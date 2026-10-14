"""What does going legal actually cost? Estimate it for ZERO submissions.

We replaced the (rules-violating) prevalence pin with train-only Platt calibration and a
literal 0.5 cut. The pin forced the realized test positive rate to 0.649; the legal cut
lands wherever the calibrated model says it lands. The difference is a set of rows that
flip from 1 to 0 (or back), and `TargetF1` is scored ONLY on that set.

We cannot compute the true cost -- that needs test labels. But we can bound it tightly,
because the whole quantity depends on one unknown: the PRECISION of the rows that flip.

    F1 = 2*TP / (k + P)          k = predicted positives, P = true positives (fixed)

Moving from k_pinned to k_legal changes TP by (number flipped) x (their precision). We
sweep that precision over its whole plausible range and report the implied score change,
so the decision is made against a range rather than a guess.

Usage:
    python tools/compliance_diff.py --variant seq_a_xview
    python tools/compliance_diff.py --variant seq_a_xview --true-prevalence 0.65
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import calibrate_legal, target_prevalence_shift  # noqa: E402
from src.utils import get_logger, load_config, resolve_path  # noqa: E402

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="preds bundle name, e.g. seq_a_xview")
    ap.add_argument("--preds-dir", default=None)
    ap.add_argument("--prevalence", type=float, default=None,
                    help="the pinned target we are replacing (default: config)")
    ap.add_argument("--true-prevalence", type=float, default=0.65,
                    help="believed TRUE test positive rate, for the P term")
    ap.add_argument("--public-frac", type=float, default=0.30,
                    help="public slice fraction, for the per-slice sensitivity note")
    args = ap.parse_args()

    cfg = load_config()
    preds_dir = Path(args.preds_dir) if args.preds_dir else resolve_path(cfg, "submissions_dir") / "preds"
    pin = args.prevalence if args.prevalence is not None else float(cfg["calibration"]["prevalence_target"])

    f = preds_dir / f"preds_{args.variant}.npz"
    if not f.exists():
        raise SystemExit(f"no bundle at {f}")
    d = np.load(f, allow_pickle=True)
    y, oof, p_test = d["y"], d["oof_prob"], d["p_test_raw"]

    # The two operating points, on identical raw scores.
    f1_legal, rauc_legal, diag = calibrate_legal(y, oof, p_test)
    p_pinned, _ = target_prevalence_shift(p_test, pin)
    f1_pinned = (p_pinned >= 0.5).astype(int)

    n = len(p_test)
    k_legal, k_pinned = int(f1_legal.sum()), int(f1_pinned.sum())
    flipped = int((f1_legal != f1_pinned).sum())
    P = args.true_prevalence * n           # true positive count (believed)

    log.info("")
    log.info("=== COMPLIANCE DIFF: %s ===", args.variant)
    log.info("  rows                          %d", n)
    log.info("  pinned  k (predicted pos)     %d   (rate %.4f)  <- rules-violating", k_pinned, k_pinned / n)
    log.info("  LEGAL   k (predicted pos)     %d   (rate %.4f)  <- train-only calibration, 0.5 cut",
             k_legal, k_legal / n)
    log.info("  rows whose TargetF1 flips     %d", flipped)
    log.info("  train prior                   %.4f", diag["train_prior"])

    if k_legal == k_pinned:
        log.info("  -> identical operating point; compliance is FREE for this variant.")
        return

    # Sweep the one unknown: what fraction of the flipped rows were true positives?
    # Everything else in F1 is determined.
    log.info("")
    log.info("  Implied score change, swept over the precision of the flipped rows.")
    log.info("  (F1 = 2*TP/(k+P) with P = %.0f at true prevalence %.2f; score weight 0.6)",
             P, args.true_prevalence)
    log.info("")
    log.info("    %-28s %10s %10s", "precision of flipped rows", "dF1", "d(score)")
    # Anchor TP so that the pinned F1 matches a plausible level, then perturb.
    # We do not know TP; but dF1 depends on it only weakly, so we solve TP from an
    # assumed pinned F1 and report the sensitivity of the ANSWER, not of TP.
    for f1_pin_assumed in (0.87,):
        tp_pin = f1_pin_assumed * (k_pinned + P) / 2.0
        for prec in (0.30, 0.50, 0.65, 0.80, 0.95):
            d_tp = (k_legal - k_pinned) * prec         # signed: negative if legal selects fewer
            f1_new = 2 * (tp_pin + d_tp) / (k_legal + P)
            d_f1 = f1_new - f1_pin_assumed
            log.info("    %-28.2f %10.4f %10.4f", prec, d_f1, 0.6 * d_f1)

    log.info("")
    log.info("  READ: the marginal rows at a near-optimal cut are close to coin flips, so the")
    log.info("  middle rows (0.50-0.65) are the realistic case. F1 is FLAT near its optimum,")
    log.info("  which is why moving k by %+d rows costs far less than the raw row count suggests.",
             k_legal - k_pinned)
    log.info("  Public slice is ~%.0f rows, so the observed delta carries ~%.0f seed-noise sd on top.",
             args.public_frac * n, 0.0191 / 0.010)


if __name__ == "__main__":
    main()
