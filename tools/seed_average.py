"""Seed-averaged submission — the answer to the iter15 finding.

WHY THIS EXISTS
---------------
On 2026-07-22 we finally measured what had been assumed for fifteen iterations: our seed-to-seed
spread. The CHAMPION CONFIGURATION, changing nothing but the RNG seed, scored

    seed 42 -> 0.8955      seed 7 -> 0.8764      delta = 0.0191

That single number voids almost the entire experiment ledger. Every verdict we recorded except the
GBDT->Transformer swap (+0.05) and per-cell detrend (-0.0514) has an effect size SMALLER than one
seed swing -- including relative-time (+0.0128), which we celebrated as the breakthrough, and the
cross-view invariance win (+0.0047) that made the current model "champion" at all.

The strategic consequence is not subtle. When run-to-run variance dominates every real effect, the
highest-value move is not another architectural probe -- it is to STOP SAMPLING ONE DRAW.
Averaging M seeds shrinks the variance of the pooled prediction by roughly 1/M, and because the
leaderboard metric is RANK-ONLY after the prevalence pin (F1 = 2*TP/(P_hat+P) is monotone in
precision@k; AUC is rank-only by definition), averaging moves the submitted ranking toward the
EXPECTED ranking of the configuration rather than the ranking of one lucky RNG draw.

EXPECT THE PUBLIC SCORE TO FALL, AND WANT IT TO.
    0.8955 is the better of two draws from a distribution with sd ~0.013. It is very likely an
    UPWARD fluctuation. A seed-averaged submission should land nearer the configuration's true
    mean -- lower on the 309-row public slice, but far more reliable on the 721-row private slice
    that actually decides the competition and that we never get to see. Chasing the public number
    here is precisely the mistake that produces a shake-up.

METHOD (CHANGED 2026-07-28 when the prevalence pin was removed as a rules violation)
    Calibrate each seed on its OWN out-of-fold predictions (Platt), then average those
    probabilities, then cut at a literal 0.5.

    We used to rank-average and then pin the prevalence. That was right WHILE the pin existed:
    it re-derived the cut afterwards, so only the ORDER mattered and ranks were the scale-free
    way to pool seeds whose calibration drifts. Under a literal 0.5 cut the LEVEL is the entire
    TargetF1 column, and rank-transforming each seed's OOF and test SEPARATELY erases the
    train-vs-test level difference -- driving the positive rate back to the train prior (0.402
    observed) instead of the model's honest estimate (~0.55). Per-seed calibration puts every
    seed on the common probability scale that ranking was only ever a proxy for.

USAGE
    python tools/seed_average.py --variant seq_a_xview --name champion_seedavg
    python tools/seed_average.py --variant seq_a_xview --seeds 42 7 13 --name champ_3seed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import calibrated_pool  # noqa: E402
from src.utils import get_logger, load_config, resolve_path  # noqa: E402

log = get_logger()


def _ranks(p: np.ndarray) -> np.ndarray:
    """Normalized ranks in (0,1); scale-free, so per-seed calibration drift cannot dominate."""
    p = np.asarray(p, dtype=float)
    return (np.argsort(np.argsort(p)) + 0.5) / len(p)


def collect(preds_dir: Path, variant: str, seeds: List[str] | None) -> List[Path]:
    """All seed bundles for one config: preds_<v>.npz plus preds_<v>_s<N>.npz."""
    if seeds:
        out = []
        for s in seeds:
            cand = preds_dir / (f"preds_{variant}.npz" if s in ("42", "base")
                                else f"preds_{variant}_s{s}.npz")
            if cand.exists():
                out.append(cand)
            else:
                log.warning("missing bundle for seed %s: %s", s, cand)
        return out
    return sorted(preds_dir.glob(f"preds_{variant}.npz")) + \
        sorted(preds_dir.glob(f"preds_{variant}_s[0-9]*.npz"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="seq_a_xview", help="config whose seeds to pool")
    ap.add_argument("--seeds", nargs="*", default=None,
                    help="specific seeds (e.g. 42 7 13); default = every bundle found")
    ap.add_argument("--name", default=None, help="output name; default <variant>_seedavg")
    ap.add_argument("--preds-dir", default=None)
    ap.add_argument("--prevalence-target", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config()
    sub_dir = resolve_path(cfg, "submissions_dir")
    preds_dir = Path(args.preds_dir) if args.preds_dir else sub_dir / "preds"
    prevalence = (args.prevalence_target if args.prevalence_target is not None
                  else float(cfg["calibration"]["prevalence_target"]))
    name = args.name or f"{args.variant}_seedavg"

    files = collect(preds_dir, args.variant, args.seeds)
    if len(files) < 2:
        raise SystemExit(f"need >=2 seed bundles for {args.variant}, found {len(files)}. "
                         f"Run the seed replicates first.")

    log.info("Seed-averaging %d bundles for %r:", len(files), args.variant)
    mats, raw_oof, raw_test, ids, y = [], [], [], None, None
    for f in files:
        d = np.load(f, allow_pickle=True)
        mats.append(_ranks(d["p_test_raw"]))        # for the rank-disagreement diagnostic
        raw_oof.append(np.asarray(d["oof_prob"]))   # RAW: level intact, for calibration
        raw_test.append(np.asarray(d["p_test_raw"]))
        if y is None:
            y = d["y"]
        tid = d["test_ids"] if "test_ids" in d.files else None
        if ids is None:
            ids = tid
        elif tid is not None and not np.array_equal(ids, tid):
            raise SystemExit(f"test_ids mismatch in {f.name}; bundles are not comparable")
        log.info("    %s", f.name)
    if ids is None:
        raise SystemExit("bundles carry no test_ids; cannot write a submission")

    R = np.vstack(mats)

    # Diagnostic: how much do the seeds actually disagree on ORDER? This is the quantity that
    # matters, since the leaderboard sees nothing else.
    if len(R) >= 2:
        cs = []
        for i in range(len(R)):
            for j in range(i + 1, len(R)):
                a, b = R[i] - R[i].mean(), R[j] - R[j].mean()
                cs.append(float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())))
        log.info("Pairwise rank correlation between seeds: mean=%.4f min=%.4f "
                 "(1.0 would mean the seed changes nothing)", float(np.mean(cs)), float(np.min(cs)))

    p_avg = R.mean(axis=0)          # rank-pooled: used ONLY for the disagreement diagnostic

    # RULES-COMPLIANT operating point. Calibrate each SEED on its own OOF, then average the
    # probabilities -- NOT the ranks. Under a literal 0.5 cut the level is the whole TargetF1
    # column, and rank-transforming each seed's OOF and test separately erases the train-vs-test
    # level difference, collapsing the positive rate onto the train prior. See calibrated_pool().
    log.info("")
    log.info("=== LEGAL POOL: per-seed calibration, then probability average ===")
    p_pooled, pdiag = calibrated_pool([(y, o, t) for o, t in zip(raw_oof, raw_test)])
    target_f1 = (p_pooled >= 0.5).astype(int)
    target_rauc = p_pooled
    log.info("POOLED: %d seeds | test pos-rate %.4f (train prior %.4f)",
             pdiag["n_members"], float(target_f1.mean()), float(np.mean(y)))

    sub = pd.DataFrame({"ID": ids, "TargetF1": target_f1,
                        "TargetRAUC": target_rauc})
    sample = pd.read_csv(resolve_path(cfg, "raw_dir") / "SampleSubmission.csv")
    order = {i: k for k, i in enumerate(sample["ID"])}
    sub = sub.sort_values("ID", key=lambda s: s.map(order)).reset_index(drop=True)
    if len(sub) != len(sample):
        raise SystemExit(f"row count {len(sub)} != SampleSubmission {len(sample)}")

    out = sub_dir / f"submission_{name}.csv"
    sub.to_csv(out, index=False)
    log.info("Wrote %s  (%d rows, pos-rate %.3f)", out, len(sub), float(sub["TargetF1"].mean()))
    log.info("")
    log.info("REMINDER: a LOWER public score than 0.8955 is the EXPECTED and DESIRABLE outcome "
             "here. 0.8955 is the better of two draws from a distribution with sd ~0.013 and is "
             "probably an upward fluctuation; the pooled submission trades that luck for "
             "reliability on the unseen 721-row private slice.")


if __name__ == "__main__":
    main()
