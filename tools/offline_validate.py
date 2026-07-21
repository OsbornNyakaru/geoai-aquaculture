"""Offline LB-predictive validators — the round-06 "Q1" unlock.

WHY THIS EXISTS
---------------
Our binding constraint is measurement, not compute or submissions. The public LB is
~309 rows (~±0.01 noise), so a real +0.005 gain is indistinguishable from noise without
spending a submission — and local OOF is not merely blind but ANTI-correlated with the LB
(our best-LB run has our lowest OOF). Both round-06 research reports independently
concluded the same thing: estimate test performance from the 1030 UNLABELED test rows,
then **prove the estimator works by retro-fitting it to the experiments whose LB we
already know** before trusting it on anything new.

That retro-fit is the whole point. This tool is self-certifying: if no estimator ranks the
known anchors correctly, we learn that for zero submissions and the noise floor stands.

ESTIMATORS (all label-free on test; computed from a run's saved preds bundle)
  ATC   Average Thresholded Confidence (Garg et al., ICLR 2022, arXiv:2201.04234).
        Pick threshold t on a source confidence score so that the fraction of OOF rows
        scoring below t equals the OOF error; predict test accuracy as the fraction of
        TEST rows scoring at or above t. Score function = negative entropy (Garg et al.
        found it at least as good as max-confidence, and it degrades more gracefully for
        a saturated/overconfident net like ours).
  DIS   Pairwise disagreement between two runs of the SAME config at different seeds
        (Generalization Disagreement Equality / agreement-on-the-line: Jiang et al. 2022,
        Baek et al. NeurIPS 2022). The rate at which the pair disagrees on the unlabeled
        test rows estimates test error. Needs >=2 bundles sharing a `variant` name.
  MARG  Test-margin mass: mean |p - 0.5| over test rows AFTER the prevalence shift. A
        cheap sanity baseline, not from the literature — included so the retro-fit has a
        naive control to beat. If MARG wins, the "sophisticated" estimators earned nothing.

USAGE
    python tools/offline_validate.py --preds-dir submissions/preds --anchors experiments/anchors.tsv

`anchors.tsv` is the ground truth: one row per historical run with its known public LB.
Estimators are scored by Spearman correlation against that column.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import target_prevalence_shift  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger()

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Score functions
# --------------------------------------------------------------------------- #
def _neg_entropy(p: np.ndarray) -> np.ndarray:
    """Negative binary entropy of each prediction; higher = more confident."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return p * np.log(p) + (1 - p) * np.log1p(-p)


def atc_estimate(oof_prob: np.ndarray, y: np.ndarray, p_test: np.ndarray,
                 threshold: float = 0.5) -> float:
    """ATC: predicted TEST accuracy from unlabeled test predictions.

    Garg et al. (ICLR 2022). Choose t such that
        #{OOF rows with score < t} / n_oof  ==  OOF error rate,
    then predict test accuracy = #{TEST rows with score >= t} / n_test.

    We select t as the empirical quantile of the OOF score distribution at the OOF error
    rate, which satisfies the defining equation exactly (up to ties).
    """
    oof_prob = np.asarray(oof_prob, dtype=float)
    y = np.asarray(y).astype(int)
    err = float((( oof_prob >= threshold).astype(int) != y).mean())
    s_oof = _neg_entropy(oof_prob)
    s_test = _neg_entropy(p_test)
    if err <= 0.0:
        # A perfect source fit gives no usable threshold; fall back to the score minimum.
        t = float(s_oof.min())
    else:
        t = float(np.quantile(s_oof, err))
    return float((s_test >= t).mean())


def disagreement_estimate(p_test_a: np.ndarray, p_test_b: np.ndarray,
                          threshold: float = 0.5) -> float:
    """Predicted TEST accuracy = 1 - disagreement rate between two seeds of one config."""
    a = (np.asarray(p_test_a, dtype=float) >= threshold).astype(int)
    b = (np.asarray(p_test_b, dtype=float) >= threshold).astype(int)
    return float(1.0 - (a != b).mean())


def margin_estimate(p_test: np.ndarray, prevalence_target: float | None = 0.649) -> float:
    """Naive control: mean |p - 0.5| after the prevalence shift (higher = more decisive)."""
    p = np.asarray(p_test, dtype=float)
    if prevalence_target is not None:
        p, _ = target_prevalence_shift(p, float(prevalence_target))
    return float(np.abs(p - 0.5).mean())


# --------------------------------------------------------------------------- #
# Retro-fit scoring
# --------------------------------------------------------------------------- #
def spearman(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def load_anchors(path: Path) -> List[Dict]:
    """Read the ground-truth table: variant<TAB>lb  (# comments and blanks ignored)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        try:
            rows.append({"variant": parts[0], "lb": float(parts[1])})
        except ValueError:
            continue          # header row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", default="submissions/preds")
    ap.add_argument("--anchors", default="experiments/anchors.tsv")
    ap.add_argument("--prevalence-target", type=float, default=0.649)
    args = ap.parse_args()

    anchors = load_anchors(Path(args.anchors))
    if not anchors:
        raise SystemExit(f"no anchors parsed from {args.anchors}")

    preds_dir = Path(args.preds_dir)
    est: Dict[str, Dict[str, float]] = {}
    seeds: Dict[str, List[np.ndarray]] = {}

    for a in anchors:
        v = a["variant"]
        # a variant may have several seed runs: preds_<v>.npz, preds_<v>_s7.npz, ...
        files = sorted(preds_dir.glob(f"preds_{v}.npz")) + sorted(preds_dir.glob(f"preds_{v}_s*.npz"))
        if not files:
            log.warning("MISSING bundle for variant %s (looked in %s) - skipped", v, preds_dir)
            continue
        d = np.load(files[0], allow_pickle=True)
        oof, y, p_test = d["oof_prob"], d["y"], d["p_test_raw"]
        est[v] = {
            "lb": a["lb"],
            "atc": atc_estimate(oof, y, p_test),
            "marg": margin_estimate(p_test, args.prevalence_target),
        }
        seeds[v] = [np.load(f, allow_pickle=True)["p_test_raw"] for f in files]
        if len(seeds[v]) >= 2:
            est[v]["dis"] = disagreement_estimate(seeds[v][0], seeds[v][1])

    if not est:
        raise SystemExit("no preds bundles found - run the anchor regeneration first")

    order = sorted(est, key=lambda v: est[v]["lb"])
    log.info("")
    log.info("%-22s %8s %8s %8s %8s", "variant", "LB", "ATC", "DIS", "MARG")
    for v in order:
        e = est[v]
        log.info("%-22s %8.4f %8.4f %8s %8.4f", v, e["lb"], e["atc"],
                 f"{e['dis']:.4f}" if "dis" in e else "-", e["marg"])

    lb = [est[v]["lb"] for v in order]
    log.info("")
    log.info("=== RETRO-FIT: Spearman rho vs known public LB (n=%d) ===", len(order))
    verdicts = {}
    for key in ("atc", "dis", "marg"):
        have = [v for v in order if key in est[v]]
        if len(have) < 3:
            log.info("  %-5s : insufficient bundles (%d)", key.upper(), len(have))
            continue
        rho = spearman([est[v]["lb"] for v in have], [est[v][key] for v in have])
        verdicts[key] = rho
        log.info("  %-5s : rho = %+.3f   (n=%d)", key.upper(), rho, len(have))

    # ---- the pre-committed gate (RESPONSE_06.md section 6) ----
    log.info("")
    need = {"seq_a_detrend", "seq_a_k4", "seq_a_reltime", "seq_a_xview"}
    if need.issubset(set(est)):
        for key, rho in verdicts.items():
            bad = max(est["seq_a_detrend"][key], est["seq_a_k4"][key])
            good = min(est["seq_a_reltime"][key], est["seq_a_xview"][key])
            ok = bad < good
            log.info("GATE %-5s: detrend/K4 below reltime/xview? %s  (rho=%+.3f)",
                     key.upper(), "PASS" if ok else "FAIL", rho)
            if ok and rho > 0.7:
                log.info("  -> %s CLEARS THE BAR: usable offline screen. Open the gated backlog.",
                         key.upper())
    else:
        log.info("GATE: need all 4 anchor variants %s to evaluate", sorted(need))
    log.info("")
    log.info("Decision rule: rho > 0.7 AND gate PASS -> screen candidates offline, submit only "
             "when >=2 estimators beat the champion. Otherwise Q1 failed (cost: 0 submissions) "
             "and we fund only ideas with plausible effect >= +0.013.")


if __name__ == "__main__":
    main()
