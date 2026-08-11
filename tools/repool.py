"""repool.py — COMBINER DIAGNOSTIC. Compare pooling orders on saved member bundles. 0 submissions.

WHY THIS EXISTS
---------------
`src/calibration.py::calibrated_pool` pools members by calibrating each one on its OWN
out-of-fold predictions and then averaging those probabilities. That is a LINEAR OPINION POOL
of calibrated forecasts, and Ranjan & Gneiting (2010, JRSS-B 72(1):71-91,
DOI 10.1111/j.1467-9868.2009.00726.x) prove that any non-trivial weighted average of distinct
calibrated probability forecasts is *necessarily uncalibrated and lacks sharpness* -- linear
pooling requires recalibration even when every member is individually calibrated. Rahaman &
Thiery (NeurIPS 2021, arXiv:2007.08792) show the same for deep ensembles and prescribe
Pool-Then-Calibrate; Wu & Gales (arXiv:2101.05397) prove calibrated members do not give a
calibrated ensemble.

Under a HARD 0.5 cut the sharpness loss is not cosmetic: it moves the realized positive rate,
which IS the entire TargetF1 column. Averaging preserves rank (AUC fine) but can compress the
pooled distribution so the fixed cut lands in the wrong place.

THE CASE THIS WAS BUILT TO ADJUDICATE (iteration 41, ARM T)
    tcons members  F1 0.9013 and 0.8975   ->  pooled F1 0.8718   (BELOW BOTH)
    tcons members AUC 0.9334 and 0.9259   ->  pooled AUC 0.9267  (normally BETWEEN)
That is the canonical underconfidence-after-averaging signature: the ranking pooled fine, the
operating point did not. ARM T holds our two highest F1 values ever and was discarded on the
strength of that pooled number. If the combiner caused it, we killed a working method.

A PREDICTION MADE HERE AND IMMEDIATELY FALSIFIED -- recorded because the correction is the point.
    The first draft of this file predicted the defect would bite "only where member Platt slopes
    DIVERGE." That is provably wrong: per-member Platt scaling is SCALE-INVARIANT, so combiner A
    cannot see a slope difference at all. A synthetic check (5 members, slopes {1,1,1,1,1} vs
    {0.35,0.6,1.0,1.7,2.6}) returned BIT-IDENTICAL pooled OOF F1 0.62529 and pos-rate 0.2033 for A
    in both settings, while B beat A in both (+0.039 and +0.031 F1).
    The real mechanism is simpler and GENERIC: an arithmetic mean of independently-noisy
    probabilities is shrunk toward the members' centre of mass, so the pooled distribution is
    narrower than any member's and a FIXED 0.5 cut catches fewer rows. It needs no slope
    heterogeneity -- only independent member noise, which is what multi-seed pooling IS.
    Consequence: this is NOT merely an ARM T diagnostic. If it reproduces on real bundles it is a
    live defect on the CHAMPION pool too.
    The ledger's +0.0055..+0.0061 pooling gains do not argue against this, and citing them as a
    rebuttal (as the first draft did) was a category error: those measured POOL vs SINGLE MEMBER,
    not combiner A vs combiner B. Pooling can be worth +0.006 and still be leaving more on the
    table via the wrong combiner. Nothing in the ledger has ever compared the two orders.
    Slope spread is still printed below -- now as a member-DISPERSION readout, not as a predictor
    of the defect.

WHAT IT COMPARES (both combiners are strictly legal; neither touches the 0.5 cut)
    A  CURRENT   per-member Platt on own OOF  ->  average PROBABILITIES        (shipping today)
    B  PROPOSED  average member LOGITS of the RAW scores -> ONE Platt fit on
                 the pooled OOF                                               (pool-then-calibrate)

Combiner B follows the logarithmic-opinion-pool form (Satopaa et al. 2014, Int. J. Forecasting
30(2):344-356): the geometric mean of odds does not compress toward the centre the way an
arithmetic mean of probabilities does, and the single post-hoc Platt map restores calibration
AFTER aggregation, which is the order the literature requires.

HOW TO READ THE OUTPUT
    The decisive line is `pooled OOF F1@0.5` against the per-member column above it.
      * Combiner A pooled F1 BELOW every member, B's between/above  -> combiner defect CONFIRMED.
      * Both pool normally                                          -> ARM T died of seed noise,
                                                                       not of the combiner; drop it.
    `test pos-rate` is a DIAGNOSTIC ONLY. It is never a tuning target and must never be swept.

USAGE
    python tools/repool.py --variant tcons                       # every preds_tcons*.npz found
    python tools/repool.py --variant tcons --seeds 42 7 13 21 29
    python tools/repool.py --members preds_a.npz preds_b.npz     # explicit paths

This reads saved bundles only. It trains nothing, writes no submission, and costs 0 uploads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import _logit, calibrated_pool, platt_calibrate  # noqa: E402
from src.utils import f1_at, get_logger, load_config, resolve_path, roc_auc  # noqa: E402

log = get_logger()


def collect(preds_dir: Path, variant: str, seeds) -> list:
    """All member bundles for one config: preds_<v>.npz plus preds_<v>_s<N>.npz.

    Same discovery rule as tools/seed_average.py, so this diagnostic sees exactly the members
    the submission pipeline would have pooled -- otherwise the comparison is not the one we care
    about.
    """
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


def pool_then_calibrate(y, raw_oof, raw_test):
    """COMBINER B: average member logits of the RAW scores, then fit ONE Platt on the pooled OOF.

    Averaging in log-odds space is the logarithmic opinion pool. Unlike the arithmetic mean of
    probabilities it does not compress the pooled distribution toward the members' centre of
    mass, so it preserves sharpness; the single Platt map fit on the POOLED OOF then supplies
    calibration after aggregation, which is the order Ranjan & Gneiting / Rahaman & Thiery
    require. Provenance is unchanged from the current path: the map is fit on training OOF and
    training labels only, and the cut stays a literal 0.5.
    """
    z_oof = np.mean([_logit(np.asarray(o, dtype=float)) for o in raw_oof], axis=0)
    z_test = np.mean([_logit(np.asarray(t, dtype=float)) for t in raw_test], axis=0)
    # platt_calibrate takes probabilities and logits them internally, so map back through the
    # sigmoid rather than re-implementing the fit and risking a second, divergent Platt.
    p_oof_avg = 1.0 / (1.0 + np.exp(-z_oof))
    p_test_avg = 1.0 / (1.0 + np.exp(-z_test))
    o_cal, t_cal, slope = platt_calibrate(y, p_oof_avg, p_test_avg)
    return t_cal, {"p_oof_pooled": o_cal, "slopes": [slope], "n_members": len(raw_oof)}


def _report(tag: str, y, p_oof, p_test) -> dict:
    """Print the three numbers that decide this, and return them."""
    stats = {
        "oof_auc": roc_auc(y, p_oof),
        "oof_f1": f1_at(y, p_oof, 0.5),
        "pos_rate": float((np.asarray(p_test) >= 0.5).mean()),
    }
    log.info("  %-38s OOF AUC %.6f | OOF F1@0.5 %.6f | test pos-rate %.4f",
             tag, stats["oof_auc"], stats["oof_f1"], stats["pos_rate"])
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare pooling orders on saved member bundles.")
    ap.add_argument("--variant", default=None, help="config whose seed bundles to pool")
    ap.add_argument("--seeds", nargs="*", default=None, help="specific seeds, e.g. 42 7 13")
    ap.add_argument("--members", nargs="*", default=None,
                    help="explicit bundle paths, instead of --variant discovery")
    ap.add_argument("--preds-dir", default=None)
    args = ap.parse_args()

    cfg = load_config()
    preds_dir = (Path(args.preds_dir) if args.preds_dir
                 else resolve_path(cfg, "submissions_dir") / "preds")

    if args.members:
        files = [Path(m) if Path(m).exists() else preds_dir / m for m in args.members]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise SystemExit("missing bundles: " + ", ".join(missing))
    elif args.variant:
        files = collect(preds_dir, args.variant, args.seeds)
    else:
        raise SystemExit("pass --variant or --members")

    if len(files) < 2:
        raise SystemExit(
            f"need >=2 member bundles, found {len(files)} in {preds_dir}. "
            "Colab's working directory is ephemeral -- if the run that produced these has ended, "
            "the bundles are gone and the member runs must be re-executed first.")

    y, raw_oof, raw_test = None, [], []
    log.info("Loading %d member bundles:", len(files))
    for f in files:
        d = np.load(f, allow_pickle=True)
        raw_oof.append(np.asarray(d["oof_prob"]))
        raw_test.append(np.asarray(d["p_test_raw"]))
        if y is None:
            y = np.asarray(d["y"])
        elif not np.array_equal(y, np.asarray(d["y"])):
            raise SystemExit(f"label vector mismatch in {f.name}; bundles are not comparable")
        log.info("    %s", f.name)

    # ---- per-member reference. The pooled numbers below are only interpretable against these. ----
    log.info("")
    log.info("=== MEMBERS (each calibrated on its own OOF, as the current pipeline does) ===")
    member_f1, slopes = [], []
    for f, o, t in zip(files, raw_oof, raw_test):
        o_cal, t_cal, slope = platt_calibrate(y, o, t)
        slopes.append(slope)
        s = _report(f"{f.stem}  (slope {slope:.3f})", y, o_cal, t_cal)
        member_f1.append(s["oof_f1"])
    lo, hi = min(member_f1), max(member_f1)
    spread = max(slopes) - min(slopes)
    log.info("")
    log.info("  Platt slope spread across members: %.3f  (min %.3f, max %.3f)",
             spread, min(slopes), max(slopes))
    log.info("  -> dispersion readout ONLY. Combiner A is scale-invariant (per-member Platt removes")
    log.info("     each member's slope), so a large spread does NOT predict a large A-vs-B gap. The")
    log.info("     A/B difference is driven by independent member NOISE, not by slope heterogeneity.")

    # ---- the two combiners ----
    log.info("")
    log.info("=== COMBINER A (CURRENT): per-member Platt -> average PROBABILITIES ===")
    pa, da = calibrated_pool([(y, o, t) for o, t in zip(raw_oof, raw_test)], log_each=False)
    sa = _report("pooled", y, da["p_oof_pooled"], pa)

    log.info("")
    log.info("=== COMBINER B (PROPOSED): average LOGITS -> ONE Platt on the pooled OOF ===")
    pb, db = pool_then_calibrate(y, raw_oof, raw_test)
    sb = _report(f"pooled (slope {db['slopes'][0]:.3f})", y, db["p_oof_pooled"], pb)

    # ---- verdict ----
    flips = int((( np.asarray(pa) >= 0.5) != (np.asarray(pb) >= 0.5)).sum())
    log.info("")
    log.info("=== VERDICT ===")
    log.info("  member OOF F1@0.5 range        : [%.6f, %.6f]", lo, hi)
    log.info("  A pooled OOF F1@0.5            : %.6f  %s", sa["oof_f1"],
             "BELOW EVERY MEMBER <-- the defect signature" if sa["oof_f1"] < lo else "within/above range")
    log.info("  B pooled OOF F1@0.5            : %.6f  %s", sb["oof_f1"],
             "BELOW EVERY MEMBER" if sb["oof_f1"] < lo else "within/above range")
    log.info("  B - A on pooled OOF F1@0.5     : %+.6f", sb["oof_f1"] - sa["oof_f1"])
    log.info("  test rows whose binary flips   : %d of %d  (pos-rate %.4f -> %.4f)",
             flips, len(pa), sa["pos_rate"], sb["pos_rate"])
    if sa["oof_f1"] < lo <= sb["oof_f1"]:
        log.info("  READ: combiner defect CONFIRMED -- A collapses below its own members and B does not.")
    elif sa["oof_f1"] >= lo:
        log.info("  READ: A pools normally here. No defect on THIS pool; B is not needed for it.")
    else:
        log.info("  READ: BOTH combiners collapse -> the failure is NOT the pooling order. Look "
                 "elsewhere (member degeneracy, e.g. a variance penalty driving logits to a constant).")
    log.info("")
    log.info("  OOF is BLIND to the leaderboard here (historically anti-correlated), so treat these")
    log.info("  as MECHANISM evidence about the combiner, not as a score prediction. pos-rate is")
    log.info("  reported for diagnosis only and is never a tuning target.")


if __name__ == "__main__":
    main()

