"""Regime-matched calibration — fit Platt on a calibration set shaped like deployment.

THE DEFECT
----------
`run_seq_cv` builds the two vectors Platt bridges with DIFFERENT averaging structure:

                        window views          fold-models
    OOF row             mean of R=2           1
    test row            1 (the real window)   mean of n_splits

Each side is variance-shrunk on the axis the other is not. Platt is a 1-D logistic fit on the OOF
logits and applied to the test logits, so it inherits both mismatches. Under the OLD prevalence pin
this was harmless -- the cut was re-derived downstream and only the ORDER survived. Under a LITERAL
0.5 cut the Platt slope IS the operating point, and a slope fit on a smoother vector than it is
applied to misplaces the 0.5 crossing.

WHAT THIS FIXES, AND WHAT IT DOES NOT
-------------------------------------
Fixed: the WINDOW axis. A test row shows exactly one window, so the OOF row that calibrates it must
be one window -- R=1. `src/seq_model.py` now records every individual view, so the R=1 calibration
set is reconstructed here offline at zero extra training cost.

NOT fixed: the MODEL axis (1 fold-model vs n_splits). Closing it needs n_repeats>1, which also
changes p_test_raw and the ranking -- a confounded lever, and confounded levers are how this project
has burned submissions before. It is MEASURED below (see the test_per_fold diagnostic) and left
alone.

RULES COMPLIANCE (the three-prong test, argued before any number was seen)
--------------------------------------------------------------------------
(a) The decision rule stays a literal 0.5. `R` does not appear in src/calibration.py.
(b) The knob is fixed by a TRAIN-ONLY criterion stated a priori: *the calibration set's averaging
    structure must match deployment's*. It is argued from the data-generating process, never from a
    realized positive rate and never from leaderboard feedback.

    ⚠️ PRE-COMMITMENT: R=1 ships whatever positive rate it produces. Choosing between R=1 and R=2
    AFTER seeing their positive rates would convert this into threshold tuning by the back door and
    void the arm. The R=all column below is printed for UNDERSTANDING ONLY. It is not a candidate.
(c) It corrects a demonstrably mis-specified model -- the asymmetry is a fact of the code, not a
    hypothesis -- rather than relabelling a fixed estimate.

CONTROL
-------
`--views all` must reproduce tools/seed_average.py BIT-FOR-BIT. If it does not, the reconstruction
is wrong and the R=1 number means nothing. Run it first; the verification section of the iter44 plan
requires it.

USAGE
    python tools/regime_match.py --variant amix --name champion_regimematch
    python tools/regime_match.py --variant amix --name _control --views all   # must match seed_average
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seed_average import collect  # noqa: E402  (same globbing convention, one source of truth)
from src.calibration import calibrated_pool, platt_calibrate  # noqa: E402
from src.utils import f1_at, get_logger, load_config, resolve_path, roc_auc  # noqa: E402

log = get_logger()

_EPS = 1e-6
_VIEW_KEYS = ("oof_view_p", "oof_view_owner", "oof_view_k")


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, _EPS, 1 - _EPS) / (1 - np.clip(p, _EPS, 1 - _EPS)))


def rebuild_oof(d, n_rows: int, views: str) -> np.ndarray:
    """OOF vector at a chosen view depth, from the per-view record.

    views='1'   -> view k=0 only, averaged over repeats: ONE window per row, matching a test row.
    views='all' -> every view, averaged: reproduces the bundle's own `oof_prob` BIT-FOR-BIT.

    The arithmetic deliberately mirrors run_seq_cv's exactly -- a per-(repeat, row) `.mean()` over
    the raw predictions in their native float32, then a float64 accumulate-and-divide across
    repeats, matching `oof_sum[va] += prob_rows` / `oof_sum / oof_cnt`. Doing the obvious thing
    instead (upcast to float64, one np.add.at over everything) changes the summation order and the
    accumulator width; it agrees to ~1e-9, which is invisible in the OOF metrics but survives Platt
    into the TargetRAUC decimals and breaks the bit-for-bit control. The control is worth more than
    the brevity.
    """
    missing = [k for k in _VIEW_KEYS if k not in d.files]
    if missing:
        raise SystemExit(
            f"bundle lacks {missing}; it predates the iter44 per-view instrumentation. "
            "Regenerate it on Colab -- do NOT mix instrumented and un-instrumented members, "
            "the pool would silently average two different calibration regimes.")
    p, own, kk = d["oof_view_p"], d["oof_view_owner"], d["oof_view_k"]
    rep = d["oof_view_rep"] if "oof_view_rep" in d.files else np.zeros(len(p), np.int16)
    keep = np.ones(len(p), bool) if views == "all" else (kk == 0)

    num = np.zeros(n_rows, dtype=np.float64)
    den = np.zeros(n_rows, dtype=np.float64)
    for r in np.unique(rep):
        m = keep & (rep == r)
        own_r, p_r = own[m], p[m]
        order = np.argsort(own_r, kind="stable")
        own_s, p_s = own_r[order], p_r[order]
        bounds = np.flatnonzero(np.diff(own_s)) + 1
        for grp, idx in zip(np.split(p_s, bounds), np.split(own_s, bounds)):
            num[idx[0]] += grp.mean()      # native-dtype mean, exactly as run_seq_cv does it
            den[idx[0]] += 1.0
    if (den == 0).any():
        raise SystemExit(f"{int((den == 0).sum())} rows have no view at views={views!r}")
    return num / den


def binormal_b(y: np.ndarray, p: np.ndarray) -> tuple:
    """Unequal-variance binormal b = sd(logit|neg)/sd(logit|pos), and the AUC it implies.

    Direct moment estimator on the logit scale. Deliberately NOT the profile-likelihood/EM fit
    attempted earlier in this project: that one had no labels, railed to the boundary for every
    artifact, and gave b=20.550 vs b=0.018 for two seeds of the SAME architecture. With labels the
    moment estimator is well-posed and needs no optimizer.

    b is a property of the RANKING, and Platt is affine on the logit scale, so b is invariant to
    calibration -- which is exactly why no calibration change can move it. Reported so the implied
    AUC can be checked against the empirical one: if they disagree badly, the binormal model does
    not describe these scores and any ceiling computed from it is void.
    """
    z = _logit(np.asarray(p, dtype=float))
    y = np.asarray(y).astype(int)
    zp, zn = z[y == 1], z[y == 0]
    sp, sn = float(zp.std(ddof=1)), float(zn.std(ddof=1))
    if sp <= 0:
        return float("nan"), float("nan"), float("nan")
    b = sn / sp
    a = (float(zp.mean()) - float(zn.mean())) / sp
    from math import erf, sqrt
    implied = 0.5 * (1.0 + erf((a / sqrt(1.0 + b * b)) / sqrt(2.0)))
    return b, a, implied


def describe(tag: str, y, oof, p_test, cfg) -> dict:
    """Everything we can learn about one calibration regime, on LABELLED data."""
    oof_cal, test_cal, slope = platt_calibrate(y, oof, p_test)
    grid = np.arange(0.05, 0.95 + 1e-9, 0.005)
    f1s = np.array([f1_at(y, oof_cal, t) for t in grid])
    f_star = float(f1s.max())
    t_at = float(grid[int(f1s.argmax())])
    b, a, implied = binormal_b(y, oof_cal)
    near = lambda v, lo, hi: int(((v >= lo) & (v <= hi)).sum())  # noqa: E731
    d = {
        "slope": slope,
        "pos_rate": float((test_cal >= 0.5).mean()),
        "oof_f1_at_half": f1_at(y, oof_cal, 0.5),
        "oof_auc": roc_auc(y, oof_cal),
        "f_star": f_star,
        "t_at_f_star": t_at,
        "delta_hat": f_star / 2.0,          # Lipton et al. (arXiv:1402.1892): F1-optimal cut = F*/2
        "b": b, "a": a, "implied_auc": implied,
        "test_near_cut": near(test_cal, 0.45, 0.55),
        "test_near_wide": near(test_cal, 0.40, 0.60),
        "test_cal": test_cal,
    }
    log.info("--- %s ---", tag)
    log.info("  Platt slope %.4f | realized test pos-rate %.4f | OOF f1@0.5 %.4f | OOF AUC %.4f",
             d["slope"], d["pos_rate"], d["oof_f1_at_half"], d["oof_auc"])
    log.info("  F*_masked %.4f at t=%.3f  ->  delta_hat = F*/2 = %.4f", f_star, t_at, d["delta_hat"])
    log.info("  binormal b=%.3f a=%.3f | implied AUC %.4f vs empirical %.4f (gap %+.4f)",
             b, a, implied, d["oof_auc"], implied - d["oof_auc"])
    log.info("  test rows in [0.45,0.55]: %d | in [0.40,0.60]: %d  (of %d)",
             d["test_near_cut"], d["test_near_wide"], len(test_cal))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="config whose seed bundles to pool")
    ap.add_argument("--seeds", nargs="*", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--preds-dir", default=None)
    ap.add_argument("--views", choices=["1", "all"], default="1",
                    help="'1' = regime-matched (SHIPS). 'all' = control, must equal seed_average.")
    args = ap.parse_args()

    cfg = load_config()
    sub_dir = resolve_path(cfg, "submissions_dir")
    preds_dir = Path(args.preds_dir) if args.preds_dir else sub_dir / "preds"
    name = args.name or f"{args.variant}_regimematch"

    files = collect(preds_dir, args.variant, args.seeds)
    if len(files) < 2:
        raise SystemExit(f"need >=2 seed bundles for {args.variant}, found {len(files)}")

    log.info("Regime-matched calibration over %d bundles for %r (views=%s)",
             len(files), args.variant, args.views)
    oof_1, oof_all, raw_test, tpf, ids, y = [], [], [], [], None, None
    for f in files:
        d = np.load(f, allow_pickle=True)
        if y is None:
            y = d["y"]
        n_rows = len(d["oof_prob"])
        rebuilt_all = rebuild_oof(d, n_rows, "all")
        drift = float(np.abs(rebuilt_all - d["oof_prob"]).max())
        # EXACT, not approximate. The all-view rebuild is the same arithmetic on the same numbers,
        # so anything but 0.0 means the per-view record does not describe how oof_prob was formed
        # -- and then the R=1 vector it is used to build describes nothing either.
        if not np.array_equal(rebuilt_all, np.asarray(d["oof_prob"])):
            raise SystemExit(
                f"{f.name}: all-view rebuild is not bit-identical to oof_prob (max drift "
                f"{drift:.2e}). The per-view record is not faithful; refusing to continue.")
        oof_all.append(rebuilt_all)
        oof_1.append(rebuild_oof(d, n_rows, "1"))
        raw_test.append(np.asarray(d["p_test_raw"]))
        if "test_per_fold" in d.files:
            tpf.append(np.asarray(d["test_per_fold"]))
        tid = d["test_ids"] if "test_ids" in d.files else None
        if ids is None:
            ids = tid
        elif tid is not None and not np.array_equal(ids, tid):
            raise SystemExit(f"test_ids mismatch in {f.name}; bundles are not comparable")
        log.info("    %s  (rebuild max-drift %.1e)", f.name, drift)
    if ids is None:
        raise SystemExit("bundles carry no test_ids; cannot write a submission")

    # ---- Diagnostics on the POOLED-equivalent single member (seed 0) is not enough; report the
    #      per-seed mean of each quantity so one odd seed cannot carry the read. ----
    log.info("")
    log.info("=== CALIBRATION REGIMES (per-seed mean over %d seeds) ===", len(files))
    stats = {}
    for tag, vecs in (("R=all  (current, CONTROL)", oof_all), ("R=1    (regime-matched)", oof_1)):
        per = [describe(f"{tag} :: seed {i}", y, o, t, cfg)
               for i, (o, t) in enumerate(zip(vecs, raw_test))]
        stats[tag] = per

    log.info("")
    log.info("=== SUMMARY (means across seeds) ===")
    for tag, per in stats.items():
        log.info("  %-26s slope %.4f | pos-rate %.4f | delta_hat %.4f | b %.3f",
                 tag, np.mean([p["slope"] for p in per]), np.mean([p["pos_rate"] for p in per]),
                 np.mean([p["delta_hat"] for p in per]), np.mean([p["b"] for p in per]))

    # ---- MODEL-AXIS mismatch, measured (diagnostic only; not corrected this round). ----
    if tpf:
        t0 = tpf[0]                                    # [n_folds, n_test]
        sd_single = float(np.mean(np.std(t0, axis=0, ddof=1)))
        log.info("")
        log.info("MODEL-AXIS (uncorrected): %d fold-models | mean per-row sd across folds %.4f "
                 "-> sd of their mean ~%.4f (shrink %.2fx). The OOF side averages ONE model, so "
                 "this much model noise is present in OOF and absent from the test vector.",
                 t0.shape[0], sd_single, sd_single / np.sqrt(t0.shape[0]), np.sqrt(t0.shape[0]))

    # ---- The submission. Same legal path as seed_average: per-member Platt, then average the
    #      probabilities, then a literal 0.5 cut. The ONLY difference is which OOF vector each
    #      member is calibrated on. ----
    chosen = oof_1 if args.views == "1" else oof_all
    log.info("")
    log.info("=== LEGAL POOL (views=%s): per-seed calibration, then probability average ===",
             args.views)
    p_pooled, pdiag = calibrated_pool([(y, o, t) for o, t in zip(chosen, raw_test)])
    target_f1 = (p_pooled >= 0.5).astype(int)

    # Direct size of the intervention: how many rows actually change side.
    p_ctrl, _ = calibrated_pool([(y, o, t) for o, t in zip(oof_all, raw_test)], log_each=False)
    flips = int(((p_pooled >= 0.5) != (p_ctrl >= 0.5)).sum())
    log.info("POOLED: %d seeds | test pos-rate %.4f (train prior %.4f)",
             pdiag["n_members"], float(target_f1.mean()), float(np.mean(y)))
    log.info("ROWS CROSSING 0.5 vs the R=all control: %d of %d full test (~%.1f on the 309-row "
             "public slice). The +0.006 composite bar is ~1.9 TP, and crossings are a TP/FP mix.",
             flips, len(p_pooled), flips * 309.0 / len(p_pooled))

    # ---- ISOLATION. `R` is read only on the held-out path, so it cannot change any single
    #      member's ranking: Platt is strictly monotone, and changing its slope re-labels a
    #      member's probabilities without reordering them. THAT is the assertion with teeth, and
    #      it is checked per member below -- a failure there means R leaked into training or
    #      inference and the arm is void.
    #
    #      It does NOT follow that the POOLED ranking is preserved, and an earlier version of this
    #      tool wrongly asserted that it was. calibrated_pool averages PROBABILITIES after a
    #      per-member Platt map. Give two members different slopes and you rescale what is being
    #      averaged, so the average can reorder even though neither input did. The pooled rank
    #      correlation is therefore a DIAGNOSTIC, not a gate: this arm is not exactly AUC-neutral
    #      for a multi-seed artifact, only very nearly so. Report the size and move on.
    for i, (o1, oa, t) in enumerate(zip(oof_1, oof_all, raw_test)):
        _, tc1, _ = platt_calibrate(y, o1, t)
        _, tca, _ = platt_calibrate(y, oa, t)
        if not np.array_equal(np.argsort(np.argsort(tc1)), np.argsort(np.argsort(tca))):
            raise SystemExit(
                f"member {i}: the R=1 and R=all calibrated test scores are NOT rank-identical. "
                "Platt is strictly monotone and R is read only on the held-out path, so this is "
                "impossible unless R leaked into training or inference. THE ARM IS VOID.")
    log.info("Isolation OK: per-member rankings are exactly identical (%d members), so R reached "
             "nothing but the calibration map.", len(oof_1))

    if args.views == "all":
        log.info("Pooled rank correlation vs control: 1.0 trivially (this IS the control run).")
    else:
        rho = float(np.corrcoef(np.argsort(np.argsort(p_pooled)),
                                np.argsort(np.argsort(p_ctrl)))[0, 1])
        log.info("Pooled rank correlation vs control: %.8f. Below 1.0, and that is EXPECTED, not a "
                 "fault -- pooling averages per-member-calibrated PROBABILITIES, so per-member "
                 "slope changes reshape the average even though no member reordered. Budget a "
                 "small AUC-column drift; this arm is near-AUC-neutral, not exactly AUC-neutral.",
                 rho)

    sub = pd.DataFrame({"ID": ids, "TargetF1": target_f1, "TargetRAUC": p_pooled})
    sample = pd.read_csv(resolve_path(cfg, "raw_dir") / "SampleSubmission.csv")
    order = {i: k for k, i in enumerate(sample["ID"])}
    sub = sub.sort_values("ID", key=lambda s: s.map(order)).reset_index(drop=True)
    if len(sub) != len(sample):
        raise SystemExit(f"row count {len(sub)} != SampleSubmission {len(sample)}")

    out = sub_dir / f"submission_{name}.csv"
    sub.to_csv(out, index=False)
    log.info("Wrote %s  (%d rows, pos-rate %.3f)", out, len(sub), float(sub["TargetF1"].mean()))
    log.info("")
    log.info("READ THIS BEFORE INTERPRETING THE SCORE. Expected outcome is a NULL: our calibrated "
             "scores are saturated (~3% of mass within 0.05 of the cut), so a slope change of this "
             "size moves very few rows. A null here is a real measurement of the boundary lane, not "
             "a failed experiment -- record it as such and do not rescue it.")


if __name__ == "__main__":
    main()
