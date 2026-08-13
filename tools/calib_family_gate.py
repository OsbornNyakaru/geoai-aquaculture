"""Calibrator-family gate — does a NON-AFFINE calibration map buy anything at our fixed 0.5 cut?

WHY THIS TOOL EXISTS
--------------------
The Platt Annihilation Theorem (LB_LOG iter43) says a train-refit Platt exactly removes any AFFINE
reparameterization of the logit:  sigma(a(alpha*z + beta) + b) = sigma((a*alpha)z + (a*beta + b)).
That closed every additive-logit and F-measure-surrogate loss lane in this project.

An external research memo (gemini_loop/findings/God_mode.md, 2026-08-13) correctly identified the
theorem's scope limit: it covers only affine maps, and **the calibrator we deploy is itself a map on
the logit**. Swap Platt for a monotone but NON-affine family and the 0.5 crossing lands on rows no
Platt fit can reach, while the ranking -- and therefore the whole TargetRAUC column -- is preserved.
The memo ranked this its Slot 1: cheapest lever, best AUC safety, cleanest compliance.

It also supplied a kill condition to run BEFORE spending a submission. This tool is that check.

THE SHARP TEST (this tool's contribution beyond the memo)
---------------------------------------------------------
Beta calibration (Kull, Silva Filho & Flach, AISTATS 2017, PMLR 54:623-631) fits

        p = sigma( a*ln(s) + b*(-ln(1-s)) + c ).

Set a == b and it becomes  sigma(a*(ln s - ln(1-s)) + c) = sigma(a*logit(s) + c)  -- which IS Platt.
So beta does not merely differ from Platt: **it CONTAINS Platt as its a==b submodel**, and the entire
non-affine content of the lever is the gap |a - b|. That makes "is this lever real?" a NESTED MODEL
COMPARISON with a standard answer -- a likelihood-ratio test on 1 df -- rather than a matter of
opinion. The memo did not frame it this way; framed this way it is decidable offline, for free.

WHAT IT REPORTS
---------------
Q1  Likelihood-ratio test of beta vs Platt, per member and on the pooled score. H0: the map is
    affine on the logit. This is the memo's "is the map affine-equivalent on your data" question,
    made into a test with a p-value instead of an eyeball on the flip count.
Q2  DIRECTION of the 0.5 crossings on the test rows. The memo's own central argument (its sec 2) is
    that our cut sits too HIGH and suppresses true positives, so the lever it wants must move rows
    UP across the cut. A lever that moves them DOWN is not a weak version of the proposal -- it is
    the proposal with its sign reversed, and it costs us F1. Direction is not a detail here.
Q3  Isotonic in-sample vs 5-fold CROSS-FITTED OOF AUC. Isotonic fit on the same OOF rows it is then
    scored on is guaranteed optimistic; Niculescu-Mizil & Caruana (ICML 2005) put the Platt/isotonic
    crossover at ~1000 calibration points and we have 1817, which is close enough to the boundary
    that the honest number and the in-sample number can disagree in SIGN. Only the cross-fitted one
    may be read.

DIAGNOSIS ONLY. Nothing printed here may be fed back into the operating point. The cut stays a
literal 0.5; the only question on the table is WHICH TRAIN-ONLY MAP produces the probability, and
that choice is made on the offline evidence below, pre-committed, before any submission.

USAGE
    python tools/calib_family_gate.py --variant amix
    python tools/calib_family_gate.py --variant dpa --views 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from regime_match import rebuild_oof  # noqa: E402
from seed_average import collect  # noqa: E402
from src.calibration import platt_calibrate  # noqa: E402
from src.utils import f1_at, get_logger, resolve_path, load_config, roc_auc  # noqa: E402

log = get_logger()

_EPS = 1e-6


def _beta_design(s: np.ndarray) -> np.ndarray:
    """The beta-calibration design matrix [ln s, -ln(1-s)]. With equal coefficients it is logit(s)."""
    s = np.clip(np.asarray(s, dtype=float), _EPS, 1 - _EPS)
    return np.column_stack([np.log(s), -np.log(1 - s)])


def _loglik(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def beta_fit(y: np.ndarray, s: np.ndarray):
    """Kull/Silva Filho/Flach beta calibration. Returns (predict_fn, a, b, c, note).

    Monotonicity of the map requires a >= 0 and b >= 0. When the unconstrained fit puts one
    coefficient negative, the published remedy is to drop that term and refit -- not to clip it,
    which would leave the intercept fitted against a coefficient that is no longer there.
    """
    y = np.asarray(y).astype(int)
    X = _beta_design(s)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(X, y)
    a, b = (float(v) for v in lr.coef_[0])
    c = float(lr.intercept_[0])
    note = "full"
    if a < 0 or b < 0:
        keep = 1 if a < 0 else 0
        lr2 = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(X[:, [keep]], y)
        co = float(lr2.coef_[0][0])
        a, b = (0.0, co) if keep == 1 else (co, 0.0)
        c = float(lr2.intercept_[0])
        note = "refit(a<0)" if keep == 1 else "refit(b<0)"

    def predict(t: np.ndarray) -> np.ndarray:
        Xt = _beta_design(t)
        return 1.0 / (1.0 + np.exp(-(a * Xt[:, 0] + b * Xt[:, 1] + c)))

    return predict, a, b, c, note


def lr_test(y: np.ndarray, s: np.ndarray):
    """2*(LL_beta - LL_platt) ~ chi2(1) under H0: a == b, i.e. the map is Platt.

    Platt is EXACTLY the a==b submodel of beta, so the two fits are nested and Wilks applies. A
    non-significant statistic says the extra parameter buys no likelihood: the non-affine degree of
    freedom is not doing measurable work on these scores.
    """
    from scipy.stats import chi2

    fb, a, b, _c, note = beta_fit(y, s)
    p_platt, _, _ = platt_calibrate(y, s, s)
    stat = 2.0 * (_loglik(y, fb(s)) - _loglik(y, p_platt))
    return float(stat), float(chi2.sf(max(stat, 0.0), 1)), a, b, note


def iso_fit(y: np.ndarray, s: np.ndarray):
    m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    m.fit(np.asarray(s, dtype=float), np.asarray(y).astype(int))
    return lambda t: m.predict(np.asarray(t, dtype=float))


def run(variant: str, views: str, seeds, preds_dir: Path) -> None:
    files = collect(preds_dir, variant, seeds)
    if len(files) < 2:
        raise SystemExit(f"need >=2 seed bundles for {variant!r}, found {len(files)}")

    y, oof_l, test_l = None, [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if y is None:
            y = np.asarray(d["y"]).astype(int)
        oof_l.append(rebuild_oof(d, len(d["oof_prob"]), views))
        test_l.append(np.asarray(d["p_test_raw"], dtype=float))

    n_test = len(test_l[0])
    log.info("=" * 84)
    log.info("CALIBRATOR-FAMILY GATE | variant=%r | %d seeds | OOF views=%s", variant,
             len(files), views)
    log.info("n_oof=%d prior=%.4f | n_test=%d", len(y), float(y.mean()), n_test)
    log.info("=" * 84)

    # ---- Q1: is the non-affine degree of freedom doing measurable work? ----
    log.info("")
    log.info("Q1  LIKELIHOOD-RATIO TEST, beta vs Platt.  H0: a == b (the map is affine on the logit)")
    log.info("    %4s %8s %8s   %7s %7s", "seed", "LRstat", "p", "a", "b")
    n_rej = 0
    for i, o in enumerate(oof_l):
        st, pv, a, b, note = lr_test(y, o)
        n_rej += pv < 0.05
        log.info("    %4d %8.3f %8.4f   %7.3f %7.3f  %s%s", i, st, pv, a, b, note,
                 "   <- rejects H0" if pv < 0.05 else "")
    o_raw = np.vstack(oof_l).mean(axis=0)
    st, pv, a, b, note = lr_test(y, o_raw)
    log.info("    %4s %8.3f %8.4f   %7.3f %7.3f  %s", "POOL", st, pv, a, b, note)
    log.info("    -> %d/%d members reject at 0.05 (expected by chance %.1f). The POOLED row is the "
             "one that matters: it is the fit an actual artifact would ship.",
             n_rej, len(oof_l), 0.05 * len(oof_l))

    # ---- the three pooled artifacts, built the same legal way: per-member map, then average ----
    def pooled(mapper) -> np.ndarray:
        return np.vstack([mapper(o, t) for o, t in zip(oof_l, test_l)]).mean(axis=0)

    P = pooled(lambda o, t: platt_calibrate(y, o, t)[1])
    B = pooled(lambda o, t: beta_fit(y, o)[0](t))
    I = pooled(lambda o, t: iso_fit(y, o)(t))

    # ---- Q2: WHICH WAY do the crossings go? ----
    log.info("")
    log.info("Q2  DIRECTION of the 0.5 crossings on %d test rows, vs the shipped Platt pool", n_test)
    log.info("    %-9s %9s %10s %12s %6s %12s %11s", "map", "pos-rate", "up (0->1)", "down (1->0)",
             "net", "~net public", "rank-corr")
    for tag, v in (("platt", P), ("beta", B), ("isotonic", I)):
        up = int(((v >= 0.5) & (P < 0.5)).sum())
        dn = int(((v < 0.5) & (P >= 0.5)).sum())
        rho = float(np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(P)))[0, 1])
        log.info("    %-9s %9.4f %10d %12d %+6d %+12.1f %11.6f", tag, float((v >= 0.5).mean()),
                 up, dn, up - dn, (up - dn) * 309.0 / n_test, rho)
    log.info("    READ THE SIGN. God_mode sec 2 argues our cut sits too HIGH and suppresses true")
    log.info("    positives, so the lever it wants must move rows UP. A net-negative column is the")
    log.info("    proposal with its sign reversed, and it costs F1 rather than buying it.")

    # ---- Q3: isotonic, honestly evaluated ----
    log.info("")
    log.info("Q3  ISOTONIC: in-sample vs 5-fold CROSS-FITTED OOF AUC (mean over members)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pl, ins, cf = [], [], []
    for o in oof_l:
        pl.append(roc_auc(y, platt_calibrate(y, o, o)[1]))
        ins.append(roc_auc(y, iso_fit(y, o)(o)))
        held = np.zeros_like(o)
        for tr, va in skf.split(o.reshape(-1, 1), y):
            held[va] = iso_fit(y[tr], o[tr])(o[va])
        cf.append(roc_auc(y, held))
    log.info("    Platt              %.5f", np.mean(pl))
    log.info("    isotonic in-sample %.5f  (%+.5f)   <- fit and scored on the same rows; optimistic",
             np.mean(ins), np.mean(ins) - np.mean(pl))
    log.info("    isotonic cross-fit %.5f  (%+.5f)   <- THE HONEST NUMBER", np.mean(cf),
             np.mean(cf) - np.mean(pl))
    if np.sign(np.mean(ins) - np.mean(pl)) != np.sign(np.mean(cf) - np.mean(pl)):
        log.info("    The two disagree in SIGN. Isotonic is overfitting the calibration set, exactly")
        log.info("    the n<~1000 pathology of Niculescu-Mizil & Caruana (ICML 2005) showing up at")
        log.info("    n=%d. Any read taken from the in-sample number would have been backwards.",
                 len(y))

    # ---- the offline predictor the memo asked us to pre-register ----
    log.info("")
    log.info("OOF F1@0.5 on the R=%s replica (pooled over members) -- the memo's pre-registered "
             "predictor:", views)
    for tag, mk in (("platt", lambda o: platt_calibrate(y, o, o)[1]),
                    ("beta", lambda o: beta_fit(y, o)[0](o))):
        po = np.vstack([mk(o) for o in oof_l]).mean(axis=0)
        log.info("    %-9s f1@0.5 %.5f | AUC %.5f | pos-rate %.4f", tag, f1_at(y, po, 0.5),
                 roc_auc(y, po), float((po >= 0.5).mean()))

    log.info("")
    log.info("VERDICT TEMPLATE. The memo's Slot-1 kill condition was: OOF AUC delta >= -0.0005 AND "
             "OOF 0.5-flip count > 0. Both can pass while the lever is still worthless or harmful, "
             "because neither reads the DIRECTION of the test crossings (Q2) and neither notices "
             "that the extra parameter is insignificant (Q1). Require all three.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seeds", nargs="*", default=None)
    ap.add_argument("--views", choices=["1", "all"], default="1",
                    help="'1' = the regime-matched OOF vector that the shipped artifact uses.")
    ap.add_argument("--preds-dir", default=None)
    args = ap.parse_args()

    cfg = load_config()
    preds = Path(args.preds_dir) if args.preds_dir else resolve_path(cfg, "submissions_dir") / "preds"
    run(args.variant, args.views, args.seeds, preds)


if __name__ == "__main__":
    main()
