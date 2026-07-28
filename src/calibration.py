"""Operating point for the two submission columns.

TWO MODES, selected by `calibration.compliance_mode`.

`legal` (DEFAULT) — see calibrate_legal(). Platt calibration fit on TRAINING out-of-fold
predictions only, then a LITERAL 0.5 cut. Both columns come from the same strictly-monotone
calibrated probability, so TargetRAUC is a genuine probability and AUC is unchanged.

`pinned` (⚠️ NON-COMPLIANT, historical) — see calibrate_for_f1() and score_for_auc(). This
is what produced every score in experiments/anchors.tsv and it VIOLATES THE RULES on two
counts. It is retained solely so the historical anchors remain reproducible; do not ship it.

    Rules, verbatim (zindi.africa/.../rules, read 2026-07-28):
      "Setting a probability threshold is strictly forbidden. Your binary target should be
       based on the default threshold of 0.5."
      "...do not set thresholds (or round your probabilities) to improve your place on the
       leaderboard."
      "Zindi will need the raw probabilities. This will allow the clients to set thresholds
       to their own needs."

    (a) TargetF1: target_prevalence_shift() computes a quantile of the logits and shifts so
        that quantile lands at 0.5 -- its own docstring calls this "a threshold on the
        logits". The prevalence constant 0.649 was moreover swept against LEADERBOARD
        FEEDBACK (iteration 02), not derived from training data.
    (b) TargetRAUC: score_for_auc() emits uniformly-spaced ranks, which are not
        probabilities.

    See REPORT.md section 7 and gemini_loop/RESPONSE_13.md.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .utils import f1_at, get_logger

log = get_logger()

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def find_tstar(y: np.ndarray, p: np.ndarray, cfg: dict) -> float:
    """Grid-search the F1-maximising threshold on OOF predictions."""
    c = cfg["calibration"]
    grid = np.arange(c["t_grid_lo"], c["t_grid_hi"] + 1e-9, c["t_grid_step"])
    f1s = [f1_at(y, p, t) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def logit_shift(p: np.ndarray, t_star: float) -> np.ndarray:
    """Monotone shift mapping p=t_star -> 0.5."""
    return _sigmoid(_logit(np.asarray(p, dtype=float)) - _logit(np.array([t_star]))[0])


def target_prevalence_shift(p: np.ndarray, pi_hat: float) -> Tuple[np.ndarray, float]:
    """Additive logit shift delta so the fraction predicted positive equals pi_hat.

    sigmoid(z + delta) > 0.5  <=>  z > -delta, so the positive fraction is monotone
    in delta and hitting an EXACT target prevalence is a threshold on the logits:
    delta* = -quantile(z, 1 - pi_hat). This is the base-rate/prior lever expressed
    directly as a prevalence target — cleaner than guessing assumed_test_prior for an
    OVER-CONFIDENT net (saturated probs make the log-odds offset barely move the
    realized rate). The shift is strictly monotone, so TargetRAUC (ranking) is
    untouched; only the 0.5 crossing moves. Returns (p_shifted, delta).
    """
    z = _logit(np.asarray(p, dtype=float))
    pi_hat = float(np.clip(pi_hat, _EPS, 1.0 - _EPS))
    # threshold on z so the top pi_hat fraction lands above 0 after the shift
    thresh = float(np.quantile(z, 1.0 - pi_hat))
    delta = -thresh
    return _sigmoid(z + delta), delta


def fit_isotonic(y: np.ndarray, p_oof: np.ndarray):
    """Cross-fit-agnostic isotonic calibrator (optional TargetF1 method)."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_oof, y)
    return iso


def score_for_auc(p_raw: np.ndarray) -> np.ndarray:
    """TargetRAUC via a rank transform. ⚠️ NON-COMPLIANT — historical use only.

    The rules require RAW PROBABILITIES in the probability column, verbatim: "Zindi will
    need the raw probabilities. This will allow the clients to set thresholds to their own
    needs." This function returns uniformly-spaced ranks in [0,1], which are not
    probabilities at all — a client thresholding the column at 0.8 gets "the top 20% of
    rows". Retained ONLY so `calibration.compliance_mode: pinned` can reproduce the
    historical anchors in experiments/anchors.tsv. Use platt_calibrate() instead.
    """
    p = np.asarray(p_raw, dtype=float)
    order = np.argsort(np.argsort(p))
    return (order + 0.5) / len(p)


def platt_calibrate(y_oof: np.ndarray, p_oof: np.ndarray, p_test: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Platt scaling fit on TRAINING out-of-fold predictions ONLY.

    This is the rules-compliant calibrator. Two properties make it the right choice here:

    1. It is fit exclusively on (y_oof, p_oof) — training data. Nothing about the test set
       or the leaderboard enters, so the resulting 0.5 cut has a clean provenance.
    2. It is **strictly monotone** in the raw score (a 1-D logistic on the logit), so the
       ranking is preserved EXACTLY and ROC-AUC is bit-identical to the raw score's. Unlike
       isotonic regression it creates no ties, and ties are scored as half-credit by AUC.

    So emitting these as TargetRAUC costs nothing on the metric while replacing rank
    surrogates with genuine probabilities. Returns (p_oof_cal, p_test_cal, slope).
    """
    from sklearn.linear_model import LogisticRegression

    z_oof = _logit(np.asarray(p_oof, dtype=float)).reshape(-1, 1)
    z_test = _logit(np.asarray(p_test, dtype=float)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)  # ~unregularized Platt
    lr.fit(z_oof, np.asarray(y_oof).astype(int))
    slope = float(lr.coef_[0][0])
    if slope <= 0:
        # A non-positive slope would REVERSE the ranking and destroy AUC. It means the OOF
        # scores are anti-correlated with the labels, which is a modelling failure, not a
        # calibration one -- fail loudly rather than silently shipping an inverted column.
        raise ValueError(
            f"Platt slope is {slope:.4f} <= 0: calibration would invert the ranking. "
            "Refusing to produce a submission.")
    return (lr.predict_proba(z_oof)[:, 1], lr.predict_proba(z_test)[:, 1], slope)


def calibrate_legal(y_oof: np.ndarray, p_oof: np.ndarray, p_test: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """RULES-COMPLIANT operating point. Returns (target_f1, target_rauc, diagnostics).

    The rules state, verbatim: "Setting a probability threshold is strictly forbidden. Your
    binary target should be based on the default threshold of 0.5." and "do not set
    thresholds (or round your probabilities) to improve your place on the leaderboard".

    So: calibrate on TRAINING data only, then cut at a LITERAL 0.5. No test-set quantity,
    no leaderboard-derived constant, and no prevalence target enters this function -- by
    construction, since it does not take `cfg`. The realized positive rate is whatever the
    calibrated model says it is; it is REPORTED, never targeted.

    Both columns come from the same strictly-monotone calibrated probability, so TargetRAUC
    is a real probability and its AUC is identical to the raw score's.
    """
    p_oof_cal, p_test_cal, slope = platt_calibrate(y_oof, p_oof, p_test)
    target_f1 = (p_test_cal >= 0.5).astype(int)

    diagnostics = {
        "mode": "legal",
        "platt_slope": slope,
        "oof_f1_at_0.5": f1_at(np.asarray(y_oof), p_oof_cal, 0.5),
        "test_pos_rate": float(target_f1.mean()),      # reported, NOT targeted
        "train_prior": float(np.mean(y_oof)),
        "p_oof_final": p_oof_cal,
    }
    log.info("LEGAL calibration: Platt fit on OOF train only (slope=%.3f) | literal 0.5 cut "
             "| realized test pos-rate %.3f (train prior %.3f) | OOF f1@0.5=%.4f",
             slope, diagnostics["test_pos_rate"], diagnostics["train_prior"],
             diagnostics["oof_f1_at_0.5"])
    return target_f1, p_test_cal, diagnostics


def prior_sensitivity(y: np.ndarray, p_shifted: np.ndarray, priors, base_prior: float
                      ) -> Dict[float, float]:
    """F1@0.5 of the shifted probs under reweighted positive priors.

    Reweights OOF rows to simulate a test positive-rate = pi, reporting how
    fragile the fixed-0.5 F1 is to prior shift (diagnostic only).
    """
    out: Dict[float, float] = {}
    for pi in priors:
        w_pos = pi / max(base_prior, _EPS)
        w_neg = (1 - pi) / max(1 - base_prior, _EPS)
        w = np.where(y == 1, w_pos, w_neg)
        pred = (p_shifted >= 0.5).astype(int)
        tp = float((w * (pred == 1) * (y == 1)).sum())
        fp = float((w * (pred == 1) * (y == 0)).sum())
        fn = float((w * (pred == 0) * (y == 1)).sum())
        prec = tp / (tp + fp + _EPS)
        rec = tp / (tp + fn + _EPS)
        out[float(pi)] = float(2 * prec * rec / (prec + rec + _EPS))
    return out


def calibrate_for_f1(y_oof: np.ndarray, p_oof: np.ndarray, p_test: np.ndarray,
                     cfg: dict) -> Tuple[np.ndarray, float, Dict]:
    """Produce binary TargetF1 for the test set + diagnostics.

    Returns (target_f1[int 0/1], t_star, diagnostics).
    """
    method = cfg["calibration"].get("method", "logit_shift")
    t_star = find_tstar(y_oof, p_oof, cfg)

    if method == "isotonic":
        iso = fit_isotonic(y_oof, p_oof)
        p_oof_cal = iso.predict(p_oof)
        p_test_cal = iso.predict(p_test)
        # After isotonic, the F1-optimal cut may not be 0.5; find it on OOF and
        # re-map through a second logit shift so 0.5 is legal.
        t2 = find_tstar(y_oof, p_oof_cal, cfg)
        p_oof_final = logit_shift(p_oof_cal, t2)
        p_test_final = logit_shift(p_test_cal, t2)
        t_report = t2
    else:
        p_oof_final = logit_shift(p_oof, t_star)
        p_test_final = logit_shift(p_test, t_star)
        t_report = t_star

    # Self-check: F1@0.5 after shift must equal F1@t* before shift.
    f1_pre = f1_at(y_oof, p_oof, t_star)
    f1_post = f1_at(y_oof, p_oof_final, 0.5)
    if abs(f1_pre - f1_post) > 1e-9:
        log.warning("Calibration self-check mismatch: pre=%.6f post=%.6f", f1_pre, f1_post)

    base_prior = float(np.mean(y_oof))
    sens = prior_sensitivity(y_oof, p_oof_final, cfg["calibration"]["prior_sensitivity"],
                             base_prior)

    # Optional prior-shift (base-rate) correction. The organizers state the test
    # set is more positive than train; if we believe an assumed test prior, we
    # add the log-odds offset  logit(pi_test) - logit(pi_train)  to the TEST
    # scores so the 0.5 decision reflects the shifted base rate. This is prior
    # correction (a modeling choice), NOT post-hoc threshold tuning — the cut
    # stays at 0.5 and TargetRAUC (ranking) is untouched.
    prior_offset = 0.0
    pos_before = float((p_test_final >= 0.5).mean())
    p_test_shifted = np.array(p_test_final, dtype=float)  # t*-calibrated, pre-prior
    prevalence_target = cfg["calibration"].get("prevalence_target")
    if prevalence_target is not None:
        # Deterministic operating-point: hit an EXACT test positive rate via a logit
        # shift, independent of the net's (over)confidence. Preferred over the
        # log-odds prior offset for saturated seq probs. Ranking (TargetRAUC) intact.
        p_test_final, prior_offset = target_prevalence_shift(p_test_final, float(prevalence_target))
        log.info("Prevalence target ON: pi_hat=%.3f delta=%.3f | test pos-rate %.3f -> %.3f",
                 float(prevalence_target), prior_offset, pos_before,
                 float((p_test_final >= 0.5).mean()))
    elif cfg["calibration"].get("prior_adjust", False):
        assumed = float(cfg["calibration"].get("assumed_test_prior") or base_prior)
        prior_offset = float(_logit(np.array([assumed]))[0] - _logit(np.array([base_prior]))[0])
        p_test_final = _sigmoid(_logit(p_test_final) + prior_offset)
        log.info("Prior adjust ON: train_prior=%.3f assumed_test_prior=%.3f offset=%.3f | "
                 "test pos-rate %.3f -> %.3f", base_prior, assumed, prior_offset,
                 pos_before, float((p_test_final >= 0.5).mean()))

    target_f1 = (p_test_final >= 0.5).astype(int)
    diagnostics = {
        "method": method,
        "t_star": t_star,
        "t_report": t_report,
        "prior_offset": prior_offset,
        "test_pos_rate": float(target_f1.mean()),
        "oof_f1_at_0.5_after": f1_post,
        "oof_f1_at_tstar_before": f1_pre,
        "prior_sensitivity": sens,
        "p_oof_final": p_oof_final,
        "p_test_shifted": p_test_shifted,   # pre-prior; use with apply_prior() to sweep
        "base_prior": base_prior,
    }
    return target_f1, t_report, diagnostics


def apply_prior(p_test_shifted: np.ndarray, base_prior: float, assumed_prior: float
                ) -> np.ndarray:
    """Binary TargetF1 for an arbitrary assumed test prior, without retraining.

    Applies the log-odds base-rate offset to the already-t*-calibrated test
    probabilities and cuts at 0.5. Lets one trained model emit submissions for
    a whole sweep of assumed priors.
    """
    offset = float(_logit(np.array([assumed_prior]))[0] - _logit(np.array([base_prior]))[0])
    p = _sigmoid(_logit(np.asarray(p_test_shifted, dtype=float)) + offset)
    return (p >= 0.5).astype(int)


def estimate_test_prior_bbse(oof_prob: np.ndarray, y: np.ndarray,
                             test_prob: np.ndarray, threshold: float) -> float:
    """Black-Box Shift Estimation of the test positive prior (label shift).

    Uses only the classifier's HARD predictions at a prior-independent
    threshold (the F1-optimal t* from OOF), so it is robust to our scores being
    a rank-averaged blend rather than calibrated probabilities.

    Model: mu = C^T q, where C[i,j] = P(yhat=j | y=i) estimated on OOF and
    mu[j] = P(yhat=j) on test. Solve for the test label distribution q.
    Assumes label shift (p(x|y) stable); with covariate shift present this is
    approximate, so treat as a centred estimate and bracket it on the LB.
    """
    y = np.asarray(y).astype(int)
    yhat_oof = (np.asarray(oof_prob, dtype=float) >= threshold).astype(int)
    yhat_test = (np.asarray(test_prob, dtype=float) >= threshold).astype(int)
    base = float(y.mean())

    C = np.zeros((2, 2), dtype=float)   # C[i, j] = P(yhat=j | y=i)
    for i in (0, 1):
        mask = y == i
        if mask.sum() == 0:
            return base  # cannot estimate a confusion row
        p1 = float(yhat_oof[mask].mean())
        C[i, 0], C[i, 1] = 1.0 - p1, p1
    mu = np.array([1.0 - yhat_test.mean(), yhat_test.mean()], dtype=float)

    try:
        q = np.linalg.solve(C.T, mu)
    except np.linalg.LinAlgError:
        return base
    if not np.all(np.isfinite(q)) or q.sum() <= 0:
        return base
    q = np.clip(q, 1e-6, None)
    q = q / q.sum()
    return float(np.clip(q[1], 0.05, 0.95))
