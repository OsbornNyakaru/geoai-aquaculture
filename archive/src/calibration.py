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


class PoolingDefect(RuntimeError):
    """A pool that ranks like its members but decides unlike them. See assert_pool_sane()."""


# Widening applied to the members' realized-positive-rate envelope before the pool is judged
# to have escaped it. max(sd of the member rates, POS_RATE_FLOOR). The floor keeps a family
# whose members happen to agree to within a couple of rows from tripping on rounding.
POS_RATE_FLOOR = 0.005


def assert_pool_sane(y, member_oof_cal, member_test_cal, pooled_oof, pooled_test,
                     on_defect: str = "raise") -> Dict:
    """FAIL LOUDLY when a pool ranks like its members but DECIDES unlike them.

    WHY. Under a literal 0.5 cut, F1 is 60% of the metric and is a function of WHERE the cut
    lands, while AUC is a function of the ORDER alone. A pool can therefore average the order
    perfectly and still move the operating point off the end of its own members' range -- and
    every diagnostic we habitually look at (AUC, member agreement, pooling "gain") stays green
    while it happens. That is exactly what killed ARM T in iteration 41: the tcons 5-seed pool
    predicted 643 positives of 1030 against member counts of 575 and 601, bought 4 extra true
    positives with ~40 extra false ones, and lost 0.0161 F1 with its AUC still inside the member
    range. We read the pooled composite, concluded the arm was a single-seed mirage, and
    discarded the arm that held our best public AND best private submission of all 91.

    TWO CHECKS, both label-free on test and train-only on OOF.

    (1) RANK-OK-BUT-DECISION-BAD, on OOF. pooled F1@0.5 below EVERY member's while pooled AUC
        sits INSIDE the member range. This is the signature the post-mortem was asked to guard.
        Honest limitation, measured: on all five families on disk the calibrated OOF positive
        rate is pinned to the train prior (0.397-0.403 against a train prior of 0.4023) while
        the TEST rate is 0.55-0.60. On OOF the operating point is right by construction, so
        this check is structurally weak -- it never fired on any healthy family, but there is
        no local evidence that it would have fired on tcons either. It is cheap and it is a
        true positive when it fires; it is not the load-bearing check.

    (2) OPERATING-POINT ESCAPE, on test. The pooled realized positive rate must lie inside the
        members' own realized range, widened by max(sd of the member rates, POS_RATE_FLOOR).
        This uses NO test labels and NO knowledge of the true prevalence -- only the members'
        own outputs -- so it is legal to run before any submission. This is the check that
        would have caught ARM T.

    CALIBRATION OF (2), STATED PLAINLY. The tolerance was chosen with the ARM T case in view,
    which is fitting a threshold to one positive example and must be read as such. What
    defends it is the margin, not the fit: on the five families on disk the pool overshoots
    its member envelope by 0 rows (dpa, amix, teacher_perm, jtt) and 2 rows (presto, whose two
    members agree to within 3 rows), while the tcons pool overshoots by ~24 rows beyond the
    widened envelope. There is an order of magnitude between the null cases and the positive
    one, so the verdict does not depend on where in that gap the line is drawn.

    `on_defect` is "raise" (default) or "warn". Returns the diagnostics either way.
    """
    y = np.asarray(y)
    m_f1 = [f1_at(y, o, 0.5) for o in member_oof_cal]
    m_auc = [_auc(y, o) for o in member_oof_cal]
    m_rate = [float((np.asarray(t) >= 0.5).mean()) for t in member_test_cal]
    p_f1, p_auc = f1_at(y, pooled_oof, 0.5), _auc(y, pooled_oof)
    p_rate = float((np.asarray(pooled_test) >= 0.5).mean())
    n_test = len(np.asarray(pooled_test))

    tol = max(float(np.std(m_rate, ddof=1)) if len(m_rate) > 1 else 0.0, POS_RATE_FLOOR)
    lo_r, hi_r = min(m_rate) - tol, max(m_rate) + tol

    faults = []
    if p_f1 < min(m_f1) and min(m_auc) <= p_auc <= max(m_auc):
        faults.append(
            f"RANK-OK-BUT-DECISION-BAD: pooled OOF F1@0.5 {p_f1:.6f} is BELOW every member "
            f"({min(m_f1):.6f}..{max(m_f1):.6f}) while pooled OOF AUC {p_auc:.6f} sits INSIDE "
            f"the member range ({min(m_auc):.6f}..{max(m_auc):.6f}). The pool averaged the "
            f"ORDER fine and moved the OPERATING POINT.")
    if not (lo_r <= p_rate <= hi_r):
        over = (p_rate - hi_r) if p_rate > hi_r else (p_rate - lo_r)
        faults.append(
            f"OPERATING-POINT ESCAPE: pooled test positive rate {p_rate:.4f} "
            f"({int(round(p_rate*n_test))} of {n_test}) is outside the members' own range "
            f"[{min(m_rate):.4f}, {max(m_rate):.4f}] widened by {tol:.4f} -> "
            f"[{lo_r:.4f}, {hi_r:.4f}]; overshoot {over:+.4f} "
            f"({int(round(over*n_test)):+d} rows). A pool is supposed to sit BETWEEN its "
            f"members, not past them.")

    diag = {"member_oof_f1": m_f1, "member_oof_auc": m_auc, "member_test_pos_rate": m_rate,
            "pooled_oof_f1": p_f1, "pooled_oof_auc": p_auc, "pooled_test_pos_rate": p_rate,
            "pos_rate_tol": tol, "faults": faults}

    if faults:
        msg = ("POOLING GUARD TRIPPED -- do NOT submit this pool without deciding, on the "
               "record, that the pool is right and its members are wrong:\n  * "
               + "\n  * ".join(faults)
               + "\n  Remedies that keep the 0.5 cut literal: submit the members individually, "
                 "or pool in a space that does not move the level. Note (measured 2026-08-17 on "
                 "5 families) that arithmetic-probability, geometric-odds and pool-then-calibrate "
                 "all land within 2 rows of each other, so swapping the combiner is NOT a "
                 "remedy -- the escape means the members disagree about the level, and that is "
                 "a modelling fact, not an averaging artefact.")
        if on_defect == "raise":
            raise PoolingDefect(msg)
        log.warning("%s", msg)
    return diag


def _auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y), np.asarray(p)))


def calibrated_pool(members, log_each: bool = True,
                    guard: str = "raise") -> Tuple[np.ndarray, Dict]:
    """Pool several models by averaging their INDIVIDUALLY CALIBRATED probabilities.

    `members` is an iterable of (y_oof, p_oof, p_test) triples.

    WHY NOT RANK-AVERAGE. Under the old prevalence pin we always rank-averaged, because the
    pin re-derived the cut afterwards and so only the ORDER mattered; normalized ranks were
    the scale-free way to pool models with drifting calibration. **That reasoning dies with
    the pin.** Under a literal 0.5 cut the LEVEL is the entire TargetF1 column.

    A rank transform maps a member's OOF and its test scores to uniform [0,1] SEPARATELY.
    That destroys exactly the information we need: our models score test rows higher than
    train rows (test really is more pond-rich), and ranking each set on its own erases that
    difference. Calibrating the pooled ranks then drives the positive rate back to the TRAIN
    prior (~0.402 observed) instead of the model's honest test estimate (~0.55).

    So: calibrate each member on ITS OWN out-of-fold predictions first -- which puts every
    member on the common, meaningful probability scale that rank-averaging was a proxy for --
    then average those probabilities. Level is preserved, members remain commensurable, and
    a single vector serves both columns.

    MEASURED 2026-08-17, so nobody re-litigates the combiner. On all five member families still
    on disk (dpa x5, amix x10, teacher_perm x5, jtt x3, presto x2) this arithmetic pool, a
    geometric mean of odds, and pool-then-calibrate produce realized positive counts within TWO
    rows of each other out of 1030, and the pooled OOF F1 is above every member in the three
    seed families. There is no combiner defect here. What CAN go wrong is the pool's operating
    point escaping its members' range -- see assert_pool_sane(), which every call runs.

    `guard` is "raise" (default), "warn", or "off". Diagnostic tools that deliberately study a
    defective pool should pass "warn"; nothing that writes a submission should.
    """
    cal_test, cal_oof, slopes = [], [], []
    y_ref = None
    for i, (y_m, oof_m, test_m) in enumerate(members):
        o_cal, t_cal, slope = platt_calibrate(y_m, oof_m, test_m)
        cal_test.append(t_cal)
        cal_oof.append(o_cal)
        slopes.append(slope)
        if y_ref is None:
            y_ref = np.asarray(y_m)
        if log_each:
            log.info("  member %d: Platt slope=%.3f | its own test pos-rate %.3f",
                     i, slope, float((t_cal >= 0.5).mean()))
    p_test = np.vstack(cal_test).mean(axis=0)
    p_oof = np.vstack(cal_oof).mean(axis=0)
    out = {"p_oof_pooled": p_oof, "slopes": slopes, "n_members": len(slopes)}
    if guard != "off" and len(cal_oof) > 1:
        out["guard"] = assert_pool_sane(y_ref, cal_oof, cal_test, p_oof, p_test,
                                        on_defect=guard)
    return p_test, out


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
        # After isotonic, the F1-optimal cut may not be 0.5; find it on OOF and re-map
        # through a second logit shift so the comparison happens at 0.5.
        # ⚠️ THIS IS NOT "LEGAL" — an earlier comment here claimed it was, and that claim was
        # WRONG. Searching for the F1-optimal cut and then moving it onto 0.5 is threshold
        # tuning with a wrapper: the literal 0.5 is cosmetic. The whole of calibrate_for_f1 is
        # the NON-COMPLIANT `pinned` path, retained only so experiments/anchors.tsv stays
        # reproducible. Nothing it returns may be submitted. See REPORT.md section 8.
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
