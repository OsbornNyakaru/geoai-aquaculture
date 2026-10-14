#!/usr/bin/env python
"""label_shift_gate.py -- the OFFLINE, ZERO-SUBMISSION label-shift diagnostic (round-16 research).

WHY THIS EXISTS
---------------
Feature engineering is exhausted (adding channels overfits the shift). The one remaining LEGAL lever
that touches the 0.6*F1 half of the metric is a PRIOR-SHIFT correction: recalibrate the model's
probabilities from the train prior pi_s to an ESTIMATED test prior pi_t (Saerens-Latinne-Decaestecker
2002 / MLLS, Alexandari 2020), then keep the literal 0.5 cut. This is the SAME legal category as the
Platt-on-OOF step we already ship -- pi_t is estimated from test FEATURES only, never from the LB, and
the transform is strictly monotone so ROC-AUC / the TargetRAUC column are UNCHANGED (only F1 moves).

But the correction is only SAFE under (approximately) PURE LABEL SHIFT: p(x|y) fixed, only p(y) moves.
Round-16 research settled the key subtlety: adversarial-AUC ~0.89 does NOT prove conditional shift --
pure label shift ALSO moves the marginal p(x) when classes are separable, so adv-AUC is the wrong
instrument. The correct GO/NO-GO is a MIXTURE GOODNESS-OF-FIT test: if the test score distribution is
reproduced by a reweighted mixture pi*g1 + (1-pi)*g0 of the per-class OOF score shapes, label shift is
plausible and the correction is safe; if no single pi reproduces the test histogram (shape, not just
mean), the per-class conditionals moved -> conditional shift -> do NOT trust the correction.

WHAT IT DOES (reads a preds bundle written by run_pipeline.py -- oof_prob, y, p_test_raw, test_ids):
  1. Platt-calibrate OOF->y, apply to test (Alexandari's "calibrate first" precondition).
  2. Estimate pi_t three legal ways: MLLS/EM on soft test scores, BBSE confusion-matrix inversion, and
     the current realized pos-rate at 0.5 (= the no-correction baseline).
  3. RUN THE GATE: bootstrap KS goodness-of-fit of the test scores against the pi_t-mixture of per-class
     OOF scores. Report the KS statistic and a bootstrap p-value.
  4. Report the Saerens-corrected test pos-rate and the F1 operating-point move.
  5. VERDICT: PASS (ship a conservatively shrunk pi_t) / FAIL (conditional shift -> fallback or drop).
  6. Optional --emit-submission: write a Saerens-corrected submission at a chosen/shrunk pi_t.

Everything here uses ONLY train OOF + UNLABELLED test scores -> 0 submissions, fully legal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def platt(y_oof, p_oof, p_test):
    """Monotone Platt scaling fit on OOF only (mirrors src.calibration.platt_calibrate)."""
    from sklearn.linear_model import LogisticRegression
    z_oof = _logit(p_oof).reshape(-1, 1)
    z_test = _logit(p_test).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z_oof, np.asarray(y_oof).astype(int))
    slope = float(lr.coef_[0][0])
    if slope <= 0:
        raise SystemExit(f"Platt slope {slope:.4f} <= 0: OOF anti-correlated with y; refusing.")
    return lr.predict_proba(z_oof)[:, 1], lr.predict_proba(z_test)[:, 1], slope


def saerens_em(p_test_cal, pi_s, max_iter=1000, tol=1e-8):
    """Saerens-Latinne-Decaestecker EM (= MLLS) for the test prior pi_t from calibrated soft scores.

    E: q_i = (pi/pi_s)p_i / [ (pi/pi_s)p_i + ((1-pi)/(1-pi_s))(1-p_i) ]
    M: pi <- mean_i q_i.  Iterate from pi_s to a fixpoint.
    Returns (pi_t, q) where q are the corrected posteriors under pi_t.
    """
    p = np.clip(np.asarray(p_test_cal, dtype=float), 1e-9, 1 - 1e-9)
    pi = float(pi_s)
    for _ in range(max_iter):
        a = (pi / pi_s) * p
        b = ((1 - pi) / (1 - pi_s)) * (1 - p)
        q = a / (a + b)
        new_pi = float(q.mean())
        if abs(new_pi - pi) < tol:
            pi = new_pi
            break
        pi = new_pi
    a = (pi / pi_s) * p
    b = ((1 - pi) / (1 - pi_s)) * (1 - p)
    return pi, a / (a + b)


def bbse_binary(p_oof_cal, y, p_test_cal):
    """BBSE (Lipton 2018) via a hard 0.5 black-box predictor. Returns pi_t (may be outside [0,1])."""
    y = np.asarray(y).astype(int)
    h_oof = (p_oof_cal >= 0.5).astype(int)
    h_test = (p_test_cal >= 0.5).astype(int)
    # Confusion C[i,j] = P(h=i | y=j) on OOF
    C = np.zeros((2, 2))
    for j in (0, 1):
        m = (y == j)
        if m.sum() == 0:
            return float("nan")
        for i in (0, 1):
            C[i, j] = ((h_oof[m] == i).mean())
    mu = np.array([(h_test == 0).mean(), (h_test == 1).mean()])
    try:
        w_times_pis = np.linalg.solve(C, mu)     # = [P_t(y=0), P_t(y=1)] under label-shift identity
    except np.linalg.LinAlgError:
        return float("nan")
    return float(np.clip(w_times_pis[1], 0.0, 1.0))


def _ecdf(x, grid):
    x = np.sort(np.asarray(x, dtype=float))
    return np.searchsorted(x, grid, side="right") / x.size


def mixture_gof(p_oof_cal, y, p_test_cal, pi_t, n_boot=2000, seed=0):
    """Bootstrap KS goodness-of-fit: are the test scores a pi_t-mixture of per-class OOF scores?

    Null model: test scores ~ pi_t * g1 + (1-pi_t) * g0, where g1/g0 are the per-class OOF score
    distributions for y=1 / y=0. D_obs = sup|F_test - F_mix_hat|, F_mix_hat built from the OOF pools.

    DOUBLE bootstrap for the null: D_obs's variance comes from BOTH the test sample (n) AND the finite
    OOF pools that estimate F_mix. Each null draw therefore resamples a fresh test-sized mixture AND a
    fresh reference (resampled pos/neg pools), so the null reflects reference-set uncertainty and the
    test is not over-rejected when the pools are the true generator. p = P(D_null >= D_obs); p>alpha =>
    label-shift mixture NOT rejected => correction SAFE.
    """
    y = np.asarray(y).astype(int)
    pos = np.asarray(p_oof_cal, dtype=float)[y == 1]
    neg = np.asarray(p_oof_cal, dtype=float)[y == 0]
    test = np.asarray(p_test_cal, dtype=float)
    n = test.size
    grid = np.linspace(0.0, 1.0, 512)
    F_mix = pi_t * _ecdf(pos, grid) + (1 - pi_t) * _ecdf(neg, grid)
    D_obs = float(np.max(np.abs(_ecdf(test, grid) - F_mix)))
    l1 = float(np.mean(np.abs(_ecdf(test, grid) - F_mix)))
    if len(pos) == 0 or len(neg) == 0:
        return D_obs, float("nan"), l1
    rng = np.random.default_rng(seed)
    D_null = np.empty(n_boot)
    for b in range(n_boot):
        k1 = rng.binomial(n, pi_t)
        samp = np.concatenate([rng.choice(pos, size=k1, replace=True),
                               rng.choice(neg, size=n - k1, replace=True)])
        pos_b = rng.choice(pos, size=len(pos), replace=True)   # fresh reference estimate
        neg_b = rng.choice(neg, size=len(neg), replace=True)
        F_mix_b = pi_t * _ecdf(pos_b, grid) + (1 - pi_t) * _ecdf(neg_b, grid)
        D_null[b] = float(np.max(np.abs(_ecdf(samp, grid) - F_mix_b)))
    p_val = float((D_null >= D_obs).mean())
    return D_obs, p_val, l1


def main():
    ap = argparse.ArgumentParser(description="Offline label-shift gate + Saerens prior-shift diagnostic.")
    ap.add_argument("--bundle", default="submissions/preds/preds_c_perm_single.npz",
                    help="preds bundle from run_pipeline.py (oof_prob, y, p_test_raw, test_ids)")
    ap.add_argument("--alpha", type=float, default=0.05, help="GoF significance (pass if p_val > alpha)")
    ap.add_argument("--shrink", type=float, default=0.5,
                    help="ship pi_t = pi_s + shrink*(pi_t_hat - pi_s); Agent-6 recommends ~0.5 (conservative)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--emit-submission", metavar="NAME",
                    help="if set (and gate PASSES), write submissions/submission_<NAME>.csv with the "
                         "Saerens-corrected operating point at the shrunk pi_t")
    args = ap.parse_args()

    b = np.load(args.bundle, allow_pickle=True)
    oof, y, ptest = b["oof_prob"], b["y"], b["p_test_raw"]
    pi_s = float(np.asarray(y).astype(int).mean())

    oof_cal, test_cal, slope = platt(y, oof, ptest)
    posrate0 = float((test_cal >= 0.5).mean())

    pi_mlls, q_mlls = saerens_em(test_cal, pi_s)
    pi_bbse = bbse_binary(oof_cal, y, test_cal)
    D, p_val, l1 = mixture_gof(oof_cal, y, test_cal, pi_mlls, n_boot=args.n_boot)

    # Saerens-corrected pos-rate at the raw EM fixpoint and at the shrunk (shipped) prior
    posrate_mlls = float((q_mlls >= 0.5).mean())
    pi_ship = pi_s + args.shrink * (pi_mlls - pi_s)
    _, q_ship = saerens_em(test_cal, pi_s)               # q under fixpoint
    a = (pi_ship / pi_s) * np.clip(test_cal, 1e-9, 1 - 1e-9)
    bb = ((1 - pi_ship) / (1 - pi_s)) * (1 - np.clip(test_cal, 1e-9, 1 - 1e-9))
    q_ship = a / (a + bb)
    posrate_ship = float((q_ship >= 0.5).mean())

    passed = p_val > args.alpha

    print("=" * 70)
    print(f"LABEL-SHIFT GATE  |  bundle: {args.bundle}")
    print("=" * 70)
    print(f"train prior pi_s                 : {pi_s:.4f}   (Platt slope {slope:.3f})")
    print(f"current test pos-rate @0.5       : {posrate0:.4f}   (no correction -- what we ship now)")
    print("-" * 70)
    print(f"pi_t  (MLLS / Saerens EM)        : {pi_mlls:.4f}")
    print(f"pi_t  (BBSE inversion)           : {pi_bbse:.4f}")
    print(f"  -> estimators agree?           : {'YES' if abs(pi_mlls-pi_bbse)<0.05 else 'NO (caution)'}")
    print("-" * 70)
    print(f"MIXTURE GoF  KS D                : {D:.4f}")
    print(f"MIXTURE GoF  bootstrap p-value   : {p_val:.4f}   (pass if > alpha={args.alpha})")
    print(f"MIXTURE GoF  L1(CDF)             : {l1:.4f}")
    print("-" * 70)
    print(f"Saerens pos-rate @ EM fixpoint   : {posrate_mlls:.4f}   (pi_t={pi_mlls:.3f})")
    print(f"Saerens pos-rate @ shrunk prior  : {posrate_ship:.4f}   (pi_ship={pi_ship:.3f}, shrink={args.shrink})")
    print("=" * 70)
    if passed:
        print(f"VERDICT: [PASS] -- test scores ARE consistent with a pi_t-mixture (label shift plausible).")
        print(f"         The Saerens correction is SAFE. Recommend shipping the SHRUNK prior pi_ship="
              f"{pi_ship:.3f} (moves pos-rate {posrate0:.3f} -> {posrate_ship:.3f}, toward the believed ~0.65).")
        print(f"         F1 geometry (Agent 6): a 0.55->~0.60 move is worth ~+0.010..+0.019 LB (>0.013 floor).")
    else:
        print(f"VERDICT: [FAIL] -- no single pi reproduces the test score SHAPE (p={p_val:.3f} <= {args.alpha}).")
        print(f"         => CONDITIONAL shift present; Saerens/MLLS is NOT trustworthy. Do the safe fallback")
        print(f"         only (one fixed conservative pi~0.55-0.60) or keep the literal 0.5 cut. Do NOT iterate EM.")
    print("=" * 70)

    if args.emit_submission:
        if not passed:
            raise SystemExit("Refusing to emit: gate FAILED. Re-run with a passing bundle or ship the 0.5 cut.")
        import pandas as pd
        ids = np.asarray(b["test_ids"])
        tf1 = (q_ship >= 0.5).astype(int)
        # TargetRAUC = corrected probability; monotone in the raw score at fixed prior -> AUC unchanged.
        df = pd.DataFrame({"ID": ids, "TargetF1": tf1, "TargetRAUC": np.clip(q_ship, 0, 1)})
        out = Path("submissions") / f"submission_{args.emit_submission}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"wrote {out}  (pos-rate {posrate_ship:.4f}, pi_ship={pi_ship:.3f})")


if __name__ == "__main__":
    main()
