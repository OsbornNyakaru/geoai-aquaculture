#!/usr/bin/env python
"""adversarial_cv.py -- FREE, LEGAL, offline GO/NO-GO gate for the TREE (GBDT) lane (round-17).

WHY. iter30's CatBoost collapsed OOF 0.995 -> LB 0.70. It already used windowed features + n-invariant
gating, so windowing was NOT the fix. MODE-A (shift_diagnostics) showed why: there is REAL covariate shift
on the SAR level that windowing does NOT reproduce, so a masking-aware StratifiedKFold OOF is still
optimistic. Before spending ANY submission on a re-tuned tree, this tool answers the make-or-break question
with 0 submissions: does a model validated on the MOST TEST-LIKE train rows still transfer, or does it
collapse (=> conditional shift, no tree tuning will save it)?

WHAT IT DOES (legal: train values+labels + UNLABELLED test features only; no LB, no external data):
  1. Build the WINDOWED train feature matrix (test-like masking) + the test feature matrix (n-invariant
     legal bank).
  2. JOINT adversarial AUC: 5-fold CatBoost predicting is-test on [train_win ; test]. High (~0.9) => the
     feature space separates train from test => any tree can key on the shift. Prints top shift-carrier
     features by importance (candidates for feature-shift removal).
  3. TEST-LIKE HOLDOUT vs RANDOM HOLDOUT of the LABEL model. Using out-of-fold P(test), hold out the most
     test-like ~30% of train rows; train a shift-robust CatBoost on the rest; score F1@0.5 + AUC on that
     test-like holdout AND on a random 30% holdout. If test-like ~ random and both healthy => trees
     transfer (GO). If test-like collapses far below random => conditional shift dominates (NO-GO).
  4. VERDICT + the pos-rate the test-like holdout implies (a legal, non-LB prevalence sanity check).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_bundle                              # noqa: E402
from src.features import build_test_matrix, build_train_matrix  # noqa: E402
from src.utils import get_logger, load_config                 # noqa: E402

log = get_logger()


def _catboost(seed, depth, l2, iters, ordered=True):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        loss_function="Logloss", random_seed=seed, iterations=iters,
        depth=depth, l2_leaf_reg=l2, learning_rate=0.03,
        boosting_type=("Ordered" if ordered else "Plain"),
        bootstrap_type="Bernoulli", subsample=0.75, rsm=0.7,
        verbose=False, allow_writing_files=False,
    )


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def _f1_at(y, p, thr=0.5):
    from sklearn.metrics import f1_score
    return float(f1_score(y, (p >= thr).astype(int)))


def main():
    ap = argparse.ArgumentParser(description="Offline adversarial-validation gate for the GBDT tree lane.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--l2", type=float, default=20.0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--holdout-frac", type=float, default=0.30)
    args = ap.parse_args()

    from sklearn.model_selection import StratifiedKFold

    cfg = load_config()
    # Legal n-invariant feature bank (no shift-carriers) + the permanence CDF profile.
    cfg["features"]["n_invariant_only"] = True
    cfg["features"]["vh_cdf_profile"] = True
    cfg.setdefault("features", {}).setdefault("cdf_taus", [-22.0, -21.0, -20.0, -19.0])

    b = load_bundle(cfg)
    train_cube, y = b.train_cube, np.asarray(b.y).astype(int)
    test_cube, schema, wd = b.test_cube, b.schema, b.window_dist

    # One test-like window per train row (matches test's one-view-per-row).
    Xtr, ytr, groups, names = build_train_matrix(train_cube, y, schema, wd, cfg, K=1, seed=args.seed)
    Xte, _ = build_test_matrix(test_cube, schema, cfg)
    log.info("feature matrix: train_win %s, test %s, %d features", Xtr.shape, Xte.shape, len(names))

    # ---- 1-2. JOINT adversarial AUC + shift-carrier importances (out-of-fold) ----
    Xall = np.vstack([Xtr, Xte])
    is_test = np.concatenate([np.zeros(len(Xtr)), np.ones(len(Xte))]).astype(int)
    adv_oof = np.zeros(len(Xall))
    imp = np.zeros(Xtr.shape[1])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for tr, va in skf.split(Xall, is_test):
        m = _catboost(args.seed, args.depth, args.l2, args.iters, ordered=False)
        m.fit(Xall[tr], is_test[tr])
        adv_oof[va] = m.predict_proba(Xall[va])[:, 1]
        imp += np.asarray(m.get_feature_importance())
    adv_auc = _auc(is_test, adv_oof)
    top = np.argsort(imp)[::-1][:12]

    print("=" * 76)
    print("TREE-LANE ADVERSARIAL-VALIDATION GATE (free, offline, legal)")
    print("=" * 76)
    print(f"JOINT adversarial AUC (train_win vs test): {adv_auc:.4f}")
    print("  (0.5 = indistinguishable; ~0.9 = strong shift the trees will exploit)")
    print("top shift-carrier features (adv-importance) -> candidates to DROP:")
    for j in top:
        print(f"    {names[j]:<28} imp={imp[j]/5:.2f}")

    # per-train-row P(test) = its adv OOF score
    ptest_tr = adv_oof[:len(Xtr)]

    # ---- 3. TEST-LIKE holdout vs RANDOM holdout of the LABEL model ----
    n = len(Xtr)
    k = int(args.holdout_frac * n)
    testlike_idx = np.argsort(ptest_tr)[::-1][:k]         # most test-like rows
    rng = np.random.default_rng(args.seed)
    random_idx = rng.choice(n, size=k, replace=False)

    def _eval(hold_idx, tag):
        mask = np.zeros(n, dtype=bool); mask[hold_idx] = True
        m = _catboost(args.seed, args.depth, args.l2, args.iters, ordered=True)
        m.fit(Xtr[~mask], ytr[~mask])
        p = m.predict_proba(Xtr[mask])[:, 1]
        yh = ytr[mask]
        auc, f1 = _auc(yh, p), _f1_at(yh, p)
        posrate = float((p >= 0.5).mean())
        print(f"  {tag:<18} AUC={auc:.4f}  F1@0.5={f1:.4f}  pred-pos-rate={posrate:.3f}  "
              f"(holdout true-pos={yh.mean():.3f})")
        return auc, f1

    print("-" * 76)
    print(f"LABEL-MODEL holdout comparison (CatBoost depth={args.depth}, l2={args.l2}, Ordered):")
    a_rand, f_rand = _eval(random_idx, "random 30%")
    a_test, f_test = _eval(testlike_idx, "test-like 30%")
    print("-" * 76)
    dref = a_rand - a_test
    print(f"transfer gap (random AUC - test-like AUC): {dref:+.4f}")
    print("=" * 76)
    # ---- 4. VERDICT ----
    if a_test >= 0.75 and dref <= 0.06:
        print("VERDICT: [GO] the tree transfers to the most-test-like rows (test-like AUC healthy, gap small)")
        print("  -> build the shift-robust CatBoost lane (monotone constraints + feature-shift removal +")
        print("     early-stop on the test-like holdout) and spend ONE seed-averaged submission.")
    elif a_test >= 0.70:
        print("VERDICT: [WEAK] partial transfer; the shift is partly conditional. Try feature-shift removal")
        print("  (drop the top adv-carriers above) and re-run this gate; submit only if the gap closes.")
    else:
        print("VERDICT: [NO-GO] the tree COLLAPSES on test-like rows -> conditional shift dominates. No tree")
        print("  tuning or prior correction will transfer; do NOT spend a submission. Stay with the Transformer.")
    print("  (Legal: this used train labels + UNLABELLED test features only; no LB, no external data.)")


if __name__ == "__main__":
    main()
