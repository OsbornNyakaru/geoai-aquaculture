#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 40 — THE SHIFT-ROBUST CATBOOST LANE (gate said GO).   *** up to 2 uploads ***
#
#   WHY NOW. The Transformer lane is CAPPED: three single-seed-42 highs (perm 0.9065, vhsq 0.9133,
#   mean_min 0.9128) ALL washed to ~0.8995 on seed-averaging (seed-robust ceiling firmly ~0.8995).
#   The tree lane is the only lane with a DIFFERENT ceiling, and iter39's adversarial-validation gate
#   returned [GO]: a shift-robust CatBoost transferred to the most-test-like 30% of train (AUC 0.960,
#   F1 0.884, only +0.024 below random) -- the opposite of iter30's 0.995->0.70 illusion. The leader's
#   ~0.94 is on CatBoost, and trees can use the ratio/rank features our mean-pool provably cannot.
#
#   WHAT'S NEW vs the iter30 CatBoost that collapsed (0.995 OOF -> 0.70 LB):
#     (1) FEATURE-SHIFT REMOVAL -- drop the 27 optical-index / VH-VV-ratio / swir-std columns the gate
#         flagged as the top train/test shift-carriers (joint adv-AUC 0.9775). Keeps the transferable
#         SAR level + permanence + water-occupancy (78 -> 51 features). This is the key addition.
#     (2) SHIFT-ROBUST CatBoost: depth 4 (was 6), l2_leaf_reg 20 (was 3), Bernoulli subsample 0.75,
#         rsm 0.7, Ordered boosting -- all now the config default for the tree lane.
#     (3) n-invariant feature bank + VH permanence CDF profile; 5-seed bag; legal Platt + 0.5 cut.
#   Windowed (test-like) CV already existed and was NOT the missing piece -- the covariate shift was.
#
#   ARM 1 -- champion_catboost_sr: the full recipe (shift-robust + feature-shift removal).
#   ARM 2 -- champion_catboost_sr_nodrop: same but WITHOUT the feature drop (control -> isolates the
#            value of feature-shift removal).
#
#   THE READ (committed). Prior legal tree LB: catblend5 0.886, standalone c_catboost 0.698.
#     champion_catboost_sr >= 0.90  -> trees TRANSFER and rival the Transformer; a genuine 2nd model
#         class -> iter41 = monotone constraints + tuning + a Transformer(perm) x CatBoost blend (two
#         decorrelated model classes -> the E[max] finalist dream).
#     0.86 - 0.90  -> partial transfer; the feature drop / regularization help but a gap remains ->
#         iter41 tightens (more feature-shift removal, monotone constraints).
#     < 0.86 (and not clearly > nodrop) -> trees do NOT transfer despite everything -> conditional shift
#         is fatal to the tree lane; lock the Transformer finalists + Phase-Two writeup.
#     sr - sr_nodrop  -> the measured value of feature-shift removal.
# =====================================================================
set -euo pipefail

# Shift-robust CatBoost is now the config DEFAULT for the tree lane (depth4/l2=20/Ordered/subsample).
GBDT="--full --model gbdt --set models.use=[catboost] --set models.bag_seeds=[42,7,13,21,29]"
NINV="--set features.n_invariant_only=true --set features.vh_cdf_profile=true"
DROP="--set features.drop_name_substrings=[vv_minus_vh,mndwi,ndvi,ndwi,awei,sdwi,swir1,swir2]"

# ---- ARM 1: full shift-robust recipe (feature-shift removal ON). ----
python run_pipeline.py $GBDT $NINV $DROP --name champion_catboost_sr

# ---- ARM 2: control WITHOUT feature drop (isolates the drop's value). ----
python run_pipeline.py $GBDT $NINV --name champion_catboost_sr_nodrop

cat <<'NEXT'
=====================================================================
 PASTE BACK both run summary lines. champion_catboost_sr logs ~51 features (78 - 27 dropped); the OOF
 AUC will look high -- IGNORE it (OOF is anti-correlated with LB for trees; the iter30 illusion). Only
 the paired LB is truth.

 UPLOAD (budget 5/day):
   1. submissions/submission_champion_catboost_sr.csv          <- the shift-robust tree (THE test)
   2. submissions/submission_champion_catboost_sr_nodrop.csv   <- control (value of feature-shift removal)

 COMMITTED READ (prior legal tree LB: catblend5 0.886, standalone c_catboost 0.698):
   sr >= 0.90  -> trees TRANSFER & rival the Transformer -> iter41 = monotone constraints + a decorrelated
                  Transformer(perm) x CatBoost blend (the two-model-class E[max] finalist).
   0.86-0.90   -> partial transfer -> iter41 tightens (more drop + monotone).
   < 0.86      -> trees don't transfer (conditional shift fatal) -> lock Transformer finalists + writeup.
   sr - nodrop -> the measured value of feature-shift removal.

 NOTE seed-robust Transformer ceiling is ~0.8995 (perm/vhsq/mean_min seed-avgs all 0.8995-0.8999);
 finalists stay {perm seed-avg 0.899882, archblend4 0.899643} UNLESS the CatBoost clears ~0.90 AND is
 decorrelated (different model class) -> then it becomes a finalist leg.
=====================================================================
NEXT
