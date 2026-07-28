#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 30 — THE LEGAL CATBOOST LANE (a different model class, done right).  *** up to 2 subs ***
#
#   WHY NOW. iter29 closed within-class pooling: adding weak transformer members to archblend4 LOST
#   (-0.0047). The ONLY remaining pooling gain needs a member that is BOTH competent AND a different
#   model class. Two independent lines point at CatBoost:
#     - The public-LB leader (~0.94) uses plain CatBoost and says "the model isn't the bottleneck";
#       our ~0.04 gap is in FEATURES (RESEARCH_14, UPDATE_13).
#     - Removing the pin made calibration ~60% of the metric, and trees vs nets differ sharply there
#       -- our pin-era GBDT rejection (-0.0155) is now void (it was measured under the pin).
#     - The closest analogous WINNER (Zindi AgriFieldNet) is CatBoost+LGBM+XGB on temporal aggregates
#       of spectral indices. Farm Pin's winner kept a weaker Random-Forest member for its DIFFERENT
#       BIAS (+1%) -- exactly the role we need filled.
#
#   WHAT IS NEW AND LEGAL. The old GBDT feature matrix was full of n-DEPENDENT shift-carriers
#   (min/max/range, window start/end/center, counts, run-lengths) -- illegal under the masking trap
#   (train=12mo, test=4-6mo). `features.n_invariant_only` restricts to statistics unbiased at every
#   window length (mean/median/std/interior-quantiles/fractions). Plus `vh_cdf_profile`: the
#   permanence feature F(tau)=fraction of observed VH months below tau -- Class-A, literature-anchored
#   (Ottinger ~-21.5 dB), not affine-spanned. CatBoost runs with ORDERED boosting (small-n lever) and
#   the LEGAL calibration (Platt-on-OOF + literal 0.5). Smoke-verified: legal, n_features 78.
#
#   SMOKE FINDING TO WATCH. Legal CatBoost is well-calibrated (Platt slope ~4.4) so its 0.5 cut
#   under-selects: pos-rate ~0.40 vs the transformer's 0.548 and the ~0.649 true prevalence. Its F1
#   column may look weak standalone; its RANKING (AUC) and its DIFFERENT BIAS are what it brings to a
#   blend. Read the blend, not just the standalone.
#
#   THE READS (I make the upload calls from your paste):
#     - c_catboost standalone LB near/above the transformer's 0.8897 -> "model isn't the bottleneck"
#       confirmed; trees are competitive legally.
#     - rho(catboost, xview) LOW (trees vs net -> expect <0.9) -> genuinely decorrelated.
#     - champion_catblend5 > archblend4's 0.899643 -> the different-bias member LIFTS the blend: the
#       first cross-class LEVEL gain, and a new finalist. This is the whole point.
#     - c_catboost_noidx vs c_catboost -> do S2 spectral indices help a TREE? (Our -0.075 index verdict
#       was Transformer-only; trees split on indices directly. Re-test, do not inherit the veto.)
# =====================================================================
set -euo pipefail

# Legal CatBoost common flags. Single bag seed keeps ordered boosting affordable; raise later if it
# proves worth it. Indices ON by default (sar_basic + water_indices + sdwi) -- the ablation turns them off.
# cv.n_repeats=1 (not 3): OOF only feeds the Platt fit; test preds come from the separately-fit full
# model, so one CV repeat is a safe ~3x speedup for ordered boosting (~5 min/arm vs ~15).
CB="--full --model gbdt --set models.use=[catboost] --set models.bag_seeds=[42] \
 --set cv.n_repeats=1 --set features.n_invariant_only=true --set features.vh_cdf_profile=true \
 --set models.catboost.boosting_type=Ordered"
COMMON="--full --model seq"

# ---- 1. Transformer blend members (fast on GPU). Same set as archblend4 + its seed spread. ----
python run_pipeline.py $COMMON --name seq_a_reltime --set seq.consistency_lambda=0
python run_pipeline.py $COMMON --name seq_a_nope    --set seq.consistency_lambda=0 --set seq.pos_encoding=none
python run_pipeline.py $COMMON --name seq_a_l3      --set seq.consistency_lambda=3
python run_pipeline.py $COMMON --name seq_a_xview                           # CHAMPION (legal 0.889686)
python run_pipeline.py $COMMON --name seq_a_reltime_s7 --set seed=7 --set seq.consistency_lambda=0
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_xview_s${SD}" --set seed=$SD
done

# ---- 2. Legal CatBoost arms. ----
echo "=== c_catboost: legal recipe (n-invariant + VH CDF + indices + ordered boosting) ==="
python run_pipeline.py $CB --name c_catboost
echo "=== c_catboost_noidx: same, S2 spectral indices OFF (the tree indices re-test) ==="
python run_pipeline.py $CB --name c_catboost_noidx \
  --set features.water_indices=false --set features.sdwi=false
echo "=== c_catboost_spw: address 0.5 under-selection via class weight (training-side, legal) ==="
python run_pipeline.py $CB --name c_catboost_spw --set models.scale_pos_weight=2.2

# ---- 3. CONTROL + CANDIDATE blends. ----
# archblend4 CONTROL must reproduce pos-rate 0.5670 / LB 0.899643, else the comparison is void.
echo "=== CONTROL: archblend4 (known LB 0.899643, expect pos-rate 0.5670) ==="
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --name champion_archblend4
echo "=== CANDIDATE: catblend5 = archblend4 members + legal CatBoost (the different-bias member) ==="
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview c_catboost \
  --diag-extra c_catboost_noidx c_catboost_spw \
  --name champion_catblend5

cat <<'NEXT'
=====================================================================
 PASTE BACK:
   (i)   the `run: c_catboost`, `run: c_catboost_noidx`, `run: c_catboost_spw` summary blocks
         (final_oof, oof_auc, and each one's realized test pos-rate line).
   (ii)  the CONTROL archblend4 pooled pos-rate (MUST be 0.5670, else VOID -- stop and say so).
   (iii) the CANDIDATE catblend5 block: per-member Platt slopes, POOLED pos-rate, and the 5x5
         correlation matrix (the 'catboost' row = rho vs each transformer = the go/no-go).

 THEN UPLOAD (I pick from your numbers; up to 2):
   - submission_champion_catblend5.csv   -> vs archblend4's 0.899643 (the main event)
   - submission_c_catboost.csv           -> the standalone "is the model the bottleneck?" read

 COMMITTED READS:
   - catblend5 >= 0.9056  -> CONFIDENT cross-class WIN. A competent, decorrelated member lifts the
                             blend; new finalist, and the tree/feature lane is alive -> push features.
   - catblend5 0.9002-0.9056 -> suggestive; confirm with a weight tweak or a seed pool.
   - catblend5 <= 0.8990  -> no lift. Trees don't add here even legally -> the ceiling is features
                             inside ONE model, or we are done and it's writeup time.
   - c_catboost standalone near 0.889 -> trees ARE competitive; the leader's 0.94 is FEATURES, and
                             the ratio/permanence feature battery (iter31) is the real lane.
   - c_catboost_noidx >= c_catboost -> S2 indices are dead even for a tree; drop them, SAR-only.
=====================================================================
NEXT
