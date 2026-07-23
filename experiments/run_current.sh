#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 19 — DISPERSION / LOWER-TAIL POOLING.  *** SCREEN FIRST, 0 SUBMISSIONS FROM THIS RUN ***
#
#   WHY THIS, NOW. The round-09 deep research produced a THREE-WAY convergence, the strongest signal
#   we have had:
#     - Claude Research: replace the masked-mean pool with MOMENTS (mean+std+min+max). Physics: a
#       pond is a PERMANENT low scatterer -> LOW temporal dispersion + a low tail; the bare mean
#       literally discards both. Capacity-light, channel-REPLACING (not the toxic amplitude ADD).
#     - Gemini Deep Research: replace the pool with attention pooling (PMA) -- same target, the pool
#       is the lossy bottleneck -- by a different route.
#     - OUR OWN iter12 probe: `mean_min` (lower tail) is the ONLY candidate that ever cleared the
#       noise floor (+0.0672 ATC-F1, +0.0109 LB) -- but DIS disagreed and it had no seed replicate,
#       so the seed-noise guard held it. This run gives it the proper 2-seed test.
#
#   iter18 established that pooling transformer-VARIANTS cannot buy level (cross-arch rank-corr
#   0.9395, ~ the 0.9511 seed baseline; a member needs rho<0.9 to add ensemble level). So the lever
#   is not more variants -- it is a better REPRESENTATION (this run) and, next, a decorrelated
#   model-class member (MiniRocket/CropNet, iter20).
#
#   THE ISOLATED CHANGE is `seq.pooling` only; everything else is the champion. The head expansion
#   is identity-preserving at init (mean half re-init at the champion's 1/sqrt(d) scale, the new-
#   statistic halves zeroed), so any delta is EARNED by the new statistic, not an init-scale
#   confound. Off (`mean`) reproduces 0.8955 exactly.
#
#   DECISION: this is a SCREEN. Submit a pooling variant only if >=2 cleared estimators (ATC-F1,
#   DIS) beat champion AND the margin EXCEEDS the estimator's seed sd (the guard that correctly held
#   mean_min last time). A HOLD costs nothing.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- 1. Anchors: re-certify the estimators against THIS code (and keep the ensemble members fresh). ----
PRE="--set seq.relative_time=false --set seq.consistency_lambda=0"
python run_pipeline.py $COMMON --name seq_a_detrend $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4      $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_base    $PRE
python run_pipeline.py $COMMON --name seq_a_reltime --set seq.consistency_lambda=0
python run_pipeline.py $COMMON --name seq_a_nope    --set seq.consistency_lambda=0 --set seq.pos_encoding=none
python run_pipeline.py $COMMON --name seq_a_l3      --set seq.consistency_lambda=3
python run_pipeline.py $COMMON --name seq_a_xview                                   # CHAMPION
python run_pipeline.py $COMMON --name seq_a_detrend_s7 --set seed=7 $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4_s7      --set seed=7 $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_reltime_s7 --set seed=7 --set seq.consistency_lambda=0
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_xview_s${SD}" --set seed=$SD
done

# ---- 2. THE POOLING CANDIDATES. Isolated change = seq.pooling only; champion defaults otherwise. ----
#         2 seeds each so DIS is scoreable AND the seed-noise guard applies. Watch the logged input
#         width / pooling line: mean_min/mean_std must show pooled=2*d, moments must show 4*d.
python run_pipeline.py $COMMON --name c_meanmin      --set seq.pooling=mean_min
python run_pipeline.py $COMMON --name c_meanmin_s7   --set seq.pooling=mean_min --set seed=7
python run_pipeline.py $COMMON --name c_meanstd      --set seq.pooling=mean_std
python run_pipeline.py $COMMON --name c_meanstd_s7   --set seq.pooling=mean_std --set seed=7
python run_pipeline.py $COMMON --name c_moments      --set seq.pooling=moments
python run_pipeline.py $COMMON --name c_moments_s7   --set seq.pooling=moments  --set seed=7

# ---- 3. Retro-fit gate + seed floor + SCREEN the pooling candidates. ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --seed-spread seq_a_xview \
  --screen c_meanmin c_meanstd c_moments

# ---- 4. Keep the low-variance finalists current (private-slice insurance, per RESPONSE_09 policy). ----
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --diag-extra seq_a_k4 seq_a_base \
  --name champion_archblend4
python tools/seed_average.py --variant seq_a_xview --name champion_seedavg5

cat <<'NEXT'
=====================================================================
 Paste back: the RETRO-FIT + GATE, the SEED SPREAD block, and the SCREEN lines for
 c_meanmin / c_meanstd / c_moments.

 DECISION RULE (pre-committed): submit a pooling variant ONLY if >=2 cleared estimators (ATC-F1,
 DIS) beat the champion AND the winning margin EXCEEDS that estimator's own seed sd (shown in the
 SEED SPREAD block). That guard is exactly what held mean_min last time; a real signal must clear it.
   - >=2 votes AND margin > seed sd  -> SUBMIT that variant (best margin first).
   - otherwise                       -> HOLD (0 cost) and we proceed to iter20 (a decorrelated
                                        MiniRocket/CropNet member for the ensemble).

 Separately (already decided, do this once regardless of the screen): upload
 submission_champion_archblend4.csv to bank our lowest-variance finalist and confirm it lands within
 noise of champion.
=====================================================================
NEXT
