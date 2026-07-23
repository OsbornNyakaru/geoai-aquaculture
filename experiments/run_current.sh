#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 20 — mean_min AS A DECORRELATED ENSEMBLE MEMBER.  *** SCREEN + AT MOST 1 SUBMISSION ***
#
#   WHY. iter19 screened dispersion pooling; `mean_min` was the standout (ATC-F1 +0.0672, the only
#   candidate margin ever to clear the seed floor, replicated). The seed-PAIRED LB test settled it:
#         xview seed42 = 0.8955   vs   c_meanmin seed42 = 0.8986   (+0.0031, INSIDE noise)
#   So mean_min is NOT a standalone level gain (ATC-F1 overpredicted; DIS's -0.035 was flat wrong),
#   but it is AT LEAST as good as champion and physics-backed (Ottinger: ponds = persistent low
#   backscatter = a low-order statistic the mean-pool discards). Its remaining value is as an
#   ENSEMBLE MEMBER on a NEW axis of diversity we have never mixed: POOLING, not architecture.
#
#   iter18 showed the four mean-pool architecture variants are only ~0.94 rank-correlated -> too
#   alike to buy level. THE OPEN QUESTION iter20 ANSWERS: is a min-pool model decorrelated enough
#   from the mean-pool cluster (target rho < 0.9) to finally add ENSEMBLE LEVEL, not just variance?
#   The cross-correlation matrix that arch_blend prints is the free go/no-go.
#
#   DECISION: read the correlation row for c_meanmin. If its mean rho vs the mean-pool members is
#   < ~0.90, champion_archblend5 (which includes it) is a genuine level candidate -> screen + maybe
#   submit. If ~0.94 like the others, pooling-diversity is exhausted too and we pivot to
#   instance-expansion (iter21, a data-model change). champion_meanmin_seedavg5 is banked as a
#   finalist either way (mean_min is our highest single public draw, seed-collapsed for reliability).
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- 1. Anchors: re-certify estimators AND supply the mean-pool ensemble members. ----
PRE="--set seq.relative_time=false --set seq.consistency_lambda=0"
python run_pipeline.py $COMMON --name seq_a_detrend $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4      $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_base    $PRE
python run_pipeline.py $COMMON --name seq_a_reltime --set seq.consistency_lambda=0
python run_pipeline.py $COMMON --name seq_a_nope    --set seq.consistency_lambda=0 --set seq.pos_encoding=none
python run_pipeline.py $COMMON --name seq_a_l3      --set seq.consistency_lambda=3
python run_pipeline.py $COMMON --name seq_a_xview                                   # CHAMPION (mean-pool)
python run_pipeline.py $COMMON --name seq_a_detrend_s7 --set seed=7 $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4_s7      --set seed=7 $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_reltime_s7 --set seed=7 --set seq.consistency_lambda=0
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_xview_s${SD}" --set seed=$SD
done

# ---- 2. The min-pool member at 5 seeds (matched to xview) so it can be seed-pooled + seed-averaged. ----
python run_pipeline.py $COMMON --name c_meanmin --set seq.pooling=mean_min
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "c_meanmin_s${SD}" --set seq.pooling=mean_min --set seed=$SD
done

# ---- 3. The test: does POOLING-diversity decorrelate where architecture-diversity did not? ----
#         archblend5 ADDS c_meanmin as a 5th member; its correlation row is the go/no-go.
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview c_meanmin \
  --diag-extra seq_a_k4 \
  --name champion_archblend5
# The pure min-pool seed-average -- our highest single public draw (0.8986), variance-collapsed.
python tools/seed_average.py --variant c_meanmin --name champion_meanmin_seedavg5
# Keep the 4-member blend + xview seed-avg as the incumbent finalists for comparison.
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --diag-extra seq_a_k4 seq_a_base \
  --name champion_archblend4
python tools/seed_average.py --variant seq_a_xview --name champion_seedavg5

# ---- 4. Retro-fit gate + seed floor + screen the new blend and the min-pool member. ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --seed-spread seq_a_xview c_meanmin \
  --screen c_meanmin champion_archblend5

cat <<'NEXT'
=====================================================================
 Paste back: the CROSS-ARCHITECTURE RANK CORRELATION matrix (now including c_meanmin), the
 RETRO-FIT + GATE, the SEED SPREAD block, and the SCREEN lines.

 THE GO/NO-GO is c_meanmin's correlation row vs the mean-pool members {reltime,nope,l3,xview}:
   mean rho < ~0.90  -> min-pool is a genuinely decorrelated member; champion_archblend5 can buy
                        LEVEL. If its ATC-F1 also beats archblend4, UPLOAD submission_champion_archblend5.csv
                        (this run's one submission).
   mean rho ~ 0.94   -> pooling-diversity is as exhausted as architecture-diversity; do NOT submit;
                        we pivot to iter21 = instance-expansion (a data-model change).

 Regardless: submission_champion_meanmin_seedavg5.csv is banked as a finalist candidate (highest
 single public draw, seed-collapsed). Compare its future LB to archblend4's 0.8946 when you upload it.
=====================================================================
NEXT
