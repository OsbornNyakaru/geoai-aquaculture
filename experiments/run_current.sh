#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 12 — THE FIRST OFFLINE SCREEN.  *** COSTS ZERO SUBMISSIONS ***
#
#   iter11 PASSED. Two estimators cleared the retro-fit on the 7 known-LB anchors:
#       ATC-F1  15/15 concordant, rho = +0.964   (exact null p ~ 0.005)
#       DIS      5/5  concordant, rho = +1.000   (n=4; exact null p ~ 0.042 -> second vote only)
#   and, confirming the rank-only proof, the two CONFIDENCE-based estimators came out NEGATIVE:
#       ATC     6/15, rho = -0.429      MARG    8/15, rho = -0.321
#   The leaderboard cannot see calibration, so estimators that measure saturation mislead here.
#
#   So we can now RANK CANDIDATES OFFLINE. This run screens five of them and submits none.
#   Each candidate runs at 2 seeds because DIS needs a seed pair to be computable.
#
#   THE PRE-COMMITTED RULE (do not renegotiate after seeing the numbers):
#     >=2 cleared estimators above champion -> submit it (1 submission), ranked by vote margin.
#     1 or 0 votes                          -> HOLD. Costs nothing.
#     ALL candidates HOLD                   -> a real result: the structural lane is exhausted
#                                             and the Presto lane (RESEARCH_07.md 5e) is next.
#
#   The committed config is UNCHANGED and still the exact 0.8955 champion — every variant below
#   is produced by --set overrides. Verified: all new flags OFF reproduces the champion tensor
#   bit-for-bit (24 channels), rank_replace keeps 24, compact_missing gives 14.
#
#   Runtime: the iter11 log measured ~35-70 s per run, so ~21 runs is roughly 20 minutes.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- Anchors: historical variants with known public LB (experiments/anchors.tsv) ----
# Regenerated every time so the retro-fit re-certifies the estimators against THIS code.
# If ATC-F1 or DIS stops clearing the gate below, the code changed something that matters and
# the screen results are VOID — report that rather than the screen table.
PRE="--set seq.relative_time=false --set seq.consistency_lambda=0"
python run_pipeline.py $COMMON --name seq_a_detrend $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4      $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_base    $PRE
python run_pipeline.py $COMMON --name seq_a_reltime --set seq.consistency_lambda=0
python run_pipeline.py $COMMON --name seq_a_nope    --set seq.consistency_lambda=0 --set seq.pos_encoding=none
python run_pipeline.py $COMMON --name seq_a_l3      --set seq.consistency_lambda=3
python run_pipeline.py $COMMON --name seq_a_xview                                   # CHAMPION, LB 0.8955

# Second seeds for the four gate variants -> makes DIS scoreable (>=3 variants needed for a rho).
python run_pipeline.py $COMMON --name seq_a_xview_s7   --set seed=7
python run_pipeline.py $COMMON --name seq_a_detrend_s7 --set seed=7 $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4_s7      --set seed=7 $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_reltime_s7 --set seed=7 --set seq.consistency_lambda=0

# ---- CANDIDATES (all vs the champion; 2 seeds each so DIS is computable) ----
# c_meanstd / c_meanmax : the round-07 pooling disagreement, settled by measurement not argument.
# c_compact             : 24 -> 14 channels. Capacity-REDUCING; two agents derived it independently.
# c_rank                : the FIRST genuine test of the amplitude question (per_cell_detrend never
#                         removed amplitude, it appended — so the toxicity law is unevidenced).
# c_antithetic          : makes the cross-view penalty informative on every row (overlap 2.37->1.28).
for S in "" "_s7"; do
  SEED=""; [ -n "$S" ] && SEED="--set seed=7"
  python run_pipeline.py $COMMON --name "c_meanstd$S"     $SEED --set seq.pooling=mean_std
  python run_pipeline.py $COMMON --name "c_meanmax$S"     $SEED --set seq.pooling=mean_max
  python run_pipeline.py $COMMON --name "c_compact$S"     $SEED --set seq.compact_missing=true
  python run_pipeline.py $COMMON --name "c_rank$S"        $SEED --set seq.channels.rank_replace=true
  python run_pipeline.py $COMMON --name "c_antithetic$S"  $SEED --set seq.antithetic_views=true
done

# ---- Retro-fit (re-certify the estimators) + SCREEN the candidates ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --screen c_meanstd c_meanmax c_compact c_rank c_antithetic

echo "=== done. NO UPLOAD. Paste back the RETRO-FIT table, the GATE lines, and the SCREEN table. ==="
