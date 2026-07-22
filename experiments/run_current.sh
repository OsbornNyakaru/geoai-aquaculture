#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 13 — SECOND OFFLINE SCREEN.  *** COSTS ZERO SUBMISSIONS ***
#
#   iter12 outcome: ALL FIVE CANDIDATES HELD (no candidate got >=2 votes). The pre-committed
#   rule worked exactly as intended -- ATC-F1 liked mean_max a lot (+0.0838) but DIS disagreed
#   (-0.0301), so we spent zero submissions on it. Two real findings came out of it:
#
#     1. c_rank COLLAPSED: OOF 0.9753 -> 0.857/0.865 across both seeds, ATC-F1 -0.1703.
#        Replacing absolute band values with within-series RANK destroys the model. So
#        AMPLITUDE IS THE PRIMARY SIGNAL -- the pond discriminator really is "persistently
#        LOW backscatter", an absolute level. The amplitude question, reopened when we found
#        per_cell_detrend had only ever APPENDED channels, is now genuinely ANSWERED, and the
#        rank/ordinal feature family is closed with evidence rather than by assumption.
#     2. c_compact WAS NEVER TESTED -- a config-path bug (seq.compact_missing instead of
#        seq.channels.compact_missing) meant the flag never reached to_inputs(). The run came
#        out bit-identical to the champion and the screen scored the no-op as a 0.0000 tie.
#        FIXED; it is re-tested below, and the pipeline now logs the ACTUAL input width.
#
#   THE ESTIMATORS RE-CERTIFIED IDENTICALLY on both runs (ATC-F1 15/15 rho=+0.964; DIS 5/5
#   rho=+1.000), so the screen itself is trustworthy. Note DIV FAILED (2/15, rho=-0.857):
#   fold-diversity is strongly ANTI-correlated with LB, the OPPOSITE of hypothesis H1.
#
#   THIS ROUND screens four candidates:
#     c_compact     the genuine re-test of the 24 -> 14 channel deletion (expect n=14 in the log)
#     c_meanmax_l0  mean_max WITHOUT cross-view invariance. ATC-F1 loved mean_max but DIS said it
#                   was seed-unstable; the zero-init second-moment head plus a variance penalty may
#                   be fighting each other. This separates the two.
#     c_k3          K=3 views. K=2 beat K=4, but K=3 was never tried and the optimum was declared
#                   "sharp" from two points. Capacity-neutral in parameters.
#     c_dropout3    dropout 0.2 -> 0.3. Exactly parameter-neutral, and under a design law that says
#                   "less fit transfers better" this is the most on-thesis knob in the repo -- and
#                   it has never been touched once in twelve iterations.
#
#   PRE-COMMITTED RULE (unchanged; do not renegotiate after seeing numbers):
#     >=2 cleared estimators above champion -> submit it. 1 or 0 -> HOLD, costs nothing.
#     If everything HOLDS again, the structural lane is exhausted and the Presto lane
#     (RESEARCH_07.md 5e) becomes the next spend.
#
#   Committed config remains the exact 0.8955 champion; every variant comes from --set.
#   Runtime ~21 runs, roughly 20 minutes.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- Anchors: re-certify the estimators against THIS code every run. ----
# If ATC-F1 or DIS stops clearing, something regressed and the SCREEN BELOW IS VOID.
PRE="--set seq.relative_time=false --set seq.consistency_lambda=0"
python run_pipeline.py $COMMON --name seq_a_detrend $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4      $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_base    $PRE
python run_pipeline.py $COMMON --name seq_a_reltime --set seq.consistency_lambda=0
python run_pipeline.py $COMMON --name seq_a_nope    --set seq.consistency_lambda=0 --set seq.pos_encoding=none
python run_pipeline.py $COMMON --name seq_a_l3      --set seq.consistency_lambda=3
python run_pipeline.py $COMMON --name seq_a_xview                                   # CHAMPION, LB 0.8955

# Second seeds for the four gate variants -> makes DIS scoreable.
python run_pipeline.py $COMMON --name seq_a_xview_s7   --set seed=7
python run_pipeline.py $COMMON --name seq_a_detrend_s7 --set seed=7 $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4_s7      --set seed=7 $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_reltime_s7 --set seed=7 --set seq.consistency_lambda=0

# ---- CANDIDATES (2 seeds each so DIS is computable) ----
# WATCH THE LOG: c_compact MUST print "seq input width: 14 channels/month". If it prints 24 the
# flag did not take effect again and its screen row is meaningless -- report that, do not read it.
for S in "" "_s7"; do
  SEED=""; [ -n "$S" ] && SEED="--set seed=7"
  python run_pipeline.py $COMMON --name "c_compact$S"    $SEED --set seq.channels.compact_missing=true
  python run_pipeline.py $COMMON --name "c_meanmax_l0$S" $SEED --set seq.pooling=mean_max --set seq.consistency_lambda=0
  python run_pipeline.py $COMMON --name "c_k3$S"         $SEED --set seq.K=3
  python run_pipeline.py $COMMON --name "c_dropout3$S"   $SEED --set seq.dropout=0.3
done

# ---- Retro-fit (re-certify) + SCREEN ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --screen c_compact c_meanmax_l0 c_k3 c_dropout3

echo "=== done. NO UPLOAD. Paste back the RETRO-FIT table, the GATE lines, and the SCREEN table. ==="
