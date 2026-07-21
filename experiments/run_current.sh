#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 11 — BUILD AN OFFLINE LB-PREDICTING VALIDATOR.  *** COSTS ZERO SUBMISSIONS ***
#
#   Round-06 research (gemini_loop/RESPONSE_06.md) had BOTH reports independently rank this
#   #1: our binding constraint is MEASUREMENT (309-row public LB, ~+-0.01 noise, and local OOF
#   is anti-correlated), so the highest-value move is a local signal that predicts the LB.
#
#   The method is self-certifying. We regenerate 7 historical runs whose public LB we ALREADY
#   KNOW (0.8266 ... 0.8955), estimate each one's test performance from the 1030 UNLABELED test
#   rows, and check whether the estimator RANKS them correctly. No submission is involved, and
#   if every estimator fails we have learned that for free.
#
#   Estimators (tools/offline_validate.py):
#     ATC   Average Thresholded Confidence  (Garg et al., ICLR 2022) - verified on synthetic
#           data to rank true test accuracy at Spearman rho = 1.0.
#     DIS   two-seed disagreement on test   (GDE / agreement-on-the-line, Baek et al. 2022)
#     MARG  mean post-shift |p-0.5|         - deliberate naive control the others must beat.
#
#   PRE-COMMITTED GATE (RESPONSE_06.md section 6):
#     An estimator must place detrend (LB 0.8266) and K=4 (0.8665) BELOW reltime (0.8908) and
#     xview (0.8955), with Spearman rho > 0.7.
#       PASS -> the noise floor is broken. Screen the gated backlog offline (fold-ensemble
#               deletion, group-DRO, VH-VV, AUC surrogate) and submit only when >=2 estimators
#               beat the champion. The ~80 remaining submissions become a real search budget.
#       FAIL -> Q1 is dead for 0 submissions; fund only ideas with plausible effect >= +0.013.
#               iter12 (dispersion pooling) and iter13 (focal loss) still ship on their merits.
#
#   Runtime ~8 x 7 min. Nothing here is uploaded to Zindi.
#   NOTE: the champion config in config.yaml is UNCHANGED - every variant below is produced by
#   --set overrides, so the repo's committed state stays the exact 0.8955 champion.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- Anchors: historical variants with known public LB (see experiments/anchors.tsv) ----
# pre-relative-time era (relative_time=false, consistency_lambda=0)
python run_pipeline.py $COMMON --name seq_a_detrend \
  --set seq.relative_time=false --set seq.consistency_lambda=0 \
  --set seq.channels.per_cell_detrend=true                        # LB 0.8266
python run_pipeline.py $COMMON --name seq_a_k4 \
  --set seq.relative_time=false --set seq.consistency_lambda=0 --set seq.K=4   # LB 0.8665
python run_pipeline.py $COMMON --name seq_a_base \
  --set seq.relative_time=false --set seq.consistency_lambda=0    # LB 0.8780

# relative-time era
python run_pipeline.py $COMMON --name seq_a_reltime \
  --set seq.consistency_lambda=0                                  # LB 0.8908
python run_pipeline.py $COMMON --name seq_a_nope \
  --set seq.consistency_lambda=0 --set seq.pos_encoding=none      # LB 0.8917
python run_pipeline.py $COMMON --name seq_a_l3 \
  --set seq.consistency_lambda=3                                  # LB 0.8921
python run_pipeline.py $COMMON --name seq_a_xview                 # LB 0.8955 (CHAMPION)

# second seed of the champion -> enables the two-seed disagreement estimator.
# (This is also the queued seed-replication run: submitting submission_seq_a_xview_s7.csv
#  later would measure our true seed-to-seed LB spread for 1 submission.)
python run_pipeline.py $COMMON --name seq_a_xview_s7 --set seed=7

# ---- The retro-fit: do any of these estimators rank the known LB correctly? ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv

echo "=== done. NO UPLOAD NEEDED. Paste the RETRO-FIT table + GATE lines back to the agent. ==="
