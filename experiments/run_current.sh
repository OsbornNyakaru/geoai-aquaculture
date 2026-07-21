#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 6 — MC temporal-dropout TTA (RESPONSE_04 idea C), CAPACITY-NEUTRAL,
#   INFERENCE-ONLY, stacked on the NEW champion.
#
#   *** ITER5 WON. *** Relative-time reframing scored 0.8908 vs 0.8780 (+0.0128,
#   clears the ±0.01 public-LB noise) — the first improvement since the champion and
#   the first confirmation that a capacity-NEUTRAL structural change transfers where
#   every capacity-ADDING change lost. relative_time is now the champion (config default
#   true), held at prevalence_target 0.649.
#
#   Iter6 banks the first robustness move on top of it. At inference, each fold's model
#   predicts the clean test view PLUS n_views=8 views that each mask 1-2 random ACTIVE
#   months per test row, then soft-votes. No new params/dims — it only averages over
#   WHICH months a short 4-6mo window happens to observe (the axis the covariate shift
#   rides on). OOF is left untouched (blind). seq.tta.enable=true is the ONLY new variable
#   vs the 0.8908 champion; enable=false reproduces it bit-for-bit.
#
#   DECISION RULE: upload submission_seq_reltime_tta.csv, gate vs 0.8908.
#     >= 0.8908 (or within ~0.004, i.e. no real regression) -> KEEP as standing default:
#         it's variance insurance for the private split even if public is within noise;
#         then bank iter7 = multi-seed bagging (seq.n_repeats up, capacity-neutral).
#     clear drop (< ~0.885) -> TTA distorts short-window preds -> revert (enable:false);
#         go straight to iter7 multi-seed bagging.
# =====================================================================
set -euo pipefail

python run_pipeline.py --full --model seq --name seq_reltime_tta

echo "=== done. Upload submissions/submission_seq_reltime_tta.csv (realized pos-rate 0.649) and paste the LB score ==="
