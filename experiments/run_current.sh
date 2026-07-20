#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file,
# so the notebook never changes; the experiment is version-controlled here.
#
# ITERATION 3 — Step 3: seq + per_cell_detrend (level-invariant channel).
#   Step 2 (GBDT+seq blend) was DISCARDED: best blend 0.8705 < baseline 0.8780
#   (GBDT dilutes seq transfer despite higher OOF AUC — see experiments/LB_LOG.md).
#
#   Hypothesis: the domain shift is a per-series LEVEL offset (adversarial AUC
#   0.99 -> 0.94 on region-normalized indices). `per_cell_detrend` subtracts each
#   cell's own per-band temporal mean, removing that level, so the seq model should
#   transfer better. config.yaml sets seq.channels.per_cell_detrend=true and
#   calibration.prevalence_target=0.649 to hold the operating point at the EXACT
#   realized pos-rate that scored 0.8780 — so detrend is the ONLY variable vs the
#   0.8780 reference (clean isolation). This is also the first live test of the
#   Step-1 prevalence_target mechanism.
#
#   DECISION RULE: upload submission_seq_detrend.csv, gate vs 0.8780.
#     > 0.8780  -> detrend transfers -> KEEP the channel; next probe adds `deltas`.
#     <=0.8780  -> DISCARD detrend; next push tries `deltas` alone instead.
# =====================================================================
set -euo pipefail

python run_pipeline.py --full --model seq --name seq_detrend

echo "=== done. Upload submissions/submission_seq_detrend.csv (realized pos-rate 0.649) and paste the LB score ==="
