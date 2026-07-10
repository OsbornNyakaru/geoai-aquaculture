#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file,
# so the notebook never changes; the experiment is version-controlled here.
#
# ITERATION: Step 2 — GBDT + seq rank-average blend (fill empty preds/, probe).
# Config operating point is assumed_test_prior 0.65 (realized ~0.65 peak).
# After it runs, read the "OOF rank correlation" line from tools/blend.py:
#   < ~0.90  -> GBDT adds decorrelated signal -> upload submission_seq_gbdt*.csv
#              (pick the sweep file whose logged pos-rate is closest to 0.65),
#              gate vs current best LB 0.8780.
#   ~1.0     -> skip the blend, next push moves to Step 3 (invariant inputs).
# =====================================================================
set -euo pipefail

python run_pipeline.py --full --model gbdt --name gbdt_p65
python run_pipeline.py --full --model seq  --name seq_v3

python tools/blend.py \
  --preds submissions/preds/preds_seq_v3.npz submissions/preds/preds_gbdt_p65.npz \
  --weights 0.7 0.3 --name seq_gbdt --sweep 0.63 0.65 0.67

echo "=== done. Upload the submission_seq_gbdt*.csv nearest realized pos-rate 0.65 ==="
