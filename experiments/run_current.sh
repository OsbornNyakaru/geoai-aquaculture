#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file,
# so the notebook never changes; the experiment is version-controlled here.
#
# ITERATION 4 — Robustness via MORE masking augmentation (seq.K 2 -> 4).
#   Prior probes both DISCARDED (see experiments/LB_LOG.md):
#     - Iter2 GBDT+seq blend: 0.8705  (adding a model class dilutes seq transfer)
#     - Iter3 per_cell_detrend: 0.8266 (adding input channels overfits source)
#   Pattern: this problem PUNISHES added capacity; OOF is blind to it.
#
#   So iter4 adds NOTHING to the model — it strengthens the DATA-side lever that
#   beat GBDT in the first place: train each row into MORE test-like masked views
#   (seq.K 2->4). No new input dims, no new model. Operating point still held at
#   prevalence_target 0.649, so K is the only variable vs the 0.8780 reference.
#
#   DECISION RULE: upload submission_seq_k4.csv, gate vs 0.8780.
#     > 0.8780  -> more augmentation transfers -> KEEP K=4; consider K=6 / n_repeats↑.
#     <=0.8780  -> revert K=2. THREE failed toggles => STOP guessing; escalate to the
#                 research loop: write gemini_loop/UPDATE_04.md for fresh sourced ideas.
# =====================================================================
set -euo pipefail

python run_pipeline.py --full --model seq --name seq_k4

echo "=== done. Upload submissions/submission_seq_k4.csv (realized pos-rate 0.649) and paste the LB score ==="
