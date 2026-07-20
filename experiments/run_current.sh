#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 5 — Relative-time reframing (RESPONSE_04 idea A), CAPACITY-NEUTRAL.
#   Round-04 Deep Research proposed 5 ideas; triaged in gemini_loop/RESPONSE_04.md.
#   REJECTED as proven dead-ends: Saerens-EM prior (label-shift assumption broken by
#   our covariate shift; prevalence_target already hits the LB-verified 0.649), and
#   the Zou-et-al hardcoded-threshold water tree + EVI (non-transferable / already
#   failed). ACCEPTED & queued: MC temporal-dropout TTA + multi-seed bagging (both
#   capacity-neutral, banked for after a keeper).
#
#   Testing idea A FIRST because it's the only NEW capacity-neutral idea with a
#   plausible LARGE effect (only large effects clear the ~±0.01 public-LB noise):
#   left-align each observed 4-6mo window to t_rel=0 so the Transformer's positional
#   embeddings encode RELATIVE step, not calendar month — removing the calendar-
#   specific spectral memorization that the domain shift punishes. No added dims/params.
#   seq.relative_time=true; held at prevalence_target 0.649 so it's the ONLY variable
#   vs the 0.8780 champion. (relative_time=false reproduces champion bit-for-bit — verified.)
#
#   DECISION RULE: upload submission_seq_reltime.csv, gate vs 0.8780.
#     > 0.8780  -> relative-time transfers -> KEEP; then bank TTA + multi-seed bagging.
#     <=0.8780  -> revert; next probe = MC temporal-dropout TTA (inference-only, safest).
# =====================================================================
set -euo pipefail

python run_pipeline.py --full --model seq --name seq_reltime

echo "=== done. Upload submissions/submission_seq_reltime.csv (realized pos-rate 0.649) and paste the LB score ==="
