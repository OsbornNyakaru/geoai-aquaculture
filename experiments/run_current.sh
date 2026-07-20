#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# LOOP PAUSED FOR RESEARCH (2026-07-20). Three straight blind toggles lost:
#     Iter2 GBDT+seq blend 0.8705 · Iter3 per_cell_detrend 0.8266 · Iter4 K=4 0.8665.
#   Champion is UNCHANGED: seq K=2 @ realized 0.649 = 0.8780. See:
#     - gemini_loop/UPDATE_04.md  (research brief — paste into Gemini/Claude Deep Research)
#     - experiments/LB_LOG.md     (the three negatives + the noise-floor finding)
#     - gemini_loop/AGENT_BRIEF.md(meta-lesson: added capacity hurts; OOF is anti-correlated)
#
#   Do NOT spend more single-toggle submissions until a research round returns a
#   LARGE-effect, rule-legal idea. This script now only REGENERATES the champion
#   submission (config reverted to K=2, channels off, prevalence_target 0.649) so
#   there is a clean known-good file to submit if needed. Running it is safe/cheap
#   and re-confirms the 0.8780 anchor; it does not test anything new.
# =====================================================================
set -euo pipefail

python run_pipeline.py --full --model seq --name seq_champion_k2

echo "=== champion regenerated: submissions/submission_seq_champion_k2.csv (realized 0.649). ==="
echo "=== Next real move: run gemini_loop/UPDATE_04.md through Deep Research; paste findings back. ==="
