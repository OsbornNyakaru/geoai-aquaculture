#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# LOOP PAUSED for research round 05. Champion = clean relative-time net (LB 0.8908):
#   seq.relative_time=true, seq.tta.enable=false, held at prevalence_target 0.649.
#   Running this file as-is REGENERATES the champion submission_seq_reltime.csv (0.8908) —
#   safe to re-run to re-confirm the anchor; nothing new to submit until iter7 is staged.
#
#   STATUS: iter5 relative-time WON 0.8908 (+0.0128, broke the 0.8780 plateau). iter6 MC
#   temporal-dropout TTA DISCARDED (0.8885, within noise, did not beat champion). The user
#   chose a research round over banking more robustness — because the ONLY changes that have
#   moved this LB are capacity-neutral STRUCTURAL reframes, and the transferable axis is
#   temporal/POSITIONAL (relative-time +0.013), NOT amplitude (detrend −0.051).
#
#   NEXT: paste gemini_loop/UPDATE_05.md into Deep Research → the coding agent triages the
#   reply into RESPONSE_05.md and stages iter7 = the next positional-family reframe here.
# =====================================================================
set -euo pipefail

python run_pipeline.py --full --model seq --name seq_reltime

echo "=== champion regen. submissions/submission_seq_reltime.csv should reproduce LB 0.8908. Loop paused pending UPDATE_05 research. ==="
