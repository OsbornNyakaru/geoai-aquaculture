#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 7 — Duration-normalized fractional positional encoding, CAPACITY-NEUTRAL.
#   Round-05 Deep Research (Gemini + Claude, both independent) ranked THIS idea #1 —
#   triaged in gemini_loop/RESPONSE_05.md. It is the literal next deletion on the axis
#   that just won: iter5 relative-time removed absolute window START (+0.0128); this
#   removes absolute window LENGTH.
#
#   Mechanism: after relative_time left-aligns the observed window to offsets 0..L-1,
#   re-index each position by the fractional coordinate p = offset/(L-1) in [0,1] so a
#   4-, 5-, and 6-month window all share ONE relative frame; read the positional vector
#   by linearly interpolating the existing learned length-12 table at index j = p*11.
#   PARAMETER-NEUTRAL (reuses the champion's table; reduces to the champion exactly when
#   L=12). Touches only the TIME coordinate — stays off the toxic amplitude axis
#   (detrend was -0.0514). seq.pos_encoding=dnorm is the ONLY variable vs the 0.8908
#   champion; pos_encoding=learned reproduces it bit-for-bit. Held at prevalence_target 0.649.
#
#   Rejected re-treads from round 05 (see RESPONSE_05.md): Saerens-EM/MLLS prior (3rd
#   rejection; label-shift assumption), Zou water-tree / WIF / EVI indices (dead-end +
#   toxic amplitude axis), CAST self-training (ESS-collapse family), CropNet blend + big-bang
#   bundle. Banked for later: NoPE set encoder (iter8, seq.pos_encoding=none, already coded),
#   cross-view invariance objective (iter9), one-time prevalence sweep.
#
#   DECISION RULE: upload submission_seq_dnorm.csv, gate vs 0.8908.
#     >= 0.8908 (esp. >= +0.005) -> length reframe transfers -> KEEP; bank NoPE (iter8).
#     within noise            -> keep as a diversity candidate, don't iterate; go to NoPE.
#     clear drop              -> revert (pos_encoding:learned); go to NoPE.
# =====================================================================
set -euo pipefail

python run_pipeline.py --full --model seq --name seq_dnorm

echo "=== done. Upload submissions/submission_seq_dnorm.csv (realized pos-rate 0.649) and paste the LB score ==="
