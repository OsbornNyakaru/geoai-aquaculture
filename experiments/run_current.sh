#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 9 — Cross-view invariance objective, CAPACITY-NEUTRAL (objective-level).
#   RESPONSE_05 idea #4. The positional lane is EXHAUSTED (iter5 relative-time +0.0128 WON;
#   iter7 dnorm −0.0064 and iter8 NoPE +0.0009 both < +0.003 -> stop-rule). Budget shifts
#   to the objective lever.
#
#   Mechanism: each train row already spawns K=2 masked "views". Add a consistency penalty
#   on the two views of the SAME row: L = BCE + lambda*mean((logit_v1 - logit_v2)^2). This
#   teaches the model label-invariance to WHICH observation window it sees — attacking the
#   window-identity nuisance the designed shift exploits. It is an OBJECTIVE / inductive-bias
#   change (structurally like the winning reframe), NOT an inference robustness add-on like
#   TTA (which was within-noise). Capacity-neutral: no new params/dims. seq.consistency_lambda=1.0
#   is the ONLY variable vs the 0.8908 champion; lambda=0 reproduces it bit-for-bit. Built on the
#   champion base (pos_encoding=learned, relative_time=true), held at prevalence_target 0.649.
#
#   NOTE: NoPE (iter8, submission_seq_nope.csv, 0.8917) is LOCKED as the diverse private-LB finalist.
#
#   DECISION RULE: upload submission_seq_xview.csv, gate vs 0.8908.
#     > 0.8908 (esp. >= +0.005) -> invariance objective transfers -> KEEP.
#     within noise / clear drop -> revert (consistency_lambda:0). The capacity-neutral lane is then
#        largely exhausted -> ENDGAME: one-time prevalence sweep (pick plateau center) + finalize
#        the two diverse picks (champion relative-time + NoPE). Confirm Zindi's finalist mechanism.
# =====================================================================
set -euo pipefail

python run_pipeline.py --full --model seq --name seq_xview

echo "=== done. Upload submissions/submission_seq_xview.csv (realized pos-rate 0.649) and paste the LB score ==="
