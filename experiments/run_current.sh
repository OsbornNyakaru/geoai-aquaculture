#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 10 — Cross-view invariance strength probe (lambda 1.0 -> 3.0), CAPACITY-NEUTRAL.
#
#   *** ITER9 WON (soft). *** consistency_lambda=1.0 scored 0.8955 — NEW BEST (+0.0047 vs the
#   0.8908 relative-time champion; our highest public score). Mechanism: the invariance penalty
#   REDUCED OVERCONFIDENCE (oof_auc 0.9936->0.9894, prevalence delta 2.03->1.30, t*=0.445) — it
#   hit the model's diagnosed weakness ("strong ranker, poor probabilities"). +0.0047 is at the
#   edge of the +-0.01 noise band, so this probe tests whether the lever is REAL and has MORE to give.
#
#   Push the invariance harder: lambda=3.0. Hypothesis: if reduced overconfidence is genuinely
#   improving transfer, more invariance -> a clearer, noise-resolvable gain. Risk: too much pull
#   collapses the ranker (AUC is 40% of the metric). consistency_lambda=3.0 is the ONLY variable
#   vs the 0.8955 champion (relative-time + lambda=1.0); held at prevalence_target 0.649.
#
#   NoPE (submission_seq_nope.csv, 0.8917) stays LOCKED as the diverse private-LB finalist.
#
#   DECISION RULE: upload submission_seq_xview_l3.csv, gate vs 0.8955.
#     clearly > 0.8955 (>= +0.005) -> lever is real & scales -> KEEP; consider a tight lambda sweep.
#     ~tie (within noise)          -> lambda=1.0 is the setting -> revert to 1.0; go to ENDGAME.
#     drop                         -> over-regularized -> revert to 1.0 (champion); go to ENDGAME.
#   ENDGAME = one-time prevalence sweep (plateau center) + finalize picks (xview lambda=1.0 + NoPE).
# =====================================================================
set -euo pipefail

python run_pipeline.py --full --model seq --name seq_xview_l3

echo "=== done. Upload submissions/submission_seq_xview_l3.csv (realized pos-rate 0.649) and paste the LB score ==="
