#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 45 — THE FINALIST LOCK. One artifact, one upload, then stop.
#
#   iter44 RESULT: NULL on both regime-matched arms, exactly as pre-registered.
#       champion_alphamix10_regimematch   0.907109506  (+0.001006)
#       champion_dualpol_add_regimematch  0.910446704  (+0.002831)
#   Both inside +-0.006. The boundary-calibration lane is CLOSED, and closed on a MEASUREMENT:
#   delta_hat = F*_masked/2 landed in [0.4791, 0.4852] across all 30 seed x regime combinations,
#   decisively above the pre-registered 0.47 kill threshold, on the branch we declared ROBUST in
#   advance (delta_hat is a LOWER bound, so "the cut is fine" cannot be an artifact of the bound).
#
#   THREE INDEPENDENT INSTRUMENTS NOW AGREE THE OPERATING POINT IS RIGHT: the graph gate (implied
#   test pos-rate 0.591 vs our realized 0.587), delta_hat (>= 0.48 against a 0.50 cut), and moving
#   the cut directly (+0.001 to +0.003). The hypothesis that we are short ~9 true positives BECAUSE
#   the cut is too conservative is dead three ways. **Do not reopen it.**
#
#   WHAT THIS ITERATION IS FOR. Not an experiment. iter44 surfaced one unresolved FINALIST question
#   and this settles it, then we stop.
#
#   THE QUESTION. Our two best artifacts differ on which half of the metric they win:
#       champion_distill_alphamix10        0.906104   AUC 0.942680   10 distinct seeds, alpha mixed
#       champion_dualpol_add_regimematch   0.910447   AUC 0.946387    5 seeds,  alpha = 0.7
#   The composite gap (+0.0033) is INSIDE the ~0.006 paired noise band -- chasing it would be the
#   exact mistake this ledger warns about. But the AUC gap (+0.0038) is NOT noise: across every
#   experiment we have run, the AUC term has been the stable one (the entire alpha ladder moved it
#   0.0017 total; alpha=0.7 at 5 and at 10 seeds were BIT-IDENTICAL at 0.944024425). A +0.0038 AUC
#   edge sits outside that term's whole observed spread. And AUC is 40% of the metric, applies to
#   all 721 private rows, and never coin-flips at a cut.
#
#   So the dual-pol gate really does rank better -- iter43 and iter44 both measured it, and it is
#   the only artifact we have ever produced whose AUC beats the public leader's 0.944897. What it
#   lacks is seeds. Rather than trade the ranking for the variance reduction, take both.
#
#   THE ARTIFACT: champion_dualpolmix10 = the alphamix10 recipe EXACTLY, plus the dual-pol gate.
#   Same 10 distinct seeds, same alpha marginalization over {0.7, 1.5}, same teacher, same
#   calibration. It differs from the current finalist #1 in ONE variable: the gate channel
#   (26 channels vs 25). That is the cleanest comparison this project has ever been able to set up.
#
#   Why marginalize alpha rather than pick 0.7: iter42 closed the alpha knob EXACTLY -- a 5x sweep
#   moved TP by one row out of 309. When a knob is that flat you do not pick a setting, you average
#   over it, which removes the choice from the private-slice gamble at zero cost. That reasoning is
#   what made ARM E finalist #1 in the first place; this applies it to the better-ranking base.
#
#   CALIBRATION: R=1 (regime-matched), per the standing pre-commitment in tools/regime_match.py --
#   "R=1 ships whatever positive rate it produces." iter44's null says the choice barely matters
#   (2-3 rows); it does NOT say R=2 is better. Reverting a principled a-priori choice because it
#   failed to pay would be exactly the post-hoc renegotiation we forbid ourselves.
#
#   ⚠️ ONE ROUND OF DISTILLATION ONLY. Teacher is the NON-distilled 5-seed permanence pool, as in
#   every previous round. Do NOT re-teach from a distilled student (Kumar/Ma/Liang: error compounds).
# =====================================================================
set -euo pipefail

BASE="--full --model seq"
PERM="--set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"
DP="--set seq.channels.dualpol_gate=true"
SEEDS5="42 7 13 21 29"

# ---- STEP 1: rebuild the 5-seed TEACHER in-run (submissions/preds/ is gitignored). ----
#      Byte-identical to iter41-44, so every number below stays comparable to the banked scores.
for s in $SEEDS5; do
  python run_pipeline.py $BASE $PERM --set seed=$s --name teacher_perm_s$s
done
python tools/seed_average.py --variant teacher_perm --name teacher_perm5
TEACHER="submissions/preds/preds_teacher_perm5.npz"

DISTILL="--set seq.distill.enable=true --set seq.distill.teacher=$TEACHER"

# ---- STEP 2: champion_dualpolmix10 — 10 DISTINCT seeds, alpha marginalized, dual-pol gate ON. ----
#      Seeds and alphas are IDENTICAL to champion_distill_alphamix10 so the two artifacts are a
#      clean one-variable pair. Width MUST log 26 channels/month on every run; 25 would mean the
#      gate silently failed to attach and the whole iteration is void.
for s in $SEEDS5; do
  python run_pipeline.py $BASE $PERM $DP --set seed=$s \
    $DISTILL --set seq.distill.alpha=0.7 --name dpam_s$s
done
for s in 3 17 23 31 37; do
  python run_pipeline.py $BASE $PERM $DP --set seed=$s \
    $DISTILL --set seq.distill.alpha=1.5 --name dpam_s$s
done
python tools/seed_average.py --variant dpam --name champion_dualpolmix10

# ---- STEP 3: THE CONTROL. Must reproduce seed_average bit-for-bit before we trust R=1. ----
python tools/regime_match.py --variant dpam --name _control_dpam --views all
if cmp -s "submissions/submission_champion_dualpolmix10.csv" \
          "submissions/submission__control_dpam.csv"; then
  echo "CONTROL PASS (dpam): regime_match --views all == seed_average, bit for bit."
else
  echo "CONTROL FAIL (dpam): the per-view rebuild does not reproduce seed_average. ABORTING."
  exit 1
fi

# ---- STEP 4: the shipped calibration. ----
python tools/regime_match.py --variant dpam --name champion_dualpolmix10_regimematch

cat <<'NEXT'
=====================================================================
 PASTE BACK:
   - the "seq input width: NN channels/month" line for ANY dpam run. It MUST say 26.
     If it says 25 the dual-pol gate did not attach and this iteration is VOID.
   - the "Pairwise rank correlation between seeds" line from the dpam pool.
   - the "=== SUMMARY (means across seeds) ===" block and the "ROWS CROSSING" line.
   - the "CONTROL PASS/FAIL (dpam)" line.
   - the Zindi AUC and F1 columns for the upload (not just the composite).
   - ⚠️ whether Cell 5b printed "copied to MyDrive/geoai-preds/".

 UPLOAD (1 file only):
   submissions/submission_champion_dualpolmix10_regimematch.csv
 Do NOT upload champion_dualpolmix10 (the R=all version) or _control_dpam. We already measured the
 R=1 vs R=all difference in iter44 and it was a null; re-measuring it would waste a slot.

 COMMITTED READ (pre-registered; do not renegotiate after seeing the numbers):

 ⚠️ THIS IS A VARIANCE DECISION, NOT A LEVEL DECISION. It is deliberately NOT gated on beating
 0.910447. Both candidate finalists sit inside one noise band of each other on composite; what is
 being bought is the RANKING edge (real, +0.0038 AUC) plus 10-seed variance reduction (real), and
 neither is visible in a single public composite. Expect it to land anywhere in 0.905-0.913. That
 whole range is two or three rows.

   AUC >= ~0.945           -> the gate's ranking edge SURVIVED seed expansion. champion_dualpolmix10
                              _regimematch becomes FINALIST #1. This is the expected outcome and
                              the reason for the run: iter42 showed 5-vs-10 seeds leaves AUC
                              bit-identical, so the gate's AUC should carry over intact.
   AUC ~0.943 (amix level) -> the edge did NOT survive; it was a 5-seed artifact after all. Keep
                              champion_distill_alphamix10 as finalist #1 and record that the iter43
                              and iter44 AUC readings were correlated, not independent.
   composite < ~0.902      -> something is actually WRONG (not noisy). Investigate before
                              designating anything; do not just pick the other artifact.

 Note the AUC branch, not the composite, is the deciding one. That is the whole point: we are
 choosing on the term that is stable and applies to all 721 private rows, not on the term that
 resolves in 0.0055 quanta on 309 public rows.

 FINALIST #2 stays champion_archblend4 (0.899643) either way. It is the DECORRELATED hedge: every
 distill artifact is built on the same teacher, so they are not second opinions of each other, and
 archblend4 predates the whole distillation lane.

 DEADLINE 2026-08-16. THIS IS THE LAST RUN. Whatever it returns, iter46 is the code-review package
 (35% of the final score) and nothing else. No further experiments.
=====================================================================
NEXT
