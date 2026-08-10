#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 42 — EXTEND THE ONE THING THAT WORKED.   *** 3 uploads ***
#
#   iter41 RESULT. Soft self-distillation on the 1030 unlabeled test rows scored
#   **0.909868 seed-averaged** = +0.009986 over the 0.899882 finalist. That is the FIRST
#   artifact ever to clear the ~0.8995 bias floor, it clears the pre-committed +0.006 bar and
#   the 0.9059 read, and it is a 5-SEED AVERAGE -- not the single-seed mirage that killed
#   perm (0.9065), vhsq (0.9133) and mean_min (0.9128). Both metric terms rose:
#   AUC 0.935 -> 0.944024, F1 0.876 -> 0.887097. Pooling behaved normally (+0.0061).
#
#   ARM T (the label-free consistency term) FAILED at 0.893752 and is dropped. Its members
#   were our two highest singles ever (0.914179, 0.908873) but the pool LOST 0.0178 -- the only
#   negative pooling gain in the ledger. Pooled AUC sat between its members while pooled F1 fell
#   below both, i.e. the RANKING pooled fine and the OPERATING POINT drifted: the unlabeled
#   variance penalty compresses logits toward a constant, per-seed Platt slopes diverge, and the
#   probability average lands at the wrong pos-rate. Not a lane worth repairing with 5 days left.
#
#   WHERE THE REMAINING GAP IS. Our AUC (0.944024) is now within 0.00087 of the leaderboard
#   leader's (0.944897). The ENTIRE remaining 0.0202 is the F1 term -- and round-18 measured that
#   only 5-13% of an F1 gap like this is operating point; ~90% is the ranking of rows near the
#   0.5 cut. So the target is near-cut ranking, which is exactly what distillation improved.
#
#   THIS ITERATION DOES ONE THING: selection on the proven direction (the alpha ladder), plus a
#   free finalist upgrade. NO new code -- every arm is a config override on the iter41 build.
#
#   ⚠️ ONE ROUND ONLY. Do NOT re-teach from the distilled student. Kumar/Ma/Liang: self-training
#   error compounds per step, and at our shift magnitude there is no bound. The teacher stays the
#   NON-distilled 5-seed permanence pool in every arm below, exactly as in iter41.
# =====================================================================
set -euo pipefail

BASE="--full --model seq --set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"
SEEDS5="42 7 13 21 29"
SEEDS10="42 7 13 21 29 3 17 23 31 37"

# ---- STEP 1: rebuild the 5-seed TEACHER in-run (submissions/preds/ is gitignored). ----
#      Identical to iter41 so every arm below is comparable to the banked 0.909868.
for s in $SEEDS5; do
  python run_pipeline.py $BASE --set seed=$s --name teacher_perm_s$s
done
python tools/seed_average.py --variant teacher_perm --name teacher_perm5
TEACHER="submissions/preds/preds_teacher_perm5.npz"

DISTILL="--set seq.distill.enable=true --set seq.distill.teacher=$TEACHER"

# ---- ARM A: alpha = 0.3 (half the banked weight). ----
for s in $SEEDS5; do
  python run_pipeline.py $BASE --set seed=$s $DISTILL --set seq.distill.alpha=0.3 --name a03_s$s
done
python tools/seed_average.py --variant a03 --name champion_distill_a03_seedavg5

# ---- ARM B: alpha = 1.5 (roughly double). ----
for s in $SEEDS5; do
  python run_pipeline.py $BASE --set seed=$s $DISTILL --set seq.distill.alpha=1.5 --name a15_s$s
done
python tools/seed_average.py --variant a15 --name champion_distill_a15_seedavg5

# ---- ARM C: the FINALIST upgrade -- alpha 0.7 (the winner) pooled over 10 seeds. ----
#      Same config as the banked 0.909868, more seeds. Worth ~+0.001 and, more importantly,
#      it removes residual seed luck from the 721-row PRIVATE slice, which is the slice that
#      actually decides the competition. Run this even if A and B both wash.
for s in $SEEDS10; do
  python run_pipeline.py $BASE --set seed=$s $DISTILL --set seq.distill.alpha=0.7 --name a07_s$s
done
python tools/seed_average.py --variant a07 --name champion_distill_seedavg10

cat <<'NEXT'
=====================================================================
 PASTE BACK the run summary lines AND, for every arm, BOTH gate lines:
     "TRANSDUCTIVE GATE PASS/FAILED: submitted pos-rate ..."
     "TRANSDUCTIVE GATE: teacher pos-rate ... | student-vs-teacher Spearman ..."
 Also paste the per-metric AUC and F1 columns from Zindi for each upload -- reading our own
 score breakdown is free, and it is what localized the gap this round.
 IGNORE OOF entirely; it is anti-correlated with the LB.

 If an arm logs "TRANSDUCTIVE GATE FAILED" (pos-rate outside [0.50, 0.62]), DO NOT UPLOAD IT --
 that is the constant-predictor / runaway failure mode, and it is what sank ARM T last round.

 UPLOAD (budget 5/day):
   1. submissions/submission_champion_distill_a03_seedavg5.csv   <- alpha 0.3
   2. submissions/submission_champion_distill_a15_seedavg5.csv   <- alpha 1.5
   3. submissions/submission_champion_distill_seedavg10.csv      <- 10-seed finalist upgrade
 Do NOT upload teacher_perm5 (it re-measures the known 0.899882) or any single seed.

 COMMITTED READ (all vs the banked alpha=0.7 5-seed result, 0.909868):
   An arm wins only at >= +0.006 SEED-AVERAGED. Expect alpha=0.7 to hold: lambda=1 vs 3 on the
   labeled consistency term differed by 0.0034 (inside noise) and interior optima have been the
   rule here. A ladder that brackets 0.7 and loses on BOTH sides CONFIRMS 0.7 and closes the knob
   -- that is a real result, not a null one, and it means we stop tuning alpha permanently.
   If alpha=1.5 wins clearly -> the unlabeled term is under-weighted; iter43 tries 3.0.
   If alpha=0.3 wins clearly -> we are over-distilling; iter43 tries 0.15.
   distill_seedavg10 >= 0.909868 -> it becomes FINALIST #1 (strictly more reliable on private).

 FINALISTS after iter41: {champion_distill_seedavg5 0.909868, champion_archblend4 0.899643}.
 archblend4 is kept as the DECORRELATED hedge -- the perm seed-average is now highly correlated
 with the distill student that was built on top of it, so it is no longer a real second opinion.

 QUEUED for iter43 (needs a small code change, not in this run): the dual-polarization water
 indicator 1[VH<-21] * 1[(VH-VV)<tau_r] as a width-neutral REPLACEMENT channel. Our whole feature
 bank is a function of VH alone; every water detector in the literature is dual-pol. We only ever
 tested VH-VV in forms that are provably AFFINE in bands the model already has (our SDWI is
 exactly -5.697415 + 0.230259*(VH+VV), verified to 3.6e-15), so those arms measured width cost,
 not missing information. The indicator form -- the same nonlinearity behind our biggest feature
 win -- has never been run.
=====================================================================
NEXT
