#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 41 — THE TRANSDUCTIVE LANE: use the 1030 UNLABELED TEST ROWS.  *** 2 uploads ***
#
#   WHY NOW. Round-18 deep research (8 agents) converged from two independent directions on the
#   SAME conclusion, and it retires three lanes at once:
#     * 0.8995 is a BIAS floor, not a variance floor. Four different constructions seed-average to
#       0.899882 / 0.899643 / 0.899512 / 0.899512 -- a spread of 0.00037 against a public-LB binomial
#       noise of +-0.012. Pooling/snapshots/bagging have <= +0.005 left in TOTAL, forever.
#     * The bias IS the covariate shift. The only levers that touch it are (i) use test-distribution
#       information, or (ii) change the inductive bias at zero width. This iteration is (i).
#   The 1030 test rows are 57% of our labeled set and are the ONLY target-domain data we have. No
#   other lever in this project adds any. Test FEATURES are supplied unlabeled -> transduction is
#   legal by construction: no external data, no test labels, no threshold tuning. The literal 0.5
#   cut and train-only Platt calibration are untouched.
#
#   WHY THIS SHAPE. The ledger's design law is that CAPACITY-ADDS LOSE (per_cell_detrend -0.0514,
#   every iter34 second-channel arm -0.006..-0.018) while OBJECTIVE CHANGES WIN (cross-view
#   invariance +0.0047, relative-time reframe +0.0128). BOTH arms below add exactly ZERO parameters.
#
#   ARM T (transductive consistency) -- point our already-proven Var_k(logit) cross-view penalty at
#     the TEST rows. Forms NO label, so it is structurally immune to confirmation bias, pseudo-label
#     prior drift, and the confirmed conditional-shift bias. Views are contiguous SUB-WINDOWS of each
#     row's visible block (measured test support L in {4,5,6}: 345/343/342 rows, 1030/1030 contiguous)
#     -- never hole-punching, which is off-manifold and is the diagnosed cause of the iter6 TTA loss.
#     345 rows at L=4 have only one legal view, so their term is 0 by construction; 685 carry it.
#   ARM D (soft self-distillation) -- train against the banked 5-seed teacher's SOFT probabilities on
#     test rows (T=1, never thresholded). Chen/Wei/Kumar/Ma 2020 prove self-training on unlabeled
#     TARGET data drives a classifier OFF features that correlate with the label in the source domain
#     but not the target -- a line-by-line description of our failure mode -- given a decent source
#     classifier, which at 0.90 we have.
#
#   VERIFIED BEFORE PUSH: with both flags off the pipeline is BIT-FOR-BIT the champion (pristine HEAD
#   and this tree both give fold scores 0.89040 / 0.89470, Platt slope 1.673, pos-rate 0.600).
# =====================================================================
set -euo pipefail

# The champion: single-tau permanence, relative-time, cross-view lambda=1.0, legal 0.5 cut.
BASE="--full --model seq --set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"
SEEDS="42 7 13 21 29"

# ---- STEP 1: rebuild the 5-seed TEACHER in-run. submissions/preds/ is gitignored, so the teacher
#      must be regenerated here rather than pulled from a previous session. This also re-measures the
#      0.899882 baseline for a matched-seed paired comparison. ----
for s in $SEEDS; do
  python run_pipeline.py $BASE --set seed=$s --name teacher_perm_s$s
done
python tools/seed_average.py --variant teacher_perm --name teacher_perm5

TEACHER="submissions/preds/preds_teacher_perm5.npz"

# ---- ARM T: transductive cross-view consistency on the test rows (no labels formed). ----
for s in $SEEDS; do
  python run_pipeline.py $BASE --set seed=$s \
    --set seq.transduct.enable=true --set seq.transduct.lambda_u=0.5 \
    --name tcons_s$s
done
python tools/seed_average.py --variant tcons --name champion_tcons_seedavg5

# ---- ARM D: soft self-distillation against the 5-seed teacher. ----
for s in $SEEDS; do
  python run_pipeline.py $BASE --set seed=$s \
    --set seq.distill.enable=true --set seq.distill.alpha=0.7 \
    --set seq.distill.teacher=$TEACHER \
    --name distill_s$s
done
python tools/seed_average.py --variant distill --name champion_distill_seedavg5

cat <<'NEXT'
=====================================================================
 PASTE BACK the run summary lines AND, for each arm, the two gate lines:
     "TRANSDUCTIVE GATE PASS/FAILED: submitted pos-rate ..."
     "TRANSDUCTIVE: raw test pos-rate ... | oof_auc ..."
 IGNORE OOF entirely -- it is anti-correlated with the LB. Only the paired LB is truth.

 FREE ABORT GATE (already automated, costs no submission): if an arm logs
 "TRANSDUCTIVE GATE FAILED" (submitted pos-rate outside [0.50, 0.62]) then the unlabeled term has
 won -- either constant-predictor collapse or self-training runaway. DO NOT UPLOAD THAT ARM.

 UPLOAD (budget 5/day, upload only arms that PASSED the gate):
   1. submissions/submission_champion_tcons_seedavg5.csv     <- ARM T (label-free; the safer bet)
   2. submissions/submission_champion_distill_seedavg5.csv   <- ARM D (soft self-distillation)
 Do NOT upload teacher_perm5 -- it re-measures the known 0.899882 baseline and would waste a slot.

 COMMITTED READ (matched 5-seed paired vs the 0.899882 finalist; bar is +0.006 = the seed-noise gate):
   >= 0.9059  -> a REAL seed-robust win. Transduction is the lever -> iter42 stacks the two arms and
                 runs the alpha / lambda_u ladder, then re-designates finalists.
   0.894-0.906 -> wash. The transductive lane is capped like everything else -> stop LB-chasing and
                 spend the remaining days on the Phase-Two writeup (35% of the top-5 score, UNBUILT).
   <= 0.894   -> the unlabeled term is actively harmful -> revert, keep the two current finalists.
   ARM T vs ARM D tells us whether forming a (soft) label helps or hurts under conditional shift.

 NOTE finalists stay {perm seed-avg 0.899882, archblend4 0.899643} unless an arm clears 0.9059
 SEED-AVERAGED. Three single-seed-42 highs (0.9065, 0.9133, 0.9128) already washed to ~0.8995 --
 never designate on one lucky draw.
=====================================================================
NEXT
