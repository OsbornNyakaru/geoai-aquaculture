#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 44 — INSTRUMENT THE CALIBRATION SET, THEN CORRECT ONE MEASURABLE DEFECT.  *** 2 uploads ***
#
#   iter43 RESULT: the VH−VV lane is CLOSED (three independent forms of the same quantity failed).
#   ARM E banked as finalist #1 at 0.906104. ARM G posted AUC 0.946460 — our highest ever and above
#   the leader's 0.944897 — while still losing at the cut. The read stands: our RANKING has caught
#   the leader; the whole remaining gap is ~9 true positives they convert at the boundary and we do
#   not. So the only live question is whether anything sharpens the LOCAL boundary.
#
#   THE PLANNED iter44 (sigmoidF1) IS CANCELLED. Three independent lines killed it:
#     1. PLATT ANNIHILATION (a theorem, not a worry). If a loss change induces any affine logit
#        reparameterization z' = a*z + b, then sigma(A(az+b)+B) = sigma((Aa)z + (Ab+B)) — refitting
#        Platt's two parameters recovers the identical function. sigmoidF1's entire boundary effect,
#        logit-adjusted loss and balanced softmax all lie exactly in Platt's span, and our pipeline
#        refits Platt on the very next line. The arm would have returned a null for a plumbing
#        reason having nothing to do with the loss.
#     2. NO PUBLISHED EVIDENCE any F-surrogate beats BCE at a pre-specified fixed 0.5 with every
#        hyperparameter fixed a priori. sigmoidF1's own fixed-0.5 result is defeated because its
#        eta is a logit offset, so the grid search over eta IS a threshold search.
#     3. MEASURED SCORE DENSITY (local, on our own artifacts). Only 29–38 of 1030 rows lie in
#        [0.45,0.55]. Reaching the F1 optimum needs a threshold-equivalent move of 0.21–0.33;
#        sigmoidF1 blended at w=0.5 supplies ~0.006, which crosses 0.3–0.6 public-slice rows against
#        a +0.006 bar that needs ~1.9 TP.
#
#   ALSO RETRACTED: the "refit Platt on a masked train replica" proposal. Reading the code, that
#   already happens — src/seq_model.py builds OOF through _mask_views(..., oof=True), so held-out
#   rows are ALREADY masked to contiguous 4–6 month windows drawn from the measured test window
#   distribution. The proposal was a no-op.
#
#   WHAT THIS ITERATION DOES INSTEAD. Two things, no third:
#
#   (1) STOP THROWING AWAY THE EVIDENCE.  run_pipeline.py has always written
#       submissions/preds/preds_<name>.npz (oof_prob, y, p_test_raw, test_per_fold). But Cell 5 only
#       downloaded CSVs and submissions/preds/ is gitignored, so EVERY bundle has died with the
#       Colab VM. That one missing copy is why the binormal b, the F1-optimal cut F*/2 and the test
#       positive count P have all been ARGUED from leaderboard arithmetic instead of MEASURED on
#       labelled data. Cell 5b now ships them to Drive. src/seq_model.py additionally records the
#       PER-VIEW OOF predictions (verified inert: --smoke final_oof is bit-identical, 0.86777).
#
#   (2) CORRECT THE ONE CALIBRATION DEFECT THAT IS A FACT OF THE CODE, NOT A HYPOTHESIS.
#       OOF and test scores have different AVERAGING STRUCTURE, and Platt is fit across the mismatch:
#
#                            window views          fold-models
#           OOF row          mean of R=2           1
#           test row         1 (the real window)   mean of n_splits
#
#       Each side is variance-shrunk on the axis the other is not. Under the OLD prevalence pin this
#       was harmless — the cut was re-derived downstream and only the ORDER survived. Under a LITERAL
#       0.5 cut the Platt slope IS the operating point. tools/regime_match.py rebuilds the
#       calibration set at R=1 — one window per row, exactly like a test row — from the per-view
#       record, offline, at ZERO extra training cost.
#
#   ⚠️ PRE-COMMITMENT, MADE BEFORE ANY NUMBER EXISTS. R=1 ships whatever positive rate it produces.
#      Choosing between R=1 and R=2 after seeing their positive rates would be threshold tuning by
#      the back door. That is why regime_match runs INSIDE this script: the choice is made here, in
#      version control, not after the fact. The R=all column it prints is UNDERSTANDING ONLY.
#
#   LEGALITY (three prongs, all argued from the data-generating process):
#     (a) the decision rule stays a literal 0.5 — `R` does not appear in src/calibration.py;
#     (b) the knob is fixed by a TRAIN-ONLY criterion stated a priori — "the calibration set's
#         averaging structure must match deployment's" — never against a realized pos-rate or the LB;
#     (c) it corrects a demonstrably mis-specified model (the table above is code, not conjecture).
#
#   NOT FIXED, deliberately: the MODEL axis (1 fold-model vs n_splits). Closing it needs
#   n_repeats>1, which also changes p_test_raw and the RANKING — a confounded lever. regime_match
#   MEASURES it from test_per_fold and leaves it alone.
#
#   ⚠️ ONE ROUND OF DISTILLATION ONLY, as before. The teacher is the NON-distilled 5-seed permanence
#   pool in every arm. Do NOT re-teach from a distilled student (Kumar/Ma/Liang: error compounds).
# =====================================================================
set -euo pipefail

BASE="--full --model seq"
PERM="--set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"
DP="--set seq.channels.dualpol_gate=true"
SEEDS5="42 7 13 21 29"

# ---- STEP 1: rebuild the 5-seed TEACHER in-run (submissions/preds/ is gitignored). ----
#      Byte-identical to iter41/42/43 so every arm below stays comparable to the banked numbers.
for s in $SEEDS5; do
  python run_pipeline.py $BASE $PERM --set seed=$s --name teacher_perm_s$s
done
python tools/seed_average.py --variant teacher_perm --name teacher_perm5
TEACHER="submissions/preds/preds_teacher_perm5.npz"

DISTILL="--set seq.distill.enable=true --set seq.distill.teacher=$TEACHER"

# ---- STEP 2: rebuild FINALIST #1 (ARM E, alpha-marginalized, 10 DISTINCT seeds). ----
#      Identical flags and seeds to iter43, so this MUST reproduce 0.906104. That makes the rebuild
#      a free reproducibility proof for the Phase-2 code review — and it is what attaches the
#      per-view instrumentation to the artifact we actually care about.
for s in $SEEDS5; do
  python run_pipeline.py $BASE $PERM --set seed=$s $DISTILL --set seq.distill.alpha=0.7 --name amix_s$s
done
for s in 3 17 23 31 37; do
  python run_pipeline.py $BASE $PERM --set seed=$s $DISTILL --set seq.distill.alpha=1.5 --name amix_s$s
done
python tools/seed_average.py --variant amix --name champion_distill_alphamix10

# ---- STEP 3: rebuild ARM G (dual-pol ADDED, 26 ch) — our highest-AUC artifact ever (0.946460). ----
#      Not a new experiment: the gate is closed on composite. It is rebuilt because it is the
#      artifact with the best RANKING we have ever produced, and the regime-matched cut is a
#      boundary intervention — if the correction is real, the base with the best ranking is where
#      it should pay most. This gives the same single intervention two independent readings.
for s in $SEEDS5; do
  python run_pipeline.py $BASE $PERM $DP --set seed=$s \
    $DISTILL --set seq.distill.alpha=0.7 --name dpa_s$s
done
python tools/seed_average.py --variant dpa --name champion_dualpol_add_seedavg5

# ---- STEP 4: THE CONTROL. regime_match --views all must reproduce seed_average BIT-FOR-BIT. ----
#      If it does not, the per-view reconstruction is wrong and the R=1 artifacts below are
#      meaningless. Hard-fail here rather than shipping a submission built on a broken rebuild.
for V in amix dpa; do
  case $V in
    amix) REF="submissions/submission_champion_distill_alphamix10.csv" ;;
    dpa)  REF="submissions/submission_champion_dualpol_add_seedavg5.csv" ;;
  esac
  python tools/regime_match.py --variant $V --name _control_$V --views all
  if cmp -s "$REF" "submissions/submission__control_$V.csv"; then
    echo "CONTROL PASS ($V): regime_match --views all == seed_average, bit for bit."
  else
    echo "CONTROL FAIL ($V): the per-view rebuild does not reproduce seed_average. ABORTING."
    exit 1
  fi
done

# ---- STEP 5: THE ARM. Regime-matched calibration (R=1) on both bases. ----
python tools/regime_match.py --variant amix --name champion_alphamix10_regimematch
python tools/regime_match.py --variant dpa  --name champion_dualpol_add_regimematch

cat <<'NEXT'
=====================================================================
 PASTE BACK, for BOTH regime_match runs:
   - the whole "=== SUMMARY (means across seeds) ===" block (slope / pos-rate / delta_hat / b)
   - the "ROWS CROSSING 0.5 vs the R=all control" line
   - the "MODEL-AXIS (uncorrected)" line
   - the "Isolation OK" line, and the pooled rank correlation
   - both "CONTROL PASS/FAIL" lines from STEP 4
 Also paste the per-metric AUC and F1 columns from Zindi for each upload. Reading our own score
 breakdown is free and it is what closed the alpha knob in iter42 and the VH-VV lane in iter43.
 IGNORE OOF level entirely; it is anti-correlated with the LB. (delta_hat and b are read from the
 OOF *shape*, which is a different claim — see the caveat below.)

 ⚠️ AND: confirm Cell 5b printed "copied to MyDrive/geoai-preds/" and listed the .npz files.
 That copy is the actual point of this iteration. If it failed, say so FIRST — the analysis lane
 stays blocked and everything below is secondary.

 UPLOAD (budget 5/day):
   1. submissions/submission_champion_alphamix10_regimematch.csv     <- PRIMARY (finalist #1 base)
   2. submissions/submission_champion_dualpol_add_regimematch.csv    <- replication (best-AUC base)
 Do NOT upload the rebuilds (champion_distill_alphamix10, champion_dualpol_add_seedavg5) or the
 _control_* files: they re-measure the known 0.906104 / 0.907616 and would waste two slots.

 COMMITTED READ (pre-registered; do not renegotiate after seeing the numbers):

 The comparators are the artifacts' OWN rebuilt scores: 0.906104 (amix) and 0.907616 (dpa).
 This is a PAIRED delta between two artifacts differing in one scalar per member, so the ~0.006
 paired SE applies, not the ~0.012 unpaired one.

   >= +0.006 on the PRIMARY   -> the averaging asymmetry was really misplacing the cut. The
                                 regime-matched artifact becomes finalist #1. Treat dpa as
                                 confirmation, not as a second independent result.
   within +-0.006 on BOTH     -> NULL, and this is the EXPECTED outcome. Say so plainly. With only
                                 ~3% of test mass within 0.05 of the cut, a slope change of this
                                 size cannot move enough rows. That closes the boundary-calibration
                                 lane on a MEASUREMENT rather than an argument, which is exactly
                                 what iterations 42 and 43 could not do for their lanes.
   <= -0.006                  -> the R=2 shrinkage was load-bearing IN OUR FAVOUR. That is a real
                                 finding about our own pipeline, not a failure; record it and keep
                                 R=2. Do NOT then go hunting for the "best" R.
   PRIMARY and dpa DISAGREE   -> the correction is base-dependent, i.e. noise at our resolution.
                                 Read it as a null and stop.

 CAVEAT ON delta_hat, stated before the number arrives. delta_hat = F*_masked/2 is Lipton's
 F1-optimal cut. Our OOF prior is 0.4023 while the deployment pos-rate is ~0.587, and F1 rises with
 prevalence, so the measured delta_hat is a LOWER BOUND on the deployment-optimal cut. Therefore:
 "delta_hat > 0.47, the cut is fine, kill the lane" is a ROBUST verdict; "delta_hat < 0.45, the cut
 is misplaced" is the FRAGILE one and must not be acted on by itself.

 FINALISTS going in: {champion_distill_alphamix10 0.906104, champion_archblend4 0.899643}.
   archblend4 stays as the DECORRELATED hedge: every distill artifact is built on the same teacher,
   so they are not second opinions of each other. Replace #1 only on the >= +0.006 branch above.

 DEADLINE 2026-08-16. iter45 is the code-review package and the final finalist lock — NOT another
 experiment. If this returns a null, that is the signal to stop experimenting entirely.
=====================================================================
NEXT
