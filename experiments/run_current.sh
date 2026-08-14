#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 49 — JTT (Just Train Twice): the LAST live candidate for the F1 gap.
#
#   WHY THIS IS THE ONLY THING LEFT, AND WHY THAT IS A MEASURED CLAIM, NOT A MOOD.
#   Round 23 closed every OPERATOR-level lane on arithmetic:
#     - the THRESHOLD is worth +0.0004 F1. F1 is at a MAXIMUM at t*, so the penalty for sitting at
#       0.5 is second order, and its coefficient is the near-cut score density -- which we measured
#       at ~15 rows of 1030. Confirmed two independent ways. ~60x below the noise floor.
#       Prevalence-matching to the estimated 0.618 would LOSE 0.0021 F1.
#     - the CALIBRATOR FAMILY reversed direction (iter46 part 1: beta 15 down/0 up).
#     - both POOLING OPERATORS moved 4-13 rows of 1030 (iter46 part 2; iter48 majority vote,
#       +0.00004/+0.00080, and dead even under the illegal post-hoc-k version of itself).
#     - EVERY POINTWISE LOSS is order-invariant at the population optimum. Focal, ASL, LDAM,
#       PolyLoss and label smoothing all minimize to T(eta(x)) for ONE fixed monotone T, so ROC-AUC
#       is EXACTLY unchanged and the F1 effect is a pure threshold slide. Focal's warp is proven
#       strictly order-preserving: Charoenphakdee et al., CVPR 2021, arXiv:2011.09172, Thm 3/5/11 +
#       Lemma 14. ⚠️ BOTH external round-23 reports ranked Asymmetric Loss as their #1
#       recommendation. That recommendation is refuted by this theorem and we are NOT running it.
#
#   WHY JTT ESCAPES THE THEOREM. Liu et al., ICML 2021, arXiv:2107.09044. The theorem assumes ONE
#   fixed pair (l1, l0) shared by every x. JTT's weight depends on whether the stage-1 model erred
#   on row i, so it is x-dependent AND class-asymmetric. The pointwise objective becomes
#   eta*w1(x)*l1 + (1-eta)*w0(x)*l0, minimized at T(eta_eff) with
#   eta_eff = eta*w1 / (eta*w1 + (1-eta)*w0). Since w1/w0 varies with x, eta_eff is NOT monotone in
#   eta alone. JTT genuinely REORDERS -- the only cheap candidate round 23 found that does.
#
#   ⚠️⚠️ THE REORDERING IS PROVABLE. ITS SIGN IS NOT. Stated before the numbers, as always.
#   Three specific reasons this may well LOSE, all measured before the run:
#     1. |E| = 38 rows of 1817 (2.09%). `balance` therefore sets lambda_up = 46.8, putting HALF the
#        gradient mass on 38 rows in a 71k-parameter model. That is a memorization setup, and
#        "added capacity fitted to these shifted rows hurts" is this project's most reliable law.
#     2. OOF UNDER-REPRESENTS THE DEPLOYMENT FAILURE RATE BY ~7x. We see a 2.09% error rate on
#        held-out train; the leaderboard says deployment recall is 0.859, i.e. ~14% of positives
#        missed. JTT can only upweight failures we can SEE, and most of ours are created by the
#        covariate shift, which OOF is blind to. So it addresses a fraction of the problem at best.
#     3. Those 38 rows may simply be mislabeled or intrinsically ambiguous, in which case
#        upweighting them 47x is fitting noise very hard.
#   The one thing genuinely in its favour: our OOF is computed on MASKED 4-6 month held-out views
#   (_mask_views(..., oof=True)), so the 38 errors are rows that fail UNDER TEST-LIKE TRUNCATION.
#
#   THE ERROR SET, measured before the run (free, train-only):
#     |E| = 38 of 1817 (2.09%) = 24 false NEGATIVES + 14 false positives.
#     The 24 missed positives are CONFIDENTLY missed -- median OOF 0.170, ten of them below 0.10,
#     and ZERO in [0.45, 0.50). That is direct confirmation of round 23's central finding: the rows
#     we need are nowhere near the boundary, which is exactly why no operator-level fix can reach
#     them, and exactly what JTT is designed to attack.
#
#   THREE RUNS, ONE VARIABLE. The control is run LOCALLY ALONGSIDE the arms rather than compared to
#   a banked Colab-era bundle, so the comparison is exact and shares every seed, fold split and
#   teacher. lambda_up is the ONLY thing that differs between ARM A and ARM B, and both values were
#   fixed from a TRAIN-ONLY quantity (|E|) before any score existed -- never against F1, a realized
#   positive rate, or the leaderboard.
# =====================================================================
set -euo pipefail

# ⚠️ FIXED 2026-08-14, AFTER THIS SCRIPT FAILED ON COLAB. The first version pointed straight at
# `submissions/preds/preds_teacher_perm5.npz` and `preds_champion_distill_alphamix10.npz`. Those
# live in `submissions/preds/`, which is **GITIGNORED** — it holds arrays derived from the
# competition CSVs, which we do not redistribute. They existed on the machine the arm was developed
# on and do not exist on a fresh clone, so Colab died four minutes in with a numpy FileNotFoundError
# from inside `_load_teacher`. `config.yaml` states this constraint verbatim ("Must be regenerated
# in the same run: submissions/preds/ is gitignored") and this script ignored it.
#
# Two fixes. (1) STEP 0 below REGENERATES both dependencies, so the file is self-contained on a
# bare clone — which is also what the code review requires. (2) The preflight fails in SECONDS with
# a readable message instead of minutes in with a stack trace.
#
# ⚠️ AND A HONEST CAVEAT ABOUT RE-RUNNING THIS. The committed read below was pre-registered against
# a SPECIFIC stage-1 bundle for which |E| = 38. A rebuild on different hardware will not necessarily
# reproduce that error set exactly. **iter49 has already been run and measured** (see LB_LOG iter49);
# a rebuild is a REPRODUCTION, not the measurement. If a rebuild yields |E| != 38, that is expected
# and is NOT a void of the recorded result — it voids only the rebuild's claim to be the same arm.

PERM="--set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"
DP="--set seq.channels.dualpol_gate=true"
# TEACHER RESOLUTION. `preds_teacher_perm5.npz` is the exact bundle the RECORDED iter49 measurement
# used; `preds_champion_perm_seedavg5.npz` is the canonical name STAGE 1a of reproduce_champion.sh
# writes. They are the same recipe under two names. Prefer the recorded one when it exists so a
# re-run on the development machine reproduces the measurement rather than silently rebuilding a
# near-identical but not bit-identical teacher.
TEACHER="submissions/preds/preds_teacher_perm5.npz"
[ -f "$TEACHER" ] || TEACHER="submissions/preds/preds_champion_perm_seedavg5.npz"
DISTILL="--set seq.distill.enable=true --set seq.distill.teacher=$TEACHER --set seq.distill.alpha=0.7"
STAGE1="submissions/preds/preds_champion_distill_alphamix10.npz"
echo "teacher resolved to: $TEACHER"

# ---- STEP 0: regenerate the two gitignored bundles this arm depends on. ----
#      Skipped individually if already present, so re-running the file locally costs nothing.
if [ ! -f "$TEACHER" ]; then
  echo "=== STEP 0a: rebuilding the 5-seed permanence TEACHER (5 runs, 25 channels) ==="
  for SD in 42 7 13 21 29; do
    python run_pipeline.py --full --model seq $PERM --set seed=$SD --name "perm_single_s${SD}"
  done
  python tools/seed_average.py --variant perm_single --name champion_perm_seedavg5
else
  echo "STEP 0a: teacher already present, skipping rebuild."
fi

if [ ! -f "$STAGE1" ]; then
  echo "=== STEP 0b: rebuilding the JTT stage-1 source (10 runs, 26 channels) ==="
  for SD in 42 7 13 21 29; do
    python run_pipeline.py --full --model seq $PERM $DP --set seed=$SD \
      $DISTILL --set seq.distill.alpha=0.7 --name "amix_s${SD}"
  done
  for SD in 3 17 23 31 37; do
    python run_pipeline.py --full --model seq $PERM $DP --set seed=$SD \
      $DISTILL --set seq.distill.alpha=1.5 --name "amix_s${SD}"
  done
  python tools/seed_average.py --variant amix --name champion_distill_alphamix10
else
  echo "STEP 0b: JTT stage-1 bundle already present, skipping rebuild."
fi

# ---- PREFLIGHT: fail in seconds, not minutes, and say exactly what is missing. ----
for f in "$TEACHER" "$STAGE1"; do
  [ -f "$f" ] || { echo "PREFLIGHT FAIL: $f still missing after STEP 0. Not starting the arms."; exit 1; }
done
echo "PREFLIGHT OK: both gitignored dependencies present."

# ---- CONTROL: the champion single-member recipe at seed 42, no JTT. ----
python run_pipeline.py --full --model seq $PERM $DP $DISTILL --set seed=42 \
  --name jtt_control_s42

# ---- ARM A: JTT with the PRE-COMMITTED parameter-free `balance` rule (lambda_up = 46.8). ----
python run_pipeline.py --full --model seq $PERM $DP $DISTILL --set seed=42 \
  --set seq.jtt.enable=true --set seq.jtt.source=$STAGE1 --set seq.jtt.lambda_up=balance \
  --name jtt_balance_s42

# ---- ARM B: JTT at lambda_up = 5, the conservative end of Liu et al.'s own published range. ----
#      Included because |E| is small enough that `balance` is aggressive; this is a pre-registered
#      SENSITIVITY pair, not a knob to be selected after the fact.
python run_pipeline.py --full --model seq $PERM $DP $DISTILL --set seed=42 \
  --set seq.jtt.enable=true --set seq.jtt.source=$STAGE1 --set seq.jtt.lambda_up=5 \
  --name jtt_lam5_s42

cat <<'NEXT'
=====================================================================
 PASTE BACK:
   - the "JTT stage-1 error set" line and the "JTT lambda_up" line from BOTH arms.
     ARM A must read lambda_up = 46.816; ARM B must read 5.000. If ARM A's |E| is not 38, the
     stage-1 bundle is not the one this read was pre-registered against and the arm is VOID.
   - the "seq input width" line from all three. It MUST be 26 channels/month on all three -- if
     JTT changed the width, something other than the loss weighting changed and the arm is void.
   - "oof_f1@0.5", "oof_auc", "oof_combined" for all three.
   - the LEGAL calibration line (Platt slope + realized test pos-rate) for all three.
   - the Zindi AUC and F1 columns for whatever gets uploaded, never just the composite.

 UPLOAD: NOTHING AUTOMATICALLY. Read the offline block below first; it may spend zero slots.

 COMMITTED READ (pre-registered; do NOT renegotiate after seeing the numbers):

 STEP 1 -- THE VOID CHECKS, before any comparison.
   - width 26 on all three, |E| = 38, lambda_up 46.816 / 5.000. Any mismatch => VOID, not negative.
   - CONTROL's OOF must land near the champion's usual ~0.975-0.980 combined. If the control is
     itself anomalous, the whole iteration is void and nothing is uploaded.

 STEP 2 -- THE OFFLINE READ, which may close this for ZERO submissions.
   Compute, against the CONTROL, on the 1030 test rows:
     (a) Spearman rank correlation. If rho > 0.999 for an arm, JTT did NOT meaningfully reorder --
         it collapsed to a threshold slide, the mechanism did not fire, and that arm is CLOSED with
         no submission. This is the single most likely outcome and it is a real result.
     (b) the number of rows changing side at the literal 0.5 cut.
     (c) OOF recall at 0.5 on the 24 known false negatives. JTT's ENTIRE thesis is that it recovers
         them. ⚠️ This is IN-SAMPLE for the arms (those rows were upweighted during training), so a
         high number here is NOT evidence of transfer -- it is only a check that the mechanism did
         what it claims mechanically. If it did NOT recover them even in-sample, the arm is dead.

 STEP 3 -- SPEND A SLOT ONLY IF STEP 2 SHOWS A REAL REORDERING.
   Upload at most ONE arm. Prefer ARM B (lambda 5) if both reordered, on the standing rule that the
   less aggressive intervention is the safer unmeasured artifact.

   Reading the LB result of the uploaded arm against the CONTROL's own LB (upload the control too
   ONLY if a slot is spare; otherwise compare to champion 0.907368983 and say so):
     >= +0.006     JTT works. This would be the first thing to move the F1 term in 8 iterations.
                   Do NOT designate off one seed -- expand to five and re-read, exactly as the
                   alpha ladder taught us at iter42.
     within ±0.006 inconclusive at our noise floor. Report as such; do NOT call it a win. The lane
                   closes and the report records that the last live F1 candidate was measured.
     <= -0.006     the capacity law holds again, now demonstrated on an x-dependent reweighting.
                   This is a GOOD outcome for the writeup: it would mean the confidently-missed
                   positives are not recoverable from the source distribution at all, which is the
                   strongest possible form of round 23's conclusion.

 ⚠️ FINALISTS ARE NOT AT RISK. champion_dualpolmix10_regimematch (0.907368983) and
 champion_archblend4 (0.899643) stay designated. A single-seed arm cannot replace a 10-seed
 measured artifact under the standing rule, whatever it scores.

 DEADLINE 2026-08-16.
=====================================================================
NEXT
