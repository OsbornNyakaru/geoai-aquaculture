#!/usr/bin/env bash
# =====================================================================
# REPRODUCE THE SUBMITTED FINALISTS — one command, for reviewers.
#
# WHY THIS FILE EXISTS
#   `run_pipeline.py --model` defaults to `gbdt`, the SUPERSEDED baseline. A reviewer running a
#   bare `python run_pipeline.py --full` would reproduce the ~0.826-0.878 gradient-boosted model,
#   NOT our submission, and would reasonably conclude the solution does not reproduce.
#
#   This is not hypothetical. In the EY Biodiversity Challenge (same platform, same rulebook
#   template), only 2 of the top 10 finishers survived post-challenge code review -- the rest were
#   eliminated for "failing to submit code within the deadline or having reproducibility concerns."
#   Our standing is 65% leaderboard + 35% code review of the top 5: a LARGER weight against a
#   SMALLER field. See gemini_loop/RESEARCH_08_EY.md.
#
# ⚠️ READ THIS IF YOU ARE COMPARING AGAINST config/config.yaml
#   The committed config defaults reproduce the ORIGINAL 24-channel champion. Both designated
#   finalists are BUILT ON TOP of it with explicit `--set` flags, shown below and passed here.
#   In particular `seq.channels.permanence=true` with a SINGLE threshold `cdf_taus=[-21.0]` is
#   what makes the 25-channel permanence model; it is deliberately NOT the config default, so
#   that the historical anchors in experiments/anchors.tsv stay reproducible from the same file.
#   Every flag that differs from the committed default is passed explicitly on the command line
#   in this script -- there are no hidden defaults behind either finalist.
#
# OPERATING POINT: runs `calibration.compliance_mode: legal` -- Platt calibration fit on TRAINING
#   out-of-fold predictions only, then a LITERAL 0.5 cut, genuine probabilities in both columns.
#   The rules state: "Setting a probability threshold is strictly forbidden. Your binary target
#   should be based on the default threshold of 0.5." An earlier revision used a prevalence pin
#   that violated that; it survives ONLY as `compliance_mode: pinned` so historical anchors stay
#   reproducible, and its output must never be submitted. See REPORT.md section 8.
#
# STAGE 1 reproduces FINALIST #1 (15 runs + 2 pools + a calibration rebuild).  STAGE 2 reproduces
#   FINALIST #2 (4 runs + a pool). Pass --quick to stop after stage 1.
# =====================================================================
set -euo pipefail

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

# The permanence champion = committed config + these two flags. Used by every stage-1 run.
PERM="--set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"
# The iteration-43 dual-pol gate 1[VH<-21]*1[(VH-VV)<-8]. Used by the STUDENTS only, not the teacher.
DP="--set seq.channels.dualpol_gate=true"

cat <<'BANNER'
=====================================================================
 Reproducing the SUBMITTED finalists: from-scratch temporal Transformer

   FINALIST #1  champion_dualpolmix10_regimematch   public LB 0.907368983   (stage 1)
                = a 5-seed permanence TEACHER (25 ch), then 10 distilled students on 10 DISTINCT
                  seeds WITH the dual-pol gate (26 ch), alpha marginalized over {0.7, 1.5},
                  legally pooled, then Platt refit on an R=1 regime-matched calibration set
   FINALIST #2  champion_archblend4                 public LB 0.899643      (stage 2)
                = 4 architecture variants, legally pooled

 WHY BOTH FINALISTS ARE POOLED ARTIFACTS, AND WHY NEITHER IS OUR BEST PUBLIC SCORE
   Our measured seed-to-seed leaderboard sd is 0.0191. Four separate single-seed "records"
   (0.906492, 0.913263, 0.912759, 0.914179) ALL collapsed to ~0.8995-0.906 when averaged over 5
   seeds. We therefore designate pooled, low-variance artifacts for the 721-row private slice
   rather than the best number we ever saw on the 309-row public slice. See REPORT.md section 4.

   Finalist #1 is NOT our best public composite. Two artifacts beat it there and both were declined
   on the SAME pre-registered ground:
     champion_distill_a15_seedavg5    0.910837   5 seeds, one alpha
     champion_dualpol_add_regimematch 0.910447   5 seeds, this same recipe
   Iteration 42 measured what a 5-vs-10-seed public gap actually is: at alpha=0.7 the 5-seed and
   10-seed pools returned BIT-IDENTICAL AUC (0.944024425 -- the same concordant-pair count), so the
   entire composite difference was ONE row crossing the 0.5 cut. A 5-seed public edge of that size
   is cut luck on 309 rows; the 10-distinct-seed variance reduction applies to all 721 private rows.
   Choosing the lower public number here is deliberate and pre-registered. See REPORT.md section 4.

   What finalist #1 DOES win on is ranking. Its AUC is 0.945841814, above the public leader's
   0.944897, and +0.003271 over the otherwise-identical 25-channel build -- the dual-pol gate buys
   pure ranking and moves the cut not at all (identical F1 cell, 328/372). AUC is 40% of the metric
   and it applies to every private row without a coin-flip at the threshold.

 STAGE 1 fingerprints to verify in the log below:
   seq relative_time ON: observed window left-aligned to t_rel=0
   seq cross-view invariance ON: lambda=1
   seq input width: 25 channels/month        <- the TEACHER: 24 + the permanence indicator
   seq input width: 26 channels/month        <- the STUDENTS: + the dual-pol gate. If a student
                                                logs 25, the gate did not attach and the
                                                reproduction is INVALID, not merely different.
   t_star        0.5000        <- LITERAL. Not fitted. This is the compliance claim.
   TRANSDUCTIVE GATE ... submitted pos-rate  <- must land inside [0.50, 0.62] for every student
   POOLED: 5 seeds  (the teacher)  then  POOLED: 10 seeds  (the pre-calibration pool)
   CONTROL PASS      <- the R=1 rebuild must reproduce seed_average bit-for-bit at --views all
   final pooled test pos-rate ~0.583

 STAGE 2 fingerprints:
   seq input width: 24 channels/month        <- archblend4 predates the permanence channel
   mean pairwise rho = 0.9524
   POOLED: 4 members | test pos-rate 0.5670

 TWO HONEST NOTES FOR THE REVIEWER
 1. final_oof is deliberately NOT a proxy for leaderboard performance here -- our best-LB models
    have our LOWEST OOF, and local OOF (~0.975) has been ANTI-correlated with the LB (~0.90).
    That is the designed covariate shift doing what it was designed to do. See README.
 2. Two DIFFERENT numbers are easy to conflate here, so both are stated:
      SEED-to-seed spread is 0.0191. That is the spread across different CHOICES of seed. It does
        not apply to this script, which pins the same seeds we submitted.
      RUN-to-run variation, same seeds, is what a reproduction actually faces. We do not set
        torch deterministic algorithms for the GPU seq path, so we cannot GUARANTEE bit-exactness.
        But we have now measured it: iteration 45 happened to re-run ten iteration-44 configs in a
        different Colab session on a different GPU allocation, and ALL TEN reproduced to every
        logged decimal (teachers 0.97347/0.97437/0.97143/0.97161/0.97414, students
        0.97594/0.97773/0.97403/0.97350/0.97317).
    So: expect this script to match closely, not loosely. A reproduction drifting by ~0.02 is NOT
    "within seed noise" -- it means something differs (channel width, teacher, alpha, seed set) and
    is worth chasing. Check the width fingerprints above first. See REPORT.md sections 4 and 8.3.
=====================================================================
BANNER

# Confirm the committed config really is the champion BASE before spending the compute.
python - <<'PY'
import sys
# Use the pipeline's own loader rather than re-reading the file here. It sets
# encoding="utf-8" explicitly (src/utils.py); a bare open() picks up the platform default,
# which is cp1252 on Windows and crashes on the non-ASCII in config.yaml.
from src.utils import load_config
cfg = load_config()
s, c = cfg["seq"], cfg["calibration"]
want = {
    "seq.K":                        (s["K"], 2),
    "seq.relative_time":            (s["relative_time"], True),
    "seq.pos_encoding":             (s["pos_encoding"], "learned"),
    "seq.consistency_lambda":       (float(s["consistency_lambda"]), 1.0),
    "seq.pooling":                  (s.get("pooling", "mean"), "mean"),
    "seq.dropout":                  (float(s["dropout"]), 0.2),
    "seq.channels.drop_bands":      (list(s.get("channels", {}).get("drop_bands") or []), []),
    # The transductive terms (iteration 41) must be OFF **in the committed config**, so that the
    # file on disk reproduces the original 24-channel champion and the historical anchors stay
    # valid. Finalist #1 DOES use soft self-distillation, but it is switched on explicitly by
    # `--set seq.distill.enable=true` in stage 1 below -- never by a default. That is the point of
    # asserting False here: every flag behind a finalist is visible on a command line in this file.
    "seq.transduct.enable":         (bool((s.get("transduct") or {}).get("enable", False)), False),
    "seq.distill.enable":           (bool((s.get("distill") or {}).get("enable", False)), False),
    # The iteration-43 dual-pol gate must be OFF **in the committed config**. Finalist #1's students
    # DO use it, switched on explicitly by `--set seq.channels.dualpol_gate=true` in stage 1b -- and
    # the teacher in stage 1a deliberately does NOT, which is why it is a flag and not a default.
    "seq.channels.dualpol_gate":    (bool(s["channels"].get("dualpol_gate", False)), False),
    # THE COMPLIANCE ASSERTION. If this is not 'legal', the run produces a rules-violating
    # submission and the script must refuse rather than silently emit one.
    "calibration.compliance_mode":  (c.get("compliance_mode", "legal"), "legal"),
}
bad = {k: v for k, (v, exp) in want.items() if v != exp}
for k, (v, exp) in want.items():
    print(f"  {'OK ' if v == exp else 'BAD'} {k:32s} = {v!r:10} (expected {exp!r})")
if bad:
    sys.exit("\nCommitted config is NOT the champion base; refusing to run. Offending keys: "
             + ", ".join(bad))
print("\nCommitted config verified as the champion base, in RULES-COMPLIANT mode.\n")
PY

# Reusable compliance audit: readable straight off a submission CSV, trusting none of our code.
audit () {
  python - "$1" "$2" <<'PY'
import pandas as pd, sys
path, tag = sys.argv[1], sys.argv[2]
df = pd.read_csv(path)
ok = (df["TargetF1"].astype(int) == (df["TargetRAUC"] >= 0.5).astype(int)).all()
print(f"\n  COMPLIANCE AUDIT ({tag})")
print(f"    file                                         : {path}")
print(f"    TargetF1 == (TargetRAUC >= 0.5) on every row : {ok}")
print(f"    distinct TargetRAUC values                   : {df['TargetRAUC'].nunique()} / {len(df)}")
print(f"    realized positive rate                       : {df['TargetF1'].mean():.4f}")
if not ok:
    sys.exit("  FAILED: the binary column is not the literal 0.5 cut of the probability column.")
PY
}

# ---- STAGE 1: FINALIST #1 -- champion_dualpolmix10_regimematch. ----
# Three sub-stages, because the finalist is a distilled student, the teacher is not shipped, and the
# shipped artifact is the regime-matched recalibration of the pool rather than the pool itself:
#   1a. the 5-seed permanence TEACHER, 25 ch  (this is champion_perm_seedavg5, public LB 0.899882)
#   1b. 10 students on 10 DISTINCT seeds WITH the dual-pol gate, 26 ch, alpha over {0.7, 1.5}, pooled
#   1c. the R=1 regime-matched Platt refit -- with its own bit-for-bit control
#
# ⚠️ ONE ROUND OF DISTILLATION ONLY. The teacher is the NON-distilled pool from 1a in every student.
# Re-teaching from a distilled student compounds error (Kumar/Ma/Liang) and is NOT what we shipped.
#
# The teacher is rebuilt here rather than shipped because submissions/preds/ is gitignored -- it
# holds arrays derived from the competition CSVs, which we do not redistribute. Rebuilding it is
# deterministic given the seeds below.
echo "=== STAGE 1a/3: the 5-seed permanence TEACHER -- 5 runs + pool (25 channels) ==="
for SD in 42 7 13 21 29; do
  python run_pipeline.py --full --model seq $PERM --set seed=$SD --name "perm_single_s${SD}"
done
python tools/seed_average.py --variant perm_single --name champion_perm_seedavg5
audit "submissions/submission_champion_perm_seedavg5.csv" "the teacher (LB 0.899882)"

TEACHER="submissions/preds/preds_champion_perm_seedavg5.npz"
DISTILL="--set seq.distill.enable=true --set seq.distill.teacher=$TEACHER"

echo
echo "=== STAGE 1b/3: the 10-seed dual-pol pool -- 10 runs + pool (26 channels each) ==="
# alpha is marginalized, not tuned: iteration 42 swept it over a 5x range and the entire ladder
# moved ONE true positive out of 309 public rows. When a knob is that flat you average over it
# rather than picking a value, which removes the choice from the private-slice gamble at zero cost.
for SD in 42 7 13 21 29; do
  python run_pipeline.py --full --model seq $PERM $DP --set seed=$SD \
    $DISTILL --set seq.distill.alpha=0.7 --name "dpam_s${SD}"
done
for SD in 3 17 23 31 37; do
  python run_pipeline.py --full --model seq $PERM $DP --set seed=$SD \
    $DISTILL --set seq.distill.alpha=1.5 --name "dpam_s${SD}"
done
python tools/seed_average.py --variant dpam --name champion_dualpolmix10

echo
echo "=== STAGE 1c/3: the R=1 regime-matched Platt refit (the SHIPPED artifact) ==="
# WHY THIS STEP EXISTS. Platt is fit on out-of-fold predictions, but OOF rows and test rows do not
# have the same averaging structure: an OOF row is the mean of R=2 masked window views from ONE
# fold-model, while a test row is ONE real window averaged over n_splits fold-models. Under a
# LITERAL 0.5 cut the Platt slope IS the operating point, so fitting it across that mismatch is a
# genuine defect. tools/regime_match.py rebuilds the calibration set at R=1 and refits offline, at
# zero extra training cost. R is read solely on the held-out path, which makes it the one operating-
# point lever in the pipeline that cannot see a realized test positive rate or any leaderboard
# feedback -- it was pre-committed in version control before the score existed. See REPORT.md 8.
#
# THE CONTROL COMES FIRST. At --views all the rebuild must reproduce tools/seed_average.py
# BIT-FOR-BIT; if it does not, the rebuild arithmetic is wrong and the R=1 output is meaningless.
# (Getting this right required replicating the native-float32 reduction exactly -- the obvious
# float64 np.add.at rewrite agrees only to ~1e-9, which is invisible in OOF metrics but survives
# Platt into the reported TargetRAUC decimals.)
python tools/regime_match.py --variant dpam --name _control_dpam --views all
if cmp -s "submissions/submission_champion_dualpolmix10.csv" \
          "submissions/submission__control_dpam.csv"; then
  echo "  CONTROL PASS: regime_match --views all == seed_average, bit for bit."
else
  echo "  CONTROL FAIL: the per-view rebuild does not reproduce seed_average. ABORTING."
  exit 1
fi
python tools/regime_match.py --variant dpam --name champion_dualpolmix10_regimematch
audit "submissions/submission_champion_dualpolmix10_regimematch.csv" "finalist #1"

if [ "$QUICK" = "1" ]; then
  echo; echo "=== --quick: stopping before stage 2. ==="; exit 0
fi

# ---- STAGE 2: FINALIST #2 -- champion_archblend4 (4 architecture variants, legally pooled). ----
# NOTE these members predate the permanence channel and are reproduced WITHOUT $PERM, at the
# 24-channel width they were actually submitted at.
echo
echo "=== STAGE 2: champion_archblend4 (finalist #2) -- 4 runs + pool (24 channels) ==="
python run_pipeline.py --full --model seq --name seq_a_reltime --set seq.consistency_lambda=0
python run_pipeline.py --full --model seq --name seq_a_nope --set seq.consistency_lambda=0 --set seq.pos_encoding=none
python run_pipeline.py --full --model seq --name seq_a_l3   --set seq.consistency_lambda=3
python run_pipeline.py --full --model seq --name seq_a_xview

python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --name champion_archblend4
audit "submissions/submission_champion_archblend4.csv" "finalist #2"

echo
echo "=== Done. ==="
echo "  finalist #1  submissions/submission_champion_dualpolmix10_regimematch.csv  (LB 0.907368983)"
echo "  finalist #2  submissions/submission_champion_archblend4.csv                (LB 0.899643)"
echo
echo "  Stage 1 also emits three files that are NOT designated finalists:"
echo "    submission_champion_perm_seedavg5.csv     LB 0.899882  -- the TEACHER (1a)"
echo "    submission_champion_dualpolmix10.csv      unscored     -- the pool BEFORE the R=1 refit"
echo "    submission__control_dpam.csv              unscored     -- the bit-for-bit control (1c)"
echo "  The last two must be byte-identical to each other; that is what CONTROL PASS asserts."
echo "  Compare the printed fingerprints against the banner above."
