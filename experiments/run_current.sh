#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 43 — THE LAST FEATURE SHOT + THE VARIANCE-OPTIMAL FINALIST.  *** 3 uploads ***
#
#   iter42 RESULT: THE alpha KNOB IS CLOSED. The ladder around the banked alpha=0.7 (0.909868):
#       alpha=0.3, 5 seeds   0.907370   (AUC 0.942280  F1 0.884097)   -0.002498
#       alpha=1.5, 5 seeds   0.910837   (AUC 0.942861  F1 0.889488)   +0.000969
#       alpha=0.7, 10 seeds  0.906642   (AUC 0.944024  F1 0.881720)   -0.003226
#   Nothing clears the +0.006 bar over a 5x range of alpha. Total spread 0.0035. Zindi's F1 column
#   is a small-denominator rational, so we can read the ladder EXACTLY: 55/62, 330/371 and 82/93 are
#   TP=165, TP=165, TP=164 at predicted-positive counts differing by one. THE ENTIRE LADDER IS ONE
#   TRUE POSITIVE AND ONE PREDICTED POSITIVE OUT OF 309 PUBLIC ROWS. Stop tuning alpha. Permanently.
#
#   THE DECISIVE DETAIL: alpha=0.7 at 5 seeds and at 10 seeds have BIT-IDENTICAL AUC (0.944024425).
#   Identical concordant-pair count => adding 5 seeds did not degrade the RANKING at all. The whole
#   -0.0032 is one row crossing the 0.5 cut -- an operating-point coin flip that re-flips independently
#   on the 721 private rows. So the 5-seed public edge carries ZERO ranking information, and the
#   10-seed artifact is strictly the better bet on private. See ARM E.
#
#   PREDICTION SCORECARD (logged before the scores arrived, and it FAILED): from the iter42 run log
#   the inter-seed rank correlation rose monotonically with alpha (0.9666 / 0.9770 / 0.9863), so I
#   predicted alpha=1.5 would lose pooling gain and underperform. It came top instead -- by one row.
#   The homogenization mechanism is not refuted, it is simply unmeasurable at this resolution. Logged
#   as a miss; no rescue attempted.
#
#   WHAT THIS ITERATION DOES. Two things, no third:
#     E. Lock in the variance-optimal finalist (alpha-marginalized, 10 DISTINCT seeds). Near-free.
#     F/G. Spend the one remaining feature idea: the DUAL-POLARIZATION gate. Replacement and added.
#
#   ⚠️ ONE ROUND OF DISTILLATION ONLY, as before. The teacher is the NON-distilled 5-seed permanence
#   pool in every arm. Do NOT re-teach from a distilled student (Kumar/Ma/Liang: error compounds).
# =====================================================================
set -euo pipefail

BASE="--full --model seq"
PERM="--set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"
SEEDS5="42 7 13 21 29"

# ---- STEP 1: rebuild the 5-seed TEACHER in-run (submissions/preds/ is gitignored). ----
#      Identical to iter41/42 so every arm below stays comparable to the banked 0.909868.
for s in $SEEDS5; do
  python run_pipeline.py $BASE $PERM --set seed=$s --name teacher_perm_s$s
done
python tools/seed_average.py --variant teacher_perm --name teacher_perm5
TEACHER="submissions/preds/preds_teacher_perm5.npz"

DISTILL="--set seq.distill.enable=true --set seq.distill.teacher=$TEACHER"

# ---- ARM E: the ALPHA-MARGINALIZED finalist. 10 members, 10 DISTINCT seeds. ----
#      iter42 proved alpha is a nuisance parameter (spread = one row). When a knob is flat you do not
#      pick a setting, you AVERAGE OVER IT -- that removes the alpha choice from the private-slice
#      gamble at zero cost. Crucially this uses TEN DISTINCT SEEDS, so it dominates BOTH iter42
#      candidates: alpha=0.7 x 10 seeds marginalized only seed; alpha=1.5 x 5 seeds only had 5 seeds.
#      Seed variance (sd 0.019) is the dominant error term, so distinct seeds are what matter most.
#      All 10 pool under one `amix` prefix so seed_average globs them together.
for s in $SEEDS5; do
  python run_pipeline.py $BASE $PERM --set seed=$s $DISTILL --set seq.distill.alpha=0.7 --name amix_s$s
done
for s in 3 17 23 31 37; do
  python run_pipeline.py $BASE $PERM --set seed=$s $DISTILL --set seq.distill.alpha=1.5 --name amix_s$s
done
python tools/seed_average.py --variant amix --name champion_distill_alphamix10

# ---- ARM F: DUAL-POL GATE as a WIDTH-NEUTRAL REPLACEMENT (25 channels, same as champion). ----
#      1[VH<-21] . 1[(VH-VV)<-8] replaces 1[VH<-21]. The gate CONTAINS the permanence clause, so this
#      is a pure refinement at identical width -- no capacity change, the only family that ever wins here.
#
#      WHY NOW, AFTER cross_pol SCORED -0.0228. That arm added RAW VH-VV, which is affine-spanned by
#      bands the model already has (our SDWI is exactly -5.697415 + 0.230259*(VH+VV), verified to
#      3.6e-15). It measured WIDTH COST, not information. The INDICATOR form -- the same nonlinearity
#      behind our only feature win -- had never been run until now.
#
#      TRAIN-ONLY SCREEN run this round (window-matched through _mask_views, so the 4-6 month test
#      window is NOT confounded with the domain). Univariate AUC of the mean-pooled fraction:
#          VH-only  1[VH<-21]       0.8012      <- the current champion channel
#          ratio    1[(VH-VV)<-8]   0.7556
#          AND gate                 0.8487      <- +0.0475, and NEITHER clause alone comes close
#      The AND is therefore a genuine interaction, not a threshold reshuffle. Best VH-only threshold
#      anywhere in [-26,-12] is 0.7917, so this is not reachable by retuning tau.
#
#      THE HONEST RISK, stated before the run: the gate's marginal shifts HARDER than permanence
#      (window-matched train->test mean -0.122 vs +0.061; per-channel adversarial AUC 0.597 vs 0.550).
#      Under A8 (shift is additive across many weak marginals) that is a cost. Under A3 (adv-AUC is
#      RETIRED as a selection criterion -- it correlated POSITIVELY with realized transfer, +0.68 across
#      transforms and +1.00 across modalities) it is not evidence against. The two read opposite, which
#      is exactly why this needs the leaderboard and not another offline argument. Note also that the
#      AND SUPPRESSES the ratio's shift (adv-AUC 0.685 ratio-only -> 0.597 gated): the VH clause anchors it.
DP="--set seq.channels.dualpol_gate=true"
for s in $SEEDS5; do
  python run_pipeline.py $BASE $DP --set seq.channels.permanence=false --set seed=$s \
    $DISTILL --set seq.distill.alpha=0.7 --name dpr_s$s
done
python tools/seed_average.py --variant dpr --name champion_dualpol_rep_seedavg5

# ---- ARM G: DUAL-POL GATE **ADDED** on top of permanence (26 channels). ----
#      If the gate is genuinely informative, keeping both coordinates may beat either alone -- the two
#      channels agree only at Spearman 0.75, so they are not redundant. This costs one channel of width,
#      and width cost is precisely what killed every previous VH-VV arm, so F vs G separates
#      "the information is new" from "the width is affordable". Distillation stabilizes training, which
#      is why a 26-channel arm is worth one upload now when it would not have been before iter41.
for s in $SEEDS5; do
  python run_pipeline.py $BASE $PERM $DP --set seed=$s \
    $DISTILL --set seq.distill.alpha=0.7 --name dpa_s$s
done
python tools/seed_average.py --variant dpa --name champion_dualpol_add_seedavg5

cat <<'NEXT'
=====================================================================
 PASTE BACK the run summary lines AND, for every arm, BOTH gate lines:
     "TRANSDUCTIVE GATE PASS/FAILED: submitted pos-rate ..."
     "TRANSDUCTIVE GATE: teacher pos-rate ... | student-vs-teacher Spearman ..."
 Also paste the "seq input width: NN channels/month" line for arms F and G -- F MUST log 25 and
 G MUST log 26. If F logs 26 the replacement silently became an addition and the arm is void.
 Also paste the per-metric AUC and F1 columns from Zindi for each upload. Reading our own score
 breakdown is free and it is what closed the alpha knob this round.
 IGNORE OOF entirely; it is anti-correlated with the LB.

 If an arm logs "TRANSDUCTIVE GATE FAILED" (pos-rate outside [0.50, 0.62]), DO NOT UPLOAD IT --
 that is the constant-predictor / runaway failure mode that sank ARM T in iter41.

 UPLOAD (budget 5/day):
   1. submissions/submission_champion_distill_alphamix10.csv       <- the finalist candidate
   2. submissions/submission_champion_dualpol_rep_seedavg5.csv     <- gate REPLACES permanence (25 ch)
   3. submissions/submission_champion_dualpol_add_seedavg5.csv     <- gate ADDED to permanence (26 ch)
 Do NOT upload teacher_perm5 (it re-measures the known 0.899882) or any single seed.

 COMMITTED READ (pre-registered; do not renegotiate after seeing the numbers):

 ARM E (alphamix10). This is a VARIANCE decision, not a level decision, so it is NOT gated on
   beating 0.910837. Expect it to land anywhere in 0.905-0.912; that whole range is one or two rows.
   It becomes FINALIST #1 unless it comes in below ~0.903, which would mean something is actually
   wrong rather than noisy. Rationale is settled by the iter42 AUC identity above: the public gap
   between 5- and 10-seed pools was pure cut-crossing, zero ranking content, while the variance
   reduction is real and applies to all 721 private rows.

 ARMS F/G (dual-pol gate). Judged against the best comparable 5-seed distilled artifact, 0.910837.
   >= +0.006 on either  -> the gate is REAL. It is the first new information admitted to the feature
                           bank since permanence, and iter44 tunes tau_r (which we have never swept
                           on the leaderboard) and rebuilds the finalist on top of it.
   both within +-0.006   -> the VH-VV lane is CLOSED for good. Three independent forms of the same
                           quantity (raw, affine/SDWI, indicator) will have failed to move it, and the
                           conclusion is that VH-VV carries no transferable information here beyond
                           what VH already gives. That is a publishable negative, not a null.
   F >> G                -> width cost binds even at 26 channels with distillation; note it and stop.
   G >> F                -> the gate is ADDITIONAL information, not a refinement; permanence and the
                           gate are separate coordinates (consistent with their Spearman of only 0.75).
   either << -0.006      -> the harder marginal shift dominates, A8 beats A3 on this channel, and the
                           adv-AUC retirement gets a caveat: it does not extend to channels whose own
                           marginal moves this much.

 FINALISTS going in: {champion_distill_a15_seedavg5 0.910837, champion_archblend4 0.899643}.
   a15 is a placeholder for ARM E and should be replaced by alphamix10 per the read above.
   archblend4 stays as the DECORRELATED hedge: every distill artifact is built on the same teacher,
   so they are not second opinions of each other.

 DEADLINE 2026-08-16. This is the last iteration with room for a feature experiment; iter44 should
 be finalist consolidation and the code-review package (35% of the final score).
=====================================================================
NEXT
