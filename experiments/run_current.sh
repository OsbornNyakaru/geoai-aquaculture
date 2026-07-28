#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 29 — DOES A BIGGER POOL WIN NOW?              *** 1 SUBMISSION ***
#                + verify the reproduction claim          (0 subs)
#
#   WHAT CHANGED, AND WHY IT REOPENS A CLOSED LANE.
#   Removing the rules-violating prevalence pin did not just cost nothing -- it made the
#   ENSEMBLE work:
#
#       pinned:   archblend4 - champion  =  -0.0009      pooling bought NOTHING
#       legal:    archblend4 - champion  =  +0.0100      pooling buys LEVEL
#                 (0.899643 vs 0.889686 -- our best public score ever, and eligible)
#
#   MECHANISM. The pin overwrote every member's operating point to a fixed 0.649, so pooling
#   could only average the RANKING -- and at mean rho 0.9524 there is almost nothing
#   independent left to average. A literal 0.5 cut also averages the members' CALIBRATION,
#   where they genuinely disagree: their individual legal positive rates were
#   0.581 / 0.570 / 0.534 / 0.586. The pin was collapsing that spread to one number and
#   throwing it away.
#
#   CONSEQUENCE: iter18's "architecture pooling is MARGINAL, not a level lever" was an
#   artifact of the OPERATING POINT, not a property of the ensemble. Every ensemble rule we
#   derived under the pin is now UNVERIFIED.
#
#   THE RULE THIS ITERATION TESTS. "Gate members on LEVEL GAP, not correlation" -- derived
#   from iter22/iter24, where a weaker member dragged the blend (-0.009 for ROCKET at -0.040
#   level, -0.0155 for GBDT at -0.011 level). BOTH WERE MEASURED UNDER THE PIN, where a weak
#   member could only contribute bad ORDER and had no calibration to contribute. Under a
#   literal 0.5 cut a weaker member may still carry USEFUL, INDEPENDENT CALIBRATION.
#
#   THE EXPERIMENT. archblend6 = archblend4 + seq_a_k4 + seq_a_base. Those two are our
#   weakest same-class members (pinned 0.8665 and 0.8780, i.e. -0.029 and -0.018 against the
#   champion) but their legal positive rates, 0.534 and 0.530, sit at the EDGE of the current
#   spread -- exactly the calibration diversity the hypothesis says is now valuable.
#
#   DECISIVE EITHER WAY, at the cost of one submission:
#     archblend6 > 0.8996  -> the level-gap gate is DEAD under a literal cut. Pool
#                             aggressively; every weak-but-differently-calibrated model we
#                             have is now an asset, and the GBDT/ROCKET blends deserve
#                             re-testing.
#     archblend6 < 0.8996  -> the level-gap gate SURVIVES the regime change. archblend4 is
#                             final, and the ensemble lane closes for good.
#
#   NOT USING THE OFFLINE SCREEN. ATC-F1 is unreliable on two counts (out-of-family since
#   iter26, and defined against the operating point we deleted). It gates nothing here.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- 1. Every archblend member + the champion seed spread. Bundles feed steps 2-3. ----
# Legal mode is the default, so submission_seq_a_xview.csv is the legal champion (LB 0.889686)
# and no separate run is needed. Bundles store RAW pre-calibration probabilities, so they are
# operating-point agnostic and experiments/anchors.tsv stays reproducible.
PRE="--set seq.relative_time=false --set seq.consistency_lambda=0"
python run_pipeline.py $COMMON --name seq_a_k4      $PRE --set seq.K=4      # archblend6 member
python run_pipeline.py $COMMON --name seq_a_base    $PRE                    # archblend6 member
python run_pipeline.py $COMMON --name seq_a_reltime --set seq.consistency_lambda=0
python run_pipeline.py $COMMON --name seq_a_nope    --set seq.consistency_lambda=0 --set seq.pos_encoding=none
python run_pipeline.py $COMMON --name seq_a_l3      --set seq.consistency_lambda=3
python run_pipeline.py $COMMON --name seq_a_xview                           # CHAMPION (legal)
python run_pipeline.py $COMMON --name seq_a_reltime_s7 --set seed=7 --set seq.consistency_lambda=0
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_xview_s${SD}" --set seed=$SD
done

# ---- 2. The control and the candidate, built identically. ----
# archblend4 is the CONTROL: it must reproduce pos-rate 0.5670 and its LB is known (0.899643),
# so if it drifts, something else changed and the comparison is void.
echo "=== CONTROL: archblend4 (known LB 0.899643, expect pos-rate 0.5670) ==="
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --name champion_archblend4

echo "=== CANDIDATE: archblend6 (+k4 +base) ==="
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview seq_a_k4 seq_a_base \
  --name champion_archblend6

# ---- 3. Verify the reproduction claim REPORT.md makes (0 submissions). ----
# The claim was never tested end to end. It OOM'd on the local box (not enough RAM), and the
# script itself was stale -- its banner still asserted the pinned t_star 0.4450 and pos-rate
# 0.649, so a reviewer running it today would have concluded the solution does not reproduce.
# Rewritten; this is the first real test of it.
echo "=== REPRODUCTION CHECK (stage 1 only) ==="
bash experiments/reproduce_champion.sh --quick

cat <<'NEXT'
=====================================================================
 PASTE BACK:
   (i)   the CONTROL archblend4 block -- its per-member slopes and POOLED pos-rate.
         It MUST read pos-rate 0.5670. If it does not, something else moved and the
         comparison below is VOID -- say so and we stop.
   (ii)  the CANDIDATE archblend6 block -- per-member slopes, POOLED pos-rate, and the
         6x6 correlation matrix.
   (iii) the REPRODUCTION CHECK output, including the COMPLIANCE AUDIT lines.

 THEN UPLOAD ONE FILE:
   submissions/submission_champion_archblend6.csv   -> compare against archblend4's 0.899643

 THE READ, COMMITTED IN ADVANCE (note our paired resolution: SUGGESTIVE at 0.006,
 CONFIDENT at 0.012; the two blends share 4 of 6 members so this is strongly paired):
   >= 0.9056  -> CONFIDENT WIN. The level-gap gate is dead under a literal cut. Pool
                 aggressively and re-test the GBDT and ROCKET members, which were rejected
                 under the pin on a rule that does not survive.
   0.9002 to 0.9056 -> suggestive win; worth one confirming variant.
   0.8990 to 0.9002 -> no effect. archblend4 stays finalist #1.
   <= 0.8936  -> CONFIDENT LOSS. The level-gap gate SURVIVES. Ensemble lane closes for good,
                 archblend4 is final, and everything left goes to the writeup.

 REMINDER: finalists are already designated. This cannot cost us the current position --
 archblend6 only replaces archblend4 if it demonstrably beats it.
=====================================================================
NEXT
