#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 25 — BAND DELETION FROM THE PHASE-A 2-D SHIFT SCREEN.   *** SCREEN, 0 SUBS ***
#
#   WHERE THIS CAME FROM. Round-11 research (gemini_loop/RESEARCH_11.md, 8 agents) plus a
#   local Phase-A audit (tools/shift_audit.py) that cost ZERO submissions and produced two
#   results, one of which killed a lane and one of which opened this one.
#
#   KILLED, FOR FREE: three independent agents predicted our per-band MISSING-INDICATOR
#   channels were a shifted nuisance worth deleting. Measured, masked-train vs test:
#       indicators ALONE          adv-AUC 0.4758   (below chance)
#       S2-cloud indicators alone adv-AUC 0.4744
#       what they ADD over values         +0.0028
#   They carry essentially zero train/test information -- because apply_mask already applies
#   S2 dropout at rates MEASURED OFF TEST, so the train indicator distribution was matched by
#   construction. Lane closed, no submission spent. (Also NOT iter13's compact_missing, 24->14.)
#
#   OPENED: the shift lives in the VALUES (adv-AUC 0.8915 on masked, left-aligned values --
#   vs 0.965-0.976 for Presto on raw pixels, so masking+left-align already removed a real
#   chunk). The 2-D screen scores each band on A = separates train/test, T = predicts label:
#       VV     A=0.5907  T=0.7801   <- TOP shift-carrier, and VH dominates it on signal (0.8302)
#       blue   A=0.5344  T=0.5963   <- barely predictive; most Rayleigh-scattered band
#       VH     A=0.5622  T=0.8302   <- REPAIR, never delete (this is our primary signal)
#   Only high-A/low-T bands are free deletions. A one-axis rule would have deleted amplitude,
#   which we PROVED is fatal (c_rank collapsed OOF 0.975 -> 0.86).
#
#   TWO INDEPENDENT ROUTES NAME VV. Our data says top shift-carrier, dominated by VH. The SAR
#   literature says VV is wind-sensitive, its water threshold drifts 2.6 dB/yr vs VH's 2.1, and
#   VH is preferred because its backscatter histogram is cleanly bimodal (Ottinger 2017/2019;
#   Li 2018, Dongting). Capacity-REDUCING, which is the only class of change that has ever won.
#
#   HONEST CAVEATS. (1) VV's T sits 0.0001 BELOW the median -- a knife-edge; the physics breaks
#   the tie, not the screen. (2) The screen is a per-band LINEAR read-out; a band with weak
#   marginal signal could still matter through the cross-band attention the champion uses.
#   (3) Max single-band A is 0.59 vs a joint 0.89, so the shift is DISTRIBUTED -- band deletion
#   cannot collapse it, and we should not expect it to.
#
#   DECISION: SCREEN, 0 subs. Submit only if >=2 cleared estimators beat champion AND the
#   margin exceeds the estimator's own seed sd (0.0576 ATC-F1 == +-0.0094 LB).
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- 1. Anchors: re-certify the estimators (7 known-LB variants) + the champion seed spread. ----
PRE="--set seq.relative_time=false --set seq.consistency_lambda=0"
python run_pipeline.py $COMMON --name seq_a_detrend $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4      $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_base    $PRE
python run_pipeline.py $COMMON --name seq_a_reltime --set seq.consistency_lambda=0
python run_pipeline.py $COMMON --name seq_a_nope    --set seq.consistency_lambda=0 --set seq.pos_encoding=none
python run_pipeline.py $COMMON --name seq_a_l3      --set seq.consistency_lambda=3
python run_pipeline.py $COMMON --name seq_a_xview                                   # CHAMPION
python run_pipeline.py $COMMON --name seq_a_detrend_s7 --set seed=7 $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4_s7      --set seed=7 $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_reltime_s7 --set seed=7 --set seq.consistency_lambda=0
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_xview_s${SD}" --set seed=$SD
done

# ---- 2. THE CANDIDATES. Band deletion, 2 seeds each (for DIS + the seed-noise guard). ----
# Watch the logged `seq input width`: 24 -> 22 for one band, 24 -> 20 for two. If it still says
# 24 the flag did not take effect and the run is VOID (this is exactly the iter12 c_compact bug).
for SD in 42 7; do
  SFX=$([ "$SD" = 42 ] && echo "" || echo "_s${SD}")
  python run_pipeline.py $COMMON --name "c_dropvv${SFX}"     --set seed=$SD \
      --set seq.channels.drop_bands='["VV"]'
  python run_pipeline.py $COMMON --name "c_dropblue${SFX}"   --set seed=$SD \
      --set seq.channels.drop_bands='["blue"]'
  python run_pipeline.py $COMMON --name "c_dropvvblue${SFX}" --set seed=$SD \
      --set seq.channels.drop_bands='["VV","blue"]'
done

# ---- 3. Retro-fit gate + seed floor + screen all three candidates. ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --seed-spread seq_a_xview \
  --screen c_dropvv c_dropblue c_dropvvblue

# ---- 4. Keep the leading finalist current + check whether deletion decorrelates. ----
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --diag-extra c_dropvv c_dropvvblue \
  --name champion_archblend4
python tools/seed_average.py --variant seq_a_xview --name champion_seedavg5

cat <<'NEXT'
=====================================================================
 Paste back: (i) the logged `seq input width` for each c_drop* run (MUST be 22 / 23 / 20 --
 if any says 24 the run is VOID), (ii) the RETRO-FIT + GATE block, (iii) the three SCREEN
 lines, and (iv) the 4 arch_blend matrix (the c_drop* diag rows).

 THE DECISION (I make the single upload call from your paste, 0 subs spent here):
   - A candidate clears >=2 estimators AND its ATC-F1 margin exceeds the seed sd (0.0576)
        -> upload it. A capacity-REDUCING win would be the first level gain since the
           GBDT->Transformer swap, and it is the only change-class that has ever worked here.
   - All HOLD
        -> band deletion cannot beat the distributed shift (consistent with max single-band
           A=0.59 vs joint 0.89). Feature-space deletion is then CLOSED, and the remaining
           live levers are the two-column split (TargetF1 and TargetRAUC are INDEPENDENT
           columns -- see RESEARCH_11 1a) and the Phase-D writeup.
=====================================================================
NEXT
