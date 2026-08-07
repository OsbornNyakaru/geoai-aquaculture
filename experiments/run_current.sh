#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 35 — CAPACITY-NEUTRAL levers (round-16 research) + the REAL single-tau seed-avg.  *** up to 3 uploads ***
#
#   ROUND-16 RESEARCH (4 agents) RE-AIMED THIS ITERATION. iter34 proved adding a 2nd channel HURTS, and
#   the theory says WHY: it is a SHIFT-CARRIER / divergence tax (d_HdH + dlambda*), not mere capacity
#   noise -- every added coord raised OOF (source risk) but paid a train/test-divergence penalty the LB
#   sees (Agent 3; pos-rate sliding toward the train prior 0.40 was the tell). The cure is CAPACITY-
#   NEUTRAL. Two untested, theory-backed moves emerged; the hard-tau scan I had staged was KILLED
#   (Agent 4: tau=-21 is on the physical optimum, a hard-tau move is common-mode and sub-floor).
#
#   ARM 1 (Agent 4, TOP PICK) -- SOFT permanence. Replace hard 1[VH<-21] with sigma(0.5*(-21-VH)),
#     a FIXED-slope sigmoid = 0 new params. At n=4-6 the hard fraction is quantized to <=6 levels
#     (rank ties + train/test level-set mismatch); the soft ramp uses each month's DISTANCE below tau
#     ("permanence depth") = the optimal rank-1 log-likelihood-ratio coordinate. Capacity-neutral.
#
#   ARMS 2-3 (Agents 1&2, FLAGSHIP) -- CHANNEL REPLACEMENT. VH & VV are co-observed every month, so
#     their missing-indicators are the IDENTICAL vector (R=1, info-free). Drop the duplicate (VV) one
#     and put a NEW nonlinear coordinate in its place -> width stays 25 (the sweet spot). This is the
#     UNTESTED cell of the 2x2: adding a channel HURT (iter34), deleting a band HURT (iter26),
#     REPLACEMENT holds width and dodges both. Payloads: VH^2 (Var_t, rice-killer) and the SARxoptical
#     AND-gate 1[VH<-21].1[NDVI<0.25] (pond = dark AND not-green).
#
#   ARM 4 -- the REAL single-tau SEED-AVERAGE finalist (the 0.8969 upload was the STALE 4-tau file;
#     s42=0.9065 + s29=0.9007 -> expect >=0.90). champion_perm_seedavg5_st.
#
#   THE READ (committed, each paired vs c_perm_single = 0.906492, seed 42, directional).
#     any arm - baseline >= +0.006 -> a real capacity-neutral WIN; seed-confirm in iter36, new champion.
#     within +-0.006 -> tie at the lucky seed; capacity-neutral lane also flat -> pivot to finalize.
#     <= -0.006 -> that lever hurts; drop it.
#   OOF stays blind (iter34: highest OOF LOST) -- paired LB only. Single seed 42 (lucky) -> directional.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"
PERM="--set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"
REPL="--set seq.channels.drop_dup_s1_indicator=true"   # frees one channel (25 -> stays 25 when +1 coord)

# ---- 0. BASELINE: reproduce c_perm_single (hard single-tau permanence, seed 42 = 0.906492). ----
python run_pipeline.py $COMMON --name c_perm_single $PERM

# ---- 1. SOFT permanence (Agent 4 top pick; capacity-neutral shape refinement). ----
python run_pipeline.py $COMMON --name c_perm_soft $PERM --set seq.channels.permanence_soft=true --set seq.channels.soft_slope=0.5

# ---- 2-3. CHANNEL REPLACEMENT: drop duplicate VV indicator, add ONE new coordinate (width stays 25). ----
python run_pipeline.py $COMMON --name c_repl_vhsq     $PERM $REPL --set seq.channels.vh_sq=true
python run_pipeline.py $COMMON --name c_repl_ricegate $PERM $REPL --set seq.channels.rice_gate=true

# ---- 4. THE REAL single-tau SEED-AVERAGE (the permanence finalist). ----
python run_pipeline.py $COMMON --name seq_a_xview_perm $PERM              # seed 42 (= c_perm_single)
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_xview_perm_s${SD}" $PERM --set seed=$SD
done
python tools/seed_average.py --variant seq_a_xview_perm --name champion_perm_seedavg5_st

# ---- 5. OFFLINE LABEL-SHIFT GATE (round-16, 0 submissions) -- decides the Saerens F1 lever + the
#         pure-label-vs-conditional-shift question that gates the finalists. Runs on the champion bundle. ----
echo "=== LABEL-SHIFT GATE (offline, legal, 0 submissions) ==="
python tools/label_shift_gate.py --bundle submissions/preds/preds_c_perm_single.npz --n-boot 4000 || true

cat <<'NEXT'
=====================================================================
 PASTE BACK all summary lines. Widths MUST be: c_perm_single 25, c_perm_soft 25, c_repl_vhsq 25,
 c_repl_ricegate 25 (12 val + 11 miss + 1 perm + 1 new coord). c_perm_single MUST reproduce ~0.906492.

 UPLOAD PRIORITY (budget 5/day):
   1. submissions/submission_champion_perm_seedavg5_st.csv   <- THE FINALIST (real single-tau seed-avg)
   2. submissions/submission_c_perm_soft.csv                 <- Agent-4 top pick (best odds to clear)
   3. the better of submission_c_repl_vhsq.csv / submission_c_repl_ricegate.csv

 COMMITTED READ (each paired vs c_perm_single = 0.906492):
   >= 0.9125 -> real capacity-neutral WIN; seed-confirm (iter36), new champion candidate.
   0.9005-0.9125 -> tie at the lucky seed; capacity-neutral lane flat -> finalize.
   <= 0.9005 -> that lever hurts; drop it.
   seed-avg (champion_perm_seedavg5_st): >= 0.900 -> strong robust FINALIST; pair with archblend4.

 LABEL-SHIFT GATE (printed above): PASTE the full block back.
   [PASS] -> the Saerens prior-shift correction is SAFE. To ship it (worth ~+0.010..+0.019 on the F1
             half, capacity-free, LEGAL), run e.g.:
       python tools/label_shift_gate.py --bundle submissions/preds/preds_champion_perm_seedavg5_st.npz \
              --shrink 0.5 --emit-submission champion_perm_seedavg5_st_saerens
     then upload submission_champion_perm_seedavg5_st_saerens.csv (AUC column unchanged; only F1 moves).
   [FAIL] -> conditional shift; do NOT apply Saerens. Keep the literal 0.5 cut and weight archblend4 as
             the safer primary finalist (its no-single-feature-dependence hedges the fragile perm bet).
=====================================================================
NEXT
