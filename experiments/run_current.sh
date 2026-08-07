#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 36 — SWA / SWAD: capacity-neutral WEIGHT averaging (round-16 Agent 5).   *** up to 2 uploads ***
#
#   WHY. Prediction seed-averaging gave a REAL +0.0055 LEVEL gain on permanence (member mean 0.8944 ->
#   5-seed-avg 0.899882) -- averaging is a live lever here, not just variance. Agent 5's thesis: averaging
#   WEIGHTS along one run (SWA/SWAD) lands at a FLATTER minimum -> a level gain NOT capped by the
#   (1-rho)/N ceiling that limits prediction-averaging. Our training uses a CONSTANT LR (no scheduler),
#   so the tail iterates genuinely explore the basin -> plain tail weight-averaging IS valid SWA.
#   LayerNorm (no BatchNorm) -> nothing to recompute after averaging. Capacity-NEUTRAL: 0 new params,
#   1x inference. Code built + unit-tested (enable:false = champion bit-for-bit; enable:true averages).
#
#   THE MOVE. Take the single-tau permanence champion and turn SWA on (average the last 30 of 60 epochs,
#   dense). Run it at the SAME 5 seeds as the non-SWA seed distribution {42,7,13,21,29} and seed-average.
#     SWA="--set seq.swa.enable=true --set seq.swa.start_frac=0.5"
#
#   THE READ (committed).
#     champion_perm_swa_seedavg5  vs  the non-SWA seed-avg 0.899882:
#       >= +0.006 -> SWA buys LEVEL on top of seed-averaging -> adopt SWA for the finalist + iter37.
#       within +-0.006 -> SWA is flat here (shallow head too near-convex, Agent 5's risk) -> drop it.
#     c_perm_swa (seed 42)  vs  c_perm_single 0.906492:
#       also watch whether SWA RAISES THE LOW SEEDS (13=0.8917, 21=0.8786) = variance compression, a
#       robustness win for the finalist even if the top seed is unchanged.
#   OOF is blind (do not judge by it); the paired LB is the only truth.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"
PERM="--set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"
SWA="--set seq.swa.enable=true --set seq.swa.start_frac=0.5"

# ---- 0. BASELINE (non-SWA), reproduces c_perm_single = 0.906492. ----
python run_pipeline.py $COMMON --name c_perm_single $PERM

# ---- 1. SWA permanence at 5 seeds (42 = default). ----
python run_pipeline.py $COMMON --name c_perm_swa $PERM $SWA
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "c_perm_swa_s${SD}" $PERM $SWA --set seed=$SD
done

# ---- 2. SWA seed-average (the decisive artifact vs the non-SWA seed-avg 0.899882). ----
python tools/seed_average.py --variant c_perm_swa --name champion_perm_swa_seedavg5

cat <<'NEXT'
=====================================================================
 PASTE BACK all summary lines. Every run MUST log n_features 25; the SWA runs MUST log a line
 "SWA: averaged N tail snapshots" (else the flag didn't reach the model). c_perm_single MUST
 reproduce ~0.906492.

 UPLOAD (budget 5/day):
   1. submissions/submission_champion_perm_swa_seedavg5.csv   <- vs non-SWA seed-avg 0.899882 (THE test)
   2. submissions/submission_c_perm_swa.csv                   <- seed 42, vs c_perm_single 0.906492

 COMMITTED READ:
   swa_seedavg - 0.899882 >= +0.006 -> SWA buys LEVEL -> adopt for finalist + iter37 (SWA on archblend
                                       members / on the replacement feature).
   within +-0.006 -> SWA flat (near-convex head) -> drop; finalist stays the plain seed-avg 0.899882.
   c_perm_swa (s42) vs 0.906492 and the low seeds: RAISED low seeds = variance compression = a robust
                                       finalist even if the mean is flat.
=====================================================================
NEXT
