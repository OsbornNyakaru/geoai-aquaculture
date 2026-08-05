#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 32 — SEED-CONFIRM THE PERMANENCE WIN + first tau selection.   *** 1-2 submissions ***
#
#   THE WIN (iter31). Adding per-month VH permanence indicators 1[VH_dB(t) < tau] as Transformer
#   input channels scored c_perm = 0.901605 -- our BEST public score ever, above the 4-member
#   archblend4 (0.899643), as a single legal model. cross_pol (VH-VV) was isolated and shown TOXIC
#   (-0.0228), so it is OFF here. Direction from now on: PERMANENCE, one change at a time.
#
#   WHY SEED-CONFIRM FIRST. c_perm is a single seed-42 run, and seed 42 is our historically LUCKY
#   draw (old champion 0.8955@42 vs 0.8764@7 -- a 0.019 swing). +0.0119 sits right at the ~0.013
#   public resolution. Before we build anything on permanence we must know the SEED-AVERAGED win.
#
#   THE READ (committed in advance).
#     perm_seedavg5 - xview_seedavg5 >= +0.006 (matched 5-seed avg -> seed noise removed)
#         -> REAL, seed-robust win. Permanence is a finalist candidate (simpler + higher than
#            archblend4). iter33 = winning tau set + permanence-in-the-ensemble.
#     within +-0.006
#         -> 0.9016 was largely seed-42 luck; permanence is a tie. Reassess before pushing further.
#   NOTE: this is a REPRESENTATION change -> ATC-F1 is out-of-family and screens NOTHING. The ground
#   truth is the seed-averaged paired LB, nothing else.
# =====================================================================
set -euo pipefail

# The iter31 winner: champion (relative_time + consistency_lambda=1) + VH permanence indicators.
PERM="--full --model seq --set seq.channels.permanence=true"
XV="--full --model seq"

# ---- A. SEED-CONFIRM (the gate). c_perm at 5 seeds vs the champion at the same 5 seeds. ----
python run_pipeline.py $PERM --name c_perm                       # seed 42 (re-anchors 0.901605)
for SD in 7 13 21 29; do
  python run_pipeline.py $PERM --name "c_perm_s${SD}" --set seed=$SD
done
python run_pipeline.py $XV --name seq_a_xview                    # champion seed 42 (~0.8897)
python run_pipeline.py $XV --name seq_a_xview_s7  --set seed=7
for SD in 13 21 29; do
  python run_pipeline.py $XV --name "seq_a_xview_s${SD}" --set seed=$SD
done

# ---- B. First tau feature-selection probes (one change each, seed 42, suggestive). ----
echo "=== c_perm_single: permanence with ONE threshold tau=-21 (is one enough?) ==="
python run_pipeline.py $PERM --name c_perm_single --set seq.channels.cdf_taus="[-21.0]"
echo "=== c_perm_wide: richer profile tau=-23..-18 (does more resolution help?) ==="
python run_pipeline.py $PERM --name c_perm_wide \
  --set seq.channels.cdf_taus="[-23.0,-22.0,-21.0,-20.0,-19.0,-18.0]"

# ---- C. Seed-average both the permanence model and the champion (the reliable comparison). ----
python tools/seed_average.py --variant c_perm      --name champion_perm_seedavg5
python tools/seed_average.py --variant seq_a_xview  --name champion_seedavg5

cat <<'NEXT'
=====================================================================
 PASTE BACK: the `run:` summaries for c_perm (+ s7/s13/s21/s29), seq_a_xview (+ its 4 seeds),
 c_perm_single, c_perm_wide; AND both seed_average blocks (their pooled pos-rate + per-seed rank-corr).

 THEN UPLOAD (up to 2):
   - submission_champion_perm_seedavg5.csv   -> the reliable permanence estimate (MAIN EVENT)
   - (optional) submission_c_perm_wide.csv   -> richer-profile single-seed read

 COMMITTED READ:
   perm_seedavg5 vs champion_seedavg5 (matched 5-seed avg; seed noise removed):
     >= +0.006  -> permanence is a REAL, seed-robust win. It becomes a finalist candidate (beats
                   archblend4, and simpler). iter33 = the winning tau set + rebuild the archblend
                   members WITH permanence (a permanence ensemble).
     within +-0.006 -> 0.9016 was mostly seed-42 luck; permanence ties the champion. Reassess.
   tau probes (single-seed, directional): c_perm_single ~ c_perm -> one threshold suffices;
                   c_perm_wide > c_perm -> a richer profile helps (carry to iter33).
=====================================================================
NEXT
