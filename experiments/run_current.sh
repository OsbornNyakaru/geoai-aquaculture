#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 33 — THE PERMANENCE ENSEMBLE (stack the two +0.010 lifts).   *** 1 submission ***
#
#   CONFIRMED (iter31-32). VH permanence channels 1[VH_dB(t)<tau] lift the Transformer by a
#   SEED-ROBUST +0.010: champion_perm_seedavg5 = 0.896918 vs champion seed-avg ~0.8865. The 4-tau grid
#   {-22,-21,-20,-19} is best (6-tau `wide` scored lower). cross_pol stays OFF (toxic, -0.023).
#
#   THE MOVE. archblend4 (0.899643) got +0.010 over its members by pooling 4 ARCHITECTURES (calibration
#   diversity, iter28). Permanence gives +0.010 to each base model. If the two lifts STACK, a blend of
#   PERMANENCE-transformers should clear ~0.905+. This iteration builds exactly that and compares it,
#   same-run, against a freshly rebuilt archblend4 control.
#
#   Members mirror archblend4 exactly, but every one has permanence ON:
#     seq_a_{reltime,nope,l3,xview}_perm  (xview_perm = the iter31 winner, at 5 seeds).
#   0 new code -- just `--set seq.channels.permanence=true` on the existing member recipes.
#
#   THE READ (committed).
#     champion_perm_archblend4 - champion_archblend4 >= +0.006  -> the lifts STACK. New best, new
#         finalist #1; permanence direction has more to give (iter34 = next single feature).
#     within +-0.006  -> the +0.010 ensemble lift and the +0.010 permanence lift do NOT stack (pooling
#         already captured most of what permanence adds). Then the finalist pair is archblend4 +
#         champion_perm_seedavg5 (two ~0.897-0.900 artifacts with different error profiles), and we
#         return to single-feature permanence engineering (iter34).
#   NOTE representation change -> ATC-F1 screens nothing; the paired LB is the only truth.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"
# Per-member config recipes (identical to archblend4). Permanence is added by the _perm arms only.
r_reltime="--set seq.consistency_lambda=0"
r_nope="--set seq.consistency_lambda=0 --set seq.pos_encoding=none"
r_l3="--set seq.consistency_lambda=3"
r_xview=""                       # champion: relative_time + consistency_lambda=1 (defaults)
# iter32 feature-selection winner: a SINGLE threshold tau=-21 beat the 4-tau and 6-tau profiles
# (LB 0.9065 > 0.9016 > 0.8987). The permanence ensemble uses the single-tau config.
PERM="--set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"

# ---- 1. BASE members (control archblend4), mirroring the known 0.899643 build. ----
python run_pipeline.py $COMMON --name seq_a_reltime $r_reltime
python run_pipeline.py $COMMON --name seq_a_reltime_s7 $r_reltime --set seed=7
python run_pipeline.py $COMMON --name seq_a_nope    $r_nope
python run_pipeline.py $COMMON --name seq_a_l3      $r_l3
python run_pipeline.py $COMMON --name seq_a_xview   $r_xview
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_xview_s${SD}" $r_xview --set seed=$SD
done

# ---- 2. PERMANENCE members (same recipes + permanence channels). ----
python run_pipeline.py $COMMON --name seq_a_reltime_perm    $r_reltime $PERM
python run_pipeline.py $COMMON --name seq_a_reltime_perm_s7 $r_reltime $PERM --set seed=7
python run_pipeline.py $COMMON --name seq_a_nope_perm       $r_nope    $PERM
python run_pipeline.py $COMMON --name seq_a_l3_perm         $r_l3      $PERM
python run_pipeline.py $COMMON --name seq_a_xview_perm      $r_xview   $PERM      # = the iter31 winner
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_xview_perm_s${SD}" $r_xview $PERM --set seed=$SD
done

# ---- 3. CONTROL + CANDIDATE blends (legal calibrated pool), + the permanence seed-avg anchor. ----
echo "=== CONTROL: archblend4 (must reproduce pos-rate 0.5670 / LB 0.899643) ==="
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --name champion_archblend4
echo "=== CANDIDATE: perm_archblend4 = the same 4 architectures, all with VH permanence ==="
python tools/arch_blend.py \
  --members seq_a_reltime_perm seq_a_nope_perm seq_a_l3_perm seq_a_xview_perm \
  --name champion_perm_archblend4
python tools/seed_average.py --variant seq_a_xview_perm --name champion_perm_seedavg5

cat <<'NEXT'
=====================================================================
 PASTE BACK: the CONTROL archblend4 pooled pos-rate (MUST be 0.5670, else VOID), the CANDIDATE
 perm_archblend4 block (per-member Platt slopes, pooled pos-rate, 4x4 correlation matrix), and the
 champion_perm_seedavg5 pooled line.

 THEN UPLOAD ONE FILE:
   submissions/submission_champion_perm_archblend4.csv   -> vs archblend4's 0.899643

 COMMITTED READ:
   perm_archblend4 >= 0.9056  -> CONFIDENT: the permanence + ensemble lifts STACK. New best & new
                                 finalist #1. Permanence direction keeps giving -> iter34 next feature.
   0.9002 - 0.9056  -> suggestive stack; worth one confirming variant.
   0.8990 - 0.9002  -> the lifts do NOT stack (pooling already captured permanence). Finalists =
                       archblend4 + champion_perm_seedavg5; back to single-feature permanence (iter34).
   <= 0.8936  -> permanence hurts the ensemble (unexpected) -> archblend4 stays, reassess.
=====================================================================
NEXT
