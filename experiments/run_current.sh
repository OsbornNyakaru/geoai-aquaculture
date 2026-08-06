#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 35 — CAPACITY-NEUTRAL tau selection + the REAL single-tau seed-avg.   *** up to 3 uploads ***
#
#   WHAT iter34 SETTLED. Adding a SECOND channel to the permanence champion HURTS: pondband -0.0061,
#   vhsq -0.0178, vvperm -0.0153. Clean mechanism -- every arm's OOF rose but every LB fell -> extra
#   Linear params overfit the adv-AUC-0.89 shift. The 25-channel permanence model is a CAPACITY SWEET
#   SPOT. So we STOP adding channels and do the only lever that pays no capacity tax:
#
#   (1) CAPACITY-NEUTRAL tau feature-selection. We only ever tested tau=-21 as the SOLE permanence
#       threshold (=0.906492). Scan single tau in {-22, -20.5, -20} -- each stays at 25 channels, no
#       capacity added. Is -21 the best single cut, or is there a better one? seed 42, directional.
#
#   (2) THE REAL single-tau SEED-AVERAGE (our permanence FINALIST, still unmeasured -- the 0.8969
#       upload was the STALE iter32 4-tau file). Rebuild seq_a_xview_perm (single-tau) at 5 seeds and
#       pool. Given s42=0.9065 + s29=0.9007, expect >= 0.90. Fresh name to kill the stale-file trap.
#
#   ALSO PENDING (already generated in iter34, no rerun -- upload when slots refresh):
#       submissions/submission_c_perm_ricegate.csv  -> the last single-channel-add test (low prior).
#
#   THE READ (committed).
#     tau scan: any single-tau arm - 0.906492 >= +0.006 -> better cut; adopt it, seed-confirm in iter36.
#               within +-0.006 -> -21 stays the champion tau (feature selection DONE on this lane).
#     seed-avg: champion_perm_seedavg5_st is the finalist number. >= 0.900 -> strong robust finalist
#               (pair with champion_archblend4 0.899643, different error profiles). ~0.897 -> still our
#               best robust legal model; designate it anyway.
#   NOTE representation is FIXED here (still 25 ch, permanence only) so cross-run pairing is clean;
#   OOF stays blind (iter34: highest OOF LOST) -- the paired LB is the only truth.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"
PERMBASE="--set seq.channels.permanence=true"   # single-tau set per-run via cdf_taus

# ---- 1. CAPACITY-NEUTRAL single-tau scan (each = 25 channels, permanence only). ----
python run_pipeline.py $COMMON --name c_perm_single $PERMBASE --set seq.channels.cdf_taus=[-21.0]   # = 0.906492 baseline (repro)
python run_pipeline.py $COMMON --name c_perm_t22    $PERMBASE --set seq.channels.cdf_taus=[-22.0]
python run_pipeline.py $COMMON --name c_perm_t205   $PERMBASE --set seq.channels.cdf_taus=[-20.5]
python run_pipeline.py $COMMON --name c_perm_t20    $PERMBASE --set seq.channels.cdf_taus=[-20.0]

# ---- 2. THE REAL single-tau SEED-AVERAGE (the permanence finalist). ----
python run_pipeline.py $COMMON --name seq_a_xview_perm $PERMBASE --set seq.channels.cdf_taus=[-21.0]           # seed 42 (= c_perm_single)
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_xview_perm_s${SD}" $PERMBASE --set seq.channels.cdf_taus=[-21.0] --set seed=$SD
done
python tools/seed_average.py --variant seq_a_xview_perm --name champion_perm_seedavg5_st

cat <<'NEXT'
=====================================================================
 PASTE BACK all summary lines. Every run MUST log n_features 25 (permanence only, capacity-neutral).
 c_perm_single MUST reproduce ~0.906492 on upload, else the run is VOID.

 UPLOAD PRIORITY (budget 5/day):
   1. submissions/submission_champion_perm_seedavg5_st.csv   <- THE FINALIST (real single-tau seed-avg)
   2. the BEST single-tau scan arm, IF its OOF/your judgement suggests it may beat -21
   3. submissions/submission_c_perm_ricegate.csv             <- leftover from iter34, last add-lane test

 COMMITTED READ (each paired vs c_perm_single = 0.906492):
   tau scan  >= 0.9125 -> a better single cut than -21; adopt + seed-confirm (iter36).
             0.9005-0.9125 -> tie; -21 stays the champion tau (tau selection DONE).
             <= 0.9005 -> worse cut; -21 confirmed optimal.
   seed-avg (champion_perm_seedavg5_st): >= 0.900 -> strong robust FINALIST; pair with archblend4.
                                          ~0.897  -> still our best robust legal model; designate it.
=====================================================================
NEXT
