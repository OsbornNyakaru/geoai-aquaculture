#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 37 — CONFIRM THE WIN + free shift diagnostics.   *** up to 3 uploads ***
#
#   THE WIN (iter35). c_repl_vhsq = 0.913263 = NEW BEST EVER (+0.0068 vs c_perm_single 0.906492, clears
#   the committed >=0.9125 gate). It DROPS the duplicate VV missing-indicator (identical to VH's, R=1,
#   info-free) and adds VH^2 in its place -> CONSTANT width 25. VH^2 was toxic as a 26th ADD (iter34
#   -0.018) but wins as a REPLACEMENT: width was the enemy, not the coordinate. It is also the temporal-
#   dispersion / 2nd-moment axis all of round-17's research converged on. BUT it is single-seed-42 (lucky)
#   -> it MUST be seed-confirmed before it becomes champion/finalist.
#
#   ARM 0 (FREE, 0 submissions) -- SHIFT DIAGNOSTICS. Decides private-LB trust:
#     (A) is the 4-6 month test dropout MAR or SEASONAL? (gates ALL distributional features incl.
#         permanence). (B) adv-AUC vs label-AUC feature screen on windowed train -> pre-ranks the next
#         feature experiments (submission-free replacement for OOF, which is anti-correlated with LB).
#
#   ARM 1 -- SEED-CONFIRM the win: c_repl_vhsq at seeds {42,7,13,21,29} + seed-avg. THE result.
#   ARM 2 -- does SWA (round-17 lever, built) STACK on the new champion? same 5 seeds + SWA + seed-avg.
#
#   THE READ (committed).
#     champion_replvhsq_seedavg5 vs the perm seed-avg 0.899882:
#       >= +0.006 -> the VH^2 replacement is a REAL, seed-robust champion -> new finalist #1; carry the
#                    replacement idea to iter38 (L-scale/IQR/VV-perm replacements, adv-AUC-screened).
#       within +-0.006 -> 0.9133 was largely seed-42 luck; replacement is a wash, reassess.
#     champion_replvhsq_swa_seedavg5 vs champion_replvhsq_seedavg5:
#       >= +0.006 -> SWA stacks on the champion (adopt); else SWA is flat here (drop, per Agent-5 risk).
#   OOF is blind; the paired LB is the only truth. Diagnostics are the offline compass for iter38+.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"
# The new champion feature config: single-tau permanence + drop dup VV indicator + VH^2, width 25.
REPL="--set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0] --set seq.channels.drop_dup_s1_indicator=true --set seq.channels.vh_sq=true"
SWA="--set seq.swa.enable=true --set seq.swa.start_frac=0.5"

# ---- 0. FREE shift diagnostics (0 submissions, legal: train + UNLABELLED test features only). ----
echo "=== SHIFT DIAGNOSTICS (free, offline) ==="
python tools/shift_diagnostics.py --mode both || true

# ---- 1. SEED-CONFIRM the win (non-SWA). ----
python run_pipeline.py $COMMON --name seq_a_replvhsq $REPL              # seed 42 (= c_repl_vhsq = 0.913263)
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_replvhsq_s${SD}" $REPL --set seed=$SD
done
python tools/seed_average.py --variant seq_a_replvhsq --name champion_replvhsq_seedavg5

# ---- 2. Does SWA stack on the champion? (5 seeds + SWA). ----
python run_pipeline.py $COMMON --name seq_a_replvhsq_swa $REPL $SWA
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_replvhsq_swa_s${SD}" $REPL $SWA --set seed=$SD
done
python tools/seed_average.py --variant seq_a_replvhsq_swa --name champion_replvhsq_swa_seedavg5

cat <<'NEXT'
=====================================================================
 PASTE BACK: the FULL diagnostics block (both MODE A and MODE B tables) + all run summary lines. Every
 run MUST log n_features 25 (12 val + 11 miss + 1 perm + 1 vhsq); seq_a_replvhsq MUST reproduce ~0.913263.

 UPLOAD (budget 5/day):
   1. submissions/submission_champion_replvhsq_seedavg5.csv       <- SEED-CONFIRM the win (THE result)
   2. submissions/submission_champion_replvhsq_swa_seedavg5.csv   <- does SWA stack on the champion?
   3. (optional) submission_seq_a_replvhsq.csv                    <- repro check of the 0.913263 seed

 COMMITTED READ:
   replvhsq_seedavg5 >= 0.9059 (=+0.006 vs perm seed-avg 0.899882) -> REAL seed-robust champion; new
       finalist #1. iter38 = adv-AUC-screened REPLACEMENTS (L-scale/IQR dispersion, VV-permanence) guided
       by the MODE B table + (if trees pass MODE-A/windowed-CV) the shift-robust CatBoost lane.
   within +-0.006 -> 0.9133 was seed-42 luck; keep permanence seed-avg as finalist, reassess.
   swa_seedavg5 - replvhsq_seedavg5 >= +0.006 -> SWA stacks (adopt on champion); else drop SWA.

 DIAGNOSTICS READ (informs iter38, costs nothing now):
   MODE A: if windowing explains most of the train->test KS gap for permanence & mean -> distributional
     features (incl. permanence/VH^2) are private-LB-trustworthy. If a big gap remains -> seasonal/
     conditional shift -> down-weight distributional features.
   MODE B: the KEEP list = the ranked, submission-worthy feature candidates for iter38.
=====================================================================
NEXT
