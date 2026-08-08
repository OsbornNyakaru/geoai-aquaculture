#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 39 — SEED-CONFIRM mean_min + FREE tree-lane gate + fixed MODE-B screen.  *** up to 2 uploads ***
#
#   iter38 SCREEN: mean_min pooling = 0.912759 (+0.0063 vs c_perm_single 0.906492), clears the >=0.9125
#   gate. It adds the temporal LOW-TAIL of the hidden state = the "permanent low scatterer" pond signal,
#   and it beat mean_std (dispersion) -> the low tail matters more than spread. BUT single-seed-42 -- our
#   THIRD such candidate; the prior two (soft, vhsq) both WASHED OUT on seed-averaging. So the ONLY thing
#   that matters now is the seed-confirm.
#
#   ARM 1 -- SEED-CONFIRM mean_min: permanence single-tau + pooling=mean_min at seeds {42,7,13,21,29} +
#     seed-avg. Read vs the perm seed-avg 0.899882.
#   ARM 2 (FREE, 0 subs) -- TREE-LANE GO/NO-GO gate: adversarial validation decides if a shift-robust
#     CatBoost can transfer at all (test-like-holdout vs random-holdout). This gates the whole tree build.
#   ARM 3 (FREE, 0 subs) -- the fixed MODE-B feature screen (adv-AUC vs label-AUC) to rank iter40 features.
#
#   THE READ (committed).
#     champion_meanmin_seedavg5 - perm seed-avg 0.899882 >= +0.006 -> mean_min is a REAL seed-robust win
#       -> new champion/finalist; iter40 = the n-INVARIANT low-quantile pooling (p10/p25, robust vs min's
#          n-bias) + fold mean_min into the finalist.
#     within +-0.006 -> 0.9128 was seed-42 luck (like soft & vhsq); pooling axis tapped; keep perm seed-avg.
#   adversarial_cv VERDICT: [GO] -> build the shift-robust CatBoost lane; [NO-GO] -> trees can't transfer,
#     stay with the Transformer + move to finalists/writeup.
#   OOF is blind; paired LB only.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"
PERM="--set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"
MEANMIN="--set seq.pooling=mean_min"

# ---- 0. FREE offline gates (0 submissions) — run first so their verdicts are ready. ----
echo "=== TREE-LANE ADVERSARIAL-VALIDATION GATE (free) ==="
python tools/adversarial_cv.py || true
echo "=== MODE-B FEATURE SCREEN (fixed; free) ==="
python tools/shift_diagnostics.py --mode screen || true

# ---- 1. SEED-CONFIRM mean_min (the only thing that decides the win). ----
python run_pipeline.py $COMMON --name seq_a_meanmin $PERM $MEANMIN               # seed 42 (= 0.912759)
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_meanmin_s${SD}" $PERM $MEANMIN --set seed=$SD
done
python tools/seed_average.py --variant seq_a_meanmin --name champion_meanmin_seedavg5

cat <<'NEXT'
=====================================================================
 PASTE BACK: the adversarial_cv GATE verdict block, the MODE-B screen table, and all run summary lines.
 seq_a_meanmin MUST reproduce ~0.912759.

 UPLOAD (budget 5/day):
   1. submissions/submission_champion_meanmin_seedavg5.csv     <- SEED-CONFIRM mean_min (THE result)
   2. (still pending from iter37) submission_champion_replvhsq_swa_seedavg5.csv  -> vs 0.899512 (SWA level?)

 COMMITTED READ:
   meanmin_seedavg5 >= 0.9059 (=+0.006 vs perm seed-avg 0.899882) -> REAL seed-robust win; new finalist #1;
       iter40 = n-invariant low-quantile pooling (p10/p25) as the robust form of mean_min.
   within +-0.006 -> 0.9128 was seed-42 luck (3rd time); pooling axis tapped -> lock finalists + Phase-Two.
   adversarial_cv [GO] -> I build the shift-robust CatBoost lane next; [NO-GO] -> trees are out, pivot.
=====================================================================
NEXT
