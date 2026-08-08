#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 38 — DISPERSION POOLING on the permanence champion (config-only).   *** up to 2 uploads ***
#
#   WHERE WE ARE. Seed-robust ceiling is ~0.900 and SINGLE-CHANNEL tweaks keep washing out (iter34 adds,
#   iter35/37 replacements: the c_repl_vhsq 0.913263 was seed-42 luck; its 5-seed avg 0.899512 = plain
#   permanence 0.899882). MODE-A confirmed masking is MAR but there is REAL covariate shift on the SAR
#   level (mean_VH/VH^2 gaps not closed by windowing). So the next gains must be STRUCTURAL, not one more
#   channel. Round-17 (agents + deep-research) converged on temporal DISPERSION (ponds stable, rice swings
#   6-8 dB) as the strongest untapped axis, made first-class by a POOLING change -- and mean_std/mean_min/
#   moments are ALREADY IMPLEMENTED (seq.pooling). They were last tested in round-09, on a WORSE base model
#   under the illegal pin -> re-testing on the current permanence champion under legal calibration is a
#   genuinely NEW experiment, at ZERO new code.
#
#   THE MOVE. Permanence single-tau champion x {mean(baseline), mean_std, mean_min, moments}, seed 42
#   directional (screen), paired vs c_perm_single 0.906492. NOTE the built-in std is BIASED 1/N (window-
#   length artifact) -- if a dispersion pool shows promise but underwhelms, iter39 = a proper n-invariant
#   L-scale/GMD pooling (the round-17-preferred unbiased form).
#
#   THE READ (committed, single-seed 42 = directional; winner gets an iter39 seed-confirm before belief).
#     any pooling - 0.906492 >= +0.006 -> dispersion/tail helps -> seed-confirm in iter39 (5 seeds + avg).
#     within +-0.006 -> pooling flat on this base -> the dispersion axis needs the unbiased L-scale form
#                       (iter39) or the lever is genuinely tapped; pivot to the TREE lane.
#   OOF is blind; paired LB only. This is a cheap structural screen before the bigger tree-lane build.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"
PERM="--set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]"

# ---- 0. BASELINE (mean pooling) = c_perm_single = 0.906492. ----
python run_pipeline.py $COMMON --name c_perm_single $PERM

# ---- 1. Dispersion / tail pooling variants on the permanence champion (config-only, width unchanged). ----
python run_pipeline.py $COMMON --name c_perm_meanstd  $PERM --set seq.pooling=mean_std
python run_pipeline.py $COMMON --name c_perm_meanmin  $PERM --set seq.pooling=mean_min
python run_pipeline.py $COMMON --name c_perm_moments  $PERM --set seq.pooling=moments

cat <<'NEXT'
=====================================================================
 PASTE BACK all summary lines. c_perm_single MUST reproduce ~0.906492. The pooling runs log a larger head
 input (mean_std=2x, moments=4x the pooled width) but the same n_features 25 input channels.

 UPLOAD the best 1-2 pooling variants (paired vs c_perm_single 0.906492):
   submissions/submission_c_perm_meanstd.csv
   submissions/submission_c_perm_meanmin.csv
   submissions/submission_c_perm_moments.csv

 COMMITTED READ (single-seed 42, DIRECTIONAL):
   >= 0.9125 -> dispersion/tail pooling helps -> iter39 seed-confirms (5 seeds + seed-avg) + tries the
                unbiased L-scale/GMD pooling form (built if this screen is positive).
   0.9005-0.9125 -> flat; the biased-std pool may be masking a real dispersion signal -> iter39 L-scale.
   <= 0.9005 -> pooling hurts on this base -> dispersion axis tapped for the Transformer; pivot to trees.

 ALSO STILL PENDING FROM iter37 (already generated, 1 upload):
   submissions/submission_champion_replvhsq_swa_seedavg5.csv  -> vs 0.899512 (does SWA buy level?).
 AND re-run once for the fixed MODE-B feature screen table:
   python tools/shift_diagnostics.py --mode screen
=====================================================================
NEXT
