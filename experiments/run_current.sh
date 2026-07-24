#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 22 — A DIFFERENT MODEL CLASS: ROCKET random-convolution member.  *** GO/NO-GO, 0 SUBS ***
#
#   WHY. iter18-21 measured every in-family lane CLOSED. Positional reframes, the cross-view
#   objective, pooling variants (mean_min a rank-TWIN at rho=0.9928), and instance-expansion all sit
#   at rank-correlation 0.93-0.99 with the champion, so they can only buy VARIANCE, never LEVEL. The
#   ONLY species of change that ever cleared the ~0.010 LB floor was a *different model class*
#   (GBDT->Transformer, +0.05). ROCKET is the one remaining move of that species.
#
#   WHAT. A large bank of RANDOM convolutional kernels over the 12-month sequence, each summarized by
#   PPV + max, then a plain linear classifier. No attention, no learned representation -> decorrelated
#   from the Transformer BY CONSTRUCTION. Pure numpy + sklearn, no new dependency, trained only on
#   competition data -> rules-safe exactly like the from-scratch Transformer. It reuses the champion's
#   masking augmentation (K/R) and to_inputs representation, so the ONLY difference is the estimator.
#
#   THE GO/NO-GO IS FREE. tools/arch_blend.py prints the cross-model rank-correlation. This run spends
#   ZERO submissions; it produces the correlation + the offline screen + candidate blends, and I pick
#   ONE csv to upload from the numbers you paste back.
#     rho(rocket, xview) < ~0.90  -> genuinely decorrelated. IF rocket is also competent (ATC-F1 not
#                                    far below champion) the blend buys real private-slice variance
#                                    reduction and is the best DIVERSE finalist we can build.
#     rho >= ~0.94                -> even a foreign model class ranks these rows the same way -> the
#                                    architecture search is genuinely FINISHED. Lock archblend4 and
#                                    move to the Phase-Two reproducibility/novelty writeup.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- 1. Anchors: re-certify the estimators (7 known-LB variants) + the champion seed spread. ----
#         Identical to iter21 section 1 so the retro-fit gate + seed floor stay valid this run.
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

# ---- 2. THE NEW MODEL CLASS. ROCKET at 2 seeds (for DIS + a seed-collapsed finalist). ----
python run_pipeline.py --full --model rocket --name c_rocket
python run_pipeline.py --full --model rocket --name c_rocket_s7 --set seed=7

# ---- 3. Retro-fit gate + seed floor + screen the rocket member. ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --seed-spread seq_a_xview \
  --screen c_rocket

# ---- 4. THE GO/NO-GO + finalists. ----
# 4a. Keep the leading finalist (4 transformers) current AND print the transformer-cluster rho.
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --diag-extra seq_a_k4 seq_a_base \
  --name champion_archblend4
# 4b. THE TEST: add ROCKET as a 5th member. The matrix row for 'rocket' is the go/no-go -- read
#     rho(rocket, xview) and rho(rocket, <each transformer>). This blend is the candidate finalist.
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview c_rocket \
  --diag-extra seq_a_k4 \
  --name champion_rocketblend5
# 4c. A maximally-diverse 2-way finalist option (champion + the foreign model class, equal weight).
python tools/arch_blend.py \
  --members seq_a_xview c_rocket \
  --name champion_xview_rocket
# 4d. Seed-collapse rocket on its own (a standalone diverse finalist candidate).
python tools/seed_average.py --variant c_rocket --name champion_rocket_seedavg2 || true
python tools/seed_average.py --variant seq_a_xview --name champion_seedavg5

cat <<'NEXT'
=====================================================================
 Paste back: (i) the two ROCKET fold lines + its `run: c_rocket` summary block (final_oof, auc),
 (ii) the RETRO-FIT + GATE and the SCREEN line for c_rocket, and (iii) BOTH arch_blend correlation
 matrices (4a and 4b) -- the 'rocket' row in 4b is the whole decision.

 THE DECISION (I make the single upload call from your paste; the run itself spent 0 submissions):
   - rho(rocket, xview) < ~0.90  AND  rocket ATC-F1 within ~0.05 of champion
        -> DECORRELATED + COMPETENT. First real ensemble-LEVEL artifact since the Transformer swap.
           Upload champion_rocketblend5 (or champion_xview_rocket) -- it also becomes the best
           low-variance DIVERSE finalist for the unseen 721-row private slice.
   - rho < ~0.90 BUT rocket ATC-F1 far below champion
        -> decorrelated but weak; equal-weight blend would drag. Down-weight / hold; report only.
   - rho >= ~0.94
        -> even a foreign model class ranks these rows the same. The architecture search is FINISHED.
           Lock champion_archblend4 (0.8946) as the leading finalist and pivot to the Phase-Two
           reproducibility + novelty writeup (35% of the top-5 rubric).
=====================================================================
NEXT
