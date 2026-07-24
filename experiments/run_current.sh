#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 24 — GBDT AS THE DECORRELATED MEMBER (the last untested model class).  *** SCREEN, 0 SUBS ***
#
#   WHY (and why NOT ROCKET). iter22-23 settled ROCKET: it decorrelates (ρ0.87) ONLY because it is
#   blind to cross-band structure; teaching it that structure (multivariate) makes it stronger but
#   re-correlates it to ρ0.91 -- within this family decorrelation and competence trade off. A GBDT is
#   the opposite kind of learner: LGBM+XGB+CatBoost on the hand-engineered temporal-AGGREGATE features.
#   It USES cross-band/cross-feature structure natively (so it is NOT weak-by-blindness like ROCKET),
#   but it discards the temporal ORDERING the Transformer attends to -- so its diversity comes from a
#   genuinely different REPRESENTATION, the one profile that could be decorrelated AND competent.
#
#   PRIOR EVIDENCE. iter2 (old pipeline) measured GBDT ρ≈0.849 to seq (decorrelated) at LB 0.8780 --
#   only −0.0175 below today's champion (≈ one seed swing, so that gap is largely VOID). Strictly a
#   better decorrelated-member profile than ROCKET's −0.040. Its one prior blend (iter2, 0.8705)
#   predates the seed-variance understanding, the prevalence pin, and the clean two-level rank-blend,
#   so it is void and worth re-measuring properly. GBDT smoke re-verified locally: valid submission +
#   arch_blend-ready preds bundle.
#
#   DECISION: SCREEN, 0 subs. This is the FINAL architecture screen either way.
#     ρ(gbdt,xview) < ~0.90  AND  gbdt ATC-F1 close to champion  -> decorrelated AND competent (what
#       ROCKET could not be) -> upload champion_gbdtblend5; a blend beating the cluster = first LEVEL
#       gain since the GBDT->Transformer swap.
#     ρ >= ~0.90  OR  gbdt ATC-F1 far below champion -> the last model class is closed -> lock the
#       finalist pair (archblend4 + rocketblend5) and pivot to the Phase-Two reproducibility/novelty
#       writeup (35% of the top-5 rubric).
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

# ---- 2. THE LAST MODEL CLASS. GBDT ensemble at 2 seeds (for DIS + a seed-collapsed finalist). ----
python run_pipeline.py --full --model gbdt --name g_gbdt
python run_pipeline.py --full --model gbdt --name g_gbdt_s7 --set seed=7

# ---- 3. Retro-fit gate + seed floor + screen the GBDT member. ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --seed-spread seq_a_xview \
  --screen g_gbdt

# ---- 4. Go/no-go correlations + candidate blends. ----
# 4a. Leading finalist (4 transformers) — keep current.
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --diag-extra seq_a_k4 seq_a_base \
  --name champion_archblend4
# 4b. THE CANDIDATE: GBDT as the 5th member. Its matrix row (rho to xview / cluster) is the decision.
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview g_gbdt \
  --name champion_gbdtblend5
# 4c. Maximally-diverse 2-way option: champion + GBDT (the two strongest, most-different learners).
python tools/arch_blend.py \
  --members seq_a_xview g_gbdt \
  --name champion_xview_gbdt
# 4d. Seed-collapse GBDT (standalone diverse hedge) + keep xview seed-avg current.
python tools/seed_average.py --variant g_gbdt --name champion_gbdt_seedavg2 || true
python tools/seed_average.py --variant seq_a_xview --name champion_seedavg5

cat <<'NEXT'
=====================================================================
 Paste back: (i) the `run: g_gbdt` summary (final_oof, oof_auc), (ii) the RETRO-FIT + GATE and the
 SCREEN line for g_gbdt, and (iii) the 4b arch_blend matrix -- the 'gbdt' row is the whole decision.

 THE DECISION (final architecture screen; I make the single upload call from your paste, 0 subs spent):
   - rho(gbdt, xview) < ~0.90  AND  gbdt ATC-F1 close to champion (within ~0.02 LB)
        -> DECORRELATED + COMPETENT, the profile ROCKET could not reach. Upload champion_gbdtblend5
           (or champion_xview_gbdt). A blend that beats the cluster = first LEVEL gain since the
           GBDT->Transformer swap, and the strongest diverse finalist for the private slice.
   - rho >= ~0.90  OR  gbdt ATC-F1 far below champion
        -> the last model class is closed. The architecture search is DONE. Lock champion_archblend4
           (0.8946) + champion_rocketblend5 (0.8857) as the diverse finalist pair and pivot to the
           Phase-Two reproducibility + novelty writeup (35% of the top-5 rubric).
=====================================================================
NEXT
