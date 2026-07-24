#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 23 — MULTIVARIATE ROCKET: make the decorrelated member competitive.  *** SCREEN, 0 SUBS ***
#
#   WHERE WE ARE. iter22 proved ROCKET is the first genuinely decorrelated member ever built
#   (rho(rocket,xview)=0.8665; the whole transformer cluster is 0.82-0.87 away vs the 0.93-0.99
#   rank-twins of every in-family variant). BUT rocket is a weaker learner (ATC-F1 -0.040 LB), so the
#   1/5 blend champion_rocketblend5 only tied the cluster (LB 0.8857 ~ the seed-avg consensus). The
#   decorrelation is real; the member is just not strong enough to cash it in.
#
#   THE FIX (mechanism, not just capacity). iter22's ROCKET is UNIVARIATE: each random kernel
#   convolves ONE band, so it can never encode a cross-band signature like "low VH AND low NDVI AND
#   low NDWI" -- the actual pond fingerprint, which the Transformer captures via cross-band attention.
#   MULTIVARIATE kernels span a random SUBSET of the 24 channels and sum the per-band convs before
#   PPV/max pooling (the ROCKET-multivariate recipe). This adds genuine cross-band signal the
#   univariate form structurally lacks -> should raise ATC-F1 (true transfer), not merely OOF. Local
#   smoke already shows OOF-AUC 0.943 -> 0.970 from this one change.
#
#   ISOLATED CHANGE: rocket.multivariate (+ max_channels). false reproduces the iter22 member
#   bit-for-bit (verified locally: identical smoke final_oof/auc).
#
#   DECISION: SCREEN, 0 subs. Paired univariate (c_rocket) vs multivariate (c_rocket_mv) isolates the
#   mechanism. Submit ONLY if c_rocket_mv ATC-F1 clearly > c_rocket AND rho(mv,xview) still < ~0.90
#   (competent AND still decorrelated). Then the blend could finally beat the cluster -- the first
#   LEVEL gain since the GBDT->Transformer swap. Otherwise ROCKET's lane is exhausted: lock
#   champion_archblend4 + champion_rocketblend5 as the diverse finalist pair and pivot to the writeup.
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

# ---- 2. ROCKET: univariate baseline vs MULTIVARIATE, 2 seeds each (paired; DIS + seed-collapse). ----
python run_pipeline.py --full --model rocket --name c_rocket
python run_pipeline.py --full --model rocket --name c_rocket_s7    --set seed=7
python run_pipeline.py --full --model rocket --name c_rocket_mv    --set rocket.multivariate=true
python run_pipeline.py --full --model rocket --name c_rocket_mv_s7 --set rocket.multivariate=true --set seed=7

# ---- 3. Retro-fit gate + seed floor + screen BOTH rocket variants (the paired ATC-F1 read). ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --seed-spread seq_a_xview \
  --screen c_rocket c_rocket_mv

# ---- 4. Go/no-go correlations + candidate blends. ----
# 4a. Leading finalist (4 transformers) — keep current.
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --diag-extra seq_a_k4 seq_a_base \
  --name champion_archblend4
# 4b. iter22 univariate blend — baseline to compare the multivariate blend against.
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview c_rocket \
  --name champion_rocketblend5
# 4c. THE CANDIDATE: multivariate rocket as the 5th member. Its matrix row is the decision --
#     rho(rocket_mv, xview) must stay < ~0.90 for the added strength to still buy diversity.
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview c_rocket_mv \
  --diag-extra c_rocket \
  --name champion_rocketblend5_mv
# 4d. Maximally-diverse 2-way option with the multivariate member.
python tools/arch_blend.py \
  --members seq_a_xview c_rocket_mv \
  --name champion_xview_rocket_mv
# 4e. Seed-collapse the multivariate rocket (standalone diverse hedge) + keep xview seed-avg current.
python tools/seed_average.py --variant c_rocket_mv --name champion_rocket_mv_seedavg2 || true
python tools/seed_average.py --variant seq_a_xview --name champion_seedavg5

cat <<'NEXT'
=====================================================================
 Paste back: (i) the `run: c_rocket` and `run: c_rocket_mv` summary blocks (final_oof, oof_auc),
 (ii) the RETRO-FIT + GATE and the SCREEN lines for BOTH c_rocket and c_rocket_mv, and (iii) the
 arch_blend matrices for 4b (uni) and 4c (mv) -- the 'rocket_mv' row in 4c is the decision.

 THE PAIRED READ: c_rocket_mv vs c_rocket isolates the multivariate (cross-band) mechanism.
   - c_rocket_mv ATC-F1 clearly > c_rocket  AND  rho(rocket_mv, xview) still < ~0.90
        -> the decorrelated member is now COMPETENT. Upload champion_rocketblend5_mv (or
           champion_xview_rocket_mv). A blend that beats the cluster = first LEVEL gain since the
           GBDT->Transformer swap, and the best diverse finalist for the private slice.
   - ATC-F1 flat / within seed sd, OR rho jumps toward the cluster (strength bought by re-correlating)
        -> ROCKET's lane is exhausted. Lock champion_archblend4 (0.8946) + champion_rocketblend5
           (0.8857) as the diverse finalist pair and pivot to the Phase-Two reproducibility/novelty
           writeup (35% of the top-5 rubric).
=====================================================================
NEXT
