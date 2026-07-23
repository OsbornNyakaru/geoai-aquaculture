#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 18 — THE GRAND ENSEMBLE (cross-architecture rank-blend).  *** AT MOST 1 SUBMISSION ***
#
#   WHY THIS. iter17 killed the Presto lane for 0 submissions: the adversarial AUC on its frozen
#   embeddings came back 0.965-0.976 (>0.9 = the encoder ENCODES the designed temporal shift rather
#   than normalizing it), and ATC-F1 put it 0.044-0.059 LB BELOW champion. That closes the
#   foundation-model / model-class frontier. But it also proved, on a general-purpose representation
#   that never saw our labels, that the train->test shift is REAL and LARGE -- so the ~0.975 OOF vs
#   ~0.89 LB gap is mostly irreducible covariate shift, and our champion already carries the right
#   response (masking views + relative time + cross-view invariance are shift-invariance machinery).
#
#   The seed-average bought VARIANCE reduction but no LEVEL (0.8865 == the single-seed mean 0.8859),
#   because seeds are 95.1% rank-correlated. The one remaining cheap shot at LEVEL is to pool across
#   DIFFERENT ARCHITECTURES, which may be decorrelated where seeds are not. The top cluster is
#   statistically tied on the LB but built from genuinely different inductive biases:
#         reltime 0.8908 | nope 0.8917 | l3 0.8921 | xview 0.8955
#
#   THE GO/NO-GO IS FREE AND PRINTED FIRST -- the CROSS-ARCHITECTURE RANK-CORRELATION MATRIX:
#         mean rho ~ 0.95 (like seeds) -> the blend behaves like the seed-average; NO level gain.
#                                          Do not upload. Pivot to pseudo-labeling / ROCKET.
#         mean rho < ~0.90             -> members carry independent signal; pooling gains level with
#                                          bounded downside (the blend lands between its members).
#                                          Upload submission_champion_archblend4.csv.
#
#   DO NOT read the SCREEN for the decision here: ATC-F1's own seed sd is 0.0576 (== +-0.0094 LB),
#   coarser than any ensemble gain by construction, so the screen will HOLD regardless. The
#   correlation matrix is the instrument with resolution for this question.
#
#   Runtime: dominated by the anchor regeneration (~5 min). The blend itself is ~1 s.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- 1. Anchors: re-certify the estimators against THIS code AND give the blend fresh members. ----
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

# ---- 2. THE GRAND ENSEMBLE. Two-level rank-average (pool seeds within each architecture, then ----
#         pool the four architectures with EQUAL weight). Prints the correlation matrix first.
#         diag-extra shows k4 and base in the matrix WITHOUT pooling them (they are below the
#         cluster on the LB; we only want to see whether they are decorrelated).
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --diag-extra seq_a_k4 seq_a_base \
  --name champion_archblend4

# ---- 3. Retro-fit gate + seed floor + screen the blend (INFORMATIONAL; decision is the matrix). ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --seed-spread seq_a_xview \
  --screen champion_archblend4

# ---- 4. Keep the seed-average finalist current as the fallback artifact. ----
python tools/seed_average.py --variant seq_a_xview --name champion_seedavg5

cat <<'NEXT'
=====================================================================
 Paste back: the CROSS-ARCHITECTURE RANK CORRELATION matrix (the go/no-go), the RETRO-FIT + GATE,
 the SEED SPREAD block, and the SCREEN line for champion_archblend4.

 THE DECISION IS THE CORRELATION MATRIX, NOT THE SCREEN:
   mean rho < ~0.90  -> UPLOAD submission_champion_archblend4.csv (level gain available; bounded
                        downside; same variance-reduction category as the seed-average that already
                        validated at 0.8865). This is the at-most-one submission from this run.
   mean rho ~ 0.95   -> DO NOT upload; the blend behaves like the seed-average. Pivot next iter to
                        pseudo-labeling (transductive shift adaptation) or a ROCKET model class.
   0.90 <= rho < 0.94 -> marginal; upload only if you want to spend 1 of ~80 on a small expected gain.
=====================================================================
NEXT
