#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 15 — MEASURE THE SEED FLOOR, THEN SCREEN AGAINST IT.  *** 0 SUBMISSIONS ***
#
#   WHY THIS CHANGED. iter14 uploaded submission_c_dropout3_S7 (seed 7) rather than the seed-0
#   artifact the screen approved, and scored 0.8675 vs the champion's 0.8955 — a -0.0280 gap that
#   changes TWO variables at once (dropout AND seed). It is therefore uninterpretable as it stands,
#   and it exposes something we have never measured in fifteen iterations: OUR SEED-TO-SEED SPREAD.
#   If that spread is ~0.028, then iter8 (+0.0009), iter9 (+0.0047) and iter10 (-0.0034) are all
#   inside noise and a large part of our ledger needs re-reading.
#
#   THE LB SIDE of that question is settled by uploading submission_seq_a_xview_s7.csv (champion,
#   seed 7, nothing else changed) — see the run instructions at the bottom.
#
#   THE OFFLINE SIDE is this run: generate the champion at FIVE seeds and report how much each
#   estimator moves across seeds of an IDENTICAL config. That gives the screen a NOISE FLOOR.
#   iter13 approved dropout=0.3 on a DIS margin of just +0.0029; if DIS moves more than that
#   between two seeds of the same config, that vote never carried information. offline_validate.py
#   now marks any margin inside the seed floor with '~' and DOWNGRADES it to HOLD regardless of
#   vote count. This is the "margin condition" the round-07 math audit asked for.
#
#   NEW CANDIDATE — c_meanmin, the tail we never tested. Our mean_max probe followed the
#   "ponds are never bright" framing, but the actual pond-MAPPING literature (Ottinger 2017;
#   nation-scale S1 work) detects ponds with a LOW-order statistic — the temporal MEDIAN / p10-p25
#   of VH — because a pond is a "permanent low scatterer" and low percentiles are robust to speckle
#   and to which months happen to be observed. We tested the wrong tail. See RESEARCH_08_EY.md.
#
#   PRE-COMMITTED RULE (now with the noise floor): >=2 cleared estimators above champion AND every
#   contributing margin outside the seed floor -> submit. Otherwise HOLD.
#
#   Committed config remains the exact 0.8955 champion; every variant comes from --set.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- Anchors: re-certify the estimators against THIS code every run. ----
PRE="--set seq.relative_time=false --set seq.consistency_lambda=0"
python run_pipeline.py $COMMON --name seq_a_detrend $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4      $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_base    $PRE
python run_pipeline.py $COMMON --name seq_a_reltime --set seq.consistency_lambda=0
python run_pipeline.py $COMMON --name seq_a_nope    --set seq.consistency_lambda=0 --set seq.pos_encoding=none
python run_pipeline.py $COMMON --name seq_a_l3      --set seq.consistency_lambda=3
python run_pipeline.py $COMMON --name seq_a_xview                                   # CHAMPION, LB 0.8955
python run_pipeline.py $COMMON --name seq_a_detrend_s7 --set seed=7 $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4_s7      --set seed=7 $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_reltime_s7 --set seed=7 --set seq.consistency_lambda=0

# ---- SEED FLOOR: the champion at five seeds. seq_a_xview_s7 is ALSO the file to upload. ----
for SD in 7 13 21 29; do
  python run_pipeline.py $COMMON --name "seq_a_xview_s${SD}" --set seed=$SD
done

# ---- CANDIDATES (2 seeds each so DIS is computable) ----
for S in "" "_s7"; do
  SEED=""; [ -n "$S" ] && SEED="--set seed=7"
  python run_pipeline.py $COMMON --name "c_meanmin$S"  $SEED --set seq.pooling=mean_min
  python run_pipeline.py $COMMON --name "c_dropout3$S" $SEED --set seq.dropout=0.3
  python run_pipeline.py $COMMON --name "c_do40$S"     $SEED --set seq.dropout=0.4
  python run_pipeline.py $COMMON --name "c_wd3$S"      $SEED --set seq.weight_decay=1.0e-3
done

# ---- Retro-fit + SEED FLOOR + SCREEN ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --seed-spread seq_a_xview \
  --screen c_meanmin c_dropout3 c_do40 c_wd3

cat <<'NEXT'
=====================================================================
 Paste back: the RETRO-FIT table, the GATE lines, the SEED SPREAD block, and the SCREEN table.

 SEPARATELY AND MOST IMPORTANTLY -- upload ONE file to Zindi:

     submissions/submission_seq_a_xview_s7.csv

 That is the CHAMPION configuration at seed 7. It changes ONLY the seed, so it isolates the
 variable iter14 confounded:
     ~0.895  -> seed variance is small, so dropout 0.3 genuinely failed and the screen produced
                a false positive on a margin that was too thin.
     ~0.867  -> seed variance is ~0.028, dropout is exonerated, and much of our ledger of
                +-0.003 to +-0.013 "effects" has to be re-read as noise.
 Either answer is worth far more than one submission. Paste the LB back.
=====================================================================
NEXT
