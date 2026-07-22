#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 14 — REGULARIZATION SWEEP.  *** COSTS ZERO SUBMISSIONS ***
#   Run this WHILE the c_dropout3 submission is in flight on Zindi. The two are independent:
#   the upload measures dropout 0.3 on the real leaderboard, this screen maps the shape of the
#   whole regularization axis offline. No reason to serialize them.
#
#   WHY: iter13 produced our FIRST-EVER screen SUBMIT. c_dropout3 (dropout 0.2 -> 0.3) cleared
#   both certified estimators (ATC-F1 +0.0165, DIS +0.0029). It is EXACTLY parameter-neutral --
#   pure regularization strength -- and it is the most on-thesis knob in the repo under our own
#   design law ("less fit transfers better"), yet it had never been touched once in twelve
#   iterations. We spent those twelve on architecture while the plainest knob sat at its default.
#   That is the lesson worth acting on: sweep the axis properly rather than stopping at one point.
#
#   Caveat carried forward: the DIS margin was TINY (+0.0029). This is a 2/2 by the rule, not a
#   resounding one, and estimator deltas are NOT on the LB scale.
#
#   CANDIDATES (all exactly parameter-neutral; 2 seeds each so DIS is computable):
#     c_do35 / c_do40 / c_do50   dropout 0.35 / 0.40 / 0.50 -- is 0.3 a slope or a peak?
#     c_wd3                      weight_decay 1e-4 -> 1e-3, the OTHER untouched regularizer
#     c_ep40                     epochs 60 -> 40, i.e. regularize by early stopping instead
#   If the dropout curve keeps rising to 0.5, the real finding is "we were badly under-regularized
#   all along" and the architecture work was fighting the wrong problem. If it peaks at 0.3-0.35,
#   we take the peak and move on to the Presto lane.
#
#   PRE-COMMITTED RULE (unchanged): >=2 cleared estimators above champion -> submit.
#   1 or 0 -> HOLD. Do not renegotiate after seeing numbers.
#
#   Committed config remains the exact 0.8955 champion; every variant comes from --set.
#   ~21 runs, roughly 20 minutes.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- Anchors: re-certify the estimators against THIS code every run. ----
# If ATC-F1 or DIS stops clearing, something regressed and THE SCREEN BELOW IS VOID.
PRE="--set seq.relative_time=false --set seq.consistency_lambda=0"
python run_pipeline.py $COMMON --name seq_a_detrend $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4      $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_base    $PRE
python run_pipeline.py $COMMON --name seq_a_reltime --set seq.consistency_lambda=0
python run_pipeline.py $COMMON --name seq_a_nope    --set seq.consistency_lambda=0 --set seq.pos_encoding=none
python run_pipeline.py $COMMON --name seq_a_l3      --set seq.consistency_lambda=3
python run_pipeline.py $COMMON --name seq_a_xview                                   # CHAMPION, LB 0.8955

# Second seeds for the four gate variants -> makes DIS scoreable.
python run_pipeline.py $COMMON --name seq_a_xview_s7   --set seed=7
python run_pipeline.py $COMMON --name seq_a_detrend_s7 --set seed=7 $PRE --set seq.channels.per_cell_detrend=true
python run_pipeline.py $COMMON --name seq_a_k4_s7      --set seed=7 $PRE --set seq.K=4
python run_pipeline.py $COMMON --name seq_a_reltime_s7 --set seed=7 --set seq.consistency_lambda=0

# ---- CANDIDATES (2 seeds each) ----
# c_dropout3 is regenerated so the sweep has its 2/2 winner on the same axis for comparison.
for S in "" "_s7"; do
  SEED=""; [ -n "$S" ] && SEED="--set seed=7"
  python run_pipeline.py $COMMON --name "c_dropout3$S" $SEED --set seq.dropout=0.3
  python run_pipeline.py $COMMON --name "c_do35$S"     $SEED --set seq.dropout=0.35
  python run_pipeline.py $COMMON --name "c_do40$S"     $SEED --set seq.dropout=0.4
  python run_pipeline.py $COMMON --name "c_do50$S"     $SEED --set seq.dropout=0.5
  python run_pipeline.py $COMMON --name "c_wd3$S"      $SEED --set seq.weight_decay=1.0e-3
  python run_pipeline.py $COMMON --name "c_ep40$S"     $SEED --set seq.epochs=40
done

# ---- Retro-fit (re-certify) + SCREEN ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --screen c_dropout3 c_do35 c_do40 c_do50 c_wd3 c_ep40

echo "=== done. NO UPLOAD FROM THIS RUN. Paste back the RETRO-FIT + GATE + SCREEN tables. ==="
echo "=== SEPARATELY: upload submissions/submission_c_dropout3.csv to Zindi and paste the LB. ==="
