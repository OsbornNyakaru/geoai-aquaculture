#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 16 — THE PRESTO LANE.  *** 0 SUBMISSIONS FROM THIS RUN ***
#
#   WHY THIS AND NOTHING ELSE. iter15 closed the measurement question with two independent
#   estimates that agree: the screen resolves ~0.010-0.013 LB. In the entire project only TWO
#   effects have ever exceeded that floor --
#         GBDT -> Transformer swap  +0.0500
#         per-cell detrend          -0.0514
#   Both are MODEL-CLASS changes. Every architectural tweak, loss term, pooling variant,
#   positional reframe and regularization knob we probed is BELOW the floor and unmeasurable in
#   principle with our budget. Running more of them cannot produce information.
#
#   iter16 also confirmed the variance story: the 5-seed average scored 0.8865 against a
#   single-seed mean of 0.8859 -- i.e. we bought VARIANCE REDUCTION, not a level gain, exactly as
#   predicted, because the seeds are 95.1% rank-correlated so only ~5% of the error is independent.
#
#   So there is one fundable architectural direction left, and the rules permit it:
#       "You may use pretrained models as long as they are openly available to everyone."
#   Presto is a ~0.4M-param transformer pretrained with masked-modality SSL on 21.5M Sentinel-1/2
#   PIXEL TIME SERIES -- our exact data shape, and its pretraining objective is literally our
#   central difficulty. FROZEN, the fitted model is a ~129-parameter logistic head: LESS fitted
#   capacity than anything we have shipped, so it does not contradict the capacity law.
#
#   TWO THINGS TO WATCH IN THE LOG:
#     1. ADVERSARIAL AUC on the embeddings. This is the go/no-go and it costs nothing:
#          ~0.5  -> the frozen encoder NORMALIZED THE TEMPORAL SHIFT AWAY. Very promising.
#          >0.9  -> Presto is ENCODING the shift and the head will latch onto it. Expect failure.
#     2. month=const vs month=true. month=const deletes absolute calendar identity and keeps only
#        relative step -- our relative-time reframing applied to Presto. It is PRIMARY. The
#        research pass measured the month argument as a first-order lever (rank corr 0.46), so
#        these are two genuinely different models, not a tweak.
#
#   Runtime: Presto inference is ~2.4 s for all 2,851 rows ON CPU. The expensive part is the
#   anchor regeneration for the retro-fit, as usual.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- 0. Vendor Presto (MIT source + 3.3 MB checkpoint), patched to import standalone. ----
pip install -q einops
python tools/fetch_presto.py

# ---- 1. Anchors: re-certify the estimators against THIS code. ----
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

# ---- 2. The Presto lane. Both month modes, 2 seeds each (head seed only; encoder is frozen). ----
# NOTE: train rows are pushed through the SAME masking-window sampler as always, so the encoder
# sees train and test at the same observation density. Skipping that would manufacture a domain
# gap of our own making -- the second most likely way this lane fails.
for MM in const true; do
  for SD in 42 7; do
    SUF=""; [ "$SD" != "42" ] && SUF="_s${SD}"
    python run_presto.py --month-mode "$MM" --seed "$SD" --name "c_presto_${MM}${SUF}"
  done
done

# ---- 3. Retro-fit + seed floor + screen ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --seed-spread seq_a_xview \
  --screen c_presto_const c_presto_true

# ---- 4. Keep the variance-reduction artifact current. ----
python tools/seed_average.py --variant seq_a_xview --name champion_seedavg5

cat <<'NEXT'
=====================================================================
 Paste back: the ADVERSARIAL AUC lines from the Presto runs (the go/no-go), the RETRO-FIT + GATE,
 the SEED SPREAD block, and the SCREEN table.

 NO UPLOAD from this run unless the screen says SUBMIT.

 How to read it:
   adversarial AUC ~0.5 and a positive screen  -> Presto normalized the shift away. Fund it hard.
   adversarial AUC >0.9                        -> Presto is encoding the shift; lane likely dead,
                                                  and we will have learned that for 0 submissions.
   screen HOLD but LB-equiv margin > +0.010    -> worth one submission anyway; that is the only
                                                  band where our instrument can actually resolve.
=====================================================================
NEXT
