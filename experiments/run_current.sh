#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 21 — INSTANCE-EXPANSION via PER-EPOCH VIEW RESAMPLING.  *** SCREEN FIRST, 0 SUBS ***
#
#   WHY. iter20 closed the pooling-ensemble lane: mean_min is a rank-TWIN of the champion
#   (rho=0.9928), so every mean/min transformer variant we can build sits at rho~0.93-0.99 and
#   cannot buy LEVEL. Only a DATA-MODEL or model-class change can move the needle now. The
#   cross-examination's #1 lever is instance-expansion.
#
#   CODE-REVIEW FINDING (important, and it re-scopes the idea). Reading src/seq_model.py: when
#   consistency_lambda=0 our K masked views are ALREADY trained as independent BCE examples. So
#   `seq_a_k4` (K=4, lam=0) was already a scaled instance-expansion and scored 0.8665 -- a fixed set
#   of K views does not help. The genuinely UNTESTED and strongest form is PER-EPOCH RESAMPLING:
#   draw fresh masked windows for every row every epoch, so the model trains on ~K*epochs DISTINCT
#   (row, sub-window) instances instead of K fixed ones -- the real "multiply each row into many
#   masked sub-windows matched to the test masking".
#
#   CLEAN PAIRED CONTROL. seq_a_reltime (K=2, reltime ON, lam=0, FIXED views) = 0.8908 is the exact
#   fixed-view twin of c_iexp_rs2 (K=2, reltime ON, lam=0, RESAMPLED). Their ATC-F1 gap isolates the
#   resampling mechanism ALONE, nothing else changed.
#
#   ISOLATED CHANGE: seq.resample_per_epoch (+ K for the scaled arms). reltime ON, lam=0 throughout
#   (resampling is only wired on the independent-view path; coupling needs fixed owners). Flag OFF
#   reproduces the champion bit-for-bit.
#
#   DECISION: SCREEN. Trust ATC-F1's SIGN (its magnitude over-predicted mean_min, so discount that).
#   Submit only if >=2 cleared estimators beat champion AND the margin exceeds the seed sd. Most
#   likely per the k4 evidence this HOLDs -> then iter22 = a decorrelated MODEL-CLASS member
#   (MiniRocket/CropNet), the cross-exam's #2. A positive here would be the first data-model gain
#   since the GBDT->Transformer swap.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- 1. Anchors: re-certify estimators; seq_a_reltime is the paired fixed-view control. ----
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

# ---- 2. INSTANCE-EXPANSION. reltime ON, lam=0, per-epoch resampling. 2 seeds each for DIS. ----
IEXP="--set seq.relative_time=true --set seq.consistency_lambda=0 --set seq.resample_per_epoch=true"
# rs2: K=2 resampled -- the PAIRED test vs seq_a_reltime (K=2 fixed). Isolates resampling alone.
python run_pipeline.py $COMMON --name c_iexp_rs2     $IEXP --set seq.K=2
python run_pipeline.py $COMMON --name c_iexp_rs2_s7  $IEXP --set seq.K=2 --set seed=7
# rs6: K=6 resampled -- more distinct instances/epoch; tests whether scale helps once resampled.
python run_pipeline.py $COMMON --name c_iexp_rs6     $IEXP --set seq.K=6
python run_pipeline.py $COMMON --name c_iexp_rs6_s7  $IEXP --set seq.K=6 --set seed=7

# ---- 3. Retro-fit gate + seed floor + screen. ----
python tools/offline_validate.py \
  --preds-dir submissions/preds --anchors experiments/anchors.tsv \
  --champion seq_a_xview \
  --seed-spread seq_a_xview \
  --screen c_iexp_rs2 c_iexp_rs6

# ---- 4. Keep the leading finalists current. ----
python tools/arch_blend.py \
  --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
  --diag-extra seq_a_k4 seq_a_base \
  --name champion_archblend4
python tools/seed_average.py --variant seq_a_xview --name champion_seedavg5

cat <<'NEXT'
=====================================================================
 Paste back: the RETRO-FIT + GATE, the SEED SPREAD block, and the SCREEN lines for
 c_iexp_rs2 / c_iexp_rs6. Also note the per-fold OOF -- instance-expansion may LOWER OOF while
 (hopefully) improving transfer; ATC-F1 is the arbiter, not OOF.

 THE KEY PAIRED READ: c_iexp_rs2 vs the seq_a_reltime anchor (0.8908) isolates per-epoch resampling.
   - c_iexp_rs2 ATC-F1 clearly > seq_a_reltime, and >=2 estimators beat champion, margin > seed sd
        -> SUBMIT c_iexp_rs2 (or rs6 if stronger). First data-model gain since the Transformer swap.
   - within noise / HOLD
        -> instance-expansion via resampling is inert too; pivot to iter22 = a decorrelated
           MODEL-CLASS member (MiniRocket/CropNet), the cross-exam's #2.
=====================================================================
NEXT
