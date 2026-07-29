#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 31 — THE LAST FEATURE SHOT, IN THE MODEL THAT TRANSFERS.   *** 1 SUBMISSION ***
#
#   WHY. iter30 closed the tree/CatBoost lane hard (standalone 0.698, blend −0.0136): trees do not
#   transfer here. The ensemble lane is fully closed (n=3 cross-class fails + iter29 within-class).
#   The ONE feature thesis never tested IN THE TRANSFORMER — the model that actually transfers
#   (0.8897 legal) — is the VH permanence profile + the cross-pol ratio. RESPONSE_13 called the
#   permanence indicator "the best feature idea of the round".
#
#   WHAT. Two new INPUT CHANNELS on the champion (relative_time + consistency_lambda=1):
#     - permanence: per-month 1[VH_dB(t) < tau] for tau in {-22,-21,-20,-19}. The masked mean-pool of
#       a binary channel IS the fraction-of-months-below-tau = the VH-CDF permanence profile. Class-A
#       (n-invariant), amplitude-PRESERVING, and an indicator is NOT affine-spanned (so it escapes the
#       dead water-index / VH-VV degeneracy that a linear proj already covers).
#     - cross_pol: VH-VV (dB), the rice-canopy axis (UPDATE_13 3a), ADDED alongside level.
#   Width 24 -> 29. Smoke-verified legal; default-off reproduces the champion at width 24.
#
#   HONEST FRAMING (committed BEFORE the number). This is a REPRESENTATION change, so ATC-F1 is
#   out-of-family and screens NOTHING (that instrument mispredicted c_dropvv and c_catboost — my 3
#   strikes were all here). RESPONSE_13 itself says "expect small" (may be redundant with what the
#   encoder already extracts from standardized VH). The matched-seed A/B cancels the 0.019 seed noise,
#   leaving ~0.013 public sampling noise, so only a clear effect is even detectable.
#     c_permxpol − xview (seed 42) >= +0.013  -> REAL: the permanence feature is the first feature win
#                                               in the transformer; keep it, seed-pool it, new finalist.
#     within +-0.013                          -> the feature-in-transformer thesis is closed too. The
#                                               score is ceilinged; archblend4 (0.899643) is final and
#                                               ALL remaining effort goes to the Phase-Two writeup.
#   Either way, this is the LAST score probe. The writeup + finalist designation start now regardless.
# =====================================================================
set -euo pipefail

COMMON="--full --model seq"

# ---- Matched-seed A/B: champion vs champion+features (seed 42 cancels the seed-noise component). ----
echo "=== CONTROL: seq_a_xview (legal champion, expect LB ~0.8897) ==="
python run_pipeline.py $COMMON --name seq_a_xview

echo "=== CANDIDATE: c_permxpol = champion + permanence + cross_pol channels (width 24->29) ==="
python run_pipeline.py $COMMON --name c_permxpol \
  --set seq.channels.permanence=true --set seq.channels.cross_pol=true

echo "=== ISOLATE: c_perm = champion + permanence ONLY (attributes the effect if the candidate moves) ==="
python run_pipeline.py $COMMON --name c_perm \
  --set seq.channels.permanence=true

# Second seed of the candidate + control, so a win can be seed-confirmed (not another lucky draw).
python run_pipeline.py $COMMON --name seq_a_xview_s7 --set seed=7
python run_pipeline.py $COMMON --name c_permxpol_s7 --set seed=7 \
  --set seq.channels.permanence=true --set seq.channels.cross_pol=true

cat <<'NEXT'
=====================================================================
 PASTE BACK: the `run:` summary blocks (final_oof, oof_auc, realized test pos-rate) for
   seq_a_xview, c_permxpol, c_perm, seq_a_xview_s7, c_permxpol_s7.

 THEN UPLOAD ONE FILE:
   submissions/submission_c_permxpol.csv   -> compare against submission_seq_a_xview.csv (same run,
                                              same seed 42 -> the seed-noise component cancels).

 THE READ, COMMITTED IN ADVANCE (seed-paired resolution ~0.013 on the public slice):
   c_permxpol − seq_a_xview >= +0.013 AND the seed-7 pair agrees in sign
        -> REAL. First feature win inside the transformer. Seed-pool it, designate as a finalist,
           and the ratio/permanence feature family is worth one more iteration.
   within +-0.013 (the overwhelmingly likely outcome per RESPONSE_13's "expect small")
        -> the feature-in-transformer lane is closed. The LB score is ceilinged at ~0.90. archblend4
           (0.899643) + seq_a_xview (0.889686) are the final two finalists. STOP probing; everything
           left is the Phase-Two reproducibility/novelty writeup (35% of a top-5 score) + designating
           the finalists on Zindi before 2026-08-16.

 (The writeup + designation work begins in parallel regardless of this number.)
=====================================================================
NEXT
