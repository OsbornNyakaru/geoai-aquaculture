#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# ITERATION 47 — THE PRESTO LANE, REOPENED AND ACTUALLY SUBMITTED.
#
#   WHY THIS IS BEING REOPENED, AND WHY THAT IS NOT A RENEGOTIATION.
#   iter17 built the entire Presto lane (run_presto.py, src/presto_features.py,
#   tools/fetch_presto.py) and then killed it for ZERO submissions using three instruments:
#       - adversarial AUC on the embeddings   -> RETIRED at round 18, which recorded that
#                                                adv-AUC as a selection criterion is "DEAD ...
#                                                BACKWARDS"
#       - ATC-F1                              -> RETIRED at iter25/26, proven invalid OUT OF
#                                                FAMILY (rho +0.964 in-family -> +0.738 out), and
#                                                frozen Presto embeddings are the most
#                                                out-of-family candidate we have ever screened
#       - OOF                                 -> blind by standing rule (OOF ~0.975 for everything
#                                                while the LB spans 0.72-0.907)
#   All three have since been retired BY THIS PROJECT. So the Presto kill rests entirely on
#   evidence we have ourselves invalidated, and Presto has never been submitted even once.
#   Reopening a lane whose only evidence has been withdrawn is not renegotiation; leaving it
#   closed on retracted evidence would be the error.
#
#   ROUND-22 RESEARCH CONFIRMED THE SHAPE FITS (gemini_loop/findings/round22_pretrained_models.md):
#       - construct_single_presto_input is real, in presto/dataops/utils.py -- the official
#         partial-band entry point. Verified by reading the code, not inferred.
#       - our 12 bands (VH VV blue green nir nira re1 re2 re3 red swir1 swir2) map 1:1 onto
#         Presto's S1+S2 slots. We are missing only B9, which Presto itself drops.
#       - the mask_tokens uniform-mask assert exists ONLY in single_file_presto.py; presto/presto.py
#         uses attn_mask and handles our variable 4/5/6-month rows correctly. tools/fetch_presto.py
#         already vendors the correct one, and says so in its docstring.
#       - lat/lon is a mandatory ARG but the paper itself runs S2-Agri100 from a single shared
#         location and reports Presto "remained performant". We pass zeros.
#       - month=0 is supported; absolute calendar identity is not required.
#
#   THE ONE THING THAT WAS NEVER TRIED: FINE-TUNING. iter17 only ever ran the FROZEN encoder with a
#   ~129-parameter logistic head. PROJECT_STATE flags this gap itself. run_presto.py now has a
#   --finetune path that unfreezes all 404,160 encoder parameters and trains end-to-end.
#
#   ⚠️ WHAT THIS PROJECT'S OWN MEASURED LAW PREDICTS. "Added capacity fitted to our 1,817 shifted
#   rows hurts" is one of the most reliable findings in this ledger. Frozen Presto fits ~129
#   parameters and therefore never tested that law -- its capacity is amortized over a global
#   pretraining corpus. Fine-tuning fits 404k parameters ONTO the shifted rows, which is exactly
#   the condition the law was measured under. So the honest prior is that the fine-tuned arm LOSES
#   to the frozen arm. We are running it because it is the untried arm, not because we expect it to
#   win. Say so now, before the numbers, so we cannot claim afterwards that we expected either.
#
#   TWO ARMS, TWO UPLOADS, ONE VARIABLE BETWEEN THEM. Both go through the identical legal operating
#   point (calibrate_legal: Platt on train OOF only, literal 0.5). The frozen arm has NEVER been on
#   the leaderboard, so submitting both answers two separate questions for two slots:
#       (a) does Presto transfer to this task AT ALL?          <- the frozen arm
#       (b) does fine-tuning help or hurt on a shifted set?    <- the pair
#   Answering (b) without (a) would be uninterpretable, which is why this is two uploads and not
#   one. Single seed (42) each: this is a probe, and if it surprises us we expand to a seed pool.
#
#   ⚠️ FINALISTS ARE NOT AT RISK BY DEFAULT. champion_dualpolmix10_regimematch (0.907368983) and
#   champion_archblend4 (0.899643) stay designated unless the pre-registered read below fires the
#   top branch, which it almost certainly will not.
# =====================================================================
set -euo pipefail

# ---- STEP 0: vendor Presto (source + 3.3 MB MIT-licensed checkpoint) and verify it loads. ----
#      Pretrained WEIGHTS are explicitly legal; external DATA is not, and we add none.
#      The verify step asserts the encoder is ~404k params, so an upstream layout change fails
#      loudly here rather than silently producing garbage embeddings.
python tools/fetch_presto.py

# ---- STEP 1: ARM A — FROZEN encoder + ~129-parameter logistic head. The iter17 arm, but now
#      routed through calibrate_legal and actually submitted. ----
python run_presto.py --month-mode const --seed 42 --name presto_frozen

# ---- STEP 2: ARM B — FINE-TUNED, all 404,160 encoder params trainable. The untried arm. ----
#      Fixed 8 epochs, no early stopping and no LR search: early stopping needs a selection signal
#      and every offline signal we have is retired or blind (LB_LOG iter46). A fixed, pre-committed
#      budget is the only honest option and it keeps this to ONE variable against ARM A.
python run_presto.py --month-mode const --seed 42 --finetune \
  --ft-epochs 8 --ft-batch 64 --ft-lr-encoder 1e-4 --ft-lr-head 1e-3 --name presto_finetune

cat <<'NEXT'
=====================================================================
 PASTE BACK:
   - the "Presto encoder loaded: NNN,NNN params" line from BOTH runs. It MUST say 404,160.
   - the "adversarial AUC on Presto embeddings" line from ARM A (descriptive only - do NOT let it
     gate anything; that is precisely the iter17 mistake we are correcting).
   - the per-fold "combined@0.5" lines and the "OOF: f1@0.5=... auc=... combined=..." line for BOTH.
   - the "FITTED params = N" line from both (expect 129 for frozen, 404,289 for fine-tuned).
   - the per-epoch "train BCE" lines from ARM B — if BCE does not fall, the fine-tune did not train
     and the arm is VOID rather than negative.
   - the LEGAL calibration line from both (Platt slope + realized test pos-rate).
   - the Zindi AUC and F1 columns for BOTH uploads, not just the composite.

 UPLOAD (2 files):
   submissions/submission_presto_frozen.csv
   submissions/submission_presto_finetune.csv

 COMMITTED READ (pre-registered; do NOT renegotiate after seeing the numbers):

 Read the FROZEN arm first — it establishes whether the lane exists at all. Then read the PAIR.

   ARM A (frozen), composite:
     >= 0.913        Presto beats our best-ever public (0.910837) by more than the +-0.015 binomial
                     band. A genuinely new lane on the last weekend. THEN AND ONLY THEN reconsider
                     the finalist lock, and only after a seed pool, never off one seed.
     0.895 - 0.913   Presto is COMPETITIVE with a purpose-built champion while fitting 129
                     parameters. That is a striking result for the writeup and it does NOT make it
                     a finalist: an unmeasured artifact tying a measured one loses on the standing
                     rule (prefer the more-seed-averaged artifact, selected on fewer LB decisions).
     0.80 - 0.895    the lane transfers but loses. Expected outcome. Record and close.
     < 0.80          Presto does not transfer to this task. Also a clean result: it would say the
                     pretraining corpus (global crop/land-cover pixel series) does not cover
                     aquaculture ponds, which is worth stating in the report.

   ARM B (fine-tuned) MINUS ARM A, composite:
     <= -0.006       the "added capacity fitted to shifted rows hurts" law HOLDS, now demonstrated
                     on a pretrained model. This is the expected outcome and it is the single most
                     valuable thing this run can produce for the code review: the law was measured
                     on our own architectures, and this would extend it out-of-family.
     within +-0.006  inconclusive at our noise floor. Report as such; do NOT call it a win.
     >= +0.006       fine-tuning HELPS, which would CONTRADICT our own capacity law and is the one
                     outcome that should change what we do next. It would mean the law is about
                     RANDOM initialization, not capacity per se. Do not act on one seed; expand to
                     five and re-read.

   BOTH arms < 0.60 -> something is broken, not negative. Check the width line, the BCE trace, and
                     the Platt slope before concluding anything.

 A NOTE ON WHAT COUNTS AS SUCCESS HERE. The likeliest outcome is two mediocre scores and a clean
 negative. That is a fine result: it converts a lane we closed on RETRACTED evidence into a lane
 closed on a MEASUREMENT, which is exactly the trade this ledger exists to make, and it removes the
 last "but you never actually tried the pretrained model" objection from the code review.

 FINALISTS unless the top branch fires: champion_dualpolmix10_regimematch + champion_archblend4.
 DEADLINE 2026-08-16.
=====================================================================
NEXT
