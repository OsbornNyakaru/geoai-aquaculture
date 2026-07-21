#!/usr/bin/env bash
# =====================================================================
# CURRENT EXPERIMENT — edited + pushed by Claude each iteration.
# The Colab notebook (colab_run.ipynb, Cell 4) runs exactly this file.
#
# LOOP PAUSED -> RESEARCH ROUND 06. This is the CHAMPION REPRODUCTION run.
#
#   *** ITER10 LOST. *** consistency_lambda=3.0 scored 0.8921 vs the 0.8955 champion (-0.0034).
#   Per the pre-committed decision rule ("~tie or drop -> revert to lambda=1.0"), config is
#   reverted. Reading: lambda=1.0 is an INTERIOR OPTIMUM. lambda=3 de-saturated FURTHER
#   (t_star 0.4450->0.3400, prevalence delta 1.30->0.725) while oof_auc HELD at 0.9896 — the
#   ranker did NOT collapse; de-saturation just stops paying past lambda=1. Objective lane CLOSED.
#
#   Both structural lanes are now measured closed (positional: dnorm -0.006, NoPE +0.001;
#   objective: lambda=3 -0.003), so we are OUT of queued ideas that plausibly clear the +-0.01
#   noise floor. Budget is NOT the constraint (~130 submissions left over ~26 days) — IDEAS are.
#   Loop paused for Deep Research round 06 (gemini_loop/UPDATE_06.md).
#
#   THIS RUN COSTS NO SUBMISSION. It re-runs the exact 0.8955 champion so that a "Run all"
#   during the research pause reproduces the champion instead of a rejected probe, and doubles
#   as a free environment-consistency check.
#
#   EXPECTED LOG (must match the iter9 champion run, or the revert was incomplete):
#     seq relative_time ON  |  seq cross-view invariance ON: lambda=1
#     final_oof ~ 0.97528   |  oof_auc ~ 0.98943  |  t_star ~ 0.4450
#     Prevalence target ON: test pos-rate 0.553 -> 0.649
#
#   NO NEED TO UPLOAD submission_seq_champion.csv (it is the 0.8955 file we already scored).
#   Only submit it if you want to re-confirm the anchor on a fresh account/environment.
# =====================================================================
set -euo pipefail

python run_pipeline.py --full --model seq --name seq_champion

echo "=== done. Champion reproduction; no submission needed. Awaiting research round 06 output. ==="
