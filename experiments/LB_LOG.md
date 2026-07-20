# LB_LOG — leaderboard reward ledger

The **reward signal** for the autoresearch loop (`gemini_loop/AGENT_BRIEF.md`). After
each Colab/Kaggle run you upload a `submission_*.csv` to Zindi and paste its public LB
score here. This is the ONLY ground truth for keep/discard — local OOF is blind.

**How to fill a row:** iteration + which experiment · the exact submission file you
uploaded · its logged test pos-rate · the Zindi public LB · keep (beats prior best) or
discard. Best-so-far is the max LB in the `keep?` = ✅ rows.

| # | Date | Experiment | Submission file | pos-rate | Public LB | keep? |
|---|------|------------|-----------------|----------|-----------|-------|
| — | 2026-07-09 | baseline: temporal Transformer | submission_seq (realized 0.649) | 0.649 | **0.8780** | ✅ (best) |
| 2 | 2026-07-20 | Step 2: GBDT+seq blend (ρ=0.849; best of prior63/65/67) | submission_seq_gbdt_prior63.csv | 0.646 | 0.8705 | ❌ (−0.0075; GBDT dilutes seq transfer despite higher OOF AUC) |
| 3 | 2026-07-20 | Step 3: seq + per_cell_detrend @ prevalence_target 0.649 | submission_seq_detrend.csv | 0.649 | 0.8266 | ❌ (−0.0514; adding channels overfits source, wrecks transfer. OOF blind: −0.004 only) |
| 4 | 2026-07-20 | Robustness: seq K 2→4 (more masking augmentation) | submission_seq_k4.csv | 0.649 | 0.8665 | ❌ (−0.0115; highest OOF 0.9840 → 2nd-worst LB. Even the winning lever overshoots. K=2 is a sharp optimum) |

**Loop paused → research round.** Three straight misses (blend, detrend, K=4). Escalated
to `gemini_loop/UPDATE_04.md` for fresh sourced ideas. Champion remains **seq K=2 @ 0.649
= 0.8780**. Key new concern: public LB ≈309 rows → ~±0.01 noise → single-probe A/B can't
resolve small (+0.005) gains; only large effects or breakages are detectable.

**Round-04 research triaged** (`gemini_loop/RESPONSE_04.md`): rejected Saerens-EM (proven
dead-end, covariate≠label shift) and Zou-threshold/EVI indices (non-transferable). Accepted
capacity-neutral ideas; testing relative-time reframing first, banking TTA + multi-seed bagging.

| 5 | _pending_ | Relative-time reframing (left-align window to t_rel=0; capacity-neutral) | submission_seq_reltime.csv | 0.649 | _awaiting Zindi_ | |

**Current best: 0.8780** (temporal Transformer, realized pos-rate 0.649).

---

### Reading the Step-2 run (what to record)
1. In Cell 4's log find `OOF rank correlation between components = ρ`.
   - ρ < ~0.90 → the blend adds decorrelated signal → upload the
     `submission_seq_gbdt_priorXX.csv` whose logged pos-rate is nearest **0.65**.
   - ρ ≈ 1.0 → skip the blend (record it as discarded); advance to Step 3.
2. Optional: also upload `submission_seq_v3.csv` to re-confirm the 0.8780 anchor.
   Reproducibility check: the **full** seq run prints `final_oof ≈ 0.9827` (the
   `0.88046` in results.tsv is the `--smoke` fingerprint, not the full run).
3. Paste the Zindi score into the table; mark ✅ only if it beats 0.8780.
