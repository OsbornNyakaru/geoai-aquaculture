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
| 2 | _pending_ | Step 2: GBDT+seq blend | submission_seq_gbdt_priorXX.csv | | | |

**Current best: 0.8780** (temporal Transformer, realized pos-rate 0.649).

---

### Reading the Step-2 run (what to record)
1. In Cell 4's log find `OOF rank correlation between components = ρ`.
   - ρ < ~0.90 → the blend adds decorrelated signal → upload the
     `submission_seq_gbdt_priorXX.csv` whose logged pos-rate is nearest **0.65**.
   - ρ ≈ 1.0 → skip the blend (record it as discarded); advance to Step 3.
2. Optional: also upload `submission_seq_v3.csv` to re-confirm the 0.8780 anchor
   (its `final_oof` should print ≈ 0.88046 — reproducibility check).
3. Paste the Zindi score into the table; mark ✅ only if it beats 0.8780.
