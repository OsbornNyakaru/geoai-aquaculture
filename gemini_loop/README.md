# Gemini research loop

An iterative Claude ↔ Gemini improvement loop for the competition:

```
Claude implements & submits
   → shares results + open questions with Gemini   (UPDATE_NN.md)
   → Gemini does deep research + critique          (RESPONSE_NN.md)
   → Claude implements the best ideas
   → new leaderboard results
   → repeat (UPDATE_NN+1.md)
```

## Convention
- `UPDATE_NN.md` — Claude → Gemini. Paste into Gemini Deep Research. Each one
  reports what the last round scored, corrects any wrong advice, states the
  current approach + direction, and poses sharpened research questions.
- `RESPONSE_NN.md` — paste Gemini's findings back here (save its reply) so the
  loop has a record and Claude can implement from it.
- One cycle per **major submission / result**.

## Log
| # | Date | Best LB before | Change tested | Result |
|---|------|----------------|---------------|--------|
| 01 | 2026-07-07 | 0.7140 | prior correction 0.40→0.50 pos-rate | **0.7561 (+0.042)** — kept |
| 01 | 2026-07-07 | 0.7561 | WIF+EVI features (prior 0.50) | 0.7509 (−0.005) — **reverted** (train AUC 0.826 but no LB gain; domain-shift trap) |
| 02 | 2026-07-08 | 0.7561 | prior sweep — full F1-vs-prior curve | peak **0.8260 @ realized ~0.65**; declines both sides — prior lever **saturated** |
| 02 | 2026-07-08 | 0.8260 | from-scratch temporal Transformer (`--model seq`) | **0.8780 @ realized 0.649** (+0.052 despite identical OOF) — **new base model** |
| 03 | 2026-07-09 | 0.8780 | Gemini round-2 triage + next brief (`UPDATE_03.md`) | rejected re-treads (BBSE/WIF/TabPFN/OOF-stacking); queued blend + seq robustness (EMA/label-smooth/bag/AUC-margin) + transfer channels |
| 04 | 2026-07-10 | 0.8780 | 3-agent research (grandmaster/math/codebase) + **built Step1+Step3** | diagnosis: shift = per-series LEVEL → promoted invariant inputs to Step 3. Built `prevalence_target` δ-shift + `seq.channels.*` (detrend/deltas/indices/rank), defaults inert (smoke repro 0.88046). Confirmed no-ops: temp-scaling, importance-weighting/DANN (ESS collapse @AUC0.99). Pending Colab full runs → blend + Step3 LB probe |
| 04 | 2026-07-20 | 0.8780 | **Ran the 3 pending probes on the LB — all LOST.** Step2 GBDT+seq blend 0.8705 (−0.007); Step3 per_cell_detrend **0.8266 (−0.051)** — detrend/level-removal thesis disproven on the net; Step1 `prevalence_target 0.649` verified working. Iter4 K=2→4 0.8665 (−0.012). Meta: **added capacity hurts, OOF anti-correlated (hi-OOF=lo-LB), public LB ≈309 rows so we're inside the ±0.01 noise band.** Champion still seq K=2 @0.649. → wrote `UPDATE_04.md`; loop paused for Deep Research |
