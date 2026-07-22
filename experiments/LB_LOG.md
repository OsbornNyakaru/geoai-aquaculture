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

| 5 | 2026-07-21 | Relative-time reframing (left-align window to t_rel=0; capacity-neutral) | submission_seq_reltime.csv | 0.649 | **0.8908** | ✅ **NEW BEST** (+0.0128; clears ±0.01 noise. First win since champion — capacity-NEUTRAL structural reframe transfers where every capacity-ADDING change lost. Attacks calendar-month memorization.) |
| 6 | 2026-07-21 | MC temporal-dropout TTA on champion (inference-only; mask 1-2 active months, soft-vote 8 views) | submission_seq_reltime_tta.csv | 0.649 | 0.8885 | ❌ (−0.0023; within noise but did NOT beat champion. 2nd robustness move to land within-noise — variance reduction isn't the lever here. Reverted; kept as a private-LB hedge candidate.) |

**Current best: 0.8955** (temporal Transformer + relative-time + cross-view invariance λ=1.0, realized
pos-rate 0.649). Prior anchor 0.8908 = relative-time alone.

**Pattern after iter6:** capacity-neutral *robustness* moves (TTA now, and by extension multi-seed
bagging) land within the ±0.01 noise — they can't be validated on public and don't lift rank. The
only changes that moved LB were *structural* (GBDT→seq +0.05; relative-time reframe +0.013). Next
high-value direction = the next structural reframe, not more variance reduction.

**Round-05 research triaged** (`gemini_loop/RESPONSE_05.md`): BOTH Deep Research reports (Gemini +
Claude) independently ranked **duration-normalized positions** #1 → staged as iter7. Re-rejected
Saerens-EM (3rd time), Zou/WIF/EVI, CAST self-training, CropNet-blend big-bang. Banked: NoPE set
encoder (iter8), cross-view invariance objective (iter9), one-time prevalence sweep.

| 7 | 2026-07-21 | Duration-normalized fractional positions (share one [0,1] frame across window lengths; parameter-neutral) | submission_seq_dnorm.csv | 0.649 | 0.8844 | ❌ (−0.0064; did not beat champion. Diagnosis: window LENGTH is already train/test distribution-matched by the masking augmentation → no length-shift to remove; dnorm only normalized away possibly-informative length signal. Also confound: interpolation read untrained table indices 6-11. REVERTED.) |

**Refined positional lesson (iter5 vs iter7):** relative-time removed window START (calendar month,
which IS shifted train-vs-test) → +0.0128 WON. dnorm removed window LENGTH (already matched by
augmentation, NOT shifted) → −0.0064 LOST. **Positional reframes help ONLY when they delete a channel
that is actually shifted between train and test.** iter8 NoPE tests removing positional identity entirely.

| 8 | 2026-07-21 | NoPE / permutation-invariant SET encoder (drop positional embedding; pos_encoding=none) | submission_seq_nope.csv | 0.649 | 0.8917 | ➖ TIE (+0.0009, within noise). Removing ALL position costs nothing → order is nuisance-or-neutral (confirms SAR set-statistic physics). **LOCKED as the diverse private-LB finalist** (max-different model, ties public → fails on different rows). |

**Positional lane EXHAUSTED:** absolute-calendar 0.8780 → relative-time 0.8908 (+0.013, WON) → NoPE
0.8917 (tie). All the gain was deleting calendar-START memorization; residual position is neutral.
Two reframes now < +0.003 → stop-rule. The other big shift channel (per-series amplitude) is toxic to
touch (detrend −0.051). Budget shifts to the OBJECTIVE lever (iter9) then endgame (prevalence + finalists).

| 9 | 2026-07-21 | Cross-view invariance objective (L=BCE+λ·Var_k(logit) across K=2 views; λ=1.0) | submission_seq_xview.csv | 0.649 | **0.8955** | ✅ **NEW BEST** (+0.0047 vs 0.8908; best public score yet). Reduced overconfidence (oof_auc 0.9936→0.9894, prevalence delta 2.03→1.30) — hit the model's diagnosed weakness. At edge of noise, so iter10 probes λ=3.0 to test if the lever scales. |
| 10 | 2026-07-21 | Cross-view invariance strength probe (λ 1.0→3.0) | submission_seq_xview_l3.csv | 0.649 | 0.8921 | ❌ (−0.0034; did not beat champion. REVERTED to λ=1.0.) |

**Objective lane CLOSED at λ=1.0.** iter10 shows λ=1.0 is an *interior optimum*, not a floor.
λ=3.0 pushed de-saturation further (t\* 0.4450→0.3400, prevalence delta 1.30→0.725, raw test
pos-rate 0.553→0.583) while `oof_auc` HELD at 0.9896 — so the drop is **not** ranker collapse;
de-saturation simply stops paying past λ=1. Both structural lanes are now measured closed
(positional: dnorm −0.006 / NoPE +0.001; objective: λ=3 −0.003). **Loop PAUSED → research round 06**
(`gemini_loop/UPDATE_06.md`), briefing lanes never explored: LB-predictive local validation,
sequence-level feature engineering, untried mathematical techniques, and CV design.

**BREAKTHROUGH (2026-07-21):** relative-time reframing broke the 10-day 0.8780 plateau. The
"added capacity hurts" lesson now has its complement: capacity-NEUTRAL *structural* change (same
params, reframed coordinates) is the direction that transfers. Iter6+ bank robustness moves (TTA,
multi-seed bagging) on top; both capacity-neutral, gated vs 0.8908.

| 11 | 2026-07-22 | **Offline LB-predicting validator** (retro-fit to 7 known-LB anchors) | *no submission* | — | **n/a — 0 subs** | ✅ **PASSED** |

**iter11 PASSED — the measurement constraint is broken.** Retro-fit against the 7 anchors:

| Estimator | Concordance (pairs with \|ΔLB\| > 0.01) | Spearman ρ | Verdict |
|---|---|---|---|
| **ATC-F1** — metric-aligned, built this round from the rank-only proof | **15/15** | **+0.964** | **PASS** (exact null p ≈ 0.005) |
| **DIS** — two-seed disagreement | **5/5** | **+1.000** (n=4) | **PASS** (p ≈ 0.042 → 2nd vote only) |
| ATC — the ORIGINAL pre-committed estimator | 6/15 | **−0.429** | FAIL |
| MARG — naive control | 8/15 | −0.321 | FAIL |

**Read this carefully: the pre-repair iter11 would have failed completely.** Raw ATC is
anti-correlated, MARG is anti-correlated, and DIS would have printed "insufficient bundles" because
only the champion carried a second seed. The two estimators that cleared are exactly the two things
round 07 added (`gemini_loop/RESEARCH_07.md`).

**The two failures are themselves evidence for the rank-only proof.** ATC and MARG both measure
*confidence/saturation*, which the LB provably cannot see — and both came out **negative**. Visible
in the table: `seq_a_l3` and `seq_a_xview` have the LOWEST MARG (0.309, 0.369) yet the HIGHEST LB.
De-saturation was never the mechanism; it is an artifact that anti-correlates with what wins.

**→ SCENARIO A is live** (`RESEARCH_07.md` §6). Candidates are now screened offline and submissions
spent only on things ≥2 cleared estimators already rank above the champion. ~80 submissions remain,
and the constraint moves from measurement back to idea quality.

| 12 | 2026-07-22 | **First offline screen**: pooling `mean_std` vs `mean_max`, 24→14 channel compaction, rank-replacement (amplitude retest), antithetic views | *no submission* | — | pending | — |

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
