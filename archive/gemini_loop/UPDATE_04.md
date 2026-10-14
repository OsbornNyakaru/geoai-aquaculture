# Deep-Research Brief — Round #04 (Gemini Deep Research AND Claude Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)
**Date:** 2026-07-20 · **Current best public LB:** **0.8780** (unchanged) · **Deadline:** 2026-08-16

---

## 0. Read this first — how to be useful this round

Live competition loop: a coding agent implements + submits, you do fresh sourced research
and react to **leaderboard** results. Since round 3 we ran three experiments and **all three
lost** on the LB (details in §1). We need you to change our strategy, not micro-optimize it.

> **The overriding rule of this problem: local cross-validation (OOF) is BLIND to the deciding
> effect — often ANTI-correlated with it. The leaderboard is the only ground truth.** This round
> gave a fifth confirmation: our highest-OOF run (0.9840) was our second-WORST on the LB
> (0.8665). Any recommendation that leans on OOF for selection is discarded.

**Hard constraints (violating any makes your idea unusable):**
- Only the supplied competition data. **No external data, no models pretrained on other data**
  — bans TabPFN, ImageNet/SSL backbones, all foundation models. Train from scratch.
- **AutoML banned.** Open-source, seeded, reproducible only.
- `TargetF1` scored at a **hard 0.5 cut** — no threshold tuning. Prior/prevalence correction
  (a monotone shift so the F1-optimum lands at 0.5) *is* allowed.
- **Max 5 submissions/day** — the scarce resource. Say which ideas deserve one.

---

## 1. Results since round 3 (ground truth — all three LOST)

Current champion: **from-scratch temporal Transformer, K=2 masking-augmented views, operating
point held at realized pos-rate 0.649 → LB 0.8780.** Everything below was tested against that
reference with the operating point **held constant at 0.649** (via an exact-prevalence logit
shift), so each change is cleanly isolated.

| # | Change tested (only variable vs champion) | OOF combined | Public LB | Δ vs 0.8780 |
|---|---|---|---|---|
| 2 | + GBDT rank-average blend (0.7 seq / 0.3 GBDT; OOF rank-corr ρ=0.85) | 0.952 | 0.8705 | −0.0075 |
| 3 | + `per_cell_detrend` input channels (subtract each cell's own per-band temporal mean) | 0.979 | **0.8266** | **−0.0514** |
| 4 | seq masking views K=2 → K=4 (more test-like augmentation, no added dims) | **0.984** | 0.8665 | −0.0115 |

Note the inversion: the run with the **highest** OOF (K=4, 0.984) scored **near the bottom**.

---

## 2. The three findings you must internalize

**(A) This problem PUNISHES added capacity.** Both "add complexity" bets lost:
- Adding a *model class* (GBDT) to the blend diluted the seq model's transfer even though the
  two were decorrelated (ρ=0.85) and GBDT had higher OOF AUC.
- Adding *input channels* (12 detrended channels appended to the raw bands) cratered the LB by
  −0.051 while OOF barely moved. The transformer is small-n (1817 train rows); extra input
  dimensions appear to overfit the *source* distribution and destroy transfer.
⇒ We now treat anything that adds inputs or parameters as **low-prior**. This includes the
  other invariant-channel ideas we had queued (`deltas`, `indices`, `rank`).

**(B) The champion is a SHARP optimum.** Even *more of the winning lever* (masking augmentation
K=2→4) overshot and lost −0.0115. Parameter/augmentation nudges are not moving us up.

**(C) MEASUREMENT RESOLUTION is now the binding constraint.** Public LB ≈ **309 rows**. At
pos-rate 0.649 a few flipped rows move the combined metric ~±0.005–0.01. detrend's −0.051 is
unambiguously real, but blend (−0.0075) and K=4 (−0.0115) are close to the noise floor. **We may
be A/B-testing inside the noise band** — single-submission public-LB probes likely cannot resolve
the small (+0.005) gains we were hunting. We can reliably detect only *large* effects (the
GBDT→seq swap was +0.05) or *breakages*. This reframes the entire strategy.

---

## 3. Corrections to earlier rounds (do not re-propose)

1. **The "remove per-series level to improve transfer" thesis is EMPIRICALLY DISPROVEN for this
   model.** We diagnosed the shift as a per-series level offset (adversarial train-vs-test AUC
   0.99 → 0.94 on region-normalized indices) and predicted detrending would help. It did the
   opposite (−0.051). Do not re-recommend detrend/instance-norm/differencing as *added channels*.
   If you believe level-invariance still matters, you must explain why it must *replace* rather
   than *augment* the raw bands, and why that wouldn't also just lose transferable signal.
2. Still-standing dead ends from prior rounds: BBSE/EM prior estimation · WIF / fixed-threshold
   water features · TabPFN (pretraining) · temperature scaling · importance-weighting / DANN
   (ESS collapse at adversarial AUC 0.99) · OOF meta-stacking · group-KFold / "it's leakage"
   (the gap is designed covariate shift, proven leak-free).

---

## 4. Self-contained problem recap (for a fresh reader)

- **Task:** binary classification — is a 10 m×10 m cell an aquaculture pond? Train 1817 rows
  (post-dedup), test 1030 (public ≈309, private ≈721). **Metric: 0.6·F1 + 0.4·ROC-AUC**, with two
  submission columns: `TargetF1` (binary, hard 0.5 cut) and `TargetRAUC` (any rank-preserving score).
- **Data:** per cell a **12-month × 12-band** series — Sentinel-1 SAR (VH, VV, dB) + 10
  Sentinel-2 optical bands. **No lat/lon, no spatial neighborhood, no static covariates.**
- **The core trap — temporal masking:** train rows are fully observed (12 months); test rows expose
  only a consecutive **4/5/6-month** window (rest = −9999), with extra Sentinel-2 cloud dropout. We
  augment each train row into K masked "views" matching the measured test masking recipe.
- **Designed domain shift:** train and test are different time periods and pilot regions; adversarial
  classifier separates them at AUC ≈0.99. Genuine covariate shift, proven leak-free.
- **Champion model:** from-scratch Transformer — per-month [standardized 12 bands ⊕ 12
  missing-indicators] → linear proj d=64 → learned positional emb → 2-layer encoder (4 heads, GELU,
  dropout 0.2, `src_key_padding_mask` over fully-masked months) → masked-mean-pool → MLP head →
  sigmoid. AdamW + BCE, K=2 masking-augmented views, 5-fold (test = mean of fold-models), n_repeats=1.
  Known weakness: **overconfident** (saturated probs) — strong ranker, poor probability (we now fix
  the operating point with an exact-prevalence logit shift, so this no longer affects the F1 cut).
- **Field (verified):** top ≈0.9452, top-5 ≈0.928–0.945, rank-50 ≈0.876. We are ≈0.878, so the gap
  to top-5 is a real **~+0.05**.

---

## 5. Research questions (prioritize by expected LB-gain × feasibility; say which deserve a submission)

1. **Given measurement resolution ≈±0.01 on a 309-row public LB, how should we run experiments at
   all?** Concretely: how do we detect a true +0.005 improvement that's smaller than public-LB noise?
   Options to evaluate — repeated seeds + a paired/McNemar test on the *same* public rows across
   submissions; trusting a robust local proxy that is NOT same-distribution OOF (is there one here?);
   deliberately spending submissions to estimate the public-LB noise floor; or abandoning small gains
   entirely and only pursuing large-effect changes. What did winners of small-public-LB shifted
   competitions do to avoid fooling themselves?

2. **What LARGE-effect levers remain** (the GBDT→seq swap was +0.05; we need effects of that size, not
   +0.005)? Given "added capacity hurts" and "OOF is blind," what *structurally different* from-scratch
   approaches have produced step-changes under strong covariate shift on short (4–6-step) masked
   multivariate satellite series? Rank by expected effect size, not incremental polish.

3. **Is our test-prediction protocol itself leaving transfer on the table?** Test prediction is the
   mean of the 5 fold-models applied to each test row's *single* given masking. Would (a) test-time
   masking augmentation (predict each test row under several resampled masks consistent with its
   observed window, then average), (b) a different pooling than masked-mean, or (c) training a single
   model on all data instead of 5 folds, plausibly transfer better? These change inference, not capacity.

4. **Private-LB robustness / submission selection.** With a designed shift this strong and only ~309
   public rows, how do we choose the final 1–2 submissions to protect the **private ≈721**? Is the
   champion's public 0.8780 likely to hold privately, and how should we hedge (seed-averaging,
   conservative operating point, picking the model that is *least* tuned to public rows)?

5. **Managed-aquaculture temporal signatures invariant to WHICH months are observed** — but only if you
   can argue they'd survive as a *replacement/reframing* of the input, not an added channel (see §3.1).
   Stocking/drain/harvest phenology, water-presence persistence, SAR temporal behavior. Which, if any,
   is worth a submission given everything above?

If you disagree with any conclusion here — especially "added capacity hurts" or "we're inside the
noise band" — argue it explicitly with sources and a concrete, rule-legal, single-submission test. We
would rather be corrected than agreed with. Return prioritized, implementable recommendations.
