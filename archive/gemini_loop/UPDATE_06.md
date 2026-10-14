# Deep-Research Brief — Round #06 (Claude Fable Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)
**Date:** 2026-07-21 · **Current best public LB:** **0.8955** · **Deadline:** 2026-08-16
**This round: one researcher, maximum depth.** Prior rounds ran two models in parallel and used their
overlap as the signal. This round we want the opposite: **go deep, read the actual sources, and be
specific.** Vague strategy advice is worthless to us now — we have run 10 leaderboard-gated
experiments and we know exactly which families fail.

---

## 0. Read this first — what "useful" means this round

This is a **live competition loop**. A coding agent implements one isolated change at a time, a human
runs it and submits, and the **leaderboard score** decides keep/discard. You are the idea source.

**We have hit a wall of a specific kind, and it is not the usual one.**

- We are **not short on compute.** A full run takes ~7 minutes.
- We are **not short on submissions** any more: ~26 days × 5/day ≈ **130 remaining**.
- We are short on **ideas that are large enough to measure**, and short on **any way to measure
  small ones**. Both of our productive lanes are now empirically closed (§2).

So: **do not give us a strategy overview.** Give us specific, sourced, implementable mechanisms —
with the mathematics written out where it matters — in the four lanes of §5, which are lanes we have
**never briefed before**: local validation that predicts the LB, feature engineering *inside the
sequence model*, mathematical techniques we have not touched, and CV design + the domain physics.

> **THE OVERRIDING RULE OF THIS PROBLEM: local cross-validation (OOF) is BLIND to the deciding
> effect — and by now demonstrably ANTI-correlated with it. The leaderboard is the only ground
> truth.** The evidence is no longer anecdotal. Our three highest-OOF runs and our three
> highest-LB runs are in *opposite* order:
>
> | Run | OOF (local combined) | Public LB |
> |---|---|---|
> | K=4 augmentation | **0.9840** (highest OOF we ever got) | 0.8665 (2nd-worst) |
> | old champion | 0.9827 | 0.8780 |
> | relative-time | 0.9811 | 0.8908 |
> | cross-view invariance λ=1.0 | **0.9753** (lowest) | **0.8955** (best) |
>
> **Any recommendation whose selection step is "pick the variant with the better CV score" is
> unusable as written.** If your idea needs model selection, you must say how to select it under
> this constraint — which is exactly research question #1.

**Hard constraints (violating any makes your idea unusable):**
- **Only the supplied competition data.** No external data, no auxiliary rasters, no lat/lon lookups.
- **No pretrained or foundation models** — bans TabPFN, ImageNet/SSL backbones, remote-sensing
  foundation models (Prithvi, SatMAE, Clay, …), and any transfer of learned weights. **Train from
  scratch, in-competition, every time.**
- **AutoML is banned.** Open-source, seeded, reproducible only.
- `TargetF1` is scored at a **hard 0.5 cut** — threshold tuning is illegal. A monotone
  prior/prevalence shift (so the F1-optimum *lands* at 0.5) **is** legal and is already in use.
- **5 submissions/day.** Not currently binding, but each idea should still be testable as a
  **single isolated change** against the champion.

---

## 1. Where we are (full ledger — every LB-gated result)

Every row is isolated against the then-champion with the operating point pinned at realized positive
rate 0.649 via an exact-prevalence logit shift, so the listed change is the **only** variable.

**Phase 1 — GBDT ensemble era**
| Change | LB | Δ |
|---|---|---|
| GBDT ensemble, inherited train prior (pos-rate 0.40) | 0.7140 | baseline |
| + base-rate/prior correction → pos 0.50 | 0.7561 | +0.042 |
| + prior correction swept → pos ~0.65 | **0.8260** | ✅ GBDT peak (lever then saturated) |
| + WIF & EVI water-index features | 0.7509 | ❌ −0.075 (train AUC 0.83, zero transfer) |

**Phase 2 — swap to a from-scratch temporal Transformer**
| Realized pos-rate | 0.593 | 0.627 | **0.649** | 0.672 |
|---|---|---|---|---|
| LB | 0.8776 | 0.8732 | **0.8780** | 0.8733 |

→ **+0.05 over the GBDT peak at essentially identical OOF.** This is the single finding that defines
the competition, and note the pos-rate row: **the prevalence lever is FLAT (within noise) across
0.59–0.67** for this model.

**Phase 3–4 — ten LB-gated experiments against the Transformer**
| # | Change (only variable vs then-champion) | OOF | LB | Δ | Verdict |
|---|---|---|---|---|---|
| 2 | + GBDT rank-average blend (ρ=0.85 with seq) | 0.952 | 0.8705 | −0.0075 | ❌ blending *dilutes* transfer |
| 3 | + `per_cell_detrend` input channels (remove per-series level) | 0.979 | **0.8266** | **−0.0514** | ❌ catastrophic |
| 4 | masking augmentation K=2 → K=4 | **0.984** | 0.8665 | −0.0115 | ❌ best OOF, near-worst LB |
| 5 | **relative-time reframing** (left-align window to t_rel=0) | 0.9811 | **0.8908** | **+0.0128** | ✅ **WON** |
| 6 | MC temporal-dropout TTA (inference-only, 8 views, soft-vote) | — | 0.8885 | −0.0023 | ❌ within noise |
| 7 | duration-normalized fractional positions (share a [0,1] frame) | 0.9789 | 0.8844 | −0.0064 | ❌ |
| 8 | **NoPE** — drop positional embedding entirely (set encoder) | 0.9789 | 0.8917 | +0.0009 | ➖ TIE |
| 9 | **cross-view invariance** L=BCE+λ·Var_k(logit), λ=1.0 | 0.9753 | **0.8955** | **+0.0047** | ✅ **CHAMPION** |
| 10 | same, λ=3.0 (strength probe) | 0.9727 | 0.8921 | −0.0034 | ❌ λ=1 is an interior optimum |

**Field:** top ≈0.9452 · top-5 ≈0.928–0.945 · rank-50 ≈0.876. **We are 0.8955; the gap to top-5 is
≈ +0.033** — about three times the size of our last two wins combined.

---

## 2. The findings you must internalize (and may challenge, with evidence)

**(A) Added capacity always loses. Capacity-neutral structural *deletion* is the only thing that has
ever won.** Added a model (blend) −0.0075. Added channels (detrend) −0.0514. Added augmentation
(K=4) −0.0115. Added inference robustness (TTA) −0.0023. Versus: reframed the coordinate system with
identical parameter count (+0.0128), and changed the training objective with zero new parameters
(+0.0047).

**(B) A structural deletion helps only if it deletes a channel that is genuinely SHIFTED between
train and test.** This is the sharpest thing we learned, and it is falsifiable:
- relative-time deleted window **START** (calendar month — genuinely shifted) → **+0.0128 WON**.
- duration-norm deleted window **LENGTH** (already distribution-matched by our augmentation — *not*
  shifted) → **−0.0064 LOST**.
- NoPE deleted **position entirely** → **+0.0009 TIE**, proving nothing exploitable remains on the
  positional axis after relative-time. The positional lane is **exhausted**.

**(C) The amplitude axis is toxic.** Removing per-series level (`per_cell_detrend`) cost **−0.0514**,
our worst result. Amplitude *is* signal here — pond backscatter level carries the class. **Do not
propose instance-norm, detrending, differencing, per-window value standardization, or any other
level/amplitude normalization**, in additive *or* replacing form, unless you can explain concretely
why it would not reproduce a −0.05 failure. That is a very high bar and two prior rounds failed it.

**(D) The objective lane is now closed at its optimum.** Cross-view invariance (penalize the variance
of a row's logit across its K=2 differently-masked views) won +0.0047 by **reducing overconfidence**
— `oof_auc` 0.9936→0.9894, prevalence shift δ 2.03→1.30, F1-optimal threshold t\* 0.500→0.445. But
λ=3.0 then *lost* (−0.0034) while pushing de-saturation further (t\* →0.340, δ→0.725) **with
`oof_auc` intact at 0.9896**. So the failure is **not** ranker collapse — de-saturation simply stops
paying. λ=1.0 is an interior optimum; further λ sweeping is measuring noise.

**(E) Measurement resolution is the binding constraint.** The public LB is ≈**309 rows**, giving
roughly **±0.01** of noise. We can resolve large effects (+0.05, +0.013) and breakages (−0.05), but
**not** +0.005 — our current champion's own margin sits right at that edge. We refuse to A/B inside
the noise band, which is why having 130 spare submissions does *not* automatically help us.
**Breaking this constraint is research question #1 and the highest-value thing you can do this round.**

---

## 3. Do NOT re-propose (all already tried, refuted, or rule-illegal)

1. **Amplitude/level normalization** in any form — see (C). −0.0514.
2. **Robustness / variance reduction as a way to climb** — TTA −0.0023; multi-seed bagging, SWA/EMA,
   snapshot ensembles are the same family. They are private-LB insurance at best. Do not pitch them
   as levers.
3. **Blending / stacking / ensembling across model families** — blend −0.0075; OOF meta-stacking
   (Ridge on OOF) rejected. Our components are ρ≈0.85 correlated and the weaker one drags.
4. **Saerens–EM / MLLS / BBSE prior estimation** — **rejected three separate rounds.** It assumes
   *label* shift; ours is *covariate* shift. BBSE estimated a test prior of 0.44 when the true
   LB-verified optimum is 0.649. Do not raise it a fourth time.
5. **Importance weighting / DANN / any density-ratio reweighting of the training loss** — effective
   sample size collapses at adversarial AUC 0.99. (But see Q1: we have *not* tried this for
   **evaluation**, which is a different and open question.)
6. **Self-training / pseudo-labeling on test** (incl. CAST) — same OOD/ESS-collapse family.
7. **Temperature scaling** — the prevalence shift already owns the operating point.
8. **Zou-style water-tree / WIF / EVI / fixed-threshold water indices** — tried, −0.075.
9. **TabPFN or any pretrained/foundation model** — rule-illegal.
10. **group-KFold or any "this is leakage" framing** — the train/test gap is *designed* covariate
    shift (different time periods and pilot regions) and has been proven leak-free.
11. **Further λ sweeping** and **further positional reframes** — both lanes measured closed (§2 B, D).

---

## 4. Self-contained problem statement (assume no repo access)

**Task.** Binary classification: is a given ~10 m cell an **aquaculture pond**?
**Data.** Train **1817** rows (after dropping 4 exact duplicates), test **1030** rows (public ≈309,
private ≈721). Per cell: a **12-month × 12-band** time series — Sentinel-1 SAR (**VH, VV**, in dB)
plus 10 Sentinel-2 optical bands. **No latitude/longitude, no spatial neighbourhood, no imagery
patches, no static covariates** — each row is one isolated pixel-cell's time series.

**Metric.** `0.6 · F1 + 0.4 · ROC-AUC`, submitted as two columns: `TargetF1` (binary, cut at a hard
0.5) and `TargetRAUC` (any rank-preserving score).

**The central trap — temporal masking.** Train rows are **fully observed** (all 12 months). Test rows
expose only a **consecutive 4-, 5-, or 6-month window** (the rest are sentinel −9999), plus extra
Sentinel-2-only dropout inside the window from cloud cover. Measured on the test set:
p(L=4)=0.335, p(L=5)=0.333, p(L=6)=0.332, and per-month S2-dropout rates ranging from 0.003 to 0.28
(worst in month 10, then month 06 at 0.115). We therefore expand every training row into **K=2 masked
"views"** whose window length, start, and S2 dropout are sampled from these measured test
distributions.

**The domain shift.** Train and test come from **different time periods and different pilot regions**,
by design. An adversarial classifier separates train from test rows at **AUC ≈ 0.99**, driven mainly
by per-series amplitude/level. This is genuine covariate shift, not leakage.

**Champion architecture (0.8955), exactly.** From scratch, no pretraining:
- Input per month: **12 standardized band values ⊕ 12 binary missing-indicators = 24 channels**.
- Linear projection to **d_model = 64**.
- **Relative-time reframing:** the observed window is left-aligned to t_rel = 0 *before* the learned
  positional embedding (length 12) is added.
- **2-layer** Transformer encoder, **4 heads**, GELU, dropout 0.2, with `src_key_padding_mask` over
  fully-missing months so attention never sees them.
- **Masked mean-pool** over observed months → MLP head → sigmoid.
- **Loss:** `BCE + λ · Var_k(logit)` across the row's K=2 masked views, λ=1.0 (cross-view invariance),
  with owner-grouped batching so a row's views land in the same batch.
- AdamW, lr 1e-3, weight decay 1e-4, batch 256, 60 epochs, **5-fold CV, n_repeats=1**; test prediction
  is the mean over the 5 fold-models; R=2 masked views per held-out row for OOF.
- **Operating point:** an exact-prevalence monotone logit shift forcing the realized test positive
  rate to **0.649** (the model's raw rate is 0.553). `TargetRAUC` is left as the untouched ranking.
- **Known weakness:** overconfident / saturated probabilities — a strong ranker with poor
  calibration. The cross-view penalty partially fixed this and that is exactly why it won.

---

## 5. Research questions — this is the substance of the round

Prioritize by **expected LB gain × feasibility**. For each recommendation state: the mechanism, the
mathematics where non-obvious, why it is legal under §0, why it will not repeat a §3 failure, the
**single isolated change** that tests it, and its **plausible effect size**. We would rather be
corrected than agreed with — if you think §2 is wrong, argue it with sources.

### Q1 — (HIGHEST VALUE) Can we build a local validation signal that actually predicts the leaderboard?

This is the question that would unlock everything else. We have ~130 spare submissions and a ±0.01
measurement floor, so we currently **cannot distinguish a real +0.005 from noise** — and our champion's
own margin is +0.0047. If you give us a local score that merely **ranks candidates correctly**, we can
screen dozens of ideas offline and spend submissions only on winners.

Note carefully what we have and have not tried: we rejected density-ratio **importance weighting for
training** (ESS collapses at adversarial AUC 0.99). We have **never tried it for evaluation**, where
the variance requirements are far weaker — we need a *ranking* over candidates, not an unbiased point
estimate. Specifically address:

- **Adversarial-validation-weighted OOF.** Fit a train-vs-test discriminator, weight each OOF row by
  its "test-likeness" w(x) = p(test|x)/(1−p(test|x)), and evaluate the competition metric under those
  weights. At AUC 0.99 the weights are near-degenerate — what is the actual effective sample size,
  and are there stabilizations (weight clipping, self-normalized/Hájek estimators, temperature on the
  discriminator, calibrated soft weights) that make the *ordering* of candidate models reliable even
  when the point estimate is not? What does the literature on **model selection under covariate
  shift** (importance-weighted CV, Sugiyama et al.) actually promise at this level of separation?
- **A most-test-like holdout.** Rather than weighting, *select* the top-q% most test-like training
  rows as a dedicated validation fold. How should q be chosen without a validation signal (chicken-
  and-egg), and does the resulting fold retain enough positives to measure F1 at all?
- **Masking-matched evaluation.** Our OOF currently pools R=2 masked views per held-out row. Should
  validation instead mirror the *exact* measured test recipe (window-length mix, per-month S2 dropout
  rates), and would evaluating per-recipe-stratum expose which candidates degrade on the hard strata?
- **Do the deltas transfer?** Is there evidence that under strong covariate shift, *differences*
  between models on a weighted/selected validation set are better preserved than absolute scores?
- **Anything else** — proxies, agreement-on-test measures (e.g. disagreement-based generalization
  estimates), or "predicting out-of-distribution accuracy from unlabeled test data" work that could
  apply to 1030 unlabeled test rows.

Please rank these by how likely each is to produce a *usable ordering*, and give us a concrete
protocol we can validate against our 10 existing (change, LB-delta) pairs — **we can retro-fit any
proposed validator to those 10 known outcomes as a free sanity check before trusting it.**

### Q2 — Feature engineering *inside the sequence model* (an admitted blind spot)

The champion's per-month input is only **12 raw standardized bands ⊕ 12 missing-indicators**. Every
richer feature we ever built (water indices, SAR percentiles, SDWI, window metadata, asymmetry flags)
exists **only in the GBDT lane, which is dead**. So the Transformer has never seen a single engineered
feature. The one time we added sequence channels it was `per_cell_detrend` and it cost −0.0514 — but
that was an **amplitude-axis** channel, the axis we now know is toxic.

**Ask:** are there physically-motivated, **shift-invariant, capacity-light** per-month features for
aquaculture ponds that are *not* on the amplitude axis and would be legal here? Consider and evaluate:
- **Polarimetric ratios / SAR structure:** VH/VV ratio, cross-pol ratio behaviour over open water vs
  vegetation vs bare soil; whether a ratio is genuinely level-invariant given our dB units.
- **Speckle / texture statistics** available from a single-pixel time series (we have no spatial
  neighbourhood — is anything meaningful still computable *temporally*?).
- **Water-permanence dynamics** expressed as *shape* rather than level — e.g. ordering/rank structure
  within the observed window, sign patterns of month-to-month change, run-lengths above/below a
  per-series statistic.
- **Phenology of pond management** — fill/drain/harvest cycles (see Q4) encoded as a channel.

For each: say explicitly why it would not repeat the −0.0514 detrend failure, and whether it should be
*added* as a channel (which historically loses) or **replace** an existing channel to stay
capacity-neutral (which historically wins). Prefer replacements.

### Q3 — Mathematical techniques we have never touched

Deliberately outside our tried list. Be concrete and write the objective/estimator where it matters:
- **Distributionally robust optimization** — group-DRO, CVaR-DRO, χ²-DRO. Is DRO viable at adversarial
  AUC 0.99 without the ESS collapse that killed importance weighting, and what defines the groups when
  we have no region labels? (Could the masking recipe — window length/start — serve as the group?)
- **Optimal-transport domain alignment** — Sinkhorn/OT-based alignment of the train and test
  representation distributions, done *without* labels and without adding model capacity. Is there an
  OT formulation that aligns *representations* while leaving amplitude information intact?
- **Spectral / frequency-domain representation** of a 4–6-step series — is a series this short even
  admissible to a frequency treatment, or is that a dead end on 4 samples?
- **Ranking objectives.** ROC-AUC is **40% of our metric** and we optimize plain BCE. What is the
  evidence for pairwise/listwise (RankNet-style), AUC-surrogate, or ordinal losses improving *transfer*
  (not just AUC) under covariate shift? Note this is an **objective-level, capacity-neutral** change —
  precisely the family that produced our last win, so we consider it a priori promising.
- **Margin / smoothing objectives** as *objective changes* rather than robustness add-ons — label
  smoothing, focal loss, logit-margin penalties. Given that our one win here came from **reducing
  overconfidence**, is there a principled successor to the cross-view penalty that attacks the same
  weakness by a different route (and would therefore *not* be redundant with λ=1.0)?
- **Conformal / selective prediction** for the hard-0.5 `TargetF1` column — anything legal and useful
  given we cannot tune the threshold but *can* apply monotone transforms.

### Q4 — CV design, and the domain physics we may be structurally unable to represent

**(a) Our CV has never been LB-validated.** It was inherited and never questioned: 5 folds,
`n_repeats: 1`, K=2 masked views per training row, R=2 views per held-out row, test prediction = mean
of the 5 fold-models. Could the fold structure, the repeat count, or the asymmetry between fitted
views (K=2) and OOF views (R=2) itself be costing transfer? Note the one time we changed K it cost
−0.0115, which suggests this machinery is *sensitive* — so we want to understand it, not poke it.
Also: is averaging 5 fold-models a hidden **ensembling** step (a family that has otherwise always hurt
us), and would training a single model on all data transfer better?

**(b) Read the actual domain sources.** Go and look at: the **FAO / ITU / Zindi challenge materials**
for this specific competition (framing, data-generation description, any organizer notes on how the
train/test split and the masking were constructed — the masking recipe is clearly deliberate and its
*purpose* may hint at what generalizes); any public discussion, forum threads, or write-ups for this
challenge; and the peer-reviewed **aquaculture-pond remote sensing** literature — pond fill/drain and
harvest cycles, stocking-season timing, how ponds are discriminated from natural water bodies, rice
paddies, and salt pans in S1/S2 time series.

**The question we most want answered from that reading:** is there a **physical signature of
aquaculture ponds** — some characteristic temporal pattern, a management cycle, a specific SAR/optical
co-behaviour — that our 24-channel, mean-pooled, 2-layer encoder is **structurally incapable of
representing**? If so, what minimal, capacity-neutral change to the representation or objective would
make it representable? That would be the next +0.01-class idea, and it is the kind of thing only
domain reading will surface.

---

## 6. Output format we want

For each recommendation:
1. **Name + one-line mechanism.**
2. **Why it is legal** (§0 constraints) and **why it is not a §3 repeat**.
3. **The maths / the exact change**, concretely enough to implement — for us "add a consistency
   penalty" was only actionable because the prior round wrote `L = BCE + λ·Var_k(logit)`.
4. **Expected effect size**, and if it is below ~0.01, how we could possibly measure it (this is where
   Q1 pays off).
5. **The single isolated experiment** that tests it against the 0.8955 champion.
6. **Sources.** Prefer primary literature and the actual challenge materials over blog summaries.

Rank everything into: **fund now** (worth a submission this week) · **fund if Q1 succeeds** (real but
unmeasurable today) · **park** (interesting, wrong stage) · **rejected on our evidence**.
