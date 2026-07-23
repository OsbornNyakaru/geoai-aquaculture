# Deep-Research Brief — Round #09 (Gemini Deep Research AND Claude Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)
**Date:** 2026-07-23 · **Champion public LB:** **0.8955** single-seed / **≈0.8865 reliable** (see §1) · **Deadline:** 2026-08-16 (~24 days)

**Rounds 07–08 were run in-loop** (a team of internal research agents + an EY-competition comparison),
so the last *external* brief you saw was Round 06. **Four things have changed since then, and two of
them invert premises you were given last time.** Read §0 before anything else — if you answer against
the Round-06 constraints you will waste the round.

---

## 0. What changed since Round 06 — read this first

Round 06 told you four things that are now **corrected by measurement**:

1. **"No pretrained/foundation models."** ❌ **WRONG.** The verbatim rule is: *"You may use pretrained
   models as long as they are openly available to everyone."* Pretrained models are **legal**. External
   **data** is still banned; AutoML is still banned. — This reopened the single biggest lane, and we
   have already spent it (see #4).

2. **"The metric can see calibration / de-saturation."** ❌ **WRONG.** The metric is **rank-only**.
   After the prevalence pin fixes the predicted-positive count P̂, F1 = 2·TP/(P̂+P) is monotone in
   precision@k — a functional of the test **ranking** alone; ROC-AUC is rank-only by definition. So a
   constant logit offset, temperature, or any monotone recalibration is **invisible** to the LB. Every
   round-06 story that explained a win via "reduced overconfidence" was reading a **correlate**, not
   the cause.

3. **"Q1 (a local signal that predicts the LB) is the open problem."** ✅ **PARTIALLY SOLVED.** We built
   it. Two label-free estimators now **clear a retro-fit** against our known (change, LB) anchors:
   **ATC-F1** (a metric-aligned Average-Thresholded-Confidence estimator, Spearman ρ **+0.964**, 15/15
   concordant on informative anchor pairs) and **DIS** (two-seed disagreement, ρ **+1.000**, n=4). We
   can now screen candidates offline for **0 submissions**. **BUT** its resolution is coarse — see #4
   — so it screens *large* effects, not small ones.

4. **The binding constraint moved from "we can't measure" to "seed noise dominates almost every real
   effect."** We finally measured our **seed-to-seed spread**: the champion config, changing *only* the
   RNG seed, scored **0.8955 (seed 42) vs 0.8764 (seed 7)** — sd ≈ **0.0191**. Seed rank-correlation is
   **0.9511**. **This voids 9 of our 11 historical verdicts** (their effect sizes are smaller than one
   seed-swing), including the two "wins" we celebrated last round. The screen's own resolution,
   measured two independent ways that agree, is **≈0.010–0.013 LB**.

**Consequence for you:** the useful unit of advice is no longer "a +0.005 tweak." Nothing below
**~0.010 LB** is measurable *in principle* with our budget, and almost every architectural knob lives
below that floor. **We need ideas whose plausible effect is ≥ ~0.02, or ideas that reduce variance in a
way that survives to the private 721-row slice.** Vague strategy is worthless; give us specific, sourced,
implementable mechanisms with the math written out.

---

## 1. Where we are — the ledger, and why most of it is now noise

Every row was isolated against the then-champion with the operating point pinned at realized positive
rate 0.649. **The catch:** none of these were seed-replicated at the time, and the seed sd is 0.0191.
**Treat any |Δ| < ~0.02 as UNRESOLVED, not as a verdict.**

| # | Change (vs then-champion) | OOF | LB | Δ | Status under the seed finding |
|---|---|---|---|---|---|
| — | GBDT peak (prior-corrected, pos≈0.65) | — | 0.8260 | — | model-class baseline |
| — | **GBDT → from-scratch Transformer** | ~0.98 | **0.8780** | **+0.052** | ✅ **REAL** (≫ seed sd) |
| 3 | `per_cell_detrend` amplitude channels | 0.979 | **0.8266** | **−0.0514** | ✅ **REAL** (≫ seed sd) |
| 4 | masking K=2 → K=4 | **0.984** | 0.8665 | −0.0115 | ⚠️ inside seed noise |
| 5 | relative-time reframing | 0.9811 | 0.8908 | +0.0128 | ⚠️ inside seed noise |
| 8 | NoPE (drop positional embedding) | 0.9789 | 0.8917 | +0.0009 | ⚠️ inside seed noise |
| 9 | cross-view invariance λ=1.0 (**"champion"**) | 0.9753 | **0.8955** | +0.0047 | ⚠️ inside seed noise |
| 10 | λ=3.0 | 0.9727 | 0.8921 | −0.0034 | ⚠️ inside seed noise |

**Only two effects in the entire project exceed the noise floor, and BOTH are model-class changes:**
the GBDT→Transformer swap (+0.05) and the detrend breakage (−0.05). Every architectural tweak, loss
term, pooling variant, positional reframe and regularization knob is below the floor and **unmeasurable
in principle** at our budget. This is the **central design law** of the problem now.

**Reliable champion level ≈ 0.8865**, not 0.8955: seed-averaging 5 seeds of the champion scored
**0.88653** (we *predicted* 0.886 from the variance model — confirmed to 0.0006). 0.8955 was the better
of two draws and is very likely an upward fluctuation.

**Field (public):** top ≈ 0.9452 · top-5 ≈ 0.928–0.945 · rank-50 ≈ 0.876. **Gap from our reliable
level to top-5 ≈ +0.04** — roughly the size of the GBDT→Transformer jump. Small tweaks cannot close it;
only another model-class-scale move can.

---

## 2. The findings you must internalize (challenge them, but with evidence)

**(A) Only model-class changes clear the noise floor.** See §1. Corollary: an idea worth a submission
must be plausibly ≥ 0.02, or it is unfundable regardless of how elegant it is.

**(B) The train→test shift is REAL, LARGE, and lives in the raw data — now proven three ways.** (i) An
adversarial classifier separates train from test at AUC ≈ 0.99 on hand features. (ii) It *also* separates
them at AUC **0.965–0.976** on the **frozen embeddings of a general-purpose SSL foundation model that
never saw our labels** (Presto — see #4). (iii) Our OOF (~0.975 combined) vs LB (~0.89) gap is stable and
large. **So the ~0.085 OOF→LB gap is mostly irreducible covariate shift, not a fixable modelling bug.**
Any method that only improves the in-distribution fit will not transfer.

**(C) Amplitude IS the primary signal (the toxicity story is subtler than Round 06 said).** Round 06
claimed "amplitude normalization is toxic (−0.0514)." That is directionally right but the *mechanism* was
mis-stated: `per_cell_detrend` **appended** 12 channels (24→36) rather than removing amplitude, so it
never actually tested "remove amplitude." A later experiment that **replaced** raw values with within-
series **rank** (true amplitude removal, capacity-neutral) collapsed OOF 0.975→0.86 — confirming
**persistently-low backscatter level is the class signal.** Do not propose instance-norm, detrend,
differencing, or per-window value standardization. Very high bar.

**(D) The metric is rank-only** (§0 #2). Do not propose temperature scaling, calibration, focal/label-
smoothing *as LB levers* — the LB cannot see them. They may matter only as *side effects* on the ranking.

**(E) Ensembling has, so far, only bought variance — not level.** Seed-averaging (95.1%-correlated
members) landed *at* the member mean, not above it (variance reduction, no level gain). GBDT+seq blending
lost (−0.0075; components ρ≈0.85, weaker drags). **Open question (Q4):** does pooling *decorrelated
architectures* — as opposed to seeds or a weaker family — buy **level**? We are testing this right now.

---

## 3. What we have already spent since Round 06 (do NOT re-propose these specific things)

1. **Presto (pixel-time-series SSL foundation model), frozen encoder + logistic head.** ❌ **DEAD.**
   Presto is a ~0.4M-param transformer pretrained with masked-modality SSL on 21.5M Sentinel-1/2 pixel
   time series — our exact data shape, its pretraining objective *is* our central difficulty. Frozen, it
   is a 128-dim embedding + a 129-param head (less fitted capacity than anything we ship). Result:
   adversarial AUC **0.965–0.976** on its embeddings (it **encodes** the shift rather than normalizing
   it), ATC-F1 **−0.044 to −0.059 LB** vs champion, OOF 0.967–0.969 (already below champion). Both month-
   modes (calendar-deleted and calendar-true) HELD. **The frozen-foundation-model-as-feature-extractor
   lane is closed for Presto specifically** — but see Q2, other foundation models are NOT all tested.
2. **Seed-averaging as a level lever.** Bought variance only (§2 E). Do not pitch multi-seed bagging /
   SWA / EMA / snapshot ensembles as *climbers*; they are private-slice insurance.
3. **Everything on the Round-06 "do not re-propose" list still stands** *except* the pretrained-model ban
   (now legal) and the framing that pseudo-labeling is categorically dead (we now have a screen and a
   transductive rationale — see Q1, it is reopened as a *screenable* candidate).
4. **Round-06 refutations that remain valid:** amplitude normalization (§2 C); GBDT/family blending;
   Saerens–EM / BBSE / MLLS label-shift prior estimation (ours is covariate shift; BBSE gave 0.44 vs the
   true 0.649); importance-weighting the **training** loss (ESS collapses at adv-AUC 0.99); WIF/EVI/SDWI
   water indices (−0.075); temperature scaling; group-KFold "leakage" framing (the shift is designed).

---

## 4. Self-contained problem statement (assume no repo access)

**Task.** Binary classification: is a given ~10 m cell an **aquaculture pond**?
**Data (supplied only).** Train **1817** rows (after dropping 4 exact duplicates), test **1030** (public
≈309, private ≈721). Per cell: a **12-month × 12-band** time series — Sentinel-1 SAR (**VH, VV**, dB) +
10 Sentinel-2 optical bands. **No lat/lon, no spatial neighbourhood, no image patches, no static
covariates.** Each row is one isolated pixel-cell's time series.
**Metric.** `0.6·F1 + 0.4·ROC-AUC`. Two columns: `TargetF1` (binary, **hard 0.5 cut** — threshold tuning
illegal; a monotone prevalence shift so the F1-optimum *lands* at 0.5 is legal and in use) and
`TargetRAUC` (any rank-preserving score).
**The designed trap — temporal masking.** Train rows are fully observed (12 months). Test rows expose
only a consecutive **4/5/6-month window** (p ≈ 0.335/0.333/0.332), rest sentinel −9999, plus extra
S2-only cloud dropout inside the window (per-month rates 0.003–0.28). We expand each train row into
**K=2 masked "views"** sampled from these measured test distributions.
**The domain shift.** Train and test are from **different time periods and pilot regions**, by design.
Adversarial train-vs-test AUC ≈ 0.99 (hand features) / ≈0.97 (foundation-model embeddings). Genuine
covariate shift, proven leak-free.
**Constraints.** Supplied data only (no external data/rasters/lookups). **Pretrained models allowed
(openly available).** AutoML banned. Open-source, seeded, reproducible. 5 submissions/day, 100 total.

**Champion architecture (0.8955 single-seed), exactly — from scratch:**
- Per-month input: **12 standardized band values ⊕ 12 binary missing-indicators = 24 channels** → Linear
  to d_model = 64.
- **Relative-time reframing:** observed window left-aligned to t_rel=0 *before* a learned length-12
  positional embedding.
- **2-layer** encoder, 4 heads, GELU, dropout 0.2, `src_key_padding_mask` over fully-missing months.
- **Masked mean-pool** → MLP head → sigmoid.
- **Loss:** `BCE + λ·Var_k(logit)` across K=2 masked views, λ=1.0 (cross-view invariance), owner-grouped
  batching.
- AdamW lr 1e-3, wd 1e-4, batch 256, 60 epochs, **5-fold CV, n_repeats=1**; test = mean of 5 fold-models;
  R=2 masked OOF views per held-out row.
- **Operating point:** exact-prevalence monotone logit shift → realized test positive rate **0.649** (raw
  0.553). `TargetRAUC` = untouched ranking.

---

## 5. Research questions — the substance of the round

Prioritize by **expected LB gain × feasibility**. For each: mechanism; the math where non-obvious; why
it is legal (§4); why it is not a §3 repeat; the **single isolated experiment** vs the 0.8955 champion;
**plausible effect size**; and — critically — **how we could screen it offline** given ATC-F1/DIS (§0 #3)
have resolution ~0.010 LB. We would rather be corrected than agreed with.

### Q1 — (HIGH VALUE) Beating an *irreducible* covariate shift, given a rank-only metric
The shift is confirmed large and in the data (§2 B). Frozen foundation features **encode** it (§3).
Importance-weighting the *training* loss is dead (ESS collapse). So: **what actually transfers under a
covariate shift this severe, when the metric only sees the test ranking?** Address specifically, with
sources and the estimator/objective written out:
- **Transductive / test-time adaptation done right.** Pseudo-labeling was dismissed in Round 06 on
  ESS-collapse grounds — but we now have (a) a rank-only metric (so we only need the *order* right on
  confident rows) and (b) an offline screen. Is there a *screenable*, fold-safe pseudo-labeling or
  self-training protocol (e.g. confidence-thresholded, class-balanced, or **CBST/CRST** regularized) that
  is defensible when the confident test rows may themselves be the shifted ones? The EY 2023 organizer
  precedent used pseudo-labeling — is it transferable here?
- **Test-time / batch-norm adaptation, feature alignment (CORAL, sub-space alignment), OT alignment of
  representations** — *without* adding fitted capacity and *without* touching amplitude (§2 C). Which of
  these align the **ranking-relevant** structure rather than just matching moments?
- **DRO over the masking recipe as the group variable.** We have no region labels, but we *do* control
  the masking recipe (window length/start, per-month dropout). Can group-DRO / CVaR-DRO over
  recipe-strata harden the model against the worst strata without the ESS collapse that killed IW?
- For each, give the **offline screen** we would use to gate it before spending a submission.

### Q2 — Foundation models *other than Presto* (the lane is reopened, not exhausted)
Presto failed because a general-purpose S1/S2 pixel-TS encoder faithfully re-encodes the shift (§3). But
that is **one** model. Pretrained models are legal. We want a sourced map of what else exists and, for
each, a **prediction of whether its embeddings would encode or normalize our specific temporal-window
shift** (the property that decides the lane):
- Other **geospatial** pixel/time-series foundation models: **Prithvi (v1/v2), SatMAE / SatMAE++, Clay,
  DOFA, CROMA, Galileo, AnySat** — most are *spatial-patch* models and we have **no spatial
  neighbourhood**; say plainly which are inapplicable for that reason and which have a pixel/TS mode.
- Generic **time-series** foundation models (**MOMENT, Chronos, Mantis, TimesFM, UniTS**) — do any admit
  a 4–6-step multivariate series, and would a non-geospatial pretraining corpus plausibly be *more* shift-
  invariant (no S1/S2 amplitude prior) or *less* (no relevant inductive bias)?
- Concretely: is there a pretrained encoder whose SSL objective would give **adversarial AUC ≈ 0.5** on
  our train/test embeddings (the go/no-go we run for free)? That single property is what we are shopping
  for.

### Q3 — A genuinely different **low-capacity** model class matched to multivariate TS
Model-class changes are the only things that clear the floor (§1), and we have now tried exactly two
(GBDT, Transformer) plus one frozen foundation model. What *other* classes are worth a shot, with
evidence from TS-classification benchmarks (UCR/UEA) under distribution shift:
- **ROCKET / MiniROCKET / MultiROCKET / Hydra** — random convolutional kernels + ridge/logistic. They are
  **shift-agnostic by construction** (random, not learned, kernels) and low-capacity in the fitted head.
  Do they handle our variable 4–6-step masking, and is there evidence they transfer better than learned
  nets under covariate shift? How would we mask/normalize the input so the random kernels see train and
  test at matched density?
- **Shapelet / dictionary methods (WEASEL 2.0, TDE), 1D-CNN / InceptionTime, LightTS** — any that are
  both (a) a real inductive-bias departure from a mean-pooled attention encoder and (b) not high-capacity
  fits on 1817 rows.
- For the top candidate, give the exact input contract for a 12×12 series with sentinel-masked months.

### Q4 — When does ensembling buy **level**, not just variance? (we are testing this now)
Seed-averaging bought variance only (95.1%-correlated members). We are pooling the tied top-cluster
architectures (reltime/nope/l3/xview) and using the **cross-architecture rank-correlation matrix** as the
go/no-go. Give us the theory and the evidence:
- Under a **rank-only** metric and covariate shift, what predicts whether rank-averaging *decorrelated
  equally-good* models gains level? What mean pairwise rank-correlation threshold, and what diversity
  metric (Q-statistic, disagreement, rank-corr) actually predicts ensemble gain **on the shifted test
  set** rather than in-distribution?
- Is there a principled way to build **deliberately decorrelated** members cheaply (different masking
  recipes, different pooling, bootstrapped folds, negatively-correlated inits) that raises diversity
  without lowering individual quality?
- Weighted vs equal blending; rank-average vs logit-average vs probability-average under a rank-only
  metric — which is provably right here?

### Q5 — What did top-5 finishers actually do — in THIS and closely-analogous competitions?
Go read primary sources, not blog summaries. We want mechanisms specific enough to implement.
- **This competition:** the FAO/ITU/Zindi challenge materials, discussion forum, any public write-ups —
  especially anything on how the train/test split and the masking recipe were *constructed* and why (its
  purpose likely hints at what generalizes).
- **Analogous Kaggle/Zindi/DrivenData competitions** on Sentinel-1/2 pixel-time-series land-cover /
  water / crop / pond classification **with a designed train/test distribution shift**: what did the
  top-5 do that the middle of the field did not — architecture, augmentation-to-match-test, TTA,
  pseudo-labeling, ensembling discipline, or a domain-physics feature? Cite the solution write-ups.

### Q6 — The domain physics we may be structurally unable to represent
Is there a **physical signature of aquaculture ponds** in an S1/S2 time series that a 24-channel,
mean-pooled, 2-layer encoder **cannot represent**? Read the aquaculture-pond remote-sensing literature:
fill/drain/stocking/harvest cycles, how ponds are separated from natural water, rice paddies, and salt
pans in S1/S2 series. If there is such a signature, what **minimal, capacity-neutral** change to the
representation or objective would make it representable? Prefer *shape*-based, amplitude-preserving,
channel-**replacing** encodings (§2 C) over additive channels.

---

## 6. Output format we want

For each recommendation:
1. **Name + one-line mechanism.**
2. **Why it is legal** (§4) and **why it is not a §3 repeat**.
3. **The math / the exact change**, implementable — "add a consistency penalty" was only actionable
   because a prior round wrote `L = BCE + λ·Var_k(logit)`.
4. **Expected effect size.** If < ~0.02, state exactly **how we would screen it offline** (ATC-F1/DIS
   have ~0.010 LB resolution; the adversarial-AUC-on-embeddings go/no-go is free). If you cannot name a
   screen, say so — that itself is useful triage.
5. **The single isolated experiment** vs the 0.8955 champion, and whether it needs a submission or is
   screenable for zero.
6. **Sources.** Primary literature and the actual challenge materials over summaries.

Rank everything into: **fund now** (worth a submission this week) · **screen first** (real but needs the
offline gate before a submission) · **park** (interesting, wrong stage) · **rejected on our evidence**.
