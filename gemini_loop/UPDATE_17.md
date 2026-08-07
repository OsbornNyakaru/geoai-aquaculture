# Deep-Research Brief — Round #17 (Claude Research / Gemini Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)
**Date:** 2026-08-06 · **Best public LB: 0.906492 (legal, single model)** · **Deadline:** 2026-08-16 (10 days) ·
**Budget:** ~66 of 100 submissions · **Finalists to designate: 2**

> **Posture of this brief — read before answering.**
> **Feature engineering is NOT closed here.** Earlier internal notes called it "exhausted"; that is an
> *inference from a handful of noisy experiments*, not an established fact, and it may be wrong. Treat every
> "we parked this" below as a **rebuttable hypothesis**, not a wall. **Your job is to go through all the
> relevant verified literature thoroughly** — read the papers, not just their abstracts — and extract
> concrete, instrumental methods we can implement. **If a well-supported paper points at a direction we
> already tried and shelved, say so explicitly and make the case to re-open it — we will re-explore any
> direction the literature substantiates, including ones we thought were closed.** We would rather re-run
> a "dead" idea done *properly* than miss a real lever because of an under-powered earlier test.

---

## 0. Why our own "closed doors" are suspect (so challenge them)

Two facts make almost all of our negative results weaker than they look:

1. **Seed variance = 0.019 (measured); public LB resolution ~0.013 on ~309 rows.** *Most* of our "it
   hurt / it didn't help" verdicts were **single-seed, single-submission** reads with effect sizes *inside*
   that noise band. A single-seed −0.006 is not evidence a feature is bad — it is one draw from a
   distribution ~0.019 wide. Several parked ideas were never seed-averaged, never re-parameterized, never
   combined, and never tested as *replacements* rather than *additions*.
2. **OOF is anti-correlated with the LB here**, so we cannot pre-screen ideas offline the normal way, which
   means we have spent submissions cautiously and closed lanes fast to conserve budget. That caution is
   reasonable but it means **"closed" often just means "tested once, weakly, and shelved."**

So: bring the full weight of the literature. Where a method has strong, transferable published evidence,
we want it — even if our one-shot attempt looked flat.

---

## 1. Self-contained problem statement

Binary per-pixel classification: is this ~10 m cell a managed aquaculture pond? Input = a **12-month ×
12-band** time series per cell: Sentinel-1 SAR (**VH, VV** in dB, always co-present when a month is
observed) + 10 Sentinel-2 optical bands (individually missing under cloud). **No lat/lon, no image patch,
no static covariates** — each row is one isolated pixel's multivariate time series. Train **1,817** rows
(~40% positive), test **1,030** (public ~309 / private ~721). **True test prevalence believed ~0.65
(label shift).** Metric = **0.6·F1 + 0.4·ROC-AUC**, two columns: `TargetF1` (binary at a **hard 0.5 cut** —
threshold tuning is FORBIDDEN by the rules) and `TargetRAUC` (probability, for AUC). Only supplied data;
no external data; no AutoML; seeded/reproducible; legal calibration = Platt on train OOF + literal 0.5.

**The masking trap (shapes which features are safe).** Train series have ~12 months; **test series have
only 4–6 consecutive months** (correlated dropout). Statistics that are *biased at short n* encode "how
many months" (a shift-carrier) instead of "what kind of pixel." Safe (n-invariant): mean, median, interior
quantiles, fractions/CDF values, U-statistics, L-moments. Risky: min, max, range, counts, run-lengths,
argmax-timing, per-calendar-month columns. **Domain shift is strong: adversarial-AUC ≈ 0.89** (train vs
test on feature values).

**Current model.** From-scratch Transformer: per-timestep `Linear` → masked **mean-pool** over observed
months → small head. Best result = adding one binary channel `1[VH_dB < −21]` (a "permanence" / empirical-
CDF coordinate) → **LB 0.906492**. The public leader (~0.94) reportedly uses plain CatBoost with
relative/ratio features.

---

## 2. Confirmed observations vs rebuttable interpretations (keep these separate)

**Confirmed (reproduced, or seed-robust):**
- `1[VH<−21]` permanence channel → 0.906492; seed distribution {42:0.9065, 29:0.9007, 13:0.8917,
  21:0.8786}, **5-seed avg 0.899882** (a *real* +0.0055 pooling gain over the member mean).
- `champion_archblend4` (4-architecture calibrated pool) = 0.899643.
- Feature-selection on the permanence axis was monotone: 1τ (0.9065) > 4τ (0.9016) > 6τ (0.8987).

**Interpretations we drew — TREAT AS REBUTTABLE, challenge with literature:**
- *"The model only sees affine functions of the temporal mean, so only nonlinear-in-the-mean features
  help."* (Because Linear and mean-pool commute.) → This is exactly why we think ratios/differences died.
  **But it is contingent on the mean-pool architecture** — see Q1.
- *"Adding a 2nd feature channel overfits the shift (capacity sweet spot at 25 channels)."* From iter34,
  three single-seed additions scored −0.006/−0.015/−0.018. **Single-seed, additions-only, never
  seed-averaged, never as replacements.**
- *"Trees/GBDT are dead here"* (our CatBoost: OOF 0.995 → LB 0.70). One implementation, one lane.
- *"Prior-shift correction (Saerens) is unsafe under this shift."* We since built an offline gate that
  says it's decidable, not dead.

---

## 3. The feature-engineering lane is OPEN — help us mine it

We want features, and we want them in forms that (a) respect the masking trap (n-invariant) and (b) either
survive our current model or motivate a model change (Q1). Bring the literature on **all** of these, ranked
by expected value with concrete definitions and τ/threshold values:

**3a. SAR aquaculture / water / rice discrimination features.** What are the published, operationally
validated Sentinel-1 features that separate managed ponds from open water, flooded rice (the hardest
confuser), and wet soil? We know the −21 dB permanence line and temporal-percentile persistence (Ottinger).
What *else* has won or shipped: temporal dispersion (ponds stable, rice swings 6–8 dB), dike double-bounce
signatures, VH/VV joint structure, coherence/texture proxies computable from GRD dB, seasonality-invariant
rice detectors? Give the exact statistic and its physical basis.

**3b. Short-time-series classification features (n=4–6).** From the time-series-classification literature
(catch22/hctsa, ROCKET/MiniRocket kernels, shapelets, symbolic SAX/BOSS, tsfresh), which features are
**both discriminative and n-invariant/robust at 4–6 samples**? We want the specific safe subset, not the
whole battery. Are there threshold-crossing / level-set / order-statistic features with a track record on
short series that we have not tried?

**3c. Optical (Sentinel-2) features that survive cloud gaps.** NDVI/NDWI/MNDWI/NDRE etc. are ratios; as raw
per-month inputs they may be affine-dead in our model, but as **thresholded/CDF/occupancy features**
(`mean_t 1[NDVI>v]` = ever-green fraction) they are new coordinates. Which optical-derived features best
reject rice/vegetation and best confirm open water, and how should they be encoded to survive both the
cloud gaps and the mean-pool?

**3d. Cross-sensor (SAR×optical) fused features.** Joint-occupancy gates (`1[VH<τ]·1[NDVI<v]`), dual-sensor
permanence, etc. What is the published evidence on SAR×optical fusion specifically for aquaculture / paddy
separation, and what fused statistic is most surgical?

**3e. "Relative/ratio" features (the leader's stated lever), translated to survive mean-pooling.** The
leader named relative/ratio features. Raw ratios appear affine-dead in our model — but CDFs/thresholds of
ratios, ratio *variances*, and within-series relative positions are not. Enumerate the ratio-derived
features with the best precedent and give their mean-pool-surviving encodings.

---

## 4. Paths we PARKED and our stated reason — challenge each; we WILL re-open any the literature backs

For each: **[what we did] → [our reason for parking] → [why that reason may be wrong].** If a paper
supports the path, tell us how to re-test it properly (seed-averaged, re-parameterized, as a replacement,
softened, or with a model change).

| Parked path | Our stated reason | Why we may be wrong / how to re-test |
|---|---|---|
| **Second feature channels** (pond-band occupancy, VH², VV-permanence) | single-seed additions each −0.006/−0.015/−0.018 → "capacity overfit" | all single-seed inside the 0.019 floor; never seed-averaged; never as **replacements** at constant width; never softened. Re-test properly. |
| **VH−VV / ratios / SDWI / RVI** | "affine-spanned → the mean-pool can't use them" (VH−VV scored −0.023 once) | contingent on mean-pool (Q1); and untested as **thresholds/CDFs of the ratio**, which are NOT affine. |
| **Trees / GBDT (CatBoost)** | OOF 0.995 → LB 0.70, one run | one implementation; no shift-aware regularization, no n-invariant feature set, no proper CV design. The leader is on CatBoost — so a *correct* tree setup may be the biggest miss. |
| **Multi-τ / dense CDF profile** | monotone 1τ>4τ>6τ (single-seed) | is the axis truly saturated, or was the profile just adding correlated capacity that a bottleneck/regularizer would fix? |
| **Learnable / adaptive / per-window τ** | "drifts to a train statistic; Otsu ill-posed at n=5" | is there a *shift-robust* learnable-threshold formulation (straight-through, hard-concrete, anchored) with published OOD evidence? |
| **Foundation-model embeddings (Presto etc.)** | adv-AUC 0.97, magnitude-dependent → collapses | are there SSL/foundation features or fine-tuning recipes that are explicitly scale/shift-invariant and transfer here? |
| **Test-time adaptation / pseudo-labeling** | "collapses at adv-AUC 0.9" | which specific safe variants (normalization-only, conservative self-training with abstention) have published gains at high shift? |
| **Prior-shift / Saerens correction** | earlier called "unsafe" | we now have an offline gate that decides it; confirm the method and the safest estimator. |

---

## 5. If the model is the bottleneck, tell us how to change it (so more features become usable)

Our "affine blind spot" is a property of **mean-pool over a per-timestep linear map**. If that is what caps
feature usefulness, the highest-value move may be a *small architecture change* that lets richer features
pay off:

- **Q1 (pooling / architecture).** What pooling or attention design keeps the n-invariance and shift-
  robustness of masked mean-pool but is **not** reducible to an affine function of the temporal mean — so
  that variance, covariance, cross-band, and ratio features become usable — without blowing up capacity at
  n≈1800 under adv-AUC-0.89 shift? (mean+std pooling, attention pooling, set-transformer/Deep Sets with a
  nonlinear encoder, moment pooling, distribution/quantile pooling.) Cite OOD/few-sample evidence.
- **Q2 (capacity control that *enables* features).** Low-rank input bottleneck, decoupled weight decay,
  channel dropout, SAM/flat-minima, SWA/SWAD — which of these has published evidence that it lets a model
  *absorb an informative-but-shifting feature* that plain ERM overfits? We want the regularizer that turns
  a −0.006 feature into a net win, with the mechanism.

## 6. The other live levers (keep, don't drop feature work for them)

- **Q3 (legal operating point).** Confirm the Saerens–Latinne–Decaestecker / MLLS prior-shift correction to
  an estimated π_t (never LB-tuned), gated by an offline mixture goodness-of-fit test; the F1 upside vs the
  0.013 resolution; the safest π_t estimator (BBSE vs MLLS vs a physics-anchored estimate from the VH-
  permanence marginal); and the pure-label-vs-conditional-shift diagnostic.
- **Q4 (robust averaging).** SWA/SWAD weight-averaging vs seed/snapshot ensembling for a shallow
  Linear+mean-pool head under shift — does weight-space averaging buy *level* (we just measured a real
  +0.0055 level gain from permanence seed-averaging, so this lane is live), and the best pooling operator
  for a hard-0.5 F1 metric.
- **Q5 (finalists).** Which 2 of {`c_perm_single` 0.906492 lucky-seed, `champion_perm_seedavg5_st` 0.899882
  robust, `champion_archblend4` 0.899643 decorrelated} minimize expected private-721 regret under seed-var
  0.019 and π≈0.65 (final rank = the better of the two on private → E[max], paid for mean AND decorrelation)?

---

## 7. How we want the literature handled

- **Read the papers, extract the mechanism and the exact recipe** (features, thresholds, hyper-parameters,
  the estimator, the loss) — not just a citation. We will implement from your description.
- **Verify every citation** (arXiv ID / DOI / venue). Do not invent references; if unsure, say so.
- **Prefer transferable, empirically validated methods** over theory-only proposals, but explain the
  mechanism so we can judge fit to our regime (tiny-n, short-window, high-shift, rank+F1 metric).
- **Explicitly flag conflicts with our parked reasons.** If the evidence says a shelved path (§4) should
  work, name it, give the strongest supporting paper, and specify the proper re-test. **We will run it.**
- **Rank everything by expected value in our regime**, and remember only LB effects > ~0.013 are resolvable
  — so favor levers with a credible path past that floor, and flag which are worth a submission vs an
  offline check.

## 8. Reading list to start from (verify, extend, and go beyond)

SAR/optical mapping: Ottinger 2017 (rs9050440) & 2021 (rs13244851); Xing 2018 Dongting (PeerJ e4992,
τ=−21.56); Tsyganskaya 2018 (rs10081286); Mekong rice (rs13050921); Duan 2020 ponds-stable (rs12183086);
Zindi Farm Pin winners (median pooling). Short-TS features: catch22 (Lubba 2019), ROCKET/MiniRocket
(Dempster 2020/2021), tsfresh, SAX/BOSS, shapelets. Tabular/trees under shift: Grinsztajn 2022
(arXiv:2207.08815), TableShift (2312.07577), McElfresh 2023. Shift theory: Ben-David 2010; DomainBed
(2007.01434); Xu 2025 invariant crop (2509.03497); Sagawa 2005.04345; Nagarajan 2010.15775.
Architecture/pooling: Deep Sets (1703.06114), Set Transformer (1810.00825). Capacity/averaging: low-rank
attention (2002.07028), SWA (1803.05407), SWAD (2102.08604), dropout=adaptive-L2 (1307.1493). Operating
point: Saerens 2002; BBSE (1802.03916); MLLS (1901.06852). Prior briefs in this repo: UPDATE_15.md,
UPDATE_16.md; digests in gemini_loop/. **Go beyond this list** — it is a starting point, not a boundary.

## 9. Deliverable

A prioritized, mechanism-level answer covering §3 (open feature lane), §4 (which parked paths to re-open,
with the proper re-test), §5 (model/pooling change if that's the real cap), and §6 (operating point,
averaging, finalists). For each recommendation: the mechanism, ≥1 verified paper, the concrete recipe
(feature/threshold/hyper-parameter values), an expected LB effect vs the 0.013 floor, and whether it is a
submission or a free offline check. End with a single **ranked build order for the remaining 10 days** — and
call out the one or two ideas most likely to be *underrated* because we shelved them too early.
