# RESEARCH_14 — Web-grounded pass: feature engineering + what won in analogous comps

**Date:** 2026-07-28 · **Author:** Claude (live web search, not deep-research) · **Purpose:** while the
UPDATE_13 CatBoost deep-research answers (Q1–Q4) are still out, this is a parallel, source-cited pass
on the two things the user asked for: (a) the best feature-engineering techniques for this problem
type, and (b) what actually won in the most similar Kaggle/Zindi competitions. Every claim below is
tied to a source in §7. Read against UPDATE_13 §2–§3 (what we've already killed) before acting.

---

## 0. TL;DR — five things the evidence says

1. **The most analogous WINNER (Zindi AgriFieldNet, Sentinel-2 crop, n≈small) is CatBoost+LGBM+XGB on
   temporal aggregates of ~15 spectral indices.** It is almost exactly the lane we are about to test.
   This is strong external validation of the CatBoost direction — not folklore.
2. **A weaker tree model with *different bias* earned its place in the winning ensemble** (Farm Pin:
   Random Forest added ~1% on top of stronger CNNs "despite performing worse"). This is direct
   evidence for our iter29-motivated question: a competent-but-different member can help the blend.
3. **The winners used `median` AND `max` temporal aggregates. We can copy `median`, NOT `max`.** They
   had full time series; we have the 4–6-month masking trap, so `max`/`min`/`range` are n-dependent
   shift-carriers (UPDATE_13 §3c). **This is the one adaptation that matters most.**
4. **The aquaculture SAR literature (Ottinger et al.) is unanimous on the exact features:** per-pixel
   `median`, `std`, and `10/25/50/75/90` percentiles of **VH and VV**, with **VH the primary
   water/land separator** (bimodal), and the **median = permanence** of low scatterers = the pond
   signal. Our percentile features already encode most of this; the CDF-profile is the new part.
5. **A live tension to resolve, not assume:** we killed the water-index family at −0.075 LB — but that
   was our *Transformer on raw bands*, where the indices are affine-spanned. **Tree models split on
   explicit index features directly**, and every crop winner fed them dozens. "Indices are dead" may
   be a Transformer-only result. The CatBoost lane must re-test indices, not inherit the veto.

---

## 1. What won in the most analogous competitions (Sentinel time series → crop/land)

### 1a. 🥇 Zindi/Radiant Earth **AgriFieldNet India** (Sentinel-2, pixel/field crop classification) — the closest analog
Winning "gold" solution (`model_ecaas_agrifieldnet_gold`, Team Starlink):
- **Model:** *"1 U-Net + 8 Gradient Boosting Trees"* — the 8 GBDTs are **CatBoost, LightGBM, XGBoost
  variants**. Ensemble = **weighted geometric mean**.
- **Features:** raw bands (B01–B12) **plus a large derived-index block** — NDVI `(B08−B04)/(B08+B04)`,
  GLI, **CVI = `(B08/B03)·(B04/B03)`** (a *product of ratios* — note the arithmetic-combination
  style), SIPI, S2REP, CCCI, HUE, RENDVI, RECI, EVI, EVI2, NDWI, NPCRI.
- **Temporal aggregation:** **`median` and `max`** of every feature across the time series.
- **Takeaway for us:** the winning recipe on a near-identical task is *tree-ensemble on
  temporal-aggregated spectral indices*. It directly supports the CatBoost lane. **But their `max` is
  illegal for us** (see §0.3 / UPDATE_13 §3c) — swap it for n-invariant statistics.

### 1b. 🥇 Zindi **Farm Pin Crop Detection** (Sentinel-2 time series, South Africa) — the "different-bias member" evidence
Winners (Fadahunsi/Ponikar/Baas): *"four base models + one 2nd-layer stacking model … two 3D-CNNs
(best performers) and two Random Forests. These perform worse than the CNNs, but because they have
**different bias**, they improve the final ensemble by ~1%."* Feature style: vegetation indices with
**mean and std across time**.
- **Takeaway:** this is the cleanest external analog to our iter29 finding. A weaker model of a
  different class **earns its ensemble slot through decorrelated errors**, not through standalone
  strength — *provided it is competent enough*. Our ROCKET/pin-era GBDT were too weak or measured
  under the pin; a properly-built legal CatBoost is the Farm-Pin "Random Forest" role.

### 1c. Farm Pin 8th (simongrest, documented): NDVI + Euclidean-norm-of-bands, cloud handling via
`S2PixelCloudDetector` + **LinearInterpolation** to fill gaps + boundary fill-out. Confirms the
standard cloud-gap pipeline (we handle masking differently, via view augmentation — do not change).

---

## 2. Aquaculture-specific SAR feature literature (Ottinger, Stiller, et al.)

Consistent across the MDPI *Remote Sensing* body of work on Sentinel-1 pond mapping:
- **Feature set:** per-pixel **median, std, and 10/25/50/75/90 percentiles** of **VH and VV** over a
  dense annual time series.
- **VH > VV** for water/land separation — VH backscatter is **bimodal** (clean water peak + land
  peak), VV less so. (Matches our own finding that VV is the shift-carrier and VH the signal.)
- **The median is the permanence detector:** it isolates *permanent low scatterers* (smooth pond
  water) from *permanent high scatterers* (dikes/dams). Ponds = persistently low VH.
- **Takeaway:** our existing percentile features (`sar_percentiles: 10/25/50/75/90 of VH,VV`) are
  exactly the literature recipe — but as n-invariant interior quantiles they are **legal**, whereas
  `min`/`max`/`range` (also in some of our feature configs) are **not**. The genuinely new,
  literature-anchored, legal feature is the **CDF profile** `F(τ) = fraction of observed months with
  VH < τ` at several τ ∈ [−22, −19] dB (RESPONSE_13's permanence indicator, but as a multi-τ profile
  it is strictly richer and still Class-A / n-invariant).

---

## 3. CatBoost configuration for n≈1,800 under covariate shift (partial — deep-research pending)

Grounded so far (full answer awaits the UPDATE_13 Q1 deep-research):
- **Ordered boosting (`boosting_type='Ordered'`) is documented as *especially effective on smaller
  datasets*** — it is a permutation scheme that fights *prediction shift / target leakage* and reduces
  bias, at higher compute cost. At n≈1,800 this is the regime it was built for. (CatBoost paper;
  UWaterloo statwiki.) Its multi-permutation encoding also acts as **data augmentation** → more
  overfit-resistant, which is what we want under shift.
- **Small-data starting config** (to tune with Optuna): `depth 4–6`, `learning_rate 0.01–0.03` with
  correspondingly more `iterations`, non-trivial `l2_leaf_reg`, plus `random_strength` and
  `bagging_temperature` for regularization; `rsm` (feature subsampling) low-ish given many correlated
  features. Tune — most of these matter more than depth at this scale.
- **Missingness (open question, high value):** our `-9999` masking is *structurally different between
  train and test by design*. Whether to pass NaN (CatBoost's `nan_mode`) vs a sentinel is exactly the
  kind of thing that interacts with the shift — flagged as UPDATE_13 Q1 for the deep-research.

---

## 4. Grandmaster tabular playbook + covariate-shift tooling

- **Feature engineering that wins is arithmetic-combination-heavy:** *ratios and products* of numeric
  columns, plus cluster features (GMM). The AgriFieldNet CVI index above (`(B08/B03)·(B04/B03)`) is
  literally a product-of-ratios — the grandmaster pattern applied to bands. This is the concrete form
  of the LB-leader's "relative/ratio features" hint, and it is what our from-scratch Transformer
  never got to exploit as explicit inputs.
- **Winning stacks are multi-level and many-membered** (e.g. 33 models → meta-model → weighted
  geo/arith-mean), with **hill-climbing** weight search. Our 4-member equal-weight blend is primitive
  by comparison — once we have a competent CatBoost, hill-climbing the blend weights is a documented
  lever (and iter29 says equal-weighting weak members is exactly the failure mode).
- **Adversarial validation** (train-vs-test classifier) has two uses we should treat differently:
  (a) **per-feature shift detection** — we already built this (`tools/shift_audit.py`); (b)
  **sample reweighting** — train the model weighting/keeping the *test-like* train rows. We rejected
  (b) because ESS collapses at adv-AUC 0.99 — **but that was on hand-features; on our actual masked
  model input adv-AUC is 0.8915**, where reweighting is far more viable. Worth reconsidering as
  *soft* instance weights for the CatBoost, not hard filtering.

---

## 5. 🔑 THE ADAPTATION THAT MATTERS — why we cannot copy-paste the winners

Every winning solution above used **full-length time series** (crop comps: 8–11 clean time points per
field). They could use `max`, `min`, `range`, argmax-timing, run-lengths freely. **We cannot** — the
competition's designed trap is that **train sees 12 months, test sees a consecutive 4–6-month
window**, so any aggregate that is not unbiased at every window length becomes a *shift-carrier by
construction* (UPDATE_13 §3c). Concretely:

| winner used | we can use? | legal substitute |
|---|---|---|
| temporal **median**, mean, std | ✅ (n-invariant) | keep as-is |
| interior **percentiles** p25/p50/p75 | ✅ | keep as-is |
| **max / min / range** | ❌ n-dependent | **CDF fraction** `F(τ)`, GMD, interior IQR |
| argmax **timing**, run-lengths | ❌ Class-B | fraction-of-months-in-state |
| spectral **indices** (NDVI, CVI…) per-time then aggregate | ⚠️ re-test | only ADD, never replace level (§0.5) |

**So the CatBoost recipe for us = {the winning recipe} ∩ {n-invariant aggregates}:** CatBoost on
per-band **mean / median / std / p25 / p75**, plus a small set of **non-degenerate derived channels**
— `VH−VV` cross-pol ratio *added alongside* level (UPDATE_13 §3a: it carries the 1.3 dB rice-canopy
axis, our hardest confuser), the **CDF permanence profile** `F(τ)` on VH, and one or two independent
S2 indices (NDVI) — each reduced by the *same* n-invariant aggregates. No `max`, no timing.

---

## 6. Concrete recommendation for the CatBoost lane (iter30 candidate, when we run it)

1. **Model:** CatBoost, `boosting_type='Ordered'`, small-data config (§3), calibrated on OOF
   (Platt/isotonic) — calibration is 60% of the metric now, and it's where trees may beat our net.
2. **Features:** per-band n-invariant aggregates (mean/median/std/p25/p75) over the observed window,
   built from: the 12 raw bands **+** `VH−VV` **+** VH CDF-profile `F(τ)` at τ∈{−22,−21,−20,−19} **+**
   NDVI. Explicitly **no max/min/range/timing**. This is the winner recipe, made legal.
3. **Two uses, both worth one submission each under the legal cut:** (a) **standalone** legal CatBoost
   — does it land near the Transformer's 0.8897, confirming "the model isn't the bottleneck"? (b)
   **blend member** with `champion_archblend4` — the Farm-Pin "different-bias RF" role; iter29 says
   this is the *only* remaining pooling gain, and it needs a competent+different member, which a
   properly-built CatBoost is (unlike ROCKET/pin-era GBDT).
4. **Screen with:** adversarial-AUC (free) + a regime-matched CV (free) — **NOT ATC-F1** (out-of-
   family, and defined against the deleted operating point; see UPDATE_13 §0.4 / iter26).
5. **Re-open the indices question deliberately:** run one CatBoost with, one without the spectral
   indices. The −0.075 index verdict was Transformer-only; trees split on indices directly. This is
   cheap and settles whether the leader's "ratio features" lever is real *for a tree*.

**Honest caveat (kept, per our own anti-convergence rule):** the crop winners were classifying among
*many crop types with full series and no designed shift*. Our task is binary, masked, and
period-shifted. The recipe transfers in *form* (tree + aggregated indices) but the *shift* is ours
alone — external evidence cannot tell us the indices survive it. That is why steps 4–5 are screens,
not commitments.

---

## 7. Sources

- Zindi AgriFieldNet winner (CatBoost/LGBM/XGB + indices, median/max): [radiant.earth blog](https://radiant.earth/blog/2023/04/behind-the-agrifieldnet-model/) · [model_ecaas_agrifieldnet_gold docs](https://github.com/radiantearth/model_ecaas_agrifieldnet_gold/blob/main/docs/index.md) · [silver (weighted trees, imbalance)](https://github.com/radiantearth/model_ecaas_agrifieldnet_silver)
- Zindi Farm Pin winners (RF "different bias" +1% in ensemble; mean/std of indices): [Meet the Winners](https://zindi.africa/learn/meet-the-winners-of-the-farm-pin-crop-detection-challenge) · [simongrest 8th-place README](https://github.com/simongrest/farm-pin-crop-detection-challenge/blob/master/README.md)
- Aquaculture SAR features (median/std/percentiles of VH/VV; VH bimodal; median=permanence): [Ottinger et al., Large-Scale Assessment, RS 2017](https://www.mdpi.com/2072-4292/9/5/440) · [Mapping Aquaculture Ponds Asia S1+S2 time series](https://www.researchgate.net/publication/356602807_Mapping_Aquaculture_Ponds_for_the_Coastal_Zone_of_Asia_with_Sentinel-1_and_Sentinel-2_Time_Series) · [Nation-Scale S1 GEE, RS 2020](https://www.mdpi.com/2072-4292/12/18/3086)
- XGBoost parcel-based Sentinel-2 crop classification (+3% OA, +0.04 F1): [MDPI RS 15/20/5009](https://www.mdpi.com/2072-4292/15/20/5009)
- CatBoost ordered boosting for small data / prediction shift: [CatBoost paper (NeurIPS 2018)](https://www.researchgate.net/publication/328576065_CatBoost_gradient_boosting_with_categorical_features_support) · [UWaterloo statwiki summary](https://wiki.math.uwaterloo.ca/statwiki/index.php?title=CatBoost:_unbiased_boosting_with_categorical_features)
- tsfresh / catch22 automated TS features: [tsfresh paper](https://www.researchgate.net/publication/324948288_Time_Series_FeatuRe_Extraction_on_basis_of_Scalable_Hypothesis_tests_tsfresh_-_A_Python_package) · [FreshPRINCE pipeline](https://arxiv.org/pdf/2201.12048)
- Kaggle grandmaster tabular FE (ratios/products, GMM, multi-level stacking, hill-climbing): [NVIDIA: FE first place](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/) · [NVIDIA: stacking first place](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/)
- Adversarial validation (shift detection + test-like reweighting): [FastML part one](https://fastml.com/adversarial-validation-part-one/) · [Zak Jost](https://blog.zakjost.com/post/adversarial_val/) · [dataset-shift credit-scoring, arXiv 2112.10078](https://arxiv.org/pdf/2112.10078)
