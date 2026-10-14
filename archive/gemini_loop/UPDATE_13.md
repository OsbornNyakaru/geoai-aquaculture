# Deep-Research Brief — Round #13 (Claude Research / Gemini Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)
**Date:** 2026-07-28 · **Best public LB: 0.899643 (legal)** · **Deadline:** 2026-08-16 (18 days) ·
**Budget:** ~75 of 100 submissions

**Since Round 12 we removed a rules violation, and our score went UP.** That single change also
invalidated a family of our own conclusions, and this round is about the lane it reopens:
**gradient-boosted trees, specifically CatBoost.**

**Focus this round narrowly.** We do not need another survey. We need Q1–Q4 answered well.

---

## 0. What changed since Round 12 — read this first

**1. 🔴 Our operating point was an explicit rules violation. We fixed it and GAINED score.**
The rules state verbatim (we fetched the page): *"Setting a probability threshold is strictly
forbidden. Your binary target should be based on the default threshold of 0.5."* and *"Zindi will
need the raw probabilities. This will allow the clients to set thresholds to their own needs."*

We had been (a) shifting logits so a chosen quantile landed on 0.5 — a prevalence constant swept
against **leaderboard feedback**, and (b) emitting uniformly-spaced **ranks**, not probabilities, in
the probability column. Both fixed: Platt calibration fit on **training out-of-fold predictions
only**, then a **literal 0.5 cut**, real probabilities in both columns.

```
champion, pinned (illegal)  0.895500      archblend4, pinned (illegal)  0.894643
champion, LEGAL             0.889686      archblend4, LEGAL             0.899643  <- best ever
```

Compliance cost **−0.0058** on the single model — below our own 0.006 "suggestive" threshold. The
pin had been credited with ≈+0.07; that figure was measured in iteration 02 on a **superseded GBDT**
and never re-measured. Inverting our cost model on the observed delta, the pin was adding 104
positives that were **~49% correct — coin flips.**

**2. 🔑 Removing the pin made ENSEMBLING work, and this is the finding that reopens trees.**

```
pinned:   archblend4 − champion  =  −0.0009      pooling bought NOTHING
legal:    archblend4 − champion  =  +0.0100      pooling buys LEVEL
```

Mechanism: the pin overwrote **every member's operating point to a common 0.649**, so pooling could
only average the **ranking** — and at mean rank-correlation 0.9524 there is nothing independent left
to average. A literal 0.5 cut *also* averages the members' **calibration**, where they genuinely
disagree (individual legal positive rates 0.534 / 0.570 / 0.581 / 0.586). The pin was collapsing
that spread to one number and discarding it.

**Consequence: every conclusion we drew about model classes and ensembling was measured under an
operating point that masked calibration quality entirely.** That includes our rejection of GBDTs.

**3. Therefore the GBDT lane is REOPENED, and we want it examined properly.** Our previous verdict —
"a GBDT member dragged the blend by −0.0155, cross-class blending is closed" — was measured under
the pin. Under a literal 0.5 cut, **calibration quality is 60% of the metric**, and trees and neural
nets differ sharply there. Our Transformer needed Platt slopes of **0.59–2.13** to fix (a slope far
from 1.0 means badly mis-scaled raw probabilities).

**4. The public-LB leader (~0.94) uses plain CatBoost** and has said so on the forum: *"Don't blame
the trees — I'm on gradient boosting too (CatBoost-family, nothing exotic), so the model isn't the
bottleneck."* Their named lever is **relative/ratio-style features**. We are at 0.8996 with a
from-scratch Transformer. **The ≈0.04 gap is in the features, not the architecture.**

**5. A feature-research round has already answered part of the ratio question — and corrected it.**
We verified the two central claims arithmetically ourselves (§3). Do **not** re-derive these.

---

## 1. Self-contained problem statement

**Task.** Binary: is this ~10 m cell a managed aquaculture pond?

**Data (supplied only).** Per cell, a **12-month × 12-band** time series: Sentinel-1 SAR (**VH, VV**,
in dB, both present whenever a month is observed) + 10 Sentinel-2 optical bands (individually
missing under cloud). **No lat/lon, no spatial neighbourhood, no image patch, no static covariates.**
Each row is one isolated pixel's time series.

**Sizes.** Train **1,817** rows (after dropping 4 duplicates), ~40.2% positive. Test **1,030**
(public ≈309, private ≈721). True test prevalence believed ≈0.65.

**Metric.** `0.6·F1 + 0.4·ROC-AUC`, over two independently scored columns: `TargetF1` (binary, hard
0.5) and `TargetRAUC` (probability).

**The designed trap.** Train rows are fully observed (12 months). **Test rows expose only a
consecutive 4/5/6-month window** (p ≈ 0.335/0.333/0.332), rest `-9999`, plus extra S2-only cloud
dropout at measured per-month rates. We already expand each train row into K=2 masked views drawn
from the *measured* test distribution — **window matching is implemented; do not propose it.**

**Shift.** Train and test are **different time periods** — this is stated by the organizers.
⚠️ *We have also been assuming different **regions**. On re-reading we cannot find that stated
anywhere; it may be our inference. If you can determine it from the competition materials, say so —
it changes how much we should distrust absolute level features.* Adversarial train-vs-test AUC:
≈0.99 on hand features, **0.8915 on our actual masked model input**.

**Rules.** Supplied data only — no external rasters, DEM, OSM, JRC. **Pretrained models allowed.**
**AutoML banned.** 5 submissions/day, 100 total, 2 designated finalists. Final standing = 65%
private LB + **35% code review of the top 5**.

**Our measurement floor.** Seed-to-seed sd is **0.0191, measured** (same config, seed 42 → 0.8955,
seed 7 → 0.8764). Seed rank-correlation 0.9511, so **averaging cannot fix it**: the variance
reduction factor at 5 seeds is (1+0.9511·4)/5 = **0.961**. **Nothing below ~0.02 is measurable on the
public slice by any construction available to us.** Local CV sits at ~0.975 against an LB of ~0.89
and has been *anti*-correlated with it.

---

## 2. Do NOT re-propose (measured, or already built)

1. **Window/regime matching** — implemented.
2. **Missing-indicator deletion** — indicators alone give adversarial AUC **0.4758**, below chance.
3. **The classical water-index family** — SDWI is *exactly affine* in (VV_dB+VH_dB); AWEI exactly
   linear; EVI ≈ 2.5(NIR−Red) over water; NDWI/MNDWI are 0/0-conditioned over water. Measured
   **−0.075 LB**. A linear model already spans them.
4. **Amplitude removal** — replacing values with within-series temporal rank collapsed OOF
   0.975→0.86. **Persistently low absolute backscatter IS the class signal.** No detrending,
   differencing, instance-norm, or per-window standardization.
5. **Presto** (frozen SSL encoder) — adversarial AUC 0.965–0.976 on its own embeddings; it *encodes*
   the shift.
6. **ROCKET / MiniROCKET** — built; decorrelated (ρ0.87) but −0.040 weak.
7. **Saerens-EM / BBSE / MLLS** — label-shift estimators on a covariate-shift problem; returned a
   prior of 0.44 against a true ≈0.649.
8. **Importance weighting / DANN** — effective sample size collapses at adversarial AUC 0.99.
9. **Adversarial-AUC-driven feature selection** — the shift is *distributed*: max single-band
   adversarial separability 0.59 against a joint 0.89. Band deletion measured −0.0113.
10. **Top-k / precision@k / LambdaRank losses** — our cut sits at k/n ≈ 0.55–0.65, the **middle** of
    the score distribution; those losses target small-k and are the wrong regime (Boyd et al.,
    NeurIPS 2012).

---

## 3. Feature findings already established — build ON these, don't redo them

We ran a feature-engineering research pass and **verified its two load-bearing claims ourselves**:

**(a) The cross-pol ratio is a nuisance canceller, NOT a signal carrier.** From the Sentinel-1 Global
Backscatter Model, class means in dB:

| | VV | VH | CR = VH−VV |
|---|---|---|---|
| water | −18.85 | −26.42 | −7.57 |
| cropland | −11.87 | −19.03 | −7.16 |

Absolute VH separates water from cropland by **7.39 dB**; CR separates the same pair by **0.41 dB**.
**CR retains 5.5% of the contrast** (we computed this). So `VH − VV` must be **added alongside**
level, never substituted for it — which is a real correction to the leader's "use ratios" advice as
we had read it. The ~1.3 dB CR *does* carry is the vegetation-structure direction, i.e. exactly the
**rice-canopy** axis, which is our hardest confuser.

**(b) Only ONE independent SAR ratio exists.** With q = σVH/σVV: `CR_dB = 10log₁₀q`,
`RVI = 4q/(1+q)`, `m = (1−q)/(1+q)` are all **strictly monotone in q** (we verified numerically).
RVI, m-chi and dual-pol "vegetation indices" are CR in disguise — the same algebraic degeneracy we
found in the water-index family. Build one, not four.

**(c) The n-invariance constraint, and a solution to it.** Train rows have 12 observed months, test
rows 4–6, so any aggregate must be **unbiased at every n** or it becomes a shift-carrier by
construction. Class A (safe): mean, median, interior quantiles, std, any **fraction**. Class B
(unsafe): min, max, range, run-lengths, raw counts, autocorrelation, successive differences.
The proposed Class-A substitutes for "amplitude" are **degree-2 U-statistics**:
`GMD = 2/(n(n−1))·Σ_{i<j}|xᵢ−xⱼ|` and `SWITCH_δ = 2/(n(n−1))·Σ_{i<j}1[|xᵢ−xⱼ|>δ]`.

**(d) Rice is separable in time, and the window is long enough.** A rice cycle is ~3.5–4 months
(flood → canopy → fallow) with a 2.7–6.7 dB VH swing; a **4–6 month window is longer than one full
cycle**, so a rice pixel must contain a transition and a pond must not.

**(e) The water threshold literature spans 8 dB** (−19.9 to −27.0 dB in VH), so no single τ is
defensible → use a **CDF profile** `F(τ) = fraction of observed months below τ` at several τ. Also:
African smallholder ponds are **100–1,500 m², often <300 m²**, against a **100 m² pixel** — so
positives are largely dike/vegetation-contaminated mixed pixels sitting *well above* the −26 dB
open-water value.

**(f) No quantitative sub-Saharan African pond-mapping study exists** with Sentinel time series. All
pond-specific quantitative work is coastal East/SE Asian intensive aquaculture. Asian-derived
thresholds and the eutrophication argument are **unvalidated** for our setting.

---

## 4. Research questions

Answer these four. Depth over breadth. **We would rather be corrected than agreed with**, and "no
evidence, this is folklore" is a valuable answer.

### Q1 — 🔑 CatBoost configured for n≈1,800 under heavy covariate shift
The leader reaches ~0.94 with "CatBoost-family, nothing exotic". We need the configuration, not the
endorsement.
- **Ordered boosting** (`boosting_type='Ordered'`) was designed for small data to fight prediction
  shift. Is it right at n=1,800? What is the documented evidence, and what does it cost?
- **Overfitting control at n<2000**: `depth`, `l2_leaf_reg`, learning-rate × iterations, `rsm`,
  `bagging_temperature`, `random_strength`. Give a concrete starting configuration and say which
  parameters actually matter at this scale and which are noise.
- **Missing values.** Our missingness is *structurally different* between train and test by design.
  What does `nan_mode` do, and is the default right when missingness is distribution-shifted? Should
  the `-9999` sentinel be NaN, or its own value?
- Does CatBoost have a **documented** robustness advantage over LightGBM/XGBoost under covariate
  shift, or is that folklore? Controlled comparisons only. (TableShift, NeurIPS 2023, and Grinsztajn
  et al., NeurIPS 2022, are starting points — find others.)

### Q2 — 🔑 Calibration at a LITERAL 0.5 cut (this is 60% of the metric)
- **Are GBDTs actually better calibrated than neural nets?** Niculescu-Mizil & Caruana (ICML 2005)
  found boosted trees are *sigmoid-distorted and NOT well calibrated*; Guo et al. (ICML 2017) found
  modern nets badly *overconfident*. Reconcile these for CatBoost-with-logloss specifically.
- **What happens to a train-fitted calibrator under covariate shift?** (Ovadia et al., NeurIPS 2019,
  and anything newer.) Which model families degrade least?
- **Platt vs isotonic vs beta calibration** at n≈1,800. Note isotonic creates **ties**, and ROC-AUC
  scores ties at half credit — quantify whether that matters at our scale.
- **Does averaging several individually-calibrated probability vectors improve calibration itself**,
  or only variance? We measured +0.010 LB from exactly this and want to know the mechanism.
- **⚠️ The most important sub-question.** Lipton, Elkan & Narayanaswamy (ECML-PKDD 2014) prove that
  for well-calibrated probabilities the F1-optimal threshold is **half the optimal F1** — at F1≈0.87
  that is ≈0.435, so a perfectly calibrated model at a literal 0.5 cut **systematically
  under-selects**. We may not move the threshold. Is there a **legitimate training-side** way to make
  a well-calibrated model's 0.5 cut land nearer the F1 optimum — class weights, a modified objective,
  a different loss — that is *not* threshold tuning in disguise? Elkan (IJCAI 2001) proves
  reweighting and threshold-shifting are equivalent for calibrated models, which suggests this may be
  **impossible in principle**. **Give us the honest answer, including if it is "you cannot".**

### Q3 — GBDT vs sequence model on single-pixel Sentinel time series
Our Transformer sees the raw 12×12 series; a GBDT must see aggregates, which discards ordering.
- Is there evidence on which wins for **pixel-wise** (no neighbourhood) crop/water/land-cover
  classification from Sentinel time series, at n in the low thousands?
- Given §3(c), aggregates must be n-invariant. **Does that constraint actually favour trees**, since
  a fixed-length aggregate vector is naturally window-length-agnostic while a sequence model must
  handle variable-length masked input?
- Our GBDT scored ≈0.885 standalone under the pin and its blend lost −0.0155. **Both were measured
  under the pin.** What would you predict changes under a literal 0.5 cut, and what is the cheapest
  experiment that distinguishes "trees were genuinely worse" from "trees were penalised by an
  operating point that erased their calibration advantage"?

### Q4 — Endgame under a 0.0191 noise floor
- We designate exactly **2 finalists**. With seed sd 0.0191, a 309-row public slice and a 721-row
  private slice, is the documented best practice to pick the two **best public**, the two most
  **different**, or the two **lowest-variance**? Find cases where the choice demonstrably mattered.
- With ~75 submissions and 18 days but a floor of ~0.02, **what is actually worth spending them on?**
  Argue for or against the position that further LB probing has negative expected value and the
  remaining effort belongs in the 35% code review.

---

## 5. Output format

Per recommendation: **(1)** name + one-line mechanism · **(2)** why it is legal under §1 and not a §2
repeat · **(3)** the exact change, implementable (concrete parameter values or a formula) ·
**(4)** which column it targets — `TargetF1` (the set) or `TargetRAUC` (the ranking) · **(5)**
expected effect size, and if it is below ~0.02, how we would screen it **offline** · **(6)** the
single isolated experiment · **(7)** sources, primary over summaries.

Rank into: **fund now** · **screen first** · **park** · **rejected on our evidence**.

**On confidence.** In an earlier round, 6 of 8 agents' top recommendation was already implemented in
our repo and 3 of 8 converged on a deletion target that measured *below chance*. Convergence between
sources is not evidence — it usually means they read the same literature. **If the honest answer is
"you cannot beat this from the data you have", say that.** Given §1's ceiling and §3(f)'s missing
literature, that is a live possibility, and knowing it is worth more to us than a plausible lane that
costs a week.
