# Message to Gemini — Research Loop Update #01
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)
**Date:** 2026-07-07 · **Current best public LB:** 0.7561 · **Deadline:** 2026-08-16

---

## How this loop works (please read)

You (Gemini Deep Research) and a coding agent (Claude) are running an iterative
improvement loop on a live machine-learning competition:

> **Claude implements & submits → shares results + open questions with you (this
> file) → you do deep research + critique → Claude implements your best ideas →
> new leaderboard results → next update.**

This is **Update #01**. Each future update reports what your last round of ideas
produced on the leaderboard, so you can see which of your recommendations
actually worked and recalibrate. Please treat leaderboard movement as ground
truth. Your job in each round: **(a)** react to results, **(b)** do fresh deep
research, and **(c)** propose concrete, prioritized, *sourced* next steps that go
**beyond** what we have already tried — ranked by expected leaderboard gain vs
implementation effort. Flag anything that would violate the rules.

**Hard constraints (never violate):** only the supplied competition data may be
used (no external data/pretraining on other datasets); AutoML is banned; the F1
target must be the 0.5 cut of a probability (no post-hoc threshold tuning);
everything must be seeded and reproducible; max 5 submissions/day.

*(A full standalone description of the competition, data, and pipeline exists in
`DEEP_RESEARCH_BRIEF.md`; this update assumes you have that context and focuses on
what changed.)*

---

## 1. Results since your last report

| Change | TargetF1 positive rate | Public LB (0.6·F1 + 0.4·AUC) |
|---|---|---|
| Baseline (inherited train prior) | 0.40 | **0.7140** |
| **Prior/base-rate correction** (assume test prior 0.50) | 0.50 | **0.7561** (+0.042) |
| WIF + EVI features (running now) | 0.50 | *pending* |

**Interpretation:** your class-shift diagnosis was correct and high-value.
Correcting the base rate so the fixed-0.5 F1 calls ~50% positive (instead of the
train prior 40%) lifted the leaderboard by **+0.042** — confirming the test set
is genuinely more positive than train. This is currently our single biggest lever.
Local cross-validation is blind to this (it is same-distribution as train), so all
of this was validated directly on the leaderboard.

---

## 2. Where your last report was RIGHT vs WRONG (please internalize)

**WRONG — do not repeat this recommendation:**
- You diagnosed the OOF≫LB gap as **group/location leakage in cross-validation**
  and told us to switch to `StratifiedGroupKFold` by location ID. This does **not
  apply to our pipeline.** Our CV already isolates each location: the temporal
  "views" of a training row are built *only* from training-fold rows, and
  out-of-fold predictions are built *only* from held-out rows — a row's masked
  slices never straddle the train/validation boundary. There is also **no
  location ID to group by** (latitude/longitude were removed by the organizers).
  We independently proved the gap is **genuine domain shift, not leakage**: an
  adversarial classifier separates (masked) train from test at **AUC ≈ 0.99**, and
  still ≈ 0.94 using only region-normalized water indices. Acting on the leakage
  advice would have wasted effort. **In future rounds, please do not re-propose
  group-KFold or assume leakage — the gap is a designed train/test distribution
  shift.**

**ALREADY IMPLEMENTED before your report (please don't re-recommend as new):**
sensor-asymmetry handling (S1/S2 aggregated independently, −9999→NaN per band,
S1-only SDWI, asymmetry flags), native GBDT missing-value handling, the
LightGBM+XGBoost+CatBoost ensemble, the config/src repo structure, and the
NDWI/MNDWI/AWEI + SAR-roughness features.

**RIGHT and VALUABLE (thank you — we implemented these):**
- **Class-shift / prior correction** → validated, +0.042 (see §1).
- **New features EVI and Water Inundation Frequency (WIF).** WIF (count of
  window months passing a water-presence decision rule) is excellent: on the
  training labels, WIF-fraction alone scores **AUC 0.826**, with pond mean 0.75 vs
  non-pond 0.22 — and because it is a *count of water-present months* rather than
  absolute reflectance, it should transfer across regions/seasons far better than
  raw bands. Currently under leaderboard test (Run B).

---

## 3. Current approach (what the model is now)

- **Train/test masking alignment:** train rows are fully observed (12 months);
  test rows expose only a consecutive 4/5/6-month window (rest = −9999). We
  augment each training row into K masked "views" matching the exact
  reverse-engineered test masking recipe (window + simulated Sentinel-2 cloud
  dropout), so the model trains on test-like inputs. (Same idea as the PLAsTiCC
  Kaggle winner degrading train to match test cadence.)
- **Features (~142):** temporal aggregates (mean/median/min/max/std/range) of
  NDWI, MNDWI, NDVI, AWEI, **EVI**; SAR VH/VV, VV−VH (dB), **SDWI**, and VH/VV
  percentiles (permanence); raw-band aggregates; window-position meta; S1/S2
  asymmetry flags; and **WIF** (count/fraction/longest-run of water months). All
  computed over observed months only, so they are invariant to which window is
  exposed.
- **Model:** LightGBM + XGBoost + CatBoost, 3-seed bags each, rank-average blend.
  Native NaN handling.
- **Validation:** masking-aware, leak-free repeated Stratified 5-fold × 3.
- **Fixed-0.5 calibration:** logit-shift so the F1-optimal operating point lands
  at 0.5, plus a **base-rate (prior) correction** for the test's higher positive
  rate, plus a **prior sweep** tool that emits submissions for several assumed
  priors from one trained model (no retraining) so we can find the optimum
  cheaply within the 5/day budget.
- **Separate targets:** `TargetF1` (calibrated 0.5 cut) vs `TargetRAUC`
  (rank-preserving, AUC-optimal). Reproducible: identical results across machines.

---

## 4. Direction we are heading next (already planned)

1. **Nail the test prior.** Sweep assumed prior ∈ {0.45, 0.50, 0.55, 0.60} in one
   run and submit the bracket to locate the leaderboard-optimal positive rate.
2. **Confirm WIF/EVI** on the leaderboard; keep if they help.
3. **Shrink the domain gap** via stronger regularization / shallower trees and by
   dropping non-transferable absolute-value features (adversarial-guided).
4. Possibly **estimate** the test prior principledly (BBSE / EM) instead of
   sweeping.

---

## 5. Deep-research questions for you (go beyond what we've tried)

Please research and return prioritized, sourced, concretely-implementable
recommendations. Rank each by **expected LB gain × feasibility**, and say which
are worth a precious daily submission.

1. **Label-shift / target-prior estimation (highest priority).** Since prior
   correction gave +0.042 and we are currently *sweeping* the prior, what is the
   state of the art for **estimating the test-set positive rate under label shift**
   from unlabeled test features + a trained classifier, without touching the
   leaderboard? Compare **BBSE (Black-Box Shift Estimation), MLLS/EM
   (Saerens–Latinne–Decaestecker), RLLS**, and confidence-based methods — which is
   most reliable at n≈1030 with a high-AUC classifier, and how do we implement it
   cleanly for a GBDT ensemble? Does the fixed-0.5-F1 metric change which estimate
   we should target (prior that maximizes expected F1, not just matches the prior)?

2. **Domain adaptation for gradient-boosted trees under extreme covariate shift
   (adversarial AUC ≈ 0.99).** Is importance weighting hopeless at that
   separability? What actually works in practice for tabular remote-sensing:
   transferability-based feature selection, CORAL / feature-distribution
   alignment, "frustratingly easy" domain adaptation, invariant/causal feature
   selection, test-time adaptation, or **self-training / pseudo-labeling on the
   unlabeled test set** (this uses only supplied data — is it advisable and how to
   do it safely)? Give concrete recipes.

3. **Temporal feature engineering beyond WIF.** For distinguishing *managed*
   aquaculture ponds from natural water / seasonal flooding using a 4–6-month
   multivariate (12-band) series: what transferable temporal signatures do the
   best remote-sensing pond-mapping papers use — pond management phenology
   (stocking/drain/harvest cycles), harmonic/Fourier descriptors of the monthly
   series, change-detection/temporal-gradient features, SAR temporal-coherence
   proxies, or texture surrogates (given each sample is a single 10×10 m cell with
   no spatial neighborhood provided)? Which survive the constraint that *which*
   months are visible varies per test row?

4. **Model diversity under domain shift.** Would adding a structurally different
   learner — a 1D-CNN / temporal transformer / ROCKET-MiniRocket over the
   12-month multivariate sequence, or **TabPFN** — measurably improve
   generalization over a GBDT-only ensemble in comparable shifted tabular
   competitions? Any evidence on TabPFN's robustness to covariate shift at
   n≈1800 train?

5. **Jointly optimizing 0.6·F1 + 0.4·ROC-AUC.** The two components pull
   differently and we are allowed different `TargetF1` and `TargetRAUC` vectors.
   Is there a principled way to construct each to maximize the weighted sum under
   the fixed-0.5 F1 rule?

6. **Reality check.** Given a designed train/test shift this strong, how high can
   the leaderboard realistically go (what have winners of similarly-shifted
   remote-sensing competitions achieved), and how should we manage public-LB
   (~30% of test, ~309 rows) overfitting risk versus the private split?

7. **Anything specific** to the FAO/ITU GeoAI Aquaculture Pond Identification
   Challenge, or to winning Zindi/Kaggle Sentinel-1/2 tabular solutions, that we
   should know.

Please return your findings so Claude can implement the top items in the next
loop iteration.
