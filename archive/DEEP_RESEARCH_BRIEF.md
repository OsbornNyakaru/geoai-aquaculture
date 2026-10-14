# Deep Research Brief — GeoAI Aquaculture Pond Identification (Zindi / FAO / ITU)

> **How to use this document.** Paste it into a deep-research assistant (e.g.
> Gemini Deep Research). It is fully self-contained — it assumes no access to my
> code or prior conversation. Its purpose: (1) explain exactly how my current
> leaderboard submission (public score **0.7140**, rank **199 / ~229 active**)
> was produced, and (2) get well-sourced guidance on how to climb into the
> **top five**. The open research questions are in **Section 7** — that is what
> I most need answered. This is a **tabular** machine-learning competition with a
> deliberate **train/test domain shift** (different time periods and regions);
> the crux is generalization across that shift, plus a quirky fixed-threshold F1
> metric.

---

## 0. TL;DR

- **Task:** binary-classify each 10 m × 10 m ground cell as *managed aquaculture
  pond* (1) or *other land cover* (0), from 12 monthly composites of Sentinel-1
  radar + Sentinel-2 optical satellite bands.
- **Metric:** `0.6 · F1 + 0.4 · ROC-AUC`. F1 is scored at a **hard 0.5
  threshold — choosing any other threshold is explicitly forbidden.**
- **My status:** solid, reproducible gradient-boosted-tree pipeline. Local
  cross-validation combined score **0.983**, but public leaderboard **0.714** —
  a ~0.27 gap driven by a large, *designed* domain shift between train and test.
- **Diagnosis:** my pipeline is well-engineered (I likely beat most of the field
  on data handling and metric calibration), but I have done **no explicit domain
  adaptation**, and my F1 is probably held back by a **class-prior mismatch**
  (see Sections 4–5). I need research-backed strategy to close the gap.

---

## 1. Competition definition

- **Objective:** predict, per location, whether it is an aquaculture pond. Each
  location is a 10 m × 10 m patch. Features come only from satellite imagery.
- **Explicit temporal-generalization design:** the model is trained on data from
  one time period/region and tested on a **different** period/region. Latitude
  and longitude are removed so models cannot memorize locations.
- **Evaluation metric:** `0.6 · F1-score + 0.4 · ROC-AUC` (Phase 1, 65% of final
  standing).
- **Submission format:** two target columns per test ID —
  `TargetF1` (binary 0/1) and `TargetRAUC` (probability in [0,1]). The two may
  differ; F1 is computed on `TargetF1`, ROC-AUC on `TargetRAUC`.
- **Hard rule (critical):** *"Setting a probability threshold is strictly
  forbidden. Your binary target should be based on the default threshold of
  0.5."* So `TargetF1` must be the 0.5 cut of some probability — you cannot pick
  an arbitrary operating point.
- **Phase 2 (35% of final):** a reproducibility / clarity / innovation rubric
  (0–10). Seeded, end-to-end-reproducible code is required. **AutoML is banned.**
  **Only the supplied competition data may be used** (no external data).
- **Limits:** max 5 submissions/day, 100 total; team ≤ 4. Public leaderboard is
  ~30% of the test set; final standing uses the private ~70%.
- **Timeline:** started 8 Jun 2026; enrollment closes 7 Aug 2026; competition
  closes **16 Aug 2026**.
- **History (important):** an earlier version of the test set had a sorting/
  shuffling exploit that let people score a perfect 1.0. The organizers retired
  that test set, merged it (with labels) into the public training set, and
  released a new 1,030-row test set with the partial-observation masking
  described below. My training file already reflects that merge.

---

## 2. The data (exhaustively)

- **Sizes:** `Train.csv` = 1,821 rows × 146 cols; `Test.csv` = 1,030 rows × 145
  cols; `SampleSubmission.csv` has columns `ID, TargetF1, TargetRAUC`.
- **Columns:** `ID`, `label` (train only), then 144 feature columns named
  `{band}_{MM}` for month `MM` = 01…12. Bands per month (12 total):
  - **Sentinel-1 (radar, SAR):** `VH`, `VV` — dual-polarization backscatter, in
    **decibels (dB)** (values negative, e.g. VH median ≈ −25.5 dB, VV ≈ −15.3 dB).
  - **Sentinel-2 (optical):** `blue, green, red, re1, re2, re3, nir, nira,
    swir1, swir2` — surface reflectance in scaled digital numbers (~1000–7000).
- **Missing-data / masking scheme (the heart of the problem):**
  - Months with no valid observation are filled with **−9999 for all bands**.
  - **Training rows are fully observed** — every train row has all 12 months
    present, no −9999 anywhere.
  - **Test rows are partially masked:** each test row exposes only a
    **consecutive window of 4, 5, or 6 months** (roughly uniform: 345/343/342
    rows for 4/5/6); the remaining months are fully −9999. The window's start
    month is roughly uniform over feasible positions. No wraparound.
  - **Sentinel-2-only cloud gaps:** in 273 of 1,030 test rows, some *in-window*
    months have all optical (S2) bands = −9999 while the radar (S1) bands remain
    present — real monsoon cloud cover blocking optical while radar penetrates.
    Concentrated in month 10 (181 rows) and month 06 (75 rows); month 02 (38).
    The reverse (S1 missing, S2 present) never occurs; there is no partial-band
    masking within an observed month.
- **Class balance:** ~40.4% positive in train. Organizers state the test set
  **may have a higher proportion of positives** than train.
- **Other:** 4 exact-duplicate feature-row pairs exist in train (consistent
  labels); no row overlap between train and test; latitude/longitude removed.

---

## 3. Exactly what my current (0.714) solution does

A config-driven, reproducible pipeline. No AutoML; open-source only.

1. **Load & clean:** regex schema discovery; convert −9999 → NaN (per band, so a
   cloud-masked optical month keeps its radar values); drop the 4 duplicate rows;
   auto-detect SAR is in dB.
2. **Train/test masking alignment (key idea):** since train is fully observed
   but test shows only 4–6 months, I **augment each training row into K=4 masked
   "views"**, each masked to match the *exact* reverse-engineered test recipe
   (consecutive window of length 4/5/6, uniform start, plus simulated S2-only
   cloud dropout at the measured per-month rates). The model thus trains on
   test-like inputs. (This mirrors the winning strategy of the PLAsTiCC Kaggle
   challenge, where the winner degraded well-observed training light curves to
   match the sparse/noisy test cadence.)
3. **Features (~132, all aggregated over *observed* months only, so they are
   invariant to which window is exposed):**
   - **Optical water/vegetation indices** per active month → mean/median/min/
     max/std/range: NDWI = (green−nir)/(green+nir), MNDWI = (green−swir1)/
     (green+swir1), NDVI, AWEI (two forms).
   - **SAR features:** VH, VV, and VV−VH (a *difference* because values are in
     dB), plus **SDWI** (Sentinel-1 Dual-Polarized Water Index — an S1-only water
     index that stays usable when optical is cloud-masked), plus **10/25/50/75/90
     percentiles of VH and VV** (the aquaculture-mapping literature finds
     pixel-wise SAR median/percentiles are the top discriminator of permanent,
     temporally-stable smooth-water ponds vs land).
   - **Raw band temporal aggregates** (mean/median/min/max/std per band).
   - **Window meta:** number of active months, window start/end/center.
   - **S1/S2 asymmetry flags:** count/fraction of months with radar present but
     optical masked, plus month-06 / month-10 cloud flags.
4. **Model:** heterogeneous ensemble of **LightGBM + XGBoost + CatBoost**, each a
   3-seed bag, blended by **rank-average** (scale-free across families). Trees
   handle the NaN missingness natively (learn the best default split direction).
5. **Cross-validation:** masking-aware, leak-free. Folds are defined on the
   *original* rows; every augmented view inherits its row's fold (a row's masked
   twins never straddle the split). Each held-out row is scored on R=3
   independent masked views, averaged → one honest out-of-fold (OOF) probability.
   Repeated 5-fold × 3 repeats.
6. **Fixed-0.5 F1 calibration:** because the threshold must be 0.5, I make 0.5
   the F1-optimal operating point via a monotone transform: find
   `t* = argmax_t F1(y, p ≥ t)` on OOF, then apply the logit shift
   `p' = σ(logit(p) − logit(t*))`, which sends `p = t*` to `p' = 0.5` while
   preserving ranking. `TargetF1 = 1[p' ≥ 0.5]`. A self-check confirms
   F1@0.5(p') == F1@t*(p).
7. **TargetRAUC** is a *separate* strictly-monotone rank transform of the raw
   ensemble score (ROC-AUC is invariant to monotone maps; I avoid isotonic
   flattening that would create ties and hurt AUC).
8. **Reproducibility:** single seed (42) drives all RNGs; per-(row, view) seeds
   derived deterministically. Verified: the same run on my laptop and on Google
   Colab produced a **bit-identical** OOF score (0.98319), which is my Phase-2
   reproducibility evidence.

---

## 4. Where I stand, and why

**The numbers:** local OOF combined = **0.983** (OOF F1@0.5 = 0.975, OOF ROC-AUC
= 0.996). Public LB combined = **0.714**. Gap ≈ **0.27**.

**Why the gap is real and expected:** I ran adversarial validation — a
classifier trained to distinguish my (masked) training feature-vectors from the
real test feature-vectors. It separates them at **AUC ≈ 0.99**, and still
**≈ 0.94 using only the region-normalized water indices**, with window-position
features contributing almost nothing. This means the train and test feature
distributions genuinely differ (different regions/time), by the organizers'
design — it is **not** a bug. Consequence: **my OOF is computed on
train-distribution data and therefore overstates the leaderboard**, and
crucially **OOF cannot measure improvements to domain generalization** — only the
leaderboard can.

### 4a. Why I am likely AHEAD of competitors below me
- **Test-masking augmentation.** Many competitors compute features over the full
  12-month training data and then predict on 4–6-month test rows — a feature
  distribution their model never saw. I explicitly train on test-like masked
  views, removing that mismatch.
- **Fixed-0.5 F1 calibration.** This is the biggest silent scorer. Anyone who
  submits raw model probabilities and lets the 0.5 cut fall where it may loses
  F1, because the F1-optimal threshold is rarely 0.5. My logit-shift recovers it.
- **Correct −9999 handling.** Treating −9999 as a real numeric value (instead of
  missing) poisons every aggregate and tree split — a common beginner error I
  avoid by mapping it to NaN with native GBDT missing handling.
- **Domain-aware features.** dB-correct SAR (difference, not ratio); SDWI for
  cloud-masked months; SAR permanence percentiles; explicit S1/S2 asymmetry
  features.
- **Exploiting the dual-target rule** (calibrated F1 target vs rank-preserving
  AUC target), plus dedup, deterministic seeding, and a heterogeneous ensemble.

### 4b. Why I am BEHIND competitors above me
- **No explicit domain adaptation.** The 0.27 OOF→LB gap is the whole game, and I
  have not addressed it beyond masking alignment. Top teams likely use importance
  weighting, domain-invariant feature selection, feature alignment (e.g. CORAL),
  or much heavier regularization to transfer across the region/time gap.
- **Class-prior mismatch is probably capping my F1.** My `TargetF1` calls only
  **40% positive** (it inherited the training prior), but the organizers say the
  test set is *more* positive. If so, I am systematically under-calling ponds →
  low recall → low F1 — the most likely single reason a well-built pipeline still
  sits mid-field. (I have coded a base-rate/prior correction but not yet
  submitted it.)
- **Untapped modeling headroom:** no hyperparameter tuning, no stacking/meta-
  learner, a single feature set, no adversarial-validation-guided feature
  pruning.
- **Possible missing features:** pond-management phenology / seasonal-cycle
  signatures, richer modeling of the S1/S2 cloud asymmetry, temporal-stability
  descriptors beyond simple std/range.

---

## 5. Metric decomposition (working hypothesis)

Combined 0.714 = 0.6·F1_LB + 0.4·AUC_LB. Ranking (AUC) generally transfers across
domains better than a fixed-threshold F1, so I believe the drop from OOF is
concentrated in **F1**, and that **F1 recovery via prior correction is the
highest-expected-value next lever**. I cannot confirm the F1/AUC split because
the leaderboard reports only the combined score. (Question for research: how to
infer or bound the F1 vs AUC split from limited submissions without threshold-
probing.)

---

## 6. Levers I have already identified (to be validated on the leaderboard, since
OOF is blind to domain gains)

1. **Base-rate / prior correction** for `TargetF1` (assume test positive rate
   ~0.50–0.55 and shift the calibrated probability's log-odds accordingly, still
   cutting at 0.5). Coded, not yet submitted.
2. **Stronger regularization / shallower trees / fewer estimators** to shrink the
   generalization gap.
3. **Drop non-transferable absolute-value features** (keep region-normalized
   indices), guided by adversarial-validation feature importance.
4. **Hyperparameter tuning + light stacking** once the above stabilize.

Discipline: trust the leaderboard over OOF, ≤5 submissions/day, one change per
experiment.

---

## 7. Open research questions (what I need answered to reach the top five)

1. **Domain adaptation for shifted tabular / remote-sensing competitions.** How
   have winning solutions of Kaggle/Zindi challenges with strong train/test
   distribution shift closed a large OOF→LB gap? Give concrete, ranked
   techniques usable with gradient-boosted trees: importance weighting when the
   adversarial train-vs-test AUC is ≈0.99 (are the weights even usable that
   extreme? alternatives?), CORAL / feature-distribution alignment, invariant-
   feature selection, target/prior shift correction, self-training or
   pseudo-labeling on the test set (allowed since it uses only supplied data —
   is it advisable here?).
2. **Fixed-0.5-threshold F1 under unknown positive-rate shift.** Given the
   organizers say test is more positive and threshold tuning is banned, what are
   principled ways to set the *assumed* test prior for a base-rate correction
   **without** probing the leaderboard? (e.g., estimating test prior from the
   unlabeled test features via distribution matching / EM / BBSE — Black Box
   Shift Estimation. Which is most reliable at n=1030?)
3. **Transferable features for aquaculture-pond detection.** From the remote-
   sensing literature (Sentinel-1/2 pond mapping), which features generalize
   best across regions and seasons? Specifically: pond-management phenology
   (fill/drain/harvest cycles), SAR temporal-stability signatures, geometry/
   texture surrogates when explicit coordinates and neighborhood are unavailable
   (each sample is a single 10×10 m cell with no spatial neighborhood provided).
4. **Model diversity for robustness.** Does adding a structurally different model
   (TabPFN, regularized linear, a 1D-CNN / temporal transformer over the
   month-sequence) measurably improve generalization under domain shift versus a
   GBDT-only ensemble, in comparable competitions?
5. **Leaderboard statistics.** With ~1,030 test rows and public = ~30% (~309
   rows), how large is the noise on the public combined score? How should I
   budget submissions and avoid overfitting the public split before the private
   reveal?
6. **This specific challenge.** Any public write-ups, forum insights, or known
   winning tactics for the **FAO/ITU GeoAI Aquaculture Pond Identification
   Challenge** (Zindi), including how top participants handled the
   partial-observation masking and the fixed-0.5 F1 rule.

Please return concrete, prioritized, source-backed recommendations, and flag any
techniques that would violate the rules (no external data, no AutoML, no
threshold tuning, must remain reproducible).
