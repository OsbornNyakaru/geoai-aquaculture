# Deep Research Request — how do we close a 0.17 leaderboard gap on the GeoAI Aquaculture Pond Challenge?

**Paste this whole document into Claude and ask it to do deep research.** It is
self-contained: it assumes no access to my code or prior chats. I need a
concrete, prioritized, **sourced** plan to move a Zindi competition submission
from **public score 0.756 (rank 187)** toward the **top 5 (which currently needs
~0.928)**. The field has clearly found an approach we are missing — the core
question is *what is it*.

---

## 0. The situation in one paragraph

Zindi / FAO / ITU **GeoAI Aquaculture Pond Identification Challenge**. Binary
classification of 10 m × 10 m ground cells as aquaculture pond (1) vs other (0),
from 12 monthly composites of Sentinel-1 radar + Sentinel-2 optical bands. Metric
= `0.6·F1 + 0.4·ROC-AUC`, with F1 scored at a **hard 0.5 threshold** (threshold
tuning banned). We built a careful gradient-boosted-tree pipeline and reached
**0.756**. But the public leaderboard shows a dense field far above us. **We are
underperforming a problem that hundreds of competitors have solved to 0.85+.**
The research goal: identify the approach/features/tricks that get to 0.90+ and a
step-by-step plan to close the gap.

---

## 1. The leaderboard reality (this is the key context)

Current public leaderboard (public = ~30% of the 1,030-row test set, ~309 rows):

| Rank | Score | Subs |
|---|---|---|
| 1 | 0.9453 | 58 |
| 2 | 0.9413 | 42 |
| 3 | 0.9403 | 30 |
| 4 | 0.9315 | 62 |
| 5 | 0.9283 | 30 |
| 10 | 0.9154 | 5 |
| 20 | 0.9005 | 42 |
| 50 | 0.8761 | 10 |
| **187 (us)** | **0.7561** | **3** |

Observations to explain: (a) the top is **dense at 0.90–0.945**, so this is not
an unbeatable domain-shift ceiling; (b) rank 10 reached 0.915 in only **5
submissions**, implying a strong approach exists that gets there fast; (c) we, at
0.756, are ~0.12 below rank 50 and ~0.17 below top-5. **What do the 0.88–0.94
solutions know that we don't?**

---

## 2. The data (exhaustive, so research can reason about it)

- **Train:** 1,821 rows × 146 cols (`ID`, `label`, 144 feature cols). **Test:**
  1,030 rows × 145 cols. Submission: `ID, TargetF1 (0/1), TargetRAUC (prob)`.
- **Columns:** `{band}_{MM}` for month MM=01..12. Bands (12): Sentinel-1
  `VH, VV` (radar backscatter, in **dB**, negative); Sentinel-2
  `blue, green, red, re1, re2, re3, nir, nira, swir1, swir2` (optical reflectance,
  scaled DN ~1000–7000).
- **Masking (crucial):** months with no observation are **−9999 for all bands**.
  **Train rows are fully observed (all 12 months).** **Test rows expose only a
  consecutive 4/5/6-month window** (rest = −9999); window start ~uniform.
- **Sentinel-2 cloud gaps:** in 273 of 1,030 test rows, some in-window months
  have S2 optical = −9999 but S1 radar present (monsoon cloud), concentrated in
  October (181 rows) and June (75). Never the reverse.
- **Class balance:** train ~40% positive; organizers say **test is more
  positive**. Lat/lon removed. History: an earlier test set had a leakage/sorting
  exploit (perfect 1.0 scores); organizers reset it, merged the old labeled test
  into train, and released this new masked test set.

---

## 3. Exactly what our current 0.756 pipeline does (the baseline to beat)

1. **−9999 → NaN**, dedup 4 duplicate rows, auto-detect SAR is dB.
2. **Train/test masking alignment:** since train is fully observed but test shows
   4–6 months, we augment each train row into K=4 masked "views" matching the
   exact test masking recipe (window + simulated S2 cloud dropout), so the model
   trains on test-like inputs (à la the PLAsTiCC Kaggle winner).
3. **Features (~132), all aggregated over observed months only:** temporal
   mean/median/min/max/std/range of NDWI, MNDWI, NDVI, AWEI; SAR VH, VV, VV−VH
   (dB), SDWI, and VH/VV percentiles; raw-band aggregates; window-position meta;
   S1/S2 asymmetry flags.
4. **Model:** LightGBM + XGBoost + CatBoost, 3-seed bags each, **rank-average**
   blend, native NaN handling.
5. **CV:** masking-aware, leak-free repeated Stratified 5-fold × 3 (a row's masked
   views never straddle the train/val split). Local OOF combined ≈ **0.983**.
6. **Fixed-0.5 calibration:** logit-shift so the F1-optimal point sits at 0.5,
   plus a base-rate (prior) correction; separate rank-preserving `TargetRAUC`.
7. Fully reproducible (identical results across machines).

---

## 4. What we've learned experimentally (facts, not guesses)

- **Local OOF (0.983) hugely overstates the LB (0.756).** An adversarial
  classifier separates our (masked) train features from test features at **AUC ≈
  0.99** (≈0.94 using only normalized water indices) → strong train/test
  distribution shift. So OOF is blind to generalization; only the LB is truth.
- **Prior correction is our only big win so far:** shifting `TargetF1` from ~40%
  positive (train prior) to ~50% positive raised LB **0.714 → 0.756 (+0.042)**.
  So the test really is more positive, and the fixed-0.5 F1 is prior-sensitive.
- **A strongly train-discriminative feature FAILED on the LB:** a "Water
  Inundation Frequency" count scored AUC 0.826 on train labels but adding it moved
  LB 0.756 → 0.751 (no gain). Train-discriminativeness ≠ transferability here.
- We have not yet tried: pushing the prior above 0.50, self-training/
  pseudo-labeling on test, model families beyond GBDTs, or any explicit domain
  adaptation.

---

## 5. The central research questions (please investigate deeply and rank answers)

**The overarching question: the field reaches 0.88–0.945 and we're stuck at
0.756. What is the standard/winning approach we're missing?** Investigate:

1. **Is our masking augmentation counterproductive?** Most likely failure mode: by
   masking train down to 4–6 months we may be *discarding signal* that a simpler
   approach (compute robust aggregates over whatever months are present, train on
   full data) keeps. Do winning Sentinel time-series solutions augment like this,
   or just aggregate and let trees handle missingness? Which wins empirically?

2. **How positive is the test set really, and what positive rate maximizes F1?**
   We gained +0.042 going 40%→50%. If the optimum is 60–70%, that alone could be
   worth a lot. What is the principled way to find it, and do top solutions
   effectively predict a high positive rate?

3. **Is there a residual leak or exploitable structure** despite the reset? The
   competition has a documented leakage history. Are there ID patterns, row
   ordering, near-duplicate signatures between the merged old-test-now-train rows
   and the new test, or any structure the top-5 (0.94+, dense cluster) might be
   exploiting? How would we detect it safely?

4. **What feature representations actually transfer** across regions/seasons for
   Sentinel-1/2 pond detection? We found normalized indices still separate
   train/test at AUC 0.94 (i.e. even "invariant" features shift). What do the best
   remote-sensing pond-mapping papers and Zindi/Kaggle Sentinel winners use —
   temporal harmonics/Fourier, phenology of pond management cycles, SAR
   temporal-stability, per-window-normalized (z-scored) features, quantile
   features?

5. **Is a GBDT ensemble the wrong tool, or is our modeling weak?** Rank 10 hit
   0.915 in 5 submissions — suggesting a robust single approach. Would a
   well-tuned single LightGBM, a tabular neural net (TabPFN, RealMLP), or a
   temporal model (1D-CNN / transformer / ROCKET over the month sequence) beat our
   3-family rank-averaged blend under this shift? Is our rank-average blend or
   calibration hurting AUC or F1?

6. **Where exactly are we losing — F1 or AUC?** Combined = 0.6·F1 + 0.4·AUC. Given
   LB 0.756 and our likely high-but-shifted ranking, decompose plausible F1 vs AUC
   splits and identify which term is the bottleneck, so we fix the right thing.

7. **Domain adaptation that actually works for tabular shift at adversarial AUC
   ≈0.99:** importance weighting, self-training/pseudo-labeling (allowed — uses
   only supplied test data), feature alignment (CORAL), invariant-feature
   selection, test-time adaptation. Rank by expected LB gain and risk.

8. **Public assets:** is there a widely-used public baseline notebook, a Zindi
   forum thread, or a known winning recipe for THIS challenge (FAO/ITU GeoAI
   Aquaculture) that the 0.88+ cluster converged on? Find it.

---

## 6. Constraints (any proposal must respect these)

- **Only the supplied competition data** — no external data or pretraining on
  other datasets.
- **AutoML banned.**
- **`TargetF1` must be the 0.5 cut of a probability** — no post-hoc threshold
  tuning (base-rate/prior correction of the probability is allowed).
- **Seeded, reproducible, open-source only.** Max 5 submissions/day.

---

## 7. What I want back

A prioritized, **sourced** action plan to get from 0.756 toward 0.90+ and then
top-5, ordered by **expected LB gain × feasibility**. For each recommendation:
the mechanism, why it should transfer under this domain shift, concrete
implementation steps, and whether it's worth a scarce daily submission. Explicitly
call out the single most likely reason we're at 0.756 while the field is at
0.90+, and lead with that. Flag anything that would violate the rules. Be
skeptical of our current pipeline — assume we may be over-engineered in the wrong
direction.
