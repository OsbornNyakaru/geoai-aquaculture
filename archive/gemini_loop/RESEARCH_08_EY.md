# RESEARCH_08 — EY Biodiversity Challenge comparison

**Date:** 2026-07-22 · **Question asked:** does <https://zindi.africa/competitions/ey-biodiversity-challenge>
resemble our competition closely enough to borrow from?

## Verdict: NOT TRANSFERABLE for technique — but it is a serious cautionary tale on governance

The two share a platform, a sponsor family, a boilerplate rulebook, and a binary target. **The three
things that define our entire toolkit are all absent there.** It also **closed on 24 May 2026**, so
there is nothing to enter.

| Axis | Verdict | |
|---|---|---|
| Prediction task | SIMILAR | both binary classification of a georeferenced point |
| Data modality | **DIFFERENT** | theirs is TerraClimate 4 km monthly climate rasters sampled to a small tabular table — no SAR, no optical, no masked panel, no `-9999` structure |
| **Designed shift** | **DIFFERENT — decisive** | train and test are both Nov 2017–Nov 2019, same region, random interleaved split. Expected adversarial AUC ≈ **0.5**, versus our ≈0.99 |
| Metric | **DIFFERENT** | single F1 on a hard-label submission with a **free threshold** — the calibration game is legal and trivial there, illegal and central here |
| LB noise | SIMILAR | ~200–300 public rows (20/80 split), but 300 submissions and 10/day |
| Rules | SAME | verbatim Zindi/EY boilerplate — paperwork similarity, not problem similarity |
| Leakage history | SIMILAR in existence, **DIFFERENT in kind** | see below |

**The inverse-coordinates observation is the neatest summary:** *they removed coordinates as a rule
because there is no shift; we lost coordinates as a design because there is one.*

Their CV/LB evidence also runs **opposite** to ours — CV 0.74 → LB 0.88, CV 0.83 → LB 0.88, and once
a pipeline was clean, CV 0.9412 → LB 0.9450 with plain `StratifiedKFold`. That is a level offset, not
our rank anti-correlation. **Nobody there needed to build a validator, and nobody did.** The honest
lesson: when CV works, don't build a CV replacement.

## The 0.99 thread is a leak — but a different class from ours

Their leak was **external re-identification**: labels come from FrogID/GBIF, a *public* occurrence
dataset, and test rows were keyed by supplied lat/lon. Join the coordinates back to public records
and read off the labels. The official benchmark notebook even instructed a `cKDTree` nearest-neighbour
mapping. Legitimate TerraClimate-only ceiling was ~**0.945**; 0.99 was cheating.

**Ours was internal** (a train/test shuffling bug yielding literal 1.0) and was fixed by **rebuilding
the data** on 25 Jun. Theirs could not be fixed by any rebuild without redesigning the task, so it was
policed by rule changes, code demands, and disqualification.

## ⚠️ The one finding that should change our behaviour

**Of a 1,395-person field, only 2 of the top 10 survived post-challenge code review.** The rest were
eliminated for *"failing to submit code within the deadline or having reproducibility concerns."*
Third prize appears to have gone unpaid entirely.

**Our final score is 65% LB + 35% code review of the top 5** — a *larger* weight against a *smaller*
field. The internal audit already found our README documents the **GBDT** as "the model" while our
champion is the Transformer, and that `--full` defaults to `--model gbdt`. **A judge reproducing "the
solution" from our README today reproduces the 0.826-era model.** That is precisely the failure class
that wiped out 8 of EY's top 10.

## Actions taken into the queue

1. **Treat reproducibility as a continuously-maintained artifact, not a deadline task.** Frozen,
   seeded, single-command path for both finalists. Highest-value import from this comparison.
2. **Designate both finalists deliberately.** Zindi defaults to "best 2 public", which on a 30%
   public slice is a shake-up generator. Our certified offline validators are the right instrument
   for choosing the *non-obvious* second finalist.
3. **Audit for implicitly-banned derived features.** EY's enforcement was broader than participants
   expected — not just lat/lon but *distances, nearest-neighbour, clustering, grid binning, any
   positional encoding*. Our coordinates are absent from the files, but we should verify nothing
   reconstructs position or row identity (row-order effects, ID-derived signals, cross-set
   nearest-neighbour matching). A leak-shaped feature that survives to the top 5 and dies in code
   review costs everything.
4. **Pseudo-labeling has an organizer precedent** — an EY organizer explicitly permitted predictions
   "generated from compliant models and the provided competition data" under a near-identical
   rulebook. **Honest caveat:** we rejected CAST self-training in round 05, but on *technical*
   grounds (the OOD/ESS-collapse family at adversarial AUC 0.99), never legal ones. What has changed
   is not the legality argument but that **we can now screen it for zero submissions**. Queued as a
   screen candidate, not as a submission.
5. **Do not trust a Zindi metric description.** Theirs contradicted itself (F1 vs accuracy) and sat
   unanswered for the entire competition. We should verify our 0.6·F1 + 0.4·AUC weighting empirically
   against our own known submission/score pairs rather than trusting the page text.

## Part 2 — the solution-mining pass (prior EY editions)

**Verdict: weak technique source, strong *sociological* source.** Four of six editions are the wrong
problem (frog counts, building-damage detection, UHI point regression, water quality). Published
solutions are overwhelmingly "throw 100–260 tabular features at RandomForest/XGBoost/ExtraTrees with
a random holdout" — a methodology our own design law says would actively lose here. No EY
architecture, loss, or ensembling trick is worth importing. Five things *are*.

### ⭐ Our anti-correlated OOF is a house style, not our bug

The EY×Zindi sibling — **Urban Heat Island Challenge** — has a *designed geographic* shift (train
Santiago + Rio → test **Freetown**) and a top-of-forum notebook titled literally
**"Beat the Baseline in Under 10 Minutes (CV 0.55 vs PL 0.39)"**, plus threads "Stuck on 0.45" and
"How Public and Private test set was split". A ~0.16 F1 gap **in the same direction as ours**, in a
competition deliberately designed with a held-out domain.

**This family of competitions is built so that in-domain CV is structurally optimistic.** Our
anti-correlated OOF is the expected behaviour of the design, not a defect in our pipeline. Worth
saying plainly in the code-review writeup.

### ⭐ EY 2023 Level 1 — the one on-modality edition, and it corroborates both our laws

Binary **rice-paddy presence** from **Sentinel-1 VV/VH time series**, scored by F1. The published
top-3 solution (F1 **0.94**) used:
- `VV_mean`, `VH_mean` over **six phenology-aligned windows** = 12 **amplitude** features,
- plus 6 RVI (ratio) features,
- into a **shallow `RandomForestClassifier(n_estimators=1800, max_depth=7)`**.

Amplitude-primary and capacity-minimal — independently corroborating both of our measured laws. Note
their validation was a plain `train_test_split` reporting **99.44% accuracy** against a real LB of
0.94. *Nobody in this series solved the validation problem we solved.*

### ⭐ We tested the WRONG TAIL

The pond-mapping literature detects ponds with a **LOW-order statistic** — the temporal **median /
p10–p25 of VH** — because a pond is a *"permanent low scatterer"* and low percentiles are robust to
speckle and to which months happen to be observed. Ottinger's VH-median stacks are **bimodal** and
Otsu-thresholdable.

Our `mean_max` probe followed the physics agent's "ponds are never bright" framing. But the published
detector is the **lower** tail, and we never tested it. **`mean_min` added and queued for iter15.**
Crucially, low percentiles are computable from *any* 4–6 consecutive months without needing the
missing ones — they degrade gracefully under exactly our masking.

### Environmental blocking — the published analogue of what we invented

Roberts et al. 2017 (*Ecography* 40:913–929), "Cross-validation strategies for data with **temporal**,
spatial, hierarchical, or phylogenetic structure":
- Random CV under dependence **seriously underestimates predictive error** (our OOF 0.975 vs LB 0.8955).
- Block **even when residuals look clean and even when the model claims to account for the structure.**
- **Environmental blocking** — assign folds by dissimilarity in *predictor space*, not by coordinate —
  is recommended "when the model must predict to new climatic conditions."

That last one **survives the loss of lat/lon**: use our adversarial train-vs-test discriminator
(AUC ≈0.99) as the distance function, rank training rows by test-likeness, and validate on the most
test-like decile. That is environmental blocking with a *learned* blocking variable. It validates
that we built the right thing and gives us a citation for the code review.
Implementations: `blockCV` (Valavi et al. 2019), `mlr3spatiotempcv`.

### Two smaller steals

- **A hard-regime secondary metric.** The EY 2022 winner tracked WMAPE restricted to the hard cases
  because the primary metric was dominated by the easy majority. Our F1 term is hostage to
  calibration drift under temporal shift — track "F1 at fixed 0.5 on the most test-like slice" as a
  first-class number alongside AUC, so we know *which* component a change moved.
- **Multi-seed RFE as an explicit reduction step.** The EY 2022 winner used recursive feature
  elimination repeated across seeds; RicEns-Net (arXiv:2502.06062), built on the EY 2023 data, cut
  100+ predictors to **15**. Both independently corroborate our capacity law, and multi-seed RFE is a
  concrete procedure for deciding *which* features to cut rather than guessing.

### Explicit do-not-steals

The EY 2025 UHI playbook (260 engineered features → ExtraTrees, val R² 0.98) is the exact failure mode
this competition punishes — and those same entrants' Zindi sibling collapsed from CV 0.55 to LB 0.39.
And anything spatial (blocks, buffers, neighbourhood context, footprint fusion) does not port: we have
no coordinates.

**Evidence caveats from the agent:** `challenge.ey.com` is a JS SPA it could not render, so official
rules and Phase-2 rubric weights are unread. MDPI/T&F/SSRN returned 403, so the feature lists are
assembled from abstracts and secondary sources — **feature names are reliable, exact dB thresholds
are not and none are quoted**. No official EY winner source was ever released; every repo cited is a
self-reported participant entry.

## Also found

A **live** sibling with a $10,000 prize: **EY Water Quality Forecasting Challenge** —
<https://zindi.africa/competitions/ey-water-quality-forecasting-challenge>. Not investigated;
flagged only because it is open and in the same family.

**Sources:** competition, data, rules and discussion pages under
<https://zindi.africa/competitions/ey-biodiversity-challenge>, notably threads 32189 (the 0.99
thread), 32229 (compliance action), 32369 (derived spatial features banned), 32819 (pseudo-labeling
permitted), 31566 (LB–CV correlation), 31618 (train/test period), 33879 (winners).

**Caveat on evidence quality:** the closed leaderboard does not render, so score anchors are
reconstructed from discussion quotes rather than read off the board; row counts are inferred from
published file sizes.
