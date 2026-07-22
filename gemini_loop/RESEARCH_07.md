# RESEARCH_07 — multi-agent research round + live competition intelligence

**Date:** 2026-07-22 · **Champion:** 0.8955 · **Status of loop:** iter11 staged, awaiting Colab run.

This round differed from 01–06: instead of briefing an external Deep Research model, a team of
research agents was spawned in-session **and** the live Zindi competition site + forum were read
directly. The site reading produced the highest-value findings of the entire project to date,
including **one rule we have been enforcing incorrectly for the whole competition**.

> **Agent status note.** Four agents were launched. One (feature engineering) completed and its
> findings are folded in below. One (competition strategy / prior Kaggle competitions) was still
> running at write-time. Two were killed by an account **monthly spend-limit** error, not by any
> fault in the task. Their briefs are preserved in this file's §7 so they can be relaunched cheaply.

---

## 1. THE HEADLINE — a constraint we got wrong

Verbatim from <https://zindi.africa/competitions/geoai-aquaculture-pond-identification-challenge/rules>:

> **"You may use pretrained models as long as they are openly available to everyone."**
> "You may use only the datasets provided for this challenge."
> "Automated machine learning tools such as automl are not permitted."

Every planning document we hold — `AGENT_BRIEF.md`, `UPDATE_01`–`UPDATE_06`, `PROJECT_STATE.md` —
states *"no pretrained/foundation models (bans TabPFN, ImageNet/SSL backbones). Train from scratch."*
**That is wrong.** External *data* is banned; pretrained *models* are explicitly permitted.

**What this reopens.** TabPFN / TabPFN v2 (pretrained on synthetic priors, openly available,
designed precisely for small tabular datasets — our regime is 1,821 train rows) was rejected in
round 02 on grounds that do not exist. Remote-sensing foundation models pretrained on *pixel time
series* — **Presto** especially, which is small, handles missing modalities, and matches our data
shape almost exactly — are legal. So are Prithvi, Clay, SatMAE, and time-series FMs (MOMENT, Chronos).

**The honest counter-argument.** Our single most reliable measured law is *added capacity hurts*.
A foundation model is the largest capacity add imaginable. But it is **categorically different**:
its capacity is amortised in pretraining rather than fitted to our shifted 1,821-row train set,
which is the exact mechanism by which our capacity adds have failed. This deserves a real probe,
not an assumption in either direction.

**Action:** correct the constraint in all docs. Do **not** immediately spend submissions on it —
it enters the queue ranked against everything else in §6.

---

## 2. Confirmed data and competition facts (all verified on the live site)

| Fact | Value | Source |
|---|---|---|
| Train | **1,821 rows, 12 FULL monthly composites**, ~40% positive | data page |
| Test | **1,030 rows, only 4/5/6 CONSECUTIVE months observed**, rest `-9999` | data page |
| Test positive rate | believed **higher than train** (~0.65) | data page |
| Bands | S1 **VH, VV** (always present when month observed) + 10 S2 optical (may be absent — **cloud**) | data page |
| Coordinates | **lat/lon REMOVED** from public files | Challenge Update |
| Shift | **TEMPORAL BY DESIGN** — "trained on data from one time period and tested on data from a different one"; conditions "change across seasons and years" | overview |
| Public / private LB | **30% / 70%** of test → ~309 / ~721 rows | overview |
| Metric | 0.6·F1 (hard 0.5, tuning **forbidden**) + 0.4·ROC-AUC | evaluation |
| Submissions | 5/day, **100 total**; **choose 2 finalists** (else 2 best public) | rules |
| Final score | **65% LB + 35% code review** of top 5 (reproducibility, clarity, novelty) | evaluation |

### 2a. The 25 June data reset — and why our anchors survive it

On **25 Jun 2026** the organisers rebuilt the challenge after a leak ("Critical Data Leakage:
Shuffling Issue … 1.0 Perfect Score Exploit"). New **train = original train + original test *with
labels***; a brand-new test set was issued; lat/lon stripped; limits set to 5/day and 100 total.

**Our first submission was 2026-07-09**, two weeks after the reset. **Every one of our seven LB
anchors post-dates it**, so `experiments/anchors.tsv` is internally consistent and the **iter11
retro-fit is valid**. This was a genuine risk to the staged experiment and it is now cleared.

**Corollary — recalibrate the target.** Forum scores of **0.953 and "0.98+" are pre-reset**, earned
on leaked data, and must be **ignored**. The live competitive band is the "90s club" (thread dated
14 Jul, post-reset): roughly **0.90–0.95**. Our 0.8955 sits **just below the current bar**, not
0.033 behind it. The gap is smaller and more winnable than `PROJECT_STATE.md` currently claims.

### 2b. The asymmetry we under-weighted

**Train sees a full 12 months; test sees a 4–6 month window.** Our masking augmentation exists to
simulate this, but note what it means: the shift is *both* temporal (different period) *and*
structural (different observation length). Two distinct axes, and we have only ever probed the second.

---

## 3. What competitors are saying (post-reset, therefore credible)

From the forum threads *"90s club any tips?😭"* (14 Jul) and *"Tips for handling the temporal shift"*:

**"sdv"** — the most informative post in the forum:
- "the CV/LB gap here is **brutal**"; random K-fold unreliable because train/test are different periods.
- **"Relative and ratio-based features outperform absolute values across time shifts."**
- **"The two scored columns are independent — worth optimizing each on its own terms."**
- Trust the LB over CV. Don't treat monthly data as independent rows.
- When the OP said tree models perform poorly, sdv **disagreed**: CatBoost-family GBDTs **are**
  viable and "the bottleneck lies elsewhere."

**"chiwai"**:
- Prioritise the **validation approach first** — "a random split can look very strong but fail badly here."
- Prefer "**robust summaries** rather than relying only on raw monthly bands": temporal aggregates,
  **seasonal min/max/range/std**, and NDWI/MNDWI/NDVI-style indices.
- Handle the fixed 0.5 threshold via "**class weighting, balanced objectives, probability
  calibration**" *during training*, not post-hoc.
- "A single strong tree model did well, but an **ensemble of different tree-based models** usually
  gave more **stable** rankings."

### Three ways this challenges our own doctrine

1. **Our "GBDT lane is dead" verdict may be premature.** A credible 90s-club competitor says GBDTs
   work. Our GBDT lane was fed largely *absolute* features — exactly what sdv says fails under
   temporal shift. We may have killed the lane for the wrong reason.
2. **Our anti-ensembling law is contradicted** by a competitor reporting ensembles give more stable
   rankings. Our one blend lost 0.0075 — but that blend mixed a *weak, badly-featurised* GBDT into
   a strong sequence model, which is not the same experiment as ensembling two strong diverse models.
3. **Independent scoring of the two columns is a possible unexploited lever** — see §4.

Two ways it **confirms** us: "relative and ratio-based features beat absolute" is an independent
restatement of our hard-won amplitude-toxicity finding; and "prioritise validation first" is exactly
what iter11 does.

---

## 4. The two-column question (potentially the biggest single lever — UNRESOLVED)

The submission has **two columns**: `TargetF1` (binary 0/1) and `TargetRAUC` (probability 0–1).
They are scored by **different metrics**, and sdv says they are "independent — worth optimizing
each on its own terms."

We currently derive `TargetF1` by thresholding our (prevalence-shifted) probability at 0.5. But the
F1 column and the AUC column need not come from the same model at all: **AUC rewards ranking,
F1 rewards a good hard partition**, and these have different optimal solutions.

**The legality question must be settled before this is acted on.** The rules say "setting a
probability threshold is strictly forbidden" and the binary target "should be based on the default
threshold of 0.5." A reading where you submit a *differently-calibrated* probability for the F1
column is arguably threshold tuning by another name. **Do not act on this until a Zindi admin
answers it in the forum.** Post the question — it costs nothing and may unlock the round.

---

## 5. Findings from the feature-engineering agent (completed)

Ranked, with our design law applied. Full reasoning and citations are in the agent report; the
load-bearing items:

**R1 — Partial-S2 missingness audit (0 submissions, do first).** The training mask sampler drops
**all 10 S2 bands together** (`src/features.py:92`), and the measurement code only counts months
where *all* S2 bands are masked (`src/data.py:180-186`). But S2 bands go missing **per-band by
cloud**. If the real test cube contains *partially* masked S2 months, our training views have
**never shown the model a missingness pattern that test rows actually exhibit** — an unmodelled,
genuinely shifted axis, which is precisely the criterion that predicted relative-time's +0.0128 win.
Pure diagnostic; costs nothing. **Expected: 0 if negative, +0.01–0.02 if positive.**

**R2 — Missing-indicator compression 24 → 14 channels (1 submission).** Given whole-block masking,
the 12 missing-indicators carry ≤2 bits/month — 10 of 24 input channels are near-duplicates.
Replacing them with 2 (`s1_observed`, `s2_observed`) is a **capacity-reducing structural deletion**
of a redundant memorisation surface: the exact winning template. **−0.9% params. Expected +0.005–0.015.**
*Order-dependent: only valid if R1's audit is negative.*

**R3 — Antithetic K=2 view sampling (1 submission).** Cross-view invariance (+0.0047, our champion)
is only as strong as the *difference* between the two views, yet with L∈{4,5,6} of 12 months and iid
starts, many view pairs overlap heavily and `Var_k(logit)` is near-vacuous. Draw the second view
**maximally distant** from the first while preserving the marginal window distribution. **Exactly
zero capacity change.** This also reinterprets iter10: λ=3 amplified a *noisy* signal; improving the
signal itself is strictly the better move. **Expected +0.005–0.015.**

**R4 — iter12 refined.** Keep `mean ⊕ std` but (a) use **biased 1/N** std, (b) `sqrt(var + 1e-5)`
(infinite gradient at zero is reachable at N=4 with dropout), and (c) **zero-initialise the std half
of the head's first layer** so the network starts *bit-identical to the champion* and must earn any
std usage. That converts a capacity add into an identity-preserving expansion. **Do not lead with
split-pooling** — it deletes 32 of 64 proven mean dims, so a loss would be uninterpretable.

**R6 — VH−VV: REJECT, remove from the queue.** `(VH,VV) → (VH, VH−VV)` is an **invertible linear
map fed straight into `nn.Linear`** — the model can already represent it exactly. Only real effect is
rescaling via per-band standardisation; an optimiser nudge, not a representational change. Plausible
effect well under ±0.005, permanently unmeasurable at our noise floor.

Also explicitly killed on N=4 grounds: permutation entropy, lag-1 autocorrelation, spectral entropy
(all degenerate at 4 samples); GeM/max pooling (emphasises the harvest-drain transient that the pond
literature deliberately *suppresses*).

---

## 5b. Findings from the competition-strategy agent (prior Kaggle/Zindi competitions)

### The noise floor, finally quantified

Public n≈309 at prevalence 0.649 → ~200 pos / ~109 neg. SE(F1) ≈ √(0.9·0.1/250) ≈ 0.019;
Hanley–McNeil SE(AUC) ≈ 0.012. Combined metric **SE ≈ 0.012 (1σ)** — our ±0.01 estimate was right.
Private n≈721 → SE ≈ 0.008.

**But paired deltas are much cheaper than we assumed.** For two variants of *our own* model with
prediction correlation ρ≈0.9, SE(Δpublic) ≈ √(2(1−ρ))·0.012 ≈ **0.006**; for near-clones (ρ≈0.95+),
≈0.004. So the working protocol is:

| Comparison | Threshold |
|---|---|
| Unpaired / cross-team | ≥ 0.012 to mean anything |
| Our own A/B vs champion — **confident** | ≥ 0.012 |
| Our own A/B vs champion — **suggestive** | ≥ 0.006 |
| Below 0.006 | unmeasurable even paired |
| Expected \|public − private\| drift for one model | ≈ **0.012** (2σ swing ±0.029 is possible) |

This retroactively rates our champion's +0.0047 at ~0.8σ paired — plausible but unproven, which is
exactly how we treated it. It also means **any public gain under ~0.012 has roughly even odds of not
existing on the private 70%.**

### Precedent for our two wins — and the move we have never made

Microsoft Malware Prediction is our situation almost exactly (temporal split, adversarial AUC ≈1.0,
huge public→private shakeup). Survivors **dropped the time-drifted features** identified by
adversarial validation; public-LB chasers collapsed. **Our relative-time deletion is the same move.**

**The untried next step: iterated adversarial channel attribution.** Re-run adversarial validation
*after* relative-time, per channel group, and rank channels by shift × (1 − signal). Prime suspects
are the strongly **seasonal S2 optical bands** — under a temporal shift, phenology, sun angle,
atmosphere and cloud regime move optical channels hard, while SAR water response is comparatively
period-stable (specular reflection off water is low VH in any year). Deleting a high-shift,
low-signal channel is **capacity reduction — our only winning family**. The offline audit costs
**zero submissions and tells us whether to spend one at all**. Expected +0.005–0.013 if a dead
shifted channel exists.

### The amplitude paradox, resolved

Why did detrending lose −0.051 if amplitude is the shifted axis? Because a temporal shift adds a
*bounded offset* to per-series level, but the class signal is the *absolute* level separation (water
VH ≈ −20 dB vs land, in any year) — **much larger than the seasonal offset**. Detrending deleted a
large class separation to remove a small offset: catastrophic by construction, not a paradox.
The correct treatment of a shifted-but-signal-bearing channel is to make the *decision* robust —
which is what cross-view invariance already does. **Deletion is only for shifted channels with no
absolute-signal content.** That is precisely why calendar position won and level lost, and it gives
us a *predictive* screen for future deletion candidates rather than a post-hoc story.

### Our prevalence shift is at its theoretical optimum

Lipton et al. (arXiv:1402.1892): for calibrated probabilities the F1-optimal threshold is **F1*/2**.
Our champion's `t_star = 0.445` ≈ 0.89/2. The prevalence instrument is **saturated** — the queued
endgame prevalence sweep should be expected to confirm a plateau, not find a gain.

### The anti-ensembling law — a sharper statement

The law is over-broad as stated, and there is an inconsistency worth naming: **our champion already
IS an ensemble** (mean of 5 fold-models). Fold-averaging survives because its components are equals.
Our failed blend rank-averaged a 0.826-class GBDT into a 0.896-class Transformer at ρ=0.85 — a
0.07-weaker correlated component can only dilute. That refutes *that blend*, not blending.

Abe et al. (NeurIPS 2022) show ensemble gains are essentially indistinguishable from adding
equivalent single-model capacity — so theory and our measurement agree: **ensembling is a capacity
move, and capacity moves lose under this shift.** The real law is *"no weak or correlated additional
components."*

**Revisit trigger (precise):** the day any lane produces a second model within ~0.01 of 0.8955 with
visibly different errors — and prediction correlation ρ can be checked on test **with no labels** —
a 2-component equal-quality blend becomes the highest-expected-value submission available.

### Amendment to the finalist plan

If a pretrained lane lands within ~0.01 of the champion, **it becomes finalist #2 instead of NoPE**.
NoPE is a near-clone (very high ρ) and buys almost no private-LB variance hedge; a different model
class does. Shakeup-survivor practice is to pick two finalists from **different failure modes**,
never two public-LB neighbours.

### Architecture: we are probably not in the wrong class

Spot the Crop 2nd place used InceptionTime/XceptionTime; ITU Cropland Mapping top-3 used per-region
LGBM/CatBoost. The winning edge in these came from **features, validation and stratum handling —
not architecture**. Our from-scratch temporal Transformer is not obviously wrong. Also relevant:
ITU Cropland is the **same organiser family** as ours and weighted its report at 40%, confirming the
35% rubric here is real and worth engineering for.

---

## 5c. Findings from the internal code audit — **this section invalidates prior conclusions**

I independently verified the two most damaging claims in the code before accepting them.

### ⛔ Finding (C) "the amplitude axis is toxic" has NO evidence behind it

`src/seq_model.py:87-94`:
```python
parts = [vals, miss]          # vals = the ABSOLUTE standardized bands
if channels_cfg and ex_mean is not None:
    ex = _raw_extra_channels(cube, schema, channels_cfg)
    parts.append(ex)          # detrend channels are APPENDED
x = np.concatenate(parts, axis=2)
```
**`per_cell_detrend` never removed anything.** It *appended* 12 detrended channels on top of the
untouched absolute-value channels (24 → 36). The −0.0514 therefore measured **"adding 12 correlated
channels hurts"** — it says *nothing whatever about amplitude*.

This is the single most consequential error found in the whole project, because that one run is the
sole basis for:
- the belief that per-series level is toxic to touch;
- the blanket ban on the ratio/relative channel family (`deltas`, `indices`, `rank`) — **exactly the
  family our competitor intel says wins** ("relative and ratio-based features outperform absolute
  values across time shifts");
- the framing of every subsequent research brief, which told each researcher amplitude was proven toxic.

The competition-strategy agent independently constructed a *physical* story reconciling the −0.051
(deleting a large class separation to remove a small offset). That story is elegant and it is **also
unnecessary** — the run simply wasn't the experiment we thought it was. **The amplitude question is
reopened and untested.** `seq.channels.rank` (within-series normalized rank, maximally
amplitude-invariant, `src/seq_model.py:150-166`) has never been probed and is the clean test — as a
**replacement** for `vals`, not an addition.

### ✅ The rank-only proof, confirmed in our code

`src/calibration.py:127-128` applies `logit_shift(p_test, t_star)`; line 155 then calls
`target_prevalence_shift`, which sets `delta = −quantile(logit(p), 1−π)`. A constant logit offset
**cancels out of a quantile**. So with `prevalence_target` set, `t_star` is **inert**, and the
submission reduces to `top-k(p_test_raw)` plus a rank transform.

Two agents reached this independently, one from the metric algebra and one from our code. Every
calibration diagnostic in the ledger (`t*` 0.500→0.445, `δ` 2.03→1.30) is **invisible to the
leaderboard**. The iter9 win and iter10 loss are real; the *mechanism* we recorded for both
("de-saturation") is impossible. **"The objective lane is closed because de-saturation stops paying"
is therefore unfounded** — λ=1 and λ=3 differ by a ranking change of unknown origin.

### 🔍 The OOF anti-correlation — the leading hypothesis is now testable for free

The hypothesis I had been pushing (*OOF scored on 12-month series*) is **killed by the code**: OOF
views are drawn from the same measured test window distribution as test rows
(`src/seq_model.py:473-474` → `src/features.py:44-93`). The window regime matches.

What does **not** match is the **estimator**:

| | OOF | Test |
|---|---|---|
| models | **1** (the current fold's net) | **mean of 5** fold-models |
| views per row | **2** (averaged) | **1** (the real window) |

`src/seq_model.py:477-482` vs `489`/`499`. Fold-averaging is **not rank-preserving**, so it is a
genuine ensembling step that the LB sees and OOF never measures. That predicts the ledger's sign
pattern: a change that makes each model better but the five **more alike** raises OOF and lowers LB.
**K=4 is exactly such a change** — more views per row, each fold model converging to the same
smoothed function — and it gave our highest OOF (0.9840) and 2nd-worst LB. Cross-view invariance is
the converse: harder constraint per model (lowest OOF, 0.9753), no homogenisation — our champion.

**I implemented the free test.** `run_seq_cv` now returns per-fold test predictions, `run_pipeline.py`
saves them into the preds bundle, and `tools/offline_validate.py` has a new **DIV** estimator
(1 − mean pairwise Spearman between the 5 fold-models). It rides on the runs already staged, costs
nothing extra, and if it ranks the anchors we have an LB-predictive local signal needing no
unlabeled-data theory at all. The gate accepts an **anti**-correlated DIV too — a reliably inverted
estimator is just as usable (negate it) and would be an equally interesting finding.

### Other confirmed defects

| # | Finding | Location | Consequence |
|---|---|---|---|
| S3 | Band scaler fit **globally** on all 1,821 train rows, outside the fold loop | `src/seq_model.py:426` | Every fold's validation rows fed the scaler → OOF uniformly optimistic. Explains bias, not anti-correlation. Free fix. |
| S5 | Seed determinism incomplete for the seq path (no `use_deterministic_algorithms`, no `cudnn.deterministic`) | `src/utils.py:60-65`, `src/seq_model.py:461` | **Run-to-run spread has never been measured**, yet iter8 (+0.0009), iter9 (+0.0047) and iter10 (−0.0034) are all *smaller than a plausible seed effect*. |
| S6 | The queued endgame `prior_sweep` uses `apply_prior` (a log-odds offset), **not** `target_prevalence_shift` | `run_pipeline.py:198-208` | It would sweep a *different lever* than the one that scored 0.8955, and no entry reproduces the main submission. |
| S8 | `TargetF1` and `TargetRAUC` are both derived from the same `p_test_raw` | `run_pipeline.py:164-168` | The two columns are scored **independently**; nothing requires one ranking. Also means the −0.0075 blend was scored on *both* columns at once — its AUC contribution was never isolated. |
| — | Finding (B)'s asymmetry (relative-time deleted a *shifted* start; dnorm deleted a *matched* length) | `src/data.py:192-216`, `src/features.py:52-63` | **False**: start *and* length are both sampled from the measured test distribution. The real deleted channel was calendar **identity**, not a shifted start distribution. |
| — | Repro gaps for the 35% rubric | `README.md:28,119-141` | The README documents the **GBDT** as "the model" and `--full` defaults to `--model gbdt` — a judge reproducing "the solution" reproduces the 0.826-era model, not the champion. `results.tsv` (our strongest innovation evidence) is gitignored. |

### The amplitude-invariant features already sitting unused in the GBDT lane

Genuinely invariant (ratio / dB-difference / rank / boolean): `ndwi`, `mndwi`, `ndvi`
(`src/features.py:106-107,125-128`); **`vv_minus_vh`** (`:142` — a dB difference is a linear ratio,
the cleanest candidate); `std` and `range` aggregates (`:31,34-38` — level-invariant *dispersion*,
i.e. Ottinger's permanence signature, **already computed and never given to the Transformer**);
`_rank_months` (`src/seq_model.py:150-166`).

Frequently miscategorised as invariant but **not**: `awei_nsh`/`awei_sh` (weighted linear
combinations, `:129-130`) and **`sdwi`** (`:144-154` — `lnVV + lnVH + const` is a *sum of levels*,
i.e. *more* amplitude-sensitive than either band). Also still present in the GBDT lane:
`meta_start`/`meta_end`/`meta_center` (`:212-217`) — **calendar position, the exact channel whose
deletion won +0.0128 in the seq lane**, never deleted here.

---

## 5d. Findings from the pond-physics agent — **iter12 is challenged**

### The verdict: CHALLENGE, with one objection voided by the code audit

The physics agent raised three objections to `mean ⊕ std`. **Its lead objection is void**, and I can
only see that by crossing its report against the code audit — neither agent could do this alone:

> **(i) "std is computed over 12 months in TRAIN and 4–6 in TEST — different physical quantities,
> shifted by construction."** This would be fatal. But `src/features.py:52-63` already samples the
> training window's **length from the measured test p(L)** and its **start from p(start|L)**. Train
> views are already masked to test-like 4–6-month windows. **Objection void.** (The agent flagged it
> conditionally — "if you are not already doing it… say so and skip" — having no code access.)

**Objections (ii) and (iii) stand, and they are serious:**

- **Dispersion is genuinely season-dependent.** Ponds are permanent but *managed*: area expands in
  monsoon and contracts in summer through maintenance and evaporation (Ottinger 2025), and
  drain-for-harvest is a deliberate recurring perturbation. "Ponds have low dispersion" is true
  *annually averaged* and unreliable *within an arbitrary 4-month slice*. Wind roughening injects
  further weather-driven, class-correlated variance.
- **At n=4 a std is ~41% noisy** (relative SE = 1/√(2(n−1))), sitting on top of a residual speckle
  floor of ~1–2 dB that is comparable to the hydrological dispersion difference being sought. One
  silver lining: because dB turns multiplicative speckle into *additive, roughly class-independent*
  noise, the comparison is attenuated rather than biased.

Expected effect for iter12 as specified: **−0.02 to +0.005, mode negative**; ~25% chance of a real win.

### The proposed replacement — and an honest disagreement between two agents

The physics agent argues the literature's actual pond discriminator is **"never bright in any
month"** (an *upper-tail* statistic), not "low variance". Rice, cropland and wetland negatives are
all defined by *having* a bright canopy month — VH climbs 6–10 dB from flood to tillering — while a
pond never rises. A **max** is an order statistic with **no calendar reference at all**, so it is
immune to any train/test phase shift. It recommends **mean ⊕ max** over mean ⊕ std.

**This directly contradicts the feature-engineering agent**, which rejected max on the grounds that
the drain/harvest transient is an outlier the literature deliberately *suppresses* via the temporal
median. I am not going to paper over this. Both are partly right, and the resolution is that they
are talking about **different bright events**:

| | Bright event | Effect on max |
|---|---|---|
| Feature agent | Pond **drain** (dry bed → rough → brighter) | Adds noise to positives |
| Physics agent | Rice **canopy** (+6–10 dB, volume scattering) | Separates the main hard negative |

So max is simultaneously contaminated (by drains) and discriminative (against rice). Which dominates
is an **empirical question we cannot settle from the literature** — and it is exactly the kind of
question the iter11 validator exists to answer offline. **Do not spend a submission choosing between
mean⊕std and mean⊕max; screen both.**

### Amplitude toxicity: real physics, but now doubly reframed

The physics agent independently quantified the nuisance budget over a *permanent* pond:

| Source | Magnitude |
|---|---|
| **Incidence angle / relative orbit** (29–46° swath, ~0.2–0.3 dB/deg) | **2–5 dB** — the dominant unlabelled axis |
| **Wind roughening** (VV, episodic) | **+3–10 dB** |
| **VH noise floor** — VH over calm water sits *at or below* the −22 dB NESZ, so it is partly instrument noise | variable by sub-swath |
| S2 index drift (turbidity, algae, drawdown) | 0.1–0.3 index units |
| Absolute sensor calibration | **≤0.5 dB (3σ)** — negligible |

Against a pond-vs-non-pond contrast of order 5–10 dB, that is a large nuisance. So level *is* both
the primary signal and the primary nuisance — the physics is real.

**But note the two-layer correction now in play.** §5c showed the −0.051 detrend run never removed
amplitude at all, so it was never evidence about this. The physics here says the *conclusion* was
nonetheless roughly right — for reasons we hadn't measured. That is a much weaker footing than we
thought we had: **we have a plausible physical story and zero experimental evidence.** The
`seq.channels.rank` probe (as a *replacement* for `vals`) is the test we have never run.

### VH−VV: third independent rejection

The physics agent agrees with the feature agent and the math audit. It adds that the same argument
kills **SDWI** (∝ ln(10·VV·VH) = the *sum* in dB — also a linear map, also already representable),
and warns that the only *non*-invertible version — feeding VH−VV and **dropping** the co-pol level —
deletes signature #1, the primary pond discriminator, and should be expected to lose −0.02 to −0.05.

### The structural blind spot we must simply accept

With lat/lon removed and a per-cell encoder, the model **cannot represent geometry** — pond
rectangularity, size, dike double-bounce edges, neighbourhood arrangement. That is precisely the
mechanism the entire literature uses to separate aquaculture from natural lakes, rivers and
reservoirs (Ottinger's method is explicitly object-based). **A residual confusion with small natural
water bodies is baked into this competition and no reframe fixes it.** Likewise acquisition geometry
(incidence angle), the dominant nuisance, is unobserved and therefore un-normalisable.

### Independent convergence worth noting

The physics agent's Idea C — **collapse the ~11 collinear S2 missing-indicators to a single
"S2 observed" bit** — is the same proposal as the feature agent's **R2** (24→14 channels), reached
by a completely different route. The physics route adds a reason the feature route missed: **cloud
frequency is climatological, so the missing-indicator block is a direct proxy for calendar season
and monsoon phase** — i.e. an 11-dimensional redundant handle on exactly the phase-locked axis whose
deletion won us +0.0128. Two independent derivations of a capacity-*reducing* structural deletion is
the strongest queue signal this round produced.

---

## 6. THE DECISION TREE — what to do next, for both outcomes

Everything hinges on **iter11**: does an offline estimator rank our seven known-LB anchors correctly
(detrend + K4 below reltime + xview, Spearman ρ > 0.7)?

### STEP 0 — runs in BOTH worlds, immediately, for zero submissions

1. **R1 partial-S2 missingness audit** (§5) — is there a masking pattern in test our training views
   can never generate?
2. **Iterated adversarial channel attribution** (§5b) — per-channel-group adversarial AUC after
   relative-time, crossed with OOF permutation signal. Produces a *ranked deletion candidate list*.
3. **Post the two-column legality question** to the Zindi forum (§4).
4. **Start the pretrained lane build** (§1) — TabPFN v2 and Presto. Building costs nothing;
   *submitting* is what costs, and that decision is made per-branch below.

### ═══ SCENARIO A — iter11 PASSES (we have a working offline screen) ═══

The measurement constraint is broken: we can rank candidates locally and spend submissions only on
things that already look like winners. ~80 submissions become a real search budget.

**A1. Screen everything offline before submitting anything — starting with the pretrained lane.**
*(0 submissions)* Push TabPFN v2, Presto, R2, R3, R4, the adversarial-deletion candidate from
Step 0, and fold-ensemble deletion through the validator. Submit only what **≥2 estimators** rank
above the champion. *Reason:* the pretrained lane is the only untried lever in the +0.05 class that
our own ledger has ever seen (the GBDT→Transformer swap), and a validator is exactly what lets us
try a whole new model class without gambling submissions on it. The real failure mode here is
building the validator and then reverting to blind submission out of habit.

**A2. Re-open the GBDT lane with ratio-only features.** *(0–1 submissions)* A credible 90s-club
competitor says CatBoost works; we killed our GBDT after feeding it *absolute* features — exactly
what sdv says fails under temporal shift. Rebuild it on **strictly relative/ratio** features (band
ratios, month-to-month ratios, rank-within-series, NDWI/MNDWI) and score it offline. *Reason:* this
is the one place where credible competitor evidence directly contradicts a conclusion of ours, and
the validator makes the test free.

**A3. Build the ensemble — but only once the revisit trigger is met.** *(1–2 submissions)* If A1 or
A2 yields a second model **within ~0.01 of champion with visibly different errors** (check ρ on test
predictions, no labels needed), screen the 2-component blend offline and submit it. *Reason:* the
anti-ensembling law's real content is "no weak or correlated components" (§5b) — our champion is
already a 5-fold ensemble. An equal-quality, decorrelated partner is the case the law never tested,
and it is the strongest available hedge for a 721-row private LB we cannot see.

### ═══ SCENARIO B — iter11 FAILS (no offline signal; noise floor stands) ═══

Back to blind submission at ±0.012 unpaired resolution. Fund **only** ideas with plausible effect
≥ +0.013 — with one important relaxation from §5b: **paired A/Bs against the champion resolve at
~0.006**, so we are less blind than we thought. Strategy shifts from *search* to **getting a
validation signal by a different route**, plus a few high-conviction structural probes.

**B1. Build a TEMPORAL holdout instead of an unlabelled-data estimator.** *(0 submissions)*
iter11 estimates from unlabelled test rows. The untried alternative: train has **12 full months**,
test has **4–6 consecutive** — so we can build a holdout that *mimics the real task* by validating on
truncated windows from held-out months, and even simulate a within-train period shift. Precedent is
strong: LANL's 1st place **subsampled train to the cycles most resembling test** and trusted that CV
entirely; IEEE-CIS winners used train-months / skip-gap / predict-months schemes. *Reason:* our OOF
is anti-correlated precisely because it measures *in-period, full-window* performance — a different
quantity from the task. This is a completely different mechanism from ATC, so iter11 failing tells us
nothing about whether it works.

**B2. Fund the structural deletions, in strict order.** *(2–3 submissions)* The Step-0 **adversarial
deletion candidate** first (it is the Malware-winner move and the highest expected value at
+0.005–0.013), then **R2 (24→14 channel compression)**, then **R3 (antithetic views)**. *Reason:*
all three are capacity-neutral-or-reducing structural changes — our only winning family (+0.0128,
+0.0047) — and each targets a redundant, wasted, or genuinely shifted channel rather than adding
anything. R3 in particular *multiplies the one mechanism already proven to work*. Note §5b's screen:
delete only channels that are shifted **and** carry no absolute-signal content — that is why calendar
position won and level lost, and it is now a predictive rule rather than a post-hoc story.

**B3. Gamble one submission on the pretrained lane, then convert scarcity into rubric points.**
*(1–2 submissions)* Even blind, TabPFN or Presto is worth one submission: it is the only lever sized
like the gap. Then write the reproduction README, pin the environment, and **designate the two
finalists** — champion 0.8955 plus, per §5b, *the best cross-family model if one lands within 0.01*,
falling back to NoPE 0.8917 only if none does. *Reason:* if the LB proves unmoveable, the cheapest
remaining points are in the other **35%** — and the ITU Cropland precedent (same organiser family,
40% report weight) shows that channel is real, low-variance, and rewards exactly the documented
ledger we already have.

---

## 7. Unfinished agent briefs (killed by the spend limit — relaunch cheaply)

1. **Pond physics under temporal shift.** Ottinger 2017/2022 signatures; which are SEASON-INVARIANT
   vs season-dependent (decisive, because a phase-locked signature cannot transfer across periods);
   hard-negative separation (rice paddy is the key case — phase-locked flooding vs pond permanence);
   interannual backscatter drift; and the sharpest open question: **is low temporal dispersion itself
   season-dependent?** If it is, iter12 fails the way detrending did.
2. **Mathematics + validator audit.** The most important single item: **ATC predicts *accuracy*, but
   our metric is 0.6·F1 + 0.4·AUC, and `tools/offline_validate.py` computes ATC on *pre-shift*
   probabilities while computing MARG on *post-shift* ones.** Whether accuracy is a valid proxy here,
   and whether the gate's ρ > 0.7 on n=7 is even statistically meaningful when 4 of the 7 anchors lie
   within 0.005 of each other (i.e. inside the LB noise band), are both unresolved and both
   potentially invalidate the iter11 decision rule.
3. **Internal code audit.** Prime open question: does OOF get produced under the *same window/masking
   regime* as test predictions? Train rows have 12 months, test rows 4–6. A regime mismatch would be a
   direct mechanical explanation of the OOF anti-correlation.

---

## 8. Immediate actions

| # | Action | Cost | Branch |
|---|---|---|---|
| 1 | Correct the "no pretrained models" error in all planning docs | 0 | both |
| 2 | Recalibrate the target: pre-reset 0.95/0.98 are leaked; real bar ≈0.90–0.95 | 0 | both |
| 3 | Record that all 7 anchors post-date the 25 Jun reset → iter11 retro-fit is VALID | 0 | both |
| 4 | Adopt the measurement protocol: paired ≥0.006 suggestive / ≥0.012 confident | 0 | both |
| 5 | Run **iter11** on Colab; paste back the RETRO-FIT + GATE lines | 0 subs | — |
| 6 | Run **R1** partial-S2 missingness audit | 0 subs | both |
| 7 | Run **iterated adversarial channel attribution** → deletion candidate list | 0 subs | both |
| 8 | Post the two-column legality question to the Zindi forum | 0 | both |
| 9 | Remove **VH−VV** from the gated backlog | 0 | both |
| 10 | Drop the endgame prevalence sweep to low priority (t\*≈F1\*/2 ⇒ saturated) | 0 | both |
| 11 | Then branch on the iter11 gate into Scenario A or B above | — | — |

### Open questions carried forward

- **Does "month 01" in train mean the same calendar month as in test?** Unanswered on the forum
  (21 Jul). If the indices are *not* calendar-aligned, relative-time's win is even better explained
  and several seasonal ideas become unworkable. Worth asking.
- **ATC predicts accuracy; our metric is 0.6·F1 + 0.4·AUC**, and `tools/offline_validate.py` computes
  ATC on *pre-shift* probabilities but MARG on *post-shift* ones. Unresolved (§7.2) — this could
  weaken the iter11 decision rule even if the gate passes.
- **Is ρ > 0.7 on n=7 anchors statistically meaningful** when 4 of the 7 lie within 0.005 of each
  other (inside the noise band)? Consider restricting the gate to anchor pairs whose LB gap exceeds
  the paired floor of 0.006.

**Sources:** competition overview, data, rules, evaluation and discussion pages at
<https://zindi.africa/competitions/geoai-aquaculture-pond-identification-challenge> (accessed
2026-07-22), specifically discussion threads 33587 (Challenge Update), 33903 (90s club),
33364 (temporal shift), 33378 (data leakage), 34056 (months in datasets).
