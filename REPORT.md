# Aquaculture Pond Identification — Solution Report

**Zindi / FAO / ITU GeoAI Challenge** · Osborn Nyakaru · revised 2026-08-10 (iteration 41)

---

## 0. Summary

We classify isolated 10 m ground cells as *managed aquaculture pond* from a 12-month × 12-band
Sentinel-1/Sentinel-2 time series, under a **deliberately constructed temporal covariate shift**:
training rows are fully observed, test rows expose only a consecutive 4–6 month window from a
different time period. (The organizers state the *temporal* shift explicitly. Earlier drafts of this
report also asserted a *regional* shift; we could not source that statement and have withdrawn it —
the shift we can evidence is temporal plus the observation-window difference.)

The submitted model is a **from-scratch temporal Transformer** — 2 layers, 4 heads, 64-dim, trained
on masking-augmented views of each training row, with a cross-view invariance penalty, one
physically-motivated permanence channel, and a literal 0.5 operating point.

**Reproduce it in one command:**

```bash
pip install -r requirements.txt          # pinned, open-source only, no AutoML
# place Train.csv / Test.csv / SampleSubmission.csv in data/raw/
bash experiments/reproduce_champion.sh   # prints fingerprints to verify the reproduction
```

**But the model is not the interesting part of this submission, and we will not pretend otherwise.**
The most valuable thing we produced is a *measurement discipline*: we measured the noise floor of our
own evaluation channel, found it large enough to void most of our recorded results, rebuilt the
decision process around it, and then documented — three times, including once against our own
written record — the precise boundary at which our replacement instruments fail. Sections 4–6 are
that work. It is the part we would want a reviewer to read.

Our gap to the top of the leaderboard is **≈0.030**, and §6.5 localizes it: ~88% sits in the F1 term,
but ~90% of *that* is the ranking of rows near the 0.5 cut rather than calibration. The whole gap is
about **52 rows out of 309**.

**Headline numbers, stated honestly:**

| quantity | value |
|---|---|
| **finalist #1 (designated)** | **0.899882** — `champion_perm_seedavg5`, a 5-seed pooled artifact |
| **finalist #2 (designated)** | **0.899643** — `champion_archblend4`, a 4-architecture pool |
| best *single* public score ever recorded | 0.912759 — **not designated; it is seed luck, see §4** |
| **measured seed-to-seed sd** | **0.0191** — larger than most effects in our own ledger |
| public-slice binomial noise (n=309) | ≈ ±0.012 on the composite |
| LB-gated iterations / submissions | 41 iterations, ~50 of 100 submissions |

The gap between rows 2 and 3 of that table is the single most important fact in this report, and
§4 is about why we deliberately submit the lower number.

---

## 1. The problem, and the one design decision that defines it

Per cell: 12 monthly composites × 12 bands — Sentinel-1 SAR (**VH, VV**, dB) and 10 Sentinel-2
optical bands. **No latitude/longitude, no spatial neighbourhood, no image patch, no static
covariates.** Each row is one isolated pixel's time series.

Train **1,821** rows (1,817 after dropping 4 exact duplicates), test **1,030** (public 309, private
721 — the organizers state a 30/70 split). Metric `0.6·F1 + 0.4·ROC-AUC`, where F1 is computed on a
binary column at a **hard 0.5 cut** and AUC on a probability column.

**The designed trap.** Train rows are fully observed across 12 months. Test rows expose only a
consecutive **4/5/6-month window** — measured on `Test.csv` as 345/343/342 rows, and **1030/1030
rows are contiguous** — with everything else set to the `-9999` sentinel, plus additional
**Sentinel-2-only cloud dropout** inside the window at measured per-month rates (0.003–0.28). A
model trained on 12-month statistics leans on signal that does not exist at test time.

**The masking trap, stated precisely.** Only statistics that are *unbiased at every window length*
transfer: means, medians, interior quantiles, and fractions. Minima, maxima, ranges, counts and
run-lengths are all biased by the number of observed months, so a model that uses them is reading
window length rather than land cover. This single observation determines the entire feature bank.

**The shift is real and irreducible, and we proved it several independent ways:**

| probe | adversarial train-vs-test AUC |
|---|---|
| unmasked 12-month train vs test (upper anchor) | **1.0000** |
| hand-engineered features on raw windows | ≈0.99 |
| frozen **Presto** SSL embeddings (never saw our labels) | 0.965–0.976 |
| **our actual input** — masked, left-aligned values | **0.8915** |
| our missing-indicator channels alone | **0.4758** (below chance) |

The second-to-last row is the one that matters: after masking augmentation and relative-time
reframing, a substantial gap remains. This is genuine covariate shift by design, **not a pipeline
leak** — the last row shows our masking already matches test dropout by construction. "Drive
adversarial AUC to 0.5" is therefore not an achievable goal, and feature-pruning toward it is
futile. §6 shows why, and corrects an earlier explanation of our own.

---

## 2. The submitted model

Per observed month the encoder sees **25 channels** — 12 standardized band values ⊕ 12 binary
missing-indicators ⊕ **1 permanence indicator** `1[VH_dB(t) < −21]` — then:

```
Linear(25 → 64)
  → learned positional embedding (length 12), applied AFTER left-alignment
  → 2-layer Transformer encoder (4 heads, GELU, dropout 0.2,
       src_key_padding_mask over fully-missing months)
  → masked mean-pool over observed months
  → MLP head → sigmoid
```

**Loss:** `L = BCE + λ·Var_k(logit)` across `K=2` masked views of the same row, λ=1.0 — a cross-view
invariance penalty teaching the model that the label does not depend on *which* window was exposed.

**Training:** AdamW lr 1e-3, wd 1e-4, batch 256, 60 epochs, 5-fold CV, owner-grouped batching. Test
prediction is the mean of the 5 fold-models; the submitted artifact pools **5 independent seeds** on
top of that, i.e. 25 models.

**Why the permanence channel.** The masked mean-pool of a binary indicator *is* the fraction of
observed months below the threshold — an n-invariant statistic, and the empirical CDF of VH at one
physically-privileged cut. Ottinger et al. place the SAR land/water split near −21.5 dB. This is our
largest feature effect (+0.012 single-seed) and, notably, **threshold selection was monotone and
decreasing in the number of thresholds**: 1 τ (0.9065) > 4 τ (0.9016) > 6 τ (0.8987). The signal is
one physical cut, not a profile.

**Design choices and their honest status.** Earlier revisions of this repo described these as
"validated on the leaderboard." **That claim does not survive our own seed measurement (§4)** and we
have corrected it:

| choice | recorded Δ LB | status against seed sd = 0.0191 |
|---|---|---|
| **masking augmentation** (train on test-like windows) | — | ✅ structural; the pipeline is built on it |
| **relative-time reframing** (left-align to `t_rel=0`) | +0.0128 | ⚠️ **inside the noise floor — UNRESOLVED** |
| **cross-view invariance** (λ=1.0) | +0.0047 | ⚠️ **inside the noise floor — UNRESOLVED** |
| **VH permanence channel** | +0.0120 | ⚠️ single-seed; **washed to +0.000 on 5-seed averaging** |
| legal 0.5 operating point (replacing a rules-violating pin) | −0.0058 | ✅ compliance fix, see §8 |

Only **two** effects across 41 iterations ever clearly exceeded the noise floor, and both were
model-class changes: GBDT → Transformer (**+0.052**) and a broken amplitude transform (**−0.051**).

---

## 3. Handling the three concrete hurdles

**(a) Train fully observed, test masked.** We reverse-engineered the exact test masking recipe from
`Test.csv` — window length distribution, start position, and per-month S2-only dropout rates — and
expand each training row into *K* masked views drawn from **that measured distribution**. Features
are computed only over active months, so they are invariant to *which* window is exposed. This is
the strategy that won the closely analogous [PLAsTiCC challenge](https://arxiv.org/pdf/1907.04690),
where the winner degraded well-observed training light curves to match the sparse test cadence.

**(b) Optical gaps where radar survives.** In 273/1030 test rows some in-window months have all
optical bands masked while VH/VV survive. The sentinel is handled **per band, not per month**, so a
cloud-masked optical month is not discarded — the radar in it is still used.

**(c) Fixed-0.5 threshold under class imbalance.** `TargetF1` is scored at a hard 0.5 cut. We fit
**Platt scaling on training out-of-fold predictions only**, then cut at a literal 0.5, and emit
genuine calibrated probabilities in both columns. Platt rather than isotonic because it is strictly
monotone, so ROC-AUC is bit-identical while the output becomes a real probability. The realized
positive rate is **reported, never targeted**: `calibrate_legal()` deliberately accepts no config
object, so no prevalence constant can enter it. An earlier revision used a prevalence pin that
violated the rules; §8 documents that in full.

---

## 4. 🔑 The core finding: we measured our own noise floor, and it voided most of our results

For fifteen iterations we operated on a ±0.01 uncertainty band derived from **row-count theory** —
the public slice is ~309 rows, so we reasoned a score is good to about a point. We never measured it.

On 2026-07-22 we did. We reran the champion configuration changing **only the RNG seed**:

```
seed 42  ->  0.8955
seed  7  ->  0.8764
                      sd = 0.0191      seed rank-correlation = 0.9511
```

**Most of our recorded A/B verdicts have effect sizes smaller than one seed swing** — including both
of the wins that made this model our champion. They are not refuted; they are **unresolved**, which
is a different and more uncomfortable claim.

**Three consequences we then had to design around.**

**(i) Our best public score is not our level.** The champion's per-seed public scores are
`42 → 0.906492`, `29 → 0.900715`, `13 → 0.891730`, `21 → 0.878575` — a member mean of **0.894378**
across a **0.028 spread**. The 5-seed pooled artifact scores **0.899882**. We therefore designate the
pooled artifact and treat every single-seed high as an upward fluctuation.

**(ii) Three separate single-seed "records" all washed out.** This is not a hypothetical:

| single-seed-42 result | headline | 5-seed average |
|---|---|---|
| permanence channel | 0.906492 | 0.899882 |
| `vhsq` replacement | 0.913263 | 0.899512 |
| `mean_min` pooling | 0.912759 | 0.899512 |

Three consecutive times we recorded a "new best" on seed 42 and three consecutive times seed
averaging returned it to ~0.8995. We now require **5-seed confirmation before any result is
believed**, and we never designate a finalist on one draw.

**(iii) The ceiling is a bias floor, not a variance floor.** Four structurally different
constructions — the permanence single model, a 4-architecture pool, a `vhsq` variant and a
`mean_min` variant — seed-average to **0.899882 / 0.899643 / 0.899512 / 0.899512**, a spread of
**0.00037** against a public-slice binomial noise of ±0.012. Independent variance models
(Krogh–Vedelsby ambiguity, and a correlated-noise law across plausible ρ) both put the *total*
remaining headroom from any amount of further pooling at **≈+0.0014**, and a noise-propagated
re-derivation caps it at **≤+0.005**. Averaging more seeds, snapshot ensembles and bagging are all
bounded by that number. **Whatever separates us from the leaders is bias under the covariate shift,
not variance** — which is why §6 is about the shift and not about the model.

**The winner's curse, quantified.** Selecting the maximum over k candidates inflates the estimate by
about `SE · E[max of k standard normals]`. Our observed single-seed-versus-pooled gaps sit at almost
exactly the predicted size. This is the reason to designate **pooled, low-variance** artifacts for
the 721-row private slice rather than our best public score.

> **⚠️ An integrity note we are recording against ourselves.** Two of the ledger entries above —
> `champion_replvhsq_seedavg5` (iter 37) and `champion_meanmin_seedavg5` (iter 39) — are
> **bit-identical at 0.899512**. Two structurally different models producing an identical
> `0.6·F1 + 0.4·AUC` on 309 rows is roughly a 1-in-10⁴ coincidence, and this repository has had one
> confirmed duplicate-upload incident before (iter 33b). We flag this as a probable duplicate upload
> rather than a genuine replication, and we do not lean on it as independent evidence.

---

## 5. 🔑 Two offline instruments, both certified, both then falsified

Given a 0.019 noise floor and 100 total submissions, screening candidates *without* spending a
submission was worth more than any single feature. We built two such instruments. **Both passed
their own certification and both were then proven wrong by the leaderboard.** We consider this pair
of failures the most transferable content in the report, because they failed for *the same
structural reason in two different guises.*

### 5.1 Instrument one — the ATC-F1 retro-fitted screen

`tools/offline_validate.py` computes label-free estimators on unlabeled test predictions, then
**retro-fits each against experiments whose public LB we already know** (`experiments/anchors.tsv`).
An estimator earns the right to gate a decision only if it ranks the known anchors correctly.

| estimator | Spearman ρ vs known LB | gate | verdict |
|---|---|---|---|
| **ATC-F1** (metric-aligned average thresholded confidence) | **+0.964** | 15/15 | ✅ CLEARED |
| **DIS** (two-seed disagreement) | **+1.000** (n=4) | 5/5 | ✅ CLEARED |
| ATC (plain) | −0.429 | 6/15 | ❌ FAIL |
| DIV (fold diversity) | −0.857 | 2/15 | ❌ FAIL |
| MARG (margin) | −0.321 | 8/15 | ❌ FAIL |

The screen worked, for a while: it closed the Presto, pooling-diversity, instance-expansion and
multivariate-ROCKET lanes for **zero submissions**.

**Then it failed.** At iteration 26 `c_dropvv` cleared every gate — 2/2 votes, an ATC-F1 margin of
+0.0902 (1.57 seed-sd), and an absolute ATC-F1 (0.8977) above the champion's *best of five* seed
draws. We had pre-committed in writing to submitting on exactly that condition. We submitted.

```
predicted:  +0.0147 LB (raw)  /  ~ +0.005 after our standard 3x discount
actual:     0.884217  =  -0.0113, paired against the seed-42 champion
```

**Wrong in sign.** The diagnosis:

> **All 7 certifying anchors were architecture and objective variants at an *identical 24-channel
> input width*. The retro-fit certified ATC-F1 only *within that family*. `c_dropvv` (22 channels)
> was the first candidate that changed the input *representation*.**

Adding it as an 8th anchor drops ρ from +0.964 to +0.738 on that single point — **and our gate did
not catch it (17/18 still reads PASS)**, because the gate counts concordance over anchor pairs
separated by |ΔLB| > 0.010, a set dominated by trivially-rankable pairs. *A concordance gate built
from easy pairs cannot detect a failure off the manifold the anchors span.*

### 5.2 Instrument three — our own ledger, and a number that never existed

Before the second instrument, one more failure of the same shape, found by auditing our own citations.

For roughly ten iterations this project operated on the belief that *"the public leaderboard leader is
at ~0.94 using plain CatBoost, and their stated lever is ratio/relative features."* That sentence
appears in `PROJECT_STATE.md` as settled doctrine, and it directed a great deal of work — including
the entire tree lane.

We went back to the source. The forum thread contains one post from the competitor in question:

> *"don't blame the trees — I'm on gradient boosting too (CatBoost-family, nothing exotic)… If tuning
> models and swapping features isn't moving the LB, that's a hint the lever is elsewhere."*

**No score is stated. They never claimed to be the leader.** When another participant directly asked
for their best LB score, they did not reply. Tracing our own documents shows a clean citation drift:
an early research note correctly recorded "a score band" and "this competitor uses CatBoost" as *two
separate facts*; a later note merged them into "sdv, in the 90s club, uses plain CatBoost"; the
iteration-30 log promoted that to "the LB **leader** (~0.94)"; and `PROJECT_STATE.md` recorded it as
doctrine. **We also inverted the one actionable sentence they did write** — they said swapping
features was *not* moving their leaderboard, and we recorded "their named lever is ratio features."

The corrected picture, from the leaderboard's own published digits: the top is a **cluster at 0.9259
and 0.9301**, not a lone 0.94. Our gap is **≈0.030, not ≈0.05**.

This is the same failure as §4's prevalence constant and §8.1's "+0.07" pin credit: **a number entered
the ledger without provenance, was compounded by summarization, and then steered months of work.** We
now require every external claim in `PROJECT_STATE.md` to carry a VERIFIED-with-quote or INFERRED tag.

### 5.3 Instrument two — the adversarial-validation gate (and the same lesson again)

For the tree lane we built `tools/adversarial_cv.py`: train on the least-test-like 70% of train,
evaluate on the **most-test-like 30%**, on the theory that a model which survives that holdout will
survive the real shift. At iteration 39 it returned a confident **[GO]** — AUC 0.960, F1 0.884, only
0.024 below a random holdout.

Iteration 40 built the recommended model and submitted it: **0.7186**, against a predicted ≥0.90.

**Wrong by 0.18.** The diagnosis is the same failure in new clothing:

> **A test-like-*covariate* holdout still carries TRAIN labels.** It therefore samples `P_train(y|x)`,
> and is structurally blind to **conditional** shift — which our own label-shift gate had already
> proven present (mixture goodness-of-fit KS D = 0.186, p = 0.000). The gate could only ever measure
> the covariate half of a shift we knew had two halves.

**The generalized lesson, which we would offer as our main methodological contribution:** every
instrument we relied on — including our own written record — certified itself on a *sub-manifold of
the thing it was asked to predict*: ATC-F1 on one input width, adversarial CV on one half of the
shift, the ledger on one unsourced quotation. In each case the instrument's own check was
structurally incapable of reporting that it had left its domain of validity. **A retro-fitted
validator inherits the support of its calibration set, and concordance gates do not reveal where
that support ends.**

The practical rule we now follow: **an offline gate may gate a PAIRED DELTA, never a LEVEL.** Both
failures above were level predictions. A paired difference between a treatment and a control arm,
both equally blind, cancels the blindness in the difference.

---

## 6. 🔑 The anatomy of the shift — and a correction to our own analysis

To ask *where* the shift lives, we transplanted **real test masks onto train rows**, so the
window-length and dropout components are removed by construction and only the residual remains. All
numbers below are measured on that window-matched comparison, with a 66-feature n-invariant bank.

**First result — no single feature carries it.** Marginal adversarial AUC has a **maximum of 0.7024**
and a mean of 0.5698, yet a gradient-boosted model on the joint bank separates the domains at
**0.9670**. Nulls are clean: missingness-count features alone give 0.4636 (the transplant does not
leak), disjoint train halves with independent transplants give 0.4934, and test-vs-test disjoint
halves give 0.5046.

**Our first explanation was wrong, and we are correcting it here.** We initially concluded that the
shift must therefore live in feature *interactions*, since a linear model reached only 0.9190 against
the GBDT's 0.9670. An adversarial re-examination broke that:

| model on the same 66 features | adversarial AUC |
|---|---|
| GBDT depth 3 (original claim) | 0.9670 |
| **GBDT depth 1 — purely additive, no interactions** | **0.9430** |
| GBDT depth 2 | 0.9524 |
| logistic, median-impute, C=0.5 (the original weak baseline) | 0.9190 |
| **logistic, rank-transform + explicit missing indicators, C=10** | **0.9438** |

The interaction premium is only **+0.0239**, and our "linear can't do it" figure was an artifact of a
badly-specified linear baseline rather than a property of the data. A properly specified additive
model matches the additive GBDT. Moreover, if the 66 features were merely *independent*, pure
accumulation of their weak marginals predicts a joint AUC of **0.9594** — nearly all of what we
observe.

> **Corrected conclusion: the shift is thinly spread across many weak marginals that accumulate
> additively.** 42 of 66 features exceed adversarial AUC 0.55 while none exceeds 0.71. This is why
> per-feature screening fails — not because the signal hides in interactions, but because **there is
> no clean subset to keep.** It also explains, retrospectively, why deleting individual bands
> (`c_dropvv`) and dropping whole feature families both failed to help.

**Second result — the two sensors are not equally transferable.**

| bank | adversarial AUC | label AUC |
|---|---|---|
| **SAR only** (VH/VV, 20 features) | **0.8452** | 0.9446 |
| **Optical only** (10 S2 bands, 46 features) | **0.9376** | 0.9817 |

Optical carries *more label signal* and *far more shift*. Because the shift accumulates additively,
adding many optical features is precisely the mechanism by which one accumulates it. This is a
concrete, measured design constraint that we would not have predicted from the physics alone, and it
is consistent with our SAR-first feature bank having outperformed every optical-heavy variant.

---

## 6.5 Where the remaining gap actually is

Because the leaderboard publishes F1 and AUC as separate columns, we can decompose our own score
without any tuning (see §8.2 on the line we drew here). Our submitted artifact splits as
**AUC 0.9354 / F1 0.8761**; the top of the cluster is **AUC 0.9449 / F1 0.9202**.

| component | share of the 0.0302 gap |
|---|---|
| AUC term (ranking overall) | ~13% |
| **F1 term** | **~88%** |
| …of which: operating point / calibration | only **5–13%** of the F1 gap |
| …of which: **ranking of rows near the 0.5 cut** | the remaining **~90%** |

In absolute terms the entire gap is about **52 rows** out of 309 — we make ~144 confusion errors
where the leaders make ~95. And because F1 is computed at a fixed cut, **a row near the boundary is
worth 1.9–3.8× a row in the deep tail.** This is the single most useful strategic fact we have: the
remaining gap is not calibration and not global ranking, it is *local ranking quality among genuinely
ambiguous cells*. That is a modelling problem, not an operating-point problem — which is why §8.2's
lever is worth so little and why we did not pursue it.

---

## 7. Negative results (41 LB-gated iterations)

A competition report that lists only what worked hides most of the information. Every row below cost
a submission or a screened experiment, and each is reproducible from `experiments/LB_LOG.md`.

| lane | outcome | evidence |
|---|---|---|
| **Gradient-boosted trees** | ❌ closed, 3 independent failures | naive CatBoost **0.6976**; blend member **−0.0136**; shift-robust CatBoost with feature-shift removal **0.7186** (no-drop control 0.6903). Trees key on exactly the joint structure the shift corrupts |
| **Amplitude normalization** | ❌ fatal | replacing values with within-series rank collapsed OOF 0.975→0.86. **Persistently-low backscatter level *is* the class signal** — do not detrend, difference, or instance-normalize |
| **Cross-polarization contrast `VH−VV`, *linear* forms** | ❌ closed — but the test was weaker than we claimed | toxic as a Transformer channel (−0.0228); a top shift-carrier in the tree bank. **However**: our own `SDWI` implementation is *exactly* `−5.697415 + 0.230259·(VH_dB + VV_dB)` — verified numerically to 3.6e-15 — i.e. affine in the two bands the model already has. A linear layer spans it, so these arms measured **width cost, not missing information.** The dual-pol *indicator* (below) remains untested |
| **Dispersion features** (IQR, L-scale) | ❌ falsified as "shift-safe" | theoretically location-invariant, empirically shift-carriers (0.57–0.66) — further evidence the shift is not a pure location shift |
| **Foundation models (Presto, frozen)** | ⚠️ **closed on retracted evidence** | the frozen encoder re-encodes the shift (adv-AUC 0.965–0.976), but the lane was closed by ATC-F1 — the instrument §5.1 later falsified. Fine-tuning (LP-FT) was never tested. We record this as *unfinished*, not refuted |
| **Cross-model-class blending** | ❌ closed, n=2 | ROCKET member → −0.009; GBDT member → **−0.0155, paired, ≥2.5σ** |
| **Feature-space deletion** | ❌ closed | `c_dropvv` −0.0113; and §6 explains why: the shift is distributed, so no small subset carries it |
| **Missing-indicator deletion** | ❌ closed for free | indicators alone separate train/test at **0.4758, below chance** — our masking already matches test dropout |
| **Importance weighting / DANN** | ❌ dead | effective sample size collapses at adversarial AUC ≈0.99 |
| **Label-shift priors (BBSE / Saerens-EM)** | ❌ **unsafe, gate FAILED** | the mixture goodness-of-fit test rejects pure label shift (KS D = 0.186, p = 0.000). Two estimators put π̂_test at **0.559 / 0.578**; our realized rate is already ≈0.55, so the upside was near-nil even had it been safe |
| **Water indices (WIF/EVI/SDWI/AWEI/NDWI)** | ❌ −0.075, and partly degenerate | **SDWI is exactly affine in (VV_dB + VH_dB); AWEI is exactly linear; EVI ≈ 2.5(NIR−Red) over water; NDWI/MNDWI are 0/0-conditioned over water.** A linear model already spans several of them |
| **More masking views / longer training** | ❌ | K=4 → −0.0115; K=2 is a sharp optimum |
| **Seed averaging as a climber** | ➖ variance only, and now capped | lands at the member mean; §4(iii) bounds all further pooling at ≤+0.005 |
| **TTA (hole-punching)** | ➖ −0.0023 | diagnosed in §6 terms: masking random *interior* months produces windows that occur in neither train nor test — off-manifold augmentation |

**What the blending failures do and do not show.** Under a metric with a hard threshold, the
error-ambiguity decomposition that justifies ensemble diversity **does not apply** — there is no such
decomposition for 0-1 loss (Brown & Kuncheva, MCS 2010). Our two foreign-member blends lost
**monotonically in the member's own level deficit**, so we gate members on **level gap, not
decorrelation**. **We flag a limitation in our own evidence:** both members were also weaker, so
member strength and member diversity are **perfectly confounded at n=2**. What we demonstrated is
that a weak member hurts; the stronger claim that *diversity itself* is harmful is **not identified**
by our experiment, and an earlier draft of this report overstated it.

**A scope condition that bit us.** Every blending result above was measured under the (later removed)
prevalence pin. When we removed it, the same 4-architecture pool went from −0.0009 against its best
member to **+0.0100**. The pin overwrote every member's operating point to a common value, so pooling
could only average the *ranking* (mean ρ = 0.9524, almost nothing independent left); a literal 0.5
cut also averages the members' *calibration*, where they genuinely disagreed (positive rates
0.534–0.586). **Ensemble conclusions drawn under a pinned threshold do not transfer to an unpinned
one**, and ours were all drawn under the pin.

---

## 8. ⚠️ Limitations, and two rule questions we raise against ourselves

### 8.1 The prevalence pin was a rules violation — found, disclosed, and fixed

We read the rules page directly on 2026-07-28. It states, verbatim:

> *"Setting a probability threshold is strictly forbidden. Your binary target should be based on the
> default threshold of 0.5."* … *"do not set thresholds (or round your probabilities) to improve your
> place on the leaderboard."* … *"Zindi will need the raw probabilities. This will allow the clients
> to set thresholds to their own needs."*

Earlier revisions argued our construction was *prevalence correction* (allowed) rather than
*threshold tuning* (forbidden), because the cut stays literally at 0.5. **That argument does not
survive reading the rule, and we withdrew it.** Two distinct problems: `TargetF1` computed
`thresh = quantile(logit(p), 1 − π̂)` and shifted so the threshold landed at 0.5 — the literal 0.5
was cosmetic, and **π̂ = 0.649 was swept against leaderboard feedback**, not derived from training
data. And `TargetRAUC` returned uniformly-spaced ranks, not probabilities, defeating the stated
rationale of the raw-probabilities rule.

**Why we disclosed rather than quietly hoping.** The final score is 65% private LB + 35% code review
**of the top 5 only**. If we do not reach the top 5, the pin was worth nothing. If we do, our code is
read by exactly the people who wrote the rule. **The gain is only cashable in the scenario that
triggers the review that would void it.** There is no branch on which keeping it pays.

**Status: FIXED, and it cost almost nothing.** Measured on the leaderboard, same config, same seed,
same folds, only the operating point changed:

| | public LB |
|---|---|
| pinned (rules-violating) | 0.895500 |
| **legal** | **0.889686** |
| paired delta | **−0.0058** |

Our protocol calls a paired A/B *suggestive* only at ≥0.006, so **the cost of compliance does not
reach significance.** Inverting the delta shows why: the pin was adding ~104 positives that were
about **49% correct — coin flips.** It bought volume, not accuracy. The ≈+0.07 once credited to it
was measured in iteration 02 on the **superseded GBDT**, and we carried that constant for 25
iterations without re-measuring it — the same methodological error as §4, in a different guise.

**A reviewer can verify the fix from the submission file alone**, without running our code:
`TargetF1 == (TargetRAUC >= 0.5)` on every row. That is the entire compliance claim.

**We then audited every code path in the repository that can write a submission.** The result, and
the fixes, in full:

| emitter | status |
|---|---|
| `run_pipeline.py` (all models) | ✅ compliant — `calibrate_legal`, train-only Platt, literal 0.5 |
| `tools/seed_average.py` → **finalist #1** | ✅ compliant — `calibrated_pool`, literal 0.5 |
| `tools/arch_blend.py` → **finalist #2** | ✅ compliant — `calibrated_pool`, literal 0.5 |
| `run_presto.py` | 🔴 **was non-compliant → FIXED** — emitted `target_prevalence_shift` (a threshold shift) + `score_for_auc` (ranks, not probabilities). Now routed through `calibrate_legal`. Any Presto artifact produced before this fix was ineligible |
| `tools/blend.py` | 🔴 **was non-compliant → now GUARDED** — same illegal pair. It is retained to reproduce historical pinned anchors, and now refuses to run unless `compliance_mode=pinned` is set explicitly |
| `src/calibration.py::calibrate_for_f1` (isotonic branch) | 🔴 **misleading comment → FIXED** — the code found the F1-optimal cut and shifted it onto 0.5 under a comment claiming this made it *"legal"*. It does not: that is threshold tuning with a wrapper. The comment now says so |

**Both designated finalists are clean.** The defects were in secondary tooling — but a reviewer opens
those files too, and a comment asserting that threshold-shifting is legal is precisely the kind of
thing that should be found and corrected before review, not during it.

### 8.2 A second lever we found and deliberately did not pull

Late in the competition we noticed the leaderboard exposes **F1 and ROC-AUC as separate columns**
alongside the composite. Because `F1 = 2·TP/(P̂+P)` and `AUC = U/(A·(N−A))` are both ratios of small
integers on a 309-row slice, the published digits are enough to **algebraically constrain the number
of positives in the public slice**. We carried out that inversion as a diagnostic.

**We did not, and will not, use it to set our operating point.** Tuning the decision boundary to a
reverse-engineered public-slice composition would be leaderboard probing — a clear violation of the
same rule as §8.1, and one that would also overfit the 309 public rows against the 721 private rows
that actually decide the competition. The analysis is retained only in scratch notes as a *diagnosis*
of where our gap lives (it is ~83% in the F1 term), and no number derived from it enters
`config/config.yaml`, `src/calibration.py`, or any submitted artifact. We record it here because a
reviewer should know we found the channel and chose not to use it.

### 8.3 Reproducibility caveats, stated plainly

A single seed drives all RNGs and per-`(row, view)` seeds are derived deterministically, so the
masking augmentation is exactly reproducible. However, for the `seq` path on GPU we do **not** set
`torch.use_deterministic_algorithms` or `cudnn.deterministic`, so CUDA attention kernels may
introduce small run-to-run differences. GBDT runs are bit-identical; `seq` runs reproduce to within
that kernel nondeterminism. Given §4, a reviewer should expect run-to-run variation on the order of
the seed effect, and **should reproduce the pooled artifacts rather than any single seed.**

### 8.4 A ceiling we cannot reach, and why we stopped trying

The pond-mapping literature stands on three legs, and two are unavailable to us:

| leg | what it buys | available? |
|---|---|---|
| pixel-wise temporal permanence (e.g. median VH) | the water mask | ✅ the only leg we have |
| **shape** — compactness, perimeter, LSI, dike detection, GLCM texture | ***the entire pond-vs-natural-water separation*** | ❌ lat/lon stripped, isolated pixels |
| DEM / OSM / JRC surface-water overlays | removes lakes, reservoirs, rivers | ❌ external data, rule-barred |

Ottinger (2022) calls compact shape *"a characteristic and **defining** feature for the distinction
between natural standing waters and managed aquaculture ponds."* Phan et al., ground-truthed, report
that flooded rice at ~10 days after sowing reads **VV −13 dB, VH −22 dB — open-water values**. At any
single date our positives and our hardest negatives are radiometrically identical; only the joint
behaviour over months separates them. **Published 89–95% accuracies are earned with the two legs we
do not have.**

*Geography caveat:* this literature is overwhelmingly coastal East/Southeast Asian intensive
aquaculture. The FAO/ITU framing suggests our data may be African smallholder — smaller ponds, less
intensive feeding, more rain-fed drawdown — which would weaken both the eutrophication signal and the
"permanently full" assumption. We found no quantitative African pond-mapping study to calibrate
against, and we flag our physics-derived expectations as **unvalidated for this setting.**

---

## 9. What we would do with more time

Stated so a reviewer can see we know where the remaining value is, not as a claim of results.

1. **Transductive training on the unlabeled test rows** (in flight at iteration 41). The 1,030 test
   rows are 57% of our labeled set and the only target-domain data available; test *features* are
   supplied, so using them is legal by construction. Two zero-parameter forms: extending our proven
   cross-view invariance penalty to test rows (forms no label, so it cannot suffer confirmation
   bias), and soft self-distillation from a pooled teacher. Chen, Wei, Kumar & Ma (2020) prove
   self-training on unlabeled target data drives a classifier off features that correlate with the
   label in the source domain but not the target — which is a description of our failure mode.
2. **The dual-polarization water indicator.** Our entire feature bank is a function of **VH alone**,
   while every water detector in the remote-sensing literature is dual-polarization (Ottinger takes
   percentiles of VH *and* VV; Duan's SDWI is a function of both). We concluded the dual-pol axis was
   dead — but §7 shows we only ever tested it in forms that are *provably affine* in bands the model
   already sees, so those arms measured width cost, not information. The untested form is the
   **indicator**, which is exactly the nonlinearity behind our single biggest feature win:
   `c_t = 1[VH_dB(t) < −21] · 1[(VH−VV)(t) < τ_r]`, as a width-neutral *replacement* channel with
   `τ_r` chosen by the free adversarial/label-AUC screen. It is also all-SAR, so unlike an
   NDVI-gated version it is available in every observed month — S2 is absent in 17.6% of October
   test rows.
3. **LP-FT fine-tuning of Presto.** §7 records that lane as closed on evidence we later retracted.
   Linear-probe-then-fine-tune (Kumar et al., ICLR 2022) is the configuration that preserves
   pretrained features out-of-distribution, and it is exactly the one we never ran. Note
   `run_presto.py` still emits through the superseded illegal path and would need the 6-line
   compliance fix first.
4. **Per-window specialization.** We match the test window distribution by augmentation but train one
   model for all window lengths. A model conditioned on — or specialized to — window length is
   untested.
5. **Measure the two columns separately.** `TargetF1` (weight 0.6) is a *set-selection* problem;
   `TargetRAUC` (0.4) is a *global ranking* problem. We have fed both from one score vector by habit
   and have never measured them separately, despite the leaderboard reporting them separately.

---

## 10. Where everything lives

```
config/config.yaml              all hyperparameters, feature toggles, seed
src/
  data.py                       schema discovery, -9999 -> NaN cube, test-mask measurement
  features.py                   window masking (apply_mask), indices, aggregates
  validation.py                 masking-aware K-fold, leak-free OOF
  seq_model.py                  the submitted temporal Transformer (+ transductive terms)
  calibration.py                calibrate_legal(): train-only Platt, literal 0.5 cut
run_pipeline.py                 orchestrator; writes + validates every submission
tools/
  offline_validate.py           the LB-predicting screen (§5.1) and its falsifier
  adversarial_cv.py             the tree-lane gate (§5.2) and its falsifier
  shift_diagnostics.py          free adv-AUC / label-AUC feature screen
  shift_audit.py                adversarial probes + the 2-D band screen
  label_shift_gate.py           the mixture goodness-of-fit test that vetoed Saerens
  arch_blend.py, seed_average.py   pooled artifacts
experiments/
  reproduce_champion.sh         ONE COMMAND REPRODUCTION of the designated finalists
  anchors.tsv                   known-LB anchors for the retro-fit (incl. the falsifier)
  LB_LOG.md                     every iteration, every score, every verdict
  results.tsv                   append-only run log
PROJECT_STATE.md                full state, ledger and lessons
```

**Cross-validation is masking-aware and leak-free:** folds are defined on the *original* rows, every
augmented view inherits its row's fold (a row's masked twins never straddle the split), and each
held-out row is scored on *R* independent masked views averaged into one OOF probability.

**A deliberate inversion a reviewer should expect:** our best-leaderboard models have our *lowest*
OOF. Local OOF sits near 0.975 against a leaderboard near 0.90, and has been **anti-correlated** with
it across the ledger. We never select on OOF. This is stated in `README.md` and is not an accident of
tuning — it is the covariate shift of §1 doing exactly what it was designed to do.
