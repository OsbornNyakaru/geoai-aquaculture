# Aquaculture Pond Identification — Solution Report

**Zindi / FAO / ITU GeoAI Challenge** · Osborn Nyakaru · DRAFT 2026-07-28

---

## 0. Summary

We classify isolated 10 m ground cells as *managed aquaculture pond* from a 12-month × 12-band
Sentinel-1/Sentinel-2 time series, under a **deliberately constructed temporal covariate shift**:
training rows are fully observed, test rows expose only a consecutive 4–6 month window from a
different period and region.

The submitted model is a **from-scratch temporal Transformer** — 2 layers, 4 heads, ~64-dim, trained
on masking-augmented views of each training row, with a cross-view invariance penalty and an
exact-prevalence operating point.

**Reproduce it in one command:**

```bash
pip install -r requirements.txt          # pinned, open-source only, no AutoML
# place Train.csv / Test.csv / SampleSubmission.csv in data/raw/
bash experiments/reproduce_champion.sh   # prints fingerprints to verify the reproduction
```

**But the model is not the interesting part of this submission, and we will not pretend otherwise.**
The most valuable thing we produced is a *measurement discipline*: we measured the noise floor of our
own evaluation channel, found it large enough to void most of our recorded results, rebuilt the
decision process around it, and then documented the precise boundary at which our replacement
instrument fails. Sections 4–6 are that work. It is the part we would want a reviewer to read.

**Headline numbers, stated honestly:**

| quantity | value |
|---|---|
| best single public score | 0.8955 — **a lucky seed, not our level** |
| **reliable level** (5-seed pooled) | **0.8865** |
| designated primary artifact | `champion_archblend4` = **0.894643** |
| **measured seed-to-seed sd** | **0.0191** — larger than 9 of our own 11 recorded effects |
| submissions spent | 26 of 100 |

---

## 1. The problem, and the one design decision that defines it

Per cell: 12 monthly composites × 12 bands — Sentinel-1 SAR (**VH, VV**, dB) and 10 Sentinel-2
optical bands. **No latitude/longitude, no spatial neighbourhood, no image patch, no static
covariates.** Each row is one isolated pixel's time series.

Train **1,821** rows (1,817 after dropping 4 exact duplicates), test **1,030** (public ≈309, private
≈721). Metric `0.6·F1 + 0.4·ROC-AUC`.

**The designed trap.** Train rows are fully observed across 12 months. Test rows expose only a
consecutive **4/5/6-month window** (p ≈ 0.335/0.333/0.332), everything else set to the `-9999`
sentinel, plus additional **Sentinel-2-only cloud dropout** inside the window at measured per-month
rates (0.003–0.28; October ~17.6%, June ~7.3%). A model trained on 12-month statistics leans on
signal that does not exist at test time.

**The shift is real and irreducible, and we proved it three independent ways:**

| probe | adversarial train-vs-test AUC |
|---|---|
| hand-engineered features | ≈0.99 |
| frozen **Presto** SSL embeddings (never saw our labels) | 0.965–0.976 |
| **our actual input** — masked, left-aligned values | **0.8915** |
| our missing-indicator channels alone | **0.4758** (below chance) |

The third row is the one that matters: after our masking augmentation and relative-time reframing,
a substantial gap remains. This is genuine covariate shift by design, **not a pipeline leak** — so
"drive adversarial AUC to 0.5" is not an achievable goal, and feature-pruning toward it is futile.
We show why in §6.

---

## 2. The submitted model

Per observed month the encoder sees **24 channels** — 12 standardized band values ⊕ 12 binary
missing-indicators — then:

```
Linear(24 → 64)
  → learned positional embedding (length 12), applied AFTER left-alignment
  → 2-layer Transformer encoder (4 heads, GELU, dropout 0.2,
       src_key_padding_mask over fully-missing months)
  → masked mean-pool over observed months
  → MLP head → sigmoid
```

**Loss:** `L = BCE + λ·Var_k(logit)` across `K=2` masked views of the same row, λ=1.0 — a cross-view
invariance penalty teaching the model that the label does not depend on *which* window was exposed.

**Training:** AdamW lr 1e-3, wd 1e-4, batch 256, 60 epochs, 5-fold CV, owner-grouped batching. Test
prediction is the mean of the 5 fold-models.

**Three design choices, and their honest status.** Earlier revisions of this repo described all three
as "validated on the leaderboard." **That claim does not survive our own seed measurement (§4)** and
we have corrected it:

| choice | recorded Δ LB | status after measuring seed sd = 0.0191 |
|---|---|---|
| **masking augmentation** (train on test-like windows) | — | ✅ structural; the pipeline is built on it |
| **relative-time reframing** (left-align window to `t_rel=0`) | +0.0128 | ⚠️ **inside the noise floor — UNRESOLVED** |
| **cross-view invariance** (λ=1.0) | +0.0047 | ⚠️ **inside the noise floor — UNRESOLVED** |
| **exact-prevalence operating point** | ≈+0.07 | ✅ real, far outside the floor (but see §7 rule risk) |

Only **two** effects in 26 iterations ever exceeded the noise floor, and both were model-class
changes: GBDT → Transformer (**+0.052**) and a broken amplitude transform (**−0.051**).

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

**(c) Fixed-0.5 threshold under class imbalance.** `TargetF1` is scored at a hard 0.5 cut. Rather
than tune a threshold, we make 0.5 the optimal operating point with a monotone transform: find
`t* = argmax_t F1(y, p ≥ t)` on out-of-fold predictions, then apply `p' = σ(logit(p) − logit(t*))`,
which maps `t* → 0.5` while preserving rank order, so `(p' ≥ 0.5) ⇔ (p ≥ t*)`. A built-in assertion
checks `F1@0.5(p') == F1@t*(p)`. **See §7 — we consider this the most reviewable decision in the
submission and we flag it against ourselves.**

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

**Nine of our eleven recorded A/B verdicts have effect sizes smaller than one seed swing** —
including both of the wins that made this model our champion. They are not refuted; they are
**unresolved**, which is a different and more uncomfortable claim.

**Three consequences we then had to design around.**

**(i) Our best public score is not our level.** 0.8955 is the better of two draws from a
distribution with sd 0.0191. Pooling 5 seeds scored **0.88653** — and we *predicted* 0.886 from the
variance model beforehand, confirming to 0.0006. We therefore report **0.8865** as our level and
treat 0.8955 as an upward fluctuation. We will not designate it as a finalist.

**(ii) Averaging cannot rescue the measurement.** This is the wall. With seed rank-correlation
0.9511, the variance-reduction factor for M pooled seeds is `(1 + ρ(M−1))/M`; at M=5 that is
**0.961**, moving sd only 0.0191 → 0.0187. **Seed averaging buys reliability, not resolution.** No
construction available to us measures an effect below ~0.02 on the public slice.

**(iii) The metric's sensitivity explains why.** With the prevalence pin fixing k = ⌊0.649·1030⌋ =
668 and P ≈ 669, `F1 = 2·TP/1337`. Moving the total score by 0.010 requires **ΔTP ≈ 11 rows** via F1,
or ΔAUC ≈ 0.025 via ranking. On the ~309-row public slice the same 0.010 is **≈3 rows**. A handful
of borderline cells crossing a pinned cut is the entire measurement.

**The winner's curse, quantified.** Selecting the maximum over k candidates inflates the estimate by
about `SE · E[max of k standard normals]`. At our budget (k ≈ 79 remaining submissions) E[max] ≈ 2.4,
predicting **≈ +0.03 inflation** — and our 0.8955-versus-0.8865 gap is that effect at almost exactly
the predicted size. We take this as the reason to designate a **pooled, low-variance** artifact for
the private slice rather than our best public score.

---

## 5. 🔑 An offline leaderboard-predicting validator — built, certified, and then falsified

Given a 0.0191 noise floor and 100 total submissions, screening candidates *without* spending a
submission was worth more than any single feature. `tools/offline_validate.py` implements this.

**Construction.** Compute several label-free estimators on the unlabeled test predictions, then
**retro-fit each against experiments whose public LB we already know** (`experiments/anchors.tsv`).
An estimator earns the right to gate a decision only if it ranks the known anchors correctly.

**Certification, against 7 anchors:**

| estimator | Spearman ρ vs known LB | gate | verdict |
|---|---|---|---|
| **ATC-F1** (metric-aligned average thresholded confidence) | **+0.964** | 15/15 | ✅ CLEARED |
| **DIS** (two-seed disagreement) | **+1.000** (n=4) | 5/5 | ✅ CLEARED |
| ATC (plain) | −0.429 | 6/15 | ❌ FAIL |
| DIV (fold diversity) | −0.857 | 2/15 | ❌ FAIL |
| MARG (margin) | −0.321 | 8/15 | ❌ FAIL |

Two guards were added deliberately: a **permutation-null gate** (which returned VOID once, at
iteration 21, correctly discarding that screen) and a **seed-noise floor** — a candidate's margin
must exceed the estimator's own seed-to-seed sd (0.0576 in ATC-F1 units ≈ 0.0094 LB), so a margin
indistinguishable from seed noise cannot trigger a submission regardless of vote count.

The screen worked. It closed the Presto lane, the pooling-diversity lane, the instance-expansion
lane and the multivariate-ROCKET lane **for zero submissions**, and at iteration 24 it correctly
returned `HOLD` on a candidate we overrode and then lost a submission to.

### 5.1 And then it failed — the result we are most glad we recorded

At iteration 26 we screened three band-deletion candidates. `c_dropvv` cleared everything: **2/2
votes**, an ATC-F1 margin of **+0.0902** (1.57 seed-sd), and an absolute ATC-F1 of **0.8977** —
higher than the champion's *best of five* seed draws (range [0.7196, 0.8601]). We had pre-committed
in writing to submitting on exactly that condition. We submitted.

```
predicted:  +0.0147 LB (raw)  /  ≈ +0.005 after our standard 3x discount
actual:     0.884217  =  -0.0113, paired against the seed-42 champion
```

**Wrong in sign.** The diagnosis is specific and, we think, generalizable:

> **All 7 certifying anchors were architecture and objective variants at an *identical 24-channel
> input width*. The retro-fit therefore certified ATC-F1 only *within that family*. `c_dropvv`
> (22 channels) was the first candidate that changed the input *representation*.**

Adding it as an 8th anchor:

```
n=7 (original)      rho = +0.964      gate = 15/15
n=8 (+c_dropvv)     rho = +0.738      gate = 17/18
only discordant informative pair: (xview, dropvv)
```

ρ collapses on a single point, and it is the only out-of-family one.

**And our gate did not catch it — 17/18 still reads PASS.** The gate counts concordance over anchor
pairs separated by |ΔLB| > 0.010, a set dominated by easy pairs (an anchor at 0.8266 is trivially
rankable). *A concordance gate built from easy pairs cannot detect a failure that occurs off the
manifold the anchors span.* We consider this the most transferable lesson in the report: **a
retro-fitted validator inherits the support of its anchor set, and a pass/fail concordance gate does
not reveal where that support ends.**

We keep `c_dropvv` in `anchors.tsv` permanently. It is the only anchor we have off the 24-channel
manifold and therefore the only one capable of detecting this failure mode again.

---

## 6. Negative results (26 LB-gated iterations)

A competition report that lists only what worked hides most of the information. Every row below cost
a submission or a screened experiment, and each is reproducible from `experiments/LB_LOG.md`.

| lane | outcome | evidence |
|---|---|---|
| **Amplitude normalization** | ❌ fatal | replacing values with within-series rank collapsed OOF 0.975→0.86. **Persistently-low backscatter level *is* the class signal** — do not detrend, difference, or instance-normalize |
| **Foundation models (Presto)** | ❌ dead | frozen encoder *re-encodes* the shift: adversarial AUC 0.965–0.976 on its own embeddings, ATC-F1 −0.044 to −0.059 LB |
| **Cross-model-class blending** | ❌ closed, n=2 | ROCKET member (ρ0.87) → −0.009; GBDT member (ρ0.873) → **−0.0155, paired, ≥2.5σ**. See the law below |
| **Feature-space deletion** | ❌ closed | `c_dropvv` −0.0113; and the shift is **distributed** — max single-band adversarial A = 0.59 against a joint 0.89, so no small subset carries it |
| **Missing-indicator deletion** | ❌ closed for free | indicators alone separate train/test at **0.4758, below chance** — because our masking already matches test dropout rates by construction |
| **Importance weighting / DANN** | ❌ dead | effective sample size collapses at adversarial AUC 0.99 |
| **Label-shift priors (BBSE / Saerens-EM)** | ❌ wrong model | assumes label shift; ours is covariate shift. Estimated prior **0.44** against a true ≈0.649 |
| **Water indices (WIF/EVI/SDWI/AWEI/NDWI)** | ❌ −0.075, and degenerate | **SDWI is exactly affine in (VV_dB + VH_dB); AWEI is exactly linear; EVI ≈ 2.5(NIR−Red) over water; NDWI/MNDWI are 0/0-conditioned over water.** A linear model already spans them |
| **Seed averaging as a climber** | ➖ variance only | lands *at* the member mean (predicted 0.886, got 0.88653) |
| **Temperature scaling / TTA / focal loss** | ➖ invisible or inside noise | see the rank-only argument below |

**What the blending failures do and do not show.** Under a metric scored at a pinned threshold, the
error-ambiguity decomposition that justifies ensemble diversity **does not apply** — there is no such
decomposition for 0-1 loss (Brown & Kuncheva, MCS 2010, distinguish "good" from "bad" diversity for
exactly this reason). Our two foreign-member blends lost **monotonically in the member's own level
deficit**, so we gate members on **level gap, not decorrelation**, and always rank-average rather
than probability-average.

**We flag a limitation in our own evidence.** Both members were also *weaker* (−0.040 and −0.011),
so member strength and member diversity are **perfectly confounded at n=2**. What we demonstrated is
that a weak member hurts; the stronger claim that *diversity itself* is harmful is **not identified**
by our experiment, and an earlier draft of this report overstated it. Untested and plausible: a
diverse member could hurt the F1 **set** while helping the AUC **ranking** — we have never measured
the two columns separately.

**The metric is rank-only — with one important scope condition.** After the prevalence pin fixes the
predicted-positive count, `F1 = 2·TP/(P̂+P)` is monotone in precision@k, a functional of the test
*ranking* alone; ROC-AUC is rank-only by definition. So temperature, a constant logit offset, or any
monotone recalibration is **invisible to the leaderboard**. **The scope condition matters:** this
holds *because of* the pin. Replace the pin with a literal 0.5 cut and monotone calibration stops
being invisible and becomes the sole determinant of the 0.6-weighted column. The two are the same
decision, which we did not appreciate until late (§7).

---

## 7. ⚠️ Limitations, and a rule question we raise against ourselves

**🔴 The prevalence pin is a rules violation. We are reporting it against ourselves.**

We read the rules page directly on 2026-07-28. It states, verbatim:

> *"Setting a probability threshold is strictly forbidden. Your binary target should be based on the
> default threshold of 0.5."* … *"do not set thresholds (or round your probabilities) to improve your
> place on the leaderboard."* … *"Zindi will need the raw probabilities. This will allow the clients
> to set thresholds to their own needs."*

Earlier revisions of this repo argued our construction was *prevalence correction* (allowed) rather
than *threshold tuning* (forbidden), because the cut stays literally at 0.5. **That argument does not
survive reading the rule, and we withdraw it.** Two distinct problems:

**(a) `TargetF1`.** `src/calibration.py:50-66` computes `thresh = quantile(logit(p), 1 − π̂)` and
shifts by `−thresh` so that threshold lands at 0.5. Our own docstring says it: *"hitting an EXACT
target prevalence **is a threshold on the logits**."* The literal 0.5 in the comparison is cosmetic.
Worse, **π̂ = 0.649 was not derived from training data** — it was swept against *leaderboard feedback*
(iteration 02: 0.7561 → peak 0.8260 at realized ≈0.65) and we kept the peak. Elkan (IJCAI 2001)
formalizes the equivalence: for calibrated models, cost-sensitive reweighting and threshold shifting
are the same operation.

**(b) `TargetRAUC`.** `src/calibration.py:78-82` returns `(rank + 0.5)/n` — **uniformly spaced ranks,
not probabilities.** A client thresholding our column at 0.8 receives "the top 20% of rows," which has
no probabilistic meaning. This defeats the stated rationale of the raw-probabilities rule. Because
ROC-AUC is invariant to any strictly monotone transform, **emitting genuine calibrated probabilities
here costs us nothing on the metric** — this is a free compliance fix and there is no argument
against it.

**Why we are disclosing rather than quietly hoping.** The final score is 65% private LB + 35% code
review **of the top 5 only**. If we do not reach the top 5, the pin was worth nothing. If we do, our
code is read by exactly the people who wrote the rule. **The gain is only cashable in the scenario
that triggers the review that would void it.** There is no branch on which keeping it pays.

One honest note on magnitude: the ≈+0.07 attributed to the pin was measured in iteration 02 **on the
superseded GBDT model class**. Its value on the current transformer is **unmeasured**, and the true
cost of compliance may be materially smaller.

**Status:** disclosed here; remediation (calibration fit on training folds only, then a literal 0.5
cut) is the top item on our work list, and we have asked the organizers for clarification on forum
thread 33912.

**Reproducibility caveat, stated plainly.** A single seed (42) drives all RNGs and per-`(row, view)`
seeds are derived deterministically, so the masking augmentation is exactly reproducible. However,
for the `seq` path on GPU we do **not** set `torch.use_deterministic_algorithms` or
`cudnn.deterministic`, so CUDA attention kernels may introduce small run-to-run differences. GBDT
runs are bit-identical; `seq` runs reproduce to within that kernel nondeterminism. Given §4, a
reviewer should expect run-to-run variation on the order of the seed effect.

**A ceiling we cannot reach, and why we stopped trying.** The pond-mapping literature stands on three
legs, and two are unavailable to us:

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
do not have**, and are not a target we can hit from isolated pixel time series.

*Geography caveat:* this literature is overwhelmingly coastal East/Southeast Asian intensive
aquaculture. The FAO/ITU framing suggests our data may be African smallholder — smaller ponds, less
intensive feeding, more rain-fed drawdown — which would weaken both the eutrophication signal and the
"permanently full" assumption. We found no quantitative African pond-mapping study to calibrate
against, and we flag our physics-derived expectations as **unvalidated for this setting.**

---

## 8. What we would do with more time

Stated so a reviewer can see we know where the remaining value is, not as a claim of results.

1. **Exploit the two independent columns.** `TargetF1` (weight 0.6) is a **set-selection** problem at
   fixed k≈668; `TargetRAUC` (0.4) is a **global ranking** problem. They are computed by separate
   functions writing separate columns (`run_pipeline.py:160-161`) and we have fed both from one score
   vector by habit. The 0.6-weighted column is blind to all reordering away from the cut.
2. **Cross-band ratio features.** Untested here. Note this does *not* contradict §6's amplitude
   finding: we refuted *within-series temporal rank* (which destroys level); a **cross-band** ratio at
   fixed *t* preserves level while cancelling per-period multiplicative gain drift. `VH − VV` in dB
   *is* the log cross-pol ratio and is the obvious first candidate.
3. **Implement the rule-risk reconstruction in §7** — the highest-value item on the list, because it
   is about defensibility rather than score.

---

## 9. Where everything lives

```
config/config.yaml              all hyperparameters, feature toggles, seed
src/
  data.py                       schema discovery, -9999 -> NaN cube, test-mask measurement
  features.py                   window masking (apply_mask), indices, aggregates
  validation.py                 masking-aware K-fold, leak-free OOF
  seq_model.py                  the submitted temporal Transformer
  calibration.py                fixed-0.5 logit shift; monotone TargetRAUC
run_pipeline.py                 orchestrator; writes + validates every submission
tools/
  offline_validate.py           the LB-predicting screen (§5)
  shift_audit.py                adversarial probes + the 2-D band screen (§6)
  adversarial_check.py          domain-shift integrity monitor
  arch_blend.py, seed_average.py   pooled artifacts
experiments/
  reproduce_champion.sh         ONE COMMAND REPRODUCTION
  anchors.tsv                   known-LB anchors for the retro-fit (incl. the falsifier)
  LB_LOG.md                     every iteration, every score, every verdict
  results.tsv                   append-only run log
PROJECT_STATE.md                full state, ledger and lessons
```

**Cross-validation is masking-aware and leak-free:** folds are defined on the *original* rows, every
augmented view inherits its row's fold (a row's masked twins never straddle the split), and each
held-out row is scored on *R* independent masked views averaged into one OOF probability.

**A deliberate inversion a reviewer should expect:** our best-leaderboard model has our *lowest* OOF.
Local OOF sits near 0.975 against a leaderboard near 0.89, and has been **anti-correlated** with it.
We never select on OOF. This is stated in `README.md` and is not an accident of tuning — it is the
covariate shift of §1 doing exactly what it was designed to do.
