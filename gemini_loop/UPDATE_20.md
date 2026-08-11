# Deep-Research Brief — Round #20 (Claude Deep Research / Gemini Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)

**Date:** 2026-08-11 · **Best public LB: 0.910837 (legal, 5-seed pooled)** · **Deadline: 2026-08-16** ·
**Finalists to designate: 2** · **Supersedes `UPDATE_19.md`** (self-contained; no need to read it)

---

## 0. What this round is

Round 19 produced **the most useful external input this project has had.** It correctly identified a
real defect in our ensembling code, and we verified it against the source. This brief is the reply:
what we confirmed, what we killed, **where we were wrong**, and — the part we most want attacked —
**three questions round 19 left unexamined that we believe now dominate the outcome.**

Two ground rules, both learned the hard way:

1. **We will check every claim you make against our code and ledger.** Round 19 survived that. A
   parallel report from another system did not: it invented a leaderboard leak, a segmentation data
   modality, and an eight-row experiment history, and was rejected in full. If you are unsure of a
   fact about our setup, say so rather than filling it in.
2. **Correcting us is the highest-value thing you can do.** Two load-bearing beliefs died this week —
   one of ours, one of round 19's — and both deaths were progress. §3 lists our own errors first so
   you can calibrate how much to trust the rest.

---

## 1. Self-contained problem statement

Binary classification: is this ~10 m cell a managed aquaculture pond? Each row is **one isolated
pixel's multivariate time series** — a **12-month × 12-band** cube. Bands: Sentinel-1 SAR **VH, VV**
(dB, always co-present when a month is observed) + 10 Sentinel-2 optical bands (individually missing
under cloud). **No lat/lon, no image patch, no static covariates, no neighbourhood.** It is tabular
time series, not imagery — nothing to segment.

- **Train:** 1,821 rows → **1,817** after exact-duplicate removal. ~40.2% positive. **All 12 months present.**
- **Test:** **1,030** rows, each with only **4–6 CONTIGUOUS visible months** (measured 345/343/342;
  1,030/1,030 contiguous). Sentinel −9999. Public **309** / private **721** (30/70).
- **The designed shift is temporal-window truncation plus a genuine covariate shift on the SAR level.**
  An offline label-shift goodness-of-fit test **FAILED** (KS D=0.186, p≈0) — there is a real
  **conditional** shift component, not pure label shift. Saerens/BBSE prior correction is therefore
  invalid here, not merely non-compliant.
- **Measured test prevalence ≈0.56–0.58** (MLLS 0.578, BBSE 0.559 — two estimators agreeing). Our
  artifacts realize **0.588**. We are already at or slightly above the estimated prior: **there is no
  positive-rate gap to close.** (An earlier brief said "believed ≈0.65". That was stale text from a
  retired prevalence-pin era and it misled round 19 — see §3.2.)

**Metric:** `0.6·F1@0.5 + 0.4·ROC-AUC`. Two columns: `TargetF1` (binary, **hard 0.5 cut**) and
`TargetRAUC` (probability). **Threshold tuning is explicitly forbidden.**

**Our model** (from scratch, ~71k params): a small temporal Transformer over the 12 months. Base input
= 24 channels/month (12 standardized band values + 12 missing-indicators). First layer is a single
`nn.Linear` projection — **this matters: any channel that is an affine function of existing bands is
exactly spanned by it and adds zero representational capacity.** LB-validated structure:

- **relative-time reframing** — observed window left-aligned to t_rel=0 (+0.0128, largest structural win).
- **cross-view invariance** — each training row masked into K=2 synthetic 4–6-month windows drawn from
  the empirical test-window distribution, penalizing logit variance across views (λ=1, +0.0047).
- **permanence channel** — per-month indicator `1[VH_dB(t) < −21]` (+0.010 seed-averaged).
- **transductive soft self-distillation** — a non-distilled 5-seed teacher's soft probabilities (T=1,
  never thresholded) on the 1,030 unlabeled test rows, added to labeled BCE with weight α (+0.0100).
- **legal calibration** — Platt fit on **training OOF only**, then a **literal 0.5 cut**.
- **seed pooling** — currently per-seed Platt → probability average (**this is under review; see §2.1**).

**The masking trap:** because test windows are 4–6 months and train is 12, only **n-invariant**
statistics transfer. Means, medians, interior quantiles and fractions are safe; min/max/range/counts
are window-length biased and are not.

**The gap we are attacking.** Zindi reports F1 and AUC separately, so we can decompose exactly:

| | our best legal artifact | public LB leader | gap | worth in composite |
|---|---|---|---|---|
| **ROC-AUC** | 0.942861 | 0.944897 | +0.002036 | **+0.000814** |
| **F1 @ 0.5** | 0.889488 | ≈0.920235 *(implied)* | +0.030747 | **+0.018448** |
| composite | 0.910837 | 0.930100 | +0.019263 | |

**95.8% of the residual gap is the F1 term.** §4.1 argues this decomposition is being *misread* —
including by us — and that is the most important question in this brief.

**Measurement resolution.** Zindi's F1 column is a small-denominator rational and inverts exactly:
`55/62, 330/371, 82/93 ⇒ TP = 165, 165, 164`. One F1 row-flip ≈ **0.0055 composite**; one AUC
concordant pair ≈ **4.4e-5**. The public slice resolves F1 only in ~0.0055 quanta. Seed-to-seed LB sd
is **0.019**; public binomial noise ±0.012. *This inversion is diagnosis only — the public positive
count must never feed the operating point, which would be leaderboard probing. Do not propose
anything requiring it.*

---

## 2. Round-19 adjudication — verified against source

### 2.1 CONFIRMED: our pooling order is wrong (round 19's central claim)

Round 19 claimed we pool backwards. **We checked. It is correct.** `src/calibration.py:161-171`:

```python
for each member:  o_cal, t_cal, slope = platt_calibrate(y_m, oof_m, test_m)
p_test = np.vstack(cal_test).mean(axis=0)
```

That is a **linear opinion pool of individually-calibrated probabilities** — exactly the construction
Ranjan & Gneiting (2010, *JRSS-B* 72(1):71–91) prove is necessarily uncalibrated and unsharp, and that
Rahaman & Thiery (NeurIPS 2021, arXiv:2007.08792) remedy with Pool-Then-Calibrate. Our code's docstring
argues carefully for per-member calibration *versus rank-averaging* — a sound argument — but never
considered the third option: pool the logits, calibrate once.

We have built `tools/repool.py` to compare the two orders offline on saved member bundles (0
submissions). **It has not yet run on real bundles** — see §5.

### 2.2 KILLED: regime-matched OOF calibration (round 19's Proposal 2)

Round 19 proposed refitting Platt on OOF computed from *masked* training views rather than
fully-observed 12-month rows, and flagged its own load-bearing caveat: *verify the current OOF
observation regime first; if OOF is already masked, this does nothing.* **We verified. It fires.**

- `src/seq_model.py:980-981` predicts each held-out fold through `_mask_views(..., oof=True)`.
- `_mask_views` (line 753) draws every view from `sample_window(wd, ...)` — **the same empirical
  test-window distribution** used in training.
- `oof=True` changes **only the RNG tag** (line 751: `tag = (10000 + k) if oof else k`).

**Our OOF is already regime-matched.** A monotone refit on the same score distribution cannot move the
positive count. Dead on the facts. *This is a model for how to write a proposal: the caveat that killed
it was round 19's own, stated up front. We would rather receive three proposals with honest kill
conditions than five without.*

### 2.3 DECLINED on compliance: the Lipton F/2 score shift

Round 19 correctly cites Lipton, Elkan & Narayanaswamy (arXiv:1402.1892): for calibrated probabilities
the F1-optimal threshold is **F/2**, so at F1≈0.92 the optimum is ≈0.46 and a forced 0.5 cut is
F1-suboptimal *by construction*. It then proposes shifting the score distribution upward so 0.5
"behaves like 0.46."

**We decline this, and the reasoning generalizes.** With the calibrator already fit under the correct
observation regime (§2.2) and already landing at the measured prior (§1), an upward shift would not be
correcting a mis-specified `p(y|x)` — it would be an operating-point move chosen for its effect on the
cut. That is threshold tuning in a calibration costume. We are recording it here because it is the most
seductive argument in the report and will recur: **a legal-looking mechanism does not make a
pos-rate-motivated application of it legal.**

But the *theory* is not dismissed — it reappears, sharpened, as **Q1** in §4.1.

### 2.4 Accepted as correctly closed

Multi-round self-distillation (Mobahi et al. arXiv:2002.05715 predict under-fitting; Kumar et al.
arXiv:2002.11361 show per-step error compounding at our shift magnitude); the "leader used plain
CatBoost" hypothesis (our own three tree attempts: 0.6976 naive, blend −0.0136, 0.7186 shift-robust);
and "0.95 is not a legal target" (the honest ambition is ~0.919–0.926).

---

## 3. Our own errors this week — read these before trusting §4

### 3.1 We mis-triaged round 19's best proposal, on two bad grounds

We initially **downgraded** the pooling fix, arguing (a) the defect bites only when member Platt slopes
diverge, and (b) our ledger's +0.0055..+0.0061 pooling gains prove the combiner is fine. **Both wrong.**

- **(a) is impossible.** Per-member Platt is **scale-invariant** — fitting a logistic on each member's
  own logits removes that member's slope by construction, so the combiner cannot see slope
  heterogeneity at all. Synthetic check: 5 members at slopes {1,1,1,1,1} vs {0.35,0.6,1.0,1.7,2.6} give
  **bit-identical** pooled OOF F1 0.62529 and pos-rate 0.2033.
- **(b) was a category error.** Those gains measured **pool vs single member**, never **combiner A vs
  combiner B**. Pooling can be worth +0.006 and still leave more on the table through the wrong order.
  Nothing in our ledger has ever compared the two.

**The correct mechanism is generic:** an arithmetic mean of independently-noisy probabilities is shrunk
toward the members' centre of mass, so the pooled distribution is narrower than any member's and a
fixed 0.5 cut catches fewer rows. Needs only independent member noise. In the same synthetic, logit-
averaging + single Platt beat the current combiner by +0.039 F1 (homogeneous) and +0.031 (divergent).

**But see Q2 in §4.2 — we now doubt this transfers to our actual members, for a reason neither we nor
round 19 raised.**

### 3.2 We fed round 19 a false premise

`UPDATE_19.md` stated "believed true test prevalence ≈0.65." Our own label-shift gate had measured
0.578/0.559 four iterations earlier. Round 19's Proposal 2 was built to close a pos-rate gap that does
not exist. Corrected in §1. **If any part of this brief contradicts something you can derive from the
numbers we give you, trust the numbers and tell us.**

### 3.3 A standing pattern worth knowing

**Four separate single-seed "records" (0.906492, 0.913263, 0.912759, 0.914179) all collapsed to ~0.8995
when seed-averaged.** Our local OOF (~0.975) is *anti-correlated* with LB (~0.90). We never select on
OOF and never trust an unpooled result. Any proposal evaluated on a single seed is not evaluated.

---

## 4. What round 19 did not capture — the three questions we most want attacked

### 4.1 Q1 (HIGHEST PRIORITY) — Is "95.8% of the gap is F1" actually telling us to fix calibration?

We have been reading the decomposition as: *AUC is at parity, therefore our ranking is as good as the
leader's, therefore the whole gap is cut placement.* **We now think that inference is invalid, and
round 19 accepted it without challenge.**

AUC is a **global average over all thresholds**. Two models with identical AUC can have very different
ROC shapes, and F1@0.5 depends only on the **single operating point** where a calibrated 0.5 lands.
The leader could have an ROC curve that is *worse than ours in regions we never use* and *better than
ours precisely in the near-cut band* — identical AUC, better F1, and **no amount of recalibration would
close it**, because the deficit is local ranking, not cut placement.

This reframes everything, so we want it settled:

- **(a)** Given AUC ≈ 0.944, prevalence ≈ 0.57, and n = 1,030, what is the **achievable range of
  F1@0.5**? Where do 0.889 (ours) and ≈0.920 (leader) sit in that range? Is +0.031 F1 even attainable
  at our AUC by cut placement alone, or does it *require* a better local ROC?
- **(b)** Is there a diagnostic that separates "our cut is mis-placed" from "our ranking is worse near
  the cut" — computable from our own OOF and test scores, **without any leaderboard-inverted quantity**?
- **(c)** If it is local ranking: the right lever is **partial-AUC / near-cut ranking optimization**, not
  calibration at all. Round 19 cited Zhu et al. (arXiv:2203.00176, partial-AUC/DRO) **in its
  bibliography but used it in no proposal.** What does the pAUC literature actually support for
  concentrating discriminative power in a target FPR/TPR band, and does it transfer under covariate
  shift? Also relevant: Ye, Chai, Lee & Chieu (ICML 2012, arXiv:1206.4625) on F-measure optimization.

**This is the question we would most like answered.** If the answer is "the gap is local ranking," then
the entire calibration lane — including our own §2.1 fix — is a sideshow, and we would rather know that
before spending our last experimental slot.

### 4.2 Q2 — Does the linear-pool defect survive at ρ = 0.98?

Round 19 asserted the pooling defect and we confirmed the code, but **neither of us checked the
magnitude against our actual member correlation.**

Our pooled members have **inter-seed rank correlation ρ ≈ 0.9804** (measured, iter43 ARM E, 10 members;
similar across all pools: 0.9779, 0.9802, 0.9863). The underconfidence effect is driven by averaging
away **independent** member noise. With m members at correlation ρ, the variance of the average is
`(1 + (m−1)ρ)/m` of a member's — at m=5, ρ=0.98 that is **0.984**, i.e. the pooled distribution is
compressed by ~1.6%. Our synthetic in §3.1 used weakly-correlated members and is therefore not
informative about our case. **See the per-pool slope / pos-rate table in §5: across four independent
pools the pooled positive rate lands at or slightly above the member mean, not below it — which is the
opposite of what a biting compression defect would produce.**

- **(a)** Is the Ranjan–Gneiting / Rahaman–Thiery underconfidence result **quantitatively negligible at
  ρ ≈ 0.98**, or does it bite through some channel that correlation does not capture? Does any paper
  report the effect size *as a function of ensemble diversity*?
- **(b)** If negligible for near-duplicate seeds, does it become material for a **heterogeneous**
  ensemble — e.g. our `archblend4` (4 different architectures, mean pairwise ρ 0.9524) or a
  distill+consistency mix? Should we be pooling *more diverse* members specifically so that the
  corrected combiner has something to work with?
- **(c)** Our ARM T pool is the one case that failed badly. We hypothesise its members were individually
  **underdispersed** (a cross-view variance penalty compresses logits toward a constant — its known
  attractor), and that averaging already-underdispersed members compresses hardest. Is underdispersion,
  rather than diversity, the right predictor of when linear pooling breaks under a hard threshold?

### 4.3 Q3 — Are the distillation and consistency terms antagonistic?

Round 19's Proposal 3 says "do not choose between self-distillation and label-free consistency; combine
them," with distillation anchoring AUC and consistency sharpening F1. **Its supporting arithmetic
assumes you can hold AUC at 0.944 *and* capture the consistency arm's F1 — but our data says the
consistency arm buys F1 by *spending* AUC.**

| artifact | F1 | AUC | composite |
|---|---|---|---|
| `tcons_s42` (single seed, consistency) | **0.901333** | 0.933447 | 0.914179 |
| `tcons_s13` (single seed, consistency) | **0.897507** | 0.925923 | 0.908873 |
| `tcons` 5-seed pool | 0.871795 | 0.926687 | 0.893752 |
| best distill pool (α=1.5) | 0.889488 | **0.942861** | **0.910837** |

Those two single seeds hold **our two highest F1 values ever**, ~0.012 above anything the distillation
lane has produced. But tcons_s42 also gives up **0.0095 AUC** versus the distill pool. At its own AUC,
0.901333 F1 is worth 0.914179 — only +0.0033 over what we already have, and inside seed noise.

- **(a)** Is the F1↑/AUC↓ pattern a **known, named trade** in the consistency-regularization /
  entropy-minimization literature (Grandvalet & Bengio NIPS 2004; Chapelle & Zien AISTATS 2005; VAT;
  FixMatch)? Is it *intrinsic* — local margin sharpening necessarily costing global ordering — or an
  artifact of over-weighting λ_u?
- **(b)** If intrinsic, do the two terms **compose or cancel**? Adding a distillation term that pulls
  toward the teacher's ordering may simply undo the boundary sharpening that produced the F1 gain. Is
  there published evidence of a scheme where both are retained (e.g. consistency on unlabeled data +
  distillation on a *disjoint* term, or a schedule that anneals one against the other)?
- **(c)** Is there a variant of the consistency penalty that sharpens the **local margin without
  compressing the global logit scale** — i.e. gets the F1 gain without the AUC cost and without the
  underdispersion that we think broke its pool (§4.2c)?

---

## 5. What is running / pending right now

**Iteration 43 — three artifacts uploaded, LB scores not yet returned.** All 25 runs completed and
passed the transductive sanity gate.

- **ARM E** `champion_distill_alphamix10` — α-marginalized pool, 10 members at 10 distinct seeds
  (α=0.7 ×5 + α=1.5 ×5). Inter-seed ρ 0.9804, pos-rate 0.5883.
- **ARM F** `champion_dualpol_rep_seedavg5` — dual-pol gate `1[VH<−21]·1[(VH−VV)<−8]` **replacing**
  permanence (25 ch). ρ 0.9779, pos-rate 0.5874.
- **ARM G** `champion_dualpol_add_seedavg5` — same gate **added** to permanence (26 ch). ρ 0.9802,
  pos-rate 0.5874.

Train-only, window-matched univariate AUC of the mean-pooled gate fraction: VH-only 0.8012, ratio-only
0.7556, **AND gate 0.8487** — the AND beats both clauses, so it is a genuine interaction.

**Width audit passed on both dual-pol arms:** ARM F logged 25 input channels and ARM G logged 26, so the
"replacement" arm really replaced and the "addition" arm really added. Neither arm is void.

**Full per-pool combiner diagnostics from the iteration-43 log — direct evidence for Q2 (§4.2).** These
are the per-member Platt slopes and positive rates that combiner A (per-member Platt, then arithmetic
mean) produced, alongside the inter-member rank correlation:

| pool | m | mean ρ | min ρ | per-member Platt slopes | member pos-rates | pooled pos-rate |
|---|---|---|---|---|---|---|
| `teacher_perm5` | 5 | 0.9478 | 0.9371 | 1.520 / 1.547 / 1.459 / 1.526 / 1.520 | 0.576–0.598 | 0.5816 |
| `amix` (ARM E) | 10 | 0.9804 | 0.9714 | 1.350–1.468 (all ten) | 0.572–0.603 | 0.5883 |
| `dpr` (ARM F) | 5 | 0.9779 | 0.9704 | 1.384 / 1.419 / 1.449 / 1.476 / 1.484 | 0.572–0.596 | 0.5874 |
| `dpa` (ARM G) | 5 | 0.9802 | 0.9766 | 1.397 / 1.397 / 1.465 / 1.475 / 1.501 | 0.570–0.594 | 0.5874 |

Two things to note, both of which sharpen Q2 rather than answering it:

1. **The members are near-homogeneous in every respect that matters.** Slopes cluster in 1.35–1.55 (a
   ±6% spread) and member pos-rates in 0.570–0.603. This is the opposite of the divergent-slope
   synthetic we (wrongly) built our first triage around — and, per §3.1, per-member Platt is
   scale-invariant anyway, so slope spread is not the channel.
2. **The pooled pos-rate is not below the member pos-rates.** In every pool the pooled rate sits at or
   just above the member mean (e.g. ARM E: members average ≈0.587, pooled 0.5883). If the linear-pool
   defect were compressing the pooled distribution enough to starve a fixed 0.5 cut, we would expect the
   pooled rate to fall *below* the member mean. It does not. **This is a real, if indirect, argument
   that the defect may be quantitatively negligible for us — and it is why we are not treating §2.1 as
   a confirmed win.** The one pool that did collapse (ARM T, `tcons`) is also the only one with a
   variance-penalty term, consistent with the underdispersion hypothesis of §4.2(c) rather than with a
   generic combiner defect. We want this adjudicated.

**`tools/repool.py` has not yet run on real bundles.** Colab's working directory is ephemeral and our
notebook copies data *in* from Drive but nothing back out, so iteration 41's ARM T member `.npz` files
are lost; the comparison requires re-running those 5 seeds (compute, **zero submissions**). Treat the
combiner question as **open and unmeasured**, not as a confirmed win.

*If iteration 43's scores have arrived by the time you read this, they will be appended here.*

---

## 6. Compliance constraints that bind any proposal

Non-negotiable. Final standing is **65% private leaderboard + 35% code review of the top 5**, so a
rules-violating artifact is worse than a weak one.

1. **Threshold tuning is FORBIDDEN.** The binary column must be the literal `p ≥ 0.5` cut of the
   probability column. We run `compliance_mode: legal` — Platt on **training OOF only**, then 0.5.
2. **No leaderboard-inverted quantity may reach the operating point.** Diagnosis only.
3. **Only supplied competition data.** No external DATA. Pretrained model **weights** are legal
   (verified); Presto was tested frozen-only and fine-tuning remains untested.
4. **No AutoML.**
5. **Seeded and reproducible.** We do not set GPU-deterministic algorithms; measured seed-to-seed LB
   spread is 0.019 and we state that openly.
6. **The three data CSVs are private** and are never committed.

**Our working boundary,** refined from round 19's three-prong test, which we found genuinely useful:
a change is legal iff **(a)** the decision rule stays a literal 0.5; **(b)** every knob is fixed by a
train-only criterion, never against a realized pos-rate target or LB feedback; **and (c)** it corrects
`p(y|x)` under a *demonstrably* mis-specified model rather than relabeling a fixed estimate. Prong (c)
is what killed §2.3: once the model is correctly specified, there is no mis-specification left to
"correct," and further movement is cosmetic relabeling. **Tell us if you think this test is wrong.**

---

## 7. Closed lanes — do not re-propose without new evidence

Each cost real submissions. Argue to re-open only with a specific paper and a specific reason our test
was under-powered.

- **Tree models / CatBoost.** Three fails: 0.6976 naive → blend −0.0136 → 0.7186 shift-robust. The
  "shift-robust" adversarial GO gate was a **false positive** — a test-like *covariate* holdout still
  carries train labels, so it is blind to **conditional** shift.
- **Adversarial AUC as a selection criterion — RETIRED.** Correlates **positively** with realized
  transfer here (Spearman +0.68 across transforms, +1.00 across modalities), i.e. backwards.
- **Marginal feature alignment.** Erases the measured shift (adv-AUC 0.9291→0.3608) and buys nothing.
- **Saerens / BBSE / EM prior correction.** Gate failed — conditional shift present.
- **Regime-matched OOF calibration.** Dead: our OOF is already masked from the test-window
  distribution (§2.2).
- **Any pos-rate-motivated score shift, including the Lipton F/2 construction** (§2.3).
- **Capacity-ADDING feature channels, generally.** Repeated pattern: OOF rises, LB falls. Replacements
  have fared better than additions.
- **All affine functions of existing bands.** Our first layer is a single `nn.Linear`, so affine
  channels are exactly spanned and add zero capacity. Measured: raw `VH−VV` cost **−0.0228**. This also
  closes **AWEI** (`4(G−SWIR1) − (0.25·NIR + 2.75·SWIR2)`) and any other linear index, without a run.
  Non-affine indices (NDWI, MNDWI, NDVI, SDWI — ratios and logs) are already always-on channels.
- **Off-manifold TTA / hole-punched masking.** Diagnosed failure.
- **Multi-round self-distillation.** One round only (§2.4).
- **Single-seed "records."** Four have collapsed to ~0.8995 (§3.3).

---

## 8. Execution budget

- **Compute:** Colab GPU; a 25-run iteration (5 folds × ~60 epochs) takes ~25 minutes wall-clock.
- **Submissions:** 5/day, ~20 usable before the deadline. **Not the binding constraint.**
- **Binding constraint:** *ideas that survive contact with a 0.019-sd seed distribution.* Anything worth
  running must be expected to move **≥0.006 seed-averaged**, or be a variance reduction justifiable
  without a level read.
- **Slots:** iteration 44 is the **last experimental slot**; iteration 45 is finalist consolidation plus
  the code-review package (35% of final score). So we need **at most two or three** concrete, ranked,
  implementable interventions. Depth over breadth.

---

## 9. Deliverable format

For each proposed intervention:

1. **Mechanism** — what it changes, and why that moves F1 at a fixed 0.5 cut *specifically*.
2. **Evidence** — actual papers, what they measured, on what kind of shift. Read them; do not summarize
   abstracts. **Say plainly when evidence is thin** — round 19 did this well and it made the report more
   useful, not less.
3. **Kill condition, stated up front** — the check that would make this proposal moot, and how we run it
   *before* spending the slot. Round 19's Proposal 2 came with its own kill condition and we were able to
   settle it in ten minutes. That is the format we want.
4. **Implementation sketch** against §1 (temporal Transformer, K=2 masked views, train-only Platt, seed
   pooling).
5. **Expected effect size** in composite points, with an honest interval.
6. **Falsification** — what result proves it did not work, distinguishable from a ±0.019 seed draw. Note
   we get **one** public score per uploaded artifact; "average over 5 seeds" means *pool 5 seeds into one
   artifact*, not five uploads.
7. **Compliance check** against §6, explicitly.

**Please prioritize §4.1 (Q1).** If Q1's answer is that the gap is local ranking rather than cut
placement, say so plainly even though it invalidates most of round 19's framing *and* our own — that
would be the single most valuable finding of this round.

If your honest conclusion is that we are at the achievable ceiling under these constraints, **say
that.** With five days left and 35% of the score on code review, a well-argued "stop and consolidate"
is a valuable answer.
