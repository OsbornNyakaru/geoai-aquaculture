# Deep-Research Brief — Round #21 (Claude Deep Research / Gemini Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)

**Date:** 2026-08-13 · **Best public LB: 0.910837 (legal, 5-seed pooled)** · **Deadline: 2026-08-16
(3 days)** · **Finalists to designate: 2** · **Supersedes `UPDATE_20.md`** (self-contained; no need
to read it)

---

## 0. What this round is

Since round 20 the picture has **inverted on its central premise.** Round 20 was written when our
ROC-AUC trailed the public leader's. It no longer does: our best-ranking artifact posts **AUC
0.946460 against the leader's 0.944897.** We are ahead on ranking and still behind on the composite,
and the F1 column inverts exactly enough to say why — **the leader converts ≈173 true positives to
our ≈164 at an equal-or-better ranking.** The entire residual gap is roughly **nine rows at the
decision boundary.**

That reframes the question we need attacked. It is no longer "how do we rank better." It is: **given
a ranking that already beats the leader, is there any legal mechanism that converts ranking quality
into boundary accuracy at a fixed 0.5 cut?**

This brief also **cancels the experiment round 20 recommended** (sigmoidF1) and explains the general
theorem that killed it, because that theorem invalidates a whole *class* of proposals and you should
know it before proposing anything in that class. And it reports **new measurements from a graph-based
estimator** that bear directly on the one prevalence question round 20 declared undecidable.

Two ground rules, retained verbatim from round 20 because they worked:

1. **We will check every claim you make against our code and ledger.** Round 19 survived that; round
   20 partly did not (§2). A parallel report from another system invented a leaderboard leak, a
   segmentation data modality, and an eight-row experiment history, and was rejected in full. **If
   you are unsure of a fact about our setup, say so rather than filling it in.**
2. **Correcting us is the highest-value thing you can do.** Three of our own load-bearing beliefs
   died this week, all three self-inflicted, all three listed in §3.

---

## 1. Self-contained problem statement

Binary classification: is this ~10 m cell a managed aquaculture pond? Each row is **one isolated
pixel's multivariate time series** — a **12-month × 12-band** cube. Bands: Sentinel-1 SAR **VH, VV**
(dB, always co-present when a month is observed) + 10 Sentinel-2 optical bands (individually missing
under cloud). **No lat/lon, no image patch, no static covariates, no neighbourhood.** Verified this
round directly against the CSVs: `Train.csv` has exactly 146 columns — `ID`, `label`, and 12 bands ×
12 months — and no spatial column of any kind. IDs are random Crockford base32 (alphabet excludes
`I`,`O`,`0`,`1`), non-sequential, with zero train/test overlap; they encode nothing recoverable. **It
is tabular time series, not imagery — there is nothing to segment and no geography to exploit.**

- **Train:** 1,821 rows → **1,817** after exact-duplicate removal. **40.23%** positive. All 12 months.
- **Test:** **1,030** rows, each with only **4–6 CONTIGUOUS visible months** (measured 345/343/342;
  1,030/1,030 contiguous). Sentinel −9999. Public **309** / private **721** (30/70).
- **The designed shift is temporal-window truncation plus a genuine covariate shift on the SAR
  level.** An offline label-shift goodness-of-fit test **FAILED** (KS D = 0.186, p ≈ 0) — there is a
  real **conditional** shift component, not pure label shift. Adversarial AUC between masked-train
  and test is ≈**0.99**.
- **Test prevalence: see §4.2.** Our artifacts realize **0.587**. The two estimators that used to
  answer this (MLLS 0.578, BBSE 0.559) were **retired** at iteration 41 by the failed gate above and
  may no longer be cited — an estimator retired for correction cannot be re-used as evidence that no
  correction is needed. A **third, structurally different estimator** is new this round.

**Metric:** `0.6·F1@0.5 + 0.4·ROC-AUC`. Two independently scored columns: `TargetF1` (binary, **hard
0.5 cut**) and `TargetRAUC` (probability). **Threshold tuning is explicitly forbidden.**

**Our model** (from scratch, ~71k params): a small temporal Transformer over the 12 months. Base
input = 24 channels/month (12 standardized band values + 12 missing-indicators). The first layer is a
single `nn.Linear` projection — **this matters: any channel that is an affine function of existing
bands is exactly spanned by it and adds zero representational capacity.** LB-validated structure:

- **relative-time reframing** — observed window left-aligned to t_rel=0 (+0.0128, largest structural win).
- **cross-view invariance** — each training row masked into K=2 synthetic 4–6-month windows drawn
  from the empirical test-window distribution, penalizing logit variance across views (λ=1, +0.0047).
- **permanence channel** — per-month indicator `1[VH_dB(t) < −21]` (+0.010 seed-averaged).
- **transductive soft self-distillation** — a non-distilled 5-seed teacher's soft probabilities
  (T=1, never thresholded) on the 1,030 unlabeled test rows, added to labeled BCE with weight α
  (**+0.0100 — the largest single win in the project, and the only thing that ever broke the ceiling**).
- **legal calibration** — Platt fit on **training OOF only**, then a **literal 0.5 cut**.
- **seed pooling** — per-member Platt, then probability average, then the 0.5 cut.

**The masking trap:** because test windows are 4–6 months and train is 12, only **n-invariant**
statistics transfer. Means, medians, interior quantiles and fractions are safe; min/max/range/counts
are window-length biased and are not.

### The gap, decomposed exactly

| | our best-ranking artifact | our best composite | public LB leader |
|---|---|---|---|
| **ROC-AUC** | **0.946460** ← *above the leader* | 0.942861 | 0.944897 |
| **F1 @ 0.5** | 0.881720 | 0.889488 | ≈0.920235 *(implied)* |
| composite | 0.907616 | **0.910837** | ≈0.930100 |

**The AUC premise has flipped since round 20.** `champion_dualpol_add_seedavg5` posts AUC 0.946460,
**+0.001563 over the leader ≈ 36 concordant pairs.** AUC has been the stable term throughout (the
entire α ladder moved it 0.0017 total; two artifacts differing only in seed count were *bit-identical*
in AUC at 0.944024425), so +0.0024 over our previous maximum is well outside its observed spread.

**Measurement resolution.** Zindi's F1 column is the small-denominator rational `2·TP/(P̂+P)` and
inverts exactly, matching to 10 decimals: iterations 42–43 give `55/62, 330/371, 82/93, 328/372,
326/371` ⇒ **TP ∈ {163,164,165}** across six submissions. One F1 row-flip ≈ **0.0055 composite**; one
AUC concordant pair ≈ 4.4e-5. Seed-to-seed LB sd is **0.019**; public binomial noise ±0.012; a
**paired** delta between two ρ≈0.9 variants of our own model has SE ≈ **0.006**.

> ⚠️ **The AUC column does NOT invert.** See §3.1 — we claimed it did and were wrong. All inversion
> results below come from the F1 column only, and are **diagnosis only**: the public positive count
> must never feed the operating point. **Do not propose anything that requires it.**

---

## 2. Round-20 adjudication — verified against source

### 2.1 KILLED: the headline "~40% cut placement / ~60% local ranking, unclosable"

Round 20's central quantitative claim opens from a stated public-slice composition that **we supplied
and have since retracted** (§3.1). With that premise withdrawn, the split does not stand.

It is also **internally inconsistent**: the brief concedes that the leader's operating point is
reachable under an *unequal-variance* binormal ROC, then reverts to the *equal-variance* ceiling to
compute the 40/60 split. We ran the sweep. Under the unequal-variance model `TPR = Φ(a + b·Φ⁻¹(FPR))`,
the leader's F1 requires `b ≥ 1.75`, and at that same `b` **our** ceiling (0.9218) **exceeds** their
realized F1 (0.9202) — at every `b` our ceiling exceeds theirs by ~0.0012. The "60% is local ranking
and unclosable" conclusion does not survive its own model.

### 2.2 CANCELLED: sigmoidF1 — and the general theorem you need before proposing its siblings

Round 20's concrete recommendation was a sigmoidF1 surrogate loss. We cancelled it **before running
it**, on three independent grounds. The first generalizes and is the most useful thing in this brief:

> **PLATT ANNIHILATION.** If a loss change induces any *affine* reparameterization of the logits,
> `z' = αz + β`, then `σ(a(αz+β) + b) = σ((aα)z + (aβ + b))`. A train-refit Platt scaling — two free
> parameters on the logit — **recovers the identical function, exactly.** Our pipeline refits Platt
> on the very next line after training.

So **sigmoidF1's entire boundary effect, logit-adjusted loss, balanced softmax, and every additive
logit prior correction lie precisely in Platt's span and are annihilated by our own next step.** Any
such arm returns a null for a plumbing reason that says nothing about the method. **Do not propose
anything in this class** unless it changes the logit distribution *non-affinely*.

The corollary is a useful filter: **"survives Platt" and "changes `b` (the binormal variance ratio)"
are the same requirement**, because `b` is a property of the ranking and Platt is affine on the
logit. Of the candidate losses we surveyed, only **VS-loss's multiplicative term** (Kini et al.,
arXiv:2103.01550, Thm 1 — only the multiplicative Δ changes terminal direction) has any theoretical
route to changing `b`. **We found no paper that measures `b` for any loss.** If you know of one, that
is a high-value citation.

The other two grounds: **(ii)** we found no published evidence that any F-surrogate beats BCE at a
*pre-specified* fixed 0.5 with every hyperparameter fixed a priori — sigmoidF1's own fixed-0.5 result
is defeated on its own terms, because its `η` is a logit offset and the grid search over `η` **is** a
threshold search. **(iii)** Measured on our own artifacts, only **29–38 of 1030** rows lie in
[0.45, 0.55]; the threshold-equivalent move needed to reach the F1 optimum is **0.21–0.33**, and
sigmoidF1 blended at w=0.5 supplies ~0.006 — roughly **35× short**, crossing 0.3–0.6 public-slice rows
against a bar that needs ~1.9.

### 2.3 Its kill condition contradicts its own principle

Round 20 proposed aborting on `t* = F*/2` (Lipton et al., arXiv:1402.1892). But `F*` is a **level**
statistic, and the same brief's §Q1(b)-3 states the governing principle "trust shape, distrust level."
Our OOF `F* ≈ 0.975` at prevalence 0.4023 gives `t* ≈ 0.48`, so the kill **fires spuriously**. We
retained the *idea* and rebuilt it correctly against a masked replica (§5).

### 2.4 Accepted as correctly closed, and one thing round 20 got right

Its "stop and consolidate" advice was sound and we are largely taking it. Its observation that the
two submission columns are scored independently is correct and we have exploited it.

---

## 3. Our own errors this round — read these before trusting anything above

### 3.1 We told you the AUC column inverts. It does not.

We previously asserted that the AUC quantum `1/(P·N)` pins the public-slice positive count exactly.
**Withdrawn.** An integrality sweep over every split of 309 (plus 308/310, plus the tie half-quantum
`1/(2PN)`) returns a best max-residual of **0.070**, where 9-decimal reporting would permit ~1e-5.
Every split is rejected. `P` is **not derivable**, and the quantities we built on it — `P̂`, FP, FN,
FPR, precision, recall — **were never established.** The honest figure is **P = 190 ± 7, an
ESTIMATE** from our own logged full-test positive rate (0.5874 ⇒ E[public P̂] = 181.5, hypergeometric
sd 7.2), bounded `P ∈ [164, 208]`. What survives is only what the F1 column gives directly: the `TP`
counts, and the ~9-true-positive gap — both `P`-independent. **Round 20's headline rested on the
retracted version.** We fed you a false premise; that is on us, and it is the second round running
that we have done so.

### 3.2 We proposed "refit Platt on a masked train replica." It was already implemented.

We spent a round arguing for a masked-replica calibration set. Reading our own code:
`run_seq_cv` has always computed OOF through `_mask_views(..., oof=True)`, so held-out rows are
**already** masked to contiguous 4–6 month windows drawn from the measured test-window distribution.
The proposal was a **no-op**, and the `F*/2` statistic it was meant to enable has been computable all
along. *Lesson we are applying: check the code before theorizing about the pipeline.*

### 3.3 We claimed our new calibration arm is exactly AUC-neutral. It is not.

The arm (§5) changes only the OOF vector Platt is fitted on, so no *individual member's* ranking can
change — Platt is strictly monotone. We asserted the pooled artifact was therefore AUC-neutral too.
**Wrong.** Our pooling averages *per-member-calibrated probabilities*, so changing each member's slope
reshapes what is being averaged and the pooled ranking can shift even though no member reordered.
Measured pooled Spearman vs control: **0.99999684**. Near-neutral, not neutral. Our own test caught it.

### 3.4 A standing pattern

**Four single-seed "records" (0.906492, 0.913263, 0.912759, 0.914179) have all collapsed to
~0.8995–0.906 when seed-averaged.** Any proposal justified by a single-seed result is not yet a result.

---

## 4. The questions we most want attacked

### 4.1 Q1 (HIGHEST PRIORITY) — We now out-rank the leader. Can ranking be converted into boundary accuracy?

The premise has flipped. **Our AUC 0.946460 > leader 0.944897**, yet their F1 is ≈0.920 to our ≈0.882.
At an equal-or-better ordering they convert **≈9 more true positives** on 309 public rows.

Under a *fixed* 0.5 cut with *train-only* calibration, what legally remains?

- Both artifacts are monotone transforms of a ranking. If ours is at least as good, then either (a)
  their calibration map places 0.5 at a better point on **our** ROC than our train-only Platt does —
  which, if we cannot reach it by any train-only criterion, is a **structural** disadvantage of legal
  play rather than a modelling gap; or (b) their ranking is better *locally near the cut* while worse
  globally, so AUC is the wrong summary. **These have different remedies and we cannot distinguish
  them from the outside. Which is it, and what evidence would separate them?**
- Is there literature on **local ROC optimization** — maximizing partial AUC or ordering quality in a
  neighbourhood of one operating point rather than globally? Our metric is 0.4·global-AUC +
  0.6·F1-at-one-point, which is exactly a partial-AUC-shaped objective, yet we have optimized global
  ranking throughout.
- The honest possibility we want tested: **the ~9 rows may be irreducible.** If the answer is that a
  legal player cannot close a cut-placement gap against one who tunes, say so plainly.

### 4.2 Q2 — A third prevalence estimator disagrees with "undecidable." Is it sound?

Round 20 argued the prevalence lane is **undecidable** because every label-shift quantifier shares one
bias family, all invalidated by the same failed KS gate. We accept that for MLLS and BBSE. But we
built and ran an estimator this round that **does not belong to that family** — it assumes no label
shift at all, only feature adjacency plus train labels.

**k-NN label propagation on a similarity graph over rows** (`tools/graph_gate.py`, committed,
zero-submission, seed 42, deterministic). Train rows masked to test-like windows through the same
`_mask_views` replica; features are per-band means over *observed* months (n-invariant by
construction, so the graph measures signal rather than window length):

| block | measurement | value |
|---|---|---|
| connectivity | test rows' k=10 neighbours that are labelled | **24.5%** (random mixing 63.8%) → ~2.5 labelled neighbours per test row; domain-clustered, **not disconnected** |
| homophily | train–train edge label agreement | **0.9252** (chance 0.5191, lift **+0.4061**) |
| propagation | masked replica, parameter-free, k=10 | AUC **0.9724** / F1@0.5 **0.9276** / combined **0.9456** |
| **prevalence** | **implied test positive rate, k = 5/10/25/50** | **0.5990 / 0.5913 / 0.5961 / 0.5913** — spread **0.0078**, flat in k |
| control | same, train left **unmasked** at 12 months | bridging 22.5%, homophily 0.9518, combined 0.9684, **prevalence 0.5291** |

Two findings we want examined:

**(a) It agrees with where we already sit.** Our realized rate is 0.587; the graph says 0.591 (+0.007).
An estimator with a completely different bias structure independently lands on our operating point.
**Is this real evidence that the cut is correctly placed, or is the agreement circular** — i.e. does
k-NN in a feature space where our model is also accurate necessarily reproduce our model's prevalence,
making this no more informative than our own output?

**(b) Regime-matching moves the estimate by +0.062**, and in the direction of agreement. Comparing
12-month train rows to 4–6-month test rows gives 0.5291; matching the masking gives 0.5913. **The
naive comparison would have told us our positive rate is far too high.** We think this is a clean
demonstration that regime mismatch alone can manufacture an apparent prevalence gap. Is that reading
right, and does it generalize to how MLLS/BBSE were applied here?

**The caveat we are aware of, and want you to price:** the replica reproduces the *window masking* but
**not the temporal domain shift**, which is the larger half of the problem. So every number above is
**optimistic**. That asymmetry is why we treat "the graph agrees our cut is roughly right" as the
robust reading and "the graph says move it" as the fragile one. **Is there a principled way to correct
a prevalence estimate for the temporal component, or is the estimate simply an upper bound?**

### 4.3 Q3 — Can a *model-independent* teacher escape the one-round distillation cap?

Iteration 41 established the single most important fact in this project: **the ceiling was bias under
covariate shift, and the only lever that moved it was test-distribution information** (soft
self-distillation, **+0.0100**). But the lane is capped at **one round**, because the teacher is the
model's own 5-seed pool and re-teaching from a distilled student compounds error (Kumar/Ma/Liang).

The graph above suggests a way around the cap: a **teacher that is not the model.** k-NN label
propagation onto the test rows is parameter-free, uses only train labels and adjacency, and is
**structurally independent of our network's own predictions** — so its errors are not our errors, and
the compounding argument does not obviously apply. It needs no new model code: our `seq.distill`
already accepts a teacher as an npz of test-row probabilities.

We want this attacked rather than endorsed, because we have already measured one thing against it:
**the graph teacher is more saturated than our own model** — only **0.5–2.9%** of its probabilities
lie in [0.45, 0.55]. Soft distillation from a near-binary teacher is close to **hard pseudo-labelling
on a shifted test set**, which is the classic self-training failure mode. And a second historical
strike: as a *pool member* a weaker model has lost twice here (ROCKET −0.009, GBDT −0.0155), so it
could only enter as a teacher, never as a blend member.

- Is there literature on **co-training / multi-view distillation under covariate shift** where the
  second view is a non-parametric graph rather than a second network?
- Does teacher **sharpness** have a known effect on distillation gain? Is there a principled way to
  temper a near-binary teacher that is not just an inverse-temperature knob we would have to tune
  (which, under our rules, must be fixed by a train-only criterion — see §6)?
- **Kill condition we would accept:** if the graph teacher's ranking correlates with our current
  teacher's above some threshold, it carries no independent information and the lane is dead. We
  cannot yet compute this — see §5 — but we will be able to within a day.

---

## 5. What is running / pending right now

**Iteration 43 RESULT** (scores in, pre-registered read applied). ARM E `champion_distill_alphamix10`
**0.906104** → **banked as finalist #1**. ARM F dual-pol gate as a width-neutral *replacement*
**0.904005**. ARM G same gate *added* **0.907616** with **AUC 0.946460**. Neither gate arm cleared
+0.006, so **the VH−VV lane is closed for good** — three independent forms of the same quantity have
now failed: raw (−0.0228), affine/SDWI (provably spanned by our `nn.Linear`), and indicator (this
round). The honest split: the gate genuinely improves **ranking** (best AUC we have ever posted, above
the leader) and gives it all back at the cut.

**Iteration 44 STAGED, not yet run.** Two things:

1. **The instrumentation fix that should have happened forty iterations ago.** `run_pipeline.py` has
   always written a diagnostic bundle (`oof_prob`, `y`, `p_test_raw`, per-fold test predictions), but
   the Colab notebook only ever downloaded the submission CSVs and the bundle directory is gitignored
   — so **every bundle has died with the VM.** Local count: zero. *That single missing copy is why the
   binormal `b`, the F1-optimal cut, and `P` have all been argued from leaderboard arithmetic instead
   of measured on labelled data.* Fixed. It is also why Q3's kill condition is not yet computable.
2. **Regime-matched calibration.** Not the no-op of §3.2 — a different defect, and one that is a fact
   of the code rather than a hypothesis. Our OOF and test scores have different **averaging
   structure**, and Platt is fit across the mismatch: an OOF row is the mean of **R=2** masked window
   views from **1** fold-model, while a test row is **1** real window averaged over `n_splits`
   fold-models. Each side is variance-shrunk on the axis the other is not. Harmless under the old
   prevalence pin (the cut was re-derived downstream); under a **literal 0.5 cut the Platt slope *is*
   the operating point.** `R` is read only on the held-out path, making it the one perfectly isolated
   operating-point lever in the pipeline. `tools/regime_match.py` rebuilds the calibration set at R=1
   offline at zero extra training cost; its `--views all` control reproduces the existing pooling
   bit-for-bit.

   **We have pre-registered a NULL as the expected outcome** — with ~3% of test mass within 0.05 of
   the cut, a slope change this size cannot move enough rows. A null would still be worth having: it
   closes the boundary-calibration lane on a *measurement* rather than an argument.

**Remaining work regardless of your answer:** finalist lock and the code-review package (35% of final
score).

---

## 6. Compliance constraints that bind any proposal

Non-negotiable. Final standing is **65% private leaderboard + 35% code review of the top 5**, so a
rules-violating artifact is worse than a weak one.

1. **Threshold tuning is FORBIDDEN.** The binary column must be the literal `p ≥ 0.5` cut of the
   probability column.
2. **No leaderboard-inverted quantity may reach the operating point.** Diagnosis only.
3. **Only supplied competition data.** No external DATA. Pretrained model **weights** are legal.
4. **No AutoML.** 5. **Seeded and reproducible.** 6. **The three data CSVs are never committed.**

**Our working three-prong test**, which we still find useful: a change is legal iff **(a)** the
decision rule stays a literal 0.5; **(b)** every knob is fixed by a **train-only** criterion, never
against a realized pos-rate target or LB feedback; **and (c)** it corrects `p(y|x)` under a
*demonstrably* mis-specified model rather than relabeling a fixed estimate.

Two clarifications learned this round. **Prong (b) is the one that bites**: it is why any
hyperparameter of a proposal must come with a train-only selection rule, pre-committed before the
realized positive rate is inspected. And **transductive use of unlabeled test *features* is
established practice here** — our largest win depends on it — so proposals may use the test features;
they may not use test *labels*, inferred label counts, or leaderboard feedback.

---

## 7. Closed lanes — do not re-propose without new evidence

- **Tree models / CatBoost.** Three fails (0.6976 → −0.0136 → 0.7186). The "shift-robust" adversarial
  GO gate was a false positive: a test-like *covariate* holdout still carries train labels, so it is
  blind to **conditional** shift.
- **Adversarial AUC as a selection criterion — RETIRED.** Correlates *positively* with realized
  transfer here (Spearman +0.68 across transforms, +1.00 across modalities), i.e. backwards.
- **Saerens / BBSE / EM prior correction.** Gate failed — conditional shift present.
- **All affine functions of existing bands.** Exactly spanned by our first `nn.Linear`. This closes
  AWEI and every linear index without a run. Measured: raw `VH−VV` cost −0.0228.
- **The entire VH−VV / cross-polarization lane**, now in three independent forms (§5).
- **Additive logit corrections and F-measure surrogate losses** — annihilated by train-refit Platt
  (§2.2). **This is the class most likely to be re-proposed; please do not.**
- **Distillation weight α.** A 5× sweep moved *one true positive* out of 309 rows.
- **Capacity-ADDING feature channels, generally.** OOF rises, LB falls, repeatedly.
- **Off-manifold TTA / hole-punched masking.** Diagnosed failure.
- **Multi-round self-distillation.** One round only — but see Q3 for the one way around it.
- **Spatial / geographic graph methods.** No coordinates exist and reconstructing them needs external
  data (§1). **GNNs over months or bands are also closed**: self-attention already learns a complete
  weighted graph over months, so a GNN is a *restriction* of it plus added width.
- **Single-seed "records."** Four have collapsed (§3.4).

---

## 8. Execution budget

- **Compute:** Colab GPU; a 25-run iteration takes ~25 minutes wall-clock. Not the constraint.
- **Submissions:** 5/day, ~40 remain. **Not the constraint.**
- **Binding constraints:** (i) **three days**, and 35% of the final score is a code-review package
  that must be finished; (ii) **ideas that survive a 0.019-sd seed distribution.** Anything worth
  running must be expected to move **≥0.006 seed-averaged**.
- **Slots:** realistically **one** experimental iteration remains after 44. So we want **at most two**
  concrete, ranked, implementable interventions — or a well-argued instruction to stop.

---

## 9. Deliverable format

For each proposed intervention:

1. **Mechanism** — what it changes, and why that moves F1 *at a fixed 0.5 cut specifically*.
2. **Platt check (new, mandatory)** — show explicitly that the mechanism is **not** an affine logit
   reparameterization, or explain why it survives a train-refit Platt (§2.2). Proposals failing this
   will be rejected without a run.
3. **Evidence** — actual papers, what they measured, on what kind of shift. Read them; do not
   summarize abstracts. **Say plainly when evidence is thin.**
4. **Kill condition, stated up front** — the check that makes the proposal moot, runnable *before* the
   slot is spent. Round 19 did this and we settled its proposal in ten minutes. That is the format.
5. **Implementation sketch** against §1.
6. **Expected effect size** in composite points, with an honest interval.
7. **Falsification** — what result proves it did not work, distinguishable from a ±0.019 seed draw.
   Note we get **one** public score per artifact; "average over 5 seeds" means *pool 5 seeds into one
   artifact*, not five uploads.
8. **Compliance check** against §6, explicitly — including the train-only selection rule for every
   hyperparameter.

**Please prioritize Q1 (§4.1).** The AUC premise has inverted and we do not think round 20's framing
— or ours — has caught up with it.

**And please stress-test Q2(a).** If the graph prevalence estimate is circular, we want to know before
we cite it in a report that will be code-reviewed.

If your honest conclusion is that we are at the achievable ceiling under these constraints, **say
that.** With three days left and 35% of the score resting on the code-review package, a well-argued
"stop and consolidate" is a valuable answer, not a failure.
