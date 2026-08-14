# UPDATE_23 — ROUND 23: **F1, AND NOTHING ELSE**

**Read this as if you know nothing about this competition. It is written to be self-contained.**
Supersedes UPDATE_20 / UPDATE_21. Written 2026-08-14. Competition deadline **2026-08-16**.

---

## §0. What this round is

We have measured, with leaderboard arithmetic that we show below and that you should check, that
**100% of our remaining gap to first place is in the F1 half of the metric.** Our ranking quality
(ROC-AUC) already *beats* the public leader. Our F1 does not, by 0.037–0.048.

So this round asks exactly one question, and we want it attacked from every angle:

> **Given a scoring metric that is 60% F1 evaluated at a hard, non-negotiable 0.5 probability
> threshold, and given a model whose ROC-AUC already exceeds the leader's, what is the
> highest-value LEGAL way to raise F1 — where "legal" specifically forbids tuning the threshold,
> forbids using any leaderboard feedback to set any knob, and forbids external data?**

Everything below exists to make that question precise and to stop you from re-deriving 46 iterations
of work we have already done. **The single most valuable thing you can do is prove one of our claims
wrong.** Two of the four agents in the previous round did exactly that and it changed what we shipped.

### Ground rules (these worked in previous rounds; keep them)

1. **We will check every claim you make.** Cite DOI / arXiv ID / a URL for every empirical or
   theoretical assertion. If you cannot find a source, say "unsourced conjecture" and mark it.
2. **Distinguish "I verified this by reading the paper/code" from "I inferred this."** A previous
   round handed us a confident claim about a library API that turned out to be right, and another
   that turned out to be a hallucinated function name. Label which is which.
3. **Correcting us is the highest-value output.** If §4 or §5 contains an error, that is the finding.
4. **Do not propose anything in §6 (the closed list) without new evidence that specifically overturns
   the reason it was closed.** Re-proposing a closed lane costs us reading time we do not have.
5. **A negative, well-sourced answer is a real deliverable.** "This cannot be done legally, here is
   the theorem" is worth more to us than a speculative method.

---

## §1. The problem, stated completely

**Competition.** Zindi × FAO × ITU, *GeoAI Challenge for Aquaculture Pond Identification*. Binary
classification: does a given location contain an aquaculture pond?

**The data. This is all of it — there is no imagery, no geometry, and no coordinates.**

- `Train.csv`: **1821 rows** (1817 after exact-duplicate removal), **146 columns** =
  `ID` + `label` + **12 spectral bands × 12 months**.
- The 12 bands: `VH, VV` (Sentinel-1 SAR, decibels) and
  `blue, green, red, re1, re2, re3, nir, nira, swir1, swir2` (Sentinel-2 optical, raw DN ×10⁴).
- So each row is a **12 × 12 table** — a one-year, 12-band satellite time series for one location,
  already reduced to a single pixel/parcel value per band per month. **No spatial extent, no
  neighbours, no lat/lon, no tile ID, no region.** The organizers removed them. IDs are random
  Crockford base32 and encode nothing (we checked: non-sequential, zero train/test overlap).
- `Test.csv`: **1030 rows**, same schema minus `label`.
- **⚠️ THE DESIGNED DIFFICULTY.** Train rows show **all 12 months**. Test rows show only
  **4–6 CONTIGUOUS months**; the rest are missing. This is a deliberate, structural train→test shift,
  not noise.
- **Class prior: train 0.4023 positive.** Our best estimate of the test prior is ~0.59–0.62 (§3).

**The metric.** The submission has two columns, both scored, and the final score is

```
score = 0.6 · F1(TargetF1) + 0.4 · ROC-AUC(TargetRAUC)
```

- `TargetF1` is **binary 0/1** — thresholded predictions. Evaluated as F1.
- `TargetRAUC` is a **continuous score** — evaluated as ROC-AUC, so only its *ranking* matters.
- We verified this decomposition to 4 × 10⁻¹⁰ on six separate submissions. It is exact.

**Leaderboard mechanics.** Public LB = **309 rows**, private = **721 rows**. Max 5 submissions/day,
100 total. You designate **2 finalists**. Final standing = **65% private LB + 35% a code/report
review of the top 5**. Public-LB noise on 309 rows is **±0.015** (binomial).

**Hard constraints, all of which are ours to keep and yours to respect when proposing:**

| constraint | status |
|---|---|
| **Threshold tuning is FORBIDDEN.** The cut is a literal `0.5`, hardcoded. | self-imposed, `compliance_mode: legal` |
| No external data of any kind | competition rule |
| Pretrained model **weights** are legal (we confirmed with the organizers' rules) | allowed |
| No AutoML | competition rule |
| Everything seeded and reproducible | competition rule |
| **No leaderboard-derived quantity may set any knob** | self-imposed |

**Why the threshold rule is self-imposed and non-negotiable.** Early in this project we "pinned" the
predicted positive rate to a chosen value, which is threshold tuning in disguise. We judged it a
rules violation, disclosed it in our report, deleted every score obtained under it, and rebuilt. We
will not reintroduce it. **Any proposal whose mechanism reduces to "move the operating point" is
dead on arrival** — see the three-prong test in §5.

---

## §2. Our current model (so you know what you are improving)

A **temporal Transformer**, ~71k parameters, trained on the 12×12 tables. In rough order of how much
each contributed:

1. **Relative-time reframing** — left-align each row's observed window to *t*=0 instead of using
   calendar months. Capacity-neutral. **+0.0128 LB**, the first real win.
2. **Masked window views (K=2 per row)** — every training row is randomly cropped to a contiguous
   4–6 month window drawn from the *measured* test window-length distribution, with antithetic
   pairing. This makes training look like deployment. (**Note for you: this already exists. Two
   separate research rounds have proposed it to us as a new idea.**)
3. **Cross-view invariance loss** — penalize the variance of the logit across the K views of a row.
4. **A permanence channel** `1[VH_dB(t) < −21]` — the single best hand feature we found (+0.010).
   Physically: ponds are *permanent low scatterers* in SAR.
5. **A dual-pol gate** `1[VH < −21] · 1[(VH−VV) < −8]` — a genuine interaction (AND beats either
   clause alone: univariate AUC 0.849 vs 0.801 / 0.756).
6. **Transductive soft self-distillation** on the 1030 unlabeled test rows — train a student against
   the teacher pool's soft predictions on test. **+0.0100, the largest single win in the project.**
   Run for **exactly one round** (Kumar/Ma/Liang: error compounds per self-training step).
7. **A 10-distinct-seed pool**, probability-averaged.
8. **Platt calibration fit on training out-of-fold predictions only**, then a literal 0.5 cut.

**Scores.** Champion / finalist #1 `champion_dualpolmix10_regimematch` = **0.907368983**
(AUC **0.945841814**, F1 **0.881720430**). Best composite we ever recorded: 0.910837. Finalist #2
`champion_archblend4` = 0.899643, kept as a decorrelated private-LB hedge.

**Public leader:** composite ~**0.929–0.936**, AUC **0.944897**.

---

## §3. 🔑 THE ARITHMETIC THAT DEFINES THIS ROUND. Check it — it is the whole brief.

**Step 1 — the gap is entirely F1.** Our AUC 0.945841814 already **beats** the leader's 0.944897 by
+0.000945. Solving `0.6·F1 + 0.4·0.944897 = leader_composite`:

| leader composite | ⇒ leader F1 | our F1 | F1 gap | composite gap |
|---|---|---|---|---|
| 0.929 | 0.9184 | 0.881720 | **+0.0367** | +0.0220 |
| 0.932 | 0.9234 | 0.881720 | **+0.0417** | +0.0250 |
| 0.936 | 0.9301 | 0.881720 | **+0.0483** | +0.0290 |

**Step 2 — the F1 column inverts exactly, so we know our confusion matrix.** F1 = 2·TP/(PP + P) is a
rational with a small denominator, and Zindi reports 9 decimals (and **truncates**, it does not
round). Over all denominators in [250, 500] the inversion is unique or near-unique:

| submission | reported F1 | ⇒ (TP, PP+P) |
|---|---|---|
| champion | 0.881720430 | **164, 372** |
| a frozen-encoder probe | 0.790190735 | **145, 367** ← unique solution |
| a fine-tuned probe | 0.794520547 | **145, 365** |

**Step 3 — and therefore P, the number of true positives on the public slice.** Each artifact's
predicted-positive *rate* over all 1030 test rows is known to us locally. If the public 309 is a
uniform subsample, `PP_public ≈ rate × 309`, so `P = (PP+P) − PP_public`:

| artifact | pos-rate | PP_public | **⇒ P** |
|---|---|---|---|
| champion | 0.5830 | 180.1 | **191.9** |
| frozen probe | 0.5699 | 176.1 | **190.9** |
| fine-tuned probe | 0.5670 | 175.2 | **189.8** |

Three artifacts, two unrelated model families, all landing on **P ≈ 190–192**.

**Step 4 — the diagnosis. At P = 191, our champion's public confusion cell is:**

```
PP = 181     P = 191     TP = 164
precision = 164/181 = 0.9061
recall    = 164/191 = 0.8586
F1        = 328/372 = 0.881720
true public prevalence = 191/309 = 0.618   vs our operating positive rate 0.583
```

**Precision exceeds recall by 0.047.** For F1, that is the unambiguous signature of a decision cut
placed **too HIGH** — we are predicting roughly **ten too few positives** on the public slice.
Moving ~10 rows from negative to positive, if ~7 of them were true positives, would take F1 from 0.8817
to ~0.896 (+0.014 F1 = +0.0084 composite).

### ⛔ AND WE ARE NOT ALLOWED TO USE ANY OF STEP 2–4 TO DO ANYTHING.

Every number in steps 2–4 is derived by inverting the **leaderboard**. Our standing rule is that any
LB-inverted quantity is **diagnosis only and must never touch the operating point**, and
`compliance_mode: legal` forbids threshold tuning outright regardless. We are showing you this
arithmetic because it **localizes the problem for you**, not because it is a lever. Please do not
propose using it.

### ⚠️ And there is a genuine tension you should attack

We have a **train-only** instrument for the same question. The classical result for the F1-optimal
threshold of a calibrated classifier is `t* = F*/2`, where `F*` is the achievable maximum F1 (Lipton,
Elkan & Naryanaswamy, *Thresholding Classifiers to Maximize F1*, arXiv:1402.1892 — please verify our
reading of it). Estimated on our masked, train-only, held-out replica, we get

```
δ̂ = F*/2 ∈ [0.479, 0.485]      across all 30 seed × regime combinations
```

So our **train-only** instrument says: the optimal cut is *slightly* below 0.5 — same DIRECTION as
the leaderboard arithmetic, but a magnitude of ~0.02 where the LB arithmetic implies ~0.035.
**Which one is right, and why do they disagree?** We would very much like a principled answer,
because the train-only one is the only one we are permitted to act on.

### And the hardest constraint on any answer

We measured the **density of test scores near the cut**: only **9–28 of 1030 rows** (mean ~15,
**1.5%**) lie in [0.45, 0.55]. To gain ~7 public TPs we need ~23 rows of the full 1030 to change
side. **There are not 23 rows near the boundary.** So the rows we need are currently at, say, 0.15–0.45
and are *confidently* misclassified positives.

> **This reframes the entire problem, and it is the most important sentence in this brief:**
> the F1 gap is **not** a calibration or boundary-placement problem. It is a *recall* problem on
> positives the model is confidently wrong about. Any method that only reshapes probabilities near
> 0.5 is arithmetically incapable of closing it.

---

## §4. What we know about the shift (context for any proposal)

- **Train and test are EXACTLY separable.** A cross-fitted discriminator on all 144 features achieves
  adversarial AUC **1.0000**. Not 0.99 — 1.0.
- **Consequence: importance weighting is not merely high-variance, it is not identifiable.** With
  disjoint support there is no density ratio to converge to. Empirically, Kish effective sample size
  comes out **73.2** with one discriminator and **687.8** with another on the same data — a 9.4×
  spread that measures the regularizer, not the data.
- **The shift has a genuine CONDITIONAL component.** A mixture goodness-of-fit test rejected
  `p(x|y)`-invariance at **p ≈ 0** (KS D = 0.186). This kills the label-shift family
  (Saerens-EM/MLLS, BBSE) — we built those, ran the gate, and did not ship them.
- **Out-of-fold (OOF) validation is BLIND here.** OOF composite sits at ~0.97 for artifacts whose
  public LB spans **0.72 to 0.907**. It has been anti-correlated with LB as often as correlated. We
  do not use it for selection and neither should any method you propose.
- Test window lengths are 4–6 contiguous months; month coverage is tent-shaped (MAR), but a real
  covariate shift also exists in the SAR *levels*.

---

## §5. The legality test any proposal must pass

We apply three prongs. State explicitly how your proposal passes each:

- **(a)** The decision rule remains a literal `0.5`. Not "0.5 after rescaling," not "0.5 on a
  transformed score chosen to move the cut." A literal 0.5 on a genuine probability.
- **(b)** Every knob is fixed by a **train-only** criterion — never against a realized positive-rate
  target, never against leaderboard feedback.
- **(c)** It corrects `p(y|x)` under a **demonstrably mis-specified** model, rather than relabeling a
  fixed estimate.

### ⚠️ The Platt Annihilation Theorem — read this before proposing any loss function

Our final step fits a two-parameter Platt map `σ(a·z + b)` on train OOF logits. Therefore, for **any**
method that induces an affine reparameterization of the logit `z' = αz + β`:

```
σ(a(αz + β) + b) = σ((aα)z + (aβ + b))
```

Refitting Platt's two parameters **recovers the identical function**. The method is annihilated — it
cannot change our submission at all.

**This has already killed, on our own analysis:** class weighting, cost-sensitive learning, balanced
softmax, logit-adjusted loss, prior correction by logit offset, and sigmoidF1 (whose `η` parameter is
a logit offset, which also means its own published threshold-free result is a threshold search).

**So the bar for a loss-function proposal is:** *does it change the RANKING of test rows, or only the
affine placement of the logit?* Only the former can matter. (Focal loss, for instance, is **not**
affine in the logit — it reweights examples by a function of their own confidence. Whether that
actually helps here is one of our questions in §7.)

---

## §6. CLOSED LANES — do not re-propose without new contrary evidence

Each closed on a measurement, with the reason:

| lane | why it is closed |
|---|---|
| Threshold / prevalence tuning | **illegal** (self-imposed, disclosed in our report) |
| Saerens-EM / MLLS / BBSE label-shift correction | gate FAILED, p≈0: conditional shift present |
| Beta and isotonic calibration | nested LR test insignificant (p 0.134/0.290); **direction reversed** (beta 15 down/0 up); isotonic's in-sample and cross-fitted AUC **disagree in sign** at n=1817 |
| Pool-then-calibrate (Ranjan–Gneiting / Rahaman–Thiery) | theoretically correct, direction correct (13 up / 0 down), but **+0.0003 to +0.002** — two orders below noise |
| sigmoidF1 and F-surrogate losses | Platt annihilation; also only ~15/1030 rows near the cut |
| Importance weighting / IWCV | adversarial AUC 1.0 ⇒ not identifiable |
| Gradient-boosted trees / CatBoost | three failures: 0.6976 standalone, −0.0136 blended, 0.7186 shift-robust |
| ROCKET / random convolutional kernels | −0.009 |
| Blending a weaker decorrelated member | lost **three** times (ROCKET −0.009, GBDT −0.0155, Presto −0.10) |
| **Foundation models (Presto, frozen AND fine-tuned)** | **just closed this round: 0.805 and 0.811 vs our 0.907** |
| Adding channels / width | lost on essentially every attempt |
| The `VH − VV` cross-pol feature, three forms | null — and it is **exactly linear** in two supplied columns, so a model given both can already represent it |
| Fourier / harmonic / seasonal-decomposition features | a 12-month period is unidentifiable from a 5-month window |
| Spatial graphs / GNNs over geography | no coordinates exist, and external data is forbidden |
| Adversarial AUC as a **selection criterion** | retired: correlates **positively** with realized transfer, i.e. backwards |
| ATC / average-thresholded-confidence as a screen | valid in-family (ρ +0.964), invalid out-of-family (ρ +0.738) |
| OOF as a selection signal | blind (see §4) |
| Tuning the distillation temperature/α | closed over a 5× range of α; the whole ladder was **one** public true positive |

---

## §7. 🎯 THE QUESTIONS WE MOST WANT ATTACKED

Ranked. Q1 and Q2 are where we think the answer is.

**Q1 — The confidently-wrong-positives problem.** Per §3, we need ~23 of 1030 rows to cross the cut
but only ~15 rows sit anywhere near it. So we must recover positives the model currently scores
*confidently low*. **What is the literature on improving recall on hard/confidently-missed positives
under a distribution shift with disjoint support, when the threshold is fixed and cannot move?**
Specifically: is there any principled method that raises `p(y=1|x)` for a *structured subpopulation*
of test rows without being a disguised global threshold shift? (Our instinct is that the honest
answer is "only a better model," and if so, say that plainly and tell us which model change.)

**Q2 — Consistent F1 maximization at a FIXED threshold under shift.** There is a real literature on
plug-in rules for non-decomposable metrics: Koyejo, Natarajan, Ravikumar & Dhillon, *Consistent Binary
Classification with Generalized Performance Metrics* (NeurIPS 2014); Narasimhan et al. on consistent
plug-in F-measure estimators; Lipton et al. arXiv:1402.1892. **These all conclude the optimal rule
thresholds `p(y|x)` at a metric-dependent value — which we are forbidden to move.** So:
   - Is there a formulation where the *classifier* is trained such that the F1-optimal threshold
     **equals 0.5 by construction**, with the constraint imposed via a **train-only** criterion?
   - Does that reduce to an affine logit shift (⇒ annihilated), or is it a genuinely different rule?
   - **Resolve the δ̂ = 0.48 vs LB-implied ≈ 0.465 disagreement in §3.** Is `t* = F*/2` even valid
     under covariate shift, and what is the right estimator when the calibration set and the
     deployment set have different prevalence *and* different missingness structure?

**Q3 — Losses that survive Platt annihilation and change the ranking.** Which training objectives
provably change the **order** of test scores rather than the affine placement of the logit? Focal loss
is our leading candidate (non-affine by construction), but Mukhoti et al. (NeurIPS 2020,
arXiv:2002.09437) argue focal loss changes calibration too — which for us is either irrelevant
(annihilated) or the whole point (if it reorders). **We want a clear theoretical separation: for each
of focal loss, label smoothing, LDAM, asymmetric loss, polyloss, and recall-oriented surrogates —
does it induce an affine logit map, or does it reorder?** Cite the derivation.

**Q4 — Getting more out of the one thing that actually worked.** Transductive self-distillation on the
1030 unlabeled test rows produced our single biggest win (+0.0100) and we cap it at one round because
self-training compounds error. **Is there a second, model-INDEPENDENT teacher** (so that re-teaching
is not the model re-consuming its own bias)? And is there a distillation variant that specifically
targets **recall on the test distribution** rather than matching soft probabilities everywhere? Note
we measured that our teacher is near-binary (only 0.5–2.9% of its mass in [0.45, 0.55]), so soft
distillation from it is already close to hard pseudo-labelling.

**Q5 — Ensembling for F1 rather than for AUC.** Every pooling operator we use (probability averaging,
rank averaging) is an AUC-oriented variance reducer. **Is there a pooling rule that maximizes expected
F1 at a fixed 0.5 cut?** E.g. selecting the member whose *decision* is best rather than averaging
scores; or a majority vote on the hard labels (which is not the same as thresholding the mean).
Note the constraint from Ranjan & Gneiting (JRSS-B 2010, 72(1):71–91): any non-trivial average of
distinct *calibrated* forecasts is necessarily uncalibrated.

**Q6 — Aquaculture domain signal we have not used.** Our best hand feature is SAR permanence
`1[VH_dB < −21]` (ponds = permanent low scatterers). One candidate survives our feature-span screen
unexploited: **LASCI**, a red/red-edge slope index (span R² 0.75, window-stability ρ 0.90, univariate
AUC 0.889). **Is there published, verifiable evidence that any specific index separates aquaculture
ponds from the confusable classes — rice paddy, seasonal wetland, natural water — using ONLY a 4–6
month window of the 12 bands we listed in §1?** Constraint: it must not be computable as a linear
function of the raw band values (we would already have it), and it must be stable under truncation
to 4–6 months.

---

## §8. Our own errors, stated first

We list these because past rounds wasted effort building on premises we had gotten wrong.

1. **We mis-cited our own motivation for `VH − VV`.** The canonical aquaculture SAR feature (Ottinger
   et al., IGARSS 2018, DOI 10.1109/IGARSS.2018.8651419) is **VH alone, pixel-wise temporal median**.
   The dual-pol *ratio* is not in that pipeline at all, and Ullmann et al. (Front. Remote Sens.
   3:905713, 2022) measure polarimetric derivatives as adding **0.1%** over intensity for water. Our
   three null results were the literature's prediction.
2. **We predicted, in writing and before the run, that fine-tuning a pretrained encoder would LOSE to
   freezing it, under our own measured law that "added capacity fitted to shifted rows hurts." It
   won**, by +0.00629 — which clears our pre-registered significance bar by 0.0003, i.e. by 1/50th of
   the noise band. We report this as a **prediction miss**. We do not claim the law is refuted, and we
   do not claim it held.
3. **Our significance bar of δ = 0.019 was leaderboard-derived** (measured by submitting seed
   replicates). Using it as a gate let LB feedback reach a decision knob. The clean replacement is the
   309-row binomial figure, **≈0.015**, with no LB input. Every decision we made still holds.
4. **A gate we built failed its own control and we nearly published its output.** We used
   `median(VH − VV)` as a control that was supposed to be exactly linear in the raw values; it scored
   R² = 0.62 instead of 1.0, because a **median is nonlinear**. General rule we now hold: *if a gate's
   control does not return the value arithmetic guarantees, every other number it prints is void.*
5. **Twice now, a research round has proposed the masked-window augmentation as a new idea.** It has
   been implemented since the beginning (§2, item 2). Please read §2 before proposing augmentations.
6. A previous round's headline conclusion rested on an estimate of P we have since retracted. The
   values in §3 supersede it.

---

## §9. Budget and deliverable format

- **Time: ~2 days.** Anything requiring a large training run or many leaderboard probes is out of
  scope. We have **5 submissions/day** and intend to spend very few.
- Our two finalists are **locked on measured scores** and will not be replaced by anything unmeasured.
  So the realistic target for this round is either **(a)** a single, high-conviction, legally-clean
  change worth ≥ +0.015 composite, or **(b)** a definitive, well-sourced negative that we can put in
  the report — which is worth 35% of the final standing.

**Format your response as:**

1. **Headline** — one paragraph: the single most valuable thing you found.
2. **Corrections to this brief** — anything in §1–§8 you believe is wrong, with sources. Put this
   *second*, not last; it is the highest-value section.
3. **Per-question answers** (Q1–Q6), each with: the claim, the source (DOI/arXiv), whether you
   verified it by reading or inferred it, and **an explicit legality analysis against the three prongs
   in §5 plus the Platt annihilation test.**
4. **Ranked recommendations** — at most 3, each with: expected effect size *in composite points*, the
   measurement that would falsify it, and what it costs to try.
5. **What you looked for and could not find.** Explicitly. A gap in the literature is information.
