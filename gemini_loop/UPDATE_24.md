# UPDATE_24 — ROUND 24: **F1, AND WE THINK WE WERE WRONG TO CLOSE IT**

**Written for a reader who knows nothing about this competition. Self-contained.**
Supersedes UPDATE_23. Written 2026-08-14. Deadline **2026-08-16**.

---

## §0. Read this part first: we are asking you to break our own conclusion

Last round we concluded, in writing, that the remaining gap was **unreachable** — that the positives
our model misses are "not recoverable from the source distribution." We closed five lanes on
measurements and stopped.

**Then we checked the arithmetic behind that conclusion and found a hole in it.** §3 below shows the
work. The short version: the argument that our operating point has no headroom relied on a theorem
whose precondition **we do not satisfy**, and a bound we never actually computed turns out to be
**wide open** — somewhere between "no headroom at all" and "+0.076 F1 available", with the data we
have unable to distinguish.

So this round has one job:

> **Find the legal way to raise F1. Assume we have missed something, because we just found that we
> did.**

**Correcting us is the highest-value thing you can produce.** It has now happened four times in two
rounds and each time it changed what we did.

---

## §1. The problem, completely stated

**Competition.** Zindi × FAO × ITU, *GeoAI Challenge for Aquaculture Pond Identification*. Binary:
does this location contain an aquaculture pond?

**The data — this is all of it.**
- `Train.csv`: **1821 rows** (1817 after dedup), **146 columns** = `ID` + `label` + **12 bands × 12
  months**. Each row is a 12×12 table: one year of satellite readings for one location.
- Bands: `VH, VV` (Sentinel-1 radar, decibels) and `blue, green, red, re1, re2, re3, nir, nira,
  swir1, swir2` (Sentinel-2 optical).
- **No images, no shapes, no neighbours, no coordinates.** The organizers stripped location. IDs are
  random and encode nothing (we checked).
- `Test.csv`: **1030 rows**, no label.
- ⚠️ **The designed difficulty:** train rows show **all 12 months**; test rows show only **4–6
  contiguous months**, rest missing. A deliberate train→test shift.
- Train is **40.23%** positive. Test is roughly **59–62%** positive.

**The metric.** Two scored columns:
```
score = 0.6 × F1(TargetF1)  +  0.4 × ROC-AUC(TargetRAUC)
```
`TargetF1` is **binary 0/1**. `TargetRAUC` is a **continuous score** where only the ordering matters.
Verified exact to 4×10⁻¹⁰ on eight submissions.

*Beginner note, because it drives everything below:* **AUC only cares about the ORDER** of the 1030
predictions — shuffle the values but keep the order and AUC is unchanged. **F1 only cares about WHICH
rows you called 1**. They are almost independent problems, and we are currently winning one and
losing the other.

**Leaderboard.** Public = **309 rows**, private = **721**. 5 submissions/day. You designate **2
finalists**. Final = **65% private LB + 35% a code/report review of the top 5**. Public noise on 309
rows ≈ **±0.015**.

**Hard constraints. Respect these in every proposal.**

| rule | source |
|---|---|
| **The decision threshold is a literal `0.5`. Tuning it is forbidden.** | self-imposed, disclosed in our report |
| No external data | competition rule |
| Pretrained **weights** are allowed | competition rule |
| No AutoML | competition rule |
| Seeded / reproducible | competition rule |
| **No leaderboard-derived quantity may set any knob** | self-imposed |

*Why the threshold rule is self-imposed and non-negotiable:* early on we "pinned" the predicted
positive rate to a chosen value. That is threshold tuning wearing a hat. We judged it a rules
violation, deleted every score obtained under it, disclosed it, and rebuilt. **Any proposal that
reduces to "move the operating point" is dead on arrival.**

---

## §2. Our model and where we stand

A **temporal Transformer**, ~71k parameters. What contributed, in order:

1. **Relative-time reframing** — align each row's observed window to *t*=0 instead of using calendar
   months (+0.0128, our first real win).
2. **Masked window views (K=2)** — every training row is randomly cropped to a contiguous 4–6 month
   window drawn from the *measured* test distribution. **This already exists. Three separate research
   rounds have proposed it to us as a new idea.**
3. **Cross-view invariance loss** — penalize logit variance across a row's K views.
4. **A SAR permanence channel** `1[VH_dB < −21]` (+0.010) — ponds are permanent low radar scatterers.
5. **A dual-pol gate** `1[VH<−21]·1[(VH−VV)<−8]` — a genuine interaction.
6. **Transductive soft self-distillation** on the 1030 unlabeled test rows — **+0.0100, our biggest
   single win.** Run exactly one round.
7. A **10-seed pool**, probability-averaged.
8. **Platt calibration fit on training out-of-fold predictions only**, then a literal 0.5 cut.

**Scores.** Champion `champion_dualpolmix10_regimematch` = **0.907368983**
(AUC **0.945841814**, F1 **0.881720430**). Best ever 0.910837.
**Public leader ≈ 0.929–0.936, AUC 0.944897** — *we beat them on AUC and lose badly on F1.*

---

## §3. 🔑 THE ARITHMETIC — INCLUDING THE HOLE WE JUST FOUND IN IT

### 3.1 The whole gap is F1

Our AUC (0.945842) **beats** the leader's (0.944897). Solving for their F1:

| leader composite | their F1 | ours | gap |
|---|---|---|---|
| 0.929 | 0.918 | 0.881720 | **+0.037** |
| 0.936 | 0.930 | 0.881720 | **+0.048** |

### 3.2 We know our exact confusion matrix, because F1 inverts

F1 = 2·TP/(PP+P) is a fraction with a small denominator, and Zindi prints 9 decimals (and
**truncates**). Over all denominators in [250,500] the inversion is unique or near-unique. Three
artifacts and their positive rates pin **P ≈ 190–192** (the true number of positives among the 309
public rows). At P = 191:

```
OURS:            TP 164   FP 17   FN 27   TN 101
                 recall 0.8586   precision 0.9061   F1 0.881720
LEADER (F1 .918): TP ~180  FP ~21
                 recall ~0.942   precision ~0.895
```

> **They find 16 more true positives for 4 more false positives.** Their advantage is concentrated in
> the **high-recall corner** of the ROC curve — the region you reach by being willing to accept a few
> more false alarms. **Global AUC barely penalizes being bad there**, which is exactly how they can
> have a *lower* AUC and a *much* higher F1.

### 3.3 ⛔ AND HERE IS THE HOLE IN OUR OWN CLOSURE ARGUMENT

Last round we argued the operating point had no headroom, via two steps. **Both are now in doubt.**

**Step we made:** F1 is at a maximum at the optimal threshold `t*`, so sitting at 0.5 instead costs
only a *second-order* amount, scaled by the density of scores near the cut — which we measured to be
tiny (~15 rows of 1030 within ±0.05 of 0.5). We computed the cost at **+0.0004 F1** and closed it.

**Hole 1 — the theorem's precondition fails.** The result we used (`t* = F*/2`, Lipton, Elkan &
Naryanaswamy, arXiv:1402.1892) holds **for calibrated probabilities**. Ours are calibrated by Platt
on training out-of-fold data whose prior is **0.4023**, then deployed on a population whose prior is
**~0.618**. They are *not* calibrated on the deployment distribution. **A density measured in a
mis-calibrated coordinate does not bound anything**, and the "flat maximum" argument inherits that.

**Hole 2 — we never computed the bound we were implicitly asserting.** We now have. Using our exact
AUC and our exact confusion cell, the pairwise-concordance budget decomposes as:

```
total (pos,neg) pairs = 191 × 118 = 22538
discordant            = (1 − 0.945842) × 22538 = 1220.6
  of which FORCED (a positive below the cut vs a negative above it) = 27 × 17 = 459
  FREE, split between "both above" and "both below" blocks          = 761.6   ← UNIDENTIFIED
```

The maximum F1 achievable **on our existing ranking**, by threshold alone, depends entirely on that
unidentified split:

| assumption | AUC among rows below our cut | **max F1 reachable** | vs our 0.881720 |
|---|---|---|---|
| all free discordance in the "above" block | 1.000 | **0.9574** | **+0.0757** |
| proportional split | 0.862 | **0.8946** | **+0.0129** |
| all free discordance in the "below" block | 0.721 | 0.8817 | +0.0000 |

> **The honest statement is that we do not know.** The middle row alone would be **+0.008 composite**
> — above our +0.006 significance bar. Our published "+0.0004, the lane is closed" was a *local*
> calculation presented as a global one. **We are formally retracting it as a closure.**

**⚠️ This does NOT license threshold tuning** — that stays forbidden and we are not asking you to
propose it. What it licenses is the question: *if there is real headroom in the high-recall region,
what LEGAL mechanism reaches it?*

### 3.4 The one thing that is genuinely solid

Only ~15 of 1030 rows lie within ±0.05 of the cut, but we need roughly 23–33 rows to change side. So
the rows we need are **not near the boundary** — they are scored confidently wrong. Our error
analysis on labelled held-out data confirms it: of the positives our model misses, the median score
is **0.170**, with ten below 0.10 and **zero** in [0.45, 0.50). Whatever the fix is, it is not a
nudge.

---

## §4. What we know about the shift

- **Train and test are EXACTLY separable**: a classifier distinguishing them scores adversarial AUC
  **1.0000**. Not 0.99 — 1.0.
- Therefore **importance weighting is not identifiable** (no overlap to transport across). Empirical
  proof: Kish effective sample size comes out **73** with one discriminator and **688** with another
  on identical data.
- **The shift has a real CONDITIONAL component**: a mixture goodness-of-fit test rejected
  `p(x|y)`-invariance at **p ≈ 0**. This kills the label-shift family (Saerens-EM/MLLS, BBSE) — we
  built them, ran the gate, did not ship.
- **Out-of-fold validation is BLIND here.** OOF sits at ~0.97 for artifacts whose public score spans
  **0.72 to 0.907**. Do not propose anything that selects on OOF.
- **One offline instrument DID predict the leaderboard, once.** Not aggregate OOF — a
  *mechanism-specific, control-baselined* one (see §6). Worth knowing when you design a validation.

---

## §5. Legality — the three prongs

State explicitly how your proposal passes each:

- **(a)** the decision rule stays a literal `0.5` on a genuine probability;
- **(b)** every knob is fixed by a **train-only** criterion — never a positive-rate target, never
  leaderboard feedback;
- **(c)** it corrects `p(y|x)` under a **demonstrably mis-specified** model rather than relabelling a
  fixed estimate.

### ⚠️ Two theorems that kill most proposals before they start. Check yours against both.

**1. Platt annihilation.** We finish with a two-parameter map `σ(a·z+b)` fit on train OOF. So for any
method inducing an affine logit change `z' = αz+β`:
`σ(a(αz+β)+b) = σ((aα)z + (aβ+b))` — refitting `a,b` recovers the identical function. Annihilated.

**2. Pointwise-loss order invariance — the stronger one.** For ANY objective of the form
`Σᵢ [yᵢ·l₁(zᵢ) + (1−yᵢ)·l₀(zᵢ)]` — one fixed pair of functions applied to every example — the
population minimizer is `T(η(x))` where `η(x)=p(y=1|x)` and **T is one fixed monotone function**.
Therefore **ROC-AUC is exactly unchanged** and the F1 effect is a **pure threshold slide** along an
unchanged ordering.

*This kills, provably:* focal loss, asymmetric loss (ASL), LDAM, PolyLoss, label smoothing, class
weighting, logit adjustment, balanced softmax, sigmoidF1. For focal the warp is *proven strictly
order-preserving* (Charoenphakdee et al., CVPR 2021, arXiv:2011.09172, Thm 3/5/11 + Lemma 14).

**⚠️ Two external research reports last round both ranked ASL as their #1 recommendation. It is
refuted by this theorem and we did not run it. Do not repeat that.**

**The three escape routes** (this taxonomy is ours; challenge it):
- **E1 — non-decomposable objectives**: pairwise/listwise ranking, AUC and **partial-AUC** surrogates,
  constrained/Lagrangian metric optimization. **WE HAVE NEVER TESTED THIS CLASS.**
- **E2 — losses depending on the model's behaviour on OTHER inputs**: consistency/invariance (we have
  one), **mixup/VRM (never tried)**, contrastive auxiliaries.
- **E3 — changing the TARGET as a function of x**: distillation on soft labels. **Our single biggest
  win lives here.** That is not a coincidence — it is the only family in our history that escapes the
  theorem.

---

## §6. Closed lanes — and an honest grading of how firmly

Do not re-propose these *unless* you have specific contrary evidence. Note the confidence column:
**two of these we now consider soft.**

| lane | why closed | how firm |
|---|---|---|
| Moving the threshold / prevalence matching | illegal, self-imposed | **FIRM (rule)** — but the *arithmetic* argument is RETRACTED, §3.3 |
| "The operating point has no headroom" | +0.0004 second-order bound | ⚠️ **RETRACTED** — precondition fails, true bound is [0, +0.076] |
| Saerens-EM / MLLS / BBSE label-shift | gate failed, p≈0, conditional shift | FIRM |
| Beta / isotonic calibration | LR test insignificant; **direction reversed**; isotonic overfits at n=1817 | FIRM |
| Pool-then-calibrate | correct theory, moved 13 rows of 1030 | FIRM |
| Hard majority voting | moved 4–7 rows; dead even with post-hoc k | FIRM |
| Any pointwise loss (focal/ASL/LDAM/Poly/smoothing) | order-invariance theorem | **FIRM (theorem)** |
| JTT error-set upweighting | provably reordered (ρ 0.984) yet gained **0 true positives** on the LB | FIRM for JTT; **the E1/E2 classes are untouched** |
| Importance weighting / IWCV | adversarial AUC = 1.0 ⇒ not identifiable | FIRM |
| GBDT / CatBoost | three failures (0.6976, −0.0136, 0.7186) | FIRM |
| ROCKET | −0.009 | FIRM |
| Presto (frozen AND fine-tuned) | 0.805 / 0.811 vs our 0.907 | FIRM |
| Blending a weaker decorrelated member | lost three times | FIRM |
| Optical spectral indices | **every one tested is ≥97% linearly reachable** from the raw bands at the per-month level, so the network can already represent them | FIRM, and general |
| Fourier / harmonic / phenology features | a 12-month period is unidentifiable from a 5-month window | FIRM |
| Spatial graphs / GNNs | no coordinates exist; external data forbidden | FIRM |
| Adding channels / width | lost on nearly every attempt | soft — one add (the dual-pol gate) won |

**A general result worth carrying:** what is outside our model's reach is not the band *index* but the
**nonlinear temporal statistic** (median, min, threshold-count). Our one winning hand-feature is
exactly that: a threshold indicator, not a ratio.

---

## §7. 🎯 THE QUESTIONS. Q1 and Q6 are where we think the answer is.

**Q1 — Optimize the HIGH-RECALL REGION, not global AUC.** §3.2 shows the leader wins at recall ~0.94
where we sit at 0.859, while our *global* AUC is higher. Global AUC averages over regions the metric
does not care about. So: **what is the state of the art in partial-AUC / recall-region-targeted
ranking optimization, and does any of it survive our two theorems?** Cover pAUC surrogates,
two-way/one-way pAUC, ranking losses restricted to a TPR band, Neyman-Pearson classification, and
constrained ERM (e.g. "maximize recall subject to precision ≥ c" via Lagrangian/proxy-Lagrangian
methods). **For each: does it reduce to a threshold slide, or does it genuinely reorder?** Show the
derivation. This is class E1 and **we have never tested it.**

**Q2 — Resolve §3.3 properly.** Given a fixed ranking, an exactly-known AUC and one exactly-known
confusion cell, **how much F1 headroom exists?** Is there a sharper bound than our [0, +0.076]? Is
there a legitimate, train-only way to estimate where our ROC actually sits in the high-recall region
— given that our labelled held-out data does not reproduce the deployment shift? And: **is Lipton's
`t* = F*/2` recoverable under prior shift**, or genuinely unusable when the calibration prior (0.4023)
differs from the deployment prior (~0.618)?

**Q3 — E2 and E3, the two untested escape classes.** (a) **Mixup / vicinal risk minimization** on
time-series under distribution shift — it is non-pointwise by construction, so the theorem does not
apply. Does it help recall under covariate shift, and are there time-series-specific variants?
(b) A **second, model-INDEPENDENT teacher** for distillation. Our biggest win is one round of
self-distillation, capped because self-training compounds error. Candidates we have *not* tested:
co-training on the SAR/optical view split, tri-training, prototype/centroid teachers from target data
only. Which is best supported for *this* setting?

**Q4 — Is the label noise real?** Of the positives we confidently miss, the median predicted
probability is 0.170. Either the model is badly wrong or **some of those labels are wrong**. What is
the state of the art in identifying label noise (confident learning / Cleanlab, area-under-margin,
influence functions), and — important — **is relabelling or removing training rows legal under a "no
threshold tuning, train-only" regime?** We think yes, but argue it.

**Q5 — Attack our closures.** Pick the ones graded soft, or any you think we got wrong. Specifically:
is "every optical index is ≥97% linearly reachable" a sound argument, or an artifact of how we
measured it? Is our claim that added width always loses actually supported?

**Q6 — 🏆 KAGGLE AND ZINDI GRANDMASTER PRACTICE. This gets its own dedicated researcher.**
We have been reasoning from first principles and theorems. Competition grandmasters have empirical
playbooks we may simply not know. Find **specific, documented, winning solutions** — write-ups,
solution threads, published notebooks — for competitions with these features, and tell us what they
actually did:
  - **F1 or other threshold-dependent metrics at a FIXED, non-tunable threshold** (rare and valuable
    — most competitions let you tune, so be careful to distinguish);
  - **severe train→test distribution shift**, especially where train and test are near-separable;
  - **truncated / variable-length time series**, train longer than test;
  - **satellite / remote-sensing tabular time series** without imagery;
  - **very small tabular datasets (~1800 rows)** where the leaderboard is noisy.
For each: what was the winning trick, is it legal under §5, and does it survive the two theorems in
§5? **Concrete and cited beats general advice.** "Grandmasters use pseudo-labelling" is worthless;
"in competition X, the 1st-place solution did Y, here is the thread, and here is why it applies" is
what we need.

---

## §8. Our own errors, so you do not build on sand

1. **We closed the operating-point lane with a local argument presented as global** (§3.3). Retracted.
2. We **mis-cited our own motivation** for a feature: the canonical aquaculture SAR feature is **VH
   alone, temporal median** (Ottinger et al., IGARSS 2018, DOI 10.1109/IGARSS.2018.8651419); the
   dual-pol *ratio* is not in that pipeline at all. Our three null results were the literature's
   prediction.
3. A feature gate of ours **failed its own control** — we used `median(VH−VV)` as a supposedly linear
   control; a median is nonlinear, so the control was meaningless. General rule we now hold: *if a
   gate's control does not return the value arithmetic guarantees, every number it prints is void.*
4. A second gate was **uninterpretable until we added a control baseline**: "7 of 24 rows recovered"
   looked like success; against a control that recovers 6 anyway, it was a null — and the null
   correctly predicted the leaderboard.
5. **Our significance bar of 0.019 was leaderboard-derived**, which let LB feedback reach a decision
   knob. Replaced with the 309-row binomial figure, **≈0.015**.
6. **Three separate rounds have proposed the masked-window augmentation to us as new.** It has existed
   since the beginning. Please read §2 item 2.

---

## §9. Budget and format

- **~2 days.** No large training programs. 5 submissions/day; we will spend very few.
- Our two finalists are locked on measured scores. A realistic win is **(a)** one high-conviction
  legal change worth ≥ +0.015 composite, or **(b)** a definitive well-sourced negative for the report
  (worth 35% of final standing).

**Format your answer as:**
1. **Headline** — one paragraph: the single most valuable thing you found.
2. **Corrections to this brief** — second, not last. Highest-value section.
3. **Per-question answers**, each with: the claim, a DOI/arXiv/URL, whether you **VERIFIED by reading**
   or **INFERRED**, and an explicit check against **both theorems in §5** plus the three prongs.
4. **Ranked recommendations** — at most 3. Each with expected effect **in composite points**, the
   measurement that would falsify it, and what it costs to try.
5. **What you looked for and could not find.** A gap in the literature is information.
