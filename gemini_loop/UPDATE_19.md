# Deep-Research Brief — Round #19 (Claude Deep Research / Gemini Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)

**Date:** 2026-08-11 · **Best public LB: 0.910837 (legal, 5-seed pooled)** · **Deadline: 2026-08-16 (5 days)** ·
**Finalists to designate: 2** · **Submission budget: not binding (≈25 usable before the deadline at 5/day)**

---

## 0. Read this first — the question has changed

Every previous brief in this series asked some version of *"how do we get better?"* That question is now
**resolvable to a single term**, and we want this round aimed at it exclusively.

The metric is `0.6·F1@0.5 + 0.4·ROC-AUC`. Zindi reports F1 and AUC as **separate columns**, so we can
decompose our own gap exactly:

| | our best legal artifact | public LB leader | gap | worth in composite |
|---|---|---|---|---|
| **ROC-AUC** | 0.942861 | 0.944897 | +0.002036 | **+0.000814** |
| **F1 @ 0.5** | 0.889488 | ≈0.920235 *(implied)* | +0.030747 | **+0.018448** |
| composite | 0.910837 | 0.930100 | +0.019263 | |

**95.8% of the remaining gap is the F1 term.** Our ranking is, to within 0.002 AUC, as good as the
leader's. We are losing purely on *how many rows fall on the correct side of a hard 0.5 cut*.

So the round-19 question is narrow and concrete:

> **How do you raise F1 at a FIXED, un-tunable 0.5 threshold from ~0.889 to ~0.920, while holding
> ROC-AUC at ~0.944, on a 1,030-row target set under covariate shift — without touching the threshold?**

We are not asking for a general survey. We are asking for **instrumental methods, from papers you have
actually read**, that attack that specific quantity. If the literature says our framing is wrong, say so
and show why — but engage with *this* decomposition, not the old one.

### Posture on our own "closed" lanes
Treat every negative result below as a **rebuttable hypothesis**. Our measured seed-to-seed LB sd is
**0.019** and public-slice binomial noise is **±0.012**; most of our historical verdicts are single-seed
reads with effect sizes inside that band. If a well-supported paper points at something we shelved,
**argue to re-open it.** We would rather re-run a "dead" idea done properly than miss a real lever.

---

## 1. Self-contained problem statement

Binary classification: is this ~10 m cell a managed aquaculture pond? Each row is **one isolated pixel's
multivariate time series** — a **12-month × 12-band** cube. Bands: Sentinel-1 SAR **VH, VV** (dB, always
co-present when a month is observed) + 10 Sentinel-2 optical bands (individually missing under cloud).
**No lat/lon, no image patch, no static covariates, no neighbourhood.**

- **Train:** 1,821 rows → **1,817** after exact-duplicate removal. ~40.2% positive. **All 12 months present.**
- **Test:** **1,030** rows. Each has only **4–6 CONTIGUOUS visible months** (measured 345/343/342;
  1,030/1,030 contiguous). Sentinel value −9999. Public **309** / private **721** (30/70).
- **The designed shift is therefore temporal-window truncation plus a genuine covariate shift on the SAR
  level.** An offline label-shift goodness-of-fit test **FAILED** (KS D=0.186, p≈0), i.e. there is a real
  **conditional** shift component, not pure label shift — so Saerens/BBSE prior correction is unsafe and
  is switched off.
- **Measured test prevalence ≈0.56–0.58**; our artifacts realize ≈0.588 — i.e. we are already AT or
  slightly ABOVE the estimated test prior, and there is **no positive-rate gap to close.** The two
  estimators agree: MLLS 0.578, BBSE 0.559 (`tools/label_shift_gate.py`, iter35b). ⚠️ **CORRECTION
  (2026-08-11):** an earlier revision of this line read "believed true test prevalence ≈0.65", a stale
  figure carried over from the retired prevalence-pin era and contradicted by our own measurement. Any
  proposal whose value rests on driving the realized pos-rate up toward 0.65 is arguing from a premise
  we have already falsified — and, with the calibrator fit under the correct observation regime and
  landing at the measured prior, a further upward shift would be an operating-point move chosen for its
  effect on the 0.5 cut, i.e. threshold tuning in a calibration costume. See
  `gemini_loop/RESPONSE_19_CLAUDE.md` §Proposal 2.

**Metric:** `0.6·F1 + 0.4·AUC`. Two columns: `TargetF1` (binary, at a **hard 0.5 cut**) and `TargetRAUC`
(probability). **Threshold tuning is explicitly forbidden by the rules.**

**Our model** (from scratch, ~71k params): a small temporal Transformer over the 12 months.
Base input = 24 channels/month (12 standardized band values + 12 missing-indicators), plus optional
engineered channels. Key structural choices, each an LB-validated win:
- **relative-time reframing** — the observed window is left-aligned to t_rel=0 (+0.0128, our largest
  structural win; attacks calendar-month memorization).
- **cross-view invariance** — each training row is masked into K=2 synthetic 4–6-month windows drawn from
  the empirical test window distribution, with a penalty on logit variance across views (λ=1, +0.0047).
- **permanence channel** — per-month indicator `1[VH_dB(t) < −21]` (+0.010 seed-averaged).
- **legal calibration** — Platt fit on **training OOF only**, then a **literal 0.5 cut**.
- **seed pooling** — per-seed Platt, then probability average. Worth a reliable **+0.0055 to +0.0061**.

**The masking trap (important for any feature proposal):** because test windows are 4–6 months and train
is 12, only **n-invariant** statistics transfer. Means, medians, interior quantiles and *fractions* are
safe; min/max/range/counts are window-length biased and are not.

---

## 2. What changed since round 18 — the ceiling broke

Round 18 was written against a hard empirical wall: four structurally different constructions all landed
at **0.899882 / 0.899643 / 0.899512 / 0.899512**. We diagnosed that as a **bias floor under the covariate
shift, not a variance floor**, and predicted that only *test-distribution information* would move it.

**That prediction was confirmed.**

### 2.1 iteration 41 — transductive soft self-distillation (+0.0100)
Zero new parameters. A non-distilled 5-seed permanence pool acts as **teacher**; its soft probabilities
(T=1, **never thresholded**) on the 1,030 unlabeled test rows become a target for the student, added to
the labeled BCE with weight α and a linear warm-up ramp. The unlabeled views are **on-manifold contiguous
sub-windows** of each test row's own visible window (never hole-punched — off-manifold masking is the
diagnosed cause of an earlier TTA failure).

`champion_distill_seedavg5` = **0.909868** (AUC 0.944024, F1 0.887097) — first artifact ever above the
floor, +0.009986, clearing the pre-committed +0.006 bar on a 5-seed average. **Both terms rose.**

### 2.2 iteration 42 — the α knob is closed, exactly
| arm | AUC | F1 | composite | Δ |
|---|---|---|---|---|
| α=0.3 ×5 | 0.942280 | 0.884097 | 0.907370 | −0.002498 |
| α=1.5 ×5 | 0.942861 | 0.889488 | **0.910837** | +0.000969 |
| α=0.7 ×10 | 0.944024 | 0.881720 | 0.906642 | −0.003226 |

Nothing clears +0.006 over a **5× range of α**. Total spread 0.0035.

### 2.3 A new measurement instrument (this is genuinely useful and we had not noticed it)
Zindi's F1 column is a **small-denominator rational**, so it inverts exactly:

```
55/62 , 330/371 , 82/93   =>   TP = 165, 165, 164
```

The whole α ladder is **one true positive and one predicted positive out of 309 public rows.** One F1
row-flip ≈ **0.0055 composite**; one AUC concordant-pair ≈ **4.4e-5**. This tells us the public slice can
resolve F1 changes only in ~0.0055 quanta, and it is why every "within noise" verdict in our ledger
should be read as "within one or two rows."

We also observed that **α=0.7 at 5 seeds and at 10 seeds have bit-identical AUC (0.944024425)** — the same
concordant-pair count. Adding five seeds did not change the ranking *at all*; the entire −0.0032 was one
row crossing the cut. Public level differences at this scale carry **no ranking information**.

> **Compliance note, stated because it constrains you too:** this inversion is **diagnosis only**. The
> public slice's positive count is a leaderboard-inverted quantity and **must never feed the operating
> point** — that would be leaderboard probing. We deliberately did not solve for it. Do not propose
> anything that requires it.

### 2.4 iteration 43 — running now, results pending
Three arms, 25 runs, all completed and all passing the transductive sanity gate; **LB scores not yet in**:
- **ARM E** `champion_distill_alphamix10` — α-marginalized pool, 10 members at **10 distinct seeds**
  (α=0.7 ×5 + α=1.5 ×5). Inter-seed ρ = 0.9804, pos-rate 0.5883.
- **ARM F** `champion_dualpol_rep_seedavg5` — the dual-pol gate **replacing** permanence (25 ch).
- **ARM G** `champion_dualpol_add_seedavg5` — the same gate **added** to permanence (26 ch).

The dual-pol gate is `1[VH < −21] · 1[(VH−VV) < −8]`. Train-only, window-matched univariate AUC of the
mean-pooled fraction: VH-only **0.8012**, ratio-only **0.7556**, **AND gate 0.8487** — the AND beats both
clauses, so it is a genuine interaction; the best VH-only threshold anywhere in [−26,−12] is only 0.7917.
Risk noted pre-run: the gate's marginal shifts harder than permanence (window-matched train→test mean
−0.122 vs +0.061; per-channel adversarial AUC 0.597 vs 0.550).

---

## 3. THE CENTRAL PUZZLE — and a strong clue we already hold

We need **+0.031 F1 at fixed AUC**. Here is the clue.

In iteration 41 we also ran **ARM T**: the same cross-view logit-variance penalty, but pointed at the
*unlabeled test rows* (label-free consistency, λ_u=0.5, no teacher). As a **pool** it failed (0.893752)
and we dropped it. **But look at its members:**

| artifact | F1 | AUC | composite |
|---|---|---|---|
| `tcons_s42` (single seed) | **0.901333** | 0.933447 | **0.914179** |
| `tcons_s13` (single seed) | **0.897507** | 0.925923 | 0.908873 |
| `tcons` 5-seed pool | 0.871795 | 0.926687 | 0.893752 |
| best distill pool (α=1.5) | 0.889488 | 0.942861 | 0.910837 |

**Those two single seeds hold our two highest F1 values ever measured, ~0.012 above anything the
distillation lane has produced.** The label-free consistency term appears to buy **F1 specifically**,
at an AUC cost — precisely the trade we now want to make, since AUC is the term we have already maxed.

The pool failed for a **mechanical** reason we diagnosed: pooled AUC (0.926687) sat *between* its
members, i.e. the ranking pooled normally, but pooled F1 (0.871795) fell *below both* members. The
unlabeled variance penalty compresses logits toward a constant, per-seed Platt slopes then diverge, and
the probability-average lands at a drifted pos-rate. **We killed the method for a pooling bug.**

Arithmetic on the prize: if we could hold AUC at 0.944 and reach `tcons_s42`'s F1 of 0.9013, that is
**0.918409**. Reaching the leader's implied F1 of 0.9202 at our AUC is **0.929751** — the leader.

**Question 3A (highest priority).** What is the right way to *ensemble models whose probability
calibration differs*, when the metric applies a hard threshold to the averaged probability? Our current
`calibrated_pool` does per-member Platt → probability average. Candidate alternatives we want assessed
against the literature: pool in **rank space** then apply a single calibration; pool in **logit** space;
**temperature-match** members before averaging; median instead of mean; or fit one Platt map on the
*pooled* OOF rather than per-member. Which of these is theoretically correct when members have
heterogeneous slopes, and which preserves both AUC and threshold placement? This single fix may be worth
more than any new feature.

**Question 3B.** Why would a **consistency/smoothness penalty on unlabeled target data** improve F1 at a
fixed threshold more than it improves AUC? Our hypothesis is that it sharpens the decision boundary
locally (moving near-cut rows decisively to one side) while slightly degrading global ordering. Is this
a known, named effect? Does the semi-supervised literature (consistency regularization, VAT, entropy
minimization, FixMatch-style sharpening) predict a **local-boundary** gain that global AUC understates?
If so, what is the correct way to *combine* it with distillation rather than choosing between them?

---

## 4. Open questions, ranked

**Q1 — Threshold-fixed F1 optimization (the main event).**
AUC-optimizing training is indifferent to *where* errors sit. We need discriminative power concentrated
near the 0.5 cut. What does the literature actually support here, with evidence that transfers under
covariate shift? Specifically: cost-sensitive / class-weighted losses, focal loss, soft-F1 / dice
surrogates, ranking losses restricted to a margin band around the operating point, and proper-scoring-rule
choices that change *where* a calibrated 0.5 lands. **Which of these are legitimate modeling choices
rather than disguised threshold tuning?** (See §5 — this line matters to us.)

**Q2 — Calibration under conditional shift, legally.**
Our Platt slopes run 1.35–1.55, i.e. training OOF probabilities are systematically under-confident and
Platt is sharpening them. We realize a positive rate of ~0.588 against a believed true prevalence of
~0.65. We may **not** move the threshold. We *may* change the model and the (train-only) calibration map.
What is the principled, non-probing way to make a legally-placed 0.5 cut land better when the target
conditional differs from the source? Note the label-shift gate **failed**, so Saerens/BBSE/EM prior
correction is off the table on validity grounds, not just compliance grounds.

**Q3 — Ensembling heterogeneously-calibrated members.** As stated in Q3A above.

**Q4 — Is one round of self-distillation really the limit?**
We deliberately ran **one** round (Kumar/Ma/Liang: self-training error compounds per step, unbounded at
our shift magnitude) and never re-taught from a distilled student. Is there published evidence for a
*safe* multi-round scheme under measurable covariate shift — e.g. with a fixed anchor teacher, confidence
filtering, or a divergence constraint back to the original teacher? What is the actual stopping criterion
in the literature, and can it be evaluated without leaderboard feedback?

**Q5 — What could plausibly explain a competitor reaching F1 ≈ 0.92 on this data?**
Given the same 1,030 rows, no external data, no spatial context, a hard 0.5 cut, and AUC essentially
identical to ours. We want hypotheses that are *testable in 5 days*, not speculation. Note our previous
belief that the leader used "plain CatBoost at ~0.94" was investigated and found to be a **phantom** — the
cited forum post states no score and we had inverted its one actionable sentence. Tree models in our own
hands transfer catastrophically here (three independent attempts: 0.6976 naive, 0.7186 shift-robust).

---

## 5. Compliance constraints that bind any proposal

Non-negotiable; a proposal that violates these is unusable regardless of merit. Final standing is
**65% private leaderboard + 35% code review of the top 5**, so a rules-violating artifact is worse than
a weak one.

1. **Threshold tuning is FORBIDDEN.** The binary column must be the literal `p ≥ 0.5` cut of the
   probability column. We run `compliance_mode: legal` — Platt fit on **training OOF only**, then 0.5.
   A historical `pinned` mode exists solely to reproduce old anchors and its output is never submitted.
2. **No leaderboard-inverted quantity may reach the operating point.** Diagnosis only. See §2.3.
3. **Only supplied competition data.** No external DATA. Pretrained model **weights** are legal
   (verified) — Presto was tested frozen-only and fine-tuning remains untested.
4. **No AutoML.**
5. **Seeded and reproducible.** We do not set GPU-deterministic algorithms; measured seed-to-seed LB
   spread is 0.019 and we state that openly.
6. **The three data CSVs are private** and are never committed.

**An honest open question for you:** where exactly is the line between a *legal modeling choice that
changes the realized positive rate* (class weighting, prior-corrected loss, a different calibration
family) and an *illegal threshold shift in disguise*? Our working rule is that no knob may be tuned
against a realized pos-rate target or against leaderboard feedback. We would value a defensible
articulation of this boundary, because Q1 and Q2 both live near it.

---

## 6. Closed lanes — do not re-propose without new evidence

Each of these cost real submissions. Argue to re-open only with a specific paper and a specific reason
our test was under-powered.

- **Tree models / CatBoost.** Three fails at every sophistication level: 0.6976 naive → blend −0.0136 →
  0.7186 shift-robust. The "shift-robust" adversarial GO gate was a **false positive**: a test-like
  *covariate* holdout still carries train labels, so it is blind to **conditional** shift.
- **Adversarial AUC as a selection criterion — RETIRED.** It correlates **positively** with realized
  transfer here (Spearman +0.68 across transforms, +1.00 across modalities), i.e. backwards. It also
  saturates: a mild synthetic shift scores 0.9955 at 0.0046 AUC cost while our real shift scores 0.9670.
- **Marginal feature alignment.** Erases the measured shift almost completely (adv-AUC 0.9291→0.3608) at
  zero label cost — and buys nothing. Deployable forms scored −0.003.
- **Saerens / BBSE / EM prior correction.** Gate failed (conditional shift present).
- **The operating-point lane per se.** ~+0.0005 legally available against a −0.030 downside; ~90% of the
  F1 gap was measured to be near-cut **ranking**, not cut placement. *(Note: this measurement predates
  our AUC reaching parity with the leader's, and Q1/Q3 may reopen it — flagging honestly.)*
- **Capacity-ADDING feature channels, generally.** Repeated pattern: OOF rises, LB falls. Width cost is
  real. Replacements have fared better than additions.
- **Raw `VH−VV` and all affine forms of it.** Our SDWI is *exactly* `−5.697415 + 0.230259·(VH+VV)`
  (verified to 3.6e-15), so those arms measured width cost, not information. The **indicator** form is
  the live test (iteration 43, pending).
- **Off-manifold TTA / hole-punched masking.** Diagnosed failure.
- **Single-seed "records."** Four separate ones (0.906492, 0.913263, 0.912759, 0.914179) all collapsed to
  ~0.8995 when seed-averaged. **Never trust an unpooled result.**

---

## 7. What we can actually execute in 5 days

- **Compute:** Colab GPU; a 25-run iteration (5 folds × ~60 epochs each) takes ~25 minutes wall-clock.
  So ~25–30 runs per iteration is comfortable; 2–3 more iterations are realistic.
- **Submissions:** 5/day, ~25 usable before the deadline. Not the binding constraint.
- **Binding constraint:** *ideas that survive contact with a 0.019-sd seed distribution.* Anything worth
  running must be expected to move ≥0.006 seed-averaged, or be a variance reduction we can justify
  without a level read.
- **What we will do with your answer:** iteration 44 is the last experimental slot; iteration 45 is
  finalist consolidation plus the code-review package. So we need **at most two or three concrete,
  ranked, implementable interventions**, each with: the mechanism, the expected effect size, the failure
  mode, and how to tell success from seed noise. Depth over breadth. **Please prioritize Q1 and Q3.**

---

## 8. Deliverable format we want back

For each proposed intervention:
1. **Mechanism** — what it changes and why that should move F1 at a fixed 0.5 cut specifically.
2. **Evidence** — the actual papers, with what they measured and on what kind of shift. Read them; do
   not summarize abstracts. Say plainly when evidence is thin.
3. **Implementation sketch** against the model described in §1 (temporal Transformer, K=2 masked views,
   train-only Platt, seed pooling).
4. **Expected effect size** in composite points, and an honest interval.
5. **Falsification** — what result would prove it did not work, distinguishable from a ±0.019 seed draw.
6. **Compliance check** against §5, explicitly.

Rank your proposals. If your honest conclusion is that we are already near the achievable ceiling given
the constraints, **say that** — a well-argued "stop here and consolidate" is a valuable answer with five
days left and 35% of the score riding on the code review.
