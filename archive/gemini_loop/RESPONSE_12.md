# RESPONSE_12 — Gemini Deep Research (round 12), triaged 2026-07-28

**Verdict: 1 genuine contribution, 2 factual errors against our own measurements, 1 recommendation
that inverts our evidence, 2 restatements of the brief.** Net new information: low. The single
valuable item (§4 below) is one Gemini did not state clearly and that I have sharpened here.

**Meta-lesson for the loop.** `UPDATE_12.md` was thorough enough that most of the reply is our own §0
and §2 read back to us. A brief that pre-answers its own questions leaves nothing to add. Future
briefs should state the evidence but ask the question *open*, and should be explicit that
**restating our findings scores zero.**

---

## 1. ✅→⚠️ Two-column decoupling — right lane, WRONG assignment

**Accepted:** the columns are independent and we should stop feeding both from one score vector.
That was already our §0 #1 (we told it this), so it is a confirmation, not a finding.

**Rejected as specified:** *"rank-averaged ensemble for `TargetRAUC`, raw calibrated probabilities
from your single best model for `TargetF1`."* Three problems, in increasing severity:

1. **It misreads our diversity finding.** "Diversity is a liability at a pinned threshold" was derived
   from *cross-class* blends where the member was **weaker** (ROCKET −0.040, GBDT −0.011). Equal-strength
   within-class pooling did **not** lose: `champion_archblend4` (ρ̄ 0.9524, 4 transformers) is our
   **highest reliable artifact at 0.894643**. The corrected law is *gate on level gap, not correlation* —
   it does not license "use one model."
2. **"Single best model" is the winner's curse, and we have already quantified it.** At k≈79 candidates,
   E[max] inflation ≈ **+0.03** (RESEARCH_11 §2.4). Our "single best" is `seq_a_xview` seed 42 = 0.8955,
   a **known lucky draw** whose 5-seed reliable level is 0.8865. Gemini is recommending we put our
   **0.6-weighted** column on our single highest-variance artifact.
3. **It is backwards on variance.** `TargetF1` is a *set*. Set membership near the cut is exactly what
   seed noise perturbs (sd 0.0191 ≈ 11 rows on full test, **≈3 rows on the 309-row public slice**).
   The high-weight column is the one that most needs **variance reduction**, i.e. pooling.

**What we will actually test:** the hybrid, with the assignment chosen on evidence rather than
asserted — pooled artifact for **both** columns as the control, versus pooled-for-RAUC ×
{pooled, single} for F1. Constructible from existing `preds_*.npz` at zero training cost.

## 2. ➖ "Trust the LB, discard CV" — restatement, and the tactical advice is harmful

We wrote this in the brief (§0 #6, quoting `sdv`). No new content.

**The added tactic is actively wrong for us:** *"use your 5 daily submissions to map the test
distribution."* Our seed sd is **0.0191** and the public slice moves **≈3 rows per 0.010**. Mapping a
distribution through a ±0.019 noise channel at 5 samples/day is not estimation, it is the winner's
curse on a schedule. Our discipline — screen offline, submit rarely, one variable, seed-paired — is
correct and stays.

## 3. ❌ Feature advice — contradicts the brief's own exclusion list AND our screen

**Error A — the recommended ratios are on our excluded list.** Gemini proposes *"AWEIsh and MNDWI
ratios."* `UPDATE_12` §3 states, and we verified algebraically: **AWEI is exactly linear** in the bands
(a linear model already spans it) and **MNDWI is 0/0-conditioned over water**. These are the specific
indices we asked it to exclude. We also measured the WIF/EVI/SDWI family at **−0.075 LB**.

**Error B — "immediately delete `blue`" is refuted by our own iter25 run**, which Gemini had not seen:

```
c_dropblue    ATCF1 -0.0133~   DIS +0.0117   votes=1/2  -> HOLD
```
The ATC-F1 margin is **negative** and inside its own seed sd. Blue deletion is **not supported**.
Gemini reasoned from the T=0.5963 figure in the brief to a deletion; the measurement disagrees.
(`c_dropvvblue` did clear 2/2 — but that is the **VV** deletion carrying it, and it is a ρ=0.9841 twin
of `c_dropvv`.)

**Accepted:** keep S1 amplitude, never detrend, VH is primary. That is our §2 (D), restated.

**Still unanswered, and it was the point of Q6:** we asked for a ranked, sourced shortlist of
**non-degenerate** drift-invariant ratios with an invariance argument each. We did not get one. Deriving
it ourselves: the obvious first candidate is **`VH − VV` in dB**, which *is* the log cross-pol ratio —
it cancels any per-period multiplicative gain (calibration drift, incidence-angle and atmospheric
scaling) while preserving the cross-pol contrast, and it is Class-A n-invariant when pooled by median.
It has been queued since round 04 and never run.

## 4. 🔑 Calibration — the ONE genuine contribution, though not as Gemini framed it

**Rejected as proposed:** Saerens-EM is on our do-not-repropose list. The Saerens/BBSE family assumes
**label shift** (p(y) moves, p(x|y) fixed). Ours is **covariate shift** — adversarial AUC 0.8915 on the
values themselves. Wrong estimator class by assumption, and when we ran it, it returned prior **0.44
against a true ≈0.649.**

**But the underlying observation is sharp, and it fixes a scoping subtlety in our own law.** Our rule
reads *"the metric is rank-only, so calibration is invisible."* That is correctly scoped in
`UPDATE_09` (*"**after the prevalence pin fixes** the predicted-positive count"*) — but the scope
matters more than we have been treating it:

| column | is monotone recalibration visible? |
|---|---|
| `TargetRAUC` | **No** — rank-invariant by definition. Ever. |
| `TargetF1`, **with our prevalence pin** | **No** — the pin re-derives the cut, absorbing any monotone map. |
| `TargetF1`, **at a literal 0.5 cut** | **YES — calibration becomes the entire column.** |

**So the rule-risk remediation (Q4) and the calibration question are the same question.** The moment we
replace the prevalence pin with a literal 0.5 threshold to be defensible in the code review, monotone
calibration stops being invisible and becomes the sole determinant of the 0.6-weighted column. We have
been treating those as two separate work items. They are one, and it is the highest-stakes item on the
board because it touches **35% of a top-5 finish**.

Bias-corrected temperature scaling **fit on training data only**, then a literal 0.5 cut, is a
defensible construction. The open question — which Gemini did not address and which Claude Research
should — is whether train-only calibration can land the cut near 0.649 **under covariate shift**, or
whether it lands near the train prevalence (~0.40) and costs us the ≈+0.07 the pin is worth.

## 5. ❌ "Use CropNet / a 2-D spectral-temporal architecture" — against all our evidence

The architecture search is **closed** (iterations 18–25). The public-LB leader is at ~0.94 on **plain
CatBoost** and says outright *"the model isn't the bottleneck."* Recommending a new architecture is
the exact axis our last eight iterations proved flat. The accompanying claims ("keep the temporal
dimension", "the shift is in the values not the indicators") are verbatim from our brief.

---

## What this changes

**Nothing about iter26's direction.** The ratio battery and the two-column split were already the
plan; this round supports both without sharpening either. The one thing it *does* change is that
**the rule-risk fix is now coupled to the F1 column** (§4) and must be screened, not assumed safe.

**Still pending:** Claude Research. The questions that remain genuinely open after Gemini are **Q3**
(is *any* train-only validation possible here), **Q6** (a non-degenerate ratio shortlist), and the
**Q4 sub-question** above — whether train-only calibration survives covariate shift.
