# Round 23 — Q1 & Q2: Confidently-wrong positives, and consistent F1 at a FIXED 0.5 threshold

Agent scope: Q1 and Q2 from §7 of `UPDATE_23.md` ONLY. Q3–Q6 owned by other agents.
Started 2026-08-14. **This file is written incrementally — if it is truncated, what is here is final.**

Labelling convention used throughout:
- **VERIFIED** = I fetched and read the paper's own text (arXiv full text / published PDF / HTML).
- **INFERRED** = derived by me from something verified, or a reasoning step, not stated in a source.
- **UNSOURCED CONJECTURE** = explicitly flagged, no source found.

---

## STATUS LOG

- [DONE] Lipton/Elkan/Naryanaswamy arXiv:1402.1892 — read via ar5iv full text
- [in progress] Koyejo et al. NeurIPS 2014
- [in progress] Narasimhan et al. plug-in consistency
- [pending] Q1 literature: hard-positive recall under disjoint-support shift

---

# ★ CORRECTION 1 (HIGHEST VALUE) — §3 contains NO disagreement. Your two instruments agree exactly. What you have measured is your OOF optimism, not a conflict of estimators.

**Source: Lipton, Elkan & Naryanaswamy, "Thresholding Classifiers to Maximize F1 Score", arXiv:1402.1892
(published as Optimal Thresholding of Classifiers to Maximize F1 Measure, ECML-PKDD 2014, LNCS 8725, pp. 225–239,
DOI 10.1007/978-3-662-44851-9_15).**
**Status: VERIFIED — read the full text via ar5iv (https://ar5iv.labs.arxiv.org/html/1402.1892).**

### What the theorem actually says

Corollary 1, verbatim from the paper:

> "An instance with predicted probability *s* is assigned to the positive class by the optimal decision rule
> that maximizes F1 if and only if *s ≥ F/2* where *F = 2tp/(2tp + fn + fp)* is the F1 score achieved by
> **this optimal decision rule**."

The critical word is the one the brief drops: **`F` is the F1 score achieved by the optimal rule ON THE
DISTRIBUTION THE RULE IS APPLIED TO.** It is a **fixed point / self-consistency condition**, not a
transportable constant. The paper's own framing is per-batch: the same batch's score distribution
simultaneously determines the optimal threshold and the F1 it attains.

I re-derived it independently and it is a one-line marginal-example argument (INFERRED, but it reproduces
the paper's Corollary 1 exactly, so it is a check, not a new claim):

Let s = p(y=1|x) be calibrated **on the deployment distribution D**. For rule `predict 1 iff s ≥ t`:
```
A(t) = TP = E_D[ s · 1{s ≥ t} ]
B(t) = PP + P = P_D(s ≥ t) + E_D[s]
F(t) = 2A/B
```
Lower t to admit an infinitesimal mass dm of examples at score s. Then
```
dF = 2(A + s·dm)/(B + dm) − 2A/B  ≈  (2 dm / B²) · (sB − A)
```
so admitting them helps **iff s > A/B = F/2**. Hence the optimum satisfies `t* = F(t*)/2`. □

**Every quantity in that derivation — `E_D[s]`, `P_D(s≥t)`, `E_D[s·1{s≥t}]` — is an expectation under D,
the DEPLOYMENT distribution.** Nothing in the theorem is a property of the training set.

### Therefore: is `t* = F*/2` valid under covariate shift? YES — and this corrects the brief's framing of Q2(c).

**Claim (INFERRED, but a direct corollary of the VERIFIED theorem):** `t* = F*/2` is *distribution-free*.
It holds under arbitrary covariate shift, arbitrary prior shift, and arbitrary missingness shift, subject to
exactly one condition: **s must be calibrated with respect to the deployment distribution**, and **F\* must be
the max F1 attained on the deployment distribution.** Shift does not break the *formula*. Shift breaks the
*plug-in estimate* of F\*, and (separately) it can break calibration.

So the question "is t\* = F\*/2 even valid under covariate shift?" has the answer: **the formula is valid; your
estimate of F\* is not.** These are different failures and the brief conflates them.

### The numbers: your two instruments are the SAME instrument, read at two different F*

| instrument | F* it is using | t* = F*/2 |
|---|---|---|
| your masked train-only replica | F*_OOF ≈ **0.958 – 0.970** | 0.479 – 0.485 |
| leaderboard arithmetic (§3) | F*_test ≈ **0.930** (= leader's F1 at composite 0.936) | **0.465** |

`0.930 / 2 = 0.465`. That is the LB-implied number in your brief **to three decimals**, reproduced by the
*same* Lipton formula. There is no second estimator and no theoretical conflict. The gap
`0.485 − 0.465 = 0.020` is exactly half of `0.958 − 0.930 = 0.028`, i.e.

> **Δt* = ΔF*/2.  Your δ̂ discrepancy IS your OOF optimism, divided by two.**

This is fully consistent with §4's own statement that "OOF composite sits at ~0.97 for artifacts whose public
LB spans 0.72 to 0.907." You already knew OOF was inflated. §3 re-measures that inflation through a second
channel and mislabels it as an estimator disagreement.

**Consequence for the brief:** delete the sentence "Which one is right, and why do they disagree?" They do not
disagree. Both say `t* = F*/2`. Neither is actionable, because:
- the LB reading is LB-derived (illegal by your own prong (b)), and
- the train-only reading is the same formula fed a **biased** F\*, and the bias is in the direction that makes
  δ̂ too HIGH, i.e. **your train-only instrument systematically under-corrects.** It is not conservative by
  choice; it is conservative because OOF is optimistic. That is worth stating in your report.

### Second-order correction: the estimator is ALSO biased by the Winner's Curse, in the same direction

**VERIFIED** — §7 of arXiv:1402.1892 is titled "A Winner's Curse" and shows that empirically selecting a
threshold on a finite sample exhibits *sharp threshold behaviour*: below `n < (1−b)/b²` samples, the procedure
picks large thresholds with constant probability even though they are far from optimal, because different
thresholds converge at different rates. With b ≈ 0.40 on your train set, `(1−b)/b² = 0.60/0.16 = 3.75`, so at
n = 1817 you are far above the pathological regime and **the Winner's Curse is NOT your problem.** I checked
this specifically so you do not chase it. (INFERRED: the arithmetic; VERIFIED: the `(1−b)/b²` criterion.)

### Third correction, free: the all-positive baseline you can compute without the LB

**VERIFIED** — Theorem 4.3 of arXiv:1402.1892: an uninformative classifier's F1-optimal rule is **predict
everything positive**, attaining `E[F1] = 2b/(1+b)` for base rate b.

At your estimated test prior b ≈ 0.618 this is `1.236/1.618 = 0.7639`. Your champion's 0.8817 clears it by
0.118. Two uses: (i) it is a sanity floor for the report; (ii) it quantifies *how much of your F1 is
information vs. prevalence* — at b=0.618 a coin flip is already worth 0.764, which is why F1 differences of
0.04 between you and the leader correspond to much smaller differences in actual ranking skill. Your AUC
already beats theirs; this is the arithmetic reason both facts can be true at once.

---

# ★ CORRECTION 2 — §3's row-count arithmetic understates the difficulty. It is ~33 rows, not 23.

**Status: INFERRED (pure arithmetic on the brief's own numbers).**

§3 says: "To gain ~7 public TPs we need ~23 rows of the full 1030 to change side." Then §3 separately models
the flip as "moving ~10 rows from negative to positive, if ~7 of them were true positives."

Those two sentences use different quantities. `23 = 7 × (1030/309)` is the full-test-set count of **true
positives** that must flip. But you cannot flip only true positives — the same score-threshold movement drags
false positives with it, at the ~70% precision you assumed. The number of **rows** that must change side is
```
10 public rows × (1030 / 309) = 33.3 rows of 1030
```
not 23. Against a measured 9–28 rows (mean ~15) in [0.45, 0.55], the deficit is **33 vs 15**, i.e. you need
more than 2× the entire near-boundary population, not 1.5×.

**This strengthens your own §3 conclusion**, which is why it is worth fixing rather than glossing: the claim
"any method that only reshapes probabilities near 0.5 is arithmetically incapable of closing it" is *more*
true than you wrote. The corrected statement for your report: even flipping **every single row** in
[0.45, 0.55] to positive, at the marginal precision those rows would have (≈ the local calibration value,
≈0.5, not 0.7), yields ≈ 7–8 TPs and ≈ 7–8 FPs across the full test set, i.e. ~2.2 public TPs. That closes
roughly **one third** of the required gap while costing precision. Boundary reshaping is not merely
insufficient; it is off by a factor of ~3.

---

# ★★ CORRECTION 3 (THE HEADLINE) — The threshold is worth ≈ 0.0005 F1. Not 0.014. The entire §3 threshold discussion is numerically irrelevant, and your own sparse-density measurement is the proof.

**Status: INFERRED (my derivation + arithmetic on the brief's own measured numbers). The underlying
optimality condition is VERIFIED from arXiv:1402.1892 Corollary 1.**

### The derivative

From the same marginal-example argument as Correction 1, with `f(t)` the score density and
`B = PP-rate + prevalence`:
```
dF/dt  =  −(2 f(t) / B) · (t − F(t)/2)
```
Two things follow immediately, and they are the whole story:

**(i) F1(t) is at a MAXIMUM at t\*, therefore it is LOCALLY FLAT.** The loss from sitting at 0.5 instead of
t\* is **second order**:
```
F(t*) − F(0.5)  ≈  (f(t*) / B) · (0.5 − t*)²
```

**(ii) The coefficient is the score density near the cut — the very quantity you measured to be tiny.**
A sparse density near 0.5 is simultaneously (a) the reason you cannot flip many rows and (b) the reason you
do not need to. You treated the sparse density purely as an obstacle. It is also an *exemption*.

### Plug in your measured numbers

Your measurement: 9–28 rows (mean ~15) of 1030 in the width-0.10 band [0.45, 0.55]
⇒ `f ≈ 15 / (0.10 × 1030) = 0.146` (density in probability units).
`B = 0.583 (your positive rate) + 0.618 (est. prevalence) = 1.201`.

The honest fixed point **for your own model** is not 0.465 and not 0.485. It is obtained by solving
`t = F(t)/2` with *your* deployment F1: `F(0.5) = 0.881720` ⇒ `t* ≈ 0.441–0.443`.

```
F(t*) − F(0.5) ≈ (0.146 / 1.201) × (0.5 − 0.442)²  =  0.1216 × 0.003364  =  4.1 × 10⁻⁴
```

> **Moving the threshold from 0.5 to its true optimum is worth +0.0004 F1 = +0.00025 composite.**
> Your F1 gap is 0.037–0.048. The threshold accounts for **about 1% of 1%** of it.
> It is ~60× below your 0.015 noise floor.

### An assumption-light cross-check that does not use the density model at all

Flip every row in [0.45, 0.50) — say ~7–8 of 1030, i.e. ~2.3 public rows, at local calibrated precision ≈0.5:
```
public: TP 164 → 165.1,  PP 181 → 183.3,  P = 191
F1 = 2(165.1)/(183.3+191) = 330.2/374.3 = 0.8822   (was 0.881720)
ΔF1 = +0.0005
```
Same order of magnitude, derived two independent ways. **The threshold lane is closed by arithmetic, not by
your compliance rule.** Add it to §6 with the reason "worth 4×10⁻⁴ F1; below noise by 60×." That is a
*better* reason than "illegal", because it survives any future relaxation of the compliance rule and it is
publishable in your report as a measured negative.

---

# ★★ CORRECTION 4 — §3 Step 4 is wrong. "Precision exceeds recall ⇒ the cut is too high" is NOT a theorem for F1. Here is an explicit counterexample where the cut is EXACTLY optimal and precision exceeds recall by 0.082.

**Status: INFERRED — my own counterexample; verify it yourself, it is four lines of arithmetic.**

§3 Step 4 asserts: *"Precision exceeds recall by 0.047. For F1, that is the **unambiguous** signature of a
decision cut placed too HIGH."* This is a widely repeated folk heuristic and it is false.

**Counterexample.** Take a perfectly calibrated model whose scores take two values:
mass 0.60 at s = 0.9, mass 0.40 at s = 0.3. Prevalence = 0.6(0.9) + 0.4(0.3) = **0.66**.

| threshold | PP | TP | P | precision | recall | **F1** |
|---|---|---|---|---|---|---|
| t ≤ 0.3 (all positive) | 1.00 | 0.66 | 0.66 | 0.660 | 1.000 | 0.7952 |
| 0.3 < t ≤ 0.9 | 0.60 | 0.54 | 0.66 | **0.900** | **0.818** | **0.8571** ← optimum |
| t > 0.9 | 0 | 0 | 0.66 | – | 0 | 0 |

The optimum is the middle row. Check the fixed point: `F/2 = 0.8571/2 = 0.4286`, which indeed lies in
(0.3, 0.9], so `t = 0.5` **is** an F1-optimal threshold here. And at that exact optimum,
**precision − recall = 0.900 − 0.818 = +0.082**, larger than the +0.047 you observe. □

**The correct diagnostic** is the Lipton condition, and only that: the cut is above optimal iff `t > F(t)/2`.
For you: `0.5 > 0.881720/2 = 0.4409` — so yes, your cut IS above optimal. But the *magnitude* that matters is
not `precision − recall`; it is the **mass of rows in [F/2, 0.5) = [0.441, 0.500)**, which you measured to be
≈ 7 rows of 1030. §3 reads the 0.047 precision–recall spread as "ten too few positives on the public slice."
The Lipton condition says the number of rows that *should* flip is ≈ 7 of 1030 ≈ **2 of the public 309**.
**§3 overstates the correct flip count by roughly 5×.**

### And the prescription §3 implies is actively HARMFUL

§3 Step 4 observes "true public prevalence 0.618 vs our operating positive rate 0.583" and suggests matching
them. Test that against the optimality condition. To lift the public positive rate from 0.583 to 0.618 you
must flip ~33 rows of 1030, reaching down to a score of roughly 0.28 given your measured density. Every row
in [0.28, 0.441) has `s < F/2` and therefore, by Corollary 1, **flipping it strictly reduces F1**. Concretely,
at honest calibrated precision ≈0.40 on that band:
```
public: TP 164 → 168,  PP 181 → 191,  P = 191
F1 = 336/382 = 0.8796   (was 0.881720)   ΔF1 = −0.0021
```
> **Prevalence matching LOSES F1 here.** This is a second, independent reason the operating-point lane is
> dead, and unlike your compliance argument it is a *measurement*. It also retires the intuition — visible
> throughout §3 — that "we predict ten too few positives" is a description of a recoverable loss. It is not:
> those ten rows are, in expectation, majority-negative.

**What survives from §3.** The single most important sentence in your brief survives completely intact and is
now over-determined:

> *"The F1 gap is not a calibration or boundary-placement problem. It is a recall problem on positives the
> model is confidently wrong about."*

Corrections 2, 3 and 4 all point the same way and each is independently sufficient. You should promote this
from an inference to a **theorem-backed conclusion** in your report, citing Corollary 1 of arXiv:1402.1892
plus the flatness bound. It is the cleanest negative result in the project.

---

# Q2 — CONSISTENT F1 MAXIMIZATION AT A FIXED THRESHOLD UNDER SHIFT

## Q2(a) — Can a classifier be TRAINED so that its F1-optimal threshold is 0.5 by construction?

### Answer: Yes, and the condition is exactly `F* = 1`. There is no other way. Here is the theorem.

**Proposition (INFERRED — my derivation; the input, Corollary 1, is VERIFIED from arXiv:1402.1892).**

> Let `s` be **calibrated** on the deployment distribution D. Then
> ```
> t*  =  F*/2      and therefore      0.5 − t*  =  (1 − F*)/2
> ```
> Since `F* ≤ 1` with equality iff the classifier is perfect on D:
> **the F1-optimal threshold of a calibrated classifier is 0.5 if and only if F* = 1.**
> For every imperfect calibrated classifier, `t* < 0.5` strictly.

**This is the plain answer Q1 asked you to give.** "Train so that 0.5 is F1-optimal" and "train a perfect
classifier" are **the same sentence**. The threshold-suboptimality `0.5 − t*` is not an independent quantity
you can attack; it is *pinned to half your remaining F1 error*. Closing it and closing the model gap are one
operation, and the model half is the entirety of it.

Note this also explains, without any leaderboard input, why your δ̂ is where it is:
`δ̂ = 0.479–0.485 ⟺ F̂* = 0.958–0.970`. Your train-only instrument was never measuring a threshold. It was
measuring your OOF F1, halved. It carries exactly zero information beyond what your OOF F1 already told you —
and §4 says OOF is blind. **δ̂ is a blind instrument wearing a threshold costume.** That is the single most
useful thing to say about it in your report.

### The dichotomy: there is no third option

**Proposition (INFERRED).** Let `s` be calibrated on D with optimum `t* = F*/2 < 0.5`, and let `s' = T(s,x)`
be any alternative score for which 0.5 is F1-optimal on D. Exactly one of:

- **Case A — T preserves the ranking** (`s' = g(s)`, g strictly increasing). Then
  `{s' ≥ 0.5} = {s ≥ g⁻¹(0.5)}`, and F1-optimality forces `g⁻¹(0.5) = F*/2`. **T is definitionally the
  forbidden threshold move**, re-expressed. `F*` is unchanged; not one row's fate differs from simply setting
  the cut to F*/2. Fails prong (a) verbatim ("not 0.5 on a transformed score chosen to move the cut").
- **Case B — T changes the ranking.** Then `s'` is a *different classifier* with its own `F*'`, and by the
  Proposition above, 0.5 is optimal for it iff `F*' = 1` (given `s'` is calibrated — which your pipeline
  enforces by construction with the Platt step). If `s'` is left uncalibrated, you are back in Case A relative
  to `s'`'s own calibrated version.

**There is no method in between.** Any proposal you receive this round claiming to "make 0.5 optimal by
construction" is, on inspection, Case A. Use this as a screening test — it is faster than the Platt test and
strictly stronger.

## Q2(b) — Does it reduce to an affine logit shift? YES, and the F-measure literature says so in its own words.

**Source: Kotłowski & Dembczyński, "Surrogate regret bounds for generalized classification performance
metrics", arXiv:1504.07272 (Machine Learning 106(4), 2017, DOI 10.1007/s10994-016-5591-7).**
**Status: VERIFIED — read via ar5iv full text.**

Their proof of the surrogate regret bound routes through cost-sensitive classification, and they state
(verbatim, on the simultaneity of the bound over all cost parameters α):

> "the misclassification costs only influence the **threshold**, but not the **function**, the surrogate loss,
> or the regret bound."

**That sentence is your Platt Annihilation Theorem, published, peer-reviewed, and in the exact literature Q2
names.** Every route from "F1 is my metric" to "therefore change my training" that the consistency literature
sanctions passes through a cost-sensitive reduction, and the reduction provably touches only the threshold.
You should cite this in your report; it converts your §5 argument from an in-house derivation into a citation.

Corroborating, from the same paper (VERIFIED): they show empirically that hinge-loss minimisation retains
**non-zero Ψ-regret even as n → ∞** for F-measure, and explain why:

> "the risk minimizer for hinge loss is already a threshold function on η(x), with the threshold value set to
> 1/2. If … the optimal threshold θ* is different than 1/2, the hinge loss minimizer will necessarily have
> suboptimal Ψ-risk."

**Read this carefully — it is a warning aimed directly at you.** A loss whose Bayes risk minimiser is a hard
threshold at 1/2 (hinge, and any margin loss that is not strictly proper) **destroys the probability
information you need** and is *provably* inconsistent for F1. Your logistic/proper-loss + Platt pipeline is
the correct choice and this paper is the citation for it. Do not let a Q3-style proposal move you to a
non-proper margin loss.

Corroborating from a second source (**VERIFIED** — read via ar5iv, arXiv:1806.00640, "Binary Classification
with Karmic, Threshold-Quasi-Concave Metrics", Koyejo group), on the Koyejo et al. NeurIPS 2014 result:

> "[Koyejo et al. 2014] provided an implicit characterization of the optimal threshold, but the solution of
> which in turn requires the knowledge of the **optimal classifier**, which is unknown in practice."

i.e. the NeurIPS 2014 result Q2 cites has exactly the Lipton fixed-point structure and is **distribution-
dependent** (confirmed: "the fixed-point equation characterizing δ* depends on the confusion matrix C(f_δ) at
that threshold, which varies across different data distributions"). It gives you nothing Lipton does not, and
nothing transportable.

### Verdict on the whole Q2 family, against your three prongs + Platt

| method | prong (a) literal 0.5 | prong (b) train-only | prong (c) corrects p(y\|x) | Platt test | verdict |
|---|---|---|---|---|---|
| Plug-in threshold at t*=F*/2 (Lipton / Koyejo / Narasimhan) | **FAIL** — is a threshold move | fail (needs deployment F*) | no — relabels a fixed estimate | n/a | **DEAD** (and worth 4×10⁻⁴ anyway) |
| Cost-sensitive / cost-ratio reduction | fail | pass | no | **ANNIHILATED** (K&D quote above) | **DEAD** |
| Any strictly-increasing rescaling to "make 0.5 optimal" | **FAIL** by Case A | — | no | annihilated if affine-in-logit | **DEAD** |
| Non-proper margin loss (hinge etc.) to force a 1/2 threshold | passes (a) technically | pass | **no — destroys p(y\|x)** | n/a | **DEAD, and provably inconsistent for F1** (arXiv:1504.07272) |
| Improve F* itself (rank-changing model change) | **PASS** | pass | pass | **survives** | the only survivor — see Q1 |

**Summary of Q2: the entire consistent-F1 literature is, for your constraint set, a proof that the lane is
empty.** That is a real deliverable under your own ground rule 5, and it is now sourced three ways.

## Q2(c) — The correct estimator when calibration set and deployment set differ in prevalence AND missingness

### Step 1: the right decomposition. F1 factorises into (ROC point) × (prevalence). Use this identity.

**Status: INFERRED (elementary algebra, but I have not seen you use it and it settles the question).**

For an operating point with true-positive rate `r` and false-positive rate `f`, at prevalence `π`:
```
PP/n = π r + (1−π) f ,     P/n = π ,     TP/n = π r

            2 π r
F1(π, r, f) = ─────────────────────────
              π r + (1−π) f + π
```
`F* (π) = max over (r,f) on the ROC curve.` **F1 depends on the deployment distribution only through the ROC
curve and the prevalence — nothing else.** This is the estimator you want, because it separates the two
things that shifted.

Two immediate consequences, both checkable by hand:

**(i) F1 is strictly INCREASING in prevalence at a fixed ROC point.**
```
∂F1/∂π = 2 r f / D²  >  0        where D = π(r + 1 − f) + f
```
So going from your train prior 0.402 to the test prior ~0.618 should have *raised* your F1 for free. It did
not — F1 fell from a replica 0.958 to a deployment 0.882. **The prevalence shift is a tailwind and you still
lost 0.076.** The ROC degradation is therefore *larger* than the raw F1 drop suggests. This is a cleaner
statement of §4's "OOF is blind" than §4 gives, and it is train-only arithmetic.

**(ii) It lets you convert your replica's F̂\* into an implied AUC, and see the blindness in one number.**
Under an equal-variance binormal ROC (`r = Φ(Φ⁻¹(f) + d')`, `AUC = Φ(d'/√2)`) — **INFERRED, the binormal step
is a model, so treat the magnitude as ±, the direction as solid**:

| where | prevalence π | F* | implied d' | **implied AUC** |
|---|---|---|---|---|
| your masked train-only replica | 0.402 | 0.958 (your δ̂×2) | ≈ 3.6 | **≈ 0.994** |
| deployment (public LB) | 0.618 | ≈ 0.902 achievable / 0.882 achieved | ≈ 2.27 | **0.9458 (measured)** |

Sanity check the second row against your measured confusion cell: at AUC 0.9458 and π = 0.618 the binormal
optimum sits at `f ≈ 0.20, r ≈ 0.924`, giving `F* ≈ 0.902`; your realised point is `f = 17/118 = 0.144,
r = 0.859`, giving 0.882. Consistent.

> **Headline number: your masked replica behaves like an AUC ≈ 0.99 problem. Deployment is an AUC ≈ 0.946
> problem. The replica overstates AUC by ~0.05 and F\* by ~0.11.** Your masked-window augmentation reproduces
> the *missingness*, but it does **not** reproduce the *covariate shift in SAR levels* (§4), and that residual
> is where all the blindness lives. If you build one diagnostic this round, build the one that measures this
> gap directly (see the recommendation section).

### Step 2: is there a legal estimator of F*_deploy? Honest answer: NO, and here is the exact obstruction.

`F*_deploy = max over ROC of F1(π_test, r, f)`, so you need **(A)** the deployment ROC curve and **(B)** the
deployment prevalence π_test. Take them in turn.

**(B) π_test is not estimable under your measured conditions.** The estimable-prevalence family is exactly the
label-shift family — Saerens–Latinne–Decaestecker EM (Neural Computation 14(1):21–41, 2002,
DOI 10.1162/089976602753284446), MLLS, and BBSE (Lipton, Wang & Smola, ICML 2018, arXiv:1802.03916). Every
member assumes `p(x|y)` invariant. **You tested exactly that and rejected it at p ≈ 0** (§4). So π_test is
not identified. Your 0.59–0.62 came from the leaderboard and is illegal as a knob by prong (b). *(This is
consistent with §6 and I am not re-proposing the closed lane — I am noting that its closure is also what
closes Q2(c).)*

**(A) The deployment ROC is not estimable either, and this one is a theorem, not a measurement.**
Your adversarial AUC is **1.0000** — train and test have *disjoint support*. Under disjoint support no
train-only quantity constrains the deployment ROC at all.

**Source: Ben-David, Lu, Luu & Pál, "Impossibility Theorems for Domain Adaptation", AISTATS 2010,
PMLR 9:129–136 (`https://proceedings.mlr.press/v9/david10a.html`).** *(Status: cited from the search index and
its abstract; I did NOT read the full PDF — treat as **PARTIALLY VERIFIED**. The result I rely on is the
paper's headline: covariate-shift + low-error-joint-predictor assumptions are insufficient for domain
adaptation without a bound on the source/target overlap.)* Their setting is precisely yours: with no shared
support you cannot certify anything about target error from source data.

> **Therefore the correct answer to "what is the right estimator for t\* when calibration and deployment differ
> in both prevalence and missingness" is: THERE ISN'T ONE, and the non-existence is forced by your own two
> measurements (adversarial AUC = 1.0, and the p ≈ 0 rejection of `p(x|y)` invariance).** Together they remove
> both inputs the estimator needs. This is a clean, well-sourced negative and it belongs in your report.

**And — per Correction 3 — it does not matter, because the quantity you cannot estimate is worth 4×10⁻⁴ F1.**
That pairing is the most report-worthy thing in Q2: *the one knob you are forbidden to touch and cannot
estimate is also the one knob that is worthless.* Your compliance rule cost you nothing.

### Step 3: the fork that actually matters — World C vs World M

The Q2(c) analysis exposes a genuine fork, and **only one branch has any money in it.**

- **World C — the score is calibrated on test.** Then Corollary 1 binds: flipping any row with `s < F/2 =
  0.441` *strictly reduces* F1, only ~7 rows of 1030 lie in [0.441, 0.5), total available gain ≈ 4×10⁻⁴.
  The gap to the leader is 100% a ranking/model gap. Nothing at the operating point. **Threshold work: worthless.**
- **World M — the score is calibrated globally but MIS-calibrated LOW on a structured subpopulation.** Then
  Corollary 1's premise fails locally, a block of genuine positives sits at 0.15–0.45, and that F1 *is*
  recoverable. But recovering it requires **identifying the subpopulation and correcting `p(y|x)` on it** —
  which is prong (c) exactly, and is *not* a threshold move, because it raises scores for some rows while
  leaving others alone (**rank-changing ⇒ Case B ⇒ not Platt-annihilable**).

**§3's own evidence points to World M**: an F1 shortfall of 0.037–0.048 with an AUC that *beats* the leader is
hard to produce in World C. (In World C, better ranking ⇒ better F* ⇒ better F1 at any sane threshold.)
A model that out-ranks the leader but under-scores on F1 is close to a definition of *locally* miscalibrated.

This is the entire content of Q1, and it is where the remaining budget should go. Q1 follows.

---

