# Round 24 findings — Q3 (mixup / second teacher), Q4 (label noise), Q5 (attack the closures)

Researcher scope: Q3(a) mixup/VRM, Q3(b) model-independent teacher, Q4 label-noise detection +
legality, Q5 challenges to §6 closures.

Status: IN PROGRESS — written incrementally, appended after each sub-finding.
Started 2026-08-14.

Legend: **[VERIFIED]** = I read the paper text (arXiv abs/full text or published HTML).
**[INFERRED]** = reasoned from secondary sources or from the math, not from reading the primary.

---

## Q3(a).1 — THE CRUX: does mixup's population minimizer escape the pointwise-loss theorem?

**Answer: YES, it escapes — and I can show exactly where the escape happens. But the escape is a
double-edged sword and the naive version probably moves the ranking in the WRONG direction for you.**

### The derivation (mine, self-contained; **[VERIFIED]** against the mixup objective as defined in
Zhang et al., arXiv:1710.09412 §2, and consistent with Carratino et al., arXiv:2006.06049)

The mixup risk is

```
R_mix(f) = E_{(xi,yi),(xj,yj) ~ D x D}  E_{λ~Beta(α,α)}  ℓ( f(λ xi + (1-λ) xj),  λ yi + (1-λ) yj )
```

This is **not** of your §5 form `Σ_i [y_i·l1(z_i) + (1-y_i)·l0(z_i)]`, because the argument of `f` is
`x̃ = λ xi + (1-λ) xj`, a point that depends on **two** examples, and the target `ỹ = λ yi + (1-λ) yj`
likewise. Your theorem's hypothesis ("one fixed pair of functions applied to every example, evaluated
at that example's own `z`") is violated. **Confirmed: mixup is genuinely in class E2.**

Now the population minimizer. Push the mixing forward to get the **vicinal input law** `D̃` of `x̃`, and
define the **vicinal regression function**

```
η̃(x̃)  =  E[ λ yi + (1-λ) yj  |  λ xi + (1-λ) xj = x̃ ]
```

For cross-entropy (any strictly proper composite loss) the unconstrained pointwise minimizer over `D̃`
is `f*(x̃) = T(η̃(x̃))` with `T` the link. So:

> **The mixup population minimizer is a monotone transform of `η̃`, NOT of `η`.**

And `η̃` is **not** a monotone transform of `η`, for a concrete structural reason: `η̃(x)` is an average
over the whole *set of (i, j, λ) triples whose interpolant lands at x*, weighted by their density. It
therefore depends on the **local geometry and density of the training set around x**, not only on the
value `η(x)`. Two points with identical `η(x)` receive different `η̃` when they sit in differently
populated neighbourhoods. **That is a genuine reordering, not a threshold slide.** ROC-AUC is not
preserved. ✅ Theorem 2 does not apply.

**Theorem 1 (Platt annihilation) also does not apply**: the map `η → η̃` is not affine in the logit —
it is a density-weighted local average, which is nonlinear and *x*-dependent. Refitting `σ(az+b)`
cannot undo it. ✅

### ⚠️ But now the sign of the effect, which is what actually matters to you

`η̃` is a **local average** of `η` under the mixing kernel. Averaging is a **contraction toward the
local mean**. Your problem is a set of positives scored at median 0.170 — i.e. positives sitting in
regions the model reads as negative-dominated. Under `η̃` those points get averaged with their
(negative-dominated) mixing partners and are pulled **further down**, not up. Vanilla mixup is a
smoother; smoothers hurt isolated minority points.

> **Prediction I will stake: plain input-space mixup on your 1817 rows will move AUC by ~0 to slightly
> negative and will NOT recover the confidently-missed positives.** If you run it and it recovers
> them, my model of the mechanism is wrong and that itself is worth knowing.

The variants that matter are the ones that **break the symmetry of the averaging** — either by mixing
*across the domain gap* (Q3(a).3) or by *not* mixing the labels the same way you mix the inputs
(Q3(a).4). Those are the two I recommend. Plain mixup I recommend against.

### One more consequence you should care about more than the reordering

**[VERIFIED]** Wang et al., *On the Pitfall of Mixup for Uncertainty Calibration*, CVPR 2023
(IEEE DOI 10.1109/CVPR52729.2023.00742; openaccess.thecvf.com CVPR2023 Wang_On_the_Pitfall_of_Mixup...).
Finding: **mixup-trained models are usually LESS *calibratable* than vanilla ERM** — i.e. after you
apply post-hoc calibration (temperature/Platt), mixup ends up *worse* than ERM+post-hoc, because mixup
conflates data uncertainty with model uncertainty and induces underconfidence that a monotone post-hoc
map cannot repair. (Read via search-result extraction of the paper's claims, not the full PDF — CVF
returned 403; grading this **[VERIFIED-abstract-level]**.)

**Why this is decisive for you specifically:** your pipeline ends in a Platt map fit on train OOF and a
literal 0.5 cut. Therefore:

1. **Every "mixup improves calibration" result in the literature is worth exactly zero to you** — a
   monotone 2-parameter map sits downstream and absorbs it (your Theorem 1).
2. The *only* channel by which mixup can help you is **reordering**.
3. And the Wang et al. result says the residual non-monotone part of mixup's distortion tends to be
   *harmful* after post-hoc calibration.

So the literature's headline benefit of mixup is annihilated by your own pipeline, and its known
post-hoc pathology is not. **This is a strong prior against plain mixup here.**

---

## Q3(a).2 — 🔑 THE VERSION OF MIXUP THAT IS ACTUALLY BUILT FOR YOUR PROBLEM: cross-domain mixup as *manufactured overlap*, i.e. **gradual domain adaptation**

This is my headline and it **merges Q3(a) and Q3(b) into one method.**

### Correction to the brief, §4, second bullet — please read this first

> Brief says: *"Train and test are EXACTLY separable (adversarial AUC 1.0000). **Therefore importance
> weighting is not identifiable (no overlap to transport across).**"*

The conclusion (IW is dead) is **right**. The parenthetical reason is **wrong, and the wrong reason is
closing a door that is actually open.**

Importance weighting needs the density ratio `dP_target/dP_source`, which requires **absolute
continuity** — target support ⊆ source support. Adversarial AUC = 1.0 says that fails, so IW dies.
Correct.

But **optimal transport does not require absolute continuity.** The Wasserstein distance and its
geodesics are defined between **mutually singular** measures — that is the entire point of OT and the
reason it is used where f-divergences and density ratios blow up to ∞. "No overlap" is not "nothing to
transport across"; it is *the canonical OT setting*. You have written down a true statement about
importance weighting and then generalized it, in the same sentence, into a false statement about
transport. **[VERIFIED — this is textbook OT; e.g. Villani, and it is the explicit premise of the GDA
papers below.]**

**Consequence:** the family of methods that operates by *building a path between two disjoint supports*
is not merely alive, it is **exactly the family designed for adversarial-AUC-1.0 regimes**.

### The method: Gradual Domain Adaptation (GDA)

**[VERIFIED]** Kumar, Ma & Liang, *Understanding Self-Training for Gradual Domain Adaptation*,
ICML 2020, arXiv:2002.11361. Abstract, verbatim in relevant part: *"We prove the first non-vacuous
upper bound on the error of self-training with gradual shifts, **under settings where directly adapting
to the target domain can result in unbounded error**. The theoretical analysis leads to algorithmic
insights, highlighting that **regularization and label sharpening are essential even when we have
infinite data**, and suggesting that self-training works particularly well for shifts with small
Wasserstein-infinity distance."*

Read that against your situation: you have (i) an adversarial AUC of exactly 1.0 — the "directly
adapting can result in unbounded error" regime, verbatim; (ii) one round of self-distillation that was
your **largest single win (+0.0100)**; (iii) a self-imposed cap of one round because "self-training
compounds error." **Kumar et al. is the theory that says the cap is the right instinct for a
single big jump and the wrong instinct if you first build a path.**

Also note the second clause: *label sharpening is essential*. Your teacher is near-binary (0.5–2.9%
of mass in [0.45,0.55]). Under Kumar et al. that is **not** a defect — sharpening is one of the two
things their theory says you need. See Q3(b).4.

### The construction that makes it usable with no intermediate domains available

**[VERIFIED]** He, Wang, Wang & Zhao, *Gradual Domain Adaptation: Theory and Algorithms* (GOAT),
arXiv:2310.13852 (JMLR 2024). I read the HTML full text. Two things you need:

**(1) The bound (their Thm 1), for neural nets:**
```
ε_T(h_T)  ≤  ε_0(h_0) + O( T·Δ + T/√n + T·√(log(1/δ)/n) + 1/√(nT) + √( (log nT)^(3L-2) / (nT) ) )
```
`T` = number of intermediate domains, `Δ` = average p-Wasserstein distance between **consecutive**
domains, `n` = samples per domain, `L` = depth. The dominant term is `T·Δ` = **total path length**.
This is *linear* in T, an exponential improvement over Kumar et al.'s `e^{O(T)}`.

**(2) The trade-off, which is the part you must respect:** `T·Δ ≈ W(source,target)` is roughly
**fixed** by your data, so more intermediate domains does not reduce the path term — but it *does*
inflate the `T/√n` variance terms. Hence there is an **interior optimum**, which they derive:
```
T* = max{ L_W/Δ_max ,  Õ( (1/(1 + Δ_max·√n))^{2/3} ) }
```
(`L_W` = Wasserstein distance source→target.) Their empirical section confirms the non-monotonicity:
"the more intermediate domains for adaptation, the worse performance" past the optimum.
**Translation for you: T is small. Think T = 2 or 3, not 10.** You currently run T = 1 (one direct
self-distillation jump). **The literature says T = 1 is very likely below the optimum.**

**(3) How GOAT builds the intermediate domains — this is the mixup connection (their Eq. 29):**
```
μ_t  =  (1/m) Σ_{i,j} γ*_ij · δ( ((T−t)/T)·x_i^source  +  (t/T)·x_j^target )
```
where `γ*` is the **optimal-transport plan** between the source and target empirical measures (an LP).
So: **an intermediate domain is exactly mixup between source and target rows — but with the mixing
partner chosen by the OT plan rather than uniformly at random, and with λ fixed to t/T rather than
drawn from Beta.** That is why this is the right mixup variant for you: the OT coupling is what stops
the averaging from being a symmetric smoother (the failure mode in Q3(a).1) and makes it a *directed*
move along the geodesic toward the deployment distribution.

**(4) Their numbers (their tables), gradual self-training with 2 given intermediates vs GOAT:**

| dataset | GST | GOAT | Δ |
|---|---|---|---|
| Rotated MNIST | 61.6 | 70.3 | **+8.7 pp** |
| **Color-Shift MNIST** | 67.6 | **90.3** | **+22.7 pp** |
| Portraits | 77.0 | 79.9 | +2.9 |
| Cover Type | 66.9 | 69.8 | +2.9 |

Color-Shift MNIST is the closest analogue to your setting (a pure, near-separable covariate shift in
the input channel statistics) and it is where the method pays the most. Cover Type is the closest
analogue in *modality* (small tabular) and pays +2.9 pp.

### Check against your two theorems and three prongs

- **Theorem 2 (pointwise-loss order invariance): ESCAPES, twice over.** The training objective is
  evaluated at interpolated points `x̃` that depend on two examples (E2), against targets that are
  model-produced soft labels on target rows (E3). It is neither pointwise nor a fixed target.
- **Theorem 1 (Platt annihilation): ESCAPES.** The induced change in the score function is a
  geometry-dependent, x-varying reweighting, not `z → αz+β`.
- **Prong (a) literal 0.5 on a genuine probability: PASSES** — nothing about GDA touches the decision
  rule; you still Platt on train OOF and cut at 0.5.
- **Prong (b) every knob train-only: PASSES with one thing to be careful about.** The only new knob is
  `T`. Set it by the GOAT formula from the *measured* Wasserstein distance between train and test
  **features** — no labels, no LB. Pre-register the value before submitting. Do **not** pick T by
  trying 2 and 3 on the leaderboard; that is LB feedback and it would be your fifth error.
- **Prong (c) corrects p(y|x) under a demonstrably mis-specified model: PASSES, and this is the
  cleanest prong-(c) story in your whole method list** — your §4 mixture goodness-of-fit test rejected
  `p(x|y)`-invariance at p≈0, i.e. you have *proved* the conditional is shifted. GDA is a method for
  moving the conditional along a path, not a relabelling of a fixed estimate.

### Honest risks

- Your "domains" are not naturally ordered in time; you are *synthesizing* the path. That is exactly
  GOAT's contribution, so it is supported, but it is one more layer of construction.
- `n` per intermediate domain is small (1817 / 1030). The `T/√n` variance term bites early. Another
  reason T ∈ {2,3}.
- OT plan on 1817×1030 is a trivially small LP — compute cost is not an issue.

