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


---

# Q3(b) â€” A SECOND, MODEL-INDEPENDENT TEACHER

## Q3(b).0 â€” ðŸ”‘ READ THIS BEFORE THE REST: your biggest win does not work by the mechanism you think it does

**This is a correction to Â§2 item 6 of the brief and it reframes the whole of Q3(b). [DERIVED],
checkable in three lines.**

Your transductive soft self-distillation adds, for each test row `x`, the term

```
L_distill(x) = âˆ’ q(x)Â·log Ïƒ(z_s(x))  âˆ’  (1âˆ’q(x))Â·log(1 âˆ’ Ïƒ(z_s(x)))
```

where `q(x)` is the teacher's probability and `z_s` the student logit. The gradient is
`âˆ‚L/âˆ‚z_s = Ïƒ(z_s) âˆ’ q`, so the **unconstrained population minimizer on the test rows is exactly**

```
Ïƒ(z_s*(x)) = q(x)      i.e.      z_s*(x) = z_t(x)
```

> **An unconstrained student reproduces the teacher's ranking exactly. AUC on the test rows would be
> IDENTICAL to the teacher's, and F1 would differ only by a threshold slide that your Platt layer then
> undoes.** By your own Theorem 1 and Theorem 2, *idealised single-teacher soft distillation is
> annihilated.*

So the observed **+0.0100 did not come from the target-matching channel.** It can only have come from
the three channels where the idealisation fails:

1. **Capacity / regularization (Mobahi channel).** The 71k-parameter student cannot interpolate `q`,
   so it returns a smoothed version of it â€” which *is* a reordering. **[VERIFIED]** Mobahi, Farajtabar
   & Bartlett, *Self-Distillation Amplifies Regularization in Hilbert Space*, NeurIPS 2020,
   arXiv:2002.05715. Abstract, verbatim: *"Self-distillation iterations modify regularization by
   progressively limiting the number of basis functions that can be used to represent the solution.
   This implies (as we also verify empirically) that **while a few rounds of self-distillation may
   reduce over-fitting, further rounds may lead to under-fitting** and thus worse performance."* Their
   Â§3.2 proves the iteration cannot continue indefinitely â€” *"at some point the solution collapses"* â€”
   and Prop. 4 gives a lower bound `tÌ„` on the number of *non-trivial* rounds.
   **This is the exact theory for your "capped at one round because self-training compounds error."
   It says the cap is right in the limit but that the optimum is `t > 1`, not `t = 1`** â€” the same
   conclusion the GOAT analysis in Q3(a).2 reached from the transport side. Two independent theories
   now say **T = 1 is below your optimum.**
2. **Augmentation under distillation.** If your masked-window views are applied to the *test* rows
   during distillation, the student must reproduce `q` from 4â€“6-month crops. That is a genuinely new
   constraint, it is not implied by `z_s = z_t`, and it is *exactly the deployment condition*. I
   suspect this is where most of the +0.0100 actually lives.
3. **Joint training with the labelled rows.** The test-row targets act as an anchor that forces the
   shared representation to be well-behaved on target-domain inputs.

### The consequence for Q3(b), stated as a requirement

Because a single teacher's contribution through channel (1) is a *self*-smoothing of its own errors,
adding a second teacher only helps if it supplies information channel (1) cannot manufacture:

> **A second teacher is worth having if and only if its score function is NOT a monotone transform of
> the first teacher's score.** If it is monotone in `z_t`, the combined target is monotone in `z_t`,
> the population minimizer is monotone in `z_t`, and Theorem 1 annihilates the whole thing.
> Conversely, **the pointwise average of two non-concordant probability functions is not a monotone
> function of either** â€” that arithmetic fact is the entire theoretical case for a second teacher, and
> it is stated in the vocabulary of your own theorem. [DERIVED]

**Falsifiable test that costs you nothing and needs no leaderboard:** compute Spearman Ï between
teacher-1 and candidate-teacher-2 scores on the 1030 test rows. If Ï > ~0.98 the candidate is a
monotone re-expression of what you already have and **must not be shipped** â€” this is the same
diagnostic that correctly predicted JTT's null (Ï 0.984, zero TPs gained). Note the sign: for JTT high
Ï predicted failure; here high Ï *also* predicts failure, and low Ï is necessary but **not** sufficient
(a decorrelated *worse* teacher is your "blending a weaker decorrelated member" lane, lost 3Ã—). What
you need is **low Ï at comparable quality.**

---

## Q3(b).1 â€” Co-training on the SAR/optical split: does Blum & Mitchell's precondition hold?

### What the precondition actually is

**[VERIFIED]** Blum & Mitchell, *Combining Labeled and Unlabeled Data with Co-Training*, COLT 1998,
DOI 10.1145/279943.279962. Two assumptions, and people routinely forget the first:

- **(S) Sufficiency / compatibility:** the instance space splits as `x = (x1, x2)` and **each view alone
  is sufficient for correct classification** â€” there exist `f1, f2` with `f1(x1) = f2(x2) = y` for
  (almost) all `x`.
- **(I) Conditional independence:** `x1 âŠ¥ x2 | y`.

Under (S)+(I) they prove the strong result: a *weakly-useful predictor* (barely better than chance) plus
unlabeled data suffices to learn the target. **(I) is the load-bearing one, and it is also the one
essentially nothing in the real world satisfies.**

### Judging (S) for your two views

- **SAR view (VH, VV Ã— 12 months = 24 values).** Sufficient on the literature's own account:
  Ottinger et al., IGARSS 2018, DOI 10.1109/IGARSS.2018.8651419 build an aquaculture-pond detector from
  **VH temporal statistics alone** â€” this is the paper the brief Â§8.2 already cites against itself. Your
  own best hand-feature (`1[VH_dB < âˆ’21]`, +0.010) lives entirely in this view. **(S) plausibly holds.**
- **Optical view (10 bands Ã— 12 = 120 values).** Water/pond discrimination from Sentinel-2 spectra is a
  standard task (NDWI/MNDWI family). **(S) plausibly holds.**

**So the unusual thing about your problem is that (S) is genuinely satisfied on both sides.** That is
rare and it is the strongest part of the case. Most real "views" fail (S), not (I).

### Judging (I) â€” and here is the honest answer

**(I) is violated, and I do not think it is close.** Both sensors observe the *same physical scene at
the same time*. The nuisance variables are shared:

| shared nuisance | effect on SAR | effect on optical | correlated given y? |
|---|---|---|---|
| water level / pond drainage & harvest cycle | VH backscatter â†‘ when drained | reflectance â†‘, NDWI â†“ | **strongly** |
| wind roughening the surface | VH â†‘ | sun-glint / texture change | yes |
| vegetation encroachment on the pond | VH â†‘ | NIR â†‘, red â†“ | **strongly** |
| season / date | both | both | yes |
| cloud, haze, aerosol | **none** | destroys the optical row | **no â€” this one is independent** |
| SAR speckle, incidence angle | yes | none | **no â€” independent** |

So the dependence structure is: **a large shared component (the physical state of the pond, which is
exactly what `y` fails to screen off because `y` is a *static* label and the pond's state is
*time-varying*) plus two genuinely sensor-private noise components (cloud, speckle).** The team's
intuition â€” "radar and optical measure genuinely different physics" â€” is true about the *measurement
operator* but that is not what (I) asks. **(I) asks about the residual dependence after conditioning on
the label, and a time-varying scene state that the static label does not determine is precisely a
conditioning failure.**

**Verdict on the literal precondition: NOT satisfied. Do not claim it in the report.**

### But the literal precondition is not the operative one

**[VERIFIED â€” via the NeurIPS 2004 paper's own framing]** Balcan, Blum & Yang, *Co-Training and
Expansion: Towards Bridging Theory and Practice*, NIPS 2004
(proceedings.neurips.cc/paper_files/paper/2004/file/9457fc28ceb408103e13533e4a5b6bd1-Paper.pdf).
They replace (I) with **Îµ-expansion**: informally, for the graph on unlabeled points induced by the two
views' confident regions, every subset `S` must leak at least `ÎµÂ·min(|S|,|SÌ„|)` mass to its complement.
Expansion is a *conductance* condition â€” it says confident labels can propagate â€” and they argue it is
in a sense **necessary**, and that it is what motivates co-training's *iterative* structure (whereas
(I) is so strong it makes a *one-shot* co-train sufficient).
âš ï¸ Grading caveat: my fetch of this PDF returned a paraphrase with reconstructed quotation marks, so I
grade the *definitions* **[INFERRED]** and only the existence/direction of the result **[VERIFIED]**.
Do not put the quoted strings in the report; cite the result, not my quotes.

**[VERIFIED â€” search-level]** Wang & Zhou, *Analyzing Co-Training Style Algorithms*, ECML 2007
(lamda.nju.edu.cn/publication/ecml07a.pdf) go further: **two views are not required at all; what is
required is that the two learners have large diversity.** This is the theoretical licence for
single-view co-training, and it collapses co-training and the Ï-diagnostic in Q3(b).0 into the same
condition.

### â›” The finding that should make you cautious, and it is directly on your sample size

**[VERIFIED â€” abstract/search level; both PDF fetches returned undecodable binary, so INFERRED on the
numeric thresholds]** Du, Ling & Zhou, *When Does Co-Training Work in Real Data?*, PAKDD 2009 /
IEEE TKDE 2011 (lamda.nju.edu.cn/publication/tkde10.pdf). They build empirical tests for (S) and (I)
and then report the sting:

> *"Given a whole or a large labeled training set, their view verification and splitting methods are
> quite effective. **Unfortunately, co-training is called for precisely when the labeled training set is
> small. However, given small labeled training sets, the two co-training assumptions are difficult to
> verify, and view splitting is unreliable.**"*

This is a **paper-shaped statement of your own core problem** (Â§4: "OOF validation is blind here"). You
cannot verify the co-training preconditions on 1817 rows, and even if you could, your OOF estimate of
whether it helped is worthless. **Co-training would be a method you ship blind.** Given a 5-submission
budget on the deadline day, that is a bad trade.

### Verdict on Q3(b) co-training

| criterion | verdict |
|---|---|
| Sufficiency (S) on both views | **plausibly YES** â€” unusually good, cite Ottinger for the SAR side |
| Conditional independence (I) | **NO** â€” shared time-varying scene state is not screened off by a static label |
| Weaker operative condition (expansion / diversity) | **plausibly yes**, but unverifiable at n=1817 |
| Escapes Theorem 2? | **YES** â€” targets are model-produced and x-dependent (E3), and the *other* view's model produces them, so the objective is not pointwise in one fixed pair of functions |
| Escapes Theorem 1? | **YES iff** the two views' scores are non-concordant (Ï well below 1); this is checkable for free |
| Prong (a) 0.5 threshold | passes, untouched |
| Prong (b) train-only knobs | âš ï¸ **the confidence threshold for accepting a pseudo-label is a new knob and it is dangerously close to a positive-rate target.** It must be pre-registered from train OOF, never adjusted |
| Prong (c) | passes â€” it corrects `p(y|x)` using target-domain inputs |
| **Shippable today?** | **NO.** Two full trainings, an unverifiable precondition, and blind validation |
| **Report-grade?** | **YES, high.** "We identified a genuine two-view structure, checked Blum & Mitchell's actual precondition rather than the folklore version, found (S) holds and (I) fails, identified the weaker expansion condition, and found the Duâ€“Lingâ€“Zhou result showing the preconditions are unverifiable at our sample size" is a strong methodology paragraph. |

---

## Q3(b).2 â€” Tri-training and asymmetric tri-training

**[VERIFIED â€” abstract]** Saito, Ushiku & Harada, *Asymmetric Tri-training for Unsupervised Domain
Adaptation*, ICML 2017, arXiv:1702.08400, PMLR v70 pp. 2988â€“2997. Abstract, verbatim in relevant part:
*"Tri-training leverages three classifiers equally to give pseudo-labels to unlabeled samples, but the
method does not assume labeling samples generated from a different domain. In this paper, we propose an
asymmetric tri-training method for unsupervised domain adaptationâ€¦ **By asymmetric, we mean that two
networks are used to label unlabeled target samples and one network is trained by the samples to obtain
target-discriminative representations.**"*

**Why this is a better structural fit than plain tri-training for you:** the asymmetry is *exactly* the
source/target asymmetry you have. Two labelers trained on the labelled 1817 rows agree-and-label the
1030 target rows; a third net is trained **only on the pseudo-labelled target rows** and is therefore
free to become target-specific rather than a compromise between two domains. Given that your adversarial
AUC is 1.0 â€” i.e. the domains do not overlap at all â€” a model that is allowed to be *purely* target-side
is more sensible than one forced to serve both.

**The catch, and it is decisive for today:** the labelling rule is *agreement of the two labelers*, and
agreement-based selection is a **high-precision, low-recall** filter. Your problem is a **recall**
problem in the high-recall corner of the ROC (Â§3.2: you need 16 more TPs). Agreement filtering will
confidently pseudo-label exactly the rows you already get right and will stay silent on the ~27 FNs.
**[DERIVED]** Structurally, agreement-based tri-training pushes you *deeper into the precision corner*,
which is the wrong direction on your metric. I rank it **below** co-training and well below Q3(b).3.

Also: three networks Ã— the seed pool = 3Ã— training cost, on deadline day, with blind validation.
**Not shippable. Mention in the report as considered-and-reasoned-against, with the recall argument â€”
that argument is worth marks precisely because it is metric-aware.**

---

## Q3(b).3 â€” ðŸ† The one I would actually pick: a prototype/centroid teacher built from target data only (SHOT)

**[VERIFIED â€” abstract + method summary]** Liang, Hu & Feng, *Do We Really Need to Access the Source
Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (SHOT)*, ICML 2020,
arXiv:2002.08546. SHOT **freezes the source classifier head** and adapts only the feature extractor on
target data, using (i) an **information-maximization** objective and (ii) **self-supervised pseudo-labels
obtained by weighted k-means clustering / a nearest-centroid classifier in the target feature space.**

### Why the centroid teacher is the right second teacher for *your* pathology

The centroid pseudo-label of a target row is
```
Å·(x) = argmin_k  d( g(x),  c_k )        c_k = Î£_x Î´_k(x)Â·g(x) / Î£_x Î´_k(x)
```
â€” a **distance to a target-domain centroid**, not a value of the classifier head. Three properties, all
of which you need. **[DERIVED]**

1. **It is not a monotone transform of `z_t`.** The head's logit and the distance-to-centroid ratio are
   different functionals of the same embedding. Spearman Ï between them will not be ~1. **Theorem 1 does
   not annihilate it. Theorem 2 does not apply â€” the target depends on the whole target set through the
   centroids, so the objective is not pointwise.** This is the cleanest theorem-escape of any candidate
   in Q3(b): the centroids are *global statistics of the unlabeled test set*, so `L` is not of the form
   `Î£áµ¢ [yáµ¢ lâ‚(záµ¢) + (1âˆ’yáµ¢) lâ‚€(záµ¢)]` even in principle.
2. **The centroids are computed on the 1030 test rows, whose positive rate is ~59â€“62%.** A centroid
   teacher's decision surface therefore sits where the *target* classes actually separate, not where the
   40.23%-prior source classes separate. Your 27 false negatives at median score 0.170 are, on the
   centroid view, *not* necessarily far from the positive centroid â€” they are far from the *source
   head's* boundary. **This is the only mechanism in the whole of Q3 that can plausibly move a row from
   0.17 to above 0.5 without touching the threshold.** It is also why it targets the high-recall corner
   specifically (Q1's target).
3. âš ï¸ **And this is also why you must be careful about prong (b).** A 2-means clustering of the target
   set will split it near the target prior. That is *dangerously adjacent* to prevalence matching â€” the
   thing you deleted scores over. **The distinction that makes it legal:** k-means places the boundary
   where the target feature *density* is thin; it does **not** take a positive-rate number as input and
   does not solve for one. Prevalence matching sets `#positives = pÂ·n` by construction; a centroid
   teacher's resulting positive rate is an *output*, unconstrained and free to come out at 0.45 or 0.75.
   **You must say this explicitly in the report, and you must NOT check the resulting rate against 0.60
   and iterate â€” checking it once and re-running is how a legal method becomes an illegal one.**
   Pre-register: "we fit 2-means on the target embedding, we distil from the resulting soft
   memberships, we do not inspect the induced positive rate before submitting."

### Legality summary
- **(a)** untouched â€” still Platt-on-train-OOF then a literal 0.5.
- **(b)** knobs are: k (=2, forced by the problem), the distance metric (cosine, SHOT's default,
  pre-registered), and the mixing weight between the head teacher and the centroid teacher. **Set the
  mixing weight to 0.5 a priori, or fix it by cross-fitted OOF NLL on train rows only.** Do not tune it.
- **(c)** passes strongly: the source head is *demonstrably* mis-specified on the target (your Â§4
  mixture GoF test rejects `p(x|y)` invariance at pâ‰ˆ0), and SHOT is literally the method for "the source
  hypothesis is wrong on the target, keep the representation and re-derive the labels from target
  structure."

**Cost:** one k-means on a 1030Ã—d embedding you already compute, plus one distillation run you already
have code for. **This is the only Q3(b) candidate that is plausibly shippable inside a deadline day.**

---

## Q3(b).4 â€” âš ï¸ THE SATURATED TEACHER: your diagnosis is right and the standard remedy is a trap

You report only **0.5â€“2.9% of the teacher's mass in [0.45, 0.55]**. Three things follow.

### (i) The usual citation does NOT transfer to binary â€” do not use it

The famous result is **[VERIFIED]** MÃ¼ller, Kornblith & Hinton, *When Does Label Smoothing Help?*,
NeurIPS 2019, arXiv:1906.02629: a teacher trained with label smoothing distils *worse*, because
smoothing *"encourages the representations of training examples from the same class to group in tight
clusters, resulting in loss of information in the logits about resemblances between instances of
different classes."*

**That mechanism does not exist in binary classification. [DERIVED]** "Dark knowledge" in MÃ¼ller's sense
is the *relative* structure among the `Kâˆ’1` wrong-class logits; with `K = 2` there is exactly one degree
of freedom and no relative structure to destroy. **If you cite MÃ¼ller here you will be wrong and a
reviewer who knows the paper will notice.** (The same caveat applies to Yuan et al., *Revisiting
Knowledge Distillation via Label Smoothing Regularization*, CVPR 2020, arXiv:1909.11723 â€” also a
multi-class-logit-structure argument.)

### (ii) What saturation actually destroys in binary â€” and it is worse for you than the multiclass case

In binary, the *only* information a soft target carries beyond a hard label is **the variation of the
scalar `q` across `x`** â€” i.e. the teacher's *ranking*. So:

> **A saturated binary teacher transmits a hard pseudo-label plus, effectively, nothing about the
> ordering of the rows inside each confident mass.** [DERIVED]

Now put that next to Â§3.4. Your missing positives sit at median 0.170 with ten below 0.10 â€” i.e. **deep
inside the confidently-negative mass, which is exactly the region a saturated teacher has flattened.**
The teacher is silent precisely where you need it to speak. **This is, I think, the sharpest single
diagnosis available for why one round of self-distillation gave +0.0100 and then stopped paying.**

### (iii) The standard remedy is temperature â€” and â›” your Theorem 1 kills it. Show this in the report.

The textbook fix (Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network*,
arXiv:1503.02531) is to soften the teacher: `q_T = Ïƒ(z_t/T)` with `T > 1`. Check it against your own
theorem. **[DERIVED]**

```
student CE against q_T  â‡’  population minimizer  Ïƒ(z_s*) = q_T = Ïƒ(z_t/T)  â‡’  z_s* = z_t / T
```

`z_s* = z_t/T` is an **affine** map of the teacher logit (Î± = 1/T, Î² = 0). Your finishing Platt map
`Ïƒ(aÂ·z+b)` refits `a â†’ aT` and recovers the identical function. **Temperature is Platt-annihilated,
exactly.** It is not a null in the loose sense â€” it is a null *by the same two-line argument you already
use to kill focal loss*, and the fact that your own theorem disposes of the field's standard remedy is a
genuinely good report paragraph.

Temperature can still act through the **finite-capacity** channel (it rescales the per-example gradient
`Ïƒ(z_s) âˆ’ q_T`, changing which rows the student spends its limited capacity on). But that is a
second-order training-dynamics effect with no guarantee of sign, it is a new tunable knob, and you have
no validation instrument that can measure it. **Recommendation: do not spend a submission on
temperature.**

### (iv) The tension you should resolve explicitly, because two good papers disagree

- Kumar, Ma & Liang (arXiv:2002.11361, Q3(a).2) say **label sharpening is essential** to gradual
  self-training, *"even when we have infinite data."*
- Mobahi et al. (arXiv:2002.05715) say the benefit of self-distillation *is* the soft, over-regularizing
  smoothing, and that too much of it under-fits.

**They are not actually in conflict, and the resolution tells you what to do. [DERIVED]** Sharpening
controls the *bias* of the pseudo-label (a confident wrong label is catastrophic; a 0.5 label teaches
nothing); smoothing controls the *variance* of the student. Kumar et al.'s sharpening operates
**per-step along a path**, where each step is short and the teacher is nearly right; Mobahi's smoothing
accumulates **across rounds**. So the joint prescription is:
**several short, sharply-labelled steps â€” not one long soft one, and not many.** That is `T âˆˆ {2,3}`
intermediate domains with a sharp teacher, which is precisely the GOAT prescription in Q3(a).2, and it
means **your near-binary teacher is an asset for that design and a liability for the design you
currently run.** Your saturation is not a bug to fix; it is a signal that you are running the wrong
number of steps.

### (v) The remedy that IS legal and IS supported: mixup as the anti-confirmation-bias regularizer

**[VERIFIED]** Arazo, Ortego, Albert, O'Connor & McGuinness, *Pseudo-Labeling and Confirmation Bias in
Deep Semi-Supervised Learning*, IJCNN 2020, arXiv:1908.02983. Abstract, verbatim: *"Weâ€¦ propose to learn
from unlabeled data by **generating soft pseudo-labels using the network predictions**. We show that a
naive pseudo-labeling overfits to incorrect pseudo-labels due to the so-called **confirmation bias** and
demonstrate that **mixup augmentation and setting a minimum number of labeled samples per mini-batch are
effective regularization techniques for reducing it.**"*

This paper is worth more to you than any other single citation in Q3(b), for four reasons:
1. It is **exactly your method** â€” soft pseudo-labels from the network's own predictions.
2. It names **exactly your stated reason for capping at one round** â€” confirmation bias â€” and treats it
   as a thing to be *regularized away*, not a hard ceiling.
3. Its remedy is **mixup**, which is Q3(a) â€” so Q3(a) and Q3(b) are not two options, they are one
   method: *mixup is the thing that lets you take round 2.*
4. The second remedy â€” **a minimum number of labelled rows per mini-batch** â€” is a free, zero-knob,
   zero-risk change that directly guards against the student drifting onto its own pseudo-labels. If you
   ship nothing else from this section, ship that: it costs one line, it has no tunable parameter beyond
   "at least some", and it is a pure train-only construction.

**Theorem check on the Arazo recipe:** mixup escapes Theorem 2 (Q3(a).1, derived); soft pseudo-labels are
class E3; the minimum-labelled-per-batch rule changes the *sampling distribution* of the empirical risk,
which is not a reweighting of a pointwise loss over a fixed dataset in the sense Theorem 2 requires
â€” but note âš ï¸ it *is* close: per-batch class-balanced sampling is asymptotically equivalent to class
weighting, which Theorem 2 **does** kill. The version that survives is "â‰¥ m **labelled** rows per batch"
(mixing labelled and pseudo-labelled *sources*), **not** "balance positives and negatives per batch."
Keep that distinction; it is the difference between a legal regularizer and a re-derivation of class
weighting.

---

## Q3(b).5 â€” Ranked verdict for Q3(b)

| rank | candidate | reorders? (escapes Thm 1&2) | shippable today | report value |
|---|---|---|---|---|
| **1** | **Centroid/prototype teacher on target only (SHOT-style), averaged with the existing head teacher** | **yes, strongly** â€” target is a global statistic of the test set | **yes** â€” one k-means + one distillation run | high |
| **2** | **Arazo's two regularizers (mixup on the pseudo-labelled rows + â‰¥m labelled rows per batch), enabling round 2** | yes | **yes** â€” the min-labelled-per-batch part is one line | high |
| 3 | Co-training on the SAR/optical split | yes, if Ï is low | no â€” 2 trainings, unverifiable precondition | **very high as a written negative** |
| 4 | Asymmetric tri-training | yes | no â€” 3 trainings | medium; the recall argument is the valuable part |
| âœ— | Raising the teacher temperature | **NO â€” Platt-annihilated, proven above** | n/a | high as a written negative |

**One-line summary of Q3(b):** the second teacher must be *non-concordant*, not *better*; the only
non-concordant teacher you can build before the deadline is a target-side centroid teacher; and your
teacher's saturation is not a defect to be temperature-fixed (that is provably annihilated) but evidence
that you are running one long step where the theory calls for two or three short ones.

---

# Q4 â€” IS THE LABEL NOISE REAL, AND IS CLEANING IT LEGAL?

**Short answer, up front.** (1) Yes, noise is almost certainly present at the few-percent level, and I
can give you *domain-specific* reasons rather than generic ones. (2) One of your own experiments â€” the
iter49 JTT null â€” is **evidence for the noise hypothesis** and you have not drawn that inference. (3)
Cleaning is **legal**, under a protocol I set out in Q4.4, and the protocol is exactly your own Â§8.4
control-baseline rule applied to a new instrument. (4) âš ï¸ **But the expected effect size is below your
0.015 bar and the downside risk is asymmetric against you.** My recommendation is to run it as a
*diagnostic and a report section*, not as a deadline-day submission.

---

## Q4.1 â€” Is the noise real? Four independent lines of evidence

### (a) The base rate

**[VERIFIED â€” abstract + reported figure]** Northcutt, Athalye & Mueller, *Pervasive Label Errors in
Test Sets Destabilize Machine Learning Benchmarks*, NeurIPS 2021 Datasets & Benchmarks,
arXiv:2103.14749. They found label errors in the test sets of **10 of the most commonly used CV/NLP/
audio benchmarks**, at an **average of 3.4%**, with **6% in the ImageNet validation set** â€” and these
are the *curated, heavily audited* datasets of the field. Flags were found by confident learning and
then **human-verified by crowdsourcing**, so the figure is not a model artifact.

**Calibration for you: 3.4% of 1817 = ~62 rows.** That is the prior you should carry into this section.
A Zindi/FAO/ITU competition dataset assembled at speed has no reason to be *cleaner* than ImageNet.

### (b) ðŸ”‘ The domain-specific argument, which is much stronger than the base rate

The remote-sensing literature on this exact task says the label is intrinsically ambiguous, and names
the confusers.

**[VERIFIED â€” search-level across four papers]**
- Ottinger, Clauss & Kuenzer, *Large-Scale Assessment of Coastal Aquaculture Ponds with Sentinel-1 Time
  Series Data*, Remote Sensing 9(5):440, 2017, DOI 10.3390/rs9050440.
- *Nation-Scale Mapping of Coastal Aquaculture Ponds with Sentinel-1 SAR Data Using Google Earth
  Engine*, Remote Sensing 12(18):3086, 2020, DOI 10.3390/rs12183086.
- *Improving Satellite Retrieval of Coastal Aquaculture Pond by Adding Water Quality Parameters*,
  Remote Sensing 14(14):3306, 2022, DOI 10.3390/rs14143306.
- *Automated extraction of aquaculture ponds from Sentinel-2 seasonal imagery â€” a validated case study
  in central Thailand*, 2022, DOI 10.1016/j.ophoto.2022.100017.

The consistent findings:

> *"Water bodies with similar morphology (e.g., saltworks, rice fields, and small reservoirs) â€¦ are
> difficult to distinguish from aquaculture ponds [and] cause a lot of omission/commission errors in
> areas with complex land-use types."*
> *"Most existing studies may fail to distinguish aquaculture ponds from farmland reservoirs or
> isolated ponds due to their similar spectral and geometrical features."*
> And the 2022 water-quality paper reports that distinguishing ponds from **salt pans, rice fields and
> wetland parks** reaches **F1 > 85%** *only once extra water-quality parameters are added.*

Read that last number carefully. **The published state of the art at separating aquaculture ponds from
their near-neighbours is F1 â‰ˆ 0.85, using information you do not have.** Your model is at F1 0.882 and
the leader at ~0.918â€“0.930. **You are already operating at or above the published accuracy of the
process that most plausibly generated your labels.** That is the single most important sentence in this
section: at that point, further gains are competing with the annotation process itself, and residual
label noise is not a hypothesis â€” it is close to a necessity.

### (c) The temporal mechanism, which ties back to your own Â§6

The same literature says the *discriminating* signal between ponds and their confusers is the
**seasonality and timing of flooding and draining** â€” *"these features have different temporal patterns,
as the seasonality and timing of flooding and draining differs for each type of farming."* Two
consequences:
1. It independently confirms your Â§6 closing note that the reachable-but-unlearned quantity is a
   **nonlinear temporal statistic**, not a band ratio.
2. âš ï¸ **A 4â€“6 month contiguous window can miss the drain event entirely.** So for a genuine pond whose
   drain-fill cycle falls outside the observed window, the *information required to distinguish it from
   a reservoir is absent from the test row.* That is a **third** hypothesis alongside "wrong label" and
   "bad model": **irreducible censoring by the window.** It is the one that best explains why the
   missing positives sit at median 0.170 rather than near the boundary â€” a censored row does not look
   *marginally* like a pond, it looks like a reservoir.

### (d) ðŸ”‘ Your own iter49 JTT null is evidence, and you have not used it

JTT upweights the rows the model gets wrong. Your log: *"JTT provably reorders (Ï 0.984) and still
fails on its own target set â€” recovered ZERO true positives."* **[DERIVED]** Set out the hypotheses:

| hypothesis about the FN set | JTT's predicted effect | observed |
|---|---|---|
| H1: hard but learnable, model under-weights them | upweighting recovers some | **âœ— recovered 0** |
| H2: labels are wrong | upweighting teaches the model a false pattern; no gain, possibly harm | âœ“ consistent |
| H3: information absent from the features (window censoring, no separating signal) | upweighting cannot help; no gain | âœ“ consistent |

> **JTT's null is a clean refutation of H1 and leaves {H2, H3}.** That is a real narrowing and it should
> be in the report. It is also the honest framing: your failed experiment was informative, it just was
> not informative in the direction you were hoping.

âš ï¸ And note what {H2, H3} jointly imply: **both are unfixable by any change to the loss or the
architecture.** H2 is fixed only by touching the data; H3 is fixed by nothing at all. If the split is
mostly H3, the correct conclusion for the report is that the remaining gap is *partly irreducible*, and
you can now say that with an argument instead of an assertion â€” which is what Â§3.3 retracted.

---

## Q4.2 â€” Which detector is sound at n = 1817? A method-by-method verdict

### Confident learning / Cleanlab â€” **the best fit, with one real caveat**

**[VERIFIED]** Northcutt, Jiang & Chuang, *Confident Learning: Estimating Uncertainty in Dataset
Labels*, JAIR 70:1373â€“1411, 2021, arXiv:1911.00068. Abstract, verbatim in relevant part: *"Confident
learning (CL) â€¦ focuses instead on label quality by characterizing and identifying label errors in
datasets, based on the principles of **pruning noisy data, counting with probabilistic thresholds to
estimate noise, and ranking examples to train with confidence** â€¦ building on **the assumption of a
class-conditional noise process** to directly estimate the joint distribution between noisy (given)
labels and uncorrupted (unknown) labels â€¦ **provably consistent** â€¦ We present sufficient conditions
where CL exactly finds label errors."*

**Why it is sound at n = 1817. [DERIVED]** CL's estimand is the joint distribution `Q[á»¹, y*]`, which in
the binary case is a **2Ã—2 table â€” three free parameters.** Estimating three parameters from 1817 rows
is not a small-sample problem in any meaningful sense. Compare this to isotonic calibration, which you
correctly closed as overfitting at n=1817: isotonic fits `O(n)` parameters, CL fits 3. **The
small-sample objection that killed isotonic does not transfer to CL, and you should say so explicitly
rather than letting the reader assume "n=1817 â‡’ nothing statistical works here."**

**Two mechanics you must get right:**
- CL's per-class thresholds are the **average self-confidence of each class** â€” `t_j = mean{ pÌ‚(y=j|x) :
  á»¹ = j }` â€” not a global 0.5. âš ï¸ This is a threshold, and a reviewer will ask. **It is not a decision
  threshold**: it is a *noise-estimation* statistic computed entirely on training rows with training
  labels, and it never touches the deployed decision rule, which stays a literal 0.5. State the
  distinction; it is clean and it holds.
- CL requires **out-of-sample (cross-validated) predicted probabilities.** In-sample probabilities make
  the whole procedure circular by construction. **This requirement is the single most important
  anti-circularity guard in the method and it is built into the algorithm, not bolted on.** You already
  produce train OOF predictions, so you have the input for free.

**âš ï¸ The caveat that matters here: the class-conditional noise (CCN) assumption is violated in your
setting.** CCN says the flip probability depends only on the true class, not on `x`. But Q4.1(b) says
the flips concentrate on *reservoirs, rice paddies and salt pans* â€” i.e. on a **specific region of
feature space**. That is **instance-dependent noise**, and CL's exactness theorem does not cover it. CL
will still *rank* suspicious rows usefully; its *estimate of the noise rate* is not trustworthy. Use CL
as a ranker, not as an oracle, and say so.

### Area Under the Margin â€” **the statistic is fine, the calibration procedure is not usable here**

**[VERIFIED â€” ar5iv full text]** Pleiss, Zhang, Elenberg & Weinberger, *Identifying Mislabeled Data
using the Area Under the Margin Ranking*, NeurIPS 2020, arXiv:2001.10528.
```
M^(t)(x, y) = z_y^(t)(x) âˆ’ max_{iâ‰ y} z_i^(t)(x)          AUM(x,y) = (1/T) Î£_t M^(t)(x,y)
```
The abstract: *"A simple procedure â€” **adding an extra class populated with purposefully mislabeled
threshold samples** â€” learns an AUM upper bound that isolates mislabeled data â€¦ On WebVision50 our
method removes 17% of training data, yielding a 1.6% (absolute) improvement in test error. On CIFAR100
removing 13% of the data leads to a 1.2% drop in error."*

Their conceptual contribution is exactly the distinction you need: **three** sample types â€” easy-clean,
**hard-clean** (rare but genuine), and mislabeled â€” with the note that *"both easy and hard
correctly-labeled samples improve model generalization, whereas mislabeled examples hurt."* The
mechanism they give is that **mislabeled samples show *persistent negative* margins from the opposing
gradients of similar correctly-labelled neighbours**, whereas hard-clean samples start low and recover.
**That is the discriminant you want, and small-loss / early-learning criteria do not have it** â€” they
conflate hard-clean with mislabeled, which is precisely the failure mode you are worried about.

**â›” Why the paper's own calibration is unusable for you. [DERIVED]** The threshold-sample allocation is
`N/(c+1)` rows moved to a fake class `c+1`. With `c = 2`: **1817/3 â‰ˆ 606 rows, one third of your
dataset, sacrificed to calibration** â€” and it requires a 3-output head, which your single-logit binary
model does not have. That is not a small inconvenience; it is disqualifying.

**The adaptation that works, and it is your own Â§8.4 rule. [DERIVED]** Replace the extra class with an
**injected-noise control**: flip a known random Î· = 5% of the training labels, train, and use the AUM
distribution of the *known-flipped* rows as the null. Take the 99th percentile of that null as the
cutoff, exactly as the paper does with threshold samples. This
(i) needs no architecture change, (ii) costs 5% of labels rather than 33%, (iii) **calibrates the
instrument against a known ground truth**, which is exactly the control baseline your brief Â§8.4 says
every gate must have and whose absence voided two of your previous gates. **I rate this the single most
report-worthy construction in Q4.**

### Influence functions â€” **do not**

**[VERIFIED â€” title/venue]** Koh & Liang, *Understanding Black-box Predictions via Influence Functions*,
ICML 2017, arXiv:1703.04730, versus **[VERIFIED â€” title/venue]** Basu, Pope & Feizi, *Influence
Functions in Deep Learning Are Fragile*, ICLR 2021, arXiv:2006.14651, which gives a large-scale
empirical study of when influence estimates fail in non-convex deep models. Add: your Hessian is
71kÃ—71k, the loss is non-convex, and you have no validation instrument to check the output. **Cost high,
reliability contested, diagnostic value zero over CL+AUM. Skip, and say why in the report â€” a
one-sentence, correctly-cited rejection is worth more than a bad experiment.**

### Small-loss / early-learning â€” **cheap, but it is the criterion AUM was written to replace**

Co-teaching (Han et al., NeurIPS 2018, arXiv:1804.06872) and ELR (Liu et al., NeurIPS 2020,
arXiv:2007.00151) both exploit the early-learning phenomenon (Arpit et al., ICML 2017, arXiv:1706.05394:
networks fit clean patterns before memorizing noise). Two notes:
- **Theorem check.** âš ï¸ Small-loss *selection* is a per-example 0/1 weight that depends on the current
  model's loss â€” so it is **not** a fixed pointwise loss and it **escapes Theorem 2**, by the same
  mechanism JTT did. **And JTT is its exact mirror image: JTT up-weights the high-loss set, small-loss
  selection down-weights it.** Your JTT null therefore tells you something about this family too â€” the
  high-loss set is *not* a lever in either direction â€” though not decisively, because the two
  interventions are not symmetric in their effect on the boundary.
- ELR's regularizer targets the model's own **temporal-ensemble prediction**, which is class E3 and
  escapes both theorems. It is the most interesting member of this family for you. But it is a training
  change with a new coefficient and no way to validate it. **Not on deadline day.**

### Summary table

| method | sound at n=1817? | separates hard-clean from mislabeled? | cost | verdict |
|---|---|---|---|---|
| **Confident learning / Cleanlab** | **yes** â€” 3 parameters, and you already have the OOF probs | partly (uses only final probabilities) | **hours** | **use as ranker** |
| **AUM + injected-noise control** (adapted) | **yes** | **yes â€” this is its whole point** | 2 training runs | **use; best report value** |
| AUM as published (extra class) | **no** â€” needs 33% of data + a 3-class head | yes | high | reject, and explain why |
| Influence functions | no â€” fragile, non-convex, 71k params | weakly | very high | reject |
| Small-loss / co-teaching | yes | **no â€” conflates them** | medium | reject; AUM strictly dominates |
| ELR | yes | partly | medium | interesting, out of time |
| Dataset Cartography (Swayamdipta et al., EMNLP 2020, arXiv:2009.10795) | yes | **yes** â€” confidence Ã— variability map separates easy / **ambiguous** / **hard-to-learn** | low (same training dynamics as AUM) | **free add-on to AUM; the plot is a good report figure** |

---

## Q4.3 â€” âš ï¸ THE DIRECTION PROBLEM: cleaning is not neutral with respect to your metric

**[DERIVED]. This is the part I most want you to read, and I have not seen it stated in your logs.**

Your gap is a **recall** gap (Â§3.2: 16 more TPs). The rows a model-based detector will flag are, by
construction, concentrated on **positives the model scores low** â€” which is *the same set* you are
trying to recover. So cleaning acts directly on the contested set, and the two hypotheses give opposite
signs:

- If H2 (labels wrong): removing them sharpens the boundary and is a genuine gain.
- If H3 (censored / hard-clean): removing them deletes **exactly the boundary-defining minority
  instances**, and the retrained model becomes *more* confident that this region is negative. Recall
  falls. **You would be paying for the diagnosis with the thing you are trying to buy.**

The label-noise-under-imbalance literature agrees on the direction of the risk.
**[INFERRED â€” search-level; the ScienceDirect full texts returned 403]**
*Imbalanced classification with label noise: a systematic review and comparative analysis*,
DOI 10.1016/j.mlwa.2025.100... (ScienceDirect S2405959525001481) and *A comparative study on noise
filtering of imbalanced data sets*, Knowledge-Based Systems (S0950705124008700): *"the literature
generally agrees that deleting from the minority class poses a high risk, as it can lead to the loss of
crucial information â€¦ undersampling risks discarding critical boundary-defining instances."* Also
relevant: *Robust Data Pruning under Label Noise* (arXiv:2311.01002) notes that uncertainty-based
pruning fails under label noise because *"noisy examples also exhibit high uncertainty and could be
wrongly considered informative."*

### The prior-shift worry, and why it is NOT the real worry

You might expect the danger to be the induced prior change. Work it out. Train is 731 positives /
1086 negatives (40.23%). If a detector flags 45 positives and 17 negatives (a 3:1 skew, which is what a
recall-limited model will produce), the new prior is `686/1755 = 0.3909`, a âˆ’1.1 pp shift. In logit
terms the intercept moves by
```
log(0.3909/0.6091) âˆ’ log(0.4023/0.5977)  =  âˆ’0.4434 + 0.3960  =  âˆ’0.047
```
â€” **an affine shift of the logit, which your Platt map refits away exactly. Theorem 1 annihilates the
prior channel.** [DERIVED]

> **So the prior worry is a non-worry, and for once your own annihilation theorem is working in your
> favour.** What Theorem 1 does *not* absorb is the change in the **shape** of the decision function
> near the deleted region â€” that is `x`-dependent and non-affine. **The whole risk of cleaning is
> concentrated in exactly the place Theorem 1 cannot reach, and exactly the place where your missing
> TPs live.** Put that sentence in the report; it is a precise statement of a real asymmetry.

### Expected effect size â€” the sober number

AUM: *"removes 17% of training data, yielding a 1.6% (absolute) improvement in test error"* (WebVision50);
13% removed â†’ 1.2% error drop (CIFAR100). CL: *"moderately increase model accuracy."* These are
**order-1-percentage-point** effects. On 309 public rows, 1 pp â‰ˆ **3 rows**. You need ~16.

> **Expected effect of data cleaning: well below your 0.015 significance bar, with a real chance of a
> negative sign.** That is not a reason to skip the analysis â€” it is a reason not to *ship* it.

---

## Q4.4 â€” ðŸ”’ THE LEGALITY ARGUMENT, MADE PROPERLY

### The three prongs

**Prong (a) â€” the decision rule stays a literal 0.5 on a genuine probability. PASSES, trivially.**
Nothing in data cleaning touches inference. The model is retrained on a different training set and then
Platt-calibrated on that set's OOF and cut at 0.5. âš ï¸ One thing a reviewer *will* probe: CL's per-class
thresholds `t_j`. Pre-empt it: *"CL's thresholds are noise-estimation statistics computed from training
labels and out-of-fold probabilities; they are inputs to a data-quality decision, are never applied to
a test row, and never enter the decision rule."*

**Prong (b) â€” every knob fixed by a train-only criterion. PASSES, conditionally, and the condition is
the whole argument.** The inputs are: the 1817 training rows, their given labels, cross-validated
predictions from those rows, and (for AUM) training dynamics on those rows. **No test row, no test
label, no leaderboard number, and no positive-rate target enters at any point.** âš ï¸ The condition:
*the flagging rule must be fixed once, before any submission, and must not be revised after seeing a
score.* "We removed 62 rows, scored 0.903, so we tried removing 30 instead" is leaderboard-derived knob
setting and it is the same class of error as the prevalence pinning you already deleted scores over.

**Prong (c) â€” corrects `p(y|x)` under a demonstrably mis-specified model, rather than relabelling a
fixed estimate. PASSES, but be honest that the prong doesn't quite fit.** Prong (c) was written for
*inference-time* interventions, to rule out post-hoc relabelling of a frozen prediction. Data cleaning
is a *training-time* intervention: it changes the estimator's input and the estimate is re-derived from
scratch. It is therefore not "relabelling a fixed estimate" in the sense the prong forbids â€” it is a
different category, and I would say so in the report rather than force it. **The mis-specification
claim is separately supportable**: Q4.1(b) shows the published state of the art at separating ponds from
their confusers is F1 â‰ˆ 0.85 *with extra information*, so the *labelling process itself* is
demonstrably imperfect. That is the cleanest prong-(c) argument available and it is a citation, not an
assertion.

### âš ï¸ The failure mode, named precisely

> **The detector's input is the model's own out-of-fold predictions. So the flagged set is, by
> construction, approximately the model's error set. Delete it and retrain, and you obtain a model
> fitted to a dataset it already agreed with. Training loss falls, OOF rises, and the model becomes
> *more confident in precisely the beliefs that generated the errors.* This is a fixed-point iteration
> toward the current hypothesis, not a measurement of the data.**

Two things make this worse in your specific case than in general:
1. **It is degenerate self-distillation with hard targets and unit weight â€” applied to the labelled rows,
   where you actually have ground truth to destroy.** Compare Q3(b).0: the value of distillation came
   from *smoothing*, not from agreement. Deleting disagreements is the anti-smoothing operation.
2. **Your OOF is blind (Â§4).** OOF sits at ~0.97 for artifacts spanning 0.72â€“0.907 on the LB. So the
   *only* signal cleaning produces â€” a higher OOF â€” is the signal you have already established is
   uninformative, **and it is guaranteed to go up whether or not the cleaning was correct.** You would
   be running an experiment whose readout is known in advance.

### The protocol that makes it defensible â€” six requirements, all cheap

**R1 â€” Out-of-sample probabilities only, no exceptions.** Mandatory in CL by construction. Any
in-sample variant is circular and indefensible.

**R2 â€” A model-independent second detector, and take the INTERSECTION.** Run CL/AUM with the deployed
transformer *and* with a structurally different learner (e.g. logistic regression on temporal summary
statistics, or a k-NN / nearest-centroid detector in raw feature space). Flag a row only if **both**
dislike it. A row disliked by two different inductive biases is evidence about the *data*; a row
disliked by one is evidence about *that model*. **This is Wang & Zhou's diversity condition (Q3(b).1)
reused as a noise-detection condition, and it is the direct structural answer to the circularity
objection.** It is also the honest reply to "you deleted the rows your model happens to dislike":
*no â€” we deleted the rows two unrelated models independently disliked.*

**R3 â€” An injected-noise control, which is your own Â§8.3/Â§8.4 rule.** Flip a known random 5% of training
labels; run the detector; report **(i) recall on the known flips and (ii) false-positive rate on the
untouched rows.** This converts the detector from an opinion into a *calibrated instrument with a
measured operating characteristic.* Recovering 60% of injected flips at a 5% FPR is a defensible
instrument; 20% at 15% is a row-deleter and you stop. **Your brief Â§8.3 states the rule: "if a gate's
control does not return the value arithmetic guarantees, every number it prints is void." This is that
control. Doing it is the difference between a methodology section that scores and one that gets
attacked.**

**R4 â€” Class-symmetric reporting.** Report flag counts separately for `y=1` and `y=0` and *publish the
ratio*. If the skew toward the class you under-recall exceeds roughly 3:1 â€” beyond what the 40/60 class
sizes explain â€” treat that as a positive finding of circularity, not as a result.

**R5 â€” Remove, do not relabel.** Removal asserts only *"this row's label is unreliable"*; relabelling
asserts *"the true label is the other one"*, which is a strictly stronger claim, and in binary it is
doubly strong because it simultaneously subtracts from one class and adds to the other. Removal is the
epistemically weaker operation and therefore the defensible one â€” and it is CL's own default (prune).

**R6 â€” Pre-register, run once, disclose.** One threshold, one cleaned dataset, one submission. Write the
threshold down before you look at anything. Then disclose the whole procedure, including R3's operating
characteristic, in the report.

**A protocol satisfying R1â€“R6 is legal under all three prongs and I would defend it in a code review.**
The version *without* R2 and R3 is not defensible, and a reviewer who knows this literature will say so.

---

## Q4.5 â€” Verdict on Q4

| question | answer |
|---|---|
| Is the noise real? | **Very likely yes, ~2â€“5% of rows.** Base rate 3.4% (Northcutt 2021) + domain confusers (rice paddy / salt pan / reservoir) + the published SOTA for that discrimination being F1â‰ˆ0.85 **with** extra data you lack |
| Do we know these specific 27 FNs are mislabeled? | **No.** JTT's null refutes "hard but learnable" and leaves {mislabeled, information-censored-by-the-4â€“6-month-window}. The second is unfixable |
| Which detectors are sound at n=1817? | **CL (3 parameters â€” the isotonic small-sample objection does not transfer) and the AUM statistic with an injected-noise control.** Not AUM's published extra-class calibration (needs 33% of your data + a 3-class head). Not influence functions |
| Is cleaning legal? | **Yes, under R1â€“R6.** Prongs (a) and (b) pass cleanly; (c) passes but the prong was written for inference-time interventions and you should say so rather than force the fit |
| Does it escape the two theorems? | **Not applicable in the useful sense** â€” cleaning is not a loss change, so Theorem 2 has no purchase. âš ï¸ But note Theorem 1 **annihilates the prior-shift channel** (an affine intercept move), leaving only the non-affine decision-function-shape channel â€” which is where all the risk is |
| **Should you ship it today?** | **No.** Expected effect â‰ˆ 1 pp accuracy â‰ˆ 3 rows of 309, versus a 0.015 bar needing ~16, with a plausible negative sign because the flagged set is your minority boundary set |
| **Should you write it up?** | **Yes, prominently.** A cleaning analysis with an injected-noise operating characteristic (R3), a two-detector intersection (R2), a class-symmetry table (R4), and an explicit refusal to ship because the measured effect is below the pre-registered bar is *exactly* what a code review rewards. **A disciplined negative, properly instrumented, is worth more than an unvalidated positive.** |

---
---

# Q4.6 -- CORRECTION TO MY OWN Q4: THE PUBLIC CONFUSION CELL WAS WRONG, AND THE SWAP CHANGES THE ANSWER

**Written after Q4 was finished, against `tools/lb_cell_solve.py`. This is a correction, not a patch.
Everything below supersedes the numbers in Q4.3 and Q4.5; the reasoning in Q4.1, Q4.2 and Q4.4
survives, and in two places gets *stronger*.**

## Q4.6.0 -- What changed, stated exactly

| quantity | what Q4 and UPDATE_24 Sec 3.2 said | what is now established | source |
|---|---|---|---|
| `n_public` | 309 (inferred as 30% of 1030, never measured) | **333** | `tools/lb_cell_solve.py` steps 2-3 |
| `P_public` | 191 | **181** | same |
| `N_public` | 118 | **152** | same |
| true public prevalence | 0.618 | **0.5435** | 181/333 |
| champion cell | TP 164, **FP 17, FN 27** | TP 164, **FP 27, FN 17** | same |
| precision / recall | 0.9061 / 0.8586 | **0.8586 / 0.9061** | swapped |
| our realized pos-rate | -- | **0.5736 vs a true 0.5435: we OVER-predict** | 191/333 |
| dominant error mode | recall deficit | **precision deficit, 1.6:1** | 27 FP vs 17 FN |

The refutation of 309 is not a judgement call. Finite-sample ROC-AUC is exactly `C/(P*N)` with `C` a
half-integer under the Mann-Whitney tie convention `sklearn.roc_auc_score` implements, so a printed
9-decimal AUC is a rational with denominator `2*P*N`. No `(P,N)` with `P+N=309` lands inside the 1e-9
display window of 0.945841814; the nearest miss is 1.9e-7, i.e. **76x the window**, and our own
`P=191` misses by 5.2e-6, **2080x**. Five reported `(AUC,F1)` pairs sieved jointly leave one scaling
family, and the full-test predicted-positive counts on disk select `n=333` within it. **TP and PP are
invariant across the whole surviving family**, so precision 0.8586 / recall 0.9061 does not depend on
`n=333` being exactly right.

### The three sentences of my own Q4 that are now false

1. **Q4.5, row 2:** *"Do we know these specific 27 FNs are mislabeled?"* -- there is no set of 27 FNs.
   There are **17** FNs and **27 FPs**. The question I answered was about the smaller pile.
2. **Q4.3, effect size:** *"On 309 public rows, 1 pp is about 3 rows. You need ~16."* -- both numbers
   are wrong; recomputed in Q4.6.1 below. The bar is **9-11 rows of 333**, not 16 of 309, so the bar
   is *easier* than I stated, and my "well below the bar" verdict has to be re-derived, not assumed.
3. **Q4.3, opening:** *"Your gap is a recall gap (Sec 3.2: 16 more TPs)."* -- the premise of the whole
   direction-problem section. Re-derived from scratch in Q4.6.4.

### And one correction that is not mine but is the most consequential in the round

**UPDATE_24 Sec 3.2's reading of the leader is unsupported.** It says *"They find 16 more true
positives for 4 more false positives ... their advantage is concentrated in the high-recall corner."*
Inverting F1 = 0.918 at `P = 181` gives a **family** of cells, not one:

| leader cell | FP | FN | precision | recall | reading |
|---|---|---|---|---|---|
| TP 168, PP 185 | 17 | 13 | 0.9081 | 0.9282 | **nearest to ours: +4 TP and -10 FP** |
| TP 174, PP 198 | 24 | 7 | 0.8788 | 0.9613 | high-recall |
| TP 179, PP 209 | 30 | 2 | 0.8565 | **0.9890** | the Sec 3.2 reading: they miss 2 positives of 181 |
| TP 163, PP 174 | 11 | 18 | 0.9368 | 0.9006 | pure-precision: same recall as us, half the FPs |
| TP 157, PP 161 | 4 | 24 | 0.9752 | 0.8674 | *worse* recall than us, far better precision |

> **The "high-recall corner" reading is now one of five, and it is no longer the natural one.** The
> nearest-neighbour cell to ours says the leader's edge is **predominantly a PRECISION edge**, and the
> cell that keeps Sec 3.2's story requires the leader to miss only **2 positives out of 181** while
> running 30 false positives -- a detector far outside anything in this ledger. We over-predict
> (0.5736 against a true 0.5435); every leader cell except the last two over-predicts *less* than we do.
>
> **This bears directly on Q1 and on `round24_partial_auc.md`.** Q1's premise -- "optimize the
> high-recall region, the leader lives at recall 0.94 where we sit at 0.859" -- is built on the swapped
> cell. We sit at recall **0.906**, not 0.859, and the region where we are demonstrably worse than the
> leader is the **low-FPR** region, not the high-TPR region. A one-way pAUC restricted to a high-TPR
> band is optimizing the half of the curve where the corrected arithmetic says we are already fine.
> **The partial-AUC lane should be re-aimed at the low-FPR / high-precision end, or its premise
> restated as unknown.** I flag this rather than resolve it; it is that agent's question.

## Q4.6.1 -- The effect-size bar, recomputed exactly at n = 333

Marginal value of one row, from `F1 = 2*TP/S` with `S = PP + P = 372`:

```
remove one FP (S -> S-1):        dF1 = F1/(S-1)            = 0.8817204/371  = +0.00237660
recover one FN (TP+1, S+1):      dF1 = 2(S-TP)/(S(S+1))    = 416/138756     = +0.00299807
ratio = (FN recovery)/(FP removal) = (S-TP)(S-1)/(TP(S+1)) = 1.26149
```

> **They are NOT worth the same.** In the large-`S` limit the ratio is exactly `2/F1 - 1`, which at
> F1 = 0.8817 is **1.2683**. **One recovered false negative is worth 1.26 false positives suppressed.**
> This is a general fact about F1 that we have never written down: recovering a miss both raises the
> numerator and is charged once in the denominator, while suppressing a false alarm only shrinks the
> denominator. The asymmetry grows as F1 falls (`2/F1 - 1 -> 1` only as `F1 -> 1`).

Rows needed to clear the significance bar, with `d(composite) = 0.6 * dF1` when only the `TargetF1`
column moves (`TargetRAUC` untouched, hence `dAUC = 0`):

| rows moved | via FN recovery | via FP suppression |
|---|---|---|
| 5 | F1 0.89655, **+0.00890 comp** | F1 0.89373, +0.00721 comp |
| 8 | F1 0.90526, +0.01413 comp | F1 0.90110, +0.01163 comp |
| **9** | F1 0.90814, **+0.01585 comp -- CLEARS** | F1 0.90358, +0.01312 comp |
| **11** | F1 0.91384, +0.01927 comp | F1 0.90859, **+0.01612 comp -- CLEARS** |

**The bar in rows: 9 recovered false negatives, or 11 suppressed false positives, out of 333 public
rows.** Not "~16 of 309". Two further notes:

- The binomial noise floor scales as `sqrt(309/333) = 0.963`, so **0.015 becomes about 0.0145**. I keep
  0.015 as the bar because loosening a significance bar after learning the sample is bigger is the same
  species of error as Sec 8.5 (an LB-derived bar). Keep 0.015; note 0.0145 and do not use it.
- **The public sample is 32.3% of the test set, not 30%.** So a mechanism that flips `X` rows test-wide
  shows up as `0.323X` public rows. **To move 11 public FPs you must suppress about 34 FPs across all
  1030 test rows.** Our test-wide error budget scales to roughly **84 FPs and 53 FNs**. So the bar,
  stated in the currency an actual method has to pay in, is: **eliminate ~40% of every false positive
  we make**, or **recover ~53% of every positive we miss.** That is the number to hold every proposal
  in this round against, and it is far harsher than "9 rows" sounds.
- Maximum reachable by fixing each pile completely: all 27 FPs -> F1 **0.95072**; all 17 FNs -> F1
  **0.93059**. The FP pile is the larger prize (+0.0690 vs +0.0489 F1) purely because it is more
  numerous, *despite* each individual FN being worth 1.26x more.

## Q4.6.2 -- THE BIG ONE: is the label noise class-asymmetric, and does that give cleaning a sign?

**This was R4, a hygiene check. Under the swap it is load-bearing, so I am doing it properly and I am
going to kill most of it.**

### (i) First, fix the direction. Two annotation error types, and they are not interchangeable

The remote-sensing literature in Q4.1(b) names the confusers -- **rice paddies, salt pans, small
reservoirs, wetland parks** -- and it names them as *negatives that look like ponds*. That produces two
error types with **opposite** consequences, and the whole section turns on keeping them apart:

```
COMMISSION  (0 -> 1):  a salt pan / rice paddy annotated as a POND.   Row carries y = 1, truth 0.
OMISSION    (1 -> 0):  a real pond the annotator missed.              Row carries y = 0, truth 1.
```

The cited sources say the confusers *"cause a lot of omission/commission errors"* -- both. But the
domain argument is specifically about **abundant negatives that resemble the positive class**, and that
is a commission mechanism. A liberal detector run over a landscape full of rice paddies makes
commission errors. **So the domain asymmetry argument in my own Q4.1(b) predicts COMMISSION-dominant
noise: rows labelled 1 that are truly 0.**

### (ii) The channel the task brief proposed -- and it runs the wrong way

The task put to me was: *if the noise is asymmetric in the confuser direction, some of our 27 false
positives may be correctly-classified rows with wrong labels.* **Work the arrow through and it does not
hold.** A false positive is `predicted 1, key says 0`. For that row to be *correctly classified*, its
truth must be 1 while the key says 0 -- that is an **omission** error, not a commission error. The
confuser mechanism produces the opposite:

| noise in the TEST KEY | mislabelled row looks like | a good model predicts | scored against the key as |
|---|---|---|---|
| **commission** (confuser labelled pond) | not a pond | **0** | **FALSE NEGATIVE** |
| **omission** (real pond labelled non-pond) | a pond | **1** | **FALSE POSITIVE** |

> **So the domain-confuser asymmetry inflates our FN pile, not our FP pile -- and the FN pile is the
> SMALL one (17 against 27).** Under the *old*, wrong cell (FN 27 > FP 17) this argument would have
> looked beautifully confirmatory, and I would very likely have written it up as one. The swap inverts
> it: **the observed asymmetry runs AGAINST the direction the domain argument predicts.** That is a
> negative for the hypothesis as posed, and it is exactly the kind of thing the corrected cell is for.

Magnitudes, so this is not just a sign argument. With `n = 333`, roughly 166 true positives and 167 true
negatives, and our own accuracy on each side around 0.90, the number of *manufactured* errors at a key
noise rate `r` is about `150 * r` on each side:

| key noise rate | manufactured FPs (omission) | manufactured FNs (commission) | share of our 27 FPs | share of our 17 FNs |
|---|---|---|---|---|
| 2.0% | ~3 | ~3 | 11% | 18% |
| **3.4%** (Northcutt base rate) | **~5** | **~5** | **19%** | **29%** |
| 5.0% | ~7.5 | ~7.5 | 28% | 44% |

**At the literature base rate, key noise explains at most about a fifth of our false positives and
about a third of our misses. It cannot be the story.** And it explains proportionally *more* of the
smaller pile, which is the opposite of a useful finding.

### (iii) The bound this DOES buy, and it is worth having: the noisy-key F1 ceiling

An oracle that predicts the *truth* perfectly still loses points against a noisy key. At symmetric rate
`r` the oracle's cell is `TP = P - commission`, `FP = omission`, `FN = commission`:

| key noise rate | oracle cell | **oracle F1** |
|---|---|---|
| 2.0% | TP 177, FP 3, FN 4 | **0.98061** |
| **3.4%** | TP 175, FP 5, FN 6 | **0.96953** |
| 5.0% | TP 172, FP 8, FN 9 | **0.95291** |

> **This closes a lane we might otherwise have leaned on in the report.** Even at a *generous* 5% key
> noise the ceiling is F1 0.953, and the leader is at ~0.918. **Test-key label noise does not explain
> the leader's gap over us and cannot be offered as an excuse for it.** There is ~0.036 F1 between us
> and the leader that sits strictly below the noise ceiling and is therefore genuinely reachable by
> *someone*. State that plainly; it is the honest version of Sec 3.3's retraction and it removes the
> "we are at the annotation ceiling" comfort story.

### (iv) The channel that DOES have the right sign, and it is a different channel

There is a second, entirely separate route from the same asymmetry, and it is the one with teeth. It
runs through the **training set**, not the key:

```
commission noise in TRAIN  ->  train rows labelled 1 that are actually rice paddies / salt pans
                           ->  the model learns an OVER-INCLUSIVE pond concept
                           ->  at test time it fires on confusers
                           ->  FALSE POSITIVES, and a realized positive rate above the truth.
```

**That is precisely the error signature the corrected cell shows: FP 27 vs FN 17 (1.6:1), and a realized
public positive rate of 0.5736 against a true 0.5435.** And it is a *non-trivial* prediction, because
we train at prevalence 0.4023 and deploy at 0.5435 -- naively we should *under*-predict, and we do not.

> **So the asymmetry hypothesis is not dead. It changes channel.** The confuser asymmetry does not mean
> "our FPs are secretly correct" (Q4.6.2(ii) kills that); it means **"our FPs are real, and a
> commission-biased training set is the mechanism that manufactured them."** That version has a
> predicted sign -- **cleaning helps** -- where Q4 had to leave it ambiguous, and the prediction it
> makes about the LB cell is *confirmed* by the corrected cell rather than assumed from it.

**Legality checkpoint, and it is urgent.** The corrected cell hands us a direction: *we over-predict.*
Any proposal of the form "so make the model more conservative" is **prevalence pinning wearing a third
hat** -- an LB-derived quantity setting a knob, prong (b), the exact violation this project deleted six
scores over. The commission hypothesis is legal **only** because it was derived from the domain
literature (train-only reasoning, Q4.1(b), written before the cell was corrected) and *predicts* the
over-prediction. Confirming a prediction is legal; tuning to a target is not. **The moment any knob is
sized to reduce the FP count, the whole thing is void.** Write that sentence into the report next to
the cell; a reviewer who sees us discover FP > FN and then ship something more conservative will
otherwise assume the worst, and would be right to.

### (v) THE KILL: the detector is blind in exactly the regime where the payoff is large

Now put Q4.6.2(iv) together with Q4.2's caveat about the class-conditional-noise assumption, which I
raised and then did not follow through.

**Proposition (detector blindness under systematic noise). [DERIVED]** Every detector on my Q4.2 list --
confident learning, AUM, small-loss/co-teaching, ELR, JTT -- computes a statistic of the form *"the
model, trained without this row, disagrees with this row's label."* Let the noise be instance-dependent
and supported on a feature-space region `R` (the confuser manifold). Two regimes:

- **Random / atypical noise (the CCN idealization).** A flipped row is an isolated contradiction to its
  neighbourhood. It can only be fitted by memorization, and Arpit et al. (ICML 2017, arXiv:1706.05394)
  is exactly the statement that memorization comes late and does not generalize. So the *out-of-fold*
  prediction on that row follows the clean neighbourhood and **disagrees with the noisy label**.
  Detector power is high. This is the regime every one of those papers is benchmarked in.
- **Systematic, feature-clustered noise (`R` = the confusers, our regime by hypothesis).** The flipped
  rows are *not* isolated: they are a coherent sub-population with mutual support. A learner with the
  capacity to represent `R` fits them **as signal, and it generalizes across folds**. The out-of-fold
  prediction on a confuser row therefore **agrees with its noisy label**. **Detector power on `R`
  collapses toward zero.**

> **Detector power is highest exactly where the noise is random, and lowest exactly where it is
> systematic. The systematic component is the one with the domain argument behind it. So the
> instrument's sensitivity is ANTI-CORRELATED with the noise we have actual reason to believe exists.**

This is not a refinement of the CCN caveat, it reverses its practical import. In Q4.2 I wrote that CL
"will still rank suspicious rows usefully; only its noise-*rate* estimate is untrustworthy." **That is
wrong for feature-clustered noise: the ranking is what fails, because the flagged set systematically
excludes `R`.** Correct that line.

**The dichotomy, which is the finding:**

| regime | can the detector see it? | payoff from cleaning it |
|---|---|---|
| random / CCN | **yes** | **negligible** -- see the arithmetic below |
| systematic / confuser-clustered | **no** | potentially large |

> **Cleaning pays off only in the regime where our instruments are blind, and our instruments work only
> in the regime where the payoff is negligible.** That single sentence is the strongest form of the Q4
> negative, it is a *mechanism* rather than an effect-size complaint, and it is the version I would put
> in the report.

### (vi) The arithmetic on the visible regime, so the negative rests on numbers too

At the Northcutt base rate, commission-mislabelled train positives number `0.034 * 731 = ~25 rows`,
1.4% of the 1817-row training set. Two channels:

- **Prior channel:** removing 25 positives moves the train prior 0.4023 -> `706/1792 = 0.3940`, a -0.8pp
  shift, i.e. a logit intercept move of `log(0.3940/0.6060) - log(0.4023/0.5977) = -0.0439`. **Affine.
  Theorem 1 annihilates it exactly.** (Q4.3 got this right and it survives the swap unchanged.)
- **Shape channel, extrapolated from the only measurements anyone reports:** AUM removes **17%** of
  training data for a **1.6%** absolute test-error improvement (WebVision50); 13% -> 1.2% (CIFAR100).
  Linear in removal fraction, `1.4% / 17% * 1.6% = 0.13%` absolute, which on 1030 test rows is
  **1.4 rows total across both error types, i.e. about 0.45 public rows.**

**Against a bar of 11 public FPs (~34 test-wide), the visible-regime effect is 0.45 public rows. The
shortfall is a factor of ~25, and that is before noting that AUM's headline numbers come from datasets
with 20-40% noise where the removal fraction is 12x ours.** Stated as leverage: we would need cleaning
1.4% of training labels to eliminate ~40% of all our false positives -- a leverage of ~29x -- where the
best-documented measurement in the literature shows leverage of about **0.09x**.

**Verdict on Q4.6.2, and I am willing to be this blunt about my own strongest remaining candidate:**

1. The specific claim put to me -- *some of our 27 FPs are correctly-classified rows with wrong labels*
   -- is **refuted by the direction of the arrow**. The confuser mechanism is commission, and commission
   in the key manufactures **false negatives**. (Q4.6.2(ii))
2. The asymmetry nevertheless **does** acquire a predicted sign through a *different* channel --
   commission in **train** producing an over-inclusive model -- and that prediction is **confirmed** by
   the corrected cell (FP > FN, realized rate above true). This is a genuine strengthening of Q4 and it
   is the one part of this section I would defend hard. (Q4.6.2(iv))
3. **But the sign is worth nothing, because the payoff and the observability are mutually exclusive.**
   (Q4.6.2(v)) And in the observable regime the effect is ~25x below the bar. (Q4.6.2(vi))
4. So the Q4 recommendation is **UNCHANGED -- do not ship, do write it up -- but the reason is now
   much better.** Q4 declined on an ambiguous sign and a small effect. Q4.6 declines on a *structural*
   argument: the instrument cannot see the thing worth cleaning. **That is a report-grade negative and
   it is worth more than the ambiguous version it replaces.**

## Q4.6.3 -- Q4.3's "direction problem" under the swap: the conclusion INVERTS, in which class carries the risk

Q4.3 argued: *"The rows a model-based detector will flag are concentrated on positives the model scores
low -- the same set you are trying to recover. Removing them deletes boundary-defining minority
instances and recall falls."* Every clause of that is conditioned on the recall framing. Re-derive it.

A disagreement detector produces **two** flag types, and Q4.3 only considered the first:

| flag type | what it is | contains (good to delete) | contains (catastrophic to delete) |
|---|---|---|---|
| **(i)** `y=1`, scored low | positives the model rejects | **commission errors** -- confusers annotated as ponds | **H3 censored true ponds** -- real ponds whose drain event fell outside the window |
| **(ii)** `y=0`, scored high | negatives the model likes | omission errors -- real ponds annotated non-pond | **the confuser exemplars themselves** -- the only rows teaching the model that a salt pan is not a pond |

Now read the two columns against the corrected cell:

- **Type (ii) is the dangerous direction, and Q4.3 never mentioned it.** Deleting flagged `y=0` rows
  removes precisely the negatives that sit on the pond boundary. The model then has nothing left
  contradicting "anything pond-shaped is a pond", becomes **more** over-inclusive, and **false positives
  rise** -- our dominant, 27-row error mode. Three things make this worse than the old worry: it acts on
  the **larger** error pile; the deleted rows carry the discriminative information the domain literature
  says is already scarce; and there are 1086 negatives, so a detector run at a fixed flag rate produces
  1.49x as many type-(ii) flags as type-(i) flags before any model effect.
- **Type (i) is now the LESS dangerous direction.** It was Q4.3's whole warning. Under the commission
  hypothesis its "good to delete" component -- confuser rows annotated as ponds -- is the *dominant*
  hypothesized noise mode, so the mixture is more favourable than Q4.3 assumed.

> **So Q4.3's specific warning inverts: deleting flagged positives is the safer half and deleting
> flagged negatives is the reckless half. Q4.3 argued the exact opposite.** Its *general* lesson
> survives and is strengthened: cleaning is not class-neutral, the risk is entirely in the
> `x`-dependent, non-affine reshaping of the decision function, and that is exactly the channel
> Theorem 1 cannot annihilate. That paragraph in Q4.3 was right and should be kept verbatim; only the
> class it points at changes.

**The tempting move, and why we must not make it.** The above licenses an *asymmetric* cleaning rule --
*delete flagged `y=1` rows only, never flagged `y=0` rows* -- and that rule now has a defensible
train-only prior behind it (the domain literature says commission dominates). It is the strongest
concrete version of Q4 that survives everything above.

**We still should not ship it, and the reason is an integrity reason rather than a technical one.** The
rule reduces predicted positives. We now know, from the leaderboard, that we over-predict. Whatever
order we actually reasoned in, **the corrected cell landed before the rule was written, so the
pre-registration prong (b) requires cannot honestly be claimed.** A reviewer reading the commit
timestamps will see LB information available upstream of a knob whose direction happens to match the
LB's diagnosis. The correct move is to **disclose the ordering and decline**, not to run it and argue.
This is the same failure the prevalence pin was, one abstraction layer up, and the only defence against
it is to name it before someone else does.

## Q4.6.4 -- The protocol, revised: R3 becomes the decisive experiment and R4 becomes directional

Q4.4's R1-R6 stand, with three changes forced by Q4.6.2(v).

**R3 (revised) -- inject STRUCTURED noise, not just random noise. This is now the whole experiment.**
The original R3 flips a uniformly random 5% of labels and measures detector recall against them. Under
the blindness proposition that control measures the detector on the *easy* regime and will return a
flattering operating characteristic that does not transfer to the noise we actually care about. **A
control that only exercises the regime where the instrument works is not a control.** Run both arms:

```
ARM A (random, the old R3):   flip a uniform random 5% of the 1817 training labels.
ARM B (structured, new):      k-means (k=8, seeded) on per-row temporal summary statistics
                              (per-band median / min / max over the 12 months -- MODEL-FREE);
                              take the cluster C with the highest label entropy, i.e. the most
                              pond/non-pond-mixed cluster, which is the confuser cluster by
                              construction; flip 5% of the dataset's labels ALL INSIDE C and ALL
                              in the 0 -> 1 direction, to mimic commission.
REPORT for each arm:          detector recall on the known flips, and FPR on the untouched rows.
```

> **Prediction, stated before the run so it can fail: `recall(ARM B) << recall(ARM A)`, and quite
> possibly `recall(ARM B)` at or below the untouched-row FPR, i.e. no power at all.** If that is what
> comes out, the cleaning lane closes **on a measurement of our own instrument** rather than on an
> argument, and it closes for every detector in the family at once, because they all share the
> disagreement statistic. Cost: two extra training runs, **zero submissions**. Against Sec 8.3's rule
> -- *"if a gate's control does not return the value arithmetic guarantees, every number it prints is
> void"* -- this is the control, and it is the single highest-value cheap thing left in Q4.

**R4 (revised) -- the class-symmetry table is now a DIRECTIONAL test, and its old decision rule is
backwards.** Q4.4 said: *"if the skew toward the class you under-recall exceeds ~3:1, treat that as a
positive finding of circularity."* Under the swap there is no under-recalled class, and the commission
hypothesis makes a **positive prediction**: flags should skew toward **`y=1`** rows. Circularity makes
the *same* prediction whenever the model's OOF errors skew that way, so **R4 on its own cannot separate
the two** -- it never could, and under the old cell that was hidden. The table must be crossed with R2:

| transformer flags skew | independent detector flags skew | reading |
|---|---|---|
| `y=1` | `y=1` | **evidence about the DATA** -- two inductive biases agree; commission hypothesis supported |
| `y=1` | flat or `y=0` | **evidence about the MODEL** -- circularity; report as a null |
| flat | flat | no detectable asymmetry; the commission hypothesis is not supported at this sample size |

**R7 (new) -- model-free corroboration.** Report what fraction of the flagged rows fall inside ARM B's
high-entropy cluster `C`, against the base rate. `C` is defined by k-means on raw temporal statistics
with no model involvement, so an over-representation of flags inside `C` is **independent** evidence
that the flags track the confuser manifold rather than the transformer's idiosyncrasies. This is the
cheapest possible answer to "you deleted the rows your own model dislikes" and it costs one k-means.

## Q4.6.5 -- Revised verdict table for Q4 (supersedes Q4.5)

| question | answer after the correction |
|---|---|
| Is the noise real? | **Unchanged: very likely, ~2-5%.** Base rate 3.4% + domain confusers + published SOTA of F1~0.85 at that discrimination *with* data we lack |
| Is it class-asymmetric? | **The literature predicts COMMISSION-dominant** (confusers annotated as ponds). Which is `y=1` noise in train, and `FN`-manufacturing noise in the test key |
| Do we know some of the 27 FPs are correctly-classified rows with wrong labels? | **No -- the arrow runs the other way.** Commission noise in the key manufactures **false negatives**, and the FN pile is the small one. The claim as posed is refuted |
| Does the asymmetry give cleaning a predicted sign? | **Yes, through a different channel: commission in TRAIN -> over-inclusive model -> false positives.** That mechanism predicts our exact signature (FP:FN = 1.6:1, realized 0.5736 vs true 0.5435), and the prediction is confirmed |
| Then does cleaning help? | **In principle yes; in practice no.** Payoff is large only for systematic, feature-clustered noise, and every available detector is provably blind on exactly that set because the model fits it as signal and it generalizes out-of-fold |
| What does it buy in the visible regime? | ~0.45 public rows, against a bar of 11. **~25x short** |
| Does key noise explain the leader's gap? | **No.** Even at 5% key noise the oracle F1 ceiling is 0.953 and the leader is at ~0.918. ~0.036 F1 is genuinely reachable |
| What is the bar, exactly? | **9 recovered FNs or 11 suppressed FPs of 333** = ~34 FP suppressions test-wide = **eliminating ~40% of all our false positives** |
| Is an FN worth the same as an FP? | **No: `2/F1 - 1 = 1.268`.** One recovered miss is worth 1.27 suppressed false alarms |
| Ship it? | **No, and now for a structural reason rather than an effect-size reason** |
| Write it up? | **Yes, and it is a better section than before.** The dichotomy in Q4.6.2(v) plus the ARM-A/ARM-B control in R3 is a genuine methodological contribution and costs zero submissions |
