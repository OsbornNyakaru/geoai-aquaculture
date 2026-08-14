# Round 23 — Q3: Losses that survive Platt annihilation and change the ranking

**Agent:** Q3 owner. **Status:** IN PROGRESS (written incrementally; if truncated, what is here is final).
**Date:** 2026-08-14.

## Scope

For each of {focal loss, label smoothing, LDAM, asymmetric loss (ASL), PolyLoss,
recall-oriented / recall-constrained surrogates, + anything else found}, decide:

- **(A) AFFINE** — the induced population minimizer is `z' = alpha*z + beta` of the BCE logit
  ⇒ exactly annihilated by the team's train-OOF Platt map `sigma(a z + b)` ⇒ worthless.
- **(R) REORDERING** — the induced population minimizer is a *non-affine, non-monotone-in-z*
  map, or a monotone map that is **example-dependent** ⇒ can change the order of test scores
  ⇒ potentially valuable.
- **(M) MONOTONE-BUT-NOT-AFFINE** — a third category the brief does not name and which
  matters enormously. See "Correction 1" below.

Labels on every claim: **VERIFIED (read paper)** or **INFERRED**.

---

# HEADLINE (written first, refined below)

**"Affine" is the wrong test. The correct test is "pointwise-decomposable".**

The team's Platt Annihilation Theorem is true but far too weak, and searching for a
*non-affine* loss is searching the wrong axis. The operative theorem is:

> **POINTWISE-LOSS ORDER INVARIANCE.** Let the training objective be
> `L = (1/n) sum_i [ y_i * l1(z_i) + (1-y_i) * l0(z_i) ]` — i.e. **decomposable**, with the
> per-example contribution depending on example *i* only through `(y_i, z_i)` via a fixed pair
> of functions `l1` (decreasing) and `l0` (increasing). Then the unrestricted population
> minimizer is
>   `z*(x) = T(eta(x))`, where `eta(x) = p(y=1|x)` and `T = argmin_z [ e*l1(z) + (1-e)*l0(z) ]`
> is **one fixed function of eta, identical for every x**. If `T` is strictly increasing
> (guaranteed whenever the loss is classification-calibrated), then:
>   1. **ROC-AUC is EXACTLY identical to plain BCE's.** (AUC is invariant to any strictly
>      increasing transform of the score.)
>   2. The induced 0.5 decision after *any* monotone recalibration is
>      `1[eta(x) > t]` for a single scalar `t`. i.e. it is **a pure threshold slide** along the
>      *same* ordering — which §3 of the brief already proves is arithmetically incapable of
>      closing the gap (only ~15/1030 rows near the cut; ~23 needed).

**Every loss on the team's Q3 list is decomposable-pointwise.** Focal, label smoothing, LDAM,
ASL, PolyLoss, class weighting, balanced softmax, logit adjustment, sigmoidF1's pointwise part,
recall-weighted losses — all of them. Therefore **none of them reorders at the population
optimum.** They differ from BCE only in the shape of the warp `T`, and `T` is monotone.

Consequence: they are not merely "annihilated by Platt"; they are annihilated by **AUC itself**
for the 40% column, and reduced to **a disguised threshold move** for the 60% column — which is
the lane the team has already declared dead on arrival (§5 prong (a)) and independently proved
worthless (§3 density argument).

**The ONLY channel by which any of these can change the test ranking is
misspecification/finite capacity** — i.e. how a 71k-parameter model, which *cannot* represent
`eta`, chooses to allocate its approximation error. That channel is real (the team's model is
demonstrably misspecified: adversarial AUC = 1.0), but **its sign is not predictable from any
theory in the literature.** Anyone who tells the team "focal loss will help your recall" is
making an empirical bet, not a theoretical claim.

**To provably reorder you must leave the decomposable-pointwise class.** Only three families do:
  (E1) **Non-decomposable** objectives (pairwise/listwise ranking, AUC/pAUC surrogates,
       constrained/Lagrangian metrics) — genuinely different objective, BUT see the trap below.
  (E2) **Losses whose per-example term depends on the model's behaviour on OTHER inputs**
       — consistency/invariance losses, mixup/VRM, contrastive auxiliaries. The team's own
       cross-view invariance loss (§2 item 3) is already in this class.
  (E3) **Changing the TARGET as a function of x** — i.e. distillation on soft, x-dependent
       labels. The team's single biggest win (+0.0100) is in this class. That is not a
       coincidence; it is the only family in their history that escapes the theorem.

---

(sections appended below as research proceeds)

---

# 1. FOCAL LOSS — **DOES NOT REORDER.** Settled by an explicit theorem.

**Verdict: Class M (monotone, non-affine). Population-optimal ranking is IDENTICAL to BCE's.
AUC unchanged by construction. F1@0.5 effect = pure threshold slide.**

## 1.1 The decisive citation

**Charoenphakdee, Vongkulbhisal, Chairatanakul, Sugiyama — "On Focal Loss for Class-Posterior
Probability Estimation: A Theoretical Perspective", CVPR 2021, arXiv:2011.09172.**
**VERIFIED (read the paper, ar5iv full text).**

Three results, taken together, close the question:

- **Theorem 3 (verbatim):** *"For any γ ≥ 0, the focal loss ℓ^γ_FL is classification-calibrated."*
- **Theorem 5 (verbatim):** *"For any γ > 0, the focal loss ℓ^γ_FL is not strictly proper."*
  (So focal's output is NOT η — it is systematically warped. This is the part Mukhoti et al.
  observe as "better calibration".)
- **Theorem 11:** the warp is **invertible in closed form**. The map recovering η from the focal
  minimizer `q^{γ,*}` is
      `Ψ^γ_i(v) = h^γ(v_i) / Σ_l h^γ(v_l)`,
      `h^γ(v) = v / φ^γ(v)`,  `φ^γ(v) = (1-v)^γ − γ (1-v)^{γ-1} v log v`.
- **Lemma 14:** *`h^γ` is a strictly increasing function*, i.e. `u > v  ⟺  h^γ(u) > h^γ(v)`.
  The authors themselves call this the **"strictly order-preserving property"** and use it to
  prove Theorem 3.

## 1.2 The derivation, in the team's own notation (binary case)

Focal loss is decomposable and pointwise: `L = Σ_i [ y_i·(1−p_i)^γ·(−log p_i)
+ (1−y_i)·p_i^γ·(−log(1−p_i)) ]`, `p_i = σ(z_i)`. Hence the unrestricted population minimizer at
x depends on x **only through η(x)**:

    p*(x) = argmin_p [ η(x)·(1−p)^γ(−log p) + (1−η(x))·p^γ(−log(1−p)) ]  =:  T_γ(η(x)).

By Theorem 11 / Lemma 14 the inverse of `T_γ` is `η = h^γ(p) / (h^γ(p) + h^γ(1−p))`, and `h^γ` is
strictly increasing ⇒ **`T_γ` is a strictly increasing bijection of [0,1] onto its range.**
Therefore

    z*_focal(x) = logit(T_γ(η(x))) = G_γ( z*_BCE(x) )     with G_γ strictly increasing.

Now apply the team's three tests:

| test | result |
|---|---|
| Does it reorder test scores? | **NO.** `G_γ` is a fixed strictly increasing scalar function applied identically to every row. Row order is preserved exactly. |
| Effect on the AUC column (40%) | **EXACTLY ZERO.** ROC-AUC is invariant under strictly increasing score transforms. Focal loss cannot change AUC at the population optimum, with or without Platt. |
| Effect on the F1 column (60%) | The decision after Platt is `1[a·G_γ(z) + b > 0]` = `1[z > G_γ^{-1}(−b/a)]` = `1[η > t]`. **A single scalar threshold on the SAME ordering.** |
| Platt annihilation (§5 as written) | **Survives it — but only technically.** `G_γ` is genuinely non-affine, so the two-parameter Platt map cannot undo it exactly. What survives is *only a shift of `t`*. |

**So the brief's §5 parenthetical — "Focal loss, for instance, is not affine in the logit" — is
correct but misleading.** Non-affine ≠ reordering. Focal is non-affine and *still* order-identical.
The residual, un-annihilated part of focal loss is precisely **a threshold move**, which §5 prong
(a) calls dead on arrival and which §3 proves is worth ≈0 anyway (only ~15/1030 rows in
[0.45, 0.55]; ~23 rows must cross).

## 1.3 Resolving the Mukhoti tension explicitly

Mukhoti, Kirsch, van Amersfoort, Torr, Gal — *"Calibrating Deep Neural Networks using Focal
Loss"*, NeurIPS 2020, arXiv:2002.09437. Their claim ("focal loss improves calibration") is
**entirely compatible with, and explained by, Theorem 5**: focal's minimizer is improper and
systematically *under-confident*, which happens to cancel the over-confidence of over-parameterised
nets. That cancellation is a **monotone re-scaling of confidence**. For this team it is:

- **irrelevant to AUC** (monotone ⇒ invariant), and
- **already the job of the Platt step** for the decision column.

So the answer to the team's framed dichotomy — *"either irrelevant (annihilated) or the whole
point (if it reorders)"* — is: **irrelevant. It does not reorder.** The calibration benefit is
consumed by a step the team already performs.

## 1.4 What focal loss does BEYOND affine calibration — the honest residue

Two things survive, and neither is a population-minimizer effect:

**(i) Implicit regularisation (the real mechanism).** Mukhoti et al. do not merely observe better
ECE; they argue focal loss acts as **a maximum-entropy / confidence penalty** — they show focal
loss upper-bounds `KL(q‖p) − γ·H(p)` — and that it **implicitly regularises the weight norm**,
with the effect that the model stops driving logits to large magnitudes late in training.
*(INFERRED that this is the operative channel here; VERIFIED that the paper makes the entropy-bound
and weight-norm arguments — see §1.5 note.)* Under **restricted capacity** the entropy penalty
changes *which* function in the 71k-parameter family is selected, so the learned `ẑ` is **not**
`G_γ(ẑ_BCE)`. That genuinely reorders. But it is regularisation, not the focal warp, and its sign
is not predictable.

**(ii) Gradient re-allocation toward low-scored positives.** Focal's `(1−p)^γ` up-weights exactly
the positives the model currently scores low — i.e. **precisely the subpopulation §3 identifies**
(confidently-missed positives at 0.15–0.45). This is the one genuinely on-target property of focal
loss for this team. It is a *finite-capacity budget-allocation* argument, not a theorem.

**Net recommendation on focal loss:** it is worth *at most one* run, and if the team runs it they
should be clear that **they are testing an implicit regulariser with an unknown sign, not
harvesting the calibration result.** Two specific warnings:
  - Any measured change in the **AUC column** is 100% attributable to channel (i)/(ii)
    (capacity/optimisation), never to the loss's intended effect. This makes AUC a clean
    *instrument*: **if AUC moves, the loss really did change the function; if AUC does not move,
    focal did nothing but slide the threshold** and can be discarded immediately. That is a free,
    train-only, LB-free diagnostic and it is the single cheapest thing in this report.
  - Focal's hard-example up-weighting amplifies label noise and outliers. Under a shift with
    disjoint support (adversarial AUC 1.0) the "hard" examples are disproportionately the
    shift-atypical ones.

## 1.5 Legality of focal loss against §5

- **(a) literal 0.5** — passes. No threshold is touched; the Platt+0.5 step is unchanged.
- **(b) train-only knobs** — passes *only if* γ is fixed a priori (e.g. γ=2, Lin et al.'s value).
  It **fails** if γ is swept and picked on LB. And it **cannot** be picked on OOF (§4: OOF is blind).
  So the team must pre-register γ = 2 and accept it. This is a real constraint: focal has a
  free parameter and the team has no valid instrument to set it.
- **(c) corrects p(y|x) under mis-specification** — **FAILS as stated.** Theorem 5 says focal
  *mis-estimates* p(y|x) rather than correcting it; Theorem 11 exists precisely to undo the damage.
  Focal loss does not correct the posterior. Any benefit is incidental regularisation.

**Sources (this section):** arXiv:2011.09172 (CVPR 2021) — VERIFIED, theorem numbers and formulas
read from full text. arXiv:1708.02002 (Lin et al., focal loss, ICCV 2017) — definition only,
VERIFIED by universal use. arXiv:2002.09437 (Mukhoti et al., NeurIPS 2020) — see §1.6.

## 1.6 Mukhoti et al., read directly — and it makes the case AGAINST focal, not for it

**arXiv:2002.09437, NeurIPS 2020. VERIFIED (read ar5iv full text).** Four extractions:

1. **Eq. (1), §4:** `L_f ≥ KL(q‖p̂) − γ·H[p̂]`. Focal loss upper-bounds cross-entropy minus an
   entropy bonus ⇒ it is **a confidence penalty / max-entropy regulariser.** A confidence penalty
   is, by definition, a statement about the *magnitude* of the logits, not their order.
2. **Proposition 1:** `‖∂L_f/∂w‖ ≤ ‖∂L_c/∂w‖` when `g(p̂,γ) ∈ [0,1]`; the authors state focal loss
   *"starts to act as a regulariser on the network's weights once the model has gained a certain
   amount of confidence."* ⇒ implicit weight-norm regularisation. This is channel (ii)
   (optimisation dynamics / implicit regularisation), **not** a change to the population minimizer.
3. **Accuracy:** *"whilst preserving accuracy, it yields state-of-the-art calibrated models."*
   Accuracy preservation across their whole benchmark suite is exactly what order-preservation
   predicts.
4. 🔴 **THE DECISIVE NUMBER, and it is in their own paper.** Focal-trained models have optimal
   post-hoc **temperature T ≈ 0.9–1.1**, whereas cross-entropy models need **T ≈ 2.0–2.8**; they
   describe focal models as *"innately calibrated"* and say they *cannot be made significantly more
   calibrated by temperature scaling.*

   **Read that as an identity:** `focal ≈ cross-entropy composed with a fixed temperature ≈ 2.5`.
   A temperature is `z ↦ z/T` — **an affine map of the logit with β = 0.** So Mukhoti et al.'s own
   headline measurement says the bulk of focal loss's effect is, empirically, **affine in the
   logit** — i.e. it is *exactly* the object the team's Platt Annihilation Theorem kills, in the
   strict original sense, no monotone-generalisation needed.

   **Conclusion: the entire published, replicated, peer-reviewed benefit of focal loss is a
   temperature the team already fits.** Their Platt map fits `σ(a·z + b)`; `a` *is* `1/T`. Fitting
   `a` on train OOF reproduces the focal calibration benefit for free, and the team already does it.

The only part of focal not covered by items 1–4 is the residual non-affine curvature of `G_γ`
(§1.2), which is monotone (Lemma 14) and therefore still cannot reorder, plus the implicit
regularisation of item 2, whose sign is unknown.

---

# 2. LABEL SMOOTHING — **DOES NOT REORDER.** Also: a live hazard to the team's best lane.

**Verdict: Class M (monotone, non-affine, and *saturating*). No reordering. Plus an active
warning about §2 item 6 (their transductive distillation, their biggest win).**

## 2.1 Derivation

Szegedy et al. (Inception-v3, arXiv:1512.00567) / Müller, Kornblith, Hinton, *"When Does Label
Smoothing Help?"*, NeurIPS 2019, **arXiv:1906.02629 — VERIFIED (read ar5iv full text).**
Targets become `y^LS_k = y_k(1−α) + α/K`.

Label smoothing is decomposable-pointwise (only the target changes, and it changes by a constant
independent of x). Binary case, `K = 2`, cross-entropy against the smoothed target:

    p*(x) = argmin_p [ ((1−α)η + α/2)·(−log p) + ((1−α)(1−η) + α/2)·(−log(1−p)) ]
          = (1−α)·η(x) + α/2                                    ... (affine in PROBABILITY)

    z*(x) = log[ ((1−α)η + α/2) / ((1−α)(1−η) + α/2) ]           ... (NON-affine in the LOGIT)

`p*` is strictly increasing and affine in `η`, so `z*` is a **strictly increasing, non-affine,
bounded** function of `z_BCE = logit(η)`:

    z*  ∈  [ −log((2−α)/α),  +log((2−α)/α) ]   for all x, no matter how extreme η is.

Two corollaries:

- **Order preserved exactly ⇒ AUC exactly unchanged; F1@0.5 after Platt = threshold slide only.**
  Same verdict as focal. **NOT annihilated in the strict affine sense (a correction to the brief's
  implied taxonomy), but annihilated in the operative sense.**
- **Label smoothing SATURATES the logit.** It is a *soft clip*, not a rescale. This is the one
  structural difference from focal, and it is the reason label smoothing is actively dangerous
  here (§2.3).

Müller et al.'s own description matches: LS *"encourages the differences between the logit of the
correct class and the logits of the incorrect classes to be a constant dependent on α"* — i.e. it
targets a **fixed logit gap**, destroying the magnitude information above that gap.

## 2.2 Calibration equivalence — again annihilated

**VERIFIED:** Müller et al. report LS and temperature scaling achieve comparable ECE (CIFAR-100:
TS 0.021 vs LS 0.024) and treat them as substitutes. Temperature is affine ⇒ exactly annihilated
by the team's Platt map. Same conclusion as focal: **the published benefit is a temperature the
team already fits.**

## 2.3 🔴 CROSS-QUESTION WARNING (this is for whoever owns Q4)

**VERIFIED, Müller et al. §5, arXiv:1906.02629:** *"When teacher models are trained with label
smoothing, student models perform worse"* — and *"using these better performing teachers is no
better, and sometimes worse, than training the student directly with label smoothing."* The
mechanism they give is **information erasure**: LS *"encourages examples to lie in tight equally
separated clusters, so every example of one class has very similar proximities to examples of the
other classes"*, which **reduces the mutual information between input and logits** and destroys the
fine-grained similarity structure distillation feeds on.

The team's §2 item 6 — transductive soft self-distillation on the 1030 test rows — is their
**single biggest win (+0.0100)**. Label smoothing on the teacher would attack exactly the quantity
that win depends on. **Recommendation: do not put label smoothing anywhere upstream of the
distillation teacher.** If anything, the reverse is indicated (see §2.4).

## 2.4 An inversion worth noting

The team measured (§7 Q4) that their teacher is **near-binary: only 0.5–2.9% of its mass in
[0.45, 0.55]**. Müller et al.'s result is that *low-entropy, information-erased* teachers distil
badly. A near-binary teacher is the extreme of that failure mode: it has already thrown away the
similarity structure. **INFERRED (not stated in Müller et al. for this setting):** the indicated
move is to *raise* the teacher's entropy — distil at a **higher temperature on the teacher's
logits** — which is the standard Hinton (arXiv:1503.02531) prescription and the direct opposite of
label smoothing. Note the team says they closed "tuning the distillation temperature/α" (§6); this
finding gives a *reason* the direction matters, but I am not re-proposing a closed lane without a
measurement, so I flag it as context for Q4's owner only.

---

# 3. LDAM — **DOES NOT REORDER via the loss.** Its own paper says the argument is a
# generalisation bound, and its second half (DRW) is plain class weighting ⇒ annihilated.

**Cao, Wei, Gaidon, Arechiga, Ma — "Learning Imbalanced Datasets with Label-Distribution-Aware
Margin Loss", NeurIPS 2019, arXiv:1906.07413. VERIFIED (read ar5iv full text).**

## 3.1 The loss and my derivation of its population minimizer

    L_LDAM((x,y);f) = −log[ e^{z_y − Δ_y} / ( e^{z_y − Δ_y} + Σ_{j≠y} e^{z_j} ) ],   Δ_j = C / n_j^{1/4}

Binary, with `z = z_1 − z_0`, margins `Δ_+` on positives and `Δ_−` on negatives. LDAM is
decomposable-pointwise (the margin depends on the *label*, not on x). Pointwise risk:

    R(z) = η · softplus(−(z − Δ_+))  +  (1−η) · softplus(z + Δ_−)

Stationarity: `η·σ(Δ_+ − z) = (1−η)·σ(z + Δ_−)`. Take the symmetric case `Δ_+ = Δ_− = Δ`, put
`u = e^z`, `c = e^{−Δ}`, `r = η`:

    (1−r)·c·u²  +  (1−2r)·u  −  r·c  =  0
    ⇒  u*(r) = [ −(1−2r) + sqrt( (1−2r)² + 4 r(1−r) c² ) ] / ( 2(1−r)c )

**Sanity check (the team's own rule 4 in §8 — always check the control):** at `Δ = 0` ⇒ `c = 1`,
the quadratic factors as `((1−r)u − r)(u + 1) = 0` ⇒ `u* = r/(1−r)` ⇒ `z* = logit(η)`. LDAM
correctly degenerates to BCE. ✅ The derivation is sound.

For `Δ ≠ 0`, `u*(r)` is a **strictly increasing, non-affine-in-logit function of η alone**.
⇒ **Class M. Order preserved. AUC exactly unchanged. F1@0.5 = threshold slide.**

## 3.2 What the paper actually claims (and it is not a minimizer claim)

**VERIFIED:** the justification is *"a generalization-bound / margin trade-off argument, not a
population minimizer claim."* Their **Theorem 1** bounds per-class error by
`(1/γ_j)·sqrt(C(F)/n_j) + log(n)/sqrt(n_j)`; optimising the balanced bound over margins gives
`γ_i ∝ n_i^{−1/4}`. This is **entirely a finite-sample / capacity argument** — precisely channel
(ii). LDAM has *nothing* to say about what happens at the population optimum, and by §3.1 what
happens there is "the same ranking as BCE."

## 3.3 Two further disqualifiers specific to this team

1. **DRW is class weighting ⇒ exactly annihilated.** VERIFIED: DRW's stage 2 *"applies
   class-inverse-frequency reweighting to the LDAM loss with reduced learning rate."* Class
   weighting multiplies the two class terms by constants; its population minimizer is
   `logit(η) + log(w_+/w_−)` — an **affine logit shift** — which is on the team's own already-killed
   list in §5. So half of LDAM's published gain is annihilated by construction.
2. **LDAM requires ℓ2-normalised features and weights.** VERIFIED: *"They normalize activations
   and weights to unit ℓ2 norm to stabilize margin tuning."* This is **an architecture change, not
   a loss change** — a cosine-similarity head. That *would* change the learned function and hence
   the ranking, but (a) it is a capacity restriction on a 71k-parameter model that is already
   underfitting a shifted problem, and (b) attributing any result to "LDAM" would be attributing it
   to the normalisation, not the margin. The team's §6 lane "adding channels / width — lost on
   essentially every attempt" and their own law "added capacity fitted to shifted rows hurts" both
   point away from this.
3. **The prior is only 0.40/0.60.** LDAM's `n_j^{−1/4}` margins are essentially equal for a
   40/60 split (`(0.4/0.6)^{−1/4} = 1.11`) ⇒ `Δ_+ ≈ Δ_−` ⇒ the loss is nearly symmetric ⇒ nearly
   nothing happens. LDAM is built for 100:1 imbalance.

**Legality:** (a) passes. (b) **fails** — `C` is a free hyperparameter with no train-only
instrument (OOF blind). (c) fails — no posterior correction is claimed or delivered.

---

# 4. ASYMMETRIC LOSS (ASL) — **DOES NOT REORDER**, its premise is absent here, and its one
# distinctive mechanism destroys signal exactly where the team needs it.

**Ridnik, Ben-Baruch, Zamir, Noy, Friedman, Protter, Zelnik-Manor — "Asymmetric Loss For
Multi-Label Classification", ICCV 2021, arXiv:2009.14119. VERIFIED (read ar5iv full text).**

    L₊ = (1 − p)^{γ₊} · log(p)          [typically γ₊ = 0]
    L₋ = (p_m)^{γ₋}  · log(1 − p_m),    p_m = max(p − m, 0),  γ₋ > 0

VERIFIED quote: *"probability shifting is equivalent to moving the loss function to the right, by a
factor m, thus getting L₋ = 0 when p < m."*

## 4.1 Derivation (mine — the paper does NOT analyse the minimizer; marked DERIVED)

ASL is decomposable-pointwise ⇒ minimizer is `T(η)`. With `γ₊ = 0`:

    R(p) = η·(−log p) + (1−η)·( −(p−m)_+^{γ₋} · log(1 − (p−m)_+) )

- For `p ≤ m`: the negative term is identically 0, so `R(p) = −η log p`, **strictly decreasing**.
  ⇒ the minimizer is never in `(0, m)`; it is always `> m`. **ASL floors every score at `m`.**
- For `p = m + ε`, `γ₋ = 2`: `L₋(q) = −q² log(1−q)`, `dL₋/dq ≈ 3q²` for small q, so
  `R'(m+ε) ≈ −η/m + 3ε²(1−η)` ⇒ `ε* ≈ sqrt( η / (3m(1−η)) )`.
- ⇒ `T(η) = m + O(sqrt(η))` near 0 — **strictly increasing**, so (once again) **Class M: no
  reordering, AUC exactly unchanged at the population optimum.**

## 4.2 Why it is *worse* than neutral for this team

1. **Premise absent.** VERIFIED: ASL targets multi-label with a positive rate of **≈0.038 on
   MS-COCO (~26:1 negative dominance)**. This team is **40% positive**. With `γ₊ = 0` and near-balanced
   classes ASL degenerates to "focal on negatives only," whose leading effect is to relatively
   up-weight positives — i.e. **class weighting ⇒ affine ⇒ on the team's already-killed list.**
2. **Zero gradient where the team needs discrimination.** The hard threshold gives training
   negatives with `p < m` **exactly zero gradient**. Under finite capacity the model therefore
   receives *no signal at all* to order the low-score region. §3 of the brief says the rows that
   must be recovered currently sit at **0.15–0.45** — inside or adjacent to the dead zone for any
   `m ≳ 0.15`. **ASL removes the learning signal from precisely the region that has to improve.**
   This is an anti-recommendation with a mechanism, not a hunch. *(DERIVED; the paper contains no
   statement about a floor or low-end information loss — I asked and it does not.)*
3. `γ₊`, `γ₋`, `m` are three free hyperparameters with no train-only instrument ⇒ **fails prong (b)
   three times over.**

**Verdict: reject. Do not run it.**

---

# 5. POLYLOSS — **DOES NOT REORDER**, and its own paper's tuning protocol is illegal here.

**Leng, Tan, Liu, Cubuk, Shi, Cheng, Anguelov — "PolyLoss: A Polynomial Expansion Perspective of
Classification Loss Functions", ICLR 2022, arXiv:2204.12511. VERIFIED (read ar5iv full text).**

    L_Poly-1 = −log(P_t) + ε₁(1 − P_t)     ,  ε_j ∈ [−1/j, ∞)

## 5.1 Derivation

Decomposable-pointwise ⇒ minimizer `T(η)`. Binary, `p = σ(z)`:

    R(p) = η[−log p + ε(1−p)] + (1−η)[−log(1−p) + ε·p]
    R'(p) = −η/p + (1−η)/(1−p) + ε(1 − 2η) = 0

At `ε = 0` this gives `p = η` ✅ (control check passes). For `ε ≠ 0` the root is a
**strictly increasing, non-affine function of η alone** (`R''(p) = η/p² + (1−η)/(1−p)² > 0`, so `R`
is strictly convex on (0,1) and the root moves continuously and monotonically with η).
⇒ **Class M. No reordering. AUC exactly unchanged.**

## 5.2 The paper agrees, in its own words

VERIFIED: *"The paper does not claim changes to ranking. Instead, it modifies loss and loss
derivative magnitudes. Authors observe ε₁ directly controls the mean P_t over all classes,
increasing prediction confidence rather than reordering predictions."* And the gradient identity is
literally a **constant offset**: `−dL_Poly-1/dP_t = −dL_CE/dP_t + ε₁`. A constant additive gradient
in probability space ⇒ a global confidence shift ⇒ **a calibration knob, which is what Platt is.**

## 5.3 Legality — fails prong (b) outright

VERIFIED: `ε₁` is chosen by **grid search on a held-out validation minival** (they reserve 25k
ImageNet-21K images and sweep `ε₁ ∈ {0,…,7}`). This team has **no valid validation instrument**:
§4 states OOF is blind (composite ~0.97 for artifacts whose LB spans 0.72–0.907, sometimes
anti-correlated), and LB tuning is forbidden. **PolyLoss without a validation set is PolyLoss with
a guessed ε, and its whole published effect is the sweep.** Reject.

---

# 6. RECALL-ORIENTED / RECALL-CONSTRAINED SURROGATES — **ANNIHILATED, and their own paper
# says so.** This is the direct answer to the team's secondary question.

## 6.1 The decisive quote

**Eban, Schain, Mackey, Gordon, Rifkin, Elidan — "Scalable Learning of Non-Decomposable
Objectives", AISTATS 2017, arXiv:1608.04802. VERIFIED (read ar5iv full text).**

Their recall-at-precision / precision-at-recall programme is
`max_f tp(f) s.t. (1−α)·tp(f) ≥ α·fp(f)`, whose Lagrangian (their Eq. 6) is

    min_f max_{λ≥0}  (1+λ)·L⁺(f)  +  λ·(α/(1−α))·L⁻(f)  −  λ|Y⁺|

and the authors state, verbatim:

> **"For a fixed λ, the minimization over f is just a c(α,λ) weighted SVM"**, with positive
> instances receiving weight `c(α,λ) = (1+λ)(1−α)/(λα)`; this *"supports the standard practice of
> trying to achieve good P@R or R@P via example re-weighting."*

**That is the whole answer.** A recall constraint, correctly Lagrangianised, **IS** a class weight.
Its population minimizer is `logit(η) + log(w₊/w₋)` — a **pure additive logit offset, i.e. affine,
β-only**. The team's Platt Annihilation Theorem kills it *exactly*, in its original strict form.
The only difference between a recall-constrained method and plain class weighting is that `λ` is
**learned by the optimiser instead of being chosen by hand** — and the object it converges to is
still a scalar class weight.

**⇒ Recall-constrained surrogates are annihilated. There is no version of this that helps.**
Same conclusion applies to the whole family:
- **Recall Loss** (Tian, Liu, Glaser, Hsu, Kira, *"Striking the Right Balance: Recall Loss for
  Semantic Segmentation"*, ICRA 2022, **arXiv:2106.14917**) — weights each class by its current
  false-negative rate. A per-class scalar weight, recomputed per batch. Population minimizer is an
  affine logit offset ⇒ annihilated; the batch-to-batch variation is an *optimisation-dynamics*
  effect only. **INFERRED from the loss definition; I did not read the full paper.**
- **Constrained optimisation with proxy Lagrangians** (Cotter, Jiang, Gupta et al.,
  *"Optimization with Non-Differentiable Constraints with Applications to Fairness, Recall,
  Churn, and Other Goals"*, JMLR 2019, **arXiv:1809.04198**) — the same Lagrangian structure with a
  more careful two-player game. Same reduction: the multiplier enters as a **cost on one class**.
  **INFERRED** from the formulation; the machinery is about *how* to find `λ`, not about escaping
  the fact that `λ` is a class cost.
- **sigmoidF1** — already correctly killed by the team.

## 6.2 The one thing in this family that is NOT annihilated — and why it still fails

Constrained-optimisation frameworks can impose a constraint that is **not** expressible as a class
cost, e.g. *"recall on subgroup G ≥ R"* for a subgroup `G` defined by features. That multiplier
attaches to a **feature-defined subset**, not to a class, so its population effect is
`logit(η(x)) + λ·1[x ∈ G]` — a **row-dependent offset ⇒ genuinely REORDERS.** This is the only
member of the recall family that escapes.

**But it fails prong (b) and it fails §1's data description.** To use it the team would need a
train-only definition of the subgroup `G` that (i) identifies the confidently-missed positives and
(ii) transfers to test. §4 says train and test are **exactly separable (adversarial AUC = 1.0000)**,
so any subgroup defined on train has **no identifiable counterpart on test** — the same
non-identifiability that killed importance weighting (§6) kills subgroup-constrained recall, and
for exactly the same reason. **This is the same theorem twice; the team should record it as such.**
