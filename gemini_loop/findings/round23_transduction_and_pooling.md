# Round 23 findings — Q4 (transductive distillation) and Q5 (F1-oriented pooling)

Agent scope: **Q4 and Q5 only**. Written incrementally; if this file ends abruptly the run was cut off.

Started: 2026-08-14.

Labels used throughout:
- **VERIFIED** = I fetched and read the paper (arXiv abs/full text or publisher page) and the claim is stated there.
- **INFERRED** = derived by me from things I verified, or read only at abstract level.
- **UNSOURCED CONJECTURE** = my own reasoning, no citation.

---

## Status log

- [ ] Q4a independent teacher
- [ ] Q4b recall-targeted distillation
- [ ] Q4c saturated teacher / temperature
- [ ] Q4d Kumar-Ma-Liang verification, "one round only"
- [ ] Q5 Ranjan-Gneiting verification
- [ ] Q5 majority vote analysis
- [ ] Q5 decision-level fusion

(Sections appended below as they complete.)

---

# CORRECTION 1 (highest value) — the "one round only" cap rests on a mis-cited theorem

**Brief §2 item 6 says:** *"Run for exactly one round (Kumar/Ma/Liang: error compounds per self-training step)."*

**This citation does not support that cap.** Kumar, Ma & Liang, *Understanding Self-Training for
Gradual Domain Adaptation*, ICML 2020, **arXiv:2002.11361** (PMLR v119, kumar20c).
**VERIFIED (read the ar5iv full text of arXiv:2002.11361).**

What the paper actually proves:

- **Theorem 3.2:** `L_r(θ′, Q) ≤ [2/(1 − ρR)] · L_r(θ, P) + α* + [4BR + √(2 log(2/δ))]/√n`
- **Corollary 3.3:** `L_r(θ, P_T) ≤ β^(T+1) · ( α₀ + [4BR + √(2 log(2T/δ))]/√n )`, with `β = 2/(1 − ρR)`.
  The paper summarises this as *"the error of gradual self-training is ≲ exp(cT)·α₀."*

**The index `T` counts INTERMEDIATE DOMAINS `P₀ → P₁ → … → P_T`, not self-training rounds on a fixed
target.** Each step in that recursion is *one* self-training pass onto a *new, slightly-shifted*
unlabeled distribution. The exponential is the price of *traversing a shift path*, not the price of
*re-teaching on the same 1030 rows*. Applying `β^(T+1)` to "how many times do I distill onto Test.csv"
is a category error: there is only ever one domain here, so `T = 1` in their indexing no matter how
many times you re-distill.

Two further things in that same paper cut **against** the team's current setup, and one cuts **for** it:

1. **Against the whole lane, not just round 2.** Their **Example 3.1** is exactly this team's
   situation — a single large jump with no intermediate domains — and it shows
   *"Self-training directly on the target does not help: `L_r(ST(θ,P₂), P₂) = 1`."* Worst-case theory
   says the team's +0.0100 should not have happened at all. It did. So this paper's worst case is
   **not binding at their scale** and cannot be used to cap rounds either way. (See Q4d below.)
2. **For hard/sharp targets.** The paper explicitly states that **label sharpening is essential**:
   *"soft labels … may never update the parameters"*, and it recommends hard labels. The team's §7-Q4
   worry that "our teacher is near-binary so soft distillation ≈ hard pseudo-labelling" is, under the
   only theory they cite, a **feature rather than a defect**. Their teacher being saturated is what
   makes self-training move the parameters at all.
3. **Regularization is the load-bearing ingredient**, not softness: *"regularization is important for
   maintaining a margin even with infinite data."*

**The correct citation for a round cap** is Mobahi, Farajtabar & Bartlett, *Self-Distillation Amplifies
Regularization in Hilbert Space*, NeurIPS 2020, **arXiv:2002.05715**.
**VERIFIED (read the abstract + mechanism on arXiv).** Abstract: *"while a few rounds of
self-distillation may reduce over-fitting, further rounds may lead to under-fitting and thus worse
performance."* Mechanism: each self-distillation round **progressively restricts the number of basis
functions** available to represent the solution, i.e. it monotonically amplifies the effective
regulariser. Setting: fitting a nonlinear function in a Hilbert space under ℓ₂ regularization in
function space (regression, closed form) — *in-distribution*, no shift.

So the honest statement of the cap is:

> Repeated self-distillation on a fixed target is a **regularization ladder with an interior optimum**
> (Mobahi et al., arXiv:2002.05715), not an error-compounding cascade (Kumar et al., arXiv:2002.11361).
> The optimum is at *some* finite number of rounds ≥ 1; nothing the team has cited pins it at exactly 1.

**Why this matters practically:** the failure mode of round 2 under the correct theory is
**under-fitting**, i.e. the student becomes *smoother / lower-variance / less confident* than the
teacher. Under-fitting a saturated teacher **de-saturates** the score distribution and moves mass back
toward 0.5. Given that this team's stated blocker (§3) is "there are not 23 rows near the boundary",
a *controlled* second round is one of the very few legal mechanisms that puts rows back near the
boundary at all — and, unlike a threshold move, it does so by changing the function, not the cut.
See Q4d for the caveat that direction is not guaranteed.

---

# Q4(a) — Is there a second, MODEL-INDEPENDENT teacher?

Short answer: **yes, and this team has an unusually clean one available that nobody seems to have
noticed — a genuine two-view split (SAR vs optical) that satisfies the co-training precondition
better than almost any tabular problem does.** Ranked options below, strongest first.

## Option A1 — Co-training on the SAR / optical view split (RECOMMENDED)

**Source.** Blum & Mitchell, *Combining Labeled and Unlabeled Data with Co-Training*, COLT 1998,
**DOI 10.1145/279943.279962**. **INFERRED at the level of the classical statement** (I did not re-read
the 1998 proof this run; the conditional-independence + each-view-sufficient precondition is
textbook). The precondition: the instance splits as `x = (x⁽¹⁾, x⁽²⁾)`, each view is *sufficient* for
the label, and the two views are *conditionally independent given the label*. Then a classifier on
view 1 can label examples for a classifier on view 2 and the pair provably boosts a weak initial
classifier using unlabeled data alone.

**Why it fits here, specifically.** The 12 bands in §1 are **two physically distinct sensors**:

- View S (SAR): `VH, VV` — Sentinel-1, active C-band microwave, decibels.
- View O (optical): `blue, green, red, re1, re2, re3, nir, nira, swir1, swir2` — Sentinel-2, passive
  reflectance.

These are not two arbitrary column subsets. They are different physics, different satellites,
different acquisition geometry, and different noise processes (speckle vs atmospheric/cloud).
Conditional independence given "is this an aquaculture pond" is *not exact* — both are driven by the
same water surface — but the *error processes* are close to independent, which is what the co-training
argument actually needs.

**Why it is genuinely model-independent, which is the thing Q4 asks for.** The team's champion is
SAR-dominated: its two best hand features are `1[VH_dB < −21]` (permanence) and the dual-pol gate
`1[VH<−21]·1[(VH−VV)<−8]` (§2 items 4–5). A teacher trained on **view O only has never seen VH or
VV**. It therefore *cannot* re-consume the champion's SAR bias — this is stronger independence than
"a different seed" or "a different architecture", both of which still see the same columns. This is
the precise sense in which round 2 stops being "the model re-eating its own errors".

**Relaxed precondition (important — the independence assumption is known to be too strong).**
Balcan, Blum & Yang, *Co-Training and Expansion: Towards Bridging Theory and Practice*, NIPS 2004
(**[NeurIPS 2004 proceedings]**, also DBLP conf/nips/BalcanBY04). **INFERRED (abstract-level).** They
replace conditional independence with a much weaker **ε-expansion** condition on the underlying
distribution: informally, for any partition, the "confident on view 1 but not view 2" region has
non-trivial probability. This is the theoretically defensible version to cite in the report, because
SAR and optical are *not* conditionally independent and the team will be asked about it.

**Legality analysis (§5 three prongs + Platt annihilation).**
- **(a) literal 0.5** — PASSES. Co-training changes what function is learned. The final student is
  still Platt-calibrated on train OOF and cut at a literal 0.5. No operating point is touched.
- **(b) train-only knobs** — PASSES *if and only if* the pseudo-label acceptance rule is a fixed
  agreement rule with a confidence level fixed a priori or by a train-only criterion, and **not** a
  "take the top-k% most confident" or "keep the predicted positive rate at r" rule. **Class-balanced
  pseudo-label selection (CBST, Zou et al., arXiv:1810.07911) is DISQUALIFIED for this team** — its
  class-wise quantile selection is literally a per-class threshold fit to a target positive rate.
  Use *agreement + fixed confidence*, never *quantile*.
- **(c) corrects p(y|x) under a mis-specified model** — PASSES, and this is its strongest prong. The
  view-O teacher is a *different hypothesis class over a disjoint feature set*. Where it disagrees
  with the SAR-driven student, the disagreement is evidence about model mis-specification, not
  relabeling of a fixed estimate.
- **Platt annihilation** — SURVIVES. Co-training changes which function is fit, hence the **ranking**
  of test rows, not the affine placement of a fixed logit. `σ(a·z_new + b)` with a refit `(a,b)`
  cannot recover `z_old` because `z_new` is not an affine image of `z_old`.

**The specific prediction that makes it worth trying.** The rows this team needs (§3: confidently-low
true positives, not boundary rows) are, by construction, rows where the **SAR view is confidently
wrong**. Ponds whose VH never drops below −21 dB in the observed 4–6 month window — e.g. shallow,
vegetated, wind-roughened, or recently drained ponds — are exactly the SAR failure mode. An optical
teacher (turbidity / chlorophyll / red-edge structure) is the sensor that *can* see those. This is
the only mechanism I found in the whole of Q4/Q5 whose failure mode is aligned with their measured
deficit rather than orthogonal to it.

**Falsifier / cost.** Train a view-O-only model on train (12 optical months → masked windows, same
recipe). Measure its **standalone** train-OOF AUC and, more importantly, the **disagreement set** on
the 1030 test rows: `D = {i : SAR-student says negative confidently (p<0.45) AND optical teacher says
positive confidently (p>0.55)}`. If `|D| < 20`, the lane cannot deliver the ~23 flips §3 needs and
should be abandoned immediately — this is a **free, LB-free, one-hour measurement**. If `|D| ∈
[25, 80]`, proceed. **Do this measurement before writing any training code.**

⚠️ Note against §6: this is **not** "blending a weaker decorrelated member" (closed, lost 3×). The
optical model is never *blended into the score*. It is used only to *manufacture targets* for a
second distillation round, after which it is discarded. The failure mode of blending (a weak member
drags the pooled score) does not apply; the failure mode here is bad pseudo-labels, which the
agreement rule bounds.

## Option A2 — Tri-training / asymmetric tri-training (the standard formal answer to Q4a)

**Source (classical).** Zhou & Li, *Tri-Training: Exploiting Unlabeled Data Using Three Classifiers*,
IEEE TKDE 17(11):1529–1541, 2005, **DOI 10.1109/TKDE.2005.186**. **INFERRED (abstract-level).** The
defining property is exactly what Q4 asks for: a point is pseudo-labeled **for** classifier `h_i`
only when the **other two** classifiers `h_j, h_k` agree on it. The teacher for any given student is
by construction a function of models that student is not. It removes the need for two views.

**Source (the domain-adaptation version, and the better one here).** Saito, Ushiku & Harada,
*Asymmetric Tri-training for Unsupervised Domain Adaptation*, ICML 2017, **arXiv:1702.08400**.
**VERIFIED (read the ar5iv full text).** Exact mechanism:

- Loss: `E(θ_F, θ_F1, θ_F2) = (1/n) Σ_i [ L_y(F1∘F(x_i), y_i) + L_y(F2∘F(x_i), y_i) ] + λ|W1ᵀW2|`
- The regulariser is **λ·|W1ᵀW2|** on the first fully-connected layers of the two labeling heads. Its
  stated purpose: it *"penalizes similarity between the weight matrices"* so the two heads *"learn
  from different feature representations, creating distinct views of the data."* **This is a
  synthetic view split** — it manufactures the co-training precondition when no natural one exists.
- Acceptance rule: pseudo-label accepted only when **`C1 = C2`** (*"two different classifiers agree
  with the prediction"*) **AND** the max probability exceeds a threshold **set to 0.9 or 0.95**
  (0.95 for the hardest shift, MNIST→SVHN).
- Third network `F_t` is trained **only** on pseudo-labeled target data → target-discriminative.

**Legality.** Same as A1 on prongs (a) and (c). Prong (b): the 0.9/0.95 confidence level is a
**published default, not a tuned quantile** — cite Saito et al. for the value and do not tune it.
That is the clean way to keep it train-only. **Caution:** the agreement+confidence rule *does* change
how many rows get pseudo-labeled, but it does **not** change the final decision rule, which remains
`σ(a·z+b) > 0.5`. It is not a threshold move.

⚠️ Note the `|W1ᵀW2|` orthogonality penalty is a **weaker** independence guarantee than A1's physical
sensor split. If the team can only do one, do A1.

## Option A3 — Prototype/centroid teacher from target data only (SHOT), with one term REMOVED

**Source.** Liang, Hu & Feng, *Do We Really Need to Access the Source Data? Source Hypothesis Transfer
for Unsupervised Domain Adaptation (SHOT)*, ICML 2020, **arXiv:2002.08546**.
**VERIFIED (read the ar5iv full text).** Exact equations:

- Centroids (Eq. 4): `c_k⁽⁰⁾ = Σ_{x_t∈X_t} δ_k(f̂_t(x_t))·ĝ_t(x_t) / Σ_{x_t∈X_t} δ_k(f̂_t(x_t))`
  — a softmax-weighted feature mean, i.e. weighted k-means on the **target** features.
- Labels (Eq. 5): `ŷ_t = argmin_k D_f(ĝ_t(x_t), c_k⁽⁰⁾)`, **cosine distance**, nearest centroid.
- Rounds: *"we update the centroids and labels in Eq. (6) for multiple rounds. However, experiments
  verify that updating for **once** gives sufficiently good pseudo labels."* — an independent,
  published "one round is enough" data point, but for the **centroid** loop, not the distillation loop.
- Ablation (Table 6), Office / Office-Home / VisDA-C: source-only 79.3 / 60.2 / 46.6 → IM only
  87.3 / 70.5 / 80.4 → full SHOT (IM + pseudo-label) 88.6 / 71.8 / 82.9. **Read this carefully: the
  centroid pseudo-labeling contributes only +1.3 / +1.3 / +2.5 points on top of IM. Almost all of
  SHOT's gain is the IM loss, not the independent teacher.**

**⛔ LEGALITY — ONE TERM OF SHOT IS DISQUALIFIED FOR THIS TEAM.** The IM loss has two parts:
- `L_ent = −E_x Σ_k δ_k(f_t(x)) log δ_k(f_t(x))` — entropy minimisation. Legal-ish (see below).
- `L_div = Σ_k p̂_k log p̂_k = D_KL(p̂, (1/K)·1_K) − log K` — **this term drives the predicted class
  marginal toward UNIFORM**. For K=2 that is a hard pull of the predicted positive rate toward 0.500.
  That is *exactly* the "pinning the predicted positive rate" manoeuvre the team disclosed as a rules
  violation in §1. **It fails prong (b) outright and must not be used.** Worse, its direction here is
  harmful: it would drag the operating rate from 0.583 **down** toward 0.500, i.e. fewer positives —
  the opposite of what §3 says they need.

  Since the ablation shows IM carries nearly all of SHOT's gain and `L_div` is half of IM, **the
  legal residue of SHOT is small**: entropy minimisation plus a +1.3-to-+2.5-point centroid teacher.
  I rank SHOT **below** A1 and A2 for this reason.

**And a second problem with entropy minimisation specifically for this team.** `L_ent` *sharpens* the
score distribution — it pushes mass away from 0.5. Their teacher is already saturated (0.5–2.9% of
mass in [0.45, 0.55]). Entropy minimisation makes saturation *worse*, and §3 says their blocker is
that there is nothing near the boundary to move. **`L_ent` is directionally wrong for this problem.**
This is worth stating in the report as a reasoned rejection.

**Prong (c):** the centroid teacher does partially pass — it is a *generative/prototype* hypothesis
(nearest-centroid in feature space) rather than the discriminative head, so it is not "relabeling a
fixed estimate". But it reuses the student's own features `ĝ_t` and its own softmax as the weights,
so it is **not model-independent** in the strong sense Q4 asks for. Call it *head-independent*, not
*model-independent*. A1 is the only strongly-independent option.

---

# Q4(b) — Is there a distillation variant that targets RECALL on the target distribution?

**Answer: yes, one exists and is published — and it is ANNIHILATED by your Platt step. I can prove
it, and the proof also retrodicts a null result you already measured.** This is the strongest
negative in my scope, and I think it should go in the report.

## The published method

Wang, Zhou et al., *Knowledge Distillation with Adaptive Asymmetric Label Sharpening for
Semi-supervised Fracture Detection in Chest X-rays*, IPMI 2021, **arXiv:2012.15359**,
DOI 10.1007/978-3-030-78191-0_46. **VERIFIED (fetched the arXiv HTML; formula quoted below).**

It is designed for **exactly your problem statement**: *"the teacher model, biased towards negatives,
might produce low-sensitivity pseudo-labels for true positives … the pseudo ground truth tends to
have low sensitivity (low probabilities at positive sites)."* That is your §3 diagnosis verbatim, in
a different field.

Its sharpening operator, applied to the teacher's soft label `y'`:

```
S(y') = expit( a · logit(y')  +  (1 − a) · logit(t) )
```

with `expit = σ`, `a` = sharpening strength, `t` = sharpening centre; adaptive strength
`a = a₀ − (a₀ − 1)·y'_max`; and the paper states *"since the asymmetric sharpening aims to enhance
low probabilities, `t < 0.5` should be used (`t = 0.4` is used in the experiments)."* The final
operator is `max(S(y'), y')`.

## Why it is annihilated — the derivation

Write the teacher logit `z_T = logit(y')`. Then

```
logit(S(y')) = a·z_T + (1 − a)·logit(t)          ← an EXACTLY AFFINE map in z_T
```

This is `z' = αz + β` with `α = a`, `β = (1−a)·logit(t)` — **the literal canonical form of your §5
Platt Annihilation Theorem**. The `t = 0.4` centre is not a new mechanism; it *is* a decision-centre
choice, i.e. a threshold move written in probability space.

The `max(S(y'), y')` clamp does not save it. Both branches are strictly increasing in `y'`, so
`max(S(y'), y')` is a **monotone** reparameterization of the teacher score, not merely an affine one.
Under soft-target cross-entropy the pointwise Bayes-optimal student output *is* the target value, so
in the well-specified / high-capacity limit the student's score is a monotone transform of `z_T`,
hence **rank-identical to the teacher**. Your pipeline then applies Platt (monotone) and cuts at 0.5,
so the whole composition reduces to `z_T > c` for some constant `c` — and `c` is set by the Platt fit
on train OOF, not by AALS. **The method cannot change your submission.**
**Label: INFERRED (derivation is mine; the inputs — the AALS formula and your Platt step — are both
verified/given).**

## The general theorem this suggests (state it in your report)

> **Monotone-Target Annihilation.** Let a distillation scheme differ from soft self-distillation only
> by replacing the teacher's target `y'ᵢ` with `g(y'ᵢ)` for a fixed strictly-increasing
> `g : (0,1) → (0,1)`. In the capacity-unconstrained limit the student's score is a monotone image of
> the teacher's score, so the induced ordering of test rows is unchanged. Composing any monotone
> calibrator and a fixed cut therefore yields a decision of the form `z_T > c`, with `c` determined by
> the calibrator alone. The scheme is annihilated.

**Corollaries you can check against your own logs:**
1. **Teacher temperature is the special case `a = 1/T`, `t = 0.5`.** So your §6 entry *"Tuning the
   distillation temperature/α — closed over a 5× range of α; the whole ladder was one public true
   positive"* is not an unlucky measurement. **It is the predicted result.** Temperature *could not*
   have moved anything. This converts an empirical null into a theoretical one — a genuinely stronger
   report claim. (Note `α`, the *mixing weight* between the supervised and distillation terms, is
   **not** covered by the theorem: it is a regularization-strength knob, not a target
   reparameterization. That half of your ladder was a real test and it came back null on its own.)
2. **Class-asymmetric distillation weights, focal-weighted distillation on the teacher probability,
   "positive-boosted" soft labels — all of the recall-oriented KD family I could find reduces to a
   monotone `g`.** They are all annihilated.

## What is NOT annihilated, and it is a short list

The theorem's escape hatches are exactly three, and only three:

1. **Change the teacher's hypothesis class or feature set** ⇒ the ordering changes. This is Q4(a),
   Option A1 (SAR/optical co-training). *This is why A1 is my top recommendation for Q4.*
2. **Change the SET of rows that are distilled on** (which rows carry a target, and with what weight
   that depends on something other than the teacher's own score). Weighting by a quantity independent
   of `z_T` — e.g. the *cross-view disagreement* from A1, or the per-row *seed variance* of the pool —
   is not a function of `z_T`, so it is not a monotone reparameterization and it does reorder.
3. **Constrain the student's capacity / regularization** ⇒ the monotone-limit argument fails and the
   student's ordering genuinely differs from the teacher's. This is the Mobahi et al. mechanism
   (arXiv:2002.05715) and it is the honest reason round 1 gave +0.0100 at all.

**⛔ One recall-oriented family to explicitly rule out on legality, not on annihilation.** Class-balanced
self-training / curriculum pseudo-labelling (Zou, Yu, Kumar & Wang, ECCV 2018, **arXiv:1810.07911**;
CRST, ICCV 2019, **arXiv:1908.09822**) raises recall by selecting pseudo-labels at a **per-class
quantile** of the confidence distribution. **INFERRED (abstract-level).** That is a direct
prevalence knob: it fixes the predicted positive rate by construction. It **fails prong (b)** and is
the same manoeuvre you already disclosed and retracted. Do not use it. Also do not use FixMatch-style
fixed-confidence thresholds *for the positive class only*.

---

# Q4(c) — Distilling from a SATURATED teacher: what the literature says

Your measurement (0.5–2.9% of teacher mass in [0.45, 0.55]) is diagnostic of something the KD
literature has studied directly, and the news is bad for the current setup but points somewhere
specific.

## Finding C1 — a saturated teacher carries no distillable information beyond its hard labels

Müller, Kornblith & Hinton, *When Does Label Smoothing Help?*, NeurIPS 2019, **arXiv:1906.02629**.
**VERIFIED (abstract + NeurIPS slide deck).** The relevant result: *"if a teacher network is trained
with label smoothing, knowledge distillation into a student network is much less effective"*, because
smoothing *"encourages the representations of training examples from the same class to group in tight
clusters, which results in loss of information in the logits about resemblances between instances of
different classes, which is necessary for distillation."*

The mechanism is **collapse of within-class logit spread**. Your teacher exhibits the same symptom
from the opposite cause (over-training / seed agreement rather than smoothing). Read contrapositively:
**the informativeness of a soft teacher is carried by the *spread* of its logits, and your pool has
almost none.** The +0.0100 you got from round 1 therefore did **not** come from "dark knowledge" — it
came from *hard pseudo-labels acting as a transductive regulariser on the target support*. That is
consistent with Kumar et al.'s *"label sharpening is essential"* (Correction 1 above) and with Yuan
et al., *Revisiting Knowledge Distillation via Label Smoothing Regularization*, CVPR 2020,
**arXiv:1909.11723** (**INFERRED, abstract-level**), which shows KD's benefit is largely a
regularization effect that survives even with a poorly-trained or near-random teacher.

**Practical consequence:** stop treating the softness as the active ingredient. It isn't. Any effort
spent making the teacher softer (temperature, smoothing, entropy penalties) is spent on the wrong
variable — and per Q4(b) it is annihilated anyway.

## Finding C2 — the published remedy for an over-confident teacher is EARLY-STOPPING it, not softening it

Cho & Hariharan, *On the Efficacy of Knowledge Distillation*, ICCV 2019, **arXiv:1910.01348**.
**VERIFIED (abstract-level).** Findings: *"more accurate teachers often don't make good teachers"*;
capacity mismatch means *"small students [are] unable to mimic large teachers"*; and the remedy they
validate is **stopping the teacher's training early**. They also report that *"typical ways of
circumventing this problem (such as performing a sequence of knowledge distillation steps)"* are
**ineffective** — see Q4(d), this cuts against multi-round.

**Legality of an early-stopped teacher (§5):**
- **(a)** PASSES — no operating point is touched.
- **(b)** PASSES *only if* the stopping epoch is chosen by a train-only criterion. ⚠️ **Your §4 says
  OOF is blind here**, so you have no honest instrument for picking the epoch. That is a real blocker.
  The defensible version is a **pre-registered fixed fraction** (e.g. teacher = 60% of the champion's
  epoch budget), declared before running, not selected.
- **(c)** PASSES — an early-stopped network is a genuinely different function.
- **Platt annihilation:** SURVIVES. An early-stopped model is not a monotone reparameterization of the
  fully-trained one; it reorders rows.
- **⚠️ Cost:** the teacher's AUC will drop, and your metric pays 0.4× for AUC. Since your AUC already
  *beats* the leader by only +0.000945, you have almost no AUC headroom to spend. **I rank this
  BELOW A1** for that reason, and I would not spend a submission on it.

## Finding C3 — the right response to saturation is to distill the ENSEMBLE'S SPREAD, not its mean

Malinin, Mlodozeniec & Gales, *Ensemble Distribution Distillation* (EnD²), ICLR 2020,
**arXiv:1905.00076**. **VERIFIED (abstract-level).** Core claim: *"Information about the diversity of
the ensemble … is lost in traditional ensemble distillation approaches"*, because standard
distillation *"collapses an ensemble of conditional distributions over classes into a single
point-estimate."* EnD² instead fits a **Prior Network** parameterizing a **Dirichlet** over the
categorical simplex — for K=2 this is a **Beta distribution over the per-row probability** — thereby
preserving the member-to-member spread.

**Why this is the most interesting Q4(c) option and why it is NOT a temperature move.** Your teacher
target is currently `p̄ᵢ = mean over 10 seeds`. Averaging 10 saturated members produces a saturated
mean *and destroys the one genuinely informative signal you have*: **which rows the seeds disagree
about**. Per-row seed disagreement is (i) not a function of `p̄ᵢ`, so per Q4(b) escape hatch #2 it is
**not a monotone target reparameterization and is not annihilated**; and (ii) it is a pure
*epistemic* signal that is highest exactly on rows the model is unsure about *as a family* — which,
on a support-disjoint target, is the closest legal proxy you have for "rows where the model is
mis-specified" (prong (c)).

**Legality (§5):** (a) PASSES — the student still emits a probability, Platt-calibrated, cut at 0.5.
(b) PASSES — the target is the empirical Beta moments of your own 10 seeds on the test rows; no
LB quantity, no positive-rate target, no threshold. (c) PASSES strongly — it is the only construct in
my scope that explicitly represents *model uncertainty* rather than relabeling a point estimate.
**Platt annihilation: SURVIVES**, per escape hatch #2.

**⚠️ THE FREE MEASUREMENT THAT DECIDES C3 (do this first, it costs nothing and no submission).**
Compute, over the 1030 test rows, the per-row standard deviation of the probability across your 10
seeds, `sᵢ = sd(p_i1 … p_i10)`.

- If **`sᵢ ≈ 0` for nearly all rows** — i.e. the 10 seeds essentially agree — then your "10-distinct-seed
  pool, probability-averaged" (§2 item 7) is arithmetically almost a single model, C3 has no signal to
  distill, **and Q5 collapses too** (all pooling operators coincide when members agree; see Q5 below).
  That would be the most important single fact in this round, and it is one line of numpy.
- If `sᵢ` is materially non-zero on a few dozen rows, those rows are your candidate set — and you can
  check immediately whether they overlap the ~23 rows §3 says must flip.

I flag this because your own measurement (only 1.5% of the *pooled* mass in [0.45,0.55]) is
**suspicious for near-total seed agreement**: a mean of 10 mutually-disagreeing saturated members
would be quantized to multiples of 0.1 and would necessarily put substantial mass at 0.4/0.5/0.6.
It does not. **UNSOURCED CONJECTURE, but it is arithmetic and you can falsify it in a minute.**

---

# Q4(d) — Verify or refute the "one round only" cap

**Verdict: the cap is not supported by the cited theorem (Correction 1), and it is not binding at your
scale either — but I am NOT recommending you lift it, because the independent evidence is split and
you have two days. The right move is to spend the round on an INDEPENDENT teacher (Q4a), not on a
second round of the same teacher.**

## d1. The cited bound is *vacuous*, not *binding*, at adversarial AUC 1.0

From Corollary 3.3 (**VERIFIED**, arXiv:2002.11361):
`L_r(θ, P_T) ≤ β^(T+1)·( α₀ + [4BR + √(2 log(2T/δ))]/√n )`, with **`β = 2/(1 − ρR)`**, `ρ` the
Wasserstein-∞ shift magnitude and `R` a model-complexity bound.

- **The bound requires `ρR < 1`.** Your §4 reports a cross-fitted adversarial AUC of **1.0000** on all
  144 features — *disjoint support*. That is the regime where `ρ` is large. If `ρR ≥ 1`, then
  `1 − ρR ≤ 0` and `β` is negative or undefined: **the theorem does not produce a bound at all.** You
  cannot invoke it to cap rounds, and equally you cannot invoke its absence as permission. It is
  simply silent about you. **INFERRED (my reading of the quoted formula; the inequality condition
  `ρR<1` is forced by the algebra).**
- **The `T`-dependence at your `n` is negligible where it exists.** The only place `T` appears outside
  `β` is `√(2 log(2T/δ))/√n`. At `n = 1030`, `δ = 0.05`: `T=1` → ≈ 0.085; `T=3` → ≈ 0.091. **A 0.006
  change in a loss bound.** So even taking the theorem at face value, *the finite-sample cost of a
  second round at 1030 rows is arithmetically nil*. All the cost lives in `β^(T+1)` — which, per
  Correction 1, prices **domain traversal**, and you traverse exactly one domain no matter how many
  times you re-distill.

## d2. The independent evidence on multi-round self-distillation is genuinely split

**Against more rounds:**
- Mobahi, Farajtabar & Bartlett, **arXiv:2002.05715** (NeurIPS 2020). **VERIFIED.** *"while a few
  rounds of self-distillation may reduce over-fitting, further rounds may lead to under-fitting and
  thus worse performance."* Mechanism: each round monotonically restricts the basis. Note: *"a few"*,
  not *"one"*. Their setting is in-distribution ℓ₂-regularized regression in a Hilbert space.
- Cho & Hariharan, **arXiv:1910.01348** (ICCV 2019). **VERIFIED (abstract).** They tested *"performing
  a sequence of knowledge distillation steps"* as a remedy for teacher over-confidence and found it
  **ineffective**. This is the closest published test to "does a second round help when the teacher
  is too confident?", and the answer was no.
- Arazo, Ortego, Albert, O'Connor & McGuinness, *Pseudo-Labeling and Confirmation Bias in Deep
  Semi-Supervised Learning*, IJCNN 2020, **arXiv:1908.02983**. **INFERRED (abstract-level).**
  Confirmation bias: the network reinforces its own mistakes across pseudo-label rounds; their
  remedies are **mixup** and a **minimum-unlabeled-samples-per-minibatch** constraint — i.e. the fix
  is *input-space regularization*, not fewer rounds.

**For more rounds:**
- Furlanello et al., *Born-Again Neural Networks*, ICML 2018, **arXiv:1805.04770**.
  **VERIFIED (abstract-level).** Students *identically parameterized to their teachers* trained on the
  teacher's outputs **outperform the teachers**, and the procedure is explicitly **sequential over
  generations**, each re-initialized from a **different random seed**, with *"additional gains … with
  an ensemble of multiple student generations."* This is direct empirical refutation of a hard
  one-round cap in the in-distribution case. ⚠️ Note the BAN protocol re-initializes each generation
  from a fresh seed — that is the diversity mechanism, and it is cheap for you.
- Xie, Luong, Hovy & Le, *Self-training with Noisy Student*, CVPR 2020, **arXiv:1911.04252**.
  **INFERRED (well-known result, abstract-level).** Three iterations of teacher→student, each
  improving, with **noise injected into the student** (dropout / augmentation / stochastic depth) as
  the essential ingredient. The rule of thumb it establishes: *a second round pays only if the student
  is noisier than the teacher was.*
- **arXiv:2406.11206**, *Retraining with Predicted Hard Labels Provably Increases Model Accuracy*.
  **VERIFIED (read the abstract).** Proves, in linearly separable binary classification with randomly
  corrupted labels, that **retraining on the model's own predicted hard labels provably improves
  population accuracy**. Also introduces **consensus-based retraining** — retrain only where the
  predicted label *matches* the given label. ⚠️ Scope caveats that matter for you: linearly separable,
  **in-distribution**, and about *label noise*, not covariate shift. It is a genuine counterweight to
  "self-training compounds error", but it is not about your shift.

## d3. My reading of the split, applied to your case

The two camps differ on one variable: **whether the student differs from the teacher by something
other than the targets.** BANs differ by seed. Noisy Student differs by injected noise. Mobahi's
collapsing case and Cho–Hariharan's failed sequential KD do not. **A second round with a
*re-seeded, noisier* student is the version with published support; a second round with the same
recipe is the version with published failure.** **INFERRED (my synthesis).**

This is the same conclusion Q4(b)'s Monotone-Target-Annihilation theorem reaches from the other
direction: the target values are not where the leverage is. **The leverage is in making round 2's
learner different from round 1's** — different view (A1), different seeds (BAN), or different noise
(Noisy Student). And of those three, **only A1 attacks the failure mode §3 actually identifies**
(SAR-confident-wrong positives).

## Legality of a second round (§5), for completeness

- **(a)** PASSES. Rounds do not touch the cut.
- **(b)** PASSES *only if the number of rounds is pre-registered*, not selected. ⚠️ **This is the real
  binding constraint on you, and it is a legality constraint, not a theory constraint.** With OOF
  blind (§4) and LB feedback forbidden, **you have no legal instrument to choose between 1 and 2
  rounds after the fact.** Mobahi says the optimum is at *some* finite round count; you cannot
  measure which. Therefore the honest position in your report is:
  > *"The round count is unidentifiable under our compliance regime. We pre-registered T=1 on the
  > (mis-attributed) Kumar et al. bound. The correct attribution is Mobahi et al., which predicts an
  > interior optimum we cannot legally locate. We therefore retain T=1 as a pre-registered choice,
  > not as a theoretically-forced one."*
  That is a defensible, honest paragraph and it is worth report credit.
- **(c)** A second round of the *same* teacher **FAILS prong (c)**: it relabels a fixed estimate rather
  than correcting `p(y|x)` under a mis-specified model. A second round with an **independent** teacher
  (A1) passes (c).
- **Platt annihilation:** a same-teacher second round survives annihilation formally (the student's
  function changes non-affinely) but per Q4(b) the *targets* contribute nothing new; only the
  regularization effect does.

**Bottom line for Q4:** keep the cap at one round *of self-distillation*, and spend the second round
on a **cross-view teacher** instead. That is not "round 2"; it is "round 1 of a different teacher".

---
