# Round 22 — Offline Transfer Instruments: can anything predict LB rank without spending submissions?

## STATUS: COMPLETE

**Target claim under refutation:** "No offline instrument can predict which of our candidate
models will transfer to the leaderboard, so we must spend scarce submissions to learn anything."

**Date:** 2026-08-13. Deadline 2026-08-16.

---

## 0. LEAD DELIVERABLE — the back-test protocol to run first

**VERDICT UP FRONT.** The claim survives in its strong form but is FALSE in its literal form, and
the difference is worth a submission. Correct statement:

> No offline instrument can predict our LB *score*. That is a published impossibility result
> (Garg et al., ICLR 2022, Proposition 1 + Corollary 1) and our data violates both escape hatches
> (label shift, covariate shift). **But "predict the score" is not what we need.** Ranking, and in
> particular *vetoing a catastrophe*, is a strictly weaker question, it is empirically testable
> today at zero submission cost, and the literature evaluates exactly this quantity by rank
> correlation. Our iteration-39 CatBoost submission (0.7186) is the class of error a working
> instrument catches.

Also: **one of our stated premises is probably wrong.** See §2 and log 9 — the evidence more likely
supports "OOF has near-zero variance across artifacts and therefore near-zero information" than
"OOF and LB are anti-correlated". Different diagnosis, different fix. **Tier-0 below settles it in
ten minutes and should be run before anything else in this document.**

---

### TIER 0 — run this first, ~10 minutes, settles a premise

Over the ~45 artifacts compute and record:
- `sd(OOF_i)`, `sd(LB_i)`, `range(OOF_i)`, `range(LB_i)`
- Kendall tau-b(OOF_i, LB_i) with a permutation p-value
- the same after dropping non-Transformer artifacts

If `sd(OOF) < ~0.005` while `sd(LB) ~ 0.05`, the "anti-correlation" story is an artifact of fitting
a sign to a near-vertical scatter, and the honest description is *degenerate ID axis* — the exact
Camelyon17-WILDS pathology named in Miller et al. ICML 2021 (log 9). Put this number in the code
review either way; it is the single most informative number we can produce for free.

---

### THE PROTOCOL — "PRA-δ back-test with artifact-level permutation and Westfall–Young control"

**Inputs.** For each artifact i in 1..N (~45): `oof_prob_i` (1817,), `y_true` (1817,),
`test_prob_i` (1030,), `test_ids`, and the realized public-LB composite `LB_i`.

**Step 1 — PRE-REGISTER, then git-commit, BEFORE computing anything.**
Write one file containing: the exact list of K candidate instruments (below), the value of δ, the
pass thresholds, B, and the RNG seed. `git commit` it. Record the hash. This costs nothing and it
is the whole defence against self-deception; without it the analysis is uninterpretable and a code
reviewer is right to discount it. **Do not add an instrument after seeing a result.**

**Step 2 — the instrument set (K, pre-registered).** Each instrument is a function
`s_i = g(oof_prob_i, y_true, test_prob_i, {pool})` using **no test labels and no LB**:
1. `OOF_composite` — the incumbent. **Mandatory as the null baseline.**
2. `ATC` — threshold t fixed on OOF so that `mean(conf_oof > t)` equals OOF accuracy; report
   `mean(conf_test > t)`. (Garg et al. 2022. Accuracy-only — see §1.)
3. `AC` — mean max-confidence on test (average-confidence baseline).
4. `DoC` — OOF accuracy minus (mean OOF confidence − mean test confidence).
5. `NegEntropy` — mean negative predictive entropy on test.
6. `Margin` — mean |p_test − 0.5|.
7. `ALine-S` and 8. `ALine-D` — the agreement estimators (Baek et al. 2022), fitted over the
   INIT-SEED-ONLY sub-pool (see §2 — this restriction is load-bearing), probit-scaled.
9. `PoolAgree` — agreement of artifact i's hard test labels with the pool's majority vote.
10. `PredPosRate` — fraction of test predicted positive at 0.5. **COMPLIANCE FLAG:** include as a
    diagnostic only; it is a prevalence statistic and the prevalence-pinning lane is already
    ruled illegal. It must be marked non-actionable in the pre-registration.

**Step 3 — the primary statistic: PRA-δ (noise-aware pairwise ranking accuracy).**
```
D(δ) = { (i,j) : |LB_i − LB_j| > δ }
PRA_k(δ) = #{ (i,j) ∈ D(δ) : sign(s_ki − s_kj) == sign(LB_i − LB_j) } / |D(δ)|
```
Chance = 0.5.

*Why PRA and not Spearman.* PRA is the estimand that matches the decision. We never consume a
predicted score; we consume the answer to "is A better than B". Kendall tau-b **is** the rescaled
pairwise concordance rate, so PRA is tau-b restricted to discriminable pairs — report tau-b as a
secondary, and Spearman third, but pre-register PRA as primary. Do **not** use Pearson: the
CatBoost point at 0.7186 is a 10-sigma outlier that would dominate any moment-based statistic.

*δ: drop, do not weight.* **Drop.** Three reasons: (a) weighting requires choosing a weight
function, which is a fresh knob and a prong-(b) exposure; (b) non-discriminable pairs contribute
pure noise and attenuate PRA toward 0.5 by an unknown factor, destroying the scale so no threshold
can be pre-set; (c) "among pairs we can actually tell apart, how often is the order right" is a
clean, communicable estimand.
*The cost of dropping, and its fix.* D(δ) over-represents easy extreme pairs, so PRA is optimistic
relative to a real finalist choice. **Pre-register a second, stratified read: PRA restricted to the
hard band `δ < |ΔLB| < 0.04`.** Both must be reported. Passing only the wide version means we built
a disaster detector, not a selector — which is still useful, but must be labelled as such.

*Choice of δ.* **Use δ = 0.015 derived from the binomial arithmetic of a 309-row public LB, NOT
δ = 0.019.** Rationale in §5: 0.019 is a seed-sd we measured *by submitting to the LB*, so using it
imports LB feedback into the analysis, whereas 309-row binomial noise is pure combinatorics
requiring no LB observation. Report δ ∈ {0.010, 0.015, 0.019, 0.025} as a pre-registered
sensitivity sweep, with 0.015 as the single primary. The sweep is declared in advance so it cannot
become a search.

**Step 4 — inference: permute at the ARTIFACT level, never the pair level.**
The ~1000 pairs are massively overlapping (each artifact appears in ~44 of them) and the artifacts
themselves are not independent (iteration k+1 is usually a child of k). A binomial CI over pairs
would be wildly anti-conservative — do not compute one.
```
for b in 1..B (B = 10000):
    π_b = random permutation of {1..N}
    LB^(b) = LB[π_b]                    # relabel LB across artifacts
    recompute D(δ) and PRA_k^(b) from LB^(b) for every k, using the SAME π_b for all k
```
Regenerating D(δ) from the permuted vector is what makes this exact: the pair-overlap structure is
reproduced under the null, so no independence assumption is needed anywhere.

**Step 5 — multiple comparisons: Westfall–Young step-down max-T, shared permutations.**
```
M^(b) = max_k PRA_k^(b)
p_adj(winner) = (1 + #{b : M^(b) >= PRA_winner_obs}) / (B + 1)
then drop the winner and repeat the max over the remaining instruments (step-down)
```
Because all K instruments are evaluated on the *same* permutations, the procedure absorbs the (very
high) correlation between instruments — most of ours are confidence functionals of the same
probability vector, so a Bonferroni correction would be needlessly brutal and an uncorrected test
grossly optimistic. Citation: **P. H. Westfall & S. S. Young (1993), _Resampling-Based Multiple
Testing: Examples and Methods for p-Value Adjustment_, John Wiley & Sons** — maxT procedure pp. 50
and 114; step-down permutation algorithm 4.1, pp. 116–117. Strong FWER control.

**Step 6 — the noise ceiling. A perfect instrument cannot score 1.0.**
`LB_i` is itself a noisy measurement of the artifact's true expected LB, so PRA is attenuated even
for an oracle. Estimate the ceiling by simulation:
```
treat observed LB as a stand-in for true T          # biased upward-in-spread; state this
for b in 1..10000:
    LB'_i = LB_i + N(0, σ²),  σ = 0.019             # seed sd, used here only as a noise scale
    ceiling_b = PRA( s = LB (the oracle), reference = LB', δ )
PRA_ceiling = mean(ceiling_b)
```
Report the **normalized score PRA_obs / PRA_ceiling** alongside the raw. Expect the ceiling around
0.85–0.92 for δ=0.015; if it comes out below ~0.75 the whole exercise is underpowered and the
honest conclusion is "45 points is not enough", which is itself a defensible code-review finding.
(Noise-ceiling analysis of this form is standard practice in representational similarity analysis —
cf. Nili et al., PLoS Comput Biol 2014, "A Toolbox for Representational Similarity Analysis".
**I did not fetch this citation; verify before quoting it.** The method stands on its own arithmetic
regardless of the cite.)

**Step 7 — PASS THRESHOLD (pre-committed; an instrument passes iff ALL five hold).**
1. `PRA(δ=0.015) >= 0.70`
2. Westfall–Young step-down adjusted `p < 0.05`
3. Hard-band `PRA(0.015 < |ΔLB| < 0.04) >= 0.60` — guards against an instrument that can only spot
   CatBoost-class disasters being sold as a finalist selector
4. Beats `OOF_composite` by `>= 0.10` PRA on the identical pair set (paired comparison)
5. Leave-one-lineage-out stable: partition the 45 artifacts into their ~5–8 lineage clusters, drop
   each cluster in turn, require `PRA >= 0.65` in every fold

*Why 0.70.* With N=45 the artifact-level permutation null for PRA has sd roughly 0.06–0.08, so 0.70
sits ~2.5–3 null sd above chance; and against an expected ceiling near 0.88 it is ~80% of the
achievable maximum. It also has a plain-language reading: right on 7 of 10 discriminable calls.
Commit to it before computing, not after.

**Step 8 — what a PASS buys, and what it does not.**
A passing instrument at PRA ≈ 0.70 is a **veto tool, not a selector.** Use it in exactly one way:
if a new candidate scores below the 10th percentile of the passing instrument's distribution over
the historical pool, do not spend a submission on it. It is *not* licensed to choose between two
finalists separated by 0.005 — that gap is inside the noise floor and no instrument validated on 45
noisy points can resolve it. At iteration 39 this would have saved one submission. With 3 days and
5 submissions/day left, that is the realistic size of the prize. Say so plainly rather than
overselling.

**Step 9 — if everything fails.** A clean, FWER-corrected null result over 10 pre-registered
instruments is a genuinely strong code-review artifact: it converts "we burned submissions" from an
admission into a *measured finding*, backed by Garg et al.'s Proposition 1 as the theoretical
reason. Write it up with the same care as a pass.

---

## 1. Average Thresholded Confidence (ATC) — Garg et al., ICLR 2022, arXiv:2201.04234

**MECHANISM.** Fit a scalar threshold `t` on labeled source data so that the fraction of source
points with confidence above `t` equals the source accuracy. Then predict target accuracy as the
fraction of *unlabeled target* points with confidence above the same `t`. One number, one pass.

**WHAT IT ASSUMES, AND WHETHER WE VIOLATE IT.** Both of its two tractable regimes, as the authors
state them:
- Label shift — "the class-conditional distribution does not change (p_s(x|y)=p_t(x|y))".
  **VIOLATED, already measured:** our KS gate rejected p(x|y)-invariance at p≈0 (MLLS/BBSE lane).
- Covariate shift — "the target marginal support is a subset of the source marginal support and …
  p_s(y|x)=p_t(y|x)". **VIOLATED twice over:** adversarial train-vs-test AUC ≈0.99 means
  near-separable, hence not support-contained; and our entire iteration-39 diagnosis is that
  p_s(y|x) ≠ p_t(y|x).

**EVIDENCE.** The authors prove the general case is hopeless:
> **Proposition 1** — "Absent further assumptions, accuracy on the target is identifiable iff
> p_t(y|x) is uniquely identified given p_s(x,y) and p_t(x)."
> **Corollary 1** — "Absent assumptions on the classifier f, no method of estimating accuracy will
> work in all scenarios, i.e., for different nature of distribution shifts."

Stated failure mode: "Shifting the support of target class conditional p_t(x_inv|y) may introduce a
bias in ATC estimates." On their hardest realistic benchmark (Breeds, novel subpopulations) "MAE is
much higher … with all methods."

**DOES IT EXTEND TO AUC OR F1? Not in this paper — it is accuracy-only throughout.** There is
adjacent work that does extend to confusion-matrix metrics: Białek, Kivimäki, Kuberski & Perrakis,
"Estimating Model Performance Under Covariate Shift Without Labels" (arXiv:2401.08348), proposing
**PAPE**, which "can be applied to any performance metric defined with elements of the confusion
matrix" and "does not need any assumptions about the nature of covariate shift, learning directly
from data instead." Note carefully: *no assumption about the nature of the covariate shift* is not
*no assumption of covariate shift* — the title says covariate shift and the method is squarely
inside that regime. It buys us metric coverage (F1), not assumption relief, and F1 is not our
problem. Also: arXiv preprint from the NannyML authors, not a peer-reviewed venue as far as I could
verify — weigh accordingly.

**Two structural reasons ATC could not reach our composite even if the assumptions held:**
- *F1 at a hard 0.5 cut is prevalence-sensitive.* Estimating it requires pinning the target
  positive rate — the exact quantity our KS gate said we cannot estimate, and the exact quantity
  prevalence-pinning was ruled illegal for.
- *AUC is a set-level ranking functional*, not an average of per-example indicators. ATC's
  statistic is a sum of per-example thresholded indicators; there is no route from it to a
  ranking-over-pairs quantity. This is not a gap in the paper, it is a type mismatch.

**BACK-TEST.** Instrument #2 in §0 Step 2. Fit t on OOF, evaluate on test probs, PRA-δ vs LB.
**PASS THRESHOLD.** The §0 Step 7 five conditions. **EXPECTED VALUE if it passes:** low — even a
passing ATC ranks by an accuracy proxy while we are scored on 0.6·F1+0.4·AUC. Keep it in the
pre-registered set as a cheap baseline; do not build on it.
**COMPLIANCE (prong b).** Clean as a *diagnostic*: t is fixed by a train-only criterion (match OOF
accuracy), it never sees the LB, and it never touches a prediction. It becomes non-compliant the
moment its output is used to move a knob that reaches a submission — see §5.

**VERDICT: ATC does not refute the claim.** Its value to us is the opposite of what we hoped:
Proposition 1 is the best available citation for *why* the claim is largely true, and belongs in
the code-review writeup.

---

## 2. Agreement-on-the-Line / Accuracy-on-the-Line — THE ONE WORTH BUILDING

**MECHANISM.** Baek, Jiang, Raghunathan & Kolter, NeurIPS 2022, arXiv:2206.13089. For every pair of
models compute `Agr(h,h′)=E_{x~D}[1{h(x)=h′(x)}]` on ID data and on *unlabeled* OOD data. Probit-
transform both. If the ID-vs-OOD **agreement** scatter is linear, the ID-vs-OOD **accuracy** line
has (empirically) the same slope and bias — so the accuracy line can be recovered with no labels.
ALine-S reads slope/bias off the agreement fit; ALine-D solves a least-squares system over all
pairs (needs ≥3 models; we have ~45).

**WHAT IT ASSUMES, AND OUR STANDING ON EACH.**

*(a) Accuracy-on-the-line must hold.* Verbatim: "When ID vs. OOD accuracy observes strong linear
correlation (≥0.95 R² values), ID vs. OOD agreement is also strongly linearly correlated." Their
own named failures are **Camelyon17-wilds** (R²=0.263 acc / 0.226 agr) and **iWildCam-wilds**
(R²=0.738 / 0.424), with "a mean absolute estimation error of around 5%" when the line is absent.
**Those two are the WILDS datasets closest in kind to ours** — unseen-hospital and unseen-camera-
location shifts, i.e. spatial-domain generalization, which is exactly our "no lat/lon, train and
test are different places" situation. Strong prior that ALine's *point estimate* of our LB will be
bad. Say so honestly.

*(b) Diversity must come from initialization.* Saxena, Kim, Mehra, Baek, Kolter & Raghunathan,
arXiv:2404.01542, verbatim: "the choice of randomness during training (linear head initialization,
data ordering, and data subsetting) can lead to drastically different levels of agreement-on-the-
line in the resulting ensemble. Surprisingly, **only random head initialization is able to reliably
induce agreement-on-the-line**." **WE VIOLATE THIS AS CURRENTLY BUILT.** One "seed" in our pipeline
simultaneously changes init, batch order, *and* the StratifiedKFold assignment — one valid source
mixed with the two documented-invalid ones. **Fix, and it is cheap: build the AGL pool by varying
the init seed with the fold split and data order HELD FIXED.** This one design detail is the
difference between a meaningful AGL back-test and a repeat of iteration 39.

*(c) Neural classifiers only.* "agreement-on-the-line appears to only hold for neural network
classifiers." Exclude tree artifacts from the line fit outright. This is also a clean published
account of *why* the iteration-39 instrument mis-ranked CatBoost against the Transformer — the
comparison was outside the phenomenon's support.

**WHY THIS IS THE REFUTATION, DESPITE (a).** The claim under attack is that *no* offline instrument
can tell us anything about transfer. AGL is an offline, label-free instrument whose *documented
intended use* includes a validity check, not only a point estimate. Verbatim from the paper:
> "check if ID and OOD agreement are linearly correlated, in order to know if our model selection
> criterion based on ID accuracy is valid"
and
> "When accuracy-on-the-line holds … we can simply pick the model with highest ID accuracy."

So AGL answers, for free, a strictly weaker but decision-relevant question: **is my OOF ranking
trustworthy at all?** On our data it will very likely answer *no* — and that answer, produced before
submitting, is precisely what would have blocked the iteration-39 CatBoost submission that returned
0.7186. An instrument that reliably says "do not spend a submission here" converts a wasted
submission into a saved one. That is a real refutation of the literal claim.

**DOES IT SURVIVE CONDITIONAL SHIFT?** No theoretical guarantee — none is claimed anywhere in the
paper; accuracy-on-the-line and agreement-on-the-line are *empirical phenomena*, established by
large-scale observation, not theorems. Under conditional shift they may or may not hold, and the
only way to know is to check the agreement scatter, which is exactly the free check. Do not claim
theoretical coverage we do not have.

**BACK-TEST.** Instruments #7 (ALine-S), #8 (ALine-D), #9 (PoolAgree) in §0, plus a standalone
diagnostic that is arguably more valuable than any of them: **report the probit-space R² of the
ID-vs-OOD agreement scatter over the init-seed-only pool.** If that R² ≪ 0.95, we have a published,
citable, label-free demonstration that OOF-based model selection is invalid on this competition —
which is a first-class code-review finding independent of whether any instrument passes.

**PASS THRESHOLD.** §0 Step 7 for the instruments. For the diagnostic, the paper's own bar:
agreement R² ≥ 0.95 to declare the line present; below ~0.5 declare it decisively absent.
**EXPECTED VALUE.** Highest of anything here, but bounded: realistically a veto tool plus a strong
writeup finding. Do not expect it to separate two finalists 0.005 apart.
**COMPLIANCE (prong b).** Clean in its natural form. Agreement is computed from OOF and unlabeled
test probabilities only. No LB, no realized positive rate, no threshold moved — the 0.5 cut is
untouched. The only exposure is at the point where its output influences a submission (§5).

---

## 3. Importance-weighted cross-validation — Sugiyama, Krauledat & Müller, JMLR 8 (2007) 985–1005

**MECHANISM.** Reweight each source CV example by `w(x)=p_t(x)/p_s(x)`, making the CV estimate an
unbiased estimate of *target* risk.

**WHAT IT ASSUMES — and this is definitional, not incidental.** Verbatim from the paper: covariate
shift is "the situation where the training input points and test input points follow different
distributions **while the conditional distribution of output values given input points is
unchanged**", and they prove IWCV's "unbiasedness **even under the covariate shift**". The theorem
is *for* covariate shift. There is no IWCV theorem for conditional shift, because importance
weighting only reweights the covariate marginal — **it is arithmetically incapable of moving
p(y|x).**

**OUR DATA VIOLATES IT, AND WE HAVE ALREADY PAID TO FIND OUT.** Our stated diagnosis of iteration 39
— that a covariate-selected holdout "still carries TRAIN labels drawn from the train conditional
p(y|x), so it is structurally blind to CONDITIONAL shift" — is exactly the negation of the covariate
shift assumption. **Reframe iteration 39 for the writeup: the adversarial "most test-like 30% of
train" holdout IS a crude nonparametric IWCV — hard 0/1 weights instead of smooth density-ratio
weights. Its 0.7186 is therefore not an isolated mishap; it is an empirical measurement that the
entire importance-weighting family is blind on this problem.** That is a far stronger sentence than
"our holdout didn't work", and it is defensible.

**IS THE NEAR-DEGENERATE DENSITY RATIO FATAL? Yes, independently — a second, sufficient kill.**
Adversarial AUC ≈0.99 ⇒ near-separable ⇒ w is near-zero on most train rows and enormous on a few.
- Cortes, Mansour & Mohri, "Learning Bounds for Importance Weighting", NIPS 2010 (Adv. NeurIPS 23,
  pp. 442–450): they "[identify] simple cases where importance weighting can fail", and guarantees
  for unbounded weights hold only "under the weak assumption that the second moment is bounded, a
  condition related to the Rényi divergence of the training and test distributions." The operative
  quantity, `E_s[w²] = exp(d₂(P_t‖P_s))`, diverges as support separates.
- Reis et al., arXiv:2010.01184: "importance weighting … may fail, according to common wisdom, due
  to small effective sample sizes (ESS). Previous research argues this scenario is more common in
  high-dimensional settings." (146 columns, 12 bands × 12 months — we are the high-dimensional case.)

**HONESTY NOTE ON THE ESS RESULT.** I could not render either PDF and therefore do **not** have a
verbatim ESS formula from a primary source. The universally-used expression is Kong's
`ESS = (Σᵢwᵢ)² / Σᵢwᵢ²`, but I have not verified it here against a quotable source — **do not
attribute a specific ESS theorem to the literature in the writeup.** Instead **compute ours**: fit
the train-vs-test discriminator, take `w_i = p̂/(1−p̂)` on the 1817 train rows, report the ESS
directly. At adv AUC ≈0.99 the prediction is ESS ≪ 100 out of 1817. This is a ~20-line computation,
it should be run today, it is the cheapest empirical kill-shot on the whole reweighting family, and
a self-computed number beats a borrowed one in a code review.

**THE λ TRAP — do not take it.** The standard remedy is flattened weights `w^λ`, λ∈[0,1]. Under
prong (b) this is inadmissible: λ is a knob with no train-only criterion fixing it, and the
conventional way to select λ is by IWCV itself — i.e. by the broken instrument. Leave it alone.

**VERDICT: does not refute.** Two independent failures: wrong assumption, and a destroyed effective
sample size even if the assumption held.

---

## 4. Does ANYTHING detect CONDITIONAL shift without target labels? — the crux

**THE HONEST ANSWER IS NO, AND IT IS A THEOREM, NOT AN OPINION.**

Garg et al. (ICLR 2022), **Proposition 1**: "Absent further assumptions, accuracy on the target is
identifiable iff p_t(y|x) is uniquely identified given p_s(x,y) and p_t(x)."

Read it as the impossibility statement it is. Conditional shift *means* p_t(y|x) ≠ p_s(y|x). Your
observables are p_s(x,y) — labeled train — and p_t(x) — unlabeled test. Nothing in that pair
constrains p_t(y|x): for any candidate target conditional you care to name, there is a joint target
distribution consistent with the same p_t(x). The function is unidentified, so any estimator of it
is reporting an assumption, not a measurement. **Corollary 1**: "Absent assumptions on the
classifier f, no method of estimating accuracy will work in all scenarios."

This is why every method in this literature is named after its assumption — label shift, covariate
shift, sparse joint shift, generalized label shift. The assumption *is* the method. **And we have
empirically rejected the two standard ones**: KS gate rejected p(x|y)-invariance at p≈0 (kills the
label-shift family: BBSE, MLLS, RLLS), and adversarial AUC ≈0.99 plus the iteration-39 result kill
the covariate-shift family (IWCV, ATC, PAPE, adversarially-selected holdouts).

**THE ONE NEAR-POSITIVE RESULT, reported for completeness and honestly discounted.**
Chen, Zaharia & Zou, **Sparse Joint Shift (SJS)**, NeurIPS 2022, arXiv:2209.08436. Verbatim:
> "we propose a new distribution shift model, Sparse Joint Shift (SJS), which considers the **joint
> shift of both labels and a few features**. This unifies and generalizes several existing shift
> models including label shift and sparse covariate shift … We describe mathematical conditions
> under which SJS is identifiable. We further propose SEES, an algorithmic framework to characterize
> the distribution shift under SJS and to estimate a model's performance on new data without any
> labels."

It buys identifiability beyond covariate shift by assuming the shift is confined to y plus a
**sparse** feature set. **Our shift is the opposite of sparse**: test rows expose only 4–6
contiguous months of 12, so the shift lives in the observation *mask* and touches all 12 bands ×
12 months at once. There is one speculative reframing — if the sparse shifting "feature" is taken
to be the observation-window variable itself, the shift becomes sparse in a reparameterized space.
**Genuinely interesting, entirely untested, not implementable in three days. Log it, do not chase
it.** I did not obtain the verbatim identifiability conditions, so do not assert what they are.

**WHAT TO WRITE IN THE CODE REVIEW (this is worth real marks).** Not "our validation didn't work",
but: *we identified that the shift in this competition includes a conditional component; we cite
the ICLR 2022 result that target performance is then unidentifiable from labeled source plus
unlabeled target; we tested the two standard escape assumptions and rejected both empirically (KS
gate p≈0; adversarial AUC 0.99 and a measured ESS of X); we therefore pre-registered and ran a
label-free ranking back-test over K instruments with FWER control, and here is the result.* That is
a description of competent science under a genuine constraint, and it is exactly the reasoning a
reviewer is looking for.

**THE ONE THING THAT DOES SURVIVE.** Note what Proposition 1 does *not* say. It concerns
identifying *accuracy* — a cardinal quantity. It says nothing about *ordinal* comparison of two
classifiers, and the agreement-on-the-line literature is empirical evidence that ordinal structure
sometimes survives shifts where cardinal estimation fails. That gap is the entire justification for
§0. It is a gap, not a guarantee.

---

## 5. Compliance under prong (b) — using past LB scores to validate an instrument

The subtlety is real and deserves a straight answer: **calibrating or selecting an instrument
against our own past LB scores IS leaderboard feedback, and indirection does not launder it.**
Three uses, sharply distinguished:

**(U1) LB → instrument → submission. ILLEGAL.** Fitting the ALine slope/bias to LB history, or
choosing δ by what makes the correlation look best, or picking which instrument to trust by its LB
correlation and then submitting on its advice. The chain terminates in a submission, so a knob that
reaches the prediction was fixed by LB feedback, one indirection removed. Prong (b) is violated.

**(U2) Measure, report, discard. UNAMBIGUOUSLY CLEAN.** Run the back-test purely as science, put
the number in the writeup, and let it change nothing about what we submit. No knob reaches the
submission; there is nothing to launder. Real value against the 35% code review, zero risk.

**(U3) The clean middle path — pre-commit the instrument, use LB only as a post-hoc audit.**
This is the version worth having, and it is available:
1. Define the instrument **completely** from train-only + unlabeled-test quantities. No LB anywhere
   in its definition: ATC's threshold from OOF, agreement from OOF/test probabilities, the pool
   from init-seed variation.
2. Write the definition, every constant, and the pass thresholds to a file and **git-commit it
   before running the back-test**. Record the hash in the writeup.
3. Run the back-test once. LB is used only to *audit* an already-frozen instrument — an audit is
   not a fit. Same logic as a pre-registered trial.
4. If it passes, the instrument may be applied to **new** candidates whose LB is unknown, to veto.
   Its parameters were never fit to LB.

**δ = 0.019 IS ITSELF LB-DERIVED — flagged, and here is the fix.** The seed sd of 0.019 was measured
*by submitting seed replicates*. Using it imports LB feedback. Defence: it is a noise *scale*, not a
*direction* — it says nothing about which model is better, only how precisely the LB measures, and
it only selects which pairs enter a diagnostic, never touching a prediction. That is defensible.
**But a strictly cleaner option costs nothing: derive δ from the binomial arithmetic of a 309-row
public LB (≈0.012–0.015), which requires no LB observation at all.** Use **δ = 0.015** as primary,
report 0.019 as a pre-registered sensitivity, and disclose both. Take the free win.

**THE RESIDUAL LEAK, disclosed rather than denied.** Even U3 leaks one bit: *the decision to use the
instrument at all* is conditioned on it passing an LB-based audit. This is vastly smaller than
fitting, but it is not zero, and claiming zero would be false. The Westfall–Young correction is what
bounds it: over a pre-registered list of K instruments with strong FWER control, the leak is exactly
"we selected the winner among K pre-specified hypotheses at level 0.05". **Quantify it and state it
in those words.** A reviewer will credit the disclosure far more than a claim of purity.

**DEFINITIVELY OUT — any one of these voids the analysis:** re-running the back-test after seeing it
fail; adding instruments post hoc; moving a pass threshold; tuning δ outside the pre-registered
sweep. One shot, or it means nothing.

---

## 6. CAVEATS

**Corrections to our own stated beliefs — read these first.**

1. **"OOF and LB are anti-correlated across 45 iterations" is probably the wrong description.** If
   OOF sits at ~0.975 for essentially every artifact while LB spans 0.72–0.907, the scatter is
   near-vertical and a fitted sign is noise. The published name for this is the Camelyon17-WILDS
   pathology (Miller et al., ICML 2021): "Models with 95% ID accuracy have OOD accuracies that range
   from about 50% (random chance) to 95% … there is no precise linear trend." The correct claim is
   *near-zero ID variance ⇒ near-zero ID information*, which is a different diagnosis with a
   different fix (spread the ID axis) than "the sign is inverted" (which suggests, wrongly, that
   picking the *worst* OOF model would help). **Tier 0 in §0 settles this in ten minutes.**
2. **Iteration 39 was not a one-off; it was a measurement.** The adversarial test-like holdout is
   nonparametric IWCV with hard weights. Its failure is evidence about the entire importance-
   weighting family, and should be written up that way.
3. **The strong form of the target claim survives.** No offline instrument can predict our LB
   *score*. That is Garg et al.'s Proposition 1, and we violate both escape assumptions. What is
   refuted is only the literal claim that *nothing* offline is informative — ordinal/veto
   information is testable today at zero submission cost. Do not oversell this to a reviewer.

**Limits of the evidence in this document.**

4. **Two PDFs would not render** (Sugiyama JMLR 2007; the ATC PDF). ATC content came from the ar5iv
   HTML, which I regard as reliable; Sugiyama content came from the JMLR abstract page only. I have
   **no verbatim ESS formula from a primary source** — the Kong `ESS=(Σw)²/Σw²` expression in log 6
   is unverified here. Compute ours instead of citing theirs.
5. **Unverified citations, marked as such:** Nili et al. (PLoS Comput Biol 2014) for noise-ceiling
   methodology — not fetched; the method stands on its own arithmetic. The "linear models on CLIP
   features exhibit AGL" line came from a search summary and was **not** confirmed in the
   arXiv:2404.01542 abstract — do not cite it. SJS identifiability conditions were not obtained
   verbatim — do not assert what they are.
6. **Quote fidelity.** Several quotes were relayed through the fetch tool's summarizer, which
   imposes a short quote limit. The Proposition 1 / Corollary 1 / AGL-precondition /
   arXiv:2404.01542-abstract quotes are the load-bearing ones; **re-verify those four against the
   primary sources before they go into the code-review writeup.** Everything else is supporting.
7. **Westfall & Young (1993) is a book**, cited here for the maxT procedure (pp. 50, 114) and the
   step-down permutation algorithm 4.1 (pp. 116–117) via secondary sources (the Bioconductor
   `multtest` documentation and the Berkeley tech report). I did not consult the book. The
   procedure as written in §0 Step 5 is standard and self-contained.

**Limits of the protocol itself.**

8. **N=45 is small and the points are not independent** (lineage: iteration k+1 is usually a child
   of k). The artifact-level permutation and leave-one-lineage-out fold are the mitigations, but
   the effective N is materially below 45 and the power is correspondingly low. A null result may
   mean "no signal" or "not enough points" — the noise-ceiling computation (Step 6) is what
   distinguishes them, so do not skip it.
9. **A PASS yields a veto tool, not a finalist selector.** Nothing validated on 45 noisy points can
   resolve a 0.005 gap. State the intended use narrowly or the instrument will be over-trusted
   exactly when it matters.
10. **The AGL pool restriction (init-seed only, folds fixed) may require re-runs.** If no existing
    artifacts share a fold-split seed while differing in init seed, the AGL back-test cannot be run
    on the stored npz files as they are. Check this before planning around it; it is a data-
    availability question, not a research question.
11. **Three days to deadline.** Tier 0 and the ESS computation are hours of work and pay off in the
    writeup regardless of outcome. The full pre-registered K=10 back-test is a bigger lift. If time
    forces a choice: **Tier 0, then the ESS number, then the AGL agreement-R² diagnostic.** Those
    three alone make the code-review argument, and none of them touches a submission.

---

## STATUS: COMPLETE

---

## RAW RESEARCH LOG (append-only, newest at bottom)

### [log 1] ATC — Garg, Balakrishnan, Lipton, Neyshabur, Sedghi. "Leveraging Unlabeled Data to Predict Out-of-Distribution Performance." ICLR 2022. arXiv:2201.04234.

VERBATIM ABSTRACT:
> "Real-world machine learning deployments are characterized by mismatches between the source
> (training) and target (test) distributions that may cause performance drops. In this work, we
> investigate methods for predicting the target domain accuracy using only labeled source data and
> unlabeled target data. We propose Average Thresholded Confidence (ATC), a practical method that
> learns a threshold on the model's confidence, predicting accuracy as the fraction of unlabeled
> examples for which model confidence exceeds that threshold. ATC outperforms previous methods
> across several model architectures, types of distribution shifts (e.g., due to synthetic
> corruptions, dataset reproduction, or novel subpopulations), and datasets (Wilds, ImageNet,
> Breeds, CIFAR, and MNIST). In our experiments, ATC estimates target performance 2–4x more
> accurately than prior methods. **We also explore the theoretical foundations of the problem,
> proving that, in general, identifying the accuracy is just as hard as identifying the optimal
> predictor and thus, the efficacy of any method rests upon (perhaps unstated) assumptions on the
> nature of the shift.** Finally, analyzing our method on some toy distributions, we provide
> insights concerning when it works."

FIRST-ORDER READ: the bolded sentence is *already a partial confirmation of our belief*, not a
refutation — the authors themselves prove an impossibility result. But note the exact scope: it
says identifying **accuracy** is as hard as identifying the optimal predictor **in general**, i.e.
absent assumptions. It does NOT say *ranking* two models is as hard. That gap is where our
refutation must live. Need the theorem statement verbatim.

Note also: ATC predicts ACCURACY. Our metric is 0.6*F1(0.5 cut) + 0.4*AUC. Abstract page gave no
statement about extension to AUC/F1 — must check the paper body.

### [log 2] Agreement-on-the-Line — Baek, Jiang, Raghunathan, Kolter. NeurIPS 2022. arXiv:2206.13089.

VERBATIM ABSTRACT:
> "Recently, Miller et al. showed that a model's in-distribution (ID) accuracy has a strong linear
> correlation with its out-of-distribution (OOD) accuracy on several OOD benchmarks -- a phenomenon
> they dubbed ''accuracy-on-the-line''. While a useful tool for model selection (i.e., the model
> most likely to perform the best OOD is the one with highest ID accuracy), this fact does not help
> estimate the actual OOD performance of models without access to a labeled OOD validation set. In
> this paper, we show a similar but surprising phenomenon also holds for the agreement between
> pairs of neural network classifiers: **whenever accuracy-on-the-line holds, we observe that the
> OOD agreement between the predictions of any two pairs of neural networks (with potentially
> different architectures) also observes a strong linear correlation with their ID agreement.**
> Furthermore, we observe that the slope and bias of OOD vs ID agreement closely matches that of
> OOD vs ID accuracy. This phenomenon, which we call agreement-on-the-line, has important practical
> applications: without any labeled data, we can predict the OOD accuracy of classifiers, since OOD
> agreement can be estimated with just unlabeled data. Our prediction algorithm outperforms previous
> methods both in shifts where agreement-on-the-line holds and, surprisingly, when accuracy is not
> on the line. **This phenomenon also provides new insights into deep neural networks: unlike
> accuracy-on-the-line, agreement-on-the-line appears to only hold for neural network classifiers.**"

FIRST-ORDER READ — two things directly relevant to us:
(a) The precondition is explicitly conditional: "**whenever** accuracy-on-the-line holds". Our
    iteration-39 CatBoost disaster (adversarial holdout said GO, LB said 0.7186) plus the reported
    OOF/LB anti-correlation across 45 iterations is *prima facie* evidence that accuracy-on-the-line
    does NOT hold in our setting. That is exactly the regime AGL disclaims.
(b) BUT — crucially, AGL is a *diagnostic that can be checked without labels*. The ID-vs-OOD
    agreement scatter across model pairs is computable from our stored npz files at zero cost, and
    the paper's own claim is that the AGREEMENT line's slope/bias MATCHES the accuracy line's. That
    means the agreement scatter is itself a TEST of whether the accuracy line exists. This is the
    single most actionable thread — pursue.
(c) "only hold for neural network classifiers" — our finalists are Transformers, so we are inside
    the supported model class; CatBoost (iter39) is NOT, which is a candidate *post-hoc explanation*
    for why the iter39 instrument mis-ranked a tree model against a neural one.

### [log 3] ATC full text (ar5iv 2201.04234) — THE IMPOSSIBILITY RESULT, verbatim

> **Proposition 1**: "Absent further assumptions, accuracy on the target is identifiable iff
> p_t(y|x) is uniquely identified given p_s(x,y) and p_t(x)."

> **Corollary 1**: "Absent assumptions on the classifier f, no method of estimating accuracy will
> work in all scenarios, i.e., for different nature of distribution shifts."

**THIS IS THE ANSWER TO QUESTION 4, AND IT IS A CONFIRMATION, NOT A REFUTATION.** Proposition 1 is
precisely a statement that *conditional shift is not detectable from p_s(x,y) and p_t(x) alone*.
Under conditional shift p_t(y|x) != p_s(y|x), and by construction p_t(y|x) is NOT identified by
p_s(x,y) + p_t(x) — so target accuracy is not identifiable. Every unlabeled-target performance
estimator (ATC, DoC, GDE, AC, IM, ...) is therefore, without exception, purchasing its validity
with an assumption that rules conditional shift out. Cite this in the code-review writeup: it is a
published, peer-reviewed (ICLR 2022) impossibility statement in exactly our situation.

Conditions ATC needs (their own framing of the two tractable regimes):
- Label shift: "the class-conditional distribution does not change (p_s(x|y)=p_t(x|y))"
  -> WE ALREADY REJECTED THIS. Our KS gate rejected p(x|y)-invariance at p~0 (MLLS/BBSE lane).
- Covariate shift: "the target marginal support is a subset of the source marginal support and ...
  p_s(y|x)=p_t(y|x)"
  -> WE VIOLATE BOTH HALVES. Adversarial AUC ~0.99 means the support-subset condition fails
     (near-separable = near-disjoint support), and our whole diagnosis of the iter39 CatBoost
     failure is that p_s(y|x) != p_t(y|x).

ATC failure mode they state: "Shifting the support of target class conditional p_t(x_inv|y) may
introduce a bias in ATC estimates." Their hardest real benchmark (Breeds, novel subpopulations)
gives "MAE is much higher ... with all methods."

Metric coverage: **ATC is accuracy-only.** The ar5iv pass found no treatment of F1, AUC, precision
or recall anywhere in the paper. So ATC does not even nominally target our 0.6*F1 + 0.4*AUC
composite. Any F1/AUC extension would be our own unpublished invention — DO NOT claim literature
support for it. (See log 6 for what literature does exist on non-accuracy extensions.)

VERDICT ON ATC: **does not refute.** Its own theory says it cannot help us, its assumptions are the
two we have already empirically rejected, and it estimates the wrong functional. Do not spend
implementation time here. Its VALUE to us is rhetorical/scientific: Proposition 1 + Corollary 1 are
the citation that makes our "we had to spend submissions" claim a principled statement rather than
an excuse — worth real points in the 35% code review.

### [log 4] Agreement-on-the-Line full text (ar5iv 2206.13089) — THE ACTIONABLE LEVER

Definition (verbatim): "Agr(h,h')=E_{x~D}[1{h(x)=h'(x)}]" — computable on UNLABELED target.

Precondition (verbatim): "When ID vs. OOD accuracy observes strong linear correlation (>=0.95 R^2
values), ID vs. OOD agreement is also strongly linearly correlated."

Failure cases they report by name — and note WHICH datasets these are:
- Camelyon17-wilds: R^2 = 0.263 (accuracy), R^2 = 0.226 (agreement)
- iWildCam-wilds:   R^2 = 0.738 (accuracy), R^2 = 0.424 (agreement)
- "On datasets where we do not observe a linear correlation between ID and OOD agreement (and
  accuracy), ALine does not perform very well, with a mean absolute estimation error of around 5%."

**Camelyon17 and iWildCam are the two WILDS datasets whose shift is closest in character to ours
(unseen geographic domains / unseen camera-trap locations = a spatial-domain shift, exactly our
"no lat/lon, train and test are different places" situation). AGL's two named failures are our two
nearest neighbours in benchmark space.** That is a strong prior that ALine's *point estimate* of
our LB score will be bad. Be honest about this in the writeup.

**BUT — the refutation lives here, and it is a real one.** The paper's own recommended use is not
only point estimation. Verbatim:

> "check if ID and OOD agreement are linearly correlated, in order to know if our model selection
> criterion based on ID accuracy is valid"

and

> "When accuracy-on-the-line holds ... we can simply pick the model with highest ID accuracy."

So AGL supplies a **label-free VALIDITY TEST for OOF-based model selection**. The claim we were
asked to refute says "no offline instrument can predict which model will transfer". AGL is an
offline instrument that, at zero submission cost, answers a strictly weaker but decision-relevant
question: *is my OOF ranking trustworthy at all?* On our data it will almost certainly return NO —
and that itself is an offline instrument output that would have PREVENTED the iteration-39 CatBoost
submission (which cost us a submission and returned 0.7186). An instrument that reliably says
"do not trust this" is not nothing; it converts a wasted submission into a saved one.

Mechanics that matter for our implementation:
- Probit scaling: they apply Phi^-1(.) (inverse standard normal CDF) to BOTH accuracy and agreement
  before fitting the line, following Miller et al. 2021. Do this; do not fit in raw accuracy space.
- ALine-D needs >= 3 models for a unique solution. We have ~45 artifacts — ample.
- Diversity requirement is real: their >=150 models per shift span "convolutional networks and
  Vision Transformers with varying architectures, not just different seeds of identical
  architectures." OUR ~45 ARTIFACTS ARE MOSTLY SEED/CONFIG VARIANTS OF ONE ~71k TRANSFORMER. This
  is a genuine violation, and it biases us toward *artificially high* agreement (correlated
  errors), which will flatten the agreement line and can make AGL look like it holds when it does
  not. Must be flagged as a caveat and, if possible, mitigated by restricting to the most
  architecturally distinct subset of artifacts.
- AGL "appears to only hold for neural network classifiers" — so any tree-model artifacts in our 45
  must be EXCLUDED from the line fit, not merely noted. This is also a clean post-hoc account of
  why the iter39 adversarial holdout mis-ranked CatBoost against the Transformer.

### [log 5] IWCV — Sugiyama, Krauledat & Müller, JMLR 8 (2007) 985–1005

Verbatim from the paper's abstract (via jmlr.org/papers/v8/sugiyama07a.html):
> "the situation where the training input points and test input points follow different
> distributions **while the conditional distribution of output values given input points is
> unchanged** is called the covariate shift"

and they prove IWCV's "**unbiasedness even under the covariate shift**".

**This is dispositive and it is a CONFIRMATION of our belief, on the definitional level.** IWCV's
unbiasedness theorem is stated *for covariate shift*, and covariate shift is DEFINED by
p_s(y|x) = p_t(y|x). Our central diagnosis — that the iter39 test-like holdout failed because it
"still carries TRAIN labels drawn from the train conditional p(y|x)" — is exactly the statement
that we are NOT under covariate shift. IWCV has no theorem for us. Importance weighting the train
rows re-weights the covariate marginal only; it cannot move p(y|x). **The iter39 adversarial
holdout is in fact a crude nonparametric IWCV (hard 0/1 weights instead of smooth density-ratio
weights), so its 0.7186 failure is an empirical datapoint that the whole IWCV family is blind
here.** Worth stating exactly that way in the code review — it turns a wasted submission into
evidence.

Second, independent, fatal problem: VARIANCE. Adversarial AUC ~0.99 => near-separable => the
density ratio w(x)=p_t(x)/p_s(x) is near-degenerate (near-zero on most train rows, huge on a few).
Relevant published results:
- Cortes, Mansour & Mohri, "Learning Bounds for Importance Weighting", NIPS 2010 (Adv. NeurIPS 23,
  pp. 442–450). They "[identify] simple cases where importance weighting can fail", and their
  guarantees for unbounded weights hold only "under the weak assumption that the second moment is
  bounded, a condition related to the Rényi divergence of the training and test distributions."
  Near-disjoint support drives the Rényi-2 divergence (hence the second moment of w) up, and the
  bound degrades correspondingly.
- See log 6 for the explicit effective-sample-size quantification.

Practical note (still in Sugiyama's own line of work): the standard remedy is the FLATTENED weight
w^lambda with lambda in [0,1], trading bias for variance. lambda=0 recovers plain CV, lambda=1
recovers full IWCV. **Under prong (b) this is a trap**: lambda is a knob, and there is no
train-only criterion that fixes it — the usual selection of lambda is itself by IWCV, which is the
thing that is broken. Do not introduce lambda.

VERDICT ON IWCV: **does not refute.** Two independent failures — wrong assumption (needs p(y|x)
invariant, which we have already diagnosed as false) and, even granting the assumption, a
near-degenerate density ratio destroying the effective sample size.

### [log 6] Effective sample size — the citation to use

Cortes, Mansour & Mohri, "Learning Bounds for Importance Weighting", NIPS 2010 (Adv. NeurIPS 23,
pp. 442–450). Contributions per the proceedings page: they "[identify] simple cases where
importance weighting can fail"; guarantees for unbounded importance weights require "the weak
assumption that the second moment is bounded, a condition related to the Rényi divergence of the
training and test distributions."
-> The operative quantity is E_s[w^2] = exp(d_2(P_t || P_s)) (Rényi-2). Near-disjoint support sends
   this to infinity. This is the rigorous form of "the density ratio is degenerate".

Also: Reis, Maia, et al. (arXiv:2010.01184), "Effective Sample Size, Dimensionality, and
Generalization in Covariate Shift Adaptation". Verbatim abstract:
> "Covariate shift adaptation yields good generalization performance when domains differ only by
> the marginal distribution of features. Covariate shift adaptation is usually implemented using
> importance weighting, which may fail, according to common wisdom, due to small effective sample
> sizes (ESS). Previous research argues this scenario is more common in high-dimensional settings."

HONEST CAVEAT: I did not retrieve an explicit ESS formula verbatim from either PDF (both PDFs
failed to render through the fetch tool). The formula in universal use is Kong's
**ESS = (sum_i w_i)^2 / sum_i w_i^2**, which is standard in importance sampling but which I have
NOT verified here against a primary source with a quote. **Do NOT cite a specific ESS number to the
literature.** Instead COMPUTE it on our own data: fit a train-vs-test discriminator, take
w_i = p_hat/(1-p_hat) on the 1817 train rows, and report ESS directly. Given adv AUC ~0.99, the
predicted result is ESS << 100 — a self-generated, verifiable number that is far more persuasive in
a code review than a borrowed one. **This is a 20-line computation and should be run today**; it is
the cheapest possible empirical kill-shot on the whole reweighting family and it doubles as
evidence for the writeup.

### [log 7] Sparse Joint Shift (SJS) — Chen, Zaharia & Zou, NeurIPS 2022, arXiv:2209.08436

Verbatim abstract (excerpt):
> "In this paper, we propose a new distribution shift model, Sparse Joint Shift (SJS), which
> considers the **joint shift of both labels and a few features**. This unifies and generalizes
> several existing shift models including label shift and sparse covariate shift, where only
> marginal feature or label distribution shifts are considered. We describe mathematical conditions
> under which SJS is identifiable. We further propose SEES, an algorithmic framework to characterize
> the distribution shift under SJS and to estimate a model's performance on new data without any
> labels."

WHY THIS MATTERS TO OUR QUESTION 4: SJS is the closest thing in the literature to a *positive*
result for label-involving (i.e. beyond-covariate) shift with no target labels. It buys
identifiability by assuming the shift is confined to y plus a SMALL, SPARSE set of features.

WHETHER IT FITS US — probably not, and here is the specific reason. Our shift is not sparse in
feature space: adversarial train-vs-test AUC ~0.99 with a *structural* cause (test rows show only
4–6 contiguous months out of 12, so the shift is in the MASK/observation pattern, which touches
every one of the 12 bands x 12 months simultaneously). That is the opposite of sparse. Also, our
own KS gate already rejected p(x|y)-invariance at p~0, and SJS's identifiability conditions are
built on top of a label-shift-like backbone.
HOWEVER — there is one framing under which it could be argued to fit: if the sparse shifting
"feature" is taken to be the observation-window variable itself (which month-window is visible),
then the shift IS sparse in a re-parameterised feature space. **This is a genuine, non-obvious
research direction but it is NOT implementable in 3 days and it is untested. Log it; do not chase
it now.** Flagging honestly: I did not obtain the verbatim identifiability theorem statement, so do
not assert what its conditions are.

### [log 8] AGL follow-up — the DIVERSITY precondition (arXiv:2404.01542)

"Predicting the Performance of Foundation Models via Agreement-on-the-Line" (2024). Reported
findings, which are DIRECTLY DECISIVE for our back-test design:
- "The diversity induced via random head initialization yields AGL, while the diversity induced via
  data reordering or data subsetting does not."
- "AGL-based methods can only accurately predict the OOD performance of models in ensembles with
  diverse initialization, and cannot with data subsetting or ordering."
- Contra the original paper's neural-only claim: "on top of CLIP features, linear models can
  exhibit AGL with random initialization."

**READ THIS AGAINST OUR ARTIFACT POOL.** Our ~45 artifacts and our 10-seed pool vary by seed. A
seed in our pipeline changes (i) weight initialization AND (ii) the StratifiedKFold split, i.e. data
subsetting, AND (iii) batch ordering. Per 2404.01542 only (i) generates AGL-valid diversity; (ii)
and (iii) actively do not. So a naive AGL fit over our seed pool is CONTAMINATED — it mixes the one
diversity source that works with the two that are documented not to.
**CONCRETE MITIGATION, and it is cheap:** build the AGL model pool by varying INITIALIZATION ONLY —
fix the fold assignment and the data order across the pool, vary only the init seed. If we have
artifacts that already share a fold-split seed but differ in init seed, use those; otherwise this
is a re-run, not a re-search. This single design detail is the difference between an AGL back-test
that means something and one that does not, and it is the kind of specificity that was missing from
our iteration-39 attempt.
NOTE: I have this paper's findings via search-result summary, not from the PDF. Quotes above are
reported as paraphrase-grade; VERIFY against arXiv:2404.01542 before quoting in the writeup.

### [log 8b] arXiv:2404.01542 — VERIFIED verbatim abstract

Saxena, Kim, Mehra, Baek, Kolter, Raghunathan. "Predicting the Performance of Foundation Models
via Agreement-on-the-Line." arXiv:2404.01542.
> "Estimating the out-of-distribution performance in regimes where labels are scarce is critical to
> safely deploy foundation models. Recently, it was shown that ensembles of neural networks observe
> the phenomena 'agreement-on-the-line', which can be leveraged to reliably predict OOD performance
> without labels. However, in contrast to classical neural networks that are trained on
> in-distribution data from scratch for numerous epochs, foundation models undergo minimal
> finetuning from heavily pretrained weights, which may reduce the ensemble diversity needed to
> observe agreement-on-the-line. In our work, we demonstrate that when lightly finetuning multiple
> runs from a single foundation model, **the choice of randomness during training (linear head
> initialization, data ordering, and data subsetting) can lead to drastically different levels of
> agreement-on-the-line in the resulting ensemble. Surprisingly, only random head initialization is
> able to reliably induce agreement-on-the-line** in finetuned foundation models across vision and
> language benchmarks."

CONFIRMED. The head-initialization-only requirement is verbatim in the abstract. The log-8 point
stands: our seed pool conflates head init with data ordering and fold subsetting, and the
literature says two of those three do not produce AGL-valid diversity. **The AGL pool must vary
init seed with the fold split HELD FIXED.**
(The "linear models on CLIP features exhibit AGL" line from the search summary was NOT confirmed in
the abstract — treat as unverified, do not cite.)

### [log 9] Accuracy-on-the-Line — Miller, Taori, Raghunathan, Sagawa, Koh, Shankar, Liang, Carmon, Schmidt. ICML 2021, PMLR v139, arXiv:2107.04649

Where the line HOLDS: CIFAR-10 and ImageNet variants, a YCB-derived pose task, FMoW-WILDS
satellite imagery, iWildCam-WILDS. Holds "across model architectures, hyperparameters, training set
size, and training duration".

Where it FAILS — and this is our diagnosis in someone else's paper:
- Camelyon17-WILDS (hospital-to-hospital staining/imaging shift) and some CIFAR-10-C corruptions.
- Reported behaviour on Camelyon17: **"Models with 95% ID accuracy have OOD accuracies that range
  from about 50% (random chance) to 95%. OOD accuracy is highly variable across the spectrum of ID
  accuracies, and there is no precise linear trend."**

**THIS IS OUR SITUATION, ALMOST NUMERICALLY.** Our OOF ("ID") sits at ~0.975 for essentially every
artifact while LB ("OOD") ranges from 0.7186 (CatBoost) to 0.907. A near-constant ID axis with a
wildly variable OOD axis is precisely the Camelyon17 pathology: the ID coordinate carries almost no
information about the OOD coordinate, so ANY instrument whose input is an ID-side score is
regressing on a degenerate predictor. That, not conditional shift alone, is the *proximate*
statistical reason our OOF is useless — and it is a published, named, benchmarked failure mode we
can cite by name. Note the further consequence: with the ID axis compressed into a
0.97–0.98 band, even a *true* underlying line could not be estimated — the leverage is gone. This
also predicts the observed "anti-correlation": in a near-vertical scatter, the fitted sign is
noise, and reporting it as "anti-correlated" over 45 points is over-reading. **Correction to our
stated belief: the evidence probably does NOT support "OOF and LB are anti-correlated"; it supports
"OOF has near-zero variance and therefore near-zero information about LB." Those are different
claims with different remedies, and the second one is fixable (make the ID axis spread out).**

RELATED, worth a look but not yet read: "Accuracy on the wrong line: On the pitfalls of noisy data
for OOD generalisation" (OpenReview id=uqj8qBNQla) — argues noisy data breaks the line. Unread.


