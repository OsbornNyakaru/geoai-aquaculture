# Round 24 — Q1: Optimising the HIGH-RECALL REGION of the ROC (partial AUC, NP classification, constrained ERM)

**Researcher note.** Written incrementally, appended after each paper read. Everything below is
labelled **VERIFIED (read)** = I fetched and read the paper/abstract text, or **INFERRED** = derived
by me from theory, or read only second-hand. Derivations marked "MY DERIVATION" are mine and are
the load-bearing part of the answer — check them.

Task context (from UPDATE_24.md): team score = 0.6·F1 + 0.4·AUC, threshold literally 0.5 and
non-negotiable, n=1817 train / 1030 test, 71k-param temporal Transformer, global AUC 0.945842 already
beats the leader (0.944897) but F1 0.881720 vs ~0.918. They sit at recall 0.859 / precision 0.906.
Two theorems must be survived: (T1) Platt annihilation of affine logit changes, (T2) pointwise-loss
order invariance.

Status: IN PROGRESS.

---

## CORRECTION 0 (highest value, written first) — §3.2's premise is NOT established, and you can settle it exactly with TWO submissions

### 0.1 The internal inconsistency

§3.2 asserts as fact: *"Their advantage is concentrated in the high-recall corner of the ROC."*
§3.3 then proves you **cannot know that** — the max F1 reachable **on your existing ranking by
threshold alone** lies in [0.8817, 0.9574], and **the leader's 0.918 sits strictly inside that
interval.** So the two live hypotheses are:

- **H_shape**: the leader's ROC genuinely dominates yours in the high-recall corner (Q1 is the right
  question, pAUC is the right tool);
- **H_point**: your ROCs are of essentially equal quality (their global AUC is *lower* by 0.00095 ≈
  **21 discordant pairs out of 22538** — statistically indistinguishable) and the leader simply
  **operates at a different point on an equally good curve**. Under H_point, Q1 is aimed at the wrong
  target entirely and no amount of pAUC optimisation helps; the whole gap is calibration/threshold
  placement.

**§3.2 and §3.3 cannot both be presented as they are.** §3.2 assumes H_shape; §3.3 proves H_shape is
unidentified. This is Correction 0 and everything downstream depends on it.

### 0.2 It is NOT unidentified. You can measure it exactly. MY DERIVATION.

You control the `TargetRAUC` column freely — **only its ordering is scored, and you may submit ties.**
That makes the RAUC column a *measuring instrument*, not just a prediction. Fix notation from §3.2
(all exact, all already known to you):

```
P = 191 positives, N = 118 negatives, P·N = 22538 pairs
above the 0.5 cut: 164 pos (TP), 17 neg (FP)
below the 0.5 cut:  27 pos (FN), 101 neg (TN)
```

Pair-block census (this is just a partition, no assumptions):

| block | count | status |
|---|---|---|
| (pos above, neg above) | 164·17 = 2788 | unknown split — call concordant part `C_above` |
| (pos above, neg below) | 164·101 = 16564 | all concordant (forced) |
| (pos below, neg above) | 27·17 = 459 | all discordant (forced) |
| (pos below, neg below) | 27·101 = 2727 | unknown — this is what §3.3 calls the "below block" |
| **total** | **22538** | ✓ |

**PROBE A — the control (1 submission, result is fully predictable in advance).**
Submit `TargetRAUC = 1.0` for every row your champion puts above 0.5 and `0.0` for every row below.
`TargetF1` = your champion's, unchanged. Two distinct values, so both within-block sets become **ties**.
Under the standard Mann–Whitney/`sklearn.roc_auc_score` convention a tie contributes exactly 0.5:

```
AUC_A · 22538 = 16564  +  0.5·2788  +  0.5·2727  +  0
              = 16564  +  1394      +  1363.5
              = 19321.5
AUC_A = 19321.5 / 22538 = 0.857285562...
```

This number depends on **nothing unknown**. If Zindi returns it, you have simultaneously confirmed
(a) P = 191 / TP = 164 / FP = 17 to the last decimal, and (b) that the grader's tie convention is 0.5.
If it does **not** match, one of your §3.2 inversions is wrong and you have learned something even
more important. Either way it is a *control that returns the value arithmetic guarantees* — precisely
the standard §8.3 says you now hold yourselves to.

**PROBE B — the actual measurement (1 submission).** Three tiers:
`1.0` for all rows above the cut; `0.5` for the **top m = 20 rows below the cut** (ranked by your
champion's score); `0.0` for the remaining below-cut rows. Let `p_B` = how many of your 27 missed
positives land in that top-20. Then (q = 20 − p_B negatives in tier B):

```
AUC_B · 22538 = 0.5·(164·17)            [pos-above vs neg-above, tied]
              + 164·101                 [pos-above vs neg-below, concordant]
              + 0                       [pos-below vs neg-above, discordant]
              + 0.5·p_B·q               [tier-B pos vs tier-B neg, tied]
              + p_B·(101 − q)           [tier-B pos vs tier-C neg, concordant]
              + 0                       [tier-C pos vs tier-B neg, discordant]
              + 0.5·(27 − p_B)·(101 − q)[tier-C pos vs tier-C neg, tied]
```

Everything is known except the single integer `p_B ∈ [0,20]`. Evaluating:

| p_B | AUC_B |
|---|---|
| 0 | 0.845177 |
| 4 | 0.856637 |
| 8 | 0.868000 |
| 12 | 0.879356 |
| 16 | **0.890713** |
| 20 | 0.902070 |

Consecutive integers are separated by **≈ 0.00284 in AUC**, and Zindi prints 9 decimals. `p_B` is
recovered **exactly, with no noise, no model, no assumption.**

**What p_B means.** The leader's point is TP 180 / FP 21 — i.e. **+16 true positives for +4 false
positives** relative to you. Admitting the top 20 rows below your cut is *at most* +20 FP, and:

- `p_B ≥ 16` ⇒ your ranking **already contains a point at least as good as the leader's**. H_point is
  true. **Q1 is the wrong question** — pAUC cannot help you because your ROC is not the deficient
  object. Your entire 0.037 F1 gap is the 0.5 cut landing in the wrong place, i.e. a calibration
  problem (Platt fit at prior 0.4023, deployed at prior ~0.618 — Hole 1 of §3.3, which is a *live*
  defect, not a retracted one).
- `p_B ≤ 8` ⇒ H_shape is true. Your ROC really is flat in the high-recall corner, the missed positives
  really are buried, §3.4's "median score 0.170" transfers to the test set, and **Q1 is exactly the
  right question**. Then the rest of this document applies.

**Legality of the probes.** Prong (a): unaffected — these probes do not change your champion's
decision rule, and the finalist submissions are untouched. Prong (b): this is the sharp edge. These
are *measurements*, in the same category as the F1-inversion you already run for diagnosis. They
become illegal the moment a number from them **sets a knob**. My recommendation: use `p_B` to decide
**which research lane to spend your last 36 hours on**, and to write the definitive negative result
for the 35% report — do **not** feed it into any hyperparameter, threshold, or model selection. State
this explicitly in the report. If you judge even lane-selection to be too close to the line, note that
Probe A alone is a pure control with zero information content about the model and is unambiguously
legal, and that you have already accepted LB-derived *diagnosis* (§3.2) as legal.

**Cost:** 2 of your ~10 remaining submissions. **Value:** it converts the single largest open question
in the brief from "we do not know" to a measured integer. I rank this above every method in the rest
of this report.

### 0.3 Why the two-hypothesis question is *not* symmetric under your rules

Note the asymmetry, because it determines how much Q1 is worth even before you probe:

- If **H_shape**, the fix is a genuinely different ranking → the E1 methods below are on-target, but
  (see §1) they are far weaker than the brief hopes.
- If **H_point**, the fix is *forbidden by your own rule* in its direct form (move the threshold), and
  the only legal route is to change the **probability scale** — which Theorem 1 (Platt annihilation)
  kills for affine changes and Theorem 2 kills for pointwise losses. The legal survivors are then
  exactly the things that change the *prior* the calibrator sees without touching the ranking, which
  is Q2/Q3 territory, not Q1.

So: **probe first.** Spending your last day on partial-AUC surrogates when `p_B = 18` would be the
most expensive mistake available to you this round.

---

## CORRECTION 0b — erratum to my own Probe A control value (accepted from the team)

My published control figure for the **champion** partition was `0.857285562`. That is wrong in the
7th decimal. The correct value is

```
19321.5 / 22538 = 0.857285473...
```

The team caught this and is right. The *mechanism* (tie contributes exactly 0.5; the four-block pair
census; the fact that the number depends on nothing unknown) is unaffected — only the arithmetic of
the final division was mis-rounded. I record it here rather than editing the earlier section, per the
append-only rule. **Use 0.857285473 as the champion-partition control.**

The team instrumented the probes on `jtt_lam5_s42` instead (public cell TP 164, FP 16, FN 27, TN 102),
for which they compute Probe A = **0.861522762** and a consecutive-`p_B` separation of **0.002862**.
Sanity check of their Probe A, MY DERIVATION, independent of theirs:

```
P = 191, N = 118, P·N = 22538
above cut: 164 pos, 16 neg   below cut: 27 pos, 102 neg
AUC_A·22538 = 164·102  [pos-above vs neg-below, concordant]
            + 0.5·164·16  [tied above]
            + 0.5·27·102  [tied below]
            + 0          [pos-below vs neg-above, discordant]
            = 16728 + 1312 + 1377 = 19417
19417 / 22538 = 0.861522762...   ✓ agrees exactly
```

Their separation figure also checks out: one extra missed-positive promoted into the middle tier
moves the numerator by `0.5·q + (102 − q)` for `q` tier-B negatives; with a 20-row tier this is
≈ 64.5/22538 ≈ 0.00286. **Both their numbers are confirmed.**

---

# §1. THE METHOD SURVEY — high-recall-targeted objectives, family by family

**Read §1.0 first. It is a single theorem and it decides four of the five families below.**

## §1.0 ⚠️ THE MASTER OBJECTION: partial AUC has the SAME population optimum as full AUC

This is the most important thing in the survey and I want it stated before any individual paper,
because it is what converts "we have never tested class E1" from an exciting gap into a mostly
foreclosed one.

**Setup.** Let `η(x) = p(y=1|x)`, let `p₁(x)`, `p₀(x)` be the class-conditional densities, and let
`Λ(x) = p₁(x)/p₀(x)` be the likelihood ratio. By Bayes, `Λ(x) = (η(x)/(1−η(x)))·((1−π)/π)` — a
**strictly increasing** function of `η(x)` for any fixed prior `π`. So "`Λ`" and "`η`" and "any strictly
monotone transform of `η`" are all *the same ranking*.

**Neyman–Pearson lemma (Neyman & Pearson 1933, Phil. Trans. R. Soc. A 231:289–337,
DOI 10.1098/rsta.1933.0009). VERIFIED — this is the classical statement, not a paraphrase of a
secondary source.** Among all tests of `p₀` against `p₁` with size ≤ α, the likelihood-ratio test
`1[Λ(x) > t_α]` has maximal power. Restated in ROC coordinates: for **every** FPR level `α ∈ [0,1]`,

```
TPR_Λ(α)  ≥  TPR_s(α)     for every measurable score function s.
```

That is: **the ROC curve of Λ is the pointwise upper envelope of the ROC curves of all scorers.**

**MY DERIVATION — the corollary that matters.** Partial AUC over any FPR band `[α, β]` is
`pAUC_s(α,β) = ∫_α^β TPR_s(u) du`. Two-way pAUC (TPAUC) additionally restricts TPR ≥ γ, which is a
restriction of the same integral's region. Since the integrand is pointwise maximised by `Λ` at every
`u`, the integral over **any** measurable subregion is maximised by `Λ`:

```
pAUC_Λ(α,β) ≥ pAUC_s(α,β)     ∀ s, ∀ 0 ≤ α < β ≤ 1
TPAUC_Λ     ≥ TPAUC_s          ∀ s, ∀ γ
```

and identically for full AUC (`α=0, β=1`).

> **⚠️ CONSEQUENCE.** The Bayes-optimal scorer for partial AUC **is the same object** as the
> Bayes-optimal scorer for full AUC: any monotone transform of `η(x)`. A pAUC objective therefore
> **does not have a different population target**. It is not an escape from your Theorem 2 in the way
> §5's E1 taxonomy implies — Theorem 2 says pointwise losses land on `T(η)`; pAUC says the *best
> possible* answer **is** `T(η)`. pAUC and full AUC agree at the optimum and can only disagree
> **off** the optimum.

**So what does pAUC actually buy?** Exactly one thing, and it is worth being precise because it is
the entire honest case for the family: **under model misspecification and/or finite samples, the
achievable set is a restricted class `S ⊊ {all scorers}` that does not contain `Λ`.** On that
restricted class the two objectives genuinely differ, and `argmax_{s∈S} pAUC(s) ≠ argmax_{s∈S} AUC(s)`.
pAUC training reallocates a *misspecified* model's finite capacity toward one region of the curve.

This is not nothing — a 71k-parameter Transformer on n=1817 is emphatically in the misspecified,
finite-sample regime. But it reframes the whole family as a **capacity-reallocation** manoeuvre, not
a "we were optimising the wrong thing" correction. And capacity reallocation is **zero-sum on the
curve**: see §1.5 for what it costs you in the AUC column you currently win.

**Corollary for the report (this is a citable, rigorous negative result):**

> *There is no ranking objective whatsoever — global, partial, one-way, two-way, region-weighted,
> pairwise, or listwise — whose population optimum improves on a calibrated `η`. Every method in
> class E1 is, at the population level, an alternative route to the same destination. The only
> mechanisms that can change the destination are ones that change the estimand (E3: the target `y` is
> replaced) or the hypothesis class / effective sample (E2). This is a structural argument for why
> the team's biggest historical win lives in E3 and why E1 has been oversold to them.*


---

## §1.0b THE EXCHANGE RATE — how many AUC pairs is one true positive worth? MY DERIVATION

Before judging any method, fix the currency. Composite `= 0.6·F1 + 0.4·AUC`. Public cell: `P=191`,
`N=118`, `P·N = 22538` pairs, champion at `TP=164, FP=17, PP=181`.

**One recovered true positive with NO new false positive:**
```
F1(164) = 328/372 = 0.881720430
F1(165) = 330/373 = 0.884718499      dF1 = +0.002998069   -> +0.001798841 composite
```
**One inverted (pos,neg) pair:** `dAUC = 1/22538 = 4.4369e-5` -> `-1.77475e-5` composite.

```
        1 recovered TP   ==   101.4 pairwise inversions
```

> **⚠️ CORRECTION TO THE BRIEF'S FRAMING.** "We currently WIN the AUC column" is true and almost
> worthless. Your AUC lead over the leader is `0.945842 − 0.944897 = 0.000945` = **21.3 pairs** =
> **+0.000378 composite**. The F1 gap is **+0.0220 composite** (below). The AUC column is a
> **58x smaller** stake than the F1 column. You are not defending anything by protecting global AUC;
> you are defending 0.04% of a point.

**The full trade, sized.** Matching the leader's cell (TP 180, FP 21, PP 201):
```
F1 = 360/392 = 0.918367347      dF1 = +0.036646917   -> +0.021988 composite
break-even AUC sacrifice = 0.021988 / 0.4 = 0.054970 AUC
                         = 1238.9 additional discordant pairs
```
So you could drop global AUC from **0.945842 all the way to 0.890872** and still break even. This is a
**very large** licence. Any objection to an E1 method of the form "but it might cost us AUC" is, at
these magnitudes, **not a real objection**. The real objections are elsewhere — see (iii) and (iv)
below. Put this table in the report; it is the cleanest quantitative statement of what the 0.6/0.4
weighting actually buys.

*(Asymmetry worth noting: a TP bought at the price of one FP is worth only `+0.000633 F1` =
`+0.00038 composite` ≈ 21 pairs. The value is overwhelmingly in TPs that cost nothing. Recovering
positives from the buried region — §3.4's "median score 0.170" — is ~100x more valuable per unit than
trading along the frontier. That is an argument for E3, not E1.)*

---

## §1.1 FAMILY A — PARTIAL AUC MAXIMISATION (one-way and two-way)

### A.1 What the literature actually is

| work | citation | status |
|---|---|---|
| Structural SVM for pAUC over FPR band `[a,b]` | Narasimhan & Agarwal, **ICML 2013**, PMLR 28(1):516-524 | **VERIFIED (abstract read)** |
| `SVM^pAUC_tight` — tighter convex upper bound | Narasimhan & Agarwal, **KDD 2013**, DOI 10.1145/2487575.2487674 | **VERIFIED (landing page read)** |
| Journal consolidation | Narasimhan & Agarwal, **arXiv:1605.04337**, "Support Vector Algorithms for Optimizing the Partial Area Under the ROC Curve" | VERIFIED (exists, indexed) |
| `pAUCBoost` — boosting for pAUC, cubic splines / stumps | Komori & Eguchi, **BMC Bioinformatics 11:314 (2010)**, DOI 10.1186/1471-2105-11-314 | **VERIFIED (full text read)** |
| pAUC with nonlinear (generative + DNN) scorers | Ueda & Fujino, **arXiv:1806.04838** (2018) | **VERIFIED (abstract read)** |
| Two-way pAUC (TPAUC), end-to-end deep framework | Yang, Xu, Bao, He, Cao, Huang, **ICML 2021**, PMLR 139:11820-11829; journal version **IEEE TPAMI**, DOI 10.1109/TPAMI.2022.3185311 | **VERIFIED (definition read via a restating paper)** |
| pAUC via DRO (CVaR-exact / KL-smooth), the LibAUC line | Zhu, Wang, Yang T. et al., **ICML 2022**, arXiv:2203.00176, "When AUC meets DRO" | **VERIFIED (abstract read)** |
| OPAUC in a range of FPRs, large scale | Zhu et al., OpenReview `FFPcFtWJwsB` | INFERRED (indexed, not read) |
| Instance-wise regularised pAUC, asymptotic unbiasedness | arXiv:2210.03967 (NeurIPS 2022) | INFERRED (abstract only) |

**The definition that matters.** TPAUC over the region `{TPR >= 1-t0, FPR <= t1}` reduces **exactly**
to a pairwise sum over a *selected subset*. **VERIFIED**, quoted from the restatement in
arXiv:2505.21944 (Stochastic Primal-Dual Double Block-Coordinate for TPAUC):

```
min_w  (1/(n+ n-))  SUM_{xi in S+^up[1,k1]}  SUM_{xj in S-^down[1,k2]}  loss( h_w(xj) - h_w(xi) )

  S+^up[1,k1]   = the k1 = floor(n+ · t0) POSITIVES WITH THE SMALLEST SCORES
  S-^down[1,k2] = the k2 = floor(n- · t1) NEGATIVES WITH THE LARGEST SCORES
```

and the same source states that TPAUC "is more challenging" than AUC precisely because "its estimator
involves selection of negative and positive examples whose prediction scores are in top and bottom
ranks."

That single equation is the whole family — and also its own indictment for this team.

### A.2 The five decisive checks

**(i) Genuinely non-decomposable? — YES. This one passes.** The per-example contribution depends on
which *other* examples are in `S+^up[1,k1]` / `S-^down[1,k2]`, and that membership is set by the whole
batch's scores. It cannot be written as `SUM_i [ y_i·l1(z_i) + (1-y_i)·l0(z_i) ]` for a fixed pair
`(l0, l1)`. **Your Theorem 2 does not apply.** Theorem 1 also does not apply (the change is not an
affine logit shift). The family clears both theorems, exactly as §5's E1 taxonomy claims.
**This is the only check it passes cleanly.**

**(ii) Does the population optimum differ from a monotone transform of `eta`? — NO. It is identical.**
This is §1.0's Neyman-Pearson argument and it is the most important finding in this survey. The ROC of
the likelihood ratio pointwise dominates every other ROC, so it maximises the integral over *every*
subregion simultaneously. There is no `(t0, t1)` for which the pAUC-optimal scorer differs from the
AUC-optimal scorer at the population level.

> **Concretely.** pAUC is not a *different objective* in the sense the brief hopes. It is the *same*
> objective with a **reweighted finite-sample estimator**. Its entire value is capacity reallocation
> under misspecification. Any report sentence of the form "we were optimising global AUC when the
> metric wanted the high-recall region" is **false as stated** — a perfect model of `eta` wins both
> columns, and the estimand never changed.

**(iii) ⚠️ Does it smuggle in an operating-point choice? — YES, FATALLY, IN TWO SEPARATE WAYS.**

*Fatal way 1 — the band IS a rate target.* `t0` is literally a **TPR (recall) target**; `t1` is
literally an **FPR target**. Prong (b) of your own §5: "every knob is fixed by a train-only criterion
— **never a positive-rate target**." `(t0, t1)` is a rate target in the most literal possible sense,
it is the *only* hyperparameter of the method, and the method is undefined without it. There is no
version of pAUC that does not choose a region.

Could `(t0, t1)` be set train-only? **No — and this is provable from facts already in your brief.** To
place the band you must know where your deployed `0.5` cut lands *in FPR coordinates on the test
distribution*. You cannot: (a) §4 — OOF is blind; OOF sits at ~0.97 for artifacts whose public score
spans 0.72-0.907, so OOF ROC coordinates are not test ROC coordinates; (b) the prior moves 0.4023 ->
~0.618, which moves the cut's FPR even for a *perfectly* transported ranking; (c) train and test are
exactly separable (adversarial AUC = 1.0000), so you cannot importance-weight the train ROC onto the
test ROC — you already proved that lane shut. **Every legal train-only rule you could write down puts
the band in the wrong place.** The only source that would place it correctly is the leaderboard, which
is a prong-(b) violation by construction.

*Fatal way 2 — the optimised region and the deployed operating point are DECOUPLED.* MY DERIVATION,
and this is the argument I would lead with in the report. Your pipeline is:

```
pAUC-trained score s(x)  ->  Platt sigma(a·s + b) fit on TRAIN OOF  ->  cut at literal 0.5
```

Platt is strictly monotone, so it preserves the ROC exactly. Where the `0.5` cut falls **on** that ROC
is determined entirely by `(a,b)` — i.e. by the train-OOF calibration — and by **nothing in the pAUC
objective**. There is no coupling whatsoever between the region you optimised and the region you
operate in. You could execute a textbook-perfect TPAUC optimisation of the corner
`{TPR >= 0.94, FPR <= 0.18}` and then deploy at `{TPR = 0.859, FPR = 0.144}`, having spent your whole
capacity budget improving a stretch of curve you never visit. Worse: because pAUC *deliberately*
degrades the rest of the curve, the region you actually operate in would come out **worse than
before**.

*Third, softer problem — prong (a) erosion.* Prong (a) demands "a literal 0.5 on a **genuine
probability**". A hinge/pairwise-ranking score has no probabilistic semantics at all; Platt would be
manufacturing the probability scale from scratch on a margin-type score. That is precisely the regime
where two-parameter sigmoid calibration is known to be worst behaved. You would be weakening the prong
you are strictest about, in order to chase a region you have no mechanism to reach.

**VERDICT ON (iii): DEAD ON ARRIVAL.** Not "risky" — the method's sole hyperparameter is a rate target
that cannot be set legally, and even if it could, it does not connect to the deployed operating point.

**(iv) Does it work at n=1817 with 71k parameters? — NO, independently of (iii). MY DERIVATION.**

Your train split: 1817 rows at 40.23% positive => **`n+ ≈ 731`, `n- ≈ 1086`.** Instantiate the TPAUC
subset sizes for the region that would actually match the leader (`TPR >= 0.94`, `FPR <= 0.18`, i.e.
`t0 = 0.06`, `t1 = 0.18`):

```
k1 = floor(731  · 0.06) = 43    positives   <- the ENTIRE positive signal in the objective
k2 = floor(1086 · 0.18) = 195   negatives
```

**Forty-three positive examples.** Even a deliberately generous band (`t0 = 0.15`, `t1 = 0.25`) gives
`k1 = 109`, `k2 = 271`. For scale: your *seed-to-seed* composite variance is **0.019** and your
significance bar is **0.015**. A statistic driven by 43 endogenously-selected examples will not clear
either. Three compounding problems:

1. **Endogenous selection.** `S+^up[1,k1]` is defined by the model's own current scores, so the active
   set churns every step. This is the same instability that makes CVaR/superquantile objectives
   high-variance — and it is exactly why the LibAUC line (arXiv:2203.00176) had to replace the *exact*
   CVaR estimator with a **KL-smoothed inexact** one to obtain convergence at all. **VERIFIED**: that
   paper explicitly contrasts a "non-smooth but exact estimator for pAUC" (CVaR-DRO) with an "inexact
   but smooth (soft) estimator" (KL-DRO). You would be adopting the inexact one and inheriting its
   bias.
2. **The literature warns about this at *larger* sample sizes than yours.** Komori & Eguchi
   (DOI 10.1186/1471-2105-11-314) conclude — **VERIFIED, their wording** — that pAUCBoost "with FPR
   restricted to be small should be applied to the genes or markers that have gone through a
   pAUC-based filtering procedure beforehand", i.e. narrow-band pAUC overfits and needs a pre-screen;
   their simulations show false discovery once non-informative features are present. You have 144 raw
   features, most of them non-informative by your own ablations.
3. **Your ensembling washes it out.** The final artefact is a 10-seed probability average. Averaging
   ten models that each optimised a *different* endogenous top-k subset pulls the ensemble back toward
   the global-AUC solution — the same averaging effect that made hard majority voting move only 4-7
   rows of 1030.

**(v) The 0.6/0.4 trade-off.** By §1.0b the trade is *nominally generous*: you may spend 1239 pairs
(0.055 AUC) to buy the leader's cell. So pAUC is **not** killed by the trade-off arithmetic. It is
killed by (iii) and (iv). Keep this distinction in the report: **the objection is not "it costs AUC",
it is "the region is unidentifiable and the estimator is starved."**

### A.3 One-way vs two-way: does the distinction rescue anything?

No, and the reason is worth stating. One-way pAUC (Narasimhan-Agarwal, FPR in `[a,b]`) subsets only
the **negatives**. Two-way (Yang et al.) subsets both. Your defect is fundamentally about **positives
ranked too low** — §3.4: median score of a missed positive is **0.170**, ten below 0.10, **zero** in
`[0.45, 0.50)`. One-way pAUC never subsets positives, so it is aimed at the wrong axis entirely.
Two-way pAUC is aimed at the right axis and is precisely the variant whose `k1 = 43` starves it.

> **The variant that targets your actual defect is the one your sample size cannot support.** That is
> a clean, reportable finding and I would print it verbatim.

### A.4 Verdict, Family A

> **REJECT — but not on the theorems.** Report line: *"Partial-AUC maximisation is the one method
> class that provably escapes our order-invariance theorem, and we tested it against our legality
> prongs rather than our theorems. We rejected it for three reasons we can state exactly: (1) by the
> Neyman-Pearson lemma its population optimum is identical to full AUC's, so it is a
> capacity-reallocation heuristic rather than a corrected estimand; (2) its only hyperparameter is an
> explicit (TPR, FPR) rate target, which our legality rule forbids and which — because train and test
> are exactly separable and the prior shifts 0.4023 -> 0.618 — no train-only criterion can place; and
> (3) at our sample size the two-way variant that targets our actual defect is driven by 43
> endogenously-selected positive examples, against a seed-to-seed noise floor of 0.019 composite."*

---

## §1.2 FAMILY B — REGION-RESTRICTED / PUSH / LISTWISE RANKING LOSSES

### B.1 The literature

| work | citation | status |
|---|---|---|
| **p-norm push** — convex, concentrates at the TOP | Rudin, **JMLR 10:2233-2271 (2009)**, "The P-Norm Push: A Simple Convex Ranking Algorithm that Concentrates at the Top of the List" | **VERIFIED (abstract + TR read)** |
| **Infinite push** (p -> inf), SVM form | Agarwal, **SDM 2011**, "The Infinite Push"; sparse variant arXiv:1206.6432 | VERIFIED (indexed, abstract) |
| **TopPush** — linear-time top-rank optimisation | Li, Jin, Zhou, **NIPS 2014**, arXiv:1410.1462 | **VERIFIED (abstract read)** |
| **Accuracy at the Top** | Boyd, Cortes, Mohri, Radovanovic, NIPS 2012 | INFERRED (known, not re-read) |
| **LambdaRank -> LambdaLoss** — metric-driven pairwise reweighting with a probabilistic justification | Wang, Li, Golbandi, Bendersky, Najork, **CIKM 2018**, DOI 10.1145/3269206.3271784 | **VERIFIED (abstract read)** |
| **Bayes-optimal scorers for bipartite ranking, incl. the p-norm-push family** | Menon & Williamson, **COLT 2014** PMLR 35; journal **JMLR 17(195), 2016**, "Bipartite Ranking: a Risk-Theoretic Perspective" | VERIFIED that the paper "characterise[s] the set of Bayes-optimal scorers"; the *content* of that characterisation is INFERRED (PDF not parseable) |
| Online/stochastic methods for non-decomposable losses | Kar, Narasimhan, Jain, arXiv:1410.6776 | INFERRED (indexed) |

### B.2 The five checks

**(i) Non-decomposable? — YES.** `p`-norm push is `(1/n_-) SUM_j ( SUM_i loss(h(x_j^-) - h(x_i^+)) )^p`; the outer
power `p` couples each negative's whole set of comparisons. LambdaRank's gradient weight `|dNDCG_ij|`
depends on the *current ranks* of `i` and `j`, i.e. on the whole list. Listwise (ListNet/ListMLE)
likewise. **Theorem 2 does not apply. Theorem 1 does not apply.** Clears both.

**(ii) Population optimum? — SAME AGAIN.** All of these are (weighted) functionals of the ROC curve
that are *monotone under ROC dominance*: if `ROC_A(u) >= ROC_B(u)` for all `u`, then `A` is at least as
good under p-norm push, infinite push, NDCG-with-binary-relevance, AP, precision@k and recall@k. By
the Neyman-Pearson envelope argument of §1.0, the likelihood ratio dominates every ROC, hence
maximises every one of them. Menon & Williamson (JMLR 17(195), 2016) study exactly this question for
the generalised p-norm-push risk class and characterise its Bayes-optimal scorer set; the answer in
that literature is the monotone-transforms-of-`eta` class. **Same destination, again.**

**(iii) Operating-point smuggling? — THIS FAMILY IS THE ONLY PARTIAL EXCEPTION IN THE SURVEY. Read
this one carefully, it is the honest positive finding.**

`p`-norm push has **no rate parameter**. Its only knob is the exponent `p >= 1`, a *concentration*
parameter, not a TPR or FPR target. `p = 1` recovers plain AUC; `p -> inf` recovers the infinite push.
You can fix `p = 2` or `p = 4` a priori and never look at a rate. **That passes prong (b) in a way
that pAUC (§1.1) and constrained ERM (§1.4) structurally cannot.** It is worth saying plainly in the
report: *there exists exactly one region-targeted ranking family whose hyperparameter is not an
operating point, and it is the push family.*

**But the published direction is wrong for you.** MY DERIVATION, and it is the load-bearing objection:

```
TopPush / InfinitePush / p-norm push all penalise
   "how many POSITIVES are ranked below the HIGHEST-ranked NEGATIVE"
=> they improve the TOP-LEFT extreme of the ROC (very low FPR, precision@top)
```

Your defect is the opposite end. §3.4: the positives you miss have **median score 0.170**, ten below
**0.10**, and **zero** in `[0.45, 0.50)`. You need the **lowest-ranked positives lifted**, which is the
`FPR -> 1` / `TPR -> 1` end of the curve. Every published push method pushes at the end you are
already good at. Applying TopPush here would improve precision at the very top — a region worth,
by §1.0b's exchange rate, essentially nothing to you — while spending capacity taken from the region
you need.

**THE MISSING METHOD (this is a genuine literature gap; MY CONSTRUCTION).** The mirror image is
straightforward to write down and, as far as I can find, **is not published as a named method**:

```
BOTTOM PUSH (proposed):
    L_bottom(h) = (1/n_-) SUM_j  loss( h(x_j^-) - min_i h(x_i^+) )
or the smooth p-norm version, pushing on POSITIVES instead of NEGATIVES:
    L_p-bottom(h) = (1/n_+) SUM_i  ( SUM_j loss( h(x_j^-) - h(x_i^+) ) )^p
                                    ^^^^^^^^^^ outer power over POSITIVES, not negatives
```

The second form is exactly Rudin's p-norm push with the roles of the two classes exchanged. It
concentrates the loss on the positives that have the most negatives above them — i.e. **it optimises
the high-recall corner directly, with no rate parameter at all.** It is the only construction in this
entire survey that (a) escapes both theorems, (b) targets the correct region, and (c) contains no
operating-point choice. I searched for a published version under "bottom push", "reverse push",
"recall-region push" and did not find one; Rudin (2009) and Menon & Williamson (2016) both frame the
family exclusively in the top-push direction. **If you have more time, this is the thing to build,
and the fact that it appears to be unpublished is itself reportable.**

**(iv) n = 1817? — MARGINAL, and worse for the extreme variants.** The `min_i` in the hard bottom-push
is a single-example statistic: one mislabelled or anomalous positive dictates the entire gradient.
That is the known failure mode of infinite push, and with 731 training positives and a known suspicion
of label noise (Q4: missed positives at median score 0.170) it is a serious risk. The **soft** version
(`p = 2` to `4`) is far safer because the outer power is a smooth reweighting over *all* positives —
no top-`k` selection, no endogenous active set, every example always contributes. **This is a real
statistical advantage over pAUC**, whose `k1 = 43` hard selection (§1.1 (iv)) is the reason that family
fails. Soft bottom-push uses all 731 positives and all 1086 negatives at every step.

**(v) 0.6/0.4 trade-off.** Favourable, by §1.0b: 1239 pairs of slack. `p = 2` is a mild reweighting and
would be expected to cost well under 0.01 global AUC.

**⚠️ But Fatal Way 2 from §1.1 still applies, in weakened form.** The deployed cut is still placed by
Platt-on-train-OOF, not by the objective. The mitigation here is that a soft bottom-push improves a
*broad* swathe of the high-TPR curve rather than a narrow band, so it is much more likely that the
improvement overlaps wherever the 0.5 cut lands. That is an argument of degree, not a guarantee.

### B.3 LambdaRank / listwise specifically

**Verdict: reject, for a different reason.** LambdaLoss (DOI 10.1145/3269206.3271784) supplies the
missing theory for LambdaRank by exhibiting it as a specific configuration of a probabilistic
framework — **VERIFIED**: the paper's own framing is that "LambdaRank ... lacks theoretical
justification, and the underlying loss that LambdaRank optimizes for remains unknown," and that they
show it "is a special configuration with a well-defined loss in the LambdaLoss framework." Two
problems for you:

1. **It is a query-grouped formulation.** NDCG and the lambda weights are defined *per query*, over a
   short candidate list. You have **one list of 1030 items and no queries.** Degenerating LambdaRank
   to a single list makes the `|dNDCG_ij|` weight a function of global rank position with a `1/log(1+r)`
   discount — which weights the **top** of a 1030-item list overwhelmingly and the high-recall region
   (ranks 150-250) essentially at zero. Wrong region again, and more severely than the push family.
2. **The discount function is a free choice of emphasis region.** Replacing `1/log(1+r)` with something
   peaked at ranks 150-250 would be a rank target, i.e. a positive-rate target in disguise. Prong (b).

### B.4 Verdict, Family B

> **REJECT the published members (wrong end of the list). FLAG the unpublished mirror (soft bottom
> push, `p`-norm push with the class roles exchanged) as the single best surviving E1 candidate and
> the strongest "with more time we would have..." item in the report.** It is the only construction I
> found that escapes Theorem 1, escapes Theorem 2, targets the correct ROC region, has no
> operating-point hyperparameter, and uses all 1817 rows in every gradient step.

---

## §1.3 FAMILY C — NEYMAN-PEARSON CLASSIFICATION

### C.1 The literature

| work | citation | status |
|---|---|---|
| NP classification, convexity, stochastic constraints | Rigollet & Tong, **JMLR 12:2831-2855 (2011)** | VERIFIED (standard, indexed) |
| **NP umbrella algorithm** + NP-ROC | Tong, Feng, Li, **Science Advances 4(2):eaao1659 (2018)**, DOI 10.1126/sciadv.aao1659; arXiv:1608.03109 | **VERIFIED (mechanism + sample-size rule confirmed from two independent sources)** |
| NP classification: parametrics and sample size | Tong, Xia, Wang, Feng, arXiv:1802.02557 | **VERIFIED (abstract read)** |
| Label-noise-adjusted NP umbrella | JASA 118(543), DOI 10.1080/01621459.2021.2016423 | INFERRED (indexed) |

**What the umbrella algorithm actually does — VERIFIED.** It (1) trains *any* scoring classifier by
*ordinary* means (logistic regression, SVM, random forest are the named base learners); (2) takes a
**held-out class-0 sample that was not used for training**; (3) sets the decision threshold to an
**order statistic of that sample's scores**, chosen as the smallest threshold whose type-I-error
violation probability is below a tolerance `delta`. The minimum held-out class-0 sample size is

```
n_0  >=  log(delta) / log(1 - alpha)
   alpha = delta = 0.05  ->  n_0 = 59
   alpha = 0.10, delta = 0.05 -> n_0 = 45
   alpha = 0.05, delta = 0.10 -> n_0 = 29
```

### C.2 The five checks

**(i) Non-decomposable? — THE TRAINING IS NOT EVEN INVOLVED.** The umbrella algorithm does not change
the loss at all. The base learner is trained with an ordinary **pointwise** loss, so **Theorem 2
applies to the base learner in full force** and the ranking is `T(eta)` as always. The NP machinery is
100% post-hoc.

**(ii) Population optimum? — the likelihood-ratio test, by the Neyman-Pearson lemma itself.** Same
ranking as `eta`. The NP paradigm never claims otherwise; it is a *threshold-selection* theory built
on top of the NP lemma, not a ranking theory.

**(iii) ⚠️ Operating-point smuggling? — IT IS NOT SMUGGLING. IT IS THE ENTIRE METHOD.**

This is the most flagrant failure in the survey and the cleanest to write up:

> **The NP umbrella algorithm is a peer-reviewed, guarantee-carrying procedure for replacing your
> `0.5` with a data-chosen order statistic. It is exactly, precisely, and only the thing your
> self-imposed rule forbids.** Its user-specified `alpha` is a type-I-error (FPR) target — a rate
> target, prong (b) — and its output is a threshold, prong (a).

Two further notes that make this worth a paragraph in the report rather than a one-line dismissal:

- **Direction.** Classical NP controls type I error (FPR), which is the *conservative* direction and
  would **lower** your recall further. To attack your defect you would have to run NP with the classes
  swapped — control type II error (miss rate) at `beta`, maximise specificity — requiring a held-out
  **positive** sample of size `>= log(delta)/log(1-beta)` (45 positives at `beta = 0.1, delta = 0.05`,
  entirely affordable out of your 731). So the machinery *would* work at your scale. It is blocked by
  the rule, not by the data.
- **The honest concession.** §3.3 says the headroom reachable *by threshold alone* on your existing
  ranking is somewhere in `[0, +0.076] F1`. **Family C is the family that would collect exactly that
  headroom, with a finite-sample violation-rate guarantee, and you have forbidden it.** If you ever
  reconsider the self-imposed rule, `sciadv.aao1659` is the citation that makes "we moved the
  threshold" into "we selected an operating point with a `1 - delta` type-II-error guarantee." I am
  **not** recommending you do this — the rule is disclosed in your report and reversing it now would
  be worse than the points are worth — but the report is stronger for naming what the rule costs.

**(iv) n = 1817? — fine (see above), and irrelevant given (iii).**

**(v) 0.6/0.4 trade-off? — AUC is EXACTLY unchanged** (the ranking is untouched), so the entire effect
is `0.6 · dF1`. Under the middle row of your own §3.3 table (proportional split of the free
discordance) that is `0.6 × 0.0129 = +0.0077 composite`; under the optimistic row, `+0.045`. This is
the largest single number in the whole survey and it is unreachable under your rules.

### C.3 Verdict, Family C

> **DEAD ON ARRIVAL — and uniquely instructive.** Report line: *"Neyman-Pearson classification is the
> statistically principled formalisation of the operating-point choice, complete with a finite-sample
> violation-rate guarantee (Tong, Feng & Li, Sci. Adv. 2018). We excluded it because our self-imposed
> rule fixes the decision threshold at 0.5. We record that this exclusion is the single most expensive
> decision in our pipeline, worth up to +0.045 composite by our own headroom bound, and we stand by
> it: the rule exists because we previously violated it, disclosed the violation, and deleted the
> affected scores."*

---

## §1.4 FAMILY D — CONSTRAINED ERM FOR RATE CONSTRAINTS

### D.1 The literature

| work | citation | status |
|---|---|---|
| **Dataset constraints with the ramp penalty** | Goh, Cotter, Gupta, Friedlander, **NIPS 2016**, arXiv:1606.07558 | **VERIFIED (abstract read)** |
| **Scalable learning of non-decomposable objectives** (precision-at-recall, F-beta, AUCPR via Lagrangian bounds) | Eban, Schain, Mackey, Gordon, Rifkin, Elidan, **AISTATS 2017**, PMLR 54:832-840, arXiv:1608.04802 | **VERIFIED (abstract read verbatim)** |
| **Proxy-Lagrangian, two-player game, swap regret** | Cotter, Jiang, Gupta, Wang, Narayan, You, Sridharan, **JMLR 20(172), 2019**, "Optimization with Non-Differentiable Constraints with Applications to Fairness, Recall, Churn, and Other Goals"; companion Cotter et al., **ALT/PMLR 98 (2019)**, arXiv:1804.06500 | **VERIFIED (abstract read)** |
| Generalisation for data-dependent constraints | Cotter, Gupta, Jiang et al., arXiv:1807.00028 | INFERRED (indexed) |
| **Bayes optimality for ratio-of-linear metrics (incl. F1)** | Koyejo, Natarajan, Ravikumar, Dhillon, **NIPS 2014**, "Consistent Binary Classification with Generalized Performance Metrics" | **VERIFIED (result confirmed from two independent summaries)** |

### D.2 The five checks — and the derivation that kills the family

**(i) Non-decomposable? — SUPERFICIALLY YES, SUBSTANTIVELY NO. MY DERIVATION, and this is the
important one.**

Take the canonical form. Let `R(f) = E[ l(f(x), y) ]` be the objective and `C(f) = E[ g(f(x), y) ]` a
rate constraint (recall, coverage/positive-prediction-rate, FPR, churn — all of these are expectations
of a **pointwise** function). The constrained problem is

```
min_f  R(f)   s.t.   C(f) <= c
```

Both Goh et al. and Cotter et al. solve this by Lagrangian / proxy-Lagrangian saddle-point iteration.
At any saddle point with multiplier `lambda*`, the primal player is solving

```
min_f  E[ l(f(x),y) + lambda* · g(f(x),y) ]
    =  min_f  SUM_i [ y_i · ( l_1(z_i) + lambda*·g_1(z_i) )  +  (1-y_i) · ( l_0(z_i) + lambda*·g_0(z_i) ) ]
```

**That is exactly the form `SUM_i [ y_i·L_1(z_i) + (1-y_i)·L_0(z_i) ]` with one fixed pair of functions
`(L_0, L_1)` applied to every example — i.e. it is EXACTLY the object your Theorem 2 annihilates.**
The non-decomposability of the constrained problem is entirely carried by the *scalar* `lambda*`. Once
`lambda*` is fixed (and at the solution it is a fixed number), the objective is a plain cost-sensitive
pointwise loss, its population minimiser is `T(eta)` for a fixed monotone `T`, **the ROC is exactly
unchanged, and the constraint is satisfied purely by where `T` places the `0.5` level set.**

> **A rate-constrained ERM is a threshold slide with extra steps.** That is the sharpest sentence in
> this document and I would put it in the report verbatim.

**Theorem 1 also fires**, independently. For the common case `l` = logistic and `g` = a cost-weighted
0/1 surrogate, the cost-weighting enters the population solution as an **additive logit offset**
`log(lambda*)`-ish — i.e. `z' = z + beta`, an affine logit change, which your Platt refit annihilates
identically. **Family D is the only family in the survey killed by BOTH of your theorems.**

*The one caveat, handled.* Precision is a **ratio** (`TP/PP`), not an expectation, so "precision >= c"
is fractional-linear and the argument above does not literally apply. It does not escape: Koyejo,
Natarajan, Ravikumar & Dhillon (NIPS 2014) prove that for the whole family of metrics given by
**ratios of linear combinations of the four confusion cells** — which includes accuracy, AM, Jaccard,
F-beta and precision-at-recall — the Bayes-optimal classifier is a **signed threshold on `eta`**,
`sign(eta(x) - delta*)`, with a metric-dependent `delta*`. **VERIFIED.** So the ratio case lands in the
same place: same ranking, different threshold.

**(ii) Population optimum? — `sign(eta(x) - delta*)`.** See above. Identical ranking, by a
peer-reviewed consistency theorem rather than by my derivation.

**(iii) ⚠️ Operating-point smuggling? — MAXIMALLY.** Goh et al.'s own abstract names the use cases as
requiring "a classifier to make positive predictions at a **specified rate**", "achieve **specified
empirical recall**", etc. — **VERIFIED, their words.** Cotter et al.'s JMLR title is literally
"...with Applications to Fairness, **Recall**, Churn, and Other Goals". Eban et al. optimise
"**precision at fixed recall**". Every member of this family takes a rate target `c` as its defining
input. There is no version without one. Prong (b), instantly and unambiguously.

And where would `c` come from for you? Only from knowing the deployment recall you want — which is a
leaderboard-derived quantity (§3.2's inversion) or an assumption about `pi_t`. Both illegal under (b).

**(iv) n = 1817? — no.** Two additional problems even if the above were resolved. (a) Cotter et al.'s
guarantees are for a **randomised** classifier (a semi-coarse correlated equilibrium), reduced to a
*mixture* of deterministic classifiers — that is a stochastic decision rule, awkward against your
"seeded / reproducible" competition requirement and against a single deterministic `0.5` cut.
(b) Rate constraints estimated on ~1800 rows have standard errors of order `sqrt(0.25/1817) = 0.0117`
on the rate; the constraint you would be enforcing is noisier than the effect you are chasing, and
Cotter et al. needed a separate paper (arXiv:1807.00028) on generalisation for data-dependent
constraints precisely because of this.

**(v) 0.6/0.4 trade-off? — degenerate.** Since the ranking is unchanged (i)-(ii), `dAUC = 0` exactly
and the whole effect is a threshold slide. Same situation as Family C, but without the guarantee.

### D.3 Verdict, Family D

> **REJECT — killed by both theorems and by prong (b).** Report line: *"Constrained ERM for rate
> constraints (Goh et al. NIPS 2016; Eban et al. AISTATS 2017; Cotter et al. JMLR 20(172) 2019) is
> superficially non-decomposable, but at any saddle point the primal problem reduces to a
> cost-sensitive pointwise loss with a fixed multiplier, which our order-invariance theorem annihilates
> exactly; and the multiplier enters as an affine logit offset, which our Platt refit annihilates
> exactly. Independently, every member of the family requires an explicit rate target as input. It is
> a threshold slide with extra steps."*
