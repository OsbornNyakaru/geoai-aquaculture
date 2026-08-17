# Post-mortem: which offline criterion would have predicted the PRIVATE leaderboard?

Competition CLOSED 2026-08-17. Private revealed. Final: rank 120/500, private 0.910686008.
Oracle over our own 91 submissions: 0.920818161 (submission_tcons_s13). Winner 0.956900206.

Metric = 0.6*F1@0.5 + 0.4*ROC-AUC. Split solved exactly: public n=333 P=181 N=152; private n=697
P=379 N=318; true test prevalence 560/1030 = 0.5437.

**Rule of this document: I append after every sub-result. Nothing is buffered.**

---

## PLAN (written before any number was computed)

1. **Join** `submissions/preds/*.npz` (30 bundles) to `experiments/zindi_submissions_final.tsv`
   (91 rows, 84 unique filenames). Report the match rate honestly. Validate the name join by
   CONTENT where a submitted CSV survives on disk, using the exact F1-cell inversion
   (F1 = 2*TP/(npred+P) with P known and TP,npred integers) to recover each submission's realized
   predicted-positive count and compare it against the CSV.
2. **Criterion panel** per bundle, all computable with ZERO submissions, from `oof_prob`, `y`,
   `oof_view_p` (the R=1 deployment-matched masked replica) and `test_per_fold` (test INPUTS only,
   never labels):
   - plain OOF ROC-AUC (the incumbent), OOF F1@0.5, offline composite 0.6*F1@0.5+0.4*AUC
   - precision@k at the train prior (0.4023) and at the true test prevalence (0.5437, DIAGNOSTIC
     ONLY -- flagged, never used to select)
   - partial AUC in the high-specificity corner FPR in [0,0.2] and [0,0.1], raw and McClish-standardised
   - average precision; Brier; ECE; realized OOF positive rate at 0.5
   - top-of-list-specific: oracle-threshold F1 (F1*), the calibration gap F1*-F1@0.5,
     F1 at the prevalence-pinned cut, recall/precision at k
   - deployment-shift-specific: R=1 vs R=2 view gap, cross-view logit variance,
     and a TEST-side boundary-instability statistic from `test_per_fold`
     (fold disagreement about which side of 0.5 a test row falls on)
3. **Rank criteria** by Spearman/Pearson against TRUE PRIVATE composite, and -- weighted much more
   heavily -- by top-k selection accuracy: what private score would picking finalists by this
   criterion have delivered, vs our actual 0.910686008 and the oracle 0.920818161.
4. **Bootstrap everything.** n is ~13-15, not 20. A 0.05 Spearman edge on n=13 is noise. If the
   honest answer is a null, that is the result.
5. Answer: did any criterion (a) catch the weak top-of-list ordering and (b) reorder our lanes
   relative to public LB? Name it or state that none did.
6. Write the protocol for next time with a PRE-COMMITTED decision bar.

### Hard rules I am operating under
- 0.5 is literal. I am studying which offline criterion PREDICTS the outcome, not tuning a cut.
- 0.5437 may be used as a DIAGNOSTIC comparison only. Any criterion that needs it is disqualified
  from the recommendation and is labelled as such inline.
- Never commit .csv or .npz.

---

## STEP 1 RESULT -- the join (measured)

**Join rate: 13 of 30 bundles, 13 of 91 ledger rows (14.3%), 13 of 87 unique filenames.**
My pre-registered estimate was 13-15. It came in at the bottom of that range.

Joined (stem, private composite):

| stem | pub | prv | auc_prv | f1_prv |
|---|---|---|---|---|
| teacher_perm_s42 | 0.906491753 | **0.916310175** | 0.948772838 | 0.894668400 |
| jtt_balance_s42 | 0.908272635 | 0.913629695 | 0.955394035 | 0.885786802 |
| amix_s13 | **0.912035999** | 0.911317302 | 0.952622757 | 0.883780332 |
| champion_dualpol_add_seedavg5 | 0.907616148 | 0.909757431 | 0.950307827 | 0.882723833 |
| dpa_s21 | 0.906604926 | 0.907885474 | 0.951992167 | 0.878481012 |
| amix_s42 | 0.899019473 | 0.906965167 | 0.946964039 | 0.880299251 |
| amix_s31 | 0.901250519 | 0.905682122 | 0.946764905 | 0.878293601 |
| jtt_lam5_s42 | 0.908387851 | 0.904952196 | 0.949644048 | 0.875157629 |
| amix_s17 | 0.901328638 | 0.904586887 | 0.944026816 | 0.878293601 |
| jtt_control_s42 | 0.902940612 | 0.903154955 | 0.943744710 | 0.876095118 |
| amix_s37 | 0.896810817 | 0.897873972 | 0.942300990 | 0.868255959 |
| teacher_perm_s13 | 0.891730406 | 0.896412034 | 0.937845372 | 0.868789808 |
| smoke_test | 0.745903076 | 0.754214682 | 0.847600438 | 0.691957511 |

### Three corrections to the brief I was given

1. **The npz key list in the brief is wrong.** Real keys: `oof_prob`, `y`, `p_test_raw`,
   `test_ids`, `model`, `prior`, `test_per_fold`, `oof_view_p`, `oof_view_owner`,
   `oof_view_k`, `oof_view_rep`. There is no `test_prob`. Test predictions live in
   `p_test_raw` (1030,) and `test_per_fold` (5,1030).
2. **`oof_view_p` is NOT "the R=1 masked replica".** It is length **3634 = 2 x 1817** --
   two replicas per OOF row, indexed by `oof_view_owner` (row id) and `oof_view_rep`
   (replica index). The R=1 vs R=2 view gap is therefore computable, but by pairing on
   `oof_view_owner`, not by comparing a length-1817 vector to `oof_prob`.
3. **OOF length is 1817, not the 1821 train rows.** 4 rows are dropped somewhere upstream.
   Noted, not chased; it does not affect a between-bundle comparison since all 26 full
   bundles share the identical 1817-row `y`.

Coverage of the view/fold keys within the 13: `smoke_test` and
`champion_dualpol_add_seedavg5` have **no** `test_per_fold` and no `oof_view_*`. So every
deployment-shift criterion is computable on only **11 of 13**.

### THE STRUCTURAL FINDING THAT CONSTRAINS EVERYTHING BELOW

**Neither finalist is in the joined set, and neither is the oracle.**
- Our two `SEL` rows (`champion_dualpolmix10_regimematch`, `champion_archblend4`) have no bundle.
- The global oracle `tcons_s13` (0.920818161) has no bundle.

So the headline framing -- "what would criterion X have delivered vs our actual 0.910686 and
the oracle 0.920818" -- **is not answerable as posed.** No offline criterion computed from
these 13 bundles could have selected `tcons_s13`, because its predictions do not exist on disk.
The honest restatement, which is what I will actually measure:

> Within the 13-bundle universe, public LB selects **amix_s13 -> private 0.911317302**.
> The within-universe oracle is **teacher_perm_s42 -> private 0.916310175**.
> **Contestable headroom = 0.004993**, not 0.010132.

Every selection-accuracy number below is against that 0.004993 bar. Any claim about the full
0.010132 would be extrapolation from 13 points to a submission that was never bundled.

Secondary caveat, pre-registered here before computing: `smoke_test` (prv 0.754) is a
degenerate 300-row run and is a ~0.15 outlier from the rest of the pack. Including it will
inflate every correlation toward 1.0, because every criterion can tell a broken model from a
working one. I will report all correlations **both with (n=13) and without (n=12)**, and I
will treat the **n=12** figure as the honest one.

---

## STEP 1b RESULT -- content validation of the name join

Tool: `tools/offline_criterion_bakeoff.py`. 4 of the 13 joined submissions still have their
CSV on disk. For each I recovered the realized predicted-positive count from the CSV and
compared it against the count implied by inverting BOTH F1 cells (public n=333 P=181,
private n=697 P=379) with an all-solutions enumerator.

| stem | CSV npred | inv pub PP | inv prv PP | sum | verdict |
|---|---|---|---|---|---|
| jtt_balance_s42 | 597 | 188 (unique) | 409 (unique) | 597 | **OK** |
| jtt_control_s42 | 615 | 195 (unique) | 420 (unique) | 615 | **OK** |
| jtt_lam5_s42 | 604 | 190 (unique) | 414 (unique) | 604 | **OK** |
| smoke_test | 408 | {128, 231} ambiguous | 280 (unique) | 408 or 511 | **OK** (408 attained) |

**4/4 validated by content.** The name join is not a guess; for every case testable it
reproduces the exact integer confusion cell. Note `smoke_test` is a genuinely ambiguous
inversion (two public solutions) -- exactly the case the old `post_mortem.invert_f1`
would have silently collapsed to a median. My enumerator returns both and the CSV picks 128.

Independently confirmed on all 4 CSVs: `TargetF1 == (TargetRAUC >= 0.5)` exactly. The 0.5
cut was applied literally in what we actually submitted.

---

## STEPS 2-4 RESULT -- the bake-off. THE HEADLINE HYPOTHESIS IS FALSIFIED.

28 criteria; 21 computable on all 13, 7 requiring the view/fold arrays (11 bundles).
Selection rule scored: take the criterion's top-2, score the better of them -- which is how
Zindi actually scored us.

Baselines inside the 13-bundle universe:
```
universe oracle (best of any)   0.916310175   teacher_perm_s42
public-LB top-2 -> best-of-2    0.911317302   amix_s13 + jtt_lam5_s42
OOF-AUC  top-2 -> best-of-2     0.909757431   jtt_lam5_s42 + champion_dualpol_add_seedavg5
random 2-of-12 expectation      0.909981474
contestable headroom            +0.004992873
```

**Result 1 (measured): the brief's central hypothesis is false.** Top-of-list criteria did
NOT outrank global OOF AUC. At n=12:

| criterion | rho vs private | paired P[beats oof_auc] | selection |
|---|---|---|---|
| `oof_auc` (incumbent) | +0.266 | -- | 0.909757431 |
| `pauc@fpr20` (all 3 variants) | +0.294 | **0.550** | 0.909757431 |
| `avg_precision` | +0.259 | 0.455 | 0.909757431 |
| `pauc@fpr10` (all 3 variants) | +0.231 | 0.454 | 0.909757431 |
| `prec@k_prior` | **-0.188** | 0.134 | 0.909757431 |

`pauc@fpr20` beats the incumbent by rho +0.028 with a paired-bootstrap probability of
**0.550** -- a coin flip. `pauc@fpr10`, the *narrower* high-precision corner, is **worse**
than plain AUC. `prec@k_prior` is outright negative. And every one of them selects
0.909757431, i.e. **worse than the public LB we actually used**. The prediction "a criterion
sensitive to the top of the list should outrank global OOF AUC as a private-score predictor"
does not survive contact with the data.

**Result 2 (measured, and this one is solid): the OOF F1 half of the composite is
ANTI-predictive.** At n=12, `oof_f1@0.5` rho = **-0.420**, paired P[beats oof_auc] = **0.017**
(i.e. ~98% confident it is worse). `oof_composite` = -0.287 (P=0.023), `oof_f1_star` = -0.364
(P=0.033). Sign-stable from n=13 to n=12 and the only direction in the whole panel with a
paired probability far from 0.5. **The offline composite we were reading was worse than
its own AUC term.** Mixing OOF F1 in actively degraded the signal.

**Result 3: nothing except one candidate beats the public LB on selection.** 26 of 28
criteria select 0.9098-0.9137, at or below the 0.911317 that public LB delivered. Public LB
was not a bad selector inside this universe -- it beat OOF AUC by +0.00156.

**Result 4: the sole candidate.** `view_auc_gap` = -|AUC(view 0) - AUC(view 1)| across the
two masked deployment views. rho = **+0.745**, bootstrap CI **[+0.26, +0.94]** (excludes
zero), and it selects **teacher_perm_s42 = 0.916310175, the exact universe oracle, capturing
the entire +0.004993 headroom.** Stress-tested below, because that is far too good.

---

## STEP 5 RESULT -- the candidate dies. This is a NULL.

`view_auc_gap` passes the tests that are easy and fails the two that matter.

**(b) Leave-one-out: PASSES.** rho across the 11 drops stays in [+0.673, +0.830]. Not an
outlier artefact.

**(c) Multiplicity: FAILS.** Permuting the private scores 20k times:
- marginal p (this criterion alone) = **0.0045**
- **family-wise p (max rho over the 28 criteria I actually tried) = 0.0684**

I ran 28 shots at an 11-point target. Honestly accounting for that, +0.745 is not
significant at 0.05. And on the selection claim:
- **P[at least one criterion in the panel picks the oracle by chance] = 0.910.**
- P[one *pre-committed* criterion picks the oracle by chance] = 2/11 = 0.182.

With 28 criteria and 11 candidates it is a near-certainty (91%) that *something* in the panel
lands on the oracle. "It picked the oracle" is worth essentially nothing as evidence.

**(a) Noise floor: FAILS DECISIVELY.** This is the finding that closes it. Paired
row-bootstrap of the signed view-AUC difference within each bundle:

```
stem                signed gap      paired SE   |gap|/SE
amix_s13             +0.000807      0.001770      0.46
amix_s17             +0.003066      0.001887      1.63
amix_s31             +0.000637      0.002447      0.26
amix_s37             +0.000977      0.001232      0.79
amix_s42             +0.001135      0.002491      0.46
dpa_s21              -0.000529      0.002008      0.26
jtt_balance_s42      -0.000117      0.003433      0.03
jtt_control_s42      +0.001052      0.001914      0.55
jtt_lam5_s42         +0.000690      0.001848      0.37
teacher_perm_s13     -0.001177      0.002436      0.48
teacher_perm_s42     -0.000498      0.003134      0.16

between-bundle SD of |gap| : 0.000728
median paired SE           : 0.002008
signal / noise             : 0.36
```

**Zero of 11 bundles have a view-AUC gap distinguishable from zero at 2 SE.** The largest is
1.63 SE. The spread *between* bundles is 0.36x the estimator's own sampling error. The
criterion ranks eleven models by a quantity that is statistically indistinguishable from
zero in every single one of them. rho = +0.745 is a coincidence in noise, and the
family-wise p of 0.068 was already telling me so.

### VERDICT ON STEPS 3-5

**No offline criterion in this panel would have predicted the private leaderboard.** The
apparent winner is noise that survived a bootstrap CI and died on multiplicity plus a noise
floor. I am not going to soften this: with n=11-13 and a headroom of 0.005, this study is
underpowered to detect anything short of a very strong criterion, and no such criterion is
present. The result is a null.

---

## STEP 5b RESULT -- the two questions, answered directly (n=12)

Reference correlations inside the universe:
```
rho(public LB, private composite) = +0.664
rho(public LB, private F1)        = +0.550
rho(public LB, private AUC)       = +0.867
rho(private F1, private AUC)      = +0.753
```
Worth noting on its own: **the public LB tracked the private AUC column well (+0.867) and
the private F1 column poorly (+0.550).** That is consistent with the main post-mortem's
*description* -- the F1 column is where the unpredictability lived. What follows is that its
*prescription* does not work.

### (a) Did any criterion catch the weak top-of-list ordering? **No.**

Best rho against `f1_prv` among legal criteria: `pauc@fpr20` at **+0.266**. That is a weaker
correlation than plain `oof_auc` achieves against the composite, on 12 points, with no
significance to speak of. The `f1 - auc` differential column -- which is where a genuine
top-of-list criterion should show a clear positive -- gives `pauc@fpr20` **+0.014** and
`pauc@fpr10` **+0.070**. Essentially nothing. `prec@k_prior` posts +0.234 only because both
its correlations are negative (-0.132 vs -0.366): it is less bad at F1, not good at it.

And `oof_f1@0.5` -- the offline quantity that most literally mirrors the private F1 column --
correlates with `f1_prv` at **-0.466**. The offline measurement of the thing we lost points on
pointed the wrong way.

### (b) Did any criterion reorder our lanes relative to the public LB? **No -- and this is the sharpest result in the study.**

Sort the panel by rho against the public LB. Everything that carried private signal was
already redundant with the public LB; everything orthogonal to the public LB was noise or
anti-signal:

| criterion | rho(public LB) | rho(private) | reading |
|---|---|---|---|
| `oof_auc` | +0.608 | +0.266 | redundant with public LB |
| `view_auc_gap` | +0.573 | +0.745 | redundant-ish; and killed in step 5 |
| `avg_precision` | +0.434 | +0.259 | redundant |
| `pauc@fpr20` | +0.357 | +0.294 | redundant |
| `oof_f1@0.5` | -0.119 | -0.420 | orthogonal -> **anti-signal** |
| `prec@k_prior` | -0.298 | -0.188 | orthogonal -> **anti-signal** |
| `test_fold_logit_sd` | -0.073 | -0.355 | orthogonal -> **anti-signal** |
| `view_logit_var` | -0.345 | -0.364 | orthogonal -> **anti-signal** |

**Every criterion that would have reordered our lanes would have reordered them downward.**
There is no criterion in this panel that both disagrees with the public LB and is right to.

---

## THE ROOT CAUSE -- why no criterion could have worked, and this is not a power excuse

I went looking for why the whole panel is flat, and the answer is in the ledger, not the
bundles. Private scores of the 12 non-degenerate joined submissions, grouped by arm:

```
arm                              n   mean private
amix                             5     0.905285
teacher_perm                     2     0.906361
dpa                              1     0.907885
jtt_balance                      1     0.913630
jtt_control                      1     0.903155
jtt_lam5                         1     0.904952
champion_dualpol_add_seedavg5    1     0.909757

within-arm seed SD (amix, 5 seeds)   = 0.004868
between-arm SD of arm means          = 0.003514
CONTESTABLE HEADROOM                 = 0.004993   ( = 1.03 x the within-arm seed SD )
```

**The between-arm signal (0.003514) is SMALLER than the within-arm seed noise (0.004868).**
And the entire headroom an offline criterion was competing for is 1.03 seed-SDs.

Then the fact that settles it:

> **The oracle `teacher_perm_s42` (0.916310175) and the WORST submission in the universe
> `teacher_perm_s13` (0.896412034) are the same architecture, the same pipeline, and the
> same code -- two seeds. Their gap is 0.019898, which is exactly the full range of all 12.**

The best and the worst thing we have are one model with two random seeds. So the target an
offline criterion was asked to hit is not "which model is better" -- it is "which seed got
lucky on 697 held-out rows". No statistic computed from OOF predictions can know that,
because it is not a property of the model. This is why the panel is flat, why `view_auc_gap`
had to be noise, and why the honest answer is a null rather than "we needed a bigger n".

This corroborates the standing note that seed variance ~0.019 dominates, and it reframes the
question in the brief. The 0.010 composite "sitting in our own submission set" was not a
better model we failed to recognise. It was seed luck we could not have recognised.

---

## STEP 6 -- THE PROTOCOL FOR NEXT TIME (pre-committed numbers, not advice)

### Design numbers, measured here

Family-wise 0.05 critical Spearman, C independent criteria on n candidates (permutation;
conservative because real criteria are correlated -- the actual panel of 28 behaved like
roughly C=10):

```
   n     C=1     C=3    C=10    C=28
  11   0.527   0.655   0.745   0.809
  12   0.497   0.622   0.720   0.783
  15   0.443   0.554   0.650   0.714
  20   0.379   0.478   0.564   0.629
  30   0.306   0.389   0.464   0.519
  40   0.264   0.338   0.403   0.452
  60   0.214   0.275   0.331   0.372
```
Power of ONE pre-committed criterion against a true rho of 0.60: n=12 -> 0.67,
n=15 -> 0.77, **n=20 -> 0.89**, n=30 -> 0.97.

### The rules I would commit to in advance

**R1. Bundle every submission, at submit time, no exceptions.**
The binding constraint in this study was not statistics, it was bookkeeping: **13 of 91
submissions (14%) had a prediction bundle.** Neither finalist and neither oracle was
bundled. Writing an npz next to every CSV is free and it is the single highest-value change.
*Bar: bundle coverage >= 90% of submissions, checked weekly. Below that, no post-hoc
selection study is worth running.*

**R2. Pre-commit to exactly ONE offline criterion before the first submission, in writing.**
A panel of 28 needs rho >= 0.78 at n=12 to clear FW-0.05; one criterion needs 0.50. The
panel is how `view_auc_gap` got to look real. And **P[some criterion out of 28 picks the
oracle by luck] = 0.91**, versus 0.18 for a pre-committed one. Searching the panel after the
fact is how you manufacture a finding.

**R3. That one criterion is OOF ROC-AUC. Delete OOF F1@0.5 and the offline composite from the dashboard.**
Measured here: `oof_f1@0.5` rho = **-0.420** vs private, paired P[beats oof_auc] = **0.017**;
`oof_composite` -0.287; `oof_f1_star` -0.364. The offline composite is worse than its own AUC
term. We were reading a number that was actively misleading us. `pauc@fpr20` is
indistinguishable from AUC (paired P = 0.550) and does not justify the extra degree of
freedom.

**R4. Noise-floor gate -- apply BEFORE looking at any correlation.**
For any candidate criterion, bootstrap its value *within* each candidate and compute
`between-candidate SD / median within-candidate SE`. **Bar: ratio >= 2.0 to be eligible.**
`view_auc_gap` scored **0.36**, with **0 of 11** candidates distinguishable from zero at 2 SE.
That gate alone would have killed it before its rho of +0.745 was ever computed, and it costs
one bootstrap.

**R5. Selection: the public LB is the primary selector, bounded.**
Measured inside this universe: public-LB top-2 -> 0.911317, OOF-AUC top-2 -> 0.909757,
random -> 0.909981, oracle -> 0.916310. rho(public LB, private) = **+0.664**, higher than
every legal offline criterion. Public LB was not the mistake. *Bar: override the public LB
only when a pre-committed criterion (R2) disagrees AND has cleared R4 AND has rho >= the
C=1 critical value at the current n. On this data nothing ever cleared that; the correct
action was to do nothing.*

**R6. Spend the diversification budget on seeds, not on criteria.**
Within-arm seed SD 0.004868 exceeds between-arm SD 0.003514. Selection cannot beat noise
this large, but **averaging can**: the one seed-averaged entry we bundled
(`champion_dualpol_add_seedavg5`, 0.909757) sits above the amix arm mean of 0.905285 while
having ~1/sqrt(5) of the seed variance. *Bar: every finalist must be an average over >= 5
seeds. Never submit a single-seed model as a finalist.* This is the only lever in this whole
study that provably moves the private score, and it needs no criterion at all.

**R7. Declare the null out loud.**
If the pre-committed criterion does not clear its bar, the protocol output is "no reorder",
and the public LB pick stands. Write that down as a result, not as a failure to find one.

### What I am NOT recommending
- Nothing involving the test prevalence. Every `DQ_` criterion used the revealed 0.5437 and
  is disqualified by construction. For the record they were not good anyway: best
  `DQ_recall@k_truePrev` rho +0.286 at n=12, selection 0.913630 -- it does not even reach the
  oracle, so knowing the answer in advance would not have saved us.
- Nothing top-of-list. That was the hypothesis I was asked to test and it failed (see above).
