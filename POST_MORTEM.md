# Post-mortem — GeoAI Aquaculture Pond Identification Challenge (FAO / ITU)

**Final: rank 120 of 500, private 0.910686008.** Winner 0.956900206. Gap 0.046214.

Captured on 2026-08-17 from the closed competition, after the private leaderboard was revealed.
Everything below is measured, not recalled. Sources:

| artifact | what it is |
|---|---|
| `experiments/zindi_submissions_final.tsv` | all **91** of our submissions — Zindi id, timestamp, **the CSV filename we uploaded**, finalist flag, public/private composite, and the AUC and F1 sub-columns for each |
| `experiments/zindi_final_leaderboard_top75.tsv` | final board ranks 1–75 plus our row, with the same four score columns per team |
| `tools/post_mortem.py` | reproduces every number in this document (`python tools/post_mortem.py`) |

---

## 0. The scoreboard

Two finalists **were** designated: `submission_champion_dualpolmix10_regimematch.csv` (public
0.910446704 → private 0.910686008) and `submission_champion_archblend4.csv` (0.899642643 →
0.908829923). The first one scored.

| selection rule | private | finish |
|---|---|---|
| **what we scored** | **0.910686** | **~#120** |
| Zindi default — best public, designate nothing | 0.913674 | ~#108 |
| oracle over our own 91 | 0.920818 | ~#79 |
| our global AUC + median top-75 team's local ranking | 0.924961 | #64 |
| our global AUC + 1st place's F1 | 0.943422 | #20 |
| our F1 + 1st place's AUC | 0.924164 | #67 |
| 1st place | 0.956900 | #1 |

Read that table as three separate failures of very different sizes: selection cost ~12 places,
our own best-vs-chosen cost ~41, and the model cost the remaining ~78.

---

## 1. Five things we now know for certain

### 1.1 The public/private split — solved exactly, and we had it wrong for 42 iterations

The half-integer AUC sieve (`tools/lb_cell_solve.py`), run against **both** columns at once with
`n_pub + n_prv = 1030` as a hard constraint, returns exactly **one** surviving solution:

```
n_pub = 333  P_pub = 181  (prevalence 0.5435)
n_prv = 697  P_prv = 379  (prevalence 0.5438)
overall test prevalence = 560/1030 = 0.5437
```

Two long-standing repo assumptions die here:

- **`n_public = 309` (30% of 1030) was wrong.** It was inferred, never read, and it propagated into
  every confusion-cell diagnosis from iter42 onward. Round 24 caught and corrected it on the last
  day; this is the independent confirmation.
- **"test pos-rate ~0.65" was wrong.** The truth is **0.5437**. Of our prevalence estimators, BBSE
  (0.559) was the closest and MLLS (0.578) second — and we *retired both* at iter41 in favour of a
  graph estimator reading 0.59, the worst of the three. `REPORT.md` §5.4 already retracted that
  choice; the exact number now confirms it.

Our realized positive rate was 0.5736 public / 0.5854 private against a true 0.5437. **We
over-predicted by about 4 points, ~40 rows.**

### 1.2 The composite formula is exactly 0.6·F1 + 0.4·AUC

Verified on 182 independent triples (91 submissions × 2 splits); max residual 8e-10. No hidden
term. We had fitted this on 5 rows.

### 1.3 The gap to first place is 71% F1, 29% AUC

```
us   AUC 0.950158  F1 0.884371  composite 0.910686
1st  AUC 0.983854  F1 0.938931  composite 0.956900
gap  0.046214 = 0.013478 from AUC (29%) + 0.032736 from F1 (71%)
```

### 1.4 The F1 deficit is **local ranking**, not calibration — H_shape, in the high-precision corner

This is the finding that matters most, and it settles a question the repo argued about for 24
rounds without evidence.

Take every team whose **private AUC is within 0.005 of ours** — an equally good global ranking — and
invert their private confusion cell (each inversion is *unique* on n=697, P=379, so these are exact,
not estimates):

| rank | team | AUC_prv | F1_prv | TP | PP | precision | recall | pos-rate |
|---|---|---|---|---|---|---|---|---|
| 49 | pmaurente | 0.945570 | 0.921438 | 346 | 372 | 0.9301 | 0.9129 | 0.5337 |
| 45 | simonMakumi | 0.950192 | 0.919598 | 366 | 417 | 0.8777 | 0.9657 | 0.5983 |
| 56 | tw_zent | 0.949038 | 0.916230 | 350 | 385 | 0.9091 | 0.9235 | 0.5524 |
| 61 | Mutombwa | 0.952515 | 0.908163 | 356 | 405 | 0.8790 | 0.9393 | 0.5811 |
| 71 | lesleygrin | 0.947313 | 0.906702 | 345 | 382 | 0.9031 | 0.9103 | 0.5481 |
| 68 | mwarsssss | 0.950013 | 0.906005 | 347 | 387 | 0.8966 | 0.9156 | 0.5552 |
| 75 | Nayal_17 | 0.951801 | 0.901660 | 353 | 404 | 0.8738 | 0.9314 | 0.5796 |
| **120** | **us** | **0.950158** | **0.884371** | **348** | **408** | **0.8529** | **0.9182** | **0.5854** |

Median peer F1 at our AUC: 0.908163. Ours: 0.884371. **−0.024**, worth −0.0143 composite and ~56
places — from an equally good global ranking.

**And it is not the operating point.** Restrict to peers predicting within ±15 of our 408 positives:

- Mutombwa: PP 405 (−3 predictions) → TP 356 (**+8** true positives)
- Nayal_17: PP 404 (−4) → TP 353 (**+5**)
- simonMakumi: PP 417 (+9) → TP 366 (**+18**)

At the *same* operating point they capture ~6 more real ponds than we do. Since global AUC is
identical, their discordant pairs sit deep in the list where nothing is decided, and ours straddle
the boundary. Our ranking is fine on average and **bad exactly where the 0.5 cut falls.**

Two direct consequences for the record:

- `REPORT.md` §3.2 asserted the leader's advantage lived in the **high-recall** corner and §3.3
  proved that unidentified. The answer is **H_shape, high-precision corner** — the mirror. Round 24
  flipped to the FP side on the final day and was right, with no time to act.
- Round 24's partial-AUC survey rejected families A–D. It was the right *family* aimed at the wrong
  *corner*. That work is reusable, not wasted.

### 1.5 Our compliance stance was correct and cost us nothing

Verbatim from the competition Evaluation page:

> "Setting a probability threshold is strictly forbidden. Your binary target should be based on the
> default threshold of 0.5."

So `compliance_mode='legal'` was right, and the F1 gap is **not** explained by rivals tuning
thresholds while we refused. Note also that the top finishers over-predict *more* than we do
(#1 pos-rate 0.5839, #8 0.6212 vs our 0.5854) and still beat us on precision. Nothing here would
have been fixed by moving the cut.

---

## 2. Five things we did wrong

### 2.1 We killed the arm that produced both our best public and our best private entry

```
submission_tcons_s42.csv           pub 0.914179 (our best ever)   prv 0.913674  (#7 of 91)
submission_tcons_s13.csv           pub 0.908873                   prv 0.920818  (BEST of 91)
submission_champion_tcons_seedavg5 pub 0.893752                   prv 0.901721

member mean  prv 0.917246   AUC 0.939190   F1 0.902618
5-seed pool  prv 0.901721   AUC 0.940650   F1 0.875769
pooling gain     -0.015525  =  +0.000584 AUC   -0.016109 F1
```

ARM T (the `Var_k(logit)` cross-view penalty aimed at the unlabeled test rows) was written off at
iter41 as a single-seed mirage. **It wasn't** — both seeds held up on private.

The iter41 log diagnosed the mechanism *correctly*: "pooled AUC sits BETWEEN its members' so the
RANKING pooled normally, but pooled F1 is BELOW BOTH members' → the OPERATING POINT moved… per-seed
Platt slopes diverge." Private confirms every word — pooled AUC is fine (+0.0006), the entire
−0.0155 is F1.

Inverting the private cells shows *how* the operating point moved. The pool did not rank worse and
did not lose true positives — it **gained** 4 TP and a point of recall, and paid for them with
**27–39 extra false positives**:

| | TP | PP | precision | recall | F1 |
|---|---|---|---|---|---|
| tcons_s13 | 352 | 395 | 0.8911 | 0.9288 | 0.909561 |
| tcons_s42 | 352 | 407 | 0.8649 | 0.9288 | 0.895674 |
| **5-seed pool** | **356** | **434** | **0.8203** | **0.9393** | **0.875769** |

At a true prevalence of 0.5437 those extra predictions are almost all wrong, so the entire −0.0161
is a **precision** loss with the ranking intact. Note the direction: arithmetic averaging compresses
the pooled distribution toward the members' centre of mass, and because our score mass sits *above*
0.5 (every artifact realizes a positive rate of 0.55–0.60), compression pulls sub-threshold rows
**up** across the cut. The pool **over**-predicts. `tools/repool.py` originally asserted the
opposite sign without checking where the mass sits.

> **⚠️ RETRACTED 2026-08-17.** This section used to conclude: *"The broken component was
> `calibrated_pool`… averaging in rank space, or refitting one Platt on the pooled logit, fixes it
> and changes nothing else."* That was never measured, and it is **false**. Measured across all five
> families on disk (`gemini_loop/findings/postmortem_pooling.md`, Results 1–3): probability
> averaging, geometric-mean-of-odds and pool-then-calibrate land within **2 rows of 1030** of each
> other, and are **bit-identical** on `dpa` and `amix`. Rank-averaging moves the ARM T pool from 643
> predicted positives to ~610 — still past both members. Swapping the combiner would have changed
> our submitted binary column by 0–2 rows. The Ranjan & Gneiting / Rahaman & Thiery result is true
> and *operationally irrelevant* at this member correlation.
>
> The "pooled F1 below every member while pooled AUC sits inside the member range" signature occurs
> in **0 of 5** families; in the three seed families the pool's OOF F1 is *above* every member. And
> OOF F1@0.5 is structurally blind to this failure anyway: calibrated OOF positive rate is pinned to
> the train prior (0.397–0.403) while test runs at 0.55–0.60, so the OOF operating point is right by
> construction.

**We had the right diagnosis and still discarded the wrong thing — but the recoverable loss was
selection, not pooling.** `tcons_s42` was the highest public score of all 91 submissions and we did
not designate it; `tcons_s13` was the highest private. The plain Zindi default (best public) beats
our actual result by +0.002988; "top-5 by public, keep the best private" by +0.008922. We judged an
arm by its pooled artifact and threw the members away with it. Cost: our best private score.

The counterfactual "would the correct pooler have saved ARM T?" is **indeterminate**, not favourable.
Bounding from the LB cells with AUC held at 0.940650, the pool's F1 at s13's operating point lies
anywhere in [0.8191, 0.9199] — composite [0.8677, 0.9282], a band that straddles the 0.910686 we
actually scored. Any single number here would be invented.

### 2.2 We spent five weeks inside the noise band

| week of | subs | best public | best private | running best private |
|---|---|---|---|---|
| 2026-07-06 | 12 | 0.877994 | 0.891984 | 0.891984 |
| 2026-07-20 | 24 | 0.898566 | 0.905704 | 0.905704 |
| 2026-07-27 | 7 | 0.899643 | 0.908830 | 0.908830 |
| 2026-08-03 | 20 | 0.913263 | 0.919608 | 0.919608 |
| 2026-08-10 | 27 | 0.914179 | 0.920818 | 0.920818 |
| 2026-08-17 | 1 | 0.894899 | 0.904150 | 0.920818 |

Week 1 reached 0.892. Seventy-nine further submissions bought **+0.029**. The sd of the
public→private shift is **0.0121**, and the private range across five seeds of one fixed config is
**0.0134**:

```
amix_s{13,17,31,37,42}   prv 0.897874 - 0.911317  (range 0.0134)
teacher_perm_s{13,42}    prv 0.896412 - 0.916310  (range 0.0199)
tcons_s{13,42}           prv 0.913674 - 0.920818  (range 0.0071)
```

Almost every A/B we adjudicated after week 1 was smaller than the seed range of a single config.
The repo's own protocol note said the combined SE was ≈0.012 — we wrote it down and then ignored it.

### 2.3 Public leaderboard reliability — good on ranking, useless at the top

```
composite  pearson 0.9807  spearman 0.9341
AUC        pearson 0.9905  spearman 0.9789
F1         pearson 0.9669  spearman 0.8648
private is +0.012 higher than public on average (the private slice is the easier one)
```

But where it mattered: **overlap of our public top-5 with our private top-5 was 1 of 5.** Our
public #1 fell to private #7; our private #1 was only public #7; `jtt_lam5` went public #8 →
private #32. The public board ranked the field well and the *tip* of the field not at all.

### 2.4 The finalist choice lost to doing nothing

We designated two members of the **same champion family** (`dualpolmix10_regimematch` and
`archblend4`). Had we designated nothing, Zindi's default — our best public entry — would have
scored 0.913674 and ~#108, twelve places better. `top-5 by public, take the best` would have
returned 0.919608, ~#82.

The second slot was spent on a hedge that was correlated with the first and 0.011 worse on public.
A genuinely decorrelated second pick (a different model family — `tcons`, `c_perm_meanmin`,
`teacher_perm`) would have paid.

### 2.5 Roughly a sixth of the submission budget bought nothing

Of 91 uploads: 87 distinct filenames, 8 uploads were re-uploads of 4 files, and 12 uploads share an
exact (public, private) score pair with another upload — byte-equivalent decisions under a new name.

---

## 3. What to do differently next time

1. **Read platform facts; never infer them.** `n_public = 309` was a guess that survived 42
   iterations and corrupted every confusion-cell diagnosis downstream. Any number used in reasoning
   must be read off the platform or *derived with proof*. `tools/lb_cell_solve.py` does the derivation
   for any AUC+F1 competition — run it in week 1, not on the last day.

2. **Establish the noise floor before chasing anything.** Five seeds of the baseline, offline, day
   one. Publish the sd. Then pre-commit: *no lane is adjudicated on a delta smaller than 2 sd.* We
   computed SE ≈ 0.012 and then spent a month resolving 0.006 differences.

3. **Optimize the metric's actual geometry — but do NOT promote a top-of-list statistic to the
   selection criterion.** 60% of the score was F1 at a *frozen* 0.5 cut, which depends only on the
   ordering of the top ~55% of rows and where probabilities sit relative to 0.5; global AUC — 40% —
   is dominated by pairs that never touch the decision. That geometry argument is sound and it is
   why the §1 peer comparison points at the high-precision corner. It does **not** license the
   prescription this recommendation originally carried.

   > **⚠️ REWRITTEN 2026-08-17.** The original text read: *"Make precision@k and F1@0.5 the primary
   > offline metric from day 1, with AUC secondary."* Directly measured against the revealed private
   > board (`gemini_loop/findings/postmortem_offline_criteria.md`, 28 criteria × 12 bundles): the
   > corner criteria did **not** beat global OOF AUC — `pauc@fpr20` ρ = +0.294 vs `oof_auc` +0.266,
   > paired P = 0.550, a coin flip; the *narrower* corner `pauc@fpr10` is worse; `prec@k_prior` is
   > **−0.188**. And the F1 half of the prescription is actively harmful: `oof_f1@0.5` scores
   > ρ = **−0.420** with P = 0.017 that it beats `oof_auc`, i.e. it is reliably **anti**-predictive,
   > and the offline composite (−0.287) is worse than its own AUC term. Keep **OOF AUC** as the
   > single selection criterion and **delete OOF F1 and the offline composite** from the selection
   > path. Aim the *loss* at the high-precision corner if you like; do not aim the *criterion* there.

   The null is structural, not a power excuse. **Within-arm seed SD (0.004868) exceeds between-arm SD
   (0.003514).** The best and worst entries in the bundled set — private 0.916310 and 0.896412 — are
   *the same model at two seeds*, and their 0.019898 gap is the entire range of all twelve. We were
   not ranking models; we were ranking which seed got lucky on 697 rows, and no OOF statistic can
   know that. Beware also of finding a winner in a panel this wide: `view_auc_gap` posted ρ = +0.745
   and picked the exact oracle, but its family-wise p is 0.068, P[*some* criterion picks the oracle
   by chance] = 0.910, and its signal/noise is 0.36 — zero of 11 bundles have a gap distinguishable
   from zero at 2 SE. Gate on a **noise floor of ratio ≥ 2.0 before computing any correlation**, and
   pre-commit to exactly **one** criterion (at n=12, C=1 needs ρ ≥ 0.50; C=28 needs 0.78).

4. **When an ensemble underperforms its members, never judge the arm by the pool — but do not
   assume the combiner is the bug either.** *(Rewritten 2026-08-17; the original recommendation here
   said "pool in rank space by default", which the measurement in §2.1 falsifies — at this member
   correlation all three combiners agree to within 2 rows of 1030.)* The rule that survives: a pooled
   artifact is **one** sample of the arm, so score the **members** too before killing a lane. The
   check to automate is not the F1-vs-members signature (0 of 5 families show it, and OOF cannot see
   an operating-point failure) but the **operating-point escape**: assert that the pooled *test*
   positive rate sits inside the members' own range widened by `max(sd, 0.005)`. That check passes on
   all five families on disk (overshoot 0, 0, 0, 0, 2 rows) and trips on the ARM T replay at +24 rows.
   Shipped as `src/calibration.py::assert_pool_sane`, raising on every `calibrated_pool()` call.

5. **Finalist policy, decided in advance.** Slot 1 = best public, always (it beat us here). Slot 2 =
   the best entry from the most *decorrelated* model family, chosen by prediction correlation, never
   a second variant of slot 1. Designate a week before the deadline and only revise upward.

6. **Keep every prevalence estimator; never retire one because a newer one flatters you.** BBSE and
   MLLS were the two closest to the truth and we dropped both for the estimator that agreed with our
   prior.

7. **Read the room.** Ranks 3, 5, 6 and 7 posted *byte-identical* F1 on both splits (public cell
   TP=173/PP=191, private TP=364/PP=397) with *different* AUC columns — four independent teams
   producing the same binary label set. A shared public approach existed and we never went looking
   for it. Zindi publishes winning solutions after code review (due **7 Sept 2026**); the Chat tab
   was live the whole time.

8. **Bundle the predictions for ≥90% of submissions, from submission #1.** We archived a `.npz` for
   only **13 of 91** entries (14%), and neither finalist nor the oracle `tcons_s13` is among them.
   That, not statistics, is what made the post-mortem above unanswerable at full strength: the
   contestable headroom inside the bundled set is 0.004993, against the 0.010132 that actually sat in
   our submission list. An analysis you cannot run on your own best entry is not an analysis. Design
   target **n ≥ 20 bundled candidates**, which buys 0.89 power against a true ρ of 0.6.

9. **Average every finalist over ≥5 seeds — it is the only lever here that needs no criterion.**
   Given that within-arm seed SD exceeds between-arm SD, seed-averaging is worth more than any model
   choice we debated, and it is provable in advance rather than selected after the fact. This does
   not conflict with recommendation 4: pool to *reduce variance in what you ship*, score the members
   to *judge whether the arm is alive*, and let `assert_pool_sane` catch the operating-point escape
   that made us distrust pooling in the first place.

---

## 4. What went right, and should be kept

- **The integrality sieve.** Recovering the exact confusion cell from a 9-decimal leaderboard
  readout, for zero submissions, is a genuinely reusable instrument. It solved the split exactly
  here and it caught our own 42-iteration error.
- **The discipline of writing the diagnosis down.** The iter41 pooling entry was precise enough
  that, five weeks later, private data confirmed it word for word. The failure was in the response,
  not the analysis — and that is only visible because the analysis was recorded.
- **Compliance.** Verified against the live rules page, held all the way, and confirmed correct.
  35% of the final standing is code review; nothing here jeopardises it.
- **Reproducibility.** Every submission is named, seeded, and traceable to a config.
