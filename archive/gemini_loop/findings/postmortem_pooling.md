# Post-mortem: is the pooler broken, and does fixing it recover the loss?

Started 2026-08-17, after the private LB was revealed. Competition CLOSED. Nothing here can
change our standing; the point is a provable, transferable lesson.

## The question

> When a seed-averaged pool scores WORSE than its own members, is the pooler broken? And does
> fixing the pooler recover the loss?

## Facts taken as given (supplied, verified upstream)

- Final: rank 120/500, private 0.910686008. Winner 0.956900206.
- Metric = 0.6*F1 + 0.4*ROC-AUC, verified on 182 triples (max residual 8e-10).
- F1 at a HARD 0.5 cut. Rules verbatim: "Setting a probability threshold is strictly forbidden.
  Your binary target should be based on the default threshold of 0.5."
- Test split SOLVED EXACTLY: public n=333 P=181 N=152; private n=697 P=379 N=318;
  overall 560/1030 = **0.5437** true test prevalence.
- Therefore any 9-decimal F1 inverts to a unique (TP, PP) cell per slice.

## The case (ARM T, iteration 41)

    submission_tcons_s42.csv              pub 0.914179  prv 0.913674   <- best public ever
    submission_tcons_s13.csv              pub 0.908873  prv 0.920818   <- best private ever (of 91)
    submission_champion_tcons_seedavg5    pub 0.893752  prv 0.901721
    member mean prv 0.917246 (AUC 0.939190, F1 0.902618)
    5-seed pool  prv 0.901721 (AUC 0.940650, F1 0.875769)
    pooling gain -0.015525 = +0.000584 AUC, -0.016109 F1

Pooled AUC is FINE (above the member mean). The entire loss is F1 -> the operating point moved.
The arm was written off as a "single-seed mirage" on the strength of the pooled number; both
seeds held up on private, so it was not a mirage.

## Plan (appended to as results land)

1. **Quantify the defect.** For every family on disk, member vs pool realized positive rate and
   the pooled-vs-member spread on OOF F1@0.5 and OOF AUC. Does the "pooled AUC normal, pooled F1
   below both members" signature generalize, or is it specific to the variance penalty?
2. **Test the fixes.** (a) probability averaging [incumbent], (b) rank-space averaging,
   (c) logit averaging, (d) pool-then-calibrate (one Platt refit on the pooled score).
   Compare realized test positive rate against the KNOWN 0.5437, and OOF F1@0.5.
3. **Validate against ground truth.** For pooled artifacts we actually submitted, the true
   public/private cells are known exactly. Does the offline prediction reproduce them? An
   offline method that cannot reproduce a known outcome is not evidence.
4. **Ship a guard.** Assertion in the pooling path: pooled_F1 < min(member_F1) while pooled_AUC
   is within the member range -> FAIL LOUDLY.
5. **State the counterfactual honestly.** Would the correct pooler have saved ARM T? If the
   evidence does not support a clean answer, say so.

## Discipline

- 0.5 stays a literal 0.5. No threshold tuning, even now.
- Any knob must be fixed by a **train-only** criterion. We may COMPARE against the true 0.5437
  as a diagnostic; we may never SELECT on it. This distinction is kept explicit throughout.
- No per-row test-label reconstruction from LB cells.
- The tcons bundles are NOT on disk (produced on Colab, never copied back). Everything below
  uses the families that ARE on disk. Where a claim would need tcons, it is marked as
  unresolvable from local data rather than invented.

---

## Result 1 — GROUND TRUTH: the exact operating point of every relevant submission

Inverted the 9-decimal F1 columns to (TP, PP) on both slices (public n=333/P=181,
private n=697/P=379). Enumerated **all** solutions rather than taking the first, so
non-uniqueness is visible rather than hidden.

| submission | pub (TP,PP) | prv (TP,PP) | pos-rate all 1030 |
|---|---|---|---|
| tcons_s13 | (162, 180) | (352, 395) | 0.5583 |
| tcons_s42 | (169, 194) | (352, 407) | 0.5835 |
| **champion_tcons_seedavg5** | *2 solutions* | **(356, 434)** | — |
| dpa_s21 | (164, 192) | (347, 411) | 0.5854 |
| **champion_dualpol_add_seedavg5** | (164, 191) | **(350, 414)** | 0.5874 |
| amix_s13 | (163, 185) | (346, 404) | 0.5718 |
| amix_s17 | (162, 189) | (350, 418) | 0.5893 |
| amix_s31 | (163, 192) | (350, 418) | 0.5922 |
| amix_s37 | (163, 194) | (346, 418) | 0.5942 |
| amix_s42 | (165, 198) | (353, 423) | 0.6029 |
| **_control_amix (the 10-member pool)** | (164, 191) | **(349, 415)** | 0.5883 |
| champion_alphamix10_regimematch | (163, 188) | (349, 414) | 0.5845 |
| teacher_perm_s42 | (158, 177) | (344, 390) | 0.5505 |
| teacher_perm_s13 | *5 solutions* | (341, 406) | — |
| distill_s21 | (164, 192) | (348, 411) | 0.5854 |

### CORRECTION 1 (to the brief): F1 inversion is NOT always unique

The brief states "every inversion we have tried is unique". On the public slice
`champion_tcons_seedavg5` admits **2** solutions and `teacher_perm_s13` admits **5**.
Uniqueness is a property of the slice size and the particular F1 value, not of the method.
It holds everywhere on the private slice (n=697, P=379) in this sample, and almost everywhere
on public (n=333, P=181) — but not everywhere.

**This is a live bug in `tools/post_mortem.py::invert_f1`.** Its last line is

    return hits[0] if len(hits) == 1 else (hits[len(hits) // 2] if hits else None)

so when the inversion is ambiguous it silently returns the *median* candidate and the caller
cannot tell. Every downstream claim of the form "inverts exactly to TP=x, PP=y" is only sound
where `len(hits) == 1`, which the function does not report. Fixed below (Result 5).

### CORRECTION 2 (to the brief, and to `tools/repool.py`): the pool over-predicts, it does not under-predict

`tools/repool.py`'s docstring states the mechanism as: "an arithmetic mean of
independently-noisy probabilities is shrunk toward the members' centre of mass, so the pooled
distribution is narrower than any member's and a FIXED 0.5 cut **catches fewer rows**."

Ground truth says the opposite. On private, the tcons pool predicted **PP = 434** positives
against member PP of **395** and **407** — the pool caught **27 to 39 MORE** rows than either
member, not fewer.

The compression premise is right; the sign was asserted without checking where the mass sits.
Our test-score mass sits **above** 0.5 (every artifact realizes pos-rate 0.55–0.60). Shrinking
a distribution toward a centre of mass that is above 0.5 pulls sub-threshold rows **up** across
the cut. So the correct statement is:

> Arithmetic averaging compresses the pooled distribution toward the members' centre of mass.
> The realized positive rate therefore moves **toward the pooled score's own median**, i.e.
> toward pos-rate 0.5 from below and **away from 0.5 in whichever direction the mass already
> lies**. With mass above 0.5, the pool **over-predicts**.

That matters practically: at a true prevalence of 0.5437 the pool's extra 27–39 predictions are
almost all false positives, which is exactly a precision (hence F1) loss with the ranking intact.

### The tcons cell, decomposed (private, P=379, N=318)

| | TP | PP | precision | recall | F1 |
|---|---|---|---|---|---|
| tcons_s13 | 352 | 395 | 0.8911 | 0.9288 | 0.909561 |
| tcons_s42 | 352 | 407 | 0.8649 | 0.9288 | 0.895674 |
| **pool** | **356** | **434** | **0.8203** | **0.9393** | **0.875769** |

The pool did **not** rank worse and did not lose true positives — it gained 4 TP and 1 point of
recall. It bought them with **27–39 extra false positives**. The whole −0.0161 F1 is a
precision loss caused by the operating point sliding right. This confirms the brief's headline
diagnosis and sharpens it: the failure is over-prediction, not blunted discrimination.

### CORRECTION 3 (to the brief): "below BOTH members" covers 2 of 5 members

ARM T pooled **five** seeds; only **two** (s42, s13) were ever submitted individually, so only
two member scores exist. "member mean prv 0.917246" is the mean of those two, not of the pool's
membership. The three unobserved seeds have no score. This does not rescue the pool — the
observed excursion (+27 PP beyond the *maximum* observed member) is far too large to be a
membership-composition effect, and F1 0.8758 sits 0.020 below the lower observed member — but
the claim as literally stated ("below its own members") is verified against 40% of the members.
Stated so the next reader does not over-read it.


---

## Result 2 — PLAN STEP 1: the defect does NOT generalize. This is a null, reported as a null.

Every bundle family on disk, each member Platt-calibrated on its own OOF (the shipping path),
then pooled with the incumbent combiner A. n_oof = 1817, train prior 0.4023; n_test = 1030.

| family | n | member OOF AUC min..max | member OOF F1@0.5 min..max | member test pos-rate min..max | pool OOF AUC | pool OOF F1@0.5 | pool test pos-rate |
|---|---|---|---|---|---|---|---|
| dpa           | 5  | 0.988234..0.990723 | 0.962199..0.969072 | 0.5699..0.5942 | 0.992048 | 0.969780 | 0.5874 (PP 605) |
| amix          | 10 | 0.986563..0.991174 | 0.962912..0.969780 | 0.5718..0.6029 | 0.992356 | 0.973829 | 0.5883 (PP 606) |
| teacher_perm  | 5  | 0.986097..0.989846 | 0.961591..0.964948 | 0.5505..0.5981 | 0.991655 | 0.969739 | 0.5816 (PP 599) |
| jtt (hetero)  | 3  | 0.986978..0.993088 | 0.940206..0.968622 | 0.5796..0.5971 | 0.992834 | 0.960825 | 0.5874 (PP 605) |
| presto (hetero)| 2 | 0.990929..0.991676 | 0.952315..0.959338 | 0.5670..0.5699 | 0.991973 | 0.957844 | 0.5718 (PP 589) |

**The defect signature — pooled F1 below every member while pooled AUC sits inside the member
range — occurs in ZERO of the five families.** In the three seed families the pool's OOF F1 is
*above every member* and its OOF AUC is *above every member*. In the two heterogeneous families
the pooled F1 lands inside the member range. Nothing on disk reproduces ARM T's signature.

Two further measurements that bear on the mechanism:

- **The pool does not compress the realized positive rate at all.** Member test pos-rates span
  0.5699–0.5942 (dpa) and 0.5718–0.6029 (amix); the pool lands at 0.5874 and 0.5883 — i.e.
  *inside* the member spread, near its mean, not outside it. By contrast the tcons pool went to
  PP 434 on private against observed member PP of 395 and 407 — an excursion *beyond the member
  maximum*. **No family on disk does that.** So Correction 2's mechanism
  (compression pushes mass across a 0.5 cut that sits below the centre of mass) is a real
  mechanism but it is *not* what the incumbent combiner does on ordinary seed families — there
  the compression is second-order and the pooled rate simply tracks the member mean.
- **OOF is saturated and therefore cannot see any of this.** OOF AUC ≈ 0.99 and OOF F1@0.5 ≈ 0.96
  everywhere, against realized test AUC ≈ 0.95 and test F1 ≈ 0.88. Worse, the calibrated OOF
  positive rate is pinned to the train prior (members 0.398–0.403, pools 0.397–0.399, train prior
  0.4023) while the *test* positive rate is 0.55–0.60. **OOF F1@0.5 is structurally blind to the
  operating-point failure we are hunting**, because on OOF the operating point is by construction
  correct. This is a hard limit on plan step 2's "selection-legal" column and is stated up front.

### CORRECTION 4 (to `tools/repool.py`'s stated purpose)

repool.py's docstring says the defect "needs no slope heterogeneity — only independent member
noise, which is what multi-seed pooling IS" and concludes "if it reproduces on real bundles it is
a live defect on the CHAMPION pool too." **It does not reproduce on real bundles.** Five families,
25 member bundles, three of them exactly the multi-seed case the claim names — the champion dpa
pool included — and the pooled F1 is above every member in all three. The prediction was
falsifiable and it is falsified. The champion pool was not defective.

---

## Result 3 — PLAN STEP 2: four pooling orders. Three of the four are the same artifact.

Definitions used (all label-free on test; the cut stays a literal 0.5; every Platt map is fit on
training OOF and training labels only):

- **(a) prob-avg [INCUMBENT]** — per-member Platt on its own OOF, then arithmetic mean of
  probabilities. `src.calibration.calibrated_pool`.
- **(b) rank-avg (joint)** — rank-transform each member's OOF∪test *jointly* (one common scale,
  no labels used), average the ranks, then ONE Platt on the pooled OOF ranks.
- **(b′) rank-avg (separate)** — the historic form: rank OOF and test *separately* per member,
  average, one Platt on the pooled OOF ranks. Included to test `calibrated_pool`'s docstring
  claim rather than assume it.
- **(c) logit-avg of calibrated probs** — per-member Platt first (as in a), then average in
  log-odds space. Geometric mean of odds; isolates arithmetic-vs-geometric holding the
  calibration order fixed.
- **(d) pool-then-calibrate** — average the RAW member logits, then ONE Platt on the pooled OOF.
  This is `tools/repool.py`'s combiner B.

Realized test positive count PP (of 1030), and OOF F1@0.5 (the only selection-legal column):

| family | member PP range | (a) | (b) joint rank | (b′) sep rank | (c) logit-avg | (d) pool-then-cal |
|---|---|---|---|---|---|---|
| dpa (5)          | 587–612 | **605** | 571 | 398 | **605** | **605** |
| amix (10)        | 589–621 | **606** | 573 | 400 | **606** | **606** |
| teacher_perm (5) | 567–616 | **599** | 545 | 416 | **599** | 601 |
| jtt (3)          | 597–615 | **605** | 599 | 432 | **605** | 606 |
| presto (2)       | 584–587 | **589** | 577 | 419 | **589** | 591 |

OOF F1@0.5 (dpa / amix / teacher_perm): (a) 0.969780 / 0.973829 / 0.969739;
(b) 0.970994 / 0.975104 / 0.969571; (c) 0.970487 / 0.974500 / 0.969739;
(d) 0.970487 / 0.973829 / 0.969739. Spread across orders ≤ 0.0013 — **inside nothing; it is
smaller than the 4th decimal of anything that matters, and OOF's positive rate is pinned to the
train prior anyway, so OOF F1 cannot select among these orders.** That is a null, not a tie
broken in favour of the incumbent.

### The finding: (a), (c) and (d) are operationally the same combiner here

On the two clean seed families the three land on **bit-identical positive counts** (605/605/605
and 606/606/606). Arithmetic mean of probabilities, geometric mean of odds, and pool-then-
calibrate disagree by at most 2 rows out of 1030 anywhere in the table. The literature that
motivated `repool.py` (Ranjan & Gneiting 2010; Rahaman & Thiery 2021; Wu & Gales 2021) is correct
that a linear opinion pool is uncalibrated — but the *magnitude* of that miscalibration, on
members this tightly correlated, is below the resolution of a 1030-row decision. **Swapping the
combiner would have changed our submitted binary column by 0 to 2 rows. It was never worth a
submission slot.**

### CORRECTION 5: `calibrated_pool`'s docstring warning about rank-averaging is CORRECT, and is the only real effect in the table

Its docstring predicts that ranking OOF and test separately "drives the positive rate back to the
TRAIN prior (~0.402 observed) instead of the model's honest test estimate (~0.55)". Measured:
(b′) yields pos-rates 0.3864 / 0.3883 / 0.4039 / 0.4194 / 0.4068 against a train prior of 0.4023.
The prediction is quantitatively right in all five families. Ranking OOF and test separately is a
genuine, large (≈200-row) defect — and it is the one we correctly avoided.

Joint ranking (b) is different: it keeps the train-vs-test level and moves the pooled rate *down*
by 30–34 rows, landing at 0.5544 / 0.5563 / 0.5291 — closest to the true 0.5437 in every family.
**That comparison against 0.5437 is a DIAGNOSTIC and is recorded as one.** No train-only criterion
in this project distinguishes (b) from (a): their OOF F1 differ by 0.0012 in dpa in (b)'s favour
and by 0.0002 in teacher_perm in (a)'s favour. We may not select (b) on the strength of the
0.5437 comparison, and we do not.

---

## Result 4 — `tools/post_mortem.py::invert_f1` is fixed, and the fix changed nothing but honesty

The bug from Correction 1 is repaired:

- `invert_f1_all(f1, n, P) -> list[(TP, PP)]` returns the **complete** solution set. Its brackets
  are now computed in exact rational arithmetic (`2*TP/hi <= PP+P <= 2*TP/lo`) instead of a fixed
  ±2 window around a float estimate — the old window could in principle miss the very edge case
  the function exists to expose.
- `invert_f1(f1, n, P) -> (cell, n_solutions)`. **The signature changed on purpose**: it is now
  impossible to consume the result without seeing how many solutions there were.
- Every caller in `leaderboard()` updated. Two now `assert n_solutions == 1` on *our own* cell
  (the operating-point comparison in section 7 would otherwise be built on a guess); the display
  loops print `[!] NON-UNIQUE (k cells)` and the AUC-band loop prints an inversion audit line.
- `tools/offline_criterion_bakeoff.py` kept a private duplicate enumerator *because of* this bug.
  It now delegates to `post_mortem.invert_f1_all`, so the two reports cannot drift apart.

**Re-ran `python tools/post_mortem.py` and diffed against the committed version. Two lines
differ, both additions; no number and no conclusion changed:**

    +   [inversion audit] 0 of 8 rows in this band have a NON-UNIQUE (TP,PP); the rest are exact.
    -     #  8 Guelmbaye   posrate 0.6212  prec 0.8730  rec 0.9974
    +     #  8 Guelmbaye   posrate 0.6212  prec 0.8730  rec 0.9974  [!] NON-UNIQUE (2 cells)

Rank 8's private F1 0.931034482 admits **(351, 375)** and **(378, 433)** — posrate 0.5380 *or*
0.6212, and the old code printed the second as fact. The section-7 conclusion ("the top of the
board sits near the true prevalence") is unaffected and if anything strengthened, since the
alternative cell is 0.5380 against a true 0.5438. Auditing all 75 leaderboard rows: **2 of 75
private F1 values are non-unique** (rank 8 with 2 cells, rank 65 with 5); neither is in the
AUC-matched band that carries the report's argument. Section 7's "us" cell (TP 348, PP 408) is
unique and now asserted to be.

---

## Result 5 — PLAN STEP 3: the offline pipeline DOES reproduce the known cells. 13 of 13.

No test labels exist locally, so the only outcome an offline method can be held to is the
**realized predicted-positive count**, which the LB inversion fixes exactly as PP_pub + PP_prv.
That is a genuine falsification test: 1030 rows, one integer, no fitting.

| artifact | offline PP | LB-inverted PP (pub + prv) | verdict |
|---|---|---|---|
| amix_s13 | 589 | 185 + 404 = 589 | MATCH |
| amix_s17 | 607 | 189 + 418 = 607 | MATCH |
| amix_s31 | 610 | 192 + 418 = 610 | MATCH |
| amix_s37 | 612 | 194 + 418 = 612 | MATCH |
| amix_s42 | 621 | 198 + 423 = 621 | MATCH |
| dpa_s21 | 603 | 192 + 411 = 603 | MATCH |
| jtt_balance_s42 | 597 | 188 + 409 = 597 | MATCH |
| jtt_control_s42 | 615 | 195 + 420 = 615 | MATCH |
| jtt_lam5_s42 | 604 | 190 + 414 = 604 | MATCH |
| teacher_perm_s42 | 567 | 177 + 390 = 567 | MATCH |
| teacher_perm_s13 | 593 | *5 pub cells* + 406 | MATCH (and RESOLVES the ambiguity) |
| smoke_test | 408 | *2 pub cells* + 280 | MATCH (and RESOLVES the ambiguity) |
| **champion_dualpol_add_seedavg5 (re-pooled from its 5 members)** | **605** | **191 + 414 = 605** | **MATCH** |

**All 12 name-joined singles and the champion pool reproduce exactly: 13 of 13.** The offline pipeline
is therefore admissible evidence: it predicts a known outcome it was not fitted to.

### Two ambiguities from Correction 1 are now CLOSED by the offline reproduction

- `teacher_perm_s13` public: 5 candidate cells; the offline total 593 minus the unique private
  PP 406 forces pub PP = 187, which is in the candidate set and unique there. **Public cell is
  (TP 160, PP 187).**
- `smoke_test` public: 2 candidates; forced to **(TP 105, PP 128)**.

### The third, `champion_tcons_seedavg5`, is resolved by inference — labelled as inference

Its public F1 admits **(153, 170)** and **(170, 209)**; private is uniquely (356, 434), posrate
0.6227. Across the **68** ledger artifacts whose *both* cells are unique, |posrate_pub −
posrate_prv| has mean 0.0179, sd 0.0112, **max 0.0478**. Candidate A implies a gap of 0.1122 —
2.3× the largest gap ever observed across 68 artifacts on the same random split. Candidate B
implies 0.0049, dead typical. **Inferred (not measured): the public cell is (TP 170, PP 209), so
the tcons pool's total is PP = 643 of 1030, pos-rate 0.6243**, against tcons_s42 at 601 (0.5835)
and tcons_s13 at 575 (0.5583). Correction 2's over-prediction finding is *larger* than first
stated: the pool over-predicted by **42 to 68 rows** across the full test set, not 27–39.

### Bundle identification, by content

- `preds_champion_distill_alphamix10.npz` re-pools to PP = 606, and among all 87 distinct
  submitted filenames **exactly one** has a consistent total: `submission__control_amix.csv`
  (191 + 415 = 606). Identification is unique, so the "10-member amix pool" bundle is that
  submission. (`champion_alphamix10_regimematch` is 188 + 414 = 602 — the regime-match step moved
  4 rows; a *different* artifact.)
- `preds_teacher_perm5.npz` re-pools to PP = 599. The only ledger row with a consistent total is
  `c_moments_s7`, an unrelated arm from three weeks earlier. **The teacher_perm 5-seed pool was
  never submitted; its leaderboard outcome is unresolvable and is not used anywhere below.**

### A small real hazard found in passing

Feeding a *stored pool bundle* back through `calibrate_legal` (i.e. Platt-calibrating an
already-Platt-calibrated pooled score) gives PP = 606 where the artifact we actually shipped had
605. One row, but it is a silent double-calibration: `seed_average.py` writes the pooled
calibrated probability into `p_test_raw`, a field whose name promises a *raw* score. Any tool
that treats a pool bundle like a member bundle re-calibrates it. Noted; not the subject here.

### Falsification test on the true-positive counts (a second, independent check)

Ground truth fixes TP_pool − TP_member exactly. Offline fixes a = |pool \ member| and
b = |member \ pool|. Every gained TP must come from a and every lost TP from b, so consistency
*requires* −b ≤ TP_pool − TP_member ≤ a. This can fail, and would falsify the join.

| pool | member | TP_pool − TP_mem (LB) | offline a, b | admissible | |
|---|---|---|---|---|---|
| dualpol_add_seedavg5 | dpa_s21 | +3 | 9, 7 | [−7, +9] | CONSISTENT |
| _control_amix | amix_s13 | +4 | 19, 2 | [−2, +19] | CONSISTENT |
| _control_amix | amix_s17 | +1 | 5, 6 | [−6, +5] | CONSISTENT |
| _control_amix | amix_s31 | +0 | 5, 9 | [−9, +5] | CONSISTENT |
| _control_amix | amix_s37 | +4 | 5, 11 | [−11, +5] | CONSISTENT |
| _control_amix | amix_s42 | −5 | 1, 16 | [−16, +1] | CONSISTENT |

Six for six, and two of them are tight (amix_s37 needs +4 of an available +5; amix_s42 needs −5
of an available −16 but only +1 the other way). The offline reconstruction is not merely
count-matching; its per-row decisions are compatible with the hidden labels.

---

## Result 6 — PLAN STEP 4: the guard is shipped, in the real code path

`src/calibration.py` now defines `PoolingDefect` and `assert_pool_sane()`, and
**`calibrated_pool()` runs it on every call** (`guard="raise"` by default, `"warn"`, or
`"off"`). That is the pooling path used by `tools/seed_average.py`, `tools/arch_blend.py` and
`tools/regime_match.py` — i.e. by both designated finalists. `tools/repool.py` is the one caller
passing `guard="warn"`, because a diagnostic built to *inspect* a suspect pool must not abort on
finding one; that exception is annotated at the call site.

Two checks, both label-free on test and train-only on OOF:

1. **RANK-OK-BUT-DECISION-BAD** (the signature the brief asked for): pooled OOF F1@0.5 below
   *every* member while pooled OOF AUC sits *inside* the member range.
2. **OPERATING-POINT ESCAPE**: the pooled realized test positive rate must lie inside the
   members' own realized range, widened by `max(sd of member rates, 0.005)`.

**Check 1 is honest but weak, and the code says so.** Result 2 measured that the calibrated OOF
positive rate is pinned to the train prior (0.397–0.403 vs prior 0.4023) while the test rate is
0.55–0.60. On OOF the operating point is correct by construction, so this check cannot see the
failure mode in general. It never fired on any healthy family, and there is **no local evidence
that it would have fired on tcons** — those bundles are gone. Recorded as a cheap
true-positive-when-it-fires, not as the load-bearing check.

**Check 2 is the one that catches ARM T.** Verified in both directions:

| family | pooled rate | member range | tol | verdict |
|---|---|---|---|---|
| dpa (5) | 0.5874 | 0.5699–0.5942 | 0.0095 | PASS |
| amix (10) | 0.5883 | 0.5718–0.6029 | 0.0087 | PASS |
| teacher_perm (5) | 0.5816 | 0.5505–0.5981 | 0.0171 | PASS |
| jtt (3) | 0.5874 | 0.5796–0.5971 | 0.0088 | PASS |
| presto (2) | 0.5718 | 0.5670–0.5699 | 0.0050 | PASS (2 rows over the raw max, inside tol) |
| **ARM T replay** | **0.6243 (643/1030)** | 0.5583–0.5835 | 0.0178 | **TRIPPED, +24 rows past the widened envelope** |

The ARM T replay uses *only* the LB-inverted predicted-positive counts (575, 601, 643) — the one
tcons fact we still have — with a deliberately healthy synthetic OOF, so it proves check 2 fires
unaided.

**Stated plainly, because it matters: the tolerance in check 2 was chosen with the ARM T case in
view.** That is fitting a threshold to a single positive example. What defends it is the margin,
not the fit — the null families overshoot by 0, 0, 0, 0 and 2 rows; ARM T overshoots by 24. There
is an order of magnitude between them, so the verdict does not depend on where in that gap the
line sits. The same caveat is written into the function's docstring.

The error message names the remedies that keep the cut literal (submit members individually; pool
in a space that does not move the level) and — on the strength of Result 3 — explicitly tells the
reader that **swapping the combiner is not one of them**.

---

## Result 7 — PLAN STEP 5: would the correct pooler have saved ARM T? NO CLEAN ANSWER — and the question is the wrong one.

### 7a. The evidence does not determine the counterfactual score

The tcons bundles are not on disk, so no offline pooler can be run on them. The furthest the LB
cells allow us to go is a bound. Private slice, P=379, N=318; pooled AUC 0.940650 held fixed (all
four orders are order-preserving up to a couple of rows, so the AUC term barely moves).

Actual pool: PP 434, TP 356, F1 0.875769, composite 0.901721.

| counterfactual operating point | admissible TP | F1 range | composite range |
|---|---|---|---|
| pool moved to s13's PP = 395 | 317–356 | 0.8191–0.9199 | 0.8677–0.9282 |
| pool moved to s42's PP = 407 | 329–356 | 0.8372–0.9059 | 0.8786–0.9198 |
| pool moved to ~PP 412 (what joint rank-averaging would plausibly have done) | 334–356 | 0.8445–0.9001 | 0.8830–0.9163 |

(The ROC-AUC discordant-pair budget, (1 − AUC)·P·N = 7153, gives TP >= 303 at PP 395 — weaker
than the >= 317 that follows from the pool's own cell, so it adds nothing.)

We actually scored **0.910686**. Every one of those bands straddles it. **The counterfactual is
indeterminate: a rate-corrected tcons pool could have finished anywhere from far below our result
to inside the top 40.** Anyone reporting a single number here is inventing it.

The one *estimate* available, labelled as an estimate and not a measurement: the pool's marginal
precision over the rows between s42's cell and its own (4 extra TP for 27 extra predictions =
0.148), extrapolated backwards, gives TP ~ 350 at PP 395, F1 ~ 0.9050, composite **~ 0.9192** —
above what we scored, below tcons_s13's 0.920818. This assumes the positive sets nest and the
marginal precision is locally constant. Neither is guaranteed.

### 7b. What Result 3 does settle: the fix would not have been the combiner

Measured on five families: probability-averaging, geometric-mean-of-odds and pool-then-calibrate
agree to within 2 rows of 1030; on dpa, `repool.py` reports **0 of 1030 rows flipping** between
combiners A and B. Joint rank-averaging is the only order that moves the level, and it moves it
by 33 rows (median across families) — applied to tcons that is 643 -> ~610, still above the
members' 575 and 601. **So the pool-then-calibrate fix this whole investigation was launched to
install would have changed ARM T by approximately nothing, and rank-averaging would only have
halved the excursion.** The literature was right about calibration and irrelevant about magnitude.

### 7c. The question is the wrong one, and this is the actual finding

ARM T did not need saving by a pooler. **Both of its members were already on the leaderboard.**

    submission_tcons_s42.csv   public  0.914178889  <- the HIGHEST public score of all 91
    submission_tcons_s13.csv   private 0.920818161  <- the HIGHEST private score of all 91
    what we designated and scored      0.910686008

`tcons_s42` was our best public submission *ever made*, and we did not designate it a finalist.
We designated `champion_archblend4` (public 0.899643) and `champion_dualpolmix10_regimematch`
(public 0.910447). The plain Zindi default — if you designate nothing, your best public entry is
scored — would have returned **0.913674, +0.002988 on us**. "Top-5 by public, keep the best
private" returns 0.919608, **+0.008922**. The oracle over our own 91 is 0.920818, +0.010132, and
it is also a tcons member.

So the arm was not lost to a combiner. It was lost to a **selection rule that judged an arm by its
pooled artifact and then discarded the arm's members along with it.** The pooled number was a
property of the pooling step; we treated it as a property of the method. Fixing the pooler
recovers, at most, part of a counterfactual we cannot pin down. Not throwing away two
already-scored, already-leading submissions recovers +0.003 to +0.010 with certainty.

---

## THE TRANSFERABLE RULE

Short enough to paste into the next competition's checklist.

1. **A pool is a new artifact, not a verdict on its members.** Never retire an arm on the strength
   of its pooled score. Carry the members and the pool as separate candidates, always.
2. **When the metric has a hard threshold, guard the LEVEL, not just the ORDER.** Before shipping
   any ensemble: the pooled realized positive rate must sit inside the members' own realized
   range, widened by `max(sd of member rates, 0.005)`. Needs no labels and no prevalence estimate.
   If it escapes, the members disagree about the level and you must decide *on the record* whether
   the pool or the members are right. (`src/calibration.py::assert_pool_sane`.)
3. **Do not assume the combiner is the bug.** Measured here on 25 bundles in 5 families:
   probability-averaging, geometric-mean-of-odds and pool-then-calibrate differ by <= 2 rows in
   1030. The published result that linear opinion pools are uncalibrated is true and, at this
   member correlation, has no operational consequence. Measure the magnitude before you refactor.
4. **Rank-transform OOF and test TOGETHER or not at all.** Ranking them separately drove the
   positive rate to the train prior (0.39–0.42 against a true 0.5437) in all five families — a
   ~200-row error, the largest single defect measured in this entire study.
5. **A validation set is not a threshold set.** Our OOF F1@0.5 was 0.96 while our test F1 was
   0.88, because the calibrated OOF positive rate is pinned to the train prior by construction.
   Any criterion computed at a fixed cut on OOF is blind to operating-point failure under label
   shift. Diagnose the level on the *test score distribution*, which is label-free and legal.
6. **An inverse function must report its own ambiguity.** `invert_f1` silently returned the median
   candidate. Audited: of our 87 distinct submitted filenames, 11 have a non-unique public
   cell and 9 a non-unique private cell; 2 of the top-75 leaderboard rows are non-unique too.
   Reconstruction routines return the solution SET or an explicit count, never a silent pick.
7. **Designate finalists by a rule fixed in advance, and let your best public entry be one of
   them.** Beating the Zindi default (best public) requires evidence we did not have, and we lost
   0.003 to it. The 91-row ledger says: top-5 by public, keep the max private, gains +0.0089.

---

## Closing summary

- **Generalized:** nothing. The pooled-F1-below-members signature appears in 0 of 5 families.
- **Falsified:** the combiner hypothesis (Correction 4), and its stated sign (Correction 2).
- **Confirmed:** `calibrated_pool`'s own warning about separate rank transforms (Correction 5).
- **Fixed:** `invert_f1`'s silent ambiguity (Result 4); `repool.py`'s backwards docstring.
- **Shipped:** `assert_pool_sane()`, on by default in the real pooling path (Result 6).
- **Verdict on ARM T:** the counterfactual is indeterminate and is reported as indeterminate. The
  recoverable loss was in the finalist selection rule, not in the pooler (Result 7c).
