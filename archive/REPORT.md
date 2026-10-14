# Aquaculture Pond Identification — Solution Report

**Zindi / FAO / ITU GeoAI Challenge** · Osborn Nyakaru · revised 2026-08-10 (through iteration 41)

---

## 0. Summary

We classify isolated 10 m ground cells as *managed aquaculture pond* from a 12-month × 12-band
Sentinel-1/Sentinel-2 time series, under a **deliberately constructed temporal covariate shift**:
training rows are fully observed, test rows expose only a consecutive 4–6 month window from a
different time period. (The organizers state the *temporal* shift explicitly. Earlier drafts of this
report also asserted a *regional* shift; we could not source that statement and have withdrawn it —
the shift we can evidence is temporal plus the observation-window difference.)

The submitted model is a **from-scratch temporal Transformer** — 2 layers, 4 heads, 64-dim, trained
on masking-augmented views of each training row, with a cross-view invariance penalty, one
physically-motivated permanence channel, and a literal 0.5 operating point.

**Reproduce it in one command:**

```bash
pip install -r requirements.txt          # pinned, open-source only, no AutoML
# place Train.csv / Test.csv / SampleSubmission.csv in data/raw/
bash experiments/reproduce_champion.sh   # prints fingerprints to verify the reproduction
```

**But the model is not the interesting part of this submission, and we will not pretend otherwise.**
The most valuable thing we produced is a *measurement discipline*: we measured the noise floor of our
own evaluation channel, found it large enough to void most of our recorded results, rebuilt the
decision process around it, and then documented — three times, including once against our own
written record — the precise boundary at which our replacement instruments fail. Sections 4–6 are
that work. It is the part we would want a reviewer to read.

Our gap to the top of the leaderboard is **≈0.030**, and §6.5 localizes it: ~88% sits in the F1 term,
but ~90% of *that* is the ranking of rows near the 0.5 cut rather than calibration. The whole gap is
about **52 rows out of 309**.

**Headline numbers, stated honestly:**

| quantity | value |
|---|---|
| **finalist #1 (designated)** | **0.907368983** — `champion_dualpolmix10_regimematch`, transductive self-distillation + the dual-pol gate, **10 distinct seeds**, α marginalized over {0.7, 1.5}, R=1 regime-matched calibration. **AUC 0.945841814 — above the leader's 0.944897** |
| **finalist #2 (designated)** | **0.899643** — `champion_archblend4`, a 4-architecture pool (decorrelated hedge) |
| best *pooled* composite ever recorded | 0.910837 — `champion_distill_a15_seedavg5`, **not designated**; a 5-seed pool at one α. iter42 proved that class of edge carries **zero ranking information** — α=0.7 at 5 and 10 seeds returned *bit-identical* AUC (0.944024425) and the whole composite gap was **one row crossing the cut** — while a 10-distinct-seed pool reduces variance across all 721 private rows |
| runner-up not designated | 0.910446704 — `champion_dualpol_add_regimematch`, beats finalist #1 on **both** public columns (+0.003078 composite, +0.000545 AUC) but on **5** seeds. Declined for the reason in the row above; the AUC difference is ≈12 concordant pairs |
| best **AUC** ever recorded | 0.946460 — `champion_dualpol_add_seedavg5` (5 seeds, not designated). Finalist #1 is second at **0.945842**, and is the first artifact to out-rank the leader on **10 distinct seeds** |
| previous ceiling, held by 4 separate constructions | ~0.8995 — broken at iteration 41, see §4(iii) and §9 |
| best *single* public score ever recorded | 0.914179 — **not designated; it is seed luck, see §4** |
| **measured seed-to-seed sd** | **0.0191** — larger than most effects in our own ledger |
| public-slice binomial noise (n=309) | ≈ ±0.012 on the composite |
| LB-gated iterations / submissions | 45 iterations, ~61 of 100 submissions |

**Where the remaining gap is, stated precisely.** The F1 column is the small-denominator rational
`2·TP/(PP+P)` and inverts exactly (matching to 10 decimals). At an AUC we have now *matched*, the
leader converts **≈173 true positives** to our **≈164** — so the entire residual ~0.022 is **~9 true
positives at the decision boundary**, not ranking. Closing it by moving the cut toward a
leaderboard-inferred positive count is exactly the move the rules forbid, and we did not make it.
See §8.2.

The gap between rows 2 and 3 of that table is the single most important fact in this report, and
§4 is about why we deliberately submit the lower number.

---

## 1. The problem, and the one design decision that defines it

Per cell: 12 monthly composites × 12 bands — Sentinel-1 SAR (**VH, VV**, dB) and 10 Sentinel-2
optical bands. **No latitude/longitude, no spatial neighbourhood, no image patch, no static
covariates.** Each row is one isolated pixel's time series.

Train **1,821** rows (1,817 after dropping 4 exact duplicates), test **1,030** (public 309, private
721 — the organizers state a 30/70 split). Metric `0.6·F1 + 0.4·ROC-AUC`, where F1 is computed on a
binary column at a **hard 0.5 cut** and AUC on a probability column.

**The designed trap.** Train rows are fully observed across 12 months. Test rows expose only a
consecutive **4/5/6-month window** — measured on `Test.csv` as 345/343/342 rows, and **1030/1030
rows are contiguous** — with everything else set to the `-9999` sentinel, plus additional
**Sentinel-2-only cloud dropout** inside the window at measured per-month rates (0.003–0.28). A
model trained on 12-month statistics leans on signal that does not exist at test time.

**The masking trap, stated precisely.** Only statistics that are *unbiased at every window length*
transfer: means, medians, interior quantiles, and fractions. Minima, maxima, ranges, counts and
run-lengths are all biased by the number of observed months, so a model that uses them is reading
window length rather than land cover. This single observation determines the entire feature bank.

**The shift is real and irreducible, and we proved it several independent ways:**

| probe | adversarial train-vs-test AUC |
|---|---|
| unmasked 12-month train vs test (upper anchor) | **1.0000** |
| hand-engineered features on raw windows | ≈0.99 |
| frozen **Presto** SSL embeddings (never saw our labels) | 0.965–0.976 |
| **our actual input** — masked, left-aligned values | **0.8915** |
| our missing-indicator channels alone | **0.4758** (below chance) |

The second-to-last row is the one that matters: after masking augmentation and relative-time
reframing, a substantial gap remains. This is genuine covariate shift by design, **not a pipeline
leak** — the last row shows our masking already matches test dropout by construction. "Drive
adversarial AUC to 0.5" is therefore not an achievable goal, and feature-pruning toward it is
futile. §6 shows why, and corrects an earlier explanation of our own.

---

## 2. The submitted model

Per observed month the encoder sees **25 channels** — 12 standardized band values ⊕ 12 binary
missing-indicators ⊕ **1 permanence indicator** `1[VH_dB(t) < −21]` — then:

```
Linear(25 → 64)
  → learned positional embedding (length 12), applied AFTER left-alignment
  → 2-layer Transformer encoder (4 heads, GELU, dropout 0.2,
       src_key_padding_mask over fully-missing months)
  → masked mean-pool over observed months
  → MLP head → sigmoid
```

**Loss:** `L = BCE + λ·Var_k(logit)` across `K=2` masked views of the same row, λ=1.0 — a cross-view
invariance penalty teaching the model that the label does not depend on *which* window was exposed.

**Training:** AdamW lr 1e-3, wd 1e-4, batch 256, 60 epochs, 5-fold CV, owner-grouped batching. Test
prediction is the mean of the 5 fold-models; the submitted artifact pools **5 independent seeds** on
top of that, i.e. 25 models.

**The designated finalist adds one more term to this loss** — a soft self-distillation term over the
unlabeled test rows, which is what carried the score past the long-standing ~0.8995 ceiling. It adds
no parameters and changes nothing above. See §6.6.

**Why the permanence channel.** The masked mean-pool of a binary indicator *is* the fraction of
observed months below the threshold — an n-invariant statistic, and the empirical CDF of VH at one
physically-privileged cut. Ottinger et al. place the SAR land/water split near −21.5 dB. This is our
largest feature effect (+0.012 single-seed) and, notably, **threshold selection was monotone and
decreasing in the number of thresholds**: 1 τ (0.9065) > 4 τ (0.9016) > 6 τ (0.8987). The signal is
one physical cut, not a profile.

**Design choices and their honest status.** Earlier revisions of this repo described these as
"validated on the leaderboard." **That claim does not survive our own seed measurement (§4)** and we
have corrected it:

| choice | recorded Δ LB | status against seed sd = 0.0191 |
|---|---|---|
| **masking augmentation** (train on test-like windows) | — | ✅ structural; the pipeline is built on it |
| **relative-time reframing** (left-align to `t_rel=0`) | +0.0128 | ⚠️ **inside the noise floor — UNRESOLVED** |
| **cross-view invariance** (λ=1.0) | +0.0047 | ⚠️ **inside the noise floor — UNRESOLVED** |
| **VH permanence channel** | +0.0120 | ⚠️ single-seed; **washed to +0.000 on 5-seed averaging** |
| legal 0.5 operating point (replacing a rules-violating pin) | −0.0058 | ✅ compliance fix, see §8 |

Only **two** effects across 41 iterations ever clearly exceeded the noise floor, and both were
model-class changes: GBDT → Transformer (**+0.052**) and a broken amplitude transform (**−0.051**).

---

## 3. Handling the three concrete hurdles

**(a) Train fully observed, test masked.** We reverse-engineered the exact test masking recipe from
`Test.csv` — window length distribution, start position, and per-month S2-only dropout rates — and
expand each training row into *K* masked views drawn from **that measured distribution**. Features
are computed only over active months, so they are invariant to *which* window is exposed. This is
the strategy that won the closely analogous [PLAsTiCC challenge](https://arxiv.org/pdf/1907.04690),
where the winner degraded well-observed training light curves to match the sparse test cadence.

**(b) Optical gaps where radar survives.** In 273/1030 test rows some in-window months have all
optical bands masked while VH/VV survive. The sentinel is handled **per band, not per month**, so a
cloud-masked optical month is not discarded — the radar in it is still used.

**(c) Fixed-0.5 threshold under class imbalance.** `TargetF1` is scored at a hard 0.5 cut. We fit
**Platt scaling on training out-of-fold predictions only**, then cut at a literal 0.5, and emit
genuine calibrated probabilities in both columns. Platt rather than isotonic because it is strictly
monotone, so ROC-AUC is bit-identical while the output becomes a real probability. The realized
positive rate is **reported, never targeted**: `calibrate_legal()` deliberately accepts no config
object, so no prevalence constant can enter it. An earlier revision used a prevalence pin that
violated the rules; §8 documents that in full.

---

## 4. 🔑 The core finding: we measured our own noise floor, and it voided most of our results

For fifteen iterations we operated on a ±0.01 uncertainty band derived from **row-count theory** —
the public slice is ~309 rows, so we reasoned a score is good to about a point. We never measured it.

On 2026-07-22 we did. We reran the champion configuration changing **only the RNG seed**:

```
seed 42  ->  0.8955
seed  7  ->  0.8764
                      sd = 0.0191      seed rank-correlation = 0.9511
```

**Most of our recorded A/B verdicts have effect sizes smaller than one seed swing** — including both
of the wins that made this model our champion. They are not refuted; they are **unresolved**, which
is a different and more uncomfortable claim.

**Three consequences we then had to design around.**

**(i) Our best public score is not our level.** The champion's per-seed public scores are
`42 → 0.906492`, `29 → 0.900715`, `13 → 0.891730`, `21 → 0.878575` — a member mean of **0.894378**
across a **0.028 spread**. The 5-seed pooled artifact scores **0.899882**. We therefore designate the
pooled artifact and treat every single-seed high as an upward fluctuation.

**(ii) Three separate single-seed "records" all washed out.** This is not a hypothetical:

| single-seed-42 result | headline | 5-seed average |
|---|---|---|
| permanence channel | 0.906492 | 0.899882 |
| `vhsq` replacement | 0.913263 | 0.899512 |
| `mean_min` pooling | 0.912759 | 0.899512 |

Three consecutive times we recorded a "new best" on seed 42 and three consecutive times seed
averaging returned it to ~0.8995. We now require **5-seed confirmation before any result is
believed**, and we never designate a finalist on one draw.

**(iii) The ceiling is a bias floor, not a variance floor.** Four structurally different
constructions — the permanence single model, a 4-architecture pool, a `vhsq` variant and a
`mean_min` variant — seed-average to **0.899882 / 0.899643 / 0.899512 / 0.899512**, a spread of
**0.00037** against a public-slice binomial noise of ±0.012. Independent variance models
(Krogh–Vedelsby ambiguity, and a correlated-noise law across plausible ρ) both put the *total*
remaining headroom from any amount of further pooling at **≈+0.0014**, and a noise-propagated
re-derivation caps it at **≤+0.005**. Averaging more seeds, snapshot ensembles and bagging are all
bounded by that number. **Whatever separates us from the leaders is bias under the covariate shift,
not variance** — which is why §6 is about the shift and not about the model.

**The winner's curse, quantified.** Selecting the maximum over k candidates inflates the estimate by
about `SE · E[max of k standard normals]`. Our observed single-seed-versus-pooled gaps sit at almost
exactly the predicted size. This is the reason to designate **pooled, low-variance** artifacts for
the 721-row private slice rather than our best public score.

> **⚠️ An integrity note we are recording against ourselves.** Two of the ledger entries above —
> `champion_replvhsq_seedavg5` (iter 37) and `champion_meanmin_seedavg5` (iter 39) — are
> **bit-identical at 0.899512**. Two structurally different models producing an identical
> `0.6·F1 + 0.4·AUC` on 309 rows is roughly a 1-in-10⁴ coincidence, and this repository has had one
> confirmed duplicate-upload incident before (iter 33b). We flag this as a probable duplicate upload
> rather than a genuine replication, and we do not lean on it as independent evidence.

---

## 5. 🔑 Four offline instruments — and by the last day, four for four falsified

Given a 0.019 noise floor and 100 total submissions, screening candidates *without* spending a
submission was worth more than any single feature. We built two such instruments. **Both passed
their own certification and both were then proven wrong by the leaderboard.** We consider this pair
of failures the most transferable content in the report, because they failed for *the same
structural reason in two different guises.*

### 5.1 Instrument one — the ATC-F1 retro-fitted screen

`tools/offline_validate.py` computes label-free estimators on unlabeled test predictions, then
**retro-fits each against experiments whose public LB we already know** (`experiments/anchors.tsv`).
An estimator earns the right to gate a decision only if it ranks the known anchors correctly.

| estimator | Spearman ρ vs known LB | gate | verdict |
|---|---|---|---|
| **ATC-F1** (metric-aligned average thresholded confidence) | **+0.964** | 15/15 | ✅ CLEARED |
| **DIS** (two-seed disagreement) | **+1.000** (n=4) | 5/5 | ✅ CLEARED |
| ATC (plain) | −0.429 | 6/15 | ❌ FAIL |
| DIV (fold diversity) | −0.857 | 2/15 | ❌ FAIL |
| MARG (margin) | −0.321 | 8/15 | ❌ FAIL |

The screen worked, for a while: it closed the Presto, pooling-diversity, instance-expansion and
multivariate-ROCKET lanes for **zero submissions**.

**Then it failed.** At iteration 26 `c_dropvv` cleared every gate — 2/2 votes, an ATC-F1 margin of
+0.0902 (1.57 seed-sd), and an absolute ATC-F1 (0.8977) above the champion's *best of five* seed
draws. We had pre-committed in writing to submitting on exactly that condition. We submitted.

```
predicted:  +0.0147 LB (raw)  /  ~ +0.005 after our standard 3x discount
actual:     0.884217  =  -0.0113, paired against the seed-42 champion
```

**Wrong in sign.** The diagnosis:

> **All 7 certifying anchors were architecture and objective variants at an *identical 24-channel
> input width*. The retro-fit certified ATC-F1 only *within that family*. `c_dropvv` (22 channels)
> was the first candidate that changed the input *representation*.**

Adding it as an 8th anchor drops ρ from +0.964 to +0.738 on that single point — **and our gate did
not catch it (17/18 still reads PASS)**, because the gate counts concordance over anchor pairs
separated by |ΔLB| > 0.010, a set dominated by trivially-rankable pairs. *A concordance gate built
from easy pairs cannot detect a failure off the manifold the anchors span.*

### 5.2 Instrument three — our own ledger, and a number that never existed

Before the second instrument, one more failure of the same shape, found by auditing our own citations.

For roughly ten iterations this project operated on the belief that *"the public leaderboard leader is
at ~0.94 using plain CatBoost, and their stated lever is ratio/relative features."* That sentence
appears in `PROJECT_STATE.md` as settled doctrine, and it directed a great deal of work — including
the entire tree lane.

We went back to the source. The forum thread contains one post from the competitor in question:

> *"don't blame the trees — I'm on gradient boosting too (CatBoost-family, nothing exotic)… If tuning
> models and swapping features isn't moving the LB, that's a hint the lever is elsewhere."*

**No score is stated. They never claimed to be the leader.** When another participant directly asked
for their best LB score, they did not reply. Tracing our own documents shows a clean citation drift:
an early research note correctly recorded "a score band" and "this competitor uses CatBoost" as *two
separate facts*; a later note merged them into "sdv, in the 90s club, uses plain CatBoost"; the
iteration-30 log promoted that to "the LB **leader** (~0.94)"; and `PROJECT_STATE.md` recorded it as
doctrine. **We also inverted the one actionable sentence they did write** — they said swapping
features was *not* moving their leaderboard, and we recorded "their named lever is ratio features."

The corrected picture, from the leaderboard's own published digits: the top is a **cluster at 0.9259
and 0.9301**, not a lone 0.94. Our gap is **≈0.030, not ≈0.05**.

This is the same failure as §4's prevalence constant and §8.1's "+0.07" pin credit: **a number entered
the ledger without provenance, was compounded by summarization, and then steered months of work.** We
now require every external claim in `PROJECT_STATE.md` to carry a VERIFIED-with-quote or INFERRED tag.

### 5.3 Instrument two — the adversarial-validation gate (and the same lesson again)

For the tree lane we built `tools/adversarial_cv.py`: train on the least-test-like 70% of train,
evaluate on the **most-test-like 30%**, on the theory that a model which survives that holdout will
survive the real shift. At iteration 39 it returned a confident **[GO]** — AUC 0.960, F1 0.884, only
0.024 below a random holdout.

Iteration 40 built the recommended model and submitted it: **0.7186**, against a predicted ≥0.90.

**Wrong by 0.18.** The diagnosis is the same failure in new clothing:

> **A test-like-*covariate* holdout still carries TRAIN labels.** It therefore samples `P_train(y|x)`,
> and is structurally blind to **conditional** shift — which our own label-shift gate had already
> proven present (mixture goodness-of-fit KS D = 0.186, p = 0.000). The gate could only ever measure
> the covariate half of a shift we knew had two halves.

**The generalized lesson, which we would offer as our main methodological contribution:** every
instrument we relied on — including our own written record — certified itself on a *sub-manifold of
the thing it was asked to predict*: ATC-F1 on one input width, adversarial CV on one half of the
shift, the ledger on one unsourced quotation. In each case the instrument's own check was
structurally incapable of reporting that it had left its domain of validity. **A retro-fitted
validator inherits the support of its calibration set, and concordance gates do not reveal where
that support ends.**

The practical rule we now follow: **an offline gate may gate a PAIRED DELTA, never a LEVEL.** Both
failures above were level predictions. A paired difference between a treatment and a control arm,
both equally blind, cancels the blindness in the difference.

---

### 5.4 Instrument four — a graph estimator that finally answered the prevalence question

> ⚠️ **THIS INSTRUMENT WAS FALSIFIED TOO, ON THE LAST DAY. See §8.7.** Its estimate (~0.59) agreed
> with our realised rate (0.587) and we read the agreement as confirmation. The solved public cell
> puts the true prevalence at **0.5435**: we *over*-predict, and the graph estimator was reproducing
> our bias rather than checking it. Retract the claim in this heading — all four instruments were
> falsified, and the honest count is **four for four**, not three of four. The two retired
> estimators it replaced, MLLS (0.578) and BBSE (0.559), were both **closer to the truth than the
> instrument we preferred over them**, which is worth sitting with: we retired them for a sound
> methodological reason and then trusted a replacement with no independent check.

The two instruments above were certified and then falsified. This one is the exception, and it
answered a question that had been formally open in our ledger.

**The question.** Is our realized positive rate (0.587) too low? We are missing ~9 true positives at
the boundary (§6.5), and the obvious explanation is that our cut is too conservative. We could not
test it: the two estimators that would have — MLLS (0.578) and BBSE (0.559) — were **retired** when
`tools/label_shift_gate.py` rejected `p(x|y)`-invariance at p ≈ 0. An estimator retired *for*
correction cannot then be cited as evidence that *no* correction is warranted, so we struck both and
the question stayed open.

**The instrument** (`tools/graph_gate.py`, zero submissions, deterministic). Build a k-NN similarity
graph over rows — train rows masked to test-like 4–6 month windows through the same `_mask_views`
replica the pipeline calibrates on, features being per-band means over *observed* months
(n-invariant, so the graph measures signal rather than window length). Propagate train labels along
edges. This assumes **no label shift at all** — only adjacency plus labels — so the failure that
retired MLLS and BBSE does not apply to it.

| | mask-matched | unmasked control |
|---|---|---|
| test rows' k=10 neighbours that are labelled | 24.5% (random mixing 63.8%) | 22.5% |
| train–train edge label homophily | 0.9252 (chance 0.5191) | 0.9518 |
| parameter-free propagation, k=10 | combined 0.9456 | 0.9684 |
| **implied test positive rate** | **0.591** *(0.599/0.591/0.596/0.591 at k=5/10/25/50)* | **0.529** |

**Two findings.** First, the estimate is **flat in k** (spread 0.0078) and lands at **0.591 against
our realized 0.587** — an estimator with a completely different bias structure independently puts our
operating point where it already is. That is evidence *against* the "our cut is too conservative"
hypothesis, and it is why we did not chase the ~9 rows by lowering the cut.

Second, and more general: **regime-matching moves the estimate by +0.062.** Comparing 12-month train
rows to 4–6-month test rows gives 0.529; matching the masking first gives 0.591. The naive comparison
would have told us our positive rate was far too *high*. **Regime mismatch alone can manufacture an
apparent prevalence gap in either direction** — which is the same lesson, arrived at from a
non-parametric direction, that §7's regime-matched calibration row reaches from the Platt side.

**The caveat, which governs how far we take it.** The replica reproduces the *window masking* but not
the *temporal* shift, the larger half of the problem. Every figure above is therefore optimistic, an
upper bound. The asymmetry that follows is deliberate: "the graph agrees the cut is roughly right" is
robust — a shift can only make the graph worse, and it already agrees — whereas "the graph says move
the cut" would be fragile. We acted only on the robust direction, which is to say we did not act.
**Diagnosis only; nothing this instrument prints reaches the 0.5 cut.**

---

## 6. 🔑 The anatomy of the shift — and a correction to our own analysis

To ask *where* the shift lives, we transplanted **real test masks onto train rows**, so the
window-length and dropout components are removed by construction and only the residual remains. All
numbers below are measured on that window-matched comparison, with a 66-feature n-invariant bank.

**First result — no single feature carries it.** Marginal adversarial AUC has a **maximum of 0.7024**
and a mean of 0.5698, yet a gradient-boosted model on the joint bank separates the domains at
**0.9670**. Nulls are clean: missingness-count features alone give 0.4636 (the transplant does not
leak), disjoint train halves with independent transplants give 0.4934, and test-vs-test disjoint
halves give 0.5046.

**Our first explanation was wrong, and we are correcting it here.** We initially concluded that the
shift must therefore live in feature *interactions*, since a linear model reached only 0.9190 against
the GBDT's 0.9670. An adversarial re-examination broke that:

| model on the same 66 features | adversarial AUC |
|---|---|
| GBDT depth 3 (original claim) | 0.9670 |
| **GBDT depth 1 — purely additive, no interactions** | **0.9430** |
| GBDT depth 2 | 0.9524 |
| logistic, median-impute, C=0.5 (the original weak baseline) | 0.9190 |
| **logistic, rank-transform + explicit missing indicators, C=10** | **0.9438** |

The interaction premium is only **+0.0239**, and our "linear can't do it" figure was an artifact of a
badly-specified linear baseline rather than a property of the data. A properly specified additive
model matches the additive GBDT. Moreover, if the 66 features were merely *independent*, pure
accumulation of their weak marginals predicts a joint AUC of **0.9594** — nearly all of what we
observe.

> **Corrected conclusion: the shift is thinly spread across many weak marginals that accumulate
> additively.** 42 of 66 features exceed adversarial AUC 0.55 while none exceeds 0.71. This is why
> per-feature screening fails — not because the signal hides in interactions, but because **there is
> no clean subset to keep.** It also explains, retrospectively, why deleting individual bands
> (`c_dropvv`) and dropping whole feature families both failed to help.

**Second result — and it retires adversarial AUC as a selection criterion entirely.**

The natural next step was to align the marginals away. It works, completely, and it buys nothing.
Per-domain standardization drops linear adversarial AUC from **0.9291 to 0.3608 at zero label cost**
(label AUC 0.9850 → 0.9850). A transform meeting our stated target also exists — whitening both
domains reaches adv-AUC **0.6168** at label-AUC **0.9710**. So the shift is marginal, and it is
erasable.

But when we measured what these transforms do to *actual transfer* — source→target benchmarks with
known target labels — the ranking inverted:

| | |
|---|---|
| Spearman(adversarial AUC, realized transfer gain) across 9 transforms | **+0.676** |
| the same, across modality subsets | **+1.00** |
| the *best*-adversarial transform (whiten-both), realized transfer | **−0.0503** (5/5 splits negative) |
| **SAR-only** — best shift signature (0.8452) | **worst transfer: F1 0.7461** vs optical's 0.8786 |

**The correlation is positive, meaning adversarial AUC points the wrong way.** Both adv-AUC and label
AUC are driven by the same thing — information content — so suppressing a representation's
domain-detectability suppresses its signal along with it.

The clinching measurement: a deliberately *mild* synthetic shift registers **adv-AUC 0.9955** while
costing only **0.0046 AUC** of transfer. **Our real train/test adv-AUC is 0.9670 — lower than the
mildest shift we could construct.** The statistic saturates long before it becomes informative. That
is the mechanical reason iteration 39's gate returned a confident false GO, and it means every
"drop the shift-carriers" rule we ever wrote — including iteration 40's removal of 27 optical
features — was steering by a broken compass.

> **This supersedes a claim in an earlier revision of this very section.** We had written that SAR
> transfers better than optical because its adversarial signature is cleaner. Measured against real
> transfer, the opposite holds. We have killed the SAR-reweighting idea it implied.

---

## 6.4 The instrument that replaced it

What survives is a gate that compares a **paired delta** rather than a level: run a treatment arm and
a control arm through the *same* proxy-domain split with known target labels, and gate on the
difference. Both arms are equally blind to whatever the proxy fails to capture, so the blindness
cancels. Applied retrospectively, this gate would have caught iteration 40. It also decomposes any
apparent F1 gain into **ranking** versus **cut placement** — which matters because 63–98% of the
apparent F1 gains we screened turned out to be cut placement, i.e. threshold tuning in disguise.

---

## 6.5 Where the remaining gap actually is

> ⚠️ **CORRECTED IN §8.7 — read that first.** Every row count in this section is denominated in a
> public test size of **309**, which we later proved cannot be the true value, and the decomposition
> below reads our error as a *recall* deficit. Both are wrong. The correct public set is **n = 333,
> P = 181**, and our champion makes **44** confusion errors on it — **27 false positives against 17
> misses**. The qualitative conclusion of this section (the gap is local ranking near the cut, not
> global calibration) survives; the direction does not.

Because the leaderboard publishes F1 and AUC as separate columns, we can decompose our own score
without any tuning (see §8.2 on the line we drew here). Our submitted artifact splits as
**AUC 0.9354 / F1 0.8761**; the top of the cluster is **AUC 0.9449 / F1 0.9202**.

| component | share of the 0.0302 gap |
|---|---|
| AUC term (ranking overall) | ~13% |
| **F1 term** | **~88%** |
| …of which: operating point / calibration | only **5–13%** of the F1 gap |
| …of which: **ranking of rows near the 0.5 cut** | the remaining **~90%** |

In absolute terms the entire gap is about **52 rows** out of 309 — we make ~144 confusion errors
where the leaders make ~95. And because F1 is computed at a fixed cut, **a row near the boundary is
worth 1.9–3.8× a row in the deep tail.** This is the single most useful strategic fact we have: the
remaining gap is not calibration and not global ranking, it is *local ranking quality among genuinely
ambiguous cells*. That is a modelling problem, not an operating-point problem — which is why §8.2's
lever is worth so little and why we did not pursue it.

---

## 6.6 🔑 The one thing that broke the ceiling

Everything above says the barrier is *bias under the covariate shift*, that pooling cannot touch it,
and that no transformation of the input features helps. That leaves exactly one lever: **information
about the target distribution itself.** The 1,030 test rows are 57% of our labeled set, they are the
only target-domain data in existence for this problem, and their *features* are supplied — so using
them is legal by construction (no test labels, no external data, no threshold tuning).

We ran two zero-parameter forms of this at iteration 41, both as 5-seed averages against the
0.899882 finalist:

| arm | mechanism | 5-seed LB | vs finalist |
|---|---|---|---|
| **D — soft self-distillation** | train against a pooled teacher's *soft* probabilities on test rows (T=1, never thresholded) | **0.909868** | **+0.009986** ✅ |
| T — transductive consistency | the proven `Var_k(logit)` cross-view penalty pointed at test rows; forms no label | 0.893752 | −0.006130 ❌ |

**Arm D is the first artifact in this project to clear ~0.8995.** It beats the pre-committed +0.006
bar, it is a five-seed average rather than a lucky draw, and **both metric terms improved** — AUC
0.935 → 0.944024, F1 0.876 → 0.887097. That matters: an operating-point trick would move F1 alone.
This moved the ranking. Our AUC is now **within 0.00087 of the leaderboard leader's**, and the entire
remaining gap is the F1 term.

**Arm T failed, and its failure is the more interesting one.** Its individual seeds produced our two
highest single scores ever recorded (0.914179 and 0.908873), yet the pool *lost* 0.0178 — the only
negative pooling gain anywhere in our ledger. The decomposition says exactly what happened: pooled
AUC (0.926687) sits *between* its members' values, so the ranking pooled normally, while pooled F1
(0.871795) falls *below both* members. The operating point drifted. Mechanism: a variance penalty on
unlabeled rows is minimized by a constant predictor, so it compresses the logit distribution; per-seed
Platt slopes then diverge, and averaging the calibrated probabilities lands at the wrong positive
rate. **Had we designated on the single-seed 0.914179 — our best number ever — we would have shipped
an artifact whose own five-seed average is below the model it replaced.** It is the fourth time this
competition has offered us that trade, and the first time the discipline from §4 was load-bearing on
a score we would genuinely have wanted.

---

## 7. Negative results (44 LB-gated iterations)

A competition report that lists only what worked hides most of the information. Every row below cost
a submission or a screened experiment, and each is reproducible from `experiments/LB_LOG.md`.

| lane | outcome | evidence |
|---|---|---|
| **Gradient-boosted trees** | ❌ closed, 3 independent failures | naive CatBoost **0.6976**; blend member **−0.0136**; shift-robust CatBoost with feature-shift removal **0.7186** (no-drop control 0.6903). Trees key on exactly the joint structure the shift corrupts |
| **Amplitude normalization** | ❌ fatal | replacing values with within-series rank collapsed OOF 0.975→0.86. **Persistently-low backscatter level *is* the class signal** — do not detrend, difference, or instance-normalize |
| **Cross-polarization contrast `VH−VV`, *linear* forms** | ❌ closed — but the test was weaker than we claimed | toxic as a Transformer channel (−0.0228); a top shift-carrier in the tree bank. **However**: our own `SDWI` implementation is *exactly* `−5.697415 + 0.230259·(VH_dB + VV_dB)` — verified numerically to 3.6e-15 — i.e. affine in the two bands the model already has. A linear layer spans it, so these arms measured **width cost, not missing information.** The dual-pol *indicator* (below) remains untested |
| **Dispersion features** (IQR, L-scale) | ❌ falsified as "shift-safe" | theoretically location-invariant, empirically shift-carriers (0.57–0.66) — further evidence the shift is not a pure location shift |
| **Foundation models (Presto, frozen)** | ⚠️ **closed on retracted evidence** | the frozen encoder re-encodes the shift (adv-AUC 0.965–0.976), but the lane was closed by ATC-F1 — the instrument §5.1 later falsified. Fine-tuning (LP-FT) was never tested. We record this as *unfinished*, not refuted |
| **Cross-model-class blending** | ❌ closed, n=2 | ROCKET member → −0.009; GBDT member → **−0.0155, paired, ≥2.5σ** |
| **Feature-space deletion** | ❌ closed | `c_dropvv` −0.0113; and §6 explains why: the shift is distributed, so no small subset carries it |
| **Missing-indicator deletion** | ❌ closed for free | indicators alone separate train/test at **0.4758, below chance** — our masking already matches test dropout |
| **Importance weighting / DANN** | ❌ dead | effective sample size collapses at adversarial AUC ≈0.99 |
| **Label-shift priors (BBSE / Saerens-EM)** | ❌ **unsafe, gate FAILED** | the mixture goodness-of-fit test rejects pure label shift (KS D = 0.186, p = 0.000). Two estimators put π̂_test at **0.559 / 0.578**; our realized rate is already ≈0.55, so the upside was near-nil even had it been safe |
| **Water indices (WIF/EVI/SDWI/AWEI/NDWI)** | ❌ −0.075, and partly degenerate | **SDWI is exactly affine in (VV_dB + VH_dB); AWEI is exactly linear; EVI ≈ 2.5(NIR−Red) over water; NDWI/MNDWI are 0/0-conditioned over water.** A linear model already spans several of them |
| **More masking views / longer training** | ❌ | K=4 → −0.0115; K=2 is a sharp optimum |
| **Seed averaging as a climber** | ➖ variance only, and now capped | lands at the member mean; §4(iii) bounds all further pooling at ≤+0.005 |
| **TTA (hole-punching)** | ➖ −0.0023 | diagnosed in §6 terms: masking random *interior* months produces windows that occur in neither train nor test — off-manifold augmentation |
| **Distillation weight α** | ❌ closed *exactly*, iter42 | a 5× sweep of α moved nothing: 0.907370 / 0.910837 / 0.906642, total spread 0.0035 inside ±0.012 noise. The F1 column inverts to **TP = 165, 165, 164** at predicted-positive counts differing by one — **the entire ladder is one true positive out of 309 rows.** α=0.7 at 5 and at 10 seeds have *bit-identical* AUC (0.944024425), so the seed count changed the ranking not at all and the whole −0.0032 was one row crossing the cut |
| **Cross-polarization contrast, *indicator* form** | ❌ closed, iter43 — completing the lane above | the dual-pol gate `1[VH<−21]·1[(VH−VV)<−8]`, as a width-neutral replacement (0.904005) and as an addition (0.907616), neither clearing +0.006. **VH−VV is now closed with three independent forms of the same quantity failed:** raw (−0.0228), affine/SDWI (provably spanned), indicator (this round). Note the honest split: the added form posted our **best AUC ever, 0.946460, above the leader** — the gate genuinely improves *ranking* and loses it back at the cut |
| **F-measure surrogate losses (sigmoidF1)** | ❌ **cancelled before it ran**, iter44 | three independent refutations. (i) **Platt annihilation, a theorem:** any affine logit reparameterization `z'=αz+β` satisfies `σ(a(αz+β)+b) = σ((aα)z+(aβ+b))`, so a train-refit Platt recovers the identical function — sigmoidF1's boundary effect, logit-adjusted loss and balanced softmax lie *exactly* in Platt's span, and our pipeline refits Platt on the next line. (ii) **No published evidence** any F-surrogate beats BCE at a *pre-specified* fixed 0.5 with all hyperparameters fixed a priori; sigmoidF1's own fixed-0.5 result is defeated because its `η` is a logit offset, so the `η` grid search *is* a threshold search. (iii) **Measured density:** only 29–38 of 1030 rows lie in [0.45,0.55]; the move needed to reach the F1 optimum is 0.21–0.33 and the mechanism supplies ~0.006 — roughly 35× short |
| **Regime-matched calibration (`R=1`)** | 🔬 iter44, result pending | the one calibration defect that is a *fact of the code*: an OOF row is the mean of R=2 masked window views from 1 fold-model, a test row is 1 real window averaged over `n_splits` models, and Platt is fit across that mismatch. Corrected offline by `tools/regime_match.py` at zero extra training cost. **We pre-registered a null as the expected outcome** — with ~3% of test mass within 0.05 of the cut, a slope change this size cannot move enough rows |

**What the blending failures do and do not show.** Under a metric with a hard threshold, the
error-ambiguity decomposition that justifies ensemble diversity **does not apply** — there is no such
decomposition for 0-1 loss (Brown & Kuncheva, MCS 2010). Our two foreign-member blends lost
**monotonically in the member's own level deficit**, so we gate members on **level gap, not
decorrelation**. **We flag a limitation in our own evidence:** both members were also weaker, so
member strength and member diversity are **perfectly confounded at n=2**. What we demonstrated is
that a weak member hurts; the stronger claim that *diversity itself* is harmful is **not identified**
by our experiment, and an earlier draft of this report overstated it.

**A scope condition that bit us.** Every blending result above was measured under the (later removed)
prevalence pin. When we removed it, the same 4-architecture pool went from −0.0009 against its best
member to **+0.0100**. The pin overwrote every member's operating point to a common value, so pooling
could only average the *ranking* (mean ρ = 0.9524, almost nothing independent left); a literal 0.5
cut also averages the members' *calibration*, where they genuinely disagreed (positive rates
0.534–0.586). **Ensemble conclusions drawn under a pinned threshold do not transfer to an unpinned
one**, and ours were all drawn under the pin.

---

## 8. ⚠️ Limitations, and two rule questions we raise against ourselves

### 8.1 The prevalence pin was a rules violation — found, disclosed, and fixed

We read the rules page directly on 2026-07-28. It states, verbatim:

> *"Setting a probability threshold is strictly forbidden. Your binary target should be based on the
> default threshold of 0.5."* … *"do not set thresholds (or round your probabilities) to improve your
> place on the leaderboard."* … *"Zindi will need the raw probabilities. This will allow the clients
> to set thresholds to their own needs."*

Earlier revisions argued our construction was *prevalence correction* (allowed) rather than
*threshold tuning* (forbidden), because the cut stays literally at 0.5. **That argument does not
survive reading the rule, and we withdrew it.** Two distinct problems: `TargetF1` computed
`thresh = quantile(logit(p), 1 − π̂)` and shifted so the threshold landed at 0.5 — the literal 0.5
was cosmetic, and **π̂ = 0.649 was swept against leaderboard feedback**, not derived from training
data. And `TargetRAUC` returned uniformly-spaced ranks, not probabilities, defeating the stated
rationale of the raw-probabilities rule.

**Why we disclosed rather than quietly hoping.** The final score is 65% private LB + 35% code review
**of the top 5 only**. If we do not reach the top 5, the pin was worth nothing. If we do, our code is
read by exactly the people who wrote the rule. **The gain is only cashable in the scenario that
triggers the review that would void it.** There is no branch on which keeping it pays.

**Status: FIXED, and it cost almost nothing.** Measured on the leaderboard, same config, same seed,
same folds, only the operating point changed:

| | public LB |
|---|---|
| pinned (rules-violating) | 0.895500 |
| **legal** | **0.889686** |
| paired delta | **−0.0058** |

Our protocol calls a paired A/B *suggestive* only at ≥0.006, so **the cost of compliance does not
reach significance.** Inverting the delta shows why: the pin was adding ~104 positives that were
about **49% correct — coin flips.** It bought volume, not accuracy. The ≈+0.07 once credited to it
was measured in iteration 02 on the **superseded GBDT**, and we carried that constant for 25
iterations without re-measuring it — the same methodological error as §4, in a different guise.

**A reviewer can verify the fix from the submission file alone**, without running our code:
`TargetF1 == (TargetRAUC >= 0.5)` on every row. That is the entire compliance claim.

**We then audited every code path in the repository that can write a submission.** The result, and
the fixes, in full:

| emitter | status |
|---|---|
| `run_pipeline.py` (all models) | ✅ compliant — `calibrate_legal`, train-only Platt, literal 0.5 |
| `tools/seed_average.py` → the 10-seed pool behind **finalist #1** | ✅ compliant — `calibrated_pool`, literal 0.5 |
| `tools/regime_match.py` → **finalist #1** (the shipped artifact) | ✅ compliant — refits Platt on an **R=1** calibration set, literal 0.5 preserved. Audited against all three prongs below |
| `tools/arch_blend.py` → **finalist #2** | ✅ compliant — `calibrated_pool`, literal 0.5 |
| `run_presto.py` | 🔴 **was non-compliant → FIXED** — emitted `target_prevalence_shift` (a threshold shift) + `score_for_auc` (ranks, not probabilities). Now routed through `calibrate_legal`. Any Presto artifact produced before this fix was ineligible |
| `tools/blend.py` | 🔴 **was non-compliant → now GUARDED** — same illegal pair. It is retained to reproduce historical pinned anchors, and now refuses to run unless `compliance_mode=pinned` is set explicitly |
| `src/calibration.py::calibrate_for_f1` (isotonic branch) | 🔴 **misleading comment → FIXED** — the code found the F1-optimal cut and shifted it onto 0.5 under a comment claiming this made it *"legal"*. It does not: that is threshold tuning with a wrapper. The comment now says so |

**Both designated finalists are clean.** The defects were in secondary tooling — but a reviewer opens
those files too, and a comment asserting that threshold-shifting is legal is precisely the kind of
thing that should be found and corrected before review, not during it.

**`tools/regime_match.py` deserves its own paragraph, because it is the one place we touch the
operating point.** It is worth stating exactly why it is legal rather than asserting it, since
"we refit the calibration" is superficially close to "we tuned the threshold," and it is not.

The defect it fixes is real. Platt is fit on out-of-fold predictions, but OOF rows and test rows do
not share an averaging structure: an **OOF row is the mean of R=2 masked window views from ONE
fold-model**, while a **test row is ONE real window averaged over `n_splits` fold-models**. Each
side is variance-shrunk on an axis the other is not. Under the old prevalence pin this was harmless
because the cut was re-derived downstream — but under a **literal 0.5 cut the Platt slope *is* the
operating point**, so fitting it across that mismatch is a genuine bug. `regime_match` rebuilds the
calibration set at R=1 and refits offline, at zero extra training cost.

Against the three-prong test we hold ourselves to:

| prong | verdict |
|---|---|
| (a) the decision rule stays a literal `0.5` | ✅ unchanged; the emitted CSV passes the same row-wise audit as every other artifact |
| (b) every knob fixed by a **train-only** criterion, never against a realized positive rate or leaderboard feedback | ✅ `R` is read **solely on the held-out path** — it is the one operating-point lever in the pipeline that structurally cannot see test outcomes. It was pre-committed in version control (`run_current.sh`, iteration 44) with the rule *"R=1 ships whatever positive rate it produces"* **before any score for it existed** |
| (c) it corrects `p(y|x)` under a *demonstrably* mis-specified model, rather than relabeling a fixed estimate | ✅ the mis-specification is measured, not assumed: the two averaging structures are visibly different in the code, and the correction was pre-registered to produce a NULL — which it did, twice (+0.001006 and +0.002831, both inside the ±0.006 band) |

It also ships with its own **bit-for-bit control**: at `--views all` the rebuild must reproduce
`tools/seed_average.py` byte-identically, which `reproduce_champion.sh` asserts and aborts on. That
control is not decoration — getting it to pass required replicating the native-float32 reduction
exactly, because the obvious float64 `np.add.at` rewrite agrees only to ~1e-9, which is invisible in
OOF metrics but survives Platt into the reported `TargetRAUC` decimals.

### 8.2 A second lever we found and deliberately did not pull

Late in the competition we noticed the leaderboard exposes **F1 and ROC-AUC as separate columns**
alongside the composite. Because `F1 = 2·TP/(P̂+P)` is a ratio of small integers on a 309-row slice,
the published digits are enough to recover `TP` and `P̂+P` exactly. We carried out that inversion as
a diagnostic.

**⚠️ A correction to our own earlier claim, recorded here rather than quietly edited out.** An
earlier revision of this report and of `experiments/LB_LOG.md` asserted that the **AUC** column
inverts too — that its quantum `1/(P·N)` pins the public-slice positive count at `P=188, N=121`.
**It does not, and that claim is withdrawn.** An integrality sweep over every split of 309 (plus
308/310, plus the tie half-quantum `1/(2PN)`) returns a best max-residual of **0.070**, where
9-decimal reporting would allow ~1e-5. Every split is rejected, so `P` is **not** derivable from the
AUC column, and the quantities we had built on it — `P̂`, FP, FN, FPR, precision, recall — were never
established. The honest figure is **P = 190 ± 7, an ESTIMATE** from our own logged full-test positive
rate (0.5874 → E[public P̂] = 181.5, hypergeometric sd 7.2), bounded `P ∈ [164, 208]`. What survives
is only what the F1 column gives directly: the `TP` counts, and the ~9-true-positive gap in §0, both
of which are `P`-independent.

**We did not, and will not, use any of it to set our operating point.** Tuning the decision boundary
to a reverse-engineered public-slice composition would be leaderboard probing — a clear violation of
the same rule as §8.1, and one that would also overfit the 309 public rows against the 721 private
rows that actually decide the competition. The analysis is retained only as a *diagnosis* of where
our gap lives (it is entirely in the F1 term), and no number derived from it enters
`config/config.yaml`, `src/calibration.py`, or any submitted artifact. We record it here because a
reviewer should know we found the channel and chose not to use it.

**The standing rule this produced**, which governed the rest of the project: *any leaderboard-inverted
quantity is diagnosis only and must never feed the operating point.* It is what ruled out the
otherwise-obvious fix for the ~9-true-positive gap — lowering the cut until our predicted-positive
count matched the leader's — and it is why iteration 44 pursued a **train-only** calibration
correction (§7) instead.

### 8.3 Reproducibility caveats, stated plainly

A single seed drives all RNGs and per-`(row, view)` seeds are derived deterministically, so the
masking augmentation is exactly reproducible. However, for the `seq` path on GPU we do **not** set
`torch.use_deterministic_algorithms` or `cudnn.deterministic`, so CUDA attention kernels may
introduce small run-to-run differences. GBDT runs are bit-identical; `seq` runs reproduce to within
that kernel nondeterminism. Given §4, a reviewer should expect run-to-run variation on the order of
the seed effect, and **should reproduce the pooled artifacts rather than any single seed.**

**Measured, 2026-08-14 — the caveat above is conservative.** iter45 happened to re-run ten configs
that iter44 had already executed (five `teacher_perm_s{42,7,13,21,29}` and five distilled students at
α=0.7), in a *different* Colab session on a *different* GPU allocation, a day apart. **All ten
reproduced to every logged decimal:**

| config | iter44 `final_oof` | iter45 `final_oof` |
|---|---|---|
| `teacher_perm_s42 / s7 / s13 / s21 / s29` | 0.97347 / 0.97437 / 0.97143 / 0.97161 / 0.97414 | identical |
| distilled α=0.7, seeds 42 / 7 / 13 / 21 / 29 | 0.97594 / 0.97773 / 0.97403 / 0.97350 / 0.97317 | identical |

Both statements are true and a reviewer should have both: the *guarantee* is genuinely absent
(we do not set the deterministic flags, so we cannot promise bit-exactness), while the *observed*
behaviour over ten runs and two sessions was exact agreement to five decimals. The pooled-artifact
recommendation stands regardless, because it is about seed variance (§4), not kernel nondeterminism.

### 8.4 A ceiling we cannot reach, and why we stopped trying

The pond-mapping literature stands on three legs, and two are unavailable to us:

| leg | what it buys | available? |
|---|---|---|
| pixel-wise temporal permanence (e.g. median VH) | the water mask | ✅ the only leg we have |
| **shape** — compactness, perimeter, LSI, dike detection, GLCM texture | ***the entire pond-vs-natural-water separation*** | ❌ lat/lon stripped, isolated pixels |
| DEM / OSM / JRC surface-water overlays | removes lakes, reservoirs, rivers | ❌ external data, rule-barred |

Ottinger (2022) calls compact shape *"a characteristic and **defining** feature for the distinction
between natural standing waters and managed aquaculture ponds."* Phan et al., ground-truthed, report
that flooded rice at ~10 days after sowing reads **VV −13 dB, VH −22 dB — open-water values**. At any
single date our positives and our hardest negatives are radiometrically identical; only the joint
behaviour over months separates them. **Published 89–95% accuracies are earned with the two legs we
do not have.**

*Geography caveat:* this literature is overwhelmingly coastal East/Southeast Asian intensive
aquaculture. The FAO/ITU framing suggests our data may be African smallholder — smaller ponds, less
intensive feeding, more rain-fed drawdown — which would weaken both the eutrophication signal and the
"permanently full" assumption. We found no quantitative African pond-mapping study to calibrate
against, and we flag our physics-derived expectations as **unvalidated for this setting.**

---

### 8.5 The one hole in our own central theorem — found by an outside reviewer, tested, and closed

We put considerable weight on what this report calls the **Platt Annihilation Theorem**: because
`σ(a(αz+β)+b) = σ((aα)z + (aβ+b))`, a train-refit two-parameter Platt map *exactly* removes any affine
reparameterization of the logit. That single identity closed a large number of lanes for us — every
additive logit prior, logit-adjusted loss, balanced softmax, and the planned sigmoidF1 arm.

An external deep-research review found a real gap in how we were applying it, and we record the
correction because it was a good one. The theorem's scope is *affine maps only* — and **the calibrator
we deploy is itself a map on the logit.** We had only ever used Platt, which is a two-parameter
affine-on-logit map. Substituting a monotone but NON-affine family therefore moves the fixed-0.5
crossing onto rows no Platt fit can reach, while preserving the ranking and hence the whole
`TargetRAUC` column. That is not a loophole in the rules; it is a modelling choice the theorem simply
does not constrain, and we had not tested it. The review ranked it the single cheapest and cleanest
remaining lever.

**We tested it before spending a submission** (`tools/calib_family_gate.py`, zero submissions,
deterministic, on 10 `amix` and 5 `dpa` seed bundles at the R=1 regime-matched OOF). The contribution
worth recording is the reframing. Beta calibration (Kull, Silva Filho & Flach, AISTATS 2017, PMLR
54:623–631) fits

    p = σ( a·ln(s) + b·(−ln(1−s)) + c )

and at `a == b` the two log terms collapse to `a·logit(s)` — so **beta contains Platt exactly as its
`a == b` submodel.** The entire non-affine content of the lever is one degree of freedom, `|a − b|`,
which turns "is this lever real?" from a judgement call into a **nested likelihood-ratio test on 1 df**
that costs nothing to run. Three findings:

| test | result | reading |
|---|---|---|
| LR test, beta vs Platt | 1/10 and 1/5 members reject at 0.05 (0.5 / 0.25 expected by chance). **Pooled — the fit an artifact actually ships — p = 0.134 and p = 0.290** | the extra parameter buys no significant likelihood |
| **direction of the test crossings** | beta **15 down, 0 up**; isotonic **23 down, 0 up**. Not one row moves up, in any configuration | **the lever's sign is reversed** |
| isotonic OOF AUC | in-sample **+0.00197**, 5-fold cross-fitted **−0.00273** | overfits the calibration set at n=1817 |

The middle row is the decisive one, and it is the review's own argument turned against the proposal.
Its mechanism was that a calibrator anchored at the train base rate sits too HIGH at a fixed 0.5 and
suppresses true positives — so the lever must move rows **UP**. Measured, it moves them exclusively
DOWN. That is not a weaker version of the proposal; it is the proposal with its sign flipped, and it
would cost F1. **Direction was not in the proposed kill condition**, which is precisely why that kill
condition was insufficient — and why the gate we committed reports direction first.

Two further corrections we owe the record. First, the review's premise that our probabilities are
"anchored at the 40.23% train base rate" is **empirically false for this model**: our realized test
positive rate is **0.5845**, already sitting on the independent k-NN graph estimate of 0.587–0.591
(§5.4). The model's own test scores carry the base-rate shift; Platt is not pulling us back to the
train prior. That removes the mechanism behind the proposed follow-up (refitting the calibration set
to the deployment prevalence), which the review itself flagged as the compliance-fragile option under
Elkan's equivalence theorem (IJCAI 2001) — a reviewer could legitimately read prevalence-reweighting
as threshold-moving in disguise. **We did not attempt it.** Second, the isotonic result is a
methodological warning larger than the result: Niculescu-Mizil & Caruana (ICML 2005, DOI
10.1145/1102351.1102430) put the Platt/isotonic crossover near 1000 calibration points, and at 1817 we
are close enough to it that the in-sample and cross-fitted numbers **disagree in sign**. The in-sample
number is the obvious one to compute and it points the wrong way.

**Outcome: the lane is closed, no submission was spent, and the shipped operating point is unchanged**
— a train-only Platt map and a literal 0.5. We regard this as the model of how the offline instruments
in §5 are supposed to work: a plausible, well-argued, correctly-motivated idea refuted for free.

### 8.6 We mis-cited our own motivation for a feature, and built the gate that would have caught it

Iteration 43 killed the cross-polarization feature `VH − VV` in three independent forms, all null. We
recorded that as a surprise. It was not one, and the reason is worth stating against ourselves.

**We cited the wrong thing.** The canonical SAR feature in the aquaculture-mapping literature is **VH
alone, pixel-wise temporal median** — Ottinger et al. (IGARSS 2018, DOI 10.1109/IGARSS.2018.8651419):
*"we used scenes in VH polarization"*, *"the pixel-wise median was calculated … to identify permanent
and stable low scatterers"*. The dual-pol **ratio does not appear in that pipeline at all**. Ullmann
et al. (Front. Remote Sens. 3:905713, 2022) measured what polarimetric derivatives add over plain
intensity for water surfaces: **0.1%**. Our three nulls were the literature's own prediction; we had
been citing a paper for a feature it does not use. (This also explains, retroactively, why the plain
VH permanence indicator `1[VH_dB < −21]` is the one feature that ever *won* here, §6.6.)

**And mechanically it could never have helped.** `VH − VV` is an **exactly linear** function of two
columns the model already receives, so a model given both can represent it at zero cost. Handing it
over as a new input adds no information — only width, and added width has lost every time in this
project (§7). That is cheap enough to be a gate, so we built one: `tools/feature_span_gate.py`
cross-fits a ridge of each candidate feature on the 144 raw values and reports R². R² → 1 means the
feature is already inside the model's reachable span. A second, competition-specific gate reports the
Spearman correlation between the feature computed on all 12 months and on a test-like contiguous 4–6
month window drawn from the *measured* window distribution via the existing `_mask_views`; a feature
that does not survive truncation is disqualified regardless of its physics. That second gate is also a
candidate explanation for the ROCKET null (−0.009): a 12-month period is unidentifiable from a 5-month
window, which disqualifies the entire Fourier/harmonic family.

**The gate's first version failed its own control, and we are reporting that rather than the polished
table alone.** v1 used `median_over_months(VH − VV)` as the "exactly linear" control and it returned
R² = 0.6206, nowhere near the 1.0 arithmetic guarantees — because a **median is nonlinear** in the raw
values, so the row was measuring the median, not the difference. The honest control is the difference
at one fixed month (literally two of the 144 columns with coefficients +1 and −1), which returns
1.0000 / 1.0000 as it must. The general rule this bought: **if a gate's control does not return the
value arithmetic guarantees, every other number it prints is void.** We caught it before the numbers
entered this report; we would not have caught it without a control row.

| candidate | span R² | window ρ | univariate AUC |
|---|---|---|---|
| **CONTROL** VH−VV at one month | **1.0000** | **1.0000** | 0.6922 |
| VH median (Ottinger canonical) | 0.9003 | 0.9316 | 0.8338 |
| MNDWI median | 0.9307 | 0.9418 | 0.8960 |
| AWEI_nsh median (Feyisa et al., RSE 140:23–35, 2014) | 0.9086 | 0.9263 | 0.8796 |
| LASCI median | 0.7526 | 0.8992 | 0.8891 |
| SPCI median | 0.5520 | **0.6483** | 0.6131 |
| corr(VH, NIR), cross-band | 0.0474 | **0.5046** | 0.5024 |

⚠️ **A low span R² is not a go signal, and this instrument funds nothing.** It says a feature is
unreachable *linearly*, not that it helps; `univ AUC` is train-only, and train-only AUC has never
predicted transfer in this project. **The gate can only ever VETO.** LASCI is the single candidate
clearing both gates with real discriminability, and with two days left and two finalists locked on
measured scores we recorded it and did **not** build it.

---

### 8.7 🔑 We diagnosed the wrong half of our own error for seven iterations — and the leaderboard itself proved it

This is the most expensive mistake in the project, we found it on the last day, and the way it was
found is more useful than the mistake itself.

**The unexamined number.** Since iter42 we have inverted the published F1 column to recover our
public confusion matrix. `F1 = 2·TP/(PP+P)` pins the *sum* `PP+P` and `TP`, but it cannot say how
that sum splits between our predicted positives and the ground truth. To split it we needed the
public set size. We used **309** — which is exactly 30% of 1030. We never read it off the platform.
We inferred it, wrote it down, and it propagated into three documents and two tools for seven
iterations without one person asking where it came from.

**The test that caught it, which we should have been running from the start.** A reported score is
not a free real number. On a finite sample,

$$\mathrm{AUC} = \frac{C}{P\cdot N}, \qquad C \in \tfrac{1}{2}\mathbb{Z}$$

because under the Mann–Whitney convention each tied pair contributes exactly ½. So a leaderboard
that prints nine decimals is publishing a **rational with a known denominator**, and one can simply
ask whether the printed value is *reachable at all*. At `n = 309` the answer for our champion's
`AUC 0.945841814` is no: the nearest realisable value over every `P` is **1.9 × 10⁻⁷** away from a
**10⁻⁹** display window, and our own `P = 191` misses by 5.2 × 10⁻⁶. The trio was impossible.

The tell had been sitting in our own working notes in plain sight: we had computed a discordant-pair
count of **1220.6**. A pair count cannot end in `.6`. *If a count comes out fractional, the inputs
are wrong* — logged as error #7 in our ledger.

**Detection is not the interesting part; inversion is.** The same sieve that refutes 309 also solves
for the truth, because we hold **five** reported `(AUC, F1)` pairs that must all be satisfied at the
*same* `(n, P)`. That constraint cuts ~150 000 candidates to **15**. The full-test predicted-positive
counts sitting in our own submission files then finish the job: `PP_public/PP_full ≈ n/1030` gives
four independent estimates — 324.0, 326.4, 324.5, 331.6 — selecting `n = 333` over the runner-up by
70 rows to 6. **Zero submissions were spent.** Reproduce with `python tools/lb_cell_solve.py`
(exact `fractions` arithmetic; no floating point in any decision).

| public set | n | P | N | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|---|---|---|
| what we carried | 309 | 191 | 118 | 164 | 17 | 27 | 101 | 0.9061 | 0.8586 |
| **solved** | **333** | **181** | **152** | 164 | **27** | **17** | 125 | **0.8586** | **0.9061** |

**Precision and recall were swapped.** We believed we were *missing* 27 ponds. We are *inventing*
27 and missing 17 — the dominant error is false positives, 1.6 to 1, and we over-predict (realised
positive rate 0.5736 against a true 0.5435). Note this is robust to the one soft step: `TP` and `PP`
are invariant across every surviving candidate, so only `TN` depends on `n = 333`. The swap does not.

**What it cost, stated honestly.**

- §5.4's graph estimator "finally answered the prevalence question" at ~0.59 and we read its
  agreement with our operating rate as confirmation. It was agreeing with our *error*. An estimator
  that reproduces your bias is not a validation, and we had no independent check on it.
- The iter43 question *"is our operating positive rate too LOW?"* was left open. It closes in the
  **opposite** direction. Every lane motivated by pushing more rows above the cut was aimed away
  from the problem.
- Our final research round was organised around the "high-recall corner" — partial-AUC methods,
  Neyman–Pearson, and a JTT arm built to recover missed positives. All of it targeted 17 errors
  while 27 sat on the other side of the cut. The method verdicts stand (they rest on the
  order-invariance theorem, which is indifferent to which corner you aim at); the *target* moved.

**What survives untouched.** The iter49 JTT result is unaffected: it used only `PP+P`, which is
`P`-independent, so "TP identical at 164, one false positive removed, zero true positives
recovered" stands exactly as recorded. `tools/roc_probe.py` has been corrected to `P=181 / N=152`
and its guaranteed control value is now 0.864222885.

**The transferable lesson, and the reason this section exists.** *A leaderboard that prints enough
digits is an exact measuring instrument, not a noisy score.* Every reported number carries an
integrality constraint, and checking reachability costs nothing. We should have been sieving since
iter42. Instead we trusted a round number we had made up ourselves, and it silently inverted our
strategic reading of the problem for seven iterations. The credit for the catch belongs to an
outside reviewer of our own brief — the second time in this project (see §8.5) that the highest-value
contribution was someone refusing to accept one of our stated premises.

*Compliance note:* every quantity here is leaderboard-derived and is therefore **diagnosis only**
under the standing rule declared in §8.2. Nothing in this section sets a threshold, a
hyperparameter, or a model choice; the decision rule remains a literal 0.5.

---

## 9. What we would do with more time

Stated so a reviewer can see we know where the remaining value is, not as a claim of results.

0. **A graph-propagation teacher — the one lane we closed for time rather than for evidence.**
   Distillation from a pooled teacher was our single largest win (§6.6, +0.0100), but it is capped
   at one round because the teacher is the model's own prediction and re-teaching compounds error
   (Kumar, Ma & Liang). A teacher built from k-NN label propagation on a mask-matched similarity
   graph (§5.4) is **structurally independent of the network's own predictions**, so that cap does
   not obviously apply. We wrote a kill condition — *if its ranking correlates too highly with the
   current teacher's, it carries no new information* — and **the kill condition did not fire**:
   ρ = 0.79–0.88 against our teacher, versus ρ = 0.9686 *within* our own family, disagreeing on
   ~10% of all rows. It needs no new model code, since `seq.distill.teacher` already accepts a
   probability bundle.
   **Two honest reasons we did not spend the round.** Its decorrelation (ρ ≈ 0.79–0.88) sits in
   precisely the band where ROCKET (0.850) and the GBDT (0.849) both *lost* as pool members
   (−0.009, −0.0155), so it could only ever enter as a teacher, never as a member. And it is
   near-binary — only 0.5–2.9% of its mass lies within 0.05 of the cut — so soft distillation from
   it approaches hard pseudo-labelling on a shifted test set, the classic self-training failure mode.
   Both risks point down, and we had two days and an unwritten report. **This is an open lane, not a
   dead one**, and it is the first thing we would run with another week.

1. **Transductive training on the unlabeled test rows** — ✅ **DONE, and it worked — see §6.6.** The 1,030 test
   rows are 57% of our labeled set and the only target-domain data available; test *features* are
   supplied, so using them is legal by construction. Two zero-parameter forms: extending our proven
   cross-view invariance penalty to test rows (forms no label, so it cannot suffer confirmation
   bias), and soft self-distillation from a pooled teacher. Chen, Wei, Kumar & Ma (2020) prove
   self-training on unlabeled target data drives a classifier off features that correlate with the
   label in the source domain but not the target — which is a description of our failure mode.
2. **The dual-polarization water indicator.** Our entire feature bank is a function of **VH alone**,
   while every water detector in the remote-sensing literature is dual-polarization (Ottinger takes
   percentiles of VH *and* VV; Duan's SDWI is a function of both). We concluded the dual-pol axis was
   dead — but §7 shows we only ever tested it in forms that are *provably affine* in bands the model
   already sees, so those arms measured width cost, not information. The untested form is the
   **indicator**, which is exactly the nonlinearity behind our single biggest feature win:
   `c_t = 1[VH_dB(t) < −21] · 1[(VH−VV)(t) < τ_r]`, as a width-neutral *replacement* channel with
   `τ_r` chosen by the free adversarial/label-AUC screen. It is also all-SAR, so unlike an
   NDVI-gated version it is available in every observed month — S2 is absent in 17.6% of October
   test rows.
3. **LP-FT fine-tuning of Presto.** §7 records that lane as closed on evidence we later retracted.
   Linear-probe-then-fine-tune (Kumar et al., ICLR 2022) is the configuration that preserves
   pretrained features out-of-distribution, and it is exactly the one we never ran. Note
   `run_presto.py` still emits through the superseded illegal path and would need the 6-line
   compliance fix first.
4. **Per-window specialization.** We match the test window distribution by augmentation but train one
   model for all window lengths. A model conditioned on — or specialized to — window length is
   untested.
5. **Measure the two columns separately.** `TargetF1` (weight 0.6) is a *set-selection* problem;
   `TargetRAUC` (0.4) is a *global ranking* problem. We have fed both from one score vector by habit
   and have never measured them separately, despite the leaderboard reporting them separately.

---

## 10. Where everything lives

```
config/config.yaml              all hyperparameters, feature toggles, seed
src/
  data.py                       schema discovery, -9999 -> NaN cube, test-mask measurement
  features.py                   window masking (apply_mask), indices, aggregates
  validation.py                 masking-aware K-fold, leak-free OOF
  seq_model.py                  the submitted temporal Transformer (+ transductive terms)
  calibration.py                calibrate_legal(): train-only Platt, literal 0.5 cut
run_pipeline.py                 orchestrator; writes + validates every submission
tools/
  offline_validate.py           the LB-predicting screen (§5.1) and its falsifier
  adversarial_cv.py             the tree-lane gate (§5.2) and its falsifier
  shift_diagnostics.py          free adv-AUC / label-AUC feature screen
  shift_audit.py                adversarial probes + the 2-D band screen
  label_shift_gate.py           the mixture goodness-of-fit test that vetoed Saerens
  graph_gate.py                 the k-NN graph instrument (§5.4): connectivity, homophily,
                                parameter-free propagation, and the independent test-prevalence
                                estimate that answered a question MLLS/BBSE could no longer answer
  calib_family_gate.py          the calibrator-family gate (§8.5): tests whether a NON-AFFINE
                                monotone calibrator (beta, isotonic) buys anything at the fixed 0.5
                                cut. Reframes the question as a nested LR test, because beta contains
                                Platt exactly as its a==b submodel. Closed the lane at zero cost.
  feature_span_gate.py          the feature-span VETO (§8.6): cross-fitted ridge R2 of a candidate
                                feature on the 144 raw values (is it already inside the model's
                                reachable span?) plus a window-truncation stability check, since test
                                rows show only 4-6 contiguous months. Explains the VH-VV nulls
                                mechanically. It can only VETO -- a low R2 is not a go signal.
  arch_blend.py, seed_average.py   pooled artifacts
  regime_match.py               regime-matched calibration (§7): refits Platt on an OOF vector
                                whose averaging structure matches deployment. Its `--views all`
                                control reproduces seed_average.py bit-for-bit.
experiments/
  reproduce_champion.sh         ONE COMMAND REPRODUCTION of the designated finalists
  anchors.tsv                   known-LB anchors for the retro-fit (incl. the falsifier)
  LB_LOG.md                     every iteration, every score, every verdict
  results.tsv                   append-only run log
PROJECT_STATE.md                full state, ledger and lessons
```

**Cross-validation is masking-aware and leak-free:** folds are defined on the *original* rows, every
augmented view inherits its row's fold (a row's masked twins never straddle the split), and each
held-out row is scored on *R* independent masked views averaged into one OOF probability.

**One consequence of that last clause, which we found late and record against ourselves.** Averaging
*R* views makes the OOF vector a *better* estimate of each row's score — but it also makes it a
**differently-shaped** one than the vector it calibrates. A test row shows exactly **one** window
(and is averaged over `n_splits` fold-models); an OOF row is **R=2** windows from **one** model. Each
side is variance-shrunk on the axis the other is not, and Platt is fit across that mismatch. This was
harmless under the prevalence pin, because the cut was re-derived downstream and only the ranking
survived — but under a **literal 0.5 cut the Platt slope is the operating point.** `R` is read only
on the held-out path, so it is the one lever in this pipeline that can move the decision boundary
without touching any member's ranking; `tools/regime_match.py` exploits that to rebuild the
calibration set at R=1 offline, at zero extra training cost. See §7.

**A deliberate inversion a reviewer should expect:** our best-leaderboard models have our *lowest*
OOF. Local OOF sits near 0.975 against a leaderboard near 0.90, and has been **anti-correlated** with
it across the ledger. We never select on OOF. This is stated in `README.md` and is not an accident of
tuning — it is the covariate shift of §1 doing exactly what it was designed to do.
