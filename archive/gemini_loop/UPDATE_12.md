# Deep-Research Brief — Round #12 (Gemini Deep Research AND Claude Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)
**Date:** 2026-07-28 · **Leading finalist:** `champion_archblend4` **0.894643** · **reliable level ≈0.8865**
· **Deadline:** 2026-08-16 (19 days) · **Budget left:** ~79 of 100 submissions, 5/day

**Rounds 10–11 were run in-loop** (a cross-examination round, then a team of 8 internal research agents
across 8 literatures), so the last *external* brief you saw was **Round 09**. Since then we have spent
7 submissions, closed 4 more lanes, and — most importantly — **found that a premise Round 09 gave you
was incomplete in a way that opens the largest untouched lane in the project.**

Read §0 first. If you answer against the Round-09 framing you will spend the round on closed ground.

---

## 0. What changed since Round 09 — read this first

**1. 🔑 "The metric is rank-only" was RIGHT but INCOMPLETE — and the missing half is the biggest
unexploited lever we have.** Round 09 told you the metric is a functional of the test ranking. True. But
we then treated the submission as *one* ranking. It is not. There are **two independently scored
columns**, and we verified in our own code (`run_pipeline.py:160-161`) that they are computed by two
separate functions writing two separate columns:

| column | weight | what it actually is | what optimizes it |
|---|---|---|---|
| `TargetF1` | **0.6** | a **SET** — which ~668 of 1030 rows are called positive | precision@k — *local order at the cut only* |
| `TargetRAUC` | **0.4** | a **RANKING** of all 1030 rows | global AUC |

**These are a set-selection problem and a ranking problem, and they may be served by different models.**
We currently feed both columns from the same score vector — **by habit, not by necessity.** The
0.6-weighted column is blind to all reordering above and below the cut. We have never exploited this.
(Source: competition forum, user `sdv`, currently in the 90s: *"optimize the two scored columns
independently."*)

**2. The whole F1 lever is ~11 rows — and on the public slice, ~3.** With k = ⌊0.649·1030⌋ = 668 and
P ≈ 669, `F1 = 2·TP/1337`. Moving the total score by our 0.010 measurement floor via F1 alone needs
**ΔTP ≈ 11 rows**; via AUC alone it needs **ΔAUC ≈ 0.025**, a very large move. The public LB is only
~309 rows, so the same 0.010 there is **≈3 rows crossing the cut.** This retro-explains both of our
blend losses (−0.009, −0.016 ≈ 6 and 11 net rows the wrong way) and it means **the public LB is an even
noisier instrument than Round 09 assumed.**

**3. Round 09's Q4 is ANSWERED, and the answer is NO — with a mechanism.** We asked whether pooling
*decorrelated equally-good* models buys **level**. We then ran it twice, across two foreign model
classes, and both lost, monotonically in the member's own level deficit:

| blend | member | member ρ to champion | member's own level | blend LB | Δ vs archblend4 |
|---|---|---|---|---|---|
| `champion_archblend4` | 4 transformers (same class) | ρ̄ 0.9395 | all ≈champion | **0.894643** | — |
| `champion_rocketblend5` | ⅕ ROCKET | **0.87** | ≈−0.040 | 0.885661 | −0.009 |
| `champion_gbdtblend5` | ⅕ GBDT | **0.873** | ≈−0.011 (est.) | 0.879123 | **−0.0155** |

The gbdt loss is **paired and significant** (4/5 members shared, identical 309 rows, ≥2.5σ). **Corrected
law:** under a *pinned threshold*, there is no ambiguity decomposition for 0-1 loss — the standard
error-ambiguity result that justifies diversity **does not apply**. The cut acts as a filter that
selects specifically for *bad* diversity: a decorrelated member's disagreements land on the rows nearest
the boundary, which are exactly the rows that flip. **Gate ensemble members on LEVEL GAP, not on
correlation.** Cross-model-class blending is CLOSED (n=2, both directions).

**4. The 0.98 target was never real, and the physics says the published ceiling is not ours either.**
Forum scores of 0.953 / "0.98+" were posted **before the 25 Jun 2026 data reset** — earned on the
**leaked** data. The live post-reset competitive band is **≈0.90–0.95**. Independently, the pond-mapping
literature stands on three legs and **two are unavailable to us**: (i) pixel-wise temporal permanence —
✅ the only one we have, and we already found it; (ii) **shape** (compactness, perimeter, LSI, dike
detection, GLCM texture) — ❌ lat/lon stripped, isolated pixels, no neighbourhood, and Ottinger (2022)
calls compact shape the ***defining*** feature separating managed ponds from natural water; (iii) DEM /
OSM / JRC surface-water overlays — ❌ external data, rule-barred. Phan et al. ground-truthed that flooded
rice at ~10 days post-sowing reads **VV −13 dB, VH −22 dB — open-water values**. At any single date our
positives and our hardest negatives are radiometrically identical. **Published 89–95% accuracies are
earned with the two legs we do not have.** Do not aim us at them.

**5. Two lanes closed for ZERO submissions by a local audit — including the one 6 of 8 agents ranked
highest.** See §2 (F) and (G). One of them (**window matching**) turned out to be *already fully
implemented in our repo*. Please do not spend the round re-recommending it.

**6. 🔑 The #1 on the public LB has described their approach on the forum — and it says we have been
optimizing the wrong axis.** User `sdv` (thread dated 14–16 Jul 2026, post-reset), verbatim:

> *"Random K-fold basically lies to you here — train and test are different time periods, so
> in-distribution CV looks amazing and means nothing. **I tried validation that mimics the test regime
> and even that barely correlated.** In the end I treat the LB as the only honest judge."*
>
> *"Prefer signals that don't drift between periods — **relative/ratio-style features tend to survive
> the shift far better than absolute values.**"*
>
> *"Think about what physically separates the target from the things it gets confused with, and try to
> **isolate that one specific signal** rather than throwing everything at a big model."*
>
> *"The two scored columns are independent — worth optimizing each on its own terms."*
>
> *"**Don't blame the trees — I'm on gradient boosting too (CatBoost-family, nothing exotic), so the
> model isn't the bottleneck.** If tuning models and swapping features isn't moving the LB, that's a
> hint the lever is elsewhere."*
>
> *"Months aren't independent points — training each month as its own row loses how a pixel behaves
> across the year."*

**Three consequences, and they are uncomfortable for us.** (i) The leader is at ~0.94–0.945 with
**plain CatBoost**, while our GBDT scored ≈0.885 and our transformer ≈0.8865 — so the ≈0.05 gap is
**entirely in the features/representation**, not the model class. Our last 8 iterations of architecture
and ensemble search were searching an axis the leader says is flat. (ii) Regime-matched validation was
**tried by the leader and barely correlated** — see Q3, which we have rewritten accordingly. (iii)
**Ratio-style features** are the leader's named lever and we have **never tested a cross-band ratio.**
Note this does *not* contradict our amplitude finding (§2 D): we tested replacing values with
*within-series temporal rank*, which destroys level. A **cross-band** ratio at fixed time preserves
level information while cancelling per-period gain/calibration drift. **These are different
transformations and we conflated them.**

**Consequence for you:** the funding bar from Round 09 stands — nothing below **~0.010 LB** is
measurable in principle — but the *shape* of a fundable idea has changed. We are no longer shopping for
a better single ranking. We are shopping for **(a)** ways to exploit the two-column split, **(b)** ways
to move ~11 specific rows across a fixed cut, and **(c)** a defensible rule-compliant operating point.

---

## 1. Where we are — the ledger since Round 09

Every row isolated against the then-champion, operating point pinned at realized positive rate 0.649.
**Seed sd = 0.0191, measured.** Treat any |Δ| < ~0.02 as UNRESOLVED, not a verdict.

| # | Change | LB | Verdict |
|---|---|---|---|
| — | GBDT → from-scratch Transformer | **0.8780** | ✅ **REAL** (+0.052, ≫ noise) |
| 3 | `per_cell_detrend` amplitude channels | 0.8266 | ✅ **REAL** (−0.051, ≫ noise) |
| 9 | cross-view invariance λ=1.0 (**champion**, seed 42) | **0.8955** | ⚠️ inside noise; **known lucky draw** |
| 15 | **champion at seed 7, config identical** | **0.8764** | 🚨 sd 0.0191 — voids 9 of 11 verdicts |
| 16 | seed-averaged champion, 5 seeds | 0.8865 | ✅ variance only, no level (predicted 0.886) |
| 17 | **Presto** frozen foundation encoder | *screened, held* | ❌ lane dead — see §3 |
| 18 | **cross-architecture blend** (reltime/nope/l3/xview) | **0.894643** | ✅ **leading finalist** |
| 19 | dispersion pooling (mean⊕min / mean⊕std / moments) | 0.898566 | ➖ within noise; ρ=0.9928 = rank-twin |
| 20 | mean_min as ensemble member | *not submitted* | ❌ not decorrelated, lane closed |
| 21 | instance-expansion (per-epoch view resampling) | *not submitted* | ❌ inert; **screen gate VOID** |
| 22 | **ROCKET member** (random conv kernels) | 0.885661 | 🎯 first ρ<0.90 member, but −0.009 |
| 23 | multivariate ROCKET (random band subsets) | *not submitted* | ❌ strength ⊥ diversity inside the family |
| 24 | **GBDT member** (trees on temporal aggregates) | **0.879123** | ❌ −0.0155 **paired, significant** → §0 #3 |
| 25 | **Phase-A shift audit** (local, free) | *n/a* | ✅ 1 lane closed, 1 opened — §2 (F)(G) |

**Only two effects in the entire project exceed the noise floor, and both are model-class changes.** All
of iterations 18–25 — every ensemble construction, pooling variant, model class and expansion scheme —
produced nothing above the floor.

**Field (public):** top ≈0.9452 · top-5 ≈0.928–0.945 · rank-50 ≈0.876. Our reliable ≈0.8865. **Gap to
top-5 ≈ +0.04** — the size of the GBDT→Transformer jump.

**Final artifact board** (all submitted): `c_meanmin` 0.898566 (single seed, ⚠️ lucky) · `seq_a_xview`
0.8955 (⚠️ known lucky draw) · **`champion_archblend4` 0.894643 (lowest variance — finalist #1,
settled)** · `champion_seedavg5` 0.886530 · `champion_rocketblend5` 0.885661 · `champion_gbdtblend5`
0.879123.

---

## 2. The findings you must internalize (challenge them, but with evidence)

**(A) Only model-class changes clear the noise floor.** An idea worth a submission must be plausibly
≥0.02, or it must be screenable offline for free.

**(B) The two columns are independent** (§0 #1). This *supersedes* the Round-09 "rank-only" framing.

**(C) Ensemble diversity is a LIABILITY at a pinned threshold** (§0 #3). Gate on level gap. Always
rank-average, never probability-average.

**(D) Amplitude IS the primary signal.** Replacing raw values with within-series **rank** (true
amplitude removal, capacity-neutral) collapsed OOF 0.975→0.86. **Persistently-low backscatter level is
the class signal.** Do not propose instance-norm, detrend, differencing, or per-window value
standardization. Very high bar.

**(E) The shift is real, large, and lives in the VALUES — not in the missingness.** Measured this week,
masked-train vs test:

```
values only .................... adv-AUC 0.8915
ALL missing-indicators only .... adv-AUC 0.4758   <- BELOW CHANCE
S2-cloud indicators only ....... adv-AUC 0.4744
gap-count channel only ......... adv-AUC 0.4815
values + indicators ............ adv-AUC 0.8943   (indicators ADD +0.0028)
```
For reference: ≈0.99 on hand features, 0.965–0.976 on frozen Presto embeddings of raw pixels. So our
masking + left-alignment already removed a real chunk, and a large signal-side component remains.

**(F) 🔴 The missing-indicator deletion lane is CLOSED — three agents predicted the opposite.** They
proposed a good mechanism: we deliberately deleted absolute time by left-aligning, but cloud-gap
patterns encode season, so the model could have recovered the month-of-year that our single biggest win
removed. **Measured: below chance.** The reason is in our own code — `apply_mask` (`src/features.py:85-92`)
already applies S2 dropout at rates *measured off the test set*, so the train indicator distribution was
matched by construction. Do not re-propose indicator deletion.

**(G) 🔴 Window / regime matching is ALREADY IMPLEMENTED here.** Six of eight agents named it the single
highest-value item in the round ("if you take one thing, take that"). They could not have known. Our
pipeline expands each train row into K masked views drawn from the *measured* test window-length and
per-month dropout distributions (`apply_mask` + `measure_window_dist` + `match_test_distribution: true`).
The LANL-Earthquake-style "resample train to look like test" fix is done. **Do not re-propose it.**
What is *not* done is applying it to a **model-selection CV** — see Q3.

**(H) 🟡 The shift is DISTRIBUTED, so feature deletion cannot collapse it.** Per-band 2-D screen
(A = separates train/test, T = predicts label):

| band | A | T | read |
|---|---|---|---|
| **VV** | **0.5907** | 0.7801 | top shift-carrier; VH dominates it on signal → free deletion |
| VH | 0.5622 | **0.8302** | REPAIR, never delete — this is the primary signal |
| **blue** | **0.5344** | 0.5963 | barely predictive, most Rayleigh-scattered → free deletion |
| medians | 0.5179 | 0.7802 | — |

**Max single-band A = 0.59 against a joint 0.89.** No small subset carries the shift. A band-deletion
screen (`c_dropvv` / `c_dropblue` / `c_dropvvblue`) is queued at 0 submissions; we do not expect it to
clear. Chasing adversarial AUC → 0.5 by feature selection is futile here.

**(I) Our own offline screen is coarser than we treated it.** ATC-F1 retro-fits at Spearman ρ **+0.964**
— but at n=7 anchors the Fisher-z 95% CI is **[0.770, 0.995]**, and its magnitude is ~3× overstated.
Trust its **sign**; treat DIS (ρ+1.000, n=4) as a second vote only. Its resolution is ~0.010–0.013 LB.
It has returned **VOID** on its own permutation-null gate once (iter21).

---

## 3. Do NOT re-propose (spent, refuted, or already built)

Everything on the Round-09 list still stands, **plus**:

1. **Window / regime matching** — already implemented (§2 G).
2. **Missing-indicator deletion** — measured below chance (§2 F). Also not iter13's `compact_missing`
   (24→14), which failed −0.0053/−0.0252.
3. **ROCKET / MiniROCKET / MultiROCKET, univariate and multivariate** — built and screened. Decorrelated
   (ρ0.87) but −0.040 weak; strength and diversity trade off *inside* the family (iter22–23).
4. **GBDT / tree ensembles as a blend member** — built (LGBM+XGB+CatBoost on temporal aggregates),
   ρ0.873, blend **−0.0155 paired significant** (iter24).
5. **Cross-model-class blending in general** — closed at n=2, both directions, with a mechanism (§0 #3).
6. **Seed averaging, SWA, EMA, snapshot ensembles as *climbers*** — variance only, confirmed to 0.0006.
7. **Presto** frozen encoder + logistic head — adv-AUC 0.965–0.976 on its own embeddings; it *encodes*
   the shift. (Other foundation models were Round-09 Q2 and remain formally open, but see §0 #4 — we now
   doubt any pixel-TS encoder helps, and none is worth a submission without a free go/no-go.)
8. **Adversarial-AUC-driven feature selection targeting ≈0.5** — the shift is distributed (§2 H).
9. **Instance-expansion / per-epoch view resampling** — inert, behaves like the failed K=4.
10. **Dispersion pooling** (mean⊕min, mean⊕std, moments) — within noise, ρ0.9928 rank-twins.
11. **Still-valid Round-06/09 refutations:** amplitude normalization · Saerens–EM / BBSE / MLLS
    (ours is covariate shift; BBSE gave 0.44 vs true 0.649) · importance-weighting the training loss
    (ESS collapse at adv-AUC 0.99) · DANN · WIF/EVI/SDWI water indices (−0.075) · temperature scaling ·
    TTA (−0.0023) · group-KFold "leakage" framing (the shift is designed).

**Also algebraically dead, verified:** SDWI is exactly affine in (VV_dB+VH_dB); AWEI is exactly linear;
EVI ≈ 2.5(NIR−Red) over water; NDWI/MNDWI are 0/0-conditioned over water. Do not propose these indices
— a linear model already spans them.

---

## 4. Self-contained problem statement (assume no repo access)

**Task.** Binary classification: is a given ~10 m cell an **aquaculture pond**?

**Data (supplied only).** Train **1,821** rows (1,817 after dropping 4 exact duplicates), test **1,030**
(public ≈309 = 30%, private ≈721 = 70%). Per cell: a **12-month × 12-band** time series — Sentinel-1 SAR
(**VH, VV**, dB) + 10 Sentinel-2 optical bands. **No lat/lon, no spatial neighbourhood, no image
patches, no static covariates.** Each row is one isolated pixel-cell's time series.

**Metric.** `0.6·F1 + 0.4·ROC-AUC` over **two independent columns**: `TargetF1` (binary, **hard 0.5
cut**) and `TargetRAUC` (any rank-preserving score).

**The designed trap — temporal masking.** Train rows are fully observed (12 months). Test rows expose
only a consecutive **4/5/6-month window** (p ≈ 0.335/0.333/0.332), rest sentinel −9999, plus extra
S2-only cloud dropout inside the window (measured per-month rates 0.003–0.28; Oct ~17.6%, Jun ~7.3%,
Feb ~3.7%). **320 of 12,360 test row-months have S1 present but ALL S2 bands missing**; some test rows
have as few as **2 usable optical months**. We expand each train row into **K=2 masked views** sampled
from these measured test distributions.

**The domain shift.** Train and test are different time periods and pilot regions, **by design**.
Adversarial train-vs-test AUC ≈0.99 (hand features), ≈0.97 (foundation-model embeddings), **0.8915
(masked, left-aligned values — our actual input)**. Genuine covariate shift, proven leak-free.

**Constraints.** Supplied data only — **no external data, rasters, or lookups**. **Pretrained models
allowed** (openly available). AutoML banned. Open-source, seeded, reproducible. 5 submissions/day, 100
total. Final score = **65% private LB + 35% code/reproducibility review of the top 5.**

**Champion architecture (0.8955 single-seed / ≈0.8865 reliable), exactly — from scratch:**
- Per-month input: **12 standardized band values ⊕ 12 binary missing-indicators = 24 channels** → Linear
  to d_model = 64.
- **Relative-time reframing:** observed window left-aligned to t_rel=0 *before* a learned length-12
  positional embedding. (This was our single biggest capacity-neutral win, +0.0128.)
- **2-layer** encoder, 4 heads, GELU, dropout 0.2, `src_key_padding_mask` over fully-missing months.
- **Masked mean-pool** → MLP head → sigmoid.
- **Loss:** `BCE + λ·Var_k(logit)` across K=2 masked views, λ=1.0, owner-grouped batching.
- AdamW lr 1e-3, wd 1e-4, batch 256, 60 epochs, **5-fold CV, n_repeats=1**; test = mean of 5 fold-models.
- **Operating point:** exact-prevalence monotone logit shift → realized test positive rate **0.649** (raw
  0.553). `TargetRAUC` = the untouched ranking.

---

## 5. Research questions — the substance of the round

Prioritize by **expected LB gain × feasibility**. For each: mechanism; the math where non-obvious; why
it is legal (§4); why it is not a §3 repeat; the **single isolated experiment** vs champion; **plausible
effect size**; and **how we would screen it offline** (our screen has ~0.010 LB resolution; adversarial
AUC on any representation is free). **We would rather be corrected than agreed with.**

### Q1 — 🔑 (HIGHEST VALUE) Exploiting the two-column split
`TargetF1` (0.6) is a **set-selection** problem at a fixed budget of k≈668; `TargetRAUC` (0.4) is a
**global ranking** problem. We have always fed both from one score vector. Address concretely:
- **What objective actually maximizes precision@k when k is fixed and known in advance**, as opposed to
  AUC? Give the estimator/loss — e.g. **p-norm push, DCG/lambda-style top-heavy ranking losses,
  Precision@k surrogates (Boyd/Kar), or an ROC-region-restricted (partial-AUC) objective** targeted at
  exactly the FPR region containing rank 668. Which of these has evidence under covariate shift?
- **Should the two columns come from different models?** e.g. a high-recall, high-variance model for the
  set, and a low-variance rank-averaged pool for AUC. Is there a principled way to choose each — and is
  there any *risk* we are not seeing (the columns are scored on the same rows, so errors may covary)?
- **Where should the marginal modelling effort go**, given the 0.6-weighted column depends only on the
  *local order in the band straddling rank 668* (~11 rows for a 0.010 move) and is completely blind to
  reordering elsewhere?
- Is there a **selective-classification / conformal** framing (abstention, risk-controlling prediction
  sets) that produces a better *set* at fixed cardinality than thresholding a probability?

### Q2 — Moving ~11 specific rows: the boundary region under covariate shift
Given §0 #2, our entire F1 lever is the ordering of a few dozen rows near the cut, and those are by
construction the *least confident* rows — the ones where the shift bites hardest.
- Is there literature on **improving local ordering in a narrow score band** (as opposed to global
  calibration or global ranking)? **Boundary-focused / hard-example / margin-based reweighting** that
  does *not* add fitted capacity on 1,817 rows?
- **Transductive** methods that use the unlabeled 1,030 test rows to sharpen the boundary specifically —
  and are defensible in a code review. Pseudo-labeling was dismissed on ESS-collapse grounds, but we now
  have a rank-only metric and a free screen. Is **CBST/CRST** class-balanced self-training, or a
  confidence-*and*-diversity criterion, safe when the confident test rows may themselves be the shifted
  ones? Give the exact protocol and the fold-safety argument.
- What is the **variance** of a boundary-region intervention? If it moves 11 rows in expectation with an
  sd of 11 rows, it is not fundable — say so.

### Q3 — Model selection when even the *matched* CV is known to fail
We already match the observation regime in training (§2 G), and the **public LB leader reports that a
regime-mimicking validation "barely correlated"** for them either (§0 #6). So please do **not** answer
"build a regime-matched CV" — that is the obvious answer and the strongest available evidence says it
does not work. Our CV sits at ~0.975 against an LB of ~0.89 and has been **anti-correlated**.
- Given that, is there **any** train-only validation that can rank candidates here, or is the honest
  answer that *no* train-only CV works when the period/region shift leaves adv-AUC 0.8915 on the values
  alone *after* regime matching? **If the answer is "no", say so** — it would settle our biggest open
  methodological question and redirect the remaining 19 days.
- Are there **label-free** model-selection criteria with real evidence under covariate shift — ATC and
  its variants, importance-weighted CV where ESS permits, agreement-on-the-line / **prediction-based
  "GDE"** methods, or the **DEV/DEV-k** estimators? We built ATC-F1 (ρ+0.964, n=7, CI [0.770, 0.995]).
  **What would raise its resolution below 0.010 LB?** That is worth more to us than any single feature.
- Is there any principled way to use the **public LB itself** as 309 labeled-by-proxy rows without
  overfitting it? (We have ~79 submissions and a 3-row-per-0.010 sensitivity, so we are sceptical.)

### Q4 — ⚠️ Rule-compliant operating point (this is 35% of our score, not a technicality)
Forum thread 33912: participants state the rules say **threshold tuning is strictly forbidden** and
`TargetF1` must use the default 0.5. **The organizers have not answered**, and two participants have
publicly said they are self-limiting because of it. **Our prevalence pin (a monotone logit shift so the
realized positive rate lands at 0.649) is functionally threshold tuning.** It is worth ≈+0.07 to us and
we cannot simply drop it.
- What is the **defensible construction that reaches the same operating point**? Our plan:
  class-weighted / balanced-objective training so a literal 0.5 lands near the target prevalence, plus
  **Platt or isotonic calibration fit on training data only**, then a literal 0.5 cut. Is that sound, and
  is it genuinely distinguishable from threshold tuning — or is it the same thing with better
  paperwork? We want the honest answer, including if it is the latter.
- **How is prevalence estimated legitimately** when test prevalence is unknown and the shift is
  covariate, not label? (BBSE gave 0.44 vs a true ≈0.649 — we do not trust label-shift estimators here.)
- Precedent: how have organizers and top finishers in comparable competitions treated the line between
  **prior/prevalence correction** (usually allowed) and **threshold tuning** (often banned)?

### Q5 — What did top-5 finishers actually do — in THIS and closely-analogous competitions?
Primary sources, not blog summaries. Mechanisms specific enough to implement.
- **This competition:** FAO/ITU/Zindi challenge materials, the discussion forum, any public write-ups.
  Especially: how the train/test split and the masking recipe were *constructed*, and anything about the
  **25 Jun 2026 data reset** (a leak: new train = old train + old test *with labels*; new test issued;
  lat/lon stripped). **Note that any score posted before 25 Jun is on leaked data and is not a target.**
- **Analogous Sentinel-1/2 pixel-time-series competitions with a designed distribution shift** (Kaggle /
  Zindi / DrivenData): what did the top-5 do that the middle of the field did not? We are specifically
  interested in cases where the winner's edge was **operating-point discipline or validation design**
  rather than architecture — that matches our situation.

### Q6 — 🔑 Drift-invariant RATIO features (the leader's named lever, and our blind spot)
The public-LB leader says *"relative/ratio-style features tend to survive the shift far better than
absolute values"* (§0 #6), and reaches ~0.94 with plain CatBoost on them. **We have never tested a
cross-band ratio.** Our one "relative" experiment replaced values with *within-series temporal rank*,
which destroys level and collapsed OOF 0.975→0.86 (§2 D) — a different transformation entirely.
- **Which ratios are physically drift-invariant for S1/S2 over water?** A cross-band ratio at fixed time
  cancels multiplicative per-period gain (sensor calibration drift, incidence-angle and atmospheric
  scaling) while *preserving* the level contrast that is our actual signal. In dB, `VH − VV` **is** the
  log cross-pol ratio and is the obvious first candidate; we have it queued and have never run it.
- Give us a **ranked, sourced shortlist** of ratio/relative constructions with a stated invariance
  argument for each — what physical nuisance does it cancel, and what does it cost in signal? Please
  exclude the ones we have proven algebraically degenerate here (§3): SDWI, AWEI, EVI, NDWI, MNDWI.
- **Ratio-of-what-to-what:** cross-band at fixed t · same-band across t (drift-sensitive, likely wrong)
  · band-to-scene-median · band-to-its-own-window-median. Which survives a *period* shift rather than
  merely a within-scene one?
- Ratios are unstable near zero and S2 reflectances over water are small and can be negative after
  atmospheric correction. **Give the numerically safe form** (log-ratio, normalized difference,
  clipped denominator) and say which you would use here.
- **Screening:** every candidate ratio can be scored for free on our 2-D screen (A = adversarial
  separability, T = target predictiveness). A ratio that is genuinely drift-invariant should show
  **lower A than either of its constituent bands at comparable T** — that is a falsifiable prediction
  we can run at zero cost, so state it explicitly for each proposal.

### Q7 — The one leg of the physics we still have
We have **only** temporal permanence (§0 #4): shape and external overlays are gone. Within a single
isolated pixel's 12-band × 4–6-month series:
- What **shape-free, amplitude-preserving** temporal signature separates a managed pond from natural
  standing water, a rice paddy, or a salt pan? Candidates we have *not* tested:
  `max_t[(VH−VV)(t+1) − (VH−VV)(t)]` (abrupt drain/harvest), fraction-of-months-below-τ, and
  `median_t(B5/B4)` (red-edge eutrophication from feeding).
- ⚠️ **Geography caveat, stated honestly:** the pond-mapping literature is overwhelmingly coastal
  East/SE Asian *intensive* aquaculture (Mekong, Pearl, Red River, Jiangsu). The FAO/ITU framing suggests
  our data may be **African smallholder** — smaller ponds, less intensive feeding, more rain-fed
  drawdown — which weakens both the eutrophication signal and the "permanently full" assumption. **We
  found no quantitative African pond-mapping study.** If one exists, that alone would justify the round.
- Any such feature must be **capacity-neutral and channel-REPLACING**, not additive (§2 D), and must be
  **n-invariant**: our test rows have 4–6 months and our train rows 12. Class-A statistics (mean, median,
  interior quantiles, std, any *fraction*) are unbiased at every n; Class-B statistics (min, max, range,
  run-lengths, raw counts, autocorrelations) are **n-dependent and therefore shift-carriers by
  construction**. State which class your proposal is in.

---

## 6. Output format we want

For each recommendation:
1. **Name + one-line mechanism.**
2. **Why it is legal** (§4) and **why it is not a §3 repeat.**
3. **The math / the exact change**, implementable. ("Add a consistency penalty" was only actionable
   because a prior round wrote `L = BCE + λ·Var_k(logit)`.)
4. **Which column it targets** — `TargetF1` (the set), `TargetRAUC` (the ranking), or both. This is new
   this round and we want it stated explicitly.
5. **Expected effect size.** If < ~0.02, state exactly **how we would screen it offline**. If you cannot
   name a screen, say so — that is useful triage.
6. **The single isolated experiment** vs the champion, and whether it needs a submission or is free.
7. **Sources.** Primary literature and the actual challenge materials over summaries.

Rank everything into: **fund now** (worth a submission this week) · **screen first** (real, but gate it
offline) · **park** (interesting, wrong stage) · **rejected on our evidence.**

**A note on calibration of confidence.** Last round, 6 of 8 agents' top recommendation was already
implemented in our repo, and 3 of 8 converged on a deletion target that measured *below chance*. Strong
convergence between agents is not evidence — it usually means they read the same literature. **If your
answer to a question is "we cannot beat this from the data you have," say that.** Given §0 #4, that is a
live possibility for several of these, and knowing it is worth more to us than a plausible-sounding lane
that costs us a week.
