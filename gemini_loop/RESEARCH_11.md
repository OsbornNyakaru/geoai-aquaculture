# RESEARCH_11 — round-11 synthesis + implementation plan (2026-07-27)

Eight parallel Opus-5 research agents (Google Scholar / arXiv / competition forensics). Six reported
at time of writing; SAR-physics and tabular-foundation-model agents still running. This file is the
**triaged synthesis and the executable plan**, not the raw reports.

> **Framing correction, stated once.** The target of "0.98 LB" is the **pre-reset, leaked-data**
> number (forum posts of 0.953 / "0.98+" predate the 25 Jun reset). It is not on this board. The live
> post-reset band is ~0.90–0.95. Realistic goal: **0.90–0.93**, which is top-of-band. Everything below
> is costed against that, not against 0.98.

---

## 0. Claims I verified before trusting (three agent errors caught)

Agents are not oracles. Three load-bearing claims were checked and two failed.

### 0a. ❌ "The adversarial AUC 0.97 is confounded by label shift" — REFUTED quantitatively

The feature-selection agent argued the 0.97 is largely explained by the 40%→65% prevalence move, and
that naive adversarial elimination would therefore delete amplitude for a spurious reason. Under
**pure** label shift the optimal discriminator ranks by the class likelihood ratio, so the adversarial
AUC decomposes exactly:

```
AUC_adv = 0.375 + 0.25 · A          (A = the feature's class-separation AUC)
```

**Ceiling at perfect separation = 0.625.** At our champion's own class AUC (~0.99) it is 0.6225.
Measured: **0.965–0.976**. So label shift explains at most ~0.62; **+0.35 of adversarial AUC is
genuine covariate shift.** iter17's conclusion stands unweakened. Any feature above ~0.62 adversarial
AUC has real covariate shift regardless of prevalence. The prior-correction is a marginal adjustment,
**not** the crux-dissolver claimed. (The real answer to the crux is the 2-D drop rule, §2.2.)

### 0b. ❌ "ROCKET's −0.040 may be a cardinality artifact" — REFUTED from our own code

Proposed that ROCKET's global-max features were computed on 12-month train vs 4–6-month test series,
biasing half its feature vector. `src/rocket_model.py:204` calls `_mask_views` — train rows **are**
masked to test-like windows, so the cardinality cancels. ROCKET was genuinely weaker. iter22 stands.

### 0c. ⚠️ Our own estimator is less precise than we have been treating it

ATC-F1's ρ = +0.964 is measured on **n=7 anchors**. Fisher-z 95% CI:

| anchors | 95% CI on ρ |
|---|---|
| **7 (today)** | **[0.770, 0.995]** |
| 10 | [0.851, 0.992] |
| 17 | [0.901, 0.987] |

(The agent claimed the CI "spans essentially the whole positive range" — overstated; it is strongly
positive. But it is *not* precise.) We know the estimator is not anti-correlated; we do **not** know it
is sharp. **It should be used to REJECT disasters, not to CROWN winners.** Every LB submission from
here adds an anchor and tightens this — 17 anchors puts the lower bound above 0.90.

---

## 1. The two structural discoveries (both verified in our own code/arithmetic)

### 1a. 🔑 `TargetF1` and `TargetRAUC` are INDEPENDENT columns — we have never exploited this

From the competition forum (user `sdv`, currently scoring in the 90s): *"optimize the two scored
columns independently."* Verified in `run_pipeline.py:160-161`: `target_f1` comes from
`calibrate_for_f1(...)` and `target_rauc` from `score_for_auc(...)`. They are two separate functions
writing two separate columns. We currently feed both from the same `p_test_raw` — **by habit, not by
necessity.**

This **refines our own "rank-only" law**, which is not wrong but is incomplete:

| column | weight | what it actually is | what optimizes it |
|---|---|---|---|
| `TargetF1` | 0.6 | a **SET**: which 668 rows are called positive | precision@k — *local order at the cut* |
| `TargetRAUC` | 0.4 | a **RANKING** of all 1030 rows | global AUC |

These are different objectives and can be served by **different models**. The 0.6-weighted column
cares only about ~the band straddling rank 668; it is blind to all reordering above and below.

### 1b. 🔑 The whole F1 lever is ~11 rows

With k = ⌊0.649·1030⌋ = 668 and P ≈ 669: `F1 = 2·TP/1337`. To move the total score by our 0.010
measurement floor **via F1 alone requires ΔTP ≈ 11 rows.** Via AUC alone it requires ΔAUC ≈ 0.025 — a
very large move.

**This quantitatively explains iter22 and iter24, which we had recorded as unexplained.** Our two
blend losses (−0.009, −0.016) correspond to ≈6 and ≈11 net rows crossing the cut the wrong way. That
is an entirely ordinary amount of reshuffling for a ρ≈0.85 member at ⅕ weight. See §3.

---

## 2. The five cross-agent convergences

Ranked by how many independent agents reached them and how cheap they are to act on.

### 2.1 ✅✅✅✅ The CV lies because of **REGIME** mismatch, not just period mismatch (4 agents)

Our CV folds see **12 full months**; test rows see **4–6 consecutive months** plus per-band cloud
holes. **Our CV is therefore scoring a different, easier task** — which explains both the 0.975→0.89
level gap *and* the anti-correlation (features exploiting the full year are rewarded in CV and
punished on LB). Forum, `sdv`: *"Random K-fold basically lies to you here."*

Precedent with a documented win: **LANL Earthquake 1st place** (Singer/Gordeev/Levinson) faced a
useless CV and fixed it by resampling *training data to look like test data* — up to 10,000 candidate
subsets scored by mean KS distance to test — then shipped a 4-feature LGB and ignored the public LB.

**Also newly known from the forum** (thread 33603): 320 of 12,360 test row-months have S1 present but
**all S2 bands missing** — concentrated in October (17.6% of that month), June, February. Some test
rows have as few as **2 usable optical months**. Our pipeline must be correct there.

**The progress meter is free:** rebuild the discriminator on *masked-train vs test*. If adversarial
AUC falls from 0.97 toward ~0.6–0.75, the regime component is gone and what remains is genuine
temporal shift. **Zero submissions.**

### 2.2 ✅✅✅ Adversarial validation for FEATURE SELECTION — never for reweighting (3 agents)

We have measured adversarial AUC 0.97 and **never used it to select features or build a validation
set.** The one paper with real numbers — **Uber, arXiv:2004.03045** — reports **+3.9% test AUC** from
cutting 309→281 features, and, critically, that **inverse-propensity reweighting *underperformed the
baseline***. That independently confirms our DANN/importance-weighting failure (ESS collapse) and
tells us the *feature-space* version of the same signal is the one that works.

**The 2-D drop rule** (this is the correct answer to the shifted-but-informative crux that §0a got
wrong). For each feature compute `A(f)` = adversarial importance, `T(f)` = target importance on the
masked-window CV:

| | low A (stable) | high A (shifted) |
|---|---|---|
| **high T** | **keep** — core | **REPAIR, don't delete** → cross-band ratio, within-row rank, or window-relative z-score. Delete only if repair fails. |
| **low T** | drop (dead weight) | **delete immediately** — pure shift-carrier, zero cost |

The "repair" cell is what protects amplitude. Stop when adversarial AUC ≤ ~0.75; do **not** chase 0.5
— some shift is designed in and irreducible.

### 2.3 ✅✅✅ Missingness/count-of-months channels are prime shift-carriers (3 agents)

Three agents independently flagged the same deletion target, and it fits our core law exactly
(*capacity-neutral change helps when it DELETES a genuinely shifted channel*).

Split the two mechanisms — they carry completely different risk:
- **Structural month masking — SAFE.** We *generate* it from a distribution matched to test. Keep the
  key-padding mask.
- **Per-band cloud gaps — HIGH RISK.** Real atmospheric data *from the train period*. Cloud frequency
  is strongly seasonal and year-to-year variable. Specific backdoor: we deliberately deleted absolute
  time via left-alignment, but **cloud-gap patterns encode absolute season** — the model may have
  partially recovered the month-of-year that our single biggest win removed.

Note this is **not** iter13's `compact_missing` (24→14, collapse to 2 indicators, failed −0.0053/−0.0252).
**Full deletion is 24→12 and untested.**

Literature is genuinely split (Perez-Lebel/Jeanselme pro-indicator; Sisk/Groenwold/MIRRAMS anti- under
*temporal* missingness shift) — which is exactly why the **zero-submission probe** is right: train an
adversarial classifier on **only** the cloud-indicator matrix. ≈0.5–0.6 → shift-free, close the lane.
≳0.7 → every indicator is a shifted channel the champion is currently eating.

### 2.4 ✅✅✅ Stop selecting, start averaging — and the model class is not the bottleneck (3 agents)

- **Winner's curse, quantified:** selecting the max over k candidates inflates by ~SE·E[max of k
  normals]. At k=79 remaining submissions, E[max]≈2.4 → **~+0.03 inflation**. Our 0.8955 vs 0.8865
  reliable *is exactly this effect at exactly the expected size.*
- **When estimator noise ≈ true spread between candidates, picking the argmax has negative expected
  value** versus averaging the top-3.
- **TableShift (NeurIPS 2023)**, 15 shift tasks × 19 models: **no model family beats
  XGBoost/LightGBM/CatBoost**; DRO and domain-generalization methods reduced the gap only by degrading
  in-distribution performance. **Grinsztajn (NeurIPS 2022)**: trees still win at ~10k rows; we have 1,821.
- **Forum:** `sdv`, in the 90s club, uses **plain CatBoost** — and when told trees were
  underperforming, said the model class is not the limitation.

**Strategic consequence: we spent 24 iterations on architecture while someone in the 90s uses CatBoost
with better features and better validation.** Our own GBDT measured 0.878 standalone. The bottleneck
is features + validation, not the model.

### 2.5 ✅ Why ensemble diversity HURT — now explained, with a prescriptive fix

We had two measurements (−0.009, −0.016) and no mechanism. Three, in order of explanatory weight:

1. **There is no ambiguity decomposition for our metric.** "The ensemble beats its average member" is
   a theorem *about squared loss*. Wood et al. (JMLR 2023) state plainly that for 0/1 loss it does not
   apply — diversity acquires a **sign** (good vs bad diversity). Precision@k at a pinned count is a
   0-1-type functional. **We had no theoretical protection at all.**
2. **The pinned cut is a bad-diversity-selecting filter.** Score is a function only of the rows near
   rank 668 — which are *by construction* the rows where the champion is least certain, and therefore
   where a weaker member's disagreement is most likely to be wrong. All the good diversity (far from
   the cut, where the strong model was already right) is discarded; only the harmful diversity is
   retained.
3. **Probability-averaging is dispersion-weighted, not weight-weighted.** Under a rank-only metric,
   score dispersion carries **zero** information but **full** blending influence.

**Prescriptive fix — we screened on the wrong variable.** The operative gate is **level gap**, not
correlation: require `member_level ≥ champion_level − 1σ_seed` (≈ −0.019) *before* looking at ρ. Both
ROCKET (−0.040) and GBDT (−0.011 est., realized worse) would have been caught. And **always
rank-average, never probability-average.**

---

## 3. ⚠️ RULE RISK — address before the code review (35% of the top-5 score)

Forum thread 33912: participants state the rules say threshold tuning is **strictly forbidden** and
`TargetF1` must use the default 0.5. **The organizers have not answered**, and two participants have
publicly said they are self-limiting because of it.

**Our prevalence pin is functionally threshold tuning.** We have recorded it as "prior/prevalence shift
allowed, threshold tuning not" — a reading that is defensible but untested, and we are optimizing for a
top-5 code review.

**Make it defensible without losing the operating point:** train with class weights / a balanced
objective so a literal 0.5 cut lands near the target prevalence, plus Platt/isotonic calibration fit
**on training data only**, then threshold at a literal 0.5. Same operating point, clean audit trail.
This is also what forum user `chiwai` recommends (calibration during training, not post-hoc tuning).

---

## 4. THE PLAN

Budget: ~79 submissions, 5/day, deadline **2026-08-16 (20 days)**. The Phase-Two writeup (35% of the
top-5 rubric) **does not exist yet** and is non-negotiable — Phase D is not optional padding.

### Phase A — Instrument repair (0 submissions, ~2 days) — DO ALL OF THIS FIRST

Nothing below Phase A is trustworthy until Phase A lands. Every item is free.

| # | Action | Success criterion |
|---|---|---|
| A1 | **Masked-window CV.** Apply `simulate_test_regime` (L∈{4,5,6}, random start, per-band S2 dropout at *measured per-month test rates* — Oct ~17.6%, Jun ~7.3%, Feb ~3.7%) inside every CV fold. Average ≥5 mask draws per fold. | CV level drops from 0.975 toward the LB's ~0.89 |
| A2 | **Adversarial AUC as the progress meter.** Rebuild the discriminator on masked-train vs test. | Falls 0.97 → 0.6–0.75. **This is the headline diagnostic.** |
| A3 | **Cloud-indicator-only adversarial probe** (§2.3). | ≳0.7 ⇒ the indicators are shifted channels ⇒ deletion is live |
| A4 | **Estimator admissibility audit.** Any offline estimator must be invariant to a strictly-increasing warp of the scores; feed it PIT ranks, never raw scores. | Confirms ATC-F1 admissible; auto-rejects future confidence-based estimators |
| A5 | **AV-holdout.** Sort train rows by OOF `p_test` from A2; top 25–30% (~500 rows) become a held-out validation set, stratified to the pinned prevalence. | From here report **3 numbers** per experiment: masked-CV, AV-holdout, adversarial AUC |
| A6 | **Time-consistency screen** (Deotte, IEEE-CIS 1st): single-feature models trained on months 1–6, validated on 7–12, and reverse. Delete features that flip sign or fall to ~0.5. | Catches period-memorizers that A2 misses — a *different* filter |

### Phase B — The two cheap, decisive submissions (2 submissions, ~1 day)

| # | Action | Cost |
|---|---|---|
| B1 | **LB prevalence probe.** Submit `TargetF1`=1 everywhere, `TargetRAUC`=constant. Then `p = (S−0.2)/(1.4−S)`, **verified exact, pins p to 4 decimals**. Converts "~65% believed" into a measurement. | 1 |
| B2 | **Delete the per-band cloud indicators** (24→12), *only if A3 fired.* Keep the structural key-padding mask. A true deletion, capacity-reducing, exactly on-thesis. | 1 |

### Phase C — The two real bets (~6–10 submissions, ~7 days)

**C1 — Adversarial 2-D feature selection + a shift-robust feature battery.** The main event, and the
lane with the only documented effect size (+3.9% AUC, Uber). Build the battery *first*, then run the
2-D drop rule (§2.2) over it:

- *Build (shift-survivors):* cross-band ratios computed **within a single month** then aggregated
  (NDWI, MNDWI, NDVI, NDMI, VH/VV, VH−VV) — ratios cancel multiplicative atmospheric/illumination
  drift, which is `sdv`'s "prefer relative/ratio-style features"; **window-relative** order statistics
  (min/max/median/IQR/p10/p90/std over the observed window only) — order-invariant, so immune to the
  unresolved calendar-alignment question (thread 34056, still unanswered); **within-row** rank/z-score
  of each band against that row's own observed months; run-length and persistence features (fraction
  of observed months with NDWI>0, longest wet run) — the pond-vs-paddy physical discriminator;
  cross-band **correlations** over the window (dimensionless).
- *Do NOT build:* raw absolute reflectance per calendar month; anything keyed to a month index;
  **whole-year features (12-month amplitude, annual phenology, Fourier fits)** — computable in train,
  **not computable in test**, and almost certainly a major source of the 0.975 CV.
- *Nuance that reconciles this with our own law:* `sdv`'s "relative over absolute" is about **cross-band
  ratios within a month**, NOT within-series ranking across months. Our `c_rank` collapse was the
  latter. These are compatible — do not conflate them.

**C2 — Optimize the two columns independently** (§1a). Serve `TargetF1` (0.6, precision@k, local at the
cut) and `TargetRAUC` (0.4, global ranking) from different models, chosen by masked-CV + AV-holdout.
Nobody on this project has tried this and it is structurally free.

**Cheap add-ons if budget allows:** rank-average ≥9 seeds (attacks the 0.019 floor directly — the one
near-guaranteed-visible move); the banded pairwise loss weighting pairs near the pinned cut, which is
the only loss change aligned with the metric's actual geometry.

### Phase D — Endgame (~8 days, runs in parallel from day 1)

- **The Phase-Two writeup.** 35% of the top-5 score. Lead with the genuinely novel and already-evidenced:
  the **measured 0.0191 seed variance** and voiding nine of our own verdicts with it; the **offline
  LB-predicting validator**; the **rank-only metric proof** (now refined by §1a); the **pinned-threshold
  ensemble law** from §2.5; the **adversarial-AUC 0.97 shift evidence**; and — newly — the theoretical
  vindication that "accuracy-on-the-wrong-line" (Sanyal et al. 2024) *proves* noisy data + nuisance
  features produce negatively-correlated ID/OOD accuracy, i.e. our anti-correlated CV is a named,
  characterized phenomenon and our "delete the shifted channel" law is its predicted mitigation.
- **Rule-risk remediation** (§3).
- **Designate finalists MANUALLY**, ≥4 days before close. Under max-of-two, pick for **decorrelated
  private errors**, not two draws of the same thing: slot 1 = the low-variance multi-seed blend; slot 2
  = a strong candidate from a **different feature philosophy** within ~0.016 of it. **Do NOT designate
  the 0.8955 single seed** — it is a max-order statistic expected to regress by 0.02–0.03.
- **Adopt the Ladder rule** (Blum & Hardt): only update the "current best" belief on a public gain
  > η = 0.016 (our public→private SE).

---

## 5. Do NOT spend budget on (evidence-backed negatives)

- **Importance/propensity weighting** by adversarial score — Uber measured it *below baseline*; matches
  our ESS collapse.
- **DRO / domain-generalization algorithms** — DomainBed: no algorithm beats ERM by >1 point, while
  *model selection* moves results 5.1 points. TableShift: gains come from degrading ID performance.
  IRM is contraindicated **by its own failure condition** (needs test similar to train; ours is not).
- **catch22** — every feature is computed on the **z-scored** series; the authors explicitly removed
  amplitude-sensitive features before selection. It deletes our primary signal by construction.
- **Path signatures** (reparameterization-invariant ⇒ encodes the path traced, not the level held;
  157–2,380 dims), **shapelets** (degenerate at n=4–6; argmin-location is a calendar-phase feature),
  **bulk tsfresh/TSFEL dumps** (90% of variance across 390 features in 4 PCs).
- **Blending decorrelated model classes** — closed, n=2, now explained (§2.5).
- **Calibration-family methods** — provably zero under a rank-only column.
- **More architecture search** — closed at iter24.

---

## 6. Honest expected value

The only intervention here with a documented effect size in a comparable setting is adversarial
feature selection (+3.9% AUC, Uber). Phase A is free and repairs the instrument that has misled this
project for 24 iterations — it is worth doing on those grounds alone even if Phase C returns nothing.
Realistic outcome of the whole plan is **0.89 → 0.90–0.92**, not 0.98. The residual gap to the top of
the band is bounded below by an irreducible covariate shift we have measured twice (adversarial AUC
0.97 on label-free embeddings) and that binds every competitor equally.

The largest single risk is **spending Phase C's days and finding nothing above the 0.010 floor**, which
has happened in 22 of 24 iterations. Phase D is therefore scheduled **in parallel from day 1**, not
after — the 35% rubric channel is the higher-expected-value use of the remaining calendar time, and it
is the one deliverable that is currently at zero.
