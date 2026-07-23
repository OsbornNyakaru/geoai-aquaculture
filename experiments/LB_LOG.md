# LB_LOG — leaderboard reward ledger

The **reward signal** for the autoresearch loop (`gemini_loop/AGENT_BRIEF.md`). After
each Colab/Kaggle run you upload a `submission_*.csv` to Zindi and paste its public LB
score here. This is the ONLY ground truth for keep/discard — local OOF is blind.

**How to fill a row:** iteration + which experiment · the exact submission file you
uploaded · its logged test pos-rate · the Zindi public LB · keep (beats prior best) or
discard. Best-so-far is the max LB in the `keep?` = ✅ rows.

| # | Date | Experiment | Submission file | pos-rate | Public LB | keep? |
|---|------|------------|-----------------|----------|-----------|-------|
| — | 2026-07-09 | baseline: temporal Transformer | submission_seq (realized 0.649) | 0.649 | **0.8780** | ✅ (best) |
| 2 | 2026-07-20 | Step 2: GBDT+seq blend (ρ=0.849; best of prior63/65/67) | submission_seq_gbdt_prior63.csv | 0.646 | 0.8705 | ❌ (−0.0075; GBDT dilutes seq transfer despite higher OOF AUC) |
| 3 | 2026-07-20 | Step 3: seq + per_cell_detrend @ prevalence_target 0.649 | submission_seq_detrend.csv | 0.649 | 0.8266 | ❌ (−0.0514; adding channels overfits source, wrecks transfer. OOF blind: −0.004 only) |
| 4 | 2026-07-20 | Robustness: seq K 2→4 (more masking augmentation) | submission_seq_k4.csv | 0.649 | 0.8665 | ❌ (−0.0115; highest OOF 0.9840 → 2nd-worst LB. Even the winning lever overshoots. K=2 is a sharp optimum) |

**Loop paused → research round.** Three straight misses (blend, detrend, K=4). Escalated
to `gemini_loop/UPDATE_04.md` for fresh sourced ideas. Champion remains **seq K=2 @ 0.649
= 0.8780**. Key new concern: public LB ≈309 rows → ~±0.01 noise → single-probe A/B can't
resolve small (+0.005) gains; only large effects or breakages are detectable.

**Round-04 research triaged** (`gemini_loop/RESPONSE_04.md`): rejected Saerens-EM (proven
dead-end, covariate≠label shift) and Zou-threshold/EVI indices (non-transferable). Accepted
capacity-neutral ideas; testing relative-time reframing first, banking TTA + multi-seed bagging.

| 5 | 2026-07-21 | Relative-time reframing (left-align window to t_rel=0; capacity-neutral) | submission_seq_reltime.csv | 0.649 | **0.8908** | ✅ **NEW BEST** (+0.0128; clears ±0.01 noise. First win since champion — capacity-NEUTRAL structural reframe transfers where every capacity-ADDING change lost. Attacks calendar-month memorization.) |
| 6 | 2026-07-21 | MC temporal-dropout TTA on champion (inference-only; mask 1-2 active months, soft-vote 8 views) | submission_seq_reltime_tta.csv | 0.649 | 0.8885 | ❌ (−0.0023; within noise but did NOT beat champion. 2nd robustness move to land within-noise — variance reduction isn't the lever here. Reverted; kept as a private-LB hedge candidate.) |

**Current best: 0.8955** (temporal Transformer + relative-time + cross-view invariance λ=1.0, realized
pos-rate 0.649). Prior anchor 0.8908 = relative-time alone.

**Pattern after iter6:** capacity-neutral *robustness* moves (TTA now, and by extension multi-seed
bagging) land within the ±0.01 noise — they can't be validated on public and don't lift rank. The
only changes that moved LB were *structural* (GBDT→seq +0.05; relative-time reframe +0.013). Next
high-value direction = the next structural reframe, not more variance reduction.

**Round-05 research triaged** (`gemini_loop/RESPONSE_05.md`): BOTH Deep Research reports (Gemini +
Claude) independently ranked **duration-normalized positions** #1 → staged as iter7. Re-rejected
Saerens-EM (3rd time), Zou/WIF/EVI, CAST self-training, CropNet-blend big-bang. Banked: NoPE set
encoder (iter8), cross-view invariance objective (iter9), one-time prevalence sweep.

| 7 | 2026-07-21 | Duration-normalized fractional positions (share one [0,1] frame across window lengths; parameter-neutral) | submission_seq_dnorm.csv | 0.649 | 0.8844 | ❌ (−0.0064; did not beat champion. Diagnosis: window LENGTH is already train/test distribution-matched by the masking augmentation → no length-shift to remove; dnorm only normalized away possibly-informative length signal. Also confound: interpolation read untrained table indices 6-11. REVERTED.) |

**Refined positional lesson (iter5 vs iter7):** relative-time removed window START (calendar month,
which IS shifted train-vs-test) → +0.0128 WON. dnorm removed window LENGTH (already matched by
augmentation, NOT shifted) → −0.0064 LOST. **Positional reframes help ONLY when they delete a channel
that is actually shifted between train and test.** iter8 NoPE tests removing positional identity entirely.

| 8 | 2026-07-21 | NoPE / permutation-invariant SET encoder (drop positional embedding; pos_encoding=none) | submission_seq_nope.csv | 0.649 | 0.8917 | ➖ TIE (+0.0009, within noise). Removing ALL position costs nothing → order is nuisance-or-neutral (confirms SAR set-statistic physics). **LOCKED as the diverse private-LB finalist** (max-different model, ties public → fails on different rows). |

**Positional lane EXHAUSTED:** absolute-calendar 0.8780 → relative-time 0.8908 (+0.013, WON) → NoPE
0.8917 (tie). All the gain was deleting calendar-START memorization; residual position is neutral.
Two reframes now < +0.003 → stop-rule. The other big shift channel (per-series amplitude) is toxic to
touch (detrend −0.051). Budget shifts to the OBJECTIVE lever (iter9) then endgame (prevalence + finalists).

| 9 | 2026-07-21 | Cross-view invariance objective (L=BCE+λ·Var_k(logit) across K=2 views; λ=1.0) | submission_seq_xview.csv | 0.649 | **0.8955** | ✅ **NEW BEST** (+0.0047 vs 0.8908; best public score yet). Reduced overconfidence (oof_auc 0.9936→0.9894, prevalence delta 2.03→1.30) — hit the model's diagnosed weakness. At edge of noise, so iter10 probes λ=3.0 to test if the lever scales. |
| 10 | 2026-07-21 | Cross-view invariance strength probe (λ 1.0→3.0) | submission_seq_xview_l3.csv | 0.649 | 0.8921 | ❌ (−0.0034; did not beat champion. REVERTED to λ=1.0.) |

**Objective lane CLOSED at λ=1.0.** iter10 shows λ=1.0 is an *interior optimum*, not a floor.
λ=3.0 pushed de-saturation further (t\* 0.4450→0.3400, prevalence delta 1.30→0.725, raw test
pos-rate 0.553→0.583) while `oof_auc` HELD at 0.9896 — so the drop is **not** ranker collapse;
de-saturation simply stops paying past λ=1. Both structural lanes are now measured closed
(positional: dnorm −0.006 / NoPE +0.001; objective: λ=3 −0.003). **Loop PAUSED → research round 06**
(`gemini_loop/UPDATE_06.md`), briefing lanes never explored: LB-predictive local validation,
sequence-level feature engineering, untried mathematical techniques, and CV design.

**BREAKTHROUGH (2026-07-21):** relative-time reframing broke the 10-day 0.8780 plateau. The
"added capacity hurts" lesson now has its complement: capacity-NEUTRAL *structural* change (same
params, reframed coordinates) is the direction that transfers. Iter6+ bank robustness moves (TTA,
multi-seed bagging) on top; both capacity-neutral, gated vs 0.8908.

| 11 | 2026-07-22 | **Offline LB-predicting validator** (retro-fit to 7 known-LB anchors) | *no submission* | — | **n/a — 0 subs** | ✅ **PASSED** |

**iter11 PASSED — the measurement constraint is broken.** Retro-fit against the 7 anchors:

| Estimator | Concordance (pairs with \|ΔLB\| > 0.01) | Spearman ρ | Verdict |
|---|---|---|---|
| **ATC-F1** — metric-aligned, built this round from the rank-only proof | **15/15** | **+0.964** | **PASS** (exact null p ≈ 0.005) |
| **DIS** — two-seed disagreement | **5/5** | **+1.000** (n=4) | **PASS** (p ≈ 0.042 → 2nd vote only) |
| ATC — the ORIGINAL pre-committed estimator | 6/15 | **−0.429** | FAIL |
| MARG — naive control | 8/15 | −0.321 | FAIL |

**Read this carefully: the pre-repair iter11 would have failed completely.** Raw ATC is
anti-correlated, MARG is anti-correlated, and DIS would have printed "insufficient bundles" because
only the champion carried a second seed. The two estimators that cleared are exactly the two things
round 07 added (`gemini_loop/RESEARCH_07.md`).

**The two failures are themselves evidence for the rank-only proof.** ATC and MARG both measure
*confidence/saturation*, which the LB provably cannot see — and both came out **negative**. Visible
in the table: `seq_a_l3` and `seq_a_xview` have the LOWEST MARG (0.309, 0.369) yet the HIGHEST LB.
De-saturation was never the mechanism; it is an artifact that anti-correlates with what wins.

**→ SCENARIO A is live** (`RESEARCH_07.md` §6). Candidates are now screened offline and submissions
spent only on things ≥2 cleared estimators already rank above the champion. ~80 submissions remain,
and the constraint moves from measurement back to idea quality.

| 12 | 2026-07-22 | **First offline screen**: `mean_std` · `mean_max` · 24→14 compaction · rank-replacement · antithetic views | *no submission* | — | **n/a — 0 subs** | ➖ ALL HELD |

**iter12 — all five candidates HELD; no submission spent.** Estimators re-certified *identically*
to iter11 (ATC-F1 15/15 ρ=+0.964; DIS 5/5 ρ=+1.000), so the screen is trustworthy.

| Candidate | ATC-F1 Δ | DIS Δ | Votes | |
|---|---|---|---|---|
| `c_meanmax` | **+0.0838** | −0.0301 | 1/2 | HOLD |
| `c_meanstd` | +0.0185 | −0.0175 | 1/2 | HOLD |
| `c_antithetic` | +0.0060 | −0.0165 | 1/2 | HOLD |
| `c_compact` | 0.0000 | 0.0000 | 0/2 | **VOID — never ran, see below** |
| `c_rank` | **−0.1703** | −0.0796 | 0/2 | HOLD (decisively) |

**The rule earned its keep.** ATC-F1 was enthusiastic about `mean_max` (+0.0838, the largest signal
of the round) but DIS disagreed. Under the old blind regime that would have been a submission; here
it cost nothing. The two estimators disagree in sign on *every* live candidate — worth watching.

**FINDING 1 — the amplitude question is now genuinely ANSWERED.** `c_rank` (within-series rank
*replacing* absolute values) **collapsed**: OOF 0.9753 → 0.857 / 0.865 across both seeds, ATC-F1
−0.1703. Amplitude is the **primary signal** — the pond discriminator really is "persistently LOW
backscatter", an absolute level. This closes the rank/ordinal family **with evidence**, and it
retires the question reopened when we found `per_cell_detrend` had only ever *appended* channels.
Note what is and isn't established: *removing* amplitude is catastrophic; that is not the same
claim as the old "amplitude is toxic" law, which remains unevidenced.

**FINDING 2 — `c_compact` was never tested (my bug).** The flag was set at `seq.compact_missing`
but `to_inputs()` only receives `cfg["seq"]["channels"]`, so it never reached the model. The run came
out **bit-identical to the champion** (`n_features: 24`, identical fold_scores) and the screen scored
that silent no-op as a legitimate 0.0000 tie. The independently-computed `n_features` in
`run_pipeline.py` agreed with the wrong answer, so nothing caught it. Fixed (flag moved under
`seq.channels`), and the pipeline now logs the **actual** input width every run. Re-tested in iter13.

**FINDING 3 — DIV failed, and in the direction that refutes its own hypothesis.** Fold-ensemble
diversity scored 2/15 concordant, **ρ = −0.857**: *lower* diversity goes with *higher* LB — the
opposite of hypothesis H1, which predicted fold-averaging diversity was the hidden driver. H1 is not
supported. (It fails the strict all-or-nothing gate, so it is not usable negated either.)

| 13 | 2026-07-22 | **Second offline screen**: `c_compact` re-test · `mean_max` without cross-view · K=3 · dropout 0.3 | *no submission* | — | **n/a — 0 subs** | ✅ **1 SUBMIT** |

**iter13 — the screen fired for the first time.** Estimators re-certified identically for the third
consecutive run (ATC-F1 15/15 ρ=+0.964; DIS 5/5 ρ=+1.000).

| Candidate | ATC-F1 Δ | DIS Δ | Votes | |
|---|---|---|---|---|
| **`c_dropout3`** (dropout 0.2 → **0.3**) | **+0.0165** | **+0.0029** | **2/2** | ✅ **SUBMIT** |
| `c_k3` (K=3) | −0.0032 | −0.0010 | 0/2 | HOLD |
| `c_compact` (24→14 ch) | −0.0053 | −0.0252 | 0/2 | HOLD |
| `c_meanmax_l0` (mean_max, λ=0) | −0.0847 | −0.0126 | 0/2 | HOLD |

**`c_dropout3` is the first candidate ever to clear.** It is **exactly parameter-neutral** — pure
regularization strength — and it is the most on-thesis knob in the repo under our own design law
("less fit transfers better"), yet it had **never been touched once in twelve iterations**. We spent
those twelve iterations on architecture while the plainest regularization knob sat at its default.
⚠️ **Honest caveat:** the ATC-F1 margin (+0.0165) is solid but the DIS margin (+0.0029) is *tiny*.
This is a 2/2 by the pre-committed rule, not a resounding one. Estimator deltas are **not** on the
LB scale — do not read +0.0165 as an expected LB gain.

**`c_compact` genuinely ran this time and FAILED** (−0.0053 / −0.0252; non-zero deltas confirm the
flag took effect, unlike iter12's silent 0.0000 no-op). The 24→14 channel deletion was derived
**independently by two research agents** and it does not work. Independent convergence of two
reasoning agents is **not** evidence — a useful correction to how we weight that signal.

**`c_meanmax_l0` flipped hard negative (−0.0847) vs iter12's `c_meanmax` at +0.0838 with λ=1.**
Same pooling, opposite sign, the only difference being cross-view invariance. So the upper-tail
statistic *depends on* the variance penalty rather than competing with it — the opposite of the
hypothesis that motivated this probe.

**`c_k3` ≈ 0 on both estimators.** K=2 is confirmed optimal, now from three points rather than the
two that originally justified calling it a "sharp optimum."

| 14 | 2026-07-22 | dropout 0.3 **at seed 7** (⚠️ NOT the screened artifact) | submission_c_dropout3_**s7**.csv | 0.649 | **0.8675** | ⚠️ **CONFOUNDED** |

## ⚠️ iter14 is confounded — and it opens the most important question in the project

**What was uploaded was not what the screen approved.** The offline screen scored `c_dropout3`
(**seed 0**). The file submitted was `c_dropout3_**s7**` (**seed 7**). So this run changes **two**
variables at once versus the champion: dropout 0.2→0.3 **and** seed 42→7.

**0.8675 vs the champion's 0.8955 is −0.0280** — about **4.7σ** on the paired scale (SE ≈ 0.006).
Far too large to shrug off, and impossible to attribute as it stands.

### Why this might matter far more than dropout

**We have never measured our seed-to-seed spread.** It has been *assumed* to be ~±0.01 from
row-count theory since the very first plateau, and every verdict in this ledger rests on that
assumption. Two readings of 0.8675, with wildly different consequences:

| If… | Then |
|---|---|
| **dropout 0.3 is genuinely bad** | the screen produced a **false positive** on its first-ever SUBMIT, and the 2/2 rule needs a margin floor (the DIS vote was only +0.0029) |
| **seed variance is ≈0.028** | **most of this ledger is noise.** iter9's champion win (+0.0047), iter8 NoPE (+0.0009) and iter10 (−0.0034) are all *far* inside a ±0.028 band, and the "champion" may not be distinguishable from three other configs |

The second possibility is the more serious one, and it is **cheap to settle**.

### The disambiguating measurement

Upload **`submission_seq_a_xview_s7.csv`** — the *champion configuration at seed 7*, already written
by the iter13 run. It changes **only** the seed, so:

- **≈0.895** → seed variance is small ⇒ dropout 0.3 genuinely failed ⇒ the screen gave a false
  positive and needs a margin floor.
- **≈0.867** → **seed variance is ≈0.028** ⇒ dropout is exonerated, and a large part of this ledger
  needs re-reading as noise. This would be the single most consequential measurement of the project.

This is the seed-replication run that has sat queued since iter6 and was never spent. It is now the
highest-value submission available, and it costs one.

| 15 | 2026-07-22 | **SEED REPLICATION OF THE CHAMPION** — identical config, seed 42→7 | submission_seq_a_xview_s7.csv | 0.649 | **0.8764** | 🚨 **VOIDS THE LEDGER** |

# 🚨 THE SEED RESULT — read this before trusting any earlier row

**The champion configuration, changing nothing but the RNG seed, moved 0.0191.**

| Run | Config | Seed | LB |
|---|---|---|---|
| `seq_a_xview` | champion | 42 | **0.8955** |
| `seq_a_xview_s7` | **identical** | 7 | **0.8764** |
| `c_dropout3_s7` | dropout 0.3 | 7 | 0.8675 |

**Now iter14 can be attributed properly, paired at matched seed:**
- **seed effect** (dropout held at 0.2): **−0.0191**
- **dropout effect** (seed held at 7): **−0.0089**
- **The seed matters 2.15× more than the change we were testing.**

## What survives

| Verdict | Δ | Status |
|---|---|---|
| GBDT → Transformer swap | +0.0500 | ✅ SURVIVES |
| per-cell detrend | −0.0514 | ✅ SURVIVES |
| **relative-time (iter5)** | +0.0128 | ❌ **VOID** — inside one seed swing |
| K=4 (iter4) | −0.0115 | ❌ VOID |
| GBDT blend (iter2) | −0.0075 | ❌ VOID |
| dnorm (iter7) | −0.0064 | ❌ VOID |
| **cross-view invariance (iter9) — the CHAMPION CLAIM** | +0.0047 | ❌ **VOID** |
| λ=3 (iter10) | −0.0034 | ❌ VOID |
| TTA (iter6) | −0.0023 | ❌ VOID |
| NoPE (iter8) | +0.0009 | ❌ VOID |

**Two of eleven verdicts survive.** Everything else — including the relative-time "breakthrough"
that broke a 10-day plateau, and the cross-view win that made this model champion at all — is
smaller than the noise we never measured. We spent roughly ten iterations doing careful A/B design
against a measurement floor **twice as large as we believed**, and the ±0.01 estimate that governed
every decision came from row-count theory, not from a measurement.

**Our "champion" is plausibly just a lucky seed.** 0.8955 is the better of two draws from a
distribution whose sd is ~0.013. Note also that our two designated finalists (xview 0.8955, NoPE
0.8917) differ by 0.0038 — they are not two different models, they are two draws.

## The strategic consequence

When run-to-run variance dominates every real effect, the highest-value move is **not another
architectural probe — it is to stop sampling one draw.** Averaging M seeds shrinks the pooled
prediction's variance ~M-fold, and because the metric is **rank-only** after the prevalence pin,
averaging moves the submitted ordering toward the configuration's *expected* ordering rather than
one RNG draw's ordering. `tools/seed_average.py` added; iter15 pools five champion seeds.

This also **retires a "banked" idea we deprioritized for the wrong reason.** Multi-seed bagging was
parked back at iter6 as a "robustness move that lands within noise." That was exactly backwards:
the noise *is* the problem, so the variance-reduction move is the highest-value one available.

## iter15 — the seed guard fired, and the screen's resolution is now calibrated

**All four candidates HELD.** Two of them (`c_dropout3`, `c_wd3`) had **2/2 votes** and were
**downgraded to HOLD** because their ATC-F1 margins sat inside the estimator's own seed noise. The
guard built one iteration earlier caught exactly the mistake that produced iter14.

| Candidate | ATC-F1 Δ | LB-equiv | DIS Δ | Verdict |
|---|---|---|---|---|
| `c_meanmin` | **+0.0672** | **+0.0109** | −0.0214 | HOLD (1/2 — but the only margin that clears the noise floor) |
| `c_dropout3` | +0.0165 ~ | +0.0027 | +0.0107 | **HOLD** — 2/2 votes, ATC-F1 inside seed noise |
| `c_wd3` | +0.0053 ~ | +0.0009 | +0.0049 | **HOLD** — 2/2 votes, ATC-F1 inside seed noise |
| `c_do40` | −0.0021 ~ | −0.0003 | +0.0039 | HOLD |

### The calibration — two independent routes to the same noise floor

Fitting LB against ATC-F1 across the seven anchors gives **LB = 0.1628·ATCF1 + 0.7714**. So:

| Quantity | Estimator units | **LB-equivalent** |
|---|---|---|
| ATC-F1 seed sd (n=5 champion seeds) | 0.0576 | **±0.0094 LB** |
| Directly measured champion seed spread (42 vs 7) | — | **0.0191** (sd ≈0.013) |

**Those agree.** The offline estimator's noise floor and the leaderboard's measured seed noise were
derived by completely separate routes and landed in the same place. That is the strongest validation
this framework has received — and it fixes the screen's resolution at **≈0.010–0.013 LB**.

**Pairwise rank correlation between champion seeds: mean 0.9511, min 0.9467.** So ~95% of our test
ordering is reproducible and ~5% is RNG lottery — and that 5% is enough to move the LB by 0.019.

### 🚨 The strategic conclusion this forces

Only effects **larger than ~0.010 LB** are measurable at all — offline *or* on the public board.

| Effect ever measured | Δ | Measurable? |
|---|---|---|
| GBDT → Transformer swap | +0.0500 | ✅ |
| per-cell detrend | −0.0514 | ✅ |
| `c_rank` (offline, LB-equiv) | −0.0277 | ✅ |
| **everything else we have ever tested** | ≤ 0.0128 | ❌ |

**Both surviving effects are model-class changes.** Every architectural tweak, loss term, pooling
variant, positional reframe and regularization knob we have probed sits below the floor and is
*unmeasurable in principle* with our budget — not merely unproven.

**So: stop running small A/B probes.** They cannot be resolved. The two remaining moves that are
sized to the floor are:
1. **Variance reduction** — seed-averaging, which converts the 0.019 lottery into a stable estimate.
2. **A model-class change** — the same species as the +0.05 GBDT→Transformer swap. That is the
   **Presto lane** (`RESEARCH_07.md` §5e), now the only fundable architectural direction.

| 16 | 2026-07-22 | **Seed-averaged champion** (5 seeds, rank-pooled; seed rank-corr 0.951) | submission_champion_seedavg5.csv | 0.649 | **0.8865** | ✅ prediction confirmed |

## iter16 — the variance model is validated to 0.0006

| Artifact | LB |
|---|---|
| xview seed 42 | 0.8955 |
| xview seed 7 | 0.8764 |
| **single-seed mean** | **0.88593** |
| **seed-average (5 seeds)** | **0.88653** |

**Predicted ≈0.886, measured 0.88653 — off by +0.0006.** The model of this competition is correct:
0.8955 was an upward fluctuation, not a better model.

**But there was NO ensemble gain.** Seed-averaging landed *at* the member mean (+0.0006, ≈0.05σ),
not above it. We bought **variance reduction, not level**. The reason is visible in the diagnostic:
seeds are **95.1% rank-correlated**, so only ~5% of the error is independent and available to
average away. More seeds will not change this — it is diminishing returns on a small independent
component.

### What to designate as finalists — the counter-intuitive part

Shrinking each artifact's observed score toward the seed distribution (prior mean 0.886, sd 0.013;
public-slice SE 0.012):

| Artifact | Observed | **Shrunk estimate of true quality** |
|---|---|---|
| xview seed 42 | 0.8955 | **0.8911** (public luck partly removed) |
| seed-average | 0.8865 | **0.8865** (no seed luck to remove) |

The difference, **+0.0046, is inside our ~0.010 resolution floor — a statistical tie.** But the
*risk profiles differ*: the seed-average carries only public/private sampling noise, while the
single seed carries that **plus** seed noise on the unseen 721-row private slice.

**→ Designate `champion_seedavg5` + `seq_a_xview`.** That is a genuinely diverse pair — a
low-variance consensus and a point estimate — and far better than the previously-planned
xview + NoPE, which differ by 0.0038 and are two draws of the same thing rather than two models.

| 17 | 2026-07-23 | **PRESTO lane** — frozen pretrained encoder + 129-param logistic head | *screened, HELD* | 0.649 | **not submitted** | ❌ lane dead |
| 18 | 2026-07-23 | **GRAND ENSEMBLE** — cross-architecture rank-blend (reltime/nope/l3/xview) | *marginal ρ=0.94* | 0.649 | **to upload ×1** | ➖ variance-only |
| 19 | 2026-07-23 | **DISPERSION POOLING** — mean_min / mean_std / moments (replace masked-mean) | *screen first* | 0.649 | **pending** | — |

---

## iter18 result — architecture pooling is MARGINAL, not a level lever

Cross-architecture rank-correlation **mean ρ = 0.9395** (min 0.9097) — only a hair below the seed
baseline 0.9511. The four tied transformer variants are the *same model class*, so they barely
decorrelate; the variance-reduction factor (1+ρ(M−1))/M ≈ 0.96 forces the blend to the member mean,
exactly like seed-avg. The screen line (`archblend4 ATC-F1 −0.0101 LB`) is partly a rank-vs-prob
representation artifact and should not be read as "worse." **Read the matrix: marginal.** Decision:
upload `champion_archblend4` **once** to bank the lowest-variance finalist (pools seed AND architecture
noise → best private-slice artifact per the order-statistics argument), then move on.

**The lesson (confirmed by both round-09 researchers):** pooling transformer-variants cannot buy
*level*; a member needs ρ<0.9 (target 0.6–0.75) to add ensemble level, and only a *different model
class* (MiniRocket/CropNet, iter20) can supply that. The representation lever comes first (iter19).

## iter19 — dispersion / lower-tail pooling (staged)

Three-way convergence: Claude Research (moment pooling — ponds are permanent = low dispersion, mean
discards it), Gemini (replace the pool), and our own iter12 `mean_min` probe (the ONLY candidate to
ever clear the floor: +0.0672 ATC-F1). Isolated change = `seq.pooling` ∈ {mean_min, mean_std,
moments}, each at 2 seeds so DIS + the seed-noise guard apply. New `moments` mode (mean⊕std⊕min⊕max,
pooled_dim 4·d) added and unit-verified (correct widths, identity-preserving zeroed extra halves).
Submit only if ≥2 cleared estimators beat champion AND the margin exceeds the estimator's seed sd —
the guard that correctly held `mean_min` last time.

---

## iter17 — the Presto lane is DEAD, learned for 0 submissions

All four Presto configs returned **adversarial AUC 0.965–0.976** on the frozen embeddings. Our
pre-committed go/no-go: >0.9 ⇒ the encoder is *encoding* the designed temporal shift, not
normalizing it. The screen agreed independently:

| config | adv-AUC | ATC-F1 vs champ | OOF combined | verdict |
|---|---|---|---|---|
| c_presto_const | 0.9757 | **−0.0444 LB** | 0.9672 | HOLD (1/2) |
| c_presto_true  | 0.9668 | **−0.0589 LB** | 0.9693 | HOLD (1/2) |

ATC-F1 (ρ+0.964, metric-aligned) puts Presto well below champion; DIS votes up but is the weak
second vote. Presto's OOF is *already* below champion's 0.975. No version of "fund it" survives.

**The valuable byproduct.** A near-perfect train/test separator (AUC 0.97) exists even in a
general-purpose, label-free representation of the raw pixels. That is independent proof the
train→test **shift is real and large** — it lives in the data, not our pipeline. So the ~0.975 OOF
vs ~0.89 LB gap is mostly *irreducible covariate shift*, which bounds the ceiling for every model
and explains why the champion (which carries shift-invariance machinery) beats a faithful
raw-signal encoder. **Scenario B is now active: the model-class frontier via foundation models is
closed.**

## iter18 — the GRAND ENSEMBLE (staged)

Seed-averaging bought variance reduction but no level (seeds 95.1% correlated). Pooling across
**architectures** is the last cheap shot at *level*, IF they are decorrelated. `tools/arch_blend.py`
two-level rank-averages the tied top cluster (reltime/nope/l3/xview) and prints the
cross-architecture rank-correlation matrix as the **free go/no-go**: ρ < ~0.90 → upload the blend
(bounded downside, same variance-reduction category as the seed-avg); ρ ≈ 0.95 → no gain, pivot to
pseudo-labeling / ROCKET. The offline screen cannot resolve an ensemble gain (ATC-F1 seed sd 0.0576
== ±0.0094 LB), so the correlation matrix — not the screen — is the deciding instrument.

---

### Reading the Step-2 run (what to record)
1. In Cell 4's log find `OOF rank correlation between components = ρ`.
   - ρ < ~0.90 → the blend adds decorrelated signal → upload the
     `submission_seq_gbdt_priorXX.csv` whose logged pos-rate is nearest **0.65**.
   - ρ ≈ 1.0 → skip the blend (record it as discarded); advance to Step 3.
2. Optional: also upload `submission_seq_v3.csv` to re-confirm the 0.8780 anchor.
   Reproducibility check: the **full** seq run prints `final_oof ≈ 0.9827` (the
   `0.88046` in results.tsv is the `--smoke` fingerprint, not the full run).
3. Paste the Zindi score into the table; mark ✅ only if it beats 0.8780.
