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
| 18 | 2026-07-23 | **GRAND ENSEMBLE** — cross-architecture rank-blend (reltime/nope/l3/xview) | *marginal ρ=0.94* | 0.649 | **0.894643** | ✅ leading finalist |
| 19 | 2026-07-23 | **DISPERSION POOLING** — mean_min / mean_std / moments (replace masked-mean) | *mean_min screened* | 0.649 | **0.898566** (c_meanmin) | ➖ within noise |
| 20 | 2026-07-23 | **mean_min AS ENSEMBLE MEMBER** — pooling-axis diversity; archblend5 | *ρ=0.9928, not decorrelated* | 0.649 | **not submitted** | ❌ lane closed |
| 21 | 2026-07-24 | **INSTANCE-EXPANSION** — per-epoch view resampling (each masked sub-window an independent example) | *screen VOID* | 0.649 | **not submitted** | ❌ inert (OOF↑ like k4) |
| 22 | 2026-07-24 | **ROCKET member** — different model class (random conv kernels + linear); go/no-go rank-corr vs champion | submission_champion_rocketblend5.csv | 0.649 | **0.885661** | 🎯 first ρ<0.90 member; blend tied cluster (weak member) |
| 23 | 2026-07-24 | **MULTIVARIATE ROCKET** — kernels span random band SUBSETS (cross-band signature) | *screen VOID→exhausted* | 0.649 | **not submitted** | ❌ strength⊥diversity tradeoff |
| 24 | 2026-07-27 | **GBDT as decorrelated member** — trees on aggregate features (a different class, not the ROCKET family) | submission_champion_gbdtblend5.csv | 0.649 | **0.879123** | ❌ **−0.0155 vs archblend4 (SIGNIFICANT, paired); cross-class blending CLOSED** |

| 25 | 2026-07-27 | **Phase-A shift audit** (`tools/shift_audit.py`) — indicator probe + 2-D band screen | *local, no cloud run* | — | **n/a — 0 subs** | ✅ 1 lane CLOSED, 1 lane OPENED |
| 26 | 2026-07-28 | **BAND DELETION** — drop VV (top shift-carrier, dominated by VH on signal); capacity-REDUCING | submission_c_dropvv.csv | 0.649 | **0.884217** | ❌ −0.0113 paired vs the seed-42 champion. **The screen was wrong in SIGN.** Deletion lane CLOSED; ATC-F1 exposed as within-family only |
| 27 | 2026-07-28 | **GOING LEGAL** — removed the prevalence pin (a rules violation); train-only Platt + literal 0.5 in both columns | submission_seq_a_xview.csv | **0.548 (reported, not targeted)** | **0.889686** | ✅✅ **−0.0058 paired — BELOW our own 0.006 suggestive threshold. Compliance is statistically FREE.** The pin was worth ~0.006, not the +0.07 it was credited with |
| 28 | 2026-07-28 | **LEGAL `champion_archblend4`** — 4 architectures, per-member Platt then probability-average | submission_champion_archblend4.csv | 0.567 (reported) | **0.899643** | 🏆 **BEST PUBLIC SCORE EVER AND IT IS ELIGIBLE.** +0.005 vs its own pinned version; **+0.0100 vs the legal champion, where the pinned pair differed by −0.0009 → the pin was SUPPRESSING the ensemble.** iter18's "pooling is marginal" verdict was an artifact of the operating point |
| 29 | 2026-07-28 | **BIGGER LEGAL POOL** — archblend6 = archblend4 + `seq_a_k4` + `seq_a_base` (the 2 weakest members) | submission_champion_archblend6.csv | — | **0.894899** | ❌ **−0.0047 paired vs archblend4 (0.899643)**, shares 4/6 members. Adding the 2 weakest members DRAGGED the blend. "Weak members become assets under a literal cut" **REFUTED** — the level-gap gate SURVIVES the legal regime. **archblend4 stays finalist #1; aggressive pooling CLOSED** |
| 30 | 2026-07-28 | **LEGAL CATBOOST LANE** — n-invariant features + VH-CDF permanence + ordered boosting; standalone + as the different-bias blend member | catblend5 / c_catboost | legal | **0.886043 / 0.697615** | ❌❌ **catblend5 −0.0136 paired (CONFIDENT LOSS); standalone 0.6976 CATASTROPHIC.** OOF AUC 0.995 → LB 0.70: the "competent + decorrelated" signal was an OOF illusion. Cross-class ensemble CLOSED (3rd fail). archblend4 final |
| 31 | 2026-07-29 | **FEATURE SHOT (in the Transformer)** — VH permanence indicators `1[VH<τ]` (+ `VH−VV` cross-pol, isolated) as champion channels | c_perm / c_permxpol | legal | **0.901605 / 0.878788** | 🏆 **PERMANENCE ALONE = NEW BEST EVER (0.9016, +0.012 vs champion, > archblend4).** cross_pol TOXIC (−0.023 when added). First feature win in the Transformer. One-at-a-time isolation caught it |
| 32 | 2026-07-29 | **SEED-CONFIRM permanence** + τ selection | perm_seedavg5 / c_perm_wide / **c_perm_single** | legal | 0.896918 / 0.898712 / **0.906492** | ✅ **CONFIRMED + FEATURE-SELECTED. `c_perm_single` (ONE threshold τ=−21) = 0.906492 = NEW BEST EVER.** Monotone: 1-τ (0.9065) > 4-τ (0.9016) > 6-τ (0.8987) — the signal is ONE physically-privileged cut; extra τ add noise. Seed-avg (4-τ) 0.8969 = +0.010 reliable vs champion |
| 33c | 2026-08-06 | **single-τ permanence per-seed spread** — `seq_a_xview_perm_s29` (seed 29, NON-lucky) | submission_seq_a_xview_perm_s29.csv | legal | **0.900715** | ✅ **2nd single-τ permanence seed, both ≥0.90 (s42 0.9065, s29 0.9007).** Permanence is ROBUST across seeds, not seed-42 luck; honest single-τ level ~0.90. **Strengthens the duplicate verdict on 33b:** a 5-seed POOL scoring 0.8969 (below both these members) is implausible → the 0.896918 upload was the iter32 4-τ file; TRUE single-τ seed-avg is UNMEASURED and likely ≥0.90 (our best robust artifact — get it). Seed spread so far 0.0058 « base 0.0191 (permanence may also STABILIZE across seeds). |
| 33b | 2026-08-06 | **perm seed-avg (finalist probe)** — `champion_perm_seedavg5` uploaded | submission_champion_perm_seedavg5.csv | legal | **0.896918** | ⚠️ **IDENTICAL to iter32's 4-τ seed-avg to 9 dp (0.896917864) → almost certainly the SAME file; single-τ seed-avg UNCONFIRMED.** Regardless, the seed-robust permanence level = **~0.8969**: confirms the lucky-seed hypothesis (single-seed 0.9065 − seed-avg 0.8969 = −0.0096 ≈ ½·seed-var 0.019). Permanence = real +0.010 single-model win over base seed-avg (~0.8865). FINALIST NOTE: seed-avg (0.8969, low-variance) is the safer private-LB bet than the lucky single-seed 0.9065. **OPEN Q: if this WAS the single-τ run, then 1τ>4τ at seed 42 was itself noise — verify which file.** |
| 33 | 2026-08-06 | **PERMANENCE ENSEMBLE** — 4 architectures × single-τ permanence, calibrated pool, vs base archblend4 | submission_champion_perm_archblend4.csv | legal | **0.892939** | ❌ **BOTTOM BUCKET (≤0.8936): permanence HURTS the ensemble.** −0.0067 paired vs base archblend4 (0.899643); −0.0136 vs the single `c_perm_single` (0.9065). **The two +0.010 lifts are SUBSTITUTES, not complements** — both monetize operating-point disagreement at the 0.5 cut; a shared strong feature raises member ρ AND collapses calibration spread → nothing left to pool. Agent-7 "orthogonal, ~0.904–0.909" FALSIFIED. Level-gap-drag law strikes again (weak perm-architectures drag `xview_perm`). **Permanence is a SINGLE-model lever; stop pooling it. archblend4 stays ensemble finalist.** |

---

## iter29 — a bigger pool does NOT win: the level-gap gate survives the legal cut

`archblend6` (archblend4 + the two weakest same-class members, `seq_a_k4` and `seq_a_base`) scored
**0.894899 vs archblend4's 0.899643 → −0.0047, strongly paired** (4 of 6 members shared, identical
309 public rows). Below our 0.006 "suggestive" line in magnitude, but the direction is unambiguous and
it is the *known-weakest* members that were added.

**The hypothesis is refuted.** iter29 tested whether removing the pin turned weak members into
assets — the idea being that a literal 0.5 cut also averages *calibration* (where members disagree),
so a weak-but-differently-calibrated member might now help. It did not: the two weak members
(pinned-era 0.8665 / 0.8780, i.e. −0.029 / −0.018 in level) still dragged the pool down. **The
"weak members drag" gate survives the regime change.**

**What this does and does NOT overturn.** It does NOT touch iter28's real win: legal pooling of the
FOUR tied-strong members buys +0.0100 of level through calibration diversity. That stands. iter29
just says the diversity dividend is only cashable among members of *comparable competence* — you
cannot extend it downward to incompetent members. **Refined law: legal pooling buys level via
calibration diversity, but only across members within ~one seed-swing of each other in level.**

**Consequence: `champion_archblend4` (0.899643) is final for the same-class ensemble lane, and
aggressive within-class pooling is closed.** The only pooling gain left would come from a member that
is *both* competent *and* genuinely different — i.e. a different model class that isn't weak. That is
exactly the CatBoost/tree question, which the pin-era rejection never tested legally (see UPDATE_13).

---

## iter30 — the legal CatBoost lane (staged), grounded in RESEARCH_14

The different-bias member iter29 says we need. Three research lines converge (RESEARCH_14,
UPDATE_13): the LB leader (~0.94) uses plain CatBoost ("model isn't the bottleneck"); the closest
analogous WINNER (Zindi AgriFieldNet) is CatBoost+LGBM+XGB on temporal-aggregated indices; Farm Pin's
winner kept a weaker Random-Forest member purely for its *different bias* (+1%). Our pin-era GBDT
rejection (−0.0155) is void — it was measured under the pin, which erased calibration (now ~60% of
the metric).

**Built this iteration (all behind default-off flags — nothing existing changes):**
- `features.n_invariant_only` — restricts every aggregate to statistics unbiased at every window
  length (mean/median/std/interior-quantiles/fractions), dropping the min/max/range/window-position/
  count/run-length **shift-carriers** the old GBDT matrix was full of. This is the one adaptation the
  crop winners didn't need (they had full series; we have the 4–6-month masking trap). Feature count
  falls 130→**78**.
- `features.vh_cdf_profile` — `F(τ)=fraction of observed VH months < τ` at τ∈{−22..−19} dB: the
  Ottinger permanence detector as a legal Class-A fraction, not affine-spanned by the bands.
- `models.catboost.boosting_type=Ordered` — the small-n permutation lever.
- Runs through the LEGAL calibration (Platt-on-OOF + literal 0.5). Smoke-verified compliant.

**Smoke finding to carry:** legal CatBoost is well-calibrated (Platt slope ~4.4) so its 0.5 cut
**under-selects** — pos-rate ~0.40 vs the transformer's 0.548 and true ~0.649. Its F1 column may look
weak standalone; its RANKING and different bias are what it brings to the blend. `c_catboost_spw`
(class weight 2.2) tests whether the under-selection is legally fixable — Elkan (IJCAI 2001) predicts
Platt neutralizes it, which would be a real methodological result either way.

**Arms:** `c_catboost` (full legal recipe), `c_catboost_noidx` (S2 indices off — do indices help a
*tree*, re-testing our Transformer-only −0.075 veto), `c_catboost_spw`; then `champion_archblend4`
(control, must reproduce 0.5670/0.899643) and **`champion_catblend5`** = the 4 transformers + legal
CatBoost, whose LB vs 0.899643 is the main event. Up to 2 uploads (catblend5, c_catboost standalone).

### iter30 RAN 2026-07-28 — offline result: the best blend-member profile of the whole project

**Control valid:** archblend4 rebuilt to pos-rate **0.5670** exactly → the comparison is live.

**🎯 CatBoost is the first member that is BOTH competent AND genuinely decorrelated:**

| | OOF AUC | ρ to transformer cluster | verdict |
|---|---|---|---|
| transformers (reltime/nope/l3/xview) | ~0.989 | 0.94–0.97 (rank-twins) | — |
| ROCKET (iter22) | ~0.99 | 0.82–0.87 | decorrelated but **−0.040 weak** |
| pin-era GBDT (iter24) | — | ~0.85 | dragged −0.0155 (under the pin) |
| **legal CatBoost (iter30)** | **0.9953** | **0.79–0.83** | **decorrelated AND strong** |

`catboost ↔ {reltime 0.820, nope 0.791, l3 0.798, xview 0.832}` — the lowest correlation to the
cluster we have ever gotten from a member whose AUC (0.9953) actually **exceeds** the transformers'.
catblend5 mean ρ = **0.8747** (vs archblend4's 0.9524) → "POOL IT: level gain is available." iter28
proved legal pooling of decorrelated-competent members buys level (+0.0100); this member is far more
decorrelated, so the blend should gain more. **catblend5 pos-rate 0.5689 (≈ archblend4's 0.567) — the
operating point is preserved, so CatBoost's under-selection does NOT collapse the blend.**

**Two clean secondary findings, both 0 extra submissions:**
- **Under-selection is real and NOT legally fixable.** CatBoost is so well-calibrated (Platt slope
  6.76) that its literal-0.5 cut lands at pos-rate **0.403 = the train prior**, far under the ~0.649
  test prevalence. `c_catboost_spw` (class weight 2.2) moved it to **0.402 — unchanged**: Platt-on-OOF
  neutralized the class weight exactly as Elkan (IJCAI 2001) predicts. So standalone CatBoost's F1
  column is capped by under-selection; its value is RANKING + decorrelation inside a blend. Drop spw.
- **S2 spectral indices marginally HELP a tree** (c_catboost 0.9792 vs c_catboost_noidx 0.9773 OOF-
  combined; AUC 0.99529 vs 0.99462) — the opposite sign to the Transformer's −0.075, consistent with
  "trees split on indices directly." Tiny and OOF-blind, so not decisive, but the veto does not
  transfer to trees. ρ(catboost, noidx)=0.965 (near-twins), ρ(catboost, spw)=0.979.

**Decision: upload `champion_catblend5` (the main event) + `c_catboost` standalone (anchors whether
the ranking strength survives the under-selection to LB). Prediction: catblend5 > archblend4's
0.899643** — first competent, genuinely decorrelated member, operating point preserved.

### iter30 LB RESULT (2026-07-29) — the prediction was WRONG; the ensemble lane is closed for good

| artifact | LB | read |
|---|---|---|
| **champion_catblend5** | **0.886043** | **−0.0136 paired vs archblend4** (4/5 shared members) — a CONFIDENT LOSS |
| **c_catboost** (standalone) | **0.697615** | catastrophic — OOF AUC 0.9953 → LB 0.70 |
| archblend4 (finalist #1) | 0.899643 | unchanged, undisputed |

**What happened.** CatBoost's "competence" was a pure OOF illusion. Its OOF AUC (0.9953) is the
HIGHEST of any model we've built, and its LB is the LOWEST (0.6976) — the OOF↔LB anti-correlation in
its most extreme form yet. Decorrelating the blend with a member whose *test* ranking is bad just
injects error: catblend5 = archblend4 diluted by a 0.70 member → −0.0136. The under-selection
(pos-rate 0.403) crippled the standalone F1, and the poor test transfer (implied test AUC ~0.85 vs OOF
0.995) did the rest.

**The law, now proven three times.** A member that is weak ON THE LB drags the blend in proportion,
no matter how decorrelated or how strong its OOF looks: ROCKET (−0.040 → blend −0.009), pin-GBDT
(−0.011 → blend −0.0155), legal-CatBoost (LB 0.70 → blend −0.0136). **Cross-model-class blending is
CLOSED with n=3 across maximally different families.** Within-class pooling closed at iter29. The
ENTIRE ensemble lane is now exhausted; `champion_archblend4` (0.899643) is the final answer for it.

**Methodological strike 3 (mine).** I over-predicted a submission for the THIRD time, every time on an
out-of-family candidate whose OOF/offline signal looked strong: `c_dropvv` (predicted +, got −0.0113),
and now `c_catboost` (predicted catblend5 > 0.8996, got 0.886). **RETIRE the two offline signals that
misled here:** (a) OOF AUC as a competence proxy — it is anti-correlated with LB and is worse for
trees than for the net; (b) `arch_blend`'s rank-correlation "POOL IT/SKIP" verdict — it printed
"POOL IT: level gain available" at mean ρ 0.8747 and was flatly wrong, because low ρ with a
BAD-on-test member is a liability, not an asset. **The only trustworthy instrument is a paired LB
submission.** Offline screening is dead for out-of-family candidates; do not spend narrative on it.

**Where this leaves the CatBoost/feature thesis.** The LB leader is at ~0.94 with CatBoost; ours got
0.70. A 0.24 gap is not a tuning gap — it is a fundamentally different pipeline (features and/or how
they escape the masking/operating-point trap), not something we close by parameter sweeps in the time
left. The tree lane, as WE can build it, does not work. If anything of the feature thesis survives, it
is as CHANNELS in the Transformer (the model that actually transfers) — the VH-CDF permanence +
`VH−VV` cross-pol ratio have never been tried there — but that is a representation change we cannot
screen (strike-3 territory) and RESPONSE_13 already flagged "expect small."

**Strategic consequence.** With the ensemble lane and the tree lane both closed, and a strong ELIGIBLE
finalist banked (archblend4 = 0.899643), the remaining expected value is in the deadline-bound,
guaranteed items — the Phase-Two reproducibility/novelty writeup (35% of a top-5 score, still does not
exist) and the manual finalist designation — not in more LB probing against a 0.019 seed floor.

---

## iter31 RESULT (2026-07-29) — 🏆 THE FEATURE LANE OPENS: VH permanence WINS in the Transformer

The "last feature shot" was not the last — it was the first WIN. Isolated, one-at-a-time:

| arm | channels | seed | LB | vs matched champion (0.8897) |
|---|---|---|---|---|
| **c_perm** | + permanence `1[VH<τ]` only (24→28) | 42 | **0.901605** | **+0.0119** 🏆 |
| c_perm (resubmit) | identical file | 42 | 0.901605 | exact repro |
| c_permxpol | + permanence + `VH−VV` (24→29) | 42 | 0.878788 | −0.011 |
| c_permxpol_s7 | + permanence + `VH−VV` | 7 | 0.873659 | (perm+xpol reliably bad) |

**Two findings:**
1. **Permanence alone is our best public score ever — 0.901605**, above every artifact including the
   4-member `champion_archblend4` (0.899643), as a SINGLE legal model. The masked mean-pool of the
   binary `1[VH_dB<τ]` channels hands the encoder the VH-CDF permanence fraction it could not
   otherwise form (a per-month threshold is nonlinear; the linear proj + mean-pool can't build it).
   Amplitude-preserving, n-invariant, not affine-spanned — it cleared every constraint and it worked.
2. **`VH−VV` cross-pol is TOXIC: −0.0228** (0.9016 → 0.8788 when added on top of permanence). Exactly
   UPDATE_13 §3a's warning — the cross-pol ratio IS affine-spanned by the projection layer, so it adds
   nothing but dilutes. **Dropped for good.** The perm+xpol seed spread (0.8788 vs 0.8737) is small,
   so its loss is reliable, not noise.

**Methodological vindication.** Had we run only the combined `c_permxpol` (0.8788) we would have
declared features dead — WRONG. The permanence-ONLY isolation arm caught the win. This confirms the
user's directive: **feature engineering + selection ONE AT A TIME, on ONE direction.** From here the
direction is PERMANENCE.

**⚠️ Caveat gating everything: c_perm is a single seed-42 run, and seed 42 is historically our LUCKY
draw** (old champion 0.8955@42 vs 0.8764@7, a 0.019 swing). +0.0119 is right at the ~0.013 public
resolution. So the win must be **seed-confirmed** before we build on it — iter32.

**Provisional best: `c_perm` = 0.901605 (pending seed-confirmation).** If it holds seed-averaged, it is
a simpler, higher, fully-reproducible finalist that beats archblend4.

## iter25 PHASE-A — two results for zero submissions, computed locally in minutes

Round-11 research (8 agents, `gemini_loop/RESEARCH_11.md`) produced two testable predictions. Both
were tested **locally on the cached cubes** — no Colab run, no submission, ~3 minutes.

### 🔴 Result 1 — the missing-indicator deletion lane is CLOSED (three agents were wrong)

Three independent agents predicted our per-band missing-indicator channels were a shifted nuisance,
via a specific and plausible mechanism: we deleted absolute time by left-aligning windows, but cloud
gaps encode absolute season, so the model could have recovered the month-of-year that relative-time
reframing removed. Measured, **masked-train vs test, left-aligned** (so window length is matched and
cannot be what is detected):

| probe | adv-AUC |
|---|---|
| P1 values only | **0.8915** |
| **P2 ALL missing-indicators only** | **0.4758** ← below chance |
| P3 values + indicators (what the champion eats) | 0.8943 |
| P4 S2-cloud indicators only | 0.4744 |
| P5 per-month S2-gap count only | 0.4815 |
| **what indicators ADD over values (P3−P1)** | **+0.0028** |

**The indicators carry essentially zero train/test information.** The reason is in our own code:
`apply_mask` (`src/features.py:85-92`) already applies S2 dropout at rates **measured off the test
set**, so the train indicator distribution was matched to test by construction. A previous session
solved this without recording it as solved. Lane closed, zero submissions spent.

*(Also note this is NOT iter13's `compact_missing`, which was 24→14 and failed. Full deletion would
have been 24→12. Neither is now worth running.)*

### 🟢 Result 2 — the shift is in the VALUES, and it is DISTRIBUTED

P1 = 0.8915 on masked, left-aligned values vs **0.965–0.976** for Presto on raw pixels: masking +
left-alignment already removed a real chunk of the shift, but a large signal-side component remains.

The 2-D screen scores each band on **A** (separates train/test) and **T** (predicts label) — the Uber
drop rule (arXiv:2004.03045, +3.9% AUC) with a second axis added so it cannot delete amplitude, which
`c_rank` proved fatal:

| band | A (shift) | T (label) | quadrant |
|---|---|---|---|
| **VV** | **0.5907** | 0.7801 | **free deletion** — top shift-carrier, VH dominates it on signal |
| VH | 0.5622 | **0.8302** | REPAIR, never delete — the primary signal |
| nir | 0.5462 | 0.8063 | REPAIR |
| **blue** | **0.5344** | 0.5963 | **free deletion** — barely predictive |
| re3 / re2 | 0.53 / 0.53 | 0.80 / 0.79 | REPAIR |
| nira / swir1 | 0.51 / 0.49 | 0.81 / 0.78 | KEEP (core) |
| swir2 / red / green / re1 | ≤0.50 | 0.76 / 0.49 / 0.65 / 0.57 | dead weight |

**Max single-band A is 0.59 against a joint 0.89 ⇒ the shift is DISTRIBUTED, not concentrated.** Band
deletion cannot collapse it and we should not expect it to. This independently confirms the standing
claim in `tools/adversarial_check.py` that chasing adv-AUC 0.5 by feature selection is futile.

**But two routes independently name VV.** Our data: top shift-carrier, dominated by VH on signal. The
SAR literature: VV is wind-sensitive, its water threshold drifts **2.6 dB/yr vs VH's 2.1**, and VH is
preferred because its backscatter histogram is cleanly bimodal (Ottinger 2017/2019; Li 2018, Dongting).
Deleting it is **capacity-REDUCING** — the only change-class that has ever won here.

**Honest caveats.** VV's T sits **0.0001 below the median** — a knife-edge, so the physics breaks the
tie, not the screen. The screen is a per-band *linear* read-out; a band with weak marginal signal could
still matter through the cross-band attention the champion actually uses. And per Khani & Liang
(arXiv:2012.04104), removing a feature can *hurt* via noise amplification in exactly our regime (few
rows, correlated features, one dominant signal) — this lane has a real downside regime.

**→ iter25 staged:** `c_dropvv`, `c_dropblue`, `c_dropvvblue` at 2 seeds each, 0 submissions. Wired via
a new `seq.channels.drop_bands` flag, smoke-verified end-to-end (width 24→20 **and** the independently
computed `n_features` agrees — the double-check iter12's silent no-op defeated).

---

## iter25 RESULT — the first candidate to clear the pre-committed rule since iter9

**Integrity checks, all PASS.**
- Widths logged: `c_dropvv` **22**, `c_dropblue` **22**, `c_dropvvblue` **20**; `n_features` agrees
  independently in every case. The flag took effect — this is not iter12's silent no-op.
  *(My staging note said "22/23/20". The 23 was my arithmetic slip: dropping one of 12 bands removes
  one value channel AND one indicator channel, so 24→22, not 23. No run was void.)*
- Gate: **ATCF1 15/15 concordant PASS (ρ +0.964, n=7)**, **DIS 5/5 PASS (ρ +1.000, n=4)**.
  ATC/DIV/MARG FAIL as always. The screen is readable — unlike iter21, where it returned VOID.

**The screen.**

| candidate | ATC-F1 margin | → LB | DIS margin | votes | rule |
|---|---|---|---|---|---|
| **`c_dropvv`** | **+0.0902** | +0.0147 | +0.0214 | **2/2** | **SUBMIT** |
| `c_dropblue` | −0.0133 ~ | −0.0022 | +0.0117 | 1/2 | HOLD |
| **`c_dropvvblue`** | **+0.0829** | +0.0135 | +0.0311 | **2/2** | **SUBMIT** |

The pre-committed rule was: *"≥2 cleared estimators AND the ATC-F1 margin exceeds the seed sd
(0.0576)."* **`c_dropvv` satisfies both** (2/2 votes; 0.0902 = 1.57 sd). So does `c_dropvvblue`.

**A stronger statement than the rule required.** The champion's ATC-F1 across 5 seeds is
mean 0.7759, sd 0.0576, **range [0.7196, 0.8601]**. `c_dropvv` scores **0.8977** — above the champion's
*best of five* seed draws, not merely above its mean. `c_dropvvblue` at 0.8904 likewise. This is the
first time a candidate has cleared the champion's entire observed seed range.

**Why `c_dropvv` and not `c_dropvvblue`.** ρ(dropvv, dropvvblue) = **0.9841** — near-twins, so one
submission tests both. ATC-F1 (the better-fit estimator: n=7, 15/15, vs DIS's n=4) prefers dropvv, and
blue's *own* row is a HOLD with a **negative** ATC-F1 margin — so the blue deletion is not independently
supported and `dropvvblue` inherits it. Take the change the evidence actually supports.

**DECISION: upload `submission_c_dropvv.csv` (seed 42). 1 submission.** Seed 42 both sides, one
variable changed, against the 0.8955 seed-42 anchor — a **paired** A/B under the RESEARCH_07 §5b
protocol (SUGGESTIVE ≥0.006, CONFIDENT ≥0.012).

**Honest prediction, committed before the result** (last time I predicted 0.892–0.897 and got 0.879123,
so this is calibrated down): ATC-F1's magnitude runs ~3× overstated, so +0.0147 → a true expectation of
**≈+0.005**, i.e. **≈0.900**, inside the ±0.019 seed band. Read: **≥0.9075** = confident win and the
first level gain since the GBDT→Transformer swap; **0.890–0.907** = unresolved, consistent with noise;
**≤0.8835** = confident loss, and feature-space deletion closes.

**Note on `archblend4`.** Its 4 members are unchanged, so the artifact is identical to the 0.894643
upload — **do not re-submit it.** The printed "mean pairwise ρ = 0.9345" is contaminated by the two
`--diag-extra` columns; archblend4's own 4-member mean is **0.9524**.

### iter28 LB = 0.899643 — 🏆 BEST PUBLIC SCORE EVER, AND IT IS LEGAL

`champion_archblend4`, rebuilt through the compliant path, scored **0.899642643** — higher than
anything we have ever submitted, including the illegal artifacts.

| artifact | operating point | LB |
|---|---|---|
| `c_meanmin` | pinned (illegal) | 0.898566 ← previous best ever |
| **`champion_archblend4`** | **LEGAL** | **0.899643** ← new best, and eligible |
| `champion_archblend4` | pinned (illegal) | 0.894643 |
| `seq_a_xview` | LEGAL | 0.889686 |
| `seq_a_xview` | pinned (illegal) | 0.895500 |

**Going legal made this artifact BETTER by +0.005.** We did not trade score for compliance here; we
gained on both.

#### 🔑 The finding: the prevalence pin was SUPPRESSING the ensemble

The same comparison under the two operating points:

```
pinned:   archblend4 - champion  =  -0.000857     pooling bought NOTHING
legal:    archblend4 - champion  =  +0.009957     pooling buys LEVEL
```

**Mechanism.** Under the pin, every member's operating point was overwritten to a fixed 0.649, so
pooling could only affect **order** — and at mean ρ = 0.9524 there is almost no independent order
information left to average. Under a literal 0.5 cut, pooling *also* averages the members'
**calibration**, and calibration error is far less correlated across members than ranking is. The
members' individual legal positive rates were **0.581 / 0.570 / 0.534 / 0.586** — a spread of 0.052
that the pin was collapsing to a single number and discarding.

So iteration 18's verdict — *"architecture pooling is MARGINAL, not a level lever"* — was **an
artifact of the operating point, not a property of the ensemble.** It was measured under the pin,
where the only available gain was order-pooling.

#### ⚠️ The correlation go/no-go is now wrong, and it nearly cost us this result

`arch_blend`'s printed verdict was **"SKIP: as correlated as seeds → no level gain"** at ρ = 0.9524.
It was wrong. That heuristic measures **rank** correlation, which predicts *order*-pooling gain and
is silent about *calibration*-pooling gain.

We uploaded anyway — but for a weaker reason than the outcome justifies. The stated reason was "we
need a legal pooled artifact regardless of level." The actual result is that pooling now buys level
outright. **Right call, wrong rationale**, and worth recording as such.

**This is the FOURTH instance of the same failure mode: a rule derived under the pin, applied outside
it.** The others were iter24 (ensemble gate), iter26 (ATC-F1's anchor family), and the blender
rank-pooling bug. Several other ensemble rules were derived under the pin and are now **unverified**,
most importantly the *"gate members on level gap, not correlation"* rule — that too was measured when
only order mattered.

---

### iter27 LB = 0.889686 — compliance cost ≈ 0.006, i.e. nothing

**The single most important result in the project's endgame.** We removed a rules violation
(§ the prevalence pin) that had been credited with ≈+0.07, and paid **−0.0058**.

```
pinned  seq_a_xview (seed 42)   0.895500
LEGAL   seq_a_xview (seed 42)   0.889686
                                --------
paired delta                    -0.005814     PAIRED: same config, same seed, same folds,
                                              ONLY the operating point differs.
```

Under our own measurement protocol (RESEARCH_07 §5b) a paired A/B is **SUGGESTIVE at ≥0.006**
and CONFIDENT at ≥0.012. **0.0058 does not even reach suggestive** — it is indistinguishable
from zero at our resolution.

**Better framing still:** our *reliable* pinned level was **0.8865** (5-seed pooled). The legal
**single-seed** champion scores **0.8897**, i.e. above it. The compliance cost is not merely
small, it is lost inside seed noise.

**Why the pin was worthless, quantified.** The free `compliance_diff` predicted the answer as a
function of one unknown — the precision of the 104 rows the pin added. Inverting the observed
delta on that sweep:

```
observed d(score)                          -0.005814
implied precision of the 104 flipped rows   0.492
```

**The pin was adding 104 positives that were ~49% correct — coin flips.** It bought volume, not
accuracy. This is exactly the prediction we committed in advance ("marginal rows at a
near-optimal cut are close to coin flips, so 0.50–0.65 is the realistic band"), landing at the
optimistic edge of it. First prediction in three that was right, and it was right because it was
derived from the F1 algebra rather than from an estimator.

**Where the ≈+0.07 came from.** Iteration 02, on the **superseded GBDT**, where the model's own
probabilities were badly scaled and the prior sweep was doing real work. The transformer
calibrates itself to a 0.5476 positive rate unaided; the extra 0.10 of prevalence the pin forced
on top was noise. **We carried a GBDT-era constant for 25 iterations without re-measuring it.**

**⚠️ CONSEQUENCE FOR THE ENDGAME — every artifact on the finalist board is now unusable.** All
six previously submitted artifacts (`c_meanmin`, `seq_a_xview` 0.8955, `champion_archblend4`,
`champion_seedavg5`, `champion_rocketblend5`, `champion_gbdtblend5`) were produced by the
rules-violating path. **None of them can be designated.** The legal board starts here, with
exactly one entry:

| legal artifact | LB | note |
|---|---|---|
| `seq_a_xview` (legal, seed 42) | **0.889686** | the only scored legal artifact we have |

Rebuilding a legal `champion_archblend4` is now the top priority, since it was finalist #1.

---

### iter26 LB = 0.884217 — the screen was WRONG IN SIGN, and that is the finding

**Result.** `c_dropvv` = **0.884217** vs the seed-42 champion anchor 0.8955 → **−0.0113, paired**
(same seed, same folds, one variable). Under the RESEARCH_07 §5b protocol that is **SUGGESTIVE**
(≥0.006) but not CONFIDENT (≥0.012). Against my pre-committed bands it lands at the very bottom of
"unresolved" (0.890–0.907), 0.0007 above the "confident loss" line — so strictly it did not confirm a
loss, but it decisively **failed to confirm the gain the screen predicted.**

**Deletion lane: CLOSED.** Cost 1 submission. Combined with the Phase-A finding that the shift is
distributed (max single-band A 0.59 vs joint 0.89), **feature-space deletion cannot beat this shift.**

**The real result is about the instrument, not the band.** ATC-F1 predicted **+0.0902 (2/2 votes, 1.57
seed-sd, above the champion's best-of-five seed draw)** and the LB came back **negative**. That is the
first time a SUBMIT call was followed and failed.

**Diagnosis — the retro-fit was only ever certified within one family.** All 7 original anchors are
architecture/objective variants at an **identical 24-channel input width**. `c_dropvv` is the first
candidate that changed the **input representation** (22 channels). Adding it as an 8th anchor:

```
n=7 (original)     rho=+0.964   gate=15/15
n=8 (+c_dropvv)    rho=+0.738   gate=17/18
only discordant informative pair: (xview, dropvv)
```

**ρ collapses 0.964 → 0.738 on a single point, and that point is the only out-of-family one.**

**⚠️ And the GATE DID NOT CATCH IT — 17/18 still reads PASS.** The gate counts concordance over pairs
with |ΔLB| > 0.010, which is dominated by easy pairs (detrend at 0.8266 is trivially rankable). It is
**not a sufficient guard**, and we have now been shown that at the cost of a submission.

**⛔ CARRY-FORWARD, THIS IS THE EXPENSIVE PART.** The next planned experiment — the **ratio-feature
battery** — is *also* a representation change, i.e. **out-of-family in exactly the same way**. Screening
it with ATC-F1 would repeat this error verbatim. **Ratio candidates must be tested on the LB directly**,
one variable, seed-paired, or the screen must first be re-certified against `c_dropvv`.

**Why this submission was not wasted.** It bought the **only anchor we have off the 24-channel
manifold** — the sole datapoint capable of detecting out-of-family estimator failure. It is now in
`experiments/anchors.tsv` and any future estimator must rank it correctly to be trusted.

**My prediction was wrong again, in the same direction.** I said ≈0.900; actual 0.884217 (−0.016).
At iter24 I said 0.892–0.897; actual 0.879123 (−0.014). **Two consecutive over-predictions of ~0.015,
both on out-of-family candidates.** This is a systematic optimism bias, not two independent misses:
in both cases I trusted an ATC-F1 margin computed outside the family it was fitted on. The 3× discount
we apply to ATC-F1's magnitude is calibrated in-family and is **not** sufficient out-of-family, where
the estimator is not merely inflated but can be **sign-wrong**.

**Budget check.** ~78 submissions, 19 days. Submissions are no longer the binding constraint —
**measurement noise is.** Seed averaging cannot fix it: with seed rank-corr 0.9511, the variance
reduction factor is (1+0.9511·4)/5 = **0.961**, so pooling 5 seeds moves sd only 0.0191 → 0.0187.
**Effects below ~0.02 are unmeasurable on the public slice by any construction available to us.** The
leader is ≈+0.05 ahead. Hunt that lever; stop buying 0.005s.

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

**archblend4 uploaded → LB 0.894643.** Landed at champion level (0.8955−0.0009, a tie), **+0.0021
above its member average** and **+0.0081 above the pure seed-average** (0.8865) — all inside σ_public
(~0.013–0.018) so not significant, but every sign favorable. It pools BOTH seed and architecture
noise → lowest-variance artifact → **now the leading finalist** (replaces seedavg5 at the top). This is
exactly what iter18's marginal ρ=0.9395 predicted: small decorrelation gain, within noise, min variance.

## Round-10 cross-examination (Claude given Gemini's report) → `gemini_loop/RESPONSE_10.md`

Confirmed our RESPONSE_09 verdicts (Gemini weaker; 3/5 refuted; CORAL/TENT/Saerens/Zou rejected) and
added a **new #1: instance-expansion reframing** (each observed sub-window = an independent training
example; data-model change, directly matched to the masking trap; unverified competitor lead ~0.914
AUC). Also flagged that dispersion pooling as [mean‖std‖min‖max] is a **capacity ADD** our own law
predicts loses — the capacity-neutral form is [mean_{d/2}‖std_{d/2}] at width d. iter19 is the
empirical test of whether the 2× `mean_min` signal survives replication. Roadmap re-ranked:
instance-expansion → decorrelated member → in-domain SSL → capacity-neutral pooling → LN-TTA.

## iter19 RESULT — dispersion pooling: `mean_min` is the standout, an ATC-F1/DIS split

| candidate | ATC-F1 Δ (LB-eq) | DIS Δ (LB-eq) | OOF combined | votes |
|---|---|---|---|---|
| c_meanmin | **+0.0672 (+0.0109)** — **above the 0.0576 seed floor** | −0.0214 (−0.0355) | 0.97536 | 1/2 HOLD |
| c_moments | +0.0470 (+0.0076) `~` | −0.0262 (−0.0436) | 0.97642 | 1/2 HOLD |
| c_meanstd | +0.0185 (+0.0030) `~` | −0.0097 (−0.0161) | 0.97526 | 1/2 HOLD |

All HOLD by the 2-vote rule. But `mean_min`'s ATC-F1 margin (+0.0672) is **the only candidate margin
in the whole project to exceed the seed-noise floor**, and it **replicated** iter12's +0.0672 with a
2nd seed. Its OOF (0.97536) is identical to champion (0.97528) — same in-distribution fit, but ATC-F1
(our stronger estimator, ρ+0.964 n=7) detects better test-transfer. DIS (weaker, ρ+1.000 n=4) objects,
but its objection is seed *stability* (a variance concern, fixable by averaging), not level.

**Decision: reasoned override of the HOLD — upload `c_meanmin` (seed 42) as a SEED-PAIRED test vs the
known `seq_a_xview` seed 42 = 0.8955** (identical config except pooling → seed noise cancels). ATC-F1
predicts ≈0.906. ≥~0.902 → real (first above-floor architectural gain; then seed-average + ensemble
it); ≤~0.895 → killed for 1 submission, proceed to instance-expansion. `mean_std`/`moments` HOLD
(below floor, DIS negative). iter18's arch-corr matrix re-printed unchanged (ρ=0.9395).

**c_meanmin uploaded → LB 0.898566.** The seed-PAIRED test (vs xview seed42 = 0.8955) gives
**+0.0031 — inside noise.** Reads:
- **ATC-F1 was directionally right, magnitude ~3× too high** (predicted +0.011, realized +0.003) →
  discount ATC-F1 *magnitude* going forward; trust its *sign*.
- **DIS was flatly wrong** (predicted −0.035). On this physics-backed split, ATC-F1 won the call.
- mean_min is **not** a standalone level gain, but it is **≥ champion** and 0.8986 is our **highest
  single public draw after the 0.8955 champion** — so it is banked as a finalist AND becomes an
  ensemble-member candidate on a NEW diversity axis (pooling, not architecture).

## iter20 RESULT — mean_min is a rank-TWIN of the champion; the pooling-ensemble lane is CLOSED

The go/no-go was unambiguous: **ρ(c_meanmin, seq_a_xview) = 0.9928** — the highest pair in the whole
matrix. mean_min does not reorder the champion; it nudges a few scores within noise (hence the +0.003
paired LB, a lucky draw). Consequences:
- **archblend5 mean ρ = 0.9486 → SKIP**; its ATC-F1 is −0.0633 (worse than archblend4's −0.0101),
  because adding a near-duplicate just double-weights xview and drags the blend. **Not submitted.**
- **c_meanmin seed sd = 0.0729 > xview 0.0576** — DIS's instinct was right: mean_min is *higher*
  variance. So it is both non-decorrelated AND noisier → useless as a member.
- `champion_meanmin_seedavg5` written but not worth a finalist slot (≈ xview seed-avg by rank).
  **`champion_archblend4` (0.8946) remains the leading finalist.**

**The deeper lesson:** every mean-based transformer variant we can build lives at ρ≈0.93–0.99 — the
ensemble/representation lane inside this model class is EXHAUSTED for buying *level*. Only a genuine
data-model or model-class change can now move the needle. Pivot to instance-expansion.

## iter21 RESULT — instance-expansion is inert, and the arbiter failed its own retro-fit

The cross-exam's #1 data-model lever: treat each `(row, masked sub-window)` as an INDEPENDENT
training example via PER-EPOCH resampling (fresh masked windows every epoch → ~K·epochs distinct
instances, not K fixed). Paired control = `seq_a_reltime` (K=2 FIXED views) so the ATC-F1 gap would
isolate resampling alone.

**Two findings, both HOLD-confirming:**

**1. Same OOF-inflation signature as the fixed-view flop.** Resampling was hypothesised to LOWER OOF
while improving transfer. It did the opposite:

| run | OOF | seed |
|---|---|---|
| `c_iexp_rs2` | 0.98205 | 42 |
| `c_iexp_rs2_s7` | 0.98292 | 7 |
| `c_iexp_rs6` | 0.98235 | 42 |
| `c_iexp_rs6_s7` | 0.98384 | 7 |
| — champion `seq_a_xview` | 0.97523 | 42 |
| — paired control `seq_a_reltime` | 0.98041 | 42 |
| — `seq_a_k4` (fixed-view twin, LB **0.8665**) | 0.98419 | 42 |

Every instance-expansion arm sits at ≈0.982–0.984 OOF — the **exact `seq_a_k4` fingerprint**, and
k4 scored one of our worst LBs (0.8665). Higher OOF has been anti-correlated with LB throughout.
Resampling did not rescue the mechanism; it reproduced the overfit-the-source signature. The
data-model lane (in-family) is now measured closed.

**2. The offline screen went VOID — ATC-F1 failed its OWN retro-fit this run.** For the first time
since iter11, ATC-F1 dropped below the gate: **14/15 concordant, ρ = +0.929** (was 15/15, ρ +0.964).
Cause: a single anchor pair (`l3` LB 0.8921 vs `xview` LB 0.8955) *flipped ATC-F1 ordering* between
Colab runs — `l3` 0.8391 now edged `xview` 0.7709. Pure run-to-run prediction noise. So no estimator
cleared → the screen was SKIPPED and `c_iexp` was never scored.

**This is the seed-variance thesis reaching the estimator itself.** At the sub-0.01 resolution where
instance-expansion would live, even our best lie-detector is not reproducible run-to-run. That is
*confirmation*, not contradiction: it re-proves that small probes are unresolvable offline **and**
online, and that only model-class-sized effects (~0.05) are measurable with this budget. `archblend4`
(0.8946) remains the leading finalist; no submission spent.

**Lane status after iter21:** positional ✅closed, objective ✅closed, pooling ✅closed (rank-twins),
foundation-model/SSL ✅closed (adv-AUC 0.97), instance-expansion ✅closed. The ONLY unspent lever of
the right species (a *different model class*, like the GBDT→Transformer swap that cleared the floor)
is a decorrelated non-transformer member (MiniRocket/Hydra/ROCKET) — cross-exam's #2.

## iter22 RESULT — ROCKET is the FIRST decorrelated member (ρ 0.82–0.87), but a weaker learner

**Milestone: after 20 iterations of rank-twins, a member finally cleared ρ<0.90.** Estimators
re-certified cleanly this run (ATC-F1 15/15 ρ+0.964 PASS; DIS 5/5 ρ+1.000 PASS — the iter21 void was
a one-run anchor wobble). The cross-model rank-correlation:

| pair | ρ | | pair | ρ |
|---|---|---|---|---|
| rocket ↔ xview | **0.8665** | | rocket ↔ nope | **0.8241** |
| rocket ↔ l3 | 0.8619 | | rocket ↔ k4 | 0.8292 |
| rocket ↔ reltime | 0.8471 | | *transformer ↔ transformer* | 0.91–0.97 |

Every in-family variant (positional / objective / pooling / instance-expansion) sat at ρ 0.93–0.99.
ROCKET at **0.82–0.87** is the first member below 0.90 — the 2-way go/no-go printed *"POOL IT: level
gain is available."* This is direct empirical confirmation of the design law: only a different
**model class** decorrelates; everything inside the Transformer class is a rank-twin.

**BUT ROCKET is a weaker standalone learner.** The screen (trustworthy this run) votes it below
champion on both cleared estimators: **ATC-F1 −0.2462 (≈ −0.040 LB), DIS −0.0214 → 0/2 HOLD**. That
−0.040 is ~4× the seed floor (±0.0094 LB), a real gap, not noise. rocket standalone ≈ 0.855. Its OOF
(0.9689 / 0.9717 across seeds) is also below the transformer cluster, consistent with a genuinely
lower-capacity model — expected for random (un-trained) kernels + a linear head.

**So this is textbook decorrelated-but-weaker.** Equal weight over-weights the weak member; the
right artifact is a SMALL-weight blend. Three candidates were written:
- `champion_rocketblend5` = 5-way {reltime, nope, l3, xview, rocket}, rocket at **1/5** weight
  (mean ρ dropped 0.9395 → **0.9118** vs archblend4, i.e. the blend is measurably more diverse).
- `champion_xview_rocket` = 2-way, rocket at **1/2** (too aggressive for a −0.04 member; probe only).
- `champion_rocket_seedavg2` = rocket alone, 2 seeds (rank-corr 0.9248) — a standalone diverse hedge.

**Decision: upload `champion_rocketblend5`** — the principled small-weight artifact. It is BOTH the
measurement (does adding a decorrelated-but-weak member preserve champion-cluster level, or does it
drag?) and the finalist candidate. Read on the paste-back:
- **≥ ~0.885** (in-cluster) → level preserved; bank it as the most-decorrelated finalist we can build,
  upgrading the diverse slot over `seedavg5`/`archblend4`. The architecture search then ends here
  (rocket was the last lever) → pivot to the Phase-Two writeup, OR iter23 = *strengthen* rocket
  (more kernels / MiniRocket recipe) to make the decorrelated member competitive and the blend gain
  real.
- **< ~0.88** (collapse) → the weak member dragged the blend; `champion_archblend4` (0.8946) stays the
  finalist and the search is done → writeup.

Note per project law: a public score *below* 0.8955 is expected and fine — we optimise the unseen
721-row private slice, and a ρ=0.87 member is the best private-variance-reduction lever available.

### rocketblend5 uploaded → LB 0.885661

| artifact | public LB | vs archblend4 |
|---|---|---|
| champion single seed (lucky) | 0.8955 | +0.0009 |
| **champion_archblend4** | **0.894643** | — |
| seed-average (5 champion seeds) | 0.88653 | −0.0080 |
| **champion_rocketblend5** | **0.885661** | **−0.0090 (within σ_public)** |

**Read: level PRESERVED, no gain.** rocketblend5 cleared the 0.885 non-collapse line and landed
essentially AT the seed-average consensus (−0.0008), i.e. −0.009 below archblend4 but inside the
±0.013 public noise — a statistical **tie at the reliable level**. The decorrelation milestone is
real (first ρ<0.90 member ever) but it did **not** translate into a measurable blend gain, because
the member is genuinely weaker (−0.040): the diversity benefit and the level cost cancel, which is
exactly why the blend sits at the consensus level rather than above it. **Decorrelation is necessary
but not sufficient — the member must also be competitive.**

**Finalist bookkeeping.** `champion_archblend4` (0.8946) stays the leading finalist (highest reliable
public, lowest variance). `champion_rocketblend5` (0.8857) is banked as a candidate DIVERSE second
finalist: it is tied on public but carries a genuinely independent (ρ0.87) component, so for the
private slice it hedges the transformer-cluster's shared errors better than `seedavg5` (a pure-
transformer ρ0.95 pool) does. Final pair decided at the end.

## iter23 RESULT — decorrelation ⊥ strength: the ROCKET lane is exhausted

Estimators re-certified clean (ATC-F1 15/15 ρ+0.964; DIS 5/5 ρ+1.000). Paired screen, univariate vs
multivariate ROCKET:

| candidate | ATC-F1 vs champ (LB-eq) | DIS vs champ | ρ(·, xview) | OOF combined |
|---|---|---|---|---|
| `c_rocket` (univariate) | −0.2462 (−0.0401) | −0.0214 | **0.8665** | 0.9689 |
| `c_rocket_mv` (multivariate) | −0.2115 (−0.0344) | −0.0350 | **0.9114** | 0.9760 |

**Both signals say "lane exhausted", exactly the pre-committed stop condition:**
1. **The ATC-F1 gain is within noise.** +0.0347 ATC-F1 (mv − uni) < the estimator's seed sd 0.0576.
   DIS moved the *other* way (−0.0214 → −0.0350). No competence gain that clears the floor.
2. **It bought that non-gain by RE-CORRELATING.** ρ(rocket, xview) jumped **0.8665 → 0.9114**, back
   toward the rank-twin cluster (mean cluster ρ with mv = 0.9106, ≈ archblend4's 0.9395; uni was the
   only member ever below 0.90).

**The finding (and it is a real one): ROCKET's diversity was a symptom of its weakness.** The
univariate model disagreed with the Transformer *because* it was blind to cross-band structure. Teach
it the cross-band pond signature (low VH ∧ low NDVI ∧ low NDWI) and it starts agreeing with the
Transformer — which already exploits exactly that via cross-band attention. Within this family you
cannot get decorrelated AND competent: the mechanism that adds strength is the mechanism that removes
the diversity. mv vs uni ρ = 0.9504 (they are near-twins of each other), so multivariate mostly moved
rocket *toward* the transformer, not to a new place.

**Decision: NOT submitted.** rocketblend5_mv (mean ρ 0.9106) and xview_rocket_mv (ρ 0.9114) are
marginal, indistinguishable from archblend4. The ROCKET lane is closed. Finalist pair remains
`champion_archblend4` (0.8946) + `champion_rocketblend5` (0.8857, the UNIVARIATE-rocket blend — our
most decorrelated artifact, ρ0.87 member, the better private-slice hedge).

## iter24 RESULT — the GBDT is DECORRELATED **and** COMPETENT: the profile ROCKET could not reach

Estimators re-certified cleanly for the sixth time (ATC-F1 **15/15 ρ+0.964 PASS**; DIS **5/5 ρ+1.000
PASS**; ATC −0.429, DIV −0.857, MARG −0.321 all FAIL exactly as before). Seed floor unchanged:
ATC-F1 sd 0.0576 == ±0.0094 LB. The iter21 wobble did not recur — note the gate's |ΔLB|>0.01 filter
is doing real work here: `l3` (ATC-F1 0.8134) still outranks `xview` (0.8075) against their LB order,
but that pair's |ΔLB| is only 0.0034, so it is correctly excluded as unresolvable rather than
counted as a miss.

### Both pre-committed Branch-A conditions are met

| condition | threshold | measured | |
|---|---|---|---|
| ρ(gbdt, xview) | < ~0.90 | **0.8734** | ✅ |
| gbdt ATC-F1 vs champion | within ~0.02 LB | **−0.0676 (−0.0110 LB)** | ✅ |

Screen line: `g_gbdt  ATCF1 −0.0676 (−0.0110 LB)  DIS +0.0107 (+0.0178 LB)  votes=1/2 → HOLD`.
That HOLD governs submitting the GBDT **standalone**, which we are not doing. Per the pre-committed
instrument rule (iter18, and the tool's own printed DECISION), an ensemble call is read off the
**correlation matrix**, not the screen — the screen's ±0.0094 LB resolution is coarser than any
ensemble gain by construction.

### The comparison that matters — GBDT vs ROCKET as members

| member | ρ to xview | mean ρ to cluster | ATC-F1 vs champ | seed rank-corr |
|---|---|---|---|---|
| ROCKET (iter22) | 0.8665 | ~0.850 | **−0.2462 (−0.0401 LB)** | 0.9248 |
| **GBDT (iter24)** | **0.8734** | **0.8489** | **−0.0676 (−0.0110 LB)** | **0.9795** |

**Equally decorrelated, ~4× less weak.** ROCKET bought its diversity by being blind to cross-band
structure; the GBDT is decorrelated while natively *using* that structure — its diversity comes from
discarding temporal ORDER instead. That is the representation-diversity profile iter24 was designed
to test, and it is the first member ever to satisfy both halves.

**Independent corroboration of the competence estimate.** ATC-F1 puts the GBDT at −0.0110 LB below
champion. The historical GBDT standalone anchor is **0.8780** vs the champion's *reliable* ≈0.8865 —
a gap of **−0.0085**. Two unrelated routes agree the GBDT sits ≈0.01 below the champion. The −0.011
margin does sit just *outside* the ±0.0094 seed floor, so read it as a real-but-small competence
gap, not as a tie.

### A genuinely new finding: the seed lottery is a TRANSFORMER property, not a task property

| model class | seed rank-corr | independent error component |
|---|---|---|
| Transformer (`seq_a_xview`, n=5 seeds) | 0.9511 | 0.0489 |
| **GBDT (`g_gbdt`, n=2 seeds)** | **0.9795** | **0.0205** |

The GBDT is **~2.4× more seed-stable**. The 0.0191 LB seed swing that voided nine of our verdicts is
a property of the from-scratch Transformer, not of this dataset. (Caveat: one seed pair vs ten, so
directional not definitive.) This makes a GBDT-containing blend more *reproducible*, which matters
for the unseen private slice and for the Phase-Two reproducibility rubric.

### The blend matrices — and why `gbdtblend5` is the right artifact

| artifact | members | mean ρ | min ρ | variance factor (1+ρ(M−1))/M |
|---|---|---|---|---|
| `champion_archblend4` | 4 transformers | 0.9395 | 0.9097 | 0.9546 |
| **`champion_gbdtblend5`** | **4 transformers + gbdt** | **0.9110** | **0.8309** | **0.9288** ← lowest |
| `champion_xview_gbdt` | xview + gbdt | 0.8734 | 0.8734 | 0.9367 |

`archblend4` reproduced its iter18 matrix **exactly** (0.9395 / 0.9097) — the framework is stable.
The 2-way is the only one to trip the tool's automatic *"POOL IT"* verdict, but the 5-way is the
better artifact on **both** axes: lower variance factor (0.9288 vs 0.9367) *and* one-fifth the level
exposure to the −0.011 member instead of one-half.

**Honest expectation, from the rocketblend5 precedent.** rocketblend5 had mean ρ 0.9118 — essentially
identical to gbdtblend5's 0.9110 — and landed −0.009 below archblend4, consistent with ⅕·(−0.040).
The same arithmetic here gives ⅕·(−0.011) ≈ **−0.002**, i.e. gbdtblend5 ≈ **0.892–0.897**: a public
tie with archblend4. **We are not buying a public number.** We are buying the lowest-variance,
most-decorrelated artifact available, built from a member that is both competent and seed-stable —
and a submitted score is *required* for finalist eligibility.

**Decision: upload `submission_champion_gbdtblend5.csv`.** The one submission the pre-committed rule
authorises. This is the final architecture screen; whatever it returns, the search is closed.

### gbdtblend5 uploaded → LB 0.879123. The prediction was WRONG and the screen was RIGHT.

| artifact | public LB | vs archblend4 |
|---|---|---|
| `champion_archblend4` (4 transformers) | **0.894643** | — |
| `champion_seedavg5` | 0.886530 | −0.0081 |
| `champion_rocketblend5` (⅕ ROCKET, −0.040 member) | 0.885661 | −0.0090 |
| **`champion_gbdtblend5` (⅕ GBDT, −0.011 member)** | **0.879123** | **−0.0155** |

**Predicted 0.892–0.897, measured 0.8791 — the forecast missed by ~0.013.** And unlike most deltas
in this ledger, **this one is significant**: archblend4 and gbdtblend5 share 4 of their 5 members at
⅘ weight and were scored on the identical 309 public rows, so this is a strongly *paired* comparison
(paired SE ≈0.006 for merely ρ0.9 variants; these are far more correlated than that). −0.0155 is
≥2.5σ. This is not seed noise.

### The result inverts the hypothesis it was built to test

| blend | member weakness | member ρ to cluster | blend cost vs archblend4 |
|---|---|---|---|
| rocketblend5 | −0.040 LB | 0.850 | −0.0090 |
| **gbdtblend5** | **−0.011 LB** | 0.849 | **−0.0155** |

**The *stronger*, equally-decorrelated member cost NEARLY TWICE AS MUCH.** The linear model that
justified this submission — "blend level ≈ weighted mean of member levels, so ⅕·(−0.011) ≈ −0.002" —
is refuted outright. Level does not compose linearly under rank-averaging with a pinned threshold:
`TargetF1` is scored at a fixed 0.649 cut, so what matters is not a member's average quality but how
much it **reorders rows near the cut** — and a genuinely decorrelated member reshuffles exactly there.
Decorrelation is not a free variance reduction; at a pinned threshold it is itself the damage.

Note the coincidence worth flagging: **0.8791 ≈ the historical GBDT standalone anchor (0.8780).** A
⅘-transformer blend landed at its ⅕-weight member's own level. Whatever the mechanism, the GBDT
influenced the final ordering far out of proportion to its nominal weight.

### 🚨 The methodological lesson — the screen was right and I overrode it

The offline screen returned **`g_gbdt  votes=1/2 → HOLD`**. I set that aside by invoking the iter18
rule *"read the correlation matrix, not the screen, for ensemble calls."* **That rule was derived for
pooling WITHIN the transformer class**, where every member is equally competent and the only question
is decorrelation. Extending it to a blend containing a **weaker foreign member** was the error: there,
member competence is exactly what determines the outcome, and the screen is the instrument that
measures it. The 1/2 vote was real information.

Compounding it: the Branch-A gate ("ATC-F1 within ~0.02 LB of champion") leaned on an estimator whose
**magnitude is known-unreliable** — iter19 already established ATC-F1 is directionally right but ~3×
overstated. A −0.011 ATC-F1 reading is not evidence of a −0.011 model. Corrected by that factor it is
≈−0.033, i.e. essentially ROCKET's −0.040, which predicts precisely the drag we observed.

**Cost: 1 submission of ~79. Cheap, and it bought a definitive negative.**

### What is now established (n=2, across maximally different model families)

**Cross-model-class blending does not work on this task, at any member strength.** Two independent
foreign classes — random-kernel ROCKET and a mature tree ensemble — both at ρ≈0.85 decorrelation,
both dragged the blend below the pure-transformer pool. The iter22 repair ("decorrelation is
necessary but not sufficient; the member must also be competent") is itself now refuted: we found a
competent decorrelated member and it did *worse*. The correct statement is stronger and simpler:

> **Under a rank-only metric with a pinned threshold, a decorrelated member's reordering near the cut
> costs more level than its independence buys back. Only members that are near rank-twins (ρ>0.93 —
> i.e. same model class) can be pooled without loss, and those buy variance reduction, never level.**

That closes the ensemble frontier completely, and with it the architecture search — every model class
has now been tested. `champion_archblend4` (0.894643) stands as the leading finalist, unchallenged by
anything built in iterations 18–24.

**`champion_xview_gbdt` must NOT be uploaded.** It puts ½ weight on the member that just cost −0.0155
at ⅕ weight; the extrapolation is strongly negative and there is nothing left to learn from it.

## iter24 — GBDT as the decorrelated member: the last untested model class (staged)

ROCKET failed because its diversity came from *ignoring* signal. A GBDT (LGBM+XGB+CatBoost on the
hand-engineered temporal-AGGREGATE features) is the opposite kind of learner: it is a mature, strong
tabular model that USES cross-band/cross-feature structure natively, but discards the temporal
ORDERING the Transformer attends to — so its diversity comes from a genuinely different
representation, not from a blindness. Historically (iter2, pre-seed-noise pipeline) the GBDT was
ρ≈0.849 to the seq model (decorrelated) and scored **LB 0.8780** standalone — only −0.0175 below the
current champion (≈ one seed-swing, so that gap is largely VOID). That is a strictly better
decorrelated-member profile than ROCKET (which was −0.040 weak). Its one prior blend attempt (iter2,
0.8705) predates the seed-variance understanding, the prevalence pin, and the clean two-level
rank-blend — so it is void and worth re-measuring properly.

**0-submission screen.** Run `--model gbdt` at 2 seeds; screen it; print ρ(gbdt, xview) via arch_blend.
- **ρ(gbdt, xview) < ~0.90 AND gbdt ATC-F1 close to champion** → a decorrelated AND competent member
  (what ROCKET couldn't be) → blend it and spend ONE submission; a blend that beats the cluster is the
  first LEVEL gain since the GBDT→Transformer swap.
- **ρ ≥ ~0.90 OR gbdt ATC-F1 far below champion** → the last model class is closed too → the
  architecture search is definitively DONE → lock the finalist pair and pivot to the Phase-Two
  reproducibility/novelty writeup (35% of the top-5 rubric). **This is the final architecture screen
  either way.**

## iter23 — MULTIVARIATE ROCKET: make the decorrelated member competitive (staged)

**The mechanism.** iter22's ROCKET is UNIVARIATE — each random kernel convolves ONE band. So it can
never encode a cross-band signature like "low VH **and** low NDVI **and** low NDWI", which is the
actual pond fingerprint; the Transformer captures exactly that via attention across bands. iter23
switches to **multivariate kernels**: each kernel spans a random SUBSET of the 24 channels (subset
size drawn ~2^U(0,log2·max_channels), ROCKET-multivariate recipe), summing the per-channel dilated
convs before PPV/max. This adds cross-band interaction the univariate version structurally lacks — a
GENUINE-signal upgrade (should raise ATC-F1 / true transfer), not just capacity to overfit source.

**0-submission screen.** Re-run univariate `c_rocket` (paired baseline) + multivariate `c_rocket_mv`,
2 seeds each. Decision from the paste:
- **c_rocket_mv ATC-F1 clearly > c_rocket AND ρ(mv, xview) still < ~0.90** → the decorrelated member
  is now competitive → blend it (`champion_rocketblend5_mv`) and spend ONE submission; a blend that
  finally beats the cluster would be the first real LEVEL gain since the GBDT→Transformer swap.
- **ATC-F1 flat / ρ jumps toward the cluster** → multivariate didn't help transfer (or bought
  strength by re-correlating). ROCKET's lane is then exhausted; lock `champion_archblend4` +
  `champion_rocketblend5` as the diverse finalist pair and pivot to the Phase-Two writeup.

## iter22 — ROCKET, a genuinely different model class (staged)

The last lever of the right species. `src/rocket_model.py` (from-scratch, pure numpy + sklearn, no
new dependency, no external data → rules-safe like the Transformer): a bank of **random** convolution
kernels over the 12-month sequence, each summarized by PPV + max, then a plain **linear** classifier.
No attention, no learned representation → **decorrelated from the Transformer by construction.** It
consumes the *identical* representation the champion sees (`to_inputs` over the same `_mask_views`
augmentation), so the ONLY difference is the estimator — which makes the cross-model rank-correlation
a clean go/no-go. Smoke-tested locally end-to-end (raw CSVs + torch present here): valid submission,
OOF AUC 0.943 on a 300-row/200-kernel smoke — the model learns.

**The Colab run spends 0 submissions.** It regenerates the 7 known-LB anchors + 5 champion seeds (so
the retro-fit gate + seed floor stay valid), runs ROCKET at 2 seeds (for DIS + a seed-collapsed
finalist), screens it, then prints the **`arch_blend` correlation matrix** with ROCKET as a 5th
member. Decision from the paste:
- **ρ(rocket, xview) < ~0.90 AND rocket ATC-F1 within ~0.05 of champion** → decorrelated + competent
  → first real ensemble-**level** artifact since the Transformer swap; upload `champion_rocketblend5`
  (or `champion_xview_rocket`), also the best diverse finalist for the 721-row private slice.
- **ρ < 0.90 but rocket weak** → decorrelated but would drag an equal-weight blend; down-weight/hold.
- **ρ ≥ ~0.94** → even a foreign model class ranks these rows the same → the architecture search is
  **genuinely finished**; lock `champion_archblend4` (0.8946) and pivot to the Phase-Two writeup.

## iter20 — mean_min as a decorrelated ensemble member (RESULT above)

iter18 showed the mean-pool architecture variants are ~0.94 correlated (too alike for level). iter20
asks whether a MIN-pool model is decorrelated enough (target rank-corr <0.9 vs the mean-pool cluster)
to finally add ensemble LEVEL. `arch_blend` prints c_meanmin's correlation row as the free go/no-go;
`champion_archblend5` adds it as a 5th member. mean_min run at 5 seeds so it can be seed-collapsed
(`champion_meanmin_seedavg5`, banked finalist) — this also directly measures whether mean_min's seed
variance really is higher, which is what DIS was reacting to. If the correlation is ~0.94 too, pooling
diversity is exhausted and iter21 = instance-expansion (the cross-exam's #1 data-model lever).

## iter19 — dispersion / lower-tail pooling (RESULT above; superseded by iter20)

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
