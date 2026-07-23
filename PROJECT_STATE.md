# PROJECT_STATE — single source of truth (portable across cloud accounts)

> **What this file is.** The one document you carry to any cloud account. It lives in the
> git repo, so a fresh Colab/Kaggle account gets it automatically on `git pull`. It holds
> everything: how to resume anywhere, the current champion, every experiment + output + LB
> score, what improved, what declined, the lessons, and the next action.
>
> **It is updated EVERY session.** If you're reading this on a new account, it is current as
> of the "Last updated" line below. Supporting files (`experiments/LB_LOG.md`,
> `gemini_loop/AGENT_BRIEF.md`, `RUN_ON_CLOUD.md`, `RUN_ON_KAGGLE.md`) go deeper; this file
> summarizes and points to them.

- **Competition:** GeoAI Aquaculture Pond Identification (Zindi / FAO / ITU)
- **Repo:** `OsbornNyakaru/geoai-aquaculture` (private) · branch `main`
- **Deadline:** 2026-08-16 · **Submissions:** max 5/day (manual upload to Zindi; no API)
- **Last updated:** 2026-07-23 · **Champion public LB: 0.8955 single seed / reliable ≈0.8865** · **Finalists: `champion_archblend4` = 0.894643; `c_meanmin` = 0.898566 (highest single draw after champion).** · **Loop state: iter20 STAGED — mean_min as a DECORRELATED ENSEMBLE MEMBER (pooling-axis diversity). Go/no-go = c_meanmin's rank-corr row vs the mean-pool cluster: <0.90 → archblend5 buys level; ~0.94 → pivot to iter21 instance-expansion.**
- **🚨 READ FIRST if you are a fresh session — three corrections, one of them fatal to the ledger:**
  1. **SEED VARIANCE IS 0.0191, MEASURED (2026-07-22).** The champion configuration, changing *only*
     the RNG seed, scored **0.8955 (seed 42)** vs **0.8764 (seed 7)**. **Nine of our eleven recorded
     verdicts have effect sizes smaller than that** — including relative-time (+0.0128) and the
     cross-view win (+0.0047) that made this model "champion". **Do not trust any A/B in the ledger
     below ±0.019 unless it was seed-paired.** The ±0.01 floor we operated under for fifteen
     iterations came from row-count theory, never from a measurement.
  2. **Pretrained models are LEGAL** — TabPFN and Presto were rejected on a misreading of the rules.
  3. **The metric is RANK-ONLY** — the LB is blind to calibration.
  See §6, `experiments/LB_LOG.md` (the seed section), and `gemini_loop/RESEARCH_07.md`.

---

## 1. Resume in 60 seconds (any account)

1. `git pull` the repo (see §2 for the per-platform loop).
2. Read this file top-to-bottom — you're now caught up.
3. **Current next action: `Run all` → iter19 (DISPERSION POOLING screen), then paste back the
   RETRO-FIT + GATE, the SEED SPREAD, and the SCREEN lines for `c_meanmin` / `c_meanstd` /
   `c_moments`. Submit a pooling variant only if ≥2 cleared estimators beat champion AND the margin
   exceeds that estimator's seed sd. Separately: upload `submission_champion_archblend4.csv` once
   (bank the lowest-variance finalist).**

   **iter18 result — the architecture ensemble is MARGINAL.** Cross-architecture rank-corr
   ρ=0.9395 (≈ the 0.9511 seed baseline): the four tied transformer variants are the same model
   class, so they barely decorrelate and the blend lands at the member mean (variance-only, like
   seed-avg). Lesson: pooling variants can't buy *level*; only a different model class can. So the
   plan is representation first (iter19 pooling), then a decorrelated MiniRocket/CropNet member
   (iter20). Round-09 deep research (Claude + Gemini) is triaged in `gemini_loop/RESPONSE_09.md`;
   the three-way convergence on dispersion pooling is why iter19 leads.

   **iter17 result — the Presto lane is DEAD, for 0 submissions.** All four configs returned
   adversarial AUC **0.965–0.976** on the frozen embeddings (>0.9 ⇒ the encoder *encodes* the
   designed temporal shift rather than normalizing it). ATC-F1 put Presto **0.044–0.059 LB below
   champion**; both configs HOLD. Its OOF (0.967–0.969) is already below champion's 0.975. That
   closes the foundation-model / model-class frontier. **But it was worth every second:** a
   near-perfect train/test separator existing in a general-purpose, label-free representation of the
   raw pixels **independently proves the shift is real and large** — the ~0.975 OOF vs ~0.89 LB gap
   is mostly irreducible covariate shift, and our champion already carries the right response
   (masking views + relative time + cross-view invariance = shift-invariance machinery). Presto lost
   *because* it faithfully re-encodes the raw signal, shift included.

   **Why the grand ensemble now.** The seed-average bought variance reduction but no level (0.8865 ==
   the single-seed mean 0.8859), because seeds are 95.1% rank-correlated. The one remaining cheap
   shot at *level* is to pool across **different architectures**, which may be decorrelated where
   seeds are not. The go/no-go is free — the cross-architecture rank-correlation matrix
   (`tools/arch_blend.py`): ρ ≈ 0.95 ⇒ no gain (behaves like seed-avg); ρ < ~0.90 ⇒ pooling gains
   level with bounded downside. The screen cannot resolve this (ATC-F1 seed sd 0.0576 ⇒ ±0.0094 LB,
   coarser than any ensemble gain), so the matrix is the instrument.

   **iter16 result: seed-averaging scored 0.8865 against a predicted ~0.886.** The variance model
   is confirmed to 0.0006. But there was **no ensemble gain** — we bought variance reduction, not
   level, because the seeds are 95.1% rank-correlated so only ~5% of the error is independent.
   **Finalist decision (revised):** designate **`champion_seedavg5` + `seq_a_xview`**, not
   xview + NoPE. Shrunk true-quality estimates are 0.8865 vs 0.8911 — a tie inside our resolution
   floor — but the two carry *different risk profiles* (consensus vs point estimate), whereas
   xview and NoPE differ by 0.0038 and are two draws of the same thing.

   **iter15 settled the measurement question. The screen's resolution is ≈0.010–0.013 LB**, derived
   two independent ways that agree: ATC-F1's seed sd (0.0576) converts to ±0.0094 LB via the anchor
   fit `LB = 0.1628·ATCF1 + 0.7714`, and the directly measured champion seed spread is 0.0191
   (sd ≈0.013). Seed rank-correlation is **0.9511** — ~95% of our ordering is reproducible, ~5% is
   RNG, and that 5% moves the LB by 0.019.

   **🚨 THE STANDING RULE FROM HERE: stop running small A/B probes.** Only effects **> ~0.010 LB**
   are measurable *in principle* with our budget. The only two effects ever measured above that
   floor are the GBDT→Transformer swap (+0.05) and detrend (−0.05) — **both model-class changes**.
   Every architectural tweak, loss term, pooling variant, positional reframe and regularization knob
   sits below it. The two fundable directions are **(a) variance reduction** (seed averaging) and
   **(b) a model-class change** = the **Presto lane** (`RESEARCH_07.md` §5e).
   In iter15 the new seed-noise guard downgraded two **2/2-vote** candidates to HOLD because their
   margins were inside estimator seed noise — it caught exactly the iter14 mistake.

   **Superseded — kept for the record:**
   iter15 measures the estimator seed-floor across five champion seeds, screens `c_meanmin` (the
   *lower* tail — the pond literature's actual detector, which we never tested), and pools the five
   champion seeds into one rank-averaged submission via the new `tools/seed_average.py`.
   **Expect the seed-averaged public score to be BELOW 0.8955, and want it to be.** 0.8955 is the
   better of two draws from a distribution with sd ~0.013 and is probably an upward fluctuation.
   Judge the pooled submission against the **mean** of our single-seed scores (~0.886), not the max.
   Chasing the public number under this much variance is what produces a shake-up.

   **Superseded — kept for the record:**
   **(a) UPLOAD `submissions/submission_c_dropout3.csv` to Zindi** and paste the LB. The file was
   already written by the iter13 run — no rerun needed. This is our **first screen-approved
   submission** and the first LB spend since iter10.
   **(b) `Run all` → iter14** (regularization sweep, **0 submissions**) and paste back the three
   tables. Independent of (a); do not serialize them.

   **iter13: the screen FIRED for the first time.** `c_dropout3` (dropout 0.2 → **0.3**) cleared
   both certified estimators — **ATC-F1 +0.0165, DIS +0.0029, 2/2 → SUBMIT**. It is *exactly
   parameter-neutral*, the most on-thesis knob in the repo under our own "less fit transfers better"
   law — and it had **never been touched in twelve iterations**. We spent those twelve on
   architecture while the plainest regularization knob sat at its default.
   ⚠️ The DIS margin is **tiny** (+0.0029): a 2/2 by the rule, not a resounding one. Estimator
   deltas are **not** on the LB scale.
   Also from iter13: **`c_compact` genuinely ran and FAILED** (−0.0053/−0.0252) — the 24→14 deletion
   two research agents derived *independently* does not work, so independent agent convergence is
   **not** evidence. **`c_meanmax_l0` flipped to −0.0847** vs +0.0838 with λ=1, so the upper-tail
   statistic *depends on* cross-view invariance rather than competing with it. **`c_k3` ≈ 0** — K=2
   confirmed optimal from three points instead of two.

   **iter12 (first screen): all five candidates HELD, 0 submissions spent.** The rule earned its
   keep — ATC-F1 liked `mean_max` (+0.0838) but DIS disagreed (−0.0301), which under the old blind
   regime would have been a wasted submission. Three findings:
   **(a) Amplitude is the PRIMARY SIGNAL** — rank-replacement collapsed OOF 0.9753 → 0.857/0.865 on
   both seeds (ATC-F1 −0.1703). The pond discriminator is "persistently LOW backscatter", an
   absolute level. The rank/ordinal family is now closed *with evidence*. (This does **not**
   resurrect the old "amplitude is toxic" law, which remains unevidenced — *removing* amplitude
   being catastrophic is a different claim.)
   **(b) `c_compact` was never tested** — a config-path bug (`seq.compact_missing` vs
   `seq.channels.compact_missing`) meant the flag never reached the model; the run was bit-identical
   to the champion and the screen scored the no-op as a 0.0000 tie. Fixed; the pipeline now logs the
   **actual** input width. Re-tested in iter13.
   **(c) DIV failed at ρ = −0.857** (2/15) — *lower* fold-diversity goes with *higher* LB, the
   opposite of hypothesis H1. H1 is not supported.

   **iter11 PASSED (2026-07-22) — the measurement constraint is broken.** Retro-fit on the 7 anchors:
   **ATC-F1 15/15 concordant, ρ=+0.964** and **DIS 5/5, ρ=+1.000 (n=4)** both cleared, while the
   *original* pre-committed estimator **ATC FAILED at ρ=−0.429** and the naive control MARG failed at
   −0.321. The pre-repair iter11 would have failed outright; the two that cleared are exactly what
   round 07 added. And the two failures **confirm the rank-only proof** — both measure confidence,
   which the LB cannot see, so both anti-correlate.
   **→ SCENARIO A is live:** screen offline, submit only where **≥2 cleared estimators** beat the
   champion. Treat DIS as a second vote only (n=4, exact null p≈0.042); ATC-F1 is the solid one.
4. **iter11 was REPAIRED on 2026-07-22 before its first run** (`RESEARCH_07.md`). Three bugs would
   have wasted the whole spend: the DIS estimator was **unscoreable** (only the champion got a second
   seed, but ≥3 variants are needed for a rank correlation — now fixed by seeding detrend/k4/reltime);
   the gate could raise `KeyError`; and a loose glob admitted `_smoke` files as seed replicates.
   The **gate itself was replaced**: exact permutation nulls at n=7 show ρ>0.7 passes by chance at
   p=0.044 (~9% familywise over 3 estimators) **and** can reject a *perfect* validator (ρ=0.643),
   because 4 of the 7 anchors sit inside the noise band so their measured order is itself noise. It
   now scores concordance on pairs with |ΔLB| > 0.01 (exact null p=0.0048), with ρ reported descriptively.
5. **After iter11, branch on the gate** — the two-scenario decision tree is in `RESEARCH_07.md` §6.
   Zero-submission work that runs either way: the partial-S2 missingness audit, iterated adversarial
   channel attribution, posting the two-column legality question, and building the Presto/TabPFN lane.
6. **RULE FACTS (verified 2026-07-22):** (a) we **designate 2 finalist submissions** (default = 2 best
   public) → the hedge is usable; designate manually before close. (b) **100-total cap**: ≈20 used,
   ≈80 left. (c) final score = **65% LB + 35% rubric** (top-5, reproducibility/innovation) → prep a
   reproduction README at endgame; the ITU Cropland precedent (same organiser family, 40% report
   weight) shows this channel is real and low-variance. (d) **Pretrained models are LEGAL** — see §6.

---

## 2. How to run on ANY cloud account (the portable loop)

The design: **code lives in GitHub, data + secret live in the account.** Each iteration the
coding agent edits `experiments/run_current.sh` + `config/config.yaml` and pushes; you pull and
**Run all**; the notebook itself never changes. Zindi submission stays manual (5/day).

**You always need three things on a new account:**
1. **The code** — via `git pull` of the private repo (needs a GitHub token, below).
2. **The data** — `Train.csv`, `Test.csv`, `SampleSubmission.csv` (from Zindi; keep PRIVATE —
   rules allow only supplied data).
3. **A GitHub PAT** — fine-grained token, repo `geoai-aquaculture`, **Contents: Read-only**,
   short expiry, stored as a secret named **`GH_PAT`**. (Read-only + single-repo + short expiry
   because the token gets embedded in `.git/config` on the VM.)

### Google Colab
- Drive: put the 3 CSVs in `MyDrive/geoai-data/` (top-level, exact name — the notebook reads it).
- Colab Secret `GH_PAT` (🔑 sidebar, enable notebook access).
- **Runtime ▸ Change runtime type ▸ T4 GPU.**
- Open `colab_run.ipynb` → **Runtime ▸ Run all.** Cell 1 pulls; Cell 4 runs `run_current.sh`;
  Cell 5 downloads `submission_*.csv`.
- Full details: `RUN_ON_CLOUD.md`.

### Kaggle Notebooks
- Upload the 3 CSVs as a **Private Dataset** (e.g. `geoai-aqua-data`); attach via **Add Input**.
- **Add-ons ▸ Secrets** → add `GH_PAT`; **Internet: On**; **Accelerator: GPU T4 x2**.
- Paste the cells from `RUN_ON_KAGGLE.md` (git pull → deps → data → `bash run_current.sh` → FileLink).
- Kaggle gives ~30 GPU-hrs/week; a full seq run is a few minutes.

### The loop each iteration (all accounts)
`git pull` → **Run all / `bash experiments/run_current.sh`** → download the newest
`submission_*.csv` → upload on Zindi → **paste the public LB score back to the agent** (it goes
into `experiments/LB_LOG.md`, the reward signal) → agent stages the next experiment + pushes.

---

## 3. Current status

- **Champion model:** from-scratch temporal Transformer (attention over observed months via
  `src_key_padding_mask`, per-band missing-indicator channels, masked-mean-pool), **K=2**
  masking-augmented training views, **relative-time reframing ON**, **cross-view invariance objective
  (λ=1.0)**, operating point held at **realized pos-rate 0.649**.
- **Champion config** (`config/config.yaml`, LIVE — this is the reverted, exact 0.8955 state):
  `seq.K: 2`, `seq.relative_time: true`, `seq.pos_encoding: learned`, `seq.consistency_lambda: 1.0`,
  all `seq.channels.*: false`, `seq.tta.enable: false`, `calibration.prevalence_target: 0.649`.
- **Best public LB: 0.8955** (0.8780 → +0.0128 relative-time → 0.8908 → +0.0047 cross-view invariance).
- **Target RECALIBRATED 2026-07-22.** Forum scores of 0.953 / "0.98+" were posted **before the
  25 Jun data reset**, i.e. earned on the **leaked** data — ignore them. The live competitive band is
  the "90s club" (thread dated 14 Jul, post-reset): roughly **0.90–0.95**. We sit just below the bar,
  **not 0.033 behind it.** The gap is smaller and more winnable than earlier revisions of this file claimed.
- **Loop state: iter11 STAGED (offline validator, 0 submissions), awaiting a Colab run.** Both
  structural lanes measured closed — positional (dnorm −0.006, NoPE +0.001) and objective (λ=3 −0.003,
  so λ=1 is an interior optimum). Diverse finalist (NoPE 0.8917) provisionally locked — **but** if any
  lane produces a model within ~0.01 of champion with different errors, it replaces NoPE as finalist #2
  (NoPE is a near-clone and buys little private-LB variance hedge).
- **Research round 07 done** → `gemini_loop/RESEARCH_07.md`, which carries the rule correction, the
  rank-only proof, the two-scenario decision tree, and the fixes applied to iter11.

### Confirmed data facts (verified on the live Zindi site 2026-07-22)

- **TRAIN: 1,821 rows × 12 FULL months**, ~40% positive. **TEST: 1,030 rows × only 4/5/6 CONSECUTIVE
  months**, rest `-9999`; test positive rate believed ~0.65 (which is exactly our tuned prevalence).
- Bands: S1 **VH/VV** always present when a month is observed; **10 S2 optical bands may be missing
  per-band due to cloud**. **lat/lon REMOVED** — these are isolated patches with no spatial context.
- **The shift is TEMPORAL BY DESIGN:** train and test are different time periods; conditions "change
  across seasons and years." Public LB = **30%** of test (~309 rows), private = **70%** (~721 rows).
- **25 Jun 2026 data reset** after a leak (new train = old train + old test *with labels*; new test
  issued; lat/lon stripped). **Our first submission was 9 Jul**, so all 7 LB anchors post-date the
  reset and are mutually comparable — the iter11 retro-fit is valid.
- **Open question, unanswered on the forum:** does "month 01" in train mean the same calendar month
  as "month 01" in test? If not, several seasonal ideas are unworkable and relative-time's win is
  even better explained.

---

## 4. Full experiment ledger (every run, output, LB, verdict)

Metric = **0.6·F1 + 0.4·ROC-AUC**. "OOF" = local cross-val combined (⚠️ **proven BLIND / often
anti-correlated** — never used for selection). "LB" = Zindi public (~309 rows) = ground truth.

### Phase 1 — GBDT ensemble + prior correction (pre-transformer)
| Change | Operating point | LB | Verdict |
|---|---|---|---|
| GBDT ensemble, inherited train prior | pos 0.40 | 0.7140 | baseline |
| + base-rate/prior correction | pos 0.50 | 0.7561 | ✅ +0.042 |
| + prior correction (swept) | pos ~0.65 | **0.8260** | ✅ GBDT peak |
| prior 0.70 / 0.75 / 0.80 | — | 0.8216 / 0.8166 / 0.8037 | prior lever saturated |
| WIF + EVI features | pos 0.50 | 0.7509 | ❌ reverted (train AUC 0.83, no transfer) |

### Phase 2 — from-scratch temporal Transformer (the breakthrough)
| Change | Realized pos-rate | LB | Verdict |
|---|---|---|---|
| Temporal Transformer | 0.593 | 0.8776 | ✅ |
| Temporal Transformer | 0.627 | 0.8732 | |
| **Temporal Transformer** | **0.649** | **0.8780** | 👑 **CHAMPION** |
| Temporal Transformer | 0.672 | 0.8733 | |

→ +0.05 over the GBDT peak **despite identical OOF** — the finding that defines this competition.

### Phase 3 — improvement attempts, 2026-07-20 (all LOST; champion held)
| # | Experiment (only variable vs champion) | OOF | LB | Verdict |
|---|---|---|---|---|
| 2 | + GBDT rank-average blend (0.7 seq / 0.3 GBDT, ρ=0.85) | 0.952 | 0.8705 | ❌ −0.0075 |
| 3 | + `per_cell_detrend` input channels | 0.979 | **0.8266** | ❌ −0.0514 |
| 4 | seq masking views K=2 → K=4 | **0.984** | 0.8665 | ❌ −0.0115 |

Also verified this session: **Step-1 `prevalence_target 0.649` mechanism works** (holds any run
at the exact champion pos-rate → clean isolation); the Colab env **reproduces faithfully** (blend
landed exactly between its components).

### Phase 4 — capacity-CONSTRAINT direction (round-04 research, in progress)
Round-04 Deep Research triaged in `gemini_loop/RESPONSE_04.md`. Rejected proven dead-ends
(Saerens-EM prior; Zou-threshold/EVI index projection). Shifting from capacity *expansion* to
*constraint*: test capacity-neutral, structural changes one at a time.
| # | Experiment (only variable vs champion) | OOF | LB | Verdict |
|---|---|---|---|---|
| 5 | relative-time reframing (`seq.relative_time`: left-align window to t_rel=0) | 0.9811 | **0.8908** | ✅ **NEW CHAMPION** (+0.0128; first win, capacity-neutral structural reframe) |
| 6 | MC temporal-dropout TTA on champion (`seq.tta`: mask 1-2 active months, 8 views, soft-vote) | — | 0.8885 | ❌ −0.0023 (within noise, did not beat champion; reverted) |
| 7 | duration-normalized fractional positions (`seq.pos_encoding: dnorm`; share [0,1] frame across L) | 0.9789 | 0.8844 | ❌ −0.0064 (length already matched → no shift to remove; reverted) |
| 8 | NoPE / permutation-invariant SET encoder (`seq.pos_encoding: none`; drop positional embedding) | 0.9789 | 0.8917 | ➖ TIE +0.0009 (position is neutral; LOCKED as diverse finalist) |
| 9 | cross-view invariance objective (`seq.consistency_lambda: 1.0`; penalize logit var across K views) | 0.9753 | **0.8955** | ✅ **NEW BEST** +0.0047 (reduced overconfidence; edge of noise) |
| 10 | cross-view invariance strength probe (`consistency_lambda: 3.0`) | 0.9727 | 0.8921 | ❌ −0.0034 (λ=1.0 is an interior optimum; reverted; objective lane CLOSED) |
| — | research round 06 → `RESPONSE_06.md` (both reports triaged) | | | ✅ done |
| 11 | **offline LB-predicting validator** (ATC · seed-disagreement · control) retro-fit to 7 known-LB anchors | | **0 subs** | **staged** |
| 12 | queued: dispersion pooling `mean ⊕ std` (Ottinger permanence/low-std physics) | | | not yet run |
| 13 | queued: focal loss γ=3 / FLSD-53, keep λ=1, refit δ | | | not yet run |
| — | gated on iter11 PASS: fold-ensemble deletion → group-DRO → VH−VV → AUC surrogate | | | gated |
| — | endgame: prevalence sweep · designate finalists (xview + NoPE) · reproduction README | | | not yet run |

**The design compass (refined through iter7):** it is not "never change the model" — it is *added
capacity* (extra model/channels/augmentation) and *robustness moves* (TTA) that don't transfer. A
capacity-neutral structural reframe helps **only when it deletes a channel that is actually SHIFTED
train-vs-test.** Relative-time removed window START (calendar month = shifted) → +0.0128 WON.
Duration-norm removed window LENGTH (matched by augmentation = NOT shifted) → −0.0064 LOST. Before
proposing any reframe, ask first: *is this channel actually shifted?* NoPE (iter8) removes positional
identity entirely — a bigger, two-tailed bet, and the diverse finalist regardless of its public score.

---

## 5. Progress & declines — the narrative

**What moved us UP (0.714 → 0.891, +0.177 total):**
1. **Prior/base-rate correction** (+0.11 to the GBDT peak 0.826): the test set is far more
   positive (~65%) than train (~40%). Now saturated.
2. **GBDT → from-scratch Transformer** (+0.05 to 0.878): attention over *only observed months*
   transfers across the designed domain shift where flattened GBDT aggregates over-fit the source.
3. **Relative-time reframing** (+0.013 to 0.891, 2026-07-21): left-align each observed window to
   t_rel=0 so positional embeddings encode relative step, not calendar month — kills the calendar-
   specific spectral memorization the covariate shift punishes. Capacity-neutral; broke a 10-day plateau.
4. **Cross-view invariance objective** (+0.005 to 0.8955, 2026-07-21): penalize logit variance across
   a row's K=2 masked views (L=BCE+λ·Var). Reduced the model's overconfidence (its diagnosed weakness)
   and improved transfer. Objective-level, capacity-neutral. **iter10 then showed λ=1.0 is an INTERIOR
   OPTIMUM** — λ=3.0 de-saturated further (t\* 0.4450→0.3400, delta 1.30→0.725) with `oof_auc` intact
   at 0.9896, yet scored 0.8921. So the mechanism is real but bounded: *some* de-saturation transfers,
   more does not, and the failure is not ranker collapse. Lane closed at λ=1.0.

**What DECLINED (Phase 3 — everything we tried after 0.878):**
- Blend −0.0075, detrend −0.0514, K=4 −0.0115. Pattern: **every attempt that ADDED something
  (a model, input channels, more augmentation) lost.** The detrend result specifically
  **disproves** the "remove per-series level → better transfer" thesis for this model.

**Why we paused (then resumed):** public LB ≈309 rows → **~±0.01 noise**. Single-submission A/B
**cannot resolve** small (+0.005) gains; only large effects or breakages are detectable. So we
stopped guessing toggles inside the noise band and ran a research round. The output — relative-time
reframing — was a *large* effect (+0.013, above noise), which is exactly the class of change worth a
submission. Lesson: don't probe inside the noise; hunt changes big enough to clear it.

---

## 6. Lessons & DEAD ENDS (do not retry)

**Hard lessons (2026-07-20, refined 2026-07-21):**
1. **Added *capacity* hurts; capacity-neutral *structure* helps — but ONLY if it deletes a SHIFTED
   channel.** Extra model / channels / augmentation all lost (−0.008 to −0.051); robustness moves
   (TTA) land within noise. Relative-time reframing (remove window START = calendar month, which is
   shifted train-vs-test) WON +0.013. Duration-norm (remove window LENGTH, which augmentation already
   distribution-matches → NOT shifted) LOST −0.006. Compass: reframe the coordinate/inductive-bias to
   delete a channel that is *actually shifted*, never its capacity, never a matched/informative channel.
2. **OOF is anti-correlated**, not merely blind — highest-OOF run (K=4, 0.984) = 2nd-worst LB;
   the 0.8908 winner's OOF (0.9811) was *lower* than the old champion's (0.9827).
3. **Measurement resolution is the binding constraint** — 309-row public LB, ±0.01 noise. Only
   probe changes plausibly large enough to clear it; don't A/B inside the noise band.

**Do not re-propose (tried & failed, or rule-illegal):** GBDT+seq blend *(but see the caveat below —
the blend was badly constructed, not proof that blending fails)* · `per_cell_detrend` and the
additive-channel family (`deltas`/`indices`/`rank`, now low-prior) · K>2 augmentation · BBSE/EM
prior estimation · WIF / fixed-threshold water features · temperature scaling ·
importance-weighting / DANN for TRAINING (ESS collapse @ adversarial AUC 0.99) · OOF meta-stacking ·
group-KFold / "it's leakage" (the gap is designed covariate shift, proven leak-free) · VH−VV as a
replacement channel *(removed 2026-07-22: `(VH,VV)→(VH,VH−VV)` is an invertible linear map feeding a
linear layer, so the model can already represent it)*.

> **⚠️ TabPFN and pretrained/foundation models are NO LONGER on this list.** They were listed as
> "rule-banned" — that was **wrong**. See the corrected constraints below.

**Constraints (never violate) — CORRECTED 2026-07-22 from the live rules page:**
- Only the supplied datasets. **No external DATA.**
- **PRETRAINED MODELS ARE ALLOWED** — verbatim: *"You may use pretrained models as long as they are
  openly available to everyone."* Every doc before 2026-07-22 wrongly said "train from scratch, no
  pretrained models," and TabPFN was rejected on that false basis. TabPFN v2, Presto, Prithvi, Clay,
  SatMAE are all **legal**. (Caveat to verify: Zindi also says "custom packages in your submission
  notebook will not be accepted" — confirm what that means operationally before relying on a pip dep.)
- AutoML banned. Open-source, seeded, reproducible only.
- `TargetF1` scored at hard 0.5 (prior/prevalence shift allowed, threshold tuning not).
- **100 submissions total**, ≤5/day. Final score = **65% LB + 35% code review** of the top 5.

**⚠️ THE METRIC IS RANK-ONLY.** After the prevalence pin the predicted-positive count is fixed at
P̂ = 0.649·n, so F1 = 2·TP/(P̂+P) is monotone in precision@k, and AUC is rank-only by definition.
**The LB sees only how the model ORDERS the 1030 test rows — it is blind to calibration.** This
reframes the "de-saturation" story told about iter9/iter10 (cross-view invariance must have won by
changing the *ranking*, not by reducing overconfidence) and demotes any change whose mechanism is
purely calibrative (focal loss, temperature). It also means our prevalence instrument is saturated:
Lipton et al. show the F1-optimal cut is F1\*/2, and our `t_star = 0.445` ≈ 0.89/2 already.

**Measurement protocol (quantified 2026-07-22).** Combined-metric SE ≈ **0.012** on the 309-row
public LB (≈0.008 on the 721-row private). But a **paired** delta between two ρ≈0.9 variants of our
own model has SE ≈ **0.006**. So: unpaired/cross-team needs ≥0.012; our own A/B is *confident* at
≥0.012 and *suggestive* at ≥0.006; below 0.006 is unmeasurable. Expected |public − private| drift
for a single model ≈ 0.012.

---

## 7. Key files map

| File | Role |
|---|---|
| **`PROJECT_STATE.md`** | ← this file. Master state, portable across accounts. Updated every session. |
| `experiments/LB_LOG.md` | Reward ledger — paste each submission's Zindi LB here. |
| `gemini_loop/AGENT_BRIEF.md` | Standing directive for the coding agent (rules, queue, meta-lessons). |
| `gemini_loop/UPDATE_06.md` | **Current** research brief → paste into Claude Fable Deep Research (05/04 = prior rounds). |
| `JOURNEY.md` / `JOURNEY.docx` | Plain-English story of the whole project (regenerate the docx via `tools/make_journey_docx.py`). |
| `experiments/run_current.sh` | The one experiment the notebook runs each iteration (agent edits + pushes). |
| `config/config.yaml` | Single source of truth for all pipeline settings. |
| `colab_run.ipynb` / `RUN_ON_KAGGLE.md` | The pull-run loop for Colab / Kaggle. |
| `run_pipeline.py` | End-to-end: CSVs → CV → calibration → `submission.csv`. |
| `src/seq_model.py` | The champion Transformer. `src/calibration.py` | Fixed-0.5 + prevalence lever. |

---

## 8. Update protocol (keep this file current)

At the end of every session (or whenever an LB score comes in), the coding agent updates:
- **§3 Current status** (champion, best LB, loop state, next action) and the header "Last updated".
- **§4 ledger** (new rows) and **§5 narrative** (if progress/decline changed).
- **§6 lessons** (if a result adds a dead-end or overturns one).
This file is the thing you carry between accounts — it must always reflect reality. If it and a
supporting file ever disagree, trust the most recent LB score in `experiments/LB_LOG.md`.
