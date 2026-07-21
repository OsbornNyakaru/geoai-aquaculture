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
- **Last updated:** 2026-07-21 · **Champion public LB: 0.8908** · **Loop state: ACTIVE — iter8 NoPE tied (0.8917, locked as diverse finalist); positional lane exhausted; iter9 cross-view invariance staged**

---

## 1. Resume in 60 seconds (any account)

1. `git pull` the repo (see §2 for the per-platform loop).
2. Read this file top-to-bottom — you're now caught up.
3. **Current next action:** **Run all** → upload `submission_seq_xview.csv` → paste the LB score. This
   is iteration 9: cross-view invariance objective (`seq.consistency_lambda: 1.0`) — penalize the logit
   variance across a row's K=2 masked views (`gemini_loop/RESPONSE_05.md` #4). Objective-level,
   capacity-neutral, held at 0.649, isolated vs 0.8908. iter8 NoPE tied (0.8917) → locked as finalist.
4. Champion is safe & isolated: `seq.consistency_lambda: 0` reproduces the champion bit-for-bit; iter9
   sets it 1.0. If iter9 ≤ champion → ENDGAME: one-time prevalence sweep + finalize champion + NoPE picks.

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
  masking-augmented training views, **relative-time reframing ON** (observed window left-aligned
  to t_rel=0), operating point held at **realized pos-rate 0.649**.
- **Champion config** (`config/config.yaml`): `seq.K: 2`, `seq.relative_time: true`,
  `seq.pos_encoding: learned` (iter7 probe sets it `dnorm`), all `seq.channels.*: false`,
  `seq.tta.enable: false`, `calibration.prevalence_target: 0.649`.
- **Best public LB: 0.8908** (was 0.8780 for 10 days; relative-time added +0.0128). Field: top
  ≈0.9452, top-5 ≈0.928–0.945. Gap to top-5 now ≈ **+0.037**.
- **Loop state: ACTIVE.** First win since the plateau. Now banking capacity-neutral robustness
  moves (TTA → multi-seed bagging) on the new champion.

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
| 9 | cross-view invariance objective (`seq.consistency_lambda: 1.0`; penalize logit var across K views) | _pending_ | _pending_ | staged (RESPONSE_05 #4) |
| — | endgame: one-time prevalence sweep (plateau center); finalists = champion + NoPE | | | not yet run |

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

**Do not re-propose (tried & failed, or rule-illegal):** GBDT+seq blend · `per_cell_detrend` and
the additive-channel family (`deltas`/`indices`/`rank`, now low-prior) · K>2 augmentation · BBSE/EM
prior estimation · WIF / fixed-threshold water features · TabPFN & any pretrained/foundation model
(rule-banned) · temperature scaling · importance-weighting / DANN (ESS collapse @ adversarial AUC
0.99) · OOF meta-stacking · group-KFold / "it's leakage" (the gap is designed covariate shift, proven
leak-free).

**Constraints (never violate):** only supplied data; no external/pretrained models; AutoML banned;
`TargetF1` scored at hard 0.5 (prior/prevalence shift allowed, threshold tuning not); seeded &
reproducible; ≤5 submissions/day.

---

## 7. Key files map

| File | Role |
|---|---|
| **`PROJECT_STATE.md`** | ← this file. Master state, portable across accounts. Updated every session. |
| `experiments/LB_LOG.md` | Reward ledger — paste each submission's Zindi LB here. |
| `gemini_loop/AGENT_BRIEF.md` | Standing directive for the coding agent (rules, queue, meta-lessons). |
| `gemini_loop/UPDATE_05.md` | **Current** research brief → paste into Deep Research (UPDATE_04 = prior round). |
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
