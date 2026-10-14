# AGENT_BRIEF — standing directive for the LB-gated autoresearch loop

*This is the `program.md` of this project (cf. `karpathy/autoresearch`). Any fresh
agent session reads THIS file first, picks the next experiment, edits
`config/config.yaml` + `experiments/run_current.sh`, commits + pushes. A human runs
it on Colab/Kaggle and pastes the leaderboard score into `experiments/LB_LOG.md`.
Update this file whenever a result changes the queue.*

> **`../PROJECT_STATE.md` is the MASTER, portable, single-source-of-truth doc** (the one the
> user carries to any new cloud account). **Every session, update it** — §3 status, §4 ledger,
> §5 narrative, §6 lessons, and the "Last updated" line — alongside `LB_LOG.md` and this file.

**Competition:** GeoAI Aquaculture Pond Identification (Zindi / FAO / ITU).
**Best public LB:** 0.8955 (temporal Transformer + relative-time + cross-view invariance λ=1.0).
**Deadline:** 2026-08-16. **Target:** ~+0.033 to top-5 (≈ 0.928+).

---

## THE ONE HARD RULE (this is why we are not vanilla autoresearch)

`autoresearch` keeps/discards on a trustworthy local metric (val bits-per-byte).
**Here local OOF is BLIND** — two models with identical OOF differ by **+0.05 on the
LB**. So:

> Iterate autonomously on anything with a trustworthy local signal (reproducibility,
> code correctness, seed variance, OOF-decorrelation for blends, training stability,
> emitting candidate submissions). **Every keep/discard on a transfer-affecting change
> is gated on the Zindi LB score** in `LB_LOG.md`. NEVER auto-select a model because
> its OOF/combined is higher. The leaderboard is the only ground truth.

## Hard competition constraints (never violate)

- Only the supplied competition data. **No external data.**
- **PRETRAINED MODELS ARE ALLOWED** if openly available to everyone. *(CORRECTED 2026-07-22 from
  the live rules page — every doc before this date wrongly said "train from scratch, no pretrained
  models." TabPFN, Presto, Prithvi, Clay, SatMAE etc. are LEGAL. See `RESEARCH_07.md` §1.)*
- **AutoML banned.** Open-source, seeded, reproducible only.
- `TargetF1` scored at a **hard 0.5 cut** — no threshold tuning. Base-rate / prior /
  prevalence correction (a monotone shift so the F1-optimum lands at 0.5) *is* allowed.
- **100 submissions total**, max 5/day. Compute is NOT the bottleneck. Final score = **65% LB +
  35% code review** of the top 5 (reproducibility, clarity, novelty).

## Confirmed data facts (verified on the live site 2026-07-22 — see `RESEARCH_07.md` §2)

- **TRAIN: 1,821 rows × 12 FULL months**, ~40% positive. **TEST: 1,030 rows × only 4/5/6
  CONSECUTIVE months**, rest `-9999`; test positive rate believed higher (~0.65).
- Bands: S1 VH/VV always present when the month is observed; 10 S2 optical bands may be missing
  **per-band due to cloud**. **lat/lon REMOVED.**
- **The shift is TEMPORAL BY DESIGN** — train and test are different time periods; conditions
  "change across seasons and years."
- Public LB = **30%** of test (~309 rows), private = **70%** (~721 rows).
- **25 Jun 2026 data reset** after a leak (new train = old train + old test *with labels*; new test
  issued). Our first submission was 9 Jul, so **all 7 anchors post-date it and are comparable.**
- Forum scores of 0.95/0.98 are **pre-reset (leaked)** — ignore. Real bar is the "90s club" ≈0.90–0.95.

## Measurement protocol (quantified 2026-07-22)

Combined-metric SE ≈ **0.012** on public (309 rows), ≈0.008 on private. But a **paired** delta
between two ρ≈0.9 variants of our own model has SE ≈ **0.006**. So: unpaired/cross-team needs
≥0.012; our own A/B is *confident* at ≥0.012, *suggestive* at ≥0.006, unmeasurable below 0.006.
Expected |public − private| drift for one model ≈ **0.012**.

## State of play

- **Levers used:** (1) prior/base-rate correction (+0.11 total, saturated at realized pos-rate
  ~0.65); (2) GBDT → temporal Transformer swap (+0.05 at identical OOF); (3) relative-time
  reframing (+0.013, capacity-neutral structural reframe). Champion = **0.8908**. The net is a
  strong but **overconfident** ranker (saturated probs).
- **Verified inert-by-default & ready to probe:** Step 1 `prevalence_target`
  (`src/calibration.py`), Step 3 `seq.channels.*` (`src/seq_model.py`).

## DEAD ENDS — do not re-propose (already tried, failed, or rule-illegal)

BBSE/EM prior estimation · WIF / fixed-threshold water-index features · TabPFN
(pretraining) · temperature scaling · importance-weighting / DANN (ESS collapse at
adversarial AUC 0.99) · OOF meta-stacking (Ridge on OOF) · group-KFold / "it's leakage"
(the gap is designed covariate shift, proven leak-free).

## META-LESSON (2026-07-20, REFINED 2026-07-21 after iter5 WON)

Round-03: three consecutive blind toggles all LOST while OOF stayed flat/high —
blend 0.8705, per_cell_detrend 0.8266, K=4 0.8665. Round-04: after a research pause,
**iter5 relative-time reframing WON, 0.8908 (+0.0128), breaking a 10-day plateau.**

The refined law (this is the compass now):
1. **Added *capacity* hurts; capacity-neutral *structure* helps.** Extra model / channels /
   augmentation all lost. But reframing the *coordinate system* (calendar→relative time), same
   params, WON. Propose changes to the model's inductive bias / coordinate frame that DELETE a
   covariate-shift memorization channel — never changes that add parameters, models, or channels.
   The additive-channel family (`deltas`/`indices`/`rank`) stays **dead** (adds capacity).
2. **OOF is anti-correlated**, not merely blind: the 0.8908 winner's OOF (0.9811) was *lower*
   than the old champion's (0.9827); highest-OOF run (K=4, 0.984) = 2nd-worst LB.
3. **Measurement resolution is binding.** Public LB ≈309 rows → ~±0.01 noise. Only probe changes
   plausibly large enough to clear it (relative-time was +0.013); don't A/B inside the noise band.

**Champion (NEW): seq K=2 + relative_time @ realized 0.649 = 0.8908.** Standing operating-point
tool: `prevalence_target 0.649` (holds any probe at the exact champion pos-rate for clean isolation).

## CURRENT STATE: LOOP PAUSED → research round 06. Champion 0.8955. Both lanes CLOSED.

Champion = relative-time + **cross-view invariance λ=1.0**, **0.8955**. **iter10 (λ=3.0) LOST: 0.8921
(−0.0034) → reverted.** Reading: λ=1.0 is an **interior optimum**, not a floor — λ=3 de-saturated
FURTHER (t\* 0.4450→0.3400, prevalence delta 1.30→0.725, raw pos-rate 0.553→0.583) while `oof_auc`
HELD at 0.9896, so the drop is **not** ranker collapse; de-saturation just stops paying past λ=1.

**Both structural lanes are now measured closed:** positional (dnorm −0.006, NoPE +0.001) and
objective (λ=3 −0.003). Amplitude stays toxic (detrend −0.051). We are OUT of queued ideas plausibly
large enough to clear ±0.01. **Budget is no longer the constraint** (~130 submissions over ~26 days) —
**idea quality and measurement resolution are.** Hence the pause.

## EXPERIMENT QUEUE

- …~~Iter5 relative-time~~ ✅0.8908 · ~~Iter6 TTA~~ ❌0.8885 · ~~Iter7 dnorm~~ ❌0.8844 · ~~Iter8 NoPE~~ ➖0.8917 (FINALIST) · ~~Iter9 xview λ=1.0~~ ✅**0.8955 CHAMPION** · ~~Iter10 λ=3.0~~ ❌0.8921 (reverted).
~~Research round 06~~ DONE — both reports triaged in `RESPONSE_06.md`. Queue that came out of it:
1. ~~**Iter11 — OFFLINE LB-PREDICTING VALIDATOR**~~ ✅ **PASSED 2026-07-22, 0 submissions.**
   **ATC-F1 15/15 concordant, ρ=+0.964** · **DIS 5/5, ρ=+1.000 (n=4)** → both CLEARED.
   The *original* pre-committed estimator **ATC FAILED (6/15, ρ=−0.429)** and the naive control MARG
   failed (8/15, −0.321) — so the pre-repair version of this experiment would have failed outright,
   and the two that cleared are precisely what research round 07 added.
   **Those two failures confirm the rank-only proof:** ATC and MARG both measure confidence, which
   the LB provably cannot see, and both came out negative (`seq_a_l3`/`seq_a_xview` have the LOWEST
   MARG yet the HIGHEST LB). De-saturation was never the mechanism.
   **→ SCENARIO A: screen offline; submit only where ≥2 cleared estimators beat the champion.**
   Caveat: DIS rests on n=4 (exact null p≈0.042) — second vote only. ATC-F1 (p≈0.005) is the solid one.
2. **Iter12 — dispersion pooling** `mean ⊕ std` over observed months (Fable's R3; Ottinger's
   permanence/low-std physics). Fallback if within noise: split-pool (mean d/2 ⊕ std d/2) = exactly
   capacity-neutral. **NOT** mean⊕max — the drain event is an outlier the literature suppresses.
3. **Iter13 — focal loss** (γ=3 or FLSD-53, **not** γ=2), keeping λ=1; refit the prevalence δ.
   Non-redundant with cross-view (entropy reg vs variance penalty). Moderate prior: iter10 showed
   de-saturation is near-exhausted, and focal targets the same weakness by another route.
4. **Gated on iter11 passing:** fold-ensemble deletion (train-on-all; kills OOF so it is unshippable
   without a validator) → group-DRO over window-length groups → VH−VV replacement channel → pairwise
   AUC surrogate (demoted: in-domain AUC already ≈0.99).
5. **Endgame — REPRIORITIZED 2026-07-22 after the EY comparison (`RESEARCH_08_EY.md`):**
   - **Reproducibility is now a live risk, not a deadline chore.** In the EY Biodiversity Challenge
     (same platform, same rulebook template) **only 2 of the top 10 survived post-challenge code
     review**; the rest were eliminated for missing the code deadline or reproducibility concerns,
     and one prize went unpaid. Our standing is **65% LB + 35% code review of the top 5** — a larger
     weight against a smaller field. ✅ Partly addressed: `experiments/reproduce_champion.sh` (with a
     config guard that refuses to run if the committed config drifts off-champion) + README corrected
     (it had documented the **GBDT** as "the model" while `--full` defaults to `--model gbdt`, so a
     reviewer would have reproduced the 0.826-era baseline). **Still open:** pin `torch==x.y.z`,
     ship an environment lock, track `experiments/results.tsv` (currently gitignored — it is our
     strongest innovation evidence), regenerate a clean champion log.
   - **Designate both finalists deliberately** — Zindi defaults to "best 2 public", a shake-up
     generator on a 30% slice. Use the certified offline validators to pick the *non-obvious* second.
   - **Audit for implicitly-banned derived features.** EY's enforcement covered not just lat/lon but
     *distances, nearest-neighbour, clustering, grid binning, any positional encoding*. Verify
     nothing in our pipeline reconstructs position or row identity.
   - **Verify the metric empirically.** EY's own metric description contradicted itself (F1 vs
     accuracy) and was never answered. Check our 0.6·F1 + 0.4·AUC weighting against our known
     submission/score pairs rather than trusting the page text.
   - One-time prevalence sweep — **low priority**: `t_star = 0.445 ≈ F1*/2` already matches the
     Lipton et al. optimum, so expect a plateau, not a gain. Note `prior_sweep` currently uses
     `apply_prior` (a log-odds offset), **not** `target_prevalence_shift`, so it would sweep the
     wrong lever as written.
6. **Screen candidate — pseudo-labeling / self-training on the 1,030 test rows.** An EY organizer
   explicitly permitted predictions "generated from compliant models and the provided competition
   data" under a near-identical rulebook. **Honest note:** we rejected CAST self-training in round
   05, but on *technical* grounds (OOD/ESS-collapse at adversarial AUC 0.99), never legal ones. What
   changed is not the legality argument but that we can now **screen it for zero submissions**.
- **Private-LB finalists — mechanism CONFIRMED (2026-07-21, Zindi rules page):** you **choose 2
  submissions** before close; if you don't, Zindi defaults to your 2 best *public* scores. So the
  hedge is USABLE: **manually designate xview λ=1.0 (0.8955) + NoPE (0.8917)** — do NOT rely on the
  default (a lucky noise-rider from the prevalence sweep could displace NoPE). NOT the TTA variant
  (too correlated).
- **Budget correction (same source):** cap is **100 total submissions**, not 5/day×days. ≈20 used →
  **≈80 remain.** Also: final ranking = **65% LB + 35% top-5 rubric** (reproducibility/innovation) —
  our seeded, LB-logged loop is strong there; queue a reproduction README for the endgame.

## REJECTED in round 05 (do not re-propose — see RESPONSE_05.md)

Saerens-EM/MLLS prior (3rd rejection; label-shift assumption, BBSE gave 0.44 vs true 0.649) · Zou
water-tree / WIF / EVI indices (dead-end + toxic amplitude axis) · CAST self-training (OOD/ESS-collapse
family under adversarial AUC 0.99) · CropNet-blend + big-bang bundle (unattributable; blend already −0.0075)
· learned relative-position bias / ALiBi (adds capacity). CropNet as a standalone diverse finalist = low prior, never blended.

## Per-iteration protocol

1. Read `LB_LOG.md` + `experiments/results.tsv` + this file → pick next queue item.
2. Edit `config/config.yaml` + `experiments/run_current.sh`; commit + push.
3. Human: Colab/Kaggle **Run all** → upload best CSV to Zindi → paste LB into `LB_LOG.md`.
4. Update this queue (keep/discard on the pasted LB) and advance.
