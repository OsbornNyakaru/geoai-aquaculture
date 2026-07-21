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

- Only the supplied competition data. **No external data, no pretrained/foundation
  models** (bans TabPFN, ImageNet/SSL backbones). Train from scratch.
- **AutoML banned.** Open-source, seeded, reproducible only.
- `TargetF1` scored at a **hard 0.5 cut** — no threshold tuning. Base-rate / prior /
  prevalence correction (a monotone shift so the F1-optimum lands at 0.5) *is* allowed.
- **Max 5 submissions/day** — the scarce resource. Compute is NOT the bottleneck;
  submissions are. Spend each probe deliberately.

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
1. **Iter11 — OFFLINE LB-PREDICTING VALIDATOR** *(current, staged, **0 submissions**)*. Both reports
   ranked this #1 independently. Regenerate 7 known-LB variants via the new `--set` overrides, estimate
   each from the 1030 **unlabeled** test rows (ATC · two-seed disagreement · a naive margin control),
   and check the ranking against `experiments/anchors.tsv`. **Gate:** detrend+K4 must fall below
   reltime+xview with ρ>0.7. PASS → screen the backlog offline, ~80 subs become a real search budget.
   FAIL → costs nothing; revert to funding only ≥+0.013 ideas.
2. **Iter12 — dispersion pooling** `mean ⊕ std` over observed months (Fable's R3; Ottinger's
   permanence/low-std physics). Fallback if within noise: split-pool (mean d/2 ⊕ std d/2) = exactly
   capacity-neutral. **NOT** mean⊕max — the drain event is an outlier the literature suppresses.
3. **Iter13 — focal loss** (γ=3 or FLSD-53, **not** γ=2), keeping λ=1; refit the prevalence δ.
   Non-redundant with cross-view (entropy reg vs variance penalty). Moderate prior: iter10 showed
   de-saturation is near-exhausted, and focal targets the same weakness by another route.
4. **Gated on iter11 passing:** fold-ensemble deletion (train-on-all; kills OOF so it is unshippable
   without a validator) → group-DRO over window-length groups → VH−VV replacement channel → pairwise
   AUC surrogate (demoted: in-domain AUC already ≈0.99).
5. **Endgame:** one-time prevalence sweep (4 subs; needs a `calibration.prevalence_sweep` list mirroring
   `prior_sweep` but calling `target_prevalence_shift()`; the 0.649 entry must come out byte-identical
   to the main submission — pick the plateau CENTRE) · designate the 2 finalists · reproduction README.
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
