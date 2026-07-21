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
**Best public LB:** 0.8908 (temporal Transformer + relative-time reframing). **Deadline:** 2026-08-16.
**Target:** ~+0.037 to top-5 (≈ 0.928+). Prev champion 0.8780 held 10 days; relative-time added +0.0128.

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

## CURRENT STATE: positional lane EXHAUSTED; iter9 = cross-view invariance objective

Champion = relative-time net (`pos_encoding: learned`, 0.8908). **iter8 NoPE scored 0.8917 (+0.0009 =
TIE).** Removing ALL positional info costs nothing → order is nuisance-or-neutral here (confirms the
SAR set-statistic physics). `submission_seq_nope.csv` is **LOCKED as the diverse private-LB finalist**
(max-different model, ties public → fails on different rows). The arc — absolute-calendar 0.8780 →
relative-time 0.8908 (+0.013) → NoPE 0.8917 (tie) — shows ALL the gain was deleting calendar-START
memorization; residual position is neutral. Two reframes now < +0.003 → **stop-rule: positional lane
done.** The other big shift channel (per-series AMPLITUDE) is toxic to touch (detrend −0.051, it IS
signal). So "delete a shifted channel" is largely exhausted; budget shifts to the OBJECTIVE lever.
REFINED LAW (keep): a reframe helps ONLY when it deletes a channel actually SHIFTED train-vs-test
(START shifted → won; LENGTH matched → lost; POSITION-as-a-whole neutral → tie).

## EXPERIMENT QUEUE

- ~~Iter2 blend~~ ❌0.8705 · ~~Iter3 detrend~~ ❌0.8266 · ~~Iter4 K=4~~ ❌0.8665 · ~~Iter5 relative-time~~ ✅**0.8908 CHAMPION** · ~~Iter6 TTA~~ ❌0.8885 · ~~Iter7 dnorm~~ ❌0.8844 · ~~Iter8 NoPE~~ ➖0.8917 (tie, FINALIST).
1. **Iter9 — cross-view invariance objective** *(current)*: `seq.consistency_lambda=1.0`. Penalize logit
   variance across a row's K=2 masked views: L=BCE+λ·Var_k(logit). Objective-level, capacity-neutral,
   built on the champion base. `submission_seq_xview.csv`. Gate vs 0.8908; if ≤ champion → ENDGAME.
2. **ENDGAME — one-time prevalence sweep** on the champion (0.62/0.635/0.65/0.665/0.68): 0.649 was tuned
   for the OLD model; free (no retrain), isolates the 60% F1 lever. Pick plateau CENTER. Do ONCE, near deadline.
- **Private-LB finalists (SECURED-ish):** champion relative-time (0.8908) + NoPE (0.8917). Both diverse,
  both ~0.891. NOT the TTA variant (too correlated). Verify Zindi's finalist mechanism (auto best-public
  vs designate two) before deadline 2026-08-16.

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
