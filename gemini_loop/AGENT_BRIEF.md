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

## CURRENT STATE: iter5 WON; banking robustness on the new champion (iter6 TTA)

Relative-time is now the champion (`seq.relative_time: true`). Per the post-win plan, bank the two
capacity-neutral robustness moves on top, one LB probe each: MC temporal-dropout TTA (iter6, now),
then multi-seed bagging (iter7). Both are variance/private-LB insurance and may land within public
noise — gate for "no real regression," keep as standing defaults. Still: ONE variable per probe,
held at prevalence_target 0.649. Do NOT big-bang refactor; rejected dead-ends stay rejected.

## EXPERIMENT QUEUE

- ~~Iter2 blend~~ ❌0.8705 · ~~Iter3 detrend~~ ❌0.8266 · ~~Iter4 K=4~~ ❌0.8665 · ~~Iter5 relative-time~~ ✅**0.8908 CHAMPION**.
1. **Iter6 — MC temporal-dropout TTA** *(current)*: `seq.tta.enable=true`, inference-only — mask
   1–2 active months per test row, soft-vote 8 views + clean. No added capacity. `submission_seq_reltime_tta.csv`.
   Gate vs 0.8908 (keep if no real regression; it's private-split insurance).
2. **Multi-seed bagging** (`seq.n_repeats↑`, no added dims): variance reduction for the noise floor.
3. **Next structural reframe** (extend the iter5 win): other capacity-neutral inductive-bias changes
   that delete a covariate-shift channel (e.g. per-window feature standardization, duration-invariant
   pooling). Highest-value direction — hunt large effects, not toggles.
4. **Private-LB submission selection** as deadline nears (UPDATE_04.md Q4).

## Per-iteration protocol

1. Read `LB_LOG.md` + `experiments/results.tsv` + this file → pick next queue item.
2. Edit `config/config.yaml` + `experiments/run_current.sh`; commit + push.
3. Human: Colab/Kaggle **Run all** → upload best CSV to Zindi → paste LB into `LB_LOG.md`.
4. Update this queue (keep/discard on the pasted LB) and advance.
