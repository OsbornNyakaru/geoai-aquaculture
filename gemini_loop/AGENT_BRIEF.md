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
**Best public LB:** 0.8780 (from-scratch temporal Transformer). **Deadline:** 2026-08-16.
**Target:** ~+0.05 (top-5 ≈ 0.928+).

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

- **Levers exhausted:** (1) prior/base-rate correction (+0.11 total, saturated at
  realized pos-rate ~0.65); (2) GBDT → temporal Transformer swap (+0.05 at identical
  OOF). Transformer = base model at **0.8780**; it is a strong ranker but **overconfident**
  (saturated probs).
- **Verified inert-by-default & ready to probe:** Step 1 `prevalence_target`
  (`src/calibration.py`), Step 3 `seq.channels.*` (`src/seq_model.py`).

## DEAD ENDS — do not re-propose (already tried, failed, or rule-illegal)

BBSE/EM prior estimation · WIF / fixed-threshold water-index features · TabPFN
(pretraining) · temperature scaling · importance-weighting / DANN (ESS collapse at
adversarial AUC 0.99) · OOF meta-stacking (Ridge on OOF) · group-KFold / "it's leakage"
(the gap is designed covariate shift, proven leak-free).

## META-LESSON (2026-07-20, after THREE failed probes → loop paused for research)

Three consecutive blind toggles all LOST on the LB while OOF stayed flat/high:
- Iter2 GBDT+seq blend → 0.8705 (adding a model class dilutes seq transfer).
- Iter3 per_cell_detrend → 0.8266, −0.0514 (adding input channels overfits source).
- Iter4 seq K=2→4 → 0.8665, −0.0115 (even MORE of the winning lever overshoots).

Three hard conclusions:
1. **Added capacity hurts** (extra model / extra channels / extra augmentation all lost).
   The additive-channel family (`deltas`/`indices`/`rank`) is now **low-prior**.
2. **OOF is anti-correlated**, not merely blind: highest-OOF run (K=4, 0.984) = 2nd-worst LB.
3. **Measurement resolution is the binding constraint.** Public LB ≈309 rows → ~±0.01 noise.
   Single-submission A/B **cannot resolve small (+0.005) gains** — only large effects
   (GBDT→seq was +0.05) or breakages (detrend −0.05). Stop hunting incremental toggles.

**Champion (unchanged): seq K=2 @ realized 0.649 = 0.8780.** Standing operating-point tool:
`prevalence_target 0.649` (holds any probe at the exact champion pos-rate for clean isolation).

## CURRENT STATE: round-04 research triaged (`RESPONSE_04.md`); testing idea A

Rejected from the round-04 report as proven dead-ends/rule-illegal: **Saerens-EM prior**
(covariate≠label shift; `prevalence_target 0.649` already optimal) and **Zou-threshold / EVI
index projection** (hardcoded-threshold class, non-transferable; EVI already failed). Do NOT
big-bang refactor. Test ONE capacity-neutral idea at a time, held at prevalence_target 0.649.

## EXPERIMENT QUEUE

- ~~Iter2 blend~~ ❌0.8705 · ~~Iter3 detrend~~ ❌0.8266 · ~~Iter4 K=4~~ ❌0.8665.
1. **Iter5 — relative-time reframing** *(current)*: `seq.relative_time=true` left-aligns each
   observed window to t_rel=0 (positional emb sees relative step, not calendar month). Capacity-
   neutral, plausibly LARGE effect. `submission_seq_reltime.csv`. Gate vs 0.8780.
2. **MC temporal-dropout TTA** (inference-only, no added capacity): mask 1–2 active months per
   test row, soft-vote N views. Safest next probe if iter5 fails.
3. **Multi-seed bagging** (`n_repeats↑`, no added dims): variance reduction for the noise floor.
4. **Private-LB submission selection** as deadline nears (UPDATE_04.md Q4).

## Per-iteration protocol

1. Read `LB_LOG.md` + `experiments/results.tsv` + this file → pick next queue item.
2. Edit `config/config.yaml` + `experiments/run_current.sh`; commit + push.
3. Human: Colab/Kaggle **Run all** → upload best CSV to Zindi → paste LB into `LB_LOG.md`.
4. Update this queue (keep/discard on the pasted LB) and advance.
