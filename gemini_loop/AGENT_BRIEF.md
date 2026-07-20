# AGENT_BRIEF — standing directive for the LB-gated autoresearch loop

*This is the `program.md` of this project (cf. `karpathy/autoresearch`). Any fresh
agent session reads THIS file first, picks the next experiment, edits
`config/config.yaml` + `experiments/run_current.sh`, commits + pushes. A human runs
it on Colab/Kaggle and pastes the leaderboard score into `experiments/LB_LOG.md`.
Update this file whenever a result changes the queue.*

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

## EXPERIMENT QUEUE (what has NOT been tested on the LB), in order

1. **Step 2 — GBDT+seq rank-average blend.** `run_current.sh` as staged. Gate on the
   `OOF rank correlation ρ` line: ρ<~0.90 → upload blend file nearest pos-rate 0.65;
   ρ≈1.0 → skip. *(current iteration)*
2. **Step 3 — invariant channels**, ONE LB probe each, `seq.channels.*`, `--model seq`,
   order `per_cell_detrend → deltas → indices → rank`. Keep any that beat 0.8780.
3. **Step 1 — `prevalence_target`** (e.g. 0.65): exact pos-rate on the overconfident net;
   A/B vs `assumed_test_prior` at the same operating point.
4. **Seq robustness (unbuilt):** EMA/SWA, label smoothing, more seed-bagging, AUC-margin
   pairwise loss — behind config flags.
5. **Third learner** for the blend (1D-CNN/TCN or masked GRU) — only if Step 2 helps.

## Per-iteration protocol

1. Read `LB_LOG.md` + `experiments/results.tsv` + this file → pick next queue item.
2. Edit `config/config.yaml` + `experiments/run_current.sh`; commit + push.
3. Human: Colab/Kaggle **Run all** → upload best CSV to Zindi → paste LB into `LB_LOG.md`.
4. Update this queue (keep/discard on the pasted LB) and advance.
