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

- ~~**Step 2 — GBDT+seq rank-average blend.**~~ **DONE → DISCARDED (2026-07-20).** ρ=0.849
  (decorrelated), yet best blend **0.8705 < 0.8780**. GBDT dilutes seq's transfer despite
  higher OOF AUC. Lesson: don't blend in other model *classes*; improve the seq model
  itself. (Also confirmed the Colab env is faithful — blend landed between components.)
1. **Step 3 — invariant channels**, ONE LB probe each, `seq.channels.*`, `--model seq`,
   held at `prevalence_target 0.649` so the channel is the only variable vs the 0.8780
   reference. Order `per_cell_detrend → deltas → indices → rank`. Keep any that beat 0.8780.
   *(current: per_cell_detrend → submission_seq_detrend.csv)*
2. **Step 1 — `prevalence_target`** is now being exercised as the operating-point tool for
   Step 3; once a channel is chosen, A/B its exact-0.649 vs `assumed_test_prior` if useful.
3. **Seq robustness (unbuilt):** EMA/SWA, label smoothing, more seed-bagging, AUC-margin
   pairwise loss — behind config flags.
4. **Third learner** — only a *from-scratch seq-family* variant (1D-CNN/TCN, masked GRU) is
   worth blending; GBDT is ruled out (Step 2). Diversity within the better-transfer class.

## Per-iteration protocol

1. Read `LB_LOG.md` + `experiments/results.tsv` + this file → pick next queue item.
2. Edit `config/config.yaml` + `experiments/run_current.sh`; commit + push.
3. Human: Colab/Kaggle **Run all** → upload best CSV to Zindi → paste LB into `LB_LOG.md`.
4. Update this queue (keep/discard on the pasted LB) and advance.
