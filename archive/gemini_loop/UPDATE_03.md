# Deep-Research Brief — Round #03 (Gemini Deep Research AND Claude Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)
**Date:** 2026-07-09 · **Current best public LB:** **0.8780** · **Deadline:** 2026-08-16

---

## 0. Read this first — how to be useful this round

You are one half of a live competition loop: a coding agent implements + submits, you
do fresh sourced research and react to **leaderboard** results. Your round-2 plan was
thorough but **misfired in specific, costly ways**, listed in §1. Please internalize
those before proposing anything. The overriding rule of this problem:

> **Local cross-validation (OOF) is BLIND to the deciding effect. The leaderboard is the
> only ground truth.** Two models with *identical* OOF differ by **+0.05 on the LB**. Any
> recommendation — especially any model-selection rule — must respect this. Advice of the
> form "trust OOF / reject an LB gain that lowers OOF" is wrong here and will be discarded.

**Hard constraints (violating any makes your idea unusable):**
- Only the supplied competition data. **No external data and no models pretrained on
  other data** — this **bans TabPFN**, ImageNet/SSL backbones, and all foundation models.
  Train from scratch.
- **AutoML banned.** Open-source, seeded, reproducible only.
- `TargetF1` is scored at a **hard 0.5 cut** — no threshold tuning. Base-rate/prior
  correction (shifting the probability so the F1-optimal point lands at 0.5) *is* allowed.
- **Max 5 submissions/day** — the scarce resource. Say which ideas deserve one.

---

## 1. Corrections to your round-2 plan (do not repeat these)

1. **Data facts were fabricated.** Real sizes: **Train = 1821 rows × 146 cols, Test = 1030
   rows × 145 cols** (public LB ≈ 309 rows, private ≈ 721). Your "858 test / 953 train /
   257 public" numbers were invented — reason only from the figures given here.
2. **BBSE / Saerens-EM prior estimation: already built, already failed.** BBSE estimated
   the test positive rate at ~0.44; the true F1-optimal rate is ~0.65 (measured on the LB).
   Covariate shift breaks the label-shift assumption these methods require, and our nets
   aren't calibrated enough for EM. We have already **empirically mapped** the entire
   F1-vs-prior curve on the LB (peak at realized ~0.65, flat within ±0.0004 across
   0.59–0.65, declines after). **Prior estimation is a solved, closed topic — do not
   re-open it.**
3. **WIF / fixed-threshold water-index features: already tested, they HURT the LB** (WIF
   scored AUC 0.83 on train labels but moved the LB −0.005). Your IMNE rule and hardcoded
   SAR thresholds (`vv < −15 & vh < −22`) are the same non-transferable class. Do not
   propose absolute-threshold or region-calibrated spectral features.
4. **TabPFN is disqualified** (pretrained PFN — violates the no-pretraining rule).
5. **OOF-based meta-stacking (Ridge on OOF) is counterproductive here** — since OOF can't
   see transfer, an OOF-weighted stacker mis-weights the models. We use **rank-average
   blending** instead.

---

## 2. State of play (ground truth)

- **Levers fully exhausted:** (a) base-rate/prior correction (+0.11 total, saturated at
  realized ~0.65); (b) model-class swap GBDT→**from-scratch temporal Transformer** (+0.05,
  despite identical OOF). The Transformer (attention over observed months via
  `src_key_padding_mask`, per-band missing-indicator channels, masked-mean-pool,
  masking-augmented training views) is our **base model at LB 0.8780**.
- **Known weakness:** the Transformer is **overconfident** (saturated probs) — a strong
  ranker, poor probability.
- **Field (verified):** top score ≈ **0.9452**, top-5 ≈ **0.928–0.945**. Realistic target
  is **top-5 ≈ 0.928+**, i.e. a real **~+0.05** from here. (A "0.99 LB" is not a goal — it
  would require public-LB overfitting that collapses on the private 70%.)

### LB progression
| Approach | Operating point | Public LB |
|---|---|---|
| GBDT ensemble + prior correction | realized ~0.65 | 0.8260 (GBDT peak) |
| Temporal Transformer | realized ~0.593 | 0.8776 |
| Temporal Transformer | realized 0.649 | **0.8780 (best)** |
| Temporal Transformer | realized 0.672 | 0.8733 |

---

## 3. What we are already doing (go BEYOND these)

1. GBDT + Transformer **rank-average blend**, correlation-gated.
2. Transformer robustness: **EMA/SWA weight averaging, label smoothing, more seed-bagging**,
   and (accepted from your round-2) an **AUC-margin pairwise surrogate loss** on top of BCE.
3. **Transfer-oriented input channels** for the Transformer (not aggregates): **per-cell
   temporal detrending** (subtract each cell's own per-band mean to kill region offset while
   keeping temporal shape), **month-to-month deltas**, and per-month normalized-difference
   indices as extra channels — each LB-gated.

---

## 4. Questions for round-3 (prioritize by expected LB-gain × effort; say which deserve a submission)

1. **Pushing a from-scratch small-n (~1.8k) Transformer from 0.88 → 0.92+ under strong
   covariate shift.** Concrete, sourced architecture/training choices that *provably improve
   transfer* (not fit) for short 4–6-step masked multivariate satellite series — beyond
   EMA/label-smoothing/mixup. What actually won comparable shifted small-n time-series
   competitions?
2. **The most transfer-robust invariance transform for this setup.** Given the shift is
   driven by absolute-level differences (adversarial train-vs-test AUC ≈ 0.99, still ≈0.94
   on region-normalized water indices), rank these by expected transfer and give recipes:
   per-cell temporal detrend/standardize, month-to-month differencing, instance
   normalization, rank/quantile transforms, spectral-shape (harmonic) descriptors invariant
   to which months are observed.
3. **A structurally different, from-scratch third learner** most likely to add *decorrelated*
   signal to a Transformer+GBDT rank-blend on masked series: 1D-CNN/TCN, masked GRU,
   DeepSet/set-transformer over observed months, or ROCKET/MiniRocket + linear head (note: is
   MiniRocket's random-kernel bank rule-legal given "no pretraining"?). Which single one is
   the best bet per submission spent?
4. **Calibrating an overconfident from-scratch net so the fixed-0.5 F1 cut is robust on a
   ~721-row private split** — temperature scaling vs beta/Platt on OOF vs our logit-shift +
   prior. Does it change ranking (AUC/TargetRAUC)? Which survives *covariate* shift, not just
   label shift?
5. **Public-vs-private hedging** with a designed shift this strong and only ~309 public rows:
   submission-selection and seed-averaging strategies that protect the private score; what
   winners of similarly-shifted Sentinel-1/2 remote-sensing competitions actually did.
6. **Anything specific** to FAO/ITU aquaculture pond phenology detectable from a 4–6-month,
   12-band window with **no spatial neighborhood and no lat/lon** — temporal signatures
   invariant to *which* months are visible.

If you disagree with any conclusion above, argue it explicitly with sources — we would
rather be corrected than agreed with. Return prioritized, implementable, rule-legal
recommendations that go beyond §3.
