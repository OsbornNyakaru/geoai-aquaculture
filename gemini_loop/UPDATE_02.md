# Deep-Research Brief — Round #02 (for Gemini Deep Research AND Claude Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)
**Date:** 2026-07-08 · **Current best public LB:** **0.8780** (up from 0.7561 last round) · **Deadline:** 2026-08-16

---

## 0. How to use this document

This is a **live ML-competition improvement loop**. A coding agent (Claude Code)
implements and submits; you (a deep-research model) react to leaderboard results,
do fresh sourced research, and return **prioritized, concretely-implementable,
rule-legal** next steps ranked by *expected LB gain × implementation effort*.
**Treat leaderboard movement as ground truth** — our local cross-validation is
provably blind to the deciding effect (see §3).

**Hard constraints (never violate; flag anything that would):**
- Only the supplied competition data. **No external data and no models pretrained
  on other data** (this rules out satellite foundation models / ImageNet / any
  transfer weights). Training from scratch on the provided data only.
- **AutoML is banned.**
- `TargetF1` must be the **0.5 cut of a probability** — no post-hoc threshold
  tuning. (Base-rate / prior correction *is* allowed: it is a modeling choice that
  shifts the probability, and the cut stays at 0.5.)
- Everything seeded and reproducible; open-source only.
- **Max 5 submissions/day.** Submissions are the scarce resource — say which ideas
  are worth spending one on.

---

## 1. The problem (self-contained recap)

- **Task:** binary classification — is a given 10 m × 10 m cell an aquaculture
  pond? Train n≈1,817 (after dedup), test n=1,030. Public LB ≈ 30% of test (~309
  rows); private is the rest.
- **Metric:** `0.6 · F1 + 0.4 · ROC-AUC`. Two separate submission columns are
  allowed: `TargetF1` (binary, scored at a hard 0.5) and `TargetRAUC` (any
  rank-preserving score).
- **Data:** per cell, a **12-month × 12-band** time series — Sentinel-1 SAR (VH,
  VV, in dB) + 10 Sentinel-2 optical bands. **No latitude/longitude** (removed by
  organizers), **no spatial neighborhood** (single isolated cell), no static
  covariates.
- **The core trap — temporal masking:** train rows are **fully observed (12
  months)**; test rows expose only a **consecutive 4/5/6-month window** (measured
  distribution ≈ uniform over {4,5,6}), with the rest set to a −9999 sentinel.
  Sentinel-2 months additionally have cloud dropout (per-month rates measured from
  the test set; S1 is more complete than S2). So test inputs are *systematically
  more masked* than train.
- **Designed domain shift:** train and test come from **different time periods and
  different pilot regions**. An adversarial classifier separates (masked-to-match)
  train from test at **AUC ≈ 0.99** (≈ 0.94 using only region-normalized water
  indices). This is genuine covariate shift, **not a pipeline leak** — we proved
  the CV is leak-free (temporal views of a row never straddle the train/val
  boundary; there is no group ID to leak on). **Do not propose group-KFold or
  assume leakage** — that was last round's wrong turn.

---

## 2. Leaderboard progression so far (ground truth)

| Approach | Operating point | Public LB |
|---|---|---|
| GBDT ensemble, inherited train prior | pos-rate 0.40 | 0.7140 |
| GBDT + base-rate/prior correction | pos-rate 0.50 | 0.7561 |
| GBDT + prior correction | **pos-rate ~0.65** | **0.8260** (GBDT peak) |
| GBDT + prior correction | pos-rate 0.70 / 0.75 / 0.80 | 0.8216 / 0.8166 / 0.8037 |
| **Temporal Transformer (from scratch)** | realized pos-rate 0.627 | **0.8732** |
| **Temporal Transformer** | realized pos-rate 0.649 | **0.8780 (current best)** |

**Field context:** ~180 competitors; the top pack is dense at **0.88–0.945**
(top-5 ≈ 0.928–0.945, top-10 ≈ 0.915, rank-50 ≈ 0.876). We just moved from rank
~187 into striking distance of the pack, but **top-5 still needs roughly +0.05.**

---

## 3. The two big findings since last round (please internalize)

**(A) Base-rate/prior correction is fully exploited and now SATURATED.** The
F1-vs-assumed-prior curve peaks at a realized test positive rate of ~0.65 (LB
0.826 for the GBDT) and falls off on both sides. The test set is genuinely much
more positive than train (~40%). This lever took us 0.714 → 0.826 and is done;
further prior tuning yields ≤ +0.005.

**(B) THE BREAKTHROUGH — a from-scratch temporal Transformer beats the tuned GBDT
ensemble by ~+0.05 on the LB, despite IDENTICAL local CV.**
- Both models score **OOF combined ≈ 0.982** (F1 ≈ 0.974, AUC ≈ 0.995) — locally
  indistinguishable. Yet on the LB the Transformer is **0.878 vs the GBDT's
  0.826.** The GBDT's flattened temporal *aggregates* over-fit the source
  distribution; the Transformer's **attention over only the observed months**
  (via a `src_key_padding_mask`, with explicit per-band missing-indicator
  channels) is the inductive bias that actually *transfers* across the designed
  domain shift. This is the strongest possible confirmation that **OOF is blind
  here and only the LB ranks approaches.**
- The Transformer is now our **base model**, not a side experiment.

**Architecture (all trained from scratch on provided data only):** input =
[standardized 12 bands ⊕ 12 missing-indicators] per month → linear projection to
d_model=64 → learned positional embedding → 2-layer TransformerEncoder (4 heads,
GELU, dropout 0.2, `src_key_padding_mask` over fully-masked months) →
masked-mean-pool over observed months → 2-layer MLP head → sigmoid. Trained with
AdamW + BCE, masking-augmented views of train rows (K views per row matching the
test masking recipe), 5-fold (test prediction = mean of the 5 fold-models),
n_repeats=1. ~24 input features/month, ~1.8k train rows.

**Known weakness of the Transformer — overconfidence.** Its test probabilities are
saturated near 0 and 1, so the prior/base-rate lever barely moves its realized
positive rate (assumed prior 0.65→0.80 shifted realized only 0.593→0.627; to reach
realized 0.65 we needed assumed ~0.90). It is a strong *ranker* but a *poorly
calibrated* probability. We currently reach the fixed-0.5 operating point by
cranking the assumed prior; a properly calibrated model might do better and would
make the F1 cut more robust on the private split.

---

## 4. What we are about to try (already planned — go BEYOND these)

1. **Rank-average blend** of the Transformer (heavier weight) + GBDT, re-calibrated
   at the ~0.65 operating point. Rank-average specifically to neutralize the
   Transformer's overconfident magnitudes. (Built; correlation-gated.)
2. **Seed-bagging / more repeats** for the Transformer (n_repeats 1→3) to steady
   the averaged test probabilities.
3. Conservative capacity/regularization changes to the Transformer (guarded,
   because local CV can't tune them — every change costs an LB submission to judge).

---

## 5. Deep-research questions (please return prioritized, sourced, implementable answers)

Rank each by **expected LB gain × feasibility**, note which deserve a scarce daily
submission, and give concrete recipes / hyperparameters where you can.

1. **How do we push a from-scratch temporal Transformer from ~0.88 toward ~0.92+
   under strong covariate shift at n≈1,800?** Specifically: architectural and
   training choices that *improve generalization across domain shift* (not just fit)
   for short (4–6 step) irregular multivariate satellite series — e.g. attention
   variants for masked/irregular time series, time-aware positional encodings,
   heavy dropout / stochastic depth / weight averaging (SWA/EMA), mixup on the
   masked views, label smoothing, ensembling many seeds vs deeper models. What has
   actually worked in comparable small-n shifted tabular/time-series competitions?

2. **Probability calibration of an overconfident neural net when the F1 cut is
   fixed at 0.5.** Given a strong *ranker* whose probabilities are saturated, what
   is the best rule-legal way to make the 0.5 cut land at the F1 optimum *and be
   robust on the ~309-row public / larger private split*? Compare temperature
   scaling, Platt/beta calibration on OOF, vs our current logit-shift + base-rate
   correction. Does calibrating change the *ranking* (and thus AUC / TargetRAUC)?
   Which calibration best survives label + covariate shift?

3. **Model diversity for the blend.** Beyond a Transformer + GBDT rank-average,
   what *structurally different, from-scratch* learners most reliably add
   decorrelated signal on masked multivariate satellite series: 1D-CNN /
   temporal-convolutional net, ROCKET/MiniRocket + linear head, an RNN/GRU with a
   masking mask, or a set-transformer/deepset over observed months? Which are
   worth building given each blend member costs training + at least one LB probe?

4. **Test-time adaptation / self-training on the unlabeled test set** (uses only
   supplied data — is it advisable, and how to do it *safely* here?). Given
   adversarial train-vs-test AUC ≈ 0.99, do pseudo-labeling, entropy
   minimization, BatchNorm/stat adaptation, or CORAL-style feature alignment help
   or hurt at n≈1,030 test? Concrete, low-risk recipe if yes.

5. **Transferable temporal signatures of *managed* aquaculture ponds** (vs natural
   water / seasonal flooding) that a network can exploit from a 4–6-month, 12-band
   window with no spatial context: stocking/drain/harvest phenology, water-presence
   persistence, SAR temporal behavior, harmonic descriptors — and crucially which
   of these are **invariant to which months happen to be observed** (the visible
   window varies per test row). We already tried a "water-inundation-frequency"
   count feature: strong on train labels (AUC 0.83) but it did **not** transfer on
   the LB — so we are wary of features that are discriminative in-domain but
   region/season-specific.

6. **Jointly maximizing 0.6·F1 + 0.4·AUC** with separate `TargetF1`/`TargetRAUC`
   vectors. Is there a principled construction that squeezes more than treating
   them independently (calibrated 0.5-cut for F1; max-spread ranks for AUC)?

7. **Reality check & overfitting management.** With a designed shift this strong
   and a ~309-row public LB, how much of the 0.88→0.94 gap to the leaders is
   plausibly *method* vs *public-LB overfitting*? What did winners of similarly
   shifted Sentinel-1/2 or remote-sensing time-series competitions actually do, and
   how should we hedge public-vs-private (submission selection, seed-averaging,
   trusting the more conservative operating point)?

8. **Anything specific** to the FAO/ITU GeoAI Aquaculture challenge or to winning
   Zindi Sentinel time-series solutions that we should know and are missing.

Please return findings so Claude can implement the top items in the next loop
iteration. If you disagree with any conclusion above, say so explicitly and show
your reasoning — we would rather be corrected than polite.
