# Deep-Research Brief — Round #05 (Gemini Deep Research AND Claude Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)
**Date:** 2026-07-21 · **Current best public LB:** **0.8908** (NEW — up from 0.8780) · **Deadline:** 2026-08-16

---

## 0. Read this first — how to be useful this round

Live competition loop: a coding agent implements + submits, you do fresh sourced research and
react to **leaderboard** results. **Round 04 broke a 10-day plateau: your recommended
"relative-time reframing" WON (+0.0128).** We need you to find the NEXT change of that *kind*.

> **The overriding rule of this problem: local cross-validation (OOF) is BLIND to the deciding
> effect — often ANTI-correlated with it. The leaderboard is the only ground truth.** Sixth
> confirmation this round: the new champion's OOF (0.9811) was *lower* than the model it beat
> (0.9827). Any recommendation that leans on OOF for selection is discarded.

**Hard constraints (violating any makes your idea unusable):**
- Only the supplied competition data. **No external data, no models pretrained on other data**
  — bans TabPFN, ImageNet/SSL backbones, all foundation models. Train from scratch.
- **AutoML banned.** Open-source, seeded, reproducible only.
- `TargetF1` scored at a **hard 0.5 cut** — no threshold tuning. Prior/prevalence correction
  (a monotone shift so the F1-optimum lands at 0.5) *is* allowed and already optimal at 0.649.
- **Max 5 submissions/day** — the scarce resource. Say which ideas deserve one.

---

## 1. Results since round 04 (ground truth)

Champion reference each row is isolated against, operating point held at realized pos-rate 0.649
via an exact-prevalence logit shift (so each change is the ONLY variable).

| # | Change tested (only variable vs the then-champion) | OOF combined | Public LB | Δ | Verdict |
|---|---|---|---|---|---|
| 5 | **Relative-time reframing** — left-align each observed 4–6mo window to t_rel=0 so the learned positional embedding encodes RELATIVE step, not absolute calendar month. Capacity-neutral (no new params/dims). | 0.9811 | **0.8908** | **+0.0128** | ✅ **NEW CHAMPION** |
| 6 | **MC temporal-dropout TTA** — inference-only; predict clean view + 8 views each masking 1–2 random observed months, soft-vote. Capacity-neutral. | — | 0.8885 | −0.0023 | ❌ within noise, did not beat champion |

**Relative-time (#5) is the headline.** It directly attacked calendar-month memorization — the
model was partly keying on WHICH calendar months a window covered (a region/time artifact of the
designed shift) instead of the relative dynamics within the window. Removing absolute calendar
position, with zero added capacity, transferred for +0.0128 (clears the ±0.01 public-LB noise).

---

## 2. The findings you must internalize (updated)

**(A) Capacity-neutral STRUCTURAL REFRAMES transfer; added capacity and robustness moves do not.**
The full ledger now reads:
- Added a model class (GBDT blend): **−0.0075**.
- Added input channels (per-cell detrend): **−0.0514**.
- Added augmentation (K=2→4): **−0.0115**.
- Added inference-time robustness (TTA): **−0.0023** (within noise).
- **Reframed the coordinate system (calendar→relative time), same parameters: +0.0128.**
⇒ The productive lane is **changing the model's inductive bias / coordinate frame to DELETE a
covariate-shift memorization channel** — not adding models, channels, augmentation, or ensembling.

**(B) The transferable axis is TEMPORAL/POSITIONAL, not AMPLITUDE.** This is the sharp new
constraint. Relative-time (remove absolute *position*) WON +0.013. Per-cell detrend (remove
per-series *amplitude/level*) LOST −0.051. So: **do NOT propose amplitude/level normalization**
(instance-norm, detrend, differencing, per-window standardization of band values) — that axis is
empirically toxic for this model. Propose reframings of *time/position/order/duration*.

**(C) MEASUREMENT RESOLUTION is binding.** Public LB ≈ **309 rows** → ~±0.01 noise. We can only
detect *large* effects (relative-time +0.013, GBDT→seq +0.05) or *breakages* (detrend −0.05).
Robustness/variance-reduction moves (TTA; likely multi-seed bagging) are unresolvable on public —
we are NOT spending submissions to A/B inside the noise. **Only propose ideas with a plausibly
LARGE (≥ ~0.01) effect.**

---

## 3. Corrections / do-not-re-propose

1. **Amplitude/level normalization is empirically toxic** (detrend −0.051). Do not re-recommend
   detrend / instance-norm / differencing / per-window value standardization in ANY form (added
   OR replacing). If you think level-invariance matters, you must explain why it wouldn't repeat
   the −0.051 failure — a very high bar.
2. **Robustness/ensembling is within-noise here** (TTA −0.0023). Multi-seed bagging, SWA/EMA,
   test-time augmentation are private-LB insurance at best, not public-LB levers. Don't pitch them
   as ways to climb.
3. Standing dead ends: BBSE/EM prior estimation · WIF / fixed-threshold water features · TabPFN /
   any pretrained model · temperature scaling · importance-weighting / DANN (ESS collapse at
   adversarial AUC 0.99) · OOF meta-stacking · group-KFold / "it's leakage" (designed covariate
   shift, proven leak-free).

---

## 4. Self-contained problem recap (for a fresh reader)

- **Task:** binary classification — is a 10 m×10 m cell an aquaculture pond? Train 1817 rows
  (post-dedup), test 1030 (public ≈309, private ≈721). **Metric: 0.6·F1 + 0.4·ROC-AUC**, two
  columns: `TargetF1` (binary, hard 0.5 cut) and `TargetRAUC` (any rank-preserving score).
- **Data:** per cell a **12-month × 12-band** series — Sentinel-1 SAR (VH, VV, dB) + 10
  Sentinel-2 optical bands. **No lat/lon, no spatial neighborhood, no static covariates.**
- **The core trap — temporal masking:** train rows are fully observed (12 months); test rows expose
  only a consecutive **4/5/6-month** window (rest = −9999), plus extra Sentinel-2 cloud dropout.
  We augment each train row into K=2 masked "views" matching the measured test masking recipe.
- **Designed domain shift:** train/test are different time periods and pilot regions; an adversarial
  classifier separates them at AUC ≈0.99. Genuine covariate shift, proven leak-free.
- **Champion model (NOW):** from-scratch Transformer — per-month [standardized 12 bands ⊕ 12
  missing-indicators] → linear proj d=64 → **learned positional embedding (length 12)** → 2-layer
  encoder (4 heads, GELU, dropout 0.2, `src_key_padding_mask` over fully-masked months) →
  masked-mean-pool → MLP head → sigmoid. AdamW+BCE, K=2 masking-augmented views, 5-fold (test =
  mean of fold-models), n_repeats=1. **NEW: the observed window is left-aligned to t_rel=0 before
  the positional embedding is applied (relative-time reframing).** Known weakness: overconfident
  (saturated probs) — strong ranker, poor probabilities; operating point fixed via prevalence shift.
- **Field (verified):** top ≈0.9452, top-5 ≈0.928–0.945, rank-50 ≈0.876. We are 0.8908, so the gap
  to top-5 is now **~+0.037**.

---

## 5. Research questions (prioritize by expected LB-gain × feasibility; say which deserve a submission)

1. **THE MAIN ASK — what is the next capacity-neutral, POSITIONAL/TEMPORAL structural reframe?**
   Relative-time (left-align to window start) gave +0.013 by removing absolute-calendar
   memorization. What related reframings of *time/order/duration* would remove a further
   covariate-shift channel WITHOUT adding parameters or channels, and WITHOUT touching amplitude?
   Candidates to evaluate (add your own, ranked by expected effect size):
   - **Duration-normalized positions** — index the positional embedding by fractional position
     scaled to window length, so a 4-month and 6-month window share one [0,1] relative frame
     (removes window-*length* memorization on top of window-*start*).
   - **Center-aligned / symmetric relative frame** vs left-aligned — does aligning to window center
     or to a phenology anchor beat aligning to the first observed month?
   - **Order/permutation reframing** — e.g. remove the positional embedding entirely (treat the
     window as an unordered set of observed months + a scalar duration), forcing reliance on
     values/dynamics rather than position. Would this over-remove real temporal signal, or transfer
     better? Argue both ways.
   - **Learned RELATIVE positional encoding** (attention bias on step-distance) — but only if it can
     be done without meaningfully adding parameters/capacity (given (A), pure-add is low-prior).

2. **Are there LARGE-effect levers of a DIFFERENT structural kind** (the GBDT→seq swap was +0.05)?
   Given "added capacity hurts," "OOF is blind," and "amplitude-norm is toxic," what *structurally
   different from-scratch* architectures or training objectives have produced step-changes under
   strong covariate shift on short (4–6-step) masked multivariate satellite series? (e.g. a
   phenology-anchored temporal encoder, an ordinal/rank-based objective, a contrastive
   invariance objective across masked views of the SAME row.) Rank by expected effect size.

3. **Does the prevalence operating point interact with the new champion?** We hold realized
   pos-rate at 0.649 (tuned for the OLD 0.8780 model). Could the relative-time model's optimal
   operating point have shifted? A prior sweep (0.63/0.65/0.67) is cheap (no retrain, monotone
   re-shift, multiple files from one run). Worth one submission-day, or within noise?

4. **Private-LB robustness / submission selection.** With a designed shift this strong and only
   ~309 public rows, how do we choose the final 1–2 submissions to protect the **private ≈721**?
   Is 0.8908 public likely to hold privately? Given TTA (0.8885) is a low-diversity variant of the
   champion, is it a useful hedge, or should the two final picks be more diverse (e.g. champion +
   a genuinely different structural reframe)?

If you disagree with any conclusion here — especially "amplitude-norm is toxic," "robustness is
within-noise," or "positional reframing is the lane" — argue it explicitly with sources and a
concrete, rule-legal, single-submission test. We would rather be corrected than agreed with.
Return prioritized, implementable recommendations.
