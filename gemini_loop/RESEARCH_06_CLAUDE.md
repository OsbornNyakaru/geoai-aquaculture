# Research Round 06 — findings by the in-loop agent (Claude Code)
**Date:** 2026-07-21 · To be triaged in `RESPONSE_06.md` alongside the Claude Fable Deep Research
report. Same brief (`UPDATE_06.md`), independently researched with live web access.

---

## 0. RULE FINDINGS FIRST — two facts that change the plan (from the Zindi competition page)

### 0.1 The finalist question is ANSWERED: we designate 2 submissions ✅
Verbatim from the rules: *"Before the end of the challenge you need to choose 2 submissions to be
judged on for the private leaderboard. If you do not make a selection your 2 best public leaderboard
submissions will be used."*

- **The NoPE diversity hedge is USABLE and valuable.** Endgame action confirmed: designate
  **champion (xview λ=1.0, 0.8955) + NoPE (0.8917)** on the Zindi site before close.
- Caution for the endgame: the *default* is "2 best public" — so if the prevalence sweep produces a
  file that luckily out-scores the champion on public noise, the default would pick it. **We must
  designate manually**, not rely on the default.

### 0.2 Submission budget is capped at 100 TOTAL, not 5/day × days
The rules impose **max 100 submissions overall** (plus the 5/day limit). We have used ≈20 (Phase 1:
7 · Phase 2: 4 · iters 2–10: 9). **≈80 remain** — still ample, but "130" in our planning docs was
wrong and any screening protocol must respect the real cap.

### 0.3 Phase-2 rubric = 35% of the final score (top-5 only)
Final ranking is **65% leaderboard + 35% rubric** (reproducibility & innovation: 9–10 points for
"reproducible, workflow well aligned with the proposed approach"). Our loop is unusually strong
here: fully seeded, config-hashed caches, every experiment LB-logged, `PROJECT_STATE.md` as a
workflow narrative. **If we reach top-5, the rubric could move us past teams that beat us on LB.**
Queued (zero submissions): near the deadline, prepare a clean reproduction README + workflow
writeup. A ~0.92 public score could plausibly win overall on rubric strength.

---

## Q1 — A local signal that predicts the LB (the highest-value lane)

### 1.1 ATC — Average Thresholded Confidence ⭐ fund now (costs zero submissions)
[Garg et al., ICLR 2022](https://arxiv.org/abs/2201.04234). Learn a threshold `t` on the model's
confidence on **source** (our OOF) such that the fraction of OOF rows with confidence > t equals OOF
accuracy; then **predicted test accuracy = fraction of unlabeled TEST rows with confidence > t**.
2–4× more accurate than prior estimators under real dataset shifts.

- **Why it fits us exactly:** needs only `oof_prob, y, p_test_raw` — which our pipeline already
  persists in `submissions/preds/preds_*.npz` for every run. **We can retro-fit ATC to past runs
  and validate it against the 10 known (change → LB) outcomes before trusting it.** Any preds
  bundle no longer on the Colab VM is regenerable: every past config state is a git commit, and a
  full run is ~7 minutes with no submission cost.
- Adaptation: ATC predicts accuracy; our metric is 0.6·F1+0.4·AUC. Use it as a **ranking signal
  across candidates**, not a point estimate — that is all we need.

### 1.2 Seed-pair disagreement on test (GDE / agreement-on-the-line) ⭐ fund now
[Jiang et al. 2022](https://arxiv.org/pdf/2202.01851), [Baek et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/7a8d388b7a17df480856dff1cc079b08-Paper-Conference.pdf).
Train the **same candidate twice with different seeds**; the **disagreement rate of the pair on the
1030 unlabeled test rows** estimates its test error (Generalization Disagreement Equality). Two
seeds ≈ 14 min of compute, **zero submissions**.
- Caveat to respect: GDE holds when the ensemble is well-calibrated *on average*; our net is
  overconfident. Two mitigations: (a) the cross-view penalty already reduced saturation, (b) again,
  use it **only to rank candidates**, and retro-validate on the 10 known outcomes first.
- Bonus: the same two runs give us the **champion seed-replication measurement** we already queued
  — one stone, two birds (the 2-submission version tells us LB noise; the 0-submission disagreement
  tells us a validator).

### 1.3 Importance-weighted OOF (IWCV) — fund only as a third opinion
[Sugiyama et al., JMLR 2007](https://jmlr.org/papers/v8/sugiyama07a.html): unbiased under covariate
shift but **unbounded variance** — and at adversarial AUC 0.99 the raw weights are near-degenerate.
If used at all: self-normalized (Hájek) weights + clipping at the ~95th percentile + a temperature
on the discriminator, and only ever as a *ranking* signal. Strictly lower priority than 1.1/1.2;
worth computing only because it is nearly free once the discriminator exists (we already have the
adversarial-validation code path).

### 1.4 The protocol that ties Q1 together (proposed iter11, ~zero submissions)
1. Regenerate/collect `preds_*.npz` for the 10 historical runs.
2. Compute ATC, seed-pair disagreement (for 2–3 key models), and stabilized IW-OOF for each.
3. **Score each validator by Spearman correlation with the 10 known LB scores.**
4. If any validator ranks ≥8/10 correctly → adopt it as the offline screen; the ~80 remaining
   submissions become a real search budget. If none do → we have spent nothing and learned that
   the noise floor stands.

---

## Q2 — Features inside the sequence model

### 2.1 The cross-pol difference VH−VV (dB) — the one channel with a physical invariance argument ⭐
Because our SAR is in **dB**, the cross-pol *ratio* is the *difference* VH−VV. It cancels
**common-mode radiometric offsets** (calibration bias, incidence-angle level effects — the very
stuff of a per-series level shift) while remaining a *physical* quantity: low over smooth open
water (surface scattering suppresses VH), higher over vegetation
([IGARSS 2026 flood-mapping work](https://arxiv.org/pdf/2605.02153), [SentiWiki](https://sentiwiki.copernicus.eu/web/s1-applications)).
- **Why this is not detrend −0.0514:** detrend subtracted a *per-series statistical mean over
  time*, destroying the absolute level that carries class signal. VH−VV is a *per-month, cross-band
  physical* contrast — absolute level survives in the other channels; only the S1 common-mode is
  cancelled *within the ratio channel itself*.
- **Implementation discipline:** REPLACE, don't add (additions always lost). E.g. swap the VH
  channel for VH−VV, keeping 24 channels — capacity-neutral. Single isolated change, gate on LB.
- Honest counter-argument: the encoder sees standardized VH and VV and can learn the difference
  linearly, so the gain — if any — comes only from *deleting* the shifted common-mode reading of
  raw VH, per our design law. Effect size uncertain; this is a "fund if Q1 succeeds" unless Fable's
  report independently ranks it top.

### 2.2 The pond physical signature, from the literature
[Ottinger et al. 2017, Remote Sensing 9(5):440](https://www.mdpi.com/2072-4292/9/5/440) (the
canonical Sentinel-1 aquaculture paper): ponds are **"almost constantly filled with water,
partially or fully drained during harvest"** — the discriminator vs natural water is *permanence +
management events*; vs rice paddies it is the *absence of a vegetation-growth phase* (rice: flood →
canopy rise in VH/optical → drain; ponds: persistently low, flat backscatter with a transient
drain spike). Temporal **median/percentiles** are the standard permanence features.

### 2.3 The structural gap this exposes: masked MEAN-pool dilutes event signatures ⭐ candidate iter12
A 1-month drain event in a 6-month window moves the *mean* of the pooled representation by ~1/6 —
but the *extremes* carry the event. Our head never sees them. **Change: pool = mean ⊕ max over
observed months** (or mean ⊕ min ⊕ max). This is a *representation reframe* letting the head see
exactly the permanence-vs-event structure the literature says defines the class.
- Capacity note: concatenating max doubles the head's input width (64→128 first-layer fan-in) —
  a *small* capacity add, which our law penalizes. The cleaner capacity-neutral variant: **split
  pooling** — mean-pool over d/2 dims, max-pool over the other d/2. Identical parameter count,
  single isolated change.
- This is my strongest architecture-lane candidate: physically grounded (Q4 reading), structural
  not additive, and plausibly ≥ +0.01 if drain events are being averaged away today.

## Q3 — Untried mathematics

### 3.1 Pairwise AUC surrogate in the loss ⭐ strongest objective-lane candidate
AUC is **40% of the metric** and we optimize plain BCE. The ranking literature is explicit that
optimal classification gives **no stability guarantee for ranking quality** ("pointwise transfer
failure"; see [Pairwise AUC Surrogate Loss overview](https://www.emergentmind.com/topics/pairwise-auc-surrogate-loss),
[robust deep AUC maximization](https://arxiv.org/pdf/2012.03173)). Concrete, capacity-neutral:
`L = BCE + λ_v·Var_k(logit) + μ · mean_{(i∈pos, j∈neg)} max(0, m − (z_i − z_j))²`
(squared-hinge pairwise margin on within-batch positive/negative logit pairs; m≈1, μ≈0.5 to start).
Zero new parameters — the same family as our only recent win. Also plausibly *complementary* to the
cross-view penalty: xview fixed the probability axis; this addresses the ranking axis directly.
**Fund now** (single isolated change vs champion).

### 3.2 Group-DRO over the masking recipe — the "what are the groups?" answer
[Sagawa et al., ICLR 2020](https://arxiv.org/abs/1911.08731). We have no region labels, but we
*manufacture* groups every run: window length L∈{4,5,6} (and optionally S2-dropout strata). Replace
mean loss with worst-group loss over L. Capacity-neutral, needs strong regularization per the paper
(we have dropout 0.2 + weight decay). Rationale: if the private set skews toward a window length we
handle badly, worst-group training buys insurance invisible on public. Effect size on *public*
likely small → **fund if Q1 succeeds** (or as a designated-finalist alternative candidate).

### 3.3 Negative results to record (so nobody re-proposes them)
- **LogitNorm** ([Wei et al., ICML 2022](https://arxiv.org/abs/2205.09310)): the natural successor
  anti-overconfidence trick, but with our **single-logit binary head it degenerates** — constraining
  |z| to a constant maps every logit to ±τ, destroying the ranking that is 40% of the metric. Only
  viable with a 2-logit softmax head, and the cross-view penalty already owns this axis. **Reject.**
- **Spectral/frequency representation of a 4–6-step series:** 4 samples support 2 usable frequency
  bins; nothing there. **Reject.**
- **OT/Sinkhorn alignment of train→test representations:** unlabeled alignment machinery on 1030
  test rows at adversarial AUC 0.99 is the ESS-collapse family wearing a different coat, and it adds
  an alignment objective + coupling matrix (capacity/machinery). **Reject** on our evidence.

## Q4 — CV design
- The 5-fold-mean test prediction is an *implicit ensemble*; a single model trained on all data is
  the capacity-neutral alternative. But the difference is a variance-family effect — within noise on
  public. **Park unless Q1 gives us a measurable validator.**
- The K=2/R=2 asymmetry: R only affects *measurement* (OOF), not the fitted model — and we've
  established OOF doesn't drive decisions anyway. Not worth a submission. **Park.**

---

## Consolidated ranking

| # | Idea | Lane | Cost | Verdict |
|---|---|---|---|---|
| 1 | **Designate champion + NoPE finalists manually** (rule finding) | endgame | 0 subs | **do before close — confirmed usable** |
| 2 | **Validator retro-fit protocol: ATC + seed-disagreement scored against our 10 known LB outcomes** | Q1 | ~0 subs, ~1–2 hrs compute | **fund now = iter11** |
| 3 | **Pairwise AUC surrogate term in the loss** | Q3 | 1 sub | **fund now = iter12 candidate** |
| 4 | **Split pooling (mean ⊕ max), capacity-neutral** — physically grounded in pond drain events | Q2/Q4 | 1 sub | **fund now = iter12 candidate** (order vs #3 by validator if #2 succeeds) |
| 5 | VH−VV channel replacement | Q2 | 1 sub | fund if Q1 succeeds |
| 6 | Group-DRO over window-length groups | Q3 | 1 sub | fund if Q1 succeeds / finalist-diversity candidate |
| 7 | Rubric prep: reproduction README + workflow writeup | rule | 0 subs | queue for endgame — 35% of final score |
| 8 | IWCV-stabilized OOF | Q1 | 0 subs | third opinion only |
| 9 | Single-model vs fold-mean; R asymmetry | Q4 | — | park |
| 10 | LogitNorm · spectral · OT alignment | Q3 | — | rejected on our evidence |

**Budget reality check:** ≈80 submissions remain (100-cap), 26 days. The plan above spends ~2–6.

## Sources
- [Zindi — GeoAI Aquaculture Pond Identification Challenge](https://zindi.africa/competitions/geoai-aquaculture-pond-identification-challenge) (rules: 2 designated finalists, 100-sub cap, 65/35 phase split)
- [Garg et al. 2022 — Leveraging Unlabeled Data to Predict OOD Performance (ATC)](https://arxiv.org/abs/2201.04234)
- [Baek et al., NeurIPS 2022 — Agreement-on-the-Line](https://proceedings.neurips.cc/paper_files/paper/2022/file/7a8d388b7a17df480856dff1cc079b08-Paper-Conference.pdf) · [Jiang et al. 2022 — Assessing Generalization via Disagreement (note)](https://arxiv.org/pdf/2202.01851)
- [Sugiyama et al., JMLR 2007 — IWCV under covariate shift](https://jmlr.org/papers/v8/sugiyama07a.html)
- [Ottinger et al. 2017 — Coastal Aquaculture Ponds with Sentinel-1 Time Series](https://www.mdpi.com/2072-4292/9/5/440)
- [Cross-polarization VV/VH fusion for flood mapping, IGARSS 2026](https://arxiv.org/pdf/2605.02153) · [SentiWiki S1 applications](https://sentiwiki.copernicus.eu/web/s1-applications)
- [Sagawa et al., ICLR 2020 — Group DRO](https://arxiv.org/abs/1911.08731)
- [Yuan et al. — Robust Deep AUC Maximization](https://arxiv.org/pdf/2012.03173) · [Pairwise AUC surrogate losses](https://www.emergentmind.com/topics/pairwise-auc-surrogate-loss)
- [Wei et al., ICML 2022 — LogitNorm](https://arxiv.org/abs/2205.09310)
