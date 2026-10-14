# RESPONSE_10 — triage of the Round-10 cross-examination (Claude given Gemini's report)

**Date:** 2026-07-23 · Champion **0.8955** single-seed / **≈0.8865 reliable** · Input:
`findings/cross_examin_doc.md` (Claude, given Gemini's Deep Research doc, adversarial comparison).
Builds on `RESPONSE_09.md`. Loop rule unchanged: implement the best idea, reject re-treads.

---

## 0. Verdict: the cross-examination CONFIRMS our RESPONSE_09 read and upgrades the roadmap

The adversarial pass reached the same verdicts we did, with receipts, and added two things we did not
have: a **new #1 candidate (instance-expansion)** and a set of **honest self-corrections** to Claude's
own R09 that change how we should weight its earlier claims. Net: our rejections stand, our funded
list is re-ordered, and one of my staged choices (iter19 pooling) gets a specific caveat.

**What it confirmed (compounding confidence):**
- **Gemini is the weaker report — 3 of its 5 headline moves are refuted or inert:** Zou water-tree
  (−0.075 measured), duration-normalized positions (−0.0064 measured), and Saerens/MLLS (rank-invisible
  + assumes label shift when ours is covariate). Exactly our RESPONSE_09 calls.
- **CORAL rejected** (aligning 2nd-order stats suppresses the backscatter-LEVEL that IS the class
  signal), **Set Transformer/ISAB/PMA parked** (capacity-add over the already-exhausted NoPE lane).
- **It independently verified the citations** and flagged two suspect Gemini refs: **PE-Field 4D
  (arXiv:2607.15667)** is real but about video diffusion — domain-irrelevant; **arXiv:2604.19217**
  unverifiable; **LEIP (2604.01651)** is label-shift-only, so its guarantees don't hold here. Do not
  reuse these in any writeup.

**Where it CHANGED my picture (act on these):**

1. **NEW #1 — Instance-expansion reframing.** Treat each `(row, observed sub-window)` as an
   *independent* training example (its own BCE term, not coupled), aggregate per-row at inference.
   Rationale that makes it #1: it is the **only** proposal in any of the three docs that is BOTH a
   *data-model change* (the category that has ever cleared our floor — GBDT→Transformer +0.052) AND
   *directly matched to the designed masking trap* (test rows literally ARE partial windows). There is
   an unverified competitor lead of **~0.914 AUC with "each month a training data point."** Expected
   band +0.01 to +0.04. **This becomes the priority after iter19.**
   - *How it differs from our K-views (important):* our champion couples K=2 views by owner-grouping +
     the cross-view Var penalty (λ=1). Instance-expansion *decouples* — large K, independent BCE, no
     grouping. That is in tension with our winning mechanism (coupling), so it is a genuine alternative
     regime, not a tweak. **Cheap first test:** `seq.K` large + `consistency_lambda=0` + a
     no-owner-grouping flag; screen with ATC-F1 before any bigger build.

2. **iter19 caveat (dispersion pooling) — Claude flags my own staged choice.** `[mean‖std‖min‖max]`
   grows the head input 4× (and `mean_min`/`mean_std` grow it 2×) — a **capacity ADD**, which our own
   ledger law predicts loses. The identity-preserving init makes the extra "earned," but it is still
   added capacity. **Reconciliation:** the capacity-NEUTRAL form is `[mean_{d/2} ‖ std_{d/2}]` at the
   *same* total head width d (split the encoder dims), or std-only — amplitude-preserving and
   param-matched. **This does not void iter19:** our own iter12 `mean_min` (+0.0672 ATC-F1) is the only
   offline signal to ever clear the floor, so iter19 is the empirical test of whether the 2× form's
   signal survives replication + the seed guard. If it clears → the capacity concern is overridden for
   this statistic; if it evaporates → iter20 tests the capacity-neutral `[mean_{d/2}‖std_{d/2}]` split.

3. **Claude's own R09 self-corrections (recalibrate our confidence):**
   - *Variance/order-statistics:* σ_seed (0.019) and σ_public (~0.013–0.018) are SEPARATE additive
     components; a competitor re-submitting the SAME model gets the SAME public score, so max-of-n
     inflation only operates across genuinely different submissions. The **directional** claim survives
     (top-5 public gaps ~0.017 ≈ 1σ_public → shakeup plausible → pick low-variance finals), but the
     specific "honest private ~0.915–0.925" was over-precise. **Adopt the policy, drop the number.**
   - *Rank-averaging:* it is provably the **invariant** combiner (to per-member monotone reparam), not
     provably **optimal**; which of rank/prob-avg scores higher is empirical. Keep rank-avg as the safe
     default (our members are differently miscalibrated — t\* ranges 0.28–0.59), but don't oversell it.
   - *In-domain SSL:* a real bet at 2,847 rows (Presto used 21.5M); the free adversarial-AUC gate
     (<0.95) is the kill switch. Unchanged from RESPONSE_09.

---

## 1. archblend4 = 0.8946 — read through the (now-refined) variance framework

| Artifact | LB | Note |
|---|---|---|
| champion xview seed-42 (lucky draw) | 0.8955 | upward fluctuation |
| **champion_archblend4** (4 arch × seeds, rank-avg) | **0.8946** | **−0.0009 vs champion — a tie** |
| member public mean | 0.8925 | blend beat it by **+0.0021** |
| champion_seedavg5 (xview × 5 seeds) | 0.8865 | blend is **+0.0081** above it |

**Interpretation.** All four numbers sit within ~1 σ_public (0.013–0.018) of each other, so nothing is
*significant*. But the direction is uniformly favorable: the architecture blend landed at **champion
level** (not down at the seed-avg 0.8865), beat its member average, and did so while pooling BOTH seed
and architecture noise. This is exactly what iter18's marginal ρ=0.9395 predicted — a small
decorrelation gain, within noise, at minimum variance. **Consequence: `champion_archblend4` is now our
leading finalist** (champion-level public + lowest variance = best private-slice bet per the
order-statistics policy). It replaces `seedavg5` at the top of the finalist list.

*Caveat honestly stated:* 0.8946 on 309 public rows could itself be an up-fluctuation. The claim is
not "the blend is better"; it is "the blend is at least as good and far more reliable," which is what a
finalist needs to be.

---

## 2. Re-ranked roadmap (supersedes RESPONSE_09 §4)

| Rank | Move | Type | Gate before a submission | Status |
|---|---|---|---|---|
| **1** | **Instance-expansion reframing** | data-model change | OOF ATC-F1 gain > 0.013 (its own resolution); adversarial-AUC free | **iter20 — build after iter19** |
| **2** | **Decorrelated low-capacity member** (MiniRocket/Hydra **or** CropNet) rank-blended | model-class member | cross-model rank-corr < 0.9 (target 0.6–0.75) AND solo ATC-F1 within 0.02 | queued |
| **3** | **In-domain transductive SSL** (masked-value pretrain on train+test) | model-class-scale + novelty story | adversarial-AUC on embeddings < 0.95 (free) | queued |
| **4** | **Capacity-neutral dispersion pooling** `[mean_{d/2}‖std_{d/2}]` | representation | ATC-F1; fold into a variant, no dedicated submission | after iter19 verdict |
| **5** | **LN-compatible TTA** (SAR/EATA-style, LN affine only, sample-filtered) | test-time adapt | OFFLINE only; submit only if >0.02 OOF re-ranking | exploratory |
| — | Radar-only decorrelated sub-model (S1-always-present asymmetry) | member | decorrelation screen | opportunistic, feeds #2 |

**iter19 (running now)** is the empirical test that decides between rows 4-as-staged (2×/4× capacity
pooling) and 4-reconciled (capacity-neutral split). Its most likely outcome per the cross-exam is HOLD;
if `mean_min` clears the seed guard, that is a real and surprising positive.

**Both docs converge that CropNet is validated for cross-region SITS shift** (arXiv:2509.03497:
"simple spectral-temporal representations outperform … modern geospatial foundation model embeddings"
under geographic shift) — so if #2's random-kernel route underdelivers, CropNet is the on-physics
fallback member. And both flag the **S1/S2 asymmetry** (2.6% of test rows are optical-blank, never the
reverse) as under-exploited — a radar-only member is a cheap decorrelated candidate.

---

## 3. Endgame policy (unchanged in spirit, sharpened)

- **Final two = the two lowest-variance artifacts whose OOF ATC-F1 is within 0.005 of our best**, NOT
  the two highest public scores. Current leader: `champion_archblend4` (0.8946, min variance).
- **Reserve time for the Phase-Two writeup (35%, top-5 only).** The coherent novelty narrative is
  *in-domain SSL + instance-expansion + rank-averaged decorrelated ensemble* — document the transductive
  use of provided test features explicitly for the reproducibility review.
- **Kill-switches (from the cross-exam's falsification list):** instance-expansion ≤ champion ±0.01 on
  OOF ATC-F1 → the data-model thesis is wrong, pour budget into ensemble+SSL. SSL fails the <0.95 gate
  → drop it. Every decorrelated member lands at ρ>0.9 → the ensemble lane is exhausted like seeds were.
