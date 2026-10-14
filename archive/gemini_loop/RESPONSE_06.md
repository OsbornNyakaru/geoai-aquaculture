# RESPONSE_06 — triage of round-06 research
**Date:** 2026-07-21 · Champion **0.8955** · Inputs: `RESEARCH_06_CLAUDE.md` (in-loop agent, live web)
and the **Claude Fable Deep Research** report. Both answered the same brief (`UPDATE_06.md`)
independently. **Where they converge, confidence compounds; where they disagree, Fable wins on
sourcing twice — both corrections accepted below.**

---

## 1. Independent convergence (highest confidence)

| Both reports independently concluded | Consequence |
|---|---|
| **Q1 is the unlock and is tractable**: adversarial-weighted OOF + label-free OOD estimators (ATC, Agreement-on-the-Line), **validated by retro-fit to our 10 known (change → LB) pairs before being trusted** | **iter11, zero submissions.** Both flagged the retro-fit as the guardrail *unprompted* — this is the plan. |
| Weighting for **evaluation** ≠ the rejected weighting for **training**; ESS collapse is survivable when you only need an *ordering* | Re-opens a lane we closed too broadly. Fable adds the fix: **power-flattening w^β + Hájek self-normalization + ESS diagnostic**. |
| **Pond signature = permanence/level** (Ottinger 2017 median-of-time-series) — reconfirms finding (C): amplitude is signal | Amplitude normalization stays permanently rejected, now on physical grounds, not just our −0.0514. |
| **OT / Sinkhorn alignment must be rejected** | Fable's phrasing is the sharpest: the separator axis *is* the label axis, so alignment necessarily transports away the signal — a predicted −0.05-class repeat of detrend. |
| **Spectral/frequency on a 4–6-step masked series: reject** | Below any usable resolution. Closed. |
| **Rule facts** (100-sub cap, 2 designated finalists, 65/35 two-phase) | Independently verified by both. See §4. |
| Group-DRO over **masking-recipe groups** is legitimate but sub-0.01 | Gated behind Q1. |

## 2. Where Fable corrected me — both accepted

### 2.1 Pooling: **dispersion (std), not events (max)** ✅ correction accepted
I proposed `mean ⊕ max` pooling, reasoning that a 1-month **drain event** is diluted ~1/6 by
mean-pooling. Fable refutes the premise with the primary literature: the drain/harvest transient is
exactly what the standard method **discards** — Ottinger et al. (2017; 2022) and Prasad et al.
(2019) use the pixel-wise **median** *specifically to suppress it* ("to reduce confusions with
temporary inundated rice fields"). The discriminator is **permanence + LOW temporal dispersion**
(stable low scatterer), not the event.

So `max` chases a signature the domain deliberately treats as an outlier, while `std` captures the
actual one. **Fable is right and my mechanism was wrong.** It also correctly notes this *answers*
the Q4b question I posed: mean-pooling's bottleneck is **loss of DISPERSION, not loss of ORDER** —
which independently explains our own **NoPE tie** (order is neutral because the physics is
order-free). Two independent lines of evidence now agree.

**Synthesis I am carrying forward (better than either report alone):** Fable's `mean ⊕ std` widens
the head's input 64→128, i.e. a small capacity *add* — the one family that has never won for us.
So: **lead with `mean ⊕ std` concat** (preserves the full mean, matches the sourced design), and if
it lands within noise, retry as **split pooling** (mean over d/2 dims ⊕ std over d/2 dims), which is
*exactly* parameter-neutral. Note for implementation: with N=4-month windows use the biased 1/N std
or clamp, per Fable.

### 2.2 Objective: **focal loss over my AUC-surrogate** ✅ correction accepted
I ranked a pairwise AUC surrogate as the top objective candidate (AUC = 40% of the metric, we
optimize BCE). Fable demotes it with an argument I cannot answer: **our in-domain AUC is already
≈0.99**, so the surrogate mostly re-optimizes an already-saturated ranking and does nothing for the
**60%** F1 term. That is a better reading of our own numbers than mine. Demoted to "fund if Q1
succeeds."

Its replacement — **focal loss** — is well-argued as a *non-redundant* successor to our one
objective-lane win: it reduces overconfidence by **entropy regularization** (Mukhoti et al.,
NeurIPS 2020), a different mechanism from our cross-view **variance** penalty. Specific and
actionable: **γ=3 or FLSD-53, explicitly not the default γ=2**, and re-fit the prevalence shift δ
afterwards.

**My one reservation, recorded:** iter10 showed de-saturation is near-exhausted at λ=1. Focal is a
*different* de-saturation route, so it is not ruled out — but it targets the same weakness, so I
hold a **moderate, not high, prior**. Fable flags this risk itself. Screen it through Q1 if Q1 works.

## 3. What I keep from my own report that Fable did not cover
- **LogitNorm explicitly rejected**: with our **single-logit binary head** it degenerates to ±τ and
  destroys the ranking that is 40% of the metric. Fable didn't raise it; recording so nobody does.
- **Fold-ensemble deletion (R5)** — Fable's idea, and a good one (fold-averaging is a hidden
  ensemble; ensembling has always hurt us). Caveat both of us note: it **destroys OOF**, so it is
  unshippable until Q1 exists. Correctly gated.

## 4. Rule facts (both reports, verified against the Zindi page)
- **Finalists: we designate 2** (default = 2 best public) → **the NoPE hedge is usable.** Designate
  **champion + NoPE manually** before close; do not let the default hand our slot to a
  noise-riding sweep file.
- **100 total submissions**, ≈20 used → **≈80 left** (our "130" was wrong).
- **65% LB + 35% top-5 rubric** (reproducibility/innovation). Fable's strategic note is worth
  quoting: the **physically-motivated R3/R8 and the principled Q1 protocol "read as genuine
  methodological contributions in Phase Two."** Our LB-gated loop with a full ledger is already a
  rubric asset. Queue a reproduction README + workflow writeup near the deadline.

## 5. Verdict table

| # | Idea | Verdict | Cost |
|---|---|---|---|
| 1 | **Offline validator + retro-fit to the 10 knowns** (ATC · seed-disagreement/AoL · power-flattened weighted OOF) | ✅ **FUND NOW = iter11** | 0 subs |
| 2 | **Dispersion pooling** `mean ⊕ std` (split-pool fallback) | ✅ **FUND NOW = iter12** | 1 sub |
| 3 | **Focal loss** γ=3 / FLSD-53, keep λ=1, refit δ | ✅ **FUND NOW = iter13** | 1 sub |
| 4 | Designate champion + NoPE as the 2 finalists | ✅ **endgame, mandatory** | 0 subs |
| 5 | Reproduction README / workflow writeup (35% of final score) | ✅ **endgame** | 0 subs |
| 6 | Fold-ensemble deletion (train-on-all) | ⏸ gated on Q1 (kills OOF) | 1 sub |
| 7 | Group-DRO over window-length groups | ⏸ gated on Q1 | 1 sub |
| 8 | VH−VV cross-pol **replacement** channel | ⏸ gated on Q1 | 1 sub |
| 9 | Pairwise AUC surrogate | ⏸ demoted — in-domain AUC already ≈0.99 | 1 sub |
| 10 | CVaR/χ²-DRO · conformal/selective prediction | ⏸ park | — |
| 11 | **OT/Sinkhorn alignment · spectral · attention/order-sensitive pooling · LogitNorm · amplitude norm · Saerens/BBSE · self-training · temperature scaling · water indices · further λ or positional reframes** | ❌ **rejected** | — |

## 6. The gate that decides everything

**iter11 minimum bar (pre-committed, both reports agree):** a validator configuration must place
**detrend (−0.0514) and K=4 (−0.0115) BELOW relative-time (+0.0128) and cross-view (+0.0047)**, with
Spearman ρ clearly > 0 across the anchors.

- **ρ > 0.7 →** the noise floor is broken. Screen the whole gated backlog (#6–#9) offline and spend
  the ≈80 remaining submissions only on candidates where **≥2 of the 3 rankers** beat the champion.
- **Bar not cleared →** Q1 failed, we have spent **zero** submissions learning it, and the rule
  reverts to: only fund ideas with a plausible effect **≥ +0.013**. Under that rule #2 and #3 still
  ship, on their own merits, and everything else stops.
