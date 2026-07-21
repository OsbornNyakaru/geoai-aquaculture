# RESPONSE_05 — round-05 Deep Research replies (Gemini + Claude) + Claude-agent triage

*Two independent Deep Research reports received 2026-07-21 in reply to `UPDATE_05.md`. This is the
coding agent's triage against our LB evidence. Loop rule: implement the best idea, reject re-treads,
never big-bang. Full original reports are in the session log; this is the actionable distillation.*

## Headline: both reports independently rank the SAME idea #1

**Duration-normalized fractional positional encoding** — index position by `p = t_rel/(L-1) ∈ [0,1]`
so 4/5/6-month windows share ONE relative frame — is Gemini's Section 2A AND Claude's #1 (highest EV).
Two independent research passes converging on the same capacity-neutral, positional-family idea, on the
exact axis that just won (+0.0128), is the strongest prior we can get. **This is iter7.**

It is the literal next deletion on the proven axis: relative-time removed absolute *start*;
duration-normalization removes absolute *length*. Claude grounds it in the length-generalization
literature (Kazemnejad NoPE, NeurIPS 2023; Ruoss randomized-PE, ACL 2023; Foumani tAPE, 2024) and in
thermal-time re-indexing (Nyborg CVPR-W 2022; T³S 2026) as the legal in-series proxy for the
gold-standard temporal-shift fix. Stays entirely off the toxic amplitude axis.

## Full triage (both reports)

| Idea (source) | Verdict | Rationale vs our LB evidence |
|---|---|---|
| **Duration-normalized fractional positions** (Gemini §2A, **Claude #1**) | ✅ **TEST FIRST = iter7** | Capacity-neutral/removing, on the proven positional axis, both reports #1. Removes residual window-LENGTH memorization. Impl: interpolate the existing learned length-12 table at fractional indices (parameter-NEUTRAL; reduces to champion exactly when L=12). |
| **NoPE / permutation-invariant set encoder** (Claude #2) | ✅ **BANK** (iter8 + diverse finalist) | Drop positional embedding entirely → set encoder over observed months. Two-tailed/high-variance (order carries pond-vs-rice fill/drain signal) but the *ideal diverse second finalist*. Parameter-removing. |
| **Cross-view invariance objective** across K=2 views (Claude #4) | ✅ **BANK** (iter9) | `L = BCE + λ·‖logit(v1)−logit(v2)‖²`: teaches label-invariance to WHICH window is observed — an *objective/inductive-bias* change (structurally like our winner), NOT a robustness add-on like TTA. Capacity-neutral, no new params. |
| **Prevalence sweep on the new champion** (Claude RQ3) | ✅ **BANK** (one day, bounded) | 0.649 was tuned for the OLD 0.8780 model; the reframed champion's F1-optimal monotone shift may differ. Free (no retrain, one run → 5 files), isolates the 60%-weighted F1 lever, ROC-AUC untouched. Pick plateau CENTER, not argmax; keep 0.649 fallback. Do ONCE. |
| Randomized-start PE (Claude #3, Ruoss) | ⚪ Later | Generalizes our left-align; partially overlaps what relative-time already captured. Combine WITH iter7 (dnorm+randomized-start) only if dnorm helps. |
| Center/event-anchored frame (Claude #5) | ⚪ Low priority | Cheap symmetry variant; only if iter7–8 stall. Must re-index TIME only, never amplitude. |
| Learned relative-position bias / ALiBi (Claude #6) | ❌ Down-ranked | Learned RPE ADDS capacity (our ledger: added capacity hurts). ALiBi's recency bias ill-suits an order-weak median-like signal; Kazemnejad found it worse than NoPE for OOD. |
| CropNet 1D-CNN from-scratch (Gemini §5) | ⚪ Possible *diverse finalist* only | Rule-legal (from scratch). A different architecture is a valid private hedge, but NoPE is cheaper/on-axis and Claude prefers it. **Do NOT blend it** (GBDT blend already −0.0075). Low priority. |
| **Saerens EM / MLLS prior** (Gemini §1) | ❌ **REJECT — proven dead-end (3rd time)** | Assumes LABEL shift; ours is COVARIATE shift. We built & tested BBSE/Saerens-EM: estimated ~0.44 vs LB-true 0.649. `prevalence_target 0.649` already hits the LB-verified optimum. (Gemini also misattributes the +0.042 jump to this shift — that gain was the GBDT base-rate correction, a different mechanism.) |
| **Zou water-tree / WIF / EVI indices** (Gemini §3) | ❌ **REJECT — proven dead-end + toxic axis** | WIF scored −0.005; EVI already off (`evi:false`); Zou = hardcoded magic-constant thresholds = the non-transferable class killed in round 3. These are ADDED amplitude/feature channels — the −0.0514 detrend axis. Double-rejected. |
| **CAST pseudo-labeling / self-training** (Gemini §4) | ❌ **REJECT** | Self-training under adversarial train/test AUC ≈0.99: test points are OOD by design, so the KDE density gate either rejects ~everything (no effect) or admits OOD noise. Same failure family as importance-weighting/DANN (ESS collapse) we killed. Adds a complex self-training loop = capacity/complexity. Very low prior. |
| Gemini's "Transformer-CropNet blend + CAST" finalist bundle | ❌ **REJECT big-bang** | Bundles a blend (−0.0075), self-training (dead-end), and a new model into one unattributable change — exactly the round-04 refactor trap. We change ONE isolated thing, gated on the LB. |

## Finalist / private-LB strategy (both reports, Claude RQ4)

- **Drop the TTA hedge.** The TTA variant (0.8885) is a near-duplicate of the champion → correlated
  errors → ~zero private protection. Both our own note and Claude agree.
- **Two finalists = champion (0.8908) + one structurally DISTINCT reframe** (duration-norm or NoPE),
  chosen for diversity, not a 0.001 public edge. Diverse inductive biases fail on different rows.
- **Do NOT blend to climb** (variance move, unresolvable on 309 rows; added model class already lost).
- **Action item:** verify on Zindi whether the private LB auto-selects best-public or lets us designate
  two finalists. If auto-select, the "diverse hedge" collapses to "which single submission we leave on
  top" — favoring the champion unless a reframe clearly leads.

## Stop rules (Claude)
- Two consecutive positional reframes both < +0.003 → positional channel exhausted at this resolution;
  shift budget to the objective lever (cross-view invariance) + the one-time prevalence sweep, freeze finalists.
- Any single move > +0.02 → new champion; re-baseline all subsequent tests against it.

## Implemented this round
**Duration-normalized positions only**, behind `seq.pos_encoding: dnorm` (default `learned` = champion,
bit-for-bit). Interpolates the existing learned length-12 positional table at fractional indices derived
from each row's observed length L — parameter-neutral, reduces to the champion when L=12. Requires
`relative_time: true` (champion). Iter7 = `submission_seq_dnorm.csv`; gate vs 0.8908. If it wins → keep +
bank NoPE (iter8, also the diverse finalist). If within noise → keep as diversity candidate, don't iterate;
go to NoPE. If it craters → revert, go to NoPE.
