# RESPONSE_09 — triage of the Round-09 deep research (Claude Research + Gemini Deep Research)

**Date:** 2026-07-23 · Champion **0.8955** single-seed / **≈0.8865 reliable** · Inputs:
`findings/claude_research_09.md` (Claude Research, given our brief) and
`findings/Aquaculture Pond Identification ML Strategies.md` (Gemini Deep Research). This is the
coding agent's vote against our LB evidence. It also stages the ground for the upcoming
**cross-examination** (Claude given Gemini's doc). Loop rule: implement the best idea, reject
re-treads, never big-bang.

---

## 0. The headline: the two reports are NOT of equal quality this round

**Claude's report was conditioned on our findings; Gemini's was not.** Claude explicitly challenges
findings A/B/E with math and citations, accepts C/D, and every recommendation is filtered through the
rank-only metric, the seed-noise floor, and the two-phase scoring. Gemini's report is a strong,
well-sourced *generic* SITS-domain-adaptation survey that re-derives four things we have already
refuted or that are inert here — most tellingly it **opens by calling our shift "label shift"**
(§II), the exact misdiagnosis we corrected rounds ago (ours is **covariate** shift), and it closes by
recommending **Saerens EM, CORAL, TENT and the Zou water-tree** — three of which our own leaderboard
has already killed. So I weight Claude heavily and mine Gemini for the *few* places it independently
converges with Claude, because convergence-from-independence is the strongest signal we get.

---

## 1. The vote — every recommendation, both reports, against our evidence

| # | Idea | Source | My verdict | Why |
|---|---|---|---|---|
| 1 | **Dispersion / lower-tail pooling** (replace masked-mean with mean⊕std⊕min⊕max; Gemini's PMA attention-pool is the same target by another route) | **BOTH** | **FUND — screen now (iter19)** | Three-way convergence: Claude (physics: ponds = low temporal dispersion, mean-pool discards it), Gemini (replace the pool), **and our own iter12 `mean_min` probe — the ONLY candidate ever to clear the floor (+0.0672 ATC-F1, +0.0109 LB)**. Capacity-light, channel-replacing (not the toxic amplitude-add), screenable. |
| 2 | **Low-capacity decorrelated ensemble member** — MiniRocket/Hydra (Claude) / CropNet 1D-CNN (Gemini) | **BOTH** | **FUND — build next** | Both independently prescribe a random-kernel / conv model as a *diversity* member. This is the missing ingredient iter18 exposed (below). Model-class departure; screenable by cross-model rank-corr. |
| 3 | **In-domain transductive SSL** (masked-value pretraining on train+test unlabeled, then fine-tune the champion head) | Claude | **FUND — but gate first** | The one model-class-scale, legal, *unexplored* attack on the shift, and the best Phase-Two novelty story. Has a **free** go/no-go: adversarial-AUC on the SSL embeddings must drop below Presto's 0.965. Higher build cost → after #1/#2. |
| 4 | **Variance-collapse rank-averaged ensemble as the PRIVATE-rank play** (order statistics) | Claude | **ADOPT as endgame policy** | Reframes what we already built (`seed_average`, `arch_blend`): its value is not public level (≈0) but private rank + the mandatory reproducibility rubric. Directly resolves the iter18 tension (§2). |
| 5 | **Transductive Var-consistency on unlabeled test rows** (extend cross-view penalty to test) | Claude | **SCREEN** | Cheap extension of the mechanism that already won; attacks the masking component of the shift transductively. DIS-screenable. |
| 6 | **Capped, class-balanced, fold-safe pseudo-labeling (CBST, 1 round)** | Claude (+ Gemini's TimeMatch is the same family) | **SCREEN — later, higher risk** | Reopened by rank-only metric + our screen. Real confirmation-bias risk (confident rows may be the shifted ones). Cap 1 iteration; abandon if worst-fold OOF drops. |
| 7 | **Group-DRO over the masking recipe** | Claude | **PARK** | Robustness knob, likely <0.02; our K-views already sample the test masking. Feeds variance not level. |
| 8 | **Prevalence-pin re-derivation on seed-averaged OOF** | Claude | **DO (free, cross-cutting)** | Our pin was set on a single-seed sweep; re-derive per final model. Zero cost. |
| — | **Set Transformer (ISAB/PMA)** as the primary encoder swap | Gemini | **PARK** | The pooling half (PMA) converges with #1 and is worth stealing; the *encoder* swap is the NoPE/set-encoder family, which already **TIED** (+0.0009) and adds capacity. Take the pooling idea, not the whole encoder. |
| — | **Deep CORAL** feature-covariance alignment | Gemini | **REJECT** | Claude's rebuttal is decisive and matches finding (C): aligning global feature moments across train/test destroys the **absolute backscatter level that IS the pond signal** (rank-replace collapsed OOF 0.975→0.86). |
| — | **TENT test-time entropy minimization** | Gemini | **REJECT** | Two independent kills: entropy-min sharpens confident predictions ≈monotonically → **rank-invisible** under our rank-only metric; and it needs BatchNorm stats we don't have (we use LayerNorm). |
| — | **Saerens EM / MLLS prior calibration** | Gemini | **REJECT (re-tread ×4)** | Assumes **label** shift; ours is **covariate**. BBSE already estimated prior 0.44 vs the LB-true 0.649. And a monotone logit shift is **invisible** to the rank-only LB except through the prevalence pin we already apply. Refuted three prior rounds. |
| — | **Zou et al. Multi-Index Water Tree / WIF / AWEI / EVI** | Gemini | **REJECT (re-tread)** | Water indices cost us **−0.075** in the GBDT lane. The *one* salvageable fragment is the **S1 VV-10th-percentile** ("persistent low backscatter"), which converges with #1's physics — but as a low-dim invariant it belongs in the pooling/representation, not as an appended channel. |

---

## 2. Alignment vs divergence (for the cross-examination)

**Where they CONVERGE (compounding confidence):**
- **The masked-mean-pool is the lossy bottleneck.** Claude → moment pooling (dispersion); Gemini →
  attention pooling (PMA). Independent routes to the same target. *This is the strongest signal in
  either document, and our own `mean_min` probe already agrees.*
- **Add a low-capacity, decorrelated member from a different model class** (MiniRocket / CropNet).
- **Persistent low SAR backscatter + temporal permanence is the pond physics** (both cite Ottinger).
  Both agree amplitude/level is signal — do not normalize it away (our finding C).
- **Transductive use of the unlabeled test set is the legitimate attack on the shift** (Claude: SSL +
  Var-consistency + CBST; Gemini: CORAL/TENT/TimeMatch). They agree on the *target*, disagree on the
  *method*.

**Where they DIVERGE (the cross-examination should resolve):**
- **CORAL / TENT / Saerens EM:** Gemini funds; Claude rejects; **our LB evidence sides with Claude.**
  These are the questions to put to Claude-vs-Gemini directly.
- **Set Transformer as encoder:** Gemini's flagship; our ledger already tied it (NoPE).
- **Ensemble level gain:** Claude says decorrelated averaging gains OOD level (DiWA/Model-Soups,
  conditional on ρ<0.9); Gemini treats ensembling as a regularizer. iter18 is the live test.

---

## 3. What iter18 just told us, read through the research

**iter18 cross-architecture rank-corr = 0.9395 (min 0.9097), MARGINAL.** The four tied transformer
variants are only slightly less correlated than seeds (0.9511). Claude's math explains it exactly:
the variance-reduction factor (1+ρ(M−1))/M is ≈0.96 at ρ=0.94 — essentially the member mean, just
like seed-avg. **His inclusion rule is ρ<0.9 (target 0.6–0.75); our transformer variants at 0.94 do
not qualify.** So the architecture ensemble cannot buy *level* — because every member is the same
model class. **The way to get ensemble level is a member from a DIFFERENT class** (idea #2), which is
precisely what both reports prescribe. iter18 is therefore not a dead end; it is the measurement that
tells us to stop pooling transformer-variants and go get a decorrelated member.

**The archblend4 screen (ATC-F1 −0.0101 LB) is partly a representation artifact** (the blend bundle
stores rank vectors while the anchors store saturated probabilities — flagged when it was built), so
do not read it as "the blend is worse." Read the **correlation matrix**: marginal.

---

## 4. Decisions (right now)

1. **Upload `submission_champion_archblend4.csv` once.** Not to chase level (≈0), but to (a) bank our
   **lowest-variance artifact** — it pools BOTH seed and architecture noise, so per Claude's
   order-statistics argument it is the strongest *private*-slice finalist we have — and (b) get the
   clean datapoint confirming arch+seed pooling lands within noise of champion. Claude explicitly
   recommends this one confirmation submission.
2. **Stage iter19 = dispersion / lower-tail pooling** (idea #1) — the three-way convergence, already
   built, screenable for 0 further submissions. Re-run `mean_min` and `mean_std` at 2 seeds each so
   DIS is scoreable and the seed-noise guard applies (the guard is what made us hold `mean_min` last
   time; with proper replication we find out whether +0.0672 ATC-F1 survives it).
3. **Queue iter20 = a MiniRocket/CropNet decorrelated member** (idea #2), built to be rank-averaged
   into the ensemble only if its cross-model rank-corr with the transformer is <0.75.
4. **Queue the in-domain SSL** (idea #3) behind its free adversarial-AUC gate.

**Endgame policy adopted from Claude:** the two final submissions should be the two *lowest-variance*
decorrelated artifacts (not our highest public score), and we must reserve time for the Phase-Two
reproducibility + novelty writeup (35% of the final score, top-5 only). In-domain SSL is both a level
bet and the novelty story, so it earns priority once the cheap screens are done.
