# Deep-Research Brief — Round #16 (Claude Research / Gemini Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)
**Date:** 2026-08-06 · **Best public LB: 0.906492 (legal, single model)** · **Deadline:** 2026-08-16 (10 days) ·
**Budget:** ~68 of 100 submissions · **Finalists to designate: 2**

**This round we ran 8 internal research agents and they converged hard. The feature-*adding* lane is
closed — and we now understand exactly why. This brief hands you the mechanism, the three capacity-neutral
levers that replace it, one legal operating-point lever worth ~+0.019 that we nearly discarded, and the
single open question that gates all of them.** We do not need another survey. We need Q1–Q5 pressure-tested.

---

## 0. What changed since Round 15 — read this first

**Round 15 assumed we would keep adding nonlinear channels (VV permanence, pond-band, variance, rice-gate).
iter33/34 falsified that whole menu:**

```
iter33  permanence ENSEMBLE (perm-archblend4)   0.892939   permanence does NOT stack (substitutes)
iter34  + pond-band occupancy                    0.900373   -0.0061   } every added channel:
iter34  + VH^2 (temporal variance)               0.894697   -0.0178   }  OOF ROSE, LB FELL
iter34  + VV permanence                          0.891166   -0.0153   }
```

**1. 🔑 The failure is a SHIFT-CARRIER DIVERGENCE TAX, not capacity noise.** Ben-David decomposition of
the target risk when we add a mean-pooled coordinate `m_g = E_t[g(x_t)]`:

```
ΔR_test = ΔR_source(≤0, = OOF, always improves)     ← why OOF rose on every arm
        + ½·Δd_HΔH (≥0 iff m_g's marginal separates train/test  = covariate/shift-carrier)
        + Δλ*     (≥0 iff p(y|m_g) drifts             = conditional shift)
        + Δε_est  (≥0, finite-sample, small here)
```

OOF measures **only** the first term (always favorable) → it is *structurally* anti-correlated with the
LB. The tell that this is divergence, not variance: **all three iter34 losers dragged the test positive-
rate systematically toward the train prior 0.40** (away from the believed test prevalence ~0.65). Random
capacity overfitting would not produce a systematic sign. The champion feature `1[VH<−21]` wins precisely
because it is the rare coordinate with `ΔR_source<0` **and** `Δd_HΔH≈0` (a CDF fraction is n-invariant and
gain-invariant) **and** `Δλ*≈0` (a physically anchored water line). (Refs: Ben-David 2010 *Machine
Learning* 79; "Accuracy on the wrong line" arXiv:2406.19049 — noise/nuisance features flip ID↔OOD
correlation negative; Sagawa arXiv:2005.04345; Nagarajan arXiv:2010.15775 — even a linear head tilts
toward a shifting coordinate.)

**2. There is a submission-free, out-of-family feature screen.** Add a coordinate **iff**
`(2·LABEL-AUC − 1) > κ·(2·ADV-AUC − 1)`, κ≥1, where LABEL-AUC = AUC(m_g; y) and ADV-AUC = AUC(m_g; 𝟙[test])
on unlabelled test. Repo-calibrated cuts (from our iter25 band screen): require **ADV-AUC ≤ 0.56 AND
LABEL-AUC ≥ 0.75**. This is legal (unlabelled test only), and unlike our retired ATC-F1 it is defined for
representation changes (ATC-F1 mis-signed the one out-of-family case, iter26).

**3. 🟢 The lever we almost threw away: legal prior-shift correction, worth ~+0.019 LB.** F1 is scored at
a hard 0.5 cut; our models realize pos-rate ~0.55 while test prevalence is believed ~0.65 — we are
under-predicting positives, which costs F1. The Saerens–Latinne–Decaestecker / MLLS EM correction
recalibrates probabilities to an **estimated (never LB-tuned)** test prior, then keeps the literal 0.5 cut
— the *same legal category* as our existing Platt-on-OOF step. One internal agent argued it is unsafe
(adv-AUC 0.89 ⇒ conditional shift). **A second agent rebutted this rigorously and we believe the rebuttal:
adv-AUC being high is fully consistent with pure label shift** (moving π shifts the marginal `p(x)` by
construction when classes are separable), so adv-AUC is the wrong instrument. The correct go/no-go is an
**offline mixture goodness-of-fit test** (§Q1). This disagreement is the single most important thing for
you to adjudicate.

**4. Feature engineering is over; the remaining levers are capacity-neutral or operating-point.** Adding
channels overfits (25-channel sweet spot). So the endgame is: refine the *shape* of the one working
channel, *replace* dead weight rather than add, average in *weight space*, correct the *operating point*
legally, and choose *finalists* to minimize private-LB regret.

---

## 1. Self-contained problem statement

Binary per-pixel classification: is this ~10 m cell a managed aquaculture pond? Input = 12-month × 12-band
time series (Sentinel-1 VH, VV in dB, always co-present; 10 Sentinel-2 optical bands, cloud-gapped). No
lat/lon, no patch. Train 1817 (~40% pos), test 1030 (public ~309 / private ~721), believed test prevalence
~0.65 (label shift). Metric = 0.6·F1 + 0.4·ROC-AUC; F1 at a **hard 0.5 cut** (threshold tuning FORBIDDEN).
Model: from-scratch Transformer = per-timestep **Linear** → masked **mean-pool** over observed months →
small head; **LayerNorm** (no BatchNorm). Trained only on competition data; legal calibration = Platt on
train OOF + literal 0.5. Masking trap: train 12 months, **test 4–6 months** → only n-invariant statistics
(mean/median/interior-quantiles/fractions/U-statistics/L-moments) are safe. Domain shift is strong:
adversarial-AUC ≈ 0.89. Seed variance 0.019 (measured); LB resolution ~0.013; **OOF is anti-correlated
with LB.**

## 2. The affine blind spot (the theoretical spine, unchanged)

Because Linear and mean-pool commute, the model sees only **affine functions of the temporal mean** of its
inputs. A new channel `g(x_t)` adds information iff `E_t[g]` is nonlinear-in-the-mean. Empirical-CDF values
`E_t 1[x<τ]`, squares `E_t[x²]`→variance, cross-products→covariance, AND-gates→joint occupancy are the only
escaping families. Raw ratios/differences (VH−VV, SDWI, total power, RVI) are affine-dead — confirmed
(VH−VV = −0.023). The winning channel is `1[VH<−21]`; feature-selection is monotone 1τ>4τ>6τ.

## 3. What is CONFIRMED — do not re-derive

Permanence `1[VH<−21]` = 0.906492 (best, seed-robust: 0.9065@42, 0.9007@29). Trees/cross-class ensembling
closed (0-for-3, OOF illusion 0.995→0.70). Permanence does NOT ensemble (substitutes with the pooling lift,
perm-archblend4 = 0.8929). Adding a 2nd channel overfits (25-ch sweet spot). Hard-τ scan is a dead lever
(τ=−21 on the physical optimum; common-mode, sub-resolution). Seed-averaging buys variance but ~0 level
(seeds 95.1% correlated; +0.0006). archblend4 = 0.899643 (calibration-diversity pool).

---

## 4. The three capacity-neutral levers (ranked) — pressure-test these

| # | Lever | Mechanism | Status |
|---|---|---|---|
| I | **Soft permanence** `σ(s·(τ−VH))`, fixed slope | at n=4–6 the hard fraction is quantized to ≤6 levels (rank ties + train/test level-set mismatch); soft ramp uses each month's *distance* below τ = the optimal rank-1 log-likelihood-ratio coordinate | **built, staged iter35** |
| II | **Channel replacement** | VH & VV missing-indicators are the identical vector `1[obs]` (R=1, info-free) → drop one, add VH²/rice-gate at **constant width 25** = the untested 2×2 cell (add hurt, delete hurt) | **built, staged iter35** |
| III | **SWA / SWAD weight-averaging** | average **weights** along one run → flatter minimum → a *level* lever NOT capped by the (1−ρ̄)/N ceiling that flattened seed-avg. LayerNorm ⇒ no BN-stat gotcha. Must be within one run (from-scratch nets aren't mode-connected). | needs build (iter36) |

## 5. The five questions we need pressure-tested (ranked)

**Q1 — THE decision: is the shift pure-label or conditional, and does the Saerens gate pass?** This one
answer gates three things at once: (a) whether the +0.019 Saerens prior-shift lever is safe; (b) whether
the single-feature permanence bet is fragile on the private rows (if the `1[VH<−21]`↔pond relation itself
drifts) so that archblend4 should be the *primary* finalist; (c) which features transfer. Specify the exact
**mixture goodness-of-fit test**: from train OOF form per-class score densities ĝ₁, ĝ₀; fit π on test scores
by MLLS/EM; test whether `π·ĝ₁ + (1−π)·ĝ₀` reproduces the *actual* test score histogram (KS/χ²/L1). Good fit
⇒ label-shift plausible ⇒ ship. Give the estimator details, the decision statistic + threshold, and a
physics-anchored π_t estimate from the test VH-permanence marginal `F̂(−21)` (2-component mixture). Is a
correction 0.55→~0.60 expected to move LB by >0.013 at AUC~0.90? (Saerens 2002 Neural Comp 14; Lipton BBSE
arXiv:1802.03916; Alexandari MLLS arXiv:1901.06852.)

**Q2 — Soft permanence: is a fixed-slope sigmoid provably ≥ the hard indicator here?** Confirm the LLR
argument and the n=4–6 quantization argument. What is the optimal fixed slope s given ~2 dB class
separation, and is the optimum interior (not s→∞ = hard)? Any reason it could *hurt* under the label shift?

**Q3 — Channel replacement vs the capacity-sweet-spot: which payload, and does the offline screen predict
the sign?** Between VH² (→ temporal variance, rice-killer) and the AND-gate `1[VH<−21]·1[NDVI<0.25]`
(SAR×optical, pond = dark AND not-green), which is the better replacement for the duplicate indicator? Does
the ADV-AUC/LABEL-AUC screen (§0.2) correctly rank them, and would it have avoided the iter34 additions'
losses? Note the Zindi Farm Pin winner used **median** (n-invariant order statistic) pooling — is a
median-VH replacement channel competitive?

**Q4 — SWA/SWAD on a shallow Linear+mean-pool head: real gain or near-convex no-op?** SWAD's evidence is on
deep ResNets with rich basin structure; our head may be near-convex where weight-averaging ≈ the single
minimum. Predict whether flat-minima averaging buys *level* (not just variance) here, the right schedule
(cyclic-LR tail / dense sampling / EMA), and whether it stacks on permanence or substitutes like the
ensemble did. (Izmailov SWA arXiv:1803.05407; Cha SWAD arXiv:2102.08604; caution: cross-seed weight-avg is
invalid — not mode-connected.)

**Q5 — Finalist pair to minimize E[max(private_A, private_B)] regret.** Confirm the portfolio framing (final
rank = the better of the two on the private 721 rows, so we are paid for mean AND (1−ρ)). Given
`champion_perm_seedavg5_st` (real single-τ seed-avg, expect ≥0.90, seed-luck removed), `c_perm_single`
(0.9065, lucky seed), and `champion_archblend4` (0.8996, decorrelated competent, no single-feature
dependence): is {seed-avg permanence + archblend4} the regret-minimizer? Reconcile the tension — one agent
argued for two permanence variants; another showed that permanence *substituting* for the ensemble lift is
exactly the decorrelation E[max] rewards, favoring archblend4. Does the answer flip if Q1 finds conditional
shift (making the permanence bet fragile)?

---

## 6. Known dead-ends — do not propose

Adding any channel (overfits the shift); raw ratio/difference/affine features; dense multi-τ or free
learnable τ; hard-τ scanning; per-window/Otsu adaptive τ (ill-posed at n=5); trees / cross-class ensembling;
frozen foundation-model embeddings (Presto, adv-AUC 0.97); pooling permanence into the ensemble; rank-average
pooling (kills the F1 calibration lift); entropy-minimization TTA and self-training pseudo-labeling (collapse
at adv-AUC 0.9); OOF/CV or ATC-F1 as a cross-class ranker; any LB threshold/prevalence tuning (illegal);
more seeds past ~5 for level; cross-seed weight-averaging (nets not mode-connected).

## 7. Reading list (verify / extend / challenge)

**Shift theory:** Ben-David 2010 (ML 79); arXiv:2406.19049; Sagawa 2005.04345; Nagarajan 2010.15775; Shah
2006.07710; DomainBed 2007.01434; IRM 1907.02893 + risks 2010.05761; Xu 2025 invariant crop 2509.03497.
**Capacity-neutral:** low-rank bottleneck Bhojanapalli 2002.07028; dropout=adaptive-L2 Wager 1307.1493;
decoupled WD 1711.05101; Khani-Liang 2012.04104 (replace ≠ delete); LUPI/distillation Lopez-Paz 1511.03643,
Born-Again 1805.04770. **Weight-averaging:** SWA 1803.05407; SWAD 2102.08604; SWAG 1902.02476; soups
2203.05482; snapshot 1704.00109; EMA 1703.01780; Krogh-Vedelsby NIPS1995; Ovadia 1906.02530.
**Calibration/prior-shift:** Saerens 2002; Lipton BBSE 1802.03916; Alexandari MLLS 1901.06852; Elkan 2001.
**Finalist/shakeup:** Blum-Hardt Ladder 1502.04585; Dwork reusable holdout (Science 2015); Efron-Morris /
James-Stein; Markowitz 1952; Zindi Farm Pin winners writeup. **SAR physics:** Xing 2018 Dongting (PeerJ
e4992, τ=−21.56); Ottinger 2017 rs9050440 / 2021; Tsyganskaya 2018 rs10081286; Mekong rice rs13050921.

## 8. Deliverable requested

A prioritized answer to **Q1–Q5**, each with mechanism (in affine-blind-spot / Ben-David terms), the SAR or
statistical physics, ≥1 verified paper, the concrete recipe (τ/slope/π_t values, exact test statistics), and
a one-line expected LB effect (only >0.013 is resolvable). Adjudicate the Saerens safety disagreement (Q1)
explicitly. End with a single ranked build order for the remaining 10 days and the 2 finalists to designate.
