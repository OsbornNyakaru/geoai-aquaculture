# Deep-Research Brief — Round #15 (Claude Research / Gemini Deep Research)
### GeoAI Aquaculture Pond Identification Challenge (Zindi / FAO / ITU)
**Date:** 2026-08-06 · **Best public LB: 0.906492 (legal, single model)** · **Deadline:** 2026-08-16 (10 days) ·
**Budget:** ~70 of 100 submissions · **Finalists to designate: 2**

**Since Round 13 we found our first real feature win, and it changed the whole strategy.** Adding a
single per-month binary channel `1[VH_dB(t) < −21]` to the Transformer input scored **0.906492 — a new
best, as one legal model**, beating the 4-architecture ensemble (0.899643). This round is about
**squeezing the permanence direction dry, one feature at a time**, and answering the few open questions
that decide whether we can clear ~0.91.

**Focus this round narrowly.** We do not need another survey. We have run 8 internal research agents and
they converged hard (§4). We need Q1–Q6 (§6) answered with papers, math, and a ranked verdict — not a
restatement of what we already know.

---

## 0. What changed since Round 13 — read this first

**1. 🏆 The GBDT/CatBoost lane (Round 13's whole premise) is DEAD. Do not reopen it.**
We built the legal CatBoost lane exactly as Round 13 asked. Result: **OOF AUC 0.995 → public LB 0.70.**
A textbook OOF illusion under our known OOF↔LB anti-correlation. Cross-class blending is now **0-for-3**
(ROCKET, pin-GBDT, legal-CatBoost all dragged the pool). Trees are closed. The ~0.94 leader is on
CatBoost, but we cannot reproduce that from our features and the tree lane does not transfer to our
validation reality. **We win by feature engineering inside the from-scratch Transformer, not by trees.**

**2. 🔑 The breakthrough: a NONLINEAR input channel escapes the model's affine blind spot.**
Our champion is `mean_pool(Linear(x_t))`. Linear and mean **commute**, so the model can only ever see
**affine functions of the temporal mean** of its inputs. This is why every ratio/difference feature we
ever tried was redundant or toxic — `VH−VV` cross-pol scored **−0.023** because it is already in the
affine span. A **threshold** breaks the commutation:

```
mean_t 1[VH(t) < τ]   ≠   1[ mean_t VH(t) < τ ]
        └─ what the model computes: the empirical CDF value F̂_VH(τ) — a genuinely NEW coordinate.
```

The value is in the **nonlinearity**, not the exact τ. This is the single most important fact for the
whole round: **any feature that survives temporal averaging as a new statistic is candidate signal;
anything affine-in-the-mean is dead on arrival.**

**3. 📉 Feature SELECTION collapsed the profile to ONE threshold — a clean monotone signal.**

```
1 threshold  {−21}                         → 0.906492   ← BEST
4 thresholds {−22,−21,−20,−19}             → 0.901605
6 thresholds {−23..−18}                     → 0.898711
```

Monotone **1τ > 4τ > 6τ.** More thresholds on the *same axis* overfit and dilute. The winning τ = −21 dB
matches published operational SAR water thresholds (Chen 2018 Dongting −21.56; Ottinger). So: **fixed,
single, physically-anchored threshold per axis is right.** The next gain is NOT more τ on VH — it is a
**different nonlinear coordinate** (a new axis or a dispersion statistic).

**4. Seed variance is 0.019 and dominates everything.** σ_public ≈ 0.013–0.03 on ~309 public rows. Only
effects **> ~0.013** are resolvable. Every claimed win must be **seed-averaged / matched-seed** before we
believe it. The permanence win survived this test (+0.010 seed-averaged); most things do not.

**5. Local CV is anti-correlated with LB and we no longer trust it.** OOF AUC 0.995 → LB 0.70 is the
proof. The public LB is our only ground truth. This constrains the round: we cannot screen features
offline; every feature costs a submission, and we have ~70 left over 10 days. **Feature ordering
matters** — we need the research to rank candidates so we spend submissions best-first.

---

## 1. Self-contained problem statement

**Task.** Binary: is this ~10 m cell a managed aquaculture pond?

**Data (supplied only).** Per cell, a **12-month × 12-band** time series: Sentinel-1 SAR (**VH, VV**, in
dB, both present whenever a month is observed) + 10 Sentinel-2 optical bands (individually missing under
cloud). **No lat/lon, no spatial neighbourhood, no image patch, no static covariates.** Each row is one
isolated pixel's time series.

**Sizes.** Train **1,817** rows, ~40.2% positive. Test **1,030** (public ≈309, private ≈721). True test
prevalence believed ≈0.65 (label shift).

**Metric.** `0.6·F1 + 0.4·ROC-AUC`, two independently scored columns: `TargetF1` (binary at a **hard 0.5
cut** — threshold tuning is **forbidden by the rules**) and `Target` (raw probability, for AUC).

**The masking trap (defines what features are legal).** Train series have ~12 months; **test series have
only 4–6 months** (correlated dropout). Any statistic that is **biased at short n** becomes a
shift-carrier that encodes "how many months" instead of "what kind of pixel". **Safe (n-invariant):**
mean, median, interior quantiles, fractions/CDF values, U-statistics (GMD), L-moments. **Unsafe:** min,
max, range, counts, sums, run-lengths, argmax-timing, per-calendar-month columns.

**Architecture.** From-scratch Transformer: per-timestep `Linear`, masked **mean-pool** over observed
months, then a small head. Legal calibration = Platt fit on **train OOF only**, then literal 0.5 cut.

---

## 2. What is CONFIRMED — do not re-derive these

1. **Permanence channel `1[VH<−21]` = 0.906492, our best.** Seed-confirmed (+0.010 matched-seed avg).
2. **Affine channels are dead.** `VH−VV`, SDWI, total power, RVI/q/m-chi, all polarimetric decomps from
   GRD dB — all affine or monotone-in-q → the FFN reconstructs them → redundant/toxic. Proven arithmetic.
3. **Single fixed τ beats multi-τ** (monotone 1>4>6). τ=−21 dB is physically privileged.
4. **Trees / cross-class ensembling closed** (0-for-3; OOF illusion).
5. **Ensemble pooling buys +0.010** via calibration diversity (legal 0.5 cut averages members'
   operating points, not just their ranks).
6. **Seed variance 0.019; resolution ~0.013;** trust matched-seed avg + public LB only.

---

## 3. The model's blind spot, stated precisely (the theoretical spine)

For champion `y = w·mean_t(A x_t) + b` with per-timestep affine `A`:

- The model is a function of `mean_t(x_t)` **only**. Two pixels with the same temporal mean vector are
  **indistinguishable**, regardless of their within-year dynamics.
- A new input channel `g(x_t)` adds information **iff `E_t[g(x_t)]` is not an affine function of
  `E_t[x_t]`** — i.e. iff `g` is nonlinear in a way that survives averaging.
- **Empirical CDF values** `E_t 1[x_t<τ] = F̂(τ)` are the canonical such coordinate. **Squared channels**
  `E_t[x_t²]` give the model access to `Var_t` via `E[x²]−E[x]²`. **Cross products** `E_t[x_t·z_t]` give
  `Cov_t`. **AND-gates** `E_t 1[x_t<τ_1]·1[z_t<τ_2]` give joint-occupancy fractions. These are the ONLY
  families that escape the blind spot without changing the architecture.

**This is the exam question for the round: which nonlinear-in-the-mean coordinate, added as ONE channel,
most improves pond-vs-not separability under 4–6 month temporal shift?**

---

## 4. Internal convergence (8 agents) — the candidate menu, ranked

Our 8 research agents (threshold-feature learning; SAR VH physics; n-invariant statistics; nonlinear
cross-band; domain shift/validation; calibration/F1; ensembling theory; analogous winners) converged on
a short ranked list of **single-feature** candidates to test one at a time. **Validate or refute each,
with papers and math**, and re-rank.

| # | Candidate (ONE new channel, per-month, mean-pooled) | Mechanism it unlocks | Physics rationale | Agents |
|---|---|---|---|---|
| A | **VV permanence** `1[VV<τ_vv]` | 2nd CDF coordinate, new axis | VV adds specular/roughness info orthogonal to VH | 1,2,4 |
| B | **Pond-band occupancy** `1[−22 ≤ VH < −18]` | band-fraction, isolates regime | managed-pond mixed pixel (dike+water) sits ABOVE open water, BELOW dry | 2 |
| C | **VH temporal variance** via channel `VH²` (→ `Var_t`) | 2nd moment | rice swings 6–8 dB, ponds stable → variance is the **rice killer** | 2,3,4 |
| D | **SAR×optical rice-exclusion gate** `1[VH<τ_s]·1[NDVI<τ_v]` | joint occupancy | ponds are low-SAR AND never green; paddies green up → most surgical pond-vs-paddy cut | 4 |
| E | **VH CDF profile** (a few fixed τ) | richer CDF | already contains GMD dispersion (CDF↔GMD identity) | 1,3,4 |
| F | **GMD / IQR** dispersion scalar | exactly-unbiased spread | U-statistic, n-invariant rice killer | 2,3 |
| G | **Threshold-pair-U** `2/[n(n−1)]Σ 1[|xi−xj|>δ]`, δ≈4 dB | transition detector | the one dispersion signal a CDF profile canNOT encode | 3 |

**Note:** E was partly falsified already (multi-τ lost to single-τ). It stays on the list only as "does a
*small, well-chosen* second τ help", not "dense profile". C and D are the two freshest, highest-upside
bets — one attacks rice via **stability**, the other via **greenness**, and they are complementary.

---

## 5. Tensions to resolve (where our own agents disagreed)

1. **Fixed vs per-window/adaptive τ.** Agents 1&2: keep τ fixed (monthly optimum moves only ±1 dB;
   adaptive chases noise under shift — corroborated by 1τ>4τ). Agent 6: a per-window/relative τ (Otsu or
   the pixel's own low-percentile) might remove the residual period-sensitivity of a global constant.
   **Which wins under 4–6-month test truncation? Quantify the bias/variance tradeoff.**
2. **Temporal moments vs CDF — redundant?** Does adding `VH²` (variance) buy anything over a VH CDF
   profile, given the CDF already implies GMD? Or are they distinct enough that both help?
3. **All-permanence pool vs mixed pool.** Ensembling theory says the two +0.010 lifts (permanence=bias,
   pooling=variance) are largely orthogonal → most of both survives, BUT perm-members are more mutually
   correlated → the pooling lift shrinks; a **mixed {base + perm}** pool may beat all-perm. **Confirm.**
4. **Is the test shift pure label-shift or conditional too?** Decides whether the Saerens EM prior-shift
   correction (§6-Q5) is safe or harmful.

---

## 6. The six questions we need answered (ranked)

**Q1 — The single best next channel.** Of candidates A–G (§4), which ONE nonlinear-in-the-mean channel
has the strongest theoretical + empirical case to add ~+0.005 for pond-vs-not under temporal shift?
Rank them. Ground it in the affine-blind-spot math (§3) and SAR pond physics. We can only afford to test
a few; tell us the order.

**Q2 — Variance vs greenness for rice rejection.** Ponds' hardest confuser is rice paddy. Two orthogonal
attacks: (C) VH temporal **variance** (`VH²` channel; rice swings, ponds stable), and (D) a **SAR×optical
AND-gate** `1[VH<τ_s]·1[NDVI<τ_v]` (ponds never green). Which is more reliable when only 4–6 months are
observed and S2 is cloud-gapped? Is there a published SAR aquaculture/rice-discrimination result that
settles it? Give the τ values.

**Q3 — Fixed vs adaptive threshold under short-window shift.** Formalize the tradeoff (Tension 1).
Under n=4–6 month truncation with correlated dropout, does a **fixed** physical τ or a **per-window/
percentile-relative** τ generalize better? Quantify the estimator variance of a per-window Otsu at n=5.
Cite domain-generalization theory (invariant vs adaptive normalization under distribution shift).

**Q4 — Is there a nonlinear coordinate we're missing?** Beyond CDF values, squared channels, cross
products, and AND-gates (§3): is there a provably-more-informative single nonlinear temporal statistic
for a Linear+mean-pool model? (e.g. a specific U-statistic, an L-moment, an entropy that is n-invariant.)
We want the *maximally informative rank-1 nonlinear augmentation*, if one is characterizable.

**Q5 — The one legal operating-point lever.** F1 is scored at a **hard 0.5 cut** and threshold tuning is
**forbidden**. AUC is rank-invariant, so calibration alone can't change LB. Our agent flagged the
**Saerens–Latinne–Decaestecker (2002) EM prior-shift correction** to an **estimated (never LB-tuned)**
test prior π_t as the only legal way to move the 0.5 cut toward the F1-optimum. **Is this legal and safe
here?** Give the exact EM update, the estimator for π_t (BBSE / MLLS — Lipton 2018, Alexandari 2020), the
condition under which it helps vs hurts (pure label-shift vs conditional shift), and how to estimate π_t
from the **test features** (e.g. the VH-permanence marginal) without touching the LB.

**Q6 — The finalist pair.** We must designate **2** finalists for private-LB scoring. The shift playbook
says: one **diverse ensemble** + one **simple robust high-scorer**. Given our best single model
(`1[VH<−21]` = 0.906) and our best blend, what is the most robust 2-finalist choice to **minimize
private-LB regret** under the seed-variance 0.019 and the believed ≈0.65 test prevalence? Argue it from
ensemble/robustness theory.

---

## 7. Known dead ends — do not propose these

- Any **affine** cross-band feature: `VH−VV`, SDWI, total power, RVI, q, m-chi, all GRD-dB polarimetric
  decompositions (need off-diagonal `|C12|` from SLC we don't have).
- **Dense multi-τ** CDF profiles on the VH axis (overfit; monotone 1τ>4τ>6τ proven).
- **Free learnable τ** (drifts to exploit a 12-month train statistic that shifts at test).
- **GBDT / CatBoost / any tree** and **cross-class ensembling** (0-for-3, OOF illusion).
- **Frozen foundation-model embeddings** (Presto; adv-AUC 0.97, magnitude-dependent → collapses).
- **Any threshold search or prevalence pin against the LB** (illegal; overfits ~309 public rows).
- **Trusting local / single-matched-fold CV** as a feature ranker (anti-correlated with LB).
- n-**biased** statistics: min/max/range/counts/run-lengths/argmax-timing (shift-carriers under masking).

---

## 8. Reading list (verify, extend, correct)

**Threshold/tabular-nonlinearity:** Grinsztajn NeurIPS 2022 (why trees beat NN on tabular = piecewise
targets + NN smoothness bias); NODE (Popov 2020, arXiv:1909.06312); GRANDE (Marton ICLR 2024, STE hard
splits); McElfresh NeurIPS 2023 (n<50k → light-tuned GBDT ≥ NN).
**SAR pond/water physics:** Ottinger 2017 (rs9050440), Ottinger 2021 percentiles (rs13244851); Chen 2018
Dongting τ=−21.56 (PMC6015492); Nguyen 2021 rice (rs13050921); DE Africa water notebook; Duan 2020 ponds
stable-low (rs12183086).
**n-invariant statistics:** Hosking 1990 L-moments; U-statistics (Gini mean difference); Lubba 2019
catch22 (hand-pick fractions/moments only — most are n-biased).
**Domain shift / validation:** Xu 2025 "Invariant Features for Global Crop Type" (arXiv 2509.03497 —
absolute magnitude + GFM embeddings collapse; relative structure transfers); WILDS (Koh 2021); DomainBed
(Gulrajani 2021 — no DG method reliably beats ERM; model selection is the bottleneck); LSCD-TTA.
**Calibration / prior shift:** Saerens 2002 (Neural Comp 14:21); Lipton BBSE 2018 (arXiv:1802.03916);
Alexandari MLLS 2020 ICML (arXiv:1901.06852); Elkan 2001.
**Ensembling:** Krogh-Vedelsby NIPS 1995 (ambiguity decomposition); Brown 2005 (diversity); Gneiting 2007
(proper scoring for pooling).

---

## 9. Deliverable requested

A prioritized answer to **Q1–Q6**, each with: (a) the mechanism in terms of the affine-blind-spot math,
(b) the SAR/time-series physics, (c) at least one supporting paper (verified, not hallucinated), (d) the
concrete feature definition with τ / δ values, and (e) a one-line **expected LB effect** (with the caveat
that only >0.013 is resolvable). End with a single **ranked build order** for the ~10 days remaining:
which features to test, in which order, and which 2 to designate as finalists.
