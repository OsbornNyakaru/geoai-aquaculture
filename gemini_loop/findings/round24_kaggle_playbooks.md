# Round 24 — Q6: Kaggle / Zindi grandmaster empirical playbooks

**Agent:** Q6 researcher (competition practice, not theory).
**Date:** 2026-08-14.
**Scope:** documented winning solutions for competitions resembling ours —
fixed-threshold F1, severe train→test shift (adversarial AUC ~1.0), truncated/variable-length
time series, satellite tabular time series without imagery, ~1800-row tabular with noisy LB.

**Status legend:** VERIFIED = I read the write-up/thread. INFERRED = reconstructed from
secondary sources or partial reads. UNVERIFIED = could not confirm.

**Legality shorthand (per UPDATE_24 §5):**
(a) literal 0.5 cut on a genuine probability; (b) every knob train-only, no LB feedback;
(c) corrects p(y|x) rather than relabelling a fixed estimate.
**Theorem shorthand:** T1 = Platt annihilation (any affine logit map is absorbed by the
refit Platt a,b). T2 = pointwise-loss order invariance (any Σ y·l1(z)+(1-y)·l0(z) objective
leaves the ranking, hence AUC, unchanged; F1 effect is a pure threshold slide).

---

## Running log of findings

### METHOD NOTE
Kaggle is a client-side SPA; plain WebFetch returns only the `<title>`. Everything below that
is marked VERIFIED was pulled through the `r.jina.ai` reader proxy via curl, which returns the
full rendered markdown of Kaggle discussion/writeup pages. Anyone reproducing this should use:
`curl -sL https://r.jina.ai/<kaggle-url>`

---

### F1 — ICR: Identifying Age-Related Conditions (Kaggle, 2023). **1st place.** VERIFIED
- **Write-up:** https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/430843
  ("How on Earth did I win this competetion?", user `room722`). Model notebook:
  https://www.kaggle.com/room722/icr-adv-model
- **Why it resembles us:** 617 training rows (even smaller than our 1817), ~6400 teams,
  one of the most violent shake-ups in Kaggle history, metric = balanced log-loss (prior-
  insensitive by construction, i.e. the organisers hard-coded a class reweighting rather than
  letting you pick one). Public LB was pure noise.
- **What they actually did (verbatim-sourced):**
  1. DNN built on the **Variable Selection Network** from the Temporal Fusion Transformer
     (Lim et al., arXiv:1912.09363) — per-feature gating rather than a plain MLP.
  2. **No StandardScaler/MinMaxScaler.** Instead each scalar feature gets its own **linear
     projection to 8 neurons** — i.e. a learned per-feature embedding.
  3. **Very large dropout: 0.75 → 0.5 → 0.25** across the three main layers.
  4. **A second output head predicting "hardness to predict".** They first trained a baseline
     DNN, collected its OOF predictions, and defined a NEW binary label
     `hard = 1 if (y=1 and p<0.2) or (y=0 and p>0.8) else 0`, then trained the final model
     **multi-label on (target, hardness)**.
  5. 10-fold CV, each fold repeated 10–30 times, keep the **2 best models per fold**.
     They note training was so unstable that single-fold CV ranged 0.05–0.25.
  6. "Reweighting the probabilities at the end" (borrowed from
     https://www.kaggle.com/code/muelsamu/simple-tabpfn-approach-for-score-of-15-in-1-min).
  - Explicitly did NOT work for them: gradient boosting (overfit), feature engineering
    (overfit), the auxiliary "greeks" metadata (absent at test time).
- **LEGALITY / THEOREM CHECK, item by item:**
  - Items 1–3 (VSN architecture, per-feature linear projection, heavy dropout): **LEGAL**
    (a: untouched, b: architecture is a train-only knob, c: changes the hypothesis class
    hence p(y|x)). **Survives T1 and T2** — architecture changes are not affine logit maps
    and are not pointwise-loss reweightings; they genuinely reorder.
  - **Item 4 is the important one for us.** The "hardness" head is an auxiliary target that is
    a function of **x** (through a baseline model's OOF score), not of y alone. T2's premise
    is an objective `Σ y·l1(z) + (1-y)·l0(z)` — one fixed pair of functions of the single
    logit. A second head with an x-dependent target breaks that premise exactly the way the
    brief's **class E3** does. **SURVIVES BOTH THEOREMS.** **LEGAL:** the hardness label is
    constructed from train OOF only, no threshold moves, no LB feedback.
    → This is a documented 1st-place instance of the same family as the team's single
      biggest win (self-distillation), and it targets precisely the failure mode the brief
      describes in §3.4: confidently-wrong positives (their `y=1 and p<0.2` is our missed
      positives with median score 0.170).
  - Item 5 (select best models per fold on CV) — **DO NOT COPY.** The brief §4 states OOF is
    blind here (0.97 OOF vs 0.72–0.907 LB). room722 himself says "cv somehow worked in this
    competition", i.e. he was surprised. Their CV worked because ICR had no covariate shift;
    ours does.
  - Item 6 (probability reweighting) — **ILLEGAL AND DEAD.** It is a prior correction, i.e.
    an affine shift in logit space → **annihilated by T1**, and its only effect on F1 would be
    a threshold slide → **killed by T2**. Mark INAPPLICABLE.
- **Needs nothing we don't have.** ~71k-param transformer already; adding a second head and
  a per-feature input projection is a few lines.
- **Bonus empirical datum for the team's report:** the widely-copied ICR public kernels that
  *thresholded/reweighted probabilities to match the public test set* were exactly the ones
  that collapsed on private LB. This is documented, independent, empirical support for the
  team's self-imposed no-threshold-tuning rule — worth citing in the 35% report component.
  (Secondary source, INFERRED from the shake-up discussion; see
  https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/430852 )

---

### F2 — PLAsTiCC Astronomical Classification (Kaggle, 2018). **1st place, solo, Kyle Boone.** VERIFIED (read the full thread)
**⭐ This is the closest structural analogue to our competition that exists in the public record.**

- **Write-up:** https://www.kaggle.com/competitions/PLAsTiCC-2018/discussion/75033
- **Code:** https://github.com/kboone/avocado
- **Peer-reviewed paper ("Avocado"):** arXiv:1907.04690 / Astron. J. 158, 257 (2019),
  DOI 10.3847/1538-3881/ab5182
- **Why it is the closest analogue:**
  - Multi-band (6 passbands) irregular **time series in TABULAR form — no imagery**, exactly
    like our 12 bands × 12 months.
  - Train = 7848 objects; test = 3.5M. **Train is well-sampled, test is badly-sampled** —
    literally the same designed difficulty as our "train has 12 months, test has 4–6".
  - The train set was drawn by a completely different (spectroscopic) selection process, so
    train and test are trivially separable — the same regime as our adversarial AUC = 1.0.
  - Small-ish labelled set, big unlabelled test set (our 1817 / 1030).
- **What he actually did — the winning move, quoted:**
  > "Now the training set is very different from the test set. To deal with this, I took every
  > lightcurve in the training set and **degraded it up to 40 times** to get something that
  > looks like the less well-sampled lightcurves in the test set."
  The degradation had five components: (1) modify brightness; (2) modify redshift, which
  **dilates time and changes brightness**; (3) **insert large seasonal gaps like the ones in
  the real data**; (4) resample the auxiliary photo-z and its error from a model of the
  measurement process; (5) **simulate the detection/selection step** — i.e. decide which
  degraded objects would even have made it into the released dataset.
  Result: 7848 → ~270,000 training rows. Then features (200 of them, from Gaussian-process
  interpolations of the light curves), then **one LightGBM, 5-fold CV**, with all 40
  degradations of an object **kept in the same fold** (to stop leakage inflating CV).
- **RELEVANCE TO US — the three things he did that we apparently do not:**
  1. **AUGMENTATION INTENSITY.** He generated **up to 40 degraded views per training row.**
     Our masked-window augmentation is **K = 2**. He is a 20× more aggressive instance of
     the exact same idea, and he describes it as the thing that won the competition. This is
     the cheapest testable delta in this entire report: raise K.
  2. **FEATURES THAT DESCRIBE THE OBSERVATION PATTERN ITSELF.** Quoted:
     > "For poorly sampled lightcurves, the GP doesn't always give great results, so I added
     > features that let the model know how well the GP is doing. This basically boils down to
     > **counting the number of observations in different windows** around maximum light. I
     > also added features related to the **signal-to-noise in each band**."
     I.e. he gave the model explicit inputs describing *how much and what kind of data it is
     being shown*, so the model can learn a different decision function for well-observed vs
     badly-observed rows. Our analogue: window length (4/5/6), which relative months are
     present, gap positions, per-band count of non-missing months, per-band SNR proxy.
     Under the brief's own §6 general result — "what is outside our model's reach is the
     **nonlinear temporal statistic**, not the band index" — observation-count and
     missingness-geometry features are exactly nonlinear statistics of the window, and are
     **not** ≥97% linearly reachable from raw band values.
  3. **SIMULATING THE SELECTION MECHANISM**, not just the corruption. He did not merely crop;
     he modelled which rows would have survived the pipeline that produced the test set.
     Our analogue: is the 4–6-month window in test rows *uniformly* placed, or is it
     correlated with e.g. cloud cover / season / label? We already "draw from the measured
     test distribution" of window lengths (§2 item 2) — but do we match the **joint**
     distribution of (start month, length, per-band missingness), or just the marginal
     length? If only the marginal, this is a real, cheap gap.
- **LEGALITY (§5):** (a) untouched — he never moved a threshold; this is a training-data
  transformation. (b) The degradation parameters were "tuned to the training/test datasets"
  using the **unlabelled test features only** — no labels, no LB feedback. That is exactly
  what our masked-window augmentation already does and what the brief already permits.
  (c) It changes p(y|x) by changing the hypothesis the model is fit to. **PASSES.**
- **THEOREMS:** Augmentation multiplies the number of (x, y) pairs and changes *which x's*
  the loss is evaluated on. T2's premise (a fixed pair of pointwise loss functions over a
  fixed sample) does not hold — the sample itself is different. **SURVIVES T2.** Not an
  affine logit map, so **SURVIVES T1.** Adding observation-pattern features changes the
  input space entirely: survives both trivially.
- **NEEDS NOTHING WE DON'T HAVE.** No coordinates, no imagery, no external data, and a 71k-
  param transformer trains fast enough for K=8–20.
- **THE ONE ILLEGAL PART, flagged by the author himself:** for the "class 99" unknown class he
  **probed the leaderboard**. He states it "defeats the purpose" and only did it after
  confirming with organisers it was within Kaggle rules. It moved him from 0.726 to 0.670.
  **DO NOT COPY — violates §5(b) and our self-imposed no-LB-feedback rule.** Worth citing in
  our report as an example of the distinction we are drawing.

