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


---

### F3 â€” IEEE-CIS Fraud Detection (Kaggle, 2019). **1st place, Chris Deotte + Konstantin Yakovlev.** VERIFIED (read Parts 1, 2 and the UID/adversarial thread in full)
**â­ This is the canonical documented case of "adversarial validation AUC = 1.0" â€” the exact regime of our Â§4 â€” and the winners' reading of it CONTRADICTS ours.**

- **Write-ups (all read):**
  - Part 1: https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111284
  - Part 2 (technical: validation strategy, feature selection): https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111308
  - "How to find UIDs" / adversarial validation mechanics: https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111510
  - Reproducible notebook: https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600
- **Why it resembles us:** 590k train / 507k test tabular rows, metric ROC-AUC, and â€” the point â€”
  **adversarial validation returned AUC ~ 1.0 (he writes "AUC = 0.999" for a 53-column subset and
  again "AUC = 0.999" for the V-columns alone).** Train and test were, for practical purposes,
  perfectly separable. This is our Â§4 condition, in a competition that was then won outright.

- **THE CORRECTION THIS OFFERS TO OUR Â§4. VERIFIED QUOTE, first paragraph of the 1st-place post:**
  > "The reason adversarial validation has AUC=1 is **not** because the nature of fraud changes
  > radically over time but rather because the **clients** in the dataset change radically over time.
  > Once you realize this, you will know that the challenge in this competition is building a model
  > that can predict **unseen clients** (not unseen time)."

  Our brief treats adversarial AUC = 1.0 as a terminal fact ("therefore importance weighting is not
  identifiable"). The 1st-place team treats it as an **un-decomposed measurement**, and their first
  job was to find out **which coordinates carry the separation** and whether those coordinates are
  *nuisance* (identity / observation-process) or *predictive*. They did this literally: run the
  adversarial LGBM, read its feature importances, and that list ("D10n, D1n, D15n, C13, card1, D2n,
  card2, addr1, TransactionAmt, dist1") became the map of what to delete and what to aggregate over.
  - **The actionable analogue for us, and I believe it is a real gap:** our adversarial AUC of 1.0000
    is almost certainly achieved by the **missingness pattern alone** â€” train rows have 12 observed
    months, test rows have 4-6. That is a *designed nuisance coordinate*, not evidence that the
    predictive covariates fail to overlap. **The measurement that matters is adversarial AUC between
    MASKED train rows (after our own window crop) and real test rows.** If that number is well below
    1.0, then the brief's inference "adversarial AUC = 1.0 => importance weighting is not
    identifiable" is **measured in the wrong space**, and the IW lane is not closed on the grounds
    stated. If it is still 1.0, that is itself a finding â€” it says our crop does not reproduce the
    test observation process (see F2 / PLAsTiCC point 3), which is a bug in the augmentation.
    **I cannot tell from UPDATE_24 which of the two spaces was measured. If it was raw-vs-test, the
    Â§4 "FIRM" grade on importance weighting is not supported by that experiment.**
  - This is a **diagnosis-only** recommendation: one adversarial LGBM fit, no training run, no
    submission. It fits the remaining budget on deadline day and it is report-grade either way.

- **What they concretely did (the winning move):**
  1. Identified that the label is attached to a **latent group** (the credit card / client), not the
     row. 96.9% of multi-transaction clients are all-negative, 2.9% all-positive, 0.2% mixed.
  2. **Did NOT feed the group identifier to the model** â€” 68.2% of private-test clients are unseen in
     train, so a UID feature could not generalise. Instead they replaced it with **aggregations over
     the UID** (`groupby(uid)[cols].agg(['mean','std','nunique'])`) and then **deleted the uid column**.
     LB 0.9510 -> 0.9602 from ten lines of code.
  3. **Post-processing:** replace every prediction within a client by that client's mean prediction
     (LB 0.9602 -> 0.9618).
  4. Deleted features that failed a shift-transfer test (below).
- **THE TWO VALIDATION INSTRUMENTS â€” the part most relevant to our Â§4 "OOF is blind".**
  Verbatim from Part 2:
  - **"Time consistency"**: *"train a single model using a single feature (or small group of features)
    on the first month of train dataset and predict `isFraud` for the last month of train dataset.
    This evaluates whether a feature by itself is consistent over time. 95% were but we found 5% of
    columns hurt our models. They had **training AUC around 0.60 and validation AUC 0.40**."*
    A feature scoring **below 0.5** on the transfer split is actively anti-predictive out of
    distribution â€” the block `V322-V339` was deleted on this evidence alone.
  - **A battery of deliberately shift-matched splits, never one CV number:** *"Train on first 4 months
    of train, skip a month, predict last month. We also did train 2, skip 2, predict 2. We did train 1
    skip 4 predict 1 ... a CV GroupKFold using month as the group."* Plus **stratified error analysis**:
    they scored each model separately on known / unknown / questionable clients and found the three
    models had different strengths (XGB 0.99723 on known UIDs, LGBM 0.92117 on **unknown** UIDs,
    CAT 0.98834 on questionable) â€” and ensembled *because the strata differed*, not because the
    average was better.
  - **Our analogue, buildable today with no training run: a "window-consistency" test.**
    For each engineered feature or channel, fit a one-feature model on **full-window** train rows and
    score it on **masked** train rows (same labels, cropped inputs, held-out fold). Any channel whose
    transfer AUC falls below 0.5 is anti-predictive under the deployment observation process and
    should be dropped. This is **train-only, label-only, LB-free** â€” it satisfies Â§5(b) exactly â€” and
    it is the *mechanism-specific, control-baselined* kind of instrument the brief says worked once
    (Â§4, last bullet). It is a direct constructive answer to "OOF at 0.97 predicts nothing".
- **LEGALITY (Â§5):** (a) untouched â€” none of the above moves a threshold; the post-processing step (3)
  averages predictions inside a known group, and we have no groups, so it does not port anyway.
  (b) every knob â€” which features to delete, which aggregations to build â€” was set by train-internal
  transfer tests, not LB. **Note the one exception they admit, and do not copy it:** *"We reviewed LB
  scores (which is just train 6, skip 1, predict 1 and no less valid than other holdouts)."* That
  practice violates our Â§5(b). It is worth naming in our report as exactly the line we drew and a
  1st-place Kaggle team did not.
  (c) deleting anti-transfer features and adding aggregates changes the hypothesis class, i.e. it
  genuinely re-estimates p(y|x) rather than relabelling a fixed estimate.
- **THEOREMS:** feature deletion and feature construction change the **input space**. T1 applies only
  to affine maps of a fixed logit â€” irrelevant here. T2 assumes a fixed x with a reweighted pointwise
  loss; here x itself changes, so the population minimiser T(eta(x)) is a monotone function of a
  *different* eta and the ranking genuinely moves. **SURVIVES BOTH.** The adversarial-validation
  diagnostic is not a model change at all, so it is trivially safe.
- **WHAT DOES NOT PORT.** The headline "magic" (UID recovery) requires a latent entity spanning
  multiple rows. Our rows are independent locations with randomised IDs, no coordinates, and the
  brief states IDs were checked and encode nothing. **Do not chase a UID here.** What ports is
  (i) the *decomposition* of adversarial AUC and (ii) the shift-matched per-feature transfer test.
- **One more datum from the comments, VERIFIED, and it cuts against a temptation:** asked about class
  imbalance, Deotte replies that `class_weights=[1,2.5]` helped his CatBoost only, that up/downsampling
  *"didn't help my LGBM nor XGB models"*, and that it would matter *"if the train.csv had 3% isFraud
  but the test.csv had 10%"*. That is precisely our prior-shift situation â€” **and it is exactly the
  move T2 proves is a pure threshold slide on an unchanged ordering.** His own evidence is equivocal
  (helped 1 model of 3); ours is a theorem. **No contradiction to Â§6 â€” if anything it supports leaving
  class weighting alone, and it is a nice citation for the report: even a 1st-place team got only a
  model-specific wobble out of it.**


---

### F4 â€” Santander Customer Transaction Prediction (Kaggle, 2019). **1st place, fl2o + Silogram.** VERIFIED (read the full write-up and the top comments)
**Why this one is here: it is the best-documented case of a competition won by TRANSDUCTIVE FEATURE ENGINEERING on the unlabeled test set â€” the family that contains our single biggest win â€” and it gives us a measured number for how much the pseudo-labelling part was actually worth.**

- **Write-up:** https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003
- **The enabling public kernel they credit:** YaG320, "List of Fake Samples and Public/Private LB split",
  https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split
- **Their code:** NN https://www.kaggle.com/fl2ooo/nn-wo-pseudo-1-fold-seed ; data build https://www.kaggle.com/fl2ooo/create-data
- **Setup:** 200k train / 200k test, 200 anonymised numeric features, metric ROC-AUC. Public LB moved
  by ~0.02 between "no magic" and "magic", which for AUC on 200k rows is enormous.

- **What they actually did â€” the three separable mechanisms:**
  1. **TRANSDUCTIVE FEATURES COMPUTED OVER TRAIN UNION TEST (the win).** For each of the 200 raw
     features they built a categorical "has one feat" with 5 levels, one of which is literally
     *"This value is unique in data + test (only including real test samples)"*, plus 200 numeric
     "not unique feat" columns that **replace any value unique in train+test by the feature mean**.
     Documented LB ladder, quoted: uniqueness features from **training data only -> .910 LB**;
     adding the "not unique" variants -> **.914 LB**; extending the uniqueness computation to
     **train + real test -> .921 LB** and eventually .925/.927.
     **The step from train-only to train+test statistics was worth ~+0.007 AUC on its own.**
  2. **Augmentation by within-class feature shuffling.** *"duplicate and shuffle 16 times samples with
     target == 1, 4 for target == 0"* for the LGBM, and an on-the-fly batch version for the NN.
  3. **Pseudo-labelling, and note how modest it was.** LGBM: *"2700 highest predicted test points as 1
     and 2000 lowest as 0"*. NN: 5000 highest / 3000 lowest. **Measured effect on the NN, from their
     own numbers: private .92497 -> .92546 = +0.00049.** The transductive features were worth ~15x
     more than the pseudo-labels.

- **THE READ-ACROSS FOR US, and it is a genuine untested lever.** Our brief's class E3 is described as
  "changing the TARGET as a function of x" and our biggest win (self-distillation, +0.0100) sits
  there. Santander is a different, **label-free** transductive family that our brief does not
  enumerate at all: **features of x computed against the pooled train-union-test corpus.** No test
  labels, no LB feedback, no external data. Concretely portable candidates for our 12x12 table:
  per-band-month **rank / empirical quantile of the value within the pooled corpus**; per-row
  distance to its k-th nearest neighbour in the pooled corpus; per-row local density. These are
  **nonlinear statistics of the pooled distribution**, and by the brief's own Â§6 general result
  ("what is outside our model's reach is the nonlinear temporal statistic, not the band index") they
  are not reachable from raw bands by the network.
  - **Honest caveat, stated up front:** Santander's specific magic worked because the organisers had
    injected synthetic rows into test and because the 200 features were conditionally independent
    given y. **Neither holds for us.** I am not claiming the trick ports; I am claiming the
    **category** â€” label-free transductive features â€” is one our brief has not enumerated, and that
    it has a 1st-place existence proof.
  - **Also honest:** Â§6 grades "adding channels / width" as a repeated loser (soft). A pooled-corpus
    rank channel is an added channel. Expected value is therefore modest and I would not rank it
    above the diagnostics in F3 on deadline day.

- **LEGALITY (Â§5):** (a) threshold untouched. (b) The uniqueness statistics are computed from test
  **features only** â€” no labels, no LB probing. This is the same license our masked-window
  augmentation already uses when it draws window lengths "from the measured test distribution"
  (Â§2 item 2), so it is consistent with a rule the team has already accepted. Their pseudo-label
  counts (2700/2000, 5000/3000) **are hand-set magic numbers and would need a train-only rule here** â€”
  under Â§5(b) we could not pick them by LB. (c) yes: it changes the representation, so it re-estimates
  p(y|x).
- **THEOREMS:** all three mechanisms change either the input space (1), the empirical sample (2), or
  the target on extra rows (3). None is an affine logit map -> **survives T1**. None is a fixed pair of
  pointwise loss functions over a fixed sample -> **survives T2**. The pseudo-labelling in (3) is the
  same escape as our self-distillation.
- **A datum worth putting in our report:** the winner's own teammate writes that the whole 0.901 ->
  0.92x jump *"is based on the ability to identify the fake test data, which should not have been
  possible if the data had been presented correctly."* Competition-winning transductive tricks are
  frequently exploitation of a data-preparation artifact rather than transferable modelling. That is
  a fair, cited caution to include when we discuss why we did not chase test-set structure harder.


---

### F5 â€” VSB Power Line Fault Detection (Kaggle, 2019). **1st place, `mark4h`.** VERIFIED (read the full overview post; the notebook's cell source did not render through the proxy, so the threshold question below is marked UNVERIFIED)
**Why it is here: a threshold-dependent metric (Matthews correlation coefficient), a tiny dataset, a violent shake-up, and a raw time-series input â€” and the winner won with NINE hand-built NONLINEAR TEMPORAL STATISTICS. This is independent corroboration of our own Â§6 "general result".**

- **Write-up:** https://www.kaggle.com/competitions/vsb-power-line-fault-detection/discussion/87038
- **Winning notebook:** https://www.kaggle.com/code/mark4h/vsb-1st-place-solution (private score 0.71899)
- **Setup:** 8712 train signals grouped into 2904 three-phase measurements, 800k samples per signal,
  metric **MCC** â€” threshold-dependent like F1, and notoriously unstable at small n. Enormous
  public->private shake-up.

- **What he actually did, quoted from the post:**
  > "In the end my best model was a simple LightGBM model, with some preprocessing of the data and a
  > **small number (9 in total) of features** based on the patterns common to the train and test data."

  The nine features are all **counts and thresholded aggregates of detected peaks**, e.g.
  *"1. The total number of peaks; 2. The number of peaks in quarters 0 and 2; 3. The number of peaks
  in quarters 1 and 3; 4. The mean 'sawtooth' RMSE value in quarters 0 and 2; 5. The std height in
  quarters 0 and 2 ..."*. The peak detector itself is a **thresholding rule discovered from the data**:
  local maxima over a window of 51, then *"a knee point detection to find the noise floor"*.
  - He explicitly **rejected denoising**: *"I spent some time trying to denoise the traces but in the
    end came to the conclusion that it was just as likely to remove signal as it was noise."*
  - He explicitly **declined to invent a clever CV scheme**: *"I didn't use any specific CV technique
    to prevent over-fitting, instead I spent most of the time trying to understand the data."*
  - Training: LightGBM, **5-fold CV repeated 25 times with a different seed each time**.

- **WHY THIS MATTERS TO US â€” it corroborates the team's own best general finding.** UPDATE_24 Â§6 ends
  with: *"what is outside our model's reach is not the band index but the **nonlinear temporal
  statistic** (median, min, threshold-count). Our one winning hand-feature is exactly that: a
  threshold indicator, not a ratio."* The VSB winner's entire feature set is **threshold-counts of a
  raw waveform**, and it beat every deep model in that competition. Our SAR permanence channel
  `1[VH_dB < -21]` (+0.010) is the same object. **This is a clean, citable, independent 1st-place
  precedent for the single design principle we already discovered ourselves** â€” exactly the kind of
  external validation that is worth marks in the 35% report component.
  - **Untested extension it suggests, cheap:** he did not use one fixed threshold; he **derived the
    threshold per trace** from a knee-point on that trace's own sorted peak heights. Our `-21 dB` is a
    single global constant. A **per-row adaptive** version â€” e.g. the count of months in which VH sits
    below that row's own (median - k*MAD) â€” is a strictly more expressive nonlinear temporal statistic
    of the same family, and it is computed on the input, so it is not a decision threshold and does
    not touch Â§5(a). Whether it helps is unknown and it needs a training run, so on deadline day this
    is a **report/future-work item, not an action.**
  - He also **changed the unit of prediction** (measurement, not signal) because *"it looked as if all
    3 signal traces had been marked as faulty even though only one had any obvious signs of a fault"* â€”
    i.e. he handled label noise by aggregating to the level at which the label was actually assigned.
    **Does not port**: our rows have no group structure.

- **THE THRESHOLD QUESTION, answered honestly.** VSB required a **binary submission**, so competitors
  were free to pick the MCC cut and essentially all public kernels searched it on OOF. **The 1st-place
  post never mentions a threshold at all**, and I could not render the notebook's code cells through
  the proxy, so **I cannot verify whether he tuned one â€” treat VSB as INAPPLICABLE on the
  "fixed-threshold" axis.** What it is applicable to is: threshold-dependent metric + tiny data +
  noisy LB, and the winning response to that combination was **fewer features, simpler model, more
  seeds** â€” not a cleverer operating point.

- **LEGALITY (Â§5):** (a) the *feature* thresholds are properties of x, not the decision rule, so a
  literal 0.5 cut is untouched. (b) all nine features were designed from train data and from
  *"patterns common to the train and test data"* â€” visual/structural inspection of unlabeled test
  signals, no LB feedback. (c) new nonlinear features change the hypothesis class, so p(y|x) is
  genuinely re-estimated. **PASSES all three.**
- **THEOREMS:** feature construction changes the input space; not an affine logit map (**survives T1**)
  and not a pointwise loss reweighting over a fixed sample (**survives T2**). Seed-repeated CV is
  variance reduction and is already what we do with our 10-seed pool.
- **The one lane it argues against, weakly:** Â§6 grades GBDT as FIRM-closed after three failures on our
  data. VSB is a case where **9 features + LightGBM beat every neural approach** on a raw time series.
  That is not evidence our GBDT results were wrong â€” the difference is that his 9 features carried all
  the signal. It does suggest the failure mode of a GBDT here is *feature poverty*, not the model
  class. Not enough to reopen the lane on deadline day; worth one sentence in the report.


---

### F6 â€” Radiant Earth **Spot the Crop** Challenge (**Zindi**, 2021). **Top awarded solution, `kiminya`.** VERIFIED (read the README *and* the actual training source)
**â­ Closest DOMAIN analogue in the record: Zindi, Sentinel-2, per-field TABULAR band time series with no imagery, irregular/variable observation counts. And the winning code contains a MIXUP-FAMILY augmentation â€” a documented winning precedent for the brief's untested class E2 (Q3a) in our exact data type.**

- **Repo (organiser-curated winners):** https://github.com/radiantearth/spot-the-crop-challenge
- **Competition:** https://zindi.africa/competitions/radiant-earth-spot-the-crop-challenge
- **Winner README:** https://github.com/radiantearth/spot-the-crop-challenge/blob/main/2nd%20place%20-%20Kiminya/README.md
- **Training source I actually read:** `.../2nd place - Kiminya/src/train_d3c4.py`, `train_d10c4.py`,
  `train_d38c4_ftrs0.py`, `train_d38c4_ftrs1.py`
- **Setup:** classify crop type in Western Cape, South Africa from a **time series of Sentinel-2
  bands aggregated per field** â€” i.e. exactly our data shape (bands x dates per row, no pixels).
  76 unique observation dates, irregular per field. Metric = **cross-entropy** (leaderboard scores
  0.6600 / 0.6813 / 0.7188 for the three awarded teams; lower is better).
  âš ï¸ **Metric is NOT threshold-dependent â€” mark this entry INAPPLICABLE on the fixed-threshold axis.**

- **What the winner actually did:**
  1. **Model family: `XceptionTime` (arXiv:1911.03803) and `InceptionTime` (arXiv:1909.04939)**, from
     the `tsai` library â€” 1-D convolutional time-series classifiers. **Not a transformer, not a GBDT.**
     Final ensemble = 6 model configurations x 5 stratified folds = **30 models**.
  2. **ðŸ”‘ THE OBSERVATION-GRID PROBLEM, solved by making it an ENSEMBLE AXIS.** Quoted from README:
     > "Two methods are to standardize the number of observations for each field:
     > - Resample the original data to intervals of **3, 5, 7 and 10 days**. Null values are imputed
     >   with zeros.
     > - Group the 76 unique dates in the original data into **38 sets of adjacent dates**; aggregate
     >   and average the data by field and date-set.
     > This results in **5 different versions of the original data**."
     Four XceptionTime models on the four resampled grids, two InceptionTime models on the 38-set
     grid. **The temporal resolution is not tuned to one best value â€” it is diversified over.**
  3. **ðŸ”‘ `CutMix1D(1.)` â€” VERIFIED FROM SOURCE, present in every one of the four training scripts:**
     `cbs=[CutMix1D(1.), SaveModelCallback(monitor='valid_loss')]`.
     `tsai`'s `CutMix1D` is a `MixHandler` â€” it splices a contiguous time segment from another sample
     into the current one **and mixes the two labels by the same lambda**. That is vicinal risk
     minimisation on time series.
  4. **`TSStandardize(by_sample=True, by_var=True)`** â€” VERIFIED FROM SOURCE, also in all four:
     each row is standardised **against its own series, per variable**, not against a training-set
     mean/std.
  5. Different band / vegetation-index subsets per member *"to increase variation and improve
     generalization of the final ensemble."* 40 epochs, `fit_one_cycle`, `lr_max` 1e-3 to 3e-3.

- **THREE READ-ACROSSES, in order of value to us:**
  - **(i) Q3(a) now has a winning precedent in our exact domain.** The brief lists mixup/VRM as
    "never tried" and notes the two theorems do not apply to it. Here is a top Zindi solution on
    Sentinel-2 tabular band time series whose every model runs a **cut-mix / label-mixing**
    augmentation. **Theorem check: CutMix mixes both x and y, so the objective is
    `l(f(x_mix), lambda*y_i + (1-lambda)*y_j)` â€” the target depends on ANOTHER example. T2's premise
    (a fixed pair of pointwise functions `l1, l0` of one example's logit) fails outright.
    SURVIVES T2. Not an affine logit map, SURVIVES T1.** Legality: (a) untouched; (b) the mixing
    coefficient is a train-only hyperparameter, no LB; (c) it changes the fitted function, hence
    p(y|x). **PASSES all three prongs.** In `tsai` this is literally one callback â€” the cheapest
    E2 experiment available, if there were time for a training run.
  - **(ii) The observation-grid ensemble is a diversity axis we do not use, and it argues against a
    "FIRM" lane.** Â§6 closes "blending a weaker decorrelated member â€” lost three times." Kiminya's
    members are **not weaker**; they are *equally strong models fit to different temporal
    resamplings of the same data*. That is a different construction from blending in a weak learner,
    and our failures do not speak to it. Our analogue would be a pool where members see the same rows
    at different temporal granularity (monthly, 2-month means, 3-month means) rather than 10 seeds of
    one grid. **This is a real, cited gap in our ensemble design.**
  - **(iii) A direct challenge to the "optical indices are >=97% linearly reachable" closure (Â§6, Q5).**
    Kiminya deliberately feeds **different index subsets to different members** *for variance*, not
    for information. Our closure argues indices carry no information the network cannot represent.
    **Representability is not the same claim as learnability**: with 1817 rows and SGD, supplying an
    index changes the inductive bias and the optimisation path even when the function is in the span.
    The measurement that would settle it is whether two members differing only in index inputs have
    **decorrelated errors** â€” a train-only check. Our closure, as worded, does not address this.
    (I flag it for Q5's owner; it is their question, not mine.)

- **âš ï¸ WHERE I WOULD NOT COPY IT.** `TSStandardize(by_sample=True)` removes each row's own level and
  scale. For crop type that is right â€” phenological *shape* is the signal. For us it would **destroy
  the absolute SAR level**, and our single best hand-feature is `1[VH_dB < -21]`, an absolute-decibel
  threshold worth +0.010. **Do not per-sample-standardise the SAR channels.** If tried at all, it
  belongs on the optical bands only. Flagging this because it is the one item here that looks
  attractive and would silently delete our best feature.
- **On architecture:** InceptionTime/XceptionTime are a credible, small, fast alternative family to
  our 71k transformer, and `tsai` implements both. Not actionable today (needs training), but it is
  the right citation if the report needs to justify the architecture choice against domain precedent.


---

### F7 â€” PSEUDO-LABELLING: what the documented competition practice actually is, with numbers
**This is the family our biggest win (+0.0100, one round of soft self-distillation on the 1030 test rows) already lives in. Verdict up front: the competition record SUPPORTS the team's one-round cap and does NOT support expecting much more from additional rounds â€” but it does show two variants we are not using.**

- **Canonical teaching artifact (VERIFIED page + comments; the notebook's code cells would not render
  through the proxy, so items marked [INFERRED] below are from the table of contents, the comment
  thread, and secondary description, not from reading the code):**
  Chris Deotte, "Pseudo Labeling - QDA - [0.969]", Instant Gratification â€”
  https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969
  (best score 0.97059; section headings VERIFIED: *Step 1 - Build first model / Step 2 - Predict test
  data / Step 3 and 4 - Add pseudo label data and build second model / Step 5 - Predict test data /
  **Why does Pseudo Labeling work?***). Comments: https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969/comments
  - **VERIFIED quote, Deotte answering "is this a strategy that tries to overfit on the test set?":**
    > "It's more like **data augmentation**. If done correctly, you won't overfit the test set but
    > instead just gain more training data."
  - **VERIFIED quote on the confidence cut**, from the same author's practice on this problem: with a
    strict rule (`>0.99` and `<0.01`) only about **85% of test rows get a pseudo-label**; when he
    wanted all of them he assigned the label **stochastically** â€” draw `u ~ U(0,1)` and set the label
    by comparing `u` to the predicted probability. *(Reported in the search index of the comments
    thread; I did not see the code line itself â€” mark [INFERRED].)*
    â†’ **Note what that stochastic rule is: an unbiased sampler from the model's own soft label.**
    In expectation it is exactly our soft self-distillation. We are already at the better end of this
    design space; hard-thresholded pseudo-labels are the *cruder* variant.
  - **The standard correctness rule, [INFERRED] but universally stated in this literature and worth
    checking in our own code:** pseudo-labelled test rows go into the **training folds only** and must
    never enter a validation fold, or CV becomes self-fulfilling. Given the brief's Â§4 ("OOF is
    blind", OOF 0.97 vs LB 0.72-0.907), it is worth confirming our distillation pipeline does this â€”
    a leak here is one candidate explanation for why our OOF is uninformative.

- **HOW MUCH IT IS ACTUALLY WORTH, measured, from a 1st place (see F4 for the source):**
  Santander CTP 1st place: NN private **0.92497 -> 0.92546 with pseudo-labels = +0.00049**, on a
  competition where their *transductive feature* step was worth ~+0.007. **In that solution
  pseudo-labelling was ~1/15th of the transductive gain.** Our +0.0100 from one distillation round is,
  by comparison, an unusually large return for this family â€” which is itself weak evidence that the
  remaining headroom in it is small.
- **Counterexample where it was large (VERIFIED existence, not read in depth):** Eedi - Mining
  Misconceptions in Mathematics, 1st place, reports pseudo-labelling on synthetic examples worth
  **+0.044** â€” https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution
  Different modality (LLM retrieval) and the pseudo-labels were on *generated* data, not test rows.
  I flag it only so the report is not accused of cherry-picking small numbers.

- **THE DOCUMENTED FAILURE MODE, and why the team's one-round cap is defensible in writing.**
  The mechanism is **confirmation bias**: the student is trained toward the teacher's own errors, and
  because those errors are the confidently-wrong ones, repetition entrenches them.
  Canonical citation: Arazo, Ortego, Albert, O'Connor, McGuinness, *"Pseudo-Labeling and Confirmation
  Bias in Deep Semi-Supervised Learning"*, arXiv:1908.02983 (IJCNN 2020). VERIFIED as the correct
  reference for the phenomenon; their proposed mitigations are **mixup** plus a minimum number of
  labelled examples per minibatch.
  - **This is a load-bearing connection for us.** The brief's Â§3.4 says the positives we miss have
    **median score 0.170, ten below 0.10, none in [0.45,0.50)** â€” i.e. our errors are exactly the
    confidently-wrong kind that self-distillation cannot fix and a second round would harden.
    **The competition and academic records agree: do not run round two of self-distillation.** That
    is a defensible, citable justification for a cap the team currently states without a source, and
    it belongs in the report.
  - **And note the mitigation Arazo et al. name is mixup** â€” the same technique the Spot-the-Crop
    winner runs on Sentinel-2 time series (F6). Two independent routes point at the same untested
    class E2. If there were time for one training run, this is the one.

- **WHAT WE ARE NOT DOING THAT THE RECORD SUPPORTS â€” one item, honestly small:**
  Every documented recipe pseudo-labels with a **different, independently-built model** or with an
  ensemble that includes members the student does not contain. Santander used an LGBM+NN blend;
  Deotte's QDA example teaches with a model class distinct from the downstream one. **Our teacher is
  our own 10-seed pool â€” the student and teacher share an architecture, a feature set and an
  inductive bias**, which is the worst case for confirmation bias. The brief's Q3(b) already asks
  about a model-independent teacher; **the competition record backs that question**, and the cheapest
  legal instance is co-training on the SAR-only / optical-only view split, since those two views are
  genuinely different measurement physics.
- **LEGALITY (Â§5):** (a) pseudo-labelling never moves the decision threshold. (b) The one knob that
  *must* be watched is the confidence cut / how many rows to label: Santander's 2700-and-2000 was
  hand-set, and under Â§5(b) we cannot pick such a number by LB. A **soft** label has no such knob at
  all â€” another reason our existing soft variant is the right one. (c) it changes the target as a
  function of x, which is the brief's class E3.
- **THEOREMS:** the target on pseudo-labelled rows is `p_teacher(x)`, a function of x, not of a fixed
  label y. T2's premise (`sum_i y_i*l1(z_i) + (1-y_i)*l0(z_i)` with y in {0,1} and fixed l1,l0) does
  not hold. **SURVIVES T2** â€” this is precisely why it is the one family in our history that has ever
  worked. Not an affine logit map: **SURVIVES T1**.


---
---

# ⛔ BREAKING CORRECTION — inserted 2026-08-16, AFTER findings F1–F7 were written

**Everything above this line was written against a public confusion cell that is now REFUTED.**
I am not silently patching F1–F7. They stay as written. This block states what changed and which
of my own sentences are void.

**What was refuted.** `n_public = 309` was never measured — it was inferred as 30% of 1030. Finite-
sample ROC-AUC is exactly `C/(P·N)` with `C` a half-integer, so a printed 9-decimal AUC is a
rational with a known denominator. Sieving all five reported (AUC, F1) pairs jointly, plus the
full-test predicted-positive counts on disk, solves the cell exactly
(`tools/lb_cell_solve.py`, run and read by me):

```
                       OLD (carried since iter42)      NEW (solved exactly)
  n_public                    309                            333
  P (true positives in public) 191                           181
  N                           118                            152
  champion cell         TP 164  FP 17  FN 27  TN 101   TP 164  FP 27  FN 17  TN 125
  precision                 0.9061                         0.858639
  recall                    0.8586                         0.906077
  true public prevalence    0.618                          0.5435
  our realised pos-rate     0.618                          0.5736
```

**PRECISION AND RECALL ARE SWAPPED.** TP and PP are invariant across the whole surviving
`P = 181·s, PP = 191·s` family, so the swap does not depend on `n = 333` being exactly right.

**The strategic consequence, and it inverts my own priorities above.**
Our dominant error mode is **FALSE POSITIVES, 27 against 17 misses, 1.6:1**. We **OVER-predict**:
realised public positive rate **0.5736 vs a true 0.5435**. We do not have a recall deficit.

**My own sentences that are now VOID:**
- F1, bullet on item 4: "*it targets precisely the failure mode the brief describes in §3.4:
  confidently-wrong positives (their `y=1 and p<0.2` is our missed positives with median score
  0.170)*." — The hardness head is still legal and still survives both theorems, but the half of
  its definition that matters for us is now the **other** half: `hard = 1 if (y=0 and p>0.8)`,
  the confidently-wrong **negatives**. The finding survives; its **direction** flips.
- F7, "*The brief's §3.4 says the positives we miss have median score 0.170 ... our errors are
  exactly the confidently-wrong kind*." — the §3.4 measurement was made on labelled held-out
  data, which does NOT reproduce the deployment shift and does NOT know about the 27 FPs. The
  conclusion "do not run round two of self-distillation" **stands** (it rests on Arazo et al.,
  not on the cell), but its supporting sentence must be re-stated in terms of confidently-wrong
  **negatives** under prior shift. See F12 below.
- Any reading of F2/F3/F4 as "recover more positives" is void. The augmentation and
  feature-transfer arguments in F2/F3 are direction-neutral and survive unchanged.
- The brief's own §3.2 ("*they find 16 more true positives for 4 more false positives ... the
  high-recall corner*") and §7 Q1 ("*optimise the HIGH-RECALL REGION*") are aimed at the smaller
  half of our error budget. **Partial-AUC restricted to the high-TPR band is now the wrong band.**
  If a pAUC lane is pursued at all it must be the **high-specificity / low-FPR** band.

**Resuming at F8 with the corrected target: FALSE POSITIVES at a fixed 0.5 cut under prior shift.**

---

### F8 — THE CORRECTED TARGET, PRICED EXACTLY. **Own derivation. VERIFIED (I ran `tools/lb_cell_solve.py` and the arithmetic below myself).**
**Before any literature: the corrected cell lets us price a false positive, and it turns out our own leaderboard history already CONFIRMS the corrected direction. This is the single most report-grade item in my file.**

**The identity that does the work.** With `P` fixed (the number of true positives in the scored
set is a property of the data, not of us) and `TP = P - FN`:

```
F1 = 2·TP / (2·TP + FP + FN) = 1 - (FP + FN) / (2P + FP - FN)
```

F1 depends on the prediction **only through the pair (FP, FN)**. At `P = 181`, `FP = 27`,
`FN = 17` this returns 0.881720430 — the reported champion F1 to all nine digits. The
identity is therefore exact, not a model.

**MARGINAL PRICES AT OUR OPERATING POINT (public set, n=333):**

| move | new F1 | ΔF1 | Δcomposite (×0.6) |
|---|---|---|---|
| remove **1 false positive** | 0.884097 | **+0.00238** | **+0.00143** |
| remove **1 false negative** | 0.884718 | +0.00300 | +0.00180 |

A missed positive is worth 1.26× a false alarm *per unit* — but **we hold 27 false alarms and
only 17 misses**, so the FP pool is 1.6× larger and, unlike the FN pool, it is not bounded by
how many positives exist. The available F1 from clearing the whole FP pool is +0.066; from
clearing the whole FN pool, +0.051.

**WHAT THE +0.015 COMPOSITE SIGNIFICANCE BAR ACTUALLY COSTS** (this is the number to carry):

```
   FPs removed :  2      4      6      8     10     12     14     16     20
   F1          : .8865  .8913  .8962  .9011  .9061  .9111  .9162  .9213  .9318
   Δcomposite  : +.0029 +.0058 +.0087 +.0116 +.0146 +.0176 +.0207 +.0238 +.0301
```

> **We must eliminate ~10 of our 27 public false positives — 37% of them — with zero collateral
> damage, to clear the +0.015 bar.** Scaled to the full 1030-row test set that is **≈31 rows**.
> Compare the brief's §3.4 estimate that "we need roughly 23–33 rows to change side": that figure
> was computed for the wrong sign and is, by coincidence of the arithmetic, about right in
> magnitude. **The rows are the same count; they are the opposite class.**

**THE EXCHANGE RATE — how much collateral damage a precision fix may cause before it loses.**
A method that suppresses false alarms will also push some true positives below the cut. Solving
`F1(27-k, 17+j) ≥ F1(27,17)`:

| FPs removed k | new FNs tolerated j | ratio |
|---|---|---|
| 5 | 3 | 0.60 |
| 10 | 7 | 0.70 |
| 15 | 11 | 0.73 |
| 20 | 15 | 0.75 |

> **Break-even is ~0.7 lost positives per suppressed false positive**, and it *rises* as you go
> — the metric is forgiving here. Any precision intervention with a better-than-0.7 hit ratio is
> net positive on F1. That is a much weaker requirement than "zero collateral damage".

**WHERE THE LEADER'S ADVANTAGE ACTUALLY IS — and the brief's §3.2 story is void.**
F1 pins only the line `FP ≈ 32.2 − 1.178·FN`, not a point, so the leader's exact cell is not
identified. But holding one axis at ours:

| leader composite | their F1 | needs FP ≤ ... at our FN=17 | or FN ≤ ... at our FP=27 |
|---|---|---|---|
| 0.929 | 0.918402 | **12 (−15 FPs)** | 4 (−13 FNs) |
| 0.930 | 0.920069 | **11 (−16)** | 3 (−14) |
| 0.936 | 0.930069 | **7 (−20)** | **0 (−17, i.e. a perfect-recall model)** |

At the top of their range the pure-recall route requires **FN = 0**, which is not a real solution.
Their total error budget `FP+FN` is ~28–32 against our **44** at F1 0.918, and ~24–27 at F1 0.930.
**The leader is not sitting in the high-recall corner. They are simply making ~14 fewer errors,
and on the corrected cell the larger share of ours is false alarms.** UPDATE_24 §3.2's
"they find 16 more true positives for 4 more false positives" describes a cell that exists in
the F1-inversion family only at `recall ≥ 0.99`, and it was selected because the old (wrong)
`P = 191` made it look natural. **Retract it.**

**⭐ OUR OWN LEADERBOARD ALREADY CONFIRMS THE CORRECTED DIRECTION — two independent instances.**
Re-read the solved cells for the other submissions on disk:

```
  champion       TP=164 FP=27 FN=17  F1 = 0.881720430
  jtt_lam5       TP=164 FP=26 FN=17  F1 = 0.884097035   <- identical TP, ONE fewer FP
  archblend4     TP=163 FP=25 FN=18  F1 = 0.883468834   <- TWO fewer FP, ONE lost TP
```

- `f1(FP=26, FN=17) = 0.884097` reproduces the reported jtt_lam5 F1 **to all nine digits.**
- **This is a correction to our round-23 closure.** The commit message says "*iter49 LB RESULT:
  JTT recovered ZERO true positives. Round 23 is finished.*" That is literally true and was the
  right call **for the goal it was measured against** — but JTT was not null. It removed exactly
  **one false positive** and gained **+0.002376 F1**, which is +0.00143 composite: real, correctly
  signed, and simply an order of magnitude too small to see. We graded a precision effect as a
  failure because we were scoring it on recall.
- `archblend4` is the **only artifact we own with a higher F1 than the champion**, and it got
  there by trading 2 FPs for 1 TP — precisely the 0.5-below-break-even exchange the table above
  says is profitable. It was passed over because its *composite* is lower (its AUC is worse).
- **Both of our two best F1 artifacts reached their F1 by cutting false positives, and neither
  gained a single true positive.** Two out of two. This is weak evidence (n=2, moves of 1–2 rows)
  but it is *our own measured leaderboard evidence*, it is correctly signed, and it is the only
  in-domain evidence that exists.

**LEGALITY:** diagnosis only. Every number here is arithmetic on figures Zindi already printed
plus prediction files on disk. It sets no threshold, no hyperparameter, no model choice — same
category as the F1 inversion disclosed since iter42. **(a)** untouched, **(b)** nothing here may
be used to set a knob and I am not proposing that it is, **(c)** n/a.
**THEOREMS:** not a method; T1/T2 do not apply. It is the *scoreboard* against which candidate
methods below are priced.

---

### F9 — ⭐ THE CONFUSER MECHANISM IS DOCUMENTED, AND OUR BEST FEATURE IS THE THING THAT GENERATES OUR FALSE POSITIVES. **VERIFIED** (read the full paper text through the r.jina.ai proxy)

**Primary source:** Ottinger, Clauss, Kuenzer, **"Assessment of Coastal Aquaculture for India from Sentinel-1 SAR Time Series", Remote Sensing 11(3):357, 2019, DOI 10.3390/rs11030357** — https://www.mdpi.com/2072-4292/11/3/357
This is the **same author group the brief already cites** (§8 item 2, Ottinger et al. IGARSS 2018) for the canonical "VH alone, temporal median" pipeline. This paper *is* that pipeline, applied at national scale, with a published error analysis. Method: temporal median of VH sigma-nought over 2983 descending-mode scenes, Otsu land/water threshold, connected-component segmentation, shape filters.

**THE FOUR VERBATIM QUOTES THAT MATTER. All read directly from the paper.**

1. **The discriminant that defines the whole field, and it is PERMANENCE:**
   > "Aquaculture ponds can be generally described as **permanent water bodies**, in contrast to,
   > e.g., temporarily inundated rice paddy fields."

   > "While aquaculture and rice paddy fields often feature **the same form factors**, the former
   > appear **much darker in the median image due to their persistent water surface throughout the
   > year**."

2. **⛔ THE FALSE-POSITIVE MECHANISM, named explicitly:**
   > "Another problem was the **confusion of aquaculture with flat features such as salt pans,
   > which appear as dark features in the SAR image**."

   > "flat surfaces such as **salt panes, roads, or abandoned fields or ponds resulted in the same
   > dark surfaces as water ponds**."

   > "Another issue was the confusion of aquaculture with **inundated rice paddies or other flat
   > features such as salt pans, which is one of the main sources of uncertainty** to the approach
   > in agricultural dominated coastal areas in Asia."

3. **✅ THE REMEDY THEY PRESCRIBE — optical, and specifically framed as a FALSE-POSITIVE fix:**
   > "The additional **multispectral information from Sentinel-2 can help to exclude false
   > positives such as salt pans, flat areas of bare soil, backscatter shadows, etc., by
   > incorporating the spectral reflectance characteristics of water bodies**."

   (Their Figure 11 contrasts the S1 median-backscatter layer against a Sentinel-2 **NDWI** layer
   for exactly this purpose.)

4. **THE FALSE-NEGATIVE MECHANISM, also named, and it is the mirror image:**
   > "in India, traditional aquaculture systems in coastal regions **alternate between rice
   > cultivation and shrimp breeding** ... Since the **persistence of inundation throughout the year
   > is a main feature** in the applied approach to differentiate aquaculture from rice fields,
   > **those aquaculture systems were not identified here**."

**WHAT THIS SAYS ABOUT US — the most important non-arithmetic item in this file.**

Our two hand-built features are `1[VH_dB < -21]` (a **SAR permanence** channel, +0.010, our best
single feature) and `1[VH<-21]·1[(VH-VV)<-8]`. Both are **darkness-persistence detectors**. The
primary literature for exactly this pipeline states that a darkness-persistence detector's **named
failure mode is commission error on salt pans, bare flats, roads, abandoned ponds and SAR
shadow** — every one of which is dark in VH and permanently so.

> **We built a feature whose documented error mode is false positives, and we have just discovered
> that false positives are our dominant error, 27 against 17. These are the same fact seen from
> two directions.**

And the prescribed remedy is not "weaken the SAR feature". It is a **conjunction with an optical
water-reflectance criterion**: a salt pan is dark in VH *and* is not water in Sentinel-2; a pond is
dark in VH *and* is water in Sentinel-2. That conjunction is a **threshold interaction** — the one
feature family that has ever won for us (§6: *"what is outside our model's reach is not the band
index but the nonlinear temporal statistic ... a threshold indicator, not a ratio"*) — and it is
**not** what §6 closed. §6 closed *linear optical indices* on the grounds that they are ≥97%
linearly reachable from raw bands **at the per-month level**. A product of two indicator functions
of two different sensors, aggregated over time, is not linearly reachable from anything. **The
closure does not cover this object.**

> **The concrete untested feature, stated so it can be built:**
> `salt_pan_veto = (fraction of OBSERVED months with VH_dB < -21) × (1 − fraction of OBSERVED
> months with NDWI > 0)`
> — high exactly for permanently-dark, never-water locations, i.e. the documented confuser.
> `NDWI = (green − nir)/(green + nir)` (McFeeters 1996); both bands are among our 12. The
> aquaculture literature also favours `MNDWI = (green − swir1)/(green + swir1)` for turbid or
> shallow water — `swir1` is also among our 12. Feed both counts *and* the product; our history
> says the explicit indicator beats letting the network find it.

**⚠️ THE SHIFT MAKES THIS DIRECTIONAL — a mechanism for why we over-predict on test but not train.**

Permanence is a statement about **all 12 months**. Train rows have all 12; **test rows have 4–6
contiguous months**. A rice paddy observed only during its flooded window is *indistinguishable
from a permanent pond* on a permanence feature, because the evidence that would refute permanence
has been deleted by the observation process. **The designed truncation systematically converts
negatives into apparent positives, and it cannot convert positives into apparent negatives by the
same route.** That is a *directional* corruption and its direction is exactly the over-prediction
we measured (realised 0.5736 against a true 0.5435).

Nothing in UPDATE_24 states this. §4 treats the shift as a symmetric nuisance and §3 treats the
prior gap as something to be estimated. **If this mechanism is real, the "test prior is ~0.59–0.62"
belief is itself partly an artifact of our own over-prediction**, and the true test prior is closer
to the 0.5435 we now measure on the public slice.

> **FALSIFIABLE, TRAIN-ONLY, NO TRAINING RUN, minutes of pandas — run this before anything else:**
> On labelled train rows compute the permanence feature on (i) the full 12 months and (ii) each
> masked 4–6-month crop. Compare `E[feature | y=0]` and `E[feature | y=1]` under both.
> - **If cropping raises the permanence feature on NEGATIVES more than on POSITIVES**, the
>   mechanism is confirmed, the direction of our LB error is explained, and the salt-pan veto has
>   a target to hit.
> - **If it does not**, F9's causal story is dead and I want that recorded as a negative — it
>   would mean our masked-window augmentation already neutralises the confuser, which is itself
>   a useful, citable result for the report.
> - Either outcome **re-grades a closure**: the K=2 masked-window augmentation exists precisely to
>   inoculate against this. If the gap persists in the *augmented* training distribution, the
>   augmentation is under-powered (F2/PLAsTiCC used **up to 40** degraded views; we use **2**) or
>   mis-specified in its joint window distribution (F2 point 3).

**LEGALITY (§5):** **(a)** the −21 dB and NDWI>0 cuts are thresholds **on x** — properties of the
input, not the decision rule. The classifier still cuts at a literal 0.5 on the Platt-calibrated
probability. This is the *identical* legal structure to the `1[VH_dB<-21]` channel the team already
ships and has already disclosed. **(b)** both constants come from published remote-sensing
literature and from train data; NDWI>0 is the canonical McFeeters (1996) water cut, not a fitted
parameter; neither is set from a positive-rate target or from LB feedback. **(c)** it adds a
genuinely new nonlinear function of x to the hypothesis class, so p(y|x) is re-estimated rather
than relabelled. **PASSES all three prongs.**

**THEOREMS:** feature construction changes the **input space**. T1 governs affine maps of a *fixed*
logit — irrelevant here. T2 governs a reweighted pointwise loss over a *fixed* sample of x — here x
itself changes, so the population minimiser is a monotone function of a **different** η and the
ranking genuinely moves. **SURVIVES T1 AND T2.**

**COST AND HONEST EXPECTED VALUE.** Two new channels, one training run of the existing 10-seed
pool. §6 grades "adding channels/width" as a repeated loser (soft) — but it also records that the
**one** channel that won was a threshold indicator of exactly this type. Priced against F8: we need
**≈10 of 27 public false positives** for +0.015 composite, tolerating up to ~7 newly-lost positives.
I will not fake a point estimate. **Run the diagnostic first; it decides whether the feature can
possibly help before any GPU time is spent.**

**WHAT I COULD NOT VERIFY.** The paper reports overall accuracy above 80% but publishes **no
per-confuser commission breakdown**. There is no number in it for "salt pans caused X% of
commission error". I looked; it is not there. Anyone quoting such a number is inventing it.

**F9 ADDENDUM — the shift mechanism is stated in the literature almost verbatim. VERIFIED.**
Second primary source, same group: Ottinger, Clauss, Kuenzer, **"Large-Scale Assessment of Coastal
Aquaculture Ponds with Sentinel-1 Time Series Data", Remote Sensing 9(5):440, 2017,
DOI 10.3390/rs9050440** — https://www.mdpi.com/2072-4292/9/5/440 (read in full through the proxy).
Reported overall accuracies: Mekong 0.83, Red River 0.84, Pearl River 0.88, Yellow River 0.80.

> "This is a major issue for many coastal areas and specifically river deltas, where **floodplains
> or paddy rice fields might be confused with aquaculture if the temporal resolution of the time
> series is inadequate to depict hydrological characteristics and seasonality** of land cover other
> than aquaculture."

**That sentence is our competition's designed difficulty, written down by the domain authors four
years before the competition existed.** The organisers truncated test rows to 4–6 contiguous months.
The literature says that truncation's specific consequence is **confusion of rice paddies AS
aquaculture** — a commission error, a false positive. Our corrected cell says false positives are
our dominant error. Independent prediction, independent confirmation, same sign.

Also verified from the same paper, and relevant to the confuser inventory: farmers in these deltas
**switch between rice and shrimp** and operate hybrid **rice–shrimp / rice–fish** systems. So the
confuser is not merely a look-alike class — some locations genuinely alternate, which caps the
achievable precision and is a candidate explanation for irreducible label noise (Q4).

---

### F10 — ⭐⭐ HARD-NEGATIVE MINING: T2 EXTENDS TO INSTANCE REWEIGHTING, AND THE EXTENSION EXPLAINS OUR OWN JTT RESULT. **Own derivation, VERIFIED against our own measured leaderboard cells.**
**This is a correction to the brief's statement of its own theorem, a correction to our round-23 reading of iter49, and it settles task item (a) — hard-negative mining — on theory rather than on anecdote.**

**THE DERIVATION.** Take any weighted pointwise objective

```
    L(f) = E[ w(x, y) · l_y( f(x) ) ]
```

with `w > 0` and `l_1, l_0` a fixed proper composite pair (log-loss, exponential, etc.). Condition
on `x` and write `η = p(y=1|x)`, `w_1 = w(x,1)`, `w_0 = w(x,0)`. The conditional risk is

```
    η·w_1·l_1(z)  +  (1-η)·w_0·l_0(z)
```

Dividing by the positive constant `η·w_1 + (1-η)·w_0` leaves the minimiser unchanged, so the
minimiser is `T(η')` — the **same fixed link `T`** as in T2 — evaluated at the **tilted posterior**

```
                        η · w_1(x)
    η'(x)  =  ───────────────────────────────
              η · w_1(x)  +  (1-η) · w_0(x)
```

**Three corollaries, and they partition the whole reweighting family:**

| what the weight depends on | effect on `η'` | ranking / AUC |
|---|---|---|
| **`y` only** (class weighting, balanced sampling, over/under-sampling) | `w_1/w_0 = c`, so `η' = cη/(cη+1-η)`, a **strictly increasing** function of `η` | **UNCHANGED.** Confirms the brief. |
| **`x` only** (curriculum weighting, per-row importance weights, difficulty weighting applied to both classes alike) | `w_1 = w_0` cancels entirely, `η' ≡ η` | **UNCHANGED — and this is stronger than what the brief states.** |
| **`x` AND `y` jointly** (hard-negative mining, JTT, per-class-per-region weights) | `w_1/w_0` varies with `x`, so `η'` is **not** a monotone function of `η` | **GENUINELY REORDERS.** |

> **⚠️ CORRECTION TO THE BRIEF.** UPDATE_24 §5 states T2 for "`Σ y·l₁(z) + (1-y)·l₀(z)` — one fixed
> pair of functions applied to every example". The middle row above shows the theorem is
> **strictly stronger than stated**: it also annihilates every **x-dependent instance weight that is
> class-symmetric**. That is a large family the brief leaves ambiguous — importance weighting,
> curriculum learning, difficulty-based sampling, sample-selection schemes, and "train longer on
> the hard rows". All of them are order-invariant in the population limit. The brief should state
> T2 in the weighted form; it costs one line and closes a whole family for free.

**⛔ CONSEQUENCE FOR TASK ITEM (a), HARD-NEGATIVE MINING — it is mostly dead, and precisely why.**
"Mine hard negatives and upweight them" or "mine hard negatives and resample them more often"
(resampling ≡ reweighting in expectation) is a `w(x,y)` scheme, so it lands in the third row and
**does** reorder. But note what it does *not* do: it does not add information. It re-tilts a
posterior the model already estimated. **The reorder it produces is a deterministic monotone
rearrangement within each weighting region**, which is a very restricted class of change — and
below I show our own leaderboard already measured its magnitude.

**WHAT DOES SURVIVE, and this is the useful half:**
- **Hard negatives inside a PAIRWISE or SET-LEVEL loss** (contrastive, triplet, InfoNCE, ArcFace-style
  metric learning, listwise ranking). These are not of the form `Σ_i w_i l_{y_i}(z_i)` at all — the
  loss couples examples — so **no version of T2 applies.** This is the brief's class E1/E2.
  **It is also, empirically, where every documented competition win by hard-negative mining lives:**
  retrieval and detection (OHEM, Shrivastava et al., CVPR 2016, arXiv:1604.03540; Google Landmark
  2019, lyakaap, arXiv:1906.04087, whose contribution is an automated **data-cleaning** system plus
  discriminative **re-ranking**, both set-level operations). **I searched specifically for a
  documented win by hard-negative reweighting in a decomposable binary tabular classification task
  and did not find one.** I now believe that is not a gap in my search: it is what the middle and
  third rows of the table predict. **Report this as a theory-backed negative.**
- **SYNTHESISING new negatives** (as opposed to reweighting existing ones) changes the support of
  the sample, not the weights, so it escapes entirely. See F11.
- **A second head predicting "is this a confuser"** — an x-dependent auxiliary target, class E3,
  the F1/ICR mechanism. Escapes because the objective is no longer a function of one logit.

---

**⭐ THE PART THAT CORRECTS US: JTT did not fail, and the theory predicts exactly what it did.**

Our commit reads *"iter49 LB RESULT: JTT recovered ZERO true positives. Round 23 is finished."*
That is literally true and it was the honest call **for the target it was scored against**. Under
the corrected cell it is the wrong scoreboard. Put JTT in the framework above.

JTT upweights the **error set** of a first-round model: weight `1+λ` on examples the first model got
wrong. So `w_1(x) = 1 + λ·1[ŷ(x)=0]` and `w_0(x) = 1 + λ·1[ŷ(x)=1]` — the third row, jointly
x-and-y-dependent. Substituting:

```
  region we currently predict POSITIVE (ŷ=1):   η' = η / (η + (1-η)(1+λ))   <  η   → scores PUSHED DOWN
  region we currently predict NEGATIVE (ŷ=0):   η' = η(1+λ) / (η(1+λ) + 1-η) >  η   → scores PUSHED UP
```

> **JTT's population effect is a two-block compression across the old cut. In the upper block its
> job is to SUPPRESS FALSE POSITIVES; in the lower block, to RECOVER FALSE NEGATIVES.** We scored
> it only on the second job.

**And it did the first job. Measured, on our own leaderboard, from the solved cells (F8):**

```
  champion   TP=164  FP=27  FN=17   F1 = 0.881720430
  jtt_lam5   TP=164  FP=26  FN=17   F1 = 0.884097035     ← one false positive removed, zero TPs lost
```

`F1(FP=26, FN=17) = 0.884097` reproduces the reported jtt_lam5 F1 **to all nine digits**. JTT
removed **exactly one false positive and cost nothing**: +0.002376 F1, +0.00143 composite. That is
**correctly signed, exactly as the derivation predicts, and simply an order of magnitude too small
to clear a ±0.015 noise bar.** We recorded a null because we were counting the wrong class.

**Why it was so small, and what the derivation says to do about it.** JTT's two blocks fight each
other: pushing the lower block up manufactures new false positives at the same time as the upper
block sheds them. Under our corrected cell those two effects are **not** of equal value to us —
we hold 27 FPs and 17 FNs. The targeted variant is therefore:

> **UPWEIGHT ONLY THE `y=0, ŷ=1` HALF OF THE ERROR SET** (confidently-wrong negatives — the
> confusers), leaving `y=1, ŷ=0` at weight 1. Then `w_1 ≡ 1`, `w_0 = 1 + λ·1[ŷ(x)=1]`, giving a
> **one-sided** rearrangement: scores fall in the currently-positive region and are untouched
> elsewhere. No manufactured false positives, no fight.

**🔴 LEGALITY OF THAT VARIANT — and I am flagging a trap, not endorsing a shortcut.**
Choosing the FP half **because our leaderboard cell says FP dominates is a §5(b) VIOLATION.** It
routes an LB-derived quantity into a training knob. That is exactly the class of move the team
deleted scores over, and it must not be done. **It is legal only if the asymmetry is fixed by a
train-only criterion**, and there is one available: the **F9 diagnostic**. If, on labelled train
rows, masking to a 4–6-month window inflates the permanence evidence on **negatives** more than on
positives, that is a train-only, label-only measurement establishing that the deployment
observation process is commission-biased — and the one-sided weighting follows from it without
any reference to the leaderboard. **Run F9's diagnostic first, or do not run this at all.**
- λ must likewise be train-only. JTT's own paper (Liu et al., *"Just Train Twice"*, ICML 2021,
  arXiv:2107.09044) selects λ on a worst-group validation set; we have no such set and must not
  invent one from the LB. A fixed λ carried over from iter49 is the defensible choice.

**THEOREMS:** the one-sided weighting is the **third row** — `w_1/w_0` varies with x — so it is
**not** annihilated by T2, and it is not an affine logit map, so it **survives T1**. It is, however,
the *weakest* of the surviving families: it re-tilts an existing posterior rather than adding
information, and our own measurement of its full-strength two-sided version was one row.
**PRONGS:** (a) 0.5 cut untouched. (b) **conditional** — legal only via the F9 train-only route
above. (c) it changes the fitted `p(y|x)`, it does not relabel a fixed estimate.

**MY RANKING, honestly.** Given that the two-sided version measured **one row**, I would expect the
one-sided version to measure **two to four rows** — i.e. +0.003 to +0.006 composite, **below the
+0.015 bar.** I am recording it because the derivation is report-grade and because it converts an
unexplained null into a predicted one, **not** because I think it wins on deadline day.

---

### F11 — ⭐ ZINDI, F1 METRIC, BINARY SUBMISSION: **Landslide Prevention and Innovation Challenge — 1st place.** VERIFIED (I downloaded and read the winning notebook's source, cell by cell)
**This closes the biggest gap in my file: a Zindi competition scored on F1 with a binary submission, where I can read exactly how the winner set the operating point. The answer is uncomfortable and I am reporting it straight.**

- **Competition:** https://zindi.world/competitions/landslide-prevention-and-innovation-challenge
  (VERIFIED from the competition page: metric is **F1**, submission is a **binary 0/1 label**
  column, public LB = 20% of test / private = 80%.) Terrain-based landslide identification in
  Hong Kong from **tabular geospatial features** — elevation, slope, aspect, TWI, LS-factor,
  plan/profile curvature, SDOIF, geology — supplied as **25 spatial neighbours per row**
  (`1_elevation` … `25_elevation`, etc.). Structurally this is our data shape with *space* where
  we have *time*.
- **Winning solution, organiser-curated:**
  `github.com/ZindiAfrica/Machine-Learning/Classification Analysis Challenges/Landslide Prevention and Innovation Challenge/#1 place /LandslidePrediction_WinningSolution.ipynb`
  (fetched via `gh api`; 36 cells, all read).

**WHAT THE WINNER ACTUALLY DID — verbatim from the source.**

1. **Feature engineering is entirely NONLINEAR AGGREGATE STATISTICS over the neighbour axis.**
   For each of 8 base variables, over the full 25-neighbour set *and* over five 5-neighbour
   sub-blocks:
   ```python
   df[i+"_mean"], df[i+"_median"], df[i+"_min"], df[i+"_max"], df[i+"_std"],
   df[i+"_range"] = max - min,  df[i+"_ratio1"] = max/min,  df[i+"_ratio2"] = min/max
   ```
   `median`, `min`, `max`, `range` are exactly the class UPDATE_24 §6 identifies as outside a
   linear model's reach. Note also the **sub-block decomposition** (5 blocks of 5): they did not
   only aggregate globally, they aggregated over *contiguous sub-windows* of the ordered axis.
   **Our direct analogue is per-sub-window temporal statistics over the 4–6 observed months**, and
   we compute nothing of the kind that I can see.
2. **Three GBDTs (CatBoost, LightGBM, XGBoost), each fit twice, with hand-picked disjoint feature
   lists per model** (~100 features each, different per model).
3. **⛔ THE OPERATING POINT IS SET BY `scale_pos_weight`, AND BY NOTHING ELSE.**
   ```python
   for scalePos in [3.8, 4]:   cat_params = {..., "scale_pos_weight": scalePos, "eval_metric": 'F1', ...}
   for scalePos in [3.9, 4]:   lgb_params = {..., "scale_pos_weight": scalePos, ...}
   for scalePos in [3.8, 4]:   xgb  = XGBClassifier(..., scale_pos_weight = scalePos, ...)
   ```
4. **The final decision is a HARD MAJORITY VOTE at a literal 0.5.** Every member calls
   `model.predict(...)` — **hard 0/1, not `predict_proba`** — the six hard votes are summed and:
   ```python
   predBlend = (predsXGB + predsCatBoost + predsLGB) / 6
   preds = [1 if x >= 0.5 else 0 for x in predBlend]
   ```
   **There is no probability threshold search anywhere in the notebook.** No OOF threshold sweep,
   no percentile matching, no post-hoc calibration. `SEED = 42`, `seed_everything`, fully
   reproducible.

**THE HONEST READ-ACROSS, and it cuts against us.**

> A Zindi 1st place on an F1 metric moved its operating point **entirely through
> `scale_pos_weight ≈ 4`** and then cut at a literal 0.5. **By T2 that is provably a pure
> threshold slide on an unchanged ranking.** The winner did not out-model the field on
> `p(y|x)`; they out-*positioned* it, using the one knob that our three-prong test forbids.

Check `scale_pos_weight ≈ 4` against our own prongs:
- **(a) PASSES** — the decision rule really is a literal 0.5, on a hard vote.
- **(b) PASSES** — `N_neg/N_pos` is the textbook train-only balancing default; the winner used two
  nearby values and averaged, which is not an LB search. **This is the trap: a knob can be
  entirely train-only and still be a threshold slide.**
- **(c) FAILS** — T2 proves it does not correct `p(y|x)`; the population minimiser is `T(η')` with
  `η' = cη/(cη+1−η)`, a strictly increasing function of `η`, so the ranking is untouched.

**Our three-prong test therefore correctly rejects the winning move of a comparable Zindi
competition.** I think the team should say that out loud in the report rather than around it: our
self-imposed rule is stricter than the competition rule, we know what it costs, and prong (c) is
the prong doing the work. That is a much stronger report position than implying nobody does this.

**⚠️ AND A DIRECTIONAL WARNING FOR US.** If we *were* to adopt balanced class weighting — the
"principled train-only default" — our train prior is 0.4023, so positives are the minority and
`w_pos = 0.5977/0.4023 = 1.486`. **That slides us toward predicting MORE positives, which is
exactly the wrong direction: we already over-predict (0.5736 realised vs 0.5435 true) and false
positives are already our dominant error.** I checked our code: `src/seq_model.py` uses a plain
`nn.BCEWithLogitsLoss()` with **no `pos_weight`**, and the `scale_pos_weight` in `src/models.py`
belongs to the closed GBDT lane and defaults to 1.0. **We are clean here — this is a
non-finding, and it is worth recording as a non-finding so nobody "fixes" it later.**

**THE THREE THINGS HERE THAT ARE ACTUALLY USEFUL TO US:**
1. **Sub-window aggregate statistics.** The winner aggregated over five contiguous 5-element
   sub-blocks *in addition to* the whole axis. On our 4–6 observed months the analogue is
   `min/max/median/range` of each band over the first half vs the second half of the observed
   window — a **trend/permanence contrast** that is precisely the statistic that separates a
   permanent pond from a seasonally-flooded paddy (F9), computable inside a 4-month window, and
   not linearly reachable. This is the strongest *feature* idea I have that does not require the
   optical channel argument of F9. **Survives T1 and T2** (input-space change). **PASSES all three
   prongs** (statistics of x; train-only; new hypothesis class).
2. **Different disjoint feature subsets per ensemble member.** Corroborates F6/Kiminya's diversity
   axis and further weakens §6's "blending a weaker decorrelated member" closure — again, these
   members are *equally strong*, not weak.
3. **Hard majority voting at 0.5 is what a Zindi F1 winner shipped.** §6 grades hard majority
   voting FIRM-closed for us ("moved 4–7 rows; dead even with post-hoc k"). Our measurement stands
   — but it is worth one line in the report that the closure is about *our* pool's diversity, not
   about the technique, since a 1st place used exactly it.

**WHAT I COULD NOT VERIFY.** The competition page does not publish row counts or the positive
rate, and the notebook loads from a private Drive path, so I cannot state the class balance or
confirm that `scale_pos_weight ≈ 4` equals `N_neg/N_pos`. I am inferring that from the value being
a round ~4 and from three independent models converging on it; **mark that inference INFERRED.**
The page also states nothing about a train/test regional shift, so **this competition is
INAPPLICABLE on the distribution-shift axis** — do not cite it for that.
