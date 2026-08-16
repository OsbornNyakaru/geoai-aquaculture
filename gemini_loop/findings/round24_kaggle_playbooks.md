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

