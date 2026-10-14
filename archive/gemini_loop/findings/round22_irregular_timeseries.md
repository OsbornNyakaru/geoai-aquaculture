# Round 22 — Irregular / Partially-Observed Time Series: REFUTATION BRIEF

**Target claim under attack:** "Attention masking plus relative-time encoding already solves the
12-month-train / 4-6-month-test mismatch; there is nothing further to gain from the irregular-time-series
literature."

**Context:** Zindi/FAO/ITU GeoAI Aquaculture Pond ID. n=1817 train (12 months fully observed),
1030 test (4-6 CONTIGUOUS months). 12 bands. Metric 0.6*F1(0.5 cut) + 0.4*AUC.
Best public 0.907369. Platt Annihilation Theorem binding: only RANKING changes score.

## STATUS: COMPLETE

---

## 0. VERDICT UP FRONT

The claim is **HALF RIGHT, AND THE HALF THAT IS RIGHT IS THE HALF YOU CARED ABOUT LEAST.**

- **On ARCHITECTURE SWAPS the claim survives.** I could not refute it and I will not pretend otherwise.
  Every canonical irregular-time-series (ISTS) architecture — mTAND, GRU-D, SeFT, Raindrop, Neural CDE,
  Latent ODE, ContiFormer, Rough Transformers, Warpformer — is built for **SCATTERED** missingness with
  real-valued, unaligned timestamps. Not one is designed for or benchmarked on a **CONTIGUOUS BLOCK**
  window whose length and offset shift between train and test. Their machinery buys you a principled
  treatment of arbitrary inter-observation gaps; your gaps are always exactly one month. Swapping
  architecture is negative expected value at 25 min/run with 2-3 slots. **Lane closed.**
- **A theorem actively vindicates your current design.** Zhou, Balakrishnan & Lipton (AISTATS 2023):
  *"If missing data indicators are available, DAMS reduces to covariate shift."* You have the indicators.
  So there is no missing-data machinery you are lacking. Correct.
- **I then tried to refute the claim on the augmentation axis, built the argument, and then killed it
  myself by reading your code and your log.** Two independent literatures (group-theoretic orbit
  averaging; covariate-shift matching) converge on "your masked-view training covers only 2 of the 24
  legal windows, frozen". That is **true of the champion config** (`resample_per_epoch: false`, and
  hard-gated to `lam == 0.0` at `src/seq_model.py:981`). **But `LB_LOG.md:1495` shows iter21 already ran
  exactly this and found it inert with a source-overfitting fingerprint.** Both self-corrections are kept
  in full in Section 4 rather than deleted, because the *reasoning* is what tells you the lane is closed
  for a principled reason and not just empirically.
- **What genuinely survives:** **TTA aggregation** (Shanmugam et al., ICCV 2021) is one of the very few
  levers that provably **survives Platt annihilation** — and you already have the on-manifold sub-window
  machinery built (`_test_views`) but wired only to a consistency penalty, not to a prediction average.
  **It needs no retrain, so it costs no slot.** And **length-generalization positional-encoding theory**
  (Kazemnejad et al., NeurIPS 2023) *explains* your +0.013 and yields a cheap diagnostic variant.

**Bottom line: the honest recommendation is to spend ZERO slots on this literature, run P2 offline for
free, and treat Section 7's unified theory as the durable output.** Rank order in Section 8.

---

## 1. ARCHITECTURE SURVEY — BLOCK vs SCATTERED missingness (the hard gate)

The gate: does the method address **BLOCK/contiguous** missingness (our case) or only **SCATTERED**
missingness (the clinical case)? Verdicts:

| Method | Citation | Missingness regime it targets | Verdict for us |
|---|---|---|---|
| **mTAND** | Shukla & Marlin, ICLR 2021, arXiv:2101.10318 | SCATTERED — "physiological time series data in electronic health records, which are sparse, irregularly sampled, and multivariate" | **DOWNGRADE.** Interpolation-based: queries reference points *from* observed points. A 4-month block covering 1/3 of the reference grid forces **extrapolation**, not interpolation, over the other 2/3. Headline claim is "as well or better ... while offering significantly faster training times" — an efficiency win. |
| **GRU-D** | Che et al., Sci. Reports 8:6085 (2018), arXiv:1606.01865 | SCATTERED, and explicitly **exploits** informative missingness | **COUNTERINDICATED — actively harmful.** Designed to learn from the missingness pattern. Our missingness pattern is the single most train/test-divergent variable in the dataset (adversarial AUC ~0.99). It would learn "12/12 observed" and transfer nothing. |
| **SeFT** | Horn, Moor, Bock, Rieck, Borgwardt, ICML 2020, arXiv:1909.12064 | SCATTERED, unaligned | **DOWNGRADE.** Permutation-invariant set encoder: discards order, which we have for free and perfectly. Headline is again "competitive ... whilst significantly reducing runtime". |
| **Raindrop** | Zhang, Zeman, Tsiligkaridis, Zitnik, ICLR 2022, arXiv:2110.05357 | **CHANNEL** missingness ("leave-sensor-out"), not temporal-block | **GATE FAILS ON AXIS.** No sensor is ever dropped in our data — all 12 bands present inside the window. Learned inter-sensor graph is unidentifiable at n=1817, T=12. |
| **Neural CDE** | Kidger, Morrill, Foster, Lyons, NeurIPS 2020 Spotlight, arXiv:2005.08926 | SCATTERED, continuous-time | **EXECUTION-RISK NON-STARTER.** SOTA claim is *against ODE/RNN baselines*, not strong masked Transformers; "sometimes orders of magnitude slower than transformer models". |
| **Latent ODE** | Rubanova, Chen, Duvenaud, NeurIPS 2019 | SCATTERED | Same objection. |
| **ContiFormer** | NeurIPS 2023 | SCATTERED, long, continuous-time | Buys principled real-valued gaps. Our gaps are always 1. **Zero value.** |
| **Rough Transformers** | arXiv:2405.20799 (2024) | SCATTERED, long sequences | Length-12 sequences; signature machinery is overkill. |
| **Warpformer** | KDD 2023 | Multi-scale clinical irregularity, SCATTERED | Same objection. |

**Negative finding, stated plainly, because it is the most useful thing here:** the suspicion in your brief
was correct. **The ISTS literature is a scattered-missingness literature.** It does not contain a model
built for your problem, and none of these architectures is worth a submission slot.

**The literature that IS yours is a different one:** *early classification / in-season mapping of satellite
image time series.* Specifically **ELECTS** — Rußwurm, Courty, Emonet, Lefèvre, Tuia, Tavenard,
"End-to-end learned early classification of time series for in-season crop type mapping", *ISPRS Journal of
Photogrammetry and Remote Sensing* **196**:445-456 (2023), arXiv:1901.10681, code
`github.com/MarcCoru/elects`. ELECTS trains on full seasons and is accurate on **prefixes**, balancing
earliness against accuracy. It is an existence proof for the operation you need. **Caveat, honestly:** a
prefix always starts at t=0; your windows have arbitrary offsets, so ELECTS is a *weaker* version of your
problem, and its earliness-reward loss is not directly importable in your remaining time budget. Cite it
for framing and for the code-review narrative, do not spend a slot on it.

And **L-TAE** — Sainte Fare Garnot & Landrieu, AALTD@ECML 2020, arXiv:2007.00586 — matters because it
establishes the SITS field's default: sinusoidal positional encoding of the **actual calendar date**
(characteristic scale tau=1000). Your measured result contradicts the field default. Section 2 explains why
that is not a paradox.

---

## 2. WHY RELATIVE-TIME ENCODING WON BY +0.013 — MECHANISM AND A TESTABLE PREDICTION

### MECHANISM

Two effects, and I believe the second dominates.

**(i) Support shift (weak effect).** With absolute-index encoding, a test row occupies positions
`[s, s+L)` for some offset `s`. The classifier head must be accurate for every `(s, L)` configuration it
meets. With relative encoding every window is remapped onto a canonical `[0, L-1]` axis, so train and test
occupy identical positional support.

**(ii) Parameter sharing across the orbit (strong effect — and this is where it couples to your code).**
There are exactly **24** legal contiguous windows of a 12-month row (`L=4`: 9 offsets; `L=5`: 8; `L=6`: 7).
Under **absolute** encoding these 24 are 24 *distinct* configurations that the model must learn
separately. Under **relative** encoding they collapse to **3** (one per length), giving roughly an 8x
increase in effective examples per configuration. At n=1817 with a 71k-parameter model, that is precisely
the regime where a hard parameter-sharing constraint dominates. Relative-time encoding is not a
"better time representation" — **it is a translation-invariance constraint, and it is buying you variance
reduction, not bias reduction.**

### EVIDENCE

**Kazemnejad, Padhi, Natesan Ramamurthy, Das, Reddy. "The Impact of Positional Encoding on Length
Generalization in Transformers." NeurIPS 2023, arXiv:2305.19466.** Verbatim:

> "Our findings reveal that the most commonly used positional encoding methods, such as ALiBi, Rotary, and
> APE, are not well suited for length generalization in downstream tasks. More importantly, NoPE
> outperforms other explicit positional encoding methods while requiring no additional computation. We
> theoretically demonstrate that NoPE can represent both absolute and relative PEs, but when trained with
> SGD, it mostly resembles T5's relative PE attention patterns... Overall, our work suggests that explicit
> position embeddings are not essential for decoder-only Transformers to generalize well to longer
> sequences."

This is the closest published measurement of your exact intervention (absolute -> relative PE) under your
exact failure mode (train and test occupy different position ranges), and it reports the same sign. It is
not a measurement of *your* quantity — it is language modelling, decoder-only, and about generalizing to
*longer* sequences whereas you generalize to *shorter, offset* ones. Treat it as mechanism corroboration,
not as an effect-size estimate.

**Time2Vec** — Kazemi et al., arXiv:1907.05321 — is the natural "stronger variant of the same idea":
one **linear** (progression) term plus k learnable **periodic** terms, so you do not have to choose between
relative-progression and calendar-seasonal encoding. *Honest caveat: Time2Vec is a 2019 preprint that never
landed a main-track venue acceptance to my knowledge, and its reported gains are modest and
task-dependent.* It is a plausible mechanism, not a guaranteed win.

### THE TESTABLE PREDICTION (this is the payload of Section 2)

The two mechanisms make **opposite** predictions about adding calendar information back:

- If **(i) support shift** dominates, calendar/absolute information is intrinsically untransferable and
  adding it back in *any* form will hurt.
- If **(ii) parameter sharing** dominates, calendar information is fine — it was only the *positional*
  delivery that broke sharing. Delivering it as a **content feature** rather than a **position** keeps the
  sharing intact and adds real phenological signal.

**PREDICTION (P3):** add `sin(2*pi*m/12), cos(2*pi*m/12)` for the absolute calendar month `m` as **two
extra input channels concatenated to each month's token** (content), while leaving the **positional**
encoding purely relative. Under mechanism (ii) this is a small win; under mechanism (i) it is a small loss.
Either way you learn which mechanism is operating, which is worth more than the delta.

**PLATT CHECK:** PASSES. This changes the learned representation and therefore the induced ordering of test
rows. It is not an affine reparameterization of the logit.

**KILL CONDITION (offline, before any submission):** on the masked-OOF replica, if `dAUC <= 0` **and**
`dF1 <= 0`, drop it. Additionally, split masked-OOF AUC by window offset `s`: mechanism (ii) predicts the
gain concentrates in offsets whose calendar months are phenologically distinctive; mechanism (i) predicts a
uniform loss. If neither pattern appears, the theory is wrong and you should stop reasoning from it.

**A CHEAPER, STRICTLY-SAFER SIBLING:** Kazemnejad's NoPE result says *less* explicit positional information
is better under position shift. You cannot use literal NoPE (your encoder is presumably bidirectional, and
a bidirectional encoder with no PE is permutation-invariant — it would destroy order entirely). But you
**can** scale down the relative-PE norm, or ablate the PE amplitude. That is a one-scalar sweep, zero new
parameters.

**EXPECTED EFFECT SIZE:** these are second-order refinements of an intervention that already fired. Honest
bound: **-0.003 to +0.005 composite**, i.e. roughly **-1 to +2 true positives** on 309 public rows. Note
that on 309 rows one flipped prediction is worth ~0.002-0.003 F1, so anything below ~+0.004 composite is
inside seed noise (your measured seed variance is 0.019 — **larger than any effect discussed in this
entire brief**). Do not spend a submission slot on P3 alone.

---

## 3. WHY DURATION NORMALIZATION LOST BY -0.0064

You offered two hypotheses. **My answer: primarily neither as stated — it is (c), a third mechanism — with
(b) as a real secondary contributor and (a) essentially ruled out by your own experimental design.**

### (a) is ruled out by construction
"The number of observed months carries label-correlated information in train that does not transfer."
In the **unmasked** train data every row has exactly 12 months: duration has **zero variance**, so it
cannot be label-correlated and cannot be fit. In the **masked views**, the window length is drawn by
`sample_window` from the measured test `p(L)`, **independently of the label**. So duration is
label-independent in train *by construction*. Hypothesis (a) cannot be the mechanism. (Cross-check
available: regress the train label on the sampled `L` in your masked views — you should get AUC ~0.5. If
you do not, you have an RNG bug, which would be a far more important finding than anything else here.)

### (c) THE ACTUAL MECHANISM: you imposed an invariance the problem does not have
Duration normalization makes the representation **invariant to the number of observed months**. But
observed-month count is not a nuisance — **it is the amount of evidence**. `p(y | x, L=6)` is strictly
sharper than `p(y | x, L=4)`. A length-invariant model *cannot* modulate its confidence with the amount of
evidence it has seen, so it cannot shrink 4-month rows toward the base rate and sharpen 6-month rows away
from it. Under a **0.4 weight on AUC**, that is a direct ranking loss: a well-specified model should rank
a confident 6-month positive above a tentative 4-month positive, and a length-invariant one cannot.
Under a **hard 0.5 cut on the F1 term**, it is worse still — the borderline rows that a length-aware model
would have pushed across the cut stay stranded.

**EVIDENCE.** Benton, Finzi, Izmailov, Wilson, "Learning Invariances in Neural Networks", NeurIPS 2020,
arXiv:2010.11882. The entire premise of Augerino is that the right amount of invariance is unknown and that
too much is costly: *"we often do not know a priori what invariances are present in the data, or to what
extent a model should be invariant to a given augmentation"*, and the training loss "will be flat for
augmentations within the range of invariance present in the data, and then will increase sharply beyond
this range." Duration normalization is an augmentation-style invariance imposed **beyond** the range the
data supports.

### (b) is a real secondary effect
Zhang, Tozzo, Higgins, Ranganath, "Set Norm and Equivariant Skip Connections: Putting the Deep in Deep
Sets", ICML 2022, arXiv:2206.11925, establishes that "layer norm can hurt performance by removing
information useful for prediction", the mechanism being that element-wise standardization forces an
invariance under which activations differing only in scale become indistinguishable, which "reduces
representation power". If your duration normalization divided *amplitudes* (not just counts), you also
destroyed the magnitude of pond flooding/drawdown, which in a 12-band optical+SAR stack is plausibly
discriminative in its own right.

### What the literature says about length normalization for variable-length classification
It says **it depends on whether length is signal or nuisance, and you must decide which**. The clearest
counter-example is Sainte-Fare-Garnot-adjacent but from insurance: the self-attention-over-hierarchical-
claims work (arXiv:1808.10543) reports "a positive correlation of fraudulent claims with the sequence
length was observed. Thus, the aggregation method should be able to naturally scale with the sequence
length. For this reason, sum pooling seemed like the obvious choice." That is the same decision you faced,
resolved the same way your experiment resolved it: **when length carries information, do not normalize it
away.** For you the information is not label-correlation (see (a)) but **evidence quantity**, which is
subtler and, crucially, still ranking-relevant.

### DERIVED PROPOSAL (P4): stop normalizing, start *conditioning*
The opposite of duration normalization is not "ignore duration" — it is "**tell the model the duration
explicitly**". Concatenate a one-hot or scalar `L in {4,5,6}` to the pooled representation before the head.
- **PLATT CHECK: PASSES** — it changes per-row scores non-uniformly (it is an interaction with `x`, not an
  additive offset), so it reorders. *But note the sharp caveat:* if the head learns to use `L` only as an
  **additive logit offset**, the effect on ranking is confined to *between-length-group* reordering, and
  a global Platt refit will NOT remove it (Platt is one global affine map, not a per-group one) — so it
  still counts, but the gain is a group-level recalibration, which is the smallest kind.
- **KILL CONDITION:** on masked OOF, compute AUC **within** each `L` stratum. If within-stratum AUC is
  unchanged and only the pooled AUC moves, you have bought only group recalibration; that is worth ~0.001,
  not worth a slot.
- **RISK:** this hands the model the one variable most correlated with the train/test split. Since `L` is
  label-independent by construction in your masked views, the shortcut risk is low — but it is not zero if
  your masker's `p(L)` interacts with fold composition.

---

## 4. THE WINDOW MISMATCH AS DOMAIN ADAPTATION

### The framing theorem
**Zhou, Balakrishnan, Lipton. "Domain Adaptation under Missingness Shift." AISTATS 2023, PMLR v206;
arXiv:2211.02093.** Their setting is a precise description of yours: DAMS "occurs when labeled source data
and unlabeled target data would be exchangeable but for different missing data mechanisms." Their
load-bearing result: **"If missing data indicators are available, DAMS reduces to covariate shift."**

You have the indicators. Your attention mask consumes them. **So there is no exotic missing-data machinery
you are missing** — and to that extent the claim under attack is right. Their alarming negative results
(e.g. "the optimal linear source predictor can perform arbitrarily worse on the target domain than always
predicting the mean") apply to the *no-indicators* regime and are **not** your situation. Do not let anyone
cite them at you.

### Where that leaves the actual headroom
If the whole problem reduces to matching source `p(x)` to target `p(x)`, then **the quality of that match
is the only thing that matters.** So I read your code. Two findings, one dead, one live.

**DEAD — the matched-distribution question is already solved.** `WindowDist` (`src/data.py:55-60`) stores
measured `length_probs` **and** `start_probs` from `Test.csv`; `sample_window` (`src/features.py:44-63`)
samples from them under `augment.match_test_distribution` (default `True`). Moreover the measured target
distribution is essentially uniform anyway — `src/seq_model.py:776-778` records "the visible S1 length is
uniform over {4,5,6} (345/343/342 rows) and 1030/1030 rows are contiguous." **Your Question 4b has already
been implemented and there is nothing to gain. Honest negative.**

**LIVE, BUT NOT IN THE FORM I FIRST THOUGHT — SEE THE CORRECTION BELOW.**
`_mask_views` (`src/seq_model.py:731-770`) materialises the masked cube **once** per `(row i, view k)` via
a deterministic `rng_for(seed, int(i), tag)` with `tag = (10000 + k) if oof else k`. That cube is then
reused for every epoch. So with `K = R = 2` out of a **24-element** orbit, each row is seen through
**8.3% of its orbit, and the same 8.3% every single epoch.**

> ### *** SELF-CORRECTION (I checked the code before shipping this, and I was wrong) ***
> **Per-epoch resampling ALREADY EXISTS.** `src/seq_model.py:977-992` implements exactly the intervention
> I was about to propose, including an epoch-keyed RNG (`cfg["seed"] + _rep + 7919 * (ep + 1)`), added at
> **iter21** and labelled "instance-expansion". Config: `config/config.yaml:303`
> `resample_per_epoch: false  # iter21 instance-expansion: fresh masked windows/row each epoch (lam=0 only)`.
>
> **So the raw proposal "resample the mask each epoch" is NOT new and its marginal EV is zero.**
> I am flagging this loudly rather than burying it, because a brief that told you to build something you
> built five iterations ago would have cost you a slot.
>
> **What IS new is the gate on line 981:** `if bool(s.get("resample_per_epoch", False)) and lam == 0.0:`.
> Per-epoch resampling is **hard-disabled whenever the cross-view invariance penalty is active** — and that
> penalty (lam > 0) is your champion, +0.0047. **The two mechanisms are currently mutually exclusive by
> implementation, and the flag ships as `false`, so in the champion configuration the masks are frozen at
> 8.3% orbit coverage after all.** The original finding survives; only the proposed remedy changes.

**Two independent literatures say this is a defect, and they say it for different reasons:**

1. **Orbit averaging / variance reduction.** Chen, Dobriban, Lee, "A Group-Theoretic Framework for Data
   Augmentation", NeurIPS 2020. Data augmentation is characterised as learning with an **orbit-averaged
   loss**, whose benefit is **variance reduction** and which "leads to sample efficient learning". A
   two-sample Monte Carlo estimate of that orbit average that is **never redrawn** is not decaying noise —
   it is a **frozen bias**, and it lets the network memorise the specific `(row, window)` pairs. Related:
   Lyle, van der Wilk, Kwiatkowska, Gal, Weller, "On the Benefits of Invariance in Neural Networks",
   arXiv:2005.00178; and a Fourier-analysis treatment (Tahmasebi et al., arXiv:2606.24418) whose surfaced
   claim is that partial augmentation "can retain the statistical benefits of full augmentation despite
   enforcing symmetry only approximately", but that exact invariance "cannot be achieved by any strict
   subset when the hypothesis space is sufficiently expressive". **[I did not verify that last quote
   verbatim against the PDF — treat it as directional, not quotable.]**
2. **Covariate-shift matching (DAMS).** Under DAMS-reduces-to-covariate-shift, correctness requires the
   source window marginal to converge to the target window marginal. With frozen draws it converges to a
   2-atom-per-row caricature and **stops there**, no matter how long you train.

**Answering "Is R=2 enough?" directly and honestly:** the literature does not give a number for your case,
and I will not invent one. What it gives you is the *shape* of the answer: with **resampling**, R controls
gradient variance and the bias vanishes as training proceeds, so R=2 is defensible. **Without resampling,
R is the entire orbit coverage and R=2 is 8.3%.** The distinction between those two regimes is where the
loss is, not the value of R.

### PROPOSAL P1 (top ranked, RESTATED AFTER THE CORRECTION): lift the `lam == 0.0` gate
**THE CLAIM:** the mutual exclusion between per-epoch view resampling and the cross-view invariance penalty
is an **implementation artifact, not a principled restriction**, and the two mechanisms are
**complementary rather than substitutes**. Removing the gate is the intervention.

**WHY THEY ARE COMPLEMENTARY, MECHANISTICALLY.** They act on different failure modes:
- The **cross-view penalty** (lam>0) enforces invariance *within* a batch by penalising `Var_k(logit)`
  across the K views of one owner. It makes the model's answer *consistent* across the two windows it sees.
- **Per-epoch resampling** enlarges *which* windows it sees at all — from 2 frozen atoms to a converging
  Monte-Carlo sample of the 24-atom orbit. It makes the *set* of windows representative.
Consistency across a frozen, unrepresentative pair is a weaker constraint than consistency across a fresh
pair each epoch. Under Chen-Dobriban-Lee, the penalty is a *regulariser toward* the orbit-averaged
predictor while resampling is what makes the orbit average *correctly estimated*. Combining them should
strengthen the champion, not compete with it — and the existing code comment at lines 978-979 already
argues the distributional half of this correctly ("drawn from the SAME measured p(L), p(start|L), so no
new train/test shift is introduced").

**IMPLEMENTATION.** Two lines plus care:
1. Change the guard on `src/seq_model.py:981` from `and lam == 0.0` to allow `lam > 0`.
2. Ensure the resampler returns views **grouped by owner in the same K-per-owner layout** the cross-view
   penalty expects (`src/seq_model.py:571` batches by OWNER and reads `Var_k`), and that the **antithetic
   pairing is re-run inside each epoch's RNG** so view 2 stays maximally distant from view 1
   (`src/seq_model.py:745-766`). `_mask_views` already does both when called fresh — the resampler calls
   `_mask_views`, so this may require no change at all beyond the guard. **Verify the owner grouping and
   the `Var_k` term are non-vacuous after the change; that is the whole risk.**
3. Set `config/config.yaml:303` `resample_per_epoch: true`.

> ### *** SECOND SELF-CORRECTION — I ran the history check I told you to run, and P1 IS LARGELY DEAD ***
> `experiments/LB_LOG.md:1495-1532`, **"iter21 RESULT — instance-expansion is inert"**. Verbatim from your
> own log: *"The cross-exam's #1 data-model lever: treat each `(row, masked sub-window)` as an INDEPENDENT
> training example via PER-EPOCH resampling (fresh masked windows every epoch → ~K·epochs distinct
> instances, not K fixed). Paired control = `seq_a_reltime` (K=2 FIXED views) so the ATC-F1 gap would
> isolate resampling alone."* Arms `c_iexp_rs2`, `c_iexp_rs2_s7`, `c_iexp_rs6`, `c_iexp_rs6_s7` all landed
> at **OOF ≈ 0.982-0.984**, versus champion `seq_a_xview` 0.97523 and control `seq_a_reltime` 0.98041.
> Your log's read: *"Every instance-expansion arm sits at ≈0.982–0.984 OOF — the exact `seq_a_k4`
> fingerprint, and k4 scored one of our worst LBs (0.8665). Higher OOF has been anti-correlated with LB
> throughout. Resampling did not rescue the mechanism; it reproduced the overfit-the-source signature. The
> data-model lane (in-family) is now measured closed."*
>
> **I therefore withdraw P1 as a top-ranked proposal.** The orbit-coverage theory in log 10/14 is
> mechanistically sound and the two literatures do converge on it — **and it was tried, and it did not
> work.** That is exactly the outcome my own FALSIFICATION clause predicted would mean *"relative-time
> encoding already collapsed the orbit, so coverage was never the bottleneck"*. **Section 7's corollary is
> confirmed, and the claim you asked me to refute is strengthened on this axis too.**
>
> **Two honest residuals, and neither justifies a slot on its own:**
> 1. **It was never scored on the LB.** `LB_LOG.md:1522-1526`: the offline screen went **VOID** (ATC-F1
>    14/15, rho +0.929, below gate), so `c_iexp` "was never scored". The lane is closed on an *OOF proxy*
>    that your own log calls unreliable at this resolution — *"even our best lie-detector is not
>    reproducible run-to-run"*. Closed on a proxy is weaker than closed on ground truth. But given the
>    proxy points the wrong way and OOF-inflation has been LB-anti-correlated throughout, **the prior is
>    now clearly negative and I will not argue you into spending a slot on it.**
> 2. **The `lam > 0` corner is still literally untested**, because `src/seq_model.py:981` forbids it: all
>    four iter21 arms necessarily ran at lam=0, against a lam=0 control. My complementarity argument
>    (resampling makes the orbit average *correctly estimated*; the cross-view penalty *regularises toward*
>    it) predicts the penalty would suppress precisely the source-overfitting that iter21 measured. That is
>    a real, unfalsified prediction. **It is also exactly the kind of "one untested corner of a measured-
>    dead lane" reasoning that burns final submissions.** Rank it below P2, and only run it if P2 comes
>    back clean offline and you have a spare slot you would otherwise waste.

**PLATT CHECK: PASSES, decisively.** This changes what the network learns, hence the whole score function,
hence the ranking. It is the furthest thing from an affine reparameterization.

**KILL CONDITION (offline, no submission needed).** Run the masked-OOF replica with resampling on. Check
**three** things, in order:
1. Masked-OOF AUC must not fall. (Primary.)
2. Cross-view penalty magnitude `Var_k(logit)` must not collapse toward zero — if it does, you broke
   antithetic pairing and you are measuring the loss of the +0.0047 champion, not the gain from P1.
3. Train-vs-masked-OOF gap must **narrow**. This is the mechanism-specific signature: if resampling is
   doing what the orbit theory says, it acts as a regularizer and shrinks the memorisation gap. If OOF AUC
   rises but the gap does not narrow, the theory is wrong and the gain is luck — treat it with suspicion.

**EXPECTED EFFECT SIZE, AFTER THE ITER21 CORRECTION.** My pre-correction estimate was +0.001 to +0.006.
**Post-correction it is 0.000, with a left tail** — the mechanism was measured inert at lam=0 and produced
the source-overfitting fingerprint. The only unmeasured quantity is whether the cross-view penalty
suppresses that fingerprint. Honest bound on the lam>0 corner: **-0.005 to +0.004**, centred on zero,
inside seed variance (0.019) by a factor of three.

**EXECUTION RISK: LOW mechanically** (one guard, one config flag, verify `Var_k` stays non-vacuous),
**HIGH strategically** — it is a re-probe of a lane your own loop marked "measured closed", at a
resolution your own arbiter has been shown not to resolve.

**FALSIFICATION.** If per-epoch resampling leaves masked-OOF AUC flat across all 10 seeds, then the
orbit-coverage theory is wrong for this problem — most likely because the relative-time encoding *already*
collapsed the 24-element orbit into 3 equivalence classes, so 2 frozen draws were never a coverage
bottleneck in the first place. **That is a genuinely informative outcome: it would confirm the claim you
asked me to refute, and it would confirm it for a reason nobody had articulated.** Either way you learn
something. Note this also makes P1 and the relative-time win *partially redundant by construction* — which
lowers my point estimate and is why I bounded the effect at +0.006 rather than higher.

---

## 5. TEST-TIME AUGMENTATION OVER WINDOWS

### The Platt question first, because it decides everything
**TTA SURVIVES PLATT ANNIHILATION.** Averaging model outputs over *different inputs* is not an affine map
of a single logit; it is a genuinely new function of the row. The direct published evidence that it
reorders individual predictions is Shanmugam, Blalock, Balakrishnan, Guttag, **"Better Aggregation in
Test-Time Augmentation", ICCV 2021**:

> "even when test-time augmentation produces a net improvement in accuracy, it can change many correct
> predictions into incorrect predictions."

Flipping individual predictions **is** reordering. TTA is therefore one of the few remaining levers in your
whole search space that is not annihilated by the train-refit Platt. (Same group, CVPR 2025,
"Test-time Augmentation Improves Efficiency in Conformal Prediction", reinforces that TTA restructures the
score distribution rather than rescaling it.) Their earlier report is
Shanmugam et al., "When and Why Test-Time Augmentation Works", arXiv:2011.11156.

### What your code already says about this
- **Scattered TTA was tried and lost.** `src/seq_model.py:888-889` logs an MC temporal-dropout TTA
  (`n_views`, `mask_months`), and lines 779-781 record the verdict: "We deliberately do NOT hole-punch
  (mask 1-2 interior active months): interior gaps occur in neither the train masker nor the test set, and
  that off-manifold augmentation is the diagnosed cause of the iter6 TTA loss (-0.0023)." **That diagnosis
  is correct and the literature agrees with it**: TTA transformations must lie on the data manifold.
- **On-manifold contiguous sub-window views ALREADY EXIST but appear to be used for the wrong thing.**
  `_test_views` (`src/seq_model.py:773-813`) builds every legal contiguous sub-window of each test row's
  visible block, with view 0 = the full window. It is wired into the **cross-view invariance penalty**
  (semi-supervised consistency), not, as far as the code path shows, into a **prediction-time average**.

### PROPOSAL P2: average predictions over `_test_views`, aggregating in logit space
**MECHANISM.** The infrastructure is built. Predict on all `K_u` views, average, use as the test score.
Aggregate in **logit** space with view 0 (the full window) up-weighted — averaging probabilities compresses
toward 0.5 and, with a hard 0.5 cut on 60% of the metric, that compression is not free.

**BLOCK vs SCATTERED: PASSES.** Sub-windows of a contiguous block are contiguous blocks. This is the one
TTA that stays inside the measured test support, and the code comment already establishes that this is
exactly what distinguishes it from the iter6 loss.

**TWO OBJECTIONS, BOTH REAL, AND I RANK THIS BELOW P1 BECAUSE OF THEM:**
1. **Asymmetric coverage.** 345/1030 test rows have `L=4=min_len` and therefore have **exactly one legal
   view** (`src/seq_model.py:783-786`). TTA can only perturb 685/1030 rows, and only the *longer-window*
   ones. You would be applying a variance-reduction treatment to two-thirds of the rows and not the other
   third — which itself distorts the relative ordering of `L=4` vs `L=5,6` rows. On an AUC term that pools
   all rows, this is a genuine hazard, not a footnote.
2. **Cropping can delete the evidence.** From the time-series augmentation survey (arXiv:2310.10060):
   "The most used data augmentation method for time series classification is the slicing window technique,
   originally inspired by the image cropping technique for data augmentation in computer vision tasks.
   However, for time series data, one cannot make sure that the discriminative information has not been
   lost when a certain region of the time series is cropped." Cropping 6 months to 4 can remove the single
   drawdown month that identifies a pond. The resulting downward bias is **label-correlated** — it hits
   positives with temporally concentrated evidence — which is the worst possible bias structure here.

**KILL CONDITION (offline).** On the masked-OOF replica, build the identical sub-window TTA and check:
(1) pooled masked-OOF AUC rises; (2) **stratify by window length**: recall on `L=6` positives must not fall
relative to `L=4` positives — that is the exact signature of objection 2; (3) the number of 0.5-crossings
in each direction (the ICCV paper's warning made concrete): if `flips_to_wrong` is a large fraction of
`flips_to_right`, the net gain is fragile and will not survive the 309-row public sample.

**MITIGATION IF (2) FIRES:** weight view 0 (full window) at, say, 0.5 and split the remaining 0.5 across
sub-views. This preserves most of the variance reduction while bounding the crop-bias.

**EXPECTED EFFECT SIZE.** TTA in the literature is typically a "small but meaningful" accuracy gain. Here
it is applied to 66% of rows with a known bias hazard. Honest bound: **-0.003 to +0.004 composite**, i.e.
**-1 to +2 true positives** on 309 public rows. **The variance of this estimate is larger than its mean.**

**EXECUTION RISK: LOW on compute (inference only, no retrain — you can evaluate it on a banked prediction
bundle), MODERATE on judgement** (the two objections above). Because it needs no retrain, **P2 is the one
thing you can test without consuming a 25-minute slot.** Do it while P1 trains.

**FALSIFICATION.** If the masked-OOF stratified check shows `L=6` recall falling, objection 2 is confirmed
and P2 is dead — do not submit it.

---

## 6. MISSINGNESS AS SIGNAL — CLOSED ON FIRST PRINCIPLES, NOT MERELY ON CAUTION

**The literature says missingness usually helps.** "On Missingness Features in Machine Learning Models for
Critical Care: Observational Study", *JMIR Medical Informatics* 2021 (PMC8701717), across 48,336 EHRs from
the 2012 and 2019 PhysioNet Challenges, reports that including missingness features "generally improved
model performance in **retrospective** tasks". GRU-D is built on the same premise: "Missing values and
their missing patterns are often correlated with the target labels, a phenomenon known as **informative
missingness**."

**And that is precisely why it is closed for you.** Missingness features encode the *data-collection
process*. Your data-collection process is the one thing that differs hardest between domains
(adversarial AUC ~0.99). But the argument is stronger than "risky":

- On **true, unmasked train rows**, every row is 12/12. Any window-descriptive feature (length, start,
  end, centre, coverage fraction) is **constant** — **zero variance, zero fittable information.**
- On **masked views**, the window is drawn by `sample_window` from the measured `p(L), p(start|L)`,
  **independently of the label**. So any window feature is **label-independent by construction.**

**Therefore there is no legal way to extract label information from the observation window on this dataset,
because there is none to extract.** This lane is closed by the structure of the data, not by prudence. Your
`CLOSED LANES` list should absorb it permanently.

**The one genuine exception, already handled.** `WindowDist.s2_dropout_rates` (`src/data.py:60`) records
per-calendar-month Sentinel-2 dropout, which is driven by **cloud cover** — a real geographic and seasonal
variable, and geography plausibly correlates with aquaculture presence. That is a real (small) channel and
it is **already modelled inside the masker**. Do not build a second, unregularised copy of it: a
free-floating "S2 dropout rate" feature would be a geographic proxy learned at the resolution of one
scalar, which is exactly the kind of thing that fits the public 309 and dies on the private 721.

---

## 7. UNIFIED THEORY — WHY RELATIVE-TIME WON AND DURATION-NORMALIZATION LOST

**One principle, both signs:**

> **Impose invariance to transformations that are genuine symmetries of the label; never impose invariance
> to quantities that change how much evidence you have.**

- **Window TRANSLATION is a symmetry.** A pond observed in months 2-6 and the same pond observed in months
  6-10 have the same label, and (given no strong calendar phase-locking of the target) approximately the
  same conditional. Imposing translation-invariance via relative-time encoding costs **no bias** and buys
  a large **variance** reduction: it collapses a 24-element orbit into 3 equivalence classes, ~8x more
  effective data per configuration at n=1817. **-> +0.013.**
- **Window LENGTH is NOT a symmetry.** `p(y | x, L=4)` and `p(y | x, L=6)` genuinely differ: the second is
  sharper because it rests on 50% more evidence. Imposing length-invariance via duration normalization is
  **pure bias with no variance payoff** — the model loses its ability to modulate confidence with evidence
  quantity, which directly damages ranking (0.4*AUC) and strands borderline rows at the hard 0.5 cut
  (0.6*F1). Secondarily, if the normalization touched amplitudes, it also destroyed magnitude information
  (Set Norm, ICML 2022). **-> -0.0064.**

Formally this is Augerino's thesis (Benton et al., NeurIPS 2020): the loss is flat for invariances the data
possesses and "increases sharply beyond" them. You empirically located both edges of that flat region in
two experiments. **Relative-time sits inside it; duration normalization sits outside it.**

**The corollary that generates P1 — and this is the part that refutes the claim.** If relative-time
encoding won because it is a *hard-coded* translation invariance, then the *soft*, data-driven version of
the same invariance — masked-view augmentation over the window orbit — is doing the same job by a second,
independent route. And your soft route is currently running at **8.3% orbit coverage, frozen**. The two
routes are partially redundant, which is why P1's expected gain is bounded and modest — but they are not
*fully* redundant, because the augmentation route also teaches the model about **length**, which the
relative encoding deliberately does not collapse. **P1 is the completion of the mechanism that already
paid you +0.013.**

**The second corollary, which is a warning:** because relative-time already handles translation, the
*marginal* value of every further translation-flavoured intervention in this brief is small. That is the
honest reason nothing here is worth more than ~+0.006, and the honest reason your leader's ~0.929-0.936 is
not going to be reached from this direction. If you need +0.022 to lead, **it is not in this literature.**

---

## 8. RANKED PROPOSALS — EV PER UNIT EXECUTION RISK

Ordered by **expected value per unit execution risk**, after both self-corrections.

| # | Proposal | Platt | Block-gate | Est. composite | Exec risk | Slot? |
|---|---|---|---|---|---|---|
| **P2** | **Sub-window TTA over `_test_views`, logit-space, view-0 up-weighted** | **PASS** | **PASS** | -0.003 to +0.004 | **NONE on compute (no retrain)**, MOD judgement | **Run it offline — it costs no slot** |
| P5 | Shrink relative-PE amplitude (NoPE-direction scalar sweep) | PASS | PASS | ~0.000 +/- 0.003 | VERY LOW, offline | Offline sweep only, diagnostic |
| P3 | Cyclic `sin/cos(2*pi*m/12)` as **content** channels, PE stays relative | PASS | PASS | -0.003 to +0.005 | LOW (2 channels) | Only with a spare slot; value is **diagnostic**, and `LB_LOG.md:1534` already lists **positional ✅closed** |
| P4 | Concatenate `L in {4,5,6}` to the pooled representation | PASS (group-level only) | PASS | ~+0.001 | LOW | Fold into P3's run; never its own slot |
| ~~P1~~ | ~~Per-epoch resampling with `lam > 0`~~ | PASS | PASS | **-0.005 to +0.004, centred 0** | **HIGH strategically** | **NO — measured inert at iter21 (`LB_LOG.md:1495`); only the lam>0 corner is untested and the prior is negative** |
| — | mTAND / SeFT / Raindrop / GRU-D / Neural CDE / ContiFormer / Warpformer | n/a | **FAIL** | negative | HIGH | **NO** |
| — | Match masking distribution to test | n/a | n/a | **0 — already implemented** (`features.py:44-63`) | — | **NO** |
| — | Any window-derived feature as a predictor | n/a | n/a | **0 by construction** (Sec 6) | — | **NO** |

**Recommended plan for the 2-3 remaining slots, at 25 min each:**
1. **Now, free, no slot:** evaluate P2 offline on the masked-OOF replica **and** on a banked
   test-prediction bundle (`_test_views` + a stored `p_test_raw`; no training at all). Apply the three-part
   kill condition, especially the `L=6`-vs-`L=4` recall stratification.
2. **If and only if P2 passes all three checks:** submit the champion + P2 as a single change. This is an
   inference-time-only change, which also reads cleanly in the 35% code review.
3. **Do NOT spend a slot on P1, P3, P4 or any architecture swap.** Every effect in this brief is between
   3x and 20x smaller than your measured seed variance of 0.019. Your log already reached this conclusion
   independently: *"only model-class-sized effects (~0.05) are measurable with this budget"*
   (`LB_LOG.md:1531`). **The highest-value use of a remaining slot is almost certainly an extra seed on the
   existing finalist, not anything in this document.**

---

## 9. CAVEATS

1. **I did not refute the claim on its main axis, and I am not going to pretend I did.** On architecture
   swaps the claim is correct: the ISTS literature is a scattered-missingness literature and has no model
   built for a contiguous train/test-shifted observation window. That is the single most decision-relevant
   sentence in this brief and it points *away* from spending slots.
2. **Every effect discussed here is smaller than your measured seed variance (0.019).** Nothing below
   ~+0.008 composite can be confirmed by a single public-LB read on 309 rows. All kill conditions are
   therefore written for the masked-OOF replica across the 10-seed pool, not for the leaderboard.
3. **The public LB is 309 rows.** One flipped prediction is worth roughly 0.002-0.003 on the F1 term. Every
   "expected effect size" above should be read as "between -1 and +3 flipped rows", which is exactly the
   regime where you cannot distinguish a real effect from noise on one read.
4. **Effect sizes are mechanism-derived, not measured.** No paper in this brief measures your quantity:
   n=1817, T=12, 12 channels, contiguous block window shift, 0.6*F1+0.4*AUC. Kazemnejad et al. is
   decoder-only language modelling extrapolating to *longer* contexts; Chen-Dobriban-Lee is theory;
   Shanmugam et al. is image classification. **I extrapolated mechanisms across domains and I have flagged
   each place I did so.** Where I gave a range, the range is my honest uncertainty, not a confidence
   interval from a paper.
5. **Three quotes need verbatim confirmation before external use:** (a) the Fourier-analysis augmentation
   claim about strict subsets (arXiv:2606.24418); (b) the Set Norm element-wise-standardization sentence
   (arXiv:2206.11925); (c) the "94.2% AUROC from missing indicators alone" sepsis figure, which appeared in
   a search summary and which **I could not trace to a primary source — do not cite that number.**
   The Kazemnejad abstract, the mTAND abstract, and the DAMS "reduces to covariate shift" result were
   confirmed against primary or near-primary sources.
6. **I self-corrected twice, in public, and both corrections went against my own proposal.** First:
   per-epoch resampling already exists (`src/seq_model.py:977-992`, `config/config.yaml:303`). Second and
   worse: **iter21 already ran it and measured it inert** (`experiments/LB_LOG.md:1495-1532`). I have left
   the full reasoning chain in Section 4 rather than deleting it, because the derivation is what shows the
   lane is closed *for a principled reason* — but **anyone reading only the top of Section 4 would come
   away with the wrong recommendation, so read the two boxed corrections.**
7. **Only one corner of P1 is genuinely untested** — resampling with `lam > 0`, which the code forbids at
   line 981 and which all four iter21 arms therefore could not have exercised. I believe the
   complementarity argument is sound. I also believe acting on "one untested corner of a lane your own loop
   marked measured-closed" is how final submissions get burned. **Both of those beliefs are in the
   document; the second one wins.**
8. **`LB_LOG.md:1534` lists "positional ✅closed".** That directly discounts P3 and P5 relative to what I
   wrote in Section 2. I have downgraded them in the ranking; the *mechanistic explanation* in Section 2 of
   why relative-time won is still the useful output there, not the follow-on experiments.
9. **The relative-time win and the augmentation route are partially redundant by construction** (Section 7,
   second corollary). iter21's null result is the empirical confirmation of exactly that redundancy. This
   is the strongest version of the claim you asked me to refute, and I am ending on it: *relative-time
   encoding did not merely help — it appears to have consumed most of the available translation-invariance
   headroom, which is why the soft-augmentation route measured flat on top of it.* That mechanism was not
   previously articulated in the loop, and it is the real deliverable of this brief.

---

## RAW RESEARCH LOG (append-only)

### [log 1] mTAND — Shukla & Marlin, ICLR 2021, arXiv:2101.10318
OpenReview: https://openreview.net/pdf?id=4c0J6lwQ4_
Search-level summary confirmed: "mTANs are fundamentally continuous-time, interpolation-based models";
"learn an embedding of continuous time values and use an attention mechanism to produce a fixed-length
representation of a time series containing a variable number of observations."
KEY STRUCTURAL POINT FOR US: mTAND queries a FIXED set of reference time points from the observed
points. That is exactly the operation that would let a 4-6 month test window be re-expressed on the same
12-slot reference grid as a 12-month train row. Need to check: does it degrade when the observed set is a
contiguous BLOCK covering only 1/3 of the reference grid (extrapolation, not interpolation)? -> pending.

### [log 2] Block vs scattered missingness — first pass
Search surfaced (needs primary verification):
- "point, subsequence, and block" as the three canonical missingness patterns in imputation benchmarks
  (TSI-Bench, arXiv:2406.12747).
- MCAR gives much LOWER imputation error than block patterns => block missingness is the harder regime,
  and the standard clinical ISTS literature (PhysioNet/MIMIC) is dominated by SCATTERED missingness.
- Informative missingness anchor: sepsis prediction where MISSING INDICATORS ALONE reach ~94.2% AUROC
  with XGBoost. Needs primary source; this is the canonical "missingness is a shortcut" warning for our
  Section 6.

### [log 3] mTAND verbatim abstract (arXiv:2101.10318, Shukla & Marlin, ICLR 2021)
VERBATIM: "Irregular sampling occurs in many time series modeling applications where it presents a
significant challenge to standard deep learning models. This work is motivated by the analysis of
physiological time series data in electronic health records, which are sparse, irregularly sampled, and
multivariate. In this paper, we propose a new deep learning framework for this setting that we call
Multi-Time Attention Networks. Multi-Time Attention Networks learn an embedding of continuous-time values
and use an attention mechanism to produce a fixed-length representation of a time series containing a
variable number of observations. We investigate the performance of this framework on interpolation and
classification tasks using multiple datasets. Our results show that the proposed approach performs as well
or better than a range of baseline and recently proposed models while offering significantly faster
training times than current state-of-the-art methods."
READ: motivation is EHR = "sparse, irregularly sampled" = SCATTERED. The abstract never claims block/window
robustness. Its win is "as well or better ... while offering significantly faster training" — i.e. a SPEED
claim as much as an accuracy claim. Weak evidence that mTAND would beat our masked Transformer on accuracy.

### [log 4] GRU-D — Che et al., Scientific Reports 8:6085 (2018), arXiv:1606.01865
Core: "Missing values and their missing patterns are often correlated with the target labels, a phenomenon
known as informative missingness." GRU-D "exploits two representations of informative missingness patterns:
masking and time interval", with a learned temporal DECAY on inputs and hidden state.
*** THIS IS A COUNTERINDICATION FOR US, NOT AN OPPORTUNITY. *** GRU-D is explicitly designed to EXPLOIT the
missingness pattern. Our missingness pattern is the single most train/test-divergent feature in the dataset
(adversarial AUC ~0.99). A GRU-D-style model would learn the train missingness distribution (= none, 12/12
observed) and have nothing to transfer. Predicted outcome: GRU-D underperforms our masked Transformer.
This is a KILL, not a lead.

### [log 5] Raindrop — Zhang, Zeman, Tsiligkaridis, Zitnik, ICLR 2022, arXiv:2110.05357
"RAINDROP represents every sample as a separate sensor graph and models time-varying dependencies between
sensors with a novel message passing operator." Evaluated "in four challenging settings including a
leave-sensor-out setup", outperforming SOTA "by up to 11.4% (absolute F1-score points)".
*** GATE FAILS ON AXIS. *** Raindrop's hard setting is LEAVE-SENSOR-OUT = CHANNEL missingness. Our
missingness is TEMPORAL-BLOCK with ALL 12 channels present inside the window. Raindrop's inductive bias
(inter-sensor graph) buys nothing when no sensor is ever dropped. Also n=1817/length-12/12-channel is far
below the regime where a learned sensor graph is identifiable. DOWNGRADE.

### [log 6] *** THE RIGHT LITERATURE IS NOT ISTS — IT IS EARLY CLASSIFICATION OF SITS ***
This is the single most important redirection of this brief. Our problem is NOT "irregularly sampled time
series" (scattered, clinical). It is "classify from a CONTIGUOUS PARTIAL OBSERVATION WINDOW of a satellite
image time series" — which is precisely the *early classification / in-season crop mapping* literature.
- ELECTS: Rußwurm, Courty, Emonet, Lefèvre, Tuia, Tavenard. "End-to-end learned early classification of
  time series for in-season crop type mapping." ISPRS Journal of Photogrammetry and Remote Sensing,
  vol. 196, pp. 445-456, 2023. arXiv:1901.10681. Code: github.com/MarcCoru/elects.
  ELECTS "estimates a classification score and a probability of whether sufficient data has been observed
  to come to an early and still accurate decision"; loss balances earliness vs accuracy; "experiments on
  four crop classification datasets from Europe and Africa show that ELECTS allows reaching
  state-of-the-art accuracy while reducing the quantity of data to be downloaded, stored, and processed."
  => Existence proof that a model can be trained on FULL series and made accurate on PREFIXES. Our case is
  the harder generalization: not a prefix but an arbitrary-offset contiguous window.
- L-TAE: Sainte Fare Garnot & Landrieu, "Lightweight Temporal Self-Attention for Classifying Satellite
  Image Time Series", AALTD@ECML 2020, arXiv:2007.00586. Uses a sinusoidal positional encoding of the
  ACTUAL DATE (characteristic scale tau=1000), duplicated and added to each channel group.
  *** TENSION WITH OUR RESULT: the SITS consensus is ABSOLUTE-CALENDAR positional encoding (day-of-year),
  because crop phenology is calendar-locked. We measured RELATIVE-time beating it by +0.013. Either our
  target is not calendar-phase-locked, or absolute encoding is failing for a support/covariate-shift
  reason, not an information reason. This distinction generates our best testable prediction (Sec 2). ***

### [log 7] Time2Vec — Kazemi et al., arXiv:1907.05321 (2019)
"a model-agnostic vector representation for time that can be easily imported into many existing and future
architectures and improve their performances." Formulation = one LINEAR term + k PERIODIC (sine) terms with
learnable omega, phi: "The periodic function (such as sine) captures periodic behaviors in data, while the
linear term represents the progression of time and captures non-periodic patterns in the input that depend
on time."
RELEVANCE: Time2Vec is literally the union of the two encodings we are debating — a LINEAR (progression /
relative-position) term AND PERIODIC (calendar/seasonal) terms, learned. Our experiment showed the
progression term matters (+0.013). Time2Vec says do not choose; fit both and let omega/phi decide. This is
the cheapest strong variant of the winning idea.
CAUTION: Time2Vec is a 2019 workshop-style arXiv preprint (never a main-track venue acceptance to my
knowledge) and its reported gains are modest and task-dependent. It is a plausible mechanism, not a
guaranteed win. Do not oversell it.

### [log 8] *** TTA SURVIVES PLATT — Shanmugam, Blalock, Balakrishnan, Guttag, ICCV 2021 ***
"Better Aggregation in Test-Time Augmentation", ICCV 2021.
https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf
KEY VERBATIM (as surfaced): "even when test-time augmentation produces a net improvement in accuracy, it
can change many correct predictions into incorrect predictions."
*** PLATT CHECK PASSES. *** That sentence is direct evidence that TTA is NOT an affine reparameterization
of the logit: it flips individual decisions, i.e. it reorders samples. Averaging over augmented views is a
nonlinear operation on the score (mean of sigmoids != sigmoid of mean of logits, and even mean-of-logits
over DIFFERENT INPUTS is a genuinely new function). Therefore TTA is one of the few remaining levers that
is NOT annihilated by our train-refit Platt. It also comes with the paper's own warning: net gain can hide
a large number of newly-broken predictions, which at a HARD 0.5 cut on the F1 term is exactly the risk
profile we must respect.
Same group, CVPR 2025: "Test-time Augmentation Improves Efficiency in Conformal Prediction" — further
evidence TTA changes the score distribution's structure, not just its scale.

### [log 9] SeFT — Horn, Moor, Bock, Rieck, Borgwardt, ICML 2020, arXiv:1909.12064, PMLR v119
"treats each observation as a tuple comprising a time, a value, and a modality indicator, with all
observations summarized as a set using a set function"; "performs competitively on healthcare time series
datasets whilst significantly reducing runtime."
GATE: SeFT is a permutation-invariant SET encoder — it deliberately DISCARDS sequence order and relies on
the time embedding alone. It targets SCATTERED unaligned clinical measurements. Two problems for us:
(a) our sequence is perfectly regular and aligned inside the window, so set-ification throws away structure
we can use for free; (b) SeFT's headline claim is again "competitive ... significantly reducing runtime",
i.e. an EFFICIENCY win, not an accuracy win. DOWNGRADE as an architecture swap.
BUT note the one transferable idea: SeFT's aggregation is a SET aggregation over a VARIABLE-SIZED set,
which is exactly the mean-vs-sum question underlying our duration-normalization loss (see Sec 3).

### [log 10] *** DATA AUGMENTATION = ORBIT AVERAGING; R=2 MAY BE THE BUG ***
- Chen, Dobriban, Lee. "A Group-Theoretic Framework for Data Augmentation." NeurIPS 2020.
  https://papers.nips.cc/paper/2020/file/f4573fc71c731d5c362f0d7860945b88-Paper.pdf
  Data augmentation = learning with an ORBIT-AVERAGED loss; the benefit is characterised as VARIANCE
  REDUCTION, and it "leads to sample efficient learning."
- Lyle, van der Wilk, Kwiatkowska, Gal, Weller. "On the Benefits of Invariance in Neural Networks."
  arXiv:2005.00178.
- Tahmasebi et al. "Data Augmentation: A Fourier Analysis Perspective." (PMLR / arXiv:2606.24418) —
  surfaced claim: partial augmentation "can retain the statistical benefits of full augmentation despite
  enforcing symmetry only approximately", HOWEVER "enforcing exact invariance via data augmentation
  requires averaging over the entire group, and cannot be achieved by any strict subset when the hypothesis
  space is sufficiently expressive." [VERIFY VERBATIM BEFORE QUOTING]
*** THE ARITHMETIC THAT MATTERS FOR US. *** Our transformation "group" (really a finite set of window
crops) has size: L=4 -> 9 offsets, L=5 -> 8, L=6 -> 7, TOTAL = 24 distinct contiguous windows of a
12-month row. R=2 views/row/repeat samples 2/24 = 8.3% of the orbit.
- If the 2 views are RESAMPLED every epoch, the orbit-averaged loss is being Monte-Carlo estimated and the
  estimator converges over training; R=2 is then mainly a GRADIENT-VARIANCE issue, not a bias issue.
- If the 2 views are FIXED per row (drawn once), we are training on a strict, tiny subset of the orbit and
  by the theory above we get neither exact nor well-estimated invariance — a real, fixable defect.
=> HIGHEST-PRIORITY CHEAP CHECK: confirm whether our mask is resampled per epoch. If not, make it so. That
is a ~0-cost code change with a mechanism-backed reason to move the ranking.

### [log 11] CODEBASE AUDIT (done — this changes several of my proposals)
File: C:\Users\ADMIN\OneDrive\Desktop\OSBORN\AGENTIC-AI-DVPT\ZINDI-PROJECTS\geoai-aquaculture\src\seq_model.py
File: C:\Users\ADMIN\OneDrive\Desktop\OSBORN\AGENTIC-AI-DVPT\ZINDI-PROJECTS\geoai-aquaculture\src\features.py
File: C:\Users\ADMIN\OneDrive\Desktop\OSBORN\AGENTIC-AI-DVPT\ZINDI-PROJECTS\geoai-aquaculture\src\data.py

(a) MATCHED MASKING DISTRIBUTION IS ALREADY DONE. `WindowDist` (data.py:55-60) stores measured
    `length_probs` p(L) AND `start_probs` p(start|L) from Test.csv; `sample_window` (features.py:44-63)
    draws from them when `augment.match_test_distribution` is true (default True).
    => Question 4's "should the masking distribution be MATCHED to the empirical test window-length
    distribution rather than uniform over 4-6?" is ALREADY IMPLEMENTED, and moreover the measured test
    length distribution IS essentially uniform: seq_model.py:776-778 records "the visible S1 length is
    uniform over {4,5,6} (345/343/342 rows) and 1030/1030 rows are contiguous." NOTHING TO GAIN HERE.
    This is an honest negative — my Section 4 lead #1 is dead on arrival.

(b) *** MASKS ARE FIXED, NOT RESAMPLED PER EPOCH. *** `_mask_views` (seq_model.py:731-770) materialises
    the masked cube ONCE per (row i, view k) using a deterministic `rng_for(seed, i, tag)` with
    `tag = (10000 + k) if oof else k`. The masked cube is then reused for every epoch of training.
    With K=R=2 out of |orbit| = 24 legal contiguous windows, each row is seen through 8.3% of its orbit,
    and the SAME 8.3% every epoch. Under Chen-Dobriban-Lee the augmented objective is a 2-sample
    Monte-Carlo estimate of the orbit-averaged loss that is NEVER re-drawn, so its error never averages
    out across epochs — it is a fixed BIAS, not decaying noise, and the network can memorise the specific
    (row, window) pairs. This is the strongest genuinely-new lead in this brief.
    NOTE THE CONFOUND: `antithetic_views` (seq_model.py:745-766) deliberately makes view 2 maximally
    distant from view 1 to power the cross-view invariance penalty (champion, +0.0047), and that penalty
    is defined on exactly K=2. Any per-epoch resampling must PRESERVE antithetic pairing (resample the
    pair jointly each epoch) or it will destroy a measured win. See proposal P1.

(c) TTA ALREADY PARTLY EXPLORED AND PARTLY NOT.
    - seq_model.py:888-889 logs "seq MC temporal-dropout TTA ON: n_views=%d mask_months=%s
      (inference-only)" — i.e. SCATTERED hole-punching TTA.
    - seq_model.py:779-781 records the verdict: "We deliberately do NOT hole-punch (mask 1-2 interior
      active months): interior gaps occur in neither the train masker nor the test set, and that
      off-manifold augmentation is the diagnosed cause of the iter6 TTA loss (-0.0023)."
    - `_test_views` (seq_model.py:773-813) ALREADY BUILDS on-manifold contiguous SUB-WINDOW views of test
      rows — but it is used for the cross-view INVARIANCE PENALTY (semi-supervised consistency), not, as
      far as the code path shows, as a prediction-time average.
    => The live question is narrow and precise: has anyone averaged predictions over `_test_views`
    on-manifold contiguous sub-windows? The infrastructure exists. This is Section 5's proposal P2.
    IMPORTANT CAVEAT ON P2: 345/1030 test rows have L=4=min_len and therefore have EXACTLY ONE legal
    sub-window (seq_model.py:783-786). TTA can only touch 685/1030 rows (66%), and by construction it
    perturbs the LONG-window rows only — an asymmetric perturbation that could itself distort ranking
    between L=4 and L=6 rows. This is a real risk, not a footnote.

### [log 12] *** THE CITATION THAT EXPLAINS THE +0.013 ***
Kazemnejad, Padhi, Natesan Ramamurthy, Das, Reddy. "The Impact of Positional Encoding on Length
Generalization in Transformers." NeurIPS 2023. arXiv:2305.19466.
VERBATIM ABSTRACT (key sentences): "Length generalization, the ability to generalize from small training
context sizes to larger ones, is a critical challenge... Positional encoding (PE) has been identified as a
major factor influencing length generalization... Our findings reveal that the most commonly used
positional encoding methods, such as ALiBi, Rotary, and APE, are not well suited for length generalization
in downstream tasks. More importantly, NoPE outperforms other explicit positional encoding methods while
requiring no additional computation. We theoretically demonstrate that NoPE can represent both absolute and
relative PEs, but when trained with SGD, it mostly resembles T5's relative PE attention patterns... Overall,
our work suggests that explicit position embeddings are not essential for decoder-only Transformers to
generalize well to longer sequences."
WHY THIS IS OUR RESULT: we changed absolute-index -> relative-to-window encoding and gained +0.013. This
paper is the closest published measurement of exactly that intervention under exactly our failure mode
(train and test occupy different position ranges). It also supplies a FREE, SHARPER VARIANT WE HAVE NOT
TRIED: **NoPE** — drop the positional signal entirely and let the causal/attention structure infer order.
Caveat: NoPE's theory in that paper is for DECODER-ONLY (causal) transformers; a bidirectional encoder with
no PE is permutation-invariant and would lose order entirely. Our model must therefore keep SOME temporal
signal. The honest transfer of their finding is: *less explicit absolute positional information is better
when position support shifts*, which is what we already measured. This raises my prior that further
absolute-position ablation (e.g. shrinking the PE norm) helps, and lowers my prior that adding an
elaborate continuous-time module (mTAND/Time2Vec periodic terms keyed to absolute calendar month) helps.

### [log 13] Augerino — Benton, Finzi, Izmailov, Wilson. "Learning Invariances in Neural Networks."
NeurIPS 2020, arXiv:2010.11882.
"we often do not know a priori what invariances are present in the data, or to what extent a model should
be invariant to a given augmentation." Mechanism: "The training loss for the augmentation parameters will
be flat for augmentations within the range of invariance present in the data, and then will increase
sharply beyond this range."
*** THIS IS THE FORMAL STATEMENT OF WHY DURATION NORMALIZATION LOST. *** Augerino's whole premise is that
imposing an invariance the data does not have costs you accuracy — the loss "increases sharply beyond" the
true range of invariance. Duration normalization imposes invariance to the number of observed months.
Window TRANSLATION is a (near-)symmetry of our label. Window LENGTH is NOT: p(y | x, L=6) is strictly
sharper than p(y | x, L=4) because L=6 is strictly more evidence. Enforcing L-invariance is therefore
misspecification with no variance payoff, whereas enforcing translation-invariance (relative-time) is a
true symmetry with a large variance payoff. ONE PRINCIPLE, BOTH SIGNS. See Section 7.

### [log 14] *** THE THEOREM THAT TELLS US WHERE THE REMAINING HEADROOM IS ***
Zhou, Balakrishnan, Lipton. "Domain Adaptation under Missingness Shift." AISTATS 2023, PMLR v206
(proceedings.mlr.press/v206/zhou23b/zhou23b.pdf), arXiv:2211.02093.
Setting (their words, as surfaced): "The problem of Domain Adaptation under Missingness Shift (DAMS) occurs
when labeled source data and unlabeled target data would be exchangeable but for different missing data
mechanisms." That is EXACTLY our setting: train and test rows would be exchangeable but for the
observation window.
*** LOAD-BEARING RESULT: "If missing data indicators are available, DAMS reduces to covariate shift." ***
WE HAVE THE INDICATORS — the mask is fully observed on both sides, and our attention mask already consumes
them. So, honestly: this theorem PARTIALLY VINDICATES the claim we were told to refute. There is no exotic
missing-data machinery that we are missing; the problem is *plain covariate shift in p(x)*, and the correct
response to plain covariate shift is exactly what we do (reweight/resample the source so its p(x) matches
the target's). Their negative results — "(ii) the optimal linear source predictor can perform arbitrarily
worse on the target domain than always predicting the mean" — apply to the WITHOUT-indicators case, which
is not ours.
BUT THIS IS ALSO WHERE THE HEADROOM IS, AND IT CONVERGES WITH LOG 10/11(b):
if the whole problem reduces to matching the source p(x) to the target p(x), then the QUALITY OF THAT MATCH
is the only thing that matters. Our match is a 2-atom-per-row Monte Carlo approximation to a 24-atom target
window distribution, FROZEN for the whole run. Two independent literatures — DAMS (covariate-shift
matching) and Chen-Dobriban-Lee (orbit averaging) — point at the SAME single intervention: draw the window
fresh every epoch so the source window marginal converges to the measured target window marginal, instead
of being a fixed 2-sample caricature of it. That convergence of two arguments on one cheap code change is
the strongest thing in this brief.
Related, worth one line each: Stokes et al. "Domain Adaptation under MNAR Missingness" arXiv:2504.00322;
"Robust prediction under missingness shifts" arXiv:2406.16484; MIRRAMS arXiv:2507.08280 (tabular, unseen
missingness shifts).

### [log 15] Continuous-time / ODE family — Neural CDE, Latent ODE, and 2023-2025 successors
- Kidger, Morrill, Foster, Lyons. "Neural Controlled Differential Equations for Irregular Time Series."
  NeurIPS 2020 (Spotlight), arXiv:2005.08926. "directly applicable to partially-observed and irregularly
  sampled multivariate time series"; uses adjoint backprop; SOTA "against similar ODE or RNN based models".
  GATE: the comparison class is *ODE/RNN models*, not strong masked Transformers. And: "Continuous-time
  models like Neural CDE are sometimes orders of magnitude slower than transformer models, and Neural CDE
  models are typically only benchmarked in settings where time series are of relatively short length due to
  computational demands." At 25 min/run and 2-3 slots, an adjoint-ODE solve is an EXECUTION-RISK
  non-starter. DOWNGRADE HARD.
- Rubanova, Chen, Duvenaud. "Latent ODEs for Irregularly-Sampled Time Series." NeurIPS 2019. Same family,
  same objection.
- ContiFormer (NeurIPS 2023, openreview YJDz4F2AZu) and Rough Transformers (arXiv:2405.20799) are the
  2023-2024 continuous-time-Transformer successors. Both target LONG, IRREGULAR, SCATTERED series. Our
  series is length 12 and perfectly regular inside the window. The thing these models buy — a principled
  treatment of arbitrary real-valued inter-observation gaps — is worth ~zero to us because our gaps are
  always exactly 1 month.
- Warpformer (KDD 2023) targets multi-scale irregularity in clinical data; same scattered-regime objection.
*** SUMMARY OF THE HARD GATE ACROSS THE WHOLE ISTS FAMILY: every one of mTAND, GRU-D, SeFT, Raindrop,
Neural CDE, Latent ODE, ContiFormer, Rough Transformers, Warpformer is built for SCATTERED, unaligned,
sparse, real-valued-timestamp observation. Not one of them is designed for, or benchmarked on, a
CONTIGUOUS-BLOCK observation window whose length and offset shift between train and test. The competition
under attack ("nothing further to gain from the irregular-time-series literature") is, as regards
ARCHITECTURE SWAPS, essentially CORRECT. I could not refute it on that axis and I am saying so. ***
The refutation lives elsewhere: in the AUGMENTATION/COVARIATE-SHIFT-MATCHING axis (log 10/11b/14), in
positional-encoding length generalization (log 12), and in TTA aggregation (log 8).

### [log 16] Normalization removes information — Zhang, Tozzo, Higgins, Ranganath, ICML 2022
"Set Norm and Equivariant Skip Connections: Putting the Deep in Deep Sets." ICML 2022, PMLR v162,
arXiv:2206.11925.
Surfaced claim: "layer norm can hurt performance by removing information useful for prediction"; the
paper's contribution includes "a novel benefit of such connections, the reduction of information loss."
The mechanism (from the earlier surfaced text on element-wise standardization): standardizing forces "an
invariance where two elements whose activations differ in only a scale yield the same output. This
invariance reduces representation power ... and removes information which may potentially be important."
[the second sentence needs verbatim confirmation from the PDF before it is quoted externally]
=> Supports hypothesis (b) for the duration-normalization loss: normalizing by observed-month count is a
scale invariance imposed by hand; it discards amplitude, and amplitude in a 12-band optical+SAR stack over
a water body is very plausibly discriminative (pond flooding/drawdown magnitude, not just its shape).

### [log 17] Missingness as signal — the transfer warning
"On Missingness Features in Machine Learning Models for Critical Care: Observational Study."
JMIR Medical Informatics (2021), PMC8701717. 48,336 EHRs from the 2012 and 2019 PhysioNet Challenges.
Finding as surfaced: "generally improved model performance in retrospective tasks was observed when
including missingness features"; several studies find missingness features "encode useful information
about a patient's health".
*** BUT THE WORD THAT MATTERS IS "RETROSPECTIVE". *** The gain is measured in-distribution. Missingness
features are the canonical shortcut: they encode the DATA-COLLECTION PROCESS. Our data-collection process
is precisely the thing that differs between train (12/12) and test (4-6/12) — adversarial AUC ~0.99.
CONCLUSION FOR SECTION 6: any feature of the observation window that is computable on train rows is, by
construction, computed on a 12/12 window and therefore CONSTANT (zero variance) in train. It cannot be
fit. Any such feature learned from our MASKED views is learned from a mask we assigned at random,
independent of the label by construction. Therefore, on our data, "missingness as signal" is not merely
risky — for the true (unmasked) train rows it is INFORMATION-FREE, and for masked views it is
label-independent BY CONSTRUCTION. This lane is closed on first principles, not just on caution.
The ONE exception worth stating: the S2 per-calendar-month dropout rates already measured into
`WindowDist.s2_dropout_rates` (src/data.py:60) are a property of CLOUD COVER, which is geographic and
seasonal, and geography plausibly correlates with the label. That is a real (small) channel and it is
already being modelled as part of the masker. Do not build a second, unregularised copy of it.

### [log 18] TTA in time series specifically — the crop-loses-the-event objection
- Shanmugam, Blalock, Balakrishnan, Guttag. "When and Why Test-Time Augmentation Works."
  arXiv:2011.11156 (the pre-ICCV technical report of the ICCV 2021 paper).
- Iglesias et al. / "Data Augmentation for Time-Series Classification: a Comprehensive Survey",
  arXiv:2310.10060. VERBATIM (as surfaced): "The most used data augmentation method for time series
  classification is the slicing window technique, originally inspired by the image cropping technique for
  data augmentation in computer vision tasks. However, for time series data, one cannot make sure that
  the discriminative information has not been lost when a certain region of the time series is cropped."
*** THIS IS THE HONEST OBJECTION TO SUB-WINDOW TTA (proposal P2). *** Cropping a 6-month test window down
to 4 months is not a symmetric, label-preserving nuisance perturbation like an image flip. It can DELETE
the discriminative event (e.g. the single drawdown month that identifies a pond). The resulting TTA average
is then biased DOWNWARD for exactly the positives whose evidence is temporally concentrated — a
label-correlated bias, which is the worst kind. This does not kill P2 but it means P2 must be evaluated on
the masked OOF replica first, and it predicts a specific failure signature (recall loss concentrated in
long-window positives) that we can check for.

### [log 19] INTERNAL EVIDENCE THAT KILLED MY OWN TOP PROPOSAL (recorded for traceability)
- `config/config.yaml:303` — `resample_per_epoch: false  # iter21 instance-expansion: fresh masked
  windows/row each epoch (lam=0 only)`
- `src/seq_model.py:977-992` — the resampler, keyed `cfg["seed"] + _rep + 7919 * (ep + 1)`.
- `src/seq_model.py:981` — the gate: `if bool(s.get("resample_per_epoch", False)) and lam == 0.0:`
- `experiments/LB_LOG.md:1495-1532` — "iter21 RESULT — instance-expansion is inert, and the arbiter failed
  its own retro-fit". Arms c_iexp_rs2 (OOF 0.98205, seed 42), c_iexp_rs2_s7 (0.98292, seed 7),
  c_iexp_rs6 (0.98235, 42), c_iexp_rs6_s7 (0.98384, 7); champion seq_a_xview 0.97523; control
  seq_a_reltime 0.98041; fixed-view twin seq_a_k4 0.98419 with LB 0.8665.
  Verbatim: "Every instance-expansion arm sits at ≈0.982–0.984 OOF — the exact `seq_a_k4` fingerprint...
  Higher OOF has been anti-correlated with LB throughout... The data-model lane (in-family) is now measured
  closed." And: "the offline screen went VOID ... `c_iexp` was never scored" (so: closed on proxy, not LB).
  And: "only model-class-sized effects (~0.05) are measurable with this budget."
- `experiments/LB_LOG.md:1534` — "Lane status after iter21: positional ✅closed, objective ✅closed,
  pooling ✅closed (rank-twins)". This discounts P3/P5.
=> Both my augmentation lead AND my positional-encoding leads were already probed by the loop. The
ARCHITECTURE gate (Sec 1) and the UNIFIED THEORY (Sec 7) are the parts of this brief that are new.
