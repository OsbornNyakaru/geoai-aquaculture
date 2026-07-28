# RESPONSE_13 — Claude Deep Research (round 12), triaged 2026-07-28

**Verdict: the strongest research round this project has had.** It refuted three of my own proposals
correctly, produced the best feature idea anyone has offered, and its headline claim is **verified
true against the live rules page**. Contrast with `RESPONSE_12.md` (Gemini), which was largely our own
brief handed back with two factual errors.

---

## 1. 🔴 R1 — VERIFIED. The prevalence pin is a rules violation, and so is our RAUC column

**Independently confirmed 2026-07-28** by fetching two Zindi pages (rules, and the competition
overview). Both carry the text verbatim:

> **"Setting a probability threshold is strictly forbidden. Your binary target should be based on the
> default threshold of 0.5."**
>
> "If the error metric requires probabilities to be submitted, do not set thresholds (or round your
> probabilities) to improve your place on the leaderboard."
>
> "In order to ensure that the client receives the best solution Zindi will need the raw
> probabilities. **This will allow the clients to set thresholds to their own needs.**"

That last sentence is the rule's *rationale*, and it is what our pipeline defeats.

### 1a. `TargetF1` — our own code comment admits the mechanism

`src/calibration.py:50-66`, `target_prevalence_shift`:

```python
z = _logit(p)
thresh = float(np.quantile(z, 1.0 - pi_hat))   #  <-- a threshold on the logits
delta  = -thresh
return _sigmoid(z + delta), delta               #  <-- shifted so that threshold IS 0.5
```

The docstring says it outright: *"hitting an EXACT target prevalence **is a threshold on the
logits**."* We select the cut, then move the probabilities so the cut lands at 0.5. The literal 0.5
in `(p_test_final >= 0.5)` is cosmetic.

**And the aggravating fact is in our own ledger:** π̂ = 0.649 was **not derived from training data.**
Round 02 swept the prior *against leaderboard feedback* — `0.7561 → peak 0.8260 @ realized ~0.65` —
and we kept the peak. A value fitted on LB feedback and used to place a decision boundary is
threshold tuning under any reading. Elkan (IJCAI 2001) makes the equivalence formal: for calibrated
models, cost-sensitive reweighting and threshold shifting are the same operation.

### 1b. 🔴 `TargetRAUC` is ALSO non-compliant — the research did not catch this, and it is free to fix

`src/calibration.py:78-82`:

```python
def score_for_auc(p_raw):
    order = np.argsort(np.argsort(p))
    return (order + 0.5) / len(p)      # uniform ranks in [0,1] — NOT probabilities
```

We submit **uniformly-spaced ranks**, not probabilities. A client setting a threshold of 0.8 on our
column receives "the top 20% of rows," which carries no probabilistic meaning. This defeats
*"Zindi will need the raw probabilities … to set thresholds to their own needs"* as squarely as the
F1 pin does. Our README even states we chose maximal rank spread *to protect AUC* — i.e. the
representation was chosen for leaderboard reasons, which is the pattern the rule names.

**The good news: this fix is free.** ROC-AUC is invariant to any strictly monotone transform, so
submitting genuine calibrated probabilities instead of ranks **cannot change our AUC score at all.**
Zero cost, removes one violation. There is no argument for not doing it.

### 1c. The pin is self-defeating — this is the decisive argument

Final score = 65% private LB + **35% code review of the top 5 only.**

- If we do **not** reach the top 5, the pin was never worth anything — we win nothing either way.
- If we **do** reach the top 5, our code is read by exactly the people who wrote that rule.

**The +0.07 is only cashable in the precise scenario that triggers the review that would void it.**
It is not a risk/reward trade; there is no branch where keeping it pays.

**One correction to the research's framing:** it treats the pin as worth ≈+0.07 today. That figure is
inherited from round 02 and was measured **on the GBDT model class**, before the transformer existed.
Its value on the current model is **unmeasured**. The true cost of going legal may be materially
smaller than 0.07 — and measuring it is one submission.

---

## 2. ✅ Three correct refutations of MY OWN proposals

**(a) Top-k / push losses are the wrong regime — my Q1 was wrong.** I proposed p-norm push,
LambdaRank and Precision@k surrogates for the F1 column. Boyd, Cortes, Mohri & Radovanovic
(NeurIPS 2012) verbatim: *"no generalization guarantee is available for such precision@k
optimization,"* and they recommend optimizing a top *fraction*. **Our cut sits at k/n ≈ 0.649 — the
middle of the score distribution, not the top.** Top-heavy losses are designed for small-k and are
actively wrong here. That closes the whole learning-to-rank family I opened. Accepted.

**(b) My "diversity is a liability" law is NOT identified.** Correct, and I should have seen it.
Both members in my n=2 evidence were **weaker** (ROCKET −0.040, GBDT −0.011), so member strength and
member diversity are **perfectly confounded**. What I demonstrated is that *weak* members hurt — which
my own corrected gate ("level gap, not correlation") already concedes. The stronger form I wrote into
`REPORT.md` §6 overstates the evidence and is being softened. The new observation is genuinely mine
to have missed: **a diverse member can hurt the F1 SET while helping the AUC RANKING, and we have
never measured the two columns separately.**

**(c) The Fisher-z CI and the ATC citation** are handled correctly and match our own numbers.

## 3. 🟢 R4 — the permanence indicator: the best feature idea of the round

```
c(t) = 1[ VH_dB(t) < τ ]        τ ≈ −21.5 dB, swept −22 … −19, in RAW dB pre-standardization
```
replacing the `blue` channel (keeping width at 24).

Why it clears every constraint we have imposed:

| our constraint | status |
|---|---|
| capacity-neutral, channel-**replacing** | ✅ swaps blue, width unchanged |
| **n-invariant (Class A)** | ✅ **by architecture** — masked mean-pool of a binary channel *is* the fraction-of-observed-months-below-τ, a fixed-degree statistic unbiased at every window length |
| not algebraically spanned | ✅ an indicator is **not** affine in the bands, so our SDWI/AWEI/EVI/NDWI deadness proof does not reach it |
| amplitude-preserving | ✅ it *encodes* level rather than removing it |
| targets the one physics leg we have | ✅ temporal permanence |

τ is anchored in primary SAR literature, not guessed: Xing et al. (Dongting Lake, PMC6015492) put the
optimal land/water split at **VH = −21.56 dB** with >95.5% overall accuracy; Kumar et al. (*Sci. Rep.*
2025) report flooded-rice VH minima of −22.03 to −17.69 dB.

Honestly caveated by the research itself: amplitude is already our primary learned signal, so the
indicator may be largely redundant with what the encoder extracts from standardized VH. **Expect
small.** It also correctly rejects my `max_t[(VH−VV)(t+1) − (VH−VV)(t)]` candidate as **Class-B and
n-dependent** — my own rule, which I violated when I proposed it.

⚠️ **But note our iter26 lesson applies:** this is a **representation change**, so ATC-F1 is
out-of-family and cannot gate it. Screen it on adversarial AUC (free) and the regime-matched CV, not
on the ATC-F1 vote.

## 4. ⚠️ R2 (regime-matched CV) — build it, but expectations stay low

Recommended as the highest-value legal deliverable. It is **free**, and our current CV is
*anti-correlated* with the LB, so almost anything beats it. Build it.

But weigh the direct evidence higher than the analogy: **`sdv`, the actual leader in this actual
competition, tried a test-regime-mimicking validation and reported it "barely correlated."** The
research cites AmbrosM (TPS-Nov-2021) where matched CV *did* work — a different competition with a
different shift structure. Ours is a *designed period/region* shift with adv-AUC 0.8915 on the values
after regime matching. Build it because it is free and our current instrument is worse than useless;
do not plan around it succeeding.

## 5. ✅ Rejected on the rules — LB probing

I asked (sceptically) in Q3 whether the public LB's 309 rows could be used as proxy labels. The
answer is a clean no: metric-decomposition probing needs degenerate `TargetF1` columns, which collide
with the "raw probabilities" and "spirit of the challenge" clauses **and would be visible in the very
code review we would be probing our way into.** Same self-defeating structure as §1c. Rejected, and I
agree.

## 6. Numbers to treat as assumptions, not findings

- **"AUC ≈ 0.93 ⇒ the remaining AUC lane is ≤0.028 ⇒ ≥75% of the gap must come from the F1 set."**
  The *decomposition* is sound and useful. But **we do not know our test AUC** — we only know
  `0.6·F1 + 0.4·AUC = 0.8955`, which does not identify either term. The 0.93 is assumed. The
  qualitative conclusion (top teams are out-*selecting*, not out-*ranking*) is plausible but rests on it.
- **Private top-5 threshold ≈ 0.90–0.92** — explicitly flagged as wide-uncertainty. Treat as a
  planning figure only.
- **Thread 33912's interior and any organizer answer** — the research could not read them
  (login-walled) and says so. Our "organizers have not answered" remains *our* observation.

---

## What changes

1. **`REPORT.md` §7 is now wrong** and is being rewritten. It currently calls our reading
   "defensible but untested." It is **not defensible** — the rule text is explicit and we have now
   read it directly.
2. **Fix `score_for_auc` regardless of any other decision.** Free, and removes one violation.
3. **The `TargetF1` de-pinning is a strategic call with a real LB cost** — escalating rather than
   deciding unilaterally, though §1c is a strong argument that there is no branch where keeping it pays.
4. **Post to forum thread 33912.** An organizer answer would settle this; the thread exists and is
   unanswered.
5. iter27 candidate list: **R4 permanence indicator** (screened on adversarial AUC, *not* ATC-F1),
   plus the two-column separation.
