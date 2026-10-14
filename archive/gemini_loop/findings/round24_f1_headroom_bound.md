# Round 24 — Q2: F1 headroom on a fixed ranking, Lipton under prior shift, and a legal local ROC estimate

Owner: Q2 researcher. Started 2026-08-14. **Written incrementally — content is appended as it is
established. Anything below a `## ` heading is final unless marked PROVISIONAL.**

Labels used throughout:
- **VERIFIED** — I read the primary source (paper / theorem statement) and it says this.
- **DERIVED** — my own arithmetic or proof, done here because the literature is silent. Reproducible
  script paths given.
- **INFERRED** — plausible reasoning from sources that do not state it directly.

---

## Status log

- [x] Read UPDATE_24 in full.
- [x] Read round24_partial_auc.md CORRECTION 0.
- [x] **CORRECTION 1** — the (AUC, F1, n=309) trio is arithmetically impossible. Written below.
- [ ] (A) sharp bound on max-F1 given AUC + one confusion cell
- [ ] (B) Lipton t* = F*/2 under prior shift
- [ ] (C) train-only estimate of the high-recall ROC under the deployment distribution

---

# CORRECTIONS TO THE BRIEF (written first, per §9)

## CORRECTION 1 — 🔴 `AUC = 0.945841814`, `F1 = 0.881720430` and `n_public = 309` are **mutually impossible**. At least one is wrong.

**DERIVED. Reproduce with** `scratchpad/q2_refute.py`, `q2_joint.py`, `q2_scan_n.py` (paths at the
bottom of this section; all pure-`fractions` exact arithmetic, no floating point in the decision).

### 1.1 The test

ROC-AUC on a finite sample is not a free real number. With `P` positives and `N` negatives it is
**exactly** `C / (P·N)` where `C` = concordant-pair count, and `C ∈ ½·ℤ` (an integer with no ties in
the score column; a half-integer if there are ties, since each tied pair contributes exactly ½ under
the Mann–Whitney convention `sklearn.roc_auc_score` uses). So **the reported AUC must be a rational
with denominator `2·P·N`.** That is a very strong constraint when the leaderboard prints 9 decimals.

### 1.2 Step 1 — F1 pins the confusion matrix HARDER than the brief realises, and this part is solid

`F1 = 2·TP/(PP+P)`. Over every denominator `PP+P ∈ [200, 600]`, the values landing in
`[0.881720430, 0.881720431)` are **exactly four**, and they are all the same fraction:

```
PP+P = 279, TP = 123  →  82/93
PP+P = 372, TP = 164  →  82/93
PP+P = 465, TP = 205  →  82/93
PP+P = 558, TP = 246  →  82/93
```

So `F1 = 82/93` **exactly**, and `PP + P = 93k`, `TP = 41k`. With `n = 309` and a test prevalence of
59–62 %, only `k = 4` is admissible (`k=3` forces prevalence ≤ 50 %, `k=5` forces ≥ 66 %). Hence:

> **NEW, and stronger than §3.2:** `TP = 164` **and** `TN = 101` are pinned **exactly and
> independently of P**, because `FP = PP − TP = (372−P) − 164 = 208 − P` and therefore
> `TN = 309 − P − FP = 101` for *every* P. The only free quantity is P, and it trades one-for-one:
> **`FP = 208 − P`, `FN = P − 164`, so `FP + FN = 44` always.**
> The brief presents `TN = 101` as contingent on `P = 191`. It is not — it is invariant. Good news.

### 1.3 Step 2 — but then the AUC cannot be what is reported

Given `F1 = 82/93` exactly and a composite of `0.907368983` truncated at 9 dp, the admissible AUC
window is
```
AUC ∈ [ (0.907368983 − 0.6·82/93)/0.4 , (0.907368984 − 0.6·82/93)/0.4 )
     = [ 0.945841812339 , 0.945841814839 )        width 2.500e-09
```
The brief's `0.945841814` sits inside it, so the brief's *derivation* is fine. The problem is that
**no realisable AUC does.** Sweeping the entire admissible family `P ∈ [164, 208]`, `N = 309 − P`,
and taking for each P the nearest half-integer `C`:

| P | N | FP | FN | nearest realisable AUC | distance from the admissible window |
|---|---|---|---|---|---|
| 188 | 121 | 20 | 24 | 0.945841392650 | 4.20e-07 |
| 190 | 119 | 18 | 26 | 0.945842547545 | 7.33e-07 |
| **191** | **118** | **17** | **27** | **0.945847013932** | **5.20e-06** |
| 192 | 117 | 16 | 28 | 0.945846688034 | 4.87e-06 |
| 193 | 116 | 15 | 29 | 0.945841522244 | 2.90e-07 |
| 196 | 113 | 12 | 32 | 0.945841610981 | 2.01e-07 |
| … | | | | | |
| **best over all 45 values of P** | | | | | **1.907e-07** (at P = 207, FP = 1 — prevalence 0.67) |

**The window is 2.5e-9 wide. The closest any P gets is 1.9e-7 — 76× too far. The brief's own
`P = 191` misses by 5.2e-6, which is 2080× the window.**

A brute-force joint search over *all* `(P, PP, TP, C)` at `n = 309` under both the truncate and the
round-to-9dp display conventions returns **zero solutions** (`q2_joint.py`, `q2_scan_n.py`).

### 1.4 What this means, and what it does not

It does **not** mean the model is different from what the team thinks. It means one of these four
inputs is false, and the team does not currently know which:

| suspect | how load-bearing | my prior |
|---|---|---|
| **(i) `n_public = 309`** | everything in §3.2/§3.3 and both Q1 probes | **highest** — 309 = exactly 30 % of 1030, which smells *inferred* rather than read off Zindi. If Zindi rounds 30 % differently, or drops unscored rows, n moves and every pair count moves with it. |
| (ii) the composite is *not* exactly `0.6·F1 + 0.4·AUC` | the AUC recovery | medium — "verified to 4e-10 on eight submissions" is a fit, and a fit to 4e-10 over 8 points can still hide a 1e-7 model error if the design matrix is ill-conditioned |
| (iii) Zindi's F1 is not the positive-class binary F1 | step 1 | low — `82/93` is far too clean to be an accident |
| (iv) the display is neither truncate nor round-half-up at 9 dp | window width | low — tested both; a wider window (say 1e-7) still leaves only `P ∈ {193,196,207}`, all with implausible FP |

### 1.5 The consequence for §3.3 — and it is smaller than it sounds

Every number in §3.3 (`22538`, `1220.6`, `459`, `761.6`) is computed from the inconsistent premise.
Note the tell that was already visible on the page: **the brief's own "discordant = 1220.6" is not an
integer or a half-integer, and it must be one.** A pair count cannot end in `.6`. That single
observation refutes the premise without any of the machinery above, and it is sitting in §3.3 in
print. Add it to §8 as error #7: *if a count comes out fractional, the inputs are wrong.*

The good news is that the bound in Part (A) below is **structurally** unaffected and **numerically
insensitive** to this: a 5e-6 wobble in AUC is ~0.1 of one pair out of 22538. I therefore compute the
bound as a function of `(P, C)` and show it is stable across the whole admissible family.

### 1.6 Two one-submission controls that settle it — and a correction to Q1's Probe A

**CORRECTION 1a (to `round24_partial_auc.md` §0.2).** Q1's Probe A predicts
`AUC_A = 19321.5 / 22538 = 0.857285562`. **The division is wrong.** `19321.5 / 22538 =
0.857285473…`. The discrepancy is 8.9e-8 — six digits into a nine-digit readout, i.e. exactly the
scale at which a control either passes or fails. Since Probe A's entire value is "a control that
returns the value arithmetic guarantees" (§8 error #3), the guaranteed value has to be right.
**Use 0.857285473.**

**CORRECTION 1b (a strengthening of Probe A).** Q1 presents Probe A as a *confirmation* of
`P = 191`. It is much better than that: it is a **P-meter**. Under the invariant family of §1.2
(`TP=164, TN=101, FP=208−P, FN=P−164`), Probe A's returned AUC is a strictly increasing function of
P with a step of 5e-4 to 2e-3 per unit of P — three to four orders of magnitude above the readout
noise:

| P | 186 | 188 | 190 | **191** | 192 | 194 | 196 |
|---|---|---|---|---|---|---|---|
| Probe-A AUC | 0.851429321 | 0.853525585 | 0.855948695 | **0.857285473** | 0.858707265 | 0.861810847 | 0.865270002 |

**And crucially: if the returned value matches NO row of that table, `n ≠ 309` and Correction 1 is
resolved in favour of suspect (i).** Probe A is therefore not a nice-to-have control; it is the
diagnostic for the inconsistency I just found. Run it.

**PROBE P (new, 1 submission, cheaper and even more direct).** Submit `TargetRAUC` = any constant
(all rows tied ⇒ AUC = 0.5 **exactly**, by definition, no assumptions) and `TargetF1` = **all ones**.
Then `F1 = 2P/(n+P)` exactly and `composite = 0.6·2P/(n+P) + 0.2`. Solving gives `P/n` with no model
and no unknowns. At `n = 309` the predicted composites are spaced ~1.5e-3 apart in P:

| P | 188 | 190 | **191** | 192 | 194 |
|---|---|---|---|---|---|
| `F1 = 2P/(309+P)` | 0.756539235 | 0.761523046 | **0.764000000** | 0.766467066 | 0.771371769 |

**If the recovered `F1` is not `2P/(309+P)` for an integer P, then `n ≠ 309`** — and you can then
solve `2P/(n+P)` over the small set of plausible `(n, P)` to recover both. This probe costs one
submission, touches no knob, and its output is a *count*, which is the least knob-like thing there
is.

> Legality: prongs (a) and (b) both clean — these change no decision rule and set no hyperparameter.
> They are in the same category as the F1 inversion the team already runs and already treats as
> legal diagnosis (§3.2). I would additionally note in the report that they were used only to
> validate arithmetic, never to select a model.

**Scripts:** `…/scratchpad/q2_consistency.py`, `q2_consistency2.py`, `q2_joint.py`, `q2_scan_n.py`,
`q2_refute.py` (session scratchpad; copy them into the repo if you want them in the report).

---

---

# TEAM RESPONSE TO CORRECTION 1 — VERIFIED, AND THEN SOLVED

Written by the operator, 2026-08-16, after independently reproducing the refutation.
Tool: `tools/lb_cell_solve.py` (committed, exact `fractions` arithmetic throughout).

## R.1 The refutation is CONFIRMED, and it is stronger than the researcher claimed

Reproduced independently: the admissible AUC window is 2.5e-09 wide and the closest realisable
`C/(P·N)` over every `P` at `n = 309` is **1.9e-07** away. `P = 191` misses by 5.2e-06.
**`n_public = 309` is refuted.**

One strengthening. The researcher's suspect table ranks *"(ii) the composite is not exactly
0.6·F1 + 0.4·AUC"* at medium prior. **Suspect (ii) is eliminated.** Zindi reports the AUC column
*directly* — we have `AUC 0.945841814` as a printed number, not as something recovered through the
composite. Re-running the sieve against the printed AUC alone, with no composite model anywhere in
the derivation, still returns no solution at `n = 309`. The 4e-10 composite fit was never load-
bearing here. That leaves suspect (i), `n_public = 309`, essentially alone — and the researcher's
instinct about *why* was right: 309 is exactly 30% of 1030, i.e. inferred by us, never read off the
platform.

Also confirmed: **CORRECTION 1a is right**, `19321.5/22538 = 0.857285473`, and the fractional
discordant count `1220.6` in our own §3.3 was a visible tell we walked past. Logged as error #7.

## R.2 The inconsistency is not just detectable — it is INVERTIBLE. We recovered (n, P).

The researcher stopped at "one of four inputs is false and the team does not know which", and
proposed spending a submission (Probe P) to find out. **No submission is needed.** The same sieve
that refutes 309 also solves for `n`, because we hold *five* reported `(AUC, F1)` pairs that must
all be satisfied at the *same* `(n, P)`:

| step | constraint | result |
|---|---|---|
| 2 | five printed AUCs must each be `C/(P·N)`, `C ∈ ½ℤ`; five printed F1s must each be `2·TP/(PP+P)` | **15** candidate `(n, P)` survive out of ~150 000 |
| 3 | `PP_public / PP_full = n / 1030`, with `PP_full` read off the submission CSVs on disk | four independent estimates: **324.0, 326.4, 324.5, 331.6** |

Admissible `n` in the surviving `P = 181` family is `{257, 333, 409, 485, 561, 637}`. The mean
estimate 326.6 selects **`n = 333`** over the runner-up by 70 rows to 6 — a factor of 12. Every
family that step 3 does not reject (`P = 181`, and its ×2 image `P = 362`) carries the *same*
`TP` and `PP`; the rejected ones are `P = 190` at `n = 552`, `P = 380` at `n = 561`, `P = 543` at
`n = 695`, each off by 200+ rows.

> ### 🔴 THE PUBLIC CELL, SOLVED
> `n = 333`, `P = 181`, `N = 152`, **true public prevalence 0.5435**
>
> | submission | TP | FP | FN | TN | precision | recall | realised pos-rate |
> |---|---|---|---|---|---|---|---|
> | champion | 164 | **27** | **17** | 125 | 0.858639 | 0.906077 | 0.5736 |
> | jtt_lam5 | 164 | 26 | 17 | 126 | 0.863158 | 0.906077 | 0.5706 |
> | sub 0.942570 | 163 | 25 | 18 | 127 | 0.867021 | 0.900552 | 0.5646 |
> | presto_frozen | 145 | 41 | 36 | 111 | 0.779570 | 0.801105 | 0.5586 |
> | presto_ft | 145 | 39 | 36 | 113 | 0.788043 | 0.801105 | 0.5526 |

Confirming the researcher's §1.2 invariance point in its corrected form: `TP` and `PP` are fixed
across the entire family, so **the precision/recall figures above do not depend on `n` at all**.
Only `TN` moves. The cell is more robust than the `n` that indexes it.

## R.3 ⛔ WHAT THIS COSTS US — precision and recall were swapped, and round 24 aimed at the wrong side

We carried **precision 0.9061 / recall 0.8586** — "27 positives missed, we have a recall deficit".
The truth is the exact mirror: **precision 0.8586 / recall 0.9061 — 27 false positives against 17
misses.** Our dominant error mode is false positives, by 1.6 to 1.

Consequences, stated plainly because they are expensive:

1. **The round-24 framing is aimed at the smaller half of the error budget.** The "high-recall
   corner", the partial-AUC survey (Q1), JTT-to-recover-missed-positives (iter49) — all of it
   targets the 17, not the 27. This does not make Q1's method verdicts wrong (they stand on the
   order-invariance argument, which is untouched), but it does relocate where a win could come from.
2. **"Our operating pos-rate may be too LOW" is refuted, not open.** `LB_LOG` iter43 left this
   question open after MLLS/BBSE were retired at iter41. It is now closed in the opposite
   direction: we predict 0.5736 positives against a true 0.5435. **We over-predict.** Any lane whose
   motivation was "push more rows above the cut" is dead on arrival, and the plan-file graph-gate
   reading ("the graph says our pos-rate is roughly right, ~0.59") was reading an estimator that
   agreed with our error rather than with the truth.
3. **iter49's conclusion survives intact.** JTT: `TP` identical at 164, one FP removed. Unchanged
   under the corrected cell — that inversion only ever used `PP+P`, which is `P`-independent.
4. **`tools/roc_probe.py` was built on the refuted cell** and has been corrected (`P_PUBLIC` 191 →
   181, `N_PUBLIC` 118 → 152). Probe A's guaranteed control value is now **0.864222885**, not
   0.857285473 — both the researcher's figure and our accepted erratum were computed on the wrong
   partition. Probe B is demoted to secondary in the file: the mirrored probe (measuring how many
   of the 27 FPs sit just *above* the cut) is now the more valuable instrument for the same cost.
5. **Probe P is no longer needed.** Its job — recover `P` without model assumptions — is done above
   for zero submissions. Keep it in the report as the design that *would* have worked; do not spend
   a slot on it.

## R.4 Credit

The refutation is the Q2 researcher's. We had `n = 309` in print in three documents and two tools
and had never tested it. The test they applied — *a reported score is a rational with a known
denominator, so check whether it is even reachable* — is the single most productive move of the
round, and it generalises: **any leaderboard that prints enough digits is an exact measuring
instrument, and we should have been sieving against it since iter42.**

---
---

# RESUMED 2026-08-16 — PARTS A–D ON THE **CORRECTED** CELL

Everything above the "TEAM RESPONSE" divider was written against `n=309, P=191`. The cell is now
solved: **`n=333, P=181, N=152`; champion `TP=164, FP=27, FN=17, TN=125`, `PP=191`.**
Precision `0.858639`, recall `0.906077`. We **over-predict** (0.5736 vs 0.5435).
All four parts below are recomputed on that cell. Reproduce with `tools/f1_headroom_bound.py`
(exact `fractions`, no float in any decision).

---

## PART A — the sharp max-F1 bound on a fixed ranking. **VERDICT: the leader is INSIDE. Our ranking is NOT provably deficient.**

**DERIVED.** `tools/f1_headroom_bound.py`.

### A.1 The AUC readout pins the concordant-pair count to a single integer

`P·N = 181 × 152 = 27512`. `AUC = C/(P·N)`, `C ∈ ½ℤ`. Over the 9-decimal window
`[0.945841814, 0.945841815)` there is **exactly one** admissible half-integer:

```
C = 26022   (an integer — no ties in the champion score column)
AUC = 26022/27512 = 0.945841814481…
D  = 27512 − 26022 = 1490 discordant pairs
```

This is worth stating on its own: the leaderboard has handed us an **exact pair count**. There is no
estimate anywhere in Part A.

### A.2 The block decomposition — the entire degree of freedom is 1031 pairs

Split the 27512 (positive, negative) pairs by which side of the 0.5 cut each member falls on:

| block | size | status |
|---|---|---|
| TP × TN | 20500 | **all concordant** — forced, the TP is above the cut and the TN below it |
| FN × FP | 459 | **all discordant** — forced, same argument mirrored |
| TP × FP | 4428 | free; write `d_a` = # discordant (an FP outscoring a TP, both above the cut) |
| FN × TN | 2125 | free; write `d_b` = # discordant (a TN outscoring an FN, both below the cut) |

`20500 + 459 + 4428 + 2125 = 27512` ✓, and

> **`d_a + d_b = D − FN·FP = 1490 − 459 = 1031` exactly.**

`d_a` measures how badly our 27 false positives are interleaved *upward* among the 164 true
positives; `d_b` measures how badly our 17 misses are buried *downward* among the 125 true
negatives. Threshold movement can only help to the extent one of these is small. **1031 is the whole
budget, and the AUC column fixes it to the pair.**

### A.3 The sharp upper bound

Parametrise a threshold *raise* by `(m, u)`: cross above the `m` lowest FPs, which necessarily also
drops the `u = u_m` TPs sitting below the `m`-th lowest FP. Parametrise a *lower* by `(j, s)`: admit
the `j` highest FNs, which drags along the `s = s_j` TNs above the `j`-th highest FN.

```
F1_raise(m,u) = 2(164−u) / (372 − m − u)          F1_lower(j,s) = 2(164+j) / (372 + j + s)
```

Maximising over all feasible arrangements:

```
 RAISE ceiling  m=27, u=0  →  328/345 = 0.950724638      ← the max
 LOWER ceiling  j=17, s=0  →  362/389 = 0.930591260
```

> ### 🔴 **SHARP UPPER BOUND ON max-F1 BY THRESHOLD MOVEMENT = 328/345 = 0.950724638**
> Attained by exactly one configuration: **all 27 false positives ranked below all 164 true
> positives**, and the cut raised past them.

Sanity check that the AUC is not the binding constraint at the maximiser: the maximum concordant
count for *any* monotone ROC staircase through `(FP=27, TP=164)` is
`0·164 + 27·164 + 125·181 = 27053`, and `27053 ≥ 26022 = C`. **The AUC constraint has slack at the
optimum, so the bound is set by the confusion cell, not by the AUC.** (This corrects the old
`[0.8817, 0.9574]`: the upper end moves down to `0.95072`, and — importantly — the reason it is
where it is has nothing to do with the AUC.)

### A.4 The verdict, which is the number the round was asked for

The leader's composite is known only as ≈ 0.929–0.936 with `AUC 0.944897`, so
`F1_leader = (composite − 0.4·0.944897)/0.6 ∈ [0.9184020, 0.9300687]` (Part D refines this).

```
0.881720430  (us, now)
0.918402  ─┐
0.930069  ─┴─  leader's whole plausible F1 band
0.950725     ← our sharp ceiling
```

> ## ✅ **THE LEADER'S ENTIRE F1 BAND IS INSIDE OUR BOUND, with 0.0207 to 0.0323 of margin.**
> **Our ranking is NOT provably deficient. No threshold-reachability argument can explain the gap.
> The entire loss is where 0.5 lands.**

That is the yes/no the brief asked for, and it points the remaining effort at the operating point,
not at the ordering. It is also consistent with the AUC column: our AUC (0.945842) already *beats*
theirs (0.944897), so a claim that our ordering is worse would have had to be an ordering defect
invisible to global AUC but fatal locally. It is not there.

### A.5 The honest other half — the bound does NOT prove the leader's F1 is reachable

The 1031-pair budget is nowhere near tight enough to *force* headroom. An adversarial arrangement
consistent with the exact cell **and** the exact AUC can block any target:

| target F1 | `d_a` needed to block all raises | `d_b` needed to block all lowers | total | share of the 1031 budget |
|---|---|---|---|---|
| 0.8817205 (our own — i.e. block *any* gain) | 311 | 203 | **514** | 49.9 % |
| 0.90 | 174 | 85 | 259 | 25.1 % |
| **0.9184020 (leader, low)** | **74** | **16** | **90** | **8.7 %** |
| 0.9300687 (leader, high) | 33 | 1 | 34 | 3.3 % |
| 0.94 | 10 | 0 | 10 | 1.0 % |

Read the first row: an adversary needs only **half** of the available discordance to make our
current 0.8817 the *global* max over every threshold. So the guaranteed lower bound on our reachable
max-F1 is exactly the F1 we already have.

> **The interval, stated correctly and sharply: our max-F1 over all thresholds lies in
> `[82/93, 328/345] = [0.881720, 0.950725]`, and the AUC column cannot narrow it further.**
> Any narrowing must come from data we hold — see Part C.

### A.6 The two concrete moves that would match the leader — and how small they are

| route | to reach F1 ≥ 0.918402 | to reach F1 ≥ 0.930069 |
|---|---|---|
| **RAISE** the cut | the lowest **15** of our 27 FPs must sit below every TP | the lowest **20** of 27 |
| **LOWER** the cut | all **17** FNs sit at the top of the below-cut group with ≤ **5** TNs interleaved | ≤ **0** TNs interleaved |

Neither is exotic. 15 of 27 FPs being the bottom 15 rows of a 191-row predicted-positive set is a
*weak* ordering requirement. This is why the round's conclusion should be read as **"the ordering is
fine and the cut is in the wrong place"** rather than "we need a better model".

⚠️ **And it is precisely the RAISE route that is available and the LOWER route that is not.**
We over-predict: 191 predicted positives against 181 true. The move that helps is *fewer* positives,
which is a legal direction only if it arrives via a corrected `p(y|x)` and not via the cut. Part B.

---

## PART D — the leader's cell, BOUNDED (done here, out of order, because it sets the target Part A is compared against)

**DERIVED.** `tools/leader_cell_bound.py`.

### D.0 First, a methodological correction: their AUC is **not** sieve-testable

Our AUC came printed at **9 dp**; theirs at **6 dp**. At `P·N = 27512` consecutive achievable AUCs
are `1/(2·P·N) = 1.817e-05` apart, so a 6-dp window (`1e-06`) is **18× narrower than the spacing**.
The window in fact contains no half-integer at all — and **that is not evidence against anything**;
it is the expected outcome ~94 % of the time. The refutation machinery of §1 works only because
Zindi gave *us* nine digits.

> **Rule to carry into the report:** the sieve is only valid when
> `10^(-digits) ≳ 1/(2·P·N)`. At `P·N ≈ 2.8e4` that means **≥ 5 significant decimals minimum, and
> 9 dp to be decisive.** State the digit count whenever the sieve is invoked. (Add to §8 as error #8:
> *we nearly ran the sieve on a 6-dp number and would have "refuted" a true cell.*)

We therefore use their AUC only as a *value*: `C_leader ≈ 25996` against our `26022`. **Their ordering
is 26 concordant pairs worse than ours** — confirming from the other side that the gap is not AUC.

### D.1 The feasible set

`F1_leader = (composite − 0.4·0.944897)/0.6 ∈ [0.918402, 0.930069]`. Enumerating every integer cell
at `P=181, N=152` inside that band, and additionally requiring the cell to be AUC-consistent
(`TP·TN ≤ C_leader ≤ TP·N + FN·TN`, the Part-A block decomposition applied to them), leaves
**121 cells**:

| quantity | leader's feasible range | mean over the 121 | **us** |
|---|---|---|---|
| precision | 0.8498 – 1.0000 | 0.9209 | **0.858639** |
| recall | 0.8508 – 1.0000 | 0.9311 | **0.906077** |
| FP | 0 – 32 | 15.2 | **27** |
| FN | 0 – 27 | 12.5 | **17** |
| **FP + FN** | **24 – 32** | 27.7 | **44** |
| PP | 154 – 213 | 183.7 | **191** (truth 181) |

> ### 🔴 THE ONE NUMBER: **they make 24–32 total errors. We make 44.**
> That range is *tight* — `FP+FN = 2·TP·(1−F1)/F1` is nearly TP-independent across the band — so the
> gap is **12 to 20 mistakes on 333 rows**, and it is robust to everything unknown about them.

### D.2 Precision, recall, or both?

| | count / 121 | share |
|---|---|---|
| beats us on **precision** | **117** | **97 %** |
| beats us on **recall** | 80 | 66 % |
| beats us on **both** | 76 | 63 % |
| beats us on precision only | 41 | 34 % |
| beats us on recall only | **4** | **3 %** |

Mean advantage: **precision +0.0623, recall +0.0250.** So:

> **They almost certainly beat us on precision (97 % of the feasible set), probably also on recall
> (66 %), and the bulk of the advantage — 2.5× by mean, and the only near-certain half — is
> PRECISION.** The four cells where they beat us on recall but not precision are the degenerate
> `recall → 1.0` corner (`TP=181, FP=32`), which requires them to have found *every* positive.
>
> This lands on exactly the same side as R.3. Our dominant error mode is false positives, and so is
> the leader's dominant *advantage*. The two independent readings agree.

Their mean `PP` is **183.7** against a truth of 181 — essentially calibrated — while ours is **191**.
66 of 121 cells over-predict, 53 under-predict; they straddle the truth, we sit 10 rows above it.

### D.3 Checking the operator's guessed cell — **very nearly right, and the miss is instructive**

> `TP=173, FP=18, FN=8, TN=134` ⇒ `PP=191`, `F1 = 173/186 = 0.930108`, **composite = 0.936023**.

That is **2.3e-05 *above* 0.936**, i.e. it sits a hair outside the stated band and is the extreme
top corner of it. As a cell it is entirely plausible — if their composite is 0.9360 this is close to
it. Two caveats:

1. It silently assumes **`PP = 191`, the same predicted-positive count as ours.** Nothing forces
   that, and it is the least likely part of the guess: their feasible mean `PP` is 183.7, and 191 is
   the 74th percentile of their feasible `PP`. Drop that assumption and `TP` is not pinned at all.
2. Under that assumption it implies precision 0.9058 / **recall 0.9558**, i.e. a *recall*-dominated
   advantage. That is the minority reading (the 3 % corner direction). Across the full feasible set
   the advantage is precision-dominated. **So the guess's arithmetic is right to within a rounding
   edge, but its qualitative conclusion is the atypical one, and it inherits that from the `PP=191`
   assumption rather than from the data.**

**Corrected headline:** the defensible statement is `FP+FN = 24–32 vs our 44`, precision advantage
near-certain, recall advantage likely; **not** a specific `(173, 18, 8)`.
