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
