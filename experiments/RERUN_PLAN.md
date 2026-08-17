# Post-competition re-run plan — resurrecting ARM T, and testing our own recommendations

**Status:** unofficial slots. The competition closed 2026-08-16; final standing (#120/500, private
0.910686008) is locked and nothing here changes it. These runs exist to convert three *arguments*
into *measurements* before the next competition.

Everything below is pre-committed: the reads are written down **before** the scores arrive, per
recommendation 7 of `POST_MORTEM.md` ("declare nulls out loud"). If a read comes back null, it gets
logged as a null.

---

## 0. Why this round exists

The post-mortem left exactly one question indeterminate, and it is the expensive one:

> Would ARM T (`tcons`, the `Var_k(logit)` cross-view penalty on the unlabeled test rows) have
> beaten our finalist if we had not killed it?

We could not answer it offline because **the tcons bundles were never copied back from Colab**.
`submissions/preds/` is gitignored and Colab Cell 5 downloaded only CSVs, so every `.npz` from that
lane died with the VM. Agent A's Result 7 had to bound the counterfactual from leaderboard cells
alone and got composite ∈ [0.8677, 0.9282] — a band that straddles the 0.910686 we actually scored.
A single number there would have been invented.

That gap is regenerable with **compute, not submissions**.

It also matters for a second reason. Agent A measured that the choice of combiner (probability
average / geometric-mean-of-odds / pool-then-calibrate) changes the shipped binary column by **0–2
rows out of 1030** — bit-identical on `dpa` and `amix`. But every family it could test is a family
*without* the variance penalty. ARM T is the one lane whose whole mechanism is logit compression, so
it is exactly the family where that null might not hold, and exactly the family we could not check.
**Do not generalize the 0–2 row result to tcons until these bundles exist.**

---

## 1. Compute stage — zero slots

Run `colab_rerun.ipynb`. **25 runs in three stages.** Every flag is taken from
`experiments/reproduce_champion.sh`, which is the authoritative recipe — not reconstructed from
memory.

The base is not the committed config default. `run_pipeline.py` defaults to `--model gbdt` (the
superseded baseline) and the committed config reproduces the original **24**-channel model; the
permanence channel is switched on explicitly, and ARM T ran **on the permanence champion**:

```
PERM = --set seq.channels.permanence=true --set seq.channels.cdf_taus=[-21.0]     # -> 25 channels
```

| stage | tag | runs | arm |
|---|---|---|---|
| A | `perm_single_s<N>` → `champion_perm_seedavg5` | 5 | the teacher. Scored **public 0.899882**; a checkpoint on the way. |
| B | `tcons_s<N>` | 10 | ARM T = `PERM` + `seq.transduct.enable=true`, `lambda_u=0.5` |
| C | `distill15_s<N>` | 10 | ARM D = `PERM` + `seq.distill.enable=true`, `alpha=1.5`, teacher from stage A |

Seeds: `42 7 13 21 29` (stage A) and `42 13 7 21 29 3 11 17 31 37` (B, C). `seq.transduct` keeps its
iter41 defaults `K_u=2, min_len=4, warmup_frac=0.5`, so stage B is a reproduction, not a new arm.

**Stage A is not optional.** `seq.distill.enable=true` hard-exits without
`seq.distill.teacher=<preds .npz>` (`src/seq_model.py:1060`), and one round of distillation only —
the teacher is always the *non-distilled* pool.

**Width fingerprint, checked automatically by the harness:** every run must log
`seq input width: 25 channels/month`. A tcons run logging 24 means `PERM` did not attach and the run
is **invalid**, not merely different.

**The one thing that must not go wrong again: copy `submissions/preds/*.npz` to Drive before the VM
dies.** That single missing step is why this round is necessary at all. The notebook does it after
*every* run, and re-running the cell skips whatever is already in Drive.

Rough cost: 25 runs × 5 folds × 60 epochs. Minutes each on a T4; transduct is slower because the
1030 test rows join every batch. Budget ~2 h on GPU. Set the Colab runtime to GPU.

### Pre-flight, run locally 2026-08-17 — and one thing it found

A smoke run of the exact ARM T command line confirms the plumbing: `seq input width: 25
channels/month`, `mode=legal t_star=0.5000`, bundle written. The config is sound.

It also printed this, which matters more:

```
TRANSDUCTIVE GATE FAILED: submitted pos-rate 0.6223 outside [0.50, 0.62]
DO NOT SUBMIT THIS ARM; halve lambda_u / alpha and rerun.
```

That gate predates all of this analysis, and applying it to the *known* iter41 cells is the finding:

| artifact | predicted positives / 1030 | pos-rate | gate `[0.50, 0.62]` |
|---|---|---|---|
| `tcons_s13` | 575 | 0.5583 | **PASS** |
| `tcons_s42` | 601 | 0.5835 | **PASS** |
| `tcons` 5-seed pool | 643 | 0.6243 | **FAIL** |

**Both members pass. Only the pool fails.** So we already owned an instrument that isolated the
defect to the *pooled artifact*, months before `assert_pool_sane` existed — and it is an independent
confirmation of the operating-point-escape thesis, arrived at from a completely different direction.
The iter41 error is now stateable in one line: **the gate correctly condemned the pool, and we
responded by killing the arm** — discarding two members that both passed it, one of which was our
best public score of all 91 and the other our best private.

(The 0.6223 in the smoke run itself is not evidence about the full runs — smoke is a 300-row
subsample at `K=1, R=1`. The full members landed at 0.5583 and 0.5835.)

**Consequence for S2, stated in advance rather than discovered later:** `tcons_seedavg10` will
probably trip this gate again and print `DO NOT SUBMIT THIS ARM`. We upload it anyway, deliberately.
Testing whether that verdict was right *is* the experiment, and this is an unofficial slot with
nothing at stake. If the gate fires and the pool underperforms its members again, two independent
instruments have now been validated. If it fires and the pool scores well, the band `[0.50, 0.62]`
is too tight and both instruments need widening. Either way we learn something; overriding it
silently would have taught us nothing.

## 2. Offline adjudication — still zero slots

Once the bundles are local, the questions Agent A had to leave open become arithmetic:

1. Re-run the four combiners on the **real compressed-logit family** (`tools/repool.py`). Does the
   0–2 row null hold on tcons, or does it break exactly where the mechanism predicts?
2. Run `src/calibration.py::assert_pool_sane` on the live tcons pool. It trips on the *replay* at
   +24 rows past the widened member envelope. Does it trip on the real thing? This is the first
   test of that guard against data it was not calibrated on.
3. Pick the member with the highest **OOF AUC** — and only OOF AUC. Recommendation 3 says delete OOF
   F1 and the offline composite from the selection path because they are anti-predictive
   (ρ = −0.420, P = 0.017). Record what each of the three criteria *would* have picked, before any
   upload, so the comparison is honest.

## 3. Slot spend — 5 uploads, in this order

Ordered so that a stop at any point still leaves a coherent result. **S1 is a control and must go
first**; if it fails, every other number this round is uninterpretable.

| # | artifact | question | pre-committed read |
|---|---|---|---|
| **S1** | `tcons_s42` (reproduction) | Does the Colab pipeline still reproduce iter41? | Public **0.914179 ± 0.005** ⇒ pipeline sound, continue. Outside that ⇒ check the 25-channel width fingerprint *first*, then the teacher and seed set; > 0.02 is not noise (see below). |
| **S2** | `tcons_seedavg10` | The resurrection. Was killing ARM T the mistake? | ≥ **0.9207** private ⇒ the arm held more than our oracle and the iter41 kill cost ~40 places. 0.9107–0.9207 ⇒ real but smaller. < 0.9107 ⇒ the kill was *right for the wrong reason* — log it as such. |
| **S3** | `tcons_` best member by **OOF AUC alone** | Does our new single-criterion rule actually pick a good member? | Beats the 10-seed pool ⇒ recommendation 9 (always seed-average) is wrong for this family. Lands within the member range ⇒ OOF AUC is at least not anti-predictive. Lands at the *bottom* of the member range ⇒ criterion R3 fails prospectively and must be retracted. |
| **S4** | `distill15_seedavg10` | Is seed-averaging the lever we claimed? | vs the 5-seed `champion_distill_a15_seedavg5` at public **0.910837**. Within-arm seed SD is 0.004868, so expect a *small* gain; > +0.006 is suspicious, negative retracts recommendation 9. Note iter42 already found the 5-vs-10-seed pools at α=0.7 returned **bit-identical AUC** — the whole difference was one row crossing the cut — so the honest prior here is a near-null on public and a small real gain on private. |
| **S5** | whichever artifact `assert_pool_sane` **flags** — upload it deliberately | Is the guard a true positive or a false alarm? | Guard fires **and** the artifact scores below its members ⇒ guard validated on live data. Guard fires and it scores fine ⇒ the tolerance is too tight and the docstring's "fit to one positive example" caveat has bitten; widen or retract. If nothing fires, spend S5 on an equal-weight `tcons` + `distill15` pool (the cross-family blend we never tried). |

### Why S1's tolerance is ±0.005 and not ±0.001

Measured from our own ledger, not assumed. Two artifacts were re-run from an identical recipe and
re-uploaded, and the ledger holds both scores under the same filename:

| file | public spread | private spread |
|---|---|---|
| `submission_champion_archblend4.csv` | **0.004999** | 0.003715 |
| `submission_champion_dualpolmix10_regimematch.csv` | **0.003078** | 0.000577 |

(Two *other* filenames appear twice with byte-identical scores — those are genuine re-uploads of the
same file, which is how we know these two are real re-runs.)

So **same recipe, same seeds, different session ⇒ up to 0.005 on public.** We do not set torch
deterministic algorithms on the GPU seq path, and `reproduce_champion.sh` says so. An earlier draft
of this plan set the S1 abort bar at 0.001, which would have halted the round on ordinary
non-determinism. The bar that matters is `reproduce_champion.sh`'s: a drift of **~0.02 is not seed
noise** and means something structural differs — width, teacher, alpha, or seed set. Check the width
fingerprint first.

Note also what this implies for S2–S4: a difference smaller than ~0.005 between any two artifacts
here is **not** interpretable as a difference between arms. Combined with the public↔private shift
sd of 0.0121, single uploads resolve very little, which is exactly why the reads above are written
as coarse bands rather than point comparisons.

### Slots deliberately NOT spent

- **Combiner variants** (rank-space, logit-average, pool-then-calibrate). Agent A measured these at
  0–2 rows apart. If §2.1 confirms that holds on tcons too, uploading them would burn four slots to
  measure noise. If §2.1 shows tcons *breaks* the null, that changes and they become S6+.
- **Anything selected using the true prevalence 0.5437.** Diagnostic comparison only — the standing
  rule survives the competition's end, because the point of this round is a transferable lesson and
  a lesson selected on the answer transfers nowhere.
- **Threshold tuning.** `compliance_mode: legal`, literal 0.5, unchanged.

---

## 4. Standing constraints (unchanged)

- Never commit `.csv` or `.npz`. Check `git status --short` for **both** before every commit.
- Only supplied competition data. Pretrained weights legal; no external data; no AutoML.
- Seeded and reproducible: two consecutive runs must print identical `final_oof`.
- The cut is a literal 0.5. The true prevalence may be **compared** against, never **selected** on.
- No per-row test-label reconstruction from leaderboard cells.

## 5. Log the results here

Append a row per upload with the filename, public, private, AUC and F1 sub-columns, and — the part
that matters — **whether the pre-committed read above was met**. The failure mode this whole
document exists to prevent is deciding what a number meant after seeing it.
