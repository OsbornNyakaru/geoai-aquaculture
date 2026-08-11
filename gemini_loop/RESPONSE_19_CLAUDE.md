# RESPONSE_19_CLAUDE — triage of the Claude Deep Research report (Round 19)

Source: `C:\Users\ADMIN\Downloads\compass_artifact_wf-c8b771bc-1d23-5bb5-833d-eb087b357a77_text_markdown.md`,
produced against `gemini_loop/UPDATE_19.md`. Triaged 2026-08-11, iter43 uploads pending.

**Screening rule (from `RESPONSE_19_CHATGPT.md` §3): PASSED.** The report reproduces our actual
ledger — tcons pool 0.8718 vs members 0.9013/0.8975, pooled AUC 0.9267 between members, pos-rate
0.5883, the 0.0055 one-row-flip quantum, seed sd 0.019, 1030 test rows, the tree lane
0.6976 -> 0.7186, the alpha=1.5 finalist at 0.910837, AUC 0.9429 vs the leader's 0.9449. No
fabricated history, no invented modality, no leak. Its 17 citations carry DOIs/arXiv IDs.
Contrast with the ChatGPT report, rejected in full.

**Verdict: one proposal confirmed live, one dead on our code, one promoted to the iter44
candidate. Net: the report is right that the bottleneck is the combiner, and wrong about which
artifact the fix pays off on.**

---

## PROPOSAL 1 — pool-then-calibrate / logit-space averaging: THE CODE CRITICISM IS CORRECT

The report claims our pooling order is backwards. **It is, exactly as described.**
`src/calibration.py:140-172` (`calibrated_pool`) does, verbatim:

    for each member: o_cal, t_cal, slope = platt_calibrate(y_m, oof_m, test_m)   # line 163
    p_test = np.vstack(cal_test).mean(axis=0)                                    # line 170

That is a **linear opinion pool of individually-calibrated probabilities** — precisely the
construction Ranjan & Gneiting (2010, JRSS-B 72(1):71-91, DOI 10.1111/j.1467-9868.2009.00726.x)
prove is necessarily uncalibrated and unsharp, and that Rahaman & Thiery (NeurIPS 2021,
arXiv:2007.08792) remedy with Pool-Then-Calibrate. The docstring at lines 145-159 argues for
per-member calibration *against rank-averaging*, which is a sound argument — but it never
considered the third option (pool the logits, calibrate once), which dominates both.

**WHERE THE REPORT'S RANKING IS WRONG.** It ranks this #1 as a level lever on the champion. Our
ledger says otherwise: pooling has delivered its expected **+0.0055 to +0.0061 on every perm and
distill pool** (iter33d +0.0055, iter41 +0.0061). A combiner defect that costs nothing on eight
pools out of nine is not the champion's bottleneck. It bit **exactly once** — ARM T (tcons) — and
that is the one pool whose members' Platt slopes diverged, because the unlabeled variance penalty
compresses each seed's logit distribution by a different amount.

So the honest expected value of Proposal 1 **in isolation** is ~0 on the current champion
(members are near-homogeneous; logit-averaging is then near rank-equivalent to
probability-averaging). Its real function is as the **instrument that makes Proposal 3
measurable**. Proposals 1 and 3 are ONE experiment, not two ranked separately.

Compliance: clean, and arguably more standard-correct than what we ship. Must be a config flag
defaulting **OFF** so the already-scored finalists and every `anchors.tsv` entry reproduce
bit-for-bit; `experiments/reproduce_champion.sh` needs the corresponding gate assertion.

## PROPOSAL 2 — regime-matched OOF calibration: DEAD, twice over

**(a) Killed by the report's own load-bearing caveat.** Its Caveat 4 (line 74) says: *"if the
current OOF is in fact already computed under masked views, Proposal 2 does nothing (a monotone
refit on the same distribution cannot move the count). Verify the current OOF observation regime
before spending the slot."* Verified — it fires:

- `src/seq_model.py:980-981` — the held-out fold is predicted through
  `_mask_views(train_cube, va, schema, wd, cfg, R, cfg["seed"] + rep, oof=True)`.
- `src/seq_model.py:753` — `_mask_views` draws every view from `sample_window(wd, cfg, rng,
  schema.n_months)`, i.e. **the same empirical test-window distribution `wd`** used for training.
- `oof=True` changes **only the RNG tag** (`tag = (10000 + k) if oof else k`, line 751).

**Our OOF is already regime-matched.** There is no observation-regime mis-specification to
correct, so the Platt refit Proposal 2 describes is a monotone refit on the same score
distribution and cannot move the positive count. Dead on the facts.

**(b) Its expected-value story rests on a premise we supplied wrongly.** The proposal aims to
"move the realized positive rate from ~0.588 toward the believed ~0.65." That 0.65 came from
`UPDATE_19.md:57` ("Believed true test prevalence ~0.65") — **our error**, a stale figure carried
forward from the retired prevalence-pin era. Our own measurement contradicts it: the iter35b
label-shift gate (`tools/label_shift_gate.py`, LB_LOG row 35b) estimated the test prior at
**MLLS 0.578 / BBSE 0.559**, in agreement, and explicitly noted it was *below* the believed 0.65.
At a realized 0.588 we are already **at or slightly above** the measured test prior. There is no
pos-rate gap to close. `UPDATE_19.md` must be corrected before it goes to any further round.

**(c) The Lipton F/2 argument is real theory but must be declined here.** Lipton, Elkan &
Narayanaswamy (arXiv:1402.1892) is correctly cited: for calibrated probabilities the F1-optimal
threshold is F/2, so at F1~0.92 the optimum is ~0.46 and a forced 0.5 cut is F1-suboptimal by
construction. The report proposes recovering this by shifting the score distribution up so that
0.5 "behaves like 0.46." With (a) and (b) established, that shift would **not** be correcting a
mis-specified p(y|x) — the calibrator is already fit under the right regime and already lands at
the measured prior. It would be an operating-point move selected for its effect on the cut, which
fails the report's own prongs (b) and (c). **That is threshold tuning in a calibration costume,
and we decline it.** Recorded here because it is the most seductive argument in the report and
will recur: the fact that a legal-looking mechanism exists does not make a pos-rate-motivated
application of it legal.

## PROPOSAL 3 — consistency + distillation under the fixed combiner: THE iter44 CANDIDATE

This is the live one, and it converges with the clue we found independently before the report
arrived (see `UPDATE_19.md` §3): ARM T's members hold our **two highest F1 values ever**
(tcons_s42 0.901333, tcons_s13 0.897507, vs the best distill pool's 0.889488), and the ARM T pool
posted the **only negative pooling gain in the entire ledger**. Two independent routes now say
the same thing: we may have killed a working method for a combiner defect.

Mechanism support is properly sourced (Grandvalet & Bengio NIPS 2004; Chapelle & Zien AISTATS
2005 — low-density separation sharpens the local margin, which is an F1@0.5 lever and rank-neutral,
matching the observed AUC-fine/F1-broken signature).

**FREE CHECK, TO RUN BEFORE ANY SUBMISSION.** Re-pool the saved ARM T member `.npz` predictions
both ways — current prob-average vs logit-average + one Platt map on the pooled OOF — and read
the pooled pos-rate and OOF. If logit-averaging removes the pos-rate drift, the defect is
confirmed **mechanically, at zero submission cost**, and Proposal 3 is worth the slot. If the
drift persists, the ARM T failure was not the combiner and Proposal 3 collapses back to a
single-seed mirage. The preds are not in the local checkout (they live on the Colab run
directory), so this rides along in the iter44 script as step 0.

Standing constraint that still binds: four separate single-seed records have collapsed to ~0.8995
when seed-averaged. tcons must clear +0.006 **seed-averaged over >=5 seeds under the fixed
combiner** to count. The report states this correctly.

## Also correctly closed by the report

- **Multi-round self-distillation: do not run.** Mobahi et al. (arXiv:2002.05715) predict
  under-fitting after a few rounds; Kumar et al. (arXiv:2002.11361) show per-step error
  compounding at our W-infinity. Matches the standing iter41 note ("ONE round only").
- **The CatBoost-at-0.94 hypothesis: confirmed dead.** Agrees with our three independent fails.
- **0.95 is not a legal target.** The defensible five-day ambition is ~0.919-0.926.

## Net effect on the plan

- iter43's three uploads proceed unchanged. Its result still sets finalist #1.
- **iter44 = Proposals 1 + 3 as a single experiment**, with the free re-pooling diagnostic as
  step 0 and the combiner behind an OFF-by-default flag. Not Proposal 1 alone (~0 expected on a
  champion that already pools correctly), and not Proposal 2 (dead).
- Proposal 2 and the F/2 score-shift are **closed permanently**; do not re-litigate.
- **Fix `UPDATE_19.md:57`** before it is fed to any further research round.
