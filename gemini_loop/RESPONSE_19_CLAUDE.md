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

**MY FIRST TRIAGE OF THIS PROPOSAL WAS WRONG. CORRECTED 2026-08-11, SAME DAY.**

The first draft of this section argued the report had mis-ranked Proposal 1: that pooling has
paid +0.0055..+0.0061 on every perm/distill pool, so a defect biting eight pools out of nine was
not the champion's bottleneck; that it bit only ARM T; and that it bit there because ARM T's
member Platt slopes diverged. **Two errors, both material.**

1. **Slope divergence cannot be the mechanism.** Per-member Platt scaling is *scale-invariant* —
   fitting a logistic on each member's own logits removes that member's slope by construction. So
   combiner A cannot see slope heterogeneity at all. Verified synthetically (`tools/repool.py`
   header): 5 members at slopes {1,1,1,1,1} vs {0.35,0.6,1.0,1.7,2.6} give combiner A
   **bit-identical** pooled OOF F1 0.62529 and pos-rate 0.2033.
2. **The ledger evidence I cited was a category error.** The +0.0055..+0.0061 gains measured
   **pool vs single member**, never **combiner A vs combiner B**. Pooling can be worth +0.006 and
   still leave more on the table through the wrong combiner. *Nothing in the ledger has ever
   compared the two orders.* I used an irrelevant measurement to downgrade a live proposal.

**The correct mechanism is generic, not ARM-T-specific.** An arithmetic mean of
independently-noisy probabilities is shrunk toward the members' centre of mass, so the pooled
distribution is strictly narrower than any member's and a FIXED 0.5 cut catches fewer rows. This
requires only independent member noise — which is exactly what multi-seed pooling is. In the same
synthetic, B beat A by **+0.039 F1 (homogeneous)** and **+0.031 F1 (divergent)**, with pos-rate
0.2033 -> 0.2450 and 0.2500 respectively.

**So the report's ranking stands and mine did not: Proposal 1 is a candidate level lever on the
CHAMPION pool, not merely an instrument for Proposal 3.** Magnitudes above are synthetic and carry
no weight for our data; the direction and the genericity do. Settle it on real bundles with
`tools/repool.py` before spending a submission.

This also refines the ARM T diagnosis. The tcons members were individually *underdispersed* (a
cross-view variance penalty compresses each seed's logits toward a constant — its known
attractor), and arithmetic averaging of already-underdispersed members compresses hardest. That is
why ARM T was hit worst while ordinary pools merely under-delivered. Same defect, different
severity — not a special case.

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
- **iter44 = Proposal 1 (primary) + Proposal 3 (rider)**, with the `tools/repool.py` diagnostic as
  step 0 and the combiner behind an OFF-by-default flag. Proposal 1 is now the primary because the
  defect is generic and therefore applies to the champion pool itself; Proposal 3 rides along
  because the same fixed combiner is what makes the ARM T re-test interpretable. Not Proposal 2
  (dead).
- **Bundle persistence.** `colab_run.ipynb:55-59` copies the three CSVs IN from Drive and copies
  nothing back, so `submissions/preds/*.npz` dies with the session — iteration 41's ARM T bundles
  are already lost, which is why step 0 must regenerate its members. Add a Drive copy-out of
  `submissions/preds/` so no future diagnostic is blocked on re-running finished compute.
- Proposal 2 and the F/2 score-shift are **closed permanently**; do not re-litigate.
- **Fix `UPDATE_19.md:57`** before it is fed to any further research round.
