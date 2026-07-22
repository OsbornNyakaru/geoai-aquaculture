# RESEARCH_08 — EY Biodiversity Challenge comparison

**Date:** 2026-07-22 · **Question asked:** does <https://zindi.africa/competitions/ey-biodiversity-challenge>
resemble our competition closely enough to borrow from?

## Verdict: NOT TRANSFERABLE for technique — but it is a serious cautionary tale on governance

The two share a platform, a sponsor family, a boilerplate rulebook, and a binary target. **The three
things that define our entire toolkit are all absent there.** It also **closed on 24 May 2026**, so
there is nothing to enter.

| Axis | Verdict | |
|---|---|---|
| Prediction task | SIMILAR | both binary classification of a georeferenced point |
| Data modality | **DIFFERENT** | theirs is TerraClimate 4 km monthly climate rasters sampled to a small tabular table — no SAR, no optical, no masked panel, no `-9999` structure |
| **Designed shift** | **DIFFERENT — decisive** | train and test are both Nov 2017–Nov 2019, same region, random interleaved split. Expected adversarial AUC ≈ **0.5**, versus our ≈0.99 |
| Metric | **DIFFERENT** | single F1 on a hard-label submission with a **free threshold** — the calibration game is legal and trivial there, illegal and central here |
| LB noise | SIMILAR | ~200–300 public rows (20/80 split), but 300 submissions and 10/day |
| Rules | SAME | verbatim Zindi/EY boilerplate — paperwork similarity, not problem similarity |
| Leakage history | SIMILAR in existence, **DIFFERENT in kind** | see below |

**The inverse-coordinates observation is the neatest summary:** *they removed coordinates as a rule
because there is no shift; we lost coordinates as a design because there is one.*

Their CV/LB evidence also runs **opposite** to ours — CV 0.74 → LB 0.88, CV 0.83 → LB 0.88, and once
a pipeline was clean, CV 0.9412 → LB 0.9450 with plain `StratifiedKFold`. That is a level offset, not
our rank anti-correlation. **Nobody there needed to build a validator, and nobody did.** The honest
lesson: when CV works, don't build a CV replacement.

## The 0.99 thread is a leak — but a different class from ours

Their leak was **external re-identification**: labels come from FrogID/GBIF, a *public* occurrence
dataset, and test rows were keyed by supplied lat/lon. Join the coordinates back to public records
and read off the labels. The official benchmark notebook even instructed a `cKDTree` nearest-neighbour
mapping. Legitimate TerraClimate-only ceiling was ~**0.945**; 0.99 was cheating.

**Ours was internal** (a train/test shuffling bug yielding literal 1.0) and was fixed by **rebuilding
the data** on 25 Jun. Theirs could not be fixed by any rebuild without redesigning the task, so it was
policed by rule changes, code demands, and disqualification.

## ⚠️ The one finding that should change our behaviour

**Of a 1,395-person field, only 2 of the top 10 survived post-challenge code review.** The rest were
eliminated for *"failing to submit code within the deadline or having reproducibility concerns."*
Third prize appears to have gone unpaid entirely.

**Our final score is 65% LB + 35% code review of the top 5** — a *larger* weight against a *smaller*
field. The internal audit already found our README documents the **GBDT** as "the model" while our
champion is the Transformer, and that `--full` defaults to `--model gbdt`. **A judge reproducing "the
solution" from our README today reproduces the 0.826-era model.** That is precisely the failure class
that wiped out 8 of EY's top 10.

## Actions taken into the queue

1. **Treat reproducibility as a continuously-maintained artifact, not a deadline task.** Frozen,
   seeded, single-command path for both finalists. Highest-value import from this comparison.
2. **Designate both finalists deliberately.** Zindi defaults to "best 2 public", which on a 30%
   public slice is a shake-up generator. Our certified offline validators are the right instrument
   for choosing the *non-obvious* second finalist.
3. **Audit for implicitly-banned derived features.** EY's enforcement was broader than participants
   expected — not just lat/lon but *distances, nearest-neighbour, clustering, grid binning, any
   positional encoding*. Our coordinates are absent from the files, but we should verify nothing
   reconstructs position or row identity (row-order effects, ID-derived signals, cross-set
   nearest-neighbour matching). A leak-shaped feature that survives to the top 5 and dies in code
   review costs everything.
4. **Pseudo-labeling has an organizer precedent** — an EY organizer explicitly permitted predictions
   "generated from compliant models and the provided competition data" under a near-identical
   rulebook. **Honest caveat:** we rejected CAST self-training in round 05, but on *technical*
   grounds (the OOD/ESS-collapse family at adversarial AUC 0.99), never legal ones. What has changed
   is not the legality argument but that **we can now screen it for zero submissions**. Queued as a
   screen candidate, not as a submission.
5. **Do not trust a Zindi metric description.** Theirs contradicted itself (F1 vs accuracy) and sat
   unanswered for the entire competition. We should verify our 0.6·F1 + 0.4·AUC weighting empirically
   against our own known submission/score pairs rather than trusting the page text.

## Also found

A **live** sibling with a $10,000 prize: **EY Water Quality Forecasting Challenge** —
<https://zindi.africa/competitions/ey-water-quality-forecasting-challenge>. Not investigated;
flagged only because it is open and in the same family.

**Sources:** competition, data, rules and discussion pages under
<https://zindi.africa/competitions/ey-biodiversity-challenge>, notably threads 32189 (the 0.99
thread), 32229 (compliance action), 32369 (derived spatial features banned), 32819 (pseudo-labeling
permitted), 31566 (LB–CV correlation), 31618 (train/test period), 33879 (winners).

**Caveat on evidence quality:** the closed leaderboard does not render, so score anchors are
reconstructed from discussion quotes rather than read off the board; row counts are inferred from
published file sizes.
