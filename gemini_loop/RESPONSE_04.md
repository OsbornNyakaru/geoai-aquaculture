# RESPONSE_04 — round-04 Deep Research reply + Claude triage

*Deep-research reply to `UPDATE_04.md`, received 2026-07-20, with the coding agent's triage
against proven evidence. The loop rule: implement the best ideas, reject re-treads. Full
original report is in the session log; this is the actionable distillation.*

## The 5 proposed ideas and their verdicts

| # | Idea (report) | Verdict | Rationale (vs our LB evidence) |
|---|---|---|---|
| A | **Relative-time coordinate frame** — left-align each observed window to t_rel=0 so positional embeddings encode relative step, not calendar month | ✅ **TEST FIRST** | Capacity-NEUTRAL (no new dims/params), genuinely new, plausibly LARGE effect. Directly attacks the calendar-month memorization that our covariate shift punishes. |
| C | **MC temporal-dropout TTA** — at inference, mask 1–2 active months per test row, soft-vote over N views | ✅ Keep (bank) | Inference-side, no added capacity → dodges the failure pattern. Already in our queue (UPDATE_04 Q3). Likely within noise on public LB but helps the private split. |
| D | **Multi-seed bagging** (10–20 seeds, average) | ✅ Keep (bank) | Variance reducer, no added capacity. Just `n_repeats↑`. Safe, modest. |
| B | **Replace raw bands with 4 physical indices** (MNDWI, EVI, Zou water tree, VH/VV) | ⚠️ **Partly reject** | The "constrain input dim" instinct aligns with our lesson, BUT: EVI already failed on LB (`config: evi:false`); the **Zou et al. water tree is hardcoded magic-constant thresholds** (EVI<0.1, AWEI>−0.1) = the non-transferable class we killed in round 3. High risk of nuking the champion. |
| E | **Saerens EM prior estimation** | ❌ **Reject — proven dead-end** | Built & tested already: BBSE/Saerens-EM estimated ~0.44 vs true LB-optimal 0.65. It assumes **label** shift; ours is **covariate** shift, which breaks it. `prevalence_target 0.649` already hits the LB-verified optimum — strictly better. |

## Two errors in the report to keep in mind
1. It cites **WIF's OOF AUC 0.826 as evidence for re-adding indices** — that number is the poster
   child of this competition's core trap (high OOF, no LB transfer; WIF scored −0.005). Trusting
   it repeats the exact mistake we've disproven 5×.
2. Its **+0.03–0.05-per-idea estimates** sum to a fantasy (~+0.15). With ~±0.01 public-LB noise and
   top-5 at 0.928, these are hopeful, not evidence-based.

## What we do NOT do
The report's "refactor features.py + data.py + run_pipeline.py + models.py + calibration.py at
once" is rejected: simultaneous changes are unattributable under ±0.01 noise, bundle proven dead
ends, and risk silently destroying the 0.878 champion. We change **one isolated thing**, held at
the 0.649 operating point, judged on the LB — the only method that has worked.

## Implemented this round
**Idea A only**, behind `seq.relative_time` (default false = champion, verified bit-for-bit).
`_left_align()` in `src/seq_model.py` rolls each row's observed window to index 0; threaded through
`to_inputs`. Iteration 5 = `submission_seq_reltime.csv`. If it beats 0.8780 → keep, then bank C+D.
If not → next probe is C (MC-TD TTA, inference-only, safest).
