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

(appended as I go)

