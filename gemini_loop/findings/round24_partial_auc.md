# Round 24 — Q1: Optimising the HIGH-RECALL REGION of the ROC (partial AUC, NP classification, constrained ERM)

**Researcher note.** Written incrementally, appended after each paper read. Everything below is
labelled **VERIFIED (read)** = I fetched and read the paper/abstract text, or **INFERRED** = derived
by me from theory, or read only second-hand. Derivations marked "MY DERIVATION" are mine and are
the load-bearing part of the answer — check them.

Task context (from UPDATE_24.md): team score = 0.6·F1 + 0.4·AUC, threshold literally 0.5 and
non-negotiable, n=1817 train / 1030 test, 71k-param temporal Transformer, global AUC 0.945842 already
beats the leader (0.944897) but F1 0.881720 vs ~0.918. They sit at recall 0.859 / precision 0.906.
Two theorems must be survived: (T1) Platt annihilation of affine logit changes, (T2) pointwise-loss
order invariance.

Status: IN PROGRESS.

---

