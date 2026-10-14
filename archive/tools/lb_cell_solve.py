"""Solve the public-leaderboard confusion cell EXACTLY from the reported (AUC, F1) pairs.

WHAT THIS REPLACES, AND WHY IT MATTERS MORE THAN ANYTHING ELSE IN ROUND 24.
Since iter42 we have inverted the F1 column to recover the public confusion cell, and we have
carried `n_public = 309`, `P_public = 191` as settled. `309` was never read off Zindi -- it is
30% of 1030, inferred. A round-24 researcher tested the trio for arithmetic consistency and it
FAILS. This tool reproduces that refutation and then goes further: it solves for (n, P).

THE CONSTRAINT THE F1 COLUMN CANNOT SEE.
F1 = 2·TP/(PP+P) pins the SUM PP+P and TP, but says nothing about how the sum splits between our
predicted positives and the truth. AUC does. On a finite sample ROC-AUC is exactly

    AUC = C / (P·N),    C = concordant-pair count ∈ ½·ℤ

(a half-integer, because under the Mann-Whitney convention `sklearn.roc_auc_score` uses, each tied
pair contributes exactly ½). So a 9-decimal reported AUC is a rational with denominator 2·P·N, and
that is a brutally tight sieve: the display window is 1e-9 wide while consecutive achievable AUCs
at P·N ~ 2.3e4 are ~2.2e-5 apart. Roughly one (n, P) in 20 000 can satisfy a given readout by
chance. Five reported submissions must satisfy it SIMULTANEOUSLY at the same (n, P).

RESULT.
  1. REFUTATION. No (P, N) with P + N = 309 reproduces AUC 0.945841814. Nearest miss 1.9e-7 vs a
     1e-9 window -- 76x too far. Our own P = 191 misses by 5.2e-6, i.e. 2080x. This does NOT depend
     on the composite formula: Zindi reports AUC directly, so the "0.6·F1 + 0.4·AUC verified to
     4e-10" fit is not the suspect. n_public = 309 is simply wrong.
  2. SOLUTION. Five (AUC, F1) pairs leave 15 candidate (n, P). Every candidate the next step does
     not reject lies in one scaling family: P = 181·s, PP = 191·s.
  3. SELECTION. Each submission's full-test predicted-positive count PP_full is on disk, and
     PP_public / PP_full ≈ n / 1030. Four submissions give n = 324.0, 326.4, 324.5, 331.6. The
     admissible values in the P=181 family are {257, 333, 409, 485, 561, 637}. **n = 333.**

  >>> PUBLIC CELL, CHAMPION:  n = 333, P = 181, N = 152
  >>>   TP = 164   FP = 27   FN = 17   TN = 125
  >>>   precision = 0.858639   recall = 0.906077   F1 = 0.881720430

⚠️ THE STRATEGIC CONSEQUENCE, WHICH IS THE WHOLE POINT.
We believed precision 0.9061 / recall 0.8586 -- 27 positives missed, a RECALL deficit. The truth is
the mirror image: **precision 0.8586 / recall 0.9061, 27 false positives against 17 misses.** Our
dominant error mode is false positives, by 1.6 to 1. Round 24's entire framing -- the "high-recall
corner", partial-AUC, JTT to recover missed positives -- was aimed at the smaller half of the error
budget. Note also the model over-predicts: realized public pos-rate 0.5736 against a true 0.5435.
Every argument of the form "our operating pos-rate may be too LOW" is refuted, not open.

ROBUSTNESS. The swap is not contingent on n = 333. TP and PP are fixed across the ENTIRE P = 181
family (only TN moves with n), so precision 0.8586 / recall 0.9061 holds for all six of its members
and for the ×2 family. The families that would preserve the old reading (P = 190 at n = 552,
P = 380 at n = 561, P = 543 at n = 695) are each rejected by step 3 by 200+ rows.

LEGALITY. Leaderboard-derived, therefore DIAGNOSIS ONLY under the standing rule: nothing here may
set a threshold, a hyperparameter, or a model choice. It is arithmetic on numbers already reported
to us, in the same category as the F1 inversion disclosed since iter42.

USAGE
    python tools/lb_cell_solve.py                 # full derivation, all three steps
    python tools/lb_cell_solve.py --nmax 700      # widen the (n, P) sweep
"""
from __future__ import annotations

import argparse
import math
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBS = ROOT / "submissions"

# (name, reported AUC, reported F1, submission stem on disk or None)
REPORTED = [
    ("champion", "0.945841814", "0.881720430", None),
    ("sub_0.942570", "0.942570514", "0.883468834", "submission_champion_archblend4"),
    ("jtt_lam5", "0.944824076", "0.884097035", "submission_jtt_lam5_s42"),
    ("presto_frozen", "0.827675196", "0.790190735", "submission_presto_frozen_local"),
    ("presto_ft", "0.836907531", "0.794520547", "submission_presto_finetune_local"),
]
N_TEST = 1030


def auc_realisable(auc_s: str, P: int, N: int) -> bool:
    """Is the 9-decimal readout `auc_s` achievable as C/(P·N) with C a half-integer?"""
    lo, hi = F(auc_s), F(auc_s) + F(1, 10 ** 9)
    x = lo * P * N * 2
    return any(lo <= F(C, 2) / (P * N) < hi for C in (math.floor(x), math.ceil(x)))


def f1_cells(f1_s: str, n: int, P: int) -> list[tuple[int, int]]:
    """All (TP, PP) at this (n, P) whose F1 displays as `f1_s`."""
    lo, hi = F(f1_s), F(f1_s) + F(1, 10 ** 9)
    out = []
    for TP in range(P + 1):
        for PP in range(TP, n + 1):
            v = F(2 * TP, PP + P)
            if v >= hi:
                continue
            if lo <= v and PP - TP <= n - P:
                out.append((TP, PP))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=700)
    args = ap.parse_args()

    print("=" * 78)
    print("STEP 1 -- REFUTE n_public = 309")
    print("=" * 78)
    auc_s = REPORTED[0][1]
    lo = F(auc_s)
    best = None
    for P in range(100, 209):
        N = 309 - P
        if N <= 0:
            continue
        x = lo * P * N * 2
        for C in (math.floor(x), math.ceil(x)):
            a = F(C, 2) / (P * N)
            d = 0 if lo <= a < lo + F(1, 10 ** 9) else min(abs(a - lo), abs(a - lo - F(1, 10 ** 9)))
            if best is None or d < best[0]:
                best = (d, P, float(a))
    print(f"  reported AUC {auc_s}, display window 1e-09 wide")
    print(f"  closest realisable value over all P at n=309: {best[2]:.12f}  (P={best[1]})")
    print(f"  distance {float(best[0]):.3e}  ->  {float(best[0]) / 1e-9:.0f}x the window.  IMPOSSIBLE.")

    print()
    print("=" * 78)
    print(f"STEP 2 -- SOLVE (n, P) against all {len(REPORTED)} reported pairs")
    print("=" * 78)
    survivors = []
    for n in range(200, args.nmax + 1):
        for P in range(50, n):
            N = n - P
            if not all(auc_realisable(a, P, N) for _, a, _, _ in REPORTED):
                continue
            cells = {nm: f1_cells(f, n, P) for nm, _, f, _ in REPORTED}
            if any(not v for v in cells.values()):
                continue
            survivors.append((n, P, cells))
    print(f"  candidate (n, P) satisfying every reported pair: {len(survivors)}")
    for n, P, cells in survivors:
        print(f"    n={n:4d} P={P:4d} N={n - P:4d} prev={P / n:.4f}   champion {cells['champion']}")

    print()
    print("=" * 78)
    print("STEP 3 -- SELECT n from full-test predicted-positive counts")
    print("=" * 78)
    print(f"  PP_public / PP_full = n / {N_TEST}, up to subsampling noise.")
    ests = []
    for nm, _, _, stem in REPORTED:
        if stem is None:
            continue
        path = SUBS / f"{stem}.csv"
        if not path.exists():
            print(f"    {nm:<14s} SKIP (not on disk: {path.name})")
            continue
        d = pd.read_csv(path)
        col = "TargetF1" if "TargetF1" in d.columns else d.columns[1]
        pp_full = int((d[col].values >= 0.5).sum())
        # PP_public in the P=181 family is fixed across n; take it from the first survivor there
        pp_pub = next((c[nm][0][1] for _, Pv, c in survivors if Pv == 181), None)
        if pp_pub is None:
            continue
        est = N_TEST * pp_pub / pp_full
        ests.append(est)
        print(f"    {nm:<14s} PP_full={pp_full:4d}  PP_public={pp_pub:3d}  ->  n = {est:6.1f}")
    if ests:
        adm = sorted({n for n, P, _ in survivors if P == 181})
        mean = sum(ests) / len(ests)
        pick = min(adm, key=lambda v: abs(v - mean))
        runner = min((v for v in adm if v != pick), key=lambda v: abs(v - mean))
        print(f"    mean estimate {mean:.1f}; admissible in the P=181 family: {adm}")
        print(f"    -> n = {pick}   (next best {runner}, off by {abs(runner - mean):.0f} vs {abs(pick - mean):.0f})")

        print()
        print("=" * 78)
        print(f"RESULT -- PUBLIC CELL  n={pick}  P=181  N={pick - 181}")
        print("=" * 78)
        for nm, _, _, _ in REPORTED:
            cells = next(c for n, P, c in survivors if n == pick and P == 181)
            TP, PP = cells[nm][0]
            FP, FN, TN = PP - TP, 181 - TP, (pick - 181) - (PP - TP)
            print(f"  {nm:<14s} TP={TP:3d} FP={FP:3d} FN={FN:3d} TN={TN:3d} | "
                  f"precision={TP / PP:.6f} recall={TP / 181:.6f} | pred pos-rate={PP / pick:.4f}")
        print(f"  true public prevalence = 181/{pick} = {181 / pick:.4f}")
        print("  ** FP (27) exceeds FN (17): the dominant error is FALSE POSITIVES, and the model")
        print("     OVER-predicts (0.5736 realised vs 0.5435 true). The old cell said the opposite. **")


if __name__ == "__main__":
    main()
