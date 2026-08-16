"""PART A -- sharp max-F1 bound on a FIXED ranking, given the SOLVED public cell.

Cell: n=333, P=181, N=152, champion TP=164 FP=27 FN=17 TN=125, PP=191.
AUC printed 0.945841814 (9dp) => C/(P*N) with C a half-integer.
"""
from fractions import Fraction as Fr
import math

P, N, n = 181, 152, 333
TP, FP, FN, TN = 164, 27, 17, 125
PP = TP + FP
PN = P * N
print(f"P*N = {PN}, total (pos,neg) pairs")

lo = Fr("0.945841814"); hi = lo + Fr(1, 10**9)
sols = [C for C in (math.floor(lo*PN*2), math.ceil(lo*PN*2), math.ceil(lo*PN*2)+1)
        if lo <= Fr(C, 2)/PN < hi]
print("half-integer C (x2) satisfying the 9dp readout:", sols)
C2 = sols[0]                      # C = C2/2
C = Fr(C2, 2)
print(f"C = {C}  (concordant pairs, half-integer)   AUC = {float(C/PN):.12f}")
D = PN - C
print(f"D = discordant pairs = {D}")

# Block decomposition around the 0.5 cut
blk_TPxTN = TP*TN      # all concordant, forced
blk_FNxFP = FN*FP      # all discordant, forced
blk_TPxFP = TP*FP      # unknown split
blk_FNxTN = FN*TN      # unknown split
assert blk_TPxTN + blk_FNxFP + blk_TPxFP + blk_FNxTN == PN
print(f"\nBLOCK DECOMPOSITION of the {PN} pairs at the 0.5 cut")
print(f"  TP x TN = {blk_TPxTN:6d}  ALL concordant (forced)")
print(f"  FN x FP = {blk_FNxFP:6d}  ALL discordant (forced)")
print(f"  TP x FP = {blk_TPxFP:6d}  free  -> d_a discordant")
print(f"  FN x TN = {blk_FNxTN:6d}  free  -> d_b discordant")
Dwithin = D - blk_FNxFP
print(f"  d_a + d_b = D - FN*FP = {D} - {blk_FNxFP} = {Dwithin}   << the ENTIRE degree of freedom")
assert 0 <= Dwithin <= blk_TPxFP + blk_FNxTN

F1_now = Fr(2*TP, PP+P)
print(f"\ncurrent F1 = {F1_now} = {float(F1_now):.9f}")

# ---------------------------------------------------------------- sharp UPPER bound
# Raising the cut: remove the bottom m FPs (u_m = #TPs below the m-th lowest FP get removed too)
# Lowering the cut: admit the top j FNs (s_j = #TNs above the j-th highest FN come too)
best = (F1_now, "no move")
for m in range(0, FP+1):
    for u in range(0, TP+1):
        f1 = Fr(2*(TP-u), (PP-m-u) + P)
        if f1 > best[0] and (FP-m)*u <= Dwithin:   # necessary feasibility: retained FPs sit above the u removed TPs
            best = (f1, f"RAISE: drop {m} FP + {u} TP")
for j in range(0, FN+1):
    for s in range(0, TN+1):
        f1 = Fr(2*(TP+j), (PP+j+s) + P)
        if f1 > best[0] and (FN-j)*0 <= Dwithin:
            best = (f1, f"LOWER: admit {j} FN + {s} TN")
print(f"\nSHARP UPPER BOUND on max-F1 by threshold movement = {best[0]} = {float(best[0]):.9f}")
print(f"  attained by: {best[1]}")
print(f"  (RAISE ceiling 2*164/(345) = {float(Fr(328,345)):.9f};  LOWER ceiling 2*181/389 = {float(Fr(362,389)):.9f})")

# Does the AUC constraint bind on the maximiser?  Max area through (0,164) and (27,164):
maxC_thru_point = 0*164 + 27*164 + (N-27)*P
print(f"\n  max concordant pairs for ANY curve through (FP=27,TP=164) = {maxC_thru_point}"
      f"  >= C = {C}  -> the AUC constraint does NOT bind at the maximiser")

# ---------------------------------------------------------------- how much budget blocks a target
def budget_to_block(T):
    """min d_a+d_b an adversary needs so that NO threshold reaches F1 >= T."""
    da = 0
    for m in range(1, FP+1):
        u = 0
        while Fr(2*(TP-u), (PP-m-u)+P) >= T:
            u += 1
        da += u                      # u_m must be at least this; non-decreasing in m automatically
    db = 0
    for j in range(1, FN+1):
        s = 0
        while Fr(2*(TP+j), (PP+j+s)+P) >= T:
            s += 1
        db += s
    return da, db

print("\n" + "="*72)
print("HOW MUCH OF THE d_a+d_b BUDGET IS NEEDED TO MAKE A TARGET F1 UNREACHABLE")
print("="*72)
print(f"  available budget d_a + d_b = {Dwithin}")
for T in ["0.8817205", "0.90", "0.9184020", "0.9250", "0.9300687", "0.9400"]:
    da, db = budget_to_block(Fr(T))
    tot = da+db
    verdict = "BLOCKABLE" if tot <= Dwithin else "UNREACHABLE-PROOF IMPOSSIBLE (target always reachable)"
    print(f"  T={T}:  block raises needs d_a>={da:5d}, block lowers needs d_b>={db:4d}, "
          f"total {tot:5d}  ({100*tot/Dwithin:5.1f}% of budget)  {verdict}")

# ---------------------------------------------------------------- minimal move that reaches leader
print("\n" + "="*72)
print("CHEAPEST THRESHOLD MOVES THAT REACH THE LEADER'S F1 BAND")
print("="*72)
for T, lbl in [(Fr("0.9184020"), "leader @ composite 0.929"), (Fr("0.9300687"), "leader @ composite 0.936")]:
    ms = [m for m in range(FP+1) if Fr(2*TP, (PP-m)+P) >= T]
    js = [(j, s) for j in range(FN+1) for s in range(TN+1)
          if Fr(2*(TP+j), (PP+j+s)+P) >= T]
    smax = max((s for j, s in js if j == FN), default=None)
    print(f"  {lbl}  (F1 >= {float(T):.7f})")
    print(f"     RAISE route: drop the lowest {min(ms) if ms else 'n/a'} FPs with ZERO TP loss")
    print(f"     LOWER route: admit all {FN} FNs with at most {smax} TNs riding along")
