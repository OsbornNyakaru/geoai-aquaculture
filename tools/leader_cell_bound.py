"""PART D -- BOUND the leader's public confusion cell on the solved partition.

Known: public cell n=333, P=181, N=152 (tools/lb_cell_solve.py).
Leader: AUC = 0.944897 (6 dp only -- NOT sieve-tight), composite only known as ~0.929-0.936.
So we BOUND rather than solve.  Constraints used, all exact:

  1. composite = 0.6*F1 + 0.4*AUC  =>  F1 in [(0.929-0.4A)/0.6, (0.936-0.4A)/0.6]
  2. F1 = 2*TP/(PP+P) with integer TP<=P, FP=PP-TP<=N
  3. AUC-consistency of the CELL (the block decomposition of Part A, applied to their cell):
        C_leader = TP*TN + a + b,  0<=a<=TP*FP, 0<=b<=FN*TN
     =>  TP*TN <= C  <=  TP*N + FN*TN
     i.e. a cell is only compatible with their AUC if those brackets contain C.
"""
from fractions import Fraction as Fr
import math

P, N, n = 181, 152, 333
OUR = dict(TP=164, FP=27, FN=17, TN=125)
A_LEAD = Fr("0.944897")
COMP_LO, COMP_HI = Fr("0.929"), Fr("0.936")

print("=" * 78)
print("D.0  is the leader's AUC readout even sieve-testable at (P,N) = (181,152)?")
print("=" * 78)
PN = P * N
lo, hi = A_LEAD, A_LEAD + Fr(1, 10**6)          # 6 dp => 1e-6 window
step = Fr(1, 2 * PN)
hits = [C for C in range(math.floor(lo * PN * 2) - 1, math.ceil(hi * PN * 2) + 2)
        if lo <= Fr(C, 2) / PN < hi]
print(f"  spacing of achievable AUCs at P*N={PN}: 1/(2PN) = {float(step):.3e}")
print(f"  6-dp display window: 1.0e-06  ->  window/spacing = {float(Fr(1,10**6)/step):.3f}")
print(f"  half-integers C*2 inside the window: {hits}")
print("  => a 6-dp readout is ~18x NARROWER than the spacing, so a miss would prove NOTHING.")
print("     Their AUC cannot be used as a sieve; we only use it as a value. (Ours was 9 dp.)")
C_lead_lo = A_LEAD * PN
C_lead_hi = (A_LEAD + Fr(1, 10**6)) * PN
print(f"  C_leader in [{float(C_lead_lo):.1f}, {float(C_lead_hi):.1f}]   "
      f"(ours: 26022 exactly).  D_leader ~ {PN - float(C_lead_lo):.0f} vs our 1490.")

print()
print("=" * 78)
print("D.1  enumerate every cell compatible with composite in [0.929, 0.936]")
print("=" * 78)
F1_LO = (COMP_LO - Fr(2, 5) * A_LEAD) / Fr(3, 5)
F1_HI = (COMP_HI - Fr(2, 5) * A_LEAD) / Fr(3, 5)
print(f"  F1_leader in [{float(F1_LO):.7f}, {float(F1_HI):.7f}]   (ours 0.881720430)")

cells = []
for TP in range(0, P + 1):
    for FP in range(0, N + 1):
        PP = TP + FP
        if PP == 0:
            continue
        f1 = Fr(2 * TP, PP + P)
        if not (F1_LO <= f1 <= F1_HI):
            continue
        FN, TN = P - TP, N - FP
        if not (TP * TN <= C_lead_hi and C_lead_lo <= TP * N + FN * TN):
            continue                      # AUC-consistency of the block decomposition
        cells.append((TP, FP, FN, TN, PP, f1, Fr(TP, PP), Fr(TP, P)))

print(f"  cells surviving F1 band + AUC-consistency: {len(cells)}")
prec = [float(c[6]) for c in cells]
rec = [float(c[7]) for c in cells]
pps = [c[4] for c in cells]
errs = [c[1] + c[2] for c in cells]
print(f"  precision range : [{min(prec):.4f}, {max(prec):.4f}]   ours 0.858639")
print(f"  recall    range : [{min(rec):.4f}, {max(rec):.4f}]   ours 0.906077")
print(f"  PP        range : [{min(pps)}, {max(pps)}]                 ours 191, truth P=181")
print(f"  FP+FN     range : [{min(errs)}, {max(errs)}]                   ours 44")

print()
print("  --- do they beat us on precision, on recall, or on both? ---")
both = [c for c in cells if c[6] > Fr(OUR['TP'], OUR['TP'] + OUR['FP']) and c[7] > Fr(OUR['TP'], P)]
prec_only = [c for c in cells if c[6] > Fr(OUR['TP'], OUR['TP'] + OUR['FP']) and c[7] <= Fr(OUR['TP'], P)]
rec_only = [c for c in cells if c[6] <= Fr(OUR['TP'], OUR['TP'] + OUR['FP']) and c[7] > Fr(OUR['TP'], P)]
print(f"    cells beating us on BOTH        : {len(both):3d}  ({100*len(both)/len(cells):.0f}%)")
print(f"    cells beating us on PRECISION only: {len(prec_only):3d}")
print(f"    cells beating us on RECALL only   : {len(rec_only):3d}")
dprec = [float(c[6]) - OUR['TP'] / (OUR['TP'] + OUR['FP']) for c in cells]
drec = [float(c[7]) - OUR['TP'] / P for c in cells]
print(f"    precision advantage over us: [{min(dprec):+.4f}, {max(dprec):+.4f}]  mean {sum(dprec)/len(dprec):+.4f}")
print(f"    recall    advantage over us: [{min(drec):+.4f}, {max(drec):+.4f}]  mean {sum(drec)/len(drec):+.4f}")

print()
print("  --- the two composite endpoints, cell by cell ---")
for lbl, band in [("composite ~0.936 (F1 hi)", [c for c in cells if float(c[5]) > float(F1_HI) - 0.0012]),
                  ("composite ~0.929 (F1 lo)", [c for c in cells if float(c[5]) < float(F1_LO) + 0.0012])]:
    print(f"   {lbl}:")
    for c in sorted(band, key=lambda c: -c[0])[:8]:
        TP, FP, FN, TN, PP, f1, pr, rc = c
        print(f"     TP={TP:3d} FP={FP:3d} FN={FN:3d} TN={TN:3d} | PP={PP:3d} | "
              f"F1={float(f1):.6f} prec={float(pr):.4f} rec={float(rc):.4f} "
              f"| composite={float(Fr(3,5)*f1 + Fr(2,5)*A_LEAD):.6f}")

print()
print("  --- CHECK the operator's guess TP=173 FP=18 FN=8 ---")
TP, FP = 173, 18
f1 = Fr(2 * TP, TP + FP + P)
comp = Fr(3, 5) * f1 + Fr(2, 5) * A_LEAD
print(f"    TP=173 FP=18 FN=8 TN=134: PP=191, F1={f1}={float(f1):.6f}, "
      f"composite={float(comp):.6f}  ->  {'IN' if COMP_LO <= comp <= COMP_HI else 'OUT OF'} the 0.929-0.936 band")
print(f"    precision={float(Fr(TP, TP+FP)):.4f} recall={float(Fr(TP, P)):.4f}; "
      f"predicted pos-rate={float(Fr(TP+FP, n)):.4f} vs true {float(Fr(P,n)):.4f}")

print()
print("=" * 78)
print("D.2  where is the bulk of their advantage?  (error-budget decomposition)")
print("=" * 78)
print(f"  us     : FP=27  FN=17  total 44")
for lbl, cs in [("their band, min-error corner", [min(cells, key=lambda c: c[1] + c[2])]),
                ("their band, max-error corner", [max(cells, key=lambda c: c[1] + c[2])])]:
    c = cs[0]
    print(f"  {lbl}: FP={c[1]:3d} FN={c[2]:3d} total {c[1]+c[2]:3d}")
fpr = [c[1] for c in cells]
fnr = [c[2] for c in cells]
print(f"  FP range {min(fpr)}-{max(fpr)} (ours 27), FN range {min(fnr)}-{max(fnr)} (ours 17)")
print(f"  median cell by FP: {sorted(fpr)[len(fpr)//2]},  by FN: {sorted(fnr)[len(fnr)//2]}")
