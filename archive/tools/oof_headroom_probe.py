"""PARTS B & C -- what the MASKED OOF replica says about the operating point.

`oof_prob` in submissions/preds/*.npz is already the deployment-matched replica: held-out rows are
passed through `_mask_views(..., oof=True)` (4-6 contiguous months, the measured p(L), p(start|L))
and R views are averaged. So it is the right surface on which to ask the two questions the
leaderboard cell raised:

  (B) Is the over-prediction (public PP=191 against P=181) reproduced train-only, and through which
      channel -- affine (dead under T1) or non-affine (alive)?
  (C) What does the HIGH-PRECISION / low-FPR corner of the ROC look like under deployment masking?

Everything here is TRAIN-ONLY. No leaderboard quantity sets anything; LB numbers appear only as
printed comparators.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "submissions" / "preds"


def f1_at(y, p, t):
    yh = (p >= t)
    tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum()); fn = int((~yh & (y == 1)).sum())
    return (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0, tp, fp, fn


def auc(y, p):
    o = np.argsort(p, kind="mergesort")
    r = np.empty(len(p), float)
    ys = p[o]
    i = 0
    while i < len(ys):
        j = i
        while j + 1 < len(ys) and ys[j + 1] == ys[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    P, N = int((y == 1).sum()), int((y == 0).sum())
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def main(stems):
    for stem in stems:
        f = PRED / f"{stem}.npz"
        if not f.exists():
            print(f"MISSING {f}"); continue
        d = np.load(f, allow_pickle=True)
        p, y = np.asarray(d["oof_prob"], float), np.asarray(d["y"], int)
        n, P, N = len(y), int(y.sum()), int((y == 0).sum())
        prior = P / n
        print("=" * 78)
        print(f"{stem}   n={n}  P={P}  N={N}  prior={prior:.4f}")
        print("=" * 78)

        # ---------------- B.1 does the over-prediction reproduce train-only?
        F1, tp, fp, fn = f1_at(y, p, 0.5)
        PPr = tp + fp
        print(f"  @0.5 : TP={tp} FP={fp} FN={fn} TN={n-PPr-fn}  F1={F1:.6f} "
              f"prec={tp/max(PPr,1):.4f} rec={tp/P:.4f}")
        print(f"         predicted pos-rate {PPr/n:.4f}  vs true prior {prior:.4f}  "
              f"-> {'OVER' if PPr > P else 'UNDER'}-predicts by {PPr-P:+d} rows "
              f"({100*(PPr-P)/P:+.1f}%)")
        print(f"  AUC(masked OOF) = {auc(y, p):.6f}")

        # ---------------- B.2 calibration: sum of probabilities vs count of positives
        print(f"  CALIBRATION  sum(p)={p.sum():.1f} vs P={P}   -> mass ratio {p.sum()/P:.4f}")
        print(f"               mean p over predicted-positives = {p[p >= .5].mean():.4f} "
              f"vs realised precision {tp/max(PPr,1):.4f}  "
              f"(over-confident by {p[p>=.5].mean()-tp/max(PPr,1):+.4f})")
        print(f"               mean p over predicted-negatives = {p[p < .5].mean():.4f} "
              f"vs realised FOR {fn/max(n-PPr,1):.4f}  "
              f"({p[p<.5].mean()-fn/max(n-PPr,1):+.4f})")
        print("  reliability (equal-count deciles of p):")
        o = np.argsort(p)
        for b in range(10):
            idx = o[b * n // 10:(b + 1) * n // 10]
            print(f"     bin {b}: p in [{p[idx].min():.3f},{p[idx].max():.3f}] "
                  f"mean_p={p[idx].mean():.4f}  emp={y[idx].mean():.4f}  gap={p[idx].mean()-y[idx].mean():+.4f}")

        # ---------------- B.3 Lipton t* = F*/2 on this surface
        ts = np.unique(np.round(p, 6))
        best = max(((f1_at(y, p, t)[0], t) for t in ts))
        print(f"\n  LIPTON CHECK   max-F1 over thresholds = {best[0]:.6f} at t = {best[1]:.4f}")
        print(f"                 theorem says t* = F*/2 = {best[0]/2:.4f}   "
              f"|argmax - F*/2| = {abs(best[1]-best[0]/2):.4f}")
        f1_44, *_ = f1_at(y, p, best[0] / 2)
        print(f"                 F1 at t=F*/2 : {f1_44:.6f}   F1 at t=0.5 : {F1:.6f}   "
              f"delta = {f1_44-F1:+.6f}")
        print(f"                 rows with p in [F*/2, 0.5) = {int(((p >= best[0]/2) & (p < .5)).sum())}, "
              f"of which positive = {int(y[(p >= best[0]/2) & (p < .5)].sum())} "
              f"(rate {y[(p >= best[0]/2) & (p < .5)].mean() if ((p>=best[0]/2)&(p<.5)).any() else float('nan'):.4f} "
              f"vs the F*/2 = {best[0]/2:.4f} break-even)")

        # ---------------- C.1 high-precision / low-FPR corner
        print("\n  PART C -- HIGH-PRECISION / LOW-FPR CORNER under deployment masking")
        print("   target_prec |  thresh | recall |  TPR   |  FPR   | #pred+ | F1")
        for tgt in (0.999, 0.99, 0.98, 0.95, 0.92, 0.90, 0.88, 0.86):
            ok = None
            for t in np.sort(np.unique(p))[::-1]:
                yh = p >= t
                tpx, fpx = int((yh & (y == 1)).sum()), int((yh & (y == 0)).sum())
                if tpx + fpx == 0:
                    continue
                if tpx / (tpx + fpx) >= tgt:
                    ok = (t, tpx, fpx)
            if ok is None:
                print(f"      {tgt:.3f}    |  --- unreachable at any threshold ---")
                continue
            t, tpx, fpx = ok
            f1x = 2 * tpx / (2 * tpx + fpx + (P - tpx))
            print(f"      {tgt:.3f}    | {t:.4f}  | {tpx/P:.4f} | {tpx/P:.4f} | {fpx/N:.4f} "
                  f"| {tpx+fpx:5d}  | {f1x:.4f}")
        # partial AUC over the low-FPR region
        for cap in (0.02, 0.05, 0.10, 0.20):
            sel = np.sort(np.unique(p))[::-1]
            xs, ys_ = [0.0], [0.0]
            for t in sel:
                yh = p >= t
                xs.append(int((yh & (y == 0)).sum()) / N); ys_.append(int((yh & (y == 1)).sum()) / P)
            xs, ys_ = np.array(xs), np.array(ys_)
            m = xs <= cap
            pa = np.trapz(ys_[m], xs[m]) / cap if m.sum() > 1 else 0.0
            print(f"      standardised pAUC(FPR<={cap:.2f}) = {pa:.4f}   "
                  f"(0.5 = chance-equivalent inside the region)")

        # ---------------- C.2 the d_a analogue: how separable are the FPs above the cut?
        above = p >= 0.5
        pa_, ya_ = p[above], y[above]
        tp_s, fp_s = pa_[ya_ == 1], pa_[ya_ == 0]
        da = sum(int((tp_s < s).sum()) for s in fp_s)
        print(f"\n  d_a analogue (FP outranking TP, both above the cut) = {da} of {len(tp_s)*len(fp_s)} "
              f"pairs = {da/max(len(tp_s)*len(fp_s),1):.4f}")
        below = ~above
        pb_, yb_ = p[below], y[below]
        fn_s, tn_s = pb_[yb_ == 1], pb_[yb_ == 0]
        db = sum(int((fn_s < s).sum()) for s in tn_s)
        print(f"  d_b analogue (TN outranking FN, both below the cut) = {db} of {len(fn_s)*len(tn_s)} "
              f"pairs = {db/max(len(fn_s)*len(tn_s),1):.4f}")
        print(f"  -> LB-scaled prediction of the public d_a : "
              f"{da/max(len(tp_s)*len(fp_s),1)*164*27:.0f} of 4428   "
              f"(budget d_a+d_b = 1031)")
        print(f"  -> LB-scaled prediction of the public d_b : "
              f"{db/max(len(fn_s)*len(tn_s),1)*17*125:.0f} of 2125")

        # ---------------- C.3 how far can a RAISE actually get, measured train-only?
        print("\n  RAISE curve measured on the masked OOF (the Part-A route that is available):")
        base = F1
        rows = []
        for t in np.linspace(0.50, 0.80, 31):
            f1x, tpx, fpx, fnx = f1_at(y, p, t)
            rows.append((t, f1x, tpx, fpx))
        bt = max(rows, key=lambda r: r[1])
        for t, f1x, tpx, fpx in rows[::5]:
            print(f"     t={t:.3f}  F1={f1x:.6f} ({f1x-base:+.6f})  TP={tpx} FP={fpx}")
        print(f"     best RAISE: t={bt[0]:.3f} F1={bt[1]:.6f} ({bt[1]-base:+.6f})")
        lo = [(t, *f1_at(y, p, t)) for t in np.linspace(0.20, 0.50, 31)]
        bl = max(lo, key=lambda r: r[1])
        print(f"     best LOWER: t={bl[0]:.3f} F1={bl[1]:.6f} ({bl[1]-base:+.6f})")
        print()


if __name__ == "__main__":
    stems = sys.argv[1:] or ["preds_champion_dualpol_add_seedavg5",
                             "preds_champion_distill_alphamix10",
                             "preds_jtt_lam5_s42"]
    main(stems)
