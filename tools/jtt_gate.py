"""JTT read-out — did the reweighting actually REORDER, or did it collapse to a threshold slide?

This computes exactly the three quantities pre-registered in `experiments/run_current.sh` STEP 2,
and nothing else. It exists so the decision to spend a submission slot is made against numbers that
were specified before the runs finished.

WHAT IT IS TESTING, AND WHY THAT IS THE WHOLE QUESTION.
Round 23's operative theorem is POINTWISE-LOSS ORDER INVARIANCE: for any decomposable objective the
population minimizer is `T(eta(x))` for one fixed monotone `T`, so ROC-AUC is exactly unchanged and
the F1 effect is a pure threshold slide along an unchanged ranking. Focal, ASL, LDAM, PolyLoss and
label smoothing are all in that class. JTT is supposed to ESCAPE it, because its weight is
x-dependent and class-asymmetric, giving an effective posterior
`eta_eff = eta*w1 / (eta*w1 + (1-eta)*w0)` that is not monotone in `eta` alone.

**That escape is a claim about the population optimum, and it can fail in practice.** If the network
simply cannot exploit the reweighting -- too little capacity, too few upweighted rows, the weights
washing out -- the trained model lands on essentially the same ranking and JTT degenerates to the
very thing the theorem describes. `rho > 0.999` against the control is the signature of that
degeneration, and it CLOSES the arm for zero submissions.

⚠️ (c) IS IN-SAMPLE AND MUST NOT BE READ AS TRANSFER. The recovered-false-negative count is measured
on the same 24 rows the arms upweighted during training. A high number proves only that the
mechanism fired mechanically; it is not evidence the arm generalizes. A LOW number, however, is
decisive the other way: if JTT cannot recover the rows it explicitly optimized for, it is dead.

USAGE
    python tools/jtt_gate.py --control jtt_control_s42 --arms jtt_balance_s42 jtt_lam5_s42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import spearmanr  # noqa: E402

from src.utils import combined_score, f1_at, get_logger, roc_auc  # noqa: E402

log = get_logger()
PREDS = Path(__file__).resolve().parents[1] / "submissions" / "preds"

RHO_DEGENERATE = 0.999          # pre-registered in run_current.sh STEP 2(a)


def load(name: str):
    p = PREDS / f"preds_{name}.npz"
    if not p.exists():
        raise SystemExit(f"missing bundle: {p}")
    d = np.load(p, allow_pickle=True)
    return d["y"].astype(int), d["oof_prob"].astype(float), d["p_test_raw"].astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="jtt_control_s42")
    ap.add_argument("--arms", nargs="+", default=["jtt_balance_s42", "jtt_lam5_s42"])
    ap.add_argument("--stage1", default="champion_distill_alphamix10",
                    help="bundle whose OOF errors defined the JTT error set")
    args = ap.parse_args()

    y, oof_c, test_c = load(args.control)
    _, oof_s1, _ = load(args.stage1)

    # The error set the arms were trained to fix, recomputed here rather than trusted.
    err = (oof_s1 >= 0.5).astype(int) != y
    fn = err & (y == 1)
    log.info("stage-1 error set: |E|=%d (%d false NEG, %d false POS)",
             int(err.sum()), int(fn.sum()), int((err & (y == 0)).sum()))

    def row(nm, oof, test):
        f1, auc = f1_at(y, oof, 0.5), roc_auc(y, oof)
        return (f"{nm:<20s} OOF f1 {f1:.5f} auc {auc:.5f} comb {combined_score(f1, auc):.5f} "
                f"| test pos-rate {float((test >= 0.5).mean()):.4f}")

    log.info("")
    log.info("=== OOF (blind for LEVEL -- reported, not used to decide) ===")
    log.info("  %s", row("CONTROL", oof_c, test_c))

    log.info("")
    log.info("=== STEP 2 PRE-REGISTERED READ ===")
    for a in args.arms:
        _, oof_a, test_a = load(a)
        log.info("  %s", row(a, oof_a, test_a))

        rho = spearmanr(test_c, test_a).statistic
        crossed = int(((test_c >= 0.5) != (test_a >= 0.5)).sum())
        rec = int(((oof_a >= 0.5) & fn).sum())

        log.info("      (a) rank corr vs control : %.6f%s", rho,
                 "   <- DEGENERATE: no reordering, arm CLOSED, spend no slot"
                 if rho > RHO_DEGENERATE else "   <- genuinely reordered")
        log.info("      (b) test rows changing side at 0.5 : %d / %d  (~%.1f public rows)",
                 crossed, len(test_c), crossed * 309 / 1030)
        log.info("      (c) false negatives recovered IN-SAMPLE : %d / %d  "
                 "(⚠️ in-sample, NOT transfer evidence)", rec, int(fn.sum()))
        log.info("      OOF delta vs control: f1 %+.5f  auc %+.5f  combined %+.5f",
                 f1_at(y, oof_a, 0.5) - f1_at(y, oof_c, 0.5),
                 roc_auc(y, oof_a) - roc_auc(y, oof_c),
                 combined_score(f1_at(y, oof_a, 0.5), roc_auc(y, oof_a))
                 - combined_score(f1_at(y, oof_c, 0.5), roc_auc(y, oof_c)))
        log.info("")

    log.info("READING (pre-registered, run_current.sh STEP 2/3).")
    log.info("  rho > %.3f on an arm  => JTT collapsed to a threshold slide. CLOSE it, spend no", RHO_DEGENERATE)
    log.info("  slot; that is a real result and the most likely outcome.")
    log.info("  Otherwise upload AT MOST ONE arm, preferring lambda=5 if both reordered (the less")
    log.info("  aggressive intervention is the safer unmeasured artifact).")
    log.info("  ⚠️ The OOF deltas above are BLIND for level -- OOF has spanned 0.72-0.907 public at")
    log.info("  a constant ~0.97 in this project. They are printed for the record, not to decide.")
    log.info("  ⚠️ Finalists are not at risk: a single-seed arm cannot replace a 10-seed measured")
    log.info("  artifact under the standing rule, whatever it scores.")


if __name__ == "__main__":
    main()
