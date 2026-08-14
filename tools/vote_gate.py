"""Decision-level pooling gate — does a HARD MAJORITY VOTE beat probability averaging at 0.5?

WHY THIS EXISTS
---------------
Round 23 converged on this from three independent directions, which is why it is worth a free
measurement rather than an argument:

  * Our shipped pooling operator averages per-member CALIBRATED probabilities and then cuts at a
    literal 0.5. Ranjan & Gneiting (JRSS-B 2010, 72(1):71-91) prove that any non-trivial average of
    distinct CALIBRATED forecasts is necessarily UNCALIBRATED and under-dispersed toward the centre.
    So the operator we ship is provably mis-specified -- which is prong (c) of our own legality test.
  * The failure mode is concrete and it is exactly our failure mode. If six members say 0.55 and four
    say 0.10, the mean is 0.37 and the row is NEGATIVE, even though a clear majority of members
    individually call it positive. Averaging lets a confident minority veto a weak majority. Since
    our public confusion cell shows we UNDER-predict positives, this is the right direction to probe.
  * Both external round-23 reports and our own Q5 agent independently proposed decision-level
    pooling, and it is one of the very few candidates that is NOT killed by the pointwise-loss
    theorem (see below) -- because it is not a pointwise loss at all.

⚠️ WHY MOST OF ROUND 23's OTHER CANDIDATES ARE DEAD, AND THIS ONE IS NOT.
The operative theorem from round 23 is POINTWISE-LOSS ORDER INVARIANCE: for any decomposable
objective `sum_i [y_i*l1(z_i) + (1-y_i)*l0(z_i)]`, the population minimizer is `T(eta(x))` for ONE
fixed monotone `T`, so ROC-AUC is exactly unchanged and the F1 effect is a pure threshold slide along
an unchanged ranking. Focal loss, ASL, label smoothing, LDAM and PolyLoss are all in that class
(Charoenphakdee et al., CVPR 2021, arXiv:2011.09172, Thm 3/5/11 + Lemma 14: focal's warp is
*strictly order-preserving*). A hard vote is not a pointwise loss and does not reorder a score at
all -- it replaces the aggregation with a different partition of the rows. It escapes the theorem.

THE TWO-COLUMN DESIGN, WHICH IS THE POINT NEITHER EXTERNAL REPORT MADE.
The metric scores two separate columns. A vote COUNT takes only `n+1` distinct values, so using it as
`TargetRAUC` would collapse our ranking and torch the 40% of the score we are already winning
(our AUC 0.945842 beats the leader's 0.944897). So the vote may only ever touch `TargetF1`:

    TargetRAUC  <- the averaged calibrated probability   (UNCHANGED, AUC bit-identical)
    TargetF1    <- 1[ #members with p_i >= 0.5  >=  ceil(n/2) ]

The two columns are scored independently, so this is a free, exactly-isolated change to the 60% term.

LEGALITY (our three prongs). (a) every member cuts at a LITERAL 0.5 and the aggregation rule is the
unweighted strict majority -- no threshold anywhere is fitted. (b) train-only: nothing here reads the
leaderboard or a realized positive-rate target. (c) it corrects a DEMONSTRABLY mis-specified operator
(Ranjan-Gneiting), rather than relabelling a fixed estimate.

⚠️ THE ONE KNOB, AND IT IS PRE-COMMITTED. The vote fraction `k/n` is a free parameter and tuning it
against F1 WOULD be threshold tuning. It is therefore fixed a priori at the strict majority
`k = ceil(n/2)`. The full sweep over `k` is printed for DIAGNOSIS ONLY and must not select `k`.

HONESTY ABOUT THE INSTRUMENT. OOF is BLIND for LEVEL in this project (OOF ~0.97 for artifacts
spanning 0.72-0.907 public). This gate is therefore only trustworthy as a VETO: if the vote loses on
cross-fitted OOF it is dead, and if it wins the win does not transfer automatically. Platt is
CROSS-FITTED here so the comparison is not read in-sample -- iter46 showed in-sample and cross-fitted
calibration numbers can disagree in SIGN at n=1817.

USAGE
    python tools/vote_gate.py                  # the amix10 pool (finalist's one-variable pair)
    python tools/vote_gate.py --pool dpa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.utils import combined_score, f1_at, get_logger, roc_auc  # noqa: E402

log = get_logger()

PREDS = Path(__file__).resolve().parents[1] / "submissions" / "preds"
_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def platt_xfit(y, p_oof, p_test, seed=42, n_splits=5):
    """Cross-fitted Platt on OOF + a full-fit map for test.

    The held-out OOF probabilities are what the operator comparison is read on, so that neither
    operator is scored on rows its calibrator has already seen.
    """
    y = np.asarray(y).astype(int)
    z = _logit(p_oof).reshape(-1, 1)
    oof_cal = np.zeros(len(y))
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(z, y):
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z[tr], y[tr])
        oof_cal[va] = lr.predict_proba(z[va])[:, 1]
    full = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)
    test_cal = full.predict_proba(_logit(p_test).reshape(-1, 1))[:, 1]
    return oof_cal, test_cal, float(full.coef_[0][0])


def load_pool(tag: str):
    files = sorted(PREDS.glob(f"preds_{tag}_s*.npz"))
    if not files:
        raise SystemExit(f"no bundles matching preds_{tag}_s*.npz in {PREDS}")
    y = None
    oof, test, names = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if y is None:
            y = d["y"].astype(int)
        elif not np.array_equal(y, d["y"].astype(int)):
            raise SystemExit(f"{f.name}: label vector differs from the rest of the pool")
        oof.append(d["oof_prob"].astype(float))
        test.append(d["p_test_raw"].astype(float))
        names.append(f.stem.replace("preds_", ""))
    return y, np.array(oof), np.array(test), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="amix", help="bundle prefix, e.g. amix or dpa")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    y, oof_raw, test_raw, names = load_pool(args.pool)
    n = len(names)
    log.info("pool '%s': %d members -> %s", args.pool, n, ", ".join(names))
    log.info("rows: %d train (prior %.4f), %d test", len(y), y.mean(), test_raw.shape[1])

    oof_cal, test_cal = [], []
    for i, nm in enumerate(names):
        o, t, slope = platt_xfit(y, oof_raw[i], test_raw[i], seed=args.seed)
        oof_cal.append(o)
        test_cal.append(t)
        log.info("  %-10s xfit-Platt slope %.3f | member OOF f1@0.5 %.4f | member test pos-rate %.4f",
                 nm, slope, f1_at(y, o, 0.5), float((t >= 0.5).mean()))
    oof_cal, test_cal = np.array(oof_cal), np.array(test_cal)

    # ---- the two operators -------------------------------------------------------------------
    oof_mean, test_mean = oof_cal.mean(0), test_cal.mean(0)
    oof_votes = (oof_cal >= 0.5).sum(0)
    test_votes = (test_cal >= 0.5).sum(0)
    k_major = int(np.ceil(n / 2))          # PRE-COMMITTED. Not selected from anything below.

    f1_mean = f1_at(y, oof_mean, 0.5)
    auc_mean = roc_auc(y, oof_mean)
    f1_vote = f1_at(y, (oof_votes >= k_major).astype(float), 0.5)

    log.info("")
    log.info("=== OPERATOR COMPARISON on CROSS-FITTED OOF (n=%d rows) ===", len(y))
    log.info("  A shipped  (mean prob >= 0.5) : f1 %.5f | auc %.5f | combined %.5f | pos-rate %.4f",
             f1_mean, auc_mean, combined_score(f1_mean, auc_mean), float((oof_mean >= 0.5).mean()))
    log.info("  B majority (>= %d of %d votes) : f1 %.5f | auc %.5f (UNCHANGED by design) | "
             "combined %.5f | pos-rate %.4f",
             k_major, n, f1_vote, auc_mean, combined_score(f1_vote, auc_mean),
             float((oof_votes >= k_major).mean()))
    log.info("  DELTA f1 = %+.5f   ->  composite %+.5f", f1_vote - f1_mean,
             0.6 * (f1_vote - f1_mean))

    # ---- where the two operators actually differ ---------------------------------------------
    a_oof = (oof_mean >= 0.5).astype(int)
    b_oof = (oof_votes >= k_major).astype(int)
    diff = a_oof != b_oof
    log.info("")
    log.info("  rows where the operators DISAGREE on OOF: %d / %d (%.2f%%)",
             diff.sum(), len(y), 100 * diff.mean())
    if diff.sum():
        up = diff & (b_oof == 1)
        dn = diff & (b_oof == 0)
        log.info("    vote flips UP   (0->1): %4d rows, of which %d are TRUE positives (%.1f%% precision)",
                 up.sum(), int(y[up].sum()), 100 * y[up].mean() if up.sum() else 0.0)
        log.info("    vote flips DOWN (1->0): %4d rows, of which %d are TRUE positives (%.1f%% = loss rate)",
                 dn.sum(), int(y[dn].sum()), 100 * y[dn].mean() if dn.sum() else 0.0)
        log.info("    NET true positives gained by voting: %+d",
                 int(y[up].sum()) - int(y[dn].sum()))

    # ---- the k sweep: DIAGNOSIS ONLY ----------------------------------------------------------
    log.info("")
    log.info("  k sweep (⚠️ DIAGNOSIS ONLY -- k is PRE-COMMITTED to %d, never selected from here):",
             k_major)
    for k in range(1, n + 1):
        f1k = f1_at(y, (oof_votes >= k).astype(float), 0.5)
        mark = "   <- PRE-COMMITTED strict majority" if k == k_major else ""
        log.info("    k=%2d  OOF f1 %.5f  pos-rate %.4f%s", k, f1k,
                 float((oof_votes >= k).mean()), mark)

    # ---- what it would do to the actual submission -------------------------------------------
    a_test = (test_mean >= 0.5).astype(int)
    b_test = (test_votes >= k_major).astype(int)
    log.info("")
    log.info("=== ON THE 1030 TEST ROWS (no labels -- magnitude only) ===")
    log.info("  A shipped  pos-rate %.4f | B majority pos-rate %.4f", a_test.mean(), b_test.mean())
    log.info("  rows changing decision: %d / %d (%.2f%%)  [up %d, down %d]",
             int((a_test != b_test).sum()), len(a_test), 100 * (a_test != b_test).mean(),
             int(((a_test != b_test) & (b_test == 1)).sum()),
             int(((a_test != b_test) & (b_test == 0)).sum()))
    log.info("  ~public-slice rows affected (x309/1030): %.1f",
             (a_test != b_test).sum() * 309 / 1030)

    log.info("")
    log.info("READING. The AUC column is UNCHANGED by construction -- the vote touches TargetF1 only,")
    log.info("because a vote COUNT has n+1 distinct values and would destroy the ranking we already")
    log.info("win. So the composite delta is exactly 0.6 x (F1 delta).")
    log.info("⚠️ OOF IS BLIND FOR LEVEL in this project. Treat this gate as a VETO: a LOSS here kills")
    log.info("the operator for free; a WIN does not transfer automatically and still has to clear the")
    log.info("0.015 binomial bar on 309 public rows to be worth a submission slot.")


if __name__ == "__main__":
    main()
