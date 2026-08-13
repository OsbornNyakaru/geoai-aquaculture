#!/usr/bin/env python
"""graph_gate.py -- the OFFLINE, ZERO-SUBMISSION graph-structure diagnostic (round-21).

WHY THIS EXISTS
---------------
Two separate questions, one instrument.

(1) CAN GRAPH ML PLAY ANY ROLE HERE? The natural graph for pond detection is SPATIAL -- ponds
    cluster, so a cell beside a pond is likely a pond. It does not exist for us: Train.csv is
    ID + label + 12 bands x 12 months and carries NO lat/lon/tile/region column, and the IDs are
    random Crockford base32 (no I/O/0/1), non-sequential, with zero train/test overlap. Rebuilding
    location would need external data, which the rules forbid. What remains is a SIMILARITY graph
    over rows in feature space, and whether that is usable is an empirical question, not an
    argument. This tool measures it.

(2) IS OUR OPERATING POINT IN THE RIGHT PLACE? This is the more valuable question and the reason
    the tool exists at all. LB_LOG iteration 43 states outright that "whether our operating pos-rate
    is too LOW is OPEN, not closed": the two estimators that could settle it (MLLS 0.578, BBSE
    0.559) were RETIRED at iteration 41 when label_shift_gate.py's KS test rejected p(x|y)
    invariance at p~=0, and an estimator retired for correction cannot be re-used as evidence that
    no correction is warranted. We have had no valid independent estimate since.

    k-NN label propagation is a THIRD, structurally different estimator. It assumes no label shift
    at all -- only feature adjacency plus train labels -- so the failure that retired MLLS and BBSE
    does not apply to it. That makes its implied test positive rate genuinely new evidence.

WHAT IT MEASURES (all on the MASK-MATCHED replica -- train rows masked to test-like contiguous 4-6
month windows through the same _mask_views(..., oof=True) the pipeline calibrates on):

  1. CONNECTIVITY  what fraction of a test row's k neighbours are LABELLED, vs the random-mixing
                   baseline. This is a precondition: near 0% and every graph method is dead on
                   arrival because propagation has no source.
  2. HOMOPHILY     do adjacent rows share a label, against the chance rate. Near chance and the
                   graph carries no signal regardless of connectivity.
  3. PROPAGATION   5-fold, held-out rows MASKED, neighbours drawn from the UNMASKED labelled pool.
                   That is the train(12mo, labelled) -> test(4-6mo) direction, run parameter-free.
  4. PREVALENCE    the implied test positive rate, swept over k, printed beside our champion's own
                   realized rate. This is the answer to question (2).

    ⚠️ THE CAVEAT THAT GOVERNS EVERY NUMBER BELOW. The replica reproduces the WINDOW MASKING but NOT
    the temporal domain shift, which is the larger half of the problem (adversarial AUC ~0.99). So
    every figure here is OPTIMISTIC -- an upper bound on what the graph can do in deployment. The
    asymmetry that follows is the whole discipline of reading this tool: "the graph agrees our
    pos-rate is roughly right" is the ROBUST conclusion, because a shift can only make the graph
    worse and it already agrees. "The graph says move the operating point" would be the FRAGILE one
    and must not be acted on alone. Same structure as delta_hat in tools/regime_match.py.

COMPLIANCE. Train labels + UNLABELLED test features only. Transductive use of test features is
established practice in this pipeline (seq.transduct, seq.distill, iteration 41). Nothing this tool
prints may be fed to the operating point -- it is DIAGNOSIS ONLY, exactly like the leaderboard-
inverted quantities in REPORT.md section 8.2. In particular, the prevalence estimate in block 4 must
never be used to move the 0.5 cut; it is reported so we know whether the cut is already right.

USAGE
    python tools/graph_gate.py                       # the full gate, seed 42
    python tools/graph_gate.py --k 10 --seed 7
    python tools/graph_gate.py --no-mask             # CONTROL: isolates the cost of masking
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_bundle  # noqa: E402
from src.seq_model import _mask_views  # noqa: E402
from src.utils import combined_score, f1_at, get_logger, load_config, roc_auc  # noqa: E402

log = get_logger()

# Our champion's own realized test positive rate, logged across iterations 41-43. Printed as the
# comparator in block 4. This is OUR number from OUR run log -- not a leaderboard-derived quantity.
CHAMPION_POS_RATE = 0.587


def pooled(cube: np.ndarray) -> np.ndarray:
    """Per-band mean over OBSERVED months -> [n, n_bands].

    n-INVARIANT by construction, and that is the whole point. A test row shows 4-6 months and a
    train row shows 12; any statistic whose VALUE depends on how many months went into it would
    make the graph measure the masking pattern instead of the signal, which is precisely why the
    adversarial AUC is ~0.99. A mean over observed months is the same quantity at any window length.
    """
    with np.errstate(invalid="ignore"):
        return np.nanmean(cube, axis=1)


def _standardize(X: np.ndarray, mu=None, sd=None):
    """Impute column means for all-NaN bands, then z-score. Stats come from the REFERENCE set."""
    X = np.asarray(X, dtype=float)
    col = np.nanmean(X, axis=0)
    col = np.where(np.isnan(col), 0.0, col)
    X = np.where(np.isnan(X), col, X)
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


def _knn(ref: np.ndarray, query: np.ndarray, k: int):
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors(n_neighbors=k).fit(ref).kneighbors(query)


def block_connectivity(Ftr, Fte, k):
    """Can label information reach the test cloud at all?"""
    X = np.vstack([Ftr, Fte])
    Xs, _, _ = _standardize(X)
    ntr = len(Ftr)
    _, idx = _knn(Xs, Xs, k + 1)
    idx = idx[:, 1:]                                    # drop self
    is_tr = np.arange(len(Xs)) < ntr
    frac_lab = float((idx[~is_tr] < ntr).mean())
    baseline = ntr / (len(Xs) - 1.0)                    # what random mixing would give
    log.info("")
    log.info("[1] CONNECTIVITY  (can label information reach the test cloud?)")
    log.info("    test-row neighbours that are LABELLED : %.1f%%", 100 * frac_lab)
    log.info("    random-mixing baseline                : %.1f%%", 100 * baseline)
    log.info("    -> test rows are %.1fx more likely to neighbour other TEST rows than chance;",
             (1 - frac_lab) / (1 - baseline))
    log.info("       ~%.1f labelled neighbours per test row.", frac_lab * k)
    if frac_lab < 0.02:
        log.info("    VERDICT: DEAD. Propagation has no source -- do not pursue any graph method.")
    else:
        log.info("    VERDICT: propagation is WELL-POSED (a source exists). Necessary, not sufficient")
        log.info("             -- block 2 decides whether adjacency carries label signal.")
    return frac_lab, baseline


def block_homophily(Ftr, y, k):
    """Does adjacency carry label signal, or is it just proximity?"""
    Xs, _, _ = _standardize(Ftr)
    _, idx = _knn(Xs, Xs, k + 1)
    idx = idx[:, 1:]
    hom = float((y[idx] == y[:, None]).mean())
    chance = float(y.mean() ** 2 + (1 - y.mean()) ** 2)
    log.info("")
    log.info("[2] HOMOPHILY  (does adjacency carry label signal?)")
    log.info("    train-train edge label agreement : %.4f", hom)
    log.info("    chance rate at prior %.4f       : %.4f  (lift %+.4f)", y.mean(), chance, hom - chance)
    if hom - chance < 0.05:
        log.info("    VERDICT: DEAD. Neighbours are no more alike than random pairs.")
    else:
        log.info("    VERDICT: strong. But note this restates IN-DISTRIBUTION separability, which")
        log.info("             this project has already shown does NOT imply transfer (OOF 0.975 vs")
        log.info("             LB 0.90). Block 3 is the one that tests the deployment direction.")
    return hom, chance


def block_propagation(Ffull, Fmask, y, ks, seed):
    """Parameter-free label propagation, held-out rows masked to the deployment regime."""
    from sklearn.model_selection import StratifiedKFold
    log.info("")
    log.info("[3] PROPAGATION  (held-out rows MASKED, neighbours from the UNMASKED labelled pool)")
    out = {}
    for k in ks:
        oof = np.zeros(len(y))
        for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
            Xa, mu, sd = _standardize(Ffull[tr_i])
            Xv, _, _ = _standardize(Fmask[va_i], mu, sd)
            d, idx = _knn(Xa, Xv, k)
            w = 1.0 / (d + 1e-6)                        # distance-weighted vote
            oof[va_i] = (w * y[tr_i][idx]).sum(1) / w.sum(1)
        f1, auc = f1_at(y, oof, 0.5), roc_auc(y, oof)
        out[k] = (auc, f1, combined_score(f1, auc))
        log.info("    k=%-3d AUC %.4f | F1@0.5 %.4f | combined %.4f", k, auc, f1, out[k][2])
    log.info("    (parameter-free: no training, no calibration, no fitted threshold)")
    return out


def block_prevalence(Ftr, Fte, y, ks):
    """The independent test-prevalence estimate -- the reason this tool is worth having."""
    Xtr, mu, sd = _standardize(Ftr)
    Xte, _, _ = _standardize(Fte, mu, sd)
    log.info("")
    log.info("[4] PREVALENCE  (an estimator that assumes NO label shift -- unlike MLLS/BBSE)")
    log.info("    train prior                        : %.4f", y.mean())
    log.info("    our champion's realized test rate  : %.4f   (from our own run log)", CHAMPION_POS_RATE)
    rates = []
    for k in ks:
        d, idx = _knn(Xtr, Xte, k)
        w = 1.0 / (d + 1e-6)
        p = (w * y[idx]).sum(1) / w.sum(1)
        rate = float((p >= 0.5).mean())
        rates.append(rate)
        log.info("    k=%-3d implied test pos-rate %.4f | mean p %.4f | %4.1f%% of mass in [.45,.55]",
                 k, rate, p.mean(), 100 * float(((p >= 0.45) & (p <= 0.55)).mean()))
    spread = max(rates) - min(rates)
    log.info("    spread across k: %.4f %s", spread,
             "(FLAT in k -> not an artifact of the neighbourhood size)" if spread < 0.02
             else "(SENSITIVE to k -> treat the estimate as unstable)")
    log.info("    delta vs our operating point: %+.4f", float(np.mean(rates)) - CHAMPION_POS_RATE)
    return rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="neighbourhood size for blocks 1-2")
    ap.add_argument("--ks", type=int, nargs="*", default=[5, 10, 25, 50], help="sweep for blocks 3-4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mask", action="store_true",
                    help="CONTROL: leave train rows at 12 months. Isolates the cost of masking; the "
                         "numbers become a strict upper bound with no deployment meaning.")
    args = ap.parse_args()

    cfg = load_config()
    cfg["seed"] = args.seed
    b = load_bundle(cfg)
    tr, te, y = b.train_cube, b.test_cube, b.y

    log.info("=" * 78)
    log.info("GRAPH GATE -- offline, zero submissions.  seed=%d  k=%d  %s",
             args.seed, args.k, "NO-MASK CONTROL" if args.no_mask else "mask-matched replica")
    log.info("train %s | test %s | train prior %.4f", tr.shape, te.shape, y.mean())
    log.info("=" * 78)

    Ffull = pooled(tr)
    if args.no_mask:
        Fmask = Ffull
    else:
        # The same masked replica the pipeline calibrates on: contiguous 4-6 month windows drawn
        # from the MEASURED test window distribution, seeded per (row, view) via rng_for.
        trm, _ = _mask_views(tr, np.arange(len(tr)), b.schema, b.window_dist, cfg, 1, args.seed,
                             oof=True)
        Fmask = pooled(trm)
    Fte = pooled(te)

    block_connectivity(Fmask, Fte, args.k)
    block_homophily(Fmask, y, args.k)
    block_propagation(Ffull, Fmask, y, args.ks, args.seed)
    block_prevalence(Fmask, Fte, y, args.ks)

    log.info("")
    log.info("=" * 78)
    log.info("HOW TO READ THIS. The replica reproduces the WINDOW MASKING but NOT the temporal")
    log.info("domain shift (adversarial AUC ~0.99), which is the larger half of the problem. Every")
    log.info("number above is therefore OPTIMISTIC -- an upper bound on deployment behaviour.")
    log.info("")
    log.info("The asymmetry that follows is the whole discipline of this tool:")
    log.info("  ROBUST   'the graph AGREES our operating point is roughly right' -- a shift can only")
    log.info("           make the graph worse, and it already agrees.")
    log.info("  FRAGILE  'the graph says MOVE the operating point' -- never act on this alone.")
    log.info("")
    log.info("DIAGNOSIS ONLY. Nothing here may reach the 0.5 cut. See REPORT.md section 8.2.")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
