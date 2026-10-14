"""Which offline, zero-submission criterion would have predicted the PRIVATE leaderboard?

Post-mortem tool. Competition CLOSED. Nothing here touches Zindi.

Joins `submissions/preds/*.npz` to `experiments/zindi_submissions_final.tsv` by filename,
validates the join by CSV content via exact F1-cell inversion, computes a panel of
zero-submission criteria from OOF predictions only, and ranks those criteria against the
TRUE private composite by (a) rank correlation and (b) -- weighted far more heavily --
top-2 selection accuracy.

Hard rules honoured here:
  * 0.5 is a literal 0.5 everywhere. We study which criterion PREDICTS, we do not tune a cut.
  * The true test prevalence 0.5437 is DIAGNOSTIC ONLY. Criteria that use it are prefixed
    `DQ_` and are excluded from every recommendation.
  * No test labels are used or reconstructed. `test_per_fold` supplies model outputs only.

Usage:  python tools/offline_criterion_bakeoff.py
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from fractions import Fraction as F

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import combined_score, f1_at, roc_auc  # noqa: E402

from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import average_precision_score, roc_curve  # noqa: E402

LEDGER = os.path.join(ROOT, "experiments", "zindi_submissions_final.tsv")
PREDS = os.path.join(ROOT, "submissions", "preds")
SUBS = os.path.join(ROOT, "submissions")

# Split solved exactly by the half-integer AUC sieve (see tools/post_mortem.py section 2).
N_PUB, P_PUB = 333, 181
N_PRV, P_PRV = 697, 379

TRAIN_PRIOR = 0.4023          # legal: known from the training labels
TRUE_TEST_PREV = 0.5437       # DIAGNOSTIC ONLY -- never available before the reveal

RNG = np.random.default_rng(20260817)
N_BOOT = 20000


# ----------------------------------------------------------------------------- #
# F1-cell inversion -- local enumerator returning ALL solutions
# ----------------------------------------------------------------------------- #
def invert_f1_all(f1: float, n: int, P: int) -> list[tuple[int, int]]:
    """Every (TP, PP) consistent with a 9-decimal F1 on n rows containing P positives.

    F1 = 2*TP / (PP + P). This used to be a private copy, because
    tools.post_mortem.invert_f1 silently returned the median candidate on an ambiguous
    inversion. That bug is fixed (2026-08-17): post_mortem now exposes invert_f1_all()
    returning the whole solution set, and invert_f1() returning (cell, n_solutions). One
    enumerator, so the two reports cannot disagree.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from post_mortem import invert_f1_all as _impl   # same tools/ directory
    return _impl(f1, n, P)


# ----------------------------------------------------------------------------- #
# Criterion helpers
# ----------------------------------------------------------------------------- #
def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def partial_auc(y, p, fmax):
    """Area under the ROC curve restricted to FPR in [0, fmax], plus McClish standardisation.

    Returns (raw_area, normalised, mcclish). `normalised` = raw / fmax, i.e. the average
    TPR over that FPR band. McClish A' = 0.5 * (1 + (raw - min) / (max - min)) with
    min = fmax**2 / 2 (chance) and max = fmax (perfect).
    """
    fpr, tpr, _ = roc_curve(y, p)
    keep = fpr <= fmax
    f = np.concatenate([fpr[keep], [fmax]])
    t = np.concatenate([tpr[keep], [np.interp(fmax, fpr, tpr)]])
    raw = float(np.trapezoid(t, f)) if hasattr(np, "trapezoid") else float(np.trapz(t, f))
    lo, hi = fmax**2 / 2.0, fmax
    return raw, raw / fmax, 0.5 * (1.0 + (raw - lo) / (hi - lo))


def prec_rec_at_k(y, p, k):
    """Precision and recall when the top-k scoring rows are called positive."""
    k = int(k)
    idx = np.argsort(-np.asarray(p, dtype=float), kind="stable")[:k]
    tp = int(np.sum(np.asarray(y)[idx]))
    return tp / max(k, 1), tp / max(int(np.sum(y)), 1)


def f1_at_topk(y, p, k):
    """F1 when exactly the top-k rows are called positive (a rate-pinned operating point)."""
    k = int(k)
    order = np.argsort(-np.asarray(p, dtype=float), kind="stable")
    pred = np.zeros(len(y), dtype=int)
    pred[order[:k]] = 1
    tp = int(np.sum(pred * np.asarray(y)))
    return 2 * tp / (k + int(np.sum(y))) if (k + np.sum(y)) else 0.0


def best_f1(y, p):
    """Oracle-threshold F1: the best F1 achievable by any cut on these scores."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p, dtype=float), kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    return float(np.max(2 * tp / (k + y.sum())))


def ece(y, p, bins=10):
    """Expected calibration error, 10 equal-width bins on [0, 1]."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _rowwise_corr(A, B):
    """Pearson correlation of each row of A against the matching row of B."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    out = np.full(len(num), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def boot_spearman(v, prv, I):
    """Vectorised bootstrap Spearman: one value per resample row of index matrix I."""
    return _rowwise_corr(rankdata(np.asarray(v, float)[I], axis=1),
                         rankdata(np.asarray(prv, float)[I], axis=1))


def pearson(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


# ----------------------------------------------------------------------------- #
# Load
# ----------------------------------------------------------------------------- #
def load_ledger():
    cols = ["id", "when_raw", "file", "sel", "pub", "prv",
            "auc_pub", "auc_prv", "f1_pub", "f1_prv"]
    out = []
    with open(LEDGER, newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            d = dict(zip(cols, row))
            for c in cols[4:]:
                d[c] = float(d[c])
            d["stem"] = d["file"][len("submission_"):-4] if d["file"].startswith("submission_") else None
            out.append(d)
    return out


def build_panel(npz):
    """Every zero-submission criterion, from OOF predictions and (optionally) fold/view arrays."""
    y = npz["y"].astype(int)
    p = npz["oof_prob"].astype(float)
    n = len(y)
    c: dict[str, float] = {}

    auc = roc_auc(y, p)
    f1 = f1_at(y, p, 0.5)                       # 0.5 is literal, always
    c["oof_auc"] = auc                          # <- the incumbent criterion
    c["oof_f1@0.5"] = f1
    c["oof_composite"] = combined_score(f1, auc)

    c["avg_precision"] = float(average_precision_score(y, p))
    c["brier"] = -float(np.mean((p - y) ** 2))  # negated: higher is better everywhere
    c["ece"] = -ece(y, p)                       # negated
    c["oof_posrate@0.5"] = float(np.mean(p >= 0.5))

    fstar = best_f1(y, p)
    c["oof_f1_star"] = fstar
    c["calib_gap(F1*-F1@.5)"] = -(fstar - f1)   # negated: a small gap is the good sign

    for fmax, tag in ((0.20, "20"), (0.10, "10")):
        raw, norm, mcc = partial_auc(y, p, fmax)
        c[f"pauc@fpr{tag}_raw"] = raw
        c[f"pauc@fpr{tag}_norm"] = norm
        c[f"pauc@fpr{tag}_mcclish"] = mcc

    k_prior = int(round(TRAIN_PRIOR * n))
    pr, rc = prec_rec_at_k(y, p, k_prior)
    c["prec@k_prior"] = pr
    c["recall@k_prior"] = rc
    c["f1@k_prior"] = f1_at_topk(y, p, k_prior)

    k_true = int(round(TRUE_TEST_PREV * n))
    pr2, rc2 = prec_rec_at_k(y, p, k_true)
    c["DQ_prec@k_truePrev"] = pr2               # DIAGNOSTIC ONLY -- uses the revealed 0.5437
    c["DQ_recall@k_truePrev"] = rc2             # DIAGNOSTIC ONLY
    c["DQ_f1@k_truePrev"] = f1_at_topk(y, p, k_true)  # DIAGNOSTIC ONLY

    # ---- deployment-shift statistics (only where the arrays exist) ----
    if "oof_view_p" in npz.files and len(npz["oof_view_p"]) == 2 * n:
        vp = npz["oof_view_p"].astype(float)
        own = npz["oof_view_owner"].astype(int)
        # NB: the discriminating index is `oof_view_k` (0/1 = the two masked views).
        # `oof_view_rep` is identically 0 in every bundle and carries no information.
        rep = npz["oof_view_k"].astype(int)
        reps = np.unique(rep)
        if len(reps) == 2:
            a = np.full(n, np.nan)
            b = np.full(n, np.nan)
            a[own[rep == reps[0]]] = vp[rep == reps[0]]
            b[own[rep == reps[1]]] = vp[rep == reps[1]]
            ok = ~(np.isnan(a) | np.isnan(b))
            c["view_absdiff"] = -float(np.mean(np.abs(a[ok] - b[ok])))       # negated
            c["view_logit_var"] = -float(np.mean(((_logit(a[ok]) - _logit(b[ok])) / 2) ** 2))
            c["view_auc_gap"] = -abs(roc_auc(y[ok], a[ok]) - roc_auc(y[ok], b[ok]))
            c["view_mean_auc"] = 0.5 * (roc_auc(y[ok], a[ok]) + roc_auc(y[ok], b[ok]))
            c["view_flip@0.5"] = -float(np.mean((a[ok] >= 0.5) != (b[ok] >= 0.5)))

    if "test_per_fold" in npz.files:
        tf = npz["test_per_fold"].astype(float)                # (n_folds, 1030) -- OUTPUTS only
        side = (tf >= 0.5).sum(axis=0)
        c["test_boundary_stability"] = -float(np.mean((side > 0) & (side < tf.shape[0])))
        c["test_fold_logit_sd"] = -float(np.mean(np.std(_logit(tf), axis=0)))

    return c


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main() -> None:
    ledger = load_ledger()
    by_stem = {r["stem"]: r for r in ledger if r["stem"]}
    bundles = {os.path.basename(p)[len("preds_"):-4]: p
               for p in sorted(glob.glob(os.path.join(PREDS, "*.npz")))}

    joined = sorted(set(by_stem) & set(bundles))
    print("=" * 78)
    print("STEP 1  JOIN")
    print("=" * 78)
    print(f"ledger rows        : {len(ledger)}")
    print(f"unique filenames   : {len(by_stem)}")
    print(f"bundles on disk    : {len(bundles)}")
    print(f"JOINED             : {len(joined)}  "
          f"({100*len(joined)/len(bundles):.0f}% of bundles, "
          f"{100*len(joined)/len(ledger):.0f}% of ledger rows)")

    # ---- content validation of the name join, where a CSV survives ----
    print("\nSTEP 1b  CONTENT VALIDATION of the name join (CSV vs inverted F1 cells)")
    print(f"{'stem':<32} {'csv_npred':>9} {'inv_pub':>9} {'inv_prv':>9} {'sum':>7} "
          f"{'match':>6} {'p>=.5==TargetF1':>16}")
    n_val = n_ok = 0
    for s in joined:
        csv_path = os.path.join(SUBS, f"submission_{s}.csv")
        if not os.path.exists(csv_path):
            continue
        n_val += 1
        ids, hard, prob = [], [], []
        with open(csv_path, newline="") as fh:
            rd = csv.DictReader(fh)
            for row in rd:
                ids.append(row["ID"])
                hard.append(int(row["TargetF1"]))
                prob.append(float(row["TargetRAUC"]))
        hard = np.array(hard)
        prob = np.array(prob)
        r = by_stem[s]
        pub = invert_f1_all(r["f1_pub"], N_PUB, P_PUB)
        prv = invert_f1_all(r["f1_prv"], N_PRV, P_PRV)
        pub_pp = sorted({pp for _, pp in pub})
        prv_pp = sorted({pp for _, pp in prv})
        tot = [a + b for a in pub_pp for b in prv_pp]
        ok = int(hard.sum()) in tot
        n_ok += ok
        thr_ok = bool(np.all(hard == (prob >= 0.5).astype(int)))
        print(f"{s:<32} {hard.sum():>9} "
              f"{str(pub_pp if len(pub_pp) < 4 else f'{len(pub_pp)} cands'):>9} "
              f"{str(prv_pp if len(prv_pp) < 4 else f'{len(prv_pp)} cands'):>9} "
              f"{str(sorted(set(tot))[:3]):>7} {'OK' if ok else 'FAIL':>6} {str(thr_ok):>16}")
    print(f"  -> {n_ok}/{n_val} joins with a surviving CSV validated by content.")

    # ---- panels ----
    print("\n" + "=" * 78)
    print("STEP 2  CRITERION PANEL")
    print("=" * 78)
    rows = []
    for s in joined:
        npz = np.load(bundles[s], allow_pickle=True)
        c = build_panel(npz)
        r = by_stem[s]
        rows.append({"stem": s, "pub": r["pub"], "prv": r["prv"],
                     "auc_prv": r["auc_prv"], "f1_prv": r["f1_prv"],
                     "n_oof": len(npz["y"]), "_path": bundles[s], **c})
    crits = [k for k in rows[0] if k not in
             ("stem", "pub", "prv", "auc_prv", "f1_prv", "n_oof", "_path")]
    full = [k for k in crits if all(k in r for r in rows)]
    partial = [k for k in crits if k not in full]
    print(f"criteria computable on all {len(rows)} bundles : {len(full)}")
    print(f"criteria on a subset only                : {partial}")

    for tag, subset in (("n=13 (incl. smoke_test)", rows),
                        ("n=12 (smoke_test dropped)", [r for r in rows if r["stem"] != "smoke_test"])):
        analyse(tag, subset, crits)

    stress_test(rows, crits, winner="view_auc_gap")
    decompose(rows, crits)


def analyse(tag, rows, crits):
    print("\n" + "=" * 78)
    print(f"STEP 3/4  RANKING + BOOTSTRAP  --  {tag}")
    print("=" * 78)
    prv = np.array([r["prv"] for r in rows])
    pub = np.array([r["pub"] for r in rows])
    n = len(rows)
    stems = [r["stem"] for r in rows]

    oracle = float(prv.max())
    oracle_stem = stems[int(prv.argmax())]
    # "select 2, Zindi scores the better of them" -- as the competition actually worked
    pub_pick = _top2(pub, stems, prv)
    incumb = _top2(np.array([r["oof_auc"] for r in rows]), stems, prv)

    print(f"universe oracle (best-of-any) : {oracle:.9f}  [{oracle_stem}]")
    print(f"public-LB top-2 -> best-of-2  : {pub_pick[0]:.9f}  {pub_pick[1]}")
    print(f"OOF-AUC top-2   -> best-of-2  : {incumb[0]:.9f}  {incumb[1]}")
    print(f"random 2-of-{n} expectation    : {_random_exp(prv, 2):.9f}")
    print(f"contestable headroom (oracle - public-LB pick) : {oracle - pub_pick[0]:+.9f}")

    # One shared resample matrix: keeps every criterion on identical resamples, so the
    # paired comparisons below are genuinely paired rather than independently noisy.
    I = RNG.integers(0, n, size=(N_BOOT, n))
    base_boot = boot_spearman([r["oof_auc"] for r in rows], prv, I)

    # Selection accuracy under resampling of the *universe* is meaningless (the universe is
    # what it is), so selection is reported as a point value and the CI is on rho only.
    res = []
    for k in crits:
        sub = [r for r in rows if k in r]
        if len(sub) < 6:
            continue
        v = np.array([r[k] for r in sub])
        pv = np.array([r["prv"] for r in sub])
        sp, pe = spearman(v, pv), pearson(v, pv)
        pick = _top2(v, [r["stem"] for r in sub], pv)
        if len(sub) == n:
            b = boot_spearman(v, prv, I)
            lo, hi = np.nanpercentile(b, [2.5, 97.5])
            beats = float(np.nanmean(b > base_boot))
        else:
            # partial-coverage criterion: bootstrap on its own reduced universe
            J = RNG.integers(0, len(sub), size=(N_BOOT, len(sub)))
            b = boot_spearman(v, pv, J)
            lo, hi = np.nanpercentile(b, [2.5, 97.5])
            beats = None
        res.append((k, sp, pe, pick[0], pick[1], len(sub), lo, hi, beats))

    res.sort(key=lambda t: (-t[3], -t[1]))
    print(f"\n{'criterion':<26} {'rho':>6} {'[95% boot CI]':>16} {'r':>6} "
          f"{'sel(best-of-2)':>15} {'vs pubLB':>10} {'P>auc':>6}  pick")
    for k, sp, pe, sel, who, m, lo, hi, beats in res:
        ci = f"[{lo:+.2f},{hi:+.2f}]" if lo is not None else f"(n={m})"
        bt = "self" if k == "oof_auc" else (f"{beats:.3f}" if beats is not None else "  -  ")
        flag = " *DQ*" if k.startswith("DQ_") else ""
        print(f"{k:<26} {sp:+.3f} {ci:>16} {pe:+.3f} {sel:.9f} "
              f"{sel-pub_pick[0]:+.6f} {bt:>6}  {who}{flag}")
    print("\n  rho  = Spearman vs TRUE private composite; CI = 20k-resample bootstrap.")
    print("  P>auc = paired-bootstrap P[rho_criterion > rho_oof_auc] on shared resamples.")
    print("  sel   = private score of the better of the top-2 this criterion would pick.")
    print("  *DQ*  = uses the revealed 0.5437 test prevalence. DIAGNOSTIC ONLY, never a recommendation.")


def _top2(crit, stems, prv):
    """Best private score among the two submissions this criterion would have selected."""
    i = np.argsort(-np.asarray(crit, dtype=float), kind="stable")[:2]
    j = i[int(np.argmax(prv[i]))]
    return float(prv[j]), f"{stems[i[0]]}+{stems[i[1]]}"


def _random_exp(prv, k):
    """E[max of k drawn without replacement] -- the do-nothing baseline."""
    s = np.sort(prv)
    n = len(s)
    from math import comb
    return float(sum(s[r] * comb(r, k - 1) for r in range(k - 1, n)) / comb(n, k))




# ----------------------------------------------------------------------------- #
# STEP 5  Stress test -- is the apparent winner real, or 28 shots at an 11-point target?
# ----------------------------------------------------------------------------- #
def stress_test(rows, crits, winner="view_auc_gap"):
    """Three ways for a lucky criterion to die: noise floor, leave-one-out, multiplicity."""
    print("\n" + "=" * 78)
    print(f"STEP 5  STRESS TEST of `{winner}`")
    print("=" * 78)

    sub = [r for r in rows if winner in r]
    prv = np.array([r["prv"] for r in sub])
    v = np.array([r[winner] for r in sub])
    stems = [r["stem"] for r in sub]
    m = len(sub)

    # (a) noise floor: is the between-bundle spread of the criterion bigger than the
    #     sampling error of the criterion itself?
    print("\n(a) NOISE FLOOR -- within-bundle bootstrap SE of the criterion vs between-bundle SD")
    ses = []
    for r in sub:
        npz = np.load(r["_path"], allow_pickle=True)
        y = npz["y"].astype(int)
        n = len(y)
        vp = npz["oof_view_p"].astype(float)
        own = npz["oof_view_owner"].astype(int)
        kk = npz["oof_view_k"].astype(int)
        a = np.empty(n); b = np.empty(n)
        a[own[kk == 0]] = vp[kk == 0]
        b[own[kk == 1]] = vp[kk == 1]
        g = []
        for _ in range(400):
            i = RNG.integers(0, n, n)
            if len(np.unique(y[i])) < 2:
                continue
            g.append(abs(roc_auc(y[i], a[i]) - roc_auc(y[i], b[i])))
        ses.append(np.std(g))
    print(f"    between-bundle SD of |AUC(v0)-AUC(v1)| : {np.std(-v):.6f}")
    print(f"    median within-bundle bootstrap SE      : {np.median(ses):.6f}")
    print(f"    ratio (signal / noise)                 : {np.std(-v)/np.median(ses):.2f}")

    # (b) leave-one-out: does one bundle carry the whole correlation?
    print("\n(b) LEAVE-ONE-OUT Spearman")
    full_rho = spearman(v, prv)
    print(f"    all {m}: rho = {full_rho:+.3f}")
    loo = []
    for i in range(m):
        keep = [j for j in range(m) if j != i]
        rr = spearman(v[keep], prv[keep])
        loo.append(rr)
        print(f"    drop {stems[i]:<22} rho = {rr:+.3f}  ({rr-full_rho:+.3f})")
    print(f"    LOO range: [{min(loo):+.3f}, {max(loo):+.3f}]")

    # (c) multiplicity: the whole panel gets 28 shots at an 11-point target.
    print("\n(c) FAMILY-WISE PERMUTATION TEST (private scores shuffled, 20k permutations)")
    common = [k for k in crits if all(k in r for r in sub)]
    V = np.array([[r[k] for r in sub] for k in common])          # (C, m)
    print(f"    panel size entered into the family : {len(common)} criteria on {m} bundles")
    Rv = rankdata(V, axis=1)
    maxrho = np.empty(N_BOOT)
    hits_oracle = np.zeros(N_BOOT, dtype=bool)
    win_i = common.index(winner)
    marg = np.empty(N_BOOT)
    for t in range(N_BOOT):
        pp = RNG.permutation(prv)
        rp = rankdata(pp)
        rho = _rowwise_corr(Rv, np.broadcast_to(rp, Rv.shape))
        maxrho[t] = np.nanmax(rho)
        marg[t] = rho[win_i]
        # did ANY criterion's top-2 contain the (permuted) best submission?
        best = int(np.argmax(pp))
        top2 = np.argsort(-V, axis=1, kind="stable")[:, :2]
        hits_oracle[t] = bool((top2 == best).any())
    p_marg = float(np.mean(marg >= full_rho))
    p_fw = float(np.mean(maxrho >= full_rho))
    print(f"    marginal   p (this criterion alone)     : {p_marg:.4f}")
    print(f"    FAMILY-WISE p (max over {len(common)} criteria)  : {p_fw:.4f}")
    print(f"    P[some criterion in the panel picks the oracle by chance] : "
          f"{hits_oracle.mean():.3f}")
    print(f"    P[one *fixed* criterion picks the oracle by chance]       : {2/m:.3f}")



# ----------------------------------------------------------------------------- #
# STEP 5b  The two questions, asked directly
# ----------------------------------------------------------------------------- #
def decompose(rows, crits):
    """(a) did anything track the private F1 column?  (b) did anything reorder our lanes?

    The post-mortem's claim is that our private deficit lived in F1 (local ranking in the
    high-precision corner), not in AUC. If so, some offline criterion should track
    `f1_prv` better than it tracks `auc_prv`. This asks that question head-on.
    """
    print("\n" + "=" * 78)
    print("STEP 5b  DECOMPOSITION -- does anything track the private F1 column specifically?")
    print("=" * 78)
    rows = [r for r in rows if r["stem"] != "smoke_test"]      # n=12, the honest universe
    pub = np.array([r["pub"] for r in rows])
    prv = np.array([r["prv"] for r in rows])
    f1p = np.array([r["f1_prv"] for r in rows])
    aucp = np.array([r["auc_prv"] for r in rows])

    print(f"\nreference: rho(public LB, private composite) = {spearman(pub, prv):+.3f}")
    print(f"           rho(public LB, private F1)        = {spearman(pub, f1p):+.3f}")
    print(f"           rho(public LB, private AUC)       = {spearman(pub, aucp):+.3f}")
    print(f"           rho(private F1, private AUC)      = {spearman(f1p, aucp):+.3f}")

    print(f"\n{'criterion':<26} {'rho(f1_prv)':>11} {'rho(auc_prv)':>12} "
          f"{'f1 - auc':>9} {'rho(pubLB)':>10}")
    out = []
    for k in crits:
        sub = [r for r in rows if k in r]
        if len(sub) < 6:
            continue
        v = np.array([r[k] for r in sub])
        i = [rows.index(r) for r in sub]
        a, b = spearman(v, f1p[i]), spearman(v, aucp[i])
        out.append((k, a, b, a - b, spearman(v, pub[i])))
    for k, a, b, d, pb in sorted(out, key=lambda t: -t[1]):
        flag = " *DQ*" if k.startswith("DQ_") else ""
        print(f"{k:<26} {a:+11.3f} {b:+12.3f} {d:+9.3f} {pb:+10.3f}{flag}")

    print("\nrho(pubLB) near +1 means the criterion is redundant with the public LB --")
    print("  it cannot reorder our lanes. Near 0 or negative means it says something new.")


if __name__ == "__main__":
    main()
