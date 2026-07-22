"""Offline LB-predictive validators — the round-06 "Q1" unlock.

WHY THIS EXISTS
---------------
Our binding constraint is measurement, not compute or submissions. The public LB is
~309 rows (~±0.01 noise), so a real +0.005 gain is indistinguishable from noise without
spending a submission — and local OOF is not merely blind but ANTI-correlated with the LB
(our best-LB run has our lowest OOF). Both round-06 research reports independently
concluded the same thing: estimate test performance from the 1030 UNLABELED test rows,
then **prove the estimator works by retro-fitting it to the experiments whose LB we
already know** before trusting it on anything new.

That retro-fit is the whole point. This tool is self-certifying: if no estimator ranks the
known anchors correctly, we learn that for zero submissions and the noise floor stands.

ESTIMATORS (all label-free on test; computed from a run's saved preds bundle)
  ATC   Average Thresholded Confidence (Garg et al., ICLR 2022, arXiv:2201.04234).
        Pick threshold t on a source confidence score so that the fraction of OOF rows
        scoring below t equals the OOF error; predict test accuracy as the fraction of
        TEST rows scoring at or above t. Score function = negative entropy (Garg et al.
        found it at least as good as max-confidence, and it degrades more gracefully for
        a saturated/overconfident net like ours).
  DIS   Pairwise disagreement between two runs of the SAME config at different seeds
        (Generalization Disagreement Equality / agreement-on-the-line: Jiang et al. 2022,
        Baek et al. NeurIPS 2022). The rate at which the pair disagrees on the unlabeled
        test rows estimates test error. Needs >=2 bundles sharing a `variant` name.
  MARG  Test-margin mass: mean |p - 0.5| over test rows AFTER the prevalence shift. A
        cheap sanity baseline, not from the literature — included so the retro-fit has a
        naive control to beat. If MARG wins, the "sophisticated" estimators earned nothing.

USAGE
    python tools/offline_validate.py --preds-dir submissions/preds --anchors experiments/anchors.tsv

`anchors.tsv` is the ground truth: one row per historical run with its known public LB.
Estimators are scored by Spearman correlation against that column.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import target_prevalence_shift  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger()

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Score functions
# --------------------------------------------------------------------------- #
def _neg_entropy(p: np.ndarray) -> np.ndarray:
    """Negative binary entropy of each prediction; higher = more confident."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return p * np.log(p) + (1 - p) * np.log1p(-p)


def atc_estimate(oof_prob: np.ndarray, y: np.ndarray, p_test: np.ndarray,
                 threshold: float = 0.5) -> float:
    """ATC: predicted TEST accuracy from unlabeled test predictions.

    Garg et al. (ICLR 2022). Choose t such that
        #{OOF rows with score < t} / n_oof  ==  OOF error rate,
    then predict test accuracy = #{TEST rows with score >= t} / n_test.

    We select t as the empirical quantile of the OOF score distribution at the OOF error
    rate, which satisfies the defining equation exactly (up to ties).
    """
    oof_prob = np.asarray(oof_prob, dtype=float)
    y = np.asarray(y).astype(int)
    err = float((( oof_prob >= threshold).astype(int) != y).mean())
    s_oof = _neg_entropy(oof_prob)
    s_test = _neg_entropy(p_test)
    if err <= 0.0:
        # A perfect source fit gives no usable threshold; fall back to the score minimum.
        t = float(s_oof.min())
    else:
        t = float(np.quantile(s_oof, err))
    return float((s_test >= t).mean())


def disagreement_estimate(p_test_a: np.ndarray, p_test_b: np.ndarray,
                          threshold: float = 0.5) -> float:
    """Predicted TEST accuracy = 1 - disagreement rate between two seeds of one config."""
    a = (np.asarray(p_test_a, dtype=float) >= threshold).astype(int)
    b = (np.asarray(p_test_b, dtype=float) >= threshold).astype(int)
    return float(1.0 - (a != b).mean())


def margin_estimate(p_test: np.ndarray, prevalence_target: float | None = 0.649) -> float:
    """Naive control: mean |p - 0.5| after the prevalence shift (higher = more decisive)."""
    p = np.asarray(p_test, dtype=float)
    if prevalence_target is not None:
        p, _ = target_prevalence_shift(p, float(prevalence_target))
    return float(np.abs(p - 0.5).mean())


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p) - np.log1p(-p)


def atc_f1_estimate(oof_prob: np.ndarray, y: np.ndarray, p_test: np.ndarray,
                    prevalence_target: float = 0.649) -> float:
    """METRIC-ALIGNED ATC: estimate the deployed F1, not raw-0.5 accuracy.

    WHY (RESEARCH_07.md, math audit Finding 1). After the prevalence pin the submission has a
    FIXED predicted-positive count P_hat = prevalence * n_test, so

        F1 = 2*TP / (2*TP + FP + FN) = 2*TP / (P_hat + P)

    with P (the true positive count) fixed by ground truth. F1 is therefore monotone in TP among
    the top-P_hat rows -- i.e. precision@k, a functional of the test RANKING alone. ROC-AUC is
    rank-only by definition. So the leaderboard sees ONLY the ranking, and is blind to calibration.

    Plain `atc_estimate` scores confidence at the raw 0.5 cut, so it separates variants largely by
    saturation -- exactly the axis the LB cannot see, and exactly the axis our variants differ on
    (cross-view invariance moved t_star 0.500 -> 0.445 with AUC intact). That makes it confounded
    as a proxy. This estimator instead measures margin from the DEPLOYED decision boundary on both
    sides, and recombines per-class ATC into an F1 estimate.
    """
    oof_prob = np.asarray(oof_prob, dtype=float)
    y = np.asarray(y).astype(int)

    # Put OOF and test on the same footing: pin both to the deployed positive rate, then the
    # decision boundary is 0.5 on each and the margin is comparable across the two populations.
    p_oof_s, _ = target_prevalence_shift(oof_prob, float(prevalence_target))
    p_test_s, _ = target_prevalence_shift(np.asarray(p_test, dtype=float), float(prevalence_target))

    z_oof, z_test = _logit(p_oof_s), _logit(p_test_s)
    yhat_o, yhat_t = (z_oof >= 0).astype(int), (z_test >= 0).astype(int)
    s_oof, s_test = np.abs(z_oof), np.abs(z_test)

    def _class_acc(cls: int) -> float:
        """ATC restricted to the rows PREDICTED to be `cls`: estimated accuracy within that class."""
        mo, mt = yhat_o == cls, yhat_t == cls
        if mo.sum() == 0 or mt.sum() == 0:
            return float("nan")
        err = float((y[mo] != cls).mean())
        t = float(np.quantile(s_oof[mo], err)) if err > 0 else float(s_oof[mo].min())
        return float((s_test[mt] >= t).mean())

    prec, neg_acc = _class_acc(1), _class_acc(0)
    if not np.isfinite(prec) or not np.isfinite(neg_acc):
        return float("nan")
    n_pos_pred = float(yhat_t.sum())
    n_neg_pred = float((1 - yhat_t).sum())
    tp = prec * n_pos_pred
    fp = n_pos_pred - tp
    fn = (1.0 - neg_acc) * n_neg_pred
    denom = 2.0 * tp + fp + fn
    return float(2.0 * tp / denom) if denom > 0 else float("nan")


def diversity_estimate(test_per_fold: np.ndarray) -> float:
    """FOLD-ENSEMBLE DIVERSITY: 1 - mean pairwise Spearman between the 5 fold-models on test.

    WHY (RESEARCH_07.md, code audit H1 -- the leading explanation of the OOF anti-correlation).
    Our reported OOF scores a SINGLE fold-model (averaged over R=2 masked views), but the
    submitted probability is the MEAN OF 5 fold-models. Averaging is not rank-preserving, and
    after the prevalence pin the leaderboard is a pure function of the RANKING -- so fold
    averaging is a real ensembling step that the LB sees and that OOF never measures.

    That predicts the ledger's sign pattern. A change that makes each individual model better
    but the five models MORE ALIKE raises OOF while lowering the LB, because it destroys the
    ensemble diversity the submission depends on. K=4 masking augmentation is exactly such a
    change (more views per row -> each fold model converges to the same smoothed function), and
    it produced our HIGHEST OOF (0.9840) and 2nd-WORST LB. Cross-view invariance is the converse:
    it constrains each model harder (lowest OOF, 0.9753) without homogenizing the five, and it is
    our champion.

    If this quantity ranks the anchors, we have an LB-predictive local signal that needs no
    unlabeled-data theory at all -- and it rides free on runs we are already doing.
    """
    a = np.asarray(test_per_fold, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2:
        return float("nan")
    ranks = np.apply_along_axis(lambda r: np.argsort(np.argsort(r)).astype(float), 1, a)
    ranks -= ranks.mean(axis=1, keepdims=True)
    norms = np.sqrt((ranks ** 2).sum(axis=1))
    ok = norms > 0
    if ok.sum() < 2:
        return float("nan")
    ranks, norms = ranks[ok], norms[ok]
    corr = (ranks @ ranks.T) / np.outer(norms, norms)
    iu = np.triu_indices(corr.shape[0], k=1)
    return float(1.0 - corr[iu].mean())


def informative_pairs(order: List[str], est: Dict[str, Dict[str, float]],
                      min_gap: float = 0.01) -> List[tuple]:
    """Anchor pairs whose LB gap EXCEEDS the noise floor -- the only pairs worth scoring.

    Four of our seven anchors (0.8908 / 0.8917 / 0.8921 / 0.8955) lie within 0.005 of each other,
    i.e. INSIDE the +-0.01 public-LB noise band, so their measured LB ORDER is itself mostly noise.
    Demanding that an estimator reproduce that order is demanding it reproduce noise: a PERFECT
    validator can score Spearman rho = 0.643 (< 0.7) purely by disagreeing with a noise-scrambled
    cluster, and so would fail the original gate through no fault of its own.
    Concordance on the informative pairs alone has exact null p = 0.0048 (vs 0.044 for rho > 0.7).
    """
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            if abs(est[b]["lb"] - est[a]["lb"]) > min_gap:
                pairs.append((a, b))
    return pairs


# --------------------------------------------------------------------------- #
# Retro-fit scoring
# --------------------------------------------------------------------------- #
def spearman(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def load_anchors(path: Path) -> List[Dict]:
    """Read the ground-truth table: variant<TAB>lb  (# comments and blanks ignored)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        try:
            rows.append({"variant": parts[0], "lb": float(parts[1])})
        except ValueError:
            continue          # header row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", default="submissions/preds")
    ap.add_argument("--anchors", default="experiments/anchors.tsv")
    ap.add_argument("--prevalence-target", type=float, default=0.649)
    ap.add_argument("--screen", nargs="*", default=None, metavar="VARIANT",
                    help="CANDIDATE variants to score against the champion using the estimators "
                         "that CLEARED the retro-fit. Submit only when >=2 cleared estimators rank "
                         "a candidate above the champion.")
    ap.add_argument("--champion", default="seq_a_xview",
                    help="variant the candidates must beat (must be present in --anchors)")
    ap.add_argument("--min-gap", type=float, default=0.01,
                    help="anchor pairs closer than this in LB are inside the noise band and are "
                         "excluded from the gate (their measured order is itself noise)")
    args = ap.parse_args()

    anchors = load_anchors(Path(args.anchors))
    if not anchors:
        raise SystemExit(f"no anchors parsed from {args.anchors}")

    preds_dir = Path(args.preds_dir)
    est: Dict[str, Dict[str, float]] = {}
    seeds: Dict[str, List[np.ndarray]] = {}

    for a in anchors:
        v = a["variant"]
        # a variant may have several seed runs: preds_<v>.npz, preds_<v>_s7.npz, ...
        # `_s[0-9]*` not `_s*`: the loose glob also matched preds_<v>_smoke.npz, which would have
        # silently entered the two-seed disagreement estimator as if it were a seed replicate.
        files = sorted(preds_dir.glob(f"preds_{v}.npz")) + sorted(preds_dir.glob(f"preds_{v}_s[0-9]*.npz"))
        if not files:
            log.warning("MISSING bundle for variant %s (looked in %s) - skipped", v, preds_dir)
            continue
        d = np.load(files[0])
        oof, y, p_test = d["oof_prob"], d["y"], d["p_test_raw"]
        est[v] = {
            "lb": a["lb"],
            "atc": atc_estimate(oof, y, p_test),
            "atcf1": atc_f1_estimate(oof, y, p_test, args.prevalence_target),
            "marg": margin_estimate(p_test, args.prevalence_target),
        }
        if "test_per_fold" in getattr(d, "files", []):
            est[v]["div"] = diversity_estimate(d["test_per_fold"])
        seeds[v] = [np.load(f)["p_test_raw"] for f in files]
        if len(seeds[v]) >= 2:
            est[v]["dis"] = disagreement_estimate(seeds[v][0], seeds[v][1])
        else:
            log.warning("variant %s has only %d seed bundle(s) - DIS not computable for it",
                        v, len(seeds[v]))

    if not est:
        raise SystemExit("no preds bundles found - run the anchor regeneration first")

    keys = ("atc", "atcf1", "dis", "div", "marg")
    order = sorted(est, key=lambda v: est[v]["lb"])
    log.info("")
    log.info("%-22s %8s %8s %8s %8s %8s %8s",
             "variant", "LB", "ATC", "ATC-F1", "DIS", "DIV", "MARG")
    for v in order:
        e = est[v]

        def _f(k):
            return f"{e[k]:.4f}" if k in e and np.isfinite(e[k]) else "-"

        log.info("%-22s %8.4f %8s %8s %8s %8s %8s", v, e["lb"],
                 _f("atc"), _f("atcf1"), _f("dis"), _f("div"), _f("marg"))

    log.info("")
    log.info("=== RETRO-FIT: Spearman rho vs known public LB (n=%d) ===", len(order))
    verdicts: Dict[str, float] = {}
    for key in keys:
        have = [v for v in order if key in est[v] and np.isfinite(est[v][key])]
        if len(have) < 3:
            log.info("  %-6s : insufficient bundles (%d) - not scored", key.upper(), len(have))
            continue
        rho = spearman([est[v]["lb"] for v in have], [est[v][key] for v in have])
        verdicts[key] = rho
        log.info("  %-6s : rho = %+.3f   (n=%d)", key.upper(), rho, len(have))

    # ---- GATE (REVISED 2026-07-22 -- RESEARCH_07.md, math audit section 3) ----
    # The original gate was "detrend/K4 below reltime/xview AND rho > 0.7". Exact permutation
    # nulls at n=7 show rho > 0.7 alone passes by CHANCE with p = 0.044, and across 3 estimators
    # the familywise false-unlock rate is ~9%. Worse, it can REJECT a perfect validator: 4 of the
    # 7 anchors sit inside the LB noise band, and a true-skill oracle that disagrees with that
    # noise-scrambled cluster scores rho = 0.643 < 0.7. We therefore score only the anchor pairs
    # whose LB gap EXCEEDS the noise floor (exact null p = 0.0048), and report rho descriptively.
    log.info("")
    pairs = informative_pairs(order, est, min_gap=args.min_gap)
    log.info("=== GATE: concordance on the %d anchor pairs with |dLB| > %.3f ===",
             len(pairs), args.min_gap)
    if not pairs:
        log.info("  no informative pairs - cannot evaluate the gate")
    cleared = []
    for key in keys:
        scorable = [(a, b) for a, b in pairs
                    if key in est[a] and key in est[b]
                    and np.isfinite(est[a][key]) and np.isfinite(est[b][key])]
        if not scorable:
            log.info("  %-6s : not scorable (missing bundles)", key.upper())
            continue
        # est[a]["lb"] < est[b]["lb"] by construction of `order`, so concordant means est_a < est_b.
        # We accept ANTI-concordance too: an estimator that reliably ranks the anchors BACKWARDS
        # is exactly as useful as one that ranks them forwards -- you negate it. Only an estimator
        # that ranks them INCONSISTENTLY carries no signal. (Relevant to DIV, whose sign is a live
        # empirical question: H1 predicts more fold-diversity -> better LB, but the converse would
        # be an equally usable screen and an equally interesting finding.)
        n = len(scorable)
        good = sum(1 for a, b in scorable if est[a][key] < est[b][key])
        ok, sign = (good == n or good == 0), ("+" if good == n else "-")
        log.info("  %-6s : %d/%d concordant  %s%s   (rho = %s)", key.upper(), good, n,
                 "PASS" if ok else "FAIL",
                 f" [sign {sign}]" if ok else "",
                 f"{verdicts[key]:+.3f}" if key in verdicts else "n/a")
        if ok:
            cleared.append(key if sign == "+" else f"{key}(negated)")

    log.info("")
    if len(cleared) >= 1:
        log.info("CLEARED: %s", ", ".join(k.upper() for k in cleared))
        log.info("-> The noise floor is broken. Screen the gated backlog offline and submit only "
                 "when >=2 estimators rank a candidate above the champion.")
        if set(cleared) <= {"marg", "marg(negated)"}:
            log.info("-> CAVEAT: only the NAIVE control cleared. The 'sophisticated' estimators "
                     "earned nothing; treat this as a weak screen, not a validator.")
    else:
        log.info("NO ESTIMATOR CLEARED -> Q1 failed, at a cost of 0 submissions. Fall back to "
                 "Scenario B (RESEARCH_07.md section 6): build a TEMPORAL holdout instead, and "
                 "fund only structural deletions with plausible effect >= +0.013.")
    log.info("")
    log.info("Measurement protocol (RESEARCH_07.md 5b): a paired A/B vs champion is SUGGESTIVE at "
             ">=0.006 and CONFIDENT at >=0.012; unpaired comparisons need >=0.012.")

    # ------------------------------------------------------------------ #
    # SCREEN: score new candidates with the estimators that CLEARED above.
    # ------------------------------------------------------------------ #
    if not args.screen:
        return
    usable = [k for k in cleared if not k.endswith("(negated)")]
    negated = [k[:-len("(negated)")] for k in cleared if k.endswith("(negated)")]
    if not usable and not negated:
        log.info("")
        log.info("SCREEN SKIPPED: no estimator cleared the retro-fit, so none is trustworthy.")
        return
    if args.champion not in est:
        raise SystemExit(f"champion {args.champion!r} not among the scored anchors")

    log.info("")
    log.info("=== SCREEN: candidates vs champion %s (LB %.4f) ===",
             args.champion, est[args.champion]["lb"])
    log.info("Using only the estimators that cleared: %s",
             ", ".join(sorted(usable + [f"{k}(neg)" for k in negated])))

    for v in args.screen:
        files = sorted(preds_dir.glob(f"preds_{v}.npz")) + \
            sorted(preds_dir.glob(f"preds_{v}_s[0-9]*.npz"))
        if not files:
            log.warning("SCREEN: no bundle for candidate %s - skipped", v)
            continue
        d = np.load(files[0])
        oof, y, p_test = d["oof_prob"], d["y"], d["p_test_raw"]
        cand: Dict[str, float] = {
            "atc": atc_estimate(oof, y, p_test),
            "atcf1": atc_f1_estimate(oof, y, p_test, args.prevalence_target),
            "marg": margin_estimate(p_test, args.prevalence_target),
        }
        if "test_per_fold" in getattr(d, "files", []):
            cand["div"] = diversity_estimate(d["test_per_fold"])
        if len(files) >= 2:
            cand["dis"] = disagreement_estimate(
                np.load(files[0])["p_test_raw"], np.load(files[1])["p_test_raw"])

        votes, detail = 0, []
        for key in usable + negated:
            if key not in cand or key not in est[args.champion]:
                detail.append(f"{key.upper()}=n/a")
                continue
            delta = cand[key] - est[args.champion][key]
            if key in negated:
                delta = -delta
            votes += int(delta > 0)
            detail.append(f"{key.upper()}{'+' if delta > 0 else ''}{delta:.4f}")
        n_avail = sum(1 for k in usable + negated if k in cand)
        verdict = ("SUBMIT" if votes >= 2 and votes == n_avail else
                   "SUBMIT (majority)" if votes >= 2 else "HOLD")
        log.info("  %-26s %-40s votes=%d/%d  -> %s",
                 v, " ".join(detail), votes, n_avail, verdict)

    log.info("")
    log.info("Rule: submit ONLY a candidate with >=2 cleared estimators above the champion. "
             "A HOLD costs nothing; a wrong submission costs 1 of ~80 and a day of latency.")


if __name__ == "__main__":
    main()
