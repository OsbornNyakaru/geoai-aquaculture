"""Phase-A shift audit — where does the train/test separability actually LIVE?

WHY THIS EXISTS (distinct from tools/adversarial_check.py, which is about the GBDT
tabular matrix). Round-11 research (gemini_loop/RESEARCH_11.md) produced one
recommendation that three independent agents reached separately: our per-band
MISSING-INDICATOR channels may be a shifted nuisance the champion is currently
eating. The argument:

  * Structural month masking is SAFE -- we generate it ourselves from a
    distribution measured off test, so it carries ~no train-specific information.
  * Per-band S2 CLOUD gaps are REAL ATMOSPHERIC DATA FROM THE TRAIN PERIOD.
    Cloud frequency is strongly seasonal and year-to-year variable. And there is
    a specific backdoor: we deliberately deleted absolute time by left-aligning
    windows to t_rel=0, but CLOUD-GAP PATTERNS ENCODE ABSOLUTE SEASON. A model
    that learns "months 2-3 of the window have S2 missing" has partially
    recovered the month-of-year that relative-time reframing (our single biggest
    win, +0.0128) was designed to remove.

The literature is genuinely split (Perez-Lebel/Jeanselme pro-indicator;
Sisk/Groenwold/MIRRAMS anti- under TEMPORAL missingness shift), which is exactly
why this is a measurement and not an argument. It costs zero submissions.

THE DECISIVE COMPARISON. Every probe below uses MASKED train (what the model
actually sees) vs real test, so window length is already matched and cannot be
the thing being detected.

  P1  values only, no indicators   -- the irreducible signal-side shift
  P2  INDICATORS ONLY              -- the question. >=0.7 => indicators alone are
                                      a shift detector => every one of them is a
                                      shifted channel we are feeding the model.
  P3  values + indicators          -- what the champion actually consumes
  P4  S2-cloud indicators only     -- isolates the atmospheric component from the
                                      structural one
  P5  per-month S2-gap COUNT only  -- the coarse scalar alternative (Step-3 of the
                                      plan): does "how cloudy" leak as much as the
                                      full band-pattern signature?

Read: P2 >> 0.5 is the finding. P3 - P1 is what the indicators ADD on top of the
irreducible gap -- if that delta is ~0, deleting them costs nothing and the lane
closes cheaply either way.

Run:  python tools/shift_audit.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import S1_BANDS, S2_BANDS, load_bundle, month_active_mask
from src.seq_model import _mask_views
from src.utils import get_logger, load_config, set_global_seeds

log = get_logger()


def _adv_auc(X_tr: np.ndarray, X_te: np.ndarray, seed: int, tag: str) -> float:
    """OOF AUC of a train-vs-test discriminator. 1 = test."""
    X = np.vstack([X_tr, X_te]).astype(np.float64)
    y = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_te))])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.shape[1] == 0:
        return float("nan")
    # Logistic regression, not a GBDT: we want to know whether the shift is
    # LINEARLY readable off these channels. A tree could manufacture separability
    # from interactions that a linear read-out (what the model's first layer is)
    # could not. This is the conservative choice.
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, prob))
    log.info("  %-46s adv-AUC = %.4f   (%d features)", tag, auc, X.shape[1])
    return auc


def _left_align(cube: np.ndarray) -> np.ndarray:
    """Shift each row so its first observed month sits at index 0 (as the model does)."""
    out = np.full_like(cube, np.nan)
    act = month_active_mask(cube)
    for i in range(cube.shape[0]):
        idx = np.flatnonzero(act[i])
        if idx.size:
            out[i, : idx.size] = cube[i, idx]
    return out


def _blocks(cube: np.ndarray, schema, s2_idx, s1_idx):
    """Return (values, all_indicators, s2_indicators, s2_gap_count) flattened per row."""
    n, m, b = cube.shape
    miss = np.isnan(cube).astype(np.float32)                 # [n, M, B]
    vals = np.nan_to_num(cube, nan=0.0).astype(np.float32)
    # per-month count of missing S2 bands, among months that are observed at all
    active = month_active_mask(cube).astype(np.float32)      # [n, M]
    s2_gap_count = miss[:, :, s2_idx].sum(axis=2) * active   # [n, M]
    return (vals.reshape(n, -1),
            miss.reshape(n, -1),
            miss[:, :, s2_idx].reshape(n, -1),
            s2_gap_count)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config()
    set_global_seeds(args.seed)
    bundle = load_bundle(cfg, use_cache=True)
    schema, wd = bundle.schema, bundle.window_dist
    train_cube, test_cube = bundle.train_cube, bundle.test_cube

    s1_idx = [schema.bands.index(x) for x in S1_BANDS if x in schema.bands]
    s2_idx = [schema.bands.index(x) for x in S2_BANDS if x in schema.bands]

    log.info("bands=%d (S1=%d, S2=%d) months=%d | train=%d test=%d",
             len(schema.bands), len(s1_idx), len(s2_idx), schema.n_months,
             len(train_cube), len(test_cube))

    # ---- Mask train EXACTLY as the model sees it (K=1 view per row), then left-align
    # both sides. Without this we would only be re-detecting 12-vs-4 months, which is
    # a masking artifact and not a fact about the data.
    rows = np.arange(len(train_cube))
    masked_tr, _ = _mask_views(train_cube, rows, schema, wd, cfg,
                               K=1, seed=args.seed, oof=False)
    log.info("masked train views: %s (should match test's 4-6 observed months)",
             masked_tr.shape)

    a_tr = _left_align(masked_tr)
    a_te = _left_align(test_cube)

    v_tr, i_tr, s2i_tr, c_tr = _blocks(a_tr, schema, s2_idx, s1_idx)
    v_te, i_te, s2i_te, c_te = _blocks(a_te, schema, s2_idx, s1_idx)

    # sanity: observed-month counts must now agree, else the comparison is void
    lt = month_active_mask(a_tr).sum(axis=1)
    le = month_active_mask(a_te).sum(axis=1)
    log.info("observed months  train(masked): mean=%.2f  test: mean=%.2f  %s",
             lt.mean(), le.mean(),
             "OK" if abs(lt.mean() - le.mean()) < 0.5 else "*** MISMATCH - probes are void ***")

    log.info("")
    log.info("=== ADVERSARIAL PROBES (masked-train vs test, left-aligned) ===")
    p1 = _adv_auc(v_tr,  v_te,  args.seed, "P1  values only (no indicators)")
    p2 = _adv_auc(i_tr,  i_te,  args.seed, "P2  ALL missing-indicators only")
    p3 = _adv_auc(np.hstack([v_tr, i_tr]), np.hstack([v_te, i_te]), args.seed,
                  "P3  values + indicators (what champion eats)")
    p4 = _adv_auc(s2i_tr, s2i_te, args.seed, "P4  S2-cloud indicators only")
    p5 = _adv_auc(c_tr,  c_te,  args.seed, "P5  per-month S2-gap COUNT only")

    log.info("")
    log.info("=== READ ===")
    log.info("  irreducible signal-side shift (P1)        : %.4f", p1)
    log.info("  indicators ALONE (P2)                     : %.4f", p2)
    log.info("  what indicators ADD on top of P1 (P3-P1)  : %+.4f", p3 - p1)
    log.info("  atmospheric-only component (P4)           : %.4f", p4)
    log.info("  coarse scalar alternative (P5)            : %.4f", p5)
    log.info("")
    if p2 >= 0.70:
        log.info("  -> P2 >= 0.70: the indicators are a SHIFT DETECTOR on their own.")
        log.info("     Every indicator channel is a shifted channel the champion consumes.")
        log.info("     Our core law says DELETE. Next: 24->12 full deletion, 1 submission.")
        log.info("     (NOTE: this is NOT iter13's compact_missing, which was 24->14.)")
    elif p2 <= 0.60:
        log.info("  -> P2 <= 0.60: indicators are approximately shift-free. Lane CLOSED,")
        log.info("     zero submissions spent. Keep them; look elsewhere.")
    else:
        log.info("  -> P2 in (0.60, 0.70): ambiguous. Use P3-P1 as the tiebreak --")
        log.info("     if it is ~0 the indicators add nothing, so deletion is free either way.")
    log.info("")
    log.info("  Compare against the iter17 anchor: adversarial AUC on frozen Presto")
    log.info("  embeddings of the RAW pixels was 0.965-0.976. P1 near that number means")
    log.info("  the signal-side shift is irreducible and already fully present.")

    # ------------------------------------------------------------------ #
    # PHASE 2 -- the 2-D screen. The shift lives in the VALUES (P1), so ask
    # per band: how SHIFTED is it (A) vs how PREDICTIVE is it (T)?
    #
    # This is the Uber adversarial-validation drop rule (arXiv:2004.03045,
    # +3.9% AUC from cutting 309->281 features), modified per RESEARCH_11 2.2
    # so it cannot delete our primary signal:
    #
    #            | low A (stable)      | high A (shifted)
    #   high T   | KEEP  (core)        | REPAIR, don't delete (ratio / rank /
    #            |                     | window-relative transform)
    #   low  T   | drop (dead weight)  | DELETE (pure shift-carrier, free)
    #
    # A naive one-axis rule ("drop the most adversarially-important") would
    # delete amplitude first, which we PROVED is the primary signal (c_rank
    # collapsed OOF 0.975 -> 0.86). The second axis is what protects it.
    # ------------------------------------------------------------------ #
    log.info("")
    log.info("=== 2-D BAND SCREEN: shifted (A) vs predictive (T) ===")
    y_tr = bundle.y

    a_scores, t_scores = {}, {}
    for bi, band in enumerate(schema.bands):
        Xtr_b = np.nan_to_num(a_tr[:, :, bi], nan=0.0)
        Xte_b = np.nan_to_num(a_te[:, :, bi], nan=0.0)
        # A: can this band alone tell train from test?
        a_scores[band] = _adv_auc_quiet(Xtr_b, Xte_b, args.seed)
        # T: can this band alone predict the label? (train only, honest OOF)
        t_scores[band] = _target_auc(Xtr_b, y_tr, args.seed)

    a_vals = np.array([a_scores[b] for b in schema.bands])
    t_vals = np.array([t_scores[b] for b in schema.bands])
    a_med, t_med = float(np.median(a_vals)), float(np.median(t_vals))

    log.info("  %-8s %8s %8s   %s", "band", "A(shift)", "T(label)", "quadrant")
    order = np.argsort(-a_vals)
    for bi in order:
        band = schema.bands[bi]
        hi_a, hi_t = a_vals[bi] >= a_med, t_vals[bi] >= t_med
        quad = ("REPAIR (shifted+predictive)" if hi_a and hi_t else
                "DELETE (shifted, not predictive)" if hi_a and not hi_t else
                "KEEP (core)" if hi_t else "drop (dead weight)")
        log.info("  %-8s %8.4f %8.4f   %s", band, a_vals[bi], t_vals[bi], quad)

    log.info("")
    log.info("  medians: A=%.4f  T=%.4f", a_med, t_med)
    delete_set = [schema.bands[i] for i in range(len(schema.bands))
                  if a_vals[i] >= a_med and t_vals[i] < t_med]
    log.info("  -> free-deletion candidates (high A, low T): %s",
             delete_set if delete_set else "(none)")
    log.info("  -> these cost nothing to remove and are the only unambiguous drops.")
    log.info("     Everything in REPAIR must be TRANSFORMED, never deleted.")
    return 0


def _adv_auc_quiet(X_tr, X_te, seed) -> float:
    X = np.vstack([X_tr, X_te]).astype(np.float64)
    y = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_te))])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=seed))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def _target_auc(X, y, seed) -> float:
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=seed))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, np.nan_to_num(X, nan=0.0), y, cv=cv,
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


if __name__ == "__main__":
    raise SystemExit(main())
