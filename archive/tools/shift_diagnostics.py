#!/usr/bin/env python
"""shift_diagnostics.py -- FREE, LEGAL, offline shift diagnostics (round-17).

Two 0-submission diagnostics that aim the whole campaign at PRIVATE-LB robustness:

  MODE A -- DROPOUT (MAR vs SEASONAL). Every distributional feature we rely on (permanence, dispersion,
    occupancy fractions) assumes the 4-6 OBSERVED test months are a REPRESENTATIVE sub-sample of the
    12-month distribution. If test dropout is SEASONAL (only certain calendar months survive) rather than
    MAR, even n-unbiased estimators become biased and ALL distributional features -- permanence included --
    are compromised. Test: mask train exactly like test (canonical sample_window/apply_mask), then compare
    each feature's distribution on FULL-12 train vs WINDOWED train vs TEST via KS. If windowing closes most
    of the train->test gap, masking is ~MAR and our features (and windowed CV) are trustworthy; if a large
    gap remains after windowing, the shift is seasonal/conditional and distributional features are risky.

  MODE B -- FEATURE SCREEN (adv-AUC vs label-AUC). Pre-rank candidate per-pixel features BEFORE spending a
    submission, computed on WINDOWED train (to mimic the n=4-6 test regime). A feature is worth a slot iff
    it is informative (high label-AUC) AND transfers (low adv-AUC, i.e. does not separate train from test).
    Round-16 rule: KEEP if label-AUC >= 0.75 AND adv-AUC <= 0.56. This is the submission-free ranker that
    replaces OOF (which is anti-correlated with the LB here).

LEGALITY: uses ONLY the supplied train data (values + labels, for label-AUC) and the UNLABELLED test
FEATURES (for adv-AUC / the KS comparison). No leaderboard feedback, no external data, no threshold tuning
of a submission -- the AUCs are internal diagnostics. Fully within the rules.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_bundle, month_active_mask          # noqa: E402
from src.features import apply_mask, sample_window            # noqa: E402
from src.utils import get_logger, load_config                 # noqa: E402

log = get_logger()


# --------------------------------------------------------------------------- #
# test-like masking of the train cube (reuses the canonical functions)
# --------------------------------------------------------------------------- #
def mask_like_test(train_cube, schema, wd, cfg, seed=0):
    rng = np.random.default_rng(seed)
    n, M, B = train_cube.shape
    out = np.full_like(train_cube, np.nan)
    for i in range(n):
        start, length = sample_window(wd, cfg, rng, M)
        out[i] = apply_mask(train_cube[i], start, length, schema, wd, cfg, rng)
    return out


# --------------------------------------------------------------------------- #
# per-pixel, n-invariant features over OBSERVED months (all masking-safe forms)
# --------------------------------------------------------------------------- #
def _band(cube, schema, name):
    if name not in schema.bands:
        return None
    return cube[:, :, schema.bands.index(name)]


def _rowstat(series, fn):
    """Apply fn to each row's observed (non-NaN) values; NaN if <1 obs."""
    out = np.full(series.shape[0], np.nan, dtype=float)
    for i in range(series.shape[0]):
        v = series[i][~np.isnan(series[i])]
        if v.size >= 1:
            out[i] = fn(v)
    return out


def _lscale(v):
    """L-scale = 1/2 * Gini mean difference = 1/2 * mean_{i<j}|xi-xj|. U-stat, unbiased at all n>=2."""
    if v.size < 2:
        return np.nan
    return 0.5 * np.abs(v[:, None] - v[None, :]).sum() / (v.size * (v.size - 1))


def _ndvi(cube, schema):
    nir, red = _band(cube, schema, "nir"), _band(cube, schema, "red")
    if nir is None or red is None:
        return None
    return (nir - red) / (nir + red + 1e-6)


def _mndwi(cube, schema):
    g, s = _band(cube, schema, "green"), _band(cube, schema, "swir1")
    if g is None or s is None:
        return None
    return (g - s) / (g + s + 1e-6)


def features(cube, schema, vhmvv_c=None):
    """Return {name: [n] feature vector} for the candidate feature bank."""
    VH, VV = _band(cube, schema, "VH"), _band(cube, schema, "VV")
    feats = {}
    feats["mean_VH"] = _rowstat(VH, np.mean)
    feats["median_VH"] = _rowstat(VH, np.median)
    feats["perm_VH<-21"] = _rowstat(VH, lambda v: np.mean(v < -21.0))
    feats["IQR_VH"] = _rowstat(VH, lambda v: np.subtract(*np.percentile(v, [75, 25])))
    feats["Lscale_VH"] = _rowstat(VH, _lscale)
    feats["VHsq_mean"] = _rowstat(VH, lambda v: np.mean(v * v))     # the iter35 winner's channel
    if VV is not None:
        feats["perm_VV<-15.82"] = _rowstat(VV, lambda v: np.mean(v < -15.82))
        feats["Lscale_VV"] = _rowstat(VV, _lscale)
        d = VH - VV
        c = vhmvv_c if vhmvv_c is not None else float(np.nanmedian(d))
        feats[f"VH-VV<{c:.1f}"] = _rowstat(d, lambda v: np.mean(v < c))
        feats["_vhmvv_c"] = c
    ndvi = _ndvi(cube, schema)
    if ndvi is not None:
        feats["evergreen_NDVI>0.3"] = _rowstat(ndvi, lambda v: np.mean(v > 0.3))
    mndwi = _mndwi(cube, schema)
    if mndwi is not None:
        feats["water_MNDWI>0"] = _rowstat(mndwi, lambda v: np.mean(v > 0.0))
    if VH is not None and ndvi is not None:
        jg = np.where(np.isnan(VH) | np.isnan(ndvi), np.nan,
                      ((VH < -21.0) & (ndvi < 0.2)).astype(float))
        feats["joint[VH<-21 & NDVI<0.2]"] = _rowstat(jg, np.mean)
    return feats


def _auc(x, label):
    """ROC-AUC of scalar x against binary label, dropping NaN rows."""
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(x)
    if m.sum() < 10 or len(np.unique(label[m])) < 2:
        return np.nan
    try:
        a = roc_auc_score(label[m].astype(int), x[m])
    except Exception:
        return np.nan
    return max(a, 1.0 - a)          # direction-agnostic separability


def _ks(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size < 5 or b.size < 5:
        return np.nan
    from scipy.stats import ks_2samp
    return float(ks_2samp(a, b).statistic)


def main():
    ap = argparse.ArgumentParser(description="Free, legal, offline shift diagnostics.")
    ap.add_argument("--mode", choices=["dropout", "screen", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    b = load_bundle(cfg)
    train_cube, y = b.train_cube, np.asarray(b.y).astype(int)
    test_cube, schema, wd = b.test_cube, b.schema, b.window_dist

    tr_win = mask_like_test(train_cube, schema, wd, cfg, seed=args.seed)
    # anchor the VH-VV cut at the FULL-train median so all three views use the same threshold
    d_full = (_band(train_cube, schema, "VH") - _band(train_cube, schema, "VV"))
    vhmvv_c = float(np.nanmedian(d_full))

    f_full = features(train_cube, schema, vhmvv_c)
    f_win = features(tr_win, schema, vhmvv_c)
    f_test = features(test_cube, schema, vhmvv_c)
    names = [k for k in f_full if not k.startswith("_")]

    if args.mode in ("dropout", "both"):
        print("=" * 78)
        print("MODE A -- DROPOUT: is the 4-6 month test masking MAR, or SEASONAL/conditional?")
        print("=" * 78)
        act = month_active_mask(test_cube)                     # [N,M] observed
        freq = act.mean(axis=0)
        print("test per-calendar-month observation frequency (uniform-ish => MAR window; "
              "spiky => seasonal):")
        print("  " + "  ".join(f"m{m}:{freq[m]:.2f}" for m in range(len(freq))))
        print(f"test window-length p(L): {dict(sorted(wd.length_probs.items()))}")
        print("-" * 78)
        print(f"{'feature':<26}{'KS(full,test)':>15}{'KS(win,test)':>15}{'masking explains':>18}")
        for k in names:
            ks_ft = _ks(f_full[k], f_test[k])
            ks_wt = _ks(f_win[k], f_test[k])
            expl = "" if (np.isnan(ks_ft) or np.isnan(ks_wt) or ks_ft < 1e-9) \
                else f"{100*(1-ks_wt/ks_ft):5.0f}% of gap"
            print(f"{k:<26}{ks_ft:>15.4f}{ks_wt:>15.4f}{expl:>18}")
        print("-" * 78)
        print("READ: if KS(win,test) << KS(full,test) (masking explains most of the gap) for permanence &")
        print("  mean, the dropout is ~MAR -> distributional features + windowed CV are TRUSTWORTHY. If a")
        print("  large KS(win,test) remains, the shift is SEASONAL/conditional -> down-weight distributional")
        print("  features (permanence included) and prefer features with low KS(win,test).")

    if args.mode in ("screen", "both"):
        print("=" * 78)
        print("MODE B -- FEATURE SCREEN (on WINDOWED train, n=4-6 regime). ADD iff labelAUC>=0.75 & advAUC<=0.56")
        print("=" * 78)
        adv_label = np.concatenate([np.zeros(tr_win.shape[0]), np.ones(test_cube.shape[0])])
        rows = []
        for k in names:
            lab = _auc(f_win[k], y)                                    # informative? (windowed regime)
            adv = _auc(np.concatenate([f_win[k], f_test[k]]), adv_label)  # transfers? (train vs test)
            keep = bool(not np.isnan(lab) and not np.isnan(adv) and lab >= 0.75 and adv <= 0.56)
            rows.append((k, lab, adv, keep))
        rows.sort(key=lambda r: (0 if r[3] else 1, -(float(r[1]) if not np.isnan(r[1]) else 0.0)))
        print(f"{'feature':<26}{'label-AUC':>11}{'adv-AUC':>10}  verdict")
        for k, lab, adv, keep in rows:
            v = "KEEP" if keep else ("shift-carrier" if (not np.isnan(adv) and adv > 0.56) else "weak")
            print(f"{k:<26}{lab:>11.3f}{adv:>10.3f}  {v}")
        print("-" * 78)
        print("READ: KEEP features are the submission-worthy candidates (informative AND transfer). Rank the")
        print("  next feature-channel experiments by this list; do NOT spend a submission on a shift-carrier")
        print("  (high adv-AUC) no matter how high its label-AUC. VHsq_mean corroborates the c_repl_vhsq win.")


if __name__ == "__main__":
    main()
