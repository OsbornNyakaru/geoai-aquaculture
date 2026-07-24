"""From-scratch ROCKET random-convolution member (rules-safe, run via --model rocket).

WHY THIS EXISTS (iter22). Every change inside the Transformer model class has been measured
closed: positional reframes, the objective term, pooling variants, instance-expansion. They all
sit at rank-correlation 0.93-0.99 with the champion, so they can only buy VARIANCE reduction, never
LEVEL (see experiments/LB_LOG.md, iter18-21). The only species of change that ever cleared the
~0.010 LB noise floor was a *different model class* (GBDT -> Transformer, +0.05). This module is the
one remaining move of that species: a genuinely different inductive bias over the SAME 12-month
sequence.

WHAT IT IS. ROCKET (Dempster et al., 2020): a large bank of RANDOM convolutional kernels (random
length, dilation, weights, bias) is applied to each channel of the time series; each kernel is
summarized by two features -- PPV (proportion of positive convolution outputs) and max -- and a
plain LINEAR classifier is fit on those features. No learned representation, no attention, no
gradient-trained kernels: the decorrelation from the Transformer comes for free from the completely
different function class. It is fast (closed-form-ish, CPU), self-contained (pure numpy + sklearn,
no new pip dependency, no external data), and therefore rules-safe in exactly the same way the
from-scratch Transformer is.

HOW IT PLUGS IN. It consumes the IDENTICAL representation the Transformer sees -- `to_inputs`
(standardized bands + missing-indicators, optionally left-aligned by relative_time) over the SAME
masking-augmented views (`_mask_views`) -- so the ONLY thing that differs from the champion is the
model. That is what makes the cross-model rank-correlation (printed by tools/arch_blend.py) a clean
go/no-go: rho < ~0.90 => decorrelated => the blend buys real private-slice variance reduction and we
bank it as a diverse finalist; rho >= ~0.94 => even a foreign model class ranks these rows the same
way, and the search is genuinely finished.

Returns (oof_prob, test_prob, fold_scores, test_per_fold) -- the same contract as run_seq_cv, so the
downstream fixed-0.5 calibration, prevalence pin, submission and preds-bundle code are untouched.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .data import Schema, WindowDist
from .seq_model import band_stats, extra_channel_stats, to_inputs, _mask_views
from .utils import combined_score, f1_at, get_logger, roc_auc

log = get_logger()


# --------------------------------------------------------------------------- #
# ROCKET transform (pure numpy)
# --------------------------------------------------------------------------- #
def _make_kernels(n_kernels: int, n_channels: int, M: int,
                  lengths, seed: int, multivariate: bool = False,
                  max_channels: int = 9):
    """Sample `n_kernels` random kernels. Each is (channels[array], W[n_sub,klen], d, bias, pad).

    UNIVARIATE (multivariate=False): each kernel convolves ONE random band. Simple and robust; with
    ~24 channels and thousands of kernels every band still gets hundreds of kernels. This is the
    iter22 member, reproduced bit-for-bit here (the RNG draw order is unchanged; the 1-D weight is
    only reshaped to [1, klen], which consumes no randomness).

    MULTIVARIATE (multivariate=True, iter23): each kernel spans a random SUBSET of bands, subset size
    ~ 2^U(0, log2(max_channels)) capped at n_channels, and the per-band dilated convolutions are
    SUMMED before pooling -- the ROCKET-multivariate recipe. This lets a single kernel encode a
    cross-band signature (e.g. low VH AND low NDVI), which the univariate form structurally cannot,
    and which is the actual pond fingerprint the Transformer captures via cross-band attention.

    Weights are mean-centered over the whole block (ROCKET), dilation is drawn on a log2 scale so the
    receptive field fits in M, bias is U(-1,1), padding is a random coin flip.
    """
    rng = np.random.default_rng(seed)
    lengths = [int(k) for k in lengths if 1 < int(k) <= M]
    max_c = max(1, min(int(max_channels), n_channels))
    kernels = []
    for _ in range(n_kernels):
        if multivariate:
            klen = int(rng.choice(lengths))
            n_sub = int(2 ** rng.uniform(0.0, np.log2(max_c))) if max_c > 1 else 1
            n_sub = min(max(n_sub, 1), n_channels)
            chans = np.sort(rng.choice(n_channels, size=n_sub, replace=False))
            max_exp = np.log2((M - 1) / (klen - 1))      # so (klen-1)*d + 1 <= M
            d = max(int(2 ** rng.uniform(0.0, max(max_exp, 0.0))), 1)
            w = rng.standard_normal((n_sub, klen)).astype(np.float32)
        else:
            # EXACT iter22 draw order (c, klen, d, w) so univariate reproduces iter22 bit-for-bit;
            # the [None,:] reshape unifies the stored shape and consumes no randomness.
            c = int(rng.integers(0, n_channels))
            klen = int(rng.choice(lengths))
            max_exp = np.log2((M - 1) / (klen - 1))
            d = max(int(2 ** rng.uniform(0.0, max(max_exp, 0.0))), 1)
            chans = np.array([c])
            w = rng.standard_normal(klen).astype(np.float32)[None, :]
        w -= w.mean()                                    # mean-centered over the whole block
        b = np.float32(rng.uniform(-1.0, 1.0))
        use_pad = bool(rng.integers(0, 2))
        kernels.append((chans, w, d, b, use_pad))
    return kernels


def _dilated_conv(series: np.ndarray, w: np.ndarray, d: int, use_pad: bool) -> np.ndarray:
    """Valid/'same' dilated 1-D convolution, vectorized over rows.

    series [n, M] -> conv [n, T]. conv[t] = sum_j w[j] * s[t + j*d] over the (optionally zero-padded)
    series s. With M=12 this is a handful of cheap numpy adds per kernel.
    """
    n, M = series.shape
    klen = len(w)
    span = (klen - 1) * d
    if use_pad:
        p = span // 2
        s = np.pad(series, ((0, 0), (p, span - p)), mode="constant")
    else:
        s = series
    T = s.shape[1] - span
    if T < 1:                                            # kernel longer than (unpadded) series
        s = np.pad(series, ((0, 0), (0, span - M + 1)), mode="constant")
        T = s.shape[1] - span
    out = np.zeros((n, T), dtype=np.float32)
    for j in range(klen):
        out += w[j] * s[:, j * d: j * d + T]
    return out


def _transform(X: np.ndarray, kernels) -> np.ndarray:
    """X [n, C, M] -> features [n, 2*n_kernels] = (PPV, max) per kernel.

    Each kernel spans `chans` (1 band for univariate, a subset for multivariate); the per-band
    dilated convolutions share (klen, d, pad) so they align and are SUMMED before pooling.
    """
    n = X.shape[0]
    feats = np.empty((n, 2 * len(kernels)), dtype=np.float32)
    for ki, (chans, w, d, b, use_pad) in enumerate(kernels):
        conv = None
        for ci, ch in enumerate(chans):
            c = _dilated_conv(X[:, ch, :], w[ci], d, use_pad)
            conv = c if conv is None else conv + c
        conv = conv + b
        feats[:, 2 * ki] = (conv > 0).mean(axis=1)       # PPV
        feats[:, 2 * ki + 1] = conv.max(axis=1)          # max
    return feats


# --------------------------------------------------------------------------- #
# CV
# --------------------------------------------------------------------------- #
def run_rocket_cv(train_cube, y, test_cube, schema: Schema, wd: WindowDist,
                  cfg: dict, smoke: bool = False):
    """Masking-aware CV for the ROCKET member.

    Mirrors run_seq_cv's fold structure and augmentation exactly (same StratifiedKFold seeds, same
    _mask_views K/R views) so the two models are comparable row-for-row. Only the estimator differs:
    ROCKET transform + a standardized linear classifier per fold, mean-of-folds for the test.
    Returns (oof_prob [N], test_prob [Ntest], fold_scores, test_per_fold [n_models, Ntest]).
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    s = cfg["seq"]                                       # reuse the champion's augmentation config
    rk = cfg.get("rocket") or {}
    n_kernels = int(rk.get("n_kernels", 2000))
    lengths = rk.get("kernel_lengths", [3, 5, 7])
    multivariate = bool(rk.get("multivariate", False))
    max_channels = int(rk.get("max_channels", 9))

    n_splits = 2 if smoke else s["n_splits"]
    n_repeats = 1 if smoke else s["n_repeats"]
    K = 1 if smoke else s["K"]
    R = 1 if smoke else s["R"]
    if smoke:
        n_kernels = min(n_kernels, 200)

    mean, std = band_stats(train_cube)
    channels_cfg = s.get("channels") or {}
    rel = bool(s.get("relative_time", False))
    if rel:
        log.info("rocket relative_time ON: observed window left-aligned to t_rel=0")
    ex_mean, ex_std = (extra_channel_stats(train_cube, schema, channels_cfg)
                       if channels_cfg else (None, None))

    def _feats(cube: np.ndarray, kernels) -> np.ndarray:
        X, _pad = to_inputs(cube, mean, std, schema, channels_cfg, ex_mean, ex_std,
                            relative_time=rel)
        return _transform(np.transpose(X, (0, 2, 1)), kernels)   # [n, M, D] -> [n, D, M]

    # Probe channel count once (defaults => 2*n_bands = 24) and build the kernel bank.
    X0, _ = to_inputs(test_cube[:1], mean, std, schema, channels_cfg, ex_mean, ex_std,
                      relative_time=rel)
    n_channels = X0.shape[2]
    kernels = _make_kernels(n_kernels, n_channels, schema.n_months, lengths, cfg["seed"],
                            multivariate=multivariate, max_channels=max_channels)
    log.info("rocket: %d kernels x (PPV,max) = %d features | %d input channels/month | lengths=%s "
             "| %s", n_kernels, 2 * n_kernels, n_channels, lengths,
             f"MULTIVARIATE (band subsets, max_channels={min(max_channels, n_channels)})"
             if multivariate else "univariate (1 band/kernel)")

    Fte = _feats(test_cube, kernels)                     # test features (fixed across folds)

    n = len(y)
    oof_sum = np.zeros(n); oof_cnt = np.zeros(n)
    test_accum = np.zeros(test_cube.shape[0]); n_models = 0
    fold_scores: List[float] = []
    test_per_fold: List[np.ndarray] = []

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=cfg["seed"] + rep)
        for fold, (tr, va) in enumerate(skf.split(np.zeros(n), y)):
            tr_cube, tr_owner = _mask_views(train_cube, tr, schema, wd, cfg,
                                            K, cfg["seed"] + rep, oof=False)
            ytr = y[tr_owner]
            Ftr = _feats(tr_cube, kernels)

            scaler = StandardScaler().fit(Ftr)
            Ftr_s = scaler.transform(Ftr)
            clf = LogisticRegressionCV(
                Cs=np.logspace(-4, 2, 7), cv=3, scoring="roc_auc",
                max_iter=2000, n_jobs=-1, random_state=cfg["seed"])
            clf.fit(Ftr_s, ytr)

            va_cube, va_owner = _mask_views(train_cube, va, schema, wd, cfg,
                                            R, cfg["seed"] + rep, oof=True)
            pv = clf.predict_proba(scaler.transform(_feats(va_cube, kernels)))[:, 1]
            prob_rows = np.zeros(len(va))
            for pos, i in enumerate(va):
                prob_rows[pos] = pv[va_owner == i].mean()
            oof_sum[va] += prob_rows; oof_cnt[va] += 1

            p_fold = clf.predict_proba(scaler.transform(Fte))[:, 1]
            test_accum += p_fold
            test_per_fold.append(np.asarray(p_fold, dtype=np.float32))
            n_models += 1

            fs = combined_score(f1_at(y[va], prob_rows, 0.5), roc_auc(y[va], prob_rows))
            fold_scores.append(fs)
            log.info("rocket rep %d fold %d: combined@0.5=%.5f (train views=%d)",
                     rep, fold, fs, len(ytr))

    oof_prob = oof_sum / np.maximum(oof_cnt, 1)
    test_prob = test_accum / max(n_models, 1)
    return oof_prob, test_prob, fold_scores, np.asarray(test_per_fold, dtype=np.float32)
