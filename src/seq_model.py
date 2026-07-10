"""From-scratch temporal Transformer over the 12-month satellite sequence.

Rules-safe alternative to a pretrained tabular model: trained ONLY on the
competition data, no external data, no AutoML. The point is the inductive bias
GBDTs lack — a `src_key_padding_mask` lets self-attention operate over *only the
observed months*, which is exactly the test-time regime (a consecutive 4-6 month
window; the rest are −9999). Sentinel-2-only cloud gaps are handled per band via
explicit missing-indicator channels.

Consumes the raw month×band cube directly (not the aggregate feature matrix), so
attention can learn temporal patterns of inundation/vegetation. Produces OOF and
test probabilities that feed the same fixed-0.5 calibration + submission code as
the GBDT path.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .data import Schema, WindowDist, month_active_mask
from .features import apply_mask, sample_window
from .utils import combined_score, f1_at, get_logger, roc_auc

log = get_logger()


# --------------------------------------------------------------------------- #
# Tensor preparation
# --------------------------------------------------------------------------- #
def band_stats(train_cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std over observed (non-NaN) values, for standardization."""
    B = train_cube.shape[2]
    mean = np.zeros(B, dtype=np.float32)
    std = np.ones(B, dtype=np.float32)
    for b in range(B):
        v = train_cube[:, :, b]
        v = v[~np.isnan(v)]
        if v.size:
            mean[b] = v.mean()
            s = v.std()
            std[b] = s if s > 1e-6 else 1.0
    return mean, std


def to_inputs(cube: np.ndarray, mean: np.ndarray, std: np.ndarray,
              schema: "Schema | None" = None, channels_cfg: dict | None = None,
              ex_mean: np.ndarray | None = None, ex_std: np.ndarray | None = None):
    """cube [n, M, B] -> (x [n, M, D], pad_mask [n, M] True=masked month).

    Base x = standardized band values (NaN->0) concatenated with per-band missing
    indicators, so the model sees both the value and whether it was observed
    (D = 2B). If ``channels_cfg`` enables transfer-oriented channels (per-series
    detrend / deltas / indices / rank) and ``ex_mean/ex_std`` (train-derived) are
    supplied, they are standardized and appended (D = 2B + C). Defaults reproduce
    the original 2B behaviour bit-for-bit.
    """
    miss = np.isnan(cube).astype(np.float32)            # [n, M, B]
    vals = (cube - mean) / std
    vals = np.where(np.isnan(vals), 0.0, vals).astype(np.float32)
    parts = [vals, miss]
    if channels_cfg and ex_mean is not None:
        ex = _raw_extra_channels(cube, schema, channels_cfg)
        if ex is not None:
            ex = (ex - ex_mean) / ex_std
            ex = np.where(np.isnan(ex), 0.0, ex).astype(np.float32)
            parts.append(ex)
    x = np.concatenate(parts, axis=2)                   # [n, M, D]
    pad = np.isnan(cube).all(axis=2)                    # [n, M] month fully masked
    return x, pad


# --------------------------------------------------------------------------- #
# Transfer-oriented input channels (Step 3): per-series LEVEL-invariant views.
# The domain shift is dominated by absolute per-series offset (adversarial AUC
# 0.99 -> 0.94 on region-normalized indices), so removing each cell's own level
# should transfer better than global standardization. Each group is toggled by
# `seq.channels.*` (all default false). Missingness (NaN) is preserved through
# these transforms and zeroed only after standardization, exactly like the base
# `vals` channel; the base missing-indicators already flag observed vs masked.
# --------------------------------------------------------------------------- #
def _cband(cube: np.ndarray, schema, name: str) -> np.ndarray:
    """One band's [n, M] series from the cube (NaN if the band is absent)."""
    if schema is None or name not in schema.bands:
        return np.full(cube.shape[:2], np.nan, dtype=np.float32)
    return cube[:, :, schema.bands.index(name)]


def _detrend(cube: np.ndarray) -> np.ndarray:
    """Subtract each cell's own per-band temporal mean over observed months."""
    mask = ~np.isnan(cube)
    cnt = mask.sum(axis=1, keepdims=True)
    s = np.where(mask, cube, 0.0).sum(axis=1, keepdims=True)
    mu = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return cube - mu                                     # NaN preserved where masked


def _deltas(cube: np.ndarray) -> np.ndarray:
    """Month-to-month difference per band (additive-offset invariant)."""
    d = np.full_like(cube, np.nan)
    d[:, 1:, :] = cube[:, 1:, :] - cube[:, :-1, :]
    return d


def _index_channels(cube: np.ndarray, schema) -> np.ndarray:
    """Per-month normalized-difference / SAR indices -> [n, M, 5]."""
    eps = 1e-6
    G = _cband(cube, schema, "green"); NIR = _cband(cube, schema, "nir")
    RED = _cband(cube, schema, "red"); SWIR1 = _cband(cube, schema, "swir1")
    VH = _cband(cube, schema, "VH"); VV = _cband(cube, schema, "VV")
    ndwi = (G - NIR) / (G + NIR + eps)
    mndwi = (G - SWIR1) / (G + SWIR1 + eps)
    ndvi = (NIR - RED) / (NIR + RED + eps)
    vv_minus_vh = VV - VH
    if schema is not None and getattr(schema, "sar_units", "db") == "db":
        vv_lin = np.power(10.0, VV / 10.0); vh_lin = np.power(10.0, VH / 10.0)
    else:
        vv_lin, vh_lin = VV, VH
    with np.errstate(divide="ignore", invalid="ignore"):
        sdwi = np.log(10.0 * vv_lin * vh_lin + eps) - 8.0
    return np.stack([ndwi, mndwi, ndvi, vv_minus_vh, sdwi], axis=2).astype(np.float32)


def _rank_months(cube: np.ndarray) -> np.ndarray:
    """Per (row, band) within-series rank over observed months, normalized [0,1]."""
    n, M, B = cube.shape
    out = np.full_like(cube, np.nan)
    for i in range(n):
        for b in range(B):
            col = cube[i, :, b]
            obs = ~np.isnan(col)
            k = int(obs.sum())
            if k == 0:
                continue
            if k == 1:
                out[i, obs, b] = 0.5
                continue
            order = np.argsort(np.argsort(col[obs]))
            out[i, obs, b] = (order + 0.5) / k
    return out


def _raw_extra_channels(cube: np.ndarray, schema, ch: dict):
    """Concatenate enabled transfer channels -> [n, M, C] (pre-standardization)."""
    parts = []
    if ch.get("per_cell_detrend"):
        parts.append(_detrend(cube))
    if ch.get("deltas"):
        parts.append(_deltas(cube))
    if ch.get("indices"):
        parts.append(_index_channels(cube, schema))
    if ch.get("rank"):
        parts.append(_rank_months(cube))
    if not parts:
        return None
    return np.concatenate(parts, axis=2).astype(np.float32)


def extra_channel_stats(train_cube: np.ndarray, schema, ch: dict):
    """Train-derived per-channel mean/std for the enabled transfer channels."""
    ex = _raw_extra_channels(train_cube, schema, ch)
    if ex is None:
        return None, None
    C = ex.shape[2]
    mean = np.zeros(C, dtype=np.float32); std = np.ones(C, dtype=np.float32)
    for c in range(C):
        v = ex[:, :, c]; v = v[~np.isnan(v)]
        if v.size:
            mean[c] = v.mean()
            sd = v.std()
            std[c] = sd if sd > 1e-6 else 1.0
    return mean, std


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _build_model(n_months: int, in_dim: int, cfg: dict):
    import torch
    import torch.nn as nn

    s = cfg["seq"]
    d = s["d_model"]

    class PondTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_dim, d)
            self.pos = nn.Parameter(torch.zeros(1, n_months, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=s["nhead"], dim_feedforward=d * 2,
                dropout=s["dropout"], batch_first=True, activation="gelu",
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=s["layers"])
            self.head = nn.Sequential(
                nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(s["dropout"]),
                nn.Linear(d // 2, 1),
            )

        def forward(self, x, pad):
            h = self.proj(x) + self.pos
            h = self.enc(h, src_key_padding_mask=pad)     # ignore masked months
            keep = (~pad).unsqueeze(-1).float()           # [n, M, 1]
            pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
            return self.head(pooled).squeeze(-1)          # logits [n]

    return PondTransformer()


def _train(model, x, pad, y, cfg, device):
    import torch
    import torch.nn as nn

    s = cfg["seq"]
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(x).to(device)
    pt = torch.from_numpy(pad).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    n = len(y)
    bs = s["batch_size"]
    g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    for _ in range(s["epochs"]):
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(xt[idx], pt[idx])
            loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()
    return model


def _predict(model, x, pad, device):
    import torch

    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device)
        pt = torch.from_numpy(pad).to(device)
        logits = model(xt, pt)
        return torch.sigmoid(logits).cpu().numpy()


# --------------------------------------------------------------------------- #
# Masking-augmented view builders (reuse the GBDT masking recipe)
# --------------------------------------------------------------------------- #
def _mask_views(cube: np.ndarray, rows: np.ndarray, schema: Schema,
                wd: WindowDist, cfg: dict, K: int, seed: int, oof: bool):
    """Return (masked_cube [len*K, M, B], owner_row_index [len*K])."""
    from .utils import rng_for

    out, owners = [], []
    for i in rows:
        for k in range(K):
            tag = (10000 + k) if oof else k
            rng = rng_for(seed, int(i), tag)
            start, L = sample_window(wd, cfg, rng, schema.n_months)
            out.append(apply_mask(cube[i], start, L, schema, wd, cfg, rng))
            owners.append(int(i))
    return np.stack(out), np.array(owners)


# --------------------------------------------------------------------------- #
# CV
# --------------------------------------------------------------------------- #
def run_seq_cv(train_cube, y, test_cube, schema: Schema, wd: WindowDist,
               cfg: dict, smoke: bool = False):
    """Masking-aware CV for the sequence model.

    Returns (oof_prob [N], test_prob [Ntest], fold_scores).
    """
    import torch

    s = cfg["seq"]
    from sklearn.model_selection import StratifiedKFold

    device = s.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("seq model device=%s", device)

    n_splits = 2 if smoke else s["n_splits"]
    n_repeats = 1 if smoke else s["n_repeats"]
    K = 1 if smoke else s["K"]
    R = 1 if smoke else s["R"]
    epochs_backup = s["epochs"]
    if smoke:
        s["epochs"] = 3

    mean, std = band_stats(train_cube)
    channels_cfg = s.get("channels") or {}
    ex_mean, ex_std = (extra_channel_stats(train_cube, schema, channels_cfg)
                       if channels_cfg else (None, None))
    if ex_mean is not None:
        log.info("seq transfer channels enabled: %s (+%d channels)",
                 [k for k, v in channels_cfg.items() if v], ex_mean.shape[0])
    Xte, pad_te = to_inputs(test_cube, mean, std, schema, channels_cfg, ex_mean, ex_std)

    n = len(y)
    oof_sum = np.zeros(n); oof_cnt = np.zeros(n)
    test_accum = np.zeros(test_cube.shape[0]); n_models = 0
    fold_scores: List[float] = []

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=cfg["seed"] + rep)
        for fold, (tr, va) in enumerate(skf.split(np.zeros(n), y)):
            torch.manual_seed(cfg["seed"] + rep * 100 + fold)
            np.random.seed(cfg["seed"] + rep * 100 + fold)

            tr_cube, tr_owner = _mask_views(train_cube, tr, schema, wd, cfg,
                                            K, cfg["seed"] + rep, oof=False)
            ytr = y[tr_owner]
            Xtr, pad_tr = to_inputs(tr_cube, mean, std, schema, channels_cfg, ex_mean, ex_std)

            model = _build_model(schema.n_months, Xtr.shape[2], cfg)
            model = _train(model, Xtr, pad_tr, ytr, cfg, device)

            va_cube, va_owner = _mask_views(train_cube, va, schema, wd, cfg,
                                            R, cfg["seed"] + rep, oof=True)
            Xva, pad_va = to_inputs(va_cube, mean, std, schema, channels_cfg, ex_mean, ex_std)
            pv = _predict(model, Xva, pad_va, device)
            # average R views per held-out row
            prob_rows = np.zeros(len(va))
            for pos, i in enumerate(va):
                prob_rows[pos] = pv[va_owner == i].mean()
            oof_sum[va] += prob_rows; oof_cnt[va] += 1

            test_accum += _predict(model, Xte, pad_te, device); n_models += 1

            fs = combined_score(f1_at(y[va], prob_rows, 0.5), roc_auc(y[va], prob_rows))
            fold_scores.append(fs)
            log.info("seq rep %d fold %d: combined@0.5=%.5f (train views=%d)",
                     rep, fold, fs, len(ytr))

    s["epochs"] = epochs_backup
    oof_prob = oof_sum / np.maximum(oof_cnt, 1)
    test_prob = test_accum / max(n_models, 1)
    return oof_prob, test_prob, fold_scores
