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

import math
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


def _left_align(cube: np.ndarray) -> np.ndarray:
    """Relative-time reframing (Step A): roll each row so its first observed month
    sits at index 0. Test windows are a consecutive 4-6 month block at a random
    calendar start; under absolute positions the learned positional embedding can
    memorize calendar-specific spectral patterns (region overfit). Left-aligning
    makes the encoder see relative step index (duration/transition dynamics) instead
    of calendar month, without adding any capacity. Padding (post-window NaN) is
    preserved, so the src_key_padding_mask and masked-mean-pool are unaffected.
    """
    n, M, _ = cube.shape
    out = np.full_like(cube, np.nan)
    fully = np.isnan(cube).all(axis=2)                  # [n, M] True=month fully masked
    for i in range(n):
        active = np.where(~fully[i])[0]
        if active.size == 0:
            continue
        s = int(active[0])
        out[i, : M - s] = cube[i, s:]                   # window -> front; tail stays NaN
    return out


def to_inputs(cube: np.ndarray, mean: np.ndarray, std: np.ndarray,
              schema: "Schema | None" = None, channels_cfg: dict | None = None,
              ex_mean: np.ndarray | None = None, ex_std: np.ndarray | None = None,
              relative_time: bool = False):
    """cube [n, M, B] -> (x [n, M, D], pad_mask [n, M] True=masked month).

    Base x = standardized band values (NaN->0) concatenated with per-band missing
    indicators, so the model sees both the value and whether it was observed
    (D = 2B). If ``channels_cfg`` enables transfer-oriented channels (per-series
    detrend / deltas / indices / rank) and ``ex_mean/ex_std`` (train-derived) are
    supplied, they are standardized and appended (D = 2B + C). If ``relative_time``
    the observed window is left-aligned to index 0 first (per-band stats are
    position-invariant, so standardization is unaffected). Defaults (relative_time
    off, no channels) reproduce the original 2B behaviour bit-for-bit.
    """
    if relative_time:
        cube = _left_align(cube)
    miss = np.isnan(cube).astype(np.float32)            # [n, M, B]
    vals = (cube - mean) / std
    vals = np.where(np.isnan(vals), 0.0, vals).astype(np.float32)

    ch = channels_cfg or {}
    if ch.get("rank_replace"):
        # AMPLITUDE-INVARIANT REPLACEMENT (not an addition). Substitute the within-series
        # normalized RANK of each band for its standardized absolute value, keeping the channel
        # count identical -- so this is a capacity-NEUTRAL structural swap, the only family that
        # has ever won here.
        # This is the first genuine test of the amplitude question. `per_cell_detrend` was
        # believed to have tested it, but it APPENDED its channels on top of `vals` (24->36) and
        # so only measured "adding 12 correlated channels hurts". See RESEARCH_07.md 5c.
        rk = _rank_months(cube)
        vals = np.where(np.isnan(rk), 0.0, rk).astype(np.float32)

    if ch.get("compact_missing"):
        # Collapse the 12 per-band missing indicators to 2: [s1_observed, s2_observed].
        # S2 bands drop out together by scene when cloudy, so the 12 indicators are near-perfectly
        # collinear -- 10 of 24 input channels are duplicates. Beyond redundancy, cloud frequency
        # is CLIMATOLOGICAL, making the block a wide handle on calendar season, i.e. exactly the
        # phase-locked axis whose deletion won +0.0128. Capacity-REDUCING (24 -> 14 channels).
        # Derived independently by two round-07 research agents. See RESEARCH_07.md 5d.
        s1_idx = [i for i, b in enumerate(schema.bands) if b in ("VH", "VV")]
        s2_idx = [i for i in range(cube.shape[2]) if i not in s1_idx]
        s1_obs = (~np.isnan(cube[:, :, s1_idx])).any(axis=2, keepdims=True)
        s2_obs = (~np.isnan(cube[:, :, s2_idx])).any(axis=2, keepdims=True)
        miss = np.concatenate([1.0 - s1_obs, 1.0 - s2_obs], axis=2).astype(np.float32)

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
    pos_mode = s.get("pos_encoding", "learned")           # learned | dnorm | none

    class PondTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_dim, d)
            self.pos_mode = pos_mode
            self.n_months = n_months
            # Always allocate the learned length-M table: "learned" uses it directly;
            # "dnorm" interpolates it at fractional indices (parameter-neutral); "none"
            # (NoPE) leaves it unused. Kept so a saved champion state_dict still loads.
            self.pos = nn.Parameter(torch.zeros(1, n_months, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=s["nhead"], dim_feedforward=d * 2,
                dropout=s["dropout"], batch_first=True, activation="gelu",
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=s["layers"])
            # Pooling over observed months. "mean" is the champion.
            #   mean_std  second moment  -- ponds are permanent -> LOW temporal dispersion
            #   mean_max  upper tail     -- ponds are "never bright"; rice/cropland ALWAYS have a
            #                               bright canopy month (VH +6-10 dB), so max separates the
            #                               main hard negative, and an order statistic carries NO
            #                               calendar reference (immune to the temporal shift).
            # Round 07 produced a genuine disagreement between two research agents on which is
            # right (drain events contaminate max; canopy events make it discriminative), so both
            # are implemented and SCREENED OFFLINE rather than chosen blind. See RESEARCH_07.md 5d.
            self.pooling = s.get("pooling", "mean")
            # width of the pooled vector in units of d: mean=1, pairwise=2, full moments=4.
            _pool_mult = {"mean": 1, "moments": 4}.get(self.pooling, 2)
            pooled_dim = _pool_mult * d
            self.head = nn.Sequential(
                nn.Linear(pooled_dim, d // 2), nn.GELU(), nn.Dropout(s["dropout"]),
                nn.Linear(d // 2, 1),
            )
            if self.pooling != "mean":
                # Make the expansion IDENTITY-PRESERVING at init: the second-moment half is zeroed
                # so it contributes nothing, and the mean half is re-initialized at the CHAMPION's
                # scale. Without the re-init, nn.Linear's default bound = 1/sqrt(fan_in) would use
                # fan_in = 2d instead of d, shrinking the mean weights by 1/sqrt(2) -- so the model
                # would NOT start equivalent to the champion and any LB delta would be confounded
                # with an init-scale change. (Verified: without this, outputs differ at init.)
                # Net effect: the new statistic must be EARNED, which is what makes this a
                # capacity-neutral probe rather than the kind of capacity ADD our design law
                # penalizes (+2048 params, ~2.9%).
                with torch.no_grad():
                    bound = 1.0 / math.sqrt(d)
                    self.head[0].weight[:, :d].uniform_(-bound, bound)
                    self.head[0].weight[:, d:].zero_()

        def _dnorm_pos(self, pad):
            """Duration-normalized positional encoding (iter7, RESPONSE_05 idea #1).

            Assumes relative_time left-alignment (observed months at relative offsets
            0..L-1). Re-index each observed position by the fractional coordinate
            p = offset/(L-1) in [0,1] so every window length L shares ONE frame, then read
            the positional vector by linearly interpolating the learned length-M table at
            table-index j = p*(M-1). Removes the residual window-LENGTH memorization channel
            with NO new parameters; identical to the learned champion when L = M.
            """
            n, M = pad.shape
            obs = (~pad).float()                          # [n, M] 1=observed month
            L = obs.sum(1, keepdim=True)                  # [n, 1] window length
            idx = torch.arange(M, device=pad.device).unsqueeze(0).to(obs.dtype)  # [1, M]
            idx = idx.expand(n, M)                        # relative offset (left-aligned)
            p = (idx / (L - 1.0).clamp(min=1.0)).clamp(0.0, 1.0)   # [n, M] frac position
            j = p * (M - 1)                               # [n, M] fractional table index
            j0 = torch.floor(j).long().clamp(0, M - 1)
            j1 = (j0 + 1).clamp(0, M - 1)
            w = (j - j0.to(j.dtype)).unsqueeze(-1)        # [n, M, 1]
            table = self.pos.squeeze(0)                   # [M, d]
            return (1.0 - w) * table[j0] + w * table[j1]  # [n, M, d]

        def forward(self, x, pad):
            if self.pos_mode == "dnorm":
                h = self.proj(x) + self._dnorm_pos(pad)
            elif self.pos_mode == "none":
                h = self.proj(x)                          # NoPE / set encoder
            else:
                h = self.proj(x) + self.pos
            h = self.enc(h, src_key_padding_mask=pad)     # ignore masked months
            keep = (~pad).unsqueeze(-1).float()           # [n, M, 1]
            n_obs = keep.sum(1).clamp(min=1.0)            # [n, 1] observed months
            mu = (h * keep).sum(1) / n_obs
            if self.pooling == "mean":
                pooled = mu
            elif self.pooling == "mean_std":
                # BIASED 1/N (not 1/(N-1)): the Bessel factor is 1.155 at N=4 vs 1.043 at N=12,
                # an ~11% multiplicative window-LENGTH artifact -- larger than biased's. And
                # sqrt has infinite gradient at 0, reachable at N=4 with dropout, so clamp.
                var = ((h - mu.unsqueeze(1)) ** 2 * keep).sum(1) / n_obs
                pooled = torch.cat([mu, (var + 1e-5).sqrt()], dim=-1)
            elif self.pooling == "mean_max":
                mx = h.masked_fill(pad.unsqueeze(-1), float("-inf")).max(dim=1).values
                # A row with zero observed months would be all -inf; keep it finite.
                mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
                pooled = torch.cat([mu, mx], dim=-1)
            elif self.pooling == "mean_min":
                # THE LOWER TAIL -- the one we never tested. Our mean_max probe took the physics
                # agent's "ponds are never bright" framing, but the actual pond-MAPPING literature
                # (Ottinger 2017 and the nation-scale S1 work) detects ponds with a LOW-order
                # statistic: the temporal MEDIAN / p10-p25 of VH, because a pond is a "permanent
                # low scatterer" and the low percentiles are robust to speckle and to which months
                # happen to be observed. Max is the wrong tail for the published detector.
                mn = h.masked_fill(pad.unsqueeze(-1), float("inf")).min(dim=1).values
                mn = torch.where(torch.isfinite(mn), mn, torch.zeros_like(mn))
                pooled = torch.cat([mu, mn], dim=-1)
            elif self.pooling == "moments":
                # FULL DISPERSION STACK: mean + std + min + max, the form BOTH round-09 researchers
                # converged on (Claude's moment pool; Gemini's PMA targets the same lossy pool by a
                # different route). "Permanence" (low std) and the low tail (min) are the pond
                # discriminators that a bare mean literally discards; the high tail (max) separates
                # the seasonal rice/cropland swing. Mask-aware throughout; sqrt/inf guarded exactly
                # as in the pairwise modes above.
                var = ((h - mu.unsqueeze(1)) ** 2 * keep).sum(1) / n_obs
                sd = (var + 1e-5).sqrt()
                mn = h.masked_fill(pad.unsqueeze(-1), float("inf")).min(dim=1).values
                mn = torch.where(torch.isfinite(mn), mn, torch.zeros_like(mn))
                mx = h.masked_fill(pad.unsqueeze(-1), float("-inf")).max(dim=1).values
                mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
                pooled = torch.cat([mu, sd, mn, mx], dim=-1)
            else:
                raise ValueError(f"unknown seq.pooling: {self.pooling!r}")
            return self.head(pooled).squeeze(-1)          # logits [n]

    return PondTransformer()


def _train(model, x, pad, y, cfg, device, K: int = 1, resampler=None):
    import torch
    import torch.nn as nn

    s = cfg["seq"]
    lam = float(s.get("consistency_lambda", 0.0))
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    lossf = nn.BCEWithLogitsLoss()

    def _to_dev(xn, padn, yn):
        return (torch.from_numpy(xn).to(device),
                torch.from_numpy(padn).to(device),
                torch.from_numpy(yn.astype(np.float32)).to(device))

    xt, pt, yt = _to_dev(x, pad, y)
    n = len(y)
    bs = s["batch_size"]
    g = torch.Generator(device="cpu").manual_seed(cfg["seed"])

    # Cross-view invariance (iter9): the K masking views of each row are contiguous blocks
    # (see _mask_views), so batch by OWNER and add a penalty on the variance of the K logits
    # of the same row: L = BCE + lam * mean_owner Var_k(logit). Teaches label-invariance to
    # WHICH window is observed. Only when lam>0 and K>1; else the original per-view path runs
    # unchanged (champion bit-for-bit).
    if lam > 0.0 and K > 1 and n % K == 0:
        n_owner = n // K
        obs = max(1, bs // K)
        ar = torch.arange(K)
        for _ in range(s["epochs"]):
            operm = torch.randperm(n_owner, generator=g)
            for i in range(0, n_owner, obs):
                oidx = operm[i:i + obs]
                vidx = (oidx.unsqueeze(1) * K + ar).reshape(-1).to(device)
                opt.zero_grad()
                out = model(xt[vidx], pt[vidx])
                loss = lossf(out, yt[vidx])
                out_g = out.view(len(oidx), K)
                cons = ((out_g - out_g.mean(dim=1, keepdim=True)) ** 2).mean()
                loss = loss + lam * cons
                loss.backward()
                opt.step()
        return model

    # INSTANCE-EXPANSION (iter21). The independent-view path (lam=0) already trains each of the K
    # masked views as its own BCE example. `resampler`, when supplied, goes one step further: it
    # regenerates a FRESH set of K masked windows per row at the start of every epoch, so the model
    # sees ~K*epochs distinct (row, sub-window) instances instead of K fixed ones -- the purest form
    # of "multiply each row into many masked sub-windows matched to the test masking". resampler is
    # only wired up on the lam=0 path (coupling needs a fixed owner structure); resampler=None
    # reproduces the champion bit-for-bit.
    for ep in range(s["epochs"]):
        if resampler is not None:
            xe, pade, ye = resampler(ep)
            xt, pt, yt = _to_dev(xe, pade, ye)
            n = len(ye)
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


def _tta_predict(model, test_cube, mean, std, schema, channels_cfg,
                 ex_mean, ex_std, rel, cfg, device, seed):
    """Inference-only MC temporal-dropout TTA (Iter6, RESPONSE_04 idea C).

    Predict the clean test view PLUS N views in each of which 1-2 randomly chosen
    ACTIVE months per row are fully masked (set NaN across all bands), then soft-vote
    (mean of sigmoid). Masking happens on the raw cube in calendar space, so it flows
    through `to_inputs` exactly like a real test window: left-align (if relative_time),
    missing-indicators, src_key_padding_mask and masked-mean-pool all handle the extra
    gaps. Capacity-NEUTRAL (no new params/dims) — it only smooths over WHICH specific
    months a short test window happens to observe, the axis the covariate shift rides on.
    OOF is deliberately left untouched (blind/diagnostic only); this changes test preds
    only. `enable:false` reproduces the champion bit-for-bit.
    """
    from .utils import rng_for

    tcfg = cfg["seq"].get("tta") or {}
    n_views = int(tcfg.get("n_views", 8))
    sizes = list(tcfg.get("mask_months", [1, 2]))

    Xc, padc = to_inputs(test_cube, mean, std, schema, channels_cfg,
                         ex_mean, ex_std, relative_time=rel)
    acc = _predict(model, Xc, padc, device)
    cnt = 1
    fully = np.isnan(test_cube).all(axis=2)             # [n, M] True=month fully masked
    n = test_cube.shape[0]
    for v in range(n_views):
        cube_v = test_cube.copy()
        for i in range(n):
            active = np.where(~fully[i])[0]
            if active.size <= 1:                        # never blank a row entirely
                continue
            rng = rng_for(seed, int(i), 20000 + v)
            m = int(rng.choice(sizes))
            m = min(m, active.size - 1)                 # always leave >=1 observed month
            if m <= 0:
                continue
            drop = rng.choice(active, size=m, replace=False)
            cube_v[i, drop, :] = np.nan
        Xv, padv = to_inputs(cube_v, mean, std, schema, channels_cfg,
                             ex_mean, ex_std, relative_time=rel)
        acc += _predict(model, Xv, padv, device)
        cnt += 1
    return acc / cnt


# --------------------------------------------------------------------------- #
# Masking-augmented view builders (reuse the GBDT masking recipe)
# --------------------------------------------------------------------------- #
def _mask_views(cube: np.ndarray, rows: np.ndarray, schema: Schema,
                wd: WindowDist, cfg: dict, K: int, seed: int, oof: bool):
    """Return (masked_cube [len*K, M, B], owner_row_index [len*K])."""
    from .utils import rng_for

    # ANTITHETIC PAIRING (K=2 only). The cross-view invariance penalty -- our champion, +0.0047 --
    # is only as strong as the DIFFERENCE between the two views. With L in {4,5,6} of 12 months and
    # iid starts, many pairs overlap heavily, so Var_k(logit) is near-vacuous for those rows and the
    # gradient carries little signal. Drawing view 2 maximally distant from view 1 makes the penalty
    # informative on every row. This is textbook antithetic variates: the MARGINAL window
    # distribution is unchanged (we resample from the same measured p(L), p(start|L)), so no new
    # train/test shift is introduced and no parameter is added -- exactly zero capacity change.
    # It also reinterprets iter10: lambda=3 amplified a NOISY signal; improving the signal itself
    # is the strictly better move. See RESEARCH_07.md 5a (R3).
    antithetic = bool(cfg["seq"].get("antithetic_views", False)) and K == 2

    out, owners = [], []
    for i in rows:
        prev_start, prev_L = None, None
        for k in range(K):
            tag = (10000 + k) if oof else k
            rng = rng_for(seed, int(i), tag)
            start, L = sample_window(wd, cfg, rng, schema.n_months)
            if antithetic and k == 1 and prev_start is not None:
                # Resample from the SAME distribution until the second window is far from the
                # first; keep the best-separated draw if the target is never met.
                need = max(1.0, prev_L / 2.0)
                best, best_gap = (start, L), abs(start - prev_start)
                for _ in range(16):
                    if best_gap >= need:
                        break
                    cand_s, cand_L = sample_window(wd, cfg, rng, schema.n_months)
                    gap = abs(cand_s - prev_start)
                    if gap > best_gap:
                        best, best_gap = (cand_s, cand_L), gap
                start, L = best
            prev_start, prev_L = start, L
            out.append(apply_mask(cube[i], start, L, schema, wd, cfg, rng))
            owners.append(int(i))
    return np.stack(out), np.array(owners)


# --------------------------------------------------------------------------- #
# CV
# --------------------------------------------------------------------------- #
def run_seq_cv(train_cube, y, test_cube, schema: Schema, wd: WindowDist,
               cfg: dict, smoke: bool = False):
    """Masking-aware CV for the sequence model.

    Returns (oof_prob [N], test_prob [Ntest], fold_scores, test_per_fold [n_models, Ntest]).
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
    rel = bool(s.get("relative_time", False))
    if rel:
        log.info("seq relative_time ON: observed window left-aligned to t_rel=0")
    pos_mode = s.get("pos_encoding", "learned")
    if pos_mode != "learned":
        log.info("seq pos_encoding=%s (%s)", pos_mode,
                 "duration-normalized fractional positions" if pos_mode == "dnorm"
                 else "NoPE / permutation-invariant set encoder")
    lam = float(s.get("consistency_lambda", 0.0))
    if lam > 0.0:
        log.info("seq cross-view invariance ON: lambda=%.3g (penalize logit variance across K views)", lam)
    tta_on = bool((s.get("tta") or {}).get("enable", False))
    if tta_on:
        _t = s["tta"]
        log.info("seq MC temporal-dropout TTA ON: n_views=%d mask_months=%s (inference-only)",
                 int(_t.get("n_views", 8)), _t.get("mask_months", [1, 2]))
    ex_mean, ex_std = (extra_channel_stats(train_cube, schema, channels_cfg)
                       if channels_cfg else (None, None))
    if ex_mean is not None:
        log.info("seq transfer channels enabled: %s (+%d channels)",
                 [k for k, v in channels_cfg.items() if v], ex_mean.shape[0])
    Xte, pad_te = to_inputs(test_cube, mean, std, schema, channels_cfg, ex_mean, ex_std,
                            relative_time=rel)
    # Log the ACTUAL per-month input width and the flags that shaped it. iter12 set a flag at the
    # wrong config path (seq.compact_missing instead of seq.channels.compact_missing), so it never
    # reached to_inputs; the run came out bit-identical to the champion and the offline screen
    # scored that silent no-op as a legitimate 0.0000 tie. The separately-computed `n_features` in
    # run_pipeline.py agreed with the wrong answer, so nothing caught it. Print ground truth.
    log.info("seq input width: %d channels/month | pooling=%s | channel flags: %s",
             Xte.shape[2], s.get("pooling", "mean"),
             {k: v for k, v in channels_cfg.items() if v} or "none")

    n = len(y)
    oof_sum = np.zeros(n); oof_cnt = np.zeros(n)
    test_accum = np.zeros(test_cube.shape[0]); n_models = 0
    fold_scores: List[float] = []
    # Per-fold test predictions, kept so we can measure ENSEMBLE DIVERSITY offline.
    # Rationale (RESEARCH_07.md, code audit H1): OOF scores a SINGLE fold-model while the
    # submission is the mean of 5 -- and fold-averaging changes the ranking, which is the only
    # thing the leaderboard sees. So a change that makes each model better but the five MORE
    # ALIKE raises OOF and lowers LB. That is the sign pattern the whole ledger shows (K=4:
    # highest OOF 0.984, 2nd-worst LB). Saving these costs nothing and lets us test it directly.
    test_per_fold: List[np.ndarray] = []

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=cfg["seed"] + rep)
        for fold, (tr, va) in enumerate(skf.split(np.zeros(n), y)):
            torch.manual_seed(cfg["seed"] + rep * 100 + fold)
            np.random.seed(cfg["seed"] + rep * 100 + fold)

            tr_cube, tr_owner = _mask_views(train_cube, tr, schema, wd, cfg,
                                            K, cfg["seed"] + rep, oof=False)
            ytr = y[tr_owner]
            Xtr, pad_tr = to_inputs(tr_cube, mean, std, schema, channels_cfg, ex_mean, ex_std,
                                    relative_time=rel)

            model = _build_model(schema.n_months, Xtr.shape[2], cfg)

            # Instance-expansion: fresh masked windows every epoch (lam=0 path only). Each epoch's
            # views are drawn from the SAME measured p(L), p(start|L), so no new train/test shift is
            # introduced -- only the NUMBER of distinct masked instances the model trains on grows.
            resampler = None
            if bool(s.get("resample_per_epoch", False)) and lam == 0.0:
                _tr, _rep = tr, rep

                def resampler(ep, _tr=_tr, _rep=_rep):
                    c, own = _mask_views(train_cube, _tr, schema, wd, cfg, K,
                                         cfg["seed"] + _rep + 7919 * (ep + 1), oof=False)
                    Xe, pe = to_inputs(c, mean, std, schema, channels_cfg, ex_mean, ex_std,
                                       relative_time=rel)
                    return Xe, pe, y[own]
                if fold == 0 and rep == 0:
                    log.info("seq instance-expansion: per-epoch view resampling ON "
                             "(K=%d fresh views/row each of %d epochs)", K, s["epochs"])

            model = _train(model, Xtr, pad_tr, ytr, cfg, device, K=K, resampler=resampler)

            va_cube, va_owner = _mask_views(train_cube, va, schema, wd, cfg,
                                            R, cfg["seed"] + rep, oof=True)
            Xva, pad_va = to_inputs(va_cube, mean, std, schema, channels_cfg, ex_mean, ex_std,
                                    relative_time=rel)
            pv = _predict(model, Xva, pad_va, device)
            # average R views per held-out row
            prob_rows = np.zeros(len(va))
            for pos, i in enumerate(va):
                prob_rows[pos] = pv[va_owner == i].mean()
            oof_sum[va] += prob_rows; oof_cnt[va] += 1

            if tta_on:
                p_fold = _tta_predict(model, test_cube, mean, std, schema,
                                      channels_cfg, ex_mean, ex_std, rel, cfg,
                                      device, seed=cfg["seed"] + rep * 100 + fold)
            else:
                p_fold = _predict(model, Xte, pad_te, device)
            test_accum += p_fold
            test_per_fold.append(np.asarray(p_fold, dtype=np.float32))
            n_models += 1

            fs = combined_score(f1_at(y[va], prob_rows, 0.5), roc_auc(y[va], prob_rows))
            fold_scores.append(fs)
            log.info("seq rep %d fold %d: combined@0.5=%.5f (train views=%d)",
                     rep, fold, fs, len(ytr))

    s["epochs"] = epochs_backup
    oof_prob = oof_sum / np.maximum(oof_cnt, 1)
    test_prob = test_accum / max(n_models, 1)
    return oof_prob, test_prob, fold_scores, np.asarray(test_per_fold, dtype=np.float32)
