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
from typing import Dict, List, Tuple

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

    drop = ch.get("drop_bands") or []
    if drop:
        # CAPACITY-REDUCING BAND DELETION (iter25). Derived from the Phase-A 2-D screen
        # (`tools/shift_audit.py`), which scores every band on two axes: A = how well that
        # band ALONE separates masked-train from test (shift), T = how well it ALONE predicts
        # the label (signal). The rule -- high A AND low T -> delete, because the channel is a
        # pure shift-carrier costing nothing to remove. A one-axis rule ("drop the most
        # adversarially-important") would delete VH/amplitude first, which we PROVED is the
        # primary signal (c_rank collapsed OOF 0.975 -> 0.86); the second axis is what
        # protects it. See gemini_loop/RESEARCH_11.md 2.2.
        #
        # MEASURED (seed 42, logistic read-out, masked+left-aligned):
        #   VV     A=0.5907  T=0.7801   <- top shift-carrier; VH dominates it on signal (0.8302)
        #   blue   A=0.5344  T=0.5963   <- barely predictive, and the most Rayleigh-scattered
        #                                  band, so the most aerosol/illumination sensitive
        # VV is independently the SAR literature's rejected channel: it is wind-sensitive, its
        # water threshold drifts 2.6 dB/yr vs VH's 2.1, and VH is preferred because its
        # backscatter histogram is cleanly bimodal (Ottinger 2017/2019; Li 2018 Dongting).
        # Two independent routes -- our own data and published physics -- name the same channel.
        #
        # HONEST CAVEAT: VV's T sits 0.0001 BELOW the median, so its quadrant assignment is a
        # knife-edge by this screen alone. The physics is what breaks the tie. Also note the
        # screen is a per-band LINEAR read-out; a band with weak marginal signal could still
        # matter through the cross-band attention the champion actually uses.
        keep = [i for i, b in enumerate(schema.bands) if b not in drop]
        dropped = [b for b in schema.bands if b in drop]
        vals = vals[:, :, keep]
        # `compact_missing` has already collapsed `miss` to 2 summary channels that are not
        # per-band, so band-indexing it would be wrong. Leave it alone in that case.
        if not ch.get("compact_missing"):
            miss = miss[:, :, keep]
        log.info("seq drop_bands: removed %s -> %d bands (%d value + %d missing channels)",
                 dropped, len(keep), len(keep), miss.shape[2])

    if ch.get("drop_dup_s1_indicator") and not ch.get("compact_missing") \
            and miss.shape[2] == len(schema.bands):
        # iter35 (round-16 research, Agents 1&2): VH and VV are co-observed every month, so their two
        # missing-indicator columns are the IDENTICAL vector 1[month observed] -- an exact collinearity
        # (redundancy R=1, zero unique information). Dropping ONE is a provably information-free capacity
        # slot. Purpose: REPLACE dead weight with a new nonlinear coordinate at CONSTANT width (25 ch),
        # testing the untested cell of the 2x2 (adding a channel HURT iter34, deleting a band HURT iter26;
        # replacement holds width at the sweet spot and dodges both).
        vv_i = schema.bands.index("VV")
        miss = np.delete(miss, vv_i, axis=2)

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
    if ch.get("cross_pol"):
        # VH - VV (dB): the cross-pol ratio. UPDATE_13 3a -- carries the ~1.3 dB rice-canopy axis
        # (our hardest confuser). Amplitude-PRESERVING and ADDED, never a substitute for level.
        VH = _cband(cube, schema, "VH"); VV = _cband(cube, schema, "VV")
        parts.append((VH - VV)[:, :, None])
    if ch.get("permanence"):
        # Per-month permanence indicators c_tau(t) = 1[VH_dB(t) < tau]. The masked mean-pool of a
        # binary channel IS the fraction-of-observed-months-below-tau -- a fixed-degree (Class-A,
        # n-invariant) statistic, and an indicator is NOT affine-spanned by the bands, so it escapes
        # the dead water-index degeneracy. Threshold on RAW dB; NaN preserved where VH is masked.
        # RESPONSE_13's "best feature idea", generalized to a multi-threshold profile. tau anchored
        # in SAR literature (land/water split ~ -21.5 dB; flooded-rice VH minima -22..-18 dB).
        VH = _cband(cube, schema, "VH")
        taus = ch.get("cdf_taus") or [-22.0, -21.0, -20.0, -19.0]
        if ch.get("permanence_soft"):
            # iter35 (round-16 research, Agent 4): capacity-NEUTRAL SHAPE refinement of the one working
            # lever. Replace the hard indicator with a FIXED-slope sigmoid c_t = sigma(s*(tau - VH_t)).
            # At the test window (n=4-6 months) the hard fraction is quantized to <=6 levels -> massive
            # rank ties AND a train/test level-set mismatch (13 vs 6 levels) even though the mean is
            # unbiased. The soft ramp uses each month's DISTANCE below tau ("permanence depth") -> the
            # optimal rank-1 log-likelihood-ratio coordinate. s is a CONSTANT (not learned) -> 0 new params.
            s = float(ch.get("soft_slope", 0.5))
            ind = np.stack(
                [np.where(np.isnan(VH), np.nan,
                          (1.0 / (1.0 + np.exp(-s * (float(t) - VH)))).astype(np.float32))
                 for t in taus], axis=2)
        else:
            ind = np.stack(
                [np.where(np.isnan(VH), np.nan, (VH < float(t)).astype(np.float32)) for t in taus],
                axis=2)
        parts.append(ind)
    # --- iter34 single-feature candidates (UPDATE_15 menu). Each is nonlinear-in-the-mean
    #     (escapes the affine blind spot), n-invariant (a per-month fraction or squared value whose
    #     masked mean-pool is an unbiased statistic at any window length), and tested ONE AT A TIME on
    #     top of the single-tau permanence champion. All gated OFF by default. ---
    if ch.get("vv_permanence"):
        # (A) VV permanence 1[VV_dB(t) < tau_vv]: a SECOND CDF coordinate on the co-pol axis, orthogonal
        #     to VH permanence. VV floor sits ~6 dB above VH; open-water/pond VV ~ -15..-17 dB.
        VV = _cband(cube, schema, "VV")
        tau = float(ch.get("vv_tau", -15.0))
        parts.append(np.where(np.isnan(VV), np.nan, (VV < tau).astype(np.float32))[:, :, None])
    if ch.get("pond_band"):
        # (B) Pond-band occupancy 1[lo <= VH < hi]: fraction of months in the MANAGED-POND regime
        #     (mixed dike+water pixel sits ABOVE open water, BELOW dry). Agent-2's top new feature.
        VH = _cband(cube, schema, "VH")
        lo = float(ch.get("pond_band_lo", -22.0)); hi = float(ch.get("pond_band_hi", -18.0))
        band = ((VH >= lo) & (VH < hi)).astype(np.float32)
        parts.append(np.where(np.isnan(VH), np.nan, band)[:, :, None])
    if ch.get("vh_sq"):
        # (C) VH^2: gives the mean-pool model access to Var_t(VH)=E[VH^2]-E[VH]^2. Rice swings 6-8 dB,
        #     ponds are stable -> temporal variance is the textbook rice killer. Squared dB, standardized.
        VH = _cband(cube, schema, "VH")
        parts.append((VH * VH)[:, :, None])
    if ch.get("dualpol_gate"):
        # (E, iter43) DUAL-POLARIZATION water indicator 1[VH<tau_s] . 1[(VH-VV)<tau_r].
        #
        # WHY THIS IS NOT A REPEAT OF THE DEAD cross_pol ARM. Our entire feature bank is a
        # function of VH alone. VH-VV was only ever tested in forms provably AFFINE in bands the
        # model already has (our SDWI is exactly -5.697415 + 0.230259*(VH+VV), verified to 3.6e-15),
        # so those arms measured WIDTH COST, not missing information. The AND-gate is the indicator
        # form -- the same nonlinearity behind our only feature win -- and has never been run.
        #
        # TRAIN-ONLY SCREEN (iter43, window-matched via _mask_views so the 4-6 month test window is
        # not confounded with the domain). Univariate AUC of the mean-pooled fraction:
        #     VH-only  1[VH<-21]        0.8012   <- the champion permanence channel
        #     ratio    1[(VH-VV)<-8]    0.7556
        #     AND gate ts=-21, tr=-8    0.8487   <- +0.0475 over the champion channel
        # Neither clause alone approaches the AND, so the gain is a genuine INTERACTION, not a
        # threshold reshuffle. The AND also SUPPRESSES the ratio's domain shift (per-channel
        # adversarial AUC 0.685 ratio-only -> 0.597 gated) because the VH clause anchors it.
        # tau_r is anchored physically: open water depolarizes weakly (VH-VV strongly negative),
        # rice/vegetation canopy volume-scatters (VH-VV near -5..-8 dB), which is our hardest confuser.
        #
        # NaN wherever EITHER polarization is masked -> the masked mean-pool is the fraction over
        # months where BOTH are observed, i.e. still a Class-A n-invariant statistic.
        VH = _cband(cube, schema, "VH"); VV = _cband(cube, schema, "VV")
        ts = float(ch.get("dualpol_vh_tau", -21.0)); tr = float(ch.get("dualpol_ratio_tau", -8.0))
        gate = ((VH < ts) & ((VH - VV) < tr)).astype(np.float32)
        bad = np.isnan(VH) | np.isnan(VV)
        parts.append(np.where(bad, np.nan, gate)[:, :, None])
    if ch.get("rice_gate"):
        # (D) SAR x optical rice-exclusion AND-gate 1[VH<tau_s] . 1[NDVI<tau_v]: ponds are low-SAR AND
        #     never green; paddies green up. The AND is nonlinear (not affine-spanned). NaN where either
        #     sensor is masked -> mean-pool over months where BOTH are observed.
        VH = _cband(cube, schema, "VH")
        NIR = _cband(cube, schema, "nir"); RED = _cband(cube, schema, "red")
        ndvi = (NIR - RED) / (NIR + RED + 1e-6)
        ts = float(ch.get("rice_vh_tau", -21.0)); tv = float(ch.get("rice_ndvi_tau", 0.3))
        gate = ((VH < ts) & (ndvi < tv)).astype(np.float32)
        bad = np.isnan(VH) | np.isnan(ndvi)
        parts.append(np.where(bad, np.nan, gate)[:, :, None])
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


def _train(model, x, pad, y, cfg, device, K: int = 1, resampler=None, u: dict | None = None):
    """Train one fold-model.

    `u` (optional) carries the TRANSDUCTIVE terms over the unlabeled test rows — see
    `run_seq_cv` for how it is built. Keys: x, pad (K_u views per test row, owner-major),
    Ku, lam_u (cross-view invariance on test rows), alpha + p_teacher (soft self-distillation),
    warmup_frac. u=None reproduces the champion bit-for-bit.
    """
    import torch
    import torch.nn as nn

    s = cfg["seq"]
    lam = float(s.get("consistency_lambda", 0.0))
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    lossf = nn.BCEWithLogitsLoss()

    # ---- SWA / SWAD: capacity-NEUTRAL weight averaging over the training tail (round-16, Agent 5).
    #      Training here uses a CONSTANT LR (no scheduler), so the tail iterates genuinely explore the
    #      loss basin; averaging their WEIGHTS lands at a flatter minimum -> a LEVEL/generalization
    #      lever that is NOT capped by the (1-rho)/N ceiling that flattened prediction seed-averaging.
    #      LayerNorm has no running buffers, so there is nothing to recompute after averaging. Averaging
    #      happens in-place at the very end; enable:false reproduces the champion BIT-FOR-BIT. ----
    _swa = s.get("swa") or {}
    swa_on = bool(_swa.get("enable"))
    swa_start_ep = int(float(_swa.get("start_frac", 0.75)) * int(s["epochs"]))
    swa_every = max(1, int(_swa.get("every", 1)))
    swa_lr = _swa.get("lr", None)
    _sw = {"params": None, "n": 0, "step": 0}

    def _swa_maybe(ep):
        if not swa_on or ep < swa_start_ep:
            return
        _sw["step"] += 1
        if (_sw["step"] - 1) % swa_every != 0:
            return
        with torch.no_grad():
            cur = [p.detach().clone() for p in model.parameters()]
            if _sw["params"] is None:
                _sw["params"] = cur
            else:
                m = _sw["n"]
                for a, c in zip(_sw["params"], cur):
                    a.add_(c.sub_(a).div_(m + 1))     # running mean of weights
            _sw["n"] += 1

    def _swa_lr(ep):
        if swa_on and swa_lr is not None and ep >= swa_start_ep:
            for pg in opt.param_groups:
                pg["lr"] = float(swa_lr)

    def _swa_finalize():
        if swa_on and _sw["n"] and _sw["params"] is not None:
            with torch.no_grad():
                for p, a in zip(model.parameters(), _sw["params"]):
                    p.copy_(a)
            log.info("SWA: averaged %d tail snapshots (start_ep=%d, every=%d)",
                     _sw["n"], swa_start_ep, swa_every)

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

        # ---- TRANSDUCTIVE terms over the 1030 UNLABELED test rows (iter41). Two independent,
        #      ZERO-PARAMETER additions to the objective — the only class of change that has ever
        #      won here (the winners were a structural reframe +0.0128 and an objective change
        #      +0.0047; every capacity ADD lost). Both terms see test FEATURES only, never labels.
        #        lam_u : the same Var_k(logit) cross-view penalty we already run on train rows,
        #                pointed at test rows. Forms no label, so it cannot suffer confirmation
        #                bias or pseudo-label prior drift.
        #        alpha : soft self-distillation against a banked teacher probability (SOFT targets,
        #                T=1, never thresholded — hard pseudo-labels would re-inject the 0.5-cut
        #                error and destroy the probability information the variance argument rests on).
        #      Both are RAMPED linearly from 0 over warmup_frac of the epochs: an un-ramped
        #      unlabeled term on an untrained network pulls it toward a constant function. ----
        u_on = bool(u) and u.get("x") is not None
        if u_on:
            uxt = torch.from_numpy(u["x"]).to(device)
            upt = torch.from_numpy(u["pad"]).to(device)
            Ku = int(u.get("Ku", 2))
            n_uown = uxt.shape[0] // Ku
            lam_u = float(u.get("lam_u", 0.0))
            alpha = float(u.get("alpha", 0.0))
            warm = max(1, int(float(u.get("warmup_frac", 0.5)) * int(s["epochs"])))
            ar_u = torch.arange(Ku)
            # sample test owners at their natural 1030/1821 ratio, so lam_u/alpha are the only knobs
            u_obs = max(1, int(round(obs * n_uown / max(n_owner, 1))))
            pT = None
            if u.get("p_teacher") is not None:
                pT = torch.from_numpy(
                    np.asarray(u["p_teacher"], dtype=np.float32)).to(device)
            log.info("seq transductive: %d test owners x K_u=%d views | lambda_u=%.3g alpha=%.3g "
                     "| ramp over %d/%d epochs | %d test owners per step",
                     n_uown, Ku, lam_u, alpha, warm, int(s["epochs"]), u_obs)

        for ep in range(s["epochs"]):
            _swa_lr(ep)
            w_u = (min(1.0, (ep + 1) / warm) if u_on else 0.0)
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
                if u_on:
                    uo = torch.randint(0, n_uown, (u_obs,), generator=g)
                    uidx = (uo.unsqueeze(1) * Ku + ar_u).reshape(-1).to(device)
                    uout = model(uxt[uidx], upt[uidx]).view(len(uo), Ku)
                    if lam_u > 0.0:
                        loss = loss + (w_u * lam_u) * (
                            (uout - uout.mean(dim=1, keepdim=True)) ** 2).mean()
                    if alpha > 0.0 and pT is not None:
                        tgt = pT[uo.to(device)].unsqueeze(1).expand(-1, Ku).reshape(-1)
                        loss = loss + (w_u * alpha) * lossf(uout.reshape(-1), tgt)
                loss.backward()
                opt.step()
                _swa_maybe(ep)
        _swa_finalize()
        return model

    if u:
        log.warning("seq transductive terms are wired only on the coupled (consistency_lambda>0, "
                    "K>1) path — the champion path. They are being IGNORED for this run.")

    # INSTANCE-EXPANSION (iter21). The independent-view path (lam=0) already trains each of the K
    # masked views as its own BCE example. `resampler`, when supplied, goes one step further: it
    # regenerates a FRESH set of K masked windows per row at the start of every epoch, so the model
    # sees ~K*epochs distinct (row, sub-window) instances instead of K fixed ones -- the purest form
    # of "multiply each row into many masked sub-windows matched to the test masking". resampler is
    # only wired up on the lam=0 path (coupling needs a fixed owner structure); resampler=None
    # reproduces the champion bit-for-bit.
    for ep in range(s["epochs"]):
        _swa_lr(ep)
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
            _swa_maybe(ep)
    _swa_finalize()
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


def _test_views(test_cube: np.ndarray, K_u: int, seed: int, min_len: int = 4):
    """K_u ON-MANIFOLD masking views of every TEST row -> (cube [n*K_u, M, B], owners).

    A test row already shows a contiguous window; measured on Test.csv, the visible S1 length
    is uniform over {4,5,6} (345/343/342 rows) and 1030/1030 rows are contiguous. The only
    augmentation that stays inside that measured support is a contiguous SUB-window of length
    l in [min_len, L]. We deliberately do NOT hole-punch (mask 1-2 interior active months):
    interior gaps occur in neither the train masker nor the test set, and that off-manifold
    augmentation is the diagnosed cause of the iter6 TTA loss (-0.0023).

    View 0 is always the full visible window (the "weak" view). Rows with L == min_len have
    exactly one legal view, so their cross-view term is identically zero -- that is correct,
    not a bug: there is no legal augmentation for them. At min_len=4 that is 345/1030 rows;
    the other 685 carry the penalty.
    """
    from .utils import rng_for

    n, M, _ = test_cube.shape
    fully = np.isnan(test_cube).all(axis=2)                 # [n, M] True = month fully masked
    out, owners = [], []
    for i in range(n):
        active = np.where(~fully[i])[0]
        L = int(active.size)
        # every legal (start, length) sub-window of the visible block
        cand = [(int(active[0]) + o, ln)
                for ln in range(min_len, L + 1) for o in range(L - ln + 1)]
        if not cand:                                        # L < min_len (does not occur in Test)
            cand = [(int(active[0]), L)] if L else [(0, 0)]
        rng = rng_for(seed, int(i), 30000)
        pick = [(int(active[0]), L)] if L else [(0, 0)]     # view 0 = the full visible window
        others = [c for c in cand if c != pick[0]]
        for k in range(1, K_u):
            pick.append(others[int(rng.integers(0, len(others)))] if others else pick[0])
        for (s0, ln) in pick:
            c = test_cube[i].copy()
            keep = np.zeros(M, dtype=bool)
            keep[s0:s0 + ln] = True
            c[~keep] = np.nan
            out.append(c)
            owners.append(i)
    return np.stack(out), np.array(owners)


def _load_teacher(path, test_ids, n_test: int) -> np.ndarray:
    """Banked teacher probabilities for test rows, realigned to the cube's row order.

    The teacher is a preds bundle written by run_pipeline.py or tools/seed_average.py
    (`p_test_raw` + `test_ids`). We NEVER trust positional order across runs: if the bundle
    carries ids we reindex by them and fail loudly on any mismatch.
    """
    from pathlib import Path

    d = np.load(Path(path), allow_pickle=True)
    p = np.asarray(d["p_test_raw"], dtype=np.float32).ravel()
    if p.size != n_test:
        raise SystemExit(f"teacher {path}: {p.size} rows, test cube has {n_test}")
    tid = d["test_ids"] if "test_ids" in d.files else None
    if tid is not None and test_ids is not None:
        pos = {str(v): k for k, v in enumerate(np.asarray(tid).ravel())}
        try:
            order = np.array([pos[str(v)] for v in np.asarray(test_ids).ravel()])
        except KeyError as e:
            raise SystemExit(f"teacher {path}: test id {e} missing from the bundle")
        p = p[order]
    elif tid is None:
        log.warning("teacher %s carries no test_ids; assuming native row order", path)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    log.info("seq distill teacher: %s | mean p=%.4f | pos-rate@0.5=%.4f",
             path, float(p.mean()), float((p >= 0.5).mean()))
    return p


# --------------------------------------------------------------------------- #
# CV
# --------------------------------------------------------------------------- #
def run_seq_cv(train_cube, y, test_cube, schema: Schema, wd: WindowDist,
               cfg: dict, smoke: bool = False, test_ids=None):
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

    # ---- TRANSDUCTIVE bundle (iter41). Built ONCE: the test cube is fixed across folds, and the
    #      views are seeded off cfg["seed"] only, so every fold-model sees the same unlabeled set
    #      (the folds partition LABELS, and there are none here to leak). Standardization uses the
    #      TRAIN-derived mean/std/ex_* -- re-fitting them on test would be a different, unmeasured
    #      intervention and would confound the arm. ----
    _tr_cfg = s.get("transduct") or {}
    _di_cfg = s.get("distill") or {}
    lam_u = float(_tr_cfg.get("lambda_u", 0.0)) if _tr_cfg.get("enable") else 0.0
    alpha = float(_di_cfg.get("alpha", 0.0)) if _di_cfg.get("enable") else 0.0
    u_bundle = None
    if (lam_u > 0.0 or alpha > 0.0) and not smoke:
        K_u = max(1, int(_tr_cfg.get("K_u", 2)))
        if lam_u > 0.0 and K_u < 2:
            raise SystemExit("seq.transduct.lambda_u>0 needs seq.transduct.K_u>=2")
        u_cube, _ = _test_views(test_cube, K_u, int(cfg["seed"]),
                                min_len=int(_tr_cfg.get("min_len", 4)))
        Xu, pad_u = to_inputs(u_cube, mean, std, schema, channels_cfg, ex_mean, ex_std,
                              relative_time=rel)
        u_bundle = {"x": Xu, "pad": pad_u, "Ku": K_u, "lam_u": lam_u, "alpha": alpha,
                    "warmup_frac": float(_tr_cfg.get("warmup_frac",
                                                     _di_cfg.get("warmup_frac", 0.5)))}
        if alpha > 0.0:
            teacher = _di_cfg.get("teacher")
            if not teacher:
                raise SystemExit("seq.distill.enable=true needs seq.distill.teacher=<preds .npz>")
            u_bundle["p_teacher"] = _load_teacher(teacher, test_ids, test_cube.shape[0])

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

    # ---- PER-VIEW OOF (iter44). `oof_prob` below averages R masked window views per held-out
    #      row, but a TEST row shows exactly ONE window. So the vector Platt is FIT on is
    #      variance-shrunk on the window axis relative to the vector it is APPLIED to, and under a
    #      literal 0.5 cut the Platt slope IS the operating point. Keeping the individual view
    #      predictions lets tools/regime_match.py rebuild the calibration set at any R offline --
    #      in particular R=1, which matches deployment exactly -- with no extra training.
    #
    #      `R` is read ONLY here, on the held-out path. It cannot reach p_test_raw, the ranking or
    #      the AUC column; it moves the 0.5 crossing and nothing else. That isolation is what makes
    #      the offline reconstruction legitimate, and it is asserted in the verification below.
    #
    #      Purely additive: no RNG is drawn here, so `oof_prob` is bit-identical to before.
    oof_view_p: List[np.ndarray] = []
    oof_view_owner: List[np.ndarray] = []
    oof_view_k: List[np.ndarray] = []       # view index within (row, repeat)
    oof_view_rep: List[np.ndarray] = []

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

            model = _train(model, Xtr, pad_tr, ytr, cfg, device, K=K, resampler=resampler,
                           u=u_bundle)

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

            # Per-view record (see the block above run_seq_cv's fold loop). The view index is
            # recovered by counting occurrences of each owner rather than assuming _mask_views'
            # emission order, so this stays correct if that loop is ever reordered.
            _seen: Dict[int, int] = {}
            _kidx = np.empty(len(va_owner), dtype=np.int16)
            for _j, _o in enumerate(va_owner):
                _kidx[_j] = _seen.get(int(_o), 0)
                _seen[int(_o)] = int(_kidx[_j]) + 1
            # Stored in pv's NATIVE dtype (float32, from torch) with no conversion. regime_match.py
            # rebuilds `oof_prob` from this and its control run must reproduce seed_average.py
            # BIT-FOR-BIT, so the rebuild has to replicate `pv[va_owner == i].mean()` exactly --
            # including the float32 accumulator. Upcasting here would silently change that
            # reduction's rounding and break the control for no gain.
            oof_view_p.append(np.asarray(pv))
            oof_view_owner.append(np.asarray(va_owner, dtype=np.int32))
            oof_view_k.append(_kidx)
            oof_view_rep.append(np.full(len(va_owner), rep, dtype=np.int16))

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

    # ---- FREE abort gate for the transductive arms (costs no submission). A variance penalty on
    #      unlabeled rows is minimized by a CONSTANT predictor, and self-training amplifies its own
    #      error; both failure modes show up first in the realized positive rate. Pos-rate direction
    #      correctly predicted the sign of every iter34 arm, so this is our most reliable free signal.
    if u_bundle is not None:
        # Diagnostics only. The ABORT GATE itself lives in run_pipeline.py, on the CALIBRATED
        # column: the [0.50, 0.62] band and the ~0.55 reference are both properties of the
        # post-Platt submitted probabilities, not of these raw pre-calibration scores.
        log.info("TRANSDUCTIVE: raw test pos-rate %.4f (train prior %.4f) | oof_auc %.4f",
                 float((test_prob >= 0.5).mean()), float(np.mean(y)), roc_auc(y, oof_prob))
        if u_bundle.get("p_teacher") is not None:
            pt = np.asarray(u_bundle["p_teacher"])
            log.info("TRANSDUCTIVE GATE: teacher pos-rate %.4f | student-vs-teacher Spearman %.4f",
                     float((pt >= 0.5).mean()),
                     float(np.corrcoef(np.argsort(np.argsort(pt)),
                                       np.argsort(np.argsort(test_prob)))[0, 1]))
    # 5th element: the per-view OOF record. Consumed by run_pipeline.py -> preds_*.npz ->
    # tools/regime_match.py. Empty dict if the fold loop never ran (cannot happen in practice).
    oof_views = {
        "oof_view_p": np.concatenate(oof_view_p) if oof_view_p else np.zeros(0, np.float32),
        "oof_view_owner": np.concatenate(oof_view_owner) if oof_view_owner else np.zeros(0, np.int32),
        "oof_view_k": np.concatenate(oof_view_k) if oof_view_k else np.zeros(0, np.int16),
        "oof_view_rep": np.concatenate(oof_view_rep) if oof_view_rep else np.zeros(0, np.int16),
    }
    return (oof_prob, test_prob, fold_scores,
            np.asarray(test_per_fold, dtype=np.float32), oof_views)
