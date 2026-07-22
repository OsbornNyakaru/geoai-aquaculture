"""Presto adapter — convert our monthly cube into Presto's input contract.

WHY THIS LANE
-------------
iter15 closed the measurement question: the screen's resolution is ~0.010-0.013 LB, established
two independent ways that agree (ATC-F1 seed sd 0.0576 -> +-0.0094 LB via the anchor fit, and a
directly measured champion seed spread of 0.0191). Only TWO effects in the entire project have
ever exceeded that floor:

    GBDT -> Transformer swap   +0.0500
    per-cell detrend           -0.0514

Both are MODEL-CLASS changes. Every architectural tweak, loss term, pooling variant, positional
reframe and regularization knob we probed sits below the floor and is unmeasurable in principle
with our submission budget. So the only fundable architectural direction left is another
model-class change -- and the rules permit exactly one we have not tried:

    "You may use pretrained models as long as they are openly available to everyone."

Presto (Tseng et al., arXiv:2304.14065, github.com/nasaharvest/presto, MIT) is a ~0.4M-parameter
transformer pretrained with masked-modality self-supervision on 21.5M Sentinel-1/2 PIXEL TIME
SERIES. That is our exact data shape, and its pretraining objective -- reconstructing masked
timesteps and channels -- is literally our central difficulty. Frozen, the fitted model is a
~129-parameter logistic head, LESS fitted capacity than anything we have shipped, so it does not
contradict the "added capacity hurts" law: that law was measured on capacity fitted to our shifted
1,821 rows, whereas Presto's capacity is amortized over a global pretraining corpus.

INPUT CONTRACT (verified against the repo by the round-07 research pass)
    x            (N, T, 17) float, normalized
    mask         (N, T, 17) float, 1 = MASKED/ignore, 0 = observed
    dynamic_world(N, T)     long, fill 9 = "absent" sentinel
    latlons      (N, 2)     float degrees -- MANDATORY and NOT maskable
    month        int or (N,) long -- the START month, 0-indexed

    NORMED_BANDS order:
      [VV, VH, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12,
       temperature_2m, total_precipitation, elevation, slope, NDVI]

    normalization: (x + ADD_BY) / DIVIDE_BY, then NDVI recomputed from B8/B4.
      S1 VV,VH : (x + 25) / 25      -- expects dB. src/data.py detects `db`. OK.
      S2 all   : x / 1e4            -- expects raw DN. src/features.py asserts this. OK.

TWO HAZARDS, both measured by the research pass
  1. `month` is a FIRST-ORDER lever on the output ordering (rank corr 0.46 between month=const and
     month=true). Setting month=0 for every row is precisely our relative-time reframing applied to
     Presto. It MUST be the primary variant; month=true is a separate experiment.
  2. Presto masks at TOKEN-GROUP granularity, so losing `red` alone drops the whole S2_RGB token
     for that month. Our champion carries per-band indicators and can represent partial cloud;
     Presto structurally cannot. This is a real handicap and the likeliest way the lane fails.

TRAIN/TEST SYMMETRY -- do not skip. Train rows have 12 months, test rows 4-6. Feed train through
the SAME masking-window sampler we already use, or we manufacture a domain gap ourselves.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .data import BAND_ORDER

# Our BAND_ORDER is VH-first; Presto is VV-first. This index list maps ours -> Presto slots.
# Getting this backwards silently swaps the two SAR channels, so it is asserted below.
OUR_TO_PRESTO = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

PRESTO_BANDS = ["VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
                "temperature_2m", "total_precipitation", "elevation", "slope", "NDVI"]
N_PRESTO = len(PRESTO_BANDS)

ADD_BY = np.array([25.0, 25.0] + [0.0] * 10 + [0.0, 0.0, 0.0, 0.0] + [0.0], dtype=np.float32)
DIV_BY = np.array([25.0, 25.0] + [1e4] * 10 + [35.0, 0.03, 2000.0, 50.0] + [1.0], dtype=np.float32)

_B8, _B4, _NDVI = 8, 4, 16
# Bands we cannot supply at all: ERA5 (2) + SRTM (2). Masked everywhere.
_UNSUPPLIED = [12, 13, 14, 15]


def _check_band_order() -> None:
    """Fail loudly if BAND_ORDER ever changes under us — a silent VV/VH swap would be invisible."""
    expected = ["VH", "VV", "blue", "green", "red", "re1", "re2", "re3",
                "nir", "nira", "swir1", "swir2"]
    if list(BAND_ORDER) != expected:
        raise RuntimeError(
            f"BAND_ORDER changed: {list(BAND_ORDER)}. OUR_TO_PRESTO is hand-derived from the "
            f"order {expected} and must be re-derived before this adapter can be trusted.")


def to_presto(cube: np.ndarray, sar_units: str = "db") -> Tuple[np.ndarray, np.ndarray]:
    """(N, M, 12) cube with NaN for missing -> (x, mask) in Presto's contract.

    `mask` follows Presto's convention: **1 = masked/ignore**, 0 = observed.
    """
    _check_band_order()
    if sar_units != "db":
        raise RuntimeError(
            f"sar_units={sar_units!r}; Presto's S1 normalization (x+25)/25 assumes dB. "
            f"Applying it to linear power would be meaningless.")
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3 or cube.shape[2] != len(BAND_ORDER):
        raise ValueError(f"expected (N, M, {len(BAND_ORDER)}), got {cube.shape}")

    n, t, _ = cube.shape
    raw = np.zeros((n, t, N_PRESTO), dtype=np.float32)
    mask = np.ones((n, t, N_PRESTO), dtype=np.float32)      # 1 = masked

    observed = ~np.isnan(cube)
    raw[:, :, OUR_TO_PRESTO] = np.nan_to_num(cube, nan=0.0)
    mask[:, :, OUR_TO_PRESTO] = (~observed).astype(np.float32)

    x = (raw + ADD_BY) / DIV_BY

    # NDVI is derived, not supplied; it needs BOTH B8 and B4 present.
    b8, b4 = x[:, :, _B8], x[:, :, _B4]
    den = b8 + b4
    x[:, :, _NDVI] = np.where(den == 0, 0.0, (b8 - b4) / np.where(den == 0, 1.0, den))
    mask[:, :, _NDVI] = np.maximum(mask[:, :, _B8], mask[:, :, _B4])

    # ERA5 + SRTM: we have none. Masked everywhere (a supported missing-modality condition).
    mask[:, :, _UNSUPPLIED] = 1.0
    x[:, :, _UNSUPPLIED] = 0.0

    return x.astype(np.float32), mask.astype(np.float32)


def embed(cube: np.ndarray, encoder, device, sar_units: str = "db",
          month_mode: str = "const", batch_size: int = 256) -> np.ndarray:
    """Frozen Presto embeddings for a cube -> (N, 128).

    month_mode:
      "const" -- month=0 for every row. This DELETES absolute calendar identity and keeps only
                 relative step, i.e. exactly the relative-time reframing. PRIMARY variant.
      "true"  -- the true start month per row. Reintroduces the calendar channel; a separate
                 experiment, never the default.
    """
    import torch

    x, mask = to_presto(cube, sar_units=sar_units)
    n, t, _ = x.shape
    if month_mode not in ("const", "true"):
        raise ValueError(f"month_mode must be 'const' or 'true', got {month_mode!r}")

    xt = torch.from_numpy(x)
    mt = torch.from_numpy(mask)
    dw = torch.full((n, t), 9, dtype=torch.long)            # 9 = DynamicWorld "absent"
    ll = torch.zeros(n, 2, dtype=torch.float32)             # no coordinates exist; constant
    if month_mode == "const":
        mo = torch.zeros(n, dtype=torch.long)
    else:
        first = np.argmax(mask[:, :, 0] == 0, axis=1)       # first month with VV observed
        mo = torch.from_numpy(first.astype(np.int64))

    encoder = encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, i + batch_size)
            out.append(encoder(
                xt[sl].to(device), dynamic_world=dw[sl].to(device), mask=mt[sl].to(device),
                latlons=ll[sl].to(device), month=mo[sl].to(device), eval_task=True
            ).cpu().numpy())
    return np.concatenate(out, axis=0)
