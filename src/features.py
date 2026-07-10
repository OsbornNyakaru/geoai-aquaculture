"""Feature engineering + training-time masking augmentation.

Design principles (see plan):
  * Every feature is an aggregate over *active* months only, so it is
    invariant to which window the sample happens to expose. No raw per-month
    band columns (those would leak window position).
  * Train rows are fully observed (12 months); test rows expose only a
    consecutive 4/5/6-month window. We close that gap by masking each train
    row down to a test-like window BEFORE computing features
    ("window-view augmentation", the PLAsTiCC-winning strategy).
  * SAR is in dB: the polarisation feature is a difference (VV - VH), and
    SDWI is computed after converting dB -> linear power.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import (BAND_ORDER, S1_BANDS, S2_BANDS, Schema, WindowDist,
                   month_active_mask)
from .utils import rng_for

# Aggregate functions over the active-month axis (NaN-aware).
_AGGS = {
    "mean": np.nanmean,
    "median": np.nanmedian,
    "min": np.nanmin,
    "max": np.nanmax,
    "std": np.nanstd,
}


def _nan_range(x, axis):
    return np.nanmax(x, axis=axis) - np.nanmin(x, axis=axis)


_AGGS_WITH_RANGE = dict(_AGGS, range=_nan_range)


# --------------------------------------------------------------------------- #
# Masking (augmentation core)
# --------------------------------------------------------------------------- #
def sample_window(window_dist: WindowDist, cfg: dict, rng: np.random.Generator,
                  n_months: int) -> Tuple[int, int]:
    """Sample (start, length) from the measured test distribution.

    Falls back to uniform over configured lengths / feasible starts if the
    measured distribution is unavailable.
    """
    lengths = cfg["augment"]["window_lengths"]
    if cfg["augment"].get("match_test_distribution", True) and window_dist.length_probs:
        Ls = sorted(window_dist.length_probs)
        ps = np.array([window_dist.length_probs[L] for L in Ls], dtype=float)
        ps = ps / ps.sum()
        L = int(rng.choice(Ls, p=ps))
        starts_map = window_dist.start_probs.get(L, {})
        if starts_map:
            ss = sorted(starts_map)
            sp = np.array([starts_map[s] for s in ss], dtype=float)
            sp = sp / sp.sum()
            start = int(rng.choice(ss, p=sp))
            return start, L
    else:
        L = int(rng.choice(lengths))
    max_start = n_months - L
    start = int(rng.integers(0, max_start + 1))
    return start, L


def apply_mask(cube_row: np.ndarray, start: int, length: int, schema: Schema,
               window_dist: WindowDist, cfg: dict,
               rng: np.random.Generator) -> np.ndarray:
    """Return a masked copy of one row's cube [M, B].

    Months outside [start, start+length) become all-NaN. Inside the window,
    with per-month probability from the measured S2-dropout rates, the S2
    (optical) bands of a month are additionally masked (S1 stays), simulating
    cloud cover.
    """
    m, b = cube_row.shape
    out = np.full_like(cube_row, np.nan)
    out[start:start + length] = cube_row[start:start + length]

    if cfg["augment"].get("simulate_s2_dropout", True):
        s2_idx = [schema.bands.index(x) for x in S2_BANDS if x in schema.bands]
        rates = window_dist.s2_dropout_rates or cfg["augment"].get("s2_dropout_rates", {})
        for mi in range(start, start + length):
            month = schema.months[mi]
            rate = float(rates.get(month, 0.0))
            if rate > 0.0 and rng.random() < rate:
                out[mi, s2_idx] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Index computation (per active month -> [M] series with NaN where masked)
# --------------------------------------------------------------------------- #
def _band(cube_row: np.ndarray, schema: Schema, name: str) -> np.ndarray:
    """Extract one band's monthly series [M] (NaN where masked/absent)."""
    if name not in schema.bands:
        return np.full(cube_row.shape[0], np.nan, dtype=np.float32)
    return cube_row[:, schema.bands.index(name)]


def _ratio(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    return (a - b) / (a + b + eps)


def compute_index_series(cube_row: np.ndarray, schema: Schema, cfg: dict
                         ) -> Dict[str, np.ndarray]:
    """Compute all per-month index/derived series for one masked row."""
    eps = cfg["features"]["epsilon"]
    G = _band(cube_row, schema, "green")
    NIR = _band(cube_row, schema, "nir")
    SWIR1 = _band(cube_row, schema, "swir1")
    SWIR2 = _band(cube_row, schema, "swir2")
    BLUE = _band(cube_row, schema, "blue")
    RED = _band(cube_row, schema, "red")
    VH = _band(cube_row, schema, "VH")
    VV = _band(cube_row, schema, "VV")

    series: Dict[str, np.ndarray] = {}

    if cfg["features"].get("water_indices", True):
        series["ndwi"] = _ratio(G, NIR, eps)
        series["mndwi"] = _ratio(G, SWIR1, eps)
        series["ndvi"] = _ratio(NIR, RED, eps)
        series["awei_nsh"] = 4.0 * (G - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)
        series["awei_sh"] = BLUE + 2.5 * G - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2

    if cfg["features"].get("evi", False):
        # EVI assumes reflectance in [0,1]; S2 DN are scaled by 1e4.
        s = 1.0e4
        nir_r, red_r, blue_r = NIR / s, RED / s, BLUE / s
        with np.errstate(divide="ignore", invalid="ignore"):
            series["evi"] = 2.5 * (nir_r - red_r) / (nir_r + 6.0 * red_r - 7.5 * blue_r + 1.0)

    if cfg["features"].get("sar_basic", True):
        series["vh"] = VH
        series["vv"] = VV
        series["vv_minus_vh"] = VV - VH   # dB polarisation difference

    if cfg["features"].get("sdwi", True):
        # SDWI (Sentinel-1 Dual-Polarised Water Index). dB -> linear power,
        # then SDWI = ln(10 * VV_lin * VH_lin) - 8.  Robust to cloud cover
        # because it needs only S1.
        if schema.sar_units == "db":
            vv_lin = np.power(10.0, VV / 10.0)
            vh_lin = np.power(10.0, VH / 10.0)
        else:
            vv_lin, vh_lin = VV, VH
        with np.errstate(divide="ignore", invalid="ignore"):
            series["sdwi"] = np.log(10.0 * vv_lin * vh_lin + eps) - 8.0

    return series


# --------------------------------------------------------------------------- #
# Feature vector for one masked row
# --------------------------------------------------------------------------- #
def compute_features(cube_row: np.ndarray, schema: Schema, cfg: dict
                     ) -> Tuple[np.ndarray, List[str]]:
    """Aggregate a single masked cube row [M, B] into a flat feature vector."""
    feats: List[float] = []
    names: List[str] = []
    aggs = cfg["features"]["aggregates"]
    active = ~np.isnan(cube_row).all(axis=1)   # [M] active months
    n_active = int(active.sum())

    def add_series_aggs(prefix: str, s: np.ndarray) -> None:
        # A month is valid for this series if the series value is finite.
        for agg in aggs:
            fn = _AGGS_WITH_RANGE[agg]
            with np.errstate(all="ignore"):
                if np.isfinite(s).any():
                    val = float(fn(np.where(np.isfinite(s), s, np.nan), axis=0))
                    if not np.isfinite(val):
                        val = np.nan
                else:
                    val = np.nan
            feats.append(val)
            names.append(f"{prefix}_{agg}")

    # A/B/ SDWI: index + SAR series aggregates
    series = compute_index_series(cube_row, schema, cfg)
    for key in series:
        add_series_aggs(key, series[key])

    # SAR percentiles (permanence discriminator)
    if cfg["features"].get("sar_percentiles", True):
        for band in ("VH", "VV"):
            s = _band(cube_row, schema, band)
            for q in (10, 25, 50, 75, 90):
                with np.errstate(all="ignore"):
                    val = float(np.nanpercentile(s, q)) if np.isfinite(s).any() else np.nan
                feats.append(val if np.isfinite(val) else np.nan)
                names.append(f"{band.lower()}_p{q}")

    # C: raw band temporal aggregates (mean/median/min/max/std)
    if cfg["features"].get("raw_band_aggregates", True):
        for band in schema.bands:
            s = _band(cube_row, schema, band)
            for agg in ("mean", "median", "min", "max", "std"):
                fn = _AGGS[agg]
                with np.errstate(all="ignore"):
                    val = float(fn(s, axis=0)) if np.isfinite(s).any() else np.nan
                feats.append(val if np.isfinite(val) else np.nan)
                names.append(f"raw_{band}_{agg}")

    # D: window meta
    if cfg["features"].get("window_meta", True):
        idx = np.where(active)[0]
        start = int(idx[0]) if len(idx) else -1
        end = int(idx[-1]) if len(idx) else -1
        feats += [float(n_active), float(start), float(end), float((start + end) / 2.0)]
        names += ["meta_n_active", "meta_start", "meta_end", "meta_center"]

    # E: S1-present/S2-masked asymmetry
    if cfg["features"].get("asymmetry_flags", True):
        s1_idx = [schema.bands.index(x) for x in S1_BANDS if x in schema.bands]
        s2_idx = [schema.bands.index(x) for x in S2_BANDS if x in schema.bands]
        s1_present = ~np.isnan(cube_row[:, s1_idx]).all(axis=1)
        s2_masked = np.isnan(cube_row[:, s2_idx]).all(axis=1)
        gap = s1_present & s2_masked        # [M]
        n_gap = int(gap.sum())
        feats += [float(n_gap), float(n_gap / max(n_active, 1))]
        names += ["asym_gap_count", "asym_gap_frac"]
        for month in ("06", "10"):
            if month in schema.months:
                mi = schema.months.index(month)
                feats.append(float(gap[mi]))
                names.append(f"asym_gap_m{month}")

    # F: Water Inundation Frequency (WIF). Per active-S2 month, a water-presence
    # decision rule; managed ponds stay flooded across the window, seasonal
    # fields do not. Count/fraction are region-robust (per-month booleans on
    # normalized/scaled indices, not absolute reflectance).
    if cfg["features"].get("wif", False):
        s = 1.0e4
        G = _band(cube_row, schema, "green") / s
        NIR = _band(cube_row, schema, "nir") / s
        SWIR1 = _band(cube_row, schema, "swir1") / s
        SWIR2 = _band(cube_row, schema, "swir2") / s
        BLUE = _band(cube_row, schema, "blue") / s
        RED = _band(cube_row, schema, "red") / s
        with np.errstate(divide="ignore", invalid="ignore"):
            mndwi = _ratio(G, SWIR1, cfg["features"]["epsilon"])
            ndvi = _ratio(NIR, RED, cfg["features"]["epsilon"])
            evi = 2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0)
        valid = np.isfinite(mndwi) & np.isfinite(evi) & np.isfinite(ndvi)
        # Zhou IMNE rule (improved multi-index): (MNDWI > NDVI) or (MNDWI > EVI).
        # Deliberately DROPS Zou's EVI<0.1 cap, which false-negatives on eutrophic
        # aquaculture ponds whose algae blooms raise EVI/NDVI (the reason the
        # earlier EVI<0.1 WIF gave no LB lift). This keeps algae-heavy ponds.
        water = ((mndwi > ndvi) | (mndwi > evi)) & valid
        n_valid = int(valid.sum())
        wif = int(np.nansum(water))
        # longest consecutive run of water months within the window
        longest, cur = 0, 0
        for w in water:
            cur = cur + 1 if w else 0
            longest = max(longest, cur)
        feats += [float(wif), float(wif / max(n_valid, 1)), float(longest),
                  float(n_valid)]
        names += ["wif_count", "wif_frac", "wif_longest_run", "wif_n_valid_s2"]

    return np.array(feats, dtype=np.float32), names


# --------------------------------------------------------------------------- #
# Matrix builders
# --------------------------------------------------------------------------- #
def build_test_matrix(test_cube: np.ndarray, schema: Schema, cfg: dict
                      ) -> Tuple[np.ndarray, List[str]]:
    """Test rows keep their native masking: one feature vector per row."""
    rows, names = [], None
    for i in range(test_cube.shape[0]):
        f, names = compute_features(test_cube[i], schema, cfg)
        rows.append(f)
    return np.vstack(rows), names


def build_train_matrix(train_cube: np.ndarray, y: np.ndarray, schema: Schema,
                       window_dist: WindowDist, cfg: dict, K: int, seed: int,
                       row_ids: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Expand each train row into K masked views.

    Returns (X[N*K, F], y_aug[N*K], group_row[N*K], names). group_row holds
    the ORIGINAL row index so folds can keep all views of a row together.
    """
    n = train_cube.shape[0]
    if row_ids is None:
        row_ids = np.arange(n)
    X_rows, y_rows, groups = [], [], []
    names = None
    for i in range(n):
        for k in range(K):
            rng = rng_for(seed, int(row_ids[i]), k)
            start, L = sample_window(window_dist, cfg, rng, schema.n_months)
            masked = apply_mask(train_cube[i], start, L, schema, window_dist, cfg, rng)
            f, names = compute_features(masked, schema, cfg)
            X_rows.append(f)
            y_rows.append(y[i])
            groups.append(int(row_ids[i]))
    return (np.vstack(X_rows), np.array(y_rows, dtype=int),
            np.array(groups, dtype=int), names)


def build_oof_views(train_cube: np.ndarray, schema: Schema,
                    window_dist: WindowDist, cfg: dict, R: int, seed: int,
                    row_indices: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build R masked views for each held-out row (honest, test-like OOF).

    Returns (X[len*R, F], owner_row[len*R], names) where owner_row maps each
    view back to its position within row_indices for averaging.
    """
    X_rows, owners = [], []
    names = None
    for pos, i in enumerate(row_indices):
        for r in range(R):
            # Distinct seed namespace from training views (offset by 10000).
            rng = rng_for(seed, int(i), 10000 + r)
            start, L = sample_window(window_dist, cfg, rng, schema.n_months)
            masked = apply_mask(train_cube[i], start, L, schema, window_dist, cfg, rng)
            f, names = compute_features(masked, schema, cfg)
            X_rows.append(f)
            owners.append(pos)
    return np.vstack(X_rows), np.array(owners, dtype=int), names
