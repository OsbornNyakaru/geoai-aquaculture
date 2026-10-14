"""Data loading, schema discovery, and the observation-cube representation.

The pipeline never hardcodes the 144 band-month column names. It discovers
them by regex so a schema change (renamed band, reordered columns) is caught
loudly instead of silently mis-indexed.

Core representation: a dense float cube of shape (N, n_months, n_bands) with
the -9999 sentinel converted to NaN. Bands are ordered as `BAND_ORDER`; S1
bands (VH, VV) come first so downstream code can slice optical vs radar.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import config_hash, get_logger, resolve_path, save_npz_atomic

log = get_logger()

# Canonical band order. S1 radar first, then the 10 S2 optical bands.
S1_BANDS = ["VH", "VV"]
S2_BANDS = ["blue", "green", "red", "re1", "re2", "re3", "nir", "nira", "swir1", "swir2"]
BAND_ORDER = S1_BANDS + S2_BANDS

# Column pattern: <band>_<MM>, e.g. VH_01, swir2_12
_COL_RE = re.compile(r"^([A-Za-z0-9]+)_(\d{2})$")


@dataclass
class Schema:
    """Discovered dataset schema."""

    id_col: str
    label_col: Optional[str]
    bands: List[str]
    months: List[str]           # e.g. ["01", ..., "12"]
    band_cols: Dict[Tuple[str, str], str]  # (band, month) -> column name
    sar_units: str = "unknown"  # "db" | "linear"

    @property
    def n_months(self) -> int:
        return len(self.months)

    @property
    def n_bands(self) -> int:
        return len(self.bands)


@dataclass
class WindowDist:
    """Empirically-measured test masking distribution."""

    length_probs: Dict[int, float] = field(default_factory=dict)         # p(L)
    start_probs: Dict[int, Dict[int, float]] = field(default_factory=dict)  # p(start | L)
    s2_dropout_rates: Dict[str, float] = field(default_factory=dict)     # per calendar month


# --------------------------------------------------------------------------- #
# Schema discovery
# --------------------------------------------------------------------------- #
def discover_schema(df: pd.DataFrame, cfg: dict) -> Schema:
    """Discover ID/label/band-month columns from a dataframe by regex."""
    id_col = cfg["data"]["id_col"]
    label_col = cfg["data"]["label_col"]
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found. Columns: {list(df.columns)[:5]}...")
    has_label = label_col in df.columns

    band_cols: Dict[Tuple[str, str], str] = {}
    bands_seen: List[str] = []
    months_seen: set[str] = set()
    for col in df.columns:
        m = _COL_RE.match(col)
        if not m:
            continue
        band, month = m.group(1), m.group(2)
        band_cols[(band, month)] = col
        if band not in bands_seen:
            bands_seen.append(band)
        months_seen.add(month)

    # Order bands canonically where known, appending any unexpected extras.
    ordered_bands = [b for b in BAND_ORDER if b in bands_seen]
    extras = [b for b in bands_seen if b not in BAND_ORDER]
    if extras:
        log.warning("Unexpected bands not in BAND_ORDER: %s", extras)
        ordered_bands += extras
    months = sorted(months_seen)

    expected = len(ordered_bands) * len(months)
    if len(band_cols) != expected:
        log.warning(
            "band_cols count %d != bands(%d)*months(%d)=%d — ragged schema",
            len(band_cols), len(ordered_bands), len(months), expected,
        )
    return Schema(
        id_col=id_col,
        label_col=label_col if has_label else None,
        bands=ordered_bands,
        months=months,
        band_cols=band_cols,
    )


def _detect_sar_units(cube: np.ndarray, schema: Schema, override: str = "auto") -> str:
    """Decide dB vs linear from S1 value range. dB backscatter is negative."""
    if override in ("db", "linear"):
        return override
    s1_idx = [schema.bands.index(b) for b in S1_BANDS if b in schema.bands]
    if not s1_idx:
        return "unknown"
    vals = cube[:, :, s1_idx]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return "unknown"
    # dB backscatter typically in [-40, +5]; linear power is strictly positive
    # and usually << 1. Median < 0 is a decisive dB signal.
    return "db" if np.nanmedian(vals) < 0 else "linear"


# --------------------------------------------------------------------------- #
# Loading and cube construction
# --------------------------------------------------------------------------- #
def load_raw(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read Train/Test/SampleSubmission CSVs."""
    raw = resolve_path(cfg, "raw_dir")
    train = pd.read_csv(raw / cfg["paths"]["train_csv"])
    test = pd.read_csv(raw / cfg["paths"]["test_csv"])
    sample = pd.read_csv(raw / cfg["paths"]["sample_submission_csv"])
    log.info("Loaded train=%s test=%s sample=%s", train.shape, test.shape, sample.shape)
    return train, test, sample


def dedup_train(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Drop exact-duplicate feature-rows (artifact of the merged retired test set)."""
    feat_cols = [c for c in df.columns if c not in (schema.id_col, schema.label_col)]
    n_dup = int(df.duplicated(subset=feat_cols).sum())
    if n_dup:
        # Warn if any duplicate group has inconsistent labels (should not happen).
        if schema.label_col:
            grp = df.groupby(feat_cols, sort=False)[schema.label_col].nunique()
            inconsistent = int((grp > 1).sum())
            if inconsistent:
                log.warning("%d duplicate groups have inconsistent labels", inconsistent)
        df = df.drop_duplicates(subset=feat_cols, keep="first").reset_index(drop=True)
        log.info("Dropped %d exact-duplicate train rows -> %d", n_dup, len(df))
    return df


def to_cube(df: pd.DataFrame, schema: Schema) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Wide dataframe -> (cube[N, M, B] float32 with NaN, ids, labels|None)."""
    sentinel = -9999
    n, m, b = len(df), schema.n_months, schema.n_bands
    cube = np.full((n, m, b), np.nan, dtype=np.float32)
    for bi, band in enumerate(schema.bands):
        for mi, month in enumerate(schema.months):
            col = schema.band_cols.get((band, month))
            if col is None:
                continue  # ragged: leave NaN
            vals = df[col].to_numpy(dtype=np.float32)
            cube[:, mi, bi] = vals
    # Sentinel -> NaN (handles partial-band masking: only masked bands go NaN).
    cube[cube == sentinel] = np.nan
    ids = df[schema.id_col].to_numpy()
    has_label = schema.label_col is not None and schema.label_col in df.columns
    y = df[schema.label_col].to_numpy().astype(int) if has_label else None
    return cube, ids, y


def month_active_mask(cube: np.ndarray) -> np.ndarray:
    """Bool [N, M]: a month is 'active' if ANY band is observed (not NaN)."""
    return ~np.isnan(cube).all(axis=2)


def s2_only_gap_mask(cube: np.ndarray, schema: Schema) -> np.ndarray:
    """Bool [N, M]: S1 present but ALL S2 bands masked (cloud-cover case)."""
    s1_idx = [schema.bands.index(x) for x in S1_BANDS if x in schema.bands]
    s2_idx = [schema.bands.index(x) for x in S2_BANDS if x in schema.bands]
    s1_present = ~np.isnan(cube[:, :, s1_idx]).all(axis=2)
    s2_masked = np.isnan(cube[:, :, s2_idx]).all(axis=2)
    return s1_present & s2_masked


# --------------------------------------------------------------------------- #
# Test masking distribution measurement
# --------------------------------------------------------------------------- #
def measure_window_dist(test_cube: np.ndarray, schema: Schema, cfg: dict) -> WindowDist:
    """Measure p(L), p(start|L) and per-month S2-dropout rate from test."""
    active = month_active_mask(test_cube)          # [N, M]
    n, mth = active.shape
    lengths = active.sum(axis=1)
    # window start = first active month index (0-based)
    starts = np.array([np.argmax(active[i]) if lengths[i] > 0 else -1 for i in range(n)])

    length_probs: Dict[int, float] = {}
    start_probs: Dict[int, Dict[int, float]] = {}
    for L in np.unique(lengths):
        L = int(L)
        sel = lengths == L
        length_probs[L] = float(sel.mean())
        s_vals, s_counts = np.unique(starts[sel], return_counts=True)
        start_probs[L] = {int(s): float(c / sel.sum()) for s, c in zip(s_vals, s_counts)}

    # S2 dropout rate per calendar month, among rows where that month is active.
    s2gap = s2_only_gap_mask(test_cube, schema)     # [N, M]
    rates: Dict[str, float] = {}
    for mi, month in enumerate(schema.months):
        denom = int(active[:, mi].sum())
        rates[month] = float(s2gap[:, mi].sum() / denom) if denom else 0.0

    return WindowDist(length_probs=length_probs, start_probs=start_probs, s2_dropout_rates=rates)


# --------------------------------------------------------------------------- #
# Cached bundle
# --------------------------------------------------------------------------- #
@dataclass
class DataBundle:
    train_cube: np.ndarray
    train_ids: np.ndarray
    y: np.ndarray
    test_cube: np.ndarray
    test_ids: np.ndarray
    schema: Schema
    window_dist: WindowDist
    sample_submission: pd.DataFrame


def _data_cache_key(cfg: dict) -> str:
    return config_hash({
        "data": cfg["data"],
        "sar_units": cfg["features"]["sar_units"],
        "paths": {k: cfg["paths"][k] for k in ("train_csv", "test_csv")},
    })


def load_bundle(cfg: dict, use_cache: bool = True) -> DataBundle:
    """Full load -> cubes + schema + window distribution, with npz caching."""
    cache_dir = resolve_path(cfg, "cache_dir")
    key = _data_cache_key(cfg)
    cache = cache_dir / f"cubes_{key}.npz"

    train, test, sample = load_raw(cfg)
    schema = discover_schema(train, cfg)
    test_schema = discover_schema(test, cfg)
    if schema.bands != test_schema.bands or schema.months != test_schema.months:
        raise ValueError("Train/test schema mismatch in bands or months.")

    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        schema.sar_units = str(z["sar_units"])
        wd = z["window_dist"].item()
        log.info("Loaded cubes from cache %s (sar_units=%s)", cache.name, schema.sar_units)
        return DataBundle(
            train_cube=z["train_cube"], train_ids=z["train_ids"], y=z["y"],
            test_cube=z["test_cube"], test_ids=z["test_ids"],
            schema=schema, window_dist=wd, sample_submission=sample,
        )

    train = dedup_train(train, schema)
    train_cube, train_ids, y = to_cube(train, schema)
    test_cube, test_ids, _ = to_cube(test, schema)
    schema.sar_units = _detect_sar_units(train_cube, schema, cfg["features"]["sar_units"])
    log.info("SAR units detected: %s", schema.sar_units)

    window_dist = measure_window_dist(test_cube, schema, cfg)
    log.info("Test window length p(L): %s",
             {k: round(v, 3) for k, v in sorted(window_dist.length_probs.items())})

    save_npz_atomic(
        cache,
        train_cube=train_cube, train_ids=train_ids, y=y,
        test_cube=test_cube, test_ids=test_ids,
        sar_units=schema.sar_units,
        window_dist=np.array(window_dist, dtype=object),
    )
    return DataBundle(
        train_cube=train_cube, train_ids=train_ids, y=y,
        test_cube=test_cube, test_ids=test_ids,
        schema=schema, window_dist=window_dist, sample_submission=sample,
    )
