"""Shared utilities: config loading, deterministic seeding, hashing,
metrics, the greppable run-summary block, and the results.tsv appender.

These follow the house "autoresearch" conventions (see the starter pack):
every run ends with a parseable metric block and appends one row to an
append-only experiment log.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import yaml

# Repository root = parent of the src/ directory that holds this file.
ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the YAML config. Relative paths are resolved against ROOT."""
    if path is None:
        path = ROOT / "config" / "config.yaml"
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve a path from cfg['paths'][key] against ROOT."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def config_hash(obj: Any, length: int = 12) -> str:
    """Stable short hash of any JSON-serialisable config subset.

    Used to key on-disk caches so that changing an unrelated knob does not
    invalidate an artifact it does not affect.
    """
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:length]


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def set_global_seeds(seed: int = 42) -> None:
    """Lock python-hash, random and numpy RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def derive_seed(base: int, *tags: int) -> int:
    """Deterministic 32-bit seed from a base seed plus integer tags.

    Lets us give every (row, view) its own reproducible RNG without any
    global mutable state.
    """
    h = hashlib.sha1(("_".join(map(str, (base, *tags)))).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def rng_for(base: int, *tags: int) -> np.random.Generator:
    """A numpy Generator seeded deterministically from base + tags."""
    return np.random.default_rng(derive_seed(base, *tags))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def f1_at(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    """F1 for the positive class at a given probability threshold."""
    from sklearn.metrics import f1_score

    pred = (np.asarray(prob) >= threshold).astype(int)
    return float(f1_score(y_true, pred, zero_division=0))


def roc_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, prob))


def combined_score(f1: float, auc: float) -> float:
    """Competition metric: 0.6 * F1 + 0.4 * ROC-AUC."""
    return 0.6 * f1 + 0.4 * auc


def evaluate_oof(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Bundle the three headline numbers at a given threshold."""
    f1 = f1_at(y_true, prob, threshold)
    auc = roc_auc(y_true, prob)
    return {"f1": f1, "auc": auc, "combined": combined_score(f1, auc)}


# --------------------------------------------------------------------------- #
# Logging + greppable summary + results.tsv
# --------------------------------------------------------------------------- #
def get_logger(name: str = "geoai") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def print_summary(name: str, metrics: Dict[str, Any], n_features: int) -> None:
    """Emit the canonical, greppable end-of-run block.

    Keys mirror the starter-pack convention (final_oof/mean_fold/std_fold/
    n_features) plus competition-specific lines.
    """
    fold_scores = metrics.get("fold_scores", [])
    fold_str = ",".join(f"{s:.5f}" for s in fold_scores)
    print("=" * 60)
    print(f"run: {name}")
    print(f"final_oof: {metrics.get('final_oof', float('nan')):.5f}")
    print(f"mean_fold: {metrics.get('mean_fold', float('nan')):.5f}")
    print(f"std_fold: {metrics.get('std_fold', float('nan')):.5f}")
    print(f"n_features: {n_features}")
    print(f"oof_f1@0.5: {metrics.get('oof_f1', float('nan')):.5f}")
    print(f"oof_auc: {metrics.get('oof_auc', float('nan')):.5f}")
    print(f"oof_combined: {metrics.get('oof_combined', float('nan')):.5f}")
    print(f"t_star: {metrics.get('t_star', float('nan')):.4f}")
    print(f"fold_scores: {fold_str}")
    print("=" * 60)


def append_results_tsv(
    path: str | Path,
    *,
    commit: str,
    final_oof: float,
    mean_fold: float,
    std_fold: float,
    n_features: int,
    status: str,
    description: str,
) -> None:
    """Append one experiment row (creates header on first write)."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "commit\tfinal_oof\tmean_fold\tstd_fold\tn_features\tstatus\tdescription\n"
    new_file = not path.exists()
    row = (
        f"{commit}\t{final_oof:.5f}\t{mean_fold:.5f}\t{std_fold:.5f}\t"
        f"{n_features}\t{status}\t{description}\n"
    )
    with open(path, "a", encoding="utf-8") as fh:
        if new_file:
            fh.write(header)
        fh.write(row)


def git_commit_short() -> str:
    """Best-effort short git hash; 'nogit' if unavailable."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def save_npz_atomic(path: str | Path, retries: int = 5, **arrays) -> None:
    """np.savez_compressed with retry-on-PermissionError (OneDrive locks)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(retries):
        try:
            # Write through an open handle so numpy does not append ".npz"
            # to our ".tmp" filename (which would break the os.replace).
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **arrays)
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))
