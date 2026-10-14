"""prepare.py — FROZEN harness for the GeoAI aquaculture loop. Do not edit: the molab runner refuses a run whose
prepare.py does not hash to PREPARE_SHA256 in notebooks/molab_runner.py, and exp.py refuses to compare results made
under different hashes.

Data: data/raw/Train.csv (1,821 rows, 12 full months), data/raw/Test.csv (1,030 rows, 4-6 consecutive months),
data/raw/folds.csv (the pinned fold of every training row; written by scripts/push_data_to_hf.py so molab's own
scikit-learn cannot reshuffle them).

Metric: 0.6 * F1(TargetF1) + 0.4 * ROC-AUC(TargetRAUC). TargetF1 must be TargetRAUC >= 0.5: the rules forbid
choosing a threshold, so build_submission derives it and validate_submission checks it.

Local CV is NOT a leaderboard proxy here (3rd place built nine protocols and all nine misranked a known pair; our own
OOF was anti-correlated with the board). It is reported, never used to adopt. Adoption is by Zindi score: exp.py lb.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ID_COL, TARGET = "ID", "label"
N_FOLDS, FOLD_SEED = 5, 42
FOLD_HASH = "e1200165fc24"   # make_folds() with scikit-learn 1.9.0; see fold_hash()
BANDS = ["blue", "green", "red", "nir", "nira", "swir1", "swir2", "re1", "re2", "re3", "VH", "VV"]
OPT = BANDS[:10]
SAR = ["VH", "VV"]
SENTINEL = -9999
VALIDATION = ("StratifiedKFold(5, shuffle, seed 42) pinned in folds.csv; OOF is a smoke test only. "
              "Adoption is by Zindi leaderboard score (exp.py lb).")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw frames with the -9999 sentinel intact (experiment code decides how to treat it)."""
    train = pd.read_csv(RAW / "Train.csv")
    test = pd.read_csv(RAW / "Test.csv")
    cols = [f"{b}_{m:02d}" for m in range(1, 13) for b in BANDS]
    missing = [c for c in cols if c not in test.columns or c not in train.columns]
    if missing:
        raise RuntimeError(f"schema changed: {missing[:5]}")
    if len(train) != 1821 or len(test) != 1030:
        raise RuntimeError(f"unexpected sizes: train {len(train)}, test {len(test)}")
    return train, test


def make_folds(train: pd.DataFrame) -> np.ndarray:
    """Fold index per training row. Only scripts/push_data_to_hf.py calls this; runs read folds.csv instead."""
    from sklearn.model_selection import StratifiedKFold

    fold = np.full(len(train), -1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(train)), train[TARGET].to_numpy())):
        fold[va] = k
    return fold


def fold_hash(ids, fold) -> str:
    text = "\n".join(f"{i},{k}" for i, k in zip(ids, fold))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def folds(train: pd.DataFrame) -> np.ndarray:
    """The pinned folds from data/raw/folds.csv, checked against FOLD_HASH."""
    f = pd.read_csv(RAW / "folds.csv")
    if list(f[ID_COL]) != list(train[ID_COL]):
        raise RuntimeError("folds.csv is not in Train.csv order")
    h = fold_hash(f[ID_COL], f["fold"])
    if h != FOLD_HASH:
        raise RuntimeError(f"folds.csv hash {h} != frozen {FOLD_HASH}")
    return f["fold"].to_numpy()


def score(y: np.ndarray, p: np.ndarray) -> dict:
    """The competition metric on probabilities p, with the legal 0.5 cut for the F1 half."""
    f1 = f1_score(y, (np.asarray(p) >= 0.5).astype(int))
    auc = roc_auc_score(y, p)
    return {"score": 0.6 * f1 + 0.4 * auc, "f1": float(f1), "auc": float(auc)}


def build_submission(test: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    p = np.asarray(p, dtype=float)
    return pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), "TargetF1": (p >= 0.5).astype(int), "TargetRAUC": p})


def validate_submission(sub: pd.DataFrame) -> None:
    sample = pd.read_csv(RAW / "SampleSubmission.csv")
    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample) and set(sub[ID_COL]) == set(sample[ID_COL]), "IDs do not match the sample"
    assert sub["TargetRAUC"].between(0, 1).all() and sub["TargetRAUC"].notna().all(), "TargetRAUC outside [0, 1]"
    assert (sub["TargetF1"] == (sub["TargetRAUC"] >= 0.5).astype(int)).all(), "TargetF1 must be TargetRAUC >= 0.5"


def harness_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
