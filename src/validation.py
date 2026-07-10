"""Cross-validation with masking-aware, leak-free out-of-fold estimation.

The subtle part: augmentation must not let a row's masked twin appear in both
train and validation. So folds are defined on the ORIGINAL rows; every
augmented view inherits its row's fold. Held-out rows are scored on R
independent masked views (test-like), averaged per row -> one honest OOF
probability per original row.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .data import Schema, WindowDist
from .features import build_oof_views, build_train_matrix
from .models import Ensemble
from .utils import (combined_score, evaluate_oof, f1_at, get_logger, roc_auc)

log = get_logger()


@dataclass
class CVResult:
    oof_prob: np.ndarray            # [N] honest OOF probability per original row
    y: np.ndarray                   # [N]
    fold_scores: List[float] = field(default_factory=list)  # combined @0.5 per fold-repeat
    n_features: int = 0
    feature_names: List[str] = field(default_factory=list)


def make_fold_iter(y: np.ndarray, cfg: dict, smoke: bool = False):
    """Yield (repeat, fold, train_rows, val_rows) over original rows."""
    n_splits = cfg["run"]["smoke_splits"] if smoke else cfg["cv"]["n_splits"]
    n_repeats = 1 if smoke else cfg["cv"]["n_repeats"]
    base_seed = cfg["seed"]
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=cfg["cv"]["shuffle"],
                              random_state=base_seed + rep)
        for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
            yield rep, fold, tr, va


def run_cv(train_cube: np.ndarray, y: np.ndarray, schema: Schema,
           window_dist: WindowDist, cfg: dict, smoke: bool = False) -> CVResult:
    """Masking-aware repeated Stratified K-fold CV.

    Returns per-row OOF probabilities averaged across repeats.
    """
    K = cfg["run"]["smoke_K"] if smoke else cfg["augment"]["K"]
    R = cfg["run"]["smoke_R"] if smoke else cfg["augment"]["R"]
    n_estimators = cfg["run"]["smoke_trees"] if smoke else None
    base_seed = cfg["seed"]
    n = len(y)

    oof_sum = np.zeros(n, dtype=np.float64)
    oof_cnt = np.zeros(n, dtype=np.int64)
    fold_scores: List[float] = []
    feature_names: List[str] = []

    for rep, fold, tr_rows, va_rows in make_fold_iter(y, cfg, smoke):
        # Build augmented training matrix from training rows only.
        X_tr, y_tr, _, feature_names = build_train_matrix(
            train_cube[tr_rows], y[tr_rows], schema, window_dist, cfg,
            K=K, seed=base_seed + rep, row_ids=tr_rows,
        )
        # Build R masked views per held-out row (honest, test-like).
        X_va, owners, _ = build_oof_views(
            train_cube, schema, window_dist, cfg, R=R,
            seed=base_seed + rep, row_indices=va_rows,
        )
        model = Ensemble(cfg, n_estimators=n_estimators).fit(X_tr, y_tr)
        prob_views = model.predict_proba(X_va)         # [len(va)*R]
        # Average the R views per held-out row.
        prob_rows = np.zeros(len(va_rows), dtype=np.float64)
        for pos in range(len(va_rows)):
            prob_rows[pos] = prob_views[owners == pos].mean()

        oof_sum[va_rows] += prob_rows
        oof_cnt[va_rows] += 1

        fs = combined_score(f1_at(y[va_rows], prob_rows, 0.5), roc_auc(y[va_rows], prob_rows))
        fold_scores.append(fs)
        log.info("rep %d fold %d: combined@0.5=%.5f (train views=%d)",
                 rep, fold, fs, len(y_tr))

    oof_prob = oof_sum / np.maximum(oof_cnt, 1)
    return CVResult(oof_prob=oof_prob, y=y, fold_scores=fold_scores,
                    n_features=len(feature_names), feature_names=feature_names)
