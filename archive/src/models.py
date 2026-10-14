"""Gradient-boosted model factories and a rank-averaging ensemble.

Three heterogeneous GBDT families (LightGBM, XGBoost, CatBoost). All handle
missing values natively; we pass NaN (converted from the -9999 sentinel) and
let each library learn the optimal default split direction. Determinism knobs
are set so a fixed seed reproduces the same model.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .utils import get_logger

log = get_logger()


# --------------------------------------------------------------------------- #
# Factories — each returns an unfitted estimator with a sklearn-like API.
# --------------------------------------------------------------------------- #
def make_lightgbm(cfg: dict, seed: int, n_estimators: int | None = None):
    import lightgbm as lgb

    p = dict(cfg["models"]["lightgbm"])
    if n_estimators is not None:
        p["n_estimators"] = n_estimators
    return lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        scale_pos_weight=cfg["models"].get("scale_pos_weight", 1.0),
        **p,
    )


def make_xgboost(cfg: dict, seed: int, n_estimators: int | None = None):
    import xgboost as xgb

    p = dict(cfg["models"]["xgboost"])
    if n_estimators is not None:
        p["n_estimators"] = n_estimators
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=cfg["models"].get("scale_pos_weight", 1.0),
        missing=np.nan,
        **p,
    )


def make_catboost(cfg: dict, seed: int, n_estimators: int | None = None):
    from catboost import CatBoostClassifier

    p = dict(cfg["models"]["catboost"])
    if n_estimators is not None:
        p["iterations"] = n_estimators
    spw = cfg["models"].get("scale_pos_weight", 1.0)
    return CatBoostClassifier(
        loss_function="Logloss",
        random_seed=seed,
        scale_pos_weight=spw,
        **p,
    )


_FACTORY = {
    "lightgbm": make_lightgbm,
    "xgboost": make_xgboost,
    "catboost": make_catboost,
}


def _rankit(p: np.ndarray) -> np.ndarray:
    """Map scores to (0,1) by rank — scale-free, ties broken by order."""
    order = np.argsort(np.argsort(p))
    return (order + 0.5) / len(p)


class Ensemble:
    """Multi-family, multi-seed GBDT bag with rank-average blending.

    fit(X, y) trains, for each requested family, one model per bag seed.
    predict_proba(X) returns the blended positive-class probability. With
    blend='rank' the per-family scores are rank-normalised before averaging
    (robust to scale/calibration differences between families); with 'mean'
    they are averaged directly.
    """

    def __init__(self, cfg: dict, n_estimators: int | None = None):
        self.cfg = cfg
        self.families: List[str] = cfg["models"]["use"]
        self.bag_seeds: List[int] = cfg["models"]["bag_seeds"]
        self.blend: str = cfg["models"].get("blend", "rank")
        self.n_estimators = n_estimators
        self.models: Dict[str, list] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ensemble":
        for fam in self.families:
            self.models[fam] = []
            for seed in self.bag_seeds:
                model = _FACTORY[fam](self.cfg, seed, self.n_estimators)
                model.fit(X, y)
                self.models[fam].append(model)
        return self

    def _family_score(self, fam: str, X: np.ndarray) -> np.ndarray:
        preds = [m.predict_proba(X)[:, 1] for m in self.models[fam]]
        return np.mean(preds, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        fam_scores = [self._family_score(fam, X) for fam in self.families]
        if self.blend == "rank":
            fam_scores = [_rankit(s) for s in fam_scores]
        return np.mean(fam_scores, axis=0)
