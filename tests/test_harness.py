"""Harness tests. Data-dependent tests skip when data/raw/ is absent (a fresh public clone)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
import experiment as E  # noqa: E402
import prepare as P  # noqa: E402

needs_data = pytest.mark.skipif(not (P.RAW / "Train.csv").exists(), reason="competition data not present")


def test_metric_is_the_competition_formula():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.8, 0.9])
    s = P.score(y, p)
    assert s["score"] == pytest.approx(0.6 * s["f1"] + 0.4 * s["auc"])
    assert s["f1"] == pytest.approx(2 / 3)    # cut at 0.5: TP 2, FP 1, FN 1


def test_saerens_keeps_ranking_and_moves_prior_up_when_test_is_positive_heavy():
    rng = np.random.default_rng(0)
    p = np.clip(rng.beta(3, 2, 2000), 0.01, 0.99)
    q, prior = E.saerens(p, 0.40)
    assert prior > 0.40
    assert np.array_equal(np.argsort(p, kind="stable"), np.argsort(q, kind="stable"))


def test_phases_follow_pl_rounds():
    assert [n for n, _ in E.phases(E.DEFAULT_CONFIG)] == ["base", "pl_r1", "pl_r2", "pl_r3"]


@needs_data
def test_submission_is_legal():
    _, test = P.load()
    sub = P.build_submission(test, np.linspace(0, 1, len(test)))
    P.validate_submission(sub)
    bad = sub.copy()
    bad.loc[0, "TargetF1"] = 1 - bad.loc[0, "TargetF1"]
    with pytest.raises(AssertionError):
        P.validate_submission(bad)


@needs_data
def test_folds_are_pinned():
    train, _ = P.load()
    fold = P.folds(train)
    assert sorted(np.bincount(fold)) == [364, 364, 364, 364, 365]


@needs_data
def test_masking_matches_test_observation_statistics():
    """3rd place's verification table: masked train vs test, mean observed months ~5.0, radar-only ~0.31."""
    train, test = P.load()
    t_opt, t_sar = E.test_patterns(test)
    tm = E.mask_train_like_test(train.replace(P.SENTINEL, np.nan), t_opt, t_sar, 42)
    X = E.extract_features(tm, {"perm": False})
    assert X.shape[1] == 316
    assert X["n_months_observed"].mean() == pytest.approx(4.997, abs=0.1)
    assert X["n_saronly_months"].mean() == pytest.approx(0.311, abs=0.05)
    assert E.extract_features(tm, {"perm": True}).shape[1] == 320


@needs_data
def test_test_features_have_no_nan():
    _, test = P.load()
    X = E.extract_features(test.replace(P.SENTINEL, np.nan), {"perm": True})
    assert not X.isna().any().any() and len(X) == 1030
    assert isinstance(X, pd.DataFrame)
