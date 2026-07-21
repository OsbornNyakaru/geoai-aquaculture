"""End-to-end pipeline: raw CSVs -> submission.csv.

Stages: load & cube -> masking-aware CV (honest OOF) -> fixed-0.5 calibration
-> fit full ensemble on all augmented train views -> predict test ->
write + validate submission -> print greppable summary -> append results.tsv.

Usage:
    python run_pipeline.py --smoke     # fast (<1 min) end-to-end sanity run
    python run_pipeline.py --full      # full 5x3 CV + full ensemble
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # silence benign sklearn feature-name notes

from src.calibration import (calibrate_for_f1, estimate_test_prior_bbse,
                             find_tstar, score_for_auc)
from src.data import load_bundle
from src.features import build_test_matrix, build_train_matrix
from src.models import Ensemble
from src.utils import (append_results_tsv, config_hash, evaluate_oof,
                       get_logger, git_commit_short, load_config, print_summary,
                       resolve_path, save_npz_atomic, set_global_seeds)
from src.validation import run_cv

log = get_logger()


def apply_overrides(cfg: dict, overrides) -> None:
    """Apply `--set dotted.key=value` overrides onto the loaded config, in place.

    Values are parsed as YAML so `false`, `0`, `1.0`, `none` and `[4,5,6]` all keep their
    natural types. The key must already exist — a typo should fail loudly rather than
    silently create a dead config entry that the pipeline then ignores.
    """
    import yaml as _yaml

    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item!r}")
        dotted, raw = item.split("=", 1)
        parts = dotted.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                raise SystemExit(f"--set: unknown config path {dotted!r}")
            node = node[p]
        leaf = parts[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise SystemExit(f"--set: unknown config path {dotted!r}")
        node[leaf] = _yaml.safe_load(raw)
        log.info("config override: %s = %r", dotted, node[leaf])


def validate_submission(df: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Assert the submission matches the required schema exactly."""
    assert list(df.columns) == ["ID", "TargetF1", "TargetRAUC"], f"bad cols: {list(df.columns)}"
    assert len(df) == len(sample), f"row count {len(df)} != {len(sample)}"
    assert list(df["ID"]) == list(sample["ID"]), "ID order must match SampleSubmission"
    assert set(df["TargetF1"].unique()).issubset({0, 1}), "TargetF1 must be binary"
    assert df["TargetRAUC"].between(0, 1).all(), "TargetRAUC must be in [0,1]"
    assert df["TargetRAUC"].isna().sum() == 0 and df["TargetF1"].isna().sum() == 0
    log.info("Submission schema OK: %d rows, %d unique RAUC values",
             len(df), df["TargetRAUC"].nunique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast subsampled run")
    ap.add_argument("--full", action="store_true", help="full run (default)")
    ap.add_argument("--name", default=None, help="experiment name for logs")
    ap.add_argument("--no-cache", action="store_true", help="ignore data cache")
    ap.add_argument("--model", default="gbdt", choices=["gbdt", "seq"],
                    help="gbdt = LGBM/XGB/CatBoost ensemble; seq = temporal Transformer")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config entry by dotted path, e.g. "
                         "--set seq.consistency_lambda=0 --set seq.relative_time=false. "
                         "Repeatable. Lets one run reproduce any historical variant without "
                         "editing config.yaml (used by the offline-validator retro-fit).")
    args = ap.parse_args()
    smoke = args.smoke and not args.full

    cfg = load_config()
    apply_overrides(cfg, args.set)
    set_global_seeds(cfg["seed"])
    name = args.name or ("smoke" if smoke else "full")

    bundle = load_bundle(cfg, use_cache=not args.no_cache)
    train_cube, y = bundle.train_cube, bundle.y
    test_cube, schema, wd = bundle.test_cube, bundle.schema, bundle.window_dist

    if smoke:
        rng = np.random.default_rng(cfg["seed"])
        sub = rng.choice(len(y), cfg["run"]["smoke_subsample"], replace=False)
        train_cube, y = train_cube[sub], y[sub]

    # ---- Model path: GBDT ensemble or temporal Transformer ----
    class _CV:  # lightweight holder shared by both paths
        pass

    if args.model == "seq":
        from src.seq_model import run_seq_cv
        oof_prob, p_test_raw, fold_scores = run_seq_cv(
            train_cube, y, test_cube, schema, wd, cfg, smoke=smoke)
        cv = _CV()
        cv.oof_prob, cv.y, cv.fold_scores = oof_prob, y, fold_scores
        # per-month input dim = values + missing (2B) + enabled transfer channels
        _ch = cfg["seq"].get("channels") or {}
        _extra = ((schema.n_bands if _ch.get("per_cell_detrend") else 0)
                  + (schema.n_bands if _ch.get("deltas") else 0)
                  + (5 if _ch.get("indices") else 0)
                  + (schema.n_bands if _ch.get("rank") else 0))
        cv.n_features = 2 * schema.n_bands + _extra
    else:
        # Masking-aware OOF, then fit the full ensemble on all augmented views.
        cv = run_cv(train_cube, y, schema, wd, cfg, smoke=smoke)
        K = cfg["run"]["smoke_K"] if smoke else cfg["augment"]["K"]
        n_est = cfg["run"]["smoke_trees"] if smoke else None
        X_tr, y_tr, _, feat_names = build_train_matrix(
            train_cube, y, schema, wd, cfg, K=K, seed=cfg["seed"], row_ids=np.arange(len(y)))
        X_te, _ = build_test_matrix(test_cube, schema, cfg)
        full_model = Ensemble(cfg, n_estimators=n_est).fit(X_tr, y_tr)
        p_test_raw = full_model.predict_proba(X_te)

    m_raw = evaluate_oof(cv.y, cv.oof_prob, 0.5)
    log.info("CV OOF (uncalibrated): f1@0.5=%.4f auc=%.4f combined=%.4f",
             m_raw["f1"], m_raw["auc"], m_raw["combined"])

    # ---- Optional: estimate the test prior (label shift) instead of guessing ----
    if cfg["calibration"].get("estimate_prior", False):
        t_hat = find_tstar(cv.y, cv.oof_prob, cfg)
        q_hat = estimate_test_prior_bbse(cv.oof_prob, cv.y, p_test_raw, t_hat)
        log.info("BBSE estimated test prior = %.3f (train prior %.3f) -> assumed_test_prior",
                 q_hat, float(cv.y.mean()))
        cfg["calibration"]["prior_adjust"] = True
        cfg["calibration"]["assumed_test_prior"] = q_hat
        if not cfg["calibration"].get("prior_sweep"):
            cfg["calibration"]["prior_sweep"] = [
                round(min(max(q_hat - 0.05, 0.05), 0.95), 2),
                round(min(max(q_hat + 0.05, 0.05), 0.95), 2),
            ]

    # ---- Fixed-0.5 calibration for TargetF1; monotone score for TargetRAUC ----
    target_f1, t_star, diag = calibrate_for_f1(cv.y, cv.oof_prob, p_test_raw, cfg)
    target_rauc = score_for_auc(p_test_raw)

    # Headline metrics reflect the CALIBRATED submission: F1 from the shifted
    # OOF at 0.5; AUC is invariant to the monotone shift (use raw).
    m = {
        "f1": diag["oof_f1_at_0.5_after"],
        "auc": m_raw["auc"],
        "combined": 0.6 * diag["oof_f1_at_0.5_after"] + 0.4 * m_raw["auc"],
    }
    log.info("t_star=%.4f | calibrated OOF: f1@0.5=%.4f auc=%.4f combined=%.4f | prior-sens=%s",
             t_star, m["f1"], m["auc"], m["combined"],
             {k: round(v, 3) for k, v in diag["prior_sensitivity"].items()})

    # ---- Write + validate submission ----
    sub_df = pd.DataFrame({
        "ID": bundle.test_ids,
        "TargetF1": target_f1.astype(int),
        "TargetRAUC": target_rauc,
    })
    # Enforce SampleSubmission ID order.
    order = {i: k for k, i in enumerate(bundle.sample_submission["ID"])}
    sub_df = sub_df.sort_values("ID", key=lambda s: s.map(order)).reset_index(drop=True)
    validate_submission(sub_df, bundle.sample_submission)

    out_dir = resolve_path(cfg, "submissions_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"submission_{name}.csv")
    sub_df.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)

    # ---- Persist raw OOF + test probabilities for cross-model blending ----
    # tools/blend.py rank-averages these across model families (GBDT + seq),
    # then re-runs fixed-0.5 calibration on the blend. RAW (pre-calibration)
    # probs are saved so the blend controls its own operating point; rank
    # transform inside the blend neutralizes the seq model's overconfidence.
    preds_dir = out_dir / "preds"
    save_npz_atomic(
        preds_dir / f"preds_{name}.npz",
        oof_prob=cv.oof_prob, y=cv.y, p_test_raw=p_test_raw,
        test_ids=np.asarray(bundle.test_ids),
        model=np.array(args.model), prior=np.array(cfg["calibration"].get("assumed_test_prior", 0.5)),
    )
    log.info("Wrote preds bundle %s", preds_dir / f"preds_{name}.npz")

    # ---- Optional prior sweep: one trained model -> many candidate submissions ----
    # The assumed test prior is pure post-processing on the calibrated test
    # probabilities, so we can emit a whole bracket without retraining. Submit
    # whichever scores best on the leaderboard.
    sweep = cfg["calibration"].get("prior_sweep") or []
    from src.calibration import apply_prior
    for pr in sweep:
        tf1 = apply_prior(diag["p_test_shifted"], diag["base_prior"], float(pr))
        sw = pd.DataFrame({"ID": bundle.test_ids, "TargetF1": tf1.astype(int),
                           "TargetRAUC": target_rauc})
        sw = sw.sort_values("ID", key=lambda s: s.map(order)).reset_index(drop=True)
        validate_submission(sw, bundle.sample_submission)
        tag = f"{name}_prior{int(round(float(pr) * 100)):02d}"
        sw.to_csv(out_dir / f"submission_{tag}.csv", index=False)
        log.info("Wrote sweep submission %s (pos-rate %.3f)", tag, float(tf1.mean()))

    # ---- Greppable summary + results.tsv ----
    fold = np.array(cv.fold_scores, dtype=float)
    metrics = {
        "final_oof": m["combined"],
        "mean_fold": float(fold.mean()) if fold.size else float("nan"),
        "std_fold": float(fold.std()) if fold.size else float("nan"),
        "oof_f1": m["f1"],
        "oof_auc": m["auc"],
        "oof_combined": m["combined"],
        "t_star": t_star,
        "fold_scores": cv.fold_scores,
    }
    print_summary(name, metrics, cv.n_features)
    append_results_tsv(
        resolve_path(cfg, "results_tsv"),
        commit=git_commit_short(),
        final_oof=m["combined"], mean_fold=metrics["mean_fold"],
        std_fold=metrics["std_fold"], n_features=cv.n_features,
        status="keep" if not smoke else "smoke",
        description=name + f" model={args.model} "
                    f"cal={cfg['calibration']['method']} "
                    f"prior_adj={cfg['calibration'].get('prior_adjust')}",
    )


if __name__ == "__main__":
    main()
