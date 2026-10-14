"""experiment.py — the one file an experiment edits. DEFAULT_CONFIG reproduces the 3rd-place solution
(github.com/zinmori/geoai-aquaculture-pond-identification, script.py v9, private 0.947): sensor-wise masking from real
test patterns, 316 temporal descriptors, LightGBM + CatBoost + XGBoost per fold, Platt per model, blend weights on OOF,
seed average, Saerens/EM prior, then pseudo-labelling rounds. Feature extraction and masking are checked against the
original byte for byte in tests/test_harness.py.

Work is split into phases (base, pl_r1, ...) and, inside a phase, seeds, so autoresearch/remote.py can upload each
finished seed and resume after a molab disconnect. A config key that is not used here is a bug: add it or drop it.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

import prepare as P

DEFAULT_CONFIG = {
    "model": "gbdt3",
    "seeds": [42, 123, 456, 789, 1337, 2024, 7, 99],
    "features": {
        "perm": False,          # our VH-permanence block: share of radar months with VH < -19/-21/-23 dB, dual-pol gate
    },
    "pl": [[0.90, 0.10], [0.85, 0.15], [0.80, 0.20]],   # pseudo-label rounds: (positive >=, negative <=)
    "prior": "saerens",          # "saerens" | "none"
    "blend_step": 0.05,
    "params": {
        "lgb": {"n_estimators": 1200, "learning_rate": 0.01, "max_depth": 6, "scale_pos_weight": 1.5,
                "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
                "reg_lambda": 1.0},
        "cb": {"iterations": 1200, "learning_rate": 0.01, "depth": 6, "auto_class_weights": "Balanced",
               "l2_leaf_reg": 5, "bagging_temperature": 0.5, "subsample": 0.8},
        "xgb": {"n_estimators": 1200, "learning_rate": 0.01, "max_depth": 6, "scale_pos_weight": 1.5,
                "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
                "reg_lambda": 1.0},
        "early_stopping": 50,
    },
}


def phases(cfg: dict) -> list[tuple[str, tuple[float, float] | None]]:
    return [("base", None)] + [(f"pl_r{i}", (hi, lo)) for i, (hi, lo) in enumerate(cfg["pl"], 1)]


# ── masking ──────────────────────────────────────────────────────────────────


def test_patterns(test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per row and month: is the optical stack present, is the radar stack present (real test acquisitions)."""
    n = len(test)
    t_opt = np.zeros((n, 12), dtype=bool)
    t_sar = np.zeros((n, 12), dtype=bool)
    for m in range(1, 13):
        ms = f"{m:02d}"
        t_sar[:, m - 1] = (test[f"VH_{ms}"].values != P.SENTINEL) & (test[f"VV_{ms}"].values != P.SENTINEL)
        t_opt[:, m - 1] = np.all([test[f"{b}_{ms}"].values != P.SENTINEL
                                  for b in ["blue", "green", "red", "nir", "swir1"]], axis=0)
    return t_opt, t_sar


def mask_train_like_test(train_nan: pd.DataFrame, t_opt, t_sar, seed: int) -> pd.DataFrame:
    """Each training row borrows one random real test row's observation pattern, sensor by sensor."""
    rng = random.Random(seed)
    tm = train_nan.copy()
    idx = [rng.randrange(len(t_opt)) for _ in range(len(tm))]
    o, s = t_opt[idx], t_sar[idx]
    for m in range(1, 13):
        ms = f"{m:02d}"
        rows = np.where(~o[:, m - 1])[0]
        if len(rows):
            tm.loc[tm.index[rows], [f"{b}_{ms}" for b in P.OPT]] = np.nan
        rows = np.where(~s[:, m - 1])[0]
        if len(rows):
            tm.loc[tm.index[rows], [f"{b}_{ms}" for b in P.SAR]] = np.nan
    return tm


# ── features ─────────────────────────────────────────────────────────────────


def _row_corr(x, y, valid):
    n_v = np.sum(valid, axis=1)
    mx = np.sum(np.where(valid, x, 0.0), axis=1) / (n_v + 1e-12)
    my = np.sum(np.where(valid, y, 0.0), axis=1) / (n_v + 1e-12)
    xm = np.where(valid, x - mx[:, None], 0.0)
    ym = np.where(valid, y - my[:, None], 0.0)
    den = np.sqrt(np.sum(xm**2, axis=1) * np.sum(ym**2, axis=1))
    corr = np.zeros(len(x))
    ok = (n_v > 1) & (den > 1e-10)
    corr[ok] = np.sum(xm * ym, axis=1)[ok] / den[ok]
    return corr


def extract_features(df: pd.DataFrame, feat: dict) -> pd.DataFrame:
    """316 descriptors over each series' own observed months (+4 with feat['perm']). df has NaN for missing."""
    N = len(df)
    a = {b: np.stack([df[f"{b}_{m:02d}"].values for m in range(1, 13)], axis=1).astype(float) for b in P.BANDS}
    opt_mask = np.ones((N, 12), dtype=bool)
    for b in ["blue", "green", "red", "nir", "nira", "swir1", "swir2", "re1", "re2"]:
        opt_mask &= ~np.isnan(a[b])
    sar_mask = ~np.isnan(a["VH"]) & ~np.isnan(a["VV"])
    vh_lin = 10.0 ** (a["VH"] / 10.0)
    vv_lin = 10.0 ** (a["VV"] / 10.0)

    with np.errstate(all="ignore"):
        mndwi = (a["green"] - a["swir1"]) / (a["green"] + a["swir1"] + 1e-8)
        ndvi = (a["nir"] - a["red"]) / (a["nir"] + a["red"] + 1e-8)
        awei = 4.0 * (a["green"] - a["swir1"]) - (0.25 * a["nir"] + 2.75 * a["swir2"])
        series = {
            "mndwi": mndwi, "ndvi": ndvi, "sar": vh_lin + vv_lin,
            "evi": 2.5 * (a["nir"] - a["red"]) / (a["nir"] + 6.0 * a["red"] - 7.5 * a["blue"] + 1.0 + 1e-8),
            "sdwi": np.log(10.0 * vv_lin * vh_lin + 1e-8) - 8.0,
            "sabi": (a["nir"] - a["red"]) / (a["green"] + a["blue"] + 1e-8),
            "cdom": a["green"] / (a["red"] + 1e-8),
            "mci": a["re1"] - a["red"] - 0.5333 * (a["re2"] - a["red"]),
            "twobda": a["re1"] / (a["red"] + 1e-8),
            "ndsi": (a["green"] - a["swir2"]) / (a["green"] + a["swir2"] + 1e-8),
            "awei": awei, "vh_vv": a["VH"] - a["VV"],
            "nira_ndvi": (a["nira"] - a["red"]) / (a["nira"] + a["red"] + 1e-8),
            "rvi": 4.0 * vh_lin / (vh_lin + vv_lin + 1e-8),
            "vh_vv_ratio": vh_lin / (vv_lin + 1e-8),
            "water_product": mndwi * (1.0 - ndvi),
            "awei_mndwi_diff": awei - mndwi,
        }
    sar_series = {"sar", "sdwi", "vh_vv", "rvi", "vh_vv_ratio"}
    smask = {k: (sar_mask if k in sar_series else opt_mask) for k in series}
    for k in series:
        series[k] = np.where(smask[k], series[k], np.nan)

    f = {}
    for name, arr in series.items():
        n_s = np.sum(smask[name], axis=1)
        with np.errstate(all="ignore"):
            stats = {f"p{q}": np.nanpercentile(arr, q, axis=1) for q in (5, 10, 25, 50, 75, 90, 95)}
            mean_val = np.nanmean(arr, axis=1, keepdims=True)
            diff = np.nan_to_num(arr - mean_val, nan=0.0)
            n_col = np.maximum(n_s, 1)[:, None]
            m2 = (np.sum(diff**2, axis=1, keepdims=True) / n_col).squeeze(-1)
            m3 = (np.sum(diff**3, axis=1, keepdims=True) / n_col).squeeze(-1)
            m4 = (np.sum(diff**4, axis=1, keepdims=True) / n_col).squeeze(-1)
            sd = np.sqrt(np.maximum(m2, 0.0))
            std_val = np.nanstd(arr, axis=1)
            stats.update({
                "min": np.nanmin(arr, axis=1), "max": np.nanmax(arr, axis=1), "std": std_val,
                "frac_pos": np.nanmean(np.where(np.isnan(arr), np.nan, arr > 0), axis=1),
                "skew": np.where((n_s >= 3) & (sd > 1e-10), m3 / m2**1.5, 0.0),
                "kurtosis": np.where((n_s >= 4) & (sd > 1e-10), m4 / m2**2 - 3.0, 0.0),
                "cv": std_val / (np.abs(mean_val.squeeze(-1)) + 1e-8),
            })
        for nm, v in stats.items():
            f[f"{name}_{nm}"] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        f[f"{name}_range"] = f[f"{name}_max"] - f[f"{name}_min"]
        f[f"{name}_iqr"] = f[f"{name}_p75"] - f[f"{name}_p25"]

    grad_names = ["mndwi", "ndvi", "sar", "ndsi", "rvi", "water_product"]
    ac_names = ["mndwi", "ndvi", "sar", "sdwi"]
    for name in grad_names:
        for k in ("grad_mean", "grad_std", "grad_max"):
            f[f"{name}_{k}"] = np.zeros(N)
    for name in ac_names:
        f[f"{name}_autocorr"] = np.zeros(N)
        f[f"{name}_masd"] = np.zeros(N)
    for name in set(grad_names) | set(ac_names):
        m, arr = smask[name], series[name]
        for i in range(N):
            vals = arr[i, np.where(m[i])[0]]
            if len(vals) >= 2 and name in grad_names:
                d = vals[1:] - vals[:-1]
                f[f"{name}_grad_mean"][i] = np.mean(d)
                f[f"{name}_grad_std"][i] = np.std(d)
                f[f"{name}_grad_max"][i] = np.max(np.abs(d))
            if len(vals) >= 4 and name in ac_names:
                if np.std(vals) > 1e-10:
                    xm, ym = vals[:-1] - np.mean(vals[:-1]), vals[1:] - np.mean(vals[1:])
                    r_den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
                    if r_den > 1e-10:
                        r = np.sum(xm * ym) / r_den
                        f[f"{name}_autocorr"][i] = r if np.isfinite(r) else 0.0
                else:
                    f[f"{name}_autocorr"][i] = 1.0
                f[f"{name}_masd"][i] = np.mean(np.abs(vals[1:] - vals[:-1]))

    both = opt_mask & sar_mask
    s = {k: np.nan_to_num(v) for k, v in series.items()}
    with np.errstate(all="ignore"):
        n_o = np.maximum(np.sum(opt_mask, axis=1), 1)
        n_sr = np.maximum(np.sum(sar_mask, axis=1), 1)
        n_b = np.maximum(np.sum(both, axis=1), 1)
        f["water_freq_strict"] = np.sum(both & (series["mndwi"] > 0.5) & (series["sar"] < 0.1), axis=1) / n_b
        f["water_freq_sdwi"] = np.sum(sar_mask & (series["sdwi"] > -1.5), axis=1) / n_sr
        f["mndwi_sar_corr"] = _row_corr(s["mndwi"], s["sar"], both)
        f["mndwi_ndvi_corr"] = _row_corr(s["mndwi"], s["ndvi"], opt_mask)
        f["ndsi_mndwi_corr"] = _row_corr(s["ndsi"], s["mndwi"], opt_mask)
        f["water_score"] = f["mndwi_p25"] * (1 - np.minimum(f["sar_p25"], 1.0))
        f["water_floor"] = f["mndwi_p10"] - f["sar_p10"]
        f["water_consist"] = f["mndwi_frac_pos"] * (1 - f["sar_frac_pos"])
        f["rvi_mndwi_corr"] = _row_corr(s["rvi"], s["mndwi"], both)
        f["water_perm"] = np.sum(opt_mask & (series["mndwi"] > 0) & (series["ndvi"] < 0.3), axis=1) / n_o
        if feat.get("perm"):
            vh = np.where(sar_mask, a["VH"], np.nan)
            vv = np.where(sar_mask, a["VV"], np.nan)
            f["vh_perm21"] = np.sum(sar_mask & (vh < -21), axis=1) / n_sr
            f["dualpol_gate"] = np.sum(sar_mask & (vh < -21) & ((vh - vv) < -8), axis=1) / n_sr
            f["vh_perm19"] = np.sum(sar_mask & (vh < -19), axis=1) / n_sr
            f["vh_perm23"] = np.sum(sar_mask & (vh < -23), axis=1) / n_sr

    vany = opt_mask | sar_mask
    n_v = np.sum(vany, axis=1)
    first = np.where(n_v > 0, np.argmax(vany, axis=1) + 1, 0)
    last = np.where(n_v > 0, 12 - np.argmax(vany[:, ::-1], axis=1), 0)
    block = np.where(n_v > 0, last - first + 1, 0)
    f["n_months_observed"] = n_v
    f["first_observed_month"] = first
    f["last_observed_month"] = last
    f["block_length"] = block
    f["obs_month_mid"] = np.where(n_v > 0, np.sum(vany * np.arange(1, 13), axis=1) / np.maximum(n_v, 1), 0.0)
    f["obs_density"] = np.where(block > 0, n_v / (block + 1e-8), 0.0)
    f["n_optical_months"] = np.sum(opt_mask, axis=1).astype(float)
    f["n_saronly_months"] = np.sum(sar_mask & ~opt_mask, axis=1).astype(float)
    return pd.DataFrame(f).replace([np.inf, -np.inf], np.nan).fillna(0)


def build_views(cfg: dict, train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """{seed: masked-train features} and the test features. Deterministic given the config."""
    t_opt, t_sar = test_patterns(test)
    train_nan = train.replace(P.SENTINEL, np.nan)
    X_test = extract_features(test.replace(P.SENTINEL, np.nan), cfg["features"])
    views = {sd: extract_features(mask_train_like_test(train_nan, t_opt, t_sar, sd), cfg["features"])
             for sd in cfg["seeds"]}
    return views, X_test


# ── models ───────────────────────────────────────────────────────────────────


def fit_fold(cfg, X_tr, y_tr, X_val, y_val, X_test, seed, fold, threads=1):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    pr, es = cfg["params"], cfg["params"]["early_stopping"]
    m_lgb = lgb.LGBMClassifier(random_state=seed + fold, verbose=-1, n_jobs=threads, **pr["lgb"])
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(es, verbose=False)])
    m_cb = CatBoostClassifier(random_seed=seed + fold, thread_count=threads, verbose=0, **pr["cb"])
    m_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=es, verbose=False)
    m_xgb = xgb.XGBClassifier(random_state=seed + fold, n_jobs=threads, eval_metric="logloss",
                              early_stopping_rounds=es, **pr["xgb"])
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    ms = (m_lgb, m_cb, m_xgb)
    return [m.predict_proba(X_val)[:, 1] for m in ms], [m.predict_proba(X_test)[:, 1] for m in ms]


def blend_and_calibrate(oof_raw, test_raw, y, step=0.05):
    """Platt per model on its full OOF, then the simplex weights maximising 0.6*F1 + 0.4*AUC on OOF."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score

    oof_c, test_c = [], []
    for o, t in zip(oof_raw, test_raw):
        cal = LogisticRegression(C=1e5).fit(o.reshape(-1, 1), y)
        oof_c.append(cal.predict_proba(o.reshape(-1, 1))[:, 1])
        test_c.append(cal.predict_proba(np.mean(t, axis=0).reshape(-1, 1))[:, 1])
    best, bw = 0, (1 / 3, 1 / 3, 1 / 3)
    for w1 in np.arange(0, 1.01, step):
        for w2 in np.arange(0, 1.01 - w1, step):
            w3 = max(1 - w1 - w2, 0)
            ens = w1 * oof_c[0] + w2 * oof_c[1] + w3 * oof_c[2]
            sc = 0.6 * f1_score(y, ens >= 0.5) + 0.4 * roc_auc_score(y, ens)
            if sc > best:
                best, bw = sc, (w1, w2, w3)
    w1, w2, w3 = bw
    return (w1 * oof_c[0] + w2 * oof_c[1] + w3 * oof_c[2], w1 * test_c[0] + w2 * test_c[1] + w3 * test_c[2],
            bw, best)


def run_seed(cfg, seed, X_train, X_test, y, fold, threads=1, X_pseudo=None, y_pseudo=None) -> dict:
    """One masking seed in one phase: 5 folds x 3 models, Platt, blend."""
    oof_raw = [np.zeros(len(y)) for _ in range(3)]
    test_raw = [[] for _ in range(3)]
    for k in range(P.N_FOLDS):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        X_tr, y_tr = X_train.iloc[tr], y[tr]
        if X_pseudo is not None:
            X_tr = pd.concat([X_tr, X_pseudo], ignore_index=True)
            y_tr = np.concatenate([y_tr, y_pseudo])
        ov, tv = fit_fold(cfg, X_tr, y_tr, X_train.iloc[va], y[va], X_test, seed, k, threads)
        for j in range(3):
            oof_raw[j][va] = ov[j]
            test_raw[j].append(tv[j])
    oof, test_p, w, cv = blend_and_calibrate(oof_raw, test_raw, y, cfg["blend_step"])
    return {"seed": seed, "oof": oof, "test": test_p, "weights": list(w), "cv": float(cv)}


def pseudo_set(X_test: pd.DataFrame, p_ref: np.ndarray, hi: float, lo: float):
    pos, neg = p_ref >= hi, p_ref <= lo
    X = pd.concat([X_test[pos], X_test[neg]], ignore_index=True)
    return X, np.concatenate([np.ones(pos.sum()), np.zeros(neg.sum())])


def saerens(p_test: np.ndarray, train_prior: float, max_iter=100, tol=1e-6) -> tuple[np.ndarray, float]:
    """EM estimate of the test positive rate, then odds rescaling of the probabilities. The cut stays at 0.5."""
    p_test = np.clip(p_test, 1e-6, 1 - 1e-6)
    prior = train_prior
    for _ in range(max_iter):
        w = prior * (1 - train_prior) / (train_prior * (1 - prior) + 1e-12)
        new = (p_test * w / (p_test * w + (1 - p_test) + 1e-12)).mean()
        done = abs(new - prior) < tol
        prior = new
        if done:
            break
    w = prior * (1 - train_prior) / (train_prior * (1 - prior) + 1e-12)
    return p_test * w / (p_test * w + (1 - p_test) + 1e-12), float(prior)


def finish_phase(cfg, seed_results: list[dict], train_prior: float) -> tuple[np.ndarray, dict]:
    """Average the seeds, apply the prior correction. Returns final test probabilities and a summary."""
    p = np.mean([r["test"] for r in seed_results], axis=0)
    prior = None
    if cfg["prior"] == "saerens":
        p, prior = saerens(p, train_prior)
    return p, {"em_prior": prior, "positives": int((p >= 0.5).sum()),
               "cv_mean": float(np.mean([r["cv"] for r in seed_results]))}
