"""Rank-average blend of GBDT + seq predictions -> a calibrated submission.

Reads two or more `preds_*.npz` bundles written by run_pipeline.py, rank-averages
their OOF and test scores across model families, then re-runs the same fixed-0.5
calibration + prior correction on the blend.

Rank-averaging (not probability averaging) is deliberate: it uses only each
model's ORDERING, which neutralizes the seq Transformer's overconfident, saturated
probabilities so it cannot dominate the GBDT. The blend's operating point is set
by the shared calibration (assumed_test_prior in config), and --sweep emits extra
candidate submissions at other priors without recomputation.

Usage:
    python tools/blend.py --preds submissions/preds/preds_gbdt_p65.npz \
                                  submissions/preds/preds_seq_v2.npz \
                          --name gbdt_seq --sweep 0.60 0.65 0.70
    # optional relative weights (default equal): --weights 0.6 0.4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> import src

from src.calibration import apply_prior, calibrate_for_f1, score_for_auc
from src.utils import evaluate_oof, get_logger, load_config, resolve_path

log = get_logger()


def _rank(p: np.ndarray) -> np.ndarray:
    """Strictly-monotone rank transform to [0,1] — magnitude-free."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(np.argsort(p))
    return (order + 0.5) / len(p)


def _validate(df: pd.DataFrame, sample: pd.DataFrame) -> None:
    assert list(df.columns) == ["ID", "TargetF1", "TargetRAUC"], f"bad cols: {list(df.columns)}"
    assert len(df) == len(sample), f"row count {len(df)} != {len(sample)}"
    assert list(df["ID"]) == list(sample["ID"]), "ID order must match SampleSubmission"
    assert set(df["TargetF1"].unique()).issubset({0, 1}), "TargetF1 must be binary"
    assert df["TargetRAUC"].between(0, 1).all(), "TargetRAUC must be in [0,1]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="preds_*.npz bundles to blend")
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="relative weights per preds file (default: equal)")
    ap.add_argument("--name", default="blend", help="submission name")
    ap.add_argument("--sweep", nargs="*", type=float, default=None,
                    help="assumed test priors -> one extra submission each")
    args = ap.parse_args()

    cfg = load_config()
    bundles = [np.load(p, allow_pickle=True) for p in args.preds]

    y = bundles[0]["y"]
    test_ids = bundles[0]["test_ids"]
    for p, b in zip(args.preds[1:], bundles[1:]):
        assert np.array_equal(b["y"], y), f"OOF label mismatch in {p}"
        assert np.array_equal(b["test_ids"], test_ids), f"test_id order mismatch in {p}"

    w = np.array(args.weights if args.weights else [1.0] * len(bundles), dtype=float)
    assert len(w) == len(bundles), "weights count != preds count"
    w = w / w.sum()

    # Diagnostic: how correlated are the components? (rank corr on OOF)
    oof_ranks = [_rank(b["oof_prob"]) for b in bundles]
    if len(bundles) == 2:
        rho = float(np.corrcoef(oof_ranks[0], oof_ranks[1])[0, 1])
        log.info("OOF rank correlation between components = %.4f "
                 "(lower = more decorrelated = better blend)", rho)

    oof_blend = np.zeros(len(y))
    test_blend = np.zeros(len(test_ids))
    for wi, path, b in zip(w, args.preds, bundles):
        oof_blend += wi * _rank(b["oof_prob"])
        test_blend += wi * _rank(b["p_test_raw"])
        mc = evaluate_oof(y, b["oof_prob"], 0.5)
        log.info("component w=%.3f model=%s prior=%s | oof auc=%.4f | %s",
                 wi, b["model"], b["prior"], mc["auc"], Path(path).name)

    m = evaluate_oof(y, oof_blend, 0.5)
    log.info("BLEND OOF (uncalibrated): f1@0.5=%.4f auc=%.4f combined=%.4f",
             m["f1"], m["auc"], m["combined"])

    # ⚠️ NON-COMPLIANT EMITTER. This tool still writes through the `pinned` path:
    # calibrate_for_f1() sets a threshold and logit-shifts it onto 0.5, and score_for_auc()
    # emits uniformly-spaced RANKS rather than probabilities. Both violate the stated rules
    # ("Setting a probability threshold is strictly forbidden"; "Zindi will need the raw
    # probabilities"). It is kept only to reproduce historical pinned anchors. The compliant
    # pooling path is src.calibration.calibrated_pool(), used by tools/arch_blend.py and
    # tools/seed_average.py -- prefer those. See REPORT.md section 8.
    if cfg["calibration"].get("compliance_mode", "legal") != "pinned":
        raise SystemExit(
            "tools/blend.py emits a RULES-VIOLATING submission (threshold shift + rank column).\n"
            "Refusing to run under compliance_mode='legal'. Use tools/arch_blend.py or\n"
            "tools/seed_average.py, which pool via calibrated_pool() and cut at a literal 0.5.\n"
            "To reproduce a historical pinned anchor anyway, set calibration.compliance_mode=pinned."
        )
    log.warning("blend.py: emitting through the NON-COMPLIANT pinned path -- DO NOT SUBMIT.")
    target_f1, t_star, diag = calibrate_for_f1(y, oof_blend, test_blend, cfg)
    target_rauc = score_for_auc(test_blend)

    sample = pd.read_csv(resolve_path(cfg, "raw_dir") / cfg["paths"]["sample_submission_csv"])
    order = {i: k for k, i in enumerate(sample["ID"])}
    out_dir = resolve_path(cfg, "submissions_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(tf1: np.ndarray, tag: str) -> None:
        df = pd.DataFrame({"ID": test_ids, "TargetF1": tf1.astype(int), "TargetRAUC": target_rauc})
        df = df.sort_values("ID", key=lambda s: s.map(order)).reset_index(drop=True)
        _validate(df, sample)
        df.to_csv(out_dir / f"submission_{tag}.csv", index=False)
        log.info("Wrote submission_%s.csv (pos-rate %.3f)", tag, float(tf1.mean()))

    _write(target_f1, args.name)
    for pr in (args.sweep or []):
        tf1 = apply_prior(diag["p_test_shifted"], diag["base_prior"], float(pr))
        _write(tf1, f"{args.name}_prior{int(round(float(pr) * 100)):02d}")


if __name__ == "__main__":
    main()
