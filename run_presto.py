"""Presto lane driver — frozen pretrained encoder + a ~129-parameter logistic head.

Run:
    python run_presto.py --month-mode const --seed 42 --name c_presto_const

Reuses the existing pipeline wholesale: the same `load_bundle`, the same `_mask_views` masking
sampler (so the encoder sees train and test at matched observation density), the same LEGAL
`calibrate_legal` operating point (train-only Platt + a literal 0.5 cut), and the same
preds-bundle format so the offline validator can screen this exactly like any other candidate.

The one number that decides this lane is printed early and costs nothing: the ADVERSARIAL AUC on
the embeddings. If a discriminator can separate train-embeddings from test-embeddings at ~0.5, the
frozen encoder has normalized the designed temporal shift away and this lane is worth funding
hard. If it is >0.9, Presto is *encoding* the shift, the head will latch onto it, and we will have
learned that for zero submissions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.calibration import calibrate_legal  # noqa: E402
from src.data import load_bundle  # noqa: E402
from src.presto_features import embed  # noqa: E402
from src.seq_model import _mask_views  # noqa: E402
from src.utils import (combined_score, f1_at, get_logger, load_config,  # noqa: E402
                       resolve_path, roc_auc, save_npz_atomic, set_global_seeds)

log = get_logger()


def _load_encoder():
    """Import the vendored Presto and return the frozen encoder + torch device."""
    import torch
    vendor = Path(__file__).resolve().parent / "vendor"
    if not (vendor / "presto" / "presto_core.py").exists():
        raise SystemExit("vendor/presto not found — run `python tools/fetch_presto.py` first.")
    sys.path.insert(0, str(vendor))
    import importlib
    core = importlib.import_module("presto.presto_core")
    enc = core.Presto.load_pretrained().encoder
    for p in enc.parameters():
        p.requires_grad_(False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = sum(p.numel() for p in enc.parameters())
    log.info("Presto encoder loaded: %s frozen params on %s", f"{n:,}", dev)
    return enc.to(dev).eval(), dev


def adversarial_auc(Xtr: np.ndarray, Xte: np.ndarray, seed: int) -> float:
    """Can a linear model tell train-embeddings from test-embeddings? ~0.5 = shift normalized."""
    X = np.vstack([Xtr, Xte])
    d = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    oof = np.zeros(len(d))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, d):
        clf.fit(X[tr], d[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(d, oof))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month-mode", default="const", choices=["const", "true"],
                    help="const = calendar identity DELETED (relative-time applied to Presto)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--C", type=float, default=1.0, help="logistic head regularization")
    args = ap.parse_args()

    cfg = load_config()
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    set_global_seeds(cfg["seed"])
    name = args.name or f"presto_{args.month_mode}"

    bundle = load_bundle(cfg)
    train_cube, y = bundle.train_cube, bundle.y
    test_cube, schema, wd = bundle.test_cube, bundle.schema, bundle.window_dist
    log.info("Presto lane: month_mode=%s seed=%d | train=%s test=%s sar_units=%s",
             args.month_mode, cfg["seed"], train_cube.shape, test_cube.shape, schema.sar_units)

    enc, dev = _load_encoder()

    # TRAIN/TEST SYMMETRY: push train rows through the SAME window sampler used everywhere else,
    # so the encoder sees 4-6 observed months on both sides. Feeding full 12-month train against
    # 4-6-month test would be a domain gap we manufactured ourselves.
    K = int(cfg["seq"]["K"])
    tr_cube, tr_owner = _mask_views(train_cube, np.arange(len(y)), schema, wd, cfg,
                                    K, cfg["seed"], oof=False)
    log.info("masked train views: %s (K=%d per row)", tr_cube.shape, K)

    Xtr = embed(tr_cube, enc, dev, sar_units=schema.sar_units, month_mode=args.month_mode)
    Xte = embed(test_cube, enc, dev, sar_units=schema.sar_units, month_mode=args.month_mode)
    log.info("embeddings: train %s test %s", Xtr.shape, Xte.shape)

    # ---- THE GO/NO-GO, and it costs nothing ----
    adv = adversarial_auc(Xtr, Xte, cfg["seed"])
    verdict = ("SHIFT NORMALIZED AWAY - fund this lane" if adv < 0.65 else
               "partial - the head may still latch on" if adv < 0.90 else
               "PRESTO IS ENCODING THE SHIFT - expect this lane to fail")
    log.info("ADVERSARIAL AUC on Presto embeddings = %.4f  -> %s", adv, verdict)

    # ---- 5-fold CV on the head, grouped by owner row so a row's K views never straddle ----
    ytr = y[tr_owner]
    oof = np.zeros(len(y))
    test_per_fold = []
    for fold, (tr, va) in enumerate(
            StratifiedKFold(cfg["seq"]["n_splits"], shuffle=True,
                            random_state=cfg["seed"]).split(np.zeros(len(y)), y)):
        m_tr = np.isin(tr_owner, tr)
        m_va = np.isin(tr_owner, va)
        head = make_pipeline(StandardScaler(),
                             LogisticRegression(C=args.C, max_iter=2000,
                                                class_weight="balanced"))
        head.fit(Xtr[m_tr], ytr[m_tr])
        pv = head.predict_proba(Xtr[m_va])[:, 1]
        for pos, i in enumerate(va):                     # average this row's K views
            oof[i] = pv[tr_owner[m_va] == i].mean()
        pf = head.predict_proba(Xte)[:, 1]
        test_per_fold.append(pf.astype(np.float32))
        log.info("  fold %d: combined@0.5=%.5f", fold,
                 combined_score(f1_at(y[va], oof[va], 0.5), roc_auc(y[va], oof[va])))

    p_test_raw = np.mean(test_per_fold, axis=0)
    n_fitted = Xtr.shape[1] + 1
    log.info("OOF: f1@0.5=%.4f auc=%.4f combined=%.4f | FITTED params = %d (vs ~71k for the seq net)",
             f1_at(y, oof, 0.5), roc_auc(y, oof),
             combined_score(f1_at(y, oof, 0.5), roc_auc(y, oof)), n_fitted)

    # ---- Operating point: the SAME legal path every shippable submission uses. ----
    # This file predated the 2026-07-28 compliance fix and until now still emitted through
    # `target_prevalence_shift` (a threshold shift onto 0.5, forbidden) plus `score_for_auc`
    # (uniformly-spaced ranks, not probabilities). Any Presto artifact produced by the old
    # code was therefore INELIGIBLE for designation. Now routed through calibrate_legal():
    # Platt fit on TRAINING out-of-fold predictions only, then a literal 0.5 cut, with real
    # calibrated probabilities in both columns. See REPORT.md section 8.
    target_f1, target_rauc, cal_diag = calibrate_legal(y, oof, p_test_raw)

    sub = pd.DataFrame({"ID": bundle.test_ids,
                        "TargetF1": target_f1.astype(int),
                        "TargetRAUC": target_rauc})
    order = {i: k for k, i in enumerate(bundle.sample_submission["ID"])}
    sub = sub.sort_values("ID", key=lambda s: s.map(order)).reset_index(drop=True)

    out_dir = resolve_path(cfg, "submissions_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_dir / f"submission_{name}.csv", index=False)
    save_npz_atomic(out_dir / "preds" / f"preds_{name}.npz",
                    oof_prob=oof, y=y, p_test_raw=p_test_raw,
                    test_ids=np.asarray(bundle.test_ids),
                    test_per_fold=np.asarray(test_per_fold, dtype=np.float32),
                    model=np.array("presto"), adversarial_auc=np.array(adv))
    log.info("Wrote submission_%s.csv and preds_%s.npz", name, name)


if __name__ == "__main__":
    main()
