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


def _load_encoder(freeze: bool = True):
    """Import the vendored Presto and return the encoder + torch device.

    `freeze=False` leaves requires_grad on, for the fine-tuning path. A FRESH encoder must be
    loaded for every fold in that mode — otherwise fold k inherits fold k-1's weights, which have
    already seen fold k's validation rows, and the OOF vector is contaminated.
    """
    import torch
    vendor = Path(__file__).resolve().parent / "vendor"
    if not (vendor / "presto" / "presto_core.py").exists():
        raise SystemExit("vendor/presto not found — run `python tools/fetch_presto.py` first.")
    sys.path.insert(0, str(vendor))
    import importlib
    core = importlib.import_module("presto.presto_core")
    enc = core.Presto.load_pretrained().encoder
    if freeze:
        for p in enc.parameters():
            p.requires_grad_(False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = sum(p.numel() for p in enc.parameters())
    log.info("Presto encoder loaded: %s params on %s (%s)", f"{n:,}", dev,
             "frozen" if freeze else "TRAINABLE")
    return (enc.to(dev).eval() if freeze else enc.to(dev)), dev


def _presto_tensors(cube, sar_units, month_mode, device):
    """Build Presto's five input tensors once, on device. Mirrors src.presto_features.embed."""
    import torch
    from src.presto_features import to_presto

    x, mask = to_presto(cube, sar_units=sar_units)
    n, t, _ = x.shape
    if month_mode == "const":
        mo = np.zeros(n, dtype=np.int64)
    else:
        mo = np.argmax(mask[:, :, 0] == 0, axis=1).astype(np.int64)
    return {
        "x": torch.from_numpy(x).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "dynamic_world": torch.full((n, t), 9, dtype=torch.long, device=device),
        "latlons": torch.zeros(n, 2, dtype=torch.float32, device=device),
        "month": torch.from_numpy(mo).to(device),
    }


def _fwd(enc, T, idx, head=None):
    """Encoder forward on a row subset, optionally through a linear head."""
    z = enc(T["x"][idx], dynamic_world=T["dynamic_world"][idx], mask=T["mask"][idx],
            latlons=T["latlons"][idx], month=T["month"][idx], eval_task=True)
    return z if head is None else head(z).squeeze(-1)


def finetune_fold(T_tr, T_te, ytr, m_tr, args, cfg, seed, fold):
    """End-to-end fine-tune of the Presto encoder + a linear head, for ONE fold.

    WHY THIS IS THE EXPERIMENT AND THE FROZEN RUN IS NOT. Frozen Presto fits ~129 parameters, so it
    could never contradict this project's measured "added capacity hurts" law — its capacity is
    amortized over Presto's global pretraining corpus, not fitted to our 1,817 shifted rows.
    Fine-tuning moves ~404k parameters ONTO those rows, which is precisely the condition the law was
    measured under. So this run is a genuine test of the law, and the law predicts it loses.

    Deliberately plain: fixed epoch count, no early stopping, no LR schedule search, no
    hyperparameter sweep. Early stopping would need a selection signal, and every offline signal we
    have is either retired or blind (see LB_LOG iter46). A fixed, pre-committed budget is the only
    honest option, and it keeps the arm to ONE variable versus the frozen run.

    Plain BCE, no class weighting: a class-prior reweight is an additive logit shift and the Platt
    refit downstream annihilates it exactly, so it could not change the submission. Omitting it
    keeps this comparable to the frozen path, which uses class_weight='balanced', WITHOUT making
    the comparison two-variable.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed * 1000 + fold)
    enc, dev = _load_encoder(freeze=False)        # FRESH weights per fold — see _load_encoder
    head = nn.Linear(128, 1).to(dev)
    opt = torch.optim.AdamW(
        [{"params": enc.parameters(), "lr": args.ft_lr_encoder},
         {"params": head.parameters(), "lr": args.ft_lr_head}], weight_decay=args.ft_wd)
    lossf = nn.BCEWithLogitsLoss()

    idx_tr = np.flatnonzero(m_tr)
    yt = torch.from_numpy(ytr[m_tr].astype(np.float32)).to(dev)
    g = np.random.default_rng(seed * 1000 + fold)

    enc.train(); head.train()
    for ep in range(args.ft_epochs):
        perm = g.permutation(len(idx_tr))
        tot = 0.0
        for b in range(0, len(perm), args.ft_batch):
            sel = perm[b:b + args.ft_batch]
            opt.zero_grad(set_to_none=True)
            logit = _fwd(enc, T_tr, idx_tr[sel], head)
            loss = lossf(logit, yt[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(head.parameters()), args.ft_clip)
            opt.step()
            tot += float(loss.detach()) * len(sel)
        log.info("    fold %d epoch %d/%d  train BCE %.5f", fold, ep + 1, args.ft_epochs,
                 tot / len(idx_tr))

    enc.eval(); head.eval()

    def predict(T, n):
        out = []
        with torch.no_grad():
            for i in range(0, n, 256):
                ii = np.arange(i, min(i + 256, n))
                out.append(torch.sigmoid(_fwd(enc, T, ii, head)).cpu().numpy())
        return np.concatenate(out)

    return predict


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
    ap.add_argument("--finetune", action="store_true",
                    help="unfreeze the encoder and train end-to-end (the untried arm)")
    ap.add_argument("--ft-epochs", type=int, default=8)
    ap.add_argument("--ft-batch", type=int, default=64)
    ap.add_argument("--ft-lr-encoder", type=float, default=1e-4)
    ap.add_argument("--ft-lr-head", type=float, default=1e-3)
    ap.add_argument("--ft-wd", type=float, default=0.01)
    ap.add_argument("--ft-clip", type=float, default=1.0)
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

    # TRAIN/TEST SYMMETRY: push train rows through the SAME window sampler used everywhere else,
    # so the encoder sees 4-6 observed months on both sides. Feeding full 12-month train against
    # 4-6-month test would be a domain gap we manufactured ourselves.
    K = int(cfg["seq"]["K"])
    tr_cube, tr_owner = _mask_views(train_cube, np.arange(len(y)), schema, wd, cfg,
                                    K, cfg["seed"], oof=False)
    log.info("masked train views: %s (K=%d per row)", tr_cube.shape, K)

    ytr = y[tr_owner]
    oof = np.zeros(len(y))
    test_per_fold = []
    folds = list(StratifiedKFold(cfg["seq"]["n_splits"], shuffle=True,
                                 random_state=cfg["seed"]).split(np.zeros(len(y)), y))

    if args.finetune:
        # ---- THE UNTRIED ARM: end-to-end fine-tuning. ----
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        T_tr = _presto_tensors(tr_cube, schema.sar_units, args.month_mode, dev)
        T_te = _presto_tensors(test_cube, schema.sar_units, args.month_mode, dev)
        log.info("FINE-TUNE: %d epochs, batch %d, lr enc %.1e head %.1e, wd %.3g, clip %.2g",
                 args.ft_epochs, args.ft_batch, args.ft_lr_encoder, args.ft_lr_head,
                 args.ft_wd, args.ft_clip)
        adv = float("nan")   # the frozen-embedding adv-AUC is not defined for a per-fold encoder
        for fold, (tr, va) in enumerate(folds):
            m_tr, m_va = np.isin(tr_owner, tr), np.isin(tr_owner, va)
            predict = finetune_fold(T_tr, T_te, ytr, m_tr, args, cfg, cfg["seed"], fold)
            pv_all = predict(T_tr, len(tr_owner))
            for i in va:                                  # average this row's K views
                oof[i] = pv_all[m_va & (tr_owner == i)].mean()
            test_per_fold.append(predict(T_te, len(test_cube)).astype(np.float32))
            log.info("  fold %d: combined@0.5=%.5f", fold,
                     combined_score(f1_at(y[va], oof[va], 0.5), roc_auc(y[va], oof[va])))
        n_fitted = 404_160 + 129        # encoder (verified by tools/fetch_presto.py) + linear head
    else:
        # ---- the iter17 arm: frozen encoder, ~129-parameter logistic head. ----
        enc, dev = _load_encoder(freeze=True)
        Xtr = embed(tr_cube, enc, dev, sar_units=schema.sar_units, month_mode=args.month_mode)
        Xte = embed(test_cube, enc, dev, sar_units=schema.sar_units, month_mode=args.month_mode)
        log.info("embeddings: train %s test %s", Xtr.shape, Xte.shape)

        # ---- the iter17 GO/NO-GO. ⚠️ RETIRED AS A SELECTION CRITERION (round 18: adversarial AUC
        #      is "DEAD ... BACKWARDS" as a gate). Printed as a DESCRIPTIVE statistic only; it must
        #      not be used to kill or fund this lane, which is exactly what iter17 did with it.
        adv = adversarial_auc(Xtr, Xte, cfg["seed"])
        log.info("adversarial AUC on Presto embeddings = %.4f  (DESCRIPTIVE ONLY - retired as a "
                 "selection criterion at round 18; do NOT gate on it)", adv)

        for fold, (tr, va) in enumerate(folds):
            m_tr, m_va = np.isin(tr_owner, tr), np.isin(tr_owner, va)
            head = make_pipeline(StandardScaler(),
                                 LogisticRegression(C=args.C, max_iter=2000,
                                                    class_weight="balanced"))
            head.fit(Xtr[m_tr], ytr[m_tr])
            pv = head.predict_proba(Xtr[m_va])[:, 1]
            for i in va:                                  # average this row's K views
                oof[i] = pv[tr_owner[m_va] == i].mean()
            test_per_fold.append(head.predict_proba(Xte)[:, 1].astype(np.float32))
            log.info("  fold %d: combined@0.5=%.5f", fold,
                     combined_score(f1_at(y[va], oof[va], 0.5), roc_auc(y[va], oof[va])))
        n_fitted = Xtr.shape[1] + 1

    p_test_raw = np.mean(test_per_fold, axis=0)
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
