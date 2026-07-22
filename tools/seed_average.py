"""Seed-averaged submission — the answer to the iter15 finding.

WHY THIS EXISTS
---------------
On 2026-07-22 we finally measured what had been assumed for fifteen iterations: our seed-to-seed
spread. The CHAMPION CONFIGURATION, changing nothing but the RNG seed, scored

    seed 42 -> 0.8955      seed 7 -> 0.8764      delta = 0.0191

That single number voids almost the entire experiment ledger. Every verdict we recorded except the
GBDT->Transformer swap (+0.05) and per-cell detrend (-0.0514) has an effect size SMALLER than one
seed swing -- including relative-time (+0.0128), which we celebrated as the breakthrough, and the
cross-view invariance win (+0.0047) that made the current model "champion" at all.

The strategic consequence is not subtle. When run-to-run variance dominates every real effect, the
highest-value move is not another architectural probe -- it is to STOP SAMPLING ONE DRAW.
Averaging M seeds shrinks the variance of the pooled prediction by roughly 1/M, and because the
leaderboard metric is RANK-ONLY after the prevalence pin (F1 = 2*TP/(P_hat+P) is monotone in
precision@k; AUC is rank-only by definition), averaging moves the submitted ranking toward the
EXPECTED ranking of the configuration rather than the ranking of one lucky RNG draw.

EXPECT THE PUBLIC SCORE TO FALL, AND WANT IT TO.
    0.8955 is the better of two draws from a distribution with sd ~0.013. It is very likely an
    UPWARD fluctuation. A seed-averaged submission should land nearer the configuration's true
    mean -- lower on the 309-row public slice, but far more reliable on the 721-row private slice
    that actually decides the competition and that we never get to see. Chasing the public number
    here is precisely the mistake that produces a shake-up.

METHOD
    Rank-average the per-seed test probabilities, then apply the standard prevalence pin.
    Rank averaging (not probability averaging) because the metric only sees order, and because
    per-seed probability calibration drifts while order is the quantity we actually want to pool.

USAGE
    python tools/seed_average.py --variant seq_a_xview --name champion_seedavg
    python tools/seed_average.py --variant seq_a_xview --seeds 42 7 13 --name champ_3seed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import score_for_auc, target_prevalence_shift  # noqa: E402
from src.utils import get_logger, load_config, resolve_path  # noqa: E402

log = get_logger()


def _ranks(p: np.ndarray) -> np.ndarray:
    """Normalized ranks in (0,1); scale-free, so per-seed calibration drift cannot dominate."""
    p = np.asarray(p, dtype=float)
    return (np.argsort(np.argsort(p)) + 0.5) / len(p)


def collect(preds_dir: Path, variant: str, seeds: List[str] | None) -> List[Path]:
    """All seed bundles for one config: preds_<v>.npz plus preds_<v>_s<N>.npz."""
    if seeds:
        out = []
        for s in seeds:
            cand = preds_dir / (f"preds_{variant}.npz" if s in ("42", "base")
                                else f"preds_{variant}_s{s}.npz")
            if cand.exists():
                out.append(cand)
            else:
                log.warning("missing bundle for seed %s: %s", s, cand)
        return out
    return sorted(preds_dir.glob(f"preds_{variant}.npz")) + \
        sorted(preds_dir.glob(f"preds_{variant}_s[0-9]*.npz"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="seq_a_xview", help="config whose seeds to pool")
    ap.add_argument("--seeds", nargs="*", default=None,
                    help="specific seeds (e.g. 42 7 13); default = every bundle found")
    ap.add_argument("--name", default=None, help="output name; default <variant>_seedavg")
    ap.add_argument("--preds-dir", default=None)
    ap.add_argument("--prevalence-target", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config()
    sub_dir = resolve_path(cfg, "submissions_dir")
    preds_dir = Path(args.preds_dir) if args.preds_dir else sub_dir / "preds"
    prevalence = (args.prevalence_target if args.prevalence_target is not None
                  else float(cfg["calibration"]["prevalence_target"]))
    name = args.name or f"{args.variant}_seedavg"

    files = collect(preds_dir, args.variant, args.seeds)
    if len(files) < 2:
        raise SystemExit(f"need >=2 seed bundles for {args.variant}, found {len(files)}. "
                         f"Run the seed replicates first.")

    log.info("Seed-averaging %d bundles for %r:", len(files), args.variant)
    mats, ids = [], None
    for f in files:
        d = np.load(f, allow_pickle=True)
        mats.append(_ranks(d["p_test_raw"]))
        tid = d["test_ids"] if "test_ids" in d.files else None
        if ids is None:
            ids = tid
        elif tid is not None and not np.array_equal(ids, tid):
            raise SystemExit(f"test_ids mismatch in {f.name}; bundles are not comparable")
        log.info("    %s", f.name)
    if ids is None:
        raise SystemExit("bundles carry no test_ids; cannot write a submission")

    R = np.vstack(mats)

    # Diagnostic: how much do the seeds actually disagree on ORDER? This is the quantity that
    # matters, since the leaderboard sees nothing else.
    if len(R) >= 2:
        cs = []
        for i in range(len(R)):
            for j in range(i + 1, len(R)):
                a, b = R[i] - R[i].mean(), R[j] - R[j].mean()
                cs.append(float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())))
        log.info("Pairwise rank correlation between seeds: mean=%.4f min=%.4f "
                 "(1.0 would mean the seed changes nothing)", float(np.mean(cs)), float(np.min(cs)))

    p_avg = R.mean(axis=0)
    p_shift, delta = target_prevalence_shift(p_avg, prevalence)
    target_f1 = (p_shift >= 0.5).astype(int)
    log.info("Prevalence pin: target=%.3f delta=%.3f | pooled pos-rate %.3f -> %.3f",
             prevalence, delta, float((p_avg >= 0.5).mean()), float(target_f1.mean()))

    sub = pd.DataFrame({"ID": ids, "TargetF1": target_f1,
                        "TargetRAUC": score_for_auc(p_avg)})
    sample = pd.read_csv(resolve_path(cfg, "raw_dir") / "SampleSubmission.csv")
    order = {i: k for k, i in enumerate(sample["ID"])}
    sub = sub.sort_values("ID", key=lambda s: s.map(order)).reset_index(drop=True)
    if len(sub) != len(sample):
        raise SystemExit(f"row count {len(sub)} != SampleSubmission {len(sample)}")

    out = sub_dir / f"submission_{name}.csv"
    sub.to_csv(out, index=False)
    log.info("Wrote %s  (%d rows, pos-rate %.3f)", out, len(sub), float(sub["TargetF1"].mean()))
    log.info("")
    log.info("REMINDER: a LOWER public score than 0.8955 is the EXPECTED and DESIRABLE outcome "
             "here. 0.8955 is the better of two draws from a distribution with sd ~0.013 and is "
             "probably an upward fluctuation; the pooled submission trades that luck for "
             "reliability on the unseen 721-row private slice.")


if __name__ == "__main__":
    main()
