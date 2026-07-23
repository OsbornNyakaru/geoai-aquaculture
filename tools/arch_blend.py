"""Cross-architecture rank-blend — the iter18 answer to Presto's death.

WHY THIS EXISTS (distinct from the older tools/blend.py, which blended GBDT+seq via the retired
calibrate_for_f1 path). Seed-averaging (tools/seed_average.py) pooled ONE config across RNG seeds.
It bought variance reduction but NO level gain: it landed at the single-seed mean (0.8865 vs
0.8859), exactly as predicted, because seeds are 95.1% rank-correlated so only ~5% of the error is
independent and available to average away.

This tool pools DIFFERENT ARCHITECTURES instead. The top cluster -- reltime (0.8908), nope
(0.8917), l3 (0.8921), xview (0.8955) -- is statistically tied (all within one seed-swing of each
other), but the four are genuinely different inductive biases: nope removes positional encoding
entirely, l3/xview differ in cross-view invariance strength, reltime in the time reframing. IF they
make DIFFERENT errors, averaging cancels error and buys LEVEL where the seed-average could not.

THE GO/NO-GO IS FREE AND PRINTED FIRST: the pairwise rank-correlation matrix across the members.
    mean rho ~ 0.95 (as correlated as seeds) -> the blend behaves like the seed-average; no level
                                                 gain is available here. Do not spend a submission.
    mean rho < ~0.90                         -> members carry independent signal; pooling them is a
                                                 variance-AND-bias reducing move with bounded
                                                 downside (the blend lands between its members).

METHOD -- two-level rank averaging, equal weight per architecture
    Level 1: within each architecture, rank-average its seed replicates (kills seed noise).
    Level 2: rank-average the per-architecture vectors with EQUAL weight (so a config that happens
             to have 5 seed runs does not outvote one with a single run).
    Rank (not probability) averaging because the leaderboard is rank-only after the prevalence pin,
    and because per-architecture calibration drifts hard (t_star ranges 0.435 -> 0.500 across the
    cluster) while the ORDER is the quantity we actually want to pool.

USAGE
    python tools/arch_blend.py --members seq_a_reltime seq_a_nope seq_a_l3 seq_a_xview \
        --diag-extra seq_a_k4 --name champion_archblend4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import score_for_auc, target_prevalence_shift  # noqa: E402
from src.utils import get_logger, load_config, resolve_path  # noqa: E402

log = get_logger()


def _ranks(p: np.ndarray) -> np.ndarray:
    """Normalized ranks in (0,1); scale-free, so per-model calibration drift cannot dominate."""
    p = np.asarray(p, dtype=float)
    return (np.argsort(np.argsort(p)) + 0.5) / len(p)


def _seed_bundles(preds_dir: Path, member: str) -> List[Path]:
    """Every seed bundle for one architecture: preds_<m>.npz plus preds_<m>_s<N>.npz.

    `_s[0-9]*` not `_s*`, so preds_<m>_smoke.npz (or any non-seed suffix) cannot sneak in.
    """
    return sorted(preds_dir.glob(f"preds_{member}.npz")) + \
        sorted(preds_dir.glob(f"preds_{member}_s[0-9]*.npz"))


def _member_ranks(preds_dir: Path, member: str, field: str) -> Tuple[np.ndarray, np.ndarray | None, int]:
    """Level-1 pool: mean rank over a member's seeds. Returns (rank_vector, test_ids_or_None, n)."""
    files = _seed_bundles(preds_dir, member)
    if not files:
        raise SystemExit(f"no preds bundles for member {member!r} in {preds_dir}")
    mats, ids = [], None
    for f in files:
        d = np.load(f, allow_pickle=True)
        if field not in d.files:
            raise SystemExit(f"{f.name} has no field {field!r}")
        mats.append(_ranks(d[field]))
        if field == "p_test_raw":
            tid = d["test_ids"] if "test_ids" in d.files else None
            if ids is None:
                ids = tid
            elif tid is not None and not np.array_equal(ids, tid):
                raise SystemExit(f"test_ids mismatch in {f.name}; bundles not comparable")
    return np.vstack(mats).mean(axis=0), ids, len(files)


def _pairwise_rankcorr(vectors: Dict[str, np.ndarray]) -> None:
    """Print the pairwise rank-correlation matrix -- THE go/no-go for whether pooling gains level."""
    names = list(vectors)
    log.info("")
    log.info("=== CROSS-ARCHITECTURE RANK CORRELATION (the go/no-go) ===")
    log.info("    (seed noise already averaged out within each member; this is architecture-only)")
    log.info("            %s", " ".join(f"{n.split('_')[-1]:>9}" for n in names))
    offdiag = []
    for a in names:
        row = []
        for b in names:
            va, vb = vectors[a] - vectors[a].mean(), vectors[b] - vectors[b].mean()
            r = float((va * vb).sum() / np.sqrt((va ** 2).sum() * (vb ** 2).sum()))
            row.append(r)
            if a < b:
                offdiag.append(r)
        log.info("  %-9s %s", a.split("_")[-1], " ".join(f"{x:>9.4f}" for x in row))
    if offdiag:
        mean_rho = float(np.mean(offdiag))
        log.info("")
        log.info("  mean pairwise rho = %.4f   min = %.4f   (seed baseline = 0.9511)",
                 mean_rho, float(np.min(offdiag)))
        verdict = ("POOL IT: members are meaningfully decorrelated -> level gain is available."
                   if mean_rho < 0.90 else
                   "MARGINAL: between seed-correlated and decorrelated; small gain at best."
                   if mean_rho < 0.94 else
                   "SKIP: as correlated as seeds -> the blend behaves like the seed-average, no "
                   "level gain. Pivot to the next lane.")
        log.info("  -> %s", verdict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True,
                    help="architecture variant names to pool (each may have seed replicates)")
    ap.add_argument("--diag-extra", nargs="*", default=None,
                    help="extra variants to SHOW in the correlation matrix but NOT pool "
                         "(e.g. a weaker/more-different config, to see if it is decorrelated)")
    ap.add_argument("--name", default="archblend", help="output submission/bundle name")
    ap.add_argument("--preds-dir", default=None)
    ap.add_argument("--prevalence-target", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config()
    sub_dir = resolve_path(cfg, "submissions_dir")
    preds_dir = Path(args.preds_dir) if args.preds_dir else sub_dir / "preds"
    prevalence = (args.prevalence_target if args.prevalence_target is not None
                  else float(cfg["calibration"]["prevalence_target"]))

    # ---- Level-1 pool per member, on BOTH test and oof (oof needed so the screen can score it) ----
    test_vec: Dict[str, np.ndarray] = {}
    oof_vec: Dict[str, np.ndarray] = {}
    ids = None
    for m in args.members:
        tv, tid, n = _member_ranks(preds_dir, m, "p_test_raw")
        ov, _, _ = _member_ranks(preds_dir, m, "oof_prob")
        test_vec[m], oof_vec[m] = tv, ov
        if ids is None:
            ids = tid
        log.info("member %-16s pooled over %d seed bundle(s)", m, n)

    # ---- The go/no-go: correlation across members (plus any diag-only extras) ----
    diag = dict(test_vec)
    for e in (args.diag_extra or []):
        try:
            diag[e] = _member_ranks(preds_dir, e, "p_test_raw")[0]
        except SystemExit as exc:
            log.warning("diag-extra %s skipped: %s", e, exc)
    _pairwise_rankcorr(diag)

    # ---- Level-2 pool: equal weight per architecture ----
    grand_test = np.vstack([test_vec[m] for m in args.members]).mean(axis=0)
    grand_oof = np.vstack([oof_vec[m] for m in args.members]).mean(axis=0)

    # y is identical across members (same dedup'd train set); take it from the first member's bundle.
    y = np.load(_seed_bundles(preds_dir, args.members[0])[0], allow_pickle=True)["y"]
    if ids is None:
        raise SystemExit("no test_ids found on any member bundle; cannot write a submission")

    # ---- Prevalence pin, identical operating point to every other submission ----
    p_shift, delta = target_prevalence_shift(grand_test, prevalence)
    target_f1 = (p_shift >= 0.5).astype(int)
    log.info("")
    log.info("Prevalence pin: target=%.3f delta=%.3f | blended pos-rate %.3f -> %.3f",
             prevalence, delta, float((grand_test >= 0.5).mean()), float(target_f1.mean()))

    sub = pd.DataFrame({"ID": ids, "TargetF1": target_f1, "TargetRAUC": score_for_auc(grand_test)})
    sample = pd.read_csv(resolve_path(cfg, "raw_dir") / "SampleSubmission.csv")
    order = {i: k for k, i in enumerate(sample["ID"])}
    sub = sub.sort_values("ID", key=lambda s: s.map(order)).reset_index(drop=True)
    if len(sub) != len(sample):
        raise SystemExit(f"row count {len(sub)} != SampleSubmission {len(sample)}")

    out = sub_dir / f"submission_{args.name}.csv"
    sub.to_csv(out, index=False)

    # ---- Also write a preds bundle so offline_validate.py can screen the blend like any candidate.
    # grand_test/grand_oof are already rank vectors in (0,1); they ARE valid inputs for the rank-only
    # estimators (ATC-F1 pins both sides to the same prevalence, so the representation is internally
    # consistent). No test_per_fold: DIV never cleared the gate, so nothing consumes it.
    preds_out = sub_dir / "preds" / f"preds_{args.name}.npz"
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(preds_out, oof_prob=grand_oof, y=y, p_test_raw=grand_test,
             test_ids=np.asarray(ids), model=np.array("archblend"))

    log.info("Wrote %s  (%d rows, pos-rate %.3f)", out, len(sub), float(sub["TargetF1"].mean()))
    log.info("Wrote %s", preds_out)
    log.info("")
    log.info("DECISION: read the correlation matrix above, NOT the screen -- the screen's ATC-F1 "
             "resolution (seed sd 0.0576 == +-0.0094 LB) is coarser than any ensemble gain by "
             "construction. If mean rho < ~0.90, upload submission_%s.csv: it is a variance-"
             "reduction artifact with bounded downside, the same category as the seed-average that "
             "already validated. If rho ~ 0.95, do NOT upload; pivot to pseudo-labeling or ROCKET.",
             args.name)


if __name__ == "__main__":
    main()
