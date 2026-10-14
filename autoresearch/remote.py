"""remote.py — runs one queued spec, phase by phase and seed by seed. Called by notebooks/molab_runner.py from the
repository as it was at the spec's git commit, so a run always uses the code the spec was written against.

    store   where results go: exists(name), get(name) -> local path, put({name: bytes | str | DataFrame})
            (names are relative to runs/<EXP>/ in the private Hugging Face runs repo)
    log     callable(str)

Each finished seed is uploaded at once as <phase>/seed<S>.npz, so a run that dies resumes at the next seed. After a
phase: <phase>.csv (the submission), <phase>_oof.npy (seed-averaged OOF) and <phase>.json (EM prior, positives, CV).
Pseudo-label phases read the previous phase's final probabilities, which are deterministic, so resuming is exact.

Locally: `python autoresearch/remote.py EXP-001 --seeds 1 --phases 1` runs the same code on this machine into
experiments/runs/<EXP>-local/ (a smoke test; not recorded).
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import experiment as E  # noqa: E402
import prepare as P  # noqa: E402


def _npz(r: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, oof=r["oof"], test=r["test"], weights=np.array(r["weights"]), cv=np.array(r["cv"]),
                        seed=np.array(r["seed"]))
    return buf.getvalue()


def _load_npz(path: str) -> dict:
    z = np.load(path)
    return {"seed": int(z["seed"]), "oof": z["oof"], "test": z["test"], "weights": list(z["weights"]), "cv": float(z["cv"])}


def run_spec(spec: dict, store, log=print, n_jobs: int | None = None, max_minutes: float | None = None, **_) -> dict:
    from joblib import Parallel, delayed

    t0 = time.time()
    deadline = t0 + 60 * float(max_minutes or spec.get("max_minutes") or 10_000)
    cfg = spec["config"]
    train, test = P.load()
    fold = P.folds(train)
    y = train[P.TARGET].to_numpy()
    prior0 = float(y.mean())
    jobs = max(1, min(int(n_jobs or os.cpu_count() or 1), len(cfg["seeds"])))
    threads = max(1, (os.cpu_count() or 1) // jobs)
    log(f"{len(cfg['seeds'])} seeds, {jobs} in parallel x {threads} threads; building masked views")
    views, X_test = E.build_views(cfg, train, test)
    log(f"{X_test.shape[1]} features")

    p_prev, summary = None, {}
    for name, thr in E.phases(cfg):
        if store.exists(f"{name}.json"):
            p_prev = pd.read_csv(store.get(f"{name}.csv"))["TargetRAUC"].to_numpy()
            summary[name] = json.loads(Path(store.get(f"{name}.json")).read_text())
            log(f"{name}: already done ({summary[name]['positives']} positives), skipping")
            continue
        X_ps, y_ps = (None, None) if thr is None else E.pseudo_set(X_test, p_prev, *thr)
        if thr is not None:
            log(f"{name}: {int(y_ps.sum())} pseudo-positives, {int(len(y_ps) - y_ps.sum())} pseudo-negatives")
        done = {sd: _load_npz(store.get(f"{name}/seed{sd}.npz")) for sd in cfg["seeds"] if store.exists(f"{name}/seed{sd}.npz")}
        todo = [sd for sd in cfg["seeds"] if sd not in done]
        if todo and time.time() > deadline:
            log(f"{name}: time limit reached")
            break
        tp = time.time()
        gen = Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(E.run_seed)(cfg, sd, views[sd], X_test, y, fold, threads, X_ps, y_ps) for sd in todo)
        for r in gen:
            store.put({f"{name}/seed{r['seed']}.npz": _npz(r)})
            done[r["seed"]] = r
            log(f"{name}: seed {r['seed']} done (OOF {r['cv']:.4f}; weights "
                f"{', '.join(f'{w:.2f}' for w in r['weights'])}) [{len(done)}/{len(cfg['seeds'])}]")
        results = [done[sd] for sd in cfg["seeds"]]
        p, info = E.finish_phase(cfg, results, prior0)
        oof = np.mean([r["oof"] for r in results], axis=0)
        info.update({"oof": P.score(y, oof), "minutes": round((time.time() - tp) / 60, 1)})
        sub = P.build_submission(test, p)
        P.validate_submission(sub)
        store.put({f"{name}.csv": sub, f"{name}_oof.npy": _npy(oof), f"{name}.json": json.dumps(info, indent=2)})
        summary[name], p_prev = info, p
        log(f"{name}: EM prior {info['em_prior']}, {info['positives']} positives, OOF {info['oof']['score']:.4f} "
            f"in {info['minutes']} min")
    finished = [n for n, _ in E.phases(cfg) if n in summary]
    return {"phases": summary, "status": "ok" if len(finished) == len(E.phases(cfg)) else "incomplete",
            "seconds": round(time.time() - t0, 1)}


def _npy(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a)
    return buf.getvalue()


class LocalStore:
    """Writes into a local folder (dry runs)."""

    def __init__(self, folder: Path):
        self.d = folder
        self.d.mkdir(parents=True, exist_ok=True)

    def exists(self, name):
        return (self.d / name).exists()

    def get(self, name):
        return str(self.d / name)

    def put(self, files: dict):
        for name, v in files.items():
            (self.d / name).parent.mkdir(parents=True, exist_ok=True)
            if isinstance(v, pd.DataFrame):
                v.to_csv(self.d / name, index=False)
            elif isinstance(v, bytes):
                (self.d / name).write_bytes(v)
            else:
                (self.d / name).write_text(v)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--seeds", type=int, help="keep only the first N seeds (smoke test)")
    ap.add_argument("--phases", type=int, help="keep only the first N phases (1 = base only)")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--set", action="append", default=[], help="config override key=value for a quick check, "
                    "e.g. params.lgb.n_estimators=50")
    a = ap.parse_args()
    root = HERE.parent
    s = json.loads((root / "experiments" / "specs" / f"{a.exp}.json").read_text())
    for kv in a.set:
        k, _, v = kv.partition("=")
        *path, last = k.split(".")
        node = s["config"]
        for p in path:
            node = node[p]
        node[last] = json.loads(v)
    if a.seeds:
        s["config"]["seeds"] = s["config"]["seeds"][: a.seeds]
    if a.phases:
        s["config"]["pl"] = s["config"]["pl"][: a.phases - 1]
    out = root / "experiments" / "runs" / f"{a.exp}-local"
    print(json.dumps(run_spec(s, LocalStore(out), n_jobs=a.jobs), indent=2, default=str))
