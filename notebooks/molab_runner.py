# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.16",
#     "numpy==1.26.4",
#     "pandas==2.2.3",
#     "scikit-learn==1.9.0",
#     "scipy==1.14.1",
#     "lightgbm==4.7.0",
#     "xgboost==3.3.0",
#     "catboost==1.2.10",
#     "joblib>=1.4",
#     "huggingface_hub>=0.35",
# ]
# ///
"""molab_runner.py — the experiment runner for the GeoAI aquaculture loop. You never edit experiments here.

  1. Locally: `exp.py new ...`, commit, `git push`, then `exp.py submit` puts the spec in the private Hugging Face
     dataset nyakaruosborn/geoai-aquaculture-runs under queue/.
  2. Here (molab, CPU is enough): paste a Hugging Face token, press **Run queue**. For each pending spec the runner
     downloads the GitHub repository at the spec's commit, refuses to run if autoresearch/prepare.py is not the frozen
     one, and calls autoresearch/remote.py, which trains seed by seed and uploads each finished seed.
  3. Locally: `exp.py pull` fetches the submissions into submissions/ and updates the ledger.

So the code that runs is always the code pushed to GitHub; this notebook only moves files. It never writes to GitHub.
Competition data comes from the private dataset nyakaruosborn/geoai-aquaculture-data.
"""

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", app_title="GeoAI aquaculture — runner")


@app.cell
def _():
    import hashlib
    import importlib
    import io
    import json
    import os
    import sys
    import tarfile
    import threading
    import time
    import traceback
    import urllib.request
    import uuid
    from datetime import datetime, timezone
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return (Path, datetime, hashlib, importlib, io, json, mo, os, pd, sys, tarfile, threading, time, timezone,
            traceback, urllib, uuid)


@app.cell
def _(importlib, mo, sys):
    # molab does not install the header's dependencies into the kernel (2026-10-14: no lightgbm, no catboost,
    # numpy 2.4.6 / pandas 3.0.3), so install whatever is missing. numpy and pandas are left as molab has them:
    # swapping them under a running kernel breaks pickling to the worker processes. Features were checked
    # identical under numpy 1.26/pandas 2.2 and numpy 2.4.6/pandas 3.0.3.
    import shutil
    import subprocess
    from importlib.util import find_spec as _find_spec

    REQUIRED = {"lightgbm": "lightgbm==4.7.0", "catboost": "catboost==1.2.10", "xgboost": "xgboost==3.3.0",
                "sklearn": "scikit-learn==1.9.0", "scipy": "scipy", "joblib": "joblib>=1.4",
                "huggingface_hub": "huggingface_hub>=0.35"}
    _missing = [spec for mod, spec in REQUIRED.items() if _find_spec(mod) is None]
    _err = ""
    if _missing:
        _cmd = ([shutil.which("uv"), "pip", "install", "--python", sys.executable, *_missing] if shutil.which("uv")
                else [sys.executable, "-m", "pip", "install", "-q", *_missing])
        _p = subprocess.run(_cmd, capture_output=True, text=True)
        _err = "" if _p.returncode == 0 else _p.stderr[-1500:]
        importlib.invalidate_caches()
    deps_versions = {}
    for _m in ("numpy", "pandas", *REQUIRED):
        try:
            deps_versions[_m] = importlib.import_module(_m).__version__
        except Exception:
            deps_versions[_m] = None
    deps_ok = all(deps_versions[m] for m in REQUIRED)
    mo.md(f"Packages: **{'ready' if deps_ok else 'MISSING'}**"
          + (f" (installed {', '.join(_missing)})" if _missing else "") + " · "
          + ", ".join(f"{k} {v}" for k, v in deps_versions.items())
          + (f"\n\n```\n{_err}\n```" if _err else ""))
    return deps_ok, deps_versions


@app.cell
def _(mo):
    mo.md("""
    # GeoAI aquaculture — experiment runner

    1. Paste a Hugging Face token with **read** on `geoai-aquaculture-data` and **write** on `geoai-aquaculture-runs`.
    2. Press **Run queue**. A full spec is 8 seeds × 4 phases; each seed-phase is ~20 CPU-minutes, so more cores
       finish sooner. Finished seeds are uploaded as they land: if the session drops, press Run queue again to resume.
    """)
    return


@app.cell
def _(mo, os):
    hf_token_input = mo.ui.text(kind="password", value="", label="Hugging Face token (this session only)", full_width=True)
    data_repo_input = mo.ui.text(value=os.environ.get("HF_DATA_REPO", "nyakaruosborn/geoai-aquaculture-data"), label="data repo", full_width=True)
    runs_repo_input = mo.ui.text(value=os.environ.get("HF_RUNS_REPO", "nyakaruosborn/geoai-aquaculture-runs"), label="runs repo", full_width=True)
    mo.vstack([hf_token_input, data_repo_input, runs_repo_input])
    return data_repo_input, hf_token_input, runs_repo_input


@app.cell
def _(data_repo_input, hf_token_input, mo, os, runs_repo_input):
    if hf_token_input.value.strip():
        os.environ["HF_TOKEN"] = hf_token_input.value.strip()
    DATA_REPO = data_repo_input.value.strip()
    RUNS_REPO = runs_repo_input.value.strip()
    GITHUB_REPO = "OsbornNyakaru/geoai-aquaculture"
    # The frozen harness: autoresearch/prepare.py must hash to this or the runner refuses.
    PREPARE_SHA256 = "d5b76b4ba743e34fe926f95ad5816722dda181c5e3ea49c639a30a5ddf0c44ea"
    RUNNER_VERSION = "2026-10-14b"

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi()
    try:
        _who = api.whoami()["name"]
        _msg = mo.md(f"Hugging Face: signed in as **{_who}** · runner **{RUNNER_VERSION}** · **{os.cpu_count()} CPU cores**")
    except Exception as _e:
        _msg = mo.callout(mo.md(f"Hugging Face: **not signed in** ({type(_e).__name__}). Paste a token above."), kind="danger")
    _msg
    return CommitOperationAdd, DATA_REPO, GITHUB_REPO, PREPARE_SHA256, RUNNER_VERSION, RUNS_REPO, api, hf_hub_download


@app.cell
def _(DATA_REPO, Path, mo):
    WORK = Path.cwd() / "runner_work"
    DATA = WORK / "data"
    RAW = DATA / "raw"
    RAW.mkdir(parents=True, exist_ok=True)
    _err = ""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=str(RAW),
                          allow_patterns=["Train.csv", "Test.csv", "SampleSubmission.csv", "folds.csv", "manifest.json"])
    except Exception as _e:
        _err = f" ({type(_e).__name__}: {_e})"
    data_ok = all((RAW / f).exists() for f in ("Train.csv", "Test.csv", "SampleSubmission.csv", "folds.csv"))
    mo.md(f"Data: **{'ready' if data_ok else 'missing'}** in `{RAW}`{_err}")
    return DATA, WORK, data_ok


@app.cell
def _(
    CommitOperationAdd, DATA, GITHUB_REPO, PREPARE_SHA256, RUNS_REPO, WORK, api, hashlib, hf_hub_download, importlib,
    io, json, pd, sys, tarfile, time, urllib,
):
    def fetch_code(sha: str):
        """The repository at `sha` (GitHub's immutable tarball), with data/ linked to the downloaded data."""
        dest = WORK / "code" / sha
        if not (dest / ".ok").exists():
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(f"https://codeload.github.com/{GITHUB_REPO}/tar.gz/{sha}", timeout=120) as r:
                        blob = r.read()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(5 * (attempt + 1))
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
                tf.extractall(dest, filter="data")
            (dest / ".ok").write_text("ok")
        root = next(p for p in dest.iterdir() if p.is_dir())
        if not (root / "data").exists():
            (root / "data").symlink_to(DATA, target_is_directory=True)
        return root

    def check_harness(root):
        h = hashlib.sha256((root / "autoresearch" / "prepare.py").read_bytes()).hexdigest()
        if h != PREPARE_SHA256:
            raise RuntimeError(f"prepare.py hash {h[:12]} != frozen {PREPARE_SHA256[:12]}: harness changed, refusing to run")

    def import_remote(root):
        """remote.py from this checkout; drop modules imported from an earlier checkout first."""
        for name in [m for m in sys.modules if m in {"prepare", "experiment", "remote"}]:
            del sys.modules[name]
        sys.path[:] = [p for p in sys.path if "/runner_work/code/" not in p]
        sys.path.insert(0, str(root / "autoresearch"))
        importlib.invalidate_caches()
        return importlib.import_module("remote")

    def hf_files() -> list[str]:
        return api.list_repo_files(RUNS_REPO, repo_type="dataset")

    def hf_json(path: str) -> dict:
        return json.loads(open(hf_hub_download(RUNS_REPO, path, repo_type="dataset", force_download=True), encoding="utf-8").read())

    def hf_put(files: dict, message: str):
        ops = []
        for path, content in files.items():
            if isinstance(content, pd.DataFrame):
                content = content.to_csv(index=False).encode("utf-8")
            elif isinstance(content, str):
                content = content.encode("utf-8")
            ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=content))
        for attempt in range(4):
            try:
                return api.create_commit(RUNS_REPO, repo_type="dataset", operations=ops, commit_message=message)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))

    class Store:
        """runs/<EXP>/ in the runs repo, as remote.py expects it."""

        def __init__(self, exp: str):
            self.base = f"runs/{exp}"
            self.files = set(hf_files())

        def exists(self, name):
            return f"{self.base}/{name}" in self.files

        def get(self, name):
            return hf_hub_download(RUNS_REPO, f"{self.base}/{name}", repo_type="dataset")

        def put(self, files: dict):
            hf_put({f"{self.base}/{k}": v for k, v in files.items()}, f"{self.base}: {', '.join(files)}")
            self.files |= {f"{self.base}/{k}" for k in files}

    return Store, check_harness, fetch_code, hf_files, hf_json, hf_put, import_remote


@app.cell
def _(
    RUNNER_VERSION, Store, check_harness, datetime, fetch_code, hf_files, hf_json, hf_put, import_remote, json, mo, os,
    threading, time, timezone, traceback, uuid,
):
    SESSION = uuid.uuid4().hex[:8]
    CLAIM_EVERY, CLAIM_STALE = 240, 900   # seconds; a reopened tab must not start a second runner on the same spec

    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _versions():
        out = {}
        for m in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost"):
            try:
                out[m] = __import__(m).__version__
            except Exception:
                pass
        return out

    def pending() -> list[dict]:
        files = hf_files()
        done = {f.split("/")[1] for f in files if f.startswith("runs/") and f.endswith("/result.json")}
        specs = [hf_json(f) for f in files if f.startswith("queue/") and f.endswith(".json")]
        fresh = []
        for s in (s for s in specs if s["exp"] not in done):
            c = f"runs/{s['exp']}/claim.json"
            if c in files:
                try:
                    cl = hf_json(c)
                    if cl.get("session") != SESSION and time.time() - float(cl.get("t", 0)) < CLAIM_STALE:
                        continue
                except Exception:
                    pass
            fresh.append(s)
        return sorted(fresh, key=lambda s: s.get("submitted", ""))

    def _claim(exp):
        hf_put({f"runs/{exp}/claim.json": json.dumps({"session": SESSION, "t": time.time(), "at": _now(), "cpus": os.cpu_count()})},
               f"{exp}: claim {SESSION}")

    def keep_claim(exp):
        stop = threading.Event()

        def beat():
            while not stop.wait(CLAIM_EVERY):
                w = threading.Thread(target=lambda: _quiet(_claim, exp), daemon=True)
                w.start()
                w.join(60)

        threading.Thread(target=beat, daemon=True).start()
        return stop

    def _quiet(fn, *a):
        try:
            fn(*a)
        except Exception:
            pass

    def run_one(spec, log, n_jobs) -> dict:
        exp = spec["exp"]
        lines = []

        def _log(msg):
            line = f"[{time.strftime('%H:%M:%S')}] {exp} {msg}"
            lines.append(line)
            log(line)

        t0 = time.time()
        result = {"exp": exp, "git_sha": spec["git_sha"], "started": _now(), "cpus": os.cpu_count(), "runner": RUNNER_VERSION}
        try:
            root = fetch_code(spec["git_sha"])
            check_harness(root)
            remote = import_remote(root)
            result.update(remote.run_spec(spec, Store(exp), log=_log, n_jobs=n_jobs))
        except Exception as e:
            result["status"] = "crash"
            result["error"] = "".join(traceback.format_exception(e))[-4000:]
            _log(f"CRASH: {type(e).__name__}: {e}")
        result.update(finished=_now(), seconds=round(time.time() - t0, 1), versions=_versions())
        files = {f"runs/{exp}/log_{SESSION}.txt": "\n".join(lines[-2000:])}
        if result.get("status") in ("ok", "crash"):
            files[f"runs/{exp}/result.json"] = json.dumps(result, indent=2, default=str)
        else:
            files[f"runs/{exp}/session_{SESSION}.json"] = json.dumps(result, indent=2, default=str)
        hf_put(files, f"{exp}: {result.get('status')}")
        return result

    def run_queue(watch=True, idle_minutes=30, n_jobs=None, log=print):
        hf_put({"_runner/ping.json": json.dumps({"at": _now(), "cpus": os.cpu_count(), "version": RUNNER_VERSION})}, "runner ping")
        results, idle_since = [], time.time()
        while True:
            todo = pending()
            if not todo:
                if not watch or time.time() - idle_since > 60 * idle_minutes:
                    break
                time.sleep(60)
                continue
            spec = todo[0]
            _claim(spec["exp"])
            time.sleep(8)
            if hf_json(f"runs/{spec['exp']}/claim.json").get("session") != SESSION:
                log(f"· {spec['exp']} taken by another runner")
                continue
            log(f"▶ {spec['exp']} ({len(todo)} pending): {spec['description']}")
            beat = keep_claim(spec["exp"])
            try:
                results.append(run_one(spec, log, n_jobs))
            finally:
                beat.set()
            log(f"■ {spec['exp']}: {results[-1].get('status')}")
            if results[-1].get("status") == "incomplete":
                break
            idle_since = time.time()
        return results

    mo.md("Runner ready.")
    return (run_queue,)


@app.cell
def _(mo, os):
    run_btn = mo.ui.run_button(label="Run queue")
    watch_switch = mo.ui.switch(value=True, label="keep watching for new specs (every 60 s)")
    idle_input = mo.ui.number(start=5, stop=600, step=5, value=30, label="stop after N idle minutes")
    jobs_input = mo.ui.number(start=1, stop=64, step=1, value=os.cpu_count() or 1, label="seeds in parallel")
    mo.hstack([run_btn, watch_switch, idle_input, jobs_input], justify="start", wrap=True)
    return idle_input, jobs_input, run_btn, watch_switch


@app.cell
def _(data_ok, deps_ok, idle_input, jobs_input, mo, run_btn, run_queue, watch_switch):
    mo.stop(not run_btn.value, mo.md("_Press **Run queue** to start._"))
    mo.stop(not deps_ok, mo.callout(mo.md("Packages are missing: see the Packages line above."), kind="danger"))
    mo.stop(not data_ok, mo.callout(mo.md("Data is missing: check the token and the data repo above."), kind="danger"))
    _lines = []

    def _show(line):
        _lines.append(line)
        mo.output.replace(mo.md("```\n" + "\n".join(_lines[-40:]) + "\n```"))

    _res = run_queue(watch=watch_switch.value, idle_minutes=idle_input.value, n_jobs=int(jobs_input.value), log=_show)
    mo.md(f"Done: {len(_res)} run(s): " + ", ".join(f"`{r['exp']}` {r.get('status')}" for r in _res)
          + ". Now run `exp.py pull` on the Mac.")
    return


if __name__ == "__main__":
    app.run()
