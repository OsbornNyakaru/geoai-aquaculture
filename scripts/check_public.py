"""check_public.py — refuse to commit competition data to the public repository.

The GitHub repository is public (molab syncs the runner notebook from it), and Zindi's terms forbid sharing the data.
This blocks:
  * data files (.csv, .npz, .npy, .parquet, .zip, .pdf) and anything under data/, experiments/runs/, submissions/;
  * any text that quotes a Train or Test row ID (every row of a prediction file carries one).

    python scripts/check_public.py            # the files staged for commit (the pre-commit hook runs this)
    python scripts/check_public.py --all      # every tracked file
Install the hook once with `make hooks`. Without data/raw/Train.csv it only checks file types.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TEXT = {".md", ".py", ".json", ".jsonl", ".txt", ".tsv", ".toml", ".yaml", ".yml", ".cfg", ".html", ".ipynb", ".sh", ""}
BANNED_SUFFIX = {".csv", ".npz", ".npy", ".parquet", ".zip", ".pdf", ".pt", ".pkl"}
BANNED_DIRS = ("data/", "experiments/runs/", "submissions/", "vendor/")
ID_RE = re.compile(r"ID_[A-Za-z0-9]{6,}")


def git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [x for x in out.split("\0") if x]


def competition_ids() -> set[str]:
    import csv

    ids: set[str] = set()
    for name in ("Train.csv", "Test.csv"):
        if (RAW / name).exists():
            with open(RAW / name, encoding="utf-8-sig") as f:
                ids.update(r["ID"] for r in csv.DictReader(f))
    return ids


def problems(path: str, text: str | None, ids: set[str]) -> list[str]:
    out = []
    if path.startswith(BANNED_DIRS):
        out.append("lives in a data folder")
    if Path(path).suffix.lower() in BANNED_SUFFIX:
        out.append(f"{Path(path).suffix} files may hold competition data")
    if text and ids:
        hits = set(ID_RE.findall(text)) & ids
        if hits:
            out.append(f"quotes {len(hits)} competition row ID(s), e.g. {sorted(hits)[0][:6]}...")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        files = git("ls-files", "-z")
        read = lambda f: (ROOT / f).read_text(errors="ignore") if (ROOT / f).exists() else ""  # noqa: E731
    else:
        files = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
        read = lambda f: subprocess.run(["git", "show", f":{f}"], cwd=ROOT, capture_output=True).stdout.decode(errors="ignore")  # noqa: E731
    ids = competition_ids()
    if not ids:
        print("check_public: data/raw/Train.csv not found; checking file types only", file=sys.stderr)
    bad = 0
    for f in files:
        text = read(f) if Path(f).suffix.lower() in TEXT else None
        for msg in problems(f, text, ids):
            print(f"BLOCKED {f}: {msg}", file=sys.stderr)
            bad += 1
    if bad:
        print(f"check_public: {bad} problem(s). The repository is public; take competition data out and retry.", file=sys.stderr)
        return 1
    print(f"check_public: {len(files)} file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
