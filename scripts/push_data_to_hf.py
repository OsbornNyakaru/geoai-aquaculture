"""Upload the competition files to a PRIVATE Hugging Face dataset, for the molab runner.

The GitHub repository is public and Zindi's terms forbid sharing the data, so molab downloads it from this private
dataset instead (notebooks/molab_runner.py, with a token pasted in the session). Refuses if the dataset is public.
Also writes data/raw/folds.csv (the pinned folds) with the local, pinned scikit-learn.

    .venv/bin/python scripts/push_data_to_hf.py            # -> <you>/geoai-aquaculture-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FILES = ("Train.csv", "Test.csv", "SampleSubmission.csv", "folds.csv")
sys.path.insert(0, str(ROOT / "autoresearch"))
import prepare as P  # noqa: E402

CARD = """---
license: other
pretty_name: geoai-aquaculture competition data (private mirror)
---

Private mirror of the Zindi GeoAI Aquaculture Pond Identification Challenge files, for the molab runner only.
Do **not** make this repository public: Zindi's terms forbid redistributing the data. SHA-256 in `manifest.json`.
`folds.csv` holds the pinned cross-validation fold of every training row (autoresearch/prepare.py).
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_folds() -> None:
    train, _ = P.load()
    fold = P.make_folds(train)
    if P.fold_hash(train[P.ID_COL], fold) != P.FOLD_HASH:
        raise SystemExit(f"local folds {P.fold_hash(train[P.ID_COL], fold)} != frozen {P.FOLD_HASH}: run in the pinned .venv")
    pd.DataFrame({P.ID_COL: train[P.ID_COL], "fold": fold}).to_csv(RAW / "folds.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--only", nargs="*", help="upload only these files")
    a = ap.parse_args()
    write_folds()
    missing = [f for f in FILES if not (RAW / f).exists()]
    if missing:
        print(f"missing in {RAW}: {missing}", file=sys.stderr)
        return 1
    api = HfApi()
    repo = a.repo or f"{api.whoami()['name']}/geoai-aquaculture-data"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        print(f"refusing to upload: {repo} is PUBLIC", file=sys.stderr)
        return 1
    stage = ROOT / "data" / "hf_card"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "manifest.json").write_text(json.dumps({f: sha256(RAW / f) for f in FILES}, indent=2))
    (stage / "README.md").write_text(CARD)
    for f in (a.only or FILES):
        api.upload_file(path_or_fileobj=str(RAW / f), path_in_repo=f, repo_id=repo, repo_type="dataset", commit_message=f"upload {f}")
    for f in ("manifest.json", "README.md"):
        api.upload_file(path_or_fileobj=str(stage / f), path_in_repo=f, repo_id=repo, repo_type="dataset", commit_message=f"upload {f}")
    remote = {s.rfilename: s for s in api.dataset_info(repo, files_metadata=True).siblings}
    bad = [f for f in FILES if f not in remote or remote[f].size != (RAW / f).stat().st_size]
    print(f"{repo} (private): {sorted(remote)}")
    if bad:
        print(f"size mismatch or missing after upload: {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
