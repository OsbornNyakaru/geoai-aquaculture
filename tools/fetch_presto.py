"""Vendor Presto into vendor/presto/ — source + pretrained weights, patched to import cleanly.

WHY VENDOR RATHER THAN pip install
    `pip install -e .` on the Presto repo drags earthengine-api, openmapflow, cropharvest,
    rioxarray and google-cloud-storage, and PINS torch==2.0 / numpy==1.23.5. That is both a
    dependency nightmare on Colab and a direct conflict with Zindi's "use the most recent versions
    of packages" instruction. Presto is MIT-licensed, so copying two source files plus the 3.3 MB
    checkpoint into the repo is cleaner, fully reproducible, and needs only torch + numpy + einops.

    It also satisfies the "no custom packages in your submission notebook" rule in the way that
    matters: a reviewer can re-run everything from this repository alone, with no private wheel
    and no runtime download from us.

WHY NOT single_file_presto.py
    That file is an older revision whose `mask_tokens` asserts every row in a batch has the SAME
    number of masked tokens (`assert summed.max() == summed.min()`). Our variable 4/5/6-month
    windows with per-band cloud dropout violate that on essentially every batch. The
    `presto/presto.py` revision uses `attn_mask` and handles variable masking correctly.

USAGE
    python tools/fetch_presto.py            # clone, copy, patch, verify
    python tools/fetch_presto.py --check    # verify an existing vendored copy only
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "https://github.com/nasaharvest/presto"
# Pin the revision so a reviewer gets byte-identical source. Update deliberately, never silently.
#
# ⚠️ FIXED 2026-08-13 (iter47). This said REV = "main" while the comment above claimed it was a
# pin. It was not: `main` moves, so two reviewers running this a week apart could vendor different
# source and different weights, and the "byte-identical" promise was false. Now pinned to a commit
# SHA, with the checkpoint's SHA-256 verified after copy so a silent upstream change cannot slip
# through even if the tag were somehow reused.
REV = "11e207a668a34336ced1d8e492a1bd5849b96c4a"
CKPT_SHA256 = "559c9d76f525bdf2dd3774da8b1e7f1510769b73baacac12e9e4d8d5c67fbe65"

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "presto"

# The three imports in presto.py that reach into the heavy dataops/utils packages, and the
# literal definitions that replace them. BANDS_GROUPS_IDX drives token grouping, so it must match
# the upstream definition exactly.
_SHIM = '''
# ---- vendored shim (tools/fetch_presto.py) -------------------------------------------------
# Upstream imports these from .dataops / .utils, which pull in earthengine-api, cropharvest and
# friends. The literals below are copied from those modules so the file stands alone.
from collections import OrderedDict as _OrderedDict
from pathlib import Path as _Path
import torch as _torch

BANDS_GROUPS_IDX = _OrderedDict([
    ("S1", [0, 1]),
    ("S2_RGB", [2, 3, 4]),
    ("S2_Red_Edge", [5, 6, 7]),
    ("S2_NIR_10m", [8]),
    ("S2_NIR_20m", [9]),
    ("S2_SWIR", [10, 11]),
    ("ERA5", [12, 13]),
    ("SRTM", [14, 15]),
    ("NDVI", [16]),
])


class DynamicWorld2020_2021:
    class_amount = 9


default_model_path = _Path(__file__).resolve().parent / "default_model.pt"
device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
# ---- end vendored shim ---------------------------------------------------------------------
'''

_DROP_PREFIXES = (
    "from .dataops",
    "from .utils import",
    "from presto.dataops",
    "from presto.utils import",
)


def _patch(src: str, add_shim: bool) -> str:
    """Comment out the package-internal imports and inject the literal definitions."""
    out, injected = [], False
    for line in src.splitlines():
        if any(line.strip().startswith(p) for p in _DROP_PREFIXES):
            out.append(f"# [vendored] {line}")
            if add_shim and not injected:
                out.append(_SHIM)
                injected = True
            continue
        out.append(line)
    if add_shim and not injected:          # imports may be absent in some revisions
        out.insert(0, _SHIM)
    return "\n".join(out) + "\n"


def verify() -> None:
    """Import the vendored module and confirm the checkpoint loads with the expected shape."""
    sys.path.insert(0, str(VENDOR.parent))
    import importlib
    m = importlib.import_module("presto.presto_core")
    enc = m.Presto.load_pretrained().encoder
    n = sum(p.numel() for p in enc.parameters())
    print(f"  encoder params: {n:,}")
    if not (300_000 < n < 600_000):
        raise SystemExit(f"unexpected encoder size {n:,}; expected ~404k. Upstream may have moved.")
    print("  vendored Presto imports and loads OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify an existing copy, do not fetch")
    ap.add_argument("--rev", default=REV)
    args = ap.parse_args()

    if args.check:
        verify()
        return

    VENDOR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        # `git clone --branch` accepts tags and branch names but NOT commit SHAs, so a pinned
        # revision needs fetch-then-checkout. Falls back to --branch for a symbolic ref.
        print(f"cloning {REPO} @ {args.rev} ...")
        if len(args.rev) == 40 and all(c in "0123456789abcdef" for c in args.rev.lower()):
            subprocess.run(["git", "init", "-q", td], check=True, capture_output=True)
            subprocess.run(["git", "-C", td, "remote", "add", "origin", REPO],
                           check=True, capture_output=True)
            subprocess.run(["git", "-C", td, "fetch", "-q", "--depth", "1", "origin", args.rev],
                           check=True, capture_output=True)
            subprocess.run(["git", "-C", td, "checkout", "-q", "FETCH_HEAD"],
                           check=True, capture_output=True)
        else:
            subprocess.run(["git", "clone", "--depth", "1", "--branch", args.rev, REPO, td],
                           check=True, capture_output=True)
        src = Path(td)
        # NOTE: model.py MUST keep its upstream filename. presto.py does
        # `from .model import ...`, so the vendored copy has to be importable as
        # `presto.model` — renaming it (e.g. to presto_model.py) orphans that import.
        # presto.py itself can be renamed freely: nothing imports `.presto`.
        pairs = [
            (src / "presto" / "presto.py", VENDOR / "presto_core.py", True),
            (src / "presto" / "model.py", VENDOR / "model.py", True),
        ]
        for a, b, shim in pairs:
            if not a.exists():
                raise SystemExit(f"expected {a} in the upstream repo; layout changed.")
            b.write_text(_patch(a.read_text(encoding="utf-8"), shim), encoding="utf-8")
            print(f"  wrote {b.relative_to(ROOT)}")

        ckpt = src / "data" / "default_model.pt"
        if not ckpt.exists():
            raise SystemExit(f"checkpoint not found at {ckpt}")
        shutil.copy2(ckpt, VENDOR / "default_model.pt")
        got = hashlib.sha256((VENDOR / "default_model.pt").read_bytes()).hexdigest()
        if CKPT_SHA256 and got != CKPT_SHA256:
            raise SystemExit(
                f"checkpoint SHA-256 mismatch.\n  expected {CKPT_SHA256}\n  got      {got}\n"
                "The pretrained weights are not the ones this project was built and scored "
                "against. Refusing to continue.")
        print(f"  wrote {(VENDOR / 'default_model.pt').relative_to(ROOT)} "
              f"({ckpt.stat().st_size / 1e6:.1f} MB, sha256 verified)")

        lic = src / "LICENSE"
        if lic.exists():
            shutil.copy2(lic, VENDOR / "LICENSE")     # MIT — redistribution requires keeping it
            print("  wrote vendor/presto/LICENSE")

    (VENDOR / "__init__.py").write_text("", encoding="utf-8")
    print("verifying ...")
    verify()


if __name__ == "__main__":
    main()
