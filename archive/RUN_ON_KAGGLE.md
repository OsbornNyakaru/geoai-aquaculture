# Running the loop on Kaggle Notebooks

Same pull-run loop as Colab (`colab_run.ipynb`), adapted to Kaggle. The differences
are all Kaggle-specific plumbing:

| Concern | Colab | Kaggle |
|---|---|---|
| Code delivery | `git pull` (private repo) | `git pull` (private repo) — needs **Internet ON** |
| Secret (`GH_PAT`) | Colab Secrets → `userdata` | **Add-ons ▸ Secrets** → `kaggle_secrets` |
| Data (3 CSVs) | Google Drive mount | **private Kaggle Dataset** attached at `/kaggle/input/…` |
| GPU | Runtime ▸ T4 GPU | **Settings ▸ Accelerator ▸ GPU T4 x2 / P100** |
| Download CSV | `files.download()` | Output tab / `FileLink` |

The experiment is still `experiments/run_current.sh` (Claude edits + pushes it); the
notebook only pulls and runs it. Zindi submission stays manual (upload CSV, max 5/day).

---

## One-time setup (≈10 min)

1. **Phone-verify your Kaggle account** (Settings) — required to enable Internet and GPU.
2. **Private data dataset:** Kaggle ▸ Datasets ▸ **New Dataset** ▸ upload `Train.csv`,
   `Test.csv`, `SampleSubmission.csv`. Keep it **Private** (competition rules allow only
   the supplied data — do not share it). Name it e.g. `geoai-aqua-data`.
3. **GitHub PAT:** github.com ▸ Settings ▸ Developer settings ▸ **Fine-grained token** →
   repo `geoai-aquaculture`, **Contents: Read-only**, short expiry. Copy it.
4. **Add the secret:** in a new Kaggle Notebook, **Add-ons ▸ Secrets ▸ Add secret**,
   label **`GH_PAT`**, value = the token, and **attach it to the notebook**.
5. **New Notebook settings (right sidebar):**
   - **Internet: On** (needed for `git pull` + pip).
   - **Accelerator: GPU T4 x2** (or P100).
   - **Add Input ▸** your `geoai-aqua-data` dataset (mounts read-only under `/kaggle/input/`).

---

## Notebook cells

```python
# Cell 1 — pull latest code from the private repo (token never printed)
import os, subprocess
from kaggle_secrets import UserSecretsClient
PAT  = UserSecretsClient().get_secret("GH_PAT")
REPO = "OsbornNyakaru/geoai-aquaculture"
DEST = "/kaggle/working/geoai"
if not os.path.exists(f"{DEST}/.git"):
    subprocess.run(["git", "clone", f"https://{PAT}@github.com/{REPO}.git", DEST], check=True)
%cd /kaggle/working/geoai
!git pull --ff-only
!git log --oneline -1        # confirms which experiment you're running
```

```python
# Cell 2 — GPU check + the two libs Kaggle may lack (torch/xgboost/sklearn preinstalled)
!nvidia-smi -L || echo "NO GPU — Settings ▸ Accelerator ▸ GPU T4 x2"
!pip -q install lightgbm==4.6.0 catboost==1.2.10 pyyaml
import torch; print("CUDA available:", torch.cuda.is_available())
```

```python
# Cell 3 — data from the attached private Kaggle Dataset
import glob, shutil, os
os.makedirs("/kaggle/working/geoai/data/raw", exist_ok=True)
hit = glob.glob("/kaggle/input/*/Train.csv")
assert hit, "Attach your geoai-aqua-data dataset via 'Add Input' first."
src = os.path.dirname(hit[0])
for f in ["Train.csv", "Test.csv", "SampleSubmission.csv"]:
    shutil.copy(f"{src}/{f}", f"/kaggle/working/geoai/data/raw/{f}")
print(os.listdir("/kaggle/working/geoai/data/raw"))
```

```python
# Cell 4 — run THIS iteration's experiment (Claude sets the commands in run_current.sh)
%cd /kaggle/working/geoai
!bash experiments/run_current.sh
```

```python
# Cell 5 — grab the newest submission(s) to upload on Zindi
from IPython.display import FileLink, display
import glob, os
for f in sorted(glob.glob("submissions/submission_*.csv"), key=os.path.getmtime)[-4:]:
    print(f); display(FileLink(f))
# Or: Save Version, then download from the notebook's Output / Data pane.
```

```python
# Cell 6 (optional) — reproducibility proof for the Phase-2 rubric
# Re-run Cell 4 once more; the two `final_oof` values must be byte-identical.
```

---

## Notes & gotchas

- **Keep the notebook private and don't share its output** — the clone URL embeds the
  token in `.git/config`. Using a short-expiry, Contents-only PAT limits the blast radius.
- **First-run dry check:** if Claude has set `run_current.sh` to a `--smoke` run, Cell 4 is
  quick and validates the token clone, dataset attach, GPU, and download before you spend a
  real GPU run.
- **GPU quota:** Kaggle gives ~30 GPU-hours/week. A full seq run is a few minutes, so this is
  plenty; turn the accelerator off for pure-GBDT experiments to save quota.
- **Reading the Step-2 blend result:** in Cell 4's log find
  `OOF rank correlation between components = …` → `< ~0.90` means the GBDT+seq blend adds
  decorrelated signal (upload the `submission_seq_gbdt*.csv` whose pos-rate is nearest 0.65);
  `≈ 1.0` means skip it. Gate every submission vs the current best LB **0.8780**.
- **Internet off?** Then `git pull` and `pip install` fail. If you must run offline, fall back
  to the zip flow in `RUN_ON_CLOUD.md` (Option B), but the git loop is preferred.
