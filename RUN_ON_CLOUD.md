# Running on Google Colab / Kaggle and submitting your first file

Two things are separate:
1. **Producing `submission.csv`** — run the pipeline (locally, Colab, or Kaggle).
2. **Submitting** — upload that CSV on the Zindi competition **Submissions** page
   (max 5/day). Zindi scores it and places you on the leaderboard. Colab/Kaggle
   are only for *producing* the file faster and reproducibly.

You need two things in the cloud: **the code** (`geoai-aquaculture-code.zip`,
already built in this folder) and **the data** (`Train.csv`, `Test.csv`,
`SampleSubmission.csv` from the Zindi Data page).

---

## Option A — Google Colab (simplest)

1. Open <https://colab.research.google.com> → New notebook.
2. **Upload the code zip and the 3 CSVs.**

   > ⚠️ If the left-sidebar file uploader throws
   > `Unexpected token 'P', "PK"... is not valid JSON`, that is a known Colab
   > widget bug (the "PK" is the zip header), not a bad file. Bypass it with a
   > code-cell upload instead:
   > ```python
   > from google.colab import files
   > files.upload()   # pick the zip + 3 CSVs
   > ```
   > or use the **Google Drive** route below (most robust — drive.google.com's
   > own uploader does not have this bug).

3. Paste these cells:

```python
# Cell 1 — unpack code + install the two libs Colab lacks
!mkdir -p /content/geoai && unzip -o -q /content/geoai-aquaculture-code.zip -d /content/geoai
!pip -q install lightgbm==4.6.0 catboost==1.2.10 pyyaml   # numpy/pandas/sklearn/xgboost preinstalled
```

```python
# Cell 2 — put the data where the pipeline expects it
!mkdir -p /content/geoai/data/raw
!cp /content/Train.csv /content/Test.csv /content/SampleSubmission.csv /content/geoai/data/raw/
```

```python
# Cell 3 — run the full pipeline (~a few minutes on Colab CPU)
%cd /content/geoai
!python run_pipeline.py --full --name colab
```

```python
# Cell 4 — download the submission
from google.colab import files
files.download('/content/geoai/submissions/submission_colab.csv')
```

4. Upload `submission_colab.csv` on the Zindi **Submissions** page.

> Tip: run **Cell 3** twice — the two `final_oof` values in the printout must be
> identical (the run is fully seeded). That is your reproducibility proof for
> Phase 2.

---

## Option B — Kaggle Notebooks

Kaggle notebooks have no internet by default, so attach code + data as
**Datasets** instead of `pip install`-ing.

1. **Create a code dataset**: Kaggle → Datasets → New Dataset → upload
   `geoai-aquaculture-code.zip`. Name it e.g. `geoai-aqua-code`.
2. **Create a data dataset**: New Dataset → upload the 3 Zindi CSVs. Name it
   e.g. `geoai-aqua-data`. *(Do not make either public — the rules allow only
   the supplied data, and keeping it private avoids sharing it.)*
3. New Notebook → **Add data** → attach both datasets. They mount read-only at
   `/kaggle/input/geoai-aqua-code/` and `/kaggle/input/geoai-aqua-data/`.
4. Enable **Settings → Accelerator: None (CPU)** (this is a tree model; no GPU
   needed) and leave internet off.
5. Paste:

```python
import subprocess, shutil, os
# copy code to a writable dir and unzip
shutil.copytree('/kaggle/input/geoai-aqua-code', '/kaggle/working/geoai', dirs_exist_ok=True)
subprocess.run('cd /kaggle/working/geoai && unzip -o -q geoai-aquaculture-code.zip', shell=True)
# lightgbm & catboost are preinstalled on Kaggle; if not: !pip install lightgbm catboost
os.makedirs('/kaggle/working/geoai/data/raw', exist_ok=True)
for f in ['Train.csv','Test.csv','SampleSubmission.csv']:
    shutil.copy(f'/kaggle/input/geoai-aqua-data/{f}', f'/kaggle/working/geoai/data/raw/{f}')
```

```python
!cd /kaggle/working/geoai && python run_pipeline.py --full --name kaggle
```

6. The file is at `/kaggle/working/geoai/submissions/submission_kaggle.csv` —
   download it from the notebook's **Output** tab and upload to Zindi.

---

## Fastest path to your FIRST submission (no cloud needed)

A full run is already producing `submissions/submission_full.csv` on your
machine. When it finishes you can submit that immediately:

1. Open the Zindi competition → **Submissions** tab.
2. Upload `geoai-aquaculture/submissions/submission_full.csv`.
3. Note the leaderboard score — because of the built-in train/test domain shift,
   expect it to be **below** the local OOF number (that is normal here; trust
   the LB). This first score is your baseline to improve on.

Use Colab/Kaggle after that for faster iteration.

---

## Sanity checks before uploading any submission
- Header is exactly `ID,TargetF1,TargetRAUC`.
- 1030 data rows (+1 header = 1031 lines).
- `TargetF1` is 0/1 only; `TargetRAUC` is a probability in [0,1].
The pipeline's `validate_submission()` already asserts all of this, so a file
that was written means it passed.
