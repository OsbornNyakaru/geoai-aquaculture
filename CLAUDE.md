# CLAUDE.md — agent entry point

GeoAI Aquaculture Pond Identification (Zindi / FAO / ITU): is a 10 m Sentinel-1/-2 patch an aquaculture pond?
Train: 1,821 rows, 12 full months. Test: 1,030 rows, only 4-6 consecutive months visible. Metric
`0.6·F1 + 0.4·ROC-AUC`, F1 at a fixed 0.5 cut (choosing a threshold is forbidden).
**Closed 16 Aug 2026.** We finished ~120th (private 0.9107). This repository is the post-competition practice loop:
reproduce the 3rd-place solution (private 0.947), then improve on it with late submissions.

## Read in this order

1. `archive/POST_MORTEM.md`: what we got wrong in the competition, measured.
2. `experiments/RESULTS.md`: the ledger (runs and Zindi scores).
3. `autoresearch/experiment.py`: the pipeline and its config.

## Files

- `autoresearch/prepare.py`: FROZEN. Data loading, pinned folds, the metric, submission build and checks.
- `autoresearch/experiment.py`: the one file an experiment edits. `DEFAULT_CONFIG` = the 3rd-place recipe.
- `autoresearch/remote.py`: runs one spec phase by phase (base, pl_r1..3) and seed by seed, resumable.
- `notebooks/molab_runner.py`: the molab notebook (synced from this public repo). It runs queued specs.
- `scripts/exp.py`: `new`, `submit`, `pull`, `status`, `lb`.
- `archive/`: the competition-era Transformer pipeline, research rounds and reports. Read-only.

## The loop

1. `exp.py new --from best --set <key>=<value> --desc "..."`: one change per experiment.
2. Commit and `git push` (molab runs the code at the spec's commit, never local edits).
3. `exp.py submit`: queues the spec in the private HF dataset `nyakaruosborn/geoai-aquaculture-runs`.
4. On molab: open `notebooks/molab_runner.py`, paste an HF token, **Run queue**.
5. `exp.py pull`: submissions land in `submissions/EXP-NNN_<phase>.csv`.
6. The user uploads to Zindi; `exp.py lb --file ... --public ... --private ...` records the score and adopts.

## Principles

1. **The Zindi score decides; OOF never does.** Local CV ranked pipelines backwards for us and for 3rd place.
   OOF, EM prior and positive count are logged as smoke tests only.
2. **One change per experiment**, against the best. `exp.py new` refuses more unless `--allow-multi`.
3. **A gain under 0.005 is not a gain** until it holds across a re-run at other seeds (POST_MORTEM §2.2).
4. **Legal output only**: `TargetF1 = TargetRAUC >= 0.5`, enforced by `prepare.validate_submission`.
5. **Do not edit `prepare.py`.** Its SHA-256 is pinned in `notebooks/molab_runner.py`; the runner refuses a mismatch.

## Environment

- Python: `.venv/bin/python` (`make setup`). Heavy runs go to molab, not this laptop (the user's choice);
  `make smoke` (~4 min, tiny models) is the only local training.
- **The GitHub repository is public** (`OsbornNyakaru/geoai-aquaculture`, so molab can sync). Competition data and
  predictions stay out of git and live in private Hugging Face datasets (`geoai-aquaculture-data`, `-runs`).
  `make hooks` installs `scripts/check_public.py` as a pre-commit check; run it in every fresh clone.
- `data/raw/Test.csv` was rebuilt from the competition-era cache (`archive/` pipeline) and checked exact.
- Report times in EAT (UTC+3). Commits carry no AI attribution lines. No tokens or keys in code.
