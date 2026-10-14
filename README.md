# GeoAI Aquaculture Pond Identification — practice loop

Post-competition work on the Zindi / FAO / ITU **GeoAI Aquaculture Pond Identification Challenge**: decide whether a
10 m × 10 m patch is an aquaculture pond from 12 monthly Sentinel-1 (radar) and Sentinel-2 (optical) composites,
when the test patches show only 4–6 consecutive months.

- In the competition we finished ~120th of 500 (private 0.9107) with a from-scratch temporal Transformer.
  What went wrong is in [`archive/POST_MORTEM.md`](archive/POST_MORTEM.md).
- The 3rd-place solution ([zinmori](https://github.com/zinmori/geoai-aquaculture-pond-identification), private 0.947)
  used gradient-boosted trees, chlorophyll/red-edge features, an EM estimate of the test pond share and three rounds
  of pseudo-labelling. `autoresearch/experiment.py` reproduces it as the default config; experiments change one
  thing at a time on top of it.

## How a run works

```
Mac                                   GitHub (public)            Hugging Face (private)         molab
exp.py new  -> spec JSON
git push  --------------------------> code at commit SHA
exp.py submit ---------------------------------------------------> queue/EXP-NNN.json
                                                                                       runner: Run queue
                                      code @ SHA  <------------------------------------ downloads code + data
                                                                   runs/EXP-NNN/...  <-- uploads each seed
exp.py pull  <---------------------------------------------------- submissions
upload to Zindi, exp.py lb --file ... --public ... --private ...
```

The runner notebook (`notebooks/molab_runner.py`) only moves files; the code that runs is always the pushed commit,
and it refuses to run if the frozen harness (`autoresearch/prepare.py`) has changed.

## Quick start

```bash
make setup          # .venv with pinned packages
make hooks          # pre-commit check: no competition data in this public repo
make test
make push-data      # once: Train/Test/SampleSubmission + pinned folds -> private HF dataset
make status
```

Competition data is not in this repository (Zindi's terms); place the CSVs in `data/raw/`.
