PY ?= .venv/bin/python

.PHONY: help setup test status submit pull hooks check-public push-data smoke

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-13s %s\n", $$1, $$2}'

setup:  ## Create .venv with the pinned packages.
	uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt

test:  ## Run the tests.
	$(PY) -m pytest -q tests

status:  ## List experiments, their Zindi scores and the current best.
	$(PY) scripts/exp.py status

submit:  ## Queue specs on Hugging Face for molab (commit and push first).
	$(PY) scripts/exp.py submit

pull:  ## Fetch molab results into experiments/runs/ and submissions/.
	$(PY) scripts/exp.py pull

smoke:  ## One-minute local check of the remote path with tiny models (EXP=EXP-001).
	$(PY) autoresearch/remote.py $${EXP:-EXP-001} --seeds 2 --phases 2 --jobs 2 --set params.lgb.n_estimators=50 \
	  --set params.cb.iterations=50 --set params.xgb.n_estimators=50 --set blend_step=0.25

push-data:  ## Upload the competition files and pinned folds to the PRIVATE Hugging Face data repo.
	$(PY) scripts/push_data_to_hf.py

hooks:  ## Install the pre-commit check that keeps competition data out of the public repository.
	printf '#!/bin/sh\nexec .venv/bin/python scripts/check_public.py\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

check-public:  ## Scan every tracked file for competition data.
	$(PY) scripts/check_public.py --all
