PYTHON ?= python3.12
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
TRAIN := data/raw/train/dico_nli_train_track1_participant_labeled.csv
DEV := data/raw/dev/dico_nli_dev_track1_participant_labeled.csv
DEV_REF := data/raw/dev/dico_nli_dev_track1_reference.csv

.PHONY: setup fetch-data test lint train-dev score-dev

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e '.[dev]'

fetch-data:
	$(VENV_PYTHON) scripts/fetch_official_assets.py --include-scorer

test:
	$(VENV_PYTHON) -m pytest

lint:
	$(VENV_PYTHON) -m ruff check src tests scripts

train-dev:
	$(VENV_PYTHON) -m dico_nli.train \
		--train-file $(TRAIN) \
		--dev-file $(DEV) \
		--reference-file $(DEV_REF) \
		--output-dir results/deberta-v3-base \
		--model-name microsoft/deberta-v3-base

score-dev:
	$(VENV_PYTHON) -m dico_nli.score \
		--reference-file $(DEV_REF) \
		--prediction-file $(PREDICTIONS)
