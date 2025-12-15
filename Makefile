CONFIG ?= configs/config.yaml
PYTHON = .venv/bin/python

.PHONY: env data train

env:
	uv sync

split:
	$(PYTHON) scripts/split_dataset.py --config $(CONFIG)

preprocess: split
	$(PYTHON) scripts/data_prep.py --config $(CONFIG)

data: preprocess

train: data
	$(PYTHON) scripts/train.py --config $(CONFIG)
