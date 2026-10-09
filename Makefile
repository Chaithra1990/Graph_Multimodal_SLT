PYTHON ?= python
export PYTHONPATH := src

compile:
	$(PYTHON) -m compileall -q src scripts tests

prepare-phoenix:
	$(PYTHON) scripts/prepare_annotations.py --dataset phoenix2014t --config configs/phoenix2014t.yaml --data-root $(DATA_ROOT)

prepare-csldaily:
	$(PYTHON) scripts/prepare_annotations.py --dataset csldaily --config configs/csldaily.yaml --data-root $(DATA_ROOT)

test:
	$(PYTHON) -m pytest -q
