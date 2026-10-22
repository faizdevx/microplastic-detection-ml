PY ?= python

.PHONY: install data audit test features train-cls controls train-seg report-phase2

install:
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) -m mp.cli fetch

audit:
	$(PY) -m mp.cli groups
	$(PY) -m mp.cli audit
	$(PY) -m mp.cli splits

test:
	$(PY) -m pytest

features:
	$(PY) -m mp.cli features

train-cls:
	$(PY) -m mp.cli train-cls

controls:
	$(PY) -m mp.cli controls

train-seg:
	$(PY) -m mp.cli segment
	$(PY) -m mp.cli train-seg --all

report-phase2:
	$(PY) -m mp.cli report-phase2
