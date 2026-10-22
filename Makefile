PY ?= python

.PHONY: install data audit test

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
