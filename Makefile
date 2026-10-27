PY ?= python

.PHONY: all install data audit test lint features train-cls controls train-seg-full \
	report-phase2 sim sim-analyse train-reg report-phase3 export benchmark app \
	paper-assets bundle release-check

# Whole chain from a clean clone. Never evaluates the hold-out (report-phase2 has no
# --holdout here) and never runs the hours-long YOLO training (see train-seg-full).
all: data audit features train-cls controls sim sim-analyse train-reg report-phase2 report-phase3

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

lint:
	$(PY) -m ruff check .

features:
	$(PY) -m mp.cli features

train-cls:
	$(PY) -m mp.cli train-cls

controls:
	$(PY) -m mp.cli controls

# GPU recommended: 150 epochs on all cross-validation images with auto masks.
train-seg-full:
	$(PY) -m mp.cli segment
	$(PY) -m mp.cli train-seg --all

report-phase2:
	$(PY) -m mp.cli report-phase2

sim:
	$(PY) -m mp.cli scatter-generate

sim-analyse:
	$(PY) -m mp.cli scatter-analyse

train-reg:
	$(PY) -m mp.cli train-reg

report-phase3:
	$(PY) -m mp.cli report-phase3

export:
	$(PY) -m mp.cli export

benchmark:
	$(PY) -m mp.cli benchmark

app:
	$(PY) app/app.py

paper-assets:
	$(PY) -m mp.cli paper-assets

bundle:
	$(PY) tools/make_bundle.py

release-check:
	$(PY) tools/release_check.py
