# `mp analyse`

```bash
python -m mp.cli analyse --image IMG.jpg [--sensors S1 S2 S3 S4] [--adc mcp3008|ads1115] [--json]
```

At least one of `--image` and `--sensors` is required. The JSON output follows
`docs/analyse.schema.json` (validated in `tests/test_analyse.py`).

| Field | Meaning |
|---|---|
| `polymer.label` | `PVC`, `NYLON`, or `null` when a quality gate refuses (see `polymer.reason`) |
| `polymer.p_nylon`, `polymer.confidence` | mean particle probability and `max(p, 1-p)`; not well calibrated (`confidence_note` carries the hold-out ECE) |
| `particles.count` | particles found by the automatic segmentation |
| `particles.size_hist_px` | histogram of equivalent diameters in pixels; `um_per_px` is `null` until the pixel size is calibrated |
| `concentration.mg_L`, `interval_90` | estimate and split-conformal 90 % interval, or `null` with `reason` |
| `concentration.note` | `rig model estimate, not a measurement` |
| `warnings` | free text, e.g. the background/exposure caveat from `MODEL_CARD.md` |
| `versions`, `git_commit` | provenance |

## Quality gates (polymer)

The label is withheld, with a reason, when the image is not 4:3, its resolution was not
seen in training, its mean brightness is outside the training range, or fewer than 3
particles are found.

## Refusals (concentration)

No estimate is returned when a detector is at full scale, when S1 and S2 are both below the
blank level, or when an engineered feature lies outside the range seen in training.

## Data needed

- Polymer: `data/processed/particles_resample.csv.gz` (`make features`) and
  `results/phase2_cv.json`. The fitted model is cached in `data/models/baseline_a.joblib`
  together with hashes of the training inputs; the cache is rebuilt when they change.
- Concentration: `data/processed/scattering_v1.csv` (`make sim`), or
  `scattering_v1_ads1115.csv` (`python -m mp.cli scatter-generate --adc ads1115`).
  The model is the one in `results/phase3_selected.json`.
