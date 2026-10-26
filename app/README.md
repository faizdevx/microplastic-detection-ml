# Demo app

```bash
pip install -e ".[dev]"
make features train-cls sim        # data the app reads (polymer model inputs, scattering table)
python app/app.py                  # http://127.0.0.1:7860, --share false is the default
```

Tabs:

- **Polymer**: upload a 4:3 fluorescence image. Returns a polymer label, or "cannot decide"
  with the reason (not 4:3, unseen resolution, brightness outside the training range, fewer
  than 3 particles), the particle outlines and a size histogram in pixels.
- **Concentration**: four detector counts and an ADC option. Returns an estimate with a 90 %
  interval, or a refusal (saturated detector, no signal above blank, out-of-range features).
  Every estimate carries the line "rig model estimate, not a measurement".
- **About**: limits from `MODEL_CARD.md` and the confound-control table from the generated
  results.

The app runs on CPU and needs no internet access once the dependencies are installed
(`GRADIO_ANALYTICS_ENABLED` is set to `False`).

## Deploying to Hugging Face Spaces (not done)

1. Create a Space with the Gradio SDK.
2. Copy the repository (or add it as a submodule), plus the generated files the app reads:
   `results/phase2_cv.json`, `results/phase2_holdout.json`, `results/phase2_controls.json`,
   `results/phase3_selected.json`, `data/processed/particles_resample.csv.gz`,
   `data/processed/image_stats.csv`, `data/processed/groups.csv`, `data/processed/splits.json`,
   `data/processed/manifest.csv` equivalents and `data/processed/scattering_v1.csv`.
3. Add a `requirements.txt` with `-e .` (or the pinned dependency list) and set `app_file: app/app.py`.
4. Do not enable the dataset download in the Space; keep `data/raw` out of it.

The example images in `app/examples/` are CC-BY-4.0 (see their README).
