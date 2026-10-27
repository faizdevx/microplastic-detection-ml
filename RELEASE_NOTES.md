# Release notes: v0.2.0

Code DOI: `TODO-after-upload`. Dataset DOI (new version): `TODO-after-upload`.
The dataset record 10.5281/zenodo.21073653 holds the original archive.

## What changed

**Data pipeline.** Download with checksum verification, near-duplicate grouping of the
images, an audit report, and leak-free splits made by group (`splits.json`).
The archive holds 2040 files that form 600 near-duplicate groups.

**Polymer identification.** Per-particle classical classifier with group-aware
cross-validation (pooled accuracy 0.976), confound controls, an
explainability report, and a single hold-out evaluation (accuracy 0.987).
YOLOv11 segmentation was trained on a GPU with 30-epoch budget with patience 10, batch 6, final-epoch weights (mask mAP50 0.739 ± 0.019 over five folds, automatic masks as labels; `results/yolo_summary.md`).

**Concentration estimation.** A multi-angle scattering model of the rig (Mie theory, photon
Monte Carlo, photodiode and ADC chain), a regression pipeline with split-conformal intervals,
and analyses of linearity, geometry sensitivity and detection limits. Best model in the
cross-validated comparison: lasso (R² 0.81).

**Robustness of polymer identification.** Three variants were compared with the baseline on the same controls, using the cross-validation images only: local normalisation, exposure randomisation and a patch network. None improves on both the cross-resolution and the background-disc control (`results/phase5_summary.md`); the baseline stays the default. Shape features alone reach CV accuracy 0.864 with background discs at chance. YOLOv11 training is reported in `results/yolo_summary.md`.

**Release preparation.** A table of the quantitative statements in the response to reviewers with the matching evidence in the repository (`docs/claims_vs_evidence.md`), generated tables, figures and named numbers for the manuscript (`make paper-assets`), a staged bundle with checksums for a new dataset version (`make bundle`), an acquisition protocol, an annotation quality command (`mp annotate-qa`) and a release check (`make release-check`).

**Tooling.** `mp analyse` for an image and/or four detector counts (JSON schema in
`docs/analyse.schema.json`), ONNX export, a benchmark script, a Gradio demo, hardware notes,
continuous integration, a Dockerfile and a claims-audit script.

## Reproduce

```bash
pip install -e ".[dev]"
make all          # data, audit, features, classifier, controls, scattering model, regression
python app/app.py
```

`make all` does not evaluate the hold-out and does not train YOLOv11-seg (`make train-seg` does, in about 4.5 hours on a 4 GB GPU with the short schedule)
(`make train-seg-full`, GPU recommended).

## Known limits (from MODEL_CARD.md)

### Not for
- Environmental or weathered samples, biofouled or UV-aged particles.
- Any polymer other than PVC and Nylon. There is no "other" class, so an unknown particle is
  forced into one of the two.
- Any claim about chemical identity. Fluorescence colour and brightness are indirect.

### Failure modes
- Touching or overlapping particles are split or merged by the watershed.
- Faint particles below the signal floor are missed; border-touching particles are dropped.
- Sensitive to exposure and illumination (raw intensity features).
- Probabilities are not well calibrated (ECE 0.125).

### Not evaluated
- **YOLOv11-seg against human-drawn masks:** not available; no corrected masks exist.
- **Human-corrected masks:** none imported; masks are auto-generated.
- **Pixel calibration:** none.

See `MODEL_CARD.md` and `DATASET_CARD.md` for the controls, the confound findings and the
data provenance.
