# Annotation guide: correcting particle masks

The auto masks (`data/annotations/auto/`) come from the classical segmentation in
`src/mp/imaging/segment.py` (background correction, Otsu on the red/Nile Red channel,
watershed). They are a starting point, not ground truth. A human pass on a stratified
subset gives (a) a measure of how good the auto masks are (mask IoU) and (b) better
training labels for YOLOv11-seg.

## Workflow

```bash
python -m mp.cli segment                 # auto labels + 640x480 images in data/yolo/
python -m mp.cli annotate-export         # writes data/annotations/to_correct/{images,coco_auto.json}
# correct the masks in CVAT or Label Studio (import coco_auto.json as "COCO 1.0"),
# export the result as COCO 1.0 / instance segmentation
python -m mp.cli annotate-import corrected.json   # YOLO labels + results/annotation_quality.json
python -m mp.cli train-seg --all         # data/yolo/labels now prefers corrected masks
```

The export holds one representative image per near-duplicate group, 150 groups per class
(`--n-per-class` changes this). Every image carries its split (`cv_fold_k` or `holdout`) in
the COCO `images[].split` field. Corrected hold-out images may be used to measure
auto-label quality and counting error, never to train.

## What counts as a particle

- One bright, compact fluorescent object = one polygon, regardless of brightness.
- Draw the outline where the object's brightness drops to roughly half of its peak, not at
  the faint halo.
- Class = the class of the image folder (`pvc` or `nylon`). Do not relabel by eye.

## Touching and overlapping particles

- Particles that touch but have visible separate cores: one polygon each.
- A clump with no visible separation between cores: **one** polygon for the clump. Do not
  guess hidden boundaries.
- Fibres or elongated fragments: one polygon along the whole length.

## Out-of-focus, faint, small

- Clearly blurred objects with a recognisable core: annotate the core.
- Faint specks you cannot tell from sensor noise: do not annotate.
- Minimum size: 20 pixels of area at 640×480 (matches `segmentation.min_area_px`). Smaller
  objects are ignored.
- Objects cut by the image border: do not annotate (the pipeline drops them too).

## Quality checks

- Correct at least 30 images per class before reading `annotation_quality.json`.
- `semantic_iou_mean` compares the union of all particle pixels; `instance_best_iou_mean`
  matches each corrected polygon to its best auto polygon; `instance_recall_at_iou50` is the
  share of corrected particles the auto mask found with IoU ≥ 0.5.
- If a second person corrects 10 % of the same images, report their mutual IoU as the
  inter-annotator agreement.

## Inter-annotator agreement

Two people correct the same 10 % of the exported images independently (separate COCO files).
Compare them with the same tool, using one person's file as the reference:

```bash
python -m mp.cli annotate-import annotator_a.json     # YOLO labels for annotator A
python -m mp.cli annotate-qa annotator_b.json         # auto masks vs annotator B
```

For agreement between the two people, convert one of them to the "auto" slot by copying their
YOLO txt files to `data/annotations/auto/` in a scratch checkout, then run `annotate-qa` with
the other file; the semantic IoU and the missed/spurious counts are then the agreement between
the annotators.

## Quality check commands

```bash
python -m mp.cli annotate-export                 # COCO set with the auto masks
# ... correct in CVAT or Label Studio, export COCO 1.0 ...
python -m mp.cli annotate-import corrected.json  # YOLO labels, semantic and instance IoU
python -m mp.cli annotate-qa corrected.json      # per image IoU, count error, missed, spurious
```

`annotate-qa` writes `results/annotation_quality.json`: mean semantic IoU, mean absolute and
signed count error per image, the share of images with at least one missed or spurious
particle (instances are matched one to one at IoU 0.5), and one row per image.
Until a corrected file is imported, no number about mask quality exists.
