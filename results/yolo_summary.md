## YOLOv11 instance segmentation (generated)

Schedule `short`: up to 30 epochs, patience 10, batch 6, device 0, final epoch (last.pt) weights. Cross-validation folds from `splits.json`; the hold-out was not used.

**The labels are the classical segmentation's output (automatic, uncorrected masks), so the model learns to imitate that segmentation plus the class label; mAP measures agreement with it, not with human-drawn particle outlines.**

### 5 cross-validation folds (mean ± std)

| Metric | Value |
|---|---|
| Mask mAP50 | 0.739 ± 0.019 |
| Mask mAP50-95 | 0.321 ± 0.013 |
| Box mAP50 | 0.849 ± 0.019 |
| Box mAP50-95 | 0.513 ± 0.013 |
| PVC precision / recall | 0.713 ± 0.051 / 0.735 ± 0.027 |
| Nylon precision / recall | 0.790 ± 0.018 / 0.654 ± 0.056 |
| Image accuracy (aggregated detections) | 0.887 ± 0.121 |
| Image balanced accuracy | 0.894 ± 0.112 |

### Resolution-matched folds (640x480 and 1280x960 only)

Mask mAP50 0.724 ± 0.007; image accuracy 0.991 ± 0.008; image balanced accuracy 0.981 ± 0.017.

### Cross-resolution control (balanced accuracy of the image decision)

| Train to test | YOLO | Per-particle classifier | Metadata shortcut |
|---|---|---|---|
| 640 to 1280 | 0.970 | 0.864 | 0.506 |
| 1280 to 640 | 0.875 | 0.762 | 0.679 |

### Model on all cross-validation images

30 epochs; mask mAP50 0.764, mask mAP50-95 0.338, measured on its own training images (validated on its own training images; a fit check, not a generalisation result).

