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

### Controls (retrained fold models)

The phase 2 fold weights were not kept, so the five fold models were retrained with the same schedule: mask mAP50 0.737 ± 0.015 (stored 0.739 ± 0.019), image balanced accuracy 0.937 ± 0.077 (stored 0.894 ± 0.112).

| Control (image balanced accuracy) | YOLO | Per-particle classifier |
|---|---|---|
| Particles replaced by class-independent discs | 0.797 ± 0.128 | 0.811 |
| Resolution-matched folds, per-image normalisation | 0.980 ± 0.019 | 0.966 |
| Resolution-matched folds, no normalisation | 0.981 ± 0.017 | 0.966 |

With every particle replaced by a disc of one colour, YOLO still separates the classes well above chance (0.5). The per-particle classifier's value is its own background control from phase 5 (pseudo-particles taken from background pixels), which lands at a similar level. Much of the image-level decision therefore comes from the background and acquisition context, not from particle appearance. Per-image normalisation does not change the resolution-matched result (mask mAP50 0.700 ± 0.019 with normalisation).

### Longer schedule on fold 1

| Schedule | Epochs run | Mask mAP50 | Mask mAP50-95 | Image balanced accuracy |
|---|---|---|---|---|
| short (budget 30, patience 10) | 21 | 0.713 | 0.289 | 0.784 |
| full (budget 150, patience 30) | 150 | 0.777 | 0.340 | 0.995 |

### YOLO against the per-particle classifier (same folds, image level)

865 paired images: YOLO accuracy 0.933, classifier accuracy 0.979. Only YOLO correct: 8; only the classifier correct: 48; exact McNemar p = 4.69e-08. Accuracy difference (YOLO minus classifier) -0.046, 95 % interval [-0.066, -0.027] from 2000 bootstrap resamples of 510 image groups.

