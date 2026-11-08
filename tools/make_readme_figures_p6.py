"""Figures for the README from stored results: confound controls, feature sets, YOLOv11-seg per fold
and the YOLOv11-seg controls.

Reads results/*.json and paper_assets/figures (`python -m mp.cli paper-assets`); writes results/figures/.
"""
from __future__ import annotations

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mp.config import resolve  # noqa: E402


def yolo_folds() -> None:
    res = resolve("results")
    runs = [json.loads((res / f"phase2_yolo_fold_{k}.json").read_text(encoding="utf8")) for k in range(5)]
    summ = json.loads((res / "yolo_summary.json").read_text(encoding="utf8"))
    x = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x, [r["seg"]["map50"] for r in runs], color="#0072B2")
    axes[0].set_xticks(x, [f"fold {k}" for k in range(5)])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("mask mAP50 (automatic masks)")
    axes[0].set_title("Mask mAP50 per fold")
    axes[1].bar(x - 0.2, [r["image_level"]["accuracy"] for r in runs], 0.4, color="#009E73",
                label="image accuracy")
    axes[1].bar(x + 0.2, [r["image_level"]["balanced_accuracy"] for r in runs], 0.4, color="#E69F00",
                label="balanced accuracy")
    axes[1].set_xticks(x, [f"fold {k}" for k in range(5)])
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[1].set_title("Image-level decision per fold")
    axes[1].legend(loc="lower right")
    cr = summ["cross_resolution"]
    fig.suptitle("YOLOv11-seg, short schedule: "
                 f"mask mAP50 {summ['folds']['mask_map50']['mean']:.3f}, "
                 f"640 to 1280 {cr['cross_a_to_b']['image_balanced_accuracy']:.3f}, "
                 f"1280 to 640 {cr['cross_b_to_a']['image_balanced_accuracy']:.3f} (balanced accuracy)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(resolve("results/figures/yolo_folds.png"), dpi=110)


def yolo_controls() -> None:
    c = json.loads(resolve("results/phase7_yolo_controls.json").read_text(encoding="utf8"))
    d, n = c["background_discs"], c["normalised_resolution_matched"]
    x = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x - 0.2, [f["balanced_accuracy"] for f in d["original_images_per_fold"]], 0.4,
                color="#0072B2", label="original images")
    axes[0].bar(x + 0.2, [f["balanced_accuracy"] for f in d["per_fold"]], 0.4, color="#D55E00",
                label="background discs")
    axes[0].axhline(d["classifier_balanced_accuracy"], color="#009E73", lw=1.2,
                    label="per-particle classifier, discs")
    axes[0].set_title("Background-disc control (same fold models)")
    axes[1].bar(x, [f["balanced_accuracy"] for f in n["per_fold"]], color="#0072B2")
    axes[1].axhline(n["unnormalised_balanced_accuracy"]["mean"], color="#E69F00", lw=1.2,
                    label="not normalised (mean)")
    axes[1].axhline(n["classifier_balanced_accuracy"], color="#009E73", lw=1.2,
                    label="per-particle classifier")
    axes[1].set_title("Resolution-matched folds, per-image normalisation")
    for ax in axes:
        ax.set_xticks(x, [f"fold {k}" for k in range(5)])
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_ylabel("image balanced accuracy")
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(resolve("results/figures/yolo_controls.png"), dpi=110)


def main() -> None:
    src = resolve("paper_assets/figures")
    dst = resolve("results/figures")
    for name in ("controls_by_variant.png", "feature_sets.png"):
        shutil.copyfile(src / name, dst / name)
    yolo_folds()
    yolo_controls()


if __name__ == "__main__":
    main()
