"""Figures for the README that are not produced by the pipeline itself.

segmentation_examples.png: raw fluorescence image next to the auto-segmentation overlay,
two PVC and two Nylon images. Needs the dataset (`python -m mp.cli fetch`).
"""
from __future__ import annotations

import warnings

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mp.config import load_config, load_named, resolve  # noqa: E402
from mp.data.dataset import load_pool  # noqa: E402
from mp.imaging import features as F  # noqa: E402
from mp.imaging.segment import segment  # noqa: E402

warnings.filterwarnings("ignore")


def main(seed: int = 3) -> None:
    cfg, ccfg = load_config(), load_named("classifier")
    pool = load_pool(cfg)
    imgs = pool.images
    rng = np.random.default_rng(seed)
    pick = []
    for cls in ("PVC", "NYLON"):
        sub = imgs[(imgs["class"] == cls) & (imgs["width"].isin([640, 742, 755]))]
        pick += list(sub.sample(2, random_state=int(rng.integers(1e6)))["rel_path"])
    root = resolve(cfg["paths"]["extract_dir"])
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for col, rel in enumerate(pick):
        a = F.prepare(root / rel, "resample", ccfg["preprocess"])
        labels = segment(a["bgsub"], ccfg["segmentation"])
        vis = a["geo"].copy()
        for i in range(1, int(labels.max()) + 1):
            cnts, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)
        axes[0, col].imshow(a["geo"])
        axes[0, col].set_title(rel.split("/")[-1], fontsize=9)
        axes[1, col].imshow(vis)
        axes[1, col].set_title(f"{int(labels.max())} particles", fontsize=9)
        for r in (0, 1):
            axes[r, col].axis("off")
    fig.suptitle("Fluorescence images (top) and automatic particle segmentation (bottom)")
    fig.tight_layout()
    out = resolve("results/figures/segmentation_examples.png")
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
