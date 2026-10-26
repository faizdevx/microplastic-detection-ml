"""CI smoke test for the YOLO-seg training path on 20 synthetic images (no dataset, CPU)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from mp.config import load_named
from mp.models import yolo


def make_dataset(root: Path, n: int = 20) -> Path:
    rng = np.random.default_rng(0)
    lists = {"train": [], "val": []}
    for i in range(n):
        split = "val" if i % 5 == 0 else "train"
        img = np.zeros((240, 320, 3), np.uint8)
        label_lines = []
        for _ in range(int(rng.integers(3, 8))):
            cx, cy, r = int(rng.integers(30, 290)), int(rng.integers(30, 210)), int(rng.integers(8, 16))
            cv2.circle(img, (cx, cy), r, (200, 90, 10), -1)
            ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            pts = np.stack([(cx + r * np.cos(ang)) / 320, (cy + r * np.sin(ang)) / 240], 1)
            label_lines.append(f"{i % 2} " + " ".join(f"{v:.5f}" for v in pts.reshape(-1)))
        ip = root / "images" / f"s{i:02d}.jpg"
        lp = root / "labels" / f"s{i:02d}.txt"
        ip.parent.mkdir(parents=True, exist_ok=True)
        lp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ip), img)
        lp.write_text("\n".join(label_lines) + "\n")
        lists[split].append(ip.resolve().as_posix())
    for k, v in lists.items():
        (root / f"{k}.txt").write_text("\n".join(v) + "\n")
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"path": root.resolve().as_posix(), "train": "train.txt",
                                    "val": "val.txt", "names": {0: "pvc", 1: "nylon"}}))
    return data


def main() -> None:
    from ultralytics import YOLO

    ycfg = load_named("yolo_seg")
    with tempfile.TemporaryDirectory() as tmp:
        data = make_dataset(Path(tmp))
        args = yolo.train_args(ycfg, data, "ci_smoke", smoke=True)
        args["project"] = str(Path(tmp) / "runs")
        model = YOLO(ycfg["smoke"]["model"])
        model.train(**args)
        print("YOLO-seg smoke training finished")


if __name__ == "__main__":
    main()
