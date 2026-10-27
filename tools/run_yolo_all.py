"""Train and evaluate YOLOv11-seg on every dataset key, one process per key.

    python tools/run_yolo_all.py [--schedule short|config] [--force]

Results go to results/phase2_yolo_<key>.json (written by `mp train-seg`); keys that already
have a result file are skipped unless --force is given.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ([f"fold_{k}" for k in range(5)] + ["cross_a_to_b", "cross_b_to_a"]
        + [f"res_matched_{k}" for k in range(5)] + ["all_cv"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", choices=["short", "config"], default="short")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    for key in args.only or KEYS:
        out = ROOT / "results" / f"phase2_yolo_{key}.json"
        if out.exists() and not args.force:
            print(f"skip {key}: {out.name} exists", flush=True)
            continue
        cmd = [sys.executable, "-W", "ignore", "-m", "mp.cli", "train-seg", "--data", key]
        if args.schedule == "short":
            cmd += ["--schedule", "short"]
        t = time.time()
        print(f"start {key}", flush=True)
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        print(f"done {key} rc={rc} in {(time.time() - t) / 60:.1f} min", flush=True)
        if rc != 0:
            return rc
    print("ALL_YOLO_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
