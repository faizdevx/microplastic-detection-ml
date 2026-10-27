"""Latency, memory and throughput of segmentation + classification on this machine."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from mp.config import load_config, load_named, resolve
from mp.eval.metrics import run_meta


def device_info() -> dict:
    model = ""
    dt = Path("/proc/device-tree/model")
    if dt.exists():
        model = dt.read_text(errors="ignore").strip("\x00\n ")
    return {
        "device_model": model or f"{platform.system()} computer",
        "is_raspberry_pi": "raspberry pi" in model.lower(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "os": platform.platform(),
        "python": sys.version.split()[0],
    }


def peak_rss_mb() -> float:
    info = psutil.Process(os.getpid()).memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak:
        return peak / 1e6
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB
    except Exception:
        return info.rss / 1e6


def _sample_images(n: int) -> list[Path]:
    root = resolve(load_config()["paths"]["extract_dir"])
    imgs = sorted(root.rglob("*.jpg"))
    if not imgs:
        raise FileNotFoundError("dataset not found; run `make data`")
    return imgs[:: max(1, len(imgs) // n)][:n]


def _stats(v) -> dict:
    return {"median_ms": float(np.median(v) * 1e3), "p95_ms": float(np.percentile(v, 95) * 1e3)}


def run(n_runs: int = 50, warmup: int = 3, pi: bool = False) -> dict:
    from mp import analyse
    from mp.imaging import features as F
    from mp.imaging import segment as sg

    dev = device_info()
    if pi and not dev["is_raspberry_pi"]:
        raise SystemExit("--pi requested but this is not a Raspberry Pi; nothing written")
    fit = analyse._polymer_bundle()["fit"]
    ccfg = load_named("classifier")
    imgs = _sample_images(10)
    t_seg, t_cls, t_all = [], [], []
    for i in range(n_runs + warmup):
        path = imgs[i % len(imgs)]
        t0 = time.perf_counter()
        a = F.prepare(path, "resample", ccfg["preprocess"])
        labels = sg.segment(a["bgsub"], ccfg["segmentation"])
        parts = F.particle_features(labels, a["bgsub"], a["norm"])
        t1 = time.perf_counter()
        if len(parts):
            fit["model"].predict_proba(parts[fit["features"]])
        t2 = time.perf_counter()
        if i >= warmup:
            t_seg.append(t1 - t0)
            t_cls.append(t2 - t1)
            t_all.append(t2 - t0)
    out = {
        "meta": run_meta(), "device": dev,
        "input": {"size": "640x480 after letterbox", "n_distinct_images": len(imgs)},
        "runs": n_runs, "warmup_runs": warmup,
        "segmentation_and_features": _stats(t_seg), "classification": _stats(t_cls),
        "total": _stats(t_all),
        "throughput_images_per_s": float(1.0 / np.mean(t_all)),
        "peak_rss_mb": peak_rss_mb(),
        "libraries": {m: __import__(m).__version__ for m in ("numpy", "sklearn", "cv2")},
    }
    onnx = _yolo_onnx_latency(n_runs, warmup)
    if onnx:
        out["yolo_onnx_untrained_weights"] = onnx
    name = "deploy_benchmark_pi.json" if pi else "deploy_benchmark.json"
    (resolve("results") / name).write_text(json.dumps(out, indent=2), encoding="utf8")
    return out


def _yolo_onnx_latency(n_runs: int, warmup: int) -> dict | None:
    """Forward-pass time of the exported YOLO graph (the weights are not trained)."""
    from mp.deploy.export import export_dir

    cands = sorted(export_dir().glob("yolo_seg_*.onnx"))
    if not cands:
        return None
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(cands[0]), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        x = np.random.default_rng(0).random(shape, dtype=np.float32)
        ts = []
        for i in range(n_runs + warmup):
            t = time.perf_counter()
            sess.run(None, {inp.name: x})
            if i >= warmup:
                ts.append(time.perf_counter() - t)
        return {"file": cands[0].name, "input_shape": shape, "runs": n_runs, **_stats(ts)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}
