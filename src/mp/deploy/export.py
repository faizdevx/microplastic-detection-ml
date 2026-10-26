"""Export models for edge use. Writes results/deploy_export.json describing what was exported."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from mp.annotate.auto import yolo_root
from mp.config import load_named, repo_root, resolve
from mp.eval.metrics import run_meta


def export_dir() -> Path:
    d = repo_root() / "data" / "models" / "export"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rel(p: Path | str) -> str:
    """Repo-relative posix path, so result files do not carry machine-specific paths."""
    try:
        return Path(p).resolve().relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        return Path(p).name


def yolo_weights() -> tuple[Path | None, str]:
    """Trained weights if they exist, otherwise the smoke-test weights (not a trained model)."""
    trained = yolo_root() / "runs" / "all_cv" / "weights" / "best.pt"
    if trained.exists():
        return trained, "trained"
    smoke = yolo_root() / "runs" / "smoke" / "weights" / "best.pt"
    return (smoke, "smoke_test_2_epochs_not_trained") if smoke.exists() else (None, "missing")


def export_yolo(imgsz: int | None = None, ncnn: bool = False) -> dict:
    w, kind = yolo_weights()
    out = {"weights": rel(w) if w else None, "kind": kind, "formats": {}}
    if w is None:
        out["note"] = "no YOLO weights found; run `mp train-seg --smoke` or `mp train-seg --all`"
        return out
    if kind != "trained":
        out["note"] = "these weights come from a 2-epoch smoke test; they are not a trained model"
    from ultralytics import YOLO

    imgsz = imgsz or load_named("yolo_seg")["imgsz"]
    tag = "trained" if kind == "trained" else "smoke"
    for fmt in ["onnx"] + (["ncnn"] if ncnn else []):
        try:
            produced = Path(str(YOLO(str(w)).export(format=fmt, imgsz=imgsz, device="cpu")))
            dst = export_dir() / f"yolo_seg_{tag}{produced.suffix or '_ncnn'}"
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            shutil.move(str(produced), str(dst))
            out["formats"][fmt] = {"status": "ok", "path": rel(dst), "imgsz": imgsz}
        except Exception as exc:  # exporter missing a dependency etc.
            out["formats"][fmt] = {"status": "failed",
                                   "error": f"{type(exc).__name__}: {exc}"[:300]}
    if not ncnn:
        out["formats"]["ncnn"] = {"status": "not attempted",
                                  "note": "the exporter downloads extra binaries; use --ncnn"}
    return out


def export_baseline() -> dict:
    """Baseline-A pipeline to ONNX with skl2onnx where supported, else keep the joblib."""
    from mp.analyse import _polymer_bundle, model_cache_path

    fit = _polymer_bundle()["fit"]
    out = {"model": fit["stamp"]["model"], "n_features": len(fit["features"]),
           "joblib": rel(model_cache_path())}
    try:
        from skl2onnx import to_onnx

        sample = np.zeros((1, len(fit["features"])), dtype=np.float32)
        onx = to_onnx(fit["model"], sample, options={"zipmap": False}, target_opset=15)
        path = export_dir() / "baseline_a.onnx"
        path.write_bytes(onx.SerializeToString())
        out["onnx"] = {"status": "ok", "path": rel(path)}
    except Exception as exc:
        out["onnx"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300],
                       "fallback": "keep the joblib file"}
    return out


def run(ncnn: bool = False) -> dict:
    res = {"meta": run_meta(), "yolo": export_yolo(ncnn=ncnn), "baseline_a": export_baseline()}
    (resolve("results") / "deploy_export.json").write_text(json.dumps(res, indent=2),
                                                           encoding="utf8")
    return res
