from pathlib import Path

import pytest

from mp.config import repo_root
from mp.deploy import benchmark, capture, export


def test_capture_import_is_safe_without_camera():
    if capture.available():
        pytest.skip("camera stack present")
    with pytest.raises(RuntimeError, match="picamera2"):
        capture.capture("x.jpg")


def test_device_info_has_expected_keys():
    d = benchmark.device_info()
    assert {"device_model", "is_raspberry_pi", "cpu", "cpu_count", "ram_gb", "os", "python"} <= set(d)
    assert "faisa" not in d["device_model"].lower()


def test_pi_benchmark_refuses_elsewhere():
    if benchmark.device_info()["is_raspberry_pi"]:
        pytest.skip("running on a Pi")
    with pytest.raises(SystemExit):
        benchmark.run(n_runs=1, pi=True)


def test_peak_memory_is_positive():
    assert benchmark.peak_rss_mb() > 10


def test_export_paths_are_repo_relative():
    inside = repo_root() / "data" / "models" / "x.onnx"
    assert export.rel(inside) == "data/models/x.onnx"
    assert export.rel(Path("/somewhere/else/y.onnx")) == "y.onnx"


def test_missing_yolo_weights_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "yolo_weights", lambda: (None, "missing"))
    out = export.export_yolo()
    assert out["weights"] is None and "no YOLO weights" in out["note"]
