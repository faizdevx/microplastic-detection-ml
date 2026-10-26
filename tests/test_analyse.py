"""Quality gates, refusals and the output schema of `mp analyse` (no dataset needed)."""
import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np
import pytest

from mp import analyse
from mp.config import load_named, repo_root
from mp.imaging.features import FEATURE_GROUPS, feature_columns
from mp.models import regression as R

SCHEMA = json.loads((repo_root() / "docs" / "analyse.schema.json").read_text(encoding="utf8"))


class _Model:
    def predict_proba(self, X):
        return np.tile([0.2, 0.8], (len(X), 1))


@pytest.fixture
def polymer_bundle(monkeypatch):
    fit = {
        "model": _Model(), "features": feature_columns(list(FEATURE_GROUPS)),
        "brightness_range": [0.0, 255.0], "resolutions": [(640, 480), (755, 566)],
    }
    bundle = {"cfg": {}, "ccfg": load_named("classifier"), "fit": fit, "ece": 0.125}
    monkeypatch.setattr(analyse, "_polymer_bundle", lambda: bundle)
    return bundle


def _image(path: Path, size=(640, 480), discs=6) -> str:
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    for k in range(discs):
        cv2.circle(img, (60 + 90 * k, h // 2), 18, (200, 90, 10), -1)
    cv2.imwrite(str(path), img)
    return str(path)


def _check(r):
    jsonschema.validate(r, SCHEMA)


def test_good_image_gets_a_label(tmp_path, polymer_bundle):
    r = analyse.analyse(_image(tmp_path / "a.png"))
    _check(r)
    assert r["polymer"]["label"] == "NYLON" and r["polymer"]["confidence"] == pytest.approx(0.8)
    assert r["particles"]["count"] == 6
    assert "not well calibrated" in r["polymer"]["confidence_note"]
    assert r["particles"]["um_per_px"] is None and r["particles"]["pixel_size_note"]


def test_blank_image_is_refused(tmp_path, polymer_bundle):
    r = analyse.analyse(_image(tmp_path / "b.png", discs=0))
    _check(r)
    assert r["polymer"]["label"] is None and "particles" in r["polymer"]["reason"]


def test_non_4_3_image_is_refused(tmp_path, polymer_bundle):
    r = analyse.analyse(_image(tmp_path / "c.png", size=(720, 480)))
    _check(r)
    assert r["polymer"]["label"] is None and "4:3" in r["polymer"]["reason"]


def test_unseen_resolution_is_refused(tmp_path, polymer_bundle):
    r = analyse.analyse(_image(tmp_path / "d.png", size=(800, 600)))
    _check(r)
    assert r["polymer"]["label"] is None and "not seen in training" in r["polymer"]["reason"]


def test_dark_image_outside_brightness_range_is_refused(tmp_path, polymer_bundle):
    polymer_bundle["fit"]["brightness_range"] = [50.0, 200.0]
    r = analyse.analyse(_image(tmp_path / "e.png"))
    assert r["polymer"]["label"] is None and "brightness" in r["polymer"]["reason"]


@pytest.fixture
def conc_bundle(monkeypatch):
    cols = list(R.make_features(
        __import__("pandas").DataFrame([{f"dS{i}": 1.0 for i in range(1, 5)}])).columns)

    class M:
        def predict(self, X):
            return np.array([0.3])

    bundle = {"R": R, "sel": {"model": "lasso", "geometry": "g0"}, "model": M(), "q": 0.2,
              "blank": np.array([60.0] * 4), "lo": np.full(len(cols), -1e9),
              "hi": np.full(len(cols), 1e9), "cols": cols, "top": 1023, "adc": "mcp3008",
              "conc_range": [0.1, 10.0]}
    monkeypatch.setattr(analyse, "_conc_bundle", lambda adc: bundle)
    return bundle


def test_concentration_estimate_and_note(conc_bundle):
    r = analyse.analyse(sensors=[120, 80, 62, 61])
    _check(r)
    c = r["concentration"]
    assert c["mg_L"] == pytest.approx(10**0.3)
    assert c["interval_90"] == pytest.approx([10**0.1, 10**0.5])
    assert c["note"] == "rig model estimate, not a measurement"
    assert r["polymer"] is None


def test_saturated_detector_is_refused(conc_bundle):
    c = analyse.analyse(sensors=[1023, 80, 62, 61])["concentration"]
    assert c["mg_L"] is None and "saturated" in c["reason"]


def test_negative_signal_is_refused(conc_bundle):
    c = analyse.analyse(sensors=[40, 40, 62, 61])["concentration"]
    assert c["mg_L"] is None and "no signal" in c["reason"]


def test_out_of_range_features_are_refused(conc_bundle):
    conc_bundle["hi"] = np.full(len(conc_bundle["cols"]), 5.0)
    c = analyse.analyse(sensors=[120, 80, 62, 61])["concentration"]
    assert c["mg_L"] is None and "outside the training range" in c["reason"]


def test_needs_some_input():
    with pytest.raises(ValueError):
        analyse.analyse()


def test_schema_rejects_missing_keys():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"polymer": None}, SCHEMA)


def test_text_format_covers_refusals(conc_bundle):
    r = analyse.analyse(sensors=[1023, 80, 62, 61])
    assert "refused" in analyse.format_text(r)


@pytest.mark.skipif(not (repo_root() / "data" / "processed" / "scattering_v1.csv").exists(),
                    reason="needs `make sim`")
def test_real_concentration_model_roundtrip():
    c = analyse.estimate_concentration([160, 70, 64, 64])
    assert c["mg_L"] is not None and c["interval_90"][0] < c["mg_L"] < c["interval_90"][1]
