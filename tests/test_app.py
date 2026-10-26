import sys

import pytest

from mp.config import repo_root

sys.path.insert(0, str(repo_root() / "app"))
gradio = pytest.importorskip("gradio")
import app  # noqa: E402


def test_app_builds_with_three_tabs():
    demo = app.build_app()
    cfg = demo.get_config_file()
    tabs = [c["props"].get("label") for c in cfg["components"] if c["type"] == "tabitem"]
    assert tabs == ["Polymer", "Concentration", "About"]


def test_about_text_carries_limits():
    text = app.about_text()
    assert "Not for" in text and "Failure modes" in text


def test_estimate_note_wording():
    assert app.NOTE == "rig model estimate, not a measurement"
