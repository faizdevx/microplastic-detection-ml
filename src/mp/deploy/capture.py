"""Camera capture with fixed exposure and gain, plus a sidecar JSON.

Needs the optional extra `pip install -e ".[pi]"` and a Raspberry Pi camera. Importing this
module never fails on a machine without a camera.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

try:  # optional dependency
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None


def available() -> bool:
    return Picamera2 is not None


def capture(path: str | Path, exposure_us: int = 20000, analogue_gain: float = 1.0,
            size: tuple[int, int] = (640, 480), notes: str = "") -> dict:
    """Save one frame to `path` and write `<path>.json` with the acquisition settings."""
    if Picamera2 is None:
        raise RuntimeError("picamera2 is not installed or no camera is present")
    path = Path(path)
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": size}))
    cam.set_controls({"AeEnable": False, "ExposureTime": int(exposure_us),
                      "AnalogueGain": float(analogue_gain), "AwbEnable": False})
    cam.start()
    try:
        cam.capture_file(str(path))
        meta = cam.capture_metadata()
    finally:
        cam.stop()
        cam.close()
    side = {
        "file": path.name, "size": list(size), "exposure_us": int(exposure_us),
        "analogue_gain": float(analogue_gain), "auto_exposure": False,
        "auto_white_balance": False,
        "camera_metadata": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                            for k, v in dict(meta).items()},
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "notes": notes,
    }
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(side, indent=2), encoding="utf8")
    return side
