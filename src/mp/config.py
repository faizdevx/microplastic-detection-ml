"""Config loading. All paths in configs/data.yaml are relative to the repo root."""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def repo_root() -> Path:
    env = os.environ.get("MP_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "data.yaml").exists():
            return parent
    return Path.cwd().resolve()


def config_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "configs" / "data.yaml"


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else config_path()
    with open(p, encoding="utf8") as fh:
        return yaml.safe_load(fh)


def resolve(cfg_path: str, root: Path | None = None) -> Path:
    return (root or repo_root()) / cfg_path


def load_named(name: str) -> dict:
    """Load configs/<name>.yaml (e.g. 'classifier', 'yolo_seg')."""
    return load_config(repo_root() / "configs" / f"{name}.yaml")
