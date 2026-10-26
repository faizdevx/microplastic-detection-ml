"""sha256 and md5 of the files listed in docs/zenodo_update.md.

    python tools/checksums.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "data/processed/groups.csv",
    "data/processed/splits.json",
    "data/processed/scattering_v1.csv",
    "data/processed/scattering_v1.config.yaml",
    "results/data_audit.json",
    "DATASET_CARD.md",
    "MODEL_CARD.md",
]


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    out = {}
    print(f"{'file':45} {'bytes':>10}  md5 / sha256")
    for rel in FILES:
        p = ROOT / rel
        if not p.exists():
            print(f"{rel:45} {'missing':>10}")
            continue
        out[rel] = {"bytes": p.stat().st_size, "md5": digest(p, "md5"), "sha256": digest(p, "sha256")}
        print(f"{rel:45} {out[rel]['bytes']:>10}  {out[rel]['md5']}  {out[rel]['sha256']}")
    (ROOT / "results" / "zenodo_checksums.json").write_text(json.dumps(out, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
