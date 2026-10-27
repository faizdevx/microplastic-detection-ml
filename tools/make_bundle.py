"""Stage the files for a new Zenodo version in release_bundle/ (nothing is uploaded).

    python tools/make_bundle.py [--allow-dirty]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release_bundle"

TOP = ["README.md", "LICENSE", "CITATION.cff", "RELEASE_NOTES.md", "MODEL_CARD.md", "DATASET_CARD.md"]
DESCRIPTIONS = [
    ("source_archive", "source code of the repository at the recorded commit (git archive)"),
    ("data/processed/groups.csv", "near-duplicate group and session of every image"),
    ("data/processed/splits.json", "hold-out and cross-validation folds (leak-free, by group)"),
    ("data/processed/scattering_v1.csv", "S1-S4 table produced by the rig model (see DATASET_CARD.md)"),
    ("data/processed/scattering_v1.config.yaml", "exact configuration of that table"),
    ("results/", "generated results: JSON, summaries and figures"),
    ("configs/", "configuration files"),
    ("docs/", "documentation"),
    ("README.md", "project overview"),
    ("LICENSE", "Apache License 2.0"),
    ("CITATION.cff", "citation metadata"),
    ("RELEASE_NOTES.md", "release notes"),
    ("MODEL_CARD.md", "model card"),
    ("DATASET_CARD.md", "dataset card"),
    ("BUNDLE_INFO.json", "commit, branch and time of this bundle"),
    ("CHECKSUMS.txt", "sha256 and md5 of every file"),
]


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf8",
                          errors="replace").stdout


def digest(p: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(p, "rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def zenodo_files() -> list[str]:
    """Paths in the first column of the table in docs/zenodo_update.md."""
    text = (ROOT / "docs" / "zenodo_update.md").read_text(encoding="utf8")
    return [m for m in re.findall(r"^\| `([^`]+)` \|", text, flags=re.M)]


def describe(rel: str) -> str:
    for prefix, d in DESCRIPTIONS:
        if rel == prefix or (prefix.endswith("/") and rel.startswith(prefix)) or rel.startswith(prefix):
            return d
    return "file"


def collect() -> list[str]:
    files = set(zenodo_files()) | set(TOP)
    for base in ("results", "configs", "docs"):
        for p in (ROOT / base).rglob("*"):
            if p.is_file() and not p.name.endswith((".pyc",)):
                files.add(p.relative_to(ROOT).as_posix())
    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf8", errors="replace")
    dirty = git("status", "--porcelain", "--untracked-files=no").strip()
    if dirty and not args.allow_dirty:
        print("refusing to build: tracked files have uncommitted changes (use --allow-dirty):\n" + dirty)
        return 1
    commit = git("rev-parse", "HEAD").strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    missing, copied = [], []
    for rel in collect():
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied.append(rel)
    archive = OUT / f"source_archive_{commit[:8]}.zip"
    subprocess.run(["git", "archive", "--format=zip", "-o", str(archive), "HEAD"], cwd=ROOT, check=True)
    copied.append(archive.name)
    (OUT / "BUNDLE_INFO.json").write_text(json.dumps({
        "git_commit": commit, "branch": branch, "dirty": bool(dirty),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "missing_files": missing}, indent=2), encoding="utf8")
    copied.append("BUNDLE_INFO.json")
    rows = sorted(copied)
    (OUT / "README.txt").write_text(
        "Release bundle (staged, not uploaded).\n\n"
        + "\n".join(f"{r}: {describe(r)}" for r in rows) + "\n\nSee CHECKSUMS.txt.\n", encoding="utf8")
    rows.append("README.txt")
    lines = [f"{digest(OUT / r, 'sha256')}  {digest(OUT / r, 'md5')}  {(OUT / r).stat().st_size:>10}  {r}"
             for r in sorted(rows)]
    (OUT / "CHECKSUMS.txt").write_text("sha256  md5  bytes  file\n" + "\n".join(lines) + "\n",
                                       encoding="utf8")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"bundle: {len(rows) + 1} files, {total / 1e6:.1f} MB at {OUT.relative_to(ROOT)}; "
          f"commit {commit[:8]}")
    if missing:
        print("missing (listed in docs/zenodo_update.md but absent):", ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
