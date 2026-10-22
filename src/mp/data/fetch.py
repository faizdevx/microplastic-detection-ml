"""Download, verify and extract RAW_DATASET.zip, then build the file manifest."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from mp.config import load_config, resolve

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_EXTS = {".txt", ".json", ".xml", ".csv", ".yaml", ".yml"}
EXIF_DATETIME_TAGS = (36867, 36868, 306)  # DateTimeOriginal, Digitized, DateTime


def _hash_file(path: Path, algo: str, chunk: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def md5_of(path: Path) -> str:
    return _hash_file(path, "md5")


def sha256_of(path: Path) -> str:
    return _hash_file(path, "sha256")


def verify_md5(path: Path, expected: str) -> bool:
    """Return True if md5 matches. On mismatch the file is deleted."""
    if md5_of(path) == expected.lower():
        return True
    path.unlink(missing_ok=True)
    return False


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, resuming a partial file with an HTTP Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    pos = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:  # already complete
            return
        r.raise_for_status()
        if pos and r.status_code != 206:  # server ignored Range: restart
            pos = 0
        total = int(r.headers.get("content-length", 0)) + pos or None
        with open(dest, "ab" if pos else "wb") as fh, tqdm(
            total=total, initial=pos, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for block in r.iter_content(1 << 20):
                fh.write(block)
                bar.update(len(block))


def safe_extract(zip_path: Path, target: Path) -> None:
    """Extract every member, rejecting paths that escape `target` (zip-slip)."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        for info in infos:
            dest = (target / info.filename).resolve()
            if dest != target and target not in dest.parents:
                raise ValueError(f"unsafe path in zip: {info.filename!r}")
        for info in tqdm(infos, desc="extract", unit="file"):
            zf.extract(info, target)


def infer_class(rel_path: Path, classes: list[str]) -> str:
    """Class from the first folder component that mentions a class name."""
    for part in rel_path.parts[:-1]:
        low = part.lower()
        for c in classes:
            if c.lower() in low:
                return c
    return ""


def _exif_datetime(img: Image.Image) -> str:
    try:
        exif = img.getexif()
        ifd = exif.get_ifd(0x8769)
        for tag in EXIF_DATETIME_TAGS:
            val = ifd.get(tag) or exif.get(tag)
            if val:
                return str(val)
    except Exception:
        pass
    return ""


def build_manifest(extract_dir: Path, classes: list[str]) -> tuple[pd.DataFrame, dict]:
    rows, skipped = [], []
    other_exts: dict[str, int] = {}
    label_files: list[str] = []
    folders: set[str] = set()
    for p in sorted(extract_dir.rglob("*")):
        rel = p.relative_to(extract_dir)
        if p.is_dir():
            folders.add(rel.as_posix())
            continue
        ext = p.suffix.lower()
        if ext not in IMAGE_EXTS:
            if p.name.startswith(".verified_"):
                continue
            key = ext or "(none)"
            other_exts[key] = other_exts.get(key, 0) + 1
            if ext in LABEL_EXTS:
                label_files.append(rel.as_posix())
            continue
        try:
            with Image.open(p) as img:
                w, h = img.size
                ch = len(img.getbands())
                exif_dt = _exif_datetime(img)
                img.verify()
        except Exception as exc:  # unreadable: log and skip
            skipped.append({"rel_path": rel.as_posix(), "error": str(exc)})
            continue
        rows.append(
            {
                "rel_path": rel.as_posix(),
                "class": infer_class(rel, classes),
                "width": w,
                "height": h,
                "channels": ch,
                "file_size": p.stat().st_size,
                "sha256": sha256_of(p),
                "exif_datetime": exif_dt,
            }
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "rel_path", "class", "width", "height", "channels",
            "file_size", "sha256", "exif_datetime",
        ],
    )
    inventory = {
        "n_images": len(df),
        "n_unreadable": len(skipped),
        "unreadable": skipped,
        "n_unclassified_images": int((df["class"] == "").sum()),
        "n_images_with_exif_datetime": int((df["exif_datetime"] != "").sum()),
        "folders": sorted(folders),
        "non_image_extensions": other_exts,
        "label_like_files": label_files,
    }
    return df, inventory


def run(cfg: dict | None = None, force: bool = False) -> int:
    cfg = cfg or load_config()
    z = cfg["zenodo"]
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    zip_path = raw_dir / z["filename"]
    extract_dir = resolve(cfg["paths"]["extract_dir"])
    manifest_path = resolve(cfg["paths"]["manifest"])
    marker = extract_dir / f".verified_{z['md5']}"
    results = resolve(cfg["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)

    if marker.exists() and manifest_path.exists() and not force:
        print(f"already fetched and verified: {extract_dir}")
        return 0

    if not (zip_path.exists() and md5_of(zip_path) == z["md5"].lower()):
        download(z["url"], zip_path)
    if not verify_md5(zip_path, z["md5"]):
        print(f"md5 mismatch for {zip_path.name}; file deleted", file=sys.stderr)
        return 1
    print("md5 ok")

    if extract_dir.exists() and any(extract_dir.iterdir()):
        if not force:
            print(f"{extract_dir} exists and is not empty; re-run with --force",
                  file=sys.stderr)
            return 1
        shutil.rmtree(extract_dir)
    safe_extract(zip_path, extract_dir)

    df, inventory = build_manifest(extract_dir, cfg["classes"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    (results / "data_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf8"
    )
    marker.write_text(z["md5"])
    print(f"manifest: {len(df)} images {df['class'].value_counts().to_dict()}")
    if inventory["non_image_extensions"]:
        print(f"non-image content: {inventory['non_image_extensions']}")
    return 0
