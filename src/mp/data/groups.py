"""Near-duplicate grouping (pHash/dHash + exact sha256) and session inference."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from mp.config import load_config, resolve


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class BKTree:
    """Metric tree over integers with Hamming distance."""

    def __init__(self):
        self.root: list | None = None  # [value, [indices], {dist: child}]

    def add(self, value: int, idx: int) -> None:
        if self.root is None:
            self.root = [value, [idx], {}]
            return
        node = self.root
        while True:
            d = hamming(value, node[0])
            if d == 0:
                node[1].append(idx)
                return
            child = node[2].get(d)
            if child is None:
                node[2][d] = [value, [idx], {}]
                return
            node = child

    def query(self, value: int, max_dist: int) -> list[tuple[int, int]]:
        """All (index, distance) with distance <= max_dist."""
        out: list[tuple[int, int]] = []
        if self.root is None:
            return out
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = hamming(value, node[0])
            if d <= max_dist:
                out.extend((i, d) for i in node[1])
            for dist, child in node[2].items():
                if d - max_dist <= dist <= d + max_dist:
                    stack.append(child)
        return out


def _hash_to_int(h: imagehash.ImageHash) -> int:
    return int("".join("1" if b else "0" for b in h.hash.flatten()), 2)


def compute_hashes(paths: list[Path], hash_size: int = 8) -> list[tuple[int, int]]:
    """(phash, dhash) per image, computed on a downscaled grayscale copy."""
    out = []
    for p in tqdm(paths, desc="hashing", unit="img"):
        with Image.open(p) as img:
            g = img.convert("L")
            g.thumbnail((256, 256))
            out.append(
                (
                    _hash_to_int(imagehash.phash(g, hash_size=hash_size)),
                    _hash_to_int(imagehash.dhash(g, hash_size=hash_size)),
                )
            )
    return out


def candidate_pairs(
    phashes: list[int], shas: list[str], max_dist: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Pairs (i, j, phash_distance) with distance <= max_dist, plus exact-sha pairs.

    Exact-sha pairs are returned separately and get distance -1 in the edge list
    so they merge at every threshold.
    """
    tree = BKTree()
    edges: list[tuple[int, int, int]] = []
    for i, h in enumerate(phashes):
        for j, d in tree.query(h, max_dist):
            edges.append((j, i, d))
        tree.add(h, i)
    by_sha: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(shas):
        by_sha[s].append(i)
    exact = []
    for idxs in by_sha.values():
        for k in range(1, len(idxs)):
            exact.append((idxs[0], idxs[k]))
    return edges, exact


def cluster(
    n: int,
    classes: list[str],
    edges: list[tuple[int, int, int]],
    exact: list[tuple[int, int]],
    threshold: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Union-find grouping that never merges across classes.

    Returns group ids (0..k-1, ordered by first member) and the cross-class
    near-duplicate pairs (i, j, distance; -1 = identical bytes) that were
    refused a merge.
    """
    uf = UnionFind(n)
    cross: list[tuple[int, int, int]] = []
    for i, j, d in edges:
        if d <= threshold:
            if classes[i] == classes[j]:
                uf.union(i, j)
            else:
                cross.append((i, j, d))
    for i, j in exact:
        if classes[i] == classes[j]:
            uf.union(i, j)
        else:
            cross.append((i, j, -1))
    roots = [uf.find(i) for i in range(n)]
    remap: dict[int, int] = {}
    gid = np.empty(n, dtype=int)
    for i, r in enumerate(roots):
        gid[i] = remap.setdefault(r, len(remap))
    return gid, cross


def _filename_number(name: str) -> int | None:
    m = re.findall(r"\d+", Path(name).stem)
    return int(m[-1]) if m else None


def infer_sessions(
    df: pd.DataFrame, exif_gap_min: float, filename_gap: int | None
) -> pd.Series:
    """Probable acquisition sessions, per class.

    Key = resolution. Within a key, split sorted EXIF timestamps on gaps larger
    than `exif_gap_min` minutes when every image has one. If `filename_gap` is
    set, otherwise split sorted filename numbers on gaps larger than that. It is
    off by default: on this dataset it cuts near-duplicate copies into
    different "sessions".
    """
    session = pd.Series("", index=df.index, dtype=object)
    for (cls, w, h), sub in df.groupby(["class", "width", "height"]):
        base = f"{cls}_{w}x{h}"
        stamps = pd.to_datetime(sub["exif_datetime"], format="%Y:%m:%d %H:%M:%S",
                                errors="coerce")
        if stamps.notna().all() and len(sub) > 1:
            order = stamps.sort_values()
            gaps = order.diff().dt.total_seconds().div(60).fillna(0)
            part = (gaps > exif_gap_min).cumsum()
        else:
            nums = sub["rel_path"].map(_filename_number)
            if filename_gap is not None and nums.notna().all() and len(sub) > 1:
                order = nums.sort_values()
                part = (order.diff().fillna(0) > filename_gap).cumsum()
            else:
                part = pd.Series(0, index=sub.index)
        for idx, k in part.items():
            session.loc[idx] = f"{base}_s{int(k)}"
    return session


def sensitivity_table(
    df: pd.DataFrame,
    edges: list[tuple[int, int, int]],
    exact: list[tuple[int, int]],
    thresholds: list[int],
) -> list[dict]:
    rows = []
    classes = df["class"].tolist()
    for t in thresholds:
        gid, cross = cluster(len(df), classes, edges, exact, t)
        row = {"threshold": t, "cross_class_pairs": len(cross)}
        for c in sorted(set(classes)):
            mask = (df["class"] == c).to_numpy()
            row[f"groups_{c}"] = int(len(set(gid[mask])))
            row[f"files_{c}"] = int(mask.sum())
        rows.append(row)
    return rows


def build_groups(cfg: dict | None = None) -> dict:
    """Compute groups.csv. Returns info for the audit (sensitivity, cross-class flags)."""
    cfg = cfg or load_config()
    gcfg = cfg["groups"]
    manifest = pd.read_csv(resolve(cfg["paths"]["manifest"]), keep_default_na=False)
    extract_dir = resolve(cfg["paths"]["extract_dir"])
    df = manifest[manifest["class"].isin(cfg["classes"])].reset_index(drop=True)

    hashes = compute_hashes(
        [extract_dir / p for p in df["rel_path"]], gcfg["hash_size"]
    )
    df["phash"] = [f"{h[0]:016x}" for h in hashes]
    df["dhash"] = [f"{h[1]:016x}" for h in hashes]

    max_t = max(max(gcfg["sensitivity_thresholds"]), gcfg["hash_threshold"])
    edges, exact = candidate_pairs([h[0] for h in hashes], df["sha256"].tolist(), max_t)
    classes = df["class"].tolist()
    gid, cross = cluster(len(df), classes, edges, exact, gcfg["hash_threshold"])
    df["group_id"] = gid
    df["group_size"] = df.groupby("group_id")["group_id"].transform("size")
    df["session_id"] = infer_sessions(
        df, cfg["sessions"]["exif_gap_minutes"], cfg["sessions"]["filename_gap"]
    )

    out = resolve(cfg["paths"]["groups"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["rel_path", "class", "phash", "dhash", "group_id", "group_size",
        "session_id", "sha256"]].to_csv(out, index=False)

    paths = df["rel_path"].tolist()
    info = {
        "threshold": gcfg["hash_threshold"],
        "sensitivity": sensitivity_table(
            df, edges, exact, gcfg["sensitivity_thresholds"]
        ),
        "cross_class_pairs": [
            {"a": paths[i], "b": paths[j], "phash_distance": d} for i, j, d in cross
        ],
        "n_exact_duplicate_pairs": len(exact),
    }
    results = resolve(cfg["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)
    (results / "groups_info.json").write_text(json.dumps(info, indent=2), encoding="utf8")
    return info
