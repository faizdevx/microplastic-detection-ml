"""Leak-checked access to images by fold, with duplicate removal and weights."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from mp.config import load_config, resolve
from mp.data.splits import validate_splits

HOLDOUT = -1


@dataclass
class Pool:
    """All unique images with split membership.

    `fold` is 0..n_folds-1 for the CV pool and HOLDOUT (-1) for the hold-out.
    `weight` = 1 / (kept images in the group); class totals are balanced.
    """

    images: pd.DataFrame
    n_folds: int

    def cv(self) -> pd.DataFrame:
        return self.images[self.images["fold"] != HOLDOUT]

    def holdout(self) -> pd.DataFrame:
        return self.images[self.images["fold"] == HOLDOUT]

    def fold_split(self, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        cv = self.cv()
        return cv[cv["fold"] != k], cv[cv["fold"] == k]


def load_pool(cfg: dict | None = None, one_per_group: bool = False) -> Pool:
    cfg = cfg or load_config()
    groups = pd.read_csv(resolve(cfg["paths"]["groups"]), keep_default_na=False)
    manifest = pd.read_csv(resolve(cfg["paths"]["manifest"]), keep_default_na=False)
    splits = json.loads(resolve(cfg["paths"]["splits"]).read_text(encoding="utf8"))

    # Leak check on the complete table, before any filtering.
    validate_splits(groups, splits)

    fold_of = splits["cv_fold_of"]
    hold = set(splits["holdout"])
    df = groups.merge(
        manifest[["rel_path", "width", "height", "file_size"]], on="rel_path"
    )
    df["fold"] = [HOLDOUT if r in hold else fold_of[r] for r in df["rel_path"]]

    # Byte-identical duplicates: keep the first per sha256 (always the same group).
    df = df.sort_values("rel_path").drop_duplicates("sha256", keep="first")
    if one_per_group:
        df = df.drop_duplicates("group_id", keep="first")
    df["group_size_kept"] = df.groupby("group_id")["group_id"].transform("size")
    df["weight"] = 1.0 / df["group_size_kept"]
    # Equal total weight per class, mean weight 1.
    cls_tot = df.groupby("class")["weight"].transform("sum")
    df["weight"] = df["weight"] / cls_tot * len(df) / df["class"].nunique()
    df["y"] = (df["class"] == "NYLON").astype(int)
    df = df.reset_index(drop=True)

    # No group may live in two places after filtering either.
    where = df.groupby("group_id")["fold"].nunique()
    assert (where == 1).all(), "a group spans several folds/hold-out"
    return Pool(images=df, n_folds=splits["meta"]["n_folds"])
