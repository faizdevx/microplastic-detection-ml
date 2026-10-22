"""Leak-free splits: group-level hold-out, StratifiedGroupKFold, leave-one-session-out."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from mp.config import load_config, resolve


def make_splits(df: pd.DataFrame, cfg: dict, groups_sha256: str = "") -> dict:
    """`df` needs rel_path, class, group_id, session_id."""
    sc = cfg["splits"]
    seed = sc["seed"]

    g = df.groupby("group_id").agg(cls=("class", "first")).reset_index()
    g_cv, g_hold = train_test_split(
        g, test_size=sc["holdout_fraction"], stratify=g["cls"], random_state=seed
    )
    hold_mask = df["group_id"].isin(set(g_hold["group_id"]))
    holdout = df[hold_mask]
    cv_df = df[~hold_mask].reset_index(drop=True)

    sgkf = StratifiedGroupKFold(n_splits=sc["n_folds"], shuffle=True, random_state=seed)
    fold_of = {}
    for k, (_, val_idx) in enumerate(
        sgkf.split(cv_df, cv_df["class"], groups=cv_df["group_id"])
    ):
        for rel in cv_df.loc[val_idx, "rel_path"]:
            fold_of[rel] = k

    out = {
        "meta": {
            "split_version": sc["split_version"],
            "seed": seed,
            "holdout_fraction_groups": sc["holdout_fraction"],
            "n_folds": sc["n_folds"],
            "hash_threshold": cfg["groups"]["hash_threshold"],
            "groups_csv_sha256": groups_sha256,
            "n_images": int(len(df)),
            "n_holdout_images": int(len(holdout)),
            "n_cv_images": int(len(cv_df)),
        },
        "holdout": sorted(holdout["rel_path"]),
        "cv_fold_of": dict(sorted(fold_of.items())),
    }

    sess_per_class = cv_df.groupby("class")["session_id"].nunique()
    if len(sess_per_class) and (sess_per_class >= 3).all():
        loso = []
        for s in sorted(cv_df["session_id"].unique()):
            val = cv_df[cv_df["session_id"] == s]
            val_groups = set(val["group_id"])
            train = cv_df[(cv_df["session_id"] != s) & (~cv_df["group_id"].isin(val_groups))]
            loso.append(
                {
                    "session": s,
                    "train": sorted(train["rel_path"]),
                    "val": sorted(val["rel_path"]),
                }
            )
        out["leave_one_session_out"] = loso
    else:
        out["leave_one_session_out"] = None
        out["meta"]["loso_note"] = (
            "fewer than 3 inferred sessions in at least one class; LOSO not emitted"
        )
    validate_splits(df, out)
    return out


def validate_splits(df: pd.DataFrame, splits: dict, ratio_tol: float = 0.05) -> None:
    """Raise AssertionError on any leakage or imbalance."""
    gid = dict(zip(df["rel_path"], df["group_id"]))
    cls = dict(zip(df["rel_path"], df["class"]))
    hold = set(splits["holdout"])
    fold_of = splits["cv_fold_of"]

    assert not hold & set(fold_of), "hold-out image also in a CV fold"
    assert hold | set(fold_of) == set(df["rel_path"]), "images missing from splits"

    hold_groups = {gid[r] for r in hold}
    cv_groups = {gid[r] for r in fold_of}
    assert not hold_groups & cv_groups, "hold-out group appears in CV folds"

    n_folds = splits["meta"]["n_folds"]
    by_fold: dict[int, set] = {k: set() for k in range(n_folds)}
    for r, k in fold_of.items():
        by_fold[k].add(gid[r])
    for a in range(n_folds):
        for b in range(a + 1, n_folds):
            assert not by_fold[a] & by_fold[b], f"group shared by folds {a} and {b}"

    classes = sorted(set(cls.values()))
    first = classes[0]
    overall = sum(cls[r] == first for r in fold_of) / max(len(fold_of), 1)
    for k in range(n_folds):
        rows = [r for r, f in fold_of.items() if f == k]
        ratio = sum(cls[r] == first for r in rows) / max(len(rows), 1)
        assert abs(ratio - overall) <= ratio_tol, (
            f"fold {k} class ratio {ratio:.3f} vs global {overall:.3f}"
        )

    for entry in splits.get("leave_one_session_out") or []:
        tg = {gid[r] for r in entry["train"]}
        vg = {gid[r] for r in entry["val"]}
        assert not tg & vg, f"LOSO leak for session {entry['session']}"
        assert not (set(entry["train"]) | set(entry["val"])) & hold, "LOSO uses hold-out"


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    groups_path = resolve(cfg["paths"]["groups"])
    df = pd.read_csv(groups_path, keep_default_na=False)
    sha = hashlib.sha256(groups_path.read_bytes()).hexdigest()
    splits = make_splits(df, cfg, sha)
    out = resolve(cfg["paths"]["splits"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits), encoding="utf8")
    m = splits["meta"]
    print(f"splits: {m['n_cv_images']} CV images, {m['n_holdout_images']} hold-out")
    return splits
