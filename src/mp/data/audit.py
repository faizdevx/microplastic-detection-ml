"""Dataset audit: duplication, resolution, intensity, shortcut baseline, figures."""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tqdm import tqdm  # noqa: E402

from mp.config import load_config, resolve  # noqa: E402

CHANNEL_COLS = ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B"]
SIZE_COLS = ["width", "height", "file_size"]


def compute_image_stats(df: pd.DataFrame, extract_dir) -> pd.DataFrame:
    rows = []
    for rel in tqdm(df["rel_path"], desc="intensity stats", unit="img"):
        img = cv2.imread(str(extract_dir / rel), cv2.IMREAD_COLOR)
        if img is None:
            rows.append([np.nan] * 6)
            continue
        rgb = img[:, :, ::-1].reshape(-1, 3).astype(np.float64)
        rows.append([*rgb.mean(0), *rgb.std(0)])
    return pd.concat(
        [df[["rel_path"]].reset_index(drop=True), pd.DataFrame(rows, columns=CHANNEL_COLS)],
        axis=1,
    )


def shortcut_baseline(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int
) -> dict:
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
    }
    out = {}
    for name, model in models.items():
        s = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="accuracy")
        out[name] = {
            "fold_accuracy": [round(float(v), 4) for v in s],
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
        }
    return out


def _montage(paths, extract_dir, path_out, title, ncols=5, thumb=160):
    n = len(paths)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, rel in zip(np.atleast_1d(axes).ravel(), paths):
        img = cv2.imread(str(extract_dir / rel))
        if img is None:
            continue
        img = cv2.resize(img[:, :, ::-1], (thumb, int(thumb * img.shape[0] / img.shape[1])))
        ax.imshow(img)
        ax.set_title(rel.split("/")[-1], fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path_out, dpi=110)
    plt.close(fig)


def make_figures(df: pd.DataFrame, extract_dir, fig_dir, seed: int) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    classes = sorted(df["class"].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 3.5), squeeze=False)
    for ax, c in zip(axes[0], classes):
        sizes = df[df["class"] == c].drop_duplicates("group_id")["group_size"]
        ax.hist(sizes, bins=range(1, int(sizes.max()) + 2), edgecolor="black")
        ax.set_title(f"{c}: images per near-duplicate group")
        ax.set_xlabel("group size")
        ax.set_ylabel("n groups")
    fig.tight_layout()
    fig.savefig(fig_dir / "group_size_hist.png", dpi=120)
    plt.close(fig)

    for c in classes:
        sub = df[df["class"] == c]
        reps = sub.drop_duplicates("group_id")
        pick = reps.sample(min(25, len(reps)), random_state=seed)["rel_path"].tolist()
        _montage(pick, extract_dir, fig_dir / f"montage_{c}.png",
                 f"{c}: one representative per group (25 random groups)")
        multi = sub[sub["group_size"] > 1]["group_id"].unique()
        for k, g in enumerate(rng.permutation(multi)[:3]):
            members = sub[sub["group_id"] == g]["rel_path"].tolist()[:10]
            _montage(members, extract_dir, fig_dir / f"cluster_{c}_{k + 1}.png",
                     f"{c}: group {g} ({len(members)} shown)")


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    results = resolve(cfg["paths"]["results"])
    fig_dir = results / "figures"
    extract_dir = resolve(cfg["paths"]["extract_dir"])
    manifest = pd.read_csv(resolve(cfg["paths"]["manifest"]), keep_default_na=False)
    groups = pd.read_csv(resolve(cfg["paths"]["groups"]), keep_default_na=False)
    info = json.loads((results / "groups_info.json").read_text(encoding="utf8"))

    df = groups.merge(
        manifest[["rel_path", "width", "height", "file_size", "exif_datetime"]],
        on="rel_path",
    )
    stats_path = resolve(cfg["paths"]["image_stats"])
    if stats_path.exists() and len(pd.read_csv(stats_path)) == len(df):
        stats = pd.read_csv(stats_path)
    else:
        stats = compute_image_stats(df, extract_dir)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(stats_path, index=False)
    df = df.merge(stats, on="rel_path")

    per_class = {}
    for c, sub in df.groupby("class"):
        res = sub.groupby(["width", "height"]).size().sort_values(ascending=False)
        gs = sub.drop_duplicates("group_id")["group_size"]
        per_class[c] = {
            "total_files": int(len(sub)),
            "exact_duplicate_files": int(sub["sha256"].duplicated().sum()),
            "unique_groups": int(sub["group_id"].nunique()),
            "singleton_groups": int((gs == 1).sum()),
            "group_size": {
                "mean": round(float(gs.mean()), 2),
                "median": float(gs.median()),
                "max": int(gs.max()),
            },
            "resolutions": {f"{w}x{h}": int(n) for (w, h), n in res.items()},
            "file_size_bytes": {k: round(float(v), 1) for k, v in
                                sub["file_size"].describe()[["mean", "std", "min", "max"]].items()},
            "channel_mean_of_means": {k: round(float(sub[k].mean()), 3) for k in CHANNEL_COLS[:3]},
            "channel_mean_of_stds": {k: round(float(sub[k].mean()), 3) for k in CHANNEL_COLS[3:]},
            "n_sessions": int(sub["session_id"].nunique()),
            "sessions": {k: int(v) for k, v in sub["session_id"].value_counts().items()},
        }

    y = (df["class"] == cfg["classes"][0]).astype(int).to_numpy()
    sb = cfg["shortcut_baseline"]
    baseline = {
        "size_only": shortcut_baseline(
            df[SIZE_COLS], y, df["group_id"].to_numpy(), sb["n_folds"], cfg["seed"]
        ),
        "size_and_intensity": shortcut_baseline(
            df[SIZE_COLS + CHANNEL_COLS], y, df["group_id"].to_numpy(),
            sb["n_folds"], cfg["seed"]
        ),
    }
    worst = max(m["mean"] for v in baseline.values() for m in v.values())
    audit = {
        "hash_threshold": info["threshold"],
        "n_images": int(len(df)),
        "per_class": per_class,
        "threshold_sensitivity": info["sensitivity"],
        "cross_class_duplicate_pairs": info["cross_class_pairs"],
        "shortcut_baseline": baseline,
        "acquisition_confound": bool(worst > sb["confound_threshold"]),
        "acquisition_confound_note": (
            f"Metadata/intensity-only classifiers reach {worst:.3f} group-aware CV accuracy "
            f"(threshold {sb['confound_threshold']}): class is predictable without looking "
            "at particles." if worst > sb["confound_threshold"] else
            f"Best shortcut accuracy {worst:.3f} <= {sb['confound_threshold']}."
        ),
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
    make_figures(df, extract_dir, fig_dir, cfg["seed"])
    write_summary(audit, results / "dataset_summary.md")
    return audit


def write_summary(audit: dict, out_path) -> None:
    lines = [
        "| Class | Files | Unique groups | Exact-duplicate files | Sessions | Resolutions |",
        "|---|---|---|---|---|---|",
    ]
    for c, v in audit["per_class"].items():
        res = ", ".join(f"{k} ({n})" for k, n in list(v["resolutions"].items())[:4])
        lines.append(
            f"| {c} | {v['total_files']} | {v['unique_groups']} | "
            f"{v['exact_duplicate_files']} | {v['n_sessions']} | {res} |"
        )
    lines += ["", f"Near-duplicate threshold: pHash Hamming distance <= {audit['hash_threshold']}.", ""]
    lines += ["| Shortcut baseline (group-aware 5-fold CV) | Logistic regression | Random forest |",
              "|---|---|---|"]
    for name, v in audit["shortcut_baseline"].items():
        lines.append(
            f"| {name} | {v['logreg']['mean']:.3f} ± {v['logreg']['std']:.3f} | "
            f"{v['random_forest']['mean']:.3f} ± {v['random_forest']['std']:.3f} |"
        )
    lines += ["", audit["acquisition_confound_note"], ""]
    out_path.write_text("\n".join(lines), encoding="utf8")
