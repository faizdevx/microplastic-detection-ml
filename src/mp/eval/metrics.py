"""Image-level metrics, group bootstrap and calibration."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from mp.config import repo_root

CLASS_NAMES = ["PVC", "NYLON"]  # y = 0 / 1


def point_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    """Metrics for binary labels y (1 = NYLON) and P(NYLON)."""
    pred = (prob >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
            for i, c in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "ece": float(ece(y, prob)),
    }
    out["roc_auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None
    return out


def ece(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error on top-label confidence."""
    pred = (prob >= 0.5).astype(int)
    conf = np.maximum(prob, 1 - prob)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total, err = len(y), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            err += m.sum() / total * abs(correct[m].mean() - conf[m].mean())
    return err


def reliability(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "mean_prob": prob[m].mean(), "frac_pos": y[m].mean(),
                         "n": int(m.sum())})
    return pd.DataFrame(rows)


def group_bootstrap_ci(
    df: pd.DataFrame, n: int = 1000, seed: int = 42, alpha: float = 0.05
) -> dict:
    """95 % CI by resampling GROUPS (not images). df needs y, prob, group_id."""
    rng = np.random.default_rng(seed)
    gids = df["group_id"].unique()
    by_g = {g: d for g, d in df.groupby("group_id")}
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]
    draws = {k: [] for k in keys}
    for _ in range(n):
        pick = rng.choice(gids, size=len(gids), replace=True)
        s = pd.concat([by_g[g] for g in pick])
        if s["y"].nunique() < 2:
            continue
        m = point_metrics(s["y"].to_numpy(), s["prob"].to_numpy())
        for k in keys:
            draws[k].append(m[k])
    return {
        k: [float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))]
        for k, v in draws.items() if v
    }


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact (binomial) McNemar test on paired correctness."""
    from scipy.stats import binomtest

    ca, cb = pred_a == y, pred_b == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p}


def run_meta(device: str = "cpu") -> dict:
    """Provenance stamped into every results JSON."""
    import sklearn

    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        git = ""
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "device": device,
    }
    try:
        import torch

        meta["torch"] = torch.__version__
    except Exception:
        pass
    return meta
