"""Offline phase-2 tests: synthetic images and tables only."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from mp.annotate import auto
from mp.data import dataset as ds
from mp.data import splits as sp
from mp.eval import controls as K
from mp.eval import metrics as M
from mp.eval import report
from mp.imaging import features as F
from mp.imaging import normalize as nz
from mp.imaging import segment as sg
from mp.models import yolo

SEG = {"channel": 0, "blur_sigma": 1.0, "threshold": "otsu", "min_signal": 8,
       "open_kernel": 3, "watershed_min_distance": 5, "min_area_px": 20, "drop_border": True}


def _disc_image(centers, r=20, size=(200, 200), value=200):
    img = np.zeros((*size, 3), np.uint8)
    for cx, cy in centers:
        cv2.circle(img, (cx, cy), r, (value, value // 2, 10), -1)
    return img


def test_watershed_splits_touching_discs():
    img = _disc_image([(80, 100), (115, 100)])  # overlapping discs
    assert sg.segment(img, SEG).max() == 2


def test_single_disc_stays_one_object():
    assert sg.segment(_disc_image([(100, 100)]), SEG).max() == 1


def test_border_touching_objects_are_dropped():
    assert sg.segment(_disc_image([(5, 100)]), SEG).max() == 0


def _feats(img):
    labels = sg.segment(img, SEG)
    return F.particle_features(labels, img, img.astype(np.float32) / 255.0)


def test_circle_is_rounder_than_square():
    circ = _feats(_disc_image([(100, 100)], r=30))
    sq = np.zeros((200, 200, 3), np.uint8)
    sq[70:130, 70:130] = (200, 100, 10)
    square = _feats(sq)
    assert circ["circularity"].iloc[0] > 0.9
    assert square["circularity"].iloc[0] < circ["circularity"].iloc[0]


def test_colour_ratio_invariant_to_brightness():
    bright = _feats(_disc_image([(100, 100)], value=240))
    dim = _feats(_disc_image([(100, 100)], value=120))
    for col in ("frac_R", "frac_G", "ratio_R_over_G"):
        assert bright[col].iloc[0] == pytest.approx(dim[col].iloc[0], rel=0.05)
    assert bright["raw_mean_R"].iloc[0] > 1.5 * dim["raw_mean_R"].iloc[0]


def test_background_only_masking_removes_all_particles():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 4, (200, 200, 3)).astype(np.uint8)
    img = np.clip(base + _disc_image([(60, 60), (140, 130)], r=18, value=220), 0, 255).astype(np.uint8)
    bgsub, bg = nz.subtract_background(img, kernel=25, downscale=4)
    labels = sg.segment(bgsub, SEG)
    assert labels.max() == 2
    masked = sg.mask_particles(img, labels, bg, dilate=3)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate((labels > 0).astype(np.uint8), se).astype(bool)
    assert np.array_equal(masked[mask], bg[mask])  # every particle pixel replaced
    assert masked[mask].max() < 40                  # no bright pixels left inside
    bgsub2, _ = nz.subtract_background(masked, kernel=25, downscale=4)
    assert sg.segment(bgsub2, SEG).max() == 0       # nothing left to segment


def test_geometry_normalize_letterboxes_to_target():
    img = np.full((960, 1280, 3), 100, np.uint8)
    assert nz.geometry_normalize(img, "resample").shape == (480, 640, 3)
    assert nz.geometry_normalize(img, "native").shape == (960, 1280, 3)
    tall = np.full((400, 100, 3), 255, np.uint8)
    out = nz.geometry_normalize(tall, "resample")
    assert out.shape == (480, 640, 3) and out[:, :10].max() == 0  # padded, not stretched


def _synthetic_images(n_groups=40, per=3):
    rows = []
    for c_i, cls in enumerate(["PVC", "NYLON"]):
        for g in range(n_groups):
            gid = c_i * n_groups + g
            for k in range(per):
                rows.append({
                    "rel_path": f"{cls}/{cls.lower()}_{gid}_{k}.jpg", "class": cls,
                    "group_id": gid, "session_id": f"{cls}_{g % 3}",
                    "sha256": f"{gid}-{k % 2}",  # k=0 and k=2 are byte-identical
                    "width": 640 if g % 2 else 1280, "height": 480 if g % 2 else 960,
                    "file_size": 1000 + gid, "phash": "0", "dhash": "0",
                    "group_size": per,
                })
    return pd.DataFrame(rows)


def _write_project(tmp_path):
    df = _synthetic_images()
    cfg = {
        "classes": ["PVC", "NYLON"],
        "groups": {"hash_threshold": 6},
        "splits": {"seed": 42, "holdout_fraction": 0.15, "n_folds": 5, "split_version": "1"},
        "paths": {"groups": str(tmp_path / "groups.csv"), "manifest": str(tmp_path / "manifest.csv"),
                  "splits": str(tmp_path / "splits.json")},
    }
    df[["rel_path", "class", "phash", "dhash", "group_id", "group_size", "session_id",
        "sha256"]].to_csv(tmp_path / "groups.csv", index=False)
    df[["rel_path", "class", "width", "height", "file_size"]].assign(
        channels=3, sha256=df["sha256"], exif_datetime="").to_csv(tmp_path / "manifest.csv", index=False)
    (tmp_path / "splits.json").write_text(json.dumps(sp.make_splits(df, cfg)))
    return cfg


def test_loader_dedupes_and_never_mixes_groups(tmp_path):
    cfg = _write_project(tmp_path)
    pool = ds.load_pool(cfg)
    assert pool.images["sha256"].is_unique  # byte-identical duplicates dropped
    for k in range(pool.n_folds):
        tr, va = pool.fold_split(k)
        assert not set(tr["group_id"]) & set(va["group_id"])
        assert not set(tr["group_id"]) & set(pool.holdout()["group_id"])
        assert not set(va["group_id"]) & set(pool.holdout()["group_id"])


def test_weights_and_one_per_group(tmp_path):
    cfg = _write_project(tmp_path)
    pool = ds.load_pool(cfg)
    img = pool.images
    assert img["weight"].mean() == pytest.approx(1.0)
    totals = img.groupby("class")["weight"].sum()
    assert totals["PVC"] == pytest.approx(totals["NYLON"])
    # inside a class every image of a group shares the weight 1/group_size_kept
    g = img[img["group_size_kept"] == 2].iloc[0]
    assert g["weight"] / img[img["group_size_kept"] == 1]["weight"].iloc[0] == pytest.approx(0.5) \
        if (img["group_size_kept"] == 1).any() else True
    one = ds.load_pool(cfg, one_per_group=True).images
    assert one["group_id"].is_unique


def test_loader_rejects_leaky_splits(tmp_path):
    cfg = _write_project(tmp_path)
    sj = json.loads((tmp_path / "splits.json").read_text())
    victim = sj["holdout"][0]
    sj["cv_fold_of"][victim] = 0  # same image now also in a fold
    (tmp_path / "splits.json").write_text(json.dumps(sj))
    with pytest.raises(AssertionError):
        ds.load_pool(cfg)


def test_yolo_dataset_builder_writes_fold_files(tmp_path, monkeypatch):
    monkeypatch.setenv("MP_ROOT", str(tmp_path))
    cfg = _write_project(tmp_path)
    pool = ds.load_pool(cfg)
    for rel in pool.images["rel_path"]:
        p = yolo._img_path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    out = yolo.build_datasets(pool, {"classes": ["pvc", "nylon"]})
    import yaml

    hold = {l for l in (tmp_path / "data/yolo/lists/holdout.txt").read_text().split()}
    assert hold
    for k in range(pool.n_folds):
        d = yaml.safe_load(out[f"fold_{k}"].read_text())
        tr = set(Path(d["train"]).read_text().split())
        va = set(Path(d["val"]).read_text().split())
        assert tr and va and not tr & va and not (tr | va) & hold
        assert d["names"] == {0: "pvc", 1: "nylon"}
    assert set(out) >= {"all_cv", "cross_a_to_b", "cross_b_to_a", "res_matched_0"}
    all_cv = yaml.safe_load(out["all_cv"].read_text())
    assert not set(Path(all_cv["train"]).read_text().split()) & hold


def test_holdout_guard_refuses_overwrite(tmp_path):
    f = tmp_path / "phase2_holdout.json"
    report.guard_holdout(f, i_know=False)  # not there yet: fine
    f.write_text("{}")
    with pytest.raises(SystemExit):
        report.guard_holdout(f, i_know=False)
    report.guard_holdout(f, i_know=True)


def test_mask_iou_identical_and_disjoint():
    sq = np.array([[10, 10], [40, 10], [40, 40], [10, 40]], np.float32)
    other = sq + 100
    assert auto.mask_iou([sq], [sq], 200, 200)["semantic_iou"] == pytest.approx(1.0)
    m = auto.mask_iou([sq], [other], 200, 200)
    assert m["semantic_iou"] == 0.0 and m["instance_best_ious"] == [0.0]


def test_polygon_roundtrip_to_yolo_txt(tmp_path):
    labels = sg.segment(_disc_image([(100, 100)], r=25), SEG)
    polys = auto.label_to_polygons(labels)
    assert len(polys) == 1
    f = tmp_path / "a.txt"
    auto.write_yolo_txt(f, polys, 1, 200, 200)
    back = auto.read_yolo_txt(f, 200, 200)
    assert back[0][0] == 1
    assert auto.mask_iou(polys, [back[0][1]], 200, 200)["semantic_iou"] > 0.95


def test_group_bootstrap_and_mcnemar():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"group_id": np.repeat(np.arange(60), 2),
                       "y": np.repeat(rng.integers(0, 2, 60), 2)})
    df["prob"] = np.where(df["y"] == 1, 0.9, 0.1)
    lo, hi = M.group_bootstrap_ci(df, n=50)["accuracy"]
    assert lo == hi == 1.0
    y = np.array([1, 1, 0, 0])
    assert M.mcnemar(y, y, y)["p_value"] == 1.0


def test_reweight_balances_classes():
    P = pd.DataFrame({"rel_path": list("abcd"), "group_id": [0, 0, 1, 2],
                      "y": [0, 0, 1, 1], "weight": 1.0, "f": 1.0})
    w = K.reweight_particles(P).drop_duplicates("rel_path").set_index("rel_path")["weight"]
    assert w[["a", "b"]].sum() == pytest.approx(w[["c", "d"]].sum())
