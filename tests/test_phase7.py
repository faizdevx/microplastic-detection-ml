import numpy as np
import pandas as pd
import pytest

from mp.annotate import auto
from mp.eval import yolo_controls as Y


def _images(n_per=(30, 10, 25, 5)):
    rows = []
    for (cls, w, h), n in zip([("PVC", 640, 480), ("PVC", 1280, 960), ("NYLON", 640, 480),
                               ("NYLON", 742, 556)], n_per):
        for i in range(n):
            rows.append({"rel_path": f"{cls}/{w}_{i}.jpg", "class": cls, "width": w, "height": h,
                         "group_id": f"{cls}{w}_{i // 2}", "fold": i % 5})
    df = pd.DataFrame(rows)
    df.loc[df.index % 7 == 0, "fold"] = -1
    return df


def test_review_pick_is_stratified_cv_only_one_per_group():
    imgs = _images()
    p = auto.review_pick(imgs, n=12, seed=1)
    assert len(p) == 12
    assert (p["fold"] != -1).all() and p["group_id"].is_unique
    strata = p["class"] + p["width"].astype(str)
    assert set(strata) == {"PVC640", "PVC1280", "NYLON640", "NYLON742"}
    assert auto.review_pick(imgs, n=12, seed=1)["rel_path"].tolist() == p["rel_path"].tolist()


def test_coco_labels_round_trip():
    coco = {"images": [{"id": 1, "file_name": "NYLON/a.jpg", "width": 100, "height": 50}],
            "annotations": [{"image_id": 1, "category_id": 2,
                             "segmentation": [[10, 10, 50, 10, 50, 40, 10, 40]]}]}
    (line,) = Y._coco_labels(coco)["NYLON/a.jpg"]
    v = line.split()
    assert v[0] == "1" and float(v[1]) == pytest.approx(0.1) and float(v[6]) == pytest.approx(0.8)


def test_mcnemar_exact():
    a = np.array([True] * 10 + [False] * 2 + [True] * 5)
    b = np.array([False] * 10 + [True] * 2 + [True] * 5)
    m = Y.mcnemar(a, b)
    assert m["only_first_correct"] == 10 and m["only_second_correct"] == 2
    assert m["p_value"] == pytest.approx(0.0386, abs=1e-3)       # exact binomial, 2 of 12
    assert Y.mcnemar(a, a)["p_value"] == 1.0


def test_group_bootstrap_interval_contains_difference():
    df = pd.DataFrame({"group_id": np.repeat(np.arange(20), 3),
                       "yolo_ok": np.tile([True, True, False], 20),
                       "cls_ok": np.tile([True, False, False], 20)})
    r = Y.group_bootstrap(df, n=500)
    assert r["difference"] == pytest.approx(1 / 3)
    assert r["ci95"][0] <= r["difference"] <= r["ci95"][1] and r["n_groups"] == 20


def test_normalise_stretches_each_channel():
    img = np.dstack([np.linspace(50, 100, 400).reshape(20, 20)] * 3).astype(np.uint8)
    out = Y.normalise(img)
    assert out.min() == 0 and out.max() == 255


def test_disc_image_removes_particles_and_paints_discs():
    rng = np.random.default_rng(0)
    img = np.full((120, 160, 3), 20, np.uint8)
    lab = np.zeros((120, 160), np.int32)
    lab[50:60, 70:80] = 1
    img[lab > 0] = 240
    out = Y.disc_image(img, lab, np.array([90, 90, 90], np.uint8), rng)
    assert out[55, 75].max() < 120                      # particle gone
    assert (np.all(out == 90, axis=2)).sum() > 20       # one disc painted in the fixed colour
