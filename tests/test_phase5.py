import cv2
import numpy as np
import pandas as pd
import pytest

from mp.config import repo_root
from mp.eval import acquisition_probe as AP
from mp.eval import phase5 as P5
from mp.eval import phase5_report as R5
from mp.imaging import invariant as I
from mp.imaging import segment as sg
from mp.models import patch_cnn

SEG = {"channel": 0, "blur_sigma": 1.0, "threshold": "otsu", "min_signal": 8,
       "open_kernel": 3, "watershed_min_distance": 5, "min_area_px": 20, "drop_border": True}


def _textured(gain=1.0, size=(240, 320), seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(10, 20, (*size, 3)).astype(np.float32)
    for cx, cy, r in [(80, 120, 22), (200, 100, 18), (250, 180, 14)]:
        yy, xx = np.ogrid[:size[0], :size[1]]
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[m] = (np.array([200, 90, 15]) + rng.normal(0, 8, (m.sum(), 3)))
    return np.clip(img * gain, 0, 255).astype(np.uint8)


def test_v1_features_do_not_change_with_exposure_gain():
    a, b = _textured(1.0), _textured(0.5)
    la, lb = sg.segment(a, SEG), sg.segment(b, SEG)
    assert la.max() == lb.max() == 3
    fa, fb = I.invariant_features(la, a), I.invariant_features(lb, b)
    for c in ["v1_rel_R", "v1_rel_G", "v1_cv_R", "v1_rg", "v1_surr_R"]:
        assert fb[c].to_numpy() == pytest.approx(fa[c].to_numpy(), rel=0.08), c


def test_v1_has_no_absolute_intensity_columns():
    bad = [c for c in I.V1_FEATURES if c.startswith(("raw_", "norm_"))]
    assert bad == []
    assert not set(I.V1_EXTRA) & set(I.BASE_FEATURES)


def test_pseudo_particles_avoid_foreground_and_have_class_free_size():
    img = _textured()
    labels = sg.segment(img, SEG)
    pl = I.pseudo_labels(labels, np.random.default_rng(1))
    assert 1 <= pl.max() <= labels.max()
    near = cv2.dilate((labels > 0).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert not (near & (pl > 0)).any()
    areas = np.bincount(pl.ravel())[1:]
    assert areas.min() >= 0.9 * np.pi * 4**2 and areas.max() <= 1.1 * np.pi * 14**2


def test_pseudo_particles_are_reproducible():
    labels = sg.segment(_textured(), SEG)
    a = I.pseudo_labels(labels, np.random.default_rng(3))
    b = I.pseudo_labels(labels, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_augmentation_is_seeded_and_changes_exposure():
    img = _textured()
    a, pa = I.augment_array(img, np.random.default_rng(5))
    b, pb = I.augment_array(img, np.random.default_rng(5))
    assert np.array_equal(a, b) and pa == pb
    assert 0.5 <= pa["gain"] <= 2.0 and 0.7 <= pa["gamma"] <= 1.4
    assert a.dtype == np.uint8 and not np.array_equal(a, img)


def test_patch_is_masked_size_and_brightness_normalised():
    from skimage.measure import regionprops

    def patch(radius, value):
        img = np.zeros((200, 200, 3), np.uint8)
        cv2.circle(img, (100, 100), radius, (value, value // 2, 10), -1)
        labels = sg.segment(img, SEG)
        return I.make_patch(labels, img, regionprops(labels)[0])

    small, large = patch(12, 100), patch(40, 200)
    for p in (small, large):
        assert p.shape == (48, 48, 3) and p.dtype == np.uint8
        assert p[:4].max() == 0 and p[-4:].max() == 0 and p[:, :4].max() == 0  # outside the particle
    assert (small.any(axis=2).sum() / large.any(axis=2).sum()) == pytest.approx(1.0, rel=0.25)
    assert small.max() == pytest.approx(large.max(), abs=15)


def test_patch_cnn_trains_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 255, (80, 48, 48, 3), dtype=np.uint8)
    y = rng.integers(0, 2, 80)
    net = patch_cnn.fit(X, y, np.ones(80), epochs=1, seed=0)
    prob, emb = patch_cnn.predict(net, X[:10])
    assert prob.shape == (10,) and emb.shape == (10, 64) and ((prob >= 0) & (prob <= 1)).all()


def _images(n=40):
    rows = [{"rel_path": f"{'P' if i % 2 else 'N'}/{i}.jpg", "y": i % 2, "group_id": i // 2,
             "fold": i % 5, "class": "NYLON" if i % 2 else "PVC", "width": 640, "height": 480}
            for i in range(n)]
    return pd.DataFrame(rows)


def test_image_weights_balance_classes_and_groups():
    imgs = _images()
    imgs = imgs.drop(index=[2])  # one group now has a single image
    w = P5.image_weights(imgs)
    assert w.mean() == pytest.approx(1.0)
    by_class = w.groupby(imgs.set_index("rel_path")["y"]).sum()
    assert by_class[0] == pytest.approx(by_class[1])


def test_permuted_labels_are_constant_within_group_and_keep_counts():
    imgs = _images()
    ym = P5.permuted_labels(imgs, 1)
    s = pd.Series(ym)
    g = imgs.set_index("rel_path")["group_id"]
    assert (s.groupby(g).nunique() == 1).all()
    assert s.sum() == imgs["y"].sum() or abs(s.sum() - imgs["y"].sum()) <= len(imgs) // 2


def test_verdict_rules():
    def r(cv, a, b, ps):
        return {"cv": {"accuracy": cv}, "cross_resolution": {"a_to_b": {"balanced_accuracy": a},
                                                              "b_to_a": {"balanced_accuracy": b}},
                "pseudo_particles": {"balanced_accuracy": ps}}

    res = {"baseline": r(0.97, 0.80, 0.80, 0.90), "V1_local_normalisation": r(0.97, 0.90, 0.90, 0.60),
           "V2_exposure_randomisation": r(0.97, 0.80, 0.81, 0.90),
           "V3_patch_cnn": r(0.97, 0.70, 0.70, 0.95)}
    cfg = {"verdict": {"cross_res_min_gain": 0.02, "pseudo_min_drop": 0.05}}
    v = R5.verdict(res, cfg)
    assert v["V1_local_normalisation"]["overall"] == "improves"
    assert v["V2_exposure_randomisation"]["overall"] == "no clear improvement"
    assert v["V3_patch_cnn"]["overall"] == "worse"


def test_probe_helpers():
    assert AP._chance(np.array([0, 0, 1])) == pytest.approx(2 / 3)
    assert np.isnan(AP._cv_accuracy(np.zeros((5, 2)), np.array([0, 1, 0, 1, 0]), np.arange(5), 0))
    oof = pd.DataFrame({"rel_path": list("abcdef"), "class": ["PVC"] * 3 + ["NYLON"] * 3,
                        "prob": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]})

    class D:
        bright = pd.Series({"a": 1, "b": 2, "c": 3, "d": 1, "e": 2, "f": 3}, dtype=float)

    c = AP.prediction_coupling(oof, D())
    assert c["PVC"] == pytest.approx(1.0) and c["mean_abs"] == pytest.approx(1.0)


@pytest.mark.skipif(not (repo_root() / "data" / "phase5" / "real.csv.gz").exists(),
                    reason="needs `mp phase5-features`")
def test_tables_never_contain_holdout_images():
    from mp.config import load_config
    from mp.data.dataset import load_pool

    pool = load_pool(load_config())
    hold = set(pool.holdout()["rel_path"])
    data = P5.load_data(with_patches=False)
    for t in (data.real, data.pseudo, data.aug):
        assert not hold & set(t["rel_path"])
    assert not hold & set(data.images["rel_path"])
