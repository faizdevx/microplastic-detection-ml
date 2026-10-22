"""Offline tests for the Phase 1 data pipeline (no dataset download)."""
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mp.data import fetch, groups, splits
from mp.imaging import calibrate_scale as cs


def test_md5_mismatch_deletes_file(tmp_path):
    f = tmp_path / "a.zip"
    f.write_bytes(b"corrupted")
    assert fetch.verify_md5(f, "0" * 32) is False
    assert not f.exists()


def test_md5_match_keeps_file(tmp_path):
    f = tmp_path / "a.zip"
    f.write_bytes(b"hello")
    assert fetch.verify_md5(f, fetch.md5_of(f)) is True
    assert f.exists()


def test_zip_slip_rejected(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("ok/a.txt", "x")
        zf.writestr("../escaped.txt", "x")
    target = tmp_path / "out"
    with pytest.raises(ValueError, match="unsafe path"):
        fetch.safe_extract(z, target)
    assert not (tmp_path / "escaped.txt").exists()
    assert not (target / "ok").exists()  # nothing extracted before the check


def _blob_image(seed: int, size=(96, 128)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros(size, dtype=np.float32)
    yy, xx = np.mgrid[: size[0], : size[1]]
    for _ in range(6):
        cy, cx = rng.integers(10, size[0] - 10), rng.integers(10, size[1] - 10)
        r = rng.integers(5, 14)
        img += 200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2))
    return img


def _save(arr, path):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def test_hash_groups_cluster_copies_not_unrelated(tmp_path):
    base = _blob_image(1)
    other = _blob_image(2)
    rng = np.random.default_rng(0)
    variants = {
        "base": base,
        "flip": base[:, ::-1],
        "bright": base * 1.15 + 8,
        "noisy": base + rng.normal(0, 4, base.shape),
        "other": other,
    }
    paths = []
    for name, arr in variants.items():
        p = tmp_path / f"{name}.png"
        _save(arr, p)
        paths.append(p)
    hashes = groups.compute_hashes(paths)
    ph = [h[0] for h in hashes]
    d = lambda a, b: groups.hamming(ph[a], ph[b])  # noqa: E731
    assert d(0, 2) <= 6 and d(0, 3) <= 6  # brightness / noise copies
    assert d(0, 4) > 6  # unrelated image stays apart

    edges, exact = groups.candidate_pairs(ph, [f"s{i}" for i in range(5)], 10)
    gid, cross = groups.cluster(5, ["A"] * 5, edges, exact, 6)
    assert gid[0] == gid[2] == gid[3]
    assert gid[4] != gid[0]
    assert cross == []


def test_cluster_never_merges_across_classes():
    ph = [0b1010, 0b1010, 0b0101]
    shas = ["a", "b", "c"]
    edges, exact = groups.candidate_pairs(ph, shas, 4)
    gid, cross = groups.cluster(3, ["PVC", "NYLON", "PVC"], edges, exact, 4)
    assert gid[0] != gid[1]
    assert any({i, j} == {0, 1} for i, j, _ in cross)


def test_exact_sha_pairs_merge_at_any_threshold():
    ph = [0, (1 << 64) - 1]  # maximally different hashes
    edges, exact = groups.candidate_pairs(ph, ["same", "same"], 0)
    gid, _ = groups.cluster(2, ["PVC", "PVC"], edges, exact, 0)
    assert gid[0] == gid[1]


def test_bktree_matches_bruteforce():
    rng = np.random.default_rng(3)
    vals = [int(v) for v in rng.integers(0, 2**63, 200)]
    tree = groups.BKTree()
    for i, v in enumerate(vals):
        tree.add(v, i)
    q = vals[5] ^ 0b111
    got = {i for i, _ in tree.query(q, 20)}
    want = {i for i, v in enumerate(vals) if groups.hamming(q, v) <= 20}
    assert got == want


def _synthetic_manifest(n_groups=60, per_group=4, sessions=3):
    rows = []
    for c_i, cls in enumerate(["PVC", "NYLON"]):
        for g in range(n_groups):
            gid = c_i * n_groups + g
            for k in range(per_group):
                rows.append(
                    {
                        "rel_path": f"{cls}/{gid}_{k}.png",
                        "class": cls,
                        "group_id": gid,
                        "session_id": f"{cls}_s{g % sessions}",
                    }
                )
    return pd.DataFrame(rows)


CFG = {
    "splits": {"seed": 42, "holdout_fraction": 0.15, "n_folds": 5, "split_version": "1"},
    "groups": {"hash_threshold": 6},
}


def test_splits_invariants_and_loso():
    df = _synthetic_manifest()
    out = splits.make_splits(df, CFG, "deadbeef")
    splits.validate_splits(df, out)  # raises on any leak
    assert out["meta"]["groups_csv_sha256"] == "deadbeef"
    assert out["leave_one_session_out"] is not None
    assert len(out["leave_one_session_out"]) == 6


def test_validate_detects_group_leak():
    df = _synthetic_manifest()
    out = splits.make_splits(df, CFG)
    gid = dict(zip(df["rel_path"], df["group_id"]))
    victim = next(r for r in out["holdout"])
    same_group = [r for r, g in gid.items() if g == gid[victim] and r != victim][0]
    out["cv_fold_of"][same_group] = 0
    out["holdout"] = [r for r in out["holdout"] if r != same_group]
    with pytest.raises(AssertionError):
        splits.validate_splits(df, out)


def test_loso_skipped_with_few_sessions():
    df = _synthetic_manifest(sessions=2)
    out = splits.make_splits(df, CFG)
    assert out["leave_one_session_out"] is None


def test_infer_sessions_by_filename_gap():
    df = pd.DataFrame(
        {
            "class": ["PVC"] * 6,
            "width": [640] * 6,
            "height": [480] * 6,
            "rel_path": [f"PVC/pvc_{n:04d}.jpg" for n in (1, 2, 3, 200, 201, 202)],
            "exif_datetime": [""] * 6,
        }
    )
    s = groups.infer_sessions(df, 60, 25)
    assert s.nunique() == 2


def test_infer_sessions_filename_gap_disabled_by_default_none():
    df = pd.DataFrame(
        {
            "class": ["PVC"] * 4,
            "width": [640] * 4,
            "height": [480] * 4,
            "rel_path": [f"PVC/pvc_{n:04d}.jpg" for n in (1, 2, 300, 301)],
            "exif_datetime": [""] * 4,
        }
    )
    assert groups.infer_sessions(df, 60, None).nunique() == 1


def test_calibrate_recovers_known_spacing(tmp_path):
    period_px = 12.5
    x = np.arange(400)
    row = 128 + 100 * np.sin(2 * np.pi * x / period_px)
    img = np.tile(row, (200, 1)).astype(np.uint8)
    p = tmp_path / "micrometer.png"
    Image.fromarray(img).save(p)
    got = cs.estimate_um_per_px(p, spacing_um=10.0)
    assert got == pytest.approx(10.0 / period_px, rel=0.02)


def test_write_um_per_px_keeps_comments(tmp_path):
    cfgf = tmp_path / "data.yaml"
    cfgf.write_text("# keep me\num_per_px: null\nseed: 1\n", encoding="utf8")
    cs.write_um_per_px(0.5, cfgf)
    text = cfgf.read_text(encoding="utf8")
    assert "# keep me" in text and "um_per_px: 0.500000" in text


def test_require_um_per_px_warns_when_unset():
    with pytest.warns(UserWarning, match="not calibrated"):
        assert cs.require_um_per_px({"um_per_px": None}) is None
