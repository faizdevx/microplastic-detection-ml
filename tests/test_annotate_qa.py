import json

import numpy as np
import pytest

from mp.annotate import auto

SQ = lambda x, y, s=30: np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], np.float32)  # noqa: E731


def test_match_instances_counts_missed_and_spurious():
    a = [SQ(10, 10), SQ(100, 100)]          # second auto particle is not in the corrected set
    c = [SQ(11, 11), SQ(200, 20)]           # second corrected particle was missed by auto
    m = auto.match_instances(a, c, 240, 320)
    assert m["matched"] == 1 and m["missed"] == 1 and m["spurious"] == 1
    assert m["matched_ious"][0] > 0.7


def test_match_instances_identical_sets_have_no_errors():
    polys = [SQ(10, 10), SQ(100, 100)]
    m = auto.match_instances(polys, polys, 240, 320)
    assert m["missed"] == 0 and m["spurious"] == 0 and m["matched"] == 2


def _coco(tmp_path, corrected):
    img = {"id": 1, "file_name": "PVC/x.jpg", "width": 320, "height": 240}
    anns = [{"id": i + 1, "image_id": 1, "category_id": 1, "segmentation": [p.reshape(-1).tolist()]}
            for i, p in enumerate(corrected)]
    path = tmp_path / "coco.json"
    path.write_text(json.dumps({"images": [img], "annotations": anns,
                                "categories": [{"id": 1, "name": "pvc"}]}))
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("MP_ROOT", str(tmp_path))
    (tmp_path / "results").mkdir()
    return tmp_path


def test_qa_report_end_to_end(root):
    auto.write_yolo_txt(root / "data/annotations/auto/PVC/x.txt", [SQ(10, 10), SQ(100, 100)], 0, 320, 240)
    coco = _coco(root, [SQ(11, 11), SQ(200, 20)])
    cfg = {"paths": {"results": "results"}}
    r = auto.qa_report(coco, cfg)
    assert r["n_corrected_images"] == 1
    assert r["missed_particles"] == 1 and r["spurious_particles"] == 1
    assert r["share_images_with_missed_or_spurious"] == 1.0
    assert r["count_error_mae"] == 0.0           # two auto, two corrected
    assert 0 < r["semantic_iou_mean"] < 1
    saved = json.loads((root / "results" / "annotation_quality.json").read_text())
    assert saved["per_image"][0]["file"] == "PVC/x.jpg"


def test_qa_report_perfect_agreement(root):
    polys = [SQ(10, 10), SQ(100, 100)]
    auto.write_yolo_txt(root / "data/annotations/auto/PVC/x.txt", polys, 0, 320, 240)
    r = auto.qa_report(_coco(root, polys), {"paths": {"results": "results"}})
    assert r["share_images_with_missed_or_spurious"] == 0.0 and r["semantic_iou_mean"] > 0.99
