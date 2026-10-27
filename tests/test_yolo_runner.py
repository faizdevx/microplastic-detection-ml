import importlib.util

from mp.config import load_named, repo_root

_spec = importlib.util.spec_from_file_location("run_yolo_all", repo_root() / "tools" / "run_yolo_all.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def test_runner_covers_every_dataset_key_once():
    expected = ({f"fold_{k}" for k in range(5)} | {f"res_matched_{k}" for k in range(5)}
                | {"cross_a_to_b", "cross_b_to_a", "all_cv"})
    assert sorted(runner.KEYS) == sorted(expected) and len(runner.KEYS) == len(set(runner.KEYS))


def test_result_file_names_match_what_the_audit_looks_for():
    names = {f"phase2_yolo_{k}.json" for k in runner.KEYS}
    assert "phase2_yolo_all_cv.json" in names and "phase2_yolo_fold_0.json" in names


def test_config_keeps_full_schedule_and_defines_a_short_one():
    y = load_named("yolo_seg")
    assert y["epochs"] == 150 and y["patience"] == 30
    short = y["schedule"]["short"]
    assert short["epochs"] < y["epochs"] and short["patience"] < y["patience"]
    assert y["batch"] <= 8  # 4 GB GPU
