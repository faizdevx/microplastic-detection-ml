import numpy as np
import pandas as pd
import pytest

from mp.models import regression as R


def _table():
    rows = []
    rng = np.random.default_rng(0)
    for gid, off in (("g0", 60), ("g1", 80)):
        for rep in range(3):
            rows.append({"polymer": "blank", "conc_mg_L": 0.0, "geometry_id": gid,
                         "adc_model": "mcp3008", "replicate": rep, "saturated_any": False,
                         **{f"S{i}": off for i in range(1, 5)}})
            for c in (0.5, 1.0, 2.0):
                rows.append({"polymer": "PVC", "conc_mg_L": c, "geometry_id": gid,
                             "adc_model": "mcp3008", "replicate": rep, "saturated_any": False,
                             **{f"S{i}": off + 10 * c * i + rng.normal(0, 0.1) for i in range(1, 5)}})
    return pd.DataFrame(rows)


def test_dark_subtract_uses_geometry_blank():
    d = R.dark_subtract(_table())
    assert d[d["polymer"] == "blank"][R.DS_COLS].abs().max().max() < 1e-9
    assert d[(d.geometry_id == "g1") & (d.conc_mg_L == 1.0)]["dS1"].mean() == pytest.approx(10, abs=0.2)


def test_dark_subtract_with_fixed_baseline_keeps_offset_shift():
    d = R.dark_subtract(_table(), baseline_geometry="g0")
    shift = d[(d.geometry_id == "g1") & (d.conc_mg_L == 1.0)]["dS1"].mean()
    assert shift == pytest.approx(10 + 20, abs=0.2)  # g1 blank is 20 counts higher


def test_features_are_finite_and_complete():
    f = R.make_features(R.dark_subtract(_table()))
    assert np.isfinite(f.to_numpy()).all()
    assert {"S_mean", "S_std", "fwd_back", "r14", "r24", "r13", "logS1"} <= set(f.columns)


def test_training_rows_drop_blanks_and_saturated():
    t = _table()
    t.loc[t.index[1], "saturated_any"] = True
    d = R.training_rows(R.dark_subtract(t), {"exclude_blank_from_training": True, "drop_saturated": True})
    assert (d["polymer"] != "blank").all() and not d["saturated_any"].any()


def test_conformal_quantile_and_metrics():
    r = np.arange(1, 100) / 100.0
    assert R.conformal_q(r, 0.9) == pytest.approx(0.9, abs=0.02)
    m = R.metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert m["r2"] == 1.0 and m["rmse"] == 0.0


def test_measured_data_stubs_raise_clear_error():
    d = _table().assign(source="simulated")
    with pytest.raises(NotImplementedError, match="measured"):
        R.run_e3(d)
    with pytest.raises(NotImplementedError, match="not implemented"):
        R.run_e3(d.assign(source="measured"))
