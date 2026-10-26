"""Offline tests for the multi-angle scattering model."""
import copy
import csv

import numpy as np
import pandas as pd
import pytest

from mp.config import repo_root
from mp.scatter import detector as det
from mp.scatter import generate as G
from mp.scatter import mie
from mp.scatter import optics as O


@pytest.fixture(scope="module")
def cfg():
    c = G.load_scatter_config()
    c["size_distribution"]["nodes"] = 8
    c["monte_carlo"] = {"photons": 3000, "max_order": 6}
    return c


# ----------------------------------------------------------------------- Mie ----------
@pytest.mark.parametrize("m,x,qext", [(1.5 + 0j, 10.0, 2.8820), (1.5 + 0j, 1.0, 0.2151)])
def test_published_efficiencies(m, x, qext):
    """Wiscombe (1979) / van de Hulst reference values."""
    assert mie.cross_sections(m, x)["q_ext"] == pytest.approx(qext, abs=2e-4)


def test_optical_theorem_forward_amplitude():
    """Q_ext = 4/x^2 Re S(0): checks the amplitude routine independently of miepython."""
    m, x = mie.relative_index(1.54, 0.0, 1.331), 120.0
    s1, _ = mie.amplitudes(m, x, [0.0])
    assert 4 / x**2 * s1[0].real == pytest.approx(mie.cross_sections(m, x)["q_ext"], rel=2e-3)


def test_phase_function_integrates_to_one():
    m, x = mie.relative_index(1.54, 0.0, 1.331), 80.0
    th = np.concatenate([np.linspace(0, 3, 3001), np.linspace(3, 180, 6001)[1:]])
    p = mie.phase_function(m, x, th)
    integral = 2 * np.pi * np.trapezoid(p * np.sin(np.radians(th)), np.radians(th))
    assert integral == pytest.approx(1.0, rel=2e-3)


def test_extinction_paradox_large_x():
    m = mie.relative_index(1.54, 0.0, 1.331)
    assert mie.cross_sections(m, 600.0)["q_ext"] == pytest.approx(2.0, abs=0.1)


def test_rayleigh_angular_shape():
    m = complex(1.2, 0.0)
    th = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
    i = mie.angular_intensity(m, 0.01, th)["unpolarised"]
    assert i / i[0] == pytest.approx((1 + np.cos(np.radians(th)) ** 2) / 2, rel=1e-3)


def test_forward_peak_at_large_x():
    m, x = mie.relative_index(1.54, 0.0, 1.331), 100.0
    i = mie.angular_intensity(m, x, [20.0, 135.0])["unpolarised"]
    assert i[0] > 100 * i[1]


def test_amplitude_matches_miepython_up_to_normalisation():
    import miepython

    m, x = mie.relative_index(1.54, 0.0, 1.331), 30.0
    th = np.array([10.0, 40.0, 70.0, 120.0])
    mine = mie.angular_intensity(m, x, th)["unpolarised"]
    s1, s2 = miepython.S1_S2(m, x, np.cos(np.radians(th)))
    ref = 0.5 * (np.abs(s1) ** 2 + np.abs(s2) ** 2)
    ratio = mine / ref
    assert ratio == pytest.approx(ratio[0], rel=1e-6)  # one constant factor


# ----------------------------------------------------------------- optics -------------
def test_mass_to_number_monodisperse_by_hand():
    n = O.number_density(1.0, 1.38, np.array([50.0]))
    expected = 1.0 / (1.38e6 * np.pi / 6 * (50e-6) ** 3)  # 1 mg/L = 1 g/m^3
    assert n == pytest.approx(expected, rel=1e-12)
    assert n == pytest.approx(1.107e7, rel=1e-3)


def test_lognormal_nodes_have_requested_median():
    d = O.lognormal_nodes(50.0, 0.3, 41)
    assert np.median(d) == pytest.approx(50.0, rel=1e-6)


def _popt(cfg, polymer="PVC", mode="first_order"):
    rig = O.rig_from_config(cfg)
    nodes = O.lognormal_nodes(50.0, 0.3, 8)
    p = cfg["polymers"][polymer]
    return rig, nodes, p, O.particle_optics(p["n"], p["k"], nodes, rig, with_phase=True)


def test_signal_linear_at_low_concentration(cfg):
    rig, nodes, p, popt = _popt(cfg)
    conc = np.linspace(0.1, 1.0, 10)
    quiet = {k: False for k in cfg["noise"]}
    volts = []
    for c in conc:
        pw = O.detector_power(c, popt, rig, p["density_g_cm3"], nodes, "first_order")["P_W"]
        rng = np.random.default_rng(0)
        volts.append(det.readings(pw, rig, 0.0, G.adc_settings(cfg), quiet, 0.0, 293.0, rng)["V"])
    volts = np.array(volts)
    for d in range(2):  # the two forward detectors carry measurable signal
        r = np.corrcoef(conc, volts[:, d])[0, 1]
        assert r**2 > 0.999


def test_high_concentration_departs_from_linear_and_changes_ratios(cfg):
    rig, nodes, p, popt = _popt(cfg)
    kw = dict(rig=rig, density_g_cm3=p["density_g_cm3"], d_nodes_um=nodes)
    low = O.detector_power(1.0, popt, mode="monte_carlo", mc_cfg=cfg["monte_carlo"], seed=1, **kw)
    high = O.detector_power(1000.0, popt, mode="monte_carlo", mc_cfg=cfg["monte_carlo"], seed=1, **kw)
    assert high["tau"] > 0.1
    per_conc_low, per_conc_high = low["P_W"] / 1.0, high["P_W"] / 1000.0
    dev = np.abs(per_conc_high / per_conc_low - 1.0)
    assert dev.max() > 0.1  # signal per unit concentration is no longer constant
    ratio = (high["P_W"][0] / high["P_W"][3]) / (low["P_W"][0] / low["P_W"][3])
    assert abs(ratio - 1.0) > 0.05  # forward/backward ratio changes


def test_monte_carlo_matches_single_scattering_at_low_depth(cfg):
    rig, nodes, p, popt = _popt(cfg)
    kw = dict(rig=rig, density_g_cm3=p["density_g_cm3"], d_nodes_um=nodes)
    mc_cfg = {"photons": 20000, "max_order": 4}
    mc = O.detector_power(0.5, popt, mode="monte_carlo", mc_cfg=mc_cfg, seed=3, **kw)["P_W"]
    an = O.detector_power(0.5, popt, mode="first_order", **kw)["P_W"]
    assert mc[:2] == pytest.approx(an[:2], rel=0.08)


def test_attenuation_modes_agree_without_attenuation(cfg):
    rig, nodes, p, popt = _popt(cfg)
    kw = dict(rig=rig, density_g_cm3=p["density_g_cm3"], d_nodes_um=nodes)
    a = O.detector_power(1e-3, popt, mode="none", **kw)["P_W"]
    b = O.detector_power(1e-3, popt, mode="first_order", **kw)["P_W"]
    assert a == pytest.approx(b, rel=1e-4)


# ---------------------------------------------------------------- detector ------------
def test_adc_quantisation_and_clipping():
    c, sat = det.adc_counts(np.array([-0.1, 0.0, 1.65, 3.3, 5.0]), 10, 3.3)
    assert list(c) == [0, 0, 512, 1023, 1023]
    assert list(sat) == [False, False, False, True, True]
    step = 3.3 / 1024
    c1, _ = det.adc_counts(np.array([step * 0.99, step * 1.01]), 10, 3.3)
    assert list(c1) == [0, 1]
    c2, _ = det.adc_counts(np.array([4.0, 9.0]), 15, 4.096)
    assert list(c2) == [32000, 32767]


def test_saturated_any_flag_in_generated_rows(cfg):
    c = copy.deepcopy(cfg)
    c["detector"]["tia_Mohm"] = 2000.0  # huge gain forces saturation
    c["concentration"]["levels"] = 3
    c["geometry"]["samples"] = 0
    c["monte_carlo"]["photons"] = 500
    df = G.generate(c, mode="first_order", replicates=2, progress=False)
    assert df["saturated_any"].any()
    flagged = df[df["saturated_any"]]
    assert (flagged[["S1", "S2", "S3", "S4"]].max(axis=1) == 1023).all()


def _tiny(cfg, seed):
    c = copy.deepcopy(cfg)
    c["seed"] = seed
    c["concentration"]["levels"] = 3
    c["geometry"]["samples"] = 1
    c["size_distribution"]["options"] = c["size_distribution"]["options"][:1]
    c["monte_carlo"]["photons"] = 500
    return G.generate(c, mode="first_order", replicates=2, progress=False)


def test_same_seed_is_identical_different_seed_differs(cfg):
    a, b, c = _tiny(cfg, 5), _tiny(cfg, 5), _tiny(cfg, 6)
    pd.testing.assert_frame_equal(a, b)
    assert not a[["S1", "S2"]].equals(c[["S1", "S2"]])


def test_output_schema_and_source_column(cfg):
    df = _tiny(cfg, 1)
    need = {"sample_id", "source", "polymer", "conc_mg_L", "d_median_um", "sigma_ln",
            "replicate", "geometry_id", "theta_1", "theta_4", "P1_W", "P4_W", "V1", "V4",
            "S1", "S4", "tau", "saturated_any", "config_sha256", "seed"}
    assert need <= set(df.columns)
    assert (df["source"] == "simulated").all()
    assert (df[df["polymer"] == "blank"]["conc_mg_L"] == 0).all()


def test_config_constants_match_literature_table(cfg):
    lit = {r["name"]: r for r in csv.DictReader(open(repo_root() / "docs" / "literature_params.csv"))}
    checks = {
        ("medium", "n"): "n_water_650nm",
    }
    assert cfg["medium"]["n"] == float(lit["n_water_650nm"]["value"])
    assert cfg["polymers"]["PVC"]["n"] == float(lit["n_pvc_650nm"]["value"])
    assert cfg["polymers"]["NYLON"]["n"] == float(lit["n_nylon6_650nm"]["value"])
    assert cfg["polymers"]["PVC"]["density_g_cm3"] == float(lit["rho_pvc"]["value"])
    assert cfg["polymers"]["NYLON"]["density_g_cm3"] == float(lit["rho_nylon6"]["value"])
    assert cfg["detector"]["responsivity_A_W"] == float(lit["bpw34_responsivity_650nm"]["value"])
    assert cfg["detector"]["area_mm2"] == float(lit["bpw34_active_area"]["value"])
    assert checks
