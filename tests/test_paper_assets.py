import re

import pandas as pd
import pytest

from mp.config import repo_root
from mp.paper import assets

pytestmark = pytest.mark.skipif(not (repo_root() / "results" / "phase5_variants.json").exists(),
                                reason="needs the generated results")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("paper_assets")
    old = assets.OUT
    assets.OUT = out
    try:
        assets.run()
    finally:
        assets.OUT = old
    return out


def test_every_number_equals_its_source(built):
    parsed = assets.parse_numbers_tex(built / "numbers.tex")
    assert set(parsed) == set(assets.NUMBERS)
    for name, (_f, _k, fmt) in assets.NUMBERS.items():
        src = assets.number_value(name)
        got = float(parsed[name])
        m = re.search(r"\.(\d+)f", fmt)
        decimals = int(m.group(1)) if m else 0
        assert abs(got - float(src)) <= 0.5 * 10 ** (-decimals) + 1e-12, name


def test_numbers_file_is_marked_generated(built):
    assert (built / "numbers.tex").read_text(encoding="utf8").startswith("% Generated")


def test_tables_exist_with_csv_and_latex(built):
    for name in assets.TABLES:
        csv = pd.read_csv(built / "tables" / f"{name}.csv")
        tex = (built / "tables" / f"{name}.tex").read_text(encoding="utf8")
        assert len(csv) > 0 and "\\begin{tabular}" in tex and f"\\label{{tab:{name}}}" in tex


def test_classifier_table_matches_markdown_rounding(built):
    df = pd.read_csv(built / "tables" / "classifier.csv", dtype=str)
    row = df[(df["Geometry"] == "hold-out") & (df["Model"] == "accuracy")].iloc[0]
    assert row["CV accuracy"] == f"{assets.number_value('HoldoutAccuracy'):.3f}"


def test_figures_exist_in_png_and_pdf(built):
    names = {p.stem for p in (built / "figures").glob("*.png")}
    assert {"controls_by_variant", "feature_sets", "regression_e1", "feature_importance",
            "angular_scattering"} <= names
    for n in names:
        assert (built / "figures" / f"{n}.pdf").exists()


def test_latex_escaping():
    df = pd.DataFrame({"a_b": ["x_y", "5 %"]})
    tex = assets.to_latex(df, "cap_1", "lab")
    assert "a\\_b" in tex and "x\\_y" in tex and "5 \\%" in tex
    assert re.search(r"\\caption\{cap\\_1\}", tex)


def test_no_title_is_baked_into_generated_figures():
    import inspect

    src = inspect.getsource(assets)
    assert "set_title(" not in src and "suptitle(" not in src
