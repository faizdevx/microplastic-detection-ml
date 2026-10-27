import hashlib
import importlib.util

from mp.config import repo_root

_spec = importlib.util.spec_from_file_location("cve", repo_root() / "tools" / "claims_vs_evidence.py")
cve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cve)


def test_matches_uses_printed_precision_and_percentages():
    assert cve.matches(0.99, [0.9871], 2)
    assert not cve.matches(0.988, [0.9871], 3)
    assert cve.matches(98.71, [0.9871], 2)          # percentage against a fraction
    assert not cve.matches(98.75, [0.9871], 2)
    assert cve.decimals_of(5) == 0 and cve.decimals_of(0.988) == 3


def test_supported_requires_numbers_that_match():
    ok = {"status": "supported", "claim_numbers": [5],
          "evidence": [{"ref": "data/processed/splits.json::meta.n_folds"}]}
    assert cve.check_entry("x", ok)[1] == []
    bad = dict(ok, claim_numbers=[7])
    assert "matches no evidence" in cve.check_entry("x", bad)[1][0]
    no_numbers = {"status": "supported", "evidence": ok["evidence"]}
    assert "without claim_numbers" in cve.check_entry("x", no_numbers)[1][0]


def test_unknown_status_and_unresolvable_ref_are_reported():
    assert cve.check_entry("x", {"status": "maybe", "evidence": []})[1]
    probs = cve.check_entry("x", {"status": "contradicted", "evidence": [{"ref": "nope.json::a.b"}]})[1]
    assert "cannot resolve" in probs[0]


def test_resolve_kinds():
    assert cve.resolve("csv_rows:final_dataset.csv") == 2000
    assert cve.resolve("csv_class:final_dataset.csv:polymer:PVC") == 1000
    assert cve.resolve("grep:notebooks/00_legacy_exploration.ipynb::GroupKFold") == 0
    assert cve.resolve("grep:notebooks/00_legacy_exploration.ipynb::train_test_split") >= 1
    assert cve.resolve("results/phase2_holdout.json::metrics.confusion_matrix[0][0]") >= 0


def test_the_map_is_consistent_with_the_docx_and_the_docx_is_untouched():
    before = hashlib.sha256(cve.DOCX.read_bytes()).hexdigest()
    text, problems = cve.build()
    assert problems == []
    assert "Discrepancies to resolve" in text
    assert hashlib.sha256(cve.DOCX.read_bytes()).hexdigest() == before


def test_every_claim_has_a_known_status():
    claims, _ = cve.read_docx(cve.DOCX)
    import yaml

    mp = yaml.safe_load(cve.MAP.read_text(encoding="utf8"))
    assert claims and all(mp["claims"][c["id"]]["status"] in cve.STATUSES for c in claims)
