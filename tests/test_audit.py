import importlib.util
import subprocess
import sys

from mp.config import repo_root

_spec = importlib.util.spec_from_file_location("claims_audit", repo_root() / "tools" / "claims_audit.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def test_number_pattern_ignores_versions_and_dois():
    found = audit.NUM.findall("v0.2.0 doi 10.5281/zenodo.21073653 accuracy 0.987 and 1.5 and 2.50")
    assert found == ["0.987", "2.50"]


def test_number_matching_precision_rules():
    uni = [0.8071, 0.9871]
    assert audit.number_matches(0.987, 3, uni)        # within 0.0005
    assert not audit.number_matches(0.990, 3, uni)    # three decimals must be close
    assert audit.number_matches(0.81, 2, uni)         # two decimals: equal after rounding
    assert not audit.number_matches(0.55, 2, uni)


def test_audit_passes_on_this_repository():
    out = subprocess.run([sys.executable, str(repo_root() / "tools" / "claims_audit.py")],
                         capture_output=True, text=True, encoding="utf8", errors="replace")
    assert out.returncode == 0, out.stdout
