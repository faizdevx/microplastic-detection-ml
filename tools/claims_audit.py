"""Consistency audit of the repository's written claims. Exit code 1 on any violation.

    python tools/claims_audit.py

Checks (written to results/claims_audit.md):
  1. Every decimal number with two or more decimals in the README and the cards appears in a
     generated results file or a config (compared at its printed precision).
  2. The word "simulated" appears only where allowed.
  3. No attribution strings in tracked files or in the branch history.
  4. No YOLO mAP numbers unless trained-model result files exist.
  5. No micrometre values in generated image results while um_per_px is null.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SELF = "tools/claims_audit.py"
DOCS = ["README.md", "MODEL_CARD.md", "DATASET_CARD.md", "RELEASE_NOTES.md"]
NUM = re.compile(r"(?<![\w.\-])\d+\.\d{2,}(?![\w./]|\.\d)")
TEXT_EXT = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".cff", ".csv", ".txt", ".cfg", ".svg",
            ".ipynb", ".ini", ""}
ATTRIBUTION = [r"co-authored-by", r"generated with", r"claude", r"anthropic"]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf8", errors="replace").stdout


def tracked() -> list[str]:
    return [f for f in git("ls-files").splitlines() if f != SELF]


def read(rel: str) -> str | None:
    p = ROOT / rel
    if p.suffix.lower() not in TEXT_EXT or not p.exists():
        return None
    try:
        return p.read_text(encoding="utf8")
    except UnicodeDecodeError:
        return None


# ------------------------------------------------------------ 1. numbers -----------------
def universe() -> list[float]:
    nums: list[float] = []
    pat = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
    sources = (list((ROOT / "results").rglob("*.json")) + list((ROOT / "results").rglob("*.md"))
               + list((ROOT / "configs").glob("*.yaml")) + [ROOT / "docs" / "literature_params.csv"]
               + list((ROOT / "data" / "processed").glob("*.json")))
    for p in sources:
        if not p.exists() or p.name == "claims_audit.md":
            continue
        for m in pat.findall(p.read_text(encoding="utf8", errors="replace")):
            try:
                nums.append(float(m))
            except ValueError:
                pass
    return nums


def number_matches(x: float, decimals: int, uni: list[float]) -> bool:
    """Three or more decimals: within 0.0005. Two decimals: equal after rounding."""
    for cand in (x, x / 100.0, x * 100.0):
        for u in uni:
            if abs(u - cand) <= 0.0005:
                return True
            if decimals <= 2 and round(u, decimals) == round(cand, decimals):
                return True
    return False


def check_numbers() -> tuple[list[str], int]:
    uni = sorted(set(universe()))
    bad, checked = [], 0
    for rel in DOCS:
        text = read(rel)
        if text is None:
            continue
        in_code = False
        for ln, line in enumerate(text.splitlines(), 1):
            if line.startswith("```"):
                in_code = not in_code
            if in_code or "doi.org" in line or "zenodo" in line.lower() and "10.5281" in line:
                continue
            for m in NUM.finditer(line):
                s = m.group(0)
                checked += 1
                if not number_matches(float(s), len(s.split(".")[1]), uni):
                    bad.append(f"{rel}:{ln}: {s}  | {line.strip()[:90]}")
    return bad, checked


# ---------------------------------------------------------- 2. the word ------------------
def check_word() -> tuple[list[str], int]:
    bad, seen = [], 0
    for rel in tracked():
        text = read(rel)
        if text is None or rel.startswith("data/") or rel.endswith(".ipynb"):
            continue
        hits = [(i, ln) for i, ln in enumerate(text.splitlines(), 1) if re.search("simulat", ln, re.I)]
        seen += len(hits)
        if rel == "README.md":
            continue
        if rel == "DATASET_CARD.md":
            if len(hits) > 1:
                bad.append(f"{rel}: {len(hits)} lines mention the word (one allowed)")
            continue
        for i, ln in hits:
            if not re.search(r"source", ln, re.I):
                bad.append(f"{rel}:{i}: {ln.strip()[:100]}")
    return bad, seen


# ------------------------------------------------------ 3. attribution ------------------
def check_attribution() -> tuple[list[str], int]:
    bad = []
    pat = re.compile("|".join(ATTRIBUTION), re.I)
    files = [f for f in tracked() if read(f) is not None]
    for rel in files:
        for i, ln in enumerate(read(rel).splitlines(), 1):
            if pat.search(ln):
                bad.append(f"{rel}:{i}: {ln.strip()[:100]}")
    log = git("log", "--format=%H%n%an%n%ae%n%cn%n%ce%n%B%n--END--")
    for block in log.split("--END--"):
        if pat.search(block):
            bad.append("git log: " + block.strip().splitlines()[0][:12] + " contains an attribution string")
    return bad, len(files)


# ------------------------------------------------------------ 4. mAP ---------------------
def check_map() -> tuple[list[str], int]:
    res = ROOT / "results"
    trained = (res / "phase2_yolo_all_cv.json").exists() or any(res.glob("phase2_yolo_fold_*.json"))
    bad, seen = [], 0
    for rel in DOCS:
        text = read(rel) or ""
        for i, ln in enumerate(text.splitlines(), 1):
            if re.search(r"\bmAP", ln) and re.search(r"\d\.\d", ln):
                seen += 1
                if not trained:
                    bad.append(f"{rel}:{i}: {ln.strip()[:100]}")
    return bad, seen


# ------------------------------------------------------------ 5. um_per_px ---------------
def check_um() -> tuple[list[str], int]:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf8"))
    if cfg.get("um_per_px") is not None:
        return [], 0
    pat = re.compile(r"µm|\b\w+_um\b(?!_per)|\bmicrometre", re.I)
    files = (list((ROOT / "results").glob("phase2_*.json")) + list((ROOT / "results").glob("data_*.json"))
             + list((ROOT / "results").glob("deploy_*.json")) + [ROOT / "results" / "phase2_summary.md"])
    bad = []
    for p in files:
        if not p.exists():
            continue
        for i, ln in enumerate(p.read_text(encoding="utf8").splitlines(), 1):
            if pat.search(ln):
                bad.append(f"{p.relative_to(ROOT).as_posix()}:{i}: {ln.strip()[:100]}")
    return bad, len(files)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf8", errors="replace")
    checks = [
        ("Numbers in the documents match generated files", check_numbers),
        ("The word is only in allowed places", check_word),
        ("No attribution strings in files or history", check_attribution),
        ("No YOLO mAP numbers without trained-model results", check_map),
        ("No micrometre values while um_per_px is null", check_um),
    ]
    lines = ["# Claims audit", "", "Generated by `python tools/claims_audit.py`.", ""]
    failed = 0
    for title, fn in checks:
        bad, n = fn()
        failed += len(bad)
        lines.append(f"## {title}")
        lines.append(f"Checked {n} items; {len(bad)} violation(s).")
        lines += [f"- {b}" for b in bad] + [""]
        print(f"{'FAIL' if bad else 'ok  '} {title} ({n} checked)")
        for b in bad:
            print("   ", b)
    lines.append("Result: " + ("FAILED" if failed else "passed"))
    (ROOT / "results" / "claims_audit.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
