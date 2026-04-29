#!/usr/bin/env python
"""Run reference YAML parsers against tests.jsonl and report agreement.

Two purposes:

1. Extraction integrity check — recompute each test's label from upstream's
   per-test-dir `error`-file convention and confirm it matches what
   tests.jsonl says. Any mismatch means setup.sh's generator drifted from
   the corpus structure.

2. Reference baselines — for each available parser, compute the confusion
   matrix and MCC against the corpus labels. Useful for understanding the
   benchmark's practical ceiling and for detecting label drift on pin
   bumps (a sudden ~10pp swing from a known-baseline parser is a flag).

The parsers are all optional; missing ones are reported and skipped, not
hard-required. Install hints:

    pip install pyyaml ruamel.yaml      # Python parsers
    brew install libfyaml               # fy-tool, the strict-1.2 reference

Numbers in reference_results.md are produced by this script.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # data/specs/yaml-1.2/
TESTS_PATH = HERE / "tests.jsonl"


def mcc(tp: int, fn: int, fp: int, tn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom else 0.0


def confusion(results: list[tuple[bool, bool]]) -> tuple[int, int, int, int]:
    tp = sum(1 for exp, act in results if exp and act)
    fn = sum(1 for exp, act in results if exp and not act)
    fp = sum(1 for exp, act in results if not exp and act)
    tn = sum(1 for exp, act in results if not exp and not act)
    return tp, fn, fp, tn


def load_tests() -> list[dict]:
    if not TESTS_PATH.exists():
        sys.exit(f"tests.jsonl not found at {TESTS_PATH} — run setup.sh first")
    return [json.loads(l) for l in TESTS_PATH.read_text().splitlines()]


# --- extraction integrity check ----------------------------------------------

def integrity_check(tests: list[dict]) -> int:
    """Recompute labels from the cache directly and compare to tests.jsonl.
    Returns the number of mismatches; 0 means setup.sh's labels are faithful
    to upstream."""
    cache = HERE / "tests"  # symlink to .cache/yaml-test-suite/
    if not cache.exists():
        print(f"[skip] integrity check: {cache} symlink missing (run setup.sh)")
        return -1
    mismatches = []
    for t in tests:
        test_dir = cache / t["id"]
        upstream_invalid = (test_dir / "error").exists()
        upstream_label = "invalid" if upstream_invalid else "valid"
        if upstream_label != t["expected"]:
            mismatches.append((t["id"], t["expected"], upstream_label))
    print(f"Extraction integrity: {len(tests)} tests, "
          f"{len(mismatches)} label mismatches against upstream `error`-file presence")
    for m in mismatches[:20]:
        print(f"  {m[0]}: tests.jsonl={m[1]}, upstream={m[2]}")
    return len(mismatches)


# --- parser drivers ----------------------------------------------------------

def run_pyyaml(tests: list[dict]) -> tuple[str, list[tuple[bool, bool]]] | None:
    try:
        import yaml as pyyaml
    except ImportError:
        print("[skip] PyYAML not installed (pip install pyyaml)")
        return None
    name = f"PyYAML {pyyaml.__version__} (libyaml={pyyaml.__with_libyaml__})"
    out = []
    for t in tests:
        data = (HERE / t["input_file"]).read_bytes()
        expected = t["expected"] == "valid"
        try:
            list(pyyaml.safe_load_all(data))
            actual = True
        except Exception:
            actual = False
        out.append((expected, actual))
    return name, out


def run_ruamel(tests: list[dict]) -> tuple[str, list[tuple[bool, bool]]] | None:
    try:
        import ruamel.yaml
        from ruamel.yaml import YAML
    except ImportError:
        print("[skip] ruamel.yaml not installed (pip install ruamel.yaml)")
        return None
    # Suppress the ReusedAnchorWarning noise — those are warnings, not errors.
    import warnings
    warnings.filterwarnings("ignore", category=ruamel.yaml.error.ReusedAnchorWarning)
    loader = YAML(typ="safe")
    name = f"ruamel.yaml {ruamel.yaml.__version__} (typ=safe)"
    out = []
    for t in tests:
        data = (HERE / t["input_file"]).read_bytes()
        expected = t["expected"] == "valid"
        try:
            list(loader.load_all(data))
            actual = True
        except Exception:
            actual = False
        out.append((expected, actual))
    return name, out


def run_libfyaml(tests: list[dict]) -> tuple[str, list[tuple[bool, bool]]] | None:
    if not shutil.which("fy-tool"):
        print("[skip] libfyaml's fy-tool not on PATH (brew install libfyaml)")
        return None
    version = subprocess.run(["fy-tool", "--version"], capture_output=True, text=True).stdout.strip()
    name = f"libfyaml {version} (fy-tool --yaml-1.2 dump -)"
    out = []
    for t in tests:
        data = (HERE / t["input_file"]).read_bytes()
        expected = t["expected"] == "valid"
        proc = subprocess.run(
            ["fy-tool", "--yaml-1.2", "dump", "-"],
            input=data, capture_output=True, timeout=10,
        )
        actual = proc.returncode == 0
        out.append((expected, actual))
    return name, out


PARSERS = {
    "pyyaml":   run_pyyaml,
    "ruamel":   run_ruamel,
    "libfyaml": run_libfyaml,
}


def report(name: str, results: list[tuple[bool, bool]], tests: list[dict],
           verbose: bool) -> None:
    tp, fn, fp, tn = confusion(results)
    total = tp + fn + fp + tn
    print(f"\n{name}")
    print(f"  TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"  MCC={mcc(tp, fn, fp, tn):.3f}  agreement={tp + tn}/{total} "
          f"({100 * (tp + tn) / total:.1f}%)")
    if verbose:
        disagree = [(t["id"], exp, act) for t, (exp, act) in zip(tests, results)
                    if exp != act]
        if disagree:
            print(f"  Disagreements ({len(disagree)}):")
            for tid, exp, act in disagree:
                el = "valid" if exp else "invalid"
                al = "valid" if act else "invalid"
                print(f"    {tid:14s} corpus={el:7s} parser={al}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parser", choices=list(PARSERS) + ["all"], default="all",
                   help="Run only one parser (default: all available)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="List per-test disagreements for each parser")
    p.add_argument("--skip-integrity", action="store_true",
                   help="Skip the upstream-vs-tests.jsonl label cross-check")
    args = p.parse_args()

    tests = load_tests()
    print(f"Corpus: {len(tests)} tests "
          f"({sum(1 for t in tests if t['expected']=='valid')} valid, "
          f"{sum(1 for t in tests if t['expected']=='invalid')} invalid)")

    if not args.skip_integrity:
        n = integrity_check(tests)
        if n > 0:
            print(f"WARNING: {n} integrity mismatch(es); setup.sh extraction may have drifted")

    selected = list(PARSERS) if args.parser == "all" else [args.parser]
    for key in selected:
        result = PARSERS[key](tests)
        if result is None:
            continue
        name, rs = result
        report(name, rs, tests, args.verbose)


if __name__ == "__main__":
    main()
