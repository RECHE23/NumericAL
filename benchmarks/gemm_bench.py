#!/usr/bin/env python3
"""Reporter for the GEMM micro-benchmark collector (gemm_bench.cpp).

The C++ binary only measures and emits the canonical sciforge.bench JSON (one Case per
(N, variant), raw per-call seconds); this script reads it back with sciforge.bench and prints
the loop-order comparison table. The headline figure is the **minimum** sample per variant —
preserving the documented "best-of-N minimum" convention of NumericAL's baselines, so the numbers
stay comparable to the prior table. A result mismatch (a variant disagreeing with operator*) is
flagged and makes the run exit non-zero — a fast wrong answer is never a win.

  python benchmarks/gemm_bench.py path/to/gemm_bench    # runs the binary
  python benchmarks/gemm_bench.py --json gemm.json      # reads a saved run
Run via: make bench-gemm  (manual; not a CI gate).
"""

import argparse
import json
import subprocess
import sys

from sciforge.bench import run_from_json

VARIANTS = ["naive-ijk", "operator*", "einsum", "blocked"]


def load(args):
    if args.json:
        with open(args.json, encoding="utf-8") as handle:
            return run_from_json(json.load(handle))
    out = subprocess.run([args.binary], capture_output=True, text=True, check=True).stdout
    return run_from_json(json.loads(out))


def main():
    parser = argparse.ArgumentParser(description="NumericAL GEMM benchmark reporter")
    parser.add_argument("binary", nargs="?", help="path to the gemm_bench binary to run")
    parser.add_argument("--json", metavar="PATH", help="read a saved run instead of running the binary")
    args = parser.parse_args()
    if not args.binary and not args.json:
        parser.error("need a binary path or --json PATH")
    run = load(args)

    # Group the cases by N, then by variant (the two domain fields the C++ emitted as siblings).
    by_n = {}
    for case in run.cases:
        by_n.setdefault(case.extra["n"], {})[case.extra["variant"]] = case

    compiler = run.meta.get("compiler", "?")
    print(f"NumericAL GEMM (N×N double, best-of-N min ms, raw samples → sciforge.bench; {compiler}):")
    print(f"  {'N':>6} {'naive ijk':>11} {'operator*':>11} {'einsum':>11} {'blocked':>11} "
          f"{'op*/naive':>11} {'op*/einsum':>11}")

    mismatched = False
    for n in sorted(by_n, key=int):
        variants = by_n[n]
        if any(name not in variants for name in VARIANTS):
            print(f"  {int(n):>6}   MISSING VARIANT(S): {sorted(set(VARIANTS) - set(variants))}")
            mismatched = True
            continue
        # best-of-N minimum, seconds -> milliseconds.
        t = {name: min(variants[name].samples) * 1e3 for name in VARIANTS}
        row_mismatch = any(variants[name].result is False for name in VARIANTS)
        mismatched = mismatched or row_mismatch
        flag = "  [MISMATCH!]" if row_mismatch else ""
        print(f"  {int(n):>6} {t['naive-ijk']:>11.2f} {t['operator*']:>11.2f} {t['einsum']:>11.2f} "
              f"{t['blocked']:>11.2f} {t['naive-ijk'] / t['operator*']:>10.2f}x "
              f"{t['einsum'] / t['operator*']:>10.2f}x{flag}")

    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
