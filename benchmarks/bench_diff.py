#!/usr/bin/env python3
"""Compare the decompositions of two NumericAL trees, compiler by compiler: `make bench-diff`.

Each regression this guard exists for came at small orders and on one compiler only: Cholesky of order 17 to 47
(+25 to +85 % on clang and GCC alike) and, on x86-64, an LU solve of order 8 to 64 (+6 to +33 %), each from a change
measured at the orders it aimed at. So both trees -- a release's headers (`--base`, the last tag by default) and
the working tree's -- are built with each compiler the wheels use, and timed at the orders between the powers of
two as well, alternately, round after round, on one thread.

A cell (compiler, op, order) is suspect when both the ratio of the minima and the ratio of the medians (working
tree over base) exceed --threshold; a single noisy round moves one of the two, a real loss moves both. The suspects
are timed again in a second batch, and only a cell suspect in both batches is a regression: the run then exits 1.
A cell faster by the same margin is listed too, as information.

  python3 benchmarks/bench_diff.py                      # last tag against the working tree
  python3 benchmarks/bench_diff.py --base v2026.10.8    # another release
  python3 benchmarks/bench_diff.py --compiler gcc="g++ -O3" --rounds 8
  python3 benchmarks/bench_diff.py --self-test          # the decision rule, arm by arm

On a shared host the minimum is the reading to trust; run it where nothing else runs (a laptop on AC power).
"""

import argparse
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OPS = ["mm", "solve", "inv", "lu", "lsolve", "chol", "qr", "eigh", "svd"]
SIZES = [8, 17, 24, 33, 48, 64, 96, 128, 192, 256, 512]
# Every function at a 64-byte boundary in both builds: where the linker happened to put a kernel moved a product of
# order 24 by 8 % between two releases whose product code was byte for byte the same, and the two batches, the same
# binaries, confirmed it.
COMMON = ["-std=c++20", "-DNDEBUG", "-fwrapv", "-g0", "-pthread", "-falign-functions=64", "-ffp-contract=on"]


def default_compilers():
    """The wheels' compilers this host has: Apple clang at -O2 on macOS (and GCC 14 there without the Apple AMX,
    which the Linux arm64 wheel lacks), GCC at -O3 on x86-64 Linux, the system compiler otherwise."""
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        found = {"clang": "c++ -O2"}
        if shutil.which("g++-14"):
            found["gcc14"] = "g++-14 -O2 -DNUMERICAL_NO_AMX"
        return found
    if machine in ("x86_64", "amd64"):
        return {"gcc": "g++ -O3"}
    return {"cxx": "c++ -O2"}


def ratios(base, head):
    """(ratio of the minima, ratio of the medians), working tree over base."""
    return min(head) / min(base), statistics.median(head) / statistics.median(base)


def suspect(base, head, threshold):
    """Whether a cell is slower in the working tree: both ratios past the threshold."""
    by_min, by_median = ratios(base, head)
    return by_min > threshold and by_median > threshold


def faster(base, head, threshold):
    """Whether a cell is faster in the working tree by the same margin."""
    by_min, by_median = ratios(base, head)
    return by_min < 1 / threshold and by_median < 1 / threshold


def regressions(first, second, threshold):
    """The cells suspect in both batches: each batch maps a cell to its (base times, working-tree times)."""
    return sorted(cell for cell, (base, head) in first.items()
                  if suspect(base, head, threshold) and cell in second and suspect(*second[cell], threshold))


def build(tree, command, out):
    """Compiles ops_bench.cpp against `tree`'s headers with `command` (a compiler and its flags)."""
    words = command.split()
    subprocess.run(words[:1] + COMMON + words[1:] + ["-I" + os.path.join(tree, "include"),
                                                     os.path.join(HERE, "ops_bench.cpp"), "-o", out], check=True)


def batch(binaries, cells, rounds):
    """Times `cells` ({(compiler, op): [orders]}) `rounds` times, the two trees alternately (the first one first on
    even rounds), and returns {(compiler, op, order): ([base times], [working-tree times])}."""
    times = {}
    for r in range(rounds):
        for (compiler, op), orders in sorted(cells.items()):
            trees = ["base", "head"] if r % 2 == 0 else ["head", "base"]
            for tree in trees:
                out = subprocess.run([binaries[(compiler, tree)], "1", op] + [str(n) for n in orders],
                                     capture_output=True, text=True, check=True).stdout
                for field in out.split():
                    order, micros = field.split(":")
                    pair = times.setdefault((compiler, op, int(order)), ([], []))
                    pair[0 if tree == "base" else 1].append(float(micros))
        print(f"  round {r + 1}/{rounds}", file=sys.stderr, flush=True)
    return times


def last_tag():
    return subprocess.run(["git", "describe", "--tags", "--abbrev=0"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def self_test():
    """The decision rule, one arm at a time: each ratio past the threshold alone must not flag a cell."""
    base, slow = [10.0] * 4, [11.0] * 4
    median_alone = ([10.0] * 4, [9.0, 11.0, 11.0, 11.0])        # median ratio 1.10, min ratio 0.90
    min_alone = ([10.0, 12.0, 12.0, 12.0], [10.6, 11.0, 11.0, 11.0])  # min ratio 1.06, median ratio 0.92
    assert suspect(base, slow, 1.05)
    assert not suspect(base, base, 1.05)
    assert ratios(*median_alone)[1] > 1.05 >= ratios(*median_alone)[0] and not suspect(*median_alone, 1.05)
    assert ratios(*min_alone)[0] > 1.05 >= ratios(*min_alone)[1] and not suspect(*min_alone, 1.05)
    assert faster(base, [9.0] * 4, 1.05) and not faster(base, [9.0, 11.0, 11.0, 11.0], 1.05)
    cell = ("clang", "chol", 17)
    assert regressions({cell: (base, slow)}, {cell: (base, slow)}, 1.05) == [cell]  # suspect twice
    assert regressions({cell: (base, slow)}, {cell: (base, base)}, 1.05) == []      # not confirmed
    assert regressions({cell: (base, slow)}, {}, 1.05) == []                        # not timed again
    assert regressions({cell: (base, base)}, {cell: (base, slow)}, 1.05) == []      # the second batch alone
    print("bench_diff: self-test OK -- each arm of the rule refused alone")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", help="the release to compare against (default: the last tag)")
    parser.add_argument("--head", help="a release to compare in place of the working tree")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=1.05)
    parser.add_argument("--ops", default=",".join(OPS))
    parser.add_argument("--sizes", default=",".join(map(str, SIZES)))
    parser.add_argument("--compiler", action="append", default=[], metavar="NAME=COMMAND")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    base_ref = args.base or last_tag()
    compilers = dict(c.split("=", 1) for c in args.compiler) or default_compilers()
    ops = args.ops.split(",")
    sizes = [int(n) for n in args.sizes.split(",")]
    with tempfile.TemporaryDirectory(prefix="bench-diff-") as work:
        trees = {"base": os.path.join(work, "base"), "head": os.path.join(work, "head")}
        for tree, ref in (("base", base_ref), ("head", args.head)):
            if ref is None:
                shutil.copytree(os.path.join(ROOT, "include"), os.path.join(trees[tree], "include"))
                continue
            os.makedirs(trees[tree])
            archive = subprocess.run(["git", "archive", ref, "include"], cwd=ROOT, capture_output=True, check=True)
            subprocess.run(["tar", "-x", "-C", trees[tree]], input=archive.stdout, check=True)
        binaries = {}
        for name, command in compilers.items():
            for tree, path in trees.items():
                binaries[(name, tree)] = os.path.join(work, f"ops_{name}_{tree}")
                build(path, command, binaries[(name, tree)])
        head_name = args.head or "the working tree"
        print(f"bench-diff: {base_ref} against {head_name}, {', '.join(f'{k} ({v})' for k, v in compilers.items())}; "
              f"{args.rounds} rounds, one thread, threshold {args.threshold}", flush=True)
        first = batch(binaries, {(c, op): sizes for c in compilers for op in ops}, args.rounds)
        suspects = {}
        for (compiler, op, order), (base, head) in first.items():
            if suspect(base, head, args.threshold):
                suspects.setdefault((compiler, op), []).append(order)
        second = batch(binaries, suspects, args.rounds) if suspects else {}
    for compiler in compilers:
        print(f"\n  {compiler}: {args.head or 'working tree'} / {base_ref}, min ratio / median ratio")
        for op in ops:
            row = [f"{order}:{ratios(*first[(compiler, op, order)])[0]:.2f}/{ratios(*first[(compiler, op, order)])[1]:.2f}"
                   for order in sizes]
            print(f"    {op:7} " + "  ".join(row))
    gained = sorted(cell for cell, pair in first.items() if faster(*pair, args.threshold))
    if gained:
        print(f"\n  faster in {args.head or 'the working tree'}: " + ", ".join(f"{c} {op} {n}" for c, op, n in gained))
    lost = regressions(first, second, args.threshold)
    if lost:
        print("\nbench-diff: REGRESSION, slower in both batches: "
              + ", ".join(f"{c} {op} {n} ({ratios(*second[(c, op, n)])[0]:.2f}x)" for c, op, n in lost))
        return 1
    unconfirmed = sum(len(orders) for orders in suspects.values())
    print(f"\nbench-diff: no regression past {args.threshold} ({unconfirmed} suspect cells not confirmed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
