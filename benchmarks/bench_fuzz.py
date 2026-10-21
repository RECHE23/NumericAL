#!/usr/bin/env python3
"""Randomised comparative benchmark: NumericAL vs NumPy over drawn operations, shapes, dtypes and layouts.

Where bench.py and bench-linalg time chosen cases (and bench-linalg's default orders are powers of two, where the
kernels' tiles and panels divide evenly), this draws its cases: the operation, every dimension on a log scale from 1
to --largest (so an order of 100 or 300, between two thresholds, counts as much as 128), the dtype (float64,
complex128, float32, complex64) and how the operands arrive (a contiguous NumPy array, a transposed or a strided one,
or a Tensor already). It reports the whole distribution of the ratios, NumPy's time over NumericAL's (above 1,
NumericAL is faster), by operation, size, dtype and layout, and the worst cases with what reproduces each: the tail
a fixed benchmark never reaches.

Each library is timed in a process of its own, twice, the two alternating (NumericAL, NumPy, NumericAL, NumPy), and
each case keeps its fastest batch: OpenBLAS's threads, which spin for a while after a call, would otherwise take the
cores NumericAL's next call runs on, and a drift of the clock between the passes cancels. Every case's answers are
then compared, untimed: a case where they differ is reported and fails the run (a fast wrong answer is no win).

    make bench-fuzz                                  # 300 cases up to order 1024
    make bench-fuzz CASES=1000 SEED=7 LARGEST=2048
    python3 benchmarks/bench_fuzz.py --seed 7 --only 41,97   # the cases a report names

Run from the repository root.
"""
import argparse
import json
import math
import platform
import subprocess
import sys
import time

sys.path.insert(0, "python")
import numerical as nl  # noqa: E402
# Dependency-free statistics from SciForge's shared substrate (sciforge.bench); the sibling ../sciforge/python is on
# PYTHONPATH via the Makefile (make bench-fuzz).
from sciforge.bench import ascii_ecdf, fmt, geomean_ci  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover - a manual tool
    print("NumPy is required for this benchmark: pip install numpy")
    sys.exit(0)

DTYPES = (("float64", 0.45), ("complex128", 0.20), ("float32", 0.20), ("complex64", 0.15))
LAYOUTS = (("contiguous", 0.40), ("tensor", 0.30), ("transposed", 0.15), ("strided", 0.15))
# The work one call may take, in floating-point operations: a drawn shape past it shrinks, every dimension alike.
BUDGET = 6e8


def flops(op, dims):
    """A call's work, roughly: what keeps a drawn shape within the budget."""
    m = dims[0]
    n = dims[1] if len(dims) > 1 else m
    k = dims[2] if len(dims) > 2 else n
    return {
        "matmul": 2 * m * k * n, "einsum-product": 2 * m * k * n, "matvec": 2 * m * n, "norm": 2 * m * n,
        "einsum-reduce": m * n, "solve": (2 / 3) * m ** 3 + 2 * m * m * max(n, 1), "det": (2 / 3) * m ** 3,
        "slogdet": (2 / 3) * m ** 3, "inv": 2 * m ** 3, "cholesky": m ** 3 / 3, "qr": 4 * m * n * n,
        "eigh": 9 * m ** 3, "eigvalsh": 4 * m ** 3, "svd": 14 * m * n * n, "svdvals": 4 * m * n * n,
        "lstsq": 14 * m * n * n + 2 * m * n * k, "eig": 25 * m ** 3,
    }[op]


# Each operation: its weight in the draw, and how many dimensions it takes.
OPS = {
    "matmul": (3, 3), "einsum-product": (1, 3), "matvec": (2, 2), "norm": (1, 2), "einsum-reduce": (2, 2),
    "solve": (3, 2), "det": (2, 1), "slogdet": (1, 1), "inv": (2, 1), "cholesky": (2, 1), "qr": (2, 2),
    "eigh": (2, 1), "eigvalsh": (1, 1), "svd": (2, 2), "svdvals": (1, 2), "lstsq": (2, 3), "eig": (1, 1),
}
# Orders past which a drawn eig is not worth its time against the others.
LIMITS = {"eig": 400}


def weighted(rng, pairs):
    names = [name for name, _ in pairs]
    weights = np.array([weight for _, weight in pairs], dtype=float)
    return names[int(rng.choice(len(names), p=weights / weights.sum()))]


def log_size(rng, largest):
    return max(1, int(round(math.exp(rng.uniform(0.0, math.log(largest + 0.5))))))


def draw_case(seed, index, largest, ops):
    """Case `index` of the run seeded `seed`: the same in every process."""
    rng = np.random.default_rng([seed, index])
    op = weighted(rng, [(name, OPS[name][0]) for name in ops])
    dims = [min(log_size(rng, largest), LIMITS.get(op, largest)) for _ in range(OPS[op][1])]
    if op in ("lstsq",) and dims[0] < dims[1]:
        dims[0], dims[1] = dims[1], dims[0]  # a tall matrix, of full column rank: one least-squares answer
    if op == "lstsq":
        dims[2] = max(1, dims[2] % 9)  # a few right-hand sides
    if op == "solve":
        dims[1] = 0 if rng.random() < 0.5 else max(1, dims[1] % 17)  # a vector, or a few columns
    while flops(op, dims) > BUDGET:
        dims = [max(1, int(d * 0.8)) for d in dims]
    return {"index": index, "op": op, "dims": dims, "dtype": weighted(rng, DTYPES), "layout": weighted(rng, LAYOUTS),
            "spec": ["ij->", "ij->i", "ij->j"][int(rng.integers(3))] if op == "einsum-reduce"
            else ["ij,jk->ik", "ij,kj->ik", "ji,jk->ik"][int(rng.integers(3))] if op == "einsum-product" else None}


def matrix(rng, shape, dtype):
    real = rng.standard_normal(shape)
    if np.dtype(dtype).kind == "c":
        return (real + 1j * rng.standard_normal(shape)).astype(dtype)
    return real.astype(dtype)


def arrive(array, layout):
    """The operand as the case hands it over: as it is, in Fortran order, every other row of a larger array, or a
    Tensor already (for NumericAL; NumPy takes the array)."""
    if layout == "transposed" and array.ndim == 2:
        return np.asfortranarray(array)
    if layout == "strided" and array.ndim >= 1:
        wide = np.zeros((2 * array.shape[0],) + array.shape[1:], dtype=array.dtype)
        wide[::2] = array
        return wide[::2]
    return array


def build(case):
    """The case's operands and its two calls: (numerical_call, numpy_call, check)."""
    rng = np.random.default_rng([case["index"], 0x5eed])
    op, dims, dtype = case["op"], case["dims"], np.dtype(case["dtype"])
    m = dims[0]
    n = dims[1] if len(dims) > 1 else m

    def well(order):  # a random matrix pushed away from singularity
        return (matrix(rng, (order, order), dtype) + order * np.eye(order, dtype=dtype)).astype(dtype)

    def spd(order):
        b = matrix(rng, (order, order), dtype)
        return (b @ b.conj().T + order * np.eye(order, dtype=dtype)).astype(dtype)

    if op == "matmul":
        operands = [matrix(rng, (m, dims[2]), dtype), matrix(rng, (dims[2], n), dtype)]
        ours, theirs = (lambda a, b: nl.matmul(a, b)), (lambda a, b: np.matmul(a, b))
    elif op == "einsum-product":
        k = dims[2]
        spec = case["spec"]
        shapes = {"ij,jk->ik": ((m, k), (k, n)), "ij,kj->ik": ((m, k), (n, k)), "ji,jk->ik": ((k, m), (k, n))}[spec]
        operands = [matrix(rng, shapes[0], dtype), matrix(rng, shapes[1], dtype)]
        ours, theirs = (lambda a, b: nl.einsum(spec, a, b)), (lambda a, b: np.einsum(spec, a, b))
    elif op == "matvec":
        operands = [matrix(rng, (m, n), dtype), matrix(rng, (n,), dtype)]
        ours, theirs = (lambda a, x: nl.matmul(a, x)), (lambda a, x: np.matmul(a, x))
    elif op == "norm":
        operands = [matrix(rng, (m, n), dtype)]
        ours, theirs = nl.norm, np.linalg.norm
    elif op == "einsum-reduce":
        spec = case["spec"]
        operands = [matrix(rng, (m, n), dtype)]
        ours, theirs = (lambda a: nl.einsum(spec, a)), (lambda a: np.einsum(spec, a))
    elif op == "solve":
        rhs = matrix(rng, (m,) if n == 0 else (m, n), dtype)
        operands = [well(m), rhs]
        ours, theirs = nl.solve, np.linalg.solve
    elif op in ("det", "slogdet", "inv"):
        operands = [(well(m) / m).astype(dtype)]  # its diagonal near 1: a determinant far from overflowing
        ours, theirs = {"det": (nl.det, np.linalg.det), "slogdet": (nl.slogdet, np.linalg.slogdet),
                        "inv": (nl.inv, np.linalg.inv)}[op]
    elif op == "cholesky":
        operands = [spd(m)]
        ours, theirs = nl.cholesky, np.linalg.cholesky
    elif op == "qr":
        operands = [matrix(rng, (m, n), dtype)]
        ours, theirs = nl.qr, np.linalg.qr
    elif op in ("eigh", "eigvalsh"):
        b = matrix(rng, (m, m), dtype)
        operands = [((b + b.conj().T) / 2).astype(dtype)]
        ours, theirs = (nl.eigh, np.linalg.eigh) if op == "eigh" else (nl.eigvalsh, np.linalg.eigvalsh)
    elif op == "svd":
        operands = [matrix(rng, (m, n), dtype)]
        ours, theirs = (lambda a: nl.svd(a, full_matrices=False)), (lambda a: np.linalg.svd(a, full_matrices=False))
    elif op == "svdvals":
        operands = [matrix(rng, (m, n), dtype)]
        ours, theirs = nl.svdvals, (lambda a: np.linalg.svd(a, compute_uv=False))
    elif op == "lstsq":
        operands = [matrix(rng, (m, n), dtype), matrix(rng, (m, dims[2]), dtype)]
        ours, theirs = (lambda a, b: nl.lstsq(a, b)[0]), (lambda a, b: np.linalg.lstsq(a, b, rcond=None)[0])
    elif op == "eig":
        operands = [matrix(rng, (m, m), dtype)]
        ours, theirs = nl.eigvals, np.linalg.eigvals
    else:  # pragma: no cover - every drawn operation is listed above
        raise ValueError(op)
    arrived = [arrive(a, case["layout"]) for a in operands]
    ours_in = [nl.asarray(a) for a in arrived] if case["layout"] == "tensor" else arrived
    return (lambda: ours(*ours_in)), (lambda: theirs(*arrived)), operands


def tolerance(dtype, order):
    """The relative gap two stable algorithms' answers may show: a few units of the dtype's precision, growing with
    the order."""
    unit = 1e-3 if np.dtype(dtype) in (np.float32, np.complex64) else 1e-9
    return unit * (1 + order / 64)


def relative(got, want):
    got, want = np.asarray(got, dtype=complex), np.asarray(want, dtype=complex)
    if got.shape != want.shape:
        return math.inf
    same = got == want  # infinities of the same sign agree; the rest is measured
    got, want = np.where(same, 0, got), np.where(same, 0, want)
    scale = max(np.abs(want).max(initial=0.0), 1e-300)
    return float(np.abs(got - want).max(initial=0.0) / scale)


def agrees(case, ours, theirs, operands):
    """Whether NumericAL's answer is NumPy's within the dtype's tolerance: compared directly where the answer is
    unique, by what defines it where it is not (QR, eigenvectors, singular vectors: their reconstruction)."""
    op, dims = case["op"], case["dims"]
    tol = tolerance(case["dtype"], max(dims))
    a = np.asarray(operands[0]).astype(complex)
    if op == "qr":
        q, r = (np.asarray(x) for x in ours)
        return relative(q @ r, a) < tol
    if op == "eigh":
        values, vectors = (np.asarray(x) for x in ours)
        return relative(values, theirs[0]) < tol and relative((vectors * values) @ vectors.conj().T, a) < tol * 10
    if op == "svd":
        u, s, vh = (np.asarray(x) for x in ours)
        return relative(s, theirs[1]) < tol and relative((u * s) @ vh, a) < tol * 10
    if op == "slogdet":
        sign, log = ours
        return relative(sign, theirs[0]) < tol and relative(log, theirs[1]) < tol
    if op == "eig":
        got = np.sort_complex(np.asarray(ours).astype(complex))
        want = np.sort_complex(np.asarray(theirs).astype(complex))
        return relative(got, want) < tol * 100  # eigenvalues of a nonnormal matrix: conditioned worse
    return relative(ours, theirs) < tol


def batch_time(fn, number):
    start = time.perf_counter()
    for _ in range(number):
        fn()
    return (time.perf_counter() - start) / number


def fastest(fn, rounds, target=0.002):
    """The fastest per-call time over `rounds` batches, each long enough (`target` seconds) for the clock."""
    fn()  # a first call outside the timing: allocations, the pool's start, a lazy path
    number = 1
    while number < 1 << 20 and batch_time(fn, number) * number < target:
        number *= 2
    return min(batch_time(fn, number) for _ in range(rounds))


def side_main(args):
    """One library's times for every case, as JSON on the standard output."""
    times = {}
    for case in cases_of(args):
        ours, theirs, _ = build(case)
        times[case["index"]] = fastest(ours if args.side == "numerical" else theirs, args.rounds)
    json.dump(times, sys.stdout)


def cases_of(args):
    ops = args.ops.split(",") if args.ops else list(OPS)
    indices = [int(i) for i in args.only.split(",")] if args.only else range(args.cases)
    return [draw_case(args.seed, i, args.largest, ops) for i in indices]


def band(case):
    largest = max(case["dims"])
    for bound, name in ((16, "1-15"), (64, "16-63"), (256, "64-255"), (1024, "256-1023")):
        if largest < bound:
            return name
    return "1024+"


def table(title, groups):
    print(f"\n{title:<16} {'cases':>6} {'geomean':>9} {'95% CI':>17} {'worst':>8}")
    for name in sorted(groups):
        ratios = groups[name]
        if len(ratios) > 1:
            g, low, high = geomean_ci(ratios)
            ci = f"[{low:.2f}-{high:.2f}]"
        else:
            g, ci = ratios[0], ""
        print(f"{name:<16} {len(ratios):>6} {g:>8.2f}x {ci:>17} {min(ratios):>7.2f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--largest", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=5, help="batches per case in each pass")
    parser.add_argument("--ops", default="", help="a comma-separated subset of: " + ", ".join(OPS))
    parser.add_argument("--only", default="", help="comma-separated case indices (to reproduce a reported case)")
    parser.add_argument("--worst", type=int, default=20)
    parser.add_argument("--json", default="", help="write every case and its times here")
    parser.add_argument("--side", choices=("numerical", "numpy"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.side:
        side_main(args)
        return 0

    cases = cases_of(args)
    child = [sys.executable, __file__, "--seed", str(args.seed), "--largest", str(args.largest), "--rounds",
             str(args.rounds), "--cases", str(args.cases)] + (["--ops", args.ops] if args.ops else []) \
        + (["--only", args.only] if args.only else [])
    best = {"numerical": {}, "numpy": {}}
    started = time.perf_counter()
    for _ in range(2):
        for side in ("numerical", "numpy"):
            out = subprocess.run(child + ["--side", side], check=True, capture_output=True, text=True).stdout
            for index, seconds in json.loads(out).items():
                previous = best[side].get(int(index), math.inf)
                best[side][int(index)] = min(previous, seconds)

    print(f"NumericAL {nl.__version__} ({nl.gemm_path()}, {nl.get_num_threads()} threads) vs NumPy {np.__version__}, "
          f"Python {platform.python_version()}, {platform.machine()}")
    print(f"{len(cases)} cases, seed {args.seed}, dimensions up to {args.largest} on a log scale, "
          f"{time.perf_counter() - started:.0f} s of timing")

    rows, divergent = [], []
    for case in cases:
        ours_fn, theirs_fn, operands = build(case)
        try:
            ok = agrees(case, ours_fn(), theirs_fn(), operands)
        except Exception as caught:  # an exception where NumPy answers is a divergence too
            ok = False
            case["error"] = repr(caught)
        if not ok:
            divergent.append(case)
            continue
        t_ours, t_theirs = best["numerical"][case["index"]], best["numpy"][case["index"]]
        rows.append(dict(case, numerical=t_ours, numpy=t_theirs, ratio=t_theirs / t_ours))

    if divergent:
        print(f"\n{len(divergent)} case(s) where NumericAL's answer is not NumPy's (not timed):")
        for case in divergent:
            print(f"  #{case['index']} {case['op']} {case['dims']} {case['dtype']} {case['layout']} "
                  f"{case.get('spec') or ''} {case.get('error', '')}")
    if not rows:
        return 1

    ratios = [row["ratio"] for row in rows]
    g, low, high = geomean_ci(ratios)
    faster = sum(r > 1 for r in ratios)
    print(f"\nNumPy/NumericAL over {len(rows)} cases: geometric mean {g:.2f}x [{low:.2f}-{high:.2f}], "
          f"NumericAL faster in {faster} ({100 * faster / len(rows):.0f} %), the worst {min(ratios):.2f}x")
    for title, key in (("operation", lambda r: r["op"]), ("largest dim", band), ("dtype", lambda r: r["dtype"]),
                       ("operands", lambda r: r["layout"])):
        groups = {}
        for row in rows:
            groups.setdefault(key(row), []).append(row["ratio"])
        table(title, groups)

    print("\nratios, cumulative (log scale; left of 1, NumPy faster):")
    print(ascii_ecdf(ratios, log=True))

    print(f"\nthe {args.worst} worst cases (reproduce: --seed {args.seed} --largest {args.largest} --only N):")
    print(f"{'N':>5} {'ratio':>7} {'operation':<15} {'dims':<16} {'dtype':<11} {'operands':<11} "
          f"{'NumericAL':>10} {'NumPy':>10}")
    for row in sorted(rows, key=lambda r: r["ratio"])[:args.worst]:
        name = row["op"] + (f" {row['spec']}" if row.get("spec") else "")
        print(f"{row['index']:>5} {row['ratio']:>6.2f}x {name:<15} {str(row['dims']):<16} {row['dtype']:<11} "
              f"{row['layout']:<11} {fmt(row['numerical'], 10)} {fmt(row['numpy'], 10)}")

    if args.json:
        with open(args.json, "w") as out:
            json.dump(dict(seed=args.seed, largest=args.largest, numerical=nl.__version__, numpy=np.__version__,
                           machine=platform.machine(), rows=rows, divergent=divergent), out, indent=1)
    return 1 if divergent else 0


if __name__ == "__main__":
    sys.exit(main())
