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

With --eigen (Eigen's include directory), the cases Eigen and NumericAL's C++ API both have a function for are
also timed from C++: benchmarks/fuzz_worker.cpp, compiled here once for each library, times them in a process of its
own for each, in turn with the Python sides, each library given its operands in its own matrix type. Both C++
sides' answers pass the check NumericAL's Python answers pass, against NumPy's; a case either fails is reported and
left out of their ratio (Eigen's time over NumericAL's), and a NumericAL failure fails the run. The NumPy ratio stays
Python against Python. An operation either C++ side lacks (einsum in Eigen, a matrix norm in NumericAL's C++ API)
is reported as n/a for that comparison.

    make bench-fuzz                                  # 300 cases up to order 1024
    make bench-fuzz CASES=1000 SEED=7 LARGEST=2048
    make bench-fuzz EIGEN=path/to/eigen-3.4.0        # and Eigen against NumericAL, from C++
    make bench-fuzz EIGEN=path/to/eigen-3.4.0 EIGEN_OPENMP=1 NUMERICAL_THREADS=1
    python3 benchmarks/bench_fuzz.py --seed 7 --only 41,97   # the cases a report names

Run from the repository root.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
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
# The operations both C++ sides have a function for: Eigen has no einsum, NumericAL's C++ API no matrix norm.
CPP_NA = {"einsum-product": "no Eigen function", "einsum-reduce": "no Eigen function",
          "norm": "no NumericAL C++ function"}
CPP_OPS = tuple(op for op in OPS if op not in CPP_NA)
HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "fuzz_worker.cpp")
INCLUDE = os.path.join(os.path.dirname(HERE), "include")
# The wheels' flags (setup.py's, with the -DNDEBUG and -fwrapv the interpreter's own add): -O3 on x86-64, -O2
# elsewhere.
NUMERICAL_FLAGS = ["-O3" if platform.machine().lower() in ("x86_64", "amd64") else "-O2", "-fwrapv", "-g0",
                   "-fvisibility=hidden", "-pthread", "-ffp-contract=on", "-I" + INCLUDE]


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


# How long a timed batch lasts at least, in seconds: Eigen's worker reads it from its manifest.
TARGET = 0.002


def fastest(fn, rounds, target=TARGET):
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


def build_worker(library, command, sources):
    """The worker for `library` ("eigen" or "numerical"), built once for each command and each content of `sources`
    (under build/). Returns its path."""
    digest = hashlib.sha256("\0".join(command).encode())
    for path in sources:
        with open(path, "rb") as source:
            digest.update(source.read())
    binary = os.path.join("build", f"fuzz_worker-{library}-{digest.hexdigest()[:16]}")
    if not os.path.exists(binary):
        os.makedirs("build", exist_ok=True)
        built = subprocess.run(command + [WORKER, "-o", binary + ".part"], capture_output=True, text=True)
        if built.returncode:
            sys.exit(f"building the {library} worker failed:\n{' '.join(command)} {WORKER}\n{built.stderr}")
        os.replace(binary + ".part", binary)
    return binary


def has_openmp(cxx):
    """Whether `cxx` compiles and links a program with -fopenmp."""
    with tempfile.TemporaryDirectory() as scratch:
        probe = os.path.join(scratch, "probe.cpp")
        with open(probe, "w") as out:
            out.write("#include <omp.h>\nint main() { return omp_get_max_threads() > 0 ? 0 : 1; }\n")
        return subprocess.run([cxx, "-fopenmp", probe, "-o", os.path.join(scratch, "probe")],
                              capture_output=True).returncode == 0


def cpp_workers(include, args):
    """The two C++ workers, one compiler for both: {library: (binary, command)}. Eigen at -O3 -DNDEBUG, without
    -march=native, and without OpenMP unless --eigen-openmp asks for it (skipped where the compiler lacks it);
    NumericAL with the wheels' flags."""
    cxx = os.environ.get("CXX") or "c++"
    common = [cxx, "-std=c++20", "-DNDEBUG"]
    openmp = []
    if args.eigen_openmp:
        if has_openmp(cxx):
            openmp = ["-fopenmp"]
        else:
            print(f"--eigen-openmp: {cxx} does not build with -fopenmp; Eigen runs on one thread (CXX= names another "
                  "compiler)")
    eigen = common + ["-O3", "-I" + include] + openmp + args.eigen_flags.split() + ["-DBENCH_EIGEN"]
    numerical = common + NUMERICAL_FLAGS + ["-DBENCH_NUMERICAL"]
    package = os.path.join(INCLUDE, "numerical")
    headers = sorted(os.path.join(package, name) for name in os.listdir(package))
    return {"numerical": (build_worker("numerical", numerical, [WORKER] + headers), " ".join(numerical)),
            "eigen": (build_worker("eigen", eigen, [WORKER]), " ".join(eigen))}


def write_cpp_cases(cases, directory, args):
    """The cases both C++ sides can take, for their workers: each operand as raw little-endian elements in row-major
    order, and a manifest of them."""
    listed = []
    for case in cases:
        if case["op"] not in CPP_OPS:
            continue
        files = []
        for k, operand in enumerate(build(case)[2]):
            name = f"{case['index']}-{k}.bin"
            np.ascontiguousarray(operand, dtype=operand.dtype.newbyteorder("<")).tofile(os.path.join(directory, name))
            files.append({"file": name, "shape": list(operand.shape)})
        listed.append({"index": case["index"], "op": case["op"], "dtype": case["dtype"], "operands": files})
    with open(os.path.join(directory, "manifest.json"), "w") as out:
        json.dump({"rounds": args.rounds, "target": TARGET, "numerical_threads": args.numerical_threads,
                   "cases": listed}, out)


def cpp_answer(directory, described):
    """A case's answers as a worker wrote them: an array, or a tuple of them where NumPy returns one."""
    answers = [np.fromfile(os.path.join(directory, d["file"]), dtype=np.dtype(d["dtype"]).newbyteorder("<"))
               .reshape(d["shape"]) for d in described]
    return answers[0] if len(answers) == 1 else tuple(answers)


def report(rows, title, name, ratio_key, ours_key, theirs_key, args, reproduce="", layouts=True):
    """The distribution of `name`'s time over NumericAL's: overall, by operation, size, dtype and (with `layouts`)
    layout, and the worst cases."""
    ratios = [row[ratio_key] for row in rows]
    g, low, high = geomean_ci(ratios)
    faster = sum(r > 1 for r in ratios)
    print(f"\n{title} over {len(rows)} cases: geometric mean {g:.2f}x [{low:.2f}-{high:.2f}], "
          f"NumericAL faster in {faster} ({100 * faster / len(rows):.0f} %), the worst {min(ratios):.2f}x")
    groupings = [("operation", lambda r: r["op"]), ("largest dim", band), ("dtype", lambda r: r["dtype"])]
    for heading, key in groupings + ([("operands", lambda r: r["layout"])] if layouts else []):
        groups = {}
        for row in rows:
            groups.setdefault(key(row), []).append(row[ratio_key])
        table(heading, groups)

    print(f"\nratios, cumulative (log scale; left of 1, {name} faster):")
    print(ascii_ecdf(ratios, log=True))

    print(f"\nthe {args.worst} worst cases (reproduce: --seed {args.seed} --largest {args.largest}{reproduce} "
          "--only N):")
    operands = f" {'operands':<11}" if layouts else ""
    print(f"{'N':>5} {'ratio':>7} {'operation':<15} {'dims':<16} {'dtype':<11}{operands} {'NumericAL':>10} {name:>10}")
    for row in sorted(rows, key=lambda r: r[ratio_key])[:args.worst]:
        label = row["op"] + (f" {row['spec']}" if row.get("spec") else "")
        layout = f" {row['layout']:<11}" if layouts else ""
        print(f"{row['index']:>5} {row[ratio_key]:>6.2f}x {label:<15} {str(row['dims']):<16} {row['dtype']:<11}"
              f"{layout} {fmt(row[ours_key], 10)} {fmt(row[theirs_key], 10)}")


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
    parser.add_argument("--eigen", default="",
                        help="Eigen's include directory (the one holding Eigen/): Eigen against NumericAL, from C++")
    parser.add_argument("--eigen-flags", default="", help="compiler flags added to Eigen's worker's (as one argument)")
    parser.add_argument("--eigen-openmp", action="store_true", help="Eigen's worker with OpenMP: its threaded products")
    parser.add_argument("--numerical-threads", type=int, default=0,
                        help="the threads NumericAL's C++ worker may use (0: the hardware's)")
    parser.add_argument("--side", choices=("numerical", "numpy"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.side:
        side_main(args)
        return 0
    if not args.eigen:
        return compare(args, None)
    include = os.path.expanduser(args.eigen)
    if not os.path.isfile(os.path.join(include, "Eigen", "Dense")):
        print(f"--eigen {args.eigen}: no Eigen/Dense there; give Eigen's include directory, the one holding Eigen/")
        return 2
    workers = cpp_workers(include, args)
    with tempfile.TemporaryDirectory(prefix="bench-fuzz-cpp-") as directory:
        return compare(args, {"workers": workers, "directory": directory})


def check_cpp(case, theirs, operands, cpp):
    """Whether each C++ side's answer to `case` is NumPy's (`theirs`): {library: None, or why not}."""
    verdicts = {}
    for library in ("numerical", "eigen"):
        try:
            answer = cpp_answer(cpp["directory"], cpp["described"][library][str(case["index"])])
            verdicts[library] = None if agrees(case, answer, theirs, operands) else "differs"
        except Exception as caught:  # the same check NumericAL's Python answer passes
            verdicts[library] = repr(caught)
    return verdicts


def compare(args, cpp):
    """Times every case on each side, checks every answer, and reports; `cpp` (the C++ workers and the directory of
    their operands) adds the comparison of Eigen with NumericAL from C++."""
    cases = cases_of(args)
    child = [sys.executable, __file__, "--seed", str(args.seed), "--largest", str(args.largest), "--rounds",
             str(args.rounds), "--cases", str(args.cases)] + (["--ops", args.ops] if args.ops else []) \
        + (["--only", args.only] if args.only else [])
    sides = {"numerical": child + ["--side", "numerical"], "numpy": child + ["--side", "numpy"]}
    if cpp:
        write_cpp_cases(cases, cpp["directory"], args)
        for library, (binary, _) in cpp["workers"].items():
            sides[f"{library}-cpp"] = [binary, cpp["directory"]]
    best = {side: {} for side in sides}
    started = time.perf_counter()
    for turn in range(2):
        for side, command in sides.items():
            answers = ["--results"] if side.endswith("-cpp") and turn == 0 else []  # written once, outside the timing
            out = subprocess.run(command + answers, check=True, capture_output=True, text=True).stdout
            for index, seconds in json.loads(out).items():
                previous = best[side].get(int(index), math.inf)
                best[side][int(index)] = min(previous, seconds)

    print(f"NumericAL {nl.__version__} ({nl.gemm_path()}, {nl.get_num_threads()} threads) vs NumPy {np.__version__}, "
          f"Python {platform.python_version()}, {platform.machine()}")
    if cpp:
        cpp["info"], cpp["described"] = {}, {}
        for library, (_, command) in cpp["workers"].items():
            with open(os.path.join(cpp["directory"], f"{library}-info.json")) as info_file:
                cpp["info"][library] = dict(json.load(info_file), compiled=command)
            with open(os.path.join(cpp["directory"], f"{library}-results.json")) as results_file:
                cpp["described"][library] = json.load(results_file)
        ours, eigen = cpp["info"]["numerical"], cpp["info"]["eigen"]
        def threads(count):
            return f"{count} thread{'s' if count != 1 else ''}"

        print(f"and from C++: NumericAL ({ours['gemm_path']}, {threads(ours['threads'])}) vs Eigen {eigen['version']} "
              f"({threads(eigen['threads'])}, "
              f"{'OpenMP' if eigen['openmp'] else 'no OpenMP: single-threaded, as it ships'}, "
              f"{eigen['simd']})")
        print(f"  NumericAL built with {ours['compiled']}\n  Eigen built with {eigen['compiled']}")
    print(f"{len(cases)} cases, seed {args.seed}, dimensions up to {args.largest} on a log scale, "
          f"{time.perf_counter() - started:.0f} s of timing")

    rows, divergent, cpp_rows, cpp_divergent = [], [], [], []
    for case in cases:
        ours_fn, theirs_fn, operands = build(case)
        theirs = None
        try:
            theirs = theirs_fn()
            ok = agrees(case, ours_fn(), theirs, operands)
        except Exception as caught:  # an exception where NumPy answers is a divergence too
            ok = False
            case["error"] = repr(caught)
        if ok:
            t_ours, t_theirs = best["numerical"][case["index"]], best["numpy"][case["index"]]
            rows.append(dict(case, numerical=t_ours, numpy=t_theirs, ratio=t_theirs / t_ours))
        else:
            divergent.append(case)
        if cpp and case["op"] in CPP_OPS and theirs is not None:
            failed = {library: why for library, why in check_cpp(case, theirs, operands, cpp).items() if why}
            if failed:
                cpp_divergent.append(dict(case, failed=failed))
            else:
                t_ours, t_eigen = best["numerical-cpp"][case["index"]], best["eigen-cpp"][case["index"]]
                cpp_rows.append(dict(case, numerical_cpp=t_ours, eigen=t_eigen, eigen_ratio=t_eigen / t_ours))

    if divergent:
        print(f"\n{len(divergent)} case(s) where NumericAL's answer is not NumPy's (not timed):")
        for case in divergent:
            print(f"  #{case['index']} {case['op']} {case['dims']} {case['dtype']} {case['layout']} "
                  f"{case.get('spec') or ''} {case.get('error', '')}")
    if not rows:
        return 1

    report(rows, "NumPy/NumericAL", "NumPy", "ratio", "numerical", "numpy", args)

    cpp_failed = False
    if cpp:
        na = [case["index"] for case in cases if case["op"] not in CPP_OPS]
        reasons = "; ".join(f"{op}: {why}" for op, why in CPP_NA.items())
        print(f"\n\nFrom C++: {len(na)} case(s) n/a ({reasons}), left out of the Eigen/NumericAL ratios; the operands "
              "in each library's own matrix type, whatever the case's layout")
        if cpp_divergent:
            print(f"{len(cpp_divergent)} case(s) where a C++ answer is not NumPy's (left out of the ratios):")
            for case in cpp_divergent:
                print(f"  #{case['index']} {case['op']} {case['dims']} {case['dtype']} "
                      + ", ".join(f"{'NumericAL' if library == 'numerical' else 'Eigen'}: {why}"
                                  for library, why in case["failed"].items()))
        cpp_failed = any("numerical" in case["failed"] for case in cpp_divergent)
        if cpp_rows:
            reproduce = f" --eigen {args.eigen}" + (" --eigen-openmp" if args.eigen_openmp else "") \
                + (f" --numerical-threads {args.numerical_threads}" if args.numerical_threads else "")
            report(cpp_rows, "Eigen/NumericAL from C++", "Eigen", "eigen_ratio", "numerical_cpp", "eigen", args,
                   reproduce, layouts=False)

    if args.json:
        summary = dict(seed=args.seed, largest=args.largest, numerical=nl.__version__, numpy=np.__version__,
                       machine=platform.machine(), rows=rows, divergent=divergent)
        if cpp:
            summary.update(cpp=cpp["info"], cpp_rows=cpp_rows, cpp_na=na, cpp_divergent=cpp_divergent)
        with open(args.json, "w") as out:
            json.dump(summary, out, indent=1)
    return 1 if divergent or cpp_failed else 0


if __name__ == "__main__":
    sys.exit(main())
