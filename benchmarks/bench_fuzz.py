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
        "einsum-reduce": m * n, "einsum-each": m * n, "solve": (2 / 3) * m ** 3 + 2 * m * m * max(n, 1), "det": (2 / 3) * m ** 3,
        "slogdet": (2 / 3) * m ** 3, "inv": 2 * m ** 3, "cholesky": m ** 3 / 3, "qr": 4 * m * n * n,
        "eigh": 9 * m ** 3, "eigvalsh": 4 * m ** 3, "svd": 14 * m * n * n, "svdvals": 4 * m * n * n,
        "lstsq": 14 * m * n * n + 2 * m * n * k, "eig": 25 * m ** 3,
        "fft": 5 * m * n * math.log2(n + 1), "ifft": 5 * m * n * math.log2(n + 1), "rfft": 2.5 * m * n * math.log2(n + 1),
        "irfft": 2.5 * m * n * math.log2(n + 1), "fft2": 5 * m * n * math.log2(m * n + 1),
        "expm": 20 * m ** 3, "logm": 60 * m ** 3, "sqrtm": 30 * m ** 3, "funm": 30 * m ** 3, "expm_frechet": 40 * m ** 3,
        "ldl": m ** 3 / 3, "schur": 25 * m ** 3, "qr-pivoting": 4 * m * n * n, "polar": 14 * m ** 3,
        "solve_sylvester": 25 * (m ** 3 + n ** 3) + 2 * m * n * (m + n), "solve_continuous_lyapunov": 60 * m ** 3,
        "solve_discrete_lyapunov": 2 * m ** 6 if m <= 24 else 120 * m ** 3, "solve_continuous_are": 400 * (m + n) ** 3,
        "solve_discrete_are": 400 * (m + n) ** 3, "solve_banded": 6 * m * n * n + 4 * m * n,
        "solveh_banded": 2 * m * n * n + 4 * m * n, "solve_toeplitz": 4 * m * m, "solve_circulant": 15 * m * math.log2(m + 1),
        "solve_triangular": m * m * max(n, 1), "eigh-generalized": 14 * m ** 3,
    }[op]


# Each operation: its weight in the draw, and how many dimensions it takes.
OPS = {
    "matmul": (3, 3), "einsum-product": (1, 3), "matvec": (2, 2), "norm": (1, 2), "einsum-reduce": (2, 2),
    "einsum-each": (1, 2),
    "solve": (3, 2), "det": (2, 1), "slogdet": (1, 1), "inv": (2, 1), "cholesky": (2, 1), "qr": (2, 2),
    "eigh": (2, 1), "eigvalsh": (1, 1), "svd": (2, 2), "svdvals": (1, 2), "lstsq": (2, 3), "eig": (1, 1),
    "fft": (3, 2), "ifft": (1, 2), "rfft": (2, 2), "irfft": (1, 2), "fft2": (1, 2),
}
# The transforms along an axis (numpy.fft's): `m` lines of `n` points, along the last axis or the first.
FFT_OPS = ("fft", "ifft", "rfft", "irfft")
# The einsum calls that sum nothing, each element one operand's or a product of two: an outer product, a row or a
# column broadcast over a matrix's rows, a transpose.
EACH_SPECS = ("i,j->ij", "ij,j->ij", "ij,i->ij", "ij->ji")
# Orders past which a drawn eig is not worth its time against the others.
LIMITS = {"eig": 400}

# The SciPy family (--family scipy): scipy.linalg's functions NumericAL answers, against SciPy, a population of its own
# (the NumPy family's draw, the one BENCHMARKS.md follows, is left as it was). Weights and dimensions as OPS's: a
# matrix order, and for the equations and the band systems a second dimension (inputs, columns, half-bandwidth).
SCIPY_OPS = {
    "expm": (2, 1), "logm": (2, 1), "sqrtm": (2, 1), "funm": (1, 1), "expm_frechet": (1, 1),
    "ldl": (2, 1), "schur": (2, 1), "qr-pivoting": (2, 2), "polar": (1, 1),
    "solve_sylvester": (2, 2), "solve_continuous_lyapunov": (1, 1), "solve_discrete_lyapunov": (1, 1),
    "solve_continuous_are": (2, 2), "solve_discrete_are": (2, 2),
    "solve_banded": (2, 2), "solveh_banded": (1, 2), "solve_toeplitz": (2, 1), "solve_circulant": (1, 1),
    "solve_triangular": (2, 2), "eigh-generalized": (2, 1),
}
# Orders past which a drawn case costs more than the rest of the draw: the pencils of order 2m + n, the Schur-based
# functions, the Kronecker-free but dense Sylvester solves.
SCIPY_LIMITS = {"solve_continuous_are": 300, "solve_discrete_are": 300, "logm": 600, "sqrtm": 600, "funm": 600,
                "expm_frechet": 600, "schur": 800, "solve_sylvester": 600, "solve_continuous_lyapunov": 600,
                "solve_discrete_lyapunov": 400}
# scipy.linalg works in double and single precision alike; the drawn dtypes as the NumPy family's.
SCIPY_DTYPES = DTYPES
# The operations both C++ sides have a function for: Eigen has no einsum, NumericAL's C++ API no matrix norm.
CPP_NA = {"einsum-product": "no Eigen function", "einsum-reduce": "no Eigen function", "einsum-each": "no Eigen function",
          "norm": "no NumericAL C++ function", **{op: "no Eigen function" for op in FFT_OPS + ("fft2",)}}
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


def draw_case(seed, index, largest, ops, family="numpy"):
    """Case `index` of the run seeded `seed`: the same in every process."""
    if family == "scipy":
        return draw_scipy_case(seed, index, largest, ops)
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
            else ["ij,jk->ik", "ij,kj->ik", "ji,jk->ik"][int(rng.integers(3))] if op == "einsum-product"
            else EACH_SPECS[int(rng.integers(len(EACH_SPECS)))] if op == "einsum-each"
            else ["axis=-1", "axis=0"][int(rng.integers(2))] if op in FFT_OPS else None}


def draw_scipy_case(seed, index, largest, ops):
    """Case `index` of the SciPy family's run seeded `seed`: its operation, order (and second dimension), dtype and
    layout; a second dimension is the inputs of a Riccati equation (a few), a band's half-width (below the order), the
    columns of a right-hand side (a few) or a Sylvester equation's second order."""
    rng = np.random.default_rng([seed, index, 0x5c1])
    op = weighted(rng, [(name, SCIPY_OPS[name][0]) for name in ops])
    dims = [min(log_size(rng, largest), SCIPY_LIMITS.get(op, largest)) for _ in range(SCIPY_OPS[op][1])]
    if op in ("solve_continuous_are", "solve_discrete_are"):
        dims[1] = 1 + dims[1] % min(8, dims[0])
    elif op in ("solve_banded", "solveh_banded"):
        dims[1] = dims[1] % max(1, min(dims[0], 33))
    elif op == "solve_triangular":
        dims[1] = 1 + dims[1] % 17
    while flops(op, dims) > BUDGET:
        dims = [max(1, int(d * 0.8)) for d in dims]
    return {"index": index, "op": op, "dims": dims, "dtype": weighted(rng, SCIPY_DTYPES),
            "layout": weighted(rng, LAYOUTS), "spec": None}


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
    elif op == "einsum-each":
        spec = case["spec"]
        shapes = {"i,j->ij": ((m,), (n,)), "ij,j->ij": ((m, n), (n,)), "ij,i->ij": ((m, n), (m,)), "ij->ji": ((m, n),)}[spec]
        operands = [matrix(rng, shape, dtype) for shape in shapes]
        ours, theirs = (lambda *xs: nl.einsum(spec, *xs)), (lambda *xs: np.einsum(spec, *xs))
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
    elif op in FFT_OPS:
        axis = -1 if case["spec"] == "axis=-1" else 0
        single = dtype in (np.float32, np.complex64)
        kind = {"rfft": np.float32 if single else np.float64, "irfft": np.complex64 if single else np.complex128}.get(op, dtype)
        points = (n // 2) + 1 if op == "irfft" else n  # irfft's input: the n/2 + 1 frequencies of n points
        operands = [matrix(rng, (m, points) if axis == -1 else (points, m), kind)]
        ours_fn, theirs_fn = getattr(nl.fft, op), getattr(np.fft, op)
        if op == "irfft":
            ours, theirs = (lambda a: ours_fn(a, n, axis)), (lambda a: theirs_fn(a, n, axis))
        else:
            ours, theirs = (lambda a: ours_fn(a, axis=axis)), (lambda a: theirs_fn(a, axis=axis))
    elif op == "fft2":
        operands = [matrix(rng, (m, n), dtype)]
        ours, theirs = nl.fft.fft2, np.fft.fft2
    else:  # pragma: no cover - every drawn operation is listed above
        raise ValueError(op)
    arrived = [arrive(a, case["layout"]) for a in operands]
    ours_in = [nl.asarray(a) for a in arrived] if case["layout"] == "tensor" else arrived
    return (lambda: ours(*ours_in)), (lambda: theirs(*arrived)), operands


def build_scipy(case):
    """The SciPy family's case: its operands, built so that the answer is well defined (well-conditioned matrices,
    spectra clear of the branch cuts and of each other's negatives, stabilisable Riccati equations, diagonally dominant
    band and Toeplitz systems), and its two calls: (numerical_call, scipy_call, operands)."""
    import scipy.linalg as sl
    rng = np.random.default_rng([case["index"], 0x5eed])
    op, dims, dtype = case["op"], case["dims"], np.dtype(case["dtype"])
    m = dims[0]
    n = dims[1] if len(dims) > 1 else m

    def random(*shape):
        return matrix(rng, shape, dtype)

    def near_identity(order):  # eigenvalues within about 1 of 1: off the negative axis, apart enough
        return (np.eye(order) + random(order, order) / (2 * math.sqrt(order))).astype(dtype)

    def hermitian(order):
        b = random(order, order)
        return ((b + b.conj().T) / 2).astype(dtype)

    def spd(order):
        b = random(order, order)
        return (b @ b.conj().T / order + np.eye(order)).astype(dtype)

    def shifted(order):  # eigenvalues near `order` / 2 on the right of the imaginary axis
        return (random(order, order) + (order / 2 + 1) * np.eye(order)).astype(dtype)

    if op in ("expm", "logm", "sqrtm"):
        operands = [near_identity(m) if op != "expm" else (random(m, m) * 10.0 ** rng.uniform(-3, 0.5) / math.sqrt(m)).astype(dtype)]
        ours, theirs = getattr(nl, op), getattr(sl, op)
    elif op == "funm":
        operands = [near_identity(m)]
        ours, theirs = (lambda a: nl.funm(a, np.exp, disp=False)), (lambda a: sl.funm(a, np.exp, disp=False))
    elif op == "expm_frechet":
        operands = [(random(m, m) / math.sqrt(m)).astype(dtype), random(m, m)]
        ours = lambda a, e: nl.expm_frechet(a, e, compute_expm=False)  # noqa: E731
        theirs = lambda a, e: sl.expm_frechet(a, e, compute_expm=False)  # noqa: E731
    elif op == "ldl":
        operands = [hermitian(m)]
        ours, theirs = nl.ldl, sl.ldl
    elif op == "schur":
        operands = [random(m, m)]
        ours, theirs = nl.schur, sl.schur
    elif op == "qr-pivoting":
        operands = [random(m, n)]
        ours = lambda a: nl.qr(a, pivoting=True)  # noqa: E731
        theirs = lambda a: sl.qr(a, pivoting=True, mode="economic")  # noqa: E731
    elif op == "polar":
        operands = [shifted(m)]
        ours, theirs = nl.polar, sl.polar
    elif op == "solve_sylvester":
        operands = [shifted(m), shifted(n), random(m, n)]
        ours, theirs = nl.solve_sylvester, sl.solve_sylvester
    elif op == "solve_continuous_lyapunov":
        operands = [shifted(m), spd(m)]
        ours, theirs = nl.solve_continuous_lyapunov, sl.solve_continuous_lyapunov
    elif op == "solve_discrete_lyapunov":
        operands = [(random(m, m) / (2 * math.sqrt(m))).astype(dtype), spd(m)]  # spectral radius about 1/2
        ours, theirs = nl.solve_discrete_lyapunov, sl.solve_discrete_lyapunov
    elif op in ("solve_continuous_are", "solve_discrete_are"):
        operands = [(random(m, m) / math.sqrt(m)).astype(dtype), random(m, n), spd(m), spd(n)]
        ours, theirs = getattr(nl, op), getattr(sl, op)
    elif op == "solve_banded":
        half = n
        ab = random(2 * half + 1, m)
        ab[half] += 2 * (2 * half + 1)  # diagonally dominant: no pivot near zero
        operands = [ab.astype(dtype), random(m)]
        ours = lambda a, b: nl.solve_banded((half, half), a, b)  # noqa: E731
        theirs = lambda a, b: sl.solve_banded((half, half), a, b)  # noqa: E731
    elif op == "solveh_banded":
        half = n
        ab = random(half + 1, m)
        ab[half] = np.abs(ab[half].real) + 2 * (half + 1)  # a real, dominant diagonal: positive definite
        operands = [ab.astype(dtype), random(m)]
        ours, theirs = nl.solveh_banded, sl.solveh_banded
    elif op == "solve_toeplitz":
        c, r = random(m), random(m)
        c[0] = r[0] = 2 * m  # diagonally dominant: every leading minor nonsingular
        operands = [c.astype(dtype), r.astype(dtype), random(m)]
        ours = lambda c, r, b: nl.solve_toeplitz((c, r), b)  # noqa: E731
        theirs = lambda c, r, b: sl.solve_toeplitz((c, r), b)  # noqa: E731
    elif op == "solve_circulant":
        c = random(m)
        c[0] += 2 * m  # its eigenvalues (c's transform) clear of 0
        operands = [c.astype(dtype), random(m)]
        ours, theirs = nl.solve_circulant, sl.solve_circulant
    elif op == "solve_triangular":
        operands = [(np.triu(random(m, m)) + m * np.eye(m)).astype(dtype), random(m, n)]
        ours = lambda a, b: nl.solve_triangular(a, b)  # noqa: E731
        theirs = lambda a, b: sl.solve_triangular(a, b)  # noqa: E731
    elif op == "eigh-generalized":
        operands = [hermitian(m), spd(m)]
        ours = lambda a, b: nl.eigh(a, b=b)  # noqa: E731
        theirs = lambda a, b: sl.eigh(a, b)  # noqa: E731
    else:  # pragma: no cover - every drawn operation is listed above
        raise ValueError(op)
    arrived = [arrive(a, case["layout"]) for a in operands]
    ours_in = [nl.asarray(a) for a in arrived] if case["layout"] == "tensor" else arrived
    return (lambda: ours(*ours_in)), (lambda: theirs(*arrived)), operands


def agrees_scipy(case, ours, theirs, operands):
    """Whether NumericAL's answer is SciPy's within the dtype's tolerance (wider where the problem's conditioning
    grows: the matrix functions and equations), or by what defines it where it is not unique (LDLᴴ, the Schur form,
    QR with pivoting: their reconstruction)."""
    op, dims = case["op"], case["dims"]
    tol = tolerance(case["dtype"], max(dims))
    a = np.asarray(operands[0]).astype(complex)
    if op == "ldl":
        lu, d, perm = (np.asarray(x) for x in ours)
        triangle = lu[list(perm)]
        return relative(lu @ d @ lu.conj().T, a) < tol * 10 and np.allclose(np.diag(triangle), 1) \
            and np.allclose(triangle, np.tril(triangle))
    if op == "schur":
        t, z = (np.asarray(x) for x in ours)
        return relative(z @ t @ z.conj().T, a) < tol * 10 and np.all(np.tril(t, -2) == 0)
    if op == "qr-pivoting":
        q, r, p = ours
        q, r = np.asarray(q), np.asarray(r)
        return relative(q @ r, a[:, list(p)]) < tol * 10 and np.allclose(r, np.triu(r))
    if op == "polar":
        return all(relative(x, y) < tol * 10 for x, y in zip(ours, theirs))
    if op == "eigh-generalized":
        values, vectors = (np.asarray(x) for x in ours)
        b = np.asarray(operands[1]).astype(complex)
        return relative(values, theirs[0]) < tol * 10 and relative(a @ vectors, (b @ vectors) * values) < tol * 100
    if op == "funm":
        (f, err), (want, their_err) = ours, theirs
        return relative(f, want) < max(tol * 10, 1e3 * max(err, their_err))
    widened = op in ("logm", "sqrtm", "expm_frechet", "solve_sylvester", "solve_continuous_lyapunov",
                     "solve_discrete_lyapunov", "solve_continuous_are", "solve_discrete_are")
    return relative(ours, theirs) < tol * (100 if widened else 10)


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
        ours, theirs, _ = (build_scipy if args.family == "scipy" else build)(case)
        times[case["index"]] = fastest(ours if args.side == "numerical" else theirs, args.rounds)
    json.dump(times, sys.stdout)


def cases_of(args):
    ops = args.ops.split(",") if args.ops else list(SCIPY_OPS if args.family == "scipy" else OPS)
    indices = [int(i) for i in args.only.split(",")] if args.only else range(args.cases)
    return [draw_case(args.seed, i, args.largest, ops, args.family) for i in indices]


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
    parser.add_argument("--family", choices=("numpy", "scipy"), default="numpy",
                        help="numpy (the default): NumPy's operations against NumPy; scipy: scipy.linalg's against SciPy")
    parser.add_argument("--ops", default="", help="a comma-separated subset of: " + ", ".join(OPS)
                        + "; with --family scipy, of: " + ", ".join(SCIPY_OPS))
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
    if args.family == "scipy" and args.eigen:
        print("--eigen compares NumPy's operations from C++: not with --family scipy")
        return 2
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
             str(args.rounds), "--cases", str(args.cases), "--family", args.family] + (["--ops", args.ops] if args.ops else []) \
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

    scipy = args.family == "scipy"
    if scipy:
        import scipy as scipy_module
        reference = f"SciPy {scipy_module.__version__} (NumPy {np.__version__})"
    else:
        reference = f"NumPy {np.__version__}"
    print(f"NumericAL {nl.__version__} ({nl.gemm_path()}, {nl.get_num_threads()} threads) vs {reference}, "
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
        ours_fn, theirs_fn, operands = (build_scipy if scipy else build)(case)
        theirs = None
        try:
            theirs = theirs_fn()
            ok = (agrees_scipy if scipy else agrees)(case, ours_fn(), theirs, operands)
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
        print(f"\n{len(divergent)} case(s) where NumericAL's answer is not {'SciPy' if scipy else 'NumPy'}'s (not timed):")
        for case in divergent:
            print(f"  #{case['index']} {case['op']} {case['dims']} {case['dtype']} {case['layout']} "
                  f"{case.get('spec') or ''} {case.get('error', '')}")
    if not rows:
        return 1

    if scipy:
        report(rows, "SciPy/NumericAL", "SciPy", "ratio", "numerical", "numpy", args, " --family scipy")
    else:
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
        summary = dict(seed=args.seed, largest=args.largest, family=args.family, numerical=nl.__version__,
                       numpy=np.__version__, machine=platform.machine(), rows=rows, divergent=divergent)
        if scipy:
            summary["scipy"] = scipy_module.__version__
        if cpp:
            summary.update(cpp=cpp["info"], cpp_rows=cpp_rows, cpp_na=na, cpp_divergent=cpp_divergent)
        with open(args.json, "w") as out:
            json.dump(summary, out, indent=1)
    return 1 if divergent or cpp_failed else 0


if __name__ == "__main__":
    sys.exit(main())
