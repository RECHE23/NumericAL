"""Differential fuzzing of the decompositions against NumPy (LAPACK), on every matrix-product path.

Each case draws a matrix from a family that breaks weak algorithms (clustered or repeated spectra, decoupled
blocks, graded singular values, rank deficiency, extreme scales, exact integers, complex data) at a size biased
towards the block and leaf boundaries of the blocked and divide-and-conquer code, then checks every operation
against its defining property and against NumPy. The bounds are backward-error bounds, a multiple of the order
times the machine epsilon times the matrix's norm: what a stable algorithm guarantees whatever the
conditioning, and what a wrong merge, deflation or update misses by orders of magnitude.

Every path the processor runs is exercised in turn (portable, AVX2, AVX-512, the Apple AMX). A failure prints
the seed and the case, which reproduce it alone:

    python3 fuzz/linalg.py --seed S --case C
"""
import argparse
import itertools
import math
import os
import sys
import time

import numpy as np

import numerical

EPS = np.finfo(float).eps
SLACK = 50.0  # the multiple of n·ε·‖A‖ a backward-stable result stays within
BOUNDARIES = (1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 191, 192, 193, 255, 256, 257)
PATHS = ("portable", "avx2", "avx512", "amx")


def size(rng, largest):
    if rng.random() < 0.6:
        choices = [b for b in BOUNDARIES if b <= largest]
        return int(rng.choice(choices))
    return int(rng.integers(1, largest + 1))


def orthonormal(rng, n, cplx):
    a = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
    q, _ = np.linalg.qr(a)
    return q


def general(rng, m, n, kind, cplx):
    """An m × n matrix of the family `kind`."""
    k = min(m, n)
    if kind == "normal":
        a = rng.standard_normal((m, n))
        return a + 1j * rng.standard_normal((m, n)) if cplx else a
    if kind == "integers":
        return rng.integers(-3, 4, size=(m, n)).astype(complex if cplx else float)
    if kind in ("graded", "rankdef", "clustered_values", "repeated_values"):
        if kind == "graded":
            s = np.logspace(0, -14, k)
        elif kind == "rankdef":
            s = np.concatenate([rng.uniform(0.5, 2.0, max(k // 2, 1)), np.zeros(k - max(k // 2, 1))])
        elif kind == "clustered_values":
            s = np.repeat(rng.uniform(0.1, 2.0, 3), k)[:k] + rng.choice([0.0, 1e-15, 1e-10], k)
        else:
            s = np.repeat([1.0, 2.0, 3.0], k)[:k]
        u = orthonormal(rng, m, cplx)[:, :k]
        v = orthonormal(rng, n, cplx)[:, :k]
        return (u * s) @ v.conj().T
    if kind == "zero":
        return np.zeros((m, n), dtype=complex if cplx else float)
    if kind == "bidiagonal":
        a = np.zeros((m, n), dtype=complex if cplx else float)
        idx = np.arange(k)
        a[idx, idx] = rng.choice([0.0, 1.0, -2.0, 1e-9], k)
        a[idx[:-1], idx[:-1] + 1] = rng.choice([0.0, 1.0, 1e-12], max(k - 1, 0))
        return a
    if kind == "scaled_rows":
        return (rng.standard_normal((m, n)).T * np.logspace(-8, 8, m)).T
    raise ValueError(kind)


def hermitian(rng, n, kind, cplx):
    """An n × n Hermitian matrix of the family `kind`."""
    if kind == "normal":
        a = general(rng, n, n, "normal", cplx)
        return (a + a.conj().T) / 2
    if kind == "integers":
        a = general(rng, n, n, "integers", cplx)
        return (a + a.conj().T) / 2
    if kind in ("clustered", "repeated", "graded", "zero_eigenvalues"):
        if kind == "clustered":
            w = np.repeat(rng.uniform(-2.0, 2.0, 4), n)[:n] + rng.choice([0.0, 1e-15, 1e-12, 1e-8], n)
        elif kind == "repeated":
            w = np.repeat([-1.0, 0.0, 2.0], n)[:n]
        elif kind == "graded":
            w = np.logspace(0, -14, n) * rng.choice([-1.0, 1.0], n)
        else:
            w = np.concatenate([rng.standard_normal(max(n - n // 3, 1)), np.zeros(n // 3)])[:n]
        q = orthonormal(rng, n, cplx)
        a = (q * w) @ q.conj().T
        return (a + a.conj().T) / 2
    if kind == "blocks":  # decoupled blocks: a split falls where nothing couples
        a = np.zeros((n, n), dtype=complex if cplx else float)
        start = 0
        while start < n:
            width = int(rng.integers(1, max(n // 3, 1) + 1))
            block = general(rng, width, width, "normal", cplx)[: n - start, : n - start]
            a[start : start + block.shape[0], start : start + block.shape[0]] = (block + block.conj().T) / 2
            start += block.shape[0]
        return a
    if kind == "wilkinson":
        m = (n - 1) / 2
        a = np.diag(np.abs(np.arange(n) - m)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
        return a.astype(complex) if cplx else a
    if kind == "arrow":
        a = np.diag(rng.standard_normal(n))
        a[0, 1:] = a[1:, 0] = rng.standard_normal(n - 1) * 1e-3
        return a.astype(complex) if cplx else a
    raise ValueError(kind)


GENERAL = ("normal", "integers", "graded", "rankdef", "clustered_values", "repeated_values", "zero", "bidiagonal",
           "scaled_rows")
HERMITIAN = ("normal", "integers", "clustered", "repeated", "graded", "zero_eigenvalues", "blocks", "wilkinson",
             "arrow")
SCALES = (1.0, 1.0, 1.0, 1e150, 1e-150)


def as_np(t):
    return np.asarray(t)


def norm(a):
    return float(np.linalg.norm(a)) if a.size else 0.0


class Failure(Exception):
    pass


def check(condition, what):
    if not condition:
        raise Failure(what)


def bound(n, scale):
    return SLACK * max(n, 1) * EPS * scale


def run_eigh(a):
    n = a.shape[0]
    w, v = numerical.eigh(numerical.asarray(a))
    w, v = as_np(w), as_np(v)
    size_a = norm(a)
    check(np.all(np.diff(w) >= 0), "eigenvalues not ascending")
    check(norm(a @ v - v * w) <= bound(n, size_a), f"eigh residual {norm(a @ v - v * w) / (n * EPS * max(size_a, 1e-300)):.3g} n·ε·‖A‖")
    check(norm(v.conj().T @ v - np.eye(n)) <= bound(n, 1.0), f"eigh orthogonality {norm(v.conj().T @ v - np.eye(n)) / (n * EPS):.3g} n·ε")
    want = np.linalg.eigvalsh(a)
    check(np.max(np.abs(w - want), initial=0.0) <= bound(n, size_a), "eigenvalues differ from NumPy's")


def run_svd(a):
    m, n = a.shape
    k = min(m, n)
    u, s, vh = numerical.svd(numerical.asarray(a))
    u, s, vh = as_np(u), as_np(s), as_np(vh)
    size_a = norm(a)
    check(np.all(np.diff(s) <= 0) and np.all(s >= 0), "singular values not descending and non-negative")
    rebuilt = (u[:, :k] * s) @ vh[:k]
    check(norm(rebuilt - a) <= bound(max(m, n), size_a), f"svd reconstruction {norm(rebuilt - a) / (max(m, n) * EPS * max(size_a, 1e-300)):.3g}")
    check(norm(u[:, :k].conj().T @ u[:, :k] - np.eye(k)) <= bound(max(m, n), 1.0), "U not orthonormal")
    check(norm(vh[:k] @ vh[:k].conj().T - np.eye(k)) <= bound(max(m, n), 1.0), "V not orthonormal")
    want = np.linalg.svd(a, compute_uv=False)
    check(np.max(np.abs(s - want), initial=0.0) <= bound(max(m, n), size_a), "singular values differ from NumPy's")


def run_qr(a):
    m, n = a.shape
    q, r = numerical.qr(numerical.asarray(a))
    q, r = as_np(q), as_np(r)
    k = min(m, n)
    check(norm(q @ r - a) <= bound(max(m, n), norm(a)), "QR does not reconstruct")
    check(norm(q.conj().T @ q - np.eye(k)) <= bound(max(m, n), 1.0), "Q not orthonormal")
    check(np.allclose(np.tril(r, -1), 0), "R not upper triangular")


def well_conditioned(a, limit=1e10):
    return a.size and np.isfinite(np.linalg.cond(a)) and np.linalg.cond(a) < limit


def run_solve(a, rng):
    n = a.shape[0]
    if not well_conditioned(a):
        return
    b = rng.standard_normal((n, 3)) + (1j * rng.standard_normal((n, 3)) if np.iscomplexobj(a) else 0)
    b = b * norm(a)
    x = as_np(numerical.solve(numerical.asarray(a), numerical.asarray(b)))
    ours = norm(a @ x - b) / (norm(a) * norm(x) + norm(b))
    reference = np.linalg.solve(a, b)
    theirs = norm(a @ reference - b) / (norm(a) * norm(reference) + norm(b))
    check(ours <= max(bound(n, 1.0), 20 * theirs), f"solve backward error {ours:.3g} (NumPy {theirs:.3g})")
    inverse = as_np(numerical.inverse(numerical.asarray(a)))
    ours = norm(a @ inverse - np.eye(n)) / (norm(a) * norm(inverse))
    reference = np.linalg.inv(a)
    theirs = norm(a @ reference - np.eye(n)) / (norm(a) * norm(reference))
    check(ours <= max(bound(n, 1.0), 20 * theirs), f"inverse error {ours:.3g} (NumPy {theirs:.3g})")
    sign, logdet = numerical.slogdet(numerical.asarray(a))
    want_sign, want_logdet = np.linalg.slogdet(a)
    check(abs(logdet - want_logdet) <= 1e-8 * max(n, 1) * max(1.0, abs(want_logdet)), "slogdet differs from NumPy's")
    check(abs(sign - want_sign) <= 1e-8 * max(n, 1), "the determinant's sign differs from NumPy's")


def run_cholesky(a):
    n = a.shape[0]
    shift = norm(a) ** 2 / max(n, 1) if norm(a) > 0 else 1.0  # of the order of A Aᴴ, which may be singular
    spd = a @ a.conj().T + shift * np.eye(n)
    if not np.isfinite(spd).all():
        return  # A Aᴴ overflowed: there is no matrix to factor
    lower = as_np(numerical.cholesky(numerical.asarray(spd)))
    check(np.allclose(np.triu(lower, 1), 0), "Cholesky factor not lower triangular")
    check(norm(lower @ lower.conj().T - spd) <= bound(n, norm(spd)), "Cholesky does not reconstruct")


def run_lstsq(a, rng):
    m, n = a.shape
    if m < n or not well_conditioned(a, 1e8):
        return
    b = rng.standard_normal(m) + (1j * rng.standard_normal(m) if np.iscomplexobj(a) else 0)
    x = as_np(numerical.lstsq(numerical.asarray(a), numerical.asarray(b)))
    want = np.linalg.lstsq(a, b, rcond=None)[0]
    check(norm(x - want) <= 1e-6 * max(norm(want), 1e-300) + bound(m, 1.0), "lstsq differs from NumPy's")


def case(seed, index, largest):
    """The matrices of one case: a general m × n and a Hermitian n × n, at a common scale."""
    rng = np.random.default_rng([seed, index])
    cplx = bool(rng.random() < 0.3)
    scale = float(rng.choice(SCALES))
    m, n = size(rng, largest), size(rng, largest)
    general_kind = str(rng.choice(GENERAL))
    hermitian_kind = str(rng.choice(HERMITIAN))
    g = general(rng, m, n, general_kind, cplx) * scale
    h = hermitian(rng, n, hermitian_kind, cplx) * scale
    square = general(rng, n, n, general_kind, cplx) * scale
    label = f"{'complex ' if cplx else ''}{general_kind} {m}×{n} / {hermitian_kind} {n}, scale {scale:g}"
    return rng, label, g, h, square


def run_case(seed, index, largest):
    rng, label, g, h, square = case(seed, index, largest)
    operations = (("eigh", lambda: run_eigh(h)), ("svd", lambda: run_svd(g)), ("qr", lambda: run_qr(g)),
                  ("solve", lambda: run_solve(square, rng)), ("cholesky", lambda: run_cholesky(square)),
                  ("lstsq", lambda: run_lstsq(g, rng)))
    failures = []
    for name, operation in operations:
        try:
            operation()
        except Failure as failure:
            failures.append(f"{name}: {failure}")
        except Exception as error:  # an exception NumPy does not raise on the same input is a failure too
            failures.append(f"{name}: {type(error).__name__}: {error}")
    return label, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=int(time.time()))
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--case", type=int, help="run this one case only")
    parser.add_argument("--largest", type=int, default=257, help="the largest order drawn")
    args = parser.parse_args()
    paths = []
    for path in PATHS:
        if numerical.limit_gemm_path(path) == path:
            paths.append(path)
    indices = [args.case] if args.case is not None else range(args.cases)
    print(f"seed {args.seed}, {len(indices)} cases up to order {args.largest}, paths {', '.join(paths)}", flush=True)
    failed = 0
    started = time.time()
    with np.errstate(all="ignore"):
        for path, index in itertools.product(paths, indices):
            numerical.limit_gemm_path(path)
            label, failures = run_case(args.seed, index, args.largest)
            for failure in failures:
                failed += 1
                print(f"FAIL [{path}] case {index} ({label}) {failure}\n"
                      f"     repro: NUMERICAL_GEMM={path} python3 fuzz/linalg.py --seed {args.seed} --case {index} --largest {args.largest}",
                      flush=True)
    numerical.limit_gemm_path("amx")
    print(f"{failed} failure(s) in {time.time() - started:.0f} s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
