#!/usr/bin/env python3
"""NumericAL vs NumPy — differential validation + performance.

Each case first checks that NumericAL's result equals NumPy's within a tolerance
(a fast wrong answer is not a benchmark win); only matching cases are timed.
Inputs are built once, outside the timed region, so the figure is the operation
itself. `ratio` is numpy_time / numerical_time, so ratio < 1 means NumericAL is
slower — which is expected and honest: NumPy dispatches linear algebra to
BLAS/LAPACK (decades of blocked, vectorised kernels), whereas NumericAL is a
portable, dependency-free *reference* implementation (a naive index-walking
einsum and an unblocked LU). The value validated here is correctness within
tolerance and a measured, reproducible performance gap — not beating BLAS.

Timing, statistics and rendering come from SciForge's shared, dependency-free
benchmark substrate (sciforge.bench): `collect` gathers raw per-call samples, and
the median, the geometric-mean ratio with a bootstrap confidence interval, and the
ASCII box-plots are all derived from them. The substrate is never shipped — the
sibling ../sciforge/python is on PYTHONPATH via the Makefile (make bench-python).

Run from the repository root: make bench-python
"""
import sys

sys.path.insert(0, "python")
import numerical  # noqa: E402
# Shared dep-free bench substrate (compare/verdict + stats + box-plots); sibling on PYTHONPATH.
# compare() gates each case with the project's own `verify` callable (the subject==reference
# decision stays here, never in the core) and times subject vs reference paired.
from sciforge.bench import ascii_boxplot, compare, fmt, median_iqr, verdict  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover - benchmark is a manual dev tool
    print("NumPy is required for this benchmark: pip install numpy")
    sys.exit(0)

RTOL = 1e-7
ATOL = 1e-9

# Cases whose NumPy-vs-NumericAL sample distributions are drawn as box-plots (the gap, visually).
BOXPLOT_CASES = {"matmul (256×256)", "solve (64×64)", "eigh (48×48)", "svd (48×32)"}


# A case is (name, numerical_fn, numpy_fn, verify); `verify(numerical_result,
# numpy_result)` confirms correctness. The default compares the results directly;
# factorisations whose factors are not unique (sign/order) verify by
# reconstruction instead.
def values_match(numerical_result, numpy_result):
    return np.allclose(np.asarray(numerical_result), np.asarray(numpy_result),
                       rtol=RTOL, atol=ATOL)


def einsum_case(name, spec, *arrays):
    operands = [numerical.asarray(a) for a in arrays]
    return (name,
            lambda: numerical.einsum(spec, *operands),
            lambda: np.einsum(spec, *arrays),
            values_match)


def solve_case(name, a, b):
    sca, scb = numerical.asarray(a), numerical.asarray(b)
    return (name,
            lambda: numerical.solve(sca, scb),
            lambda: np.linalg.solve(a, b),
            values_match)


def determinant_case(name, a):
    t = numerical.asarray(a)
    return (name, lambda: numerical.determinant(t), lambda: np.linalg.det(a), values_match)


def inverse_case(name, a):
    t = numerical.asarray(a)
    return (name, lambda: numerical.inverse(t), lambda: np.linalg.inv(a), values_match)


def lstsq_case(name, a, b):
    t, tb = numerical.asarray(a), numerical.asarray(b)
    return (name, lambda: numerical.lstsq(t, tb)[0],
            lambda: np.linalg.lstsq(a, b, rcond=None)[0], values_match)


def cholesky_case(name, a):
    t = numerical.asarray(a)
    return (name, lambda: numerical.cholesky(t), lambda: np.linalg.cholesky(a), values_match)


def qr_case(name, a):
    t = numerical.asarray(a)

    def verify(numerical_result, _numpy_result):
        q, r = numerical_result
        return np.allclose(np.asarray(q) @ np.asarray(r), a, rtol=RTOL, atol=ATOL)

    return (name, lambda: numerical.qr(t), lambda: np.linalg.qr(a), verify)


def eigh_case(name, a):
    t = numerical.asarray(a)

    def verify(numerical_result, _numpy_result):
        values, vectors = numerical_result
        v = np.asarray(values)
        matrix = np.asarray(vectors)
        return np.allclose(matrix @ np.diag(v) @ matrix.T, a, rtol=RTOL, atol=ATOL)

    return (name, lambda: numerical.eigh(t), lambda: np.linalg.eigh(a), verify)


def svd_case(name, a):
    t = numerical.asarray(a)

    def verify(numerical_result, _numpy_result):
        u, s, vh = numerical_result  # Vh, as numpy.linalg.svd returns it
        return np.allclose(np.asarray(u) @ np.diag(np.asarray(s)) @ np.asarray(vh),
                           a, rtol=RTOL, atol=ATOL)

    return (name, lambda: numerical.svd(t), lambda: np.linalg.svd(a, full_matrices=False), verify)


def build_cases():
    r = np.random.default_rng(12345)
    dom = lambda n: r.random((n, n)) + n * np.eye(n)            # diagonally dominant (invertible)
    spd = lambda n: (lambda b: (b @ b.T) + n * np.eye(n))(r.random((n, n)))  # positive-definite
    sym = lambda n: (lambda b: b + b.T)(r.random((n, n)))        # symmetric
    return [
        einsum_case("dot (n=100k)", "i,i->", r.random(100_000), r.random(100_000)),
        einsum_case("outer (200×200)", "i,j->ij", r.random(200), r.random(200)),
        einsum_case("transpose (256×256)", "ij->ji", r.random((256, 256))),
        einsum_case("trace (300×300)", "ii->", r.random((300, 300))),
        einsum_case("sum-all (256×256)", "ij->", r.random((256, 256))),
        einsum_case("row-sums (256×256)", "ij->i", r.random((256, 256))),
        einsum_case("matvec (256×256)", "ij,j->i", r.random((256, 256)), r.random(256)),
        einsum_case("matmul (64×64)", "ij,jk->ik", r.random((64, 64)), r.random((64, 64))),
        einsum_case("matmul (256×256)", "ij,jk->ik", r.random((256, 256)), r.random((256, 256))),
        einsum_case("frobenius (200×200)", "ij,ij->", r.random((200, 200)), r.random((200, 200))),
        einsum_case("matmul complex (48×48)", "ij,jk->ik",
                    r.random((48, 48)) + 1j * r.random((48, 48)),
                    r.random((48, 48)) + 1j * r.random((48, 48))),
        solve_case("solve (64×64)", dom(64), r.random(64)),
        solve_case("solve complex (48×48)",
                   dom(48) + 1j * r.random((48, 48)), r.random(48) + 1j * r.random(48)),
        determinant_case("determinant (64×64)", dom(64)),
        inverse_case("inverse (64×64)", dom(64)),
        lstsq_case("lstsq (64×32)", r.random((64, 32)), r.random(64)),
        cholesky_case("cholesky (64×64)", spd(64)),
        qr_case("qr (64×32)", r.random((64, 32))),
        eigh_case("eigh (48×48)", sym(48)),
        svd_case("svd (48×32)", r.random((48, 32))),
    ]


def main():
    print(f"NumericAL {numerical.__version__} vs NumPy {np.__version__} "
          f"(Python {sys.version.split()[0]})\n")
    header = f"{'case':<26} {'NumPy':>11} {'NumericAL':>11} {'ratio':>10}"
    print(header)
    print("-" * len(header))

    cases = []      # one compare() Case per case (subject = NumericAL, reference = NumPy)
    matched = 0
    boxplots = []   # (name, numpy_samples, numerical_samples) for the shortlisted cases
    for name, numerical_fn, numpy_fn, verify in build_cases():
        # `verify(numerical_result, numpy_result) -> bool` is the equal callable: the gate stays
        # here (allclose within tolerance / factorisation reconstruction), never in the core.
        case = compare(name, numerical_fn, numpy_fn, verify)
        cases.append(case)
        if case.extra["mismatch"]:
            print(f"{name:<26} RESULT MISMATCH — correctness failure")
            continue
        matched += 1
        sc_samples = case.samples                      # subject = NumericAL
        np_samples = case.extra["reference_samples"]   # reference = NumPy
        t_sc = median_iqr(sc_samples)[0]
        t_np = median_iqr(np_samples)[0]
        print(f"{name:<26} {fmt(t_np, 8)} {fmt(t_sc, 8)} {t_np / t_sc:9.4f}x")
        if name in BOXPLOT_CASES:
            boxplots.append((name, np_samples, sc_samples))

    print("-" * len(header))
    print(f"all {matched} timed cases matched NumPy within rtol={RTOL}, atol={ATOL}")
    # The CI-aware aggregate (NumPy/NumericAL); for a reference implementation, being slower is
    # the expected, honest outcome — only a result mismatch fails. So we report the gap.
    result = verdict(cases, subject_label="NumericAL")
    if result.classification != "no-data":
        print(f"geometric-mean ratio (NumPy/NumericAL): {result.geomean:.4f}x "
              f"[{result.ci_low:.4f}-{result.ci_high:.4f}]  "
              f"⇒ NumericAL is ~{1.0 / result.geomean:.0f}× slower than NumPy on average")

    if boxplots:
        print("\nper-call time distributions — NumPy vs NumericAL (left = faster):")
        for name, np_samples, sc_samples in boxplots:
            print(f"  {name}:")
            print(ascii_boxplot([np_samples, sc_samples], ["NumPy", "NumericAL"], width=42))

    print("\nExpected: NumPy dispatches to BLAS/LAPACK; NumericAL is a portable, "
          "dependency-free reference\n(naive einsum, unblocked LU). Correctness "
          "within tolerance is the pass criterion here.")
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
