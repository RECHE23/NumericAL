#!/usr/bin/env python3
"""NumericAL vs NumPy (and SciPy's linalg, where SciPy is installed) — differential validation + performance.

Each case first checks that NumericAL's result equals NumPy's within a tolerance
(a fast wrong answer is not a benchmark win); only matching cases are timed.
Inputs are built once, outside the timed region, so the figure is the operation
itself. `ratio` is numpy_time / numerical_time: above 1, NumericAL is faster;
below 1, NumPy (its BLAS and LAPACK: OpenBLAS, or Accelerate on macOS) is. Calls
of a few microseconds move by tens of percent from one run to the next on a
laptop's cores; repeat a run before reading a small case's ratio.

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
BOXPLOT_CASES = {"matmul (256×256)", "solve (64×64)", "eigh (48×48)", "svd (48×32)", "fft (1024)"}


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

    return (name, lambda: numerical.svd(t, full_matrices=False), lambda: np.linalg.svd(a, full_matrices=False), verify)


def fft_case(name, ours, theirs, x, rtol=1e-12):
    """A transform of numerical.fft against numpy.fft's of the same name: equal within `rtol` of the largest output
    (a sum of n terms rounds by about log2(n) units of the dtype's precision)."""
    t = numerical.asarray(x)

    def verify(numerical_result, numpy_result):
        got, want = np.asarray(numerical_result), np.asarray(numpy_result)
        return got.shape == want.shape and np.abs(got - want).max() <= rtol * np.abs(want).max()

    return name, lambda: ours(t), lambda: theirs(x), verify


def build_cases():
    r = np.random.default_rng(12345)
    dom = lambda n: r.random((n, n)) + n * np.eye(n)            # diagonally dominant (invertible)
    spd = lambda n: (lambda b: (b @ b.T) + n * np.eye(n))(r.random((n, n)))  # positive-definite
    sym = lambda n: (lambda b: b + b.T)(r.random((n, n)))        # symmetric
    cplx = lambda shape: r.standard_normal(shape) + 1j * r.standard_normal(shape)  # complex Gaussian
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
        fft_case("fft (1024)", numerical.fft.fft, np.fft.fft, cplx(1024)),
        fft_case("fft (1000)", numerical.fft.fft, np.fft.fft, cplx(1000)),
        fft_case("fft (1009, Bluestein)", numerical.fft.fft, np.fft.fft, cplx(1009)),
        fft_case("fft (65536)", numerical.fft.fft, np.fft.fft, cplx(65536)),
        fft_case("ifft (4096)", numerical.fft.ifft, np.fft.ifft, cplx(4096)),
        fft_case("fft complex64 (4096)", numerical.fft.fft, np.fft.fft, cplx(4096).astype(np.complex64), rtol=1e-5),
        fft_case("fft (64×64 lines)", numerical.fft.fft, np.fft.fft, cplx((64, 64))),
        fft_case("fft axis 0 (1024×64)", lambda t: numerical.fft.fft(t, axis=0), lambda a: np.fft.fft(a, axis=0),
                 cplx((1024, 64))),
        fft_case("fft2 (512×512)", numerical.fft.fft2, np.fft.fft2, cplx((512, 512))),
        fft_case("rfft (4096)", numerical.fft.rfft, np.fft.rfft, r.standard_normal(4096)),
        fft_case("irfft (2049→4096)", numerical.fft.irfft, np.fft.irfft, cplx(2049)),
        fft_case("rfft2 (512×512)", numerical.fft.rfft2, np.fft.rfft2, r.standard_normal((512, 512))),
    ]


def build_scipy_cases(sl):
    """scipy.linalg's functions NumericAL answers, against SciPy, at fixed sizes: compared directly where the answer
    is unique, by reconstruction where it is not (LDLᴴ, the Schur form, QR with pivoting)."""
    r = np.random.default_rng(54321)
    near = lambda n: np.eye(n) + r.standard_normal((n, n)) / (2 * np.sqrt(n))  # eigenvalues near 1
    shifted = lambda n: r.standard_normal((n, n)) + (n / 2 + 1) * np.eye(n)    # right half-plane, apart
    spd = lambda n: (lambda b: b @ b.T / n + np.eye(n))(r.standard_normal((n, n)))
    sym = lambda n: (lambda b: (b + b.T) / 2)(r.standard_normal((n, n)))

    def case(name, ours, theirs, *arrays, verify=values_match):
        operands = [numerical.asarray(a) for a in arrays]
        return name, (lambda: ours(*operands)), (lambda: theirs(*arrays)), verify

    def rebuilt(product, want):
        return lambda got, _: np.allclose(product(*(np.asarray(x) for x in got[:2])), want, rtol=RTOL, atol=ATOL)

    ldl_a, schur_a, qr_a = sym(64), r.standard_normal((64, 64)), r.standard_normal((64, 32))
    band = r.standard_normal((7, 1000))
    band[3] += 14.0
    hband = r.standard_normal((4, 1000))
    hband[3] = np.abs(hband[3]) + 8.0
    toeplitz_c, toeplitz_r = r.standard_normal(500), r.standard_normal(500)
    toeplitz_c[0] = toeplitz_r[0] = 1000.0
    circulant = r.standard_normal(1024)
    circulant[0] += 2048.0
    funm_exp = lambda f: (lambda a: f(a, np.exp))  # noqa: E731
    return [
        case("expm (64×64)", numerical.expm, sl.expm, r.standard_normal((64, 64)) / 8),
        case("logm (64×64)", numerical.logm, sl.logm, near(64)),
        case("sqrtm (64×64)", numerical.sqrtm, sl.sqrtm, near(64)),
        case("funm exp (32×32)", funm_exp(numerical.funm), funm_exp(sl.funm), near(32)),
        case("expm_frechet (48×48)", lambda a, e: numerical.expm_frechet(a, e, compute_expm=False),
             lambda a, e: sl.expm_frechet(a, e, compute_expm=False), r.standard_normal((48, 48)) / 7, r.standard_normal((48, 48))),
        case("ldl (64×64)", numerical.ldl, sl.ldl, ldl_a, verify=rebuilt(lambda lu, d: lu @ d @ lu.T, ldl_a)),
        case("schur (64×64)", numerical.schur, sl.schur, schur_a, verify=rebuilt(lambda t, z: z @ t @ z.T, schur_a)),
        case("qr pivoting (64×32)", lambda a: numerical.qr(a, pivoting=True), lambda a: sl.qr(a, pivoting=True, mode="economic"),
             qr_a, verify=lambda got, _: np.allclose(np.asarray(got[0]) @ np.asarray(got[1]), qr_a[:, list(got[2])], rtol=RTOL, atol=ATOL)),
        case("polar (48×48)", numerical.polar, sl.polar, shifted(48),
             verify=lambda got, want: all(np.allclose(np.asarray(x), y, rtol=RTOL, atol=ATOL) for x, y in zip(got, want))),
        case("sylvester (48, 32)", numerical.solve_sylvester, sl.solve_sylvester, shifted(48), shifted(32), r.standard_normal((48, 32))),
        case("lyapunov (48×48)", numerical.solve_continuous_lyapunov, sl.solve_continuous_lyapunov, shifted(48), spd(48)),
        case("discrete lyapunov (32)", numerical.solve_discrete_lyapunov, sl.solve_discrete_lyapunov,
             r.standard_normal((32, 32)) / 12, spd(32)),
        case("continuous are (32, 4)", numerical.solve_continuous_are, sl.solve_continuous_are,
             r.standard_normal((32, 32)) / 6, r.standard_normal((32, 4)), spd(32), spd(4)),
        case("discrete are (32, 4)", numerical.solve_discrete_are, sl.solve_discrete_are,
             r.standard_normal((32, 32)) / 6, r.standard_normal((32, 4)), spd(32), spd(4)),
        case("solve_banded (1000, 3+3)", lambda ab, b: numerical.solve_banded((3, 3), ab, b),
             lambda ab, b: sl.solve_banded((3, 3), ab, b), band, r.standard_normal(1000)),
        case("solveh_banded (1000, 3)", numerical.solveh_banded, sl.solveh_banded, hband, r.standard_normal(1000)),
        case("solve_toeplitz (500)", lambda c, rr, b: numerical.solve_toeplitz((c, rr), b),
             lambda c, rr, b: sl.solve_toeplitz((c, rr), b), toeplitz_c, toeplitz_r, r.standard_normal(500)),
        case("solve_circulant (1024)", numerical.solve_circulant, sl.solve_circulant, circulant, r.standard_normal(1024)),
        case("solve_triangular (256, 8)", numerical.solve_triangular, sl.solve_triangular,
             np.triu(r.standard_normal((256, 256))) + 256 * np.eye(256), r.standard_normal((256, 8))),
        case("eigh(a, b) values (64)", lambda a, b: numerical.eigh(a, b=b)[0], lambda a, b: sl.eigh(a, b)[0], sym(64), spd(64)),
    ]


def run(cases, reference):
    """Each case checked against `reference` then timed; the lines, the verdict, and the cases whose samples are
    drawn. Returns whether every case matched."""
    header = f"{'case':<26} {reference:>11} {'NumericAL':>11} {'ratio':>10}"
    print(header)
    print("-" * len(header))
    compared = []
    matched = 0
    boxplots = []
    for name, numerical_fn, reference_fn, verify in cases:
        case = compare(name, numerical_fn, reference_fn, verify)
        compared.append(case)
        if case.extra["mismatch"]:
            print(f"{name:<26} RESULT MISMATCH — correctness failure")
            continue
        matched += 1
        sc_samples = case.samples
        ref_samples = case.extra["reference_samples"]
        t_sc = median_iqr(sc_samples)[0]
        t_ref = median_iqr(ref_samples)[0]
        print(f"{name:<26} {fmt(t_ref, 8)} {fmt(t_sc, 8)} {t_ref / t_sc:9.4f}x")
        if name in BOXPLOT_CASES:
            boxplots.append((name, ref_samples, sc_samples))
    print("-" * len(header))
    print(f"all {matched} timed cases matched {reference} within rtol={RTOL}, atol={ATOL}")
    result = verdict(compared, subject_label="NumericAL")
    if result.classification != "no-data":
        print(f"geometric-mean ratio ({reference}/NumericAL): {result.text}")
    return result.passed, boxplots


def main():
    print(f"NumericAL {numerical.__version__} vs NumPy {np.__version__} "
          f"(Python {sys.version.split()[0]})\n")
    # The CI-aware aggregate (NumPy/NumericAL): faster, slower, equivalent or indecisive, as its confidence interval
    # says; only a result mismatch fails.
    passed, boxplots = run(build_cases(), "NumPy")
    if boxplots:
        print("\nper-call time distributions — NumPy vs NumericAL (left = faster):")
        for name, np_samples, sc_samples in boxplots:
            print(f"  {name}:")
            print(ascii_boxplot([np_samples, sc_samples], ["NumPy", "NumericAL"], width=42))

    # scipy.linalg's functions, against SciPy where it is installed: a section and a verdict of their own, the NumPy
    # aggregate above unchanged.
    try:
        import scipy
        import scipy.linalg as sl
    except ImportError:  # pragma: no cover - SciPy is optional
        print("\n(SciPy is not installed: scipy.linalg's functions are not compared)")
    else:
        print(f"\nscipy.linalg's functions against SciPy {scipy.__version__}\n")
        scipy_passed, _ = run(build_scipy_cases(sl), "SciPy")
        passed = passed and scipy_passed

    print("\nNumPy dispatches to its BLAS/LAPACK; NumericAL runs its own kernels, in C++, without one. "
          "Correctness within\ntolerance is the pass criterion; a small case's ratio moves between runs.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
