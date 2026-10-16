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


def scaled_norm(a):
    """The Frobenius norm, taken at the scale of the largest magnitude: a plain sum of squares overflows past 1e154."""
    if not a.size:
        return 0.0
    largest = float(np.max(np.abs(a)))
    if largest == 0.0 or not np.isfinite(largest):
        return largest
    return largest * float(np.linalg.norm(a / largest))


class Failure(Exception):
    pass


def check(condition, what):
    if not condition:
        raise Failure(what)


def bound(n, scale):
    return SLACK * max(n, 1) * EPS * scale


def bit_gap(got, want):
    """How two arrays meant to agree to the bit differ: how many entries, and by how much at most."""
    if got.shape != want.shape:
        return f"shapes {got.shape} and {want.shape}"
    differ = got != want
    largest = float(np.max(np.abs(got - want), initial=0.0)) if np.isfinite(got).all() and np.isfinite(want).all() else np.nan
    return f"{int(np.count_nonzero(differ))} of {got.size} entries differ, by up to {largest:.3g}"


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
    alone = as_np(numerical.eigvalsh(numerical.asarray(a)))
    check(np.max(np.abs(alone - want), initial=0.0) <= bound(n, size_a), "eigvalsh differs from NumPy's")
    # UPLO="U" reads the upper triangle: of a Hermitian matrix, the conjugate of the lower, so the same bits.
    upper_only = np.triu(a) + np.tril(np.full((n, n), np.nan), -1)
    wu, vu = (as_np(t) for t in numerical.eigh(numerical.asarray(upper_only), "U"))
    check(np.array_equal(wu, w) and np.array_equal(vu, v),
          f"eigh(UPLO='U') differs from the lower triangle's: {bit_gap(wu, w)}, vectors {bit_gap(vu, v)}")
    via_upper = as_np(numerical.eigvalsh(numerical.asarray(upper_only), "U"))
    check(np.array_equal(via_upper, alone),
          f"eigvalsh(UPLO='U') differs from the lower triangle's: {bit_gap(via_upper, alone)}; the lower triangle's again "
          f"{bit_gap(as_np(numerical.eigvalsh(numerical.asarray(a))), alone)}")


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
    # NumPy's default, full_matrices=True: U m×m and Vh n×n orthonormal, the thin factors their first columns.
    check(u.shape == (m, m) and vh.shape == (n, n), f"svd full shapes {u.shape} {vh.shape}")
    check(norm(u.conj().T @ u - np.eye(m)) <= bound(max(m, n), 1.0), "full U not orthonormal")
    check(norm(vh @ vh.conj().T - np.eye(n)) <= bound(max(m, n), 1.0), "full V not orthonormal")
    tu, ts, tvh = (as_np(x) for x in numerical.svd(numerical.asarray(a), full_matrices=False))
    check(np.array_equal(tu, u[:, :k]) and np.array_equal(ts, s) and np.array_equal(tvh, vh[:k]),
          "thin factors are not the full ones' first columns")
    check(np.max(np.abs(as_np(numerical.svd(numerical.asarray(a), compute_uv=False)) - want), initial=0.0)
          <= bound(max(m, n), size_a), "compute_uv=False values differ from NumPy's")


def run_qr(a):
    m, n = a.shape
    q, r = numerical.qr(numerical.asarray(a))
    q, r = as_np(q), as_np(r)
    k = min(m, n)
    check(norm(q @ r - a) <= bound(max(m, n), norm(a)), "QR does not reconstruct")
    check(norm(q.conj().T @ q - np.eye(k)) <= bound(max(m, n), 1.0), "Q not orthonormal")
    check(np.allclose(np.tril(r, -1), 0), "R not upper triangular")
    # complete: Q m×m orthonormal, R m×n with zero rows past k, the reduced Q its first columns
    qc, rc = (as_np(x) for x in numerical.qr(numerical.asarray(a), "complete"))
    check(qc.shape == (m, m) and rc.shape == (m, n), f"complete QR shapes {qc.shape} {rc.shape}")
    check(norm(qc @ rc - a) <= bound(max(m, n), norm(a)), "complete QR does not reconstruct")
    check(norm(qc.conj().T @ qc - np.eye(m)) <= bound(max(m, n), 1.0), "complete Q not orthonormal")
    check(np.array_equal(qc[:, :k], q) and not np.any(rc[k:]), "complete QR does not extend the reduced one")
    check(np.array_equal(as_np(numerical.qr(numerical.asarray(a), "r")), r), "mode='r' differs from the reduced R")


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
    # upper=True reads the upper triangle, as NumPy's: the factor of the matrix that triangle makes.
    upper_only = np.triu(spd) + np.tril(np.full((n, n), np.nan), -1)
    from_upper = np.triu(spd) + np.triu(spd, 1).conj().T
    upper = as_np(numerical.cholesky(numerical.asarray(upper_only), upper=True))
    check(np.allclose(np.tril(upper, -1), 0), "upper Cholesky factor not upper triangular")
    check(norm(upper.conj().T @ upper - from_upper) <= bound(n, norm(spd)), "upper Cholesky does not reconstruct")


RCONDS = (None, None, -1.0, 0.0, 1e-12, 1e-6, 1e-2)


def run_lstsq(a, rng):
    """Any shape, any rank, any rcond. Where NumPy's rank is plain (every singular value a thousand times past its
    tolerance, one way or the other), the same minimum-norm solution to within the conditioning of the kept values;
    between, where the two may draw the rank differently, a finite answer."""
    m, n = a.shape
    if a.size == 0:
        return
    b = rng.standard_normal(m) + (1j * rng.standard_normal(m) if np.iscomplexobj(a) else 0)
    rcond = RCONDS[int(rng.integers(len(RCONDS)))]
    x = as_np(numerical.lstsq(numerical.asarray(a), numerical.asarray(b), rcond)[0])
    want = np.linalg.lstsq(a, b, rcond=rcond)[0]
    if not np.isfinite(want).all():
        return  # NumPy's own answer overflowed: nothing to compare
    sigma = np.linalg.svd(a, compute_uv=False)
    if not np.isfinite(sigma).all() or sigma[0] == 0:
        check(np.allclose(x, want), "lstsq of a zero matrix differs from NumPy's")
        return
    tolerance = sigma[0] * relative_tolerance(rcond, m, n, lapack=True)
    if plain_rank(sigma, tolerance, sigma[0] * EPS * max(m, n)) is None:
        check(np.isfinite(x).all(), "lstsq near NumPy's rank tolerance is not finite")
        return
    kept = sigma[sigma > tolerance]
    condition = kept[0] / kept[-1]
    # A least-squares solution's forward error has a second term, κ² ‖r‖ / ‖A‖, from the residual r: with |x| small
    # beside |b| (a 2 × 1 system whose x cancels to 0.0027 of b), a backward-stable QR solve is off by ε ‖r‖ / ‖A‖,
    # far more than ε |x|, and NumPy's SVD by as much on other draws.
    residual = norm(b - (a @ want))
    allowed = 50 * max(m, n) * EPS * (condition * max(norm(want), 1e-300) + condition**2 * residual / kept[0])
    check(norm(x - want) <= allowed + bound(max(m, n), 1e-300),
          f"lstsq differs from NumPy's ({norm(x - want):.3g} for |x| {norm(want):.3g}, kept κ {condition:.3g})")


def relative_tolerance(rtol, m, n, lapack=False):
    """NumPy's cutoff relative to the largest singular value: rtol, or ε max(m, n) for None (for lstsq, ε for a
    negative one, as LAPACK's)."""
    if rtol is None:
        return EPS * max(m, n)
    return EPS if lapack and rtol < 0 else rtol


def plain_rank(sigma, cutoff, floor):
    """The rank at `cutoff` where no singular value lies within a factor of a thousand of it, else None. Under `floor`,
    the rounding of the decomposition (ε max(m, n) times the largest), a value may come out zero or not: a cutoff
    under a thousand times it leaves every value under that ambiguous."""
    if cutoff < floor * 1e3:
        ambiguous = sigma < floor * 1e3
    else:
        ambiguous = (sigma > cutoff / 1e3) & (sigma < cutoff * 1e3)
    if np.any(ambiguous):
        return None
    return int(np.count_nonzero(sigma > cutoff))


def run_pinv_and_rank(a, h, rng):
    """pinv and matrix_rank, general and Hermitian, at NumPy's tolerance or a drawn one: where the rank is plain, the
    same rank, and a pseudo-inverse that meets the Penrose conditions and agrees with NumPy's to the conditioning of the
    kept values. The Hermitian forms read the lower triangle alone."""
    rtol = RCONDS[int(rng.integers(len(RCONDS)))]
    rtol = None if rtol is not None and rtol < 0 else rtol
    for matrix, hermitian in ((a, False), (h, True)):
        m, n = matrix.shape
        if matrix.size == 0 or not np.isfinite(matrix).all():
            continue
        given = np.tril(matrix) + np.triu(np.full((n, n), np.nan), 1) if hermitian else matrix
        sigma = np.abs(np.linalg.eigvalsh(matrix)) if hermitian else np.linalg.svd(matrix, compute_uv=False)
        largest = float(np.max(sigma, initial=0.0))
        if not np.isfinite(largest):
            continue
        cutoff = largest * relative_tolerance(rtol, m, n)
        floor = largest * EPS * max(m, n)
        rank = numerical.matrix_rank(numerical.asarray(given), hermitian=hermitian, rtol=rtol)
        plain = plain_rank(sigma, cutoff, floor)
        if plain is not None:
            check(rank == plain, f"matrix_rank {rank}, NumPy's {plain} (hermitian={hermitian}, rtol={rtol})")
        tol = cutoff * float(rng.choice((0.5, 2.0)))  # an absolute tol, as matrix_rank's
        plain = plain_rank(sigma, tol, floor)
        if plain is not None:
            got = numerical.matrix_rank(numerical.asarray(given), tol, hermitian)
            check(got == plain, f"matrix_rank(tol) {got}, NumPy's {plain} (hermitian={hermitian})")
        p = as_np(numerical.pinv(numerical.asarray(given), rtol, hermitian))
        check(p.shape == (n, m), f"pinv shape {p.shape}")
        if plain_rank(sigma, cutoff, floor) is None or largest == 0:
            check(np.isfinite(p).all() or largest == 0, "pinv near the tolerance is not finite")
            continue
        kept = np.sort(sigma[sigma > cutoff])[::-1]
        if kept.size == 0:
            check(not np.any(p), "pinv of a rank-0 matrix is not zero")
            continue
        condition = kept[0] / kept[-1]
        want = np.linalg.pinv(matrix, rcond=rtol, hermitian=hermitian) if rtol is not None else \
            np.linalg.pinv(matrix, hermitian=hermitian)
        allowed = 50 * max(m, n) * EPS * condition / kept[-1]
        check(scaled_norm(p - want) <= allowed * max(1.0, np.sqrt(kept.size)),
              f"pinv differs from NumPy's ({scaled_norm(p - want):.3g}, allowed {allowed:.3g}, hermitian={hermitian})")


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
                  ("lstsq", lambda: run_lstsq(g, rng)), ("pinv/rank", lambda: run_pinv_and_rank(g, h, rng)))
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
