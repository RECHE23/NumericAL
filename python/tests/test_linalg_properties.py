"""Every decomposition against its defining properties and against NumPy, over the matrix families that break
weak algorithms: Hilbert, Vandermonde, graded singular values, rank-deficient, exactly singular, complex, and
every shape. Deterministic (fixed seeds); skipped when NumPy is not importable (CI installs it).

The tolerances are backward-error bounds, a small multiple of the machine epsilon times the order: what a
stable algorithm guarantees whatever the conditioning. Values are compared to NumPy only where they are well
defined (a numerically singular matrix has a determinant of rounding noise in both libraries).
"""
import itertools
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None

EPS = 2.220446049250313e-16


def family(rng, n, kind, cplx=False):
    if kind == "normal":
        a = rng.standard_normal((n, n))
        return a + 1j * rng.standard_normal((n, n)) if cplx else a
    if kind == "hilbert":
        i = np.arange(n)
        return 1.0 / (i[:, None] + i[None, :] + 1)
    if kind == "vander":
        return np.vander(np.linspace(0, 1, n), n, increasing=True)
    if kind == "graded":
        q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
        q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
        return q1 @ np.diag(np.logspace(0, -12, n)) @ q2.T
    if kind == "rankdef":
        k = max(n // 2, 1)
        return rng.standard_normal((n, k)) @ rng.standard_normal((k, n))
    raise ValueError(kind)


@unittest.skipIf(np is None, "NumPy not installed")
class SquareProperties(unittest.TestCase):
    SIZES = (1, 2, 3, 5, 16, 17, 40)
    KINDS = ("normal", "hilbert", "vander", "graded", "rankdef")

    def setUp(self):
        self.rng = np.random.default_rng(20261011)

    def cases(self):
        for n, kind in itertools.product(self.SIZES, self.KINDS):
            yield f"{kind} {n}", n, family(self.rng, n, kind)
        for n in self.SIZES:
            yield f"complex {n}", n, family(self.rng, n, "normal", cplx=True)

    def test_solve_and_inverse_are_backward_stable(self):
        for name, n, a in self.cases():
            if np.linalg.cond(a) > 1e12:
                continue
            with self.subTest(name):
                b = self.rng.standard_normal(n)
                x = np.asarray(numerical.solve(a, b))
                self.assertLess(np.linalg.norm(a @ x - b) / (np.linalg.norm(a) * np.linalg.norm(x) + np.linalg.norm(b)),
                                1e3 * EPS * n)
                inv = np.asarray(numerical.inverse(a))
                self.assertLess(np.linalg.norm(a @ inv - np.eye(n)) / max(1.0, np.linalg.norm(a) * np.linalg.norm(inv)),
                                1e3 * EPS * n)

    def test_determinant_and_slogdet_agree_with_numpy(self):
        for name, n, a in self.cases():
            if np.linalg.cond(a) > 1e10:
                continue  # both libraries return rounding noise for a numerically singular matrix
            with self.subTest(name):
                want = np.linalg.det(a)
                self.assertTrue(np.isclose(numerical.determinant(a), want, rtol=1e-8, atol=0.0))
                sign, logabsdet = numerical.slogdet(a)
                want_sign, want_log = np.linalg.slogdet(a)
                self.assertTrue(np.isclose(sign, want_sign))
                self.assertTrue(np.isclose(logabsdet, want_log, rtol=1e-10, atol=1e-10))

    def test_eigh_is_orthonormal_and_reconstructs(self):
        for (name, n, a), method in itertools.product(self.cases(), ("divide_and_conquer", "tridiagonal_ql", "jacobi")):
            h = (a + a.conj().T) / 2
            with self.subTest(name, method=method):
                if method == "jacobi" and np.iscomplexobj(h):
                    with self.assertRaises(numerical.LinAlgError):
                        numerical.eigh(h, method=method)
                    continue
                values, vectors = (np.asarray(t) for t in numerical.eigh(h, method=method))
                scale = max(1.0, np.max(np.abs(np.linalg.eigvalsh(h))))
                self.assertLess(np.max(np.abs(values - np.linalg.eigvalsh(h))), 1e3 * EPS * n * scale)
                self.assertLess(np.linalg.norm(vectors.conj().T @ vectors - np.eye(n)), 1e3 * EPS * n)
                self.assertLess(np.linalg.norm(h @ vectors - vectors * values) / max(1.0, np.linalg.norm(h)),
                                1e3 * EPS * n)
                self.assertTrue(np.all(np.diff(values) >= 0))

    def test_cholesky_reconstructs_a_positive_definite_matrix(self):
        for name, n, a in self.cases():
            spd = a @ a.conj().T + n * np.eye(n)
            with self.subTest(name):
                lower = np.asarray(numerical.cholesky(spd))
                self.assertEqual(np.linalg.norm(np.triu(lower, 1)), 0.0)
                self.assertLess(np.linalg.norm(lower @ lower.conj().T - spd) / np.linalg.norm(spd), 1e2 * EPS * n)


class NumPyKeywords(unittest.TestCase):
    """UPLO, upper, rcond, rtol, tol and hermitian mean what they mean to numpy.linalg."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.real = rng.standard_normal((6, 6))
        self.complex = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
        self.diagonal = np.diag([1.0, 1e-3, 1e-9, 0.0])

    def test_uplo_and_upper_read_the_triangle_numpy_reads(self):
        for a in (self.real, self.complex):
            n = a.shape[0]
            spd = a @ a.conj().T + n * np.eye(n)
            noisy = np.triu(spd) + np.tril(np.full((n, n), 99.0), -1)  # only the upper triangle is meaningful
            for UPLO in ("U", "u"):
                values, vectors = (np.asarray(t) for t in numerical.eigh(noisy, UPLO))
                self.assertTrue(np.allclose(values, np.linalg.eigh(noisy, "U")[0]))
                self.assertTrue(np.allclose(np.asarray(numerical.eigvalsh(noisy, UPLO)), np.linalg.eigvalsh(noisy, "U")))
                hermitian = np.triu(noisy) + np.triu(noisy, 1).conj().T
                self.assertLess(np.linalg.norm(hermitian @ vectors - vectors * values), 1e3 * EPS * n * np.linalg.norm(spd))
            upper = np.asarray(numerical.cholesky(noisy, upper=True))
            self.assertTrue(np.allclose(upper, np.linalg.cholesky(noisy, upper=True)))
            self.assertEqual(np.linalg.norm(np.tril(upper, -1)), 0.0)
            self.assertTrue(np.array_equal(np.asarray(numerical.cholesky(spd, upper=False)), np.asarray(numerical.cholesky(spd))))
        with self.assertRaises(ValueError):
            numerical.eigh(self.real, "X")
        with self.assertRaises(ValueError):
            numerical.eigvalsh(self.real, 1)

    def test_tolerances_cut_the_singular_values_numpy_cuts(self):
        d = self.diagonal
        for kw in ({}, {"rcond": 1e-6}, {"rcond": -1}, {"rcond": 0.0}):
            x, _, rank, _ = numerical.lstsq(d, np.ones(4), **kw)
            want = np.linalg.lstsq(d, np.ones(4), **kw)
            self.assertTrue(np.allclose(np.asarray(x), want[0]))
            self.assertEqual(rank, want[2])
        for kw in ({}, {"rcond": 1e-6}, {"rtol": 1e-2}, {"hermitian": True}, {"rcond": 1e-6, "hermitian": True}):
            self.assertTrue(np.allclose(np.asarray(numerical.pinv(d, **kw)), np.linalg.pinv(d, **kw)))
        for kw in ({}, {"tol": 1e-6}, {"rtol": 1e-2}, {"hermitian": True}, {"tol": 0.5, "hermitian": True}):
            self.assertEqual(numerical.matrix_rank(d, **kw), np.linalg.matrix_rank(d, **kw))
        for v in ([0.0, 0.0, 3.0], [0.0, 0.0], 5.0, 0.0, np.zeros(0), [1j]):
            self.assertEqual(numerical.matrix_rank(v), np.linalg.matrix_rank(np.asarray(v)))

    def test_a_hermitian_matrix_is_read_from_its_lower_triangle(self):
        for a in (self.real, self.complex):
            n = a.shape[0]
            h = (a + a.conj().T) / 2
            noisy = np.tril(h) + np.triu(np.full((n, n), np.nan), 1)
            self.assertTrue(np.allclose(np.asarray(numerical.pinv(noisy, hermitian=True)), np.linalg.pinv(h)))
            self.assertEqual(numerical.matrix_rank(noisy, hermitian=True), np.linalg.matrix_rank(h))
        with self.assertRaises(numerical.LinAlgError):
            numerical.pinv(np.ones((2, 3)), hermitian=True)

    def test_tolerances_are_numbers_set_once_and_not_negative(self):
        d = self.diagonal
        with self.assertRaises(ValueError):
            numerical.pinv(d, 1e-3, rtol=1e-3)
        with self.assertRaises(ValueError):
            numerical.matrix_rank(d, tol=1.0, rtol=1.0)
        for bad in (-1.0, float("nan")):
            with self.assertRaises(ValueError):
                numerical.pinv(d, bad)
            with self.assertRaises(ValueError):
                numerical.matrix_rank(d, bad)
        with self.assertRaises(ValueError):
            numerical.lstsq(d, np.ones(4), float("nan"))
        for bad in ("1e-3", True, 1j):
            with self.assertRaises(TypeError):
                numerical.pinv(d, bad)

    def test_det_and_inv_are_numpys_names(self):
        self.assertIs(numerical.det, numerical.determinant)
        self.assertIs(numerical.inv, numerical.inverse)


class MethodNames(unittest.TestCase):
    def test_an_unknown_method_is_a_value_error(self):
        with self.assertRaises(ValueError):
            numerical.eigh([[1.0]], "qr")
        with self.assertRaises(ValueError):
            numerical.svd([[1.0]], method="tridiagonal_ql")
        with self.assertRaises(TypeError):  # the second argument is full_matrices, as in NumPy
            numerical.svd([[1.0]], "jacobi")
        with self.assertRaises(TypeError):  # the second argument is UPLO, as in NumPy
            numerical.eigh([[1.0]], "jacobi")

    def test_the_extension_refuses_an_unknown_method_too(self):
        one = numerical.asarray([[1.0]])
        with self.assertRaises(numerical.error):
            numerical._numerical.eigh(one, "qr", False)
        with self.assertRaises(numerical.error):
            numerical._numerical.svd(one, "qr", False)


@unittest.skipIf(np is None, "NumPy not installed")
class RectangularProperties(unittest.TestCase):
    SHAPES = ((1, 1), (3, 1), (1, 3), (5, 3), (3, 5), (17, 16), (16, 17), (40, 8), (8, 40))

    def setUp(self):
        self.rng = np.random.default_rng(20261012)

    def cases(self):
        for (m, n), cplx in itertools.product(self.SHAPES, (False, True)):
            a = self.rng.standard_normal((m, n))
            yield f"{m}x{n} {'c' if cplx else 'r'}", m, n, (a + 1j * self.rng.standard_normal((m, n)) if cplx else a)
        for n in (5, 16, 40):
            yield f"rankdef {n}", n, n, family(self.rng, n, "rankdef")
            yield f"graded {n}", n, n, family(self.rng, n, "graded")

    def test_svd_is_orthonormal_reconstructs_and_matches_numpy(self):
        methods = ("divide_and_conquer", "bidiagonal_qr", "jacobi")
        for (name, m, n, a), method, full in itertools.product(self.cases(), methods, (True, False)):
            k = min(m, n)
            with self.subTest(name, method=method, full=full):
                u, s, vh = (np.asarray(t) for t in numerical.svd(a, full, method=method))
                rows, cols = (m, n) if full else (k, k)
                self.assertEqual((u.shape, s.shape, vh.shape), ((m, rows), (k,), (cols, n)))
                want = np.linalg.svd(a, compute_uv=False)
                self.assertLess(np.max(np.abs(s - want)), 1e2 * EPS * max(m, n) * want[0])
                self.assertLess(np.linalg.norm((u[:, :k] * s) @ vh[:k] - a) / np.linalg.norm(a), 1e2 * EPS * max(m, n))
                self.assertLess(np.linalg.norm(u.conj().T @ u - np.eye(rows)), 1e2 * EPS * max(m, n))
                self.assertLess(np.linalg.norm(vh @ vh.conj().T - np.eye(cols)), 1e2 * EPS * max(m, n))

    def test_qr_is_orthonormal_and_reconstructs_any_shape(self):
        for name, m, n, a in self.cases():
            k = min(m, n)
            with self.subTest(name):
                q, r = (np.asarray(t) for t in numerical.qr(a))
                self.assertEqual((q.shape, r.shape), ((m, k), (k, n)))
                self.assertEqual(np.linalg.norm(np.tril(r, -1)), 0.0)
                self.assertLess(np.linalg.norm(q @ r - a) / np.linalg.norm(a), 1e2 * EPS * max(m, n))
                self.assertLess(np.linalg.norm(q.conj().T @ q - np.eye(k)), 1e2 * EPS * max(m, n))

    def test_lstsq_pinv_and_rank_match_numpy(self):
        for name, m, n, a in self.cases():
            if name.startswith("graded"):
                continue  # σ_min = 1e-12: both libraries' answers carry their own 1e-5 relative error in it
            with self.subTest(name):
                b = self.rng.standard_normal(m)
                want, want_residuals, want_rank, want_s = np.linalg.lstsq(a, b, rcond=None)
                x, residuals, rank, s = numerical.lstsq(a, b)  # the four NumPy returns, unpacked as from it
                got = np.asarray(x)
                self.assertLess(np.linalg.norm(got - want), 1e-8 * max(1.0, np.linalg.norm(want)) * np.linalg.cond(a))
                self.assertEqual(rank, want_rank)
                self.assertEqual(np.asarray(residuals).shape, want_residuals.shape)
                self.assertTrue(np.allclose(np.asarray(residuals), want_residuals, rtol=1e-8, atol=1e-10))
                self.assertTrue(np.allclose(np.asarray(s), want_s, rtol=0, atol=1e-12 * want_s[0]))
                self.assertTrue(np.allclose(np.asarray(numerical.pinv(a)), np.linalg.pinv(a), atol=1e-8))
                self.assertEqual(numerical.matrix_rank(a), np.linalg.matrix_rank(a))


@unittest.skipIf(np is None, "NumPy not installed")
class NonFinitePolicy(unittest.TestCase):
    def test_every_decomposition_refuses_a_nan_or_infinite_entry(self):
        functions = (numerical.determinant, numerical.slogdet, numerical.inverse, numerical.cholesky,
                     numerical.eigh, numerical.svd, numerical.qr, numerical.pinv, numerical.matrix_rank,
                     lambda a: numerical.solve(a, np.ones(2)), lambda a: numerical.lstsq(a, np.ones(2)))
        for bad, function in itertools.product((np.nan, np.inf, -np.inf), functions):
            with self.subTest(bad=bad, function=function):
                with self.assertRaises(numerical.LinAlgError):
                    function(np.array([[4.0, bad], [bad, 3.0]]))


if __name__ == "__main__":
    unittest.main()
