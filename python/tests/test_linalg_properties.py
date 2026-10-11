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
        for name, n, a in self.cases():
            h = (a + a.conj().T) / 2
            with self.subTest(name):
                values, vectors = (np.asarray(t) for t in numerical.eigh(h))
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
        for name, m, n, a in self.cases():
            k = min(m, n)
            with self.subTest(name):
                u, s, vh = (np.asarray(t) for t in numerical.svd(a))
                self.assertEqual((u.shape, s.shape, vh.shape), ((m, k), (k,), (k, n)))
                want = np.linalg.svd(a, compute_uv=False)
                self.assertLess(np.max(np.abs(s - want)), 1e2 * EPS * max(m, n) * want[0])
                self.assertLess(np.linalg.norm((u * s) @ vh - a) / np.linalg.norm(a), 1e2 * EPS * max(m, n))
                self.assertLess(np.linalg.norm(u.conj().T @ u - np.eye(k)), 1e2 * EPS * max(m, n))
                self.assertLess(np.linalg.norm(vh @ vh.conj().T - np.eye(k)), 1e2 * EPS * max(m, n))

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
                want = np.linalg.lstsq(a, b, rcond=None)[0]
                got = np.asarray(numerical.lstsq(a, b))
                self.assertLess(np.linalg.norm(got - want), 1e-8 * max(1.0, np.linalg.norm(want)) * np.linalg.cond(a))
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
