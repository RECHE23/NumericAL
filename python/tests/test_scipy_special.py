"""scipy.linalg's special matrices, structure tests, subspaces, pinvh and the trigonometric and hyperbolic matrix
functions against SciPy's own. Deterministic; skipped when SciPy is not importable (CI installs it)."""
import itertools
import math
import unittest

import numerical

try:
    import numpy as np
    import scipy.linalg as sl
except ImportError:
    np = sl = None


@unittest.skipIf(sl is None, "SciPy not installed")
class SpecialMatrices(unittest.TestCase):
    def same(self, got, expected, exact=True):
        expected = np.asarray(expected)
        self.assertIsInstance(got, numerical.Tensor)
        self.assertEqual(got.shape, expected.shape)
        if exact:
            self.assertTrue(np.array_equal(np.asarray(got), expected.astype(np.asarray(got).dtype)))
        else:
            self.assertTrue(np.allclose(np.asarray(got), expected, rtol=1e-14, atol=0))

    def test_each_matrix_as_scipy(self):
        for n in (1, 2, 5, 9):
            with self.subTest(n=n):
                self.same(numerical.hilbert(n), sl.hilbert(n))
                self.same(numerical.invhilbert(n), sl.invhilbert(n, exact=True))
                for kind in ("symmetric", "lower", "upper"):
                    self.same(numerical.pascal(n, kind), sl.pascal(n, kind))
                    self.same(numerical.invpascal(n, kind), sl.invpascal(n, kind))
                self.same(numerical.helmert(n), sl.helmert(n), exact=False)
                self.same(numerical.helmert(n, full=True), sl.helmert(n, full=True), exact=False)
                self.same(numerical.dft(n), sl.dft(n))
                for scale in ("sqrtn", "n"):
                    self.same(numerical.dft(n, scale), sl.dft(n, scale))
        for n in (1, 2, 8, 64):
            self.same(numerical.hadamard(n), sl.hadamard(n))
            self.same(numerical.hadamard(n, complex), sl.hadamard(n, complex))
        with self.assertRaises(ValueError):
            numerical.hadamard(6)
        self.same(numerical.invhilbert(16), sl.invhilbert(16, exact=True).astype(float))  # past int64, rounded once

    def test_polynomials_and_vectors_as_scipy(self):
        rng = np.random.default_rng(20261104)
        for n in (2, 3, 4, 7):
            a = rng.standard_normal(n)
            with self.subTest(n=n):
                self.same(numerical.companion(a), sl.companion(a))
                self.same(numerical.fiedler_companion(a), sl.fiedler_companion(a))
                self.same(numerical.fiedler(a), sl.fiedler(a))
                f, s = rng.uniform(0, 2, n), rng.uniform(0, 1, n - 1)
                self.same(numerical.leslie(f, s), sl.leslie(f, s))
                for m, mode in itertools.product((1, 2, 5, 9), ("full", "same", "valid")):
                    self.same(numerical.convolution_matrix(a, m, mode), sl.convolution_matrix(a, m, mode))
                b, c = rng.standard_normal((n, 3)), rng.standard_normal((4, 3))
                self.same(numerical.khatri_rao(b, c), sl.khatri_rao(b, c))
                s = rng.uniform(0, 1, n)
                for M, N in ((n, n + 2), (n + 3, n), (n, n)):
                    self.same(numerical.diagsvd(s, M, N), sl.diagsvd(s, M, N))
        self.same(numerical.companion([1j, 2.0, 3.0]), sl.companion([1j, 2.0, 3.0]))
        self.same(numerical.fiedler([[1.0, 4.0], [2.0, -1.0]]), sl.fiedler(np.array([[1.0, 4.0], [2.0, -1.0]])))
        for call in (lambda: numerical.companion([0.0, 1.0]), lambda: numerical.companion([1.0]),
                     lambda: numerical.leslie([1.0, 2.0], [1.0, 2.0]), lambda: numerical.diagsvd([1.0], 2, 3),
                     lambda: numerical.convolution_matrix([1.0], 0), lambda: numerical.dft(3, "x")):
            with self.assertRaises(ValueError):
                call()


@unittest.skipIf(sl is None, "SciPy not installed")
class Structure(unittest.TestCase):
    def test_bandwidth_and_symmetry_as_scipy(self):
        rng = np.random.default_rng(20261105)
        for rows, cols, lower, upper in [(5, 5, 1, 2), (4, 6, 0, 3), (6, 3, 2, 0), (3, 3, 0, 0), (0, 0, 0, 0)]:
            a = np.triu(np.tril(rng.standard_normal((rows, cols)), upper), -lower)
            with self.subTest(shape=(rows, cols), band=(lower, upper)):
                self.assertEqual(numerical.bandwidth(a), sl.bandwidth(a))
        s = rng.standard_normal((4, 4))
        s = s + s.T
        h = s + 1j * (np.triu(s, 1) - np.triu(s, 1).T)
        for a in (s, h, s + 1e-9 * np.eye(4, k=1), h.conj()):
            with self.subTest(a=a[0, 1]):
                self.assertEqual(numerical.issymmetric(a), sl.issymmetric(a))
                self.assertEqual(numerical.ishermitian(a), sl.ishermitian(a))
                self.assertEqual(numerical.issymmetric(a, atol=1e-8), sl.issymmetric(a, atol=1e-8))
                self.assertEqual(numerical.ishermitian(a, rtol=1e-6), sl.ishermitian(a, rtol=1e-6))


@unittest.skipIf(sl is None, "SciPy not installed")
class Subspaces(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261106)

    def projector(self, q):
        q = np.asarray(q)
        return q @ q.conj().T

    def test_null_space_and_orth_span_scipy_s(self):
        for m, n, rank, cplx in [(5, 7, 3, False), (7, 5, 5, False), (6, 6, 2, True), (4, 4, 4, False), (3, 5, 0, False)]:
            a = self.rng.standard_normal((m, rank)) @ self.rng.standard_normal((rank, n))
            if cplx:
                a = a + 1j * (self.rng.standard_normal((m, rank)) @ self.rng.standard_normal((rank, n)))
            with self.subTest(m=m, n=n, rank=rank, cplx=cplx):
                ns, expected = numerical.null_space(a), sl.null_space(a)
                self.assertEqual(ns.shape, expected.shape)
                self.assertTrue(np.allclose(self.projector(ns), self.projector(expected), atol=1e-12))
                o, expected = numerical.orth(a), sl.orth(a)
                self.assertEqual(o.shape, expected.shape)
                self.assertTrue(np.allclose(self.projector(o), self.projector(expected), atol=1e-12))

    def test_subspace_angles_as_scipy(self):
        for m, p, q in [(6, 2, 3), (6, 3, 2), (8, 4, 4), (5, 1, 1)]:
            a, b = self.rng.standard_normal((m, p)), self.rng.standard_normal((m, q))
            with self.subTest(m=m, p=p, q=q):
                self.assertTrue(np.allclose(np.asarray(numerical.subspace_angles(a, b)), sl.subspace_angles(a, b),
                                            atol=1e-13))
        a = self.rng.standard_normal((5, 2))
        self.assertTrue(np.allclose(np.asarray(numerical.subspace_angles(a, a @ [[1.0, 2.0], [0.0, 1.0]])), 0.0,
                                    atol=1e-14))  # one space: small angles by their sines
        # the angles largest first and their sines aligned: a space sharing two directions and a small angle (SciPy
        # 1.18 sends the small ones through the arccosine: 0 for 1e-8, its shared directions' angles at 2e-8)
        q, _ = np.linalg.qr(self.rng.standard_normal((6, 4)))
        for theta in (1e-8, 1e-6):
            third = np.cos(theta) * q[:, 2] + np.sin(theta) * q[:, 3]
            with self.subTest(theta=theta):
                angles = np.asarray(numerical.subspace_angles(q[:, :3], np.column_stack([q[:, 0], q[:, 1], third])))
                self.assertLess(abs(angles[0] - theta), 1e-6 * theta)
                self.assertTrue(np.all(np.abs(angles[1:]) <= 1e-15))

    def test_procrustes_and_pinvh_as_scipy(self):
        for n, cplx in [(4, False), (5, True)]:
            a = self.rng.standard_normal((n + 2, n)) + (1j * self.rng.standard_normal((n + 2, n)) if cplx else 0)
            b = self.rng.standard_normal((n + 2, n))
            r, scale = numerical.orthogonal_procrustes(a, b)
            r_expected, scale_expected = sl.orthogonal_procrustes(a, b)
            with self.subTest(n=n, cplx=cplx):
                self.assertTrue(np.allclose(np.asarray(r), r_expected, atol=1e-12))
                self.assertAlmostEqual(scale, scale_expected, delta=1e-12 * abs(scale_expected))
                h = a.conj().T @ a
                h[0] *= 0.0
                h[:, 0] *= 0.0  # rank deficient
                got, rank = numerical.pinvh(h, return_rank=True)
                expected, expected_rank = sl.pinvh(h, return_rank=True)
                self.assertEqual(rank, expected_rank)
                self.assertTrue(np.allclose(np.asarray(got), expected, atol=1e-12))
                self.assertTrue(np.allclose(np.asarray(numerical.pinvh(h, lower=False, atol=1.0)),
                                            sl.pinvh(h, lower=False, atol=1.0), atol=1e-12))


@unittest.skipIf(sl is None, "SciPy not installed")
class MatrixFunctions(unittest.TestCase):
    def test_trigonometric_and_hyperbolic_as_scipy(self):
        rng = np.random.default_rng(20261107)
        for n, dtype in itertools.product((1, 3, 8), (np.float64, np.complex128, np.float32)):
            a = rng.standard_normal((n, n)) / n
            if dtype is np.complex128:
                a = a + 1j * rng.standard_normal((n, n)) / n
            a = a.astype(dtype)
            for name in ("cosm", "sinm", "tanm", "coshm", "sinhm", "tanhm", "signm"):
                with self.subTest(n=n, dtype=dtype.__name__, name=name):
                    got = getattr(numerical, name)(a)
                    expected = getattr(sl, name)(a)
                    self.assertEqual(got.dtype, str(np.asarray(expected).dtype))
                    tolerance = 1e-4 if dtype is np.float32 else 1e-11
                    self.assertTrue(np.allclose(np.asarray(got), expected, rtol=tolerance, atol=tolerance))
        with self.assertRaises(ValueError):
            numerical.cosm(np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
