"""scipy.linalg.eig's arguments against SciPy's own: the left eigenvectors (``vl.conj().T @ a == w * vl.conj().T``),
the generalized problem ``a @ v = w * b @ v`` by the complex QZ (its values as SciPy's as sets, ∞ and NaN where SciPy
puts them, each vector of unit 2-norm, the left ones too), the homogeneous pairs, the answer's shape for each choice
of left and right, single precision, eigvals with b, the refusals. Deterministic; skipped when SciPy is not importable
(CI installs it)."""
import itertools
import unittest

import numerical

try:
    import numpy as np
    import scipy.linalg as sl
except ImportError:
    np = sl = None


def _matched(x, y, tolerance):
    """Whether the complex numbers `x` and `y` match one to one within `tolerance` of their size (a greedy match)."""
    x, y = list(x), list(y)
    for value in x:
        distances = [abs(value - other) / max(1.0, abs(other)) for other in y]
        k = int(np.argmin(distances))
        if distances[k] > tolerance:
            return False
        y.pop(k)
    return not y


@unittest.skipIf(sl is None, "SciPy not installed")
class SciPyEig(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261031)

    def matrix(self, n, complex_):
        a = self.rng.standard_normal((n, n))
        return a + 1j * self.rng.standard_normal((n, n)) if complex_ else a

    def test_left_eigenvectors(self):
        for n, complex_ in itertools.product((1, 4, 25), (False, True)):
            a = self.matrix(n, complex_)
            with self.subTest(n=n, complex=complex_):
                w, vl, vr = (np.asarray(x) for x in numerical.eig(a, left=True))
                self.assertLess(np.abs(vl.conj().T @ a - w[:, None] * vl.conj().T).max(), 1e-12)
                self.assertLess(np.abs(a @ vr - vr * w).max(), 1e-12)
                self.assertTrue(np.allclose(np.linalg.norm(vl, axis=0), 1.0))
                self.assertTrue(_matched(w, sl.eigvals(a), 1e-10))
                w2, vl2 = numerical.eig(a, left=True, right=False)
                self.assertTrue(np.array_equal(np.asarray(w2), w) and np.array_equal(np.asarray(vl2), vl))
                self.assertTrue(np.array_equal(np.asarray(numerical.eig(a, right=False)), np.asarray(numerical.eigvals(a))))

    def test_generalized_problem(self):
        for n, complex_ in itertools.product((1, 5, 30), (False, True)):
            a, b = self.matrix(n, complex_), self.matrix(n, complex_)
            with self.subTest(n=n, complex=complex_):
                w, vl, vr = (np.asarray(x) for x in numerical.eig(a, b, left=True))
                self.assertLess(np.abs(a @ vr - (b @ vr) * w).max(), 1e-11 * max(1.0, np.abs(w).max()))
                self.assertLess(np.abs(vl.conj().T @ a - w[:, None] * (vl.conj().T @ b)).max(), 1e-11 * max(1.0, np.abs(w).max()))
                self.assertTrue(np.allclose(np.linalg.norm(vr, axis=0), 1.0) and np.allclose(np.linalg.norm(vl, axis=0), 1.0))
                self.assertTrue(_matched(w, sl.eigvals(a, b), 1e-9))
                self.assertTrue(_matched(np.asarray(numerical.eigvals(a, b)), w, 1e-15))
                pairs = np.asarray(numerical.eigvals(a, b, homogeneous_eigvals=True))
                self.assertEqual(pairs.shape, (2, n))
        # an infinite eigenvalue where only β is 0, NaN where both are, as SciPy's
        for a, b in ((np.eye(3), np.diag([1.0, 0.0, 1.0])), (np.diag([1.0, 0.0, 2.0]), np.diag([1.0, 0.0, 1.0]))):
            got, expected = np.asarray(numerical.eigvals(a, b)), sl.eigvals(a, b)
            self.assertEqual(np.sort_complex(got).tolist().__repr__(), np.sort_complex(expected).tolist().__repr__())

    def test_shapes_precision_and_refusals(self):
        a = self.matrix(3, False)
        self.assertEqual(np.asarray(numerical.eig(a, homogeneous_eigvals=True)[0]).shape, (2, 3))
        self.assertEqual(numerical.eig(a.astype(np.float32), left=True)[1].dtype, "complex64")
        self.assertEqual(numerical.eig(a.astype(np.float32), a.astype(np.float32))[1].dtype, "complex64")
        with self.assertRaises(ValueError):
            numerical.eig(a, np.eye(2))
        with self.assertRaises(ValueError):
            numerical.eig(np.ones((2, 3)), np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
