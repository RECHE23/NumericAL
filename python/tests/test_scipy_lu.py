"""scipy.linalg.lu against SciPy's own: P @ L @ U == a for square, wide and tall matrices (each of the three ways
lu_factors takes), real and complex, L unit lower trapezoidal and U upper; P, L and U SciPy's for a real matrix (a
complex one's pivots by modulus, where LAPACK's izamax takes |re| + |im|); permute_l, p_indices, single precision,
empty matrices' shapes as SciPy's, a singular matrix, the refusals. Deterministic; skipped when SciPy is not
importable (CI installs it)."""
import itertools
import unittest

import numerical

try:
    import numpy as np
    import scipy.linalg as sl
except ImportError:
    np = sl = None


@unittest.skipIf(sl is None, "SciPy not installed")
class SciPyLu(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261031)

    def test_factors_as_scipy(self):
        for (m, n), complex_ in itertools.product([(1, 1), (4, 4), (3, 7), (7, 3), (30, 5), (5, 30), (40, 38)], (False, True)):
            a = self.rng.standard_normal((m, n)) + (1j * self.rng.standard_normal((m, n)) if complex_ else 0)
            with self.subTest(m=m, n=n, complex=complex_):
                p, l, u = (np.asarray(x) for x in numerical.lu(a))
                self.assertLess(np.abs(p @ l @ u - a).max(), 1e-13)
                self.assertTrue(np.array_equal(np.triu(l, 1), np.zeros_like(l)) and np.allclose(np.diag(l), 1.0))
                self.assertTrue(np.array_equal(np.tril(u, -1), np.zeros_like(u)))
                self.assertEqual((l.shape, u.shape, p.shape), ((m, min(m, n)), (min(m, n), n), (m, m)))
                if not complex_:
                    expected = sl.lu(a)
                    self.assertTrue(np.array_equal(p, expected[0]))
                    self.assertLess(np.abs(l - expected[1]).max(), 1e-12)
                    self.assertLess(np.abs(u - expected[2]).max(), 1e-12)
                pl, u2 = (np.asarray(x) for x in numerical.lu(a, permute_l=True))
                self.assertLess(np.abs(pl @ u2 - a).max(), 1e-13)
                rows, l3, u3 = numerical.lu(a, p_indices=True)
                self.assertLess(np.abs(np.asarray(l3)[rows] @ np.asarray(u3) - a).max(), 1e-13)

    def test_edges(self):
        singular = np.array([[1.0, 2.0], [2.0, 4.0]])
        p, l, u = (np.asarray(x) for x in numerical.lu(singular))
        self.assertLess(np.abs(p @ l @ u - singular).max(), 1e-15)
        self.assertEqual(u[1, 1], 0.0)
        for shape in ((3, 0), (0, 3)):
            got, expected = numerical.lu(np.zeros(shape)), sl.lu(np.zeros(shape))
            self.assertEqual([np.asarray(x).shape for x in got], [x.shape for x in expected])
        p, l, u = numerical.lu(np.eye(3, dtype=np.float32))
        self.assertEqual((p.dtype, l.dtype, u.dtype), ("float32", "float32", "float32"))
        for bad in (np.ones(3), np.array([[np.nan, 1.0], [1.0, 1.0]])):
            with self.assertRaises(numerical.LinAlgError):
                numerical.lu(bad)


if __name__ == "__main__":
    unittest.main()
