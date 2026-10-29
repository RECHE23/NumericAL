"""solve_sylvester and the Lyapunov solvers against scipy.linalg where it is installed, and against their equations."""

import unittest

import numerical

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import scipy.linalg as sl
except ImportError:  # pragma: no cover
    sl = None


def _drawn(shape, complex_entries, rng):
    x = rng.standard_normal(shape)
    return x + 1j * rng.standard_normal(shape) if complex_entries else x


@unittest.skipIf(np is None, "NumPy is not installed")
class Sylvester(unittest.TestCase):
    def test_the_equation_holds(self):
        # every mix of real and complex operands, square and not; single precision computed in double
        rng = np.random.default_rng(4)
        for m, n in ((1, 1), (3, 4), (20, 15)):
            for kinds in ((False, False, False), (True, True, True), (True, False, True), (False, True, False)):
                a, b, q = (_drawn(shape, kind, rng) for shape, kind in zip(((m, m), (n, n), (m, n)), kinds))
                with self.subTest(m=m, n=n, kinds=kinds):
                    x = np.asarray(numerical.solve_sylvester(a, b, q))
                    self.assertEqual(x.dtype, np.complex128 if any(kinds) else np.float64)
                    scale = np.abs(a).max() * np.abs(x).max() + np.abs(x).max() * np.abs(b).max() + np.abs(q).max()
                    self.assertLess(np.abs(a @ x + x @ b - q).max(), 1e-13 * scale)
        single = numerical.solve_sylvester(np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32), np.ones((2, 2), np.float32))
        self.assertEqual(single.dtype, "float32")
        self.assertEqual(np.asarray(single).tolist(), [[0.5, 0.5], [0.5, 0.5]])
        for bad in ((np.ones((2, 3)), np.eye(2), np.ones((2, 2))), (np.eye(2), np.eye(3), np.ones((2, 2))),
                    (np.eye(2), np.eye(2), np.full((2, 2), np.nan))):
            with self.assertRaises(numerical.LinAlgError):
                numerical.solve_sylvester(*bad)

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        # operands of one kind (SciPy pairs a real b's quasi-triangular form with the complex solver where a is complex)
        rng = np.random.default_rng(5)
        for m, n in ((3, 4), (20, 15)):
            for complex_entries in (False, True):
                a, b, q = (_drawn(shape, complex_entries, rng) for shape in ((m, m), (n, n), (m, n)))
                ours = np.asarray(numerical.solve_sylvester(a, b, q))
                np.testing.assert_allclose(ours, sl.solve_sylvester(a, b, q), rtol=1e-11, atol=1e-11 * np.abs(ours).max())

    def test_lyapunov(self):
        rng = np.random.default_rng(6)
        for n in (3, 12):
            for complex_entries in (False, True):
                a = _drawn((n, n), complex_entries, rng) * 0.3
                q = _drawn((n, n), complex_entries, rng)
                q = q + q.conj().T
                with self.subTest(n=n, complex_entries=complex_entries):
                    x = np.asarray(numerical.solve_continuous_lyapunov(a, q))
                    self.assertLess(np.abs(a @ x + x @ a.conj().T - q).max(), 1e-12 * np.abs(x).max())
                    self.assertIs(numerical.solve_lyapunov, numerical.solve_continuous_lyapunov)
                    for method in (None, "direct", "bilinear"):
                        x = np.asarray(numerical.solve_discrete_lyapunov(a, q, method))
                        self.assertLess(np.abs(a @ x @ a.conj().T - x + q).max(), 1e-12 * np.abs(x).max())
                        if sl is not None:
                            np.testing.assert_allclose(x, sl.solve_discrete_lyapunov(a, q, method), rtol=1e-11,
                                                       atol=1e-11 * np.abs(x).max())
        with self.assertRaises(ValueError):
            numerical.solve_discrete_lyapunov(np.eye(2), np.eye(2), "other")
        for bad in ((np.ones((2, 3)), np.eye(2)), (np.eye(2), np.eye(3)), (np.ones(2), np.eye(2))):
            with self.assertRaises(numerical.LinAlgError):
                numerical.solve_discrete_lyapunov(*bad)
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_continuous_lyapunov(np.ones(3), np.eye(3))


if __name__ == "__main__":
    unittest.main()
