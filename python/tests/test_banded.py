"""solve_banded and solveh_banded against scipy.linalg where it is installed, and against the dense solve
everywhere."""

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


def _band(n, lower, upper, complex_entries, rng):
    """A dense band matrix and its diagonal-ordered form."""
    a = np.zeros((n, n), complex if complex_entries else float)
    ab = np.zeros((lower + upper + 1, n), a.dtype)
    for i in range(n):
        for j in range(max(0, i - lower), min(n, i + upper + 1)):
            a[i, j] = rng.standard_normal() + (1j * rng.standard_normal() if complex_entries else 0)
            ab[upper + i - j, j] = a[i, j]
    return a, ab


@unittest.skipIf(np is None, "NumPy is not installed")
class Banded(unittest.TestCase):
    def test_solve_banded(self):
        rng = np.random.default_rng(1)
        for n, lower, upper in ((5, 1, 2), (30, 3, 1), (40, 1, 1), (12, 0, 2), (12, 2, 0), (1, 0, 0)):
            for complex_entries in (False, True):
                a, ab = _band(n, lower, upper, complex_entries, rng)
                for b in (rng.standard_normal(n), rng.standard_normal((n, 3))):
                    with self.subTest(n=n, lower=lower, upper=upper, complex_entries=complex_entries, rank=b.ndim):
                        x = np.asarray(numerical.solve_banded((lower, upper), ab, b))
                        self.assertEqual(x.shape, b.shape)
                        np.testing.assert_allclose(a @ x, b, atol=1e-10 * np.abs(x).max())
                        if sl is not None and not (n == 1 and complex_entries):  # SciPy's 1 × 1 path casts a complex quotient into b's float
                            np.testing.assert_allclose(x, sl.solve_banded((lower, upper), ab, b), rtol=1e-10, atol=1e-12)
        single = numerical.solve_banded((1, 1), np.array([[0, 1], [2, 2], [1, 0]], np.float32), np.ones(2, np.float32))
        self.assertEqual(single.dtype, "float32")
        with self.assertRaises(ValueError):
            numerical.solve_banded((1, 1), np.ones((2, 4)), np.ones(4))
        with self.assertRaises(ValueError):
            numerical.solve_banded((1, 1), np.ones((3, 4)), np.ones(5))
        with self.assertRaises(ValueError):
            numerical.solve_banded((-1, 1), np.ones((1, 4)), np.ones(4))
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_banded((1, 0), np.array([[1.0, 0.0], [1.0, 0.0]]), np.ones(2))
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_banded((0, 0), np.array([[1.0, float("nan")]]), np.ones(2))

    def test_solveh_banded(self):
        rng = np.random.default_rng(2)
        for n, kd in ((6, 2), (30, 3), (10, 1), (5, 0)):
            for complex_entries in (False, True):
                a = np.zeros((n, n), complex if complex_entries else float)
                for i in range(n):
                    for j in range(max(0, i - kd), i):
                        value = rng.standard_normal() + (1j * rng.standard_normal() if complex_entries else 0)
                        a[i, j], a[j, i] = value, np.conj(value)
                    a[i, i] = 4 * kd + 2
                upper = np.zeros((kd + 1, n), a.dtype)
                lower = np.zeros((kd + 1, n), a.dtype)
                for j in range(n):
                    for i in range(max(0, j - kd), j + 1):
                        upper[kd + i - j, j] = a[i, j]
                    for i in range(j, min(n, j + kd + 1)):
                        lower[i - j, j] = a[i, j]
                b = rng.standard_normal((n, 2))
                for form, ab in ((False, upper), (True, lower)):
                    with self.subTest(n=n, kd=kd, complex_entries=complex_entries, lower=form):
                        x = np.asarray(numerical.solveh_banded(ab, b, lower=form))
                        np.testing.assert_allclose(a @ x, b, atol=1e-12)
                        if sl is not None:
                            np.testing.assert_allclose(x, sl.solveh_banded(ab, b, lower=form), rtol=1e-11, atol=1e-13)
        with self.assertRaises(numerical.LinAlgError):
            numerical.solveh_banded(np.array([[0.0, 1.0], [1.0, -1.0]]), np.ones(2))
        with self.assertRaises(ValueError):
            numerical.solveh_banded(np.ones((2, 3)), np.ones(4))


if __name__ == "__main__":
    unittest.main()
