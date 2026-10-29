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
                        if sl is not None and n > 1:  # SciPy's 1 × 1 shortcut casts a complex quotient into b's float, and 1.11's reads a row past ab
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


@unittest.skipIf(np is None, "NumPy is not installed")
class Structured(unittest.TestCase):
    def test_solve_toeplitz_and_circulant(self):
        rng = np.random.default_rng(4)
        for n in (1, 2, 7, 40):
            for complex_entries in (False, True):
                c = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_entries else 0)
                r = rng.standard_normal(n) + (1j * rng.standard_normal(n) if complex_entries else 0)
                c[0] += n
                r[0] = c[0]
                toeplitz = np.array([[c[i - j] if i >= j else r[j - i] for j in range(n)] for i in range(n)])
                hermitian = np.array([[c[i - j] if i >= j else np.conj(c[j - i]) for j in range(n)] for i in range(n)])
                circulant = np.array([[c[(i - j) % n] for j in range(n)] for i in range(n)])
                for b in (rng.standard_normal(n), rng.standard_normal((n, 2))):
                    with self.subTest(n=n, complex_entries=complex_entries, rank=b.ndim):
                        x = np.asarray(numerical.solve_toeplitz((c, r), b))
                        np.testing.assert_allclose(toeplitz @ x, b, atol=1e-12)
                        if not complex_entries or n > 1:  # a complex 1 × 1 Hermitian Toeplitz needs a real c[0]
                            x = np.asarray(numerical.solve_toeplitz(c.real if n == 1 else c, b))
                            np.testing.assert_allclose((hermitian.real if n == 1 else hermitian) @ x, b, atol=1e-12)
                        y = np.asarray(numerical.solve_circulant(c, b))
                        self.assertEqual(y.shape, b.shape)
                        np.testing.assert_allclose(circulant @ y, b, atol=1e-12)
                        if sl is not None:
                            np.testing.assert_allclose(np.asarray(numerical.solve_toeplitz((c, r), b)), sl.solve_toeplitz((c, r), b), atol=1e-12)
                            np.testing.assert_allclose(y, sl.solve_circulant(c, b), atol=1e-12)
                            self.assertEqual(y.dtype, sl.solve_circulant(c, b).dtype)
        least = np.asarray(numerical.solve_circulant([1, 1, 0, 0], [1, 2, 3, 4], singular="lstsq"))
        np.testing.assert_allclose(least, [0.25, 1.25, 2.25, 1.25])
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_circulant([1, 1, 0, 0], [1, 2, 3, 4])
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_toeplitz(([0.0, 1.0], [0.0, 1.0]), [1.0, 2.0])
        for bad in (lambda: numerical.solve_circulant([1, 2], [1, 2], singular="other"),
                    lambda: numerical.solve_circulant([1, 2], [1, 2], baxis=1),
                    lambda: numerical.solve_circulant([1, 2], [1, 2, 3]),
                    lambda: numerical.solve_toeplitz(([1, 2], [1]), [1, 2]),
                    lambda: numerical.solve_toeplitz([1, 2], [1, 2, 3])):
            with self.assertRaises(ValueError):
                bad()


@unittest.skipIf(np is None, "NumPy is not installed")
class Constructors(unittest.TestCase):
    def test_against_scipy(self):
        expected = {
            ("toeplitz", (1, 2, 3), None): [[1, 2, 3], [2, 1, 2], [3, 2, 1]],
            ("toeplitz", (1, 2, 3), (9, 4, 5, 6)): [[1, 4, 5, 6], [2, 1, 4, 5], [3, 2, 1, 4]],
            ("hankel", (1, 2, 3), None): [[1, 2, 3], [2, 3, 0], [3, 0, 0]],
            ("hankel", (1, 2, 3), (9, 4, 5, 6)): [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
        }
        for (name, c, r), want in expected.items():
            got = getattr(numerical, name)(c, r)
            self.assertEqual(np.asarray(got).tolist(), want)
            if sl is not None:
                self.assertEqual(np.asarray(got).tolist(), getattr(sl, name)(c, r).tolist())
        self.assertEqual(np.asarray(numerical.circulant([1, 2, 3])).tolist(), [[1, 3, 2], [2, 1, 3], [3, 2, 1]])
        hermitian = np.asarray(numerical.toeplitz([1, 2j, 3]))
        np.testing.assert_array_equal(hermitian, hermitian.conj().T)
        self.assertEqual(numerical.toeplitz(np.arange(3, dtype=np.float32)).dtype, "float32")
        self.assertEqual(numerical.toeplitz([1.0, 2.0], [1.0, 3j]).dtype, "complex128")
        self.assertEqual(np.asarray(numerical.toeplitz(3.0)).tolist(), [[3.0]])
        blocks = np.asarray(numerical.block_diag([[1, 2], [3, 4]], 5, [6, 7], np.eye(1) * 1j))
        self.assertEqual(blocks.shape, (5, 6))
        self.assertEqual(blocks.dtype, np.complex128)
        if sl is not None:
            np.testing.assert_array_equal(blocks, sl.block_diag([[1, 2], [3, 4]], 5, [6, 7], np.eye(1) * 1j))
        self.assertEqual(np.asarray(numerical.block_diag()).shape, (1, 0))
        for bad in (lambda: numerical.toeplitz(np.ones((2, 2))), lambda: numerical.circulant(np.ones((2, 2))),
                    lambda: numerical.block_diag(np.ones((2, 2, 2)))):
            with self.assertRaises(ValueError):
                bad()


@unittest.skipIf(np is None, "NumPy is not installed")
class PivotedQR(unittest.TestCase):
    def test_against_scipy(self):
        # SciPy's permutation and R up to the signs of its rows (ours real and non-negative on the diagonal); every mode
        rng = np.random.default_rng(9)
        for shape in ((5, 5), (8, 3), (3, 8), (1, 1)):
            for complex_entries in (False, True):
                a = rng.standard_normal(shape) + (1j * rng.standard_normal(shape) if complex_entries else 0)
                k = min(shape)
                q, r, p = numerical.qr(a, pivoting=True)
                q, r = np.asarray(q), np.asarray(r)
                self.assertEqual((q.shape, r.shape), ((shape[0], k), (k, shape[1])))
                np.testing.assert_allclose(q @ r, a[:, list(p)], atol=1e-13)
                np.testing.assert_allclose(q.conj().T @ q, np.eye(k), atol=1e-13)
                self.assertTrue(np.all(np.diag(r).real >= 0) and np.all(np.diag(r).imag == 0))
                if sl is not None:
                    _, r_s, p_s = sl.qr(a, pivoting=True, mode="economic")
                    self.assertEqual(list(p), list(p_s))
                    np.testing.assert_allclose(np.abs(r), np.abs(r_s), atol=1e-12)
                q_full, r_full, p_full = numerical.qr(a, "complete", pivoting=True)
                self.assertEqual((np.asarray(q_full).shape, np.asarray(r_full).shape), ((shape[0], shape[0]), shape))
                r_only, p_only = numerical.qr(a, "r", pivoting=True)
                np.testing.assert_allclose(np.asarray(r_only), r, atol=1e-14)
                self.assertEqual(p_only, p)
        self.assertEqual(numerical.qr(np.eye(2, dtype=np.float32), pivoting=True)[1].dtype, "float32")


if __name__ == "__main__":
    unittest.main()
