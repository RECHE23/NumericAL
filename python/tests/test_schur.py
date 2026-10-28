"""schur(a, output) against scipy.linalg.schur where it is installed, and against its defining properties everywhere:
a = Z T Zᴴ, Z orthogonal (unitary), T quasi-triangular (real) or triangular (complex)."""

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


def _drawn(n, dtype, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n))
    if dtype.startswith("complex"):
        x = x + 1j * rng.standard_normal((n, n))
    return x.astype(dtype)


@unittest.skipIf(np is None, "NumPy is not installed")
class Schur(unittest.TestCase):
    def test_every_dtype_and_output(self):
        for dtype in ("float64", "complex128", "float32", "complex64"):
            tolerance = 1e-12 if dtype in ("float64", "complex128") else 1e-4
            for output in ("real", "r", "complex", "c"):
                complex_out = dtype.startswith("complex") or output in ("complex", "c")
                expected = {("float64", False): np.float64, ("float64", True): np.complex128,
                            ("float32", False): np.float32, ("float32", True): np.complex64}
                kind = "float64" if dtype in ("float64", "complex128") else "float32"
                for n in (1, 6, 120):
                    a = _drawn(n, dtype, n)
                    t, z = (np.asarray(m) for m in numerical.schur(a, output))
                    self.assertEqual(t.dtype, expected[(kind, complex_out)])
                    self.assertEqual(z.dtype, expected[(kind, complex_out)])
                    scale = np.abs(a).max() * n
                    self.assertLess(np.abs(z @ t @ z.conj().T - a).max(), tolerance * scale)
                    self.assertLess(np.abs(z.conj().T @ z - np.eye(n)).max(), tolerance * n)
                    self.assertTrue(np.all(np.tril(t, -1 if complex_out else -2) == 0))

    def test_errors(self):
        with self.assertRaises(numerical.LinAlgError):
            numerical.schur(np.ones((2, 3)))
        with self.assertRaises(numerical.LinAlgError):
            numerical.schur(np.array([[np.nan]]))
        with self.assertRaises(ValueError):
            numerical.schur(np.eye(2), "x")

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        for dtype in ("float64", "complex128"):
            for n in (5, 60):
                a = _drawn(n, dtype, 40 + n)
                for output in ("real", "complex"):
                    t, z = (np.asarray(m) for m in numerical.schur(a, output))
                    expected_t, expected_z = sl.schur(a, output)
                    self.assertEqual(t.dtype, expected_t.dtype)
                    self.assertEqual(t.shape, expected_t.shape)
                    # The same eigenvalues on T's diagonal (and blocks) as SciPy's, as sets.
                    # (Each to its nearest: a pair's halves sort by real parts equal to rounding.)
                    ours = np.linalg.eigvals(t)
                    theirs = np.linalg.eigvals(expected_t)
                    nearest = np.abs(ours[:, None] - theirs[None, :]).min(axis=1)
                    self.assertLess(nearest.max(), 1e-9 * np.abs(theirs).max())
                    self.assertLess(np.abs(np.sort(np.abs(ours)) - np.sort(np.abs(theirs))).max(), 1e-9 * np.abs(theirs).max())


if __name__ == "__main__":
    unittest.main()
