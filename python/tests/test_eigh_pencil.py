"""eigh(a, b=...) and eigvalsh(a, b=...), the pencil a x = λ b x, against scipy.linalg.eigh(a, b) where it is
installed, and against its defining equations everywhere."""

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


def _pencil(n, dtype, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    m = rng.standard_normal((n, n))
    if dtype.startswith("complex"):
        a = a + 1j * rng.standard_normal((n, n))
        m = m + 1j * rng.standard_normal((n, n))
    a = (a + a.conj().T) / 2
    b = m @ m.conj().T + n * np.eye(n)
    return a.astype(dtype), b.astype(dtype)


@unittest.skipIf(np is None, "NumPy is not installed")
class Pencil(unittest.TestCase):
    def test_every_dtype_and_method_solves_the_pencil(self):
        for dtype in ("float64", "complex128", "float32", "complex64"):
            tolerance = 1e-10 if dtype in ("float64", "complex128") else 2e-3
            methods = ("divide_and_conquer", "tridiagonal_ql") + (("jacobi",) if not dtype.startswith("complex") else ())
            for n in (1, 6, 70):
                a, b = _pencil(n, dtype, n)
                for method in methods:
                    for uplo in ("L", "U"):
                        values, vectors = numerical.eigh(a, uplo, b=b, method=method)
                        w, x = np.asarray(values), np.asarray(vectors)
                        self.assertEqual(w.dtype, np.float64 if dtype in ("float64", "complex128") else np.float32)
                        self.assertEqual(x.dtype, np.dtype(dtype))
                        scale = np.abs(a).max() + np.abs(b).max() * np.abs(w).max()
                        self.assertLess(np.abs(a @ x - (b @ x) * w).max(), tolerance * scale)
                        self.assertLess(np.abs(x.conj().T @ b @ x - np.eye(n)).max(), tolerance * n)
                        self.assertTrue(np.all(np.diff(w) >= 0))
                        np.testing.assert_allclose(np.asarray(numerical.eigvalsh(a, uplo, b=b)), w,
                                                   rtol=tolerance, atol=tolerance * np.abs(w).max())

    def test_only_the_named_triangle_is_read(self):
        a, b = _pencil(5, "float64", 9)
        values = np.asarray(numerical.eigh(a, b=b)[0])
        a_low, b_low = np.tril(a) + np.triu(np.full((5, 5), np.nan), 1), np.tril(b) + np.triu(np.full((5, 5), np.nan), 1)
        np.testing.assert_allclose(np.asarray(numerical.eigh(a_low, b=b_low)[0]), values, rtol=1e-13)
        np.testing.assert_allclose(np.asarray(numerical.eigvalsh(a_low.T, "U", b=b_low.T)), values, rtol=1e-13)

    def test_mixed_dtypes_promote(self):
        a, b = _pencil(4, "float64", 3)
        values, vectors = numerical.eigh(a.astype("float32"), b=b.astype("complex128"))
        self.assertEqual(np.asarray(vectors).dtype, np.complex128)

    def test_errors(self):
        a, b = _pencil(3, "float64", 4)
        with self.assertRaises(numerical.LinAlgError):
            numerical.eigh(a, b=-b)  # not positive definite
        with self.assertRaises(numerical.LinAlgError):
            numerical.eigh(a, b=np.eye(2))
        with self.assertRaises(numerical.LinAlgError):
            numerical.eigvalsh(np.stack([a, a]), b=np.stack([b, b]))
        with self.assertRaises(numerical.LinAlgError):
            numerical.eigh(a.astype(complex), b=b, method="jacobi")
        with self.assertRaises(ValueError):
            numerical.eigh(a, "X", b=b)
        with self.assertRaises(TypeError):
            numerical.eigh(a, "L", b)  # b is keyword-only: UPLO holds NumPy's place

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        for dtype in ("float64", "complex128"):
            for n in (1, 8, 90):
                a, b = _pencil(n, dtype, 20 + n)
                for lower in (True, False):
                    w, x = sl.eigh(a, b, lower=lower)
                    values, vectors = numerical.eigh(a, "L" if lower else "U", b=b)
                    np.testing.assert_allclose(np.asarray(values), w, rtol=1e-10, atol=1e-10 * np.abs(w).max())
                    v = np.asarray(vectors)
                    # Each vector to a unit-modulus factor, the values being distinct.
                    phases = np.sum(x.conj() * (b @ v), axis=0)
                    np.testing.assert_allclose(np.abs(phases), np.ones(n), rtol=1e-8)
                    np.testing.assert_allclose(v, x * phases, atol=1e-8 * np.abs(x).max())


if __name__ == "__main__":
    unittest.main()
