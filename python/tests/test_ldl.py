"""ldl against its definition, and against scipy.linalg.ldl where it is installed."""

import unittest
import warnings

import numerical

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import scipy.linalg as sl
except ImportError:  # pragma: no cover
    sl = None


def _symmetric(n, kind, rng):
    """A real symmetric, complex Hermitian or complex symmetric matrix of order `n`."""
    a = rng.standard_normal((n, n))
    if kind == "real":
        return a + a.T
    a = a + 1j * rng.standard_normal((n, n))
    return a + a.conj().T if kind == "hermitian" else a + a.T


@unittest.skipIf(np is None, "NumPy is not installed")
class Ldl(unittest.TestCase):
    def test_the_factorisation_holds(self):
        rng = np.random.default_rng(9)
        for n in (1, 4, 13, 40):
            for kind in ("real", "hermitian", "symmetric"):
                a = _symmetric(n, kind, rng)
                hermitian = kind != "symmetric"
                for lower in (True, False):
                    with self.subTest(n=n, kind=kind, lower=lower):
                        lu, d, perm = numerical.ldl(a, lower=lower, hermitian=hermitian)
                        lu, d = np.asarray(lu), np.asarray(d)
                        self.assertIsInstance(perm, tuple)
                        triangle = np.tril if lower else np.triu
                        self.assertTrue(np.array_equal(lu[list(perm)], triangle(lu[list(perm)])))
                        self.assertTrue(np.all(np.diag(lu[list(perm)]) == 1))
                        outer = lu.conj().T if hermitian else lu.T
                        self.assertLess(np.abs(lu @ d @ outer - a).max(), 1e-12 * np.abs(a).max())

    def test_single_precision_the_diagonal_and_refusals(self):
        a = np.array([[2.0, -1.0, 3.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
        lu, d, perm = numerical.ldl(a, lower=False)
        self.assertEqual(lu.dtype, "float32")
        self.assertEqual(perm, (2, 1, 0))
        self.assertEqual(np.asarray(d).tolist(), [[-5.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 2.0]])
        tilted = np.array([[2.0 + 1j, 1.0], [1.0, 3.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lu, d, _ = numerical.ldl(tilted)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))
        self.assertEqual(np.asarray(d)[0, 0], 2.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            numerical.ldl(tilted, hermitian=False)
        self.assertEqual(caught, [])
        for bad in (np.ones((2, 3)), np.array([[np.nan]])):
            with self.assertRaises(numerical.LinAlgError):
                numerical.ldl(bad)

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        # SciPy's sytrf is sytf2 up to order 64 (its block): the same pivots, the same factors to rounding
        rng = np.random.default_rng(10)
        for n in (2, 7, 30, 64):
            for kind in ("real", "hermitian", "symmetric"):
                for integers in (False, True):  # integer entries: ties decide pivots
                    a = np.round(3 * _symmetric(n, kind, rng)) if integers else _symmetric(n, kind, rng)
                    hermitian = kind != "symmetric"
                    for lower in (True, False):
                        with self.subTest(n=n, kind=kind, integers=integers, lower=lower), warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            ours = numerical.ldl(a, lower=lower, hermitian=hermitian)
                            theirs = sl.ldl(a, lower=lower, hermitian=hermitian)
                        self.assertEqual(ours[2], tuple(theirs[2].tolist()))
                        for x, y in zip(ours[:2], theirs[:2]):
                            np.testing.assert_allclose(np.asarray(x), y, rtol=0, atol=1e-12 * max(1.0, np.abs(y).max()))


if __name__ == "__main__":
    unittest.main()
