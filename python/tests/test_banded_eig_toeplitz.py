"""eig_banded, eigvals_banded and matmul_toeplitz against SciPy's answers: the eigenvalues within a few ulps of
LAPACK's, the eigenvectors up to their signs (phases); the Toeplitz products within a few ulps of SciPy's FFT.
Deterministic; the SciPy comparisons skipped without SciPy."""
import unittest

import numerical

try:
    import numpy as np
    import scipy.linalg as sl
except ImportError:
    np = sl = None


def _bands(a, width):
    """The upper and the lower diagonal-ordered forms of the band matrix `a` of `width` off-diagonals."""
    n = a.shape[0]
    upper, lower = np.zeros((width + 1, n), a.dtype), np.zeros((width + 1, n), a.dtype)
    for k in range(min(width, n - 1) + 1):  # the rows past the matrix's diagonals left zero
        upper[width - k, k:] = np.diag(a, k)
        lower[k, :n - k] = np.diag(a, -k)
    return upper, lower


class WithoutSciPy(unittest.TestCase):
    def test_values(self):
        w, v = numerical.eig_banded([[0.0, 1.0], [2.0, 2.0]])  # [[2, 1], [1, 2]]
        self.assertEqual([round(x, 12) for x in w.tolist()], [1.0, 3.0])
        self.assertEqual(v.shape, (2, 2))
        self.assertEqual(numerical.eigvals_banded([[2.0, 2.0], [1.0, 0.0]], lower=True, select="i",
                                                  select_range=[1, 1]).shape, (1,))
        self.assertEqual(numerical.eigvals_banded([[0.0, 1.0], [2.0, 2.0]], select="v", select_range=[5, 6]).shape,
                         (0,))
        self.assertEqual(numerical.eig_banded(numerical.zeros((1, 0)))[1].shape, (0, 0))
        self.assertEqual(numerical.matmul_toeplitz(([1.0, 2.0, 3.0], [1.0, 4.0]), [1.0, 1.0]).tolist(),
                         [5.0, 3.0, 5.0])
        self.assertEqual(numerical.matmul_toeplitz([1.0, 2.0], numerical.zeros((2, 0))).shape, (2, 0))
        cases = [
            (ValueError, lambda: numerical.eig_banded([1.0, 2.0])),
            (ValueError, lambda: numerical.eig_banded([[1.0, float("inf")]])),
            (ValueError, lambda: numerical.eig_banded([[1.0, 2.0]], select="x")),
            (ValueError, lambda: numerical.eig_banded([[1.0, 2.0]], select="i", select_range=[0, 2])),
            (ValueError, lambda: numerical.eig_banded([[1.0, 2.0]], select="i", select_range=[0.0, 1.0])),
            (ValueError, lambda: numerical.eig_banded([[1.0, 2.0]], select="v", select_range=[2.0, 1.0])),
            (ValueError, lambda: numerical.matmul_toeplitz([1.0, 2.0], [1.0, 2.0, 3.0])),
            (ValueError, lambda: numerical.matmul_toeplitz([1.0, 2.0], numerical.zeros((2, 2, 2)))),
        ]
        for error, call in cases:
            with self.assertRaises(error):
                call()


@unittest.skipIf(sl is None, "SciPy is not installed")
class AgainstSciPy(unittest.TestCase):
    def test_eig_banded(self):
        rng = np.random.default_rng(5)
        for n, width, complex_ in ((9, 2, False), (12, 4, True), (6, 7, False), (1, 0, False)):
            a = np.diag(rng.normal(size=n))
            for k in range(1, min(width, n - 1) + 1):
                off = rng.normal(size=n - k) + (1j * rng.normal(size=n - k) if complex_ else 0)
                a = a + np.diag(off, k) + np.diag(np.conj(off), -k)
            for band, lower in zip(_bands(a, width), (False, True)):
                for select, bounds in (("a", None), ("v", (-1.0, 1.0)), ("i", (0, n - 1)), ("i", (n // 2, n // 2)),
                                       ("v", (50.0, 60.0))):
                    label = f"{n} {width} {complex_} {lower} {select} {bounds}"
                    want_w, want_v = sl.eig_banded(band, lower=lower, select=select, select_range=bounds)
                    w, v = numerical.eig_banded(numerical.asarray(band), lower=lower, select=select,
                                                select_range=bounds)
                    w, v = np.asarray(w), np.asarray(v)
                    self.assertEqual((w.shape, v.shape), (want_w.shape, want_v.shape), label)
                    np.testing.assert_allclose(w, want_w, atol=1e-12, err_msg=label)
                    if v.size:  # one eigenvector per column, up to its phase
                        np.testing.assert_allclose(np.abs(v.conj().T @ want_v), np.eye(v.shape[1]), atol=1e-9,
                                                   err_msg=label)
                    np.testing.assert_allclose(np.asarray(numerical.eigvals_banded(band, lower=lower, select=select,
                                                                                   select_range=bounds)),
                                               want_w, atol=1e-12, err_msg=label)

    def test_matmul_toeplitz(self):
        rng = np.random.default_rng(6)
        cases = [(rng.normal(size=5), None, rng.normal(size=5)), (rng.normal(size=6), rng.normal(size=4),
                                                                 rng.normal(size=(4, 3))),
                 (rng.normal(size=3) + 1j, rng.normal(size=5), rng.normal(size=(5, 2))),
                 (rng.normal(size=4), None, rng.normal(size=(4, 2)) * 1j), (rng.normal(size=1), None, [2.0])]
        for c, r, x in cases:
            column_row = c if r is None else (c, r)
            want = sl.matmul_toeplitz(column_row, x)
            got = np.asarray(numerical.matmul_toeplitz(column_row, x))
            self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype))
            np.testing.assert_allclose(got, want, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
