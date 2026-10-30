"""scipy.linalg's hessenberg, matrix_balance, eigh_tridiagonal and eigvalsh_tridiagonal, cholesky_banded and
cho_solve_banded, rsf2csf, rq and cdf2rdf against SciPy's own: by their defining identities and, where the factor is
unique, entry by entry. Deterministic; skipped when SciPy is not importable (CI installs it)."""
import itertools
import unittest

import numerical

try:
    import numpy as np
    import scipy.linalg as sl
except ImportError:
    np = sl = None


def _close(got, expected, tolerance=1e-12):
    got, expected = np.asarray(got), np.asarray(expected)
    return got.shape == expected.shape and np.allclose(got, expected, rtol=tolerance, atol=tolerance * max(1.0, np.abs(expected).max(initial=0.0)))


@unittest.skipIf(sl is None, "SciPy not installed")
class Decompositions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261108)

    def matrix(self, n, dtype=None, m=None):
        dtype = np.float64 if dtype is None else dtype
        a = self.rng.standard_normal((n, m or n))
        if np.dtype(dtype).kind == "c":
            a = a + 1j * self.rng.standard_normal((n, m or n))
        return a.astype(dtype)

    def test_hessenberg_as_scipy(self):
        for n, dtype in itertools.product((1, 2, 3, 6, 40, 150), (np.float64, np.complex128, np.float32)):
            a = self.matrix(n, dtype)
            with self.subTest(n=n, dtype=dtype.__name__):
                h, q = numerical.hessenberg(a, calc_q=True)
                h_expected, q_expected = sl.hessenberg(a, calc_q=True)
                self.assertEqual(h.dtype, str(h_expected.dtype))
                tolerance = 1e-4 if dtype is np.float32 else 1e-11
                h, q = np.asarray(h), np.asarray(q)
                self.assertTrue(np.array_equal(np.tril(h, -2), np.zeros_like(h)))
                self.assertTrue(_close(q @ h @ q.conj().T, a, tolerance))
                self.assertTrue(_close(q.conj().T @ q, np.eye(n), tolerance))
                # unique up to the phases of Q's columns past the first (Q S, H then Sᴴ H S): real signs for a real a
                signs = np.ones(n, dtype=h.dtype)
                for k in range(1, n):
                    column = q[:, k]
                    largest = np.argmax(np.abs(column))
                    signs[k] = q_expected[largest, k] / column[largest] if column[largest] != 0 else 1
                self.assertTrue(_close(q * signs, q_expected, tolerance))
                self.assertTrue(_close(np.conj(signs)[:, None] * h * signs[None, :], h_expected, tolerance))
                self.assertTrue(_close(numerical.hessenberg(a), h, 0.0))
        with self.assertRaises(ValueError):
            numerical.hessenberg(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            numerical.hessenberg(np.array([[1.0, np.nan, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    def test_matrix_balance_as_scipy(self):
        a = np.array([[1.0, 2**30, 3.0, 2**-30, 5.0], [0.0, 2.0, 2**40, 1.0, 7.0], [0.0, 2**-40, 3.0, 2.0, 1.0],
                      [0.0, 1.0, 2**-20, 4.0, 2.0], [0.0, 0.0, 0.0, 0.0, 5.0]])
        for x, permute, scale in itertools.product((a, a + 1j * np.triu(a), self.matrix(6), a.astype(np.float32)),
                                                   (True, False), (True, False)):
            with self.subTest(dtype=str(x.dtype), permute=permute, scale=scale):
                b, t = numerical.matrix_balance(x, permute=permute, scale=scale)
                b, t = np.asarray(b), np.asarray(t)
                self.assertTrue(np.array_equal(b, np.linalg.solve(t, x @ t).astype(b.dtype)))  # an exact similarity
                b2, (scaling, perm) = numerical.matrix_balance(x, permute=permute, scale=scale, separate=True)
                self.assertTrue(np.array_equal(np.asarray(b2), b))
                self.assertEqual(sorted(perm), list(range(len(x))))
                self.assertEqual(str(np.asarray(scaling).dtype), "float64")
                b_expected, t_expected = sl.matrix_balance(x, permute=permute, scale=scale)
                self.assertTrue(np.array_equal(b, b_expected) and np.array_equal(t, t_expected))
        with self.assertRaises(ValueError):
            numerical.matrix_balance(np.ones((2, 3)))

    def test_tridiagonal_eigenproblems_as_scipy(self):
        for n in (1, 2, 5, 30, 200):
            d, e = self.rng.standard_normal(n), self.rng.standard_normal(n - 1)
            with self.subTest(n=n):
                w, v = numerical.eigh_tridiagonal(d, e)
                w_expected = sl.eigh_tridiagonal(d, e, eigvals_only=True)
                self.assertTrue(_close(w, w_expected))
                t = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
                v = np.asarray(v)
                self.assertTrue(_close(t @ v, v * np.asarray(w)))
                self.assertTrue(_close(v.T @ v, np.eye(n)))
                self.assertTrue(_close(numerical.eigvalsh_tridiagonal(d, e), w_expected))
                lo, hi = sorted(self.rng.standard_normal(2))
                for select, select_range in (("v", (lo, hi)), ("i", (0, n // 2)), ("i", (n - 1, n - 1))):
                    got = numerical.eigvalsh_tridiagonal(d, e, select=select, select_range=select_range)
                    expected = sl.eigvalsh_tridiagonal(d, e, select=select, select_range=select_range)
                    self.assertTrue(_close(got, expected))
                    w_sel, v_sel = numerical.eigh_tridiagonal(d, e, select=select, select_range=select_range)
                    self.assertEqual(np.asarray(v_sel).shape, (n, len(expected)))
                    self.assertTrue(_close(t @ np.asarray(v_sel), np.asarray(v_sel) * np.asarray(w_sel)))
        w32 = numerical.eigvalsh_tridiagonal(np.ones(3, np.float32), np.ones(2, np.float32))
        self.assertEqual(w32.dtype, "float32")
        for call, error in ((lambda: numerical.eigh_tridiagonal([1.0, 2.0], [1.0, 2.0]), ValueError),
                            (lambda: numerical.eigh_tridiagonal([1j, 2.0], [1.0]), TypeError),
                            (lambda: numerical.eigh_tridiagonal([1.0, 2.0], [1.0], select="x"), ValueError),
                            (lambda: numerical.eigh_tridiagonal([1.0, 2.0], [1.0], select="i", select_range=(0, 2)), ValueError),
                            (lambda: numerical.eigh_tridiagonal([1.0, 2.0], [1.0], select="v", select_range=(1.0, 0.0)), ValueError),
                            (lambda: numerical.eigh_tridiagonal([1.0, 2.0], [1.0], lapack_driver="x"), ValueError),
                            (lambda: numerical.eigh_tridiagonal([1.0, 2.0], [1.0], lapack_driver="sterf"), ValueError),
                            (lambda: numerical.eigh_tridiagonal([1.0, 2.0], [1.0], select="v", select_range=(0, 1),
                                                                lapack_driver="stev"), ValueError),
                            (lambda: numerical.eigh_tridiagonal([1.0, np.inf], [1.0]), ValueError)):
            with self.assertRaises(error):
                call()

    def test_banded_cholesky_as_scipy(self):
        for n, kd, dtype in itertools.product((1, 4, 30), (0, 1, 3), (np.float64, np.complex128)):
            dense = np.zeros((n, n), dtype)
            for i in range(n):
                for j in range(max(0, i - kd), i):
                    dense[i, j] = self.rng.standard_normal() + (1j * self.rng.standard_normal() if dtype is np.complex128 else 0)
                    dense[j, i] = np.conj(dense[i, j])
                dense[i, i] = 4 * kd + 2
            upper = np.zeros((kd + 1, n), dtype)
            lower = np.zeros((kd + 1, n), dtype)
            for j in range(n):
                for i in range(max(0, j - kd), j + 1):
                    upper[kd + i - j, j] = dense[i, j]
                for i in range(j, min(n, j + kd + 1)):
                    lower[i - j, j] = dense[i, j]
            b = self.rng.standard_normal((n, 2))
            with self.subTest(n=n, kd=kd, dtype=dtype.__name__):
                for form, is_lower in ((upper, False), (lower, True)):
                    cb = numerical.cholesky_banded(form, lower=is_lower)
                    self.assertTrue(_close(cb, sl.cholesky_banded(form, lower=is_lower)))
                    self.assertTrue(_close(numerical.cho_solve_banded((cb, is_lower), b),
                                           sl.cho_solve_banded((np.asarray(cb), is_lower), b)))
                    self.assertTrue(_close(numerical.cho_solve_banded((cb, is_lower), b[:, 0]), np.linalg.solve(dense, b[:, 0])))
        with self.assertRaises(numerical.LinAlgError):
            numerical.cholesky_banded([[0.0, 1.0], [1.0, -1.0]])

    def test_rsf2csf_as_scipy(self):
        for n in (1, 2, 5, 20):
            a = self.matrix(n)
            t, z = sl.schur(a)
            with self.subTest(n=n):
                got_t, got_z = numerical.rsf2csf(t, z)
                expected_t, expected_z = sl.rsf2csf(t, z)
                self.assertTrue(_close(got_t, expected_t, 1e-11) and _close(got_z, expected_z, 1e-11))
                self.assertEqual(got_t.dtype, "complex128")
        with self.assertRaises(ValueError):
            numerical.rsf2csf(np.eye(2), np.eye(3))

    def test_rq_as_scipy(self):
        for (m, n), dtype, mode in itertools.product(((3, 5), (5, 3), (4, 4), (1, 3)), (np.float64, np.complex128),
                                                     ("full", "r", "economic")):
            a = self.matrix(m, dtype, n)
            with self.subTest(shape=(m, n), dtype=dtype.__name__, mode=mode):
                got = numerical.rq(a, mode=mode)
                expected = sl.rq(a, mode=mode)
                if mode == "r":
                    got, expected = (got, None), (expected, None)
                r, q = np.asarray(got[0]), None if got[1] is None else np.asarray(got[1])
                self.assertEqual(r.shape, expected[0].shape)
                if q is not None:
                    self.assertEqual(q.shape, expected[1].shape)
                    self.assertTrue(_close(r @ q, a))
                    self.assertTrue(_close(q @ q.conj().T, np.eye(q.shape[0])))
                k = r.shape[1] - r.shape[0]
                self.assertTrue(np.allclose(np.tril(r, k - 1), 0.0))
                # unique up to the signs (phases) of R's columns and Q's rows: |R| as SciPy's
                self.assertTrue(_close(np.abs(r), np.abs(expected[0])))
        with self.assertRaises(ValueError):
            numerical.rq(np.ones((2, 2)), mode="x")

    def test_cdf2rdf_as_scipy(self):
        a = self.matrix(6)
        w, v = np.linalg.eig(a)
        wr, vr = numerical.cdf2rdf(w, v)
        wr_expected, vr_expected = sl.cdf2rdf(w, v)
        self.assertTrue(_close(wr, wr_expected) and _close(vr, vr_expected))
        stack = np.stack([a, self.matrix(6)])
        ws, vs = np.linalg.eig(stack)
        wr, vr = numerical.cdf2rdf(ws, vs)
        wr_expected, vr_expected = sl.cdf2rdf(ws, vs)
        self.assertTrue(_close(wr, wr_expected) and _close(vr, vr_expected))
        with self.assertRaises(ValueError):
            numerical.cdf2rdf(np.array([1j, 2.0]), np.eye(2))


if __name__ == "__main__":
    unittest.main()
