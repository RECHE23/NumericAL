"""solve_triangular, lu_factor / lu_solve and cho_factor / cho_solve against scipy.linalg where it is installed, and
against the products they invert everywhere."""

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


def _drawn(rows, cols, dtype, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols))
    if dtype.startswith("complex"):
        x = x + 1j * rng.standard_normal((rows, cols))
    return x.astype(dtype)


@unittest.skipIf(np is None, "NumPy is not installed")
class TriangularSolves(unittest.TestCase):
    def test_every_triangle_trans_and_diagonal_inverts_its_product(self):
        for dtype in ("float64", "complex128", "float32", "complex64"):
            tolerance = 1e-10 if dtype in ("float64", "complex128") else 1e-3
            for n, k in ((1, 1), (7, 3), (70, 5)):
                a = _drawn(n, n, dtype, 1) + n * np.eye(n, dtype=dtype)
                for lower in (True, False):
                    triangle = np.tril(a) if lower else np.triu(a)
                    for trans, op in ((0, lambda m: m), (1, lambda m: m.T), (2, lambda m: m.conj().T),
                                      ("N", lambda m: m), ("T", lambda m: m.T), ("C", lambda m: m.conj().T)):
                        for unit in (False, True):
                            used = triangle - np.diag(np.diag(triangle)) + np.eye(n, dtype=dtype) if unit else triangle
                            for b in (_drawn(n, k, dtype, 2), _drawn(n, 1, dtype, 3)[:, 0]):
                                x = np.asarray(numerical.solve_triangular(a, b, trans=trans, lower=lower, unit_diagonal=unit))
                                self.assertEqual(x.dtype, np.dtype(dtype))
                                self.assertEqual(x.shape, b.shape)
                                self.assertLess(np.max(np.abs(op(used) @ x - b)), tolerance * max(1.0, np.max(np.abs(x))))

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        a = _drawn(40, 40, "float64", 4) + 40 * np.eye(40)
        b = _drawn(40, 6, "float64", 5)
        for lower in (True, False):
            for trans in (0, 1, 2):
                ours = np.asarray(numerical.solve_triangular(a, b, trans=trans, lower=lower))
                np.testing.assert_allclose(ours, sl.solve_triangular(a, b, trans=trans, lower=lower), rtol=1e-12, atol=1e-14)

    def test_every_form_of_the_operands_gives_the_same_bits(self):
        # a NumPy a read where it lies, or converted: a Tensor, a Fortran-ordered or strided a, b a list, a Tensor, a
        # strided b too large to gather, b of another dtype
        for dtype in ("float64", "complex128"):
            n = 300
            a = _drawn(n, n, dtype, 6) + n * np.eye(n, dtype=dtype)
            b = _drawn(n, 260, dtype, 7)
            spread = np.zeros((n, 2 * n), dtype=dtype)
            spread[:, ::2] = a
            for lower in (True, False):
                for trans in (0, 1, 2):
                    expected = np.asarray(numerical.solve_triangular(a, b, trans=trans, lower=lower))
                    for left, right in ((numerical.asarray(a), b), (np.asfortranarray(a), b), (a, numerical.asarray(b)),
                                        (a, np.asfortranarray(b)), (a, b.tolist() if dtype == "float64" else b),
                                        (spread[:, ::2], b)):
                        x = np.asarray(numerical.solve_triangular(left, right, trans=trans, lower=lower))
                        self.assertTrue(np.array_equal(x, expected))
                    column = np.asarray(numerical.solve_triangular(a, b[:, 3], trans=trans, lower=lower))
                    self.assertTrue(np.array_equal(column, np.asarray(numerical.solve_triangular(a, b[:, 3].copy(), trans=trans, lower=lower))))
        # an integer b beside a float64 a, and an empty system
        a = np.array([[2.0, 0.0], [1.0, 4.0]])
        self.assertEqual(np.asarray(numerical.solve_triangular(a, np.array([2, 5]), lower=True)).tolist(), [1.0, 1.0])
        self.assertEqual(np.asarray(numerical.solve_triangular(np.zeros((0, 0)), np.zeros(0))).shape, (0,))
        # the errors of an array read where it lies
        with self.assertRaisesRegex(numerical.LinAlgError, "singular matrix: resolution failed at diagonal 2"):
            numerical.solve_triangular(np.array([[2.0, 0.0], [1.0, 0.0]]), np.array([1.0, 2.0]), lower=True)
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_triangular(np.array([[1.0, 0.0], [np.inf, 1.0]]), np.array([1.0, 2.0]), lower=True)
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_triangular(np.eye(2), np.array([1.0, np.nan]))
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_triangular(np.eye(2), np.ones(3))
        # a complex b beside a real a: both complex, as SciPy's
        mixed = np.asarray(numerical.solve_triangular(np.array([[2.0, 0.0], [1.0, 4.0]]), np.array([2.0j, 1.0 + 1.0j]), lower=True))
        self.assertEqual(mixed.dtype, np.complex128)
        self.assertEqual(mixed.tolist(), [1.0j, 0.25])

    def test_errors(self):
        a = np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 5.0, 6.0]])
        with self.assertRaisesRegex(numerical.LinAlgError, "singular matrix: resolution failed at diagonal 2"):
            numerical.solve_triangular(a, [1.0, 2.0, 3.0], lower=True)
        self.assertEqual(np.asarray(numerical.solve_triangular(a, [2.0, 1.0, 3.0], lower=True, unit_diagonal=True)).tolist(), [2.0, -1.0, 0.0])
        with self.assertRaises(ValueError):
            numerical.solve_triangular(a, [1.0, 2.0, 3.0], trans="X")
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_triangular(np.ones((2, 3)), [1.0, 2.0])
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_triangular(np.eye(2), [1.0, 2.0, 3.0])
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_triangular(np.eye(2), np.ones((2, 2, 2)))
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_triangular(np.array([[1.0, 0.0], [np.nan, 1.0]]), [1.0, 2.0], lower=True)
        # the triangle not read may hold anything
        x = numerical.solve_triangular(np.array([[1.0, np.nan], [2.0, 1.0]]), [1.0, 2.0], lower=True)
        self.assertEqual(np.asarray(x).tolist(), [1.0, 0.0])


@unittest.skipIf(np is None, "NumPy is not installed")
class Factorisations(unittest.TestCase):
    def test_lu_factor_and_solve(self):
        for dtype in ("float64", "complex128", "float32", "complex64"):
            tolerance = 1e-10 if dtype in ("float64", "complex128") else 1e-3
            a = _drawn(30, 30, dtype, 6)
            f = numerical.lu_factor(a)
            lu, piv = f
            self.assertIs(f.lu, lu)
            self.assertIs(f.piv, piv)
            self.assertEqual(len(piv), 30)
            for trans, op in ((0, lambda m: m), (1, lambda m: m.T), (2, lambda m: m.conj().T)):
                for b in (_drawn(30, 4, dtype, 7), _drawn(30, 1, dtype, 8)[:, 0]):
                    x = np.asarray(numerical.lu_solve((lu, piv), b, trans=trans))
                    self.assertLess(np.max(np.abs(op(a) @ x - b)), tolerance * max(1.0, np.max(np.abs(x))) * 30)
                    self.assertTrue(np.array_equal(x, np.asarray(f.solve(b, trans))))

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_lu_against_scipy(self):
        a = _drawn(25, 25, "float64", 9)
        lu, piv = numerical.lu_factor(a)
        their_lu, their_piv = sl.lu_factor(a)
        self.assertEqual(list(piv), list(their_piv))
        np.testing.assert_allclose(np.asarray(lu), their_lu, rtol=1e-12, atol=1e-13)
        b = _drawn(25, 3, "float64", 10)
        for trans in (0, 1, 2):
            np.testing.assert_allclose(np.asarray(numerical.lu_solve((lu, piv), b, trans)), sl.lu_solve((their_lu, their_piv), b, trans),
                                       rtol=1e-11, atol=1e-12)
        # SciPy's pair is read too
        np.testing.assert_allclose(np.asarray(numerical.lu_solve((their_lu, their_piv), b)), np.linalg.solve(a, b), rtol=1e-10)

    def test_lu_errors(self):
        singular = numerical.lu_factor(np.array([[1.0, 2.0], [2.0, 4.0]]))
        with self.assertRaises(numerical.LinAlgError):
            numerical.lu_solve(singular, [1.0, 2.0])
        with self.assertRaises(numerical.LinAlgError):
            numerical.lu_factor(np.ones((2, 3)))
        with self.assertRaises(numerical.LinAlgError):
            numerical.lu_solve((np.eye(2), (0,)), [1.0, 2.0])
        with self.assertRaises(numerical.LinAlgError):
            numerical.lu_solve((np.eye(2), (0, 5)), [1.0, 2.0])
        with self.assertRaises(numerical.LinAlgError):
            numerical.lu_solve((np.eye(2), (0, "x")), [1.0, 2.0])
        with self.assertRaises(numerical.LinAlgError):
            numerical.lu_solve((np.eye(2), 3), [1.0, 2.0])
        with self.assertRaises(numerical.LinAlgError):
            numerical.lu_solve((np.ones((2, 3)), (0, 1)), [1.0, 2.0])
        with self.assertRaises(numerical.LinAlgError):
            numerical.lu_solve((np.eye(2), (0, 1)), [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            numerical.lu_solve((np.eye(2), (0, 1)), [1.0, 2.0], trans=3)
        x = numerical.lu_solve((np.eye(2), (0, 1)), numerical.asarray([1.0, 2.0], "complex128"))
        self.assertEqual(x.dtype, "complex128")
        # An infinite or NaN entry of the factor (either triangle) or of b, refused before any solve.
        for lu, b in ((np.array([[1.0, np.nan], [0.0, 1.0]]), [1.0, 2.0]), (np.array([[1.0, 0.0], [np.inf, 1.0]]), [1.0, 2.0]),
                      (np.eye(2), [1.0, np.nan])):
            with self.assertRaises(numerical.LinAlgError) as caught:
                numerical.lu_solve((lu, (0, 1)), b)
            self.assertIn("infinite or NaN", str(caught.exception))

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_lu_solve_overflow_between_its_solves_is_the_result(self):
        # L y = b overflows (1e308 + 1e308) though the factor and b are finite: the infinity carried to x (and 0 · ∞ = NaN
        # above it), as SciPy.
        f = numerical.lu_factor(np.array([[1.0, 0.0], [-1.0, 1.0]]))
        b = np.array([1e308, 1e308])
        x = np.asarray(numerical.lu_solve(f, b))
        np.testing.assert_array_equal(x, sl.lu_solve(sl.lu_factor(np.array([[1.0, 0.0], [-1.0, 1.0]])), b))  # [nan, inf]
        self.assertEqual(x[1], np.inf)

    def test_cholesky_factor_and_solve(self):
        for dtype in ("float64", "complex128", "float32", "complex64"):
            tolerance = 1e-10 if dtype in ("float64", "complex128") else 1e-3
            m = _drawn(20, 20, dtype, 11)
            a = m @ m.conj().T + 20 * np.eye(20, dtype=dtype)
            b = _drawn(20, 3, dtype, 12)
            for lower in (False, True):
                f = numerical.cho_factor(a, lower=lower)
                c, flag = f
                self.assertEqual((f.c is c, f.lower), (True, lower))
                self.assertEqual(flag, lower)
                x = np.asarray(numerical.cho_solve((c, flag), b))
                self.assertLess(np.max(np.abs(a @ x - b)), tolerance * 20)
                self.assertTrue(np.array_equal(x, np.asarray(f.solve(b))))

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_cholesky_against_scipy(self):
        m = _drawn(15, 15, "float64", 13)
        a = m @ m.T + 15 * np.eye(15)
        b = _drawn(15, 2, "float64", 14)
        for lower in (False, True):
            np.testing.assert_allclose(np.asarray(numerical.cho_solve(numerical.cho_factor(a, lower=lower), b)),
                                       sl.cho_solve(sl.cho_factor(a, lower=lower), b), rtol=1e-12)
        with self.assertRaises(numerical.LinAlgError):
            numerical.cho_factor(-np.eye(3))


if __name__ == "__main__":
    unittest.main()
