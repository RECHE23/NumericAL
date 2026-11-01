"""scipy.linalg.solve's arguments against SciPy's own: every structure assume_a names (by each of its names), the
triangle lower picks, transposed, vector and matrix right-hand sides, stacks; the refusals (a singular matrix, one not
positive definite, an unknown structure, a complex transposed system); the reciprocal condition number's warning
where SciPy's warns. Deterministic; skipped when SciPy is not importable (CI installs it)."""
import itertools
import unittest
import warnings

import numerical

try:
    import numpy as np
    import scipy.linalg as sl
except ImportError:
    sl = None

STRUCTURES = ("gen", "general", "diagonal", "tridiagonal", "banded", "upper triangular", "lower triangular", "sym",
              "symmetric", "her", "hermitian", "pos", "positive definite")


@unittest.skipIf(sl is None, "SciPy not installed")
class AssumeA(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1101)

    def matrix(self, n, structure, complex_):
        a = self.rng.standard_normal((n, n)) + (1j * self.rng.standard_normal((n, n)) if complex_ else 0)
        if structure in ("pos", "positive definite"):
            a = a @ a.conj().T + n * np.eye(n)
        if structure == "banded":
            a = np.triu(np.tril(a, 2), -1)  # two below, one above
        if structure == "diagonal":
            a = a + 5 * np.eye(n)
        return a

    def test_every_structure_as_scipy(self):
        for n, structure, complex_, lower, shape, transposed in itertools.product(
                (2, 5, 30), STRUCTURES, (False, True), (False, True), ("vector", "matrix"), (False, True)):
            if transposed and complex_:
                continue
            a = self.matrix(n, structure, complex_)
            b = self.rng.standard_normal((n,) if shape == "vector" else (n, 3))
            with self.subTest(n=n, structure=structure, complex=complex_, lower=lower, shape=shape, transposed=transposed):
                expected = sl.solve(a, b, assume_a=structure, lower=lower, transposed=transposed)
                got = np.asarray(numerical.solve(a, b, assume_a=structure, lower=lower, transposed=transposed))
                self.assertEqual(got.shape, expected.shape)
                np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)

    def test_one_triangle_read(self):
        upper = np.array([[4.0, 1.0, 2.0], [99.0, 5.0, 1.0], [99.0, 99.0, 6.0]])  # 99s below: never read
        b = np.array([1.0, 2.0, 3.0])
        for structure in ("sym", "pos"):
            np.testing.assert_allclose(np.asarray(numerical.solve(upper, b, assume_a=structure)),
                                       sl.solve(upper, b, assume_a=structure), rtol=1e-13)
            np.testing.assert_allclose(np.asarray(numerical.solve(upper.T, b, assume_a=structure, lower=True)),
                                       sl.solve(upper.T, b, assume_a=structure, lower=True), rtol=1e-13)
        # a Hermitian matrix's diagonal taken real (from order 2 as SciPy's; its 1 × 1 one divides by the complex entry)
        her = np.array([[2 + 5j, 1j], [99.0, 3.0]])
        np.testing.assert_allclose(np.asarray(numerical.solve(her, b[:2], assume_a="her")), sl.solve(her, b[:2], assume_a="her"))
        self.assertEqual(complex(numerical.solve([[2 + 5j]], [1.0], assume_a="her")[0]), 0.5)

    def test_stacks(self):
        a = np.stack([self.matrix(4, "pos", False) for _ in range(3)])
        for b in (self.rng.standard_normal(4), self.rng.standard_normal((4, 2)), self.rng.standard_normal((3, 4, 2))):
            for structure in ("pos", "sym", "diagonal", "upper triangular", "tridiagonal"):
                with self.subTest(b=b.shape, structure=structure):
                    np.testing.assert_allclose(np.asarray(numerical.solve(a, b, assume_a=structure)),
                                               sl.solve(a, b, assume_a=structure), rtol=1e-10)
        for b in (np.ones((3, 4)), np.ones((2, 4, 1))):  # (n, nrhs) with n = 3, or another count of systems: as SciPy
            with self.subTest(b=b.shape), self.assertRaises(ValueError):
                sl.solve(a, b, assume_a="pos")
            with self.subTest(b=b.shape), self.assertRaises(ValueError):  # LinAlgError, a ValueError
                numerical.solve(a, b, assume_a="pos")

    def test_the_default_is_numpy_s(self):
        a, b = self.matrix(6, "gen", False), self.rng.standard_normal(6)
        self.assertEqual(np.asarray(numerical.solve(a, b)).tobytes(), np.asarray(numerical.solve(a, b, assume_a=None)).tobytes())

    def test_refusals(self):
        b = np.ones(2)
        for a, structure in ((np.zeros((2, 2)), "gen"), (np.zeros((2, 2)), "sym"), (np.zeros((2, 2)), "tridiagonal"),
                             (np.array([[2.0, 0.0], [0.0, 0.0]]), "diagonal"), (np.array([[1.0, 2.0], [2.0, 1.0]]), "pos"),
                             (np.zeros((2, 2)), "banded"), (np.array([[1.0, 0.0], [5.0, 0.0]]), "lower triangular"),
                             (np.array([[-2.0]]), "pos")):  # order 1 as every order, where SciPy's divides (scipy#26580)
            with self.subTest(structure=structure), self.assertRaises(numerical.LinAlgError):
                numerical.solve(a, b[:a.shape[0]], assume_a=structure)
        with self.assertRaisesRegex(ValueError, "not a recognized matrix structure"):
            numerical.solve(np.eye(2), b, assume_a="nope")
        with self.assertRaisesRegex(ValueError, "not a recognized matrix structure"):
            numerical.solve(np.eye(2), b, assume_a=3)
        with self.assertRaises(NotImplementedError):
            numerical.solve(np.eye(2) * 1j, b, transposed=True)
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve(np.ones((2, 3)), b, assume_a="pos")


@unittest.skipIf(sl is None, "SciPy not installed")
class IllConditioned(unittest.TestCase):
    """The LinAlgWarning SciPy's solve gives a system whose reciprocal condition number, estimated from the
    factorisation, falls under half the machine epsilon: given where SciPy's is, for every structure, and pointed at
    the caller."""

    @staticmethod
    def warned(solve, *args, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solve(*args, **kwargs)
        return [w for w in caught if w.category.__name__ == "LinAlgWarning"]

    def systems(self):
        """(a, structure, lower) of every structure, each well conditioned and ill conditioned (κ₁ about 10²⁰)."""
        n = 14
        hilbert = sl.hilbert(n)
        graded = np.diag(10.0 ** (-2.0 * np.arange(n)))  # 1 down to 10⁻²⁶
        upper = np.eye(n) - 100.0 * np.triu(np.ones((n, n)), 1)  # an inverse growing as 101ⁿ⁻¹
        band = np.diag(np.full(n, 1e-18)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 2), -2)
        tridiagonal = graded + np.diag(np.full(n - 1, 1e-30), 1) + np.diag(np.full(n - 1, 1e-30), -1)
        rng = np.random.default_rng(31)
        well = rng.standard_normal((n, n)) + n * np.eye(n)
        for a, structure in ((hilbert, "gen"), (hilbert, "sym"), (hilbert + 0j, "her"), (graded, "pos"),
                             (graded, "diagonal"), (upper, "upper triangular"), (upper.T, "lower triangular"),
                             (band, "banded"), (tridiagonal, "tridiagonal")):
            yield a, structure, True
        for structure in ("gen", "sym", "pos", "diagonal", "upper triangular", "lower triangular", "banded",
                          "tridiagonal"):
            a = well @ well.T if structure in ("sym", "pos") else well
            if structure == "diagonal":
                a = np.diag(np.diag(a))
            if structure in ("upper triangular", "banded", "tridiagonal"):
                a = np.triu(a) if structure == "upper triangular" else np.triu(np.tril(a, 1 if structure == "tridiagonal" else 2), -1)
            if structure == "lower triangular":
                a = np.tril(a)
            yield a, structure, False

    def test_warns_where_scipy_warns(self):
        for a, structure, ill in self.systems():
            for transposed in (False, True):
                if transposed and np.iscomplexobj(a):
                    continue
                b = np.ones(a.shape[0])
                with self.subTest(structure=structure, ill=ill, transposed=transposed):
                    expected = self.warned(sl.solve, a, b, assume_a=structure, transposed=transposed)
                    got = self.warned(numerical.solve, a, b, assume_a=structure, transposed=transposed)
                    self.assertEqual(bool(got), bool(expected))
                    self.assertEqual(bool(got), ill)
                    if got:
                        self.assertEqual(got[0].filename, __file__)  # the caller's line, as SciPy's stacklevel
                        self.assertIn("Ill-conditioned matrix (rcond=", str(got[0].message))
                        self.assertIsInstance(got[0].message, numerical.LinAlgWarning)

    def test_the_estimate_is_gecon_s(self):
        """The reciprocal condition number LAPACK's gecon, trcon and pocon give, to rounding."""
        from scipy.linalg import lapack
        rng = np.random.default_rng(7)
        for n in (1, 4, 40):
            a = rng.standard_normal((n, n))
            lu, _, _ = lapack.dgetrf(a)
            _, rcond = numerical._numerical.solve_conditioned(numerical.asarray(a), numerical.asarray(np.ones(n)))
            self.assertAlmostEqual(rcond / lapack.dgecon(lu, np.abs(a).sum(0).max())[0], 1.0, places=9)
            u = np.triu(a) + n * np.eye(n)
            self.assertAlmostEqual(numerical._numerical.factor_condition(numerical.asarray(u), numerical.asarray(u), 1, False)
                                   / lapack.dtrcon(u, uplo="U")[0], 1.0, places=9)
            s = a @ a.T + np.eye(n)
            self.assertAlmostEqual(numerical._numerical.factor_condition(numerical.asarray(np.linalg.cholesky(s)),
                                                                         numerical.asarray(s), 0, True)
                                   / lapack.dpocon(lapack.dpotrf(s)[0], np.abs(s).sum(0).max())[0], 1.0, places=9)

    def test_single_precision_s_epsilon(self):
        """A float32 system against float32's epsilon: κ₁ of 10⁸ warns there, not in float64 (as SciPy's)."""
        a = np.diag([1.0, 1e-8, 1.0])
        b = np.ones(3)
        for dtype in ("float32", "float64"):
            for structure in ("gen", "diagonal", "upper triangular"):
                with self.subTest(dtype=dtype, structure=structure):
                    expected = self.warned(sl.solve, a.astype(dtype), b.astype(dtype), assume_a=structure)
                    got = self.warned(numerical.solve, a.astype(dtype), b.astype(dtype), assume_a=structure)
                    self.assertEqual(bool(got), bool(expected))
                    self.assertEqual(bool(got), dtype == "float32")

    def test_each_system_of_a_stack(self):
        a = np.stack([np.eye(3), np.diag([1.0, 1e-20, 1.0]), np.eye(3)])
        got = self.warned(numerical.solve, a, np.ones(3), assume_a="gen")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].filename, __file__)

    def test_a_reciprocal_condition_number_of_zero_warns(self):
        """An estimate that underflows to 0 warns and answers, as SciPy 1.17's: the matrix is not singular (a diagonal's
        answer exact here)."""
        a = np.diag([1e300, 1e-300])
        for structure in ("gen", "diagonal", "upper triangular", "pos", "sym", "tridiagonal", "banded"):
            with self.subTest(structure=structure):
                got = self.warned(numerical.solve, a, np.ones(2), assume_a=structure)
                self.assertEqual(len(got), 1)
                np.testing.assert_allclose(np.asarray(numerical.solve(a, np.ones(2), assume_a=structure)), [1e-300, 1e300])

    def test_large_orders(self):
        """From order 2000 a well-conditioned system goes in mixed precision, its estimate from the float32 factors (no
        warning); an ill-conditioned one falls back to the double factorisation, and warns."""
        n = 2000
        rng = np.random.default_rng(2000)
        well = rng.standard_normal((n, n)) + np.sqrt(n) * np.eye(n)
        b = np.ones(n)
        self.assertEqual(self.warned(numerical.solve, well, b, assume_a="gen"), [])
        ill = well.copy()
        ill[:, -1] = ill[:, 0] * (1 + 1e-17)  # two columns equal to rounding
        ill[0, -1] += 1e-13
        self.assertEqual(bool(self.warned(numerical.solve, ill, b, assume_a="gen")),
                         bool(self.warned(sl.solve, ill, b, assume_a="gen")))
        self.assertTrue(self.warned(numerical.solve, ill, b, assume_a="gen"))

    def test_no_warning_without_assume_a(self):
        """NumPy's solve, the default, warns of nothing (as NumPy's)."""
        self.assertEqual(self.warned(numerical.solve, sl.hilbert(14), np.ones(14)), [])


if __name__ == "__main__":
    unittest.main()
