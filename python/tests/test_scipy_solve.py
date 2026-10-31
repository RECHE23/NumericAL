"""scipy.linalg.solve's arguments against SciPy's own: every structure assume_a names (by each of its names), the
triangle lower picks, transposed, vector and matrix right-hand sides, stacks; the refusals (a singular matrix, one not
positive definite, an unknown structure, a complex transposed system). Deterministic; skipped when SciPy is not
importable (CI installs it)."""
import itertools
import unittest

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


if __name__ == "__main__":
    unittest.main()
