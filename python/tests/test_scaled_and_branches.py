"""A fast path held to its general path's safeguards, and branches and edges as NumPy and SciPy take them: the continuous
Riccati equation's Hamiltonian path balanced; a negative fractional power of a Hermitian matrix on the principal branch;
the norms of an empty matrix and the condition numbers of a float32 stack as NumPy 2.5's; the pencils of types 2 and 3."""

import unittest

import numerical

try:
    import numpy as np
except ImportError:  # pragma: no cover - the CI images install NumPy
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class ScaledAndBranches(unittest.TestCase):
    def test_the_hamiltonian_path_is_balanced(self):
        a, b, r = np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]]), np.eye(1)
        x12 = np.sqrt(1e10)
        x22 = np.sqrt(1e10 + 2 * x12)
        exact = np.array([[x12 * x22, x12], [x12, x22]])
        x = np.asarray(numerical.solve_continuous_are(a, b, 1e10 * np.eye(2), r))
        self.assertLess(np.linalg.norm(x - exact) / np.linalg.norm(exact), 1e-14)
        rng = np.random.default_rng(5)
        failed = 0
        for _ in range(60):
            n = int(rng.choice([2, 3, 5, 8]))
            m = int(rng.integers(1, n + 1))
            a, b = rng.standard_normal((n, n)), rng.standard_normal((n, m))
            g = rng.standard_normal((n, n))
            q = g @ g.T * 10.0 ** rng.uniform(5, 10)
            g = rng.standard_normal((m, m))
            r = g @ g.T + np.eye(m)
            x = np.asarray(numerical.solve_continuous_are(a, b, q, r))
            residual = a.T @ x + x @ a - x @ b @ np.linalg.solve(r, b.T @ x) + q
            failed += not np.linalg.norm(residual) / max(np.linalg.norm(q), np.linalg.norm(x) * np.linalg.norm(a)) < 1e-8
        self.assertEqual(failed, 0)

    def test_a_negative_power_of_a_hermitian_matrix_is_principal(self):
        rng = np.random.default_rng(31)
        u = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))[0]
        h = (u * np.array([-0.5, -4.75, -9.0])) @ u.conj().T
        h = (h + h.conj().T) / 2
        w, v = np.linalg.eigh(h)
        for p in (0.5, -0.5, -1.5, 0.25, -0.75):
            with self.subTest(p=p):
                truth = (v * w.astype(complex) ** p) @ v.conj().T
                got = np.asarray(numerical.fractional_matrix_power(h, p))
                self.assertLess(np.linalg.norm(got - truth) / np.linalg.norm(truth), 1e-13)

    def test_the_norms_of_an_empty_matrix_are_numpy_2_5s(self):
        for shape in ((3, 0), (0, 3), (0, 0)):
            for order in (None, "fro", "nuc", 1, 2, np.inf):
                with self.subTest(shape=shape, order=order):
                    self.assertEqual(float(numerical.linalg.norm(numerical.zeros(shape), order)), 0.0)
        for order in (-1, -np.inf):  # the smallest sum of no column (no row) has no value
            with self.assertRaises(ValueError):
                numerical.linalg.norm(numerical.zeros((3, 0) if order == -1 else (0, 3)), order)
        self.assertEqual(float(numerical.linalg.norm(numerical.zeros((0, 3)), -1)), 0.0)
        with self.assertRaises(ValueError):
            numerical.linalg.norm(numerical.zeros((3, 0)), -2)
        self.assertEqual(numerical.linalg.norm(numerical.zeros((2, 3, 0)), 2, axis=(1, 2)).tolist(), [0.0, 0.0])

    def test_the_condition_numbers_of_a_float32_stack_are_float32(self):
        s = np.random.default_rng(0).standard_normal((2, 4, 4)).astype(np.float32) + 3 * np.eye(4, dtype=np.float32)
        for p in (1, np.inf, "fro"):
            got = numerical.linalg.cond(numerical.asarray(s), p)
            want = np.linalg.cond(s, p)
            self.assertEqual(got.dtype, str(want.dtype))
            np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5)
        self.assertEqual(numerical.linalg.cond(numerical.asarray(s.astype(np.complex64)), 1).dtype, "float32")

    def test_the_pencils_of_types_2_and_3_are_scipys(self):
        try:
            import scipy.linalg as sl
        except ImportError:  # pragma: no cover - the CI images install SciPy
            self.skipTest("SciPy is not installed")
        rng = np.random.default_rng(7)
        for cplx in (False, True):
            g = rng.standard_normal((5, 5)) + (1j * rng.standard_normal((5, 5)) if cplx else 0)
            a = g + g.conj().T
            g = rng.standard_normal((5, 5)) + (1j * rng.standard_normal((5, 5)) if cplx else 0)
            b = g @ g.conj().T + 5 * np.eye(5)
            for kind in (2, 3):
                for lower in (True, False):
                    with self.subTest(cplx=cplx, kind=kind, lower=lower):
                        triangle = np.tril if lower else np.triu  # the other triangle never read
                        w, x = numerical.eigh(triangle(a), triangle(b), lower=lower, type=kind)
                        want = sl.eigh(a, b, type=kind, eigvals_only=True)
                        np.testing.assert_allclose(np.asarray(w), want, rtol=1e-12, atol=1e-12 * np.abs(want).max())
                        x = np.asarray(x)
                        product = a @ b @ x if kind == 2 else b @ a @ x
                        np.testing.assert_allclose(product, x * np.asarray(w), atol=1e-11 * np.abs(product).max())
                        gram = x.conj().T @ (b if kind == 2 else np.linalg.inv(b)) @ x
                        np.testing.assert_allclose(gram, np.eye(5), atol=1e-12)
                        np.testing.assert_allclose(np.asarray(numerical.eigvalsh(a, b, type=kind)), want, rtol=1e-12,
                                                   atol=1e-12 * np.abs(want).max())
                        pair = numerical.eigh(a, b, type=kind, subset_by_index=[1, 2])
                        np.testing.assert_allclose(np.asarray(pair[0]), want[1:3], rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            numerical.eigh(np.eye(2), np.eye(2), type=4)


if __name__ == "__main__":
    unittest.main()
