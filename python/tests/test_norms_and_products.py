"""Norms, condition numbers, powers and products as numpy.linalg gives them: norm, vector_norm, matrix_norm, cond,
svdvals, matrix_power, multi_dot, trace, outer, cross, tensordot, vecdot, conj and svd(hermitian=True), against NumPy
on every order, axis and stack, in the four dtypes (the answer real or complex as NumPy's, of the input's precision).
The 2-norm does not overflow where NumPy's does. Skipped when NumPy is not importable.
"""
import math
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy is not importable")
class NormsAndProducts(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)
        self.a = self.rng.standard_normal((4, 5))
        self.stack = self.rng.standard_normal((2, 3, 4, 4))
        self.complex = self.rng.standard_normal((3, 4)) + 1j * self.rng.standard_normal((3, 4))

    def close(self, got, want, tolerance=1e-12):
        got = np.asarray(got) if isinstance(got, numerical.Tensor) else got
        self.assertEqual(np.shape(got), np.shape(want))
        self.assertTrue(np.allclose(got, want, rtol=tolerance, atol=tolerance), f"{got} against {want}")

    def test_norms_take_numpys_orders_axes_and_keepdims(self):
        for order in (None, "fro", "nuc", 1, -1, 2, -2, math.inf, -math.inf):
            self.close(numerical.norm(self.a, order), np.linalg.norm(self.a, order))
            self.close(numerical.norm(self.complex, order), np.linalg.norm(self.complex, order))
        for order in (None, 1, 2, 3, 0, 0.5, -1, math.inf, -math.inf):
            self.close(numerical.norm(self.a[0], order), np.linalg.norm(self.a[0], order))
        self.close(numerical.norm(self.a, axis=0), np.linalg.norm(self.a, axis=0))
        self.close(numerical.norm(self.a, axis=(0, 1), keepdims=True), np.linalg.norm(self.a, axis=(0, 1), keepdims=True))
        self.close(numerical.norm(self.stack, "nuc", axis=(-2, -1)), np.linalg.norm(self.stack, "nuc", axis=(-2, -1)))
        self.close(numerical.norm(self.stack, 1, axis=(1, 3)), np.linalg.norm(self.stack, 1, axis=(1, 3)))
        self.close(numerical.vector_norm(self.stack, axis=(0, 2), keepdims=True),
                   np.linalg.vector_norm(self.stack, axis=(0, 2), keepdims=True))
        for order in ("fro", "nuc", 1, -1, 2, -2, math.inf, -math.inf):
            self.close(numerical.matrix_norm(self.stack, ord=order), np.linalg.matrix_norm(self.stack, ord=order))
        self.close(numerical.matrix_norm(self.stack, ord=2, keepdims=True),
                   np.linalg.matrix_norm(self.stack, ord=2, keepdims=True))
        self.assertIsInstance(numerical.norm(self.a), float)
        with self.assertRaises(ValueError):
            numerical.vector_norm(self.a, ord="fro")
        with self.assertRaises(ValueError):
            numerical.matrix_norm(self.a, ord=3)

    def test_the_2_norm_does_not_overflow_and_keeps_the_precision(self):
        self.assertAlmostEqual(numerical.vector_norm([1e200, 1e200]) / 1e200, math.sqrt(2), places=14)
        self.assertAlmostEqual(numerical.norm(np.full((3, 3), 1e-200)) / 1e-200, 3.0, places=14)
        self.assertEqual(numerical.vector_norm(np.zeros(3)), 0.0)
        self.assertTrue(math.isnan(numerical.vector_norm([1.0, math.nan])))
        self.assertEqual(numerical.vector_norm([1.0, math.inf]), math.inf)
        single = numerical.vector_norm(self.a.astype(np.float32), axis=1)
        self.assertEqual(single.dtype, "float32")
        self.assertEqual(numerical.vector_norm(self.complex.astype(np.complex64), axis=0).dtype, "float32")
        self.close(single, np.linalg.vector_norm(self.a.astype(np.float32), axis=1), 1e-6)

    def test_the_2_norm_sums_squares_where_none_is_lost_and_scales_elsewhere(self):
        # Squares summed as they are where their sum lies between 2^-900 and the largest double, scaled by the largest
        # magnitude past either bound: either way the norm math.hypot gives, row by row, each row taking its own way.
        rows = np.array([[3.0, 4.0, 12.0],                       # 13, summed
                         [2.0 ** -449, 2.0 ** -449, 0.0],         # a sum of squares 2^-897: summed
                         [2.0 ** -460, 2.0 ** -460, 2.0 ** -460],  # 3 · 2^-920: scaled
                         [1e300, 1e300, 1.0]])                    # past the largest double: scaled
        got = np.asarray(numerical.vector_norm(rows, axis=1))
        for row, value in zip(rows, got):
            self.assertAlmostEqual(value / math.hypot(*row), 1.0, places=14)
        for dtype in (np.float32, np.complex128, np.complex64):  # the parts' squares, in double
            values = (self.complex if np.dtype(dtype).kind == "c" else self.a).astype(dtype)
            self.close(numerical.norm(values), np.linalg.norm(values.astype(np.complex128)), 1e-6)

    def test_condition_numbers_and_singular_values(self):
        matrix = self.stack[0, 0]
        for p in (None, 1, -1, 2, -2, math.inf, -math.inf, "fro", "nuc"):
            self.close(numerical.cond(matrix, p), np.linalg.cond(matrix, p), 1e-9)
        self.close(numerical.cond(self.stack), np.linalg.cond(self.stack), 1e-9)
        self.close(numerical.cond(self.stack, 1), np.linalg.cond(self.stack, 1), 1e-9)
        for singular in (np.zeros((2, 2)), np.diag([1.0, 0.0])):  # infinite (or 0 for -2 when only one is zero)
            for p in (None, -2, 1, "fro"):
                self.assertEqual(numerical.cond(singular, p), np.linalg.cond(singular, p))
        self.close(numerical.cond(np.zeros((2, 2, 2))), np.linalg.cond(np.zeros((2, 2, 2))))
        self.close(numerical.svdvals(self.stack), np.linalg.svdvals(self.stack))

    def test_powers_and_chained_products(self):
        for n in (0, 1, 2, 5, -3):
            self.close(numerical.matrix_power(self.stack, n), np.linalg.matrix_power(self.stack, n), 1e-9)
            self.close(numerical.matrix_power(self.stack[0, 0], n), np.linalg.matrix_power(self.stack[0, 0], n), 1e-9)
        self.assertEqual(numerical.matrix_power(self.complex[:3, :3].astype(np.complex64), 2).dtype, "complex64")
        matrices = [self.rng.standard_normal(shape) for shape in ((10, 30), (30, 5), (5, 60), (60, 4))]
        self.close(numerical.multi_dot(matrices), np.linalg.multi_dot(matrices), 1e-10)
        first = [matrices[0][0]] + matrices[1:]
        self.close(numerical.multi_dot(first), np.linalg.multi_dot(first), 1e-10)
        both = [matrices[0][0]] + matrices[1:-1] + [matrices[-1][:, 0]]
        self.close(numerical.multi_dot(both), np.linalg.multi_dot(both), 1e-10)
        self.assertEqual(numerical._chain_order([(10, 30), (30, 5), (5, 60)])[0][2], 1)  # (A B) C: 4500 + 3000
        with self.assertRaises(numerical.LinAlgError):
            numerical.matrix_power(self.a, 2)

    def test_traces_outer_cross_tensordot_vecdot(self):
        for offset in (0, 1, -1, 3, -5):
            self.close(numerical.trace(self.stack, offset=offset), np.linalg.trace(self.stack, offset=offset))
        self.close(numerical.trace(self.a), np.linalg.trace(self.a))
        self.assertEqual(numerical.trace(self.a, dtype="complex128"), complex(np.trace(self.a)))
        self.close(numerical.outer(self.a[0], self.a[1]), np.linalg.outer(self.a[0], self.a[1]))
        v, w = self.rng.standard_normal((4, 3)), self.rng.standard_normal(3)
        self.close(numerical.cross(v, w), np.linalg.cross(v, w))
        self.close(numerical.cross(v.T, v.T[:, ::-1].copy(), axis=0), np.linalg.cross(v.T, v.T[:, ::-1], axis=0))
        t1, t2 = self.rng.standard_normal((3, 4, 5)), self.rng.standard_normal((4, 5, 6))
        self.close(numerical.tensordot(t1, t2), np.linalg.tensordot(t1, t2))
        self.close(numerical.tensordot(t1, t2, axes=([1], [0])), np.linalg.tensordot(t1, t2, axes=([1], [0])))
        self.close(numerical.tensordot(t1, t1, axes=3), np.linalg.tensordot(t1, t1, axes=3))
        self.close(numerical.vecdot(self.complex, self.complex[0]), np.linalg.vecdot(self.complex, self.complex[0]))
        self.close(numerical.vecdot(self.complex, self.complex, axis=0), np.linalg.vecdot(self.complex, self.complex, axis=0))
        self.close(numerical.conj(self.complex), np.conj(self.complex))
        self.assertIs(numerical.conjugate, numerical.conj)
        with self.assertRaises(ValueError):
            numerical.cross(np.ones(2), np.ones(2))
        with self.assertRaises(ValueError):
            numerical.outer(self.a, self.a)

    def test_a_hermitian_svd_goes_through_the_eigendecomposition(self):
        h = self.complex[:, :3] @ self.complex[:, :3].conj().T + np.diag([0.0, -9.0, 4.0])
        u, s, vh = numerical.svd(h, hermitian=True)
        self.close(s, np.linalg.svd(h, hermitian=True, compute_uv=False))
        self.close(np.asarray(u) * np.asarray(s) @ np.asarray(vh), h, 1e-10)
        self.close(np.asarray(u).conj().T @ np.asarray(u), np.eye(3), 1e-12)
        stack = self.stack + np.swapaxes(self.stack, -1, -2)
        self.close(numerical.svd(stack, hermitian=True, compute_uv=False),
                   np.linalg.svd(stack, hermitian=True, compute_uv=False))
        single = numerical.svd(stack.astype(np.float32), hermitian=True)
        self.assertEqual([part.dtype for part in single], ["float32"] * 3)
        with self.assertRaises(numerical.LinAlgError):
            numerical.svd(self.a, hermitian=True)


if __name__ == "__main__":
    unittest.main()
