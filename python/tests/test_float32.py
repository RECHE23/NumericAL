"""float32 and complex64 kept as they are, as NumPy keeps them: read, stored, exchanged, promoted, computed and
returned in single precision. Element-wise arithmetic and square roots are IEEE operations, the same to the bit as
NumPy's; the decompositions compute in double precision and round their answers to the input's precision, so they are
as accurate as a single-precision answer can be, where NumPy's LAPACK computes in single precision; solve, inverse,
cholesky, lu_factor and lu_solve of a float32 matrix of order 500 or more compute in single precision, as NumPy's. Skipped when NumPy
is not importable (CI installs it).
"""
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None

EPS32 = float(np.finfo(np.float32).eps) if np is not None else 1.1920928955078125e-07
DTYPES = ("float32", "complex64", "float64", "complex128")


@unittest.skipIf(np is None, "NumPy is not importable")
class SinglePrecisionStorage(unittest.TestCase):
    def test_every_dtype_is_read_kept_and_exchanged(self):
        for dtype in DTYPES:
            a = (np.arange(12, dtype=dtype).reshape(3, 4) / 7) * (1 + (1j if dtype.startswith("complex") else 0))
            a = a.astype(dtype)
            t = numerical.asarray(a)
            self.assertEqual(t.dtype, dtype)
            for view in (np.asarray(t), np.from_dlpack(t)):
                self.assertEqual(view.dtype, np.dtype(dtype))
                self.assertTrue(np.array_equal(view, a))
            self.assertTrue(np.array_equal(np.asarray(numerical.asarray(a[:, ::-2])), a[:, ::-2]))  # strided
            self.assertTrue(np.array_equal(np.asarray(numerical.asarray(a.astype(a.dtype.newbyteorder()))), a))

    def test_asarray_converts_between_precisions_and_not_complex_to_real(self):
        single = numerical.asarray(np.array([1.5, 1e-40, 3.0e38], np.float32))
        wide = numerical.asarray(single, "float64")
        self.assertEqual(wide.dtype, "float64")
        self.assertEqual(np.asarray(wide).tolist(), np.array([1.5, 1e-40, 3.0e38], np.float32).astype(float).tolist())
        self.assertEqual(numerical.asarray(wide, "float32").dtype, "float32")
        self.assertEqual(numerical.asarray([0.1, 0.2], "float32").tolist(), np.float32([0.1, 0.2]).tolist())
        self.assertEqual(numerical.asarray(single, "complex64").dtype, "complex64")
        self.assertEqual(numerical.asarray(np.complex128([1 + 2j]), "complex64").dtype, "complex64")
        self.assertEqual(numerical.asarray(2.5, "float32").dtype, "float32")
        self.assertEqual(numerical.asarray(2j, "complex64").dtype, "complex64")
        for source in (np.complex64([1j]), numerical.asarray(np.complex64([1j])), 1j):
            for real in ("float32", "float64"):
                with self.assertRaises(numerical.error):
                    numerical.asarray(source, real)
        self.assertEqual(numerical.asarray(np.int16([1, 2])).dtype, "float64")  # integers and bools: float64, as before

    def test_constructors_take_the_four_dtypes(self):
        for dtype in DTYPES:
            self.assertEqual(numerical.zeros((2, 3), dtype).dtype, dtype)
            self.assertEqual(numerical.ones(2, dtype).dtype, dtype)
            self.assertEqual(numerical.eye(3, dtype).dtype, dtype)
            self.assertEqual(numerical.Tensor([1, 2], dtype=dtype).dtype, dtype)
        self.assertEqual(numerical.full(2, 1 + 1j, "complex64").tolist(), [1 + 1j, 1 + 1j])
        with self.assertRaises(TypeError):
            numerical.full(2, 1j, "float32")
        with self.assertRaises(ValueError):
            numerical.zeros(2, "float16")


@unittest.skipIf(np is None, "NumPy is not importable")
class Promotion(unittest.TestCase):
    """NumPy 2's: complex where one operand is, single precision only where both are; a Python number is weak."""

    def test_tensors_and_numbers_promote_as_numpy_promotes_them(self):
        operands = {dtype: np.ones(3, dtype) for dtype in DTYPES}
        for left, a in operands.items():
            for right, b in operands.items():
                with self.subTest(left=left, right=right):
                    got = numerical.asarray(a) + numerical.asarray(b)
                    self.assertEqual(got.dtype, (a + b).dtype.name)
            for number in (2, 2.0, 2j, True):
                with self.subTest(left=left, number=number):
                    self.assertEqual((numerical.asarray(a) * number).dtype, (a * number).dtype.name)
                    self.assertEqual((number - numerical.asarray(a)).dtype, (number - a).dtype.name)
            self.assertEqual((numerical.asarray(a) + [1.0, 2.0, 3.0]).dtype, (a + np.array([1.0, 2.0, 3.0])).dtype.name)

    def test_linalg_operands_promote_together(self):
        a = np.eye(3, dtype=np.float32) * 2
        self.assertEqual(numerical.solve(a, np.ones(3, np.float32)).dtype, "float32")
        self.assertEqual(numerical.solve(a, np.ones(3)).dtype, "float64")
        self.assertEqual(numerical.solve(a, np.ones(3, np.complex64)).dtype, "complex64")
        self.assertEqual(numerical.einsum("ij,j->i", a, np.ones(3, np.complex128)).dtype, "complex128")


@unittest.skipIf(np is None, "NumPy is not importable")
class SinglePrecisionArithmetic(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(32)
        self.x = rng.standard_normal((7, 9)).astype(np.float32)
        self.y = (rng.standard_normal((7, 9)) + 3).astype(np.float32)
        self.z = (rng.standard_normal((7, 9)) + 1j * rng.standard_normal((7, 9))).astype(np.complex64)

    def test_ieee_operations_are_numpys_to_the_bit(self):
        x, y = numerical.asarray(self.x), numerical.asarray(self.y)
        for got, want in ((x + y, self.x + self.y), (x - y, self.x - self.y), (x * y, self.x * self.y),
                          (x / y, self.x / self.y), (numerical.sqrt(y), np.sqrt(self.y)), (-x, -self.x),
                          (x * 0.1, self.x * np.float32(0.1)), (abs(x), np.abs(self.x))):
            self.assertEqual(got.dtype, "float32")
            self.assertTrue(np.array_equal(np.asarray(got), want))
        z = numerical.asarray(self.z)
        for got, want in ((z + z, self.z + self.z), (z - z, self.z - self.z)):
            self.assertEqual(got.dtype, "complex64")
            self.assertTrue(np.array_equal(np.asarray(got), want))

    def test_functions_and_reductions_keep_the_precision(self):
        x, y, z = numerical.asarray(self.x), numerical.asarray(self.y), numerical.asarray(self.z)
        for name in ("exp", "sin", "cos", "tanh"):
            got = getattr(numerical, name)(x)
            self.assertEqual(got.dtype, "float32")
            self.assertTrue(np.allclose(np.asarray(got), getattr(np, name)(self.x), rtol=4 * EPS32, atol=4 * EPS32))
        self.assertEqual(numerical.log(y).dtype, "float32")
        self.assertEqual(abs(z).dtype, "float32")
        self.assertEqual(numerical.exp(z).dtype, "complex64")
        self.assertEqual(numerical.sum(x, 0).dtype, "float32")
        self.assertTrue(np.allclose(np.asarray(numerical.sum(x, 1)), self.x.sum(1), rtol=1e-6, atol=1e-6))
        self.assertAlmostEqual(numerical.sum(x), float(self.x.sum()), delta=1e-5)
        self.assertTrue(np.array_equal(np.asarray(numerical.max(x, 0)), self.x.max(0)))
        self.assertEqual(numerical.min(x), float(self.x.min()))
        self.assertEqual(numerical.mean(z, 0).dtype, "complex64")
        with self.assertRaises(TypeError):
            numerical.max(z)

    def test_products_and_indexing_keep_the_precision(self):
        x = numerical.asarray(self.x)
        product = x @ x.T
        self.assertEqual(product.dtype, "float32")
        want = self.x.astype(float) @ self.x.T.astype(float)
        self.assertLess(np.max(np.abs(np.asarray(product) - want)), 32 * EPS32 * np.max(np.abs(want)))
        self.assertEqual(numerical.einsum("ij,ij->i", x, x).dtype, "float32")
        self.assertIsInstance(x[1, 2], float)
        self.assertEqual(x[1, 2], float(self.x[1, 2]))
        self.assertEqual(x[1:, ::2].dtype, "float32")
        self.assertEqual(x.reshape(63).dtype, "float32")
        self.assertTrue(numerical.array_equal(x, self.x))


@unittest.skipIf(np is None, "NumPy is not importable")
class SinglePrecisionLinearAlgebra(unittest.TestCase):
    """Computed in double and rounded: each answer within a few roundings of float32 of the double one."""

    def setUp(self):
        rng = np.random.default_rng(64)
        g = rng.standard_normal((6, 6))
        self.real = (g @ g.T + 6 * np.eye(6)).astype(np.float32)
        h = g + 1j * rng.standard_normal((6, 6))
        self.complex = (h @ h.conj().T + 6 * np.eye(6)).astype(np.complex64)
        self.tall = rng.standard_normal((9, 4)).astype(np.float32)

    def close(self, got, want, dtype):
        got = np.asarray(got)
        self.assertEqual(got.dtype, np.dtype(dtype))
        want = np.asarray(want)
        self.assertLessEqual(np.max(np.abs(got - want), initial=0.0), 2 * EPS32 * max(np.max(np.abs(want)), 1.0))

    def test_every_decomposition_answers_in_single_precision(self):
        for a, dtype, wide in ((self.real, "float32", np.float64), (self.complex, "complex64", np.complex128)):
            w = a.astype(wide)
            b = np.arange(1, 7).astype(dtype)
            self.close(numerical.solve(a, b), np.linalg.solve(w, b.astype(wide)), dtype)
            self.close(numerical.inverse(a), np.linalg.inv(w), dtype)
            self.close(numerical.cholesky(a), np.linalg.cholesky(w), dtype)
            values = np.linalg.eigvalsh(w)
            self.close(numerical.eigvalsh(a), values, "float32")
            got_values, vectors = numerical.eigh(a)
            self.close(got_values, values, "float32")
            self.assertEqual(vectors.dtype, dtype)
            self.close(numerical.svd(a, compute_uv=False), np.linalg.svd(w, compute_uv=False), "float32")
            u, s, vh = numerical.svd(a)
            self.assertEqual((u.dtype, s.dtype, vh.dtype), (dtype, "float32", dtype))
            q, r = numerical.qr(a)
            self.assertEqual((q.dtype, r.dtype), (dtype, dtype))
            self.close(numerical.pinv(a), np.linalg.pinv(w), dtype)
            self.assertAlmostEqual(abs(numerical.det(a)), abs(np.linalg.det(w)), delta=1e-6 * abs(np.linalg.det(w)))
        x, residuals, rank, s = numerical.lstsq(self.tall, np.ones(9, np.float32))
        want = np.linalg.lstsq(self.tall.astype(float), np.ones(9), rcond=None)
        self.close(x, want[0], "float32")
        self.assertEqual((residuals.dtype, rank, s.dtype), ("float32", 4, "float32"))

    def test_large_float32_factorisations_compute_in_single_precision(self):
        # From order 500, solve, inverse, cholesky, lu_factor and lu_solve factorise in float32, as NumPy's LAPACK: the
        # answers float32 and backward stable at that precision (residuals within 20 n ε of float32 and above double's).
        rng = np.random.default_rng(11)
        n = 500
        a = rng.standard_normal((n, n)).astype(np.float32)
        spd = (a @ a.T + n * np.eye(n)).astype(np.float32)
        b = rng.standard_normal((n, 3)).astype(np.float32)
        w, wb = a.astype(np.float64), b.astype(np.float64)

        def backward(x, matrix, rhs):
            x = np.asarray(x).astype(np.float64)
            return np.linalg.norm(matrix.astype(np.float64) @ x - rhs, 1) / (np.linalg.norm(matrix.astype(np.float64), 1) * np.linalg.norm(x, 1))

        x = numerical.solve(a, b)
        self.assertEqual(x.dtype, "float32")
        self.assertLess(backward(x, a, wb), 20 * n * EPS32)
        self.assertGreater(backward(x, a, wb), 1e3 * np.finfo(np.float64).eps)  # single precision, not double rounded
        inverse = numerical.inverse(a)
        self.assertEqual(inverse.dtype, "float32")
        self.assertLess(backward(inverse, a, np.eye(n)), 20 * n * EPS32)
        lower = np.asarray(numerical.cholesky(spd))
        self.assertEqual(lower.dtype, np.float32)
        self.assertLess(np.linalg.norm(lower.astype(np.float64) @ lower.T.astype(np.float64) - spd, 1) / np.linalg.norm(spd.astype(np.float64), 1), 20 * n * EPS32)
        lu, piv = numerical.lu_factor(a)
        self.assertEqual(lu.dtype, "float32")
        y = numerical.lu_solve((lu, piv), b)
        self.assertEqual(y.dtype, "float32")
        self.assertLess(backward(y, a, wb), 20 * n * EPS32)
        small = a[:499, :499]
        self.assertLess(backward(numerical.solve(small, b[:499]), small, wb[:499]), 4 * EPS32)  # below 500: double, rounded

    def test_det_and_slogdet_of_single_precision_are_the_double_answer_rounded(self):
        # The single-precision matrix's values in double, the answer rounded to single, to the bit (a plain number).
        # Compared as Python numbers: NumPy 2 would round a Python float to float32 to compare it with one.
        rng = np.random.default_rng(7)
        real = rng.standard_normal((6, 6)).astype(np.float32)
        for single, rounded, number in ((real, np.float32, float), ((real + 1j * real.T).astype(np.complex64), np.complex64, complex)):
            double = single.astype(np.complex128 if rounded is np.complex64 else np.float64)
            self.assertEqual(numerical.det(single), number(rounded(numerical.det(double))))
            sign, logarithm = numerical.slogdet(single)
            wide_sign, wide_logarithm = numerical.slogdet(double)
            self.assertEqual(sign, number(rounded(wide_sign)))
            self.assertEqual(logarithm, float(np.float32(wide_logarithm)))

    def test_a_rank_counts_float32_s_rounding_as_zero(self):
        # σ = 1, 1e-9: zero at float32's precision (NumPy's own float32 rank says 1), not at float64's.
        a = np.diag([1.0, 1e-9]).astype(np.float32)
        self.assertEqual(numerical.matrix_rank(a), np.linalg.matrix_rank(a))
        self.assertEqual(numerical.matrix_rank(a), 1)
        self.assertEqual(numerical.matrix_rank(a.astype(float)), 2)
        self.assertEqual(numerical.lstsq(a, np.ones(2, np.float32))[2], 1)
        self.assertEqual(numerical.lstsq(a, np.ones(2, np.float32), rcond=-1)[2], 1)  # LAPACK's: float32's epsilon
        self.assertTrue(np.array_equal(np.asarray(numerical.pinv(a)), np.diag([1.0, 0.0]).astype(np.float32)))
        self.assertEqual(numerical.matrix_rank(a, tol=1e-12), 2)  # a tolerance given is taken as it is


if __name__ == "__main__":
    unittest.main()
