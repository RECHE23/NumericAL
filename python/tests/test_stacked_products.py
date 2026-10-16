"""Products of stacks as NumPy computes them: ``a @ b`` and ``matmul`` broadcast the leading axes of stacks of matrices
(an extent of 1 repeated, a vector first a row and second a column), promote mixed dtypes, and refuse what NumPy
refuses; ``einsum`` reads an ellipsis as NumPy does, broadcast included. Skipped when NumPy is not importable.
"""
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy is not importable")
class StackedProducts(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def operand(self, shape, dtype="float64"):
        x = self.rng.standard_normal(shape)
        if dtype.startswith("complex"):
            x = x + 1j * self.rng.standard_normal(shape)
        return x.astype(dtype)

    def same(self, got, want, tolerance=1e-12):
        got = np.asarray(got)
        self.assertEqual(got.shape, np.shape(want))
        self.assertTrue(np.allclose(got, want, rtol=tolerance, atol=tolerance), f"{got} against {want}")

    def test_matmul_broadcasts_the_leading_axes(self):
        shapes = [((3, 4), (4, 5)), ((2, 3, 4), (4, 5)), ((4,), (2, 4, 5)), ((2, 3, 4), (4,)), ((2, 1, 3, 4), (5, 4, 6)),
                  ((1, 3, 4), (2, 4, 6)), ((0, 3, 4), (4, 2)), ((2, 3, 0), (2, 0, 5)), ((7, 1, 2, 3), (1, 6, 3, 2))]
        for sa, sb in shapes:
            for dtype in ("float64", "complex128", "float32", "complex64"):
                a, b = self.operand(sa, dtype), self.operand(sb, dtype)
                got = numerical.asarray(a) @ numerical.asarray(b)
                self.assertEqual(got.dtype, dtype)
                self.same(got, a @ b, 1e-5 if dtype in ("float32", "complex64") else 1e-12)
        self.assertEqual(numerical.matmul([1.0, 2.0], [3.0, 4.0]), 11.0)  # two vectors: a number

    def test_mixed_dtypes_promote_as_numpy(self):
        a, b = self.operand((2, 3, 4), "float32"), self.operand((4, 2), "complex128")
        got = numerical.matmul(a, b)
        self.assertEqual(got.dtype, "complex128")
        self.same(got, a @ b, 1e-6)
        self.assertEqual((numerical.asarray(self.operand((2, 2), "float32")) @ self.operand((2,), "complex64")).dtype, "complex64")

    def test_what_numpy_refuses(self):
        for sa, sb in (((2, 3, 4), (3, 4, 5)), ((2, 3, 4), (5, 4, 6)), ((), (3, 3))):
            with self.assertRaises(ValueError):
                numerical.matmul(np.ones(sa), np.ones(sb))

    def test_a_stack_answers_as_its_matrices_alone(self):
        a, b = self.operand((3, 5, 6)), self.operand((6, 4))
        stacked = np.asarray(numerical.matmul(a, b))
        for i in range(3):
            self.assertTrue(np.array_equal(stacked[i], np.asarray(numerical.matmul(a[i], b))))

    def test_einsum_reads_an_ellipsis(self):
        cases = [("...ij,...jk->...ik", [(2, 3, 4, 5), (2, 3, 5, 6)]), ("...ij,...jk->...ik", [(2, 1, 4, 5), (3, 5, 6)]),
                 ("...ij,jk", [(2, 3, 4), (4, 5)]), ("...i->...", [(2, 3, 4)]), ("i...->...", [(4, 2, 3)]),
                 ("...ii->...i", [(2, 3, 3)]), ("ij...,jk...->ik...", [(2, 3, 4), (3, 5, 4)]), ("...,...", [(2, 3), (3,)]),
                 ("a...b,b...->a...", [(2, 4, 3), (3, 4)]), ("i...j,j->i...", [(2, 1, 3), (3,)]),
                 ("ij,...jk->...ik", [(2, 3), (4, 3, 5)]), ("i,...->...i", [(3,), (2, 2)]), ("...ij->ij", [(2, 3)])]
        for spec, shapes in cases:
            for dtype in ("float64", "complex128"):
                operands = [self.operand(s, dtype) for s in shapes]
                self.same(numerical.einsum(spec, *[numerical.asarray(o) for o in operands]), np.einsum(spec, *operands))
        for spec, shapes in (("...ij,...jk->...ik", [(2, 3, 4, 5), (4, 3, 5, 6)]), ("..i", [(2,)]), ("...i...", [(2, 3)]),
                             ("...i->i", [(2, 3, 4)])):
            with self.assertRaises((ValueError, numerical.error)):
                numerical.einsum(spec, *[numerical.asarray(np.ones(s)) for s in shapes])


if __name__ == "__main__":
    unittest.main()
