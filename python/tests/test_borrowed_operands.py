"""Products of NumPy arrays read where they lie (no copy into a Tensor): the same bits as the Tensors' product, and the
converting path for any array that cannot be read so."""

import unittest

import numerical

try:
    import numpy as np
except ImportError:  # pragma: no cover - the CI images install NumPy
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class BorrowedOperands(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(5)

    def operand(self, shape, dtype):
        values = self.rng.standard_normal(shape)
        if np.dtype(dtype).kind == "c":
            values = values + 1j * self.rng.standard_normal(shape)
        return values.astype(dtype)

    def assert_same(self, got, want):
        if isinstance(want, numerical.Tensor):
            self.assertIsInstance(got, numerical.Tensor)
            self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape))
        self.assertTrue(np.array_equal(np.asarray(got), np.asarray(want)))

    def test_every_dtype_and_rank_gives_the_tensors_bits(self):
        for dtype in (np.float64, np.complex128, np.float32, np.complex64):
            for m, k, n in ((1, 1, 1), (3, 4, 5), (100, 37, 1), (1, 37, 100), (129, 200, 3), (5, 0, 4)):
                for left, right in (((m, k), (k, n)), ((m, k), (k,)), ((k,), (k, n)), ((k,), (k,))):
                    with self.subTest(dtype=np.dtype(dtype).name, left=left, right=right):
                        a, b = self.operand(left, dtype), self.operand(right, dtype)
                        want = numerical.asarray(a) @ numerical.asarray(b)
                        self.assert_same(numerical.matmul(a, b), want)
                        self.assert_same(numerical.matmul(numerical.asarray(a), b), want)
                        self.assert_same(numerical.matmul(a, numerical.asarray(b)), want)
                        self.assert_same(numerical.asarray(a) @ b, want)

    def test_arrays_that_cannot_be_read_in_place_are_converted(self):
        a, b = self.operand((6, 8), np.float64), self.operand((8, 6), np.float64)
        strided = a[:, ::2]
        swapped = a.astype(">f8")
        raw = np.zeros(6 * 8 * 8 + 1, np.uint8)[1:].view(np.float64).reshape(6, 8)  # one byte off a double's alignment
        raw[...] = a
        self.assertFalse(raw.flags.aligned)
        for left, right, want in ((strided, b[::2], a[:, ::2] @ b[::2]), (swapped, b, a @ b), (raw, b, a @ b),
                                  (a.astype(np.float32), b, a.astype(np.float32).astype(np.float64) @ b)):
            with self.subTest(left=left.dtype.str, contiguous=left.flags.c_contiguous, aligned=left.flags.aligned):
                np.testing.assert_allclose(np.asarray(numerical.matmul(left, right)), want, rtol=1e-12)
        with self.assertRaises(numerical.LinAlgError):
            numerical.matmul(a, a)

    def test_the_product_owns_its_elements(self):
        a, b = self.operand((4, 4), np.float64), self.operand((4, 4), np.float64)
        product = numerical.matmul(a, b)
        before = np.asarray(product).copy()
        a[...] = 0.0
        b[...] = 0.0
        self.assertTrue(np.array_equal(np.asarray(product), before))

    def same(self, got, want):
        if isinstance(want, tuple):
            self.assertEqual(len(got), len(want))
            for g, w in zip(got, want):
                self.same(g, w)
        elif isinstance(want, numerical.Tensor):
            self.assert_same(got, want)
        else:
            self.assertIs(type(got), type(want))
            self.assertEqual(got, want)

    def test_solve_det_slogdet_and_inverse_give_the_tensors_answers(self):
        for dtype in (np.float64, np.complex128, np.float32, np.complex64):
            for n in (1, 2, 7, 64):
                a = self.operand((n, n), dtype)
                with self.subTest(dtype=np.dtype(dtype).name, n=n):
                    tensor = numerical.asarray(a)
                    for f in (numerical.det, numerical.slogdet, numerical.inv):
                        self.same(f(a), f(tensor))
                    for rhs in ((n,), (n, 3)):
                        b = self.operand(rhs, dtype)
                        self.same(numerical.solve(a, b), numerical.solve(tensor, numerical.asarray(b)))
                        self.same(numerical.solve(a, numerical.asarray(b)), numerical.solve(tensor, numerical.asarray(b)))

    def test_decompositions_of_unreadable_or_wrong_operands_answer_as_the_tensors(self):
        def failure(f, *args):
            try:
                f(*args)
            except numerical.LinAlgError as caught:
                return str(caught)
            return None

        a = self.operand((5, 5), np.float64)
        singular, holed = np.ones((4, 4)), np.eye(3)
        holed[1, 1] = np.nan
        for f, args in ((numerical.inv, (singular,)), (numerical.det, (holed,)), (numerical.slogdet, (holed,)),
                        (numerical.inv, (np.ones((3, 4)),)), (numerical.solve, (a, np.ones(4))),
                        (numerical.solve, (singular, np.ones(4)))):
            with self.subTest(function=f.__name__, shape=args[0].shape):
                want = failure(f, *(numerical.asarray(x) for x in args))
                self.assertIsNotNone(want)
                self.assertEqual(failure(f, *args), want)
        strided = np.eye(6)[::2, ::2] * 2
        np.testing.assert_allclose(np.asarray(numerical.inv(strided)), np.eye(3) / 2)
        self.assertEqual(numerical.inv(np.stack([np.eye(3)] * 2)).shape, (2, 3, 3))
        stacked = self.operand((2, 5, 3), np.float64)  # a stack of right-hand sides, broadcast against one matrix
        self.same(numerical.solve(a, stacked), numerical.solve(numerical.asarray(a), numerical.asarray(stacked)))


if __name__ == "__main__":
    unittest.main()
