"""The advanced indexes gathered in C (numerical._numerical._gather_index): a bool mask over a tensor's leading axes and
an integer array along its first, alone in the key, and take along the first axis or of the flattened tensor; and
concatenate of parts of one dtype joined in C (_concatenate_fast). NumPy's answers, and the general path's (its errors
included) for every other key and call."""

import unittest

import numerical

try:
    import numpy as np
except ImportError:  # pragma: no cover - the CI images install NumPy
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class FastIndexing(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12)

    def operands(self, shape):
        values = self.rng.standard_normal(shape)
        yield "float64", values
        yield "float32", values.astype(np.float32)
        yield "complex128", values + 1j * values[::-1]
        yield "bool", values > 0

    def assert_like_numpy(self, got, want):
        want = np.asarray(want)
        self.assertIsInstance(got, numerical.Tensor)
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_array_equal(np.asarray(got), want)

    def test_a_mask_over_the_leading_axes_gives_numpys_elements(self):
        for shape in ((7,), (4, 5), (3, 4, 2), (0, 3), (1, 1)):
            for name, values in self.operands(shape):
                tensor = numerical.asarray(values)
                for rank in range(1, len(shape) + 1):
                    mask = self.rng.random(shape[:rank]) < 0.5
                    for key in (mask, numerical.asarray(mask), np.zeros(shape[:rank], bool), np.ones(shape[:rank], bool)):
                        with self.subTest(dtype=name, shape=shape, rank=rank, key=type(key).__name__):
                            self.assert_like_numpy(tensor[key], values[np.asarray(key)])
        a = self.rng.standard_normal((6, 5))
        mask = self.rng.random((5, 6)) < 0.5
        for view, values in ((numerical.asarray(a).T, a.T), (numerical.asarray(a)[::-1], a[::-1])):  # views read in C order
            keep = mask if view.shape == mask.shape else mask.T
            self.assert_like_numpy(view[keep], values[keep])

    def test_an_integer_array_takes_numpys_rows(self):
        a = self.rng.standard_normal((5, 3, 2))
        tensor = numerical.asarray(a)
        for index in ([0, 4, 2, 2], [-1, -5, 0], [[0, 1], [4, 3]], [], np.array([3, 0], np.int32), np.array([1, -2]),
                      np.array([[2], [1]]), numerical.asarray(np.array([4, 0, 1]), "int64")):
            with self.subTest(index=index):
                want = a[np.asarray(index, dtype=np.int64) if isinstance(index, list) else np.asarray(index)]
                self.assert_like_numpy(tensor[index], want)
        for name, values in self.operands((6, 4)):
            with self.subTest(dtype=name):
                self.assert_like_numpy(numerical.asarray(values)[np.array([5, 0, 3])], values[[5, 0, 3]])
        picked = tensor[[1, 1]]
        self.assertFalse(np.shares_memory(np.asarray(picked), np.asarray(tensor)))  # a copy, as NumPy's

    def test_take_of_the_flattened_tensor_and_along_the_first_axis(self):
        a = self.rng.standard_normal((4, 6))
        tensor = numerical.asarray(a)
        for index in ([0, 23, 5], np.array([[1, 2], [-1, 7]]), np.array([], np.int64)):
            with self.subTest(index=index):
                self.assert_like_numpy(numerical.take(tensor, index), np.take(a, index))
                self.assert_like_numpy(numerical.take(a, index), np.take(a, index))
        self.assert_like_numpy(numerical.take(tensor, [3, 0], axis=0), np.take(a, [3, 0], axis=0))
        self.assert_like_numpy(numerical.take(tensor.T, [5, 1]), np.take(a.T, [5, 1]))

    def test_other_keys_and_errors_answer_as_before(self):
        # A mask of another shape, an index out of the axis, a float array, a rank-0 index and a key of several parts
        # go the general path: its answers, and NumPy's errors in its words.
        a = self.rng.standard_normal((4, 3))
        tensor = numerical.asarray(a)
        gather = numerical._numerical._gather_index
        for key in (np.ones(8, bool)[::2], np.ones((3, 4), bool), np.array(7), np.array([0.5]), 1.5, (slice(None), [0]),
                    np.array([0, 4]), [True, False]):
            self.assertIs(gather(tensor, key), NotImplemented)
        with self.assertRaisesRegex(IndexError, "index 4 is out of bounds for axis 0 with size 4"):
            tensor[np.array([0, 4])]
        with self.assertRaisesRegex(IndexError, "index -5 is out of bounds for axis 0 with size 4"):
            tensor[[-5]]
        with self.assertRaises(IndexError):
            tensor[np.ones((3, 4), bool)]
        with self.assertRaises(IndexError):
            numerical.take(tensor, [12])
        self.assert_like_numpy(tensor[np.ones(8, bool)[::2]], a[np.ones(8, bool)[::2]])  # a strided mask, as before
        self.assert_like_numpy(tensor[[True, False, True, True]], a[[True, False, True, True]])
        self.assert_like_numpy(tensor[:, [2, 0]], a[:, [2, 0]])
        self.assertEqual(float(tensor[np.int64(1), 2]), a[1, 2])


@unittest.skipIf(np is None, "NumPy is not installed")
class FastConcatenate(unittest.TestCase):
    # concatenate of Tensors and NumPy arrays of one dtype, read where they lie and joined in C
    # (numerical._numerical._concatenate_fast); NumPy's answers, and the general path's for every other call.
    def setUp(self):
        self.rng = np.random.default_rng(13)

    def assert_like_numpy(self, got, want):
        want = np.asarray(want)
        self.assertIsInstance(got, numerical.Tensor)
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(np.asarray(got).tobytes(), want.astype(np.asarray(got).dtype).tobytes())

    def test_parts_of_one_dtype_join_along_any_axis_as_numpys(self):
        for dtype in (np.float64, np.float32, np.complex128, np.complex64):
            for shapes, axis in ((((3, 4), (5, 4)), 0), (((3, 4), (3, 1), (3, 6)), 1), (((2, 3, 4), (2, 3, 1)), -1),
                                 (((2, 3, 4), (2, 0, 4)), 1), (((7,), (2,)), 0), (((1000, 1000), (1000, 1000)), 1)):
                parts = [self.rng.standard_normal(shape).astype(dtype) for shape in shapes]
                want = np.concatenate(parts, axis=axis)
                with self.subTest(dtype=np.dtype(dtype).name, shapes=shapes, axis=axis):
                    self.assert_like_numpy(numerical.concatenate(parts, axis=axis), want)
                    self.assert_like_numpy(numerical.concatenate([numerical.asarray(p) for p in parts], axis=axis), want)
                    self.assert_like_numpy(numerical.concatenate(tuple([numerical.asarray(parts[0])] + parts[1:]), axis), want)
        a = self.rng.standard_normal((4, 6))
        for parts in ([numerical.asarray(a).T, numerical.asarray(a).T], [np.asfortranarray(a), a[::-1]],
                      [np.asfortranarray(self.rng.standard_normal((400, 300))), self.rng.standard_normal((400, 300))]):
            arrays = [np.asarray(p) for p in parts]
            self.assert_like_numpy(numerical.concatenate(parts, axis=0), np.concatenate(arrays, axis=0))
        for values in (a > 0, np.arange(24).reshape(4, 6)):  # bool and int64 Tensors joined in C, the arrays converted
            tensor = numerical.asarray(values, "int64" if values.dtype.kind == "i" else None)
            self.assert_like_numpy(numerical.concatenate([tensor, tensor], axis=1), np.concatenate([values, values], axis=1))

    def test_other_calls_answer_as_before(self):
        fast = numerical._numerical._concatenate_fast
        a = self.rng.standard_normal((3, 4))
        single = a.astype(np.float32)
        for args in (([a, single], 0), ([a, a[:, :2]], 0), ([a, a], 2), ([a, a], -3), ([], 0), ([np.float64(1.0)], 0),
                     ([a > 0, a > 0], 0), ([a, [1.0, 2.0, 3.0, 4.0]], 0), (a, 0)):
            self.assertIs(fast(*args), NotImplemented)
        self.assert_like_numpy(numerical.concatenate([a, single]), np.concatenate([a, single]))  # promoted
        self.assert_like_numpy(numerical.concatenate([a, [[1.0, 2.0, 3.0, 4.0]]]), np.concatenate([a, [[1.0, 2.0, 3.0, 4.0]]]))
        self.assert_like_numpy(numerical.concatenate([a, a], axis=None), np.concatenate([a, a], axis=None))
        self.assert_like_numpy(numerical.concatenate([a, a], dtype="float32"), np.concatenate([a, a], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "dimensions except for the concatenation axis must match"):
            numerical.concatenate([a, a[:, :2]], axis=0)
        with self.assertRaises(numerical.AxisError):
            numerical.concatenate([a, a], axis=2)
        with self.assertRaisesRegex(ValueError, "zero-dimensional"):
            numerical.concatenate([numerical.asarray(1.0), numerical.asarray(2.0)])
        with self.assertRaisesRegex(ValueError, "at least one array"):
            numerical.concatenate([])


if __name__ == "__main__":
    unittest.main()
