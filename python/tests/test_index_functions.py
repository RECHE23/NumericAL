"""argmax, argmin, argsort, nonzero, flatnonzero, argwhere, count_nonzero and searchsorted against NumPy's answers, in
int64; indexing with an int64 Tensor. Deterministic; the NumPy comparisons skipped without NumPy."""
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


class IndexFunctions(unittest.TestCase):
    def test_without_numpy(self):
        t = numerical.asarray([[3.0, 1.0, 3.0], [2.0, float("nan"), 5.0]])
        self.assertEqual(numerical.argmax(t), 4)  # the first NaN
        self.assertIs(type(numerical.argmax(t)), int)
        self.assertEqual(numerical.argmax(t, axis=0).tolist(), [0, 1, 1])
        self.assertEqual(numerical.argmax(t, axis=0).dtype, "int64")
        self.assertEqual(numerical.argmin(t, axis=1, keepdims=True).shape, (2, 1))
        self.assertEqual(numerical.argmax(t, keepdims=True).shape, (1, 1))
        self.assertEqual(numerical.argsort(t, axis=1).tolist(), [[1, 0, 2], [0, 2, 1]])
        self.assertEqual(numerical.argsort(t, axis=None).tolist(), [1, 3, 0, 2, 5, 4])
        self.assertEqual([x.tolist() for x in numerical.nonzero(t > 2)], [[0, 0, 1], [0, 2, 2]])
        self.assertEqual(numerical.flatnonzero(t > 2).tolist(), [0, 2, 5])
        self.assertEqual(numerical.argwhere(t > 2).tolist(), [[0, 0], [0, 2], [1, 2]])
        self.assertEqual(numerical.argwhere(numerical.asarray(1.0)).shape, (1, 0))
        self.assertEqual(numerical.count_nonzero(t), 6)  # a NaN is nonzero
        self.assertEqual(numerical.count_nonzero(t > 2, axis=1).tolist(), [2, 1])
        self.assertEqual(numerical.count_nonzero(t > 2, axis=(0, 1)), 3)
        self.assertEqual(numerical.count_nonzero(t > 2, axis=(), keepdims=False).dtype, "int64")
        self.assertEqual(numerical.count_nonzero(t > 2, axis=0, keepdims=True).shape, (1, 3))
        self.assertEqual(numerical.count_nonzero(numerical.asarray([1.0, 0.0]), axis=()).tolist(), [1, 0])
        self.assertEqual(numerical.searchsorted([1.0, 2.0, 2.0, 3.0], 2.0), 1)
        self.assertEqual(numerical.searchsorted([1.0, 2.0, 2.0, 3.0], [2.0, 9.0], side="right").tolist(), [3, 4])
        self.assertEqual(numerical.searchsorted([3.0, 1.0, 2.0], [2.5], sorter=[1, 2, 0]).tolist(), [2])
        self.assertEqual(numerical.argmax(numerical.asarray([True, False, True])), 0)
        self.assertEqual(numerical.argsort(numerical.asarray([True, False])).tolist(), [1, 0])
        self.assertEqual(t[numerical.argsort(t[:, 0])].tolist()[0][0], 2.0)  # an int64 Tensor indexes
        self.assertEqual(t.argmax(axis=1).tolist(), [0, 1])
        self.assertEqual([x.tolist() for x in (t > 2).nonzero()], [[0, 0, 1], [0, 2, 2]])

    def test_refusals(self):
        with self.assertRaises(ValueError):
            numerical.argmax(numerical.zeros(0))
        with self.assertRaises(ValueError):
            numerical.argmin(numerical.zeros((2, 0)), axis=1)
        with self.assertRaises(ValueError):
            numerical.nonzero(numerical.asarray(1.0))
        with self.assertRaises(TypeError):
            numerical.argmax(numerical.asarray([1j]))
        with self.assertRaises(ValueError):
            numerical.searchsorted([1.0, 2.0], 1.5, side="middle")
        with self.assertRaises(ValueError):
            numerical.searchsorted([[1.0, 2.0]], 1.5)
        with self.assertRaises(IndexError):
            numerical.asarray([1.0, 2.0])[numerical.asarray([0.0])]


@unittest.skipIf(np is None, "NumPy not installed")
class IndexFunctionsAsNumpy(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261102)

    def same(self, got, want):
        got = np.asarray(got)
        self.assertEqual(got.dtype, np.int64)
        np.testing.assert_array_equal(got, want)

    def test_drawn(self):
        x = self.rng.standard_normal((6, 7))
        x[1, 2] = np.nan
        x[3, :] = x[3, 0]
        x[4, 1] = -0.0
        x[4, 2] = 0.0
        ints = self.rng.integers(-5, 5, (6, 7))
        for array in (x, x.astype(np.float32), ints, ints > 0):
            source = numerical.asarray(array, dtype="int64") if array.dtype.kind == "i" else numerical.asarray(array)
            for axis in (None, 0, 1, -1):
                with self.subTest(dtype=array.dtype, axis=axis):
                    if axis is None:
                        self.assertEqual(numerical.argmax(source), np.argmax(array))
                        self.assertEqual(numerical.argmin(source), np.argmin(array))
                    else:
                        self.same(numerical.argmax(source, axis=axis), np.argmax(array, axis=axis))
                        self.same(numerical.argmin(source, axis=axis), np.argmin(array, axis=axis))
                    self.same(numerical.argsort(source, axis=axis), np.argsort(array, axis=axis, kind="stable"))
                    counted = numerical.count_nonzero(source, axis=axis)
                    np.testing.assert_array_equal(np.asarray(counted), np.count_nonzero(array, axis=axis))
            for got, want in zip(numerical.nonzero(source), np.nonzero(array)):
                self.same(got, want)
            self.same(numerical.flatnonzero(source), np.flatnonzero(array))
            self.same(numerical.argwhere(source), np.argwhere(array))
        z = self.rng.standard_normal(40) + 1j * self.rng.standard_normal(40)
        z[::4] = z[1]
        z[7] = complex(np.nan, 0)
        self.same(numerical.argsort(z), np.argsort(z, kind="stable"))

    def test_long_lines(self):
        big = self.rng.standard_normal(300_000)
        big[::7] = np.nan
        big[::11] = 0.0
        big[1::13] = -0.0
        self.same(numerical.argsort(big), np.argsort(big, kind="stable"))
        self.assertEqual(numerical.argmax(big), np.argmax(big))
        clean = big[~np.isnan(big)]
        self.assertEqual((numerical.argmax(clean), numerical.argmin(clean)), (np.argmax(clean), np.argmin(clean)))
        for got, want in zip(numerical.nonzero(big > 0.5), np.nonzero(big > 0.5)):
            self.same(got, want)

    def test_searchsorted(self):
        a = np.sort(self.rng.standard_normal(50))
        a[-3:] = np.nan
        v = np.concatenate([self.rng.standard_normal(20), [np.nan, a[5], a[10], -np.inf, np.inf]])
        for side in ("left", "right"):
            with self.subTest(side=side):
                self.same(numerical.searchsorted(a, v, side=side), np.searchsorted(a, v, side=side))
                self.same(numerical.searchsorted(a, v.reshape(5, 5), side=side), np.searchsorted(a, v.reshape(5, 5), side=side))
        ints = np.sort(self.rng.integers(-9, 9, 30))
        self.same(numerical.searchsorted(numerical.asarray(ints, dtype="int64"), numerical.asarray([-3, 0, 20], dtype="int64")),
                  np.searchsorted(ints, [-3, 0, 20]))
        shuffled = self.rng.permutation(a)
        sorter = np.argsort(shuffled, kind="stable")
        self.same(numerical.searchsorted(shuffled, v, sorter=sorter), np.searchsorted(shuffled, v, sorter=sorter))
        self.assertEqual(numerical.searchsorted(a, 0.25), np.searchsorted(a, 0.25))

    def test_int64_indexing(self):
        x = self.rng.standard_normal((6, 7))
        t = numerical.asarray(x)
        np.testing.assert_array_equal(np.asarray(t[numerical.argsort(t[:, 0])]), x[np.argsort(x[:, 0], kind="stable")])
        index = numerical.asarray([[0, 1], [5, -1]], dtype="int64")
        np.testing.assert_array_equal(np.asarray(t[index]), x[np.array([[0, 1], [5, -1]])])
        np.testing.assert_array_equal(np.asarray(t[numerical.nonzero(t > 0)]), x[np.nonzero(x > 0)])


if __name__ == "__main__":
    unittest.main()
