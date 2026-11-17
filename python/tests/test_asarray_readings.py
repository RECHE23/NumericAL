"""What asarray reads as NumPy reads it: a range, a buffer (array.array, a memoryview, a bytearray), an object with
__array__ (pandas, xarray, torch on the CPU), and lists of Tensors or of NumPy arrays; the integers of a buffer as int64,
NumPy's small integer dtypes absent here."""

import array
import unittest

import numerical

try:
    import numpy as np
except ImportError:  # pragma: no cover - the CI images install NumPy
    np = None


class WithArray:
    def __array__(self, dtype=None, copy=None):
        return np.arange(3.0)


@unittest.skipIf(np is None, "NumPy is not installed")
class AsarrayReadings(unittest.TestCase):
    def assert_numpys(self, source, dtype=None):
        got, want = numerical.asarray(source), np.asarray(source)
        self.assertEqual((got.shape, got.dtype, got.tolist()), (want.shape, dtype or str(want.dtype), want.tolist()))

    def test_ranges_buffers_and_array_holders(self):
        for source in (range(3), range(0), range(1, 10, 3), array.array("d", [0.0, 1.0]), array.array("f", [0.5]),
                       memoryview(np.arange(3.0)), memoryview(np.arange(6).reshape(2, 3)), WithArray()):
            with self.subTest(source=type(source).__name__):
                self.assert_numpys(source)
        self.assert_numpys(array.array("i", [1, 2]), "int64")  # int32 there
        self.assert_numpys(bytearray(b"ab"), "int64")  # uint8 there
        self.assertEqual(numerical.asarray(range(3), "float32").dtype, "float32")
        with self.assertRaises(TypeError):
            numerical.asarray(memoryview(np.zeros(2, dtype=np.complex128)))  # a format not read here

    def test_lists_of_tensors_and_of_numpy_arrays_are_stacked(self):
        for source in ([numerical.arange(2.0), numerical.arange(2.0)], [np.arange(2.0), np.arange(2.0)],
                       [np.arange(2), numerical.arange(2.0)]):
            with self.subTest(source=source):
                self.assert_numpys(source)


@unittest.skipIf(np is None, "NumPy is not installed")
class NanReductionsOfNumpyBools(unittest.TestCase):
    def test_a_numpy_bool_array_keeps_numpys_dtypes(self):
        b = np.array([[True, False], [True, True]])
        for name in ("nanmax", "nanmin", "nansum", "nanprod", "nancumsum", "nancumprod", "nanmean"):
            for axis in (None, 0):
                with self.subTest(name=name, axis=axis):
                    got = getattr(numerical, name)(b, axis=axis)
                    want = getattr(np, name)(b, axis=axis)
                    self.assertEqual((np.asarray(got).tolist(), str(np.asarray(got).dtype)),
                                     (np.asarray(want).tolist(), str(np.asarray(want).dtype)))


class RankZeroAsAFloat(unittest.TestCase):
    def test_is_integer_and_what_wants_a_float(self):
        import json
        s = numerical.asarray([1.5, 2.5, 3.0]).sum()
        self.assertTrue(s.is_integer())
        self.assertFalse(numerical.asarray([1.5, 2.0]).sum().is_integer())
        self.assertTrue(numerical.asarray([1, 2]).sum().is_integer())
        with self.assertRaises(TypeError):
            numerical.asarray([1.0, 2.0]).is_integer()  # a vector: no number
        with self.assertRaises(TypeError):
            json.dumps(s)  # documented: no Python float
        self.assertEqual(json.dumps(float(s)), "7.0")


if __name__ == "__main__":
    unittest.main()
