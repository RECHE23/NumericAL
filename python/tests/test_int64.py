"""The int64 dtype: read exactly (past 2**53), converted, exported (NumPy's array interface, DLPack, pickle), compared,
selected, moved and reduced to its extremes as NumPy's; refused where an operation computes in floating point."""
import pickle
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None

BIG = 2**62 + 1  # no float64 holds it


class Int64(unittest.TestCase):
    def test_read_exactly(self):
        for source in ([1, BIG, -3], (1, BIG, -3)):
            with self.subTest(source=type(source).__name__):
                t = numerical.asarray(source, dtype="int64")
                self.assertEqual((t.dtype, t.tolist()), ("int64", [1, BIG, -3]))
        self.assertEqual(numerical.asarray([[1, 2], [3, 4]], dtype=int).tolist(), [[1, 2], [3, 4]])
        self.assertEqual(numerical.asarray(BIG, dtype="int64").tolist(), BIG)
        self.assertEqual(numerical.asarray(True, dtype="int64").tolist(), 1)
        self.assertEqual(numerical.asarray([1, 2]).dtype, "float64")  # integers stay float64 by default until 2026.12
        self.assertEqual(numerical.Tensor([1, BIG], dtype="int64").tolist(), [1, BIG])
        with self.assertRaises(OverflowError):
            numerical.asarray([2**63], dtype="int64")
        with self.assertRaises(OverflowError):
            numerical.asarray(2**63, dtype="int64")

    def test_conversions(self):
        self.assertEqual(numerical.asarray([1.9, -1.9, 0.5], dtype="int64").tolist(), [1, -1, 0])  # toward zero
        self.assertEqual(numerical.asarray(2.7, dtype="int64").tolist(), 2)
        self.assertEqual(numerical.asarray([1.5, -2.5]).astype("int64").tolist(), [1, -2])
        self.assertEqual(numerical.asarray([1.5], dtype="float32").astype("int64").tolist(), [1])
        self.assertEqual(numerical.asarray([BIG], dtype="int64").astype("float64").tolist(), [float(BIG)])
        self.assertEqual(numerical.asarray([BIG, -1], dtype="int64").astype("complex64").dtype, "complex64")
        self.assertEqual(numerical.asarray([0, 3], dtype="int64").astype("bool").tolist(), [False, True])
        self.assertEqual(numerical.asarray([True, False]).astype(int).tolist(), [1, 0])
        for value in (float("nan"), float("inf"), -float("inf")):
            for source in ([value], numerical.asarray([value]), numerical.asarray([value], dtype="float32"), value):
                with self.subTest(value=value, source=type(source).__name__), self.assertRaises(ValueError):
                    numerical.asarray(source, dtype="int64")
        for value in (2.0**63, -2.0**63 * 2, 1e300):
            with self.subTest(value=value), self.assertRaises(OverflowError):
                numerical.asarray([value], dtype="int64")
        self.assertEqual(numerical.asarray([-2.0**63], dtype="int64").tolist(), [-2**63])  # the bound itself fits
        with self.assertRaises(numerical.error):  # a complex number is never cast to a real dtype
            numerical.asarray([1j], dtype="int64")
        with self.assertRaises(numerical.error):
            numerical.asarray(numerical.asarray([1j]), dtype="int64")

    def test_creation(self):
        self.assertEqual(numerical.zeros(3, dtype="int64").tolist(), [0, 0, 0])
        self.assertEqual(numerical.ones((2,), dtype=int).tolist(), [1, 1])
        self.assertEqual(numerical.full(2, BIG, dtype="int64").tolist(), [BIG, BIG])
        self.assertEqual(numerical.full(2, -2.7, dtype="int64").tolist(), [-2, -2])
        self.assertEqual(numerical.eye(2, dtype="int64").tolist(), [[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            numerical.full(2, float("nan"), dtype="int64")
        with self.assertRaises(OverflowError):
            numerical.full(2, 2**64, dtype="int64")

    def test_moved_compared_selected(self):
        t = numerical.asarray([[1, BIG], [3, -4]], dtype="int64")
        self.assertEqual((t.itemsize, t.nbytes, t.shape), (8, 32, (2, 2)))
        self.assertEqual(t[0, 1], BIG)
        self.assertEqual(t[:, 1].tolist(), [BIG, -4])
        self.assertEqual(t.T.tolist(), [[1, 3], [BIG, -4]])
        self.assertEqual(t.reshape(4).tolist(), [1, BIG, 3, -4])
        self.assertEqual(numerical.concatenate([t, t]).dtype, "int64")
        self.assertEqual((t == t).tolist(), [[True, True], [True, True]])
        self.assertEqual((t < 2).tolist(), [[True, False], [False, True]])
        self.assertEqual(numerical.where(t > 0, t, t).tolist(), [[1, BIG], [3, -4]])
        self.assertEqual(numerical.isnan(t).tolist(), [[False, False], [False, False]])
        self.assertEqual(pickle.loads(pickle.dumps(t)).tolist(), t.tolist())
        self.assertIn("4611686018427387905", repr(t))

    def test_extremes(self):
        t = numerical.asarray([[1, BIG], [3, -4]], dtype="int64")
        self.assertEqual((numerical.max(t), numerical.min(t)), (BIG, -4))
        self.assertIs(type(numerical.max(t)), int)
        self.assertEqual(numerical.max(t, axis=1).tolist(), [BIG, 3])
        self.assertEqual(numerical.min(t, axis=0).tolist(), [1, -4])

    def test_beside_floats(self):
        t = numerical.asarray([1, 2], dtype="int64")
        self.assertEqual((t + 0.5).tolist(), [1.5, 2.5])  # a float beside int64: float64, as NumPy 2
        self.assertEqual((t + numerical.asarray([0.5], dtype="float32")).dtype, "float64")
        self.assertEqual((t * numerical.asarray([1j])).dtype, "complex128")
        self.assertEqual((t * numerical.asarray([1j], dtype="complex64")).dtype, "complex128")
        self.assertEqual((t + 1j).dtype, "complex128")

    def test_arithmetic_wraps_as_numpy(self):
        top, bottom = 2**63 - 1, -2**63
        t = numerical.asarray([top, bottom, 7, -7], dtype="int64")
        self.assertEqual((t + 1).tolist(), [bottom, bottom + 1, 8, -6])
        self.assertEqual((t - numerical.asarray([1, 1, 1, 1], dtype="int64")).tolist(), [top - 1, top, 6, -8])
        self.assertEqual((t * 2).tolist(), [-2, 0, 14, -14])
        self.assertEqual((-t).tolist(), [-top, bottom, -7, 7])
        self.assertEqual(abs(t).tolist(), [top, bottom, 7, 7])  # |−2⁶³| wraps to itself, as NumPy's
        self.assertEqual((t // 2).tolist(), [top // 2, bottom // 2, 3, -4])
        self.assertEqual((t % 3).tolist(), [top % 3, bottom % 3, 1, 2])
        self.assertEqual((t // 0).tolist(), [0, 0, 0, 0])  # NumPy's answer by zero
        self.assertEqual((numerical.asarray([bottom], dtype="int64") // -1).tolist(), [bottom])
        self.assertEqual((t / 2).dtype, "float64")
        self.assertEqual((numerical.asarray([3, -2], dtype="int64") ** 3).tolist(), [27, -8])
        self.assertEqual((numerical.asarray([2], dtype="int64") ** 64).tolist(), [0])
        self.assertEqual((2 ** numerical.asarray([10], dtype="int64")).tolist(), [1024])
        self.assertEqual((numerical.asarray([4], dtype="int64") ** 0.5).tolist(), [2.0])
        with self.assertRaisesRegex(ValueError, "negative integer powers"):
            numerical.asarray([2], dtype="int64") ** -1
        self.assertEqual(numerical.maximum(t, 0).tolist(), [top, 0, 7, 0])
        self.assertEqual(numerical.minimum(t, numerical.asarray([0, 0, 0, 0], dtype="int64")).tolist(), [0, bottom, 0, -7])
        self.assertEqual(numerical.clip(t, -10, 10).tolist(), [10, -10, 7, -7])
        self.assertEqual(numerical.clip(t, -0.5, 10).dtype, "float64")

    def test_functions_keep_or_leave_int64_as_numpy(self):
        t = numerical.asarray([-7, 0, 3, 2**62 + 1], dtype="int64")
        for name in ("floor", "ceil", "trunc"):
            self.assertIs(getattr(numerical, name)(t), t)  # an integer its own floor, an immutable tensor itself
        self.assertEqual(numerical.sign(t).tolist(), [-1, 0, 1, 1])
        self.assertEqual(numerical.sign(t).dtype, "int64")
        self.assertEqual(numerical.square(numerical.asarray([3, -4], dtype="int64")).tolist(), [9, 16])
        for name in ("rint", "sqrt", "exp", "sin", "log1p", "deg2rad"):
            self.assertEqual(getattr(numerical, name)(numerical.asarray([1, 4], dtype="int64")).dtype, "float64")
        self.assertIs(numerical.round(t, 3), t)
        self.assertEqual(numerical.round(numerical.asarray([15, 25, -15, 1250, 1350, -1250], dtype="int64"), -1).tolist(),
                         [20, 20, -20, 1250, 1350, -1250])
        self.assertEqual(numerical.round(numerical.asarray([1250, 1350, 1251], dtype="int64"), -2).tolist(), [1200, 1400, 1300])
        big = 2**62 + 123456789
        self.assertEqual(numerical.round(numerical.asarray([big], dtype="int64"), -1).tolist(), [big - 3])  # exact (NumPy, through float64: …416, no multiple of 10)
        self.assertEqual(numerical.round(numerical.asarray([6 * 10**18, -6 * 10**18, 10**18], dtype="int64"), -19).tolist(),
                         [10**19 - 2**64, 2**64 - 10**19, 0])  # ±10¹⁹, wrapped
        self.assertEqual(numerical.round(numerical.asarray([2**63 - 1], dtype="int64"), -25).tolist(), [0])

    def test_refused_where_floating_point_computes(self):
        t = numerical.asarray([[1, 2], [3, 4]], dtype="int64")
        for name, call in (("sum", lambda: numerical.sum(t)),
                           ("mean", lambda: numerical.mean(t)), ("@", lambda: t @ t),
                           ("sort", lambda: numerical.sort(t)), ("solve", lambda: numerical.solve(t, t)),
                           ("einsum", lambda: numerical.einsum("ij->", t)), ("fft", lambda: numerical.fft.fft(t))):
            with self.subTest(name=name), self.assertRaises(TypeError):
                call()


@unittest.skipIf(np is None, "NumPy not installed")
class Int64WithNumpy(unittest.TestCase):
    def test_numpy_integers_read_exactly(self):
        for dtype in ("<i8", ">i8", "<i4", ">i4", "<i2", ">i2", "i1", "<u4", ">u2", "u1"):
            with self.subTest(dtype=dtype):
                values = [1, -5, 100] if dtype[-2] == "i" or dtype[0] == "i" else [1, 5, 100]
                a = np.array(values, dtype=dtype)
                self.assertEqual(numerical.asarray(a, dtype="int64").tolist(), a.astype(np.int64).tolist())
        for dtype in ("<i8", ">i8"):
            a = np.array([BIG, -BIG], dtype=dtype)
            self.assertEqual(numerical.asarray(a, dtype="int64").tolist(), [BIG, -BIG])
            self.assertEqual(numerical.asarray(a[::-1], dtype="int64").tolist(), [-BIG, BIG])  # strided
        for dtype in ("<u8", ">u8"):  # past int64, wrapped as NumPy's cast
            a = np.array([2**64 - 1, 3], dtype=dtype)
            self.assertEqual(numerical.asarray(a, dtype="int64").tolist(), a.astype(np.int64).tolist())
        self.assertEqual(numerical.asarray(np.array([1.9, -1.9], dtype=">f8"), dtype="int64").tolist(), [1, -1])
        self.assertEqual(numerical.asarray(np.array([1.5], dtype=">f4"), dtype="int64").tolist(), [1])
        with self.assertRaises(ValueError):
            numerical.asarray(np.array([np.nan], dtype=">f8"), dtype="int64")
        self.assertEqual(numerical.asarray(np.array([True, False]), dtype="int64").tolist(), [1, 0])
        self.assertEqual(numerical.asarray(np.int64(BIG), dtype="int64").tolist(), BIG)
        self.assertEqual(numerical.asarray(np.array([1, 2])).dtype, "float64")  # by default, until 2026.12

    def test_exported(self):
        t = numerical.asarray([[1, BIG], [3, -4]], dtype="int64")
        for exported in (np.asarray(t), np.from_dlpack(t)):
            self.assertEqual(exported.dtype, np.int64)
            np.testing.assert_array_equal(exported, np.array([[1, BIG], [3, -4]]))
        self.assertEqual(numerical.asarray([1], dtype=np.int64).dtype, "int64")
        self.assertEqual(numerical.zeros(2, dtype=np.dtype("int64")).dtype, "int64")


if __name__ == "__main__":
    unittest.main()
