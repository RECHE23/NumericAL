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
        self.assertEqual(numerical.asarray([1, 2]).dtype, "int64")  # Python's ints as NumPy reads them
        self.assertEqual(numerical.Tensor([1, BIG], dtype="int64").tolist(), [1, BIG])
        with self.assertRaises(OverflowError):
            numerical.asarray([2**63], dtype="int64")
        with self.assertRaises(OverflowError):
            numerical.asarray(2**63, dtype="int64")

    def test_nested_lists_read_natively_as_the_python_reading_reads_them(self):
        # the native walk takes ints alone, each exact; any other leaf, or a ragged list, falls to the Python reading,
        # whose answers these are
        cases = [([], (0,), []), ([[]], (1, 0), [[]]), ([True, False, 2], (3,), [1, 0, 2]),
                 ([[2**63 - 1], [-2**63]], (2, 1), [[2**63 - 1], [-2**63]]), ((1, (2, 3))[1:], (1, 2), [[2, 3]]),
                 ([1.5, 2], (2,), [1, 2]), ([1, 2.0], (2,), [1, 2])]
        for source, shape, values in cases:
            with self.subTest(source=source):
                t = numerical.asarray(source, dtype="int64")
                self.assertEqual((t.shape, t.dtype, t.tolist()), (shape, "int64", values))
        for source, error in (([2**63], OverflowError), ([-2**63 - 1], OverflowError), ([None], TypeError),
                              ([[1, 2], [3]], ValueError), ([10**30, 1], OverflowError)):
            with self.subTest(source=source), self.assertRaises(error):
                numerical.asarray(source, dtype="int64")
        self.assertEqual(numerical.asarray(list(range(-5000, 5000)), dtype="int64").tolist(), list(range(-5000, 5000)))

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

    def test_beside_a_python_int_exactly(self):
        # a Python int beside an int64 tensor is read as int64, not through float64: BIG - 1 and BIG were one there
        t = numerical.asarray([BIG - 1, 5, -7], dtype="int64")
        self.assertEqual((t == BIG).tolist(), [False, False, False])
        self.assertEqual((t != BIG).tolist(), [True, True, True])
        self.assertEqual((t < BIG).tolist(), [True, True, True])
        self.assertEqual(numerical.equal(t, BIG - 1).tolist(), [True, False, False])
        # the Python int first, the functions' form: the same, mirrored (less(k, t) is t > k)
        self.assertEqual(numerical.equal(BIG - 1, t).tolist(), [True, False, False])
        self.assertEqual(numerical.not_equal(BIG, t).tolist(), [True, True, True])
        self.assertEqual(numerical.less(BIG - 2, t).tolist(), [True, False, False])
        self.assertEqual(numerical.greater_equal(BIG - 1, t).tolist(), [True, True, True])
        self.assertEqual(numerical.maximum(t, BIG).tolist(), [BIG, BIG, BIG])
        self.assertEqual(numerical.minimum(t, -BIG).tolist(), [-BIG, -BIG, -BIG])
        self.assertEqual(numerical.clip(t, 0, BIG - 2).tolist(), [BIG - 2, 5, 0])
        self.assertEqual(numerical.isin(t, [BIG, 5]).tolist(), [False, True, False])
        self.assertEqual(numerical.isin(t, BIG - 1).tolist(), [True, False, False])
        # past int64: a comparison answers, the same for each element (NumPy 2's); arithmetic raises
        self.assertEqual((t == 2**70).tolist(), [False] * 3)
        self.assertEqual((t < 2**70).tolist(), [True] * 3)
        self.assertEqual((t >= -2**70).tolist(), [True] * 3)
        self.assertEqual((t != -2**63 - 1).tolist(), [True] * 3)
        self.assertEqual((numerical.asarray([True, False]) > 2**64).tolist(), [False, False])
        with self.assertRaises(OverflowError):
            t + 2**70

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

    def test_reductions(self):
        top = 2**63 - 1
        t = numerical.asarray([[top, 1, 2], [3, -4, 5]], dtype="int64")
        self.assertEqual(numerical.sum(t), top + 1 + 2 + 3 - 4 + 5 - 2**64)  # wrapped
        self.assertIs(type(numerical.sum(t)), int)
        self.assertEqual(numerical.sum(t, axis=0).tolist(), [top + 3 - 2**64, -3, 7])
        self.assertEqual(numerical.sum(t, axis=1, keepdims=True).shape, (2, 1))
        self.assertEqual(numerical.prod(numerical.asarray([3, -2, 5], dtype="int64")), -30)
        self.assertEqual(numerical.cumsum(t, axis=1).tolist(), [[top, top + 1 - 2**64, top + 3 - 2**64], [3, -1, 4]])
        self.assertEqual(numerical.cumprod(numerical.asarray([2, 3, 4], dtype="int64")).tolist(), [2, 6, 24])
        self.assertEqual(numerical.sort(t, axis=1).tolist(), [[1, 2, top], [-4, 3, 5]])
        self.assertEqual(numerical.mean(numerical.asarray([1, 2], dtype="int64")), 1.5)  # in float64
        self.assertEqual(numerical.var(numerical.asarray([1, 3], dtype="int64")), 1.0)
        self.assertEqual(numerical.std(numerical.asarray([1, 3], dtype="int64")), 1.0)
        self.assertEqual((numerical.any(t), numerical.all(t)), (True, True))
        self.assertEqual(numerical.all(numerical.asarray([1, 0], dtype="int64")), False)
        self.assertEqual(numerical.trace(numerical.asarray([[top, 0], [0, 1]], dtype="int64")), top + 1 - 2**64)
        self.assertEqual(numerical.ptp(t, axis=1).tolist(), [top - 1, 9])
        self.assertEqual(numerical.diff(numerical.asarray([1, 4, 9], dtype="int64")).tolist(), [3, 5])
        self.assertEqual(t.sum(dtype="float64"), float(top) + 7.0)  # accumulated in float64: no wrap

    def test_products_stay_integers_and_the_linear_algebra_floats(self):
        # NumPy's: a product of integers is an integer, wrapped; the linear algebra and the transforms compute in
        # floating point.
        t = numerical.asarray([[1, 2], [3, 4]], dtype="int64")
        self.assertEqual((t @ t).tolist(), [[7, 10], [15, 22]])
        self.assertEqual((t @ t).dtype, "int64")
        self.assertEqual(numerical.einsum("ij->", t).tolist(), 10)
        self.assertEqual(numerical.einsum("ij->", t).dtype, "int64")
        big = numerical.asarray([[2 ** 62, 0], [0, 1]], dtype="int64")
        self.assertEqual((big @ big).tolist(), [[0, 0], [0, 1]])  # 2^124 modulo 2^64
        self.assertEqual(numerical.solve(t, t).dtype, "float64")
        self.assertAlmostEqual(numerical.det(t), -2.0, places=14)
        self.assertEqual(numerical.fft.fft(numerical.asarray([1, 2], dtype="int64")).tolist(), [3 + 0j, -1 + 0j])
        self.assertEqual(numerical.norm(numerical.asarray([3, 4], dtype="int64")), 5.0)


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
        self.assertEqual(numerical.asarray(np.array([1, 2])).dtype, "int64")  # NumPy's integers as they are

    def test_arithmetic_and_reductions_as_numpy(self):
        rng = np.random.default_rng(20261102)
        a = np.concatenate([rng.integers(-2**62, 2**62, 300), rng.integers(-1000, 1000, 300)]).astype(np.int64)
        b = np.concatenate([rng.integers(-2**62, 2**62, 300), rng.integers(-50, 50, 300)]).astype(np.int64)
        x, y = numerical.asarray(a, dtype="int64"), numerical.asarray(b, dtype="int64")
        with np.errstate(all="ignore"):
            for name, got, want in (("+", x + y, a + b), ("-", x - y, a - b), ("*", x * y, a * b), ("//", x // y, a // b),
                                    ("%", x % y, a % b), ("/", x / y, a / b), ("neg", -x, -a), ("abs", abs(x), np.abs(a)),
                                    ("max", numerical.maximum(x, y), np.maximum(a, b)), ("sign", numerical.sign(x), np.sign(a))):
                with self.subTest(name=name):
                    got = np.asarray(got)
                    self.assertEqual(got.dtype, want.dtype)
                    np.testing.assert_array_equal(got, want)
        grid, wide = a.reshape(20, 30), x.reshape((20, 30))
        for axis in (None, 0, 1):
            with self.subTest(axis=axis):
                for name in ("sum", "prod", "max", "min", "any", "all", "cumsum", "cumprod", "sort"):
                    result = getattr(numerical, name)(wide, axis=axis)
                    want = getattr(np, name)(grid, axis=axis)
                    if not isinstance(result, numerical.Tensor):  # a Python number, of NumPy's scalar's kind and value
                        # (not np.asarray's dtype of it: the platform's integer, int32 on a 32-bit Windows)
                        self.assertIs(type(result), bool if want.dtype == np.bool_ else int, name)
                        self.assertEqual(result, want.item(), name)
                        continue
                    got = np.asarray(result)
                    self.assertEqual(got.dtype, np.asarray(want).dtype, name)
                    np.testing.assert_array_equal(got, want, err_msg=name)
        np.testing.assert_array_equal(np.asarray(numerical.mean(wide, axis=1)), np.mean(grid, axis=1))
        np.testing.assert_array_equal(np.asarray(numerical.var(wide)), np.var(grid))

    def test_exported(self):
        t = numerical.asarray([[1, BIG], [3, -4]], dtype="int64")
        for exported in (np.asarray(t), np.from_dlpack(t)):
            self.assertEqual(exported.dtype, np.int64)
            np.testing.assert_array_equal(exported, np.array([[1, BIG], [3, -4]]))
        self.assertEqual(numerical.asarray([1], dtype=np.int64).dtype, "int64")
        self.assertEqual(numerical.zeros(2, dtype=np.dtype("int64")).dtype, "int64")


if __name__ == "__main__":
    unittest.main()
