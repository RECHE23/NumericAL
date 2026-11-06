"""NumPy's root functions of the second set against NumPy's own answers: the real elementary functions (within a few
units in the last place: both sides take their platform's libm or their own), rounding, prod, cumsum, cumprod, sort
and diff; the cube root within half an ulp. Deterministic; the comparisons skipped without NumPy."""
import math
import random
import struct
import unittest
from fractions import Fraction

import numerical

try:
    import numpy as np
except ImportError:
    np = None


class CubeRoot(unittest.TestCase):
    @staticmethod
    def within(root, x, single):
        """Whether `root` is within 0.51 ulp of x's real cube root, by the exact cubes of the bounds."""
        def neighbor(value, toward):
            if not single:
                return math.nextafter(value, toward)
            bits = struct.unpack("<I", struct.pack("<f", value))[0]  # the magnitude's bits, after the sign
            step = 1 if (toward > value) == (value > 0) else -1
            return struct.unpack("<f", struct.pack("<I", bits + step))[0]
        low = Fraction(root) - Fraction(51, 100) * (Fraction(root) - Fraction(neighbor(root, -math.inf)))
        high = Fraction(root) + Fraction(51, 100) * (Fraction(neighbor(root, math.inf)) - Fraction(root))
        return low ** 3 <= Fraction(x) <= high ** 3

    def test_cbrt_within_half_an_ulp(self):
        # C's cbrt is up to 3 ulps off (glibc); one Newton step on an exact residual leaves half an ulp, over every
        # binade, the subnormals included
        draw = random.Random(20261106)
        values = [draw.choice((-1.0, 1.0)) * draw.random() * 2.0 ** draw.randint(-1074, 1023) for _ in range(1500)]
        values += [5e-324, -5e-324, 2.0 ** -600, math.nextafter(2.0 ** -600, 0.0), 1.7976931348623157e308, 27.0]
        for x, root in zip(values, numerical.cbrt(values).tolist()):
            with self.subTest(x=x.hex()):
                self.assertTrue(self.within(root, x, single=False), root.hex())
        singles = [struct.unpack("<f", struct.pack("<I", draw.getrandbits(31)))[0] for _ in range(1500)]
        singles = [x for x in singles if math.isfinite(x)] + [2.0 ** -149, 2.0 ** -60, 3.4028234663852886e38]
        for x, root in zip(singles, numerical.cbrt(numerical.Tensor(singles, dtype="float32")).tolist()):
            with self.subTest(x=x.hex(), dtype="float32"):
                self.assertTrue(self.within(root, x, single=True), root.hex())
        specials = numerical.cbrt([0.0, -0.0, math.inf, -math.inf, math.nan, -8.0]).tolist()
        self.assertEqual([math.copysign(1.0, v) for v in specials[:2]], [1.0, -1.0])
        self.assertEqual(specials[2:4] + specials[5:], [math.inf, -math.inf, -2.0])
        self.assertTrue(math.isnan(specials[4]))


@unittest.skipIf(np is None, "NumPy not installed")
class NumpyWave2(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261102)

    def same(self, got, expected, ulps=0):
        got, expected = np.asarray(got), np.asarray(expected)
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, expected.dtype)
        if ulps:
            np.testing.assert_array_max_ulp(got, expected, maxulp=ulps)
        else:
            np.testing.assert_array_equal(got, expected)

    def test_elementary_functions(self):
        x = self.rng.standard_normal(200) * 3
        unit = np.tanh(x)
        above = 1 + np.abs(x)
        for name, argument in (("arcsin", unit), ("arccos", unit), ("arctan", x), ("sinh", x), ("cosh", x),
                               ("arcsinh", x), ("arccosh", above), ("arctanh", unit * 0.99), ("log2", above),
                               ("log10", above), ("log1p", np.abs(x)), ("expm1", x), ("cbrt", x),
                               ("deg2rad", x * 60), ("rad2deg", x)):
            for dtype in (np.float64, np.float32):
                with self.subTest(name=name, dtype=dtype):
                    value = argument.astype(dtype)
                    self.same(getattr(numerical, name)(value), getattr(np, name)(value), ulps=4)
        self.assertIs(numerical.radians, numerical.deg2rad)
        self.assertIs(numerical.degrees, numerical.rad2deg)
        with self.assertRaises(TypeError):
            numerical.sinh(np.ones(2) * 1j)

    def test_rounding(self):
        x = np.array([-2.5, -1.5, -0.5, -0.4, 0.0, -0.0, 0.5, 1.5, 2.5, 3.7, np.nan, np.inf])
        for name in ("floor", "ceil", "trunc", "rint", "sign"):
            with self.subTest(name=name):
                got, want = np.asarray(getattr(numerical, name)(x)), getattr(np, name)(x)
                self.same(got, want)
                np.testing.assert_array_equal(np.signbit(got), np.signbit(want))
        y = self.rng.standard_normal(50) * 1000
        for decimals in (0, 1, 2, -1, -2):
            with self.subTest(decimals=decimals):
                self.same(numerical.round(y, decimals), np.round(y, decimals))
        self.assertIs(numerical.around, numerical.round)
        self.same(numerical.asarray(y).round(1), y.round(1))
        for dtype in (np.float64, np.float32, np.complex128, np.complex64):  # a complex number's parts apart
            with self.subTest(dtype=dtype):
                value = (y + 1j * y[::-1] if dtype in (np.complex128, np.complex64) else y).astype(dtype)
                self.same(numerical.round(value, 1), np.round(value, 1))

    def test_rounding_to_far_places_is_exact(self):
        # Where NumPy's product by 10ᵈ rounds off a last bit or overflows, NumPy's answer is not the nearest value to
        # the decimal rounding; ours is, a value already exact to the places being its own rounding.
        from decimal import Decimal, ROUND_HALF_EVEN, localcontext
        y = self.rng.standard_normal(300) * 100
        for decimals in (15, 17, 20):
            got = np.asarray(numerical.round(y, decimals))
            with localcontext() as context:
                context.prec = 400
                exact = [float(Decimal(v).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_EVEN)) for v in y]
            self.same(got, np.array(exact))
            self.assertTrue(np.any(np.round(y, decimals) != got))  # where NumPy's misses
        self.same(numerical.round(np.array([1e307, -1e308]), 2), np.array([1e307, -1e308]))  # NumPy's: inf
        far = np.array([1.5, -2.25, np.inf, -np.inf, np.nan, -0.0])
        self.same(numerical.round(far, 400), far)
        self.same(numerical.round(far, 10**30), far)
        down = np.asarray(numerical.round(far, -400))  # every finite value to a zero of its sign (NumPy's: NaN)
        self.same(down, np.array([0.0, -0.0, np.inf, -np.inf, np.nan, -0.0]))
        np.testing.assert_array_equal(np.signbit(down), [False, True, False, True, False, True])
        single = np.array([1.5, 1.234e-39, 3e-39], dtype=np.float32)  # past 10³⁸, through double
        self.same(numerical.round(single, 40), np.array([1.5, 1.2e-39, 3e-39], dtype=np.float32))
        self.same(numerical.round(np.array([True, False]), 0), np.array([1.0, 0.0]))

    def test_real_functions_read_a_numpy_array_where_it_lies(self):
        x = self.rng.standard_normal(3000)
        x[::9] = np.nan
        for name in ("floor", "ceil", "trunc", "rint", "sign", "deg2rad", "rad2deg", "arctan", "cbrt", "expm1"):
            for array in (x, x.astype(np.float32), x[::3], x.reshape(30, 100).T):
                with self.subTest(name=name, dtype=array.dtype, contiguous=array.flags.c_contiguous):
                    self.same(getattr(numerical, name)(array), getattr(numerical, name)(numerical.asarray(array)))
        self.same(numerical.floor(np.array([True, False])), np.array([1.0, 0.0]))  # a bool array read as float64
        with self.assertRaises(TypeError):
            numerical.floor(np.array([1j]))

    def test_floor_division_and_remainder(self):
        a = self.rng.standard_normal(2000) * 10
        b = self.rng.standard_normal(2000) * 3
        a[::50], b[::70], a[::90], b[::110], a[::130], a[1::40] = 0.0, 0.0, np.inf, -np.inf, np.nan, -0.0
        for dtype in (np.float64, np.float32):
            x, y = a.astype(dtype), b.astype(dtype)
            with np.errstate(all="ignore"):
                for name in ("floor_divide", "remainder"):
                    with self.subTest(name=name, dtype=dtype):
                        got, want = np.asarray(getattr(numerical, name)(x, y)), getattr(np, name)(x, y)
                        self.same(got, want)
                        np.testing.assert_array_equal(np.signbit(got), np.signbit(want))  # zeros of the right sign
                with self.subTest(operators=dtype):
                    t, u = numerical.asarray(x), numerical.asarray(y)
                    self.same(t // u, x // y)
                    self.same(t % u, x % y)
                    self.same(t // 2, x // 2)
                    self.same(3 % t, 3 % x)
                    self.same(divmod(t, u)[1], np.divmod(x, y)[1])
                    self.same(numerical.divmod(x, y)[0], np.divmod(x, y)[0])
        self.assertIs(numerical.mod, numerical.remainder)
        self.same(divmod(2.5, numerical.asarray([2.0]))[0], np.array([1.0]))
        with self.assertRaises(TypeError):
            numerical.asarray([1j]) // 2
        with self.assertRaises(TypeError):
            numerical.remainder(np.array([1j]), 2)

    def test_products_and_running_sums(self):
        a = self.rng.uniform(0.5, 1.5, (3, 4, 5))
        self.assertAlmostEqual(numerical.prod(a) / np.prod(a), 1.0, places=13)
        for axis in (0, 1, -1, (0, 2)):
            with self.subTest(axis=axis):
                np.testing.assert_allclose(np.asarray(numerical.prod(a, axis=axis)), np.prod(a, axis=axis), rtol=1e-14)
        self.same(numerical.prod(a, axis=1, keepdims=True).shape, np.prod(a, axis=1, keepdims=True).shape)
        self.assertEqual(numerical.prod(np.ones(0)), 1.0)
        self.same(numerical.prod(np.array([True, True, False])), np.prod(np.array([True, True, False])).astype(float))
        for function in ("cumsum", "cumprod"):
            for axis in (None, 0, 1, -1):
                with self.subTest(function=function, axis=axis):
                    self.same(getattr(numerical, function)(a, axis=axis), getattr(np, function)(a, axis=axis))
        z = a + 1j * a[::-1]
        self.same(numerical.cumsum(z, axis=2), np.cumsum(z, axis=2))
        t = numerical.asarray(a)
        self.same(t.cumsum(axis=0), a.cumsum(axis=0))
        np.testing.assert_allclose(np.asarray(t.prod(axis=2)), a.prod(axis=2), rtol=1e-14)

    def test_sort_and_differences(self):
        a = self.rng.standard_normal((4, 7))
        a[1, 3] = np.nan
        for axis in (None, 0, 1, -1):
            with self.subTest(axis=axis):
                self.same(numerical.sort(a, axis=axis), np.sort(a, axis=axis))
        z = np.array([1 + 2j, 1 - 1j, np.nan + 0j, 0 + 5j, -1 + 0j])
        self.same(numerical.sort(z), np.sort(z))
        self.same(numerical.sort(a.astype(np.float32)), np.sort(a.astype(np.float32)))
        mask = self.rng.random((5, 6)) > 0.5
        for axis in (None, 0, -1):
            with self.subTest(mask_axis=axis):
                self.same(numerical.sort(mask, axis=axis), np.sort(mask, axis=axis))
                self.same(numerical.sort(numerical.asarray(mask), axis=axis), np.sort(mask, axis=axis))
        long = self.rng.standard_normal(300_000)  # sorted among the threads
        long[::97] = np.nan
        long[::89] = -0.0
        long[1::89] = 0.0
        self.assertEqual(np.asarray(numerical.sort(long)).tobytes(), np.sort(long, kind="stable").tobytes())
        for n, axis in ((1, -1), (2, -1), (1, 0), (0, 1), (3, 1)):
            with self.subTest(n=n, axis=axis):
                self.same(numerical.diff(a, n=n, axis=axis), np.diff(a, n=n, axis=axis))
        with self.assertRaises(ValueError):
            numerical.diff(a, n=-1)
        with self.assertRaises(ValueError):
            numerical.diff(3.0)


if __name__ == "__main__":
    unittest.main()
