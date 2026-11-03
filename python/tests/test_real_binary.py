"""hypot, arctan2, copysign, nextafter, fmod, logaddexp, logaddexp2, fmax, fmin, heaviside, float_power, angle, unwrap
and nan_to_num against NumPy's answers over the edge values (zeros of both signs, infinities, NaN, the smallest
subnormal, the largest magnitudes): to the bit for the exact ones, within two ulps for those that call C's library
(hypot, atan2, exp, log1p), where NumPy's own kernels on x86-64 differ from it by an ulp. Deterministic; the NumPy
comparisons skipped without NumPy."""
import itertools
import math
import unittest
import warnings

import numerical

try:
    import numpy as np
except ImportError:
    np = None

_BINARY = ("hypot", "arctan2", "copysign", "nextafter", "fmod", "logaddexp", "logaddexp2", "fmax", "fmin", "heaviside")
_ROUNDED = ("hypot", "arctan2", "logaddexp", "logaddexp2", "angle")  # through C's library: within an ulp or two
_EDGES = [0.0, -0.0, 1.0, -1.0, 0.5, 2.5, 3.0, -7.0, 7.0, 1000.0, -1000.0, 1e300, -1e300, 5e-324, math.inf, -math.inf,
          math.nan]


def _bits(a):
    """The bytes of a NumPy array with every NaN made one NaN (NumPy's and C's NaNs differ in sign by platform)."""
    a = np.where(np.isnan(a), np.nan, a) if a.dtype.kind in "fc" else a
    return a.dtype, a.shape, a.tobytes()


class WithoutNumPy(unittest.TestCase):
    def test_values(self):
        nan, inf = math.nan, math.inf
        self.assertEqual(numerical.hypot([3.0], [4.0]).tolist(), [5.0])
        self.assertEqual(numerical.arctan2([0.0, -0.0], [-1.0, -1.0]).tolist(), [math.pi, -math.pi])
        self.assertEqual(numerical.copysign([1.0, 2.0], [-0.0, 0.0]).tolist(), [-1.0, 2.0])
        self.assertEqual(numerical.nextafter([1.0], [2.0]).tolist(), [1.0 + 2.0**-52])
        self.assertEqual(numerical.fmod([7.0, -7.0], [2.0, 2.0]).tolist(), [1.0, -1.0])
        self.assertTrue(math.isnan(numerical.fmod([1.0], [0.0]).tolist()[0]))
        self.assertEqual(numerical.logaddexp([inf, -inf], [inf, -inf]).tolist(), [inf, -inf])
        self.assertEqual(numerical.logaddexp2([1.0], [1.0]).tolist(), [2.0])
        self.assertEqual(numerical.fmax([nan, 1.0, -0.0], [2.0, nan, 0.0]).tolist(), [2.0, 1.0, 0.0])
        self.assertEqual(str(numerical.fmin([0.0], [-0.0]).tolist()[0]), "-0.0")
        self.assertEqual(str(numerical.fmax([-0.0], [0.0]).tolist()[0]), "0.0")
        self.assertTrue(math.isnan(numerical.fmax([nan], [nan]).tolist()[0]))
        h = numerical.heaviside([-2.0, -0.0, 0.0, 3.0, nan], 0.5).tolist()
        self.assertEqual(h[:4], [0.0, 0.5, 0.5, 1.0])
        self.assertTrue(math.isnan(h[4]))
        self.assertEqual(numerical.angle([1j, -1.0, -0.0]).tolist(), [math.pi / 2, math.pi, math.pi])
        self.assertEqual(numerical.angle([-1.0], deg=True).tolist(), [180.0])
        self.assertEqual(numerical.float_power(numerical.asarray([2], "int64"), -1).tolist(), [0.5])
        self.assertEqual(numerical.float_power([1j], 2).dtype, "complex128")
        self.assertEqual(numerical.nan_to_num([nan, inf, -inf, 1.0]).tolist(),
                         [0.0, 1.7976931348623157e308, -1.7976931348623157e308, 1.0])
        self.assertEqual(numerical.nan_to_num([nan, inf, -inf], nan=-1.0, posinf=9.0, neginf=-9.0).tolist(),
                         [-1.0, 9.0, -9.0])
        self.assertEqual(numerical.nan_to_num(numerical.asarray([inf], "float32")).tolist(), [3.4028234663852886e38])
        self.assertEqual(numerical.nan_to_num([complex(nan, -inf)]).tolist(), [complex(0.0, -1.7976931348623157e308)])
        self.assertEqual(numerical.unwrap([0.0, 6.0, 12.0]).tolist()[0], 0.0)
        self.assertEqual(numerical.unwrap([0.0, 3.0, 6.0], period=4.0).tolist(), [0.0, -1.0, -2.0])

    def test_dtypes(self):
        single = numerical.asarray([3.0], "float32")
        integers = numerical.asarray([7, -7, 5, -9223372036854775808], "int64")
        self.assertEqual(numerical.hypot(single, 4.0).dtype, "float32")  # a Python float is weak
        self.assertEqual(numerical.hypot(single, numerical.asarray([4.0])).dtype, "float64")
        self.assertEqual(numerical.hypot(integers, 1).dtype, "float64")
        self.assertEqual(numerical.heaviside(numerical.asarray([True]), numerical.asarray([False])).dtype, "float64")
        self.assertEqual(numerical.fmod(integers, numerical.asarray([2, 2, 0, -1], "int64")).tolist(), [1, -1, 0, 0])
        self.assertEqual(numerical.fmod(numerical.asarray([7, -7], "int64"), -2).tolist(), [1, -1])
        self.assertEqual(numerical.fmod(integers, 2.5).dtype, "float64")
        self.assertEqual(numerical.fmax(integers, 0).tolist(), [7, 0, 5, 0])
        self.assertEqual(numerical.fmin(integers, 0).dtype, "int64")
        self.assertEqual(numerical.angle(numerical.asarray([-1], "int64")).dtype, "float64")
        self.assertEqual(numerical.angle(numerical.asarray([1j], "complex64")).dtype, "float32")
        self.assertIs(numerical.nan_to_num(integers), integers)
        self.assertEqual(numerical.unwrap(numerical.asarray([0, 7], "int64")).dtype, "float64")

    def test_refusals(self):
        for name in _BINARY:
            with self.assertRaises(TypeError, msg=name):
                getattr(numerical, name)([1j], [1.0])
        with self.assertRaises(ValueError):
            numerical.hypot([1.0, 2.0], [1.0, 2.0, 3.0])
        with self.assertRaises(TypeError):
            numerical.unwrap([1j])
        with self.assertRaises(ValueError):
            numerical._numerical._real_binary("gamma", numerical.asarray([1.0]), numerical.asarray([1.0]))
        with self.assertRaises(TypeError):
            numerical._numerical._real_binary("hypot", numerical.asarray([1.0]), numerical.asarray([1.0], "float32"))


@unittest.skipIf(np is None, "NumPy is not installed")
class AgainstNumPy(unittest.TestCase):
    def assertSame(self, got, want, name, label):
        """To the bit, or for a function of C's library the same NaNs, infinities and zeros (a zero's sign too) and
        every other value within two ulps."""
        if name not in _ROUNDED:
            self.assertEqual(_bits(got), _bits(want), label)
            return
        self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape), label)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=label)
        exact = ~np.isfinite(want) | (want == 0)
        self.assertEqual(_bits(np.where(exact, got, 0)), _bits(np.where(exact, want, 0)), label)
        finite = ~exact & ~np.isnan(want)
        np.testing.assert_array_max_ulp(got[finite], want[finite], maxulp=2)

    def setUp(self):
        warnings.simplefilter("ignore")  # NumPy's warnings for its NaNs, infinities and divisions by zero
        self.addCleanup(warnings.resetwarnings)

    def test_binary_edges(self):
        pairs = list(itertools.product(_EDGES, _EDGES))
        for dtype in ("float64", "float32"):
            a = np.array([x for x, _ in pairs], dtype)
            b = np.array([y for _, y in pairs], dtype)
            for name in _BINARY:
                got = np.asarray(getattr(numerical, name)(numerical.asarray(a), numerical.asarray(b)))
                want = getattr(np, name)(a, b)
                if name in ("fmax", "fmin"):  # two zeros: NumPy's sign that of the zero its loop meets first on x86-64
                    zeros = (a == 0) & (b == 0)  # (IEEE 754-2019's order here, NumPy's on arm64: test_values)
                    got, want = np.where(zeros, 0.0, got).astype(dtype), np.where(zeros, 0.0, want).astype(dtype)
                self.assertSame(got, want, name, f"{name} {dtype}")

    def test_binary_broadcast_and_random(self):
        rng = np.random.default_rng(11)
        a = rng.normal(0, 30, (7, 1, 9))
        b = rng.normal(0, 30, (5, 1))
        for name in _BINARY:
            got = np.asarray(getattr(numerical, name)(numerical.asarray(a), numerical.asarray(b)))
            self.assertSame(got, getattr(np, name)(a, b), name, name)

    def test_integers(self):
        a = np.array([7, -7, 7, -7, 0, 5, -2**63, 3, 2**62])
        b = np.array([2, 2, -2, -2, 3, 0, -1, 7, -3])
        for name in _BINARY:
            got = np.asarray(getattr(numerical, name)(numerical.asarray(a, "int64"), numerical.asarray(b, "int64")))
            self.assertSame(got, getattr(np, name)(a, b), name, name)

    def test_angle_nan_to_num(self):
        z = np.array([1 + 1j, -1 + 0j, complex(-1, -0.0), 0j, complex(-0.0, 0.0), complex(np.nan, 1), -3j])
        for dtype in ("complex128", "complex64"):
            for deg in (False, True):
                got = np.asarray(numerical.angle(numerical.asarray(z.astype(dtype)), deg=deg))
                self.assertSame(got, np.angle(z.astype(dtype), deg=deg), "angle", f"{dtype} {deg}")
        r = np.array([1.0, -1.0, -0.0, 0.0, np.nan, -np.inf])
        self.assertSame(np.asarray(numerical.angle(numerical.asarray(r))), np.angle(r), "angle", "real")
        cases = [np.array([np.nan, np.inf, -np.inf, 1.0, -0.0]), np.array([np.nan, np.inf, -np.inf], np.float32),
                 np.array([complex(np.nan, -np.inf), complex(-0.0, np.nan), complex(np.inf, -0.0)]),
                 np.array([complex(np.inf, -0.0), complex(np.nan, 2)], np.complex64)]
        for x in cases:
            for kwargs in ({}, {"nan": 7.0, "posinf": 1.0, "neginf": -2.0}):
                got = np.asarray(numerical.nan_to_num(numerical.asarray(x), **kwargs))
                self.assertEqual(got.tobytes(), np.nan_to_num(x, **kwargs).tobytes(), f"{x.dtype} {kwargs}")
                self.assertEqual(got.dtype, x.dtype)

    def test_unwrap(self):
        rng = np.random.default_rng(5)
        for dtype in ("float64", "float32"):
            x = np.cumsum(rng.normal(0, 2.5, (5, 40)), axis=1).astype(dtype)
            x[0, 3] = x[0, 2] + np.float32(np.pi)  # a jump of half a period, kept positive
            x[1, 5] = x[1, 4] - np.pi
            for axis in (0, 1, -1):
                for kwargs in ({}, {"discont": 1.0}, {"period": 360.0}, {"period": 4}):
                    got = np.asarray(numerical.unwrap(numerical.asarray(x), axis=axis, **kwargs))
                    want = np.unwrap(x, axis=axis, **kwargs)
                    self.assertEqual(_bits(got), _bits(want), f"{dtype} {axis} {kwargs}")

    def test_float_power(self):
        a = np.array([2, -3, 0, 5])
        b = np.array([-1, 2, 3, -2])
        got = np.asarray(numerical.float_power(numerical.asarray(a, "int64"), numerical.asarray(b, "int64")))
        self.assertEqual(_bits(got), _bits(np.float_power(a, b)))


if __name__ == "__main__":
    unittest.main()
