"""NumPy's root functions of the second set against NumPy's own answers: the real elementary functions (within a few
units in the last place: both sides take their platform's libm or their own), rounding, prod, cumsum, cumprod, sort
and diff. Deterministic; skipped without NumPy."""
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


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
        for n, axis in ((1, -1), (2, -1), (1, 0), (0, 1), (3, 1)):
            with self.subTest(n=n, axis=axis):
                self.same(numerical.diff(a, n=n, axis=axis), np.diff(a, n=n, axis=axis))
        with self.assertRaises(ValueError):
            numerical.diff(a, n=-1)
        with self.assertRaises(ValueError):
            numerical.diff(3.0)


if __name__ == "__main__":
    unittest.main()
