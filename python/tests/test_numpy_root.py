"""NumPy's root functions NumericAL composes of its natives: shapes (the *_like family, copy, ravel, expand_dims,
atleast_*d, flip, swapaxes, moveaxis, column_stack), element-wise arithmetic and comparisons (abs, negative, positive,
square, divide, maximum, minimum, clip, isclose, allclose), products (dot, inner, vdot) and spreads (ptp, var, std),
each against NumPy's own answer. Deterministic; skipped without NumPy."""
import unittest
import warnings

import numerical

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy not installed")
class NumpyRoot(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261101)

    def same(self, got, expected, rtol=0.0):
        expected = np.asarray(expected)
        got = np.asarray(got)
        self.assertEqual(got.shape, expected.shape)
        if rtol:
            np.testing.assert_allclose(got, expected, rtol=rtol, atol=0)
        else:
            np.testing.assert_array_equal(got, expected)

    def test_shapes(self):
        a = self.rng.standard_normal((2, 3, 4))
        for like, expected in ((numerical.zeros_like, np.zeros_like), (numerical.ones_like, np.ones_like)):
            self.same(like(a), expected(a))
            self.same(like(a, shape=(5,)), expected(a, shape=(5,)))
        self.assertEqual(numerical.zeros_like(a, dtype="complex128").dtype, "complex128")
        self.same(numerical.empty_like(a).shape, a.shape)
        self.same(numerical.full_like(a, 2.5), np.full_like(a, 2.5))
        t = numerical.asarray(a)
        self.assertIs(numerical.copy(t), t)
        self.same(numerical.copy(a), a)
        self.same(numerical.ravel(a), np.ravel(a))
        for axis in (0, 1, -1, (0, 2), (0, 4)):
            with self.subTest(axis=axis):
                self.same(numerical.expand_dims(a, axis), np.expand_dims(a, axis))
        with self.assertRaises(ValueError):
            numerical.expand_dims(a, (1, 1))
        self.same(numerical.atleast_1d(3.0), np.atleast_1d(3.0))
        self.same(numerical.atleast_2d(np.arange(3.0)), np.atleast_2d(np.arange(3.0)))
        for x in (2.0, np.arange(3.0), np.ones((2, 3))):
            self.same(numerical.atleast_3d(x), np.atleast_3d(x))
        one, two = numerical.atleast_1d(1.0, np.ones((2, 2)))
        self.assertEqual((one.shape, two.shape), ((1,), (2, 2)))
        for axis in (None, 0, 2, (0, 1), -1):
            with self.subTest(flip=axis):
                self.same(numerical.flip(a, axis), np.flip(a, axis))
        self.same(numerical.flipud(a), np.flipud(a))
        self.same(numerical.fliplr(a), np.fliplr(a))
        self.same(numerical.flip(numerical.asarray(2.0)), np.flip(2.0))
        with self.assertRaises(ValueError):
            numerical.fliplr(np.arange(3.0))
        with self.assertRaises(ValueError):
            numerical.flipud(numerical.asarray(1.0))
        self.same(numerical.swapaxes(a, 0, 2), np.swapaxes(a, 0, 2))
        self.same(numerical.swapaxes(a, 1, 1), a)
        self.same(t.swapaxes(0, 1), a.swapaxes(0, 1))
        for source, destination in ((0, -1), (-1, 0), ((0, 1), (-1, -2)), ([2], [0])):
            with self.subTest(moveaxis=(source, destination)):
                self.same(numerical.moveaxis(a, source, destination), np.moveaxis(a, source, destination))
        with self.assertRaises(ValueError):
            numerical.moveaxis(a, (0, 1), 0)
        vectors = [np.arange(3.0), np.ones((3, 2))]
        self.same(numerical.column_stack(vectors), np.column_stack(vectors))

    def test_element_wise(self):
        x = self.rng.standard_normal((3, 4))
        x[0, 0], x[1, 1], x[2, 2] = np.nan, 0.0, -0.0
        y = self.rng.standard_normal((3, 4))
        y[0, 1], y[1, 1], y[2, 2] = np.nan, -0.0, 0.0
        self.same(numerical.abs(x), np.abs(x))
        self.same(numerical.negative(x), np.negative(x))
        self.same(numerical.positive(x), np.positive(x))
        self.same(numerical.square(x), np.square(x))
        self.same(numerical.divide(x, y), np.divide(x, y))
        self.same(numerical.true_divide(2.0, y), np.true_divide(2.0, y))
        self.same(numerical.divide(x, 2.0), np.divide(x, 2.0))
        self.same(numerical.divide(x.tolist(), y.tolist()), np.divide(x, y))
        zeros = (x == 0) & (y == 0)  # their signs as IEEE 754-2019 orders them; NumPy's are its platform's (x86: y's)
        for function, expected, negative_zero in ((numerical.maximum, np.maximum, np.signbit(x) & np.signbit(y)),
                                                  (numerical.minimum, np.minimum, np.signbit(x) | np.signbit(y))):
            got = np.asarray(function(x, y))
            self.same(got, expected(x, y))
            np.testing.assert_array_equal(np.signbit(got)[zeros], negative_zero[zeros])
            self.same(function(x, 0.25), expected(x, 0.25))
        self.same(numerical.clip(x, -0.5, 0.5), np.clip(x, -0.5, 0.5))
        self.same(numerical.clip(x, None, 0.5), np.clip(x, None, 0.5))
        self.same(numerical.clip(x, -0.5, None), np.clip(x, -0.5, None))
        self.same(numerical.asarray(x).clip(-0.5, 0.5), x.clip(-0.5, 0.5))
        with self.assertRaises(ValueError):
            numerical.clip(x, None, None)
        with self.assertRaises(TypeError):
            numerical.maximum(np.ones(2) * 1j, 1.0)
        with self.assertRaises(TypeError):
            numerical.negative(np.array([True, False]))
        with self.assertRaises(TypeError):
            numerical.positive(np.array([True]))
        a = np.array([1.0, 1.0 + 1e-9, np.inf, -np.inf, np.nan, 1.0, 1e10])
        b = np.array([1.0, 1.0, np.inf, np.inf, np.nan, np.nan, 1.00001e10])
        for equal_nan in (False, True):
            self.same(numerical.isclose(a, b, equal_nan=equal_nan), np.isclose(a, b, equal_nan=equal_nan))
            self.assertEqual(numerical.allclose(a, b, equal_nan=equal_nan), np.allclose(a, b, equal_nan=equal_nan))
        self.same(numerical.isclose(a, b, rtol=0.0, atol=0.1), np.isclose(a, b, rtol=0.0, atol=0.1))
        self.assertTrue(numerical.allclose(x[1:], x[1:] + 1e-12))

    def test_products(self):
        for sa, sb in (((4,), (4,)), ((3, 4), (4,)), ((3, 4), (4, 5)), ((4,), (4, 5)), ((2, 3, 4), (5, 4, 6)), ((3,), ())):
            a = self.rng.standard_normal(sa)
            b = self.rng.standard_normal(sb) if sb else 2.5
            with self.subTest(dot=(sa, sb)):
                self.same(numerical.dot(a, b), np.dot(a, b), rtol=1e-13)
        for sa, sb in (((4,), (4,)), ((3, 4), (4,)), ((3, 4), (5, 4)), ((2, 3, 4), (5, 4)), ((3,), ())):
            a = self.rng.standard_normal(sa)
            b = self.rng.standard_normal(sb) if sb else 2.5
            with self.subTest(inner=(sa, sb)):
                self.same(numerical.inner(a, b), np.inner(a, b), rtol=1e-13)
        a = self.rng.standard_normal((2, 3)) + 1j * self.rng.standard_normal((2, 3))
        b = self.rng.standard_normal((2, 3)) + 1j * self.rng.standard_normal((2, 3))
        self.assertAlmostEqual(complex(numerical.vdot(a, b)), complex(np.vdot(a, b)), places=13)
        self.same(numerical.asarray(a).dot(b.T), a.dot(b.T), rtol=1e-13)
        with self.assertRaises(ValueError):
            numerical.vdot(np.ones(3), np.ones(4))

    def test_spreads(self):
        x = self.rng.standard_normal((4, 6))
        for axis in (None, 0, 1, (0, 1)):
            for keepdims in (False, True):
                with self.subTest(axis=axis, keepdims=keepdims):
                    self.same(numerical.ptp(x, axis=axis, keepdims=keepdims), np.ptp(x, axis=axis, keepdims=keepdims))
                    for ddof in (0, 1):
                        self.same(numerical.var(x, axis=axis, ddof=ddof, keepdims=keepdims),
                                  np.var(x, axis=axis, ddof=ddof, keepdims=keepdims), rtol=1e-13)
                        self.same(numerical.std(x, axis=axis, ddof=ddof, keepdims=keepdims),
                                  np.std(x, axis=axis, ddof=ddof, keepdims=keepdims), rtol=1e-13)
        z = x + 1j * self.rng.standard_normal((4, 6))
        self.same(numerical.var(z, axis=0), np.var(z, axis=0), rtol=1e-13)
        self.assertEqual(numerical.var(z).__class__, float)
        self.same(numerical.var(x, dtype="float32"), np.var(x.astype(np.float32)), rtol=1e-6)
        self.same(numerical.var(np.array([True, False, True])), np.var(np.array([True, False, True])), rtol=1e-15)
        t = numerical.asarray(x)
        self.same(t.var(axis=1), x.var(axis=1), rtol=1e-13)
        self.same(t.std(), x.std(), rtol=1e-13)
        self.same(t.ptp(axis=0), np.ptp(x, axis=0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertTrue(np.isnan(numerical.var(np.ones(2), ddof=2)))
            self.assertTrue(np.all(np.isnan(np.asarray(numerical.var(np.ones((2, 3)), axis=0, ddof=2)))))
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))


if __name__ == "__main__":
    unittest.main()
