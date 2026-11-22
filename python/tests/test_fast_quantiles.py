"""The linear quantiles computed in C (_numerical._quantile_linear): the bits of the tensor operations they replace, and
the converting path for everything else."""

import unittest

import numerical
from numerical import _numerical

try:
    import numpy as np
except ImportError:  # pragma: no cover - the CI images install NumPy
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class LinearQuantiles(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.native = _numerical._quantile_linear

    def through_tensors(self, *args, **kwargs):
        _numerical._quantile_linear = lambda *_: NotImplemented
        try:
            return numerical.quantile(*args, **kwargs)
        finally:
            _numerical._quantile_linear = self.native

    def test_the_lines_quantiles_are_the_tensor_operations_bits(self):
        # float64 and float32 lines of 1 to 100 elements, every axis, infinities, signed zeros and a NaN (its line's
        # answer, the stable sort's last), at weak and Tensor quantiles of the edges (0, 1, just below 1), past eight
        # positions (the line sorted) and drawn, keepdims: the dtype, shape and bytes of the tensor operations'.
        shapes = (((1,), None), ((2,), None), ((100,), None), ((7, 13), 0), ((7, 13), 1), ((7, 13), None),
                  ((3, 4, 5), (0, 2)), ((50, 40), -1))
        quantiles = (0.0, 1.0, 0.5, 0.3, 1 - 1e-16, [0.1, 0.5, 0.9], [0.0, 1.0], list(self.rng.random(7)),
                     numerical.asarray(0.25), numerical.asarray([0.2, 0.7]))
        for dtype in (np.float64, np.float32):
            for shape, axis in shapes:
                for special in (False, True):
                    x = self.rng.standard_normal(shape).astype(dtype)
                    if special:
                        flat = x.reshape(-1)
                        flat[:4] = [np.inf, -np.inf, -0.0, 0.0][:flat.size]
                        if flat.size > 5:
                            flat[self.rng.integers(0, flat.size)] = np.nan
                    t = numerical.asarray(x)
                    for q in quantiles:
                        for keepdims in (False, True):
                            with self.subTest(dtype=np.dtype(dtype).name, shape=shape, axis=axis, special=special, q=q,
                                              keepdims=keepdims):
                                got = np.asarray(numerical.quantile(t, q, axis=axis, keepdims=keepdims))
                                want = np.asarray(self.through_tensors(t, q, axis=axis, keepdims=keepdims))
                                self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape))
                                self.assertEqual(got.tobytes(), want.tobytes())
        x = self.rng.standard_normal((9, 31))
        np.testing.assert_array_equal(np.asarray(numerical.percentile(x, [5, 50, 95], axis=1)),
                                      np.percentile(x, [5, 50, 95], axis=1))

    def test_anything_else_is_not_implemented(self):
        # Integers, complex numbers, float32 quantiles, no axis, an empty line, quantiles of rank 2 or not a Tensor:
        # the tensor operations' path.
        t = numerical.asarray(self.rng.standard_normal((3, 4)))
        q = numerical.asarray([0.5])
        for args in ((numerical.asarray([1, 2, 3], "int64"), q, True), (numerical.asarray([1j, 2j]), q, True),
                     (t, numerical.asarray([0.5], "float32"), True), (numerical.asarray(1.0), q, True),
                     (numerical.zeros((3, 0)), q, True), (t, numerical.asarray([[0.5]]), True), (t, [0.5], True),
                     ([1.0, 2.0], q, True)):
            self.assertIs(_numerical._quantile_linear(*args), NotImplemented)
        self.assertEqual(float(numerical.quantile(numerical.asarray([1, 2, 3, 4], "int64"), 0.5)), 2.5)


if __name__ == "__main__":
    unittest.main()
