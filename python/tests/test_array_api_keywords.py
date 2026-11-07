"""The array API standard's keywords, against NumPy's answers where NumPy 2 has them: asarray's and astype's `copy` and
`device`, the creation functions' `device`, the `dtype` sums, means and products accumulate in, var's and std's
`correction`, sort's and argsort's `descending`, clip's `min` and `max`, the sign of a complex number, and
`__array_api_version__`. Deterministic; the comparisons skipped without NumPy."""
import math
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


class WithoutNumPy(unittest.TestCase):
    def test_version_and_devices(self):
        self.assertIn(numerical.__array_api_version__, ("2023.12", "2024.12"))
        t = numerical.asarray([1.0, 2.0])
        self.assertIs(t.__array_namespace__(api_version=numerical.__array_api_version__), numerical)
        made = [numerical.zeros(2, device="cpu"), numerical.ones(2, device="cpu"), numerical.full(2, 3.0, device="cpu"),
                numerical.eye(2, device="cpu"), numerical.arange(2, device="cpu"),
                numerical.zeros_like(t, device="cpu"), numerical.ones_like(t, device="cpu"),
                numerical.empty_like(t, device="cpu"), numerical.full_like(t, 2.0, device="cpu"),
                numerical.asarray([1.0], device="cpu"), numerical.astype(t, "float32", device="cpu"),
                t.astype("float32", device=t.device)]
        self.assertTrue(all(m.device == "cpu" for m in made))
        calls = [lambda: numerical.zeros(2, device="gpu"), lambda: numerical.ones(2, device="gpu"),
                 lambda: numerical.full(2, 1.0, device="gpu"), lambda: numerical.eye(2, device="gpu"),
                 lambda: numerical.arange(2, device="gpu"), lambda: numerical.zeros_like(t, device="gpu"),
                 lambda: numerical.ones_like(t, device="gpu"), lambda: numerical.empty_like(t, device="gpu"),
                 lambda: numerical.full_like(t, 1.0, device="gpu"), lambda: numerical.asarray(t, device="gpu"),
                 lambda: numerical.astype(t, "float32", device="gpu")]
        for call in calls:
            with self.assertRaises(ValueError):
                call()

    def test_asarray_copy(self):
        t = numerical.asarray([1.0, 2.0])
        self.assertIs(numerical.asarray(t, copy=False), t)  # its elements shared, never written into
        self.assertIs(numerical.asarray(t, dtype="float64", copy=False), t)
        self.assertEqual(numerical.asarray(t, copy=True).tolist(), [1.0, 2.0])
        self.assertEqual(numerical.asarray([3.0], copy=None).tolist(), [3.0])
        for obj, dtype in (([1.0, 2.0], None), (t, "float32"), (2.5, None)):
            with self.subTest(obj=obj, dtype=dtype):
                with self.assertRaises(ValueError):
                    numerical.asarray(obj, dtype=dtype, copy=False)
        self.assertIs(numerical.astype(t, "float64", copy=False), t)
        self.assertEqual(t.astype("float32", copy=True).dtype, "float32")

    def test_descending_keeps_equal_elements_in_order(self):
        values = [2.0, -0.0, float("nan"), 0.0, 5.0, 2.0, -1.0]
        down = numerical.sort(values, descending=True).tolist()
        self.assertTrue(math.isnan(down[0]))
        self.assertEqual(down[1:], [5.0, 2.0, 2.0, -0.0, 0.0, -1.0])
        self.assertEqual([math.copysign(1.0, v) for v in down[4:6]], [-1.0, 1.0])  # -0 before +0, as they came
        self.assertEqual(numerical.argsort(values, descending=True).tolist(), [2, 4, 0, 5, 1, 3, 6])
        self.assertEqual(numerical.asarray(values).argsort(descending=True).tolist(), [2, 4, 0, 5, 1, 3, 6])
        matrix = numerical.asarray([[1.0, 3.0, 3.0], [2.0, 2.0, 0.0]])
        self.assertEqual(numerical.argsort(matrix, axis=1, descending=True).tolist(), [[1, 2, 0], [0, 1, 2]])
        self.assertEqual(numerical.argsort(matrix, axis=0, descending=True).tolist(), [[1, 0, 0], [0, 1, 1]])
        self.assertEqual(numerical.argsort(matrix, axis=None, descending=True).tolist(), [1, 2, 3, 4, 0, 5])
        self.assertEqual(numerical.sort(matrix, axis=None, descending=True).tolist(), [3.0, 3.0, 2.0, 2.0, 1.0, 0.0])
        self.assertEqual(numerical.sort([True, False, True], descending=True).tolist(), [True, True, False])
        self.assertEqual(numerical.argsort([1, 3, 1], descending=True).tolist(), [1, 0, 2])

    def test_correction_and_clip_bounds(self):
        values = [1.0, 2.0, 4.0]
        self.assertEqual(numerical.var(values, correction=1), numerical.var(values, ddof=1))
        self.assertEqual(numerical.std(values, correction=1.5), numerical.std(values, ddof=1.5))
        self.assertEqual(numerical.asarray(values).var(correction=1), numerical.var(values, ddof=1))
        for function in (numerical.var, numerical.std):
            with self.assertRaises(ValueError):
                function(values, ddof=1, correction=1)
        self.assertEqual(numerical.clip(values, min=2.0).tolist(), [2.0, 2.0, 4.0])
        self.assertEqual(numerical.clip(values, max=2.0).tolist(), [1.0, 2.0, 2.0])
        self.assertEqual(numerical.asarray(values).clip(min=1.5, max=3.0).tolist(), [1.5, 2.0, 3.0])
        self.assertEqual(numerical.clip(values).tolist(), values)  # neither bound: the elements as they are
        with self.assertRaises(ValueError):
            numerical.clip(values, 1.0, min=1.0)
        with self.assertRaises(ValueError):
            numerical.clip(values, None, 3.0, max=3.0)


@unittest.skipIf(np is None, "NumPy not installed")
class AgainstNumPy(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261106)

    def same(self, got, want):
        got, want = np.ascontiguousarray(np.atleast_1d(got)), np.ascontiguousarray(np.atleast_1d(want))
        self.assertEqual(got.dtype, want.dtype)
        np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8))  # to the bit

    def test_dtype_the_elements_accumulate_in(self):
        single = self.rng.standard_normal(3001).astype(np.float32) * 1e3
        # under a thousand elements: NumPy's 32-bit Windows build ends a longer float32 sum on other last bits
        double = self.rng.standard_normal((30, 31)) * 1e3
        whole = np.arange(1, 9, dtype=np.int64) * 10 ** 17
        for name in ("sum", "mean", "prod"):
            # along the last axis: NumPy adds another one row after row (a documented difference)
            for array, dtype, axis in ((single, np.float64, None), (double, np.float32, 1), (double, np.float32, None),
                                       (whole, np.float64, None), (single[:20], np.complex128, None)):
                with self.subTest(name=name, dtype=dtype, axis=axis, of=array.dtype):
                    want = getattr(np, name)(array, axis=axis, dtype=dtype)
                    got = getattr(numerical, name)(array, axis=axis, dtype=dtype)
                    method = getattr(numerical.asarray(array), name)(axis=axis, dtype=dtype)
                    if axis is None:  # a number: its value in that dtype
                        self.assertEqual(got, want.item())
                        self.assertEqual(method, want.item())
                    else:
                        self.same(got, want)
                        self.same(method, want)

    def test_complex_sign_as_numpy_2(self):
        parts = [0.0, -0.0, 1.0, -2.5, 3e-320, 1e308, 1.7e308, math.inf, -math.inf, math.nan]
        z = np.array([complex(a, b) for a in parts for b in parts])
        with np.errstate(all="ignore"):
            want = np.sign(z)
        for dtype in (np.complex128, np.complex64):
            with self.subTest(dtype=dtype):
                with np.errstate(all="ignore"):
                    expected = np.sign(z.astype(dtype))
                got = numerical.sign(z.astype(dtype))
                self.same(np.asarray(got).real, expected.real)
                self.same(np.asarray(got).imag, expected.imag)
        self.same(np.asarray(numerical.sign(numerical.asarray(z))), want)
        self.assertEqual(numerical.sign(3 + 4j).tolist(), 0.6 + 0.8j)
        self.same(np.asarray(numerical.sign(np.array([-2.0, 0.0, 3.0]))), np.array([-1.0, 0.0, 1.0]))

    def test_keywords_numpy_shares(self):
        values = self.rng.standard_normal((4, 5))
        self.same(numerical.var(values, axis=0, correction=1), np.var(values, axis=0, correction=1))
        self.same(numerical.std(values, correction=2), np.float64(np.std(values, correction=2)))
        self.same(numerical.clip(values, min=-0.5, max=0.5), np.clip(values, min=-0.5, max=0.5))
        self.same(numerical.clip(values), np.clip(values))


if __name__ == "__main__":
    unittest.main()
