"""median, quantile, percentile and their NaN-ignoring forms, the NaN-ignoring reductions, average, cov and corrcoef
against NumPy's answers: to the bit, but for sums along a leading axis (pairwise per column here, row after row in
NumPy's), within a few ulps there. Deterministic; the NumPy comparisons skipped without NumPy."""
import math
import unittest
import warnings

import numerical

try:
    import numpy as np
except ImportError:
    np = None

_METHODS = ("inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen",
            "weibull", "linear", "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint", "nearest")


class WithoutNumPy(unittest.TestCase):
    def test_values(self):
        nan = math.nan
        self.assertEqual(numerical.median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(numerical.median([[1.0, 4.0], [3.0, 2.0]], axis=0).tolist(), [2.0, 3.0])
        self.assertTrue(math.isnan(numerical.median([1.0, nan])))
        self.assertEqual(numerical.nanmedian([1.0, nan, 3.0]), 2.0)
        self.assertEqual(str(numerical.median([-0.0])), "0.0")  # NumPy's mean of the middle one: from +0
        self.assertEqual(numerical.median(numerical.asarray([1, 2], "int64")), 1.5)
        self.assertEqual(numerical.quantile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertEqual(numerical.quantile([1.0, 2.0, 3.0, 4.0], [0.0, 1.0]).tolist(), [1.0, 4.0])
        self.assertEqual(numerical.quantile([1.0, 2.0, 3.0, 4.0], 0.5, method="lower"), 2.0)
        self.assertEqual(numerical.percentile([1.0, 2.0, 3.0, 4.0], 50, method="higher"), 3.0)
        self.assertEqual(numerical.quantile(numerical.asarray([1.0, 2.0], "float32"), 0.5, keepdims=True).dtype,
                         "float32")
        self.assertEqual(numerical.quantile([[1.0, 2.0], [3.0, 4.0]], [0.5], axis=1, keepdims=True).shape, (1, 2, 1))
        self.assertEqual(numerical.nanquantile([nan, 1.0, 3.0], 0.5), 2.0)
        self.assertEqual(numerical.nansum([1.0, nan]), 1.0)
        self.assertEqual(numerical.nanprod([2.0, nan, 3.0]), 6.0)
        self.assertEqual(numerical.nancumsum([1.0, nan, 2.0]).tolist(), [1.0, 1.0, 3.0])
        self.assertEqual(numerical.nancumprod([2.0, nan, 2.0]).tolist(), [2.0, 2.0, 4.0])
        self.assertEqual(numerical.nanmax([nan, 1.0, 3.0]), 3.0)
        self.assertEqual(numerical.nanmin([[nan, 1.0], [2.0, nan]], axis=0).tolist(), [2.0, 1.0])
        self.assertEqual(numerical.nanargmax([nan, 1.0, 3.0]), 2)
        self.assertEqual(numerical.nanargmin([nan, 1.0, 3.0]), 1)
        self.assertEqual(numerical.nanmean([1.0, nan, 3.0]), 2.0)
        self.assertEqual(numerical.nanvar([1.0, nan, 3.0]), 1.0)
        self.assertEqual(numerical.nanstd([1.0, nan, 3.0], ddof=1), math.sqrt(2.0))
        self.assertEqual(numerical.average([1.0, 2.0, 4.0], weights=[3.0, 1.0, 0.0]), 1.25)
        self.assertEqual(numerical.average([1.0, 2.0], returned=True), (1.5, 2.0))
        self.assertEqual(numerical.cov([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]).tolist(), [[1.0, -1.0], [-1.0, 1.0]])
        self.assertEqual(numerical.cov([1.0, 2.0, 3.0]), 1.0)
        self.assertEqual(numerical.cov(numerical.zeros((0, 3))).shape, (0, 0))
        self.assertEqual(numerical.corrcoef([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]).tolist(), [[1.0, -1.0], [-1.0, 1.0]])
        self.assertEqual(numerical.mean(numerical.asarray([1, 2], "int64"), keepdims=True).dtype, "float64")
        self.assertEqual(numerical.any(numerical.asarray([0.0, 2.0]), axis=()).tolist(), [False, True])
        self.assertEqual(str(numerical.sum(numerical.full(9, -0.0))), "0.0")

    def test_warnings_and_errors(self):
        nan = math.nan
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertTrue(math.isnan(numerical.nanmax([nan, nan])))
            self.assertTrue(math.isnan(numerical.nanmean([nan])))
            self.assertTrue(math.isnan(numerical.nanvar([1.0, nan], ddof=1)))
            self.assertTrue(math.isnan(numerical.nanmedian([nan, nan])))
            self.assertTrue(math.isnan(numerical.nanquantile([nan], 0.5)))
            self.assertTrue(math.isnan(numerical.median(numerical.zeros(0))))
            self.assertTrue(math.isnan(numerical.cov([1.0])))
        self.assertTrue(all(issubclass(w.category, RuntimeWarning) for w in caught))
        self.assertEqual(len(caught), 7)
        cases = [
            (ValueError, lambda: numerical.nanargmax([nan, nan])),
            (ValueError, lambda: numerical.quantile([1.0], 1.5)),
            (ValueError, lambda: numerical.percentile([1.0], -1)),
            (ValueError, lambda: numerical.quantile([1.0], 0.5, method="mean")),
            (ValueError, lambda: numerical.quantile([1.0], [[[0.5]]])),
            (TypeError, lambda: numerical.quantile([1j], 0.5)),
            (NotImplementedError, lambda: numerical.quantile([1.0], 0.5, method="inverted_cdf", weights=[1.0])),
            (TypeError, lambda: numerical.median([1.0], out=numerical.zeros(1))),
            (TypeError, lambda: numerical.nansum([1.0], out=numerical.zeros(1))),
            (ValueError, lambda: numerical.cov(numerical.zeros((2, 2, 2)))),
            (ValueError, lambda: numerical.cov([1.0, 2.0], ddof=1.5)),
            (TypeError, lambda: numerical.cov([1.0, 2.0], fweights=[1.5, 1.0])),
            (RuntimeError, lambda: numerical.cov([1.0, 2.0], fweights=[1.0])),
            (RuntimeError, lambda: numerical.cov([1.0, 2.0], aweights=[[1.0, 1.0]])),
            (ValueError, lambda: numerical.cov([1.0, 2.0], aweights=[-1.0, 1.0])),
            (TypeError, lambda: numerical.average([[1.0, 2.0]], weights=[1.0, 1.0])),
            (ValueError, lambda: numerical.average([[1.0, 2.0]], axis=0, weights=[1.0, 1.0])),
            (ZeroDivisionError, lambda: numerical.average([1.0, 2.0], weights=[0.0, 0.0])),
        ]
        for error, call in cases:
            with self.assertRaises(error):
                call()


def _bits(x):
    x = np.asarray(x)
    if x.dtype.kind in "fc":
        x = np.where(np.isnan(x), np.nan, x)
    return x.dtype, x.shape, x.tobytes()


@unittest.skipIf(np is None, "NumPy is not installed")
class AgainstNumPy(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.addCleanup(warnings.resetwarnings)

    def assertSame(self, got, want, label, exact=True):
        if isinstance(got, (int, float)) and not isinstance(got, bool):  # NumPy's scalar of the dtype, a number here
            want = np.asarray(want)
            self.assertEqual(want.shape, (), label)
            self.assertIs(type(got), int if want.dtype.kind in "iu" else float, label)
            if math.isnan(got):
                self.assertTrue(np.isnan(want), label)
            else:  # the dtype's value exactly, a zero's sign too
                self.assertEqual(np.asarray(got, want.dtype).tobytes(), want.tobytes(), label)
            return
        if exact:
            self.assertEqual(_bits(got), _bits(want), label)
        else:
            got, want = np.asarray(got), np.asarray(want)
            self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype), label)
            np.testing.assert_allclose(got, want, rtol=1e-5 if got.dtype == np.float32 else 1e-13, err_msg=label)

    def test_nan_reductions(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(4, 5, 6))
        x[0, 1, 2] = np.nan
        x[2, :, 3] = np.nan
        x[1, 0, :] = np.nan
        for data in (x, x.astype(np.float32), np.round(x * 10), rng.integers(-50, 50, (4, 5, 6))):
            t = numerical.asarray(data, "int64") if data.dtype.kind == "i" else numerical.asarray(data)
            for axis in (None, 0, 1, -1, (0, 2), (1, 2), ()):
                for keepdims in (False, True):
                    for name in ("nansum", "nanprod", "nanmax", "nanmin", "nanmean", "nanvar", "nanstd", "median",
                                 "nanmedian"):
                        if axis == () and name in ("nanmax", "nanmin"):
                            continue
                        leading = name in ("nansum", "nanmean", "nanvar", "nanstd") and axis in (0, 1, (0, 2))
                        self.assertSame(getattr(numerical, name)(t, axis=axis, keepdims=keepdims),
                                        getattr(np, name)(data, axis=axis, keepdims=keepdims),
                                        f"{name} {data.dtype} {axis} {keepdims}", exact=not leading)
            for axis in (None, 0, 1, -1):
                for name in ("nancumsum", "nancumprod"):
                    self.assertSame(getattr(numerical, name)(t, axis=axis), getattr(np, name)(data, axis=axis),
                                    f"{name} {axis}")
            for axis in (0,) if data.dtype.kind == "f" else (0, 1, -1):
                for name in ("nanargmax", "nanargmin"):
                    self.assertSame(getattr(numerical, name)(t, axis=axis), getattr(np, name)(data, axis=axis),
                                    f"{name} {axis}")
            if data.dtype.kind == "f":
                for axis in (1, -1):  # x[2, :, 3] and x[1, 0, :] hold NaNs alone
                    with self.assertRaises(ValueError):
                        numerical.nanargmax(t, axis=axis)

    def test_quantiles(self):
        rng = np.random.default_rng(3)
        clean = rng.normal(size=(5, 7))
        holed = clean.copy()
        holed[1, 3] = np.nan
        holed[:, 5] = np.nan
        cases = (clean, clean.astype(np.float32), holed, holed.astype(np.float32), rng.integers(-9, 9, (5, 7)),
                 np.array([-1.0, -0.0]), np.array([3.0]))
        for data in cases:
            t = numerical.asarray(data, "int64") if data.dtype.kind == "i" else numerical.asarray(data)
            for method in _METHODS:
                for q in (0.5, 0.0, 1.0, 0.37, [0.1, 0.5, 0.99], np.array([[0.2, 0.8], [0.0, 1.0]]), 1, 0):
                    for axis in ((None, 0, 1, (0, 1)) if data.ndim == 2 else (None, 0)):
                        for name in ("quantile", "nanquantile"):
                            got = getattr(numerical, name)(t, numerical.asarray(q) if isinstance(q, np.ndarray) else q,
                                                           axis=axis, method=method)
                            self.assertSame(got, getattr(np, name)(data, q, axis=axis, method=method),
                                            f"{name} {method} {q!r} {axis} {data.dtype}")
            for q in (50, 12.5, [10, 90]):
                self.assertSame(numerical.percentile(t, q, axis=0), np.percentile(data, q, axis=0), f"percentile {q}")
                self.assertSame(numerical.nanpercentile(t, q, axis=0, keepdims=True),
                                np.nanpercentile(data, q, axis=0, keepdims=True), f"nanpercentile {q}")

    def test_average_cov_corrcoef(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=(3, 4, 5))
        t = numerical.asarray(a)
        w5, wf = rng.random(5), rng.random((3, 4, 5))
        self.assertSame(numerical.average(t), np.average(a), "average")
        self.assertSame(numerical.average(t, axis=2, weights=w5), np.average(a, axis=2, weights=w5), "weights")
        self.assertSame(numerical.average(t, weights=wf), np.average(a, weights=wf), "full weights")
        w34 = rng.random((3, 4))
        self.assertSame(numerical.average(t, axis=(1, 0), weights=w34.T), np.average(a, axis=(1, 0), weights=w34.T),
                        "weights over two axes", exact=False)
        got, scale = numerical.average(t, axis=2, weights=w5, returned=True)
        want, want_scale = np.average(a, axis=2, weights=w5, returned=True)
        self.assertSame(got, want, "returned")
        self.assertSame(scale, want_scale, "returned weights")
        got, scale = numerical.average(t, axis=1, returned=True, keepdims=True)
        want, want_scale = np.average(a, axis=1, returned=True, keepdims=True)
        self.assertSame(scale, want_scale, "returned count")
        x, y = rng.normal(size=(4, 30)), rng.normal(size=(2, 30))
        fw, aw = rng.integers(0, 4, 30), rng.random(30)
        for kwargs in ({}, {"bias": True}, {"ddof": 3}, {"rowvar": False}, {"fweights": fw}, {"aweights": aw},
                       {"fweights": fw, "aweights": aw, "ddof": 2}, {"fweights": fw, "aweights": aw, "ddof": 0}):
            ours = {k: numerical.asarray(v, "int64") if k == "fweights" else v for k, v in kwargs.items()}
            self.assertSame(numerical.cov(numerical.asarray(x), **ours), np.cov(x, **kwargs), f"cov {list(kwargs)}",
                            exact=False)  # a matrix product: OpenBLAS's order is not ours
        self.assertSame(numerical.cov(numerical.asarray(x), numerical.asarray(y)), np.cov(x, y), "cov y", exact=False)
        self.assertSame(numerical.cov(numerical.asarray(x, "float32")), np.cov(x.astype(np.float32)), "float32",
                        exact=False)
        self.assertSame(numerical.cov(numerical.asarray(x.astype(np.float32)), dtype="float32"),
                        np.cov(x.astype(np.float32), dtype=np.float32), "dtype", exact=False)
        z = x[:2] + 1j * x[2:]
        self.assertSame(numerical.cov(numerical.asarray(z)), np.cov(z), "complex", exact=False)
        self.assertSame(numerical.corrcoef(numerical.asarray(z)), np.corrcoef(z), "complex corrcoef", exact=False)
        self.assertSame(numerical.corrcoef(numerical.asarray(x), numerical.asarray(y)), np.corrcoef(x, y), "corrcoef",
                        exact=False)
        self.assertLessEqual(float(numerical.max(numerical.corrcoef(numerical.asarray(np.vstack([x[0], x[0]]))))), 1.0)


if __name__ == "__main__":
    unittest.main()
