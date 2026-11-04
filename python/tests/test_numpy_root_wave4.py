"""Indexing by int64 Tensors and take (every mode, empty axes and results), the reductions composed in Python kept in
their dtype (var and std of float32, ptp of int64 wrapping, corrcoef of a constant), nanquantile of nothing,
setdiff1d across dtypes, polymul's leading zeros, roots' real answers and correlate of bools, against NumPy's
answers: to the bit, a full reduction's number equal to NumPy's scalar; with NumPy's errors. Deterministic; the
comparisons skipped without NumPy."""
import math
import unittest
import warnings

import numerical

try:
    import numpy as np
except ImportError:
    np = None

INT64_MIN, INT64_MAX = -2 ** 63, 2 ** 63 - 1


class WithoutNumPy(unittest.TestCase):
    def test_values(self):
        t = numerical.asarray([10.0, 20.0, 30.0, 40.0])
        index = numerical.asarray([[3, -1], [0, 1]], "int64")
        self.assertEqual(t[index].tolist(), [[40.0, 40.0], [10.0, 20.0]])
        m = numerical.reshape(numerical.arange(6.0), (2, 3))
        self.assertEqual(m[numerical.asarray([1, 0], "int64"), numerical.asarray([-1], "int64")].tolist(), [5.0, 2.0])
        self.assertEqual(m[:, numerical.asarray([2, 2], "int64")].tolist(), [[2.0, 2.0], [5.0, 5.0]])
        self.assertEqual(m[numerical.asarray([1], "int64"), [0, 2]].tolist(), [3.0, 5.0])
        self.assertEqual(numerical.take(t, numerical.asarray([5, -6], "int64"), mode="wrap").tolist(), [20.0, 30.0])
        self.assertEqual(numerical.take(t, [9, -9], mode="clip").tolist(), [40.0, 10.0])
        self.assertEqual(numerical.take(m, [True, False], axis=1).tolist(), [[1.0, 0.0], [4.0, 3.0]])
        self.assertEqual(numerical.take(m, -1, axis=0).tolist(), [3.0, 4.0, 5.0])
        self.assertEqual(numerical.take(numerical.zeros((0, 2)), [7], axis=1).shape, (0, 1))
        self.assertEqual(numerical.take(numerical.zeros((2, 0)), 7, axis=0, mode="clip").shape, (0,))
        self.assertEqual(numerical.take(t, numerical.zeros((0, 2), "int64")).shape, (0, 2))
        variance = numerical.var(numerical.asarray([1.0, 2.0, 4.0], "float32"))  # divided in float32
        self.assertEqual(variance, float(numerical.asarray(variance, "float32")))
        self.assertNotEqual(variance, float(numerical.asarray(14 / 9, "float32")))
        self.assertEqual(numerical.std(numerical.asarray([0.0, 2.0], "float32"), keepdims=True).tolist(), [1.0])
        self.assertEqual(numerical.ptp(numerical.asarray([INT64_MIN, INT64_MAX], "int64")), -1)
        self.assertTrue(math.isnan(numerical.corrcoef([1.0, 1.0])))
        self.assertTrue(math.isnan(numerical.nanquantile([], [0.5, 0.9])))
        self.assertEqual(numerical.nanpercentile(numerical.zeros((0, 2)), 50, axis=0, keepdims=True).shape, (1, 2))
        self.assertEqual(numerical.setdiff1d(numerical.asarray([INT64_MIN, INT64_MIN + 1, 5], "int64"), 1.5).tolist(),
                         [INT64_MIN, INT64_MIN + 1, 5])
        self.assertEqual(numerical.polymul([0, 2, 4], [0, 3, 0, -3]).tolist(), [6, 12, -6, -12])
        self.assertEqual(numerical.polymul([0.0, 0.0], [1.0, 2.0]).tolist(), [0.0, 0.0])
        self.assertEqual(numerical.roots([1.0, -3.0, 2.0]).dtype, "float64")
        self.assertEqual(numerical.roots([1.0, 0.0, 1.0]).dtype, "complex128")
        self.assertEqual(numerical.roots([5.0, 0.0]).tolist(), [0.0])

    def test_errors(self):
        t = numerical.asarray([1.0, 2.0])
        cases = [
            (IndexError, lambda: t[numerical.asarray([2], "int64")]),
            (IndexError, lambda: t[numerical.asarray([0, -3], "int64")]),
            (IndexError, lambda: numerical.take(t, [2])),
            (IndexError, lambda: numerical.take(numerical.zeros((3, 0)), [5], axis=0)),
            (IndexError, lambda: numerical.take(numerical.zeros((2, 0)), [0], axis=1, mode="wrap")),
            (IndexError, lambda: numerical.take(numerical.zeros((0, 0)), [0], axis=0, mode="wrap")),
            (TypeError, lambda: numerical.take(t, [0.5])),
            (TypeError, lambda: numerical.take(t, numerical.asarray([0.0]))),
            (ValueError, lambda: numerical.take(t, [0], mode="nearest")),
            (ValueError, lambda: numerical.polymul([[1.0]], [1.0])),
        ]
        for error, call in cases:
            with self.assertRaises(error):
                call()
        with self.assertRaisesRegex(IndexError, "index -3 is out of bounds for axis 0 with size 2"):
            t[numerical.asarray([0, -3, 5], "int64")]


@unittest.skipIf(np is None, "NumPy is not installed")
class AgainstNumPy(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.addCleanup(warnings.resetwarnings)

    def assertSame(self, ours, theirs, label):
        try:
            want = theirs()
        except (IndexError, TypeError, ValueError) as error:
            with self.assertRaises(type(error), msg=label):
                ours()
            return
        got = ours()
        want = np.asarray(want)
        if want.dtype.kind in "iu":  # NumPy's indices are intp (int32 on a 32-bit Windows)
            want = want.astype(np.int64)
        if isinstance(got, (int, float, complex)) and not isinstance(got, bool):  # NumPy's scalar: its very number
            self.assertEqual(want.shape, (), label)
            self.assertTrue(got == want.item() or (got != got and want != want), f"{label}: {got!r} {want!r}")
            return
        got = np.asarray(got)
        self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype), label)
        self.assertEqual(got.tobytes(), want.tobytes(), label)

    @staticmethod
    def numpy_index(index):
        """NumPy's answer, its deprecation of a bound unchecked for an empty result (before 2.3) the error it became."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            try:
                return index()
            except DeprecationWarning as warning:
                raise IndexError(str(warning)) from None

    def test_indexing_by_int64_tensors(self):
        rng = np.random.default_rng(20261104)
        for case in range(400):
            shape = tuple(int(e) for e in rng.integers(0, 5, int(rng.integers(1, 4))))
            a = rng.standard_normal(shape)
            index_shape = tuple(int(e) for e in rng.integers(0, 4, int(rng.integers(0, 3))))
            i = rng.integers(-shape[0] - 1, shape[0] + 1, index_shape)
            t, ti = numerical.asarray(a), numerical.asarray(i, "int64")

            def same(ours, key, label):
                self.assertSame(ours, lambda: self.numpy_index(lambda: a[key]), f"{case} {shape} {label}")

            same(lambda: t[ti], i, f"[{i.tolist()}]")
            if len(shape) >= 2:
                j = rng.integers(-shape[1] - 1, shape[1] + 1, index_shape)
                tj = numerical.asarray(j, "int64")
                same(lambda: t[ti, tj], (i, j), f"[{i.tolist()}, {j.tolist()}]")
                same(lambda: t[:, tj], (slice(None), j), f"[:, {j.tolist()}]")
                if j.size:  # an empty list has its own shape, (0,)
                    same(lambda: t[ti, j.tolist()], (i, j), "tensor and list")
                same(lambda: t[ti, j], (i, j), "tensor and NumPy")
                same(lambda: t[ti, 0], (i, 0), "tensor and integer")

    def test_take(self):
        rng = np.random.default_rng(1104)
        for case in range(400):
            shape = tuple(int(e) for e in rng.integers(0, 5, int(rng.integers(1, 4))))
            a = rng.standard_normal(shape)
            axis = int(rng.integers(-len(shape), len(shape))) if rng.random() < 0.7 else None
            extent = a.size if axis is None else shape[axis]
            index_shape = tuple(int(e) for e in rng.integers(0, 4, int(rng.integers(0, 3))))
            i = rng.integers(-extent - 2, extent + 2, index_shape)
            t = numerical.asarray(a)
            for mode in ("raise", "wrap", "clip"):
                if mode == "wrap" and i.size and not extent:
                    continue  # NumPy loops forever there (numpy#11677)
                for kind, index in (("tensor", numerical.asarray(i, "int64")), ("numpy", i),
                                    ("list", i.tolist() if i.size else None),
                                    ("integer", int(i) if not index_shape else None)):
                    if index is None:
                        continue
                    self.assertSame(lambda: numerical.take(t, index, axis=axis, mode=mode),
                                    lambda: np.take(a, i, axis=axis, mode=mode),
                                    f"{case} {shape} {i.tolist()} {axis} {mode} {kind}")

    def test_reductions_in_their_dtype(self):
        rng = np.random.default_rng(4)
        for dtype in ("float32", "float64", "complex64"):
            x = (rng.standard_normal(1001) * 1000).astype(dtype)
            if dtype == "complex64":
                x = x + 1j * x[::-1]
            t = numerical.asarray(x)
            for ddof in (0, 1):
                self.assertSame(lambda: numerical.var(t, ddof=ddof), lambda: np.var(x, ddof=ddof), f"var {dtype}")
                self.assertSame(lambda: numerical.std(t, ddof=ddof), lambda: np.std(x, ddof=ddof), f"std {dtype}")
        wide = np.array([[18, -8], [INT64_MIN, INT64_MAX]], np.int64)
        for axis in (None, 0, 1, (0, 1)):
            for keepdims in (False, True):
                self.assertSame(lambda: numerical.ptp(numerical.asarray(wide, "int64"), axis, keepdims),
                                lambda: np.ptp(wide, axis, keepdims=keepdims), f"ptp {axis} {keepdims}")
        for x in ([1.0, 1.0], [[1.0, 2.0], [3.0, 3.0]], [2.0]):
            self.assertSame(lambda: numerical.corrcoef(x), lambda: np.corrcoef(x), f"corrcoef {x}")

    def test_quantiles_of_nothing(self):
        for shape, axis in (((0,), None), ((0, 3), 0), ((3, 0), 1), ((0, 3), 1), ((0, 3), None)):
            a = np.zeros(shape)
            for q in (0.5, [0.2, 0.9]):
                for keepdims in (False, True):
                    for name in ("nanquantile", "nanpercentile"):
                        scaled = q if name == "nanquantile" else np.multiply(q, 100).tolist()
                        self.assertSame(lambda: getattr(numerical, name)(a, scaled, axis=axis, keepdims=keepdims),
                                        lambda: getattr(np, name)(a, scaled, axis=axis, keepdims=keepdims),
                                        f"{name} {shape} {axis} {q} {keepdims}")
            self.assertSame(lambda: numerical.nanquantile(a, 0.5, axis=axis, method="inverted_cdf", weights=a),
                            lambda: np.nanquantile(a, 0.5, axis=axis, method="inverted_cdf", weights=a),
                            f"weighted {shape} {axis}")

    def test_sets_and_polynomials(self):
        ints = np.array([INT64_MIN, INT64_MIN + 1, 5, 2 ** 53, 2 ** 53 + 1], np.int64)
        for other in (1.5, [5.0], [float(2 ** 53)], numerical.asarray([INT64_MIN], "int64"), [-9.223372036854776e18]):
            self.assertSame(lambda: numerical.setdiff1d(numerical.asarray(ints, "int64"), other),
                            lambda: np.setdiff1d(ints, np.asarray(other)), f"setdiff1d {other}")
        for a1, a2 in (([0, 2, 4], [0, 3, 0, -3]), ([0.0, 0.0], [1.0, 2.0]), ([], [1, 2]), ([0], [0]),
                       ([0j, 1], [1]), ([True, False], [True])):
            self.assertSame(lambda: numerical.polymul(a1, a2), lambda: np.polymul(a1, a2), f"polymul {a1} {a2}")
        for p in ([1, -3, 2], [1.0, 0.0, 1.0], [5], [5, 0], [0, 0], [1, -3, 2, 0, 0], [2j, 1], [1, 0, -1]):
            self.assertSame(lambda: numerical.roots(p).dtype.__str__(), lambda: str(np.roots(p).dtype), f"roots {p}")
        for a, v in (([True, False, True], [True, True]), ([False, False], [True]), ([True], [False, True, True])):
            for mode in ("valid", "same", "full"):
                ta, tv = numerical.asarray(a, "bool"), numerical.asarray(v, "bool")  # a list of bools reads as float64
                self.assertSame(lambda: numerical.correlate(ta, tv, mode), lambda: np.correlate(a, v, mode),
                                f"correlate {a} {v} {mode}")
                self.assertSame(lambda: numerical.convolve(ta, tv, mode), lambda: np.convolve(a, v, mode),
                                f"convolve {a} {v} {mode}")
        p32 = np.array([1, -3, 2], np.float32)
        self.assertEqual(numerical.roots(p32).dtype, str(np.roots(p32).dtype))


if __name__ == "__main__":
    unittest.main()
