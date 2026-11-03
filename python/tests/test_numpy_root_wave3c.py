"""trapezoid, gradient, diagflat, indices, ix_, ravel_multi_index, unravel_index, correlate, convolve and
fft.next_fast_len against NumPy's and SciPy's answers: to the bit, but for a trapezoid along an axis that is not the
last (its sum pairwise per column, NumPy's row after row) and the correlations (each lag's sum in its terms' order,
NumPy's in its BLAS's), within a few ulps there; with their errors. Deterministic; the comparisons skipped without
NumPy."""
import math
import unittest
import warnings

import numerical
import numerical.fft

try:
    import numpy as np
except ImportError:
    np = None

try:
    import scipy.fft as scipy_fft
except ImportError:
    scipy_fft = None


class WithoutNumPy(unittest.TestCase):
    def test_values(self):
        self.assertEqual(numerical.trapezoid([1.0, 2.0, 3.0]), 4.0)
        self.assertEqual(numerical.trapezoid([1.0, 2.0], x=[0.0, 2.0]), 3.0)
        self.assertEqual(numerical.gradient([1.0, 2.0, 4.0, 7.0]).tolist(), [1.0, 1.5, 2.5, 3.0])
        self.assertEqual(numerical.gradient([1.0, 4.0, 9.0], edge_order=2).tolist(), [2.0, 4.0, 6.0])
        self.assertEqual(len(numerical.gradient([[1.0, 2.0], [3.0, 5.0]])), 2)
        self.assertEqual(numerical.diagflat(numerical.asarray([1, 2], "int64"), -1).tolist(),
                         [[0, 0, 0], [1, 0, 0], [0, 2, 0]])
        self.assertEqual(numerical.diag(numerical.asarray([3], "int64")).dtype, "int64")
        self.assertEqual(numerical.indices((2, 2)).tolist(), [[[0, 0], [1, 1]], [[0, 1], [0, 1]]])
        self.assertEqual([t.shape for t in numerical.indices((2, 3), sparse=True)], [(2, 1), (1, 3)])
        self.assertEqual([t.tolist() for t in numerical.ix_([True, False, True], [2])], [[[0], [2]], [[2]]])
        self.assertEqual(numerical.ravel_multi_index(([1, 2], [0, 3]), (3, 4)).tolist(), [4, 11])
        self.assertEqual(numerical.ravel_multi_index((1, 5), (3, 4), mode="clip"), 7)
        self.assertEqual([c.tolist() for c in numerical.unravel_index([4, 11], (3, 4))], [[1, 2], [0, 3]])
        self.assertEqual(numerical.unravel_index(5, (3, 4), order="F"), (2, 1))
        self.assertEqual(numerical.correlate([1.0, 2.0, 3.0], [0.0, 1.0, 0.5]).tolist(), [3.5])
        self.assertEqual(numerical.convolve([1.0, 2.0, 3.0], [0.0, 1.0, 0.5]).tolist(), [0.0, 1.0, 2.5, 4.0, 1.5])
        self.assertEqual(numerical.correlate([1j], [1j]).tolist(), [1 + 0j])
        self.assertEqual([numerical.fft.next_fast_len(n) for n in (0, 7, 13, 97)], [0, 7, 14, 98])
        self.assertEqual(numerical.fft.next_fast_len(13, real=True), 15)

    def test_errors(self):
        cases = [
            (IndexError, lambda: numerical.trapezoid([1.0, 2.0], axis=1)),
            (ValueError, lambda: numerical.gradient([1.0])),
            (ValueError, lambda: numerical.gradient([1.0, 2.0], edge_order=3)),
            (ValueError, lambda: numerical.gradient([1.0, 2.0], [1.0])),
            (ValueError, lambda: numerical.gradient([1.0, 2.0], [[1.0, 2.0]])),
            (TypeError, lambda: numerical.gradient([1.0, 2.0], 1.0, 2.0)),
            (ValueError, lambda: numerical.ix_([[1]])),
            (ValueError, lambda: numerical.ravel_multi_index(([3], [0]), (3, 4))),
            (ValueError, lambda: numerical.ravel_multi_index(([1],), (3, 4))),
            (ValueError, lambda: numerical.ravel_multi_index(([1], [1]), (3, 4), mode="bounce")),
            (ValueError, lambda: numerical.ravel_multi_index(([1], [1]), (3, 4), order="K")),
            (TypeError, lambda: numerical.ravel_multi_index(([1.5], [1]), (3, 4))),
            (ValueError, lambda: numerical.unravel_index(12, (3, 4))),
            (TypeError, lambda: numerical.unravel_index([1.5], (3,))),
            (ValueError, lambda: numerical.correlate([1.0], [1.0], "middle")),
            (ValueError, lambda: numerical.correlate([], [1.0])),
            (ValueError, lambda: numerical.convolve([1.0], [])),
            (ValueError, lambda: numerical.correlate([[1.0]], [1.0])),
            (ValueError, lambda: numerical.fft.next_fast_len(-1)),
            (TypeError, lambda: numerical.fft.next_fast_len(2.0)),
        ]
        for error, call in cases:
            with self.assertRaises(error):
                call()


def _tensor(x):
    if isinstance(x, np.ndarray):
        return numerical.asarray(x, "int64") if x.dtype.kind == "i" else numerical.asarray(x)
    return x


@unittest.skipIf(np is None, "NumPy is not installed")
class AgainstNumPy(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.addCleanup(warnings.resetwarnings)

    def assertSame(self, ours, theirs, label, close=False):
        try:
            want = theirs()
        except (IndexError, TypeError, ValueError) as error:
            with self.assertRaises(ValueError if type(error).__name__ == "AxisError" else type(error), msg=label):
                ours()
            return
        got = ours()
        if isinstance(want, tuple):
            self.assertIsInstance(got, tuple, label)
            self.assertEqual(len(got), len(want), label)
            for g, w in zip(got, want):
                self.assertSame(lambda g=g: g, lambda w=w: w, label, close)
            return
        if isinstance(got, (int, float, complex)) and not isinstance(got, bool):  # NumPy's scalar: a number here
            self.assertEqual(np.shape(want), (), label)
            got = np.asarray(got, np.asarray(want).dtype)
        got, want = np.asarray(got), np.asarray(want)
        if want.dtype.kind in "iu":  # NumPy's indices are intp (int32 on a 32-bit Windows)
            want = want.astype(np.int64)
        self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype), label)
        if close:
            np.testing.assert_allclose(got, want, rtol=1e-5 if want.dtype in (np.float32, np.complex64) else 1e-13,
                                       atol=1e-5 if want.dtype in (np.float32, np.complex64) else 1e-12,
                                       err_msg=label)
            return
        nan = np.isnan(want) if want.dtype.kind in "fc" else np.zeros(want.shape, bool)
        self.assertTrue(np.array_equal(np.isnan(got) if got.dtype.kind in "fc" else nan, nan), label)
        self.assertEqual(np.where(nan, 0, got).tobytes(), np.where(nan, 0, want).tobytes(), label)

    def drawn(self, rng, kind, shape):
        if kind == "int64":
            return rng.integers(-5, 5, size=shape, dtype=np.int64)
        x = rng.normal(size=shape)
        return (x + 1j * rng.normal(size=shape)).astype(kind) if kind.startswith("complex") else x.astype(kind)

    def test_trapezoid_and_gradient(self):
        rng = np.random.default_rng(11)
        for kind in ("float64", "float32", "int64", "complex128"):
            for shape in ((7,), (3, 4), (2, 3, 5), (2,)):
                y = self.drawn(rng, kind, shape)
                t = _tensor(y)
                for axis in range(-1, len(shape)):
                    close = axis not in (-1, len(shape) - 1)  # see the module's docstring
                    for x in (None, np.sort(rng.normal(size=shape[axis])), np.sort(rng.normal(size=shape), axis=axis)):
                        self.assertSame(lambda: numerical.trapezoid(t, _tensor(x), axis=axis),
                                        lambda: np.trapezoid(y, x, axis=axis), f"trapezoid {kind} {shape} {axis}",
                                        close)
                    self.assertSame(lambda: numerical.trapezoid(t, dx=0.5, axis=axis),
                                    lambda: np.trapezoid(y, dx=0.5, axis=axis), f"trapezoid dx {kind} {shape}", close)
                for edge_order in (1, 2, 3):
                    for axis in (None, 0, -1, (0, -1)):
                        axes = tuple(range(len(shape))) if axis is None else (axis,) if isinstance(axis, int) else axis
                        points = tuple(np.cumsum(rng.random(shape[a]) + 0.1) for a in axes)
                        for spacing in ((), (0.5,), points, tuple(np.arange(shape[a]) * 2 for a in axes)):
                            self.assertSame(lambda: numerical.gradient(t, *map(_tensor, spacing), axis=axis,
                                                                       edge_order=edge_order),
                                            lambda: np.gradient(y, *spacing, axis=axis, edge_order=edge_order),
                                            f"gradient {kind} {shape} {axis} {edge_order} {len(spacing)}")

    def test_indices_and_correlations(self):
        rng = np.random.default_rng(12)
        for v, k in (([1, 2, 3], 0), ([1.5, 2.0], 2), ([[1, 2], [3, 4]], -1), ([], 1), ([1j, 2], -3)):
            self.assertSame(lambda: numerical.diagflat(v, k), lambda: np.diagflat(v, k), f"diagflat {v} {k}")
        for dimensions in ((2, 3), (4,), (), (0, 2), (2, 1, 3)):
            for dtype in (int, float):
                for sparse in (False, True):
                    self.assertSame(lambda: numerical.indices(dimensions, dtype, sparse),
                                    lambda: np.indices(dimensions, dtype, sparse), f"indices {dimensions}")
        for args in (([0, 1], [2, 4]), ([True, False, True], [3]), ([], [1, 2]), ([[1]],)):
            self.assertSame(lambda: numerical.ix_(*args), lambda: np.ix_(*args), f"ix_ {args}")
        for coordinates, dims, mode, order in ((([3, 6, 6], [4, 5, 1]), (7, 6), "raise", "C"),
                                               (([3, 6, 6], [4, 5, 1]), (7, 6), "raise", "F"),
                                               (([3, 7], [4, 5]), (7, 6), "raise", "C"),
                                               (([3, 7], [-1, 5]), (7, 6), ("clip", "wrap"), "C"),
                                               (([3, 7], [-1, 5]), (7, 6), "wrap", "F"), ((2, 3), (4, 5), "raise", "C"),
                                               (([1], [2], [3]), (2, 3, 4), "raise", "C")):
            self.assertSame(lambda: numerical.ravel_multi_index(coordinates, dims, mode, order),
                            lambda: np.ravel_multi_index(coordinates, dims, mode, order), f"ravel {coordinates}")
        for flat, shape, order in (([22, 41, 37], (7, 6), "C"), ([22, 41, 37], (7, 6), "F"), (42, (7, 6), "C"),
                                   (5, 6, "C"), ([[1, 2], [3, 4]], (2, 3), "C")):
            self.assertSame(lambda: numerical.unravel_index(flat, shape, order),
                            lambda: np.unravel_index(flat, shape, order), f"unravel {flat} {shape} {order}")
        for la in (1, 2, 3, 5, 8):
            for lv in (1, 2, 3, 5, 8):
                for kind in ("float64", "complex128", "int64", "float32"):
                    a, v = self.drawn(rng, kind, (la,)), self.drawn(rng, kind, (lv,))
                    for mode in ("valid", "same", "full"):
                        self.assertSame(lambda: numerical.correlate(_tensor(a), _tensor(v), mode),
                                        lambda: np.correlate(a, v, mode), f"correlate {la} {lv} {kind} {mode}",
                                        kind != "int64")
                        self.assertSame(lambda: numerical.convolve(_tensor(a), _tensor(v), mode),
                                        lambda: np.convolve(a, v, mode), f"convolve {la} {lv} {kind} {mode}",
                                        kind != "int64")

    @unittest.skipIf(scipy_fft is None, "SciPy is not installed")
    def test_next_fast_len(self):
        rng = np.random.default_rng(13)
        bound = (2 ** 64 - 1) // 11 + 1  # SciPy's largest target
        for target in list(range(0, 3000)) + [int(rng.integers(0, bound)) for _ in range(200)] + [bound]:
            for real in (False, True):
                self.assertEqual(numerical.fft.next_fast_len(target, real), scipy_fft.next_fast_len(target, real),
                                 f"{target} {real}")
        for target, error in ((bound + 1, ValueError), (2 ** 63, OverflowError)):
            with self.assertRaises(error):
                scipy_fft.next_fast_len(target)
            with self.assertRaises(error):
                numerical.fft.next_fast_len(target)


if __name__ == "__main__":
    unittest.main()
