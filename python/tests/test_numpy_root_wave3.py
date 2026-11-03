"""add, subtract, multiply, the array API's aliases (acos… atanh, pow, permute_dims), amax and amin, reciprocal, fabs,
signbit, isposinf and isneginf, the complex predicates, isscalar, shape, ndim, size, tri and the triangle and diagonal
indices, meshgrid: against NumPy's answers to the bit (a complex reciprocal within 2 ulps: NumPy's compiler may fuse
its products), in every dtype they take, with NumPy's errors. Deterministic; the NumPy comparisons skipped without
NumPy."""
import math
import unittest
import warnings

import numerical

try:
    import numpy as np
except ImportError:
    np = None

_NUMPY_2_4 = np is not None and tuple(int(part) for part in np.__version__.split(".")[:2]) >= (2, 4)


class WithoutNumPy(unittest.TestCase):
    def test_values(self):
        self.assertEqual(numerical.add([1, 2], 3).tolist(), [4.0, 5.0])
        self.assertEqual(numerical.subtract(1.5, [1, 2]).tolist(), [0.5, -0.5])
        self.assertEqual(numerical.multiply(2, 3).tolist(), 6.0)
        self.assertIs(numerical.acos, numerical.arccos)
        self.assertIs(numerical.pow, numerical.power)
        self.assertEqual(numerical.permute_dims(numerical.zeros((2, 3, 4)), (2, 0, 1)).shape, (4, 2, 3))
        self.assertEqual(numerical.reciprocal(numerical.asarray([4.0, -0.0])).tolist(), [0.25, -math.inf])
        self.assertEqual(numerical.reciprocal(numerical.asarray([1, -1, 5], "int64")).tolist(), [1, -1, 0])
        self.assertEqual(numerical.reciprocal(numerical.asarray([1j])).tolist(), [-1j])
        self.assertEqual(numerical.fabs(numerical.asarray([-2], "int64")).tolist(), [2.0])
        self.assertEqual(numerical.signbit([-0.0, 0.0, -math.nan, -1.0]).tolist(), [True, False, True, True])
        self.assertEqual(numerical.isposinf([math.inf, -math.inf]).tolist(), [True, False])
        self.assertEqual(numerical.isneginf([math.inf, -math.inf]).tolist(), [False, True])
        self.assertEqual(numerical.iscomplex([1 + 0j, 1j]).tolist(), [False, True])
        self.assertEqual(numerical.isreal([1.0]).tolist(), [True])
        self.assertTrue(numerical.iscomplexobj([1j]) and numerical.isrealobj([1.0]))
        self.assertTrue(numerical.isscalar(1.5) and not numerical.isscalar(numerical.asarray(1.5)))
        self.assertEqual((numerical.shape([[1, 2]]), numerical.ndim([[1, 2]]), numerical.size([[1, 2]], 1)),
                         ((1, 2), 2, 2))
        self.assertEqual(numerical.tri(2, 3, 1, "int64").tolist(), [[1, 1, 0], [1, 1, 1]])
        self.assertEqual([t.tolist() for t in numerical.tril_indices(2)], [[0, 1, 1], [0, 0, 1]])
        self.assertEqual([t.tolist() for t in numerical.triu_indices_from(numerical.zeros((2, 3)), 1)],
                         [[0, 0, 1], [1, 2, 2]])
        self.assertEqual([t.tolist() for t in numerical.diag_indices_from(numerical.zeros((2, 2)))], [[0, 1], [0, 1]])
        x, y = numerical.meshgrid([1.0, 2.0], [3.0, 4.0, 5.0], indexing="ij", sparse=True)
        self.assertEqual((x.shape, y.shape), ((2, 1), (1, 3)))
        self.assertEqual(numerical.meshgrid(), ())

    def test_errors(self):
        cases = [
            (TypeError, lambda: numerical.reciprocal(numerical.asarray([True]))),
            (ValueError, lambda: numerical.reciprocal(numerical.asarray([0], "int64"))),
            (TypeError, lambda: numerical.fabs([1j])),
            (TypeError, lambda: numerical.fabs(numerical.asarray([True]))),
            (TypeError, lambda: numerical.signbit([1j])),
            (TypeError, lambda: numerical.isposinf([1j])),
            (TypeError, lambda: numerical.isneginf([1j])),
            (TypeError, lambda: numerical.isposinf([1.0], out=numerical.zeros(1))),
            (ValueError, lambda: numerical.tril_indices_from(numerical.zeros(2))),
            (ValueError, lambda: numerical.triu_indices_from(numerical.zeros((1, 1, 1)))),
            (ValueError, lambda: numerical.diag_indices_from(numerical.zeros((2, 3)))),
            (ValueError, lambda: numerical.diag_indices_from(numerical.zeros(2))),
            (ValueError, lambda: numerical.meshgrid([1.0], indexing="xyz")),
            (ValueError, lambda: numerical.size([1.0], 1)),
        ]
        for error, call in cases:
            with self.assertRaises(error):
                call()


def _drawn(rng, kind, shape=(4, 5)):
    if kind == "int64":
        return rng.integers(-3, 4, size=shape, dtype=np.int64)
    if kind == "bool":
        return rng.random(shape) < 0.5
    if kind.startswith("complex"):
        x, y = _drawn(rng, "float64", shape), _drawn(rng, "float64", shape)
        return (x + 1j * y).astype(kind)
    x = rng.normal(size=shape) * np.exp(rng.normal(size=shape) * 5)
    for value, share in ((0.0, 0.1), (-0.0, 0.1), (np.inf, 0.05), (-np.inf, 0.05), (np.nan, 0.05)):
        x[rng.random(shape) < share] = value
    return x.astype(kind)


@unittest.skipIf(np is None, "NumPy is not installed")
class AgainstNumPy(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.addCleanup(warnings.resetwarnings)

    def assertSame(self, ours, theirs, label, ulps=0):
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
                self.assertSame(lambda g=g: g, lambda w=w: w, label)
            return
        if not isinstance(want, (np.ndarray, np.generic)):
            self.assertEqual((type(got), got), (type(want), want), label)
            return
        if isinstance(got, (int, float, complex)) and not isinstance(got, bool):  # NumPy's scalar: a number here
            self.assertEqual(np.shape(want), (), label)
            got = np.asarray(got, np.asarray(want).dtype)
        got, want = np.asarray(got), np.asarray(want)
        if want.dtype.kind in "iu":  # NumPy's indices are intp (int32 on a 32-bit Windows)
            want = want.astype(np.int64)
        self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype), label)
        if want.dtype.kind not in "fc":
            self.assertEqual(got.tobytes(), want.tobytes(), label)
            return
        self.assertTrue(np.array_equal(np.isnan(got), np.isnan(want)), label)
        got, want = np.where(np.isnan(got), 0, got), np.where(np.isnan(want), 0, want)
        if ulps == 0:
            self.assertEqual(got.tobytes(), want.tobytes(), label)
            return
        for g, w in ((got.real, want.real), (got.imag, want.imag)):
            exact = ~np.isfinite(w) | (w == 0)
            self.assertEqual(g[exact].tobytes(), w[exact].tobytes(), label)
            self.assertTrue(np.all(np.abs(g - w)[~exact] <= ulps * np.spacing(np.abs(w))[~exact]), label)

    def test_elementwise(self):
        rng = np.random.default_rng(5)
        kinds = ("float64", "float32", "complex128", "complex64", "int64", "bool")
        for kind in kinds:
            x = _drawn(rng, kind)
            t = numerical.asarray(x, "int64") if kind == "int64" else numerical.asarray(x)
            names = ["reciprocal", "fabs", "signbit", "isposinf", "isneginf", "iscomplex", "isreal", "iscomplexobj",
                     "isrealobj", "isscalar", "shape", "ndim", "size"]
            if not kind.startswith("complex") and kind != "bool":  # see differences.md: real and no float16
                names += ["acos", "asin", "atan", "acosh", "asinh", "atanh"]
            for name in names:
                if name == "reciprocal" and kind in ("bool", "int64"):
                    continue  # NumPy's int8 for a bool, and its int64 of an infinity for a zero (see the doc)
                if name == "fabs" and kind == "bool":
                    continue  # NumPy's float16
                self.assertSame(lambda: getattr(numerical, name)(t), lambda: getattr(np, name)(x), f"{name} {kind}",
                                ulps=2 if name == "reciprocal" and kind.startswith("complex") else 0)
            if _NUMPY_2_4:  # a tuple of axes from NumPy 2.4 on
                self.assertSame(lambda: numerical.size(t, (0, 1)), lambda: np.size(x, (0, 1)), "size of axes")
            self.assertSame(lambda: numerical.permute_dims(t, (1, 0)), lambda: np.permute_dims(x, (1, 0)), "permute_dims")
            if not kind.startswith("complex"):
                self.assertSame(lambda: numerical.amax(t, axis=0), lambda: np.amax(x, axis=0), f"amax {kind}")
                self.assertSame(lambda: numerical.amin(t), lambda: np.amin(x), f"amin {kind}")
            for other in kinds:
                if kind == "bool" and other == "bool":
                    continue  # see differences.md: no bool arithmetic
                y = _drawn(rng, other)
                u = numerical.asarray(y, "int64") if other == "int64" else numerical.asarray(y)
                for name in ("add", "subtract", "multiply"):
                    self.assertSame(lambda: getattr(numerical, name)(t, u), lambda: getattr(np, name)(x, y),
                                    f"{name} {kind} {other}")
            for number in (2.5, 1j) if kind != "bool" else ():
                for name in ("add", "subtract", "multiply"):
                    self.assertSame(lambda: getattr(numerical, name)(t, number), lambda: getattr(np, name)(x, number),
                                    f"{name} {kind} {number!r}")
                    self.assertSame(lambda: getattr(numerical, name)(number, t), lambda: getattr(np, name)(number, x),
                                    f"{name} {number!r} {kind}")

    def test_constructions(self):
        for n, m, k in ((3, None, 0), (3, 5, 1), (4, 2, -1), (0, 3, 0), (3, 3, -5), (2, 2, 7)):
            for dtype in (float, "int64", bool, "float32", "complex128"):
                self.assertSame(lambda: numerical.tri(n, m, k, dtype), lambda: np.tri(n, m, k, dtype),
                                f"tri {n} {m} {k} {dtype}")
            self.assertSame(lambda: numerical.tril_indices(n, k, m), lambda: np.tril_indices(n, k, m), "tril_indices")
            self.assertSame(lambda: numerical.triu_indices(n, k, m), lambda: np.triu_indices(n, k, m), "triu_indices")
        for shape in ((3, 4), (4, 4), (2,), (2, 2, 2)):
            a = np.zeros(shape)
            for k in (-1, 0, 2):
                self.assertSame(lambda: numerical.tril_indices_from(numerical.asarray(a), k),
                                lambda: np.tril_indices_from(a, k), f"tril_indices_from {shape} {k}")
                self.assertSame(lambda: numerical.triu_indices_from(numerical.asarray(a), k),
                                lambda: np.triu_indices_from(a, k), f"triu_indices_from {shape} {k}")
            self.assertSame(lambda: numerical.diag_indices_from(numerical.asarray(a)),
                            lambda: np.diag_indices_from(a), f"diag_indices_from {shape}")
        self.assertSame(lambda: numerical.diag_indices(4, 3), lambda: np.diag_indices(4, 3), "diag_indices")
        vectors = ([1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0], [0.5])
        for count in range(5):
            for indexing in ("xy", "ij"):
                for sparse in (False, True):
                    self.assertSame(lambda: numerical.meshgrid(*vectors[:count], indexing=indexing, sparse=sparse),
                                    lambda: tuple(np.meshgrid(*map(np.array, vectors[:count]), indexing=indexing,
                                                              sparse=sparse)), f"meshgrid {count} {indexing} {sparse}")
        for value in (1, 2.5, -0.0, 1j, "text", [1, 2], np.float32(1)):
            self.assertEqual(numerical.isscalar(value), np.isscalar(value), repr(value))
        self.assertEqual(numerical.isscalar(numerical.asarray(1.0)), np.isscalar(np.asarray(1.0)))


if __name__ == "__main__":
    unittest.main()
