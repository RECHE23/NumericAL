"""Basic indexing natively; the array API's namespace (__array_namespace__, isdtype, result_type, can_cast, empty,
from_dlpack, the inspection namespace); median and the quantiles by selection; and more of NumPy's root (array, exp2,
ldexp, frexp, modf, spacing, fix, sinc, lexsort, partition, argpartition, apply_along_axis, fromfunction, fromiter,
rollaxis, real_if_close, sort_complex, array_equiv, resize) against NumPy's answers: to the bit, within an ulp where a
library's function computes; with NumPy's errors. Deterministic; the comparisons skipped without NumPy."""
import math
import unittest
import warnings

import numerical

try:
    import numpy as np
except ImportError:
    np = None


class WithoutNumPy(unittest.TestCase):
    def test_array_api(self):
        t = numerical.ones((2,))
        self.assertIs(t.__array_namespace__(), numerical)
        self.assertIs(t.__array_namespace__(api_version="2023.12"), numerical)
        with self.assertRaises(ValueError):
            t.__array_namespace__(api_version="2021.12")
        self.assertTrue(numerical.isdtype("float32", "real floating"))
        self.assertTrue(numerical.isdtype("int64", ("bool", "integral")))
        self.assertFalse(numerical.isdtype("complex64", "real floating"))
        self.assertTrue(numerical.isdtype(numerical.float64, numerical.float64))
        self.assertFalse(numerical.isdtype("bool", "numeric"))
        with self.assertRaises(TypeError):
            numerical.isdtype("float16", "numeric")
        self.assertEqual(numerical.result_type("float32", "int64"), "float64")
        self.assertEqual(numerical.result_type(numerical.ones(2, "float32"), 2.0), "float32")
        self.assertEqual(numerical.result_type(numerical.ones(2, "int64"), 2.5), "float64")
        self.assertEqual(numerical.result_type(1, 2.0), "float64")
        self.assertEqual(numerical.result_type(True), "bool")
        self.assertEqual(numerical.result_type(3), "int64")
        self.assertEqual(numerical.result_type("float32", 1j), "complex64")
        with self.assertRaises(ValueError):
            numerical.result_type()
        self.assertTrue(numerical.can_cast("int64", "float64"))
        self.assertFalse(numerical.can_cast("float64", "float32"))
        self.assertTrue(numerical.can_cast("float64", "float32", "same_kind"))
        self.assertFalse(numerical.can_cast("float64", "int64", "same_kind"))
        self.assertTrue(numerical.can_cast("complex128", "bool", "unsafe"))
        self.assertTrue(numerical.can_cast(numerical.ones(2, "bool"), "complex64"))
        self.assertFalse(numerical.can_cast("float32", "float64", "no"))
        with self.assertRaises(ValueError):
            numerical.can_cast("float32", "float64", "nearly")
        with self.assertRaises(TypeError):
            numerical.can_cast(1.0, "float64")
        self.assertEqual(numerical.empty((2, 3), "int64").tolist(), [[0, 0, 0], [0, 0, 0]])
        with self.assertRaises(ValueError):
            numerical.empty(2, device="gpu")
        self.assertIs(numerical.from_dlpack(t), t)
        self.assertEqual(numerical.from_dlpack(t, copy=True).tolist(), [1.0, 1.0])
        with self.assertRaises(ValueError):
            numerical.from_dlpack(t, copy=False)
        with self.assertRaises(TypeError):
            numerical.from_dlpack([1.0])
        info = numerical.__array_namespace_info__()
        self.assertEqual((info.default_device(), info.devices()), ("cpu", ["cpu"]))
        self.assertTrue(info.capabilities()["boolean indexing"])
        self.assertEqual(info.default_dtypes()["indexing"], "int64")
        self.assertEqual(list(info.dtypes(kind="real floating")), ["float32", "float64"])
        self.assertEqual(len(info.dtypes()), 6)

    def test_root_functions(self):
        self.assertEqual(numerical.array([1, 2], ndmin=3).shape, (1, 1, 2))
        self.assertEqual(numerical.array(5.0).shape, ())
        with self.assertRaises(ValueError):
            numerical.array([1.0], copy=False)
        self.assertEqual(numerical.ascontiguousarray(3.0).shape, (1,))
        self.assertEqual(numerical.asanyarray([1, 2], "int64").dtype, "int64")
        self.assertEqual(numerical.exp2([0.0, 3.0, -1.0]).tolist(), [1.0, 8.0, 0.5])
        self.assertEqual(numerical.ldexp([1.0, 3.0], numerical.asarray([3, -1], "int64")).tolist(), [8.0, 1.5])
        self.assertEqual(numerical.ldexp(1.0, numerical.asarray([2000], "int64")).tolist(), [math.inf])
        with self.assertRaises(TypeError):
            numerical.ldexp(1.0, 2.0)
        m, e = numerical.frexp([8.0, -0.75, 0.0, math.inf])
        self.assertEqual((m.tolist(), e.tolist()), ([0.5, -0.75, 0.0, math.inf], [4, 0, 0, 0]))
        fraction, whole = numerical.modf([2.5, -2.0, math.inf])
        self.assertEqual((fraction.tolist(), whole.tolist()), ([0.5, -0.0, 0.0], [2.0, -2.0, math.inf]))
        self.assertTrue(math.copysign(1, fraction.tolist()[1]) < 0)
        self.assertEqual(numerical.spacing([1.0]).tolist(), [2.0 ** -52])
        self.assertEqual(numerical.fix([-1.5, 1.5]).tolist(), [-1.0, 1.0])
        self.assertEqual(numerical.sinc([0.0]).tolist(), [1.0])
        self.assertEqual(numerical.lexsort([[1, 0, 1], [0, 0, 1]]).tolist(), [1, 0, 2])
        with self.assertRaises(TypeError):
            numerical.lexsort([])
        self.assertEqual(numerical.partition([3.0, 1.0, 2.0], 1).tolist()[1], 2.0)
        self.assertEqual(numerical.argpartition([3.0, 1.0, 2.0], 0).tolist()[0], 1)
        with self.assertRaises(ValueError):
            numerical.partition([1.0, 2.0], 2)
        with self.assertRaises(ValueError):
            numerical.partition(1.0, 0)
        self.assertEqual(numerical.partition([[2.0, 1.0]], 0, axis=None).tolist(), [1.0, 2.0])
        self.assertEqual(numerical.argpartition([[2.0, 1.0]], [0, 1], axis=None).tolist(), [1, 0])
        self.assertEqual(numerical.apply_along_axis(numerical.sum, 0, [[1.0, 2.0], [3.0, 4.0]]).tolist(), [4.0, 6.0])
        with self.assertRaises(ValueError):
            numerical.apply_along_axis(numerical.sum, 0, numerical.zeros((2, 0)))
        self.assertEqual(numerical.fromfunction(lambda i, j: i + j, (2, 2)).tolist(), [[0.0, 1.0], [1.0, 2.0]])
        self.assertEqual(numerical.fromiter(iter([1, 2, 3]), "int64", 2).tolist(), [1, 2])
        self.assertEqual(numerical.fromiter([], "float64").shape, (0,))
        with self.assertRaises(ValueError):
            numerical.fromiter([1], "float64", 3)
        self.assertEqual(numerical.rollaxis(numerical.zeros((2, 3, 4)), 2).shape, (4, 2, 3))
        self.assertEqual(numerical.rollaxis(numerical.zeros((2, 3, 4)), 0, 3).shape, (3, 4, 2))
        self.assertEqual(numerical.rollaxis(numerical.zeros((2, 3)), 1, -1).shape, (2, 3))
        with self.assertRaises(ValueError):
            numerical.rollaxis(numerical.zeros((2, 3)), 0, 4)
        self.assertEqual(numerical.real_if_close([1 + 1e-20j]).dtype, "float64")
        self.assertEqual(numerical.real_if_close([1 + 1e-3j], tol=1e-2).dtype, "float64")
        self.assertEqual(numerical.real_if_close([1 + 1j]).dtype, "complex128")
        self.assertEqual(numerical.real_if_close([1.0]).dtype, "float64")
        self.assertEqual(numerical.sort_complex([2.0, 1.0]).tolist(), [1 + 0j, 2 + 0j])
        self.assertEqual(numerical.sort_complex(numerical.asarray([2.0], "float32")).dtype, "complex64")
        self.assertEqual(numerical.sort_complex([1j, 0j]).tolist(), [0j, 1j])
        self.assertTrue(numerical.array_equiv([1.0, 2.0], [[1.0, 2.0], [1.0, 2.0]]))
        self.assertFalse(numerical.array_equiv([1.0, 2.0], [1.0, 2.0, 3.0]))
        self.assertEqual(numerical.resize([1.0, 2.0, 3.0], (2, 2)).tolist(), [[1.0, 2.0], [3.0, 1.0]])
        self.assertEqual(numerical.resize([], 3).tolist(), [0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            numerical.resize([1.0], (-1,))


@unittest.skipIf(np is None, "NumPy is not installed")
class BasicIndexing(unittest.TestCase):
    """Tensor.__getitem__ with integers and slices natively, and every other key through the Python indexing: the
    same answers as NumPy's either way, and NumPy's errors."""

    def test_against_numpy(self):
        a = np.arange(60.0).reshape(3, 4, 5)
        t = numerical.asarray(a)
        keys = [0, -1, 2, slice(None), slice(1, None), slice(None, None, -1), slice(-2, 1, -1), slice(5, 9),
                (1, 2), (1, 2, 3), (-1, slice(None, None, 2)), (slice(None), 3), (slice(1, 3), slice(None), -1),
                (), (Ellipsis, 1), (None, 0), np.int64(1), (np.int64(-1), 2), [0, 2], (slice(None), [1, 1]),
                True, (0, 0, slice(4, None, -2)), Ellipsis, (1, Ellipsis, 2), (Ellipsis, 0, 1, 2),
                (0, Ellipsis), (Ellipsis, slice(1, 3))]
        for key in keys:
            with self.subTest(key=key):
                got, want = t[key], a[key]
                if isinstance(got, numerical.Tensor):
                    self.assertTrue(np.array_equal(np.asarray(got), want))
                    self.assertEqual(got.shape, np.shape(want))
                else:
                    self.assertEqual(got, want)
        for key, error in ((3, IndexError), (-4, IndexError), ((0, 4), IndexError), ((0, 0, 0, 0), IndexError),
                           (slice(None, None, 0), ValueError), (1.5, IndexError), ((0, -6), IndexError),
                           ((Ellipsis, Ellipsis), IndexError), ((Ellipsis, 0, 0, 0, 0), IndexError)):
            with self.subTest(key=key), self.assertRaises(error):
                t[key]
        with self.assertRaisesRegex(IndexError, "index 3 is out of bounds for axis 0 with size 3"):
            t[3]


@unittest.skipIf(np is None, "NumPy is not installed")
class AgainstNumPy(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.addCleanup(warnings.resetwarnings)

    def assertSame(self, ours, theirs, label, ulps=0):
        try:
            want = theirs()
        except (IndexError, TypeError, ValueError) as error:
            with self.assertRaises(type(error), msg=label):
                ours()
            return
        got = ours()
        if isinstance(want, tuple):
            for g, w in zip(got, want):
                self.assertSame(lambda g=g: g, lambda w=w: w, label, ulps)
            return
        want = np.asarray(want)
        if want.dtype.kind in "iu":  # NumPy's indices are intp, its frexp exponents int32
            want = want.astype(np.int64)
        if isinstance(got, (int, float, complex)) and not isinstance(got, bool):
            self.assertEqual(want.shape, (), label)
            self.assertTrue(got == want.item() or (got != got and want != want), f"{label}: {got!r} {want!r}")
            return
        got = np.asarray(got)
        self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype), label)
        if ulps and want.dtype.kind == "f":
            finite = np.isfinite(want) & (want != 0)
            self.assertEqual(np.where(finite, 0, got).tobytes(), np.where(finite, 0, want).tobytes(), label)
            np.testing.assert_array_max_ulp(got[finite], want[finite], maxulp=ulps)
            return
        self.assertEqual(got.tobytes(), want.tobytes(), label)

    def test_dtype_functions(self):
        names = ("bool", "int64", "float32", "float64", "complex64", "complex128")
        for a in names:
            for b in names:
                self.assertEqual(numerical.result_type(a, b), str(np.result_type(a, b)), f"{a} {b}")
                for casting in ("no", "equiv", "safe", "same_kind", "unsafe"):
                    self.assertEqual(numerical.can_cast(a, b, casting), np.can_cast(a, b, casting),
                                     f"{a} {b} {casting}")
            for scalar in (True, 2, 2.5, 1j):
                if a == "bool" and scalar == 2 and not isinstance(scalar, bool):
                    continue  # a bool beside an int is float64 here (as asarray reads an int) until 2026.12.0
                self.assertEqual(numerical.result_type(numerical.ones(1, a), scalar),
                                 str(np.result_type(np.ones(1, a), scalar)), f"{a} {scalar!r}")
            self.assertEqual(numerical.result_type(np.dtype(a)), a)
            self.assertEqual(numerical.result_type(np.ones(2, a)), a)
        self.assertEqual(numerical.from_dlpack(np.arange(3.0)).tolist(), [0.0, 1.0, 2.0])

    def test_order_statistics(self):
        rng = np.random.default_rng(1105)
        specials = np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 1.0, -1.0, 2.0])
        for case in range(300):
            shape = tuple(int(e) for e in rng.integers(1, 6, int(rng.integers(1, 4))))
            a = rng.standard_normal(shape)
            pick = rng.random(shape) < 0.3
            a[pick] = specials[rng.integers(0, len(specials), int(pick.sum()))]
            if rng.random() < 0.3:
                a = np.round(a)  # repeated values
            t = numerical.asarray(a)
            axis = int(rng.integers(-len(shape), len(shape))) if rng.random() < 0.7 else None
            for name in ("median", "nanmedian"):
                self.assertSame(lambda: getattr(numerical, name)(t, axis=axis),
                                lambda: getattr(np, name)(a, axis=axis), f"{case} {name} {shape} {axis}")
            q = float(rng.random()) if rng.random() < 0.5 else rng.random(int(rng.integers(1, 12)))
            for method in ("linear", "lower", "higher", "nearest", "midpoint", "inverted_cdf", "hazen"):
                for name in ("quantile", "nanquantile"):
                    # a zero's sign where a method answers an element: NumPy's partition picks one, its sort here
                    got = lambda: numerical.where(numerical.equal(r := getattr(numerical, name)(t, q, axis=axis,
                                                                                                method=method), 0),
                                                  0.0, r)
                    want = lambda: np.where((w := getattr(np, name)(a, q, axis=axis, method=method)) == 0, 0.0, w)
                    self.assertSame(got, want, f"{case} {name} {method} {shape} {axis} {q}")
        ints = np.array([[5, -3, 7, 7, 0], [1, 2, 3, 4, -9]], np.int64)
        for axis in (None, 0, 1):
            self.assertSame(lambda: numerical.median(numerical.asarray(ints, "int64"), axis=axis),
                            lambda: np.median(ints, axis=axis), f"int64 median {axis}")

    def test_root_functions(self):
        rng = np.random.default_rng(6)
        x = np.concatenate([rng.standard_normal(50) * 10 ** rng.uniform(-300, 300, 50),
                            [0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324, -5e-324, 1.0, 0.5, 2.0 ** 1023]])
        for dtype in ("float64", "float32"):
            v = x.astype(dtype)
            t = numerical.asarray(v)
            self.assertSame(lambda: numerical.exp2(t), lambda: np.exp2(v), f"exp2 {dtype}", ulps=1)
            exponents = rng.integers(-1100, 1100, v.size).astype(np.int64)
            self.assertSame(lambda: numerical.ldexp(t, numerical.asarray(exponents, "int64")),
                            lambda: np.ldexp(v, exponents), f"ldexp {dtype}")
            # an infinity's or a NaN's exponent is C's to choose: 0 from glibc and macOS, as here, −1 from MSVC
            self.assertSame(lambda: numerical.frexp(t), lambda: (np.frexp(v)[0], np.where(np.isfinite(v),
                                                                                          np.frexp(v)[1], 0)),
                            f"frexp {dtype}")
            self.assertSame(lambda: numerical.modf(t), lambda: np.modf(v), f"modf {dtype}")
            self.assertSame(lambda: numerical.spacing(t), lambda: np.spacing(v), f"spacing {dtype}")
            self.assertSame(lambda: numerical.fix(t), lambda: np.fix(v), f"fix {dtype}")
        s = rng.standard_normal(40)
        self.assertSame(lambda: numerical.sinc(numerical.asarray(s)), lambda: np.sinc(s), "sinc", ulps=2)
        keys = [rng.integers(0, 3, 30), rng.integers(0, 3, 30), rng.standard_normal(30).round()]
        self.assertSame(lambda: numerical.lexsort([numerical.asarray(k) for k in keys]), lambda: np.lexsort(keys),
                        "lexsort")
        m = rng.standard_normal((3, 4))
        self.assertSame(lambda: numerical.lexsort(numerical.asarray(m)), lambda: np.lexsort(m), "lexsort rows")
        # keys of every real dtype together (ties, both zeros and NaNs among them), along either axis of matrices, and
        # long enough for a radix sort among the threads
        for shape, axis in (((60, 50), 0), ((60, 50), 1), ((60, 50), -1), ((200001,), 0)):
            mixed = [rng.integers(0, 3, shape), rng.random(shape) < 0.5,
                     np.where(rng.random(shape) < 0.2, np.nan, rng.integers(-2, 3, shape) * 0.5),
                     np.where(rng.random(shape) < 0.3, -0.0, rng.integers(0, 2, shape)).astype(np.float32)]
            ours = [numerical.asarray(k, "int64") if k.dtype.kind == "i" else numerical.asarray(k) for k in mixed]
            self.assertSame(lambda: numerical.lexsort(ours, axis=axis), lambda: np.lexsort(mixed, axis=axis),
                            f"lexsort {shape} {axis}")
        # numbers as keys (a 1-D array's rows), none (an empty one's) and an axis that is not an integer, as NumPy's
        for keys in (np.array([3.0, 1.0, 2.0]), np.array([[1.0, 2.0], [2.0, 1.0]]), np.zeros((0,)), np.array(5.0),
                     m[:, :0], np.zeros((0, 3))):
            for axis in (-1, 0, 1, 2, None):
                for given in (numerical.asarray(keys), keys):
                    with self.subTest(keys=keys.shape, axis=axis, given=type(given).__name__):
                        try:
                            want = np.lexsort(keys, axis=axis)
                        except (TypeError, ValueError) as error:  # an AxisError a ValueError
                            with self.assertRaises(type(error)):
                                numerical.lexsort(given, axis=axis)
                            continue
                        got = np.asarray(numerical.lexsort(given, axis=axis))
                        self.assertEqual((got.tolist(), got.shape), (want.tolist(), want.shape))
        for axis in (-1, 0, None):  # a number's argsort: one element's, as NumPy's
            self.assertEqual(numerical.argsort(numerical.asarray(5.0), axis=axis).tolist(), [0])
        self.assertEqual(numerical.argsort(numerical.asarray(5.0), descending=True).tolist(), [0])
        with self.assertRaises(numerical.AxisError):
            numerical.argsort(numerical.asarray(5.0), axis=1)
        for kth in (0, 2, -1, [0, 3]):
            got = numerical.partition(numerical.asarray(m), kth, axis=1)
            want = np.partition(m, kth, axis=1)
            for k in ([kth] if isinstance(kth, int) else kth):
                np.testing.assert_array_equal(np.asarray(got)[:, k], want[:, k])
            order = np.asarray(numerical.argpartition(numerical.asarray(m), kth, axis=1))
            for k in ([kth] if isinstance(kth, int) else kth):
                np.testing.assert_array_equal(np.take_along_axis(m, order, 1)[:, k], want[:, k])
        self.assertSame(lambda: numerical.partition(m, 4), lambda: np.partition(m, 4), "partition kth past")
        for axis in (0, 1, -1):
            self.assertSame(lambda: numerical.apply_along_axis(lambda r: numerical.stack([numerical.max(r), 0.0]),
                                                               axis, m),
                            lambda: np.apply_along_axis(lambda r: np.array([r.max(), 0.0]), axis, m),
                            f"apply_along_axis {axis}")
            self.assertSame(lambda: numerical.apply_along_axis(numerical.sum, axis, m),
                            lambda: np.apply_along_axis(np.sum, axis, m), f"apply_along_axis sum {axis}")
        cube = rng.standard_normal((2, 3, 4))
        self.assertSame(lambda: numerical.apply_along_axis(lambda r: numerical.outer(r, r), 1, cube),
                        lambda: np.apply_along_axis(lambda r: np.outer(r, r), 1, cube), "apply_along_axis outer")
        for axis in range(-3, 3):
            for start in range(-3, 4):
                self.assertSame(lambda: numerical.rollaxis(numerical.asarray(cube), axis, start).shape,
                                lambda: np.rollaxis(cube, axis, start).shape, f"rollaxis {axis} {start}")
        for z in ([1 + 1e-14j, 2.0], [1 + 1e-10j], [1.0 + 0j]):
            self.assertSame(lambda: numerical.real_if_close(z), lambda: np.real_if_close(z), f"real_if_close {z}")
        c = rng.standard_normal(10).round() + 1j * rng.standard_normal(10).round()
        self.assertSame(lambda: numerical.sort_complex(numerical.asarray(c)), lambda: np.sort_complex(c),
                        "sort_complex")
        for shape in ((7,), (2, 5), (0,), 4):
            self.assertSame(lambda: numerical.resize(numerical.asarray(s[:3]), shape), lambda: np.resize(s[:3], shape),
                            f"resize {shape}")
        for b in ([1.0, 2.0], [[1.0, 2.0]], [1.0], [1.0, 3.0]):
            self.assertEqual(numerical.array_equiv([1.0, 2.0], b), np.array_equiv([1.0, 2.0], b), f"equiv {b}")


@unittest.skipIf(np is None, "NumPy is not installed")
class TensorMethods(unittest.TestCase):
    def test_the_methods_answer_as_ndarrays(self):
        # argpartition, choose, compress, repeat and take as NumPy's methods of those names, their arguments in
        # NumPy's order (compress's condition first, take's `out` before `mode`)
        values = np.array([[3.0, -1.0, 2.0], [0.5, 7.0, -4.0]])
        t = numerical.asarray(values)
        index = np.array([[0, 1, 1], [1, 0, 2]], dtype=np.int64)  # a Tensor's integers
        cases = [
            ("argpartition", values, (1,), {}),
            ("argpartition", values, (0,), {"axis": 0}),
            ("argpartition", values, (2,), {"axis": None}),
            ("choose", index, ([values[0], values[1], values[0] * 2],), {}),
            ("choose", index + 2, ([values[0], values[1], values[0] * 2],), {"mode": "wrap"}),
            ("compress", values, ([True, False, True],), {"axis": 1}),
            ("compress", values, ([False, True],), {}),
            ("repeat", values, (2,), {}),
            ("repeat", values, ([1, 3],), {"axis": 0}),
            ("take", values, ([4, 0, -1],), {}),
            ("take", values, ([[2, 0]],), {"axis": 1}),
            ("take", values, ([5, -9],), {"axis": 1, "mode": "clip"}),
            ("take", values, ([7],), {"axis": 0, "mode": "wrap"}),
            ("take", values, (1, 1, None, "wrap"), {}),
        ]
        for name, array, args, kwargs in cases:
            with self.subTest(name=name, args=args, kwargs=kwargs):
                reference = array.astype(np.intp) if name == "choose" else array  # NumPy's indices are a pointer's size
                want = getattr(reference, name)(*args, **kwargs)
                got = getattr(numerical.asarray(array, str(array.dtype)), name)(*args, **kwargs)
                if name == "argpartition":  # one arrangement among those that partition: the stable sort's
                    kth, axis = args[0], kwargs.get("axis", -1)
                    flat = array.ravel() if axis is None else array
                    got = np.asarray(got)
                    np.testing.assert_array_equal(np.take_along_axis(flat, got, axis=-1 if axis is None else axis)
                                                  .take(kth, axis=-1 if axis is None else axis),
                                                  np.partition(flat, kth, axis=-1 if axis is None else axis)
                                                  .take(kth, axis=-1 if axis is None else axis))
                    continue
                self.assertEqual(np.asarray(got).dtype, want.dtype)
                np.testing.assert_array_equal(np.asarray(got), want)
        for name, args in (("take", ([0], None, numerical.zeros((1,)))), ("compress", ([True], None, t)),
                           ("choose", ([values[0]], t))):
            with self.subTest(name=name, out=True):
                with self.assertRaises(TypeError):
                    getattr(t, name)(*args)
        with self.assertRaises(IndexError):
            t.take([6])
        with self.assertRaises(ValueError):
            t.argpartition(3)


if __name__ == "__main__":
    unittest.main()
