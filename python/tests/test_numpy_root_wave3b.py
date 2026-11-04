"""linspace (in full), logspace, geomspace, identity, vander, the sets (unique and its forms, intersect1d, union1d,
setdiff1d, setxor1d, isin), digitize, cumulative_sum and cumulative_prod, unstack, matvec, vecmat, ediff1d,
trim_zeros, select, and the complex prod and cumprod: against NumPy's answers to the bit (logspace and geomspace
within 2 ulps, their powers C's library's; matvec and vecmat within a few: NumPy's go through its BLAS), with
NumPy's errors. Deterministic; the NumPy comparisons skipped without NumPy."""
import math
import unittest
import warnings

import numerical

try:
    import numpy as np
except ImportError:
    np = None

_NUMPY_2_3 = np is not None and tuple(int(part) for part in np.__version__.split(".")[:2]) >= (2, 3)
_NUMPY_2_4 = np is not None and tuple(int(part) for part in np.__version__.split(".")[:2]) >= (2, 4)


class WithoutNumPy(unittest.TestCase):
    def test_values(self):
        self.assertEqual(numerical.linspace(0.0, 1.0, 4, endpoint=False, retstep=True)[1], 0.25)
        self.assertEqual(numerical.linspace([0.0, 10.0], 20.0, 3, axis=-1).tolist(), [[0.0, 10.0, 20.0],
                                                                                      [10.0, 15.0, 20.0]])
        self.assertEqual(numerical.linspace(0, 5, 3, dtype="int64").tolist(), [0, 2, 5])
        self.assertTrue(math.isnan(numerical.linspace(0.0, 1.0, 1, retstep=True)[1]))
        self.assertEqual(numerical.logspace(0.0, 2.0, 3, base=2.0).tolist(), [1.0, 2.0, 4.0])
        self.assertEqual(numerical.geomspace(-1.0, -100.0, 3).tolist(), [-1.0, -10.0, -100.0])
        self.assertEqual(numerical.identity(2, "int64").tolist(), [[1, 0], [0, 1]])
        self.assertEqual(numerical.vander(numerical.asarray([2, 3], "int64")).tolist(), [[2, 1], [3, 1]])
        self.assertEqual(numerical.unique([3.0, 1.0, 3.0]).tolist(), [1.0, 3.0])
        values, index, inverse, counts = numerical.unique([3.0, 1.0, 3.0], True, True, True)
        self.assertEqual((index.tolist(), inverse.tolist(), counts.tolist()), ([1, 0], [1, 0, 1], [1, 2]))
        self.assertEqual(numerical.unique([[1.0, 2.0], [1.0, 2.0], [0.0, 5.0]], axis=0).tolist(), [[0.0, 5.0],
                                                                                                   [1.0, 2.0]])
        self.assertEqual(len(numerical.unique([math.nan, math.nan])), 1)
        self.assertEqual(len(numerical.unique_values([math.nan, math.nan])), 2)
        self.assertEqual(numerical.unique_all([2.0, 2.0]).counts.tolist(), [2])
        self.assertEqual(numerical.intersect1d([1.0, 2.0, 3.0], [3.0, 1.0]).tolist(), [1.0, 3.0])
        self.assertEqual(numerical.union1d([2.0], [1.0]).tolist(), [1.0, 2.0])
        self.assertEqual(numerical.setdiff1d([1.0, 2.0], [2.0]).tolist(), [1.0])
        self.assertEqual(numerical.setxor1d([1.0, 2.0], [2.0, 3.0]).tolist(), [1.0, 3.0])
        self.assertEqual(numerical.isin([[1.0, math.nan]], [math.nan, 1.0]).tolist(), [[True, False]])
        self.assertEqual(numerical.digitize([0.5, 2.5], [3.0, 2.0, 1.0]).tolist(), [3, 1])
        self.assertEqual(numerical.cumulative_sum([1.0, 2.0], include_initial=True).tolist(), [0.0, 1.0, 3.0])
        self.assertEqual([t.tolist() for t in numerical.unstack([[1.0, 2.0]], axis=1)], [[1.0], [2.0]])
        self.assertEqual(numerical.ediff1d([1.0, 4.0], to_begin=-1.0).tolist(), [-1.0, 3.0])
        self.assertEqual(numerical.trim_zeros([0, 1, 0]), [1])
        self.assertEqual(numerical.select([numerical.asarray([True, False])], [1.0], -1).tolist(), [1.0, -1.0])
        inf = math.inf
        self.assertTrue(math.isnan(numerical.prod(numerical.asarray([complex(inf, -inf), 3 + 0j])).real))
        alone = numerical.prod(numerical.asarray([complex(inf, 1)]))  # 1 · (inf + j), from NumPy's identity
        self.assertTrue(alone.real == inf and math.isnan(alone.imag))

    def test_errors(self):
        cases = [
            (ValueError, lambda: numerical.linspace(0.0, 1.0, -1)),
            (ValueError, lambda: numerical.geomspace(0.0, 1.0)),
            (TypeError, lambda: numerical.geomspace(1j, 2j)),
            (ValueError, lambda: numerical.vander([[1.0]])),
            (TypeError, lambda: numerical.identity(2, like=[1.0])),
            (ValueError, lambda: numerical.cumulative_sum([[1.0]])),
            (ValueError, lambda: numerical.unstack(1.0)),
            (ValueError, lambda: numerical.matvec([1.0], [1.0])),
            (ValueError, lambda: numerical.vecmat([1.0], [1.0])),
            (TypeError, lambda: numerical.ediff1d(numerical.asarray([1, 2], "int64"), to_end=0.5)),
            (ValueError, lambda: numerical.trim_zeros([0, 1], "x")),
            (ValueError, lambda: numerical.select([], [])),
            (ValueError, lambda: numerical.select([numerical.asarray([True])], [])),
            (TypeError, lambda: numerical.select([numerical.asarray([1.0])], [1.0])),
            (TypeError, lambda: numerical.digitize([1j], [0.0, 1.0])),
            (ValueError, lambda: numerical.digitize([1.0], [0.0, 2.0, 1.0])),
            (ValueError, lambda: numerical.digitize([1.0], [[0.0, 1.0]])),
            (ValueError, lambda: numerical.isin([1.0], [1.0], kind="hash")),
            (ValueError, lambda: numerical.isin([1.0], [1.0], kind="table")),
        ]
        for error, call in cases:
            with self.assertRaises(error):
                call()


def _tensor(x):
    return numerical.asarray(x, "int64") if isinstance(x, np.ndarray) and x.dtype.kind == "i" else (
        numerical.asarray(x) if isinstance(x, np.ndarray) else x)


@unittest.skipIf(np is None, "NumPy is not installed")
class AgainstNumPy(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.addCleanup(warnings.resetwarnings)
        self.errstate = np.errstate(all="ignore")
        self.errstate.__enter__()
        self.addCleanup(self.errstate.__exit__, None, None, None)

    def assertSame(self, ours, theirs, label, ulps=0):
        try:
            want = theirs()
        except (IndexError, TypeError, ValueError, RuntimeError) as error:
            with self.assertRaises(ValueError if type(error).__name__ == "AxisError" else type(error), msg=label):
                ours()
            return
        self.assertBits(ours(), want, label, ulps)

    def assertBits(self, got, want, label, ulps=0):
        """To the bit; or, `ulps` given, NaNs, infinities and zeros (a zero's sign too) to the bit and every other
        value within that many ulps: a power is C's library's here, NumPy's own vector loop's on x86-64."""
        if isinstance(want, tuple):
            self.assertIsInstance(got, tuple, label)
            self.assertEqual(len(got), len(want), label)
            for g, w in zip(got, want):
                self.assertBits(g, w, label)
            return
        if isinstance(want, list):
            self.assertEqual(got, want, label)
            return
        want = np.asarray(want)
        if want.dtype.kind in "iu":  # NumPy's indices are intp (int32 on a 32-bit Windows)
            want = want.astype(np.int64)
        if isinstance(got, (int, float, complex)) and not isinstance(got, bool):  # NumPy's scalar: a number here
            self.assertEqual(want.shape, (), label)
            got = np.asarray(got, want.dtype)
        got = np.asarray(got)
        self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype), label)
        if want.dtype.kind in "fc":
            self.assertTrue(np.array_equal(np.isnan(got), np.isnan(want)), label)
            got, want = np.where(np.isnan(got), 0, got), np.where(np.isnan(want), 0, want)
        if ulps and want.dtype.kind == "f":
            exact = ~np.isfinite(want) | (want == 0)
            self.assertEqual(np.where(exact, got, 0).tobytes(), np.where(exact, want, 0).tobytes(), label)
            try:
                np.testing.assert_array_max_ulp(got[~exact], want[~exact], maxulp=ulps)
            except AssertionError as error:
                self.fail(f"{label}: {error}")
            return
        self.assertEqual(got.tobytes(), want.tobytes(), label)

    def test_spaces(self):
        ends = (0, 1, -2.5, 3.0, 1e-310, np.array([0.0, 1.0, 2.0]), np.array([1.5, -2.0], dtype=np.float32),
                np.array([[1.0], [2.0]]), 1 + 2j)
        for start in ends:
            for stop in ends:
                for num in (0, 1, 2, 5, 50):
                    for endpoint in (True, False):
                        for dtype in (None, "float32", "int64", "complex128"):
                            if dtype in ("float32", "int64") and (np.iscomplexobj(start) or np.iscomplexobj(stop)):
                                continue  # see differences.md: no complex cast to a real dtype
                            for axis in (0, -1):
                                self.assertSame(
                                    lambda: numerical.linspace(_tensor(start), _tensor(stop), num, endpoint, True,
                                                               dtype, axis),
                                    lambda: np.linspace(start, stop, num, endpoint, True, dtype, axis),
                                    f"linspace {start} {stop} {num} {endpoint} {dtype} {axis}")
                    if np.iscomplexobj(start) or np.iscomplexobj(stop):
                        continue  # a complex power's last bits are its library's
                    for base in (10.0, 2, np.array([2.0, 3.0, 4.0])):
                        self.assertSame(lambda: numerical.logspace(_tensor(start), _tensor(stop), num, endpoint,
                                                                   _tensor(base)),
                                        lambda: np.logspace(start, stop, num, endpoint, base),
                                        f"logspace {start} {stop} {num} {base}", ulps=2)
        for start in (1, 2.5, -3, np.array([1.0, 10.0]), np.array([1.0, -4.0], dtype=np.float32)):
            for stop in (1000, -8.0, np.array([5.0, 7.0]), 1e300):
                for num in (0, 1, 2, 7):
                    for endpoint in (True, False):
                        for dtype in (None, "float32"):
                            for axis in (0, -1):
                                self.assertSame(lambda: numerical.geomspace(_tensor(start), _tensor(stop), num,
                                                                            endpoint, dtype, axis),
                                                lambda: np.geomspace(start, stop, num, endpoint, dtype, axis),
                                                f"geomspace {start} {stop} {num} {dtype} {axis}", ulps=2)
        for dtype in (None, "float32", "int64", bool, "complex128"):
            self.assertSame(lambda: numerical.identity(3, dtype), lambda: np.identity(3, dtype), f"identity {dtype}")

    def drawn(self, rng, kind, shape):
        if kind == "int64":
            return rng.integers(-3, 4, size=shape, dtype=np.int64)
        if kind == "bool":
            return rng.random(shape) < 0.5
        x = np.round(rng.normal(size=shape) * 2) + 0.0  # ties, and no -0 beside a +0 (NumPy's sort is not stable)
        x[rng.random(shape) < 0.1] = np.nan
        x[rng.random(shape) < 0.05] = np.inf
        if kind == "complex128":
            return x + 1j * (np.round(rng.normal(size=shape)) + 0.0)
        return x.astype(kind)

    def test_sets(self):
        rng = np.random.default_rng(7)
        kinds = ("float64", "float32", "int64", "bool", "complex128")
        for kind in kinds:
            for shape in ((7,), (3, 4), (0,), (2, 3, 2)):
                x = self.drawn(rng, kind, shape)
                t = _tensor(x)
                for axis in (None, 0, 1, -1):
                    for flags in ((False, False, False), (True, True, True), (True, False, False), (False, True, True)):
                        for equal_nan in (True, False):
                            if equal_nan and not _NUMPY_2_4 and np.isnan(np.asarray(x, complex)).any():
                                continue  # NumPy before 2.4 keeps a float32's NaNs apart there
                            self.assertSame(lambda: numerical.unique(t, *flags, axis, equal_nan=equal_nan),
                                            lambda: np.unique(x, *flags, axis, equal_nan=equal_nan),
                                            f"unique {kind} {shape} {axis} {flags} {equal_nan}")
                for name in ("unique_counts", "unique_inverse", "unique_all"):
                    self.assertSame(lambda: tuple(getattr(numerical, name)(t)), lambda: tuple(getattr(np, name)(x)),
                                    f"{name} {kind} {shape}")
                if not np.isnan(np.asarray(x, complex)).any():  # NumPy's order of unique_values is its hash's
                    self.assertSame(lambda: numerical.unique_values(t), lambda: np.sort(np.unique_values(x)),
                                    f"unique_values {kind} {shape}")
                for other in kinds:
                    y = self.drawn(rng, other, (5,))
                    u = _tensor(y)
                    for name in ("union1d", "setdiff1d", "setxor1d"):
                        self.assertSame(lambda: getattr(numerical, name)(t, u), lambda: getattr(np, name)(x, y),
                                        f"{name} {kind} {other}")
                    self.assertSame(lambda: numerical.intersect1d(t, u, return_indices=True),
                                    lambda: np.intersect1d(x, y, return_indices=True), f"intersect1d {kind} {other}")
                    for invert in (False, True):
                        for method in (None, "sort", "table"):
                            self.assertSame(lambda: numerical.isin(t, u, invert=invert, kind=method),
                                            lambda: np.isin(x, y, invert=invert, kind=method),
                                            f"isin {kind} {other} {invert} {method}")
        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, np.nan])
        for bins in ([0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [np.nan, 1.0],
                     [1.0, np.nan], [0.0], [3.0, 2.0, 2.0, 1.0]):
            for right in (False, True):
                self.assertSame(lambda: numerical.digitize(numerical.asarray(x), numerical.asarray(bins), right),
                                lambda: np.digitize(x, bins, right), f"digitize {bins} {right}")

    def test_array_functions(self):
        rng = np.random.default_rng(8)
        for kind in ("float64", "float32", "int64", "complex128"):
            for shape in ((7,), (3, 4), (0,), (2, 3, 2)):
                x = self.drawn(rng, kind, shape)
                t = _tensor(x)
                for axis in (None, 0, -1, 1, 5):
                    for include_initial in (False, True):
                        for name in ("cumulative_sum", "cumulative_prod"):
                            self.assertSame(lambda: getattr(numerical, name)(t, axis=axis,
                                                                             include_initial=include_initial),
                                            lambda: getattr(np, name)(x, axis=axis, include_initial=include_initial),
                                            f"{name} {kind} {shape} {axis} {include_initial}")
                for axis in (0, 1, -1):
                    if len(shape) > 1:  # a vector's slices are numbers here, NumPy's scalars there
                        self.assertSame(lambda: numerical.unstack(t, axis=axis), lambda: np.unstack(x, axis=axis),
                                        f"unstack {kind} {shape} {axis}")
                for extra in ({}, {"to_begin": [9]}, {"to_end": 5, "to_begin": [[1, 2]]}, {"to_end": 2.5},
                              {"to_begin": 1j}):
                    self.assertSame(lambda: numerical.ediff1d(t, **extra), lambda: np.ediff1d(x, **extra),
                                    f"ediff1d {kind} {shape} {extra}")
            for n in (None, 0, 1, 3, 6):
                for increasing in (False, True):
                    x = self.drawn(rng, kind, (4,))
                    self.assertSame(lambda: numerical.vander(_tensor(x), n, increasing),
                                    lambda: np.vander(x, n, increasing), f"vander {kind} {n} {increasing}")
        for filt in ([0, 0, 1, 2, 0], [0.0, 0.0], [1, 0, 3], np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                     np.array([[0, 1], [0, 0]])):
            for trim in ("fb", "f", "b", "bF"):
                for axis in ((None, 0, -1) + (((),) if isinstance(filt, np.ndarray) else ())) if _NUMPY_2_4 else (None,):
                    ours = filt if isinstance(filt, list) else _tensor(filt)  # `axis` from NumPy 2.4 on
                    self.assertSame(lambda: numerical.trim_zeros(ours, trim, axis),
                                    lambda: np.trim_zeros(filt, trim, *(() if axis is None else (axis,))),
                                    f"trim_zeros {filt} {trim} {axis}")
        conditions = [np.array([True, False, True]), np.array([False, True, True])]
        for choices, default in (([1, 2], 0), ([np.array([1.0, 2.0, 3.0]), 2.5], -1),
                                 ([np.array([1.0, 2.0, 3.0], dtype=np.float32), 2.5], 0),
                                 ([np.array([1, 2, 3]), np.array([4, 5, 6])], 0.5), ([1j, 2], 0)):
            self.assertSame(lambda: numerical.select([numerical.asarray(c) for c in conditions],
                                                     [_tensor(c) for c in choices], default),
                            lambda: np.select(conditions, choices, default), f"select {choices} {default}")
        a, v = rng.normal(size=(2, 3, 4)), rng.normal(size=(4,))
        np.testing.assert_allclose(np.asarray(numerical.matvec(a, v)), np.matvec(a, v), rtol=1e-14, atol=1e-15)
        z, m = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3)), rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
        np.testing.assert_allclose(np.asarray(numerical.vecmat(z, m)), np.vecmat(z, m), rtol=1e-14, atol=1e-15)

    def test_complex_sums(self):
        rng = np.random.default_rng(10)
        for dtype in (np.complex128, np.complex64):
            for length in list(range(1, 70)) + [100, 129, 200, 1000, 5000] + ([70000, 300000] if _NUMPY_2_3 else []):
                # NumPy before 2.3 sums past 8192 elements a buffer of them at a time
                x = (rng.normal(size=length) + 1j * rng.normal(size=length)).astype(dtype)
                self.assertBits(numerical.sum(numerical.asarray(x)), np.sum(x), f"sum {dtype.__name__} {length}")
            m = (rng.normal(size=(5, 300)) + 1j * rng.normal(size=(5, 300))).astype(dtype)
            self.assertBits(numerical.sum(numerical.asarray(m), axis=-1), np.sum(m, axis=-1), "sum along the last axis")
            self.assertBits(numerical.mean(numerical.asarray(m)), np.mean(m), "mean")

    def test_complex_products(self):
        rng = np.random.default_rng(9)
        for shape in ((30,), (30, 1), (30, 2), (30, 3), (6, 30, 3), (6, 30, 1), (3, 30), (1, 30), (2, 2, 30), (2,), (3,),
                      (2, 3), (3, 2), (0, 3)):
            x = rng.normal(size=shape) + 1j * rng.normal(size=shape)
            x.ravel()[rng.random(x.size) < 0.1] = complex(np.inf, -np.inf)
            x.ravel()[rng.random(x.size) < 0.05] = complex(np.nan, 1.0)
            t = numerical.asarray(x)
            self.assertSame(lambda: numerical.prod(t), lambda: np.prod(x), f"prod {shape}")
            self.assertSame(lambda: numerical.cumprod(t), lambda: np.cumprod(x), f"cumprod {shape}")
            for axis in range(len(shape)):
                self.assertSame(lambda: numerical.prod(t, axis=axis), lambda: np.prod(x, axis=axis),
                                f"prod {shape} {axis}")
                self.assertSame(lambda: numerical.cumprod(t, axis=axis), lambda: np.cumprod(x, axis=axis),
                                f"cumprod {shape} {axis}")
            if len(shape) == 3:
                for axes in ((0, 1), (1, 2), (0, 2)):
                    self.assertSame(lambda: numerical.prod(t, axis=axes), lambda: np.prod(x, axis=axes),
                                    f"prod {shape} {axes}")


if __name__ == "__main__":
    unittest.main()
