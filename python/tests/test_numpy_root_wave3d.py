"""interp, the windows (hamming, hanning, blackman, bartlett, kaiser) and i0, the polynomials (polyval, polyder,
polyint, polyadd, polysub, polymul, polydiv, roots, polyfit), histogramdd and histogram2d, compress, extract and choose
against NumPy's answers: to the bit, but within a few ulps for what goes through a library's cos, exp or sqrt (NumPy's
own loops on x86-64), a convolution or a least-squares fit; with NumPy's errors. Deterministic; the comparisons skipped
without NumPy."""
import math
import unittest
import warnings

import numerical

try:
    import numpy as np
except ImportError:
    np = None


class WithoutNumPy(unittest.TestCase):
    def test_values(self):
        self.assertEqual(numerical.interp([0.5, 2.5, -1.0], [0.0, 1.0, 2.0], [0.0, 10.0, 20.0]).tolist(),
                         [5.0, 20.0, 0.0])
        self.assertEqual(numerical.interp(0.5, [0.0, 1.0], [0.0, 2.0], left=-1.0), 1.0)
        self.assertTrue(math.isnan(numerical.interp(math.nan, [0.0, 1.0], [0.0, 2.0])))
        self.assertEqual(numerical.interp([5.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], period=4).tolist(), [1.0])
        self.assertEqual(numerical.interp([0.5], [0.0, 1.0], [0.0, 2j]).tolist(), [1j])
        self.assertEqual(numerical.hamming(1).tolist(), [1.0])
        self.assertEqual(numerical.bartlett(3).tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(numerical.hanning(0).shape, (0,))
        self.assertEqual(numerical.i0(0.0), 1.0)
        self.assertEqual(numerical.kaiser(1, 5.0).tolist(), [1.0])
        self.assertEqual(numerical.polyval([1, 2, 3], 2), 11)
        self.assertEqual(numerical.polyder([1, 2, 3]).tolist(), [2, 2])
        self.assertEqual(numerical.polyint([3.0, 2.0]).tolist(), [1.5, 2.0, 0.0])
        self.assertEqual(numerical.polyadd([1, 2], [1, 1, 1]).tolist(), [1, 2, 3])
        self.assertEqual(numerical.polysub([1, 2], [1]).tolist(), [1, 1])
        self.assertEqual(numerical.polymul([1, 1], [1, -1]).tolist(), [1, 0, -1])
        quotient, remainder = numerical.polydiv([1.0, 0.0, -1.0], [1.0, 1.0])
        self.assertEqual((quotient.tolist(), remainder.tolist()), ([1.0, -1.0], [0.0]))
        self.assertEqual(sorted(z.real for z in numerical.roots([1.0, -3.0, 2.0]).tolist()), [1.0, 2.0])
        self.assertEqual(numerical.roots([0.0, 0.0]).shape, (0,))
        self.assertEqual(len(numerical.polyfit([0.0, 1.0, 2.0], [1.0, 3.0, 5.0], 1)), 2)
        hist, edges = numerical.histogramdd(numerical.asarray([[0.0, 0.0], [1.0, 1.0]]), bins=2)  # two points
        self.assertEqual((hist.tolist(), [e.tolist() for e in edges]), ([[1.0, 0.0], [0.0, 1.0]],
                                                                       [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]))
        hist, _ = numerical.histogramdd([[0.0, 0.0], [1.0, 1.0]], bins=2)  # a list: two coordinate vectors
        self.assertEqual(hist.tolist(), [[0.0, 0.0], [0.0, 2.0]])
        self.assertEqual(numerical.histogram2d([0.0, 1.0], [0.0, 1.0], bins=2)[0].tolist(), [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(numerical.compress([True, False, True], [1.0, 2.0, 3.0]).tolist(), [1.0, 3.0])
        self.assertEqual(numerical.extract([[True, False], [False, True]], [[1.0, 2.0], [3.0, 4.0]]).tolist(),
                         [1.0, 4.0])
        self.assertEqual(numerical.choose([0, 1, 1], [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]).tolist(),
                         [1.0, 20.0, 30.0])

    def test_errors(self):
        cases = [
            (ValueError, lambda: numerical.interp([0.0], [], [])),
            (ValueError, lambda: numerical.interp([0.0], [0.0, 1.0], [0.0])),
            (ValueError, lambda: numerical.interp([0.0], [0.0, 1.0], [0.0, 1.0], period=0)),
            (TypeError, lambda: numerical.i0([1j])),
            (ValueError, lambda: numerical.polyder([1.0], -1)),
            (ValueError, lambda: numerical.polyint([1.0], -1)),
            (ValueError, lambda: numerical.polyint([1.0], 3, k=[1.0, 2.0])),
            (ValueError, lambda: numerical.roots([[1.0]])),
            (ValueError, lambda: numerical.polyfit([0.0, 1.0], [0.0, 1.0], -1)),
            (TypeError, lambda: numerical.polyfit([[0.0]], [0.0], 1)),
            (TypeError, lambda: numerical.polyfit([0.0, 1.0], [0.0], 1)),
            (ValueError, lambda: numerical.histogramdd([[0.0, 1.0]], bins=[2, 2, 2])),
            (ValueError, lambda: numerical.histogramdd([[0.0, 1.0]], bins=0)),
            (ValueError, lambda: numerical.histogram2d([0.0], [0.0, 1.0])),
            (IndexError, lambda: numerical.compress([False, False, False, True], [1.0, 2.0, 3.0])),
            (IndexError, lambda: numerical.extract([True, False, True, True], [1.0, 2.0, 3.0])),
            (ValueError, lambda: numerical.choose([2], [[1.0], [2.0]])),
            (ValueError, lambda: numerical.choose([0], [[1.0]], mode="bounce")),
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

    def assertSame(self, ours, theirs, label, ulps=0, close=False):
        try:
            want = theirs()
        except (IndexError, TypeError, ValueError) as error:
            with self.assertRaises(ValueError if type(error).__name__ == "AxisError" else type(error), msg=label):
                ours()
            return
        got = ours()
        if isinstance(want, (tuple, list)):
            self.assertEqual(len(got), len(want), label)
            for g, w in zip(got, want):
                self.assertSame(lambda g=g: g, lambda w=w: w, label, ulps, close)
            return
        if isinstance(got, (int, float, complex)) and not isinstance(got, bool):  # NumPy's scalar: a number here
            self.assertEqual(np.shape(want), (), label)
            got = np.asarray(got, np.asarray(want).dtype)
        got, want = np.asarray(got), np.asarray(want)
        if want.dtype.kind in "iu":  # NumPy's indices are intp (int32 on a 32-bit Windows)
            want = want.astype(np.int64)
        self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype), label)
        if close:
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9, err_msg=label)
            return
        if want.dtype.kind in "fc":
            self.assertTrue(np.array_equal(np.isnan(got), np.isnan(want)), label)
            got, want = np.where(np.isnan(got), 0, got), np.where(np.isnan(want), 0, want)
        if ulps:
            for g, w in ((got.real, want.real), (got.imag, want.imag)) if want.dtype.kind == "c" else ((got, want),):
                exact = ~np.isfinite(w) | (w == 0)
                self.assertEqual(np.where(exact, g, 0).tobytes(), np.where(exact, w, 0).tobytes(), label)
                try:
                    np.testing.assert_array_max_ulp(g[~exact], w[~exact], maxulp=ulps)
                except AssertionError as error:
                    self.fail(f"{label}: {error}")
            return
        self.assertEqual(got.tobytes(), want.tobytes(), label)

    def test_interp(self):
        rng = np.random.default_rng(15)
        for count in (1, 2, 3, 4, 5, 9, 50):
            xp = np.sort(rng.normal(size=count))
            for fp in (rng.normal(size=count), rng.normal(size=count) + 1j * rng.normal(size=count),
                       rng.integers(-5, 5, size=count)):
                for x in (rng.uniform(-3, 3, size=200), np.concatenate([xp, [np.nan, np.inf, -np.inf]]), 0.25):
                    for left, right in ((None, None), (-7.0, 9.0)):
                        if np.iscomplexobj(fp) and left is not None:
                            left, right = complex(left, 1), complex(right, -1)
                        self.assertSame(lambda: numerical.interp(_tensor(x), _tensor(xp), _tensor(fp), left, right),
                                        lambda: np.interp(x, xp, fp, left, right), f"interp {count} {fp.dtype}")
                    for period in (2.5, -4.0):
                        self.assertSame(lambda: numerical.interp(_tensor(x), _tensor(xp), _tensor(fp), period=period),
                                        lambda: np.interp(x, xp, fp, period=period), f"interp period {count}")
        xp = np.array([0.0, 1.0, 1.0, 2.0])  # a repeated point: NumPy's search takes the later
        fp = np.array([0.0, 1.0, 5.0, 6.0])
        self.assertSame(lambda: numerical.interp(numerical.asarray([1.0, 0.9, 1.1]), numerical.asarray(xp),
                                                 numerical.asarray(fp)),
                        lambda: np.interp([1.0, 0.9, 1.1], xp, fp), "interp repeated")

    def test_windows(self):
        for M in (0, 1, 2, 5, 12, 101, -3, 4.0):
            for name in ("hamming", "hanning", "blackman", "bartlett"):
                self.assertSame(lambda: getattr(numerical, name)(M), lambda: getattr(np, name)(M), f"{name} {M}",
                                ulps=0 if name == "bartlett" else 4)
            for beta in (0.0, 5.0, 14.0):
                if M == 0 or M < 0:
                    continue
                self.assertSame(lambda: numerical.kaiser(M, beta), lambda: np.kaiser(M, beta), f"kaiser {M} {beta}",
                                ulps=8)
        x = np.array([0.0, -0.5, 1.0, 7.9, 8.0, 8.1, 30.0, 700.0, np.nan, np.inf])
        for values in (x, x.astype(np.float32), np.array([1, 2, 3]), 2.5):
            self.assertSame(lambda: numerical.i0(_tensor(values)), lambda: np.i0(values), f"i0 {values}", ulps=8)

    def test_polynomials(self):
        rng = np.random.default_rng(16)
        for p in ([1.0, 2.0, 3.0], [3, 0, -1, 2], rng.normal(size=6), [2.0], [1j, 2.0]):
            p = np.asarray(p)
            for x in (2.5, np.array([0.0, 1.5, -2.0]), np.array([1, 2, 3]), np.array([[1.0, 2.0]])):
                self.assertSame(lambda: numerical.polyval(_tensor(p), _tensor(x)), lambda: np.polyval(p, x),
                                f"polyval {p} {x}")
            for m in (0, 1, 2, 5):
                self.assertSame(lambda: numerical.polyder(_tensor(p), m), lambda: np.polyder(p, m), f"polyder {m}")
                if not np.iscomplexobj(p):
                    for k in (None, 3.0, [1.0, 2.0, 3.0, 4.0, 5.0]):
                        self.assertSame(lambda: numerical.polyint(_tensor(p), m, k), lambda: np.polyint(p, m, k),
                                        f"polyint {m} {k}")
            for q in ([1.0, -1.0], [2, 3, 4, 5, 6, 7, 8], [5.0]):
                q = np.asarray(q)
                for name in ("polyadd", "polysub"):
                    self.assertSame(lambda: getattr(numerical, name)(_tensor(p), _tensor(q)),
                                    lambda: getattr(np, name)(p, q), f"{name} {p} {q}")
                self.assertSame(lambda: numerical.polymul(_tensor(p), _tensor(q)), lambda: np.polymul(p, q),
                                f"polymul {p} {q}", close=True)
                self.assertSame(lambda: numerical.polydiv(_tensor(p), _tensor(q)), lambda: np.polydiv(p, q),
                                f"polydiv {p} {q}", close=True)
            ours = np.sort_complex(np.asarray(numerical.roots(_tensor(p)), complex))
            theirs = np.sort_complex(np.asarray(np.roots(p), complex))
            np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9, err_msg=f"roots {p}")
        for x, y in ((np.linspace(0, 1, 20), rng.normal(size=20)),
                     (np.linspace(-1, 2, 15), rng.normal(size=(15, 3)))):
            for deg in (0, 1, 3):
                self.assertSame(lambda: numerical.polyfit(_tensor(x), _tensor(y), deg),
                                lambda: np.polyfit(x, y, deg), f"polyfit {deg}", close=True)
                self.assertSame(lambda: numerical.polyfit(_tensor(x), _tensor(y), deg, w=_tensor(x + 1.0), cov=True),
                                lambda: np.polyfit(x, y, deg, w=x + 1.0, cov=True), f"polyfit cov {deg}", close=True)
            c, residues, rank, singular, rcond = numerical.polyfit(_tensor(x), _tensor(y), 2, full=True)
            theirs = np.polyfit(x, y, 2, full=True)
            self.assertEqual(rank, theirs[2])
            np.testing.assert_allclose(np.asarray(residues), theirs[1], rtol=1e-9)

    def test_histograms_and_selections(self):
        rng = np.random.default_rng(17)
        sample = rng.normal(size=(500, 3))
        for bins in (5, (3, 4, 5), [np.array([-3.0, 0.0, 3.0]), 4, np.linspace(-2, 2, 7)]):
            for limits in (None, [(-1, 1), (-2, 2), (-3, 3)]):
                for weights in (None, rng.random(500)):
                    for density in (None, True):
                        self.assertSame(
                            lambda: numerical.histogramdd(_tensor(sample), [_tensor(b) for b in bins]
                                                          if isinstance(bins, list) else bins, limits, density,
                                                          _tensor(weights)),
                            lambda: tuple([np.histogramdd(sample, bins, limits, density, weights)[0],
                                           np.histogramdd(sample, bins, limits, density, weights)[1]]),
                            f"histogramdd {bins} {limits} {density}")
        x, y = rng.normal(size=300), rng.normal(size=300)
        for bins in (4, (3, 6), np.array([-2.0, 0.0, 2.0])):
            self.assertSame(lambda: numerical.histogram2d(_tensor(x), _tensor(y), _tensor(bins)),
                            lambda: np.histogram2d(x, y, bins), f"histogram2d {bins}")
        a = rng.normal(size=(3, 4))
        for condition in ([True, False, True], [False, True], [True] * 4, [False, False, False, True]):
            for axis in (None, 0, 1):
                self.assertSame(lambda: numerical.compress(condition, _tensor(a), axis),
                                lambda: np.compress(condition, a, axis), f"compress {condition} {axis}")
        for condition in ([True, False, True], np.ones((3, 4), bool), (a > 0)):
            self.assertSame(lambda: numerical.extract(_tensor(condition) if isinstance(condition, np.ndarray)
                                                      else condition, _tensor(a)),
                            lambda: np.extract(condition, a), f"extract {condition}")
        choices = [rng.normal(size=4), rng.normal(size=4), 7.5]
        for index in (np.array([0, 1, 2, 1]), np.array([[0, 2, 1, 0]]), np.array([3, -1, 0, 5])):
            for mode in ("raise", "wrap", "clip"):
                self.assertSame(lambda: numerical.choose(_tensor(index), [_tensor(c) for c in choices], mode=mode),
                                lambda: np.choose(index, choices, mode=mode), f"choose {index} {mode}")


if __name__ == "__main__":
    unittest.main()
