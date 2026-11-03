"""numerical.linalg against numpy.linalg: the same names, the root's functions, and cross and outer as
numpy.linalg's (3-vectors along one axis, vectors) where the root's take numpy.cross's and numpy.outer's arguments,
every bit of the products as NumPy's (zeros' signs, infinities). Deterministic; the NumPy comparisons skipped
without NumPy."""
import unittest
import warnings

import numerical

try:
    import numpy as np
except ImportError:
    np = None

_NUMPY_2_5 = np is not None and tuple(int(part) for part in np.__version__.split(".")[:2]) >= (2, 5)


class WithoutNumPy(unittest.TestCase):
    def test_names(self):
        for name in numerical.linalg.__all__:
            if name not in ("cross", "outer"):
                self.assertIs(getattr(numerical.linalg, name), getattr(numerical, name), name)
        self.assertIs(numerical.linalg.LinAlgError, numerical.LinAlgError)

    def test_cross_and_outer(self):
        self.assertEqual(numerical.linalg.cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]).tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(numerical.linalg.outer([1.0, 2.0], [3.0]).tolist(), [[3.0], [6.0]])
        self.assertEqual(numerical.cross([[1.0], [0.0], [0.0]], [[0.0, 1.0, 0.0]], axisa=0).tolist(),
                         [[0.0, 0.0, 1.0]])
        self.assertEqual(numerical.outer([[1.0, 2.0]], [[3.0], [4.0]]).tolist(), [[3.0, 4.0], [6.0, 8.0]])
        cases = [
            (ValueError, lambda: numerical.linalg.cross([1.0, 2.0], [3.0, 4.0])),
            (ValueError, lambda: numerical.linalg.cross([1.0, 2.0, 3.0], 1.0)),
            (TypeError, lambda: numerical.linalg.cross([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], axis=(0,))),
            (ValueError, lambda: numerical.linalg.outer([[1.0]], [1.0])),
            (ValueError, lambda: numerical.cross([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0])),
            (ValueError, lambda: numerical.cross([1.0, 2.0], [1.0, 2.0, 3.0])),
            (ValueError, lambda: numerical.cross(1.0, [1.0, 2.0, 3.0])),
            (ValueError, lambda: numerical.cross([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], axisc=1)),
            (TypeError, lambda: numerical.outer([1.0], [1.0], out=numerical.zeros((1, 1)))),
        ]
        for error, call in cases:
            with self.assertRaises(error):
                call()


@unittest.skipIf(np is None, "NumPy is not installed")
class AgainstNumPy(unittest.TestCase):
    def test_names(self):
        self.assertEqual(sorted(numerical.linalg.__all__), sorted(np.linalg.__all__))

    def test_cross_and_outer_bits(self):
        rng = np.random.default_rng(4)

        def drawn(shape, kind):
            if kind == "int64":
                return rng.integers(-5, 5, size=shape, dtype=np.int64)
            x = rng.normal(size=shape)
            x[rng.random(shape) < 0.15] = 0.0
            x[rng.random(shape) < 0.1] = -0.0
            x[rng.random(shape) < 0.05] = np.inf
            if kind == "complex128":
                return x + 1j * rng.normal(size=shape)
            return x.astype(kind)

        def outcome(call):
            try:
                return call()
            except (IndexError, TypeError, ValueError) as error:
                return ValueError if type(error).__name__ == "AxisError" else type(error)

        shapes = ((3,), (2,), (4, 3), (4, 2), (3, 4), (2, 4), (5, 1, 3), (1, 6, 2), (3, 3), (2, 2), (4,), (), (2, 3, 2))
        arguments = ({}, {"axis": 0}, {"axisa": 0}, {"axisb": 0, "axisc": 0}, {"axisc": -2}, {"axis": 1},
                     {"axisa": -2, "axisb": 0, "axisc": 1})
        with warnings.catch_warnings(), np.errstate(all="ignore"):  # NumPy warns of its infinities times zeros
            warnings.simplefilter("ignore", DeprecationWarning)
            for kind in ("float64", "float32", "complex128", "int64"):
                for first in shapes:
                    for second in shapes:
                        if not _NUMPY_2_5 and 2 in (first[-1:] + second[-1:] + first[:1] + second[:1]):
                            continue  # 2-vectors: NumPy before 2.5 crossed them (deprecated), 2.5 refuses them
                        x, y = drawn(first, kind), drawn(second, kind)
                        t, u = (numerical.asarray(v, "int64" if kind == "int64" else None) for v in (x, y))
                        pairs = [(lambda kw=kw: numerical.cross(t, u, **kw), lambda kw=kw: np.cross(x, y, **kw),
                                  f"cross {kw}") for kw in arguments]
                        if _NUMPY_2_5 or (first and second):  # a rank 0: IndexError before NumPy 2.5
                            pairs.append((lambda: numerical.linalg.cross(t, u), lambda: np.linalg.cross(x, y),
                                          "linalg.cross"))
                        pairs += [(lambda: numerical.outer(t, u), lambda: np.outer(x, y), "outer"),
                                  (lambda: numerical.linalg.outer(t, u), lambda: np.linalg.outer(x, y), "linalg.outer")]
                        for ours, theirs, name in pairs:
                            label = f"{name} {kind} {first} {second}"
                            want, got = outcome(theirs), outcome(ours)
                            if isinstance(want, type):
                                self.assertTrue(isinstance(got, type) and issubclass(got, want), label)
                                continue
                            got = np.asarray(got)
                            self.assertEqual((got.dtype, got.shape, got.tobytes()),
                                             (want.dtype, want.shape, want.tobytes()), label)


if __name__ == "__main__":
    unittest.main()
