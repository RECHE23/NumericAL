"""NumPy's keywords a call written for it passes: einsum's optimize (taken, not followed), diff's prepend and append,
initial for max, min, nanmax and nanmin (cast to the reduction's dtype, alone over no element), dtype for concatenate,
stack, cumsum and cumprod, correction for nanvar and nanstd; against NumPy's answers, to the bit. Deterministic;
skipped without NumPy."""
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy not installed")
class AgainstNumPy(unittest.TestCase):
    def test_keywords(self):
        a = np.arange(6.0).reshape(2, 3)
        nans = np.array([[np.nan, np.nan], [1.0, np.nan]])
        cases = {
            "einsum optimize": lambda m: m.einsum("ij,jk->ik", a, a.T, optimize=True),
            "einsum path": lambda m: m.einsum("ij,jk->ik", a, a.T, optimize=["einsum_path", (0, 1)]),
            "diff prepend": lambda m: m.diff(a, prepend=0),
            "diff append": lambda m: m.diff(a, axis=0, append=a[:1]),
            "diff both": lambda m: m.diff(a, n=2, prepend=-1.5, append=9),
            "diff none": lambda m: m.diff(a, n=0, prepend=1),
            "max initial": lambda m: m.max(a, axis=1, initial=4.5),
            "max empty": lambda m: m.max(np.zeros((0, 3)), axis=0, initial=-1),
            "max nan": lambda m: m.max(np.array([np.nan, 1.0]), initial=5.0),
            "min keepdims": lambda m: m.min(a, axis=0, keepdims=True, initial=0.5),
            "min empty whole": lambda m: m.min(np.zeros((2, 0)), initial=7.0),
            "nanmax": lambda m: m.nanmax(nans, axis=1, initial=0.5),
            "nanmin": lambda m: m.nanmin(np.array([np.nan]), initial=3.0),
            "cumsum": lambda m: m.cumsum(a, 1, np.float32),
            "cumprod": lambda m: m.cumprod(a, None, "float32"),
            "concatenate": lambda m: m.concatenate([a, a], 0, None, dtype=np.float32),
            "stack": lambda m: m.stack([a, a], dtype="complex128"),
            "nanvar": lambda m: m.nanvar(np.array([1.0, np.nan, 3.0, 4.0]), correction=1),
            "nanstd": lambda m: m.nanstd(a, axis=0, correction=1),
        }
        for name, call in cases.items():
            with self.subTest(name=name):
                want = np.asarray(call(np))
                got = np.asarray(call(numerical))
                self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape))
                np.testing.assert_array_equal(got, want)
        integers = numerical.asarray(np.array([1, 2]), "int64")
        self.assertEqual(numerical.max(integers, initial=2.5), 2)  # NumPy's cast of `initial`: toward 0
        with self.assertRaises(TypeError):
            numerical.einsum("i->", a[0], optimize="fastest")
        for name in ("concatenate", "stack", "cumsum", "cumprod"):
            with self.subTest(out=name), self.assertRaises(TypeError):
                getattr(numerical, name)([a, a] if name in ("concatenate", "stack") else a, 0, out=numerical.zeros(1))


if __name__ == "__main__":
    unittest.main()
