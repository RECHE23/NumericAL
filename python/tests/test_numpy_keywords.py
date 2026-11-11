"""NumPy's keywords a call written for it passes: einsum's optimize (taken, not followed), diff's prepend and append,
initial for sum, prod, max, min, nanmax and nanmin (cast to the reduction's dtype, alone over no element), dtype for concatenate,
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

    def test_sum_and_prod_from_initial(self):
        # prod's first factor everywhere and sum's start where NumPy's order is NumericAL's, to the bit: the reduced
        # axes the last ones (NumPy's pairwise sum, its accumulator from initial), or columns of fewer than 8 terms
        # (4 complex ones, NumPy's pairwise sum in order there); longer columns pairwise from it, near NumPy's
        rng = np.random.default_rng(5)
        for trial in range(600):
            rank = int(rng.integers(0, 4))
            shape = tuple(int(rng.choice([0, 1, 2, 3, 5, 9])) for _ in range(rank))
            dtype = ("float64", "float32", "complex128", "complex64")[trial % 4]
            a = (1 + 0.3 * rng.standard_normal(shape) + (0.3j * rng.standard_normal(shape) if dtype[0] == "c" else 0))
            a = a.astype(dtype)
            t = numerical.asarray(a)
            if trial % 5 == 0 and rank and shape[-1] > 1:  # a strided view
                a, t = a[..., ::2], t[..., ::2]
            initial = (0.5, -0.0, 3.0, 1e-7)[trial % 4] + (0.25j if dtype[0] == "c" and trial % 3 else 0)
            count = int(rng.integers(-1, len(a.shape) + 1))
            axis = None if count < 0 else tuple(sorted(int(k) for k in rng.choice(len(a.shape), count, replace=False)))
            keepdims = trial % 3 == 1
            for name in ("sum", "prod"):
                with self.subTest(name=name, dtype=dtype, shape=a.shape, axis=axis, keepdims=keepdims, initial=initial):
                    want = np.asarray(getattr(np, name)(a, axis=axis, keepdims=keepdims, initial=initial))
                    got = np.asarray(getattr(numerical, name)(t, axis, keepdims, initial=initial), want.dtype)
                    self.assertEqual(got.shape, want.shape)
                    axes = range(len(a.shape)) if axis is None else axis
                    order = [k for k, extent in enumerate(a.shape) if extent != 1]
                    reduced = [k for k in order if k in axes]
                    terms = (int(np.prod([a.shape[k] for k in axes])) + 1) * (2 if dtype[0] == "c" else 1)
                    if name == "prod" or order[len(order) - len(reduced):] == reduced or terms < 8:
                        self.assertEqual(got.tobytes(), want.tobytes())
                    else:
                        np.testing.assert_allclose(got, want, rtol=1e-5 if dtype in ("float32", "complex64") else 1e-13)

    def test_initial_as_numpy_casts_it(self):
        a = numerical.asarray([[1.5, 2.0], [3.0, 4.0]])
        self.assertEqual(a.sum(initial=1), np.sum(np.asarray(a), initial=1))
        self.assertEqual(np.prod(a, axis=0, initial=-2).tolist(), [-9.0, -16.0])  # NumPy's call of the method
        self.assertEqual(a.max(axis=1, initial=3).tolist(), [3.0, 4.0])
        self.assertEqual(a.min(initial=None), 1.5)  # None: no initial, as NumPy's
        self.assertEqual(numerical.sum(a, initial=None), 10.5)
        self.assertEqual(numerical.prod(a, initial=None), 36.0)
        integers = numerical.asarray(np.array([1, 2]), "int64")
        self.assertEqual(numerical.sum(integers, initial=2.9), 5)  # truncated toward 0, as NumPy's
        self.assertEqual(numerical.prod(integers, initial=-2.9), -4)
        self.assertEqual(numerical.sum(a, dtype="float32", initial=0.1), np.sum(np.asarray(a), dtype=np.float32,
                                                                              initial=0.1))
        self.assertEqual(numerical.prod(a, dtype="complex128", initial=1j), 36j)
        self.assertEqual(np.signbit(numerical.sum(numerical.zeros((0,)), initial=-0.0)), True)  # alone, -0.0 kept
        self.assertEqual(numerical.sum(numerical.asarray([True, True]), initial=1), 3)
        for name in ("sum", "prod", "max", "min"):
            with self.subTest(name=name), self.assertRaises(TypeError):  # a complex start for a real reduction
                getattr(numerical, name)(a, initial=1j)
        with self.assertRaises(TypeError):
            a.mean(initial=1.0)  # NumPy's mean takes none


if __name__ == "__main__":
    unittest.main()
