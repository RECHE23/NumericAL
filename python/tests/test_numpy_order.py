"""NumPy's argument order and names, for a call written for NumPy: the reductions' ``axis, dtype, out, keepdims``
(``axis, out, keepdims`` for max, min, any, all, ptp; ``out`` before ``ddof`` for var and std, before ``mode`` for
take), the ``*_like`` functions' ``order`` and ``subok`` before ``shape``, and NumPy's parameter names (``fill_value``,
``a1``/``a2``, ``prototype``, ``val``, tensordot's ``a``, ``b`` and positional ``axes``); against NumPy's answers, to
the bit. `out` None alone, as a Tensor is never written into. Deterministic; skipped without NumPy."""
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy not installed")
class AgainstNumPy(unittest.TestCase):
    def test_positions(self):
        a = np.arange(6.0).reshape(2, 3) - 2.5
        mask = a > 0
        cases = {
            "sum dtype": lambda m: m.sum(a, 0, np.float32),
            "sum keepdims": lambda m: m.sum(a, 1, None, None, True),
            "mean dtype": lambda m: m.mean(a, 1, "float32"),
            "prod": lambda m: m.prod(a, 0, None, None, True),
            "max keepdims": lambda m: m.max(a, 0, None, True),
            "max initial": lambda m: m.max(a, 1, None, False, 0.75),
            "min keepdims": lambda m: m.min(a, 1, None, True),
            "amax": lambda m: m.amax(a, 0, None, True),
            "any": lambda m: m.any(mask, 0, None, True),
            "all": lambda m: m.all(mask, 1, None, True),
            "ptp": lambda m: m.ptp(a, 1, None, True),
            "argmax": lambda m: m.argmax(a, 1, None),
            "argmin": lambda m: m.argmin(a, 0, None, keepdims=True),
            "var": lambda m: m.var(a, 0, None, None, 1),
            "std keepdims": lambda m: m.std(a, 1, np.float32, None, 1, True),
            "take": lambda m: m.take(a, [0, 4], 1, None, "clip"),
            "zeros_like": lambda m: m.zeros_like(a, np.float32, "C", True, (2, 2)),
            "ones_like": lambda m: m.ones_like(a, None, "K", True, 3),
            "empty_like": lambda m: m.empty_like(prototype=a, shape=(1, 2)).shape,
            "full_like": lambda m: m.full_like(a, 2.5, None, "F", True, (3,)),
            "full": lambda m: m.full((2, 2), fill_value=1.5, dtype=np.float32),
            "full order": lambda m: m.full(3, 2.0, None, "F"),
            "zeros order": lambda m: m.zeros(2, "float32", "F"),
            "ones order": lambda m: m.ones((1, 2), None, "C"),
            "asarray": lambda m: m.asarray(a=[1.0, 2.0], dtype=None, order="K"),
            "array_equal": lambda m: m.array_equal(a1=a, a2=a.copy()),
            "real": lambda m: m.real(val=a + 1j),
            "imag": lambda m: m.imag(val=a + 2j),
            "tensordot": lambda m: m.tensordot(a, a.T, 1),
            "tensordot named": lambda m: m.tensordot(a=a, b=a, axes=([0, 1], [0, 1])),
        }
        for name, call in cases.items():
            with self.subTest(name=name):
                want = np.asarray(call(np))
                got = np.asarray(call(numerical))
                self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape))
                np.testing.assert_array_equal(got, want)

    def test_methods(self):
        a = np.arange(6.0).reshape(2, 3) - 2.5
        t = numerical.asarray(a)
        cases = {
            "sum": lambda x: x.sum(0, np.float32, None, True),
            "mean": lambda x: x.mean(1, None, None, True),
            "prod": lambda x: x.prod(1, "float32"),
            "max": lambda x: x.max(0, None, True),
            "max initial": lambda x: x.max(1, None, False, 1.25),
            "min": lambda x: x.min(1, None, True),
            "any": lambda x: (x > 0).any(0, None, keepdims=True),  # NumPy's method reads a second positional as a dtype
            "all": lambda x: (x > 0).all(1, None, keepdims=False),
            "var": lambda x: x.var(0, None, None, 1, True),
            "std": lambda x: x.std(1, None, None, 0),
            "ptp": lambda x: x.ptp(0, None, True) if hasattr(x, "ptp") else np.ptp(x, 0, None, True),
            "argmax": lambda x: x.argmax(1, None),
            "take": lambda x: x.take([5, 0], None, None, "wrap"),
        }
        for name, call in cases.items():
            with self.subTest(name=name):
                want = np.asarray(call(a))
                got = np.asarray(call(t))
                self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape))
                np.testing.assert_array_equal(got, want)

    def test_out_and_order_refused(self):
        t = numerical.ones((2, 3))
        out = numerical.zeros(3)
        for name, call in {
            "sum": lambda: numerical.sum(t, 0, None, out), "max": lambda: numerical.max(t, 0, out),
            "any": lambda: numerical.any(t, 0, out), "var": lambda: numerical.var(t, 0, None, out),
            "ptp": lambda: numerical.ptp(t, 0, out), "argmax": lambda: numerical.argmax(t, 0, out),
            "take": lambda: numerical.take(t, [0], 0, out), "method": lambda: t.max(0, out),
            "like": lambda: numerical.full((2,), 1.0, like=np.zeros(2)),
        }.items():
            with self.subTest(name=name), self.assertRaises(TypeError):
                call()
        for call in (lambda: numerical.zeros(2, None, "X"), lambda: numerical.asarray([1.0], None, "Q"),
                     lambda: numerical.ones_like(t, None, "Z")):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
