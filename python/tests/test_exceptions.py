"""NumPy's exception classes for the same mistakes: an axis out of range is an AxisError (a ValueError and an
IndexError, and numpy.exceptions.AxisError where NumPy is loaded) wherever an axis is read, natively or in Python; a
zero step of arange a ZeroDivisionError; a rank past 64 NumPy's ValueError, from nested lists however deep (or holding
themselves) and from any shape, where the recursions over the axes would overflow the stack. Deterministic; the
comparisons skipped without NumPy."""
import pickle
import threading
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


class WithoutNumPy(unittest.TestCase):
    def test_axis_error(self):
        t = numerical.ones((2, 3))
        for call in (lambda: numerical.sum(t, axis=2), lambda: numerical.sort(t, axis=-3),
                     lambda: numerical.concatenate([t, t], axis=2), lambda: numerical.fft.fft(t, axis=2),
                     lambda: numerical.cumsum(t, axis=2), lambda: numerical.expand_dims(t, 3)):
            with self.assertRaises(numerical.AxisError) as caught:
                call()
            for kind in (ValueError, IndexError, numerical.error):
                self.assertIsInstance(caught.exception, kind)
            self.assertIn("out of bounds", str(caught.exception))
        error = numerical.AxisError("axis 3 is out of bounds")
        self.assertEqual(str(pickle.loads(pickle.dumps(error))), "axis 3 is out of bounds")

    def test_rank_past_64(self):
        def nested(depth):
            value = [1.0]
            for _ in range(depth - 1):
                value = [value]
            return value

        self.assertEqual(numerical.asarray(nested(64)).ndim, 64)
        self.assertEqual(numerical.zeros((1,) * 64).ndim, 64)
        itself = []
        itself.append(itself)
        outcomes = []

        def deep():  # on a small stack, which a recursion per level overflowed at some thousands of levels
            for call in (lambda: numerical.asarray(nested(100_000)), lambda: numerical.asarray(nested(65)),
                         lambda: numerical.asarray(nested(65), "int64"), lambda: numerical.asarray(itself),
                         lambda: numerical.zeros((1,) * 65), lambda: numerical.reshape(numerical.ones(1), (1,) * 4000),
                         lambda: numerical.expand_dims(numerical.zeros((1,) * 64), 0)):
                try:
                    call()
                    outcomes.append("answered")
                except ValueError as refused:
                    outcomes.append("64" in str(refused))

        size = threading.stack_size(256 * 1024)
        try:
            thread = threading.Thread(target=deep)
            thread.start()
            thread.join()
        finally:
            threading.stack_size(size)
        self.assertEqual(outcomes, [True] * 7)
        with self.assertRaises(numerical.error):
            numerical.Tensor([1.0], shape=(1,) * 65)
        self.assertEqual(numerical.__array_namespace_info__().capabilities()["max dimensions"], 64)

    def test_numpy_positions_refused_loudly(self):
        # NumPy's third argument of the reductions is dtype, here keepdims (named dtype here); its fourth of var and
        # std is out, here ddof: a call in NumPy's order raises, where it kept the axes and the dtype silently
        t = numerical.ones((2, 3))
        for call in (lambda: numerical.sum(t, 0, "float32"), lambda: numerical.mean(t, 1, float),
                     lambda: numerical.prod(t, 0, "float32"), lambda: numerical.max(t, 0, numerical.float32),
                     lambda: numerical.var(t, 0, None, None, 1), lambda: numerical.std(t, 0, None, None)):
            with self.assertRaises(TypeError):
                call()
        self.assertEqual(numerical.sum(t, 0, True).shape, (1, 3))
        self.assertEqual(numerical.sum(t, 0, 0).shape, (3,))
        self.assertEqual(numerical.sum(t, 0, dtype="float32").dtype, "float32")

    def test_zero_step(self):
        for kind in (ZeroDivisionError, ValueError, numerical.error):
            with self.assertRaises(kind):
                numerical.arange(0, 1, 0)


@unittest.skipIf(np is None, "NumPy not installed")
class AgainstNumPy(unittest.TestCase):
    def test_numpy_except_clauses_catch_ours(self):
        # each mistake made on both sides: NumericAL's error is NumPy's class or one of its subclasses
        a = np.arange(6.0).reshape(2, 3)
        t = numerical.asarray(a)
        calls = {
            "sum": lambda m, x: m.sum(x, axis=2), "mean": lambda m, x: m.mean(x, axis=-3),
            "max": lambda m, x: m.max(x, axis=2), "sort": lambda m, x: m.sort(x, axis=2),
            "argsort": lambda m, x: m.argsort(x, axis=2), "median": lambda m, x: m.median(x, axis=2),
            "moveaxis": lambda m, x: m.moveaxis(x, 2, 0), "expand_dims": lambda m, x: m.expand_dims(x, 3),
            "unique": lambda m, x: m.unique(x, axis=2), "concatenate": lambda m, x: m.concatenate([x, x], axis=2),
            "stack": lambda m, x: m.stack([x, x], axis=4), "append": lambda m, x: m.append(x, x, axis=2),
            "cumsum": lambda m, x: m.cumsum(x, axis=2), "diff": lambda m, x: m.diff(x, axis=2),
            "take": lambda m, x: m.take(x, [0], axis=2), "flip": lambda m, x: m.flip(x, axis=2),
            "roll": lambda m, x: m.roll(x, 1, axis=2), "squeeze": lambda m, x: m.squeeze(x, axis=2),
            "argmax": lambda m, x: m.argmax(x, axis=2), "quantile": lambda m, x: m.quantile(x, 0.5, axis=2),
            "swapaxes": lambda m, x: m.swapaxes(x, 0, 2), "split": lambda m, x: m.split(x, 1, axis=2),
            "count_nonzero": lambda m, x: m.count_nonzero(x, axis=2), "fft": lambda m, x: m.fft.fft(x, axis=2),
            "norm": lambda m, x: m.linalg.norm(x, axis=2), "delete": lambda m, x: m.delete(x, 0, axis=2),
            "arange": lambda m, x: m.arange(0, 1, 0),
        }
        for name, call in calls.items():
            with self.subTest(name=name):
                with self.assertRaises(Exception) as numpy_side:
                    call(np, a)
                with self.assertRaises(type(numpy_side.exception)):
                    call(numerical, t)
        with self.assertRaises(np.exceptions.AxisError):
            numerical.sum(t, axis=2)


if __name__ == "__main__":
    unittest.main()
