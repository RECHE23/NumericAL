"""tile, repeat, roll, rot90, the splits, dstack, append, block, take, take_along_axis and the broadcasts against
NumPy's answers, in float64, bool and int64, with NumPy's errors. Deterministic; the NumPy comparisons skipped without
NumPy."""
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


class WithoutNumPy(unittest.TestCase):
    def test_values(self):
        self.assertEqual(numerical.tile([1, 2], 2).tolist(), [1.0, 2.0, 1.0, 2.0])
        self.assertEqual(numerical.tile(numerical.asarray([True]), (2, 1)).dtype, "bool")
        self.assertEqual(numerical.repeat([[1, 2]], 2, axis=1).tolist(), [[1.0, 1.0, 2.0, 2.0]])
        self.assertEqual(numerical.repeat(numerical.asarray([5, 6], "int64"), [0, 3]).tolist(), [6, 6, 6])
        self.assertEqual(numerical.roll([1, 2, 3], -1).tolist(), [2.0, 3.0, 1.0])
        self.assertEqual(numerical.roll([[1, 2], [3, 4]], 1).tolist(), [[4.0, 1.0], [2.0, 3.0]])
        self.assertEqual(numerical.roll([[1, 2], [3, 4]], (1, 1), axis=(1, 1)).tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(numerical.roll(numerical.zeros((0, 2)), 3, axis=0).shape, (0, 2))
        self.assertEqual(numerical.rot90([[1, 2], [3, 4]], -1).tolist(), [[3.0, 1.0], [4.0, 2.0]])
        self.assertEqual(numerical.rot90([[1, 2], [3, 4]], 2).tolist(), [[4.0, 3.0], [2.0, 1.0]])
        self.assertEqual([t.tolist() for t in numerical.array_split([1, 2, 3], 2)], [[1.0, 2.0], [3.0]])
        self.assertEqual([t.tolist() for t in numerical.split([1, 2, 3], [1, 1])], [[1.0], [], [2.0, 3.0]])
        self.assertEqual(numerical.dstack([[1, 2], [3, 4]]).shape, (1, 2, 2))
        self.assertEqual(numerical.append([1], [[2, 3]]).tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(numerical.block([[1, 2], [3, 4]]).tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(numerical.block([[1.0, numerical.asarray([2, 3], "int64")]]).dtype, "float64")
        self.assertEqual(numerical.take([[1, 2], [3, 4]], [[0], [3]]).tolist(), [[1.0], [4.0]])
        self.assertEqual(numerical.take([1, 2, 3], -1), 3.0)
        self.assertEqual(numerical.take([1, 2, 3], [7, -7], mode="clip").tolist(), [3.0, 1.0])
        a = numerical.asarray([[3.0, 1.0, 2.0]])
        self.assertEqual(numerical.take_along_axis(a, [[2, 0]], axis=1).tolist(), [[2.0, 3.0]])
        self.assertEqual(numerical.take_along_axis(a, numerical.argsort(a, axis=None), axis=None).tolist(),
                         [1.0, 2.0, 3.0])
        self.assertEqual(numerical.broadcast_to([1, 2], (2, 2)).tolist(), [[1.0, 2.0], [1.0, 2.0]])
        self.assertEqual(numerical.broadcast_to(numerical.asarray(7, "int64"), (0, 3)).shape, (0, 3))
        self.assertEqual(numerical.broadcast_shapes(), ())
        x, y = numerical.broadcast_arrays([[1], [2]], [1, 2, 3])
        self.assertEqual((x.shape, y.shape), ((2, 3), (2, 3)))

    def test_errors(self):
        cases = [
            (ValueError, lambda: numerical.block([[1.0], 2.0])),
            (ValueError, lambda: numerical.block([[]])),
            (TypeError, lambda: numerical.block(([1.0],))),
            (ValueError, lambda: numerical.split([1, 2, 3], 2)),
            (ValueError, lambda: numerical.array_split([1, 2, 3], 0)),
            (ValueError, lambda: numerical.broadcast_to([1, 2], (3,))),
            (ValueError, lambda: numerical.broadcast_to([1, 2], (-1, 2))),
            (ValueError, lambda: numerical.broadcast_shapes((2,), (3,))),
            (ValueError, lambda: numerical.repeat([1, 2], [1, 2, 3])),
            (ValueError, lambda: numerical.repeat([1, 2], -1)),
            (TypeError, lambda: numerical.repeat([1, 2], [1.5, 2])),
            (TypeError, lambda: numerical.repeat([1, 2], numerical.asarray([1.0, 2.0]))),
            (ValueError, lambda: numerical.rot90([[1]], axes=(0, 0))),
            (ValueError, lambda: numerical.rot90([[1]], axes=(0,))),
            (ValueError, lambda: numerical.rot90([[1]], axes=(0, 2))),
            (ValueError, lambda: numerical.roll([1, 2], (1, 2), axis=(0, 0, 0))),
            (IndexError, lambda: numerical.take([1, 2], [5])),
            (IndexError, lambda: numerical.take(numerical.zeros(0), [1], mode="wrap")),
            (ValueError, lambda: numerical.take([1, 2], [0], mode="bounce")),
            (IndexError, lambda: numerical.take_along_axis([1.0, 2.0], [0.5], axis=0)),
            (ValueError, lambda: numerical.take_along_axis([[1.0]], [0], axis=0)),
            (ValueError, lambda: numerical.take_along_axis([[1.0]], [[0]], axis=None)),
            (ValueError, lambda: numerical.hsplit(1.0, 1)),
            (ValueError, lambda: numerical.vsplit([1.0], 1)),
            (ValueError, lambda: numerical.dsplit([[1.0]], 1)),
        ]
        for error, call in cases:
            with self.assertRaises(error):
                call()


@unittest.skipIf(np is None, "NumPy is not installed")
class AgainstNumPy(unittest.TestCase):
    def assertSame(self, got, want, label):
        if isinstance(want, (list, tuple)):
            self.assertEqual(len(got), len(want), label)
            for x, y in zip(got, want):
                self.assertSame(x, y, label)
            return
        got = np.asarray(got)
        self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype), label)
        self.assertTrue(np.array_equal(got, want, equal_nan=True), label)

    def test_against_numpy(self):
        rng = np.random.default_rng(0)
        for x in (rng.normal(size=(3, 4, 5)), rng.random((3, 4)) > 0.5, np.arange(24).reshape(2, 3, 4)):
            t = numerical.asarray(x, "int64") if x.dtype.kind == "i" else numerical.asarray(x)
            last = x.shape[-1]
            for reps in (2, (2, 1), (1, 2, 3), (2, 1, 1, 2), (0, 2), ()):
                self.assertSame(numerical.tile(t, reps), np.tile(x, reps), f"tile {reps}")
            for counts, axis in ((2, None), (3, 0), (0, 1), ([1, 0, 2], 0 if x.shape[0] == 3 else None),
                                 (np.arange(last) % 3, -1)):
                if axis is None and not isinstance(counts, int):
                    continue
                self.assertSame(numerical.repeat(t, counts, axis=axis), np.repeat(x, counts, axis=axis),
                                f"repeat {counts} {axis}")
            for shift, axis in ((1, None), (-3, None), (2, 0), (7, 1), ((1, 2), (0, 1)), (1, (0, 1)), ((1, -2), 0)):
                self.assertSame(numerical.roll(t, shift, axis=axis), np.roll(x, shift, axis=axis), f"roll {shift}")
            for k in range(-1, 6):
                for axes in ((0, 1), (1, 0), (-1, 0)):
                    self.assertSame(numerical.rot90(t, k, axes), np.rot90(x, k, axes), f"rot90 {k} {axes}")
            for sections, axis in ((3, 0), ([1, 2], 1), ([2, -1], 1), ([0, 5, 1], 0), (np.array([1, 3]), 1), (1, -1)):
                self.assertSame(numerical.array_split(t, sections, axis), np.array_split(x, sections, axis),
                                f"array_split {sections} {axis}")
            for indices, axis, mode in (([0, 1], 0, "raise"), ([[0, 1], [2, 2]], 1, "raise"), (1, 0, "raise"),
                                        ([5, -7], 1, "wrap"), ([9, -9], 0, "clip"), ([3, 4], None, "raise")):
                self.assertSame(numerical.take(t, indices, axis=axis, mode=mode),
                                np.take(x, indices, axis=axis, mode=mode), f"take {indices} {axis} {mode}")
            for axis in (0, 1, -1):
                order = np.argsort(x, axis=axis, kind="stable")
                self.assertSame(numerical.take_along_axis(t, numerical.asarray(order, "int64"), axis),
                                np.take_along_axis(x, order, axis), f"take_along_axis {axis}")
            self.assertSame(numerical.append(t, t, axis=0), np.append(x, x, axis=0), "append")
            self.assertSame(numerical.broadcast_to(t[0], (2,) + x[0].shape), np.broadcast_to(x[0], (2,) + x[0].shape),
                            "broadcast_to")
        a = rng.normal(size=(2, 3, 4))
        self.assertSame(numerical.dsplit(numerical.asarray(a), [2]), np.dsplit(a, [2]), "dsplit")
        self.assertSame(numerical.hsplit(numerical.asarray(a), 3), np.hsplit(a, 3), "hsplit")
        self.assertSame(numerical.vsplit(numerical.asarray(a), 2), np.vsplit(a, 2), "vsplit")
        self.assertSame(numerical.split(numerical.asarray(a), [1], 0), np.split(a, [1], 0), "split")
        self.assertSame(numerical.dstack([a[0], a[1]]), np.dstack([a[0], a[1]]), "dstack")
        nested = [[[np.ones((2, 2))], [np.zeros((1, 2))]]]
        self.assertSame(numerical.block(nested), np.block(nested), "block of rank 3")
        layout = [[np.eye(2), np.zeros((2, 1))], [np.ones((1, 3))]]
        self.assertSame(numerical.block(layout), np.block(layout), "block")
        self.assertSame(numerical.block(np.ones(3)), np.block(np.ones(3)), "block of one array")
        self.assertEqual(numerical.broadcast_shapes((3, 1), (4,), 1), np.broadcast_shapes((3, 1), (4,), 1))


if __name__ == "__main__":
    unittest.main()
