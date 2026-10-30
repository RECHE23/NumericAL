"""The ndarray methods a Tensor has, against NumPy's: pickling and copying, the axis moves (transpose, swapaxes,
mT, squeeze, ravel), real and imag, diagonal and trace along any two axes, dot, item, membership and the sizes, over
every dtype. Deterministic; skipped when NumPy is not importable (CI installs it)."""
import copy
import itertools
import pickle
import sys
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None

DTYPES = ("float64", "complex128", "float32", "complex64", "bool")


@unittest.skipIf(np is None, "NumPy not installed")
class Methods(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261103)

    def array(self, shape, dtype="float64"):
        x = self.rng.standard_normal(shape)
        if dtype == "bool":
            return x > 0
        if dtype.startswith("complex"):
            x = x + 1j * self.rng.standard_normal(shape)
        return x.astype(dtype)

    def same(self, got, expected):
        expected = np.asarray(expected)
        self.assertIsInstance(got, numerical.Tensor)
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, str(expected.dtype))
        self.assertTrue(np.array_equal(np.asarray(got), expected, equal_nan=expected.dtype != bool))

    def test_a_pickle_and_a_copy_hold_every_bit(self):
        special = np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 5e-324, 1.5])
        for dtype, shape in itertools.product(DTYPES, [(), (0,), (7,), (2, 3, 4)]):
            x = self.array(shape, dtype)
            if shape == (7,) and dtype != "bool":
                x = (special + (1j * special[::-1] if dtype.startswith("complex") else 0)).astype(dtype)
            t = numerical.asarray(x, dtype)
            with self.subTest(dtype=dtype, shape=shape):
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    back = pickle.loads(pickle.dumps(t, protocol))
                    self.same(back, x)
                    self.assertEqual(back.tobytes(), x.tobytes())  # the same bits, NaN payloads and -0 included
                self.assertIs(copy.copy(t), t)  # never written into: itself is its copy
                self.assertIs(copy.deepcopy([t])[0], t)
                self.assertIs(t.copy(), t)

    def test_a_pickle_written_in_the_other_byte_order_reads_back(self):
        for dtype in DTYPES:
            x = self.array((3, 2), dtype)
            other = "big" if sys.byteorder == "little" else "little"
            swapped = x.astype(x.dtype.newbyteorder(">" if other == "big" else "<")).tobytes()
            with self.subTest(dtype=dtype):
                self.same(numerical._rebuilt(swapped, (3, 2), dtype, other), x)
        with self.assertRaises(ValueError):
            numerical._rebuilt(b"\0" * 7, (1,), "float64", sys.byteorder)
        self.assertEqual(numerical._rebuilt(b"\x02\x00", (2,), "bool", sys.byteorder).tolist(), [True, False])

    def test_the_axes_move_as_numpy(self):
        for dtype in DTYPES:
            x = self.array((2, 3, 4), dtype)
            t = numerical.asarray(x, dtype)
            with self.subTest(dtype=dtype):
                self.same(t.transpose(), x.transpose())
                self.same(t.transpose(1, 0, 2), x.transpose(1, 0, 2))
                self.same(t.transpose((2, 0, 1)), x.transpose((2, 0, 1)))
                self.same(t.transpose([-1, 0, 1]), x.transpose([-1, 0, 1]))
                self.same(numerical.transpose(x, (0, 2, 1)) if dtype != "bool" else t.transpose(0, 2, 1),
                          np.transpose(x, (0, 2, 1)))
                self.same(t.swapaxes(0, 2), x.swapaxes(0, 2))
                self.same(t.swapaxes(-1, 1), x.swapaxes(-1, 1))
                self.assertIs(t.swapaxes(1, 1), t)
                self.same(t.mT, x.mT)
                self.same(t.ravel(), x.ravel())
                self.same(t.flatten(), x.flatten())
        with self.assertRaises(ValueError):
            numerical.ones((2, 3)).transpose(0, 0)
        with self.assertRaises(ValueError):
            numerical.ones((2, 3)).transpose(0)

    def test_squeeze_as_numpy(self):
        x = self.array((1, 3, 1, 2))
        t = numerical.asarray(x)
        for axis in (None, 0, 2, -2, (0, 2)):
            with self.subTest(axis=axis):
                self.same(t.squeeze(axis), x.squeeze(axis))
                self.same(numerical.squeeze(x, axis), np.squeeze(x, axis))
        v = numerical.ones(3)
        self.assertIs(v.squeeze(), v)  # nothing to drop: the tensor itself
        with self.assertRaises(ValueError):
            t.squeeze(1)

    def test_real_and_imag_as_numpy(self):
        for dtype in DTYPES:
            x = self.array((2, 3), dtype)
            t = numerical.asarray(x, dtype)
            with self.subTest(dtype=dtype):
                self.same(t.real, x.real)
                self.same(t.imag, x.imag)
                self.same(numerical.real(x) if dtype != "bool" else t.real, np.real(x))
                self.same(numerical.imag(t), np.imag(x))
                if dtype != "bool":
                    self.same(t.conj(), x.conj())
                    self.same(t.conjugate(), x.conjugate())

    def test_diagonal_and_trace_along_any_two_axes(self):
        for dtype in ("float64", "complex128", "float32"):
            x = self.array((3, 4, 5), dtype)
            t = numerical.asarray(x)
            for (axis1, axis2), offset in itertools.product([(0, 1), (1, 0), (0, 2), (2, 1), (-1, 0)], (-2, 0, 1, 7)):
                with self.subTest(dtype=dtype, axes=(axis1, axis2), offset=offset):
                    self.same(t.diagonal(offset, axis1, axis2), x.diagonal(offset, axis1, axis2))
                    expected = x.trace(offset, axis1, axis2)
                    got = t.trace(offset, axis1, axis2)
                    tolerance = 1e-5 if dtype == "float32" else 1e-12
                    self.assertTrue(np.allclose(np.asarray(got), expected, rtol=tolerance, atol=tolerance))
        m = numerical.asarray([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(m.trace(), 5.0)
        self.assertEqual(m.trace(dtype=np.complex128), 5 + 0j)
        with self.assertRaises(ValueError):
            m.diagonal(0, 1, 1)

    def test_dot_as_numpy(self):
        for sa, sb in [((3,), (3,)), ((2, 3), (3,)), ((3,), (3, 4)), ((2, 3), (3, 4)), ((2, 2, 3), (3,)),
                       ((2, 2, 3), (4, 3, 5)), ((), (3,)), ((2, 3), ())]:
            x, y = self.array(sa), self.array(sb)
            with self.subTest(sa=sa, sb=sb):
                got = numerical.asarray(x).dot(y)
                expected = x.dot(y)
                self.assertTrue(np.allclose(np.asarray(got), expected, rtol=1e-12, atol=1e-12))
                self.assertEqual(np.shape(got), np.shape(expected))

    def test_item_membership_and_sizes(self):
        for dtype in DTYPES:
            x = self.array((2, 3), dtype)
            t = numerical.asarray(x, dtype)
            with self.subTest(dtype=dtype):
                self.assertEqual(t.item(4), x.item(4))
                self.assertEqual(t.item(-1), x.item(-1))
                self.assertEqual(t.item(1, 2), x.item(1, 2))
                self.assertEqual(t.item((0, 1)), x.item((0, 1)))
                self.assertEqual(numerical.asarray(x[:1, :1], dtype).item(), x[:1, :1].item())
                self.assertEqual(t.itemsize, x.itemsize)
                self.assertEqual(t.nbytes, x.nbytes)
                self.assertIn(x.item(3), t)
        with self.assertRaises(ValueError):
            numerical.ones(2).item()
        with self.assertRaises(IndexError):
            numerical.ones(2).item(2)
        t = numerical.asarray([1.0, float("nan"), 3.0])
        self.assertIn(3, t)
        self.assertNotIn(2.0, t)
        self.assertNotIn(float("nan"), t)  # a NaN equals nothing, as NumPy's `in`
        self.assertEqual(t.device, "cpu")
        self.assertIs(t.to_device("cpu"), t)
        with self.assertRaises(ValueError):
            t.to_device("cuda")


if __name__ == "__main__":
    unittest.main()
