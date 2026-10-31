"""A Tensor's advanced indexing against NumPy's own: integer lists and arrays, bool masks (lists, NumPy arrays, bool
Tensors, of any rank, a lone bool), mixed with integers, slices, ``...`` and ``None``, over shapes with empty axes and
every dtype; where the result's axes go (together or at the front); NumPy's errors as IndexError; the refusals of the
two natives behind it. Deterministic; skipped when NumPy is not importable (CI installs it)."""
import random
import sys
import unittest
import warnings

import numerical

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy not installed")
class AdvancedIndexing(unittest.TestCase):
    def tensor(self, a):
        return numerical.asarray(a, "bool") if a.dtype == bool else numerical.asarray(a)

    def same(self, t, a, key):
        """t[key] is a[key] (a bool Tensor in the key read by NumPy as its array), or both raise IndexError. NumPy before
        2.3 read an index out of bounds into an empty result with a DeprecationWarning saying it would raise, as it
        has since: the warning counts as its refusal."""
        numpy_key = tuple(np.asarray(k) if isinstance(k, numerical.Tensor) else k for k in key) \
            if isinstance(key, tuple) else np.asarray(key) if isinstance(key, numerical.Tensor) else key
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                expected = a[numpy_key]
        except (IndexError, DeprecationWarning):
            with self.assertRaises(IndexError):
                t[key]
            return
        got = np.asarray(t[key])
        self.assertEqual(got.shape, expected.shape)
        if expected.ndim:  # a number is a Python one, as basic indexing gives
            self.assertEqual(got.dtype, expected.dtype)
        np.testing.assert_array_equal(got, expected)

    def random_index(self, rng, extent):
        choice = rng.randrange(10)
        if choice == 0:
            return slice(rng.randrange(-extent - 1, extent + 2), None, rng.choice([1, 2, -1]))
        if choice == 1:
            return rng.randrange(-extent, extent) if extent else 0
        if choice == 2:
            return [rng.randrange(-extent - 1, extent + 1) for _ in range(rng.randrange(4))] if extent else []
        if choice == 3:
            return np.array([[rng.randrange(-extent, extent) for _ in range(2)]] * 2) if extent else np.zeros(0, int)
        if choice == 4:
            return None
        if choice == 5:
            return rng.random() < 0.5
        mask = [rng.random() < 0.5 for _ in range(extent)]
        return (mask, np.array(mask, bool), numerical.asarray(mask, "bool"), slice(None))[choice - 6]

    def test_random_keys_as_numpy(self):
        rng = random.Random(1031)
        for _ in range(4000):
            shape = tuple(rng.randrange(5) for _ in range(rng.randrange(5)))
            a = np.arange(1.0, 1.0 + np.prod(shape)).reshape(shape)
            a = (a + 1j * a, a % 3 == 0, a.astype(np.float32), a)[rng.randrange(4)]
            key, axis = [], 0
            for _ in range(rng.randrange(1, 4)):
                if rng.random() < 0.1:
                    key.append(Ellipsis)
                    continue
                key.append(self.random_index(rng, shape[axis] if axis < len(shape) else 1))
                axis += 1
            if rng.random() < 0.15 and len(shape) >= 2:  # a mask over two axes
                key[:2] = [np.array([[rng.random() < 0.5 for _ in range(shape[1])] for _ in range(shape[0])],
                                    bool).reshape(shape[:2])]
            with self.subTest(shape=shape, key=key):
                self.same(self.tensor(a), a, tuple(key))

    def test_where_the_advanced_axes_go(self):
        a = np.arange(120.0).reshape(2, 3, 4, 5)
        t = numerical.asarray(a)
        for key in ((slice(None), [0, 2], [1, 3]),                # together: in their place
                    ([0, 1], slice(None), [1, 3]),                # apart: at the front
                    (slice(None), [[0], [2]], [1, 3], slice(1, 3)),  # broadcast to 2 × 2
                    (0, slice(None), [1, 2]),                     # an integer beside an array is advanced too
                    (slice(None), 0, [1, 2]),
                    ([1], None, [2]),                             # a new axis between them parts them
                    (Ellipsis, [4, 0]),
                    (a[:, :, 0, 0] > 30,),                        # a mask over the first two axes
                    (slice(None), a[0, :, :, 0] > 3),
                    (True, [0, 1]), (False, Ellipsis), (np.True_,)):
            with self.subTest(key=key):
                self.same(t, a, key)

    def test_masks_and_lists_of_every_kind(self):
        a = np.arange(12.0).reshape(3, 4)
        t = numerical.asarray(a)
        mask = a > 4.5
        for key in (mask, numerical.asarray(mask, "bool"), mask.tolist(), [True, False, True], np.array([2, -1]),
                    np.array([0, 2], np.uint8), [], [[0, 1], [2, 2]], ([0, 2], [3, -4]), (a[:, 1] > 2, 1)):
            with self.subTest(key=key):
                self.same(t, a, key)
        self.same(numerical.asarray(2.5), np.asarray(2.5), True)  # a rank-0 tensor takes a lone bool
        self.same(numerical.asarray(2.5), np.asarray(2.5), (False, None))

    def test_numpy_checks_bounds_only_for_a_result_with_elements(self):
        a = np.zeros((2, 3))
        t = numerical.asarray(a)
        self.same(t, a, ([], [5]))            # an empty broadcast: 5 never read
        self.same(t, a, ([[5]], np.zeros((0, 1), int)))
        self.same(t, a, (slice(0, 0), [5]))   # the slice is empty, the array not: checked
        self.same(numerical.asarray(np.zeros((4, 2))), np.zeros((4, 2)), (np.zeros(0, bool),))  # a mask of no element
        with self.assertRaisesRegex(IndexError, "index 5 is out of bounds for axis 1 with size 3"):
            t[:, [5]]
        with self.assertRaisesRegex(IndexError, "index -4 is out of bounds for axis 1 with size 3"):
            t[:, np.array([0, -4])]
        with self.assertRaisesRegex(IndexError, "index 9 is out of bounds for axis 1 with size 3"):
            t[:, np.array([9], np.uint64)]
        with self.assertRaisesRegex(IndexError, "index 7 is out of bounds for axis 1 with size 3"):
            t[[], 7]                          # an integer: always checked

    def test_the_errors_are_numpys(self):
        t = numerical.asarray(np.zeros((3, 4)))
        for key, words in ((numerical.ones(2), "integer \\(or boolean\\) type, not float64"),
                           ([0.5], "integer \\(or boolean\\) type"), (np.array([0.5]), "not float64"),
                           ([[0, 1], [2]], "rectangular"), ([0, None], "rectangular"),
                           (([0, 1], [0, 1, 2]), "could not be broadcast together with shapes \\(2,\\) \\(3,\\)"),
                           ([True, False], "size of axis is 3 but size of corresponding boolean axis is 2"),
                           (np.ones((3, 3), bool), "along axis 1; size of axis is 4"),
                           ((Ellipsis, [0], Ellipsis), "single ellipsis"),
                           ((0, 0, [0]), "too many indices for a tensor of rank 2: 3")):
            with self.subTest(key=key), self.assertRaisesRegex(IndexError, words):
                t[key]

    def test_large_takes_as_numpy(self):
        rng = np.random.default_rng(1032)
        a = rng.standard_normal((300, 400))
        t = numerical.asarray(a)
        rows, columns = rng.integers(-300, 300, 5000), rng.integers(0, 400, 5000)
        for key in ((rows,), (slice(None), columns[:700] % 400), (rows, columns), (a > 0,), (a[:, 0] > 0, slice(None, None, -3)),
                    (rows.reshape(50, 100), slice(2, 9))):
            with self.subTest(key=[getattr(k, "shape", k) for k in key]):
                self.same(t, a, key)
        flat = rng.standard_normal(10**6)
        index = rng.integers(0, 10**6, 10**6)
        np.testing.assert_array_equal(np.asarray(numerical.asarray(flat)[index]), flat[index])


class Natives(unittest.TestCase):
    """The two natives refuse what the Python side never hands them, rather than read outside the tensor."""

    def test_gather_refusals(self):
        gather, t = numerical._numerical._gather, numerical.ones((2, 3))
        offsets = (b"\x00" * 8, (0, 1, 3))
        self.assertEqual(numerical.asarray(gather(t, list(offsets), [1, 3])).shape, (1, 3))
        for groups, shape, error in (([], [0], numerical.error), ([b"\x00" * 7], [0], numerical.error),
                                     ([(0, 1)], [0], numerical.error), ([(0, 1, -1)], [0], numerical.error),
                                     ([(0, 1, 2)], [3], numerical.error), ([(0, 1, 7)], [7], IndexError),
                                     ([(-1, 1, 2)], [2], IndexError), ([(4, 1, 1), (0, 1, 3)], [1, 3], IndexError),
                                     ([1.5], [1], TypeError), (5, [1], TypeError)):
            with self.subTest(groups=groups), self.assertRaises(error):
                gather(t, groups, shape)
        with self.assertRaises(numerical.error):
            gather(5, [(0, 1, 1)], [1])
        with self.assertRaises(numerical.error):
            gather(t, [(0, 1, 1)], [-1])
        self.assertEqual(numerical.asarray(gather(numerical.ones((0, 3)), [(0, 1, 0), (0, 1, 3)], [0, 3])).shape, (0, 3))

    def test_true_offsets_refusals(self):
        offsets = numerical._numerical._true_offsets
        mask = numerical.asarray([[True, False], [False, True]], "bool")
        self.assertEqual(offsets(mask, [10, 1]), (0).to_bytes(8, sys.byteorder) + (11).to_bytes(8, sys.byteorder))
        for args, error in (((numerical.ones(2), [1]), numerical.error), ((mask, [1]), numerical.error),
                            ((mask, [1, "a"]), TypeError), ((5, [1]), numerical.error), ((mask, 5), TypeError)):
            with self.subTest(args=args), self.assertRaises(error):
                offsets(*args)


if __name__ == "__main__":
    unittest.main()
