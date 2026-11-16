"""Arguments no caller should pass, answered at once as NumPy and SciPy answer them: an order past any length, a
transform or a matrix past memory, a bound past int64, positions into nothing. Each of these held on for minutes, or
took gigabytes, before it was refused; a test that ends at all is the check."""

import unittest

import numerical
import numerical.fft as nfft

try:
    import numpy as np
except ImportError:  # pragma: no cover - the CI images install NumPy
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class AbsurdArguments(unittest.TestCase):
    def assert_same(self, got, want):
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.dtype, str(want.dtype))
        self.assertEqual(got.tolist(), want.tolist())

    def test_clip_bounds_past_int64_bound_nothing_on_their_harmless_side(self):
        a = np.array([1, 5])
        t = numerical.asarray([1, 5], dtype="int64")
        for low, high in ((-2 ** 70, None), (3, 2 ** 70), (-2 ** 70, 2 ** 70), (3, 2 ** 64), (-2 ** 64, 3),
                          (3, 2 ** 63), (-2 ** 63 - 1, 3)):
            self.assert_same(numerical.clip(t, low, high), np.clip(a, low, high))
        for low, high in ((None, -2 ** 70), (2 ** 70, None), (2 ** 70, 3), (2 ** 70, -2 ** 70)):
            with self.assertRaises(OverflowError):
                np.clip(a, low, high)
            with self.assertRaises(OverflowError):
                numerical.clip(t, low, high)

    def test_partition_of_nothing_checks_no_position_and_an_empty_list_is_no_integer(self):
        for shape, kth, axis in (((0,), 0, -1), ((0,), -1, -1), ((0,), 5, -1), ((0, 3), 3, 1)):
            self.assert_same(numerical.partition(numerical.zeros(shape), kth, axis=axis),
                             np.partition(np.zeros(shape), kth, axis=axis))
            self.assert_same(numerical.argpartition(numerical.zeros(shape), kth, axis=axis),
                             np.argpartition(np.zeros(shape), kth, axis=axis))
        for kth in ([], ()):
            with self.assertRaisesRegex(TypeError, "Partition index must be integer"):
                np.partition(np.array([3.0, 1.0, 2.0]), kth)
            with self.assertRaisesRegex(TypeError, "Partition index must be integer"):
                numerical.partition(numerical.asarray([3.0, 1.0, 2.0]), kth)
            with self.assertRaisesRegex(TypeError, "Partition index must be integer"):
                numerical.argpartition(numerical.asarray([3.0, 1.0, 2.0]), kth)
        for empty in (np.array([], dtype=np.int64), numerical.asarray([], dtype="int64")):
            self.assert_same(numerical.partition(numerical.asarray([3.0, 1.0]), empty),
                             np.partition(np.array([3.0, 1.0]), np.array([], dtype=np.int64)))

    def test_diff_of_bools_is_numpys_not_equal_and_an_order_past_the_axis_ends(self):
        x = [[True, False, True, True], [False, False, True, False]]
        for n in range(1, 6):
            for axis in (0, 1):
                self.assert_same(numerical.diff(x, n, axis=axis), np.diff(x, n, axis=axis))
                self.assert_same(numerical.diff(numerical.asarray(x), n, axis=axis), np.diff(x, n, axis=axis))
        self.assert_same(numerical.diff([True, False], prepend=True, append=False),
                         np.diff([True, False], prepend=True, append=False))
        self.assert_same(numerical.diff(numerical.asarray([1.0, 2.0, 4.0]), 10 ** 12),
                         np.diff(np.array([1.0, 2.0, 4.0]), 3))

    def test_polyder_of_an_order_past_the_coefficients_ends(self):
        p = numerical.asarray([1, 2, 3], dtype="int64")
        for m in (3, 4, 10 ** 12):
            self.assert_same(numerical.polyder(p, m), np.polyder(np.array([1, 2, 3]), 5))

    def test_pascal_matrices_refuse_a_negative_order_and_one_past_memory(self):
        for function in (numerical.pascal, numerical.invpascal, numerical.invhilbert):
            with self.assertRaises(ValueError):
                function(-1)
            for n in (2 ** 33, 2 ** 64):
                with self.assertRaises((ValueError, OverflowError, MemoryError)):
                    function(n)
            self.assertEqual(function(0).shape, (0, 0))

    def test_repeat_refuses_a_total_past_memory_before_building_it(self):
        for counts in ((2 ** 62, 2 ** 62), 2 ** 62, [2 ** 61]):
            with self.assertRaises(ValueError):
                numerical.repeat(numerical.asarray([1.0, 2.0]), counts)
        for counts in ((2 ** 63, 1), 2 ** 63):
            with self.assertRaises(OverflowError):
                np.repeat(np.array([1.0, 2.0]), counts)
            with self.assertRaises(OverflowError):
                numerical.repeat(numerical.asarray([1.0, 2.0]), counts)
        rng = np.random.default_rng(5)
        for shape, axis in (((5,), 0), ((3, 4), 0), ((3, 4), 1), ((2, 3, 4), 1), ((0, 3), 1), ((3, 0), 0)):
            x = rng.standard_normal(shape)
            for _ in range(4):
                counts = rng.integers(0, 4, shape[axis]).tolist()
                self.assert_same(numerical.repeat(numerical.asarray(x), counts, axis=axis),
                                 np.repeat(x, counts, axis=axis))

    def test_frequencies_past_memory_are_refused_and_the_others_numpys_bits(self):
        for n in (10 ** 30, 2 ** 63 - 1):
            for function in (nfft.fftfreq, nfft.rfftfreq):
                with self.assertRaises(ValueError):
                    getattr(np.fft, function.__name__)(n)
                with self.assertRaises(ValueError):
                    function(n)
        for function in (nfft.fftfreq, nfft.rfftfreq):
            with self.assertRaisesRegex(ValueError, "n should be an integer"):
                function(2.5)
            for n in (1, 2, 7, 1000, np.int8(3)):
                for d in (1.0, 0.1, 3, np.float32(0.3)):
                    self.assertEqual(np.asarray(function(n, d)).tobytes(),
                                     getattr(np.fft, function.__name__)(n, d).tobytes())
        self.assertEqual(nfft.rfftfreq(-3).shape, (0,))
        with self.assertRaisesRegex(ValueError, "negative dimensions"):
            nfft.fftfreq(-3)

    def test_hadamard_past_memory_is_refused_before_its_doublings(self):
        for n in (2 ** 64, 2 ** 32):
            with self.assertRaisesRegex(ValueError, "array is too big"):
                numerical.hadamard(n)

    def test_a_transform_past_memory_is_refused_before_its_plan(self):
        x = numerical.asarray([[1.0, 2.0], [3.0, 5.0]])
        for name in ("fft", "ifft", "rfft", "irfft", "hfft", "ihfft", "dct", "idst"):
            for n in (2 ** 63 - 1, 2 ** 62):
                with self.assertRaisesRegex(ValueError, "is too big"):
                    getattr(nfft, name)(x, n=n)
                with self.assertRaisesRegex(ValueError, "is too big"):
                    getattr(nfft, name)(np.asarray(x), n=n)
        self.assertEqual(nfft.rfft(numerical.zeros((0, 3)), n=2 ** 40).shape, (0, 2 ** 39 + 1))


if __name__ == "__main__":
    unittest.main()
