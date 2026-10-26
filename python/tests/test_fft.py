"""numerical.fft against numpy.fft: the same names, arguments, shapes, dtypes and errors, the values within a few units
in the last place of NumPy's (both are within a few of the exact transform; their bits differ), the frequencies and
shifts to the bit."""

import unittest

import numerical
import numerical.fft as nfft

try:
    import numpy as np
except ImportError:  # pragma: no cover - the CI images install NumPy
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class FourierTransforms(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)

    def operand(self, shape, dtype):
        values = self.rng.standard_normal(shape)
        if np.dtype(dtype).kind == "c":
            values = values + 1j * self.rng.standard_normal(shape)
        return values.astype(dtype)

    def assert_close(self, got, want, single=False):
        got = np.asarray(got)
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.dtype, want.dtype)
        scale = max(1.0, float(np.max(np.abs(want)))) if want.size else 1.0
        np.testing.assert_allclose(got, want, rtol=0, atol=(2e-5 if single else 1e-12) * scale)

    def test_fft_and_ifft_match_numpy_on_every_kind_of_length(self):
        # Powers of two, mixed radices (2, 3, 5, 7), a large prime (Bluestein's), in the four dtypes.
        for n in (1, 2, 8, 16, 64, 1024, 6, 12, 100, 360, 1000, 2187, 4800, 7, 67, 1009, 2018):
            for dtype in (np.float64, np.complex128, np.float32, np.complex64):
                x = self.operand(n, dtype)
                single = np.dtype(dtype) in (np.float32, np.complex64)
                self.assert_close(nfft.fft(x), np.fft.fft(x), single)
                self.assert_close(nfft.ifft(x), np.fft.ifft(x), single)

    def test_points_axes_and_norms_are_numpys(self):
        x = self.operand((4, 6, 5), np.complex128)
        for axis in (0, 1, 2, -1, -3):
            for n in (None, 3, 8):
                for norm in (None, "backward", "ortho", "forward"):
                    self.assert_close(nfft.fft(x, n, axis, norm), np.fft.fft(x, n, axis, norm))
                    self.assert_close(nfft.ifft(x, n, axis, norm), np.fft.ifft(x, n, axis, norm))
        self.assert_close(nfft.fft(numerical.asarray(x)), np.fft.fft(x))  # a Tensor as well as an array
        self.assert_close(nfft.fft([1, 2, 3, 4]), np.fft.fft([1, 2, 3, 4]))  # integers as complex128

    def test_n_dimensional_transforms_are_numpys(self):
        x = self.operand((6, 8, 5), np.complex128)
        self.assert_close(nfft.fftn(x), np.fft.fftn(x))
        self.assert_close(nfft.ifftn(x), np.fft.ifftn(x))
        self.assert_close(nfft.fftn(x, s=(4, 10), axes=(1, 2)), np.fft.fftn(x, s=(4, 10), axes=(1, 2)))
        self.assert_close(nfft.fftn(x, axes=(0, 2), norm="ortho"), np.fft.fftn(x, axes=(0, 2), norm="ortho"))
        self.assert_close(nfft.fft2(x), np.fft.fft2(x))
        self.assert_close(nfft.ifft2(x, s=(3, 3), axes=(-2, -1)), np.fft.ifft2(x, s=(3, 3), axes=(-2, -1)))
        with self.assertRaises(ValueError):
            nfft.fftn(x, s=(4, 10), axes=(0,))

    def test_frequencies_and_shifts_are_numpys_to_the_bit(self):
        for n in (1, 2, 7, 8, 1000):
            for d in (1.0, 0.1, 3.0):
                self.assertEqual(np.asarray(nfft.fftfreq(n, d)).tobytes(), np.fft.fftfreq(n, d).tobytes())
                self.assertEqual(np.asarray(nfft.rfftfreq(n, d)).tobytes(), np.fft.rfftfreq(n, d).tobytes())
        x = self.operand((5, 6, 7), np.complex128)
        for axes in (None, 0, 1, -1, (0, 2)):
            self.assertTrue(np.array_equal(np.asarray(nfft.fftshift(x, axes)), np.fft.fftshift(x, axes)))
            self.assertTrue(np.array_equal(np.asarray(nfft.ifftshift(x, axes)), np.fft.ifftshift(x, axes)))

    def test_errors_are_numpys(self):
        x = self.operand((3, 4), np.float64)
        with self.assertRaisesRegex(ValueError, "Invalid norm value spam"):
            nfft.fft(x, norm="spam")
        with self.assertRaisesRegex(ValueError, r"Invalid number of FFT data points \(0\) specified."):
            nfft.ifft(x, 0)
        with self.assertRaisesRegex(ValueError, r"Invalid number of FFT data points \(0\) specified."):
            nfft.fft([])
        with self.assertRaises(IndexError):
            nfft.fft(x, axis=2)
        with self.assertRaises(IndexError):
            nfft.fftshift(x, -3)
        with self.assertRaises(TypeError):
            nfft.fft(x, out=np.empty(4, complex))
        with self.assertRaises(ZeroDivisionError):
            nfft.fftfreq(0)
        with self.assertRaises(ValueError):
            nfft.fftfreq(4, device="gpu")
        with self.assertRaises(ValueError):
            nfft.rfftfreq(4, device="gpu")
        self.assertIs(numerical.fft, nfft)


if __name__ == "__main__":
    unittest.main()
