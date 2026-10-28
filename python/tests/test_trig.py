"""numerical.fft's dct, idct, dst and idst (and their n-dimensional forms) against scipy.fft where it is installed, and
against their direct sums and inverses everywhere."""

import math
import unittest

import numerical
import numerical.fft as nfft

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import scipy.fft as sf
except ImportError:  # pragma: no cover
    sf = None


def _direct(sine, type, x):
    """The unscaled transform of the list `x` by its sum."""
    n = len(x)
    out = []
    for k in range(n):
        total = 0.0
        for j, value in enumerate(x):
            if not sine:
                if type == 1:
                    term = value * math.cos(math.pi * j * k / (n - 1)) * (1 if j in (0, n - 1) else 2)
                elif type == 2:
                    term = 2 * value * math.cos(math.pi * k * (2 * j + 1) / (2 * n))
                elif type == 3:
                    term = value if j == 0 else 2 * value * math.cos(math.pi * j * (2 * k + 1) / (2 * n))
                else:
                    term = 2 * value * math.cos(math.pi * (2 * j + 1) * (2 * k + 1) / (4 * n))
            elif type == 1:
                term = 2 * value * math.sin(math.pi * (j + 1) * (k + 1) / (n + 1))
            elif type == 2:
                term = 2 * value * math.sin(math.pi * (k + 1) * (2 * j + 1) / (2 * n))
            elif type == 3:
                term = (-1) ** k * value if j == n - 1 else 2 * value * math.sin(math.pi * (j + 1) * (2 * k + 1) / (2 * n))
            else:
                term = 2 * value * math.sin(math.pi * (2 * j + 1) * (2 * k + 1) / (4 * n))
            total += term
        out.append(total)
    return out


class DirectSums(unittest.TestCase):
    def test_every_family_and_type_is_its_sum_and_inverts(self):
        for n in (1, 2, 3, 4, 5, 8, 9, 16, 17, 30):
            x = [math.sin(1 + 0.7 * j) for j in range(n)]
            for sine, forward, inverse in ((False, nfft.dct, nfft.idct), (True, nfft.dst, nfft.idst)):
                for type in (1, 2, 3, 4):
                    if not sine and type == 1 and n < 2:
                        continue
                    y = forward(x, type).tolist()
                    expected = _direct(sine, type, x)
                    scale = max(1.0, max(abs(v) for v in expected))
                    for got, want in zip(y, expected):
                        self.assertAlmostEqual(got, want, delta=1e-12 * scale)
                    for got, want in zip(inverse(y, type).tolist(), x):
                        self.assertAlmostEqual(got, want, delta=1e-12)

    def test_errors(self):
        with self.assertRaises(ValueError):
            nfft.dct([1.0, 2.0], 5)
        with self.assertRaises(ValueError):
            nfft.dct([1.0], 1)
        with self.assertRaises(ValueError):
            nfft.dst([1.0, 2.0], n=0)
        with self.assertRaises(ValueError):
            nfft.dct([1.0, 2.0], norm="none")
        with self.assertRaises(IndexError):
            nfft.dct([1.0, 2.0], axis=1)


@unittest.skipIf(np is None or sf is None, "NumPy or SciPy is not installed")
class AgainstSciPy(unittest.TestCase):
    def test_every_option(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((3, 11, 8))
        for name in ("dct", "idct", "dst", "idst"):
            ours, theirs = getattr(nfft, name), getattr(sf, name)
            for type in (1, 2, 3, 4):
                for norm in (None, "backward", "ortho", "forward"):
                    for orthogonalize in (None, False, True):
                        for axis, n in ((-1, None), (1, None), (0, 5), (1, 14), (-1, 3)):
                            expected = theirs(x, type, n, axis, norm, orthogonalize=orthogonalize)
                            got = np.asarray(ours(x, type, n, axis, norm, orthogonalize=orthogonalize))
                            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max(),
                                                       err_msg=f"{name} {type} {norm} {orthogonalize} {axis} {n}")

    def test_dtypes(self):
        rng = np.random.default_rng(6)
        for dtype in ("float64", "float32", "complex128", "complex64", "int64"):
            x = rng.standard_normal((4, 9)) * 10
            if dtype.startswith("complex"):
                x = x + 1j * rng.standard_normal((4, 9))
            x = x.astype(dtype)
            got = np.asarray(nfft.dct(x, 2, norm="ortho"))
            expected = sf.dct(x, 2, norm="ortho")
            self.assertEqual(got.dtype, expected.dtype)
            tolerance = 1e-12 if expected.dtype in (np.float64, np.complex128) else 1e-5
            np.testing.assert_allclose(got, expected, rtol=tolerance, atol=tolerance * np.abs(expected).max())

    def test_n_dimensional(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((5, 6, 7))
        for name in ("dctn", "idctn", "dstn", "idstn"):
            ours, theirs = getattr(nfft, name), getattr(sf, name)
            for type in (1, 2, 3, 4):
                for s, axes in ((None, None), (None, (0, 2)), ((4, 9), None), ((8,), 1), (None, -1)):
                    expected = theirs(x, type, s, axes, "ortho")
                    got = np.asarray(ours(x, type, s, axes, "ortho"))
                    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max(),
                                               err_msg=f"{name} {type} {s} {axes}")

    def test_lengths_of_every_plan(self):
        rng = np.random.default_rng(8)
        for n in (2, 6, 7, 64, 97, 128, 1000, 1009, 4096):
            x = rng.standard_normal(n)
            for type in (1, 2, 3, 4):
                for name in ("dct", "dst"):
                    expected = getattr(sf, name)(x, type)
                    got = np.asarray(getattr(nfft, name)(x, type))
                    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11 * np.abs(expected).max(),
                                               err_msg=f"{name} {type} {n}")


if __name__ == "__main__":
    unittest.main()
