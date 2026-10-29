"""expm against scipy.linalg.expm where it is installed, and against closed forms everywhere."""

import math
import unittest

import numerical

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import scipy.linalg as sl
except ImportError:  # pragma: no cover
    sl = None


@unittest.skipIf(np is None, "NumPy is not installed")
class Expm(unittest.TestCase):
    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        # every degree of the approximant and the squarings, real and complex, single precision and stacks
        rng = np.random.default_rng(7)
        for n in (1, 2, 5, 30):
            for scale in (1e-3, 0.1, 1.0, 5.0, 60.0):
                for dtype in ("float64", "complex128", "float32", "complex64"):
                    a = rng.standard_normal((n, n)) * scale / math.sqrt(n)
                    if dtype.startswith("complex"):
                        a = a + 1j * rng.standard_normal((n, n)) * scale / math.sqrt(n)
                    a = a.astype(dtype)
                    with self.subTest(n=n, scale=scale, dtype=dtype):
                        ours = np.asarray(numerical.expm(a))
                        self.assertEqual(ours.dtype, sl.expm(np.eye(2, dtype=dtype)).dtype)
                        # single precision computed in double, as SciPy's double (its single overflows at 60 to NaN)
                        wide = "complex128" if dtype.startswith("complex") else "float64"
                        theirs = sl.expm(a.astype(wide))
                        tolerance = 1e-12 if dtype in ("float64", "complex128") else 1e-4
                        self.assertLessEqual(np.abs(ours - theirs).max(), tolerance * max(np.abs(theirs).max(), 1.0))
        stack = rng.standard_normal((2, 3, 4, 4))
        np.testing.assert_allclose(np.asarray(numerical.expm(stack)), sl.expm(stack), rtol=1e-12, atol=1e-14)

    def test_closed_forms_and_refusals(self):
        angle = 0.7
        rotation = np.asarray(numerical.expm([[0.0, -angle], [angle, 0.0]]))
        np.testing.assert_allclose(rotation, [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
                                   rtol=1e-15, atol=1e-15)
        self.assertEqual(np.asarray(numerical.expm([[0, 1], [0, 0]])).tolist(), [[1.0, 1.0], [0.0, 1.0]])  # integers as float64
        self.assertEqual(np.asarray(numerical.expm(np.zeros((0, 0)))).shape, (0, 0))
        for bad in (np.ones((2, 3)), np.ones(3), np.ones((2, 3, 3, 2))):
            with self.assertRaises(numerical.LinAlgError):
                numerical.expm(bad)
        with self.assertRaises(numerical.LinAlgError):
            numerical.expm([[1.0, float("nan")], [0.0, 1.0]])


@unittest.skipIf(np is None, "NumPy is not installed")
class Sqrtm(unittest.TestCase):
    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        # positive definite, general (complex pairs), complex, with real eigenvalues below 0 (a complex root, +i √|λ|
        # as SciPy's), in single precision, and stacks of each (complex for all where any needs it)
        rng = np.random.default_rng(3)
        for n in (1, 2, 5, 30):
            for kind in ("definite", "general", "complex", "negative"):
                a = rng.standard_normal((n, n))
                if kind == "definite":
                    a = a @ a.T + np.eye(n)
                elif kind == "complex":
                    a = a + 1j * rng.standard_normal((n, n))
                elif kind == "negative":
                    a = -(a @ a.T + np.eye(n))
                with self.subTest(n=n, kind=kind):
                    ours = np.asarray(numerical.sqrtm(a))
                    theirs = sl.sqrtm(a)
                    self.assertEqual(ours.dtype, theirs.dtype)
                    self.assertLess(np.abs(ours - theirs).max(), 1e-11 * np.abs(theirs).max())
                    single = numerical.sqrtm(a.astype(np.complex64 if kind == "complex" else np.float32))
                    self.assertEqual(np.asarray(single).dtype, sl.sqrtm(a.astype(np.complex64 if kind == "complex" else np.float32)).dtype)
        stack = np.stack([np.eye(3) * 4.0, -np.eye(3)])
        np.testing.assert_array_equal(np.asarray(numerical.sqrtm(stack)), sl.sqrtm(stack))
        stack = np.stack([np.eye(3) * 4.0, np.eye(3)])
        self.assertEqual(np.asarray(numerical.sqrtm(stack)).dtype, np.float64)

    def test_edges_and_refusals(self):
        jordan = np.asarray(numerical.sqrtm([[0.0, 1.0], [0.0, 0.0]]))
        self.assertTrue(np.isinf(jordan[0, 1]) and jordan[0, 0] == 0.0 and jordan[1, 1] == 0.0)
        self.assertEqual(np.asarray(numerical.sqrtm(np.zeros((2, 2)))).tolist(), [[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(np.asarray(numerical.sqrtm([[4]])).tolist(), [[2.0]])
        self.assertEqual(np.asarray(numerical.sqrtm(np.zeros((0, 0)))).shape, (0, 0))
        for bad in (np.ones((2, 3)), np.ones(3), [[float("inf")]]):
            with self.assertRaises(numerical.LinAlgError):
                numerical.sqrtm(bad)


if __name__ == "__main__":
    unittest.main()
