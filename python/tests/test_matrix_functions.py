"""expm, expm_frechet, expm_cond, funm, logm, sqrtm and polar against scipy.linalg where it is installed, and against closed
forms everywhere."""

import math
import unittest
import warnings

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


def _sqrtm_dtype(theirs, given):
    """The dtype SciPy's sqrtm answers in for an operand of dtype `given`: an older SciPy's extended precision complex
    (complex256, whatever the operand's precision) read as the complex dtype of the operand's precision."""
    if theirs.dtype == np.clongdouble and np.clongdouble != np.complex128:
        return np.result_type(given, np.complex64)
    return theirs.dtype


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
                    self.assertEqual(ours.dtype, _sqrtm_dtype(theirs, a.dtype))
                    self.assertLess(np.abs(ours - theirs).max(), 1e-11 * np.abs(theirs).max())
                    narrow = a.astype(np.complex64 if kind == "complex" else np.float32)  # in single precision, the double answer's kind
                    single = numerical.sqrtm(narrow)
                    self.assertEqual(np.asarray(single).dtype, np.complex64 if ours.dtype == np.complex128 else np.float32)
        stack = np.stack([np.eye(3) * 4.0, -np.eye(3)])
        try:
            theirs = sl.sqrtm(stack)
        except ValueError:  # an older SciPy takes no stack ("Non-matrix input to matrix function.")
            theirs = np.stack([sl.sqrtm(m).astype(complex) for m in stack])
        np.testing.assert_array_equal(np.asarray(numerical.sqrtm(stack)), theirs)
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


@unittest.skipIf(np is None, "NumPy is not installed")
class Polar(unittest.TestCase):
    def test_factors(self):
        # a = u p (right) or p u (left), u with orthonormal columns or rows, p Hermitian positive semidefinite; against
        # SciPy where installed; square, tall and wide, real and complex
        rng = np.random.default_rng(8)
        for shape in ((4, 4), (5, 3), (3, 5), (1, 1)):
            for complex_entries in (False, True):
                a = rng.standard_normal(shape)
                if complex_entries:
                    a = a + 1j * rng.standard_normal(shape)
                for side in ("right", "left"):
                    with self.subTest(shape=shape, complex_entries=complex_entries, side=side):
                        u, p = (np.asarray(t) for t in numerical.polar(a, side))
                        np.testing.assert_allclose(u @ p if side == "right" else p @ u, a, atol=1e-13)
                        np.testing.assert_allclose(p, p.conj().T, atol=1e-13)
                        self.assertGreaterEqual(np.linalg.eigvalsh(p).min(), -1e-12)
                        if sl is not None:
                            u_s, p_s = sl.polar(a, side)
                            np.testing.assert_allclose(u, u_s, atol=1e-13)
                            np.testing.assert_allclose(p, p_s, atol=1e-13)
        with self.assertRaises(ValueError):
            numerical.polar(np.eye(2), "top")
        with self.assertRaises(ValueError):
            numerical.polar(np.ones(3))



@unittest.skipIf(np is None, "NumPy is not installed")
class ExpmFrechet(unittest.TestCase):
    def test_the_derivative(self):
        # L(A, E) against a central difference of expm, every degree and the squarings; blockEnlarge the same
        rng = np.random.default_rng(12)
        for scale in (1e-9, 1e-3, 0.05, 0.3, 1.0, 3.0, 40.0):
            for complex_entries in (False, True):
                a = rng.standard_normal((5, 5)) * scale
                e = rng.standard_normal((5, 5))
                if complex_entries:
                    a = a + 1j * rng.standard_normal((5, 5)) * scale
                    e = e + 1j * rng.standard_normal((5, 5))
                with self.subTest(scale=scale, complex=complex_entries):
                    x, l = (np.asarray(v) for v in numerical.expm_frechet(a, e))
                    np.testing.assert_allclose(x, np.asarray(numerical.expm(a)), rtol=1e-12, atol=1e-13 * np.abs(x).max())
                    h = 1e-6
                    difference = (np.asarray(numerical.expm(a + h * e)) - np.asarray(numerical.expm(a - h * e))) / (2 * h)
                    np.testing.assert_allclose(l, difference, rtol=0, atol=1e-6 * max(1.0, np.abs(l).max()))
                    block = np.asarray(numerical.expm_frechet(a, e, method="blockEnlarge", compute_expm=False))
                    np.testing.assert_allclose(block, l, rtol=0, atol=1e-11 * max(1.0, np.abs(l).max()))
        single = numerical.expm_frechet(np.zeros((2, 2), np.float32), np.eye(2, dtype=np.float32))
        self.assertEqual(single[1].dtype, "float32")
        self.assertEqual(np.asarray(single[1]).tolist(), [[1.0, 0.0], [0.0, 1.0]])
        for bad in ((np.ones((2, 3)), np.ones((2, 3))), (np.eye(2), np.eye(3)), (np.eye(2), np.full((2, 2), np.nan))):
            with self.assertRaises(numerical.LinAlgError):
                numerical.expm_frechet(*bad)
        with self.assertRaises(ValueError):
            numerical.expm_frechet(np.eye(2), np.eye(2), method="other")

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        rng = np.random.default_rng(13)
        for n in (1, 3, 8):
            for scale in (1e-3, 0.5, 2.0, 30.0):
                a = rng.standard_normal((n, n)) * scale
                e = rng.standard_normal((n, n))
                for method in ("SPS", "blockEnlarge"):
                    ours = [np.asarray(v) for v in numerical.expm_frechet(a, e, method=method)]
                    theirs = sl.expm_frechet(a, e, method=method)
                    for x, y in zip(ours, theirs):
                        np.testing.assert_allclose(x, y, rtol=0, atol=1e-11 * max(1.0, np.abs(y).max()))
                if n <= 3:
                    self.assertAlmostEqual(numerical.expm_cond(a) / sl.expm_cond(a), 1.0, places=10)

    def test_cond_closed_form(self):
        # a scalar's condition number is |a|: e^a's derivative e^a, relative to e^a and a
        self.assertAlmostEqual(numerical.expm_cond([[2.0]]), 2.0, places=13)
        with self.assertRaises(numerical.LinAlgError):
            numerical.expm_cond(np.ones((2, 3)))


def _cosm_closed_form(n, lam):
    """cos of the Jordan block of order n at lam: cos⁽ᵏ⁾(lam) / k! on the k-th superdiagonal."""
    derivatives = [math.cos(lam), -math.sin(lam), -math.cos(lam), math.sin(lam)]
    out = np.zeros((n, n))
    for k in range(n):
        out += np.diag([derivatives[k % 4] / math.factorial(k)] * (n - k), k)
    return out


@unittest.skipIf(np is None, "NumPy is not installed")
class Funm(unittest.TestCase):
    def test_closed_forms(self):
        # SciPy's example (x² is A²), exp against expm, a complex function of a real matrix complex, a repeated
        # eigenvalue's undivided entries reported inaccurate
        a = np.array([[1.0, 3.0], [1.0, 4.0]])
        f = numerical.funm(a, lambda x: x * x)
        self.assertEqual(f.dtype, "float64")
        np.testing.assert_allclose(np.asarray(f), a @ a, rtol=1e-13)
        rng = np.random.default_rng(14)
        b = rng.standard_normal((6, 6))
        f, err = numerical.funm(b, np.exp, disp=False)
        np.testing.assert_allclose(np.asarray(f), np.asarray(numerical.expm(b)), rtol=0, atol=1e3 * err * np.abs(f).max())
        root = numerical.funm(np.diag([-4.0, 9.0]), np.sqrt)
        self.assertEqual(root.dtype, "complex128")
        np.testing.assert_allclose(np.asarray(root), np.diag([2j, 3.0]), atol=1e-15)
        jordan = np.array([[2.0, 1.0], [0.0, 2.0]])  # a repeated eigenvalue: Cauchy's integral on its block
        f, err = numerical.funm(jordan, np.exp, disp=False)
        np.testing.assert_allclose(np.asarray(f), np.exp(2.0) * np.array([[1.0, 1.0], [0.0, 1.0]]), rtol=1e-14)
        self.assertLess(err, 1e-15)
        cosine = numerical.funm(np.diag([2.0] * 8) + np.diag([1.0] * 7, 1), np.cos)  # SciPy's is off by 0.9
        self.assertLess(np.abs(np.asarray(cosine) - _cosm_closed_form(8, 2.0)).max(), 1e-14)
        single = numerical.funm(np.eye(2, dtype=np.float32), np.exp)
        self.assertEqual(single.dtype, "float32")
        with self.assertRaises(numerical.LinAlgError):
            numerical.funm(np.ones((2, 3)), np.exp)
        with self.assertRaises(numerical.LinAlgError):
            numerical.funm(np.eye(3), lambda x: x[:2])

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        rng = np.random.default_rng(15)
        for n in (1, 4, 10):
            for complex_entries in (False, True):
                a = rng.standard_normal((n, n))
                if complex_entries:
                    a = a + 1j * rng.standard_normal((n, n))
                for func in (np.exp, np.sin, lambda x: x ** 3 + 2 * x):
                    ours, err = numerical.funm(a, func, disp=False)
                    theirs, their_err = sl.funm(a, func, disp=False)
                    self.assertEqual(ours.dtype, str(theirs.dtype))
                    np.testing.assert_allclose(np.asarray(ours), theirs, rtol=0, atol=1e3 * max(err, their_err) * max(1.0, np.abs(theirs).max()))



@unittest.skipIf(np is None, "NumPy is not installed")
class Logm(unittest.TestCase):
    def test_inverts_expm_and_warns(self):
        rng = np.random.default_rng(16)
        for n in (1, 4, 12):
            a = rng.standard_normal((n, n)) + n * np.eye(n)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                f = numerical.logm(a)
            self.assertEqual(f.dtype, "float64")
            np.testing.assert_allclose(np.asarray(numerical.expm(f)), a, rtol=0, atol=1e-12 * np.abs(a).max())
        minus = np.asarray(numerical.logm(-np.eye(2)))
        np.testing.assert_allclose(minus, np.pi * 1j * np.eye(2), atol=1e-15)
        stack = numerical.logm(np.stack([2 * np.eye(2), -np.eye(2)]))
        self.assertEqual(stack.dtype, "complex128")
        self.assertEqual(numerical.logm(np.eye(2, dtype=np.float32)).dtype, "float32")
        with warnings.catch_warnings(record=True) as caught:  # a zero eigenvalue taken as 1e-20: A rebuilt closely
            warnings.simplefilter("always")
            numerical.logm([[0.0, 1.0], [0.0, 1.0]])
        self.assertEqual({w.category for w in caught}, {numerical.LogmExactlySingularWarning})
        with warnings.catch_warnings(record=True) as caught:  # the zero matrix: ‖expm(F) − A‖₁ / ‖A‖₁ infinite
            warnings.simplefilter("always")
            numerical.logm(np.zeros((2, 2)))
        self.assertEqual({w.category for w in caught}, {numerical.LogmExactlySingularWarning, RuntimeWarning})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            numerical.logm([[1e-25, 0.0], [0.0, 1.0]])
        self.assertIn(numerical.LogmNearlySingularWarning, {w.category for w in caught})
        for bad in (np.ones((2, 3)), [[float("nan")]]):
            with self.assertRaises(numerical.LinAlgError):
                numerical.logm(bad)

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        rng = np.random.default_rng(17)
        for n in (1, 3, 9):
            for kind in ("real", "negative", "complex", "triangular", "near identity"):
                a = rng.standard_normal((n, n))
                if kind == "real":
                    a = a + n * np.eye(n)
                elif kind == "complex":
                    a = a + 1j * rng.standard_normal((n, n))
                elif kind == "triangular":
                    a = np.triu(a) + 2 * np.eye(n)
                elif kind == "near identity":
                    a = np.eye(n) + 1e-4 * a
                with self.subTest(n=n, kind=kind), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ours = np.asarray(numerical.logm(a))
                    theirs = sl.logm(a)
                self.assertEqual(ours.dtype, theirs.dtype)
                np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-11 * max(1.0, np.abs(theirs).max()))


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(np is None, "NumPy is not installed")
class FractionalMatrixPower(unittest.TestCase):
    def test_inverts_whole_powers(self):
        a = np.array([[4.0, 1.0, 0.5], [0.3, 3.0, 0.2], [0.1, 0.4, 2.0]])
        for k in (2, 3, 4):
            x = np.asarray(numerical.fractional_matrix_power(a, 1.0 / k))
            self.assertEqual(x.dtype, np.float64)  # real eigenvalues above 0: real
            self.assertLess(np.abs(np.linalg.matrix_power(x, k) - a).max(), 1e-13)
        negative = np.array([[-2.0, 1.0], [0.0, 3.0]])
        root = np.asarray(numerical.fractional_matrix_power(negative, 0.5))
        self.assertEqual(root.dtype, np.complex128)
        self.assertLess(np.abs(root @ root - negative).max(), 1e-13)  # no principal branch: no exact diagonal
        single = np.asarray(numerical.fractional_matrix_power(a.astype(np.float32), 0.5))
        self.assertEqual(single.dtype, np.float32)
        with self.assertRaises(numerical.LinAlgError):
            numerical.fractional_matrix_power([[0.0, 1.0], [0.0, 4.0]], -0.5)  # a negative power of a singular matrix
        with self.assertRaises(ValueError):
            numerical.fractional_matrix_power(a, float("nan"))
        with self.assertRaises(numerical.LinAlgError):
            numerical.fractional_matrix_power([[1.0, 2.0]], 0.5)

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        rng = np.random.default_rng(11)
        for n in (1, 2, 5, 20):
            for p in (0.5, -0.5, 1.3, 2.7, -1.25, 3.0, 1.0 / 3.0):
                for cplx in (False, True):
                    a = rng.standard_normal((n, n)) / math.sqrt(n) + 2.0 * np.eye(n)
                    if cplx:
                        a = a + 1j * rng.standard_normal((n, n)) / math.sqrt(n)
                    with self.subTest(n=n, p=p, cplx=cplx):
                        ours = np.asarray(numerical.fractional_matrix_power(a, p))
                        theirs = sl.fractional_matrix_power(a, p)
                        self.assertLess(np.abs(ours - theirs).max() / np.abs(theirs).max(), 1e-12)
