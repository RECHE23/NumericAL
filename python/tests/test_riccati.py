"""solve_continuous_are and solve_discrete_are against their equations, and against scipy.linalg where it is installed
(but for the operands its balancing gets wrong: e with balanced=True, scipy/scipy#26555)."""

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


def _drawn(shape, complex_entries, rng):
    x = rng.standard_normal(shape)
    return x + 1j * rng.standard_normal(shape) if complex_entries else x


def _operands(m, n, complex_entries, rng):
    """An equation of order `m` with `n` inputs: `q` and `r` positive definite, `e` well away from singular."""
    q0 = _drawn((m, m), complex_entries, rng)
    r0 = _drawn((n, n), complex_entries, rng)
    return (0.3 * _drawn((m, m), complex_entries, rng), _drawn((m, n), complex_entries, rng),
            q0.conj().T @ q0 + np.eye(m), r0.conj().T @ r0 + np.eye(n),
            _drawn((m, m), complex_entries, rng) + (m + 1) * np.eye(m), 0.1 * _drawn((m, n), complex_entries, rng))


def _continuous_residual(a, b, q, r, e, s, x):
    """The equation's residual over its terms' sizes, infinite where the closed loop is not stable."""
    h = lambda m: m.conj().T  # noqa: E731
    g = h(e) @ x @ b + s
    gain = np.linalg.solve(r, h(g))
    if np.linalg.eigvals(np.linalg.solve(e, a - b @ gain)).real.max() >= 0:
        return np.inf
    terms = 2 * np.abs(a).max() * np.abs(x).max() * np.abs(e).max() + np.abs(g @ gain).max() + np.abs(q).max()
    return np.abs(h(a) @ x @ e + h(e) @ x @ a - g @ gain + q).max() / terms


def _discrete_residual(a, b, q, r, e, s, x):
    """The equation's residual over its terms' sizes, infinite where the closed loop is not stable."""
    h = lambda m: m.conj().T  # noqa: E731
    g = h(a) @ x @ b + s
    gain = np.linalg.solve(r + h(b) @ x @ b, h(g))
    if np.abs(np.linalg.eigvals(np.linalg.solve(e, a - b @ gain))).max() >= 1:
        return np.inf
    terms = (np.abs(a).max() ** 2 + np.abs(e).max() ** 2) * np.abs(x).max() + np.abs(g @ gain).max() + np.abs(q).max()
    return np.abs(h(a) @ x @ a - h(e) @ x @ e - g @ gain + q).max() / terms


@unittest.skipIf(np is None, "NumPy is not installed")
class Riccati(unittest.TestCase):
    def test_the_equations_hold(self):
        # real and complex, e and s each given or not, balanced or not: Hermitian, stabilising, to rounding
        rng = np.random.default_rng(7)
        for m, n in ((1, 1), (3, 2), (12, 4)):
            for complex_entries in (False, True):
                a, b, q, r, e, s = _operands(m, n, complex_entries, rng)
                for with_e in (False, True):
                    for with_s in (False, True):
                        for balanced in (True, False):
                            options = dict(e=e if with_e else None, s=s if with_s else None, balanced=balanced)
                            ee = e if with_e else np.eye(m)
                            ss = s if with_s else np.zeros((m, n))
                            with self.subTest(m=m, complex=complex_entries, e=with_e, s=with_s, balanced=balanced):
                                x = np.asarray(numerical.solve_continuous_are(a, b, q, r, **options))
                                self.assertEqual(x.dtype, np.complex128 if complex_entries else np.float64)
                                self.assertTrue(np.array_equal(x, x.conj().T))
                                self.assertLess(_continuous_residual(a, b, q, r, ee, ss, x), 1e-11)
                                y = np.asarray(numerical.solve_discrete_are(a, b, q, r, **options))
                                self.assertTrue(np.array_equal(y, y.conj().T))
                                self.assertLess(_discrete_residual(a, b, q, r, ee, ss, y), 1e-11)

    def test_single_precision_and_mixed_operands(self):
        # float32 operands computed in double and rounded; a complex q makes the answer complex
        a, b, q, r = (np.array(v, np.float32) for v in ([[-1.0]], [[1.0]], [[3.0]], [[1.0]]))
        x = numerical.solve_continuous_are(a, b, q, r)
        self.assertEqual(x.dtype, "float32")
        self.assertAlmostEqual(float(np.asarray(x)[0, 0]), 1.0, places=6)  # −2X − X² + 3 = 0
        z = numerical.solve_continuous_are([[-1.0]], [[1.0]], np.array([[3.0 + 0j]]), [[1.0]])
        self.assertEqual(z.dtype, "complex128")

    def test_single_precision_hermitian_at_its_precision(self):
        # complex64 q and r Hermitian to float32's rounding (a product B Bᴴ formed in it) accepted, as SciPy checks them
        # at their own precision; one off by more refused
        rng = np.random.default_rng(31)
        for m, n in ((9, 8), (24, 5), (55, 1)):
            b0 = (rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))).astype(np.complex64)
            q = (b0 @ b0.conj().T / m + np.eye(m)).astype(np.complex64)
            r0 = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(np.complex64)
            r = (r0 @ r0.conj().T / n + np.eye(n)).astype(np.complex64)
            a = (rng.standard_normal((m, m)) / np.sqrt(m) - 1.5 * np.eye(m)).astype(np.complex64)
            b = rng.standard_normal((m, n)).astype(np.complex64)
            x = numerical.solve_continuous_are(a, b, q, r)
            self.assertEqual(x.dtype, "complex64")
            y = numerical.solve_discrete_are((a + 1.5 * np.eye(m, dtype=np.complex64)) / 2, b, q, r)
            self.assertEqual(y.dtype, "complex64")
        skewed = np.eye(2, dtype=np.complex64)
        skewed[0, 1] = 1e-3
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_continuous_are(-np.eye(2, dtype=np.complex64), np.ones((2, 1), np.complex64), skewed,
                                           np.eye(1, dtype=np.complex64))

    def test_balancing_scales_e(self):
        # SciPy 1.18 answers this one with residuals 3 (continuous) and 2 (discrete)
        a = np.array([[3.0, -3.0], [-1.0, -1.0]])
        b = np.array([[3.0], [-2.0]])
        e = np.array([[1.0, -2.0], [0.0, 100.0]])
        q, r, s = np.eye(2), np.eye(1), np.zeros((2, 1))
        x = np.asarray(numerical.solve_continuous_are(a, b, q, r, e=e))
        self.assertLess(_continuous_residual(a, b, q, r, e, s, x), 1e-14)
        y = np.asarray(numerical.solve_discrete_are(a, b, q, r, e=e))
        self.assertLess(_discrete_residual(a, b, q, r, e, s, y), 1e-14)

    def test_graded_continuous_equation(self):
        # the generalized Schur form's answer has a residual of 1.3e-7 of its terms (SciPy 1.18's 9.8e-9); a step of
        # Newton's method brings it to rounding
        a = np.array([[-2.5723633516085407e+08, 4.4375165314791058e-04],
                      [1.2072959802483012e+03, -6.2102344766426814e+08]])
        b = np.array([[0.44663486401116476], [0.285525462894814]])
        q = np.array([[2.998057577496829, 0.5524690732831364], [0.5524690732831364, 3.127499944446211]])
        r = np.array([[1.0528016657039019]])
        x = np.asarray(numerical.solve_continuous_are(a, b, q, r))
        self.assertLess(_continuous_residual(a, b, q, r, np.eye(2), np.zeros((2, 1)), x), 4 * np.finfo(float).eps)

    def test_refusals(self):
        for solver in (numerical.solve_continuous_are, numerical.solve_discrete_are):
            for bad in (([[1.0, 2.0]], [[1.0]], [[1.0]], [[1.0]]),                       # a not square
                        ([[1.0]], [[1.0], [1.0]], [[1.0]], [[1.0]]),                     # b of other rows
                        ([[1.0]], [[1.0]], [[1.0]], [[1.0, 0.0], [0.0, 1.0]]),           # r of other order
                        ([[np.nan]], [[1.0]], [[1.0]], [[1.0]]),                         # not finite
                        (np.eye(2), [[1.0], [0.0]], [[1.0, 1.0], [0.0, 1.0]], [[1.0]]),  # q not symmetric
                        ([[1.0]], [[1.0, 1.0]], [[1.0]], [[1.0, 2.0], [0.0, 1.0]])):     # r not symmetric
                with self.subTest(solver=solver.__name__, bad=bad), self.assertRaises(numerical.LinAlgError):
                    solver(*bad)
            with self.assertRaises(numerical.LinAlgError):  # e singular
                solver(np.eye(2), [[1.0], [0.0]], np.eye(2), [[1.0]], e=[[1.0, 2.0], [2.0, 4.0]])
            with self.assertRaises(numerical.LinAlgError):  # s of other shape
                solver(np.eye(2), [[1.0], [0.0]], np.eye(2), [[1.0]], s=np.eye(2))
        for r in ([[1.0, 2.0], [2.0, 4.0]], np.zeros((2, 2))):  # the continuous equation's R⁻¹, as SciPy refuses it
            with self.assertRaises(numerical.LinAlgError):
                numerical.solve_continuous_are(-np.eye(2), np.eye(2), np.eye(2), r)
        with self.assertRaises(numerical.LinAlgError):  # an unstable mode b cannot reach, decoupled (q zero there)
            numerical.solve_continuous_are([[1.0]], [[0.0]], [[0.0]], [[1.0]])
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve_discrete_are([[2.0, 0.0], [0.0, 0.5]], [[0.0], [1.0]], [[0.0, 0.0], [0.0, 1.0]], [[1.0]])
        for q in (-0.5, -2.0):  # the pencil's eigenvalues on the imaginary axis, on the unit circle
            with self.assertRaises(numerical.LinAlgError):
                numerical.solve_continuous_are([[0.0]], [[1.0]], [[q]], [[1.0]])
            with self.assertRaises(numerical.LinAlgError):
                numerical.solve_discrete_are([[1.0]], [[1.0]], [[q]], [[1.0]])

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        # where SciPy's balancing is right: without e, or unbalanced
        rng = np.random.default_rng(8)
        for m, n in ((2, 1), (6, 3), (15, 4)):
            for complex_entries in (False, True):
                a, b, q, r, e, s = _operands(m, n, complex_entries, rng)
                for options in (dict(), dict(s=s), dict(e=e, balanced=False), dict(e=e, s=s, balanced=False)):
                    for ours, theirs in ((numerical.solve_continuous_are, sl.solve_continuous_are),
                                         (numerical.solve_discrete_are, sl.solve_discrete_are)):
                        with self.subTest(m=m, complex=complex_entries, options=sorted(options), solver=ours.__name__):
                            x = np.asarray(ours(a, b, q, r, **options))
                            np.testing.assert_allclose(x, theirs(a, b, q, r, **options), rtol=1e-9, atol=1e-9 * np.abs(x).max())


if __name__ == "__main__":
    unittest.main()
