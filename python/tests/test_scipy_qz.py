"""scipy.linalg's qz and ordqz against SciPy's own: the decomposition by its identities and structure, the eigenvalues
as SciPy's (as sets, the order of a QZ's deflations its own), the selection of ordqz by each sort and a callable, the
real and complex forms, single precision, a singular B (infinite eigenvalues) and the refusals. Deterministic; skipped
when SciPy is not importable (CI installs it)."""
import itertools
import unittest

import numerical

try:
    import numpy as np
    import scipy.linalg as sl
except ImportError:
    np = sl = None


def _close_sets(x, y, tolerance):
    """Whether the complex numbers `x` and `y` match one to one within `tolerance` of their size (a greedy match)."""
    x, y = list(x), list(y)
    if len(x) != len(y):
        return False
    for value in x:
        distances = [abs(value - other) if np.isfinite(value) else (0.0 if not np.isfinite(other) else np.inf) for other in y]
        k = int(np.argmin(distances))
        if distances[k] > tolerance * max(1.0, abs(value) if np.isfinite(value) else 1.0):
            return False
        y.pop(k)
    return True


def _ratios(alpha, beta):
    alpha, beta = np.asarray(alpha), np.asarray(beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(beta != 0, alpha / np.where(beta != 0, beta, 1), np.inf)


@unittest.skipIf(sl is None, "SciPy not installed")
class QZ(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261109)

    def pencil(self, n, dtype):
        a, b = self.rng.standard_normal((n, n)), self.rng.standard_normal((n, n))
        if np.dtype(dtype).kind == "c":
            a, b = a + 1j * self.rng.standard_normal((n, n)), b + 1j * self.rng.standard_normal((n, n))
        return a.astype(dtype), b.astype(dtype)

    def check(self, a, b, aa, bb, q, z, real_form):
        aa, bb, q, z = (np.asarray(x) for x in (aa, bb, q, z))
        tolerance = 1e-4 if aa.dtype in (np.float32, np.complex64) else 1e-12
        self.assertTrue(np.allclose(q @ aa @ z.conj().T, a, atol=tolerance * max(1.0, np.abs(a).max())))
        self.assertTrue(np.allclose(q @ bb @ z.conj().T, b, atol=tolerance * max(1.0, np.abs(b).max())))
        self.assertTrue(np.allclose(q.conj().T @ q, np.eye(len(a)), atol=tolerance))
        self.assertTrue(np.allclose(z.conj().T @ z, np.eye(len(a)), atol=tolerance))
        self.assertTrue(np.array_equal(np.tril(bb, -1), np.zeros_like(bb)))
        # quasi-triangular (the 2 × 2 blocks' subdiagonal) for a real form, else triangular
        self.assertTrue(np.array_equal(np.tril(aa, -2 if real_form else -1), np.zeros_like(aa)))

    def test_qz_as_scipy(self):
        for n, dtype, output in itertools.product((1, 2, 5, 12, 40), (np.float64, np.complex128, np.float32), ("real", "complex")):
            a, b = self.pencil(n, dtype)
            with self.subTest(n=n, dtype=np.dtype(dtype).name, output=output):
                aa, bb, q, z = numerical.qz(a, b, output=output)
                expected = sl.qz(a, b, output=output)
                self.assertEqual(np.asarray(aa).dtype, expected[0].dtype)
                real_form = output == "real" and np.dtype(dtype).kind == "f"
                self.check(a, b, aa, bb, q, z, real_form)
                # the eigenvalues as SciPy's: the ratios of the diagonals (a real form's 2 × 2 blocks by ordqz's alpha)
                ours = numerical.ordqz(a, b, sort=lambda x, y: np.zeros(len(x), bool), output=output)
                theirs = sl.ordqz(a, b, sort=lambda x, y: np.zeros(len(x), bool), output=output)
                tolerance = 1e-3 if np.dtype(dtype) in (np.float32, np.complex64) else 1e-9
                self.assertTrue(_close_sets(_ratios(ours[2], ours[3]), _ratios(theirs[2], theirs[3]), tolerance))

    def test_ordqz_selects_as_scipy(self):
        sorts = ["lhp", "rhp", "iuc", "ouc", lambda x, y: np.real(x) > 0.5 * np.abs(y)]
        for n, dtype, output, sort in itertools.product((3, 8, 25), (np.float64, np.complex128), ("real", "complex"), sorts):
            a, b = self.pencil(n, dtype)
            with self.subTest(n=n, dtype=np.dtype(dtype).name, output=output, sort=sort if isinstance(sort, str) else "callable"):
                aa, bb, alpha, beta, q, z = numerical.ordqz(a, b, sort=sort, output=output)
                expected = sl.ordqz(a, b, sort=sort, output=output)
                self.check(a, b, aa, bb, q, z, output == "real" and np.dtype(dtype).kind == "f")
                select = sort if callable(sort) else {"lhp": sl._decomp_qz._lhp, "rhp": sl._decomp_qz._rhp,
                                                      "iuc": sl._decomp_qz._iuc, "ouc": sl._decomp_qz._ouc}[sort]
                ours = np.asarray(select(np.asarray(alpha), np.asarray(beta)))
                theirs = np.asarray(select(expected[2], expected[3]))
                count = int(theirs.sum())
                self.assertEqual(int(ours.sum()), count)
                self.assertTrue(ours[:count].all())  # the selected ones lead
                self.assertTrue(_close_sets(_ratios(alpha, beta)[:count], _ratios(expected[2], expected[3])[:count], 1e-9))

    def test_singular_b_gives_infinite_eigenvalues(self):
        a, b = self.pencil(6, np.float64)
        b[:, :2] = 0.0  # rank 4: two infinite eigenvalues
        for output in ("real", "complex"):
            with self.subTest(output=output):
                aa, bb, alpha, beta, q, z = numerical.ordqz(a, b, sort="ouc", output=output)
                self.check(a, b, aa, bb, q, z, output == "real")
                infinite = np.abs(np.asarray(beta)) <= 1e-12 * np.abs(b).max()
                self.assertEqual(int(np.sum(infinite)), 2)
                chosen = int(np.sum(sl._decomp_qz._ouc(np.asarray(alpha), np.asarray(beta))))
                self.assertTrue(np.all(np.flatnonzero(infinite) < chosen))  # "ouc" takes them among the leading ones

    def test_refusals(self):
        a, b = self.pencil(3, np.float64)
        with self.assertRaises(ValueError):
            numerical.qz(a, b, sort="lhp")
        with self.assertRaises(ValueError):
            numerical.qz(a, b, output="x")
        with self.assertRaises(ValueError):
            numerical.ordqz(a, b, sort="x")
        with self.assertRaises(ValueError):
            numerical.qz(a, b[:2])
        with self.assertRaises(numerical.LinAlgError):
            numerical.qz(a, np.full((3, 3), np.nan))
        with self.assertRaises(ValueError):
            numerical.ordqz(a, b, sort=lambda x, y: [True])  # one truth per eigenvalue
        with self.assertRaises(ZeroDivisionError):
            numerical.ordqz(a, b, sort=lambda x, y: 1 / 0)  # the callable's error as it is


if __name__ == "__main__":
    unittest.main()
