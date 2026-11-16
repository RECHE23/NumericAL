"""A gallery of classical hard matrices through every decomposition and each of its methods, at scales from the
subnormal to 2¹⁰⁰⁰: each answer finite (or an explicit LinAlgError where the matrix has none), within a backward error
of order n ε ‖A‖, its factors orthonormal. Random matrices, which the differential fuzzers draw, missed four defects
these find: a skew-symmetric tridiagonal matrix's bulges left subnormal entries (eig), columns of a matrix of rank 1
stayed along each other to rounding (Jacobi SVD), a stop at 10⁻¹² of the norm (Jacobi eigh), a zero pencil (eig(a, b))."""
import os
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


def _skew_tridiagonal(n):
    return np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)


def _wilkinson(n):  # W⁺: |i − m| on the diagonal, ones beside it
    m = (n - 1) / 2.0
    return np.diag(np.abs(np.arange(n) - m)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)


def _glued_wilkinson(n):  # copies of W⁺₇ glued by 1e-10: clusters of eigenvalues as close as that
    block = _wilkinson(7)
    copies = max(1, n // 7)
    a = np.zeros((7 * copies, 7 * copies))
    for k in range(copies):
        a[7 * k:7 * k + 7, 7 * k:7 * k + 7] = block
        if k:
            a[7 * k, 7 * k - 1] = a[7 * k - 1, 7 * k] = 1e-10
    return a


def _kahan(n, theta=1.2):
    s, c = np.sin(theta), np.cos(theta)
    return np.diag(s ** np.arange(n)) @ (np.eye(n) - c * np.triu(np.ones((n, n)), 1))


def _grcar(n):
    return np.eye(n, k=-1) * -1 + np.triu(np.ones((n, n))) - np.triu(np.ones((n, n)), 4)


def _frank(n):
    i, j = np.indices((n, n))
    return np.where(j >= i - 1, n - np.maximum(i, j), 0.0).astype(float)


def _hilbert(n):
    i, j = np.indices((n, n))
    return 1.0 / (i + j + 1.0)


def _minij(n):  # min(i, j) + 1: positive definite
    i, j = np.indices((n, n))
    return np.minimum(i, j) + 1.0


def _rank(n, k):  # an exact integer matrix of rank k
    r = np.random.default_rng(n * 31 + k)
    return (r.integers(-3, 4, (n, k)) @ r.integers(-3, 4, (k, n))).astype(float)


def _jordan(n):  # one eigenvalue 2, a single block: defective
    return 2.0 * np.eye(n) + np.eye(n, k=1)


def _companion(n):  # of (x − 1)(x − 2)…(x − n) scaled: distinct real roots, far from normal
    roots = np.arange(1, n + 1) / n
    c = np.poly(roots)
    a = np.zeros((n, n))
    a[0, :] = -c[1:]
    a[1:, :-1] = np.eye(n - 1)
    return a


_SQUARE = {
    "skew tridiagonal": (_skew_tridiagonal, (2, 23, 55)),
    "Wilkinson": (_wilkinson, (3, 21)),
    "glued Wilkinson": (_glued_wilkinson, (21, 42)),
    "Kahan": (_kahan, (5, 30)),
    "Grcar": (_grcar, (2, 10, 40)),
    "Frank": (_frank, (5, 20)),
    "Hilbert": (_hilbert, (5, 12)),
    "rank 1": (lambda n: _rank(n, 1), (2, 7, 40)),
    "rank 3": (lambda n: _rank(n, 3), (10, 40)),
    "Jordan": (_jordan, (2, 8)),
    "companion": (_companion, (4, 10)),
    "zero": (lambda n: np.zeros((n, n)), (1, 4)),
    "ones": (lambda n: np.ones((n, n)), (3, 40)),
}

_SYMMETRIC = {
    "Wilkinson": (_wilkinson, (3, 21)),
    "glued Wilkinson": (_glued_wilkinson, (21, 42)),
    "Hilbert": (_hilbert, (5, 12)),
    "min(i, j) + 1": (_minij, (7, 40)),
    "rank 1": (lambda n: (lambda r: r @ r.T)(_rank(n, 1)), (2, 40)),
    "ones": (lambda n: np.ones((n, n)), (3, 40)),
    "zero": (lambda n: np.zeros((n, n)), (1, 4)),
    "skew squared": (lambda n: _skew_tridiagonal(n) @ _skew_tridiagonal(n).T, (23, 55)),
}

def _random(n):  # dense, of normal entries: past order 150 the blocked Hessenberg reduction, past 200 eig's deflation
    return np.random.default_rng(n).standard_normal((n, n))


# NUMERICAL_GALLERY_LARGE (make gallery-long, the weekly long fuzzing): orders past each algorithm's thresholds, where
# the defects lay (the blocked Hessenberg reduction from 150, the Jacobi SVD of rank 1 from 160, eig's aggressive
# deflation from 200); a minute or so, beside the suite's 0.3 s.
if os.environ.get("NUMERICAL_GALLERY_LARGE") and np is not None:
    for _name, _orders in (("skew tridiagonal", (150, 200, 256)), ("rank 1", (160, 200, 256)), ("ones", (200,)),
                           ("glued Wilkinson", (210,)), ("Grcar", (200,))):
        _SQUARE[_name] = (_SQUARE[_name][0], _SQUARE[_name][1] + _orders)
    _SQUARE["random"] = (_random, (150, 200, 256))
    for _name, _orders in (("min(i, j) + 1", (200,)), ("rank 1", (200,)), ("glued Wilkinson", (210,)),
                           ("skew squared", (200,))):
        _SYMMETRIC[_name] = (_SYMMETRIC[_name][0], _SYMMETRIC[_name][1] + _orders)

# 2⁻¹⁰⁰⁰ and 2¹⁰⁰⁰ exact powers of two; 2⁻¹⁰⁶⁰ puts most entries among the subnormal numbers
_SCALES = (1.0, 2.0 ** -1000, 2.0 ** -1060, 2.0 ** 1000)
_EPS = np.finfo(float).eps if np is not None else 2.0 ** -52
_TINY = 5e-324


def _norm(x):
    return float(np.abs(np.asarray(x)).max()) if np.asarray(x).size else 0.0


@unittest.skipIf(np is None, "NumPy not installed")
class Gallery(unittest.TestCase):
    def assertSmall(self, gap, scale, n, label, factor=64.0):
        """|gap| within factor n ε scale, and some subnormal spacings: the error a backward stable method leaves."""
        self.assertTrue(np.isfinite(gap), f"{label}: not finite")
        self.assertLessEqual(gap, factor * n * _EPS * scale + 1e3 * n * _TINY, label)

    def assertOrthonormal(self, q, label):
        q = np.asarray(q)
        k = q.shape[1]
        self.assertTrue(np.isfinite(q).all(), f"{label}: not finite")
        self.assertLessEqual(_norm(q.conj().T @ q - np.eye(k)), 64.0 * max(q.shape) * _EPS, label)

    def cases(self, gallery):
        for name, (make, orders) in gallery.items():
            for n in orders:
                base = make(n)
                for scale in _SCALES:
                    yield f"{name} {n} × {scale:.0e}", base * scale

    def test_eig_eigvals_schur_hessenberg(self):
        for label, a in self.cases(_SQUARE):
            with self.subTest(label):
                n, size = a.shape[0], _norm(a)
                w, v = (np.asarray(x) for x in numerical.eig(a))
                self.assertTrue(np.isfinite(w).all() and np.isfinite(v).all(), f"{label}: eig not finite")
                # each column of unit norm: the residual of a backward stable eigendecomposition
                self.assertSmall(_norm(a @ v - v * w), size, n, f"{label}: eig residual", factor=256.0)
                self.assertTrue(np.isfinite(np.asarray(numerical.eigvals(a))).all(), f"{label}: eigvals not finite")
                t, z = (np.asarray(x) for x in numerical.schur(a))
                self.assertOrthonormal(z, f"{label}: schur Z")
                self.assertSmall(_norm(z @ t @ z.T - a), size, n, f"{label}: schur")
                h, q = (np.asarray(x) for x in numerical.hessenberg(a, calc_q=True))
                self.assertOrthonormal(q, f"{label}: hessenberg Q")
                self.assertSmall(_norm(q @ h @ q.T - a), size, n, f"{label}: hessenberg")
                self.assertEqual(_norm(np.tril(h, -2)), 0.0, f"{label}: hessenberg form")

    def test_eigh_each_method(self):
        for label, a in self.cases(_SYMMETRIC):
            for method in ("divide_and_conquer", "tridiagonal_ql", "jacobi"):
                with self.subTest(label, method=method):
                    n, size = a.shape[0], _norm(a)
                    w, v = (np.asarray(x) for x in numerical.eigh(a, method=method))
                    self.assertOrthonormal(v, f"{label} {method}: vectors")
                    self.assertSmall(_norm(a @ v - v * w), size, n, f"{label} {method}: residual")

    def test_svd_each_method(self):
        shapes = dict(_SQUARE)
        shapes["rank 1, 40 × 25"] = (lambda n: _rank(40, 1)[:, :25], (40,))
        shapes["ones, 25 × 28"] = (lambda n: np.ones((25, 28)), (25,))
        for label, a in self.cases(shapes):
            for method in ("divide_and_conquer", "bidiagonal_qr", "jacobi"):
                with self.subTest(label, method=method):
                    size = _norm(a)
                    u, s, vh = (np.asarray(x) for x in numerical.svd(a, full_matrices=False, method=method))
                    self.assertOrthonormal(u, f"{label} {method}: U")
                    self.assertOrthonormal(vh.conj().T, f"{label} {method}: V")
                    self.assertTrue((s >= 0).all() and (np.diff(s) <= 0).all(), f"{label} {method}: values")
                    self.assertSmall(_norm((u * s) @ vh - a), size, max(a.shape), f"{label} {method}: residual")

    def test_qr_and_solve(self):
        for label, a in self.cases(_SQUARE):
            with self.subTest(label):
                n, size = a.shape[0], _norm(a)
                q, r = (np.asarray(x) for x in numerical.qr(a))
                self.assertOrthonormal(q, f"{label}: Q")
                self.assertSmall(_norm(q @ r - a), size, n, f"{label}: QR")
                b = np.arange(1.0, n + 1.0) * (size if size else 1.0)
                try:
                    x = np.asarray(numerical.solve(a, b))
                except numerical.LinAlgError:
                    continue  # singular to the bit: no solution
                if np.isfinite(x).all():  # a singular matrix may give infinities, as LAPACK's
                    # the backward error, and the subnormal spacing the entries hold, which A x rounds to as well
                    bound = 64.0 * n * _EPS * (size * _norm(x) + _norm(b)) + 1e3 * n * _TINY * _norm(x)
                    self.assertLessEqual(_norm(a @ x - b), bound, f"{label}: solve backward error")

    def test_cholesky_of_definite_matrices(self):
        for name, make, orders in (("Hilbert", _hilbert, (5, 12)), ("min(i, j) + 1", _minij, (7, 40)),
                                   ("Wilkinson + 6 I", lambda n: _wilkinson(n) + 6.0 * np.eye(n), (21,))):
            for n in orders:
                for scale in _SCALES:
                    a = make(n) * scale
                    with self.subTest(f"{name} {n} × {scale:.0e}"):
                        try:
                            l = np.asarray(numerical.cholesky(a))
                        except numerical.LinAlgError:
                            # rounded to the subnormal numbers' few bits, an ill-conditioned one is definite no
                            # more (LAPACK refuses it too); anywhere else, a refusal is a defect
                            self.assertEqual(scale, 2.0 ** -1060, f"{name} {n}: refused")
                            continue
                        self.assertSmall(_norm(l @ l.T - a), _norm(a), n, f"{name}: cholesky")

    def test_generalized_eig_of_zero_and_singular_pencils(self):
        r = np.random.default_rng(5)
        other = r.standard_normal((6, 6)) + 6.0 * np.eye(6)
        for label, (a, b) in (("a zero", (np.zeros((6, 6)), other)), ("b zero", (other, np.zeros((6, 6)))),
                              ("b singular", (other, _rank(6, 2))), ("rank 1 pencil", (_rank(6, 1), _rank(6, 1)))):
            with self.subTest(label):
                w, vl, vr = (np.asarray(x) for x in numerical.eig(a, b, left=True))
                self.assertTrue(np.isfinite(vl).all() and np.isfinite(vr).all(), f"{label}: vectors not finite")
                self.assertTrue(np.allclose(np.linalg.norm(vr, axis=0), 1.0), f"{label}: unit vectors")


if __name__ == "__main__":
    unittest.main()
