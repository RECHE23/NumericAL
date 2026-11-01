"""scipy.linalg's arguments where its names are NumPy's too: eigh(a, b) positional, lower, eigvals_only and driver;
cholesky's lower; qr's "economic" and "full"; svd's lapack_driver; lstsq's cond; pinv's atol and return_rank; the
overwrite and check_finite flags accepted. Each against SciPy's own answer. Deterministic; skipped without SciPy."""
import unittest

import numerical

try:
    import numpy as np
    import scipy.linalg as sl
except ImportError:
    np = sl = None


@unittest.skipIf(sl is None, "SciPy not installed")
class ScipyArguments(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261101)
        m = self.rng.standard_normal((6, 6))
        self.a = m + m.T
        k = self.rng.standard_normal((6, 6))
        self.b = k @ k.T + 6 * np.eye(6)

    def test_eigh_as_scipy(self):
        a, b = self.a, self.b
        w, v = numerical.eigh(a, b)  # SciPy's b in NumPy's UPLO place
        np.testing.assert_allclose(np.asarray(w), sl.eigh(a, b)[0], rtol=1e-12)
        np.testing.assert_allclose(np.asarray(v).T @ b @ np.asarray(v), np.eye(6), atol=1e-12)
        upper = np.triu(a) + np.tril(self.rng.standard_normal((6, 6)), -1)  # only the upper triangle meaningful
        np.testing.assert_allclose(np.asarray(numerical.eigh(upper, lower=False)[0]), sl.eigh(upper, lower=False)[0], rtol=1e-12)
        np.testing.assert_allclose(np.asarray(numerical.eigvalsh(upper, lower=False)), sl.eigvalsh(upper, lower=False), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(numerical.eigh(a, eigvals_only=True)), sl.eigh(a, eigvals_only=True), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(numerical.eigh(a, b, eigvals_only=True)), sl.eigh(a, b, eigvals_only=True), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(numerical.eigh(a, b, eigvals_only=True, subset_by_index=[0, 2])),
                                   sl.eigh(a, b, eigvals_only=True, subset_by_index=[0, 2]), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(numerical.eigh(a, eigvals_only=True, subset_by_index=[1, 3])),
                                   sl.eigh(a, eigvals_only=True, subset_by_index=[1, 3]), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(numerical.eigvalsh(a, b)), sl.eigvalsh(a, b), rtol=1e-12)
        for driver in ("ev", "evd", "evr", "evx"):
            with self.subTest(driver=driver):
                np.testing.assert_allclose(np.asarray(numerical.eigh(a, driver=driver, overwrite_a=True, check_finite=False)[0]),
                                           sl.eigh(a, driver=driver)[0], rtol=1e-12)
                np.testing.assert_allclose(np.asarray(numerical.eigvalsh(a, driver=driver)), sl.eigvalsh(a, driver=driver), rtol=1e-12)
        with self.assertRaises(ValueError):
            numerical.eigh(a, b, type=2)
        with self.assertRaises(ValueError):
            numerical.eigvalsh(a, driver="other")
        with self.assertRaises(TypeError):
            numerical.eigh(a, b, b=b)

    def test_factorisations_as_scipy(self):
        a, b = self.a, self.b
        np.testing.assert_allclose(np.asarray(numerical.cholesky(b, lower=False)), sl.cholesky(b), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(numerical.cholesky(b, lower=True, overwrite_a=True, check_finite=False)),
                                   sl.cholesky(b, lower=True), rtol=1e-12)
        tall = self.rng.standard_normal((7, 4))
        for mode, shapes in (("economic", ((7, 4), (4, 4))), ("full", ((7, 7), (7, 4)))):
            q, r = numerical.qr(tall, mode=mode, overwrite_a=True, lwork=None, check_finite=False)
            self.assertEqual((q.shape, r.shape), shapes)
            np.testing.assert_allclose(np.asarray(q) @ np.asarray(r), tall, atol=1e-13)
        np.testing.assert_allclose(np.asarray(numerical.inv(b, overwrite_a=True, check_finite=False)), sl.inv(b), rtol=1e-12)
        self.assertAlmostEqual(numerical.det(b, overwrite_a=True, check_finite=False) / sl.det(b), 1.0, places=12)
        for driver in ("gesdd", "gesvd"):
            with self.subTest(driver=driver):
                np.testing.assert_allclose(np.asarray(numerical.svd(tall, lapack_driver=driver, overwrite_a=True, check_finite=False)[1]),
                                           sl.svd(tall, lapack_driver=driver)[1], rtol=1e-12)
        with self.assertRaises(ValueError):
            numerical.svd(tall, lapack_driver="gesvx")
        del a

    def test_least_squares_as_scipy(self):
        a = self.rng.standard_normal((8, 5))
        a[:, 4] = a[:, 0] + a[:, 1]  # rank 4
        y = self.rng.standard_normal(8)
        for driver in ("gelsd", "gelsy", "gelss", None):
            with self.subTest(driver=driver):
                x, _, rank, _ = numerical.lstsq(a, y, cond=1e-10, lapack_driver=driver, overwrite_a=True, overwrite_b=True,
                                                check_finite=False)
                expected = sl.lstsq(a, y, cond=1e-10, lapack_driver=driver or "gelsd")
                np.testing.assert_allclose(np.asarray(x), expected[0], atol=1e-12)
                self.assertEqual(rank, expected[2])
        with self.assertRaises(ValueError):
            numerical.lstsq(a, y, 1e-10, cond=1e-10)
        with self.assertRaises(ValueError):
            numerical.lstsq(a, y, lapack_driver="gels")
        for kwargs in ({"atol": 1e-8}, {"atol": 0.5}, {"atol": 1e-8, "rtol": 0.3}, {}):
            with self.subTest(**kwargs):
                got, rank = numerical.pinv(a, return_rank=True, check_finite=False, **kwargs)
                expected, expected_rank = sl.pinv(a, return_rank=True, **kwargs)
                np.testing.assert_allclose(np.asarray(got), expected, atol=1e-12)
                self.assertEqual(rank, expected_rank)
                np.testing.assert_allclose(np.asarray(numerical.pinv(a, **kwargs)), expected, atol=1e-12)
        h = a @ a.T
        np.testing.assert_allclose(np.asarray(numerical.pinv(h, hermitian=True, atol=1e-8)), sl.pinvh(h, atol=1e-8), atol=1e-12)
        zero, rank = numerical.pinv(np.zeros((3, 2)), return_rank=True)
        self.assertEqual((zero.shape, rank), ((2, 3), 0))
        with self.assertRaises(numerical.LinAlgError):
            numerical.pinv(np.ones((2, 3, 3)), atol=1.0)


if __name__ == "__main__":
    unittest.main()
