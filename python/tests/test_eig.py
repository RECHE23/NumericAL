"""eig and eigvals of a general square matrix as numpy.linalg gives them: the same eigenvalues (as a multiset: their
order is the Schur form's), eigenvectors with A V = V Λ, unit norm and the largest component real, real answers where
the matrix and all its eigenvalues are, on stacks and in the four dtypes; and the matrices that strain an
eigensolver: defective, triangular either way, graded, companion, zero. Complex answers whatever the eigenvalues, as
NumPy 2.5's (NumPy before it answers real where it can: the values are compared, not the dtypes). Skipped when NumPy is not importable.
"""
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy is not importable")
class Eig(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)

    def check(self, a, tolerance=1e-12):
        w, v = (np.asarray(part) for part in numerical.eig(a))
        want = np.linalg.eigvals(a)
        self.assertEqual((w.dtype, v.dtype), (np.complex128, np.complex128))
        scale = max(1.0, float(np.abs(a).max())) if a.size else 1.0
        self.assertTrue(np.allclose(np.sort_complex(w.ravel()), np.sort_complex(want.ravel()), rtol=0,
                                    atol=tolerance * scale * max(1, a.shape[-1])), f"{w} against {want}")
        self.assertTrue(np.allclose(a @ v, v * w[..., None, :], rtol=0, atol=tolerance * scale * max(1, a.shape[-1])))
        if a.size:
            self.assertTrue(np.allclose(np.linalg.norm(v, axis=-2), 1.0, atol=1e-13))
            largest = np.take_along_axis(v, np.argmax(np.abs(v), axis=-2)[..., None, :], axis=-2)
            self.assertTrue(np.all(np.imag(largest) == 0))
        self.assertTrue(np.array_equal(np.asarray(numerical.eigvals(a)), w))  # the same Schur form either way
        return w, v

    def test_random_matrices_real_and_complex(self):
        for n in (1, 2, 3, 5, 8, 17, 40, 90):
            self.check(self.rng.standard_normal((n, n)))
            self.check(self.rng.standard_normal((n, n)) + 1j * self.rng.standard_normal((n, n)))

    def test_complex_answers_and_conjugate_pairs(self):
        w, v = self.check(np.diag([3.0, 1.0, 2.0]) + np.triu(np.ones((3, 3)), 1))
        self.assertTrue(np.all(w.imag == 0) and np.all(v.imag == 0))  # real eigenpairs, held as complex
        w, v = self.check(np.array([[0.0, -1.0], [1.0, 0.0]]))
        self.assertEqual(sorted(w.tolist(), key=lambda z: z.imag), [-1j, 1j])
        self.assertEqual(w[0], np.conj(w[1]))  # a pair, the positive imaginary part first
        self.assertTrue(np.array_equal(v[:, 0], np.conj(v[:, 1])))

    def test_stacks_answer_as_each_matrix_alone(self):
        stack = self.rng.standard_normal((2, 3, 6, 6))
        w, v = self.check(stack)
        alone_w, alone_v = (np.asarray(part) for part in numerical.eig(stack[1, 2]))
        self.assertTrue(np.array_equal(w[1, 2], alone_w))
        self.assertTrue(np.array_equal(v[1, 2], alone_v))
        empty_w, empty_v = numerical.eig(np.zeros((0, 4, 4)))
        self.assertEqual((empty_w.shape, empty_v.shape), ((0, 4), (0, 4, 4)))

    def test_single_precision_keeps_its_dtype(self):
        a = self.rng.standard_normal((5, 5))
        w, v = numerical.eig(a.astype(np.float32))
        self.assertEqual((w.dtype, v.dtype), ("complex64", "complex64"))
        self.assertTrue(np.allclose(np.sort_complex(np.asarray(w)), np.sort_complex(np.linalg.eigvals(a)), atol=1e-5))
        w, v = numerical.eig((a + 1j * a.T).astype(np.complex64))
        self.assertEqual((w.dtype, v.dtype), ("complex64", "complex64"))

    def test_matrices_that_strain_an_eigensolver(self):
        self.check(np.triu(self.rng.standard_normal((7, 7))))  # isolated by the balancing: exact
        lower = np.tril(self.rng.standard_normal((7, 7)))
        w, _ = self.check(lower)
        self.assertTrue(np.all(w.imag == 0))
        self.assertEqual(sorted(w.real.tolist()), sorted(np.diag(lower).tolist()))  # exact: isolated by the balancing
        self.check(np.zeros((4, 4)))
        jordan = np.eye(12) + np.diag(np.ones(11), 1)  # defective: one eigenvector, twelve times
        w, v = numerical.eig(jordan)
        self.assertTrue(np.all(np.isfinite(np.asarray(v))))
        companion = np.zeros((12, 12))
        companion[1:, :-1] = np.eye(11)
        companion[:, -1] = self.rng.standard_normal(12)
        self.check(companion)
        graded = self.rng.standard_normal((9, 9)) * np.logspace(-9, 9, 9)[:, None] / np.logspace(-9, 9, 9)[None, :]
        self.check(graded, 1e-10)
        tiny = self.rng.standard_normal((6, 6)) * 1e-300
        w, _ = numerical.eig(tiny)
        self.assertTrue(np.allclose(np.sort_complex(np.asarray(w) * 1e300), np.sort_complex(np.linalg.eigvals(tiny * 1e300))))
        huge = self.rng.standard_normal((6, 6)) * 1e300
        w, _ = numerical.eig(huge)
        self.assertTrue(np.all(np.isfinite(np.asarray(w))))

    def test_rejections(self):
        with self.assertRaises(numerical.LinAlgError):
            numerical.eig(np.ones((2, 3)))
        with self.assertRaises(numerical.LinAlgError):
            numerical.eigvals(np.ones(3))
        with self.assertRaises(numerical.LinAlgError):
            numerical.eig(np.array([[1.0, np.nan], [0.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
