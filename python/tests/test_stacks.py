"""Stacks of matrices, (..., m, n), as NumPy's linalg takes them: every matrix's answer is the one it has alone, to the
bit, whether the stack goes to the threads one matrix each (small matrices) or a matrix at a time with the threads
(past 128); the answers stack along the leading axes, solve's right-hand sides broadcast as NumPy 2 broadcasts them,
and an empty stack answers with the shapes a matrix's would have. Skipped when NumPy is not importable.
"""
import itertools
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


def per_matrix(function, stack):
    """`function` of each matrix of `stack` alone, as NumPy arrays nested along its leading axes."""
    stack = np.asarray(stack)
    answers = [function(stack[index]) for index in itertools.product(*map(range, stack.shape[:-2]))]
    if isinstance(answers[0], tuple):
        return tuple(np.array([np.asarray(part[k]) for part in answers]).reshape(stack.shape[:-2] + np.shape(answers[0][k]))
                     for k in range(len(answers[0])))
    return np.array([np.asarray(answer) for answer in answers]).reshape(stack.shape[:-2] + np.shape(answers[0]))


@unittest.skipIf(np is None, "NumPy is not importable")
class Stacks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

    def general(self, shape, dtype=np.float64):
        a = self.rng.standard_normal(shape)
        if np.dtype(dtype).kind == "c":
            a = a + 1j * self.rng.standard_normal(shape)
        return a.astype(dtype)

    def hermitian_positive(self, shape, dtype=np.float64):
        a = self.general(shape, dtype)
        return (a @ np.swapaxes(a.conj(), -1, -2) + shape[-1] * np.eye(shape[-1])).astype(dtype)

    def same(self, got, want):
        for g, w in zip(got if isinstance(got, tuple) else (got,), want if isinstance(want, tuple) else (want,)):
            self.assertEqual(np.asarray(g).shape, np.asarray(w).shape)
            self.assertTrue(np.array_equal(np.asarray(g), np.asarray(w)))

    def test_every_matrix_answers_as_it_does_alone(self):
        # 300 matrices of order 6 go to the threads together, and 8 of order 100 (each large enough to release the GIL
        # alone, which a worker never holds); 2 of order 140, one at a time with the threads.
        for shape, dtype in (((300, 6, 6), np.float64), ((3, 2, 5, 5), np.complex128), ((40, 7, 7), np.float32),
                             ((8, 100, 100), np.float64), ((2, 140, 140), np.float64)):
            with self.subTest(shape=shape, dtype=dtype):
                a = self.general(shape, dtype)
                h = self.hermitian_positive(shape, dtype)
                for name, function, operand in (("inv", numerical.inv, a), ("det", numerical.det, a),
                                                ("slogdet", numerical.slogdet, a), ("cholesky", numerical.cholesky, h),
                                                ("eigvalsh", numerical.eigvalsh, h), ("eigh", numerical.eigh, h),
                                                ("svd", numerical.svd, a), ("qr", numerical.qr, a),
                                                ("pinv", numerical.pinv, a),
                                                ("values", lambda m: numerical.svd(m, compute_uv=False), a)):
                    with self.subTest(name):
                        self.same(function(operand), per_matrix(function, operand))
                self.assertEqual(numerical.matrix_rank(a), per_matrix(numerical.matrix_rank, a).tolist())

    def test_the_answers_match_numpys(self):
        a = self.general((4, 3, 5, 5))
        h = self.hermitian_positive((4, 3, 5, 5))
        for got, want in ((numerical.inv(a), np.linalg.inv(a)), (numerical.det(a), np.linalg.det(a)),
                          (numerical.eigvalsh(h), np.linalg.eigvalsh(h)), (numerical.cholesky(h), np.linalg.cholesky(h)),
                          (numerical.svd(a, compute_uv=False), np.linalg.svd(a, compute_uv=False)),
                          (numerical.pinv(a), np.linalg.pinv(a))):
            self.assertTrue(np.allclose(np.asarray(got), want, rtol=1e-10, atol=1e-10))
        sign, logarithm = numerical.slogdet(a)
        self.assertTrue(np.allclose(np.asarray(logarithm), np.linalg.slogdet(a)[1], rtol=1e-12))
        self.assertTrue(np.array_equal(np.asarray(sign), np.linalg.slogdet(a)[0]))

    def test_solve_broadcasts_the_leading_axes(self):
        a = self.general((2, 3, 4, 4))
        cases = [np.ones(4),                       # a vector, against every matrix
                 self.general((4, 2)),             # one set of right-hand sides, for every matrix
                 self.general((3, 4, 2)),          # the last leading axis
                 self.general((2, 1, 4, 1)),       # a broadcast axis of 1
                 self.general((5, 2, 3, 4, 2))]    # more leading axes than a
        for b in cases:
            with self.subTest(b=b.shape):
                got = np.asarray(numerical.solve(a, b))
                want = np.linalg.solve(a, b)
                self.assertEqual(got.shape, want.shape)
                self.assertTrue(np.allclose(got, want, rtol=1e-10, atol=1e-10))
        alone = np.asarray(numerical.solve(a[1, 2], np.ones(4)))
        self.assertTrue(np.array_equal(np.asarray(numerical.solve(a, np.ones(4)))[1, 2], alone))
        self.assertTrue(np.allclose(np.asarray(numerical.solve(a[0, 0], self.general((6, 4, 3)))).shape, (6, 4, 3)))
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve(a, self.general((4, 4, 2)))  # 4 against 3: no broadcast

    def test_an_empty_stack_answers_with_the_shapes_of_a_matrix(self):
        for shape in ((0, 3, 3), (2, 0, 3, 3)):
            a = np.zeros(shape)
            self.assertEqual(numerical.inv(a).shape, shape)
            self.assertEqual(numerical.det(a).shape, shape[:-2])
            self.assertEqual(numerical.eigvalsh(a).shape, shape[:-1])
            self.assertEqual([part.shape for part in numerical.svd(np.zeros(shape[:-2] + (3, 5)))],
                             [shape[:-2] + (3, 3), shape[:-2] + (3,), shape[:-2] + (5, 5)])
            self.assertEqual(numerical.solve(a, np.zeros(shape[:-1] + (2,))).shape, shape[:-1] + (2,))

    def test_one_failing_matrix_fails_the_stack_and_a_rank_below_two_is_refused(self):
        a = self.general((5, 3, 3))
        a[3] = 0.0  # singular
        with self.assertRaises(numerical.LinAlgError):
            numerical.inv(a)
        b = self.general((4, 3, 3))
        b[1, 0, 0] = np.nan
        with self.assertRaises(numerical.LinAlgError):
            numerical.svd(b)
        with self.assertRaises(numerical.LinAlgError):
            numerical.inv(np.ones(3))

    def test_an_answer_does_not_depend_on_what_ran_before(self):
        # The same matrices in two orders, other allocations between: the same bits. GCC's default fusion of
        # multiply-adds (-ffp-contract=fast, which the build replaces by on) rounded an element otherwise in a loop's
        # vectorised version than in its scalar one, the one run chosen by the operands' addresses: on AVX-512, one
        # eigvalsh in a hundred and fifty changed bits with the heap's history.
        h = self.hermitian_positive((300, 6, 6))
        first = [np.asarray(numerical.eigvalsh(m)) for m in h]
        clutter = [numerical.asarray(np.ones(k)) for k in range(1, 40)]
        again = {i: np.asarray(numerical.eigvalsh(h[i])) for i in itertools.chain(range(0, 300, 7), range(300))}
        del clutter
        stacked = np.asarray(numerical.eigvalsh(h))
        for i in range(300):
            self.assertTrue(np.array_equal(first[i], again[i]) and np.array_equal(first[i], stacked[i]), f"matrix {i}")

    def test_the_threads_give_the_same_bits_as_one(self):
        a = self.general((500, 8, 8))
        many = np.asarray(numerical.inv(a))
        numerical.set_num_threads(1)
        try:
            one = np.asarray(numerical.inv(a))
        finally:
            numerical.set_num_threads(0)
        self.assertTrue(np.array_equal(many, one))


if __name__ == "__main__":
    unittest.main()
