"""NumPy's assembly and triangles against NumPy's own: concatenate (and concat), stack, vstack, hstack over shapes,
axes and every pair of dtypes (the result's NumPy's promotion), tril and triu over offsets and stacks, diag of vectors
and matrices, kron over ranks; bool tensors kept bool; the refusals. Deterministic; skipped when NumPy is not
importable (CI installs it)."""
import itertools
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None

DTYPES = ["float64", "float32", "complex128", "bool"]


@unittest.skipIf(np is None, "NumPy not installed")
class Assembly(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1030)

    def array(self, shape, dtype):
        x = self.rng.standard_normal(shape)
        if dtype == "complex128":
            x = x + 1j * self.rng.standard_normal(shape)
        return x > 0 if dtype == "bool" else x.astype(dtype)

    def same(self, got, expected):
        got = np.asarray(got)
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(str(got.dtype), str(expected.dtype))
        self.assertTrue(np.array_equal(got, expected))

    def test_concatenate_and_stack_as_numpy(self):
        cases = [([(2, 3), (4, 3)], 0), ([(2, 3), (2, 1)], 1), ([(2, 3), (2, 2)], -1), ([(3,), (0,), (2,)], 0),
                 ([(2, 2, 2), (2, 1, 2)], 1), ([(2, 3), (1, 3)], None)]
        for (shapes, axis), (first, second) in itertools.product(cases, itertools.product(DTYPES, DTYPES)):
            parts = [self.array(shapes[0], first)] + [self.array(s, second) for s in shapes[1:]]
            with self.subTest(shapes=shapes, axis=axis, dtypes=(first, second)):
                tensors = [numerical.asarray(p, "bool" if p.dtype == bool else None) for p in parts]
                self.same(numerical.concatenate(tensors, axis=axis), np.concatenate(parts, axis=axis))
        self.assertIs(numerical.concat, numerical.concatenate)
        for (shape, axis), dtype in itertools.product([((3,), 0), ((3,), 1), ((3,), -1), ((2, 3), 2), ((), 0)], DTYPES):
            parts = [self.array(shape, dtype) for _ in range(3)]
            with self.subTest(shape=shape, axis=axis, dtype=dtype):
                tensors = [numerical.asarray(p, dtype) for p in parts]
                self.same(numerical.stack(tensors, axis=axis), np.stack(parts, axis=axis))

    def test_vstack_and_hstack_as_numpy(self):
        for parts in ([(3,), (3,)], [(2, 3), (3,)], [(), ()]):
            arrays = [self.array(s, "float64") for s in parts]
            self.same(numerical.vstack(arrays), np.vstack(arrays))
        for parts in ([(3,), (2,)], [(2, 3), (2, 1)], [(), (2,)]):
            arrays = [self.array(s, "float64") for s in parts]
            self.same(numerical.hstack(arrays), np.hstack(arrays))

    def test_triangles_and_diagonals_as_numpy(self):
        for shape, dtype in itertools.product([(4, 4), (3, 5), (5, 3), (2, 3, 4), (4,), (0, 3)], DTYPES):
            x = self.array(shape, dtype)
            t = numerical.asarray(x, dtype)
            for k in (-2, -1, 0, 1, 3, 9, -9):
                with self.subTest(shape=shape, dtype=dtype, k=k):
                    self.same(numerical.tril(t, k), np.tril(x, k))
                    self.same(numerical.triu(t, k), np.triu(x, k))
        for dtype in DTYPES:
            for n, k in itertools.product((0, 1, 4), (-2, 0, 1, 3)):
                v = self.array((n,), dtype)
                with self.subTest(dtype=dtype, n=n, k=k):
                    self.same(numerical.diag(numerical.asarray(v, dtype), k), np.diag(v, k))
            m = self.array((3, 5), dtype)
            for k in (-4, -1, 0, 2, 7):
                with self.subTest(dtype=dtype, k=k):
                    self.same(numerical.diag(numerical.asarray(m, dtype), k), np.diag(m, k))

    def test_kron_as_numpy(self):
        for (sa, sb), (da, db) in itertools.product([((2, 2), (3, 3)), ((2,), (3,)), ((2, 3), (4,)), ((3,), (2, 2)),
                                                     ((2, 1, 2), (1, 3)), ((), (2, 2))],
                                                    [("float64", "float64"), ("float32", "complex128"), ("float64", "float32")]):
            a, b = self.array(sa, da), self.array(sb, db)
            with self.subTest(sa=sa, sb=sb, dtypes=(da, db)):
                self.same(numerical.kron(a, b), np.kron(a, b))

    def test_refusals(self):
        for call in (lambda: numerical.concatenate([]), lambda: numerical.concatenate([np.ones(()), np.ones(())]),
                     lambda: numerical.concatenate([np.ones((2, 2)), np.ones((3, 3))]),
                     lambda: numerical.concatenate([np.ones((2, 2))], axis=2),
                     lambda: numerical.concatenate([np.ones(2), np.ones((2, 2))]),
                     lambda: numerical.stack([np.ones(2), np.ones(3)]), lambda: numerical.stack([np.ones(2)], axis=3),
                     lambda: numerical.diag(np.ones((2, 2, 2))), lambda: numerical.tril(np.float64(1.0))):
            with self.assertRaises(ValueError):
                call()
        with self.assertRaises(TypeError):
            numerical.concatenate(numerical.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
