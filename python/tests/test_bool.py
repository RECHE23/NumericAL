"""The bool dtype against NumPy's: comparisons, the logical operators, where, isnan/isinf/isfinite, any/all and the
reductions of a mask, over every pair of a set of shapes that broadcast (and pairs that do not), with NaN, signed
zeros and infinities among the values; its conversions and its exchange with NumPy; and the arithmetic and linear
algebra it is kept out of. Deterministic; skipped when NumPy is not importable (CI installs it)."""
import itertools
import operator
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None

SHAPES = [(), (1,), (3,), (2, 1), (1, 3), (2, 3), (4, 1, 3), (0,), (2, 0)]
COMPARISONS = [operator.lt, operator.le, operator.eq, operator.ne, operator.gt, operator.ge]
SPECIAL = [0.0, -0.0, 1.0, -1.0, np.nan if np else 0.0, np.inf if np else 0.0, -np.inf if np else 0.0]


@unittest.skipIf(np is None, "NumPy not installed")
class Comparisons(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261030)

    def values(self, shape, dtype=None):
        """Elements drawn from SPECIAL and a few small integers, so that pairs often tie."""
        pool = np.array(SPECIAL + [2.0, 3.0])
        return self.rng.choice(pool, size=shape).astype(np.float64 if dtype is None else dtype)

    def check(self, got, expected):
        self.assertIsInstance(got, numerical.Tensor)
        self.assertEqual(got.dtype, "bool")
        self.assertEqual(got.shape, expected.shape)
        self.assertTrue(np.array_equal(np.asarray(got), expected))

    def test_every_pair_of_shapes_compares_as_numpy(self):
        for (sa, sb), compare in itertools.product(itertools.product(SHAPES, SHAPES), COMPARISONS):
            x, y = self.values(sa), self.values(sb)
            try:
                expected = compare(x, y)
            except ValueError:
                with self.subTest(sa=sa, sb=sb, op=compare.__name__), self.assertRaises(ValueError):
                    compare(numerical.asarray(x), numerical.asarray(y))
                continue
            with self.subTest(sa=sa, sb=sb, op=compare.__name__):
                self.check(compare(numerical.asarray(x), numerical.asarray(y)), np.asarray(expected))

    def test_numbers_and_array_likes_compare_on_either_side(self):
        x = self.values((2, 3))
        t = numerical.asarray(x)
        for compare, other in itertools.product(COMPARISONS, (1.0, 2, -0.0, np.nan, [1.0, 2.0, 3.0], x[0])):
            with self.subTest(op=compare.__name__, other=other):
                self.check(compare(t, other), np.asarray(compare(x, other)))
                if isinstance(other, np.ndarray):  # NumPy's operator answers first, reading the Tensor, as for + and *
                    self.assertTrue(np.array_equal(compare(other, t), compare(other, x)))
                else:
                    self.check(compare(other, t), np.asarray(compare(other, x)))

    def test_single_precision_compares_in_its_own_precision(self):
        x = self.values((5,), np.float32)
        t = numerical.asarray(x)
        for compare in COMPARISONS:
            with self.subTest(op=compare.__name__):
                self.check(compare(t, 0.1), np.asarray(compare(x, np.float32(0.1))))
                self.check(compare(t, numerical.asarray(x.astype(np.float64))), compare(x.astype(np.float64), x))

    def test_complex_tensors_compare_for_equality_only(self):
        x = self.values((2, 3)) + 1j * self.values((2, 3))
        y = x.copy()
        y[0, 0] = 5.0
        a, b = numerical.asarray(x), numerical.asarray(y)
        self.check(a == b, x == y)
        self.check(a != b, x != y)
        self.check(a == 1.0, x == 1.0)
        for compare in (operator.lt, operator.le, operator.gt, operator.ge):
            with self.subTest(op=compare.__name__), self.assertRaisesRegex(TypeError, "no order"):
                compare(a, b)

    def test_bool_tensors_compare_as_numpy(self):
        x = self.rng.integers(0, 2, (3, 4)).astype(bool)
        y = self.rng.integers(0, 2, (4,)).astype(bool)
        for compare in COMPARISONS:
            with self.subTest(op=compare.__name__):
                self.check(compare(numerical.asarray(x, "bool"), numerical.asarray(y, "bool")), compare(x, y))
                self.check(compare(numerical.asarray(x, "bool"), True), compare(x, True))

    def test_something_unreadable_is_neither_equal_nor_ordered(self):
        t = numerical.ones(2)
        self.assertFalse(t == None)  # noqa: E711  (Python's own answer once both sides decline)
        self.assertTrue(t != "a")
        with self.assertRaises(TypeError):
            t < None  # noqa: B015

    def test_the_named_comparisons_are_the_operators(self):
        x, y = self.values((2, 3)), self.values((3,))
        for name, compare in zip(("less", "less_equal", "equal", "not_equal", "greater", "greater_equal"), COMPARISONS):
            with self.subTest(name=name):
                self.check(getattr(numerical, name)(x, y), compare(x, y))
                self.check(getattr(numerical, name)(x.tolist(), 2.0), compare(x, 2.0))


@unittest.skipIf(np is None, "NumPy not installed")
class Logical(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261031)

    def mask(self, shape):
        return self.rng.integers(0, 2, shape).astype(bool)

    def check(self, got, expected):
        self.assertEqual(got.dtype, "bool")
        self.assertTrue(np.array_equal(np.asarray(got), np.asarray(expected)))

    def test_and_or_xor_broadcast_as_numpy(self):
        for (sa, sb), op in itertools.product(itertools.product(SHAPES, SHAPES), (operator.and_, operator.or_, operator.xor)):
            x, y = self.mask(sa), self.mask(sb)
            a, b = numerical.asarray(x, "bool"), numerical.asarray(y, "bool")
            try:
                expected = op(x, y)
            except ValueError:
                with self.subTest(sa=sa, sb=sb, op=op.__name__), self.assertRaises(ValueError):
                    op(a, b)
                continue
            with self.subTest(sa=sa, sb=sb, op=op.__name__):
                self.check(op(a, b), expected)
                self.check(op(a, y), expected)  # a NumPy bool array
                self.check(op(x, b), expected)

    def test_bools_and_lists_of_bools_take_part(self):
        x = self.mask((2, 3))
        a = numerical.asarray(x, "bool")
        for op in (operator.and_, operator.or_, operator.xor):
            with self.subTest(op=op.__name__):
                self.check(op(a, True), op(x, True))
                self.check(op(False, a), op(False, x))
                self.check(op(a, [True, False, True]), op(x, [True, False, True]))
        self.check(~a, ~x)

    def test_numbers_have_no_bitwise_operators(self):
        a = numerical.asarray(self.mask((3,)), "bool")
        f = numerical.ones(3)
        for op in (operator.and_, operator.or_, operator.xor):
            for x, y in ((a, f), (f, a), (a, 1.0), (a, [1.0, 0.0, 1.0])):
                with self.subTest(op=op.__name__, x=type(x).__name__, y=type(y).__name__), self.assertRaises(TypeError):
                    op(x, y)
        with self.assertRaisesRegex(TypeError, "bool tensors, not float64"):
            ~f

    def test_the_logical_functions_read_any_number_as_its_truth(self):
        x = np.array([[0.0, 1.0, np.nan], [-0.0, 2.0, 0.0]])
        y = np.array([1.0, 0.0, 0.0])
        for name in ("logical_and", "logical_or", "logical_xor"):
            with self.subTest(name=name):
                self.check(getattr(numerical, name)(x, y), getattr(np, name)(x, y))
        self.check(numerical.logical_not(x), np.logical_not(x))
        self.check(numerical.logical_not(x + 1j * y), np.logical_not(x + 1j * y))


@unittest.skipIf(np is None, "NumPy not installed")
class Where(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261101)

    def test_every_triple_of_shapes_broadcasts_as_numpy(self):
        for sc, sx, sy in itertools.product(SHAPES[:7], repeat=3):
            c = self.rng.integers(0, 2, sc).astype(bool)
            x, y = self.rng.standard_normal(sx), self.rng.standard_normal(sy)
            try:
                expected = np.where(c, x, y)
            except ValueError:
                with self.subTest(sc=sc, sx=sx, sy=sy), self.assertRaises(ValueError):
                    numerical.where(numerical.asarray(c, "bool"), x, y)
                continue
            with self.subTest(sc=sc, sx=sx, sy=sy):
                got = numerical.where(numerical.asarray(c, "bool"), x, y)
                self.assertEqual(got.shape, expected.shape)
                self.assertTrue(np.array_equal(np.asarray(got), expected))

    def test_the_branches_promote_as_arithmetic(self):
        c = np.array([True, False, True])
        x32 = np.array([1.5, -0.0, 3.0], np.float32)
        for x, y, dtype in ((x32, 0.25, "float32"), (x32, np.zeros(3), "float64"), (x32, 1j, "complex64"),
                            (np.ones(3), np.zeros(3, complex), "complex128")):
            with self.subTest(x=x.dtype, y=type(y).__name__):
                got = numerical.where(c, x, y)
                self.assertEqual(got.dtype, dtype)
                self.assertTrue(np.array_equal(np.asarray(got), np.where(c, x, y).astype(dtype)))
        masks = numerical.where(c, numerical.asarray([True] * 3, "bool"), numerical.zeros(3, "bool"))
        self.assertEqual(masks.dtype, "bool")
        self.assertEqual(masks.tolist(), [True, False, True])

    def test_a_number_condition_is_its_truth(self):
        c = np.array([[0.0, np.nan], [-0.0, 2.0]])
        got = numerical.where(c, 1.0, -1.0)
        self.assertTrue(np.array_equal(np.asarray(got), np.where(c, 1.0, -1.0)))
        self.assertEqual(numerical.where(numerical.asarray(c) > 0, numerical.asarray(c), 0.0).tolist(), [[0.0, 0.0], [0.0, 2.0]])


@unittest.skipIf(np is None, "NumPy not installed")
class Classification(unittest.TestCase):
    def test_every_special_value_and_dtype_classifies_as_numpy(self):
        real = np.array(SPECIAL)
        cplx = np.array([complex(a, b) for a, b in itertools.product(SPECIAL, repeat=2)])
        for name, x in itertools.product(("isnan", "isinf", "isfinite"), (real, real.astype(np.float32), cplx,
                                                                         cplx.astype(np.complex64), real > 0)):
            with self.subTest(name=name, dtype=str(x.dtype)):
                source = numerical.asarray(x, "bool") if x.dtype == bool else x
                got = getattr(numerical, name)(source)
                self.assertEqual(got.dtype, "bool")
                self.assertTrue(np.array_equal(np.asarray(got), getattr(np, name)(x)))


@unittest.skipIf(np is None, "NumPy not installed")
class Reductions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261102)

    def test_any_and_all_reduce_as_numpy(self):
        for shape in [(), (3,), (2, 3), (4, 1, 3), (0,), (2, 0), (3, 0, 2)]:
            x = self.rng.integers(0, 2, shape).astype(bool)
            numbers = np.where(x, self.rng.choice([np.nan, -1.0, 2.0], shape), 0.0)
            axes = [None, *range(-len(shape), len(shape)), *itertools.combinations(range(len(shape)), 2)]
            for name, axis, keepdims, source in itertools.product(("any", "all"), axes, (False, True), (x, numbers)):
                expected = getattr(np, name)(source, axis=axis, keepdims=keepdims)
                operand = numerical.asarray(source, "bool") if source.dtype == bool else source
                with self.subTest(name=name, shape=shape, axis=axis, keepdims=keepdims, dtype=str(source.dtype)):
                    got = getattr(numerical, name)(operand, axis=axis, keepdims=keepdims)
                    every = axis is None or (isinstance(axis, tuple) and len(axis) == len(shape))
                    if every and not keepdims:  # a number, as sum's; along some axes a Tensor, of rank 0 at least
                        self.assertIs(got, bool(expected))
                    else:
                        self.assertEqual(got.dtype, "bool")
                        self.assertEqual(got.shape, np.shape(expected))
                        self.assertTrue(np.array_equal(np.asarray(got), expected))
        mask = numerical.asarray([[True, False]], "bool")
        self.assertIs(mask.any(), True)
        self.assertEqual(mask.all(axis=0).tolist(), [True, False])
        empty = numerical.zeros((2, 0), "bool")
        for name, axis in itertools.product(("max", "min"), (None, 1)):
            with self.subTest(name=name, axis=axis), self.assertRaisesRegex(ValueError, "zero-size"):
                getattr(numerical, name)(empty, axis=axis)
        self.assertEqual(numerical.max(empty, axis=0).shape, (0,))

    def test_sum_mean_max_and_min_of_a_mask_as_numpy(self):
        x = self.rng.integers(0, 2, (3, 4)).astype(bool)
        mask = numerical.asarray(x, "bool")
        for name, axis in itertools.product(("sum", "mean", "max", "min"), (None, 0, 1, (0, 1))):
            expected = getattr(np, name)(x, axis=axis)
            with self.subTest(name=name, axis=axis):
                got = getattr(numerical, name)(mask, axis=axis)
                if axis in (0, 1):  # along one of the two axes; every axis gives a number
                    self.assertTrue(np.array_equal(np.asarray(got), expected))
                    self.assertEqual(got.dtype, "bool" if name in ("max", "min") else "float64")
                else:
                    self.assertEqual(got, expected)
                    self.assertIsInstance(got, bool if name in ("max", "min") else float)


@unittest.skipIf(np is None, "NumPy not installed")
class Conversions(unittest.TestCase):
    def test_astype_converts_as_numpy(self):
        values = np.array([0.0, -0.0, 1.5, np.nan, np.inf, -2.0])
        for source, dtype in itertools.product((values, values.astype(np.float32), values + 1j * values[::-1]),
                                               ("bool", "float64", "complex128", "float32", "complex64")):
            if np.iscomplexobj(source) and dtype in ("float64", "float32"):
                with self.subTest(source=str(source.dtype), dtype=dtype), self.assertRaises(numerical.error):
                    numerical.astype(source, dtype)
                continue
            with self.subTest(source=str(source.dtype), dtype=dtype):
                got = numerical.asarray(source).astype(dtype)
                self.assertEqual(got.dtype, dtype)
                self.assertTrue(np.array_equal(np.asarray(got), source.astype(dtype), equal_nan=dtype != "bool"))
                back = got.astype(str(source.dtype)) if dtype == "bool" else None
                if back is not None:
                    self.assertTrue(np.array_equal(np.asarray(back), source.astype(bool).astype(source.dtype)))
        self.assertEqual(numerical.astype([1.0], np.bool_).dtype, "bool")
        mask = numerical.asarray([True], "bool")
        self.assertIs(mask.astype("bool"), mask)

    def test_construction_of_a_bool_tensor(self):
        self.assertEqual(numerical.Tensor([1, 0, 2.5, 0j], dtype="bool").tolist(), [True, False, True, False])
        self.assertEqual(numerical.Tensor([True, False, True, True], (2, 2), "bool").tolist(), [[True, False], [True, True]])
        self.assertEqual(numerical.zeros((2,), "bool").tolist(), [False, False])
        self.assertEqual(numerical.ones(2, dtype="bool").tolist(), [True, True])
        self.assertEqual(numerical.full((1, 2), True).dtype, "bool")
        self.assertEqual(numerical.full(2, 3j, "bool").tolist(), [True, True])
        self.assertEqual(numerical.eye(3, "bool").tolist(), np.eye(3, dtype=bool).tolist())
        self.assertEqual(numerical.asarray([1.0, 0.0], "bool").tolist(), [True, False])
        self.assertEqual(numerical.asarray(2.0, "bool").tolist(), True)
        self.assertEqual(repr(numerical.asarray([True, False], "bool")), "numerical.Tensor([True, False], dtype=bool)")

    def test_numpy_reads_a_mask_where_it_lies_and_hands_one_back(self):
        x = np.array([[True, False, True], [False, False, True]])
        mask = numerical.asarray(x, "bool")
        self.assertEqual(mask.dtype, "bool")
        for view in (np.asarray(mask), np.from_dlpack(mask)):
            self.assertEqual(view.dtype, np.bool_)
            self.assertTrue(np.array_equal(view, x))
        self.assertEqual(mask.__array_interface__["typestr"], "|b1")
        strided = x[:, ::-1].T  # neither C nor Fortran order
        self.assertTrue(np.array_equal(np.asarray(numerical.asarray(strided, "bool")), strided))
        self.assertEqual(numerical.inv(np.eye(2, dtype=bool)).tolist(), [[1.0, 0.0], [0.0, 1.0]])

    def test_asarray_keeps_bools_bool_as_numpy(self):
        x = np.array([[True, False], [False, True]])
        for source in (x, x.tolist(), True, np.True_, [True], ((False,),), x[:, ::-1].T):
            with self.subTest(source=source):
                got = numerical.asarray(source)
                self.assertEqual(got.dtype, "bool")
                self.assertTrue(np.array_equal(np.asarray(got), np.asarray(source)))
        for source, dtype in (([], "float64"), ([True, 1], "float64"), ([True, 2.5], "float64"), ([True, 1j], "complex128"),
                              (1, "float64"), (np.int8([1]), "float64")):
            with self.subTest(source=source):
                self.assertEqual(numerical.asarray(source).dtype, dtype)
                self.assertEqual(numerical.asarray(source).dtype, str(np.asarray(source).dtype)
                                 if np.asarray(source).dtype.kind not in "iu" else "float64")

    def test_a_bool_array_like_is_computed_on_as_float64(self):
        # A function that computes reads a bool array-like as float64, as NumPy's arithmetic and linear algebra do;
        # a bool Tensor has no arithmetic and is refused there.
        x = np.array([[True, False], [True, True]])
        for name, call in (("inv", lambda a: numerical.inv(a)), ("solve", lambda a: numerical.solve(a, [1.0, 2.0])),
                           ("det", lambda a: numerical.det(a)), ("exp", lambda a: numerical.exp(a)),
                           ("norm", lambda a: numerical.norm(a)), ("matmul", lambda a: numerical.matmul(a, a)),
                           ("fft", lambda a: numerical.fft.fft(a)), ("kron", lambda a: numerical.kron(a, [[2.0]])),
                           ("sum", lambda a: numerical.sum(a)), ("expm", lambda a: numerical.expm(a))):
            with self.subTest(name=name):
                expected = np.asarray(call(x.astype(float)))
                for source in (x, x.tolist()):
                    self.assertTrue(np.allclose(np.asarray(call(source)), expected))
        with self.assertRaisesRegex(TypeError, "bool tensor has no arithmetic"):
            numerical.inv(numerical.asarray(x))

    def test_a_bool_array_like_is_moved_as_bool(self):
        # A function that moves elements keeps a bool array-like bool, as NumPy's.
        x = np.array([[True, False, True], [False, False, True]])
        for name, call, expected in (("concatenate", lambda a: numerical.concatenate([a, a]), np.concatenate([x, x])),
                                     ("stack", lambda a: numerical.stack([a, a]), np.stack([x, x])),
                                     ("tril", lambda a: numerical.tril(a), np.tril(x)),
                                     ("triu", lambda a: numerical.triu(a, 1), np.triu(x, 1)),
                                     ("transpose", lambda a: numerical.transpose(a), x.T),
                                     ("reshape", lambda a: numerical.reshape(a, (3, 2)), x.reshape(3, 2)),
                                     ("squeeze", lambda a: numerical.squeeze(a[None]), x),
                                     ("diagonal", lambda a: numerical.diagonal(a), np.diagonal(x))):
            with self.subTest(name=name):
                got = call(x)
                self.assertEqual(got.dtype, "bool")
                self.assertTrue(np.array_equal(np.asarray(got), expected))

    def test_a_mask_is_indexed_reshaped_and_transposed_as_any_tensor(self):
        x = np.array([[[True, False], [False, False]], [[True, True], [False, True]]])
        mask = numerical.asarray(x, "bool")
        self.assertIs(mask[1, 0, 1], True)
        self.assertEqual(mask[:, 1].tolist(), x[:, 1].tolist())
        self.assertEqual(mask[..., None].shape, (2, 2, 2, 1))
        self.assertEqual(mask.reshape(4, 2).tolist(), x.reshape(4, 2).tolist())
        self.assertEqual(mask.T.tolist(), x.T.tolist())
        self.assertEqual([row.tolist() for row in mask], [row.tolist() for row in x])
        self.assertTrue(numerical.array_equal(mask, x))
        self.assertFalse(numerical.array_equal(mask, ~x))
        self.assertIs(bool(numerical.asarray([True], "bool")), True)

    def test_a_masks_diagonal_conjugate_and_trace_as_numpy(self):
        x = np.array([[[True, False, True], [False, True, False]], [[False, False, True], [True, True, True]]])
        mask = numerical.asarray(x, "bool")
        for offset in (-2, -1, 0, 1, 2, 5):
            with self.subTest(offset=offset):
                got = numerical.diagonal(mask, offset=offset)
                self.assertEqual(got.dtype, "bool")
                self.assertEqual(got.tolist(), np.linalg.diagonal(x, offset=offset).tolist())
                self.assertEqual(mask.diagonal(offset, 1, 2).tolist(), x.diagonal(offset, 1, 2).tolist())
                trace = numerical.trace(mask, offset=offset)  # the trues counted, as a float64 (sum's convention)
                self.assertEqual(trace.dtype, "float64")
                self.assertEqual(trace.tolist(), np.linalg.trace(x, offset=offset).astype(float).tolist())
        self.assertIs(numerical.conj(mask), mask)
        self.assertEqual(mask.conj().tolist(), x.tolist())


@unittest.skipIf(np is None, "NumPy not installed")
class KeptOutOfTheArithmetic(unittest.TestCase):
    def test_a_mask_beside_a_number_takes_its_dtype(self):
        x = np.array([True, False, True])
        mask = numerical.asarray(x, "bool")
        f32 = np.array([1.5, np.inf, -2.0], np.float32)
        for other in (2.5, 3, 1j, np.array([1.0, -0.0, np.nan]), f32):
            for op in (operator.add, operator.mul, operator.truediv):
                expected = op(x, other)
                with self.subTest(other=other, op=op.__name__):
                    got = op(mask, other)
                    # NumPy's integer (int64, or int32 under a 32-bit CPython or NumPy 1 on Windows) is float64 here
                    self.assertEqual(got.dtype, str(expected.dtype) if expected.dtype.kind not in "iu" else "float64")
                    self.assertTrue(np.array_equal(np.asarray(got), expected, equal_nan=True))

    def test_two_masks_and_the_functions_refuse(self):
        mask = numerical.asarray([[True, False], [False, True]], "bool")
        refused = [lambda: mask + mask, lambda: mask * True, lambda: -mask, lambda: abs(mask), lambda: mask @ mask,
                   lambda: numerical.exp(mask), lambda: numerical.inv(mask), lambda: numerical.solve(mask, mask),
                   lambda: numerical.einsum("ij->", mask), lambda: numerical.det(mask), lambda: numerical.fft.fft(mask),
                   lambda: numerical.norm(mask), lambda: numerical.eigh(mask), lambda: numerical.expm(mask)]
        for i, call in enumerate(refused):
            with self.subTest(i=i), self.assertRaisesRegex(TypeError, "bool tensor has no arithmetic"):
                call()


if __name__ == "__main__":
    unittest.main()
