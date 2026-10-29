"""NumericAL Python binding tests (stdlib unittest, no third-party requirement).

The NumPy / DLPack interop is verified twice: once with pure-Python checks that
need nothing installed (the __array_interface__ dict, the DLPack capsule and
device), and once against NumPy proper when it happens to be importable.
"""
import os
import platform
import subprocess
import sys
import unittest

import numerical

try:
    import numpy as np
except ImportError:  # NumPy is optional; the interop round-trips are then skipped
    np = None


class TensorBasics(unittest.TestCase):
    def test_construct_and_introspect(self):
        t = numerical.Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(t.ndim, 2)
        self.assertEqual(t.size, 6)
        self.assertEqual(t.dtype, "float64")
        self.assertEqual(t.tolist(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_default_shape_is_one_dimensional(self):
        v = numerical.Tensor([1, 2, 3])
        self.assertEqual(v.shape, (3,))
        self.assertEqual(v.tolist(), [1.0, 2.0, 3.0])

    def test_complex_dtype(self):
        t = numerical.Tensor([1 + 2j, 3 - 1j], dtype="complex128")
        self.assertEqual(t.dtype, "complex128")
        self.assertEqual(t.tolist(), [1 + 2j, 3 - 1j])

    def test_rank_zero_tolist_is_a_scalar(self):
        s = numerical.einsum("i,i->", numerical.Tensor([1, 2, 3]), numerical.Tensor([1, 2, 3]))
        self.assertEqual(s.ndim, 0)
        self.assertEqual(s.shape, ())
        self.assertEqual(s.tolist(), 14.0)

    def test_repr(self):
        self.assertEqual(repr(numerical.Tensor([1, 2])), "numerical.Tensor([1.0, 2.0], dtype=float64)")
        self.assertEqual(repr(numerical.asarray([[1.0, 2.0], [3.0, 4.0]])),
                         "numerical.Tensor([[1.0, 2.0],\n                  [3.0, 4.0]], dtype=float64)")
        self.assertEqual(repr(numerical.asarray(1j)), "numerical.Tensor(1j, dtype=complex128)")
        # Past 1000 elements each long axis keeps three entries at each end.
        self.assertEqual(repr(numerical.arange(1001)),
                         "numerical.Tensor([0.0, 1.0, 2.0, ..., 998.0, 999.0, 1000.0], dtype=float64)")
        self.assertEqual(repr(numerical.zeros((1000, 2))).count("..."), 1)

    def test_bad_dtype_raises(self):
        with self.assertRaises(numerical.error):
            numerical.Tensor([1, 2], dtype="float16")

    def test_shape_mismatch_raises(self):
        with self.assertRaises(numerical.error):
            numerical.Tensor([1, 2, 3], shape=(2, 2))


class Einsum(unittest.TestCase):
    def test_matrix_product(self):
        a = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        b = numerical.Tensor([5, 6, 7, 8], shape=(2, 2))
        self.assertEqual(numerical.einsum("ij,jk->ik", a, b).tolist(),
                         [[19.0, 22.0], [43.0, 50.0]])

    def test_transpose_and_trace(self):
        m = numerical.Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
        self.assertEqual(numerical.einsum("ij->ji", m).shape, (3, 2))
        sq = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        self.assertEqual(numerical.einsum("ii->", sq).tolist(), 5.0)

    def test_three_operands(self):
        a = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        i = numerical.Tensor([1, 0, 0, 1], shape=(2, 2))
        self.assertEqual(numerical.einsum("ij,jk,kl->il", a, i, i).tolist(),
                         [[1.0, 2.0], [3.0, 4.0]])

    def test_complex_operands(self):
        a = numerical.Tensor([1j, 1], dtype="complex128")
        out = numerical.einsum("i,i->", a, a)
        self.assertEqual(out.tolist(), 0j)  # i*i + 1*1 = -1 + 1

    def test_too_few_arguments_raises(self):
        with self.assertRaises(numerical.error):
            numerical.einsum("i->i")

    def test_mixed_dtype_promotes_to_complex(self):
        out = numerical.einsum("i,i->", numerical.Tensor([2.0]), numerical.Tensor([1j], dtype="complex128"))
        self.assertEqual(out.dtype, "complex128")
        self.assertEqual(out.tolist(), 2j)

    def test_bad_spec_raises(self):
        for spec in ("i1", "ijk", "ij->k"):  # a digit, a rank the operand lacks, a label no input has
            with self.subTest(spec=spec), self.assertRaises(numerical.error):
                numerical.einsum(spec, numerical.Tensor([1, 2, 3, 4], shape=(2, 2)))


@unittest.skipIf(np is None, "NumPy is the reference")
class HighRanks(unittest.TestCase):
    """Specs built in Python past 26 axes: the 52 labels einsum takes, the matrices' own labels apart from the
    batch's (a batch reaching them read a diagonal, or refused), and a clear refusal past 52."""

    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_stacks_whose_batch_reaches_the_matrices_labels(self):
        stack = self.rng.standard_normal((1,) * 22 + (2,) + (2, 2))
        np.testing.assert_allclose(np.asarray(numerical.matrix_power(numerical.asarray(stack), 2)),
                                   np.linalg.matrix_power(stack, 2), rtol=1e-12)
        deep = self.rng.standard_normal((1,) * 25 + (3,) + (3, 3))
        np.testing.assert_allclose(np.asarray(numerical.trace(numerical.asarray(deep))),
                                   np.trace(deep, axis1=-2, axis2=-1), rtol=1e-12)

    def test_transposes_and_tensordots_past_26_axes(self):
        values = self.rng.standard_normal((1,) * 26 + (2,))
        self.assertEqual(numerical.asarray(values).T.shape, values.T.shape)
        np.testing.assert_array_equal(np.asarray(numerical.asarray(values).T), values.T)
        p = self.rng.standard_normal((1,) * 14 + (2,))
        q = self.rng.standard_normal((2,) + (1,) * 13)
        np.testing.assert_allclose(np.asarray(numerical.tensordot(numerical.asarray(p), numerical.asarray(q), axes=([14], [0]))),
                                   np.tensordot(p, q, axes=([14], [0])), rtol=1e-12)
        with self.assertRaises(ValueError):
            _ = numerical.zeros((1,) * 53).T

    def test_cross_keeps_each_component_apart(self):
        # Copied side by side, where products with unit vectors spread an infinite component as NaN into the others.
        with np.errstate(all="ignore"):
            for a, b in (([np.inf, 1, 0], [0, 1, 0]), ([0, 0, 1], [np.inf, 0, 0]), ([1e308, 1e308, 0], [1e308, -1e308, 0])):
                a, b = np.array(a, float), np.array(b, float)
                np.testing.assert_array_equal(np.asarray(numerical.cross(numerical.asarray(a), numerical.asarray(b))), np.cross(a, b))

    def test_outer_keeps_each_product_s_sign(self):
        x, y = np.array([-0.0, 2.0, np.inf]), np.array([1.0, -0.0, 3.0])
        got = np.asarray(numerical.outer(numerical.asarray(x), numerical.asarray(y)))
        with np.errstate(all="ignore"):
            want = np.outer(x, y)
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(np.signbit(got), np.signbit(want))

    def test_complex_products_and_quotients_are_numpys_to_the_bit(self):
        # NumPy's loops, fused where NumPy's are on this processor: every pair of a grid of special parts (zeros of
        # either sign, infinities, a NaN, values near overflow, a subnormal) and drawn pairs, in both complex types.
        values = [0.0, -0.0, np.inf, -np.inf, np.nan, 1e308, -1e308, 1.0, -2.5, 1e-310]
        grid = [complex(r, i) for r in values for i in values]
        rng = np.random.default_rng(17)
        drawn = rng.standard_normal((500, 4)) * np.exp(rng.uniform(-5, 5, (500, 4)))
        left = [x for x in grid for _ in grid] + list(drawn[:, 0] + 1j * drawn[:, 1])
        right = [y for _ in grid for y in grid] + list(drawn[:, 2] + 1j * drawn[:, 3])
        for dtype in (np.complex128, np.complex64):
            with np.errstate(all="ignore"):
                a, b = np.array(left).astype(dtype), np.array(right).astype(dtype)
                for want, got in ((a * b, numerical.asarray(a) * numerical.asarray(b)),
                                  (a / b, numerical.asarray(a) / numerical.asarray(b))):
                    got = np.asarray(got)
                    for x, y in ((want.real, got.real), (want.imag, got.imag)):
                        same = (np.isnan(x) & np.isnan(y)) | ((x == y) & (np.signbit(x) == np.signbit(y)))
                        self.assertTrue(np.all(same), (np.dtype(dtype).name, int(np.sum(~same))))

    def test_one_element_products_take_numpys_loop(self):
        # One element from an operand of rank above 0 broadcast to a higher rank is NumPy's scalar loop's, which x86-64
        # does not fuse; from equal ranks or a rank-0 operand, its vector loop's. (1 + e)(1 - e) - 1 is -e² fused.
        for dtype in (np.complex128, np.complex64):
            e = np.finfo(dtype).eps * 4
            x, y = complex(1 + e, 1), complex(1 - e, 1)
            for left, right in (((1, 1), (1,)), ((1,), (1, 1)), ((1, 1, 1), (1,)), ((1, 1), ()), ((1, 1), (1, 1)),
                                ((1,), (1,)), ((1, 1), (3,))):
                a = np.full(left, x, dtype=dtype)
                b = np.full(right, y, dtype=dtype)
                got = np.asarray(numerical.asarray(a) * numerical.asarray(b))
                np.testing.assert_array_equal(got, a * b, (np.dtype(dtype).name, left, right))

    def test_products_follow_numpys_loops_as_measured(self):
        # Fused or rounded as the NumPy beside it fuses, which the binding measures: forced either way, then measured.
        e = 2.0**-30
        a = numerical.asarray(np.array([complex(1 + e, 1)] * 2))
        b = numerical.asarray(np.array([complex(1 - e, 1)] * 2))
        try:
            for fused, real in ((False, 0.0), (True, -(e * e))):
                numerical._numerical._set_numpy_loops(fused, fused, fused)
                self.assertEqual(np.asarray(a * b)[0].real, real)
        finally:
            numerical._numpy_loops()
        np.testing.assert_array_equal(np.asarray(a * b), np.asarray(a) * np.asarray(b))

    def test_complex_powers_are_numpys_to_the_bit(self):
        # ** 2 and ** -1 of a Python integer are NumPy's square and reciprocal, ** 0.5 its square root, any other
        # integer exponent below 100 its products (npy_cpow): on a grid of special parts and drawn values.
        values = [0.0, -0.0, np.inf, -np.inf, np.nan, 1e308, -1e308, 1.0, -2.5, 1e-310, 1 + 2.0**-30]
        rng = np.random.default_rng(23)
        drawn = rng.standard_normal((200, 2)) * np.exp(rng.uniform(-5, 5, (200, 2)))
        z = np.array([complex(r, i) for r in values for i in values] + list(drawn[:, 0] + 1j * drawn[:, 1]))
        for dtype in (np.complex128, np.complex64):
            with np.errstate(all="ignore"):
                a = z.astype(dtype)
                t = numerical.asarray(a)
                for exponent in (2, -1, 2.0, -1.0, 3, -2, 5, -7, 0, 1):
                    want, got = a**exponent, np.asarray(t**exponent)
                    for x, y in ((want.real, got.real), (want.imag, got.imag)):
                        same = (np.isnan(x) & np.isnan(y)) | ((x == y) & (np.signbit(x) == np.signbit(y)))
                        self.assertTrue(np.all(same), (np.dtype(dtype).name, exponent, int(np.sum(~same))))

    def test_complex_square_roots_are_glibcs(self):
        # NumPy's under glibc to the bit (its csqrt); within two units in the last place of the others' (macOS's
        # libm drops an imaginary part far smaller than the real one, npymath on Windows halves a scaled one).
        values = [0.0, -0.0, np.inf, -np.inf, np.nan, 1e308, -1e308, 1.0, -1.0, -1e20, 1e-310, 3.0]
        z = np.array([complex(r, i) for r in values for i in values])
        for dtype in (np.complex128, np.complex64):
            with np.errstate(all="ignore"):
                a = z.astype(dtype)
                for got in (np.asarray(numerical.sqrt(numerical.asarray(a))), np.asarray(numerical.asarray(a) ** 0.5)):
                    self.assertEqual(complex(got[values.index(-1.0) * len(values) + values.index(0.0)]), 1j)
                    near = complex(got[values.index(-1e20) * len(values) + values.index(1.0)])
                    self.assertLess(abs(near.real - 5e-11), 1e-15)
                    if platform.libc_ver()[0] == "glibc":  # musl's csqrt is its own
                        want = np.sqrt(a)
                        for x, y in ((want.real, got.real), (want.imag, got.imag)):
                            same = (np.isnan(x) & np.isnan(y)) | ((x == y) & (np.signbit(x) == np.signbit(y)))
                            self.assertTrue(np.all(same), (np.dtype(dtype).name, int(np.sum(~same))))

    def test_complex_elementary_functions_are_glibcs(self):
        # NumPy's under glibc to the bit (its cexp, clog, csin, ccos, ctan, ctanh); elsewhere NumPy takes the platform's
        # library, which chooses otherwise among infinities and NaNs and strays further near the unit circle: the
        # finite answers agree within a few dozen units in the last place of the larger part.
        values = [0.0, -0.0, np.inf, -np.inf, np.nan, 1e308, -1e308, 1.0, -1.0, 1e-310, 710.0, -745.0, 40.0, 1e-20]
        rng = np.random.default_rng(29)
        drawn = rng.standard_normal((300, 2)) * np.exp(rng.uniform(-6, 6, (300, 2)))
        z = np.array([complex(r, i) for r in values for i in values] + list(drawn[:, 0] + 1j * drawn[:, 1]))
        glibc = platform.libc_ver()[0] == "glibc"
        for dtype in (np.complex128, np.complex64):
            a = z.astype(dtype)
            eps = np.finfo(dtype).eps
            for name in ("exp", "log", "sin", "cos", "tan", "tanh"):
                with np.errstate(all="ignore"):
                    want = getattr(np, name)(a)
                    got = np.asarray(getattr(numerical, name)(numerical.asarray(a)))
                if glibc:
                    for x, y in ((want.real, got.real), (want.imag, got.imag)):
                        same = (np.isnan(x) & np.isnan(y)) | ((x == y) & (np.signbit(x) == np.signbit(y)))
                        self.assertTrue(np.all(same), (np.dtype(dtype).name, name, int(np.sum(~same))))
                    continue
                finite = np.isfinite(want) & np.isfinite(got)
                with np.errstate(all="ignore"):
                    scale = np.maximum(np.abs(want.real), np.abs(want.imag))
                    off = np.maximum(np.abs(want.real - got.real), np.abs(want.imag - got.imag))
                self.assertTrue(np.all(off[finite] <= 64 * eps * scale[finite]), (np.dtype(dtype).name, name))
        # the losses of libc++'s forms, whatever the platform
        tangent = complex(numerical.tan(numerical.asarray(np.array([1e308 + 0j])))[0])
        self.assertLess(abs(tangent.real - np.tan(1e308)), 4e-16)  # NaN under libc++, from 1e308 + 0j
        self.assertEqual(tangent.imag, 0.0)
        self.assertEqual(complex(numerical.sin(numerical.asarray(np.array([1e308j])))[0]).real, 0.0)
        self.assertGreater(complex(numerical.log(numerical.asarray(np.array([1 + 1e-20j])))[0]).real, 0.0)

    def test_reductions_over_several_axes_and_kept_dims(self):
        # a tuple of axes reduced together (one pass over them moved last), () none, keepdims each an extent of 1
        b = np.random.default_rng(31).standard_normal((3, 4, 5))
        t = numerical.asarray(b)
        for name in ("sum", "mean", "max", "min"):
            for axis in (None, 1, (0, 2), (2, 0), (0, 1, 2), (), -1):
                for keepdims in (False, True):
                    want = getattr(np, name)(b, axis=axis, keepdims=keepdims)
                    got = np.asarray(getattr(numerical, name)(t, axis=axis, keepdims=keepdims))
                    self.assertEqual(got.shape, np.shape(want), (name, axis, keepdims))
                    np.testing.assert_allclose(got, want, rtol=1e-12, err_msg=f"{name} {axis} {keepdims}")
            self.assertEqual(np.asarray(getattr(t, name)(axis=(0, 1), keepdims=True)).shape, (1, 1, 5))
            with self.assertRaises(ValueError):  # a repeated axis, as NumPy's
                getattr(numerical, name)(t, axis=(0, 0))

    def test_numpys_reductions_take_a_tensor(self):
        # numpy.sum(t) and the like call the Tensor's method with `out` (and `dtype` for a mean), as for any object
        a = np.random.default_rng(41).standard_normal((3, 4))
        t = numerical.asarray(a)
        for f in (np.sum, np.mean, np.max, np.min):
            self.assertAlmostEqual(f(t), f(a), places=12)
            np.testing.assert_allclose(np.asarray(f(t, axis=1, keepdims=True)), f(a, axis=1, keepdims=True), rtol=1e-12)
        self.assertEqual(np.asarray(np.mean(t, axis=0, dtype=np.complex128)).dtype, np.complex128)
        with self.assertRaises(TypeError):
            np.sum(t, out=np.zeros(()))

    def test_ellipsis_and_new_axes_index_as_numpy(self):
        a = np.arange(24.0).reshape(2, 3, 4)
        t = numerical.asarray(a)
        for key in (Ellipsis, (Ellipsis, 1), (0, Ellipsis), (None, 0), (0, None, 1, None), (Ellipsis, None), (0, 1, 2, None),
                    (None, Ellipsis, 1), (slice(1, 2), None, Ellipsis)):
            np.testing.assert_array_equal(np.asarray(t[key]), a[key])
        for key in ((Ellipsis, Ellipsis), (0, 1, 2, 3, Ellipsis)):
            with self.assertRaises(IndexError):
                t[key]

    def test_diagonals_transposes_and_tensor_systems_as_numpy(self):
        rng = np.random.default_rng(37)
        for x in (rng.standard_normal((4, 4)), rng.standard_normal((2, 3, 5)), rng.standard_normal((3, 5)) + 1j):
            for offset in (0, 2, -1, 9):
                np.testing.assert_array_equal(np.asarray(numerical.diagonal(numerical.asarray(x), offset=offset)),
                                              np.linalg.diagonal(x, offset=offset))
            np.testing.assert_array_equal(np.asarray(numerical.matrix_transpose(numerical.asarray(x))), np.linalg.matrix_transpose(x))
        a = rng.standard_normal((6, 2, 3)) + np.eye(6).reshape(6, 2, 3)
        b = rng.standard_normal(6)
        np.testing.assert_allclose(np.asarray(numerical.tensorsolve(numerical.asarray(a), numerical.asarray(b))),
                                   np.linalg.tensorsolve(a, b), rtol=1e-9)
        c = np.eye(24).reshape(4, 6, 8, 3) * 3 + rng.standard_normal((4, 6, 8, 3))
        np.testing.assert_allclose(np.asarray(numerical.tensorinv(numerical.asarray(c))), np.linalg.tensorinv(c), rtol=1e-8)
        with self.assertRaises(numerical.LinAlgError):
            numerical.tensorinv(numerical.asarray(rng.standard_normal((2, 3, 4))), ind=1)
        with self.assertRaises(ValueError):
            numerical.tensorinv(numerical.asarray(c), ind=0)
        with self.assertRaises(ValueError):
            numerical.matrix_transpose(numerical.asarray(b))

    def test_means_and_norms_where_numpy_says_nothing_or_nan(self):
        # A complex mean divided as NumPy divides, by n + 0j: a NaN part makes both NaN, to NumPy's bits; the smallest
        # magnitude over no element refused, as NumPy's minimum; a result empty along a non-empty axis allowed.
        a = np.array([[complex(np.nan, 1.0), 1 + 2j, 3 - 1j], [0.5 + 0.25j, -1 + 0j, 2 + 2j]])
        for dtype in (np.complex128, np.complex64):
            for axis in (None, 0, 1):
                with np.errstate(all="ignore"):
                    want = np.mean(a.astype(dtype), axis=axis)
                got = np.asarray(numerical.mean(numerical.asarray(a.astype(dtype)), axis=axis))
                np.testing.assert_array_equal(got, want)
        for shape, axis in (((0,), None), ((2, 0), 1), ((0, 3), 0)):
            with self.assertRaises(ValueError):
                numerical.vector_norm(numerical.zeros(shape), ord=-np.inf, axis=axis)
        self.assertEqual(numerical.vector_norm(numerical.zeros((0, 3)), ord=-np.inf, axis=1).shape, (0,))
        with_nan = np.array([[0.0, np.nan, 2.0], [-0.0, 0.0, np.inf]])
        for axis in (None, 0, 1):  # ord=0 counts a NaN, which is not zero; every other order is NaN where one is
            for order in (0.0, 1.0, 2.0, np.inf):
                with np.errstate(all="ignore"):
                    want = np.linalg.vector_norm(with_nan, ord=order, axis=axis)
                got = np.asarray(numerical.vector_norm(numerical.asarray(with_nan), ord=order, axis=axis))
                np.testing.assert_array_equal(got, want)

    def test_one_dimensional_cross_and_complex_trace_keep_numpys_values(self):
        # A cross of two vectors in their own dtype (a part read as a Python number computed in double); a trace of
        # complex numbers with an infinite part (a complex 1 times each element made 0 · inf a NaN).
        x, y = np.array([1.1, -2.3, 0.7], np.float32), np.array([0.3, 4.1, -1.9], np.float32)
        got = numerical.cross(numerical.asarray(x), numerical.asarray(y))
        self.assertEqual(got.dtype, "float32")
        np.testing.assert_array_equal(np.asarray(got), np.cross(x, y))
        m = np.array([[complex(np.inf, 1.0), 0j], [0j, -0.0 - 1j]])
        np.testing.assert_array_equal(np.asarray(numerical.trace(numerical.asarray(m))), np.trace(m))

    def test_cross_takes_one_axis(self):
        with self.assertRaises(TypeError):
            numerical.cross(numerical.asarray(np.ones((3, 3))), numerical.asarray(np.ones((3, 3))), axis=(0, 1))


class HugeEmptyShapes(unittest.TestCase):
    """A tensor of no elements along one axis, any number along the others: a sum keeping extents whose product
    overflows refuses (it wrote past a buffer of the wrapped size), one keeping an empty axis is empty."""

    def test_sums_and_norms(self):
        big = 2 ** ((sys.maxsize.bit_length() + 1) // 2)  # its square past std::size_t: 2**32, or 2**16 on 32 bits
        t = numerical.zeros((big, 0, big))
        with self.assertRaises(numerical.error):
            numerical.sum(t, axis=1)
        with self.assertRaises(numerical.error):
            numerical.vector_norm(t, axis=1)
        self.assertEqual(numerical.sum(t, axis=0).shape, (0, big))
        self.assertEqual(numerical.zeros((big // 2, 0)).shape, (big // 2, 0))  # 2**31: past a 32-bit long
        with self.assertRaises(OverflowError):  # an extent past Py_ssize_t
            numerical.zeros((sys.maxsize + 1, 0))

    def test_empty_matrices_decompose_as_numpys(self):
        # svd of 0 × 0 read the largest of no band of rows; each factor has NumPy's shape, thin or full
        for shape in ((0, 0), (0, 3), (3, 0)):
            for full in (True, False):
                got = numerical.svd(numerical.zeros(shape), full_matrices=full)
                want = np.linalg.svd(np.zeros(shape), full_matrices=full)
                self.assertEqual([np.asarray(x).shape for x in got], [x.shape for x in want])
            with self.assertRaises(numerical.LinAlgError):  # as NumPy's: cond is not defined on empty arrays
                numerical.cond(numerical.zeros(shape))
        self.assertEqual(np.asarray(numerical.cond(numerical.zeros((0, 2, 2)))).shape, (0,))


class Solve(unittest.TestCase):
    def test_solves_a_linear_system(self):
        a = numerical.Tensor([2, 1, 1, 1, 3, 1, 1, 1, 4], shape=(3, 3))
        b = numerical.Tensor([7, 10, 15])
        x = numerical.solve(a, b)
        for got, want in zip(x.tolist(), [1.0, 2.0, 3.0]):
            self.assertAlmostEqual(got, want)

    def test_rank_requirements(self):
        with self.assertRaises(numerical.error):
            numerical.solve(numerical.Tensor([1, 2, 3]), numerical.Tensor([1, 2, 3]))

    def test_singular_raises(self):
        a = numerical.Tensor([1, 2, 2, 4], shape=(2, 2))
        with self.assertRaises(numerical.error):
            numerical.solve(a, numerical.Tensor([1, 2]))


class NonFinite(unittest.TestCase):
    """Every decomposition refuses an infinite or NaN entry where it starts, rather than stop at once
    and return plausible values (eigh of [[1, nan], [nan, 2]] used to read as 1 and 2)."""

    def test_every_decomposition_refuses_a_non_finite_entry(self):
        rhs = numerical.Tensor([1.0, 1.0], [2])
        calls = {
            "eigh": lambda m: numerical.eigh(m),
            "svd": lambda m: numerical.svd(m),
            "qr": lambda m: numerical.qr(m),
            "cholesky": lambda m: numerical.cholesky(m),
            "inverse": lambda m: numerical.inverse(m),
            "determinant": lambda m: numerical.determinant(m),
            "solve": lambda m: numerical.solve(m, rhs),
            "lstsq": lambda m: numerical.lstsq(m, rhs),
        }
        for bad in (float("inf"), float("-inf"), float("nan")):
            for data in ([1.0, bad, bad, 2.0], [bad, 0.0, 0.0, 1.0]):
                for name, call in calls.items():
                    with self.subTest(name=name, data=data):
                        with self.assertRaises(numerical.error) as caught:
                            call(numerical.Tensor(data, [2, 2]))
                        self.assertIn("infinite or NaN", str(caught.exception))


class ArrayInterface(unittest.TestCase):
    def test_dict_describes_the_buffer(self):
        t = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        ai = t.__array_interface__
        self.assertEqual(ai["shape"], (2, 2))
        self.assertEqual(ai["version"], 3)
        self.assertTrue(ai["typestr"].endswith("f8"))
        self.assertIsInstance(ai["data"][0], int)
        self.assertTrue(ai["data"][1])  # read-only

    def test_complex_typestr(self):
        t = numerical.Tensor([1j], dtype="complex128")
        self.assertTrue(t.__array_interface__["typestr"].endswith("c16"))

    def test_asarray_of_a_tensor_returns_it(self):
        t = numerical.Tensor([1, 2, 3])
        self.assertIs(numerical.asarray(t), t)

    def test_asarray_rejects_plain_objects(self):
        # A TypeError, as NumPy raises for an object it cannot read, and still a numerical.error.
        for obj in (object(), "abc", None, {"a": 1}):
            with self.subTest(obj=obj):
                with self.assertRaises(TypeError) as caught:
                    numerical.asarray(obj)
                self.assertIsInstance(caught.exception, numerical.error)
                self.assertIn(f"a {type(obj).__name__} is not a tensor", str(caught.exception))
        with self.assertRaises(numerical.error):  # the native reader still refuses one directly
            numerical._numerical.asarray(object())

    def test_asarray_widens_real_to_complex(self):
        t = numerical.asarray(numerical.Tensor([1.0, 2.0]), dtype="complex128")
        self.assertEqual(t.dtype, "complex128")

    def test_asarray_refuses_complex_to_float64(self):
        with self.assertRaises(numerical.error):
            numerical.asarray(numerical.Tensor([1j, 2j], dtype="complex128"), dtype="float64")

    def test_asarray_matching_dtype_returns_same_object(self):
        t = numerical.Tensor([1.0, 2.0])
        self.assertIs(numerical.asarray(t, dtype="float64"), t)

    def test_a_tensor_refuses_a_second_init(self):
        # Views of the buffer (np.asarray, __dlpack__) outlive the call that made them; a second
        # __init__ would free their memory, so it is refused and the tensor keeps its value.
        t = numerical.Tensor([1.0, 2.0])
        interface = t.__array_interface__
        with self.assertRaises(TypeError):
            t.__init__([3.0])
        self.assertEqual(t.tolist(), [1.0, 2.0])
        self.assertEqual(t.__array_interface__["data"], interface["data"])

    def test_asarray_reads_either_byte_order(self):
        # A typestr names its byte order: a big-endian `>f8` source is byte-swapped on read, not
        # copied as native doubles; an unknown order is refused, not guessed.
        import ctypes
        import struct

        class Interface:
            def __init__(self, typestr, raw, count):
                self.buffer = ctypes.create_string_buffer(raw, len(raw))
                self.__array_interface__ = {"version": 3, "typestr": typestr, "shape": (count,),
                                            "data": (ctypes.addressof(self.buffer), True)}

        values = [1.5, -2.25, 1e300]
        for order in "<>":
            source = Interface(order + "f8", struct.pack(order + "3d", *values), 3)
            with self.subTest(order=order):
                self.assertEqual(numerical.asarray(source).tolist(), values)
        pairs = struct.pack(">4d", 1.0, -2.0, 0.5, 3.0)
        self.assertEqual(numerical.asarray(Interface(">c16", pairs, 2)).tolist(), [1 - 2j, 0.5 + 3j])
        with self.assertRaises(numerical.error):
            numerical.asarray(Interface("|f8", struct.pack("<d", 1.0), 1))
        # A size too large to convert (it aborted the interpreter) is no known type either.
        with self.assertRaises(numerical.error):
            numerical.asarray(Interface("<f99999999999999999999999", struct.pack("<d", 1.0), 1))

    def test_asarray_refuses_a_null_address_and_an_empty_data_tuple(self):
        # A null data address with elements to read crashed the interpreter; an empty array may have one.
        class Raw:
            def __init__(self, typestr, data, count):
                self.__array_interface__ = {"version": 3, "typestr": typestr, "shape": (count,), "data": data}

        for typestr in ("<f8", "<c16"):
            with self.subTest(typestr=typestr):
                with self.assertRaises(numerical.error):
                    numerical.asarray(Raw(typestr, (0, True), 4))
                self.assertEqual(numerical.asarray(Raw(typestr, (0, True), 0)).tolist(), [])
        with self.assertRaises(numerical.error):
            numerical.asarray(Raw("<f8", (), 1))
        # Rows of no elements at a null address: nothing is read.
        empty_rows = Raw("<f8", (0, True), 3)
        empty_rows.__array_interface__["shape"] = (3, 0)
        self.assertEqual(numerical.asarray(empty_rows).shape, (3, 0))

    def test_asarray_reads_nested_lists_and_numbers(self):
        self.assertEqual(numerical.asarray([[1, 2], [3, 4]]).shape, (2, 2))
        self.assertEqual(numerical.asarray([[1, 2], [3, 4]]).tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(numerical.asarray((True, 2.5)).tolist(), [1.0, 2.5])
        self.assertEqual(numerical.asarray([1 + 2j, 3]).tolist(), [1 + 2j, 3 + 0j])
        self.assertEqual(numerical.asarray([1, 2], dtype="complex128").tolist(), [1 + 0j, 2 + 0j])
        scalar = numerical.asarray(5)
        self.assertEqual((scalar.shape, scalar.tolist()), ((), 5.0))
        self.assertEqual(numerical.asarray([]).shape, (0,))
        with self.assertRaises(ValueError):
            numerical.asarray([[1, 2], [3]])
        with self.assertRaises(TypeError):
            numerical.asarray([1, "2"])
        with self.assertRaises(numerical.error):
            numerical.asarray([1j], dtype="float64")

    def test_a_nested_list_reads_as_the_python_reading_gives(self):
        # The C walk of floats, ints and complex numbers against asarray's Python reading of the same values; what the
        # walk leaves to that reading (another kind of number, a ragged list, an int past a double) reads or fails as it did.
        import fractions

        class Real(float):
            pass

        cases = [[1, 2, 3], [[1.5, 2], [3, 4]], [], [[], []], [[1, 2j], [3, 4]], ((1, 2), (3, 4)), [True, False],
                 [[[1]]], [2 ** 70, -0.0], [Real(2.5), 1], [complex(1, -0.0), 2], [fractions.Fraction(1, 3)]]
        for case in cases:
            for dtype in (None, "float64", "float32", "complex128", "complex64"):
                with self.subTest(case=case, dtype=dtype):
                    values, shape = numerical._nested(case)
                    slow = numerical._NestedBuffer(values, shape)
                    try:
                        want = numerical._asarray(slow, dtype) if dtype else numerical._asarray(slow)
                    except numerical.error:
                        with self.assertRaises(numerical.error):
                            numerical.asarray(case, dtype)
                        continue
                    got = numerical.asarray(case, dtype)
                    self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape))
                    self.assertEqual(repr(got.tolist()), repr(want.tolist()))
        self.assertIsNone(numerical._asarray_nested([fractions.Fraction(1, 3)]))
        self.assertIsNone(numerical._asarray_nested([1, [2]]))
        self.assertIsNone(numerical._asarray_nested(3.0))
        with self.assertRaises(numerical.error):
            numerical._asarray_nested([1.0], "float16")
        with self.assertRaises(OverflowError):
            numerical.asarray([2 ** 1100])
        with self.assertRaises(ValueError):
            numerical.asarray([1, [2]])
        with self.assertRaises(ValueError):
            numerical.asarray([[1, 2], [3, 4, 5]])

    def test_asarray_reads_every_numeric_typestr_at_any_stride(self):
        # Without NumPy: an __array_interface__ over a packed buffer of each element type, read through
        # strides that walk it backwards.
        import ctypes
        import struct

        class Interface:
            def __init__(self, typestr, raw, shape, strides=None):
                self.buffer = ctypes.create_string_buffer(raw, len(raw))
                address = ctypes.addressof(self.buffer)
                offset = 0 if strides is None or strides[0] > 0 else len(raw) + strides[0]
                self.__array_interface__ = {"version": 3, "typestr": typestr, "shape": shape,
                                            "strides": strides, "data": (address + offset, True)}

        for typestr, fmt in (("|b1", "?"), ("|i1", "b"), ("<i2", "h"), ("<i4", "i"), ("<i8", "q"),
                             ("|u1", "B"), ("<u2", "H"), ("<u4", "I"), ("<u8", "Q"), ("<f4", "f"),
                             (">i4", "i"), (">f8", "d")):
            order = ">" if typestr[0] == ">" else "<"
            raw = struct.pack(order + "3" + fmt, *([1, 0, 1] if fmt == "?" else [1, 2, 3]))
            want = [1.0, 0.0, 1.0] if fmt == "?" else [1.0, 2.0, 3.0]
            with self.subTest(typestr=typestr):
                self.assertEqual(numerical.asarray(Interface(typestr, raw, (3,))).tolist(), want)
                size = struct.calcsize(fmt)
                self.assertEqual(numerical.asarray(Interface(typestr, raw, (3,), (-size,))).tolist(), want[::-1])
        pairs = struct.pack(">4f", 1.0, -2.0, 0.5, 3.0)
        self.assertEqual(numerical.asarray(Interface(">c8", pairs, (2,))).tolist(), [1 - 2j, 0.5 + 3j])
        for bad in ("<f2", "|f8", "<U4", "<i3", "xf8", "<f"):
            with self.subTest(bad=bad):
                with self.assertRaises(numerical.error):
                    numerical.asarray(Interface(bad, b"\0" * 8, (1,)))
        with self.assertRaises(numerical.error):
            numerical.asarray(Interface("<f8", b"\0" * 16, (2,), (8, 8)))  # strides of another rank

    def test_asarray_rejects_unknown_dtype(self):
        with self.assertRaises(numerical.error):
            numerical.asarray(numerical.Tensor([1.0]), dtype="float16")


class GilRelease(unittest.TestCase):
    def test_a_large_decomposition_lets_another_thread_run(self):
        # Another thread wakes 1 ms after the call starts. Holding the GIL, the call keeps it from running
        # until it returns; releasing it, the thread runs while the SVD still computes. The SVD must outlast
        # the sleep by far: a 1 ms sleep takes up to 16 ms on Windows, and order 600 takes about 300 ms.
        import threading
        import time

        n = 600
        a = numerical.Tensor([float((i * 7 + j * 13) % 17) + (n if i == j else 0) for i in range(n) for j in range(n)],
                             shape=(n, n))
        started = threading.Event()
        woke = []

        def other():
            started.wait()
            time.sleep(0.001)
            woke.append(time.perf_counter())

        thread = threading.Thread(target=other)
        thread.start()
        begin = time.perf_counter()
        started.set()
        numerical.svd(a)
        end = time.perf_counter()
        thread.join()
        self.assertLess(woke[0] - begin, (end - begin) / 2)

    def test_a_large_elementwise_function_lets_another_thread_run(self):
        # As the decomposition: a complex tangent of 8M elements on one thread (80 ms) outlasts the sleep by far.
        import threading
        import time

        z = numerical.asarray(np.full(1 << 23, 0.5 + 0.25j)) if np is not None else None
        if z is None:
            self.skipTest("NumPy builds the operand")
        threads = numerical.get_num_threads()
        numerical.set_num_threads(1)
        try:
            started = threading.Event()
            woke = []

            def other():
                started.wait()
                time.sleep(0.001)
                woke.append(time.perf_counter())

            thread = threading.Thread(target=other)
            thread.start()
            begin = time.perf_counter()
            started.set()
            numerical.tan(z)
            end = time.perf_counter()
            thread.join()
        finally:
            numerical.set_num_threads(threads)
        self.assertLess(woke[0] - begin, (end - begin) / 2)


class DLPack(unittest.TestCase):
    def test_capsule_and_device(self):
        t = numerical.Tensor([1, 2, 3])
        capsule = t.__dlpack__()
        self.assertIn("PyCapsule", repr(type(capsule)))
        self.assertEqual(t.__dlpack_device__(), (1, 0))  # (kDLCPU, 0)


@unittest.skipIf(np is None, "NumPy not installed")
class NumpyInterop(unittest.TestCase):
    def test_asarray_reads_numpy_arrays_of_every_dtype_in_either_byte_order(self):
        # NumPy's arrays and scalars are read through __array_struct__: every element type, both byte orders,
        # transposed and reversed views, and a rank-0 scalar, as NumPy holds them.
        for kind in ("?", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "c8", "c16"):
            for order in ("<", ">"):
                dtype = np.dtype(kind if kind in ("?", "i1", "u1") else order + kind)
                base = (np.arange(6) % 2 == 1) if kind == "?" else np.arange(6).astype(dtype)
                if kind.startswith("c"):
                    base = (base + 1j * np.arange(6)).astype(dtype)
                matrix = base.astype(dtype).reshape(2, 3)
                for view in (matrix, matrix.T, matrix[::-1, ::-2]):
                    with self.subTest(dtype=dtype.str, strides=view.strides):
                        want = view.astype(complex if kind.startswith("c") else float).tolist()
                        self.assertEqual(numerical.asarray(view).tolist(), want)
        self.assertEqual(numerical.asarray(np.float32(1.5)).tolist(), 1.5)
        self.assertEqual(numerical.asarray(np.arange(3.0), dtype="complex128").tolist(), [0j, 1 + 0j, 2 + 0j])
        for bad in (np.zeros(2, np.float16), np.array(["a"])):
            with self.subTest(dtype=bad.dtype.str):
                with self.assertRaises(numerical.error):
                    numerical.asarray(bad)
        with self.assertRaises(numerical.error):
            numerical.asarray(np.array([1j]), dtype="float64")

    def test_array_interface_zero_copy_view(self):
        t = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        arr = np.asarray(t)
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.dtype, np.float64)
        self.assertEqual(arr.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_dlpack_round_trip(self):
        t = numerical.Tensor([1.5, 2.5, 3.5])
        arr = np.from_dlpack(t)
        self.assertEqual(arr.tolist(), [1.5, 2.5, 3.5])

    def test_asarray_from_numpy(self):
        source = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = numerical.asarray(source)
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_asarray_widens_numpy_real_to_complex(self):
        source = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = numerical.asarray(source, dtype="complex128")
        self.assertEqual(t.dtype, "complex128")
        self.assertTrue(np.allclose(np.asarray(t), source.astype(np.complex128)))

    def test_asarray_refuses_numpy_complex_to_float64(self):
        with self.assertRaises(numerical.error):
            numerical.asarray(np.array([1 + 2j, 3 + 4j]), dtype="float64")

    def test_einsum_matches_numpy(self):
        a = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        b = numerical.Tensor([5, 6, 7, 8], shape=(2, 2))
        want = np.einsum("ij,jk->ik", np.asarray(a), np.asarray(b))
        self.assertEqual(np.asarray(numerical.einsum("ij,jk->ik", a, b)).tolist(),
                         want.tolist())


class LinalgSurface(unittest.TestCase):
    """Shape, dtype and structure of the decomposition layer (no NumPy needed)."""

    def test_determinant_value_and_dtype(self):
        a = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        self.assertAlmostEqual(numerical.determinant(a), -2.0)  # 1*4 - 2*3
        c = numerical.Tensor([1, 0, 0, 1], shape=(2, 2), dtype="complex128")
        self.assertAlmostEqual(numerical.determinant(c), 1 + 0j)

    def test_inverse_and_cholesky_return_matrices(self):
        a = numerical.Tensor([4, 2, 2, 3], shape=(2, 2))
        self.assertEqual(numerical.inverse(a).shape, (2, 2))
        self.assertEqual(numerical.cholesky(a).shape, (2, 2))

    def test_factorisations_return_tuples(self):
        a = numerical.Tensor([1, 1, 1, 2, 1, 3], shape=(3, 2))
        q, r = numerical.qr(a)
        self.assertEqual((q.shape, r.shape), ((3, 2), (2, 2)))
        sym = numerical.Tensor([2, 1, 1, 2], shape=(2, 2))
        values, vectors = numerical.eigh(sym)
        self.assertEqual((values.shape, vectors.shape), ((2,), (2, 2)))
        u, s, vh = numerical.svd(a)  # NumPy's full_matrices=True: U m×m, Vh n×n
        self.assertEqual((u.shape, s.shape, vh.shape), ((3, 3), (2,), (2, 2)))
        wide = numerical.Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
        q, r = numerical.qr(wide)  # NumPy's reduced QR: Q m×k, R k×n
        self.assertEqual((q.shape, r.shape), ((2, 2), (2, 3)))
        u, s, vh = numerical.svd(wide)
        self.assertEqual((u.shape, s.shape, vh.shape), ((2, 2), (2,), (3, 3)))
        u, s, vh = numerical.svd(wide, full_matrices=False)
        self.assertEqual((u.shape, s.shape, vh.shape), ((2, 2), (2,), (2, 3)))

    def test_eigh_takes_a_hermitian_matrix_and_svd_rejects_a_vector(self):
        h = numerical.Tensor([2, 1 - 1j, 1 + 1j, 3], shape=(2, 2), dtype="complex128")
        values, vectors = numerical.eigh(h)
        self.assertEqual((values.dtype, vectors.dtype), ("float64", "complex128"))
        self.assertAlmostEqual(values.tolist()[0], 1.0)
        self.assertAlmostEqual(values.tolist()[1], 4.0)
        with self.assertRaises(numerical.error):
            numerical.svd(numerical.Tensor([1, 2, 3], shape=(3,)))

    def test_inverse_rejects_singular(self):
        with self.assertRaises(numerical.error):
            numerical.inverse(numerical.Tensor([1, 2, 2, 4], shape=(2, 2)))


class TensorProtocol(unittest.TestCase):
    """A Tensor indexes, iterates, transposes and multiplies as a NumPy array of rank 0 to 3 does."""

    def test_errors_name_the_problem(self):
        t = numerical.asarray([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(IndexError):
            t[2]
        with self.assertRaises(IndexError):
            t[0, 0, 0]
        with self.assertRaises(TypeError):
            t[0.5]
        scalar = numerical.asarray(3.0)
        with self.assertRaises(TypeError):
            len(scalar)
        with self.assertRaises(TypeError):
            iter(scalar)
        with self.assertRaises(numerical.LinAlgError):
            numerical.asarray([[[1.0]]]) @ t
        with self.assertRaises(IndexError):  # a direct call outside the resolved bounds is refused, not read
            numerical._numerical._take(t, [0, 5], [1, 1], [1, 1], [1, 1])

    def test_an_empty_slice_exports_a_non_null_address(self):
        empty = numerical.asarray([[1.0, 2.0], [3.0, 4.0]])[1:1]
        self.assertEqual(empty.shape, (0, 2))
        self.assertNotEqual(empty.__array_interface__["data"][0], 0)  # NumPy 2.2 refuses a null one
        if np is not None:
            self.assertEqual(np.asarray(empty).shape, (0, 2))

    def test_len_iter_transpose_and_matmul(self):
        t = numerical.asarray([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(len(t), 2)
        self.assertEqual([row.tolist() for row in t], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(list(numerical.asarray([5.0, 6.0])), [5.0, 6.0])
        self.assertEqual(t.T.tolist(), [[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual((t @ t).tolist(), [[7.0, 10.0], [15.0, 22.0]])
        self.assertEqual((t @ [1.0, 1.0]).tolist(), [3.0, 7.0])
        self.assertEqual(([1.0, 0.0] @ t).tolist(), [1.0, 2.0])
        self.assertEqual(numerical.asarray([1.0, 2.0]) @ [3.0, 4.0], 11.0)
        self.assertEqual(numerical.asarray(2.5)[()], 2.5)  # a rank-0 tensor's number, as NumPy's a[()]

    def test_matmul_of_two_tensors_is_the_einsum_and_the_rest_converts_or_refuses(self):
        # Two tensors of one dtype take the type's own product; any other operand goes through the conversions,
        # with the same answers and the same refusals.
        real = numerical.asarray([[1.0, 2.0], [3.0, 4.0]])
        cplx = numerical.asarray([[1j, 2.0], [3.0, -1j]])
        for m in (real, cplx):
            v = numerical.asarray([1.0, 2.0], dtype=m.dtype)
            self.assertEqual((m @ m).tolist(), numerical.einsum("ij,jk->ik", m, m).tolist())
            self.assertEqual((m @ v).tolist(), numerical.einsum("ij,j->i", m, v).tolist())
            self.assertEqual((v @ m).tolist(), numerical.einsum("i,ij->j", v, m).tolist())
            self.assertEqual(v @ v, 5.0)
            self.assertIsInstance(v @ v, float if m is real else complex)
        self.assertEqual((real @ [[1.0], [1.0]]).tolist(), [[3.0], [7.0]])
        with self.assertRaises(numerical.LinAlgError):
            numerical.asarray([[[1.0]]]) @ real
        with self.assertRaises(numerical.LinAlgError):
            real @ numerical.asarray(1.0)
        mixed = real @ cplx  # two dtypes: promoted, as einsum promotes them
        self.assertEqual((mixed.dtype, mixed.tolist()), ("complex128", numerical.einsum("ij,jk->ik", real, cplx).tolist()))
        with self.assertRaises(numerical.error):  # two tensors: the type's own product, in C
            real @ numerical.asarray([[1.0, 2.0, 3.0]])
        with self.assertRaises(numerical.LinAlgError):  # a list converted first: a ValueError, as NumPy's matmul raises
            real @ [[1.0, 2.0, 3.0]]

    @unittest.skipIf(np is None, "NumPy not installed")
    def test_indexing_as_numpy(self):
        rng = np.random.default_rng(3)
        for shape in ((5,), (4, 6), (3, 4, 5)):
            for cplx in (False, True):
                a = rng.standard_normal(shape) + (1j * rng.standard_normal(shape) if cplx else 0)
                t = numerical.asarray(a)
                for _ in range(200):
                    key = []
                    for extent in shape[:rng.integers(1, len(shape) + 1)]:
                        if rng.integers(0, 2):
                            key.append(int(rng.integers(-extent, extent)))
                        else:
                            bound = lambda: None if rng.integers(0, 3) == 0 else int(rng.integers(-extent - 2, extent + 2))
                            step = [None, 1, 2, -1, -3][rng.integers(0, 5)]
                            key.append(slice(bound(), bound(), step))
                    key = tuple(key)
                    got, want = t[key], a[key]
                    got = np.asarray(got) if isinstance(got, numerical.Tensor) else got
                    self.assertTrue(np.array_equal(got, want), msg=f"{shape} {key}")
                self.assertTrue(np.array_equal(np.asarray(t.T), a.T))


class ImpostorTensors(unittest.TestCase):
    """An object that claims to be a Tensor through its __class__ holds none: it is refused, never read
    (a mock with spec=Tensor crashed the interpreter in determinant, slogdet and asarray)."""

    def test_a_mock_tensor_is_refused(self):
        import unittest.mock as mock
        impostor = mock.MagicMock(spec=numerical.Tensor)
        self.assertIsInstance(impostor, numerical.Tensor)  # what the check must not trust
        for function in (numerical.determinant, numerical.slogdet, numerical.inverse):
            with self.assertRaises((numerical.error, TypeError)):
                function(impostor)
        with self.assertRaises((numerical.error, TypeError, ValueError)):
            numerical.asarray(impostor)


class ArrayLikeArguments(unittest.TestCase):
    """Every function takes what NumPy's take: nested lists and numbers too, not only Tensors; and its failures
    are LinAlgError, a ValueError as numpy.linalg.LinAlgError is, and still a numerical.error."""

    def test_nested_lists_are_accepted_everywhere(self):
        a = [[4.0, 2.0], [2.0, 3.0]]
        self.assertAlmostEqual(numerical.determinant(a), 8.0)
        self.assertAlmostEqual(numerical.slogdet(a)[1], numerical.slogdet(numerical.asarray(a))[1])
        self.assertEqual(numerical.solve(a, [2.0, 1.0]).tolist(), numerical.solve(numerical.asarray(a), numerical.asarray([2.0, 1.0])).tolist())
        for function in (numerical.inverse, numerical.cholesky, numerical.pinv):
            self.assertEqual(function(a).shape, (2, 2))
        self.assertEqual(numerical.matrix_rank(a), 2)
        self.assertEqual(len(numerical.qr(a)), 2)
        self.assertEqual(len(numerical.eigh(a)), 2)
        self.assertEqual(len(numerical.svd(a)), 3)
        self.assertEqual(numerical.lstsq(a, [2.0, 1.0])[0].shape, (2,))
        self.assertEqual(numerical.einsum("ij,j->i", a, [1.0, 0.0]).tolist(), [4.0, 2.0])

    def test_a_matrix_right_hand_side_solves_every_column(self):
        a = [[3.0, 1.0], [1.0, 2.0]]
        x = numerical.solve(a, [[9.0, 1.0], [8.0, 0.0]])
        self.assertEqual(x.shape, (2, 2))
        for column, rhs in ((0, [9.0, 8.0]), (1, [1.0, 0.0])):
            single = numerical.solve(a, rhs).tolist()
            for row in range(2):
                self.assertAlmostEqual(x.tolist()[row][column], single[row])
        wide = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        y = numerical.lstsq(wide, [[1.0, 0.0], [0.0, 1.0]])[0]
        self.assertEqual(y.shape, (3, 2))
        for column, rhs in ((0, [1.0, 0.0]), (1, [0.0, 1.0])):
            single = numerical.lstsq(wide, rhs)[0].tolist()
            for row in range(3):
                self.assertAlmostEqual(y.tolist()[row][column], single[row])
        with self.assertRaises(numerical.LinAlgError):
            numerical.solve(a, [[[1.0]]])  # rank 3

    def test_a_real_operand_is_widened_to_meet_a_complex_one(self):
        x = numerical.solve([[2.0, 0.0], [0.0, 1.0]], [2j, 1.0])
        self.assertEqual(x.dtype, "complex128")
        self.assertEqual(x.tolist(), [1j, 1.0])

    def test_a_tensor_argument_is_passed_as_it_is(self):
        a = numerical.Tensor([4, 2, 2, 3], shape=(2, 2))
        self.assertAlmostEqual(numerical.determinant(a), 8.0)

    def test_failures_are_linalg_errors(self):
        self.assertTrue(issubclass(numerical.LinAlgError, ValueError))
        self.assertTrue(issubclass(numerical.LinAlgError, numerical.error))
        with self.assertRaises(numerical.LinAlgError) as caught:
            numerical.inverse([[1.0, 2.0], [2.0, 4.0]])
        self.assertIn("singular", str(caught.exception))
        with self.assertRaises(ValueError):
            numerical.cholesky([[1.0, 2.0], [2.0, 1.0]])
        with self.assertRaises(TypeError):  # not an array-like at all: asarray's own error, unchanged
            numerical.inverse([["a"]])

    def test_the_wrappers_keep_their_names_and_documentation(self):
        self.assertEqual(numerical.solve.__name__, "solve")
        self.assertTrue(numerical.solve.__doc__.startswith("solve(a, b)"))

    @unittest.skipIf(np is None, "NumPy not installed")
    def test_numpy_arrays_are_accepted(self):
        a = np.array([[4.0, 2.0], [2.0, 3.0]])
        np.testing.assert_allclose(np.asarray(numerical.inverse(a)), np.linalg.inv(a))
        np.testing.assert_allclose(np.asarray(numerical.solve(a, np.array([2.0, 1.0]))), np.linalg.solve(a, [2.0, 1.0]))


@unittest.skipIf(np is None, "NumPy not installed")
class LinalgVsNumpy(unittest.TestCase):
    """Validate the decomposition layer against numpy.linalg."""

    def test_determinant_and_inverse(self):
        a = np.array([[4.0, 7.0], [2.0, 6.0]])
        t = numerical.asarray(a)
        self.assertAlmostEqual(numerical.determinant(t), np.linalg.det(a))
        self.assertTrue(np.allclose(np.asarray(numerical.inverse(t)), np.linalg.inv(a)))

    def test_lstsq(self):
        a = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b = np.array([2.0, 3.0, 5.0])
        got = np.asarray(numerical.lstsq(numerical.asarray(a), numerical.asarray(b))[0])
        want = np.linalg.lstsq(a, b, rcond=None)[0]
        self.assertTrue(np.allclose(got, want))

    def test_cholesky(self):
        a = np.array([[4.0, 2.0], [2.0, 3.0]])
        self.assertTrue(np.allclose(np.asarray(numerical.cholesky(numerical.asarray(a))),
                                    np.linalg.cholesky(a)))

    def test_qr_reconstructs(self):
        a = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        q, r = numerical.qr(numerical.asarray(a))
        self.assertTrue(np.allclose(np.asarray(q) @ np.asarray(r), a)) # signs may differ; reconstruct

    def test_qr_mode_r_is_the_reduced_r_to_the_bit(self):
        # mode "r" never forms Q: its R is the reduced decomposition's, bit for bit, in each dtype and over a stack.
        rng = np.random.default_rng(5)
        for shape in ((1, 1), (5, 3), (3, 5), (70, 65), (2, 9, 7)):
            base = rng.standard_normal(shape)
            for values in (base, base + 1j * base[..., ::-1], base.astype(np.float32), (base - 2j * base).astype(np.complex64)):
                t = numerical.asarray(values)
                r = np.asarray(numerical.qr(t, mode="r"))
                want = np.asarray(numerical.qr(t)[1])
                self.assertEqual((r.shape, r.dtype), (want.shape, want.dtype), shape)
                self.assertTrue(np.array_equal(r, want), (shape, values.dtype))
        with self.assertRaises(numerical.LinAlgError):
            numerical.qr(numerical.asarray(np.array([[1.0, np.nan], [0.0, 1.0]])), mode="r")

    def test_eigh_reconstructs(self):
        a = np.array([[2.0, 1.0], [1.0, 2.0]])
        values, vectors = numerical.eigh(numerical.asarray(a))
        v = np.asarray(values)
        matrix = np.asarray(vectors)
        self.assertTrue(np.allclose(np.sort(v), np.linalg.eigvalsh(a)))
        self.assertTrue(np.allclose(matrix @ np.diag(v) @ matrix.T, a))

    def test_slogdet_matrix_rank_and_pinv_as_numpy(self):
        rng = np.random.default_rng(11)
        for a in (np.eye(200) * 1e-2, np.array([[0.0, 2.0], [1.0, 0.0]]), np.array([[1.0, 2.0], [2.0, 4.0]]),
                  rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))):
            sign, logabsdet = numerical.slogdet(numerical.asarray(a))
            want_sign, want_log = np.linalg.slogdet(a)
            self.assertAlmostEqual(sign, want_sign)
            self.assertTrue(np.isclose(logabsdet, want_log) or logabsdet == want_log == -np.inf)
        for a in (np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), rng.standard_normal((5, 3)),
                  rng.standard_normal((3, 5)), rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))):
            self.assertEqual(numerical.matrix_rank(numerical.asarray(a)), np.linalg.matrix_rank(a))
            self.assertTrue(np.allclose(np.asarray(numerical.pinv(numerical.asarray(a))), np.linalg.pinv(a)))

    def test_eigh_reads_the_lower_triangle_as_numpy(self):
        a = np.array([[2.0, 99.0, -7.0], [1.0, 3.0, 50.0], [0.5, -1.0, 4.0]])
        values, _ = numerical.eigh(numerical.asarray(a))
        self.assertTrue(np.allclose(np.asarray(values), np.linalg.eigvalsh(a)))

    def test_eigh_of_a_hermitian_matrix_as_numpy(self):
        rng = np.random.default_rng(5)
        m = rng.standard_normal((12, 12)) + 1j * rng.standard_normal((12, 12))
        h = m + m.conj().T
        values, vectors = (np.asarray(t) for t in numerical.eigh(h))
        self.assertTrue(np.allclose(values, np.linalg.eigvalsh(h)))
        self.assertTrue(np.allclose(vectors @ np.diag(values) @ vectors.conj().T, h))
        self.assertTrue(np.allclose(vectors.conj().T @ vectors, np.eye(12)))
        upper_ignored = np.tril(h) + np.triu(np.full((12, 12), 9 + 9j), 1)  # lower triangle read, as NumPy's
        self.assertTrue(np.allclose(np.asarray(numerical.eigh(upper_ignored)[0]), np.linalg.eigvalsh(upper_ignored)))

    def test_eigvalsh_is_eighs_values_alone_as_numpy(self):
        rng = np.random.default_rng(6)
        for n in (1, 2, 9, 40, 130):
            m = rng.standard_normal((n, n))
            c = m + 1j * rng.standard_normal((n, n))
            for a in (m + m.T, c + c.conj().T):
                got = np.asarray(numerical.eigvalsh(a))
                self.assertEqual(got.dtype, np.float64)
                self.assertTrue(np.allclose(got, np.linalg.eigvalsh(a), rtol=0, atol=1e-12 * np.abs(a).max() * n))
                self.assertTrue(np.allclose(got, np.asarray(numerical.eigh(a)[0]), rtol=0, atol=1e-12 * np.abs(a).max() * n))
        upper_ignored = np.tril(m) + np.triu(np.full((130, 130), 7.0), 1)  # lower triangle read, as NumPy's
        self.assertTrue(np.allclose(np.asarray(numerical.eigvalsh(upper_ignored)), np.linalg.eigvalsh(upper_ignored)))
        self.assertEqual(np.asarray(numerical.eigvalsh(np.zeros((0, 0)))).shape, (0,))
        with self.assertRaises(numerical.LinAlgError):
            numerical.eigvalsh(np.ones((2, 3)))
        with self.assertRaises(numerical.LinAlgError):
            numerical.eigvalsh(np.ones(3))
        with self.assertRaises(numerical.LinAlgError):
            numerical.eigvalsh(np.array([[1.0, 0.0], [np.nan, 1.0]]))

    def test_asarray_takes_numpy_dtypes_and_layouts(self):
        a = np.arange(12, dtype=np.int64).reshape(3, 4)
        for view in (a, a.T, a[::-1, ::2], np.asfortranarray(a)):
            self.assertEqual(numerical.asarray(view).tolist(), view.astype(float).tolist())
        for dtype in (np.float32, np.int8, np.uint32, np.bool_, np.complex64):
            values = np.array([1, 0, 2], dtype=dtype)
            got = np.asarray(numerical.asarray(values))
            self.assertTrue(np.array_equal(got, values.astype(np.complex128 if dtype == np.complex64 else float)))

    def test_asarray_copies_every_dtype_and_layout_as_numpy_reads_it(self):
        # Every element type into float64 and complex128, through every layout the copy treats apart: a
        # contiguous run (copied whole when the type is the output's), strided rows, a reversed axis, column
        # order, and rank 0.
        base = np.arange(24).reshape(2, 3, 4) - 7
        for dtype in (np.float64, np.float32, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint16,
                      np.uint8, np.bool_, np.complex128, np.complex64):
            values = base.astype(dtype)
            for view in (values, values[:, ::2, :], values[::-1, :, ::-1], np.asfortranarray(values),
                         values[1, 2, 3]):
                for out in ("float64", "complex128"):
                    if out == "float64" and np.dtype(dtype).kind == "c":
                        continue
                    with self.subTest(dtype=np.dtype(dtype).name, shape=np.shape(view), out=out):
                        got = np.asarray(numerical.asarray(np.asarray(view), out))
                        self.assertTrue(np.array_equal(got, np.asarray(view).astype(out)))

    def test_svd_reconstructs(self):
        a = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        u, s, vh = numerical.svd(numerical.asarray(a), full_matrices=False)
        reconstructed = np.asarray(u) @ np.diag(np.asarray(s)) @ np.asarray(vh)
        self.assertTrue(np.allclose(reconstructed, a))
        self.assertTrue(np.allclose(np.asarray(s), np.linalg.svd(a, compute_uv=False)))

    def test_svd_third_factor_is_vh_as_numpy_returns_it(self):
        # numpy's idiom (U * S) @ Vh reconstructs a, wide or tall, real or complex.
        rng = np.random.default_rng(7)
        for a in (rng.standard_normal((4, 3)), rng.standard_normal((3, 5)),
                  rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))):
            u, s, vh = (np.asarray(f) for f in numerical.svd(numerical.asarray(a), full_matrices=False))
            self.assertTrue(np.allclose((u * s) @ vh, a))
            self.assertTrue(np.allclose(s, np.linalg.svd(a, compute_uv=False)))

    def test_svdvals_of_exact_ranks_are_lapacks(self):
        # Exactly rank-deficient matrices whose reflections LAPACK's dlarfg leaves exact: NumPy's values to the bit,
        # their zeros exact (Julia's cond(·, 2) of [0 0; 1 1] Inf), below and past the blocked reduction's crossover.
        ones = np.ones((5, 5))
        ones[0] = 0.0
        cases = [np.array([[0.0, 0.0], [1.0, 1.0]]), ones, np.array([[0, 0], [1, 1j]])]
        for n in (130, 200):
            b = np.eye(n)
            b[0, 0], b[1, 0], b[1, 1] = 0.0, 1.0, 1.0
            cases.append(b)
        for a in cases:
            self.assertTrue(np.array_equal(np.asarray(numerical.svdvals(a)), np.linalg.svd(a, compute_uv=False)))

    def test_svd_keeps_small_singular_values(self):
        # sigma down to 1e-12 under random rotations, against numpy (which uses LAPACK's gesdd).
        rng = np.random.default_rng(3)
        q1, _ = np.linalg.qr(rng.standard_normal((6, 4)))
        q2, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        a = q1 @ np.diag([1.0, 1e-4, 1e-8, 1e-12]) @ q2.T
        s = np.asarray(numerical.svd(numerical.asarray(a))[1])
        self.assertTrue(np.allclose(s, np.linalg.svd(a, compute_uv=False), rtol=1e-3, atol=0))

    def test_lstsq_rank_deficient_and_wide_give_numpys_minimum_norm(self):
        for a, b in ((np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), np.array([1.0, 2.0, 4.0])),
                     (np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]), np.array([3.0, 1.0]))):
            got = np.asarray(numerical.lstsq(numerical.asarray(a), numerical.asarray(b))[0])
            self.assertTrue(np.allclose(got, np.linalg.lstsq(a, b, rcond=None)[0]))




class Threads(unittest.TestCase):
    def test_a_cap_changes_the_threads_not_the_bits(self):
        a = numerical.asarray([[float((i * 7 + j * 3) % 11) - 5.0 for j in range(300)] for i in range(300)])
        default = (a @ a).tolist()
        numerical.set_num_threads(1)
        try:
            self.assertEqual(numerical.get_num_threads(), 1)
            self.assertEqual((a @ a).tolist(), default)
        finally:
            numerical.set_num_threads(0)
        self.assertGreaterEqual(numerical.get_num_threads(), 1)

    def test_every_gemm_path_agrees_to_rounding(self):
        a = numerical.asarray([[((i * 7 + j * 3) % 11) / 3.0 - 1.5 for j in range(200)] for i in range(200)])
        self.assertEqual(numerical.limit_gemm_path("portable"), "portable")
        plain = (a @ a).tolist()
        for path in ("neon", "avx2", "avx512", "amx"):
            got = numerical.limit_gemm_path(path)
            self.assertIn(got, ("portable", "neon", "avx2", "avx512", "amx"))
            worst = max(abs(x - y) for row_f, row_p in zip((a @ a).tolist(), plain) for x, y in zip(row_f, row_p))
            self.assertLess(worst, 1e-11)
        self.assertEqual(numerical.get_config()["gemm_path"], numerical.gemm_path())
        with self.assertRaises(ValueError):
            numerical.limit_gemm_path("sse9")

    def test_the_highest_cap_restores_the_best_path(self):
        # In a process no environment caps: the path at import is the best one, and the last name gives it back.
        env = {k: v for k, v in os.environ.items() if k not in ("NUMERICAL_GEMM", "NUMERICAL_AMX")}
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        script = ("import numerical; best = numerical.gemm_path(); numerical.limit_gemm_path('portable'); "
                  "print(best, numerical.limit_gemm_path('amx'))")
        best, restored = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True,
                                        check=True).stdout.split()
        self.assertEqual(restored, best)

    def test_the_environment_limits_the_gemm_path_at_import(self):
        import subprocess

        def run(env, code="import numerical; print(numerical.gemm_path())"):
            out = subprocess.run([sys.executable, "-W", "always", "-c", code], env=dict(os.environ, **env),
                                 capture_output=True, text=True, check=True)
            return out.stdout.strip(), out.stderr

        self.assertEqual(run({"NUMERICAL_GEMM": "portable"})[0], "portable")
        self.assertEqual(run({"NUMERICAL_GEMM": "PORTABLE"})[0], "portable")
        self.assertNotEqual(run({"NUMERICAL_AMX": "0"})[0], "amx")
        self.assertNotEqual(run({"NUMERICAL_AMX": "off", "NUMERICAL_GEMM": "amx"})[0], "amx")  # the more portable wins
        stdout, stderr = run({"NUMERICAL_GEMM": "fastest"})
        self.assertIn("NUMERICAL_GEMM", stderr)
        self.assertIn("NUMERICAL_AMX", run({"NUMERICAL_AMX": "maybe"})[1])
        # Under NUMERICAL_AMX=0 the AMX's self-test never runs, a product and the path's report included.
        probe = ("import numerical, numerical._numerical as m; a = numerical.asarray([[1.0] * 64] * 64); a @ a; "
                 "numerical.gemm_path(); numerical.get_config(); print(m._amx_probed())")
        self.assertEqual(run({"NUMERICAL_AMX": "0"}, probe)[0], "False")
        self.assertEqual(run({"NUMERICAL_GEMM": "portable"}, probe)[0], "False")

    def test_thread_counts_are_validated(self):
        import subprocess

        for bad in (-1, True, 2.0, "2"):
            with self.assertRaises(ValueError):
                numerical.set_num_threads(bad)
        out = subprocess.run([sys.executable, "-W", "always", "-c", "import numerical; print(numerical.get_num_threads())"],
                             env=dict(os.environ, NUMERICAL_NUM_THREADS="\u00b2"), capture_output=True, text=True,
                             check=True)
        self.assertIn("NUMERICAL_NUM_THREADS", out.stderr)
        out = subprocess.run([sys.executable, "-c", "import numerical; print(numerical.get_num_threads())"],
                             env=dict(os.environ, NUMERICAL_NUM_THREADS="1"), capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "1")

    def test_the_environment_sets_the_cap_at_import(self):
        import subprocess
        env = dict(os.environ, NUMERICAL_NUM_THREADS="1")
        out = subprocess.run([sys.executable, "-c", "import numerical; print(numerical.get_num_threads())"],
                             env=env, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "1")


class FirstGestures(unittest.TestCase):
    def test_a_mismatched_product_names_both_shapes(self):
        a = numerical.Tensor([1.0] * 6, shape=(2, 3))
        for left, right, words in ((a, a, "a is 2 × 3 and b is 2 × 3: a's 3 columns do not match b's 2 rows"),
                                   (a, numerical.Tensor([1.0, 2.0]), "b's 2 entries")):
            with self.subTest(words=words):
                with self.assertRaises(numerical.LinAlgError) as caught:
                    left @ right
                self.assertIn(words, str(caught.exception))
                with self.assertRaises(numerical.LinAlgError) as caught:
                    numerical.matmul(left, right)
                self.assertIn(words, str(caught.exception))
        if np is not None:
            with self.assertRaises(numerical.LinAlgError) as caught:
                numerical.matmul(np.ones((4, 5)), np.ones((4, 5)))
            self.assertIn("a is 4 × 5 and b is 4 × 5", str(caught.exception))

    def test_writing_into_a_tensor_says_why_not(self):
        t = numerical.Tensor([1.0, 2.0])
        with self.assertRaises(TypeError) as caught:
            t[0] = 5.0
        self.assertIn("immutable", str(caught.exception))

if __name__ == "__main__":
    unittest.main()
