"""The Tensor operators' short path (numerical._numerical._fast_binary and _fast_compare): two Tensors of one dtype,
or a float or complex Tensor and a weak Python number, computed in C as the promoting path computes them; NumPy's
answers, and the promoting path's for every operand the short path leaves to it."""

import operator
import sys
import unittest

import numerical

try:
    import numpy as np
except ImportError:  # pragma: no cover - the CI images install NumPy
    np = None

ARITHMETIC = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "**": operator.pow,
              "//": operator.floordiv, "%": operator.mod}
COMPARISONS = {"<": operator.lt, "<=": operator.le, "==": operator.eq, "!=": operator.ne, ">": operator.gt,
               ">=": operator.ge}


@unittest.skipIf(np is None, "NumPy is not installed")
class FastOperators(unittest.TestCase):
    def assert_like_numpy(self, got, want):
        want = np.asarray(want)
        self.assertIsInstance(got, numerical.Tensor)
        self.assertEqual((got.dtype, got.shape), (str(want.dtype), want.shape))
        np.testing.assert_array_equal(np.asarray(got), want)  # NaNs where NumPy's are

    def test_a_weak_number_keeps_the_tensors_dtype_on_either_side(self):
        # float32 beside 0.1 compares and computes 0.1 rounded to float32; a complex Tensor takes a real number as its
        # real part; an int is read as a float of the dtype; reflected operators put the number first.
        values = np.array([-2.5, -0.0, 0.0, 0.1, 1.0, 3.0, np.inf, np.nan])
        with np.errstate(all="ignore"):
            for dtype in (np.float64, np.float32, np.complex128, np.complex64):
                a = values.astype(dtype) if np.dtype(dtype).kind == "f" else (values + 1j * values[::-1]).astype(dtype)
                t = numerical.asarray(a)
                for number in (0.1, 2, -3, 2.5, 2**60 + 1):
                    for name, f in ARITHMETIC.items():
                        if np.dtype(dtype).kind == "c" and name in ("//", "%"):
                            continue
                        with self.subTest(dtype=np.dtype(dtype).name, number=number, op=name):
                            if name == "**":
                                # the power of the number as a rank-0 Tensor of the dtype: NumPy's own real power
                                # differs by a unit in the last place from one processor to another (0.1 ** -3 is
                                # 999.9999999999998 on a GitHub runner, 1000.0 on an AVX2 core, the C library's
                                # 999.9999999999999), and test_a_complex_power_is_numpys holds the complex one
                                as_tensor = numerical.asarray(number, t.dtype)
                                if np.dtype(dtype).kind == "c" and number in (2, -1, 0.5):  # NumPy's shortcuts
                                    self.assert_like_numpy(t ** number, a ** number)
                                else:
                                    self.assertEqual(np.asarray(t ** number).tobytes(), np.asarray(t ** as_tensor).tobytes())
                                self.assertEqual(np.asarray(number ** t).tobytes(), np.asarray(as_tensor ** t).tobytes())
                                continue
                            self.assert_like_numpy(f(t, number), f(a, number))
                            self.assert_like_numpy(f(number, t), f(number, a))
                    for name, f in COMPARISONS.items():
                        if np.dtype(dtype).kind == "c" and name not in ("==", "!="):
                            continue
                        with self.subTest(dtype=np.dtype(dtype).name, number=number, op=name):
                            self.assert_like_numpy(f(t, number), f(a, number))
                            self.assert_like_numpy(f(number, t), f(number, a))

    def test_a_complex_power_is_numpys(self):
        # A complex power that is not by products (an exponent not a small integer) is the C library's cpow, which
        # NumPy calls: its bits under macOS (where libc++'s std::pow strayed from it for half the bases) and glibc;
        # MSVC's std::pow is its own, within 100 epsilons times the exponent's size.
        rng = np.random.default_rng(9)
        for dtype in (np.complex128, np.complex64):
            bases = (rng.standard_normal(4000) + 1j * rng.standard_normal(4000)).astype(dtype)
            t = numerical.asarray(bases)
            for exponent in (2.5, 0.1, -1.7, 1j, 0.3 + 0.2j, 150.0):
                with self.subTest(dtype=np.dtype(dtype).name, exponent=exponent):
                    with np.errstate(all="ignore"):
                        want = bases ** np.asarray(exponent, dtype)
                    got = np.asarray(t ** numerical.asarray(np.asarray(exponent, dtype)))
                    if sys.platform == "win32":  # an error relative to the exponent's size: 1.1e-13 to the 150th
                        limits = np.finfo(np.complex64 if dtype == np.complex64 else np.complex128)
                        finite = np.isfinite(got) & np.isfinite(want)
                        # relative to the exponent's size, absolute below the smallest normal number (a subnormal
                        # result keeps a few bits: one apart is a large fraction of it)
                        np.testing.assert_allclose(got[finite], want[finite], atol=limits.smallest_normal,
                                                   rtol=100 * limits.eps * max(1.0, abs(exponent)))
                        # one side past the largest number where the other is near it: a rounding apart
                        edge = np.where(np.isfinite(got), got, want)[~finite & (np.isfinite(got) | np.isfinite(want))]
                        self.assertTrue(np.all(np.maximum(abs(edge.real), abs(edge.imag)) > limits.max / 2))
                    else:
                        np.testing.assert_array_equal(got, want)

    def test_a_numpy_scalar_first_gives_a_tensor_of_numpys_dtype(self):
        # A NumPy scalar is of its own dtype (NumPy 2's strong scalars): first beside a Tensor it is read as a rank-0
        # Tensor and promoted, where its own operator answered a NumPy array.
        for scalar in (np.float64(2.5), np.float32(2.5), np.int64(3), np.complex128(1 - 2j), np.bool_(True)):
            for values in (np.array([1.0, -2.0, 4.0]), np.array([1.0, -2.0, 4.0], np.float32)):
                t = numerical.asarray(values)
                for name in ("add", "subtract", "multiply", "divide", "maximum", "minimum", "greater", "equal"):
                    if name in ("maximum", "minimum", "greater") and isinstance(scalar, np.complexfloating):
                        continue
                    with self.subTest(scalar=repr(scalar), dtype=t.dtype, function=name):
                        got, want = getattr(numerical, name)(scalar, t), getattr(np, name)(scalar, values)
                        self.assertIsInstance(got, numerical.Tensor)
                        self.assertEqual(got.dtype, str(want.dtype))
                        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-6)

    def test_a_comparison_reads_numpy_arrays_where_they_lie(self):
        # A comparison with a NumPy array operand, by the functions and a Tensor's operators: the Tensors' answer, for
        # each dtype (a complex one by == and != alone), broadcast, beside a Python number, NaNs included; a complex
        # order refused and shapes that do not broadcast as before.
        functions = {"<": numerical.less, "<=": numerical.less_equal, "==": numerical.equal,
                     "!=": numerical.not_equal, ">": numerical.greater, ">=": numerical.greater_equal}
        rng = np.random.default_rng(14)
        for dtype in (np.float64, np.float32, np.complex128, np.complex64):
            for left, right in (((3, 4), (3, 4)), ((3, 4), (4,)), ((1,), (5,)), ((), (6,)), ((2, 3), ())):
                a = np.round(rng.standard_normal(left) * 2).astype(dtype)
                b = np.round(rng.standard_normal(right) * 2).astype(dtype)
                if a.size:
                    a.reshape(-1)[0] = np.nan
                ta, tb = numerical.asarray(a), numerical.asarray(b)
                for name, f in functions.items():
                    if np.dtype(dtype).kind == "c" and name not in ("==", "!="):
                        with self.assertRaises(TypeError):
                            f(a, b)
                        continue
                    with self.subTest(dtype=np.dtype(dtype).name, left=left, right=right, op=name):
                        want = np.asarray(f(ta, tb))
                        for got in (f(a, b), f(ta, b), f(a, tb), COMPARISONS[name](ta, b)):
                            self.assertIsInstance(got, numerical.Tensor)
                            self.assertEqual(got.dtype, "bool")
                            np.testing.assert_array_equal(np.asarray(got), want)
                        for number in (0.5, 1, -2):
                            np.testing.assert_array_equal(np.asarray(f(a, number)), np.asarray(f(ta, number)))
                            np.testing.assert_array_equal(np.asarray(f(number, a)), np.asarray(f(number, ta)))
        with self.assertRaises(ValueError):
            numerical.less(np.ones(3), np.ones(4))
        self.assertIs(numerical._numerical._borrowed_compare("<", np.ones(2), np.ones(2, np.float32)), NotImplemented)
        self.assertIs(numerical._numerical._borrowed_compare("<>", np.ones(2), np.ones(2)), NotImplemented)

    def test_two_tensors_of_one_dtype_broadcast_as_numpy(self):
        rng = np.random.default_rng(8)
        for dtype in (np.float64, np.float32, np.complex128, np.int64):
            a = (rng.standard_normal((3, 1)) * 4).astype(dtype)
            b = (rng.standard_normal((1, 4)) * 4 + 5).astype(dtype)
            ta, tb = numerical.asarray(a), numerical.asarray(b)
            if np.dtype(dtype).kind == "i" and ta.dtype != "int64":  # a release without int64 arrays reads them as floats
                continue
            for name in ("+", "-", "*", "/"):
                with self.subTest(dtype=np.dtype(dtype).name, op=name):
                    self.assert_like_numpy(ARITHMETIC[name](ta, tb), ARITHMETIC[name](a, b))
            for name in ("==", "!="):
                with self.subTest(dtype=np.dtype(dtype).name, op=name):
                    self.assert_like_numpy(COMPARISONS[name](ta, tb), COMPARISONS[name](a, b))
        with self.assertRaises(ValueError):
            numerical.ones(3) + numerical.ones(4)
        with self.assertRaises(ValueError):
            numerical.ones(3) < numerical.ones(4)

    def test_the_operands_the_short_path_leaves_answer_as_before(self):
        # Two dtypes promote; a bool Tensor and an int64 one beside a number take the promoting path (an int there is
        # exact, past int64 compared as NumPy 2 compares it); a complex number promotes; a complex Tensor has no order,
        # floor division or remainder; an int no double holds overflows; anything else is NotImplemented.
        fast_binary, fast_compare = numerical._numerical._fast_binary, numerical._numerical._fast_compare
        t, single = numerical.asarray([1.0, 2.0]), numerical.asarray(np.array([1.0, 2.0], np.float32))
        mask = numerical.asarray([True, False])
        self.assertEqual((t + single).dtype, "float64")
        self.assertEqual((single + 2j).dtype, "complex64")
        self.assertEqual((mask + 1.5).dtype, "float64")
        np.testing.assert_array_equal(np.asarray(mask < 0.5), [False, True])
        with self.assertRaises(OverflowError):
            t + 10**400
        z = numerical.asarray([1 + 1j])
        with self.assertRaises(TypeError):
            z < 1.0
        with self.assertRaises(TypeError):
            z // 2.0
        for args in (("+", t, single), ("+", mask, mask), ("+", mask, 1.0), ("+", 1.0, 2.0), ("+", t, 2j), ("+", t, True),
                     ("+", t, [1.0]), ("@", t, t), ("+", [1.0], t)):
            self.assertIs(fast_binary(*args), NotImplemented)
        for args in (("<", t, single), ("<", mask, 1.0), ("<", 1.0, t), ("<", t, 2j), ("<", t, True), ("<", t, [1.0])):
            self.assertIs(fast_compare(*args), NotImplemented)
        if numerical.asarray(np.arange(2)).dtype == "int64":
            ints = numerical.asarray(np.array([2**60, -(2**60)]))
            np.testing.assert_array_equal(np.asarray(ints + 1), [2**60 + 1, -(2**60) + 1])
            np.testing.assert_array_equal(np.asarray(ints < 2**70), [True, True])


if __name__ == "__main__":
    unittest.main()
