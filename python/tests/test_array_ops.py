"""Element-wise arithmetic, reductions, reshaping and creation of Tensors against NumPy's own, over every pair
of a set of shapes that broadcast (and the pairs that do not), both dtypes, every axis. Deterministic; skipped
when NumPy is not importable (CI installs it)."""
import itertools
import math
import operator
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None

SHAPES = [(), (1,), (3,), (2, 1), (1, 3), (2, 3), (4, 1, 3), (1, 2, 1), (0,), (2, 0)]
OPERATORS = [operator.add, operator.sub, operator.mul, operator.truediv]


@unittest.skipIf(np is None, "NumPy not installed")
class Arithmetic(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261012)

    def array(self, shape, cplx=False):
        a = self.rng.uniform(0.5, 2.0, shape)
        return a + 1j * self.rng.uniform(0.5, 2.0, shape) if cplx else a

    def test_every_pair_of_shapes_broadcasts_as_numpy(self):
        for (sa, sb), op, (ca, cb) in itertools.product(itertools.product(SHAPES, SHAPES), OPERATORS,
                                                       [(False, False), (False, True), (True, True)]):
            a, b = self.array(sa, ca), self.array(sb, cb)
            with self.subTest(a=sa, b=sb, op=op.__name__, complex=(ca, cb)):
                try:
                    want = op(a, b)
                except ValueError:
                    with self.assertRaises(ValueError):
                        op(numerical.asarray(a), numerical.asarray(b))
                    continue
                got = op(numerical.asarray(a), numerical.asarray(b))
                self.assertEqual(got.shape, want.shape)
                self.assertEqual(got.dtype, str(want.dtype))
                if cb and op in (operator.mul, operator.truediv):
                    # A complex product or quotient is the C++ library's (C's Annex G scaling), NumPy's its own
                    # formula (Smith's for the quotient): they agree to a few units in the last place.
                    np.testing.assert_allclose(np.asarray(got), want, rtol=4 * np.finfo(float).eps, atol=0)
                else:
                    np.testing.assert_array_equal(np.asarray(got), want)

    def test_numbers_lists_and_reflected_operators(self):
        t = numerical.asarray([[1.0, 2.0], [3.0, 4.0]])
        a = np.asarray(t)
        for got, want in [(t + 1, a + 1), (1 + t, 1 + a), (2 - t, 2 - a), (t * [10, 0], a * [10, 0]),
                          ([1.0, 2.0] / t, [1.0, 2.0] / a), (t * 1j, a * 1j), (-t, -a), (+t, +a)]:
            np.testing.assert_array_equal(np.asarray(got), want)

    def test_an_unreadable_operand_is_not_implemented(self):
        with self.assertRaises(TypeError):
            numerical.asarray([1.0]) + "text"
        with self.assertRaises(TypeError):
            "text" - numerical.asarray([1.0])

    def test_a_broadcast_failure_is_a_value_error_and_a_numerical_error(self):
        with self.assertRaises(numerical.error):
            numerical.ones((2, 3)) + numerical.ones(4)


@unittest.skipIf(np is None, "NumPy not installed")
class Reductions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_every_axis_as_numpy(self):
        for shape in [(5,), (3, 4), (2, 3, 4), (1, 5, 1)]:
            for cplx in (False, True):
                a = self.rng.standard_normal(shape) + (1j * self.rng.standard_normal(shape) if cplx else 0)
                t = numerical.asarray(a)
                for axis in [None] + list(range(-len(shape), len(shape))):
                    names = ["sum", "mean"] + ([] if cplx else ["max", "min"])
                    for name in names:
                        with self.subTest(shape=shape, axis=axis, op=name, complex=cplx):
                            got = getattr(numerical, name)(t, axis)
                            want = getattr(np, name)(a, axis=axis)
                            np.testing.assert_allclose(np.asarray(got), want, rtol=1e-13, atol=1e-13)
                            method = getattr(t, name)(axis=axis)  # the method is the function
                            np.testing.assert_array_equal(np.asarray(method), np.asarray(got))

    def test_a_long_sum_is_pairwise(self):
        tenths = numerical.full(10 ** 6, 0.1)
        self.assertEqual(numerical.sum(tenths), np.sum(np.full(10 ** 6, 0.1)))

    def test_nan_wins_and_empty_extremes_are_refused(self):
        t = numerical.asarray([1.0, float("nan"), 3.0])
        self.assertTrue(math.isnan(numerical.max(t)) and math.isnan(numerical.min(t)))
        with self.assertRaises(ValueError):
            numerical.max(numerical.zeros(0))
        with self.assertRaises(ValueError):
            numerical.sum(numerical.ones((2, 3)), axis=2)
        with self.assertRaises(TypeError):
            numerical.max(numerical.ones(2, dtype="complex128"))


@unittest.skipIf(np is None, "NumPy not installed")
class ShapesAndCreation(unittest.TestCase):
    def test_reshape_as_numpy(self):
        a = np.arange(24.0)
        t = numerical.asarray(a)
        for shape in [(24,), (2, 12), (2, 3, 4), (4, -1), (-1, 2, 3), (1, 24, 1)]:
            with self.subTest(shape=shape):
                np.testing.assert_array_equal(np.asarray(numerical.reshape(t, shape)), a.reshape(shape))
                np.testing.assert_array_equal(np.asarray(t.reshape(*shape)), a.reshape(shape))
        self.assertEqual(t.reshape(24).shape, (24,))
        for bad in [(5,), (5, -1), (-1, -1), (0, -1)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                numerical.reshape(t, bad)

    def test_creation_as_numpy(self):
        cases = [(numerical.zeros((2, 3)), np.zeros((2, 3))), (numerical.ones(4), np.ones(4)),
                 (numerical.zeros(2, dtype="complex128"), np.zeros(2, dtype=complex)),
                 (numerical.full((2,), 2.5), np.full((2,), 2.5)), (numerical.full(3, 1j), np.full(3, 1j)),
                 (numerical.eye(3), np.eye(3)), (numerical.arange(5), np.arange(5.0)),
                 (numerical.arange(1, 2, 0.25), np.arange(1, 2, 0.25)), (numerical.arange(3, 0, -1), np.arange(3.0, 0, -1)),
                 (numerical.linspace(0, 1, 7), np.linspace(0, 1, 7)), (numerical.linspace(-2, 5), np.linspace(-2, 5)),
                 (numerical.linspace(0, 1, 1), np.linspace(0, 1, 1)), (numerical.linspace(0, 1, 0), np.linspace(0, 1, 0))]
        for got, want in cases:
            with self.subTest(want=want):
                self.assertEqual(got.dtype, str(want.dtype))
                np.testing.assert_array_equal(np.asarray(got), want)

    def test_creation_refuses_what_numpy_refuses(self):
        for call in [lambda: numerical.zeros(-1), lambda: numerical.eye(-2), lambda: numerical.arange(0, 1, 0),
                     lambda: numerical.linspace(0, 1, -1), lambda: numerical.zeros(2, dtype="int32")]:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


@unittest.skipIf(np is None, "NumPy not installed")
class ElementwiseFunctions(unittest.TestCase):
    """The C++ library's functions against NumPy's own implementations: equal to a few units in the last
    place, not to the bit (NumPy computes some of them with its own vector code)."""

    def test_functions_and_power_as_numpy(self):
        rng = np.random.default_rng(11)
        real = rng.uniform(0.1, 3.0, (4, 5))
        cplx = real + 1j * rng.uniform(-2.0, 2.0, (4, 5))
        for a in (real, cplx):
            t = numerical.asarray(a)
            for name, reference in [("exp", np.exp), ("log", np.log), ("sqrt", np.sqrt), ("sin", np.sin),
                                    ("cos", np.cos), ("tan", np.tan), ("tanh", np.tanh),
                                    ("absolute", np.abs)]:
                with self.subTest(function=name, dtype=a.dtype):
                    got = getattr(numerical, name)(t)
                    want = reference(a)
                    self.assertEqual(got.dtype, str(want.dtype))
                    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-14, atol=1e-15)
            with self.subTest(power=a.dtype):
                np.testing.assert_allclose(np.asarray(t ** 2.5), a ** 2.5, rtol=1e-14)
                np.testing.assert_allclose(np.asarray(1.5 ** t), 1.5 ** a, rtol=1e-14)
                np.testing.assert_allclose(np.asarray(numerical.power(t, t)), np.power(a, a), rtol=1e-14)
                np.testing.assert_allclose(np.asarray(abs(t)), np.abs(a), rtol=1e-15)

    def test_outside_the_real_domain_is_nan_as_numpy(self):
        t = numerical.asarray([-1.0, 0.0])
        self.assertTrue(math.isnan(numerical.sqrt(t)[0]) and math.isnan(numerical.log(t)[0]))
        self.assertEqual(numerical.log(t)[1], -math.inf)


if __name__ == "__main__":
    unittest.main()
