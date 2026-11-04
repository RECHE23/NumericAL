"""The integers' bit operations (bitwise_and, bitwise_or, bitwise_xor, invert, left_shift, right_shift, gcd, lcm,
bitwise_count and the operators & | ^ ~ << >>) and nonzero across chunks, against NumPy's answers: to the bit, on
int64's extremes, shifts by 64 places or more and negative counts; bools by their logical operations, a bool beside
an integer as an integer. Deterministic; the comparisons skipped without NumPy."""
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None

INT64_MIN, INT64_MAX = -2 ** 63, 2 ** 63 - 1


class WithoutNumPy(unittest.TestCase):
    def test_values(self):
        a = numerical.asarray([12, -1, INT64_MIN], "int64")
        b = numerical.asarray([10, 3, 1], "int64")
        self.assertEqual(numerical.bitwise_and(a, b).tolist(), [8, 3, 0])
        self.assertEqual((a | b).tolist(), [14, -1, INT64_MIN + 1])
        self.assertEqual((a ^ 1).tolist(), [13, -2, INT64_MIN + 1])
        self.assertEqual((~a).tolist(), [-13, 0, INT64_MAX])
        self.assertEqual((a << 1).tolist(), [24, -2, 0])
        self.assertEqual((a >> 70).tolist(), [0, -1, -1])
        self.assertEqual((2 << numerical.asarray([1, 64, -1], "int64")).tolist(), [4, 0, 0])
        self.assertEqual((1 >> numerical.asarray([0], "int64")).tolist(), [1])
        self.assertEqual(numerical.gcd(a, 0).tolist(), [12, 1, INT64_MIN])
        self.assertEqual(numerical.lcm([4, 0, -6], [6, 5, 4]).tolist(), [12, 0, 12])
        self.assertEqual(numerical.bitwise_count(a).tolist(), [2, 1, 1])
        self.assertEqual(numerical.bitwise_and([True, False], [True, True]).tolist(), [True, False])
        self.assertEqual(numerical.bitwise_xor(True, numerical.asarray([3], "int64")).tolist(), [2])
        self.assertEqual(numerical.invert([True, False]).tolist(), [False, True])
        self.assertEqual(numerical.invert(numerical.asarray([5], "int64")).tolist(), [-6])
        self.assertEqual(numerical.bitwise_not, numerical.invert)
        for call in (lambda: numerical.bitwise_and([1.5], [1]), lambda: numerical.invert([1.5]),
                     lambda: numerical.bitwise_count(numerical.ones(2)), lambda: numerical.ones(2) & 1,
                     lambda: numerical.gcd(1.0, 2)):
            with self.assertRaises(TypeError):
                call()


@unittest.skipIf(np is None, "NumPy is not installed")
class AgainstNumPy(unittest.TestCase):
    def assertSame(self, got, want, label):
        want = np.asarray(want)
        want = want.astype(np.int64) if want.dtype.kind in "iu" else want
        got = np.asarray(got)
        self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype), label)
        self.assertEqual(got.tobytes(), want.tobytes(), label)

    def test_against_numpy(self):
        rng = np.random.default_rng(1104)
        specials = np.array([INT64_MIN, INT64_MAX, -1, 0, 1, 63, 64, 65, -64, 2 ** 53 + 1], np.int64)
        for case in range(200):
            shape = tuple(int(e) for e in rng.integers(0, 4, int(rng.integers(0, 3))))
            a = rng.integers(-1000, 1000, shape).astype(np.int64)
            b = rng.integers(-70, 70, shape[-1:] if shape and rng.random() < 0.3 else shape).astype(np.int64)
            pick = rng.random(a.shape) < 0.3
            a[pick] = specials[rng.integers(0, len(specials), int(pick.sum()))]
            ta, tb = numerical.asarray(a, "int64"), numerical.asarray(b, "int64")
            with np.errstate(all="ignore"):
                for name in ("bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift", "gcd", "lcm"):
                    self.assertSame(getattr(numerical, name)(ta, tb), getattr(np, name)(a, b), f"{case} {name}")
                    self.assertSame(getattr(numerical, name)(tb, ta), getattr(np, name)(b, a), f"{case} {name} r")
                self.assertSame(~ta, ~a, f"{case} ~")
                self.assertSame(ta << tb, a << b, f"{case} <<")
                self.assertSame(ta >> 3, a >> 3, f"{case} >> 3")
                self.assertSame(ta & True, a & True, f"{case} & True")
                self.assertSame(numerical.bitwise_count(ta), np.bitwise_count(a), f"{case} count")
        bools = rng.random(9) < 0.5
        tb = numerical.asarray(bools, "bool")
        for name in ("bitwise_and", "bitwise_or", "bitwise_xor"):
            self.assertSame(getattr(numerical, name)(tb, tb[::-1]), getattr(np, name)(bools, bools[::-1]), name)
            self.assertSame(getattr(numerical, name)(tb, numerical.asarray(np.arange(9), "int64")),
                            getattr(np, name)(bools, np.arange(9)), f"{name} bool and int")
        self.assertSame(numerical.invert(tb), np.invert(bools), "invert bool")

    def test_nonzero(self):
        rng = np.random.default_rng(5)
        for shape in ((70001,), (3, 7, 5003), (2, 0, 4), (4, 9)):
            a = np.where(rng.random(shape) < 0.4, 0.0, rng.standard_normal(shape))
            got = numerical.nonzero(numerical.asarray(a))
            for g, w in zip(got, np.nonzero(a)):
                self.assertSame(g, w, f"nonzero {shape}")


if __name__ == "__main__":
    unittest.main()
