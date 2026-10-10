"""Differential parity: NumericAL must agree with NumPy within a tolerance.

A broad corpus of einsum specifications and linear solves is evaluated with both
NumericAL and NumPy on the same inputs; the results must match within a documented
floating-point tolerance (not bitwise — both accumulate rounding differently).
Skipped when NumPy is not importable; in CI the Python job installs it.
"""
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None

RTOL = 1e-9
ATOL = 1e-11


@unittest.skipIf(np is None, "NumPy not installed")
class EinsumParity(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260615)

    def assertEinsum(self, spec, *arrays):
        operands = [numerical.asarray(a) for a in arrays]
        got = np.asarray(numerical.einsum(spec, *operands))
        want = np.einsum(spec, *arrays)
        self.assertTrue(np.allclose(got, want, rtol=RTOL, atol=ATOL),
                        msg=f"{spec}: {got} != {want}")

    def test_real_specs(self):
        r = self.rng
        a, b = r.random(7), r.random(7)
        self.assertEinsum("i,i->", a, b)              # dot
        self.assertEinsum("i,j->ij", r.random(4), r.random(5))  # outer
        m = r.random((4, 5))
        self.assertEinsum("ij->ji", m)                # transpose
        self.assertEinsum("ij->", m)                  # sum all
        self.assertEinsum("ij->i", m)                 # row sums
        self.assertEinsum("ij->j", m)                 # column sums
        self.assertEinsum("ij,jk->ik", r.random((3, 4)), r.random((4, 6)))  # matmul
        self.assertEinsum("ij,j->i", m, r.random(5))  # matrix–vector
        self.assertEinsum("ij,ij->", m, r.random((4, 5)))  # Frobenius
        sq = r.random((5, 5))
        self.assertEinsum("ii->", sq)                 # trace
        self.assertEinsum("ii->i", sq)                # diagonal
        self.assertEinsum("ijk,kl->ijl", r.random((2, 3, 4)), r.random((4, 5)))  # batched
        self.assertEinsum("ij,jk,kl->il", r.random((2, 3)), r.random((3, 4)),
                          r.random((4, 2)))            # three operands

    def test_complex_specs(self):
        r = self.rng
        a = r.random(6) + 1j * r.random(6)
        b = r.random(6) + 1j * r.random(6)
        self.assertEinsum("i,i->", a, b)
        x = r.random((4, 4)) + 1j * r.random((4, 4))
        y = r.random((4, 4)) + 1j * r.random((4, 4))
        self.assertEinsum("ij,jk->ik", x, y)


@unittest.skipIf(np is None, "NumPy not installed")
class SolveParity(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(987654321)

    def assertSolve(self, a, b):
        got = np.asarray(numerical.solve(numerical.asarray(a), numerical.asarray(b)))
        want = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(got, want, rtol=1e-7, atol=1e-9),
                        msg=f"solve: {got} != {want}")

    def test_real_systems(self):
        for n in (2, 5, 16, 40):
            a = self.rng.random((n, n)) + n * np.eye(n)  # diagonally dominant
            self.assertSolve(a, self.rng.random(n))

    def test_complex_system(self):
        n = 6
        a = self.rng.random((n, n)) + 1j * self.rng.random((n, n)) + n * np.eye(n)
        b = self.rng.random(n) + 1j * self.rng.random(n)
        self.assertSolve(a, b)


if __name__ == "__main__":
    unittest.main()
