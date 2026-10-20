"""numerical_blas as the package carries it: located by blas_library(), its Fortran routines called through ctypes
as a BLAS caller would (every argument by address, column-major, 32-bit integers), never loaded by the import."""
import ctypes
import os
import subprocess
import sys
import unittest

import numerical

ROUTINES = ["ddot_", "dsyrk_", "dgemv_", "dtrsm_", "dsyevd_", "dgesdd_", "dgeqrf_", "dorgqr_", "dgelsd_", "dgeev_"] + [
    prefix + name for name in ("gemm_", "gesv_", "getrf_", "getrs_", "potrf_", "potrs_") for prefix in "sdcz"]

LINUX = sys.platform.startswith("linux")


def doubles(*values):
    return (ctypes.c_double * len(values))(*values)


@unittest.skipUnless(LINUX, "numerical_blas is built for Linux alone")
class BlasLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = numerical.blas_library()
        cls.lib = ctypes.CDLL(cls.path)

    def test_located_in_the_package(self):
        self.assertTrue(os.path.isabs(self.path))
        self.assertEqual(os.path.dirname(self.path), os.path.join(os.path.dirname(numerical.__file__), "lib"))

    def test_every_routine_exported(self):
        self.assertEqual(len(ROUTINES), 34)
        for name in ROUTINES:
            self.assertTrue(hasattr(self.lib, name), name)

    def test_ddot(self):
        ddot = self.lib.ddot_
        ddot.restype = ctypes.c_double
        n, one, two = ctypes.c_int(3), ctypes.c_int(1), ctypes.c_int(2)
        self.assertEqual(ddot(ctypes.byref(n), doubles(1, 2, 3), ctypes.byref(one), doubles(4, 0, 5, 0, 6),
                              ctypes.byref(two)), 32.0)

    def test_dgemm(self):
        # C = 2 Aᵀ B + C: A is 2 × 3 (so Aᵀ is 3 × 2), B 2 × 2, C 3 × 2, all column-major.
        a, b, c = doubles(1, 2, 3, 4, 5, 6), doubles(1, 0, 1, 1), doubles(1, 1, 1, 1, 1, 1)
        m, n, k = ctypes.c_int(3), ctypes.c_int(2), ctypes.c_int(2)
        alpha, beta = ctypes.c_double(2.0), ctypes.c_double(1.0)
        self.lib.dgemm_(b"T", b"N", ctypes.byref(m), ctypes.byref(n), ctypes.byref(k), ctypes.byref(alpha), a,
                        ctypes.byref(k), b, ctypes.byref(k), ctypes.byref(beta), c, ctypes.byref(m),
                        ctypes.c_size_t(1), ctypes.c_size_t(1))
        rows = [[1, 2], [3, 4], [5, 6]]  # Aᵀ
        want = [2 * (rows[i][0] * [1, 1][j] + rows[i][1] * [0, 1][j]) + 1 for j in range(2) for i in range(3)]
        self.assertEqual(list(c), want)

    def test_dgesv_and_a_singular_matrix(self):
        n, nrhs, info = ctypes.c_int(2), ctypes.c_int(1), ctypes.c_int(-7)
        pivots = (ctypes.c_int * 2)()
        a, b = doubles(4, 2, 1, 3), doubles(9, 11)  # [[4, 1], [2, 3]] x = [9, 11]
        self.lib.dgesv_(ctypes.byref(n), ctypes.byref(nrhs), a, ctypes.byref(n), pivots, b, ctypes.byref(n),
                        ctypes.byref(info))
        self.assertEqual(info.value, 0)
        self.assertAlmostEqual(b[0], 1.6, places=14)
        self.assertAlmostEqual(b[1], 2.6, places=14)
        singular = doubles(1, 2, 2, 4)
        self.lib.dgesv_(ctypes.byref(n), ctypes.byref(nrhs), singular, ctypes.byref(n), pivots, doubles(1, 1),
                        ctypes.byref(n), ctypes.byref(info))
        self.assertEqual(info.value, 2)

    def test_not_loaded_by_the_import(self):
        probe = ("import numerical, numerical._numerical\n"
                 "maps = open('/proc/self/maps').read()\n"
                 "print('libnumerical_blas' in maps, '_numerical' in maps)")
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True,
                             env={**os.environ, "LD_PRELOAD": ""}).stdout.split()
        self.assertEqual(out, ["False", "True"])


@unittest.skipIf(LINUX, "elsewhere the package carries no numerical_blas")
class NoBlasLibrary(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(numerical.blas_library())


if __name__ == "__main__":
    unittest.main()
