"""The same bits from AVX2 and AVX-512: every function's answer under limit_gemm_path("avx2") and under "avx512",
byte for byte, on a processor with AVX-512 (skipped elsewhere): the kernels of both units sum in one order."""
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None or numerical.gemm_path() != "avx512", "an AVX-512 processor and NumPy needed")
class SameBitsOnBothUnits(unittest.TestCase):
    def test_every_function(self):
        r = np.random.default_rng(31)
        for n in (37, 200):
            a, b = r.standard_normal((n, n)), r.standard_normal((n, n))
            s, p = a + a.T, a @ a.T + n * np.eye(n)
            c = a + 1j * b
            h, v, f32 = c + c.conj().T, r.standard_normal(n), a.astype(np.float32)
            tall, wide = r.standard_normal((3 * n, n // 2)), r.standard_normal((n // 2, 3 * n))
            cases = {
                "matmul": lambda: numerical.matmul(a, b), "matmul f32": lambda: numerical.matmul(f32, f32),
                "matmul complex": lambda: numerical.matmul(c, c), "A @ x": lambda: numerical.matmul(a, v),
                "A.T @ x": lambda: numerical.matmul(a.T, v), "vecdot": lambda: numerical.vecdot(v, v),
                "f32 A @ x": lambda: numerical.matmul(f32, v.astype(np.float32)),
                "f32 A.T @ x": lambda: numerical.matmul(f32.T, v.astype(np.float32)),
                "solve": lambda: numerical.solve(a, v), "inv": lambda: numerical.inv(a), "det": lambda: numerical.det(a),
                "lu tall": lambda: numerical.lu(tall)[1], "lu wide": lambda: numerical.lu(wide)[2],
                "cholesky": lambda: numerical.cholesky(p), "qr": lambda: numerical.qr(a)[1],
                "eigh": lambda: numerical.eigh(s)[1], "eigvalsh": lambda: numerical.eigvalsh(s),
                "eigh complex": lambda: numerical.eigh(h)[1], "svd": lambda: numerical.svd(a)[0],
                "eig": lambda: numerical.eig(a)[1], "eig(a, b)": lambda: numerical.eig(a, b)[1],
                "qz": lambda: numerical.qz(a, b)[0], "hessenberg": lambda: numerical.hessenberg(a),
                "schur": lambda: numerical.schur(a)[0], "expm": lambda: numerical.expm(a / n),
                "logm": lambda: numerical.logm(p), "sqrtm": lambda: numerical.sqrtm(p),
                "fractional_matrix_power": lambda: numerical.fractional_matrix_power(p, 0.3),
                "ldl": lambda: numerical.ldl(s)[0], "pinv": lambda: numerical.pinv(tall),
                "norm 2": lambda: numerical.matrix_norm(a, ord=2),
            }
            answers = {}
            try:
                for path in ("avx2", "avx512"):
                    numerical.limit_gemm_path(path)
                    answers[path] = {name: np.ascontiguousarray(np.asarray(f())).tobytes() for name, f in cases.items()}
            finally:
                numerical.limit_gemm_path("amx")  # the best path again
            for name in cases:
                with self.subTest(n=n, function=name):
                    self.assertEqual(answers["avx2"][name], answers["avx512"][name])


if __name__ == "__main__":
    unittest.main()
