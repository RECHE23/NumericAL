# Temps NumPy/OpenBLAS par opération, aux tailles données, pour comparer au pilote C++.
import sys, time, numpy as np
for n in map(int, sys.argv[1:]):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n)); b = rng.standard_normal((n, n)); r = rng.standard_normal((n, 4))
    za = a + 1j * b; zb = b + 1j * a
    s = a.T @ a + n * np.eye(n); h = a + a.T; t = rng.standard_normal((2 * n, n)); y = rng.standard_normal(2 * n)
    ops = {"matmul": lambda: a @ b, "zmatmul": lambda: za @ zb, "solve": lambda: np.linalg.solve(a, r),
           "inverse": lambda: np.linalg.inv(a), "cholesky": lambda: np.linalg.cholesky(s), "qr": lambda: np.linalg.qr(a),
           "eigh": lambda: np.linalg.eigh(h), "svd": lambda: np.linalg.svd(a, full_matrices=False),
           "lstsq": lambda: np.linalg.lstsq(t, y, rcond=None)}
    for k, f in ops.items():
        f(); best = 1e9
        for _ in range(3):
            t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
        print(f"{k} n={n} : {best*1e3:.3f} ms")
