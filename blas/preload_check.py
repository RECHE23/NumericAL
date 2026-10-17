"""The numerical_blas library standing in for a system's BLAS and LAPACK: run by the system's NumPy and SciPy
(linked against libblas.so.3 and liblapack.so.3) with the library in LD_PRELOAD, its answers checked against their
defining properties, a non-positive-definite matrix still reported. Whether NumPy's calls reach the library is checked
apart, by the dynamic linker's bindings (the CI job's LD_DEBUG step)."""
import numpy as np
from scipy.linalg import solve_triangular

rng = np.random.default_rng(0)
a = rng.standard_normal((300, 300))
b = rng.standard_normal((300, 4))
s = a @ a.T + 300 * np.eye(300)
x = np.linalg.solve(a, b)
assert np.abs(a @ x - b).max() < 1e-9, "solve"
lower = np.linalg.cholesky(s)
assert np.abs(lower @ lower.T - s).max() < 1e-9, "cholesky"
assert np.abs(a @ a.T - np.einsum("ij,kj->ik", a, a)).max() < 1e-9, "matmul"
v = b[:, 0]
assert np.abs(a @ v - np.einsum("ij,j->i", a, v)).max() < 1e-9, "matrix times vector"
assert np.abs(v @ a - np.einsum("i,ij->j", v, a)).max() < 1e-9, "vector times matrix"
long = rng.standard_normal(100000)
assert abs(long @ long - np.einsum("i,i->", long, long)) < 1e-7, "dot"
values, vectors = np.linalg.eigh(s)
assert np.abs(s @ vectors - vectors * values).max() < 1e-9, "eigh"
assert np.abs(np.linalg.eigvalsh(s) - values).max() < 1e-9, "eigvalsh"
for full in (False, True):
    u, sigma, vh = np.linalg.svd(a[:, :200], full_matrices=full)
    assert np.abs((u[:, :200] * sigma) @ vh[:200] - a[:, :200]).max() < 1e-9, "svd"
assert np.abs(np.linalg.svd(a, compute_uv=False) - np.linalg.svd(a)[1]).max() < 1e-9, "singular values"
for mode in ("reduced", "complete"):
    q, r = np.linalg.qr(a[:, :200], mode=mode)
    assert np.abs(q @ r - a[:, :200]).max() < 1e-9, "qr"
    assert np.abs(q.T @ q - np.eye(q.shape[1])).max() < 1e-12, "qr orthonormal"
tall = a[:, :200]
x, residuals, rank, sigma = np.linalg.lstsq(tall, b, rcond=None)
assert np.abs(tall.T @ (tall @ x - b)).max() < 1e-9, "lstsq normal equations"
assert np.abs(residuals - ((tall @ x - b) ** 2).sum(axis=0)).max() < 1e-9, "lstsq residuals"
assert rank == 200 and np.abs(sigma - np.linalg.svd(tall, compute_uv=False)).max() < 1e-9, "lstsq rank"
deficient = tall[:, :50] @ rng.standard_normal((50, 120))
x, residuals, rank, _ = np.linalg.lstsq(deficient, b, rcond=None)
assert rank == 50 and residuals.size == 0, "lstsq deficient rank"
cutoff = np.finfo(float).eps * max(deficient.shape)  # lstsq's rcond=None: pinv's default keeps the rounding noise
assert np.abs(x - np.linalg.pinv(deficient, rcond=cutoff) @ b).max() < 1e-9, "lstsq minimum norm"
values, vectors = np.linalg.eig(a)
assert np.abs(a @ vectors - vectors * values).max() < 1e-9, "eig"
assert np.iscomplexobj(values) and np.abs(np.sort_complex(np.linalg.eigvals(a)) - np.sort_complex(values)).max() < 1e-9, "eigvals"
try:
    np.linalg.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
    raise AssertionError("an indefinite matrix was factorised")
except np.linalg.LinAlgError:
    pass
lower = np.tril(a) + 300 * np.eye(300)
for trans in (0, 1):
    x = solve_triangular(lower, b, lower=True, trans=trans)
    assert np.abs((lower.T if trans else lower) @ x - b).max() < 1e-9, "solve_triangular"
print("numerical_blas answers as the reference BLAS and LAPACK do")
