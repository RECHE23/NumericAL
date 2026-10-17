"""The numerical_blas library standing in for a system's BLAS and LAPACK: run by the system's NumPy (linked against
libblas.so.3 and liblapack.so.3) with the library in LD_PRELOAD, its answers checked against their defining
properties, a non-positive-definite matrix still reported. Whether NumPy's calls reach the library is checked
apart, by the dynamic linker's bindings (the CI job's LD_DEBUG step)."""
import numpy as np

rng = np.random.default_rng(0)
a = rng.standard_normal((300, 300))
b = rng.standard_normal((300, 4))
s = a @ a.T + 300 * np.eye(300)
x = np.linalg.solve(a, b)
assert np.abs(a @ x - b).max() < 1e-9, "solve"
lower = np.linalg.cholesky(s)
assert np.abs(lower @ lower.T - s).max() < 1e-9, "cholesky"
assert np.abs(a @ a.T - np.einsum("ij,kj->ik", a, a)).max() < 1e-9, "matmul"
values, vectors = np.linalg.eigh(s)
assert np.abs(s @ vectors - vectors * values).max() < 1e-9, "eigh"
assert np.abs(np.linalg.eigvalsh(s) - values).max() < 1e-9, "eigvalsh"
for full in (False, True):
    u, sigma, vh = np.linalg.svd(a[:, :200], full_matrices=full)
    assert np.abs((u[:, :200] * sigma) @ vh[:200] - a[:, :200]).max() < 1e-9, "svd"
assert np.abs(np.linalg.svd(a, compute_uv=False) - np.linalg.svd(a)[1]).max() < 1e-9, "singular values"
try:
    np.linalg.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
    raise AssertionError("an indefinite matrix was factorised")
except np.linalg.LinAlgError:
    pass
print("numerical_blas answers as the reference BLAS and LAPACK do")
