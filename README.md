# NumericAL

A pure, header-only C++20 **numerical computing** library — a leaf of the
SciLang ecosystem (a sibling of [REAL](https://github.com/RECHE23/real-regex)),
generic over its scalar type and depending on nothing but the C++ standard
library.

NumericAL is **agnostic of expressions and of any symbolic layer**: it computes
with concrete numeric structures — vectors, matrices, tensors — and the
algorithms over them. SciLang consumes it as a numeric backend through its
module seam; the symbolic↔numeric reconciliation stays in SciLang.

## Capabilities

Small and complete, grown measured, honest about what is not yet here:

- `numerical::vector<T>` — a dense, value-semantic vector: element-wise arithmetic,
  scalar multiplication, dot product, tolerance comparison, shape checks.
- `numerical::matrix<T>` — a dense, row-major, value-semantic matrix: element-wise
  arithmetic, scalar / matrix–vector / matrix–matrix products, transpose,
  tolerance comparison, shape checks.
- `numerical::lu_decomposition<T>` — an LU factorisation (partial pivoting) computed
  once and reused: `solve` (any number of right-hand sides), `determinant`, and
  `inverse`. Free `solve` / `determinant` / `inverse` are one-shot conveniences.
- `numerical::cholesky_decomposition<T>` — `A = L Lᴴ` for a Hermitian
  positive-definite matrix (about twice as cheap as LU): the factor `lower()`,
  `solve`, and `determinant`; throws `not_positive_definite_error` otherwise.
- `numerical::qr_decomposition<T>` — `A = Q R` (Householder reflections, so `Q` stays
  orthonormal to rounding however ill-conditioned `A` is) for an `m × n` matrix with
  `m ≥ n`: the factors `q()` / `r()` and a least-squares `solve` of `A x ≈ b`, which
  throws `singular_matrix_error` when `A` is rank-deficient.
- `numerical::symmetric_eigen` — eigenvalues (ascending) and eigenvectors of a real
  symmetric matrix by cyclic Jacobi rotations (`A = V Λ Vᵀ`).
- `numerical::svd_decomposition<T>` — the singular value decomposition `A = U Σ Vᴴ`
  of a real or complex `m × n` matrix, any shape: a Householder QR, then one-sided
  Jacobi on `R`, so every singular value comes to high relative accuracy, the small
  ones too. `singular_values()` (descending), `u()` (`m × k`), `v()` (`n × k`) and
  `vh()`, with `k = min(m, n)`; `U` and `V` stay orthonormal for a rank-deficient `A`.
- `numerical::lstsq(a, b)` — the minimum-norm least-squares solution of `A x ≈ b`,
  as NumPy's `lstsq` gives it, for any shape and rank (from the SVD); `pinv(a)` and
  `matrix_rank(a)` from the same SVD, and `lu_decomposition::log_determinant()` —
  NumPy's `slogdet`, which does not underflow where the determinant does.
- `numerical::tensor<T>` — a dense, row-major, N-dimensional tensor (rank-1 is a
  vector, rank-2 a matrix, rank-0 a scalar; built directly or from a `vector` /
  `matrix`), with **`einsum`** — Einstein-summation index notation for
  contractions and general products, as in NumPy / PyTorch. `einsum` is the
  unifying primitive of the algebra: `"ij,jk->ik"` is the matrix product,
  `"i,i->"` the dot product, `"ij->ji"` the transpose, `"ii->"` the trace,
  `"i,j->ij"` the outer product, `"ij->i"` a row reduction. It takes one or more
  operands and sums over every label absent from the output, in a deterministic
  order.

**Grows in as measured:** sparse storage, complex
Hermitian eigen, non-symmetric eigenvalues, FFT, numeric autodiff, iterative
solvers, numerical integration, arbitrary precision.

## Build

```bash
make test        # build and run the test suite
make coverage    # line-coverage summary + HTML report
make sanitize    # tests under AddressSanitizer + UndefinedBehaviorSanitizer
make lint        # clang-tidy
make format      # uncrustify, in place
make doc         # API reference (Doxygen) with embedded coverage
```

Override the compiler with `make test CXX=g++-14`.

`numerical::numerical` is the CMake target — `add_subdirectory`, `FetchContent`, or an
installed config package:

```cmake
# After `cmake --install <build> --prefix <prefix>`:
find_package(numerical CONFIG REQUIRED)
target_link_libraries(app PRIVATE numerical::numerical)
```

## Releasing

`make release` computes the next calendar version `YYYY.M.PATCH` — the patch
resets each month, the first release of a month is `.0` (PEP 440 drops leading
zeros, so `2026.6.1`, never `2026.06.001`) — bumps it in `pyproject.toml` and
`python/numerical/__init__.py`, then commits, tags and pushes from a clean `main`.
Pushing the tag drives `release.yml`, which checks the tag matches the version,
builds the abi3 wheels (Linux x86-64/aarch64, macOS universal, Windows) and the
sdist, and publishes to PyPI via Trusted Publishing (OIDC, no stored secret).

One-time PyPI setup (before the first release): create a
[Trusted Publisher](https://docs.pypi.org/trusted-publishers/) for the project
`numeric-al` — owner `RECHE23`, repository `NumericAL`, workflow `release.yml`,
environment `pypi` — and create the matching GitHub Environment named `pypi`.
Until that exists the build/sdist jobs still run, but the publish step fails;
the pushed tag remains a valid versioned snapshot regardless.

## Python binding

```sh
pip install numeric-al        # the distribution is numeric-al; the module is numerical
```

A CPython binding (stable ABI, one `cp310` abi3 wheel serves 3.10+) exposes the
unifying object `numerical.Tensor` (rank-1 a vector, rank-2 a matrix, rank-0 a
scalar; `float64` or `complex128`) and the algebra over it:

```python
import numerical
a = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
b = numerical.Tensor([5, 6, 7, 8], shape=(2, 2))
numerical.einsum("ij,jk->ik", a, b)             # matrix product
numerical.einsum("i,i->", numerical.Tensor([1, 2, 3]), numerical.Tensor([1, 2, 3]))  # dot
numerical.solve(a, numerical.Tensor([1.0, 1.0]))   # a @ x = b
```

The decomposition layer is exposed too (a `numpy.linalg`-style surface):
`determinant(a)`, `slogdet(a)` → `(sign, logabsdet)`, `inverse(a)`, `pinv(a)`,
`matrix_rank(a)`, `lstsq(a, b)`, `cholesky(a)`, `qr(a)` → `(Q, R)`,
`eigh(a)` → `(values, vectors)`, `svd(a)` → `(U, S, Vh)` — NumPy's
`svd(a, full_matrices=False)`, so `(U * S) @ Vh` is `a`. Every one accepts `float64`
and `complex128` but `eigh`, which is `float64` only (matching the C++ layer).

A `Tensor` is a citizen of the scientific Python stack, **zero-copy**:

```python
import numpy as np, torch
np.asarray(t)          # via __array_interface__ (a read-only view)
np.from_dlpack(t)      # via __dlpack__
torch.from_dlpack(t)   # the same DLPack capsule — JAX / CuPy consume it too
numerical.asarray(np.eye(3))   # the other direction (a contiguous-array copy)
```

`numerical.get_include()` returns the header directory, so the C++ library can be
located through its Python install. `make python` builds the extension in place;
`make python-test` runs the binding suite — including a seeded **differential
fuzzer** that checks hundreds of random einsum specifications and linear-algebra
problems against NumPy within tolerance (when NumPy is importable, as in CI). PETSc (petsc4py) has no such exchange protocol — bridging it
would require an explicit copy into a PETSc object.

## Benchmarks

NumericAL is validated against **NumPy** for both result and speed — `make
bench-python` checks every operation matches NumPy within a tolerance, then times
it; `make python-test` runs the correctness parity suite alone (also in CI). The
results and methodology, with an honest reading of the gap, are in
[BENCHMARKS.md](BENCHMARKS.md): NumericAL is a portable, dependency-free reference
implementation: correct within tolerance, and competitive with NumPy's own
`einsum`/`linalg` (it beats `np.einsum`'s non-BLAS path and the decompositions
land within ~1–4×). Against hand-tuned BLAS (`np.matmul`) its cache-blocked,
`std::thread`-parallel GEMM is ~10× slower but steady (no cache cliff) — that gap
is BLAS's per-architecture SIMD micro-kernel, which a header-only library does not
chase. See [BENCHMARKS.md](BENCHMARKS.md) for the honest details.

## Numerical discipline

Floating-point accuracy is documented, never claimed exact where it cannot be;
results are deterministic given inputs and any tolerance/seed; differential
tests compare against references **within a documented tolerance**. Same gate
bar as the rest of the ecosystem: 100% coverage, clang **and** g++-14, lint,
format, and sanitizers green.

## License

MIT — see [LICENSE](LICENSE).

## Author

René Chenard
