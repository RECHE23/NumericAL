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
  orthonormal to rounding however ill-conditioned `A` is) for an `m × n` matrix of any
  shape, as NumPy's reduced `qr`: the factors `q()` (`m × k`) / `r()` (`k × n`), `k = min(m, n)`
  (`qr_mode::complete`: `q()` `m × m`, `r()` `m × n`, the reduced factors extended), and for `m ≥ n` a least-squares `solve` of `A x ≈ b`, which throws `singular_matrix_error`
  when `A` is rank-deficient.
- `numerical::hermitian_eigen` — the same for a complex Hermitian matrix (real eigenvalues,
  a unitary `V`), reduced to a real tridiagonal by Householder reflections and unit phases, then
  diagonalised by divide and conquer, as LAPACK's `zheevd` does.
- `numerical::symmetric_eigen` — eigenvalues (ascending) and eigenvectors of a real
  symmetric matrix (`A = V Λ Vᵀ`): Householder tridiagonalisation by panels then divide and conquer,
  as LAPACK's `dsyevd` (most of the work in matrix products); implicit QL (`eigen_method::tridiagonal_ql`,
  `dsyev`'s) or cyclic Jacobi rotations (`eigen_method::jacobi`) on request.
- `numerical::svd_decomposition<T>` — the singular value decomposition `A = U Σ Vᴴ`
  of a real or complex `m × n` matrix, any shape: a Householder QR, then Golub–Kahan
  bidiagonalisation of `R` (by panels, for a real one) and divide and conquer, as LAPACK's `dgesdd`;
  implicit-shift QR (`svd_method::bidiagonal_qr`, `dgesvd`'s) or one-sided Jacobi on `R`
  (`svd_method::jacobi`, which gives every singular value to high relative accuracy) on request. `singular_values()` (descending), `u()` (`m × k`), `v()` (`n × k`) and
  `vh()`, with `k = min(m, n)` (`full = true`: `u()` `m × m` and `v()` `n × n`, the thin factors extended
  by an orthonormal basis of the complement); `U` and `V` stay orthonormal for a rank-deficient `A`.
- `numerical::lstsq(a, b)` — the minimum-norm least-squares solution of `A x ≈ b`,
  as NumPy's `lstsq` gives it, for any shape and rank (from the SVD), `b` a vector or a
  matrix of right-hand sides (as `solve(a, b)` takes too); `least_squares(a, b)` adds the residuals, the
  rank and the singular values NumPy returns beside it; `pinv(a)` from the same SVD, `singular_values(a)`
  without the singular vectors (and `matrix_rank(a)` from them), `symmetric_eigenvalues(a)` without the
  eigenvectors (as NumPy's `eigvalsh`), and `lu_decomposition::log_determinant()` —
  NumPy's `slogdet`, which does not underflow where the determinant does. `lstsq`, `least_squares`, `pinv` and
  `matrix_rank` take a `rank_cutoff`: NumPy's default, `rank_cutoff::relative(r)` (`rcond`, `rtol`) or
  `rank_cutoff::absolute(t)` (`tol`); `pinv(a, cutoff, true)` and `matrix_rank(a, cutoff, true)` take a Hermitian
  `a` by its eigendecomposition, its lower triangle alone read.
- `numerical::tensor<T>` — a dense, row-major, N-dimensional tensor (rank-1 is a
  vector, rank-2 a matrix, rank-0 a scalar; built directly or from a `vector` /
  `matrix`), with **`einsum`** — Einstein-summation index notation for
  contractions and general products, as in NumPy / PyTorch. `einsum` is the
  unifying primitive of the algebra: `"ij,jk->ik"` is the matrix product,
  `"i,i->"` the dot product, `"ij->ji"` the transpose, `"ii->"` the trace,
  `"i,j->ij"` the outer product, `"ij->i"` a row reduction. It takes one or more
  operands and sums over every label absent from the output, in a deterministic
  order.
- Arithmetic on tensors (`tensor_ops.hpp`): `+ - * /` element by element with
  NumPy's broadcasting, `sum`, `mean`, `max`, `min` of every element or along an
  axis (sums by NumPy's pairwise blocks), `reshape`, and `zeros`, `ones`, `full`,
  `eye`, `arange`, `linspace`.

**Grows in as measured:** sparse storage, non-symmetric eigenvalues, FFT, numeric autodiff, iterative
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
scalar; `float64`, `complex128`, `float32` or `complex64`) and the algebra over it. Every function takes a
`Tensor` or anything `asarray` reads — a NumPy array, a nested list, a number:

```python
import numerical
a = [[1, 2], [3, 4]]
numerical.einsum("ij,jk->ik", a, [[5, 6], [7, 8]])   # matrix product
numerical.einsum("i,i->", [1, 2, 3], [1, 2, 3])       # dot
numerical.einsum("ij,jk", a, [[5, 6], [7, 8]])        # implicit output, as NumPy: "ij,jk->ik"
numerical.einsum("...ij,...jk->...ik", s, t)          # an ellipsis, its axes broadcast, as NumPy's
numerical.solve(a, [1.0, 1.0])                         # a @ x = b
```

A failure is a `numerical.LinAlgError`: a `ValueError`, as `numpy.linalg.LinAlgError` is,
and a `numerical.error`. A `Tensor` indexes (`t[i, j]`, `t[1:5:2, ::-1]`), iterates, transposes
(`t.T`) and multiplies (`a @ b`, `numerical.matmul`) as a NumPy array does, stacks of matrices along leading axes
broadcast against each other, and computes as one:

```python
t = numerical.arange(6).reshape(2, 3)
(t + 1) * [10, 0, 1] / 2      # element-wise, broadcast as NumPy's
t.sum(), t.mean(axis=0), t.max(axis=-1)
numerical.zeros((2, 2)), numerical.eye(3), numerical.linspace(0, 1, 5)
numerical.exp(t), numerical.sqrt(t), abs(t), t ** 2
```

Shapes that do not broadcast raise a `ValueError` worded as NumPy's. `max` and `min` take
real dtypes only (NumPy orders complex numbers lexicographically; NumericAL does not order them). Operands of two dtypes
promote as NumPy 2 promotes them: complex where one is, single precision only where both are, a Python number keeping
the tensor's precision (`float32 + 2.0` is float32). float32 and complex64 arithmetic is single precision, as NumPy's:
`+ - * /` and `sqrt` give its very bits.

The decomposition layer is exposed too (a `numpy.linalg`-style surface):
`determinant(a)` (`det`), `slogdet(a)` → `(sign, logabsdet)`, `inverse(a)` (`inv`), `pinv(a, rcond, hermitian)`,
`matrix_rank(a, tol, hermitian)`, `eigvalsh(a, UPLO)`, `lstsq(a, b, rcond)` → `(x, residuals, rank, s)`,
`cholesky(a, upper=False)`, `qr(a, mode="reduced")` → `(Q, R)` (`"complete"` as NumPy's, `"r"` → `R` alone),
`eigh(a, UPLO)` → `(values, vectors)`, `eig(a)` → `(values, vectors)` and `eigvals(a)` of a general square matrix
(LAPACK's `dgeev`/`zgeev`: balancing, blocked Hessenberg reduction, multishift QR with aggressive early deflation;
complex answers whatever the eigenvalues, as NumPy 2.5's, where NumPy before 2.5 made them real where it could; the
eigenvalues in the Schur form's order, a real matrix's pairs conjugate to the bit; each vector of unit norm, its
largest component real),
`svd(a, full_matrices=True, compute_uv=True, hermitian=False)` → `(U, S, Vh)` as NumPy's (`full_matrices=False`:
the thin factors, so `(U * S) @ Vh` is `a`; `compute_uv=False`: `S` alone; `hermitian=True`: from the
eigendecomposition, the singular values the eigenvalues' magnitudes), `svdvals(a)`. The keywords mean what they mean to NumPy 2:
`UPLO="U"` and `upper=True` read the upper triangle, `rcond`/`rtol` cut the singular values relative to the
largest and `tol` absolutely, `hermitian=True` takes the eigendecomposition. Every one accepts the four dtypes
and answers in the input's: a float32 or complex64 matrix is decomposed in double precision and its answers rounded
to single, where NumPy's LAPACK computes in single (so they are as accurate as single precision holds, and the default
tolerances are float32's epsilon, as NumPy's). `eigh` of a complex Hermitian matrix gives real values and complex
vectors.
Every one but `lstsq` (2-D in NumPy too) takes a stack of matrices, `(..., m, n)`, as NumPy's: each matrix answers
as it does alone, to the bit, and the answers stack along the leading axes (`det` and `slogdet` give tensors, a stack's
`matrix_rank` nested integers); `solve` broadcasts the leading axes of `a` and `b` as NumPy 2 does, `b` a vector only
when it is 1-D. Small matrices, or as many as threads, go to the threads one each; larger ones one at a time, each
with the threads (10 000 inverses of order 4 on eight arm64 cores: 2.0 ms, NumPy 4.4 ms).
`eigh(a, method=...)` and `svd(a, method=...)` choose the algorithm: divide and conquer by default (LAPACK's
`dsyevd`/`zheevd`, `dgesdd`), `"tridiagonal_ql"` or `"bidiagonal_qr"` (`dsyev`'s, `dgesvd`'s), or `"jacobi"`.
The rest of `numpy.linalg` answers as NumPy 2's, on stacks too: `norm(x, ord, axis, keepdims)`, `vector_norm`,
`matrix_norm` (`"fro"`, `"nuc"`, 1, -1, 2, -2, ±inf), `cond(x, p)` (infinite for a singular matrix),
`matrix_power(a, n)` (`n` negative through the inverse), `multi_dot(arrays)` (the cheapest order of the chain),
`trace(x, offset, dtype)`, `outer`, `cross(x1, x2, axis)`, `tensordot(a, b, axes)`, `vecdot(x1, x2, axis)`
(conjugating `x1`), and `conj` (`conjugate`). A 2-norm is scaled by its largest magnitude, so it neither overflows nor
underflows: `vector_norm([1e200, 1e200])` is 1.414e200, where NumPy's is infinite.
A matrix holding a NaN or an infinity (in the triangle read, where only one is) raises
`numerical.LinAlgError`, where NumPy returns NaN factors or raises depending on the routine: a decomposition
of a non-finite matrix has no meaning, and saying so beats propagating NaN into every factor. A negative or NaN
tolerance raises too, where NumPy's `pinv` and `matrix_rank` return infinities or a rank of 0.

A `Tensor` is a citizen of the scientific Python stack, **zero-copy**:

```python
import numpy as np, torch
np.asarray(t)          # via __array_interface__ (a read-only view)
np.from_dlpack(t)      # via __dlpack__
torch.from_dlpack(t)   # the same DLPack capsule — JAX / CuPy consume it too
numerical.asarray(np.eye(3))   # the other direction (a copy)
```

`asarray` copies any NumPy array — bool, integer, float32/64 or complex64/128 elements, in either
byte order, at any strides (a transposed, sliced or reversed view) — as well as a nested list or
tuple of numbers and a plain number (rank 0): a float32, complex64 or complex128 array as it is, anything else into
float64 (complex128 for a complex list or number), or into the `dtype` asked; a complex input is never cast to a
real dtype.

Compiled with GCC, the headers want `-ffp-contract=on` (Clang's default; the wheels and the sdist set it): GCC's own
default, `fast`, fuses multiply-adds across statements as its passes find them, so a loop's vectorised and scalar
versions, one or the other chosen at run time by the operands' addresses, can round an element differently, and the
same operands then give other last bits from one call to the next.

`numerical.get_include()` returns the header directory, so the C++ library can be
located through its Python install. `make python` builds the extension in place;
`make python-test` runs the binding suite — including a seeded **differential
fuzzer** that checks hundreds of random einsum specifications and linear-algebra
problems against NumPy within tolerance (when NumPy is importable, as in CI). PETSc (petsc4py) has no such exchange protocol — bridging it
would require an explicit copy into a PETSc object.

## Threads and kernels

The kernels share a pool of threads, started the first time a kernel's work pays for more than one: one per
physical core the process may run on, where the system tells them (Linux, macOS; elsewhere one per hardware
thread), since two threads on one core's hyperthreads share its multiply-add units.
`numerical.set_num_threads(n)` (C++: `numerical::set_max_threads`), or `NUMERICAL_NUM_THREADS` read at import,
caps the threads a kernel uses, the caller's included, as `OMP_NUM_THREADS` caps BLAS; at 1 no thread is
started. A process forked while the pool exists runs its kernels alone. Each library that embeds NumericAL's
headers has a pool and a cap of its own.

`float64` and `complex128` matrix products, and the decompositions through them, run on a kernel chosen at run time, which
`numerical.gemm_path()` (C++: `gemm_path_in_use`) reports:

- `"portable"` — the compiler's own vectors, on any processor;
- `"neon"` — on arm64, its vectors with each multiply-add fused, whatever the compiler would contract (`float32`
  products too, there and on the AMX path, whose coprocessor takes `float64` alone);
- `"avx2"`, `"avx512"` — on x86-64, where `cpuid` reports the instructions and the system saves their registers
  (the wheels are built for the baseline x86-64 and carry these tiles beside it);
- `"amx"` — on Apple M1 to M4, the matrix coprocessor Apple's Accelerate uses, which Apple does not document; a
  product of few rows or columns, which it would compute as whole tiles of 32 × 16, runs on the NEON tile.
  It is used only on a known chip family, outside a virtual machine, and once a product computed on it equals
  `std::fma` in order to the bit; `NUMERICAL_AMX=0`, read by the Python package at import, keeps it off (its
  self-test included). C++ reads no environment: `limit_gemm_path` below `amx` before the first product does the
  same, and compiling with `NUMERICAL_NO_AMX` removes it.

`numerical.limit_gemm_path(name)` (C++: `limit_gemm_path`), or `NUMERICAL_GEMM` at import, caps the kernel at
`name` or a more portable one. Every element of a product is summed in ascending order of the depth on any
number of threads, so a result is the same, to the bit, whatever the thread count; across kernels it differs in
the last bits, since the NEON, AVX and AMX kernels fuse each multiply-add and the portable one rounds as the
compiler does (under the wheels' `-ffp-contract=on`, Clang fuses `c + a·b` there and GCC does not). Past the portable kernel, a complex product runs as one real product of twice the order, `[Re A | −Im A]`
times `[[Re B, Im B], [Im B, −Re B]]` (by blocks), on that kernel and its threads: each element then sums the
products of A's real parts, then those of its imaginary parts, where the portable kernel sums whole complex
products in order.

## Benchmarks

`make bench-linalg` times NumericAL against the BLAS/LAPACK NumPy ships, each library alone in its process,
every answer checked first; `make fuzz-linalg` checks every decomposition against NumPy on adversarial matrices,
on every kernel. On an M1 Pro against Accelerate, at order 1024, matrix products, `solve` and `inverse` are as
fast or faster (0.94×), Cholesky, QR and `eigh` within 1.0–1.2×, `svd` and `lstsq` 1.3–1.4×; small problems
(order 64–256) take 1.5–3× Accelerate's time. Against OpenBLAS on x86-64 (i9-10900X, i9-9900K) large problems
(order 512 and 1024) take 1.1–2.5×, `eigh` 2.6–2.9×. The tables, the method and where the gaps are:
[BENCHMARKS.md](https://github.com/RECHE23/NumericAL/blob/main/BENCHMARKS.md).

## Numerical discipline

Floating-point accuracy is documented, never claimed exact where it cannot be: every decomposition is backward
stable, checked against its defining property and against NumPy within bounds of the order times the machine
epsilon. A result is the same, to the bit, on any number of threads; between kernels and processors it may
differ in the last bits (see Threads and kernels). Same gate bar as the rest of the ecosystem: 100% coverage,
clang **and** g++-14, lint, format, MISRA-oriented analysis and sanitizers green.

## License

MIT — see [LICENSE](LICENSE).

## Author

René Chenard
