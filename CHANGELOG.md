# Changelog

All notable changes to NumericAL. Versions are calendar-based (`YYYY.M.PATCH`, the patch resetting each
month; PEP 440 drops leading zeros). The project holds the SciLang-stack gate: 100%-4D coverage of
`include/`, clang and GCC, sanitizers, MISRA, and a NumPy differential.

## Unreleased

### Added
- **Every function takes array-likes, as NumPy's do**: a NumPy array, a nested list or a number goes through
  `asarray` (zero-copy where it can); `nm.solve(np_array, [1.0, 2.0])` raised `TypeError: expected a
  numerical.Tensor`.
- **`LinAlgError`**, raised by every linear-algebra function where it cannot compute its answer (singular, not
  positive definite, non-finite, no convergence, wrong shape). It is a `ValueError`, as
  `numpy.linalg.LinAlgError` is, and a `numerical.error`, so existing `except numerical.error` keeps working.

### Changed
- **`svd` is 48 times faster** (512 × 512: 11.2 → 0.24 s; 256 × 256: 1.0 → 0.031 s; arm64, Apple clang 16,
  `-O2`): after the Householder QR, `R` is reduced to bidiagonal form and diagonalised by implicit-shift QR,
  the method of LAPACK's `dgesvd`; the reconstruction error of a 512 × 512 falls from 6e-13 to 5e-14. One-sided
  Jacobi, which gives every singular value to high relative accuracy, stays on request and is itself 6.6 times
  faster (its sweeps rotate contiguous rows). C++: `svd_decomposition` takes the method as a second parameter
  (`svd_method::bidiagonal_qr`, the default, or `svd_method::jacobi`). `lstsq`, `pinv` and `matrix_rank` go
  through the default.
- **`eigh` is 100 times faster and more accurate** (512 × 512: 14.6 s → 0.12 s; 256 × 256: 785 → 14 ms; arm64,
  Apple clang 16, `-O2`): the matrix is reduced to tridiagonal form by Householder reflections, then diagonalised
  by implicit QL, the method of LAPACK's `dsyev`. Cyclic Jacobi took O(n³) per sweep and stopped at 1e-12 of the
  matrix norm; the residual `|AV − VΛ|` of a 256 × 256 falls from 4e-13 to 5e-14. C++: `symmetric_eigen`'s
  second parameter is now the method (`eigen_method::tridiagonal_ql`, the default, or `eigen_method::jacobi`);
  it was a sweep bound.
- **QR is 9 to 14 times faster** (512 × 512: 630 → 44 ms; 256 × 256: 48 → 5.6 ms; arm64, Apple clang 16, `-O2`):
  each Householder reflection walked the row-major matrix column by column; it now walks it row by row, on
  contiguous memory. `qr`, `lstsq`, `pinv`, `matrix_rank` and `svd` (which starts with a QR) all use it.

## 2026.10.1 — 2026-10-11

### Added
- **`asarray` reads what NumPy users pass it**: bool, integer, float32 and complex64 arrays (as
  float64 or complex128), in either byte order, at any strides (`a.T`, `a[::-1, ::2]`, Fortran order),
  nested lists and tuples of numbers, and plain numbers. It refused everything but C-contiguous float64
  and complex128.
- **`einsum(spec, operands)` over a `std::vector` of operands** counted at run time, for a caller that
  holds them in a container (SciLang's interpreter called `detail::einsum_impl` for this).
- **`slogdet`, `pinv` and `matrix_rank`**, as `numpy.linalg` gives them (C++: `lu_decomposition::
  log_determinant`, `pinv`, `matrix_rank`). `determinant` of `1e-2·I` of order 200 underflows to 0.0;
  its `slogdet` is `(1.0, -921.03)`.

### Changed
- **SciForge v2026.10.2**: CI pins and the `sciforge-build` floor (was 2026.6.x), so an isolated wheel
  build compiles against the hardened binding headers.
- **`svd` returns `(U, S, Vh)`, as NumPy's `svd(a, full_matrices=False)` does** (Python; C++ adds
  `svd_decomposition::vh()`). The third factor was `V`, so NumPy's idiom `(U * S) @ Vh` reconstructed a
  wrong matrix without an error. Code that transposed the third factor itself must stop doing so.
- **The SVD takes any shape and a complex matrix, and keeps its small singular values.** It was
  the eigendecomposition of `Aᵀ A`, which squares the condition number: true singular values
  `[1, 1e-4, 1e-8, 1e-12]` came back as `[…, 9.7e-9, 0]`, and `U` lost its orthonormality where
  `A` was rank-deficient. It is now a Householder QR then one-sided Jacobi on `R`, which gives each
  singular value to high relative accuracy; `U` (`m × k`) and `V` (`n × k`), `k = min(m, n)`, stay
  orthonormal at any rank. `svd_decomposition` is a class template (`svd_decomposition<T>`, deduced
  from its argument).
- **QR is computed by Householder reflections.** Modified Gram–Schmidt lost orthogonality on an
  ill-conditioned matrix (Hilbert-10: `|QᵀQ − I|` 2.2e-4) and refused Hilbert-12 as "linearly
  dependent", which now factorises with `|QᵀQ − I|` 1.8e-15. A rank-deficient matrix factorises too
  (with a zero on `R`'s diagonal); `qr(a).solve(b)` refuses it, and `lstsq` solves it.
- **`lstsq` gives NumPy's minimum-norm least-squares solution, for any shape and rank**, from the
  SVD. Through QR it lost accuracy on a polynomial fit (Vandermonde of degree 11: a coefficient off
  by 0.60; the test's degree-11 fit now recovers each within 1e-9) and refused a rank-deficient or
  wide system.
- **`eigh` reads the lower triangle, as `numpy.linalg.eigh` does by default**; it read the upper one
  of a matrix that is not symmetric.
- **A large computation releases the GIL** (from 64 × 64 elements): `solve`, `inverse`, `lstsq`,
  `cholesky`, `qr`, `eigh`, `svd`, `determinant` and `einsum` run in parallel from several Python
  threads.
- **`einsum` over three operands or more contracts them two at a time**, after summing each over the
  labels only it carries: `ij,jk,kl->il` of order 60 takes 0.11 ms instead of 29 (arm64, Apple clang 16,
  `-O3`), as it ran one loop over every label. `i,j,k,l,m->` over five vectors of 65 536 ones gives 2⁸⁰
  exactly, where it raised a loop-count overflow.
- **A parallel loop starts as many threads as its work pays for.** It started one per hardware thread
  as soon as it had 16 or 64 indices, whatever their work: a 16 × 16 `inverse` took 177 µs instead of
  8, a 64 × 64 matrix product 95 µs instead of 35.

### Fixed
- **Every decomposition refuses an infinite or NaN entry** (`non_finite_error`), named after the
  decomposition asked for. `eigh` of `[[1, nan], [nan, 2]]` returned `[1, 2]`.
- **An exception in a parallel worker reaches the caller** once every worker has joined; it called
  `std::terminate`.
- **`asarray` refuses an `__array_interface__` whose data address is null** while the shape has
  elements, and one whose `data` tuple is empty; both crashed the interpreter.
- **`einsum` no longer wraps its count of terms**: five summed labels of 65 536 wrapped it to 0 and
  returned 0.0 (the count is checked, and such a spec now sums each operand alone first).
- **The Jacobi methods bound their sweeps** (`convergence_error`) instead of looping without end.

## 2026.10.0 — 2026-10-10

First release under the name NumericAL (formerly SciNum): dense vectors, matrices and tensors with
`einsum`, LU, Cholesky, QR, symmetric eigen and SVD decompositions, and a zero-copy Python binding
(`pip install numeric-al`, `import numerical`).
