# Changelog

All notable changes to NumericAL. Versions are calendar-based (`YYYY.M.PATCH`, the patch resetting each
month; PEP 440 drops leading zeros). The project holds the SciLang-stack gate: 100%-4D coverage of
`include/`, clang and GCC, sanitizers, MISRA, and a NumPy differential.

## Unreleased

### Changed
- **`asarray` is 10–20× faster** on a NumPy array (1e6 float64: 11 → 0.8 ms; arm64, Apple clang 16, `-O2`): the
  element type is resolved once per copy rather than per element, the innermost axis is one strided loop (a
  contiguous run of the output's type one `memcpy`), and the GIL is released during the copy.
- **Element-wise arithmetic, powers and extremes at NumPy's pace** (1e6 float64, NumPy's time in brackets):
  `x * 2.0` 3.4 → 0.80 ms (0.63), `M + row` 3.4 → 0.84 (0.80), `x ** 2` 18 → 0.80 (0.69), `z ** 2` (complex)
  80 → 0.68 (0.52), `max` 2.5 → 0.24 (0.14), `min` along axis 0 2.4 → 0.34 (0.23). Broadcasting runs the
  innermost axis as one loop; a single exponent of 2, 0.5 or −1 is `x·x`, `sqrt(x)` or `1/x`, as NumPy computes
  them (so `(1+1j) ** 2` is exactly `2j`); `max`/`min` compare over native vectors.

- **The kernels share a pool of threads started once**, rather than starting threads on every call: each
  run hands its chunks out as threads free up (a slower core takes fewer), a run nested in another or started
  while the pool is busy runs on its caller, and results are the same to the bit for any thread count. Matrix
  product of order 128–256: 5.4–6.1× → 4.2–5.0× and 3.0× → 2.4× NumPy's time (arm64, Apple clang 16, `-O2`).

- **Matrix products use more of the cores**: both operands are packed in parallel (one thread packed them
  while the others waited) into buffers no longer zeroed first. M1 Pro, 8 threads, float64: order 512
  153 → 178 GFLOPS, 1024 195 → 207 (one core: 41, about 80 % of its vector peak).

### Added
- **`numerical.set_num_threads(n)` and `get_num_threads()`** (C++: `numerical::set_max_threads`,
  `max_threads`), and `NUMERICAL_NUM_THREADS` read at import, as `OMP_NUM_THREADS` caps BLAS: a process per core
  can keep the kernels to one thread each.

### Fixed
- **`max`/`min` along an empty axis raise**, as NumPy's, even when the result would be empty too
  (`min(zeros((3, 0, 0)), axis=1)` returned an empty tensor).

## 2026.10.3 — 2026-10-11

### Fixed
- **The sdist builds.** It carried no `include/` (no `MANIFEST.in`), so `pip install` on a platform without a
  wheel (free-threaded CPython, ppc64le, s390x, BSD…) failed on a missing header. CI now builds the sdist,
  installs it from source in a clean venv and runs the tests.
- **`lu_decomposition<long double>` compiles** (and `solve`, `determinant`, `inverse` with it): the pivot
  search held its magnitudes in a `double`, a narrowing for `long double` and `std::complex<long double>`.
  A test now instantiates every decomposition for `float`, `double`, `long double` and their complex forms,
  which compile cleanly under MSVC `/WX` and on x86-64, where `long double` is wider: every magnitude the
  library carries in `double` is converted to it explicitly, and a `complex<float>` is built from a `float`.
- **A boolean is not an index**: `t[True]` read row 1 and `t[False]` row 0 (`numpy.True_` too, before NumPy
  2.5), where NumPy reads a boolean as a mask. It raises `IndexError` now.
- **A Tensor's truth value and comparisons no longer answer wrongly**: `bool(t)` was `len(t) > 0`, and `a == b`
  compared identity. As NumPy's, `bool` gives the value of a one-element tensor and raises `ValueError` for
  any other size; `==`, `!=`, `<`, `<=`, `>`, `>=` raise `TypeError` until comparisons are element-wise, and a
  Tensor is unhashable. `numerical.array_equal(a, b)` compares two tensors whole, as NumPy's.
- **`arange` steps by the step its first two elements hold**, `(start + step) − start`, as NumPy's, and rounds
  each element once. The previous release said it equalled NumPy's to the bit; it differed by up to 1 280 units
  in the last place (`arange(1e16, 1e16 + 100, 1.0)`). It now equals NumPy's on arm64, where NumPy fuses the
  multiply-add; a NumPy build without fusion rounds twice and can differ in the last bits.

### Added
- **`repr(t)` shows the values**, one row per line, as NumPy's: `numerical.Tensor([[1.0, 2.0],` / `[3.0, 4.0]],
  dtype=float64)`; past 1000 elements each long axis keeps its first and last three entries around `...`.
- **`float(t)`, `int(t)`, `complex(t)` of a rank-0 tensor** (a full reduction along an axis of a vector, say),
  as NumPy's; any other rank raises `TypeError`.

### Changed
- **The extension is compiled without debug information** (the Linux wheel's was 7.8 MB with it).

## 2026.10.2 — 2026-10-11

### Fixed
- **An object that only claims to be a `Tensor` no longer crashes the interpreter.** `determinant`,
  `slogdet` and `asarray` trusted `isinstance`, which a mock with `spec=Tensor` passes, and read a tensor
  out of it (a segmentation fault). The object's real type is checked now; the impostor is refused.

### Added
- **A `Tensor` computes as a NumPy array**: `+ - * /` element by element with NumPy's broadcasting (numbers,
  lists and the reflected forms too; a real operand meeting a complex one is widened), `-t`, `t.sum()`,
  `t.mean()`, `t.max()`, `t.min()` of every element or along an `axis` (also as module functions),
  `t.reshape(…)` with one `-1`, and `numerical.zeros`, `ones`, `full`, `eye`, `arange`, `linspace`. Real
  arithmetic and creation equal NumPy's to the bit; sums follow NumPy's pairwise algorithm (its vector builds
  group terms their own way, so a long sum can differ in the last bits); a complex product or quotient agrees to
  a few units in the last place. `max` and `min` refuse complex128. C++: `tensor_ops.hpp`.
- **Element-wise functions**: `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `tanh`, `absolute` (`abs(t)`, float64 for
  either dtype), and `power` (`t ** p`, `p ** t`), as NumPy's to a few units in the last place. C++:
  `numerical::map` and `numerical::pow`.
- **Every function takes array-likes, as NumPy's do**: a NumPy array, a nested list or a number goes through
  `asarray` (zero-copy where it can); `nm.solve(np_array, [1.0, 2.0])` raised `TypeError: expected a
  numerical.Tensor`.
- **A matrix right-hand side for `solve` and `lstsq`** (`b` of shape `n × k`, one system per column), as
  NumPy's: one factorisation, then every column. C++: `solve(matrix, matrix)` and `lstsq(matrix, matrix)`.
- **`qr` of a wide matrix** (`m < n`), as NumPy's reduced `qr`: `Q` is `m × m`, `R` is `m × n` upper
  trapezoidal; `solve` of a wide `qr_decomposition` refuses its underdetermined system (`lstsq` solves it).
- **A real operand meets a complex one**: the real ones are widened to `complex128`, as NumPy promotes them
  (`einsum`, `solve` and `lstsq` refused operands of two dtypes).
- **A `Tensor` indexes, iterates, transposes and multiplies as a NumPy array**: `t[i]`, `t[i, j]`,
  `t[1:5:2, ::-1]` (basic indexing, negative indices and steps; all integers give a number), `len(t)`,
  `for row in t`, `t.T`, `a @ b` for vectors and matrices (two vectors give a number), `t[()]` for a rank-0
  tensor's number. A Tensor is never written through, so its indexed copies read as NumPy's views.
- **`eigh` of a complex Hermitian matrix**, as `numpy.linalg.eigh` takes it: real eigenvalues, complex
  eigenvectors (C++: `hermitian_eigen`). It refused anything but `float64`.
- **`LinAlgError`**, raised by every linear-algebra function where it cannot compute its answer (singular, not
  positive definite, non-finite, no convergence, wrong shape). It is a `ValueError`, as
  `numpy.linalg.LinAlgError` is, and a `numerical.error`, so existing `except numerical.error` keeps working.

### Changed
- **The matrix product is 1.3 to 3.2 times faster** (`a * b`, `einsum("ij,jk->ik", …)`, and the Python `@`
  and `einsum` over them): a packed, register-tiled kernel in the loop structure of BLIS (`gemm.hpp`), the tile
  held in native vectors for `float` and `double` under GCC and Clang. `double`, best of N: arm64 with 8 cores,
  Apple clang 16, `-O2`, 1.3× at 128 to 3.2× at 2048 (121 ms, 142 GFLOP/s); x86-64 AVX2 with 2 cores, GCC 14
  `-O2` without `-march`, 1.7× at 128 to 3.0× at 2048. Against BLAS (`np.matmul`) the gap goes from about 10×
  to about 3×. Every element is still summed in ascending order of the inner index, the order of the loop it
  replaces, on any number of threads; `a * b` and `einsum` now run the same code and agree to the bit. Products below
  8³ multiply-adds or shallower than 8 keep the plain loop.
- **The LU factorisation is 4.9 times faster at order 1024** (80.8 → 16.6 ms; 1.8× at 256, 3.0× at 512), and
  with it `determinant`, `slogdet`, `solve` and `inverse`: blocked as LAPACK's `dgetrf`, panels of 64 columns
  factorised in a column-major copy and the trailing block updated by the matrix product kernel. It regroups
  the unblocked algorithm's updates without reordering any, so without multiply-add fusion the factors are the
  same to the bit. Against LAPACK the gap goes from 6–13× to 2.4–3.5× (arm64 with 8 cores, Apple clang 16,
  `-O2`).
- **`inverse` is 5.2 times faster at order 1024** (241 → 46 ms; 3.3× at 512), and `solve` with a matrix
  right-hand side with it: every column is solved at once on the factors, block of rows by block of rows,
  through the matrix product kernel. A column agrees with the vector `solve` to rounding (the backward pass
  takes the farther blocks' terms first). Against LAPACK the gap goes from 11× to 2.2×. C++:
  `lu_decomposition::solve(matrix)`.
- **`svd` is 1.7 times faster again at order 1024** (1.67 → 1.01 s) and **`eigh` 1.2 times** (0.98 → 0.80 s):
  the singular vectors are formed from the bidiagonal reduction's reflectors by blocks through the matrix product
  kernel, and the rotations of the QR and QL iterations are recorded and applied once, in parallel by columns,
  each element receiving them in the same order. What remains of the gap to LAPACK (about 4× for both) is the
  unblocked bidiagonal and tridiagonal reductions.
- **`qr` is 3.9 times faster at order 1024** (393 → 101 ms; 2.7× at 512): blocked as LAPACK's `dgeqrf`, the
  reflectors of a panel of 32 columns applied to the rest at once as `I − V T Vᴴ` through the matrix product
  kernel, and `Q` formed the same way. Against LAPACK the gap goes from 6.3× to 1.7×.
- **`cholesky` is 6.2 times faster at order 1024** (139 → 22.5 ms; 1.9× at 256, 3.5× at 512): blocked on the
  same kernel, each panel's rows below its diagonal block in parallel; without multiply-add fusion the factor is
  the unblocked one to the bit. Against LAPACK the gap goes from about 12× to 1.8× at 1024.
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
