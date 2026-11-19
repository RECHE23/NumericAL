# The C++ library

`#include <numerical/numerical.hpp>` brings in the whole library, in namespace `numerical`; {doc}`../reference/cpp`
documents every class and the Doxygen reference every free function. This page says what is there and which
algorithm each part runs.

## Using it

The headers need a C++20 compiler and the threads library. `numerical::numerical` is the CMake target, through
`add_subdirectory`, `FetchContent`, or an installed config package:

```cmake
# After `cmake --install <build> --prefix <prefix>`:
find_package(numerical CONFIG REQUIRED)
target_link_libraries(app PRIVATE numerical::numerical)
```

`-DNUMERICAL_BUILD_BLAS=ON` also builds the `numerical_blas` shared library ({doc}`blas`). Under GCC, compile with
`-ffp-contract=on`, and under Clang for arm64 with `-mllvm -force-ordered-reductions=false`; {doc}`reproducibility`
says why.

## Supported toolchains

These are the toolchains each push is built and tested with; an older one may work, and is not tested:

| Toolchain | Where |
| --- | --- |
| GCC 14 | Linux x86-64: the test suite, under ASan and UBSan too |
| Clang 21 | Linux x86-64: the test suite, under ASan and UBSan too |
| GCC 13 (Ubuntu 24.04's) | Linux arm64: the CMake build, tests and install |
| MSVC (Visual Studio 2022) | Windows x64: the CMake build, tests and install |
| Apple Clang 16 | macOS arm64: the local gate (`make check-push`), not CI |

CMake 3.16 or newer; Python 3.10 or newer for the binding. The x86-64 kernels are checked on an emulated AVX-512
processor in CI.

## Containers

- `numerical::vector<T>`: a dense, value-semantic vector: element-wise arithmetic, scalar multiplication, dot
  product, tolerance comparison, shape checks.
- `numerical::matrix<T>`: a dense, row-major, value-semantic matrix: element-wise arithmetic, scalar,
  matrix–vector and matrix–matrix products, transpose, tolerance comparison, shape checks.
- `numerical::tensor<T>`: a dense, row-major, N-dimensional tensor (rank 1 a vector, rank 2 a matrix, rank 0 a
  scalar; built directly or from a `vector` or a `matrix`). `tensor::transposed(axes)` and `tensor::sliced` give
  views ({doc}`tensors`), `contiguous()` says whether `data()` copies.
- **`einsum`**: Einstein-summation index notation for contractions and general products, as in NumPy and PyTorch,
  the unifying primitive of the algebra: `"ij,jk->ik"` is the matrix product, `"i,i->"` the dot product, `"ij->ji"`
  the transpose, `"ii->"` the trace, `"i,j->ij"` the outer product, `"ij->i"` a row reduction. It takes one or more
  operands and sums over every label absent from the output, in a deterministic order.
- Arithmetic on tensors (`tensor_ops.hpp`): `+ - * /` element by element with NumPy's broadcasting; `sum`, `mean`,
  `max`, `min` of every element or along an axis (sums by NumPy's pairwise blocks); `reshape`; and `zeros`, `ones`,
  `full`, `eye`, `arange`, `linspace`.

## Factorisations and solvers

- `lu_decomposition<T>`: an LU factorisation (partial pivoting) computed once and reused: `solve` (any number of
  right-hand sides), `determinant`, `inverse`, and `log_determinant()` (NumPy's `slogdet`, which does not underflow
  where the determinant does). The free `solve`, `determinant` and `inverse` are one-shot conveniences.
- `cholesky_decomposition<T>`: `A = L Lᴴ` for a Hermitian positive-definite matrix (about twice as cheap as LU): the
  factor `lower()`, `solve` and `determinant`; it throws `not_positive_definite_error` otherwise.
- `ldl_decomposition<T>`: the symmetric indefinite factorisation `A = L D Lᴴ` (or `L D Lᵀ`) of a symmetric or
  Hermitian matrix, as `scipy.linalg.ldl` (Bunch and Kaufman's pivoting, LAPACK's `sytrf`/`hetrf`, by panels from
  order 384; `D` with blocks of order 1 and 2), from either triangle.
- `qr_decomposition<T>`: `A = Q R` by Householder reflections, so `Q` stays orthonormal to rounding however
  ill-conditioned `A` is, for an `m × n` matrix of any shape, as NumPy's reduced `qr`: `q()` (`m × k`) and `r()`
  (`k × n`), `k = min(m, n)` (`qr_mode::complete`: `q()` `m × m`, `r()` `m × n`, the reduced factors extended), and
  for `m ≥ n` a least-squares `solve` of `A x ≈ b`, which throws `singular_matrix_error` when `A` is rank-deficient.
- `pivoted_qr_decomposition<T>`: `A P = Q R` with column pivoting, as `scipy.linalg.qr(a, pivoting=True)` (LAPACK's
  `geqp3`), `R`'s diagonal revealing the rank.
- `solve_triangular(a, b, lower, op, unit_diagonal)`: `A X = B`, `Aᵀ X = B` or `Aᴴ X = B` (`triangular_op`) for a
  triangular `A`, as SciPy's: its lower or upper triangle read, the other never; `n² k` operations, where `solve`
  factorises first.
- `solve_banded(l, u, ab, b)` and `solveh_banded(ab, b, lower)`: banded systems in LAPACK's diagonal-ordered form, as
  `scipy.linalg`'s (`gbsv`, `pbsv`); `solve_toeplitz(c, r, b)` (Levinson's recursion) and `solve_circulant(c, b)`
  (the Fourier transform); the matrices `toeplitz`, `hankel`, `circulant` and `block_diag`.
- `lstsq(a, b)`: the minimum-norm least-squares solution of `A x ≈ b`, as NumPy's, for any shape and rank (from the
  SVD), `b` a vector or a matrix of right-hand sides (as `solve(a, b)` takes too); `least_squares(a, b)` adds the
  residuals, the rank and the singular values NumPy returns beside it; `pinv(a)` from the same SVD. `lstsq`,
  `least_squares`, `pinv` and `matrix_rank` take a `rank_cutoff`: NumPy's default, `rank_cutoff::relative(r)`
  (`rcond`, `rtol`) or `rank_cutoff::absolute(t)` (`tol`); `pinv(a, cutoff, true)` and `matrix_rank(a, cutoff, true)`
  take a Hermitian `a` by its eigendecomposition, its lower triangle alone read.

## Scalar types

The factorisations and solvers take a floating-point field (`numerical::field`: `float`, `double`, `long double` and
their complex forms), so that an integer matrix is refused at compile time. How far each carries its scalar:

- LU (`lu_decomposition`, `solve`), Cholesky and the unpivoted QR compute in the scalar itself: a `long double`
  matrix is factorised to `long double`'s precision.
- The SVD takes any field and computes in `double`: its singular values are `double`, and a `long double` matrix's
  factors hold a `double`'s precision. Where `long double` reaches past `double`'s range (x86-64's 80 bits), a
  matrix one of whose nonzero singular values lies outside `double`'s normal numbers is refused (`argument_error`)
  rather than answered 0 or ∞.
- The eigendecompositions (`eigen_decomposition`, `symmetric_eigen`, `hermitian_eigen`), the Schur form, the pivoted QR,
  LDLᴴ and the generalized forms are of `double` and `std::complex<double>`; the class templates among them take
  `numerical::double_field` and refuse another scalar by that name. A `float` matrix through the Python binding is
  computed in `double` and narrowed, below order 500; from order 500, `solve`, `inv`, `det`, `slogdet`, `cholesky`,
  `lu_factor` and `lu_solve` factorise it in `float` (see {doc}`linalg`).

## Eigenvalues and singular values

- `symmetric_eigen`: eigenvalues (ascending) and eigenvectors of a real symmetric matrix (`A = V Λ Vᵀ`):
  Householder tridiagonalisation by panels, then divide and conquer, as LAPACK's `dsyevd` (most of the work in matrix
  products); implicit QL (`eigen_method::tridiagonal_ql`, `dsyev`'s) or cyclic Jacobi rotations
  (`eigen_method::jacobi`) on request. `symmetric_eigenvalues(a)` gives the values alone (NumPy's `eigvalsh`).
- `hermitian_eigen`: the same for a complex Hermitian matrix (real eigenvalues, a unitary `V`), reduced to a real
  tridiagonal by Householder reflections and unit phases, then diagonalised by divide and conquer, as LAPACK's
  `zheevd`.
- `symmetric_eigen(a, b)`, `hermitian_eigen(a, b)`: the pencil `A x = λ B x`, `B` positive definite, as
  `scipy.linalg.eigh(a, b)` (LAPACK's `sygvd`): `B = L Lᴴ`, `L⁻¹ A L⁻ᴴ` diagonalised, the vectors `B`-orthonormal.
- `eigen_decomposition<T>`: the eigenvalues and right eigenvectors of a general real or complex square matrix, complex
  either way, as `numpy.linalg.eig` (LAPACK's `dgeev`/`zgeev`: balancing, blocked Hessenberg reduction, the
  multishift QR algorithm with aggressive early deflation); `values()`, `vectors()`, and `eigenvalues(a)` without the
  vectors (as `eigvals`).
- `schur_decomposition<T>`: `A = Z T Zᴴ`, as `scipy.linalg.schur` (LAPACK's `dgees`/`zgees`, from the same reduction
  and iterations): `t()` quasi-triangular for a real matrix (a pair a 2 × 2 block in standard form), triangular for a
  complex one, `z()` orthogonal or unitary; `values()` in `T`'s order; given a predicate on the eigenvalues, or by
  `sort(select)` and `reorder(chosen)` afterwards, those selected moved to the leading block (LAPACK's `trsen`,
  MATLAB's `ordschur`), `leading()` their count.
- `generalized_schur`: the generalized Schur decomposition of a complex pencil, `A = Q S Zᴴ` and `B = Q T Zᴴ` with `S`
  and `T` triangular, as `scipy.linalg.qz(a, b, output="complex")` (LAPACK's `zgges`: QZ iterations with single
  shifts), reordered as `ordqz` (`ztgsen`'s exchanges, refused where they would not keep the pencil triangular).
- `real_generalized_schur`: the generalized real Schur decomposition of a real pencil, `A = Q S Zᵀ` and `B = Q T Zᵀ`
  with `S` quasi-triangular (2 × 2 blocks for conjugate pairs) and `T` triangular, as `scipy.linalg.qz(a, b)`
  (LAPACK's `dgges`: the operands scaled into range, `dhgeqz`'s double shifts), reordered as `ordqz` (`dtgsen`'s
  exchanges of 1 × 1 and 2 × 2 blocks, refused as LAPACK refuses them).
- `generalized_eigenvectors(d, left)`: the right (or left) eigenvectors of a pencil from its generalized Schur form,
  complex or real (`real_generalized_schur`, its 2 × 2 blocks made triangular first), as LAPACK's `ztgevc` and
  `scipy.linalg.eig(a, b)` (unit 2-norm).
- `svd_decomposition<T>`: the singular value decomposition `A = U Σ Vᴴ` of a real or complex `m × n` matrix, any
  shape: a Householder QR, then Golub–Kahan bidiagonalisation of `R` (by panels, for a real one) and divide and
  conquer, as LAPACK's `dgesdd`; implicit-shift QR (`svd_method::bidiagonal_qr`, `dgesvd`'s) or one-sided Jacobi on
  `R` (`svd_method::jacobi`, which gives every singular value to high relative accuracy) on request.
  `singular_values()` (descending), `u()` (`m × k`), `v()` (`n × k`) and `vh()`, with `k = min(m, n)` (`full = true`:
  `u()` `m × m` and `v()` `n × n`, the thin factors extended by an orthonormal basis of the complement); `U` and `V`
  stay orthonormal for a rank-deficient `A`.
- `singular_values(a)`: the singular values without the vectors, by dqds as LAPACK's `dlasq1` (a bidiagonal's each to
  high relative accuracy), and `matrix_rank(a)` from them.

## Matrix functions and equations

- `expm(a)`: the matrix exponential, as `scipy.linalg.expm` (Al-Mohy and Higham's scaling and squaring of a Padé
  approximant, 2009); `expm_frechet(a, e)` and `expm_cond(a)`, its Fréchet derivative and condition number (Al-Mohy
  and Higham's algorithm, 2009, or the exponential of `[[A, E], [0, A]]`).
- `logm(a)`: the principal logarithm, as `scipy.linalg.logm` (inverse scaling and squaring, Al-Mohy and Higham 2012,
  on the Schur form; the norms exact where SciPy estimates them).
- `sqrtm(a)`: the principal square root, as `scipy.linalg.sqrtm` (the Schur method, real for a real matrix with no
  real eigenvalue below 0; `complex_sqrtm` the complex one of a real matrix).
- `fractional_matrix_power(a, p)`: `A` to a real power, as `scipy.linalg.fractional_matrix_power` (Higham and Lin's
  Schur–Padé algorithm, 2013).
- `funm(a, f)`: a function of a matrix, as `scipy.linalg.funm`: Parlett's recurrence on the Schur form where the
  eigenvalues are well apart (SciPy's error estimate); where they gather, Davies and Higham's Schur–Parlett algorithm
  with each cluster by Cauchy's integral (`f` then also evaluated on circles about them); a Hermitian matrix by its
  eigendecomposition. `plan_funm` and `finish_funm` split it around the evaluation of `f`.
- `solve_sylvester(a, b, q)`: `A X + X B = Q`, as `scipy.linalg.solve_sylvester` (Bartels and Stewart's method, the
  triangular equation by blocks as LAPACK's `trsyl`).
- `solve_continuous_are(a, b, q, r, e, s)` and `solve_discrete_are`: the stabilising solutions of the algebraic
  Riccati equations, as `scipy.linalg`'s (the extended pencil, balanced, its stable deflating subspace from the sorted
  generalized Schur form, real for real operands; with `e`, the balancing scales it as it scales the pencil, where
  SciPy 1.18 does not, scipy/scipy#26555).

## Fourier transforms

- `numerical::fft` (`fft.hpp`): the discrete Fourier transform along an axis, as `numpy.fft`: `fft`, `ifft`, `rfft`, `irfft`, `hfft`,
  `ihfft` and `fftshift`, NumPy's points (cut or padded with zeros) and scalings (`fft_norm`). Any length: Stockham
  passes of radices 2 to 16 and every prime to 61, Bluestein's convolution past them; twiddle factors computed in
  double-double, so each is the correctly rounded root. Two sets of bits only, each the same on every machine and
  compiler: the portable path's (products rounded apart) and the vector paths' (NEON, AMX, AVX2 and AVX-512: one fused
  complex product).
- `dct`, `idct`, `dst`, `idst`: the cosine and sine transforms of `scipy.fft`, types 1 to 4, its norms and
  `orthogonalize`, by the real (types 1 to 3) or complex (type 4) transform: DCT-II by Makhoul's reordering, DCT-III
  its inverse, the sines from the cosines.

## Building from source

The test suite, the binding and the lint tooling need SciForge checked out beside the repository;
[CONTRIBUTING.md](https://github.com/RECHE23/NumericAL/blob/main/CONTRIBUTING.md) gives the layout and the gates run
before each commit, push and release, and `RELEASING.md` the release procedure. The single steps:

```bash
make test             # build and run the test suite
make coverage         # line-coverage summary + HTML report
make sanitize         # tests under AddressSanitizer + UndefinedBehaviorSanitizer
make python           # the Python extension, in place
make python-test      # the binding's suite
make python-sanitize  # the Python suite under the sanitizers (Linux, GCC)
make lint             # clang-tidy
make format           # uncrustify, in place
make doc              # API reference (Doxygen) with embedded coverage
make help             # every other target
```

`make test CXX=g++-14` overrides the compiler. The binding's suite includes a seeded differential fuzzer that checks
hundreds of random `einsum` specifications and linear-algebra problems against NumPy within tolerance, when NumPy is
importable, as in CI.
