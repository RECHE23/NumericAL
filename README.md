# NumericAL

A pure, header-only C++20 **numerical computing** library — a leaf of the
SciLang ecosystem (a sibling of [REAL](https://github.com/RECHE23/real-regex)),
generic over its scalar type and depending on nothing but the C++ standard
library.

NumericAL is **agnostic of expressions and of any symbolic layer**: it computes
with concrete numeric structures — vectors, matrices, tensors — and the
algorithms over them. SciLang consumes it as a numeric backend through its
module seam; the symbolic↔numeric reconciliation stays in SciLang.

The API reference, generated from the headers: <https://reche23.github.io/NumericAL/>.

## Quick start (C++)

Header-only: add `include/` to the include path, or after `cmake --install`, in your `CMakeLists.txt`:

```cmake
find_package(numerical CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE numerical::numerical) # C++20, and the threads library
```

then (this is `cmake/smoke/main.cpp`, which CI builds against the installed package):

```cpp
#include <cmath>
#include <complex>
#include <cstdio>

#include <numerical/numerical.hpp>

int main()
{
  const numerical::matrix<double> a {{4.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 2.0}};
  const numerical::vector<double> b {1.0, 2.0, 3.0};

  const numerical::vector<double>              x       {numerical::solve(a, b)}; // LU, partial pivoting
  const numerical::symmetric_eigen             eigen   {a};                      // A = V L V^T, as numpy.linalg.eigh
  const numerical::eigen_decomposition<double> general {a};                      // as numpy.linalg.eig
  const numerical::svd_decomposition<double>   svd     {a};                      // A = U S V^H

  std::printf("x = %.6f %.6f %.6f\n", x[0], x[1], x[2]);
  std::printf("eigh: %.6f %.6f %.6f\n", eigen.values()[0], eigen.values()[1], eigen.values()[2]);
  std::printf("eig: %.6f%+.6fi, its vectors %zu x %zu\n", general.values()[0].real(), general.values()[0].imag(),
              general.vectors().rows(), general.vectors().cols());
  std::printf("svd: %.6f %.6f %.6f\n", svd.singular_values()[0], svd.singular_values()[1], svd.singular_values()[2]);
  const numerical::vector<double> residual {(a * x) - b};
  return std::abs(residual[0]) + std::abs(residual[1]) + std::abs(residual[2]) < 1e-12 ? 0 : 1;
}
```

The Python package, `pip install numeric-al`, follows NumPy's API (see Python binding, below).

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
- `symmetric_eigen(a, b)`, `hermitian_eigen(a, b)` — the pencil `A x = λ B x`, `B` positive definite, as
  `scipy.linalg.eigh(a, b)` (LAPACK's `sygvd`): `B = L Lᴴ`, `L⁻¹ A L⁻ᴴ` diagonalised, the vectors `B`-orthonormal.
- `numerical::eigen_decomposition<T>` — the eigenvalues and right eigenvectors of a general real or complex
  square matrix, complex either way, as `numpy.linalg.eig` (LAPACK's `dgeev`/`zgeev`: balancing, blocked
  Hessenberg reduction, the multishift QR algorithm with aggressive early deflation); `values()`, `vectors()`,
  and `eigenvalues(a)` without the vectors (as `eigvals`).
- `numerical::schur_decomposition<T>` — `A = Z T Zᴴ`, as `scipy.linalg.schur` (LAPACK's `dgees`/`zgees`, from the
  same reduction and iterations): `t()` quasi-triangular for a real matrix (a pair a 2 × 2 block in standard form),
  triangular for a complex one, `z()` orthogonal or unitary; `values()` in `T`'s order; given a predicate on the
  eigenvalues, or by `sort(select)` and `reorder(chosen)` afterwards, those selected moved to the leading block
  (LAPACK's `trsen`, MATLAB's `ordschur`), `leading()` their count.
- `numerical::ldl_decomposition<T>` — the symmetric indefinite factorisation `A = L D Lᴴ` (or `L D Lᵀ`) of a
  symmetric or Hermitian matrix, as `scipy.linalg.ldl` (Bunch and Kaufman's pivoting, LAPACK's `sytf2`/`hetf2`; `D`
  with blocks of order 1 and 2), from either triangle.
- `numerical::pivoted_qr_decomposition<T>` — `A P = Q R` with column pivoting, as `scipy.linalg.qr(a, pivoting=True)`
  (LAPACK's `geqp3`), `R`'s diagonal revealing the rank.
- `numerical::solve_banded(l, u, ab, b)` and `numerical::solveh_banded(ab, b, lower)` — banded systems in LAPACK's
  diagonal-ordered form, as `scipy.linalg`'s (`gbsv`, `pbsv`); `numerical::solve_toeplitz(c, r, b)` (Levinson's
  recursion) and `numerical::solve_circulant(c, b)` (the Fourier transform); the matrices `toeplitz`, `hankel`,
  `circulant` and `block_diag`.
- `numerical::expm(a)` — the matrix exponential, as `scipy.linalg.expm` (Al-Mohy and Higham's scaling and squaring
  of a Padé approximant, 2009).
- `numerical::expm_frechet(a, e)` and `expm_cond(a)` — the Fréchet derivative of the exponential and its condition
  number, as `scipy.linalg`'s (Al-Mohy and Higham's algorithm, 2009, or the exponential of `[[A, E], [0, A]]`).
- `numerical::logm(a)` — the principal logarithm, as `scipy.linalg.logm` (inverse scaling and squaring, Al-Mohy and
  Higham 2012, on the Schur form; the norms exact where SciPy estimates them).
- `numerical::fractional_matrix_power(a, p)` — `A` to a real power, as `scipy.linalg.fractional_matrix_power` (Higham
  and Lin's Schur–Padé algorithm, 2013).
- `numerical::funm(a, f)` — a function of a matrix, as `scipy.linalg.funm`: Parlett's recurrence on the Schur form
  where the eigenvalues are well apart (SciPy's error estimate); where they gather, Davies and Higham's Schur–Parlett
  algorithm with each cluster by Cauchy's integral (`f` then also evaluated on circles about them); a Hermitian
  matrix by its eigendecomposition. `plan_funm` and `finish_funm` split it around the evaluation of `f`.
- `numerical::sqrtm(a)` — the principal square root, as `scipy.linalg.sqrtm` (the Schur method, real for a real
  matrix with no real eigenvalue below 0; `complex_sqrtm` the complex one of a real matrix).
- `numerical::solve_sylvester(a, b, q)` — `A X + X B = Q`, as `scipy.linalg.solve_sylvester` (Bartels and Stewart's
  method, the triangular equation by blocks as LAPACK's `trsyl`).
- `numerical::generalized_schur` — the generalized Schur decomposition of a complex pencil, `A = Q S Zᴴ` and
  `B = Q T Zᴴ` with `S` and `T` triangular, as `scipy.linalg.qz(a, b, output="complex")` (LAPACK's `zgges`: QZ
  iterations with single shifts), reordered as `ordqz` (`ztgsen`'s exchanges, refused where they would not keep the
  pencil triangular).
- `numerical::solve_continuous_are(a, b, q, r, e, s)` and `solve_discrete_are` — the stabilising solutions of the
  algebraic Riccati equations, as `scipy.linalg`'s (the extended pencil, balanced, its stable deflating subspace from
  the sorted generalized Schur form; with `e`, the balancing scales it as it scales the pencil, where SciPy 1.18 does
  not, scipy/scipy#26555).
- `numerical::svd_decomposition<T>` — the singular value decomposition `A = U Σ Vᴴ`
  of a real or complex `m × n` matrix, any shape: a Householder QR, then Golub–Kahan
  bidiagonalisation of `R` (by panels, for a real one) and divide and conquer, as LAPACK's `dgesdd`;
  implicit-shift QR (`svd_method::bidiagonal_qr`, `dgesvd`'s) or one-sided Jacobi on `R`
  (`svd_method::jacobi`, which gives every singular value to high relative accuracy) on request. `singular_values()` (descending), `u()` (`m × k`), `v()` (`n × k`) and
  `vh()`, with `k = min(m, n)` (`full = true`: `u()` `m × m` and `v()` `n × n`, the thin factors extended
  by an orthonormal basis of the complement); `U` and `V` stay orthonormal for a rank-deficient `A`.
- `numerical::solve_triangular(a, b, lower, op, unit_diagonal)` — `A X = B`, `Aᵀ X = B` or `Aᴴ X = B`
  (`triangular_op`) for a triangular `A`, as SciPy's `solve_triangular`: its lower or upper triangle read, the other
  never; `n² k` operations, where `solve` factorises first.
- `numerical::lstsq(a, b)` — the minimum-norm least-squares solution of `A x ≈ b`,
  as NumPy's `lstsq` gives it, for any shape and rank (from the SVD), `b` a vector or a
  matrix of right-hand sides (as `solve(a, b)` takes too); `least_squares(a, b)` adds the residuals, the
  rank and the singular values NumPy returns beside it; `pinv(a)` from the same SVD, `singular_values(a)`
  without the singular vectors (by dqds, as LAPACK's `dlasq1`: a bidiagonal's each to high relative accuracy; and
  `matrix_rank(a)` from them), `symmetric_eigenvalues(a)` without the
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
- `numerical::fft` (`fft.hpp`) — the discrete Fourier transform along an axis, as `numpy.fft`: `fft`, `ifft`, `rfft`,
  `irfft`, `hfft`, `ihfft` and `fftshift`, NumPy's points (cut or padded with zeros) and scalings (`fft_norm`). Any
  length: Stockham passes of radices 2 to 16 and every prime to 61, Bluestein's convolution past them; twiddle factors
  computed in double-double, so each is the correctly rounded root. Two sets of bits only, each the same on every
  machine and compiler: the portable path's (products rounded apart) and the vector paths' (NEON, AMX, AVX2 and
  AVX-512: one fused complex product).
- `dct`, `idct`, `dst`, `idst` (`fft.hpp`) — the cosine and sine transforms of `scipy.fft`, types 1 to 4, its norms and
  `orthogonalize`, by the real (types 1 to 3) or complex (type 4) transform: DCT-II by Makhoul's reordering, DCT-III
  its inverse, the sines from the cosines.
- Arithmetic on tensors (`tensor_ops.hpp`): `+ - * /` element by element with
  NumPy's broadcasting, `sum`, `mean`, `max`, `min` of every element or along an
  axis (sums by NumPy's pairwise blocks), `reshape`, and `zeros`, `ones`, `full`,
  `eye`, `arange`, `linspace`.

- `numerical::blas` (`blas.hpp`) — `dot`, `gemm` (of `float`, `double` and their complex numbers), `syrk`, `gemv`,
  `trsm`, `gesv`, `getrf`, `getrs`, `potrf` and `potrs` (the last five, as `gemm`, of `float`, `double` and their
  complex numbers), `syevd`, `gesdd`, `geqrf`, `orgqr`, `gelsd` and `geev` with the column-major semantics of BLAS's
  `ddot`, `dgemm`, `dsyrk`, `dgemv` and `dtrsm` and LAPACK's routines of those names; the `numerical_blas` shared
  library exports them as `dgemm_`, `dgesv_`… (Fortran calling convention, 32-bit integers), to stand in for those
  routines under a program linked against BLAS and LAPACK (`LD_PRELOAD`) or a switch such as FlexiBLAS. The Linux
  wheels carry it (see *BLAS and LAPACK for other programs*); CMake builds it with `-DNUMERICAL_BUILD_BLAS=ON`.

**Grows in as measured:** sparse storage, a real QZ, the rest of the BLAS/LAPACK interface, numeric autodiff,
iterative solvers, numerical integration, arbitrary precision.

## Build

Maintainers: the release procedure and the versioning rules are in `RELEASING.md`, beside this file.

```bash
make test        # build and run the test suite
make coverage    # line-coverage summary + HTML report
make sanitize    # tests under AddressSanitizer + UndefinedBehaviorSanitizer
make lint        # clang-tidy
make format      # uncrustify, in place
make doc         # API reference (Doxygen) with embedded coverage
```

Override the compiler with `make test CXX=g++-14`.

### Supported toolchains

A C++20 compiler is required. These are the toolchains each push is built and tested with; an older one may work,
and is not tested:

| Toolchain | Where |
| --- | --- |
| GCC 14 | Linux x86-64: the test suite, under ASan and UBSan too |
| Clang 18 | Linux x86-64: the test suite, under ASan and UBSan too |
| GCC 13 (Ubuntu 24.04's) | Linux arm64: the CMake build, tests and install |
| MSVC (Visual Studio 2022) | Windows x64: the CMake build, tests and install |
| Apple Clang 16 | macOS arm64: the local gate (`make check-push`), not CI |

CMake 3.16 or newer; Python 3.10 or newer for the binding. The x86-64 kernels are checked on an emulated AVX-512
processor in CI.

`numerical::numerical` is the CMake target — `add_subdirectory`, `FetchContent`, or an
installed config package:

```cmake
# After `cmake --install <build> --prefix <prefix>`:
find_package(numerical CONFIG REQUIRED)
target_link_libraries(app PRIVATE numerical::numerical)
```

## Python binding

```sh
pip install numeric-al        # the distribution is numeric-al; the module is numerical
```

A CPython binding (stable ABI, one `cp310` abi3 wheel serves 3.10+; a free-threaded 3.14t or 3.15t takes its own
wheel and runs without the GIL) exposes the
unifying object `numerical.Tensor` (rank-1 a vector, rank-2 a matrix, rank-0 a
scalar; `float64`, `complex128`, `float32`, `complex64`, or `bool` for a comparison's answer) and the algebra over it. Every function takes a
`Tensor` or anything `asarray` reads — a NumPy array, a nested list, a number:

```python
import numerical
a = [[1, 2], [3, 4]]
numerical.einsum("ij,jk->ik", a, [[5, 6], [7, 8]])   # matrix product
numerical.einsum("i,i->", [1, 2, 3], [1, 2, 3])       # dot
numerical.einsum("ij,jk", a, [[5, 6], [7, 8]])        # implicit output, as NumPy: "ij,jk->ik"
numerical.einsum("...ij,...jk->...ik", s, t)          # an ellipsis, its axes broadcast, as NumPy's
numerical.solve(a, [1.0, 1.0])                         # a @ x = b
numerical.fft.rfft([1.0, 2.0, 3.0, 4.0])               # numpy.fft's names: fft, ifft, fftn, rfft, irfft, fftfreq…
numerical.fft.dct([1.0, 2.0, 3.0, 4.0], norm="ortho")  # and scipy.fft's dct, idct, dst, idst, dctn…
```

A failure is a `numerical.LinAlgError`: a `ValueError`, as `numpy.linalg.LinAlgError` is,
and a `numerical.error`. A `Tensor` indexes (`t[i, j]`, `t[1:5:2, ::-1]`, `t[..., 0]`, `t[None, i]`), iterates, transposes
(`t.T`) and multiplies (`a @ b`, `numerical.matmul`) as a NumPy array does, stacks of matrices along leading axes
broadcast against each other, and computes as one:

```python
t = numerical.arange(6).reshape(2, 3)
(t + 1) * [10, 0, 1] / 2      # element-wise, broadcast as NumPy's
t.sum(), t.mean(axis=0), t.max(axis=-1), t.sum(axis=(0, 1), keepdims=True)
numerical.zeros((2, 2)), numerical.eye(3), numerical.linspace(0, 1, 5)
numerical.exp(t), numerical.sqrt(t), abs(t), t ** 2
mask = (t > 1) & ~numerical.isnan(t)                     # a bool Tensor, broadcast as NumPy's
numerical.where(mask, t, 0.0), mask.any(axis=0), mask.sum(), mask.astype("float64")
```

A comparison (`< <= == != > >=`, `numerical.less` and the rest) gives a `bool` Tensor, element by element, a NaN
unequal to everything, as NumPy's; a complex tensor compares with `==` and `!=` only. `&`, `|`, `^` and `~` take bool
tensors (`logical_and` and the rest any, read as NumPy reads them: nonzero is true), `where` picks between two
operands promoted as arithmetic promotes them, `isnan`, `isinf` and `isfinite` classify, and `any`, `all` reduce as
`sum` does. A mask beside a number takes the number's dtype (`mask * t`), and its `sum` and `mean` count its trues, as
NumPy's; two masks have no arithmetic, and neither has a mask in the linear algebra (a `TypeError` names
`astype("float64")`), as the array API leaves them: `asarray` still reads a NumPy bool array into float64, unless
`dtype="bool"` asks.

Shapes that do not broadcast raise a `ValueError` worded as NumPy's. `max` and `min` take
real dtypes only (NumPy orders complex numbers lexicographically; NumericAL does not order them); `max` gives +0 and `min`
−0 where both zeros are among the elements, as IEEE 754-2019 and NumPy on arm64 (NumPy on x86-64 keeps a zero's sign as
its loops meet them). Operands of two dtypes
promote as NumPy 2 promotes them: complex where one is, single precision only where both are, a Python number keeping
the tensor's precision (`float32 + 2.0` is float32). float32 and complex64 arithmetic is single precision, as NumPy's:
`+ - * /` and `**` by an integer give its very bits, complex ones included (each fused as the NumPy beside it
fuses it); a complex number's square root, `exp`, `log`, `sin`, `cos`, `tan` and `tanh` are glibc's, NumPy's on Linux to the bit
(macOS's and Windows' libraries, NumPy's there, stray from them among infinities and NaNs and past their last bits),
and its magnitude NumPy's own.

The decomposition layer is exposed too (a `numpy.linalg`-style surface):
`determinant(a)` (`det`), `slogdet(a)` → `(sign, logabsdet)`, `inverse(a)` (`inv`), `pinv(a, rcond, hermitian)`,
`matrix_rank(a, tol, hermitian)`, `eigvalsh(a, UPLO)`, `lstsq(a, b, rcond)` → `(x, residuals, rank, s)`,
`cholesky(a, upper=False)`, `ldl(A, lower, hermitian)` → `(lu, d, perm)` as `scipy.linalg.ldl`, `qr(a, mode="reduced")` → `(Q, R)` (`"complete"` as NumPy's, `"r"` → `R` alone),
`eigh(a, UPLO)` → `(values, vectors)` (with `b=`, a positive-definite matrix, the pencil `a x = λ b x` as
`scipy.linalg.eigh(a, b)`; `eigvalsh(a, b=b)` its values), `schur(a, output="real")` → `(T, Z)` as
`scipy.linalg.schur` (`"complex"` the triangular form of a real matrix; with `sort=` `"lhp"`, `"rhp"`, `"iuc"`,
`"ouc"` or a callable, → `(T, Z, sdim)`, the selected eigenvalues leading), `expm(a)`, `logm(a)` and `sqrtm(a)` as `scipy.linalg`'s (a
stack's matrices each), `expm_frechet(A, E, method, compute_expm)`, `expm_cond(A)`, `fractional_matrix_power(A, t)`, `funm(A, func, disp)`, `polar(a, side)`, `solve_banded((l, u), ab, b)`, `solveh_banded(ab, b, lower=False)`,
`solve_toeplitz(c_or_cr, b)`, `solve_circulant(c, b, singular)`, `toeplitz`, `hankel`, `circulant`, `block_diag`,
the special matrices `hilbert`, `invhilbert`, `pascal`, `invpascal`, `hadamard`, `helmert`, `leslie`, `companion`,
`fiedler`, `fiedler_companion`, `convolution_matrix`, `dft`, `khatri_rao` and `diagsvd` (float64 where SciPy's exact
ones are integers: each entry computed exactly, rounded once), `bandwidth`, `issymmetric`, `ishermitian`,
`null_space`, `orth`, `subspace_angles`, `orthogonal_procrustes`, `pinvh`, `cosm`, `sinm`, `tanm`, `coshm`, `sinhm`,
`tanhm`, `signm`, `hessenberg`, `matrix_balance`, `eigh_tridiagonal`, `eigvalsh_tridiagonal`, `cholesky_banded`,
`cho_solve_banded`, `rsf2csf`, `rq`, `cdf2rdf`, `qz` and `ordqz` (the real generalized Schur form, as `dgges`/`dtgsen`, or the
complex one),
`solve_sylvester(a, b, q)`, `solve_continuous_lyapunov(a, q)`,
`solve_discrete_lyapunov(a, q, method)`, `solve_continuous_are(a, b, q, r, e, s, balanced)` and
`solve_discrete_are(a, b, q, r, e, s, balanced)` as `scipy.linalg`'s, `eig(a)` → `(values, vectors)` and `eigvals(a)` of a general square matrix
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
to single (so they are as accurate as single precision holds, and the default tolerances are float32's epsilon, as
NumPy's), except that `solve`, `inv`, `cholesky`, `lu_factor` and `lu_solve` factorise a float32 matrix of order 500
or more in single precision, as NumPy's LAPACK does at every order: there the factorisation's time dominates and single
precision halves it, its answers backward stable at float32's precision. `eigh` of a complex Hermitian matrix gives real values and complex
vectors.
From `scipy.linalg`, with its signatures: `solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False)`;
`lu_factor(a)` → an `LUFactor`, the pair `(lu, piv)` SciPy returns (LAPACK's interchanges, a tuple of ints) with
`.solve(b)`, and `lu_solve((lu, piv), b, trans=0)`; `cho_factor(a, lower=False)` → a `CholeskyFactor`, `(c, lower)`,
and `cho_solve((c, lower), b)`: one factorisation, then `n² k` per solve. Their differences: a zero on `U`'s diagonal
raises in `lu_solve`, where SciPy's divides by it, and `cho_factor` leaves zeros in the triangle it does not use.
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
`trace(x, offset, dtype)`, `diagonal(x, offset)`, `matrix_transpose`, `outer`, `cross(x1, x2, axis)`,
`tensordot(a, b, axes)`, `tensorsolve(a, b, axes)`, `tensorinv(a, ind)`, `vecdot(x1, x2, axis)` (conjugating `x1`),
and `conj` (`conjugate`). A 2-norm is scaled by its largest magnitude, so it neither overflows nor
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
real dtype. `matmul` and `@` read a NumPy array where it lies, with no copy, when it is row-major, contiguous, aligned,
in this machine's byte order and of one of the four numeric dtypes, the other operand of the same dtype (a product of
order 512 by a vector: 231 → 18 µs); `solve`, `det`, `slogdet` and `inv` read a square float64 or complex128 array so
too, into the one copy their factorisation takes. Any other array is copied first, with the same answer. NumericAL makes
no views: `einsum("ij->ji", a)` returns a new tensor where NumPy returns a view of `a` (a copy, faster than NumPy's
copy of `a.T`), so writing to it never reaches `a`.

Compiled with GCC, the headers want `-ffp-contract=on` (Clang's default; the wheels and the sdist set it): GCC's own
default, `fast`, fuses multiply-adds across statements as its passes find them, so a loop's vectorised and scalar
versions, one or the other chosen at run time by the operands' addresses, can round an element differently, and the
same operands then give other last bits from one call to the next.

Compiled with Clang for arm64, the headers want `-mllvm -force-ordered-reductions=false` too (the wheels and the sdist set it
where the compiler takes it): Clang otherwise vectorises some loops that sum products in order, rounding their
products apart where the same loop inlined elsewhere fuses them, so a sum's last bits depend on its caller (a row of a
complex matrix times a vector against that row times the vector alone).

`numerical.get_include()` returns the header directory, so the C++ library can be
located through its Python install. `make python` builds the extension in place;
`make python-test` runs the binding suite — including a seeded **differential
fuzzer** that checks hundreds of random einsum specifications and linear-algebra
problems against NumPy within tolerance (when NumPy is importable, as in CI). PETSc (petsc4py) has no such exchange protocol — bridging it
would require an explicit copy into a PETSc object.

### BLAS and LAPACK for other programs

**Experimental.** The Linux wheels carry `numerical_blas`, a shared library of 39 BLAS and LAPACK routines under
their Fortran names: `?gemm_`, `?gesv_`, `?getrf_`, `?getrs_`, `?potrf_` and `?potrs_` in the four precisions,
`sdot_`, `ddot_`, `cdotu_`, `cdotc_`, `zdotu_` and `zdotc_`, and `dsyrk_`, `dgemv_`, `dtrsm_`, `dsyevd_`, `dgesdd_`,
`dgeqrf_`, `dorgqr_`, `dgelsd_` and `dgeev_` (gfortran's calling convention, a complex result returned by value);
each with 32-bit integers, and under its name ending in `64_` (`dgemm_64_`) with 64-bit ones, as OpenBLAS's ILP64
build names them. `numerical.blas_library()` gives its path (`None` on macOS and Windows,
where the wheels carry none: macOS's dyld binds each routine a program imports to the library that provides it,
which a library inserted by `DYLD_INSERT_LIBRARIES`, `DYLD_FORCE_FLAT_NAMESPACE` or not, does not replace; there,
CMake's build is linked by name, `-lnumerical_blas`). Preloaded, it answers those calls in a program linked against the system's BLAS and
LAPACK, which keep every other routine:

```bash
LD_PRELOAD=$(python -c "import numerical; print(numerical.blas_library())") python script.py
```

Julia reaches the library through libblastrampoline, which loads it by its path and forwards it the routines it has
(those it lacks stay OpenBLAS's). Twelve groups of Julia's LinearAlgebra tests pass with it on Linux, but for two that
compare `svd`'s vectors with `gesvd`'s column by column: they agree up to the columns' signs, which LAPACK leaves open.

```julia
using LinearAlgebra
BLAS.lbt_forward(path; clear=false, suffix_hint="64_")  # path: numerical.blas_library()
```

The program must call BLAS by those names: a NumPy or SciPy from the system's packages (Debian, Ubuntu, Fedora…) or
from FlexiBLAS, R, Octave, a Fortran or C program. NumPy and SciPy wheels from PyPI carry an OpenBLAS whose symbols
are renamed, which a preloaded library does not reach. Importing `numerical` never loads it.

Its routines run NumPy's and SciPy's linear algebra test suites in CI, and Julia's (`julia.yml`, weekly and on a
change to them); an argument is checked as the reference BLAS checks it (the reference XERBLA's message, then the call
returns). A NaN or an infinity, which NumericAL's own functions refuse, is taken as the reference LAPACK (3.12) takes
it, where it takes it one way: `getrf`, `gesv`, `geqrf`, `orgqr` and the solvers carry it through, `potrf` reports the
order its factorisation stops at, `syevd` gives NaN eigenvalues and `INFO` `n`. `gesdd`, `geev` and `gelsd` report the
matrix as an illegal argument (`INFO` `-4`), as the reference's `gesdd` and `geev` report a NaN; on an infinity the
reference loops without end or gives NaN, and on a NaN its `gelsd` fails inside.

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
on every kernel, and `make fuzz-edges` the element-wise operations and reductions on signed zeros, infinities, NaNs,
values near overflow and empty axes. On an M1 Pro against Accelerate, at order 1024, matrix products, `solve` and `inverse` are as
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

### Accuracy, function by function

The bounds the differential fuzzer (`make fuzz-linalg`, and every week `fuzz-long.yml`) holds every answer to, on
adversarial matrices (clustered, repeated and graded spectra, rank deficiency, exact integers, entries scaled to
1e±150, complex data) at every kernel the processor has. Throughout, `n` is the larger dimension, `ε = 2⁻⁵³`,
`‖·‖` the Frobenius norm and `c = 50 n ε`: a backward-stable algorithm's error is a small multiple of `n ε` whatever
the conditioning, and a wrong update or deflation misses it by orders of magnitude.

| Function | What holds |
| --- | --- |
| `eigh`, `eigvalsh` | `‖A V − V Λ‖ ≤ c ‖A‖`, `‖Vᴴ V − I‖ ≤ c`, eigenvalues ascending and within `c ‖A‖` of NumPy's; `UPLO="U"`, reading the upper triangle alone, the same bits as the lower triangle's |
| `svd`, `svdvals` | `‖U Σ Vᴴ − A‖ ≤ c ‖A‖`, `U` and `V` orthonormal within `c` (full and thin), values descending, non-negative, within `c ‖A‖` of NumPy's |
| `qr` | `‖Q R − A‖ ≤ c ‖A‖`, `‖Qᴴ Q − I‖ ≤ c`, `R` upper triangular; the complete factors extend the reduced ones to the bit |
| `cholesky` | `‖L Lᴴ − A‖ ≤ c ‖A‖` (and `upper=True`'s `Uᴴ U`) |
| `solve`, `inv` | backward error `‖A X − B‖ / (‖A‖ ‖X‖ + ‖B‖)` within `c`, or within 20 times NumPy's, below κ = 10¹⁰ |
| `det`, `slogdet` | the logarithm within `10⁻⁸ n · max(1, |log det|)` of NumPy's, the sign NumPy's |
| `lstsq`, `pinv`, `matrix_rank` | the solution within the perturbation bound of the rank kept (`c` times κ and κ² times the residual), the rank NumPy's |
| `eig`, `eigvals` | each pair's residual within `c ‖A‖` (on `D⁻¹ A D` for a graded matrix, as LAPACK balances it), unit vectors whose largest component is real; each of NumPy's well-conditioned eigenvalues within twice that bound times its condition number; complex conjugates paired to the bit |
| `norm`, `cond`, `svdvals`, products | within the rounding of their sums: `c` times the scale, `cond` (2-norm) within `c κ²` below κ = 10⁶; a 2-norm never overflows or underflows where its value is in range |
| `float32`, `complex64` | the `double` answer of the same values rounded to single, to the bit |
| any of them | the same bits on one thread as on all; a NaN or an infinity in what is read raises `LinAlgError` |

## License

MIT — see [LICENSE](LICENSE). Several algorithms follow LAPACK's routines closely; its BSD license, which their redistribution carries, is in [NOTICE](NOTICE).

## Author

René Chenard
