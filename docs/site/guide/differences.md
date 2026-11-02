# Differences from NumPy and SciPy

NumericAL answers as NumPy 2 and SciPy do wherever it carries their names. This page lists where a program written
for them meets something else: a default chosen between the two, a refusal where they compute, a dtype or an
operation that is not there yet. The differential fuzzers (`make fuzz-linalg`, `make fuzz-scipy`, `make fuzz-edges`)
hold everything else to NumPy's and SciPy's answers.

## Names that NumPy and SciPy both use

One namespace carries both libraries' names. Where they disagree on a default, NumericAL takes NumPy's and accepts
SciPy's arguments:

| Function | NumericAL, as NumPy | SciPy's default | The SciPy call |
| --- | --- | --- | --- |
| `cholesky(a)` | the lower factor `L` | the upper factor `U` | `cholesky(a, lower=False)` |
| `qr(a)` | the reduced factors | the full factors | `qr(a, mode="full")`, `mode="economic"` |
| `eigh(a, b)` | — | the pencil `a x = λ b x` | the same: a matrix in NumPy's `UPLO` place is SciPy's `b` |
| `lstsq(a, b)` | `rcond` as NumPy's, the residue of a vector `b` a one-element tensor | `cond`, the residue a number | `lstsq(a, b, cond=…)` answers as SciPy's |
| `svd(a)` | the full factors, as both | — | `lapack_driver="gesvd"` takes `gesvd`'s algorithm |

A port from SciPy that relies on `cholesky`'s or `qr`'s default changes its answer without a word: pass the argument
explicitly.

## Refusals where NumPy and SciPy compute

- **A NaN or an infinity** in a matrix a decomposition reads (in the triangle read, where only one is) raises
  {class}`numerical.LinAlgError`, where NumPy returns NaN factors or raises depending on the routine.
- **A negative or NaN tolerance** raises, where NumPy's `pinv` and `matrix_rank` return infinities or a rank of 0.
- **`lu_solve` with a zero on `U`'s diagonal** raises, where SciPy's divides by it.
- **A singular system** raises in `solve` and `inv`, as NumPy's; an ill-conditioned one warns with
  {class}`numerical.LinAlgWarning`, as SciPy's.

Where NumPy is loaded, a `LinAlgError` is also a `numpy.linalg.LinAlgError` (SciPy's is the same class), so
`except np.linalg.LinAlgError` catches it; where `scipy.linalg` is loaded, the warning is also a
`scipy.linalg.LinAlgWarning`.

## Answers that differ on purpose

- **2-norms never overflow or underflow** where their value is in range: `vector_norm([1e200, 1e200])` is
  1.414e200, where NumPy's is infinite.
- **`maximum`, `minimum`, `max` and `min` order the zeros** as IEEE 754-2019: +0 above −0, whatever the order they
  come in. NumPy on arm64 agrees; NumPy on x86-64 keeps the sign its loops meet first.
- **`eig` and `eigvals` answer in complex numbers** whatever the eigenvalues, as NumPy 2.5's and SciPy's; NumPy
  before 2.5 made them real where it could.
- **A float32 or complex64 decomposition** is computed in double precision and rounded to single, below order 500
  (see {doc}`linalg`): the double answer of the same values, rounded once, where NumPy's LAPACK computes in single.
- **A sum along an axis that is not the last** (`sum(a, axis=0)` of a matrix, and the `mean`, `var` and `std` over
  it) adds each column pairwise, as NumPy adds along the last axis; NumPy adds such an axis row after row, in order.
  The last bits differ where the terms are of very different sizes, ours nearer the exact sum.
- **`round` to many places, and an integer's to tens or hundreds, is exact**: NumPy's scales by 10ᵈ in floating point,
  which moves a last bit (`round(x, 15)` near 10) or overflows (`round(1e307, 2)`: inf), and rounds an integer
  through float64 (not exact past 2⁵³).
- **`sort`** keeps equal elements in their order whatever `kind`; NumPy's default sort may order them otherwise, which
  only a zero's sign or a NaN's payload shows.
- **The special matrices** (`hilbert`, `pascal`, `invpascal`…) are float64 where SciPy's exact ones are integers:
  each entry computed exactly, then rounded once.
- **The complex elementary functions** (`exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `tanh` of a complex number) are
  glibc's, NumPy's on Linux to the bit; macOS's and Windows' libraries, NumPy's there, differ among infinities and
  NaNs and past their last bits.

## What is not there yet

- **Integer dtypes.** NumPy's integer arrays are read as float64 unless `dtype="int64"` is asked (by default from
  2026.12.0); an int64 tensor compares, selects, moves, computes and reduces as NumPy's integers, but not yet
  through `@`, `einsum` or the linear algebra, and the index functions answer in it. A real converted to int64 that
  it cannot hold (NaN, an infinity, a value out of range) raises, as Python's `int()`; NumPy's answer depends on the
  processor.
- **Views and assignment.** A `Tensor` is immutable: no `t[i] = x`, and no operation returns a view of another tensor.
- **Complex ordering.** `max`, `min`, `maximum`, `minimum` and `clip` take real dtypes only; NumPy orders complex
  numbers by real, then imaginary part (`sort` does too).
- **Namespaces.** No `numerical.linalg` and no `__array_namespace__` yet; everything lives at the root and in
  `numerical.fft`.
- **Domains.** No `random`, no sparse matrices, no `optimize`, `stats`, `signal`, `interpolate` or `special`: they are
  outside NumericAL's scope, for libraries built on it.
