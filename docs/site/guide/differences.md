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
`except np.linalg.LinAlgError` catches it, and an axis out of range, a {class}`numerical.AxisError` (a `ValueError`
and an `IndexError`), is also a `numpy.exceptions.AxisError`; where `scipy.linalg` is loaded, the warning is also a
`scipy.linalg.LinAlgWarning`.

## Answers that differ on purpose

- **2-norms never overflow or underflow** where their value is in range: `vector_norm([1e200, 1e200])` is
  1.414e200, where NumPy's is infinite.
- **`maximum`, `minimum`, `fmax`, `fmin`, `max` and `min` order the zeros** as IEEE 754-2019: +0 above −0,
  whatever the order they come in. NumPy on arm64 agrees; NumPy on x86-64 keeps the sign its loops meet first.
- **`power` of float64 on an x86-64 processor with AVX-512** is the C library's, as NumPy's on other processors;
  NumPy's there is its own vector kernel, whose last bit differs from it in some 5 % of the powers.
- **`exp` of float64** is NumericAL's own, in vectors (a table of 2^(j/128) and a polynomial): within 0.506 ulp of the
  exact value, the same bits on every processor with fused multiply-add, three times faster than the C
  library's on one core. NumPy's is the C library's (its own vector kernel on x86-64 with AVX-512): the last bit differs in some
  0.2 % of the values. A subnormal or zero result is the C library's; an x86-64 processor without FMA, and a build by
  MSVC, take the C library's throughout.
- **`eig` and `eigvals` answer in complex numbers** whatever the eigenvalues, as NumPy 2.5's and SciPy's; NumPy
  before 2.5 made them real where it could.
- **A float32 or complex64 decomposition** is computed in double precision and rounded to single, below order 500
  (see {doc}`linalg`): the double answer of the same values, rounded once, where NumPy's LAPACK computes in single.
- **A float32 sum of more than a thousand elements on 32-bit Windows**: NumPy's build there ends on other last bits
  than its other builds, whose bits NumericAL's are everywhere.
- **A sum along an axis that is not the last** (`sum(a, axis=0)` of a matrix, and the `mean`, `var` and `std` over
  it) adds each column pairwise, as NumPy adds along the last axis; NumPy adds such an axis row after row, in order.
  The last bits differ where the terms are of very different sizes, ours nearer the exact sum. With `initial=`, each
  column starts with it, the first term of its pairwise sum, where NumPy's rows add to it in order: the same bits
  below 8 terms (4 complex ones), where a pairwise sum is in order.
- **A sum in another dtype** (`sum(a, dtype="float64")` of a float32 `a`) converts every element and adds them
  pairwise, as a sum of that dtype; NumPy converts 8192 elements at a time, adds each block pairwise and the blocks in
  order. The last bits differ past 8192 elements.
- **`round` to many places, and an integer's to tens or hundreds, is exact**: NumPy's scales by 10ᵈ in floating point,
  which moves a last bit (`round(x, 15)` near 10) or overflows (`round(1e307, 2)`: inf), and rounds an integer
  through float64 (not exact past 2⁵³).
- **`sort`** keeps equal elements in their order whatever `kind`; NumPy's default sort may order them otherwise, which
  only a zero's sign or a NaN's payload shows.
  A `histogram` with weights between given edges sums the weights of equal values in their order too; NumPy's, in
  its sort's, which can move a last bit.
- **`median`, `nanmedian` and the quantiles of the largest values** answer them exactly: NumPy averages a middle
  value with itself (or interpolates between two equal ones), which overflows past half the largest float, so
  `nanmedian([1.7976931348623157e308])` is infinite there.
- **`geomspace` is within 3 ulps of the exact values**: NumPy's takes `10 ** y` between the `log10` of its ends,
  whose roundings carry into each value up to some 2·(|ln start| + |ln stop|) ulps (1500 over 10^±300); ours give
  their exponents' roundings back. The last bits differ where NumPy's are that far.
- **`take` with `mode="wrap"` from an empty axis** raises IndexError, as NumPy does for a result with elements; for
  one without, NumPy's loop never ends (numpy/numpy#11677).
- **`partition` and `argpartition`** sort each line, one of the arrangements NumPy's allows (the element at each
  `kth` the sorted one, none larger before it, none smaller after); NumPy's own leaves the rest in an order it does
  not specify.
- **`bitwise_count` answers int64**, NumPy's uint8 (NumericAL has no unsigned dtype). An int64 shifted by 64 places
  or more (or a negative count) is 0, or −1 to the right of a negative number, as NumPy defines it; its 32-bit Windows
  build shifts as the processor does, the count taken modulo 64.
- **`frexp`'s exponents are int64**, NumPy's int32, and 0 for an infinity or a NaN, where NumPy's are its C
  library's (0 from glibc and macOS, −1 from MSVC); `empty` gives zeros (no element of a Tensor is ever unset).
- **Floating-point warnings.** An invalid operation, a division by zero or an overflow answers IEEE 754's NaN or
  infinity without NumPy's `RuntimeWarning` ("invalid value encountered…"); there is no `errstate`.
- **`insert` before a list of bools** inserts before each true position, as NumPy does for a bool array; NumPy 2.4
  raises AttributeError for a list (it asks the list for its rank).
- **`nanquantile` and `nanpercentile` with weights of the axis's shape** answer as with those weights broadcast to
  the array's; NumPy raises IndexError there (numpy/numpy#29709).
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
- **Bool arithmetic and the small dtypes.** `+`, `-`, `*` (and `add`, `multiply`…) refuse two bool tensors, where
  NumPy's are a logical or and a logical and; there is no float16 nor int8, NumPy's dtype for `fabs`, `arcsin`… or
  `reciprocal` of a bool, which raise here.
- **The complex inverse trigonometric and hyperbolic functions** (`arcsin`, `arccos`, `arctan`, `arcsinh`,
  `arccosh`, `arctanh`) take real dtypes only.
- **Views and assignment.** A `Tensor` is immutable: no `t[i] = x`, and no operation returns a view of another tensor.
- **Complex ordering.** `max`, `min`, `maximum`, `minimum` and `clip` take real dtypes only; NumPy orders complex
  numbers by real, then imaginary part (`sort` does too).
- **Namespaces.** Everything lives at the root, with `numpy.fft`'s names in `numerical.fft` and `numpy.linalg`'s in
  `numerical.linalg`; the root is also a Tensor's array API namespace (`__array_namespace__`), with the standard's
  keywords (`device`, `copy`, `correction`, `descending`…) beside NumPy's.
- **`diagonal` and `trace` at the root** are `numpy.linalg`'s: the diagonals of the last two axes, where
  `numpy.diagonal` and `numpy.trace` take the first two (`axis1=0`, `axis2=1`). A matrix's are the same.
- **Domains.** No `random`, no sparse matrices, no `optimize`, `stats`, `signal`, `interpolate` or `special`: they are
  outside NumericAL's scope, for libraries built on it.
