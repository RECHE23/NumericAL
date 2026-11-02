# Linear algebra

NumericAL's linear algebra carries NumPy's `numpy.linalg` names with NumPy 2's signatures, and most of
`scipy.linalg` with SciPy's. Where the two libraries share a name (`cholesky`, `qr`, `eigh`, `lstsq`, `svd`), the
function takes the arguments of both and keeps NumPy's defaults where they disagree; {doc}`differences` lists those
cases.

## Stacks

Every function but `lstsq` (2-D in NumPy too) takes a stack of matrices, `(..., m, n)`, as NumPy's: each matrix answers
as it does alone, to the bit, and the answers stack along the leading axes. `solve` broadcasts the leading axes of `a`
and `b` as NumPy 2 does. Small matrices, or as many as threads, go to the threads one each; larger ones run one at a
time, each with the threads.

```pycon
>>> import numerical
>>> stack = numerical.asarray([[[2.0, 0.0], [0.0, 2.0]], [[4.0, 0.0], [0.0, 1.0]]])
>>> numerical.det(stack)
numerical.Tensor([4.0, 4.0], dtype=float64)
>>> numerical.inv(stack)[1]
numerical.Tensor([[0.25, 0.0],
                  [0.0, 1.0]], dtype=float64)
```

## Single precision

Every function accepts the four dtypes and answers in the input's. A float32 or complex64 matrix is decomposed in
double precision and its answers rounded to single, so they are as accurate as single precision holds, and the
default tolerances are float32's epsilon, as NumPy's. From order 500, `solve`, `inv`, `det`, `slogdet`, `cholesky`,
`lu_factor` and `lu_solve` factorise a float32 matrix in single precision, as NumPy's LAPACK does at every order:
there the factorisation's time dominates, and single precision halves it.

## Choosing the algorithm

`eigh(a, method=...)` and `svd(a, method=...)` choose the algorithm: divide and conquer by default (LAPACK's
`dsyevd`/`zheevd`, `dgesdd`), `"tridiagonal_ql"` or `"bidiagonal_qr"` (`dsyev`'s, `dgesvd`'s), or `"jacobi"`. SciPy's
`driver` and `lapack_driver` arguments name the same choices.

## Factor once, solve many

`lu_factor` and `cho_factor` return SciPy's pairs, `(lu, piv)` and `(c, lower)`, and `lu_solve` and `cho_solve` take
them; one factorisation, then `n² k` per solve.

```pycon
>>> factor = numerical.lu_factor([[4.0, 3.0], [6.0, 3.0]])
>>> numerical.lu_solve(factor, [10.0, 12.0])
numerical.Tensor([1.0, 2.0], dtype=float64)
```

## Matrix functions and equations

`expm`, `logm`, `sqrtm`, `funm`, `fractional_matrix_power`, the trigonometric and hyperbolic matrix functions,
`expm_multiply`, `expm_frechet` and `expm_cond` follow `scipy.linalg`'s; so do the Sylvester, Lyapunov and Riccati
solvers, `schur`, `qz`, `ordqz`, `hessenberg`, `polar` and the special matrices. `funm` estimates its own error and,
with `disp=False`, returns it beside the answer, as SciPy's.
