# Linear algebra

NumericAL's linear algebra carries NumPy's `numpy.linalg` names with NumPy 2's signatures, and most of
`scipy.linalg` with SciPy's. Where the two libraries share a name (`cholesky`, `qr`, `eigh`, `lstsq`, `svd`), the
function takes the arguments of both and keeps NumPy's defaults where they disagree; {doc}`differences` lists those
cases.

## Stacks

Every function but `lstsq` (2-D in NumPy too) takes a stack of matrices, `(..., m, n)`, as NumPy's: each matrix answers
as it does alone, to the bit, and the answers stack along the leading axes. `solve` broadcasts the leading axes of `a`
and `b` as NumPy 2 does, `b` a vector only when it is 1-D; `det` and `slogdet` of a stack give tensors, and a stack's
`matrix_rank` nested integers. Small matrices, or as many as threads, go to the threads one each; larger ones run one
at a time, each with the threads (10 000 inverses of order 4 on eight arm64 cores: 2.0 ms, NumPy 4.4 ms).

```pycon
>>> import numerical
>>> stack = numerical.asarray([[[2.0, 0.0], [0.0, 2.0]], [[4.0, 0.0], [0.0, 1.0]]])
>>> numerical.det(stack)
numerical.Tensor([4.0, 4.0], dtype=float64)
>>> numerical.inv(stack)[1]
numerical.Tensor([[0.25, 0.0],
                  [0.0, 1.0]], dtype=float64)
```

## The functions

From `numpy.linalg`, with NumPy 2's signatures and answers: `det` (`determinant`), `slogdet` → `(sign, logabsdet)`,
`inv` (`inverse`), `solve`, `pinv(a, rcond, hermitian)`, `matrix_rank(a, tol, hermitian)`, `lstsq(a, b, rcond)` →
`(x, residuals, rank, s)`, `cholesky(a, upper=False)`, `qr(a, mode="reduced")` → `(Q, R)` (`"complete"` as NumPy's,
`"r"` → `R` alone), `eigh(a, UPLO)` → `(values, vectors)` and `eigvalsh(a, UPLO)`, `eig(a)` → `(values, vectors)` and
`eigvals(a)`, `svd(a, full_matrices=True, compute_uv=True, hermitian=False)` → `(U, S, Vh)` and `svdvals(a)`,
`norm(x, ord, axis, keepdims)`, `vector_norm`, `matrix_norm` (`"fro"`, `"nuc"`, 1, -1, 2, -2, ±inf), `cond(x, p)`
(infinite for a singular matrix), `matrix_power(a, n)` (`n` negative through the inverse), `multi_dot(arrays)` (the
cheapest order of the chain), `trace(x, offset, dtype)`, `diagonal(x, offset)`, `matrix_transpose`, `outer`,
`cross(x1, x2, axis)`, `tensordot(a, b, axes)`, `tensorsolve(a, b, axes)`, `tensorinv(a, ind)`, `vecdot(x1, x2,
axis)` (conjugating `x1`) and `conj` (`conjugate`).

The keywords mean what they mean to NumPy 2: `UPLO="U"` and `upper=True` read the upper triangle, `rcond`/`rtol` cut
the singular values relative to the largest and `tol` absolutely, `hermitian=True` takes the eigendecomposition.
`svd`'s `full_matrices=False` gives the thin factors, so `(U * S) @ Vh` is `a`; `compute_uv=False` gives `S` alone;
`hermitian=True` takes the singular values as the eigenvalues' magnitudes. `eigh` of a complex Hermitian matrix gives
real values and complex vectors. `eig` gives its eigenvalues in the Schur form's order, a real matrix's pairs
conjugate to the bit, each vector of unit norm with its largest component real.

From `scipy.linalg`, with SciPy's signatures: `lu(a, permute_l, p_indices)` → `(P, L, U)` (any shape),
`ldl(A, lower, hermitian)` → `(lu, d, perm)`, `eigh(a, b)` (the pencil `a x = λ b x`, `b` positive definite) and
`eigvalsh(a, b=b)`, `schur(a, output="real")` → `(T, Z)` (`"complex"` the triangular form of a real matrix; with
`sort=` `"lhp"`, `"rhp"`, `"iuc"`, `"ouc"` or a callable, → `(T, Z, sdim)`, the selected eigenvalues leading), `qz` and
`ordqz` (the real generalized Schur form, as `dgges`/`dtgsen`, or the complex one), `rsf2csf`, `cdf2rdf`, `hessenberg`,
`matrix_balance`, `rq`, `polar(a, side)`, `null_space`, `orth`, `subspace_angles`, `orthogonal_procrustes`, `pinvh`,
`bandwidth`, `issymmetric`, `ishermitian`, `solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False)`,
`solve_banded((l, u), ab, b)`, `solveh_banded(ab, b, lower=False)`, `cholesky_banded`, `cho_solve_banded`,
`solve_toeplitz(c_or_cr, b)`, `solve_circulant(c, b, singular)`, `eigh_tridiagonal`, `eigvalsh_tridiagonal`; the
matrices `toeplitz`, `hankel`, `circulant`, `block_diag`, `hilbert`, `invhilbert`, `pascal`, `invpascal`, `hadamard`,
`helmert`, `leslie`, `companion`, `fiedler`, `fiedler_companion`, `convolution_matrix`, `dft`, `khatri_rao` and
`diagsvd`; and the matrix functions and equations below. {doc}`../reference/python` documents each one, and {doc}`cpp`
the algorithm each runs.

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

`lu_factor` and `cho_factor` return SciPy's pairs, `(lu, piv)` and `(c, lower)`, and `lu_solve(factor, b, trans=0)`
and `cho_solve` take them; one factorisation, then `n² k` per solve. The pairs are an `LUFactor` (`piv` LAPACK's
interchanges, a tuple of ints) and a `CholeskyFactor`, each with a `.solve(b)`. A zero on `U`'s diagonal raises in
`lu_solve`, where SciPy's divides by it, and `cho_factor` leaves zeros in the triangle it does not use.

```pycon
>>> factor = numerical.lu_factor([[4.0, 3.0], [6.0, 3.0]])
>>> numerical.lu_solve(factor, [10.0, 12.0])
numerical.Tensor([1.0, 2.0], dtype=float64)
```

## Matrix functions and equations

`expm`, `logm`, `sqrtm`, `funm(A, func, disp)`, `fractional_matrix_power(A, t)`, the trigonometric and hyperbolic
matrix functions (`cosm`, `sinm`, `tanm`, `coshm`, `sinhm`, `tanhm`) and `signm`, `expm_multiply`,
`expm_frechet(A, E, method, compute_expm)` and `expm_cond(A)` follow `scipy.linalg`'s, a stack's matrices each; so do
the Sylvester, Lyapunov and Riccati solvers (`solve_sylvester(a, b, q)`, `solve_continuous_lyapunov(a, q)`,
`solve_discrete_lyapunov(a, q, method)`, `solve_continuous_are(a, b, q, r, e, s, balanced)`,
`solve_discrete_are(a, b, q, r, e, s, balanced)`), `schur`, `qz`, `ordqz`, `hessenberg`, `polar` and the special matrices. `funm` estimates its own error and,
with `disp=False`, returns it beside the answer, as SciPy's.
