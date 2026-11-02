# Accuracy

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
