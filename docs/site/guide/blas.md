# BLAS and LAPACK for other programs

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
