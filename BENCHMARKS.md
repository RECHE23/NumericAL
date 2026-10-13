# NumericAL — benchmarks vs NumPy

NumericAL is validated against **NumPy** on two axes: **correctness** (results agree
within a documented floating-point tolerance) and **performance** (a measured,
reproducible gap). Reproduce with:

```bash
make bench-linalg      # dense linear algebra against BLAS/LAPACK, each library alone (the first tables)
make bench-python      # differential validation + timing, one process (the history's tables)
make python-test       # the parity suite alone (also runs in CI)
```

Correctness is gated automatically: `python/tests/test_numpy_parity.py` checks a
broad corpus of einsum specifications and linear solves (real and complex)
against NumPy within `rtol=1e-9`, and runs in CI (the Python job installs NumPy).

## Dense linear algebra against BLAS/LAPACK

NumericAL aims to match and then beat the BLAS/LAPACK NumPy ships, in portable C++ with no dependency. `make
bench-linalg` measures where it stands: each library in its own process with all its threads (a BLAS whose threads
still spin after a call would otherwise slow the next library's), every NumericAL answer checked before it is timed
(against its defining property to a backward-error bound; `lstsq` against NumPy's solution), the median of 15 calls
per process (5 from order 512) and, in the tables below, the median of 5 processes per library (`ROUNDS=5`; the
default is 3). `solve` takes 4 right-hand sides, `lstsq` a 2n × n matrix and one, `svd` the thin factors; every
operand is float64. Accelerate and OpenBLAS run on their default threads. The ratio is NumericAL's time over
NumPy's: **below 1, NumericAL is faster**. Correctness on adversarial matrices is checked separately, by `make
fuzz-linalg` (every decomposition against NumPy, on every matrix-product path).

```bash
make bench-linalg                      # SIZES=64,128,256,512,1024 ROUNDS=3; JSON in build/bench-linalg.json
make fuzz-linalg CASES=1000 SEED=2     # the decompositions on adversarial matrices, every GEMM path
```

**Apple M1 Pro** (8 cores; NumPy 2.4.2 on Accelerate; NumericAL 95028d8 built locally by Apple clang 16, the
AMX path), 2026-10-12, beside a desktop session (load 8 to 13 across the run):

| operation | n = 64 | n = 128 | n = 256 | n = 512 | n = 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| matmul | 0.00742 ms (2.34×) | 0.0359 ms (2.35×) | 0.14 ms (1.37×) | 0.764 ms (1.26×) | 4.85 ms (**0.94×**) |
| solve | 0.0518 ms (2.03×) | 0.186 ms (2.52×) | 0.743 ms (2.58×) | 3.01 ms (1.60×) | 12.2 ms (**0.76×**) |
| inverse | 0.0613 ms (1.80×) | 0.28 ms (2.32×) | 1.18 ms (1.66×) | 5.46 ms (1.20×) | 23.7 ms (**0.66×**) |
| cholesky | 0.0221 ms (1.63×) | 0.0802 ms (1.84×) | 0.37 ms (1.29×) | 1.86 ms (**0.99×**) | 8.61 ms (**0.75×**) |
| qr | 0.084 ms (1.06×) | 0.307 ms (**0.84×**) | 1.57 ms (**0.80×**) | 8.75 ms (**0.82×**) | 48.8 ms (**0.79×**) |
| eigh | 0.303 ms (**0.98×**) | 1.12 ms (1.19×) | 5.06 ms (1.12×) | 23.3 ms (**0.91×**) | 113 ms (**0.62×**) |
| svd | 0.622 ms (1.64×) | 2.54 ms (1.75×) | 12.1 ms (1.31×) | 60.2 ms (1.20×) | 352 ms (1.27×) |
| lstsq | 0.197 ms (**0.41×**) | 0.804 ms (**0.35×**) | 3.82 ms (**0.36×**) | 18.4 ms (**0.34×**) | 97.4 ms (**0.26×**) |

**Intel i9-10900X** (10 cores, AVX-512; NumPy 2.5.4 on OpenBLAS 0.3.34; NumericAL 95028d8 built locally by GCC
15.2, not the PyPI wheel; the AVX-512 path), same day, idle. A shared machine: figures move by some 10 %, and
OpenBLAS's product of order 256 on its twenty threads swings tenfold between runs (from 0.1 to 1 ms in three runs
that day; 0.098 ms here):

| operation | n = 64 | n = 128 | n = 256 | n = 512 | n = 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| matmul | 0.0148 ms (1.96×) | 0.0796 ms (3.09×) | 0.26 ms (2.65×) | 1.36 ms (2.59×) | 6.74 ms (1.47×) |
| solve | 0.05 ms (1.44×) | 0.207 ms (**0.36×**) | 1.15 ms (**0.77×**) | 4.68 ms (1.16×) | 13.8 ms (**0.87×**) |
| inverse | 0.0524 ms (**0.61×**) | 0.508 ms (**0.77×**) | 1.98 ms (1.08×) | 7.4 ms (1.06×) | 23.1 ms (**0.76×**) |
| cholesky | 0.0396 ms (1.99×) | 0.144 ms (**0.80×**) | 0.669 ms (1.15×) | 2.94 ms (**0.99×**) | 12.5 ms (**0.72×**) |
| qr | 0.132 ms (**0.94×**) | 0.667 ms (**0.40×**) | 2.51 ms (**0.64×**) | 11.9 ms (**0.76×**) | 48.1 ms (**0.74×**) |
| eigh | 0.48 ms (**0.80×**) | 1.95 ms (1.12×) | 7.76 ms (1.17×) | 29.9 ms (1.37×) | 114 ms (1.13×) |
| svd | 1.13 ms (1.20×) | 4.82 ms (1.30×) | 20.8 ms (1.44×) | 94.5 ms (1.14×) | 421 ms (1.14×) |
| lstsq | 0.272 ms (**0.24×**) | 1.87 ms (**0.36×**) | 7.72 ms (**0.45×**) | 33.8 ms (**0.39×**) | 155 ms (**0.50×**) |

**Intel i9-9900K** (8 cores, AVX2; NumPy 2.5.4 on OpenBLAS 0.3.34; NumericAL 95028d8 built locally by GCC 15.3;
the AVX2 path), same day, idle:

| operation | n = 64 | n = 128 | n = 256 | n = 512 | n = 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| matmul | 0.0169 ms (1.02×) | 0.0531 ms (1.62×) | 0.17 ms (1.29×) | 1.1 ms (1.27×) | 7.8 ms (1.10×) |
| solve | 0.0361 ms (**0.86×**) | 0.18 ms (**0.89×**) | 1 ms (1.39×) | 3.53 ms (1.19×) | 13.2 ms (1.05×) |
| inverse | 0.0539 ms (**0.67×**) | 0.398 ms (1.41×) | 1.59 ms (1.35×) | 5.34 ms (**0.95×**) | 22.5 ms (**0.62×**) |
| cholesky | 0.0285 ms (1.34×) | 0.122 ms (**0.92×**) | 0.623 ms (1.14×) | 2.39 ms (1.15×) | 10.9 ms (**0.98×**) |
| qr | 0.112 ms (**0.71×**) | 0.656 ms (**0.44×**) | 2.55 ms (**0.63×**) | 10.9 ms (**0.76×**) | 51 ms (**0.79×**) |
| eigh | 0.4 ms (**0.74×**) | 1.75 ms (**0.97×**) | 7.05 ms (1.05×) | 26.7 ms (1.23×) | 109 ms (1.07×) |
| svd | 1.25 ms (1.41×) | 4.42 ms (1.17×) | 19.6 ms (1.21×) | 87.7 ms (1.24×) | 368 ms (1.18×) |
| lstsq | 0.231 ms (**0.20×**) | 1.43 ms (**0.28×**) | 7.12 ms (**0.39×**) | 28.4 ms (**0.38×**) | 123 ms (**0.44×**) |

Where the gaps are, and what closes them:

- **`lstsq` and `qr` are ahead everywhere** (0.2–0.8×): `lstsq` takes a QR where a bound on the condition number
  shows the rank full, the singular value decomposition elsewhere; NumPy's `gelsd` always takes the latter. At
  n = 1024 the inverse is ahead on every machine, and Cholesky and `solve` on most.
- **Small problems** (n ≤ 256): up to 2–3× NumPy's time for the product of order 128, and up to 2.6× for the
  decompositions built on it (M1 Pro's `solve`, against Accelerate). A call of a few microseconds waits on its threads; the cut-off that decides how
  many to start is one weight for every shape. It wants a criterion by shape that counts the bytes a thread must
  fetch, not only the multiply-adds.
- **One x86-64 core** runs the product at about 77–80 GFLOP/s against OpenBLAS's 82–99: the loop order now keeps
  the A sliver in L1 (BLIS's order), and what is left is the wait on C from memory at each tile's start and the
  packing passes (about a tenth of the time each). Fusing the packing into the first compute pass (GEMMFIP) is the
  next step to try.
- **`eigh` and `svd` from n = 256** (1.05–1.45×): of `eigh`'s order 1024 on the i9-9900K, the tridiagonal reduction
  takes about 45 ms to LAPACK's 31 (its panel's matrix-vector work runs on one core), divide and conquer 36 to
  `dstedc`'s 20, and every call maps some 60 MB of fresh pages, about a microsecond each on Linux. The bidiagonal
  reduction of `svd` has the same panel.
- **Shared machines**: both x86-64 hosts serve other users, whose load moves single figures by 30 % or more. A
  figure that departs from its row is re-measured alone before it is read as a regression; `uptime` before a
  run says whether it was alone.

## History

The sections below record how each kernel got where it is, with the figures of their day. Their tables come from
`make bench-python`, which times NumericAL against NumPy in one process and writes the opposite ratio,
`numpy_time / numerical_time`: there, **below 1 means NumericAL is slower**.

### Indicative results

Apple Silicon (arm64, macOS), CPython 3.14, NumPy 2.4 — single run, per-call
medians. Absolute figures vary by machine; the **shape** of the gap is the point.
`ratio` is `numpy_time / numerical_time` (so `< 1` means slower); the last column
reads it as "NumericAL slower by". `naive` is the ratio of the very first engine, for
reference (see the evolution below).

| case                  |   NumPy |  NumericAL |   ratio |   naive | NumericAL slower by |
| --------------------- | ------: | ------: | ------: | ------: | ---------------: |
| dot (n=100k)          | 64.2 µs |  172 µs | 0.3737× | 0.0166× |             ~2.7× |
| outer (200×200)       | 13.1 µs |  129 µs | 0.1016× | 0.0079× |            ~10× |
| transpose (256×256)   |  632 ns |  189 µs | 0.0033× | 0.0004× |            ~300× |
| trace (300×300)       |  1.3 µs |  1.3 µs | 1.0340× | 0.1953× |  **0.97× (faster)** |
| sum-all (256×256)     |  9.1 µs | 75.8 µs | 0.1200× | 0.0060× |             ~8× |
| row-sums (256×256)    |  8.1 µs | 74.5 µs | 0.1087× | 0.0056× |             ~9× |
| matvec (256×256)      | 23.7 µs | 80.5 µs | 0.2942× | 0.0088× |            ~3.4× |
| **matmul (64×64)**    | 48.6 µs | 19.7 µs | **2.4745×** | 0.0046× | **0.40× (faster)** |
| **matmul (256×256)**  | 2.75 ms |  362 µs | **7.5938×** |    — | **0.13× (faster)** |
| frobenius (200×200)   | 26.3 µs | 69.9 µs | 0.3756× | 0.0155× |             ~2.7× |
| **matmul complex (48²)** | 293 µs | 122 µs | **2.4058×** | 0.0629× | **0.42× (faster)** |
| solve (64×64)         | 19.5 µs | 23.2 µs | 0.8382× | 0.8849× |            ~1.2× |
| solve complex (48×48) | 47.1 µs | 84.4 µs | 0.5586× | 0.5721× |            ~1.8× |

(One `make bench-python` run, 2026-10-11, arm64 with 8 cores, Apple clang 16, `-O2`.)

> **On the matmul baseline.** The table compares `numerical.einsum` against
> `np.einsum`, which by default (`optimize=False`) runs NumPy's own C loops, **not
> BLAS** — that is what the "faster than NumPy" matmul figures beat. Against real
> BLAS (`np.matmul` / `@`, here Apple's Accelerate) NumericAL's product is about
> **3× slower**: 113–151 GFLOP/s from n = 256 to 2048 against 330–490. It was 10×
> (33 GFLOP/s) before the packed kernel of the next sections; what remains is the
> per-architecture assembly, and here a matrix coprocessor, that a portable,
> header-only library did not use then (the AMX path and the x86-64 kernels chosen at
> run time came later: see the first section).

### Decompositions (vs `numpy.linalg`)

The decomposition layer — LU (`determinant`, `inverse`), Cholesky, QR, symmetric
eigen, and SVD — measured the same way (result checked, then timed; the
non-unique factorisations are checked by reconstruction).

| case                | NumPy   | NumericAL  |   ratio | NumericAL slower by |
| ------------------- | ------: | ------: | ------: | ---------------: |
| determinant (64×64) | 14.9 µs | 18.4 µs | 0.8102× |            ~1.2× |
| inverse (64×64)     | 26.7 µs |  231 µs | 0.1155× |            ~9×   |
| cholesky (64×64)    | 13.7 µs | 22.6 µs | 0.6039× |            ~1.7× |
| qr (64×32)          | 34.6 µs | 56.4 µs | 0.6145× |            ~1.6× |
| svd (48×32)         |  133 µs |  304 µs | 0.4359× |            ~2.3× |
| eigh (48×48)        |  188 µs |  836 µs | 0.2249× |            ~4.4× |
| **lstsq (64×32)**   |  132 µs | 55.9 µs | 2.3644× | **0.42× (faster)** |

These land much closer to LAPACK than the data-movement einsum cases: an LU-backed
`determinant` is near parity; `cholesky` / `qr` / `svd` are within ~1.6–2.3×;
`eigh` (cyclic Jacobi vs LAPACK's `syevd`) is ~4.4×; `inverse` (n back-solves) is
the laggard. `lstsq` actually **beats** NumPy here — `numpy.linalg.lstsq` uses a
full SVD (`gelsd`), heavier than our Gram–Schmidt QR for a small overdetermined
system.

At larger orders the LU behind `determinant`, `slogdet`, `solve` and `inverse` is
blocked, as LAPACK's `dgetrf`: panels of 64 columns are factorised with partial
pivoting in a column-major copy, and the trailing block takes its update from the
matrix product kernel below. It regroups the unblocked algorithm's updates without
reordering any, so without multiply-add fusion the factors are the same to the bit
(`tests/test_decomposition.cpp`).
`slogdet` of a diagonally dominant `double` matrix, best of N:

| n    | NumPy (LAPACK) | unblocked (before) | blocked | NumericAL slower by |
| ---: | -------------: | -----------------: | ------: | ------------------: |
|  256 |        0.20 ms |            1.24 ms | 0.69 ms |               ~3.5× |
|  512 |        1.15 ms |            10.5 ms | 3.47 ms |               ~3.0× |
| 1024 |        5.03 ms |            80.8 ms | 16.6 ms |               ~3.3× |
| 2048 |        38.3 ms |                  — | 93.8 ms |               ~2.4× |

(2026-10-11, arm64 with 8 cores, Apple clang 16, `-O2`; NumPy through Apple's Accelerate.)

`inverse`, and `solve` with a matrix right-hand side, solve every column at once on the
factors: within a block of rows one row after another (in parallel chunks of columns),
across blocks through the product kernel. `inverse` of the same matrices, best of N:

| n    | NumPy (LAPACK) | column by column (before) | blocked | NumericAL slower by |
| ---: | -------------: | ------------------------: | ------: | ------------------: |
|  256 |        0.61 ms |                   3.96 ms | 2.01 ms |               ~3.3× |
|  512 |        3.67 ms |                   31.5 ms | 9.66 ms |               ~2.6× |
| 1024 |        20.9 ms |                    241 ms | 46.2 ms |               ~2.2× |
| 2048 |         168 ms |                         — |  286 ms |               ~1.7× |

`qr` is blocked as LAPACK's `dgeqrf`: the 32 reflectors of a panel act on it one by
one, then on the columns right of it at once as `I − V T Vᴴ` through the product
kernel, and `Q` is formed block by block. Square matrices of the same kind, best of N:

| n    | NumPy (LAPACK) | unblocked (before) | blocked | NumericAL slower by |
| ---: | -------------: | -----------------: | ------: | ------------------: |
|  256 |        1.81 ms |            6.03 ms | 3.41 ms |               ~1.9× |
|  512 |        10.2 ms |            47.4 ms | 17.6 ms |               ~1.7× |
| 1024 |        60.6 ms |             393 ms |  101 ms |               ~1.7× |

`cholesky` is blocked the same way: each panel's diagonal block row by row, the rows
below it in parallel, and the trailing lower part updated through the product kernel,
every element again receiving its terms in the unblocked order. `M Mᵀ + n·I`, best of N:

| n    | NumPy (LAPACK) | unblocked (before) | blocked | NumericAL slower by |
| ---: | -------------: | -----------------: | ------: | ------------------: |
|  256 |        0.27 ms |            1.58 ms | 0.82 ms |               ~3.0× |
|  512 |        1.71 ms |            15.2 ms | 4.30 ms |               ~2.5× |
| 1024 |        12.2 ms |             139 ms | 22.5 ms |               ~1.8× |
| 2048 |        73.6 ms |                  — |  142 ms |               ~1.9× |

All timed cases (array ops and decompositions) matched NumPy within tolerance.
Geometric-mean ratio over the whole suite ≈ **0.38×** (NumericAL ~2.6× slower on
average), pulled down mostly by the memory-bound data-movement cases above.

### Multithreading

The embarrassingly-parallel kernels run on `std::thread` (the only portable,
dependency-free standard parallelism — no TBB, no execution policies): each
parallelised index writes a **disjoint** slice of the output, so there are no
locks and the result is **identical to the sequential run** (deterministic).
Each kernel states the work of one index, and a range runs on one thread per
2¹⁶ multiply-adds of it (a quarter of that for a vectorised inner loop), up to the
hardware's count, sequentially below two: a thread costs about 10 µs to start, and
counting indices instead of work started eight of them for a 16 × 16 `inverse`
(177 µs instead of 8). Parallelised at the time: the **matrix product** (blocks of rows
by columns of the result, below) and `inverse` (column solves).

On an 8-core machine the effect is stark where the work is genuinely parallel and
compute-bound: **matmul 256×256 went 0.89× → 2.76×** (~3× faster than NumPy),
small matmul stays ≥ parity. The decompositions gain less: their cores —
LU/Cholesky/QR factorisation and the Jacobi eigen sweeps — are **inherently
sequential** (each step depends on the previous), so only the independent parts
(`inverse`'s solves, the SVD's U columns) parallelise; full parallelism there
would need blocked (Level-3) algorithms — the step taken since (the first section). This is the
one BLAS technique a header-only leaf can adopt portably; the hand-tuned
per-architecture assembly microkernels behind most of BLAS's remaining edge are
not.

### The matrix product kernel

`a * b` and a canonical `einsum("ij,jk->ik", …)` run one kernel, `include/numerical/gemm.hpp`,
in the loop structure of BLIS: the inner dimension is cut into panels of 256, both operands'
panels are copied once into the layout the inner loop reads at unit stride, and a register
tile of 4 × 12 elements of the result (4 × 24 for `float`) accumulates a whole panel before it
is written back. For `float` and `double` under GCC and Clang the tile is held in vectors of
the target's native width (16 bytes, 32 with AVX); elsewhere, and for `complex` and integers,
it is a plain array. Blocks of 64 rows by 48 columns of the result run in parallel.

Each element of the result is read into the tile at the start of a panel and written back at
its end, so it is summed in ascending order of the inner index, exactly as the plain i-k-j
loop sums it, on any number of threads. With the compiler's multiply-add fusion off the two
are the same to the bit (`tests/test_gemm.cpp`, across panels, column blocks, partial tiles
and four scalar types); with it on, the compiler may fuse one loop and not the other.
Products below 8³ multiply-adds, or shallower than 8, keep the plain loop: a 1000 × 1000 × 1
product ran six times slower packed.

Measured with `make bench-gemm` (`double`, best of N; `ikj` is the loop `a * b` ran before,
in the same binary):

| N    | arm64, 8 cores: ikj | packed | speed-up | x86-64 (AVX2), 2 cores: ikj | packed | speed-up |
| ---: | ------------------: | -----: | -------: | --------------------------: | -----: | -------: |
|  128 |             0.17 ms | 0.13 ms |    1.3× |                     0.85 ms | 0.50 ms |    1.7× |
|  256 |             0.74 ms | 0.45 ms |    1.7× |                     6.87 ms | 4.03 ms |    1.7× |
|  512 |             5.66 ms | 2.34 ms |    2.4× |                     46.9 ms | 23.2 ms |    2.0× |
| 1024 |             34.3 ms | 15.5 ms |    2.2× |                      446 ms |  160 ms |    2.8× |
| 2048 |              390 ms |  121 ms |    3.2× |                     3850 ms | 1278 ms |    3.0× |

(2026-10-11. arm64: Apple clang 16, `-O2`; x86-64: GCC 14.2, `-O2` without `-march`, the
build of the Python wheels.) On one x86-64 core (GCC 13, `-O2`) the kernel does 11.9 GFLOP/s
against 3.3 for the i-k-j loop; with `-march=native` (AVX2 and FMA) 37 against 5.

Under MSVC (`/O2`), where the tile is a plain array, the kernel against the i-k-j loop:
2.3× at 64, 2.6× at 128, 3.6× at 256, 4.0× at 512, 4.3× at 1024 (a Windows virtual machine).

Two things the kernel needs from GCC, each measured on that core: its loops are unrolled
explicitly (GCC at `-O2` does not, and then keeps the tile in memory: 4.8 GFLOP/s), and the
tile moves into and out of the vectors one vector at a time (a single copy of the whole array
also kept it in memory: 7.4 GFLOP/s).

#### Python binding glue copies (measured, documented)

The Python paths (`do_solve`, `matrix_from`, `do_lstsq`) each do:

    numerical::matrix<T> m {r, c, a.data()};

The by-value constructor copies (O(n²)). Measured (copy against solve time):

- 32   : 0.08 µs / 3.58 µs   → 2.3 %
- 64   : 0.54 µs / 20.2 µs   → 2.7 %
- 128  : 2.08 µs / 132 µs    → 1.6 %
- 256  : 8.25 µs / 1253 µs   → 0.7 %
- 512  : 154 µs / 10305 µs   → 1.5 %

Conclusion: **document only**. No const-ref ctor and no borrowing view is added. A non-owning
view would add API surface and dilute the deliberate value-semantics of
`matrix`/`vector` (a noted library strength). The solve paths already copy the
data internally (LU/QR etc.), so this is one extra O(n²) on top of an
unavoidable O(n²)+O(n³). <3 % on a non-bottleneck → documented, not "fixed".

### The einsum engine, in three steps

1. **Incremental odometer.** The first cut allocated a fresh index vector and did
   `std::map<char,std::size_t>` lookups **per output element**. It was replaced by
   a single integer index space walked with an incremental mixed-radix odometer:
   per-label stride coefficients computed once, each operand's flat offset carried
   in O(1), no allocation and no lookup in the hot loop. Geomean **0.021× →
   0.142×** (~48× → ~7× slower).
2. **Contiguous-axis fast path.** The innermost free label and the innermost
   summed label are peeled out of the odometer into a tight unit-of-stride inner
   loop, amortising one odometer step over a whole axis. Geomean **0.142× →
   0.194×** (~7× → ~5× slower), ~1.7× per arithmetic-bearing op; `trace` overtakes
   NumPy.
3. **Cache-blocked GEMM.** A plain two-operand single contraction in canonical
   layout (matmul / matrix–vector) is recognised and routed to a tiled matrix
   kernel — m-k-n order, accumulating into the output, 32-wide blocks — so every
   access is unit-stride and stays cache-resident. The sum still runs in ascending
   order, so the result is **bit-identical** to the general engine. Geomean
   **0.194× → 0.29×**, and **matmul now matches or beats NumPy**: 64×64 at 1.03×,
   256×256 at 0.89× (≈ BLAS), complex 48² at 1.78×. A 1×1 result (a plain `dot`) is
   left to the general engine, where its tight loop is faster than tiling.
4. **Packed kernel.** The GEMM path and `a * b` now share the packed, register-tiled
   kernel of the section above, which still sums in ascending order. matmul 256×256
   **2.76× → 7.6×**, 64×64 **1.03× → 2.5×**, complex 48² **1.78× → 2.4×**, matvec
   256×256 **~10× → ~3.4× slower** (2026-10-11 run).

All four steps are purely structural — one generic engine with a recognised
GEMM shape, every rank/spec/scalar, **no intrinsics and no dependency** (the
kernel's vectors are GCC and Clang's portable vector extension, with a plain-array
fallback). The
standard library offers no numeric SIMD primitive (unlike `memmem` /
`boyer_moore_searcher` for strings); the wins are algorithmic, and the resulting
clean loops are what the compiler auto-vectorises.

### What this told us (2026-10-11)

- **`matmul` and `solve` were at the level of NumPy's own loops** without BLAS: a packed GEMM and
  an unblocked LU capture most of the kernel-level advantage at these sizes.
- **Pure data-movement stays the outlier** — `transpose` (~300×), and the
  reductions and `outer` (~8–10×). These are memory-bound: a transpose reads
  one operand column-wise (one cache line per element) and cannot be contiguous on
  both sides; the reductions touch every input once to produce little output. Only
  **cache blocking the general engine** (as the GEMM path does for products) would
  help, a further localised optimisation. Beyond it lie opt_einsum-style pairwise
  path optimisation and a BLAS reduction, not adopted here: they trade
  generic simplicity and zero dependencies for raw speed. Correctness and clarity
  come first.
