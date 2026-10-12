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
bench-linalg` measures where it stands: each library in its own process with all its threads (a BLAS whose
threads still spin after a call would otherwise slow the next library's), every NumericAL answer checked against
its defining property before it is timed, the median of 15 calls per process (5 at order 1024) and the median of
5 processes per library. The ratio is NumericAL's time over NumPy's: **below 1, NumericAL is faster**. Correctness
on adversarial matrices is checked separately, by `make fuzz-linalg` (every decomposition against NumPy, on every
matrix-product path).

```bash
make bench-linalg                      # SIZES=64,128,256,512,1024 ROUNDS=3; JSON in build/bench-linalg.json
make fuzz-linalg CASES=1000 SEED=2     # the decompositions on adversarial matrices, every GEMM path
```

**Apple M1 Pro** (8 cores; NumPy 2.4.2 on Accelerate; NumericAL 1c8805a, the AMX path), 2026-10-12:

| operation | n = 64 | n = 128 | n = 256 | n = 512 | n = 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| matmul | 0.0075 ms (2.38×) | 0.0505 ms (3.31×) | 0.15 ms (1.46×) | 0.79 ms (1.36×) | 5.14 ms (**0.94×**) |
| solve | 0.047 ms (1.87×) | 0.162 ms (2.21×) | 0.701 ms (2.43×) | 3.27 ms (1.80×) | 15.6 ms (**0.94×**) |
| inverse | 0.0597 ms (1.79×) | 0.255 ms (2.11×) | 1.36 ms (1.79×) | 6.92 ms (1.47×) | 34.8 ms (**0.95×**) |
| cholesky | 0.0309 ms (2.26×) | 0.11 ms (2.51×) | 0.446 ms (1.48×) | 2.85 ms (1.57×) | 13.9 ms (1.15×) |
| qr | 0.102 ms (1.29×) | 0.391 ms (1.07×) | 2.16 ms (1.10×) | 12.6 ms (1.21×) | 66.6 ms (1.04×) |
| eigh | 0.342 ms (1.11×) | 1.46 ms (1.55×) | 7.03 ms (1.53×) | 37.4 ms (1.42×) | 193 ms (1.06×) |
| svd | 0.673 ms (1.79×) | 2.89 ms (1.95×) | 13.6 ms (1.45×) | 71.7 ms (1.43×) | 393 ms (1.39×) |
| lstsq | 0.805 ms (1.70×) | 3.58 ms (1.55×) | 17.3 ms (1.60×) | 87.2 ms (1.62×) | 503 ms (1.28×) |

**Intel i9-10900X** (10 cores, AVX-512; NumPy 2.5.4 on OpenBLAS 0.3.34; GCC 15 wheel, the AVX-512 path), same
code and day. A shared machine: figures move by some 10 %, and OpenBLAS's product of order 256 on its twenty
threads takes ten times its order-512 rate in every process (its own, reproducibly):

| operation | n = 64 | n = 128 | n = 256 | n = 512 | n = 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| matmul | 0.0165 ms (2.15×) | 0.115 ms (4.04×) | 0.298 ms (**0.10×**) | 1.34 ms (2.49×) | 6.75 ms (1.77×) |
| solve | 0.0684 ms (1.95×) | 0.241 ms (**0.42×**) | 1.14 ms (**0.77×**) | 4.74 ms (1.16×) | 26.4 ms (1.89×) |
| inverse | 0.0949 ms (1.06×) | 0.422 ms (**0.63×**) | 2.54 ms (1.41×) | 12 ms (1.74×) | 51.3 ms (1.66×) |
| cholesky | 0.0436 ms (2.10×) | 0.174 ms (1.00×) | 0.757 ms (1.25×) | 3.92 ms (1.37×) | 21.5 ms (1.25×) |
| qr | 0.152 ms (1.10×) | 0.636 ms (**0.39×**) | 3.73 ms (**0.94×**) | 21.2 ms (1.32×) | 91.2 ms (1.36×) |
| eigh | 0.525 ms (**0.93×**) | 2.54 ms (1.41×) | 11.1 ms (1.64×) | 62.1 ms (2.79×) | 253 ms (2.61×) |
| svd | 1.16 ms (1.26×) | 4.91 ms (1.33×) | 23.6 ms (1.64×) | 125 ms (1.50×) | 535 ms (1.49×) |
| lstsq | 1.32 ms (1.16×) | 5.69 ms (1.10×) | 30 ms (1.72×) | 155 ms (1.81×) | 640 ms (2.07×) |

**Intel i9-9900K** (8 cores, AVX2; NumPy 2.5.4 on OpenBLAS 0.3.34; GCC 15 wheel, the AVX2 path), same code and
day:

| operation | n = 64 | n = 128 | n = 256 | n = 512 | n = 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| matmul | 0.0234 ms (1.44×) | 0.124 ms (3.81×) | 0.24 ms (1.84×) | 1.21 ms (1.39×) | 7.99 ms (1.11×) |
| solve | 0.0662 ms (1.62×) | 0.306 ms (1.52×) | 1.18 ms (1.61×) | 4.73 ms (1.67×) | 19.7 ms (1.61×) |
| inverse | 0.109 ms (1.37×) | 0.522 ms (1.80×) | 2.72 ms (2.29×) | 12.8 ms (2.31×) | 54.5 ms (1.49×) |
| cholesky | 0.0472 ms (2.22×) | 0.192 ms (1.41×) | 0.866 ms (1.53×) | 4.29 ms (2.16×) | 22.2 ms (1.98×) |
| qr | 0.182 ms (1.14×) | 0.792 ms (**0.52×**) | 4.26 ms (1.04×) | 19.4 ms (1.34×) | 80.6 ms (1.25×) |
| eigh | 0.587 ms (1.11×) | 2.56 ms (1.40×) | 12 ms (1.74×) | 62 ms (2.89×) | 265 ms (2.62×) |
| svd | 1.32 ms (1.51×) | 5.3 ms (1.40×) | 25.8 ms (1.61×) | 125 ms (1.76×) | 531 ms (1.67×) |
| lstsq | 1.56 ms (1.38×) | 6.44 ms (1.26×) | 33 ms (1.84×) | 151 ms (2.04×) | 633 ms (2.28×) |

Where the gaps are, and what closes them:

- **Small matrix products** (n ≤ 256): two to four times NumPy's time. A product of a few microseconds waits on its
  threads; the cut-off that decides how many to start is one weight for every product shape, and lowering it
  slowed QR down on both architectures. It wants a criterion by shape.
- **One x86-64 core** runs the product at about 0.7 of OpenBLAS's rate: the tile reaches 103–119 GFLOP/s in the
  L1 cache, the whole product about 80, stalled on the B operand from L3. Large products on every core inherit it.
- **`eigh` on x86-64** (2.6–2.9× at n ≥ 512): the tridiagonal reduction's matrix-vector products run on SSE2, the
  baseline the wheels are built for (AVX2 gives 21 % on that step), and its parallel steps do not pay below order
  1024. On arm64, Accelerate's lead is 1.1–1.5×.
- **`svd` and `lstsq`** carry the bidiagonal reduction's two matrix-vector passes per step, which the panels do not
  remove (LAPACK's `dgebrd` keeps them too); 1.3–2.3× throughout.

## History

The sections below record how each kernel got where it is, with the figures of their day. Their tables come from
`make bench-python`, which times NumericAL against NumPy in one process and writes the opposite ratio,
`numpy_time / numerical_time`: there, **below 1 means NumericAL is slower**.

## Indicative results

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
> header-only library does not use.

## Decompositions (vs `numpy.linalg`)

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

## Multithreading

The embarrassingly-parallel kernels run on `std::thread` (the only portable,
dependency-free standard parallelism — no TBB, no execution policies): each
parallelised index writes a **disjoint** slice of the output, so there are no
locks and the result is **identical to the sequential run** (deterministic).
Each kernel states the work of one index, and a range runs on one thread per
2¹⁶ multiply-adds of it (a quarter of that for a vectorised inner loop), up to the
hardware's count, sequentially below two: a thread costs about 10 µs to start, and
counting indices instead of work started eight of them for a 16 × 16 `inverse`
(177 µs instead of 8). Parallelised so far: the **matrix product** (blocks of rows
by columns of the result, below) and `inverse` (column solves).

On an 8-core machine the effect is stark where the work is genuinely parallel and
compute-bound: **matmul 256×256 went 0.89× → 2.76×** (~3× faster than NumPy),
small matmul stays ≥ parity. The decompositions gain less: their cores —
LU/Cholesky/QR factorisation and the Jacobi eigen sweeps — are **inherently
sequential** (each step depends on the previous), so only the independent parts
(`inverse`'s solves, the SVD's U columns) parallelise; full parallelism there
would need blocked (Level-3) algorithms — a larger step, not taken yet. This is the
one BLAS technique a header-only leaf can adopt portably; the hand-tuned
per-architecture assembly microkernels behind most of BLAS's remaining edge are
not.

## The matrix product kernel

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

### Python binding glue copies (measured, documented)

The Python paths (`do_solve`, `matrix_from`, `do_lstsq`) each do:

    numerical::matrix<T> m {r, c, a.data()};

The by-value constructor copies (O(n²)). Measured this session (copy vs solve time):

- 32   : 0.08 µs / 3.58 µs   → 2.3 %
- 64   : 0.54 µs / 20.2 µs   → 2.7 %
- 128  : 2.08 µs / 132 µs    → 1.6 %
- 256  : 8.25 µs / 1253 µs   → 0.7 %
- 512  : 154 µs / 10305 µs   → 1.5 %

Decision (supervisor, consistent with the measurement + grown-measured rule):
**document only**. No const-ref ctor and no borrowing view is added. A non-owning
view would add API surface and dilute the deliberate value-semantics of
`matrix`/`vector` (a noted library strength). The solve paths already copy the
data internally (LU/QR etc.), so this is one extra O(n²) on top of an
unavoidable O(n²)+O(n³). <3 % on a non-bottleneck → documented, not "fixed".

See also the earlier session measurement that reached the same conclusion.

## The einsum engine, in three steps

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

## What this tells us

- **`matmul` and `solve` are now at NumPy's level** without BLAS: a packed GEMM and
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
