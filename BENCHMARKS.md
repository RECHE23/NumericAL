# NumericAL — benchmarks vs NumPy

NumericAL is validated against **NumPy** on two axes: **correctness** (results agree
within a documented floating-point tolerance) and **performance** (a measured,
reproducible gap). Reproduce with:

```bash
make bench-python      # differential validation + timing (this file's table)
make python-test       # the parity suite alone (also runs in CI)
```

Correctness is gated automatically: `python/tests/test_numpy_parity.py` checks a
broad corpus of einsum specifications and linear solves (real and complex)
against NumPy within `rtol=1e-9`, and runs in CI (the Python job installs NumPy).

## How to read the numbers

`make bench-python` builds each input once (outside the timed region), checks
NumericAL's result equals NumPy's within tolerance, then times the operation. The
reported `ratio` is `numpy_time / numerical_time`, so **`ratio < 1` means NumericAL is
slower**.

**NumericAL is, and is expected to be, slower than NumPy** — this is not a fair
fight and is not meant to be one. NumPy dispatches linear algebra to
**BLAS/LAPACK**: decades of blocked, hand-vectorised kernels. NumericAL is a
portable, header-only, **dependency-free reference** implementation: a naive
index-walking `einsum` and an unblocked LU. Its value is correctness, genericity
over the scalar type, determinism, zero dependencies, and the symbolic↔numeric
bridge into SciLang — not raw throughput. The benchmark exists to **prove the
results are right** and to **quantify the gap** (a regression guard and an
optimisation compass), not to claim a win.

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
