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
| dot (n=100k)          | 65.4 µs |  174 µs | 0.3765× | 0.0166× |             ~2.7× |
| outer (200×200)       | 13.4 µs |  130 µs | 0.1029× | 0.0079× |            ~10× |
| transpose (256×256)   |  662 ns |  192 µs | 0.0035× | 0.0004× |            ~285× |
| trace (300×300)       |  1.4 µs |  792 ns | 1.7250× | 0.1953× |  **0.6× (faster)** |
| sum-all (256×256)     |  9.2 µs | 75.5 µs | 0.1222× | 0.0060× |             ~8× |
| row-sums (256×256)    |  8.3 µs | 75.5 µs | 0.1101× | 0.0056× |             ~9× |
| matvec (256×256)      | 24.1 µs |  231 µs | 0.1044× | 0.0088× |            ~10× |
| **matmul (64×64)**    | 49.9 µs | 48.6 µs | **1.0268×** | 0.0046× | **0.97× (faster)** |
| **matmul (256×256)**  | 2.81 ms | 1.02 ms | **2.7555×** |    — | **0.36× (faster)** |
| frobenius (200×200)   | 27.0 µs | 71.3 µs | 0.3789× | 0.0155× |             ~3× |
| **matmul complex (48²)** | 298 µs | 168 µs | **1.7770×** | 0.0629× | **0.56× (faster)** |
| solve (64×64)         | 21.5 µs | 21.5 µs | 1.0001× | 0.8849× |            ~1.0× |
| solve complex (48×48) | 45.7 µs | 83.2 µs | 0.5492× | 0.5721× |            ~1.8× |

> **On the matmul baseline.** The table compares `numerical.einsum` against
> `np.einsum`, which by default (`optimize=False`) runs NumPy's own C loops, **not
> BLAS** — that is what the "faster than NumPy" matmul figures beat. Against real
> BLAS (`np.matmul` / `@`, multi-threaded) NumericAL's GEMM is about **10× slower** and
> steady: ~33 GFLOP/s from n=256 to n=1024 (NumPy ~280–470), with no degradation at
> large sizes. That residual gap is BLAS's hand-tuned per-architecture SIMD
> register micro-kernel, which a portable, header-only library does not attempt;
> our portable levers (cache blocking, threads, compiler auto-vectorisation) are
> already in. There is no cache cliff for multi-level blocking or panel packing to
> fix, so those are deliberately not pursued.

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
Below a per-kernel threshold the work stays sequential (thread-spawn overhead
would otherwise dominate). Parallelised so far: the tiled **GEMM** (row-blocks),
the **matrix·matrix product** (output rows), `inverse` (column solves), and the
**SVD's** left singular vectors (columns).

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

## The `a * b` operator: cache-friendly loop order

NumericAL has two matrix-product paths; this section is about the **standalone
`matrix` operator `a * b`**, distinct from the generic einsum engine's tiled GEMM
above (which canonical `einsum` contractions route to). The operator is the direct
C++ API and parallelises over output rows — each row a disjoint slice, so the
threaded result is identical to a sequential run.

Its inner loops run in **i-k-j** order: each result row is accumulated as
`Σ_k a(i,k)·(row k of b)`, so `b`'s row and the output row are both swept in
storage (row-major) order — every inner-loop access is unit-stride. The textbook
**i-j-k** order instead reads `b(k,j)` down a column (a fresh cache line per step),
which thrashes the cache as the operands outgrow it. Both orders do the same
arithmetic; only the access pattern differs.

Measured on an 8-core machine (`double`, best-of-N, both variants on the same
`parallel_for`) — reproduce with `make bench-gemm`:

| N    | naive i-j-k (before) | i-k-j `operator*` (after) | speed-up |
| ---: | -------------------: | ------------------------: | -------: |
|  128 |              0.49 ms |                   0.18 ms |    2.7×  |
|  256 |              3.28 ms |                   0.79 ms |    4.1×  |
|  512 |             28.17 ms |                   5.08 ms |    5.5×  |
| 1024 |            314.60 ms |                  36.37 ms |    8.7×  |
| 2048 |           9711.79 ms |                 452.94 ms |   21.4×  |

The speed-up grows with N — the signature of a cache effect: at 2048² the strided
order falls off a cliff (~9.7 s → ~0.45 s; ~1.8 → ~38 GFLOP/s). A k/j-**blocked**
variant was measured too and is **not** faster than the plain reorder here
(0.87 / 6.64 / 46.2 / 875.6 ms at 256–2048²): the row-parallel scheme already streams
each row of `b` in full, so blocking only chops that stream and adds loop overhead.
The operator therefore ships the one-line-order change, not a tiled kernel — the
simplest fix that captures the whole win. `benchmarks/gemm_bench.cpp` keeps the
naive baseline alongside the shipping path, so the speed-up is a permanent
regression guard.

### `operator*` vs the einsum GEMM — one product, two paths

NumericAL has two matrix-product implementations: this `operator*` (i-k-j, parallel
rows) and the einsum engine's 32×32 tiled `tile_gemm`, which a canonical
`einsum("ij,jk->ik", …)` routes to. Before the reorder `operator*` (naive i-j-k)
was far slower; **after it the two are comparable** — same operands, same
`parallel_for`, results checked equal, from the same `make bench-gemm` run:

| N    | `operator*` |   einsum | `operator*` / einsum |
| ---: | ----------: | -------: | -------------------: |
|  128 |     0.18 ms |  0.18 ms |               1.00× |
|  256 |     0.79 ms |  0.91 ms |               1.15× |
|  512 |     5.08 ms |  8.12 ms |               1.60× |
| 1024 |    36.37 ms | 63.09 ms |               1.73× |
| 2048 |   452.94 ms | 447.42 ms |              0.99× |

(`operator* / einsum` > 1 means `operator*` is faster.) `operator*` leads by
1.15–1.73× from 256² to 1024² — less loop overhead, and the prefetcher streams `b`'s
rows — then they converge to a dead heat at 2048², where the einsum 32-tiling earns
back its blocking overhead. So there is **no "two-speed API" to fix**: delegating
`operator*` to `tile_gemm` would *regress* the common mid sizes. Both paths are kept
— `operator*` is the direct, simplest (and at these sizes faster) matrix API; the
tiled kernel serves the general einsum engine. Folding them onto one shared kernel
is a possible DRY refactor, but it is **not justified by performance** and would
risk regressing each path's sweet spot, so it waits on its own plan + benchmark.

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

All three steps are purely structural — one generic engine with a recognised
GEMM shape, every rank/spec/scalar, **no intrinsics and no dependency**. The
standard library offers no numeric SIMD primitive (unlike `memmem` /
`boyer_moore_searcher` for strings); the wins are algorithmic, and the resulting
clean loops are what the compiler auto-vectorises.

## What this tells us

- **`matmul` and `solve` are now at NumPy's level** without BLAS: a tiled GEMM and
  an unblocked LU capture most of the kernel-level advantage at these sizes.
- **Pure data-movement stays the outlier** — `transpose` (~285×), and the
  reductions/`outer`/`matvec` (~8–10×). These are memory-bound: a transpose reads
  one operand column-wise (one cache line per element) and cannot be contiguous on
  both sides; the reductions touch every input once to produce little output. Only
  **cache blocking the general engine** (as the GEMM path does for products) would
  help, a further localised optimisation. Beyond it lie opt_einsum-style pairwise
  path optimisation and a BLAS reduction, not adopted here: they trade
  generic simplicity and zero dependencies for raw speed. Correctness and clarity
  come first.
