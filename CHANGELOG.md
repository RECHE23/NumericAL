# Changelog

All notable changes to NumericAL. Versions are calendar-based (`YYYY.M.PATCH`, the patch resetting each
month; PEP 440 drops leading zeros). The project holds the SciLang-stack gate: 100%-4D coverage of
`include/`, clang and GCC, sanitizers, MISRA, and a NumPy differential.

## Unreleased

### Changed
- **Decompositions take over an expiring matrix and hand their factors over without copies**: `matrix::elements()
  &&`, `tensor(matrix&&)`, constructors of LU, Cholesky and QR from a `matrix&&`, and `lower()`, `q()`, `r()`,
  `vectors()`, `u()` on an expiring decomposition move their storage; `solve`, `inverse`, `determinant` and
  `cholesky` take their matrix by value. The Python package copies each operand once and each result not at all,
  where it copied each of them twice or thrice (each copy a fresh buffer of 8 MB at order 1024, its pages faulted
  in). i9-9900K, from Python at order 1024: the inverse 32.5 → 22.1 ms (NumPy with OpenBLAS: 35.6), `solve` 20.7 →
  13.6 ms, Cholesky 15.6 → 12.3 ms, QR 57.7 → 50.2 ms, `svd` 394 → 380 ms.
- **The AVX2 tile pays for more threads**: each kernel weighs its work by a speedup of its own, a quarter of the
  others' for AVX2 (on physical cores, threads pay sooner; on the AMX, fewer threads stayed faster). i9-9900K: a
  product of order 128 73 → 39 µs, of order 192 107 → 90 µs; from Python at order 1024, Cholesky 20 → 16 ms,
  `lstsq` 148 → 131 ms, `svd` 415 → 406 ms. Its packing is weighed the same way, so a small product packs in
  parallel too: order 128 39 → 30 µs (NumPy with OpenBLAS: 23), 192 90 → 62 µs (67). AVX-512 keeps its weight
  until measured.
- **The thread pool runs one thread per physical core** where the system tells them (Linux: the distinct cores of
  the processors in the process's affinity, so `taskset` and cgroups count; macOS: `hw.physicalcpu`), not one per
  hyperthread: two threads on one core's hyperthreads share its multiply-add units. i9-9900K (eight cores, sixteen
  hyperthreads): a product of order 128 140 → 71 µs, of order 256 193 → 137 µs (NumPy with OpenBLAS: 130), the
  same at order 2048; from Python at order 1024, QR 80 → 66 ms (NumPy: 66), `eigh` 185 → 150 ms, `svd` 534 →
  426 ms, `lstsq` 173 → 145 ms, `solve` 26 → 21 ms. `get_num_threads()` reports the physical cores.
- **`einsum` of three operands or more contracts them in the cheapest order** (as `opt_einsum` chooses it):
  every order up to four operands, the cheapest in multiply-adds and then the smaller largest intermediate; past
  four, the cheapest pair sharing a label at each step. Each pair runs as the matrix products below, and an operand
  with nothing of its own to sum is no longer copied first. The order depends on the shapes alone, so a result is
  the same on any number of threads; it differs from the left-to-right order in the last bits where the order
  changes. M1 Pro: `"ij,jk,kl->il"` (200 × 10, 10 × 200, 200 × 200) 0.30 → 0.047 ms (NumPy's
  `einsum(optimize=True)`: 0.037), `"i,ij,j->"` (500) 0.85 → 0.10 ms (0.08), five 30 × 30 matrices 0.018 ms
  (0.050).
- **Every two-operand `einsum` contraction runs as matrix products** (as `opt_einsum` maps a pair to BLAS):
  each operand's axes are regrouped, labels shared with the output as batches, labels of one operand and the
  output as rows or columns, summed labels in ascending order as the depth, and the result laid out in the
  output's order. Each element takes the same terms in the same order as the canonical layout, so the bits are
  those of the canonical product; a dot product per batch sums as `"i,i->"` does. Only a label summed by one
  operand alone, or repeated within one, keeps the general loop. M1 Pro, 256 × 256: `"ij,kj->ik"` 29 → 0.18 ms
  (NumPy's `einsum(optimize=True)`: 0.12), `"ji,jk->ik"` 34 → 0.19 ms; `"bij,bjk->bik"` (32 × 64²) 15 → 0.08 ms
  (0.08); `"ijk,jkl->il"` (40³) 4.5 → 0.13 ms; `"bi,bi->b"` (1000 × 100) 0.34 → 0.021 ms (0.024).
- **A dot product (`einsum("i,i->")`) runs on vectors and every core**: chunks of 16 384 elements in parallel,
  each by several vector accumulators (AVX-512 or AVX2 for `float64` on x86-64), their sums added in the chunks'
  order: the same on any number of threads, no longer the ascending sum in the last bits. M1 Pro, from Python
  on Tensors: 100 000 elements 168 → 20 µs (NumPy's `dot`: 20, `einsum`: 63), 1 000 000 elements 1.68 ms →
  65 µs (NumPy's `dot`: 244). NumPy arrays passed in are copied into Tensors first (some 0.5 ms a million
  elements), as before.

### Fixed
- **The x86-64 kernels align the buffers they store whole registers into.** GCC 14, in C++20 under the address
  sanitizer, stored a register into an 8-byte-aligned buffer as if 64-byte-aligned: complex `eigh` crashed on an
  AVX-512 processor in such a build. Builds without the sanitizer, the published wheels among them, placed the
  buffer aligned and were not affected.

## 2026.10.5 — 2026-10-12

Faster dense linear algebra throughout (see BENCHMARKS.md for the tables against NumPy's BLAS/LAPACK).
Results that differ from 2026.10.4 in the last bits, each still the same on any number of threads: on x86-64
with AVX2 or AVX-512, the decompositions' row updates (now fused multiply-adds), real and complex; the
inverse and `solve` (a block's terms in ascending order); `lstsq` (a QR where the rank is surely full); and
complex products and decompositions past the portable path. `limit_gemm_path("portable")` keeps the classical
sums of the matrix product and of the row updates.

### Changed
- **`eigh` reduces to tridiagonal form faster**: each row of the trailing block is read once for both halves of
  its symmetric product (as `dsymv` reads it), on AVX-512 or AVX2 with FMA where the matrix product's kernel
  runs there (`gemm_path`), and the trailing update after each panel touches the lower triangle alone, the only
  one the reduction reads (as `dsyr2k`), in one product of twice the depth. Each step's partial sums no longer
  map their buffer anew. i9-10900X (AVX-512), order 1024: the reduction 243 → 170 ms on one thread, 153 → 125
  ms on twenty; `eigh` 275 → 237 ms. M1 Pro: the reduction 125 → 100 ms on one core.
- **`eigh`'s divide and conquer joins its halves on every core**: Löwner's weights and the eigenvectors'
  lengths are computed per pole and per vector in parallel, and the eigenvectors are put in ascending order row
  by row (they were gathered and written column by column, a stride of the whole order each). The same
  operations in the same order: the results are the same to the bit. Order 1024: i9-10900X 88 → 62 ms (137 →
  113 on one thread), M1 Pro 64 → 42 ms. Each join's matrix product then skips the zeros of the halves'
  block-diagonal eigenvectors, a product per half on the columns that reach it, as `dlaed3` multiplies them:
  i9-10900X 123 → 105 ms on one thread.
- **`eigh` and `svd` form their bases transposed**, as they return them: each block of reflectors is applied
  from the right to `Qᵀ` (or `Uᵀ`, `Vᵀ`), so that all three of its products have as many rows as the basis,
  where the block's few rows (32) made the first a product the kernel runs poorly; and no transposition is
  left at the end. i9-10900X, order 1024: the tridiagonal reduction 128 → 120 ms.
- **The decompositions' row updates and dot products run on AVX-512 or AVX2 with FMA** on x86-64, where
  the matrix product's kernel runs there (`gemm_path`; `limit_gemm_path("portable")` keeps them portable): a
  wheel is built for the baseline x86-64, two `double` per vector and no fused multiply-add. Each element is
  then one fused multiply-add, so LU, Cholesky, QR, the inverse, `eigh` and `svd` differ in the last bits from
  2026.10.4 on such processors. i9-10900X (a shared machine): `solve` of order 1024 about 30 → 25 ms, the
  inverse 63 → 57 ms, `svd` of order 512 about 165 → 134 ms; `eigh` 3–5 % faster.
- **Complex row updates run on AVX-512 or AVX2 with FMA** on x86-64, where the matrix product's kernel runs
  there: the triangular solves' row updates (`y −= Σ f·x`, each term `Re f · x` then `Im f` times the swapped
  parts, a fused multiply-add per part, the row held in registers through the terms), the single-term updates of
  the decompositions, and each row of `eigh`'s Hermitian product in one pass (`out += conj(row) · vᵢ` and
  `Σ row · v`). Complex results there differ in the last bits; the portable path and arm64 keep the template's
  complex products. i9-10900X, `complex128`: the inverse of order 256 13 → 6.8 ms (NumPy with OpenBLAS: 2.1),
  of order 1024 202 → 154 ms (64); `eigh` of order 512 145–188 → 112–125 ms (63), of order 1024 756–947 → 510 ms
  (335).
- **The matrix product keeps each A sliver in the L1 cache across its task's B slivers**, as BLIS orders its
  loops (the B sliver, 32 KiB on AVX-512, filled the L1 when it was the outer one). The same results to the bit.
  One core: i9-10900X (AVX-512) 72 → 77 GFLOP/s at order 2048, i5-8500 (AVX2) 47 → 49, M1 Pro (AMX) 205 → 214;
  all cores: i5-8500 207 → 222, M1 Pro 510 → 570 at order 1024.
- **LU looks one panel ahead** where the trailing block is at most 1024 columns wide: the next panel's columns
  are updated first, then the panel is factorised on the calling thread while the workers update the rest; its row
  swaps reach those columns once both are done. The same arithmetic in the same order: the factors are the same
  to the bit. M1 Pro: LU of order 512 2.33 → 2.05 ms, 1024 10.4 → 8.6 ms; x86-64 within noise.
- **`lstsq` answers by one QR factorisation where the rank is surely full**: `R⁻¹` gives a bound on the
  condition number, `‖R‖_F ‖R⁻¹‖_F`, and where it stays a hundred times under the reciprocal of NumPy's rank
  tolerance, `X = R⁻¹ Qᴴ B` (or `Q R⁻ᴴ B`, from a QR of `Aᴴ`, for a wide `A`) is the singular value
  decomposition's solution to rounding; elsewhere the decomposition decides, as before. `Q` is never formed: its
  blocks of reflectors are applied to `B`. The solution differs from 2026.10.4 in the last bits. M1 Pro:
  256 × 256 12.7 → 2.0 ms (NumPy with Accelerate: 7.1), 2000 × 1000 353 → 87 ms (208), 500 × 1000 87 → 20 ms
  (42), 4000 × 200 48 → 27 ms (23). i9-10900X: 256 × 256 21 → 4.8 ms (NumPy with OpenBLAS: 11.6), 2000 × 1000
  667 → 167 ms (239), 500 × 1000 121 → 31 ms (68), 4000 × 200 69 → 38 ms (33).
- **QR reflects each panel in a column-major copy**, where a reflector meets each column at unit stride (row by
  row, each row's part of the 32-column panel was a call of its own): QR 8–12 % faster on M1 Pro.
- **Complex products run on the real kernel** past the portable path (AVX2, AVX-512, AMX): one real product
  of twice the order, `[Re A | −Im A]·[[Re B, Im B], [Im B, −Re B]]`, by blocks of 512 columns and 256 of depth,
  where the complex tile was scalar and single-threaded in effect. Each element sums the products of A's real
  parts, then of its imaginary parts, in a fixed order on any number of threads: complex products, and the
  complex decompositions through them, differ in the last bits; `limit_gemm_path("portable")` keeps the classical
  sums. M1 Pro, `complex128`: a product of order 1024 188 → 23 ms (Accelerate: 30), the inverse 260 → 117 ms,
  `eigh` 642 → 418 ms. i9-10900X: the product 118 → 25–34 ms (OpenBLAS: 18), the inverse of order 512 84 → 49
  ms, `eigh` 298 → 170 ms.
- **Every matrix product and parallel loop asks the system for its thread count once**: glibc reads a file
  for it on each call, three system calls per product, which a container's file system (LXC's lxcfs) serves
  slowly. i5-8500 (six AVX2 cores) in a container: a product of order 128 by 64 290 → 52 µs, the inverse of
  order 128 1.8 → 0.41 ms and of order 1024 64 → 33 ms. Outside one, i9-10900X: the inverse of order 64 0.12 →
  0.06 ms.
- **The inverse, `solve` and LU solve their triangles a block at a time, in parallel**: within each block of
  64 rows, every chunk of columns runs the whole block's substitution at once (its columns depend on no other's),
  each row taking all its terms in registers on AVX-512 or AVX2 (a call per term, and a 16-row leaf through the
  matrix product, before). The inverse also leaves out the products with the zeros of `L⁻¹`'s upper triangle
  (as `dgetri`), and permutes its columns at the end. Each element still takes its terms in a fixed order, the
  same on any number of threads; the backward pass takes a block's in ascending order, so the inverse and `solve`
  differ in the last bits. i9-10900X (a shared machine), twenty threads: the inverse of order 1024 54 → 41 ms,
  of order 2048 235 → 180 ms (NumPy with OpenBLAS: 31, 219 ms); `solve` with 1024 right-hand sides 48 → 33 ms,
  with 16 of order 128 0.46 → 0.24 ms. M1 Pro: the inverse of order 2048 165 → 128 ms.
- **The matrix product zeroes no tile it computes in place**: a partial tile still goes through a zeroed copy,
  but whole tiles (nearly all of a large product) took the zeroing too, 1.5 KiB per AVX-512 tile (4 KiB on the
  AMX), a twelfth of an x86-64 inverse's samples. The same results to the bit.
- **`eigh` no longer zeroes buffers it writes whole** (each join's root gaps and eigenvector weights, megabytes
  at the top join; the panel's and the reflector blocks' copies): 3.4 % fewer instructions at order 512
  (callgrind), the same results to the bit.

### Fixed
- `make bench-linalg` without `SIZES` measured order 64 alone: the default list's commas split the arguments of
  the `$(or …)` it sat in.


## 2026.10.4 — 2026-10-12

### Changed
- **`eigh` and `svd` default to divide and conquer** (see Added), as LAPACK's `dsyevd`/`zheevd` and `dgesdd`, and
  matrix products run on a kernel chosen at run time (AVX-512, AVX2 or the Apple AMX, each element a chain of
  fused multiply-adds). Results differ from 2026.10.3 in the last bits, and within a repeated eigenvalue or
  singular value the basis may differ (both are orthonormal to working precision); `lstsq`, `pinv` and
  `matrix_rank` go through `svd`. The previous methods remain: `eigh(a, "tridiagonal_ql")` and
  `svd(a, "bidiagonal_qr")` (C++: `eigen_method::tridiagonal_ql`, `svd_method::bidiagonal_qr`); the new
  enumerators come last, so existing values keep their meaning.
  `limit_gemm_path("portable")` (or `NUMERICAL_GEMM=portable`) keeps products off the run-time kernels.
- **A matrix product no longer maps its packing storage anew each time**: the calling thread keeps it for the
  next product (up to 32 MiB). Fresh megabytes were faulted in page by page on every product: order 512 on 20
  x86-64 threads took 2.9 ms instead of 0.89. The packed operands start a cache line, so no vector load of
  them straddles two.
- **`asarray` is 10–20× faster** on a NumPy array (1e6 float64: 11 → 0.8 ms; arm64, Apple clang 16, `-O2`): the
  element type is resolved once per copy rather than per element, the innermost axis is one strided loop (a
  contiguous run of the output's type one `memcpy`), and the GIL is released during the copy.
- **Element-wise arithmetic, powers and extremes at NumPy's pace** (1e6 float64, NumPy's time in brackets):
  `x * 2.0` 3.4 → 0.80 ms (0.63), `M + row` 3.4 → 0.84 (0.80), `x ** 2` 18 → 0.80 (0.69), `z ** 2` (complex)
  80 → 0.68 (0.52), `max` 2.5 → 0.24 (0.14), `min` along axis 0 2.4 → 0.34 (0.23). Broadcasting runs the
  innermost axis as one loop; a single exponent of 2, 0.5 or −1 is `x·x`, `sqrt(x)` or `1/x`, as NumPy computes
  them (so `(1+1j) ** 2` is exactly `2j`); `max`/`min` compare over native vectors.

- **The kernels share a pool of threads started once**, rather than starting threads on every call: each
  run hands its chunks out as threads free up (a slower core takes fewer), a run nested in another or started
  while the pool is busy runs on its caller, and results are the same to the bit for any thread count. Matrix
  product of order 128–256: 5.4–6.1× → 4.2–5.0× and 3.0× → 2.4× NumPy's time (arm64, Apple clang 16, `-O2`).

- **Matrix products use more of the cores**: both operands are packed in parallel (one thread packed them
  while the others waited) into buffers no longer zeroed first. M1 Pro, 8 threads, float64: order 512
  153 → 178 GFLOPS, 1024 195 → 207 (one core: 41, about 80 % of its vector peak).

- **Medium matrix products wait less for their threads**: an idle thread polls for 0.1 ms before it sleeps, so
  the passes of a product (packing, then computing, panel after panel) find it awake, and a product too small
  to give each thread four tasks of whole blocks is cut into tasks of fewer whole tiles. Twenty x86-64 threads,
  float64: order 192 131 → 107 µs, 256 279 → 105, 384 490 → 349. M1 Pro: order 128 47 → 28 µs, 256 124 → 96;
  inverse, Cholesky and QR of order 128–384 5–17 % faster.

- **LU solves, inverses and QR faster when built by GCC** (the Linux wheels): the row updates of the
  factorisation, of the triangular solves and of the Householder reflections run over native vectors, where
  GCC's `-O2` left them scalar (Clang vectorises them itself, and keeps its loops). Each element is computed as
  before. x86-64, float64: `solve` of order 128–512 1.2–1.4× faster, `inverse` of order 256–512 1.2–1.3×.

- **The inverse sends more of its triangular solves through the matrix product kernel**: each block of 64
  rows is solved by leaves of 16, from a leaf to the rest of its block by a product. x86-64 (twenty threads,
  float64): order 128 1.1 → 0.55 ms, 192 2.7 → 1.6, 384 13 → 8.5, 512 20 → 16; arm64 unchanged.

- **LU factorisation solves its rows of U by leaves**, a product between them, factorises a tall panel
  recursively as LAPACK's dgetrf2 (its bottom updated by products, in parallel), and its panel's column updates
  run over native vectors under GCC. float64 `solve`: x86-64 (twenty threads) order 192 0.96 → 0.43 ms, 256
  1.85 → 1.03, 512 8.2 → 4.3–5.4; M1 Pro order 128–1024 5–15 % faster. The factors are unchanged where no product
  is fused.

- **Cholesky solves the rows below each panel by leaves**, a product for each leaf's terms from the panel's
  earlier columns. x86-64 float64 (twenty threads): order 128 0.25 → 0.16 ms, 256 0.93 → 0.72, 384 3.3 → 2.8;
  arm64 and orders of 512 and more unchanged. The factor is unchanged where no product is fused.

- **`eigh` reduces to tridiagonal form by blocks and forms its vectors by blocks**, as LAPACK's `dsytrd` and
  `dorgtr`: each panel of 32 reflections updates the trailing block at once through the matrix product kernel,
  the product `A v` of each reflection reads the lower triangle alone, each row once for its two uses (as
  `dsymv`), on every core, and the accumulated reflections go through the kernel by blocks. The rotations of a QL
  or QR sweep reach the vectors as LAPACK's `dlasr` applies them, each row read and written once per sweep. M1
  Pro, float64: order 256 13.8 → 9.1 ms, 512 107 → 54 (Accelerate 26); the reduction alone at order 1024 on
  twenty x86-64 threads 218 → 123 ms.

- **`svd` bidiagonalises on every core**: each reflection's columns (from the left) or rows (from the right)
  run in parallel, and a right reflection's products sum over native vectors. M1 Pro, float64, order 256 21.7 →
  18.3 ms, 512 146 → 114 (Accelerate 49).

### Added
- **`eigh` diagonalises the tridiagonal by divide and conquer**, as LAPACK's `dsyevd` and `zheevd`: the
  halves are joined by a rank-one update (Gu and Eisenstat's deflation, the secular equation solved from its
  nearer pole, Löwner's formula for orthogonal vectors), most of the work in matrix products. The default for
  `symmetric_eigen` (`eigen_method::divide_and_conquer`; implicit QL stays as `tridiagonal_ql`) and for
  `hermitian_eigen`; up to order 32, QL alone, as before. M1 Pro, float64, order 512: 54 → 40 ms (Accelerate
  26), the tridiagonal step 30 → 7.5 ms.
- **`svd` reduces a real matrix to bidiagonal form by panels**, as LAPACK's `dgebrd`: each panel of 32
  reflections from each side updates the trailing block at once through the matrix product kernel, and the
  block reflectors' Gram matrix (QR, `eigh` and `svd` form their bases from it) is one product too. M1 Pro,
  float64, order 512: `svd` 88 → 67 ms (Accelerate 49), `eigh` 40 → 37.
- **`svd` decomposes the bidiagonal by divide and conquer**, as LAPACK's `dgesdd`: a bidiagonal splits at a
  row into two with an extra column each side, whose null vectors rotate into one, and the arrow left in the
  middle is decomposed through its secular equation (solved in the squares from the nearer pole, so that both
  `d − σ` and `d + σ` are accurate) and Löwner's formula. The default (`svd_method::divide_and_conquer`;
  implicit-shift QR stays as `bidiagonal_qr`). M1 Pro, float64, order 512: 108 → 88 ms (Accelerate 49), the
  bidiagonal step 45 → 9. A zero arrow, which has no weight on its zero pole and so no root, is its own
  decomposition.
- **`make fuzz-linalg`**: every decomposition against NumPy on adversarial matrices (clustered, repeated and
  graded spectra, decoupled blocks, rank deficiency, extreme scales, complex) past the block and leaf
  boundaries, on every matrix-product kernel the processor runs, each answer held to a backward-error bound; a
  short fixed campaign runs with the Python tests. **`make bench-linalg`**: NumericAL against NumPy's
  BLAS/LAPACK, each library alone in its process (the tables in BENCHMARKS.md).
- **`float64` matrix products on the Apple AMX** (M1 to M4), the matrix coprocessor Accelerate uses. M1 Pro,
  8 threads, GFLOPS against Accelerate's, the product alone timed in C++: order 2048 423 (383), 1024 395
  (388), 512 277 (431), 256 209 (282); the vector units gave 204, 207, 178 and 145. One core: 160–180 (was
  41). Apple documents no instruction for it, so the path is guarded: compiled only for arm64 macOS (not at
  all under `NUMERICAL_NO_AMX`), used only on a CPU family of the M1 to M4 line, and only once a product
  computed on it equals, to the bit, the same product by `std::fma` in ascending order. On the AMX each element
  of a product is a chain of fused multiply-adds in ascending order of the depth, the same on any number of
  threads; it differs from the vector path in the last bits, as any fused product does. LU, Cholesky, QR, the
  inverse, `eigh` and `svd` take it through their matrix products.
- **`float64` matrix products on AVX-512 and AVX2 with FMA** (x86-64), chosen at run time: a wheel built for
  the baseline x86-64 compiles each tile for its instruction set and uses it only once `cpuid` reports the
  instructions and `xgetbv` the system saving their registers. Each element is a chain of fused multiply-adds
  in ascending order of the depth, the same on any number of threads. GFLOPS against OpenBLAS 0.3.34's
  (NumPy's), float64, the product alone timed in C++, each library alone in its process (`make bench-linalg`
  times it through Python, with its conversions: BENCHMARKS.md):
  i9-10900X (AVX-512, 10 cores), order 1024 540 (720), 2048 420 (590); one core 66–79 (82–99), was 11.
  i9-9900K (AVX2, 8 cores), order 1024 271 (353), 2048 213 (304); one core 54–58 (69).
- **`eigh(a, method)` and `svd(a, method)`** choose the algorithm, as the C++ classes did: `"divide_and_conquer"`
  (the default), `"tridiagonal_ql"` or `"jacobi"` for `eigh`, `"bidiagonal_qr"` or `"jacobi"` for `svd`; a name
  neither knows is a `ValueError`. `hermitian_eigen` takes an `eigen_method` too (divide and conquer or implicit
  QL; the Jacobi method, real only, is refused with `std::invalid_argument`).
- **`numerical.gemm_path()` and `limit_gemm_path(name)`** (C++: `numerical::gemm_path_in_use`,
  `limit_gemm_path`) report and cap the kernel float64 products run on, `"portable"`, `"avx2"`, `"avx512"` or
  `"amx"`: a cap takes the most capable path at or below it that runs here; `"portable"` is the vectors the
  compiler makes of `c + a·b`, fused or not as it compiles them. `NUMERICAL_GEMM` sets the cap at import,
  `NUMERICAL_AMX=0` keeps products off the AMX alone, and `get_config()["gemm_path"]` reports it.
- **`numerical.set_num_threads(n)` and `get_num_threads()`** (C++: `numerical::set_max_threads`,
  `max_threads`), and `NUMERICAL_NUM_THREADS` read at import, as `OMP_NUM_THREADS` caps BLAS: a process per core
  can keep the kernels to one thread each.

### Fixed
- **`linspace` equals NumPy's to the bit on arm64 Linux too.** The aarch64 wheel (GCC) fused each `i · step`
  into the addition after it, as GCC does across statements wherever a fused multiply-add exists: 19 of
  `linspace(-2, 5)`'s 50 values differed in the last bit. The product is now computed rounded, by
  `fma(i, step, -0)`, which no compiler can fuse. A CI leg runs the Python suite on arm64 Linux, built by GCC
  as the wheels are.
- **The thread cap holds**: under `set_num_threads(n)` (`set_max_threads`) a run takes the caller and at most
  `n − 1` workers (every worker joined every run, whatever the cap); at 1, and to answer `get_num_threads`, no
  thread is started. A process forked while the pool exists runs its kernels alone, rather than on workers it
  does not have (a child could hang on the parent's lock).
- **Below the AMX, the AMX is never probed**: `limit_gemm_path` under it, `NUMERICAL_AMX=0` or
  `NUMERICAL_GEMM=portable` keep even its self-test from running (a cap probed it first), and it is not used in a
  virtual machine, whose host decides what the processor offers. `NUMERICAL_AMX` also reads false, no and off, the
  more portable of it and `NUMERICAL_GEMM` wins, and a value neither reads, like an invalid
  `NUMERICAL_NUM_THREADS`, is ignored with a warning; `set_num_threads` refuses a negative count or a boolean
  with `ValueError`.
- **Subnormal entries no longer cost the bases their orthogonality** (in 2026.10.3 too): a Householder vector or a
  phase computed among subnormal numbers missed unit length by the coarse grid's error, which the input scaling
  does not remove when one column alone is tiny (`eigh` of a matrix with a column of 1e-320: ‖VᵀV − I‖ 1e-3;
  `qr` of a 1e-318 matrix: 6e-6; a Hermitian tridiagonal with 1e-321 couplings: 0.3). Each is now scaled up by an
  exact power of two first, as LAPACK's `dlarfg` rescales; nothing changes for normal numbers.
- **`asarray` of an `__array_interface__` whose type names an absurd size aborted the interpreter** (since
  2026.10.1); it is refused as an unknown type. A source with rows of no elements at a null address no longer
  reaches `memcpy` with it.
- **`eigh` reads only what it uses**: the lower triangle and the diagonal's real part, as `numpy.linalg.eigh`. A
  NaN in the upper triangle (or in the diagonal's imaginary part) raised; the lower triangle's are refused as before.
- **`max`/`min` along an empty axis raise**, as NumPy's, even when the result would be empty too
  (`min(zeros((3, 0, 0)), axis=1)` returned an empty tensor).

## 2026.10.3 — 2026-10-11

### Fixed
- **The sdist builds.** It carried no `include/` (no `MANIFEST.in`), so `pip install` on a platform without a
  wheel (free-threaded CPython, ppc64le, s390x, BSD…) failed on a missing header. CI now builds the sdist,
  installs it from source in a clean venv and runs the tests.
- **`lu_decomposition<long double>` compiles** (and `solve`, `determinant`, `inverse` with it): the pivot
  search held its magnitudes in a `double`, a narrowing for `long double` and `std::complex<long double>`.
  A test now instantiates every decomposition for `float`, `double`, `long double` and their complex forms,
  which compile cleanly under MSVC `/WX` and on x86-64, where `long double` is wider: every magnitude the
  library carries in `double` is converted to it explicitly, and a `complex<float>` is built from a `float`.
- **A boolean is not an index**: `t[True]` read row 1 and `t[False]` row 0 (`numpy.True_` too, before NumPy
  2.5), where NumPy reads a boolean as a mask. It raises `IndexError` now.
- **A Tensor's truth value and comparisons no longer answer wrongly**: `bool(t)` was `len(t) > 0`, and `a == b`
  compared identity. As NumPy's, `bool` gives the value of a one-element tensor and raises `ValueError` for
  any other size; `==`, `!=`, `<`, `<=`, `>`, `>=` raise `TypeError` until comparisons are element-wise, and a
  Tensor is unhashable. `numerical.array_equal(a, b)` compares two tensors whole, as NumPy's.
- **`arange` steps by the step its first two elements hold**, `(start + step) − start`, as NumPy's, and rounds
  each element once. The previous release said it equalled NumPy's to the bit; it differed by up to 1 280 units
  in the last place (`arange(1e16, 1e16 + 100, 1.0)`). It now equals NumPy's on arm64, where NumPy fuses the
  multiply-add; a NumPy build without fusion rounds twice and can differ in the last bits.

### Added
- **`repr(t)` shows the values**, one row per line, as NumPy's: `numerical.Tensor([[1.0, 2.0],` / `[3.0, 4.0]],
  dtype=float64)`; past 1000 elements each long axis keeps its first and last three entries around `...`.
- **`float(t)`, `int(t)`, `complex(t)` of a rank-0 tensor** (a full reduction along an axis of a vector, say),
  as NumPy's; any other rank raises `TypeError`.

### Changed
- **The extension is compiled without debug information** (the Linux wheel's was 7.8 MB with it).

## 2026.10.2 — 2026-10-11

### Fixed
- **An object that only claims to be a `Tensor` no longer crashes the interpreter.** `determinant`,
  `slogdet` and `asarray` trusted `isinstance`, which a mock with `spec=Tensor` passes, and read a tensor
  out of it (a segmentation fault). The object's real type is checked now; the impostor is refused.

### Added
- **A `Tensor` computes as a NumPy array**: `+ - * /` element by element with NumPy's broadcasting (numbers,
  lists and the reflected forms too; a real operand meeting a complex one is widened), `-t`, `t.sum()`,
  `t.mean()`, `t.max()`, `t.min()` of every element or along an `axis` (also as module functions),
  `t.reshape(…)` with one `-1`, and `numerical.zeros`, `ones`, `full`, `eye`, `arange`, `linspace`. Real
  arithmetic and creation equal NumPy's to the bit; sums follow NumPy's pairwise algorithm (its vector builds
  group terms their own way, so a long sum can differ in the last bits); a complex product or quotient agrees to
  a few units in the last place. `max` and `min` refuse complex128. C++: `tensor_ops.hpp`.
- **Element-wise functions**: `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `tanh`, `absolute` (`abs(t)`, float64 for
  either dtype), and `power` (`t ** p`, `p ** t`), as NumPy's to a few units in the last place. C++:
  `numerical::map` and `numerical::pow`.
- **Every function takes array-likes, as NumPy's do**: a NumPy array, a nested list or a number goes through
  `asarray` (zero-copy where it can); `nm.solve(np_array, [1.0, 2.0])` raised `TypeError: expected a
  numerical.Tensor`.
- **A matrix right-hand side for `solve` and `lstsq`** (`b` of shape `n × k`, one system per column), as
  NumPy's: one factorisation, then every column. C++: `solve(matrix, matrix)` and `lstsq(matrix, matrix)`.
- **`qr` of a wide matrix** (`m < n`), as NumPy's reduced `qr`: `Q` is `m × m`, `R` is `m × n` upper
  trapezoidal; `solve` of a wide `qr_decomposition` refuses its underdetermined system (`lstsq` solves it).
- **A real operand meets a complex one**: the real ones are widened to `complex128`, as NumPy promotes them
  (`einsum`, `solve` and `lstsq` refused operands of two dtypes).
- **A `Tensor` indexes, iterates, transposes and multiplies as a NumPy array**: `t[i]`, `t[i, j]`,
  `t[1:5:2, ::-1]` (basic indexing, negative indices and steps; all integers give a number), `len(t)`,
  `for row in t`, `t.T`, `a @ b` for vectors and matrices (two vectors give a number), `t[()]` for a rank-0
  tensor's number. A Tensor is never written through, so its indexed copies read as NumPy's views.
- **`eigh` of a complex Hermitian matrix**, as `numpy.linalg.eigh` takes it: real eigenvalues, complex
  eigenvectors (C++: `hermitian_eigen`). It refused anything but `float64`.
- **`LinAlgError`**, raised by every linear-algebra function where it cannot compute its answer (singular, not
  positive definite, non-finite, no convergence, wrong shape). It is a `ValueError`, as
  `numpy.linalg.LinAlgError` is, and a `numerical.error`, so existing `except numerical.error` keeps working.

### Changed
- **The matrix product is 1.3 to 3.2 times faster** (`a * b`, `einsum("ij,jk->ik", …)`, and the Python `@`
  and `einsum` over them): a packed, register-tiled kernel in the loop structure of BLIS (`gemm.hpp`), the tile
  held in native vectors for `float` and `double` under GCC and Clang. `double`, best of N: arm64 with 8 cores,
  Apple clang 16, `-O2`, 1.3× at 128 to 3.2× at 2048 (121 ms, 142 GFLOP/s); x86-64 AVX2 with 2 cores, GCC 14
  `-O2` without `-march`, 1.7× at 128 to 3.0× at 2048. Against BLAS (`np.matmul`) the gap goes from about 10×
  to about 3×. Every element is still summed in ascending order of the inner index, the order of the loop it
  replaces, on any number of threads; `a * b` and `einsum` now run the same code and agree to the bit. Products below
  8³ multiply-adds or shallower than 8 keep the plain loop.
- **The LU factorisation is 4.9 times faster at order 1024** (80.8 → 16.6 ms; 1.8× at 256, 3.0× at 512), and
  with it `determinant`, `slogdet`, `solve` and `inverse`: blocked as LAPACK's `dgetrf`, panels of 64 columns
  factorised in a column-major copy and the trailing block updated by the matrix product kernel. It regroups
  the unblocked algorithm's updates without reordering any, so without multiply-add fusion the factors are the
  same to the bit. Against LAPACK the gap goes from 6–13× to 2.4–3.5× (arm64 with 8 cores, Apple clang 16,
  `-O2`).
- **`inverse` is 5.2 times faster at order 1024** (241 → 46 ms; 3.3× at 512), and `solve` with a matrix
  right-hand side with it: every column is solved at once on the factors, block of rows by block of rows,
  through the matrix product kernel. A column agrees with the vector `solve` to rounding (the backward pass
  takes the farther blocks' terms first). Against LAPACK the gap goes from 11× to 2.2×. C++:
  `lu_decomposition::solve(matrix)`.
- **`svd` is 1.7 times faster again at order 1024** (1.67 → 1.01 s) and **`eigh` 1.2 times** (0.98 → 0.80 s):
  the singular vectors are formed from the bidiagonal reduction's reflectors by blocks through the matrix product
  kernel, and the rotations of the QR and QL iterations are recorded and applied once, in parallel by columns,
  each element receiving them in the same order. What remains of the gap to LAPACK (about 4× for both) is the
  unblocked bidiagonal and tridiagonal reductions.
- **`qr` is 3.9 times faster at order 1024** (393 → 101 ms; 2.7× at 512): blocked as LAPACK's `dgeqrf`, the
  reflectors of a panel of 32 columns applied to the rest at once as `I − V T Vᴴ` through the matrix product
  kernel, and `Q` formed the same way. Against LAPACK the gap goes from 6.3× to 1.7×.
- **`cholesky` is 6.2 times faster at order 1024** (139 → 22.5 ms; 1.9× at 256, 3.5× at 512): blocked on the
  same kernel, each panel's rows below its diagonal block in parallel; without multiply-add fusion the factor is
  the unblocked one to the bit. Against LAPACK the gap goes from about 12× to 1.8× at 1024.
- **`svd` is 48 times faster** (512 × 512: 11.2 → 0.24 s; 256 × 256: 1.0 → 0.031 s; arm64, Apple clang 16,
  `-O2`): after the Householder QR, `R` is reduced to bidiagonal form and diagonalised by implicit-shift QR,
  the method of LAPACK's `dgesvd`; the reconstruction error of a 512 × 512 falls from 6e-13 to 5e-14. One-sided
  Jacobi, which gives every singular value to high relative accuracy, stays on request and is itself 6.6 times
  faster (its sweeps rotate contiguous rows). C++: `svd_decomposition` takes the method as a second parameter
  (`svd_method::bidiagonal_qr`, the default, or `svd_method::jacobi`). `lstsq`, `pinv` and `matrix_rank` go
  through the default.
- **`eigh` is 100 times faster and more accurate** (512 × 512: 14.6 s → 0.12 s; 256 × 256: 785 → 14 ms; arm64,
  Apple clang 16, `-O2`): the matrix is reduced to tridiagonal form by Householder reflections, then diagonalised
  by implicit QL, the method of LAPACK's `dsyev`. Cyclic Jacobi took O(n³) per sweep and stopped at 1e-12 of the
  matrix norm; the residual `|AV − VΛ|` of a 256 × 256 falls from 4e-13 to 5e-14. C++: `symmetric_eigen`'s
  second parameter is now the method (`eigen_method::tridiagonal_ql`, the default, or `eigen_method::jacobi`);
  it was a sweep bound.
- **QR is 9 to 14 times faster** (512 × 512: 630 → 44 ms; 256 × 256: 48 → 5.6 ms; arm64, Apple clang 16, `-O2`):
  each Householder reflection walked the row-major matrix column by column; it now walks it row by row, on
  contiguous memory. `qr`, `lstsq`, `pinv`, `matrix_rank` and `svd` (which starts with a QR) all use it.

## 2026.10.1 — 2026-10-11

### Added
- **`asarray` reads what NumPy users pass it**: bool, integer, float32 and complex64 arrays (as
  float64 or complex128), in either byte order, at any strides (`a.T`, `a[::-1, ::2]`, Fortran order),
  nested lists and tuples of numbers, and plain numbers. It refused everything but C-contiguous float64
  and complex128.
- **`einsum(spec, operands)` over a `std::vector` of operands** counted at run time, for a caller that
  holds them in a container (SciLang's interpreter called `detail::einsum_impl` for this).
- **`slogdet`, `pinv` and `matrix_rank`**, as `numpy.linalg` gives them (C++: `lu_decomposition::
  log_determinant`, `pinv`, `matrix_rank`). `determinant` of `1e-2·I` of order 200 underflows to 0.0;
  its `slogdet` is `(1.0, -921.03)`.

### Changed
- **SciForge v2026.10.2**: CI pins and the `sciforge-build` floor (was 2026.6.x), so an isolated wheel
  build compiles against the hardened binding headers.
- **`svd` returns `(U, S, Vh)`, as NumPy's `svd(a, full_matrices=False)` does** (Python; C++ adds
  `svd_decomposition::vh()`). The third factor was `V`, so NumPy's idiom `(U * S) @ Vh` reconstructed a
  wrong matrix without an error. Code that transposed the third factor itself must stop doing so.
- **The SVD takes any shape and a complex matrix, and keeps its small singular values.** It was
  the eigendecomposition of `Aᵀ A`, which squares the condition number: true singular values
  `[1, 1e-4, 1e-8, 1e-12]` came back as `[…, 9.7e-9, 0]`, and `U` lost its orthonormality where
  `A` was rank-deficient. It is now a Householder QR then one-sided Jacobi on `R`, which gives each
  singular value to high relative accuracy; `U` (`m × k`) and `V` (`n × k`), `k = min(m, n)`, stay
  orthonormal at any rank. `svd_decomposition` is a class template (`svd_decomposition<T>`, deduced
  from its argument).
- **QR is computed by Householder reflections.** Modified Gram–Schmidt lost orthogonality on an
  ill-conditioned matrix (Hilbert-10: `|QᵀQ − I|` 2.2e-4) and refused Hilbert-12 as "linearly
  dependent", which now factorises with `|QᵀQ − I|` 1.8e-15. A rank-deficient matrix factorises too
  (with a zero on `R`'s diagonal); `qr(a).solve(b)` refuses it, and `lstsq` solves it.
- **`lstsq` gives NumPy's minimum-norm least-squares solution, for any shape and rank**, from the
  SVD. Through QR it lost accuracy on a polynomial fit (Vandermonde of degree 11: a coefficient off
  by 0.60; the test's degree-11 fit now recovers each within 1e-9) and refused a rank-deficient or
  wide system.
- **`eigh` reads the lower triangle, as `numpy.linalg.eigh` does by default**; it read the upper one
  of a matrix that is not symmetric.
- **A large computation releases the GIL** (from 64 × 64 elements): `solve`, `inverse`, `lstsq`,
  `cholesky`, `qr`, `eigh`, `svd`, `determinant` and `einsum` run in parallel from several Python
  threads.
- **`einsum` over three operands or more contracts them two at a time**, after summing each over the
  labels only it carries: `ij,jk,kl->il` of order 60 takes 0.11 ms instead of 29 (arm64, Apple clang 16,
  `-O3`), as it ran one loop over every label. `i,j,k,l,m->` over five vectors of 65 536 ones gives 2⁸⁰
  exactly, where it raised a loop-count overflow.
- **A parallel loop starts as many threads as its work pays for.** It started one per hardware thread
  as soon as it had 16 or 64 indices, whatever their work: a 16 × 16 `inverse` took 177 µs instead of
  8, a 64 × 64 matrix product 95 µs instead of 35.

### Fixed
- **Every decomposition refuses an infinite or NaN entry** (`non_finite_error`), named after the
  decomposition asked for. `eigh` of `[[1, nan], [nan, 2]]` returned `[1, 2]`.
- **An exception in a parallel worker reaches the caller** once every worker has joined; it called
  `std::terminate`.
- **`asarray` refuses an `__array_interface__` whose data address is null** while the shape has
  elements, and one whose `data` tuple is empty; both crashed the interpreter.
- **`einsum` no longer wraps its count of terms**: five summed labels of 65 536 wrapped it to 0 and
  returned 0.0 (the count is checked, and such a spec now sums each operand alone first).
- **The Jacobi methods bound their sweeps** (`convergence_error`) instead of looping without end.

## 2026.10.0 — 2026-10-10

First release under the name NumericAL (formerly SciNum): dense vectors, matrices and tensors with
`einsum`, LU, Cholesky, QR, symmetric eigen and SVD decompositions, and a zero-copy Python binding
(`pip install numeric-al`, `import numerical`).
