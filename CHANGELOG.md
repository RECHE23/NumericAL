# Changelog

All notable changes to NumericAL. Versions are calendar-based (`YYYY.M.PATCH`, the patch resetting each
month; PEP 440 drops leading zeros). The project holds the SciLang-stack gate: 100%-4D coverage of
`include/`, clang and GCC, sanitizers, MISRA, and a NumPy differential.
A change that can break code is listed under **Breaking** and held for a month's first release (see
`RELEASING.md`).

## Unreleased

### Changed
- **`fft` and `ifft` of a contiguous line load and finish it vectorised** (`fft_load_line`, `fft_finish_line`): the
  copy into the passes' input, the conjugation of an inverse and the scaling, where each element was copied alone, its
  bounds and flags reloaded at every element. The same bits. One thread, M1 Pro: 1024 `complex64` points 2.14 →
  1.56 µs, 4096 `complex128` 15.2 → 13.8 µs; ten AVX-512 cores: 1024 `complex64` 2.38 → 1.44 µs, 4096
  `complex128` 18.9 → 14.7 µs (FFTW 10.4).
- **`rfft` of an even length checks its input's finiteness in one vectorised pass** (`detail::all_finite`, now for
  `float` as for `double`, moved to `simd.hpp`) and copies it as one block, where each pair of values was checked
  and copied alone. The same bits. One thread, ten AVX-512 cores: 4096 `float32` points 6.2 → 5.2 µs under Clang,
  7.7 → 5.7 µs under GCC; M1 Pro 5.75 → 4.94 µs.
- **Small complex products on x86-64 stay on the calling thread** (below 70³ multiply-adds of `complex128` and 84³ of
  `complex64` on the AVX-512 path, 52³ on the AVX2 path), where the threads cost more than they shared: ten AVX-512
  cores, all threads, order 48 `complex128` 19.9 → 12.1 µs, `complex64` 9.7 → 6.0 µs, order 64 `complex64` 16.0 →
  14.3 µs; six AVX2 cores, order 48 21.9 → 19.5 µs. The same bits (a product's bits do not depend on its threads).
- **`asarray` of a nested list or tuple of floats, ints and complex numbers reads it in one walk in C**
  (`_asarray_nested`), where Python gathered its values and shape first: M1 Pro, 1000 × 64 floats 53 → 1.4 ms (NumPy
  2.3 ms), 2 × 2 8.4 → 0.44 µs. The same values; anything else (another kind of number, a ragged list, an int past a
  double's range) is read as before, with the same errors.
- **`svd_decomposition`, `symmetric_eigen` and `eigen_decomposition` of an expiring matrix scale it in its own
  storage** (`matrix&&` constructors, as LU, Cholesky and QR had), where they copied it first: one matrix less at the
  peak of `eigh` and `eig` (from Python: `eigh` of order 3000 9.4 → 8.5 times a matrix's size, `eig` of order 2000
  12.1 → 11.3), and one pass less over the matrix; a wide `svd` takes its adjoint the same way. The peak of `svd`
  falls later, unchanged. The same bits.
- **The symmetric divide and conquer (`eigh`, `eigvalsh`'s vectors) solves blocks of up to 25 rows by QL**, LAPACK's
  `dstedc` size, where it took them up to 32: their QL's error, which grows with the block, weighed on the whole. On
  Wilkinson's W61 the residual `‖T V − V Λ‖` falls from 0.22 to 0.10 `n ε ‖T‖` (LAPACK's `dstevd` 0.098), on a
  random tridiagonal of order 60 from 0.30 to 0.19; ten AVX-512 cores, orders 40 to 1000, the same time. The last
  bits of `eigh` change.

## 2026.10.39 — 2026-10-27

### Added
- **The fast Fourier transform, as `numpy.fft`** (`fft.hpp`; `numerical.fft` in Python): `fft`, `ifft`, `rfft`,
  `irfft`, `hfft`, `ihfft` and `fftshift` along an axis, in C++; `numerical.fft`'s `fft`, `ifft`, `fft2`, `ifft2`,
  `fftn`, `ifftn`, `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`, `hfft`, `ihfft`, `fftfreq`, `rfftfreq`,
  `fftshift` and `ifftshift` with NumPy's points, axes, scalings (`norm`) and errors; `float32` and `complex64` in
  single precision. Any length: Stockham passes of radices 2, 3, 4, 5, 8 and 16, every other prime to 61 by its own
  sum, Bluestein's convolution past them; twiddle factors computed in double-double (each the correctly rounded root);
  a real transform of even length by the complex one of half its length. Within a fraction of ε·log₂ n of the exact
  transform (RMS). Two sets of bits only, each the same on every machine and compiler: the portable path's, its
  products rounded apart, and the vector paths' (NEON, AMX, AVX2, AVX-512), one fused complex product; a line has the
  same bits alone, in a batch and on any threads. Plans kept per length and type (the 32 most recent). Six AVX2 cores,
  one line: 1000 complex points in 4.3 µs, 1.2–2.0× FFTW's time for complex transforms, rfft 1.7–1.9× faster than fft.
  Lines of a batch transformed a register of them at a time on the vector paths, each line's bits its own: M1 Pro,
  `complex64` lines 2.1–2.9× faster than one at a time, 64 lines of 64 points 16 → 9 µs; six AVX2 cores, 1024 lines
  of 16 points 82 → 43 µs, along the first axis of 64 × 1024 346 → 222 µs. A line of 4 MiB and more in four steps
  that the threads share, its rows written down the output's columns as they are transformed (M1 Pro, `complex128`
  2²⁰ points 21.8 → 11.6 ms on one thread, 5.2 ms on all; NumPy 19.1; 2¹⁸ points 3.6 → 2.1 ms on one thread; ten
  AVX-512 cores, 10⁶ points 26.0 → 19.7 ms on one thread, 9.2 → 4.6 ms on all);
  AVX-512's registers on its path (ten AVX-512 cores, `complex128` 4096 points 14.0 → 9.3 µs, FFTW 10.0); NEON's on
  arm64 (M1 Pro, one line of `complex64` 2.0–2.4× faster, 2²⁰ points 17.9 → 8.3 ms; `complex128` 1.1–1.3×), its
  passes of radix 3, 5 and 8 too (`complex64` 1000 points 4.1 → 2.4 µs, 5⁶ 96 → 62 µs). A real input's `fft` and
  `ifft` by its real transform and the Hermitian symmetry (M1 Pro, 1000 points 5.0 → 3.0 µs, 65536 640 → 354 µs); a
  NumPy array of one of the four dtypes read where it lies, not copied first (1000 lines of 1024 `complex128`
  4.1 → 2.9 ms). Bluestein's convolution over a length of factors 2, 3 and 5 where the power of two would
  nearly double it (ten AVX-512 cores, 65537 points 7.3 → 4.3 ms, 16411 1.24 → 0.66 ms), and, for a length with a
  prime factor `p` past 61 at least 16 times `p`, over `p` points alone, in two steps (1009·64 points 2.8 → 1.5 ms,
  67·2¹⁴ 112 → 47 ms). The passes of the odd radices from 7 to 61 in the vector units' registers too (ten AVX-512
  cores, one thread: 44 100 points 1.18 → 0.62 ms, 30 030 1.05 → 0.62 ms, 61·256 0.62 → 0.21 ms, `complex64` 4.8×;
  M1 Pro 1.5–2.0×, `complex128` with radix 31 or 61 even). A transform's plan built with no lock held (a transform
  of another length no longer waits on it), its roots of unity computed on the threads (M1 Pro, 2²² points 61 → 27 ms,
  10⁶ 14.5 → 6.4 ms; the same roots); `fftshift` and `ifftshift` of several axes in one copy that the threads
  share (`numerical::fftshift(a, axes)`; M1 Pro, two axes of 2048² `complex128` 31.3 → 6.8 ms, 256² 308 → 34 µs).

- **`cholesky_decomposition::solve` and `qr_decomposition::solve` of a matrix** of right-hand sides, every column at
  once (C++): Cholesky's row by row over the columns, the threads sharing them, each column the vector solve's bits;
  QR's `Qᴴ B` through the matrix product kernel, then `R`'s back substitution over the columns.

### Changed

- **The eigenvectors of `eig` on the threads**: each vector's triangular solve, its normalisation and the final
  transpose (the same bits on any number of threads; M1 Pro, order 800: real 284 → 215 ms, complex 968 → 715 ms).
  `svd`'s `vh()`, the conjugate transpose of a wide matrix, by blocks on the threads; `svdvals` and `eigvalsh` find
  their scale and copy the matrix by bands on the threads (a wide one by blocks); a complex matrix's finiteness check
  on the threads, as a real one's. The same bits.

- **`svd`, `svdvals`, `eigh` and `eigvalsh` scale the matrix by a power of two** (its largest entry's), as `eig`
  already did, where they divided it by its largest entry: the scaling and its undoing are now exact (a diagonal
  matrix of 7, 14 and 25 gives exactly 7, 14 and 25, where 7/25·25 and 14/25·25 missed); their last bits change.
- **`dgeev`'s left eigenvectors from the same Schur form as the right ones** (`eigen_decomposition`'s new
  `left_vectors()`, as LAPACK's `dtrevc` side L: `S = J Tᴴ J` solved as the right vectors are), where a second
  decomposition of `Aᵀ` was matched to the eigenvalues by distance: each left vector now belongs to its eigenvalue
  by construction, a cluster or a multiple eigenvalue included, and `jobvl = 'V'` takes one decomposition, not two.
  Their last bits change.
- **A rank-deficient `lstsq` through the matrix product kernel**: its coefficients `Uᴴ B` and its solution `V·(Σ⁺Uᴴ B)`
  (ten AVX-512 cores, order 512, 64 right-hand sides: 212 → 84 ms); its last bits change.

- **`dgetrs` and `dpotrs` of eight right-hand sides and more by two triangular solves of blocks** (`dtrsm`'s, as
  LAPACK's), where each column was substituted alone (M1 Pro, order 512, 512 columns: `dgetrs` 36 → 6.3 ms, with
  `T` 143 → 7.8 ms, `dpotrs` 177 → 4.2 ms; ten AVX-512 cores 35 → 4.4, 118 → 6.1, 154 → 3.2 ms); fewer columns as
  before. Their last bits change.

- **Buffers from 1 MiB keep their blocks when freed** (four at most, each under 16 MiB on Linux, where huge pages take
  over, and under 32 MiB elsewhere): the next buffer one fits takes it back, its pages already in place, where the
  system allocator handed back fresh ones, each faulted in and zeroed at its first write (M1 Pro: FFT of 4096 lines
  of 64 points 380 → 182 µs, rfft of 2²⁰ points 3.5 → 2.3 ms, a sum of two vectors of 2²⁰ doubles 375 → 150 µs, the
  decompositions of order 512 5–10 % faster under Clang; Linux, glibc reusing blocks of one size already, 2–6 %).
  At most 128 MiB is kept on macOS.

### Fixed

- **The determinant of a singular matrix whose other pivots overflow** is 0, as documented, not NaN (∞·0): a finite
  product is kept as it was, signed zero included.
- **A complex LU with a subnormal pivot** (below the smallest normal number) divides by it, as LAPACK's `zgetf2` does,
  where its reciprocal overflowed to infinity and the factorisation and solves gave NaN; every other pivot keeps its
  bits.
- **The SVD of a matrix graded up** (its bidiagonal small at the top, large at the bottom, as a diagonal from 2⁻⁴¹⁶
  to 1 in steps of 2²⁶) converges: the bidiagonal QR iteration chases each block's bulge from its larger end, as
  LAPACK's `dbdsqr` does, where from the top alone it never converged and `svd_decomposition` (divide and conquer,
  whose small blocks it decomposes, and `bidiagonal_qr`) threw `convergence_error`. The last bits of a decomposition
  whose block is larger at its bottom change.
- **The singular values alone** (`singular_values`, `svdvals`, `svd(compute_uv=False)`, and those `lstsq` reports
  beside a QR solution) by the dqds algorithm on the bidiagonal (LAPACK's `dlasq1`, which NumPy's `dgesdd` takes for
  them), where the implicit-shift QR iteration found them to absolute accuracy only: a bidiagonal's small values to
  high relative accuracy, as NumPy gives them, where QR took one of 5·10⁻³³ of the largest for zero (the bidiagonal
  with diagonal 1, superdiagonal 2 and a last entry of 2⁻¹⁰⁰). Where a shifted transform's numbers are not finite
  (entries 10²⁹⁰ apart), it is retried unshifted and then guarded against overflow, as LAPACK means to: LAPACK's own
  test of that case never fires, and it returns NaN (so does NumPy on such bidiagonals). The
  last bits of every singular value change. QR in its place if dqds does not converge within 100 steps per value (not
  observed).

## 2026.10.38 — 2026-10-25

### Changed
- **A complex vector times a matrix (`x @ B`, `einsum("j,jk->k")`) has the bits of its row in a product of a matrix by
  B** past the portable path, as on it already for finite parts: its terms rounded as the path's complex product rounds
  them (the x86-64 complex tiles' panels, `Re x · B` and `Im x · B` from zero and then crossed; arm64's real kernel's
  blocks, `Re x · Re B` before `−Im x · Im B`), where each row of B was added by the compiler's complex arithmetic. B
  row-major or column-major (a Fortran-ordered NumPy matrix, read where it lies). Faster as well: M1 Pro, `complex128`
  500 × 500 93 → 47 µs, 4096 × 64 170 → 67 µs, `complex64` 2000 × 2000 1.07 → 0.75 ms; ten AVX-512 cores, `complex64`
  1000 × 1000 294 → 56 µs, `complex128` 500 × 500 158 → 66 µs; slower in two corners: under GCC 14 on arm64, `complex64`
  300 × 100 and 64 × 4096 by 4–8 % (B's rows read twice, its real parts' terms first), and on six AVX2 cores
  `complex128` 2000 × 2000 2.13 → 2.33 ms. Complex `x @ B`'s last bits move on the NEON, AMX, AVX2 and AVX-512 paths.
- **A real vector times a matrix has the bits of its row in a product of a matrix by it as well, and a complex one on the
  portable path where an operand is infinite**: each term rounded as `gemm` rounds it on the path (one fused
  multiply-add on the AVX, NEON and AMX paths, `float` included; on the portable path as its plain loop rounds `sum +
  a·b`, a complex product's infinities recovered by the library). `float` differed on every vector path, and under Clang
  the portable path's `double` and `float`, where a row's vectors were fused and its last elements not; a complex
  product of infinities read NaN. As fast or faster: M1 Pro, `float` 4096 × 64 37 → 26 µs under GCC 14, `double` 4096 ×
  64 50 → 40 µs. The last bits of a real `x @ B` move on those paths.

## 2026.10.37 — 2026-10-25

### Changed
- **A transposed complex NumPy matrix times a vector (`a.T @ x`) is read where it lies**, as a real one is, with its
  C-ordered copy's bits on every path (the slots its row kernel keeps, kept a column at a time over rows): M1 Pro,
  `complex64` 500 × 500 195 → 18 µs (NumPy 27), `complex128` 1000 × 500 815 → 71–94 µs (NumPy 121–153); the kernel
  from C++ on ten AVX-512 cores, `complex64` 500 × 500 18–20 µs.
- **A complex `einsum` product that sums nothing (`ij,j->ij`, `ijk,ik->ijk`) runs element by element**, as a real one
  does, where it went one matrix product of depth one per element of its batch: each element rounded as the path's
  complex product rounds it (the x86-64 complex tiles' parts from zero, arm64's real kernel's fused second term), so
  the same bits. M1 Pro, `complex64` 363 × 183 times a row 517 → 36 µs (NumPy 163–208), `complex128` 1000 × 1000
  9.8 → 0.33 ms (NumPy 2.5–3.5).
- **A sum of products rounds the same wherever its loop is compiled, under Clang on arm64** (the macOS wheels):
  Clang vectorised some such loops in order, their products rounded apart where the same loop elsewhere fused them,
  as its callers' inlining decided. A row of a `complex128` matrix times a vector now has the bits of that row times
  the vector alone, where 732 of 1680 shapes from 2 × 2 differed in their last bits. The wheels build with
  `-mllvm -force-ordered-reductions=false` where the compiler takes it; 300 drawn cases on an M1 Pro run as fast
  (1.002). Some results on macOS arm64 move in their last bits.
- **On arm64 a `double` row update fuses every multiply-add explicitly** (the NEON and AMX paths), its vectors' and its
  scalar tail's alike, as the AVX2 and AVX-512 kernels do, where the compiler's contraction decided: Clang fused the
  vectors and not the tail, GCC neither, so an LU's bits moved with its panel width and the macOS and Linux aarch64
  wheels' factors differed. Now they are the same at any panel width and under both compilers, and arm64 takes the LU
  panels of 128 columns from order 3072 too: order 4096 on eight cores 316 → 251 ms; under GCC 14 the LU, `inv`, `solve`
  and `qr` 5–26 % faster. arm64's last bits move.
- **A transposed NumPy matrix times a vector (`a.T @ x`) is read where it lies on arm64 too**, `float64` and
  `float32`, with its C-ordered copy's bits (the portable vectors' partial sums kept a vector of rows at a time),
  except where the row-major product takes the AMX: M1 Pro, 129 × 7 2.13 → 0.66 µs (NumPy 0.69), 1000 × 500
  `float64` 235 → 43 (NumPy 63), `float32` 207 × 119 12.6 → 2.9.
- **A transposed matrix times a vector takes its rows by blocks of 128**, where 256 left a matrix of a few hundred rows
  to two threads: six AVX2 cores, 500 × 500 `float64` 33 → 19.6 µs. The same bits.
- **`norm` sums its squares on the AVX2 units of x86-64** (the AVX-512 path's too), the same eight partial sums
  and the same bits: six AVX2 cores, 1000 × 1000 `float64` 90 → 57 µs (one thread 429 → 239), `float32` 73 → 30,
  101 × 901 `float32` 8.6 → 5.5.
- **A complex SVD reduces its matrix to bidiagonal form by panels** past order 128, as a real one did (`zgebrd`,
  `zlabrd`): the trailing block updated by complex matrix products after each panel of 32 reflections, where each
  reflection went over the whole of it; the bidiagonal is turned real by the same unit phases. Six AVX2 cores:
  `svdvals` of a complex 328 × 328 40.1 → 18.4 ms, 200 × 200 11.6 → 7.3. Real matrices keep their bits.
- **`float` dot products and matrix-vector products run on the AVX2 and AVX-512 units** of x86-64, as `double`'s
  do (four accumulators of 8 or 16, fused multiply-adds), where the baseline's SSE2 vectors ran: their last bits change
  on those units. A transposed `float32` NumPy matrix times a vector (`W.T @ x`) is read where it lies, as `float64`'s
  was, the bits of its C-ordered copy's product: six AVX2 cores, 1000 × 500 491 → 28 µs, 207 × 119 29 → 7.7.
- **A NumPy array converted into a Tensor is copied once, by groups of rows on the threads**: a real tensor's elements
  are no longer zeroed first, and the copy (with the first touch of its pages) is shared among the threads. M1 Pro,
  500 × 1000: `float64` 240–273 → 157–165 µs (NumPy's own copy 305), `float32` 102–123 → 82–92, a transposed
  `float32` 249–252 → 109–111; 48 × 275 `float64` 4.5 → 3.5. A complex one, zeroed by its own constructor, keeps its
  copy on one thread, which there is faster.

## 2026.10.36 — 2026-10-25

### Changed
- **`einsum` of two NumPy arrays reads both where they lie** when each label appears once and none is an operand's
  own (products, broadcasts, batches): `detail::einsum_pair`, the route a pair of Tensors takes, so the same bits.
  M1 Pro: `ij,j->ij` on 1000 × 1000 1127 → 333 µs (NumPy 864), `ij,i->ij` on 112 × 204 15.7 → 9.5, a 64 × 64
  product 9.0 → 4.1, 16 products of 32 × 32 34.5 → 18.9. A column-major matrix is still copied.
- **`einsum` that permutes a NumPy array's axes reads it where it lies** (`ij->ji`, `ijk->kij`), a column-major one
  with its labels reversed, where the array was copied into a Tensor first: M1 Pro, 1000 × 1000 1797 → 698 µs
  (NumPy's copy of `a.T` 798), 339 × 24 12.2 → 3.7 µs. NumPy answers with a view, which NumericAL does not make.
- **A dot product of three chunks or more shares them among the threads from 49 152 elements**, where one weight an
  element kept it on one thread to 131 072: the same chunks, so the same bits. M1 Pro, 10⁵ elements: 19.6 → 13.7 µs
  (NumPy 19.6); `norm`'s blocks likewise, 15.6 → 11.0 µs.
- **`einsum` of two operands sums each one's own labels first** (A-02), as three or more did: `ijk,jl->il` is
  `ijk->ij` then a matrix product, I·J·K + I·J·L terms where one loop over every label ran I·J·K·L. 64⁴ on an M1 Pro:
  21.6 ms → 63 µs. Each element's sum takes another order, so its last bits; fewer terms in each sum.
- **`norm` with no axis sums its squares by fixed blocks of 2¹⁴ parts on the threads**, each block's eight partial
  sums and tail as the one pass took them, the blocks' sums added in order: the same on any number of threads, and an
  array of one block keeps its bits; a larger one's change. A column-major array takes the same blocks in its C
  order. M1 Pro, 10⁶ elements: `float64` 158 → 40 µs (NumPy 160), `float32` 156 → 41 (NumPy 30); 3000 × 3000
  1444 → 645 µs.
- **A complex Cholesky divides by its diagonal as the real number it is**: each part over it, one rounding each, in
  the factor and in both triangular solves, where a complex division took the C library's scaled one (several
  roundings under libc++, other bits under another library). M1 Pro: order 48 27.5 → 16.6 µs, 128 247 → 162, 500
  3240 → 2750. Real Cholesky keeps its bits.
- **A product of one to four rows on x86-64 runs by blocks of columns on the threads** (`x @ W` with `x` a vector or a
  few rows): each block keeps its sums in registers over the whole depth with B read where it lies, where the packed
  kernel copied the whole of B for those few rows and a product by rows of four took one thread. The same bits. All
  threads: 1 × 1000 by 1000 × 512 on six AVX2 cores 173 → 33 µs, on ten AVX-512 cores 131 → 34; 4 × 300 by 300 × 300
  26 → 10 µs; one thread 1.3 to 3 times as fast.
- **`einsum("ij,ij->ij")` runs element by element**, as the batches of products that sum nothing do: the same bits. M1
  Pro, 1000 × 1000: 527 → 343 µs, the time of `a * b`.
- **`sum`, `mean`, `max` and `min` of every element of a NumPy array read it where it lies**, where it was copied into a
  Tensor first: the Tensor's answer to the bit. M1 Pro, 10^6 doubles: `sum` 791 → 52 µs, `mean` 811 → 44, `max` and
  `min` 700–796 → 237.
- **`exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `tanh` of a NumPy array, and `absolute` of a real one, read it where it
  lies**: the Tensor's result to the byte. M1 Pro, 10^6 doubles: `exp` 1581 → 644 µs, `log` 1889 → 1106, `sqrt` 1080 →
  309, `sin` 2009 → 1121.
- **`asarray` hands a NumPy array or a Tensor to the extension at once**, before the slower tests for numbers: every
  function converting its operands pays less. M1 Pro, `einsum("ij,j->ij")` of an 11 × 3 array and a row: 4.44 → 2.84
  µs.
- **A matrix times a vector on x86-64 shares its rows among more threads**: each element is read once, so the
  memory, not the arithmetic, sets the time, and the rows went to too few threads. The same bits (each row its own dot
  product). Six AVX2 cores: a `double` matrix of 1000 × 100 19.1 → 6.6 µs, 256 × 256 10.6 → 5.2, complex 671 × 36
  16.4 → 6.1, 539 × 68 14.3 → 9.8.
- **A transposed NumPy matrix times a vector (`a.T @ x`) runs without a copy on x86-64** (`double`): each row's sum
  the kernel's for its C-ordered copy, from the same partial sums updated a column at a time. The same bits. Six AVX2
  cores: 500 × 500 266 → 38 µs (NumPy 32), 1000 × 100 105 → 12.3 µs (NumPy 16.2).

### Deprecated
- **NumPy's loops over tensors move to `numerical::detail`**: `numpy_loops`, `numpy_wheel_loops`, `numpy_multiply`,
  `numpy_divide`, `numpy_power`, `numpy_square`, `numpy_reciprocal` and `numpy_absolute`. They follow the loops of the
  NumPy beside the binding, which changed between NumPy 2.2 and 2.3, and no part of the library's own algebra uses
  them: the operators and `pow` keep C's Annex G. Their names in `numerical` remain, each forwarding to its
  `detail` form under `[[deprecated]]`, until 2026.11.0 removes them.

### Fixed
- **A `long double` QR or Cholesky answers to `long double`'s precision**: the norms, reflector lengths and
  Cholesky's diagonal of a `long double` scalar, real or complex, are carried in `long double` (`detail::carried_t`),
  where `double` cut them, and with them `Q`, `R`, `L` and every solve built on them, to `double`'s precision. On
  x86-64 (80-bit `long double`): QR of order 12, `‖QR − A‖` 1419 ε → under 768 ε as `float` and `double` are held,
  `Q`'s orthogonality 5397 ε → likewise; of order 100, 2496 and 13 302 ε. LU already answered to it. `float` and
  `double` keep the same bits: the same type, the same code. Singular values stay carried in `double` whatever the
  scalar.

## 2026.10.35 — 2026-10-24

### Changed
- **A product of up to 24 columns runs by rows** (8 under Clang on arm64), where the packed kernel's tile of 16 or more
  columns left most of its width idle: the same bits, each term rounded as the tile rounds it. On 200 random products of
  1 to 24 columns, 0.42 to 0.64 times as long on x86-64 (ten AVX-512 cores, six AVX2 cores), 0.44 to 0.54 on arm64
  under GCC, 0.60 under Clang; 169 × 579 by 579 × 2 on one AVX-512 core 82 → 24 µs (OpenBLAS 13).
- **A small product runs by rows too, on the threads** (up to 2^20 multiply-adds from 4 rows, on x86-64 and under GCC
  on arm64; Clang's AMX tiles on arm64 keep it ahead already): the same bits. On 200 random products of 1 to 256 in
  each dimension, 0.78 times as long in all on six AVX2 cores and on arm64 under GCC; order 64 on six AVX2 cores
  14.5 → 7.2 µs, 96 24.5 → 12.7.
- **`max` and `min` of contiguous `double` and `float` run on AVX2 where the processor has it**, where the x86-64
  wheels' baseline build compared two `double` at a time: the same answer (each lane its own best, a NaN flagged apart,
  a zero's sign settled after). Six AVX2 cores, one thread: 4096 elements 0.36 → 0.09 ns each, 262 144 0.41 → 0.20.
- **A matrix product no longer zeroes its result on the calling thread first** (`a * b`, `matmul`, `@` and `einsum`'s
  products; `detail::c_start::zero` in C++): the blocks by rows start their sums at zero, and on the threads each
  packed task zeroes its own block of the result before its first panel, where the threads then took the result's
  cache lines from the caller's cache. The same bits. Six AVX2 cores, all threads: order 64 7.1 → 5.2 µs, 96 12.7 →
  9.4, 128 32.7 → 28.8, 256 152 → 143; one thread, order 24 to 96 5 to 9 % faster.
- **A product on threads runs by rows up to 2^19 multiply-adds a thread** where its blocks of rows give at least half
  the threads one each (x86-64, and arm64 under GCC): the packed product's pass of packing and its tasks scaled less
  well there. The same bits. Order 128 on six AVX2 cores 31 → 24 µs, on ten AVX-512 cores 32 → 21, on eight arm64 cores
  under GCC 72 → 30; of 150 drawn products of 2^18 to 2^23 multiply-adds, those it moves take 0.64, 0.58 and 0.38 times
  as long. One thread keeps the packed product.
- **`matmul` reads a NumPy matrix whose rows lie apart where it lies** (every other row of a larger array, each row's
  elements together), its rows' distance the kernels' leading dimension, where one of more than 65 536 elements went
  through `asarray`'s copy: the same bits, on every route (a matrix times a vector, a vector times a matrix, the
  product, beside a column-major operand). One M1 Pro: 169 × 579 by 579 × 2 67 → 31 µs, order 300 241 → 171 µs, the
  contiguous product's times.
- **A matrix times a vector on x86-64 runs its rows four at a time**: each row's blocks as its dot product sums them, then
  the four rows' remainders added in order together, their chains interleaved; the instruction set read once a product,
  where each row paid a call and a load of the path. Each row the bits of its own dot product. Six AVX2 cores, one
  thread: 925 × 3 8.2 → 1.7 µs, 152 × 59 2.7 → 1.9; ten AVX-512 cores: 925 × 3 13.3 → 2.5, 498 × 19 14.2 → 3.8; complex
  `double` 652 × 3 7.4 → 3.3.
- **A stack of matrix products shares its threads with its products**: fewer products than threads run one after
  another, each on all of them, where the stack's products in parallel held each to one (`matmul` and `einsum`'s
  batches); and a stack whose second operand is shared runs as one product of all the first's rows, that operand
  packed once. The same bits. Ten AVX-512 cores: two products of order 1000 28.0 → 7.5 ms, three of 600 6.5 → 2.8 ms;
  16 × 32 × 512 by 512 × 512 1013 → 509 µs, 1000 × 3 × 64 by 64 × 64 165 → 87 µs.
- **`sum` of many elements runs on the threads, to the bit**: the pairwise tree's subtrees of up to 2^15 elements summed
  in parallel, each as before, then added in the tree's order, the subtrees set by the count alone (the whole tensor's
  sum, and an axis's of single-element rows). M1 Pro: 2^20 doubles 181 → 45 µs, 2^24 2.97 → 1.17 ms; ten AVX-512 cores
  2^20 247 → 44 µs, 2^24 12.5 → 2.2 ms.
- **Complex `double` products by rows run on AVX2 on x86-64** (the panels' products of the complex LU and Cholesky
  factorisations), where they ran scalar: each term the library's product `(ac − bd, ad + bc)`, its products rounded
  apart, added to its sum, two columns a vector; a block with a NaN part among its products (the library rescues a
  product of infinities) is left to the scalar loop. The same bits. Six AVX2 cores, one thread: complex `cholesky` of
  order 286 2.66 → 1.87 ms, 512 10.6 → 7.9 ms; `det` of order 512 16.0 → 13.1 ms; ten AVX-512 cores, all threads:
  `cholesky` of order 512 4.1 → 3.4 ms, `det` 10.3 → 7.0 ms.
- **A transpose whose rows lie a multiple of 1 KiB apart (x86-64) or 8 KiB (arm64) copies its tiles by pieces of 8 ×
  8** (`einsum("ij->ji")`, `transpose` of a matrix, the BLAS interface's copies between column-major arguments and
  row-major storage): a tile's 32 rows that far apart fell in a few sets of the L1 cache and evicted one another. Four
  AVX2 cores, from Python: 256 × 256 doubles 163 → 52 µs, 1024 × 1024 2.65 → 2.45 ms; ten AVX-512 cores 256 × 256
  142 → 51 µs; M1 Pro 1024 × 1024 1.39 → 1.05 ms; other orders unchanged.
- **`norm` of a transposed NumPy matrix reads it where it lies** (a column-major array, as `a.T` hands it), eight rows
  at a time copied into a band in their C order, where the whole matrix was copied into a Tensor first: the same bits
  as its contiguous copy's. One M1 Pro: 252 × 175 26.5 → 17.3 µs, 371 × 208 47.1 → 30.4 µs, 1000 × 1000 1.23 → 0.58 ms.
- **A large buffer freed on Linux is kept for the next that fits** (two blocks at most, each under 64 MiB, the newest
  kept), where each buffer of 16 MiB or more took fresh huge pages, each faulted in and zeroed by the system at its
  first write: a sum of two vectors of 2^21 to 2^22 doubles 1.9–1.4 → 0.7–0.8 ns an element on six AVX2 cores,
  1.25–0.95 → 0.23–0.33 on ten AVX-512 cores (the time of a block glibc reuses below 32 MiB); a Cholesky factorization
  of order 1800 on ten AVX-512 cores 15.0 → 12.2 ms, products of order 1800 to 2400 1 to 14 % faster there; the
  decompositions on six AVX2 cores the same within noise. A program holds up to 128 MiB of such blocks after freeing them.
- **Broadcasting merges the axes both operands step through as one, and runs short rows repeated over a matrix as one
  loop**: `(1000, 4, 3) + (1, 4, 3)` runs as rows of 12; a row of under 32 elements repeated over a matrix's rows (`(n,
  3) + (3,)`) runs the matrix's rows as one loop against that row repeated, where each row of three turned the loop over
  the axes. Every element the same operation on the same operands. Six AVX2 cores, one thread: 10^6 rows of 3 plus a row
  7.1 → 3.0 ms (the time of a plain addition of as many elements); all threads: (1000, 4, 3) + (1, 4, 3) 23.7 → 5.6 µs,
  (4, 1000, 1000) + (1000, 1000) 6.2 → 4.3 ms. A column against a matrix of rows under 32 (`(n, 3) + (n, 1)`) spreads
  its elements over a run of rows, one loop against them: all six AVX2 cores, 10^6 rows of 2 4.1 → 1.9 ms, of 3 5.9 →
  4.4 ms; M1 Pro 1.09 → 0.40 and 1.83 → 1.25 ms.
- **An `einsum` batch of products that sum nothing runs element by element** (`ij,j->ij`, `ijk,ik->ijk`; an outer
  product, `i,j->ij`, stays one matrix product of depth one), each
  element its operands' product added to 0 as the matrix product rounds its first term, where it went one matrix
  product of depth one per element of the batch: the same bits, a zero's sign included (real types; complex ones keep
  their route). Six AVX2 cores: 1000 × 1000 times a row of 1000 19.7 → 0.55 ms (the time of `x * v`).
- **The thread pool runs no more threads than a CPU quota allows** (a container started with `--cpus=2`, a
  `CPUQuota=`, on Linux: cgroup v2's `cpu.max` of the process's group and those above it, or cgroup v1's
  `cpu.cfs_quota_us`), where a thread a core exhausted the quota and then waited it out. Six AVX2 cores under a quota
  of two processors: products of order 128 100 → 62 µs, 256 460 → 357 µs.
- **`a @ a.T` and `a.T @ a` of a float64 NumPy array compute one triangle and mirror it**: the matrix product's tiles
  on the lower triangle (as `syrk` does), each element the same products in the same order as its mirror, so the same
  bits as the whole product, with half its multiply-adds. 1000 × 1000: M1 Pro 4.33 → 2.89 ms (NumPy's 2.71), six AVX2
  cores 8.0 → 5.6 ms (NumPy's 5.8); 300 × 2000 on the M1 Pro 1.08 → 0.79 ms.
- **A vector times a narrow matrix holds each column's sum in registers** (fewer than 16 columns on arm64, 9 on
  x86-64), where it added the matrix's rows through memory: the same bits, each column's products added in the same
  order by the same arithmetic. One M1 Pro, from Python: a float64 vector of 764 times a 764 × 2 matrix 4.6 → 1.1 µs,
  4096 × 3 26.5 → 4.3 µs; a complex128 4096 × 3 28.7 → 8.0 µs (NumPy's 14.6).
- **A deep `double` product whose result lies within one AMX tile runs on the vector tile** (fewer than 32 rows and
  16 columns, on Apple's AMX): the AMX computed its whole tile at every step of the depth; both tiles sum each element
  by the same chain of fused multiply-adds, so the same bits. One M1 Pro: 8 × 1024 by 1024 × 8 24.3 → 7.9 µs,
  4 × 4096 by 4096 × 15 101.5 → 28.8 µs, a symmetric rank update of order 8, 4096 deep, 56 → 32 µs.
- **`qr(a, mode="r")` no longer forms `Q`**: `R` alone (`numerical::qr_r` in C++), the reduced decomposition's to the
  bit. One M1 Pro, order 1000: 34.8 → 20.6 ms (NumPy's `mode="r"` 19.6).
- **A stack of small matrices goes to the threads**: each matrix counts its copy and its answer's tensor beside its
  arithmetic, where a stack of 1000 matrices of order 4 stayed on one thread. `inv` and `det` of 256 to 1000 matrices
  of order 2 to 6 take 0.4 to 0.6 times as long; a stack of 16 or fewer the same. Each matrix's answer is the same.
- **The LU of order 3072 and more takes panels of 128 columns on x86-64**, where 64: each trailing update rereads the
  whole trailing block, and there its memory traffic sets the time. The same factors to the bit. Order 4096 on ten
  AVX-512 cores 322 → 279 ms, on six AVX2 cores 416 → 330; order 3072 10 and 12 % faster. arm64 keeps 64 columns.
- `eig` returns its eigenvectors without copying them, and `solve` with several right-hand sides solves them in their
  own storage, where both copied a matrix more.

## 2026.10.34 — 2026-10-23

### Fixed
- **`pow` of a complex tensor to a single 2 or −1 is the operators' square and reciprocal** (C's Annex G, as `*` and
  `/`), where the square took the school formula with each product rounded, neither the library's `*` nor NumPy's:
  `pow(1e308 + 1e308j, 2)` is now `z * z`; NumPy's loops stay `numpy_square`, `numpy_reciprocal` and `numpy_power`,
  which the binding takes. Its documentation said NumPy's.
- **`numpy.sum(t)`, `numpy.mean`, `numpy.max` and `numpy.min` of a Tensor answer**: NumPy calls the object's method
  with `out` (and `dtype` for a mean), which the Tensor methods refused with a `TypeError`; they now take `dtype` (the
  result converted) and an `out` of None (a Tensor is never written into).

### Added
- **`diagonal`, `matrix_transpose`, `tensorsolve` and `tensorinv`**, as `numpy.linalg`'s: the rest of its functions.
  **`sum`, `mean`, `max` and `min` take several axes and `keepdims`**, as NumPy's (`axis=(0, 2)`, `()` for none, a
  repeated axis a `ValueError`): the axes are moved last and reduced together, one pass, so a mean divides by their
  whole count and a sum adds by pairwise halving over them. **A `Tensor` indexes with `...` and `None`**, as NumPy's
  basic indexing: an `...` stands for the whole axes the other indices leave, a `None` adds an axis of extent 1.

### Changed
- **The element-wise functions share a large tensor among the threads** (`exp`, `log`, `sqrt`, `sin`, `cos`, `tan`,
  `tanh`, negation, `pow`'s single exponents, the `numpy_` forms), by the chunks the element-wise arithmetic takes,
  each element as before; from Python they release the GIL as the arithmetic does. One M1 Pro, a complex128 `tan` of
  4M elements: 41 → 10 ms. `map` itself stays sequential: its function may hold state. A complex function's sine and
  cosine are the system's `__sincos` on macOS, where GCC fused a `sin` and a `cos` into it in one inlining context and
  not in another: the same element took bits a unit apart.

### Development
- **`decomposition.hpp` gathers two headers**: `factorizations.hpp` (LU, Cholesky, QR, the symmetric and Hermitian
  eigen-decompositions) and `svd.hpp` (the singular value decomposition, least squares, the pseudo-inverse, the rank),
  the same declarations.

## 2026.10.33 — 2026-10-22

### Fixed
- **A complex number's `exp`, `log`, `sin`, `cos`, `tan` and `tanh` are computed as the GNU C library computes them**
  (`numerical/complex_math.hpp`), where they were `std::`'s, libc++'s on macOS and MSVC's on Windows, which lose a part
  glibc's keeps: `tan(1e308 + 0j)` was NaN for −0.509, `sin(1e308j)` had a NaN real part for 0, `log(1 + 1e-20j)` a
  real part of 0 for 5e-41. Annex G's infinities and NaNs, the parts scaled against overflow, the logarithm near the
  unit circle from an almost exact `x² + y² − 1`, the sums of products fused where glibc's arm64 build fuses them:
  NumPy's on Linux to the bit, x86-64 and arm64, glibc 2.28 (the manylinux wheels') to 2.41, both complex types, the
  sine and cosine of a part taken together by glibc's `sincos` as its own functions take them. `numerical::exp`,
  `log`, `sin`, `cos`, `tan` and `tanh` of a tensor take them. **A complex magnitude from Python is NumPy's** (`numpy_absolute`: the larger part times
  `sqrt(ratio² + 1)`, fused as its vector loop), where `std::abs` differed from it by two units in the last place.

### Development
- **The wheels are built and tested every Sunday without publishing** (the release workflow on a schedule; publishing
  takes a tag): the 32-bit Windows and musllinux wheels, which no push builds, met their first test at a tag.

## 2026.10.32 — 2026-10-22

2026.10.31 was tagged but never published: its musllinux and 32-bit Windows wheels failed two tests (a square root
compared with NumPy's to the bit outside glibc, an extent past a 32-bit `Py_ssize_t`). This release carries all of it.

### Development
- **The test suite builds under GCC 15**, which took a vector set element by element in the complex dot products and
  scaled adds for maybe used uninitialised: the swapped parts are built in an array and loaded as the operands are, the
  same code under clang and in GCC 14's loops.

## 2026.10.31 — 2026-10-22

### Fixed
- **`mean` of `float32` and `complex64` divides by an exact count**, in double precision and back, as NumPy's:
  the count was rounded to a `float`, so 2**24 + 1 ones averaged to 1 where NumPy gives 0.99999994.
- **`max` gives +0 and `min` −0 wherever both zeros are among the elements**, whatever their order — IEEE
  754-2019's maximum and minimum, and NumPy's on arm64 — where the first zero met kept its sign; a NaN still wins.
  NumPy on x86-64 keeps the sign of a zero as its vector loops meet them.
- **Complex `*` and `/` from Python give NumPy's bits, its infinities and NaNs included**: NumPy's own loops — the
  product's four part products and the quotient by Smith's algorithm, each fused as the NumPy beside them fuses it,
  measured once NumPy is imported (its wheel's build decides, not the processor alone: the scalar loop that takes one
  element from an operand of rank above 0 broadcast to a higher rank is fused by clang and recent MSVC builds, not by
  GCC's) — where `std::complex`'s operators recover infinities as C's Annex G does: `1j * (inf + inf j)` gave
  `-inf + inf j` where NumPy gives `nan + nan j`, and complex64 products differed in their last bit where NumPy fuses.
  The C++ operators keep Annex G; `numpy_multiply` and `numpy_divide` are NumPy's (`numpy_loops` says which fuse).
  **A complex `mean` divides by its count as NumPy divides**, by `n + 0j`: a NaN part makes both parts NaN.
- **`svd` of a 0 × 0 matrix gives empty factors**, as NumPy's, where it read the largest of no band of rows (an access
  at address 0 under AddressSanitizer), in C++ and from Python; `cond` of an empty matrix raises `LinAlgError`, as
  NumPy's, where a bare `ValueError` did.
- **A complex square root is computed as glibc's `csqrt`**: `sqrt`, and `** 0.5`, took `std::sqrt`, whose libc++
  form goes through the polar one — `sqrt(-1e20 + 1j)` in single precision had a real part of 755 for 5e-11, and
  `sqrt(-1 + 0j)` a real part of 6e-17 for 0. Now exact on the axes, Annex G's infinities and NaNs, scaled near
  overflow and among subnormal parts: NumPy's on Linux to the bit. **A complex `** 2`, `** -1` and `** n` from Python
  are NumPy's**: the square by its product's vector loop, `** -1` by its own reciprocal (`1 / (inf + inf j)` is
  NaN there, where `std::complex`'s division gave `0 − 0j`), an integer exponent below 100 by products and a
  quotient of 1 as `npy_cpow`, each fused as the NumPy beside them fuses it (`numpy_square`, `numpy_reciprocal`,
  `numpy_power`): `(1e308 + 1e308j) ** 2` was `nan + inf j` where NumPy gives `-inf + inf j`.
- **`einsum`'s general loop starts each product from its first factor**, where it multiplied a 1 by it: a complex 1
  times an infinity gave a NaN (0 · inf) and −0 came out +0, so a complex `trace` or sum of infinities or signed zeros
  went wrong. **`vector_norm` with `ord=0` counts a NaN** (it is not zero), as NumPy's, where it gave the NaN; with
  `ord=-inf` over no element it raises `ValueError`, as NumPy's smallest of nothing, where it gave inf. **The cross
  product of two 3-vectors is computed in their dtype**, where its components came from Python numbers, in double and
  by Python's complex product.
- **`outer` keeps each product's sign**, as NumPy's: computed through an einsum, a −0 product was added to a zero and
  came out +0. **A mean of an integer tensor over no element refuses** (`dimension_error`), where the division by
  a count of 0 was undefined (an integer has no NaN to give).
- **`cross` keeps each component apart**: the components were assembled by products with unit vectors, so an
  infinite one, or one overflowing from finite operands, turned the others NaN (`[1e308, 1e308, 0] × [1e308, -1e308, 0]`
  gave NaNs where NumPy gives `[0, 0, -inf]`).
- **A reduction keeping axes whose extents multiply past `std::size_t` refuses** (`dimension_error`), where a
  tensor holding no element along another axis — `(2**32, 0, 2**32)` — made `sum` and `mean` along that axis, an
  `einsum` such as `ijk->ik` and a norm over it allocate a buffer of the wrapped size and write past it. One keeping
  an empty axis is empty, whatever its other extents; a product or broadcast past `std::size_t` refuses before
  writing; `arange` refuses a length past `std::size_t`, or an infinite span, where the conversion was undefined.
- **Specs built in Python take the 52 labels einsum has**: `Tensor.T` past 26 axes, `tensordot` past 26 axes in
  all, and a stack's matrices labelled apart from its batch, where `matrix_power` of a stack of rank 25 refused and
  `trace` of one of rank 28 read a diagonal of its batch, silently wrong. Past 52 axes, a `ValueError` that says so.
  `cross` takes one integer axis, a `TypeError` otherwise, as NumPy's; a shape's extents are read as `Py_ssize_t`,
  where Windows' 32-bit `long` refused one past 2**31 − 1.

### Changed
- **Element-wise arithmetic shares a large tensor among the threads** (a same-shape pair by chunks, a broadcast one by
  rows, each task's odometer set from its first row), with the GIL released from Python, and **`transpose` copies by
  blocks of 32 among the threads**, where a row of the result walked a column of the source: each element computed or
  copied as before. One M1 Pro, 2000 × 2000 doubles: `a + b` 3.1 → 1.4 ms, `a + row` 2.8 → 1.2 ms, `transpose` 6.1
  → 1.7 ms. **BLAS's `cdotu`, `cdotc`, `zdotu` and `zdotc` of contiguous vectors run on vectors of their parts**, as
  the library's complex dot products, where they summed term by term: their last bits move. BLAS's `dgemv` works in `y`
  itself where it is contiguous, and reads `x` where it lies for `Aᵀ x` with α 1: the same values, without the copies.
- **`einsum` of a Fortran-ordered (transposed) NumPy matrix over one of its axes reads it where it lies** (`ij->i`,
  `ij->j`), its transpose row-major with the labels reversed: each sum the pairwise sum of the same elements, the bits
  of its C-ordered copy, where the matrix was copied first. 564 × 774 doubles by columns, one M1 Pro: 567 → 88 µs
  (NumPy 45).
- **The Hermitian reduction of complex `eigh` and `eigvalsh` runs on vectors**: its conjugate dot products over
  vectors of the parts (the dot product's four sums, combined as the conjugate's), its panel corrections spelled out
  as the complex product's statements, a reflector divided by its real norm part by part, and on arm64 each row of
  the Hermitian product read once for both of its uses (NEON, as AVX2's and AVX-512's rows). The last bits of complex
  `eigh`, `eigvalsh` and of the complex blocked QR move; they stay the same on any number of threads. One M1 Pro,
  complex128 `eigvalsh`: order 64 430 → 277 µs (NumPy 162), order 152 3.02 → 1.75 ms (1.54); complex64 order 321
  19.7 → 10.9 ms (14.7).

### Development
- **`make fuzz-edges`, a differential fuzz of the element-wise operations and reductions against NumPy** on the
  values and shapes that break them — zeros of either sign, infinities, NaNs, values near overflow, subnormal ones,
  empty axes, extents past a vector's or a task's width — to the bit where both compute the same expression, the NaNs
  and infinities exact elsewhere. Its fixed seeds run with the Python suite, a campaign in `check-release` and every
  week in `fuzz-long.yml`, with the shapes too large for a push (a float32 mean of 2^24 + 1 elements, sums keeping
  extents whose product overflows). Where NumPy's own answer depends on the processor or on its BLAS — a zero
  extreme's sign on x86-64, a NaN part of a complex dot product with an infinite term — it accepts either; NumPy on Windows
  halves the imaginary part of a complex square root it scaled against overflow (npymath rescales the real part
  alone), which it undoes before comparing. It found the complex arithmetic, mean, `einsum`, norm and cross fixes above.

## 2026.10.30 — 2026-10-21

### Changed
- **A complex `y += f · x` runs on AVX2 vectors on the x86-64 units** (a vector times a matrix, and the complex
  factorizations' updates), where the extension's baseline build ran SSE or scalar code: the same arithmetic, each
  part product rounded apart and no multiply-add fused, so the same bits. Six AVX2 cores (GCC 13), one pass of
  complex64 rows of 583 over 406 rows 182 → 69 µs; a complex64 vector of 406 times a 406 × 583 matrix 112 → 75 µs.
- **A small strided NumPy array is copied in C order where a product, a norm, an `einsum` reduction or a solve reads
  it** (up to 65 536 elements, aligned, in this machine's byte order), as the converting path copied it but without
  the Python calls around the copy: the same answers. A 4 × 31 strided matrix times a strided vector: one M1 Pro
  5.1 → 1.1 µs (NumPy 0.8); six AVX2 cores 6.7 → 1.2 µs (NumPy 1.1).
- **A vector, or a matrix of one row, times a Fortran-ordered (transposed) NumPy matrix reads it where it lies**, a
  column at a time, eight columns' sums at once, each element summed as a vector times a row-major matrix sums it
  (its terms in order, the same arithmetic: fused on the x86-64 units for float64, as there): the same bits, where the
  matrix was copied first. 1 × 406 by 406 × 583, one M1 Pro: complex64 382 → 66 µs (NumPy 55), float64 25.7 µs
  (NumPy 22); six AVX2 cores: complex64 331 → 102 µs, float64 23.7 µs.
- **A product of matrices in any dtype reads a Fortran-ordered (transposed) NumPy operand where it lies**, as
  float64's did: packed from there for the tiles, or copied into its parts from there for a complex product through
  the real kernel, the bits of the product of its C-ordered copy. One M1 Pro, B Fortran-ordered: complex64 3 × 803
  by 803 × 247 453 → 263 µs (C-ordered 221), complex128 712 → 341 µs (310), float32 300 × 300 × 300 172 → 109 µs.
- **Complex64 products on the x86-64 units run on complex tiles of their own** (AVX2 and AVX-512, as complex128's:
  each row accumulates `Re a · b` and `Im a · b` apart, by fused multiply-adds, and forms their sum once), where they
  went through the real kernel, copying B's parts for every product. The last bits of complex64 products on x86-64
  move; they stay the same on any number of threads and whatever the shape of the product. Six AVX2 cores (GCC 13):
  1 × 406 by 406 × 583 411 → 30 µs, 3 × 803 by 803 × 247 574 → 47, 16 × 512 by 512 × 512 389 → 128, order 128
  119 → 41, order 512 2.71 → 2.00 ms.
- **A complex product of a few rows of A runs four real products of each block's parts** (`Re C += Re A·Re B`, then
  `−Im A·Im B`; `Im C += Re A·Im B`, then `Im A·Re B`), where one real product of its expanded operands needed a copy
  of B four times its size, built for every product: the same chains, the same bits. On the AMX path up to 64 rows,
  elsewhere up to one row of the real tile, past which the expanded product stays. And a product whose A is one row of
  tiles reads B where it lies on the NEON and x86-64 tiles too, as the AMX's did for a block of rows, packing only a
  partial sliver (a group of slivers ending in one was packed whole). One M1 Pro, complex64 3 × 803 by 803 × 247
  376 → 133 µs, complex128 16 × 512 by 512 × 512 582 → 257 µs, float32 3 × 803 by 803 × 247 62 → 27 µs; six AVX2
  cores (GCC 13), complex64 572 → 380 µs, complex128 1 × 406 by 406 × 583 180 → 113 µs, float64 3 × 803 by 803 × 247
  114 → 90 µs; no other shape slower beyond noise.
- **`einsum` of one contiguous NumPy array summed over a run of its axes (`ij->i`, `ij->`, `ij->j`, `ijk->ik`) reads
  it where it lies**, where it was copied into a Tensor first: the same pairwise sums, the same bits. 344 × 961
  doubles by rows, one M1 Pro: 217 → 57 µs (NumPy 34).

## 2026.10.29 — 2026-10-21

### Changed
- **`einsum` sums one operand over a run of its axes (`ij->`, `ij->i`, `ij->j`, `ijk->ik`) by pairwise halving**, as
  `sum` adds, where its general loop added the terms one after another: the bits of `sum()` and `sum(axis)`. Order 256,
  one M1 Pro: `ij->` and `ij->i` 97 → 16 µs (NumPy's `einsum` 11.7, its `sum` 16.7), `ij->j` 56 → 16 µs (NumPy 22.7).
- **A complex `y += f · x` (a vector times a matrix, its rows added in order) spells out its part products**, each
  rounded apart as libc++'s operator rounds them, which Clang then vectorises, and GCC runs over vectors of the parts;
  every element the same arithmetic, the bits libc++'s. One M1 Pro, a complex64 vector of 406 times a 406 × 583
  matrix: 121 → 39 µs (NumPy 47); complex128 131 → 83 µs (NumPy 67).
- **A complex dot product sums over vectors of the parts**, its real and imaginary sums apart (`re·re`, `im·im`,
  `re·im`, `im·re`), where a running complex sum waited on each addition and tested each product for NaN; on the x86-64
  units with AVX2 their fused multiply-adds. A complex matrix times a vector shares its rows among the threads by
  their weight (a complex multiply-add four real ones), in groups of fewer rows where there are few. The last bits
  of complex dot products move. A complex128 matrix of 632 × 351 times a vector, one M1 Pro: 32 µs (NumPy 87); six
  AVX2 cores, each library timed in a process of its own: 25 µs (NumPy 16), and 189 × 206 12 µs (5.3).
- **Converting a NumPy array asks for its `__array_struct__` before its `__array_interface__`**, a capsule where the
  other is a dictionary built anew at every access, and `@` of two Tensors past the conversions calls the extension
  itself: a 4 × 31 strided matrix times a strided vector, six AVX2 cores, 11.7 → 6.7 µs.
- **A product of float64 matrices reads a Fortran-ordered (transposed) NumPy operand where it lies**, packed from there
  as gemm packs it, so with the bits of the C-ordered product, where it was copied first; and a Fortran-ordered matrix
  of 32 rows or more converts by tiles where it is copied (256 × 256 doubles 49.7 → 32.3 µs, NumPy's copy 43).
- **The 2-norm (`norm`, `vector_norm`, `matrix_norm`'s Frobenius) sums the squares in one pass**, in double, where
  their sum lies between 2^-900 and the largest double; outside (a NaN, an infinity, a sum past either bound) the
  scaled sum as before, so no overflow and no underflow still. Each element was copied, divided by the largest and
  indexed through every axis: a 371 × 208 float64 norm 361 → 13 µs (NumPy 13), complex128 100 × 100 4.1 µs (NumPy 7.7).
  The last bits move (a division fewer per element). A contiguous NumPy array is read where it lies, where it was
  copied into a Tensor first (six AVX2 cores, 371 × 208 doubles: 39.5 of the 56 µs).
- **A product whose A is one block of rows reads B in place whatever its stride** on the AMX, where a B more than
  512 doubles wide was packed for every call, a pass the product then made again; and a partial AMX tile takes the
  products of the accumulators its rows and columns fill, where it took all of them. The same bits. One M1 Pro, A of
  2 to 24 rows by a B of 700 × 800 doubles: 308 → 63 to 97 µs (NumPy 27 to 79).
- **A complex LU divides by each pivot once, as LAPACK's `zgetf2`**, its reciprocal multiplying the column and the
  solves' rows, where every element went through the C library's robust complex division (a quarter of a complex LU
  of order 48); the factors' and solutions' last bits move. Under libc++ the complex products of the LU, its solves
  and the small products run in line, the operator's own statements and its rescue of a product whose parts are both
  NaN copied (`detail::operator_times`, the same bits), where a translation unit as large as the Python extension left
  them a call (45 % of a complex LU of order 48). One M1 Pro, against NumPy (Accelerate), complex `solve` of order
  48 0.71× → 0.88×, `det` 0.75× → 0.97×, `inv` 0.95× → 1.29×; order 128 `solve` 0.66× → 0.81×, `inv` 1.58× → 1.86×.
- **`sum` and `mean` along an axis other than the last add whole rows**, each column's pairwise sum the same to the
  bit as when it was walked down alone: `sum(axis=0)` of order 256 55 → 14.5 µs (NumPy 16.2).

### Added
- **The Linux wheels carry `numerical_blas`** (experimental), the BLAS and LAPACK routines under their Fortran names —
  `?gemm_`, `?gesv_`, `?getrf_`, `?getrs_`, `?potrf_` and `?potrs_` in the four precisions, `ddot_`, `dsyrk_`,
  `dgemv_`, `dtrsm_`, `dsyevd_`, `dgesdd_`, `dgeqrf_`, `dorgqr_`, `dgelsd_` and `dgeev_` — and
  `numerical.blas_library()` gives its path, for `LD_PRELOAD` under a program linked against the system's BLAS and
  LAPACK (a NumPy or SciPy from the system's packages, R, Octave) or a switch such as FlexiBLAS; the import never loads
  it. It exports those routines alone (CMake's build of it too), so a process holding another build of NumericAL binds
  none of its code to the library's. The wheel grows by about 0.6 MB.
- **`numerical_blas` under the ILP64 names too**, its routines again with 64-bit integers as OpenBLAS's `64_` build
  names them (`dgemm_64_`), and with `sdot_`, `cdotu_`, `cdotc_`, `zdotu_` and `zdotc_` (a complex result returned by
  value, as gfortran returns it; `numerical::blas::complex_dot` and a `float` `dot` in C++): 78 routines, which Julia's
  libblastrampoline takes (`BLAS.lbt_forward(path; clear=false, suffix_hint="64_")`). Under it, twelve groups of
  Julia 1.11's LinearAlgebra tests (blas, matmul, dense, lapack, cholesky, lu, qr, svd, eigen, symmetriceigen, pinv,
  generic) pass but for two comparisons of `svd`'s vectors with `gesvd`'s, which are equal up to the signs of their
  columns, as LAPACK leaves them. The routines are declared once (`blas/numerical_blas.hpp`), each definition checked
  against its declaration: the 64-bit names of the complex and `float` routines had taken 32-bit integers.

### Fixed
- **`numerical_blas` takes a NaN or an infinity as the reference LAPACK (3.12) does**, where it reported the matrix as
  an illegal argument: `getrf` and `gesv` carry it through (`dgetf2`'s column by column factors, `getrs` after), as
  `geqrf` and `orgqr` do; `potrf` factorises column by column as `dpotf2` and reports the order it stops at (the
  reference's `INFO` on the same matrices, NaN, ∞ and −∞ on and off the diagonal); `syevd` gives NaN eigenvalues and
  `INFO` `n`. Julia's `lu` of a matrix of NaN, under `numerical_blas`, raised. `gesdd`, `geev` and `gelsd` still
  report it (`INFO` `-4`), as the reference's `gesdd` and `geev` report a NaN.
- **The sdist builds with setuptools 77 or later**, the first to read the SPDX `license` and `license-files` the
  project declares, where `pyproject.toml` asked for 68.

### Development
- **`make bench-fuzz`**: drawn cases against NumPy (an operation of seventeen, every dimension on a log scale up to
  `LARGEST`, the four dtypes, the operands contiguous, transposed, strided or already a Tensor), each library timed in
  a process of its own, twice in turn; every answer checked first. It reports the ratios' distribution by operation,
  size, dtype and layout, and the worst cases with their seed: where the fixed benchmarks, at hand-picked and
  power-of-two orders, put NumericAL at 1.45× NumPy on this Mac, 600 drawn cases put it at 0.94× (0.89× on six AVX2
  cores against OpenBLAS), and found the norm, the narrow products and the complex ones above; after them, 1.07× on
  both [1.00-1.14].
- `make bench-python` reports SciForge's verdict on the ratios, where it said "slower" whatever they were.
- The test binary runs the tests whose name holds its first argument (all without one).
- The AVX-512 workflow logs its host's processor and may take two hours: the emulator's speed is the host's (10
  minutes on one runner, 55 on another).
- Julia's LinearAlgebra tests run on `numerical_blas` (`julia.yml`: on a change to the BLAS interface, weekly, on
  demand), twelve groups under libblastrampoline, through `blas/julia_check.sh`: a failure fails the run unless the
  script lists it as a known divergence, and a listed one that stops failing fails it too.
- The AVX-512 emulator's run is a workflow of its own (`avx512.yml`): on a push that touches the x86-64 kernels,
  weekly, on demand, and required on the commit a release tags (`RELEASE_CI`), where every push waited on its sixteen
  minutes; the C++ suite runs once under the emulator, instrumented for `coverage-x86`, where it ran twice.

## 2026.10.28 — 2026-10-20

### Changed
- **The reductions behind `eigh`, `eigvalsh`, `svd` and `svdvals` share their steps' serial work with the threads**:
  each step of a panel computes its coefficients among the threads' matrix-vector products, then takes its
  corrections, and the next column's terms of the earlier steps, by bands of rows in one pass, where they ran on the
  caller between two products; every element the same operations in the same order, so the same bits. Six x86-64
  AVX2 cores: the tridiagonal reduction of order 1024 34.8 → 29.9 ms (`eigh` −5 to −8 % at 512–1024), the bidiagonal
  reduction 108.9 → 95.5 ms; an M1 Pro on eight cores, `eigh` of order 768–1024 −2 to −11 %. From 256 rows on
  x86-64, 512 on arm64 for the tridiagonal step; the bidiagonal step on x86-64 alone, where it gains.
- **`cholesky` reads a square float64 or complex128 NumPy array where it lies**, as `solve`, `det`, `slogdet` and `inv`
  do: its one copy is the one the factorisation takes, where the converted Tensor was a copy before it; the same
  factor and the same errors. From NumPy arrays, one M1 Pro: order 64 13 → 9 µs, 1024 4.42 → 3.15 ms, 2048 20.7 →
  16.5 ms (NumPy 11.3 and 72 ms).

### Development
- The linear-algebra fuzzer's `borrowed` section hands NumPy arrays of every layout (contiguous, Fortran order,
  strided, byte-swapped, misaligned, a block of rows of a larger array) to the products, `solve`, `det`, `slogdet`
  and `inv`, and requires the answer or the error the same values give in a Tensor, to the bit.

## 2026.10.27 — 2026-10-20

### Changed
- **A `float32` product runs on the AVX2 and AVX-512 tiles on x86-64**, where it ran on the portable path (the wheels'
  baseline SSE2, no fused multiply-add): tiles of 6 × 16 and 12 × 32 with each term fused in ascending order, and the
  plain loop of small and narrow `float32` products fused too, so that an element is the same whatever the shape of
  its product. Six x86-64 AVX2 cores: order 1024 17 → 4.0 ms, `matmul` of `float32` arrays from 3.7–4.9 to
  1.12–1.23 times OpenBLAS's time; one core 24 → 106 GFLOP/s. Their last bits change on those paths (fused now).

## 2026.10.26 — 2026-10-20

### Development
- `NOTICE` credits LAPACK, whose routines several algorithms follow closely, with its BSD license; the wheels and the
  sdist carry it beside `LICENSE`.
- `make bench-linalg` times `eig`, `eigvalsh`, `slogdet`, `svdvals`, `cond`, a `float32` product and a stack of inverses
  too, each answer checked before it is timed.
- **The API reference is published** at <https://reche23.github.io/NumericAL/> (`docs.yml`: Doxygen on every push to
  `main` that touches the headers or the pages it reads, every warning fatal under Ubuntu 24.04's Doxygen 1.9.8, whose
  stricter resolution of six references the headers now meet).

### Changed
- **`matmul` and `@` read NumPy arrays where they lie**, with no copy into a Tensor, when each is row-major, contiguous,
  aligned, in native byte order and of one of the four dtypes, both of one dtype, a vector or a matrix: the same
  kernels and bits as the product of two Tensors. M1 Pro, from NumPy arrays: a matrix of order 512 times a vector
  231 → 18 µs (NumPy 14), order 64 7.2 → 0.9 µs; a matrix product of order 1024 at NumPy's time.
- **`solve`, `det`, `slogdet` and `inv` read a square float64 or complex128 NumPy array where it lies**: its one copy is
  the one the factorisation takes in place, where the converted Tensor was a copy before it; the same answers and
  errors. From NumPy arrays of order 1024: `solve` 9.07 → 6.82 ms, `slogdet` 7.66 → 6.90 ms (NumPy 7.4), `inv`
  15.6 → 14.2 ms.
- `numerical.matmul(a, b)` goes through the same native method as `a @ b`, where it ran the Python fallback for
  every call: two tensors of order 8 2.3 → 0.5 µs.
- **The NaN and infinity check before a decomposition runs on vectors** under GCC and Clang (eight values a step), where
  GCC at the arm64 wheels' -O2 did not vectorise it at all: GCC 14 on one arm64 core, Cholesky of order 24 to 100 −11
  to −17 % (order 33 2.58 → 2.18 µs, below 2026.10.25's 2.40); x86-64 GCC 13 -O3, the check itself −40 %.
- **A `float` product runs on Apple's AMX** where the chip offers it, as a `double` one already did: tiles of 32 × 32
  in four 16 × 16 accumulators of fused multiply-adds, each element's terms in ascending order (the bits of the NEON
  tile, on one thread as on all). A narrow product with partial tiles stays on the NEON tile (`amx_suits_f32`); the
  AMX's self-test checks the `float` arithmetic too before it is used. M1 Pro: products of order 64 −80 % (12.3 → 2.5
  µs), 128 −89 % (59 → 6.3 µs), 512 −81 %, 1024 −77 % (6.0 → 1.39 ms).
- **A `double` product too narrow or too shallow to pay for packing runs in the plain loop** (which rounds as the tiles
  do: the same bits): on the NEON tile, a depth of 16 or less, or of 64 or less for 64 rows or fewer; on the AVX2 and
  AVX-512 tiles, 16 rows or fewer to a depth of 64, or 32 to 32, at least 64 columns wide. GCC 14 on one arm64 core
  (the NEON path the aarch64 wheels take): products of order 24 to 64 −32 to −50 %, Cholesky of order 48 to 512 −10
  to −31 %, QR −5 to −21 %, `eigh` −3 to −12 %, `inv` −3 to −11 %; x86-64 AVX2, a QR of order 96 −3 %.
- **The LU factorisation's panels update their right half through the product's plain loop**, which rounds as the
  tiles do (the same bits), where the packed product packed a sliver of the panel's whole height for every few
  columns. Six x86-64 AVX2 cores: order 300 607 → 577 µs, 1000 6.86 → 6.36 ms; one core, order 100 62 → 59 µs. One
  arm64 core: order 128 −5 % (Clang and GCC), 512 −4 % under GCC.
- **The Cholesky factorisation's rows below each panel solve faster**: their products by the earlier columns go through
  the product's plain loop, which rounds as the tiles do (the same bits), where the packed product packed slivers too
  small to pay for it; and on x86-64 the factor's rows and `−L₂₁ᴴ`'s columns are written one after the other, each
  along. One x86-64 AVX2 core, order 1000: that pass 2.82 → 1.97 ms, the factorisation 11.06 → 10.19 ms; six cores
  3.43 → 3.23 ms. arm64, one core: order 256 −5 % under GCC 14, −2 to −8 % under Clang from order 192.
- **The check for NaN and infinities before a decomposition reads a quarter faster**: an exponent of all ones is the
  one that one more carries into the sign bit, so the pass ANDs, adds and ORs, where it compared every value. One M1
  Pro core, a matrix of order 100 1.56 → 1.17 µs, 300 14.0 → 10.5 µs; x86-64 under GCC 13 % faster, under Clang 40 %.
  The BLAS ABI's `geqrf_` uses the same pass, where it kept a copy of its own.
- **On the AMX, a product whose rows are not a multiple of 32 ends in place**: its last tile of eight rows or fewer
  went through a copy the core wrote and the AMX read, a microsecond each. One M1 Pro core: products of order 33 3.7
  → 2.5 µs, 100 19.2 → 15.9 µs, 129 32.9 → 28.8 µs, 200 70.9 → 64.6 µs; an SVD and a QR of order 33 −16 %, order 8
  level. The same operations in the same order: the same bits.
- **`einsum`, `solve`, `det` (`determinant`), `slogdet` and `inv` (`inverse`) have their signatures**, positional as
  NumPy's: `einsum(subscripts, /, *operands)`, `solve(a, b, /)` and `(a, /)` for the others, in the stubs and at run
  time (`inspect.signature`), where both took `*args`. A type checker now refuses `solve(a)` or `einsum(a, a)`, and
  `stubtest` holds the stubs to the functions. Each call goes through one more Python frame, 22 ns on a 2 × 2
  `solve` of 830 ns.

## 2026.10.25 — 2026-10-19

### Fixed
- **A product's elements are the same whatever its shape, under GCC too, so a real inverse is the same to the bit on
  one thread as on all.** The plain loop of the small products rounded each term as the compiler does, and GCC under
  -ffp-contract=on (the wheels' flag) left its products unfused where the AVX and NEON tiles fuse every one: an LU
  inverse, which splits its columns by the threads, moved by up to an ulp on the Linux wheels (order 97 to 257, x86-64
  and aarch64). The plain loop now rounds as the tile of the path in use does: one fused multiply-add on the AVX2,
  AVX-512, NEON and AMX paths (on x86-64 in code built for AVX2 with FMA, so no call in a wheel built for any
  x86-64), and on the portable path as the vector tile rounds. One x86-64 AVX2 core, GCC 13: products of order 8 0.78
  → 0.67 µs, `qr` 3.93 → 3.52 µs, the rest level; one arm64 core: an order-4 product 10.6 → 11.5 ns, the rest level.
- **A complex inverse is the same to the bit on one thread as on all.** Past the portable path a complex product ran
  in the plain loop when small and through the real kernel when large, which round its terms differently, and the
  LU solve splits its right-hand sides' columns by the threads: at order 131 or 257 on arm64, `inv` of a complex
  matrix moved by up to an ulp of its largest entries between one thread and eight. Every complex product there now
  runs on the one kernel whatever its shape (the real kernel on arm64, the complex tiles on x86-64). Small complex
  decompositions got faster with it: one arm64 core, `qr` of order 8 6.5 → 4.8 µs, 16 32.4 → 23.3 µs, `eigh` of 16
  68 → 57 µs.

### Development
- A weekly long fuzzing campaign (`fuzz-long.yml`): an hour of the differential fuzzer on x86-64 and on arm64, built
  by GCC as the Linux wheels are, a fresh seed every hundred cases. `RELEASING.md` names the stable surface (C++
  `numerical`, Python `__all__`), what is tuning (the kernel paths), experimental (`numerical_blas`) or internal, and
  keeps the correctness record whose six quiet weeks make the library mature.
- `make tsan`, in `make check-push`: the suite under ThreadSanitizer (the pool's workers, nested parallel loops, the
  buffers each thread keeps), after a race planted in a parallel loop is reported, so that a runtime unable to watch
  the threads cannot pass for a clean run. The suite has no race under it.
- `SECURITY.md` (the threat model: values, shapes, the BLAS ABI, threads) and `CONTRIBUTING.md`.
- The differential fuzzer covers the rest of the NumPy surface and two contracts: `svdvals`, `cond` (below κ = 10⁶,
  its error growing as n·ε·κ²), `multi_dot`, `trace`, `outer`, `cross`, `tensordot` and `vecdot` against NumPy's
  within the bound of the sums they form; a NaN or an infinity in what a decomposition reads raising `LinAlgError`;
  and the same bits on one thread as on every one, factors and all. Each arm fails alone when its function is made
  to answer otherwise.

## 2026.10.24 — 2026-10-19

### Changed
- On the x86-64 units the Cholesky factorisation's trailing update runs on the triangle's tiles alone
  (`detail::gemm_triangle`), where its blocks computed the part of each diagonal block past the diagonal; each
  element's sum is the same, so the factor's bits. Six x86-64 AVX2 cores, `dpotrf_`: order 300 363 → 332 µs, 500 787 →
  721 µs, 1000 4.10 → 3.82 ms; one core, 500 2.04 → 1.89 ms. On arm64, where the AMX takes `−L₂₁` packed once for every
  block, the blocks stay (the triangle there: order 500 on every core 0.85 → 1.03 ms).
- Small steps cost less: the instruction set the next product takes is kept with the cap it was found under (one
  load), where each small update walked the paths down through their probes' guards; `dgetrf_` checks a square matrix
  for NaN and infinities once, in the decomposition, where it also checked it entry by entry across its columns. One
  x86-64 AVX2 core: `dgetrf_` order 100 88 → 79 µs, 300 0.99 → 0.93 ms; `dgesv_`, `dsyevd_`, `dgesdd_` of order 100
  2 to 4 % faster.
- The AVX2 tile unrolls its depth loop by four (it was by two) and fetches its slivers of A and B 512 bytes ahead, as
  OpenBLAS's Haswell kernel does: the same operations in the same order, so the same bits. One x86-64 AVX2 core,
  against OpenBLAS 0.3.26 and MKL 2024.2: `dgemm_` order 1000 41.8 → 38.0 ms (35.2, 34.9), 300 1.12 → 1.07 ms; `dsyrk_`
  1000 22.4 → 21.1 ms; six cores, `dgemm_` 1000 8.89 → 8.56 ms (8.39, 7.58).
- The LU, Cholesky and QR decompositions of a `const matrix&` copy it on the threads, its pages faulted in by
  several, where the matrix's copy faulted every one on the caller's: on every core of an M1 Pro, `cholesky` of order
  1000 3.58 → 3.22 ms, 2000 15.0 → 13.3 ms, `lu` 1000 5.42 → 5.14 ms, 2000 30.0 → 27.7 ms; six x86-64 AVX2 cores,
  `cholesky` 500 733 → 676 µs, 1000 3.55 → 3.29 ms. From 16 MiB on Linux, where a buffer takes 2 MiB pages, the
  threads' zeroing and copies take a page each: by eight 4 KiB pages a thread they waited on one another's fault
  (six x86-64 cores, 32 MiB zeroed into fresh memory 3.6 → 2.5 ms, a product of order 2000 −2 %; the copy before a
  decomposition of order 2000 was 2 to 4 % slower than one thread's, now level with it).

## 2026.10.23 — 2026-10-19

### Changed
- The SVD scales and copies its input, and turns its bases' rows into columns, by bands of rows on the threads, the
  transposition by tiles (`detail::transpose_into`, which the QR's copies share), where each ran on one thread and the
  transposition read its source a row apart. Six x86-64 AVX2 cores, `svd` of order 1000: 218 → 199 ms (OpenBLAS
  `dgesdd` 188); the transpositions 9.5 → 2.1 ms on six cores, 2.4 on one.
- The bidiagonal reduction of the SVD (past order 128) gathers each panel's columns once, row by row, where each step
  read its column across rows `n` apart, and takes a step's terms of the panel's earlier ones a family at a time in one
  update through their rows (`detail::subtract_products`; the reflectors and what they move interleaved), in the same
  order. The bits are the same on x86-64 and under Clang on arm64; under GCC on arm64 those terms are now fused. Six
  x86-64 AVX2 cores, `svd`: order 300 12.7 → 12.2 ms, 1000 210 → 203 ms (OpenBLAS `dgesdd` 11.3 and 188).
- The thread pool rings each worker it wants for a run, asleep, on a bell of its own, where every run woke them all:
  on Linux the worker pinned to the caller's core, which sits the run out, was woken on that core every time, held
  the caller off it, and slept again — about 4 µs a run. Six x86-64 AVX2 cores: an empty run 5.3 → 1.1–2.6 µs;
  `eigvalsh` order 300 4.05 → 3.73 ms, `dgetrf_` 300 625 → 590 µs, `dpotrf_` 300 398 → 371 µs. macOS, whose workers
  are not pinned and sit nothing out, is unchanged.
- The Cholesky factorisation's trailing update reads `L₂₁` where the factor holds it and takes the sign on its right
  operand (`L₂₁ (−L₂₁ᴴ)`, the same products exactly), where the rows below each panel were also written negated into a
  copy; that operand's storage is kept from panel to panel. The factor is the same to the bit. One x86-64 AVX2 core,
  `dpotrf_`: order 100 48 → 45 µs, 300 608 → 587 µs, 1000 12.5 → 12.2 ms; six cores 1000 4.31 → 4.16 ms; arm64, one
  core, 100 35 → 31 µs, 300 331 → 310 µs, 1000 5.1–5.4 → 4.6–5.0 ms.
- The passes bound by the memory around a factorisation run on the threads: the check for NaN and infinities of a
  `double` matrix past two chunks of 32 768 elements (`detail::all_finite_shared`: LU, Cholesky, QR, SVD, the
  eigensolvers), the zeroing of a Cholesky factor's upper triangle, and `dpotrf_`'s copies of its triangle. Six x86-64
  AVX2 cores, `dpotrf_`: order 300 426 → 399 µs, 500 1.02 → 0.82 ms, 1000 5.02 → 4.29 ms; `dgetrf_` 1000 7.0 → 6.7
  ms; one core unchanged. arm64, every core, `dpotrf_` 1000 4.1–4.6 → 3.7–3.9 ms; one core order 300 +5 %, a placement
  of the code (at `-falign-functions=128` or 256, +1 %; no inlining lost).
- The Cholesky factorisation's diagonal blocks go four rows at a time over the columns before them, which depend on
  the rows above alone: four chains of sums where row by row each term waited on the last (`cholesky_diagonal_rows`).
  Each element takes the same terms in the same order: on x86-64, and for complex matrices, the factor is the same to
  the bit; on arm64 Clang fuses other terms of a real one's sums, which move by rounding. One x86-64 AVX2 core,
  `dpotrf_`: order 300 701 → 635 µs (the diagonal blocks 128 → 65), 500 2.28 → 2.16 ms, 1000 13.4 → 13.1 ms; six
  cores, 300 504 → 440 µs; arm64, one core, 300 354 → 318 µs, 500 1.00 → 0.95 ms. The rows' loops are unrolled
  (GCC kept their sums in memory) and a block narrower than 16 goes row by row: arm64 under GCC 14, order 17 0.95 →
  0.79 µs, 33 4.6 → 2.8 µs; under Clang 17 0.80 → 0.62 µs.
- The Hermitian tridiagonal reduction (`eigvalsh`, `eigh`, `dsyevd_`) gathers each panel's columns once, row by row,
  where each step read its column across rows `n` apart, and takes a step's terms of the panel's earlier ones in one
  update through their rows (`detail::subtract_products`, the vectors and their products interleaved), in the same
  order; its trailing update runs on the triangle's tiles alone (`detail::gemm_triangle`) where bands of rows computed
  what the reduction never reads. The bits are the same on x86-64 and under Clang on arm64; under GCC on arm64 (the
  aarch64 Linux wheels) the panel's terms are now fused, and a real reduction moves by rounding. `eigvalsh` against OpenBLAS 0.3.26 on six
  x86-64 AVX2 cores: one core, order 100 0.47 → 0.42 ms (0.37), 300 4.61 → 4.06 ms (3.48), 1000 88 → 77 ms (67); six,
  300 4.38 → 4.04 ms (3.01), 1000 53 → 48 ms (36.6). arm64, order 300 3.16 → 2.93 ms, 1000 59 → 56 ms.
- `dgesv_` with `B` the identity to the bit (NumPy's `inv`) solves through the LU's inverse of the rows in their
  pivoted order (`lu_decomposition::pivoted_inverse`), whose forward pass leaves out the identity's zeros: the same
  terms otherwise, so the same bits, a test holding it at orders 9 to 301. Its columns go to the interchanged places as
  they are copied out. The inverse (`lu_decomposition::inverse`, `numerical::inverse`) writes its identity and applies
  the interchanges in place, by bands of rows on the threads, where a fresh `n × n` was faulted in and filled on one
  thread; its column slices weigh alike (`n² + (n − c)²` a column) where the first did twice the last's work;
  `inverse(buffer)` takes the result's storage. `inverse` on six x86-64 AVX2 cores: order 300 970 → 430 µs, 1000 15.3
  → 9.1 ms. NumPy's `inv`, against OpenBLAS 0.3.26: order 100 201 → 173 µs (81), 300 1.30 → 1.12 ms (0.74), 1000 24.0
  → 20.1 ms (17.7); arm64 100 123 → 110 µs, 300 980 → 935 µs, 1000 17.3 → 12.3 ms.
- The BLAS routines' transposing copies of their column-major arguments run in parallel by bands of tiles past a few
  hundred thousand elements (`dgesv_`, `dgetrf_`, `dpotrf_`, `dsyrk_`'s and `dgemm_`'s copies, …): NumPy's `inv` on six
  x86-64 AVX2 cores, order 300 1.50 → 1.29 ms, 1000 24.4 → 24.0 ms; arm64 300 1.23 → 1.15 ms, 1000 19.3 → 18.3 ms.
- The Cholesky factorisation's rows below a diagonal block take each column's terms of its sub-block with the column
  held in registers (`detail::subtract_products`), where each term read and wrote it again; the same multiply-adds in
  the same order, so without fusion the factor is still the unblocked algorithm's to the bit. One x86-64 AVX2 core:
  order 300 696 → 659 µs, 500 2.26 → 2.12 ms; arm64 300 350 → 332 µs, 1000 4.67 → 4.36 ms. The bits are the same on
  x86-64; on arm64 a few elements of large orders move by rounding where the column's last elements were fused
  otherwise.
- `dsyrk_` takes its triangle in one pass (`detail::gemm_triangle`): both operands packed once a panel of depth, the
  tiles outside the triangle skipped, a tile across the diagonal run on a copy of its block, where products by halves
  packed them again for every block and ran the diagonal blocks one after another. Each element's sum is the product's;
  with `β = 0` (NumPy's `a @ a.T`) the bits are the same, with another `β` the diagonal blocks' elements move by
  rounding (`β C` now starts their sums, as it does the others'). Six x86-64 AVX2 cores, against OpenBLAS 0.3.26: order
  200 110 → 61 µs (63), 300 386 → 181 µs (179), 400 594 → 335 µs (350), 500 949 → 667 µs (671), 700 2.46 → 1.66 ms
  (1.66), 1000 6.4 → 5.2 ms (4.7), 1500 19.8 → 17.3 ms (15.7); arm64 order 300 320 → 185 µs, 1000 4.6 → 2.3 ms.
- The matrix product takes either operand given as its transpose (`detail::gemm_at`, `gemm_atbt`, `double`): A packed
  from the transpose's rows, as B already is (`gemm_bt`), the bits those of the product by copies. `dgemm_` with a
  transposed `B` and `α = 1` (NumPy's `a.T @ b`), whether or not `A` is transposed too, and `dsyrk_` with `N` and
  `α = 1` copy no operand any more: the same bits. Six x86-64 AVX2 cores, against OpenBLAS 0.3.26: `dgemm_` `N`, `T`
  order 100 34.5 → 23.3 µs (OpenBLAS 15.4), 300 376 → 291 µs (234), 1000 10.1 → 8.75 ms (8.16); `T`, `T` 100 45.0 →
  24.4 µs (15.2), 300 977 → 295 µs (234), 1000 17.3 → 8.55 ms (8.16).

- The matrix product takes its right operand given as its transpose (`detail::gemm_bt`, `double`): B is packed from
  the transpose's rows, the tiles and so the bits are the product's by a copy of it. `dsyrk_` with `T` and `α = 1`
  (NumPy's `a @ a.T`) reads both operands from `A` in place, and `dgemm_` with a transposed `A` (NumPy's `a @ b.T`)
  copies it no more: the same bits. Six x86-64 AVX2 cores, against OpenBLAS 0.3.26: `dsyrk_` order 100 33.7 → 27.2 µs
  (OpenBLAS 14.1), 300 421 → 386 µs (179), 1000 8.62 → 6.33 ms (4.71), 1500 24.8 → 20.0 ms (15.7); `dgemm_` `T`, `N`
  order 100 32.7 → 25.6 µs (27.7), 300 376 → 293 µs (231), 1000 10.2 → 8.64 ms (8.11).

### Fixed
- The SVD by one-sided Jacobi (`svd_method::jacobi`) converges on a matrix whose rows or columns are of scales far apart
  (past a ratio of about 1e150), where it threw `convergence_error`: divided by its largest magnitude, the small rows'
  squares underflowed, so a pair of columns was judged on sums of squares of 0 and never found orthogonal, and past
  1e150 the rotation's tangent overflowed to no rotation at all. Such a pair is judged on the columns over their norms
  (their cosine, and the rotation from the ratio of the norms), a tangent past `|ζ| = 1e150` is `1 / (2|ζ|)`, and a
  rotation below the smallest double is skipped. To a ratio of 1e250 the factors stay orthonormal to rounding, each row
  reconstructed to its own scale (1e-13); at 1e300 the small rows reach the subnormals and `U` is orthonormal to about
  1e-5. Other matrices take the same rotations, to the bit.

## 2026.10.22 — 2026-10-18

### Changed
- On the AVX2 units a matrix product's task covers at most 36 rows over a panel of depth up to 128, 72 deeper: its block
  of A then stays in the L1 cache beside its slivers of B. Six AVX2 cores, 936 × 936 by 64 (an LU's trailing update):
  one core 44.7 → 47.9 GFLOP/s, six 235 → 255; by 32, 11 and 14 % faster; by 192 and more, unchanged. The tiles, and so
  the bits, are the same.
- On the AVX2 units a matrix product prefetches the next row of tiles' block of C while a tile computes: within a row
  the tiles walk one page's lines, which the processor follows, the next row lands on other pages. Six AVX2 cores, order
  1000: one core 49.2 → 50.0 GFLOP/s, six 228 → 232; 936 × 936 by 32 to 256, 1 to 5 %.
- The Cholesky factorisation solves the rows below each diagonal block by sub-blocks of 16 columns, the terms of the
  sub-blocks before one through the matrix product kernel (that solve took 37 % of the time for a tenth of the work),
  and packs −L₂₁ once for the tiles of the trailing update where a block of its rows serves several (past 512 rows); the
  factor is the unblocked algorithm's to the bit still. One AVX2 core: order 300 0.76 → 0.67 ms, 1000 13.9 → 12.7 ms,
  2000 86 → 76 ms; six cores 3 to 5 % faster; arm64, order 512, 16 %.
- On the x86-64 units the QR factorisation runs on the matrix column-major, as LAPACK's `dgeqrf`: each panel is
  factorised in place (it was copied out and back), and the columns right of it take its block of reflectors through
  products whose rows are those columns, where the row-major form's first product had the block's 32; `Q` is formed the
  same way. On arm64 the row-major form stays, faster there (the AMX tile reads the projection's right operand in
  place). `dgeqrf_` and `dorgqr_` copy their matrix in and out of that layout in storage the thread keeps (they
  transposed it into fresh storage, column by column), test it for NaN on vectors, and `dorgqr_` gathers its blocks'
  `T` through the matrix product kernel and forms `Q` as `qr()` does, so that its `Q` of `dgeqrf_`'s factors is
  `qr()`'s to the bit. Against OpenBLAS 0.3.26 on six x86-64 AVX2 cores, `dgeqrf_` one core 1.89 → 1.27× its time at
  order 300, 1.54 → 1.21× at 1000, 1.32 → 1.13× at 2000, six cores 1.47 → 1.24×, 1.6 → 1.24×, 1.16 → 1.01×;
  `dorgqr_` one core 1.78 → 1.14×, 1.62 → 1.17×, 1.39 → 1.10×, six cores 1.64 → 1.04×, 1.83 → 1.06×, 1.26 → 0.93×.
  One arm64 core, order 512: `dgeqrf_` 3.8 → 3.3 ms, `dorgqr_` 4.4 → 2.4 ms. On x86-64, real and complex factors move
  by rounding (the products run in other kernels), as accurate as before; on arm64 they are the same, and only
  `dorgqr_`'s `Q` moves, to `qr()`'s.
- `dgemv_` shares its work out in tasks of 64 rows (or columns), weighed as the memory-bound product it is (a quarter
  of a compute-bound product's multiply-adds), so that an order of a few hundred runs on several cores; `A x` holds each
  task's rows of `y` in registers through every column (the multiply-adds of a column at a time, in the same order:
  the same bits, but for arm64 under GCC, where they are now fused). Six x86-64 AVX2 cores against OpenBLAS 0.3.26:
  `Aᵀ x` (NumPy's `a @ v`) order 300 17.2 → 7.5 µs (OpenBLAS 3.0), 1000 73 → 61 µs (54); `A x` order 100 2.1 → 1.0 µs
  (1.7), 300 24 → 5.5 µs (6.8), 1000 135 → 99 µs (64). arm64, every core: `Aᵀ x` order 300 10.4 → 5.1 µs, `A x`
  300 13.4 → 5.4 µs, 1000 85 → 36 µs.
- `dpotrf_` copies `A`'s triangle into storage the thread keeps (a fresh zeroed one was faulted in page by page on
  every call), the lower one by tiles, and the factor back the same way (a row read across columns `lda` apart missed
  on every element). One x86-64 AVX2 core: order 200 370 → 255 µs, 300 980 → 710 µs, 500 3.06 → 2.35 ms.

### Fixed
- A program built with GCC 14 under `-Werror` (`-Wall`, glibc's fortified `memcpy`) no longer fails on a false
  `-Warray-bounds` in the buffers' release of huge pages, where a small buffer of a size known at compile time was
  inlined (a QR of constant 70 × 40 complex columns): that release is out of line.

## 2026.10.21 — 2026-10-17

### Added
- **An optional BLAS/LAPACK library, `numerical_blas`** (CMake `-DNUMERICAL_BUILD_BLAS=ON`): `ddot_`, `dgemm_` (and
  `sgemm_`, `cgemm_`, `zgemm_`, the conjugate transposes conjugated), `dsyrk_`, `dgemv_`, `dtrsm_`, `dgesv_`, `dgetrf_`,
  `dgetrs_` (and their `s`, `c` and `z` routines, `C` solving with `Aᴴ`), `dpotrf_`, `dpotrs_` (and their `s`, `c` and
  `z` routines, a complex matrix Hermitian, its upper triangle `Uᴴ U`), `dsyevd_`, `dgesdd_`, `dgeqrf_`, `dorgqr_`,
  `dgelsd_` and `dgeev_` with the Fortran calling convention (32-bit integers, hidden lengths of character arguments),
  over NumericAL's kernels, so that a program linked against BLAS and LAPACK, or a switch such as FlexiBLAS, runs on
  them. Their column-major semantics are `numerical/blas.hpp`'s, as the reference routines define them: arguments
  checked in the reference's order (an illegal one reported as `XERBLA` reports it, and in `INFO`), `β = 0` not reading
  `C`, `dsyrk` writing one triangle, `dgemv`'s negative increments, every side, triangle and transposition of `dtrsm`,
  `dgesv`'s factors and interchanges as `dgetrf` leaves them (a rectangular matrix too), `dpotrf`'s `INFO` the failing
  minor's order, `dsyevd`'s and `dgesdd`'s workspace queries (`LWORK = -1`) and minimums, every `JOBZ` of `dgesdd`,
  `dgeqrf`'s reflectors in the form `dorgqr` and `dormqr` read (and `dorgqr` forming `Q` from any implementation's),
  `dgelsd`'s rank (its cutoff relative to LAPACK's precision, 2⁻⁵³), singular values and, below the solution, the
  residual NumPy's `lstsq` reads, `dgeev`'s packed conjugate pairs and left eigenvectors. A matrix holding a NaN or an
  infinity, which NumericAL's decompositions refuse, is reported as an illegal value of that argument. Preloaded under
  Ubuntu's NumPy 1.26, which calls the reference BLAS and LAPACK, on six x86-64 cores at order 300: `solve` 5.3 → 1.6
  ms, `cholesky` 3.2 → 1.4 ms, `a @ a` 14.4 → 0.3 ms, `a @ a.T` (NumPy's `dsyrk`) 12.7 → 0.5 ms, `a @ v` at order 1000
  1.02 → 0.08 ms, `v @ w` of 100 000 elements 102 → 25 µs, `eigh` 41.9 → 8.1 ms, `eigvalsh` 11.0 → 4.5 ms, `svd` 84.1 →
  15.7 ms, `qr` 27.5 → 4.0 ms, `lstsq` (300 × 200) 17.6 → 6.2 ms, `eig` 182.6 → 42.6 ms, SciPy's `solve_triangular`
  (1000 × 200) 98.9 → 3.9 ms, `a @ a` of order 500 in `float32` 64.9 → 1.9 ms, `complex64` 225 → 8.5 ms, `complex128`
  190 → 3.9 ms, `solve` and `inv` of order 500 in `float32` 25.0 → 4.6 and 75.9 → 6.7 ms, `complex64` 71.7 → 12.6 and
  223 → 24.4 ms, `complex128` 69.0 → 9.6 and 217 → 24.0 ms, `cholesky` of order 500 in `float32` 15.0 → 2.5 ms,
  `complex128` 36.7 → 8.8 ms, the same answers to rounding; at order 1000, `inv` 577 → 29 ms, `det` 182 → 11.8 ms, `a @
  a.T` 508 → 9.1 ms. The reference LAPACK, linked against the BLAS, then runs on these kernels too: NumPy's and SciPy's
  linear algebra test suites (`numpy.linalg`, `scipy.linalg`, 4007 tests) pass with the library preloaded, in CI.
- `lu_decomposition::factors()`, `pivots()` and `singular()`; `not_positive_definite_error::order()`, the order of the
  first leading minor found not positive definite.
- `lu_decomposition::solve(matrix&&)`, which solves in the right-hand sides' own storage (their rows permuted in place,
  no copy), and `factors() &&`, which moves the factors out of an expiring decomposition.

### Changed
- The LU factorisation packs −L₂₁ once per panel for the matrix product's tile and shares it with every block of the
  trailing update, which packed it again each (the tiles, and so the factors, are the same to the bit); every panel is
  factorised in one storage, allocated once. Six x86-64 cores (AVX2), order 1000: 7.75 → 6.94 ms; on arm64 the LU and
  solves of order 128 to 256 are 5 to 8 % faster.
- On the x86-64 units the LU takes panels of 32 columns up to order 384 (not 256), where the panel is its critical path
  (order 300: 0.70 → 0.64 ms), and an LU solve gives each thread 48 right-hand sides or more (not 64), so that 300 of
  them use six threads rather than four (0.62 → 0.54 ms); arm64 keeps both, which it measured slower.

### Fixed
- QR (and the SVD and least squares built on it) forms its Householder reflectors as LAPACK's `dlarfg` does, `w₀ = 1`
  and `τ`, the identity where nothing lies below the diagonal: a unit Householder vector rounded exact inputs, so the
  columns of an identity came out as `1 − 2⁻⁵²` and `R`'s signs differed from the reference's. QR of order 8 to 24 is
  4 to 7 % faster for it; no other order changes.

## 2026.10.20 — 2026-10-17

### Changed
- **Arithmetic between tensors of one dtype, or with a Python `float` or `int`, skips the promotion**: the operands
  went through NumPy's promotion rules and `asarray` each call. A 4 × 4 `a + b` 1.71 → 0.37 µs (NumPy 0.29),
  `a * 2.0` 1.67 → 0.71 (NumPy 0.42); every other operand takes the rules as before.
- **The Cholesky factorisation's trailing update runs as tiles shared in one parallel loop**, of 64 columns by 256
  rows, each one product on one thread, where a parallel product for each block of columns started and joined the
  threads some thirty times a panel; each element's terms in the order of one product, the same bits. All threads:
  six x86-64 AVX2 cores, order 512 −18 %, 2048 −17 %; eight arm64 cores, order 2048 −45 % (Clang) and −40 % (GCC 14),
  1024 −10 to −30 % (Clang) and −44 % (GCC 14); one thread 1 to 4 % slower, each tile packing its columns again.
- **The Cholesky factorisation writes its trailing update's operands where it computes them**: the rows below a panel,
  factorised in parallel chunks, now each write their share of `−L₂₁` and `L₂₁ᴴ`, which a copy on one thread made
  afterwards (15 to 20 % of the factorisation on six cores). Six x86-64 AVX2 cores: order 512 −12 %, 1024 −14 %,
  2048 −13 %; eight arm64 cores, GCC 14, −6 to −9 %; the same bits.
- **Buffers of 16 MiB or more ask Linux for huge pages** (`madvise(MADV_HUGEPAGE)`; most distributions leave
  transparent huge pages to the program), each at another offset past its 2 MiB boundary so that they do not fall on
  the same cache sets: fewer address translations for the decompositions' strided passes. Six x86-64 AVX2 cores, order
  2048: `inverse` −23 %, `solve` −19 %, `cholesky` −23 %, LU −18 %, a product −5 %; order 3072 −14 to −20 %; below
  16 MiB unchanged, where a fresh result's huge pages, each zeroed on one thread, made a product slower.
- **A `float64` matrix of 2^15 to 2^18 elements times a vector runs on the Apple AMX**, eight rows at a time, each
  row's elements on eight lanes of fused multiply-adds (`fma64` in vector mode), the lanes added in order; the AMX's
  self-test now checks that mode too, to the bit against `std::fma`, and a failure keeps the AMX off. One M1 Pro
  core, from Python: order 192 6.5 → 5.2 µs, 256 11.1 → 7.6, 384 17 → 13.9 (Accelerate: 4.0 at 256). Smaller
  matrices sit in the L1 cache, where the NEON rows are faster; larger ones stay on the NEON rows' threads. Their
  last bits change on the AMX path (the lanes' order).

## 2026.10.19 — 2026-10-17

### Changed
- **The workers poll 500 µs, not 100, before they sleep**: a factorisation's steps last about that long at order 256,
  so the workers one step had left out slept through to the next, whose caller paid about 10 µs to wake each. LU of
  order 192 to 512 on six x86-64 AVX2 cores 5 to 13 % faster (256: 560 → 500 µs), `solve` 256 8 %; on eight arm64
  cores LU 256 and 512 7 to 10 %.
- **One copy of the operand, its pages faulted in on threads, before a decomposition from Python**: the routines that
  read one triangle (`cholesky`, `eigh`, `eigvalsh`) copied the matrix twice, and every copy into fresh memory took its
  page faults on one thread (a 2 MiB copy: 0.85 ms on one x86-64 core in a container, a third of a Cholesky
  factorisation of order 512). Six x86-64 AVX2 cores, against NumPy's OpenBLAS: `cholesky` of order 512 1.86 ms (2.02;
  was 4.5), 1024 8.9 ms (10.9; was 15.2); on eight arm64 cores, 3 to 14 % less from order 512.
- **`float32` products on arm64 run on a NEON tile of their own**, each multiply-add fused by lane, on the NEON path
  and on the AMX path (whose coprocessor takes `float64` alone), where they ran the portable tile, which GCC fuses
  nothing on: the Linux arm64 wheels ran them at half speed. One arm64 thread, GCC 14: order 128 27 → 47 GFLOP/s, 512
  33 → 65 (Clang 67). Their last bits change on Linux arm64, now fused; not under the macOS wheels' Clang, which fused
  them already.

## 2026.10.18 — 2026-10-16

### Added
- **`a @ b` and `matmul` of stacks of matrices**, as NumPy's: the leading axes broadcast against each other (an extent
  of 1 repeated, read where it is, not copied), a vector first a row and second a column, mixed dtypes promoted; one
  product per matrix of the stack through the packed kernel, the products in parallel. `@` took ranks 1 and 2 alone.
  Eight arm64 cores, against NumPy: 1000 products of order 4 0.019 ms (0.045), of order 8 0.050 ms (0.092), of order
  16 0.14 ms (0.27); 64 of order 128 0.91 ms (1.32); 100 of order 64 by one matrix 0.22 ms (0.30); behind on 8 of
  order 512, 5.4 ms (4.5).
- **An ellipsis in `einsum`**, as NumPy reads it: `...` stands for an operand's axes its letters leave, aligned from the
  right across the operands and broadcast; with no `->`, the output is the broadcast axes then the letters that appear
  once. An output without `...` where an operand has such axes is refused, as NumPy refuses it.

### Changed
- **Small products by blocks of rows held in registers**: below 2^13 multiply-adds on arm64 and 2^11 on x86-64 (it
  was 512), a product runs the plain loop four rows of C by eight columns at a time, each sum in a register over the
  whole depth, where it ran the packed kernel or the plain loop of unknown width: the same sums in the same order. One
  arm64 thread against 10.17: a product of order 8 0.11 times as long (Clang; GCC 0.28), 17 0.30 (GCC 0.22); `qr` of
  order 8 and 17 0.42 to 0.73, `eigh` 0.78 to 0.83, `svd` 0.75 to 0.94; one x86-64 thread, order 8 0.73. Where a
  product moves from the AVX2, AVX-512 or AMX kernel, which fuses each multiply-add, to the plain loop, which rounds
  as the compiler does, its last bits can differ from 10.17's: on x86-64 from 512 to 2047 multiply-adds, on Linux
  arm64 from 512 to 8191; not under the macOS wheels' Clang, which fuses both.
- **A NEON path for `double` products on arm64** (`gemm_path()` answers `"neon"` where the AMX is not in use): the
  vector tile with each multiply-add fused, by A's element in its lane. GCC fuses no arithmetic on vector extensions,
  even under the wheels' `-ffp-contract=on`, so on the portable path its tile ran a product and a sum per term, its
  accumulators spilled: the Linux arm64 wheels ran every matrix product at half speed. One arm64 thread, GCC 14,
  against the portable path: a product of order 128 0.55 times as long, 512 0.52; `lu`, `solve`, `inv`, `cholesky`,
  `qr`, `eigh` and `svd` of order 512 0.64 to 0.73 times; a complex product of order 192 0.23 times, through the real
  kernel where it ran the scalar complex tile. Their last bits change there, now fused as on the AMX and AVX paths.
  Paths from the most portable: `portable`, `neon`, `avx2`, `avx512`, `amx` (the C++ enumerators after `portable`
  are one higher); a cap at `avx2` or `avx512` on arm64, or `NUMERICAL_AMX=0`, keeps the NEON tile.
- **The AMX runs only the products it suits**: 16 rows and 16 columns at least, or 2^16 multiply-adds. A product of a
  few rows or columns computed whole tiles of 32 × 16 on it, and ran up to 5.5 times slower than on the vector tile
  (8 × 166 by 166 × 8: 7.0 µs, 1.3 on the vector tile); it now runs on the NEON tile, which rounds as the AMX does.
- **Complex `double` products on x86-64 run on complex tiles of their own** (AVX2 3 × 4, AVX-512 6 × 8), where they went
  through four real products (`gemm_split`): each row of a tile accumulates `Re a · b` and `Im a · b` apart, combined
  once. Six AVX2 cores, against NumPy's OpenBLAS: a complex product of order 512 4.9 ms (3.7; was 7.6), 1024 37 ms
  (30; was 52); complex `eig` of order 512 422 ms (499; was 491), 1024 1.80 s (1.95; was 2.19).

## 2026.10.17 — 2026-10-16

### Added
- **`eig` and `eigvals` of a general square matrix**, real or complex, and of stacks, as `numpy.linalg`'s, by LAPACK's
  method (`dgeev`, `zgeev`): balancing by permutations and powers of two; Householder reduction to Hessenberg form, by
  panels whose update is a matrix product past order 128 (`dgehrd`); the Schur form by the multishift QR algorithm with
  aggressive early deflation (`dlaqr0`, `dlaqr3`, `dlaqr5`) from order 200 for a real matrix and 100 for a complex one,
  below them Francis's double shift (`dlahqr`) or Wilkinson's single shift (`zlahqr`), each where it measured the
  faster; the eigenvectors by back-substitution in the Schur form (its sums vectorised), carried back by a matrix
  product. A real matrix's
  complex eigenvalues come in conjugate pairs, their vectors conjugate to the bit; `eigvals` gives `eig`'s eigenvalues
  to the bit. Complex answers whatever the eigenvalues, as NumPy 2.5's (NumPy before 2.5 made them real where they all
  were); each vector of unit norm with its largest component real. As accurate as LAPACK's on the differential fuzzer
  (each pair's residual within the backward-error bound of the balanced matrix `D⁻¹ A D`, each of NumPy's
  well-conditioned eigenvalues near one of ours). Eight arm64 cores, against NumPy: `eig` of order 64 0.7 ms (1.2),
  128 3.8 ms (9.5), 512 117 ms (247), 1024 569 ms (985); `eigvals` of order 1024 433 ms (770); a complex `eig` of
  order 512 329 ms (694); a stack of 1000 of order 16 7.4 ms (67). Six x86-64 AVX2 cores, against NumPy's OpenBLAS:
  `eig` of order 512 174 ms (233), 1024 647 ms (832); a complex `eig` stays behind past order 256 (1024: 2.2 s,
  NumPy 1.9).
- **The rest of `numpy.linalg`, as NumPy 2 answers it, stacks included**: `norm`, `vector_norm` and `matrix_norm` on
  every order, axis and `keepdims`; `cond` (infinite for a singular matrix, as NumPy's); `svdvals`; `matrix_power`
  (negative powers through the inverse); `multi_dot`, multiplying the chain in its cheapest order; `trace`, `outer`,
  `cross`, `tensordot` and `vecdot`; `conj` (`conjugate`). The norms keep a float32 or complex64 input's precision.
- `svd(..., hermitian=True)`, from the eigendecomposition: the singular values are the eigenvalues' magnitudes in
  descending order, `Vh` the eigenvectors signed by the eigenvalues.
- A 2-norm is scaled by the largest magnitude, so it neither overflows nor underflows: `vector_norm([1e200, 1e200])` is 1.414e200, where NumPy's is infinite.

## 2026.10.16 — 2026-10-16

### Fixed
- **The same operands give the same bits from one call to the next, under GCC too**: the wheels and the sdist are
  built with `-ffp-contract=on` (Clang's default), where GCC's default, `fast`, fused multiply-adds across statements as
  its passes found them, so a loop's vectorised and scalar versions, the one or the other run as the operands'
  addresses fell, rounded an element differently. Seen as an `eigvalsh` (or any decomposition through its symmetric
  product) changing its last bits with what ran before, on AVX-512 under GCC -O3: the Linux x86-64 wheels since the
  AVX-512 kernels. Compiling the headers with GCC, pass `-ffp-contract=on` (GCC before 14 takes it as off).
- **The complete `qr`'s `Q`, and the full `svd`'s `U`, extend the reduced ones to the bit under any fusion of
  multiply-adds**: their first columns are formed as the reduced `Q`'s, the complement apart, where formed together
  the products' tiles differed and GCC 14 on aarch64 rounded some columns otherwise.

### Added
- **Stacks of matrices, `(..., m, n)`, in every decomposition and solver but `lstsq`** (2-D in NumPy too): `solve`,
  `inv`, `det`, `slogdet`, `cholesky`, `qr`, `eigh`, `eigvalsh`, `svd`, `pinv` and `matrix_rank`, each matrix
  answering as it does alone, to the bit, the answers stacked along the leading axes (`det` and `slogdet` give
  tensors; a stack's `matrix_rank`, integers nested as its leading axes). `solve` broadcasts the leading axes of `a`
  and `b` as NumPy 2 does, `b` a vector only when it is 1-D. Matrices of order 128 or less, or as many as threads, go
  to the threads one each (the GIL released once for the stack); larger ones one at a time, each with the threads.
  Eight arm64 cores, against NumPy: 10 000 inverses of order 4 2.0 ms (4.4), 2 000 `eigvalsh` of order 16 4.1 ms
  (17.3), 1 000 `svd` of order 32 25 ms (119), 16 inverses of order 400 12.6 ms (24.8).
- `det` and `slogdet` of a float32 or complex64 matrix are rounded to its precision, as its decompositions' answers.
- **float32 and complex64**, kept as they are, as NumPy keeps them: `asarray` keeps a float32, complex64 or complex128
  array's dtype (and takes any of the four as `dtype`), `Tensor`, `zeros`, `ones`, `full` and `eye` make them, and
  `__array_interface__` and DLPack export them without a copy. Arithmetic, functions, reductions, indexing, products
  and `einsum` compute in single precision, as NumPy's (`+ - * /` and `sqrt` to its bits). The decompositions compute
  in double and round their answers to the input's precision: as accurate as single precision holds, where NumPy's
  LAPACK computes in it; their default tolerances (`matrix_rank`, `pinv`, `lstsq`) are float32's epsilon, as NumPy's.
- **NumPy 2's type promotion**: complex where one operand is, single precision only where both are; a Python number
  keeps the tensor's precision (`float32 + 2.0` is float32, `float32 + 1j` complex64).

### Breaking
- **`asarray` keeps float32 and complex64** where it widened them to float64 and complex128: code that relied on a
  float64 result from a float32 array asks for it with `dtype="float64"`. Bool and integer arrays still read as
  float64.

## 2026.10.15 — 2026-10-16

### Breaking
- **`eigh(a, UPLO="L", *, method=...)`**: the second argument is NumPy's `UPLO`, and the method goes by keyword; a
  method's name in second place raises `TypeError` naming `method=`.
- **`svd(a)` returns NumPy's default, the full factors**: `svd(a, full_matrices=True, compute_uv=True)`, `U` `m × m`
  and `Vh` `n × n`, the thin ones extended by an orthonormal basis of the complement, where it returned the thin
  factors of `full_matrices=False`. Code that relied on `(U * S) @ Vh` being `a` passes `full_matrices=False`;
  `compute_uv=False` returns the singular values alone. The algorithm is now `method=`, by keyword: a string in
  second place raises `TypeError` naming it. C++: `svd_decomposition(a, method, full)`.

### Added
- **NumPy's keywords on the decompositions**: `eigh(a, UPLO)` and `eigvalsh(a, UPLO)` (`"U"` reads the upper
  triangle), `cholesky(a, upper=True)` (`U` with `a = Uᴴ U`, from the upper triangle), `lstsq(a, b, rcond)` (a negative
  `rcond` is machine epsilon, as LAPACK's), `pinv(a, rcond, hermitian, *, rtol)` and
  `matrix_rank(A, tol, hermitian, *, rtol)`; `hermitian=True` takes the eigendecomposition, cheaper, and reads the
  lower triangle alone. `matrix_rank` of a vector or a number is NumPy's (1 unless all zero). `det` and `inv` name
  `determinant` and `inverse`. C++: `rank_cutoff` (NumPy's default, `relative`, `absolute`) for `lstsq`,
  `least_squares`, `pinv` and `matrix_rank`, which take `hermitian` too. A negative or NaN tolerance raises, where
  NumPy returns infinities or a rank of 0.
- **`qr(a, mode)`** takes NumPy's `"reduced"` (the default), `"complete"` (`Q` `m × m`, `R` `m × n`, the reduced
  factors extended) and `"r"` (`R` alone); `"raw"` raises `ValueError`. C++: `qr_decomposition(a, qr_mode::complete)`.
- The differential fuzzer holds the full factors orthonormal and the thin ones their first columns, and the
  complete QR an extension of the reduced one, bit for bit.

### Fixed
- **C++: the singular value decomposition of a `float` or `std::complex<float>` matrix converges**: a zero in its
  bidiagonal took a NaN phase (the factor that rescues a tiny magnitude was a `double`'s, infinite in `float`), and
  one-sided Jacobi sought an orthogonality to `double`'s epsilon that `float` columns do not hold; both ran out of
  steps and threw. The Python binding, in `float64` and `complex128`, was not touched.
- **`cholesky` reads its lower triangle alone, NaN and infinities included**, as it reads the values: a NaN above the
  diagonal raised, where NumPy ignores it. One vectorised pass over the whole matrix still comes first, the triangle
  entry by entry only behind a NaN or an infinity; GCC 14 on arm64 pays about 20 ns at order 8 (0.205 → 0.224 µs,
  the one cell `make bench-diff` flags), Clang nothing.

### Changed
- **An LU solve of up to sixteen right-hand sides goes backward by groups of eight rows too**: a group's terms in the
  blocks past its own go together, in the order each row took them, its own block's after, row by row. The same bits.
  One thread, four right-hand sides on a factor built once: arm64, order 128 −36 %, 256 −51 %, 1024 −59 % (Clang;
  GCC −36 to −50 %); six AVX2 cores, order 128 −31 %, 512 −45 %, 1024 −46 %. `solve` with its factorisation: 2–8 %.
- **A complex matrix product keeps its real operands from one product to the next, on each thread**, where each
  product mapped them anew, page by page; past 2 MiB an operand is let go after its product. The same bits. Six AVX2
  cores, one thread: order 64 0.12 → 0.07 ms, 128 0.76 → 0.44, 256 4.6 → 3.1; all threads, order 128 0.52 → 0.17 ms;
  arm64, order 256 15–22 % faster, smaller orders unchanged.
- **`svd` keeps the buffers of its divide and conquer's joins from one call to the next, on each thread**, as `eigh`
  does, the sorted bases swapped with the old ones so that their storage serves the next join; past 2 MiB a buffer is
  let go after the call. The same bits. Six AVX2 cores, one thread and all: orders 64 to 1024 5–12 % faster.
- **An LU solve of up to sixteen right-hand sides goes forward by groups of eight rows**: their terms before the
  group are independent and go together, each value of the solution read once for all eight, where each row's
  multiply-adds waited on its last one. The same bits. One thread, four right-hand sides on a factor built once: six
  AVX2 cores, order 128 −22 %, 512 −33 %, 1024 −30 % (Clang; GCC alike); arm64, order 128 −21 %, and past 192, where
  the blocked passes took over, the rows by groups now beat them (order 256 −35 % under Clang, −55 % under GCC; order
  1024 on every thread −9 % and −38 %). `solve` with its factorisation: 2–4 %.
- **`eigh` keeps the buffers of its divide and conquer from one call to the next, on each thread**, where every call
  mapped them anew, page by page; a buffer past 2 MiB (an order past about 512) is let go after the call, so a thread
  keeps at most 8 MiB. The same bits. Six AVX2 cores, all threads: order 256 6.2 → 5.5 ms, 512 21.3 → 20.0 ms; one
  thread, order 512 38.8 → 36.0 ms; arm64 1–3 % (its pages cost less to map).

## 2026.10.14 — 2026-10-15

### Changed
- **`eigh` and `svd` share their divide and conquer among the threads**: the leaves of its tree and the joins of one
  level are independent, and go to the threads together, each block depth-first on one thread below the depth that
  gives every thread one, where one block after another had run alone and the leaves and the lower joins did not
  scale at all. On one thread the order is the recursion's. The same bits on any number of threads (a new test holds
  it). All threads, order 128 to 512: `eigh` up to 18 % faster (six AVX2 cores 256: 6.5 → 5.9 ms; ten AVX-512 cores 7.5
  → 6.5; eight arm64 cores 3.3 → 2.9), `svd` 5–16 % (six AVX2 cores 128: 3.29 → 2.90 ms; eight arm64 cores 512: 38.7
  → 34.7 ms).
- **On Linux, a worker joins a parallel run by its core's rank rather than by arrival**: a run of few threads takes the
  same cores from one run to the next, their caches still warm, where the first workers to wake took it, on other
  cores each time. Ten AVX-512 cores, an LU of order 1024 on two threads 21.5 → 14.6 ms (one thread took 18), on four
  18 → 11.4 ms; six AVX2 cores unchanged; the regions sized for a few threads gain at the default count too (LU of
  order 512 on all ten: 3.2 → 2.8 ms).
- **The LU's row swaps reach the columns left of each panel at the end, in parallel**, where each swap held up the
  panel the other threads wait for, and on x86-64 a panel of 256 rows or more is copied by tiles of rows. The same
  bits. Six AVX2 cores: order 384 1.33 → 1.25 ms, 768 5.02 → 4.65 ms; arm64 and one thread unchanged.

## 2026.10.13 — 2026-10-15

### Changed
- **C++: `vector`, `matrix` and `tensor` store their elements in `numerical::buffer<T>`** (`buffer.hpp`), a `std::vector`
  whose elements constructed without a value are left uninitialised, so that a result written in full need not be
  zeroed first, on one thread, before the threads that compute it write it (the next entries use it). `tensor::data()`
  and `matrix::elements()` return a `buffer<T>`, and the iterators are `buffer<T>::const_iterator`; code that used
  `auto` is unchanged. The constructors still take a `std::vector<T>` (copied in; a `buffer<T>` moves in), and a
  braced list of elements; a `buffer<T>` compares equal to a `std::vector<T>` holding the same elements. Nothing is
  computed differently: every result is the same to the bit.
- **An LU solve of many right-hand sides, and so the inverse, gives each thread its own columns** (slices of 64 or
  more), each through every block of rows, where every block joined the threads for products too small to share
  them. The same bits. Inverse, eight arm64 threads: order 256 698 → 503 µs, 512 4.26 → 2.97 ms, 768 9.75 → 6.16 ms
  (GCC 14 without the AMX, the Linux wheels': 256 2.23 → 1.44 ms); six AVX2 cores, up to order 384 where the threads'
  rereads of the factor still pay: 192 980 → 825 µs, 256 1.87 → 1.60 ms. `solve` with 128 right-hand sides, arm64,
  order 512: 643 → 475 µs.
- **A product's result is zeroed by the threads that will compute it** (`detail::parallel_zero`), where it was zeroed on
  one before the product started: a fresh result's pages are first touched across the threads. The C++ `matrix *
  matrix`, `A @ B` and `einsum`'s products; the element-wise operations, transposes, reductions, `arange` and
  `linspace`, which write every element, no longer zero first. M1 Pro, eight threads, `matrix * matrix`: order 512
  680 → 580 µs, 1024 4.46 → 4.13 ms; ten AVX-512 cores, 728 → 647 µs and 4.36 → 3.99 ms; one thread unchanged. The
  same bits.
- **The C++ `matrix * vector` goes the way `A @ v` goes** (`detail::matrix_times_vector`, shared by both): each element
  its row's dot product, four rows at a time, the rows on every core, where a scalar loop summed each row in order.
  M1 Pro: order 64 1.5 → 0.37 µs, 256 34.7 → 8.0, 1024 873 → 45. Its last bits can move, as `A @ v`'s did.
- **A matrix times a vector, a vector times a matrix and a transpose take paths of their own**, where `einsum` packed
  the whole matrix for one product per element, or wrote a transpose a row apart at every element. Each row of
  `A @ v` is its dot product with `v`, as `einsum("i,i->")` sums it (four rows at a time off the x86-64 units), so its
  last bits can move; `v @ A` adds `A`'s rows in order, as before; a transpose copies by tiles, and `.T` calls the
  extension directly. Against NumPy, from orders 64 to 1024: `A @ v` 3.7–7.8× → 0.96–2.0× (M1 Pro), 1.7–4.9× →
  0.48–1.14× (six AVX2 cores), 2.6–13.4× → 0.74–2.25× (ten AVX-512 cores); the transpose, against a contiguous
  copy of NumPy's view, 2.7–5.8× → 0.62–1.35×, 3.1–6.5× → 0.38–1.6×, 2.4–6.7× → 0.60–2.6×.
- **The NumPy comparison (`benchmarks/linalg_bench.py`) times `A @ v` and the transpose**, and takes `--threads N`
  for both libraries: on a host with hyperthreads or other users, one thread per core under `taskset`, where
  OpenBLAS's threads on every hyperthread had swung its times by up to 40×.
- **`make bench-diff` compiles both trees with every function on a 64-byte boundary**: where the linker put a kernel
  moved a product of order 24 by 8 % between 2026.10.11 and 2026.10.12, whose product code was byte for byte the
  same, and stopped the release (two batches of the same binaries confirm a placement as surely as a change).
  Aligned, that cell reads 0.99, and the decompositions' real gains still show.

## 2026.10.12 — 2026-10-14

### Added
- **`numerical.eigvalsh(a)` and `numerical::symmetric_eigenvalues(a)`**: the eigenvalues alone of a real symmetric or
  complex Hermitian matrix, ascending, as `numpy.linalg.eigvalsh` (the lower triangle read): the tridiagonal form
  without its basis, then the implicit QL iteration without rotations kept. M1 Pro, eight threads: 1.4–1.7× faster
  than `eigh`'s values (order 256 2.2 ms against 3.4). In the NumPy differential beside `eigh`.

### Changed
- **The secular equation of `eigh` and `svd` sums each side of its poles in a loop of its own**, where each term chose
  its sum by a branch: M1 Pro, one core, `eigh` 13–21 % faster from
  order 33 (128: 1.02 → 0.82 ms, 512: 21.1 → 17.6), `svd` 7–11 % (128: 1.87 → 1.68 ms); GCC 14 as fast as before.
  The terms round differently, so under either compiler the values and vectors can move in their last bits; the
  reconstruction and orthogonality errors are unchanged, and the NumPy differential passes (1 500 cases).
- **The bidiagonalisation behind `svd`, `singular_values` and `lstsq` holds each chunk of `Bᵀ v` through its terms**,
  where each row's term stored it back: singular values of order 512 on one x86-64 core 46 → 42 ms, `svd` on an M1
  Pro 2–6 % from order 128. The same additions in the same order: unchanged to the bit.
- **On x86-64, those two matrix-vector products are shared between threads from a quarter of the work they took**:
  six cores, singular values of order 256 8.1 → 7.3 ms, 512 35 → 30, `svd` 4–8 %. Each column and each row is still
  computed whole by one thread: unchanged to the bit. On arm64 the threshold stays (sharing them slowed order 256).
- **On x86-64, an LU's trailing update weighs its work as the AVX tiles run it**, where it counted the AMX tile's
  speed and so shared fewer threads: six cores, order 256 690 → 622 µs, 384 to 1024 4 % faster. The factors are
  unchanged to the bit.
- **The singular values alone record no rotation**: the iteration on the bidiagonal kept every rotation for bases it
  did not form. `lstsq` 20 % faster on x86-64 (six cores, order 256 17.6 → 13.7 ms, 512 71.6 → 58.3), 7–14 % on an
  M1 Pro; `singular_values` and `matrix_rank` likewise. The values are unchanged to the bit.

## 2026.10.11 — 2026-10-14

### Added
- **`make bench-diff`**, run first by `make release`: the last release against the working tree, with each compiler
  the host has, at small orders too; a decomposition slower by 5 % in two batches stops the release. The two
  regressions 2026.10.10 fixed (Cholesky of order 17 to 47, an x86-64 LU solve of order 8 to 64) each came from a
  change measured only at the orders it aimed at. On a Mac it times Apple clang and GCC 14; the x86-64 wheel's
  compiler is timed by running it on an x86-64 host.
- **`numerical::singular_values(a)`**: the singular values without the singular vectors, as
  `numpy.linalg.svd(a, compute_uv=False)` (neither basis is formed, nor rotated through the iteration).
- **`einsum` takes a spec without `->`**, as NumPy does: the output is every label that appears once, in ASCII order
  (`"ij,jk"` is `"ij,jk->ik"`, `"ii"` the trace, `"ba"` the transpose), in C++ and Python alike. It was refused.
- **`numerical::least_squares(a, b)`**: `lstsq`'s solution with the residual sums of squares, the rank and the
  singular values beside it (`least_squares_solution`).

### Changed
- **Breaking: `numerical.lstsq(a, b)` returns `(x, residuals, rank, s)`, as `numpy.linalg.lstsq` does**, where it
  returned `x` alone. Write `x = numerical.lstsq(a, b)[0]` where only the solution is wanted. The old form was a trap
  for code ported from NumPy: `x, residuals, rank, s = lstsq(a, b)` unpacked the solution itself when it held four
  values, and failed otherwise. The residuals are given, as NumPy gives them, for a tall matrix of full column rank;
  the singular values come from the QR the solution already took when it took one, so the four cost what NumPy's
  four cost or less (M1 Pro, all cores, 2n × n: 0.88–0.95× NumPy from n = 64 to 512). The C++ `lstsq` is unchanged.
- **`matrix_rank` computes the singular values alone**, without the vectors it discarded: order 512 52 → 31 ms
  (NumPy 31 ms), 256 9.6 → 6.1 ms, 128 1.9 → 1.1 ms (M1 Pro, all cores).
- **A product whose order is no multiple of 16, on the AMX, computes its edge tiles in C itself** where it copied each
  one out and back: the copy, written by the core and read by the coprocessor, waited on its lines' write-back.
  M1 Pro, one core: order 24 2.0 → 1.2 µs, 33 4.7 → 3.3, 48 4.2 → 2.8, 65 10.3 → 7.7, 100 21.5 → 17.3; a tile
  whose last column cuts its second half row keeps the copy, which measured faster there (31, 47, 63), and so does
  one of eight rows or fewer, whose rows the core has just written (a QR or an SVD of order 8).
- **The LU's pivot search goes by native vectors**, each lane keeping its first largest magnitude: the same pivot
  as the element-by-element search, so the same factors to the bit. One core, order 256: M1 Pro 341 → 306 µs
  (clang), 389 → 366 (GCC 14); order 128 83 → 74 µs (clang); x86-64 (GCC 13) 2–4 % from order 64.
- **An LU's trailing update cuts its columns by the threads its work pays for**, not the threads the machine has:
  each cut packs L₂₁ again, and a product too small to share ran one cut a thread one after another. Eight M1 Pro
  threads: order 256 329 → 311 µs, 512 1.72 → 1.64 ms, 768 3.99 → 3.79; six x86-64 threads: 256 −9 %. One thread
  is unchanged, and so are the factors.
- **`a @ b` of two tensors runs in the extension**, with no Python frame: 2 × 2 1.29 → 0.40 µs (NumPy 0.59),
  16 × 16 2.19 → 1.07, 64 × 64 4.76 → 3.64 (M1 Pro). Any other operand (a list, a NumPy array, a rank past 2)
  takes the conversions and refusals it took.
- **A solve with a few right-hand sides takes their columns together**: under eight, each column waited on its
  own previous term, one column after another. The solution is the same to the bit. M1 Pro, one core, four
  columns: the solve from the factors 2.0–2.3× faster from order 48 to 192 (clang and GCC 14), `solve` of order
  128 112 → 92 µs.
- **A Cholesky factorisation updates its trailing block by products at least 64 columns wide**, where each
  panel's 16 or 32 columns packed every row below them again: on the AMX the packing outweighed the products. M1 Pro,
  one core: order 64 13.7 → 12.7 µs, 96 31.5 → 28.8, 128 61 → 53; GCC 14 and x86-64 neutral to 9 % faster. Real
  factors are unchanged to the bit; a complex one can move in its last bits (its products round as the complex
  kernel or the plain loop does, by their shape).
- **The row combinations of the LU's solves hold 32 elements through their terms**, where 8 held four chains of
  multiply-adds each waiting on its last: the inverse 6–11 % faster from order 33 (M1 Pro, clang; order 128 151 →
  137 µs), LU and `solve` up to 6 % at 512. The same arithmetic in the same order: the results are unchanged to the
  bit. GCC takes its own vector path there and is unchanged.
- **A product on the AMX packs A's slivers two columns at a time**, each pair of a row read whole and its lanes
  swapped, where each element was read alone: the packing took a third of a product of order 128. M1 Pro, one
  core: order 32 1.05 → 0.92 µs, 64 3.14 → 2.76, 96 8.5 → 7.7, 128 18.4 → 16.6, 256 −4 %. The same elements in the
  same places: the products are unchanged to the bit.

## 2026.10.10 — 2026-10-14

2026.10.9 was tagged but never published: its 32-bit Windows wheel failed a test. This release carries all of it.

### Fixed
- **The test suite on a 32-bit interpreter**: a `linspace` count no index of that width holds is refused by Python
  itself, as an `OverflowError`, before NumericAL sees it; the test now accepts that refusal as well.

## 2026.10.9 — 2026-10-14

### Added
- **The package is typed** (PEP 561): `py.typed` and stubs for `numerical` and its extension ship in the wheel and
  the sdist, so a type checker reads `Tensor`'s properties and operators and each function's signature. `make
  python-stubtest` and the CI hold the stubs to the runtime (`mypy.stubtest`).

### Fixed
- **`asarray` of an object it cannot read raised a bare `numerical.error`**: it raises a `TypeError`, as NumPy
  does, which is still a `numerical.error`. The module no longer exports `os`.
- **`eigh` of a graded matrix returned NaN, or its divide and conquer did not converge** (2026.10.7 and 2026.10.8):
  a rotation's length was taken as the plain root of its squares at every magnitude, but a QL shift's ratio on a
  matrix spanning 1e-200 to 1 reaches 5e199, whose square overflows. Past `2^500`, as below `2^-500`, the length is
  `std::hypot`'s again. `[[1e-200, 1e-200], [1e-200, 1]]` gives `[1e-200, 1]`, as NumPy and 2026.10.6 do.
- **A product whose result is empty crashed** (2026.10.6 to 2026.10.8): `a @ b` with `b` of shape `(3, 0)`, and an
  `einsum` contraction with no output element, read past an empty operand. It is an empty tensor of its shape.
- **`linspace` with a count past what memory holds aborted the interpreter**, and so could negation and the
  element-wise functions on a refused allocation: each raises a Python exception.
- **`eigh` of a Hermitian matrix scaled by the imaginary parts of its diagonal**, which it otherwise ignores (as
  NumPy's does): an infinite one made every value NaN, a huge one wrong. The scale is the real part's.
- **GCC 15 refused to build the tests** (`-Werror=array-bounds` on `core_of`, a subscript it read before its check).
- **The row-by-row LU solve's equality to the blocked solve, to the bit, was stated too broadly**: it holds where
  the blocked passes' products run the matrix kernel; a product small enough for the plain loop rounds each term
  apart on x86-64, where the rows fuse it, so a column's last bits may differ with the number of right-hand sides.
- **Cholesky of a small order took up to 85 % longer than in 2026.10.6** (2026.10.7 and 2026.10.8): its panels of
  16 columns, which win from order 48 on arm64 and 40 on x86-64, brought below that a trailing product whose
  packing the few panels do not repay. Below those orders the factorisation is one panel again. One thread,
  order 17: 1.23 → 0.80 µs on an M1 Pro (Clang), 2.45 → 1.31 µs on an i5-8500 (GCC 13 `-O3`); the factor is
  2026.10.6's, to the bit.
- **On x86-64, solving with a reused LU factor of a small order took up to 33 % longer than in 2026.10.6**: each
  row found its panel by an integer division, the panel width being chosen at run time since 2026.10.7. The width
  is a power of two, and a mask finds it. One thread, i5-8500, four right-hand sides: order 8 0.34 → 0.25 µs,
  17 0.80 → 0.62 µs, 64 5.7 → 5.1 µs (2026.10.6: 0.25, 0.62, 5.0).

### Changed
- **`pinv` forms `V Σ⁺ Uᴴ` through the matrix product kernel**, where it summed it term by term: M1 Pro, all cores,
  order 512 279 → 48 ms (NumPy 52 ms), 256 28 → 9.3 ms (9.2), 128 3.4 → 2.0 ms (1.5).
- **`array_equal` compares the elements natively**, where it compared Python lists: order 1000, 269 → 0.34 ms
  (NumPy 1.1 ms).
- **Every decomposition checks a `double` or complex input for infinities and NaNs in one vectorised pass** over
  the exponent bits, where it tested each entry and stopped at the first: a tenth of an LU of order 128. M1 Pro,
  orders 64 to 256: LU −3 to −5 %, Cholesky −5 to −9 %, `solve` −3 to −8 %.
- **On the AMX path, A's packed slivers are written past the core's L1** (non-temporal pair stores) once a block
  holds 2 048 elements: the AMX reads its operands from the L2, and the lines ordinary stores left dirty in the L1
  held each product's first tiles back while they were written back. M1 Pro, all cores, `matmul` of order 64,
  128 and 256: −22 %, −10 % and −11 % (3.98 → 3.1 µs at 64); a smaller block keeps ordinary stores, which a
  product as shallow as Cholesky's small panels wants (64 × 16 by 16 × 64: +17 % streamed). The products are the
  same to the bit.
- **The product packs a whole sliver of A without testing each element for padding**: Clang vectorises the copy
  then. M1 Pro, all cores, `matmul` of order 32 to 512: −5 to −8 % (order 64: 4.32 → 3.98 µs). x86-64 and GCC on
  arm64 move within ±3 %.

## 2026.10.8 — 2026-10-13

### Fixed
- **Built by GCC on arm64 (the Linux aarch64 wheels), 2026.10.7's LU and inverse ran up to 70 % longer than
  2026.10.6's**: the row combinations it moved them to (an LU solve's rows, the panel's rows of U) hold eight
  elements through their terms, which GCC at `-O2` left unvectorised. Under GCC they now go by blocks of native
  vectors, each element the same statement in the same order, so the same value. GCC 14, arm64, all cores,
  orders 64, 128 and 512: `inverse` 135 → 43 µs, 522 → 176 µs, 8.1 → 5.2 ms (2026.10.6: 78 µs, 305 µs,
  5.2 ms); the LU factorisation 35 → 23 µs, 157 → 92 µs, 2.9 → 2.1 ms; `solve` 196 → 130 µs at 128,
  3.4 → 2.5 ms at 512. Clang, which vectorised them, is unchanged.

### Changed
- **`asarray` reads a NumPy array through its `__array_struct__`, and a Python number directly**, where it read
  the `__array_interface__` dict NumPy builds on every access (twice) and passed a number through a nested
  list: what every operation with a number or an array of another library pays first. M1 Pro: `asarray` of a
  one-element array 2.0 → 0.61 µs, of `2.0` 2.1 → 0.23 µs; `x ** 2` and `x * 2.0` of one element 2.3 → 0.64
  µs and 2.5 → 0.66 µs. An object with an `__array_interface__` alone is read as before.
- **The x86-64 wheels are built at `-O3`**: GCC's unrolling and vectorising take 7–18 % off the decompositions
  of order 64 to 512 (six x86-64 cores, GCC 13, the baseline x86-64 build: `cholesky` −12 to −18 %, `qr`
  −9 to −14 %, `solve` −9 to −13 %, `eigh` −6 to −16 %, `inverse` −8 to −13 %). On arm64 it moved some
  operations up and some down, so `-O2` stays there.

## 2026.10.7 — 2026-10-13

### Changed
- **The AMX path reads B's rows in place**, where it packed them, when they are at most 512 elements long: its X
  registers load a row's 16 elements as they lie, and packing them took a fifth of a small product. Partial
  slivers are still packed, and the products are the same to the bit. M1 Pro, all cores: `matmul` of order 64,
  128 and 256 −23 %, −17 % and −17 % (5.5 → 4.3 µs, 26 → 22 µs, 130 → 104 µs), `qr` of order 256 and 512
  −11 % and −13 %; rows further apart stay packed (in place, order 1024 took 8 % longer and 2048 35 %). The
  x86-64 kernels keep packing: read in place, AVX2's product of order 256 took twice as long.
- **Off the x86-64 units, an LU solve with at most 16 right-hand sides goes row by row up to order 192**, as it does
  on x86-64 at any order, and the row combinations it is made of hold eight elements in registers through their
  terms, where each term was a call through the whole row. The solution is the blocked solve's to the bit. M1 Pro,
  four right-hand sides, the solve after the factorisation: 6.3 → 2.0 µs at order 32, 18 → 8.5 µs at 64,
  49 → 37 µs at 128; the factorisations gain with the row combinations (LU of order 64 and 128 −10 % and −8 %,
  `inverse` −17 % and −15 %). Past order 192 the blocked products stay (row by row was 10 % slower at 256).
- **Cholesky takes panels of 16 columns up to order 128 and of 32 up to 256**, where it took 64: there the
  panels, row by row on one thread, are most of the time. The widths stay powers of two, which the product
  kernels' tiles divide (24, 40 or 56 columns left the AMX's tiles part empty: order 320 took 20 % longer).
  The factor keeps its bits' rule. All cores, orders 64, 128, 192 and 256: six x86-64 cores (i5-8500)
  30 → 20 µs, 124 → 88 µs, 289 → 231 µs, 486 → 461 µs; M1 Pro 20 → 15 µs, 78 → 66 µs, 195 → 141 µs,
  298 → 278 µs.
- **Up to order 256, the LU takes panels of 32 columns** where it took 64: there the panels, factorised a column at
  a time on one thread, are most of the time, and narrower ones halve their work. The factors keep their
  bits' rule: each element receives its updates in ascending order of the pivot. All cores, `solve` with four
  right-hand sides at orders 128, 192 and 256: six x86-64 cores (i5-8500) 182 → 136 µs, 426 → 336 µs,
  802 → 743 µs; M1 Pro 181 → 150 µs, 722 → 611 µs at 256. `inverse` at orders 64 and 128: six x86-64 cores
  68 → 52 µs, 501 → 363 µs; M1 Pro 58 → 50 µs, 255 → 196 µs. From order 320 the panels of 64 stay (32 was
  7–13 % slower there on x86-64).
- **On the AMX path, a product under 2²⁴ multiply-adds runs on the calling thread alone**: a cluster of cores
  shares one AMX unit, so threads beside the caller pay only once the product spans clusters. M1 Pro, all cores,
  `matmul` of order 128 and 192: 36 → 26 µs and 84 → 70 µs; from order 256 the threads stay and pay (149 µs on
  one, 129 on them). The decompositions' panel products gain with it: `qr` of order 384 −17 %, `eigh` of order
  256 and 384 −10 %, `svd` of order 128 to 512 −4 to −6 %. Cholesky and `solve` are unchanged.
- **The QR and QL iterations of `svd` and `eigh` take a rotation's length as the plain root of its sum of
  squares**, as LAPACK's `dlartg` does, where `std::hypot` rescaled both values every time; it remains where the
  larger is below 2⁻⁵⁰⁰, so no square underflows (the matrices are scaled to their largest entry, so none
  overflows).
- **Below order 64, the SVD forms its bidiagonal bases a reflector at a time**, as LAPACK's `dorgbr` falls
  back to `dorg2r`, where blocks of 32 reflectors went through the matrix product kernel.
  With the item above, all cores, orders 24, 32, 48, 64 and 128: `svd` on six x86-64 cores (i5-8500)
  105 → 75 µs, 174 → 141 µs, 351 → 320 µs, 681 → 678 µs, 3.27 → 3.20 ms, and on M1 Pro 92 → 65 µs,
  129 → 99 µs, 276 → 234 µs, 446 → 416 µs, 2.00 → 1.94 ms; `eigh` on six x86-64 cores 56 → 47 µs,
  109 → 89 µs, 220 → 198 µs, 373 → 339 µs, 1.61 → 1.53 ms, and on M1 Pro 49 → 43 µs, 77 → 67 µs,
  178 → 168 µs, 279 → 260 µs, 0.98 → 0.94 ms.
- **Below order 128, the real SVD reduces to bidiagonal form one reflection at a time**, as LAPACK's `dgebrd`
  falls back to `dgebd2`: the blocked reduction's panels and trailing products cost more than they save on a
  small matrix. The bidiagonal reduction alone, orders 32, 64 and 96: six x86-64 cores (i5-8500) 116 → 63 µs,
  355 → 238 µs, 756 → 613 µs; M1 Pro 153 → 130 µs, 508 → 470 µs, 1127 → 1057 µs. From order 128 on the
  blocked reduction stays: one reflection at a time is 30 % slower at order 160 on arm64.
- **The SVD's divide and conquer multiplies each half of its bases apart**, as LAPACK's `dlasd3` does: the rows of
  a half take only the columns that reach them, half the operations of the whole product (the others are zeros
  there; the results are the same to the bit). Six x86-64 cores, order 1024: 314 → 306 ms.
- **The SVD's divide and conquer joins its halves on every core**: the Löwner weights by blocks of poles, the
  singular vectors by rows of the poles, their lengths by rows, and the sorted vectors row by row, where a
  single core read each of them a column at a time. The results are the same to the bit. All cores, order 1024:
  M1 Pro 268 → 240 ms, six x86-64 cores (i5-8500) 361 → 334 ms.
- **The SVD of a square matrix reduces it to bidiagonal form as it is**, as LAPACK's `dgesdd` does short of a tall
  matrix, where it first took a QR, formed its Q and multiplied it in at the end. All cores, orders 256, 512 and
  1024: M1 Pro 11.8 → 10.0 ms, 60 → 49 ms, 300 → 265 ms; six x86-64 cores (i5-8500) 19.1 → 15.8 ms, 79 → 71 ms,
  422 → 361 ms. A tall matrix keeps its QR first.
- **On x86-64, an LU solve with at most 16 right-hand sides goes row by row**, each row of the solution held in
  registers through its terms, where blocks of the factor were negated, packed and multiplied for a few columns;
  each element takes its terms in the blocked passes' order, so the solution is the same to the bit. Six x86-64
  cores, four right-hand sides: the solve after the factorisation 112 → 72 µs at order 256, 477 → 279 µs at 512.
  On arm64 the product kernel stays faster (row by row there was 2.5 times as long).
- **`eigh` copies less around its decomposition**: it copies the lower triangle alone (the reduction reads
  nothing above it), builds no identity it does not use, and lays the eigenvectors out as columns by
  ascending value in one copy by tiles on every core, where it transposed them element by element and then
  permuted every column, each write a row apart. The results are the same to the bit. All cores, order 512
  and 1024: six x86-64 cores (i5-8500) 26.5 → 23.2 ms and 121 → 102 ms; M1 Pro 22.2 → 19.5 ms and
  105 → 91 ms.

## 2026.10.6 — 2026-10-13

### Changed
- **`eigh`'s divide and conquer maps its buffers once and touches them on every core**: a page costs about a
  microsecond on Linux the first time it is written, and the joins waited on some 30 MB of them one thread at a
  time. All cores, `eigh` of order 1024: i9-9900K 123 → 115 ms, i9-10900X 126 → 112 ms; M1 Pro unchanged. The
  results are the same to the bit.
- **On Linux, the thread pool pins a worker to each physical core, and the caller's core sits each run out**:
  free to move, two of its threads at times shared one core's hyperthreads, and a product of order 1024 on eight
  AVX2 cores took 10.4 ms in most processes, 8.1 in the others; it now takes 8.1 ms in seven of eight (NumPy with
  OpenBLAS: 7.6). The caller's own thread is left where it runs; the worker that sits a run out sleeps rather than
  polls beside it (polling, it slowed `eigh` of order 1024 by a fifth); a worker the system refuses to pin stays free.
- **A Householder vector's norm is one vectorised pass** for `double` and `std::complex<double>`: the plain sum of
  squares where it can neither overflow nor underflow (between 2⁻⁹⁰⁰ and 2⁹⁰⁰), summed at the scale of the largest
  element only outside, where it took two passes and a division per element everywhere; nor is the column copied
  for its norm. The tridiagonal reduction's trailing update packs its factors into two buffers for the whole
  reduction, row by row. All cores, `eigh`: M1 Pro order 512 23.5 → 22.3 ms, 1024 110 → 104 ms; i9-9900K 256
  7.9 → 7.1 ms, 1024 129 → 121 ms; i9-10900X 1024 137 → 127 ms.
- **`eigh` applies its reduction's reflectors to the eigenvectors**, as LAPACK's `dormtr`, where it formed the
  reduction's unitary factor and multiplied by it: 2 n³ operations where there were 3⅓ n³, and two `n × n` buffers
  fewer. Divide and conquer works in one matrix, each half in its own diagonal block (as `dlaed0`), its joins'
  buffers kept from one to the next; the eigenvectors' Löwner weights go by blocks of poles, in place of the gaps
  to the roots. The results of divide and conquer are the same to the bit. All cores, `eigh`: M1 Pro order 512
  26 → 23.5 ms, 1024 126 → 112 ms; i9-9900K 512 32.2 → 28.4 ms, 1024 145 → 130 ms; i9-10900X 1024 155 → 134 ms.
- **`eigh`'s symmetric matrix-vector product takes four rows at once**: the vector and the partial product are read
  once for the four rows, where each row read them again (AVX-512, AVX2, and the compiler's vectors). One core,
  the product of order 1024: i9-9900K 110 → 84 µs (OpenBLAS's `dsymv`: 74), M1 Pro 137 → 94 µs. All cores, `eigh`:
  M1 Pro order 512 28.2 → 25.7 ms, 1024 141 → 125 ms; i9-9900K 512 33.5 → 32.2 ms, 1024 148 → 142 ms.
- **`eigh`'s reduction to tridiagonal form uses more threads on x86-64**: its symmetric matrix-vector product,
  paced there by each core's loads, is weighed four times its multiply-adds on AVX2 and AVX-512 (arm64 keeps its
  weight: there it gained nothing). All cores, `eigh` of order 512: i9-9900K 36 → 33 ms, i9-10900X 43.6 → 40.6 ms;
  256 and 1024 unchanged.
- **Cholesky solves the rows below each panel along columns**: chunks of 64 rows in parallel, each in a
  column-major copy where every column takes its terms for all the chunk's rows at once (on AVX-512 or AVX2 one
  fused multiply-add each), where each row's sums waited on their last addition. Each element's arithmetic is the
  unblocked step's, in its order. All cores, i9-9900K: order 256 0.70 → 0.55 ms, 512 3.2 → 2.15 ms, 1024 12.2 →
  10.1 ms, 2048 62 → 56 ms; M1 Pro: order 256 0.42 → 0.33 ms, 512 2.03 → 1.52 ms, 2048 54 → 49.5 ms.
- **Decompositions take over an expiring matrix and hand their factors over without copies**: `matrix::elements()
  &&`, `tensor(matrix&&)`, constructors of LU, Cholesky and QR from a `matrix&&`, and `lower()`, `q()`, `r()`,
  `vectors()`, `u()` on an expiring decomposition move their storage; `solve`, `inverse`, `determinant` and
  `cholesky` take their matrix by value. The Python package copies each operand once and each result not at all,
  where it copied each of them twice or thrice (each copy a fresh buffer of 8 MB at order 1024, its pages faulted
  in). i9-9900K, from Python at order 1024: the inverse 32.5 → 22.1 ms (NumPy with OpenBLAS: 35.6), `solve` 20.7 →
  13.6 ms, Cholesky 15.6 → 12.3 ms, QR 57.7 → 50.2 ms, `svd` 394 → 380 ms.
- **The AVX2 and AVX-512 tiles pay for more threads**: each kernel weighs its work by a speedup of its own, a
  quarter of the others' for AVX2 and AVX-512 (on physical cores, threads pay sooner; on the AMX, fewer threads
  stayed faster). i9-9900K: a product of order 128 73 → 39 µs, of order 192 107 → 90 µs; from Python at order
  1024, Cholesky 20 → 16 ms, `lstsq` 148 → 131 ms, `svd` 415 → 406 ms. Its packing is weighed the same way, so a small product packs in
  parallel too: order 128 39 → 30 µs (NumPy with OpenBLAS: 23), 192 90 → 62 µs (67). i9-10900X (AVX-512, ten
  cores): order 128 90 → 51 µs, 192 155 → 94 µs, 384 580 → 450 µs, 512 940 → 860 µs.
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
