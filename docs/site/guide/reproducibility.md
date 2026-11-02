# Reproducibility and threads

## What is guaranteed

- **The same bits on any number of threads.** Every sum runs in one written order, whether one thread or all of them
  compute it; the pool only schedules the pieces.
- **The same bits on every x86-64 processor with AVX2 or AVX-512**, for one build: the AVX2 and AVX-512 kernels sum
  each block of lanes by one written tree (41 functions checked byte for byte, from `matmul` and `solve` to `eigh`,
  `svd`, `eig`, `qz` and `logm`).
- **Not guaranteed:** the same bits between kernels (the portable one, NEON, AVX, the AMX coprocessor), between
  processor families, or between two compilers' builds; they differ in the last bits. From 2026.12.0, NumericAL is
  to be built with `-ffp-contract=off`, so that GCC's and Clang's builds agree too.

## Threads and kernels

The kernels share a pool of threads, started the first time a kernel's work pays for more than one: one per
physical core the process may run on, where the system tells them (Linux, macOS; elsewhere one per hardware
thread), since two threads on one core's hyperthreads share its multiply-add units.
`numerical.set_num_threads(n)` (C++: `numerical::set_max_threads`), or `NUMERICAL_NUM_THREADS` read at import,
caps the threads a kernel uses, the caller's included, as `OMP_NUM_THREADS` caps BLAS; at 1 no thread is
started. A process forked while the pool exists runs its kernels alone. Each library that embeds NumericAL's
headers has a pool and a cap of its own.

`float64` and `complex128` matrix products, and the decompositions through them, run on a kernel chosen at run time, which
`numerical.gemm_path()` (C++: `gemm_path_in_use`) reports:

- `"portable"` — the compiler's own vectors, on any processor;
- `"neon"` — on arm64, its vectors with each multiply-add fused, whatever the compiler would contract (`float32`
  products too, there and on the AMX path, whose coprocessor takes `float64` alone);
- `"avx2"`, `"avx512"` — on x86-64, where `cpuid` reports the instructions and the system saves their registers
  (the wheels are built for the baseline x86-64 and carry these tiles beside it);
- `"amx"` — on Apple M1 to M4, the matrix coprocessor Apple's Accelerate uses, which Apple does not document; a
  product of few rows or columns, which it would compute as whole tiles of 32 × 16, runs on the NEON tile.
  It is used only on a known chip family, outside a virtual machine, and once a product computed on it equals
  `std::fma` in order to the bit; `NUMERICAL_AMX=0`, read by the Python package at import, keeps it off (its
  self-test included). C++ reads no environment: `limit_gemm_path` below `amx` before the first product does the
  same, and compiling with `NUMERICAL_NO_AMX` removes it.

Each thread keeps the workspaces its kernels last needed, and freed blocks of a few megabytes are kept for the next
buffer that fits one, so a repeated call faults no fresh page in. `numerical.release_memory()` (C++:
`release_memory`) gives that memory back, as oneMKL's `mkl_free_buffers`: the calling thread's workspaces, each pool
worker's (emptied on its own thread between two runs), the kept blocks, then the allocator's free pages (a session of
order 1500 on one thread: 148 MB kept, 2 MB after it). `OMP_NUM_THREADS` caps the pool where `NUMERICAL_NUM_THREADS`
is unset, and threadpoolctl's `threadpool_limits` caps it as a BLAS library.

`numerical.limit_gemm_path(name)` (C++: `limit_gemm_path`), or `NUMERICAL_GEMM` at import, caps the kernel at
`name` or a more portable one. Every element of a product is summed in ascending order of the depth on any
number of threads, so a result is the same, to the bit, whatever the thread count. The AVX2 and AVX-512 kernels
also sum in one order (the dot products, the symmetric and Hermitian rows and the products' tiles alike, each block
of lanes by one written tree), so one binary, built by one compiler, gives the same bits on every x86-64 processor
with AVX2, whether it has AVX-512 or not, on any number of threads (41 functions checked byte for byte, from
`matmul` and `solve` to `eigh`, `svd`, `eig`, `qz` and `logm`); two compilers' builds, and the portable kernel, may
differ in the last bits.
Across other kernels a result differs in the last bits, since the NEON, AVX and AMX kernels fuse each multiply-add
and the portable one rounds as the compiler does (under the wheels' `-ffp-contract=on`, Clang fuses `c + a·b` there and GCC does not). Past the portable kernel, a complex product runs as one real product of twice the order, `[Re A | −Im A]`
times `[[Re B, Im B], [Im B, −Re B]]` (by blocks), on that kernel and its threads: each element then sums the
products of A's real parts, then those of its imaginary parts, where the portable kernel sums whole complex
products in order.

## Compiler settings

Compiled with GCC, the headers want `-ffp-contract=on` (Clang's default; the wheels and the sdist set it): GCC's own
default, `fast`, fuses multiply-adds across statements as its passes find them, so a loop's vectorised and scalar
versions, one or the other chosen at run time by the operands' addresses, can round an element differently, and the
same operands then give other last bits from one call to the next.

Compiled with Clang for arm64, the headers want `-mllvm -force-ordered-reductions=false` too (the wheels and the sdist set it
where the compiler takes it): Clang otherwise vectorises some loops that sum products in order, rounding their
products apart where the same loop inlined elsewhere fuses them, so a sum's last bits depend on its caller (a row of a
complex matrix times a vector against that row times the vector alone).

