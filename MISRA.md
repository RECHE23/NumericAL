# MISRA C++:2023 deviations

`make misra` runs clang-tidy with the **shared SciForge profile** (`lint/clang-tidy-misra` in SciForge), which covers
a material subset of MISRA C++:2023 through the cppcoreguidelines, cert, bugprone, hicpp and misc modules. The scope
is NumericAL's own headers (`--header-filter='include/numerical/.*'`; a synthetic translation unit includes
`<numerical/numerical.hpp>`); `make lint` runs clang-tidy over the tests and `blas/` as well. The profile's
deviations, the checks left disabled because they conflict with idiomatic header-library choices, apply to
NumericAL as to the rest of the ecosystem and are documented in SciForge's own MISRA notes.

NumericAL disables no further check. Each suppression in the source is a `NOLINT` naming its check and its reason on
the line; they fall into these families:

| Check | Where | Why it is suppressed rather than rewritten |
| --- | --- | --- |
| `cppcoreguidelines-pro-type-reinterpret-cast` | `x86.hpp`, `simd.hpp`, `gemm.hpp`, `fft*.hpp`, `amx.hpp`, `factorizations.hpp` | the SIMD intrinsics take `double*`, `__m256d*` and the like, and `std::complex<T>` is read as `T[2]`, which the standard guarantees ([complex.numbers.general]); no cast-free spelling exists |
| `hicpp-no-assembler` | `amx.hpp` | Apple's matrix coprocessor has no intrinsics: its instructions are emitted as encoded words, behind a run-time self-test |
| `cppcoreguidelines-macro-usage` | `simd.hpp`, `x86.hpp`, `gemm.hpp`, `buffer.hpp`, `fft.hpp` | attributes (`always_inline`, `target`, `noinline`), which no function or constant can stand for |
| `cppcoreguidelines-pro-type-member-init`, `hicpp-member-init` | `gemm.hpp`, `fft_kernels.hpp` | scratch arrays in the innermost kernels, every element written before it is read; zeroing them is a measured cost |
| `hicpp-signed-bitwise` | `parallel.hpp` | the system's `CPU_SET`, `CPU_ISSET` and `CPU_ZERO` macros |
| `cppcoreguidelines-owning-memory` | `buffer.hpp` | the cache of freed blocks lives as long as the process, so it is never destroyed while a thread could still return a block to it |
| `hicpp-explicit-conversions`, `google-explicit-constructor` | `buffer.hpp` | the allocator's rebinding constructor, which the standard containers call implicitly |
| `cppcoreguidelines-pro-type-vararg`, `concurrency-mt-unsafe`, `cppcoreguidelines-no-malloc`, `bugprone-throwing-static-initialization` | `blas/numerical_blas.cpp` | LAPACK's `xerbla` message on standard error, the environment read once at load, `_dupenv_s`'s allocation freed, and a `noexcept` initializer |
| `google-build-using-namespace` | `python/src/_numerical.cpp` | the module's registration names every binding function |

Every other MISRA-relevant concern is handled by keeping the check enabled and satisfying it.
