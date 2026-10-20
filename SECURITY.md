# Security Policy

## Supported versions

NumericAL ships under CalVer (`YYYY.M.PATCH`). Only the **most recent release** receives fixes; older
releases do not.

| Version | Supported |
| --- | --- |
| latest release | ✅ |
| older releases | ❌ |

## Reporting a vulnerability

Please report security issues privately through GitHub's
**[Report a vulnerability](https://github.com/RECHE23/NumericAL/security/advisories/new)** (private
vulnerability reporting). Do **not** open a public issue for a security report.

## Threat model

NumericAL is a dense linear-algebra library: a header-only C++20 core, an abi3 Python binding
(`numeric-al`) and a BLAS/LAPACK ABI library (`numerical_blas`, in the Linux wheels and built by CMake on request). Its input is
numbers and shapes, and the boundary is the same through all three.

**Values.** Any value is accepted, adversarial ones included (extreme scales, denormals, exact integers,
repeated or clustered spectra): the differential fuzzer (`make fuzz-linalg`) runs every decomposition on
such matrices against NumPy on every kernel the processor has. A matrix holding a NaN or an infinity in
what a decomposition reads raises `numerical::non_finite_error` (Python: `numerical.LinAlgError`) before
any work, rather than spreading NaN through the factors. A wrong answer beyond the documented backward
error on any input is a bug; a crash, a hang, or a read or write outside the operands is in scope here.

**Shapes.** Element counts are multiplied with an overflow check (`detail::checked_mul`) before any
allocation, so a shape whose product overflows `size_t` raises instead of allocating too little. The work
is polynomial in the order (a decomposition of order `n` is O(n³) time, O(n²) memory) and has no
input-dependent blow-up past that: an iteration that does not converge (an eigenvalue or SVD sweep) is
bounded and raises. Bound the order of what you accept from untrusted callers by the time and memory you
can give it; a shape that makes the library exceed those bounds, or any memory-safety issue from a shape,
is in scope.

**The BLAS ABI.** `numerical_blas` checks its arguments as reference BLAS/LAPACK do (`xerbla`, `info < 0`)
and reads and writes only the leading dimensions it is given. A caller that passes a leading dimension or a
pointer that does not match its buffers is outside what any BLAS can defend against; an out-of-bounds
access with consistent arguments is in scope.

**Threads.** The pool runs a kernel's work on worker threads that it starts once; the results are the same
to the bit on any number of threads (`set_max_threads` / `numerical.set_num_threads`), which the fuzzer
checks. `make check-push` runs the suite under AddressSanitizer with UndefinedBehaviorSanitizer and again under
ThreadSanitizer (`make tsan`, which first requires a race planted in the pool to be reported); a data race found
by any means is in scope.
