# Contributing to NumericAL

NumericAL is a header-only C++20 dense linear-algebra library with an abi3 Python binding (`numeric-al`) and a
BLAS/LAPACK ABI library (`numerical_blas`). A few conventions keep its answers exact and its speed honest.

## Building and testing

The library needs only a C++20 compiler. The test suite, the binding and the lint tooling need
[SciForge](https://github.com/RECHE23/sciforge) checked out **as a sibling** (the test harness, the lint
configuration and the shared `make/quality.mk`):

```
Projects/
  numerical/       <- here
  sciforge/        <- git clone https://github.com/RECHE23/sciforge
```

Then, from `numerical/`:

```sh
make check-commit   # before each commit: format, the suite under coverage at 100 % on every dimension
make check-push     # before each push: the above, g++-14 (on the AMX and on NEON alone), ASan + UBSan, TSan, MISRA, lint, doc, the Python suite
make check-release  # before a release: the above, bench-diff against the last release, fuzz-linalg, fuzz-edges
```

`make help` lists the single steps (`make test`, `make python-test`, `make fuzz-linalg`, `make fuzz-edges`, `make bench-linalg`).

## Conventions the gates and the tests hold

- **Every element is summed in one order.** A product, a decomposition or a solve gives each element its
  terms in ascending order of the depth whatever the blocking, the kernel or the number of threads, so the
  answer is the same to the bit on one thread as on all (the fuzzer's thread arm checks it). A product's
  element must not depend on the shape of the product it falls in: callers split their columns by the
  threads. Fusion may differ between instruction sets and compilers; within one build it may not.
- **No relaxing that order for speed.** A faster ordering that moves the bits (an inverse diagonal block, a
  reassociated sum) is out unless the release notes call it out and it is decided as such.
- **Measure with the wheels' flags.** Those of `setup.py` (`-O3` on x86-64, `-O2` on arm64, `-g0`,
  `-ffp-contract=on`) over the interpreter's `-DNDEBUG -fwrapv`, plus `-falign-functions=64` so that where the
  linker puts a kernel does not pass for a gain or a loss (`benchmarks/bench_diff.py`); under Clang **and** GCC
  (the Linux wheels are GCC's), on arm64 and x86-64, each library in its own process. A speed claim in
  `CHANGELOG.md` names the machine class, the compiler, the order and the before and after.
- **Test under GCC too.** The suite's CMake build holds `-ffp-contract=off`; the wheels use `on`, where GCC and
  Clang contract differently. A change to the product kernels is checked under both compilers.
- **A gate's green means the gate ran.** A new check is shown to fail once, with its subject broken on purpose,
  before it is trusted.
- **Comments state constraints, not history.** Say why the code must be as it is (a measured cost, a
  compiler's behaviour, a bound); the history belongs in the commit message.
- **A change that can break code** is listed under **Breaking** in `CHANGELOG.md` and held for a month's
  first release (`RELEASING.md`).

## Reporting

Bugs and wrong answers: an issue with the input (or the seed of `make fuzz-linalg` that finds it) and the
expected result. Security issues: see `SECURITY.md`.
