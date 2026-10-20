# Releasing NumericAL

`make release` computes the next calendar version `YYYY.M.PATCH` — the patch
resets each month, the first release of a month is `.0` (PEP 440 drops leading
zeros, so `2026.6.1`, never `2026.06.001`) — bumps it in `pyproject.toml` and
`python/numerical/__init__.py`, then commits, tags and pushes from a clean `main`.
Pushing the tag drives `release.yml`, which checks the tag matches the version,
builds the abi3 wheels (Linux x86-64/aarch64, macOS universal, Windows) and the
sdist, and publishes to PyPI via Trusted Publishing (OIDC, no stored secret).

One-time PyPI setup (before the first release): create a
[Trusted Publisher](https://docs.pypi.org/trusted-publishers/) for the project
`numeric-al` — owner `RECHE23`, repository `NumericAL`, workflow `release.yml`,
environment `pypi` — and create the matching GitHub Environment named `pypi`.
Until that exists the build/sdist jobs still run, but the publish step fails;
the pushed tag remains a valid versioned snapshot regardless.

## Verification tiers

The checks run in three tiers, through SciForge's shared `make/quality.mk` (the sibling `../sciforge` checkout):
`make check-commit` before each commit (format, the suite under coverage at 100 %), `make check-push` before each
push (that, then g++-14, the sanitizers, MISRA, lint, the documentation, the Python suite and its stubs) and
`make check-release` before a release (both, then `bench-diff` against the last release and `fuzz-linalg`).
`make release` runs `bench-diff` itself as well. Each tier stops at its first red step, prints each step's time,
refuses to run beside another gate in the same tree, and voids its verdict if the tree moved while it ran.

## Versions and breaking changes

Versions are calendar-based, `YYYY.M.PATCH`. A change that can break a user's code or results it depends on — a
removed or renamed function, a changed signature or default, a C++ enumerator's value, a different answer for the
same input beyond its last bits — is listed under a **Breaking** heading of its release's CHANGELOG entry, and is
held for the first release of a month (`YYYY.M.0`), so that the patch releases within a month only add and fix.
Changes in the last bits of a floating-point result, from a faster kernel summing in another order, are not
breaking: each is noted under **Changed**.

## Release cadence

A change goes out when it is ready: a correctness fix, an addition or a speed-up is released as soon as the release
checks pass (`make check-release`), carrying `main` as it is. Only a change that can break existing code waits: it is
announced under **Breaking** in the CHANGELOG's `## Unreleased` section and released with the first release of a month
(`YYYY.M.0`), so that the releases within a month only add, speed up and fix.

## Stable surface

What the policy above protects, and what it does not:

- **Stable — C++.** Every name in namespace `numerical` that `<numerical/numerical.hpp>` brings in (`matrix`,
  `vector`, `tensor`, the decompositions and their results, the free functions, the exceptions,
  `set_max_threads`, `max_threads`), with its signature, its defaults and the meaning of its answer.
- **Stable — Python.** Every name in `numerical.__all__` but the four below, with NumPy's semantics where the
  name is NumPy's and the divergences the README lists (a NaN or an infinity raises; `float32` stays `float32`).
- **Tuning, not a contract.** `gemm_path`, `limit_gemm_path` and `get_config` (Python), `limit_gemm_path` and
  `gemm_path_in_use` (C++), the `NUMERICAL_GEMM` variable: they name the kernels this release has, which a later
  one may add to, rename or retire.
- **Experimental.** `numerical_blas`, the BLAS/LAPACK ABI library, and `numerical.blas_library()`, which locates it:
  carried by the Linux wheels and built by CMake on request (`-DNUMERICAL_BUILD_BLAS=ON`), its routines and their
  coverage still growing.
- **Internal.** Namespace `numerical::detail`, every `NUMERICAL_*` macro, and the layout of the headers
  themselves (include `<numerical/numerical.hpp>`, not a file within it).
- **The answers.** On one build, the same input gives the same bits on any number of threads. Across releases a
  result may move in its last bits (a faster kernel summing in another order, noted under **Changed**); a larger
  change is **Breaking**.

## Correctness record

The day of the last correctness finding — a wrong answer, a crash, or a broken guarantee such as the same bits on
any number of threads — in a released version, and the release that fixed it. The library is called mature once
six weeks have passed without one, six weekly long fuzzing campaigns green on x86-64 and arm64
(`.github/workflows/fuzz-long.yml`) beside the differential suites of every push.

| Last finding | What | Fixed in |
| --- | --- | --- |
| 2026-10-19 | `inv` not the same to the bit on one thread as on several (complex on arm64; real on the GCC-built Linux wheels), found by the fuzzer's thread arm | 2026.10.25 |
