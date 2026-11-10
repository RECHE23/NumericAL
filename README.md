# NumericAL

A pure, header-only C++20 **numerical computing** library: tensors, Einstein summation, dense linear algebra,
matrix functions and FFT, generic over its scalar type and depending on nothing but the C++ standard library. Its
answers are the same to the bit on any number of threads, and its Python binding follows NumPy and SciPy. It is
one of the Algorithmic Libraries of the [SciForge](https://github.com/RECHE23/sciforge) ecosystem, beside
[REAL](https://github.com/RECHE23/real-regex).

NumericAL computes with concrete numeric structures (vectors, matrices, tensors) and knows nothing of expressions
or of any symbolic layer; the tools built on top of it own that.

**Documentation: <https://reche23.github.io/NumericAL/>**, with the guides, the Python and C++ references, a page of differences from NumPy and
SciPy, and the Doxygen reference of every header under [`api/`](https://reche23.github.io/NumericAL/api/).

## Scope and non-goals

- **In scope:** dense linear algebra (NumPy's `linalg` and most of `scipy.linalg`), tensors and `einsum`, the FFT,
  matrix functions, and the element-wise functions and reductions over them, in float32, float64 and their complex
  types, with bool for comparisons and int64 for integers (not yet through the linear algebra); a BLAS and LAPACK ABI
  (`numerical_blas`).
- **Views, not assignment:** a transpose, a permutation of axes (`.T`, `transpose`, a permuting `einsum`) and a basic
  slice (`t[1:500]`, `t[::2]`, `t[::-1]`) are views that read the elements where they lie. A `Tensor` is immutable
  (`t[i] = x` raises `TypeError`), so unlike a NumPy view, a view never writes through to its base.
- **Not yet:** the small dtypes (float16, int8, unsigned integers), NumPy's file input and output, printing options
  and datetimes; [the differences page](https://reche23.github.io/NumericAL/guide/differences.html) lists them, and the CHANGELOG what each release
  adds.
- **Non-goals:** the GPU, automatic differentiation, and the sparse, optimization, statistics and signal domains,
  which belong in libraries of their own built on NumericAL.

## Install

```sh
pip install numeric-al        # the distribution is numeric-al; the module is numerical
```

One `cp310` abi3 wheel serves CPython 3.10 and later; the free-threaded 3.14t and 3.15t take their own and run without
the GIL. For C++, add `include/` to the include path, or after `cmake --install`, in your `CMakeLists.txt`:

```cmake
find_package(numerical CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE numerical::numerical) # C++20, and the threads library
```

`add_subdirectory` and `FetchContent` give the same target; `numerical.get_include()` gives the headers of an
installed Python package.

## Quick start

C++ (this is `cmake/smoke/main.cpp`, which CI builds against the installed package):

```cpp
#include <cmath>
#include <complex>
#include <cstdio>

#include <numerical/numerical.hpp>

int main()
{
  const numerical::matrix<double> a {{4.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 2.0}};
  const numerical::vector<double> b {1.0, 2.0, 3.0};

  const numerical::vector<double>              x       {numerical::solve(a, b)}; // LU, partial pivoting
  const numerical::symmetric_eigen             eigen   {a};                      // A = V L V^T, as numpy.linalg.eigh
  const numerical::eigen_decomposition<double> general {a};                      // as numpy.linalg.eig
  const numerical::svd_decomposition<double>   svd     {a};                      // A = U S V^H

  std::printf("x = %.6f %.6f %.6f\n", x[0], x[1], x[2]);
  std::printf("eigh: %.6f %.6f %.6f\n", eigen.values()[0], eigen.values()[1], eigen.values()[2]);
  std::printf("eig: %.6f%+.6fi, its vectors %zu x %zu\n", general.values()[0].real(), general.values()[0].imag(),
              general.vectors().rows(), general.vectors().cols());
  std::printf("svd: %.6f %.6f %.6f\n", svd.singular_values()[0], svd.singular_values()[1], svd.singular_values()[2]);
  const numerical::vector<double> residual {(a * x) - b};
  return std::abs(residual[0]) + std::abs(residual[1]) + std::abs(residual[2]) < 1e-12 ? 0 : 1;
}
```

Python, where every function takes a `Tensor` or anything `asarray` reads (a NumPy array, a nested list, a number):

```python
import numerical
a = [[4.0, 1.0], [1.0, 3.0]]
x = numerical.solve(a, [1.0, 2.0])                 # a @ x = b
values, vectors = numerical.eigh(a)                # numpy.linalg's names and signatures
numerical.einsum("ij,jk->ik", a, a)                # the matrix product
numerical.fft.rfft([1.0, 2.0, 3.0, 4.0])           # numpy.fft's names, and scipy.fft's dct, dst…
numerical.arange(6).reshape(2, 3).sum(axis=0)      # element-wise and reductions, broadcast as NumPy's
```

A `Tensor` passes to NumPy, PyTorch, JAX or CuPy without a copy (`np.asarray(t)`, `np.from_dlpack(t)`).

## Capabilities

- **Linear algebra** as `numpy.linalg` and most of `scipy.linalg`: LU, Cholesky, LDLᵀ, QR (with pivoting), the
  symmetric, Hermitian and general eigenproblems, the SVD, Schur and generalized Schur (QZ) forms, least squares,
  banded, Toeplitz and triangular solvers, on stacks of matrices as NumPy's
  ([linear algebra](https://reche23.github.io/NumericAL/guide/linalg.html)).
- **Matrix functions and equations**: `expm`, `logm`, `sqrtm`, `funm`, fractional powers, the Sylvester, Lyapunov
  and Riccati solvers.
- **Tensors**: `einsum`, NumPy's broadcasting, element-wise functions, reductions, comparisons and masks, the array
  API namespace (`__array_namespace__`) and `numerical.linalg` ([tensors and dtypes](https://reche23.github.io/NumericAL/guide/tensors.html)).
- **FFT** as `numpy.fft` and the cosine and sine transforms of `scipy.fft`, any length.
- **`numerical_blas`** (experimental): 39 BLAS and LAPACK routines under their Fortran names, carried by the Linux
  wheels, to preload under a program linked against BLAS ([BLAS and LAPACK](https://reche23.github.io/NumericAL/guide/blas.html)).

[The C++ library](https://reche23.github.io/NumericAL/guide/cpp.html) lists every class and the algorithm it runs.

## Reproducibility and accuracy

A result is the same, to the bit, on any number of threads, and on any x86-64 processor with AVX2 or AVX-512 for one
build; between other kernels (portable, NEON, AMX) it may differ in the last bits. Every decomposition is backward
stable, held by differential fuzzers against NumPy and SciPy on adversarial matrices; a NaN or an infinity in a matrix
raises `LinAlgError`. Compiled outside the wheels, the sdist and the CMake target, which set it, the headers want
`-ffp-contract=off`, so that no multiply-add is fused that the code does not write.

[Reproducibility and threads](https://reche23.github.io/NumericAL/guide/reproducibility.html) says why, and how to cap the threads and choose the
kernel; [Accuracy](https://reche23.github.io/NumericAL/guide/accuracy.html) gives the bound each function is held to.

## Toolchains

Each push is built and tested with GCC 14 and Clang 21 (Linux x86-64, under ASan and UBSan too), GCC 13 (Linux
arm64), MSVC 2022 (Windows x64) and Apple Clang 16 (macOS arm64, locally). CMake 3.16 or newer; Python 3.10 or newer
for the binding. SciForge's quality bar holds: 100 % coverage, Clang and GCC, lint, format, MISRA-oriented analysis
and sanitizers green. [CONTRIBUTING.md](https://github.com/RECHE23/NumericAL/blob/main/CONTRIBUTING.md) gives the
build and the gates.

## Benchmarks

On an M1 Pro against Accelerate, at order 1024, matrix products, `solve` and `inverse` are as fast or faster (0.94×),
Cholesky, QR and `eigh` within 1.0–1.2×, `svd` and `lstsq` 1.3–1.4×; small problems (order 64–256) take 1.5–3×
Accelerate's time. Against OpenBLAS on x86-64 (i9-10900X, i9-9900K) large problems (order 512 and 1024) take
1.1–2.5×, `eigh` 2.6–2.9×. `make bench-linalg` times NumericAL against the BLAS and LAPACK NumPy ships, each library
alone in its process, every answer checked first. The tables, the method and where the gaps are:
[BENCHMARKS.md](https://github.com/RECHE23/NumericAL/blob/main/BENCHMARKS.md).

## Changes

[CHANGELOG.md](https://github.com/RECHE23/NumericAL/blob/main/CHANGELOG.md) lists what each release adds, changes
and fixes.

## License

MIT — see [LICENSE](LICENSE). Several algorithms follow LAPACK's routines closely; its BSD license, which their
redistribution carries, is in [NOTICE](NOTICE).

## Author

René Chenard
