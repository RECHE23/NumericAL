# Getting started

## Python

```sh
pip install numeric-al        # the distribution is numeric-al; the module is numerical
```

The wheels serve CPython 3.10 and later through the stable ABI, and the free-threaded 3.14t and 3.15t without the
GIL, on Linux (x86-64, arm64), macOS (arm64, x86-64) and Windows (x86-64). Every function takes a
{class}`numerical.Tensor` or anything {func}`numerical.asarray` reads: a NumPy array, a nested list, a number.

```pycon
>>> import numerical
>>> a = [[4.0, 1.0], [1.0, 3.0]]
>>> numerical.solve(a, [1.0, 2.0])
numerical.Tensor([0.09090909090909091, 0.6363636363636364], dtype=float64)
>>> values, vectors = numerical.eigh(a)
>>> numerical.round(values, 12)
numerical.Tensor([2.38196601125, 4.61803398875], dtype=float64)
>>> t = numerical.arange(6).reshape(2, 3)
>>> t.sum(axis=0)
numerical.Tensor([3.0, 5.0, 7.0], dtype=float64)
>>> numerical.fft.rfft([1.0, 2.0, 3.0, 4.0])
numerical.Tensor([(10+0j), (-2+2j), (-2+0j)], dtype=complex128)
```

A `Tensor` passes to NumPy, PyTorch, JAX or CuPy without a copy (`np.asarray(t)`, `np.from_dlpack(t)`,
`torch.from_dlpack(t)`); `numerical.asarray(array)` copies the other way. {doc}`guide/tensors` says what a `Tensor`
holds and how the dtypes promote; {doc}`guide/differences` lists what a program written for NumPy or SciPy meets.

## C++

Header-only: add `include/` to the include path, or after `cmake --install`, in your `CMakeLists.txt`:

```cmake
find_package(numerical CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE numerical::numerical) # C++20, and the threads library
```

then

```cpp
#include <cstdio>

#include <numerical/numerical.hpp>

int main()
{
  const numerical::matrix<double> a {{4.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 2.0}};
  const numerical::vector<double> b {1.0, 2.0, 3.0};

  const numerical::vector<double>  x     {numerical::solve(a, b)}; // LU, partial pivoting
  const numerical::symmetric_eigen eigen {a};                      // A = V L V^T, as numpy.linalg.eigh
  std::printf("x = (%g, %g, %g), smallest eigenvalue %g\n", x[0], x[1], x[2], eigen.values()[0]);
}
```

`numerical.get_include()` gives the header directory of an installed Python package, so a C++ extension can build
against the same headers. With GCC, compile with `-ffp-contract=on` (the wheels' setting; {doc}`guide/reproducibility`
says why).
