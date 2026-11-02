# NumericAL

**Dense linear algebra, tensors and the FFT whose answers are the same to the bit on any number of threads.**

NumericAL is a header-only C++20 library with a Python binding that follows NumPy and SciPy: `solve`, `eigh`, `svd`,
`expm`, `einsum`, `fft` and some two hundred more, on float32, float64 and their complex types. It depends on nothing
but the C++ standard library; the Python package imports in about 10 ms and starts no thread until a computation
pays for one.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Get started
:link: getting-started
:link-type: doc

Install the Python package or the headers, and solve your first system.
:::

:::{grid-item-card} Differences from NumPy and SciPy
:link: guide/differences
:link-type: doc

What a ported program meets: defaults, refusals, dtypes, and what is not there yet.
:::

:::{grid-item-card} Guides
:link: guide/index
:link-type: doc

Tensors and dtypes, the linear algebra, reproducibility and threads, accuracy, the BLAS ABI.
:::

:::{grid-item-card} Python reference
:link: reference/python
:link-type: doc

Every public function and class of `numerical` and `numerical.fft`.
:::

:::{grid-item-card} C++ reference
:link: reference/cpp
:link-type: doc

`numerical::matrix`, `tensor`, the decompositions, the free functions and the exceptions.
:::

:::{grid-item-card} Benchmarks
:link: https://github.com/RECHE23/NumericAL/blob/main/BENCHMARKS.md
:link-type: url

Against NumPy, SciPy, OpenBLAS, Accelerate and Eigen, every answer checked first.
:::
::::

## What it guarantees

- **The same bits on one thread as on all**, and on every x86-64 processor with AVX2 or AVX-512 for one build: each
  sum runs in one written order, whatever the threads or the vector width.
- **A refusal rather than a meaningless answer.** A NaN or an infinity in a matrix raises `LinAlgError`; a singular
  system raises; an ill-conditioned one warns.
- **Backward-stable answers within stated bounds**, held by differential fuzzers against NumPy and SciPy on
  adversarial matrices (see {doc}`guide/accuracy`).
- **No dependency and no idle spinning**: no BLAS underneath, no thread busy-waiting between calls, safe after
  `fork`.

## A first look

```pycon
>>> import numerical
>>> a = numerical.asarray([[2.0, 0.0], [0.0, 4.0]])
>>> numerical.solve(a, [2.0, 2.0])
numerical.Tensor([1.0, 0.5], dtype=float64)
>>> numerical.einsum("ii->", a)
numerical.Tensor(6.0, dtype=float64)
>>> numerical.vector_norm([1e200, 1e200])
1.414213562373095e+200
```

```{toctree}
:hidden:
:maxdepth: 2

getting-started
guide/index
reference/index
```
