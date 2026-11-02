# Tensors and dtypes

A {class}`numerical.Tensor` is an immutable array of one dtype: rank 0 a scalar, rank 1 a vector, rank 2 a matrix,
and stacks of matrices along leading axes. Its dtypes are `float64`, `complex128`, `float32`, `complex64`, and `bool`
for a comparison's answer.

## Reading arrays

{func}`numerical.asarray` copies any NumPy array (bool, integer, float32 or float64, complex64 or complex128, in
either byte order, at any strides), a nested list or tuple of numbers, or a number: a float32, complex64, complex128
or bool array as it is, anything else into float64 (complex128 for a complex list or number, bool for bools alone),
or into the `dtype` asked. A complex input is never cast to a real dtype.

```pycon
>>> import numerical
>>> numerical.asarray([[1, 2], [3, 4]]).dtype
'float64'
>>> numerical.asarray([1j, 2]).dtype
'complex128'
>>> numerical.asarray([True, False]).dtype
'bool'
```

`matmul` and `@` read a NumPy array where it lies, without a copy, when it is row-major, contiguous, aligned, in this
machine's byte order and of one of the four numeric dtypes; `solve`, `det`, `slogdet` and `inv` read a square float64
or complex128 array so too, into the one copy their factorisation takes.

## Computing

A `Tensor` indexes (`t[i, j]`, `t[1:5:2, ::-1]`, `t[..., 0]`, `t[None, i]`, `t[[2, 0]]`, `t[t > 0]`), iterates,
transposes (`t.T`) and computes as a NumPy array does, broadcasting as NumPy's:

```pycon
>>> t = numerical.arange(6).reshape(2, 3)
>>> (t + 1) * [10, 0, 1] / 2
numerical.Tensor([[5.0, 0.0, 1.5],
                  [20.0, 0.0, 3.0]], dtype=float64)
>>> t.max(axis=-1)
numerical.Tensor([2.0, 5.0], dtype=float64)
>>> mask = t > 1
>>> numerical.where(mask, t, 0.0)
numerical.Tensor([[0.0, 0.0, 2.0],
                  [3.0, 4.0, 5.0]], dtype=float64)
>>> int(mask.sum())
4
```

## Promotion

Operands of two dtypes promote as NumPy 2 promotes them (NEP 50): complex where one is, single precision only where
both are, a Python number keeping the tensor's precision (`float32 + 2.0` is float32). float32 and complex64
arithmetic is single precision, as NumPy's: `+ - * /`, and `**` by an integer, give NumPy's very bits.

```pycon
>>> single = numerical.asarray([1.0, 2.0], "float32")
>>> (single + 2.0).dtype
'float32'
>>> (single + numerical.asarray([1j, 0])).dtype
'complex128'
```

## Exchanging without a copy

A `Tensor` is read where it lies by NumPy (`np.asarray(t)`, through `__array_interface__`, a read-only view) and by
anything that takes DLPack (`np.from_dlpack(t)`, `torch.from_dlpack(t)`, JAX, CuPy). NumericAL makes no views of its
own: `einsum("ij->ji", a)` returns a new tensor where NumPy returns a view, so writing to the result never reaches
`a`.
