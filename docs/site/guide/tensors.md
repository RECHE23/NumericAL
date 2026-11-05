# Tensors and dtypes

A {class}`numerical.Tensor` is an immutable array of one dtype: rank 0 a scalar, rank 1 a vector, rank 2 a matrix,
and stacks of matrices along leading axes. Its dtypes are `float64`, `complex128`, `float32`, `complex64`, `bool`
for a comparison's answer, and `int64`, NumPy's integers and Python's read as it: exactly, past 2⁵³ too, it
compares, selects, reshapes, concatenates, indexes, finds its largest and smallest elements, and computes as NumPy's
integers (wrapping modulo 2⁶⁴: `+ - * // % **`, the sums and products, `@`, `einsum` and `dot`; `/`, `mean`, `var`,
`std`, the elementary functions, the linear algebra and the transforms in float64). A bool's sums and products are
int64 too, as NumPy's.

## Reading arrays

{func}`numerical.asarray` copies any NumPy array (bool, integer, float32 or float64, complex64 or complex128, in
either byte order, at any strides), a nested list or tuple of numbers, or a number: a float32, complex64, complex128,
bool or signed integer array as it is (its integers int64, and unsigned ones but uint64, which int64 does not hold
and NumPy promotes beside int64 to float64), Python's ints as int64 (OverflowError past it), anything else into
float64 (complex128 for a complex list or number, bool for bools alone), or into the `dtype` asked. A complex input is never cast to a real dtype. Into int64, a real is truncated toward
zero, as NumPy's, and NaN, an infinity or a value outside int64's range is refused (`ValueError`, `OverflowError`),
as Python's `int()` refuses them, where NumPy's answer depends on the processor.

```pycon
>>> import numerical
>>> numerical.asarray([[1, 2], [3, 4]]).dtype
'float64'
>>> numerical.asarray([1j, 2]).dtype
'complex128'
>>> numerical.asarray([True, False]).dtype
'bool'
>>> numerical.asarray([2**62 + 1, -3], dtype="int64")
numerical.Tensor([4611686018427387905, -3], dtype=int64)
```

`matmul` and `@` read a NumPy array where it lies, without a copy, when it is row-major, contiguous, aligned, in this
machine's byte order and of one of the four numeric dtypes; `solve`, `det`, `slogdet` and `inv` read a square float64
or complex128 array so too, into the one copy their factorisation takes (a product of order 512 by a vector: 18 µs,
against 231 µs through a copy). Any other array is copied first, with the same answer.

## Computing

A `Tensor` indexes (`t[i, j]`, `t[1:5:2, ::-1]`, `t[..., 0]`, `t[None, i]`, `t[[2, 0]]`, `t[:, [1, 0], 2]`,
`t[t > 0]`), iterates, transposes (`t.T`), multiplies (`a @ b`, `numerical.matmul`, stacks of matrices along leading
axes broadcast against each other) and computes as a NumPy array does, broadcasting as NumPy's; shapes that do not
broadcast raise a `ValueError` worded as NumPy's. NumPy's constructors (`zeros`, `ones`, `full`, `eye`, `arange`,
`linspace`…) and its functions that move elements (`concatenate` or `concat`, `stack`, `vstack`, `hstack`, `tril`,
`triu`, `diag`, `kron`…) carry their names:

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

## Comparisons and masks

A comparison (`< <= == != > >=`, `numerical.less` and the rest) gives a `bool` Tensor, element by element, a NaN
unequal to everything, as NumPy's; a complex tensor compares with `==` and `!=` only. `&`, `|`, `^` and `~` take bool
tensors (`logical_and` and the rest any, read as NumPy reads them: nonzero is true), `where` picks between two
operands promoted as arithmetic promotes them, `isnan`, `isinf` and `isfinite` classify, and `any` and `all` reduce as
`sum` does.

A mask beside a number takes the number's dtype (`mask * t`), and its `sum` and `mean` count its trues, as NumPy's;
two masks have no arithmetic, and neither has a mask in the linear algebra (a `TypeError` names `astype("float64")`),
as the array API leaves them. `asarray` keeps a NumPy bool array, a bool and a list of bools bool, as NumPy's; a
function that computes reads such an array-like as float64 (`inv(np.eye(2, dtype=bool))`), as NumPy's arithmetic and
linear algebra do, but beside an operand of another dtype it takes that one (a float32 Tensor plus a NumPy mask is
float32), and one that moves elements (`concatenate`, `tril`, `transpose`) keeps it bool.

```pycon
>>> mask = (t > 1) & ~numerical.isnan(t)
>>> mask.any(axis=0)
numerical.Tensor([True, True, True], dtype=bool)
>>> (mask * t).dtype
'float64'
>>> mask.mean()
0.6666666666666666
```

## Promotion

Operands of two dtypes promote as NumPy 2 promotes them (NEP 50): complex where one is, single precision only where
both are, a Python number keeping the tensor's precision (`float32 + 2.0` is float32). float32 and complex64
arithmetic is single precision, as NumPy's: `+ - * /`, and `**` by an integer, give NumPy's very bits, complex ones
included (each fused as the NumPy beside it fuses it), and a complex number's magnitude is NumPy's own;
{doc}`differences` says where the complex elementary functions come from.

```pycon
>>> single = numerical.asarray([1.0, 2.0], "float32")
>>> (single + 2.0).dtype
'float32'
>>> (single + numerical.asarray([1j, 0])).dtype
'complex128'
```

## Einstein summation

`einsum` is the unifying primitive of the algebra, as NumPy's: `"ij,jk->ik"` is the matrix product, `"i,i->"` the dot
product, `"ij->ji"` the transpose (a view), `"ii->"` the trace, `"i,j->ij"` the outer product, `"ij->i"` a row
reduction. It sums over every label absent from the output, in one order; without `->` the output is NumPy's implicit
one, and an ellipsis broadcasts its axes as NumPy's.

```pycon
>>> a = [[1.0, 2.0], [3.0, 4.0]]
>>> numerical.einsum("ij,jk->ik", a, [[5.0, 6.0], [7.0, 8.0]])
numerical.Tensor([[19.0, 22.0],
                  [43.0, 50.0]], dtype=float64)
>>> numerical.einsum("ij,jk", a, [[5.0, 6.0], [7.0, 8.0]]).shape
(2, 2)
>>> numerical.einsum("i,i->", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
numerical.Tensor(14.0, dtype=float64)
>>> numerical.einsum("...ij,...jk->...ik", numerical.ones((3, 2, 2)), a).shape
(3, 2, 2)
```

## Exchanging without a copy

A `Tensor` is read where it lies by NumPy (`np.asarray(t)`, through `__array_interface__`, a read-only view) and by
anything that takes DLPack (`np.from_dlpack(t)`, `torch.from_dlpack(t)`, JAX, CuPy). A permutation of a tensor's
axes (`t.T`, `t.mT`, `transpose`, `swapaxes`, `moveaxis`, `matrix_transpose`, `einsum("ij->ji", t)`) is a view of its
own, as NumPy's: no element is copied until an operation needs them in the new order, then once; a product, and
NumPy (`np.asarray`, `np.from_dlpack`), read a transpose where it lies. So is a slice (`t[1:3]`, `t[:, ::2]`,
`t[::-1, 2]`) of more than 512 elements, a product reading its rows apart (`t[:, :k] @ b`) where they lie, NumPy a
slice along its strides; DLPack shares no reversed one (PyTorch refuses a negative stride) and gives a copy. A smaller
slice is copied at once. An element-wise operation steps through a slice's elements where they lie; a transpose is
copied out by blocks first. As NumPy's, a view keeps the whole tensor it reads alive; its copy (`t.copy()`,
`numerical.copy(t)`, `asarray(t, copy=True)`) holds its own elements and keeps nothing else.

The other way, `numerical.asarray(array)` copies a NumPy array (or anything of NumPy's array protocols) into a
Tensor, and `numerical.from_dlpack(x)` the elements of any DLPack producer on the CPU (a PyTorch or JAX tensor, a
CuPy array moved to the host), read from its capsule along its strides: a Tensor holds its own elements, so
`copy=False` is refused.

A tensor is never written: there is no assignment by index (`t[i] = x` raises `TypeError`), so a view never writes
through to its base, where a NumPy view does. A NumPy array passed in is copied, so writing to it later never reaches a
tensor. PETSc (petsc4py) has no such exchange protocol; bridging it takes an explicit copy into a PETSc object.
