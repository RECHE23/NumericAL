r"""NumericAL — a generic, header-only numerical library, exposed to Python.

The one object is :class:`Tensor` — an N-dimensional, row-major array of
``float64`` or ``complex128`` (rank-1 is a vector, rank-2 a matrix, rank-0 a
scalar). The algebra over it is reached through module functions, which take a
Tensor or anything :func:`asarray` reads (a NumPy array, a nested list, a number):

    import numerical
    a = [[1, 2], [3, 4]]
    numerical.einsum("ij,jk->ik", a, [[5, 6], [7, 8]])  # matrix product
    numerical.einsum("i,i->", [1, 2, 3], [1, 2, 3])      # dot
    numerical.solve(a, [1.0, 1.0])                        # a @ x = b

A failure is a :class:`LinAlgError`, a ``ValueError`` as NumPy's is.

A :class:`Tensor` interoperates zero-copy with the scientific Python stack:

    import numpy as np
    np.asarray(a)            # via __array_interface__ (read-only view)
    np.from_dlpack(a)        # via __dlpack__
    numerical.asarray(np.eye(3))  # the other direction (a contiguous-array copy)

PyTorch / JAX / CuPy consume the same ``__dlpack__`` (``torch.from_dlpack(a)``).
PETSc (petsc4py) has no such protocol; bridging it would require an explicit
copy into a PETSc object.
"""
import array as _array
import functools as _functools
import numbers as _numbers
import operator as _operator
import os
import sys as _sys

from numerical import _numerical
from numerical._numerical import Tensor, asarray as _asarray, error

__all__ = [
    "Tensor", "einsum", "solve", "asarray", "error", "LinAlgError",
    "determinant", "slogdet", "inverse", "lstsq", "pinv", "matrix_rank", "cholesky", "qr", "eigh", "svd",
    "get_include", "get_config",
]


class LinAlgError(error, ValueError):
    """Raised by the linear-algebra functions when they cannot compute their answer: a singular matrix, one
    not positive definite, an infinite or NaN entry, an iteration that did not converge, or an operand of the
    wrong shape. As ``numpy.linalg.LinAlgError`` it is a :class:`ValueError`; it is also a
    :class:`numerical.error`, so code catching that keeps working."""


__version__ = "2026.10.1"


def get_include():
    """Return the directory to add to a C++ include path for NumericAL's headers.

    ``#include <numerical/numerical.hpp>`` resolves against this directory. NumericAL is
    header-only and shipped inside the installed package, so a project can
    compile against it located through its Python install. NumericAL is a leaf
    library — it needs no other include paths.

    Returns:
        str: Absolute path to NumericAL's include directory.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    packaged = os.path.join(here, "include")
    if os.path.isdir(os.path.join(packaged, "numerical")):
        return packaged
    return os.path.normpath(os.path.join(here, os.pardir, os.pardir, "include"))


def get_config():
    """Return metadata for embedding the C++ library.

    Returns:
        dict: ``version`` (str), ``include`` (str, see :func:`get_include`), and
        ``cxx_standard`` (str).
    """
    return {
        "version": __version__,
        "include": get_include(),
        "cxx_standard": "c++20",
    }


class _NestedBuffer:
    """Doubles in C order behind an ``__array_interface__``: how a nested list or a scalar reaches the
    same reader as an array (real values as float64, complex ones as complex128)."""

    def __init__(self, values, shape):
        is_complex = any(isinstance(value, complex) or (isinstance(value, _numbers.Complex)
                                                         and not isinstance(value, _numbers.Real))
                         for value in values)
        flat = []
        for value in values:
            if is_complex:
                number = complex(value)
                flat += [number.real, number.imag]
            else:
                flat.append(float(value))
        self._data = _array.array("d", flat)
        order = "<" if _sys.byteorder == "little" else ">"
        self.__array_interface__ = {"version": 3, "typestr": order + ("c16" if is_complex else "f8"),
                                    "shape": tuple(shape), "data": (self._data.buffer_info()[0], True)}


def _nested(obj):
    """The values of a nested list (or tuple) of numbers in C order, and its shape."""
    if isinstance(obj, (list, tuple)):
        parts = [_nested(item) for item in obj]
        if not parts:
            return [], (0,)
        if len({shape for _, shape in parts}) != 1:
            raise ValueError("asarray: a nested list must be rectangular (every row of one length)")
        return [value for values, _ in parts for value in values], (len(parts),) + parts[0][1]
    if isinstance(obj, _numbers.Number):
        return [obj], ()
    raise TypeError(f"asarray: a nested list holds numbers, not {type(obj).__name__}")


def _linalg(native):
    """`native` taking array-likes: every argument that is not a :class:`Tensor` or a string goes through
    :func:`asarray` (a NumPy array, a nested list, a number); when one operand is complex the real ones are
    widened to complex128, as NumPy promotes them; and its failures are :class:`LinAlgError`."""

    @_functools.wraps(native)
    def call(*args):
        operands = [arg if isinstance(arg, (Tensor, str)) else asarray(arg) for arg in args]
        if any(isinstance(operand, Tensor) and operand.dtype == "complex128" for operand in operands):
            operands = [asarray(operand, "complex128") if isinstance(operand, Tensor) else operand
                        for operand in operands]
        try:
            return native(*operands)
        except error as caught:
            raise LinAlgError(*caught.args) from None

    return call


def _getitem(self, key):
    """``t[i]``, ``t[i, j]``, ``t[1:5:2, ::-1]``: integers drop an axis, slices keep one, missing trailing
    indices take a whole axis, as NumPy's basic indexing; a negative index counts from the end. All integers
    give a number, anything else a Tensor (a copy: a Tensor is never written through, so it reads as NumPy's
    view would)."""
    if not isinstance(key, tuple):
        key = (key,)
    shape = self.shape
    if len(key) > len(shape):
        raise IndexError(f"too many indices for a tensor of rank {len(shape)}: {len(key)} were given")
    starts, steps, counts, keep = [], [], [], []
    for axis, extent in enumerate(shape):
        index = key[axis] if axis < len(key) else slice(None)
        if isinstance(index, slice):
            span = range(*index.indices(extent))
            starts.append(span.start if len(span) else 0)
            steps.append(span.step)
            counts.append(len(span))
            keep.append(1)
            continue
        position = _operator.index(index)
        if position < 0:
            position += extent
        if not 0 <= position < extent:
            raise IndexError(f"index {index} is out of bounds for axis {axis} with size {extent}")
        starts.append(position)
        steps.append(1)
        counts.append(1)
        keep.append(0)
    return _numerical._take(self, starts, steps, counts, keep)


def _len(self):
    """The extent of the first axis, as ``len`` of a NumPy array; a rank-0 tensor has none."""
    if not self.shape:
        raise TypeError("len() of a rank-0 tensor")
    return self.shape[0]


def _iter(self):
    """The tensor's slices along its first axis (numbers for a vector), as iterating a NumPy array."""
    if not self.shape:
        raise TypeError("iteration over a rank-0 tensor")
    return (self[i] for i in range(self.shape[0]))


def _transpose(self):
    """The tensor with its axes reversed: a matrix's transpose, as NumPy's ``.T``."""
    letters = "abcdefghijklmnopqrstuvwxyz"[:len(self.shape)]
    return einsum(f"{letters}->{letters[::-1]}", self)


_MATMUL = {(1, 1): "i,i->", (2, 1): "ij,j->i", (1, 2): "i,ij->j", (2, 2): "ij,jk->ik"}


def _matmul(self, other):
    """``a @ b`` for vectors and matrices, as NumPy's matmul of rank 1 and 2 (a number for two vectors)."""
    other = other if isinstance(other, Tensor) else asarray(other)
    spec = _MATMUL.get((len(self.shape), len(other.shape)))
    if spec is None:
        raise LinAlgError(f"@ takes vectors and matrices, not ranks {len(self.shape)} and {len(other.shape)}")
    product = einsum(spec, self, other)
    return product[()] if not product.shape else product  # two vectors give a number, as NumPy's


def _rmatmul(self, other):
    return _matmul(asarray(other), self)


Tensor.__getitem__ = _getitem
Tensor.__len__ = _len
Tensor.__iter__ = _iter
Tensor.T = property(_transpose)
Tensor.__matmul__ = _matmul
Tensor.__rmatmul__ = _rmatmul

cholesky = _linalg(_numerical.cholesky)
determinant = _linalg(_numerical.determinant)
eigh = _linalg(_numerical.eigh)
einsum = _linalg(_numerical.einsum)
inverse = _linalg(_numerical.inverse)
lstsq = _linalg(_numerical.lstsq)
matrix_rank = _linalg(_numerical.matrix_rank)
pinv = _linalg(_numerical.pinv)
qr = _linalg(_numerical.qr)
slogdet = _linalg(_numerical.slogdet)
solve = _linalg(_numerical.solve)
svd = _linalg(_numerical.svd)


def asarray(obj, dtype=None):
    """A :class:`Tensor` from `obj`: a Tensor (the same object when `dtype` matches), an object with an
    ``__array_interface__`` (NumPy: bool, integer, float32/64 and complex64/128 elements, any strides),
    a nested list or tuple of numbers, or a number (rank 0). `dtype` is ``"float64"`` or
    ``"complex128"``; by default float64, or complex128 for complex input. A complex input is never cast
    to float64 (that would discard the imaginary part).

    Raises:
        ValueError: if a nested list is not rectangular.
        TypeError: if a nested list holds something other than numbers.
    """
    if not hasattr(obj, "__array_interface__") and (isinstance(obj, (list, tuple))
                                                    or isinstance(obj, _numbers.Number)):
        values, shape = _nested(obj)
        obj = _NestedBuffer(values, shape)
    return _asarray(obj, dtype) if dtype is not None else _asarray(obj)



class _ShapeError(error, ValueError):
    """An array operation's operands do not fit (shapes that do not broadcast, a reshape to another size, an
    axis out of range, an empty maximum): a ``ValueError`` as NumPy raises, and a :class:`numerical.error`."""


def _native(call, *args):
    """`call(*args)`, its :class:`numerical.error` re-raised as a ``ValueError`` (a :class:`_ShapeError`)."""
    try:
        return call(*args)
    except error as caught:
        raise _ShapeError(*caught.args) from None


def _same_dtype(a, b):
    """`a` and `b` as Tensors of one dtype: array-likes through :func:`asarray`, a real one widened to
    complex128 when the other is complex, as NumPy promotes them."""
    a = a if isinstance(a, Tensor) else asarray(a)
    b = b if isinstance(b, Tensor) else asarray(b)
    if a.dtype != b.dtype:
        a, b = asarray(a, "complex128"), asarray(b, "complex128")
    return a, b


def _arithmetic(op, reflected=False):
    """The Tensor method for `op` ("+", "-", "*" or "/"), element-wise and broadcast as NumPy's; a reflected
    one (``2 - t``) puts the other operand first. Anything :func:`asarray` cannot read is NotImplemented."""

    def method(self, other):
        try:
            a, b = _same_dtype(self, other)
        except (TypeError, error):  # asarray cannot read it: let the other operand try
            return NotImplemented
        if reflected:
            a, b = b, a
        return _native(_numerical._binary, op, a, b)

    return method


Tensor.__add__ = _arithmetic("+")
Tensor.__radd__ = _arithmetic("+", reflected=True)
Tensor.__sub__ = _arithmetic("-")
Tensor.__rsub__ = _arithmetic("-", reflected=True)
Tensor.__mul__ = _arithmetic("*")
Tensor.__rmul__ = _arithmetic("*", reflected=True)
Tensor.__truediv__ = _arithmetic("/")
Tensor.__rtruediv__ = _arithmetic("/", reflected=True)
Tensor.__neg__ = lambda self: _numerical._negate(self)
Tensor.__pos__ = lambda self: self


def _axis(t, axis):
    """`axis` as the native side takes it: -1 for every element, else a non-negative axis of `t`."""
    if axis is None:
        return -1
    axis = _operator.index(axis)
    rank = len(t.shape)
    if not -rank <= axis < rank:
        raise _ShapeError(f"axis {axis} is out of bounds for a tensor of rank {rank}")
    return axis + rank if axis < 0 else axis


def _reduction(op, doc):
    def reduce(a, axis=None):
        a = a if isinstance(a, Tensor) else asarray(a)
        return _native(_numerical._reduce, op, a, _axis(a, axis))

    reduce.__name__ = op
    reduce.__doc__ = doc
    return reduce


sum = _reduction("sum", """The sum of every element (a number), or the sums along `axis` (a Tensor without it),
as ``numpy.sum``; added by pairwise halving, as NumPy adds.""")
mean = _reduction("mean", """The mean of every element, or the means along `axis`, as ``numpy.mean``.""")
max = _reduction("max", """The largest element, or the largest along `axis`, as ``numpy.max``; a NaN wins. float64
only (complex128 has no order).""")
min = _reduction("min", """The smallest element, or the smallest along `axis`, as ``numpy.min``; a NaN wins.
float64 only.""")
Tensor.sum = lambda self, axis=None: sum(self, axis)
Tensor.mean = lambda self, axis=None: mean(self, axis)
Tensor.max = lambda self, axis=None: max(self, axis)
Tensor.min = lambda self, axis=None: min(self, axis)


def _shape(shape):
    """A shape argument as a tuple: an integer is a rank-1 shape, as NumPy reads it."""
    if isinstance(shape, _numbers.Integral):
        return (_operator.index(shape),)
    return tuple(_operator.index(extent) for extent in shape)


def reshape(a, shape):
    """The elements of `a` in the same order under `shape`, as ``numpy.reshape``; one extent may be -1, the
    one the others leave."""
    a = a if isinstance(a, Tensor) else asarray(a)
    shape = _shape(shape)
    unknown = [axis for axis, extent in enumerate(shape) if extent == -1]
    if len(unknown) > 1:
        raise _ShapeError("can only specify one unknown dimension")
    if unknown:
        known = 1
        for extent in shape:
            known *= extent if extent != -1 else 1
        if known == 0 or a.size % known != 0:
            raise _ShapeError(f"cannot reshape a tensor of size {a.size} into shape {shape}")
        shape = tuple(a.size // known if extent == -1 else extent for extent in shape)
    return _native(_numerical._reshape, a, shape)


Tensor.reshape = lambda self, *shape: reshape(self, shape[0] if len(shape) == 1 else shape)


def _is_complex_dtype(dtype, value=0.0):
    if dtype is None:
        return isinstance(value, complex)
    if dtype not in ("float64", "complex128"):
        raise ValueError("dtype must be 'float64' or 'complex128'")
    return dtype == "complex128"


def full(shape, value, dtype=None):
    """A Tensor of `shape` whose every element is `value`, as ``numpy.full``; complex128 for a complex value
    unless `dtype` says otherwise."""
    return _native(_numerical._full, _shape(shape), value, _is_complex_dtype(dtype, value))


def zeros(shape, dtype="float64"):
    """A Tensor of `shape` full of zeros, as ``numpy.zeros``."""
    return _native(_numerical._full, _shape(shape), 0.0, _is_complex_dtype(dtype))


def ones(shape, dtype="float64"):
    """A Tensor of `shape` full of ones, as ``numpy.ones``."""
    return _native(_numerical._full, _shape(shape), 1.0, _is_complex_dtype(dtype))


def eye(n, dtype="float64"):
    """The `n × n` identity, as ``numpy.eye(n)``."""
    return _native(_numerical._eye, _operator.index(n), _is_complex_dtype(dtype))


def arange(start, stop=None, step=1.0):
    """``start, start + step, …`` short of `stop`, as ``numpy.arange``: ``arange(stop)`` starts at 0. float64."""
    if stop is None:
        start, stop = 0.0, start
    return _native(_numerical._arange, float(start), float(stop), float(step))


def linspace(start, stop, num=50):
    """`num` values evenly spaced from `start` to `stop`, both included, as ``numpy.linspace``. float64."""
    return _native(_numerical._linspace, float(start), float(stop), _operator.index(num))


__all__ += ["sum", "mean", "max", "min", "reshape", "full", "zeros", "ones", "eye", "arange", "linspace"]
