r"""NumericAL — a generic, header-only numerical library, exposed to Python.

The one object is :class:`Tensor` — an N-dimensional, row-major array of
``float64`` or ``complex128`` (rank-1 is a vector, rank-2 a matrix, rank-0 a
scalar). The algebra over it is reached through module functions:

    import numerical
    a = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
    b = numerical.Tensor([5, 6, 7, 8], shape=(2, 2))
    numerical.einsum("ij,jk->ik", a, b)        # matrix product
    numerical.einsum("i,i->", numerical.Tensor([1, 2, 3]), numerical.Tensor([1, 2, 3]))  # dot
    numerical.solve(a, numerical.Tensor([1.0, 1.0]))  # a @ x = b

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
import numbers as _numbers
import os
import sys as _sys

from numerical._numerical import (
    Tensor,
    asarray as _asarray,
    cholesky,
    determinant,
    eigh,
    einsum,
    error,
    inverse,
    lstsq,
    matrix_rank,
    pinv,
    qr,
    slogdet,
    solve,
    svd,
)

__all__ = [
    "Tensor", "einsum", "solve", "asarray", "error",
    "determinant", "slogdet", "inverse", "lstsq", "pinv", "matrix_rank", "cholesky", "qr", "eigh", "svd",
    "get_include", "get_config",
]

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

