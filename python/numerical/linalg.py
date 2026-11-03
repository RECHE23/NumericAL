"""Linear algebra, as :mod:`numpy.linalg`: its names, so that code written against ``numpy.linalg`` (``xp.linalg`` in
the array API) runs unchanged. Each is the function of that name at the root of :mod:`numerical` (the same object:
NumPy's defaults, SciPy's further arguments accepted too), but for :func:`cross`, along one axis, and :func:`outer`,
of vectors alone, where ``numpy.cross`` and ``numpy.outer`` take more."""

import operator as _operator

import numerical as _root

from numerical import (LinAlgError, cholesky, cond, det, diagonal, eig, eigh, eigvals, eigvalsh, inv, lstsq,
                       matmul, matrix_norm, matrix_power, matrix_rank, matrix_transpose, multi_dot, norm, pinv,
                       qr, slogdet, solve, svd, svdvals, tensordot, tensorinv, tensorsolve, trace, vecdot, vector_norm)

__all__ = ["matrix_power", "solve", "tensorsolve", "tensorinv", "inv", "cholesky", "eigvals", "eigvalsh", "pinv",
           "slogdet", "det", "svd", "svdvals", "eig", "eigh", "lstsq", "norm", "qr", "cond", "matrix_rank",
           "LinAlgError", "multi_dot", "trace", "diagonal", "cross", "outer", "tensordot", "matmul", "matrix_transpose",
           "matrix_norm", "vector_norm", "vecdot"]


def cross(x1, x2, /, *, axis=-1):
    """The cross product of 3-vectors along `axis`, broadcast over the other axes, as ``numpy.linalg.cross``
    (NumPy 2.5's): :func:`numerical.cross` along one axis for both operands and the result."""
    return _root.cross(x1, x2, axis=_operator.index(axis))  # one axis: TypeError for any other, as NumPy's


def outer(x1, x2, /):
    """The outer product of two vectors, ``x1[i] * x2[j]``, as ``numpy.linalg.outer``: :func:`numerical.outer` of
    vectors alone."""
    a, b = (_root.asarray(x) if not isinstance(x, _root.Tensor) else x for x in (x1, x2))
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError("outer: two vectors (numpy.linalg.outer)")
    return _root.outer(a, b)
