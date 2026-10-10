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
import os

from numerical._numerical import (
    Tensor,
    asarray,
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

__version__ = "2026.10.0"


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
