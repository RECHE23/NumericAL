r"""NumericAL — a generic, header-only numerical library, exposed to Python.

The one object is :class:`Tensor` — an N-dimensional, row-major array of
``float64``, ``complex128``, ``float32`` or ``complex64`` (rank-1 is a vector, rank-2 a matrix, rank-0 a
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
import builtins as _builtins
import functools as _functools
import math as _math
import numbers as _numbers
import operator as _operator
import os as _os
import sys as _sys
import warnings as _warnings

from numerical import _numerical
from numerical._numerical import Tensor, asarray as _asarray, _asarray_nested, error, get_num_threads
from numerical._numerical import _gemm_path, _limit_gemm_path, _set_num_threads

__all__ = [
    "Tensor", "einsum", "solve", "asarray", "error", "LinAlgError",
    "determinant", "det", "slogdet", "inverse", "inv", "solve_banded", "solveh_banded", "solve_toeplitz", "solve_circulant",
    "toeplitz", "hankel", "circulant", "block_diag",
    "expm", "expm_frechet", "expm_cond", "fractional_matrix_power", "funm", "logm", "LogmRankWarning", "LogmExactlySingularWarning",
    "LogmNearlySingularWarning", "sqrtm", "polar",
    "solve_sylvester", "solve_continuous_lyapunov", "solve_lyapunov", "solve_discrete_lyapunov",
    "solve_continuous_are", "solve_discrete_are", "lstsq", "pinv",
    "matrix_rank", "cholesky", "ldl", "qr", "eig",
    "eigvals", "eigh", "eigvalsh", "svd", "svdvals", "norm", "vector_norm", "matrix_norm", "cond", "matrix_power", "multi_dot",
    "trace", "outer", "cross", "tensordot", "vecdot", "conj", "conjugate", "matmul",
    "array_equal", "set_num_threads", "get_num_threads", "gemm_path", "limit_gemm_path",
    "get_include", "get_config", "blas_library",
    "schur", "solve_triangular", "lu_factor", "lu_solve", "cho_factor", "cho_solve", "LUFactor", "CholeskyFactor",
]


class LinAlgError(error, ValueError):
    """Raised by the linear-algebra functions when they cannot compute their answer: a singular matrix, one
    not positive definite, an infinite or NaN entry, an iteration that did not converge, or an operand of the
    wrong shape. As ``numpy.linalg.LinAlgError`` it is a :class:`ValueError`; it is also a
    :class:`numerical.error`, so code catching that keeps working."""


__version__ = "2026.10.41"

def set_num_threads(n):
    """Cap the threads a kernel uses at `n`, the caller's included; ``0`` lifts the cap. At ``1`` no thread is
    started. For a process per core, say, so that together they do not ask for more threads than the machine
    has, as ``OMP_NUM_THREADS`` caps BLAS."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"set_num_threads: a count of threads is a non-negative integer, not {n!r}")
    _set_num_threads(n if n < 2**31 else 2**31)  # not min(): this module defines its own, a reduction


def _environment_threads():
    """NUMERICAL_NUM_THREADS, read once at import: a count of threads, or None (with a warning if it is not one)."""
    value = _os.environ.get("NUMERICAL_NUM_THREADS", "").strip()
    if not value:
        return None
    if value.isascii() and value.isdigit():
        return int(value)
    _warnings.warn(f"NUMERICAL_NUM_THREADS={value!r} is not a count of threads; ignored", RuntimeWarning, stacklevel=2)
    return None


_threads = _environment_threads()
if _threads is not None:
    set_num_threads(_threads)
del _threads

_GEMM_PATHS = ("portable", "neon", "avx2", "avx512", "amx")


def gemm_path():
    """The kernel float64 matrix products run on: ``"portable"`` (the compiler's vectors), ``"neon"`` (arm64's
    vectors, each multiply-add fused), ``"avx2"``, ``"avx512"`` (x86-64, chosen at run time from cpuid) or ``"amx"``
    (Apple M1 to M4, an undocumented coprocessor, used only on a known chip that passes a self-test, the products it
    does not suit on NEON; see numerical/amx.hpp)."""
    return _GEMM_PATHS[_gemm_path()]


def limit_gemm_path(most):
    """Keep float64 matrix products on `most` or a more portable path (one of :func:`gemm_path`'s names): the most
    capable path at or below it that runs on this processor. ``"amx"`` restores the best; ``"portable"`` keeps
    products on the compiler's own vectors. Paths below the cap are not probed (under ``"avx512"`` or below, the
    Apple AMX's self-test never runs). Returns the path now in use."""
    if most not in _GEMM_PATHS:
        raise ValueError(f"limit_gemm_path: {most!r} is none of {_GEMM_PATHS}")
    return _GEMM_PATHS[_limit_gemm_path(_GEMM_PATHS.index(most))]


def _environment_gemm_cap():
    """The cap NUMERICAL_GEMM and NUMERICAL_AMX set, the more portable of the two: NUMERICAL_GEMM names a path
    (portable, neon, avx2, avx512, amx); NUMERICAL_AMX set to 0, false, no or off keeps products off the Apple AMX, its
    self-test included. A value neither reads is ignored, with a warning."""
    cap = len(_GEMM_PATHS) - 1
    amx = _os.environ.get("NUMERICAL_AMX", "").strip().lower()
    if amx in ("0", "false", "no", "off"):
        cap = _GEMM_PATHS.index("avx512")
    elif amx not in ("", "1", "true", "yes", "on"):
        _warnings.warn(f"NUMERICAL_AMX={amx!r} is neither on nor off; ignored", RuntimeWarning, stacklevel=2)
    gemm = _os.environ.get("NUMERICAL_GEMM", "").strip().lower()
    if gemm in _GEMM_PATHS:
        cap = cap if cap < _GEMM_PATHS.index(gemm) else _GEMM_PATHS.index(gemm)
    elif gemm:
        _warnings.warn(f"NUMERICAL_GEMM={gemm!r} is none of {_GEMM_PATHS}; ignored", RuntimeWarning, stacklevel=2)
    return _GEMM_PATHS[cap]


# Applied only when it caps: the default probes nothing until a product needs it.
_cap = _environment_gemm_cap()
if _cap != _GEMM_PATHS[-1]:
    limit_gemm_path(_cap)
del _cap


def get_include():
    """Return the directory to add to a C++ include path for NumericAL's headers.

    ``#include <numerical/numerical.hpp>`` resolves against this directory. NumericAL is
    header-only and shipped inside the installed package, so a project can
    compile against it located through its Python install. NumericAL is a leaf
    library — it needs no other include paths.

    Returns:
        str: Absolute path to NumericAL's include directory.
    """
    here = _os.path.dirname(_os.path.abspath(__file__))
    packaged = _os.path.join(here, "include")
    if _os.path.isdir(_os.path.join(packaged, "numerical")):
        return packaged
    return _os.path.normpath(_os.path.join(here, _os.pardir, _os.pardir, "include"))


def blas_library():
    """Return the path of the numerical_blas shared library, or ``None`` where the package carries none.

    The library exports 39 BLAS and LAPACK routines under their Fortran names and calling convention:
    ``?gemm_``, ``?gesv_``, ``?getrf_``, ``?getrs_``, ``?potrf_`` and ``?potrs_`` in every precision (``s``, ``d``,
    ``c``, ``z``), ``sdot_``, ``ddot_``, ``cdotu_``, ``cdotc_``, ``zdotu_`` and ``zdotc_``, and ``dsyrk_``,
    ``dgemv_``, ``dtrsm_``, ``dsyevd_``, ``dgesdd_``, ``dgeqrf_``, ``dorgqr_``, ``dgelsd_`` and ``dgeev_`` in double
    precision; each with 32-bit integers (LP64), and again under its name ending in ``64_`` with 64-bit ones (ILP64,
    as OpenBLAS's ``64_`` build), which Julia's libblastrampoline takes::

        BLAS.lbt_forward(path; clear=false, suffix_hint="64_") Preloaded ahead of the system's BLAS and LAPACK, it answers those calls and leaves the others to them::

        LD_PRELOAD=$(python -c "import numerical; print(numerical.blas_library())") python script.py

    A program reaches it only through a BLAS it calls by those names: a NumPy or SciPy installed by the system's
    package manager or switched by FlexiBLAS, not the wheels from PyPI, whose OpenBLAS is renamed inside them.
    Importing :mod:`numerical` never loads it. It is built for Linux alone; elsewhere this returns ``None``.
    Experimental: its routines are checked by NumPy's and SciPy's linear algebra suites, and more may join.

    Returns:
        str | None: Absolute path to ``libnumerical_blas.so``, or ``None``.
    """
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "lib", "libnumerical_blas.so")
    return path if _os.path.isfile(path) else None


def get_config():
    """Return metadata for embedding the C++ library.

    Returns:
        dict: ``version`` (str), ``include`` (str, see :func:`get_include`), ``cxx_standard`` (str),
        ``gemm_path`` (str, the kernel float64 matrix products run on, see :func:`gemm_path`) and ``threads`` (int).
    """
    return {
        "version": __version__,
        "include": get_include(),
        "cxx_standard": "c++20",
        "gemm_path": gemm_path(),
        "threads": get_num_threads(),
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


_DTYPES = ("float64", "complex128", "float32", "complex64")
_SINGLE = ("float32", "complex64")


def _promoted(dtypes):
    """NumPy's result dtype of tensors of `dtypes`: complex where one is, single precision only where all are."""
    single = all(dtype in _SINGLE for dtype in dtypes)
    if any(dtype.startswith("complex") for dtype in dtypes):
        return "complex64" if single else "complex128"
    return "float32" if single else "float64"


def _weak(value):
    """Whether `value` is a Python number, which NumPy 2 promotes as weak: it keeps the tensor's precision."""
    return type(value) in _NATIVE_NUMBERS


def _beside_weak(dtype, value):
    """The dtype of a tensor of `dtype` beside the Python number `value`: its own, complex for a complex `value`."""
    if isinstance(value, complex) and not dtype.startswith("complex"):
        return "complex64" if dtype in _SINGLE else "complex128"
    return dtype


def _linalg(native, arrays=None):
    """`native` taking array-likes: every argument that is not a :class:`Tensor` or a string goes through
    :func:`asarray` (a NumPy array, a nested list, a number), or only the first `arrays` of them, the options after
    passed as they are; the operands are brought to one dtype as NumPy promotes them (complex where one is, single
    precision only where all are); and its failures are :class:`LinAlgError`."""

    @_functools.wraps(native)
    def call(*args):
        count = len(args) if arrays is None else arrays
        operands = [arg if isinstance(arg, (Tensor, str)) else asarray(arg) for arg in args[:count]]
        tensors = [operand for operand in operands if isinstance(operand, Tensor)]
        if len({tensor.dtype for tensor in tensors}) > 1:
            dtype = _promoted([tensor.dtype for tensor in tensors])
            operands = [asarray(operand, dtype) if isinstance(operand, Tensor) else operand for operand in operands]
        try:
            return native(*operands, *args[count:])
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
    if any(k is Ellipsis for k in key) or any(k is None for k in key):
        return _getitem_extended(self, key)
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
        if isinstance(index, bool) or (type(index).__module__ == "numpy" and type(index).__name__ in ("bool", "bool_")):
            # operator.index(True) is 1 (and numpy.True_ is 1 before NumPy 2.5), but NumPy reads a boolean as a
            # mask, which a Tensor does not take.
            raise IndexError("boolean indexing is not supported: index with integers or slices")
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


def _getitem_extended(self, key):
    """``t[..., i]`` and ``t[None, i]``: an ``...`` stands for as many whole axes as the other indices leave (once), a
    ``None`` adds an axis of extent 1 where it stands, as NumPy's basic indexing."""
    if _builtins.sum(1 for k in key if k is Ellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    named = _builtins.sum(1 for k in key if k is not None and k is not Ellipsis)
    if named > len(self.shape):
        raise IndexError(f"too many indices for a tensor of rank {len(self.shape)}: {named} were given")
    expanded = []
    for k in key:
        expanded += [slice(None)] * (len(self.shape) - named) if k is Ellipsis else [k]
    plain = tuple(k for k in expanded if k is not None)
    picked = _getitem(self, plain) if plain else self
    if not any(k is None for k in expanded):
        return picked
    out, axis = [], 0  # the result's extents: a slice keeps its axis, an integer drops one, a None adds 1
    picked_shape = picked.shape if isinstance(picked, Tensor) else ()
    for k in expanded:
        if k is None:
            out.append(1)
        elif isinstance(k, slice):
            out.append(picked_shape[axis])
            axis += 1
    out += list(picked_shape[axis:])  # the axes no index named
    if not isinstance(picked, Tensor):
        picked = asarray([picked], self.dtype)
    return reshape(picked, tuple(out))


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


# The labels einsum takes: the 52 ASCII letters, as the extension's.
_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_REVERSALS = [f"{_LETTERS[:rank]}->{_LETTERS[:rank][::-1]}" for rank in range(len(_LETTERS) + 1)]


def _labels(count, what):
    """The first `count` labels, distinct; ValueError past the 52 an einsum spec has."""
    if count > len(_LETTERS):
        raise ValueError(f"{what}: at most {len(_LETTERS)} axes, the labels an einsum spec has ({count} asked)")
    return _LETTERS[:count]


def _transpose(self):
    """The tensor with its axes reversed: a matrix's transpose, as NumPy's ``.T``. The spec is built once per rank and
    the extension called directly: a transposition has no error for the wrapper to translate."""
    rank = len(self.shape)
    if rank >= len(_REVERSALS):
        _labels(rank, "T")  # raises
    return _numerical.einsum(_REVERSALS[rank], self)


_MATMUL = {(1, 1): "i,i->", (2, 1): "ij,j->i", (1, 2): "i,ij->j", (2, 2): "ij,jk->ik"}


def _matmul_shapes(a_shape, b_shape):
    """Refuses the product of operands of shapes `a_shape` and `b_shape` whose inner dimensions differ, naming both: a's
    last against b's first (a vector) or second to last."""
    a_shape, b_shape = tuple(a_shape), tuple(b_shape)
    if not a_shape or not b_shape:
        raise LinAlgError("matmul: a rank-0 operand has no matrix to multiply (multiply it with *)")
    inner = b_shape[0] if len(b_shape) == 1 else b_shape[-2]
    if a_shape[-1] != inner:
        def text(shape):
            return " × ".join(str(d) for d in shape)
        across = "rows" if len(b_shape) > 1 else "entries"
        raise LinAlgError(f"matmul: a is {text(a_shape)} and b is {text(b_shape)}: a's {a_shape[-1]} columns do not match "
                          f"b's {inner} {across}")


def _matmul(self, other):
    """``a @ b`` as NumPy's matmul: vectors and matrices, and stacks of matrices along leading axes broadcast against
    each other (a number for two vectors); the dtypes promoted as NumPy promotes them."""
    other = other if isinstance(other, Tensor) else asarray(other)
    _matmul_shapes(self.shape, other.shape)
    dtype = _promoted((self.dtype, other.dtype))
    a, b = (t if t.dtype == dtype else asarray(t, dtype=dtype) for t in (self, other))
    spec = _MATMUL.get((len(a.shape), len(b.shape)))
    if spec is None:
        return _matmul_nd(a, b)
    try:  # two Tensors of one dtype already: the extension itself, its failures LinAlgError as _einsum's
        product = _numerical.einsum(spec, a, b)
    except error as caught:
        raise LinAlgError(*caught.args) from None
    return product[()] if not product.shape else product  # two vectors give a number, as NumPy's


def matmul(x1, x2, /):
    """The matrix product of `x1` and `x2`, as ``numpy.matmul`` (and ``x1 @ x2``): stacks of matrices along leading axes
    broadcast against each other, a vector a row first and a column second, two vectors their dot product."""
    try:
        product = _numerical._borrowed_product(x1, x2)  # Tensors and NumPy arrays read where they lie, when they can be
    except error as caught:  # shapes that do not fit, said with both (checked here only, off the products' path)
        shapes = (getattr(x1, "shape", None), getattr(x2, "shape", None))
        if shapes[0] is not None and shapes[1] is not None:
            _matmul_shapes(*shapes)
        raise LinAlgError(*caught.args) from None
    return _tensor(x1)._matmul(x2) if product is NotImplemented else product  # else as ``@``


def _rmatmul(self, other):
    return _matmul(asarray(other), self)


def _flat(nested):
    """The numbers of a nested list (as ``tolist`` gives it) in C order."""
    if not isinstance(nested, list):
        return [nested]
    return [value for item in nested for value in _flat(item)]


def _bool(self):
    """The truth value of a one-element tensor, as NumPy's; any other size is ambiguous and raises."""
    if self.size != 1:
        kind = "an empty tensor" if self.size == 0 else "a tensor with more than one element"
        raise ValueError(f"the truth value of {kind} is ambiguous: test .size, or compare whole with "
                         "numerical.array_equal")
    return bool(_flat(self.tolist())[0])


def _scalar(convert):
    def method(self):
        if self.shape:
            raise TypeError("only rank-0 tensors can be converted to Python scalars")
        return convert(self.tolist())
    return method


def _uncomparable(self, other):
    raise TypeError("tensors have no element-wise comparison yet: numerical.array_equal(a, b) compares two "
                    "tensors whole")


_REPR_EDGE = 3        # elements kept at each end of a summarised axis
_REPR_THRESHOLD = 1000  # a larger tensor is summarised, as NumPy's default


def _summary(t, prefix, shape):
    """A nested list of t's elements with every long axis cut to its ends around an ``...``."""
    if not shape:
        return t[prefix] if prefix else t[()]
    extent = shape[0]
    if extent > 2 * _REPR_EDGE:
        indices = list(range(_REPR_EDGE)) + [None] + list(range(extent - _REPR_EDGE, extent))
    else:
        indices = list(range(extent))
    return [Ellipsis if i is None else _summary(t, prefix + (i,), shape[1:]) for i in indices]


def _format(nested, indent):
    if not isinstance(nested, list):
        return "..." if nested is Ellipsis else repr(nested)
    if not any(isinstance(item, list) for item in nested):
        return "[" + ", ".join(_format(item, indent) for item in nested) + "]"
    return "[" + (",\n" + " " * (indent + 1)).join(_format(item, indent + 1) for item in nested) + "]"


def _repr(self):
    """``numerical.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=float64)``, one row per line; past 1000 elements
    each long axis shows its first and last three entries around ``...``, as NumPy's repr."""
    head = "numerical.Tensor("
    values = self.tolist() if self.size <= _REPR_THRESHOLD else _summary(self, (), self.shape)
    return f"{head}{_format(values, len(head))}, dtype={self.dtype})"


def array_equal(a, b):
    """``True`` when `a` and `b` (Tensors or array-likes) have the same shape and equal elements, as
    ``numpy.array_equal``: a NaN equals nothing, itself included."""
    a, b = _same_dtype(a, b)
    return a.shape == b.shape and _numerical._equal(a, b)


Tensor.__getitem__ = _getitem


def _setitem(self, key, value):
    """A Tensor is immutable (its buffer may be shared with others and with NumPy, without a copy): writing into one
    raises, saying how to build the changed one instead."""
    del key, value
    raise TypeError("a numerical.Tensor is immutable (its data may be shared without a copy): build a new one, for "
                    "instance from a NumPy array (numpy.array(t), changed, then numerical.asarray), or by an expression")


Tensor.__setitem__ = _setitem
Tensor.__len__ = _len
Tensor.__iter__ = _iter
Tensor.T = property(_transpose)
_numerical._set_matmul_fallback(_matmul)
Tensor.__matmul__ = Tensor._matmul  # a method of the type: no Python frame for two tensors
Tensor.__rmatmul__ = _rmatmul
Tensor.__bool__ = _bool
Tensor.__float__ = _scalar(float)
Tensor.__complex__ = _scalar(complex)
Tensor.__int__ = _scalar(int)
Tensor.__repr__ = _repr
for _name in ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"):
    setattr(Tensor, _name, _uncomparable)
Tensor.__hash__ = None  # unhashable, as a NumPy array: equality will be element-wise

_cholesky = _linalg(_numerical.cholesky, arrays=1)
_determinant = _linalg(_numerical.determinant)
_einsum = _linalg(_numerical.einsum)
_inverse = _linalg(_numerical.inverse)
_lstsq = _linalg(_numerical.lstsq, arrays=2)
_eigvalsh = _linalg(_numerical.eigvalsh, arrays=1)
_matrix_rank = _linalg(_numerical.matrix_rank, arrays=1)
_matrix_ranks = _linalg(_numerical._matrix_ranks, arrays=1)
_pinv = _linalg(_numerical.pinv, arrays=1)
_qr = _linalg(_numerical.qr, arrays=1)
_qr_r = _linalg(_numerical.qr_r, arrays=1)
_slogdet = _linalg(_numerical.slogdet)
_solve = _linalg(_numerical.solve)


# The public functions over the native ones, with the signatures their stubs give, so that stubtest holds the stubs to
# them (a native function taking *args has none to compare); each keeps the native one's documentation.
def einsum(subscripts, /, *operands):
    if len(operands) == 1 and isinstance(subscripts, str):
        summed = _numerical._borrowed_einsum(subscripts, operands[0])  # a run of axes summed where the operand lies
        if summed is not NotImplemented:
            return summed
    if len(operands) == 2 and isinstance(subscripts, str):
        paired = _numerical._borrowed_einsum_pair(subscripts, *operands)  # both read where they lie
        if paired is not NotImplemented:
            return paired
    return _einsum(subscripts, *operands)


def _borrowed(name, convert, *operands):
    """`name` of a NumPy array read where it lies, its one copy the one the factorisation takes (where the converted
    Tensor was a copy before it), else `convert`'s path for anything else: the same answer either way."""
    try:
        result = _numerical._borrowed_linalg(name, *operands)
    except error as caught:
        raise LinAlgError(*caught.args) from None
    return convert(*operands) if result is NotImplemented else result


def solve(a, b, /):
    return _borrowed("solve", _solve, a, b)


def determinant(a, /):
    return _borrowed("determinant", _determinant, a)


def slogdet(a, /):
    return _borrowed("slogdet", _slogdet, a)


def inverse(a, /):
    return _borrowed("inverse", _inverse, a)


for _public, _native in ((einsum, _numerical.einsum), (solve, _numerical.solve), (determinant, _numerical.determinant),
                         (slogdet, _numerical.slogdet), (inverse, _numerical.inverse)):
    _public.__doc__ = _native.__doc__
del _public, _native

det = determinant
inv = inverse


_expm = _linalg(_numerical.expm, arrays=1)
_expm_frechet = _linalg(_numerical.expm_frechet, arrays=2)
_expm_cond = _linalg(_numerical.expm_cond, arrays=1)
_funm_plan = _linalg(_numerical.funm_plan, arrays=1)
_logm = _linalg(_numerical.logm, arrays=1)
_fractional_matrix_power = _linalg(_numerical.fractional_matrix_power, arrays=1)
_qr_pivoted = _linalg(_numerical.qr_pivoted, arrays=1)
_ldl = _linalg(_numerical.ldl, arrays=1)
_solve_banded = _linalg(_numerical.solve_banded, arrays=2)
_solveh_banded = _linalg(_numerical.solveh_banded, arrays=2)
_solve_toeplitz = _linalg(_numerical.solve_toeplitz, arrays=3)
_solve_circulant = _linalg(_numerical.solve_circulant, arrays=2)
_solve_sylvester = _linalg(_numerical.solve_sylvester, arrays=3)
_riccati = _linalg(_numerical.riccati, arrays=6)
_sqrtm = _linalg(_numerical.sqrtm, arrays=1)


def solve_banded(l_and_u, ab, b, overwrite_ab=False, overwrite_b=False, check_finite=True):
    """The solution `x` of ``a x = b`` for a banded `a` of `l` subdiagonals and `u` superdiagonals, as
    ``scipy.linalg.solve_banded``: `ab` its diagonal-ordered form, ``(l + u + 1, n)``, ``ab[u + i - j, j] = a[i, j]``
    (its corners not read); `b` a vector or a matrix of right-hand sides. The LU factorisation with partial pivoting
    in LAPACK's band storage (``gbsv``), ``O(n l (l + u))`` operations; for ``l = u = 1``, ``gtsv``, as SciPy. `overwrite_ab`, `overwrite_b` and
    `check_finite` are accepted and not needed: an infinite or NaN entry raises always.

    Raises:
        ValueError: if `ab` has not ``l + u + 1`` rows or `b` not as many rows as `ab` has columns.
        LinAlgError: if `a` is singular or an entry is infinite or NaN.
    """
    del overwrite_ab, overwrite_b, check_finite
    lower, upper = (_operator.index(count) for count in l_and_u)
    ab = ab if isinstance(ab, Tensor) else asarray(ab)
    b = b if isinstance(b, Tensor) else asarray(b)
    if lower < 0 or upper < 0 or ab.ndim != 2 or ab.shape[0] != lower + upper + 1:
        raise ValueError(f"invalid values for the number of lower and upper diagonals: l+u+1 ({lower + upper + 1}) "
                         f"does not equal ab.shape[0] ({ab.shape[0] if ab.ndim else 0})")
    if b.ndim not in (1, 2) or b.shape[0] != ab.shape[1]:
        raise ValueError("shapes of ab and b are not compatible.")
    return _solve_banded(ab, b, lower, upper)


def solveh_banded(ab, b, overwrite_ab=False, overwrite_b=False, lower=False, check_finite=True):
    """The solution `x` of ``a x = b`` for a Hermitian positive-definite banded `a`, as
    ``scipy.linalg.solveh_banded``: `ab` its upper diagonal-ordered form, ``(u + 1, n)``, ``ab[u + i - j, j] = a[i,
    j]`` for ``i <= j``, or with `lower` its lower one, ``ab[i - j, j] = a[i, j]`` for ``i >= j`` (the diagonal's
    imaginary part and the corners not read). The band's Cholesky factor (LAPACK's ``pbsv``); for one off-diagonal,
    ``L D Lᴴ`` (``ptsv``), as SciPy. `overwrite_ab`,
    `overwrite_b` and `check_finite` are accepted and not needed.

    Raises:
        ValueError: if `b` has not as many rows as `ab` has columns.
        LinAlgError: if `a` is not positive definite or an entry is infinite or NaN.
    """
    del overwrite_ab, overwrite_b, check_finite
    ab = ab if isinstance(ab, Tensor) else asarray(ab)
    b = b if isinstance(b, Tensor) else asarray(b)
    if ab.ndim != 2 or ab.shape[0] == 0 or b.ndim not in (1, 2) or b.shape[0] != ab.shape[1]:
        raise ValueError("shapes of ab and b are not compatible.")
    return _solveh_banded(ab, b, bool(lower))


def solve_toeplitz(c_or_cr, b, check_finite=True):
    """The solution `x` of ``T x = b`` for a Toeplitz `T`, as ``scipy.linalg.solve_toeplitz``: `c_or_cr` its first
    column `c`, or the pair ``(c, r)`` of its first column and first row (``r[0]`` not read; ``r = c.conj()`` by
    default, a Hermitian `T`); `b` a vector or a matrix of right-hand sides. Levinson's recursion, ``O(n²)`` a column
    where a dense solve takes ``n³ / 3``: every leading principal minor must be nonsingular, and it may lose accuracy
    unless `T` is well-conditioned and symmetric positive definite, as SciPy's. `check_finite` is accepted and not
    needed.

    Raises:
        ValueError: if `c` and `r` differ in length or `b` has not as many rows.
        LinAlgError: if a leading principal minor is singular or an entry is infinite or NaN.
    """
    del check_finite
    if isinstance(c_or_cr, tuple):
        c, r = c_or_cr
        c = c if isinstance(c, Tensor) else asarray(c)
        r = r if isinstance(r, Tensor) else asarray(r)
    else:
        c = c_or_cr if isinstance(c_or_cr, Tensor) else asarray(c_or_cr)
        r = conj(c) if c.dtype.startswith("complex") else c
    b = b if isinstance(b, Tensor) else asarray(b)
    c, r = c.reshape((c.size,)), r.reshape((r.size,))
    if r.shape != c.shape or b.ndim not in (1, 2) or b.shape[0] != c.shape[0]:
        raise ValueError("Incompatible c, r and b shapes.")
    return _solve_toeplitz(c, r, b)


def solve_circulant(c, b, singular="raise", tol=None, caxis=-1, baxis=0, outaxis=0):
    """The solution `x` of ``C x = b`` for the circulant `C` of first column `c`, as ``scipy.linalg.solve_circulant``:
    the Fourier transform diagonalises it, ``x = ifft(fft(b) / fft(c))``, ``O(n log n)`` a column. An eigenvalue
    ``fft(c)`` of modulus at most `tol` (by default ``n eps max |fft(c)|``) makes `C` near singular: ``singular="raise"``
    refuses it, ``"lstsq"`` leaves those terms out (the minimum-norm least-squares solution). Real `c` and `b` give a
    real `x`. `c` a vector and `b` a vector or a matrix of right-hand sides by rows (SciPy's default axes alone).

    Raises:
        ValueError: for another `singular`, axes or shapes.
        LinAlgError: if `C` is near singular under ``"raise"``, or an entry is infinite or NaN.
    """
    if singular not in ("raise", "lstsq"):
        raise ValueError("singular must be 'raise' or 'lstsq'")
    if caxis not in (-1, 0) or baxis != 0 or outaxis != 0:
        raise ValueError("solve_circulant takes c's only axis, b's first and the solution's first")
    c = c if isinstance(c, Tensor) else asarray(c)
    b = b if isinstance(b, Tensor) else asarray(b)
    if c.ndim != 1 or b.ndim not in (1, 2) or b.shape[0] != c.shape[0]:
        raise ValueError(f"Shapes of c {c.shape} and b {b.shape} are incompatible")
    return _solve_circulant(c, b, singular == "lstsq", -1.0 if tol is None else float(tol))


def _vectors(name, *values):
    """`values` as vectors (a number as one entry) of one dtype, as NumPy promotes them."""
    tensors = [value if isinstance(value, Tensor) else asarray(value) for value in values]
    if any(tensor.ndim > 1 for tensor in tensors):
        raise ValueError(f"{name} takes vectors: SciPy's batches of them are not taken")
    tensors = [tensor.reshape((tensor.size,)) for tensor in tensors]
    dtype = _promoted([tensor.dtype for tensor in tensors])
    return [tensor if tensor.dtype == dtype else asarray(tensor, dtype) for tensor in tensors]


def toeplitz(c, r=None):
    """The Toeplitz matrix of first column `c` and first row `r`, as ``scipy.linalg.toeplitz``: ``T[i, j] = c[i - j]``
    for ``i >= j``, ``r[j - i]`` above (``r[0]`` not read); ``r = conj(c)`` by default, a Hermitian matrix. Integers
    and booleans as ``float64``."""
    if r is None:
        (c,) = _vectors("toeplitz", c)
        r = conj(c) if c.dtype.startswith("complex") else c
    c, r = _vectors("toeplitz", c, r)
    return _numerical._structured(c, r, 0)


def hankel(c, r=None):
    """The Hankel matrix of first column `c` and last row `r`, as ``scipy.linalg.hankel``: ``H[i, j] = c[i + j]``
    while ``i + j < len(c)``, ``r[i + j - len(c) + 1]`` past it (``r[0]`` not read); `r` zeros by default."""
    if r is None:
        (c,) = _vectors("hankel", c)
        r = zeros(c.shape, c.dtype)
    c, r = _vectors("hankel", c, r)
    return _numerical._structured(c, r, 1)


def circulant(c):
    """The circulant matrix of first column `c`, as ``scipy.linalg.circulant``: ``C[i, j] = c[(i - j) % n]``."""
    (c,) = _vectors("circulant", c)
    return _numerical._structured(c, c, 2)


def block_diag(*arrs):
    """The block diagonal matrix of `arrs`, in order, as ``scipy.linalg.block_diag``: each a matrix (a vector one row,
    a number one entry), zeros off the blocks, one dtype as NumPy promotes them; none gives a ``1 × 0`` matrix."""
    tensors = [arr if isinstance(arr, Tensor) else asarray(arr) for arr in (arrs or ([],))]
    if any(tensor.ndim > 2 for tensor in tensors):
        raise ValueError("block_diag takes matrices: SciPy's batches of them are not taken")
    tensors = [tensor.reshape((1, tensor.size)) if tensor.ndim < 2 else tensor for tensor in tensors]
    dtype = _promoted([tensor.dtype for tensor in tensors])
    return _numerical._block_diag([tensor if tensor.dtype == dtype else asarray(tensor, dtype) for tensor in tensors])


def expm(A):
    """The matrix exponential ``e^A``, as ``scipy.linalg.expm``: of a square matrix, or of each matrix of a stack
    (``(..., n, n)``); integers and booleans as ``float64``. Scaling and squaring of a Padé approximant of degree 3 to
    13, as Al-Mohy and Higham's algorithm (2009) that SciPy follows: the same values to rounding, not to the bit. An
    infinite or NaN entry raises, where SciPy returns NaNs.

    Raises:
        LinAlgError: if `A` is not square (or a stack of square matrices) or holds an infinite or NaN entry.
    """
    return _expm(A)


def expm_frechet(A, E, method=None, compute_expm=True, check_finite=True):
    """The Fréchet derivative of the matrix exponential at `A` along `E`, as ``scipy.linalg.expm_frechet``:
    ``(expm(A), L)``, or `L` alone where `compute_expm` is false. ``method=None`` or ``"SPS"`` is Al-Mohy and Higham's
    algorithm 6.4 (2009) as SciPy's: the Padé approximant of degree 3, 5, 7 or 9 where ``‖A‖₁`` is within its bound,
    else of degree 13 on ``2⁻ˢ A``, then `s` squarings of the pair; ``"blockEnlarge"`` the exponential of the block
    matrix ``[[A, E], [0, A]]``, whose blocks are ``expm(A)`` and `L`. Matrices only (SciPy's batches are not taken);
    `check_finite` is accepted and not needed.

    Raises:
        LinAlgError: if `A` or `E` is not square, their shapes differ, or an entry is infinite or NaN.
        ValueError: for another `method`.
    """
    del check_finite
    if method not in (None, "SPS", "blockEnlarge"):
        raise ValueError(f"Unknown implementation {method}")
    x, l = _expm_frechet(A, E, method == "blockEnlarge")
    return (x, l) if compute_expm else l


def funm(A, func, disp=True):
    """A function of a square matrix, as ``scipy.linalg.funm``: the Schur form ``A = Z T Zᴴ`` (a real matrix's real
    form made complex), `func` called once with `T`'s diagonal (a NumPy array where NumPy is installed, else a
    :class:`Tensor`; it returns one value per entry), then Parlett's recurrence for the rest of ``f(T)`` and
    ``Z f(T) Zᴴ``; real where `A` is and the imaginary parts are negligible. Where eigenvalues gather (within 1e-3 of
    the largest modulus, or of 1), Davies and Higham's Schur–Parlett algorithm: the Schur form reordered by clusters,
    each cluster's block by Cauchy's integral on a circle about it (`func` then also gets those circles' points, after
    the eigenvalues: it must be analytic on their disks), the rest by the block recurrence. A Hermitian `A`:
    ``V f(Λ) Vᴴ`` from its eigendecomposition. With ``disp=True`` (the default) a likely inaccurate result is reported
    on standard output, as SciPy's; with ``disp=False`` ``(F, err)`` is returned, `err` the estimate of
    ``‖error‖₁ / ‖A‖₁``.

    Raises:
        LinAlgError: if `A` is not square or holds an infinite or NaN entry, or `func` does not return one value per
            point.
    """
    A = A if isinstance(A, Tensor) else asarray(A)
    t, z, points, hermitian = _funm_plan(A)
    arguments = [complex(x) for x in _flat(points.tolist())] if points.size else []
    try:
        import numpy as _np
    except ImportError:  # pragma: no cover
        argument = asarray(arguments, "complex128")
    else:
        argument = _np.array(arguments, dtype=complex)
    values = [complex(x) for x in _flat(asarray(func(argument)).tolist())] if arguments else []
    values = asarray(values, "complex128")
    try:
        f, err = _numerical.funm_finish(t, z, points, values, hermitian, not A.dtype.startswith("complex"))
    except error as caught:
        raise LinAlgError(*caught.args) from None
    if A.dtype in ("float32", "complex64"):
        f = asarray(f, "float32" if f.dtype == "float64" else "complex64")
    if disp:
        if err > 1000 * 2.220446049250313e-16:
            print("funm result may be inaccurate, approximate err =", err)
        return f
    return f, err


class LogmRankWarning(UserWarning):
    """A warning of :func:`logm` about a singular or nearly singular matrix, as SciPy's."""


class LogmExactlySingularWarning(LogmRankWarning):
    """:func:`logm` met a zero eigenvalue (no logarithm exists; it was taken as 1e-20), as SciPy's."""


class LogmNearlySingularWarning(LogmRankWarning):
    """:func:`logm` met an eigenvalue below 1e-20 in modulus, as SciPy's."""


def logm(A):
    """The principal matrix logarithm, as ``scipy.linalg.logm``: an upper triangular `A` directly, else its Schur form
    (a real matrix's real form made complex) and ``Z log(T) Zᴴ``, ``log T`` by Al-Mohy and Higham's inverse scaling
    and squaring (2012); the norms that choose the square roots and the Padé degree are exact where SciPy estimates
    them (``onenormest``). Real where `A` is and the imaginary parts are negligible; a stack's matrices each (complex
    for all where any is). Warns as SciPy does: :class:`LogmExactlySingularWarning` where an eigenvalue is 0 (taken as
    1e-20), :class:`LogmNearlySingularWarning` where one is below 1e-20, and a ``RuntimeWarning`` where
    ``‖expm(F) − A‖₁ / ‖A‖₁`` is not below 1000 ε.

    Raises:
        LinAlgError: if `A` is not square (or a stack of square matrices) or holds an infinite or NaN entry.
    """
    A = A if isinstance(A, Tensor) else asarray(A)
    if A.ndim > 2:
        parts = [logm(A[i]) for i in range(A.shape[0])]
        dtype = _promoted([part.dtype for part in parts]) if parts else A.dtype
        return asarray([part.tolist() for part in parts], dtype) if parts else A
    f, err, exact, near = _logm(A)
    if exact:
        _warnings.warn("The logm input matrix is exactly singular.", LogmExactlySingularWarning, stacklevel=2)
    elif near:
        _warnings.warn("The logm input matrix may be nearly singular.", LogmNearlySingularWarning, stacklevel=2)
    if not (err < 1000 * 2.220446049250313e-16):  # NaN too
        _warnings.warn(f"logm result may be inaccurate, approximate err = {err}", RuntimeWarning, stacklevel=2)
    return f


def fractional_matrix_power(A, t):
    """`A` to the real power `t`, as ``scipy.linalg.fractional_matrix_power``: a whole `t` by repeated squaring; else
    ``A^a A^b``, `a` the floor or the ceiling of `t` as `A`'s condition chooses it, ``A^b`` (``|b| < 1``) by Higham and
    Lin's Schur–Padé algorithm (2013) on the Schur form, the diagonal and superdiagonal written exactly after each
    squaring (SciPy writes them before the first only). Where `A` is singular, a positive `t` falls back to
    :func:`funm`, as SciPy's; a negative one raises (SciPy answers a matrix of NaN). Real where `A` is and every
    imaginary part is 0, else complex.

    Raises:
        LinAlgError: if `A` is not square, holds an infinite or NaN entry, or is singular for a negative `t`.
        ValueError: if `t` is infinite or NaN.
    """
    t = float(t)
    if not _math.isfinite(t):
        raise ValueError("fractional_matrix_power: the power must be finite")
    return _fractional_matrix_power(A, t)


def expm_cond(A, check_finite=True):
    """The relative condition number of the matrix exponential in the Frobenius norm, as ``scipy.linalg.expm_cond``:
    ``‖K‖₂ ‖A‖_F / ‖expm(A)‖_F``, `K` the ``n² × n²`` Kronecker form of the Fréchet derivative (one
    :func:`expm_frechet` a column). Matrices only; `check_finite` is accepted and not needed.

    Raises:
        LinAlgError: if `A` is not square or holds an infinite or NaN entry.
    """
    del check_finite
    return _expm_cond(A)
_eigh = _linalg(_numerical.eigh, arrays=1)
_eigh_pencil = _linalg(_numerical.eigh_pencil, arrays=2)
_eigvalsh_pencil = _linalg(_numerical.eigvalsh_pencil, arrays=2)
_svd = _linalg(_numerical.svd, arrays=1)
_singular_values = _linalg(_numerical._singular_values)
_svd_hermitian = _linalg(_numerical._svd_hermitian)
_matmul_nd = _linalg(_numerical._matmul_nd)
_eig = _linalg(_numerical._eig)
_schur = _linalg(_numerical.schur, arrays=1)
_schur_sorted = _linalg(_numerical.schur_sorted, arrays=1)
_eigvals = _linalg(_numerical._eigvals)
_QR_MODES = ("reduced", "complete", "r")
_EIGH_METHODS = ("divide_and_conquer", "tridiagonal_ql", "jacobi")
_SVD_METHODS = ("divide_and_conquer", "bidiagonal_qr", "jacobi")


def sqrtm(A):
    """The principal square root `X` of a square matrix (``X @ X = A``, its eigenvalues of nonnegative real part), as
    ``scipy.linalg.sqrtm``: of each matrix of a stack (``(..., n, n)``); integers and booleans as ``float64``. By the
    Schur method, in real arithmetic for a real matrix (Higham, 1987), whose root is real unless it has a real
    eigenvalue below 0, then complex (for a whole stack); where there is no square root (a zero eigenvalue in a Jordan
    block), the entries that would need one infinite, as SciPy's. SciPy's deprecated `disp` and `blocksize` are not
    taken.

    Raises:
        LinAlgError: if `A` is not square (or a stack of square matrices) or holds an infinite or NaN entry.
    """
    return _sqrtm(A)


def polar(a, side="right"):
    """The polar decomposition of a matrix, as ``scipy.linalg.polar``: ``(u, p)`` with ``a = u p`` (``side="right"``)
    or ``a = p u`` (``"left"``), `u` with orthonormal columns (rows, for a wide `a`), `p` Hermitian positive
    semidefinite: from the singular value decomposition ``a = W Σ Vᴴ`` (``full_matrices=False``), ``u = W Vᴴ`` and
    ``p = V Σ Vᴴ`` (right) or ``W Σ Wᴴ`` (left), as SciPy computes them.

    Raises:
        ValueError: if `a` is not a matrix or `side` is neither ``"right"`` nor ``"left"``.
        LinAlgError: if `a` holds an infinite or NaN entry.
    """
    if side not in ("right", "left"):
        raise ValueError("`side` must be either 'right' or 'left'")
    a = a if isinstance(a, Tensor) else asarray(a)
    if a.ndim != 2:
        raise ValueError("`a` must be a 2-D array.")
    w, s, vh = svd(a, full_matrices=False)
    u = w @ vh
    if side == "right":
        return u, (_adjoint(vh) * s) @ vh
    return u, (w * s) @ _adjoint(w)


def solve_sylvester(a, b, q):
    """The solution `X` of the Sylvester equation ``A X + X B = Q``, as ``scipy.linalg.solve_sylvester``: Bartels and
    Stewart's method, the Schur forms of `a` and `b` and the triangular equation between them solved by blocks
    (LAPACK's ``trsyl``); complex where an operand is. Where `a` and ``-b`` share an eigenvalue there is no unique
    solution, and the answer grows as LAPACK's does.

    Raises:
        LinAlgError: if `a` or `b` is not square, `q` is not ``a.shape[0] × b.shape[0]``, or an entry is infinite or
            NaN.
    """
    return _solve_sylvester(a, b, q)


def _adjoint(a):
    """`a`'s conjugate transpose (its transpose where real)."""
    return conj(a.T) if a.dtype.startswith("complex") else a.T


def solve_continuous_lyapunov(a, q):
    """The solution `X` of ``A X + X Aᴴ = Q``, as ``scipy.linalg.solve_continuous_lyapunov``: the Sylvester equation
    with ``b = a.conj().T`` (:func:`solve_sylvester`)."""
    a = a if isinstance(a, Tensor) else asarray(a)
    if a.ndim != 2:
        raise LinAlgError("solve_continuous_lyapunov: a must be a square matrix")
    return solve_sylvester(a, _adjoint(a), q)


solve_lyapunov = solve_continuous_lyapunov


def solve_discrete_lyapunov(a, q, method=None):
    """The solution `X` of ``A X Aᴴ − X + Q = 0``, as ``scipy.linalg.solve_discrete_lyapunov``: ``method="direct"``
    (by default below order 10, as SciPy's) solves the Kronecker system ``(I − A ⊗ conj(A)) vec(X) = vec(Q)``,
    ``"bilinear"`` (from order 10) the continuous equation the bilinear transformation gives, ``Bᴴ X + X B = −C`` for
    ``B = (Aᴴ − I)(Aᴴ + I)⁻¹`` and ``C = 2 (A + I)⁻¹ Q (Aᴴ + I)⁻¹``.

    Raises:
        LinAlgError: if `a` is not square or `q` not of its shape, an entry is infinite or NaN, or ``I − A ⊗ conj(A)``
            (direct) or ``A + I`` (bilinear) is singular.
        ValueError: for another `method`.
    """
    a = a if isinstance(a, Tensor) else asarray(a)
    q = q if isinstance(q, Tensor) else asarray(q)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or q.shape != a.shape:
        raise LinAlgError("solve_discrete_lyapunov: a must be square and q of its shape")
    n = a.shape[0]
    if method is None:
        method = "direct" if n < 10 else "bilinear"
    if method == "direct":
        kron = einsum("ij,kl->ikjl", a, conj(a) if a.dtype.startswith("complex") else a).reshape((n * n, n * n))
        return solve(eye(n * n, kron.dtype) - kron, q.reshape((n * n,))).reshape((n, n))
    if method == "bilinear":
        identity = eye(n, a.dtype)
        a_h_inverse = inverse(_adjoint(a) + identity)
        b = (_adjoint(a) - identity) @ a_h_inverse
        c = 2 * ((inverse(a + identity) @ q) @ a_h_inverse)
        return solve_continuous_lyapunov(_adjoint(b), -c)
    raise ValueError(f"Unknown solver {method}")


_MACHINE_EPSILON = {"float64": 2.220446049250313e-16, "float32": 1.1920928955078125e-07}  # complex's parts share them


def _single_hermitian(x, name):
    """A single-precision `q` or `r` checked Hermitian as SciPy checks it, at its own precision (``‖X − Xᴴ‖₁`` within
    100 float32 spacings of ``‖X‖₁``), then made exactly so, ``(X + Xᴴ) / 2``: the double solver its widened values go
    to checks at double's spacing, which float32 rounding passes."""
    size = float(norm(x, 1))
    spacing = 2.0 ** (_math.frexp(size)[1] - 24) if size > 0 else 2.0 ** -149
    if float(norm(x - _adjoint(x), 1)) > 100 * spacing:
        raise LinAlgError(f"Matrix {name} should be symmetric/hermitian.")
    return (x + _adjoint(x)) / 2


def _are(a, b, q, r, e, s, balanced, discrete):
    """The native Riccati solver: `e` and `s` absent passed as `a` with their flags clear; single-precision `q` and `r`
    checked Hermitian at their own precision."""
    a, b, q, r = (x if isinstance(x, Tensor) else asarray(x) for x in (a, b, q, r))
    given = [x if x is None or isinstance(x, Tensor) else asarray(x) for x in (e, s)]
    if all(x.dtype in ("float32", "complex64") for x in [a, b, q, r] + [g for g in given if g is not None]):
        q, r = _single_hermitian(q, "q"), _single_hermitian(r, "r")
    e, s = given
    flags = (1 if discrete else 0) | (0 if e is None else 2) | (0 if s is None else 4) | (8 if balanced else 0)
    return _riccati(a, b, q, r, a if e is None else e, a if s is None else s, flags)


def solve_continuous_are(a, b, q, r, e=None, s=None, balanced=True):
    """The stabilising solution `X` of the continuous algebraic Riccati equation
    ``Aᴴ X E + Eᴴ X A − (Eᴴ X B + S) R⁻¹ (Bᴴ X E + Sᴴ) + Q = 0`` (``E = I`` and ``S = 0`` when `e` and `s` are
    None), as ``scipy.linalg.solve_continuous_are``: the extended Hamiltonian pencil, balanced unless `balanced` is
    false, its stable deflating subspace from the generalized Schur form sorted with the left half-plane first, `R`
    never inverted. Complex where an operand is. With `e`, the balancing scales `e` as it scales the pencil (SciPy
    1.18 does not, and answers a wrong `X` or raises there when the balancing is not trivial: scipy/scipy#26555).

    Raises:
        LinAlgError: if a shape is not ``a`` m×m, ``b`` m×n, ``q`` m×m, ``r`` n×n, ``e`` m×m, ``s`` m×n (m, n ≥ 1),
            an entry is infinite or NaN, `q` or `r` is not Hermitian, `e` is numerically singular, there is no finite
            solution, or the pencil has eigenvalues too close to the imaginary axis.
    """
    return _are(a, b, q, r, e, s, balanced, False)


def solve_discrete_are(a, b, q, r, e=None, s=None, balanced=True):
    """The stabilising solution `X` of the discrete algebraic Riccati equation
    ``Aᴴ X A − Eᴴ X E − (Aᴴ X B + S) (R + Bᴴ X B)⁻¹ (Bᴴ X A + Sᴴ) + Q = 0`` (``E = I`` and ``S = 0`` when `e` and `s`
    are None), as ``scipy.linalg.solve_discrete_are``: the extended symplectic pencil, balanced unless `balanced` is
    false, its stable deflating subspace from the generalized Schur form sorted with the inside of the unit circle
    first, nothing inverted. Complex where an operand is; with `e`, the balancing scales it as
    :func:`solve_continuous_are`'s. A pencil with eigenvalues on the unit circle is refused also where SciPy, in real
    arithmetic with ``m = 1``, answers.

    Raises:
        LinAlgError: as :func:`solve_continuous_are`'s, the eigenvalues too close to the unit circle.
    """
    return _are(a, b, q, r, e, s, balanced, True)


def _upper(name, UPLO):
    """Whether NumPy's `UPLO` names the upper triangle: ``"L"`` or ``"U"``, in either case."""
    if isinstance(UPLO, str) and UPLO.upper() in ("L", "U"):
        return UPLO.upper() == "U"
    if UPLO in _EIGH_METHODS:
        raise TypeError(f"{name}: the second argument is UPLO, as in NumPy; give the method by name, "
                        f"{name}(a, method={UPLO!r})")
    raise ValueError(f"{name}: UPLO must be 'L' or 'U', not {UPLO!r}")


def _tolerance(name, value):
    """A tolerance as a float: a real number, not a bool; None stays None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, _numbers.Real):
        raise TypeError(f"{name}: a tolerance is a real number, not {type(value).__name__}")
    return float(value)


def eigh(a, UPLO="L", *, b=None, method="divide_and_conquer"):
    """The eigenvalues (real, ascending) and eigenvectors (columns) of a real symmetric or complex Hermitian
    matrix, as ``numpy.linalg.eigh``: its lower triangle is read (``UPLO="L"``), or its upper one (``"U"``).
    `method` diagonalises the tridiagonal form: ``"divide_and_conquer"`` (LAPACK's ``dsyevd``/``zheevd``),
    ``"tridiagonal_ql"`` (implicit QL, ``dsyev``'s, the default before 2026.10.4) or ``"jacobi"`` (cyclic Jacobi
    rotations, real matrices only).

    With `b`, a positive-definite matrix of `a`'s order, the pencil ``a x = λ b x``, as ``scipy.linalg.eigh(a, b)``:
    ``b = L Lᴴ`` and ``L⁻¹ a L⁻ᴴ`` diagonalised, the vectors ``b``-orthonormal (``Xᴴ b X = I``); each matrix read in
    the triangle `UPLO` names. Two matrices, not stacks; `b` keyword-only, `UPLO` holding NumPy's place."""
    upper = _upper("eigh", UPLO)
    if method not in _EIGH_METHODS:
        raise ValueError(f"eigh: method {method!r} is none of {_EIGH_METHODS}")
    if b is not None:
        return _eigh_pencil(a, b, method, upper)
    return _eigh(a, method, upper)


def eigvalsh(a, UPLO="L", *, b=None):
    """The eigenvalues alone of a real symmetric or complex Hermitian matrix, real and ascending, as
    ``numpy.linalg.eigvalsh``: its lower triangle is read (``UPLO="L"``), or its upper one (``"U"``). With `b`,
    those of the pencil ``a x = λ b x``, as :func:`eigh`'s."""
    upper = _upper("eigvalsh", UPLO)
    if b is not None:
        return _eigvalsh_pencil(a, b, upper)
    return _eigvalsh(a, upper)


def eig(a):
    """The eigenvalues and right eigenvectors (columns) of a general square matrix, or of each of a stack, as
    ``numpy.linalg.eig``: ``a @ v[:, j] == w[j] * v[:, j]``, each vector of unit norm with its component of largest
    modulus real. Complex whatever the eigenvalues, as NumPy 2.5's (complex64 for a single-precision matrix); in the
    order LAPACK's ``dgeev`` and ``zgeev`` give them (balancing, Hessenberg reduction, shifted QR), a real matrix's
    conjugate pairs together, the positive imaginary part first."""
    return _eig(a)


def eigvals(a):
    """The eigenvalues alone of a general square matrix, or of each of a stack, as ``numpy.linalg.eigvals``: complex
    whatever they are, as NumPy 2.5's; the same values as ``eig``'s, to the bit."""
    return _eigvals(a)


_SCHUR_SORTS = {"lhp": 0, "rhp": 1, "iuc": 2, "ouc": 3}


def schur(a, output="real", lwork=None, overwrite_a=False, sort=None, check_finite=True):
    """The Schur decomposition ``a = Z @ T @ Z.conj().T`` of a square matrix, as ``scipy.linalg.schur``: `Z`
    orthogonal (unitary), `T` upper quasi-triangular for a real matrix (a complex pair a 2 × 2 block, its diagonal
    equal), upper triangular with ``output="complex"`` (``"c"``) or for a complex matrix. Returns ``(T, Z)``. Not
    balanced, as LAPACK's ``dgees``.

    With `sort`, the eigenvalues it selects are moved to the leading block of `T` and ``(T, Z, sdim)`` returned,
    `sdim` their count: ``"lhp"`` (real part below 0), ``"rhp"`` (at or above 0), ``"iuc"`` (modulus at most 1),
    ``"ouc"`` (above 1), or a callable, given the real and imaginary parts of each eigenvalue for a real `T` and the
    eigenvalue for a complex one; a complex pair moves, and counts twice, where either of its two is selected.
    `lwork`, `overwrite_a` and `check_finite` are accepted and not needed.

    Raises:
        LinAlgError: if `a` is not square, holds an infinite or NaN entry, two blocks cannot be exchanged, or the
            exchanges' rounding leaves a selected eigenvalue after one not selected (real).
        ValueError: for another `output` or `sort`.
    """
    del lwork, overwrite_a, check_finite
    if output not in ("real", "r", "complex", "c"):
        raise ValueError(f"schur: output must be 'real' or 'complex', not {output!r}")
    complex_output = output in ("complex", "c")
    if sort is None:
        return _schur(a, complex_output)
    if callable(sort):
        kind, callback = 4, sort
    elif isinstance(sort, str) and sort in _SCHUR_SORTS:
        kind, callback = _SCHUR_SORTS[sort], None
    else:
        raise ValueError("'sort' parameter must either be 'None', or a callable, or one of ('lhp','rhp','iuc','ouc')")
    return _schur_sorted(a, complex_output, kind, callback)


def cholesky(a, /, *, upper=False):
    """The Cholesky factor of a Hermitian positive-definite matrix, as ``numpy.linalg.cholesky``: the lower `L` with
    ``a = L @ L.conj().T``, read from the lower triangle; with ``upper=True``, the upper `U` with ``a = U.conj().T @ U``,
    read from the upper triangle."""
    return _borrowed("cholesky", lambda x, flag: _cholesky(x, flag), a, bool(upper))


_solve_triangular = _linalg(_numerical.solve_triangular, arrays=2)
_lu_factor = _linalg(_numerical.lu_factor, arrays=1)
_TRANS = {0: 0, 1: 1, 2: 2, "N": 0, "T": 1, "C": 2}


def _trans(trans):
    """SciPy's `trans` as 0, 1 or 2: an integer or ``'N'``, ``'T'``, ``'C'``."""
    try:
        return _TRANS[trans]
    except (KeyError, TypeError):
        raise ValueError(f"trans must be 0, 1, 2, 'N', 'T' or 'C', not {trans!r}") from None


def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, check_finite=True):
    """The solution `x` of ``a @ x = b`` for a triangular `a`, as ``scipy.linalg.solve_triangular``: its `lower`
    triangle, or by default its upper one, is read and the other never; ``trans`` 1 (``'T'``) solves
    ``a.T @ x = b`` and 2 (``'C'``) ``a.conj().T @ x = b``; with ``unit_diagonal`` the diagonal is taken as ones.
    `b` is a vector or a matrix of right-hand sides. ``n² k`` operations where :func:`solve` factorises `a` first.
    `overwrite_b` is accepted and ignored (a Tensor is never written through); `check_finite` too, an infinite or
    NaN entry raising always.

    Raises:
        LinAlgError: if a diagonal entry read is zero ("singular matrix: resolution failed at diagonal i", 1-based,
            as SciPy's), an entry read is infinite or NaN, or the shapes do not fit.
    """
    del overwrite_b, check_finite
    operation = _trans(trans)
    try:  # a NumPy array read where it lies
        result = _numerical._borrowed_solve_triangular(a, b, bool(lower), operation, bool(unit_diagonal))
    except error as caught:
        raise LinAlgError(*caught.args) from None
    if result is not NotImplemented:
        return result
    return _solve_triangular(a, b, bool(lower), operation, bool(unit_diagonal))


class LUFactor(tuple):
    """:func:`lu_factor`'s answer: the pair ``(lu, piv)`` SciPy's ``lu_factor`` returns, unpacked the same way, and
    the factorisation itself: ``f.solve(b)`` is ``lu_solve(f, b)``."""

    __slots__ = ()

    def __new__(cls, lu, piv):
        return super().__new__(cls, (lu, piv))

    @property
    def lu(self):
        """`L` (its unit diagonal not stored) and `U` in one matrix."""
        return self[0]

    @property
    def piv(self):
        """LAPACK's row interchanges: row ``i`` swapped with row ``piv[i]``, in turn from the first."""
        return self[1]

    def solve(self, b, trans=0):
        """:func:`lu_solve` with this factorisation."""
        return lu_solve(self, b, trans)


def lu_factor(a, overwrite_a=False, check_finite=True):
    """The pivoted LU factorisation ``P a = L U`` of a square matrix, as ``scipy.linalg.lu_factor``: a
    :class:`LUFactor`, the pair ``(lu, piv)`` with `L` (unit diagonal not stored) and `U` in one matrix and LAPACK's
    row interchanges (a tuple of ints, where SciPy's is an integer array), to solve many systems with one
    factorisation (:func:`lu_solve`). A singular matrix factorises, a zero on `U`'s diagonal, as SciPy's (which warns).

    Raises:
        LinAlgError: if `a` is not square or holds an infinite or NaN entry.
    """
    del overwrite_a, check_finite
    lu, piv = _lu_factor(a)
    return LUFactor(lu, tuple(piv))


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """The solution of ``a @ x = b`` (``trans`` 0), ``a.T @ x = b`` (1, ``'T'``) or ``a.conj().T @ x = b`` (2,
    ``'C'``) from :func:`lu_factor`'s ``(lu, piv)`` (or any such pair), as ``scipy.linalg.lu_solve``: ``n² k``
    operations. A zero on `U`'s diagonal raises, where SciPy's divides by it.

    Raises:
        LinAlgError: for a zero on `U`'s diagonal, an infinite or NaN entry, or shapes that do not fit.
    """
    del overwrite_b, check_finite
    lu, piv = lu_and_piv
    operation = _trans(trans)
    try:  # Tensors and NumPy arrays read where they lie, b copied once
        result = _numerical._borrowed_factor_solve(lu, piv, b, False, operation)
    except error as caught:
        raise LinAlgError(*caught.args) from None
    if result is not NotImplemented:
        return result
    lu = lu if isinstance(lu, Tensor) else asarray(lu)
    b = b if isinstance(b, Tensor) else asarray(b)
    if lu.dtype != b.dtype:
        dtype = _promoted((lu.dtype, b.dtype))
        lu, b = asarray(lu, dtype), asarray(b, dtype)
    try:
        return _numerical.lu_solve(lu, piv, b, operation)
    except error as caught:
        raise LinAlgError(*caught.args) from None


class CholeskyFactor(tuple):
    """:func:`cho_factor`'s answer: the pair ``(c, lower)`` SciPy's ``cho_factor`` returns, unpacked the same way,
    and the factorisation itself: ``f.solve(b)`` is ``cho_solve(f, b)``."""

    __slots__ = ()

    def __new__(cls, c, lower):
        return super().__new__(cls, (c, lower))

    @property
    def c(self):
        """The factor: `L` in the lower triangle where :attr:`lower`, else `U` in the upper one, zeros elsewhere."""
        return self[0]

    @property
    def lower(self):
        """Whether :attr:`c` holds `L` (``a = L Lᴴ``), else `U` (``a = Uᴴ U``)."""
        return self[1]

    def solve(self, b):
        """:func:`cho_solve` with this factorisation."""
        return cho_solve(self, b)


def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """The Cholesky factorisation of a Hermitian positive-definite matrix, as ``scipy.linalg.cho_factor``: a
    :class:`CholeskyFactor`, the pair ``(c, lower)``, `c` the upper factor `U` (``a = Uᴴ U``) read from `a`'s upper
    triangle, or with ``lower`` the lower `L` (``a = L Lᴴ``) from its lower one; the other triangle zeros, where SciPy's
    leaves whatever `a` held there. For :func:`cho_solve`.

    Raises:
        LinAlgError: if `a` is not positive definite, not square, or holds an infinite or NaN entry.
    """
    del overwrite_a, check_finite
    return CholeskyFactor(cholesky(a, upper=not lower), bool(lower))


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """The solution of ``a @ x = b`` from :func:`cho_factor`'s ``(c, lower)``, as ``scipy.linalg.cho_solve``: two
    triangular solves (:func:`solve_triangular`'s bits), ``2 n² k`` operations; only `c`'s triangle is read, checked
    once with `b`, and an overflow between the two solves is the result's.

    Raises:
        LinAlgError: for a zero on `c`'s diagonal, an infinite or NaN entry, or shapes that do not fit.
    """
    del overwrite_b, check_finite
    c, lower = c_and_lower
    try:  # Tensors and NumPy arrays read where they lie, b copied once
        result = _numerical._borrowed_factor_solve(c, None, b, bool(lower), 0)
    except error as caught:
        raise LinAlgError(*caught.args) from None
    if result is not NotImplemented:
        return result
    if lower:
        return solve_triangular(c, solve_triangular(c, b, lower=True), trans=2, lower=True)
    return solve_triangular(c, solve_triangular(c, b, trans=2, lower=False), lower=False)


def lstsq(a, b, rcond=None):
    """``(x, residuals, rank, s)``, as ``numpy.linalg.lstsq``: the minimum-norm least-squares solution of
    ``a @ x ≈ b``, the residual sums of squares (for a tall `a` of full column rank, else empty), the rank and the
    singular values of `a`. A singular value at or under `rcond` times the largest counts as zero: by default
    (``None``) machine epsilon times max(m, n), NumPy's; a negative `rcond` is machine epsilon, as LAPACK's. The
    epsilon is the operands' precision's (float32's for float32 and complex64)."""
    rcond = _tolerance("lstsq", rcond)
    a = a if isinstance(a, Tensor) else asarray(a)
    b = b if isinstance(b, Tensor) else asarray(b)
    if rcond is not None and rcond < 0:
        rcond = _MACHINE_EPSILON["float32" if _promoted((a.dtype, b.dtype)) in _SINGLE else "float64"]
    return _lstsq(a, b, rcond)


def pinv(a, rcond=None, hermitian=False, *, rtol=None):
    """The Moore–Penrose pseudo-inverse, as ``numpy.linalg.pinv``: a singular value at or under `rcond` (or `rtol`,
    its NumPy 2 name; not both) times the largest counts as zero, by default machine epsilon times max(m, n). With
    ``hermitian=True`` `a` is taken as real symmetric or complex Hermitian, its lower triangle read, and the
    pseudo-inverse comes from its eigendecomposition, cheaper. A negative or NaN tolerance raises, where NumPy's
    result is infinite or NaN."""
    if rcond is not None and rtol is not None:
        raise ValueError("pinv: rtol and rcond can't be both set")
    return _pinv(a, _tolerance("pinv", rcond if rcond is not None else rtol), bool(hermitian))


def matrix_rank(A, tol=None, hermitian=False, *, rtol=None):
    """The numerical rank, as ``numpy.linalg.matrix_rank``: the singular values above `tol`, or above `rtol` times the
    largest (not both), by default machine epsilon times max(m, n) times the largest. With ``hermitian=True`` `A` is
    taken as real symmetric or complex Hermitian, its lower triangle read, its singular values the magnitudes of its
    eigenvalues, cheaper. A vector or a number has rank 1 unless all zero. A negative or NaN tolerance raises."""
    if tol is not None and rtol is not None:
        raise ValueError("matrix_rank: tol and rtol can't be both set")
    tol, rtol = _tolerance("matrix_rank", tol), _tolerance("matrix_rank", rtol)
    if not isinstance(A, Tensor):
        A = asarray(A)
    if A.ndim < 2:  # NumPy's: a vector or a number is of rank 1 unless all zero
        values = A.tolist()
        return int(any(value != 0 for value in (values if A.ndim == 1 else [values])))
    if A.ndim > 2:  # a stack: its ranks, integers nested along its leading axes (NumPy's an integer array)
        return _as_integers(_matrix_ranks(A, rtol, tol, bool(hermitian)).tolist())
    return _matrix_rank(A, rtol, tol, bool(hermitian))


def _as_integers(nested):
    """The numbers of a nested list as integers, nested alike."""
    return [_as_integers(item) for item in nested] if isinstance(nested, list) else int(nested)


def svd(a, full_matrices=True, compute_uv=True, hermitian=False, *, method="divide_and_conquer"):
    """``(U, S, Vh)`` with ``a = U @ diag(S) @ Vh``, the singular values descending, as ``numpy.linalg.svd``: for an
    m×n `a`, `U` is m×m and `Vh` n×n (``full_matrices=True``, NumPy's default: the columns past min(m, n) an
    orthonormal basis of the complement), or m×k and k×n, k = min(m, n) (``full_matrices=False``); the singular values
    alone with ``compute_uv=False``. `method` decomposes the bidiagonal form: ``"divide_and_conquer"`` (LAPACK's
    ``dgesdd``), ``"bidiagonal_qr"`` (implicit-shift QR, ``dgesvd``'s) or ``"jacobi"`` (one-sided Jacobi, each singular
    value to high relative accuracy). ``hermitian=True`` takes `a` as real symmetric or complex Hermitian (its lower
    triangle read) and decomposes it through its eigendecomposition, cheaper: the singular values the magnitudes of
    the eigenvalues, ``Vh = (U * sign(w)).conj().T``, as NumPy's."""
    if isinstance(full_matrices, str):
        raise TypeError("svd: the second argument is full_matrices, as in NumPy; give the method by name, "
                        f"svd(a, method={full_matrices!r})")
    if method not in _SVD_METHODS:
        raise ValueError(f"svd: method {method!r} is none of {_SVD_METHODS}")
    if hermitian:
        answer = _svd_hermitian(a)
        return answer if compute_uv else answer[1]
    if not compute_uv:
        return _singular_values(a)
    return _svd(a, method, bool(full_matrices))


def ldl(A, lower=True, hermitian=True, overwrite_a=False, check_finite=True):
    """The symmetric indefinite factorisation of a symmetric or Hermitian matrix, as ``scipy.linalg.ldl``:
    ``A = lu @ d @ lu.conj().T`` (``lu.T`` where `hermitian` is false, a complex symmetric `A`), Bunch and Kaufman's
    diagonal pivoting (LAPACK's ``sytf2`` and ``hetf2``) on the lower triangle, or on the upper with ``lower=False``;
    only that triangle is read. Returns ``(lu, d, perm)``: `d` block diagonal with blocks of order 1 and 2, ``lu[perm]``
    triangular with a unit diagonal, `perm` a tuple (SciPy's an integer array). A Hermitian matrix's diagonal
    imaginary parts are ignored with a ``ComplexWarning``, as SciPy's. `overwrite_a` and `check_finite` are accepted
    and not needed.

    Raises:
        LinAlgError: if `A` is not square or holds an infinite or NaN entry.
    """
    del overwrite_a, check_finite
    lu, d, perm, ignored = _ldl(A, bool(lower), bool(hermitian))
    if ignored:
        _warnings.warn("scipy.linalg.ldl():\nThe imaginary parts of the diagonal are ignored. Use \"hermitian=False\" for "
                       "factorization of complex symmetric arrays.", _complex_warning(), stacklevel=2)
    return lu, d, tuple(perm)


def _complex_warning():
    """NumPy's ``ComplexWarning`` where NumPy is installed, else :class:`RuntimeWarning` (its base)."""
    try:
        from numpy.exceptions import ComplexWarning
    except ImportError:  # pragma: no cover
        return RuntimeWarning
    return ComplexWarning


def qr(a, mode="reduced", *, pivoting=False):
    """``(Q, R)`` with ``a = Q @ R``, as ``numpy.linalg.qr``: for an m×n `a` and k = min(m, n), `Q` is m×k with
    orthonormal columns and `R` k×n upper trapezoidal (``mode="reduced"``, the default); `Q` m×m and `R` m×n
    (``"complete"``); `R` alone, k×n (``"r"``). `R`'s diagonal is real and non-negative.

    With ``pivoting=True``, as ``scipy.linalg.qr(a, pivoting=True)`` (LAPACK's ``geqp3``): ``a[:, P] = Q @ R`` with the
    column of largest remaining norm taken at each step, so that `R`'s diagonal does not grow down it and reveals the
    numerical rank; ``(Q, R, P)``, or ``(R, P)`` for ``"r"``, `P` a tuple of the columns' origins (SciPy's an integer
    array; its modes ``"economic"`` and ``"full"`` are ``"reduced"`` and ``"complete"`` here). `R`'s diagonal is real
    and non-negative here too, where LAPACK's may be negative."""
    if mode not in _QR_MODES:
        raise ValueError(f"qr: mode {mode!r} is none of {_QR_MODES} (the LAPACK reflectors of \"raw\" are not offered)")
    if pivoting:
        q, r, p = _qr_pivoted(a, mode == "complete")
        return (r, tuple(p)) if mode == "r" else (q, r, tuple(p))
    if mode == "r":
        return _qr_r(a)
    return _qr(a, mode)


def _tensor(x):
    return x if isinstance(x, Tensor) else asarray(x)


def _number_if_scalar(t):
    """A rank-0 tensor as the Python number it holds, as NumPy answers a full reduction; any other as it is."""
    return t[()] if isinstance(t, Tensor) and not t.shape else t


def _resolved_axes(t, axis, name):
    """`axis` (None for every axis, an integer or a tuple of them) as distinct non-negative axes of `t`."""
    rank = len(t.shape)
    if axis is None:
        return tuple(range(rank))
    axes = []
    for each in axis if isinstance(axis, tuple) else (axis,):
        each = _operator.index(each)
        if not -rank <= each < rank:
            raise _ShapeError(f"{name}: axis {each} is out of bounds for a tensor of rank {rank}")
        axes.append(each + rank if each < 0 else each)
    if len(set(axes)) != len(axes):
        raise ValueError(f"{name}: repeated axis")
    return tuple(axes)


def _permuted(t, order):
    """`t` with its axes in `order` (a permutation): through einsum, which copies."""
    letters = _labels(len(t.shape), "permute")
    return _numerical.einsum(f"{letters}->{''.join(letters[a] for a in order)}", t)


def _matmul_stack(a, b):
    """``a @ b`` for two stacks of one shape (..., m, k) and (..., k, n), or two matrices."""
    if len(a.shape) == 2 and len(b.shape) == 2:
        return a @ b
    labels = _labels(len(a.shape) + 1, "matmul")  # the batch's, then the matrices' three, apart from them
    batch, (y, z, w) = labels[:-3], labels[-3:]
    return _einsum(f"{batch}{y}{z},{batch}{z}{w}->{batch}{y}{w}", a, b)


def conj(x):
    """The complex conjugate of every element, as ``numpy.conj``; a real tensor's own elements."""
    return _numerical._unary("conj", _tensor(x))


conjugate = conj


def _vector_order(ord):
    if isinstance(ord, (str, bool)) or not isinstance(ord, _numbers.Real):
        raise ValueError(f"Invalid norm order {ord!r} for vectors")
    return float(ord)


def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
    """The `ord` norm of `x` over `axis` (None: every element; an integer or a tuple of them), as
    ``numpy.linalg.vector_norm``: ``(Σ |x|^ord)^(1/ord)``, ``inf`` the largest magnitude, ``-inf`` the smallest, 0 how
    many are not zero; real, of `x`'s precision. Summed at the scale of the largest magnitude, it does not overflow
    where NumPy's 2-norm of entries past 1e154 is infinite."""
    if axis is None and ord == 2 and not keepdims:
        plain = _numerical._borrowed_norm(x)  # read where it lies: a NumPy array is not copied
        if plain is not NotImplemented:
            return plain
    x = _tensor(x)
    order, axes = _vector_order(ord), list(_resolved_axes(x, axis, "vector_norm"))
    if order == -_math.inf and any(x.shape[a] == 0 for a in axes):  # the smallest of nothing, as NumPy's minimum
        raise ValueError("vector_norm: ord=-inf over no element has no value (a minimum of nothing)")
    return _native(_numerical._vector_norm, x, order, axes, bool(keepdims))


def svdvals(x, /):
    """The singular values of a matrix or of every matrix of a stack, descending, as ``numpy.linalg.svdvals``."""
    return svd(x, compute_uv=False)


def _matrix_norm(x, ord, row, col, keepdims):
    """The `ord` norm of the matrices of `x` over its axes `row` and `col` (distinct, non-negative)."""
    rank = len(x.shape)
    if ord in (None, "fro"):
        return vector_norm(x, axis=(row, col), keepdims=keepdims)
    if ord in ("nuc", 2, -2):
        moved = _permuted(x, [a for a in range(rank) if a not in (row, col)] + [row, col])
        values = svdvals(moved)
        reduced = sum(values, axis=-1) if ord == "nuc" else (max if ord == 2 else min)(values, axis=-1)
    elif ord in (1, -1, _math.inf, -_math.inf):
        inner, outer = (row, col) if ord in (1, -1) else (col, row)  # column sums for 1, row sums for inf
        sums = vector_norm(x, axis=inner, ord=1)
        reduced = (max if ord > 0 else min)(sums, axis=outer - (1 if outer > inner else 0))
    else:
        raise ValueError(f"Invalid norm order {ord!r} for matrices")
    if keepdims:
        shape = list(x.shape)
        shape[row] = shape[col] = 1
        return reshape(reduced, tuple(shape))
    return _number_if_scalar(reduced)


def matrix_norm(x, /, *, keepdims=False, ord="fro"):
    """The `ord` norm of a matrix or of every matrix of a stack (its last two axes), as ``numpy.linalg.matrix_norm``:
    ``"fro"`` (Frobenius, without overflow), ``"nuc"`` (the sum of the singular values), 2 and -2 (the largest and the
    smallest singular value), 1 and -1 (the largest and smallest column sum of magnitudes), ``inf`` and ``-inf``
    (row sums)."""
    x = _tensor(x)
    if len(x.shape) < 2:
        raise LinAlgError("matrix_norm: a matrix or a stack of matrices (rank ≥ 2)")
    rank = len(x.shape)
    return _matrix_norm(x, ord, rank - 2, rank - 1, keepdims)


def norm(x, ord=None, axis=None, keepdims=False):
    """A vector or a matrix norm, as ``numpy.linalg.norm``: with no `axis`, the 2-norm of every element for no `ord`,
    else the `ord` norm of the vector or the matrix `x` is; with an integer `axis`, the vector norms along it; with a
    pair, the matrix norms over it (:func:`vector_norm`, :func:`matrix_norm`)."""
    if axis is None and ord is None and not keepdims:
        plain = _numerical._borrowed_norm(x)  # every element's 2-norm, a NumPy array read where it lies
        if plain is not NotImplemented:
            return plain
    x = _tensor(x)
    rank = len(x.shape)
    if axis is None:
        if ord is None:
            return vector_norm(x, keepdims=keepdims)
        if rank == 1:
            return vector_norm(x, ord=ord, keepdims=keepdims)
        if rank == 2:
            return _matrix_norm(x, ord, 0, 1, keepdims)
        raise ValueError("norm: an ord with no axis takes a vector or a matrix")
    axes = _resolved_axes(x, axis, "norm")
    if len(axes) == 1:
        return vector_norm(x, axis=axes[0], keepdims=keepdims, ord=2 if ord is None else ord)
    if len(axes) == 2:
        return _matrix_norm(x, ord, axes[0], axes[1], keepdims)
    raise ValueError("norm: axis is an integer or a pair of them")


def cond(x, p=None):
    """The condition number of a matrix or of every matrix of a stack in the `p` norm, as ``numpy.linalg.cond``: for
    None, 2 and -2 the ratio of the extreme singular values (infinite for a singular matrix), else
    ``norm(x, p) * norm(inv(x), p)`` (infinite where `x` is singular)."""
    x = _tensor(x)
    if len(x.shape) < 2:
        raise LinAlgError("cond: a matrix or a stack of matrices (rank ≥ 2)")
    if 0 in x.shape[-2:]:  # as NumPy's
        raise LinAlgError("cond is not defined on empty arrays")
    if p in (None, 2, -2):
        values = svdvals(x)
        largest, smallest = _number_if_scalar(max(values, axis=-1)), _number_if_scalar(min(values, axis=-1))
        top, bottom = (largest, smallest) if p != -2 else (smallest, largest)

        def ratio(t, b):  # a singular matrix's, zero over zero included, is infinite, as NumPy's
            return _math.inf if b == 0 else t / b

        if not isinstance(top, Tensor):
            return ratio(top, bottom)
        return asarray(_paired(top.tolist(), bottom.tolist(), ratio), top.dtype)
    if x.shape[-1] != x.shape[-2]:
        raise LinAlgError(f"cond: the {p!r} norm takes square matrices")
    if len(x.shape) == 2:
        try:
            return norm(x, p) * norm(inv(x), p)
        except LinAlgError:
            return _math.inf
    stack = reshape(x, (-1,) + tuple(x.shape[-2:]))
    return reshape(asarray([cond(stack[i], p) for i in range(stack.shape[0])]), tuple(x.shape[:-2]))


def _paired(x, y, f):
    """`f` of the numbers of two nested lists of one shape, nested alike."""
    return [_paired(u, v, f) for u, v in zip(x, y)] if isinstance(x, list) else f(x, y)


def matrix_power(a, n):
    """`a` to the integer power `n` by repeated squaring, as ``numpy.linalg.matrix_power``: the identity for 0, the
    inverse's powers for a negative `n`; a matrix or a stack of square matrices."""
    a = _tensor(a)
    n = _operator.index(n)
    if len(a.shape) < 2 or a.shape[-1] != a.shape[-2]:
        raise LinAlgError("matrix_power: a square matrix or a stack of them")
    if n == 0:
        return eye(a.shape[-1], a.dtype) + zeros(a.shape, a.dtype)
    if n < 0:
        a, n = inv(a), -n
    result, square = None, a
    while n:
        if n & 1:
            result = square if result is None else _matmul_stack(result, square)
        n >>= 1
        if n:
            square = _matmul_stack(square, square)
    return result


def _chain_order(shapes):
    """The split of each range of the cheapest product of matrices of `shapes`, as NumPy's multi_dot finds it: by
    dynamic programming over the multiply-adds."""
    dims = [shapes[0][0]] + [shape[1] for shape in shapes]
    count = len(shapes)
    cost = [[0] * count for _ in range(count)]
    split = [[0] * count for _ in range(count)]
    for length in range(1, count):
        for i in range(count - length):
            j = i + length
            cost[i][j] = _math.inf
            for k in range(i, j):
                c = cost[i][k] + cost[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if c < cost[i][j]:
                    cost[i][j], split[i][j] = c, k
    return split


def multi_dot(arrays, *, out=None):
    """The product of two or more matrices in the cheapest order, as ``numpy.linalg.multi_dot``: the first may be a
    vector (a row), the last a vector (a column). `out` is not taken."""
    if out is not None:
        raise TypeError("multi_dot: out= is not taken; the product is returned")
    operands = [_tensor(array) for array in arrays]
    if len(operands) < 2:
        raise ValueError("multi_dot: at least two arrays")
    first_vector, last_vector = len(operands[0].shape) == 1, len(operands[-1].shape) == 1
    if first_vector:
        operands[0] = reshape(operands[0], (1, operands[0].shape[0]))
    if last_vector:
        operands[-1] = reshape(operands[-1], (operands[-1].shape[0], 1))
    if any(len(operand.shape) != 2 for operand in operands):
        raise LinAlgError("multi_dot: matrices, the first and last of which may be vectors")
    split = _chain_order([operand.shape for operand in operands])

    def product(i, j):
        return operands[i] if i == j else product(i, split[i][j]) @ product(split[i][j] + 1, j)

    result = product(0, len(operands) - 1)
    if first_vector and last_vector:
        return result[0, 0]
    if first_vector:
        return result[0]
    if last_vector:
        return reshape(result, (result.shape[0],))
    return result


def trace(x, /, *, offset=0, dtype=None):
    """The sum of the `offset` diagonal of a matrix or of every matrix of a stack (its last two axes), as
    ``numpy.linalg.trace``: above the main diagonal for a positive `offset`, below for a negative one; in `dtype` when
    given."""
    x = _tensor(x)
    if len(x.shape) < 2:
        raise _ShapeError("trace: a matrix or a stack of matrices (rank ≥ 2)")
    offset = _operator.index(offset)
    rows, cols = x.shape[-2:]
    r0, c0 = _builtins.max(-offset, 0), _builtins.max(offset, 0)
    length = _builtins.max(_builtins.min(rows - r0, cols - c0), 0)
    labels = _labels(len(x.shape) - 1, "trace")  # the batch's, then the diagonal's, apart from them
    batch, diagonal = labels[:-1], labels[-1]
    if length == 0:
        result = zeros(tuple(x.shape[:-2]), x.dtype)
    else:
        block = x[(slice(None),) * (len(x.shape) - 2) + (slice(r0, r0 + length), slice(c0, c0 + length))]
        result = _numerical.einsum(f"{batch}{diagonal}{diagonal}->{batch}", block)
    if dtype is not None:
        result = asarray(result, dtype)
    return _number_if_scalar(result)


def diagonal(x, /, *, offset=0):
    """The `offset` diagonal of a matrix or of every matrix of a stack (its last two axes), as
    ``numpy.linalg.diagonal``: above the main diagonal for a positive `offset`, below for a negative one."""
    x = _tensor(x)
    if len(x.shape) < 2:
        raise _ShapeError("diagonal: a matrix or a stack of matrices (rank ≥ 2)")
    offset = _operator.index(offset)
    rows, cols = x.shape[-2:]
    r0, c0 = _builtins.max(-offset, 0), _builtins.max(offset, 0)
    length = _builtins.max(_builtins.min(rows - r0, cols - c0), 0)
    if length == 0:
        return zeros(tuple(x.shape[:-2]) + (0,), x.dtype)
    labels = _labels(len(x.shape) - 1, "diagonal")
    batch, along = labels[:-1], labels[-1]
    block = x[(slice(None),) * (len(x.shape) - 2) + (slice(r0, r0 + length), slice(c0, c0 + length))]
    return _numerical.einsum(f"{batch}{along}{along}->{batch}{along}", block)


def matrix_transpose(x, /):
    """A matrix, or every matrix of a stack, transposed (its last two axes swapped), as
    ``numpy.linalg.matrix_transpose``."""
    x = _tensor(x)
    rank = len(x.shape)
    if rank < 2:
        raise _ShapeError("matrix_transpose: a matrix or a stack of matrices (rank ≥ 2)")
    return _permuted(x, list(range(rank - 2)) + [rank - 1, rank - 2])


def tensorsolve(a, b, axes=None):
    """`x` with ``tensordot(a, x, axes=x.ndim) == b``, as ``numpy.linalg.tensorsolve``: `a`'s leading axes those of
    `b`, its others `x`'s, their products equal (`axes`, when given, are moved to the end of `a` first)."""
    a, b = _same_dtype(a, b)
    rank = len(a.shape)
    if axes is not None:
        moved = [k for k in range(rank) if k not in [ax % rank for ax in axes]] + [ax % rank for ax in axes]
        a = _permuted(a, moved)
    shape = tuple(a.shape[len(b.shape):])
    rows = _math.prod(a.shape[:len(b.shape)])
    cols = _math.prod(shape)
    if rows != cols:
        raise LinAlgError(f"tensorsolve: a's leading axes hold {rows} elements and its trailing ones {cols}: no "
                          "square system")
    return reshape(solve(reshape(a, (rows, cols)), reshape(b, (rows,))), shape)


def tensorinv(a, ind=2):
    """The inverse of `a` read as a matrix of its first `ind` axes by the others, as ``numpy.linalg.tensorinv``:
    ``tensordot(tensorinv(a, ind), a, ind)`` is the identity of `a`'s trailing axes."""
    a = _tensor(a)
    ind = _operator.index(ind)
    if ind <= 0:
        raise ValueError("tensorinv: ind must be a positive integer")
    left, right = a.shape[:ind], a.shape[ind:]
    rows, cols = _math.prod(left), _math.prod(right)
    if rows != cols:
        raise LinAlgError(f"tensorinv: the first {ind} axes hold {rows} elements and the others {cols}: no square "
                          "matrix to invert")
    return reshape(inv(reshape(a, (rows, cols))), tuple(right) + tuple(left))


def outer(x1, x2, /):
    """The outer product of two vectors, ``x1[i] * x2[j]``, as ``numpy.linalg.outer``."""
    a, b = _same_dtype(x1, x2)
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError("outer: two vectors (numpy.linalg.outer)")
    # each element its own product, broadcast, as NumPy's multiply.outer: an einsum adds it to a zero, which loses −0
    return reshape(a, (a.shape[0], 1)) * reshape(b, (1, b.shape[0]))


def cross(x1, x2, /, *, axis=-1):
    """The cross product of 3-vectors along `axis`, broadcast over the other axes, as ``numpy.linalg.cross``."""
    axis = _operator.index(axis)  # one axis: TypeError for any other, as NumPy's
    a, b = _same_dtype(x1, x2)
    if len(a.shape) == 1 and len(b.shape) == 1:  # as rows, so a part is a tensor of the dtype, not a Python number
        rows = cross(reshape(a, (1,) + tuple(a.shape)), reshape(b, (1,) + tuple(b.shape)), axis=-1)
        return reshape(rows, tuple(a.shape))
    moved = []
    for operand in (a, b):
        if not operand.shape or operand.shape[_resolved_axes(operand, axis, "cross")[0]] != 3:
            raise ValueError("cross: both operands have 3 elements along axis (numpy.linalg.cross)")
        rank, at = len(operand.shape), _resolved_axes(operand, axis, "cross")[0]
        moved.append(_permuted(operand, [k for k in range(rank) if k != at] + [at]))
    a, b = moved

    def part(t, i):
        return t[(slice(None),) * (len(t.shape) - 1) + (i,)]

    components = [_tensor(c) for c in (part(a, 1) * part(b, 2) - part(a, 2) * part(b, 1),
                                       part(a, 2) * part(b, 0) - part(a, 0) * part(b, 2),
                                       part(a, 0) * part(b, 1) - part(a, 1) * part(b, 0))]
    # side by side, copied: products with unit vectors multiplied an infinite component by their zeros (NaN elsewhere)
    result = _numerical._stack_last(components)
    rank = len(result.shape)
    at = axis + rank if axis < 0 else axis
    if at != rank - 1:  # the vectors' axis back where it was
        order = list(range(rank - 1))
        order.insert(at, rank - 1)
        result = _permuted(result, order)
    return result


def tensordot(x1, x2, /, *, axes=2):
    """The sum of products over `axes` of `x1` and `x2`, as ``numpy.linalg.tensordot``: an integer N contracts the
    last N axes of `x1` with the first N of `x2`, a pair of sequences the axes it names."""
    a, b = _same_dtype(x1, x2)
    if isinstance(axes, (tuple, list)):
        left = [axes[0]] if isinstance(axes[0], int) else list(axes[0])
        right = [axes[1]] if isinstance(axes[1], int) else list(axes[1])
    else:
        count = _operator.index(axes)
        left, right = list(range(len(a.shape) - count, len(a.shape))), list(range(count))
    left = [k + len(a.shape) if k < 0 else k for k in left]
    right = [k + len(b.shape) if k < 0 else k for k in right]
    if len(left) != len(right) or any(a.shape[i] != b.shape[j] for i, j in zip(left, right)):
        raise ValueError("tensordot: the contracted axes do not match")
    labels = _labels(len(a.shape) + len(b.shape), "tensordot")
    letters_a = list(labels[:len(a.shape)])
    letters_b = list(labels[len(a.shape):])
    for i, j in zip(left, right):
        letters_b[j] = letters_a[i]
    kept = [letters_a[i] for i in range(len(a.shape)) if i not in left] + \
           [letters_b[j] for j in range(len(b.shape)) if j not in right]
    return _number_if_scalar(_numerical.einsum(f"{''.join(letters_a)},{''.join(letters_b)}->{''.join(kept)}", a, b))


def vecdot(x1, x2, /, *, axis=-1):
    """``Σ conj(x1) · x2`` along `axis`, broadcast over the other axes, as ``numpy.linalg.vecdot``."""
    a, b = _same_dtype(x1, x2)
    product = conj(a) * b
    return sum(product, axis=_resolved_axes(product, axis, "vecdot")[0])


_NATIVE_NUMBERS = (float, int, bool, complex)


def asarray(obj, dtype=None):
    """A :class:`Tensor` from `obj`: a Tensor (the same object when `dtype` matches), an object with an
    ``__array_interface__`` (NumPy: bool, integer, float32/64 and complex64/128 elements, any strides),
    a nested list or tuple of numbers, or a number (rank 0). `dtype` is ``"float64"``, ``"complex128"``,
    ``"float32"`` or ``"complex64"``; by default an array's own for float32, complex64 and complex128, else
    float64 (bool and integers included), and complex128 for a complex number or list. A complex input is never
    cast to a real dtype (that would discard the imaginary part).

    Raises:
        ValueError: if a nested list is not rectangular.
        TypeError: if `obj` is none of these, or a nested list holds something other than numbers.
    """
    # A float, int, bool or complex, an array (its __array_struct__ or __array_interface__) and a Tensor go to C
    # as they are; a nested list, or another number (a Fraction), through its values. A Tensor or an array's struct
    # first: the abstract Number test below took a third of a small einsum's call.
    if type(obj) is Tensor or hasattr(obj, "__array_struct__"):
        return _asarray(obj, dtype) if dtype is not None else _asarray(obj)
    if isinstance(obj, (list, tuple)):  # floats, ints and complex numbers in one walk in C; anything else read here
        fast = _asarray_nested(obj, dtype) if dtype is not None else _asarray_nested(obj)
        if fast is not None:
            return fast
    if isinstance(obj, (list, tuple)) or (type(obj) not in _NATIVE_NUMBERS and isinstance(obj, _numbers.Number)
                                          and not hasattr(obj, "__array_interface__")):
        values, shape = _nested(obj)
        obj = _NestedBuffer(values, shape)
    elif not (isinstance(obj, Tensor) or type(obj) in _NATIVE_NUMBERS or hasattr(obj, "__array_struct__")
              or hasattr(obj, "__array_interface__")):  # a NumPy array's struct a capsule, its interface a dict built anew
        raise _UnreadableError(f"asarray: a {type(obj).__name__} is not a tensor, an array, a nested list of numbers "
                               "or a number")
    return _asarray(obj, dtype) if dtype is not None else _asarray(obj)


class _UnreadableError(error, TypeError):
    """An object :func:`asarray` cannot read: a ``TypeError`` as NumPy raises, and a :class:`numerical.error`."""



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
    """`a` and `b` as Tensors of one dtype, as NumPy 2 promotes them: array-likes through :func:`asarray`; two
    tensors to the dtype both fit (:func:`_promoted`); a Python number keeps the tensor's precision, made complex
    only by a complex number (``float32 + 2.0`` is float32)."""
    weak_a, weak_b = _weak(a) and not _weak(b), _weak(b) and not _weak(a)
    tensor_a = a if isinstance(a, Tensor) else asarray(a)
    tensor_b = b if isinstance(b, Tensor) else asarray(b)
    if weak_a:
        dtype = _beside_weak(tensor_b.dtype, a)
    elif weak_b:
        dtype = _beside_weak(tensor_a.dtype, b)
    else:
        dtype = _promoted((tensor_a.dtype, tensor_b.dtype))
    return asarray(tensor_a, dtype), asarray(tensor_b, dtype)


_numpy_loops_measured = False
_numpy_shortcuts_any_number = False  # NumPy before 2.3 takes ** of any number 2, -1 or 0.5 by a shortcut, not of exact ones


def _numpy_loops():
    """Measures, once NumPy is imported, which of its complex loops fuse their multiply-adds, so that ``*`` and ``/``
    give its bits: they follow how its wheel was built (clang and MSVC contract its scalar loop, GCC does not), not
    the processor alone. Each probe's exact answer is a 0 where the loop rounds each product, and not where it fuses."""
    global _numpy_loops_measured, _numpy_shortcuts_any_number
    np = _sys.modules.get("numpy")
    if np is None:
        return
    _numpy_loops_measured = True
    e = 2.0**-30
    x, y = complex(1 + e, 1), complex(1 - e, 1)  # the real part (1 + e)(1 - e) - 1: -e² fused, 0 rounded
    # Smith's a / (1 + r j) adds a.imag · r to a.real: (1 - 2^-30)² sits 2^-60 past 1 - 2^-29 fused, on it rounded
    r = 1 - 2.0**-30
    with np.errstate(all="ignore"):
        products = np.multiply(np.array([x, x]), np.array([y, y]))[0].real != 0  # the vector loop
        scalar_products = np.multiply(np.array([[x]]), np.array([y]))[0, 0].real != 0  # one element, broadcast
        quotients = np.divide(np.array([complex(-(1 - 2.0**-29), r)]), np.array([complex(1, r)]))[0].real != 0
        # (inf + 0j) ** -1.0: 0 by the reciprocal a shortcut takes, NaN by the power (its 1 · inf is 0 · inf in part)
        _numpy_shortcuts_any_number = not np.isnan((np.array([complex(np.inf, 0.0)]) ** -1.0)[0].real)
    _numerical._set_numpy_loops(bool(products), bool(scalar_products), bool(quotients))


def _arithmetic(op, reflected=False):
    """The Tensor method for `op` ("+", "-", "*", "/", or "^" for ``**``), element-wise and broadcast as NumPy's; a reflected
    one (``2 - t``) puts the other operand first. Anything :func:`asarray` cannot read is NotImplemented."""

    def method(self, other):
        kind = type(other)
        if op == "^" and not reflected and self.dtype.startswith("complex"):
            # NumPy takes an array to the Python integer 2 or -1, or the float 0.5, by its square, its reciprocal or its
            # square root, not by its power; before 2.3, to any real number equal to them
            if not _numpy_loops_measured:
                _numpy_loops()
            if _numpy_shortcuts_any_number:
                if isinstance(other, _numbers.Real) and other == 1:  # its positive: the elements themselves, -0 kept
                    return self
                shortcut = {2: "2", -1: "r", 0.5: "h"}.get(other) if isinstance(other, _numbers.Real) else None
            else:
                shortcut = {(int, 2): "2", (int, -1): "r", (float, 0.5): "h"}.get((kind, other)) if kind in (int, float) else None
            if shortcut is not None:
                return _native(_numerical._power_shortcut, shortcut, self)
        if kind is Tensor and other.dtype == self.dtype:  # nothing to promote: the common cases, straight through
            a, b = self, other
        elif kind is float or kind is int:  # a real Python number is weak: it takes the tensor's dtype
            a, b = self, asarray(other, self.dtype)
        else:
            try:
                a, b = _same_dtype(self, other)
            except (TypeError, error):  # asarray cannot read it: let the other operand try
                return NotImplemented
        if reflected:
            a, b = b, a
        if not _numpy_loops_measured and op in ("*", "/", "^"):
            _numpy_loops()
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
Tensor.__pow__ = _arithmetic("^")
Tensor.__rpow__ = _arithmetic("^", reflected=True)
Tensor.__abs__ = lambda self: absolute(self)
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
    def reduce(a, axis=None, keepdims=False):
        if axis is None and not keepdims and not isinstance(a, Tensor):
            whole = _numerical._borrowed_reduce(op, a)  # every element of a NumPy array read where it lies
            if whole is not NotImplemented:
                return whole
        a = a if isinstance(a, Tensor) else asarray(a)
        rank = len(a.shape)
        if isinstance(axis, tuple):  # several axes: moved last and run together, one native reduction over them
            axes = _resolved_axes(a, axis, op)
            if not axes:
                return a
            kept = [k for k in range(rank) if k not in axes]
            if kept:
                moved = _permuted(a, kept + sorted(axes)) if kept + sorted(axes) != list(range(rank)) else a
                count = _math.prod(a.shape[k] for k in axes)
                joined = reshape(moved, tuple(a.shape[k] for k in kept) + (count,))
                result = _native(_numerical._reduce, op, joined, len(kept))
            else:
                result = _native(_numerical._reduce, op, a, -1)
        else:
            axes = _resolved_axes(a, axis, op)
            result = _native(_numerical._reduce, op, a, _axis(a, axis))
        if not keepdims:
            return result
        shape = tuple(1 if k in axes else a.shape[k] for k in range(rank))  # each reduced axis kept with extent 1
        if not isinstance(result, Tensor):
            return full(shape, result, a.dtype)
        return reshape(result, shape)

    reduce.__name__ = op
    reduce.__doc__ = doc + """ `axis` may name several axes (a tuple, `()` for none), and `keepdims` keeps each reduced
axis with extent 1, as NumPy's."""
    return reduce


sum = _reduction("sum", """The sum of every element (a number), or the sums along `axis` (a Tensor without it),
as ``numpy.sum``; added by pairwise halving, as NumPy adds.""")
mean = _reduction("mean", """The mean of every element, or the means along `axis`, as ``numpy.mean``.""")
max = _reduction("max", """The largest element, or the largest along `axis`, as ``numpy.max``; a NaN wins. Real
dtypes only (a complex one has no order).""")
min = _reduction("min", """The smallest element, or the smallest along `axis`, as ``numpy.min``; a NaN wins. Real
dtypes only.""")
def _dtype_name(dtype):
    """A dtype's name as a Tensor takes it: a string as it is, ``numpy.float64`` or ``numpy.dtype("float32")`` by name."""
    if isinstance(dtype, str):
        return dtype
    return getattr(dtype, "__name__", None) or str(dtype)


def _reduction_method(reduce):
    """The Tensor method of `reduce`, which NumPy's `numpy.sum(t)` and the like call with `out` (and `dtype` for a
    mean): None the only `out` a Tensor takes, since it is never written into; `dtype` converts the result."""
    def method(self, axis=None, dtype=None, out=None, keepdims=False):
        if out is not None:
            raise TypeError(f"{reduce.__name__}: a Tensor takes no `out` (it is never written into)")
        result = reduce(self, axis, keepdims)
        if dtype is None:
            return result
        converted = asarray(result, dtype=_dtype_name(dtype))
        return converted if isinstance(result, Tensor) else converted[()]

    method.__name__ = reduce.__name__
    method.__doc__ = reduce.__doc__
    return method


Tensor.sum = _reduction_method(sum)
Tensor.mean = _reduction_method(mean)
Tensor.max = _reduction_method(max)
Tensor.min = _reduction_method(min)


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


def _dtype_of(dtype, value=0.0):
    """`dtype` checked, one of float64, complex128, float32 and complex64; ``None``: complex128 for a complex
    `value`, else float64, as NumPy infers it."""
    if dtype is None:
        return "complex128" if isinstance(value, complex) else "float64"
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {_DTYPES}, not {dtype!r}")
    if not dtype.startswith("complex") and isinstance(value, complex):
        raise TypeError(f"full: a complex value does not fit a {dtype} tensor")
    return dtype


def full(shape, value, dtype=None):
    """A Tensor of `shape` whose every element is `value`, as ``numpy.full``; complex128 for a complex value and
    float64 otherwise unless `dtype` says (float64, complex128, float32 or complex64)."""
    return _native(_numerical._full, _shape(shape), value, _dtype_of(dtype, value))


def zeros(shape, dtype="float64"):
    """A Tensor of `shape` full of zeros, as ``numpy.zeros``."""
    return _native(_numerical._full, _shape(shape), 0.0, _dtype_of(dtype))


def ones(shape, dtype="float64"):
    """A Tensor of `shape` full of ones, as ``numpy.ones``."""
    return _native(_numerical._full, _shape(shape), 1.0, _dtype_of(dtype))


def eye(n, dtype="float64"):
    """The `n × n` identity, as ``numpy.eye(n)``."""
    return _native(_numerical._eye, _operator.index(n), _dtype_of(dtype))


def arange(start, stop=None, step=1.0):
    """``start, start + step, …`` short of `stop`, as ``numpy.arange``: ``arange(stop)`` starts at 0. float64."""
    if stop is None:
        start, stop = 0.0, start
    return _native(_numerical._arange, float(start), float(stop), float(step))


def linspace(start, stop, num=50):
    """`num` values evenly spaced from `start` to `stop`, both included, as ``numpy.linspace``. float64."""
    return _native(_numerical._linspace, float(start), float(stop), _operator.index(num))


def _elementwise(name, doc):
    def function(a):
        if name == "abs" and not _numpy_loops_measured:  # a complex magnitude fused as NumPy's vector loop
            _numpy_loops()
        if not isinstance(a, Tensor):
            mapped = _numerical._borrowed_unary(name, a)  # a NumPy array read where it lies
            if mapped is not NotImplemented:
                return mapped
        return _numerical._unary(name, a if isinstance(a, Tensor) else asarray(a))

    function.__name__ = name
    function.__doc__ = doc
    return function


exp = _elementwise("exp", "e to the power of every element, as ``numpy.exp``.")
log = _elementwise("log", "The natural logarithm of every element, as ``numpy.log`` (NaN below 0 for float64).")
sqrt = _elementwise("sqrt", "The square root of every element, as ``numpy.sqrt`` (NaN below 0 for float64).")
sin = _elementwise("sin", "The sine of every element, as ``numpy.sin``.")
cos = _elementwise("cos", "The cosine of every element, as ``numpy.cos``.")
tan = _elementwise("tan", "The tangent of every element, as ``numpy.tan``.")
tanh = _elementwise("tanh", "The hyperbolic tangent of every element, as ``numpy.tanh``.")
absolute = _elementwise("abs", "The magnitude of every element, real of the same precision, as ``numpy.abs``.")
absolute.__name__ = "absolute"


def power(a, b):
    """`a` to the power `b` element by element, broadcast, as ``numpy.power``."""
    a, b = _same_dtype(a, b)
    if not _numpy_loops_measured:
        _numpy_loops()
    return _native(_numerical._binary, "^", a, b)


__all__ += ["diagonal", "matrix_transpose", "tensorsolve", "tensorinv"]
__all__ += ["sum", "mean", "max", "min", "reshape", "full", "zeros", "ones", "eye", "arange", "linspace",
            "exp", "log", "sqrt", "sin", "cos", "tan", "tanh", "absolute", "power"]

from numerical import fft  # noqa: E402  (the module numpy.fft's names live in, as numpy.fft is NumPy's)
