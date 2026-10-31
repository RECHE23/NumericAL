r"""NumericAL — a generic, header-only numerical library, exposed to Python.

The one object is :class:`Tensor` — an N-dimensional, row-major array of
``float64``, ``complex128``, ``float32``, ``complex64`` or ``bool`` (rank-1 is a vector, rank-2 a matrix, rank-0 a
scalar; ``bool`` the answer of a comparison, a mask). The algebra over it is reached through module functions, which take a
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
    "schur", "solve_triangular", "lu", "lu_factor", "lu_solve", "cho_factor", "cho_solve", "LUFactor", "CholeskyFactor",
]


class LinAlgError(error, ValueError):
    """Raised by the linear-algebra functions when they cannot compute their answer: a singular matrix, one
    not positive definite, an infinite or NaN entry, an iteration that did not converge, or an operand of the
    wrong shape. As ``numpy.linalg.LinAlgError`` it is a :class:`ValueError`; it is also a
    :class:`numerical.error`, so code catching that keeps working."""


__version__ = "2026.10.45"

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
        is_complex = _builtins.any(isinstance(value, complex) or (isinstance(value, _numbers.Complex)
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


_DTYPES = ("float64", "complex128", "float32", "complex64", "bool")
_SINGLE = ("float32", "complex64")


def _promoted(dtypes):
    """NumPy's result dtype of tensors of `dtypes`: complex where one is, single precision only where all are; a bool
    one takes the others' dtype, and bools alone stay bool."""
    dtypes = [dtype for dtype in dtypes if dtype != "bool"]
    if not dtypes:
        return "bool"
    single = _builtins.all(dtype in _SINGLE for dtype in dtypes)
    if _builtins.any(dtype.startswith("complex") for dtype in dtypes):
        return "complex64" if single else "complex128"
    return "float32" if single else "float64"


def _weak(value):
    """Whether `value` is a Python number, which NumPy 2 promotes as weak: it keeps the tensor's precision."""
    return type(value) in _NATIVE_NUMBERS


def _beside_weak(dtype, value):
    """The dtype of a tensor of `dtype` beside the Python number `value`: its own, complex for a complex `value`; a
    bool tensor's beside a bool, else the number's own (float64 for an int, as :func:`asarray` reads one)."""
    if dtype == "bool":
        return "bool" if isinstance(value, bool) else "complex128" if isinstance(value, complex) else "float64"
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
    view would). An integer or bool array in the key (a list, a NumPy array, a bool Tensor, a lone bool) makes it
    NumPy's advanced indexing (:func:`_getitem_advanced`)."""
    if not isinstance(key, tuple):
        key = (key,)
    shape = self.shape
    if not _builtins.all(type(k) is int or type(k) is slice for k in key):
        arrays = [None if k is None or k is Ellipsis or type(k) is int or type(k) is slice else _index_array(k)
                  for k in key]
        if _builtins.any(array is not None for array in arrays):
            return _getitem_advanced(self, key, arrays)
    if _builtins.any(k is Ellipsis for k in key) or _builtins.any(k is None for k in key):
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


def _index_array(index):
    """An index NumPy reads as an array: ``("mask", t)`` for a bool one (a bool Tensor, a NumPy bool array, a list of
    bools, a lone bool, whose mask has rank 0), ``("ints", values, shape)`` for integers (a list or a NumPy integer array,
    its values in C order); None for an integer, a slice, ``...`` or ``None``. Any other array raises IndexError, as
    NumPy's (a float one included)."""
    if isinstance(index, bool) or (type(index).__module__ == "numpy" and type(index).__name__ in ("bool", "bool_")):
        return "mask", asarray(bool(index), "bool")  # before an integer: operator.index(True) is 1
    if isinstance(index, Tensor):
        if index.dtype != "bool":
            raise IndexError(f"arrays used as indices must be of integer (or boolean) type, not {index.dtype}")
        return "mask", index
    if isinstance(index, (list, tuple)):
        try:
            values, shape = _nested(index)
        except (TypeError, ValueError) as failure:
            raise IndexError(f"an index list holds integers or bools, in a rectangular nesting: {failure}") from None
        if values and _builtins.all(isinstance(v, bool) for v in values):
            return "mask", asarray(index, "bool")
        if not _builtins.all(isinstance(v, _numbers.Integral) for v in values):
            raise IndexError("arrays used as indices must be of integer (or boolean) type")
        return "ints", [int(v) for v in values], shape
    if type(index).__name__ == "ndarray" and getattr(index, "ndim", 0) > 0:
        kind = index.dtype.kind
        if kind == "b":
            return "mask", asarray(index, "bool")
        if kind not in "iu":
            raise IndexError(f"arrays used as indices must be of integer (or boolean) type, not {index.dtype}")
        return "ints", index.ravel(), tuple(index.shape)
    if type(index).__name__ == "ndarray" and index.dtype.kind == "b":  # a rank-0 bool array, as a lone bool
        return "mask", asarray(bool(index), "bool")
    return None


def _broadcast_shape(shapes):
    """The shape the advanced indices' `shapes` broadcast to, as NumPy's; IndexError where they do not."""
    rank = _builtins.max(len(shape) for shape in shapes)
    out = []
    for extents in zip(*((1,) * (rank - len(shape)) + tuple(shape) for shape in shapes)):
        wide = {extent for extent in extents if extent != 1}
        if len(wide) > 1:
            raise IndexError("shape mismatch: indexing arrays could not be broadcast together with shapes "
                             + " ".join(str(tuple(shape)) for shape in shapes))
        out.append(wide.pop() if wide else 1)
    return tuple(out)


def _broadcast_offsets(parts, out):
    """The sum of the advanced indices' offsets ``(values, shape)``, each broadcast to the shape `out`."""
    if _builtins.all(len(values) == len(parts[0][0]) for values, _ in parts) and len(parts[0][0]) == _math.prod(out):
        return [_builtins.sum(column) for column in zip(*(values for values, _ in parts))]  # no axis broadcast
    total = _math.prod(out)
    summed = [0] * total
    for values, shape in parts:
        steps, step = [], 1  # each axis' step in this part's values, 0 where it is broadcast
        for extent in reversed((1,) * (len(out) - len(shape)) + tuple(shape)):
            steps.append(step if extent != 1 else 0)
            step *= extent
        steps.reverse()
        for flat in range(total):
            rest, at = flat, 0
            for axis in range(len(out) - 1, -1, -1):
                at += (rest % out[axis]) * steps[axis]
                rest //= out[axis]
            summed[flat] += values[at]
    return summed


_NATIVE_INT64 = ("<" if _sys.byteorder == "little" else ">") + "i8"  # NumPy's dtype.str of this machine's int64


def _offsets_bytes(parts, extents):
    """The advanced indices' offsets ``(values, shape)`` summed over their broadcast shape `extents`, as int64 bytes for
    :func:`_numerical._gather`; the values bytes (a mask's), a NumPy array (an integer array's) or a list."""
    if len(parts) == 1:
        values = parts[0][0]
        if isinstance(values, bytes):
            return values
        return _array.array("q", values).tobytes() if isinstance(values, list) else values.tobytes()
    arrays = [values for values, _ in parts if type(values).__name__ == "ndarray"]
    if arrays:  # NumPy's broadcasting, one of its arrays being an index
        numpy = _sys.modules[type(arrays[0]).__module__.split(".")[0]]
        summed = numpy.zeros(extents, _NATIVE_INT64)
        for values, shape in parts:
            summed += numpy.frombuffer(values, _NATIVE_INT64).reshape(shape) if isinstance(values, bytes) \
                else numpy.asarray(values, _NATIVE_INT64).reshape(shape)
        return summed.tobytes()
    lists = [(_array.array("q", values) if isinstance(values, bytes) else values, shape) for values, shape in parts]
    return _array.array("q", _broadcast_offsets(lists, extents)).tobytes()


def _bounded_offsets(values, extent, axis, stride):
    """The offsets of the integers `values` along an axis of `extent` and `stride`, a negative one counted from the end;
    IndexError for one outside the axis, in NumPy's words."""
    if type(values).__name__ == "ndarray":  # NumPy's own arithmetic, at its speed
        if values.size and (values.max() >= extent or values.min() < -extent):
            value = values[(values >= extent) | (values < -extent)][0]
            raise IndexError(f"index {value} is out of bounds for axis {axis} with size {extent}")
        if values.dtype.str == _NATIVE_INT64 and stride == 1 and (not values.size or values.min() >= 0):
            return values  # the offsets themselves
        at = values.astype(_NATIVE_INT64)
        at[at < 0] += extent
        return at * stride
    offsets = []
    for value in values:
        at = value + extent if value < 0 else value
        if not 0 <= at < extent:
            raise IndexError(f"index {value} is out of bounds for axis {axis} with size {extent}")
        offsets.append(at * stride)
    return offsets


def _axes_taken(entry):
    """How many of the tensor's axes an index entry of :func:`_getitem_advanced` takes: a mask its rank, ``None`` and
    ``...`` none (``...`` stands for the rest), any other one."""
    if entry[0] == "mask":
        return len(entry[1].shape)
    if entry[1] is None or entry[1] is Ellipsis:
        return 0
    return 1


def _getitem_advanced(self, key, arrays):
    """``t[[0, 2]]``, ``t[mask]``, ``t[:, [1, 0], 2]``: NumPy's advanced indexing. Every integer or bool array, and
    every integer beside one, is an advanced index: their shapes broadcast together (a bool mask counts as the
    coordinates of its true elements, a lone bool as a mask of rank 0 over a new axis), and the result has that shape
    where the advanced indices stand when they stand together, at the front otherwise, the other indices' axes as basic
    indexing gives them. A copy, as NumPy's."""
    shape = self.shape
    entries = []
    for index, array in zip(key, arrays):
        if array is not None:
            entries.append(array)
        elif index is None or index is Ellipsis or isinstance(index, slice):
            entries.append(("basic", index))
        else:
            entries.append(("int", _operator.index(index)))
    if _builtins.sum(1 for kind, index, *_ in entries if index is Ellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    used = _builtins.sum(_axes_taken(entry) for entry in entries)
    if used > len(shape):
        raise IndexError(f"too many indices for a tensor of rank {len(shape)}: {used} were given")
    expanded = []
    for entry in entries:
        expanded += [("basic", slice(None))] * (len(shape) - used) if entry[1] is Ellipsis else [entry]
    if not _builtins.any(entry[1] is Ellipsis for entry in entries):
        expanded += [("basic", slice(None))] * (len(shape) - used)
    strides = [_math.prod(shape[axis + 1:]) for axis in range(len(shape))]
    slots, parts, positions, axis = [], [], [], 0
    for position, entry in enumerate(expanded):
        kind, index = entry[0], entry[1]
        if kind == "basic" and index is None:
            slots.append(((0, 0, 1), 1))
        elif kind == "basic":
            span = range(*index.indices(shape[axis]))
            slots.append(((span.start * strides[axis], span.step * strides[axis], len(span)), len(span)))
            axis += 1
        elif kind == "mask":
            rank = len(index.shape)
            for along, (extent, size) in enumerate(zip(shape[axis:axis + rank], index.shape), axis):
                if extent != size and index.size != 0:  # NumPy holds a mask without elements to no extent
                    raise IndexError(f"boolean index did not match indexed tensor along axis {along}; size of axis is "
                                     f"{extent} but size of corresponding boolean axis is {size}")
            offsets = _numerical._true_offsets(index, strides[axis:axis + rank])
            parts.append((offsets, (len(offsets) // 8,)))
            positions.append(position)
            axis += rank
        else:
            if kind == "int":  # NumPy checks an integer's bound at once, an array's only for a result with elements
                parts.append((_bounded_offsets([index], shape[axis], axis, strides[axis]), ()))
            else:
                parts.append((index, entry[2], axis))
            positions.append(position)
            axis += 1
    extents = _broadcast_shape([part[1] for part in parts])
    if _math.prod(extents) == 0:  # NumPy checks no bound then
        block = (b"", extents)
    else:
        parts = [part if len(part) == 2 else (_bounded_offsets(part[0], shape[part[2]], part[2], strides[part[2]]),
                                              part[1]) for part in parts]
        block = (_offsets_bytes(parts, extents), extents)
    together = positions == list(range(positions[0], positions[0] + len(positions)))
    first = positions[0] if together else 0
    groups, out_shape, slot = [], [], iter(slots)
    for position in range(len(expanded)):
        if position == first:
            groups.append(block[0])
            out_shape += block[1]
        if position not in positions:
            offsets, extent = next(slot)
            groups.append(offsets)
            out_shape.append(extent)
    return _numerical._gather(self, groups, out_shape)


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
    if not _builtins.any(k is None for k in expanded):
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
    try:
        return _numerical.einsum(_REVERSALS[rank], self)
    except TypeError:  # a bool tensor, which einsum's arithmetic refuses: its elements moved alone
        return _numerical._permute(self, range(rank - 1, -1, -1))


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


def _comparison(op):
    """The Tensor method for the comparison `op` ("<", "<=", "==", "!=", ">", ">="), element-wise and broadcast as
    NumPy's, a bool tensor; the operands promoted as for arithmetic. Anything :func:`asarray` cannot read is
    NotImplemented (so ``t == None`` is False, as Python decides)."""

    def method(self, other):
        if type(other) is Tensor and other.dtype == self.dtype:
            a, b = self, other
        else:
            try:
                a, b = _same_dtype(self, other)
            except (TypeError, error):
                return NotImplemented
        return _native(_numerical._compare, op, a, b)

    return method


def _boolean(value):
    """`value` as a bool Tensor: a Tensor, or what :func:`asarray` reads, of bools (a NumPy bool array, a Python bool, a
    nested list of them); None for anything else, which the logical operators leave to the other operand."""
    if isinstance(value, Tensor):
        return value if value.dtype == "bool" else None
    if isinstance(value, bool) or str(getattr(value, "dtype", "")) == "bool":
        return asarray(value, "bool")
    if isinstance(value, (list, tuple)):
        try:
            values, _ = _nested(value)
        except (TypeError, ValueError):
            return None
        return asarray(value, "bool") if _builtins.all(isinstance(v, bool) for v in values) else None
    return None


def _logical(op, reflected=False):
    """The Tensor method for ``&``, ``|`` or ``^`` (`op`) of bool tensors, broadcast; NotImplemented for another
    operand or dtype, as Python then raises TypeError (NumPy's bitwise operators take no float)."""

    def method(self, other):
        a, b = _boolean(self), _boolean(other)
        if a is None or b is None:
            return NotImplemented
        return _native(_numerical._logical, op, *((b, a) if reflected else (a, b)))

    return method


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
    if not _builtins.any(isinstance(item, list) for item in nested):
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
for _name, _op in (("__eq__", "=="), ("__ne__", "!="), ("__lt__", "<"), ("__le__", "<="), ("__gt__", ">"),
                  ("__ge__", ">=")):
    setattr(Tensor, _name, _comparison(_op))
for _name, _op in (("and", "&"), ("or", "|"), ("xor", "^")):
    setattr(Tensor, f"__{_name}__", _logical(_op))
    setattr(Tensor, f"__r{_name}__", _logical(_op, reflected=True))
del _name, _op
Tensor.__invert__ = lambda self: _numerical._invert(self)
Tensor.__hash__ = None  # unhashable, as a NumPy array: equality is element-wise

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
    if _builtins.any(tensor.ndim > 1 for tensor in tensors):
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
    if _builtins.any(tensor.ndim > 2 for tensor in tensors):
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
    form made complex), `func` called with `T`'s diagonal (a NumPy array where NumPy is installed, else a
    :class:`Tensor`; it returns one value per entry), then Parlett's recurrence for the rest of ``f(T)`` and
    ``Z f(T) Zᴴ``; real where `A` is and the imaginary parts are negligible. Where eigenvalues gather (within 0.1 of
    the largest modulus, or of 1), Davies and Higham's Schur–Parlett algorithm: the Schur form reordered by clusters,
    each cluster's block by Cauchy's integral on a circle about it (`func` is then called again with those circles'
    points, more of them where an integral's own estimate asks: it must be analytic on their disks), the rest by the
    block recurrence; where no circle can keep the other clusters out, or the estimate stays past ``1e3 eps``, the
    clusters within 0.01 instead. A Hermitian `A`:
    ``V f(Λ) Vᴴ`` from its eigendecomposition. With ``disp=True`` (the default) a likely inaccurate result is reported
    on standard output, as SciPy's; with ``disp=False`` ``(F, err)`` is returned, `err` the estimate of
    ``‖error‖₁ / ‖A‖₁``.

    Raises:
        LinAlgError: if `A` is not square or holds an infinite or NaN entry, or `func` does not return one value per
            point.
    """
    A = A if isinstance(A, Tensor) else asarray(A)
    try:
        import numpy as _np
    except ImportError:  # pragma: no cover
        _np = None

    def evaluate(points):  # f at the points funm asks for: a NumPy array of them where NumPy is installed
        argument = _np.asarray(points) if _np is not None else points
        values = [complex(x) for x in _flat(asarray(func(argument)).tolist())]
        return asarray(values, "complex128") if values else zeros((0,), "complex128")

    try:
        f, err = _numerical.funm_call(A, evaluate)
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
_eig_left = _linalg(_numerical._eig_left, arrays=1)
_geneig = _linalg(_numerical._geneig, arrays=2)


def _eigenvalues_of(alpha, beta, homogeneous):
    """``alpha / beta`` as ``scipy.linalg``'s ``_make_eigvals`` gives them: ∞ where only β is 0, NaN where both are
    (complex NaN where an α is not real); the pair ``(alpha, beta)`` stacked where `homogeneous`."""
    if homogeneous:
        return stack([alpha, beta])
    alphas, betas = alpha.tolist(), beta.tolist()
    real = _builtins.all(complex(x).imag == 0 for x in alphas)
    values = [complex(x) / complex(y) if y != 0 else complex(_math.inf, 0) if x != 0
              else complex(_math.nan, 0 if real else _math.nan) for x, y in zip(alphas, betas)]
    return asarray(values, alpha.dtype)
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
    if _builtins.all(x.dtype in ("float32", "complex64") for x in [a, b, q, r] + [g for g in given if g is not None]):
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


def eig(a, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True,
        homogeneous_eigvals=False):
    """The eigenvalues and right eigenvectors (columns) of a general square matrix, or of each of a stack, as
    ``numpy.linalg.eig``: ``a @ v[:, j] == w[j] * v[:, j]``, each vector of unit norm with its component of largest
    modulus real. Complex whatever the eigenvalues, as NumPy 2.5's (complex64 for a single-precision matrix); in the
    order LAPACK's ``dgeev`` and ``zgeev`` give them (balancing, Hessenberg reduction, shifted QR), a real matrix's
    conjugate pairs together, the positive imaginary part first.

    With ``scipy.linalg.eig``'s arguments (a matrix): `left` the left eigenvectors too (``vl[:, j].conj() @ a ==
    w[j] * vl[:, j].conj()``, as ``geev``'s), `right` False for none of the right ones, `b` the generalized problem
    ``a @ v = w * b @ v`` by the complex QZ (``w = alpha / beta``, ∞ where only β is 0, NaN where both are; each vector
    of unit 2-norm, as SciPy normalises ``ggev``'s), `homogeneous_eigvals` the pairs ``(alpha, beta)`` stacked; the
    answer ``w``, ``(w, vl)``, ``(w, vr)`` or ``(w, vl, vr)`` as SciPy's. The vectors are complex always, where SciPy's
    are real for a real matrix with real eigenvalues. `overwrite_a`, `overwrite_b` and `check_finite` are accepted (an
    infinite or NaN entry raises always).
    """
    del overwrite_a, overwrite_b, check_finite
    if b is None and not left and right and not homogeneous_eigvals:
        return _eig(a)
    a = _tensor(a)
    if b is None:
        if left:
            w, vl, vr = _eig_left(a)
        elif right:
            (w, vr), vl = _eig(a), None
        else:
            w, vl, vr = _eigvals(a), None, None
        if homogeneous_eigvals:
            w = stack([w, ones(w.shape, w.dtype)])
    else:
        b = _tensor(b)
        if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("expected square matrix")
        if a.shape != b.shape:
            raise ValueError("a and b must have the same shape")
        alpha, beta, vl, vr = _geneig(a, b, left, right)
        w = _eigenvalues_of(alpha, beta, homogeneous_eigvals)
    if not (left or right):
        return w
    if left:
        return (w, vl, vr) if right else (w, vl)
    return w, vr


def eigvals(a, b=None, overwrite_a=False, check_finite=True, homogeneous_eigvals=False):
    """The eigenvalues alone of a general square matrix, or of each of a stack, as ``numpy.linalg.eigvals``: complex
    whatever they are, as NumPy 2.5's; the same values as ``eig``'s, to the bit. With ``scipy.linalg.eigvals``'s
    arguments: `b` the generalized problem's (:func:`eig`'s), `homogeneous_eigvals` the pairs ``(alpha, beta)``."""
    del overwrite_a, check_finite
    if b is None and not homogeneous_eigvals:
        return _eigvals(a)
    return eig(a, b, left=False, right=False, homogeneous_eigvals=homogeneous_eigvals)


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
_lu_full = _linalg(_numerical.lu_full, arrays=1)
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


def lu(a, permute_l=False, overwrite_a=False, check_finite=True, p_indices=False):
    """The pivoted LU factorisation ``a = P @ L @ U`` of an m × n matrix, as ``scipy.linalg.lu``: `L` m × k lower
    trapezoidal with a unit diagonal, `U` k × n upper trapezoidal, ``k = min(m, n)``; `P` the m × m permutation matrix,
    or with ``p_indices=True`` the rows themselves (``a == L[P] @ U``, an int list where SciPy's is an array); with
    ``permute_l=True`` the pair ``(P @ L, U)``. A singular matrix factorises (a zero pivot leaves its column of `L`
    zero). `overwrite_a` and `check_finite` are accepted (an infinite or NaN entry raises always).

    Raises:
        LinAlgError: if `a` is not a matrix or holds an infinite or NaN entry.
    """
    del overwrite_a, check_finite
    a = _tensor(a)
    rows, lower, upper = _lu_full(a)
    where = [0] * len(rows)  # where row r of a lies in L U: P's inverse
    for i, r in enumerate(rows):
        where[r] = i
    if permute_l:
        return (lower[where] if a.size else lower), upper
    if p_indices:
        return (where if a.size else []), lower, upper
    dtype = "float32" if a.dtype in _SINGLE else "float64"
    m = len(rows) if a.size else 0  # SciPy's P of an empty matrix is 0 × 0 whatever its rows
    return (eye(m, dtype=dtype)[:, rows] if m else zeros((0, 0), dtype)), lower, upper  # P[i, j] = 1 where rows[j] == i


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
        return int(_builtins.any(value != 0 for value in (values if A.ndim == 1 else [values])))
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
    diagonal pivoting (LAPACK's ``sytrf`` and ``hetrf``, by panels from order 384) on the lower triangle, or on the
    upper with ``lower=False``; only that triangle is read. Returns ``(lu, d, perm)``: `d` block diagonal with blocks of order 1 and 2, ``lu[perm]``
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
    """`t` with its axes in `order` (a permutation): through einsum, which copies; a bool one's elements moved alone."""
    letters = _labels(len(t.shape), "permute")
    try:
        return _numerical.einsum(f"{letters}->{''.join(letters[a] for a in order)}", t)
    except TypeError:  # a bool tensor, which einsum's arithmetic refuses
        return _numerical._permute(t, order)


def _matmul_stack(a, b):
    """``a @ b`` for two stacks of one shape (..., m, k) and (..., k, n), or two matrices."""
    if len(a.shape) == 2 and len(b.shape) == 2:
        return a @ b
    labels = _labels(len(a.shape) + 1, "matmul")  # the batch's, then the matrices' three, apart from them
    batch, (y, z, w) = labels[:-3], labels[-3:]
    return _einsum(f"{batch}{y}{z},{batch}{z}{w}->{batch}{y}{w}", a, b)


def conj(x):
    """The complex conjugate of every element, as ``numpy.conj``; a real or bool tensor's own elements."""
    x = _tensor(x)
    try:
        return _numerical._unary("conj", x)
    except TypeError:
        if x.dtype == "bool":  # the identity, as NumPy's
            return x
        raise


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
    if order == -_math.inf and _builtins.any(x.shape[a] == 0 for a in axes):  # the smallest of nothing, as NumPy's minimum
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
    if _builtins.any(len(operand.shape) != 2 for operand in operands):
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
    given. A bool tensor's trace counts its trues, a float64 as :func:`sum`'s (NumPy's an integer)."""
    x = _tensor(x)
    if len(x.shape) < 2:
        raise _ShapeError("trace: a matrix or a stack of matrices (rank ≥ 2)")
    if x.dtype == "bool":
        x = asarray(x, "float64")
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
    if x.dtype == "bool":  # its entries, through float64 exactly
        return asarray(_numerical.einsum(f"{batch}{along}{along}->{batch}{along}", asarray(block, "float64")), "bool")
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
    if len(left) != len(right) or _builtins.any(a.shape[i] != b.shape[j] for i, j in zip(left, right)):
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
    ``"float32"``, ``"complex64"`` or ``"bool"`` (true where nonzero); by default an array's own for float32,
    complex64 and complex128, else float64 (bool and integers included), and complex128 for a complex number or list.
    A complex input is never cast to a real dtype (that would discard the imaginary part).

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
        elif (kind is float or kind is int) and (dtype := self.dtype) != "bool":  # a real Python number is weak: it
            a, b = self, asarray(other, dtype)                                     # takes the tensor's dtype
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
        if not isinstance(result, Tensor):  # a bool's dtype its own (any, all; a mask's max), a mask's sum float64
            return full(shape, result, None if isinstance(result, bool) or a.dtype == "bool" else a.dtype)
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
any = _reduction("any", """Whether any element is true (nonzero), or any along `axis` (a bool Tensor without it), as
``numpy.any``.""")
all = _reduction("all", """Whether every element is true (nonzero), or every one along `axis` (a bool Tensor without
it), as ``numpy.all``.""")
def _dtype_name(dtype):
    """A dtype's name as a Tensor takes it: a string as it is, ``numpy.float64`` or ``numpy.dtype("float32")`` by name,
    Python's ``float``, ``complex`` and ``bool`` as NumPy names them."""
    if isinstance(dtype, str):
        return dtype
    if dtype in (float, complex, bool):
        return {float: "float64", complex: "complex128", bool: "bool"}[dtype]
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
Tensor.any = _reduction_method(any)
Tensor.all = _reduction_method(all)


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
    """`dtype` checked, one of float64, complex128, float32, complex64 and bool; ``None``: complex128 for a complex
    `value`, bool for a bool one, else float64, as NumPy infers it."""
    if dtype is None:
        return "complex128" if isinstance(value, complex) else "bool" if isinstance(value, bool) else "float64"
    dtype = _dtype_name(dtype)
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {_DTYPES}, not {dtype!r}")
    if dtype in ("float64", "float32") and isinstance(value, complex):
        raise TypeError(f"full: a complex value does not fit a {dtype} tensor")
    return dtype


def full(shape, value, dtype=None):
    """A Tensor of `shape` whose every element is `value`, as ``numpy.full``; complex128 for a complex value, bool for
    a bool one and float64 otherwise unless `dtype` says (float64, complex128, float32, complex64 or bool)."""
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


def astype(x, dtype, /):
    """`x` in `dtype` (float64, complex128, float32, complex64 or bool), as ``numpy.astype``: a bool is 0 or 1, a number
    is true where it is nonzero (either part of a complex one); a complex `x` has no real dtype (its imaginary part
    would be lost). The same Tensor when it has `dtype` already: a Tensor is never written into."""
    return asarray(x, _dtype_name(dtype))


Tensor.astype = lambda self, dtype: asarray(self, _dtype_name(dtype))
Tensor.astype.__doc__ = astype.__doc__


def where(condition, x, y, /):
    """`x`'s element where `condition` holds, else `y`'s, the three broadcast, as ``numpy.where``: `condition` a bool
    Tensor or what :func:`asarray` reads (a number true where nonzero); `x` and `y` promoted to one dtype, the
    answer's, as arithmetic promotes them."""
    condition = condition if isinstance(condition, Tensor) else asarray(condition)
    if condition.dtype != "bool":
        condition = asarray(condition, "bool")
    x, y = _same_dtype(x, y)
    return _native(_numerical._where, condition, x, y)


def _classifier(name, doc):
    def function(x, /):
        return _numerical._classify(name, x if isinstance(x, Tensor) else asarray(x))

    function.__name__ = name
    function.__doc__ = doc
    return function


isnan = _classifier("isnan", "Whether each element is a NaN (either part of a complex one), as ``numpy.isnan``: a "
                             "bool Tensor.")
isinf = _classifier("isinf", "Whether each element is infinite (either part of a complex one), as ``numpy.isinf``: "
                             "a bool Tensor.")
isfinite = _classifier("isfinite", "Whether each element is finite (both parts of a complex one), as "
                                   "``numpy.isfinite``: a bool Tensor.")


def _compared(op, name):
    def function(x1, x2, /):
        return _comparison(op)(x1 if isinstance(x1, Tensor) else asarray(x1), x2)

    function.__name__ = name
    function.__doc__ = f"``x1 {op} x2`` element by element, broadcast, as ``numpy.{name}``: a bool Tensor."
    return function


equal = _compared("==", "equal")
not_equal = _compared("!=", "not_equal")
less = _compared("<", "less")
less_equal = _compared("<=", "less_equal")
greater = _compared(">", "greater")
greater_equal = _compared(">=", "greater_equal")


def _truth(x):
    """`x` as a bool Tensor, true where it is nonzero, as NumPy's logical functions read their operands."""
    x = x if isinstance(x, Tensor) else asarray(x)
    return x if x.dtype == "bool" else asarray(x, "bool")


def logical_and(x1, x2, /):
    """Whether both elements are true (nonzero), element by element, broadcast, as ``numpy.logical_and``."""
    return _native(_numerical._logical, "&", _truth(x1), _truth(x2))


def logical_or(x1, x2, /):
    """Whether either element is true (nonzero), element by element, broadcast, as ``numpy.logical_or``."""
    return _native(_numerical._logical, "|", _truth(x1), _truth(x2))


def logical_xor(x1, x2, /):
    """Whether exactly one element is true (nonzero), element by element, broadcast, as ``numpy.logical_xor``."""
    return _native(_numerical._logical, "^", _truth(x1), _truth(x2))


def logical_not(x, /):
    """Whether each element is false (zero), as ``numpy.logical_not``."""
    return _numerical._invert(_truth(x))


def power(a, b):
    """`a` to the power `b` element by element, broadcast, as ``numpy.power``."""
    a, b = _same_dtype(a, b)
    if not _numpy_loops_measured:
        _numpy_loops()
    return _native(_numerical._binary, "^", a, b)


def real(x, /):
    """The real part of every element, of the parts' precision, as ``numpy.real``; a real or bool tensor itself."""
    x = _tensor(x)
    return _numerical._unary("real", x) if x.dtype.startswith("complex") else x


def imag(x, /):
    """The imaginary part of every element, of the parts' precision, as ``numpy.imag``; zeros of a real or bool
    tensor's dtype."""
    x = _tensor(x)
    return _numerical._unary("imag", x) if x.dtype.startswith("complex") else zeros(x.shape, x.dtype)


def transpose(a, axes=None):
    """`a` with its axes in the order `axes` (a permutation of them, negative ones counted from the end), reversed
    by default, as ``numpy.transpose``: a copy."""
    a = _tensor(a)
    if axes is None:
        return a.T
    rank = len(a.shape)
    order = [_operator.index(axis) for axis in axes]
    order = [axis + rank if axis < 0 else axis for axis in order]
    if sorted(order) != list(range(rank)):
        raise ValueError(f"transpose: axes {tuple(axes)} are not a permutation of a rank-{rank} tensor's")
    return _permuted(a, order) if order != list(range(rank)) else a


def squeeze(a, axis=None):
    """`a` without its axes of extent 1, or without the ones `axis` names (an integer or a tuple), as
    ``numpy.squeeze``.

    Raises:
        ValueError: if an axis `axis` names has another extent.
    """
    a = _tensor(a)
    shape = a.shape
    if axis is None:
        drop = {k for k, extent in enumerate(shape) if extent == 1}
    else:
        drop = set(_resolved_axes(a, axis, "squeeze"))
        if _builtins.any(shape[k] != 1 for k in drop):
            raise ValueError("squeeze: cannot select an axis to squeeze out which has size not equal to one")
    return reshape(a, tuple(extent for k, extent in enumerate(shape) if k not in drop)) if drop else a


def _rebuilt(data, shape, dtype, byteorder):
    """The Tensor a pickle holds (:meth:`Tensor.__reduce__`): its elements' bytes, written in `byteorder`."""
    return _numerical._frombytes(data, shape, dtype, byteorder != _sys.byteorder)


def _moved_last(t, axis1, axis2, name):
    """`t` with its axes `axis1` and `axis2` moved last, in that order, the others kept in theirs."""
    rank = len(t.shape)
    if rank < 2:
        raise _ShapeError(f"{name}: a tensor of rank 2 at least")
    first, second = _resolved_axes(t, (axis1, axis2), name)
    order = [k for k in range(rank) if k not in (first, second)] + [first, second]
    return _permuted(t, order) if order != list(range(rank)) else t


def _diagonal_method(self, offset=0, axis1=0, axis2=1):
    """The `offset` diagonal of the matrices the axes `axis1` and `axis2` hold, as ``ndarray.diagonal``: along the
    other axes' end."""
    return diagonal(_moved_last(self, axis1, axis2, "diagonal"), offset=offset)


def _trace_method(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """The sum of the `offset` diagonal of the matrices the axes `axis1` and `axis2` hold, as ``ndarray.trace``."""
    if out is not None:
        raise TypeError("trace: a Tensor takes no `out` (it is never written into)")
    return trace(_moved_last(self, axis1, axis2, "trace"), offset=offset,
                 dtype=None if dtype is None else _dtype_name(dtype))


def _dot(self, b):
    """``ndarray.dot``: the matrix product for vectors and matrices, else the sum over the last axis of `self` and
    the second to last of `b` (its only one for a vector)."""
    b = _tensor(b)
    if len(self.shape) <= 2 and len(b.shape) <= 2 and self.shape and b.shape:
        return _matmul(self, b)
    if not self.shape or not b.shape:
        return self * b
    return tensordot(self, b, axes=([len(self.shape) - 1], [_builtins.max(len(b.shape) - 2, 0)]))


def _item(self, *args):
    """The one element of a one-element tensor, or the element at a flat index or a multi-index, as a Python
    number, as ``ndarray.item``."""
    if not args:
        if self.size != 1:
            raise ValueError("can only convert an array of size 1 to a Python scalar")
        return reshape(self, -1)[0]
    if len(args) == 1 and not isinstance(args[0], tuple):
        index = _operator.index(args[0])
        if not -self.size <= index < self.size:
            raise IndexError(f"index {index} is out of bounds for size {self.size}")
        return reshape(self, -1)[index]
    return self[args[0] if len(args) == 1 else args]


def _swapaxes(self, axis1, axis2):
    """`self` with its axes `axis1` and `axis2` exchanged, as ``ndarray.swapaxes``: a copy."""
    rank = len(self.shape)
    first, second = (_resolved_axes(self, axis, "swapaxes")[0] for axis in (axis1, axis2))
    order = list(range(rank))
    order[first], order[second] = second, first
    return _permuted(self, order) if first != second else self


def _to_device(self, device, /, *, stream=None):
    """The tensor itself on the CPU, the only device a Tensor has (the array API's ``to_device``)."""
    del stream
    if device != "cpu":
        raise ValueError(f"to_device: a Tensor lives on the CPU, not {device!r}")
    return self


def _contains(self, value):
    """``value in t``: whether an element equals `value`, as NumPy's ``(t == value).any()``."""
    return bool((self == value).any())


_ITEMSIZE = {"float64": 8, "complex128": 16, "float32": 4, "complex64": 8, "bool": 1}

Tensor.__reduce__ = lambda self: (_rebuilt, (_numerical._tobytes(self), self.shape, self.dtype, _sys.byteorder))
Tensor.__copy__ = lambda self: self  # a Tensor is never written into: itself is its copy
Tensor.__deepcopy__ = lambda self, memo: self
Tensor.copy = lambda self: self
Tensor.copy.__doc__ = """The tensor itself, as ``ndarray.copy`` would give: a Tensor is never written into."""
Tensor.tobytes = lambda self: _numerical._tobytes(self)
Tensor.tobytes.__doc__ = """The elements' bytes, row-major, in this machine's byte order, as ``ndarray.tobytes``."""
Tensor.transpose = lambda self, *axes: transpose(self, axes[0] if len(axes) == 1 and not isinstance(axes[0], int)
                                                  else (axes or None))
Tensor.swapaxes = _swapaxes
Tensor.mT = property(matrix_transpose)
Tensor.flatten = Tensor.ravel = lambda self: reshape(self, -1)
Tensor.squeeze = lambda self, axis=None: squeeze(self, axis)
Tensor.squeeze.__doc__ = squeeze.__doc__
Tensor.item = _item
Tensor.real = property(real)
Tensor.imag = property(imag)
Tensor.conj = Tensor.conjugate = lambda self: conj(self)
Tensor.conj.__doc__ = conj.__doc__
Tensor.diagonal = _diagonal_method
Tensor.trace = _trace_method
Tensor.dot = _dot
Tensor.itemsize = property(lambda self: _ITEMSIZE[self.dtype])
Tensor.nbytes = property(lambda self: _ITEMSIZE[self.dtype] * self.size)
Tensor.device = property(lambda self: "cpu")
Tensor.to_device = _to_device
Tensor.__contains__ = _contains


__all__ += ["diagonal", "matrix_transpose", "tensorsolve", "tensorinv", "real", "imag", "transpose", "squeeze"]
__all__ += ["any", "all", "astype", "where", "isnan", "isinf", "isfinite", "equal", "not_equal", "less", "less_equal",
            "greater", "greater_equal", "logical_and", "logical_or", "logical_xor", "logical_not"]
__all__ += ["sum", "mean", "max", "min", "reshape", "full", "zeros", "ones", "eye", "arange", "linspace",
            "exp", "log", "sqrt", "sin", "cos", "tan", "tanh", "absolute", "power"]


# --- scipy.linalg: special matrices, structure, subspaces and matrix functions over the decompositions ------------


class LinAlgWarning(RuntimeWarning):
    """The warning ``scipy.linalg.LinAlgWarning`` names: an ill-conditioned operation whose answer may be
    inaccurate."""


def _grid(n):
    """The row and column indices of an `n × n` matrix, a column and a row of float64, to broadcast."""
    k = arange(n)
    return reshape(k, (n, 1)), k


def hilbert(n):
    """The `n × n` Hilbert matrix, ``H[i, j] = 1 / (i + j + 1)``, as ``scipy.linalg.hilbert``."""
    n = _operator.index(n)
    values = 1.0 / (1.0 + arange(_builtins.max(2 * n - 1, 0)))
    return hankel(values[:n], values[_builtins.max(n - 1, 0):]) if n else zeros((0, 0))


def _exact(rows, n):
    """The float64 `n × n` matrix of the Python integers `rows` (a function of i and j), each rounded once."""
    return asarray([[float(rows(i, j)) for j in range(n)] for i in range(n)]) if n else zeros((0, 0))


def invhilbert(n, exact=False):
    """The inverse of the `n × n` Hilbert matrix, as ``scipy.linalg.invhilbert``: its integer entries computed
    exactly and each rounded once to float64 (SciPy rounds the binomials it multiplies where `exact` is false, and
    gives integers where it is true; float64 holds them exactly up to 2**53, about `n = 14`)."""
    del exact
    n = _operator.index(n)
    comb = _math.comb
    return _exact(lambda i, j: (-1) ** (i + j) * (i + j + 1) * comb(n + i, n - j - 1) * comb(n + j, n - i - 1)
                  * comb(i + j, _builtins.min(i, j)) ** 2, n)


def _pascal_kind(kind):
    if kind not in ("symmetric", "lower", "upper"):
        raise ValueError("kind must be 'symmetric', 'lower', or 'upper'")


def pascal(n, kind="symmetric", exact=True):
    """The `n × n` Pascal matrix, as ``scipy.linalg.pascal``: binomial coefficients, ``lower`` ``C(i, j)``, ``upper``
    its transpose, ``symmetric`` ``C(i + j, i)``; float64, each entry exact and rounded once (exact integers up to
    2**53)."""
    del exact
    _pascal_kind(kind)
    n = _operator.index(n)
    comb = _math.comb
    if kind == "lower":
        return _exact(lambda i, j: comb(i, j), n)
    if kind == "upper":
        return _exact(lambda i, j: comb(j, i), n)
    return _exact(lambda i, j: comb(i + j, i), n)


def invpascal(n, kind="symmetric", exact=True):
    """The inverse of the `n × n` Pascal matrix, as ``scipy.linalg.invpascal``: float64, each entry exact and
    rounded once."""
    del exact
    _pascal_kind(kind)
    n = _operator.index(n)
    comb = _math.comb
    if kind == "lower":
        return _exact(lambda i, j: (-1) ** (i - j) * comb(i, j), n)
    if kind == "upper":
        return _exact(lambda i, j: (-1) ** (j - i) * comb(j, i), n)

    def symmetric(i, j):
        i, j = _builtins.max(i, j), _builtins.min(i, j)
        return (-1) ** (i - j) * _builtins.sum(comb(i + k, k) * comb(i + k, i + k - j) for k in range(n - i))

    return _exact(symmetric, n)


def hadamard(n, dtype="float64"):
    """The `n × n` Hadamard matrix of Sylvester's construction (`n` a power of 2), as ``scipy.linalg.hadamard``: its
    ±1 entries in `dtype` (float64 by default, where SciPy's is an integer)."""
    n = _operator.index(n)
    if n < 1 or n & (n - 1):
        raise ValueError("n must be an positive integer, and n must be a power of 2")
    dtype = _dtype_of(None if dtype in (int, "int", "int64") else dtype)
    h = ones((1, 1), dtype)
    sylvester = asarray([[1.0, 1.0], [1.0, -1.0]], dtype)
    while h.shape[0] < n:
        size = 2 * h.shape[0]
        h = reshape(einsum("ij,kl->ikjl", sylvester, h), (size, size))
    return h


def helmert(n, full=False):
    """The `n × n` Helmert matrix, or its last `n - 1` rows unless `full`, as ``scipy.linalg.helmert``."""
    n = _operator.index(n)
    i, j = _grid(n)
    h = where(i > j, 1.0, 0.0) - where(i == j, i, 0.0)
    d = arange(n) * arange(1, n + 1)
    first = (arange(n) == 0)
    h = where(reshape(first, (n, 1)), 1.0, h)
    d = where(first, float(n), d)
    h = h / reshape(sqrt(d), (n, 1))
    return h if full else h[1:]


def leslie(f, s):
    """The Leslie matrix of fecundities `f` and survival rates `s` (one fewer), as ``scipy.linalg.leslie``: `f` its
    first row, `s` its subdiagonal."""
    f, s = _same_dtype(f, s)
    if len(f.shape) != 1 or len(s.shape) != 1 or f.shape[0] != s.shape[0] + 1:
        raise ValueError("Incorrect lengths for f and s. The length of s along the last axis must be one less than "
                         "the length of f.")
    if s.shape[0] == 0:
        raise ValueError("The length of s must be at least 1.")
    n = f.shape[0]
    i, j = _grid(n)
    survival = asarray(s.tolist() + [0.0], f.dtype)  # s[j] under column j, nothing under the last
    below = where(i == j + 1, reshape(survival, (1, n)), zeros((1, n), f.dtype))
    return where(i == 0, reshape(f, (1, n)), below)


def companion(a):
    """The companion matrix of the polynomial of coefficients `a` (highest degree first), as
    ``scipy.linalg.companion``: ``-a[1:] / a[0]`` its first row, ones on its subdiagonal."""
    (a,) = _vectors("companion", a)
    n = a.shape[0]
    if n < 2:
        raise ValueError("The length of `a` along the last axis must be at least 2.")
    if a[0] == 0:
        raise ValueError("The first coefficient(s) of `a` (i.e. elements of `a[..., 0]`) must not be zero.")
    first = -a[1:] / (1.0 * a[0])
    i, j = _grid(n - 1)
    return where(i == 0, reshape(first, (1, n - 1)), where(i == j + 1, 1.0, 0.0).astype(first.dtype))


def fiedler(a):
    """The symmetric Fiedler matrix of `a`, ``F[i, j] = |a[i] - a[j]|``, as ``scipy.linalg.fiedler`` (along the last
    axis of a stack)."""
    a = _tensor(a)
    if not a.shape:
        a = reshape(a, (1,))
    if a.size == 0:
        return zeros((0,))
    if a.size == 1:
        return zeros((1, 1))
    return absolute(a[..., :, None] - a[..., None, :])


def fiedler_companion(a):
    """The Fiedler companion matrix of the polynomial of coefficients `a` (highest degree first), as
    ``scipy.linalg.fiedler_companion``: pentadiagonal, its eigenvalues the polynomial's roots."""
    (a,) = _vectors("fiedler_companion", a)
    if a.size <= 2:
        return asarray([[-(a[-1] / a[0])]], a.dtype) if a.size == 2 else zeros((0,), a.dtype)
    if a[0] == 0:
        raise ValueError("Leading coefficient is zero.")
    dtype = a.dtype
    a = (a / a[0]).tolist()
    n = len(a) - 1
    c = [[0.0] * n for _ in range(n)]
    for i, j in zip(range(3, n, 2), range(1, n - 2, 2)):
        c[i][j] = 1.0
    for i, j, k in zip(range(2, n, 2), range(1, n - 1, 2), range(3, n + 1, 2)):
        c[i][j] = -a[k]
    for i, j in zip(range(0, n - 2, 2), range(2, n, 2)):
        c[i][j] = 1.0
    for i, j, k in zip(range(0, n - 1, 2), range(1, n, 2), range(2, n + 1, 2)):
        c[i][j] = -a[k]
    c[0][0], c[1][0] = -a[1], 1.0
    return asarray(c, dtype)


def convolution_matrix(a, n, mode="full"):
    """The Toeplitz matrix whose product with a vector of length `n` is its convolution with `a`, as
    ``scipy.linalg.convolution_matrix``: ``full`` (``len(a) + n - 1`` rows), ``same`` (``max(n, len(a))``) or
    ``valid`` (``max(n, len(a)) - min(n, len(a)) + 1``)."""
    n = _operator.index(n)
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    (a,) = _vectors("convolution_matrix", a)
    if a.size == 0:
        raise ValueError("len(a) must be at least 1.")
    if mode not in ("full", "valid", "same"):
        raise ValueError("'mode' argument must be one of ('full', 'valid', 'same')")
    values, zero = a.tolist(), 0.0
    az = values + [zero] * (n - 1)
    raz = values[::-1] + [zero] * (n - 1)
    if mode == "full":
        col0, row0 = az, raz[-n:]
    else:
        trim = _builtins.min(n, len(values)) - 1
        tb = trim // 2 if mode == "same" else trim
        te = trim - tb if mode == "same" else trim
        col0, row0 = az[tb:len(az) - te], raz[len(raz) - n - tb:len(raz) - tb]
    return toeplitz(asarray(col0, a.dtype), asarray(row0, a.dtype))


def dft(n, scale=None):
    """The `n × n` discrete Fourier transform matrix, ``exp(-2πi jk / n)``, as ``scipy.linalg.dft``: divided by
    ``sqrt(n)`` (``scale="sqrtn"``) or `n` (``"n"``)."""
    if scale not in (None, "sqrtn", "n"):
        raise ValueError(f"scale must be None, 'sqrtn', or 'n'; {scale!r} is not valid.")
    n = _operator.index(n)
    omegas = reshape(exp(-2j * _math.pi * arange(n) / n), (n, 1))
    m = omegas ** arange(n)
    if scale == "sqrtn":
        m = m / _math.sqrt(n)
    elif scale == "n":
        m = m / n
    return m


def khatri_rao(a, b):
    """The column-wise Kronecker product of `a` (m × k) and `b` (p × k), ``(m p) × k``, as
    ``scipy.linalg.khatri_rao``."""
    a, b = _same_dtype(a, b)
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("The both arrays should be 2-dimensional.")
    if a.shape[1] != b.shape[1]:
        raise ValueError("The number of columns for both arrays should be equal.")
    return reshape(a[:, None, :] * b[None, :, :], (a.shape[0] * b.shape[0], a.shape[1]))


def diagsvd(s, M, N):
    """The `M × N` matrix of the singular values `s` on its diagonal, as ``scipy.linalg.diagsvd`` (`s` of length `M`
    or `N`)."""
    (s,) = _vectors("diagsvd", s)
    M, N, k = _operator.index(M), _operator.index(N), s.shape[0]
    if k not in (M, N):
        raise ValueError("Length of s must be M or N.")
    i, j = _grid(_builtins.max(M, N))
    square = where(i == j, reshape(asarray(s.tolist() + [0.0] * (_builtins.max(M, N) - k), s.dtype), (1, -1)),
                   zeros((1, 1), s.dtype))
    return square[:M, :N]


def bandwidth(a):
    """The lower and upper bandwidths of the matrix `a` (the farthest diagonals below and above the main one holding
    a nonzero entry, 0 where none), as ``scipy.linalg.bandwidth``."""
    a = _tensor(a)
    if len(a.shape) != 2:
        raise ValueError("Input array must be a 2D array.")
    rows, cols = a.shape
    i, j = reshape(arange(rows), (rows, 1)), arange(cols)
    nonzero = a != 0
    lower = max(where(nonzero, i - j, 0.0)) if a.size else 0.0
    upper = max(where(nonzero, j - i, 0.0)) if a.size else 0.0
    return int(lower), int(upper)


def _symmetric(a, atol, rtol, hermitian):
    a = _tensor(a)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input array must be square.")
    other = matrix_transpose(a)
    if hermitian and a.dtype.startswith("complex"):
        other = conj(other)
    if atol is None and rtol is None:
        return bool(all(a == other))
    return bool(all(absolute(a - other) <= (0.0 if atol is None else atol)
                    + (0.0 if rtol is None else rtol) * absolute(other)))


def issymmetric(a, atol=None, rtol=None):
    """Whether the square matrix `a` equals its transpose: exactly, or within ``atol + rtol |aᵀ|`` where a tolerance is
    given, as ``scipy.linalg.issymmetric``."""
    return _symmetric(a, atol, rtol, False)


def ishermitian(a, atol=None, rtol=None):
    """Whether the square matrix `a` equals its conjugate transpose: exactly, or within ``atol + rtol |aᴴ|`` where a
    tolerance is given, as ``scipy.linalg.ishermitian``."""
    return _symmetric(a, atol, rtol, True)


_EPSILON = {"float64": 2.0 ** -52, "complex128": 2.0 ** -52, "float32": 2.0 ** -23, "complex64": 2.0 ** -23}


def null_space(A, rcond=None, *, overwrite_a=False, check_finite=True, lapack_driver="gesdd"):
    """An orthonormal basis of the null space of `A` (its columns), as ``scipy.linalg.null_space``: the right singular
    vectors of the singular values at most `rcond` times the largest (``eps max(M, N)`` by default)."""
    del overwrite_a, check_finite, lapack_driver
    u, s, vh = svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = _EPSILON[s.dtype] * _builtins.max(M, N)
    tol = (max(s) if s.size else 0.0) * rcond
    num = _builtins.sum(1 for value in s.tolist() if value > tol)
    return conj(matrix_transpose(vh[num:, :]))


def orth(A, rcond=None):
    """An orthonormal basis of the range of `A` (its columns), as ``scipy.linalg.orth``: the left singular vectors of
    the singular values above `rcond` times the largest (``eps max(M, N)`` by default)."""
    u, s, vh = svd(A, full_matrices=False)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = _EPSILON[s.dtype] * _builtins.max(M, N)
    tol = (max(s) if s.size else 0.0) * rcond
    num = _builtins.sum(1 for value in s.tolist() if value > tol)
    return u[:, :num]


def subspace_angles(A, B):
    """The principal angles between the column spaces of `A` and `B`, largest first, as
    ``scipy.linalg.subspace_angles`` (Knyazev and Argentati's algorithm, the sines for the small angles)."""
    A, B = _tensor(A), _tensor(B)
    if len(A.shape) != 2:
        raise ValueError(f"expected 2D array, got shape {A.shape}")
    QA = orth(A)
    if len(B.shape) != 2:
        raise ValueError(f"expected 2D array, got shape {B.shape}")
    if B.shape[0] != QA.shape[0]:
        raise ValueError(f"A and B must have the same number of rows, got {QA.shape[0]} and {B.shape[0]}")
    QB = orth(B)
    QA_H_QB = conj(matrix_transpose(QA)) @ QB
    sigma = svdvals(QA_H_QB).tolist()
    if QA.shape[1] >= QB.shape[1]:
        B = QB - QA @ QA_H_QB
    else:
        B = QA - QB @ conj(matrix_transpose(QA_H_QB))
    # the angles come largest first, their cosines sigma smallest first: angle k's cosine is sigma[::-1][k] (SciPy
    # builds the mask on sigma unreversed and sends the small angles through the arccosine)
    mask = [value ** 2 >= 0.5 for value in reversed(sigma)]
    sines = svdvals(B).tolist() if _builtins.any(mask) else [0.0] * len(sigma)
    theta = [_math.asin(_builtins.min(_builtins.max(sines[k], -1.0), 1.0)) if mask[k]
             else _math.acos(_builtins.min(_builtins.max(sigma[::-1][k], -1.0), 1.0)) for k in range(len(sigma))]
    return asarray(theta)


def orthogonal_procrustes(A, B, check_finite=True):
    """The unitary `R` nearest to mapping `A` onto `B` (minimising ``‖A R − B‖_F``) and the sum of the singular values
    of ``Aᴴ B``, as ``scipy.linalg.orthogonal_procrustes``."""
    del check_finite
    A, B = _same_dtype(A, B)
    if len(A.shape) != 2:
        raise ValueError(f"expected ndim to be 2, but observed {len(A.shape)}")
    if A.shape != B.shape:
        raise ValueError(f"the shapes of A and B differ ({A.shape} vs {B.shape})")
    u, w, vt = svd(matrix_transpose(matrix_transpose(B) @ conj(A)))
    return u @ vt, sum(w)


def pinvh(a, atol=None, rtol=None, lower=True, return_rank=False, check_finite=True):
    """The pseudo-inverse of the Hermitian matrix `a` from its eigendecomposition, as ``scipy.linalg.pinvh``: the
    eigenvalues above ``atol + rtol max|λ|`` inverted (``rtol`` ``max(M, N) eps`` by default), with the rank where
    `return_rank`."""
    del check_finite
    a = _tensor(a)
    s, u = eigh(a, UPLO="L" if lower else "U")
    values = s.tolist()
    largest = _builtins.max((abs(v) for v in values), default=0.0)
    atol = 0.0 if atol is None else atol
    rtol = _builtins.max(a.shape) * _EPSILON[u.dtype] if rtol is None else rtol
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol values must be positive.")
    kept = absolute(s) > atol + largest * rtol
    inverted = where(kept, 1.0 / where(kept, s, 1.0), 0.0)  # a dropped eigenvalue's column weighs nothing
    B = (u * inverted.astype(u.dtype)) @ conj(matrix_transpose(u))
    return (B, int(sum(kept))) if return_rank else B


def _maybe_real(A, B):
    """`B` real where `A` is and `B`'s imaginary parts are within SciPy's tolerance of 0, as ``_maybe_real``."""
    if not A.dtype.startswith("complex") and B.dtype.startswith("complex"):
        tol = 1e3 * 2.0 ** -23 if B.dtype == "complex64" else 1e6 * 2.0 ** -52
        if all(absolute(B.imag) <= tol):
            return B.real
    return B


def _square(name, A):
    A = _tensor(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name}: expected a square matrix")
    return A


def cosm(A):
    """The matrix cosine, as ``scipy.linalg.cosm``: the real part of ``expm(iA)`` for a real `A`."""
    A = _square("cosm", A)
    if A.dtype.startswith("complex"):
        return 0.5 * (expm(1j * A) + expm(-1j * A))
    return expm(1j * A).real


def sinm(A):
    """The matrix sine, as ``scipy.linalg.sinm``: the imaginary part of ``expm(iA)`` for a real `A`."""
    A = _square("sinm", A)
    if A.dtype.startswith("complex"):
        return -0.5j * (expm(1j * A) - expm(-1j * A))
    return expm(1j * A).imag


def tanm(A):
    """The matrix tangent, ``cosm(A)⁻¹ sinm(A)``, as ``scipy.linalg.tanm``: the exponentials :func:`cosm` and
    :func:`sinm` take, computed once for both."""
    A = _square("tanm", A)
    if A.dtype.startswith("complex"):
        ep, em = expm(1j * A), expm(-1j * A)
        return _maybe_real(A, solve(0.5 * (ep + em), -0.5j * (ep - em)))
    e = expm(1j * A)
    return _maybe_real(A, solve(e.real, e.imag))


def coshm(A):
    """The matrix hyperbolic cosine, ``(expm(A) + expm(-A)) / 2``, as ``scipy.linalg.coshm``."""
    A = _square("coshm", A)
    return _maybe_real(A, 0.5 * (expm(A) + expm(-A)))


def sinhm(A):
    """The matrix hyperbolic sine, ``(expm(A) − expm(-A)) / 2``, as ``scipy.linalg.sinhm``."""
    A = _square("sinhm", A)
    return _maybe_real(A, 0.5 * (expm(A) - expm(-A)))


def tanhm(A):
    """The matrix hyperbolic tangent, ``coshm(A)⁻¹ sinhm(A)``, as ``scipy.linalg.tanhm``: the exponentials
    :func:`coshm` and :func:`sinhm` take, computed once for both."""
    A = _square("tanhm", A)
    ep, em = expm(A), expm(-A)
    return _maybe_real(A, solve(_maybe_real(A, 0.5 * (ep + em)), _maybe_real(A, 0.5 * (ep - em))))


def signm(A, disp=True):
    """The matrix sign function, as ``scipy.linalg.signm``: :func:`funm` of the sign of each eigenvalue's real part
    (0 within ``1e3 eps`` of the largest), then, where its error estimate passes ``1e3 eps``, Newton's iteration
    ``S ← (S + S⁻¹) / 2`` from `A` shifted by half the reciprocal of its largest singular value; with ``disp=False``
    the estimate beside the answer."""
    A = _square("signm", A)

    def rounded_sign(x):  # SciPy's: 0 where |Re λ| is not past c, NumPy's complex maximum ordering (Re, then Im)
        values = [complex(v) for v in _flat(asarray(x).tolist())]
        c = 1e3 * 2.0 ** -52 * _builtins.max(values, key=lambda v: (v.real, v.imag))
        return [(1.0 if v.real > 0 else -1.0) if (abs(v.real), 0.0) > (c.real, c.imag) else 0.0 for v in values]

    result, errest = funm(A, rounded_sign, disp=False)
    errtol = 1e3 * (2.0 ** -23 if result.dtype in _SINGLE else 2.0 ** -52)
    if errest < errtol:
        return result if disp else (result, errest)
    vals = svdvals(A)
    c = 0.5 / max(vals)
    S0 = A + c * eye(A.shape[0], A.dtype)
    prev_errest = errest
    for _ in range(100):
        S0 = 0.5 * (S0 + inv(S0))
        Pp = 0.5 * (S0 @ S0 + S0)
        errest = norm(Pp @ Pp - Pp, 1)
        if errest < errtol or prev_errest == errest:
            break
        prev_errest = errest
    if disp:
        if not _math.isfinite(errest) or errest >= errtol:
            print("signm result may be inaccurate, approximate err =", errest)
        return S0
    return S0, errest


__all__ += ["LinAlgWarning", "hilbert", "invhilbert", "pascal", "invpascal", "hadamard", "helmert", "leslie", "companion",
            "fiedler", "fiedler_companion", "convolution_matrix", "dft", "khatri_rao", "diagsvd", "bandwidth",
            "issymmetric", "ishermitian", "null_space", "orth", "subspace_angles", "orthogonal_procrustes", "pinvh",
            "cosm", "sinm", "tanm", "coshm", "sinhm", "tanhm", "signm"]


# --- scipy.linalg: Hessenberg, balancing, tridiagonal eigenproblems, banded Cholesky, RQ and the real-complex forms ---

_qz = _linalg(_numerical.qz, arrays=2)


def _qz_output(output):
    if output not in ("real", "complex", "r", "c"):
        raise ValueError("argument must be 'real', or 'complex'")
    return output in ("complex", "c")


def qz(A, B, output="real", lwork=None, sort=None, overwrite_a=False, overwrite_b=False, check_finite=True):
    """The generalized Schur decomposition of the pencil ``(A, B)``, ``A = Q AA Zᴴ`` and ``B = Q BB Zᴴ``, as
    ``scipy.linalg.qz`` (LAPACK's ``xgges``): `Q` and `Z` orthogonal (unitary), `BB` upper triangular with a
    non-negative diagonal, `AA` quasi-triangular for a real pencil (a 2 × 2 block for each conjugate pair of
    eigenvalues, `BB` diagonal there) and triangular with ``output="complex"`` or for a complex one. The operands are
    first scaled into range as ``xgges`` scales them. Returns ``(AA, BB, Q, Z)``. `lwork`, `overwrite_a`, `overwrite_b`
    and `check_finite` are accepted and not needed.

    Raises:
        ValueError: if `sort` is given (as SciPy, which leaves it to :func:`ordqz`) or `output` is another.
        LinAlgError: if `A` and `B` are not square and of one order, an entry is infinite or NaN, or the iterations do
            not converge.
    """
    del lwork, overwrite_a, overwrite_b, check_finite
    if sort is not None:
        raise ValueError("The 'sort' input of qz() has to be None and will be removed in a future release. Use ordqz instead.")
    AA, BB, _, _, Q, Z = _qz(A, B, _qz_output(output), None)
    return AA, BB, Q, Z


def _lhp(x, y):
    return y != 0 and (x / y).real < 0.0


def _rhp(x, y):
    return y != 0 and (x / y).real > 0.0


def _iuc(x, y):
    return y != 0 and abs(x / y) < 1.0


def _ouc(x, y):
    return x != 0 if y == 0 else abs(x / y) > 1.0


_ORDQZ_SORTS = {"lhp": _lhp, "rhp": _rhp, "iuc": _iuc, "ouc": _ouc}


def _ordqz_select(sort):
    """The selection the native ordqz calls with the eigenvalues' `alpha` and `beta` (Tensors): one of SciPy's four,
    taken element by element as SciPy's (a zero β none of them but "ouc", which takes a nonzero α over it), or the
    callable `sort` given them as NumPy arrays where NumPy is installed, as SciPy gives them."""
    if callable(sort):
        def select(alpha, beta):
            try:
                import numpy as np
            except ImportError:  # pragma: no cover
                chosen = sort(alpha, beta)
            else:
                chosen = np.asarray(sort(np.asarray(alpha), np.asarray(beta))).tolist()
            return _flat(asarray(chosen).tolist()) if isinstance(chosen, Tensor) else list(chosen)
        return select
    if not isinstance(sort, str) or sort not in _ORDQZ_SORTS:
        raise ValueError("sort parameter must be None, a callable, or one of ('lhp','rhp','iuc','ouc')")
    test = _ORDQZ_SORTS[sort]
    return lambda alpha, beta: [test(complex(x), complex(y)) for x, y in zip(alpha.tolist(), beta.tolist())]


def ordqz(A, B, sort="lhp", output="real", overwrite_a=False, overwrite_b=False, check_finite=True):
    """The generalized Schur decomposition of :func:`qz` with the eigenvalues `sort` selects moved to the leading
    block, as ``scipy.linalg.ordqz`` (LAPACK's ``xgges`` then ``xtgsen``): ``"lhp"`` (real part of α / β below 0),
    ``"rhp"`` (above 0), ``"iuc"`` (modulus below 1), ``"ouc"`` (above 1, an infinite one included), or a callable
    given the arrays `alpha` and `beta` and returning one truth per eigenvalue; a conjugate pair moved whole where
    either of its two is selected. Returns ``(AA, BB, alpha, beta, Q, Z)``, the eigenvalues ``alpha / beta`` (`alpha`
    complex, `beta` real for a real pencil's real form).

    Raises:
        ValueError: for another `sort` or `output`; LinAlgError (a ValueError) if two blocks cannot be exchanged, or
            as :func:`qz`.
    """
    del overwrite_a, overwrite_b, check_finite
    complex_output = _qz_output(output)
    return _qz(A, B, complex_output, _ordqz_select(sort))


_cholesky_banded = _linalg(_numerical.cholesky_banded, arrays=1)
_cho_solve_banded = _linalg(_numerical.cho_solve_banded, arrays=2)
_hessenberg = _linalg(_numerical.hessenberg, arrays=1)
_matrix_balance = _linalg(_numerical.matrix_balance, arrays=1)
_tridiagonal_eigen = _linalg(_numerical.tridiagonal_eigen)
_tridiagonal_select = _linalg(_numerical.tridiagonal_select)
_rsf2csf = _linalg(_numerical.rsf2csf, arrays=2)


def hessenberg(a, calc_q=False, overwrite_a=False, check_finite=True):
    """The upper Hessenberg form `H` of the square `a`, ``A = Q H Qᴴ``, and with `calc_q` the unitary `Q`, as
    ``scipy.linalg.hessenberg`` (Householder reflections, LAPACK's ``gehrd``, no balancing); a matrix of order 2 or
    less is its own, `Q` the float64 identity, as SciPy's. `overwrite_a` and `check_finite` are accepted and not
    needed: an infinite or NaN entry raises always.

    Raises:
        ValueError: if `a` is not square or holds an infinite or NaN entry.
    """
    del overwrite_a, check_finite
    a = _tensor(a)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("expected square matrix")
    if a.shape[0] <= 2:
        if not _builtins.all(_math.isfinite(abs(x)) for x in _flat(a.tolist())):
            raise ValueError("array must not contain infs or NaNs")
        return (a, eye(a.shape[0])) if calc_q else a
    h, q = _hessenberg(a, bool(calc_q))
    return (h, q) if calc_q else h


def matrix_balance(A, permute=True, scale=True, separate=False, overwrite_a=False):
    """The square `A` balanced, ``B = T⁻¹ A T`` with ``T = P D`` a permutation (`permute`) times powers of two
    (`scale`), and `T`, as ``scipy.linalg.matrix_balance`` (LAPACK's ``gebal``): every entry of `B` an entry of `A`
    scaled exactly. With `separate`, ``(scaling, permutation)`` in place of `T`, the permutation a list of
    integers.

    Raises:
        ValueError: if `A` is not square or holds an infinite or NaN entry.
    """
    del overwrite_a
    A = _tensor(A)
    if len(A.shape) == 1:
        A = reshape(A, (1, -1))
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("The data matrix for balancing should be square.")
    n = A.shape[0]
    B, scaling, permutation = _matrix_balance(A, bool(permute), bool(scale))
    if separate:
        return B, (scaling, list(permutation))
    rows = reshape(arange(n), (n, 1))
    return B, where(rows == asarray([float(p) for p in permutation]) if n else rows == 0.0,
                    reshape(scaling, (1, n)), 0.0)


_SELECT = {"a": 0, "all": 0, 0: 0, "v": 1, "value": 1, 1: 1, "i": 2, "index": 2, 2: 2}


def eigh_tridiagonal(d, e, eigvals_only=False, select="a", select_range=None, check_finite=True, tol=0.0,
                     lapack_driver="auto"):
    """The eigenvalues, ascending, and with them the orthonormal eigenvectors (columns) of the real symmetric
    tridiagonal matrix of diagonal `d` and off-diagonal `e`, as ``scipy.linalg.eigh_tridiagonal``: every one by divide
    and conquer (LAPACK's ``stevd``, SciPy's default), or those in the half-open interval ``(vl, vu]`` of
    ``select="v"``, or those of the indices ``il … iu`` of ``select="i"``. `tol` and the drivers are accepted (the
    answer is ``stevd``'s whichever is named, within each driver's rules); `check_finite` too: an infinite or NaN
    entry raises always.

    Raises:
        ValueError: if `d` and `e` are not vectors of lengths n and n − 1, `select` or `select_range` is invalid, a
            driver is named outside its rules, or an entry is infinite or NaN.
        TypeError: if `d` or `e` is complex.
    """
    del check_finite, tol
    d, e = _tensor(d), _tensor(e)
    for check in (d, e):
        if len(check.shape) != 1:
            raise ValueError("expected a 1-D array")
        if check.dtype.startswith("complex"):
            raise TypeError("Only real arrays currently supported")
    if d.size != e.size + 1:
        raise ValueError(f"d ({d.size}) must have one more element than e ({e.size})")
    try:
        kind = _SELECT[select.lower() if isinstance(select, str) else select]
    except (KeyError, TypeError):
        raise ValueError("invalid argument for select") from None
    if kind:
        bounds = list(_flat(asarray(select_range).tolist())) if select_range is not None else []
        if len(bounds) != 2 or bounds[1] < bounds[0]:
            raise ValueError("select_range must be a 2-element array-like in nondecreasing order")
        if kind == 2:
            if not _builtins.all(isinstance(b, int) or float(b).is_integer() for b in bounds) or \
                    not _builtins.all(isinstance(b, _numbers.Integral) for b in select_range):
                raise ValueError("when using select=\"i\", select_range must contain integers")
            if _builtins.min(bounds) < 0 or _builtins.max(bounds) >= d.size:
                raise ValueError("select_range out of bounds")
    if not isinstance(lapack_driver, str):
        raise TypeError("lapack_driver must be str")
    drivers = ("auto", "stemr", "sterf", "stebz", "stev", "stevd")
    if lapack_driver not in drivers:
        raise ValueError(f"lapack_driver must be one of {drivers}, got {lapack_driver}")
    if lapack_driver in ("sterf", "stev", "stevd") and kind:
        raise ValueError(f'{lapack_driver} can only be used when select == "a"')
    if lapack_driver == "sterf" and not eigvals_only:
        raise ValueError("sterf can only be used when eigvals_only is True")
    single = d.dtype == "float32" and e.dtype == "float32"
    if kind and eigvals_only:  # bisection for a few of them (dstebz, SciPy's driver there)
        w = _tridiagonal_select(asarray(d, "float64"), asarray(e, "float64"), kind == 1, float(bounds[0]), float(bounds[1]))
        return asarray(w, "float32") if single else w
    w, v = _tridiagonal_eigen(asarray(d, "float64"), asarray(e, "float64"), not eigvals_only)
    if kind:
        values = w.tolist()
        if kind == 1:
            kept = [k for k, value in enumerate(values) if bounds[0] < value <= bounds[1]]
        else:
            kept = list(range(int(bounds[0]), int(bounds[1]) + 1))
        # w ascending: the kept indices run from kept[0] to kept[-1]
        w = w[kept[0]:kept[-1] + 1] if kept else zeros((0,))
        if not eigvals_only:
            v = v[:, kept[0]:kept[-1] + 1] if kept else zeros((d.size, 0))
    if single:
        w = asarray(w, "float32")
        v = v if eigvals_only else asarray(v, "float32")
    return w if eigvals_only else (w, v)


def eigvalsh_tridiagonal(d, e, select="a", select_range=None, check_finite=True, tol=0.0, lapack_driver="auto"):
    """The eigenvalues alone of the real symmetric tridiagonal matrix of diagonal `d` and off-diagonal `e`, ascending,
    as ``scipy.linalg.eigvalsh_tridiagonal`` (:func:`eigh_tridiagonal` with ``eigvals_only=True``)."""
    return eigh_tridiagonal(d, e, eigvals_only=True, select=select, select_range=select_range,
                            check_finite=check_finite, tol=tol, lapack_driver=lapack_driver)


def cholesky_banded(ab, overwrite_ab=False, lower=False, check_finite=True):
    """The Cholesky factor of the Hermitian positive-definite band `ab`, in its own form, as
    ``scipy.linalg.cholesky_banded`` (LAPACK's ``pbtrf``): ``U`` (``A = Uᴴ U``) of the upper form, ``L``
    (``A = L Lᴴ``) of the lower one; the corners outside the band as `ab` has them.

    Raises:
        LinAlgError: if `A` is not positive definite or an entry is infinite or NaN.
    """
    del overwrite_ab, check_finite
    return _cholesky_banded(ab, bool(lower))


def cho_solve_banded(cb_and_lower, b, overwrite_b=False, check_finite=True):
    """The solution `x` of ``A x = b`` from :func:`cholesky_banded`'s ``(cb, lower)``, as
    ``scipy.linalg.cho_solve_banded`` (LAPACK's ``pbtrs``).

    Raises:
        ValueError: if `b` has not as many rows as `cb` has columns.
        LinAlgError: if an entry is infinite or NaN.
    """
    del overwrite_b, check_finite
    cb, lower = cb_and_lower
    return _cho_solve_banded(cb, b, bool(lower))


def rsf2csf(T, Z, check_finite=True):
    """The complex Schur form ``(T, Z)`` of the real one, as ``scipy.linalg.rsf2csf``: each 2 × 2 block of `T` (a
    conjugate pair) triangularised by a unitary rotation of its rows and columns, `Z`'s columns turning with it; a
    subdiagonal entry within ``eps`` of its diagonal neighbours' magnitudes is taken as 0.

    Raises:
        ValueError: if `T` and `Z` are not square matrices of one shape, or an entry is infinite or NaN.
    """
    del check_finite
    T, Z = _same_dtype(T, Z)
    for name, X in (("Z", Z), ("T", T)):
        if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
            raise ValueError(f"Input '{name}' must be square.")
    if T.shape[0] != Z.shape[0]:
        raise ValueError(f"Input array shapes must match: Z: {Z.shape} vs. T: {T.shape}")
    if T.dtype.startswith("complex"):
        raise ValueError("rsf2csf: T and Z are a real Schur form (real matrices)")
    return _rsf2csf(T, Z)


def _flipped(t, axis):
    """`t` with its rows (``axis=0``) or columns (``axis=1``) in reverse order."""
    return t[::-1] if axis == 0 else t[:, ::-1]


def rq(a, overwrite_a=False, lwork=None, mode="full", check_finite=True):
    """The RQ factorisation ``A = R Q`` of the `M × N` `a`, `R` upper trapezoidal (its triangle at the right) and `Q`
    unitary, as ``scipy.linalg.rq`` (LAPACK's ``gerqf``): from the QR factorisation of ``(J A)ᴴ``, `J` the reversal,
    ``R = J R₁ᴴ J`` and ``Q = J Q₁ᴴ``. ``mode="r"`` gives `R` alone, ``"economic"`` `R` ``M × K`` and `Q` ``K × N``,
    ``K = min(M, N)``.

    Raises:
        ValueError: if `mode` is not one of these, `a` is not a matrix or holds an infinite or NaN entry.
    """
    del overwrite_a, lwork, check_finite
    if mode not in ("full", "r", "economic"):
        raise ValueError("Mode argument should be one of ['full', 'r', 'economic']")
    a = _tensor(a)
    if len(a.shape) != 2:
        raise ValueError("expected matrix")
    M, N = a.shape
    economic = mode == "economic" and N >= M
    b = conj(matrix_transpose(_flipped(a, 0)))  # (J A)ᴴ, N × M
    q1, r1 = qr(b, mode="reduced" if economic else "complete")
    r = _flipped(_flipped(conj(matrix_transpose(r1)), 0), 1)
    q = _flipped(conj(matrix_transpose(q1)), 0)
    return r if mode == "r" else (r, q)


def cdf2rdf(w, v):
    """The real block-diagonal eigenvalues and real eigenvectors of a real matrix from its complex ones `w` and `v`
    (``A v = v diag(w)``, conjugate pairs adjacent), as ``scipy.linalg.cdf2rdf``: ``A vr = vr wr``, each pair a
    2 × 2 block ``[[re, im], [-im, re]]``; over a stack's leading axes.

    Raises:
        ValueError: if the shapes do not match or a complex eigenvalue is not in a conjugate pair.
    """
    w, v = _tensor(w), _tensor(v)
    if len(w.shape) < 1:
        raise ValueError("expected w to be at least 1D")
    if len(v.shape) < 2:
        raise ValueError("expected v to be at least 2D")
    if len(v.shape) != len(w.shape) + 1:
        raise ValueError("expected eigenvectors array to have exactly one dimension more than eigenvalues array")
    n = w.shape[-1]
    if v.shape[-2] != v.shape[-1]:
        raise ValueError("expected v to be a square matrix or stacked square matrices: v.shape[-2] = v.shape[-1]")
    if v.shape[-1] != n:
        raise ValueError("expected the same number of eigenvalues as eigenvectors")
    batch = w.shape[:-1]
    count = _math.prod(batch)
    flat_w = reshape(w, (count, n))
    flat_v = reshape(v, (count, n, n))
    blocks, vectors = [], []
    i, j = _grid(n)
    for k in range(count):
        values = [complex(x) for x in flat_w[k].tolist()] if n else []
        complex_at = [m for m, x in enumerate(values) if x.imag != 0]
        if len(complex_at) % 2:
            raise ValueError("expected complex-conjugate pairs of eigenvalues")
        partner = list(range(n))
        role = [0.0] * n  # 1 the first of a pair, -1 the second
        for first, second in zip(complex_at[0::2], complex_at[1::2]):
            partner[first], partner[second] = second, first
            role[first], role[second] = 1.0, -1.0
        partner_col = reshape(asarray([float(p) for p in partner]), (n, 1))
        role_col = reshape(asarray(role), (n, 1))
        real_row = reshape(asarray([x.real for x in values]), (1, n))
        imag_col = reshape(asarray([x.imag for x in values]), (n, 1))
        paired = (j == partner_col) & (role_col != 0.0)
        blocks.append(where(i == j, real_row, where(paired, imag_col, 0.0)))
        diagonal = where(role_col > 0.0, 0.5j, where(role_col < 0.0, 0.5 + 0j, 1.0 + 0j))
        off = where(role_col > 0.0, 0.5 + 0j, -0.5j)
        u = where(i == j, diagonal, where(paired, off, 0j))
        vectors.append(real(flat_v[k] @ u))
    if not batch:
        return blocks[0], vectors[0]
    return (reshape(_stacked_first(blocks), batch + (n, n)), reshape(_stacked_first(vectors), batch + (n, n)))


def _stacked_first(parts):
    """The tensors `parts` (one shape) along a new first axis."""
    columns = _numerical._stack_last([reshape(part, (-1,)) for part in parts])  # (size, count)
    return reshape(matrix_transpose(columns), (len(parts),) + parts[0].shape)


__all__ += ["hessenberg", "matrix_balance", "eigh_tridiagonal", "eigvalsh_tridiagonal", "cholesky_banded",
            "cho_solve_banded", "rsf2csf", "rq", "cdf2rdf", "qz", "ordqz"]

# --- NumPy's assembly and triangles: concatenate, stack, vstack, hstack, tril, triu, diag, kron ---


def _joined(arrays, name):
    """The tensors of `arrays` in the dtype they all fit (NumPy's promotion, :func:`_promoted`)."""
    if isinstance(arrays, (Tensor, str)) or not hasattr(arrays, "__iter__"):
        raise TypeError(f"{name}: a sequence of arrays")
    tensors = [_tensor(x) for x in arrays]
    if not tensors:
        raise ValueError("need at least one array to concatenate")
    dtype = _promoted([t.dtype for t in tensors])
    return [t if t.dtype == dtype else asarray(t, dtype) for t in tensors]


def concatenate(arrays, /, axis=0):
    """The arrays one after the other along `axis` (every other axis of one extent), as ``numpy.concatenate``; with
    ``axis=None`` each flattened first. The result in the dtype they all fit, as NumPy promotes them.

    Raises:
        ValueError: if there is no array, one is rank 0, the axis is out of their rank, or the other axes differ.
    """
    tensors = _joined(arrays, "concatenate")
    if axis is None:
        tensors, axis = [reshape(t, (-1,)) for t in tensors], 0
    rank = len(tensors[0].shape)
    if rank == 0:
        raise ValueError("zero-dimensional arrays cannot be concatenated")
    axis = _operator.index(axis)
    if not -rank <= axis < rank:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {rank}")
    if _builtins.any(len(t.shape) != rank for t in tensors):
        raise ValueError("all the input arrays must have same number of dimensions")
    return _numerical._concatenate(tensors, axis % rank)


concat = concatenate


def stack(arrays, /, axis=0):
    """The arrays (one shape) along a new axis `axis` of the result, as ``numpy.stack``.

    Raises:
        ValueError: if there is no array, their shapes differ, or the axis is out of the result's rank.
    """
    tensors = _joined(arrays, "stack")
    shape = tensors[0].shape
    if _builtins.any(t.shape != shape for t in tensors):
        raise ValueError("all input arrays must have the same shape")
    rank = len(shape) + 1
    axis = _operator.index(axis)
    if not -rank <= axis < rank:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {rank}")
    axis %= rank
    expanded = shape[:axis] + (1,) + shape[axis:]
    return _numerical._concatenate([reshape(t, expanded) for t in tensors], axis)


def vstack(tup):
    """The arrays row-wise, as ``numpy.vstack``: a vector as a row (1 × n), along the first axis."""
    tensors = _joined(tup, "vstack")
    return concatenate([reshape(t, (1, t.shape[0] if t.shape else 1)) if len(t.shape) < 2 else t for t in tensors], axis=0)


def hstack(tup):
    """The arrays column-wise, as ``numpy.hstack``: vectors one after the other, else along the second axis."""
    tensors = _joined(tup, "hstack")
    tensors = [reshape(t, (1,)) if len(t.shape) == 0 else t for t in tensors]
    return concatenate(tensors, axis=0 if len(tensors[0].shape) == 1 else 1)


def _band_mask(shape, k, lower):
    """True on and below the `k` diagonal (above where not `lower`) of matrices of the last two axes of `shape`."""
    rows, cols = shape[-2], shape[-1]
    i = reshape(arange(rows), (rows, 1))
    j = reshape(arange(cols), (1, cols))
    return (j - i <= k) if lower else (j - i >= k)


def _triangle(m, k, lower, name):
    m = _tensor(m)
    k = _operator.index(k)
    if len(m.shape) == 0:
        raise ValueError(f"{name}: the input must be at least 1-D")
    shape = m.shape
    if len(shape) == 1:  # NumPy's: the vector as every row of a square
        m, shape = reshape(m, (1, shape[0])), (shape[0], shape[0])
    return where(_band_mask(shape, k, lower), m, False if m.dtype == "bool" else 0.0)


def tril(m, k=0):
    """The matrices of `m` (its last two axes) zero above their `k` diagonal, as ``numpy.tril`` (a vector broadcast to
    a square first, as NumPy's)."""
    return _triangle(m, k, True, "tril")


def triu(m, k=0):
    """The matrices of `m` (its last two axes) zero below their `k` diagonal, as ``numpy.triu``."""
    return _triangle(m, k, False, "triu")


def diag(v, k=0):
    """``numpy.diag``: a matrix with the vector `v` on its `k` diagonal (above the main one for k > 0), or a matrix's
    `k` diagonal as a vector (a copy).

    Raises:
        ValueError: if `v` is neither a vector nor a matrix.
    """
    v = _tensor(v)
    k = _operator.index(k)
    if len(v.shape) == 2:
        return diagonal(v, offset=k)
    if len(v.shape) != 1:
        raise ValueError("Input must be 1- or 2-d.")
    n = v.shape[0] + _builtins.abs(k)
    if v.shape[0] == 0:
        return zeros((n, n), v.dtype)
    zero = zeros((_builtins.abs(k),), v.dtype)
    column = concatenate([v, zero] if k >= 0 else [zero, v])  # row i's value: v[i] (k ≥ 0) or v[i + k]
    i = reshape(arange(n), (n, 1))
    j = reshape(arange(n), (1, n))
    return where(j - i == k, reshape(column, (n, 1)), False if v.dtype == "bool" else 0.0)


def kron(a, b):
    """The Kronecker product, as ``numpy.kron``: blocks ``a[i, j] * b`` (over every axis, the shorter shape led by
    ones)."""
    a, b = _tensor(a), _tensor(b)
    rank = _builtins.max(len(a.shape), len(b.shape))
    sa = (1,) * (rank - len(a.shape)) + a.shape
    sb = (1,) * (rank - len(b.shape)) + b.shape
    a = reshape(a, tuple(x for pair in zip(sa, (1,) * rank) for x in pair))
    b = reshape(b, tuple(x for pair in zip((1,) * rank, sb) for x in pair))
    return reshape(a * b, tuple(x * y for x, y in zip(sa, sb)))


__all__ += ["concatenate", "concat", "stack", "vstack", "hstack", "tril", "triu", "diag", "kron"]

from numerical import fft  # noqa: E402  (the module numpy.fft's names live in, as numpy.fft is NumPy's)
