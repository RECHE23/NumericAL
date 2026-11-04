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
import collections as _collections
import functools as _functools
import math as _math
import numbers as _numbers
import operator as _operator
import os as _os
import sys as _sys
import warnings as _warnings

from numerical import _numerical
from numerical._numerical import Tensor, asarray as _asarray, _asarray_nested, error, get_num_threads
from numerical._numerical import _gemm_path, _limit_gemm_path, _set_num_threads, release_memory

__all__ = [
    "Tensor", "einsum", "solve", "asarray", "error", "LinAlgError",
    "determinant", "det", "slogdet", "inverse", "inv", "solve_banded", "solveh_banded", "solve_toeplitz", "solve_circulant",
    "toeplitz", "hankel", "circulant", "block_diag",
    "expm", "expm_multiply", "expm_frechet", "expm_cond", "fractional_matrix_power", "funm", "logm", "LogmRankWarning", "LogmExactlySingularWarning",
    "LogmNearlySingularWarning", "sqrtm", "polar",
    "solve_sylvester", "solve_continuous_lyapunov", "solve_lyapunov", "solve_discrete_lyapunov",
    "solve_continuous_are", "solve_discrete_are", "lstsq", "pinv",
    "matrix_rank", "cholesky", "ldl", "qr", "eig",
    "eigvals", "eigh", "eigvalsh", "svd", "svdvals", "norm", "vector_norm", "matrix_norm", "cond", "matrix_power", "multi_dot",
    "trace", "outer", "cross", "tensordot", "vecdot", "conj", "conjugate", "matmul",
    "array_equal", "set_num_threads", "get_num_threads", "gemm_path", "limit_gemm_path", "release_memory",
    "get_include", "get_config", "blas_library",
    "schur", "solve_triangular", "lu", "lu_factor", "lu_solve", "cho_factor", "cho_solve", "LUFactor", "CholeskyFactor",
]


class LinAlgError(error, ValueError):
    """Raised by the linear-algebra functions when they cannot compute their answer: a singular matrix, one
    not positive definite, an infinite or NaN entry, an iteration that did not converge, or an operand of the
    wrong shape. As ``numpy.linalg.LinAlgError`` it is a :class:`ValueError`; it is also a
    :class:`numerical.error`, so code catching that keeps working. Where NumPy is loaded, it is also a
    ``numpy.linalg.LinAlgError`` (and so SciPy's, the same class): code written for them catches it as theirs,
    without NumericAL importing NumPy."""

    def __new__(cls, *args, **kwargs):
        if cls is LinAlgError:
            numpy_linalg = _sys.modules.get("numpy.linalg")
            if numpy_linalg is not None and isinstance(getattr(numpy_linalg, "LinAlgError", None), type):
                cls = _beside(LinAlgError, numpy_linalg.LinAlgError)
        return super().__new__(cls, *args, **kwargs)


@_functools.cache
def _beside(ours, theirs):
    """A subclass of NumericAL's class `ours` and of NumPy's or SciPy's `theirs`, made once; pickled as `ours`, the
    name it is found by (unpickled where NumPy is loaded, it is this subclass again)."""
    def reduce(self):
        return (ours, self.args) + ((self.__dict__,) if self.__dict__ else ())

    return type(ours.__name__, (ours, theirs),
                {"__module__": ours.__module__, "__doc__": ours.__doc__, "__reduce__": reduce})


__version__ = "2026.11.5"

def set_num_threads(n):
    """Cap the threads a kernel uses at `n`, the caller's included; ``0`` lifts the cap. At ``1`` no thread is
    started. For a process per core, say, so that together they do not ask for more threads than the machine
    has, as ``OMP_NUM_THREADS`` caps BLAS."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"set_num_threads: a count of threads is a non-negative integer, not {n!r}")
    _set_num_threads(n if n < 2**31 else 2**31)  # not min(): this module defines its own, a reduction


def _environment_threads():
    """NUMERICAL_NUM_THREADS, read once at import: a count of threads, or None (with a warning if it is not one).
    Without it, OMP_NUM_THREADS's first count, as OpenBLAS reads it (joblib and loky set it in their workers), a value
    that is not one ignored."""
    value = _os.environ.get("NUMERICAL_NUM_THREADS", "").strip()
    if not value:
        first = _os.environ.get("OMP_NUM_THREADS", "").split(",")[0].strip()
        return int(first) if first.isascii() and first.isdigit() and int(first) > 0 else None
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
    """Numbers in C order behind an ``__array_interface__``: how a nested list or a scalar reaches the
    same reader as an array (real values as float64, complex ones as complex128, bools alone as bool, integers
    that int64 holds as int64, exactly, which the reader takes as float64 unless int64 is asked)."""

    def __init__(self, values, shape):
        order = "<" if _sys.byteorder == "little" else ">"
        if values and _builtins.all(isinstance(value, bool) for value in values):
            self._data = _array.array("B", values)
            self.__array_interface__ = {"version": 3, "typestr": "|b1", "shape": tuple(shape),
                                        "data": (self._data.buffer_info()[0], True)}
            return
        if values and _builtins.all(isinstance(value, _numbers.Integral) and -2**63 <= value < 2**63 for value in values):
            self._data = _array.array("q", [int(value) for value in values])
            self.__array_interface__ = {"version": 3, "typestr": order + "i8", "shape": tuple(shape),
                                        "data": (self._data.buffer_info()[0], True)}
            return
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


_DTYPES = ("float64", "complex128", "float32", "complex64", "bool", "int64")
_SINGLE = ("float32", "complex64")


def _promoted(dtypes):
    """NumPy's result dtype of tensors of `dtypes`: complex where one is, single precision only where all are; a bool
    one takes the others' dtype, and bools alone stay bool; int64 alone (bools beside) stays int64, beside a float or
    complex dtype the double precision of its kind (int64 and float32: float64, as NumPy's)."""
    dtypes = [dtype for dtype in dtypes if dtype != "bool"]
    if not dtypes:
        return "bool"
    numbers = [dtype for dtype in dtypes if dtype != "int64"]
    if not numbers:
        return "int64"
    single = len(numbers) == len(dtypes) and _builtins.all(dtype in _SINGLE for dtype in numbers)
    if _builtins.any(dtype.startswith("complex") for dtype in dtypes):
        return "complex64" if single else "complex128"
    return "float32" if single else "float64"


def _weak(value):
    """Whether `value` is a Python number, which NumPy 2 promotes as weak: it keeps the tensor's precision."""
    return type(value) in _NATIVE_NUMBERS


def _beside_weak(dtype, value):
    """The dtype of a tensor of `dtype` beside the Python number `value`: its own, complex for a complex `value`; a
    bool tensor's beside a bool, else the number's own (float64 for an int, as :func:`asarray` reads one); an int64
    tensor's beside an int, float64 beside a float."""
    if dtype == "bool":
        return "bool" if isinstance(value, bool) else "complex128" if isinstance(value, complex) else "float64"
    if dtype == "int64" and not isinstance(value, int):
        return "complex128" if isinstance(value, complex) else "float64"
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
        operands = [arg if isinstance(arg, (Tensor, str)) else _read(arg) for arg in args[:count]]
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
        if index.dtype == "int64":
            return "ints", reshape(index, (-1,)), index.shape
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
    if _builtins.any(isinstance(values, Tensor) for values, _ in parts):  # broadcast natively, as int64 Tensors
        summed = None
        for values, shape in parts:
            if isinstance(values, bytes):
                values = _numerical._frombytes(values, (len(values) // 8,), "int64", False)
            elif not isinstance(values, Tensor):
                values = asarray(values, "int64")
            values = reshape(values, shape)
            summed = values if summed is None else summed + values
        return broadcast_to(summed, extents).tobytes()
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
    if isinstance(values, Tensor):  # an int64 Tensor, natively
        if values.size:
            outside = (values >= extent) | (values < -extent)
            if any(outside):
                value = int(values[int(argmax(outside))])
                raise IndexError(f"index {value} is out of bounds for axis {axis} with size {extent}")
            if int(min(values)) < 0:
                values = where(values < 0, values + extent, values)
        return values if stride == 1 else values * stride
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
    other = other if isinstance(other, Tensor) else _read(other)
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
    """The matrix product of `x1` and `x2`, as ``numpy.matmul`` (and ``x1 @ x2``).

    Stacks of matrices along leading axes broadcast against each other, a vector a row first and a column second, two
    vectors their dot product. The dtypes are promoted as NumPy promotes them; a bool operand is read as float64.

    Args:
        x1: The left operand, a Tensor or an array-like of rank 1 or more.
        x2: The right operand, a Tensor or an array-like of rank 1 or more.

    Returns:
        Tensor | float | complex: The product; a number for two vectors.

    Raises:
        LinAlgError: if an operand is of rank 0 or the inner dimensions differ (the message names both shapes).
        TypeError: if an operand is nothing :func:`asarray` reads.

    Examples:
        >>> import numerical
        >>> numerical.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        numerical.Tensor([[19.0, 22.0],
                          [43.0, 50.0]], dtype=float64)
        >>> numerical.matmul([1, 2], [3, 4])
        11.0
        >>> numerical.matmul(numerical.zeros((3, 2, 2)), numerical.eye(2)).shape
        (3, 2, 2)
    """
    try:
        product = _numerical._borrowed_product(x1, x2)  # Tensors and NumPy arrays read where they lie, when they can be
    except error as caught:  # shapes that do not fit, said with both (checked here only, off the products' path)
        shapes = (getattr(x1, "shape", None), getattr(x2, "shape", None))
        if shapes[0] is not None and shapes[1] is not None:
            _matmul_shapes(*shapes)
        raise LinAlgError(*caught.args) from None
    return _operand(x1)._matmul(x2) if product is NotImplemented else product  # else as ``@``


def _rmatmul(self, other):
    return _matmul(_read(other), self)


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


def array_equal(a, b, equal_nan=False):
    """``True`` when `a` and `b` (Tensors or array-likes) have the same shape and equal elements, as
    ``numpy.array_equal``: a NaN equals nothing, itself included, unless `equal_nan` (a complex one with a NaN part
    equal to any other)."""
    a, b = _same_dtype(a, b)
    if a.shape != b.shape:
        return False
    if not equal_nan or a.dtype in ("int64", "bool"):
        return _numerical._equal(a, b)
    return bool(all(equal(a, b) | (isnan(a) & isnan(b))))


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
# them (a native function taking *args has none to compare).
def einsum(subscripts, /, *operands):
    """Einstein summation over the operands, as ``numpy.einsum``: ``"ij,jk->ik"`` (a matrix product), ``"i,i->"``
    (a dot product), ``"ij->ji"`` (a transpose), ``"ii->i"`` (a diagonal).

    Without ``->`` the output takes the labels that appear once, in alphabetical order, as NumPy's; ``...`` stands for
    the axes no label names. The operands are brought to one dtype as NumPy promotes them (complex where one is,
    single precision only where all are); a bool array-like is read as float64, a bool Tensor refused.

    Args:
        subscripts: The labels of each operand's axes, comma-separated, then optionally ``->`` and the output's: at
            most the 52 ASCII letters.
        *operands: The Tensors or array-likes (NumPy arrays, nested lists, numbers), one per comma-separated term.

    Returns:
        Tensor: The summed product, of rank 0 for a full contraction.

    Raises:
        LinAlgError: if the subscripts do not parse, their terms are not one per operand, an operand's rank does not
            match its subscripts, or a repeated label has two extents.
        TypeError: if `subscripts` is not a string, or an operand is a bool Tensor or nothing :func:`asarray` reads.

    Examples:
        >>> import numerical
        >>> numerical.einsum("ij,jk->ik", [[1, 0], [0, 2]], [[1, 2], [3, 4]])
        numerical.Tensor([[1.0, 2.0],
                          [6.0, 8.0]], dtype=float64)
        >>> numerical.einsum("i,i->", [1, 2, 3], [1, 2, 3])
        numerical.Tensor(14.0, dtype=float64)
        >>> numerical.einsum("ii->i", [[1, 2], [3, 4]])
        numerical.Tensor([1.0, 4.0], dtype=float64)
    """
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


# scipy.linalg.solve's structures, by every name it takes for each.
_STRUCTURES = {"general": "gen", "gen": "gen", "diagonal": "diagonal", "tridiagonal": "tridiagonal", "banded": "banded",
               "upper triangular": "upper", "lower triangular": "lower", "symmetric": "sym", "sym": "sym",
               "hermitian": "her", "her": "her", "positive definite": "pos", "pos": "pos"}


def solve(a, b, /, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a=None,
          transposed=False):
    """The solution `x` of ``a @ x = b``, as ``numpy.linalg.solve``, or ``scipy.linalg.solve`` with its arguments.

    Without `assume_a` or `transposed`, NumPy's ``numpy.linalg.solve``: a triangular `a` detected and solved by
    substitution, else the LU factorisation; a stack of systems broadcast. With them, ``scipy.linalg.solve``'s:
    `assume_a` names `a`'s structure and its solver, and SciPy's reciprocal condition number
    ``1 / (‖a‖₁ ‖a⁻¹‖₁)``, the inverse's norm estimated from the factorisation (LAPACK's ``gecon``, ``pocon``,
    ``trcon``, ``gbcon``; a diagonal's exact), under half the machine epsilon warns (:class:`LinAlgWarning`: the
    answer may have no correct digit). `a` and `b` are never written, and a NaN or an infinity always raises.

    Args:
        a: The square matrix, or a stack of them (``(..., n, n)``): a Tensor or an array-like.
        b: The right-hand side: a vector or a matrix of right-hand sides (one per column), or a stack of them.
        lower: With `assume_a` ``"sym"``, ``"her"`` or ``"pos"``, read `a` from its lower triangle rather than its
            upper one.
        overwrite_a: Accepted for SciPy's signature; ignored.
        overwrite_b: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).
        assume_a: `a`'s structure, as SciPy names it: ``"diagonal"`` (a division), ``"tridiagonal"`` and
            ``"banded"`` (the band LU, of the bandwidth `a` has), ``"upper triangular"`` and ``"lower triangular"``
            (substitution), ``"sym"``, ``"her"`` (the matrix read from one triangle, by default the upper one,
            `lower` the lower; a Hermitian one's diagonal taken real; then the LU, which here outruns the symmetric
            indefinite factorisation), ``"pos"`` (Cholesky's, from that triangle) or ``"gen"``; SciPy's long names
            (``"general"``, ``"symmetric"``, ``"hermitian"``, ``"positive definite"``) too.
        transposed: Solve ``a.T @ x = b``.

    Returns:
        Tensor: `x`, of `b`'s shape (broadcast over a stack), in the dtype `a` and `b` promote to.

    Raises:
        LinAlgError: if `a` is singular, not positive definite for ``"pos"``, not square, holds an infinite or NaN
            entry, or `b` does not fit it.
        ValueError: if `assume_a` names no structure.
        NotImplementedError: for `transposed` with a complex `a`, as SciPy's.

    Examples:
        >>> import numerical
        >>> numerical.solve([[2.0, 0.0], [0.0, 4.0]], [2.0, 2.0])
        numerical.Tensor([1.0, 0.5], dtype=float64)
        >>> numerical.solve([[4.0, 0.0], [0.0, 2.0]], [[4.0, 8.0], [2.0, 2.0]])
        numerical.Tensor([[1.0, 2.0],
                          [1.0, 1.0]], dtype=float64)
        >>> numerical.solve([[2.0, 0.0], [1.0, 4.0]], [2.0, 9.0], assume_a="lower triangular")
        numerical.Tensor([1.0, 2.0], dtype=float64)
    """
    del overwrite_a, overwrite_b, check_finite
    if assume_a is None and not transposed:
        return _borrowed("solve", _solve, a, b)
    structure = "gen" if assume_a is None else _STRUCTURES.get(assume_a) if isinstance(assume_a, str) else None
    if structure is None:
        raise ValueError(f"{assume_a} is not a recognized matrix structure")
    a, b = _operand(a), _operand(b)
    if transposed and a.dtype in ("complex128", "complex64"):
        raise NotImplementedError("solve cannot solve a.T @ x = b or a.conj().T @ x = b for a complex a, as SciPy's")
    if transposed and structure not in ("sym", "her", "pos"):  # those are their transposes (a real `a` here)
        a = a.mT
        structure = {"upper": "lower", "lower": "upper"}.get(structure, structure)
    return _stacked_solve(a, b, structure, lower, 3)


def _stacked_solve(a, b, structure, lower, level):
    """:func:`_structured_solve` of each system of the stack `a` in turn, as SciPy's: `b` a vector or a matrix (`n`
    rows) they share, or one per system (of `a`'s rank). `level` the frames from here to the caller of :func:`solve`
    (minus the warning's own), to point a warning there."""
    if len(a.shape) <= 2:
        return _structured_solve(a, b, structure, lower, level + 1)
    if len(b.shape) <= 2:
        return stack([_stacked_solve(a[i], b, structure, lower, level + 1) for i in range(a.shape[0])])
    if len(b.shape) != len(a.shape) or b.shape[0] != a.shape[0]:
        raise LinAlgError(f"solve: b of shape {b.shape} fits no system of the stack of shape {a.shape}")
    return stack([_stacked_solve(a[i], b[i], structure, lower, level + 1) for i in range(a.shape[0])])


def _conditioned(x, rcond, dtype, level):
    """`x`, the solution of a system whose reciprocal condition number is `rcond`, as SciPy's ``solve`` judges it:
    under half the machine epsilon of `dtype` (LAPACK's ``lamch('E')``), 0 included (an exactly singular matrix raised
    before), warns (:class:`LinAlgWarning`) at the caller, `level` frames up."""
    if not rcond >= _MACHINE_EPSILON["float32" if dtype in _SINGLE else "float64"] / 2.0:
        _warnings.warn(f"Ill-conditioned matrix (rcond={rcond:.6g}): result may not be accurate.", _linalg_warning(),
                       stacklevel=level)
    return x


def _structured_solve(a, b, structure, lower, level):
    """:func:`solve` of one system of the `structure` :data:`_STRUCTURES` names, with its reciprocal condition number
    (:func:`_conditioned`)."""
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise LinAlgError(f"solve: a is a square matrix, not of shape {a.shape}")
    n = a.shape[0]
    a, b = _operand(a), _operand(b)
    if a.dtype != b.dtype:
        dtype = _promoted((a.dtype, b.dtype))
        a, b = asarray(a, dtype), asarray(b, dtype)
    if n == 0 or b.size == 0:
        return _borrowed("solve", _solve, a, b)
    if structure in ("upper", "lower"):
        x = solve_triangular(a, b, lower=structure == "lower")
        return _conditioned(x, _factor_condition(a, a, 1, structure == "lower"), a.dtype, level + 1)
    if structure == "pos":
        factor = cho_factor(a, lower=lower)
        x = cho_solve(factor, b)
        return _conditioned(x, _factor_condition(factor.c, a, 0, bool(lower)), a.dtype, level + 1)
    if structure == "diagonal":
        d = diagonal(a)
        if not all(d != 0.0):
            raise LinAlgError("solve: a singular matrix (a zero on the diagonal)")
        moduli = absolute(d)
        x = b / (d if len(b.shape) == 1 else reshape(d, (n, 1)))
        return _conditioned(x, float(min(moduli)) / float(max(moduli)), a.dtype, level + 1)
    if structure in ("tridiagonal", "banded"):
        below, above = (1, 1) if structure == "tridiagonal" else bandwidth(a)
        zero = zeros((1,), a.dtype)
        rows = []  # row above - k of LAPACK's band storage: the diagonal at offset k, shifted to its columns
        for k in range(above, -below - 1, -1):
            band = diagonal(a, offset=k) if n > _builtins.abs(k) else zeros((0,), a.dtype)
            pad = [zero] * _builtins.min(_builtins.abs(k), n)
            rows.append(concatenate(pad + [band] if k > 0 else [band] + pad) if k else band)
        x, rcond = _solve_banded_conditioned(stack(rows), b, below, above)
        return _conditioned(x, rcond, a.dtype, level + 1)
    if structure in ("sym", "her"):  # the matrix of one triangle, its other the transpose (conjugate, Hermitian), a Hermitian diagonal real
        strict = tril(a, -1) if lower else triu(a, 1)
        mirror = strict.mT if structure == "sym" else conj(strict).mT
        middle = diagonal(a) if structure == "sym" or a.dtype in ("float64", "float32") else asarray(real(diagonal(a)), a.dtype)
        a = strict + mirror + diag(middle)
    x, rcond = _solve_conditioned(a, b)
    return _conditioned(x, rcond, a.dtype, level + 1)


def determinant(a, /, *, overwrite_a=False, check_finite=True):
    """The determinant of a square matrix or a stack of them, as ``numpy.linalg.det`` and ``scipy.linalg.det``.

    From the LU factorisation with partial pivoting. A singular matrix's determinant is 0, not an error.

    Args:
        a: The square matrix, or a stack of them (``(..., n, n)``): a Tensor or an array-like, real or complex.
        overwrite_a: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).

    Returns:
        float | complex | Tensor: The determinant, a number (complex for a complex matrix); for a stack, a Tensor of
        the stack's leading shape.

    Raises:
        LinAlgError: if `a` is not a square matrix (or a stack of them) or holds an infinite or NaN entry.

    Examples:
        >>> import numerical
        >>> numerical.det([[2.0, 0.0], [0.0, 4.0]])
        8.0
        >>> numerical.det([[1, 2], [3, 4]])
        -2.0
        >>> numerical.det([[[2.0, 0.0], [0.0, 4.0]], [[1.0, 0.0], [0.0, 3.0]]])
        numerical.Tensor([8.0, 3.0], dtype=float64)
    """
    del overwrite_a, check_finite  # scipy.linalg.det's: an infinite or NaN entry raises always
    return _borrowed("determinant", _determinant, a)


def slogdet(a, /):
    """The sign and the natural logarithm of the absolute value of the determinant, as ``numpy.linalg.slogdet``.

    It does not underflow or overflow where :func:`determinant` does: the determinant is ``sign * exp(logabsdet)``.

    Args:
        a: The square matrix, or a stack of them (``(..., n, n)``): a Tensor or an array-like, real or complex.

    Returns:
        tuple: ``(sign, logabsdet)``. For a real matrix `sign` is 1.0, -1.0 or 0.0 (a singular matrix, whose
        `logabsdet` is ``-inf``); for a complex one, a complex number of modulus 1. For a stack, two Tensors of the
        stack's leading shape.

    Raises:
        LinAlgError: if `a` is not a square matrix (or a stack of them) or holds an infinite or NaN entry.

    Examples:
        >>> import numerical
        >>> numerical.slogdet([[-1.0, 0.0], [0.0, 1.0]])
        (-1.0, 0.0)
        >>> numerical.slogdet([[0.0, 0.0], [0.0, 4.0]])
        (0.0, -inf)
    """
    return _borrowed("slogdet", _slogdet, a)


def inverse(a, /, *, overwrite_a=False, check_finite=True):
    """The inverse of a square matrix, or of each matrix of a stack, as ``numpy.linalg.inv`` and ``scipy.linalg.inv``.

    Args:
        a: The square matrix, or a stack of them (``(..., n, n)``): a Tensor or an array-like, real or complex.
        overwrite_a: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).

    Returns:
        Tensor: The inverse, of `a`'s shape and dtype (float64 for integers or bools).

    Raises:
        LinAlgError: if `a` is singular, not a square matrix (or a stack of them), or holds an infinite or NaN entry.

    Examples:
        >>> import numerical
        >>> numerical.inv([[2.0, 0.0], [0.0, 4.0]])
        numerical.Tensor([[0.5, 0.0],
                          [0.0, 0.25]], dtype=float64)
        >>> numerical.inv([[1.0, 1.0], [0.0, 1.0]])
        numerical.Tensor([[1.0, -1.0],
                          [0.0, 1.0]], dtype=float64)
    """
    del overwrite_a, check_finite  # scipy.linalg.inv's: an infinite or NaN entry raises always
    return _borrowed("inverse", _inverse, a)


det = determinant
inv = inverse


_expm = _linalg(_numerical.expm, arrays=1)
_expm_multiply = _linalg(_numerical.expm_multiply, arrays=2)
_expm_frechet = _linalg(_numerical.expm_frechet, arrays=2)
_expm_cond = _linalg(_numerical.expm_cond, arrays=1)
_logm = _linalg(_numerical.logm, arrays=1)
_fractional_matrix_power = _linalg(_numerical.fractional_matrix_power, arrays=1)
_qr_pivoted = _linalg(_numerical.qr_pivoted, arrays=1)
_ldl = _linalg(_numerical.ldl, arrays=1)
_solve_banded = _linalg(_numerical.solve_banded, arrays=2)
_solve_banded_conditioned = _linalg(_numerical.solve_banded_conditioned, arrays=2)
_solve_conditioned = _linalg(_numerical.solve_conditioned, arrays=2)
_factor_condition = _linalg(_numerical.factor_condition, arrays=2)
_solveh_banded = _linalg(_numerical.solveh_banded, arrays=2)
_solve_toeplitz = _linalg(_numerical.solve_toeplitz, arrays=3)
_solve_circulant = _linalg(_numerical.solve_circulant, arrays=2)
_solve_sylvester = _linalg(_numerical.solve_sylvester, arrays=3)
_solve_continuous_lyapunov = _linalg(_numerical.solve_continuous_lyapunov, arrays=2)
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
    ab = ab if isinstance(ab, Tensor) else _read(ab)
    b = b if isinstance(b, Tensor) else _read(b)
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
    ab = ab if isinstance(ab, Tensor) else _read(ab)
    b = b if isinstance(b, Tensor) else _read(b)
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
        c = c if isinstance(c, Tensor) else _read(c)
        r = r if isinstance(r, Tensor) else _read(r)
    else:
        c = c_or_cr if isinstance(c_or_cr, Tensor) else _read(c_or_cr)
        r = conj(c) if c.dtype.startswith("complex") else c
    b = b if isinstance(b, Tensor) else _read(b)
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
    c = c if isinstance(c, Tensor) else _read(c)
    b = b if isinstance(b, Tensor) else _read(b)
    if c.ndim != 1 or b.ndim not in (1, 2) or b.shape[0] != c.shape[0]:
        raise ValueError(f"Shapes of c {c.shape} and b {b.shape} are incompatible")
    return _solve_circulant(c, b, singular == "lstsq", -1.0 if tol is None else float(tol))


def _vectors(name, *values):
    """`values` as vectors (a number as one entry) of one dtype, as NumPy promotes them."""
    tensors = [value if isinstance(value, Tensor) else _read(value) for value in values]
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
    tensors = [arr if isinstance(arr, Tensor) else _read(arr) for arr in (arrs or ([],))]
    if _builtins.any(tensor.ndim > 2 for tensor in tensors):
        raise ValueError("block_diag takes matrices: SciPy's batches of them are not taken")
    tensors = [tensor.reshape((1, tensor.size)) if tensor.ndim < 2 else tensor for tensor in tensors]
    dtype = _promoted([tensor.dtype for tensor in tensors])
    return _numerical._block_diag([tensor if tensor.dtype == dtype else asarray(tensor, dtype) for tensor in tensors])


def expm(A):
    """The matrix exponential ``e^A``, as ``scipy.linalg.expm``.

    Scaling and squaring of a Padé approximant of degree 3 to 13, as Al-Mohy and Higham's algorithm (2009) that SciPy
    follows: the same values to rounding, not to the bit. An infinite or NaN entry raises, where SciPy returns NaNs.

    Args:
        A: The square matrix, or a stack of them (``(..., n, n)``): a Tensor or an array-like; integers and booleans
            as ``float64``.

    Returns:
        Tensor: ``e^A``, of `A`'s shape and dtype.

    Raises:
        LinAlgError: if `A` is not square (or a stack of square matrices) or holds an infinite or NaN entry.

    Examples:
        >>> import numerical
        >>> numerical.expm([[0.0, 0.0], [0.0, 0.0]])
        numerical.Tensor([[1.0, 0.0],
                          [0.0, 1.0]], dtype=float64)
        >>> numerical.expm([[0.0, 1.0], [0.0, 0.0]])
        numerical.Tensor([[1.0, 1.0],
                          [0.0, 1.0]], dtype=float64)
    """
    return _expm(A)


def expm_multiply(A, B, start=None, stop=None, num=None, endpoint=None, traceA=None):
    """``e^{tA} B`` without forming ``e^{tA}``, as ``scipy.sparse.linalg.expm_multiply``: at ``t = 1``, or at each of
    the times ``numpy.linspace(start, stop, num, endpoint)`` gives (``start``, ``stop``, ``num`` and ``endpoint`` as
    ``linspace`` takes them, at least two times), stacked along a first axis. Al-Mohy and Higham's algorithm 3.2
    (2011), as SciPy's: `A` less its trace over `n`, then `s` steps of at most `m` terms of the Taylor series, each a
    product of `A` by `B`'s block, `m` and `s` the fewest products for double's unit roundoff; the times after the
    first each a step ``h`` from the one before, as SciPy's (its algorithm 5.2), except where the error the steps
    carry, grown at most ``‖e^{hA}‖₁`` times by each (estimated once, as ``onenormest``), would pass ``2⁻⁴⁰`` of the
    action: that time is computed from `B` again and the steps go on from it. Where `A` is far from normal or the time
    negative, the steps alone lose digits by orders (SciPy's: 7·10⁻¹⁰ where each time from `B` is right to 10⁻¹⁵).
    `A` dense; `traceA` is accepted (the trace is exact here).

    Raises:
        LinAlgError: if `A` is not square, `B` not a vector or matrix of its rows, or an entry is infinite or NaN.
        ValueError: for fewer than two times.
    """
    del traceA
    if start is None and stop is None and num is None and endpoint is None:
        return _expm_multiply(A, B, 1.0)
    linspace = {"retstep": True}
    if num is not None:
        linspace["num"] = num
    if endpoint is not None:
        linspace["endpoint"] = endpoint
    times, step = _linspace_times(start, stop, linspace)
    if len(times) < 2:
        raise ValueError("at least two time points are required")
    A = A if isinstance(A, Tensor) else _read(A)
    B = B if isinstance(B, Tensor) else _read(B)
    actions = [_expm_multiply(A, B, float(times[0]))]
    growth = _step_growth(A, float(step))  # ‖e^{hA}‖₁, estimated: how a step grows the rounding it carries
    carried = _EXPM_MULTIPLY_UNIT * _column_norm(actions[0])  # the error the steps carry, bounded
    for t in times[1:]:
        last = actions[-1]
        action = _expm_multiply(A, last, float(step))
        carried = growth * (carried + _EXPM_MULTIPLY_UNIT * _column_norm(last))
        if carried > _EXPM_MULTIPLY_KEPT * _column_norm(action):  # the steps' rounding grown past it: from B
            action = _expm_multiply(A, B, float(t))
            carried = _EXPM_MULTIPLY_UNIT * _column_norm(action)
        actions.append(action)
    return stack(actions)


_EXPM_MULTIPLY_UNIT = 2.0**-53  # double's unit roundoff
_EXPM_MULTIPLY_KEPT = 2.0**-40  # the error bound relative to an action past which a time is computed from B again


def _column_norm(x):
    """``‖x‖₁`` of a vector, or the largest column sum of moduli of a matrix (0 for none)."""
    if not x.size:
        return 0.0
    sums = sum(absolute(x), axis=0)
    return float(sums) if len(x.shape) == 1 else float(max(sums))


def _step_growth(A, h):
    """``‖e^{hA}‖₁`` estimated as Hager and Higham's ``onenormest`` (LAPACK's ``lacn2``), from products of ``e^{hA}``
    and its adjoint with vectors (:func:`_expm_multiply`), at most five rounds: a step of a sequence of times grows
    the error it carries by at most this."""
    n = A.shape[0]
    if n == 0:
        return 0.0
    adjoint = _adjoint(A)
    complex_ = A.dtype.startswith("complex")
    x = full((n,), 1.0 / n, A.dtype)
    estimate = 0.0
    for _ in range(5):
        y = _expm_multiply(A, x, h)
        estimate = _builtins.max(estimate, _column_norm(y))
        values = y.tolist()
        signs = [v / _builtins.abs(v) if v != 0 else 1.0 for v in values] if complex_ else [1.0 if v >= 0 else -1.0 for v in values]
        z = _expm_multiply(adjoint, asarray(signs, A.dtype), h).tolist()
        moduli = [_builtins.abs(v) for v in z]
        j = moduli.index(_builtins.max(moduli))
        here = _builtins.sum((complex(zi).conjugate() * xi).real for zi, xi in zip(z, x.tolist()))
        if moduli[j] <= here:
            break
        unit = [0.0] * n
        unit[j] = 1.0
        x = asarray(unit, A.dtype)
    return estimate


def _linspace_times(start, stop, options):
    """``numpy.linspace(start, stop, **options)``'s times and step as Python floats, without NumPy: num 50 and the
    endpoint included by default."""
    num = _operator.index(options.get("num", 50))
    endpoint = bool(options.get("endpoint", True))
    if num < 0:
        raise ValueError(f"Number of samples, {num}, must be non-negative.")
    start, stop = float(start), float(stop)
    div = (num - 1) if endpoint else num
    step = (stop - start) / div if div > 0 else _math.nan
    times = [start + (i * step) for i in range(num)] if div > 0 else [start] * num
    if endpoint and num > 1:
        times[-1] = stop
    return times, step


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
    """A function of a square matrix, as ``scipy.linalg.funm``.

    The Schur form ``A = Z T Zᴴ`` (a real matrix's real form made complex), `func` called with `T`'s diagonal, then
    Parlett's recurrence for the rest of ``f(T)`` and ``Z f(T) Zᴴ``. Where eigenvalues gather (within 0.1 of the
    largest modulus, or of 1), Davies and Higham's Schur–Parlett algorithm: the Schur form reordered by clusters, each
    cluster's block by Cauchy's integral on a circle about it (`func` is then called again with those circles'
    points, more of them where an integral's own estimate asks: it must be analytic on their disks), the rest by the
    block recurrence; where no circle can keep the other clusters out, or the estimate stays past ``1e3 eps``, the
    clusters within 0.01 instead. A Hermitian `A`: ``V f(Λ) Vᴴ`` from its eigendecomposition.

    Args:
        A: The square matrix: a Tensor or an array-like.
        func: The scalar function, called with an array of points (a NumPy array where NumPy is installed, else a
            :class:`Tensor`); it returns one value per point.
        disp: With true (the default) a likely inaccurate result (an estimate past ``1000 eps``) is reported on standard
            output, as SciPy's; with false the estimate is returned with the result.

    Returns:
        Tensor | tuple: ``f(A)``, real where `A` is and the imaginary parts are negligible, single precision for a
        single-precision `A`. With ``disp=False``, ``(F, err)``, `err` the estimate of ``‖error‖₁ / ‖A‖₁``: the
        recurrence run again on its own roundings and the integrals' errors, never below SciPy's (which far from
        normality can say 10⁻¹³ of an error of 10⁻²).

    Raises:
        LinAlgError: if `A` is not a square matrix or holds an infinite or NaN entry, or `func` does not return one
            value per point.

    Examples:
        >>> import numerical
        >>> numerical.funm([[1.0, 0.0], [0.0, 4.0]], lambda x: x * x)
        numerical.Tensor([[1.0, 0.0],
                          [0.0, 16.0]], dtype=float64)
    """
    A = A if isinstance(A, Tensor) else _read(A)
    try:
        import numpy as _np
    except ImportError:  # pragma: no cover
        _np = None

    def evaluate(points):  # f at the points funm asks for: a NumPy array of them where NumPy is installed
        argument = _np.asarray(points) if _np is not None else points
        values = [complex(x) for x in _flat(_read(func(argument)).tolist())]
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
    """The principal matrix logarithm, as ``scipy.linalg.logm``.

    An upper triangular `A` directly, else its Schur form (a real matrix's real form made complex) and
    ``Z log(T) Zᴴ``, ``log T`` by Al-Mohy and Higham's inverse scaling and squaring (2012); the norms that choose the
    square roots and the Padé degree are exact where SciPy estimates them (``onenormest``). Warns as SciPy does:
    :class:`LogmExactlySingularWarning` where an eigenvalue is 0 (taken as 1e-20), :class:`LogmNearlySingularWarning`
    where one is below 1e-20, and a ``RuntimeWarning`` where ``‖expm(F) − A‖₁ / ‖A‖₁`` is not below 1000 ε.

    Args:
        A: The square matrix, or a stack of them (``(..., n, n)``): a Tensor or an array-like.

    Returns:
        Tensor: The logarithm, real where `A` is and the imaginary parts are negligible, else complex; a stack's
        matrices each (complex for all where any is).

    Raises:
        LinAlgError: if `A` is not square (or a stack of square matrices) or holds an infinite or NaN entry.

    Examples:
        >>> import numerical
        >>> numerical.logm([[1.0, 1.0], [0.0, 1.0]])
        numerical.Tensor([[0.0, 1.0],
                          [0.0, 0.0]], dtype=float64)
    """
    A = A if isinstance(A, Tensor) else _read(A)
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
_eigh_subset = _linalg(_numerical.eigh_subset, arrays=1)
_tridiagonal_vectors = _linalg(_numerical.tridiagonal_vectors, arrays=3)
_eigh_pencil_subset = _linalg(_numerical.eigh_pencil_subset, arrays=2)
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
    """The principal square root `X` of a square matrix, as ``scipy.linalg.sqrtm``.

    ``X @ X = A``, its eigenvalues of nonnegative real part. By the Schur method, in real arithmetic for a real matrix
    (Higham, 1987), whose root is real unless it has a real eigenvalue below 0, then complex (for a whole stack); where
    there is no square root (a zero eigenvalue in a Jordan block), the entries that would need one infinite, as
    SciPy's. SciPy's deprecated `disp` and `blocksize` are not taken.

    Args:
        A: The square matrix, or a stack of them (``(..., n, n)``): a Tensor or an array-like; integers and booleans
            as ``float64``.

    Returns:
        Tensor: `X`, of `A`'s shape: real or complex as above.

    Raises:
        LinAlgError: if `A` is not square (or a stack of square matrices) or holds an infinite or NaN entry.

    Examples:
        >>> import numerical
        >>> numerical.sqrtm([[4.0, 0.0], [0.0, 9.0]])
        numerical.Tensor([[2.0, 0.0],
                          [0.0, 3.0]], dtype=float64)
        >>> numerical.sqrtm([[-4.0, 0.0], [0.0, 9.0]])
        numerical.Tensor([[2j, 0j],
                          [0j, (3+0j)]], dtype=complex128)
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
    a = a if isinstance(a, Tensor) else _read(a)
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
    """The solution `X` of ``A X + X Aᴴ = Q``, as ``scipy.linalg.solve_continuous_lyapunov``: one Schur form
    ``A = U R Uᴴ``, which gives ``Aᴴ = U Rᴴ Uᴴ`` too, then ``R Y + Y Rᴴ = Uᴴ Q U`` by blocks (LAPACK's ``trsyl``) and
    ``X = U Y Uᴴ``; complex where an operand is.

    Raises:
        LinAlgError: if `a` is not square, `q` not of its shape, or an entry is infinite or NaN.
    """
    return _solve_continuous_lyapunov(a, q)


solve_lyapunov = solve_continuous_lyapunov


def solve_discrete_lyapunov(a, q, method=None):
    """The solution `X` of ``A X Aᴴ − X + Q = 0``, as ``scipy.linalg.solve_discrete_lyapunov``: ``method="direct"``
    (by default below order 10, as SciPy's) solves the Kronecker system ``(I − A ⊗ conj(A)) vec(X) = vec(Q)``,
    ``"bilinear"`` (from order 10) the continuous equation the bilinear transformation gives, ``Bᴴ X + X B = −C`` for
    ``B = (Aᴴ − I)(Aᴴ + I)⁻¹`` and ``C = 2 (A + I)⁻¹ Q (Aᴴ + I)⁻¹``, from one LU factorisation of ``A + I``.

    Raises:
        LinAlgError: if `a` is not square or `q` not of its shape, an entry is infinite or NaN, or ``I − A ⊗ conj(A)``
            (direct) or ``A + I`` (bilinear) is singular.
        ValueError: for another `method`.
    """
    a = a if isinstance(a, Tensor) else _read(a)
    q = q if isinstance(q, Tensor) else _read(q)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or q.shape != a.shape:
        raise LinAlgError("solve_discrete_lyapunov: a must be square and q of its shape")
    n = a.shape[0]
    if method is None:
        method = "direct" if n < 10 else "bilinear"
    if method == "direct":
        kron = einsum("ij,kl->ikjl", a, conj(a) if a.dtype.startswith("complex") else a).reshape((n * n, n * n))
        return solve(eye(n * n, kron.dtype) - kron, q.reshape((n * n,))).reshape((n, n))
    if method == "bilinear":  # one factorisation of A + I: (Aᴴ + I)⁻¹ is its inverse's adjoint
        identity = eye(n, a.dtype)
        factor = lu_factor(a + identity)
        left = lu_solve(factor, q)  # (A + I)⁻¹ Q
        c = 2 * _adjoint(lu_solve(factor, _adjoint(left)))  # (A + I)⁻¹ Q (Aᴴ + I)⁻¹
        return solve_continuous_lyapunov(lu_solve(factor, a - identity), -c)  # Bᴴ = (A + I)⁻¹ (A − I)
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
    a, b, q, r = (x if isinstance(x, Tensor) else _read(x) for x in (a, b, q, r))
    given = [x if x is None or isinstance(x, Tensor) else _read(x) for x in (e, s)]
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


def _subset(name, a, subset_by_index, subset_by_value):
    """``(by_value, low, high)`` for :func:`eigh`'s and :func:`eigvalsh`'s subsets, checked as ``scipy.linalg.eigh``
    checks them, or None for the whole spectrum; `a` read for its order."""
    if subset_by_index is None and subset_by_value is None:
        return None
    if subset_by_index is not None and subset_by_value is not None:
        raise ValueError("Either index or value subset can be requested.")
    a = _operand(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise LinAlgError(f"{name}: a subset of the spectrum of a square matrix, not of shape {a.shape}")
    n = a.shape[0]
    if subset_by_index is not None:
        lo, hi = (int(x) for x in subset_by_index)
        if not 0 <= lo <= hi < n:
            raise ValueError("Requested eigenvalue indices are not valid. "
                             f"Valid range is [0, {n - 1}] and start <= end, but start={lo}, end={hi} is given")
        return False, float(lo), float(hi)
    lo, hi = subset_by_value
    if not -_math.inf <= lo < hi <= _math.inf:
        raise ValueError("Requested eigenvalue bounds are not valid. "
                         f"Valid range is (-inf, inf) and low < high, but low={lo}, high={hi} is given")
    return True, float(lo), float(hi)


def eigh(a, UPLO="L", *, b=None, method="divide_and_conquer", subset_by_index=None, subset_by_value=None, lower=None,
         eigvals_only=False, overwrite_a=False, overwrite_b=False, type=1, check_finite=True, driver=None):
    """The eigenvalues and eigenvectors of a real symmetric or complex Hermitian matrix, as ``numpy.linalg.eigh``.

    The eigenvalues are real and ascending, the eigenvectors the columns of the second matrix. With `b`, a
    positive-definite matrix of `a`'s order, the pencil ``a x = λ b x``, as ``scipy.linalg.eigh(a, b)``: ``b = L Lᴴ``
    and ``L⁻¹ a L⁻ᴴ`` diagonalised, the vectors ``b``-orthonormal (``Xᴴ b X = I``); each matrix read in the triangle
    `UPLO` names. Two matrices, not stacks; `b` keyword-only, `UPLO` holding NumPy's place.

    A subset of the spectrum (`subset_by_index` or `subset_by_value`, as ``scipy.linalg.eigh``'s) comes from the
    tridiagonal form, the eigenvalues by bisection, the vectors by inverse iteration (LAPACK's ``dstebz`` and
    ``dstein``): ``n² k`` operations past the reduction for ``k`` of them, where `method` takes ``n³`` and more for
    all; a matrix (and `b`), not a stack. With `eigvals_only` or a subset, `method` is checked but not used.

    ``scipy.linalg.eigh``'s arguments are taken too: `b` as the second positional argument (a matrix, where `UPLO` is
    a string), `lower`, `eigvals_only`, `driver`, `type` 1 only (``a b x = λ x`` and ``b a x = λ x`` are not offered),
    and `overwrite_a`, `overwrite_b` and `check_finite`.

    Args:
        a: The symmetric or Hermitian matrix, or a stack of them (``(..., n, n)``) without `b` or a subset: a Tensor
            or an array-like. Only the triangle `UPLO` names is read.
        UPLO: ``"L"`` (the default) reads the lower triangle, ``"U"`` the upper one, in either case. A matrix there is
            SciPy's `b`.
        b: The positive-definite matrix of the pencil ``a x = λ b x``, of `a`'s order.
        method: How the tridiagonal form is diagonalised: ``"divide_and_conquer"`` (LAPACK's ``dsyevd``/``zheevd``),
            ``"tridiagonal_ql"`` (implicit QL, ``dsyev``'s) or ``"jacobi"`` (cyclic Jacobi rotations, real matrices
            only).
        subset_by_index: ``[lo, hi]``: the eigenpairs of indices ``lo`` to ``hi`` (ascending, from 0).
        subset_by_value: ``[low, high]``: the eigenpairs of the eigenvalues in ``(low, high]``.
        lower: SciPy's name for the triangle: true for ``UPLO="L"``, false for ``"U"``; it overrides `UPLO`.
        eigvals_only: Return the eigenvalues alone.
        overwrite_a: Accepted for SciPy's signature; ignored.
        overwrite_b: Accepted for SciPy's signature; ignored.
        type: SciPy's problem type: 1 only (``a x = λ b x``).
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).
        driver: SciPy's driver, as a `method`: ``"ev"`` and ``"gv"`` the implicit QL, ``"evd"`` and SciPy's others
            divide and conquer, or the subset's bisection where one is asked.

    Returns:
        tuple | Tensor: ``(w, v)``: `w` the eigenvalues, real, of `a`'s precision; `v` the eigenvectors as columns, of
        `a`'s dtype (float64 for integers or bools). `w` alone with `eigvals_only`.

    Raises:
        LinAlgError: if `a` is not square, holds an infinite or NaN entry, `b` is not positive definite or not of
            `a`'s order, or ``"jacobi"`` is asked of a complex matrix.
        ValueError: for a `UPLO`, `method`, `driver` or `type` not offered, both subsets, or a subset out of range.
        TypeError: if a method's name is given as the second argument, or `b` is given twice.

    Examples:
        >>> import numerical
        >>> w, v = numerical.eigh([[2.0, 0.0], [0.0, 1.0]])
        >>> w
        numerical.Tensor([1.0, 2.0], dtype=float64)
        >>> v
        numerical.Tensor([[0.0, 1.0],
                          [1.0, 0.0]], dtype=float64)
        >>> numerical.eigh([[2.0, 0.0], [0.0, 1.0]], b=[[4.0, 0.0], [0.0, 1.0]], eigvals_only=True)
        numerical.Tensor([0.5, 1.0], dtype=float64)
    """
    UPLO, b = _scipy_pencil("eigh", UPLO, b, lower, type)
    upper = _upper("eigh", UPLO)
    method = _eigh_driver(driver, method)
    del overwrite_a, overwrite_b, check_finite
    if method not in _EIGH_METHODS:
        raise ValueError(f"eigh: method {method!r} is none of {_EIGH_METHODS}")
    subset = _subset("eigh", a, subset_by_index, subset_by_value)
    if subset is not None:
        pairs = (_eigh_pencil_subset(a, b, *subset, not eigvals_only, upper) if b is not None
                 else _eigh_subset(a, *subset, not eigvals_only, upper))
        return pairs[0] if eigvals_only else pairs
    if eigvals_only:
        return _eigvalsh_pencil(a, b, upper) if b is not None else _eigvalsh(a, upper)
    if b is not None:
        return _eigh_pencil(a, b, method, upper)
    return _eigh(a, method, upper)


_EIGH_DRIVERS = {"ev": "tridiagonal_ql", "evd": "divide_and_conquer", "evr": "divide_and_conquer",
                 "evx": "divide_and_conquer", "gv": "tridiagonal_ql", "gvd": "divide_and_conquer",
                 "gvx": "divide_and_conquer"}


def _eigh_driver(driver, method):
    """``scipy.linalg.eigh``'s `driver` as a `method` (None keeps `method`)."""
    if driver is None:
        return method
    if driver not in _EIGH_DRIVERS:
        raise ValueError(f"driver must be one of {tuple(_EIGH_DRIVERS)}, not {driver!r}")
    return _EIGH_DRIVERS[driver]


def _scipy_pencil(name, UPLO, b, lower, type):
    """`UPLO` and `b` from NumPy's or SciPy's arguments: a second positional argument that is not a string is
    SciPy's `b`; `lower` names the triangle; `type` must be 1."""
    if type != 1:
        raise ValueError(f"{name}: type {type!r}: only type 1 (a x = λ b x) is offered")
    if UPLO is not None and not isinstance(UPLO, str):
        if b is not None:
            raise TypeError(f"{name}: b given twice (as the second argument and by name)")
        UPLO, b = "L", UPLO
    if lower is not None:
        UPLO = "L" if lower else "U"
    return UPLO, b


def eigvalsh(a, UPLO="L", *, b=None, subset_by_index=None, subset_by_value=None, lower=None, overwrite_a=False,
             overwrite_b=False, type=1, check_finite=True, driver=None):
    """The eigenvalues alone of a real symmetric or complex Hermitian matrix, as ``numpy.linalg.eigvalsh``.

    Real and ascending. With `b`, those of the pencil ``a x = λ b x``, as :func:`eigh`'s. A subset (`subset_by_index`
    or `subset_by_value`, as :func:`eigh`'s) by bisection on the tridiagonal form alone. ``scipy.linalg.eigvalsh``'s
    arguments as :func:`eigh` takes them.

    Args:
        a: The symmetric or Hermitian matrix, or a stack of them (``(..., n, n)``) without `b` or a subset: a Tensor
            or an array-like. Only the triangle `UPLO` names is read.
        UPLO: ``"L"`` (the default) reads the lower triangle, ``"U"`` the upper one, in either case. A matrix there is
            SciPy's `b`.
        b: The positive-definite matrix of the pencil ``a x = λ b x``, of `a`'s order.
        subset_by_index: ``[lo, hi]``: the eigenvalues of indices ``lo`` to ``hi`` (ascending, from 0).
        subset_by_value: ``[low, high]``: the eigenvalues in ``(low, high]``.
        lower: SciPy's name for the triangle: true for ``UPLO="L"``, false for ``"U"``; it overrides `UPLO`.
        overwrite_a: Accepted for SciPy's signature; ignored.
        overwrite_b: Accepted for SciPy's signature; ignored.
        type: SciPy's problem type: 1 only (``a x = λ b x``).
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).
        driver: SciPy's driver: checked against :func:`eigh`'s names, it does not change the computation.

    Returns:
        Tensor: The eigenvalues, real, of `a`'s precision; for a stack, one row per matrix.

    Raises:
        LinAlgError: if `a` is not square, holds an infinite or NaN entry, or `b` is not positive definite or not of
            `a`'s order.
        ValueError: for a `UPLO`, `driver` or `type` not offered, both subsets, or a subset out of range.
        TypeError: if a method's name is given as the second argument, or `b` is given twice.

    Examples:
        >>> import numerical
        >>> numerical.eigvalsh([[2.0, 7.0], [0.0, 1.0]])
        numerical.Tensor([1.0, 2.0], dtype=float64)
        >>> numerical.eigvalsh([[2.0, 0.0], [7.0, 1.0]], "U")
        numerical.Tensor([1.0, 2.0], dtype=float64)
    """
    UPLO, b = _scipy_pencil("eigvalsh", UPLO, b, lower, type)
    upper = _upper("eigvalsh", UPLO)
    _eigh_driver(driver, None)
    del overwrite_a, overwrite_b, check_finite
    subset = _subset("eigvalsh", a, subset_by_index, subset_by_value)
    if subset is not None:
        return (_eigh_pencil_subset(a, b, *subset, False, upper) if b is not None
                else _eigh_subset(a, *subset, False, upper))[0]
    if b is not None:
        return _eigvalsh_pencil(a, b, upper)
    return _eigvalsh(a, upper)


def eig(a, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True,
        homogeneous_eigvals=False):
    """The eigenvalues and right eigenvectors of a general square matrix or a stack of them, as ``numpy.linalg.eig``.

    ``a @ v[:, j] == w[j] * v[:, j]``, each vector of unit norm with its component of largest modulus real. Complex
    whatever the eigenvalues, as NumPy 2.5's (complex64 for a single-precision matrix); in the order LAPACK's ``dgeev``
    and ``zgeev`` give them (balancing, Hessenberg reduction, shifted QR), a real matrix's conjugate pairs together, the
    positive imaginary part first.

    With ``scipy.linalg.eig``'s arguments (a matrix, not a stack): `left`, `right`, `b` and `homogeneous_eigvals`, the
    answer ``w``, ``(w, vl)``, ``(w, vr)`` or ``(w, vl, vr)`` as SciPy's. The vectors are complex always, where SciPy's
    are real for a real matrix with real eigenvalues.

    Args:
        a: The square matrix, or a stack of them (``(..., n, n)``) with NumPy's arguments alone: a Tensor or an
            array-like.
        b: The generalized problem ``a @ v = w * b @ v``, by the complex QZ: ``w = alpha / beta``, ∞ where only β is 0,
            NaN where both are; each vector of unit 2-norm, as SciPy normalises ``ggev``'s. A matrix of `a`'s shape.
        left: Return the left eigenvectors too (``vl[:, j].conj() @ a == w[j] * vl[:, j].conj()``, as ``geev``'s).
        right: Return the right eigenvectors (the default); false for none of them.
        overwrite_a: Accepted for SciPy's signature; ignored.
        overwrite_b: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).
        homogeneous_eigvals: Return the eigenvalues as the pairs ``(alpha, beta)`` stacked (a ``(2, n)`` Tensor; β 1
            without `b`).

    Returns:
        tuple | Tensor: ``(w, v)`` by default: `w` the eigenvalues, `v` the eigenvectors as columns, both complex. With
        SciPy's arguments, ``w``, ``(w, vl)``, ``(w, vr)`` or ``(w, vl, vr)``.

    Raises:
        LinAlgError: if `a` is not square (or a stack of square matrices) or holds an infinite or NaN entry.
        ValueError: if, with `b`, `a` is not a square matrix or `b` is not of its shape.

    Examples:
        >>> import numerical
        >>> w, v = numerical.eig([[2.0, 0.0], [0.0, 1.0]])
        >>> w
        numerical.Tensor([(2+0j), (1+0j)], dtype=complex128)
        >>> numerical.eig([[0.0, -1.0], [1.0, 0.0]], right=False)
        numerical.Tensor([1j, -1j], dtype=complex128)
    """
    del overwrite_a, overwrite_b, check_finite
    if b is None and not left and right and not homogeneous_eigvals:
        return _eig(a)
    a = _operand(a)
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
        b = _operand(b)
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
    """The eigenvalues alone of a general square matrix, or of each of a stack, as ``numpy.linalg.eigvals``.

    Complex whatever they are, as NumPy 2.5's; the same values as :func:`eig`'s, to the bit.

    Args:
        a: The square matrix, or a stack of them (``(..., n, n)``) without `b` or `homogeneous_eigvals`: a Tensor or
            an array-like.
        b: The generalized problem's matrix, as ``scipy.linalg.eigvals`` (:func:`eig`'s).
        overwrite_a: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).
        homogeneous_eigvals: Return the pairs ``(alpha, beta)`` stacked, as :func:`eig`'s.

    Returns:
        Tensor: The eigenvalues, complex (complex64 for a single-precision matrix); for a stack, one row per matrix.

    Raises:
        LinAlgError: if `a` is not square (or a stack of square matrices) or holds an infinite or NaN entry.
        ValueError: if, with `b`, `a` is not a square matrix or `b` is not of its shape.

    Examples:
        >>> import numerical
        >>> numerical.eigvals([[0.0, -1.0], [1.0, 0.0]])
        numerical.Tensor([1j, -1j], dtype=complex128)
        >>> numerical.eigvals([[2.0, 0.0], [0.0, 3.0]], b=[[1.0, 0.0], [0.0, 0.0]])
        numerical.Tensor([(2+0j), (inf+0j)], dtype=complex128)
    """
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


def cholesky(a, /, *, upper=False, lower=None, overwrite_a=False, check_finite=True):
    """The Cholesky factor of a Hermitian positive-definite matrix, as ``numpy.linalg.cholesky``.

    The lower `L` with ``a = L @ L.conj().T``, read from the lower triangle, or the upper `U` with
    ``a = U.conj().T @ U``, read from the upper triangle. NumPy's default stays, the lower factor, where
    ``scipy.linalg.cholesky``'s is the upper one.

    Args:
        a: The Hermitian positive-definite matrix, or a stack of them (``(..., n, n)``): a Tensor or an array-like.
        upper: Return the upper factor `U`.
        lower: ``scipy.linalg.cholesky``'s: the negation of `upper`, which it overrides.
        overwrite_a: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).

    Returns:
        Tensor: The factor, of `a`'s shape and dtype (float64 for integers or bools), zeros in its other triangle.

    Raises:
        LinAlgError: if `a` is not positive definite, not square, or holds an infinite or NaN entry.

    Examples:
        >>> import numerical
        >>> numerical.cholesky([[4.0, 2.0], [2.0, 5.0]])
        numerical.Tensor([[2.0, 0.0],
                          [1.0, 2.0]], dtype=float64)
        >>> numerical.cholesky([[4.0, 2.0], [2.0, 5.0]], upper=True)
        numerical.Tensor([[2.0, 1.0],
                          [0.0, 2.0]], dtype=float64)
    """
    if lower is not None:
        upper = not lower
    del overwrite_a, check_finite
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
    """The solution `x` of ``a @ x = b`` for a triangular `a`, as ``scipy.linalg.solve_triangular``.

    The triangle `lower` names is read and the other never. ``n² k`` operations where :func:`solve` factorises `a`
    first.

    Args:
        a: The triangular square matrix: a Tensor or an array-like.
        b: The right-hand side: a vector or a matrix of right-hand sides.
        trans: 0 (``'N'``) solves ``a @ x = b``, 1 (``'T'``) ``a.T @ x = b``, 2 (``'C'``) ``a.conj().T @ x = b``.
        lower: Read the lower triangle; by default the upper one.
        unit_diagonal: Take the diagonal as ones (it is not read).
        overwrite_b: Accepted for SciPy's signature; ignored (a Tensor is never written through).
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).

    Returns:
        Tensor: `x`, of `b`'s shape.

    Raises:
        LinAlgError: if a diagonal entry read is zero ("singular matrix: resolution failed at diagonal i", 1-based,
            as SciPy's), an entry read is infinite or NaN, or the shapes do not fit.
        ValueError: for another `trans`.

    Examples:
        >>> import numerical
        >>> numerical.solve_triangular([[2.0, 1.0], [0.0, 4.0]], [4.0, 8.0])
        numerical.Tensor([1.0, 2.0], dtype=float64)
        >>> numerical.solve_triangular([[2.0, 0.0], [1.0, 4.0]], [2.0, 9.0], lower=True)
        numerical.Tensor([1.0, 2.0], dtype=float64)
        >>> numerical.solve_triangular([[1.0, 0.0], [1.0, 1.0]], [1.0, 2.0], lower=True, unit_diagonal=True)
        numerical.Tensor([1.0, 1.0], dtype=float64)
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
    """The pivoted LU factorisation ``a = P @ L @ U`` of an m × n matrix, as ``scipy.linalg.lu``.

    A singular matrix factorises (a zero pivot leaves its column of `L` zero).

    Args:
        a: The m × n matrix: a Tensor or an array-like.
        permute_l: Return the pair ``(P @ L, U)``.
        overwrite_a: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).
        p_indices: Return the rows themselves in place of `P` (``a == L[P] @ U``, an int list where SciPy's is an
            array).

    Returns:
        tuple: ``(P, L, U)``: `P` the m × m permutation matrix (real, of `a`'s precision), `L` m × k lower trapezoidal
        with a unit diagonal, `U` k × n upper trapezoidal, ``k = min(m, n)``. ``(P @ L, U)`` with `permute_l`.

    Raises:
        LinAlgError: if `a` is not a matrix or holds an infinite or NaN entry.

    Examples:
        >>> import numerical
        >>> p, l, u = numerical.lu([[0.0, 1.0], [2.0, 0.0]])
        >>> p
        numerical.Tensor([[0.0, 1.0],
                          [1.0, 0.0]], dtype=float64)
        >>> u
        numerical.Tensor([[2.0, 0.0],
                          [0.0, 1.0]], dtype=float64)
        >>> numerical.lu([[0.0, 1.0], [2.0, 0.0]], p_indices=True)[0]
        [1, 0]
    """
    del overwrite_a, check_finite
    a = _operand(a)
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
    """The pivoted LU factorisation ``P a = L U`` of a square matrix, as ``scipy.linalg.lu_factor``.

    To solve many systems with one factorisation (:func:`lu_solve`). A singular matrix factorises, a zero on `U`'s
    diagonal, as SciPy's (which warns).

    Args:
        a: The square matrix: a Tensor or an array-like.
        overwrite_a: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).

    Returns:
        LUFactor: The pair ``(lu, piv)``: `L` (unit diagonal not stored) and `U` in one matrix, and LAPACK's row
        interchanges (a tuple of ints, where SciPy's is an integer array).

    Raises:
        LinAlgError: if `a` is not square or holds an infinite or NaN entry.

    Examples:
        >>> import numerical
        >>> lu, piv = numerical.lu_factor([[0.0, 1.0], [2.0, 0.0]])
        >>> lu
        numerical.Tensor([[2.0, 0.0],
                          [0.0, 1.0]], dtype=float64)
        >>> piv
        (1, 1)
    """
    del overwrite_a, check_finite
    lu, piv = _lu_factor(a)
    return LUFactor(lu, tuple(piv))


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """The solution of a system from :func:`lu_factor`'s factorisation, as ``scipy.linalg.lu_solve``.

    ``n² k`` operations. A zero on `U`'s diagonal raises, where SciPy's divides by it.

    Args:
        lu_and_piv: :func:`lu_factor`'s ``(lu, piv)``, or any such pair.
        b: The right-hand side: a vector or a matrix of right-hand sides.
        trans: 0 (``'N'``) solves ``a @ x = b``, 1 (``'T'``) ``a.T @ x = b``, 2 (``'C'``) ``a.conj().T @ x = b``.
        overwrite_b: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).

    Returns:
        Tensor: `x`, of `b`'s shape, in the dtype `lu` and `b` promote to.

    Raises:
        LinAlgError: for a zero on `U`'s diagonal, an infinite or NaN entry, or shapes that do not fit.
        ValueError: for another `trans`.

    Examples:
        >>> import numerical
        >>> factor = numerical.lu_factor([[0.0, 1.0], [2.0, 0.0]])
        >>> numerical.lu_solve(factor, [1.0, 4.0])
        numerical.Tensor([2.0, 1.0], dtype=float64)
        >>> numerical.lu_solve(numerical.lu_factor([[1.0, 2.0], [0.0, 4.0]]), [1.0, 4.0], trans=1)
        numerical.Tensor([1.0, 0.5], dtype=float64)
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
    lu = lu if isinstance(lu, Tensor) else _read(lu)
    b = b if isinstance(b, Tensor) else _read(b)
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
    """The Cholesky factorisation of a Hermitian positive-definite matrix, as ``scipy.linalg.cho_factor``.

    For :func:`cho_solve`. The other triangle of the factor is zeros, where SciPy's leaves whatever `a` held there.

    Args:
        a: The Hermitian positive-definite matrix: a Tensor or an array-like.
        lower: Return the lower factor `L` (``a = L Lᴴ``) read from `a`'s lower triangle; by default the upper `U`
            (``a = Uᴴ U``) read from its upper one.
        overwrite_a: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).

    Returns:
        CholeskyFactor: The pair ``(c, lower)``: `c` the factor, `lower` a bool.

    Raises:
        LinAlgError: if `a` is not positive definite, not square, or holds an infinite or NaN entry.

    Examples:
        >>> import numerical
        >>> numerical.cho_factor([[4.0, 2.0], [2.0, 5.0]])
        (numerical.Tensor([[2.0, 1.0],
                          [0.0, 2.0]], dtype=float64), False)
    """
    del overwrite_a, check_finite
    return CholeskyFactor(cholesky(a, upper=not lower), bool(lower))


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """The solution of ``a @ x = b`` from :func:`cho_factor`'s factorisation, as ``scipy.linalg.cho_solve``.

    Two triangular solves (:func:`solve_triangular`'s bits), ``2 n² k`` operations; only `c`'s triangle is read,
    checked once with `b`, and an overflow between the two solves is the result's.

    Args:
        c_and_lower: :func:`cho_factor`'s ``(c, lower)``, or any such pair.
        b: The right-hand side: a vector or a matrix of right-hand sides.
        overwrite_b: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).

    Returns:
        Tensor: `x`, of `b`'s shape.

    Raises:
        LinAlgError: for a zero on `c`'s diagonal, an infinite or NaN entry, or shapes that do not fit.

    Examples:
        >>> import numerical
        >>> factor = numerical.cho_factor([[4.0, 2.0], [2.0, 5.0]])
        >>> numerical.cho_solve(factor, [6.0, 12.0])
        numerical.Tensor([0.375, 2.25], dtype=float64)
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


def lstsq(a, b, rcond=None, *, cond=None, overwrite_a=False, overwrite_b=False, check_finite=True, lapack_driver=None):
    """The minimum-norm least-squares solution of ``a @ x ≈ b``, as ``numpy.linalg.lstsq``.

    A singular value at or under `rcond` times the largest counts as zero. The epsilon is the operands' precision's
    (float32's for float32 and complex64). ``scipy.linalg.lstsq``'s arguments are taken; the SVD solves whichever
    driver is named.

    Args:
        a: The m×n matrix: a Tensor or an array-like.
        b: The right-hand side: a vector of m entries, or an m×k matrix of right-hand sides.
        rcond: The relative cutoff: by default (``None``) machine epsilon times max(m, n), NumPy's; a negative `rcond`
            is machine epsilon, as LAPACK's.
        cond: ``scipy.linalg.lstsq``'s name for `rcond` (not both).
        overwrite_a: Accepted for SciPy's signature; ignored.
        overwrite_b: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored.
        lapack_driver: ``"gelsd"``, ``"gelsy"`` or ``"gelss"``, as SciPy's: the SVD here, whichever.

    Returns:
        tuple: ``(x, residuals, rank, s)``: the solution (a Tensor), the residual sums of squares (a Tensor, for a tall
        `a` of full column rank, else empty), the rank (an int) and the singular values of `a` (a Tensor). With `cond`
        or `lapack_driver` (a SciPy-style call) and a vector `b`, the residual is a number, as SciPy's.

    Raises:
        LinAlgError: if `a` is not a matrix, `b` does not fit it, an entry is infinite or NaN, or the tolerance is NaN.
        ValueError: if both `rcond` and `cond` are given, or for another `lapack_driver`.
        TypeError: if the tolerance is not a real number.

    Examples:
        >>> import numerical
        >>> x, residuals, rank, s = numerical.lstsq([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], [1.0, 2.0, 3.0])
        >>> x
        numerical.Tensor([1.0, 1.0], dtype=float64)
        >>> residuals, rank, s
        (numerical.Tensor([9.0], dtype=float64), 2, numerical.Tensor([2.0, 1.0], dtype=float64))
    """
    del overwrite_a, overwrite_b, check_finite
    if lapack_driver not in (None, "gelsd", "gelsy", "gelss"):
        raise ValueError(f"LAPACK driver {lapack_driver!r} is not found")
    if cond is not None:
        if rcond is not None:
            raise ValueError("lstsq: rcond and cond can't be both set")
        rcond = cond
    rcond = _tolerance("lstsq", rcond)
    a = a if isinstance(a, Tensor) else _read(a)
    b = b if isinstance(b, Tensor) else _read(b)
    if rcond is not None and rcond < 0:
        rcond = _MACHINE_EPSILON["float32" if _promoted((a.dtype, b.dtype)) in _SINGLE else "float64"]
    answer = _lstsq(a, b, rcond)
    if (cond is not None or lapack_driver is not None) and len(b.shape) == 1 and answer[1].shape == (1,):
        x, residuals, rank, values = answer  # a SciPy-style call: SciPy's residue of a vector b is a number
        return x, float(residuals.tolist()[0]), rank, values
    return answer


def pinv(a, rcond=None, hermitian=False, *, rtol=None, atol=None, return_rank=False, check_finite=True):
    """The Moore–Penrose pseudo-inverse, as ``numpy.linalg.pinv``.

    A singular value at or under the relative tolerance times the largest counts as zero. A negative or NaN tolerance
    raises, where NumPy's result is infinite or NaN. ``scipy.linalg.pinv``'s `atol`, `return_rank` and `check_finite`
    are taken.

    Args:
        a: The matrix, or a stack of them (``(..., m, n)``) without `atol` or `return_rank`: a Tensor or an array-like.
        rcond: The relative tolerance, by default machine epsilon times max(m, n).
        hermitian: Take `a` as real symmetric or complex Hermitian, its lower triangle read: the pseudo-inverse comes
            from its eigendecomposition, cheaper.
        rtol: NumPy 2's name for `rcond` (not both).
        atol: ``scipy.linalg.pinv``'s absolute bound: a singular value at or under ``max(atol, rtol σ₁)`` counts as
            zero, `rtol` then 0 by default, as SciPy's.
        return_rank: Return the pair ``(pinv, rank)``, as SciPy's.
        check_finite: Accepted for SciPy's signature; ignored.

    Returns:
        Tensor | tuple: The pseudo-inverse, n×m (of `a`'s dtype, float64 for integers or bools); with `return_rank`,
        ``(pinv, rank)``, `rank` an int.

    Raises:
        LinAlgError: for a negative or NaN tolerance, an infinite or NaN entry, a non-square `a` with `hermitian`, or
            a stack with `atol` or `return_rank`.
        ValueError: if both `rcond` and `rtol` are given.
        TypeError: if a tolerance is not a real number (a bool included).

    Examples:
        >>> import numerical
        >>> numerical.pinv([[2.0, 0.0], [0.0, 4.0]])
        numerical.Tensor([[0.5, 0.0],
                          [0.0, 0.25]], dtype=float64)
        >>> numerical.pinv([[2.0, 0.0], [0.0, 0.0]], return_rank=True)
        (numerical.Tensor([[0.5, 0.0],
                          [0.0, 0.0]], dtype=float64), 1)
    """
    del check_finite
    if rcond is not None and rtol is not None:
        raise ValueError("pinv: rtol and rcond can't be both set")
    relative = _tolerance("pinv", rcond if rcond is not None else rtol)
    atol = _tolerance("pinv", atol)
    if atol is None and not return_rank:
        return _pinv(a, relative, bool(hermitian))
    a = _operand(a)
    if len(a.shape) != 2:
        raise LinAlgError("pinv: atol and return_rank take a matrix")
    values = [_builtins.abs(v) for v in (eigvalsh(a) if hermitian else svdvals(a)).tolist()]
    largest = _builtins.max(values, default=0.0)
    if relative is None:
        relative = 0.0 if atol else _builtins.max(a.shape) * _MACHINE_EPSILON["float32" if a.dtype in _SINGLE else "float64"]
    cutoff = _builtins.max(atol or 0.0, relative * largest)
    # the relative bound pinv applies, the absolute one folded in (a zero matrix: any)
    inverse_ = _pinv(a, cutoff / largest if largest > 0 else relative, bool(hermitian))
    if not return_rank:
        return inverse_
    return inverse_, _builtins.sum(1 for v in values if v > cutoff)


def matrix_rank(A, tol=None, hermitian=False, *, rtol=None):
    """The numerical rank, as ``numpy.linalg.matrix_rank``.

    The number of singular values above the tolerance. A vector or a number has rank 1 unless all zero. A negative or
    NaN tolerance raises.

    Args:
        A: The matrix, or a stack of them (``(..., m, n)``), a vector or a number: a Tensor or an array-like.
        tol: The absolute tolerance; by default machine epsilon times max(m, n) times the largest singular value.
        hermitian: Take `A` as real symmetric or complex Hermitian, its lower triangle read, its singular values the
            magnitudes of its eigenvalues, cheaper.
        rtol: The tolerance relative to the largest singular value (not with `tol`).

    Returns:
        int | list: The rank; for a stack, the ranks as integers nested along its leading axes (NumPy's an integer
        array).

    Raises:
        LinAlgError: for a negative or NaN tolerance, or an infinite or NaN entry.
        ValueError: if both `tol` and `rtol` are given.
        TypeError: if a tolerance is not a real number (a bool included).

    Examples:
        >>> import numerical
        >>> numerical.matrix_rank([[1.0, 2.0], [2.0, 4.0]])
        1
        >>> numerical.matrix_rank([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]])
        [2, 1]
        >>> numerical.matrix_rank([0.0, 0.0])
        0
    """
    if tol is not None and rtol is not None:
        raise ValueError("matrix_rank: tol and rtol can't be both set")
    tol, rtol = _tolerance("matrix_rank", tol), _tolerance("matrix_rank", rtol)
    if not isinstance(A, Tensor):
        A = _read(A)
    if A.ndim < 2:  # NumPy's: a vector or a number is of rank 1 unless all zero
        values = A.tolist()
        return int(_builtins.any(value != 0 for value in (values if A.ndim == 1 else [values])))
    if A.ndim > 2:  # a stack: its ranks, integers nested along its leading axes (NumPy's an integer array)
        return _as_integers(_matrix_ranks(A, rtol, tol, bool(hermitian)).tolist())
    return _matrix_rank(A, rtol, tol, bool(hermitian))


def _as_integers(nested):
    """The numbers of a nested list as integers, nested alike."""
    return [_as_integers(item) for item in nested] if isinstance(nested, list) else int(nested)


def svd(a, full_matrices=True, compute_uv=True, hermitian=False, *, method="divide_and_conquer", overwrite_a=False,
        check_finite=True, lapack_driver=None):
    """The singular value decomposition ``a = U @ diag(S) @ Vh``, as ``numpy.linalg.svd``.

    The singular values descending. ``hermitian=True`` decomposes `a` through its eigendecomposition, cheaper: the
    singular values the magnitudes of the eigenvalues, ``Vh = (U * sign(w)).conj().T``, as NumPy's.

    Args:
        a: The m×n matrix, or a stack of them (``(..., m, n)``): a Tensor or an array-like.
        full_matrices: `U` m×m and `Vh` n×n (true, NumPy's default: the columns past min(m, n) an orthonormal basis of
            the complement), or m×k and k×n, k = min(m, n) (false).
        compute_uv: Return the factors; false for the singular values alone.
        hermitian: Take `a` as real symmetric or complex Hermitian (its lower triangle read).
        method: How the bidiagonal form is decomposed: ``"divide_and_conquer"`` (LAPACK's ``dgesdd``),
            ``"bidiagonal_qr"`` (implicit-shift QR, ``dgesvd``'s) or ``"jacobi"`` (one-sided Jacobi, each singular
            value to high relative accuracy).
        overwrite_a: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).
        lapack_driver: ``scipy.linalg.svd``'s: ``"gesdd"`` (divide and conquer) or ``"gesvd"`` (bidiagonal QR); it
            replaces `method`.

    Returns:
        tuple | Tensor: ``(U, S, Vh)``: `S` real, of `a`'s precision, `U` and `Vh` of `a`'s dtype (float64 for
        integers or bools). `S` alone with ``compute_uv=False``.

    Raises:
        LinAlgError: if `a` is not a matrix (or a stack of them) or holds an infinite or NaN entry.
        ValueError: for a `method` or `lapack_driver` not offered.
        TypeError: if a method's name is given as the second argument.

    Examples:
        >>> import numerical
        >>> u, s, vh = numerical.svd([[3.0, 0.0], [0.0, 4.0]])
        >>> s
        numerical.Tensor([4.0, 3.0], dtype=float64)
        >>> u
        numerical.Tensor([[0.0, 1.0],
                          [1.0, 0.0]], dtype=float64)
        >>> numerical.svd([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], full_matrices=False)[2].shape
        (2, 3)
    """
    if isinstance(full_matrices, str):
        raise TypeError("svd: the second argument is full_matrices, as in NumPy; give the method by name, "
                        f"svd(a, method={full_matrices!r})")
    del overwrite_a, check_finite
    if lapack_driver is not None:  # scipy.linalg.svd's
        if lapack_driver not in ("gesdd", "gesvd"):
            raise ValueError(f'lapack_driver must be "gesdd" or "gesvd", not "{lapack_driver}"')
        method = "divide_and_conquer" if lapack_driver == "gesdd" else "bidiagonal_qr"
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


def qr(a, mode="reduced", *, pivoting=False, overwrite_a=False, lwork=None, check_finite=True):
    """The QR factorisation ``a = Q @ R``, as ``numpy.linalg.qr``.

    `R`'s diagonal is real and non-negative. With ``pivoting=True``, as ``scipy.linalg.qr(a, pivoting=True)``
    (LAPACK's ``geqp3``): ``a[:, P] = Q @ R`` with the column of largest remaining norm taken at each step, so that
    `R`'s diagonal does not grow down it and reveals the numerical rank; `R`'s diagonal is real and non-negative there
    too, where LAPACK's may be negative.

    Args:
        a: The m×n matrix, or a stack of them (``(..., m, n)``): a Tensor or an array-like.
        mode: For k = min(m, n): ``"reduced"`` (the default) `Q` m×k with orthonormal columns and `R` k×n upper
            trapezoidal; ``"complete"``, `Q` m×m and `R` m×n; ``"r"``, `R` alone, k×n. SciPy's names ``"economic"``
            and ``"full"`` are taken as ``"reduced"`` and ``"complete"`` (NumPy's default stays, the reduced one).
        pivoting: Pivot the columns, as SciPy's.
        overwrite_a: Accepted for SciPy's signature; ignored.
        lwork: Accepted for SciPy's signature; ignored.
        check_finite: Accepted for SciPy's signature; ignored (an infinite or NaN entry always raises).

    Returns:
        tuple | Tensor: ``(Q, R)``, or `R` alone for ``"r"``. With `pivoting`, ``(Q, R, P)``, or ``(R, P)`` for
        ``"r"``, `P` a tuple of the columns' origins (SciPy's an integer array).

    Raises:
        LinAlgError: if `a` is not a matrix (or a stack of them) or holds an infinite or NaN entry.
        ValueError: for a `mode` not offered (LAPACK's reflectors, ``"raw"``, are not).

    Examples:
        >>> import numerical
        >>> q, r = numerical.qr([[1.0, 0.0], [0.0, 2.0]])
        >>> q
        numerical.Tensor([[1.0, 0.0],
                          [0.0, 1.0]], dtype=float64)
        >>> numerical.qr([[1.0, 0.0], [0.0, -2.0]], mode="r")
        numerical.Tensor([[1.0, 0.0],
                          [0.0, 2.0]], dtype=float64)
    """
    del overwrite_a, lwork, check_finite
    mode = {"economic": "reduced", "full": "complete"}.get(mode, mode)
    if mode not in _QR_MODES:
        raise ValueError(f"qr: mode {mode!r} is none of {_QR_MODES} (the LAPACK reflectors of \"raw\" are not offered)")
    if pivoting:
        q, r, p = _qr_pivoted(a, mode == "complete")
        return (r, tuple(p)) if mode == "r" else (q, r, tuple(p))
    if mode == "r":
        return _qr_r(a)
    return _qr(a, mode)


def _tensor(x):
    """`x` as a Tensor for a function that moves elements: :func:`asarray`'s, a bool array-like kept bool, as NumPy's."""
    return x if isinstance(x, Tensor) else asarray(x)


def _operand(x):
    """`x` as a Tensor for a function that computes on it: :func:`_read`'s."""
    return x if isinstance(x, Tensor) else _read(x)


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
    """The `ord` norm of `x` over `axis`, as ``numpy.linalg.vector_norm``.

    Summed at the scale of the largest magnitude, it does not overflow where NumPy's 2-norm of entries past 1e154 is
    infinite.

    Args:
        x: The Tensor or array-like.
        axis: ``None`` (every element), an integer or a tuple of them; negative ones count from the end.
        keepdims: Keep each reduced axis with extent 1.
        ord: The order, a real number: ``(Σ abs(x)^ord)^(1/ord)``; ``inf`` the largest magnitude, ``-inf`` the
            smallest, 0 how many are not zero.

    Returns:
        float | Tensor: The norm, real, of `x`'s precision: a number for a whole reduction, else a Tensor.

    Raises:
        ValueError: for an `ord` that is not a real number, an `axis` out of range or repeated, or ``ord=-inf`` over no
            element.

    Examples:
        >>> import numerical
        >>> numerical.vector_norm([3.0, 4.0])
        5.0
        >>> numerical.vector_norm([3.0, -4.0], ord=1)
        7.0
        >>> numerical.vector_norm([[3.0, 4.0], [0.0, 0.0]], axis=1)
        numerical.Tensor([5.0, 0.0], dtype=float64)
    """
    if axis is None and ord == 2 and not keepdims:
        plain = _numerical._borrowed_norm(x)  # read where it lies: a NumPy array is not copied
        if plain is not NotImplemented:
            return plain
    x = _operand(x)
    order, axes = _vector_order(ord), list(_resolved_axes(x, axis, "vector_norm"))
    if order == -_math.inf and _builtins.any(x.shape[a] == 0 for a in axes):  # the smallest of nothing, as NumPy's minimum
        raise ValueError("vector_norm: ord=-inf over no element has no value (a minimum of nothing)")
    return _native(_numerical._vector_norm, x, order, axes, bool(keepdims))


def svdvals(x, /):
    """The singular values of a matrix or of every matrix of a stack, descending, as ``numpy.linalg.svdvals``.

    Args:
        x: The matrix, or a stack of them (``(..., m, n)``): a Tensor or an array-like.

    Returns:
        Tensor: The singular values, real, of `x`'s precision; for a stack, one row per matrix.

    Raises:
        LinAlgError: if `x` is not a matrix (or a stack of them) or holds an infinite or NaN entry.

    Examples:
        >>> import numerical
        >>> numerical.svdvals([[3.0, 0.0], [0.0, -4.0]])
        numerical.Tensor([4.0, 3.0], dtype=float64)
    """
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
    x = _operand(x)
    if len(x.shape) < 2:
        raise LinAlgError("matrix_norm: a matrix or a stack of matrices (rank ≥ 2)")
    rank = len(x.shape)
    return _matrix_norm(x, ord, rank - 2, rank - 1, keepdims)


def norm(x, ord=None, axis=None, keepdims=False):
    """A vector or a matrix norm, as ``numpy.linalg.norm``.

    With no `axis`, the 2-norm of every element for no `ord`, else the `ord` norm of the vector or the matrix `x` is;
    with an integer `axis`, the vector norms along it; with a pair, the matrix norms over it (:func:`vector_norm`,
    :func:`matrix_norm`).

    Args:
        x: The Tensor or array-like.
        ord: The order: :func:`vector_norm`'s for a vector, :func:`matrix_norm`'s for a matrix (``None`` the 2-norm,
            the Frobenius norm for a matrix).
        axis: ``None``, an integer (vector norms) or a pair of integers (matrix norms); negative ones count from the
            end.
        keepdims: Keep each reduced axis with extent 1.

    Returns:
        float | Tensor: The norm, real, of `x`'s precision: a number for a whole reduction, else a Tensor.

    Raises:
        ValueError: for an `ord` with no `axis` on a rank other than 1 or 2, an `ord` not offered, an `axis` out of
            range, repeated, or of more than two axes.
        LinAlgError: if a singular value norm (2, -2, ``"nuc"``) meets an infinite or NaN entry.

    Examples:
        >>> import numerical
        >>> numerical.norm([3.0, 4.0])
        5.0
        >>> numerical.norm([[1.0, 2.0], [3.0, 4.0]], 1)
        6.0
        >>> numerical.norm([[3.0, 4.0], [0.0, 0.0]], axis=1)
        numerical.Tensor([5.0, 0.0], dtype=float64)
    """
    if axis is None and ord is None and not keepdims:
        plain = _numerical._borrowed_norm(x)  # every element's 2-norm, a NumPy array read where it lies
        if plain is not NotImplemented:
            return plain
    x = _operand(x)
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
    """The condition number of a matrix or of every matrix of a stack in the `p` norm, as ``numpy.linalg.cond``.

    Args:
        x: The matrix, or a stack of them (``(..., m, n)``): a Tensor or an array-like.
        p: The norm: ``None``, 2 and -2 the ratio of the extreme singular values (infinite for a singular matrix), else
            ``norm(x, p) * norm(inv(x), p)`` (infinite where `x` is singular), `p` one of :func:`matrix_norm`'s orders
            for a square matrix.

    Returns:
        float | Tensor: The condition number; for a stack, a Tensor of its leading shape.

    Raises:
        LinAlgError: if `x` is not a matrix or a stack of them, is empty, is not square for a `p` other than
            ``None``, 2 and -2, or holds an infinite or NaN entry for those three.
        ValueError: for a `p` not offered.

    Examples:
        >>> import numerical
        >>> numerical.cond([[2.0, 0.0], [0.0, 4.0]])
        2.0
        >>> numerical.cond([[1.0, 0.0], [0.0, 0.0]])
        inf
        >>> numerical.cond([[2.0, 0.0], [0.0, 4.0]], 1)
        2.0
    """
    x = _operand(x)
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
    return reshape(_read([cond(stack[i], p) for i in range(stack.shape[0])]), tuple(x.shape[:-2]))


def _paired(x, y, f):
    """`f` of the numbers of two nested lists of one shape, nested alike."""
    return [_paired(u, v, f) for u, v in zip(x, y)] if isinstance(x, list) else f(x, y)


def matrix_power(a, n):
    """`a` to the integer power `n` by repeated squaring, as ``numpy.linalg.matrix_power``.

    Args:
        a: The square matrix, or a stack of them (``(..., n, n)``): a Tensor or an array-like.
        n: The power, an integer: the identity for 0, the inverse's powers for a negative `n`.

    Returns:
        Tensor: ``a`` to the power `n`, of `a`'s shape and dtype (float64 for integers).

    Raises:
        LinAlgError: if `a` is not square (or a stack of square matrices), or singular for a negative `n`.
        TypeError: if `n` is not an integer, or `a` is a bool Tensor.

    Examples:
        >>> import numerical
        >>> numerical.matrix_power([[1.0, 1.0], [0.0, 1.0]], 3)
        numerical.Tensor([[1.0, 3.0],
                          [0.0, 1.0]], dtype=float64)
        >>> numerical.matrix_power([[2.0, 0.0], [0.0, 4.0]], -1)
        numerical.Tensor([[0.5, 0.0],
                          [0.0, 0.25]], dtype=float64)
    """
    a = _operand(a)
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
    operands = [_operand(array) for array in arrays]
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
    """The sum of a diagonal of a matrix or of every matrix of a stack, as ``numpy.linalg.trace``.

    The diagonal is taken over the last two axes. A bool tensor's trace counts its trues, a float64 as :func:`sum`'s
    (NumPy's an integer).

    Args:
        x: The matrix, or a stack of them (``(..., m, n)``): a Tensor or an array-like.
        offset: The diagonal: above the main diagonal for a positive `offset`, below for a negative one.
        dtype: The result's dtype, when given.

    Returns:
        float | complex | Tensor: The trace, a number for a matrix; for a stack, a Tensor of its leading shape.

    Raises:
        ValueError: if `x` is of rank below 2.

    Examples:
        >>> import numerical
        >>> numerical.trace([[1.0, 2.0], [3.0, 4.0]])
        5.0
        >>> numerical.trace([[1.0, 2.0], [3.0, 4.0]], offset=1)
        2.0
        >>> numerical.trace([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]])
        numerical.Tensor([2.0, 4.0], dtype=float64)
    """
    x = _tensor(x)
    if len(x.shape) < 2:
        raise _ShapeError("trace: a matrix or a stack of matrices (rank ≥ 2)")
    if x.dtype == "bool":
        x = asarray(x, "float64")
    along = diagonal(x, offset=offset)
    labels = _labels(len(along.shape), "trace")  # the batch's, then the diagonal's
    if along.shape[-1] == 0:
        result = zeros(tuple(along.shape[:-1]), x.dtype)
    elif x.dtype == "int64":  # an integer sum, wrapped as NumPy's
        result = sum(along, axis=-1)
    else:
        result = _numerical.einsum(f"{labels}->{labels[:-1]}", along)
    if dtype is not None:
        result = asarray(result, dtype)
    return _number_if_scalar(result)


def diagonal(x, /, *, offset=0):
    """The `offset` diagonal of a matrix or of every matrix of a stack (its last two axes), as
    ``numpy.linalg.diagonal``: above the main diagonal for a positive `offset`, below for a negative one; read where
    it lies, in time of its length."""
    x = _tensor(x)
    if len(x.shape) < 2:
        raise _ShapeError("diagonal: a matrix or a stack of matrices (rank ≥ 2)")
    rows, cols = x.shape[-2:]
    offset = _builtins.max(_builtins.min(_operator.index(offset), cols), -rows)  # past the matrix: empty alike
    return _numerical._diagonal(x, offset)


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
    a = _operand(a)
    ind = _operator.index(ind)
    if ind <= 0:
        raise ValueError("tensorinv: ind must be a positive integer")
    left, right = a.shape[:ind], a.shape[ind:]
    rows, cols = _math.prod(left), _math.prod(right)
    if rows != cols:
        raise LinAlgError(f"tensorinv: the first {ind} axes hold {rows} elements and the others {cols}: no square "
                          "matrix to invert")
    return reshape(inv(reshape(a, (rows, cols))), tuple(right) + tuple(left))


def outer(a, b, out=None):
    """The outer product of `a` and `b` flattened, ``a[i] * b[j]``, as ``numpy.outer``.

    Each element is its own product, as NumPy's ``multiply.outer`` (a −0 kept).

    Args:
        a: The first operand, flattened: a Tensor or an array-like.
        b: The second operand, flattened: a Tensor or an array-like.
        out: NumPy's; None alone (a Tensor is never written into).

    Returns:
        Tensor: The ``a.size × b.size`` matrix, in the dtype the two promote to.

    Raises:
        TypeError: if an operand is a bool Tensor or nothing :func:`asarray` reads, or for an `out`.

    Examples:
        >>> import numerical
        >>> numerical.outer([1, 2], [3, 4])
        numerical.Tensor([[3.0, 4.0],
                          [6.0, 8.0]], dtype=float64)
    """
    _no_out(out, "outer")
    a, b = _same_dtype(a, b)
    a, b = reshape(a, (-1,)), reshape(b, (-1,))
    # each element its own product, broadcast, as NumPy's multiply.outer: an einsum adds it to a zero, which loses −0
    return reshape(a, (a.shape[0], 1)) * reshape(b, (1, b.shape[0]))


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """The cross product of the 3-vectors of `a` along `axisa` and of `b` along `axisb`, broadcast over the other
    axes, as ``numpy.cross`` (NumPy 2.5's: 2-vectors refused, as they are since); the products' vectors along
    `axisc`, `axis` for all three. Each component by NumPy's operations.

    Examples:
        >>> import numerical
        >>> numerical.cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        numerical.Tensor([0.0, 0.0, 1.0], dtype=float64)
    """
    if axis is not None:
        axisa = axisb = axisc = axis
    a, b = _same_dtype(a, b)
    if not a.shape or not b.shape:
        raise ValueError("At least one array has zero dimension")
    vectors = []
    for operand, along in ((a, axisa), (b, axisb)):
        rank, at = len(operand.shape), _resolved_axes(operand, _operator.index(along), "cross")[0]
        # as rows, so a part is a tensor of the dtype, not a Python number
        moved = _permuted(operand, [k for k in range(rank) if k != at] + [at])
        vectors.append(reshape(moved, (1,) + moved.shape))
    a, b = vectors
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("Both input arrays must be (arrays of) 3-dimensional vectors, but they are "
                         f"{a.shape[-1]} and {b.shape[-1]} dimensional instead.")

    def part(t, i):
        return t[(slice(None),) * (len(t.shape) - 1) + (i,)]

    a0, a1, a2, b0, b1, b2 = part(a, 0), part(a, 1), part(a, 2), part(b, 0), part(b, 1), part(b, 2)
    first, second, third = a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0
    # side by side, copied: products with unit vectors multiplied an infinite component by their zeros (NaN elsewhere)
    result = _numerical._stack_last([_operand(c) for c in (first, second, third)])
    result = reshape(result, result.shape[1:])
    rank = len(result.shape)
    at = _resolved_axes(result, _operator.index(axisc), "cross")[0]
    if at != rank - 1:  # the vectors' axis where `axisc` puts it
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
    """A :class:`Tensor` from `obj`, as ``numpy.asarray``.

    A complex input is never cast to a real dtype (that would discard the imaginary part).

    Args:
        obj: A Tensor (the same object when `dtype` matches), an object with an ``__array_interface__`` (NumPy: bool,
            integer, float32/64 and complex64/128 elements, any strides), a nested list or tuple of numbers, or a
            number (rank 0).
        dtype: ``"float64"``, ``"complex128"``, ``"float32"``, ``"complex64"`` or ``"bool"`` (true where nonzero), or
            the NumPy type or dtype of that name, or Python's ``float``, ``complex`` or ``bool``; by default an array's own for float32, complex64, complex128 and bool, else float64 (integers included);
            complex128 for a complex number or a list with one, bool for a bool or a list of bools alone, as NumPy's.

    Returns:
        Tensor: The elements of `obj`, a copy unless `obj` is a Tensor of `dtype` already.

    Raises:
        ValueError: if a nested list is not rectangular.
        TypeError: if `obj` is none of these, or a nested list holds something other than numbers.
        TypeError: for a `dtype` not offered (also a :class:`numerical.error` and a ``ValueError``).
        numerical.error: for a complex input cast to a real dtype.

    Examples:
        >>> import numerical
        >>> numerical.asarray([[1, 2], [3, 4]])
        numerical.Tensor([[1.0, 2.0],
                          [3.0, 4.0]], dtype=float64)
        >>> numerical.asarray([1, 0], dtype="bool")
        numerical.Tensor([True, False], dtype=bool)
        >>> numerical.asarray(2.5).shape
        ()
    """
    if dtype is not None and (type(dtype) is not str or dtype not in _DTYPES):
        dtype = _checked_dtype(dtype, "asarray: ")
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


def _read(obj, dtype=None):
    """:func:`asarray` for an operand computed on: without a `dtype`, a bool array-like (not a bool Tensor) read as
    float64, as NumPy's arithmetic and linear algebra compute on one (a bool Tensor has no arithmetic)."""
    tensor = asarray(obj, dtype)
    if dtype is None and tensor.dtype == "bool" and not isinstance(obj, Tensor):
        return asarray(tensor, "float64")
    return tensor


class _UnreadableError(error, TypeError):
    """An object :func:`asarray` cannot read: a ``TypeError`` as NumPy raises, and a :class:`numerical.error`."""



class _DtypeError(error, TypeError, ValueError):
    """A dtype a Tensor does not offer: a ``TypeError`` as NumPy raises for a dtype it does not understand, a
    :class:`numerical.error` as :func:`asarray` raised before, and a ``ValueError`` as the creation functions did."""


def _checked_dtype(dtype, prefix=""):
    """`dtype`'s name (see :func:`_dtype_name`), one of :data:`_DTYPES`, or :class:`_DtypeError`."""
    name = _dtype_name(dtype)
    if name not in _DTYPES:
        raise _DtypeError(f"{prefix}dtype must be one of {_DTYPES}, not {dtype!r}")
    return name


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
    tensor_a = a if isinstance(a, Tensor) else _read(a)
    tensor_b = b if isinstance(b, Tensor) else _read(b)
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
        reductions = np.multiply.reduce(np.array([x, y])).real != 0  # prod's and cumprod's loop: out its own operand
        # its imaginary part ar·bi + ai·br is e² by MSVC's contraction (ai·br fused), 0 by the others' (ar·bi fused)
        late = np.multiply.reduce(np.array([complex(1, 1 + e), complex(-(1 - e), 1)])).imag != 0
        # (inf + 0j) ** -1.0: 0 by the reciprocal a shortcut takes, NaN by the power (its 1 · inf is 0 · inf in part)
        _numpy_shortcuts_any_number = not np.isnan((np.array([complex(np.inf, 0.0)]) ** -1.0)[0].real)
    _numerical._set_numpy_loops(bool(products), bool(scalar_products), bool(quotients), bool(reductions),
                                bool(reductions and late))


def _arithmetic(op, reflected=False):
    """The Tensor method for `op` ("+", "-", "*", "/", "^" for ``**``, "|" for ``//``, "%"), element-wise and broadcast as NumPy's; a reflected
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
        elif (kind is int or kind is float) and (dtype := self.dtype) != "bool" and (kind is int or dtype != "int64"):
            a, b = self, asarray(other, dtype)  # a real Python number is weak: it takes the tensor's dtype
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
Tensor.__floordiv__ = _arithmetic("|")
Tensor.__rfloordiv__ = _arithmetic("|", reflected=True)
Tensor.__mod__ = _arithmetic("%")
Tensor.__rmod__ = _arithmetic("%", reflected=True)
Tensor.__divmod__ = lambda self, other: (self // other, self % other)
Tensor.__rdivmod__ = lambda self, other: (other // self, other % self)
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
        a = a if isinstance(a, Tensor) else _read(a)
        rank = len(a.shape)
        if isinstance(axis, tuple):  # several axes: moved last and run together, one native reduction over them
            axes = _resolved_axes(a, axis, op)
            if not axes:  # each element its own reduction, as NumPy's: a sum from +0, a truth a bool
                if op in ("any", "all"):
                    return asarray(a, "bool")
                if op == "mean" and a.dtype == "int64":
                    return asarray(a, "float64")
                return a + 0 if op in ("sum", "mean") and a.dtype not in ("int64", "bool") else a
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
            if op == "mean" and a.dtype == "int64":  # an integer's mean in float64, as NumPy's
                return full(shape, result, "float64")
            return full(shape, result, None if isinstance(result, bool) or a.dtype == "bool" else a.dtype)
        return reshape(result, shape)

    reduce.__name__ = op
    reduce.__doc__ = doc
    return reduce


sum = _reduction("sum", """The sum of every element, or the sums along `axis`, as ``numpy.sum``.

    Added by pairwise halving, as NumPy adds.

    Args:
        a: The Tensor or array-like (a bool array-like read as float64).
        axis: ``None`` (every element), an integer, or a tuple of them (``()`` for none); negative ones count from the
            end.
        keepdims: Keep each reduced axis with extent 1, as NumPy's.

    Returns:
        float | complex | Tensor: A number for every element, else a Tensor of `a`'s dtype (float64 for a bool one).

    Raises:
        ValueError: for an axis out of range or repeated.

    Examples:
        >>> import numerical
        >>> numerical.sum([[1.0, 2.0], [3.0, 4.0]])
        10.0
        >>> numerical.sum([[1.0, 2.0], [3.0, 4.0]], axis=0)
        numerical.Tensor([4.0, 6.0], dtype=float64)
        >>> numerical.sum([[1.0, 2.0], [3.0, 4.0]], axis=1, keepdims=True)
        numerical.Tensor([[3.0],
                          [7.0]], dtype=float64)
    """)
mean = _reduction("mean", """The mean of every element, or the means along `axis`, as ``numpy.mean``.

    Args:
        a: The Tensor or array-like (a bool array-like read as float64).
        axis: ``None`` (every element), an integer, or a tuple of them (``()`` for none); negative ones count from the
            end.
        keepdims: Keep each reduced axis with extent 1, as NumPy's.

    Returns:
        float | complex | Tensor: A number for every element, else a Tensor of `a`'s dtype (float64 for a bool one).

    Raises:
        ValueError: for an axis out of range or repeated.

    Examples:
        >>> import numerical
        >>> numerical.mean([[1.0, 2.0], [3.0, 4.0]])
        2.5
        >>> numerical.mean([[1.0, 2.0], [3.0, 4.0]], axis=1)
        numerical.Tensor([1.5, 3.5], dtype=float64)
    """)
max = _reduction("max", """The largest element, or the largest along `axis`, as ``numpy.max``.

    A NaN wins. Real dtypes only (a complex one has no order).

    Args:
        a: The Tensor or array-like (a bool array-like read as float64).
        axis: ``None`` (every element), an integer, or a tuple of them (``()`` for none); negative ones count from the
            end.
        keepdims: Keep each reduced axis with extent 1, as NumPy's.

    Returns:
        float | bool | Tensor: A number for every element, else a Tensor of `a`'s dtype.

    Raises:
        ValueError: for an axis out of range or repeated, or a reduction over no element.
        TypeError: for a complex `a`.

    Examples:
        >>> import numerical
        >>> numerical.max([[1.0, 2.0], [3.0, 4.0]])
        4.0
        >>> numerical.max([[1.0, 2.0], [3.0, 4.0]], axis=0)
        numerical.Tensor([3.0, 4.0], dtype=float64)
        >>> numerical.max([1.0, float("nan")])
        nan
    """)
min = _reduction("min", """The smallest element, or the smallest along `axis`, as ``numpy.min``.

    A NaN wins. Real dtypes only.

    Args:
        a: The Tensor or array-like (a bool array-like read as float64).
        axis: ``None`` (every element), an integer, or a tuple of them (``()`` for none); negative ones count from the
            end.
        keepdims: Keep each reduced axis with extent 1, as NumPy's.

    Returns:
        float | bool | Tensor: A number for every element, else a Tensor of `a`'s dtype.

    Raises:
        ValueError: for an axis out of range or repeated, or a reduction over no element.
        TypeError: for a complex `a`.

    Examples:
        >>> import numerical
        >>> numerical.min([[1.0, 2.0], [3.0, 4.0]])
        1.0
        >>> numerical.min([[1.0, 2.0], [3.0, 4.0]], axis=1)
        numerical.Tensor([1.0, 3.0], dtype=float64)
    """)
any = _reduction("any", """Whether any element is true (nonzero), or any along `axis` (a bool Tensor without it), as
``numpy.any``. `axis` may name several axes (a tuple, `()` for none), and `keepdims` keeps each reduced axis with
extent 1, as NumPy's.""")
all = _reduction("all", """Whether every element is true (nonzero), or every one along `axis` (a bool Tensor without
it), as ``numpy.all``. `axis` may name several axes (a tuple, `()` for none), and `keepdims` keeps each reduced axis
with extent 1, as NumPy's.""")
def _dtype_name(dtype):
    """A dtype's name as a Tensor takes it: a string as it is, ``numpy.float64`` or ``numpy.dtype("float32")`` by name,
    Python's ``float``, ``complex`` and ``bool`` as NumPy names them."""
    if isinstance(dtype, str):
        return dtype
    # By identity: a NumPy dtype compares equal to the Python type it stands for, numpy.dtype("float64") == float.
    for python_type, name in ((float, "float64"), (complex, "complex128"), (bool, "bool"), (int, "int64")):
        if dtype is python_type:
            return name
    name = getattr(dtype, "__name__", None) or str(dtype)
    return "bool" if name == "bool_" else name  # NumPy 1's numpy.bool_


def _reduction_method(reduce):
    """The Tensor method of `reduce`, which NumPy's `numpy.sum(t)` and the like call with `out` (and `dtype` for a
    mean): None the only `out` a Tensor takes, since it is never written into; `dtype` converts the result, and an
    int64 tensor's elements first (NumPy's: they accumulate in that dtype, where int64's would wrap)."""
    def method(self, axis=None, dtype=None, out=None, keepdims=False):
        if out is not None:
            raise TypeError(f"{reduce.__name__}: a Tensor takes no `out` (it is never written into)")
        if dtype is not None and self.dtype == "int64" and _dtype_name(dtype) != "int64":
            self = asarray(self, _dtype_name(dtype))
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
    """The elements of `a` in the same order under `shape`, as ``numpy.reshape``.

    Args:
        a: The Tensor or array-like.
        shape: The new extents: an integer or a sequence of them, of `a`'s size; one extent may be -1, the one the
            others leave.

    Returns:
        Tensor: The reshaped elements, of `a`'s dtype.

    Raises:
        ValueError: if `shape` holds another number of elements, or more than one -1.

    Examples:
        >>> import numerical
        >>> numerical.reshape(numerical.arange(6), (2, 3))
        numerical.Tensor([[0.0, 1.0, 2.0],
                          [3.0, 4.0, 5.0]], dtype=float64)
        >>> numerical.reshape(numerical.arange(6), (3, -1)).shape
        (3, 2)
    """
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
    dtype = _checked_dtype(dtype)
    if dtype in ("float64", "float32") and isinstance(value, complex):
        raise TypeError(f"full: a complex value does not fit a {dtype} tensor")
    return dtype


def full(shape, value, dtype=None):
    """A Tensor of `shape` whose every element is `value`, as ``numpy.full``.

    Args:
        shape: The extents: an integer (a vector) or a sequence of non-negative integers.
        value: The number every element holds.
        dtype: ``"float64"``, ``"complex128"``, ``"float32"``, ``"complex64"`` or ``"bool"``; by default complex128
            for a complex value, bool for a bool one and float64 otherwise.

    Returns:
        Tensor: The filled Tensor.

    Raises:
        ValueError: for a negative extent or a `dtype` not offered.
        TypeError: for a complex `value` with a real `dtype`, or a `value` that is not a number.

    Examples:
        >>> import numerical
        >>> numerical.full((2,), 7)
        numerical.Tensor([7.0, 7.0], dtype=float64)
        >>> numerical.full(2, 1j)
        numerical.Tensor([1j, 1j], dtype=complex128)
    """
    return _native(_numerical._full, _shape(shape), value, _dtype_of(dtype, value))


def zeros(shape, dtype="float64"):
    """A Tensor of `shape` full of zeros, as ``numpy.zeros``.

    Args:
        shape: The extents: an integer (a vector) or a sequence of non-negative integers.
        dtype: ``"float64"`` (the default), ``"complex128"``, ``"float32"``, ``"complex64"`` or ``"bool"``; NumPy's
            dtype objects and Python's ``float``, ``complex`` and ``bool`` name them too.

    Returns:
        Tensor: The zeros.

    Raises:
        ValueError: for a negative extent or a `dtype` not offered.
        TypeError: if `shape` is neither an integer nor a sequence of them.

    Examples:
        >>> import numerical
        >>> numerical.zeros(3)
        numerical.Tensor([0.0, 0.0, 0.0], dtype=float64)
        >>> numerical.zeros((2, 2), "complex128")
        numerical.Tensor([[0j, 0j],
                          [0j, 0j]], dtype=complex128)
    """
    return _native(_numerical._full, _shape(shape), 0.0, _dtype_of(dtype))


def ones(shape, dtype="float64"):
    """A Tensor of `shape` full of ones, as ``numpy.ones``.

    Args:
        shape: The extents: an integer (a vector) or a sequence of non-negative integers.
        dtype: ``"float64"`` (the default), ``"complex128"``, ``"float32"``, ``"complex64"`` or ``"bool"``; NumPy's
            dtype objects and Python's ``float``, ``complex`` and ``bool`` name them too.

    Returns:
        Tensor: The ones.

    Raises:
        ValueError: for a negative extent or a `dtype` not offered.
        TypeError: if `shape` is neither an integer nor a sequence of them.

    Examples:
        >>> import numerical
        >>> numerical.ones((2,), dtype="float32")
        numerical.Tensor([1.0, 1.0], dtype=float32)
    """
    return _native(_numerical._full, _shape(shape), 1.0, _dtype_of(dtype))


def eye(n, dtype="float64"):
    """The `n × n` identity, as ``numpy.eye(n)``.

    Args:
        n: The order, a non-negative integer.
        dtype: ``"float64"`` (the default), ``"complex128"``, ``"float32"``, ``"complex64"`` or ``"bool"``; NumPy's
            dtype objects and Python's ``float``, ``complex`` and ``bool`` name them too.

    Returns:
        Tensor: The identity matrix.

    Raises:
        ValueError: for a negative `n` or a `dtype` not offered.
        TypeError: if `n` is not an integer.

    Examples:
        >>> import numerical
        >>> numerical.eye(2)
        numerical.Tensor([[1.0, 0.0],
                          [0.0, 1.0]], dtype=float64)
    """
    return _native(_numerical._eye, _operator.index(n), _dtype_of(dtype))


def arange(start, stop=None, step=1.0):
    """``start, start + step, …`` short of `stop`, as ``numpy.arange``.

    Args:
        start: The first value; with no `stop`, the stop itself, the values starting at 0 (``arange(stop)``).
        stop: The bound, not included.
        step: The spacing, finite and nonzero, negative for a descending range.

    Returns:
        Tensor: The values, float64.

    Raises:
        ValueError: for a bound or a step that is not finite, or a zero step.

    Examples:
        >>> import numerical
        >>> numerical.arange(4)
        numerical.Tensor([0.0, 1.0, 2.0, 3.0], dtype=float64)
        >>> numerical.arange(1, 2, 0.25)
        numerical.Tensor([1.0, 1.25, 1.5, 1.75], dtype=float64)
    """
    if stop is None:
        start, stop = 0.0, start
    return _native(_numerical._arange, float(start), float(stop), float(step), _arange_fuses())


def _result_dtype(*values):
    """NumPy 2's result dtype of `values`: a Tensor's or an array's (a list read by :func:`asarray`) strong, a Python
    number weak (its kind alone: an int beside bools int64, a float beside integers float64, a complex beside reals
    the complex of their precision)."""
    strong = [_dtype_name(v.dtype) if hasattr(v, "dtype") else _tensor(v).dtype for v in values if not _weak(v)]
    weak = [v for v in values if _weak(v)]
    dtype = _promoted(strong) if strong else None
    for kind, name in ((bool, "bool"), (int, "int64"), (float, "float64"), (complex, "complex128")):
        if not _builtins.any(type(v) is kind for v in weak):
            continue
        if dtype is None or name == "bool":
            dtype = _promoted([dtype or name, name])
        elif name == "int64" and dtype == "bool":
            dtype = "int64"
        elif name == "float64" and dtype in ("bool", "int64"):
            dtype = "float64"
        elif name == "complex128" and not dtype.startswith("complex"):
            dtype = "complex64" if dtype == "float32" else "complex128"
    return dtype


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *, device=None):
    """`num` values evenly spaced from `start` to `stop`, as ``numpy.linspace`` and by its steps: ``arange(num) ·
    step + start``, through ``/ div · delta`` where a step is 0, the last value `stop` itself; `start` and `stop`
    tensors for a tensor of ramps along `axis`; computed in the inexact dtype they promote to (a Python number weak),
    floored for an integer `dtype`.

    Args:
        start: The first value: a number or an array-like.
        stop: The last value (past the last, without `endpoint`).
        num: How many values, a non-negative integer (50 by default).
        endpoint: Whether `stop` is the last value; else ``num`` steps of ``(stop − start) / num``.
        retstep: Whether to return the step too.
        dtype: The result's dtype; by default that of the computation.
        axis: Where the values' axis goes among the axes of `start` and `stop`.
        device: The array API's; None or "cpu" alone.

    Returns:
        Tensor: The values (with the step, a number or a tensor, NaN for fewer than two steps, where `retstep`).

    Raises:
        ValueError: for a negative `num`.

    Examples:
        >>> import numerical
        >>> numerical.linspace(0, 1, 5)
        numerical.Tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float64)
        >>> numerical.linspace([0.0, 1.0], 2.0, 3, axis=1)
        numerical.Tensor([[0.0, 1.0, 2.0],
                          [1.0, 1.5, 2.0]], dtype=float64)
    """
    if device not in (None, "cpu"):
        raise ValueError(f"linspace: unsupported device {device!r}")
    num = _operator.index(num)
    if num < 0:
        raise ValueError(f"Number of samples, {num}, must be non-negative.")
    if (type(start) in (int, float) and type(stop) in (int, float) and endpoint and not retstep and axis == 0
            and dtype in (None, float, "float64")):
        return _native(_numerical._linspace, float(start), float(stop), num)
    div = (num - 1) if endpoint else num
    computed = _result_dtype(start, stop)
    computed = "float64" if computed in ("bool", "int64") else computed
    first = start if _weak(start) else asarray(_tensor(start) if not isinstance(start, Tensor) else start, computed)
    last = asarray(stop if not _weak(stop) else [stop], computed)
    last = last if not _weak(stop) else reshape(last, ())
    delta = last - first
    y = reshape(asarray(arange(num), computed), (-1,) + (1,) * len(delta.shape))
    if div > 0:
        step = delta / div
        y = (y / div) * delta if any(step == 0) else y * step
    else:
        step = _math.nan
        y = y * delta
    y = y + first
    if endpoint and num > 1:
        y = concatenate([y[:-1], reshape(asarray(broadcast_to(last, y.shape[1:]), computed), (1,) + y.shape[1:])])
    if axis != 0:
        y = moveaxis(y, 0, axis)
    target = computed if dtype is None else _dtype_name(dtype)
    if target == "int64":
        y = floor(y)
    y = asarray(y, target)
    if retstep:
        return y, step[()] if isinstance(step, Tensor) and not step.shape else step
    return y


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """``base ** linspace(start, stop, num, endpoint, axis=axis)``, as ``numpy.logspace``: a tensor `base` broadcast
    with `start` and `stop`, its axis of the values at `axis`."""
    if not isinstance(base, (float, int)) and len(shape(base)):
        rank = len(broadcast_shapes(shape(start), shape(stop), shape(base)))
        start, stop, base = (reshape(_tensor(x), (1,) * (rank - len(shape(x))) + shape(x)) for x in (start, stop, base))
        base = expand_dims(base, axis)
    y = linspace(start, stop, num=num, endpoint=endpoint, axis=axis)
    result = power(base, y)
    return result if dtype is None else asarray(result, _dtype_name(dtype))


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """`num` values evenly spaced on a log scale from `start` to `stop` (both nonzero, negative ones of one sign), as
    ``numpy.geomspace`` by its steps: :func:`logspace` of their ``log10`` in float64 at least, the ends `start` and
    `stop` themselves. Real dtypes."""
    # NumPy's: `start` and `stop` read as arrays first (a Python number of full strength), beside float(num) and
    # the requested dtype (float64 by default)
    ends = ["int64" if type(x) is int else _result_dtype(x) for x in (start, stop)]
    computed = _promoted(ends + ["float64" if dtype is None else _dtype_name(dtype)])
    computed = "float64" if computed in ("bool", "int64") else computed
    if computed.startswith("complex"):
        raise TypeError("geomspace: real numbers only here (NumPy's takes complex ones too)")
    first, last = (asarray(_tensor(x) if not isinstance(x, Tensor) else x, computed) for x in (start, stop))
    if any(first == 0) or any(last == 0):
        raise ValueError("Geometric sequence cannot include zero")
    sign_ = sign(first)
    first = first / sign_
    last = last / sign_
    result = logspace(log10(first), log10(last), num=num, endpoint=endpoint, base=10.0, dtype=computed)
    shape_ = result.shape[1:]
    if num > 0:
        pieces = [reshape(broadcast_to(first, shape_), (1,) + shape_), result[1:]]
        if num > 1 and endpoint:
            pieces = [pieces[0], result[1:-1], reshape(broadcast_to(last, shape_), (1,) + shape_)]
        result = concatenate(pieces)
    result = result * sign_
    if axis != 0:
        result = moveaxis(result, 0, axis)
    return asarray(result, computed if dtype is None else _dtype_name(dtype))


def _elementwise(name, doc):
    def function(a):
        if name == "abs" and not _numpy_loops_measured:  # a complex magnitude fused as NumPy's vector loop
            _numpy_loops()
        if not isinstance(a, Tensor):
            mapped = _numerical._borrowed_unary(name, a)  # a NumPy array read where it lies
            if mapped is not NotImplemented:
                return mapped
        elif a.dtype == "int64":  # NumPy's: an integer's magnitude an integer (wrapped), the rest in float64
            return where(a < 0, -a, a) if name == "abs" else _numerical._unary(name, asarray(a, "float64"))
        return _numerical._unary(name, a if isinstance(a, Tensor) else _read(a))

    function.__name__ = name
    function.__doc__ = doc
    return function


exp = _elementwise("exp", """e to the power of every element, as ``numpy.exp``.

    Args:
        a: The Tensor or array-like (a bool array-like read as float64; a bool Tensor refused).

    Returns:
        Tensor: Of `a`'s shape and dtype (float64 for integers).

    Raises:
        TypeError: if `a` is a bool Tensor or nothing :func:`asarray` reads.

    Examples:
        >>> import numerical
        >>> numerical.exp([0.0, 0.0])
        numerical.Tensor([1.0, 1.0], dtype=float64)
        >>> numerical.exp([0j])
        numerical.Tensor([(1+0j)], dtype=complex128)
    """)
log = _elementwise("log", """The natural logarithm of every element, as ``numpy.log``.

    NaN below 0 for a real dtype, ``-inf`` at 0; a complex element's principal logarithm.

    Args:
        a: The Tensor or array-like (a bool array-like read as float64; a bool Tensor refused).

    Returns:
        Tensor: Of `a`'s shape and dtype (float64 for integers).

    Raises:
        TypeError: if `a` is a bool Tensor or nothing :func:`asarray` reads.

    Examples:
        >>> import numerical
        >>> numerical.log([1.0, 0.0, -1.0])
        numerical.Tensor([0.0, -inf, nan], dtype=float64)
    """)
sqrt = _elementwise("sqrt", """The square root of every element, as ``numpy.sqrt``.

    NaN below 0 for a real dtype; a complex element's principal root.

    Args:
        a: The Tensor or array-like (a bool array-like read as float64; a bool Tensor refused).

    Returns:
        Tensor: Of `a`'s shape and dtype (float64 for integers).

    Raises:
        TypeError: if `a` is a bool Tensor or nothing :func:`asarray` reads.

    Examples:
        >>> import numerical
        >>> numerical.sqrt([4.0, 9.0, -1.0])
        numerical.Tensor([2.0, 3.0, nan], dtype=float64)
        >>> numerical.sqrt([-1.0 + 0j])
        numerical.Tensor([1j], dtype=complex128)
    """)
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
    """`x`'s element where `condition` holds, else `y`'s, the three broadcast, as ``numpy.where``.

    Args:
        condition: A bool Tensor or what :func:`asarray` reads (a number true where nonzero).
        x: The elements taken where `condition` holds: a Tensor or an array-like.
        y: The elements taken elsewhere: a Tensor or an array-like.

    Returns:
        Tensor: Of the shape the three broadcast to, in the dtype `x` and `y` promote to, as arithmetic promotes them.

    Raises:
        ValueError: if the shapes do not broadcast.
        TypeError: if an operand is nothing :func:`asarray` reads.

    Examples:
        >>> import numerical
        >>> numerical.where([True, False], [1, 2], [3, 4])
        numerical.Tensor([1.0, 4.0], dtype=float64)
        >>> numerical.where([[True], [False]], [1, 2], 0)
        numerical.Tensor([[1.0, 2.0],
                          [0.0, 0.0]], dtype=float64)
    """
    condition = condition if isinstance(condition, Tensor) else _read(condition)
    if condition.dtype != "bool":
        condition = asarray(condition, "bool")
    x, y = _same_dtype(x, y)
    return _native(_numerical._where, condition, x, y)


def _classifier(name, doc):
    def function(x, /):
        return _numerical._classify(name, x if isinstance(x, Tensor) else _read(x))

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
        return _comparison(op)(x1 if isinstance(x1, Tensor) else _read(x1), x2)

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
    x = x if isinstance(x, Tensor) else _read(x)
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
    """`a` with its axes permuted, as ``numpy.transpose``: a copy.

    Args:
        a: The Tensor or array-like (a bool one kept bool).
        axes: A permutation of `a`'s axes, negative ones counted from the end; by default the axes reversed.

    Returns:
        Tensor: The permuted elements, of `a`'s dtype.

    Raises:
        ValueError: if `axes` is not a permutation of `a`'s axes.

    Examples:
        >>> import numerical
        >>> numerical.transpose([[1, 2], [3, 4]])
        numerical.Tensor([[1.0, 3.0],
                          [2.0, 4.0]], dtype=float64)
        >>> numerical.transpose(numerical.zeros((2, 3, 4)), (1, 0, 2)).shape
        (3, 2, 4)
    """
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
    b = _operand(b)
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


_ITEMSIZE = {"float64": 8, "complex128": 16, "float32": 4, "complex64": 8, "bool": 1, "int64": 8}

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
    inaccurate. Where SciPy's linalg is loaded, the warning issued is also a ``scipy.linalg.LinAlgWarning``, so a
    filter set on SciPy's class takes it."""


def _linalg_warning():
    """The category NumericAL warns of an ill-conditioned operation with: :class:`LinAlgWarning`, beside SciPy's
    where ``scipy.linalg`` is loaded."""
    scipy_linalg = _sys.modules.get("scipy.linalg")
    theirs = getattr(scipy_linalg, "LinAlgWarning", None) if scipy_linalg is not None else None
    return _beside(LinAlgWarning, theirs) if isinstance(theirs, type) else LinAlgWarning


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
    return _read([[float(rows(i, j)) for j in range(n)] for i in range(n)]) if n else zeros((0, 0))


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
    a = _operand(a)
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
    return _numerical._bandwidth(_tensor(a))


def _symmetric(a, atol, rtol, hermitian):
    a = _operand(a)
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
    A, B = _operand(A), _operand(B)
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
    return _read(theta)


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
    a = _operand(a)
    s, u = eigh(a, UPLO="L" if lower else "U")
    values = s.tolist()
    largest = _builtins.max((_builtins.abs(v) for v in values), default=0.0)
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
    A = _operand(A)
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


def _sign_newton(A, steps=30):
    """The sign of `A` by Newton's iteration ``S ← (μ S + (μ S)⁻¹) / 2``, each step scaled by ``μ = |det S|^(-1/n)``
    (Higham's determinantal scaling) until the steps shrink below 10⁻², then unscaled: or None where `S` turns
    singular or `steps` do not converge, an eigenvalue then on or near the imaginary axis."""
    n = A.shape[0]
    if n == 0:
        return None
    tolerance = n * (2.0 ** -23 if A.dtype in _SINGLE else 2.0 ** -52)
    S, scaled, last = A, True, False
    for _ in range(steps):
        try:
            inverse_S = inv(S)
            if scaled:
                sign, logabsdet = slogdet(S)
                mu = _math.exp(-float(logabsdet) / n)
                following = 0.5 * (mu * S + inverse_S / mu)
            else:
                following = 0.5 * (S + inverse_S)
        except LinAlgError:
            return None
        size = norm(following, 1)
        if size == 0.0:  # S + S⁻¹ = 0: S² = −I, every eigenvalue on the imaginary axis
            return None
        change = norm(following - S, 1) / size
        S = following
        if last or not _math.isfinite(change):
            break
        scaled = scaled and change > 1e-2
        last = change <= _math.sqrt(tolerance)  # quadratic: one more step brings the change under the tolerance
    else:
        return None
    return S if last and _math.isfinite(change) else None


def signm(A, disp=True):
    """The matrix sign function, as ``scipy.linalg.signm``: Newton's iteration ``S ← (μ S + (μ S)⁻¹) / 2`` with
    determinantal scaling, 12 to 20 steps of one inverse each; where it does not converge in 30 (an eigenvalue on or
    near the imaginary axis), SciPy's algorithm: :func:`funm` of the sign of each eigenvalue's real part (0 within
    ``1e3 eps`` of the largest), then, where its error estimate passes ``1e3 eps``, the unscaled iteration from `A`
    shifted by half the reciprocal of its largest singular value. With ``disp=False`` the error estimate beside the
    answer: ``‖P² − P‖₁`` of the projector ``P = (S² + S) / 2``, as SciPy's, which grows as ``‖S‖²`` with the
    sign's conditioning; only the fallback reports an inaccurate answer on standard output, as SciPy's does."""
    A = _square("signm", A)
    errtol = 1e3 * (2.0 ** -23 if A.dtype in _SINGLE else 2.0 ** -52)
    newton = _sign_newton(A)
    if newton is not None:  # converged: as accurate as the sign's conditioning allows, whatever SciPy's estimate says
        if disp:
            return newton
        projector = 0.5 * (newton @ newton + newton)
        return newton, norm(projector @ projector - projector, 1)

    def rounded_sign(x):  # SciPy's: 0 where |Re λ| is not past c, NumPy's complex maximum ordering (Re, then Im)
        values = [complex(v) for v in _flat(_read(x).tolist())]
        c = 1e3 * 2.0 ** -52 * _builtins.max(values, key=lambda v: (v.real, v.imag))
        return [(1.0 if v.real > 0 else -1.0) if (_builtins.abs(v.real), 0.0) > (c.real, c.imag) else 0.0 for v in values]

    result, errest = funm(A, rounded_sign, disp=False)
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
    return y != 0 and _builtins.abs(x / y) < 1.0


def _ouc(x, y):
    return x != 0 if y == 0 else _builtins.abs(x / y) > 1.0


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
            return _flat(_read(chosen).tolist()) if isinstance(chosen, Tensor) else list(chosen)
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
_band_tridiagonal = _linalg(_numerical.band_tridiagonal)
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
    a = _operand(a)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("expected square matrix")
    if a.shape[0] <= 2:
        if not _builtins.all(_math.isfinite(_builtins.abs(x)) for x in _flat(a.tolist())):
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
    A = _operand(A)
    if len(A.shape) == 1:
        A = reshape(A, (1, -1))
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("The data matrix for balancing should be square.")
    n = A.shape[0]
    B, scaling, permutation = _matrix_balance(A, bool(permute), bool(scale))
    if separate:
        return B, (scaling, list(permutation))
    rows = reshape(arange(n), (n, 1))
    return B, where(rows == _read([float(p) for p in permutation]) if n else rows == 0.0,
                    reshape(scaling, (1, n)), 0.0)


_SELECT = {"a": 0, "all": 0, 0: 0, "v": 1, "value": 1, 1: 1, "i": 2, "index": 2, 2: 2}


def eigh_tridiagonal(d, e, eigvals_only=False, select="a", select_range=None, check_finite=True, tol=0.0,
                     lapack_driver="auto"):
    """The eigenvalues, ascending, and with them the orthonormal eigenvectors (columns) of the real symmetric
    tridiagonal matrix of diagonal `d` and off-diagonal `e`, as ``scipy.linalg.eigh_tridiagonal``: every one by divide
    and conquer (LAPACK's ``stevd``, SciPy's default), or those in the half-open interval ``(vl, vu]`` of
    ``select="v"``, or those of the indices ``il … iu`` of ``select="i"``, by bisection (``stebz``, or the whole
    spectrum's QL iteration where they are many) and their vectors by inverse iteration (``stein``). `tol` and the
    drivers are accepted (the answer is the same whichever is named, within each driver's rules); `check_finite` too:
    an infinite or NaN entry raises always.

    Raises:
        ValueError: if `d` and `e` are not vectors of lengths n and n − 1, `select` or `select_range` is invalid, a
            driver is named outside its rules, or an entry is infinite or NaN.
        TypeError: if `d` or `e` is complex.
    """
    del check_finite, tol
    d, e = _operand(d), _operand(e)
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
        bounds = list(_flat(_read(select_range).tolist())) if select_range is not None else []
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
    if kind:  # bisection for a few of them (dstebz), their vectors by inverse iteration (dstein)
        w = _tridiagonal_select(asarray(d, "float64"), asarray(e, "float64"), kind == 1, float(bounds[0]), float(bounds[1]))
        v = None if eigvals_only else _tridiagonal_vectors(asarray(d, "float64"), asarray(e, "float64"), w)
    else:
        w, v = _tridiagonal_eigen(asarray(d, "float64"), asarray(e, "float64"), not eigvals_only)
    if single:
        w = asarray(w, "float32")
        v = v if eigvals_only else asarray(v, "float32")
    return w if eigvals_only else (w, v)


_SELECTIONS = {0: 0, 1: 1, 2: 2, "all": 0, "value": 1, "index": 2, "a": 0, "v": 1, "i": 2}


def _band_reduction_pays(width, size, eigvals_only):
    """Whether the band's reduction to tridiagonal form (plane rotations, ``n² u`` operations) beats the dense
    matrix's (blocked reflections, ``4/3 n³``): for the eigenvalues alone of a band of at most 16 diagonals each side
    (M1 Pro: order 3000, u = 2, 170 ms against 760; even near u = 20 at order 1000 and 40 at 3000); not for the
    vectors, whose rotations' product (``n³`` on one thread) loses to the dense path's blocked one."""
    return eigvals_only and 1 <= width <= 16 and size >= 4 * width


def _eig_banded_by_reduction(band, width, lower, eigvals_only, select, subset_by_index, subset_by_value, real_dtype):
    """:func:`eig_banded` through the band's real tridiagonal form (LAPACK's ``sbtrd``/``hbtrd``) and
    :func:`eigh_tridiagonal`: the eigenvectors the reduction's basis times the tridiagonal's."""
    size = band.shape[1]
    computed = "complex128" if band.dtype.startswith("complex") else "float64"
    if lower:
        lower_band = asarray(band, computed)
    else:  # the lower form's diagonal d: the conjugates of the upper form's row u − d, shifted d to the left
        rows = []
        for d in range(width + 1):
            row = asarray(band[width - d][d:], computed)
            row = conj(row) if computed == "complex128" else row
            rows.append(concatenate([row, zeros((d,), computed)]))
        lower_band = stack(rows)
    d, e, basis = _band_tridiagonal(lower_band, not eigvals_only)
    kind = {0: "a", 1: "v", 2: "i"}[select]
    bounds = subset_by_value if select == 1 else subset_by_index
    found = eigh_tridiagonal(d, e, eigvals_only=eigvals_only, select=kind, select_range=bounds)
    if eigvals_only:
        return asarray(found, real_dtype)
    values, vectors = found
    vectors = basis @ (asarray(vectors, computed) if computed == "complex128" else vectors)
    vector_dtype = band.dtype if band.dtype.startswith("complex") else real_dtype
    return asarray(values, real_dtype), asarray(vectors, vector_dtype)


def eig_banded(a_band, lower=False, eigvals_only=False, overwrite_a_band=False, select="a", select_range=None,
               max_ev=0, check_finite=True):
    """The eigenvalues (ascending) and eigenvectors of the real symmetric or complex Hermitian band matrix whose
    upper diagonal-ordered form is `a_band`, ``(u + 1, n)``, ``a_band[u + i - j, j] = a[i, j]`` for ``i <= j`` (with
    `lower` its lower form, ``a_band[i - j, j] = a[i, j]`` for ``i >= j``), as ``scipy.linalg.eig_banded``: all of
    them (`select` "a"), those in ``(min, max]`` ("v") or of indices ``min`` to ``max`` ("i") of `select_range`.
    A real tridiagonal band (``u = 1``) through :func:`eigh_tridiagonal`; the eigenvalues alone of a narrow one
    through its own reduction to real tridiagonal form by plane rotations (LAPACK's ``sbtrd`` and ``hbtrd``: ``n² u``
    operations, where the dense matrix takes ``n³``); the rest through :func:`eigh` of the matrix it describes. `max_ev`,
    `overwrite_a_band` and `check_finite` are SciPy's and change nothing here.

    Raises:
        ValueError: if `a_band` is not a matrix, `select` or `select_range` is invalid, or an entry is infinite or NaN.
    """
    del overwrite_a_band, max_ev, check_finite
    band = a_band if isinstance(a_band, Tensor) else _read(a_band)
    if band.ndim != 2:
        raise ValueError("expected a 2-D array")
    if not all(isfinite(band)):
        raise ValueError("array must not contain infs or NaNs")
    if isinstance(select, str):
        select = select.lower()
    if select not in _SELECTIONS:
        raise ValueError("invalid argument for select")
    select = _SELECTIONS[select]
    size = band.shape[1]
    real_dtype = "float32" if band.dtype in ("float32", "complex64") else "float64"
    if band.size == 0:
        return zeros((0,), real_dtype) if eigvals_only else (zeros((0,), real_dtype), zeros((0, 0), band.dtype))
    subset_by_index = subset_by_value = None
    if select:
        bounds = select_range if isinstance(select_range, Tensor) else asarray(
            select_range, "int64" if select == 2 and _builtins.all(
                isinstance(v, _numbers.Integral) for v in _nested(select_range)[0]) else None)
        if bounds.ndim != 1 or bounds.size != 2 or bounds[1] < bounds[0]:
            raise ValueError("select_range must be a 2-element array-like in nondecreasing order")
        if select == 1:
            subset_by_value = [float(bounds[0]), float(bounds[1])]
        else:
            if bounds.dtype != "int64":
                raise ValueError(f'when using select="i", select_range must contain integers, got dtype {bounds.dtype}')
            low, high = int(bounds[0]), int(bounds[1])
            if low < 0 or high >= size:
                raise ValueError("select_range out of bounds")
            subset_by_index = [low, high]
    width = band.shape[0] - 1
    if width == 1 and not band.dtype.startswith("complex"):  # tridiagonal: in n² operations, not n³
        diagonal, off = (band[0], band[1][:size - 1]) if lower else (band[1], band[0][1:])
        kind = {0: "a", 1: "v", 2: "i"}[select]
        bounds = subset_by_value if select == 1 else subset_by_index
        return eigh_tridiagonal(diagonal, off, eigvals_only=eigvals_only, select=kind, select_range=bounds)
    if _band_reduction_pays(width, size, eigvals_only):  # the band's own reduction, in n² u operations
        return _eig_banded_by_reduction(band, width, lower, eigvals_only, select, subset_by_index, subset_by_value,
                                        real_dtype)
    dense = zeros((size, size), band.dtype)
    for k in range(_builtins.min(width, size - 1) + 1):  # the k-th diagonal off the main one, in the triangle read
        row = band[k] if lower else band[width - k]
        dense = dense + diag(row[:size - k] if lower else row[k:], -k if lower else k)
    return eigh(dense, UPLO="L" if lower else "U", eigvals_only=eigvals_only, subset_by_index=subset_by_index,
                subset_by_value=subset_by_value)


def eigvals_banded(a_band, lower=False, overwrite_a_band=False, select="a", select_range=None, check_finite=True):
    """The eigenvalues alone of a symmetric or Hermitian band matrix, as ``scipy.linalg.eigvals_banded``
    (:func:`eig_banded` with ``eigvals_only=True``)."""
    return eig_banded(a_band, lower=lower, eigvals_only=True, overwrite_a_band=overwrite_a_band, select=select,
                      select_range=select_range, check_finite=check_finite)


def matmul_toeplitz(c_or_cr, x, check_finite=False, workers=None):
    """``T @ x`` for the Toeplitz matrix `T` of first column `c` and first row `r` (``r = conj(c)`` by default; `T` of
    ``len(c)`` rows and ``len(r)`` columns), as ``scipy.linalg.matmul_toeplitz`` and by its steps: the product by the
    circulant matrix that embeds `T`, through the FFT, ``O((m + n) log(m + n))`` a column where a product takes
    ``m n``. In float64 or complex128; `x` a vector or a matrix. `check_finite` and `workers` are SciPy's and change
    nothing here.

    Raises:
        ValueError: if `x` has not as many rows as `T` has columns, or more than two axes (SciPy's batches are not
            taken).

    Examples:
        >>> import numerical
        >>> numerical.matmul_toeplitz(([1.0, 2.0, 3.0], [1.0, 4.0]), [1.0, 1.0])
        numerical.Tensor([5.0, 3.0, 5.0], dtype=float64)
    """
    del check_finite, workers
    if isinstance(c_or_cr, tuple):
        c, r = (v if isinstance(v, Tensor) else _read(v) for v in c_or_cr)
    else:
        c = c_or_cr if isinstance(c_or_cr, Tensor) else _read(c_or_cr)
        r = conj(c) if c.dtype.startswith("complex") else c
    x = x if isinstance(x, Tensor) else _read(x)
    if c.ndim > 1 or r.ndim > 1 or x.ndim > 2:
        raise ValueError("matmul_toeplitz takes vectors c and r and a vector or a matrix x: SciPy's batches are not "
                         "taken")
    c, r = reshape(c, (-1,)), reshape(r, (-1,))
    if not x.shape or x.shape[0] != r.shape[0]:
        raise ValueError("Incompatible dimensions.")
    shape = x.shape
    dtype = "complex128" if _builtins.any(t.dtype.startswith("complex") for t in (c, r, x)) else "float64"
    c, r, x = (asarray(t, dtype) for t in (c, r, x))
    x = reshape(x, (-1, 1)) if len(shape) == 1 else x
    rows, size = c.shape[0], r.shape[0] + c.shape[0] - 1
    result_shape = (rows,) if len(shape) == 1 else (rows, x.shape[1])
    if x.size == 0:
        return zeros(result_shape, dtype)
    embedded = concatenate([c, r[r.shape[0] - 1:0:-1]])
    if dtype == "complex128":
        spectrum = reshape(fft.fft(embedded, axis=0), (-1, 1))
        product = fft.ifft(spectrum * fft.fft(x, n=size, axis=0), axis=0)[:rows, :]
    else:
        spectrum = reshape(fft.rfft(embedded, axis=0), (-1, 1))
        product = fft.irfft(spectrum * fft.rfft(x, n=size, axis=0), axis=0, n=size)[:rows, :]
    return reshape(product, result_shape)


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
    a = _operand(a)
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
    w, v = _operand(w), _operand(v)
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
        partner_col = reshape(_read([float(p) for p in partner]), (n, 1))
        role_col = reshape(_read(role), (n, 1))
        real_row = reshape(_read([x.real for x in values]), (1, n))
        imag_col = reshape(_read([x.imag for x in values]), (n, 1))
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


__all__ += ["hessenberg", "matrix_balance", "eigh_tridiagonal", "eigvalsh_tridiagonal", "eig_banded", "eigvals_banded",
            "matmul_toeplitz", "cholesky_banded",
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
    """The arrays one after the other along `axis`, as ``numpy.concatenate``.

    Every other axis is of one extent. The result is in the dtype they all fit, as NumPy promotes them.

    Args:
        arrays: A sequence of Tensors or array-likes, of one rank.
        axis: The axis to join along (negative counted from the end); ``None`` flattens each first.

    Returns:
        Tensor: The joined arrays.

    Raises:
        ValueError: if there is no array, one is rank 0, the axis is out of their rank, or the other axes differ.
        TypeError: if `arrays` is a Tensor, a string or not a sequence.

    Examples:
        >>> import numerical
        >>> numerical.concatenate([[1, 2], [3]])
        numerical.Tensor([1.0, 2.0, 3.0], dtype=float64)
        >>> numerical.concatenate([[[1, 2]], [[3, 4]]], axis=1)
        numerical.Tensor([[1.0, 2.0, 3.0, 4.0]], dtype=float64)
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

    The result is in the dtype they all fit, as NumPy promotes them.

    Args:
        arrays: A sequence of Tensors or array-likes, of one shape.
        axis: Where the new axis stands in the result (negative counted from the end).

    Returns:
        Tensor: The stacked arrays, of rank one more than theirs.

    Raises:
        ValueError: if there is no array, their shapes differ, or the axis is out of the result's rank.
        TypeError: if `arrays` is a Tensor, a string or not a sequence.

    Examples:
        >>> import numerical
        >>> numerical.stack([[1, 2], [3, 4]])
        numerical.Tensor([[1.0, 2.0],
                          [3.0, 4.0]], dtype=float64)
        >>> numerical.stack([[1, 2], [3, 4]], axis=1)
        numerical.Tensor([[1.0, 3.0],
                          [2.0, 4.0]], dtype=float64)
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
    return where(_band_mask(shape, k, lower), m, zeros((), m.dtype))


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
    return where(j - i == k, reshape(column, (n, 1)), zeros((), v.dtype))  # zeros of v's dtype: an int64 stays one


def kron(a, b):
    """The Kronecker product, as ``numpy.kron``: blocks ``a[i, j] * b`` (over every axis, the shorter shape led by
    ones)."""
    a, b = _operand(a), _operand(b)
    rank = _builtins.max(len(a.shape), len(b.shape))
    sa = (1,) * (rank - len(a.shape)) + a.shape
    sb = (1,) * (rank - len(b.shape)) + b.shape
    a = reshape(a, tuple(x for pair in zip(sa, (1,) * rank) for x in pair))
    b = reshape(b, tuple(x for pair in zip((1,) * rank, sb) for x in pair))
    return reshape(a * b, tuple(x * y for x, y in zip(sa, sb)))


__all__ += ["concatenate", "concat", "stack", "vstack", "hstack", "tril", "triu", "diag", "kron"]


# --- NumPy's root functions composed of the natives above: shapes, element-wise comparisons, products, spreads ---


abs = absolute  # numpy.abs, numpy.absolute's other name


def negative(x, /):
    """``-x`` element-wise, as ``numpy.negative`` (a bool tensor refused, as NumPy's)."""
    x = _tensor(x)
    if x.dtype == "bool":
        raise TypeError("The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the "
                        "logical_not function instead.")
    return -x


def positive(x, /):
    """`x` itself, as ``numpy.positive`` (a Tensor is never written, so it needs no copy)."""
    x = _tensor(x)
    if x.dtype == "bool":
        raise TypeError("The numpy boolean positive is not supported")
    return x


def square(x, /):
    """``x * x`` element-wise, as ``numpy.square``."""
    x = _operand(x)
    return x * x


def divide(x1, x2, /):
    """``x1 / x2`` element-wise, as ``numpy.divide`` (``numpy.true_divide``)."""
    if not isinstance(x1, (Tensor, _numbers.Number)):
        x1 = _operand(x1)
    if not isinstance(x2, (Tensor, _numbers.Number)):
        x2 = _operand(x2)
    if not isinstance(x1, Tensor) and not isinstance(x2, Tensor):
        x1 = _operand(x1)
    return x1 / x2


true_divide = divide


def _binary_function(operator):
    """``operator(x1, x2)`` of NumPy's binary function: through the Tensor operator, an operand that is neither a
    Tensor nor a number read first (one of the two at least a Tensor)."""
    def function(x1, x2, /):
        if not isinstance(x1, (Tensor, _numbers.Number)):
            x1 = _operand(x1)
        if not isinstance(x2, (Tensor, _numbers.Number)):
            x2 = _operand(x2)
        if not isinstance(x1, Tensor) and not isinstance(x2, Tensor):
            x1 = _operand(x1)
        return operator(x1, x2)

    return function


def floor_divide(x1, x2, /):
    """``x1 // x2`` element-wise, the quotient floored, as ``numpy.floor_divide``: NumPy's ``divmod`` for reals (by
    zero ``x1 / x2``: ±inf or NaN). Real dtypes, as NumPy's.

    Examples:
        >>> import numerical
        >>> numerical.floor_divide(numerical.asarray([7.0, -7.0]), 2)
        numerical.Tensor([3.0, -4.0], dtype=float64)
    """
    return _floor_divide(x1, x2)


def remainder(x1, x2, /):
    """``x1 % x2`` element-wise, of `x2`'s sign as Python's, as ``numpy.remainder`` (``numpy.mod``): NumPy's ``divmod``
    for reals (by zero NaN). Real dtypes, as NumPy's.

    Examples:
        >>> import numerical
        >>> numerical.remainder(numerical.asarray([7.0, -7.0]), 2)
        numerical.Tensor([1.0, 1.0], dtype=float64)
    """
    return _remainder(x1, x2)


def divmod(x1, x2, /):  # noqa: A001 (numpy.divmod's name)
    """``(x1 // x2, x1 % x2)`` element-wise, as ``numpy.divmod``."""
    return floor_divide(x1, x2), remainder(x1, x2)


_floor_divide = _binary_function(_operator.floordiv)
_remainder = _binary_function(_operator.mod)
mod = remainder


def _like(a, shape, dtype):
    """The shape and dtype of a ``*_like`` function: `a`'s unless given."""
    a = a if isinstance(a, Tensor) else _tensor(a)
    return (a.shape if shape is None else _shape(shape)), (a.dtype if dtype is None else dtype)


def zeros_like(a, dtype=None, shape=None):
    """Zeros of `a`'s shape and dtype (or those given), as ``numpy.zeros_like``."""
    shape, dtype = _like(a, shape, dtype)
    return zeros(shape, dtype)


def ones_like(a, dtype=None, shape=None):
    """Ones of `a`'s shape and dtype (or those given), as ``numpy.ones_like``."""
    shape, dtype = _like(a, shape, dtype)
    return ones(shape, dtype)


def empty_like(a, dtype=None, shape=None):
    """A tensor of `a`'s shape and dtype (or those given), as ``numpy.empty_like``; its elements zeros here, where
    NumPy's are whatever the memory held."""
    shape, dtype = _like(a, shape, dtype)
    return zeros(shape, dtype)


def full_like(a, fill_value, dtype=None, shape=None):
    """`fill_value` in `a`'s shape and dtype (or those given), as ``numpy.full_like``."""
    shape, dtype = _like(a, shape, dtype)
    return full(shape, fill_value, dtype)


def copy(a):
    """`a` as a Tensor, as ``numpy.copy``: an array-like read into one; a Tensor itself, never written, so it serves
    as its own copy."""
    return a if isinstance(a, Tensor) else _tensor(a)


def ravel(a):
    """The elements of `a` in row-major order as a vector, as ``numpy.ravel``."""
    return reshape(_tensor(a), (-1,))


def _normalized_axes(axes, rank, name):
    """`axes` (an integer or a sequence) as distinct non-negative axes of a rank-`rank` result."""
    axes = (axes,) if isinstance(axes, _numbers.Integral) else tuple(axes)
    resolved = []
    for axis in axes:
        axis = _operator.index(axis)
        if not -rank <= axis < rank:
            raise _ShapeError(f"{name}: axis {axis} is out of bounds for array of dimension {rank}")
        resolved.append(axis % rank)
    if len(set(resolved)) != len(resolved):
        raise ValueError("repeated axis")
    return resolved


def expand_dims(a, axis):
    """`a` with an axis of extent 1 inserted at each of `axis` (an integer or a tuple, in the result's numbering),
    as ``numpy.expand_dims``."""
    a = _tensor(a)
    rank = len(a.shape) + (1 if isinstance(axis, _numbers.Integral) else len(tuple(axis)))
    inserted = set(_normalized_axes(axis, rank, "expand_dims"))
    extents = iter(a.shape)
    return reshape(a, tuple(1 if k in inserted else next(extents) for k in range(rank)))


def _at_least(rank, arrays, shape_of):
    tensors = [_tensor(a) for a in arrays]
    shaped = [t if len(t.shape) >= rank else reshape(t, shape_of(t.shape)) for t in tensors]
    return shaped[0] if len(shaped) == 1 else shaped


def atleast_1d(*arys):
    """Each array with at least one axis (a number a vector of one), as ``numpy.atleast_1d``: one tensor for one
    argument, else a list."""
    return _at_least(1, arys, lambda shape: (1,))


def atleast_2d(*arys):
    """Each array with at least two axes (a vector a row), as ``numpy.atleast_2d``."""
    return _at_least(2, arys, lambda shape: (1,) * (2 - len(shape)) + shape)


def atleast_3d(*arys):
    """Each array with at least three axes, as ``numpy.atleast_3d``: a vector of n ``(1, n, 1)``, a matrix
    ``(m, n, 1)``."""
    return _at_least(3, arys, lambda shape: {0: (1, 1, 1), 1: (1,) + shape + (1,), 2: shape + (1,)}[len(shape)])


def flip(m, axis=None):
    """`m` with its elements in reverse order along `axis` (an integer, a tuple, or None for every axis), as
    ``numpy.flip``."""
    m = _tensor(m)
    rank = len(m.shape)
    axes = range(rank) if axis is None else _normalized_axes(axis, rank, "flip")
    if not rank:
        return m
    return m[tuple(slice(None, None, -1) if k in axes else slice(None) for k in range(rank))]


def flipud(m):
    """`m` upside down (its first axis reversed), as ``numpy.flipud``."""
    m = _tensor(m)
    if len(m.shape) < 1:
        raise ValueError("Input must be >= 1-d.")
    return flip(m, 0)


def fliplr(m):
    """`m` left to right (its second axis reversed), as ``numpy.fliplr``."""
    m = _tensor(m)
    if len(m.shape) < 2:
        raise ValueError("Input must be >= 2-d.")
    return flip(m, 1)


def swapaxes(a, axis1, axis2):
    """`a` with the axes `axis1` and `axis2` exchanged, as ``numpy.swapaxes``."""
    a = _tensor(a)
    rank = len(a.shape)
    i, j = (_normalized_axes(axis, rank, "swapaxes")[0] for axis in (axis1, axis2))
    order = list(range(rank))
    order[i], order[j] = order[j], order[i]
    return a if i == j else transpose(a, order)


def moveaxis(a, source, destination):
    """`a` with the axes `source` moved to the places `destination` (integers or sequences of one length), the others
    in their order, as ``numpy.moveaxis``."""
    a = _tensor(a)
    rank = len(a.shape)
    source = _normalized_axes(source, rank, "moveaxis")
    destination = _normalized_axes(destination, rank, "moveaxis")
    if len(source) != len(destination):
        raise ValueError("`source` and `destination` arguments must have the same number of elements")
    order = [k for k in range(rank) if k not in source]
    for place, axis in sorted(zip(destination, source)):
        order.insert(place, axis)
    return transpose(a, order)


def column_stack(tup):
    """Vectors as columns and matrices as they are, side by side, as ``numpy.column_stack``."""
    tensors = _joined(tup, "column_stack")
    return concatenate([reshape(t, (t.shape[0], 1)) if len(t.shape) < 2 else t for t in tensors], axis=1)


def _ordered(name, x1, x2):
    """`x1` and `x2` of one real dtype for an element-wise order (a complex one refused: no order here)."""
    x1, x2 = _same_dtype(x1, x2)
    if x1.dtype.startswith("complex"):
        raise TypeError(f"{name}: complex numbers are not ordered here (NumPy orders them by real, then imaginary part)")
    return x1, x2


def maximum(x1, x2, /):
    """The larger of `x1` and `x2` element-wise, broadcast, as ``numpy.maximum``: a NaN wins (the first's where
    both), and +0 over −0 whatever their order, as IEEE 754-2019's maximum (NumPy's on arm64, and :func:`max`'s).
    Real dtypes."""
    x1, x2 = _ordered("maximum", x1, x2)
    if x1.dtype != "bool":
        return _numerical._extreme_pair(True, x1, x2)  # one pass, broadcast
    return logical_or(x1, x2)


def minimum(x1, x2, /):
    """The smaller of `x1` and `x2` element-wise, broadcast, as ``numpy.minimum``: a NaN wins (the first's where
    both), and −0 under +0 whatever their order, as IEEE 754-2019's minimum. Real dtypes."""
    x1, x2 = _ordered("minimum", x1, x2)
    if x1.dtype != "bool":
        return _numerical._extreme_pair(False, x1, x2)  # one pass, broadcast
    return logical_and(x1, x2)


def clip(a, a_min=None, a_max=None):
    """`a` limited to ``[a_min, a_max]`` element-wise, as ``numpy.clip``: ``minimum(maximum(a, a_min), a_max)``, either
    bound None for none (not both). A NaN in `a` stays."""
    if a_min is None and a_max is None:
        raise ValueError("One of max or min must be given")
    a = _operand(a)
    if a_min is not None:
        a = maximum(a, a_min)
    return a if a_max is None else minimum(a, a_max)


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Where ``|a − b| ≤ atol + rtol |b|`` element-wise, as ``numpy.isclose``: infinities close only to themselves,
    NaNs to nothing unless `equal_nan`."""
    b = (float(b) if isinstance(b, int) else b) if _weak(b) else _tensor(b)
    if isinstance(b, Tensor) and b.dtype in ("int64", "bool"):  # NumPy's: `b` inexact, so that no difference wraps
        b = asarray(b, "float64")
    a, b = _same_dtype(a, b)
    close = logical_and(less_equal(absolute(a - b), atol + rtol * absolute(b)), isfinite(b))
    close = logical_or(close, equal(a, b))
    if equal_nan:
        close = logical_or(close, logical_and(isnan(a), isnan(b)))
    return close


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Whether every pair is close (:func:`isclose`), as ``numpy.allclose``."""
    return bool(all(isclose(a, b, rtol, atol, equal_nan)))


def dot(a, b):
    """The dot product, as ``numpy.dot``.

    A number times an array; two vectors' sum of products; a matrix product; for more axes the sum over `a`'s last
    axis and `b`'s second-to-last.

    Args:
        a: The left operand: a Tensor or an array-like.
        b: The right operand: a Tensor or an array-like.

    Returns:
        Tensor | float | complex: The product; a number for two vectors.

    Raises:
        LinAlgError: if the summed axes differ in extent.
        TypeError: if an operand is nothing :func:`asarray` reads.

    Examples:
        >>> import numerical
        >>> numerical.dot([1, 2], [3, 4])
        11.0
        >>> numerical.dot(2, [1, 2])
        numerical.Tensor([2.0, 4.0], dtype=float64)
        >>> numerical.dot([[1, 2], [3, 4]], [1, 1])
        numerical.Tensor([3.0, 7.0], dtype=float64)
    """
    a, b = _operand(a), _operand(b)
    ra, rb = len(a.shape), len(b.shape)
    if ra == 0 or rb == 0:
        return a * b
    if rb == 1:
        letters = _labels(ra, "dot")
        return _number_if_scalar(einsum(f"{letters},{letters[-1]}->{letters[:-1]}", a, b))
    letters = _labels(ra + rb - 1, "dot")
    summed = letters[-1]  # a's last axis and b's second-to-last
    batch, last = letters[ra - 1:ra + rb - 3], letters[ra + rb - 3]
    return einsum(f"{letters[:ra - 1]}{summed},{batch}{summed}{last}->{letters[:ra - 1]}{batch}{last}", a, b)


def inner(a, b, /):
    """The sum of products over the last axes of `a` and `b`, as ``numpy.inner``: a number times an array where one
    is a number."""
    a, b = _operand(a), _operand(b)
    ra, rb = len(a.shape), len(b.shape)
    if ra == 0 or rb == 0:
        return a * b
    letters = _labels(ra + rb - 1, "inner")
    left = letters[:ra]
    right = letters[ra:ra + rb - 1] + letters[ra - 1]
    return _number_if_scalar(einsum(f"{left},{right}->{left[:-1]}{right[:-1]}", a, b))


def vdot(a, b, /):
    """The sum of ``conj(a) · b`` over both arrays flattened, as ``numpy.vdot``."""
    a, b = ravel(_operand(a)), ravel(_operand(b))
    if a.shape != b.shape:
        raise ValueError("cannot reshape array: vdot requires arrays of one size")
    return _number_if_scalar(einsum("i,i->", conj(a), b))


def ptp(a, axis=None, keepdims=False):
    """The range, maximum less minimum, over every element or along `axis`, as ``numpy.ptp``."""
    a = _operand(a)
    axes = _resolved_axes(a, axis, "ptp")
    return _finished(max(a, axis=axis, keepdims=True) - min(a, axis=axis, keepdims=True), axes, keepdims)


def _finished(kept, axes, keepdims):
    """A reduction over `axes` computed with them kept (`kept`, so that its arithmetic stays in its dtype: an int64
    wrapping, a float32 rounding, a 0/0 a NaN), in the shape the caller asked for: a number for every axis without
    `keepdims`, as a full reduction answers."""
    if keepdims:
        return kept
    return _number_if_scalar(reshape(kept, tuple(e for k, e in enumerate(kept.shape) if k not in axes)))


def _variance(a, axis, dtype, ddof):
    """:func:`var` with the reduced axes kept, and those axes."""
    a = _operand(a)
    if dtype is not None:
        a = asarray(a, dtype)
    if a.dtype in ("bool", "int64"):  # NumPy's: computed in float64
        a = asarray(a, "float64")
    axes = _resolved_axes(a, axis, "var")
    count = _math.prod(a.shape[k] for k in axes)
    deviation = a - mean(a, axis=axis, keepdims=True)
    squares = real(deviation * conj(deviation)) if a.dtype.startswith("complex") else deviation * deviation
    total = sum(squares, axis=axis, keepdims=True)
    divisor = _builtins.max(count - ddof, 0)
    if divisor == 0:
        _warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=3)
    return (total / divisor if divisor else total * _math.nan), axes


def var(a, axis=None, dtype=None, ddof=0, keepdims=False):
    """The variance, ``Σ |x − mean|² / (N − ddof)`` over every element or along `axis`, as ``numpy.var``: real, of
    `a`'s precision (a complex one's parts' squares summed); `dtype` the one to compute in."""
    variance, axes = _variance(a, axis, dtype, ddof)
    return _finished(variance, axes, keepdims)


def std(a, axis=None, dtype=None, ddof=0, keepdims=False):
    """The standard deviation, the square root of :func:`var`, as ``numpy.std``."""
    variance, axes = _variance(a, axis, dtype, ddof)
    return _finished(sqrt(variance), axes, keepdims)


def _real_function(name, doc):
    def function(x, /):
        if not isinstance(x, Tensor):
            mapped = _numerical._borrowed_unary(name, x)  # a real NumPy array read where it lies
            if mapped is not NotImplemented:
                return mapped
        x = _operand(x)
        if x.dtype.startswith("complex"):
            raise TypeError(f"{name}: real numbers only here (NumPy's takes complex ones too)")
        if x.dtype == "int64":  # NumPy's: an integer its own floor, ceiling and truncation, its sign an integer
            if name in ("floor", "ceil", "trunc"):
                return x
            if name == "sign":
                return asarray(x > 0, "int64") - asarray(x < 0, "int64")
            x = asarray(x, "float64")
        return _numerical._real_unary(name, x)

    function.__name__ = name
    function.__doc__ = doc + " Real dtypes; C's function where it has one, within an ulp or two of NumPy's own."
    return function


floor = _real_function("floor", "The largest integer at or below every element, as ``numpy.floor``.")
ceil = _real_function("ceil", "The smallest integer at or above every element, as ``numpy.ceil``.")
trunc = _real_function("trunc", "Every element rounded toward zero, as ``numpy.trunc``.")
rint = _real_function("rint", "Every element rounded to the nearest integer, half to even, as ``numpy.rint``.")
sign = _real_function("sign", "−1, 0 or 1 by every element's sign (+0 for either zero, a NaN kept), as ``numpy.sign``.")
arcsin = _real_function("arcsin", "The inverse sine of every element, as ``numpy.arcsin``.")
arccos = _real_function("arccos", "The inverse cosine of every element, as ``numpy.arccos``.")
arctan = _real_function("arctan", "The inverse tangent of every element, as ``numpy.arctan``.")
sinh = _real_function("sinh", "The hyperbolic sine of every element, as ``numpy.sinh``.")
cosh = _real_function("cosh", "The hyperbolic cosine of every element, as ``numpy.cosh``.")
arcsinh = _real_function("arcsinh", "The inverse hyperbolic sine of every element, as ``numpy.arcsinh``.")
arccosh = _real_function("arccosh", "The inverse hyperbolic cosine of every element, as ``numpy.arccosh``.")
arctanh = _real_function("arctanh", "The inverse hyperbolic tangent of every element, as ``numpy.arctanh``.")
log2 = _real_function("log2", "The base-2 logarithm of every element, as ``numpy.log2``.")
log10 = _real_function("log10", "The base-10 logarithm of every element, as ``numpy.log10``.")
log1p = _real_function("log1p", "``log(1 + x)`` of every element, accurate near 0, as ``numpy.log1p``.")
expm1 = _real_function("expm1", "``exp(x) − 1`` of every element, accurate near 0, as ``numpy.expm1``.")
cbrt = _real_function("cbrt", "The cube root of every element, as ``numpy.cbrt``.")
deg2rad = _real_function("deg2rad", "Every element in degrees to radians, ``x · π/180``, as ``numpy.deg2rad``.")
rad2deg = _real_function("rad2deg", "Every element in radians to degrees, ``x · 180/π``, as ``numpy.rad2deg``.")
radians = deg2rad
degrees = rad2deg


def round(a, decimals=0):  # noqa: A001 (numpy.round's name)
    """Every element rounded to `decimals` places (tens, hundreds for a negative one), half to even, as
    ``numpy.round``: ``rint(a · 10ᵈ) / 10ᵈ`` (``· 10⁻ᵈ`` for a negative `decimals`), NumPy's own way, with its
    rounding of the scaled value; a complex number's parts apart. A value already exact to those places (its ulp
    coarser than them) is kept, where NumPy's product can overflow (``round(1e307, 2)``: inf) or move its last bit,
    and a finite value rounds to 0 where ``10⁻ᵈ`` overflows (NumPy's: NaN)."""
    decimals = _operator.index(decimals)
    a = _operand(a)
    if a.dtype == "bool":
        a = asarray(a, "float64")
    if a.dtype == "int64":  # NumPy's: an integer to tens, hundreds, exactly, half to even; itself to places
        if decimals >= 0:
            return a
        if -decimals >= 20:  # half of 10²⁰ past int64: every integer rounds to 0
            return zeros(a.shape, "int64")
        if -decimals == 19:  # past ±5·10¹⁸ to ±10¹⁹, wrapped as int64's arithmetic wraps; else 0
            ten = 10**19 - 2**64
            return where(a > 5 * 10**18, asarray(ten, "int64"), where(a < -5 * 10**18, asarray(-ten, "int64"), 0))
        unit = 10**-decimals
        below = a % unit  # 0 ≤ below < unit
        base = a - below
        up = (below > unit // 2) | ((below == unit // 2) & ((base // unit) % 2 == 1))  # past half, or half to even
        return where(up, base + unit, base)
    return _numerical._round(a, _builtins.max(-400, _builtins.min(decimals, 400)))  # past ±400, any dtype's answer is the same


around = round


def _along(name, a, axis):
    """The native along-axis operation `name` of `a`: every element flattened where `axis` is None."""
    a = _operand(a)
    if axis is None:
        return _native(_numerical._axis_op, name, reshape(a, (-1,)), 0)
    return _native(_numerical._axis_op, name, a, _axis(a, axis))


def prod(a, axis=None, keepdims=False):
    """The product of every element (a number), or the products along `axis`, as ``numpy.prod``: in order (NumPy's
    multiplications are not paired as its sums are). An empty one's is 1."""
    a = _operand(a)
    if a.dtype == "bool":
        a = asarray(a, "float64")
    if a.dtype.startswith("complex") and not _numpy_loops_measured:  # each product as NumPy's loops fuse it
        _numpy_loops()
    if axis is None:
        result = _numerical._axis_op("prod", a, -1)
        return full((1,) * len(a.shape), result, a.dtype) if keepdims else result
    axes = _resolved_axes(a, axis, "prod")
    if len(axes) > 1:  # NumPy's order: the reduced axes' elements in turn, as one axis of them in row-major order
        rank = len(a.shape)
        kept = [k for k in range(rank) if k not in axes]
        # before the kept axes where NumPy's innermost loop runs along a kept one (its vector loop, between slices),
        # after them where it runs along a reduced one (its reduction loop): the product fuses as that loop does
        innermost = next((k for k in range(rank - 1, -1, -1) if a.shape[k] != 1), None)
        if innermost in kept:
            moved = _permuted(a, sorted(axes) + kept)
            joined = reshape(moved, (_math.prod(moved.shape[:len(axes)]),) + moved.shape[len(axes):])
            result = _native(_numerical._axis_op, "prod", joined, 0)
        else:
            joined, _ = _last_axis(a, axes, "prod")
            result = _native(_numerical._axis_op, "prod", joined, len(joined.shape) - 1)
    else:
        result = a if not axes else _native(_numerical._axis_op, "prod", a, axes[0])
    if keepdims:
        result = reshape(result, tuple(1 if k in axes else a.shape[k] for k in range(len(a.shape))))
    return result


def cumsum(a, axis=None):
    """The running sums along `axis` (every element flattened where None), in order, as ``numpy.cumsum``."""
    a = _operand(a)
    return _along("cumsum", asarray(a, "float64") if a.dtype == "bool" else a, axis)


def cumprod(a, axis=None):
    """The running products along `axis` (every element flattened where None), in order, as ``numpy.cumprod``."""
    a = _operand(a)
    if a.dtype.startswith("complex") and not _numpy_loops_measured:  # each product as NumPy's loops fuse it
        _numpy_loops()
    return _along("cumprod", asarray(a, "float64") if a.dtype == "bool" else a, axis)


def sort(a, axis=-1, kind=None, stable=None):
    """The elements sorted along `axis` (every element flattened where None), as ``numpy.sort``: ascending, NaNs
    last, a complex number by its real part then its imaginary part, False before True; equal elements keep their order whatever `kind`
    (NumPy's default sort may order them otherwise, which only a zero's sign or a NaN's payload shows)."""
    del kind, stable
    a = _tensor(a)
    if a.dtype == "bool":  # False before True: sorted as 0 and 1, which keeps no other element to tell apart
        return asarray(_along("sort", asarray(a, "float64"), axis), "bool")
    return _along("sort", a, axis)


def _ordered_operand(a, name):
    """`a` as a Tensor to order: a bool one as int64 (False before True), a complex one refused where `name` takes a
    real order alone (argmax, argmin)."""
    a = _tensor(a)
    if a.dtype == "bool":
        a = asarray(a, "int64")
    return a


def _arg_extreme(largest, a, axis, keepdims, name):
    a = _ordered_operand(a, name)
    if axis is None:
        index = _native(_numerical._arg_extreme, largest, a, -1)
        return full((1,) * len(a.shape), index, "int64") if keepdims else index
    axis = _axis(a, axis)
    result = _native(_numerical._arg_extreme, largest, a, axis)
    return reshape(result, tuple(1 if k == axis else e for k, e in enumerate(a.shape))) if keepdims else result


def argmax(a, axis=None, keepdims=False):
    """The index of the largest element, flat (a Python int) or along `axis` (an int64 Tensor), as ``numpy.argmax``:
    the first of equal ones, the first NaN where there is one. Real and int64 dtypes, bool as 0 and 1.

    Raises:
        ValueError: if `a` (or the axis) is empty.

    Examples:
        >>> import numerical
        >>> numerical.argmax(numerical.asarray([[1.0, 5.0], [7.0, 2.0]]), axis=1)
        numerical.Tensor([1, 0], dtype=int64)
    """
    return _arg_extreme(True, a, axis, keepdims, "argmax")


def argmin(a, axis=None, keepdims=False):
    """The index of the smallest element, flat or along `axis`, as ``numpy.argmin``: the first of equal ones, the
    first NaN where there is one."""
    return _arg_extreme(False, a, axis, keepdims, "argmin")


def argsort(a, axis=-1, kind=None, stable=None):
    """The indices that sort `a` along `axis` (every element flattened where None), as ``numpy.argsort``: an int64
    Tensor, ascending, NaNs last, equal elements by index whatever `kind` (NumPy's ``kind="stable"``).

    Examples:
        >>> import numerical
        >>> numerical.argsort(numerical.asarray([3.0, 1.0, 2.0]))
        numerical.Tensor([1, 2, 0], dtype=int64)
    """
    del kind, stable
    a = _ordered_operand(a, "argsort")
    if axis is None:
        return _native(_numerical._argsort, reshape(a, (-1,)), 0)
    return _native(_numerical._argsort, a, _axis(a, axis))


def nonzero(a):
    """The indices of the nonzero (true) elements, one int64 Tensor per axis, in C order, as ``numpy.nonzero``: a NaN
    is nonzero, as is a complex number either part of which is.

    Raises:
        ValueError: for a rank-0 `a`, as NumPy 2's.
    """
    a = _tensor(a)
    if not a.shape:
        raise ValueError("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.")
    return _numerical._nonzero(a)


def flatnonzero(a):
    """The flat indices of the nonzero elements, an int64 Tensor, as ``numpy.flatnonzero``."""
    return nonzero(ravel(a))[0]


def argwhere(a):
    """The indices of the nonzero elements, one row each of `a`'s rank, as ``numpy.argwhere``: an int64 Tensor of
    shape ``(N, a.ndim)``."""
    a = _tensor(a)
    if not a.shape:
        return zeros((int(bool(a[()])), 0), "int64")
    return stack(_numerical._nonzero(a), axis=1)


def count_nonzero(a, axis=None, *, keepdims=False):
    """The number of nonzero elements, of every element (a Python int) or along `axis` (an int64 Tensor; a tuple of
    axes for several), as ``numpy.count_nonzero``."""
    a = _tensor(a)
    if axis is None and not keepdims:
        return _numerical._count_nonzero(a, -1)
    axes = _resolved_axes(a, axis, "count_nonzero")
    if not axes:
        return asarray(a != 0, "int64") if a.dtype != "bool" else asarray(a, "int64")
    counted = _native(_numerical._count_nonzero, a, axes[-1])
    for k in sorted(axes[:-1], reverse=True):
        counted = sum(counted, axis=k)
    if keepdims:
        counted = reshape(counted, tuple(1 if k in axes else e for k, e in enumerate(a.shape)))
    return _number_if_scalar(counted) if not keepdims and len(axes) == len(a.shape) else counted


def searchsorted(a, v, side="left", sorter=None):
    """Where each element of `v` would go in the sorted vector `a` to keep it sorted, as ``numpy.searchsorted``: the
    first such place (`side` "left") or the last ("right"), NaNs ordered last; `sorter` the indices that sort `a`
    (:func:`argsort`'s). A Python int for a number `v`, else an int64 Tensor of its shape.

    Examples:
        >>> import numerical
        >>> numerical.searchsorted(numerical.asarray([1.0, 2.0, 2.0, 3.0]), [2.0, 0.5], side="right")
        numerical.Tensor([3, 0], dtype=int64)
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right' (got {side!r})")
    a = _ordered_operand(a, "searchsorted")
    if len(a.shape) != 1:
        raise ValueError("object too deep for desired array")
    if sorter is not None:
        a = a[asarray(sorter, "int64")]
    number = isinstance(v, _numbers.Number)
    a, v = _same_dtype(a, v)
    found = _native(_numerical._searchsorted, a, v, side == "right")
    return found[()] if number else found


def diff(a, n=1, axis=-1):
    """The `n`-th differences along `axis`, ``a[1:] − a[:-1]`` repeated, as ``numpy.diff``."""
    a = _operand(a)
    n = _operator.index(n)
    if n < 0:
        raise ValueError(f"order must be non-negative but got {n}")
    if not a.shape:
        raise ValueError("diff requires input that is at least one dimensional")
    axis = _axis(a, axis)
    rank = len(a.shape)
    for _ in range(n):
        upper = tuple(slice(1, None) if k == axis else slice(None) for k in range(rank))
        lower = tuple(slice(None, -1) if k == axis else slice(None) for k in range(rank))
        a = a[upper] - a[lower]
    return a


def _real_binary_function(name, doc, integers=None):
    """NumPy's real binary function `name` of two operands promoted to one dtype (:func:`_same_dtype`): native for
    float64 and float32; int64 by `integers` where NumPy keeps integers, else, as bools, in float64."""
    def function(x1, x2, /):
        a, b = _same_dtype(x1, x2)
        if a.dtype.startswith("complex"):
            raise TypeError(f"{name}: real numbers only (NumPy's too)")
        if a.dtype == "int64" and integers is not None:
            return integers(a, b)
        if a.dtype in ("int64", "bool"):
            a, b = asarray(a, "float64"), asarray(b, "float64")
        return _native(_numerical._real_binary, name, a, b)

    function.__name__ = name
    function.__doc__ = doc
    return function


def _integer_fmod(a, b):
    """C's ``a % b`` of int64 tensors, the remainder of `a`'s sign (by zero 0, as NumPy's): Python's, of `b`'s sign,
    less `b` where the two signs differ."""
    r = remainder(a, b)
    return where((r != 0) & ((r < 0) ^ (a < 0)), r - b, r)


hypot = _real_binary_function("hypot", """``√(x1² + x2²)`` element-wise without overflow, as ``numpy.hypot`` (C's
    ``hypot``). Real dtypes; integers in float64.

    Examples:
        >>> import numerical
        >>> numerical.hypot([3.0, 1e300], [4.0, 1e300])
        numerical.Tensor([5.0, 1.4142135623730952e+300], dtype=float64)
    """)
arctan2 = _real_binary_function("arctan2", """The angle of the point ``(x2, x1)`` element-wise, in ``[−π, π]``, as
    ``numpy.arctan2`` (C's ``atan2``: the zeros' signs pick the quadrant). Real dtypes; integers in float64.

    Examples:
        >>> import numerical
        >>> numerical.arctan2([1.0, 0.0], [0.0, -1.0])
        numerical.Tensor([1.5707963267948966, 3.141592653589793], dtype=float64)
    """)
atan2 = arctan2
copysign = _real_binary_function("copysign", """`x1`'s magnitude with `x2`'s sign element-wise (a zero's and a NaN's
    sign too), as ``numpy.copysign``. Real dtypes; integers in float64.""")
nextafter = _real_binary_function("nextafter", """The next representable number after `x1` toward `x2` element-wise,
    as ``numpy.nextafter``. Real dtypes; integers in float64.""")
fmod = _real_binary_function("fmod", """C's remainder ``x1 − trunc(x1 / x2) · x2`` element-wise, of `x1`'s sign (where
    :func:`remainder` takes `x2`'s), as ``numpy.fmod``: by zero NaN, an int64 0. Real dtypes; int64 stays.

    Examples:
        >>> import numerical
        >>> numerical.fmod([7.0, -7.0], 2)
        numerical.Tensor([1.0, -1.0], dtype=float64)
    """, _integer_fmod)
logaddexp = _real_binary_function("logaddexp", """``log(exp(x1) + exp(x2))`` element-wise without overflow, as
    ``numpy.logaddexp`` and by its formula. Real dtypes; integers in float64.

    Examples:
        >>> import numerical
        >>> numerical.logaddexp([1000.0], [1000.0])
        numerical.Tensor([1000.6931471805599], dtype=float64)
    """)
logaddexp2 = _real_binary_function("logaddexp2", """``log2(2**x1 + 2**x2)`` element-wise without overflow, as
    ``numpy.logaddexp2`` and by its formula. Real dtypes; integers in float64.""")
fmax = _real_binary_function("fmax", """The larger of `x1` and `x2` element-wise, a NaN losing to a number (NaN only
    where both are), as ``numpy.fmax``; +0 over −0 whatever their order (NumPy's on arm64). Real dtypes; int64
    stays.""", lambda a, b: maximum(a, b))
fmin = _real_binary_function("fmin", """The smaller of `x1` and `x2` element-wise, a NaN losing to a number, as
    ``numpy.fmin``; −0 under +0 whatever their order. Real dtypes; int64 stays.""", lambda a, b: minimum(a, b))
heaviside = _real_binary_function("heaviside", """The Heaviside step of `x1` element-wise, as ``numpy.heaviside``:
    0 below 0, `x2` at either zero, 1 above, NaN for a NaN. Real dtypes; integers in float64.

    Examples:
        >>> import numerical
        >>> numerical.heaviside([-1.0, 0.0, 2.0], 0.5)
        numerical.Tensor([0.0, 0.5, 1.0], dtype=float64)
    """)


def float_power(x1, x2, /):
    """``x1 ** x2`` element-wise in float64 (complex128 for a complex operand), as ``numpy.float_power``: an integer
    to a negative power a fraction, where :func:`power` refuses it."""
    a, b = _same_dtype(x1, x2)
    dtype = "complex128" if a.dtype.startswith("complex") else "float64"
    return asarray(a, dtype) ** asarray(b, dtype)


def angle(z, deg=False):
    """The argument of every element, ``arctan2(imag, real)`` in ``[−π, π]``, as ``numpy.angle`` (a real's 0, or π
    where it is negative or −0); in degrees for `deg`."""
    z = _operand(z)
    if z.dtype.startswith("complex"):
        result = arctan2(imag(z), real(z))
    else:
        z = asarray(z, z.dtype if z.dtype in ("float64", "float32") else "float64")
        result = arctan2(zeros((), z.dtype), z)
    return result * (180 / _math.pi) if deg else result


def unwrap(p, discont=None, axis=-1, *, period=2 * _math.pi):
    """`p` with each jump along `axis` of more than `discont` (by default half the `period`) taken back by a whole
    number of periods, as ``numpy.unwrap`` and by its formula: each difference brought into ``[−period/2, period/2)``
    (a difference of exactly half a period kept positive where it was), the corrections summed from the start. Real
    dtypes; integers in float64."""
    p = _operand(p)
    if p.dtype.startswith("complex"):
        raise TypeError("unwrap: real numbers only (NumPy's too)")
    if p.dtype in ("int64", "bool"):
        p = asarray(p, "float64")
    axis = _axis(p, axis)
    dd = diff(p, axis=axis)
    half = period / 2
    ddmod = remainder(dd + half, period) - half
    ddmod = where((ddmod == -half) & (dd > 0), asarray(half, ddmod.dtype), ddmod)
    correction = ddmod - dd
    correction = where(absolute(dd) < (half if discont is None else discont), asarray(0, ddmod.dtype), correction)
    rank = len(p.shape)
    first = tuple(slice(0, 1) if k == axis else slice(None) for k in range(rank))
    rest = tuple(slice(1, None) if k == axis else slice(None) for k in range(rank))
    return concatenate([p[first], p[rest] + cumsum(correction, axis=axis)], axis=axis)

def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """`x` with each NaN replaced by `nan`, +inf by `posinf` and −inf by `neginf` (by default the dtype's largest
    finite number and its negative), a complex element's parts apart, as ``numpy.nan_to_num``; integers and bools
    as they are. `copy` is NumPy's and changes nothing here: a Tensor is never written into.

    Examples:
        >>> import numerical
        >>> numerical.nan_to_num([float("nan"), float("inf"), -float("inf"), 1.0])
        numerical.Tensor([0.0, 1.7976931348623157e+308, -1.7976931348623157e+308, 1.0], dtype=float64)
    """
    del copy
    x = _operand(x)
    if x.dtype in ("int64", "bool"):
        return x
    largest = 3.4028234663852886e38 if x.dtype in ("float32", "complex64") else _sys.float_info.max
    return _numerical._nan_to_num(x, float(nan), largest if posinf is None else float(posinf),
                                  -largest if neginf is None else float(neginf))


def _along_axis(axis, rank, index):
    """The key that applies `index` to the axis `axis` of a rank-`rank` tensor, every other axis whole."""
    return (slice(None),) * axis + (index,) + (slice(None),) * (rank - axis - 1)


def broadcast_shapes(*args):
    """The shape `args` (shapes, an integer a rank-1 one) broadcast to, as ``numpy.broadcast_shapes``.

    Examples:
        >>> import numerical
        >>> numerical.broadcast_shapes((3, 1), (4,), 1)
        (3, 4)
    """
    shapes = [_shape(shape) for shape in args]
    rank = _builtins.max((len(shape) for shape in shapes), default=0)
    out = []
    for extents in zip(*((1,) * (rank - len(shape)) + shape for shape in shapes)):
        wide = {extent for extent in extents if extent != 1}
        if len(wide) > 1:
            raise ValueError("shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between "
                             + " and ".join(str(shape) for shape in shapes))
        out.append(wide.pop() if wide else 1)
    return tuple(out)


def broadcast_to(array, shape):
    """`array` repeated along the axes of extent 1 (and new leading ones) to `shape`, as ``numpy.broadcast_to``: a
    copy here (a Tensor is never written, so it reads as NumPy's read-only view would)."""
    a = _tensor(array)
    shape = _shape(shape)
    if _builtins.any(extent < 0 for extent in shape):
        raise ValueError("all elements of broadcast shape must be non-negative")
    if len(a.shape) > len(shape) or _builtins.any(
            have not in (1, want) for have, want in zip(reversed(a.shape), reversed(shape))):
        raise ValueError("operands could not be broadcast together with remapped shapes [original->remapped]: "
                         f"{a.shape} and requested shape {shape}")
    if a.shape == shape:
        return a
    return _numerical._where(full(shape, True, "bool"), a, a)


def broadcast_arrays(*args):
    """The arrays broadcast to one shape (:func:`broadcast_shapes`), as ``numpy.broadcast_arrays``: a tuple."""
    tensors = [_tensor(a) for a in args]
    shape = broadcast_shapes(*(t.shape for t in tensors))
    return tuple(broadcast_to(t, shape) for t in tensors)


def tile(A, reps):  # noqa: N803 (numpy.tile's argument)
    """`A` repeated `reps` times along each axis, as ``numpy.tile``: `A` and `reps` brought to one rank by leading
    extents and counts of 1.

    Examples:
        >>> import numerical
        >>> numerical.tile([1, 2], (2, 2))
        numerical.Tensor([[1.0, 2.0, 1.0, 2.0],
                          [1.0, 2.0, 1.0, 2.0]], dtype=float64)
    """
    a = _tensor(A)
    reps = (_operator.index(reps),) if isinstance(reps, _numbers.Integral) else tuple(_operator.index(r) for r in reps)
    rank = _builtins.max(len(reps), len(a.shape))
    shape = (1,) * (rank - len(a.shape)) + a.shape
    reps = (1,) * (rank - len(reps)) + reps
    spread = reshape(a, tuple(x for extent in shape for x in (1, extent)))
    repeated = broadcast_to(spread, tuple(x for count, extent in zip(reps, shape) for x in (count, extent)))
    return reshape(repeated, tuple(count * extent for count, extent in zip(reps, shape)))


def _counts(values, name):
    """Integers of an array-like (a Tensor, a list, a NumPy array) as a Python list, refusing any other kind."""
    if isinstance(values, Tensor):
        if values.dtype not in ("int64", "bool"):
            raise TypeError(f"{name}: Cannot cast array data from dtype('{values.dtype}') to dtype('int64') according "
                            "to the rule 'safe'")
        return [int(v) for v in reshape(values, (-1,)).tolist()]
    if type(values).__name__ == "ndarray":
        if values.dtype.kind not in "iub":
            raise TypeError(f"{name}: Cannot cast array data from dtype('{values.dtype}') to dtype('int64') according "
                            "to the rule 'safe'")
        return [int(v) for v in values.ravel().tolist()]
    flat, _ = _nested(values)
    if not _builtins.all(isinstance(v, _numbers.Integral) for v in flat):
        raise TypeError(f"{name}: Cannot cast array data from dtype('float64') to dtype('int64') according to the "
                        "rule 'safe'")
    return [int(v) for v in flat]


def _index_tensor(values, name):
    """Integers of an array-like (a Tensor, a list, a NumPy array) as an int64 Tensor of its shape, a bool one read
    as 0 and 1, refusing any other kind."""
    if isinstance(values, Tensor):
        if values.dtype not in ("int64", "bool"):
            raise TypeError(f"{name}: Cannot cast array data from dtype('{values.dtype}') to dtype('int64') according "
                            "to the rule 'safe'")
        return values if values.dtype == "int64" else asarray(values, "int64")
    if type(values).__name__ == "ndarray" and values.dtype.kind not in "iub":
        raise TypeError(f"{name}: Cannot cast array data from dtype('{values.dtype}') to dtype('int64') according "
                        "to the rule 'safe'")
    if type(values).__name__ != "ndarray":
        flat, shape = _nested(values)
        if not _builtins.all(isinstance(v, _numbers.Integral) for v in flat):
            raise TypeError(f"{name}: Cannot cast array data from dtype('float64') to dtype('int64') according to "
                            "the rule 'safe'")
        return reshape(asarray([int(v) for v in flat], "int64"), shape)
    return asarray(values, "int64")


def repeat(a, repeats, axis=None):
    """Each element of `a` repeated, along `axis` (of the flattened `a` where None), as ``numpy.repeat``: `repeats`
    one count for all, or one per element along the axis.

    Examples:
        >>> import numerical
        >>> numerical.repeat([1, 2], 2)
        numerical.Tensor([1.0, 1.0, 2.0, 2.0], dtype=float64)
        >>> numerical.repeat([[1, 2], [3, 4]], [1, 2], axis=0)
        numerical.Tensor([[1.0, 2.0],
                          [3.0, 4.0],
                          [3.0, 4.0]], dtype=float64)
    """
    a = _tensor(a)
    if axis is None:
        a, axis = reshape(a, (-1,)), 0
    axis = _axis(a, axis)
    extent = a.shape[axis]
    counts = [_operator.index(repeats)] if isinstance(repeats, _numbers.Integral) else _counts(repeats, "repeat")
    if _builtins.any(count < 0 for count in counts):
        raise ValueError("repeats may not contain negative values.")
    if len(counts) == 1:  # every element as often: a broadcast, natively
        count = counts[0]
        spread = reshape(a, a.shape[:axis + 1] + (1,) + a.shape[axis + 1:])
        repeated = broadcast_to(spread, a.shape[:axis + 1] + (count,) + a.shape[axis + 1:])
        return reshape(repeated, a.shape[:axis] + (extent * count,) + a.shape[axis + 1:])
    if len(counts) != extent:
        raise ValueError(f"operands could not be broadcast together with shape ({extent},) ({len(counts)},)")
    index = [position for position, count in enumerate(counts) for _ in range(count)]
    return a[_along_axis(axis, len(a.shape), index)]


def take(a, indices, axis=None, mode="raise"):
    """The elements of `a` at `indices` along `axis` (of the flattened `a` where None), as ``numpy.take``: the
    result's shape `a`'s with that axis replaced by the indices' shape. `mode` "raise" refuses an index out of the
    axis, "wrap" takes it modulo the extent, "clip" to the nearest end."""
    a = _tensor(a)
    if axis is None:
        a, axis = reshape(a, (-1,)), 0
    axis = _axis(a, axis)
    if mode not in ("raise", "wrap", "clip"):
        raise ValueError(f"clipmode must be one of 'clip', 'raise', or 'wrap' (got {mode!r})")
    extent = a.shape[axis]
    if isinstance(indices, _numbers.Integral):
        flat, shape = asarray([_operator.index(indices)], "int64"), ()
    else:
        flat = _index_tensor(indices, "take")
        shape = flat.shape
        flat = reshape(flat, (-1,))
    before, after = _math.prod(a.shape[:axis]), _math.prod(a.shape[axis + 1:])
    out_shape = a.shape[:axis] + shape + a.shape[axis + 1:]
    # NumPy's check, of its result's size; "wrap" of an index from an empty axis loops forever there (numpy#11677)
    if extent == 0 and (before * flat.size * after or (mode == "wrap" and flat.size)):
        raise IndexError("cannot do a non-empty take from an empty axes.")
    if mode == "raise" and before and flat.size:  # NumPy's: checked wherever a slice of the other axes stands
        flat = _bounded_offsets(flat, extent, axis, 1)
    if not before * flat.size * after:
        return zeros(out_shape, a.dtype)
    if mode == "wrap":
        flat = flat % extent
    elif mode == "clip":
        flat = minimum(maximum(flat, 0), extent - 1)
    picked = a[_along_axis(axis, len(a.shape), flat)]
    return reshape(picked, out_shape)


def take_along_axis(arr, indices, axis=-1):
    """The elements of `arr` at `indices` along `axis`, the other axes matched (broadcast), as
    ``numpy.take_along_axis``: the companion of :func:`argsort` (``take_along_axis(a, argsort(a, axis), axis)`` sorts);
    `axis` None for the flattened `arr` and indices of rank 1.

    Examples:
        >>> import numerical
        >>> a = numerical.asarray([[3.0, 1.0], [2.0, 5.0]])
        >>> numerical.take_along_axis(a, numerical.argsort(a, axis=1), axis=1)
        numerical.Tensor([[1.0, 3.0],
                          [2.0, 5.0]], dtype=float64)
    """
    arr = _tensor(arr)
    indices = indices if isinstance(indices, Tensor) else asarray(indices)
    if indices.dtype != "int64":
        if indices.dtype == "float64" and _builtins.all(float(v).is_integer() for v in reshape(indices, (-1,)).tolist()):
            indices = asarray(indices, "int64")  # integers read as float64 by asarray's default
        else:
            raise IndexError("`indices` must be an integer array")
    if axis is None:
        if len(indices.shape) != 1:
            raise ValueError("when axis=None, `indices` must have a single dimension.")
        arr, axis = reshape(arr, (-1,)), 0
    axis = _axis(arr, axis)
    rank = len(arr.shape)
    if len(indices.shape) != rank:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")
    others = broadcast_shapes(arr.shape[:axis] + arr.shape[axis + 1:], indices.shape[:axis] + indices.shape[axis + 1:])
    shape_of = (lambda extent: others[:axis] + (extent,) + others[axis:])  # the other axes broadcast, as NumPy's
    if arr.shape != shape_of(arr.shape[axis]):
        arr = broadcast_to(arr, shape_of(arr.shape[axis]))
    if indices.shape != shape_of(indices.shape[axis]):
        indices = broadcast_to(indices, shape_of(indices.shape[axis]))
    extent = arr.shape[axis]
    if indices.size:
        low, high = int(min(indices)), int(max(indices))
        if low < -extent or high >= extent:
            bad = low if low < -extent else high
            raise IndexError(f"index {bad} is out of bounds for axis {axis} with size {extent}")
        if low < 0:
            indices = where(indices < 0, indices + extent, indices)
    return _native(_numerical._take_along_axis, arr, indices, axis)


def roll(a, shift, axis=None):
    """`a` with its elements shifted by `shift` along `axis` (of the flattened `a` where None), those past the end
    back at the start, as ``numpy.roll``; `shift` and `axis` may be sequences of one length (or one of them a single
    value), a repeated axis shifted by the sum.

    Examples:
        >>> import numerical
        >>> numerical.roll([1, 2, 3, 4], 1)
        numerical.Tensor([4.0, 1.0, 2.0, 3.0], dtype=float64)
    """
    a = _tensor(a)
    if axis is None:
        return reshape(roll(reshape(a, (-1,)), shift, 0), a.shape)
    shifts = (shift,) if isinstance(shift, _numbers.Integral) else tuple(shift)
    axes = (axis,) if isinstance(axis, _numbers.Integral) else tuple(axis)
    if len(shifts) == 1:
        shifts *= len(axes)
    elif len(axes) == 1:
        axes *= len(shifts)
    if len(shifts) != len(axes):
        raise ValueError("shape mismatch: objects cannot be broadcast to a single shape")
    rank = len(a.shape)
    total = [0] * rank
    for step, along in zip(shifts, axes):
        total[_normalized_axes(along, rank, "roll")[0]] += _operator.index(step)
    for along, step in enumerate(total):
        extent = a.shape[along]
        step = step % extent if extent else 0
        if step:
            a = concatenate([a[_along_axis(along, rank, slice(extent - step, None))],
                             a[_along_axis(along, rank, slice(None, extent - step))]], axis=along)
    return a


def rot90(m, k=1, axes=(0, 1)):
    """`m` rotated by 90 degrees `k` times in the plane of `axes`, from the first axis toward the second, as
    ``numpy.rot90``.

    Examples:
        >>> import numerical
        >>> numerical.rot90([[1, 2], [3, 4]])
        numerical.Tensor([[2.0, 4.0],
                          [1.0, 3.0]], dtype=float64)
    """
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")
    m = _tensor(m)
    rank = len(m.shape)
    if axes[0] == axes[1] or _builtins.abs(axes[0] - axes[1]) == rank:
        raise ValueError("Axes must be different.")
    if not (-rank <= axes[0] < rank and -rank <= axes[1] < rank):
        raise ValueError(f"Axes={axes} out of range for array of ndim={rank}.")
    k = _operator.index(k) % 4
    if k == 0:
        return m
    if k == 2:
        return flip(flip(m, axes[0]), axes[1])
    order = list(range(rank))
    order[axes[0]], order[axes[1]] = order[axes[1]], order[axes[0]]
    return transpose(flip(m, axes[1]), order) if k == 1 else flip(transpose(m, order), axes[1])


def array_split(ary, indices_or_sections, axis=0):
    """`ary` cut along `axis` into a list of tensors, as ``numpy.array_split``: at the given indices (slices between
    them, empty where they do not advance), or into that many sections, the first ``extent % sections`` one longer."""
    a = _tensor(ary)
    axis = _axis(a, axis)
    extent = a.shape[axis]
    if isinstance(indices_or_sections, _numbers.Integral):
        sections = _operator.index(indices_or_sections)
        if sections <= 0:
            raise ValueError("number sections must be larger than 0.")
        each, extras = _builtins.divmod(extent, sections)
        points = [0]
        for size in [each + 1] * extras + [each] * (sections - extras):
            points.append(points[-1] + size)
    else:
        points = [0] + _counts(indices_or_sections, "array_split") + [extent]
    rank = len(a.shape)
    return [a[_along_axis(axis, rank, slice(start, stop))] for start, stop in zip(points, points[1:])]


def split(ary, indices_or_sections, axis=0):
    """`ary` cut along `axis` into a list of tensors, as ``numpy.split``: :func:`array_split`, a number of sections
    dividing the extent.

    Examples:
        >>> import numerical
        >>> numerical.split([1, 2, 3, 4], 2)
        [numerical.Tensor([1.0, 2.0], dtype=float64), numerical.Tensor([3.0, 4.0], dtype=float64)]
    """
    a = _tensor(ary)
    if isinstance(indices_or_sections, _numbers.Integral) and indices_or_sections > 0 \
            and a.shape[_axis(a, axis)] % indices_or_sections:
        raise ValueError("array split does not result in an equal division")
    return array_split(a, indices_or_sections, axis)


def hsplit(ary, indices_or_sections):
    """:func:`split` along the second axis (the first of a vector), as ``numpy.hsplit``."""
    a = _tensor(ary)
    if not a.shape:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    return split(a, indices_or_sections, 1 if len(a.shape) > 1 else 0)


def vsplit(ary, indices_or_sections):
    """:func:`split` along the first axis, as ``numpy.vsplit`` (rank 2 or more)."""
    a = _tensor(ary)
    if len(a.shape) < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return split(a, indices_or_sections, 0)


def dsplit(ary, indices_or_sections):
    """:func:`split` along the third axis, as ``numpy.dsplit`` (rank 3 or more)."""
    a = _tensor(ary)
    if len(a.shape) < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return split(a, indices_or_sections, 2)


def dstack(tup):
    """The arrays along the third axis, as ``numpy.dstack``: each first of rank 3 (:func:`atleast_3d`)."""
    tensors = _joined(tup, "dstack")
    shaped = atleast_3d(*tensors)
    return concatenate(shaped if isinstance(shaped, list) else [shaped], axis=2)


def append(arr, values, axis=None):
    """`values` after `arr` along `axis`, as ``numpy.append``: both flattened where `axis` is None."""
    if axis is None:
        return concatenate([ravel(arr), ravel(values)], axis=0)
    return concatenate([arr, values], axis=axis)


def _positions(obj):
    """The shape of the index argument `obj`, the kind of its elements ("b" bools alone, "i" integers, "f" other
    numbers), and its numbers in C order, as NumPy reads it with ``numpy.asarray`` (an empty list as integers)."""
    if isinstance(obj, _numbers.Integral):
        return (), "b" if isinstance(obj, bool) else "i", [int(obj)]
    if isinstance(obj, Tensor):
        kind = {"bool": "b", "int64": "i"}.get(obj.dtype, "f")
        return obj.shape, kind, reshape(obj, (-1,)).tolist()
    if type(obj).__name__ == "ndarray":
        kind = {"b": "b", "i": "i", "u": "i"}.get(obj.dtype.kind, "f")
        return obj.shape, kind, obj.ravel().tolist()
    flat, shape = _nested(obj)
    if flat and _builtins.all(isinstance(v, bool) for v in flat):
        return shape, "b", flat
    return shape, "i" if _builtins.all(isinstance(v, _numbers.Integral) for v in flat) else "f", flat


_NOT_INDICES = "arrays used as indices must be of integer (or boolean) type"


def delete(arr, obj, axis=None):
    """`arr` without the entries at `obj` along `axis` (of the flattened `arr` where None), as ``numpy.delete``: `obj`
    a slice, an index, indices (negative from the end, repeated ones once) or a mask of the axis's extent.

    Examples:
        >>> import numerical
        >>> numerical.delete([[1, 2, 3], [4, 5, 6]], [0, -1], axis=1)
        numerical.Tensor([[2.0],
                          [5.0]], dtype=float64)
    """
    a = _tensor(arr)
    if axis is None:
        a, axis = reshape(a, (-1,)), 0
    axis = _axis(a, axis)
    extent, rank = a.shape[axis], len(a.shape)
    if isinstance(obj, slice):
        gone = set(range(*obj.indices(extent)))
    else:
        shape, kind, flat = _positions(obj)
        if kind == "f":
            raise IndexError(_NOT_INDICES)
        if kind == "b":
            if shape != (extent,):
                raise ValueError("boolean array argument obj to delete must be one dimensional and match the axis "
                                 f"length of {extent}")
            gone = {i for i, drop in enumerate(flat) if drop}
        else:
            for i in map(int, flat):
                if not -extent <= i < extent:
                    where = axis if len(flat) == 1 else 0  # NumPy's message: the axis, or that of its mask
                    raise IndexError(f"index {i} is out of bounds for axis {where} with size {extent}")
            gone = {int(i) % extent for i in flat}
    return a[_along_axis(axis, rank, [i for i in range(extent) if i not in gone])]


def insert(arr, obj, values, axis=None):
    """`arr` with `values` (in `arr`'s dtype) inserted before the entries at `obj` along `axis` (of the flattened `arr`
    where None), as ``numpy.insert``: `obj` a slice, an index (`values`' first axis along `axis`) or indices
    (negative from the end, the extent for the end), the values for equal indices in their order.

    Examples:
        >>> import numerical
        >>> numerical.insert([1, 2, 3], [1, 3], [9, 8])
        numerical.Tensor([1.0, 9.0, 2.0, 3.0, 8.0], dtype=float64)
        >>> numerical.insert([[1, 2], [3, 4]], 1, [7, 8], axis=1)
        numerical.Tensor([[1.0, 7.0, 2.0],
                          [3.0, 8.0, 4.0]], dtype=float64)
    """
    a = _tensor(arr)
    if axis is None:
        a, axis = reshape(a, (-1,)), 0
    axis = _axis(a, axis)
    extent, rank = a.shape[axis], len(a.shape)
    if isinstance(obj, slice):
        flat = list(range(*obj.indices(extent)))
        shape, kind = (len(flat),), "i"
    else:
        shape, kind, flat = _positions(obj)
        if kind == "b":
            if len(shape) != 1:
                raise ValueError("boolean array argument obj to insert must be one dimensional")
            shape, flat = (_builtins.sum(flat),), [i for i, chosen in enumerate(flat) if chosen]
        elif len(shape) > 1:
            raise ValueError("index array argument obj to insert must be one dimensional or scalar")
        elif kind == "f" and len(flat) != 1:  # NumPy's: several index a mask, one bounds a slice (below)
            raise IndexError(_NOT_INDICES)
    v = asarray(values, a.dtype)
    v = reshape(v, (1,) * (rank - len(v.shape)) + v.shape)
    if len(flat) == 1:
        index = flat[0]
        if not -extent <= index <= extent:
            raise IndexError(f"index {obj} is out of bounds for axis {axis} with size {extent}")
        if kind == "f":
            raise TypeError("slice indices must be integers or None or have an __index__ method")
        index += extent if index < 0 else 0
        if not shape:  # one index: values' first axis runs along `axis`
            v = moveaxis(v, 0, axis)
        indices = [index] * v.shape[axis]
        sources = list(range(index)) + list(range(extent, extent + len(indices))) + list(range(index, extent))
    else:
        # NumPy's: each index moved past the values inserted before it, in a stable order
        indices = [i + extent if i < 0 else i for i in flat]
        for shift, k in enumerate(sorted(range(len(indices)), key=indices.__getitem__)):
            indices[k] += shift
        size = extent + len(indices)
        taken = {}
        for j, i in enumerate(indices):
            if not -size <= i < size:
                raise IndexError(f"index {i} is out of bounds for axis 0 with size {size}")
            taken[i % size] = extent + j  # equal positions: the last value
        kept = [q for q in range(size) if q not in taken]
        if len(kept) != extent:
            raise ValueError(f"boolean array indexing assignment cannot assign {extent} input values to the "
                             f"{len(kept)} output values where the mask is true")
        old = iter(range(extent))
        sources = [taken[q] if q in taken else next(old) for q in range(size)]
    while len(v.shape) > rank and v.shape[0] == 1:
        v = reshape(v, v.shape[1:])
    v = broadcast_to(v, a.shape[:axis] + (len(indices),) + a.shape[axis + 1:])
    return concatenate([a, v], axis=axis)[_along_axis(axis, rank, sources)]


def block(arrays):
    """One tensor assembled from nested lists of blocks, as ``numpy.block``: the innermost lists joined along the
    last axis, the lists of them along the one before, and so on, each block first given leading axes of extent 1 up
    to the result's rank (the deeper of the nesting and the blocks' ranks).

    Examples:
        >>> import numerical
        >>> numerical.block([[numerical.eye(2), numerical.zeros((2, 1))], [numerical.ones((1, 3))]])
        numerical.Tensor([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [1.0, 1.0, 1.0]], dtype=float64)
    """
    def survey(x, index):
        if type(x) is tuple:
            raise TypeError(f"{index} is a tuple. Only lists can be used to arrange blocks, and np.block does not "
                            "allow implicit conversion from tuple to ndarray.")
        if type(x) is not list:
            return 0, len(_tensor(x).shape), index
        if not x:
            raise ValueError(f"List at {index} cannot be empty")
        found = [survey(y, f"{index}[{i}]") for i, y in enumerate(x)]
        for depth, _, where in found[1:]:
            if depth != found[0][0]:
                raise ValueError(f"List depths are mismatched. First element was at depth {found[0][0] + 1}, but "
                                 f"there is an element at depth {depth + 1} ({where})")
        return found[0][0] + 1, _builtins.max(rank for _, rank, _ in found), found[0][2]

    depth, deepest, _ = survey(arrays, "arrays")
    rank = _builtins.max(depth, deepest)

    def assembled(x, level):
        if level == depth:
            t = _tensor(x)
            return reshape(t, (1,) * (rank - len(t.shape)) + t.shape)
        return concatenate([assembled(y, level + 1) for y in x], axis=level - depth)

    return assembled(arrays, 0)


def _without_nans(a, value):
    """`a` as a Tensor with each NaN replaced by `value`, and where the NaNs were; an integer or bool `a` as it is, and
    no mask (NumPy's ``_replace_nan``: a complex element is NaN where either part is)."""
    a = _operand(a)
    if a.dtype in ("int64", "bool"):
        return a, None
    mask = isnan(a)
    return where(mask, value, a), mask


def _no_out(out, name):
    if out is not None:
        raise TypeError(f"{name}: a Tensor takes no `out` (it is never written into)")


def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    """:func:`sum` with each NaN counted as 0, as ``numpy.nansum`` (an all-NaN slice's sum 0)."""
    _no_out(out, "nansum")
    a, _ = _without_nans(a, 0)
    return sum(a if dtype is None else asarray(a, _dtype_name(dtype)), axis=axis, keepdims=keepdims)


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    """:func:`prod` with each NaN counted as 1, as ``numpy.nanprod``."""
    _no_out(out, "nanprod")
    a, _ = _without_nans(a, 1)
    return prod(a if dtype is None else asarray(a, _dtype_name(dtype)), axis=axis, keepdims=keepdims)


def nancumsum(a, axis=None, dtype=None, out=None):
    """:func:`cumsum` with each NaN counted as 0, as ``numpy.nancumsum``."""
    _no_out(out, "nancumsum")
    a, _ = _without_nans(a, 0)
    return cumsum(a if dtype is None else asarray(a, _dtype_name(dtype)), axis=axis)


def nancumprod(a, axis=None, dtype=None, out=None):
    """:func:`cumprod` with each NaN counted as 1, as ``numpy.nancumprod``."""
    _no_out(out, "nancumprod")
    a, _ = _without_nans(a, 1)
    return cumprod(a if dtype is None else asarray(a, _dtype_name(dtype)), axis=axis)


def _nan_extreme(a, axis, keepdims, larger):
    a, mask = _without_nans(a, -_math.inf if larger else _math.inf)
    result = (max if larger else min)(a, axis=axis, keepdims=keepdims)
    if mask is None:
        return result
    lost = all(mask, axis=axis, keepdims=keepdims)
    if not isinstance(lost, Tensor):
        if lost:
            _warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
            return _math.nan
        return result
    if any(lost):
        _warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
        return where(lost, _math.nan, result)
    return result


def nanmax(a, axis=None, out=None, keepdims=False):
    """:func:`max` ignoring NaNs, as ``numpy.nanmax``: NaN (with a ``RuntimeWarning``) for a slice of NaNs alone."""
    _no_out(out, "nanmax")
    return _nan_extreme(a, axis, keepdims, True)


def nanmin(a, axis=None, out=None, keepdims=False):
    """:func:`min` ignoring NaNs, as ``numpy.nanmin``: NaN (with a ``RuntimeWarning``) for a slice of NaNs alone."""
    _no_out(out, "nanmin")
    return _nan_extreme(a, axis, keepdims, False)


def _nan_arg_extreme(a, axis, keepdims, larger):
    a, mask = _without_nans(a, -_math.inf if larger else _math.inf)
    if mask is not None and any(all(mask, axis=axis)):
        raise ValueError("All-NaN slice encountered")
    return (argmax if larger else argmin)(a, axis=axis, keepdims=keepdims)


def nanargmax(a, axis=None, out=None, *, keepdims=False):
    """:func:`argmax` ignoring NaNs, as ``numpy.nanargmax``: ``ValueError`` for a slice of NaNs alone."""
    _no_out(out, "nanargmax")
    return _nan_arg_extreme(a, axis, keepdims, True)


def nanargmin(a, axis=None, out=None, *, keepdims=False):
    """:func:`argmin` ignoring NaNs, as ``numpy.nanargmin``: ``ValueError`` for a slice of NaNs alone."""
    _no_out(out, "nanargmin")
    return _nan_arg_extreme(a, axis, keepdims, False)


def _reduced_shape(result, a, axis, keepdims, name):
    """A result computed with every reduced axis kept (extent 1) in the shape `keepdims` asks: the reduced axes dropped
    without it (a number for a rank-0 result)."""
    if keepdims:
        return result
    axes = _resolved_axes(a, axis, name)
    shape = tuple(extent for k, extent in enumerate(a.shape) if k not in axes)
    return reshape(result, shape) if shape else reshape(result, ())[()]


def _by_count(total, count, dtype):
    """``total / count`` in `dtype`, as NumPy's ``_divide_by_count``: a float32 one through float64 (int64 counts),
    rounded once; NaN where `count` is 0."""
    wide = "complex128" if dtype.startswith("complex") else "float64"
    return asarray(asarray(total, wide) / asarray(count, "float64"), dtype)


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    """The mean of the elements that are not NaN, as ``numpy.nanmean``: NaN (with a ``RuntimeWarning``) for a slice
    of NaNs alone."""
    _no_out(out, "nanmean")
    arr, mask = _without_nans(a, 0)
    if dtype is not None:
        arr = asarray(arr, _dtype_name(dtype))
    if mask is None:
        return mean(arr, axis=axis, keepdims=keepdims)
    count = count_nonzero(~mask, axis=_resolved_axes(arr, axis, "nanmean"), keepdims=True)
    total = sum(arr, axis=axis, keepdims=True)
    if any(count == 0):
        _warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
    return _reduced_shape(_by_count(total, count, arr.dtype), arr, axis, keepdims, "nanmean")


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """The variance of the elements that are not NaN, as ``numpy.nanvar`` and by its steps: NaN (with a
    ``RuntimeWarning``) where they are no more than `ddof`."""
    _no_out(out, "nanvar")
    arr, mask = _without_nans(a, 0)
    if dtype is not None:
        arr = asarray(arr, _dtype_name(dtype))
    if mask is None:
        return var(arr, axis=axis, ddof=ddof, keepdims=keepdims)
    axes = _resolved_axes(arr, axis, "nanvar")
    count = count_nonzero(~mask, axis=axes, keepdims=True)
    average = _by_count(sum(arr, axis=axis, keepdims=True), count, arr.dtype)
    deviation = where(mask, 0, arr - average)
    squares = real(deviation * conj(deviation)) if arr.dtype.startswith("complex") else deviation * deviation
    total = sum(squares, axis=axis, keepdims=True)
    freedom = count - ddof
    bad = freedom <= 0
    if any(bad):
        _warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning, stacklevel=2)
    variance = where(bad, _math.nan, _by_count(total, freedom, total.dtype))
    return _reduced_shape(variance, arr, axis, keepdims, "nanvar")


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """The standard deviation of the elements that are not NaN, the square root of :func:`nanvar`, as
    ``numpy.nanstd``."""
    variance = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
    return _math.sqrt(variance) if isinstance(variance, _numbers.Number) else sqrt(variance)


def _last_axis(a, axis, name):
    """`a` with the axes `axis` (None for every one) moved last and joined into one, and the reduced axes."""
    rank = len(a.shape)
    axes = _resolved_axes(a, axis, name)
    kept = [k for k in range(rank) if k not in axes]
    if kept + sorted(axes) != list(range(rank)):
        a = _permuted(a, kept + sorted(axes))
    return reshape(a, tuple(a.shape[:len(kept)]) + (_math.prod(a.shape[len(kept):]),)), axes


def _kept_dims(result, a, axes, lead, keepdims):
    """`result` of shape ``lead + kept axes`` with each reduced axis of `a` back with extent 1 where `keepdims`; a
    number for a rank-0 result."""
    if keepdims:
        result = reshape(result, lead + tuple(1 if k in axes else extent for k, extent in enumerate(a.shape)))
    return result[()] if isinstance(result, Tensor) and not result.shape else result


def _median_of(a, axis, keepdims, ignore_nans):
    a = _tensor(a)
    if a.dtype == "bool":
        a = asarray(a, "float64")
    joined, axes = _last_axis(a, axis, "median")
    s = sort(joined, axis=-1)
    rest, extent = s.shape[:-1], s.shape[-1]
    inexact = a.dtype != "int64"
    wide = a.dtype if inexact else "float64"  # NumPy's mean of integers in float64
    if extent == 0:
        _warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=3)
        return _kept_dims(full(rest, _math.nan, wide), a, axes, (), keepdims)
    if ignore_nans and inexact:
        count = count_nonzero(~isnan(joined), axis=-1, keepdims=True)
        low = take_along_axis(s, where(count == 0, 0, (count - 1) // 2), axis=-1)
        high = take_along_axis(s, count // 2 - (count == 0), axis=-1)
        result = reshape(where(count % 2 == 1, low + 0, (low + high + 0) / 2), rest)  # sums from +0, as NumPy's
        if any(count == 0):
            _warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
            result = where(reshape(count == 0, rest), _math.nan, result)
        return _kept_dims(result, a, axes, (), keepdims)
    middle = extent // 2  # NumPy's mean of the middle one or two: their sum from +0, so −0 becomes +0
    high = asarray(reshape(s[..., middle:middle + 1], rest), wide)
    if extent % 2:
        result = high + 0
    else:
        result = (asarray(reshape(s[..., middle - 1:middle], rest), wide) + high + 0) / 2
    if inexact:
        last = reshape(s[..., -1:], rest)
        result = where(isnan(last), last, result)
    return _kept_dims(result, a, axes, (), keepdims)


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """The median of every element, or the medians along `axis` (an integer or a tuple), as ``numpy.median``: the
    middle element of the sorted ones, or the mean of the two middle ones; NaN where there is a NaN; integers in
    float64. `overwrite_input` is NumPy's and changes nothing here (a Tensor is never written into).

    Examples:
        >>> import numerical
        >>> numerical.median([[1.0, 3.0, 2.0], [4.0, 6.0, 8.0]], axis=1)
        numerical.Tensor([2.0, 6.0], dtype=float64)
    """
    del overwrite_input
    _no_out(out, "median")
    return _median_of(a, axis, keepdims, False)


def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """:func:`median` of the elements that are not NaN, as ``numpy.nanmedian``: NaN (with a ``RuntimeWarning``) for a
    slice of NaNs alone."""
    del overwrite_input
    _no_out(out, "nanmedian")
    return _median_of(a, axis, keepdims, True)


def _virtual_index(method, n, q):
    """NumPy's virtual index of the quantiles `q` among `n` sorted values (an integer or an int64 Tensor, one per
    slice) by `method`: a float64 Tensor, or an int64 one for the methods that pick an element."""
    def computed(alpha, beta):
        return n * q + (alpha + q * (1 - alpha - beta)) - 1

    def boundaries(index, lower):
        before = floor(index)
        picked = asarray(where(lower(index - before, index), before, before + 1), "int64")
        return where(picked < 0, 0, picked)

    if method == "linear":
        return (n - 1) * q if q.dtype != "int64" else asarray((n - 1) * q, "int64")
    if method == "lower":
        return asarray(floor((n - 1) * q), "int64")
    if method == "higher":
        return asarray(ceil((n - 1) * q), "int64")
    if method == "nearest":
        return asarray(round((n - 1) * q), "int64")
    if method == "midpoint":
        return 0.5 * (floor((n - 1) * q) + ceil((n - 1) * q))
    if method == "inverted_cdf":
        return boundaries(n * q - 1, lambda gamma, _: gamma == 0)
    if method == "averaged_inverted_cdf":
        return n * q - 1
    if method == "closest_observation":
        return boundaries(n * q - 1 - 0.5, lambda gamma, index: (gamma == 0) & (floor(index) % 2 == 1))
    alpha, beta = {"interpolated_inverted_cdf": (0, 1), "hazen": (0.5, 0.5), "weibull": (0, 0),
                   "median_unbiased": (1 / 3.0, 1 / 3.0), "normal_unbiased": (3 / 8.0, 3 / 8.0)}[method]
    return computed(alpha, beta)


_QUANTILE_METHODS = ("inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf",
                     "hazen", "weibull", "linear", "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint",
                     "nearest")


def _quantiles(a, q, axis, method, keepdims, ignore_nans, weak, name):
    """NumPy's ``_quantile`` of `a` (real) at `q` (in [0, 1]) over `axis`, by `method`: the virtual indexes, their
    neighbours in the sorted values, and NumPy's interpolation ``_lerp``; NaN where a slice holds a NaN (where it holds
    nothing else, for `ignore_nans`)."""
    a = _tensor(a)
    if a.dtype.startswith("complex"):
        raise TypeError("a must be an array of real numbers")
    if ignore_nans and a.size == 0:  # NumPy's: the mean of nothing, whatever `q` and `method`
        return nanmean(a, axis, keepdims=keepdims)
    if method not in _QUANTILE_METHODS:
        raise ValueError(f"{method!r} is not a valid method. Use one of: {_QUANTILE_METHODS}")
    joined, axes = _last_axis(a, axis, name)
    inexact = a.dtype in ("float64", "float32")
    if inexact and not ignore_nans and joined.shape[-1] == 0:  # NumPy's look for a NaN at the end, whatever `q`
        raise IndexError("index -1 is out of bounds for axis 0 with size 0")
    s = sort(joined, axis=-1)
    rest = s.shape[:-1]
    lead = q.shape
    flat_q = reshape(q, (-1,))
    if ignore_nans and inexact:
        n = count_nonzero(~isnan(joined), axis=-1, keepdims=True)  # one count per slice: rest + (1,)
        q_along = reshape(flat_q, (1,) * len(rest) + (flat_q.shape[0],))
    else:
        n = s.shape[-1]
        q_along = flat_q
    index = _virtual_index(method, n, q_along)
    last = n - 1  # NumPy's -1: the last value (the last that is not NaN, for `ignore_nans`)
    if index.dtype == "int64" and method in ("linear", "lower", "higher", "nearest", "inverted_cdf",
                                             "closest_observation"):
        result = _gathered(s, where(index < 0, index + n, index), rest)
    else:
        previous = floor(index)
        following = previous + 1
        above = index >= last
        below = index < 0
        previous = where(below, 0.0, where(above, -1.0, previous))  # NumPy's order: below last
        following = where(below, 0.0, where(above, -1.0, following))
        gamma = index - previous
        if method == "averaged_inverted_cdf":
            gamma = where(gamma == 0, 0.5, 1.0)
        elif method == "midpoint":
            gamma = where(index % 1 == 0, 0.0, 0.5)
        low = _gathered(s, _as_position(previous, n), rest)
        high = _gathered(s, _as_position(following, n), rest)
        if not weak:
            result = _lerp(low, high, gamma)
        elif ignore_nans:  # NumPy's slice by slice, each gamma a Python float beside the values (weak)
            result = _lerp(low, high, gamma, a.dtype if inexact else "float64")
        else:
            result = _lerp(low, high, float(reshape(gamma, (-1,))[0]))
    if inexact:
        if ignore_nans:
            empty = n == 0
            if any(empty):
                _warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
                result = where(empty, _math.nan, result)
        else:
            last = s[..., -1:]
            result = where(isnan(last), last, result)
    # rest + (Q,) to lead + rest
    result = _permuted(result, [len(rest)] + list(range(len(rest)))) if rest else result
    result = reshape(result, lead + rest)
    return _kept_dims(result, a, axes, lead, keepdims)


def _weighted_quantiles(a, q, axis, weights, keepdims, ignore_nans, name):
    """NumPy's weighted ``_quantile`` (method "inverted_cdf") of `a` (real) at `q`: in each slice sorted, the first
    value whose share of the weights (cumulated in float64, in `q`'s float dtype) reaches `q`, a leading run of zero
    weights skipped; NaN where a slice holds a NaN (`ignore_nans`: the NaNs and their weights left out, NaN where
    nothing else is left). `weights` of `a`'s shape, or of the shape of `axis`'s extents in its order."""
    a = _tensor(a)
    if a.dtype.startswith("complex"):
        raise TypeError("a must be an array of real numbers")
    rank = len(a.shape)
    w = _tensor(weights)
    if w.shape != a.shape:
        if axis is None:
            raise TypeError("Axis must be specified when shapes of a and weights differ.")
        axes = _normalized_axes(axis, rank, name)
        if w.shape != tuple(a.shape[k] for k in axes):
            raise ValueError("Shape of weights must be consistent with shape of a along specified axis.")
        w = _permuted(w, sorted(range(len(axes)), key=axes.__getitem__))
        w = broadcast_to(reshape(w, tuple(e if k in axes else 1 for k, e in enumerate(a.shape))), a.shape)
    if any(w < 0):
        raise ValueError("Weights must be non-negative.")
    if ignore_nans and a.size == 0:  # NumPy's: the mean of nothing, whatever `q`
        return nanmean(a, axis, keepdims=keepdims)
    joined, axes = _last_axis(a, axis, name)
    order = argsort(joined, axis=-1)
    s = take_along_axis(joined, order, axis=-1)
    share = asarray(take_along_axis(_last_axis(w, axis, name)[0], order, axis=-1), "float64")
    rest = s.shape[:-1]
    inexact = a.dtype in ("float64", "float32")
    empty = None
    if ignore_nans and inexact:
        share = where(isnan(s), 0.0, share)
        empty = count_nonzero(~isnan(s), axis=-1, keepdims=True) == 0
        share = where(empty, 1.0, share)  # a slice of NaNs alone: NaN below, not an error
    cdf = cumsum(share, axis=-1)
    cdf = cdf / cdf[..., -1:]
    if any(isnan(cdf[..., -1])):
        raise ValueError("Weights included NaN, inf or were all zero.")
    if q.dtype == "float32":  # NumPy's: compared in q's precision
        cdf = asarray(cdf, "float32")
    if any(cdf[..., 0] == 0):
        cdf = where(cdf == 0, -1.0, cdf)
    flat_q = reshape(q, (-1,))
    lines = reshape(cdf, (-1, cdf.shape[-1]))
    found = [searchsorted(lines[i], flat_q, side="left") for i in range(lines.shape[0])]  # NumPy's, slice by slice
    index = reshape(stack(found) if found else zeros((0, flat_q.shape[0]), "int64"), rest + flat_q.shape)
    index = minimum(index, s.shape[-1] - 1)
    result = take_along_axis(s, index, axis=-1) if rest else s[index]
    if empty is not None:
        if any(empty):
            _warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
            result = where(empty, _math.nan, result)
    elif inexact:
        last = s[..., -1:]
        result = where(isnan(last), last, result)
    result = _permuted(result, [len(rest)] + list(range(len(rest)))) if rest else result
    result = reshape(result, q.shape + rest)
    return _kept_dims(result, a, axes, q.shape, keepdims)


def _as_position(index, n):
    """A float64 virtual index (NumPy's -1 for the last value) as int64 positions among `n` values."""
    position = asarray(index, "int64")
    return where(position < 0, position + n, position)


def _gathered(s, positions, rest):
    """The values of `s` (sorted along its last axis) at `positions`: a vector of them for every slice, or one row per
    slice (shape ``rest + (Q,)``)."""
    if len(positions.shape) <= 1:
        picked = s[..., reshape(positions, (-1,))]
        return picked
    return take_along_axis(s, broadcast_to(positions, rest + positions.shape[-1:]), axis=-1)


def _lerp(a, b, t, weak=None):
    """NumPy's ``_lerp``: ``a + (b − a) t``, from `b` where ``t ≥ 0.5`` (``b − (b − a)(1 − t)``). A float64 Tensor `t`
    with a `weak` dtype stands for one Python float per element: `t` and ``1 − t`` (computed in float64) each rounded
    to that dtype, as NumPy's weak scalars are."""
    difference = b - a
    if isinstance(t, float):
        return (b - difference * (1 - t)) if t >= 0.5 else a + difference * t
    if weak is None:
        return where(t >= 0.5, b - difference * (1 - t), a + difference * t)
    return where(t >= 0.5, b - difference * asarray(1 - t, weak), a + difference * asarray(t, weak))


def _quantile_operand(q, scale, name):
    weak = type(q) in (int, float)
    values = asarray(q, "int64") if isinstance(q, _numbers.Integral) or (
        isinstance(q, Tensor) and q.dtype == "int64") else _read(q)
    if len(values.shape) > 2:
        raise ValueError("q must be a scalar or 1d")
    if scale != 1:
        values = asarray(values, "float64") / scale
    if values.dtype == "bool":
        values = asarray(values, "float64")
    flat = reshape(values, (-1,)).tolist()
    if not _builtins.all(0.0 <= v <= 1.0 for v in flat):
        raise ValueError(f"{name} must be in the range [0, {'100' if scale != 1 else '1'}]")
    return values, weak


def quantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, *, weights=None):
    """The `q`-th quantiles (in [0, 1]) of every element or along `axis`, as ``numpy.quantile`` and by its steps (its
    13 methods, its interpolation, NumPy 2.4's weak Python `q`: a float32 kept beside it); the quantiles' axes
    first. Real dtypes. `weights` (method "inverted_cdf" alone, as NumPy's), non-negative, of `a`'s shape or of the
    extents of `axis` in its order: the first sorted value whose share of the weights reaches `q`.

    Examples:
        >>> import numerical
        >>> numerical.quantile([1.0, 2.0, 3.0, 4.0], [0.25, 0.5])
        numerical.Tensor([1.75, 2.5], dtype=float64)
        >>> numerical.quantile([1.0, 2.0, 3.0], 0.5, method="inverted_cdf", weights=[1.0, 1.0, 3.0])
        3.0
    """
    return _quantile_entry(a, q, axis, out, method, keepdims, weights, 1, False, "quantile")


def percentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, *, weights=None):
    """The `q`-th percentiles (in [0, 100]) of every element or along `axis`, as ``numpy.percentile``:
    :func:`quantile` at ``q / 100``."""
    return _quantile_entry(a, q, axis, out, method, keepdims, weights, 100, False, "percentile")


def nanquantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, *, weights=None):
    """:func:`quantile` of the elements that are not NaN, as ``numpy.nanquantile``: NaN (with a ``RuntimeWarning``)
    for a slice of NaNs alone."""
    return _quantile_entry(a, q, axis, out, method, keepdims, weights, 1, True, "nanquantile")


def nanpercentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, *, weights=None):
    """:func:`percentile` of the elements that are not NaN, as ``numpy.nanpercentile``."""
    return _quantile_entry(a, q, axis, out, method, keepdims, weights, 100, True, "nanpercentile")


def _quantile_entry(a, q, axis, out, method, keepdims, weights, scale, ignore_nans, name):
    _no_out(out, name)
    values, weak = _quantile_operand(q, scale, "Quantiles" if scale == 1 else "Percentiles")
    if weights is None:
        return _quantiles(a, values, axis, method, keepdims, ignore_nans, weak, name)
    if method != "inverted_cdf":
        raise ValueError(f"Only method 'inverted_cdf' supports weights. Got: {method}.")
    return _weighted_quantiles(a, values, axis, weights, keepdims, ignore_nans, name)


def average(a, axis=None, weights=None, returned=False, *, keepdims=False):
    """The mean of `a`, or its mean weighted by `weights`, over every element or along `axis` (an integer or a tuple),
    as ``numpy.average``: ``Σ a·w / Σ w``; with `returned`, the pair of it and the sum of the weights (the count
    without them). `weights` of `a`'s shape, or of the extents of `axis` alone.

    Examples:
        >>> import numerical
        >>> numerical.average([1.0, 2.0, 4.0], weights=[3.0, 1.0, 0.0])
        1.25
    """
    a = _operand(a)
    axes = None if axis is None else _resolved_axes(a, axis, "average")
    if weights is None:
        avg = mean(a, axis=axes, keepdims=keepdims)
        if not returned:
            return avg
        count = a.size / (avg.size if isinstance(avg, Tensor) else 1)  # NumPy's, in the mean's dtype
        if not isinstance(avg, Tensor):
            return avg, type(avg)(count)
        return avg, full(avg.shape, count, avg.dtype)
    w = _operand(weights)
    if w.shape != a.shape:
        if axes is None:
            raise TypeError("Axis must be specified when shapes of a and weights differ.")
        if w.shape != tuple(a.shape[k] for k in axes):
            raise ValueError("Shape of weights must be consistent with shape of a along specified axis.")
        order = sorted(range(len(axes)), key=lambda k: axes[k])
        w = transpose(w, order) if order != list(range(len(axes))) else w
        w = reshape(w, tuple(extent if k in axes else 1 for k, extent in enumerate(a.shape)))
    dtype = _promoted((a.dtype, w.dtype, "float64") if a.dtype in ("int64", "bool") else (a.dtype, w.dtype))
    scale = sum(asarray(w, dtype), axis=axes, keepdims=keepdims)
    if any(asarray(scale) == 0.0):
        raise ZeroDivisionError("Weights sum to zero, can't be normalized")
    avg = sum(asarray(a, dtype) * asarray(w, dtype), axis=axes, keepdims=keepdims) / scale
    if not returned:
        return avg
    if isinstance(avg, Tensor) and (not isinstance(scale, Tensor) or scale.shape != avg.shape):
        scale = broadcast_to(scale, avg.shape)
    return avg, scale


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None):
    """The covariance matrix of the variables in the rows of `m` (its columns without `rowvar`), the rows of `y`
    joined after them, as ``numpy.cov`` and by its steps: the deviations from the (weighted) means, their products
    summed (a matrix product) and scaled by the reciprocal of ``N − ddof`` (of its weighted form for `fweights`, integer
    repeat counts, and `aweights`, relative weights). One variable's a number.

    Examples:
        >>> import numerical
        >>> numerical.cov([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
        numerical.Tensor([[1.0, -1.0],
                          [-1.0, 1.0]], dtype=float64)
    """
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")
    m = _operand(m)
    if len(m.shape) > 2:
        raise ValueError("m has more than 2 dimensions")
    if y is not None:
        y = _operand(y)
        if len(y.shape) > 2:
            raise ValueError("y has more than 2 dimensions")
    if dtype is None:
        dtype = _promoted([m.dtype, "float64"] + ([] if y is None else [y.dtype]))
    dtype = _dtype_name(dtype)
    x = asarray(m, dtype)
    x = reshape(x, (1,) * (2 - len(x.shape)) + x.shape) if len(x.shape) < 2 else x
    if not rowvar and len(m.shape) != 1:
        x = transpose(x)
    if x.shape[0] == 0:
        return zeros((0, 0))
    if y is not None:
        y = asarray(y, dtype)
        y = reshape(y, (1,) * (2 - len(y.shape)) + y.shape) if len(y.shape) < 2 else y
        if not rowvar and y.shape[0] != 1:
            y = transpose(y)
        x = concatenate([x, y], axis=0)
    if ddof is None:
        ddof = 1 if bias == 0 else 0
    w = None
    if fweights is not None:
        fweights = asarray(fweights, "float64")
        if not all(fweights == round(fweights)):
            raise TypeError("fweights must be integer")
        if len(fweights.shape) > 1:
            raise RuntimeError("cannot handle multidimensional fweights")
        if fweights.shape[0] != x.shape[1]:
            raise RuntimeError("incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError("fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = asarray(aweights, "float64")
        if len(aweights.shape) > 1:
            raise RuntimeError("cannot handle multidimensional aweights")
        if aweights.shape[0] != x.shape[1]:
            raise RuntimeError("incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError("aweights cannot be negative")
        w = aweights if w is None else w * aweights
    avg, w_sum = average(x, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]
    if w is None:
        fact = x.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * sum(w * aweights) / w_sum
    if fact <= 0:
        _warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        fact = 0.0
    x = x - reshape(avg, (avg.shape[0], 1))
    x_t = transpose(x if w is None else x * w)
    c = x @ conj(x_t) if dtype.startswith("complex") else x @ x_t
    c = c * (1 / fact if fact else _math.inf)  # NumPy's product by the reciprocal; 1/0 its inf
    return c[0, 0] if c.shape == (1, 1) else c


def corrcoef(x, y=None, rowvar=True, *, dtype=None):
    """The Pearson correlation coefficients of the variables in the rows of `x` (its columns without `rowvar`), the
    rows of `y` joined after them, as ``numpy.corrcoef``: :func:`cov` divided by the products of the standard
    deviations, then clipped to [−1, 1] (each part of a complex coefficient)."""
    c = cov(x, y, rowvar, dtype=dtype)
    if not isinstance(c, Tensor) or not c.shape:
        c = c if isinstance(c, Tensor) else asarray(c)
        return _number_if_scalar(c / c)  # NaN for a constant variable, as NumPy's
    d = diagonal(c)
    stddev = sqrt(real(d))
    c = c / reshape(stddev, (stddev.shape[0], 1))
    c = c / reshape(stddev, (1, stddev.shape[0]))
    if c.dtype.startswith("complex"):
        return _from_parts(clip(real(c), -1, 1), clip(imag(c), -1, 1))
    return clip(c, -1, 1)


def _from_parts(re, im):
    """``re + im·i`` of two real tensors of one shape and dtype, exactly (each part's zero keeps its sign, where a sum
    and a product by 1j would not): their elements side by side, read as complex ones."""
    return _native(_numerical._complex_from_parts, re, im)


def _pad_pairs(x, rank, as_index=False):
    """NumPy's ``_as_pairs``: `x` (a number, a pair, or one pair per axis) as one ``(before, after)`` pair per axis;
    rounded to integers, none negative, for `as_index`."""
    if x is None:
        return [(None, None)] * rank
    if isinstance(x, Tensor):
        values, shape = reshape(x, (-1,)).tolist(), x.shape
    elif type(x).__name__ == "ndarray":
        values, shape = x.ravel().tolist(), tuple(x.shape)
    else:
        values, shape = _nested(x)
    if as_index:
        values = [int(_builtins.round(v)) for v in values]
        if values and _builtins.min(values) < 0:
            raise ValueError("index can't contain negative values")
    if len(shape) < 3 and len(values) == 1:
        return [(values[0], values[0])] * rank
    if len(shape) < 3 and len(values) == 2 and tuple(shape) != (2, 1):
        return [(values[0], values[1])] * rank
    shape = (1,) * (2 - len(shape)) + tuple(shape)
    if len(shape) > 2 or shape[0] not in (1, rank) or shape[1] not in (1, 2):
        raise ValueError("operands could not be broadcast together with remapped shapes [original->remapped]: "
                         f"{tuple(shape)} and requested shape ({rank},2)")
    return [tuple(values[(i if shape[0] != 1 else 0) * shape[1] + (j if shape[1] != 1 else 0)] for j in range(2))
            for i in range(rank)]


def _ramp(start, stop, num, dtype, axis):
    """NumPy's ``linspace(start, stop, num, endpoint=False, dtype=dtype, axis=axis)`` of a number `start` and a tensor
    `stop`, by its steps: ``arange(num) · step + start``, through ``/ num · delta`` where a step is 0; floored for an
    integer `dtype`. Computed in the dtype `start` and `stop` promote to, `start` as NumPy's ``_as_pairs`` hands it: a
    NumPy scalar, of full strength (a Python integer an int64, a float a float64: a float32 ramp is computed in
    float64)."""
    strong = getattr(start, "dtype", None)
    strong = _dtype_name(strong) if strong is not None else (
        "complex128" if isinstance(start, complex) else "float64" if isinstance(start, float) else "int64")
    computed = _promoted([strong, dtype])
    computed = "float64" if computed in ("int64", "bool") else computed
    stop = asarray(stop, computed)
    delta = stop - start
    y = reshape(asarray(arange(num), computed), (-1,) + (1,) * len(stop.shape))
    if num > 0:
        step = delta / num
        y = (y / num) * delta if any(step == 0) else y * step
    else:
        y = y * delta  # no element, of the ramp's shape
    y = y + start
    y = moveaxis(y, 0, axis) if axis else y
    return asarray(floor(y) if dtype == "int64" else y, dtype)


def _pad_reflected(t, axis, before, after, odd, period, include_edge):
    """NumPy's ``_set_reflect_both``, repeated: the values mirrored at each end of `t` along `axis` (about the edge
    for "reflect", past it for "symmetric"), a chunk of at most one period at a time; "odd" mirrors them through the
    edge value (``2 · edge − value``)."""
    rank = len(t.shape)
    offset = 1 if include_edge else 0
    while before > 0 or after > 0:
        length = t.shape[axis]
        if include_edge:
            reach = length // period * period
        else:
            reach = (length - 1) // (period - 1) * (period - 1)
        left = right = None
        if before > 0:
            chunk = _builtins.min(reach, before)
            left = flip(t[_along_axis(axis, rank, slice(1 - offset, chunk - offset + 1))], axis)
            if odd:  # back to t's dtype, as NumPy's assignment casts it (a bool true where nonzero)
                left = asarray(2 * t[_along_axis(axis, rank, slice(0, 1))] - left, t.dtype)
            before -= chunk
        if after > 0:
            chunk = _builtins.min(reach, after)
            right = flip(t[_along_axis(axis, rank, slice(length - 1 + offset - chunk, length - 1 + offset))], axis)
            if odd:
                right = asarray(2 * t[_along_axis(axis, rank, slice(length - 1, length))] - right, t.dtype)
            after -= chunk
        t = concatenate([part for part in (left, t, right) if part is not None], axis=axis)
    return t


def _pad_wrapped(t, axis, before, after, period):
    """NumPy's ``_set_wrap_both``, repeated: each end of `t` along `axis` continued by the other end's values, a whole
    number of periods at a time."""
    rank = len(t.shape)
    while before > 0 or after > 0:
        length = t.shape[axis]
        whole = length // period * period
        left = right = None
        later_before = later_after = 0
        if before > 0:
            chunk = _builtins.min(whole, before)
            left = t[_along_axis(axis, rank, slice(whole - chunk, whole))]
            later_before = before - whole if before > whole else 0
        if after > 0:
            chunk = _builtins.min(whole, after)
            right = t[_along_axis(axis, rank, slice(length - whole, length - whole + chunk))]
            later_after = after - whole if after > whole else 0
        t = concatenate([part for part in (left, t, right) if part is not None], axis=axis)
        before, after = later_before, later_after
    return t


_PAD_KEYWORDS = {"empty": (), "edge": (), "wrap": (), "constant": ("constant_values",), "linear_ramp": ("end_values",),
                 "maximum": ("stat_length",), "mean": ("stat_length",), "median": ("stat_length",),
                 "minimum": ("stat_length",), "reflect": ("reflect_type",), "symmetric": ("reflect_type",)}


def pad(array, pad_width, mode="constant", **kwargs):
    """`array` padded by `pad_width` (one count for every side, a ``(before, after)`` pair, one pair per axis, or a
    dict of them by axis) as ``numpy.pad`` and by its steps, axis after axis: "constant" (`constant_values`), "edge",
    "linear_ramp" (to `end_values`), "maximum", "minimum", "mean" and "median" (of `stat_length` values from each
    end), "reflect" and "symmetric" (`reflect_type` "even" or "odd"), "wrap", and "empty" (zeros here). A function as
    `mode` is refused: NumPy's writes into the vectors it is handed, and a Tensor is never written into.

    Examples:
        >>> import numerical
        >>> numerical.pad([1.0, 2.0, 3.0], (2, 1), mode="reflect")
        numerical.Tensor([3.0, 2.0, 1.0, 2.0, 3.0, 2.0], dtype=float64)
        >>> numerical.pad([[1.0]], 1, constant_values=9)
        numerical.Tensor([[9.0, 9.0, 9.0],
                          [9.0, 1.0, 9.0],
                          [9.0, 9.0, 9.0]], dtype=float64)
    """
    a = _tensor(array)
    rank = len(a.shape)
    if isinstance(pad_width, dict):
        widths = [(0, 0)] * rank
        for axis, width in pad_width.items():
            widths[axis] = (width, width) if isinstance(width, int) else tuple(width)
        pad_width = widths
    if isinstance(pad_width, Tensor):
        integral = pad_width.dtype == "int64"
    elif type(pad_width).__name__ == "ndarray":
        integral = pad_width.dtype.kind == "i"
    else:
        flat, _ = _nested(pad_width)
        integral = bool(flat) and _builtins.all(isinstance(v, _numbers.Integral) and not isinstance(v, bool) for v in flat)
    if not integral:  # NumPy's: an empty list reads as float64
        raise TypeError("`pad_width` must be of integral type.")
    widths = _pad_pairs(pad_width, rank, as_index=True)
    if callable(mode):
        raise TypeError("pad: a function as mode writes into the vectors it is handed; a Tensor is never written into")
    if mode not in _PAD_KEYWORDS:
        raise ValueError(f"mode '{mode}' is not supported")
    unsupported = set(kwargs) - set(_PAD_KEYWORDS[mode])
    if unsupported:
        raise ValueError(f"unsupported keyword arguments for mode '{mode}': {unsupported}")

    def shaped(t, axis, extent):
        return tuple(extent if k == axis else e for k, e in enumerate(t.shape))

    if mode in ("constant", "empty") or a.size == 0:
        if mode not in ("constant", "empty"):
            for axis, pair in enumerate(widths):
                if a.shape[axis] == 0 and _builtins.any(pair):
                    raise ValueError(f"can't extend empty axis {axis} using modes other than 'constant' or 'empty'")
        values = _pad_pairs(kwargs.get("constant_values", 0), rank) if mode == "constant" else [(0, 0)] * rank
        for axis, ((before, after), (low, high)) in enumerate(zip(widths, values)):
            a = concatenate([full(shaped(a, axis, before), low, a.dtype), a, full(shaped(a, axis, after), high, a.dtype)],
                            axis=axis)
        return a
    if mode in ("reflect", "symmetric"):
        odd = kwargs.get("reflect_type", "even") == "odd"
    lengths = _pad_pairs(kwargs.get("stat_length"), rank, as_index=True) if mode in ("maximum", "minimum", "mean",
                                                                                     "median") else None
    ends = _pad_pairs(kwargs.get("end_values", 0), rank) if mode == "linear_ramp" else None
    for axis, (before, after) in enumerate(widths):
        if not before and not after:
            continue
        extent = a.shape[axis]
        first = a[_along_axis(axis, rank, slice(0, 1))]
        last = a[_along_axis(axis, rank, slice(extent - 1, extent))]
        if mode == "edge" or (mode in ("reflect", "symmetric") and extent == 1):
            a = concatenate([broadcast_to(first, shaped(a, axis, before)), a,
                             broadcast_to(last, shaped(a, axis, after))], axis=axis)
        elif mode in ("reflect", "symmetric"):
            a = _pad_reflected(a, axis, before, after, odd, extent, mode == "symmetric")
        elif mode == "wrap":
            a = _pad_wrapped(a, axis, before, after, extent)
        elif mode == "linear_ramp":
            low = _ramp(ends[axis][0], reshape(first, tuple(e for k, e in enumerate(first.shape) if k != axis)), before,
                        a.dtype, axis)
            high = flip(_ramp(ends[axis][1], reshape(last, tuple(e for k, e in enumerate(last.shape) if k != axis)),
                              after, a.dtype, axis), axis)
            a = concatenate([low, a, high], axis=axis)
        else:
            reduce = {"maximum": max, "minimum": min, "mean": mean, "median": median}[mode]
            left_length = extent if lengths[axis][0] is None or extent < lengths[axis][0] else lengths[axis][0]
            right_length = extent if lengths[axis][1] is None or extent < lengths[axis][1] else lengths[axis][1]
            if (left_length == 0 or right_length == 0) and mode in ("maximum", "minimum"):
                raise ValueError("stat_length of 0 yields no value for padding")

            def stat(start, stop):
                value = reduce(a[_along_axis(axis, rank, slice(start, stop))], axis=axis, keepdims=True)
                return asarray(round(value), "int64") if a.dtype == "int64" else asarray(value, a.dtype)

            low = stat(0, left_length)
            high = low if left_length == right_length == extent else stat(extent - right_length, extent)
            a = concatenate([broadcast_to(low, shaped(a, axis, before)), a, broadcast_to(high, shaped(a, axis, after))],
                            axis=axis)
    return a

Tensor.nonzero = lambda self: nonzero(self)
for _name in ("swapaxes", "clip", "ptp", "var", "std", "dot", "prod", "cumsum", "cumprod", "round", "argmax", "argmin",
              "argsort", "searchsorted"):  # NumPy's methods of these names, where Tensor has none
    if not hasattr(Tensor, _name):
        setattr(Tensor, _name, (lambda function: lambda self, *args, **kwargs: function(self, *args, **kwargs))(globals()[_name]))
del _name

def _binary_operands(x1, x2):
    """`x1` and `x2` as operands of an arithmetic operator: Tensors and numbers as they are (a number weak beside a
    Tensor), anything else read by :func:`asarray`, two numbers as tensors."""
    if not isinstance(x1, (Tensor, _numbers.Number)):
        x1 = _operand(x1)
    if not isinstance(x2, (Tensor, _numbers.Number)):
        x2 = _operand(x2)
    if not isinstance(x1, Tensor) and not isinstance(x2, Tensor):
        x1 = _operand(x1)
    return x1, x2


def add(x1, x2, /):
    """``x1 + x2`` element-wise, as ``numpy.add``."""
    x1, x2 = _binary_operands(x1, x2)
    return x1 + x2


def subtract(x1, x2, /):
    """``x1 - x2`` element-wise, as ``numpy.subtract``."""
    x1, x2 = _binary_operands(x1, x2)
    return x1 - x2


def multiply(x1, x2, /):
    """``x1 * x2`` element-wise, as ``numpy.multiply``."""
    x1, x2 = _binary_operands(x1, x2)
    return x1 * x2


acos, asin, atan = arccos, arcsin, arctan  # the array API's names, NumPy 2's aliases
acosh, asinh, atanh = arccosh, arcsinh, arctanh
pow = power
amax, amin = max, min


def permute_dims(a, axes=None):
    """`a` with its axes in the order `axes` (reversed where None), as ``numpy.permute_dims`` (:func:`transpose`)."""
    return transpose(a, axes)


def reciprocal(x, /):
    """``1 / x`` element-wise, as ``numpy.reciprocal``: in `x`'s dtype, a complex one by NumPy's loop (Smith's, the
    larger part divided first, fused as NumPy's; an infinity and a NaN for a zero); an integer's truncated (0 past
    ±1), which raises for a zero as an infinity converted to int64 does."""
    x = _tensor(x)
    if x.dtype == "bool":
        raise TypeError("ufunc 'reciprocal' not supported for the input types (bool)")
    if x.dtype == "int64":
        return asarray(1.0 / asarray(x, "float64"), "int64")
    if not x.dtype.startswith("complex"):
        return 1 / x
    if not _numpy_loops_measured:  # fused as NumPy's quotient loop
        _numpy_loops()
    return _native(_numerical._power_shortcut, "r", x)


def fabs(x, /):
    """The magnitude of every real element, as ``numpy.fabs``: an integer's in float64; a complex `x` refused."""
    x = _tensor(x)
    if x.dtype.startswith("complex"):
        raise TypeError("ufunc 'fabs' not supported for the input types (complex)")
    if x.dtype == "bool":
        raise TypeError("fabs of bool is NumPy's float16, a dtype NumericAL does not have: convert to float first")
    return abs(asarray(x, "float64") if x.dtype == "int64" else x)


def signbit(x, /):
    """Whether each real element's sign bit is set (−0 and a negative NaN's too), as ``numpy.signbit``."""
    x = _tensor(x)
    if x.dtype.startswith("complex"):
        raise TypeError("ufunc 'signbit' not supported for the input types (complex)")
    if x.dtype == "bool":
        return zeros_like(x)
    if x.dtype == "int64":
        return x < 0
    return copysign(1, x) < 0


def isposinf(x, out=None):
    """Whether each real element is +∞, as ``numpy.isposinf``."""
    _no_out(out, "isposinf")
    x = _tensor(x)
    if x.dtype.startswith("complex"):
        raise TypeError(f"This operation is not supported for {x.dtype} values because it would be ambiguous.")
    return isinf(x) & ~signbit(x)


def isneginf(x, out=None):
    """Whether each real element is −∞, as ``numpy.isneginf``."""
    _no_out(out, "isneginf")
    x = _tensor(x)
    if x.dtype.startswith("complex"):
        raise TypeError(f"This operation is not supported for {x.dtype} values because it would be ambiguous.")
    return isinf(x) & signbit(x)


def iscomplexobj(x):
    """Whether `x` is of a complex dtype (or a complex number), as ``numpy.iscomplexobj``; its values unread."""
    dtype = getattr(x, "dtype", None)
    if dtype is not None:
        return _dtype_name(dtype).startswith("complex") or str(dtype).startswith("complex")
    return _tensor(x).dtype.startswith("complex")


def isrealobj(x):
    """Whether `x` is not of a complex dtype, as ``numpy.isrealobj``."""
    return not iscomplexobj(x)


def iscomplex(x):
    """Whether each element has a nonzero imaginary part, as ``numpy.iscomplex``: false throughout for a real dtype."""
    x = _tensor(x)
    return imag(x) != 0 if x.dtype.startswith("complex") else zeros_like(x, "bool")


def isreal(x):
    """Whether each element's imaginary part is zero, as ``numpy.isreal``: true throughout for a real dtype."""
    return ~iscomplex(x)


def isscalar(element):
    """Whether `element` is a scalar, as ``numpy.isscalar``: a number, a string or bytes (NumPy's scalars too), not a
    Tensor, even of rank 0."""
    return isinstance(element, (_numbers.Number, str, bytes)) or type(element).__module__ == "numpy" and \
        type(element).__name__ not in ("ndarray", "dtype") and hasattr(element, "dtype")


def shape(a):
    """`a`'s shape, as ``numpy.shape``."""
    return _tensor(a).shape if not isinstance(a, Tensor) else a.shape


def ndim(a):
    """`a`'s number of axes, as ``numpy.ndim``."""
    return len(shape(a))


def size(a, axis=None):
    """`a`'s number of elements, or its extent along `axis` (an integer or a tuple of them), as ``numpy.size``."""
    extents = shape(a)
    if axis is None:
        return _math.prod(extents)
    return _math.prod(extents[k] for k in _normalized_axes(axis, len(extents), "size"))


def tri(N, M=None, k=0, dtype=float):
    """The ``N × M`` matrix (square where `M` is None) of ones at and below the `k`-th diagonal, zeros above, as
    ``numpy.tri``."""
    N = _operator.index(N)
    M = N if M is None else _operator.index(M)
    k = _operator.index(k)
    rows = reshape(asarray(arange(N), "int64"), (N, 1))
    mask = reshape(asarray(arange(M), "int64"), (1, M)) <= rows + k
    return asarray(mask, _dtype_name(dtype))


def tril_indices(n, k=0, m=None):
    """The row and column indices of the ``n × m`` lower triangle, to and below the `k`-th diagonal, as
    ``numpy.tril_indices``: int64 tensors, in row-major order."""
    return nonzero(tri(n, m, k=k, dtype=bool))


def triu_indices(n, k=0, m=None):
    """The row and column indices of the ``n × m`` upper triangle, to and above the `k`-th diagonal, as
    ``numpy.triu_indices``."""
    return nonzero(~tri(n, m, k=k - 1, dtype=bool))


def tril_indices_from(arr, k=0):
    """:func:`tril_indices` of a matrix's shape, as ``numpy.tril_indices_from``."""
    extents = shape(arr)
    if len(extents) != 2:
        raise ValueError("input array must be 2-d")
    return tril_indices(extents[-2], k=k, m=extents[-1])


def triu_indices_from(arr, k=0):
    """:func:`triu_indices` of a matrix's shape, as ``numpy.triu_indices_from``."""
    extents = shape(arr)
    if len(extents) != 2:
        raise ValueError("input array must be 2-d")
    return triu_indices(extents[-2], k=k, m=extents[-1])


def diag_indices(n, ndim=2):
    """The indices of the main diagonal of an `n`-cube of `ndim` axes, as ``numpy.diag_indices``: ``ndim`` copies of
    ``0 … n−1`` (int64)."""
    steps = asarray(arange(_operator.index(n)), "int64")
    return (steps,) * _operator.index(ndim)


def diag_indices_from(arr):
    """:func:`diag_indices` of an array of equal extents, as ``numpy.diag_indices_from``."""
    extents = shape(arr)
    if len(extents) < 2:
        raise ValueError("input array must be at least 2-d")
    if len(set(extents)) != 1:
        raise ValueError("All dimensions of input must be of equal length")
    return diag_indices(extents[0], len(extents))


def meshgrid(*xi, copy=True, sparse=False, indexing="xy"):
    """Coordinate tensors from coordinate vectors, as ``numpy.meshgrid``: one per vector, each varying along its own
    axis (the first two swapped for ``indexing="xy"``, matrix order for ``"ij"``), broadcast to the grid unless
    `sparse`. `copy` is NumPy's and changes nothing (a Tensor is never written into).

    Examples:
        >>> import numerical
        >>> x, y = numerical.meshgrid([1.0, 2.0, 3.0], [4.0, 5.0])
        >>> x.shape, y.tolist()
        ((2, 3), [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
    """
    del copy
    if indexing not in ("xy", "ij"):
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")
    count = len(xi)
    ones = (1,) * count
    output = [reshape(_tensor(x), ones[:i] + (-1,) + ones[i + 1:]) for i, x in enumerate(xi)]
    if indexing == "xy" and count > 1:
        output[0] = reshape(output[0], (1, -1) + ones[2:])
        output[1] = reshape(output[1], (-1, 1) + ones[2:])
    if not sparse:
        output = broadcast_arrays(*output)
    return tuple(output)


__all__ += ["add", "subtract", "multiply", "acos", "asin", "atan", "acosh", "asinh", "atanh", "pow", "amax", "amin",
            "permute_dims", "reciprocal", "fabs", "signbit", "isposinf", "isneginf", "iscomplexobj", "isrealobj",
            "iscomplex", "isreal", "isscalar", "shape", "ndim", "size", "tri", "tril_indices", "triu_indices",
            "tril_indices_from", "triu_indices_from", "diag_indices", "diag_indices_from", "meshgrid"]

def _cumulative(x, scan, axis, dtype, out, include_initial, identity, name):
    """NumPy's ``_cumulative_func``: `scan` along `axis` (required past rank 1) of `x` at least of rank 1, in `dtype`,
    after the `identity` where `include_initial`."""
    _no_out(out, name)
    x = atleast_1d(_tensor(x))
    if axis is None:
        if len(x.shape) >= 2:
            raise ValueError("For arrays which have more than one dimension ``axis`` argument is required.")
        axis = 0
    axis = _axis(x, axis)
    if dtype is not None:
        x = asarray(x, _dtype_name(dtype))
    result = scan(x, axis=axis)
    if include_initial:
        result = concatenate([full(x.shape[:axis] + (1,) + x.shape[axis + 1:], identity, result.dtype), result],
                             axis=axis)
    return result


def cumulative_sum(x, /, *, axis=None, dtype=None, out=None, include_initial=False):
    """The running sums along `axis`, as ``numpy.cumulative_sum`` (the array API's): :func:`cumsum`, `axis` required
    past rank 1, a 0 first where `include_initial`."""
    return _cumulative(x, cumsum, axis, dtype, out, include_initial, 0, "cumulative_sum")


def cumulative_prod(x, /, *, axis=None, dtype=None, out=None, include_initial=False):
    """The running products along `axis`, as ``numpy.cumulative_prod``: :func:`cumprod`, `axis` required past rank 1,
    a 1 first where `include_initial`."""
    return _cumulative(x, cumprod, axis, dtype, out, include_initial, 1, "cumulative_prod")


def unstack(x, /, *, axis=0):
    """The slices of `x` along `axis`, as ``numpy.unstack``: a tuple of tensors."""
    x = _tensor(x)
    if not x.shape:
        raise ValueError("Input array must be at least 1-d.")
    moved = moveaxis(x, axis, 0)
    return tuple(moved[i] for i in range(moved.shape[0]))


def matvec(x1, x2, /):
    """The products of the matrices of `x1` (its last two axes) by the vectors of `x2` (its last), broadcast, as
    ``numpy.matvec``: ``(x1 @ x2[..., None])[..., 0]``."""
    a, b = _tensor(x1), _tensor(x2)
    if len(a.shape) < 2 or len(b.shape) < 1:
        raise ValueError("matvec: a stack of matrices and a stack of vectors")
    return matmul(a, expand_dims(b, -1))[..., 0]


def vecmat(x1, x2, /):
    """The products of the vectors of `x1`, conjugated, by the matrices of `x2`, broadcast, as ``numpy.vecmat``:
    ``(conj(x1)[..., None, :] @ x2)[..., 0, :]``."""
    a, b = _tensor(x1), _tensor(x2)
    if len(a.shape) < 1 or len(b.shape) < 2:
        raise ValueError("vecmat: a stack of vectors and a stack of matrices")
    return matmul(expand_dims(conj(a) if a.dtype.startswith("complex") else a, -2), b)[..., 0, :]


def _same_kind(source, target):
    """Whether NumPy casts `source` to `target` under the "same_kind" rule: within a kind or up the kinds (bool,
    integer, real, complex), never down."""
    order = {"bool": 0, "int64": 1, "float32": 2, "float64": 2, "complex64": 3, "complex128": 3}
    return order[source] <= order[target]


def _as_numpy_reads(x):
    """`x` as a Tensor of the dtype NumPy reads it in: a Python integer, or a nested list of them, int64 (where
    :func:`asarray` reads it as float64 by default)."""
    if isinstance(x, Tensor) or hasattr(x, "__array_interface__"):
        return _tensor(x)
    if isinstance(x, (list, tuple, _numbers.Number)):
        flat, _ = _nested(x) if isinstance(x, (list, tuple)) else ([x], ())
        if flat and _builtins.all(isinstance(v, _numbers.Integral) and not isinstance(v, bool) for v in flat):
            return asarray(x, "int64")
    return _tensor(x)


def ediff1d(ary, to_end=None, to_begin=None):
    """The differences of consecutive elements of `ary` flattened, as ``numpy.ediff1d``: `to_begin` before them and
    `to_end` after (flattened, cast to `ary`'s dtype, which they must reach under NumPy's "same_kind" rule)."""
    a = reshape(_tensor(ary), (-1,))
    pieces = []
    for extra, name in ((to_begin, "to_begin"), (to_end, "to_end")):
        if extra is None:
            pieces.append(None)
            continue
        extra = _as_numpy_reads(extra)
        if not _same_kind(extra.dtype, a.dtype):
            raise TypeError(f"dtype of `{name}` must be compatible with input `ary` under the `same_kind` rule.")
        pieces.append(asarray(reshape(extra, (-1,)), a.dtype))
    differences = a[1:] - a[:-1]
    parts = [p for p in (pieces[0], differences, pieces[1]) if p is not None]
    return concatenate(parts) if len(parts) > 1 else differences


def trim_zeros(filt, trim="fb", axis=None):
    """`filt` without its leading (``"f"``) and trailing (``"b"``) zeros, along `axis` (every one where None), as
    ``numpy.trim_zeros``: the slice of `filt` itself (a list stays a list)."""
    t = _tensor(filt) if not isinstance(filt, Tensor) else filt
    trim = trim.lower()
    if trim not in {"fb", "bf", "f", "b"}:
        raise ValueError(f"unexpected character(s) in `trim`: {trim!r}")
    rank = len(t.shape)
    axes = tuple(range(rank)) if axis is None else tuple(_normalized_axes(axis, rank, "trim_zeros"))
    if not axes:
        return filt
    found = argwhere(t)
    if found.shape[0] == 0:
        start = stop = [0] * rank
    else:
        start = [int(v) for v in min(found, axis=0).tolist()] if "f" in trim else [None] * rank
        stop = [int(v) + 1 for v in max(found, axis=0).tolist()] if "b" in trim else [None] * rank
    key = tuple(slice(start[k], stop[k]) if k in axes else slice(None) for k in range(rank))
    return filt[key[0]] if len(key) == 1 else filt[key]


def vander(x, N=None, increasing=False):
    """The Vandermonde matrix of the vector `x`, as ``numpy.vander``: its columns the powers of `x` (decreasing by
    default), each a running product (``x · x · …``, as NumPy's ``multiply.accumulate``); integers stay int64."""
    x = _tensor(x)
    if len(x.shape) != 1:
        raise ValueError("x must be a one-dimensional array or sequence.")
    n = x.shape[0] if N is None else _operator.index(N)
    dtype = _promoted([x.dtype, "int64"])  # NumPy's promote_types(x.dtype, int): float32 to float64
    x = asarray(x, dtype)
    columns = [ones((x.shape[0], _builtins.min(n, 1)), dtype)]
    if n > 1:
        columns.append(cumprod(broadcast_to(reshape(x, (-1, 1)), (x.shape[0], n - 1)), axis=1))
    v = concatenate(columns, axis=1) if n > 0 else zeros((x.shape[0], 0), dtype)
    return v if increasing else flip(v, 1)


def identity(n, dtype=None, *, like=None):
    """The ``n × n`` identity, as ``numpy.identity``: :func:`eye`, float64 by default."""
    if like is not None:
        raise TypeError("identity: `like` is NumPy's array-dispatch argument, not taken here")
    return eye(n, "float64" if dtype is None else _dtype_name(dtype))


def select(condlist, choicelist, default=0):
    """Each element from the first choice whose condition holds there, else `default`, as ``numpy.select``: the
    conditions bool, broadcast with the choices, the result in the dtype they promote to (a Python number weak)."""
    if len(condlist) != len(choicelist):
        raise ValueError("list of cases must be same length as list of conditions")
    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")
    choices = [c if _weak(c) else _tensor(c) for c in list(choicelist) + [default]]
    dtype = _result_dtype(*choices)
    conditions = broadcast_arrays(*[_tensor(c) for c in condlist])
    for i, condition in enumerate(conditions):
        if condition.dtype != "bool":
            raise TypeError(f"invalid entry {i} in condlist: should be boolean ndarray")
    choices = [asarray(c if not _weak(c) else [c], dtype) if _weak(c) else asarray(c, dtype) for c in choices]
    choices = [reshape(c, ()) if _weak(v) else c for c, v in zip(choices, list(choicelist) + [default])]
    choices = broadcast_arrays(*choices)
    target = conditions[0].shape if not choices[0].shape else broadcast_shapes(conditions[0].shape, choices[0].shape)
    result = broadcast_to(choices[-1], target)
    for choice, condition in zip(choices[-2::-1], conditions[::-1]):
        result = where(condition, choice, result)
    return asarray(result, dtype)


def _monotonicity(values):
    """NumPy's ``check_array_monotonic`` of the vector `values`: 1 increasing (or all equal), -1 decreasing, 0 neither;
    the first unequal pair decides the direction (a NaN unequal to all, and below none), each later pair must not go
    against it."""
    count = values.shape[0]
    if count < 2:
        return 1
    unequal = flatnonzero(values[1:] != values[:-1])
    if unequal.shape[0] == 0:
        return 1
    i = int(unequal[0]) + 1  # the first unequal pair: (i − 1, i)
    increasing = bool(values[i - 1] < values[i])
    earlier, later = values[i:-1], values[i + 1:]
    against = (earlier > later) if increasing else (earlier < later)
    if any(against):
        return 0
    return 1 if increasing else -1


def digitize(x, bins, right=False):
    """The index of the bin of each element of `x` among the monotonic `bins`, as ``numpy.digitize``: through
    :func:`searchsorted` (the side swapped by `right`), decreasing bins reversed and their indices counted from the
    end; int64."""
    x, bins = _tensor(x), _tensor(bins)
    if x.dtype.startswith("complex"):
        raise TypeError("x may not be complex")
    if len(bins.shape) != 1:
        raise ValueError("object too deep for desired array" if len(bins.shape) > 1 else
                         "object of too small depth for desired array")
    monotonic = _monotonicity(bins)
    if monotonic == 0:
        raise ValueError("bins must be monotonically increasing or decreasing")
    side = "left" if right else "right"
    if monotonic == -1:
        return bins.shape[0] - searchsorted(flip(bins, 0), x, side=side)
    return searchsorted(bins, x, side=side)


def _first_of_runs(aux, equal_nan):
    """NumPy's ``_unique1d`` mask over the sorted values `aux`: true at each value unlike the one before (and the
    first), every NaN after the first one false where `equal_nan` (all of a complex one's NaNs alike)."""
    count = aux.shape[0]
    if count == 0:
        return zeros(0, "bool")
    if len(aux.shape) > 1:  # rows, as NumPy's structured elements: unlike where any field is
        differs = any(aux[1:] != aux[:-1], axis=tuple(range(1, len(aux.shape))))
    else:
        differs = aux[1:] != aux[:-1]
    mask = concatenate([ones(1, "bool"), differs])
    if equal_nan and len(aux.shape) == 1 and aux.dtype not in ("bool", "int64") and bool(isnan(aux[-1])):
        first = int(count_nonzero(~isnan(aux)))  # the NaNs sorted last
        steps = asarray(arange(count), "int64")
        mask = where(steps == first, True, where(steps > first, False, mask))
    return mask


def _sorted_rows(rows):
    """The stable lexicographic order of the rows of a matrix (its columns as NumPy's fields): stable sorts from the
    last column to the first."""
    order = asarray(arange(rows.shape[0]), "int64")
    for column in range(rows.shape[1] - 1, -1, -1):
        order = order[argsort(rows[order, column])]
    return order


def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True,
           sorted=True):
    """The sorted distinct elements of `ar` flattened (or its distinct slices along `axis`, ordered as tuples), as
    ``numpy.unique``: with the index of each one's first occurrence, the index of each element's among them (of
    `ar`'s shape), and their counts, where asked (int64); NaNs as one where `equal_nan`. A zero's sign is the first
    one's met (the order is stable). `sorted` is NumPy's: the elements are sorted either way."""
    del sorted
    a = _tensor(ar)
    if axis is None or len(a.shape) == 1:
        if axis is not None:
            _axis(a, axis)
        flat = reshape(a, (-1,))
        if return_index or return_inverse:
            order = argsort(flat)
            aux = flat[order]
        else:  # the values sorted directly: the same order, the sort and argsort both stable
            order, aux = None, sort(flat)
        mask = _first_of_runs(aux, equal_nan)
        inverse_shape = a.shape
        reshape_values = None
    else:
        axis = _axis(a, axis)
        moved = moveaxis(a, axis, 0)
        rows = reshape(moved, (moved.shape[0], _math.prod(moved.shape[1:])))
        order = _sorted_rows(rows)
        aux = rows[order]
        mask = _first_of_runs(aux, False)
        inverse_shape = None

        def reshape_values(values):
            return moveaxis(reshape(values, (values.shape[0],) + moved.shape[1:]), 0, axis)
    values = aux[mask]
    result = (reshape_values(values) if reshape_values else values,)
    if return_index:
        result += (order[mask],)
    if return_inverse:
        among = cumsum(asarray(mask, "int64")) - 1
        inverse = among[argsort(order)]
        result += (reshape(inverse, inverse_shape) if inverse_shape is not None else inverse,)
    if return_counts:
        starts = concatenate([nonzero(mask)[0], asarray([mask.shape[0]], "int64")])
        result += (starts[1:] - starts[:-1],)
    return result[0] if len(result) == 1 else result


_UniqueAllResult = _collections.namedtuple("UniqueAllResult", ["values", "indices", "inverse_indices", "counts"])
_UniqueCountsResult = _collections.namedtuple("UniqueCountsResult", ["values", "counts"])
_UniqueInverseResult = _collections.namedtuple("UniqueInverseResult", ["values", "inverse_indices"])


def unique_values(x):
    """The distinct elements of `x`, as ``numpy.unique_values`` (the array API's): each NaN its own."""
    return unique(x, equal_nan=False)


def unique_counts(x):
    """The distinct elements of `x` and their counts, as ``numpy.unique_counts``: each NaN its own."""
    return _UniqueCountsResult(*unique(x, return_counts=True, equal_nan=False))


def unique_inverse(x):
    """The distinct elements of `x` and the index of each element's among them, as ``numpy.unique_inverse``."""
    return _UniqueInverseResult(*unique(x, return_inverse=True, equal_nan=False))


def unique_all(x):
    """The distinct elements of `x`, their first indices, each element's index among them and their counts, as
    ``numpy.unique_all``."""
    return _UniqueAllResult(*unique(x, return_index=True, return_inverse=True, return_counts=True, equal_nan=False))


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """The sorted elements of both `ar1` and `ar2`, as ``numpy.intersect1d``: with the indices of their first
    occurrences in each, where asked."""
    a, b = _tensor(ar1), _tensor(ar2)
    if not assume_unique:
        if return_indices:
            a, first_a = unique(a, return_index=True)
            b, first_b = unique(b, return_index=True)
        else:
            a, b = unique(a), unique(b)
    else:
        a, b = reshape(a, (-1,)), reshape(b, (-1,))
    aux = concatenate([a, b])
    order = argsort(aux)
    aux = aux[order]
    mask = aux[1:] == aux[:-1]
    both = aux[:-1][mask]
    if not return_indices:
        return both
    in_a = order[:-1][mask]
    in_b = order[1:][mask] - a.shape[0]
    if not assume_unique:
        in_a, in_b = first_a[in_a], first_b[in_b]
    return both, in_a, in_b


def union1d(ar1, ar2):
    """The sorted elements of `ar1` or `ar2`, as ``numpy.union1d``."""
    return unique(concatenate([reshape(_tensor(ar1), (-1,)), reshape(_tensor(ar2), (-1,))]))


def setdiff1d(ar1, ar2, assume_unique=False):
    """The sorted distinct elements of `ar1` not in `ar2`, as ``numpy.setdiff1d``."""
    if assume_unique:
        a = reshape(_tensor(ar1), (-1,))
    else:
        a, ar2 = unique(ar1), unique(ar2)
    return a[_isin(a, ar2, assume_unique=True, invert=True)]


def setxor1d(ar1, ar2, assume_unique=False):
    """The sorted elements of exactly one of `ar1` and `ar2`, as ``numpy.setxor1d``."""
    a, b = (_tensor(ar1), _tensor(ar2)) if assume_unique else (unique(ar1), unique(ar2))
    aux = concatenate([reshape(a, (-1,)), reshape(b, (-1,))])
    if aux.shape[0] == 0:
        return aux
    aux = sort(aux)
    flags = concatenate([ones(1, "bool"), aux[1:] != aux[:-1], ones(1, "bool")])
    return aux[flags[1:] & flags[:-1]]


def _isin(ar1, ar2, assume_unique=False, invert=False, kind=None):
    """NumPy's ``_isin`` of `ar1` and `ar2` flattened, by its sorting method (its table method answers the same)."""
    a, b = reshape(_tensor(ar1), (-1,)), reshape(_tensor(ar2), (-1,))
    if kind not in (None, "sort", "table"):
        raise ValueError(f"Invalid kind: '{kind}'. Please use None, 'sort' or 'table'.")
    integers = a.dtype in ("bool", "int64") and b.dtype in ("bool", "int64")
    if kind == "table" and not integers:
        raise ValueError("The 'table' method is only supported for boolean or integer arrays. Please select 'sort' or "
                         "None for kind.")
    if kind == "table" and b.shape[0] and int(max(asarray(b, "int64"))) - int(min(asarray(b, "int64"))) >= 2 ** 63:
        raise RuntimeError("You have specified kind='table', but the range of values in `ar2` or `ar1` exceed the "
                           "maximum integer of the datatype. Please set `kind` to None or 'sort'.")
    if integers and b.shape[0] == 0:
        return ones(a.shape, "bool") if invert else zeros(a.shape, "bool")
    # NumPy's sort of the two together: with duplicates in a, an element matches the next one. Of two dtypes, distinct
    # elements of a may meet in the promoted one (INT64_MIN and INT64_MIN + 1 in float64): each looked up in b below
    if assume_unique and a.dtype == b.dtype:
        joined = concatenate([a, b])
        order = argsort(joined)
        s = joined[order]
        same = (s[1:] != s[:-1]) if invert else (s[1:] == s[:-1])
        flags = concatenate([same, full(1, bool(invert), "bool")])
        return flags[argsort(order)][:a.shape[0]]
    # each element looked up among b's distinct sorted values: equal to one (never a NaN), as NumPy's sort finds
    b = unique(b)
    if b.shape[0] == 0:
        return ones(a.shape, "bool") if invert else zeros(a.shape, "bool")
    dtype = _promoted([a.dtype, b.dtype])
    a, b = asarray(a, dtype), asarray(b, dtype)
    at = minimum(searchsorted(b, a), b.shape[0] - 1)
    found = b[at] == a
    return ~found if invert else found


def isin(element, test_elements, assume_unique=False, invert=False, *, kind=None):
    """Whether each element of `element` is among `test_elements` (equal to one: never a NaN), as ``numpy.isin``:
    of `element`'s shape; `invert` for the opposite. `kind` is NumPy's choice of method, which changes nothing in the
    answer."""
    e = _tensor(element)
    return reshape(_isin(e, test_elements, assume_unique=assume_unique, invert=invert, kind=kind), e.shape)


__all__ += ["logspace", "geomspace", "cumulative_sum", "cumulative_prod", "unstack", "matvec", "vecmat", "ediff1d",
            "trim_zeros", "vander", "identity", "select", "digitize", "unique", "unique_values", "unique_counts",
            "unique_inverse", "unique_all", "intersect1d", "union1d", "setdiff1d", "setxor1d", "isin"]

def trapezoid(y, x=None, dx=1.0, axis=-1):
    """The integral of `y` along `axis` by the trapezoidal rule, as ``numpy.trapezoid``: over the sample points `x`
    (a vector along the axis, or of `y`'s shape) or at the spacing `dx`; ``Σ d · (y₁ + y₀) / 2``, by NumPy's steps.

    Examples:
        >>> import numerical
        >>> numerical.trapezoid([1.0, 2.0, 3.0])
        4.0
    """
    y = _tensor(y)
    rank = len(y.shape)
    if not -rank <= _operator.index(axis) < rank:
        raise IndexError("list assignment index out of range")  # NumPy's, from its slices
    axis = _axis(y, axis)
    if x is None:
        d = dx
    else:
        x = _tensor(x)
        if len(x.shape) == 1:
            d = reshape(diff(x), tuple(x.shape[0] - 1 if k == axis else 1 for k in range(rank)))
        else:
            d = diff(x, axis=axis)
    later = y[_along_axis(axis, rank, slice(1, None))]
    earlier = y[_along_axis(axis, rank, slice(None, -1))]
    result = sum(d * (later + earlier) / 2.0, axis=axis)
    return result[()] if isinstance(result, Tensor) and not result.shape else result


def _entry(t, i):
    """The element `i` of the vector `t` as a rank-0 tensor: of its dtype's full strength beside another tensor, as a
    NumPy scalar is (where a Python number would be weak)."""
    return reshape(t[i:i + 1] if i != -1 else t[-1:], ())


def gradient(f, *varargs, axis=None, edge_order=1):
    """The gradient of `f` by second-order central differences inside and first- or second-order (`edge_order`)
    one-sided ones at the ends, as ``numpy.gradient`` by its steps: one tensor per axis of `axis` (every one where
    None; the tensor alone for one axis), each spacing a number or the sample points along its axis (one for all
    axes, or one per axis); integers in float64."""
    f = _tensor(f)
    rank = len(f.shape)
    axes = tuple(range(rank)) if axis is None else tuple(_normalized_axes(axis, rank, "gradient"))
    count = len(varargs)
    if count == 0:
        spacings = [1.0] * len(axes)
    elif count == 1 and not shape(varargs[0]):
        spacings = list(varargs) * len(axes)
    elif count == len(axes):
        spacings = []
        for along, distances in zip(axes, varargs):
            if not shape(distances):
                spacings.append(distances)
                continue
            distances = _as_numpy_reads(distances)
            if len(distances.shape) != 1:
                raise ValueError("distances must be either scalars or 1d")
            if distances.shape[0] != f.shape[along]:
                raise ValueError("when 1d, distances must match the length of the corresponding dimension")
            if distances.dtype in ("int64", "bool"):
                distances = asarray(distances, "float64")
            steps = diff(distances)
            uniform = steps.shape[0] > 0 and bool(all(steps == steps[0]))
            spacings.append(_entry(steps, 0) if uniform else steps)  # NumPy's: a constant spacing as the scalar case
    else:
        raise TypeError("invalid number of arguments")
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")
    if f.dtype in ("int64", "bool"):
        f = asarray(f, "float64")
    results = []
    for along, step in zip(axes, spacings):
        if f.shape[along] < edge_order + 1:
            raise ValueError("Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) "
                             "elements are required.")
        uniform = not isinstance(step, Tensor) or not step.shape

        def part(key):
            return f[_along_axis(along, rank, key)]

        def along_axis(v):
            return reshape(v, tuple(-1 if k == along else 1 for k in range(rank)))
        if uniform:
            inner = (part(slice(2, None)) - part(slice(None, -2))) / (2.0 * step)
        else:
            dx1, dx2 = step[0:-1], step[1:]
            a = along_axis(-dx2 / (dx1 * (dx1 + dx2)))
            b = along_axis((dx2 - dx1) / (dx1 * dx2))
            c = along_axis(dx1 / (dx2 * (dx1 + dx2)))
            inner = a * part(slice(None, -2)) + b * part(slice(1, -1)) + c * part(slice(2, None))
        if edge_order == 1:
            first = (part(slice(1, 2)) - part(slice(0, 1))) / (step if uniform else _entry(step, 0))
            last = (part(slice(-1, None)) - part(slice(-2, -1))) / (step if uniform else _entry(step, -1))
        else:
            if uniform:
                a, b, c = -1.5 / step, 2.0 / step, -0.5 / step
            else:
                dx1, dx2 = _entry(step, 0), _entry(step, 1)
                a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = -dx1 / (dx2 * (dx1 + dx2))
            first = a * part(slice(0, 1)) + b * part(slice(1, 2)) + c * part(slice(2, 3))
            if uniform:
                a, b, c = 0.5 / step, -2.0 / step, 1.5 / step
            else:
                dx1, dx2 = _entry(step, -2), _entry(step, -1)
                a = dx2 / (dx1 * (dx1 + dx2))
                b = -(dx2 + dx1) / (dx1 * dx2)
                c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
            last = a * part(slice(-3, -2)) + b * part(slice(-2, -1)) + c * part(slice(-1, None))
        pieces = [asarray(piece, f.dtype) for piece in (first, inner, last)]
        if f.shape[along] == 2:  # no inside: the two ends alone, the second written last as NumPy's
            pieces = pieces[:1] + pieces[2:]
        results.append(concatenate(pieces, axis=along))
    return results[0] if len(axes) == 1 else tuple(results)


def diagflat(v, k=0):
    """The square matrix with `v` flattened on its `k`-th diagonal and zeros elsewhere, as ``numpy.diagflat``."""
    v = reshape(_as_numpy_reads(v), (-1,))
    k = _operator.index(k)
    n = v.shape[0] + _builtins.abs(k)
    if v.shape[0] == 0:
        return zeros((n, n), v.dtype)
    return diag(v, k) if n else zeros((0, 0), v.dtype)


def indices(dimensions, dtype=int, sparse=False):
    """The index grids of an array of shape `dimensions`, as ``numpy.indices``: one tensor of shape
    ``(N,) + dimensions`` (int64 by default), or a tuple of `N` broadcastable ones where `sparse`."""
    dimensions = tuple(_operator.index(d) for d in dimensions)
    name = _dtype_name(dtype)
    count = len(dimensions)
    grids = []
    for i, extent in enumerate(dimensions):
        steps = asarray(asarray(arange(extent), "int64"), name)
        grids.append(reshape(steps, (1,) * i + (extent,) + (1,) * (count - i - 1)))
    if sparse:
        return tuple(grids)
    if not grids:
        return zeros((0,), name)
    return stack([broadcast_to(g, dimensions) for g in grids])


def ix_(*args):
    """Open meshes from 1-D sequences for cross indexing, as ``numpy.ix_``: each its own axis, a bool one its true
    positions; an empty one int64."""
    out = []
    count = len(args)
    for k, new in enumerate(args):
        new = _as_numpy_reads(new)
        if new.shape == (0,) and not isinstance(args[k], Tensor):
            new = asarray(new, "int64")
        if len(new.shape) != 1:
            raise ValueError("Cross index must be 1 dimensional")
        if new.dtype == "bool":
            new = nonzero(new)[0]
        out.append(reshape(new, (1,) * k + (new.shape[0],) + (1,) * (count - k - 1)))
    return tuple(out)


def _index_modes(mode, count):
    modes = (mode,) * count if isinstance(mode, str) else tuple(mode)
    if len(modes) != count:
        raise ValueError(f"parameter mode does not have the correct number of entries ({len(modes)}, expected "
                         f"{count})")
    for m in modes:
        if m not in ("raise", "wrap", "clip"):
            raise ValueError(f"clipmode must be one of 'clip', 'raise', or 'wrap' (got {m!r})")
    return modes


def ravel_multi_index(multi_index, dims, mode="raise", order="C"):
    """The flat indices of the coordinates `multi_index` in an array of shape `dims`, as
    ``numpy.ravel_multi_index``: row-major (`order` "C") or column-major ("F"); a coordinate out of its extent
    refused ("raise"), wrapped ("wrap") or clipped ("clip"), per axis with a tuple of modes. int64."""
    dims = tuple(_operator.index(d) for d in dims)
    if len(multi_index) != len(dims):
        raise ValueError(f"parameter multi_index must be a sequence of length {len(dims)}")
    if order not in ("C", "F"):
        raise ValueError("only 'C' or 'F' order is permitted")
    modes = _index_modes(mode, len(dims))
    coordinates = broadcast_arrays(*[_as_numpy_reads(c) for c in multi_index])
    for c in coordinates:
        if c.dtype not in ("int64", "bool"):
            raise TypeError("only int indices permitted")
    strides, stride = [0] * len(dims), 1
    for k in (range(len(dims) - 1, -1, -1) if order == "C" else range(len(dims))):
        strides[k] = stride
        stride *= dims[k]
    flat = zeros(coordinates[0].shape if coordinates else (), "int64")
    for c, extent, step, m in zip(coordinates, dims, strides, modes):
        c = asarray(c, "int64")
        if m == "raise":
            if any(c < 0) or any(c >= extent):
                raise ValueError("invalid entry in coordinates array")
        elif m == "wrap":
            c = c % extent if extent else c
        else:
            c = clip(c, 0, _builtins.max(extent - 1, 0))
        flat = flat + c * step
    return flat[()] if not flat.shape else flat


def unravel_index(indices, shape, order="C"):
    """The coordinates of the flat `indices` in an array of `shape`, as ``numpy.unravel_index``: a tuple of int64
    tensors (numbers for a number), row-major (`order` "C") or column-major ("F")."""
    dims = (_operator.index(shape),) if isinstance(shape, _numbers.Integral) else tuple(
        _operator.index(d) for d in shape)
    if order not in ("C", "F"):
        raise ValueError("only 'C' or 'F' order is permitted")
    flat = _as_numpy_reads(indices)
    if flat.dtype not in ("int64", "bool"):
        raise TypeError(f"only int indices permitted")
    flat = asarray(flat, "int64")
    size_ = _math.prod(dims)
    if flat.size:
        low, high = int(min(flat)), int(max(flat))
        if low < 0 or high >= size_:
            line = reshape(flat, (-1,))
            first = int(flatnonzero((line < 0) | (line >= size_))[0])  # NumPy's: the first one out
            raise ValueError(f"index {int(line[first])} is out of bounds for array with size {size_}")
    coordinates = [None] * len(dims)
    remainder = flat
    for k in (range(len(dims) - 1, -1, -1) if order == "C" else range(len(dims))):
        coordinates[k] = remainder % dims[k] if dims[k] else remainder
        remainder = remainder // dims[k] if dims[k] else remainder
    return tuple(c[()] if not c.shape else c for c in coordinates)


def correlate(a, v, mode="valid"):
    """The cross-correlation ``c[k] = Σₙ a[n + k] · conj(v[n])`` of two vectors, as ``numpy.correlate``: the lags
    where they overlap wholly ("valid"), the length of the longer centred as NumPy's ("same"), or every lag ("full").
    Each lag's sum in order of `v`'s terms, on the threads, where NumPy's goes through its dot product (its BLAS's
    order): the last bits may differ."""
    a, v = _same_dtype(a, v)
    a, v = reshape(a, (-1,)) if not a.shape else a, reshape(v, (-1,)) if not v.shape else v
    if len(a.shape) != 1 or len(v.shape) != 1:
        raise ValueError("object too deep for desired array")
    if mode not in ("valid", "same", "full"):
        raise ValueError(f"mode must be one of 'valid', 'same', or 'full' (got {mode!r})")
    n, m = a.shape[0], v.shape[0]
    if n == 0 or m == 0:
        raise ValueError(f"{'a' if n == 0 else 'v'} cannot be empty")
    shorter = _builtins.min(n, m)
    if mode == "full":
        start, count = 0, n + m - 1
    elif mode == "valid":
        start, count = (m - 1, n - m + 1) if n >= m else (n - 1, m - n + 1)
    else:
        start, count = (shorter - 1 - shorter // 2, n) if n >= m else (shorter // 2, m)
    # the full range's lags run from −(m − 1): the lags start − (m − 1) … on, each from its own terms, natively
    if a.dtype == "bool":  # NumPy's dot of bools: whether any of a lag's products holds, counted here in int64
        return _native(_numerical._correlate, asarray(a, "int64"), asarray(v, "int64"), start - (m - 1), count) != 0
    return _native(_numerical._correlate, a, v, start - (m - 1), count)


def convolve(a, v, mode="full"):
    """The discrete linear convolution of two vectors, as ``numpy.convolve``: :func:`correlate` of the longer and
    the reversed shorter, every lag by default."""
    a, v = _same_dtype(a, v)
    a, v = atleast_1d(a), atleast_1d(v)
    if a.shape[0] == 0:
        raise ValueError("a cannot be empty")
    if v.shape[0] == 0:
        raise ValueError("v cannot be empty")
    if v.shape[0] > a.shape[0]:
        a, v = v, a
    reversed_ = flip(v, 0)
    return correlate(a, conj(reversed_) if reversed_.dtype.startswith("complex") else reversed_, mode)


__all__ += ["trapezoid", "gradient", "diagflat", "indices", "ix_", "ravel_multi_index", "unravel_index", "correlate",
            "convolve"]

def bincount(x, /, weights=None, minlength=0):
    """How often each value 0, 1, … occurs in the non-negative integers `x`, as ``numpy.bincount``: at least
    `minlength` of them (int64); or with `weights` (one per element, read as float64), each value's weights summed in
    `x`'s order, as NumPy's loop adds them. An empty `x` gives `minlength` zeros of int64, its weights unread, as
    NumPy's."""
    values = _as_numpy_reads(x)
    if values.shape == (0,) and not isinstance(x, Tensor) and not hasattr(x, "dtype"):
        values = asarray(values, "int64")  # NumPy's: an empty list read as integers
    if len(values.shape) != 1:
        raise ValueError("object too deep for desired array" if values.shape else
                         "object of too small depth for desired array")
    if values.dtype not in ("int64", "bool"):
        raise TypeError(f"Cannot cast array data from dtype('{values.dtype}') to dtype('int64') according to the rule "
                        "'safe'")
    values = asarray(values, "int64")
    minlength = _operator.index(minlength)
    if minlength < 0:
        raise ValueError("'minlength' must not be negative")
    if not values.shape[0]:  # NumPy's: no weights read, the counts' dtype
        return zeros((minlength,), "int64")
    if int(min(values)) < 0:
        raise ValueError("'list' argument must have no negative elements")
    length = _builtins.max(int(max(values)) + 1, minlength)
    if weights is not None:
        weights = _as_numpy_reads(weights)
        if weights.dtype.startswith("complex"):
            raise TypeError(f"Cannot cast array data from dtype('{weights.dtype}') to dtype('float64') according to "
                            "the rule 'safe'")
        weights = reshape(asarray(weights, "float64"), (-1,))
        if weights.shape[0] != values.shape[0]:
            raise ValueError("The weights and list don't have the same length.")
    return _native(_numerical._bincount, values, weights, length)


def _edge_value(value, dtype):
    """An edge NumPy reads from the data: a scalar of the data's dtype, of full strength (a rank-0 tensor here)."""
    return asarray(value, dtype)


def _histogram_edges(a, range_):
    """NumPy's ``_get_outer_edges``: `range_`'s ends (as given), or the data's least and greatest (of its dtype), a
    half widened each way where they are equal."""
    if range_ is not None:
        first, last = range_
        if first > last:
            raise ValueError("max must be larger than min in range parameter.")
        if not (_math.isfinite(first) and _math.isfinite(last)):
            raise ValueError(f"supplied range of [{first}, {last}] is not finite")
    elif a.shape[0] == 0:
        first, last = 0, 1
    else:
        first, last = _edge_value(min(a), a.dtype), _edge_value(max(a), a.dtype)
        if not (_math.isfinite(float(first)) and _math.isfinite(float(last))):
            raise ValueError(f"autodetected range of [{float(first)}, {float(last)}] is not finite")
    if bool(first == last):
        first, last = first - 0.5, last + 0.5
    return first, last


def _scalar_dtype(value):
    """NumPy's dtype of an edge: a tensor's own; a Python integer int64, a float float64 (``result_type`` of the
    scalar alone)."""
    if isinstance(value, Tensor):
        return value.dtype
    return "int64" if isinstance(value, _numbers.Integral) else "float64"


def _difference(b, a):
    """NumPy's ``_unsigned_subtract`` of two edges: ``b − a`` in their result dtype, an integer one through its
    unsigned type (a Python integer here, exact), a float one a strong rank-0 tensor of that dtype."""
    dtype = _promoted([_scalar_dtype(b), _scalar_dtype(a)])
    if dtype in ("int64", "bool"):
        return int(b) - int(a)
    return asarray(b, dtype) - asarray(a, dtype)


def _histogram_width(name, x, edges):
    """The bin width of NumPy's estimator `name` for the data `x` (its ``_hist_bin_*``), in its dtypes: the data's
    spread, deviation and percentiles of the data's dtype (a float32 one float32), NumPy's ``sqrt`` and ``log2`` of a
    count float64, Python's numbers weak beside them."""
    size = x.shape[0]
    dtype = "float64" if x.dtype in ("int64", "bool") else x.dtype

    def strong(value, of=dtype):  # a NumPy scalar of the dtype `of`: a rank-0 tensor
        return asarray(value, of)
    spread = _difference(_edge_value(max(x), x.dtype), _edge_value(min(x), x.dtype)) if size else 0
    spread = strong(spread, "float64") if not isinstance(spread, Tensor) else spread  # an unsigned integer's
    if name == "sqrt":
        return spread / strong(_math.sqrt(size), "float64")
    if name == "sturges":
        return spread / (strong(_math.log2(size), "float64") + 1.0)
    if name == "rice":
        return spread / (2.0 * size ** (1.0 / 3))
    if name == "scott":
        return (24.0 * _math.pi ** 0.5 / size) ** (1.0 / 3.0) * strong(std(x))
    if name == "fd":
        quartiles = percentile(x, [75, 25])
        return 2.0 * reshape(quartiles[0:1] - quartiles[1:2], ()) * size ** (-1.0 / 3.0)
    if name == "doane":
        if size > 2:
            sg1 = _math.sqrt(6.0 * (size - 2) / ((size + 1.0) * (size + 3)))
            sigma = strong(std(x))
            if float(sigma) > 0.0:
                g1 = strong(mean(power((x - strong(mean(x))) / sigma, 3)))
                return spread / (1.0 + strong(_math.log2(size), "float64") + log2(1.0 + abs(g1) / sg1))
        return 0.0
    if name == "stone":
        if size <= 1 or float(spread) == 0:
            return 0
        upper = _builtins.max(100, int(_math.sqrt(size)))

        def jhat(count):
            shares = histogram(x, bins=count, range=edges)[0] / size
            return (2 - (size + 1) * float(dot(shares, shares))) / (float(spread) / count)
        best = _builtins.min(_builtins.range(1, upper + 1), key=jhat)
        if best == upper:
            _warnings.warn("The number of bins estimated may be suboptimal.", RuntimeWarning, stacklevel=4)
        return spread / best
    fd, sturges, root = (_histogram_width(other, x, edges) for other in ("fd", "sturges", "sqrt"))  # "auto"
    corrected = fd if float(fd) >= float(root / 2) else root / 2  # Python's max and min: the first of equals
    return corrected if float(corrected) <= float(sturges) else sturges


_HISTOGRAM_ESTIMATORS = ("stone", "auto", "doane", "fd", "rice", "scott", "sqrt", "sturges")


def _histogram_bins(a, bins, range_, weights):
    """NumPy's ``_get_bin_edges``: the bin edges, and for equal bins their first and last edge and count."""
    count = None
    if isinstance(bins, str):
        if bins not in _HISTOGRAM_ESTIMATORS:
            raise ValueError(f"{bins!r} is not a valid estimator for `bins`")
        if weights is not None:
            raise TypeError("Automated estimation of the number of bins is not supported for weighted data")
        first, last = _histogram_edges(a, range_)
        if range_ is not None:
            keep = (a >= first) & (a <= last)
            if not all(keep):
                a = a[keep]
        if a.shape[0] == 0:
            count = 1
        else:
            width = _histogram_width(bins, a, (first, last))
            if float(width):
                if a.dtype == "int64" and float(width) < 1:
                    width = 1
                delta = _difference(last, first)  # a Python integer for integer edges: NumPy's unsigned one
                ratio = delta / width
                count = int(_math.ceil(float(ratio)))
            else:
                count = 1
    elif not shape(bins):
        try:
            count = _operator.index(bins)
        except TypeError as error:
            raise TypeError("`bins` must be an integer, a string, or an array") from error
        if count < 1:
            raise ValueError("`bins` must be positive, when an integer")
        first, last = _histogram_edges(a, range_)
    elif len(shape(bins)) == 1:
        edges = _as_numpy_reads(bins)
        if edges.shape[0] > 1 and any(edges[:-1] > edges[1:]):
            raise ValueError("`bins` must increase monotonically, when an array")
        return edges, None
    else:
        raise ValueError("`bins` must be 1d, when an array")
    dtype = _result_dtype(first if isinstance(first, Tensor) else first, last, a)
    dtype = "float64" if dtype in ("int64", "bool") else dtype
    edges = linspace(first, last, count + 1, endpoint=True, dtype=dtype)
    if edges.shape[0] > 1 and any(edges[:-1] >= edges[1:]):
        raise ValueError(f"Too many bins for data range. Cannot create {count} finite-sized bins.")
    return edges, (first, last, count)


def _histogram_operands(a, weights):
    a = _as_numpy_reads(a)
    if a.dtype == "bool":
        _warnings.warn("Converting input from bool to <class 'numpy.uint8'> for compatibility.", RuntimeWarning,
                       stacklevel=3)
        a = asarray(a, "int64")
    if weights is not None:
        weights = _as_numpy_reads(weights)
        if weights.shape != a.shape:
            raise ValueError("weights should have the same shape as a.")
        weights = reshape(weights, (-1,))
    return reshape(a, (-1,)), weights


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """The edges of :func:`histogram`'s bins, as ``numpy.histogram_bin_edges``."""
    a, weights = _histogram_operands(a, weights)
    return _histogram_bins(a, bins, range, weights)[0]


def histogram(a, bins=10, range=None, density=None, weights=None):
    """The histogram of `a` flattened, as ``numpy.histogram`` by its steps: `bins` equal ones (an integer, or one of
    NumPy's estimators by name: "auto", "fd", "doane", "rice", "scott", "sqrt", "sturges", "stone") over `range` (the
    data's least to greatest by default), or between the edges `bins`; the counts (int64), or the sums of `weights`
    (their dtype), or a density where `density`; each half-open bin but the last, closed. Returns ``(hist,
    bin_edges)``.

    Examples:
        >>> import numerical
        >>> counts, edges = numerical.histogram([1.0, 2.0, 2.0, 4.0], bins=3)
        >>> counts.tolist(), edges.tolist()
        ([1, 2, 1], [1.0, 2.0, 3.0, 4.0])
    """
    a, weights = _histogram_operands(a, weights)
    edges, uniform = _histogram_bins(a, bins, range, weights)
    total = "int64" if weights is None else weights.dtype
    block = 65536
    if uniform is not None:
        first, last, count = uniform
        n = zeros((count,), total)
        span = _difference(last, first)
        span = span if isinstance(span, Tensor) else asarray(span, "float64")  # NumPy's unsigned integer: a float
        for start in _builtins.range(0, a.shape[0], block):
            part = a[start:start + block]
            part_weights = None if weights is None else weights[start:start + block]
            keep = (part >= first) & (part <= last)
            if not all(keep):
                part = part[keep]
                if part_weights is not None:
                    part_weights = part_weights[keep]
            part = asarray(part, edges.dtype)
            offset = _difference(part, first) if not isinstance(first, Tensor) else part - first
            index = asarray(offset / span * count, "int64")
            index = where(index == count, index - 1, index)
            index = where(part < edges[index], index - 1, index)
            index = where((part >= edges[index + 1]) & (index != count - 1), index + 1, index)
            if total.startswith("complex"):
                parts = [asarray(bincount(index, weights=f(part_weights), minlength=count), _realpart(total))
                         for f in (real, imag)]
                n = n + _from_parts(*parts)
            elif part_weights is None:
                n = n + bincount(index, minlength=count)
            else:
                n = n + asarray(bincount(index, weights=part_weights, minlength=count), total)
    else:
        cumulative = zeros(edges.shape, total)
        for start in _builtins.range(0, a.shape[0], block):
            part = a[start:start + block]
            common = _promoted([part.dtype, edges.dtype])
            if weights is None:
                ordered = asarray(sort(part), common)
                found = _inclusive_search(ordered, asarray(edges, common))
                cumulative = cumulative + found
            else:
                order = argsort(part)
                ordered = asarray(part[order], common)
                sums = concatenate([zeros((1,), total), cumsum(weights[start:start + block][order])])
                cumulative = cumulative + sums[_inclusive_search(ordered, asarray(edges, common))]
        n = cumulative[1:] - cumulative[:-1]
    if density:
        widths = asarray(edges[1:] - edges[:-1], "float64")
        return n / widths / sum(n), edges
    return n, edges


def _realpart(dtype):
    return "float32" if dtype == "complex64" else "float64"


def _inclusive_search(ordered, edges):
    """NumPy's ``_search_sorted_inclusive``: where each edge goes in `ordered`, the last one past its equals."""
    return concatenate([searchsorted(ordered, edges[:-1], side="left"), searchsorted(ordered, edges[-1:],
                                                                                    side="right")])


__all__ += ["bincount", "histogram", "histogram_bin_edges"]

_numpy_interp_fused = None  # whether NumPy's interp fuses its line's multiply-add: measured once NumPy is imported
_numpy_arange_fused = None  # whether NumPy's arange fuses its elements' multiply-add: measured once NumPy is imported


def _contracting_wheels():
    """Whether NumPy's wheels for this platform fuse a multiply-add their C writes apart (macOS arm64's clang
    contracts them, the others' compilers do not)."""
    return _sys.platform == "darwin" and _platform_machine() in ("arm64", "aarch64")


def _arange_fuses():
    """Whether numpy.arange's fill fuses ``start + i · delta``: as NumPy computes it beside this process, else as its
    wheels for this platform do."""
    global _numpy_arange_fused
    if _numpy_arange_fused is None:
        np = _sys.modules.get("numpy")
        if np is None:
            return _contracting_wheels()
        # an element whose product and sum, rounded apart, differ from their fused rounding (by its last bit)
        start, step = -3.5, 0.3
        rounded = start + 27 * ((start + step) - start)
        _numpy_arange_fused = float(np.arange(start, 7.2, step)[27]) != rounded
    return _numpy_arange_fused


def _interp_fuses():
    """Whether numpy.interp's loop fuses ``slope · (x − x₀) + y₀``: as NumPy computes it beside this process (its
    macOS arm64 wheels' clang contracts it, the others do not), else as those wheels do."""
    global _numpy_interp_fused
    if _numpy_interp_fused is None:
        np = _sys.modules.get("numpy")
        if np is None:
            return _contracting_wheels()
        # a point whose line through (0, 0.1) and (0.3, 1.0) differs fused and rounded (by its last bit)
        x, xp, fp = 24 / 997.0, [0.0, 0.3], [0.1, 1.0]
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        rounded = slope * (x - xp[0]) + fp[0]
        _numpy_interp_fused = float(np.interp(x, xp, fp)) != rounded
    return _numpy_interp_fused


def _platform_machine():
    import platform
    return platform.machine().lower()


def interp(x, xp, fp, left=None, right=None, period=None):
    """The piecewise-linear interpolant of the points `(xp, fp)` (`xp` increasing) at `x`, as ``numpy.interp`` by its
    loop: `left` below `xp[0]` and `right` above `xp[-1]` (`fp`'s ends by default), a NaN `x` NaN; each line through
    its first point, or its second where that is NaN; with `period`, `x` and `xp` taken modulo it. float64, or
    complex128 for a complex `fp`.

    Examples:
        >>> import numerical
        >>> numerical.interp([0.5, 2.5], [0.0, 1.0, 2.0], [0.0, 10.0, 20.0]).tolist()
        [5.0, 20.0]
    """
    values = _as_numpy_reads(fp)
    dtype = "complex128" if values.dtype.startswith("complex") else "float64"
    points = asarray(_tensor(x), "float64")
    knots = asarray(_tensor(xp), "float64")
    values = asarray(values, dtype)
    if period is not None:
        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = _builtins.abs(period)
        left = right = None
        if len(knots.shape) != 1 or len(values.shape) != 1:
            raise ValueError("Data points must be 1-D sequences")
        if knots.shape[0] != values.shape[0]:
            raise ValueError("fp and xp are not of the same length")
        points = points % period
        knots = knots % period
        order = argsort(knots)
        knots, values = knots[order], values[order]
        knots = concatenate([knots[-1:] - period, knots, knots[0:1] + period])
        values = concatenate([values[-1:], values, values[0:1]])
    if len(knots.shape) != 1 or len(values.shape) != 1:
        raise ValueError("object too deep for desired array" if len(knots.shape) > 1 or len(values.shape) > 1 else
                         "object of too small depth for desired array")
    if knots.shape[0] != values.shape[0]:
        raise ValueError("fp and xp are not of the same length.")
    if knots.shape[0] == 0:
        raise ValueError("array of sample points is empty")
    fills = [default if v is None else asarray([v], dtype) for v, default in ((left, values[0:1]),
                                                                              (right, values[-1:]))]
    result = _native(_numerical._interp, points, knots, values, fills[0], fills[1], _interp_fuses())
    return result[()] if not result.shape else result


def _window_points(M):
    """NumPy's window length as a float64 (``numpy.array([0.0, M])[1]``) and its points ``1 − M, 3 − M, …, M − 1``."""
    M = float(M)
    return M, arange(1 - M, M, 2)


def hamming(M):
    """The Hamming window of `M` points, as ``numpy.hamming``: ``0.54 + 0.46 cos(π n / (M − 1))``."""
    M, n = _window_points(M)
    if M < 1:
        return zeros((0,))
    if M == 1:
        return ones((1,))
    return 0.54 + 0.46 * cos(_math.pi * n / (M - 1))


def hanning(M):
    """The Hann window of `M` points, as ``numpy.hanning``: ``0.5 + 0.5 cos(π n / (M − 1))``."""
    M, n = _window_points(M)
    if M < 1:
        return zeros((0,))
    if M == 1:
        return ones((1,))
    return 0.5 + 0.5 * cos(_math.pi * n / (M - 1))


def blackman(M):
    """The Blackman window of `M` points, as ``numpy.blackman``."""
    M, n = _window_points(M)
    if M < 1:
        return zeros((0,))
    if M == 1:
        return ones((1,))
    return 0.42 + 0.5 * cos(_math.pi * n / (M - 1)) + 0.08 * cos(2.0 * _math.pi * n / (M - 1))


def bartlett(M):
    """The Bartlett (triangular) window of `M` points, as ``numpy.bartlett``."""
    M, n = _window_points(M)
    if M < 1:
        return zeros((0,))
    if M == 1:
        return ones((1,))
    return where(n <= 0, 1 + n / (M - 1), 1 - n / (M - 1))


_I0_A = (-4.4153416464793395e-18, 3.3307945188222384e-17, -2.431279846547955e-16, 1.715391285555133e-15,
         -1.1685332877993451e-14, 7.676185498604936e-14, -4.856446783111929e-13, 2.95505266312964e-12,
         -1.726826291441556e-11, 9.675809035373237e-11, -5.189795601635263e-10, 2.6598237246823866e-09,
         -1.300025009986248e-08, 6.046995022541919e-08, -2.670793853940612e-07, 1.1173875391201037e-06,
         -4.4167383584587505e-06, 1.6448448070728896e-05, -5.754195010082104e-05, 0.00018850288509584165,
         -0.0005763755745385824, 0.0016394756169413357, -0.004324309995050576, 0.010546460394594998,
         -0.02373741480589947, 0.04930528423967071, -0.09490109704804764, 0.17162090152220877, -0.3046826723431984,
         0.6767952744094761)
_I0_B = (-7.233180487874754e-18, -4.830504485944182e-18, 4.46562142029676e-17, 3.461222867697461e-17,
         -2.8276239805165836e-16, -3.425485619677219e-16, 1.7725601330565263e-15, 3.8116806693526224e-15,
         -9.554846698828307e-15, -4.150569347287222e-14, 1.54008621752141e-14, 3.8527783827421426e-13,
         7.180124451383666e-13, -1.7941785315068062e-12, -1.3215811840447713e-11, -3.1499165279632416e-11,
         1.1889147107846439e-11, 4.94060238822497e-10, 3.3962320257083865e-09, 2.266668990498178e-08,
         2.0489185894690638e-07, 2.8913705208347567e-06, 6.889758346916825e-05, 0.0033691164782556943,
         0.8044904110141088)


def _chebyshev(x, coefficients):
    """NumPy's ``_chbevl``: the Chebyshev series of `coefficients` at `x` by Clenshaw's recurrence."""
    b0, b1, b2 = coefficients[0], 0.0, 0.0
    for c in coefficients[1:]:
        b2, b1 = b1, b0
        b0 = x * b1 - b2 + c
    return 0.5 * (b0 - b2)


def i0(x):
    """The modified Bessel function of the first kind, order 0, of every element, as ``numpy.i0`` by its steps
    (Cephes' Chebyshev series, Clenshaw's, below and above 8). Real dtypes, an integer in float64."""
    x = _tensor(x)
    if x.dtype.startswith("complex"):
        raise TypeError("i0 not supported for complex values")
    if x.dtype in ("int64", "bool"):
        x = asarray(x, "float64")
    x = abs(x)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)  # each branch computed everywhere, kept where it applies
        small = exp(x) * _chebyshev(x / 2.0 - 2, _I0_A)
        large = exp(x) * _chebyshev(32.0 / x - 2.0, _I0_B) / sqrt(x)
    result = where(x <= 8.0, small, large)
    return result[()] if not result.shape else result


def kaiser(M, beta):
    """The Kaiser window of `M` points and shape `beta`, as ``numpy.kaiser``: ``i0(β √(1 − ((n − α)/α)²)) / i0(β)``."""
    M, beta = float(M), float(beta)
    if M == 1:
        return ones((1,))
    n = arange(0, M)
    alpha = (M - 1) / 2.0
    return i0(beta * sqrt(1 - ((n - alpha) / alpha) ** 2.0)) / i0(beta)


def _coefficients(p):
    """A polynomial's coefficients as NumPy reads them, at least of rank 1 (integers int64)."""
    return atleast_1d(_as_numpy_reads(p))


def polyval(p, x):
    """The polynomial of coefficients `p` (highest power first) at `x`, as ``numpy.polyval``: Horner's
    ``y · x + pᵢ`` from a zero of `x`'s dtype."""
    p = _as_numpy_reads(p)
    x = _as_numpy_reads(x)
    y = zeros_like(x)
    for k in range(p.shape[0]):
        y = y * x + reshape(p[k:k + 1], ())
    return y[()] if not y.shape else y


def polyder(p, m=1):
    """The `m`-th derivative of the polynomial `p`, as ``numpy.polyder``."""
    m = int(m)
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")
    p = _as_numpy_reads(p)
    for _ in range(m):
        count = p.shape[0] - 1
        p = p[:-1] * asarray(arange(count, 0, -1), "int64")
    return p


def polyint(p, m=1, k=None):
    """The `m`-th antiderivative of the polynomial `p`, its constants `k` (one for all, or one per order), as
    ``numpy.polyint``."""
    m = int(m)
    if m < 0:
        raise ValueError("Order of integral must be positive (see polyder)")
    constants = zeros((m,)) if k is None else atleast_1d(_as_numpy_reads(k))
    if constants.shape[0] == 1 and m > 1:
        constants = reshape(constants[0:1], ()) * ones((m,))
    if constants.shape[0] < m:
        raise ValueError("k must be a scalar or a rank-1 array of length 1 or >m.")
    p = _as_numpy_reads(p)
    for order in range(m):
        p = concatenate([p / asarray(arange(p.shape[0], 0, -1), "int64"), asarray(constants[order:order + 1],
                                                                                    _promoted([p.dtype, "float64"]))])
    return p


def _padded_pair(a1, a2):
    a1, a2 = _coefficients(a1), _coefficients(a2)
    gap = a2.shape[0] - a1.shape[0]
    if gap > 0:
        a1 = concatenate([zeros((gap,), a1.dtype), a1])
    elif gap < 0:
        a2 = concatenate([zeros((-gap,), a2.dtype), a2])
    return a1, a2


def polyadd(a1, a2):
    """The sum of two polynomials, as ``numpy.polyadd``."""
    a1, a2 = _padded_pair(a1, a2)
    return a1 + a2


def polysub(a1, a2):
    """The difference of two polynomials, as ``numpy.polysub``."""
    a1, a2 = _padded_pair(a1, a2)
    return a1 - a2


def polymul(a1, a2):
    """The product of two polynomials, as ``numpy.polymul``: their coefficients' convolution."""
    return convolve(_trimmed_coefficients(a1), _trimmed_coefficients(a2))


def _trimmed_coefficients(p):
    """A polynomial's coefficients as ``numpy.poly1d`` holds them: rank 1, its leading zeros left out (one zero for
    the zero polynomial)."""
    c = _coefficients(p)
    if len(c.shape) > 1:
        raise ValueError("Polynomial must be 1d only.")
    found = flatnonzero(c)
    return c[int(found[0]):] if found.shape[0] else zeros((1,), c.dtype)


def polydiv(u, v):
    """The quotient and remainder of two polynomials, as ``numpy.polydiv``: long division in floats, the remainder's
    leading zeros (to 1e-14 relative) dropped."""
    u = _coefficients(u) + 0.0
    v = _coefficients(v) + 0.0
    dtype = _promoted([u.dtype, v.dtype])
    if not u.shape[0] or not v.shape[0]:  # NumPy's: its leading coefficients read first
        raise IndexError("index 0 is out of bounds for axis 0 with size 0")
    m, n = u.shape[0] - 1, v.shape[0] - 1
    scale = 1.0 / reshape(asarray(v[0:1], dtype), ())
    quotient = []
    r = asarray(u, dtype)
    for k in range(m - n + 1):
        d = scale * reshape(r[k:k + 1], ())
        quotient.append(reshape(d, (1,)))
        r = concatenate([r[:k], r[k:k + n + 1] - d * v, r[k + n + 1:]])
    q = concatenate(quotient) if quotient else zeros((1,), dtype)
    while r.shape[0] > 1 and allclose(r[0:1], 0, rtol=1e-14):
        r = r[1:]
    return q, r


def roots(p):
    """The roots of the polynomial `p`, as ``numpy.roots``: the eigenvalues of its companion matrix (its leading and
    trailing zeros set apart), complex as :func:`eigvals` answers unless `p` is real and so are they all."""
    p = _coefficients(p)
    if len(p.shape) != 1:
        raise ValueError("Input must be a rank-1 array.")
    found = flatnonzero(p)
    if found.shape[0] == 0:
        return zeros((0,))
    trailing = p.shape[0] - int(found[-1]) - 1
    p = p[int(found[0]):int(found[-1]) + 1]
    if p.dtype in ("int64", "bool"):
        p = asarray(p, "float64")
    count = p.shape[0]
    if count > 1:
        companion = diag(ones((count - 2,), p.dtype), -1) if count > 2 else zeros((1, 1), p.dtype)
        first = reshape(-p[1:] / reshape(p[0:1], ()), (1, count - 1))
        companion = concatenate([first, companion[1:]]) if count > 2 else first
        values = eigvals(companion)
        if not p.dtype.startswith("complex") and not any(imag(values) != 0):  # NumPy's: real ones if they all are
            values = real(values)
    else:
        values = zeros((0,), "float64")
    return concatenate([values, zeros((trailing,), values.dtype)])


def _rank_warning():
    np = _sys.modules.get("numpy")
    return np.exceptions.RankWarning if np is not None and hasattr(np, "exceptions") else _RankWarning


class _RankWarning(UserWarning):
    """A least-squares fit of deficient rank, as ``numpy.exceptions.RankWarning`` (which it is where NumPy is
    loaded)."""


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """The least-squares polynomial fit of degree `deg` to the points `(x, y)`, as ``numpy.polyfit`` by its steps:
    the Vandermonde matrix's columns scaled to unit norm, :func:`lstsq`, the scales undone; with `full`, the residues,
    rank, singular values and `rcond` too; with `cov`, the coefficients' covariance (unscaled, or scaled by the
    residue per degree of freedom)."""
    order = int(deg) + 1
    x = _as_numpy_reads(x) + 0.0
    y = _as_numpy_reads(y) + 0.0
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if len(x.shape) != 1:
        raise TypeError("expected 1D vector for x")
    if x.shape[0] == 0:
        raise TypeError("expected non-empty vector for x")
    if len(y.shape) < 1 or len(y.shape) > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")
    if rcond is None:
        rcond = x.shape[0] * finfo(x.dtype).eps
    lhs = vander(x, order)
    rhs = y
    if w is not None:
        w = _as_numpy_reads(w) + 0.0
        if len(w.shape) != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        lhs = lhs * reshape(w, (-1, 1))
        rhs = rhs * (reshape(w, (-1, 1)) if len(rhs.shape) == 2 else w)
    scale = sqrt(sum(lhs * lhs, axis=0))
    lhs = lhs / scale
    c, residues, rank, singular = lstsq(lhs, rhs, rcond)
    c = transpose(transpose(c) / scale)
    if rank != order and not full:
        _warnings.warn("Polyfit may be poorly conditioned", _rank_warning(), stacklevel=2)
    if full:
        return c, residues, rank, singular, rcond
    if cov:
        base = inv(dot(transpose(lhs), lhs)) / outer(scale, scale)
        if cov == "unscaled":
            factor = 1
        else:
            if x.shape[0] <= order:
                raise ValueError("the number of data points must exceed order to scale the covariance matrix")
            factor = residues / (x.shape[0] - order)
        if len(y.shape) == 1:
            return c, base * factor
        return c, reshape(base, base.shape + (1,)) * factor
    return c


def histogramdd(sample, bins=10, range=None, density=None, weights=None):
    """The multidimensional histogram of `sample` (`N` points of `D` coordinates, or a sequence of `D` coordinate
    vectors), as ``numpy.histogramdd`` by its steps: per coordinate a count of equal bins over `range` or given
    edges; each bin half-open but the last; counts (or weights' sums) in float64, or a density."""
    if isinstance(sample, Tensor) and len(sample.shape) == 2:
        points = sample
    else:
        points = _as_numpy_reads(sample)
        if len(points.shape) != 2 or not isinstance(sample, (Tensor,)) and not hasattr(sample, "shape"):
            points = transpose(atleast_2d(points))
    count, dims = points.shape
    if weights is not None:
        weights = _as_numpy_reads(weights)
    try:
        if len(bins) != dims:
            raise ValueError("The dimension of bins must be equal to the dimension of the sample x.")
    except TypeError:
        bins = dims * [bins]
    if range is None:
        range = (None,) * dims
    elif len(range) != dims:
        raise ValueError("range argument must have one entry per dimension")
    edges, extents = [], []
    for i in _builtins.range(dims):
        column = points[:, i]
        if not shape(bins[i]):
            if bins[i] < 1:
                raise ValueError(f"`bins[{i}]` must be positive, when an integer")
            low, high = _histogram_edges(column, range[i])
            try:
                n = _operator.index(bins[i])
            except TypeError as error:
                raise TypeError(f"`bins[{i}]` must be an integer, when a scalar") from error
            edge = linspace(low, high, n + 1)
        elif len(shape(bins[i])) == 1:
            edge = _as_numpy_reads(bins[i])
            if edge.shape[0] > 1 and any(edge[:-1] > edge[1:]):
                raise ValueError(f"`bins[{i}]` must be monotonically increasing, when an array")
        else:
            raise ValueError(f"`bins[{i}]` must be a scalar or 1d array")
        edges.append(edge)
        extents.append(edge.shape[0] + 1)
    places = []
    for i in _builtins.range(dims):
        common = _promoted([edges[i].dtype, points.dtype])
        column = asarray(points[:, i], common)
        found = searchsorted(asarray(edges[i], common), column, side="right")
        found = where(column == asarray(edges[i][-1:], common), found - 1, found)  # the last edge in the last bin
        places.append(found)
    flat = ravel_multi_index(places, extents) if dims else zeros((0,), "int64")
    hist = bincount(flat, weights, minlength=_math.prod(extents))
    if hist.dtype.startswith("complex"):
        raise TypeError("Cannot cast array data from dtype('complex128') to dtype('float64') according to the rule "
                        "'safe'")
    hist = asarray(reshape(hist, tuple(extents)), "float64")
    hist = hist[(slice(1, -1),) * dims]
    if density:
        total = sum(hist)
        for i in _builtins.range(dims):
            widths = reshape(asarray(edges[i][1:] - edges[i][:-1], "float64"),
                             tuple(-1 if k == i else 1 for k in _builtins.range(dims)))
            hist = hist / widths
        hist = hist / total
    return hist, edges


def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
    """The two-dimensional histogram of the points `(x, y)`, as ``numpy.histogram2d``: :func:`histogramdd` of the two
    coordinates; returns ``(hist, xedges, yedges)``."""
    x, y = _as_numpy_reads(x), _as_numpy_reads(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")
    try:
        count = len(bins)
    except TypeError:
        count = 1
    if count not in (1, 2):
        bins = [_as_numpy_reads(bins), _as_numpy_reads(bins)]
    hist, edges = histogramdd(stack([x, y], axis=1) if x.dtype == y.dtype else
                              stack([asarray(x, _promoted([x.dtype, y.dtype])), asarray(y, _promoted([x.dtype,
                                                                                                       y.dtype]))],
                                    axis=1), bins, range, density, weights)
    return hist, edges[0], edges[1]


def compress(condition, a, axis=None, out=None):
    """The slices of `a` along `axis` (of `a` flattened where None) where `condition` holds, as ``numpy.compress``: a
    shorter `condition` stops early; a true entry past the axis refused."""
    _no_out(out, "compress")
    a = _tensor(a)
    if axis is None:
        a, axis = reshape(a, (-1,)), 0
    axis = _axis(a, axis)
    chosen = nonzero(reshape(asarray(_tensor(condition), "bool"), (-1,)))[0]
    return take(a, chosen, axis=axis)  # NumPy's: no bound checked without a slice of the other axes


def extract(condition, arr):
    """The elements of `arr` flattened where `condition` (flattened) holds, as ``numpy.extract``."""
    return take(reshape(_tensor(arr), (-1,)), nonzero(reshape(asarray(_tensor(condition), "bool"), (-1,)))[0])


def choose(a, choices, out=None, mode="raise"):
    """Each element from the choice its index in `a` names, all broadcast, as ``numpy.choose``: an index out of range
    refused ("raise"), wrapped ("wrap") or clipped ("clip")."""
    _no_out(out, "choose")
    index = _as_numpy_reads(a)
    if index.dtype not in ("int64", "bool"):
        raise TypeError(f"Cannot cast array data from dtype('{index.dtype}') to dtype('int64') according to the rule "
                        "'safe'")
    if mode not in ("raise", "wrap", "clip"):
        raise ValueError(f"clipmode must be one of 'clip', 'raise', or 'wrap' (got {mode!r})")
    options = [_as_numpy_reads(c) if not _weak(c) else c for c in choices]
    if not options:
        raise ValueError("0-length sequence.")
    dtype = _result_dtype(*options)
    options = [asarray(c if not _weak(c) else [c], dtype) for c in options]
    options = [reshape(c, ()) if _weak(v) else c for c, v in zip(options, choices)]
    count = len(options)
    index = asarray(index, "int64")
    if mode == "raise":
        if index.size and (int(min(index)) < 0 or int(max(index)) >= count):
            raise ValueError("invalid entry in choice array")
    elif mode == "wrap":
        index = index % count
    else:
        index = clip(index, 0, count - 1)
    *options, index = broadcast_arrays(*options, index)
    stacked = stack(options)
    return take_along_axis(stacked, reshape(index, (1,) + index.shape), 0)[0]


__all__ += ["interp", "hamming", "hanning", "blackman", "bartlett", "kaiser", "i0", "polyval", "polyder", "polyint",
            "polyadd", "polysub", "polymul", "polydiv", "roots", "polyfit", "histogramdd", "histogram2d", "compress",
            "extract", "choose"]

pi = _math.pi
e = _math.e
inf = _math.inf
nan = _math.nan
newaxis = None

# The array API's dtype names, as the strings a Tensor's dtype is: numerical.float64 == t.dtype. Served by the module's
# __getattr__ so that `bool` and the others never shadow Python's names inside this module.
_DTYPE_NAMES = ("bool", "int64", "float32", "float64", "complex64", "complex128")


def __getattr__(name):
    if name in _DTYPE_NAMES:
        return name
    raise AttributeError(f"module 'numerical' has no attribute {name!r}")


class finfo:
    """The limits of a floating-point dtype, as ``numpy.finfo``: of a complex one, its parts' (float32 for
    complex64, float64 for complex128)."""

    _LIMITS = {"float32": (32, 23, -126, 128, 6), "float64": (64, 52, -1022, 1024, 15)}

    def __init__(self, dtype):
        name = _dtype_name(dtype.dtype if isinstance(dtype, Tensor) else dtype)
        name = {"complex64": "float32", "complex128": "float64"}.get(name, name)
        if name not in self._LIMITS:
            raise ValueError(f"data type {name!r} not inexact")
        bits, mantissa, minexp, maxexp, precision = self._LIMITS[name]
        cast = (lambda x: _array.array("f", [x])[0]) if bits == 32 else float  # the value the dtype holds
        self.dtype, self.bits, self.nmant, self.minexp, self.maxexp, self.precision = name, bits, mantissa, minexp, \
            maxexp, precision
        self.eps = cast(2.0 ** -mantissa)
        self.epsneg = cast(2.0 ** (-mantissa - 1))
        self.max = cast((2.0 - 2.0 ** -mantissa) * 2.0 ** (maxexp - 1))
        self.min = -self.max
        self.tiny = self.smallest_normal = cast(2.0 ** minexp)
        self.smallest_subnormal = cast(2.0 ** (minexp - mantissa))
        self.resolution = cast(10.0 ** -precision)

    def __repr__(self):
        return f"finfo(resolution={self.resolution}, min={self.min}, max={self.max}, dtype={self.dtype})"


class iinfo:
    """The limits of an integer dtype, as ``numpy.iinfo``: int64, NumericAL's one."""

    def __init__(self, int_type):
        name = _dtype_name(int_type.dtype if isinstance(int_type, Tensor) else int_type)
        if name != "int64":
            raise ValueError(f"Invalid integer data type {name!r}.")
        self.dtype, self.bits, self.min, self.max = name, 64, -2 ** 63, 2 ** 63 - 1

    def __repr__(self):
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})"


__all__ += ["pi", "e", "inf", "nan", "newaxis", "finfo", "iinfo"]

__all__ += ["abs", "negative", "positive", "square", "divide", "true_divide", "floor_divide", "remainder", "mod",
            "divmod", "zeros_like", "ones_like", "empty_like",
            "full_like", "copy", "ravel", "expand_dims", "atleast_1d", "atleast_2d", "atleast_3d", "flip", "flipud",
            "fliplr", "swapaxes", "moveaxis", "column_stack", "maximum", "minimum", "clip", "isclose", "allclose", "dot",
            "inner", "vdot", "ptp", "var", "std", "floor", "ceil", "trunc", "rint", "sign", "arcsin", "arccos", "arctan",
            "sinh", "cosh", "arcsinh", "arccosh", "arctanh", "log2", "log10", "log1p", "expm1", "cbrt", "deg2rad",
            "rad2deg", "radians", "degrees", "round", "around", "prod", "cumsum", "cumprod", "sort", "diff", "argmax",
            "argmin", "argsort", "nonzero", "flatnonzero", "argwhere", "count_nonzero", "searchsorted", "hypot",
            "arctan2", "atan2", "copysign", "nextafter", "fmod", "logaddexp", "logaddexp2", "fmax", "fmin", "heaviside",
            "float_power", "angle", "unwrap", "nan_to_num", "broadcast_shapes", "broadcast_to", "broadcast_arrays", "tile",
            "repeat", "take", "take_along_axis", "roll", "rot90", "array_split", "split", "hsplit", "vsplit", "dsplit",
            "dstack", "append", "insert", "delete", "block", "nansum", "nanprod", "nancumsum", "nancumprod", "nanmax",
            "nanmin", "nanargmax", "nanargmin", "nanmean", "nanvar", "nanstd", "median", "nanmedian", "quantile", "percentile", "nanquantile",
            "nanpercentile", "average", "cov", "corrcoef", "pad"]

from numerical import fft  # noqa: E402  (the module numpy.fft's names live in, as numpy.fft is NumPy's)
from numerical import linalg  # noqa: E402  (numpy.linalg's names, the root's functions)


def _register_threadpoolctl(threadpoolctl):
    """NumericAL's thread pool as a threadpoolctl library (``user_api="blas"``), so that ``threadpool_limits`` (joblib,
    scikit-learn) caps it and ``threadpool_info`` lists it."""

    class _Controller(threadpoolctl.LibController):
        user_api = "blas"
        internal_api = "numerical"
        filename_prefixes = ("_numerical",)

        def get_num_threads(self):
            return get_num_threads()

        def set_num_threads(self, num_threads):
            set_num_threads(num_threads)

        def get_version(self):
            return __version__

    threadpoolctl.register(_Controller)


class _ThreadpoolctlHook:
    """A finder on ``sys.meta_path`` that registers NumericAL with threadpoolctl once something imports it: importing it
    here would cost an import of NumericAL twice over (9 ms against 4 to 5). It finds nothing itself; it wraps the
    loader the other finders find."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if name != "threadpoolctl":
            return None
        _sys.meta_path.remove(_ThreadpoolctlHook)  # once, and not for its own search below
        import importlib.util as _util
        spec = _util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        run = loader.exec_module

        def exec_module(module):
            run(module)
            _register_threadpoolctl(module)

        loader.exec_module = exec_module
        return spec


if "threadpoolctl" in _sys.modules:
    _register_threadpoolctl(_sys.modules["threadpoolctl"])
else:
    _sys.meta_path.insert(0, _ThreadpoolctlHook)
