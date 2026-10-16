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
import functools as _functools
import numbers as _numbers
import operator as _operator
import os as _os
import sys as _sys
import warnings as _warnings

from numerical import _numerical
from numerical._numerical import Tensor, asarray as _asarray, error, get_num_threads
from numerical._numerical import _gemm_path, _limit_gemm_path, _set_num_threads

__all__ = [
    "Tensor", "einsum", "solve", "asarray", "error", "LinAlgError",
    "determinant", "det", "slogdet", "inverse", "inv", "lstsq", "pinv", "matrix_rank", "cholesky", "qr", "eigh",
    "eigvalsh", "svd",
    "array_equal", "set_num_threads", "get_num_threads", "gemm_path", "limit_gemm_path",
    "get_include", "get_config",
]


class LinAlgError(error, ValueError):
    """Raised by the linear-algebra functions when they cannot compute their answer: a singular matrix, one
    not positive definite, an infinite or NaN entry, an iteration that did not converge, or an operand of the
    wrong shape. As ``numpy.linalg.LinAlgError`` it is a :class:`ValueError`; it is also a
    :class:`numerical.error`, so code catching that keeps working."""


__version__ = "2026.10.15"

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

_GEMM_PATHS = ("portable", "avx2", "avx512", "amx")


def gemm_path():
    """The kernel float64 matrix products run on: ``"portable"`` (the compiler's vectors), ``"avx2"``,
    ``"avx512"`` (x86-64, chosen at run time from cpuid) or ``"amx"`` (Apple M1 to M4, an undocumented
    coprocessor, used only on a known chip that passes a self-test; see numerical/amx.hpp)."""
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
    (portable, avx2, avx512, amx); NUMERICAL_AMX set to 0, false, no or off keeps products off the Apple AMX, its
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


_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_REVERSALS = [f"{_LETTERS[:rank]}->{_LETTERS[:rank][::-1]}" for rank in range(len(_LETTERS) + 1)]


def _transpose(self):
    """The tensor with its axes reversed: a matrix's transpose, as NumPy's ``.T``. The spec is built once per rank and
    the extension called directly: a transposition has no error for the wrapper to translate."""
    return _numerical.einsum(_REVERSALS[len(self.shape)], self)


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
determinant = _linalg(_numerical.determinant)
det = determinant
einsum = _linalg(_numerical.einsum)
inverse = _linalg(_numerical.inverse)
inv = inverse
_lstsq = _linalg(_numerical.lstsq, arrays=2)
_eigvalsh = _linalg(_numerical.eigvalsh, arrays=1)
_matrix_rank = _linalg(_numerical.matrix_rank, arrays=1)
_matrix_ranks = _linalg(_numerical._matrix_ranks, arrays=1)
_pinv = _linalg(_numerical.pinv, arrays=1)
_qr = _linalg(_numerical.qr, arrays=1)
slogdet = _linalg(_numerical.slogdet)
solve = _linalg(_numerical.solve)
_eigh = _linalg(_numerical.eigh, arrays=1)
_svd = _linalg(_numerical.svd, arrays=1)
_singular_values = _linalg(_numerical._singular_values)
_QR_MODES = ("reduced", "complete", "r")
_EIGH_METHODS = ("divide_and_conquer", "tridiagonal_ql", "jacobi")
_SVD_METHODS = ("divide_and_conquer", "bidiagonal_qr", "jacobi")


_MACHINE_EPSILON = {"float64": 2.220446049250313e-16, "float32": 1.1920928955078125e-07}  # complex's parts share them


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


def eigh(a, UPLO="L", *, method="divide_and_conquer"):
    """The eigenvalues (real, ascending) and eigenvectors (columns) of a real symmetric or complex Hermitian
    matrix, as ``numpy.linalg.eigh``: its lower triangle is read (``UPLO="L"``), or its upper one (``"U"``).
    `method` diagonalises the tridiagonal form: ``"divide_and_conquer"`` (LAPACK's ``dsyevd``/``zheevd``),
    ``"tridiagonal_ql"`` (implicit QL, ``dsyev``'s, the default before 2026.10.4) or ``"jacobi"`` (cyclic Jacobi
    rotations, real matrices only)."""
    upper = _upper("eigh", UPLO)
    if method not in _EIGH_METHODS:
        raise ValueError(f"eigh: method {method!r} is none of {_EIGH_METHODS}")
    return _eigh(a, method, upper)


def eigvalsh(a, UPLO="L"):
    """The eigenvalues alone of a real symmetric or complex Hermitian matrix, real and ascending, as
    ``numpy.linalg.eigvalsh``: its lower triangle is read (``UPLO="L"``), or its upper one (``"U"``)."""
    return _eigvalsh(a, _upper("eigvalsh", UPLO))


def cholesky(a, /, *, upper=False):
    """The Cholesky factor of a Hermitian positive-definite matrix, as ``numpy.linalg.cholesky``: the lower `L` with
    ``a = L @ L.conj().T``, read from the lower triangle; with ``upper=True``, the upper `U` with ``a = U.conj().T @ U``,
    read from the upper triangle."""
    return _cholesky(a, bool(upper))


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


def svd(a, full_matrices=True, compute_uv=True, *, method="divide_and_conquer"):
    """``(U, S, Vh)`` with ``a = U @ diag(S) @ Vh``, the singular values descending, as ``numpy.linalg.svd``: for an
    m×n `a`, `U` is m×m and `Vh` n×n (``full_matrices=True``, NumPy's default: the columns past min(m, n) an
    orthonormal basis of the complement), or m×k and k×n, k = min(m, n) (``full_matrices=False``); the singular values
    alone with ``compute_uv=False``. `method` decomposes the bidiagonal form: ``"divide_and_conquer"`` (LAPACK's
    ``dgesdd``), ``"bidiagonal_qr"`` (implicit-shift QR, ``dgesvd``'s) or ``"jacobi"`` (one-sided Jacobi, each singular
    value to high relative accuracy)."""
    if isinstance(full_matrices, str):
        raise TypeError("svd: the second argument is full_matrices, as in NumPy; give the method by name, "
                        f"svd(a, method={full_matrices!r})")
    if method not in _SVD_METHODS:
        raise ValueError(f"svd: method {method!r} is none of {_SVD_METHODS}")
    if not compute_uv:
        return _singular_values(a)
    return _svd(a, method, bool(full_matrices))


def qr(a, mode="reduced"):
    """``(Q, R)`` with ``a = Q @ R``, as ``numpy.linalg.qr``: for an m×n `a` and k = min(m, n), `Q` is m×k with
    orthonormal columns and `R` k×n upper trapezoidal (``mode="reduced"``, the default); `Q` m×m and `R` m×n
    (``"complete"``); `R` alone, k×n (``"r"``). `R`'s diagonal is real and non-negative."""
    if mode not in _QR_MODES:
        raise ValueError(f"qr: mode {mode!r} is none of {_QR_MODES} (the LAPACK reflectors of \"raw\" are not offered)")
    if mode == "r":
        return _qr(a, "reduced")[1]
    return _qr(a, mode)


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
    # as they are; a nested list, or another number (a Fraction), through its values.
    if isinstance(obj, (list, tuple)) or (type(obj) not in _NATIVE_NUMBERS and isinstance(obj, _numbers.Number)
                                          and not hasattr(obj, "__array_interface__")):
        values, shape = _nested(obj)
        obj = _NestedBuffer(values, shape)
    elif not (isinstance(obj, Tensor) or type(obj) in _NATIVE_NUMBERS or hasattr(obj, "__array_interface__")
              or hasattr(obj, "__array_struct__")):
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


def _arithmetic(op, reflected=False):
    """The Tensor method for `op` ("+", "-", "*", "/", or "^" for ``**``), element-wise and broadcast as NumPy's; a reflected
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
    def reduce(a, axis=None):
        a = a if isinstance(a, Tensor) else asarray(a)
        return _native(_numerical._reduce, op, a, _axis(a, axis))

    reduce.__name__ = op
    reduce.__doc__ = doc
    return reduce


sum = _reduction("sum", """The sum of every element (a number), or the sums along `axis` (a Tensor without it),
as ``numpy.sum``; added by pairwise halving, as NumPy adds.""")
mean = _reduction("mean", """The mean of every element, or the means along `axis`, as ``numpy.mean``.""")
max = _reduction("max", """The largest element, or the largest along `axis`, as ``numpy.max``; a NaN wins. Real
dtypes only (a complex one has no order).""")
min = _reduction("min", """The smallest element, or the smallest along `axis`, as ``numpy.min``; a NaN wins. Real
dtypes only.""")
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
    return _native(_numerical._binary, "^", a, b)


__all__ += ["sum", "mean", "max", "min", "reshape", "full", "zeros", "ones", "eye", "arange", "linspace",
            "exp", "log", "sqrt", "sin", "cos", "tan", "tanh", "absolute", "power"]
