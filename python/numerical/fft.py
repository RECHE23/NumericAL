"""Discrete Fourier transforms, as :mod:`numpy.fft`: the same names, arguments and results.

Every length runs in O(n log n): Stockham passes of mixed radices, or Bluestein's convolution for a length with a prime
factor past 61. ``float32`` and ``complex64`` are transformed in single precision, as NumPy 2 does; integers as
``float64``. A real input's transform takes its real transform (:func:`rfft`'s) and the conjugates of its frequencies
for the others. A transform's bits are the same on any number of threads, a line alone or in a batch,
along any axis; they depend only on the path (portable, or the vector units, NEON, AVX2 and AVX-512 alike).
"""

import builtins as _builtins
import operator as _operator

from numerical import _numerical
from numerical import Tensor, asarray

__all__ = ["fft", "ifft", "fft2", "ifft2", "fftn", "ifftn", "rfft", "irfft", "rfft2", "irfft2", "rfftn", "irfftn", "hfft",
           "ihfft", "fftfreq", "rfftfreq", "fftshift", "ifftshift", "dct", "idct", "dst", "idst", "dctn", "idctn", "dstn",
           "idstn"]

_NORMS = {None: 0, "backward": 0, "ortho": 1, "forward": 2}


def _norm(norm):
    """The native code of `norm`, as NumPy reads it."""
    if norm not in _NORMS:
        raise ValueError(f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".')
    return _NORMS[norm]


def _operand(a):
    """`a` as a Tensor: a NumPy array, a nested list or a number through :func:`asarray`, integers and booleans as
    ``float64`` (NumPy transforms them as ``complex128``)."""
    tensor = a if isinstance(a, Tensor) else asarray(a)
    return tensor


def _axis(axis, ndim):
    """`axis` from 0, as NumPy reads a negative one; an IndexError past the rank."""
    axis = _operator.index(axis)
    if not -ndim <= axis < ndim:
        raise IndexError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis + ndim if axis < 0 else axis


def _no_out(out):
    """NumericAL's tensors are never written through: `out` is refused."""
    if out is not None:
        raise TypeError("out is not supported: the transform returns a new Tensor")


def _borrowed(kind, a, n, axis, norm, inverse, default_points):
    """The transform of a NumPy array read where it lies (``_numerical._borrowed_fft``, `kind` 0 fft, 1 rfft, 2
    irfft), or NotImplemented: `a` is not one, or not one so read."""
    if isinstance(a, Tensor) or not hasattr(a, "__array_struct__"):
        return NotImplemented
    axis = _axis(axis, a.ndim)
    points = default_points(a.shape[axis]) if n is None else _operator.index(n)
    if points < 1:
        raise ValueError(f"Invalid number of FFT data points ({points}) specified.")
    return _numerical._borrowed_fft(kind, a, points, axis, _norm(norm), inverse)


def _transform(a, n, axis, norm, inverse):
    """The one-axis transform of NumPy's `fft` or `ifft`."""
    result = _borrowed(0, a, n, axis, norm, inverse, lambda extent: extent)
    if result is not NotImplemented:
        return result
    tensor = _operand(a)
    axis = _axis(axis, tensor.ndim)
    points = tensor.shape[axis] if n is None else _operator.index(n)
    if points < 1:
        raise ValueError(f"Invalid number of FFT data points ({points}) specified.")
    return _numerical._fft(tensor, points, axis, _norm(norm), inverse)


def fft(a, n=None, axis=-1, norm=None, out=None):
    """The discrete Fourier transform of `a` along `axis`, as :func:`numpy.fft.fft`: `n` points, the axis cut or
    padded with zeros to them; scaled as `norm` says (none by default)."""
    _no_out(out)
    return _transform(a, n, axis, norm, False)


def ifft(a, n=None, axis=-1, norm=None, out=None):
    """The inverse discrete Fourier transform of `a` along `axis`, as :func:`numpy.fft.ifft` (scaled by 1/n by
    default)."""
    _no_out(out)
    return _transform(a, n, axis, norm, True)


def _shape_and_axes(tensor, s, axes):
    """NumPy's `s` and `axes` of an n-dimensional transform: every axis, or the last ``len(s)``, or those given."""
    if axes is None:
        axes = list(range(tensor.ndim)) if s is None else list(range(tensor.ndim - len(s), tensor.ndim))
    else:
        axes = [_axis(axis, tensor.ndim) for axis in axes]
    if s is None:
        s = [tensor.shape[axis] for axis in axes]
    elif len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    return list(s), axes


def _lent(a, kinds):
    """`a` itself where it is a NumPy array of one of the dtype kinds `kinds` (its first transform reads it where it
    lies, :func:`_borrowed`), else None."""
    if isinstance(a, Tensor) or not hasattr(a, "__array_struct__") or a.dtype.kind not in kinds:
        return None
    return a


def _transform_n(a, s, axes, norm, inverse):
    """NumPy's n-dimensional transform: each axis by the one-axis transform, the last given first; a NumPy array's
    first axis read where it lies."""
    tensor = _lent(a, "fc")
    tensor = _operand(a) if tensor is None else tensor
    s, axes = _shape_and_axes(tensor, s, axes)
    for points, axis in reversed(list(_builtins.zip(s, axes))):
        tensor = _transform(tensor, points, axis, norm, inverse)
    return _operand(a) if tensor is a else tensor


def fftn(a, s=None, axes=None, norm=None, out=None):
    """The n-dimensional discrete Fourier transform of `a`, as :func:`numpy.fft.fftn`."""
    _no_out(out)
    return _transform_n(a, s, axes, norm, False)


def ifftn(a, s=None, axes=None, norm=None, out=None):
    """The n-dimensional inverse discrete Fourier transform of `a`, as :func:`numpy.fft.ifftn`."""
    _no_out(out)
    return _transform_n(a, s, axes, norm, True)


def fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """The two-dimensional discrete Fourier transform of `a`, as :func:`numpy.fft.fft2`."""
    _no_out(out)
    return _transform_n(a, s, axes, norm, False)


def ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """The two-dimensional inverse discrete Fourier transform of `a`, as :func:`numpy.fft.ifft2`."""
    _no_out(out)
    return _transform_n(a, s, axes, norm, True)


def _real(a, who):
    """`a` as a real Tensor; a TypeError for a complex one, as NumPy's real transforms refuse it."""
    tensor = _operand(a)
    if tensor.dtype.startswith("complex"):
        raise TypeError(f"ufunc '{who}' not supported for the input types, and the inputs could not be safely coerced "
                        "to any supported types according to the casting rule ''safe''")
    return tensor


def _complex(a):
    """`a` as a complex Tensor: a real one widened to the complex of its precision."""
    tensor = _operand(a)
    if not tensor.dtype.startswith("complex"):
        tensor = asarray(tensor, "complex64" if tensor.dtype == "float32" else "complex128")
    return tensor


def _real_transform(a, n, axis, norm, hermitian):
    """NumPy's `rfft` (or `ihfft`): `n` real points, their ``n // 2 + 1`` frequencies."""
    tensor = _real(a, "rfft_n_even")
    axis = _axis(axis, tensor.ndim)
    points = tensor.shape[axis] if n is None else _operator.index(n)
    if points < 1:
        raise ValueError(f"Invalid number of FFT data points ({points}) specified.")
    return _numerical._rfft(tensor, points, axis, _norm(norm), hermitian)


def _real_inverse(a, n, axis, norm, hermitian):
    """NumPy's `irfft` (or `hfft`): `n` real points, ``2 * (m - 1)`` by default."""
    tensor = _complex(a)
    axis = _axis(axis, tensor.ndim)
    points = (tensor.shape[axis] - 1) * 2 if n is None else _operator.index(n)
    if points < 1:
        raise ValueError(f"Invalid number of FFT data points ({points}) specified.")
    return _numerical._irfft(tensor, points, axis, _norm(norm), hermitian)


def rfft(a, n=None, axis=-1, norm=None, out=None):
    """The discrete Fourier transform of real `a` along `axis`, its ``n // 2 + 1`` non-negative frequencies, as
    :func:`numpy.fft.rfft`."""
    _no_out(out)
    result = _borrowed(1, a, n, axis, norm, False, lambda extent: extent)
    return _real_transform(a, n, axis, norm, False) if result is NotImplemented else result


def irfft(a, n=None, axis=-1, norm=None, out=None):
    """The inverse of :func:`rfft`, `n` real points (``2 * (m - 1)`` by default), as :func:`numpy.fft.irfft`."""
    _no_out(out)
    result = _borrowed(2, a, n, axis, norm, False, lambda extent: (extent - 1) * 2)
    return _real_inverse(a, n, axis, norm, False) if result is NotImplemented else result


def hfft(a, n=None, axis=-1, norm=None, out=None):
    """The transform of a signal of Hermitian symmetry, `n` real points, as :func:`numpy.fft.hfft`."""
    _no_out(out)
    return _real_inverse(a, n, axis, norm, True)


def ihfft(a, n=None, axis=-1, norm=None, out=None):
    """The inverse of :func:`hfft`, as :func:`numpy.fft.ihfft`."""
    _no_out(out)
    return _real_transform(a, n, axis, norm, True)


def _real_shape_and_axes(tensor, s, axes, inverse):
    """NumPy's `s` and `axes` of an n-dimensional real transform; an inverse's last length ``2 * (m - 1)`` by default,
    an `s` of -1 the whole axis."""
    given = s is not None
    s, axes = _shape_and_axes(tensor, s, axes)
    s = [tensor.shape[axis] if size == -1 else size for size, axis in _builtins.zip(s, axes)]
    if inverse and not given:
        s[-1] = (tensor.shape[axes[-1]] - 1) * 2
    return s, axes


def rfftn(a, s=None, axes=None, norm=None, out=None):
    """The n-dimensional transform of real `a`, as :func:`numpy.fft.rfftn`: :func:`rfft` along the last axis, then
    :func:`fft` along the others, the last first."""
    _no_out(out)
    lent = _lent(a, "f")
    tensor = _real(a, "rfft_n_even") if lent is None else lent
    s, axes = _real_shape_and_axes(tensor, s, axes, False)
    first = _borrowed(1, tensor, s[-1], axes[-1], norm, False, None)
    tensor = _real_transform(tensor, s[-1], axes[-1], norm, False) if first is NotImplemented else first
    for points, axis in reversed(list(_builtins.zip(s[:-1], axes[:-1]))):
        tensor = _transform(tensor, points, axis, norm, False)
    return tensor


def irfftn(a, s=None, axes=None, norm=None, out=None):
    """The inverse of :func:`rfftn`, as :func:`numpy.fft.irfftn`: :func:`ifft` along each axis but the last, in order,
    then :func:`irfft` along the last."""
    _no_out(out)
    lent = _lent(a, "c")
    tensor = _complex(a) if lent is None else lent
    s, axes = _real_shape_and_axes(tensor, s, axes, True)
    for points, axis in _builtins.zip(s[:-1], axes[:-1]):
        tensor = _transform(tensor, points, axis, norm, True)
    return _real_inverse(tensor, s[-1], axes[-1], norm, False)


def rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """The two-dimensional transform of real `a`, as :func:`numpy.fft.rfft2`."""
    return rfftn(a, s, axes, norm, out)


def irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """The inverse of :func:`rfft2`, as :func:`numpy.fft.irfft2`."""
    return irfftn(a, s, axes, norm, out)


def fftfreq(n, d=1.0, device=None):
    """The sample frequencies of a transform of `n` points, `d` apart, as :func:`numpy.fft.fftfreq`: 0, 1, …,
    ⌈n/2⌉ − 1, then −⌊n/2⌋, …, −1, each times 1/(n·d)."""
    if device not in (None, "cpu"):
        raise ValueError(f"Device not understood. Only \"cpu\" is allowed, but received: {device}")
    step = 1.0 / (n * d)
    half = (n - 1) // 2 + 1
    return asarray([k * step for k in [*range(half), *range(-(n // 2), 0)]])


def rfftfreq(n, d=1.0, device=None):
    """The sample frequencies of a real transform of `n` points, `d` apart, as :func:`numpy.fft.rfftfreq`: 0, 1, …,
    ⌊n/2⌋, each times 1/(n·d)."""
    if device not in (None, "cpu"):
        raise ValueError(f"Device not understood. Only \"cpu\" is allowed, but received: {device}")
    step = 1.0 / (n * d)
    return asarray([k * step for k in range(n // 2 + 1)])


def _shift(x, axes, inverse):
    """NumPy's roll of `x` along `axes` (every axis by default) by half each extent."""
    tensor = _operand(x)
    if axes is None:
        axes = range(tensor.ndim)
    elif isinstance(axes, int):
        axes = [axes]
    return _numerical._fftshift(tensor, [_axis(axis, tensor.ndim) for axis in axes], inverse)


def fftshift(x, axes=None):
    """`x` with its zero frequency moved to the middle, as :func:`numpy.fft.fftshift`."""
    return _shift(x, axes, False)


def ifftshift(x, axes=None):
    """The inverse of :func:`fftshift`, as :func:`numpy.fft.ifftshift`."""
    return _shift(x, axes, True)


def _trig(x, sine, type, n, axis, norm, orthogonalize, inverse):
    """One axis of SciPy's dct, idct, dst or idst; integers and booleans as ``float64``."""
    tensor = _operand(x)
    axis = _axis(axis, tensor.ndim)
    points = tensor.shape[axis] if n is None else _operator.index(n)
    if points < 1:
        raise ValueError(f"invalid number of data points ({points}) specified")
    weighted = -1 if orthogonalize is None else int(bool(orthogonalize))
    try:
        return _numerical._trig(tensor, sine, _operator.index(type), points, axis, _norm(norm), weighted, inverse)
    except _numerical.error as caught:
        raise ValueError(*caught.args) from None


def _trig_n(x, sine, type, s, axes, norm, orthogonalize, inverse):
    """SciPy's n-dimensional dctn, idctn, dstn or idstn: the one-axis transform along each axis."""
    tensor = _operand(x)
    if isinstance(axes, int):
        axes = [axes]
    if isinstance(s, int):
        s = [s]
    s, axes = _shape_and_axes(tensor, s, axes)
    for points, axis in _builtins.zip(s, axes):
        tensor = _trig(tensor, sine, type, points, axis, norm, orthogonalize, inverse)
    return tensor


def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    """The discrete cosine transform of `x` along `axis`, as :func:`scipy.fft.dct`: type 1, 2 (the default), 3 or 4,
    each by a real or complex Fourier transform; `n` points (the axis cut or padded with zeros); `norm` ``"backward"``
    (none), ``"ortho"`` or ``"forward"``; `orthogonalize` the first (and type 1's last) point and value weighted by
    √2, by default where `norm` is ``"ortho"``. A complex `x` its real and imaginary parts each. `overwrite_x` and
    `workers` are accepted and not needed: `x` is never written, and the threads are the library's."""
    return _trig(x, False, type, n, axis, norm, orthogonalize, False)


def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    """The inverse of :func:`dct` of the same `type`, as :func:`scipy.fft.idct`: types 2 and 3 each other's inverse,
    1 and 4 their own, scaled by ``1/2n`` (``1/2(n - 1)``, type 1) by default."""
    return _trig(x, False, type, n, axis, norm, orthogonalize, True)


def dst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    """The discrete sine transform of `x` along `axis`, as :func:`scipy.fft.dst` (see :func:`dct`): `orthogonalize`
    the last point (type 3) or value (type 2) weighted by √2."""
    return _trig(x, True, type, n, axis, norm, orthogonalize, False)


def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    """The inverse of :func:`dst` of the same `type`, as :func:`scipy.fft.idst` (``1/2(n + 1)`` for type 1)."""
    return _trig(x, True, type, n, axis, norm, orthogonalize, True)


def dctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    """:func:`dct` along each of `axes` (every axis by default), `s` their points, as :func:`scipy.fft.dctn`."""
    return _trig_n(x, False, type, s, axes, norm, orthogonalize, False)


def idctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    """:func:`idct` along each of `axes`, as :func:`scipy.fft.idctn`."""
    return _trig_n(x, False, type, s, axes, norm, orthogonalize, True)


def dstn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    """:func:`dst` along each of `axes`, as :func:`scipy.fft.dstn`."""
    return _trig_n(x, True, type, s, axes, norm, orthogonalize, False)


def idstn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False, workers=None, orthogonalize=None):
    """:func:`idst` along each of `axes`, as :func:`scipy.fft.idstn`."""
    return _trig_n(x, True, type, s, axes, norm, orthogonalize, True)

