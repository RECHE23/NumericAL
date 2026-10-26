"""Discrete Fourier transforms, as :mod:`numpy.fft`: the same names, arguments and results.

Every length runs in O(n log n): Stockham passes of mixed radices, or Bluestein's convolution for a length with a prime
factor past 61. ``float32`` and ``complex64`` are transformed in single precision, as NumPy 2 does; integers and
``float64`` as ``complex128``. A transform's bits are the same on any number of threads, a line alone or in a batch,
along any axis; they depend only on the path (portable, or the vector units, NEON, AVX2 and AVX-512 alike).
"""

import builtins as _builtins
import operator as _operator

from numerical import _numerical
from numerical import Tensor, asarray

__all__ = ["fft", "ifft", "fft2", "ifft2", "fftn", "ifftn", "fftfreq", "rfftfreq", "fftshift", "ifftshift"]

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


def _transform(a, n, axis, norm, inverse):
    """The one-axis transform of NumPy's `fft` or `ifft`."""
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


def _transform_n(a, s, axes, norm, inverse):
    """NumPy's n-dimensional transform: each axis by the one-axis transform, the last given first."""
    tensor = _operand(a)
    s, axes = _shape_and_axes(tensor, s, axes)
    for points, axis in reversed(list(_builtins.zip(s, axes))):
        tensor = _transform(tensor, points, axis, norm, inverse)
    return tensor


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
    for axis in axes:
        tensor = _numerical._fftshift(tensor, _axis(axis, tensor.ndim), inverse)
    return tensor


def fftshift(x, axes=None):
    """`x` with its zero frequency moved to the middle, as :func:`numpy.fft.fftshift`."""
    return _shift(x, axes, False)


def ifftshift(x, axes=None):
    """The inverse of :func:`fftshift`, as :func:`numpy.fft.ifftshift`."""
    return _shift(x, axes, True)
