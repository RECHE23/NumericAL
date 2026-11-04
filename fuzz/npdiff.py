"""Differential fuzzing of NumericAL's NumPy functions composed in Python against NumPy: sorting and searching, sets,
statistics and quantiles, histograms, polynomials, shape and selection functions, integer arithmetic, ranges and
windows.

Each family draws its inputs at random: values among the special ones (zeros of either sign, infinities, NaN,
subnormals, the largest finite, exact halves, 2^53 + 2; INT64_MIN and INT64_MAX, ±(2^53 + 1)), shapes of rank 0 to
3 with empty axes, axes and options, then calls NumPy's function and NumericAL's on the same inputs. They must agree:
the same dtype, shape and bits (two NaNs alike, each zero's sign), a full reduction's number equal to NumPy's scalar,
or both raise. A family may allow what the documented differences name (docs/site/guide/differences.md): a few ulps
where NumericAL goes through another library's function, a zero's sign chosen by the order of a sort or a sum, a sum
along an axis other than the last added in another order.

Each family's cases are drawn from the seed and the family's name, so a family's campaign replays alone:

    python3 fuzz/npdiff.py --seed S --family F --case C
"""
import argparse
import math
import platform
import sys
import time
import warnings
import zlib

import numpy as np

import numerical

INT64_MIN, INT64_MAX = -2 ** 63, 2 ** 63 - 1
EXTENTS = (0, 1, 1, 2, 3, 3, 4, 5, 8)
NUMPY = tuple(int(part) for part in np.__version__.split(".")[:2])
# NumPy's max and min (fmax, fmin, nanmax…) of +0 and -0 give IEEE 754-2019's answer on arm64, as NumericAL's on any
# processor; on x86-64, the sign of a zero as their vector loops meet it.
EXTREME_ZEROS = {} if platform.machine().lower() in ("arm64", "aarch64") else {"zero_signs": True}


class Failure(Exception):
    pass


# The drawn inputs.

def shape(rng, low=0, high=3):
    return tuple(int(rng.choice(EXTENTS)) for _ in range(int(rng.integers(low, high + 1))))


def floats(rng, size, dtype="float64", special=0.25):
    """`size` floats of `dtype`: normal ones over many scales, a share of special values."""
    info = np.finfo(dtype)
    specials = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, float(info.max), -float(info.max), float(info.tiny) / 4,
                         -float(info.tiny), 0.5, 1.5, -2.5, 1.0, 2.0, 2.0 ** 53 + 2], dtype)
    values = (rng.standard_normal(size) * np.exp(rng.uniform(-4, 4, size))).astype(dtype)
    pick = rng.random(size) < special
    values[pick] = specials[rng.integers(0, len(specials), int(pick.sum()))]
    return values


def ints(rng, size, extremes=0.1):
    """`size` int64: small ones, a share of the extremes."""
    specials = np.array([INT64_MIN, INT64_MAX, INT64_MIN + 1, 2 ** 53 + 1, -(2 ** 53 + 1), 0, -1], np.int64)
    values = rng.integers(-20, 21, size).astype(np.int64)
    pick = rng.random(size) < extremes
    values[pick] = specials[rng.integers(0, len(specials), int(pick.sum()))]
    return values


def array(rng, dims=None, kinds=("f8", "f8", "f4", "i8", "b"), special=0.25):
    """An array of a drawn kind and shape."""
    dims = shape(rng) if dims is None else dims
    size = int(np.prod(dims)) if dims else 1
    kind = kinds[int(rng.integers(len(kinds)))]
    if kind == "i8":
        values = ints(rng, size)
    elif kind == "b":
        values = rng.random(size) < 0.5
    elif kind == "c16":
        values = np.empty(size, complex)  # each part set apart: 1j * inf would be nan + inf j
        values.real, values.imag = floats(rng, size, special=special), floats(rng, size, special=special)
    else:
        values = floats(rng, size, "float32" if kind == "f4" else "float64", special)
    return values.reshape(dims)


def small_floats(rng, dims=None):
    """Floats without special values, some repeated: what a statistic or a fit takes."""
    dims = shape(rng, 1, 2) if dims is None else dims
    size = int(np.prod(dims))
    values = np.round(rng.standard_normal(size) * 4, int(rng.integers(0, 3)))
    return values.reshape(dims)


def axis_of(rng, a, none=0.3):
    if a.ndim == 0 or rng.random() < none:
        return None
    return int(rng.integers(-a.ndim, a.ndim))


def ours(value):
    """An input as NumericAL takes it: a NumPy array of its very dtype (int64 and bool included), the rest as it is."""
    if isinstance(value, np.ndarray):
        return numerical.asarray(value, "bool" if value.dtype == bool else str(value.dtype))
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, np.ndarray) for v in value):
        return type(value)(ours(v) for v in value)
    return value


# The comparison.

def bits_equal(got, want, ulps=0, zero_signs=False):
    """`got` and `want` (NumPy arrays of one dtype and shape) to the bit, two NaNs alike; within `ulps` where both
    are finite and nonzero; a zero of either sign alike with `zero_signs`."""
    if want.dtype.kind not in "fc":
        return got.tobytes() == want.tobytes()
    parts = ((got.real, want.real), (got.imag, want.imag)) if want.dtype.kind == "c" else ((got, want),)
    for g, w in parts:
        nan = np.isnan(w)
        if not np.array_equal(nan, np.isnan(g)):
            return False
        g, w = np.where(nan, 0, g), np.where(nan, 0, w)
        if zero_signs:
            g, w = np.where(g == 0, 0, g), np.where(w == 0, 0, w)
        if ulps:
            close = np.isfinite(w) & (w != 0) & np.isfinite(g) & (np.sign(g) == np.sign(w))
            if not close.any():
                if g.tobytes() != w.tobytes():
                    return False
                continue
            if np.where(close, 0, g).tobytes() != np.where(close, 0, w).tobytes():
                return False
            itype = np.int32 if w.dtype == np.float32 else np.int64
            gap = np.abs(g[close].view(itype).astype(np.float64) - w[close].view(itype).astype(np.float64))
            if gap.max() > ulps:
                return False
        elif g.tobytes() != w.tobytes():
            return False
    return True


def number_equal(got, want):
    """A number NumericAL answers against NumPy's scalar: the same value (a zero's sign, NaN alike)."""
    if isinstance(want, (complex, np.complexfloating)) or isinstance(got, complex):
        return number_equal(complex(got).real, complex(want).real) and number_equal(complex(got).imag,
                                                                                     complex(want).imag)
    if isinstance(got, float) or isinstance(want, float):
        g, w = float(got), float(want)
        return (math.isnan(g) and math.isnan(w)) or (g == w and math.copysign(1, g) == math.copysign(1, w))
    return got == want


def agree(got, want, rule, label):
    """Raises Failure where `got` (NumericAL's) and `want` (NumPy's) differ beyond the family's `rule`."""
    if isinstance(want, (tuple, list)):
        if not isinstance(got, (tuple, list)) or len(got) != len(want):
            raise Failure(f"{label}: {type(got).__name__} of {len(got) if hasattr(got, '__len__') else '?'} "
                          f"against {len(want)} answers")
        for k, (g, w) in enumerate(zip(got, want)):
            agree(g, w, rule, f"{label}[{k}]")
        return
    if isinstance(got, (bool, int, float, complex)) and not isinstance(got, np.generic):
        w = np.asarray(want)
        if w.shape != ():
            raise Failure(f"{label}: a number {got!r} against shape {w.shape}")
        if not number_equal(got, w.item()):
            if rule.keys() & {"ulps", "zero_signs", "order", "overflow"}:
                agree(np.asarray(got, w.dtype), w, rule, label)
                return
            raise Failure(f"{label}: {got!r} against {w.item()!r} ({w.dtype})")
        return
    g, w = np.asarray(got), np.asarray(want)
    if w.dtype.kind in "iu" and w.dtype != np.int64:  # NumPy's intp (int32 on a 32-bit Windows)
        w = w.astype(np.int64)
    if (g.shape, g.dtype) != (w.shape, w.dtype):
        raise Failure(f"{label}: {g.dtype}{list(g.shape)} against {w.dtype}{list(w.shape)}")
    if rule.get("overflow") and w.dtype.kind == "f":  # NumericAL's finite answer where NumPy's overflows
        exact = np.isinf(w) & np.isfinite(g) & (np.abs(g) > np.finfo(w.dtype).max / 4) & (np.sign(g) == np.sign(w))
        g, w = np.where(exact, 0, g).astype(w.dtype), np.where(exact, 0, w).astype(w.dtype)
    if rule.get("order") and w.dtype.kind in "fc":  # another order of a sum: within its rounding errors
        eps = np.finfo(w.dtype).eps
        parts = ((g.real, w.real), (g.imag, w.imag)) if w.dtype.kind == "c" else ((g, w),)
        if rule.get("near_overflow"):  # terms near the largest: which partial sum overflows depends on the order
            parts = tuple((np.where(np.isfinite(gp) & np.isfinite(wp), gp, 0), np.where(np.isfinite(gp) &
                           np.isfinite(wp), wp, 0)) for gp, wp in parts)
        finite = np.abs(np.concatenate([part[np.isfinite(part)].ravel() for _, part in parts]))
        scale = rule.get("scale", float(finite.max()) if finite.size else 0.0)  # the terms' magnitude, or the result's
        with np.errstate(all="ignore"):
            if all(np.allclose(gp, wp, rtol=rule["order"] * eps, atol=rule["order"] * eps * scale, equal_nan=True)
                   and np.array_equal(np.isinf(gp), np.isinf(wp)) for gp, wp in parts):
                return
    if not bits_equal(g, w, rule.get("ulps", 0), rule.get("zero_signs", False)):
        with np.printoptions(precision=17, threshold=12):
            raise Failure(f"{label}: {g!r} against {w!r}")


def call(function):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                return function(), None
    except Exception as error:  # the error is the answer compared
        return None, error


SAME_ERRORS = ({"AxisError", "ValueError", "IndexError"}, {"LinAlgError", "ValueError"})


def same_error(ours_, theirs):
    a, b = type(ours_).__name__, type(theirs).__name__
    return a == b or isinstance(ours_, type(theirs)) or any(a in group and b in group for group in SAME_ERRORS)


COMPARED = [0]


def check(rule, label, numpy_call, our_call):
    COMPARED[0] += 1
    want, their_error = call(numpy_call)
    got, our_error = call(our_call)
    if their_error is not None:
        if our_error is None:
            raise Failure(f"{label}: NumPy raises {type(their_error).__name__}: {their_error}; NumericAL answers")
        if not same_error(our_error, their_error):
            raise Failure(f"{label}: NumPy raises {type(their_error).__name__}: {their_error}; NumericAL "
                          f"{type(our_error).__name__}: {our_error}")
        return
    if our_error is not None:
        raise Failure(f"{label}: NumericAL raises {type(our_error).__name__}: {our_error}")
    agree(got, want, rule, label)


def same(rule, label, name, *args, ours_name=None, **kwargs):
    """NumPy's `name` and NumericAL's (`ours_name`) on the same arguments, ours converted by :func:`ours`."""
    theirs = _resolve(np, name)
    mine = _resolve(numerical, ours_name or name)
    check(rule, f"{name}{_shown(args, kwargs)}" if not label else label,
          lambda: theirs(*args, **kwargs),
          lambda: mine(*[ours(a) for a in args], **{k: ours(v) for k, v in kwargs.items()}))


def _resolve(module, dotted):
    for part in dotted.split("."):
        module = getattr(module, part)
    return module


def _shown(args, kwargs):
    def show(value):
        if isinstance(value, np.ndarray):
            shown = f"np.array({value.tolist()!r}, {str(value.dtype)!r})"
            return shown if value.size else f"{shown}.reshape({value.shape})"
        return repr(value)
    return "(" + ", ".join([show(a) for a in args] + [f"{k}={show(v)}" for k, v in kwargs.items()]) + ")"


# The families: each draws one case from `rng` and checks it.

def integer_arithmetic(rng):
    a, b = ints(rng, 6).reshape(2, 3), ints(rng, 3)
    name = str(rng.choice(["floor_divide", "remainder", "fmod", "add", "subtract", "multiply", "absolute",
                           "negative", "sign", "maximum", "minimum"]))
    if name in ("absolute", "negative", "sign"):
        same({}, "", name, a)
    else:
        same({}, "", name, a, b)
    if rng.random() < 0.5:
        x = floats(rng, 4)
        same({}, "", name if name not in ("absolute", "negative", "sign") else "add", ints(rng, 4), x)


def powers(rng):
    base = ints(rng, 5, extremes=0.05)
    exponent = rng.integers(0, 6, 5).astype(np.int64)
    same({}, "", "power", base, exponent)
    same({}, "", "power", floats(rng, 5), floats(rng, 5, special=0.4))


def rounding(rng):
    a = array(rng, kinds=("f8", "f4"))
    decimals = int(rng.integers(-3, 4)) if rng.random() < 0.6 else 0
    same({"ulps": 1, "overflow": True} if decimals else {}, "", "round", a, decimals)  # rounded exactly here
    same({}, "", str(rng.choice(["floor", "ceil", "trunc", "rint"])), a)


def sorting(rng):
    a = array(rng, shape(rng, 1, 3), kinds=("f8", "f4", "i8", "b"), special=0.4)
    axis = int(rng.integers(-a.ndim, a.ndim)) if rng.random() < 0.8 else None
    same({"zero_signs": True}, "", "sort", a, axis=axis)
    same({}, "", "argsort", a, axis=axis, stable=True)
    same({}, "", str(rng.choice(["argmax", "argmin"])), a, axis=axis)
    same(EXTREME_ZEROS, "", str(rng.choice(["max", "min"])), a, axis=axis, keepdims=bool(rng.random() < 0.3))


def searching(rng):
    sorted_ = np.sort(array(rng, (int(rng.choice(EXTENTS)),), kinds=("f8", "i8"), special=0.3))
    values = array(rng, kinds=("f8", "i8"), special=0.3)
    side = str(rng.choice(["left", "right"]))
    same({}, "", "searchsorted", sorted_, values, side=side)
    if rng.random() < 0.5:
        unsorted = rng.permutation(sorted_)
        same({}, "", "searchsorted", unsorted, values, side=side, sorter=np.argsort(unsorted, kind="stable"))
    same({}, "", "nonzero", array(rng, shape(rng, 1, 3), special=0.5))
    same({}, "", "flatnonzero", array(rng, special=0.5))
    same({}, "", "count_nonzero", array(rng, special=0.5))
    bins = np.sort(floats(rng, int(rng.integers(1, 5)), special=0.0))
    same({}, "", "digitize", array(rng, kinds=("f8",)), bins, right=bool(rng.random() < 0.5))


def sets(rng):
    kinds = ("f8", "i8") if rng.random() < 0.7 else ("f8", "i8", "f4", "b")
    a = array(rng, shape(rng, 1, 2), kinds=kinds, special=0.3)
    b = array(rng, shape(rng, 1, 1), kinds=kinds, special=0.3)
    if rng.random() < 0.5:  # repeated values
        a = a.ravel()[rng.integers(0, max(a.size, 1), a.size)] if a.size else a
    for name in ("intersect1d", "union1d", "setdiff1d", "setxor1d"):
        same({"zero_signs": True}, "", name, a, b)
    same({}, "", "isin", a, b, invert=bool(rng.random() < 0.5))
    flags = dict(return_index=bool(rng.random() < 0.5), return_inverse=bool(rng.random() < 0.5),
                 return_counts=bool(rng.random() < 0.5))
    if a.ndim == 1 or not flags["return_inverse"] or NUMPY >= (2, 1):
        same({"zero_signs": True}, "", "unique", a, **flags)


def statistics(rng):
    a = array(rng, shape(rng, 1, 2), kinds=("f8", "f4", "i8"), special=0.15)
    axis = axis_of(rng, a)
    ddof = int(rng.integers(0, 2))
    keepdims = bool(rng.random() < 0.3)
    last = axis is None or axis in (-1, a.ndim - 1) or a.ndim == 1
    sums = {"ulps": 2} if last else other_order(a)  # NumPy adds along another axis row by row
    same(sums, "", "mean", a, axis=axis, keepdims=keepdims)
    same(sums, "", "var", a, axis=axis, ddof=ddof, keepdims=keepdims)
    same(sums, "", "std", a, axis=axis, ddof=ddof, keepdims=keepdims)
    same(EXTREME_ZEROS, "", "ptp", a, axis=axis, keepdims=keepdims)
    same({"overflow": True}, "", "median", a, axis=axis, keepdims=keepdims)  # NumPy's mean of ±max with itself
    same({"overflow": True}, "", "nanmedian", a, axis=axis, keepdims=keepdims)
    same(EXTREME_ZEROS, "", "nanmax", a, axis=axis)
    same(sums, "", "nanmean", a, axis=axis)
    same(sums, "", "nanvar", a, axis=axis, ddof=ddof)


# NumPy before 2.3 takes a Python float `q` as float64 beside float32 values (not as a weak one)
QUANTILE_KINDS = ("f8", "f4", "i8") if NUMPY >= (2, 3) else ("f8", "i8")


def quantiles(rng):
    a = array(rng, shape(rng, 1, 2), kinds=QUANTILE_KINDS, special=0.1)
    if not a.size and NUMPY < (2, 3):  # 2.2 answers some quantiles of nothing that 2.3 refuses
        return
    axis = axis_of(rng, a)
    q = float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, float(rng.random())])) if rng.random() < 0.6 else \
        rng.random(int(rng.integers(0, 4)))
    method = str(rng.choice(["linear", "lower", "higher", "nearest", "midpoint", "inverted_cdf",
                             "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen",
                             "weibull", "median_unbiased", "normal_unbiased"]))
    keepdims = bool(rng.random() < 0.3)
    same({"overflow": True}, "", "quantile", a, q, axis=axis, method=method, keepdims=keepdims)
    same({"overflow": True}, "", "nanquantile", a, q, axis=axis, method=method, keepdims=keepdims)
    same({"overflow": True}, "", "percentile", a, np.multiply(q, 100), axis=axis, method=method)
    if a.dtype.kind == "f" and rng.random() < 0.5 and NUMPY >= (2, 3):  # 2.3 refuses weights that are all zero
        weights = rng.integers(0, 4, a.shape).astype(np.float64)
        same({}, "", "quantile", a, q, axis=axis, method="inverted_cdf", weights=weights)
        same({}, "", "nanquantile", a, q, axis=axis, method="inverted_cdf", weights=weights)


def averages(rng):
    a = small_floats(rng)
    axis = axis_of(rng, a)
    weights = rng.integers(0, 4, a.shape).astype(np.float64) if rng.random() < 0.7 else None
    if a.size or weights is not None:  # NumPy divides by the mean's size first: ZeroDivisionError for no element
        same({"ulps": 2}, "", "average", a, axis=axis, weights=weights, returned=bool(rng.random() < 0.3))
    m = small_floats(rng, (int(rng.integers(1, 4)), int(rng.integers(1, 6))))
    same({"order": 64}, "", "cov", m, ddof=int(rng.integers(0, 2)) if rng.random() < 0.3 else None)
    same({"order": 64}, "", "corrcoef", m)


def reductions(rng):
    # a bool's sums are float64 here until 2026.12.0, NumPy's int64
    a = array(rng, shape(rng, 0, 3), kinds=("f8", "f4", "c16", "i8"), special=0.15)
    axis = axis_of(rng, a)
    last = axis is None or axis in (-1, a.ndim - 1) or a.ndim <= 1
    for name in ("sum", "prod", "cumsum", "cumprod"):
        if name in ("sum", "prod") and last:
            same({}, "", name, a, axis=axis)
        else:  # NumPy adds along another axis row by row, and a product as it may
            same(other_order(a) if name.endswith("sum") else {"order": 64, "zero_signs": True}, "", name, a,
                 axis=axis)


def histograms(rng):
    a = small_floats(rng, (int(rng.choice([0, 1, 5, 40])),))
    estimators = ["fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"] + (["auto"] if NUMPY >= (2, 3) else [])
    bins = int(rng.integers(1, 12)) if rng.random() < 0.6 else str(rng.choice(estimators))  # 2.3 bounds "auto"
    span = (float(rng.uniform(-6, 0)), float(rng.uniform(0, 6))) if rng.random() < 0.3 else None
    weights = rng.integers(0, 3, a.shape).astype(np.float64) if rng.random() < 0.3 and not isinstance(bins, str) \
        else None
    same({}, "", "histogram", a, bins=bins, range=span, weights=weights, density=bool(rng.random() < 0.2))
    same({}, "", "histogram_bin_edges", a, bins=bins, range=span)
    x, y = small_floats(rng, (12,)), small_floats(rng, (12,))
    same({}, "", "histogram2d", x, y, bins=int(rng.integers(1, 5)))
    counts = rng.integers(0, 6, int(rng.integers(0, 8))).astype(np.int64)
    same({}, "", "bincount", counts, minlength=int(rng.integers(0, 4)))
    same({}, "", "bincount", counts, weights=floats(rng, counts.size, special=0.0))


def interpolation(rng):
    xp = np.sort(small_floats(rng, (int(rng.integers(1, 7)),)))
    xp = np.unique(xp)
    fp = floats(rng, xp.size, special=0.1)
    x = array(rng, kinds=("f8",), special=0.2)
    same({}, "", "interp", x, xp, fp)
    same({}, "", "interp", x, xp, fp, left=-1.0, right=float(rng.random()))


def polynomials(rng):
    p = rng.integers(-3, 4, int(rng.integers(0, 5))).astype(np.float64 if rng.random() < 0.7 else np.int64)
    if rng.random() < 0.4 and p.size:
        p[0] = 0
    q = rng.integers(-3, 4, int(rng.integers(1, 4))).astype(np.float64)
    x = small_floats(rng, (4,))
    same({}, "", "polyval", p, x)
    same({}, "", "polyadd", p, q)
    same({}, "", "polysub", p, q)
    same({}, "", "polymul", p, q)
    same({}, "", "polyder", p, int(rng.integers(0, 3)))
    same({}, "", "polyint", p, int(rng.integers(0, 3)))
    if q[0] != 0:
        same({"order": 64}, "", "polydiv", p.astype(np.float64), q)
    check({}, f"roots dtype {p.tolist()}", lambda: str(np.roots(p).dtype), lambda: str(numerical.roots(ours(p)).dtype))


def padding(rng):
    a = array(rng, shape(rng, 1, 2), kinds=("f8", "i8"), special=0.1)
    if a.dtype == np.int64:  # NumPy's statistics of int64 in float64 come back as the processor converts them
        a = np.clip(a, -2 ** 53, 2 ** 53)
    mode = str(rng.choice(["constant", "edge", "reflect", "symmetric", "wrap", "linear_ramp", "maximum", "minimum",
                           "mean", "median", "empty"]))
    width = int(rng.integers(0, 4)) if rng.random() < 0.5 else [(int(rng.integers(0, 3)), int(rng.integers(0, 3)))
                                                                 for _ in range(a.ndim)]
    if mode == "empty":
        check({}, f"pad {mode} shape", lambda: np.pad(a, width, mode).shape,
              lambda: numerical.pad(ours(a), width, mode).shape)
        return
    rule = {"ulps": 1} if mode in ("mean", "linear_ramp") else {}
    same(rule, "", "pad", a, width, mode)


def shaping(rng):
    a = array(rng, shape(rng, 1, 3), kinds=("f8", "i8", "b"), special=0.1)
    axis = axis_of(rng, a)
    same({}, "", "tile", a, tuple(int(r) for r in rng.integers(0, 3, int(rng.integers(1, 3)))))
    repeats = int(rng.integers(0, 3)) if rng.random() < 0.5 or axis is None else \
        rng.integers(0, 3, a.shape[axis]).astype(np.int64)
    same({}, "", "repeat", a, repeats, axis=axis)
    same({}, "", "roll", a, int(rng.integers(-5, 6)), axis=axis)
    same({}, "", "flip", a, axis=axis)
    if axis is not None:
        same({}, "", "diff", a if a.dtype != bool else a.astype(np.int64), int(rng.integers(0, 3)), axis=axis)
    same({}, "", "squeeze", a)
    same({}, "", "moveaxis", a, 0, -1)


def selections(rng):
    a = array(rng, shape(rng, 1, 3), kinds=("f8", "i8"), special=0.1)
    axis = axis_of(rng, a)
    extent = a.size if axis is None else a.shape[axis]
    index = rng.integers(-extent - 1, extent + 1, shape(rng, 0, 2)).astype(np.int64)
    for mode in ("raise", "wrap", "clip"):
        if mode == "wrap" and index.size and not extent:
            continue  # NumPy loops forever there (numpy#11677)
        same({}, "", "take", a, index, axis=axis, mode=mode)
    if a.size:
        same({}, "", "take_along_axis", a, np.argsort(a, axis=-1, kind="stable"), axis=-1)
    flat = a.ravel()
    at = sorted(int(k) for k in rng.integers(-flat.size, flat.size + 1, int(rng.integers(0, 3))))
    same({}, "", "insert", flat, at, 7)
    if flat.size:
        same({}, "", "delete", flat, [int(k) for k in rng.integers(-flat.size, flat.size, 2)])
    condition = rng.random(a.shape) < 0.5
    same({}, "", "where", condition, a, -a if a.dtype != np.int64 else a)
    same({}, "", "extract", condition, a)
    same({}, "", "compress", rng.random(int(rng.integers(0, 4))) < 0.5, a, axis=axis)
    same({}, "", "clip", a, -1, 1)


def calculus(rng):
    a = small_floats(rng, (int(rng.integers(1, 7)),))
    same({"ulps": 2}, "", "trapezoid", a, dx=float(rng.uniform(0.1, 2)))
    if a.size >= 2:
        same({}, "", "gradient", a, edge_order=int(rng.integers(1, 3)) if a.size >= 3 else 1)
    v = small_floats(rng, (int(rng.integers(1, 5)),))
    mode = str(rng.choice(["valid", "same", "full"]))
    terms = {"order": 64, "scale": float(np.abs(a).max() * np.abs(v).max())}  # NumPy's dot product: its BLAS's order
    same(terms, "", "correlate", a, v, mode)
    same(terms, "", "convolve", a, v, mode)


def ranges(rng):
    start, stop = float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5))
    num = int(rng.integers(0, 12))
    endpoint = bool(rng.random() < 0.7)
    same({}, "", "linspace", start, stop, num, endpoint=endpoint)
    same({"ulps": 2}, "", "logspace", start, stop, num, endpoint=endpoint)
    if start * stop > 0:
        same({"ulps": 2}, "", "geomspace", start, stop, num, endpoint=endpoint)
    same({}, "", "arange", float(np.round(start, 1)), float(np.round(stop, 1)), float(rng.choice([0.5, 1, 0.3])))
    m = int(rng.integers(0, 6))
    window = str(rng.choice(["hamming", "hanning", "blackman", "bartlett"]))
    same({"ulps": 4}, "", window, m)
    same({"ulps": 4}, "", "kaiser", m, float(rng.uniform(0, 10)))


def other_order(a):
    """The rule of a sum of `a`'s elements in another order than NumPy's: within the rounding errors of its terms,
    an overflow left to the order where a term is near the largest."""
    finite = np.abs(a[np.isfinite(a)]).astype(np.float64) if a.dtype.kind in "fc" else np.abs(a.astype(np.float64))
    largest = float(finite.max()) if finite.size else 0.0
    near = a.dtype.kind in "fc" and largest > float(np.finfo(a.dtype).max) / 16
    return {"order": 64, "scale": largest * max(a.size, 1), "zero_signs": True, "near_overflow": near}


def nan_functions(rng):
    a = array(rng, shape(rng, 1, 2), kinds=("f8", "f4", "i8"), special=0.4)
    axis = axis_of(rng, a)
    last = axis is None or axis in (-1, a.ndim - 1) or a.ndim == 1
    sums = {} if last else other_order(a)
    for name in ("nansum", "nanprod"):
        same(sums, "", name, a, axis=axis)
    for name in ("nancumsum", "nancumprod"):
        same(other_order(a), "", name, a, axis=axis)
    for name in ("nanargmax", "nanargmin"):
        same({}, "", name, a, axis=axis)
    same(EXTREME_ZEROS, "", "nanmin", a, axis=axis)
    same({"ulps": 2} if last else other_order(a), "", "nanstd", a, axis=axis, ddof=int(rng.integers(0, 2)))
    if a.dtype != np.float32 or NUMPY >= (2, 3):  # NumPy's partition picks one zero or another where it is a zero
        same({"overflow": True, "zero_signs": True}, "", "nanpercentile", a, float(rng.uniform(0, 100)), axis=axis)
    same({}, "", "nan_to_num", a)
    same({}, "", "nan_to_num", a, nan=-1.0, posinf=2.0, neginf=-2.0)


def uniques(rng):
    a = array(rng, shape(rng, 1, 2), kinds=("f8", "i8", "b"), special=0.3)
    if a.size:
        a = a.ravel()[rng.integers(0, a.size, a.size)].reshape(a.shape)  # repeated values
    rule = {"zero_signs": True}
    for name in ("unique_counts", "unique_all"):
        check(rule, f"{name}{_shown((a,), {})}", lambda: tuple(np.asarray(part) for part in
                                                                _named(getattr(np, name)(a))),
              lambda: tuple(_named(getattr(numerical, name)(ours(a)))))
    # NumPy's unique_inverse sorts unstably: which of the NaNs (each its own value) an index names is arbitrary
    check(rule, f"unique_inverse{_shown((a,), {})}", lambda: _nan_blind(*np.unique_inverse(a)),
          lambda: _nan_blind(*(np.asarray(part) for part in numerical.unique_inverse(ours(a)))))
    # NumPy 2.3's unique_values hashes: its order is unspecified
    check(rule, f"unique_values{_shown((a,), {})}", lambda: np.sort(np.unique_values(a)),
          lambda: numerical.unique_values(ours(a)))
    if a.ndim == 2 and a.size and rng.random() < 0.5:
        same(rule, "", "unique", a, axis=int(rng.integers(0, 2)), return_counts=True)
    same({}, "", "trim_zeros", a.ravel(), str(rng.choice(["f", "b", "fb"])))
    same({}, "", "ediff1d", a.ravel() if a.dtype != bool else a.ravel().astype(np.int64))


def _nan_blind(values, inverse):
    values, inverse = np.asarray(values), np.asarray(inverse).astype(np.int64)
    if values.dtype.kind in "fc" and inverse.size:
        inverse = np.where(np.isnan(values[inverse]), -1, inverse)
    return values, inverse


def _named(answer):
    return tuple(answer) if isinstance(answer, tuple) else (answer,)


def indexing(rng):
    dims = tuple(int(e) for e in rng.integers(1, 5, int(rng.integers(1, 4))))
    flat = rng.integers(-1, int(np.prod(dims)) + 1, int(rng.integers(0, 6))).astype(np.int64)
    same({}, "", "unravel_index", flat, dims)
    coordinates = tuple(rng.integers(-1, e + 1, 3).astype(np.int64) for e in dims)
    mode = str(rng.choice(["raise", "wrap", "clip"]))
    same({}, "", "ravel_multi_index", coordinates, dims, mode=mode)
    same({}, "", "indices", dims)
    a = array(rng, dims, kinds=("f8", "i8", "b"), special=0.4)
    same({}, "", "argwhere", a)
    k = int(rng.integers(-2, 3))
    m = a.reshape(dims[0], -1)
    for name in ("tril", "triu"):
        same({}, "", name, m, k)
    same({}, "", "diagflat", a.ravel()[:4], k)
    same({}, "", "vander", a.ravel()[:4] if a.dtype != bool else a.ravel()[:4].astype(np.int64),
         int(rng.integers(0, 4)), increasing=bool(rng.random() < 0.5))
    choices = [array(rng, dims, kinds=("f8",)) for _ in range(3)]
    which = rng.integers(-1, 4, dims).astype(np.int64)
    same({}, "", "choose", which, choices, mode=str(rng.choice(["raise", "wrap", "clip"])))
    same({}, "", "select", [rng.random(dims) < 0.3, rng.random(dims) < 0.5], choices[:2], -1.0)


def products(rng):
    a = array(rng, (int(rng.integers(0, 4)),), kinds=("f8", "i8", "c16"), special=0.2)
    b = array(rng, (int(rng.integers(0, 4)),), kinds=("f8", "i8"), special=0.2)
    same({}, "", "outer", a, b)
    same({}, "", "kron", a, b)
    u, v = array(rng, (3,), kinds=("f8",), special=0.2), array(rng, (3,), kinds=("f8",), special=0.2)
    same({}, "", "cross", u, v)
    grids = [array(rng, (int(rng.integers(0, 4)),), kinds=("f8",)) for _ in range(int(rng.integers(1, 4)))]
    check({}, "meshgrid", lambda: np.meshgrid(*grids, indexing="ij"),
          lambda: numerical.meshgrid(*[ours(g) for g in grids], indexing="ij"))


def comparisons(rng):
    dims = shape(rng, 0, 2)
    a = array(rng, dims, kinds=("f8", "f4", "i8"), special=0.4)
    b = array(rng, dims, kinds=("f8", "f4", "i8"), special=0.4)
    if rng.random() < 0.4:
        b = a.copy()
    for name in ("isclose", "allclose"):
        same({}, "", name, a, b, equal_nan=bool(rng.random() < 0.5))
    same({}, "", "array_equal", a, b, equal_nan=bool(rng.random() < 0.5) and a.dtype.kind == "f")
    for name in ("fmax", "fmin", "copysign", "heaviside", "nextafter", "hypot", "logaddexp", "logaddexp2"):
        if name in ("copysign", "nextafter", "heaviside", "hypot", "logaddexp", "logaddexp2") and \
                "i" in (a.dtype.kind, b.dtype.kind):
            continue
        rule = {"ulps": 2} if name.startswith("logaddexp") else EXTREME_ZEROS if name in ("fmax", "fmin") else {}
        same(rule, "", name, a, b)
    for name in ("signbit", "isneginf", "isposinf", "isreal", "iscomplex", "logical_not"):
        same({}, "", name, a)


def conversions(rng):
    a = array(rng, shape(rng, 0, 2), kinds=("f8", "f4", "i8", "b"), special=0.3)
    target = str(rng.choice(["float64", "float32", "complex128", "bool", "int64"]))
    if target == "int64" and a.dtype.kind == "f":
        a = np.where(np.isfinite(a) & (np.abs(a) < 2.0 ** 62), a, 0).astype(a.dtype)  # NumPy's: as the processor
    check({}, f"astype {target}{_shown((a,), {})}", lambda: a.astype(target),
          lambda: ours(a).astype(target))
    same({"ulps": 1, "overflow": True}, "", "around", a if a.dtype.kind == "f" else a.astype(np.float64),
         int(rng.integers(0, 3)))  # rounded exactly here


FAMILIES = {function.__name__: function for function in (
    integer_arithmetic, powers, rounding, sorting, searching, sets, statistics, quantiles, averages, reductions,
    histograms, interpolation, polynomials, padding, shaping, selections, calculus, ranges, nan_functions, uniques,
    indexing, products, comparisons, conversions)}


def run(seed, family, case):
    rng = np.random.default_rng([seed, zlib.crc32(family.encode()), case])
    try:
        FAMILIES[family](rng)
    except Failure as failure:
        return str(failure)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=int(time.time()))
    parser.add_argument("--cases", type=int, default=200, help="cases per family")
    parser.add_argument("--family", choices=sorted(FAMILIES), help="one family alone")
    parser.add_argument("--case", type=int, help="replay one case (with --family)")
    args = parser.parse_args()
    families = [args.family] if args.family else sorted(FAMILIES)
    cases = [args.case] if args.case is not None else range(args.cases)
    failures, started = 0, time.time()
    for family in families:
        for case in cases:
            failure = run(args.seed, family, case)
            if failure:
                failures += 1
                print(f"seed {args.seed} --family {family} --case {case}: {failure}")
    print(f"{len(families)} families x {len(cases)} cases ({COMPARED[0]} calls), seed {args.seed}, {failures} failure(s), "
          f"{time.time() - started:.0f} s, NumPy {np.__version__}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
