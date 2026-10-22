"""Differential fuzzing of NumericAL's element-wise operations and reductions against NumPy, on the values and shapes
that break them: signed zeros, infinities, NaNs, values near overflow and subnormal ones, empty axes, and extents
past a vector's or a task's width.

Each case draws a dtype, a shape (rank 0 to 4, an axis of 0 now and then, a long one now and then) and elements from a
palette weighted towards those values, then an operation and, for a reduction, an axis. The answers must agree with
NumPy's: to the bit where both compute the same expression per element (arithmetic, max and min, cross, outer, the
real square root), the sign of a zero and the NaN taken into account (two NaNs agree; on x86-64, NumPy's max and min
leave a zero's sign to their loops' order, so there a zero extreme agrees with either); within a tolerance scaled by
the terms' magnitudes where the order of a sum may differ (sums, means, norms, dot products), the pattern of NaNs and
infinities exact unless a value near overflow makes an overflow depend on that order, or unless NumPy's complex dot
product is its BLAS's, which may make a part NaN where a term is infinite. NumPy on Windows halves the imaginary part
of a complex square root it scaled against overflow (npymath's), which is undone before comparing. A complex number's
elementary functions are glibc's in NumericAL: to the bit under glibc, elsewhere (NumPy takes the platform's library)
within a few dozen units in the last place where both answers are finite. Where NumPy
raises, NumericAL must raise too.

A failure prints the seed and the case, which reproduce it alone:

    python3 fuzz/edges.py --seed S --case C

`--huge` runs, alone, the checks too large for every push (a float32 mean of 2^24 + 1 elements, sums whose result
would overflow std::size_t).
"""
import argparse
import platform
import sys
import time
import warnings

import numpy as np

import numerical

DTYPES = (np.float64, np.float32, np.complex128, np.complex64)
EXTENTS = (0, 1, 1, 2, 2, 3, 3, 4, 5, 7, 8, 9, 16, 17, 33)
LONG_EXTENTS = (65, 130, 257, 1031)  # a vector's lanes and an element-wise task's rows past these
# NumPy's max and min give +0 and -0 where both zeros are among the elements on arm64, as NumericAL's on any processor
# (IEEE 754-2019's maximum and minimum); on x86-64, the sign of a zero as its vector loops meet them.
NUMPY_SETTLES_ZEROS = platform.machine().lower() in ("arm64", "aarch64")
# NumPy on Windows takes a complex square root from npymath, which divides a part past max / (1 + sqrt(2)) by 4 and
# then doubles the real part of the root alone: its imaginary part comes out halved.
NUMPY_HALVES_SCALED_ROOTS = platform.system() == "Windows"
GLIBC = platform.libc_ver()[0] == "glibc"  # NumPy's complex functions are glibc's there, as NumericAL's everywhere


def real_type(dtype):
    return np.float32 if dtype in (np.float32, np.complex64) else np.float64


def palette(dtype):
    """The special values of `dtype`'s parts: zeros of either sign, infinities, a NaN, the largest and the smallest
    normal and a subnormal of either sign, and small integers, whose sums are exact."""
    info = np.finfo(real_type(dtype))
    big, tiny = float(info.max), float(info.tiny)
    return [0.0, -0.0, np.inf, -np.inf, np.nan, big, -big, big / 3, tiny, -tiny, tiny / 4, -tiny / 4,
            1.0, -1.0, 2.0, 0.5, 3.0]


def draw_parts(rng, count, dtype, special):
    values = palette(dtype)
    normal = rng.standard_normal(count) * np.exp(rng.uniform(-3, 3, count))
    chosen = np.array([values[i] for i in rng.integers(0, len(values), count)], dtype=float)
    return np.where(rng.random(count) < special, chosen, normal)


def draw_array(rng, shape, dtype, special):
    count = int(np.prod(shape)) if shape else 1
    parts = draw_parts(rng, count, dtype, special)
    if np.issubdtype(dtype, np.complexfloating):
        real = parts
        parts = np.empty(count, complex)  # each part set apart: 1j * inf would be nan + inf j
        parts.real, parts.imag = real, draw_parts(rng, count, dtype, special)
    with np.errstate(all="ignore"):
        return parts.astype(dtype).reshape(shape)


def draw_shape(rng, rank):
    shape = [int(rng.choice(EXTENTS)) for _ in range(rank)]
    if rank and rng.random() < 0.15:
        shape[int(rng.integers(rank))] = int(rng.choice(LONG_EXTENTS))
    return tuple(shape)


def broadcastable(rng, shape):
    """A shape that broadcasts against `shape`: some axes 1, some dropped from the front."""
    other = [1 if rng.random() < 0.3 else extent for extent in shape]
    return tuple(other[int(rng.integers(0, len(other) + 1)):]) if other else ()


def parts_equal(a, b):
    """To the bit, but NaNs alike: the same values, each zero of the same sign."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    for x, y in ((np.real(a), np.real(b)), (np.imag(a), np.imag(b))):
        both_nan = np.isnan(x) & np.isnan(y)
        same = (x == y) & (np.signbit(x) == np.signbit(y))
        if not np.all(both_nan | same):
            return False
    return True


def equal_but_zero_signs(a, b):
    """`parts_equal`, a zero of either sign agreeing with a zero."""
    a, b = np.asarray(a), np.asarray(b)
    return parts_equal(np.where(a == 0, 0, a).astype(a.dtype), np.where(b == 0, 0, b).astype(b.dtype))


def with_blas_nans(compare, a, b):
    """`compare`, a NaN part of NumPy's complex dot product along the last axis taken for NumericAL's where a product
    of two parts is infinite (an infinite part, or an overflow): NumPy's answer there is its BLAS's, whose kernels may
    make either part of the result NaN from a product of the other's (OpenBLAS's on x86-64 and arm64), where NumericAL
    sums each part's terms."""
    with np.errstate(all="ignore"):
        products = [x * y for x in (a.real, a.imag) for y in (b.real, b.imag)]  # each in the parts' precision
    infinite = np.zeros(np.broadcast_shapes(a.shape, b.shape)[:-1], bool)
    for product in products:
        infinite |= np.any(np.isinf(product), axis=-1)

    def compare_with(got, want):
        got, want = np.asarray(got), np.array(want)
        if got.shape != want.shape or not np.iscomplexobj(want):
            return compare(got, want)
        for g, w in ((got.real, want.real), (got.imag, want.imag)):
            taken = np.isnan(w) & ~np.isnan(g) & infinite
            w[taken] = g[taken]
        return compare(got, want)

    return compare_with


def numpy_root(a):
    """NumPy's square root of `a`, its halved imaginary parts doubled where it has them (`NUMPY_HALVES_SCALED_ROOTS`)."""
    root = np.sqrt(a)
    if not NUMPY_HALVES_SCALED_ROOTS or not np.iscomplexobj(a):
        return root
    threshold = np.finfo(real_type(a.dtype)).max / (1 + np.sqrt(2.0))
    finite = np.isfinite(a.real) & np.isfinite(a.imag)
    scaled = finite & ((np.abs(a.real) >= threshold) | (np.abs(a.imag) >= threshold))
    return np.where(scaled, root.real + 2j * root.imag, root).astype(root.dtype)


def finite_close(tol):
    """Where both answers are finite, within `tol` of the larger part's magnitude; where either is not, either: NumPy
    takes the platform's library outside glibc, which chooses otherwise among infinities and NaNs."""
    def compare(got, want):
        got, want = np.asarray(got), np.asarray(want)
        if got.shape != want.shape:
            return False
        with np.errstate(all="ignore"):
            finite = np.isfinite(got) & np.isfinite(want)
            scale = np.maximum(np.abs(np.real(want)), np.abs(np.imag(want)))
            off = np.maximum(np.abs(np.real(want) - np.real(got)), np.abs(np.imag(want) - np.imag(got)))
            return bool(np.all(off[finite] <= tol * np.maximum(scale[finite], np.finfo(float).tiny)))
    return compare


def numpy_norm(a, order, axis):
    """NumPy's vector norm; the largest magnitude of no element 0, as NumPy has it since 2.3 (2.2 raised)."""
    try:
        return np.linalg.vector_norm(a, ord=order, axis=axis)
    except ValueError:
        if order != np.inf:
            raise
        return np.max(np.abs(a), axis=axis, initial=0.0).astype(real_type(a.dtype))


def numpy_power(a, exponent):
    """NumPy's `a ** exponent`, the halved imaginary part of a scaled square root undone where it has one."""
    return numpy_root(a) if exponent == 0.5 and type(exponent) is float else a**exponent


def parts_close(a, b, scale, tol, near_overflow):
    """Within `tol` times `scale` (the terms' magnitude, elementwise), the NaNs and infinities where NumPy has them;
    an infinity or a NaN on one side only where values near overflow may overflow in one order and not another."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    scale = np.broadcast_to(np.asarray(scale, dtype=float), a.shape)
    scale = np.where(np.isnan(scale), np.inf, scale)  # magnitudes overflowed to inf, then times 0: terms past 1e308
    for x, y in ((np.real(a), np.real(b)), (np.imag(a), np.imag(b))):
        finite = np.isfinite(x) & np.isfinite(y)
        special_same = (np.isnan(x) & np.isnan(y)) | ((x == y) & ~np.isnan(x))
        within = np.zeros(x.shape, bool)
        with np.errstate(all="ignore"):
            # equal ones agree whatever the scale (an infinite or NaN one where a term was 0 · inf)
            bound = tol * np.maximum(scale[finite], np.finfo(float).tiny)
            within[finite] = (x[finite] == y[finite]) | (np.abs(x[finite] - y[finite]) <= bound)
        ok = np.where(finite, within, special_same)
        if near_overflow:
            ok = ok | ~(np.isfinite(x) & np.isfinite(y))
        if not np.all(ok):
            return False
    return True


def magnitude(array):
    """|re| + |im| of each element, a NaN's or an infinity's part counted 0: the scale of a sum's terms, which one
    special element elsewhere in its line does not make NaN."""
    array = np.asarray(array)
    with np.errstate(all="ignore"):
        return (np.abs(np.nan_to_num(np.real(array).astype(float), nan=0.0, posinf=0.0, neginf=0.0))
                + np.abs(np.nan_to_num(np.imag(array).astype(float), nan=0.0, posinf=0.0, neginf=0.0)))


def norm_close(got, want, scale, tol, operands):
    """parts_close, but for the divergences vector_norm documents: summed at the scale of the largest magnitude it
    stays finite where NumPy's sum of squares overflows to inf or underflows to 0; and an element with an infinite
    part has an infinite magnitude (C's hypot), where NumPy's x·conj(x) may make it NaN."""
    got, want = np.asarray(got), np.asarray(want)
    if got.shape != want.shape:
        return False
    infinite_part = any(np.any(np.isinf(np.real(o)) | np.isinf(np.imag(o))) for o in operands)
    allowed = ((np.isfinite(got) & ((want == 0) | np.isinf(want)))
               | (np.isinf(got) & np.isnan(want) & infinite_part))
    if np.all(allowed | ~np.ones(got.shape, bool)):
        return True
    keep = ~allowed
    return parts_close(got[keep], want[keep], np.broadcast_to(np.asarray(scale, float), got.shape)[keep], tol, False)


def near_overflow(*arrays):
    for array in arrays:
        parts = np.abs(np.concatenate([np.real(array).ravel(), np.imag(array).ravel()]))
        finite = parts[np.isfinite(parts)]
        if finite.size and finite.max() > float(np.finfo(real_type(array.dtype.type)).max) / 1e6:
            return True
    return False


def check(label, numpy_call, numerical_call, compare):
    """[] where they agree, else the difference; NumPy raising means NumericAL must raise too."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # NumPy's mean of an empty slice, its NaN the answer checked
        try:
            want = numpy_call()
        except (ValueError, IndexError, TypeError, np.linalg.LinAlgError) as raised:
            try:
                numerical_call()
            except Exception:
                return []
            return [f"{label}: NumPy raised ({type(raised).__name__}), NumericAL answered"]
        try:
            got = np.asarray(numerical_call())
        except Exception as raised:
            return [f"{label}: NumericAL raised {type(raised).__name__}: {raised}"]
    if not compare(got, want):
        return [f"{label}: {difference(got, want)}"]
    return []


def difference(got, want):
    """The first elements that differ (to the bit, NaNs alike), or the shapes and dtypes."""
    got, want = np.asarray(got), np.asarray(want)
    if got.shape != want.shape:
        return f"shape {got.shape} against NumPy's {want.shape}"
    flat_got, flat_want = got.ravel(), want.ravel()
    shown = []
    for k in range(flat_got.size):
        if not parts_equal(flat_got[k:k + 1], flat_want[k:k + 1]):
            shown.append(f"[{np.unravel_index(k, got.shape)}] {flat_got[k]!r} against {flat_want[k]!r}")
        if len(shown) == 3:
            break
    return f"{got.dtype} against {want.dtype}: " + "; ".join(shown)


def run_case(seed, index, largest=None):
    """One case: (its label, the failures)."""
    with np.errstate(all="ignore"):  # the scales' own overflows
        return drawn_case(seed, index, largest)


def drawn_case(seed, index, largest):
    rng = np.random.default_rng([seed, index])
    dtype = DTYPES[int(rng.integers(len(DTYPES)))]
    eps = float(np.finfo(real_type(dtype)).eps)
    special = float(rng.choice([0.0, 0.1, 0.3, 0.7]))
    rank = int(rng.integers(0, 5))
    shape = draw_shape(rng, rank)
    a = draw_array(rng, shape, dtype, special)
    t = numerical.asarray(a)
    op = rng.choice(["add", "subtract", "multiply", "divide", "negate", "max", "min", "sum", "mean", "norm",
                     "absolute", "sqrt", "power", "elementary", "cross", "outer", "vecdot", "trace", "einsum"])
    label = f"{op} {np.dtype(dtype).name} {shape} special={special}"
    exact = parts_equal

    operands = [a]  # every operand drawn: an overflow depends on all of them

    def tolerant(scale, factor=8.0):
        return lambda got, want: parts_close(got, want, scale, factor * eps * max(1, a.size), near_overflow(*operands))

    if op in ("add", "subtract", "multiply", "divide"):
        b = draw_array(rng, broadcastable(rng, shape), dtype, special)
        operands.append(b)
        nb = numerical.asarray(b)
        numpy_op = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply, "divide": np.divide}[op]
        nl_op = {"add": lambda x, y: x + y, "subtract": lambda x, y: x - y, "multiply": lambda x, y: x * y,
                 "divide": lambda x, y: x / y}[op]
        label += f" with {b.shape}"
        return label, check(label, lambda: numpy_op(a, b), lambda: nl_op(t, nb), exact)  # complex ones NumPy's loops
    if op == "negate":
        return label, check(label, lambda: -a, lambda: -t, exact)
    if op in ("max", "min", "sum", "mean"):
        axis = None if rank == 0 or rng.random() < 0.4 else int(rng.integers(rank))
        label += f" axis={axis}"
        if np.issubdtype(dtype, np.complexfloating) and op in ("max", "min"):
            return label, []  # NumPy orders complex numbers lexicographically; NumericAL's max takes real ones
        numpy_op = {"max": np.max, "min": np.min, "sum": np.sum, "mean": np.mean}[op]
        nl_op = {"max": numerical.max, "min": numerical.min, "sum": numerical.sum, "mean": numerical.mean}[op]
        if op in ("max", "min"):
            compare = exact if NUMPY_SETTLES_ZEROS else equal_but_zero_signs
        else:
            with np.errstate(all="ignore"):
                count = 1
                if op == "mean":
                    count = (a.size if axis is None else a.shape[axis]) or 1
                scale = np.sum(magnitude(a), axis=axis) / count
            compare = tolerant(scale)
        return label, check(label, lambda: numpy_op(a, axis=axis), lambda: nl_op(t, axis=axis), compare)
    if op == "norm":
        order = rng.choice([2.0, 1.0, np.inf, -np.inf, 0.0, 3.0])
        axis = None if rank == 0 or rng.random() < 0.5 else int(rng.integers(rank))
        label += f" ord={order} axis={axis}"
        with np.errstate(all="ignore"):
            scale = np.sum(magnitude(a), axis=axis) if order not in (0.0, -np.inf) else 1.0
        return label, check(label, lambda: numpy_norm(a, order, axis),
                            lambda: numerical.vector_norm(t, ord=float(order), axis=axis),
                            lambda got, want: norm_close(got, want, scale, 16 * eps * max(1, a.size), operands))
    if op == "absolute":
        compare = exact if not np.issubdtype(dtype, np.complexfloating) else tolerant(np.abs(a), 2.0)
        return label, check(label, lambda: np.abs(a), lambda: numerical.absolute(t), compare)
    if op == "sqrt":
        compare = exact if not np.issubdtype(dtype, np.complexfloating) else tolerant(np.sqrt(np.abs(a)), 4.0)
        return label, check(label, lambda: numpy_root(a), lambda: numerical.sqrt(t), compare)
    if op == "elementary":
        name = ["exp", "log", "sin", "cos", "tan", "tanh"][int(rng.integers(6))]
        label += f" {name}"
        compare = exact
        if np.issubdtype(dtype, np.complexfloating):
            compare = exact if GLIBC else finite_close(64.0 * eps)  # NumPy's complex ones are the C library's
        else:
            compare = finite_close(4.0 * eps)  # NumPy's real ones its own vector loops, a few units from the library's
        return label, check(label, lambda: getattr(np, name)(a), lambda: getattr(numerical, name)(t), compare)
    if op == "power":
        exponent = [2, -1, 0.5, 2.0, -1.0, 3, -2, 5, -7][int(rng.integers(9))]
        label += f" ** {exponent!r}"
        compare = exact
        if exponent == 0.5 and np.issubdtype(dtype, np.complexfloating) and not GLIBC:
            compare = tolerant(np.sqrt(np.abs(a)), 4.0)  # NumPy's root is the C library's: glibc's is NumericAL's
        elif not np.issubdtype(dtype, np.complexfloating) and exponent not in (2, -1, 0.5, 2.0, -1.0):
            compare = tolerant(np.abs(a) ** exponent, 4.0)  # the C library's pow, NumPy's and std::pow alike
        return label, check(label, lambda: numpy_power(a, exponent), lambda: t**exponent, compare)
    if op == "cross":
        b = draw_array(rng, shape, dtype, special)
        operands.append(b)
        if rank == 0 or 3 not in shape:
            return label, []
        axis = int(np.where(np.array(shape) == 3)[0][0])
        label += f" axis={axis}"
        nb = numerical.asarray(b)
        compare = exact if not np.issubdtype(dtype, np.complexfloating) else tolerant(1.0)
        return label, check(label, lambda: np.linalg.cross(a, b, axis=axis), lambda: numerical.cross(t, nb, axis=axis),
                            compare if compare is exact else lambda got, want: parts_close(
                                got, want, np.abs(want) + 1.0, 16 * eps, near_overflow(*operands)))
    if op == "outer":
        x, y = a.ravel()[:40], draw_array(rng, (int(rng.choice(EXTENTS)),), dtype, special)
        operands.append(y)
        compare = exact if not np.issubdtype(dtype, np.complexfloating) else tolerant(np.abs(np.outer(x, y)), 2.0)
        return label, check(label, lambda: np.outer(x, y),
                            lambda: numerical.outer(numerical.asarray(x), numerical.asarray(y)), compare)
    if op == "vecdot":
        if rank == 0:
            return label, []
        b = draw_array(rng, shape, dtype, special)
        operands.append(b)
        with np.errstate(all="ignore"):
            scale = np.sum(magnitude(a) * magnitude(b), axis=-1)
        return label, check(label, lambda: np.linalg.vecdot(a, b), lambda: numerical.vecdot(t, numerical.asarray(b)),
                            with_blas_nans(tolerant(scale), a, b))
    if op == "trace":
        if rank < 2:
            return label, []
        with np.errstate(all="ignore"):
            scale = np.trace(magnitude(a), axis1=-2, axis2=-1)
        return label, check(label, lambda: np.linalg.trace(a), lambda: numerical.trace(t), tolerant(scale))
    if rank == 0:  # einsum: one operand summed over one axis
        return label, []
    axis = int(rng.integers(rank))
    letters = "abcd"[:rank]
    spec = f"{letters}->{letters.replace(letters[axis], '')}"
    label += f" {spec}"
    with np.errstate(all="ignore"):
        scale = np.sum(magnitude(a), axis=axis)
    return label, check(label, lambda: np.einsum(spec, a), lambda: numerical.einsum(spec, t), tolerant(scale))


def huge_checks():
    """The shapes too large for every push: a float32 mean of 2^24 + 1 ones (64 MiB) to NumPy's bits, its count exact
    where a float would round it, and sums keeping extents whose product overflows refused, where a wrapped size made
    a buffer too small. Their failures, none when they hold."""
    failures = []
    ones = np.ones((1 << 24) + 1, np.float32)
    failures += check("mean of 2^24 + 1 float32 ones", lambda: np.mean(ones),
                      lambda: numerical.mean(numerical.asarray(ones)), parts_equal)
    failures += check("mean along an axis of 2^24 + 1", lambda: np.mean(ones.reshape(1, -1), axis=1),
                      lambda: numerical.mean(numerical.asarray(ones.reshape(1, -1)), axis=1), parts_equal)
    half = (sys.maxsize.bit_length() + 1) // 2  # extents whose product is past std::size_t
    huge = numerical.zeros((1 << half, 0, 1 << half))
    for name, call in (("sum", lambda: numerical.sum(huge, axis=1)), ("mean", lambda: numerical.mean(huge, axis=1)),
                       ("einsum", lambda: numerical.einsum("ijk->ik", huge))):
        try:
            call()
            failures.append(f"{name} keeping 2^{half} × 2^{half} elements answered")
        except numerical.error:
            pass
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=int(time.time()))
    parser.add_argument("--cases", type=int, default=2000)
    parser.add_argument("--case", type=int, help="replay one case")
    parser.add_argument("--huge", action="store_true", help="the large-shape checks alone (memory: about 100 MiB)")
    args = parser.parse_args()
    if args.huge:
        found = huge_checks()
        for failure in found:
            print(failure)
        print(f"large shapes: {len(found)} failure(s)")
        return 1 if found else 0
    indices = [args.case] if args.case is not None else range(args.cases)
    failures = 0
    started = time.time()
    for index in indices:
        label, found = run_case(args.seed, index)
        for failure in found:
            failures += 1
            print(f"seed {args.seed} case {index}: {failure}")
    print(f"{len(indices)} cases, seed {args.seed}, {failures} failure(s), {time.time() - started:.0f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
