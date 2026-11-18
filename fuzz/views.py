"""Differential fuzzing of NumericAL's views against the same elements built contiguous: every operation on a view
gives the bits it gives on its copy.

A view reads its elements where they lie, and many operations take paths of their own for one: a transpose reduced,
scanned or sorted along the matching axis of its elements as laid out, two transposes alike paired as they lie, a
reshape that stays a view, a slice read row by row, a diagonal along its stride. Each path must give the bits of the
general one, which reads the elements copied out row-major.

Each case draws a dtype (float64, float32, complex128, complex64, int64, bool), a shape of rank 1 to 4 (most past
the 512 elements under which a slice is a copy of its own) and elements (NaNs now and then), then a chain of one to
three view steps (a permutation of the axes, `.T`, a slice with steps and reversals, a reshape, `expand_dims`,
`squeeze`, `diagonal`). The view's elements must be its copy's; then every operation of the dtype's list, along one
drawn axis, must give the copy's bits (two NaNs agree), or raise the same type of exception.

A failure prints the seed and the case, which reproduce it alone:

    python3 fuzz/views.py --seed S --case C
"""
import argparse
import faulthandler
import math
import random
import sys
import time

import numpy as np

import numerical as nm

DTYPES = ("float64", "float32", "complex128", "complex64", "int64", "bool")


def bits(x):
    """A result (a Tensor, a number, or a tuple of them) as comparable bytes, every NaN one NaN."""
    if isinstance(x, (tuple, list)):
        return tuple(bits(each) for each in x)
    a = np.asarray(x)
    if a.dtype.kind in "fc":
        a = np.where(np.isnan(a), np.nan, a)
    return a.dtype.str, a.shape, a.tobytes()


def outcome(run, x):
    """What `run(x)` gives: its bits, or the name of the exception it raises."""
    try:
        return bits(run(x))
    except Exception as error:  # noqa: BLE001 -- the error is the outcome compared
        return type(error).__name__


def elements(rng, dtype, shape):
    if dtype == "bool":
        return rng.random(shape) < 0.5
    if dtype == "int64":
        return rng.integers(-50, 50, shape)
    a = rng.standard_normal(shape)
    if rng.random() < 0.2:
        a[rng.random(shape) < 0.1] = np.nan
    if dtype.startswith("complex"):
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


def split(draw, count, rank):
    """`count` split into `rank` extents (each 1 or more), in a drawn order."""
    extents = []
    for _ in range(rank - 1):
        extent = draw.choice([d for d in range(1, count + 1) if count % d == 0])
        extents.append(extent)
        count //= extent
    extents.append(count)
    draw.shuffle(extents)
    return tuple(extents)


def view_step(draw, t):
    """One drawn view step of `t`: its name and the view."""
    shape = t.shape
    rank = len(shape)
    kind = draw.choice(["permute", "permute", "slice", "slice", "reshape", "reshape", "expand", "squeeze",
                        "diagonal", "T"])
    if kind == "T":
        return "T", t.T
    if kind == "permute" and rank >= 2:
        axes = list(range(rank))
        draw.shuffle(axes)
        return f"transpose{tuple(axes)}", nm.transpose(t, tuple(axes))
    if kind == "slice":
        key = []
        for extent in shape:
            pick = draw.random()
            if pick < 0.4 or extent < 2:
                key.append(slice(None))
            elif pick < 0.6:
                key.append(slice(None, None, -1))
            elif pick < 0.8:
                step = draw.choice([1, 2, 3, -2])
                key.append(slice(draw.randrange(extent), None, step) if step > 0 else slice(None, None, step))
            else:
                key.append(slice(draw.randrange(extent // 2 + 1), None))
        return f"[{', '.join(f'{k.start}:{k.stop}:{k.step}' for k in key)}]", t[tuple(key)]
    if kind == "reshape" and t.size > 0:
        into = split(draw, t.size, draw.randint(1, 4))
        return f"reshape{into}", nm.reshape(t, into)
    if kind == "expand":
        axis = draw.randint(0, rank)
        return f"expand_dims({axis})", nm.expand_dims(t, axis)
    if kind == "squeeze" and 1 in shape:
        return "squeeze", nm.squeeze(t)
    if kind == "diagonal" and rank >= 2 and min(shape[-2:]) > 0:
        offset = draw.randint(-1, 1)
        return f"diagonal({offset})", nm.diagonal(t, offset=offset)
    return "itself", t


def operations(draw, dtype, rank):
    """The operations compared on a view of `dtype` and `rank`, along one drawn axis."""
    real = dtype in ("float64", "float32")
    inexact = dtype not in ("int64", "bool")
    ax = draw.randrange(rank) - rank
    k = draw.randint(-2, 2)

    def along(run):  # a reduction that needs a line: none along an empty axis
        return lambda x: run(x) if x.shape[ax] else nm.sum(x, axis=ax)

    ops = [
        ("sum", nm.sum),
        ("any", nm.any),
        ("count_nonzero", nm.count_nonzero),
        ("tolist", lambda x: nm.asarray(x.tolist(), dtype=x.dtype)),
        ("through NumPy", lambda x: nm.asarray(np.asarray(x))),
        ("flattened", lambda x: nm.reshape(x, (-1,))),
        ("expanded and squeezed", lambda x: nm.squeeze(nm.expand_dims(x, 0), 0)),
        ("where of itself", lambda x: nm.where(x != 0, x, x)),
        ("array_equal", lambda x: nm.array_equal(x, x)),
        ("sum along", lambda x: nm.sum(x, axis=ax)),
        ("sum along, kept", lambda x: nm.sum(x, axis=ax, keepdims=True)),
        ("prod along", lambda x: nm.prod(x, axis=ax)),
        ("cumsum along", lambda x: nm.cumsum(x, axis=ax)),
        ("cumprod along", lambda x: nm.cumprod(x, axis=ax)),
        ("any along", lambda x: nm.any(x, axis=ax)),
        ("all along", lambda x: nm.all(x, axis=ax)),
        ("count_nonzero along", lambda x: nm.count_nonzero(x, axis=ax)),
        ("plus itself", lambda x: x + x),
        ("times its double transpose", lambda x: x * nm.transpose(nm.transpose(x))),
        ("triu", lambda x: nm.triu(x, k) if x.ndim >= 2 else nm.sum(x)),
        ("tril", lambda x: nm.tril(x, -1) if x.ndim >= 2 else nm.sum(x)),
    ]
    if dtype != "bool":
        ops += [
            ("max", lambda x: nm.max(x) if x.size else nm.sum(x)),
            ("max along", along(lambda x: nm.max(x, axis=ax))),
            ("min along", along(lambda x: nm.min(x, axis=ax))),
            ("mean along", lambda x: nm.mean(x, axis=ax)),
            ("plus 2", lambda x: x + 2),
            ("2 minus", lambda x: 2 - x),
        ]
    if dtype != "bool" and not dtype.startswith("complex"):
        ops += [
            ("sort along", lambda x: nm.sort(x, axis=ax)),
            ("argsort along", lambda x: nm.argsort(x, axis=ax)),
            ("argmax along", along(lambda x: nm.argmax(x, axis=ax))),
            ("argmin along", along(lambda x: nm.argmin(x, axis=ax))),
            ("less than 0", lambda x: x < 0),
            ("clip", lambda x: nm.clip(x, -1, 1)),
            ("median along", along(lambda x: nm.median(x, axis=ax))),
        ]
    if real:
        ops += [
            ("exp", nm.exp), ("expm1", nm.expm1), ("tanh", nm.tanh), ("floor", nm.floor),
            ("nanmax along", along(lambda x: nm.nanmax(nm.nan_to_num(x), axis=ax))),
            ("where positive", lambda x: nm.where(x > 0, x, 0.0)),
            ("where positive, a number", lambda x: nm.where(x > 0, 1.0, x)),
            ("maximum of a number", lambda x: nm.maximum(x, 0.25)),
            ("rfft along", along(lambda x: nm.fft.rfft(x, axis=ax))),
            ("dct along", lambda x: nm.fft.dct(x, axis=ax) if x.shape[ax] > 1 else nm.sum(x)),
        ]
    if inexact:
        ops += [
            ("fft along", along(lambda x: nm.fft.fft(x, axis=ax))),
            ("fftshift", nm.fft.fftshift),
            ("isnan", nm.isnan),
            ("nansum along", lambda x: nm.nansum(x, axis=ax)),
        ]
    if inexact and rank >= 2:
        ops += [
            ("matmul", lambda x: x @ nm.ones((x.shape[-1], 3), dtype=x.dtype)),
            ("times a vector", lambda x: x @ nm.ones((x.shape[-1],), dtype=x.dtype)),
        ]
    return ops


def run_case(seed, index):
    """Case `index` of `seed`: its failures."""
    draw = random.Random(seed * 1_000_003 + index)
    rng = np.random.default_rng([seed, index])
    dtype = draw.choice(DTYPES)
    rank = draw.randint(1, 4)
    shape = [draw.randint(1, 12) for _ in range(rank)]
    if draw.random() < 0.7:  # past 512 elements: a slice is a view, not a copy of its own
        while math.prod(shape) < 700:
            shape[draw.randrange(rank)] += draw.randint(1, 6)
    t = nm.asarray(elements(rng, dtype, tuple(shape)))
    chain, view = [], t
    try:
        for _ in range(draw.randint(1, 3)):
            name, view = view_step(draw, view)
            chain.append(name)
            if not isinstance(view, nm.Tensor):  # every axis squeezed out: a number, no view
                return []
    except Exception as error:  # noqa: BLE001 -- a step refused on the view and on its copy alike is no finding
        return [] if type(error).__name__ in ("ValueError", "IndexError") else [f"{dtype} {shape} {chain}: {error!r}"]
    built = nm.asarray(np.array(np.asarray(view)))  # the same elements, contiguous
    where = f"{dtype} {tuple(shape)} {' '.join(chain)} -> {view.shape}"
    if bits(view) != bits(built):
        return [f"{where}: the view's elements differ from its copy's"]
    if view.ndim == 0:
        return []
    failures = []
    for name, run in operations(draw, dtype, view.ndim):
        got, want = outcome(run, view), outcome(run, built)
        if got != want:
            failures.append(f"{where}, {name}: {got if isinstance(got, str) else 'bits'} where the copy gives "
                            f"{want if isinstance(want, str) else 'other bits'}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=int(time.time()))
    parser.add_argument("--cases", type=int, default=3000)
    parser.add_argument("--case", type=int, help="replay one case")
    args = parser.parse_args()
    indices = [args.case] if args.case is not None else range(args.cases)
    failures = 0
    started = time.time()
    faulthandler.enable()  # a crash prints its stack; the last case announced below brackets it
    for index in indices:
        if index % 250 == 0:
            print(f"seed {args.seed}: from case {index}", flush=True)
        for failure in run_case(args.seed, index):
            failures += 1
            print(f"seed {args.seed} case {index}: {failure}", flush=True)
    print(f"{len(indices)} cases, seed {args.seed}, {failures} failure(s), {time.time() - started:.0f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
