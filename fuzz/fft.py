"""Differential fuzzing of the Fourier transforms against NumPy (pocketfft), on every path.

Each case draws a length biased towards the boundaries of the transform's passes (powers of 2, 3 and 5, products of
the small radices, primes up to 61, which take a pass of their own, and past it, which take Bluestein's convolution),
a batch of lines along one axis of a tensor of one to three axes, a dtype, a point count that cuts or pads, a scaling
and a scale, and one of numerical.fft's transforms. It then checks:

- the answer against numpy.fft's, within a multiple of log2(n) times the dtype's epsilon times the answer's norm (two
  stable transforms each stay within a fraction of that of the exact one);
- the inverse of the answer against the input, within the same bound (the complex and the real pairs);
- each line of the batch against the same line transformed alone, and one thread against all of them, to the bit;
- the vector paths against one another, to the bit (NEON and the AMX; AVX2 and AVX-512: one set of bits);
- the input untouched.

Every path the processor runs is exercised in turn (portable, NEON, AVX2, AVX-512, the Apple AMX). A failure prints
the seed and the case, which reproduce it alone:

    python3 fuzz/fft.py --seed S --case C
"""
import argparse
import itertools
import sys
import time

import numpy as np

import numerical

SLACK = 8.0  # the multiple of log2(n)·ε·‖X‖ two stable transforms stay within of each other
PATHS = ("portable", "neon", "avx2", "avx512", "amx")
VECTOR_PATHS = ("neon", "avx2", "avx512", "amx")
SMALL_PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61)
LARGE_PRIMES = (67, 71, 97, 101, 127, 257, 1009, 4099)
SCALES = (1.0, 1.0, 1.0, 1e100, 1e-100)
SINGLE_SCALES = (1.0, 1.0, 1.0, 1e15, 1e-15)  # within float's range, sums of thousands of terms included
NORMS = ("backward", "ortho", "forward")
DTYPES = ("complex128", "complex64", "float64", "float32")
OPS = ("fft", "ifft", "rfft", "irfft", "hfft", "ihfft", "fft2", "fftn")


class Failure(Exception):
    pass


def check(condition, what):
    if not condition:
        raise Failure(what)


def length(rng, largest):
    """A length biased towards the passes' boundaries, at most `largest`."""
    kind = rng.integers(6)
    if kind == 0:
        n = 2 ** int(rng.integers(0, 15))
    elif kind == 1:
        n = (3 ** int(rng.integers(0, 6))) * (2 ** int(rng.integers(0, 8)))
    elif kind == 2:
        n = int(rng.choice(SMALL_PRIMES)) * int(rng.choice((1, 2, 4, 8, 3, 5, 16)))
    elif kind == 3:
        n = int(rng.choice(LARGE_PRIMES)) * int(rng.choice((1, 1, 2, 3)))
    elif kind == 4:
        n = (5 ** int(rng.integers(0, 4))) * (7 ** int(rng.integers(0, 3))) * (2 ** int(rng.integers(0, 6)))
    else:
        n = int(rng.integers(1, largest + 1))
    return max(1, min(n, largest))


def values(rng, shape, dtype, scale):
    real = rng.standard_normal(shape) * scale
    if np.dtype(dtype).kind == "c":
        return (real + 1j * rng.standard_normal(shape) * scale).astype(dtype)
    return real.astype(dtype)


def case(seed, index, largest):
    """One case: the transform, its input, and its arguments."""
    rng = np.random.default_rng([seed, index])
    op = str(rng.choice(OPS))
    dtype = str(rng.choice(DTYPES))
    if op in ("irfft", "hfft"):
        dtype = {"float64": "complex128", "float32": "complex64"}.get(dtype, dtype)  # a spectrum: complex
    if op in ("rfft", "ihfft"):
        dtype = {"complex128": "float64", "complex64": "float32"}.get(dtype, dtype)  # a signal: real
    n = length(rng, largest)
    rank = int(rng.integers(1, 4)) if op not in ("fft2",) else int(rng.integers(2, 4))
    axis = int(rng.integers(rank))
    shape = [max(1, int(rng.integers(1, 9))) for _ in range(rank)]
    shape[axis] = n
    if op in ("fft2", "fftn"):
        shape = [min(s, 64) if a != axis else s for a, s in enumerate(shape)]
        while int(np.prod(shape)) > 1 << 18:
            shape = [max(1, s // 2) for s in shape]
    points = None if rng.random() < 0.6 else max(1, n + int(rng.integers(-n // 2, n // 2 + 3)))
    if op in ("fft2", "fftn"):
        points = None  # their shapes, capped above, transformed whole
    norm = str(rng.choice(NORMS))
    scale = float(rng.choice(SINGLE_SCALES if dtype in ("complex64", "float32") else SCALES))
    x = values(rng, tuple(shape), dtype, scale)
    label = f"{op} {dtype} shape {tuple(shape)} axis {axis} n {points} norm {norm} scale {scale:g}"
    return op, x, axis, points, norm, label


def call(module, op, x, axis, points, norm):
    """`op` of `module` (numerical.fft or numpy.fft) over `x`."""
    fn = getattr(module, op)
    if op in ("fft2", "fftn"):
        return fn(x, norm=norm)
    return fn(x, points, axis, norm)


def inverse_of(op):
    return {"fft": "ifft", "ifft": "fft", "rfft": "irfft", "irfft": "rfft", "hfft": "ihfft", "ihfft": "hfft",
            "fft2": "ifft2", "fftn": "ifftn"}[op]


def epsilon(dtype):
    return float(np.finfo(np.dtype(dtype)).eps)


def bound(answer, n, dtype):
    """SLACK·log2(n)·ε·‖answer‖, in the answer's precision."""
    size = float(np.linalg.norm(np.asarray(answer, dtype=complex)))
    return SLACK * max(1.0, np.log2(max(n, 2))) * epsilon(dtype) * size + np.finfo(float).tiny


def distance(got, want):
    got, want = np.asarray(got, dtype=complex), np.asarray(want, dtype=complex)
    if got.shape != want.shape:
        return np.inf
    return float(np.linalg.norm(got - want))


def same_bits(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return got.shape == want.shape and got.dtype == want.dtype and got.tobytes() == want.tobytes()


def run_case(seed, index, largest):
    op, x, axis, points, norm, label = case(seed, index, largest)
    before = x.copy()
    precision = np.result_type(x.dtype, np.complex64).type(0).real.dtype
    longest = max(x.shape) if points is None else max(max(x.shape), points)
    failures = []

    def attempt(name, body):
        try:
            body()
        except Failure as failure:
            failures.append(f"{name}: {failure}")
        except Exception as error:  # an exception NumPy does not raise on the same input is a failure too
            failures.append(f"{name}: {type(error).__name__}: {error}")

    try:
        theirs = call(np.fft, op, x, axis, points, norm)
    except Exception as refusal:  # NumPy refuses the call (no points): NumericAL must refuse it alike
        try:
            call(numerical.fft, op, x, axis, points, norm)
        except type(refusal):
            return label, failures, None
        except Exception as error:
            return label, [f"refusal: NumPy raised {type(refusal).__name__}, NumericAL {type(error).__name__}"], None
        return label, [f"refusal: NumPy raised {type(refusal).__name__}, NumericAL answered"], None
    ours = np.asarray(call(numerical.fft, op, x, axis, points, norm))

    def against_numpy():
        gap, limit = distance(ours, theirs), bound(theirs, longest, precision)
        check(gap <= limit, f"‖ours − NumPy's‖ {gap:.3g} past {limit:.3g}")

    def round_trip():
        if op in ("fft2", "fftn") or points is not None:
            return  # a cut or padded transform has no inverse
        size = x.shape[axis]
        if op in ("irfft", "hfft"):
            return  # their inputs' imaginary parts at the ends are read as 0: no inverse of an arbitrary input
        inverse = getattr(numerical.fft, inverse_of(op))
        back = np.asarray(inverse(ours, size, axis, norm) if op in ("rfft", "ihfft") else inverse(ours, None, axis, norm))
        gap, limit = distance(back, x), bound(x, longest, precision)
        check(gap <= limit, f"the inverse misses the input by {gap:.3g}, past {limit:.3g}")

    def lines_alone():
        if op in ("fft2", "fftn") or x.ndim == 1:
            return
        moved = np.moveaxis(x, axis, -1)
        answer = np.moveaxis(ours, axis, -1)
        flat_in = moved.reshape(-1, moved.shape[-1])
        flat_out = answer.reshape(-1, answer.shape[-1])
        for line in sorted({0, flat_in.shape[0] - 1, flat_in.shape[0] // 2}):
            alone = np.asarray(call(numerical.fft, op, np.ascontiguousarray(flat_in[line]), -1, points, norm))
            check(same_bits(alone, np.ascontiguousarray(flat_out[line])), f"line {line} differs from its transform alone")

    def threads():
        every = numerical.get_num_threads()
        try:
            numerical.set_num_threads(1)
            alone = np.asarray(call(numerical.fft, op, x, axis, points, norm))
        finally:
            numerical.set_num_threads(every)
        check(same_bits(alone, ours), f"one thread and {every} differ")

    def untouched():
        check(same_bits(x, before), "the input was written")

    for name, body in (("numpy", against_numpy), ("round trip", round_trip), ("lines", lines_alone),
                       ("threads", threads), ("input", untouched)):
        attempt(name, body)
    return label, failures, ours


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=int(time.time()))
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--case", type=int, help="run this one case only")
    parser.add_argument("--largest", type=int, default=5000, help="the longest axis drawn")
    args = parser.parse_args()
    paths = [path for path in PATHS if numerical.limit_gemm_path(path) == path]
    indices = [args.case] if args.case is not None else range(args.cases)
    print(f"seed {args.seed}, {len(indices)} cases up to {args.largest} points, paths {', '.join(paths)}", flush=True)
    failed = 0
    started = time.time()
    vector_answers = {}  # case → (path, answer) of the first vector path: the others must match it to the bit
    with np.errstate(all="ignore"):
        for path, index in itertools.product(paths, indices):
            numerical.limit_gemm_path(path)
            label, failures, answer = run_case(args.seed, index, args.largest)
            if path in VECTOR_PATHS and answer is not None:
                first = vector_answers.setdefault(index, (path, answer))
                if not same_bits(answer, first[1]):
                    failures.append(f"paths: {path}'s bits differ from {first[0]}'s")
            for failure in failures:
                failed += 1
                print(f"FAIL [{path}] case {index} ({label}) {failure}\n"
                      f"     repro: python3 fuzz/fft.py --seed {args.seed} --case {index} --largest {args.largest}",
                      flush=True)
    numerical.limit_gemm_path("amx")
    print(f"{failed} failure(s) in {time.time() - started:.0f} s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
