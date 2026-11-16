"""Hostile arguments through the whole public namespace: every callable of numerical, numerical.linalg and
numerical.fft called with drawn values no caller should pass (None, huge and negative integers, NaN and infinities,
strings, bytes, ragged lists, absurd shapes, unknown dtypes, objects), positionally and by keyword.

The answer may be anything, and the error any of those a caller expects (TypeError, ValueError, LinAlgError,
IndexError, OverflowError, MemoryError, ZeroDivisionError, NotImplementedError and their kin). A case fails on an error
that escapes from the library's own code rather than reports the argument: SystemError (a C call made with an
exception set: full(shape, [1, 2]) raised it), AttributeError, KeyError, NameError, UnboundLocalError, AssertionError,
RecursionError; a crash of the interpreter fails the run itself. Shapes past a few thousand elements are drawn absurd
(negative, past 2⁶³) or not at all, so that no case allocates gigabytes.

Usage: python fuzz/hostile.py [--seed S] [--cases N]
"""
import argparse
import inspect
import math
import sys
import time
import warnings

import numpy as np

import numerical

ESCAPED = (SystemError, AttributeError, KeyError, NameError, UnboundLocalError, AssertionError, RecursionError)
SKIPPED = {"set_num_threads", "release_memory", "show_config", "get_include", "test"}  # state or the environment
# NumPy's own escapes, kept for parity: trim_zeros calls trim.lower() before any check, as NumPy's does
NUMPY_ALIKE = {("numerical.trim_zeros", "AttributeError")}


def callables():
    """(qualified name, callable) for each public function of the namespace and its submodules."""
    found = []
    for prefix, module in (("numerical", numerical), ("numerical.linalg", numerical.linalg),
                           ("numerical.fft", numerical.fft)):
        for name in sorted(getattr(module, "__all__", dir(module))):
            if name.startswith("_") or name in SKIPPED:
                continue
            value = getattr(module, name, None)
            if callable(value) and not inspect.isclass(value):
                found.append((f"{prefix}.{name}", value))
    return found


def hostile(rng, depth=0):
    """One drawn argument."""
    small = numerical.asarray(rng.standard_normal(int(rng.integers(0, 5))))
    choices = [
        None, 0, -1, 1, 2 ** 31, -(2 ** 63) - 1, 2 ** 64, 10 ** 30, math.nan, math.inf, -math.inf, 0.5, -0.0, 1j,
        True, "", "abc", "float64", "int8", "uint32", b"\x00", object(), (), [], [[1, 2], [3]], [1, "a"],
        [math.nan, math.inf], (-1, 2), (2 ** 62, 2 ** 62), (0, 0, 0), (3, -4), slice(None), Ellipsis,
        np.float32(1.5), np.int8(3), np.array([], dtype=np.int8), np.ones((2, 3), np.uint16), small,
        numerical.asarray([[1.0, 2.0], [3.0, 4.0]]), numerical.asarray(3.0), numerical.asarray([True, False]),
        numerical.zeros((0, 3)), numerical.asarray([1 + 2j, 3j]), numerical.asarray([2 ** 62, -5], dtype="int64"),
    ]
    if depth == 0 and rng.random() < 0.1:
        return [hostile(rng, 1) for _ in range(int(rng.integers(1, 3)))]
    return choices[int(rng.integers(len(choices)))]


def plausible(rng):
    """An argument a caller might well pass (a small shape, a tensor, a finite number, a known dtype): beside a hostile
    one it reaches the code past the first check."""
    choices = [2, 3, (2, 2), (2, 3), 1.5, -2.0, "float64", "complex128", numerical.asarray([[1.0, 2.0], [3.0, 5.0]]),
               numerical.asarray([1.0, -2.0, 3.0]), numerical.asarray([1, 2], dtype="int64"), [1.0, 2.0], 0]
    return choices[int(rng.integers(len(choices)))]


def required(function):
    """The count of positional arguments `function` requires, where its signature says (1 else)."""
    try:
        return sum(1 for p in inspect.signature(function).parameters.values()
                   if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty)
    except (TypeError, ValueError):
        return 1


def keywords(function):
    """The keyword names `function` takes, where its signature says."""
    try:
        return [p.name for p in inspect.signature(function).parameters.values()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    except (TypeError, ValueError):
        return []


def run(seed, cases):
    rng = np.random.default_rng(seed)
    functions = callables()
    failures = 0
    for case in range(cases):
        name, function = functions[int(rng.integers(len(functions)))]
        count = max(0, required(function) + int(rng.choice([-1, 0, 0, 0, 1])))
        args = [hostile(rng) if rng.random() < 0.5 else plausible(rng) for _ in range(count)]
        names = keywords(function)
        kwargs = {}
        if names and rng.random() < 0.4:
            key = names[int(rng.integers(len(names)))]
            kwargs[key] = hostile(rng)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                function(*args, **kwargs)
            except ESCAPED as error:
                if (name, type(error).__name__) in NUMPY_ALIKE:
                    continue
                failures += 1
                shown = ", ".join([repr(a)[:60] for a in args] + [f"{k}={v!r:.60}" for k, v in kwargs.items()])
                print(f"seed {seed} case {case}: {name}({shown}): {type(error).__name__}: {str(error)[:150]}")
            except BaseException as error:  # noqa: BLE001 (any other is the argument reported)
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cases", type=int, default=5000)
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else int(time.time())
    start = time.time()
    failures = run(seed, args.cases)
    print(f"{args.cases} calls, seed {seed}, {failures} failure(s), {time.time() - start:.0f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
