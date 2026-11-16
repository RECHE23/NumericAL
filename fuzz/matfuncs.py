"""Functions of matrices against a 60-digit reference (mpmath): expm, sqrtm, logm, sinhm, coshm, tanhm and funm on drawn
families where the defects of 2026-11 lay (a stiff triangle's exact band, eigenvalues gathered under one circle, a
quotient of exponentials that cancels), each answer's error against the reference beside SciPy's.

A case fails where NumericAL's relative error (Frobenius) is past both 1e-12 and ten times SciPy's, or its answer is not
finite where the reference is: the error a matrix's conditioning allows is SciPy's to show, and an answer past it by a
decade is NumericAL's defect (exp of eigenvalues 0, −30 and −500 under one circle: 10¹⁰ against SciPy's 10⁻¹⁸; a gap
of 1500 in a triangle: NaN). Usage: python fuzz/matfuncs.py [--seed S] [--cases N] [--case K]
"""
import argparse
import sys
import time
import warnings

import mpmath
import numpy as np
import scipy.linalg as sl

import numerical as nm

mpmath.mp.dps = 60


def reference(name, a):
    """The function of `a` at 60 digits, rounded to complex128: mpmath's own where it has one, the exponentials' sums
    and quotient otherwise (no cancellation at that precision)."""
    m = mpmath.matrix(a.tolist())
    if name in ("expm", "funm"):
        r = mpmath.expm(m)
    elif name == "sqrtm":
        r = mpmath.sqrtm(m)
    elif name == "logm":
        r = mpmath.logm(m)
    else:
        e, f = mpmath.expm(m), mpmath.expm(-m)
        if name == "sinhm":
            r = (e - f) / 2
        elif name == "coshm":
            r = (e + f) / 2
        else:  # tanhm: cosh⁻¹ sinh
            r = mpmath.inverse((e + f) / 2) * ((e - f) / 2)
    return np.array([[complex(r[i, j]) for j in range(r.cols)] for i in range(r.rows)])


def draw(rng):
    """A family and a matrix of it: (label, matrix, the functions that take it)."""
    n = int(rng.integers(2, 7))
    kind = int(rng.integers(7))
    upper = np.triu(rng.standard_normal((n, n)), 1)
    every = ("expm", "sinhm", "coshm", "tanhm", "funm")
    if kind == 0:  # a stiff triangle: neighbouring diagonal entries far apart (expm's exact band)
        diagonal = rng.uniform(-2000.0, 700.0, n)
        return "stiff triangle", np.diag(diagonal) + upper, ("expm", "funm")
    if kind == 1:  # a tight pair beside eigenvalues far from it (funm's clusters)
        gap = 10.0 ** rng.uniform(-13, -3)
        far = rng.uniform(-6000.0, -10.0, n - 2)
        diagonal = np.concatenate([[0.0, -gap], far])
        return "tight pair", np.diag(diagonal) + upper, ("expm", "funm")
    if kind == 2:  # a small norm: (e^A − e^−A)/2 cancels
        return "small norm", 10.0 ** rng.uniform(-10, -1) * rng.standard_normal((n, n)), every
    if kind == 3:  # positive spectrum: square roots and logarithms
        b = rng.standard_normal((n, n))
        return "positive definite", b @ b.T + n * np.eye(n), ("sqrtm", "logm") + every
    if kind == 4:  # a dense matrix of moderate norm
        return "random", rng.uniform(0.1, 8.0) * rng.standard_normal((n, n)), every
    if kind == 5:  # far from normality: V D V⁻¹, V unit triangular with integer entries (the quotient's conditioning)
        v = np.eye(n) + np.triu(rng.integers(-7, 8, (n, n)), 1)
        diagonal = rng.uniform(-60.0, 60.0, n)
        return "far from normal", (v * diagonal) @ np.round(np.linalg.inv(v)), every
    # a triangle with eigenvalues near tanh's poles iπ(k + ½), where tanh grows as the reciprocal of the distance
    poles = 1j * np.pi * (rng.integers(-2, 3, n) + 0.5) + 10.0 ** rng.uniform(-6, -1) * rng.standard_normal(n)
    return "near tanh's poles", np.diag(poles) + upper, ("tanhm",)


def ours(name, a):
    if name == "funm":
        return np.asarray(nm.funm(a, np.exp, disp=False)[0])
    return np.asarray(getattr(nm, name)(a))


def theirs(name, a):
    if name == "funm":
        return sl.funm(a, np.exp, disp=False)[0]
    return getattr(sl, name)(a)


def run_case(seed, index):
    rng = np.random.default_rng([seed, index])
    label, a, names = draw(rng)
    failures = []
    for name in names:
        try:
            exact = reference(name, a)
        except (ZeroDivisionError, ValueError):  # no reference (a pole hit): nothing to judge
            continue
        with np.errstate(over="ignore"):  # an exponential past 10¹⁵⁴: its norm overflows, nothing to judge
            scale = np.linalg.norm(exact)
        if not np.isfinite(scale) or scale == 0.0 or scale > 1e300:
            continue
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                got = ours(name, a)
            except nm.LinAlgError:
                failures.append(f"{name}: raised where the reference is finite")
                continue
            try:
                their = theirs(name, a)
                their_error = np.linalg.norm(their - exact) / scale
            except Exception:  # noqa: BLE001 (SciPy's failure leaves its error unknown: none to excuse ours)
                their_error = 0.0
        error = np.linalg.norm(got - exact) / scale if np.all(np.isfinite(got)) else np.inf
        if not np.isfinite(their_error):
            their_error = 0.0
        if error > max(1e-12, 10.0 * their_error):
            failures.append(f"{name}: relative error {error:.1e}, SciPy's {their_error:.1e}")
    return label, a, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--case", type=int, default=None)
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else int(time.time())
    start = time.time()
    cases = [args.case] if args.case is not None else range(args.cases)
    found = 0
    for index in cases:
        label, a, failures = run_case(seed, index)
        for failure in failures:
            found += 1
            print(f"seed {seed} --case {index}: {label} {a.shape[0]}x{a.shape[0]}: {failure}\n{np.array2string(a, precision=17)}")
    print(f"{len(cases)} cases, seed {seed}, {found} failure(s), {time.time() - start:.0f} s")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
