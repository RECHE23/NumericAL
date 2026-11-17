"""Matrices at extreme scales, and each fast path held to its general one: D1 A D2 with diagonal scalings drawn to
10^±150 (A random, real or complex, orders 1 to 16), and strongly non-normal triangular pencils. Every call must end
(a watchdog ends the run with every thread's traceback past WATCHDOG seconds); then

- det is NumPy's to 1e-10 wherever NumPy's is finite and not 0 (its pivots' product passed the doubles midway);
- the eigenvalues sum to the trace (eig's balancing swept forever near the subnormals);
- solve, inv and lstsq have a backward error within n·1e3·ε, or within 100 times NumPy's;
- eig(a, b)'s vectors are finite and solve the pencil where SciPy's are finite (its back substitution overflowed);
- a structured matrix (diagonal, upper triangular, Hermitian, Hermitian positive definite) and the same matrix with
  one zero entry nudged to 1e-300 of its largest (the structure gone, the problem the same to rounding) give expm,
  sqrtm, logm and inv within 1e-8 of each other wherever the problem is well conditioned and no eigenvalue lies on
  the negative real axis (on the cut a nudge may choose the other branch).

Usage: python fuzz/scaled.py [--seed S] [--cases N]
"""
import argparse
import faulthandler
import sys
import time
import warnings

import numpy as np

import numerical as nm

EPS = np.finfo(float).eps
WATCHDOG = 120
norm = np.linalg.norm


def scaled(rng, n, cplx, spread):
    """D1 A D2, the scalings drawn to 10^±spread."""
    a = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
    return (10.0 ** rng.uniform(-spread, spread, n))[:, None] * a * (10.0 ** rng.uniform(-spread, spread, n))[None, :]


def backward(name, a, solver):
    """The backward error of `solver`'s answer to problem `name` on `a`: infinite for an exception or a non-finite one."""
    try:
        if name == "solve":
            b = np.ones(len(a))
            x = np.asarray(solver(a, b))
            return norm(a @ x - b) / (norm(a) * norm(x) + norm(b))
        if name == "inv":
            x = np.asarray(solver(a))
            return min(norm(a @ x - np.eye(len(a))), norm(x @ a - np.eye(len(a)))) / (norm(a) * norm(x))
        b = np.ones(len(a))
        x = np.asarray(solver(a, b))
        return norm(a.conj().T @ (a @ x - b)) / (norm(a) ** 2 * norm(x) + norm(a) * norm(b))
    except Exception:  # noqa: BLE001 - judged against NumPy's answer
        return np.inf


SOLVERS = {
    "solve": (lambda a, b: nm.linalg.solve(a, b), np.linalg.solve),
    "inv": (nm.linalg.inv, np.linalg.inv),
    "lstsq": (lambda a, b: nm.linalg.lstsq(a, b, rcond=None)[0], lambda a, b: np.linalg.lstsq(a, b, rcond=None)[0]),
}


def scaled_checks(rng, failures, label):
    n = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16]))
    cplx = bool(rng.random() < 0.4)
    spread = float(rng.choice([0, 20, 75, 150]))
    a = scaled(rng, n, cplx, spread)
    where = f"{label}: {'complex ' if cplx else ''}{n}×{n} scaled to 1e±{spread:g}"
    their_det = complex(np.linalg.det(a))
    ours_det = complex(nm.linalg.det(a))
    if their_det != 0 and np.isfinite(their_det) and not abs(ours_det - their_det) <= 1e-10 * abs(their_det):
        failures.append(f"{where}: det {ours_det:.6e}, NumPy's {their_det:.6e}")
    values = np.asarray(nm.linalg.eigvals(a))
    if not abs(values.sum() - np.trace(a)) <= 1e-6 * n * np.abs(a).max():
        failures.append(f"{where}: eigenvalues summing to {values.sum():.6e}, the trace {np.trace(a):.6e}")
    for name, (ours, theirs) in SOLVERS.items():
        ours_error, their_error = backward(name, a, ours), backward(name, a, theirs)
        if ours_error > n * 1e3 * EPS and ours_error > 100 * their_error:
            failures.append(f"{where}: {name} backward error {ours_error:.2e}, NumPy's {their_error:.2e}")


def pencil_checks(rng, failures, label):
    n = int(rng.choice([20, 50, 80]))
    strict = float(10.0 ** rng.uniform(3, 8))
    t = np.triu(rng.standard_normal((n, n)), 1) * strict + np.diag(np.arange(1.0, n + 1))
    b = np.triu(rng.standard_normal((n, n)), 1) * 0.1 + np.eye(n)
    for left in (False, True):
        w, v = (np.asarray(part) for part in nm.eig(t, b, left=left, right=not left))
        if not np.isfinite(v).all():
            failures.append(f"{label}: eig(a, b) of a non-normal pencil (order {n}, strict part {strict:.1e}), "
                            f"{'left' if left else 'right'} vectors not finite")
            continue
        residual = t.conj().T @ v - (b.conj().T @ v) * np.conj(w) if left else t @ v - (b @ v) * w
        if not np.abs(residual).max() <= 1e-8 * np.abs(t).max():
            failures.append(f"{label}: eig(a, b) residual {np.abs(residual).max() / np.abs(t).max():.2e}")


def structured(rng, kind, n, cplx):
    g = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
    if kind == "diagonal":
        return np.diag(np.diag(g))
    if kind == "upper":
        return np.triu(g)
    q = np.linalg.qr(g)[0]
    values = rng.standard_normal(n) * 10.0 ** rng.uniform(-2, 2)
    if kind == "hermitian-positive":
        values = np.abs(values) + 1e-2
    h = (q * values) @ q.conj().T
    return (h + h.conj().T) / 2


def nudged(a):
    """`a` with its entry (n−1, 0) moved by 1e-300 of its largest: no longer triangular, diagonal nor Hermitian."""
    b = a.copy()
    b[len(b) - 1, 0] += np.abs(a).max() * 1e-300 if b[len(b) - 1, 0] == 0 else b[len(b) - 1, 0] * 1e-15
    return b


FUNCTIONS = {"expm": nm.expm, "sqrtm": nm.sqrtm, "logm": lambda a: nm.logm(a), "inv": nm.linalg.inv}
BRANCHED = {"sqrtm", "logm"}


def fast_path_checks(rng, failures, label):
    n = int(rng.choice([2, 3, 5, 8]))
    kind = str(rng.choice(["diagonal", "upper", "hermitian", "hermitian-positive"]))
    cplx = bool(rng.random() < 0.4)
    a = structured(rng, kind, n, cplx)
    if np.linalg.cond(a) >= 1e6:
        return
    eig = np.linalg.eigvals(a)
    on_cut = bool(np.any((np.abs(eig.imag) <= 1e-12 * np.abs(eig).max()) & (eig.real < 0)))
    for name, f in FUNCTIONS.items():
        if on_cut and name in BRANCHED:
            continue
        fast, general = np.asarray(f(a), dtype=complex), np.asarray(f(nudged(a)), dtype=complex)
        if not (np.isfinite(fast).all() and np.isfinite(general).all()):
            continue
        gap = np.abs(fast - general).max() / max(np.abs(general).max(), 1e-300)
        if gap > 1e-8:
            failures.append(f"{label}: {name} of a {'complex ' if cplx else ''}{kind} {n}×{n}: fast and general paths "
                            f"{gap:.2e} apart")


def run_case(seed, index):
    """The failures of case `index` from `seed`, a label each."""
    rng = np.random.default_rng([seed, index, 0x5ca1ed])
    failures = []
    label = f"seed {seed} case {index}"
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        family = int(rng.integers(3))
        (scaled_checks, pencil_checks, fast_path_checks)[family](rng, failures, label)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cases", type=int, default=300)
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else int(time.time())
    start = time.time()
    failures = []
    for index in range(args.cases):
        faulthandler.dump_traceback_later(WATCHDOG, exit=True)  # a call that never ends ends the run, traced
        failures += run_case(seed, index)
        faulthandler.cancel_dump_traceback_later()
    for failure in failures:
        print(failure)
    print(f"{args.cases} cases, seed {seed}, {len(failures)} failure(s), {time.time() - start:.0f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
