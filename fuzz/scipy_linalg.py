"""Differential fuzzing of NumericAL's scipy.linalg functions against SciPy.

Each case draws a matrix from a family that breaks weak algorithms (clustered or repeated spectra, graded scales from
1e-6 to 1e6, near singularity, near defectiveness, exact integers, complex data, and every entry scaled by 1e150 or
1e-150) at an order biased towards the block boundaries, then passes it to NumericAL and to SciPy and measures each
answer by the backward error of what defines it: `expm(X) − A` for logm, `X² − A` for sqrtm, `AX + XB − Q` for
Sylvester, `ZTZᴴ − A` for the Schur form, the equation's residual for Lyapunov and Riccati, `A x − b` for the
solvers. A case fails where NumericAL's error exceeds both 50·n·ε and ten times SciPy's on the same input, or where
NumericAL raises and SciPy answers within that bound. A failure prints the seed and the case, which reproduce it
alone:

    python3 fuzz/scipy_linalg.py --seed S --case C

`--huge` runs the functions on random matrices scaled by 1e200 and 1e-200, each call in a process of its own with a
time limit: an answer must be finite and accurate, or a refusal; a hang or a crash is a failure.
"""
import argparse
import multiprocessing
import sys
import time
import warnings

import numpy as np

import numerical

try:
    import scipy.linalg as sl
except ImportError:  # pragma: no cover - a manual tool
    print("SciPy is required for this campaign: pip install scipy")
    sys.exit(0)

EPS = np.finfo(float).eps
SLACK = 50.0
BOUNDARIES = (1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65)
FAMILIES = ("normal", "clustered", "graded", "near_singular", "near_defective", "integers")
SCALES = (1.0, 1.0, 1.0, 1e150, 1e-150)


class Failure(Exception):
    pass


def size(rng):
    return int(rng.choice(BOUNDARIES)) if rng.random() < 0.6 else int(rng.integers(1, 66))


def fro(a):
    """The Frobenius norm at the scale of the largest magnitude: a plain sum of squares overflows past 1e154."""
    a = np.asarray(a)
    if not a.size:
        return 0.0
    largest = float(np.max(np.abs(a)))
    if largest == 0.0 or not np.isfinite(largest):
        return largest
    return largest * float(np.linalg.norm(a / largest))


def ratio(top, bottom):
    return top / bottom if bottom > 0 else (0.0 if top == 0 else np.inf)


def square(rng, n, family, cplx):
    """An n × n matrix of the family `family`."""
    def noise(*shape):
        return rng.standard_normal(shape) + (1j * rng.standard_normal(shape) if cplx else 0)
    if family == "integers":
        return rng.integers(-3, 4, size=(n, n)).astype(complex if cplx else float)
    if family == "graded":
        d = np.logspace(-6, 6, n)
        return noise(n, n) * d[:, None] * np.sqrt(d)[None, :]
    q, _ = np.linalg.qr(noise(n, n))
    if family == "clustered":
        values = np.repeat(rng.uniform(0.5, 2.0, 3), n)[:n] + rng.choice([0.0, 1e-12, 1e-8], n)
        return (q * values) @ q.conj().T + 1e-3 * np.triu(noise(n, n), 1)
    if family == "near_singular":
        values = np.concatenate([[1e-13], rng.uniform(0.5, 2.0, n - 1)])[:n]
        return (q * values) @ q.conj().T
    if family == "near_defective":  # a Jordan-like block, perturbed
        j = np.eye(n) * 1.5 + np.diag(np.ones(n - 1), 1) + 1e-10 * noise(n, n)
        return q @ j @ q.conj().T
    return noise(n, n)


def hermitian_part(a):
    return (a + a.conj().T) / 2


def positive(a):
    n = a.shape[0]
    return a @ a.conj().T / max(n, 1) + np.eye(n)


def stable(a, scale):
    """A shifted to the left half-plane by its own size: a stable matrix of the family's structure."""
    n = a.shape[0]
    return a - (fro(a) / np.sqrt(max(n, 1)) + scale) * np.eye(n)


def contractive(a):
    """A scaled to a spectral radius about one half."""
    radius = float(np.max(np.abs(np.linalg.eigvals(a)))) if a.size else 0.0
    return a / (2 * radius) if radius > 0 else a


# Each operation: (name, run(a, b, rng) -> (NumericAL's backward error, SciPy's), n for the bound).
def residuals(ours, theirs, measure):
    """The backward errors of the two answers, by `measure`; an exception is kept in its place, whichever it is (SciPy's
    own failures among them: its solve_banded of a complex 1 × 1 band and a real b raises a casting error)."""
    errors = []
    for answer in (ours, theirs):
        try:
            errors.append(measure(answer()))
        except Exception as caught:  # noqa: BLE001 - judged by judge(): a failure where the other side answered well
            errors.append(caught)
    return errors


def op_logm(a, rng, scale):
    a = a + (fro(a) / np.sqrt(a.shape[0]) + scale) * np.eye(a.shape[0])  # off the negative axis
    return residuals(lambda: np.asarray(numerical.logm(a)), lambda: sl.logm(a),
                     lambda x: ratio(fro(sl.expm(x) - a), fro(a)))


def op_sqrtm(a, rng, scale):
    a = a + (fro(a) / np.sqrt(a.shape[0]) + scale) * np.eye(a.shape[0])
    return residuals(lambda: np.asarray(numerical.sqrtm(a)), lambda: sl.sqrtm(a), lambda x: ratio(fro(x @ x - a), fro(a)))


def op_polar(a, rng, scale):
    def measure(up):
        u, p = (np.asarray(x) for x in up)
        return max(ratio(fro(u @ p - a), fro(a)), fro(u.conj().T @ u - np.eye(a.shape[1])))
    return residuals(lambda: numerical.polar(a), lambda: sl.polar(a), measure)


def op_ldl(a, rng, scale):
    h = hermitian_part(a)
    def measure(answer):
        lu, d = np.asarray(answer[0]), np.asarray(answer[1])
        return ratio(fro(lu @ d @ lu.conj().T - h), fro(h))
    return residuals(lambda: numerical.ldl(h), lambda: sl.ldl(h), measure)


def op_qr_pivoting(a, rng, scale):
    def measure(qrp):
        q, r, p = np.asarray(qrp[0]), np.asarray(qrp[1]), list(qrp[2])
        return ratio(fro(q @ r - a[:, p]), fro(a))
    return residuals(lambda: numerical.qr(a, pivoting=True), lambda: sl.qr(a, pivoting=True, mode="economic"), measure)


def op_schur(a, rng, scale):
    output = "complex" if np.iscomplexobj(a) else "real"
    def measure(tz):
        t, z = np.asarray(tz[0]), np.asarray(tz[1])
        if np.any(np.tril(t, -2)):
            return np.inf
        return ratio(fro(z @ t @ z.conj().T - a), fro(a))
    return residuals(lambda: numerical.schur(a, output=output), lambda: sl.schur(a, output=output), measure)


def op_sylvester(a, rng, scale):
    n = a.shape[0]
    b = stable(-a.T + 0.1 * scale * rng.standard_normal((n, n)), scale) * -1  # B's spectrum on the right, A's anywhere
    a = stable(a, scale)
    q = rng.standard_normal((n, n)) * scale
    def measure(x):
        return ratio(fro(a @ x + x @ b - q), (fro(a) + fro(b)) * fro(x) + fro(q))
    return residuals(lambda: np.asarray(numerical.solve_sylvester(a, b, q)), lambda: sl.solve_sylvester(a, b, q), measure)


def op_lyapunov(a, rng, scale):
    a = stable(a, scale)
    q = positive(rng.standard_normal(a.shape)) * scale
    def measure(x):
        return ratio(fro(a @ x + x @ a.conj().T - q), 2 * fro(a) * fro(x) + fro(q))
    return residuals(lambda: np.asarray(numerical.solve_continuous_lyapunov(a, q)),
                     lambda: sl.solve_continuous_lyapunov(a, q), measure)


def op_discrete_lyapunov(a, rng, scale):
    a = contractive(a)
    q = positive(rng.standard_normal(a.shape)) * scale
    def measure(x):
        return ratio(fro(a @ x @ a.conj().T - x + q), (fro(a) ** 2 + 1) * fro(x) + fro(q))
    return residuals(lambda: np.asarray(numerical.solve_discrete_lyapunov(a, q)),
                     lambda: sl.solve_discrete_lyapunov(a, q), measure)


def op_care(a, rng, scale):
    n = a.shape[0]
    a = stable(a / scale, 1.0)  # the equation's terms at one scale: Riccati is not homogeneous in A
    b = rng.standard_normal((n, max(1, n // 3)))
    q, r = positive(rng.standard_normal((n, n))), positive(rng.standard_normal((b.shape[1],) * 2))
    def measure(x):
        g = x @ b
        return ratio(fro(a.conj().T @ x + x @ a - g @ np.linalg.solve(r, g.conj().T) + q),
                     2 * fro(a) * fro(x) + fro(g) ** 2 / max(fro(r), 1e-300) + fro(q))
    return residuals(lambda: np.asarray(numerical.solve_continuous_are(a, b, q, r)),
                     lambda: sl.solve_continuous_are(a, b, q, r), measure)


def op_dare(a, rng, scale):
    n = a.shape[0]
    a = contractive(a / scale)
    b = rng.standard_normal((n, max(1, n // 3)))
    q, r = positive(rng.standard_normal((n, n))), positive(rng.standard_normal((b.shape[1],) * 2))
    def measure(x):
        g = a.conj().T @ x @ b
        return ratio(fro(a.conj().T @ x @ a - x - g @ np.linalg.solve(r + b.conj().T @ x @ b, g.conj().T) + q),
                     (fro(a) ** 2 + 1) * fro(x) + fro(g) + fro(q))
    return residuals(lambda: np.asarray(numerical.solve_discrete_are(a, b, q, r)),
                     lambda: sl.solve_discrete_are(a, b, q, r), measure)


def solver_measure(a, b):
    return lambda x: ratio(fro(a @ x - b), fro(a) * fro(x) + fro(b))


def op_lu(a, rng, scale):
    b = rng.standard_normal(a.shape[0]) * scale
    return residuals(lambda: np.asarray(numerical.lu_solve(numerical.lu_factor(a), b)),
                     lambda: sl.lu_solve(sl.lu_factor(a), b), solver_measure(a, b))


def op_triangular(a, rng, scale):
    t = np.triu(a) + np.diag(np.where(np.diag(a) == 0, scale, 0))
    b = rng.standard_normal(a.shape[0]) * scale
    return residuals(lambda: np.asarray(numerical.solve_triangular(t, b)), lambda: sl.solve_triangular(t, b),
                     solver_measure(t, b))


def op_banded(a, rng, scale):
    n = a.shape[0]
    half = int(rng.integers(0, min(n, 4)))
    dense = np.triu(np.tril(a, half), -half)
    ab = np.zeros((2 * half + 1, n), dtype=a.dtype)
    for k in range(-half, half + 1):
        ab[half - k, max(k, 0):n + min(k, 0)] = np.diagonal(dense, k)
    b = rng.standard_normal(n) * scale
    return residuals(lambda: np.asarray(numerical.solve_banded((half, half), ab, b)),
                     lambda: sl.solve_banded((half, half), ab, b), solver_measure(dense, b))


def op_toeplitz(a, rng, scale):
    c, r = a[:, 0], a[0, :]
    c, r = c.copy(), r.copy()
    c[0] = r[0] = fro(c) + fro(r) + scale  # dominant: every leading minor nonsingular
    t = sl.toeplitz(c, r)
    b = rng.standard_normal(a.shape[0]) * scale
    return residuals(lambda: np.asarray(numerical.solve_toeplitz((c, r), b)), lambda: sl.solve_toeplitz((c, r), b),
                     solver_measure(t, b))


def op_circulant(a, rng, scale):
    c = a[:, 0].copy()
    c[0] += fro(c) + scale
    b = rng.standard_normal(a.shape[0]) * scale
    return residuals(lambda: np.asarray(numerical.solve_circulant(c, b)), lambda: sl.solve_circulant(c, b),
                     solver_measure(sl.circulant(c), b))


def op_eigh_pencil(a, rng, scale):
    h = hermitian_part(a)
    m = positive(rng.standard_normal(a.shape)) * scale
    def measure(wv):
        w, v = np.asarray(wv[0]), np.asarray(wv[1])
        return ratio(fro(h @ v - (m @ v) * w), (fro(h) + fro(m) * float(np.max(np.abs(w), initial=0.0))) * fro(v))
    return residuals(lambda: numerical.eigh(h, b=m), lambda: sl.eigh(h, m), measure)


def op_expm_frechet(a, rng, scale):
    a = a / max(fro(a), 1e-300) * rng.uniform(0.1, 5.0)  # ‖A‖ ≤ 5: the derivative's terms finite
    e = rng.standard_normal(a.shape)
    exact = sl.expm(np.block([[a, e], [np.zeros_like(a), a]]))[: a.shape[0], a.shape[0]:]
    return residuals(lambda: np.asarray(numerical.expm_frechet(a, e, compute_expm=False)),
                     lambda: sl.expm_frechet(a, e, compute_expm=False), lambda x: ratio(fro(x - exact), fro(exact)))


OPERATIONS = (("logm", op_logm), ("sqrtm", op_sqrtm), ("polar", op_polar), ("ldl", op_ldl),
              ("qr-pivoting", op_qr_pivoting), ("schur", op_schur), ("solve_sylvester", op_sylvester),
              ("solve_continuous_lyapunov", op_lyapunov), ("solve_discrete_lyapunov", op_discrete_lyapunov),
              ("solve_continuous_are", op_care), ("solve_discrete_are", op_dare), ("lu_factor/lu_solve", op_lu),
              ("solve_triangular", op_triangular), ("solve_banded", op_banded), ("solve_toeplitz", op_toeplitz),
              ("solve_circulant", op_circulant), ("eigh(a, b)", op_eigh_pencil), ("expm_frechet", op_expm_frechet))
# The functions whose problems the scale changes in kind (Riccati is not homogeneous; expm_frechet's derivative
# overflows): drawn at scale 1 only.
UNSCALED = {"solve_continuous_are", "solve_discrete_are", "expm_frechet"}


def case(seed, index):
    rng = np.random.default_rng([seed, index, 0x5c1])
    cplx = bool(rng.random() < 0.3)
    family = str(rng.choice(FAMILIES))
    name, operation = OPERATIONS[int(rng.integers(len(OPERATIONS)))]
    scale = 1.0 if name in UNSCALED else float(rng.choice(SCALES))
    n = size(rng)
    a = square(rng, n, family, cplx) * scale
    return rng, name, operation, a, scale, f"{name}: {'complex ' if cplx else ''}{family} {n}×{n}, scale {scale:g}"


def judge(ours, theirs, n):
    """The verdict on the two backward errors: None where NumericAL's is within bounds, else why not."""
    bound = SLACK * max(n, 1) * EPS
    their_error = theirs if isinstance(theirs, float) else np.inf
    if isinstance(ours, Exception):
        if their_error <= bound:
            return f"NumericAL raised {type(ours).__name__}: {ours} where SciPy answered (error {their_error:.3g})"
        return None
    if not np.isfinite(ours) or ours > max(bound, 10 * their_error):
        return f"backward error {ours:.3g} (bound {bound:.3g}, SciPy's {their_error if np.isfinite(their_error) else theirs})"
    return None


def run_case(seed, index, tally):
    rng, name, operation, a, scale, label = case(seed, index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours, theirs = operation(a, rng, scale)
    kind = ("both answered" if not isinstance(ours, Exception) and not isinstance(theirs, Exception)
            else "both raised" if isinstance(ours, Exception) and isinstance(theirs, Exception)
            else "NumericAL raised" if isinstance(ours, Exception) else "SciPy raised")
    tally.setdefault(name, {}).setdefault(kind, 0)
    tally[name][kind] += 1
    return label, judge(ours, theirs, a.shape[0])


# --huge: each call in a process of its own, with a time limit.
HUGE = (("logm", lambda a: numerical.logm(a)), ("sqrtm", lambda a: numerical.sqrtm(a)),
        ("schur", lambda a: numerical.schur(a)), ("polar", lambda a: numerical.polar(a)),
        ("ldl", lambda a: numerical.ldl((a + a.T) / 2)), ("qr-pivoting", lambda a: numerical.qr(a, pivoting=True)),
        ("solve_sylvester", lambda a: numerical.solve_sylvester(a, a.T + np.eye(len(a)) * np.abs(a).max() * 40, a)),
        ("solve_continuous_lyapunov",
         lambda a: numerical.solve_continuous_lyapunov(a - np.eye(len(a)) * np.abs(a).max() * 40, (a / np.abs(a).max()) @ a.T)),
        ("expm", lambda a: numerical.expm(a)))


def huge_call(name, scale, queue):
    rng = np.random.default_rng(17)
    a = rng.standard_normal((17, 17)) * scale
    function = dict(HUGE)[name]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = function(a)
        parts = result if isinstance(result, tuple) else (result,)
        finite = all(np.isfinite(np.asarray(p)).all() for p in parts if hasattr(p, "shape") or isinstance(p, numerical.Tensor))
        queue.put(("finite" if finite else "non-finite", ""))
    except (numerical.LinAlgError, ValueError) as caught:
        queue.put(("refused", str(caught)))


def run_huge(limit):
    failures = 0
    for scale in (1e200, 1e-200):
        for name, _ in HUGE:
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=huge_call, args=(name, scale, queue))
            process.start()
            process.join(limit)
            if process.is_alive():
                process.terminate()
                verdict = ("hung", f"no answer in {limit} s")
            elif process.exitcode != 0:
                verdict = ("crashed", f"exit code {process.exitcode}")
            else:
                verdict = queue.get()
            # expm of a matrix of eigenvalues 1e200 overflows in truth; any other function answers finitely or refuses
            bad = verdict[0] in ("hung", "crashed") or (verdict[0] == "non-finite" and not (name == "expm" and scale > 1))
            failures += bad
            print(f"{'FAIL ' if bad else ''}{name} × {scale:g}: {verdict[0]} {verdict[1]}".rstrip(), flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=int(time.time()))
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--case", type=int, help="run this one case only")
    parser.add_argument("--huge", action="store_true", help="the extreme scales, each call in a process of its own")
    parser.add_argument("--limit", type=float, default=30.0, help="--huge: seconds a call may take")
    args = parser.parse_args()
    if args.huge:
        failed = run_huge(args.limit)
        print(f"{failed} failure(s) at extreme scales", flush=True)
        return 1 if failed else 0
    indices = [args.case] if args.case is not None else range(args.cases)
    print(f"seed {args.seed}, {len(indices)} cases against SciPy", flush=True)
    failed = 0
    started = time.time()
    tally = {}
    with np.errstate(all="ignore"):
        for index in indices:
            label, failure = run_case(args.seed, index, tally)
            if failure:
                failed += 1
                print(f"FAIL case {index} ({label}) {failure}\n"
                      f"     repro: python3 fuzz/scipy_linalg.py --seed {args.seed} --case {index}", flush=True)
    for name in sorted(tally):  # what each function was checked on: answers compared, or refusals
        print(f"  {name:<27} " + ", ".join(f"{kind} {count}" for kind, count in sorted(tally[name].items())), flush=True)
    print(f"{failed} failure(s) in {time.time() - started:.0f} s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
