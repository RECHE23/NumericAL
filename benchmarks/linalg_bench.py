#!/usr/bin/env python3
"""NumericAL against NumPy's BLAS/LAPACK on dense linear algebra, each library alone in its own process.

Each library runs in a process of its own, with all of its threads, and the other library absent from the timing:
a BLAS whose threads still spin after a call would otherwise slow the next library's. In NumericAL's process
NumPy is held to one thread and used only to check each answer before its timing (a fast wrong answer is not a
result). Every figure is the median of `--reps` calls after a warm-up (a third of them, at least 3, from order
512), and the table takes the median over `--rounds` processes per library, so one noisy process does not decide
it.

Run from the repository root: make bench-linalg (SIZES=64,128,256,512,1024 ROUNDS=3 by default; NUMPY_INPUTS=1 calls
NumericAL on the NumPy arrays, as a NumPy user does, where it takes Tensors converted beforehand by default).
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

OPERATIONS = ("matmul", "matvec", "transpose", "solve", "inverse", "cholesky", "qr", "eigh", "svd", "lstsq", "eig",
              "eigvalsh", "slogdet", "svdvals", "cond", "matmul32", "inverse_stack")


def inputs(np, n):
    rng = np.random.default_rng(n)
    a = rng.standard_normal((n, n))
    side = max(4, n // 8)  # a stack of 64 matrices an eighth of the order
    return {"a": a, "spd": a @ a.T + n * np.eye(n), "sym": (a + a.T) / 2, "b": rng.standard_normal((n, 4)),
            "tall": rng.standard_normal((2 * n, n)), "rhs": rng.standard_normal(2 * n), "v": rng.standard_normal(n),
            "a32": a.astype(np.float32), "stack": rng.standard_normal((64, side, side)) + side * np.eye(side)}


def calls(library, np, data):
    """Each operation as a call on the library's own arrays, built outside the timing."""
    if library == "numpy":
        la = np.linalg
        a, spd, sym, b, tall, rhs, v, a32, stack = (data[k] for k in ("a", "spd", "sym", "b", "tall", "rhs", "v", "a32", "stack"))
        # NumPy's .T is a view, free and lazy; NumericAL's is a contiguous copy, so it is timed against one.
        return {"matmul": lambda: a @ a, "matvec": lambda: a @ v, "transpose": lambda: np.ascontiguousarray(a.T), "solve": lambda: la.solve(a, b), "inverse": lambda: la.inv(a),
                "cholesky": lambda: la.cholesky(spd), "qr": lambda: la.qr(a), "eigh": lambda: la.eigh(sym),
                "svd": lambda: la.svd(a, full_matrices=False), "lstsq": lambda: la.lstsq(tall, rhs, rcond=None),
                "eig": lambda: la.eig(a), "eigvalsh": lambda: la.eigvalsh(sym), "slogdet": lambda: la.slogdet(a),
                "svdvals": lambda: la.svd(a, compute_uv=False), "cond": lambda: la.cond(a), "matmul32": lambda: a32 @ a32,
                "inverse_stack": lambda: la.inv(stack)}
    import numerical as nl

    if os.environ.get("NUMERICAL_BENCH_NUMPY_INPUTS"):
        # NumericAL called on the NumPy arrays themselves, as a NumPy user calls it: each conversion is timed.
        a, spd, sym, b, tall, rhs, v, a32, stack = (data[k] for k in ("a", "spd", "sym", "b", "tall", "rhs", "v", "a32", "stack"))
        return {"matmul": lambda: nl.matmul(a, a), "matvec": lambda: nl.matmul(a, v), "transpose": lambda: nl.asarray(a).T,
                "solve": lambda: nl.solve(a, b), "inverse": lambda: nl.inverse(a), "cholesky": lambda: nl.cholesky(spd),
                "qr": lambda: nl.qr(a), "eigh": lambda: nl.eigh(sym), "svd": lambda: nl.svd(a, full_matrices=False),
                "lstsq": lambda: nl.lstsq(tall, rhs), "eig": lambda: nl.eig(a), "eigvalsh": lambda: nl.eigvalsh(sym),
                "slogdet": lambda: nl.slogdet(a), "svdvals": lambda: nl.svdvals(a), "cond": lambda: nl.cond(a),
                "matmul32": lambda: nl.matmul(a32, a32), "inverse_stack": lambda: nl.inverse(stack)}
    a, spd, sym, b, tall, rhs, v, a32, stack = (nl.asarray(data[k]) for k in ("a", "spd", "sym", "b", "tall", "rhs", "v", "a32", "stack"))
    return {"matmul": lambda: a @ a, "matvec": lambda: a @ v, "transpose": lambda: a.T, "solve": lambda: nl.solve(a, b), "inverse": lambda: nl.inverse(a),
            "cholesky": lambda: nl.cholesky(spd), "qr": lambda: nl.qr(a), "eigh": lambda: nl.eigh(sym),
            "svd": lambda: nl.svd(a, full_matrices=False), "lstsq": lambda: nl.lstsq(tall, rhs),
            "eig": lambda: nl.eig(a), "eigvalsh": lambda: nl.eigvalsh(sym), "slogdet": lambda: nl.slogdet(a),
            "svdvals": lambda: nl.svdvals(a), "cond": lambda: nl.cond(a), "matmul32": lambda: a32 @ a32,
            "inverse_stack": lambda: nl.inverse(stack)}


def agrees(np, operation, data, got):
    """NumericAL's answer against its defining property, to a backward-error bound."""
    n = data["a"].shape[0]
    tolerance = 1e3 * n * np.finfo(float).eps
    norm = np.linalg.norm
    if operation == "matmul":
        return norm(np.asarray(got) - data["a"] @ data["a"]) <= tolerance * norm(data["a"]) ** 2
    if operation == "matvec":
        return norm(np.asarray(got) - data["a"] @ data["v"]) <= tolerance * norm(data["a"]) * norm(data["v"])
    if operation == "transpose":
        return np.array_equal(np.asarray(got), data["a"].T)
    if operation == "solve":
        return norm(data["a"] @ np.asarray(got) - data["b"]) <= tolerance * norm(data["a"]) * norm(np.asarray(got))
    if operation == "inverse":
        return norm(data["a"] @ np.asarray(got) - np.eye(n)) <= tolerance * norm(data["a"]) * norm(np.asarray(got))
    if operation == "cholesky":
        lower = np.asarray(got)
        return norm(lower @ lower.T - data["spd"]) <= tolerance * norm(data["spd"])
    if operation == "qr":
        q, r = (np.asarray(x) for x in got)
        return norm(q @ r - data["a"]) <= tolerance * norm(data["a"])
    if operation == "eigh":
        w, v = (np.asarray(x) for x in got)
        return norm(data["sym"] @ v - v * w) <= tolerance * norm(data["sym"])
    if operation == "svd":
        u, s, vh = (np.asarray(x) for x in got)
        return norm((u * s) @ vh - data["a"]) <= tolerance * norm(data["a"])
    if operation == "lstsq":
        want = np.linalg.lstsq(data["tall"], data["rhs"], rcond=None)[0]
        return norm(np.asarray(got[0]) - want) <= 1e-6 * norm(want)
    if operation == "eig":
        w, v = (np.asarray(x) for x in got)
        return norm(data["a"] @ v - v * w) <= tolerance * norm(data["a"])
    if operation == "eigvalsh":
        return np.max(np.abs(np.asarray(got) - np.linalg.eigvalsh(data["sym"]))) <= tolerance * norm(data["sym"])
    if operation == "slogdet":
        sign, logdet = got
        want_sign, want_logdet = np.linalg.slogdet(data["a"])
        return sign == want_sign and abs(logdet - want_logdet) <= 1e-8 * n * max(1.0, abs(want_logdet))
    if operation == "svdvals":
        return np.max(np.abs(np.asarray(got) - np.linalg.svd(data["a"], compute_uv=False))) <= tolerance * norm(data["a"])
    if operation == "cond":
        want = np.linalg.cond(data["a"])
        return abs(float(got) - want) <= 1e-6 * want
    if operation == "matmul32":
        want = data["a"] @ data["a"]
        return norm(np.asarray(got).astype(float) - want) <= 1e3 * n * np.finfo(np.float32).eps * norm(data["a"]) ** 2
    inverses = np.asarray(got)
    return all(norm(m @ x - np.eye(m.shape[0])) <= tolerance * norm(m) * norm(x) for m, x in zip(data["stack"], inverses))


def measure(library, sizes, operations, reps):
    """One process's medians, in seconds, by operation and size."""
    import numpy as np

    if library == "numerical" and os.environ.get("NUMERICAL_BENCH_THREADS"):
        import numerical
        numerical.set_num_threads(int(os.environ["NUMERICAL_BENCH_THREADS"]))
    results = {}
    for n in sizes:
        data = inputs(np, n)
        table = calls(library, np, data)
        for operation in operations:
            call = table[operation]
            got = call()
            if library == "numerical" and not agrees(np, operation, data, got):
                results[f"{operation} {n}"] = None
                continue
            count = max(3, reps if n <= 256 else reps // 3)
            times = []
            for _ in range(count):
                start = time.perf_counter()
                call()
                times.append(time.perf_counter() - start)
            results[f"{operation} {n}"] = statistics.median(times)
    return results


def machine():
    import numpy as np

    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        else:
            with open("/proc/cpuinfo", encoding="utf-8") as info:
                cpu = next(line.split(":", 1)[1].strip() for line in info if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    blas = "unknown"
    try:
        config = np.show_config(mode="dicts")
        blas = config["Build Dependencies"]["blas"]["name"] + " " + config["Build Dependencies"]["blas"]["version"]
    except (TypeError, KeyError):
        pass
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    return {"cpu": cpu, "cores": os.cpu_count(), "numpy": np.__version__, "blas": blas, "python": platform.python_version(),
            "commit": commit, "date": time.strftime("%Y-%m-%d")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", default="64,128,256,512,1024")
    parser.add_argument("--operations", default=",".join(OPERATIONS))
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, help="the threads of both libraries (NumPy's BLAS, NumericAL's pool): "
                        "on a host with hyperthreads or other users, one per core, run under taskset")
    parser.add_argument("--numpy-inputs", action="store_true", help="call NumericAL on the NumPy arrays, each conversion "
                        "timed, where it takes Tensors converted beforehand by default")
    parser.add_argument("--json", help="write the figures here too")
    parser.add_argument("--worker", choices=("numpy", "numerical"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    sizes = [int(n) for n in args.sizes.split(",")]
    operations = args.operations.split(",")
    if args.worker:
        print(json.dumps(measure(args.worker, sizes, operations, args.reps)))
        return 0
    rounds = {"numpy": [], "numerical": []}
    for _ in range(args.rounds):
        for library in rounds:
            env = dict(os.environ)
            if library == "numerical":
                env["OPENBLAS_NUM_THREADS"] = "1"  # NumPy only checks answers here: no spinning threads
            elif args.threads:
                env["OPENBLAS_NUM_THREADS"] = str(args.threads)
            if args.threads:
                env["NUMERICAL_BENCH_THREADS"] = str(args.threads)
            if args.numpy_inputs:
                env["NUMERICAL_BENCH_NUMPY_INPUTS"] = "1"
            out = subprocess.run([sys.executable, __file__, "--worker", library, "--sizes", args.sizes, "--operations",
                                  args.operations, "--reps", str(args.reps)], env=env, capture_output=True, text=True,
                                 check=True).stdout
            rounds[library].append(json.loads(out))
    info = machine()
    print(f"{info['cpu']}, {args.threads or info['cores']} threads; NumPy {info['numpy']} ({info['blas']}); Python {info['python']}; "
          f"NumericAL {info['commit']}, {'on the NumPy arrays' if args.numpy_inputs else 'on Tensors converted beforehand'}; "
          f"{info['date']}. Median of {args.rounds} processes per library; "
          "ratio = NumericAL / NumPy (below 1: NumericAL faster).\n")
    print("| operation | " + " | ".join(f"n = {n}" for n in sizes) + " |")
    print("| --- | " + " | ".join("---:" for _ in sizes) + " |")
    figures = {}
    for operation in operations:
        cells = []
        for n in sizes:
            key = f"{operation} {n}"
            ours = [r[key] for r in rounds["numerical"]]
            theirs = statistics.median(r[key] for r in rounds["numpy"])
            if any(value is None for value in ours):
                cells.append("**wrong**")
                figures[key] = None
                continue
            mine = statistics.median(ours)
            figures[key] = {"numerical": mine, "numpy": theirs}
            cells.append(f"{mine * 1e3:.3g} ms ({mine / theirs:.2f}×)")
        print(f"| {operation} | " + " | ".join(cells) + " |")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump({"machine": info, "figures": figures}, out, indent=1)
    return 1 if any(value is None for value in figures.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
