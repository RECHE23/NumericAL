# Usage : python mm1.py run {r|c} <fils>. Produits de matrices : NumericAL contre OpenBLAS, réel et complexe, chaque bibliothèque dans son processus.
import os, sys, time, json, subprocess, pathlib
lib, dtype = sys.argv[1], sys.argv[2]
if lib == "child":
    which, dt = sys.argv[3], sys.argv[4]
    import numpy as np
    out = {}
    for n in [64, 128, 256, 512, 1024, 2048]:
        rng = np.random.default_rng(n)
        a = rng.standard_normal((n, n)); b = rng.standard_normal((n, n))
        if dt == "c":
            a = a + 1j * rng.standard_normal((n, n)); b = b + 1j * rng.standard_normal((n, n))
        if which == "numerical":
            import numerical
            ta, tb = numerical.asarray(a), numerical.asarray(b)
            f = lambda: ta @ tb
        else:
            f = lambda: a @ b
        f()
        reps = max(3, int(2e8 / n**3))
        best = 1e9
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(reps): f()
            best = min(best, (time.perf_counter() - t0) / reps)
        flops = (8 if dt == "c" else 2) * n**3
        out[n] = flops / best / 1e9
    print(json.dumps(out))
else:
    threads = sys.argv[3]
    env = dict(os.environ, OPENBLAS_NUM_THREADS=threads, NUMERICAL_NUM_THREADS=threads, PYTHONPATH=str(pathlib.Path(__file__).resolve().parents[3] / "python"))
    res = {w: json.loads(subprocess.check_output([sys.executable, __file__, "child", "x", w, dtype], env=env)) for w in ["numerical", "numpy"]}
    print(f"{'n':>5} | NumericAL GFLOP/s | OpenBLAS GFLOP/s | ratio temps")
    for n in res["numpy"]:
        print(f"{n:>5} | {res['numerical'][n]:17.1f} | {res['numpy'][n]:16.1f} | {res['numpy'][n]/res['numerical'][n]:.2f}")
