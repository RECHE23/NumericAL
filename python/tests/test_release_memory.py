"""numerical.release_memory: callable at any time, it frees the kept workspaces, and computing afterwards gives the
same answers, to the bit; on Linux, the process's resident memory falls (a fresh interpreter each)."""
import os
import subprocess
import sys
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy not installed")
class ReleaseMemory(unittest.TestCase):
    def test_answers_unchanged_after_a_release(self):
        r = np.random.default_rng(1101)
        a = r.standard_normal((300, 300))
        s = a + a.T
        before = [np.asarray(x).tobytes() for x in (numerical.matmul(a, a), numerical.solve(a, a), numerical.eigh(s)[1])]
        numerical.release_memory()
        numerical.release_memory()
        after = [np.asarray(x).tobytes() for x in (numerical.matmul(a, a), numerical.solve(a, a), numerical.eigh(s)[1])]
        self.assertEqual(before, after)

    @unittest.skipUnless(sys.platform.startswith("linux"), "/proc/self/statm")
    def test_resident_memory_falls_on_linux(self):
        code = """
import gc, os, numpy as np, numerical
def rss(): return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
r = np.random.default_rng(0); a = r.standard_normal((1000, 1000)); s = a + a.T
base = rss()
for f in (lambda: numerical.matmul(a, a), lambda: numerical.inv(a), lambda: numerical.eigh(s), lambda: numerical.svd(a)):
    x = f(); del x
gc.collect(); kept = rss() - base
numerical.release_memory()
print(kept, rss() - base)
"""
        env = dict(os.environ, NUMERICAL_NUM_THREADS="1")
        kept, after = map(int, subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                                              check=True).stdout.split())
        self.assertLess(after, kept / 2)


if __name__ == "__main__":
    unittest.main()
