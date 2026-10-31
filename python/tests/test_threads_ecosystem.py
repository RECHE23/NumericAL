"""NumericAL among other libraries' threads: OMP_NUM_THREADS read at import when NUMERICAL_NUM_THREADS is absent (as
joblib's workers set it), its first count as OpenBLAS reads it; threadpoolctl's limits applied to it, whichever of the
two is imported first. Each case in a fresh interpreter; threadpoolctl's skipped where it is not installed."""
import importlib.util
import os
import subprocess
import sys
import unittest


def run(code, **environment):
    """`code`'s output in a fresh interpreter with `environment` over this one's (an empty value removes the name)."""
    env = {k: v for k, v in os.environ.items() if k not in ("NUMERICAL_NUM_THREADS", "OMP_NUM_THREADS")}
    env.update({k: v for k, v in environment.items() if v})
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    return result.stdout.strip()


class Environment(unittest.TestCase):
    def test_omp_num_threads_when_numerical_s_is_absent(self):
        count = "import numerical; print(numerical.get_num_threads())"
        # What a cap of k gives on this machine (at most its hardware threads: a CI runner may have two)
        capped = {k: run(f"import numerical; numerical.set_num_threads({k}); print(numerical.get_num_threads())")
                  for k in (2, 3)}
        self.assertEqual(run(count, OMP_NUM_THREADS="3"), capped[3])
        self.assertEqual(run(count, OMP_NUM_THREADS="3,2"), capped[3])  # nested levels: the outermost
        self.assertEqual(run(count, OMP_NUM_THREADS="3", NUMERICAL_NUM_THREADS="2"), capped[2])
        default = run(count)
        for ignored in ("x", "0", " ", "-1"):
            with self.subTest(value=ignored):
                self.assertEqual(run(count, OMP_NUM_THREADS=ignored), default)


@unittest.skipIf(importlib.util.find_spec("threadpoolctl") is None, "threadpoolctl not installed")
class Threadpoolctl(unittest.TestCase):
    def test_limits_apply_whichever_comes_first(self):
        for first in ("numerical", "threadpoolctl"):
            code = f"""
import {first}
import numerical, threadpoolctl
listed = [i for i in threadpoolctl.threadpool_info() if i["internal_api"] == "numerical"]
before = numerical.get_num_threads()
with threadpoolctl.threadpool_limits(1, user_api="blas"):
    inside = numerical.get_num_threads()
print(len(listed), listed[0]["user_api"], listed[0]["version"] == numerical.__version__, inside,
      numerical.get_num_threads() == before)
"""
            with self.subTest(first=first):
                self.assertEqual(run(code, NUMERICAL_NUM_THREADS="4"), "1 blas True 1 True")

    def test_numerical_does_not_import_threadpoolctl(self):
        self.assertEqual(run("import numerical, sys; print('threadpoolctl' in sys.modules)"), "False")


if __name__ == "__main__":
    unittest.main()
