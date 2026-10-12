"""A short, fixed campaign of fuzz/linalg.py on every gate run: the decompositions against NumPy on adversarial
matrices past the block and leaf boundaries, on every GEMM path this processor runs. `make fuzz-linalg` runs
long campaigns. Skipped when NumPy is not importable (CI installs it).
"""
import importlib.util
import os
import unittest

try:
    import numpy as np
except ImportError:
    np = None

TOOL = os.path.join(os.path.dirname(__file__), "..", "..", "fuzz", "linalg.py")


@unittest.skipIf(np is None or not os.path.exists(TOOL), "NumPy or the fuzz tool not available")
class FuzzLinalg(unittest.TestCase):
    def test_a_fixed_campaign_passes_on_every_path(self):
        spec = importlib.util.spec_from_file_location("fuzz_linalg", TOOL)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        import numerical

        paths = [path for path in tool.PATHS if numerical.limit_gemm_path(path) == path]
        failures = []
        try:
            with np.errstate(all="ignore"):
                for path in paths:
                    numerical.limit_gemm_path(path)
                    for index in range(40):
                        label, found = tool.run_case(20261012, index, 140)
                        failures += [f"[{path}] case {index} ({label}) {failure}" for failure in found]
        finally:
            numerical.limit_gemm_path("amx")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
