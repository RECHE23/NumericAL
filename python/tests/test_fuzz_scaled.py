"""A short, fixed campaign of fuzz/scaled.py on every gate run: matrices scaled to 10^±150 (det, the eigenvalues'
sum, the solvers' backward errors), non-normal pencils' vectors, and fast paths against their general ones. `make
fuzz-scaled` runs long campaigns, the weekly fuzzing fresh seeds. Skipped when NumPy is not importable."""
import importlib.util
import os
import unittest

try:
    import numpy as np
except ImportError:
    np = None

TOOL = os.path.join(os.path.dirname(__file__), "..", "..", "fuzz", "scaled.py")


@unittest.skipIf(np is None or not os.path.exists(TOOL), "NumPy or the fuzz tool not available")
class FuzzScaled(unittest.TestCase):
    def test_a_fixed_campaign_passes(self):
        spec = importlib.util.spec_from_file_location("fuzz_scaled", TOOL)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        failures = []
        for index in range(60):
            failures += tool.run_case(20261117, index)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
