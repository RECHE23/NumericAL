"""A short, fixed campaign of fuzz/edges.py on every gate run: element-wise operations and reductions against NumPy
on signed zeros, infinities, NaNs, values near overflow, empty axes and long ones. `make fuzz-edges` runs long
campaigns, and the weekly fuzz-long workflow the large shapes. Skipped when NumPy is not importable (CI installs it).
"""
import importlib.util
import os
import unittest

try:
    import numpy as np
except ImportError:
    np = None

TOOL = os.path.join(os.path.dirname(__file__), "..", "..", "fuzz", "edges.py")


@unittest.skipIf(np is None or not os.path.exists(TOOL), "NumPy or the fuzz tool not available")
class FuzzEdges(unittest.TestCase):
    def test_a_fixed_campaign_agrees_with_numpy(self):
        spec = importlib.util.spec_from_file_location("fuzz_edges", TOOL)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        failures = []
        for index in range(600):
            label, found = tool.run_case(20261021, index)
            failures += [f"case {index} ({label}) {failure}" for failure in found]
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
