"""The package is typed (PEP 561): its marker and stubs ship beside the module, wherever it is installed."""
import os
import unittest

import numerical


class Typing(unittest.TestCase):
    def test_the_marker_and_the_stubs_ship_with_the_package(self):
        here = os.path.dirname(numerical.__file__)
        for name in ("py.typed", "__init__.pyi", "_numerical.pyi"):
            with self.subTest(name=name):
                self.assertTrue(os.path.isfile(os.path.join(here, name)))


if __name__ == "__main__":
    unittest.main()
