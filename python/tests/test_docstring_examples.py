"""The examples of the docstrings run as written: every ``>>>`` line of numerical's and numerical.fft's documentation
gives the output shown under it, so that a page rendered from them shows what a call answers."""

import doctest
import unittest

import numerical
import numerical.fft

_FLAGS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS


class DocstringExamples(unittest.TestCase):

    def test_numerical_examples_hold(self):
        failed, attempted = doctest.testmod(numerical, optionflags=_FLAGS, report=True)
        self.assertEqual(failed, 0, f"{failed} of {attempted} examples of numerical's docstrings failed (shown above)")
        self.assertGreaterEqual(attempted, 50, "the examples were not found: too few ran to hold the documentation")

    def test_fft_examples_hold(self):
        failed, attempted = doctest.testmod(numerical.fft, optionflags=_FLAGS, report=True)
        self.assertEqual(failed, 0, f"{failed} of {attempted} examples of numerical.fft's docstrings failed")


if __name__ == "__main__":
    unittest.main()
