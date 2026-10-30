"""On a free-threaded CPython (3.14t, 3.15t): importing numerical leaves the GIL disabled, and Python threads calling
its functions at once get the answers one thread gets, byte for byte. Skipped elsewhere."""

import sys
import sysconfig
import threading
import unittest

import numerical

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


@unittest.skipUnless(FREE_THREADED, "not a free-threaded CPython")
class FreeThreading(unittest.TestCase):
    def test_the_gil_stays_disabled(self):
        self.assertFalse(sys._is_gil_enabled())  # pylint: disable=protected-access

    @unittest.skipIf(np is None, "NumPy is not installed")
    def test_threads_get_one_threads_answers(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal((60, 60))
        s = a + a.T
        calls = [lambda: numerical.eigh(s)[0], lambda: numerical.solve(a, s), lambda: numerical.expm(a / 60),
                 lambda: numerical.fft.fft(a), lambda: numerical.qr(a)[1], lambda: a @ numerical.asarray(a)]
        expected = [np.asarray(call()).tobytes() for call in calls]
        mismatches = []

        def worker():
            for _ in range(50):
                for call, answer in zip(calls, expected):
                    if np.asarray(call()).tobytes() != answer:
                        mismatches.append(call)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(mismatches, [])


if __name__ == "__main__":
    unittest.main()
