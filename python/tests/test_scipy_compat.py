"""Code written for NumPy and SciPy meets NumericAL's errors and warnings as theirs: ``except
numpy.linalg.LinAlgError`` (SciPy's is the same class) catches NumericAL's where NumPy is loaded, and a warnings filter
set on ``scipy.linalg.LinAlgWarning`` takes NumericAL's ill-conditioned solve; NumericAL's own classes keep working,
and without NumPy loaded its error is its own. lstsq called SciPy's way (cond, lapack_driver) gives a vector's
residue as SciPy does, a number."""
import pickle
import subprocess
import sys
import unittest
import warnings

import numerical

try:
    import numpy as np
    import scipy.linalg as sl
except ImportError:
    np = sl = None


@unittest.skipIf(sl is None, "SciPy not installed")
class ScipyCompat(unittest.TestCase):
    def test_numpy_s_except_catches_numericals_error(self):
        for caught in (np.linalg.LinAlgError, sl.LinAlgError, numerical.LinAlgError, numerical.error, ValueError):
            with self.subTest(caught=caught.__name__):
                with self.assertRaises(caught):
                    numerical.solve(np.zeros((2, 2)), np.ones(2))
        try:
            numerical.inv(np.zeros((2, 2)))
        except np.linalg.LinAlgError as error:
            self.assertIsInstance(error, numerical.LinAlgError)
            self.assertIn("singular", str(error).lower())
        self.assertIs(type(numerical.LinAlgError("x")), type(numerical.LinAlgError("y")))  # one class, made once
        again = pickle.loads(pickle.dumps(numerical.LinAlgError("singular matrix")))  # multiprocessing's way back
        self.assertIsInstance(again, np.linalg.LinAlgError)
        self.assertEqual(again.args, ("singular matrix",))

    def test_scipy_s_filter_takes_numericals_warning(self):
        ill = np.array([[1.0, 1.0], [1.0, 1.0 + 2.0 ** -52]])  # rcond near ε/4: warned, not singular
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            numerical.solve(ill, np.ones(2), assume_a="gen")
        self.assertTrue(any(issubclass(w.category, sl.LinAlgWarning) for w in caught))
        self.assertTrue(any(issubclass(w.category, numerical.LinAlgWarning) for w in caught))
        with warnings.catch_warnings():
            warnings.simplefilter("error", sl.LinAlgWarning)
            with self.assertRaises(sl.LinAlgWarning):
                numerical.solve(ill, np.ones(2), assume_a="gen")

    def test_without_numpy_the_error_is_numericals_own(self):
        code = ("import sys, numerical\n"
                "try:\n    numerical.solve([[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0])\n"
                "except numerical.LinAlgError as e:\n"
                "    print('numpy' in sys.modules, type(e) is numerical.LinAlgError)\n")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env={**__import__("os").environ, "PYTHONPATH": ":".join(sys.path)})
        self.assertEqual(out.stdout.split(), ["False", "True"])

    def test_lstsq_s_residue_as_scipy_s(self):
        a = np.random.default_rng(3).standard_normal((6, 3))
        b = np.random.default_rng(4).standard_normal(6)
        x, residue, rank, s = numerical.lstsq(a, b, cond=1e-12)
        expected = sl.lstsq(a, b, cond=1e-12)
        self.assertIsInstance(residue, float)
        self.assertAlmostEqual(residue, float(expected[1]), places=12)
        self.assertEqual(numerical.lstsq(a, b)[1].shape, (1,))  # NumPy's way: a (1,) array
        self.assertEqual(numerical.lstsq(a, np.ones((6, 2)), cond=1e-12)[1].shape, (2,))
        del x, rank, s


if __name__ == "__main__":
    unittest.main()
