"""schur(a, output) against scipy.linalg.schur where it is installed, and against its defining properties everywhere:
a = Z T Zᴴ, Z orthogonal (unitary), T quasi-triangular (real) or triangular (complex)."""

import unittest

import numerical

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import scipy.linalg as sl
except ImportError:  # pragma: no cover
    sl = None


def _drawn(n, dtype, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n))
    if dtype.startswith("complex"):
        x = x + 1j * rng.standard_normal((n, n))
    return x.astype(dtype)


def _clear_of_boundaries(n, dtype, seed):
    """Q T₀ Qᴴ for a random orthogonal (unitary) Q and T₀ upper (quasi-)triangular, its eigenvalues cycling through
    −2.5, −0.5 ± 0.5i, 0.4, 0.2 ± 2i, −0.3, each cycle's moved off the last's away from the boundaries (distinct, so
    well conditioned): real parts from 0.2 and moduli from 0.2 off 0 and 1."""
    rng = np.random.default_rng(seed)
    complex_entries = dtype.startswith("complex")
    t0 = np.triu(rng.standard_normal((n, n)) * 0.1, 1).astype(complex if complex_entries else float)
    i = 0
    while i < n:
        slot, c = i % 6, i // 6  # each cycle's values apart from the last's, away from the boundaries
        if slot in (1, 4):
            re, im = (-0.5 - 0.01 * c, 0.5 - 0.01 * c) if slot == 1 else (0.2 + 0.05 * c, 2.0 + 0.05 * c)
            if complex_entries:
                t0[i, i] = re + 1j * im
            elif i + 1 < n:
                t0[i:i + 2, i:i + 2] = [[re, im], [-im, re]]
                i += 2
                continue
            else:
                t0[i, i] = 3.0
        else:
            t0[i, i] = {0: -2.5 - 0.05 * c, 2: 3.0 + 0.05 * c, 3: 0.4 + 0.02 * c, 5: -0.3 - 0.02 * c}[slot]
        i += 1
    x = rng.standard_normal((n, n))
    if complex_entries:
        x = x + 1j * rng.standard_normal((n, n))
    q, _ = np.linalg.qr(x)
    return (q @ t0 @ q.conj().T).astype(dtype)


_SELECTIONS = {
    "lhp": lambda z: z.real < 0.0,
    "rhp": lambda z: z.real >= 0.0,
    "iuc": lambda z: abs(z) <= 1.0,
    "ouc": lambda z: abs(z) > 1.0,
}


def _selected_count(a, sort, complex_out):
    """The eigenvalues of `a` a sort selects (each of a pair alike under these selections), from its eigenvalues."""
    values = np.linalg.eigvals(np.asarray(a, dtype=complex))
    if isinstance(sort, str):
        chosen = _SELECTIONS[sort]
    elif complex_out:
        chosen = sort
    else:
        def chosen(z):
            return sort(z.real, z.imag)
    return sum(1 for z in values if chosen(z))


def _scipy_sorts_by_modulus():
    """Whether SciPy's "iuc" takes a real matrix's eigenvalues whole (±2i outside the unit circle), as since 1.13 or so;
    an older one took their real parts alone."""
    return sl.schur(np.array([[0.0, -2.0], [2.0, 0.0]]), sort="iuc")[2] == 0


@unittest.skipIf(np is None, "NumPy is not installed")
class Schur(unittest.TestCase):
    def test_every_dtype_and_output(self):
        for dtype in ("float64", "complex128", "float32", "complex64"):
            tolerance = 1e-12 if dtype in ("float64", "complex128") else 1e-4
            for output in ("real", "r", "complex", "c"):
                complex_out = dtype.startswith("complex") or output in ("complex", "c")
                expected = {("float64", False): np.float64, ("float64", True): np.complex128,
                            ("float32", False): np.float32, ("float32", True): np.complex64}
                kind = "float64" if dtype in ("float64", "complex128") else "float32"
                for n in (1, 6, 120):
                    a = _drawn(n, dtype, n)
                    t, z = (np.asarray(m) for m in numerical.schur(a, output))
                    self.assertEqual(t.dtype, expected[(kind, complex_out)])
                    self.assertEqual(z.dtype, expected[(kind, complex_out)])
                    scale = np.abs(a).max() * n
                    self.assertLess(np.abs(z @ t @ z.conj().T - a).max(), tolerance * scale)
                    self.assertLess(np.abs(z.conj().T @ z - np.eye(n)).max(), tolerance * n)
                    self.assertTrue(np.all(np.tril(t, -1 if complex_out else -2) == 0))

    def test_errors(self):
        with self.assertRaises(numerical.LinAlgError):
            numerical.schur(np.ones((2, 3)))
        with self.assertRaises(numerical.LinAlgError):
            numerical.schur(np.array([[np.nan]]))
        with self.assertRaises(ValueError):
            numerical.schur(np.eye(2), "x")

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_against_scipy(self):
        for dtype in ("float64", "complex128"):
            for n in (5, 60):
                a = _drawn(n, dtype, 40 + n)
                for output in ("real", "complex"):
                    t, z = (np.asarray(m) for m in numerical.schur(a, output))
                    expected_t, expected_z = sl.schur(a, output)
                    self.assertEqual(t.dtype, expected_t.dtype)
                    self.assertEqual(t.shape, expected_t.shape)
                    # The same eigenvalues on T's diagonal (and blocks) as SciPy's, as sets.
                    # (Each to its nearest: a pair's halves sort by real parts equal to rounding.)
                    ours = np.linalg.eigvals(t)
                    theirs = np.linalg.eigvals(expected_t)
                    nearest = np.abs(ours[:, None] - theirs[None, :]).min(axis=1)
                    self.assertLess(nearest.max(), 1e-9 * np.abs(theirs).max())
                    self.assertLess(np.abs(np.sort(np.abs(ours)) - np.sort(np.abs(theirs))).max(), 1e-9 * np.abs(theirs).max())

    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_isolated_eigenvalues_as_scipy(self):
        # A triangular matrix is its own Schur form, Z the identity, as dgees's permutations leave it; a permuted
        # triangular one gives the diagonal SciPy gives, in its order.
        for dtype in ("float64", "complex128"):
            a = np.triu(_drawn(7, dtype, 3)) + np.diag(np.arange(1.0, 8.0))
            t, z = (np.asarray(m) for m in numerical.schur(a))
            self.assertTrue(np.array_equal(t, a))
            self.assertTrue(np.array_equal(z, np.eye(7)))
            order = np.array([3, 6, 0, 5, 1, 4, 2])
            permuted = a[np.ix_(order, order)]
            t, z = (np.asarray(m) for m in numerical.schur(permuted))
            expected_t = sl.schur(permuted)[0]
            self.assertTrue(np.array_equal(np.diag(t), np.diag(expected_t)))
            self.assertLess(np.abs(z @ t @ z.conj().T - permuted).max(), 1e-12 * np.abs(a).max() * 7)


    @unittest.skipIf(sl is None, "SciPy is not installed")
    def test_sort_moves_the_selected_eigenvalues_first_as_scipy(self):
        # Each selection SciPy names, and callables (the real and imaginary parts for a real T, the eigenvalue for a
        # complex one), on matrices whose eigenvalues lie clear of the boundaries (rounding moves none across, which
        # either library would refuse): the same count, the same leading eigenvalues, a = Z T Zᴴ still.
        selections = ["lhp", "rhp", "iuc", "ouc"]
        for dtype in ("float64", "complex128", "float32"):
            tolerance = 1e-10 if dtype != "float32" else 1e-3
            for output in ("real", "complex"):
                complex_out = dtype.startswith("complex") or output == "complex"
                chosen = (lambda z: z.real > 0.1) if complex_out else (lambda x, y: x > 0.1)
                for n in (1, 7, 60):
                    a = _clear_of_boundaries(n, dtype, n + 1)
                    for sort in selections + [chosen]:
                        with self.subTest(dtype=dtype, output=output, n=n, sort=sort if isinstance(sort, str) else "callable"):
                            t, z, sdim = numerical.schur(a, output, sort=sort)
                            t, z = np.asarray(t), np.asarray(z)
                            self.assertEqual(sdim, _selected_count(a, sort, complex_out))
                            ts, _, sdim_scipy = sl.schur(a, output, sort=sort)
                            if complex_out or sort not in ("iuc", "ouc") or _scipy_sorts_by_modulus():
                                self.assertEqual(sdim, sdim_scipy)
                            else:  # an older SciPy took a real matrix's real parts alone: compare its count only
                                continue
                            np.testing.assert_allclose(z @ t @ z.conj().T, a, atol=tolerance * n)
                            ours = list(np.linalg.eigvals(t[:sdim, :sdim].astype(complex)))
                            theirs = list(np.linalg.eigvals(ts[:sdim, :sdim].astype(complex)))
                            for value in ours:  # the same set, each matched to its nearest
                                nearest = min(range(len(theirs)), key=lambda i: abs(theirs[i] - value))
                                self.assertLess(abs(theirs.pop(nearest) - value), tolerance * 10)

    def test_sort_refusals(self):
        a = _drawn(4, "float64", 3)
        with self.assertRaises(ValueError):
            numerical.schur(a, sort="left")
        with self.assertRaises(ValueError):
            numerical.schur(a, sort=3)

        def failing(x, y):
            raise RuntimeError("chosen badly")

        with self.assertRaises(RuntimeError):
            numerical.schur(a, sort=failing)

        class Unanswerable:
            def __bool__(self):
                raise TypeError("no truth")

        with self.assertRaises(TypeError):
            numerical.schur(a, sort=lambda x, y: Unanswerable())
        with self.assertRaises(numerical.LinAlgError):
            numerical.schur(np.ones((2, 3)), sort="lhp")
        t, z, sdim = numerical.schur(np.zeros((0, 0)), sort="lhp")
        self.assertEqual((np.asarray(t).shape, np.asarray(z).shape, sdim), ((0, 0), (0, 0), 0))
        # the boundaries as SciPy draws them: 0 in the right half plane, the unit circle inside
        for sort, count in (("lhp", 1), ("rhp", 2), ("iuc", 2), ("ouc", 1)):
            for output in ("real", "complex"):
                values = np.diag([0.0, -1.0, 2.0]) if sort in ("lhp", "rhp") else np.diag([1.0, 2.0, 0.5])
                self.assertEqual(numerical.schur(values, output, sort=sort)[2], count)
                if sl is not None:
                    self.assertEqual(sl.schur(values, output, sort=sort)[2], count)
        # a real pair selected by one of its two moves whole and counts twice
        rotation = np.array([[3.0, 1.0, 0.5], [0.0, 0.0, -2.0], [0.0, 2.0, 0.0]])
        t, z, sdim = numerical.schur(rotation, sort=lambda x, y: y < 0)
        self.assertEqual(sdim, 2)
        self.assertAlmostEqual(float(np.asarray(t)[2, 2]), 3.0, places=12)


if __name__ == "__main__":
    unittest.main()
