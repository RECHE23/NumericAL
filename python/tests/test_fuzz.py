"""Differential fuzzing: NumericAL must agree with NumPy on random inputs.

Randomly generated (but seeded, so any failure reproduces) einsum specifications
and linear-algebra problems are evaluated with both NumericAL and NumPy; the results
must match within a floating-point tolerance. This is the broad net the
hand-written parity tests complement. Skipped when NumPy is absent; the CI Python
job installs it.
"""
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None

SEED = 20260615
LABELS = "abcd"


def random_einsum(rng, complex_data):
    """A valid einsum spec and operands: random labels, ranks, operands, output."""
    label_count = int(rng.integers(1, len(LABELS) + 1))
    used = list(LABELS[:label_count])
    dims = {ch: int(rng.integers(2, 5)) for ch in used}
    operand_count = int(rng.integers(1, 4))
    inputs = []
    arrays = []
    for _ in range(operand_count):
        rank = int(rng.integers(1, label_count + 1))
        subscript = list(rng.choice(used, size=rank, replace=False))  # distinct per operand
        inputs.append("".join(subscript))
        shape = tuple(dims[ch] for ch in subscript)
        block = rng.standard_normal(shape)
        if complex_data:
            block = block + 1j * rng.standard_normal(shape)
        arrays.append(block)
    present = sorted(set("".join(inputs)))
    keep = int(rng.integers(0, len(present) + 1))
    output = list(rng.choice(present, size=keep, replace=False)) if keep else []
    spec = ",".join(inputs) + "->" + "".join(output)
    return spec, arrays


@unittest.skipIf(np is None, "NumPy not installed")
class EinsumFuzz(unittest.TestCase):
    def _run(self, count, complex_data):
        rng = np.random.default_rng(SEED + (1 if complex_data else 0))
        for _ in range(count):
            spec, arrays = random_einsum(rng, complex_data)
            got = np.asarray(numerical.einsum(spec, *[numerical.asarray(a) for a in arrays]))
            want = np.einsum(spec, *arrays)
            self.assertTrue(np.allclose(got, want, rtol=1e-9, atol=1e-9),
                            msg=f"einsum '{spec}' on shapes {[a.shape for a in arrays]}")

    def test_real(self):
        self._run(500, complex_data=False)

    def test_complex(self):
        self._run(200, complex_data=True)


@unittest.skipIf(np is None, "NumPy not installed")
class LinalgFuzz(unittest.TestCase):
    def test_decompositions_match_numpy(self):
        rng = np.random.default_rng(SEED + 2)
        for _ in range(200):
            n = int(rng.integers(2, 7))
            base = rng.standard_normal((n, n))

            dominant = base + n * np.eye(n)  # invertible, well-conditioned
            self.assertTrue(np.isclose(numerical.determinant(numerical.asarray(dominant)),
                                       np.linalg.det(dominant), rtol=1e-6))
            self.assertTrue(np.allclose(np.asarray(numerical.inverse(numerical.asarray(dominant))),
                                        np.linalg.inv(dominant), rtol=1e-7, atol=1e-9))
            rhs = rng.standard_normal(n)
            self.assertTrue(np.allclose(
                np.asarray(numerical.solve(numerical.asarray(dominant), numerical.asarray(rhs))),
                np.linalg.solve(dominant, rhs), rtol=1e-7, atol=1e-9))

            spd = (base @ base.T) + n * np.eye(n)  # symmetric positive-definite
            self.assertTrue(np.allclose(np.asarray(numerical.cholesky(numerical.asarray(spd))),
                                        np.linalg.cholesky(spd), rtol=1e-7, atol=1e-9))

            sym = base + base.T
            values, vectors = numerical.eigh(numerical.asarray(sym))
            v = np.asarray(values)
            matrix = np.asarray(vectors)
            self.assertTrue(np.allclose(np.sort(v), np.linalg.eigvalsh(sym), rtol=1e-7, atol=1e-9))
            self.assertTrue(np.allclose(matrix @ np.diag(v) @ matrix.T, sym, rtol=1e-7, atol=1e-9))

            rows = n + int(rng.integers(0, 3))
            cols = int(rng.integers(1, n + 1))
            tall = rng.standard_normal((rows, cols))
            q, r = numerical.qr(numerical.asarray(tall))
            self.assertTrue(np.allclose(np.asarray(q) @ np.asarray(r), tall, rtol=1e-7, atol=1e-9))
            u, s, vh = numerical.svd(numerical.asarray(tall))
            reconstructed = np.asarray(u) @ np.diag(np.asarray(s)) @ np.asarray(vh)
            self.assertTrue(np.allclose(reconstructed, tall, rtol=1e-7, atol=1e-9))
            self.assertTrue(np.allclose(np.asarray(s),
                                        np.linalg.svd(tall, compute_uv=False), rtol=1e-7, atol=1e-9))
            target = rng.standard_normal(rows)
            self.assertTrue(np.allclose(
                np.asarray(numerical.lstsq(numerical.asarray(tall), numerical.asarray(target))[0]),
                np.linalg.lstsq(tall, target, rcond=None)[0], rtol=1e-6, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
