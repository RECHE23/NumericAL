"""A Tensor's axes permuted (.T, .mT, transpose, swapaxes, moveaxis, matrix_transpose, a permuting einsum) are a view:
NumPy's elements in NumPy's order, and every operation on it the bits it gives on the same elements built contiguous,
whatever the dtype, from one thread or several. The NumPy comparisons are skipped without NumPy."""
import threading
import unittest

import numerical

try:
    import numpy as np
except ImportError:
    np = None


def _bits(x):
    """A result (a Tensor, a number, or a tuple of them) as comparable bytes, every NaN one NaN."""
    if isinstance(x, tuple):
        return tuple(_bits(each) for each in x)
    a = np.asarray(x)
    a = np.where(np.isnan(a), np.nan, a) if a.dtype.kind in "fc" else a
    return a.dtype, a.shape, a.tobytes()


def _outcome(run, x):
    """What `run(x)` gives: its bits (_bits), or the type of the exception it raises."""
    try:
        return _bits(run(x))
    except Exception as error:  # noqa: BLE001 -- the error is the outcome compared
        return type(error)


_PERMUTATIONS = [
    ("T", lambda t: t.T, lambda a: a.T),
    ("mT", lambda t: t.mT, lambda a: np.swapaxes(a, -1, -2)),
    ("transpose", lambda t: numerical.transpose(t, (2, 0, 1)) if len(t.shape) == 3 else numerical.transpose(t),
     lambda a: np.transpose(a, (2, 0, 1)) if a.ndim == 3 else a.T),
    ("swapaxes", lambda t: numerical.swapaxes(t, 0, -1), lambda a: np.swapaxes(a, 0, -1)),
    ("moveaxis", lambda t: numerical.moveaxis(t, 0, -1), lambda a: np.moveaxis(a, 0, -1)),
    ("matrix_transpose", lambda t: numerical.matrix_transpose(t), lambda a: np.swapaxes(a, -1, -2)),
    ("einsum", lambda t: numerical.einsum("ij->ji" if len(t.shape) == 2 else "ijk->kji", t),
     lambda a: a.T),
    ("T of T", lambda t: t.T.T, lambda a: a),
]

_FLOAT_OPERATIONS = [
    ("add", lambda x: x + 1.5),
    ("square", lambda x: x * x),
    ("exp", numerical.exp),
    ("sum", numerical.sum),
    ("sum axis 0", lambda x: numerical.sum(x, axis=0)),
    ("sum axis -1", lambda x: numerical.sum(x, axis=-1)),
    ("cumsum", lambda x: numerical.cumsum(x, axis=-1)),
    ("max", numerical.max),
    ("sort", lambda x: numerical.sort(x, axis=-1)),
    ("rows", lambda x: x[1:3]),
    ("every other column", lambda x: x[..., ::2]),
    ("flattened", lambda x: numerical.reshape(x, (-1,))),
    ("einsum sum", lambda x: numerical.einsum("ij->" if len(x.shape) == 2 else "ijk->", x)),
    ("times its transpose", lambda x: x @ numerical.matrix_transpose(x)),
    ("matmul", lambda x: x @ numerical.ones((x.shape[-1], 3), dtype=x.dtype)),
    ("matmul on the right", lambda x: numerical.ones((2, x.shape[-2]), dtype=x.dtype) @ x),
    ("vector", lambda x: x @ numerical.ones((x.shape[-1],), dtype=x.dtype)),
    ("svdvals", numerical.linalg.svdvals),
    ("qr", lambda x: tuple(numerical.linalg.qr(x))),
    ("norm", numerical.linalg.norm),
    ("fft", numerical.fft.fft),
    ("tolist", lambda x: numerical.asarray(x.tolist(), dtype=x.dtype)),
]


@unittest.skipIf(np is None, "NumPy not installed")
class TestViews(unittest.TestCase):
    def _tensors(self):
        rng = np.random.default_rng(20261114)
        for dtype in ("float64", "float32", "complex128", "complex64"):
            for shape in ((7, 5), (3, 4, 6), (1, 5), (40, 33)):
                a = rng.standard_normal(shape)
                if dtype.startswith("complex"):
                    a = a + 1j * rng.standard_normal(shape)
                yield dtype, a.astype(dtype)

    def test_the_elements_in_numpys_order(self):
        for dtype, a in self._tensors():
            t = numerical.asarray(a)
            for name, permute, expected in _PERMUTATIONS:
                with self.subTest(dtype=dtype, shape=a.shape, permutation=name):
                    view = permute(t)
                    want = expected(a)
                    self.assertEqual(view.shape, want.shape)
                    self.assertEqual(_bits(view), _bits(np.ascontiguousarray(want)))
                    self.assertEqual(_bits(np.from_dlpack(view)), _bits(np.ascontiguousarray(want)))

    def test_each_operation_gives_the_contiguous_bits(self):
        for dtype, a in self._tensors():
            t = numerical.asarray(a)
            for name, permute, _ in _PERMUTATIONS:
                for operation, run in _FLOAT_OPERATIONS:
                    view = permute(t)
                    if operation == "svdvals" and len(view.shape) != 2:
                        continue
                    if operation == "qr" and len(view.shape) != 2:
                        continue
                    with self.subTest(dtype=dtype, shape=a.shape, permutation=name, operation=operation):
                        built = numerical.asarray(np.array(view))  # the same elements, contiguous
                        self.assertEqual(_outcome(run, view), _outcome(run, built))

    def test_bool_and_int64_views(self):
        x = np.arange(24).reshape(2, 3, 4)
        for source in (x, x % 3 == 0):
            t = numerical.asarray(source)
            for name, permute, expected in _PERMUTATIONS:
                if name == "einsum" and source.dtype == bool:
                    continue
                with self.subTest(dtype=str(source.dtype), permutation=name):
                    view = permute(t)
                    self.assertEqual(view.tolist(), expected(source).tolist())
                    self.assertEqual(numerical.sum(view, axis=0).tolist(), np.sum(expected(source), axis=0).tolist())

    def test_a_view_shared_by_threads(self):
        a = np.random.default_rng(7).standard_normal((300, 200))
        want = _bits(np.ascontiguousarray(a.T) * 2.0)
        for _ in range(5):
            view = numerical.asarray(a).T
            got = []

            def worker():
                got.append(_bits(view * 2.0))

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(got, [want] * 8)


if __name__ == "__main__":
    unittest.main()
