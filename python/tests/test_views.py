"""A Tensor's axes permuted (.T, .mT, transpose, swapaxes, moveaxis, matrix_transpose, a permuting einsum) and its basic
slices (ranges, steps, reversals, integer indices, and their compositions with permutations) are views, and so is a
diagonal: NumPy's elements in NumPy's order, and every operation on one (a permutation reduced along an axis as its
elements lie) the bits it gives on the same elements built contiguous,
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

_SLICES = [
    ("from row 1", lambda t: t[1:], lambda a: a[1:]),
    ("reversed", lambda t: t[::-1], lambda a: a[::-1]),
    ("every other column", lambda t: t[..., ::2], lambda a: a[..., ::2]),
    ("inner block backwards", lambda t: t[1:, -2::-2], lambda a: a[1:, -2::-2]),
    ("first row", lambda t: t[0], lambda a: a[0]),
    ("a column", lambda t: t[:, 1], lambda a: a[:, 1]),
    ("columns then T", lambda t: t[:, 1:4].T, lambda a: a[:, 1:4].T),
    ("T then rows", lambda t: t.T[::2], lambda a: a.T[::2]),
    ("slice of a slice", lambda t: t[::-1][1:, ::3], lambda a: a[::-1][1:, ::3]),
    ("the whole", lambda t: t[:], lambda a: a),
]

_VIEWS = _PERMUTATIONS + _SLICES

_FLOAT_OPERATIONS = [
    ("add", lambda x: x + 1.5),
    ("square", lambda x: x * x),
    ("exp", numerical.exp),
    ("sum", numerical.sum),
    ("sum axis 0", lambda x: numerical.sum(x, axis=0)),
    ("sum axis -1", lambda x: numerical.sum(x, axis=-1)),
    ("cumsum", lambda x: numerical.cumsum(x, axis=-1)),
    ("max", numerical.max),
    ("mean axis 1", lambda x: numerical.mean(x, axis=1)),
    ("max axis 0", lambda x: numerical.max(x, axis=0)),
    ("min axis -1", lambda x: numerical.min(x, axis=-1)),
    ("any axis 0", lambda x: numerical.any(x > 0.5, axis=0)),
    ("all axis -1", lambda x: numerical.all(x, axis=-1)),
    ("diagonal", lambda x: numerical.diagonal(x, offset=-1)),
    ("trace", numerical.trace),
    ("clip", lambda x: numerical.clip(x, -0.5, 0.5)),
    ("expm1", numerical.expm1),
    ("floor", numerical.floor),
    ("log1p", lambda x: numerical.log1p(x * x)),
    ("cumsum axis 0", lambda x: numerical.cumsum(x, axis=0)),
    ("cumprod axis 0", lambda x: numerical.cumprod(x, axis=0)),
    ("sort axis 0", lambda x: numerical.sort(x, axis=0)),
    ("argsort axis 0", lambda x: numerical.argsort(x, axis=0)),
    ("partition axis 0", lambda x: numerical.partition(x, 0, axis=0)),
    ("argpartition axis 0", lambda x: numerical.argpartition(x, 0, axis=0)),
    ("argmax axis 0", lambda x: numerical.argmax(x, axis=0)),
    ("argmin axis -1", lambda x: numerical.argmin(x, axis=-1)),
    ("count_nonzero axis 0", lambda x: numerical.count_nonzero(x > 0.5, axis=0)),
    ("min", numerical.min),
    ("any", lambda x: numerical.any(x > 0.5)),
    ("all", numerical.all),
    ("count_nonzero", lambda x: numerical.count_nonzero(x > 0.5)),
    ("maximum of a number", lambda x: numerical.maximum(x, 0.25)),
    ("minimum of a number", lambda x: numerical.minimum(x, -0.25)),
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
    ("number minus", lambda x: 2.0 - x),
    ("less than a number", lambda x: x.real < 0.25 if x.dtype.startswith("complex") else x < 0.25),
    ("plus a view alike", lambda x: x + x * 0.5),
    ("prod axis 0", lambda x: numerical.prod(x, axis=0)),
    ("triu", lambda x: numerical.triu(x, 1)),
    ("tril", lambda x: numerical.tril(x, -2)),
    ("fft axis 0", lambda x: numerical.fft.fft(x, axis=0)),
    ("ifft axis 0 padded", lambda x: numerical.fft.ifft(x, 9, axis=0)),
    ("rfft axis 0", lambda x: numerical.fft.rfft(x, axis=0)),
    ("ihfft axis 0", lambda x: numerical.fft.ihfft(x, axis=0)),
    ("dct axis 0", lambda x: numerical.fft.dct(x, axis=0)),
    ("fftshift", numerical.fft.fftshift),
    ("tolist", lambda x: numerical.asarray(x.tolist(), dtype=x.dtype)),
    ("expand_dims", lambda x: numerical.expand_dims(x, 1)),
    ("squeezed back", lambda x: numerical.squeeze(numerical.expand_dims(x, 0), 0)),
    ("atleast_3d", numerical.atleast_3d),
    ("rows joined", lambda x: numerical.reshape(x, (-1, x.shape[-1]))),
    ("last axis split", lambda x: numerical.reshape(x, x.shape[:-1] + (1, x.shape[-1], 1))),
    ("sum axis 0 kept", lambda x: numerical.sum(x, axis=0, keepdims=True)),
    ("prod axis -1", lambda x: numerical.prod(x, axis=-1)),
    ("cumprod axis -1", lambda x: numerical.cumprod(x, axis=-1)),
    ("where of a number", lambda x: numerical.where(x.real > 0, x, 0.25)),
    ("where of views alike", lambda x: numerical.where(x.real > 0, x * 2.0, x)),
    ("where of a mask alone", lambda x: numerical.where(x.real > 0, 1.5, -1.5)),
    ("nansum axis 0", lambda x: numerical.nansum(x, axis=0)),
    ("nansum axis -1 with NaNs", lambda x: numerical.nansum(numerical.where(x.real > 1.0, float("nan"), x), axis=-1)),
    ("array_equal", lambda x: numerical.array_equal(x, x * 1.0)),
    ("array_equal apart", lambda x: numerical.array_equal(x, x + 1.0)),
]


@unittest.skipIf(np is None, "NumPy not installed")
class TestViews(unittest.TestCase):

    def test_a_copy_of_a_view_holds_its_own_elements(self):
        # A Tensor holding its elements row-major is its own copy; a view's copy (copy(), asarray(copy=True),
        # astype, from_dlpack(copy=True), copy.copy) holds its elements row-major, without the array the view reads.
        import copy as copying

        base = numerical.asarray(np.arange(4000.0).reshape(40, 100))
        for view in (base.T, base[::2], base[:, 3:90]):
            for copied in (view.copy(), numerical.copy(view), numerical.asarray(view, copy=True),
                           numerical.astype(view, view.dtype), view.astype(view.dtype),
                           numerical.from_dlpack(view, copy=True), copying.copy(view), copying.deepcopy(view)):
                self.assertIsNot(copied, view)
                self.assertEqual(copied.shape, view.shape)
                self.assertTrue(np.array_equal(np.asarray(copied), np.asarray(view)))
                self.assertTrue(np.asarray(copied).flags["C_CONTIGUOUS"])
            self.assertIs(numerical.astype(view, view.dtype, copy=False), view)
            self.assertIs(numerical.asarray(view), view)
        for same in (base.copy(), numerical.copy(base), numerical.asarray(base, copy=True), base.astype(base.dtype),
                     numerical.from_dlpack(base, copy=True), copying.copy(base)):
            self.assertIs(same, base)
    def _tensors(self):
        rng = np.random.default_rng(20261114)
        for dtype in ("float64", "float32", "complex128", "complex64"):
            for shape in ((7, 5), (3, 4, 6), (1, 5), (40, 33), (9, 70, 3)):
                a = rng.standard_normal(shape)
                if dtype.startswith("complex"):
                    a = a + 1j * rng.standard_normal(shape)
                yield dtype, a.astype(dtype)

    def test_the_elements_in_numpys_order(self):
        for dtype, a in self._tensors():
            t = numerical.asarray(a)
            for name, permute, expected in _VIEWS:
                with self.subTest(dtype=dtype, shape=a.shape, view=name):
                    view = permute(t)
                    want = expected(a)
                    self.assertEqual(view.shape, want.shape)
                    self.assertEqual(_bits(view), _bits(np.ascontiguousarray(want)))
                    self.assertEqual(_bits(np.from_dlpack(view)), _bits(np.ascontiguousarray(want)))

    def test_numpy_reads_a_view_where_it_lies(self):
        # np.asarray and np.from_dlpack read a transpose in the Tensor's own buffer, read-only, along its strides; the
        # array keeps the elements alive after the view and the tensor are gone; a copy asked for is row-major.
        a = np.random.default_rng(3).standard_normal((30, 20, 7))
        t = numerical.asarray(a)
        for view, want in ((t.T, a.T), (numerical.transpose(t, (1, 2, 0)), np.transpose(a, (1, 2, 0))),
                           (numerical.swapaxes(t, 0, 1), np.swapaxes(a, 0, 1))):
            for read in (np.asarray, np.from_dlpack):
                got = read(view)
                self.assertTrue(np.shares_memory(got, np.asarray(t)))
                self.assertFalse(got.flags.writeable)
                np.testing.assert_array_equal(got, want)
            copied = np.from_dlpack(view, copy=True)
            self.assertTrue(copied.flags.c_contiguous and not np.shares_memory(copied, np.asarray(t)))
            np.testing.assert_array_equal(copied, np.asarray(view))
        kept = np.asarray(numerical.asarray(a).T)  # neither the tensor nor its view is held any more
        np.testing.assert_array_equal(kept, a.T)
        # A slice past 512 elements likewise, from its first element, a reversed axis's stride negative; a smaller one
        # is a copy of its own.
        a = np.random.default_rng(4).standard_normal((60, 40, 9))
        t = numerical.asarray(a)
        for view, want in ((t[1:, ::-2, 3], a[1:, ::-2, 3]), (t[::-1, 2], a[::-1, 2]), (t[5:9], a[5:9]), (t[:, 1:], a[:, 1:])):
            reversed_axis = any(stride < 0 for stride in want.strides)
            for read in (np.asarray, np.from_dlpack):
                got = read(view)
                # DLPack shares no reversed view (PyTorch refuses a negative stride): a copy of its own
                shared = read is np.asarray or not reversed_axis
                self.assertEqual(np.shares_memory(got, np.asarray(t)), shared)
                if shared:
                    self.assertFalse(got.flags.writeable)
                np.testing.assert_array_equal(got, want)
            if reversed_axis:
                with self.assertRaises(BufferError):
                    view.__dlpack__(max_version=(1, 0), copy=False)
        small = np.asarray(t[2:4, 1:3])
        self.assertFalse(np.shares_memory(small, np.asarray(t)))
        np.testing.assert_array_equal(small, a[2:4, 1:3])
        kept = np.asarray(numerical.asarray(a)[::-1, 3])
        np.testing.assert_array_equal(kept, a[::-1, 3])

    def test_each_operation_gives_the_contiguous_bits(self):
        for dtype, a in self._tensors():
            t = numerical.asarray(a)
            for name, permute, _ in _VIEWS:
                for operation, run in _FLOAT_OPERATIONS:
                    view = permute(t)
                    if operation in ("svdvals", "qr") and len(view.shape) != 2:
                        continue
                    if operation in ("times its transpose", "matmul on the right") and len(view.shape) < 2:
                        continue
                    with self.subTest(dtype=dtype, shape=a.shape, view=name, operation=operation):
                        built = numerical.asarray(np.array(view))  # the same elements, contiguous
                        self.assertEqual(_outcome(run, view), _outcome(run, built))

    def test_products_of_large_views(self):
        # Past the Gram products' size (a @ a.T by its triangle), on all threads and on one: the copies' bits.
        rng = np.random.default_rng(11)
        for dtype in ("float64", "complex128", "float32"):
            a = numerical.asarray(rng.standard_normal((400, 390)).astype(dtype))
            b = numerical.asarray(rng.standard_normal((400, 390)).astype(dtype))
            c = numerical.asarray(rng.standard_normal((390, 400)).astype(dtype))
            x = numerical.asarray(rng.standard_normal(400).astype(dtype))
            built = numerical.asarray(np.array(a.T))
            products = [("aT@a", lambda: a.T @ a, lambda: built @ a), ("a@aT", lambda: a @ a.T, lambda: a @ built),
                        ("aT@b", lambda: a.T @ b, lambda: built @ b), ("b@aT", lambda: b @ a.T, lambda: b @ built),
                        ("aT@cT", lambda: a.T @ c.T, lambda: built @ numerical.asarray(np.array(c.T))),
                        ("x@a", lambda: x @ a, lambda: x @ a), ("aT@x", lambda: a.T @ x, lambda: built @ x),
                        ("aT@narrow", lambda: a.T @ numerical.ones((400, 3), dtype=dtype),
                         lambda: built @ numerical.ones((400, 3), dtype=dtype)),
                        # slices read where they lie: rows apart (a[:, :300]), columns apart (a[:, :300].T)
                        ("a[:, :300]@c[:300]", lambda: a[:, :300] @ c[:300],
                         lambda: numerical.asarray(np.array(a[:, :300])) @ numerical.asarray(np.array(c[:300]))),
                        ("a[:, :300].T@b", lambda: a[:, :300].T @ b,
                         lambda: numerical.asarray(np.array(a[:, :300].T)) @ b),
                        ("a[::2]@c", lambda: a[::2] @ c, lambda: numerical.asarray(np.array(a[::2])) @ c),
                        ("b@a[1:391].T", lambda: b @ a[1:391].T,
                         lambda: b @ numerical.asarray(np.array(a[1:391].T)))]
            for threads in (0, 1):
                previous = numerical.get_num_threads()
                if threads:
                    numerical.set_num_threads(threads)
                try:
                    for name, view, copy in products:
                        with self.subTest(dtype=dtype, threads=threads or "all", product=name):
                            self.assertEqual(_bits(view()), _bits(copy()))
                finally:
                    numerical.set_num_threads(previous)

    def test_bool_and_int64_views(self):
        x = np.arange(960).reshape(6, 8, 20)  # past the elements a slice copies at once
        for source in (x, x % 3 == 0):
            t = numerical.asarray(source)
            for name, permute, expected in _VIEWS:
                if name == "einsum" and source.dtype == bool:
                    continue
                with self.subTest(dtype=str(source.dtype), view=name):
                    view = permute(t)
                    self.assertEqual(view.tolist(), expected(source).tolist())
                    for op in ("sum", "max", "min", "any", "all", "count_nonzero"):
                        self.assertEqual(numerical.asarray(getattr(numerical, op)(view)).tolist(),
                                         np.asarray(getattr(np, op)(expected(source))).tolist())
                    for op in ("sum", "prod", "max", "min", "any", "all", "mean"):
                        for axis in range(view.ndim):
                            self.assertEqual(getattr(numerical, op)(view, axis=axis).tolist(),
                                             getattr(np, op)(expected(source), axis=axis).tolist())

    def test_where_and_array_equal_of_operands_laid_apart(self):
        # Operands of one shape laid in different orders, contiguous beside a transpose, a single element of more axes
        # than the others, every one a single element, a condition broadcast: NumPy's answer whichever path reads them.
        rng = np.random.default_rng(12)
        a = rng.standard_normal((6, 6, 6))
        t = numerical.asarray(a)
        orders = ((0, 1, 2), (1, 0, 2), (2, 1, 0), (2, 0, 1))
        for c_axes in orders:
            for x_axes in orders:
                for y_axes in orders:
                    with self.subTest(c=c_axes, x=x_axes, y=y_axes):
                        c = numerical.transpose(t, c_axes) > 0
                        x = numerical.transpose(t * 2.0, x_axes)
                        y = numerical.transpose(t, y_axes)
                        want = np.where(np.transpose(a, c_axes) > 0, np.transpose(a * 2.0, x_axes), np.transpose(a, y_axes))
                        self.assertEqual(_bits(numerical.where(c, x, y)), _bits(want))
                        self.assertEqual(numerical.array_equal(x, y * 2.0), np.array_equal(np.transpose(a * 2.0, x_axes),
                                                                                           np.transpose(a, y_axes) * 2.0))
        one = numerical.asarray(np.full((1, 1, 1, 1), 3.0))
        for c, x, y, want in ((t.T > 0, one, t.T, np.where(a.T > 0, np.full((1, 1, 1, 1), 3.0), a.T)),
                              (numerical.asarray(True), one, one, np.full((1, 1, 1, 1), 3.0)),
                              (t[0] > 0, t.T, 1.0, np.where(a[0] > 0, a.T, 1.0)),
                              (t.T > 0, 1.0, numerical.asarray([[[2.0]]]), np.where(a.T > 0, 1.0, [[[2.0]]]))):
            with self.subTest(c=getattr(c, "shape", ()), x=getattr(x, "shape", ())):
                self.assertEqual(_bits(numerical.where(c, x, y)), _bits(want))

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
