"""NumericAL Python binding tests (stdlib unittest, no third-party requirement).

The NumPy / DLPack interop is verified twice: once with pure-Python checks that
need nothing installed (the __array_interface__ dict, the DLPack capsule and
device), and once against NumPy proper when it happens to be importable.
"""
import unittest

import numerical

try:
    import numpy as np
except ImportError:  # NumPy is optional; the interop round-trips are then skipped
    np = None


class TensorBasics(unittest.TestCase):
    def test_construct_and_introspect(self):
        t = numerical.Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(t.ndim, 2)
        self.assertEqual(t.size, 6)
        self.assertEqual(t.dtype, "float64")
        self.assertEqual(t.tolist(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_default_shape_is_one_dimensional(self):
        v = numerical.Tensor([1, 2, 3])
        self.assertEqual(v.shape, (3,))
        self.assertEqual(v.tolist(), [1.0, 2.0, 3.0])

    def test_complex_dtype(self):
        t = numerical.Tensor([1 + 2j, 3 - 1j], dtype="complex128")
        self.assertEqual(t.dtype, "complex128")
        self.assertEqual(t.tolist(), [1 + 2j, 3 - 1j])

    def test_rank_zero_tolist_is_a_scalar(self):
        s = numerical.einsum("i,i->", numerical.Tensor([1, 2, 3]), numerical.Tensor([1, 2, 3]))
        self.assertEqual(s.ndim, 0)
        self.assertEqual(s.shape, ())
        self.assertEqual(s.tolist(), 14.0)

    def test_repr(self):
        self.assertIn("dtype=float64", repr(numerical.Tensor([1, 2])))

    def test_bad_dtype_raises(self):
        with self.assertRaises(numerical.error):
            numerical.Tensor([1, 2], dtype="float32")

    def test_shape_mismatch_raises(self):
        with self.assertRaises(numerical.error):
            numerical.Tensor([1, 2, 3], shape=(2, 2))


class Einsum(unittest.TestCase):
    def test_matrix_product(self):
        a = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        b = numerical.Tensor([5, 6, 7, 8], shape=(2, 2))
        self.assertEqual(numerical.einsum("ij,jk->ik", a, b).tolist(),
                         [[19.0, 22.0], [43.0, 50.0]])

    def test_transpose_and_trace(self):
        m = numerical.Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
        self.assertEqual(numerical.einsum("ij->ji", m).shape, (3, 2))
        sq = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        self.assertEqual(numerical.einsum("ii->", sq).tolist(), 5.0)

    def test_three_operands(self):
        a = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        i = numerical.Tensor([1, 0, 0, 1], shape=(2, 2))
        self.assertEqual(numerical.einsum("ij,jk,kl->il", a, i, i).tolist(),
                         [[1.0, 2.0], [3.0, 4.0]])

    def test_complex_operands(self):
        a = numerical.Tensor([1j, 1], dtype="complex128")
        out = numerical.einsum("i,i->", a, a)
        self.assertEqual(out.tolist(), 0j)  # i*i + 1*1 = -1 + 1

    def test_too_few_arguments_raises(self):
        with self.assertRaises(numerical.error):
            numerical.einsum("i->i")

    def test_mixed_dtype_raises(self):
        with self.assertRaises(numerical.error):
            numerical.einsum("i,i->", numerical.Tensor([1.0]),
                          numerical.Tensor([1j], dtype="complex128"))

    def test_bad_spec_raises(self):
        with self.assertRaises(numerical.error):
            numerical.einsum("ij", numerical.Tensor([1, 2, 3, 4], shape=(2, 2)))


class Solve(unittest.TestCase):
    def test_solves_a_linear_system(self):
        a = numerical.Tensor([2, 1, 1, 1, 3, 1, 1, 1, 4], shape=(3, 3))
        b = numerical.Tensor([7, 10, 15])
        x = numerical.solve(a, b)
        for got, want in zip(x.tolist(), [1.0, 2.0, 3.0]):
            self.assertAlmostEqual(got, want)

    def test_rank_requirements(self):
        with self.assertRaises(numerical.error):
            numerical.solve(numerical.Tensor([1, 2, 3]), numerical.Tensor([1, 2, 3]))

    def test_singular_raises(self):
        a = numerical.Tensor([1, 2, 2, 4], shape=(2, 2))
        with self.assertRaises(numerical.error):
            numerical.solve(a, numerical.Tensor([1, 2]))


class NonFinite(unittest.TestCase):
    """Every decomposition refuses an infinite or NaN entry where it starts, rather than stop at once
    and return plausible values (eigh of [[1, nan], [nan, 2]] used to read as 1 and 2)."""

    def test_every_decomposition_refuses_a_non_finite_entry(self):
        rhs = numerical.Tensor([1.0, 1.0], [2])
        calls = {
            "eigh": lambda m: numerical.eigh(m),
            "svd": lambda m: numerical.svd(m),
            "qr": lambda m: numerical.qr(m),
            "cholesky": lambda m: numerical.cholesky(m),
            "inverse": lambda m: numerical.inverse(m),
            "determinant": lambda m: numerical.determinant(m),
            "solve": lambda m: numerical.solve(m, rhs),
            "lstsq": lambda m: numerical.lstsq(m, rhs),
        }
        for bad in (float("inf"), float("-inf"), float("nan")):
            for data in ([1.0, bad, bad, 2.0], [bad, 0.0, 0.0, 1.0]):
                for name, call in calls.items():
                    with self.subTest(name=name, data=data):
                        with self.assertRaises(numerical.error) as caught:
                            call(numerical.Tensor(data, [2, 2]))
                        self.assertIn("infinite or NaN", str(caught.exception))


class ArrayInterface(unittest.TestCase):
    def test_dict_describes_the_buffer(self):
        t = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        ai = t.__array_interface__
        self.assertEqual(ai["shape"], (2, 2))
        self.assertEqual(ai["version"], 3)
        self.assertTrue(ai["typestr"].endswith("f8"))
        self.assertIsInstance(ai["data"][0], int)
        self.assertTrue(ai["data"][1])  # read-only

    def test_complex_typestr(self):
        t = numerical.Tensor([1j], dtype="complex128")
        self.assertTrue(t.__array_interface__["typestr"].endswith("c16"))

    def test_asarray_of_a_tensor_returns_it(self):
        t = numerical.Tensor([1, 2, 3])
        self.assertIs(numerical.asarray(t), t)

    def test_asarray_rejects_plain_objects(self):
        with self.assertRaises(numerical.error):
            numerical.asarray(object())

    def test_asarray_widens_real_to_complex(self):
        t = numerical.asarray(numerical.Tensor([1.0, 2.0]), dtype="complex128")
        self.assertEqual(t.dtype, "complex128")

    def test_asarray_refuses_complex_to_float64(self):
        with self.assertRaises(numerical.error):
            numerical.asarray(numerical.Tensor([1j, 2j], dtype="complex128"), dtype="float64")

    def test_asarray_matching_dtype_returns_same_object(self):
        t = numerical.Tensor([1.0, 2.0])
        self.assertIs(numerical.asarray(t, dtype="float64"), t)

    def test_a_tensor_refuses_a_second_init(self):
        # Views of the buffer (np.asarray, __dlpack__) outlive the call that made them; a second
        # __init__ would free their memory, so it is refused and the tensor keeps its value.
        t = numerical.Tensor([1.0, 2.0])
        interface = t.__array_interface__
        with self.assertRaises(TypeError):
            t.__init__([3.0])
        self.assertEqual(t.tolist(), [1.0, 2.0])
        self.assertEqual(t.__array_interface__["data"], interface["data"])

    def test_asarray_reads_either_byte_order(self):
        # A typestr names its byte order: a big-endian `>f8` source is byte-swapped on read, not
        # copied as native doubles; an unknown order is refused, not guessed.
        import ctypes
        import struct

        class Interface:
            def __init__(self, typestr, raw, count):
                self.buffer = ctypes.create_string_buffer(raw, len(raw))
                self.__array_interface__ = {"version": 3, "typestr": typestr, "shape": (count,),
                                            "data": (ctypes.addressof(self.buffer), True)}

        values = [1.5, -2.25, 1e300]
        for order in "<>":
            source = Interface(order + "f8", struct.pack(order + "3d", *values), 3)
            with self.subTest(order=order):
                self.assertEqual(numerical.asarray(source).tolist(), values)
        pairs = struct.pack(">4d", 1.0, -2.0, 0.5, 3.0)
        self.assertEqual(numerical.asarray(Interface(">c16", pairs, 2)).tolist(), [1 - 2j, 0.5 + 3j])
        with self.assertRaises(numerical.error):
            numerical.asarray(Interface("|f8", struct.pack("<d", 1.0), 1))

    def test_asarray_refuses_a_null_address_and_an_empty_data_tuple(self):
        # A null data address with elements to read crashed the interpreter; an empty array may have one.
        class Raw:
            def __init__(self, typestr, data, count):
                self.__array_interface__ = {"version": 3, "typestr": typestr, "shape": (count,), "data": data}

        for typestr in ("<f8", "<c16"):
            with self.subTest(typestr=typestr):
                with self.assertRaises(numerical.error):
                    numerical.asarray(Raw(typestr, (0, True), 4))
                self.assertEqual(numerical.asarray(Raw(typestr, (0, True), 0)).tolist(), [])
        with self.assertRaises(numerical.error):
            numerical.asarray(Raw("<f8", (), 1))

    def test_asarray_rejects_unknown_dtype(self):
        with self.assertRaises(numerical.error):
            numerical.asarray(numerical.Tensor([1.0]), dtype="float32")


class GilRelease(unittest.TestCase):
    def test_a_large_decomposition_lets_another_thread_run(self):
        # Another thread wakes 1 ms after the call starts. Holding the GIL, the call keeps it from running
        # until it returns; releasing it, the thread runs while the SVD (about 100 ms) still computes.
        import threading
        import time

        n = 200
        a = numerical.Tensor([float((i * 7 + j * 13) % 17) + (n if i == j else 0) for i in range(n) for j in range(n)],
                             shape=(n, n))
        started = threading.Event()
        woke = []

        def other():
            started.wait()
            time.sleep(0.001)
            woke.append(time.perf_counter())

        thread = threading.Thread(target=other)
        thread.start()
        begin = time.perf_counter()
        started.set()
        numerical.svd(a)
        end = time.perf_counter()
        thread.join()
        self.assertLess(woke[0] - begin, (end - begin) / 2)


class DLPack(unittest.TestCase):
    def test_capsule_and_device(self):
        t = numerical.Tensor([1, 2, 3])
        capsule = t.__dlpack__()
        self.assertIn("PyCapsule", repr(type(capsule)))
        self.assertEqual(t.__dlpack_device__(), (1, 0))  # (kDLCPU, 0)


@unittest.skipIf(np is None, "NumPy not installed")
class NumpyInterop(unittest.TestCase):
    def test_array_interface_zero_copy_view(self):
        t = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        arr = np.asarray(t)
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.dtype, np.float64)
        self.assertEqual(arr.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_dlpack_round_trip(self):
        t = numerical.Tensor([1.5, 2.5, 3.5])
        arr = np.from_dlpack(t)
        self.assertEqual(arr.tolist(), [1.5, 2.5, 3.5])

    def test_asarray_from_numpy(self):
        source = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = numerical.asarray(source)
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_asarray_widens_numpy_real_to_complex(self):
        source = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = numerical.asarray(source, dtype="complex128")
        self.assertEqual(t.dtype, "complex128")
        self.assertTrue(np.allclose(np.asarray(t), source.astype(np.complex128)))

    def test_asarray_refuses_numpy_complex_to_float64(self):
        with self.assertRaises(numerical.error):
            numerical.asarray(np.array([1 + 2j, 3 + 4j]), dtype="float64")

    def test_einsum_matches_numpy(self):
        a = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        b = numerical.Tensor([5, 6, 7, 8], shape=(2, 2))
        want = np.einsum("ij,jk->ik", np.asarray(a), np.asarray(b))
        self.assertEqual(np.asarray(numerical.einsum("ij,jk->ik", a, b)).tolist(),
                         want.tolist())


class LinalgSurface(unittest.TestCase):
    """Shape, dtype and structure of the decomposition layer (no NumPy needed)."""

    def test_determinant_value_and_dtype(self):
        a = numerical.Tensor([1, 2, 3, 4], shape=(2, 2))
        self.assertAlmostEqual(numerical.determinant(a), -2.0)  # 1*4 - 2*3
        c = numerical.Tensor([1, 0, 0, 1], shape=(2, 2), dtype="complex128")
        self.assertAlmostEqual(numerical.determinant(c), 1 + 0j)

    def test_inverse_and_cholesky_return_matrices(self):
        a = numerical.Tensor([4, 2, 2, 3], shape=(2, 2))
        self.assertEqual(numerical.inverse(a).shape, (2, 2))
        self.assertEqual(numerical.cholesky(a).shape, (2, 2))

    def test_factorisations_return_tuples(self):
        a = numerical.Tensor([1, 1, 1, 2, 1, 3], shape=(3, 2))
        q, r = numerical.qr(a)
        self.assertEqual((q.shape, r.shape), ((3, 2), (2, 2)))
        sym = numerical.Tensor([2, 1, 1, 2], shape=(2, 2))
        values, vectors = numerical.eigh(sym)
        self.assertEqual((values.shape, vectors.shape), ((2,), (2, 2)))
        u, s, vh = numerical.svd(a)
        self.assertEqual((u.shape, s.shape, vh.shape), ((3, 2), (2,), (2, 2)))
        wide = numerical.Tensor([1, 2, 3, 4, 5, 6], shape=(2, 3))
        u, s, vh = numerical.svd(wide)
        self.assertEqual((u.shape, s.shape, vh.shape), ((2, 2), (2,), (2, 3)))

    def test_eigh_rejects_complex_and_svd_rejects_a_vector(self):
        c = numerical.Tensor([1, 0, 0, 1], shape=(2, 2), dtype="complex128")
        with self.assertRaises(numerical.error):
            numerical.eigh(c)
        with self.assertRaises(numerical.error):
            numerical.svd(numerical.Tensor([1, 2, 3], shape=(3,)))

    def test_inverse_rejects_singular(self):
        with self.assertRaises(numerical.error):
            numerical.inverse(numerical.Tensor([1, 2, 2, 4], shape=(2, 2)))


@unittest.skipIf(np is None, "NumPy not installed")
class LinalgVsNumpy(unittest.TestCase):
    """Validate the decomposition layer against numpy.linalg."""

    def test_determinant_and_inverse(self):
        a = np.array([[4.0, 7.0], [2.0, 6.0]])
        t = numerical.asarray(a)
        self.assertAlmostEqual(numerical.determinant(t), np.linalg.det(a))
        self.assertTrue(np.allclose(np.asarray(numerical.inverse(t)), np.linalg.inv(a)))

    def test_lstsq(self):
        a = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b = np.array([2.0, 3.0, 5.0])
        got = np.asarray(numerical.lstsq(numerical.asarray(a), numerical.asarray(b)))
        want = np.linalg.lstsq(a, b, rcond=None)[0]
        self.assertTrue(np.allclose(got, want))

    def test_cholesky(self):
        a = np.array([[4.0, 2.0], [2.0, 3.0]])
        self.assertTrue(np.allclose(np.asarray(numerical.cholesky(numerical.asarray(a))),
                                    np.linalg.cholesky(a)))

    def test_qr_reconstructs(self):
        a = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        q, r = numerical.qr(numerical.asarray(a))
        self.assertTrue(np.allclose(np.asarray(q) @ np.asarray(r), a)) # signs may differ; reconstruct

    def test_eigh_reconstructs(self):
        a = np.array([[2.0, 1.0], [1.0, 2.0]])
        values, vectors = numerical.eigh(numerical.asarray(a))
        v = np.asarray(values)
        matrix = np.asarray(vectors)
        self.assertTrue(np.allclose(np.sort(v), np.linalg.eigvalsh(a)))
        self.assertTrue(np.allclose(matrix @ np.diag(v) @ matrix.T, a))

    def test_slogdet_matrix_rank_and_pinv_as_numpy(self):
        rng = np.random.default_rng(11)
        for a in (np.eye(200) * 1e-2, np.array([[0.0, 2.0], [1.0, 0.0]]), np.array([[1.0, 2.0], [2.0, 4.0]]),
                  rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))):
            sign, logabsdet = numerical.slogdet(numerical.asarray(a))
            want_sign, want_log = np.linalg.slogdet(a)
            self.assertAlmostEqual(sign, want_sign)
            self.assertTrue(np.isclose(logabsdet, want_log) or logabsdet == want_log == -np.inf)
        for a in (np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), rng.standard_normal((5, 3)),
                  rng.standard_normal((3, 5)), rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))):
            self.assertEqual(numerical.matrix_rank(numerical.asarray(a)), np.linalg.matrix_rank(a))
            self.assertTrue(np.allclose(np.asarray(numerical.pinv(numerical.asarray(a))), np.linalg.pinv(a)))

    def test_eigh_reads_the_lower_triangle_as_numpy(self):
        a = np.array([[2.0, 99.0, -7.0], [1.0, 3.0, 50.0], [0.5, -1.0, 4.0]])
        values, _ = numerical.eigh(numerical.asarray(a))
        self.assertTrue(np.allclose(np.asarray(values), np.linalg.eigvalsh(a)))

    def test_svd_reconstructs(self):
        a = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        u, s, vh = numerical.svd(numerical.asarray(a))
        reconstructed = np.asarray(u) @ np.diag(np.asarray(s)) @ np.asarray(vh)
        self.assertTrue(np.allclose(reconstructed, a))
        self.assertTrue(np.allclose(np.asarray(s), np.linalg.svd(a, compute_uv=False)))

    def test_svd_third_factor_is_vh_as_numpy_returns_it(self):
        # numpy's idiom (U * S) @ Vh reconstructs a, wide or tall, real or complex.
        rng = np.random.default_rng(7)
        for a in (rng.standard_normal((4, 3)), rng.standard_normal((3, 5)),
                  rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))):
            u, s, vh = (np.asarray(f) for f in numerical.svd(numerical.asarray(a)))
            self.assertTrue(np.allclose((u * s) @ vh, a))
            self.assertTrue(np.allclose(s, np.linalg.svd(a, compute_uv=False)))

    def test_svd_keeps_small_singular_values(self):
        # sigma down to 1e-12 under random rotations, against numpy (which uses LAPACK's gesdd).
        rng = np.random.default_rng(3)
        q1, _ = np.linalg.qr(rng.standard_normal((6, 4)))
        q2, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        a = q1 @ np.diag([1.0, 1e-4, 1e-8, 1e-12]) @ q2.T
        s = np.asarray(numerical.svd(numerical.asarray(a))[1])
        self.assertTrue(np.allclose(s, np.linalg.svd(a, compute_uv=False), rtol=1e-3, atol=0))

    def test_lstsq_rank_deficient_and_wide_give_numpys_minimum_norm(self):
        for a, b in ((np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), np.array([1.0, 2.0, 4.0])),
                     (np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]), np.array([3.0, 1.0]))):
            got = np.asarray(numerical.lstsq(numerical.asarray(a), numerical.asarray(b)))
            self.assertTrue(np.allclose(got, np.linalg.lstsq(a, b, rcond=None)[0]))


if __name__ == "__main__":
    unittest.main()
