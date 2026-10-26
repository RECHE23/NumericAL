"""Products of NumPy arrays read where they lie (no copy into a Tensor): the same bits as the Tensors' product, and the
converting path for any array that cannot be read so."""

import math
import unittest

import numerical

try:
    import numpy as np
except ImportError:  # pragma: no cover - the CI images install NumPy
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class BorrowedOperands(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(5)

    def operand(self, shape, dtype):
        values = self.rng.standard_normal(shape)
        if np.dtype(dtype).kind == "c":
            values = values + 1j * self.rng.standard_normal(shape)
        return values.astype(dtype)

    def assert_same(self, got, want):
        if isinstance(want, numerical.Tensor):
            self.assertIsInstance(got, numerical.Tensor)
            self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape))
        self.assertTrue(np.array_equal(np.asarray(got), np.asarray(want)))

    def test_every_dtype_and_rank_gives_the_tensors_bits(self):
        for dtype in (np.float64, np.complex128, np.float32, np.complex64):
            for m, k, n in ((1, 1, 1), (3, 4, 5), (100, 37, 1), (1, 37, 100), (129, 200, 3), (5, 0, 4)):
                for left, right in (((m, k), (k, n)), ((m, k), (k,)), ((k,), (k, n)), ((k,), (k,))):
                    with self.subTest(dtype=np.dtype(dtype).name, left=left, right=right):
                        a, b = self.operand(left, dtype), self.operand(right, dtype)
                        want = numerical.asarray(a) @ numerical.asarray(b)
                        self.assert_same(numerical.matmul(a, b), want)
                        self.assert_same(numerical.matmul(numerical.asarray(a), b), want)
                        self.assert_same(numerical.matmul(a, numerical.asarray(b)), want)
                        self.assert_same(numerical.asarray(a) @ b, want)

    def test_arrays_that_cannot_be_read_in_place_are_converted(self):
        a, b = self.operand((6, 8), np.float64), self.operand((8, 6), np.float64)
        strided = a[:, ::2]
        swapped = a.astype(">f8")
        raw = np.zeros(6 * 8 * 8 + 1, np.uint8)[1:].view(np.float64).reshape(6, 8)  # one byte off a double's alignment
        raw[...] = a
        self.assertFalse(raw.flags.aligned)
        for left, right, want in ((strided, b[::2], a[:, ::2] @ b[::2]), (swapped, b, a @ b), (raw, b, a @ b),
                                  (a.astype(np.float32), b, a.astype(np.float32).astype(np.float64) @ b)):
            with self.subTest(left=left.dtype.str, contiguous=left.flags.c_contiguous, aligned=left.flags.aligned):
                np.testing.assert_allclose(np.asarray(numerical.matmul(left, right)), want, rtol=1e-12)
        with self.assertRaises(numerical.LinAlgError):
            numerical.matmul(a, a)

    def test_strided_arrays_give_their_contiguous_copies_bits(self):
        # A small strided array is copied in C order where it is read, a large one by the converting path: either way
        # the answers of its contiguous copy, to the bit, for every reader of borrowed operands.
        for rows, cols in ((4, 31), (80, 80), (400, 400)):  # the last past the elements copied in place
            for dtype in (np.float64, np.complex128, np.float32, np.complex64):
                with self.subTest(rows=rows, cols=cols, dtype=np.dtype(dtype).name):
                    whole = self.operand((2 * rows, 2 * cols), dtype)
                    strided = whole[::2, ::-2]
                    copy = np.ascontiguousarray(strided)
                    vector = self.operand((2 * cols,), dtype)[::2]
                    self.assert_same(numerical.matmul(strided, vector), numerical.matmul(copy, np.ascontiguousarray(vector)))
                    self.assert_same(numerical.matmul(vector, strided.T), numerical.matmul(np.ascontiguousarray(vector),
                                                                                           np.ascontiguousarray(copy.T)))
                    self.assertEqual(numerical.vector_norm(strided), numerical.vector_norm(copy))
                    self.assert_same(numerical.einsum("ij->j", strided), numerical.einsum("ij->j", copy))
                    if dtype in (np.float64, np.complex128) and rows == cols:
                        spread = np.zeros((2 * rows, 2 * rows), dtype)
                        spread[::2, ::2] = copy + rows * np.eye(rows)
                        square = spread[::2, ::2]  # strided, well conditioned
                        self.assert_same(numerical.solve(square, vector[:rows]),
                                         numerical.solve(np.ascontiguousarray(square), np.ascontiguousarray(vector[:rows])))

    def test_a_whole_reduction_reads_the_array_and_gives_the_tensors_bits(self):
        # sum, mean, max and min of every element of a NumPy array, read where it lies: the Tensor's answer to the bit,
        # of its type, for each dtype, from one element to a million (the sum's threads), strided (copied in place
        # where small, through a Tensor where large); max and min of a complex array and of an empty one still refuse.
        for dtype in (np.float64, np.complex128, np.float32, np.complex64):
            for shape in ((1,), (7, 3), (1000, 1000), (3, 0)):
                x = self.operand(shape, dtype)
                for view in (x, x[::2, ::-1] if len(shape) == 2 else x):
                    tensor = numerical.asarray(np.ascontiguousarray(view))
                    for op in ("sum", "mean", "max", "min"):
                        with self.subTest(dtype=np.dtype(dtype).name, shape=view.shape, op=op):
                            f = getattr(numerical, op)
                            if op in ("max", "min") and (np.dtype(dtype).kind == "c" or view.size == 0):
                                with self.assertRaises(Exception) as from_tensor:
                                    f(tensor)
                                with self.assertRaises(type(from_tensor.exception)):
                                    f(view)
                                continue
                            got, want = f(view), f(tensor)
                            self.assertEqual(type(got), type(want))
                            self.assertTrue(got == want or (got != got and want != want))
        self.assertIs(numerical._numerical._borrowed_reduce("prod", np.ones(3)), NotImplemented)
        self.assertIs(numerical._numerical._borrowed_reduce("sum", [1.0, 2.0]), NotImplemented)

    def test_an_elementwise_function_reads_the_array_and_gives_the_tensors_bits(self):
        # exp, log, sqrt, sin, cos, tan, tanh and absolute of a NumPy array, read where it lies: the Tensor's result to
        # the byte, of its dtype and shape, for each dtype, empty to a million elements, strided
        # views small and large; complex magnitudes keep NumPy's loop through a Tensor, the same answer.
        names = ("exp", "log", "sqrt", "sin", "cos", "tan", "tanh", "absolute")
        for dtype in (np.float64, np.complex128, np.float32, np.complex64):
            for shape in ((1,), (7, 3), (1000, 1000), (3, 0)):
                x = self.operand(shape, dtype)
                for view in (x, x[::2, ::-1] if len(shape) == 2 else x):
                    tensor = numerical.asarray(np.ascontiguousarray(view))
                    for name in names:
                        with self.subTest(dtype=np.dtype(dtype).name, shape=view.shape, name=name):
                            f = getattr(numerical, name)
                            got, want = f(view), f(tensor)
                            self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape))
                            self.assertEqual(np.asarray(got).tobytes(), np.asarray(want).tobytes())
        self.assertIs(numerical._numerical._borrowed_unary("conj", np.ones(3)), NotImplemented)
        self.assertIs(numerical._numerical._borrowed_unary("absolute", np.ones(3)), NotImplemented)
        self.assertIs(numerical._numerical._borrowed_unary("abs", np.ones(3, np.complex128)), NotImplemented)
        self.assertIs(numerical._numerical._borrowed_unary("exp", [1.0]), NotImplemented)

    def test_a_transposed_matrix_times_a_vector_gives_its_copys_bits(self):
        # A column-major matrix (a transpose) times a vector, read where it lies: the product of its C-ordered copy,
        # byte for byte, in the four dtypes, about the kernels' partial sums, a complex row's slots and the blocks of
        # rows.
        for dtype in (np.float64, np.complex128, np.float32, np.complex64):
            for rows, cols in ((1, 1), (3, 17), (37, 32), (48, 275), (259, 65), (129, 7), (17, 16), (300, 47)):
                with self.subTest(dtype=np.dtype(dtype).name, rows=rows, cols=cols):
                    a = self.operand((cols, rows), dtype).T
                    x = self.operand((cols,), dtype)
                    got = numerical.matmul(a, x)
                    want = numerical.matmul(np.ascontiguousarray(a), x)
                    self.assertEqual(np.asarray(got).tobytes(), np.asarray(want).tobytes())

    def test_a_transposed_matrix_norm_gives_its_contiguous_copys_bits(self):
        # A column-major matrix (a transpose) is read where it lies, its elements in its C order: the norm of its
        # C-ordered copy to the bit, by bands of eight rows, the whole blocks of eight parts and the tail after them,
        # real and complex, single and double; values whose squares overflow or underflow leave it to the scaled norm.
        for rows, cols in ((2, 3), (3, 5), (8, 8), (7, 9), (252, 175), (31, 2)):
            for dtype in (np.float64, np.complex128, np.float32, np.complex64):
                for scale in (1.0, 1e200, 1e-200):
                    if scale != 1.0 and dtype in (np.float32, np.complex64):
                        continue  # past single precision's range
                    with self.subTest(rows=rows, cols=cols, dtype=np.dtype(dtype).name, scale=scale):
                        a = (self.operand((cols, rows), dtype) * scale).astype(dtype).T
                        self.assertTrue(a.flags.f_contiguous and not a.flags.c_contiguous)
                        copy = np.ascontiguousarray(a)
                        got, want = numerical.norm(a), numerical.norm(copy)
                        self.assertEqual(got, want)
                        read = numerical._numerical._borrowed_norm(a)  # where it lies, unless the scaled norm decides
                        self.assertEqual(read is NotImplemented, scale != 1.0)
                        self.assertEqual(type(got), type(want))

    def test_a_permutation_of_axes_reads_the_array_where_it_lies(self):
        # einsum of one operand that keeps each label in another order (or the same): the elements in the output's
        # order, as from a Tensor and as NumPy's einsum; a column-major matrix read with its labels reversed (a.T to
        # "ij->ji" a copy); empty and higher-rank operands alike.
        for shape, spec in (((40, 30), "ij->ji"), ((339, 24), "ij->ji"), ((3, 4, 5), "ijk->kij"), ((3, 4, 5), "ijk->ijk"),
                            ((0, 3), "ij->ji"), ((2, 3, 4, 5), "ijkl->ljik"), ((7, 9), "ij->ij")):
            for dtype in (np.float64, np.complex128, np.float32, np.complex64):
                a = self.operand(shape, dtype)
                for x in ((a, np.asfortranarray(a), a.T) if a.ndim == 2 else (a,)):
                    with self.subTest(shape=x.shape, spec=spec, dtype=np.dtype(dtype).name, fortran=x.flags.f_contiguous):
                        got = numerical.einsum(spec, x)
                        self.assert_same(got, numerical.einsum(spec, numerical.asarray(np.ascontiguousarray(x))))
                        self.assertTrue(np.array_equal(np.asarray(got), np.einsum(spec, x)))
                        self.assertIsNot(numerical._numerical._borrowed_einsum(spec, x), NotImplemented)

    def test_a_pair_of_arrays_reads_both_where_they_lie(self):
        # einsum of two NumPy arrays of one dtype, each label once and none an operand's own: the Tensors' result to the
        # bit, read in place (products, broadcasts, batches, a dot, empty extents); a column-major matrix is copied,
        # its reversed labels a route of other bits (ij,j->i no canonical product), the same answer; own labels,
        # repeated labels, mismatched extents and implicit specs left to the converting path.
        cases = (("i,j->ij", (30,), (40,)), ("ij,j->ij", (112, 204), (204,)), ("ij,i->ij", (112, 204), (112,)),
                 ("ij,ij->ij", (50, 60), (50, 60)), ("ij,jk->ik", (37, 41), (41, 23)), ("ij,kj->ik", (37, 41), (23, 41)),
                 ("ij,j->i", (400, 200), (200,)), ("i,i->", (1000,), (1000,)), ("bij,bjk->bik", (4, 9, 11), (4, 11, 6)),
                 ("ij,jk->ki", (13, 17), (17, 19)), ("ij,ji->", (20, 30), (30, 20)), ("ij,jk->ik", (0, 4), (4, 3)),
                 (",i->i", (), (5,)))
        for spec, a_shape, b_shape in cases:
            for dtype in (np.float64, np.complex128, np.float32, np.complex64):
                a, b = self.operand(a_shape, dtype), self.operand(b_shape, dtype)
                layouts = [(a, b, True)]
                if a.ndim == 2 and a.size > 1 << 16:
                    layouts.append((np.asfortranarray(a), b, False))
                for x, y, in_place in layouts:
                    with self.subTest(spec=spec, dtype=np.dtype(dtype).name, fortran=x.flags.f_contiguous):
                        want = numerical.einsum(spec, numerical.asarray(np.array(x, order="C")),
                                                numerical.asarray(np.array(y, order="C")))
                        self.assert_same(numerical.einsum(spec, x, y), want)
                        read = numerical._numerical._borrowed_einsum_pair(spec, x, y)
                        self.assertEqual(read is not NotImplemented, in_place)
        for spec, a_shape, b_shape in (("ijk,jl->il", (3, 4, 5), (4, 6)), ("ii,i->i", (3, 3), (3,)),
                                       ("ij,jk->ik", (3, 4), (5, 6)), ("ij,jk", (3, 4), (4, 5))):
            with self.subTest(spec=spec):
                a, b = self.operand(a_shape, np.float64), self.operand(b_shape, np.float64)
                self.assertIs(numerical._numerical._borrowed_einsum_pair(spec, a, b), NotImplemented)

    def test_a_norm_sums_fixed_blocks_of_parts(self):
        # The squares are summed by blocks of 2^14 parts, whatever the threads, each block's sum taken alone and the
        # blocks' sums added in order: 1 then 2^14 squares of 2^-60 is 1 + 2^-46 exactly, where one pass added a
        # quarter of those squares to the 1 and lost them. The same blocks in a column-major array's C order.
        values = np.zeros(2 * 16384)
        values[0] = 1.0
        values[16384:] = 2.0**-30
        want = math.sqrt(1.0 + 2.0**-46)
        for operand in (values, values.reshape(2, 16384), np.asfortranarray(values.reshape(2, 16384)),
                        np.asfortranarray(values.reshape(256, 128))):
            with self.subTest(shape=operand.shape, fortran=operand.flags.f_contiguous):
                self.assertEqual(numerical.norm(operand), want)
        self.assertEqual(numerical.norm(values.view(np.complex128)), want)  # the same parts, as complex pairs

    def test_matrices_of_rows_apart_give_their_contiguous_copies_bits(self):
        # Every other row of a larger array, each row's elements together: read where it lies at any size (its rows'
        # distance the kernels' leading dimension), beside a contiguous, a column-major or another such operand, the
        # product of the contiguous copies to the bit; a single column of them is copied first, the same.
        for m, k, n in ((169, 579, 2), (300, 300, 300), (473, 38, 1), (1, 40, 30), (3, 247, 803), (9, 4, 1)):
            for dtype in (np.float64, np.complex128, np.float32, np.complex64):
                a, b = self.operand((m, k), dtype), self.operand((k, n), dtype)
                apart_a, apart_b = np.zeros((2 * m, k), dtype), np.zeros((3 * k, n), dtype)
                apart_a[::2], apart_b[::3] = a, b
                want = np.asarray(numerical.matmul(a, b))
                for left, right in ((apart_a[::2], b), (a, apart_b[::3]), (apart_a[::2], apart_b[::3]),
                                    (apart_a[::2], np.asfortranarray(b)), (np.asfortranarray(a), apart_b[::3])):
                    with self.subTest(shape=(m, k, n), dtype=np.dtype(dtype).name, left=left.flags.c_contiguous,
                                      right=right.flags.c_contiguous):
                        self.assertTrue(np.array_equal(np.asarray(numerical.matmul(left, right)), want))
                vector = self.operand((k,), dtype)
                self.assert_same(numerical.matmul(apart_a[::2], vector), numerical.matmul(a, vector))

    def test_a_gram_product_gives_the_products_bits(self):
        # a @ a.T and a.T @ a of float64, a NumPy array read where it lies: one triangle by the product's tiles, mirrored,
        # the product of a and a contiguous copy of its transpose to the bit; small ones, and other dtypes, keep their route.
        for m, k in ((300, 50), (64, 700), (129, 257), (9, 3), (200, 8)):
            for dtype in (np.float64, np.float32, np.complex128):
                with self.subTest(m=m, k=k, dtype=np.dtype(dtype).name):
                    a = self.operand((m, k), dtype)
                    self.assertTrue(np.array_equal(np.asarray(numerical.matmul(a, a.T)),
                                                   np.asarray(numerical.matmul(a, np.ascontiguousarray(a.T)))))
                    self.assertTrue(np.array_equal(np.asarray(numerical.matmul(a.T, a)),
                                                   np.asarray(numerical.matmul(np.ascontiguousarray(a.T), a))))

    def test_a_fortran_ordered_array_converts_element_for_element(self):
        # Its columns running contiguously: by tiles from 32 rows, a row at a time below; every dtype asarray reads,
        # converted as from the same array in C order.
        for shape in ((70, 45), (33, 1), (31, 50), (1, 9)):
            base = self.rng.standard_normal(shape)
            for values in (base, base + 1j * base, base.astype(np.float32), (base * 10).astype(np.int32), base > 0):
                fortran = np.asfortranarray(values)
                self.assertTrue(np.array_equal(np.asarray(numerical.asarray(fortran)),
                                               np.asarray(numerical.asarray(np.ascontiguousarray(values)))), (shape, values.dtype))

    def test_a_large_array_converts_by_groups_of_rows_element_for_element(self):
        # Past 2^14 elements the copy runs by groups of rows (or bands of a transpose's tiles) on the threads: every
        # element in its place, C-ordered, strided, reversed, of rank 3 with rows of one or of many, column-major, and
        # for each dtype the conversion reads; the same values as NumPy's own copy.
        base = self.rng.standard_normal((300, 200))
        cube = self.rng.standard_normal((40, 30, 50))
        layouts = (base, base[::2, ::3], base[::-1], np.asfortranarray(base), cube, cube[:, ::-1, 1::2],
                   self.rng.standard_normal((70000, 1)), self.rng.standard_normal((1, 70000)), cube.transpose(2, 0, 1))
        for values in layouts:
            for converted in (values, values + 1j * values, values.astype(np.float32), (values * 10).astype(np.int16),
                              values > 0):
                with self.subTest(shape=converted.shape, dtype=converted.dtype.name, strides=converted.strides):
                    got = np.asarray(numerical.asarray(converted))
                    self.assertTrue(np.array_equal(got, np.ascontiguousarray(converted).astype(got.dtype)))
        for shape in ((0, 4), (0, 3, 4), (3, 0, 4), (5, 0), (0,)):  # empty: no group of rows, no element to write
            for dtype in (np.float64, np.complex128, np.float32, np.complex64, np.int16):
                with self.subTest(shape=shape, dtype=np.dtype(dtype).name):
                    self.assertEqual(np.asarray(numerical.asarray(np.zeros(shape, dtype))).shape, shape)

    def test_a_product_of_fortran_ordered_matrices_gives_the_c_ordered_bits(self):
        # Matrices of each dtype read where they lie, either or both column-major, packed (or copied into their
        # parts) from there, and a vector or a single row times such a matrix a column at a time: the product of their
        # C-ordered copies to the bit; a single column takes the converting path, the same.
        for m, k, n in ((7, 5, 3), (64, 64, 64), (129, 40, 75), (2, 300, 2), (3, 803, 247), (1, 9, 4), (9, 4, 1)):
            real_a = self.rng.standard_normal((m, k))
            real_b = self.rng.standard_normal((k, n))
            for dtype in (np.float64, np.complex128, np.float32, np.complex64):
                a = (real_a + 1j * real_a[::-1] if np.iscomplexobj(dtype(0)) else real_a).astype(dtype)
                b = (real_b - 1j * real_b[::-1] if np.iscomplexobj(dtype(0)) else real_b).astype(dtype)
                want = np.asarray(numerical.matmul(np.ascontiguousarray(a), np.ascontiguousarray(b)))
                for left in (a, np.asfortranarray(a)):
                    for right in (b, np.asfortranarray(b)):
                        self.assertTrue(np.array_equal(np.asarray(numerical.matmul(left, right)), want),
                                        (m, k, n, np.dtype(dtype).name))
                vector = np.ascontiguousarray(a[0])  # a vector times the matrix: the bits of its C-ordered copy
                self.assertTrue(np.array_equal(np.asarray(numerical.matmul(vector, np.asfortranarray(b))),
                                               np.asarray(numerical.matmul(vector, np.ascontiguousarray(b)))),
                                (k, n, np.dtype(dtype).name))

    def test_a_vector_times_a_narrow_matrix_gives_the_bits_of_its_rows_added_in_order(self):
        # A vector times a matrix of a few columns sums each column's products in registers; a matrix of forty adds its
        # rows in order. Each element of the product depends on its column alone: the narrow product is the leading
        # columns of the wide one to the bit, on the wheels' flags, for every width under the narrow route's.
        for k in (1, 7, 300):
            real_a = self.rng.standard_normal(k)
            real_b = self.rng.standard_normal((k, 40))
            for dtype in (np.float64, np.complex128, np.float32, np.complex64):
                a = (real_a + 1j * real_a[::-1] if np.iscomplexobj(dtype(0)) else real_a).astype(dtype)
                b = (real_b - 1j * real_b[::-1] if np.iscomplexobj(dtype(0)) else real_b).astype(dtype)
                wide = np.asarray(numerical.matmul(a, b))
                for n in range(2, 17):
                    narrow = np.ascontiguousarray(b[:, :n])
                    self.assertTrue(np.array_equal(np.asarray(numerical.matmul(a, narrow)), wide[:n]),
                                    (k, n, np.dtype(dtype).name))
                    self.assertTrue(np.array_equal(np.asarray(numerical.matmul(a, np.asfortranarray(narrow))), wide[:n]),
                                    (k, n, np.dtype(dtype).name))

    def test_an_einsum_of_one_operand_summing_a_run_reads_it_where_it_lies(self):
        # A contiguous array summed over a run of its axes: read in place, the bits of the same einsum of its Tensor;
        # another spec, a strided array or a wrong one take the converting path, the same answers and refusals.
        base = self.rng.standard_normal((6, 5, 4))
        for values in (base, base + 1j * base, base.astype(np.float32), (base + 2j).astype(np.complex64)):
            as_tensor = numerical.asarray(values)
            for spec in ("ijk->ik", "ijk->i", "ijk->", "ijk->ij", "ijk->jk", "ijk->ki", "ijk->j", "i j k -> i k"):
                got = numerical.einsum(spec, values)
                want = numerical.einsum(spec, as_tensor)
                self.assertEqual(got.shape, want.shape, spec)
                self.assertTrue(np.array_equal(np.asarray(got), np.asarray(want)), spec)
        strided = base[:, ::2]
        self.assertTrue(np.array_equal(np.asarray(numerical.einsum("ijk->i", strided)),
                                       np.asarray(numerical.einsum("ijk->i", numerical.asarray(np.ascontiguousarray(strided))))))
        for bad in ("ij->i", "ijk->l", "iik->i", "ijk->ii"):
            with self.assertRaises(ValueError):
                numerical.einsum(bad, base)

    def test_an_einsum_of_a_fortran_ordered_matrix_over_one_axis_gives_the_c_ordered_bits(self):
        # A column-major matrix summed over one axis is read where it lies, its transpose row-major: each sum the
        # pairwise sum of the C-ordered copy's, to the bit; a whole sum takes the C-ordered path, the same.
        for rows, cols in ((5, 9), (130, 70), (564, 774)):
            for dtype in (np.float64, np.complex128, np.float32, np.complex64):
                with self.subTest(rows=rows, cols=cols, dtype=np.dtype(dtype).name):
                    copy = self.operand((rows, cols), dtype)
                    fortran = np.asfortranarray(copy)
                    for spec in ("ij->i", "ij->j", "ij->", "ji->j", "ji->i"):
                        self.assert_same(numerical.einsum(spec, fortran), numerical.einsum(spec, copy))

    def test_the_product_owns_its_elements(self):
        a, b = self.operand((4, 4), np.float64), self.operand((4, 4), np.float64)
        product = numerical.matmul(a, b)
        before = np.asarray(product).copy()
        a[...] = 0.0
        b[...] = 0.0
        self.assertTrue(np.array_equal(np.asarray(product), before))

    def same(self, got, want):
        if isinstance(want, tuple):
            self.assertEqual(len(got), len(want))
            for g, w in zip(got, want):
                self.same(g, w)
        elif isinstance(want, numerical.Tensor):
            self.assert_same(got, want)
        else:
            self.assertIs(type(got), type(want))
            self.assertEqual(got, want)

    def test_solve_det_slogdet_and_inverse_give_the_tensors_answers(self):
        for dtype in (np.float64, np.complex128, np.float32, np.complex64):
            for n in (1, 2, 7, 64):
                a = self.operand((n, n), dtype)
                with self.subTest(dtype=np.dtype(dtype).name, n=n):
                    tensor = numerical.asarray(a)
                    for f in (numerical.det, numerical.slogdet, numerical.inv):
                        self.same(f(a), f(tensor))
                    for rhs in ((n,), (n, 3)):
                        b = self.operand(rhs, dtype)
                        self.same(numerical.solve(a, b), numerical.solve(tensor, numerical.asarray(b)))
                        self.same(numerical.solve(a, numerical.asarray(b)), numerical.solve(tensor, numerical.asarray(b)))

    def test_cholesky_gives_the_tensors_factor_either_triangle(self):
        for dtype in (np.float64, np.complex128, np.float32, np.complex64):
            for n in (1, 2, 7, 64):
                m = self.operand((n, n), dtype)
                spd = (m @ m.conj().T + n * np.eye(n)).astype(dtype)
                with self.subTest(dtype=np.dtype(dtype).name, n=n):
                    for upper in (False, True):
                        self.same(numerical.cholesky(spd, upper=upper), numerical.cholesky(numerical.asarray(spd), upper=upper))
        indefinite = -np.eye(3)
        def failure(x):
            try:
                numerical.cholesky(x)
            except numerical.LinAlgError as caught:
                return str(caught)
            return None
        self.assertIsNotNone(failure(numerical.asarray(indefinite)))
        self.assertEqual(failure(indefinite), failure(numerical.asarray(indefinite)))
        spd = np.eye(6) * 4
        np.testing.assert_allclose(np.asarray(numerical.cholesky(spd[::2, ::2])), np.eye(3) * 2)

    def test_decompositions_of_unreadable_or_wrong_operands_answer_as_the_tensors(self):
        def failure(f, *args):
            try:
                f(*args)
            except numerical.LinAlgError as caught:
                return str(caught)
            return None

        a = self.operand((5, 5), np.float64)
        singular, holed = np.ones((4, 4)), np.eye(3)
        holed[1, 1] = np.nan
        for f, args in ((numerical.inv, (singular,)), (numerical.det, (holed,)), (numerical.slogdet, (holed,)),
                        (numerical.inv, (np.ones((3, 4)),)), (numerical.solve, (a, np.ones(4))),
                        (numerical.solve, (singular, np.ones(4)))):
            with self.subTest(function=f.__name__, shape=args[0].shape):
                want = failure(f, *(numerical.asarray(x) for x in args))
                self.assertIsNotNone(want)
                self.assertEqual(failure(f, *args), want)
        strided = np.eye(6)[::2, ::2] * 2
        np.testing.assert_allclose(np.asarray(numerical.inv(strided)), np.eye(3) / 2)
        self.assertEqual(numerical.inv(np.stack([np.eye(3)] * 2)).shape, (2, 3, 3))
        stacked = self.operand((2, 5, 3), np.float64)  # a stack of right-hand sides, broadcast against one matrix
        self.same(numerical.solve(a, stacked), numerical.solve(numerical.asarray(a), numerical.asarray(stacked)))


    def test_a_transform_reads_the_array_where_it_lies_and_gives_the_tensors_bits(self):
        import numerical.fft as nfft

        for dtype in (np.float64, np.complex128, np.float32, np.complex64):
            a = self.operand((6, 20), dtype)
            for view in (a, a[:, ::3], np.asfortranarray(a)):
                tensor = numerical.asarray(np.ascontiguousarray(view))
                functions = [nfft.fft, nfft.ifft, nfft.irfft if np.dtype(dtype).kind == "c" else nfft.rfft]
                for f in functions:
                    for axis, n, norm in ((-1, None, None), (0, 5, "ortho"), (1, 31, "forward")):
                        with self.subTest(dtype=np.dtype(dtype).name, strides=view.strides, function=f.__name__,
                                          axis=axis, n=n, norm=norm):
                            self.assert_same(f(view, n=n, axis=axis, norm=norm), f(tensor, n=n, axis=axis, norm=norm))
        # Not read in place: integers (transformed as float64), a complex rfft (refused), a real irfft (widened).
        self.assert_same(nfft.fft(np.arange(8)), nfft.fft(numerical.asarray(np.arange(8.0))))
        with self.assertRaises(TypeError):
            nfft.rfft(np.ones(4, np.complex128))
        self.assert_same(nfft.irfft(np.ones(5)), nfft.irfft(numerical.asarray(np.ones(5, np.complex128))))
        with self.assertRaises(ValueError):
            nfft.fft(np.ones(4), n=0)
        with self.assertRaises(IndexError):
            nfft.fft(np.ones(4), axis=1)

if __name__ == "__main__":
    unittest.main()
