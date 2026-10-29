// NumericAL — Python extension module (CPython Limited API, abi3): the Tensor type and the registration; the
// functions by family in linalg.cpp, eigen.cpp, fft.cpp, elementwise.cpp and products.cpp.

#include "binding.hpp"

using namespace numerical_binding; // NOLINT(google-build-using-namespace): the registration names every function

SCIFORGE_MODULE(_numerical, "numerical.error", m)
{
    // The Tensor type: a def_init constructor, the DLPack/list methods, and the
    // introspection properties (shape/__array_interface__ are computed PyObject* getters).
    // init_once: np.asarray, __array_interface__ and __dlpack__ view the held buffer in place, so a
    // second __init__ would free what those views read.
    m.type<any_tensor>("numerical.Tensor")
    .init_once()
    .def_init<&tensor_factory>(sb::arg("data"), sb::arg("shape") = sb::none, sb::arg("dtype") = "float64")
    .def<&tensor_dlpack_device>("__dlpack_device__",
                                "__dlpack_device__() -> (device_type, device_id)\nThe DLPack device (always CPU).")
    .raw("__dlpack__", reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(tensor_dlpack)),
         METH_VARARGS | METH_KEYWORDS,
         "__dlpack__(stream=None) -> PyCapsule\nExport a DLPack capsule sharing this tensor's buffer.")
    .raw("_matmul", tensor_matmul, METH_O,
         "_matmul(other) -> Tensor or number\nself @ other (Tensor.__matmul__).")
    .raw("tolist", tensor_tolist, METH_NOARGS,
         "tolist() -> list\nThe elements as nested Python lists (a bare scalar when rank-0).")
    .def_prop_ro<&get_shape>("shape", "The size of each axis, as a tuple.")
    .def_prop_ro<&get_ndim>("ndim", "The number of axes.")
    .def_prop_ro<&get_size>("size", "The total number of elements.")
    .def_prop_ro<&get_dtype>("dtype", "'float64' or 'complex128'.")
    .def_prop_ro<&get_array_interface>("__array_interface__",
                                       "NumPy array-interface (v3) describing the shared, read-only buffer.");
    // The 7 linear-algebra functions whose args/return are caster-able -> m.def.
    m.def<&set_num_threads>("_set_num_threads",
                            "_set_num_threads(n)\nCap the threads a kernel uses at n (see numerical.set_num_threads).");
    m.def<&set_numpy_loops>("_set_numpy_loops",
                            "_set_numpy_loops(products, scalar_products, quotients)\nWhich of NumPy's complex loops fuse (see numerical._numpy_loops).");
    m.def<&get_num_threads>("get_num_threads", "get_num_threads() -> int\nThe threads a kernel may use.");
    m.def<&limit_gemm_path_code>("_limit_gemm_path", "_limit_gemm_path(code) -> int\nThe path code now in use (see numerical.limit_gemm_path).");
    m.def<&gemm_path_code>("_gemm_path", "_gemm_path() -> int\nThe path code float64 matrix products take.");
    m.def<&amx_probed>("_amx_probed", "_amx_probed() -> bool\nWhether the AMX's self-test has run in this process.");
    m.def<&solve>("solve", "solve(a, b) -> Tensor\nSolve the linear system a @ x = b (a rank-2; b rank-1, or rank-2 with one right-hand side per column).");
    m.def<&inverse>("inverse", "inverse(a) -> Tensor\nThe inverse of a square matrix.");
    m.def<&lstsq>("lstsq", "lstsq(a, b, rtol) -> (x, residuals, rank, s)\nThe minimum-norm least-squares solution of a @ x ≈ b with its residual sums of squares, the rank and the singular values of a, as numpy.linalg.lstsq (SVD; a rank-2; b rank-1, or rank-2 with one right-hand side per column).");
    m.def<&solve_triangular>("solve_triangular", "solve_triangular(a, b, lower, trans, unit) -> Tensor\nop(A) X = B for the lower or upper triangle of a, op by trans (0 A, 1 Aᵀ, 2 Aᴴ), its diagonal ones where unit.");
    m.def<&lu_factor>("lu_factor", "lu_factor(a) -> (Tensor, list)\nL and U in one matrix, and LAPACK's row interchanges.");
    m.def<&lu_solve>("lu_solve", "lu_solve(lu, piv, b, trans) -> Tensor\nThe solution of A x = b (trans 0), Aᵀ x = b (1) or Aᴴ x = b (2) from lu_factor's factors.");
    m.def<&cholesky>("cholesky", "cholesky(a, upper) -> Tensor\nThe lower factor L of A = L Lᴴ (Hermitian positive-definite), or with upper the factor U of A = Uᴴ U from its upper triangle.");
    m.def<&qr>("qr", "qr(a, mode) -> (Q, R)\nThe QR factorisation of an m×n matrix, as numpy.linalg.qr: reduced, Q is m×k, R k×n, k = min(m, n); complete, Q is m×m, R m×n.");
    m.def<&qr_r>("qr_r", "qr_r(a) -> R\nqr's R alone, k×n, k = min(m, n), to the bit, Q never formed (numpy.linalg.qr's mode=\"r\").");
    m.def<&schur>("schur", "schur(a, complex_output) -> (T, Z)\nThe Schur decomposition A = Z T Zᴴ, as scipy.linalg.schur: T quasi-triangular for a real matrix (triangular with complex_output, or for a complex one), Z orthogonal or unitary.");
    m.def<&schur_sorted>("schur_sorted", "schur_sorted(a, complex_output, kind, callback) -> (T, Z, sdim)\nschur with the eigenvalues scipy.linalg.schur's sort selects moved to the leading block of T (kind 0 lhp, 1 rhp, 2 iuc, 3 ouc, 4 the callable), as LAPACK's gees; sdim their count.");
    m.def<&eigh_pencil>("eigh_pencil", "eigh_pencil(a, b, method, upper) -> (values, vectors)\nEigenpairs of the pencil A x = λ B x, A symmetric or Hermitian, B positive definite (ascending values, B-orthonormal vectors), as scipy.linalg.eigh(a, b); each lower triangle read, or with upper each upper one.");
    m.def<&eigvalsh_pencil>("eigvalsh_pencil", "eigvalsh_pencil(a, b, upper) -> Tensor\nThe eigenvalues alone of the pencil A x = λ B x, ascending.");
    m.def<&eigh>("eigh", "eigh(a, method, upper) -> (values, vectors)\nEigenpairs of a real symmetric or complex Hermitian matrix (real values, ascending); its lower triangle is read, or with upper its upper one, as numpy.linalg.eigh's UPLO.");
    m.def<&matmul_nd>("_matmul_nd", "_matmul_nd(a, b) -> Tensor\nThe matrix product of two stacks of one dtype, their leading axes broadcast, as numpy.matmul.");
    m.def<&eig>("_eig", "_eig(a) -> (w, v)\nThe eigenvalues and right eigenvectors of a general square matrix (or stack), as numpy.linalg.eig: complex, as NumPy 2.5's.");
    m.def<&eigvals>("_eigvals", "_eigvals(a) -> w\nThe eigenvalues alone of a general square matrix (or stack), as numpy.linalg.eigvals: complex.");
    m.def<&fft_axis>("_fft", "_fft(a, n, axis, norm, inverse) -> Tensor\nnumpy.fft.fft (or ifft) of n points along axis (from 0), norm 0 backward, 1 ortho, 2 forward (see numerical.fft).");
    m.raw("_fftshift", module_fftshift, METH_VARARGS,
          "_fftshift(a, axes, inverse) -> Tensor\nnumpy.fft.fftshift (or ifftshift) along each of axes (from 0), in one copy (see numerical.fft).");
    m.def<&rfft_axis>("_rfft", "_rfft(a, n, axis, norm, hermitian) -> Tensor\nnumpy.fft.rfft (or ihfft) of a real tensor along axis (from 0) (see numerical.fft).");
    m.def<&trig_axis>("_trig", "_trig(a, sine, type, n, axis, norm, orthogonalize, inverse) -> Tensor\nscipy.fft.dct, idct, dst or idst along axis (from 0), orthogonalize -1 for its default (see numerical.fft).");
    m.def<&irfft_axis>("_irfft", "_irfft(a, n, axis, norm, hermitian) -> Tensor\nnumpy.fft.irfft (or hfft) of a complex tensor along axis (from 0) (see numerical.fft).");
    m.def<&eigvalsh>("eigvalsh", "eigvalsh(a, upper) -> Tensor\nThe eigenvalues alone of a real symmetric or complex Hermitian matrix, ascending, as numpy.linalg.eigvalsh (its lower triangle is read, or with upper its upper one).");
    m.def<&matrix_ranks>("_matrix_ranks", "_matrix_ranks(a, rtol, tol, hermitian) -> Tensor\nThe rank of every matrix of a stack, as float64 numbers along its leading axes.");
    m.def<&matrix_rank>("matrix_rank", "matrix_rank(a, rtol, tol, hermitian) -> int\nThe numerical rank of a matrix, as numpy.linalg.matrix_rank: its singular values above tol, else above rtol times the largest (None, None: NumPy's default).");
    m.def<&pinv>("pinv", "pinv(a, rtol, hermitian) -> Tensor\nThe Moore–Penrose pseudo-inverse of an m×n matrix, as numpy.linalg.pinv (rtol None: NumPy's default).");
    m.def<&svd>("svd", "svd(a, method, full) -> (U, S, Vh)\nThe singular value decomposition of an m×n matrix, as numpy.linalg.svd(a, full_matrices=full).");
    m.def<&svd_hermitian_of>("_svd_hermitian", "_svd_hermitian(a) -> (U, S, Vh)\nThe SVD of a real symmetric or complex Hermitian matrix (or stack) from its eigendecomposition, as numpy.linalg.svd(a, hermitian=True).");
    m.def<&singular_values>("_singular_values", "_singular_values(a) -> Tensor\nThe singular values alone, descending, as numpy.linalg.svd(a, compute_uv=False).");
    // The 3 that stay manual: determinant (dtype-dependent scalar), einsum (variadic),
    // asarray (returns the same object when the dtype matches — Py_NewRef identity).
    m.raw("determinant", module_determinant, METH_VARARGS,
          "determinant(a) -> number\nThe determinant of a square matrix (float or complex).");
    m.raw("_take", module_take, METH_VARARGS,
          "_take(t, starts, steps, counts, keep) -> Tensor or number\nThe strided copy behind Tensor.__getitem__.");
    m.raw("_power_shortcut", module_power_shortcut, METH_VARARGS,
          "_power_shortcut(kind, a) -> Tensor\nA complex tensor squared (\"2\"), inverted (\"r\") or rooted (\"h\") as NumPy's ** 2, ** -1, ** 0.5.");
    m.raw("_binary", module_binary, METH_VARARGS,
          "_binary(op, a, b) -> Tensor\nElement-wise a op b (op one of + - * /), broadcast; one dtype.");
    m.raw("_stack_last", module_stack_last, METH_VARARGS,
          "_stack_last(parts) -> Tensor\nTensors of one shape and dtype side by side along a new last axis (a copy).");
    m.raw("_negate", module_negate, METH_VARARGS, "_negate(a) -> Tensor\nElement-wise negation.");
    m.raw("_equal", module_equal, METH_VARARGS,
          "_equal(a, b) -> bool\nWhether two tensors of one dtype and size hold equal elements (a NaN equals nothing).");
    m.raw("_unary", module_unary, METH_VARARGS,
          "_unary(name, a) -> Tensor\nexp, log, sqrt, sin, cos, tan, tanh or abs of every element.");
    m.raw("_borrowed_einsum_pair", module_borrowed_einsum_pair, METH_VARARGS,
          "_borrowed_einsum_pair(spec, a, b) -> Tensor or NotImplemented\nTwo operands of one dtype, each label once and none its own, read where they lie.");
    m.raw("_borrowed_fft", module_borrowed_fft, METH_VARARGS,
          "_borrowed_fft(kind, a, n, axis, norm, inverse) -> Tensor or NotImplemented\nfft (kind 0), rfft (1) or irfft (2) of an operand read where it lies.");
    m.raw("_borrowed_trig", module_borrowed_trig, METH_VARARGS,
          "_borrowed_trig(a, sine, type, n, axis, norm, orthogonalize, inverse) -> Tensor or NotImplemented\ndct, idct, dst or "
          "idst of real elements read where they lie.");
    m.raw("_borrowed_einsum", module_borrowed_einsum, METH_VARARGS,
          "_borrowed_einsum(spec, x) -> Tensor or NotImplemented\nOne operand summed over a run of its axes, read where it lies.");
    m.raw("_borrowed_unary", module_borrowed_unary, METH_VARARGS,
          "_borrowed_unary(name, x) -> Tensor or NotImplemented\nAn element-wise function of every element of a NumPy array read where it lies.");
    m.raw("_borrowed_reduce", module_borrowed_reduce, METH_VARARGS,
          "_borrowed_reduce(op, x) -> number or NotImplemented\nsum, mean, max or min of every element of a NumPy array read where it lies.");
    m.raw("_borrowed_norm", module_borrowed_norm, METH_O,
          "_borrowed_norm(x) -> float or NotImplemented\nThe 2-norm of every element of a Tensor or a NumPy array read where it lies.");
    m.raw("_vector_norm", module_vector_norm, METH_VARARGS,
          "_vector_norm(x, ord, axes, keepdims) -> Tensor or number\nThe ord norm over the given axes, as numpy.linalg.vector_norm, without overflow.");
    m.raw("_reduce", module_reduce, METH_VARARGS,
          "_reduce(op, a, axis) -> number or Tensor\nsum, mean, max or min of every element (axis -1) or along an axis.");
    m.raw("_reshape", module_reshape, METH_VARARGS, "_reshape(a, shape) -> Tensor\nThe elements under another shape.");
    m.raw("_full", module_full, METH_VARARGS, "_full(shape, value, dtype) -> Tensor\nA tensor of one value.");
    m.raw("_eye", module_eye, METH_VARARGS, "_eye(n, dtype) -> Tensor\nThe n×n identity.");
    m.raw("_arange", module_arange, METH_VARARGS, "_arange(start, stop, step) -> Tensor\nAs numpy.arange (float64).");
    m.raw("_linspace", module_linspace, METH_VARARGS,
          "_linspace(start, stop, num) -> Tensor\nAs numpy.linspace with the end point (float64).");
    m.raw("slogdet", module_slogdet, METH_VARARGS,
          "slogdet(a) -> (sign, logabsdet)\nThe sign and the natural log of |det(a)|, as numpy.linalg.slogdet; it\n"
          "does not underflow or overflow where determinant(a) does.");
    m.raw("_borrowed_linalg", module_borrowed_linalg, METH_VARARGS,
          "_borrowed_linalg(name, a, b=None) -> the answer or NotImplemented\ncholesky, solve, determinant, slogdet or "
          "inverse of a NumPy array read where it lies.");
    m.raw("_borrowed_solve_triangular", module_borrowed_solve_triangular, METH_VARARGS,
          "_borrowed_solve_triangular(a, b, lower, trans, unit) -> Tensor or NotImplemented\nsolve_triangular of a "
          "NumPy array read where it lies.");
    m.raw("_borrowed_factor_solve", module_borrowed_factor_solve, METH_VARARGS,
          "_borrowed_factor_solve(factor, piv, b, lower, trans) -> Tensor or NotImplemented\nlu_solve (piv a sequence) "
          "or cho_solve (piv None) of operands read where they lie.");
    m.raw("_borrowed_product", module_borrowed_product, METH_VARARGS,
          "_borrowed_product(a, b) -> Tensor, number or NotImplemented\nThe product of two operands read where they lie.");
    m.raw("_set_matmul_fallback", module_set_matmul_fallback, METH_O,
          "_set_matmul_fallback(f)\nThe Python product Tensor._matmul hands every case it does not take.");
    m.raw("einsum", module_einsum, METH_VARARGS,
          "einsum(spec, *operands) -> Tensor\n"
          "Einstein summation, e.g. 'ij,jk->ik' (matmul), 'i,i->' (dot), 'ij->ji'.");
    m.raw("_asarray_nested", reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(module_asarray_nested)),
          METH_VARARGS | METH_KEYWORDS,
          "_asarray_nested(obj, dtype=None) -> Tensor | None\n"
          "A nested list or tuple of floats, ints and complex numbers as a Tensor, or None for anything else.");
    m.raw("asarray", reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(module_asarray)),
          METH_VARARGS | METH_KEYWORDS,
          "asarray(obj, dtype=None) -> Tensor\n"
          "A Tensor (converted if dtype differs), or a copy of a C-contiguous array via\n"
          "__array_interface__. dtype is honoured: real->complex128 widens (zero imaginary\n"
          "part); complex128->float64 is refused.");
}
