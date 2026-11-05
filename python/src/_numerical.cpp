// NumericAL — Python extension module (CPython Limited API, abi3): the Tensor type and the registration; the
// functions by family in linalg.cpp, eigen.cpp, fft.cpp, elementwise.cpp and products.cpp.

#include "binding.hpp"

// A free-threaded CPython re-enables the GIL at import, with a warning alone, for a module that does not declare it
// unused: SciForge before 2026.10.7 (no SCIFORGE_BINDING_NO_GIL) never declares it.
#if defined(Py_GIL_DISABLED) && !defined(SCIFORGE_BINDING_NO_GIL)
#error "a free-threaded build needs SciForge 2026.10.7 or later (sciforge-build>=2026.10.7)"
#endif

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
    .raw("_getitem", tensor_getitem, METH_O,
         "_getitem(key) -> Tensor or number\nself[key] (Tensor.__getitem__): integers and slices here, the rest by the fallback.")
    .raw("tolist", tensor_tolist, METH_NOARGS,
         "tolist() -> list\nThe elements as nested Python lists (a bare scalar when rank-0).")
    .def_prop_ro<&get_shape>("shape", "The size of each axis, as a tuple.")
    .def_prop_ro<&get_ndim>("ndim", "The number of axes.")
    .def_prop_ro<&get_size>("size", "The total number of elements.")
    .def_prop_ro<&get_dtype>("dtype", "'float64', 'complex128', 'float32', 'complex64' or 'bool'.")
    .def_prop_ro<&get_array_interface>("__array_interface__",
                                       "NumPy array-interface (v3) describing the shared, read-only buffer.");
    // The 7 linear-algebra functions whose args/return are caster-able -> m.def.
    m.def<&release_memory>("release_memory",
                           "release_memory()\nFree the workspaces NumericAL keeps between calls: the calling thread's and the thread pool's,\nthen the blocks kept for reuse (as oneMKL's mkl_free_buffers). Another thread's stay, its own to free.");
    m.def<&set_num_threads>("_set_num_threads",
                            "_set_num_threads(n)\nCap the threads a kernel uses at n (see numerical.set_num_threads).");
    m.def<&set_numpy_loops>("_set_numpy_loops",
                            "_set_numpy_loops(products, scalar_products, quotients, reductions, late_reductions)\nWhich of NumPy's complex loops fuse (see numerical._numpy_loops).");
    m.def<&get_num_threads>("get_num_threads", "get_num_threads() -> int\nThe threads a kernel may use.");
    m.def<&limit_gemm_path_code>("_limit_gemm_path", "_limit_gemm_path(code) -> int\nThe path code now in use (see numerical.limit_gemm_path).");
    m.def<&gemm_path_code>("_gemm_path", "_gemm_path() -> int\nThe path code float64 matrix products take.");
    m.def<&amx_probed>("_amx_probed", "_amx_probed() -> bool\nWhether the AMX's self-test has run in this process.");
    m.def<&solve>("solve", "solve(a, b) -> Tensor\nSolve the linear system a @ x = b (a rank-2; b rank-1, or rank-2 with one right-hand side per column).");
    m.def<&solve_banded>("solve_banded", "solve_banded(ab, b, lower, upper) -> Tensor\nThe solution of a banded system, as scipy.linalg.solve_banded (LAPACK's gbsv; gtsv for three diagonals).");
    m.def<&solve_conditioned>("solve_conditioned", "solve_conditioned(a, b) -> (Tensor, float)\nThe solution of a @ x = b by the LU factorisation and its reciprocal condition number, estimated (LAPACK's gesv and gecon), as scipy.linalg.solve(assume_a='gen') takes them.");
    m.def<&factor_condition>("factor_condition", "factor_condition(factor, a, kind, lower) -> float\nThe reciprocal condition number of a, estimated: kind 0 from its Cholesky factor (pocon), kind 1 of its triangle (trcon).");
    m.def<&solve_banded_conditioned>("solve_banded_conditioned", "solve_banded_conditioned(ab, b, lower, upper) -> (Tensor, float)\nsolve_banded by the band LU and its reciprocal condition number, estimated (LAPACK's gbsv and gbcon).");
    m.def<&solveh_banded>("solveh_banded", "solveh_banded(ab, b, lower) -> Tensor\nThe solution of a Hermitian positive-definite banded system, as scipy.linalg.solveh_banded (LAPACK's pbsv; ptsv for three diagonals).");
    m.def<&cholesky_banded>("cholesky_banded", "cholesky_banded(ab, lower) -> Tensor\nThe Cholesky factor of a Hermitian positive-definite band in its own form, as scipy.linalg.cholesky_banded (LAPACK's pbtrf).");
    m.def<&cho_solve_banded>("cho_solve_banded", "cho_solve_banded(cb, b, lower) -> Tensor\nThe solution of A x = b from cholesky_banded's factor, as scipy.linalg.cho_solve_banded (LAPACK's pbtrs).");
    m.def<&hessenberg>("hessenberg", "hessenberg(a, calc_q) -> (H, Q)\nThe Hessenberg form A = Q H Qᴴ, as scipy.linalg.hessenberg (LAPACK's gehrd); Q 0 × 0 without calc_q.");
    m.def<&matrix_balance>("matrix_balance", "matrix_balance(a, permute, scale) -> (B, scaling, permutation)\nA balanced, B = T⁻¹ A T, T = P D, as scipy.linalg.matrix_balance(separate=True) (LAPACK's gebal).");
    m.def<&band_tridiagonal>("band_tridiagonal", "band_tridiagonal(lower, basis) -> (d, e, w)\nA symmetric or Hermitian band (lower form) reduced to real tridiagonal form, as LAPACK's dsbtrd/zhbtrd; w 0 × 0 without basis.");
    m.def<&tridiagonal_eigen>("tridiagonal_eigen", "tridiagonal_eigen(d, e, vectors) -> (w, v)\nThe eigenvalues (ascending) and eigenvectors of a real symmetric tridiagonal, as scipy.linalg.eigh_tridiagonal; v 0 × 0 without vectors.");
    m.def<&tridiagonal_select>("tridiagonal_select", "tridiagonal_select(d, e, by_value, a, b) -> Tensor\nThe eigenvalues of indices a to b (or in (a, b] where by_value) of a real symmetric tridiagonal, ascending, as scipy.linalg.eigvalsh_tridiagonal with a selection.");
    m.def<&qz>("qz", "qz(a, b, complex_output, select) -> (S, T, alpha, beta, Q, Z)\nThe generalized Schur form A = Q S Zᴴ, B = Q T Zᴴ, as scipy.linalg.qz (LAPACK's xgges), reordered by select(alpha, beta) as scipy.linalg.ordqz where it is a callable.");
    m.def<&rsf2csf>("rsf2csf", "rsf2csf(T, Z) -> (T, Z)\nA real Schur form made complex, as scipy.linalg.rsf2csf.");
    m.def<&solve_toeplitz>("solve_toeplitz", "solve_toeplitz(c, r, b) -> Tensor\nThe solution of a Toeplitz system by Levinson's recursion, as scipy.linalg.solve_toeplitz.");
    m.def<&solve_circulant>("solve_circulant", "solve_circulant(c, b, least_squares, tol) -> Tensor\nThe solution of a circulant system by the FFT, as scipy.linalg.solve_circulant.");
    m.def<&structured>("_structured", "_structured(c, r, kind) -> Tensor\nToeplitz (0), Hankel (1) or circulant (2) matrix of vectors of one dtype, as scipy.linalg's.");
    m.raw("_block_diag", module_block_diag, METH_VARARGS, "_block_diag(blocks) -> Tensor\nThe block diagonal matrix of matrices of one dtype, as scipy.linalg.block_diag.");
    m.def<&qr_pivoted>("qr_pivoted", "qr_pivoted(a, complete) -> (Q, R, P)\nQR with column pivoting, A[:, P] = Q R, as scipy.linalg.qr(a, pivoting=True).");
    m.def<&ldl>("ldl", "ldl(a, lower, hermitian) -> (lu, d, perm, ignored)\nThe symmetric indefinite factorisation A = L D Lᴴ (Lᵀ), as scipy.linalg.ldl (Bunch-Kaufman pivoting); ignored where a Hermitian diagonal's imaginary parts were set aside.");
    m.def<&inverse>("inverse", "inverse(a) -> Tensor\nThe inverse of a square matrix.");
    m.def<&solve_sylvester>("solve_sylvester", "solve_sylvester(a, b, q) -> Tensor\nX of A X + X B = Q, as scipy.linalg.solve_sylvester (Bartels-Stewart).");
    m.def<&solve_continuous_lyapunov>("solve_continuous_lyapunov", "solve_continuous_lyapunov(a, q) -> Tensor\nX of A X + X Aᴴ = Q, as scipy.linalg.solve_continuous_lyapunov (one Schur form).");
    m.def<&riccati>("riccati", "riccati(a, b, q, r, e, s, flags) -> Tensor\nThe stabilising solution of an algebraic Riccati equation, as scipy.linalg.solve_continuous_are (flags & 1 clear) or solve_discrete_are; e where flags & 2, s where flags & 4, balanced where flags & 8.");
    m.def<&sqrtm>("sqrtm", "sqrtm(a) -> Tensor\nThe principal matrix square root, as scipy.linalg.sqrtm (a stack's matrices each).");
    m.def<&expm_multiply>("expm_multiply", "expm_multiply(a, b, t) -> Tensor\ne^{tA} B without forming e^{tA}, as scipy.sparse.linalg.expm_multiply at one time (Al-Mohy and Higham's algorithm 3.2).");
    m.def<&expm>("expm", "expm(a) -> Tensor\nThe matrix exponential, as scipy.linalg.expm (a stack's matrices each).");
    m.def<&expm_frechet>("expm_frechet", "expm_frechet(a, e, block) -> (Tensor, Tensor)\nexpm(a) and the Frechet derivative of the exponential at a along e, as scipy.linalg.expm_frechet (Al-Mohy and Higham's algorithm, or the exponential of [[a, e], [0, a]] where block).");
    m.def<&expm_cond>("expm_cond", "expm_cond(a) -> float\nThe exponential's relative condition number in the Frobenius norm, as scipy.linalg.expm_cond.");
    m.def<&funm_call>("funm_call", "funm_call(a, evaluate) -> (F, err)\nA function of a square matrix, as scipy.linalg.funm: evaluate(points) returns f at the points numerical::funm asks for (the eigenvalues, then circles about gathered ones, more where their integrals ask); F real where a is real and its imaginary parts are negligible.");
    m.def<&fractional_matrix_power>("fractional_matrix_power", "fractional_matrix_power(a, p) -> Tensor\nA to the real power p, as scipy.linalg.fractional_matrix_power (Higham and Lin's Schur-Pade algorithm).");
    m.def<&logm>("logm", "logm(a) -> (F, err, exactly_singular, nearly_singular)\nThe principal matrix logarithm, as scipy.linalg.logm (inverse scaling and squaring), with its error estimate and the triangular form's singularity.");
    m.def<&lstsq>("lstsq", "lstsq(a, b, rtol) -> (x, residuals, rank, s)\nThe minimum-norm least-squares solution of a @ x ≈ b with its residual sums of squares, the rank and the singular values of a, as numpy.linalg.lstsq (SVD; a rank-2; b rank-1, or rank-2 with one right-hand side per column).");
    m.def<&solve_triangular>("solve_triangular", "solve_triangular(a, b, lower, trans, unit) -> Tensor\nop(A) X = B for the lower or upper triangle of a, op by trans (0 A, 1 Aᵀ, 2 Aᴴ), its diagonal ones where unit.");
    m.def<&lu_factor>("lu_factor", "lu_factor(a) -> (Tensor, list)\nL and U in one matrix, and LAPACK's row interchanges.");
    m.def<&lu_full>("lu_full", "lu_full(a) -> (rows, L, U)\nThe pivoted LU factorisation of an m × n matrix, as scipy.linalg.lu: row i of L @ U is row rows[i] of a.");
    m.def<&lu_solve>("lu_solve", "lu_solve(lu, piv, b, trans) -> Tensor\nThe solution of A x = b (trans 0), Aᵀ x = b (1) or Aᴴ x = b (2) from lu_factor's factors.");
    m.def<&cholesky>("cholesky", "cholesky(a, upper) -> Tensor\nThe lower factor L of A = L Lᴴ (Hermitian positive-definite), or with upper the factor U of A = Uᴴ U from its upper triangle.");
    m.def<&qr>("qr", "qr(a, mode) -> (Q, R)\nThe QR factorisation of an m×n matrix, as numpy.linalg.qr: reduced, Q is m×k, R k×n, k = min(m, n); complete, Q is m×m, R m×n.");
    m.def<&qr_r>("qr_r", "qr_r(a) -> R\nqr's R alone, k×n, k = min(m, n), to the bit, Q never formed (numpy.linalg.qr's mode=\"r\").");
    m.def<&schur>("schur", "schur(a, complex_output) -> (T, Z)\nThe Schur decomposition A = Z T Zᴴ, as scipy.linalg.schur: T quasi-triangular for a real matrix (triangular with complex_output, or for a complex one), Z orthogonal or unitary.");
    m.def<&schur_sorted>("schur_sorted", "schur_sorted(a, complex_output, kind, callback) -> (T, Z, sdim)\nschur with the eigenvalues scipy.linalg.schur's sort selects moved to the leading block of T (kind 0 lhp, 1 rhp, 2 iuc, 3 ouc, 4 the callable), as LAPACK's gees; sdim their count.");
    m.def<&eigh_subset>("eigh_subset", "eigh_subset(a, by_value, low, high, vectors, upper) -> (values, vectors)\nSome eigenpairs of a real symmetric or complex Hermitian matrix: the indices low to high, or (by_value) the eigenvalues in (low, high]; bisection and inverse iteration on its tridiagonal form.");
    m.def<&eigh_pencil_subset>("eigh_pencil_subset", "eigh_pencil_subset(a, b, by_value, low, high, vectors, upper) -> (values, vectors)\neigh_subset of the pencil A x = λ B x, B positive definite, the vectors B-orthonormal.");
    m.def<&tridiagonal_vectors>("tridiagonal_vectors", "tridiagonal_vectors(d, e, w) -> Tensor\nThe eigenvectors of a real symmetric tridiagonal for its eigenvalues w (ascending), by inverse iteration (dstein).");
    m.def<&eigh_pencil>("eigh_pencil", "eigh_pencil(a, b, method, upper) -> (values, vectors)\nEigenpairs of the pencil A x = λ B x, A symmetric or Hermitian, B positive definite (ascending values, B-orthonormal vectors), as scipy.linalg.eigh(a, b); each lower triangle read, or with upper each upper one.");
    m.def<&eigvalsh_pencil>("eigvalsh_pencil", "eigvalsh_pencil(a, b, upper) -> Tensor\nThe eigenvalues alone of the pencil A x = λ B x, ascending.");
    m.def<&eigh>("eigh", "eigh(a, method, upper) -> (values, vectors)\nEigenpairs of a real symmetric or complex Hermitian matrix (real values, ascending); its lower triangle is read, or with upper its upper one, as numpy.linalg.eigh's UPLO.");
    m.def<&matmul_nd>("_matmul_nd", "_matmul_nd(a, b) -> Tensor\nThe matrix product of two stacks of one dtype, their leading axes broadcast, as numpy.matmul.");
    m.def<&eig>("_eig", "_eig(a) -> (w, v)\nThe eigenvalues and right eigenvectors of a general square matrix (or stack), as numpy.linalg.eig: complex, as NumPy 2.5's.");
    m.def<&eigvals>("_eigvals", "_eigvals(a) -> w\nThe eigenvalues alone of a general square matrix (or stack), as numpy.linalg.eigvals: complex.");
    m.def<&eig_left>("_eig_left", "_eig_left(a) -> (w, vl, vr)\nThe eigenvalues and the left and right eigenvectors of a square matrix, as scipy.linalg.eig(a, left=True): complex.");
    m.def<&geneig>("_geneig", "_geneig(a, b, left, right) -> (alpha, beta, vl, vr)\nThe generalized eigenvalues (alpha, beta) of a pencil by the complex QZ and its left and right eigenvectors as asked (unit 2-norm; 0 x 0 otherwise), as scipy.linalg.eig(a, b).");
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
    m.raw("_tobytes", module_tobytes, METH_O, "_tobytes(t) -> bytes\nThe elements, row-major, in this machine's byte order.");
    m.raw("_frombytes", module_frombytes, METH_VARARGS,
          "_frombytes(data, shape, dtype, swapped) -> Tensor\nThe tensor of _tobytes's bytes, swapped from the other byte order.");
    m.raw("_permute", module_permute, METH_VARARGS,
          "_permute(t, order) -> Tensor\nThe tensor with its axes in order (a permutation), a copy of any dtype.");
    m.raw("_take", module_take, METH_VARARGS,
          "_take(t, starts, steps, counts, keep) -> Tensor or number\nThe strided copy behind Tensor.__getitem__.");
    m.raw("_bandwidth", module_bandwidth, METH_O, "_bandwidth(a) -> (lower, upper)\nThe matrix's lower and upper bandwidths.");
    m.raw("_gather", module_gather, METH_VARARGS,
          "_gather(t, groups, shape) -> Tensor\nThe elements at every sum of one offset per group: the advanced indexing.");
    m.raw("_true_offsets", module_true_offsets, METH_VARARGS,
          "_true_offsets(mask, strides) -> bytes\nThe int64 offsets, for those strides, of a bool mask's true elements.");
    m.raw("_power_shortcut", module_power_shortcut, METH_VARARGS,
          "_power_shortcut(kind, a) -> Tensor\nA complex tensor squared (\"2\"), inverted (\"r\") or rooted (\"h\") as NumPy's ** 2, ** -1, ** 0.5.");
    m.raw("_binary", module_binary, METH_VARARGS,
          "_binary(op, a, b) -> Tensor\nElement-wise a op b (op one of + - * /), broadcast; one dtype.");
    m.raw("_stack_last", module_stack_last, METH_VARARGS,
          "_stack_last(parts) -> Tensor\nTensors of one shape and dtype side by side along a new last axis (a copy).");
    m.raw("_concatenate", module_concatenate, METH_VARARGS,
          "_concatenate(parts, axis) -> Tensor\nTensors of one dtype and rank, one shape but along axis, one after the other along it (a copy).");
    m.raw("_negate", module_negate, METH_VARARGS, "_negate(a) -> Tensor\nElement-wise negation.");
    m.raw("_extreme_pair", module_extreme_pair, METH_VARARGS,
          "_extreme_pair(larger, a, b) -> Tensor\nnumpy.maximum (larger) or numpy.minimum of two real tensors of one dtype, broadcast.");
    m.raw("_axis_op", module_axis_op, METH_VARARGS, "_axis_op(name, a, axis) -> number or Tensor\nprod, cumsum, cumprod or sort along an axis.");
    m.raw("_arg_extreme", module_arg_extreme, METH_VARARGS, "_arg_extreme(largest, a, axis) -> int or Tensor\nargmax or argmin, of every element or along an axis.");
    m.raw("_argsort", module_argsort, METH_VARARGS, "_argsort(a, axis) -> Tensor\nThe indices that sort a along an axis, stably.");
    m.raw("_nonzero", module_nonzero, METH_VARARGS, "_nonzero(a) -> tuple\nThe indices of the nonzero elements, one int64 Tensor per axis.");
    m.raw("_count_nonzero", module_count_nonzero, METH_VARARGS, "_count_nonzero(a, axis) -> int or Tensor\nThe nonzero elements, counted.");
    m.raw("_take_along_axis", module_take_along_axis, METH_VARARGS, "_take_along_axis(a, indices, axis) -> Tensor\nThe elements at the indices along the axis.");
    m.raw("_select", module_select, METH_VARARGS,
          "_select(a, positions) -> Tensor\nThe elements a stable sort of each line along the last axis puts at positions.");
    m.raw("_correlate", module_correlate, METH_VARARGS, "_correlate(a, v, first, count) -> Tensor\nThe cross-correlation over count lags from first.");
    m.raw("_complex_from_parts", module_complex_from_parts, METH_VARARGS, "_complex_from_parts(re, im) -> Tensor\nre + im·i, exactly.");
    m.raw("_interp", module_interp, METH_VARARGS, "_interp(x, xp, fp, left, right, fused) -> Tensor\nnumpy.interp's linear interpolation.");
    m.raw("_bincount", module_bincount, METH_VARARGS, "_bincount(x, weights, length) -> Tensor\nThe counts of x's values, or their weights' sums.");
    m.raw("_searchsorted", module_searchsorted, METH_VARARGS, "_searchsorted(a, v, right) -> Tensor\nWhere v's elements go in the sorted vector a.");
    m.raw("_real_unary", module_real_unary, METH_VARARGS, "_real_unary(name, a) -> Tensor\nA real elementary function of every element.");
    m.raw("_bitwise", module_bitwise, METH_VARARGS,
          "_bitwise(op, a, b) -> Tensor\nTwo int64 tensors' & | ^, shifts (< >), gcd (g), lcm (l) or bit counts (c).");
    m.raw("_real_binary", module_real_binary, METH_VARARGS,
          "_real_binary(name, a, b) -> Tensor\nA real binary function of two tensors of one dtype, element by element, broadcast.");
    m.raw("_nan_to_num", module_nan_to_num, METH_VARARGS,
          "_nan_to_num(a, nan, posinf, neginf) -> Tensor\nEach NaN and infinity of a replaced, a complex element's parts apart.");
    m.raw("_round", module_round, METH_VARARGS, "_round(a, decimals) -> Tensor\nEvery element rounded to decimals places, half to even.");
    m.raw("_compare", module_compare, METH_VARARGS,
          "_compare(op, a, b) -> Tensor\nElement-wise a op b (op one of < <= == != > >=), broadcast; one dtype; bool.");
    m.raw("_logical", module_logical, METH_VARARGS,
          "_logical(op, a, b) -> Tensor\nElement-wise a & b, a | b or a ^ b of bool tensors, broadcast.");
    m.raw("_invert", module_invert, METH_VARARGS, "_invert(a) -> Tensor\nElement-wise negation of a bool tensor.");
    m.raw("_where", module_where, METH_VARARGS,
          "_where(condition, x, y) -> Tensor\nx where the bool condition holds, else y, the three broadcast.");
    m.raw("_classify", module_classify, METH_VARARGS,
          "_classify(name, a) -> Tensor\nisnan, isinf or isfinite of every element, as a bool tensor.");
    m.raw("_equal", module_equal, METH_VARARGS,
          "_equal(a, b) -> bool\nWhether two tensors of one dtype and size hold equal elements (a NaN equals nothing).");
    m.raw("_unary", module_unary, METH_VARARGS,
          "_unary(name, a) -> Tensor\nexp, log, sqrt, sin, cos, tan, tanh, abs, conj, real or imag of every element.");
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
          "_reduce(op, a, axis) -> number or Tensor\nsum, mean, max, min, any or all of every element (axis -1) or along an axis.");
    m.raw("_reshape", module_reshape, METH_VARARGS, "_reshape(a, shape) -> Tensor\nThe elements under another shape.");
    m.raw("_diagonal", module_diagonal, METH_VARARGS,
          "_diagonal(a, offset) -> Tensor\nThe offset diagonal of each matrix of a (its last two axes), read where it lies.");
    m.raw("_full", module_full, METH_VARARGS, "_full(shape, value, dtype) -> Tensor\nA tensor of one value.");
    m.raw("_eye", module_eye, METH_VARARGS, "_eye(n, dtype) -> Tensor\nThe n×n identity.");
    m.raw("_arange", module_arange, METH_VARARGS,
          "_arange(start, stop, step, fused=True) -> Tensor\nAs numpy.arange (float64), its elements fused or not.");
    m.raw("_linspace", module_linspace, METH_VARARGS,
          "_linspace(start, stop, num) -> Tensor\nAs numpy.linspace with the end point (float64).");
    m.raw("slogdet", module_slogdet, METH_VARARGS,
          "slogdet(a) -> (sign, logabsdet)\nThe sign and the natural log of abs(det(a)), as numpy.linalg.slogdet; it\n"
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
    m.raw("_set_getitem_fallback", module_set_getitem_fallback, METH_O,
          "_set_getitem_fallback(f)\nThe Python indexing Tensor._getitem hands every key it does not take.");
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
          "part); complex128->float64 is refused; bool is a number's truth.");
}
