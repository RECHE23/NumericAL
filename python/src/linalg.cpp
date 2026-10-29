// NumericAL — Python extension module: the linear systems, factorisations, determinants, ranks and pseudo-inverse
// (their helpers in binding.hpp).

#include "binding.hpp"

namespace numerical_binding {

// solve(a, b), sharing a dtype: a a matrix or a stack of them (..., m, m); b a vector (m,), solved against every
// matrix, or matrices (..., m, k), the leading axes broadcast as NumPy 2 broadcasts them.
any_tensor solve(const any_tensor& a, const any_tensor& b)
{
    const std::size_t rank_a {shape_of(a).size()};
    const std::size_t rank_b {shape_of(b).size()};
    if (rank_a < 2 || rank_b < 1) {
        throw std::runtime_error("solve(a, b) needs a matrix or a stack of them, and a vector or matrices b");
    }
    if (a.index() != b.index()) {
        throw std::runtime_error("solve operands must share one dtype");
    }
    if (rank_a > 2 || rank_b > 2) {
        const stack_shape              sa {stack_of(a)};
        const bool                     vector_b {rank_b == 1};
        const std::vector<std::size_t> own_b {vector_b ? std::vector<std::size_t> {} : stack_of(b).batch};
        const std::vector<std::size_t> batch {broadcast_batch(sa.batch, own_b)};
        const std::size_t              count {numerical::detail::element_count(batch.begin(), batch.end())};
        const std::size_t b_rows {vector_b ? 0 : shape_of(b)[rank_b - 2]};
        const std::size_t b_cols {shape_of(b).back()};
        const auto        right_side {[&](std::size_t i) {
            return vector_b ? b : matrix_of(b, b_rows, b_cols, broadcast_index(i, batch, own_b));
        }};
        return over_batch<any_tensor>(batch, count, sa.rows, count_of(a) + count_of(b),
                                      [&](std::size_t i) { return solve(matrix_of(a, sa.rows, sa.cols, broadcast_index(i, batch, sa.batch)), right_side(i)); },
                                      [&] { return solve(identity_like(a, sa.rows, sa.cols), vector_b ? b : identity_like(b, b_rows, b_cols)); });
    }
    if (is_single(a)) {
        return narrowed(solve(widened(a), widened(b)));
    }
    if (is_complex(a)) {
        return do_solve<std::complex<double>>(std::get<cplx_tensor>(a), std::get<cplx_tensor>(b));
    }
    return do_solve<double>(std::get<real_tensor>(a), std::get<real_tensor>(b));
}

// determinant(a) -> float or complex: a dtype-dependent scalar (a variant return no single
// caster covers), so it stays a manual .raw function; a stack's determinants, a Tensor.
PyObject* module_determinant(PyObject* /*self*/, PyObject* args)
{
    PyObject* obj = nullptr;
    if (PyArg_ParseTuple(args, "O", &obj) == 0) {
        return nullptr;
    }
    const any_tensor* h = as_tensor(obj);
    if (h == nullptr) {
        return nullptr;
    }
    if (shape_of(*h).size() < 2) {
        PyErr_SetString(sciforge_module_error(), "determinant expects a matrix or a stack of matrices (rank ≥ 2)");
        return nullptr;
    }
    try {
        if (shape_of(*h).size() > 2) {
            return wrap(over_stack<any_tensor>(*h, determinant_tensor));
        }
        std::optional<any_tensor> value; // a single-precision matrix's rounded to its precision, as a stack's
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*h))};
            value.emplace(determinant_tensor(*h));
        }
        return std::visit([](const auto& t) { return from_scalar(t.data()[0]); }, *value);
    }
    catch (...) {
        return set_cpp_error();
    }
}

// slogdet(a) -> (sign, logabsdet), as numpy.linalg.slogdet: the sign a float or complex like determinant.
PyObject* module_slogdet(PyObject* /*self*/, PyObject* args)
{
    PyObject* obj = nullptr;
    if (PyArg_ParseTuple(args, "O", &obj) == 0) {
        return nullptr;
    }
    const any_tensor* h = as_tensor(obj);
    if (h == nullptr) {
        return nullptr;
    }
    if (shape_of(*h).size() < 2) {
        PyErr_SetString(sciforge_module_error(), "slogdet expects a matrix or a stack of matrices (rank ≥ 2)");
        return nullptr;
    }
    try {
        if (shape_of(*h).size() > 2) {
            auto [sign, logarithm] {over_stack<std::tuple<any_tensor, any_tensor>>(*h, slogdet_tensors)};
            return Py_BuildValue("(NN)", wrap(std::move(sign)), wrap(std::move(logarithm)));
        }
        std::optional<std::tuple<any_tensor, any_tensor>> value; // as determinant's
        {
            const sb::scoped_gil_release unlocked {releases_gil(count_of(*h))};
            value.emplace(slogdet_tensors(*h));
        }
        const auto scalar {[](const any_tensor& v) { return std::visit([](const auto& t) { return from_scalar(t.data()[0]); }, v); }};
        PyObject* sign {scalar(std::get<0>(*value))}; // new ref
        if (sign == nullptr) {
            return nullptr;
        }
        const double logarithm {std::visit([](const auto& t) { return static_cast<double>(std::real(t.data()[0])); }, std::get<1>(*value))};
        return Py_BuildValue("(Nd)", sign, logarithm); // N: the tuple takes the reference
    }
    catch (...) {
        return set_cpp_error();
    }
}

// matrix_rank(a, rtol, tol, hermitian) -> int, as numpy.linalg.matrix_rank.
std::size_t matrix_rank(const any_tensor& v, std::optional<double> rtol, std::optional<double> tol, bool hermitian)
{
    if (shape_of(v).size() != 2) {
        throw std::runtime_error("matrix_rank expects a matrix (rank-2 tensor); a stack goes through _matrix_ranks");
    }
    if (is_single(v)) {
        return matrix_rank(widened(v), single_rtol(v, rtol, tol), tol, hermitian);
    }
    const numerical::rank_cutoff cutoff {cutoff_named(rtol, tol)};
    if (is_complex(v)) {
        return do_matrix_rank<std::complex<double>>(std::get<cplx_tensor>(v), cutoff, hermitian);
    }
    return do_matrix_rank<double>(std::get<real_tensor>(v), cutoff, hermitian);
}

// _matrix_ranks(a, rtol, tol, hermitian) -> Tensor: the rank of every matrix of the stack `a` (rank > 2), as float64
// numbers along its leading axes (the Python side makes them integers).
any_tensor matrix_ranks(const any_tensor& v, std::optional<double> rtol, std::optional<double> tol, bool hermitian)
{
    if (shape_of(v).size() <= 2) {
        throw std::runtime_error("_matrix_ranks expects a stack of matrices (rank > 2)");
    }
    return over_stack<any_tensor>(v, [&](const any_tensor& m) {
        return any_tensor {real_tensor {{}, numerical::buffer<double> {static_cast<double>(matrix_rank(m, rtol, tol, hermitian))}}};
    });
}

// pinv(a, rtol, hermitian) -> Tensor, as numpy.linalg.pinv.
any_tensor pinv(const any_tensor& v, std::optional<double> rtol, bool hermitian)
{
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("pinv expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [&](const any_tensor& m) { return pinv(m, rtol, hermitian); });
    }
    if (is_single(v)) {
        return narrowed(pinv(widened(v), single_rtol(v, rtol, std::nullopt), hermitian));
    }
    const numerical::rank_cutoff cutoff {cutoff_named(rtol, std::nullopt)};
    if (is_complex(v)) {
        return do_pinv<std::complex<double>>(std::get<cplx_tensor>(v), cutoff, hermitian);
    }
    return do_pinv<double>(std::get<real_tensor>(v), cutoff, hermitian);
}

// inverse(a) -> Tensor (the matrix inverse).
any_tensor inverse(const any_tensor& v)
{
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("inverse expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [&](const any_tensor& m) { return inverse(m); });
    }
    if (is_single(v)) {
        return narrowed(inverse(widened(v)));
    }
    if (is_complex(v)) {
        return do_inverse<std::complex<double>>(std::get<cplx_tensor>(v));
    }
    return do_inverse<double>(std::get<real_tensor>(v));
}

// lstsq(a, b, rtol) -> (x, residuals, rank, s), as numpy.linalg.lstsq (SVD; a rank-2; b rank-1, or rank-2 with one
// right-hand side per column).
lstsq_result lstsq(const any_tensor& a, const any_tensor& b, std::optional<double> rtol)
{
    if (shape_of(a).size() != 2 || (shape_of(b).size() != 1 && shape_of(b).size() != 2)) {
        throw std::runtime_error("lstsq(a, b) needs a rank-2 a and a rank-1 or rank-2 b");
    }
    if (a.index() != b.index()) {
        throw std::runtime_error("lstsq operands must share one dtype");
    }
    if (is_single(a)) {
        return narrowed(lstsq(widened(a), widened(b), single_rtol(a, rtol, std::nullopt)));
    }
    const numerical::rank_cutoff cutoff {cutoff_named(rtol, std::nullopt)};
    if (is_complex(a)) {
        return do_lstsq<std::complex<double>>(std::get<cplx_tensor>(a), std::get<cplx_tensor>(b), cutoff);
    }
    return do_lstsq<double>(std::get<real_tensor>(a), std::get<real_tensor>(b), cutoff);
}

// cholesky(a, upper) -> Tensor: the lower factor L of A = L Lᴴ, or with upper the factor U of A = Uᴴ U, from A's upper
// triangle, as numpy.linalg.cholesky(a, upper=True).
any_tensor cholesky(const any_tensor& v, bool upper)
{
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("cholesky expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [&](const any_tensor& m) { return cholesky(m, upper); });
    }
    if (is_single(v)) {
        return narrowed(cholesky(widened(v), upper));
    }
    if (is_complex(v)) {
        return do_cholesky<std::complex<double>>(std::get<cplx_tensor>(v), upper);
    }
    return do_cholesky<double>(std::get<real_tensor>(v), upper);
}

// solve_triangular(a, b, lower, trans, unit) -> Tensor, as scipy.linalg.solve_triangular: a square, b a vector or a
// matrix of a's dtype; single precision solved in double and rounded, as the other solves.
any_tensor solve_triangular(const any_tensor& a, const any_tensor& b, bool lower, int trans, bool unit)
{
    const numerical::triangular_op op {triangular_op_of(trans)};
    if (shape_of(a).size() != 2 || shape_of(a)[0] != shape_of(a)[1]) {
        throw std::runtime_error("solve_triangular: expected a square matrix");
    }
    if (shape_of(b).empty() || shape_of(b).size() > 2) {
        throw std::runtime_error("solve_triangular: b must be a vector or a matrix");
    }
    if (a.index() != b.index()) {
        throw std::runtime_error("solve_triangular operands must share one dtype");
    }
    if (is_single(a)) {
        return narrowed(solve_triangular(widened(a), widened(b), lower, trans, unit));
    }
    if (is_complex(a)) {
        return do_solve_triangular<std::complex<double>>(std::get<cplx_tensor>(a), std::get<cplx_tensor>(b), lower, op, unit);
    }
    return do_solve_triangular<double>(std::get<real_tensor>(a), std::get<real_tensor>(b), lower, op, unit);
}

std::tuple<any_tensor, std::vector<std::size_t>> lu_factor(const any_tensor& a)
{
    if (shape_of(a).size() != 2 || shape_of(a)[0] != shape_of(a)[1]) {
        throw std::runtime_error("lu_factor: expected a square matrix");
    }
    if (is_single(a)) {
        auto [lu, swaps] {lu_factor(widened(a))};
        return {narrowed(lu), swaps};
    }
    if (is_complex(a)) {
        return do_lu_factor<std::complex<double>>(std::get<cplx_tensor>(a));
    }
    return do_lu_factor<double>(std::get<real_tensor>(a));
}

// LAPACK's interchanges in `piv` (a tuple or list of ints read directly, any other sequence item by item), n of them,
// each in [0, n), into `swaps`; false, no error set, for anything else.
bool interchanges_of(PyObject* piv, std::size_t n, std::vector<std::size_t>& swaps)
{
    const bool       tuple {PyTuple_Check(piv) != 0};
    const bool       list {PyList_Check(piv) != 0};
    const Py_ssize_t length {tuple ? PyTuple_Size(piv) : list ? PyList_Size(piv) : PySequence_Size(piv)};
    if (length < 0 || static_cast<std::size_t>(length) != n) {
        PyErr_Clear();
        return false;
    }
    swaps.resize(n);
    for (std::size_t i {0}; i < n; ++i) {
        const auto       at {static_cast<Py_ssize_t>(i)};
        PyObject* const  item {tuple  ? Py_XNewRef(PyTuple_GetItem(piv, at))
                               : list ? Py_XNewRef(PyList_GetItem(piv, at))
                                      : PySequence_GetItem(piv, at)}; // new ref
        const Py_ssize_t value {item == nullptr ? -1 : PyLong_Check(item) != 0 ? PyLong_AsSsize_t(item) : PyNumber_AsSsize_t(item, nullptr)};
        Py_XDECREF(item);
        if (PyErr_Occurred() != nullptr || value < 0 || static_cast<std::size_t>(value) >= n) {
            PyErr_Clear();
            return false;
        }
        swaps[i] = static_cast<std::size_t>(value);
    }
    return true;
}

// lu_solve(lu, piv, b, trans) -> Tensor, as scipy.linalg.lu_solve((lu, piv), b, trans): `piv` a sequence of LAPACK's
// interchanges (lu_factor's), each in [0, n); a zero on U's diagonal raises, where SciPy's divides by it.
any_tensor lu_solve(const any_tensor& lu, PyObject* piv, const any_tensor& b, int trans)
{
    const numerical::triangular_op op {triangular_op_of(trans)};
    if (shape_of(lu).size() != 2 || shape_of(lu)[0] != shape_of(lu)[1]) {
        throw std::runtime_error("lu_solve: expected a square lu");
    }
    const std::size_t n {shape_of(lu)[0]};
    if (shape_of(b).empty() || shape_of(b).size() > 2 || shape_of(b)[0] != n) {
        throw std::runtime_error("lu_solve: b must be a vector or a matrix of the factor's row count");
    }
    if (lu.index() != b.index()) {
        throw std::runtime_error("lu_solve operands must share one dtype");
    }
    std::vector<std::size_t> swaps;
    if (!interchanges_of(piv, n, swaps)) {
        throw std::runtime_error("lu_solve: piv must be a sequence of the factor's row count, each interchange in [0, n)");
    }
    if (is_single(lu)) {
        return narrowed(lu_solve(widened(lu), piv, widened(b), trans));
    }
    if (is_complex(lu)) {
        return lu_solved(std::get<cplx_tensor>(lu).data().data(), swaps, std::get<cplx_tensor>(b).data().data(), shape_of(b), op);
    }
    return lu_solved(std::get<real_tensor>(lu).data().data(), swaps, std::get<real_tensor>(b).data().data(), shape_of(b), op);
}

// qr(a, mode) -> (Q, R), mode "reduced" or "complete", through the tuple-caster.
std::tuple<any_tensor, any_tensor> qr(const any_tensor& v, const std::string& name)
{
    using numerical::qr_mode;
    const qr_mode mode {method_named<qr_mode>(name, {{"reduced", qr_mode::reduced}, {"complete", qr_mode::complete}})};
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("qr expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<std::tuple<any_tensor, any_tensor>>(v, [&](const any_tensor& m) { return qr(m, name); });
    }
    if (is_single(v)) {
        return narrowed(qr(widened(v), name));
    }
    if (is_complex(v)) {
        return do_qr<std::complex<double>>(std::get<cplx_tensor>(v), mode);
    }
    return do_qr<double>(std::get<real_tensor>(v), mode);
}

// qr_r(a) -> R, k×n, qr's R to the bit with Q never formed (numpy.linalg.qr's mode "r").
any_tensor qr_r(const any_tensor& v)
{
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("qr expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [](const any_tensor& m) { return qr_r(m); });
    }
    if (is_single(v)) {
        return narrowed(qr_r(widened(v)));
    }
    if (is_complex(v)) {
        return any_tensor {numerical::tensor<std::complex<double>> {numerical::qr_r(matrix_from(std::get<cplx_tensor>(v)))}};
    }
    return any_tensor {numerical::tensor<double> {numerical::qr_r(matrix_from(std::get<real_tensor>(v)))}};
}

// _borrowed_linalg(name, a[, b]) -> the answer, or NotImplemented: cholesky (b its upper flag), solve, determinant, slogdet or inverse of a square
// float64 or complex128 NumPy array read where it lies (borrow_operand), its one copy the one the factorisation takes
// in place, where asarray's Tensor was a copy before it; b, if any, of the same dtype and rank 1 or 2 (a NumPy array
// copied, or a Tensor; its rows are the solve's to check). NotImplemented for anything else (a Tensor, a stack, single
// precision, a shape the converting path rejects), the caller then converting: each answer is the converting
// path's, from the same values, and so is each error.
PyObject* module_borrowed_linalg(PyObject* /*self*/, PyObject* args)
{
    const char* name = nullptr;
    PyObject*   a    = nullptr;
    PyObject*   b    = nullptr;
    if (PyArg_ParseTuple(args, "sO|O", &name, &a, &b) == 0) {
        return nullptr;
    }
    const std::string operation {name};
    if (operation == "cholesky") { // the third argument is the upper flag, not a right-hand side
        const int upper {b == nullptr ? 0 : PyObject_IsTrue(b)};
        if (upper < 0) {
            return nullptr;
        }
        borrowed_operand matrix;
        const bool       tensor {PyObject_TypeCheck(a, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0};
        if (tensor || !borrow_operand(a, matrix) || matrix.dtype > 1 || matrix.shape.size() != 2 || matrix.shape[0] != matrix.shape[1]
            || matrix.shape[0] == 0) {
            Py_RETURN_NOTIMPLEMENTED;
        }
        const std::size_t order {matrix.shape[0]};
        const auto factor = [&]<typename T>(std::type_identity<T>) -> PyObject* {
            std::optional<any_tensor> l;
            {
                const sb::scoped_gil_release unlocked {releases_gil(order * order)};
                l.emplace(cholesky_copied(lower_read(matrix_of_elements(static_cast<const T*>(matrix.data), order, order), upper != 0), upper != 0));
            }
            return wrap(std::move(*l));
        };
        try {
            return matrix.dtype == 0 ? factor(std::type_identity<double> {}) : factor(std::type_identity<std::complex<double>> {});
        }
        catch (...) {
            return set_cpp_error();
        }
    }
    borrowed_operand  left;
    const bool        tensor {PyObject_TypeCheck(a, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0};
    if (tensor || !borrow_operand(a, left) || left.dtype > 1 || left.shape.size() != 2 || left.shape[0] != left.shape[1]
        || left.shape[0] == 0 || (operation == "solve") != (b != nullptr)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const std::size_t n {left.shape[0]};
    borrowed_operand  right;
    if (b != nullptr && (!borrow_operand(b, right) || right.dtype != left.dtype || right.shape.empty() || right.shape.size() > 2)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const auto run = [&]<typename T>(std::type_identity<T>) -> PyObject* {
        std::optional<numerical::tensor<T>> rhs;
        if (b != nullptr) {
            const bool rhs_tensor {PyObject_TypeCheck(b, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0};
            if (rhs_tensor) {
                rhs.emplace(std::get<numerical::tensor<T>>(*reinterpret_cast<sb::wrapper<any_tensor>*>(b)->held));
            }
            else {
                rhs.emplace(owned_copy<T>(right));
            }
        }
        if (operation == "solve") {
            std::optional<any_tensor> x;
            {
                const sb::scoped_gil_release unlocked {releases_gil(n * n)};
                x.emplace(solve_copied(matrix_of_elements(static_cast<const T*>(left.data), n, n), *rhs));
            }
            return wrap(std::move(*x));
        }
        if (operation == "inverse") {
            std::optional<numerical::tensor<T>> inverse;
            {
                const sb::scoped_gil_release unlocked {releases_gil(n * n)};
                inverse.emplace(numerical::inverse(matrix_of_elements(static_cast<const T*>(left.data), n, n)));
            }
            return wrap(any_tensor {std::move(*inverse)});
        }
        if (operation == "determinant") {
            std::optional<T> value;
            {
                const sb::scoped_gil_release unlocked {releases_gil(n * n)};
                value.emplace(numerical::determinant(matrix_of_elements(static_cast<const T*>(left.data), n, n)));
            }
            return from_scalar(*value);
        }
        if (operation == "slogdet") {
            std::optional<std::pair<T, double>> value;
            {
                const sb::scoped_gil_release unlocked {releases_gil(n * n)};
                value.emplace(numerical::lu_decomposition<T> {matrix_of_elements(static_cast<const T*>(left.data), n, n)}.log_determinant());
            }
            PyObject* sign {from_scalar(value->first)}; // new ref
            if (sign == nullptr) {
                return nullptr;
            }
            return Py_BuildValue("(Nd)", sign, value->second);
        }
        Py_RETURN_NOTIMPLEMENTED;
    };
    try {
        return left.dtype == 0 ? run(std::type_identity<double> {}) : run(std::type_identity<std::complex<double>> {});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _borrowed_solve_triangular(a, b, lower, trans, unit) -> Tensor or NotImplemented: solve_triangular of a square
// float64 or complex128 NumPy array read where it lies (borrow_operand), b a Tensor or NumPy array of its dtype, rank 1
// or 2, its rows a's, copied once into the answer the solve overwrites. NotImplemented for anything else (a Tensor a,
// single precision, a shape or dtype the converting path rejects), the caller then converting: the same answer and
// the same errors either way. A NumPy a was copied into a Tensor first: order 4000 with 64 columns, M1 Pro, 14.7 ms
// where the solve takes 9.9.
PyObject* module_borrowed_solve_triangular(PyObject* /*self*/, PyObject* args)
{
    PyObject* a     = nullptr;
    PyObject* b     = nullptr;
    int       lower = 0;
    int       trans = 0;
    int       unit  = 0;
    if (PyArg_ParseTuple(args, "OOpip", &a, &b, &lower, &trans, &unit) == 0) {
        return nullptr;
    }
    borrowed_operand matrix;
    const bool       tensor {PyObject_TypeCheck(a, reinterpret_cast<PyTypeObject*>(sb::class_type<any_tensor>())) != 0};
    if (tensor || !borrow_operand(a, matrix) || matrix.dtype > 1 || matrix.shape.size() != 2 || matrix.shape[0] != matrix.shape[1]) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const std::size_t n {matrix.shape[0]};
    borrowed_operand  right;
    if (!borrow_operand(b, right) || right.dtype != matrix.dtype || right.shape.empty() || right.shape.size() > 2 || right.shape[0] != n) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const std::size_t k {right.shape.size() == 1 ? 1 : right.shape[1]};
    const auto solve = [&]<typename T>(std::type_identity<T>) -> PyObject* {
        const numerical::triangular_op op {triangular_op_of(trans)};
        const T* const                 first {static_cast<const T*>(right.data)};
        numerical::buffer<T>           x(first, first + (n * k));
        {
            const sb::scoped_gil_release unlocked {releases_gil((n * n) + (n * k))};
            numerical::detail::solve_triangular_in_place(static_cast<const T*>(matrix.data), n, lower != 0, op, unit != 0, x.data(), k);
        }
        return wrap(any_tensor {numerical::tensor<T> {right.shape, std::move(x)}});
    };
    try {
        return matrix.dtype == 0 ? solve(std::type_identity<double> {}) : solve(std::type_identity<std::complex<double>> {});
    }
    catch (...) {
        return set_cpp_error();
    }
}

// _borrowed_factor_solve(factor, piv, b, lower, trans) -> Tensor or NotImplemented: lu_solve (piv a sequence of
// interchanges) or cho_solve (piv None, `lower` the factor's triangle) of a float64 or complex128 factor and right-hand
// sides, each a Tensor or a NumPy array read where it lies (borrow_operand), b copied once into the answer. NotImplemented
// for anything else (single precision, mixed dtypes, a shape or piv the converting path rejects), the caller then
// converting: the same answer and the same errors either way.
PyObject* module_borrowed_factor_solve(PyObject* /*self*/, PyObject* args)
{
    PyObject* a     = nullptr;
    PyObject* piv   = nullptr;
    PyObject* b     = nullptr;
    int       lower = 0;
    int       trans = 0;
    if (PyArg_ParseTuple(args, "OOOpi", &a, &piv, &b, &lower, &trans) == 0) {
        return nullptr;
    }
    borrowed_operand matrix;
    if (!borrow_operand(a, matrix) || matrix.dtype > 1 || matrix.shape.size() != 2 || matrix.shape[0] != matrix.shape[1]) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const std::size_t n {matrix.shape[0]};
    borrowed_operand  right;
    if (!borrow_operand(b, right) || right.dtype != matrix.dtype || right.shape.empty() || right.shape.size() > 2 || right.shape[0] != n) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    std::vector<std::size_t> swaps;
    if (piv != Py_None && !interchanges_of(piv, n, swaps)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const auto solve = [&]<typename T>(std::type_identity<T>) -> PyObject* {
        const T* const factor {static_cast<const T*>(matrix.data)};
        const T* const rhs {static_cast<const T*>(right.data)};
        return wrap(piv == Py_None ? cho_solved(factor, lower != 0, rhs, right.shape) : lu_solved(factor, swaps, rhs, right.shape, triangular_op_of(trans)));
    };
    try {
        return matrix.dtype == 0 ? solve(std::type_identity<double> {}) : solve(std::type_identity<std::complex<double>> {});
    }
    catch (...) {
        return set_cpp_error();
    }
}

} // namespace numerical_binding
