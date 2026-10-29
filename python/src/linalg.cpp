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
            const sb::scoped_gil_release unlocked {releases_gil_cubic(count_of(*h))};
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
            const sb::scoped_gil_release unlocked {releases_gil_cubic(count_of(*h))};
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

// expm(a) -> Tensor, as scipy.linalg.expm: the exponential of a square matrix or of each of a stack's.
any_tensor expm(const any_tensor& v)
{
    if (shape_of(v).size() < 2 || shape_of(v)[shape_of(v).size() - 1] != shape_of(v)[shape_of(v).size() - 2]) {
        throw std::runtime_error("expm expects a square matrix or a stack of them");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [&](const any_tensor& m) { return expm(m); });
    }
    if (is_single(v)) {
        return narrowed(expm(widened(v)));
    }
    const sb::scoped_gil_release unlocked {releases_gil(count_of(v) * 8)};
    if (is_complex(v)) {
        return any_tensor {cplx_tensor {numerical::expm(matrix_from(std::get<cplx_tensor>(v)))}};
    }
    return any_tensor {real_tensor {numerical::expm(matrix_from(std::get<real_tensor>(v)))}};
}

// expm_frechet(a, e, block) -> (expm(a), L(a, e)), as scipy.linalg.expm_frechet ("SPS", or "blockEnlarge" where
// block): one dtype (the caller promotes), single precision computed in double.
std::tuple<any_tensor, any_tensor> expm_frechet(const any_tensor& a, const any_tensor& e, bool block)
{
    if (shape_of(a).size() != 2 || shape_of(e).size() != 2) {
        throw std::runtime_error("expm_frechet expects matrices (SciPy's batches are not taken)");
    }
    if (a.index() != e.index()) {
        throw std::runtime_error("expm_frechet operands must share one dtype");
    }
    if (is_single(a)) {
        auto [x, l] {expm_frechet(widened(a), widened(e), block)};
        return {narrowed(x), narrowed(l)};
    }
    const numerical::frechet_method method {block ? numerical::frechet_method::block_enlarge : numerical::frechet_method::scaling_pade_squaring};
    const sb::scoped_gil_release    unlocked {releases_gil(count_of(a) * 8)};
    if (is_complex(a)) {
        auto [x, l] {numerical::expm_frechet(matrix_from(std::get<cplx_tensor>(a)), matrix_from(std::get<cplx_tensor>(e)), method)};
        return {any_tensor {cplx_tensor {std::move(x)}}, any_tensor {cplx_tensor {std::move(l)}}};
    }
    auto [x, l] {numerical::expm_frechet(matrix_from(std::get<real_tensor>(a)), matrix_from(std::get<real_tensor>(e)), method)};
    return {any_tensor {real_tensor {std::move(x)}}, any_tensor {real_tensor {std::move(l)}}};
}

// expm_cond(a) -> float, as scipy.linalg.expm_cond: the exponential's relative condition number in the Frobenius norm.
double expm_cond(const any_tensor& a)
{
    if (shape_of(a).size() != 2) {
        throw std::runtime_error("expm_cond expects a matrix (SciPy's batches are not taken)");
    }
    if (is_single(a)) {
        return expm_cond(widened(a));
    }
    const sb::scoped_gil_release unlocked {releases_gil(count_of(a) * count_of(a) * 8)};
    if (is_complex(a)) {
        return numerical::expm_cond(matrix_from(std::get<cplx_tensor>(a)));
    }
    return numerical::expm_cond(matrix_from(std::get<real_tensor>(a)));
}

// funm_plan(a) -> (T, Z, points, hermitian): numerical::plan_funm's forms and the points where funm wants f (the
// eigenvalues, then the circles about gathered ones); complex128 whatever the dtype.
std::tuple<any_tensor, any_tensor, any_tensor, bool> funm_plan(const any_tensor& a)
{
    if (shape_of(a).size() != 2) {
        throw std::runtime_error("funm expects a matrix");
    }
    if (is_single(a)) {
        return funm_plan(widened(a));
    }
    const sb::scoped_gil_release unlocked {releases_gil_cubic(count_of(a))};
    numerical::funm_plan         plan {is_complex(a) ? numerical::plan_funm(matrix_from(std::get<cplx_tensor>(a)))
                                                     : numerical::plan_funm(matrix_from(std::get<real_tensor>(a)))};
    numerical::buffer<std::complex<double>> points(plan.points.begin(), plan.points.end());
    const std::size_t                       count {points.size()};
    return {any_tensor {cplx_tensor {std::move(plan.t)}}, any_tensor {cplx_tensor {std::move(plan.z)}},
            any_tensor {cplx_tensor {numerical::tensor<std::complex<double>> {{count}, std::move(points)}}}, plan.hermitian};
}

// funm_finish(t, z, points, images, hermitian, real) -> (F, err), numerical::finish_funm of funm_plan's forms and f at
// its points (complex128); F real where real (a real matrix) and every imaginary part is within 1e6 ε of 0 (SciPy's
// _maybe_real).
std::tuple<any_tensor, double> funm_finish(const any_tensor& t, const any_tensor& z, const any_tensor& points, const any_tensor& images,
                                           bool hermitian, bool real)
{
    if (!is_complex(t) || !is_complex(z) || !is_complex(points) || !is_complex(images) || is_single(t) || is_single(z) || is_single(points)
        || is_single(images)) {
        throw std::runtime_error("funm_finish takes complex128 tensors");
    }
    const auto&          p {std::get<cplx_tensor>(points).data()};
    const auto&          v {std::get<cplx_tensor>(images).data()};
    numerical::funm_plan plan {matrix_from(std::get<cplx_tensor>(t)), matrix_from(std::get<cplx_tensor>(z)),
                               std::vector<std::complex<double>>(p.begin(), p.end()), hermitian};
    const sb::scoped_gil_release unlocked {releases_gil(count_of(t) * 8)};
    numerical::funm_result r {numerical::finish_funm(plan, std::vector<std::complex<double>>(v.begin(), v.end()))};
    const double           tolerance {1e6 * std::numeric_limits<double>::epsilon()};
    const bool negligible {std::all_of(r.f.begin(), r.f.end(), [tolerance](const std::complex<double>& x) { return std::abs(x.imag()) <= tolerance; })};
    if (real && negligible) {
        numerical::buffer<double> parts(r.f.rows() * r.f.cols());
        std::transform(r.f.begin(), r.f.end(), parts.begin(), [](const std::complex<double>& x) { return x.real(); });
        return {any_tensor {real_tensor {numerical::matrix<double> {r.f.rows(), r.f.cols(), std::move(parts)}}}, r.error};
    }
    return {any_tensor {cplx_tensor {std::move(r.f)}}, r.error};
}

namespace {

    // ‖x‖₁, the largest column sum of moduli.
    template <typename T>
    double matrix_one_norm(const numerical::matrix<T>& x)
    {
        std::vector<double> sums(x.cols(), 0.0);
        for (std::size_t i {0}; i < x.rows(); ++i) {
            for (std::size_t j {0}; j < x.cols(); ++j) {
                sums[j] += std::abs(x(i, j));
            }
        }
        return sums.empty() ? 0.0 : *std::max_element(sums.begin(), sums.end());
    }

    // logm of one matrix as scipy.linalg.logm answers: F (real where a is and its imaginary parts are within 1e6 ε of
    // 0, SciPy's _maybe_real), ‖expm(F) − A‖₁ / ‖A‖₁, and the triangular form's singularity.
    template <typename T>
    std::tuple<any_tensor, double, bool, bool> logm_of(const numerical::matrix<T>& a)
    {
        const numerical::logm_result r {numerical::logm(a)};
        numerical::matrix<std::complex<double>> as_complex {a.rows(), a.cols(), numerical::buffer<std::complex<double>>(a.begin(), a.end())};
        const double error {matrix_one_norm(numerical::expm(r.log) - as_complex) / matrix_one_norm(as_complex)};
        const double tolerance {1e6 * std::numeric_limits<double>::epsilon()};
        const bool   negligible {std::all_of(r.log.begin(), r.log.end(), [tolerance](const std::complex<double>& x) { return std::abs(x.imag()) <= tolerance; })};
        if (std::is_same_v<T, double> && negligible) {
            numerical::buffer<double> parts(r.log.rows() * r.log.cols());
            std::transform(r.log.begin(), r.log.end(), parts.begin(), [](const std::complex<double>& x) { return x.real(); });
            return {any_tensor {real_tensor {numerical::matrix<double> {r.log.rows(), r.log.cols(), std::move(parts)}}}, error, r.exactly_singular,
                    r.nearly_singular};
        }
        return {any_tensor {cplx_tensor {r.log}}, error, r.exactly_singular, r.nearly_singular};
    }
} // namespace

// logm(a) -> (F, err, exactly_singular, nearly_singular), as scipy.linalg.logm of one matrix (the caller takes a
// stack's one at a time): single precision computed in double.
std::tuple<any_tensor, double, bool, bool> logm(const any_tensor& a)
{
    if (shape_of(a).size() != 2) {
        throw std::runtime_error("logm expects a matrix");
    }
    if (is_single(a)) {
        auto [f, error, exact, near] {logm(widened(a))};
        return {narrowed(f), error, exact, near};
    }
    const sb::scoped_gil_release unlocked {releases_gil(count_of(a) * 8)};
    if (is_complex(a)) {
        return logm_of(matrix_from(std::get<cplx_tensor>(a)));
    }
    return logm_of(matrix_from(std::get<real_tensor>(a)));
}

// fractional_matrix_power(a, p) -> Tensor, as scipy.linalg.fractional_matrix_power: complex, or real where a is and
// every imaginary part is 0 (the real Schur path SciPy keeps real); single precision computed in double.
any_tensor fractional_matrix_power(const any_tensor& a, double p)
{
    if (shape_of(a).size() != 2) {
        throw std::runtime_error("fractional_matrix_power expects a matrix");
    }
    if (is_single(a)) {
        return narrowed(fractional_matrix_power(widened(a), p));
    }
    const sb::scoped_gil_release           unlocked {releases_gil(count_of(a) * 8)};
    numerical::matrix<std::complex<double>> power {is_complex(a) ? numerical::fractional_matrix_power(matrix_from(std::get<cplx_tensor>(a)), p)
                                                                 : numerical::fractional_matrix_power(matrix_from(std::get<real_tensor>(a)), p)};
    const bool real {!is_complex(a) && std::all_of(power.begin(), power.end(), [](const std::complex<double>& x) { return x.imag() == 0.0; })};
    if (real) {
        numerical::buffer<double> parts(power.rows() * power.cols());
        std::transform(power.begin(), power.end(), parts.begin(), [](const std::complex<double>& x) { return x.real(); });
        return any_tensor {real_tensor {numerical::matrix<double> {power.rows(), power.cols(), std::move(parts)}}};
    }
    return any_tensor {cplx_tensor {std::move(power)}};
}

// sqrtm(a) -> Tensor, as scipy.linalg.sqrtm: the principal square root of a square matrix or of each of a stack's; a
// real operand's real where every matrix has one, else the whole complex (a real eigenvalue below 0 anywhere).
any_tensor sqrtm(const any_tensor& v)
{
    if (shape_of(v).size() < 2 || shape_of(v)[shape_of(v).size() - 1] != shape_of(v)[shape_of(v).size() - 2]) {
        throw std::runtime_error("sqrtm expects a square matrix or a stack of them");
    }
    if (is_single(v)) {
        return narrowed(sqrtm(widened(v)));
    }
    if (!is_complex(v)) {
        const auto real_root = [](const any_tensor& m) { return any_tensor {real_tensor {numerical::sqrtm(matrix_from(std::get<real_tensor>(m)))}}; };
        const auto complex_root = [](const any_tensor& m) {
            return any_tensor {cplx_tensor {numerical::complex_sqrtm(matrix_from(std::get<real_tensor>(m)))}};
        };
        try {
            if (shape_of(v).size() > 2) {
                return over_stack<any_tensor>(v, real_root);
            }
            const sb::scoped_gil_release unlocked {releases_gil(count_of(v) * 8)};
            return real_root(v);
        }
        catch (const std::domain_error&) { // a real eigenvalue below 0: every matrix's complex root, as SciPy's
            if (shape_of(v).size() > 2) {
                return over_stack<any_tensor>(v, complex_root);
            }
            const sb::scoped_gil_release unlocked {releases_gil(count_of(v) * 8)};
            return complex_root(v);
        }
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [](const any_tensor& m) { return sqrtm(m); });
    }
    const sb::scoped_gil_release unlocked {releases_gil(count_of(v) * 8)};
    return any_tensor {cplx_tensor {numerical::sqrtm(matrix_from(std::get<cplx_tensor>(v)))}};
}

// solve_sylvester(a, b, q) -> Tensor, as scipy.linalg.solve_sylvester: X of A X + X B = Q, one dtype (the caller
// promotes), single precision computed in double.
any_tensor solve_sylvester(const any_tensor& a, const any_tensor& b, const any_tensor& q)
{
    if (shape_of(a).size() != 2 || shape_of(b).size() != 2 || shape_of(q).size() != 2) {
        throw std::runtime_error("solve_sylvester expects matrices");
    }
    if (a.index() != b.index() || a.index() != q.index()) {
        throw std::runtime_error("solve_sylvester operands must share one dtype");
    }
    if (is_single(a)) {
        return narrowed(solve_sylvester(widened(a), widened(b), widened(q)));
    }
    const sb::scoped_gil_release unlocked {releases_gil(count_of(a) * 8)};
    if (is_complex(a)) {
        return any_tensor {cplx_tensor {numerical::solve_sylvester(matrix_from(std::get<cplx_tensor>(a)), matrix_from(std::get<cplx_tensor>(b)),
                                                                   matrix_from(std::get<cplx_tensor>(q)))}};
    }
    return any_tensor {real_tensor {numerical::solve_sylvester(matrix_from(std::get<real_tensor>(a)), matrix_from(std::get<real_tensor>(b)),
                                                               matrix_from(std::get<real_tensor>(q)))}};
}

namespace {

    // The Riccati solver \p flags picks, e and s passed on only where their flags are set.
    template <typename T>
    any_tensor riccati_of(const any_tensor& a, const any_tensor& b, const any_tensor& q, const any_tensor& r, const any_tensor& e,
                          const any_tensor& s, std::size_t flags)
    {
        const numerical::matrix<T> ea {matrix_from(std::get<numerical::tensor<T>>(e))};
        const numerical::matrix<T> sa {matrix_from(std::get<numerical::tensor<T>>(s))};
        const numerical::matrix<T>* ep {(flags & 2U) != 0 ? &ea : nullptr};
        const numerical::matrix<T>* sp {(flags & 4U) != 0 ? &sa : nullptr};
        const bool                  balanced {(flags & 8U) != 0};
        const auto                  operand = [](const any_tensor& v) { return matrix_from(std::get<numerical::tensor<T>>(v)); };
        if ((flags & 1U) != 0) {
            return any_tensor {numerical::tensor<T> {numerical::solve_discrete_are(operand(a), operand(b), operand(q), operand(r), ep, sp, balanced)}};
        }
        return any_tensor {numerical::tensor<T> {numerical::solve_continuous_are(operand(a), operand(b), operand(q), operand(r), ep, sp, balanced)}};
    }
} // namespace

// riccati(a, b, q, r, e, s, flags) -> Tensor, as scipy.linalg.solve_continuous_are (flags & 1 clear) or
// solve_discrete_are (set): e taken where flags & 2, s where flags & 4 (the caller passes some matrix of the dtype in
// their place otherwise), the pencil balanced where flags & 8; one dtype (the caller promotes), single precision
// computed in double.
any_tensor riccati(const any_tensor& a, const any_tensor& b, const any_tensor& q, const any_tensor& r, const any_tensor& e,
                   const any_tensor& s, std::size_t flags)
{
    for (const any_tensor* operand : {&a, &b, &q, &r, &e, &s}) {
        if (shape_of(*operand).size() != 2) {
            throw std::runtime_error("solve_continuous_are and solve_discrete_are expect matrices");
        }
        if (operand->index() != a.index()) {
            throw std::runtime_error("solve_continuous_are and solve_discrete_are operands must share one dtype");
        }
    }
    if (is_single(a)) {
        return narrowed(riccati(widened(a), widened(b), widened(q), widened(r), widened(e), widened(s), flags));
    }
    const sb::scoped_gil_release unlocked {releases_gil(count_of(a) * 64)};
    if (is_complex(a)) {
        return riccati_of<std::complex<double>>(a, b, q, r, e, s, flags);
    }
    return riccati_of<double>(a, b, q, r, e, s, flags);
}

namespace {

    // b as a matrix of right-hand sides (a vector one column), and back to its rank.
    template <typename T>
    numerical::matrix<T> right_hand_sides(const numerical::tensor<T>& b)
    {
        const std::size_t rows {shape_of(any_tensor {b})[0]};
        const std::size_t cols {b.shape().size() == 1 ? 1 : b.shape()[1]};
        return numerical::matrix<T> {rows, cols, numerical::buffer<T>(b.data().begin(), b.data().end())};
    }

    template <typename T>
    any_tensor shaped_like(numerical::matrix<T>&& x, const any_tensor& b)
    {
        return any_tensor {numerical::tensor<T> {shape_of(b), std::move(x).elements()}};
    }
} // namespace

// solve_banded(ab, b, lower, upper) -> Tensor, as scipy.linalg.solve_banded((lower, upper), ab, b): ab and b of one
// dtype (the caller promotes), b a vector or a matrix of right-hand sides; single precision computed in double.
any_tensor solve_banded(const any_tensor& ab, const any_tensor& b, std::size_t lower, std::size_t upper)
{
    if (shape_of(ab).size() != 2 || shape_of(b).empty() || shape_of(b).size() > 2 || ab.index() != b.index()) {
        throw std::runtime_error("solve_banded expects a band matrix and a vector or matrix of one dtype");
    }
    if (is_single(ab)) {
        return narrowed(solve_banded(widened(ab), widened(b), lower, upper));
    }
    const sb::scoped_gil_release unlocked {releases_gil(count_of(ab) * (lower + 1))};
    if (is_complex(ab)) {
        return shaped_like(numerical::solve_banded(lower, upper, matrix_from(std::get<cplx_tensor>(ab)), right_hand_sides(std::get<cplx_tensor>(b))), b);
    }
    return shaped_like(numerical::solve_banded(lower, upper, matrix_from(std::get<real_tensor>(ab)), right_hand_sides(std::get<real_tensor>(b))), b);
}

// solve_toeplitz(c, r, b) -> Tensor, as scipy.linalg.solve_toeplitz((c, r), b): c and r vectors, b a vector or a
// matrix, one dtype (the caller promotes); single precision computed in double.
any_tensor solve_toeplitz(const any_tensor& c, const any_tensor& r, const any_tensor& b)
{
    if (shape_of(c).size() != 1 || shape_of(r).size() != 1 || shape_of(b).empty() || shape_of(b).size() > 2 || c.index() != r.index()
        || c.index() != b.index()) {
        throw std::runtime_error("solve_toeplitz expects vectors c and r and a vector or matrix b of one dtype");
    }
    if (is_single(c)) {
        return narrowed(solve_toeplitz(widened(c), widened(r), widened(b)));
    }
    const sb::scoped_gil_release unlocked {releases_gil(count_of(c) * count_of(b))};
    if (is_complex(c)) {
        return shaped_like(numerical::solve_toeplitz(numerical::vector<std::complex<double>> {std::get<cplx_tensor>(c).data()},
                                                     numerical::vector<std::complex<double>> {std::get<cplx_tensor>(r).data()},
                                                     right_hand_sides(std::get<cplx_tensor>(b))),
                           b);
    }
    return shaped_like(numerical::solve_toeplitz(numerical::vector<double> {std::get<real_tensor>(c).data()}, numerical::vector<double> {std::get<real_tensor>(r).data()},
                                                 right_hand_sides(std::get<real_tensor>(b))),
                       b);
}

// solve_circulant(c, b, least_squares, tol) -> Tensor, as scipy.linalg.solve_circulant(c, b, singular, tol): tol
// negative for its default.
any_tensor solve_circulant(const any_tensor& c, const any_tensor& b, bool least_squares, double tol)
{
    if (shape_of(c).size() != 1 || shape_of(b).empty() || shape_of(b).size() > 2 || c.index() != b.index()) {
        throw std::runtime_error("solve_circulant expects a vector c and a vector or matrix b of one dtype");
    }
    if (is_single(c)) {
        return narrowed(solve_circulant(widened(c), widened(b), least_squares, tol));
    }
    const std::optional<double>  bound {tol < 0.0 ? std::nullopt : std::optional<double> {tol}};
    const sb::scoped_gil_release unlocked {releases_gil(count_of(b) * 8)};
    if (is_complex(c)) {
        return shaped_like(numerical::solve_circulant(numerical::vector<std::complex<double>> {std::get<cplx_tensor>(c).data()}, right_hand_sides(std::get<cplx_tensor>(b)),
                                                      least_squares, bound),
                           b);
    }
    return shaped_like(numerical::solve_circulant(numerical::vector<double> {std::get<real_tensor>(c).data()}, right_hand_sides(std::get<real_tensor>(b)), least_squares, bound),
                       b);
}

// solveh_banded(ab, b, lower) -> Tensor, as scipy.linalg.solveh_banded(ab, b, lower=lower).
any_tensor solveh_banded(const any_tensor& ab, const any_tensor& b, bool lower)
{
    if (shape_of(ab).size() != 2 || shape_of(b).empty() || shape_of(b).size() > 2 || ab.index() != b.index()) {
        throw std::runtime_error("solveh_banded expects a band matrix and a vector or matrix of one dtype");
    }
    if (is_single(ab)) {
        return narrowed(solveh_banded(widened(ab), widened(b), lower));
    }
    const sb::scoped_gil_release unlocked {releases_gil(count_of(ab) * shape_of(ab)[0])};
    if (is_complex(ab)) {
        return shaped_like(numerical::solveh_banded(matrix_from(std::get<cplx_tensor>(ab)), right_hand_sides(std::get<cplx_tensor>(b)), lower), b);
    }
    return shaped_like(numerical::solveh_banded(matrix_from(std::get<real_tensor>(ab)), right_hand_sides(std::get<real_tensor>(b)), lower), b);
}

// toeplitz(c, r), hankel(c, r) and circulant(c) -> Tensor, as scipy.linalg's: vectors of one dtype (the caller
// promotes), any of the four.
any_tensor structured(const any_tensor& c, const any_tensor& r, int kind)
{
    if (shape_of(c).size() != 1 || shape_of(r).size() != 1 || c.index() != r.index()) {
        throw std::runtime_error("toeplitz, hankel and circulant expect vectors of one dtype");
    }
    return std::visit([&](const auto& column) -> any_tensor {
                          using tensor_type = std::decay_t<decltype(column)>;
                          using T           = element_of_t<tensor_type>;
                          const numerical::vector<T> first {column.data()};
                          const numerical::vector<T> row {std::get<tensor_type>(r).data()};
                          switch (kind) {
                          case 0: return any_tensor {tensor_type {numerical::toeplitz(first, row)}};
                          case 1: return any_tensor {tensor_type {numerical::hankel(first, row)}};
                          default: return any_tensor {tensor_type {numerical::circulant(first)}};
                          }
                      }, c);
}

// _block_diag(blocks) -> Tensor, as scipy.linalg.block_diag: matrices of one dtype (the caller promotes).
PyObject* module_block_diag(PyObject* /*self*/, PyObject* args)
{
    PyObject* seq = nullptr;
    if (PyArg_ParseTuple(args, "O", &seq) == 0) {
        return nullptr;
    }
    const Py_ssize_t count {PySequence_Size(seq)};
    if (count <= 0) {
        if (count == 0) {
            PyErr_SetString(PyExc_ValueError, "_block_diag: at least one matrix");
        }
        return nullptr;
    }
    std::vector<const any_tensor*> parts;
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* item = PySequence_GetItem(seq, i); // new ref; the sequence keeps the tensor alive
        if (item == nullptr) {
            return nullptr;
        }
        const any_tensor* part = as_tensor(item);
        Py_DECREF(item);
        if (part == nullptr) {
            return nullptr;
        }
        if (shape_of(*part).size() != 2 || (!parts.empty() && part->index() != parts.front()->index())) {
            PyErr_SetString(sciforge_module_error(), "_block_diag: matrices of one dtype");
            return nullptr;
        }
        parts.push_back(part);
    }
    try {
        return std::visit([&](const auto& first) {
            using tensor_type = std::decay_t<decltype(first)>;
            using T           = element_of_t<tensor_type>;
            std::vector<numerical::matrix<T>> blocks;
            blocks.reserve(parts.size());
            for (const any_tensor* part : parts) {
                blocks.push_back(matrix_from(std::get<tensor_type>(*part)));
            }
            return wrap(any_tensor {tensor_type {numerical::block_diag(blocks)}});
        }, *parts.front());
    }
    catch (...) {
        return set_cpp_error();
    }
}

// qr_pivoted(a, complete) -> (Q, R, P), as scipy.linalg.qr(a, pivoting=True): P the columns' origins.
std::tuple<any_tensor, any_tensor, std::vector<std::size_t>> qr_pivoted(const any_tensor& a, bool complete)
{
    if (shape_of(a).size() != 2) {
        throw std::runtime_error("qr with pivoting expects a matrix");
    }
    if (is_single(a)) {
        auto [q, r, p] {qr_pivoted(widened(a), complete)};
        return {narrowed(q), narrowed(r), std::move(p)};
    }
    const numerical::qr_mode     mode {complete ? numerical::qr_mode::complete : numerical::qr_mode::reduced};
    const sb::scoped_gil_release unlocked {releases_gil(count_of(a) * 8)};
    return std::visit([&](const auto& t) -> std::tuple<any_tensor, any_tensor, std::vector<std::size_t>> {
                          using tensor_type = std::decay_t<decltype(t)>;
                          using T           = element_of_t<tensor_type>;
                          if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) {
                              const numerical::pivoted_qr_decomposition<T> f {matrix_from(t), mode};
                              return {any_tensor {tensor_type {f.q()}}, any_tensor {tensor_type {f.r()}}, f.permutation()};
                          }
                          else {
                              throw std::runtime_error("qr with pivoting: single precision is widened first");
                          }
                      }, a);
}

// ldl(a, lower, hermitian) -> (lu, d, perm, ignored), as scipy.linalg.ldl: ignored where a Hermitian complex matrix's
// diagonal has an imaginary part (SciPy's ComplexWarning); single precision computed in double.
std::tuple<any_tensor, any_tensor, std::vector<std::size_t>, bool> ldl(const any_tensor& a, bool lower, bool hermitian)
{
    if (shape_of(a).size() != 2) {
        throw std::runtime_error("ldl expects a matrix");
    }
    if (is_single(a)) {
        auto [l, d, p, ignored] {ldl(widened(a), lower, hermitian)};
        return {narrowed(l), narrowed(d), std::move(p), ignored};
    }
    const sb::scoped_gil_release unlocked {releases_gil(count_of(a) * 8)};
    return std::visit([&](const auto& t) -> std::tuple<any_tensor, any_tensor, std::vector<std::size_t>, bool> {
                          using tensor_type = std::decay_t<decltype(t)>;
                          using T           = element_of_t<tensor_type>;
                          if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) {
                              const numerical::matrix<T>               m {matrix_from(t)};
                              const numerical::ldl_decomposition<T>    f {m, lower, hermitian};
                              bool                                     ignored {false};
                              for (std::size_t i {0}; i < m.rows() && i < m.cols(); ++i) {
                                  ignored = ignored || (hermitian && std::imag(m(i, i)) != 0.0);
                              }
                              return {any_tensor {tensor_type {f.factor()}}, any_tensor {tensor_type {f.d()}}, f.permutation(), ignored};
                          }
                          else {
                              throw std::runtime_error("ldl: single precision is widened first");
                          }
                      }, a);
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
                const sb::scoped_gil_release unlocked {releases_gil_cubic(order * order)};
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
                const sb::scoped_gil_release unlocked {releases_gil_cubic(n * n)};
                x.emplace(solve_copied(matrix_of_elements(static_cast<const T*>(left.data), n, n), *rhs));
            }
            return wrap(std::move(*x));
        }
        if (operation == "inverse") {
            std::optional<numerical::tensor<T>> inverse;
            {
                const sb::scoped_gil_release unlocked {releases_gil_cubic(n * n)};
                inverse.emplace(numerical::inverse(matrix_of_elements(static_cast<const T*>(left.data), n, n)));
            }
            return wrap(any_tensor {std::move(*inverse)});
        }
        if (operation == "determinant") {
            std::optional<T> value;
            {
                const sb::scoped_gil_release unlocked {releases_gil_cubic(n * n)};
                value.emplace(numerical::determinant(matrix_of_elements(static_cast<const T*>(left.data), n, n)));
            }
            return from_scalar(*value);
        }
        if (operation == "slogdet") {
            std::optional<std::pair<T, double>> value;
            {
                const sb::scoped_gil_release unlocked {releases_gil_cubic(n * n)};
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
