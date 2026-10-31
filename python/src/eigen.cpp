// NumericAL — Python extension module: the eigenvalue, Schur and singular-value decompositions (their helpers in
// binding.hpp).

#include "binding.hpp"

namespace numerical_binding {

any_tensor eigvalsh_pencil(const any_tensor& a, const any_tensor& b, bool upper)
{
    require_pencil(a, b, "eigvalsh");
    if (is_single(a)) {
        return narrowed(eigvalsh_pencil(widened(a), widened(b), upper));
    }
    if (is_complex(a)) {
        return do_eigvalsh_pencil<std::complex<double>>(std::get<cplx_tensor>(a), std::get<cplx_tensor>(b), upper);
    }
    return do_eigvalsh_pencil<double>(std::get<real_tensor>(a), std::get<real_tensor>(b), upper);
}

any_tensor eigvalsh(const any_tensor& v, bool upper)
{
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("eigvalsh expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [&](const any_tensor& m) { return eigvalsh(m, upper); });
    }
    if (is_single(v)) {
        return narrowed(eigvalsh(widened(v), upper));
    }
    if (is_complex(v)) {
        return do_eigvalsh<std::complex<double>>(std::get<cplx_tensor>(v), upper);
    }
    return do_eigvalsh<double>(std::get<real_tensor>(v), upper);
}

namespace {

    // eigh of every matrix of the double or complex stack `t`, its values and vectors written into the two outputs
    // (for_batch): no tensor per matrix, the one-matrix path's calls and bits.
    template <typename T>
    std::tuple<any_tensor, any_tensor> eigh_stack(const numerical::tensor<T>& t, numerical::eigen_method method, bool upper)
    {
        using decomposition_type = std::conditional_t<std::is_same_v<T, double>, numerical::symmetric_eigen, numerical::hermitian_eigen>;
        const stack_shape         s {stack_of(any_tensor {t})};
        const std::size_t         n {s.rows};
        numerical::buffer<double> values(s.count * n);
        numerical::buffer<T>      vectors(s.count * n * n);
        for_batch(s.count, std::max(s.rows, s.cols), t.data().size(), [&](std::size_t i) {
            decomposition_type decomposition {lower_read(matrix_at(t.data(), s.rows, s.cols, i), upper), method};
            std::copy(decomposition.values().begin(), decomposition.values().end(), values.begin() + static_cast<std::ptrdiff_t>(i * n));
            const auto basis {std::move(decomposition).vectors()};
            std::copy(basis.begin(), basis.end(), vectors.begin() + static_cast<std::ptrdiff_t>(i * n * n));
        });
        return {any_tensor {real_tensor {batch_shape(s.batch, {n}), std::move(values)}},
                any_tensor {numerical::tensor<T> {batch_shape(s.batch, {n, n}), std::move(vectors)}}};
    }
} // namespace

// eigh(a, method, upper) -> (eigenvalues, eigenvectors) for a real symmetric or a complex Hermitian matrix: the values
// are real; its lower triangle is read, or its upper one.
std::tuple<any_tensor, any_tensor> eigh(const any_tensor& v, const std::string& name, bool upper)
{
    using numerical::eigen_method;
    const eigen_method method {method_named<eigen_method>(name, {{"divide_and_conquer", eigen_method::divide_and_conquer},
                                                                 {"tridiagonal_ql", eigen_method::tridiagonal_ql},
                                                                 {"jacobi", eigen_method::jacobi}})};
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("eigh expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (is_single(v)) {
        return narrowed(eigh(widened(v), name, upper));
    }
    if (shape_of(v).size() > 2) {
        return is_complex(v) ? eigh_stack(std::get<cplx_tensor>(v), method, upper) : eigh_stack(std::get<real_tensor>(v), method, upper);
    }
    if (is_complex(v)) {
        const cplx_tensor&               a = std::get<cplx_tensor>(v);
        const sb::scoped_gil_release     unlocked {releases_gil_cubic(a.data().size())};
        numerical::hermitian_eigen   decomposition {lower_read(a, upper), method};
        return {any_tensor {real_tensor {decomposition.values()}},
                any_tensor {cplx_tensor {std::move(decomposition).vectors()}}};
    }
    const real_tensor&               a = std::get<real_tensor>(v);
    const sb::scoped_gil_release     unlocked {releases_gil_cubic(a.data().size())};
    numerical::symmetric_eigen   decomposition {lower_read(a, upper), method};
    return {any_tensor {real_tensor {decomposition.values()}},
            any_tensor {real_tensor {std::move(decomposition).vectors()}}};
}

std::tuple<any_tensor, any_tensor> schur(const any_tensor& v, bool complex_output)
{
    if (shape_of(v).size() != 2 || shape_of(v)[0] != shape_of(v)[1]) {
        throw std::runtime_error("schur: expected a square matrix");
    }
    if (is_single(v)) {
        return narrowed(schur(widened(v), complex_output));
    }
    const sb::scoped_gil_release unlocked {releases_gil_cubic(shape_of(v)[0] * shape_of(v)[0])};
    if (is_complex(v)) {
        return do_schur(matrix_from(std::get<cplx_tensor>(v)));
    }
    if (complex_output) {
        return do_schur(matrix_from(converted<std::complex<double>>(std::get<real_tensor>(v))));
    }
    return do_schur(matrix_from(std::get<real_tensor>(v)));
}

namespace {

    // scipy.linalg.schur's sort: kind 0 to 3 "lhp" (Re z < 0), "rhp" (Re z ≥ 0), "iuc" (|z| ≤ 1), "ouc" (|z| > 1), 4
    // the Python callable `callback`, called with (Re z, Im z) for a real Schur form and with z for a complex one, as
    // SciPy calls it; its truth taken, its exception passed on.
    struct schur_selection
    {
        int       kind;
        PyObject* callback;
        bool      real_form;

        bool operator()(const std::complex<double>& z) const
        {
            switch (kind) {
            case 0: return z.real() < 0.0;
            case 1: return z.real() >= 0.0;
            case 2: return std::abs(z) <= 1.0;
            case 3: return std::abs(z) > 1.0;
            default: break;
            }
            PyObject* const answer {real_form ? PyObject_CallFunction(callback, "dd", z.real(), z.imag())
                                              : PyObject_CallFunction(callback, "D", &z)}; // new ref
            if (answer == nullptr) {
                throw sb::error_already_set {};
            }
            const int truth {PyObject_IsTrue(answer)};
            Py_DECREF(answer);
            if (truth < 0) {
                throw sb::error_already_set {};
            }
            return truth != 0;
        }
    };

    template <typename T>
    std::tuple<any_tensor, any_tensor, std::size_t> sorted_schur(numerical::matrix<T>&& a, schur_selection select)
    {
        std::optional<numerical::schur_decomposition<T>> s;
        {
            const sb::scoped_gil_release unlocked {releases_gil_cubic(a.rows() * a.rows())};
            s.emplace(std::move(a));
        }
        const std::size_t    sdim {s->sort(select)}; // the GIL held: a callable may be called
        numerical::matrix<T> z {std::move(*s).z()};
        return {any_tensor {numerical::tensor<T> {std::move(*s).t()}}, any_tensor {numerical::tensor<T> {std::move(z)}}, sdim};
    }
} // namespace

std::tuple<any_tensor, any_tensor, std::size_t> schur_sorted(const any_tensor& v, bool complex_output, int kind, PyObject* callback)
{
    if (shape_of(v).size() != 2 || shape_of(v)[0] != shape_of(v)[1]) {
        throw std::runtime_error("schur: expected a square matrix");
    }
    if (is_single(v)) {
        auto [t, z, sdim] {schur_sorted(widened(v), complex_output, kind, callback)};
        return {narrowed(t), narrowed(z), sdim};
    }
    const bool real_form {!is_complex(v) && !complex_output};
    const schur_selection select {kind, callback, real_form};
    if (is_complex(v)) {
        return sorted_schur(matrix_from(std::get<cplx_tensor>(v)), select);
    }
    if (complex_output) {
        return sorted_schur(matrix_from(converted<std::complex<double>>(std::get<real_tensor>(v))), select);
    }
    return sorted_schur(matrix_from(std::get<real_tensor>(v)), select);
}

namespace {

    using pencil_answer = std::tuple<any_tensor, any_tensor, any_tensor, any_tensor, any_tensor, any_tensor>;

    //! The choices `select(alpha, beta)` (a Python callable) makes of the eigenvalues, one truth per entry of what it
    //! returns, called with the GIL held.
    std::vector<bool> selected(PyObject* select, any_tensor alpha, any_tensor beta)
    {
        PyObject* const alpha_object {wrap(std::move(alpha))};
        PyObject* const beta_object {alpha_object == nullptr ? nullptr : wrap(std::move(beta))};
        PyObject* const answer {beta_object == nullptr ? nullptr : PyObject_CallFunctionObjArgs(select, alpha_object, beta_object, nullptr)};
        Py_XDECREF(alpha_object);
        Py_XDECREF(beta_object);
        PyObject* const items {answer == nullptr ? nullptr : PySequence_Fast(answer, "ordqz: sort returns one truth per eigenvalue")};
        Py_XDECREF(answer);
        if (items == nullptr) {
            throw sb::error_already_set {};
        }
        std::vector<bool> chosen(static_cast<std::size_t>(PySequence_Size(items)));
        for (std::size_t k {0}; k < chosen.size(); ++k) {
            PyObject* const item {PySequence_GetItem(items, static_cast<Py_ssize_t>(k))}; // new ref
            const int       truth {item == nullptr ? -1 : PyObject_IsTrue(item)};
            Py_XDECREF(item);
            if (truth < 0) {
                Py_DECREF(items);
                throw sb::error_already_set {};
            }
            chosen[k] = truth != 0;
        }
        Py_DECREF(items);
        return chosen;
    }

    // The generalized Schur form of a complex pencil, reordered where `select` is a callable.
    pencil_answer complex_qz(numerical::matrix<std::complex<double>>&& a, numerical::matrix<std::complex<double>>&& b, PyObject* select)
    {
        using C = std::complex<double>;
        std::optional<numerical::generalized_schur> d;
        {
            const sb::scoped_gil_release unlocked {releases_gil_cubic(a.rows() * a.rows())};
            d.emplace(a, b);
        }
        const auto values_of = [&] {
            numerical::buffer<C> alpha(d->values().size());
            numerical::buffer<C> beta(d->values().size());
            for (std::size_t k {0}; k < alpha.size(); ++k) {
                alpha[k] = d->values()[k].first;
                beta[k]  = d->values()[k].second;
            }
            return std::pair {any_tensor {cplx_tensor {{alpha.size()}, std::move(alpha)}}, any_tensor {cplx_tensor {{beta.size()}, std::move(beta)}}};
        };
        if (select != Py_None) {
            auto [alpha, beta] {values_of()};
            const std::vector<bool> chosen {selected(select, std::move(alpha), std::move(beta))};
            if (chosen.size() != d->values().size()) {
                throw std::runtime_error("ordqz: sort returns one truth per eigenvalue");
            }
            static_cast<void>(d->reorder(chosen));
        }
        auto [alpha, beta] {values_of()};
        return {any_tensor {cplx_tensor {d->s()}}, any_tensor {cplx_tensor {d->t()}}, std::move(alpha), std::move(beta), any_tensor {cplx_tensor {d->q()}},
                any_tensor {cplx_tensor {d->z()}}};
    }

    // The generalized real Schur form, reordered where `select` is a callable; alpha complex, beta real.
    pencil_answer real_qz(numerical::matrix<double>&& a, numerical::matrix<double>&& b, PyObject* select)
    {
        std::optional<numerical::real_generalized_schur> d;
        {
            const sb::scoped_gil_release unlocked {releases_gil_cubic(a.rows() * a.rows())};
            d.emplace(a, b);
        }
        const auto values_of = [&] {
            numerical::buffer<std::complex<double>> alpha(d->values().size());
            numerical::buffer<double>               beta(d->values().size());
            for (std::size_t k {0}; k < alpha.size(); ++k) {
                alpha[k] = {d->values()[k].alphar, d->values()[k].alphai};
                beta[k]  = d->values()[k].beta;
            }
            return std::pair {any_tensor {cplx_tensor {{alpha.size()}, std::move(alpha)}}, any_tensor {real_tensor {{beta.size()}, std::move(beta)}}};
        };
        if (select != Py_None) {
            auto [alpha, beta] {values_of()};
            const std::vector<bool> chosen {selected(select, std::move(alpha), std::move(beta))};
            if (chosen.size() != d->values().size()) {
                throw std::runtime_error("ordqz: sort returns one truth per eigenvalue");
            }
            static_cast<void>(d->reorder(chosen));
        }
        auto [alpha, beta] {values_of()};
        return {any_tensor {real_tensor {d->s()}}, any_tensor {real_tensor {d->t()}}, std::move(alpha), std::move(beta), any_tensor {real_tensor {d->q()}},
                any_tensor {real_tensor {d->z()}}};
    }
} // namespace

// qz(a, b, complex_output, select) -> (S, T, alpha, beta, Q, Z), as scipy.linalg.qz and ordqz: the generalized (real)
// Schur form, reordered by `select(alpha, beta)` where it is a callable (None: not reordered); a real pencil's in real
// arithmetic unless complex_output; single precision computed in double and rounded.
std::tuple<any_tensor, any_tensor, any_tensor, any_tensor, any_tensor, any_tensor> qz(const any_tensor& a, const any_tensor& b, bool complex_output, PyObject* select)
{
    const std::vector<std::size_t>& shape {shape_of(a)};
    if (shape.size() != 2 || shape[0] != shape[1] || shape_of(b) != shape || a.index() != b.index()) {
        throw std::runtime_error("qz: Array dimensions must be square and agree");
    }
    if (is_single(a)) {
        return narrowed(qz(widened(a), widened(b), complex_output, select));
    }
    if (is_complex(a)) {
        return complex_qz(matrix_from(std::get<cplx_tensor>(a)), matrix_from(std::get<cplx_tensor>(b)), select);
    }
    if (complex_output) {
        return complex_qz(matrix_from(converted<std::complex<double>>(std::get<real_tensor>(a))), matrix_from(converted<std::complex<double>>(std::get<real_tensor>(b))),
                          select);
    }
    return real_qz(matrix_from(std::get<real_tensor>(a)), matrix_from(std::get<real_tensor>(b)), select);
}

// eigh_pencil(a, b, method, upper) -> (values, vectors): A x = λ B x, B positive definite, the vectors B-orthonormal.
std::tuple<any_tensor, any_tensor> eigh_pencil(const any_tensor& a, const any_tensor& b, const std::string& name, bool upper)
{
    using numerical::eigen_method;
    const eigen_method method {method_named<eigen_method>(name, {{"divide_and_conquer", eigen_method::divide_and_conquer},
                                                                 {"tridiagonal_ql", eigen_method::tridiagonal_ql},
                                                                 {"jacobi", eigen_method::jacobi}})};
    require_pencil(a, b, "eigh");
    if (is_single(a)) {
        return narrowed(eigh_pencil(widened(a), widened(b), name, upper));
    }
    const sb::scoped_gil_release unlocked {releases_gil_cubic(shape_of(a)[0] * shape_of(a)[0])};
    if (is_complex(a)) {
        numerical::hermitian_eigen decomposition {lower_read(std::get<cplx_tensor>(a), upper), lower_read(std::get<cplx_tensor>(b), upper), method};
        return {any_tensor {real_tensor {decomposition.values()}}, any_tensor {cplx_tensor {std::move(decomposition).vectors()}}};
    }
    numerical::symmetric_eigen decomposition {lower_read(std::get<real_tensor>(a), upper), lower_read(std::get<real_tensor>(b), upper), method};
    return {any_tensor {real_tensor {decomposition.values()}}, any_tensor {real_tensor {std::move(decomposition).vectors()}}};
}

// svd(a, method, full) -> (U, singular_values, Vh) for an m×n matrix, as numpy.linalg.svd(a, full_matrices=full).
std::tuple<any_tensor, any_tensor, any_tensor> svd(const any_tensor& v, const std::string& name, bool full)
{
    using numerical::svd_method;
    const svd_method method {method_named<svd_method>(name, {{"divide_and_conquer", svd_method::divide_and_conquer},
                                                             {"bidiagonal_qr", svd_method::bidiagonal_qr},
                                                             {"jacobi", svd_method::jacobi}})};
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("svd expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<std::tuple<any_tensor, any_tensor, any_tensor>>(v, [&](const any_tensor& m) { return svd(m, name, full); });
    }
    if (is_single(v)) {
        return narrowed(svd(widened(v), name, full));
    }
    if (is_complex(v)) {
        return do_svd<std::complex<double>>(std::get<cplx_tensor>(v), method, full);
    }
    return do_svd<double>(std::get<real_tensor>(v), method, full);
}

// _singular_values(a) -> Tensor: the singular values alone, descending, as numpy.linalg.svd(a, compute_uv=False).
any_tensor singular_values(const any_tensor& v)
{
    if (shape_of(v).size() < 2) {
        throw std::runtime_error("svd expects a matrix or a stack of matrices (rank ≥ 2)");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<any_tensor>(v, [&](const any_tensor& m) { return singular_values(m); });
    }
    if (is_single(v)) {
        return narrowed(singular_values(widened(v)));
    }
    if (is_complex(v)) {
        const cplx_tensor&           a = std::get<cplx_tensor>(v);
        const sb::scoped_gil_release unlocked {releases_gil_cubic(a.data().size())};
        return any_tensor {real_tensor {numerical::singular_values(matrix_from(a))}};
    }
    const real_tensor&           a = std::get<real_tensor>(v);
    const sb::scoped_gil_release unlocked {releases_gil_cubic(a.data().size())};
    return any_tensor {real_tensor {numerical::singular_values(matrix_from(a))}};
}

// _svd_hermitian(a) -> (U, S, Vh): svd_hermitian of a matrix or of every matrix of a stack.
std::tuple<any_tensor, any_tensor, any_tensor> svd_hermitian_of(const any_tensor& v)
{
    if (shape_of(v).size() < 2 || shape_of(v).back() != shape_of(v)[shape_of(v).size() - 2]) {
        throw std::runtime_error("svd(hermitian=True) expects a square matrix or a stack of them");
    }
    if (shape_of(v).size() > 2) {
        return over_stack<std::tuple<any_tensor, any_tensor, any_tensor>>(v, svd_hermitian_of);
    }
    return svd_hermitian(v);
}

// _eig(a) -> (w, v), as numpy.linalg.eig.
std::tuple<any_tensor, any_tensor> eig(const any_tensor& v)
{
    return eig_of(v, true);
}

// _eigvals(a) -> w, as numpy.linalg.eigvals.
any_tensor eigvals(const any_tensor& v)
{
    return std::get<0>(eig_of(v, false));
}

// _eig_left(a) -> (w, vl, vr): the eigenvalues with the left and right eigenvectors of a square matrix, as
// scipy.linalg.eig(a, left=True): complex, in the input's precision.
std::tuple<any_tensor, any_tensor, any_tensor> eig_left(const any_tensor& v)
{
    if (shape_of(v).size() != 2 || shape_of(v)[0] != shape_of(v)[1]) {
        throw std::runtime_error("eig expects a square matrix");
    }
    if (is_single(v)) {
        auto [w, vl, vr] {eig_left(widened(v))};
        return {narrowed(std::move(w)), narrowed(std::move(vl)), narrowed(std::move(vr))};
    }
    const auto decomposed = [](const auto& a) -> std::tuple<any_tensor, any_tensor, any_tensor> {
        using T = element_of_t<std::decay_t<decltype(a)>>;
        const sb::scoped_gil_release      unlocked {releases_gil_cubic(a.data().size())};
        numerical::eigen_decomposition<T> d {matrix_from(a), true, true};
        return {any_tensor {cplx_tensor {d.values()}}, any_tensor {cplx_tensor {d.left_vectors()}}, any_tensor {cplx_tensor {d.vectors()}}};
    };
    return is_complex(v) ? decomposed(std::get<cplx_tensor>(v)) : decomposed(std::get<real_tensor>(v));
}

// _geneig(a, b, left, right) -> (alpha, beta, vl, vr): the generalized eigenvalues (α, β) of the pencil of two square
// matrices of one order by the complex QZ (numerical::generalized_schur) and, as asked, its left and right
// eigenvectors (numerical::generalized_eigenvectors, unit 2-norm), as scipy.linalg.eig(a, b): complex, in the input's
// precision; a vector not asked for 0 × 0.
std::tuple<any_tensor, any_tensor, any_tensor, any_tensor> geneig(const any_tensor& a, const any_tensor& b, bool left, bool right)
{
    if (shape_of(a).size() != 2 || shape_of(a)[0] != shape_of(a)[1] || shape_of(b) != shape_of(a)) {
        throw std::runtime_error("eig: a and b must be square matrices of one order");
    }
    if (is_single(a) && is_single(b)) {
        auto [alpha, beta, vl, vr] {geneig(widened(a), widened(b), left, right)};
        return {narrowed(std::move(alpha)), narrowed(std::move(beta)), narrowed(std::move(vl)), narrowed(std::move(vr))};
    }
    using C = std::complex<double>;
    const auto as_complex = [](const any_tensor& x) {
        return is_complex(x) ? matrix_from(std::get<cplx_tensor>(is_single(x) ? widened(x) : x))
                             : numerical::detail::as_complex(matrix_from(std::get<real_tensor>(is_single(x) ? widened(x) : x)));
    };
    const numerical::matrix<C>        ca {as_complex(a)};
    const numerical::matrix<C>        cb {as_complex(b)};
    const sb::scoped_gil_release      unlocked {releases_gil_cubic(ca.rows() * ca.rows())};
    const numerical::generalized_schur d {ca, cb};
    const std::size_t                 n {ca.rows()};
    numerical::buffer<C>              alpha(n);
    numerical::buffer<C>              beta(n);
    for (std::size_t k {0}; k < n; ++k) {
        alpha[k] = d.values()[k].first;
        beta[k]  = d.values()[k].second;
    }
    const auto vectors = [&](bool wanted, bool of_left) {
        return wanted ? any_tensor {cplx_tensor {numerical::generalized_eigenvectors(d, of_left)}} : any_tensor {cplx_tensor {{0, 0}, numerical::buffer<C> {}}};
    };
    return {any_tensor {cplx_tensor {{n}, std::move(alpha)}}, any_tensor {cplx_tensor {{n}, std::move(beta)}}, vectors(left, true), vectors(right, false)};
}

} // namespace numerical_binding
