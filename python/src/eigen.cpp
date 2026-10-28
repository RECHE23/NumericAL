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
    if (shape_of(v).size() > 2) {
        return over_stack<std::tuple<any_tensor, any_tensor>>(v, [&](const any_tensor& m) { return eigh(m, name, upper); });
    }
    if (is_single(v)) {
        return narrowed(eigh(widened(v), name, upper));
    }
    if (is_complex(v)) {
        const cplx_tensor&               a = std::get<cplx_tensor>(v);
        const sb::scoped_gil_release     unlocked {releases_gil(a.data().size())};
        numerical::hermitian_eigen   decomposition {lower_read(a, upper), method};
        return {any_tensor {real_tensor {decomposition.values()}},
                any_tensor {cplx_tensor {std::move(decomposition).vectors()}}};
    }
    const real_tensor&               a = std::get<real_tensor>(v);
    const sb::scoped_gil_release     unlocked {releases_gil(a.data().size())};
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
    const sb::scoped_gil_release unlocked {releases_gil(shape_of(v)[0] * shape_of(v)[0])};
    if (is_complex(v)) {
        return do_schur(matrix_from(std::get<cplx_tensor>(v)));
    }
    if (complex_output) {
        return do_schur(matrix_from(converted<std::complex<double>>(std::get<real_tensor>(v))));
    }
    return do_schur(matrix_from(std::get<real_tensor>(v)));
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
    const sb::scoped_gil_release unlocked {releases_gil(shape_of(a)[0] * shape_of(a)[0])};
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
        const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
        return any_tensor {real_tensor {numerical::singular_values(matrix_from(a))}};
    }
    const real_tensor&           a = std::get<real_tensor>(v);
    const sb::scoped_gil_release unlocked {releases_gil(a.data().size())};
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

} // namespace numerical_binding
