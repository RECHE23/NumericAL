// LU factorisation and what it powers: reusing one factorisation for several
// right-hand sides, the determinant (incl. a singular matrix), the inverse
// (checked by A·A⁻¹ ≈ I), and the rejections (singular, non-square).
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::approx_equal;
  using numerical::cholesky;
  using numerical::cholesky_decomposition;
  using numerical::determinant;
  using numerical::inverse;
  using numerical::lu_decomposition;
  using numerical::matrix;
  using numerical::qr;
  using numerical::solve;
  using numerical::svd_decomposition;
  using numerical::symmetric_eigen;
  using numerical::transpose;
  using numerical::vector;

  //! \brief Builds an `n × n` diagonal matrix from \p entries.
  matrix<double> diagonal_matrix(const vector<double>& entries)
  {
    const std::size_t   n {entries.size()};
    std::vector<double> data(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      data[(i * n) + i] = entries[i];
    }
    return matrix<double> {n, n, data};
  }

  // Reconstruct a symmetric matrix from its eigendecomposition: V Λ Vᵀ.
  matrix<double> reconstruct(const symmetric_eigen& e)
  {
    return e.vectors() * diagonal_matrix(e.values()) * transpose(e.vectors());
  }

  // Reconstruct a matrix from its SVD: U Σ Vᴴ.
  template <typename T>
  matrix<T> reconstruct(const svd_decomposition<T>& s)
  {
    const std::size_t k              {s.singular_values().size()};
    std::vector<T>    sigma(k * k, T {});
    for (std::size_t i {0}; i < k; ++i) {
      sigma[(i * k) + i] = T {s.singular_values()[i]};
    }
    return s.u() * matrix<T> {k, k, sigma} *s.vh();
  }

  //! \brief The largest entry of |Mᴴ M − I|: how far \p m's columns are from orthonormal.
  template <typename T>
  double orthonormality_error(const matrix<T>& m)
  {
    double worst {0.0};
    for (std::size_t i {0}; i < m.cols(); ++i) {
      for (std::size_t j {0}; j < m.cols(); ++j) {
        std::complex<double> dot {};
        for (std::size_t r {0}; r < m.rows(); ++r) {
          dot += std::conj(std::complex<double> {m(r, i)}) * std::complex<double> {m(r, j)};
        }
        worst = std::max(worst, std::abs(dot - std::complex<double> {i == j ? 1.0 : 0.0}));
      }
    }
    return worst;
  }

  //! \brief The `n × n` Hilbert matrix, whose condition number grows like e^{3.5 n}.
  matrix<double> hilbert(std::size_t n)
  {
    std::vector<double> entries(n * n);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        entries[(i * n) + j] = 1.0 / static_cast<double>(i + j + 1);
      }
    }
    return matrix<double> {n, n, entries};
  }
} // namespace

TEST(lu_factorisation_is_reused_for_solve_and_determinant)
{
  const matrix<double>           a  {{2.0, 1.0}, {1.0, 3.0}}; // no row swap needed (det = 5)
  const lu_decomposition<double> lu {a};
  const vector<double>           x1 {lu.solve(vector<double> {3.0, 5.0})};
  const vector<double>           x2 {lu.solve(vector<double> {1.0, 1.0})};
  EXPECT(approx_equal(x1, vector<double> {0.8, 1.4}));
  EXPECT(approx_equal(x2, vector<double> {0.4, 0.2}));
  EXPECT(std::abs(lu.determinant() - 5.0) < 1e-9);
}

TEST(determinant_matches_the_closed_form)
{
  const matrix<double> a        {{1.0, 2.0}, {3.0, 4.0}}; // a row swap ⇒ sign flips; det = -2
  EXPECT(std::abs(determinant(a) - (-2.0)) < 1e-9);
  const matrix<double> singular {{1.0, 2.0}, {2.0, 4.0}};
  EXPECT(std::abs(determinant(singular)) < 1e-12);        // a zero pivot ⇒ determinant 0
}

TEST(inverse_times_the_matrix_is_identity)
{
  const matrix<double> a        {{4.0, 7.0}, {2.0, 6.0}};
  const matrix<double> identity {{1.0, 0.0}, {0.0, 1.0}};
  EXPECT(approx_equal(a * inverse(a), identity));
}

TEST(inverse_of_a_large_matrix)
{
  // Large enough that inverse() solves its columns in parallel; A·A⁻¹ ≈ I checks
  // the threaded result is still correct (and deterministic).
  const std::size_t   n {40};
  std::vector<double> entries(n * n, 0.0);
  std::vector<double> unit(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    unit[(i * n) + i] = 1.0;
    for (std::size_t j {0}; j < n; ++j) {
      entries[(i * n) + j] = (i == j) ? (static_cast<double>(n) + 1.0) : 0.5; // diagonally dominant
    }
  }
  const matrix<double> a        {n, n, entries};
  const matrix<double> identity {n, n, unit};
  EXPECT(approx_equal(a * inverse(a), identity));
}

TEST(factorisation_rejects_singular_and_non_square)
{
  const matrix<double> singular   {{1.0, 2.0}, {2.0, 4.0}};
  EXPECT_THROWS(inverse(singular), numerical::singular_matrix_error);
  const matrix<double> non_square {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  EXPECT_THROWS(determinant(non_square), numerical::dimension_error); // factorisation needs a square
  EXPECT_THROWS(inverse(non_square), numerical::dimension_error);
}

TEST(cholesky_factor_reconstructs_the_matrix)
{
  const matrix<double>                 a    {{4.0, 2.0}, {2.0, 3.0}}; // symmetric, positive-definite
  const cholesky_decomposition<double> chol {a};
  EXPECT(approx_equal(chol.lower() * transpose(chol.lower()), a));    // L Lᵀ == A
}

TEST(cholesky_solves_and_gives_the_determinant)
{
  const matrix<double> a    {{4.0, 2.0}, {2.0, 3.0}};
  const auto           chol {cholesky(a)};
  EXPECT(approx_equal(chol.solve(vector<double> {1.0, 1.0}), vector<double> {0.125, 0.25}));
  EXPECT(std::abs(chol.determinant() - 8.0) < 1e-9);                                     // det = 12 - 4
  EXPECT_THROWS(chol.solve(vector<double> {1.0, 2.0, 3.0}), numerical::dimension_error); // wrong RHS size
}

TEST(cholesky_rejects_non_positive_definite_and_non_square)
{
  const matrix<double> indefinite {{1.0, 2.0}, {2.0, 1.0}}; // det = -3, not positive-definite
  EXPECT_THROWS(cholesky(indefinite), numerical::not_positive_definite_error);
  const matrix<double> non_square {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  EXPECT_THROWS(cholesky(non_square), numerical::dimension_error);
}

TEST(cholesky_is_generic_over_a_complex_hermitian_matrix)
{
  using c = std::complex<double>;
  // [[2, 1-i], [1+i, 3]] is Hermitian and positive-definite (det = 4).
  const matrix<c> a    {{c {2.0, 0.0}, c {1.0, -1.0}}, {c {1.0, 1.0}, c {3.0, 0.0}}};
  const vector<c> b    {c {1.0, 0.0}, c {0.0, 1.0}};
  const auto      chol {cholesky(a)};
  // Validate the complex factor and solve against the trusted LU.
  EXPECT(approx_equal(chol.solve(b), solve(a, b)));
  EXPECT(std::abs(chol.determinant() - determinant(a)) < 1e-9);
}

TEST(qr_factor_is_orthonormal_and_reconstructs_the_matrix)
{
  const matrix<double> a        {{1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}};     // 3×2, full column rank
  const auto           factored {qr(a)};
  const matrix<double> identity {{1.0, 0.0}, {0.0, 1.0}};
  EXPECT(approx_equal(factored.q() * factored.r(), a));                   // Q R == A
  EXPECT(approx_equal(transpose(factored.q()) * factored.q(), identity)); // Qᵀ Q == I
}

TEST(qr_solve_is_the_least_squares_solution)
{
  // Overdetermined system: QR's solution must match the normal equations.
  const matrix<double> a           {{1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}};
  const vector<double> b           {2.0, 3.0, 5.0};
  const vector<double> from_qr     {qr(a).solve(b)};
  const vector<double> from_normal {solve(transpose(a) * a, transpose(a) * b)}; // (AᵀA) x = Aᵀb
  EXPECT(approx_equal(from_qr, from_normal));
}

// A wide matrix factorises as NumPy's reduced qr does — Q m × m, R m × n upper trapezoidal — but its system
// is underdetermined, which solve leaves to lstsq.
TEST(qr_factorises_any_shape_and_solves_only_full_column_rank)
{
  const matrix<double> wide      {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};                    // 2×3, rows < columns
  const auto           short_qr  {qr(wide)};
  EXPECT(short_qr.q().rows() == 2 && short_qr.q().cols() == 2 && short_qr.r().rows() == 2 && short_qr.r().cols() == 3);
  EXPECT(approx_equal(short_qr.q() * short_qr.r(), wide));
  EXPECT(orthonormality_error(short_qr.q()) < 1e-14);
  EXPECT(short_qr.r()(1, 0) == 0.0);
         EXPECT_THROWS(short_qr.solve(vector<double> {1.0, 2.0}), numerical::dimension_error);
  // A rank-deficient matrix factorises (a zero on R's diagonal); its least squares are lstsq's.
  const matrix<double> dependent {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}};                  // column 2 = 2 × column 1
  const auto           factored  {qr(dependent)};
  EXPECT(approx_equal(factored.q() * factored.r(), dependent));
  EXPECT(std::abs(factored.r()(1, 1)) < 1e-12);
         EXPECT_THROWS(factored.solve(vector<double> {1.0, 2.0, 3.0}), numerical::singular_matrix_error);
  const matrix<double> tall      {{1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}};
  EXPECT_THROWS(qr(tall).solve(vector<double> {1.0, 2.0}), numerical::dimension_error); // wrong RHS size
}

TEST(qr_is_generic_over_a_complex_scalar)
{
  using c = std::complex<double>;
  const matrix<c> a        {{c {1.0, 1.0}, c {2.0, 0.0}}, {c {0.0, 1.0}, c {1.0, 0.0}}};
  const vector<c> b        {c {1.0, 0.0}, c {0.0, 1.0}};
  const auto      factored {qr(a)};
  EXPECT(approx_equal(factored.q() * factored.r(), a));     // Q R == A
  EXPECT(approx_equal(factored.solve(b), solve(a, b)));     // square ⇒ matches the LU solve
}

// Householder reflections keep Q orthonormal to rounding however ill-conditioned A is: Hilbert-12
// (condition ~1e16) factorises with |QᵀQ − I| near the machine epsilon, and R's diagonal is real and
// non-negative, for a complex matrix too.
TEST(qr_stays_orthonormal_on_an_ill_conditioned_matrix)
{
  for (const std::size_t n : {std::size_t {10}, std::size_t {12}, std::size_t {14}}) {
    const auto factored {qr(hilbert(n))};
    EXPECT(orthonormality_error(factored.q()) < 1e-14);
    EXPECT(approx_equal(factored.q() * factored.r(), hilbert(n), 1e-14));
  }
  using c = std::complex<double>;
  const matrix<c> a        {{c {0.0, 2.0}, c {1.0, 0.0}}, {c {-1.0, 0.0}, c {0.0, 1.0}}, {c {0.0, 0.0}, c {2.0, -1.0}}};
  const auto      factored {qr(a)};
  for (std::size_t i {0}; i < 2; ++i) {
    EXPECT(factored.r()(i, i).imag() == 0.0 && factored.r()(i, i).real() > 0.0);
  }
  EXPECT(orthonormality_error(factored.q()) < 1e-14);
}

namespace {

  constexpr std::array<numerical::eigen_method, 3> eigen_methods {numerical::eigen_method::divide_and_conquer,
                                                                  numerical::eigen_method::tridiagonal_ql,
                                                                  numerical::eigen_method::jacobi};
  constexpr std::array<numerical::svd_method, 3>   svd_methods   {numerical::svd_method::divide_and_conquer,
                                                                  numerical::svd_method::bidiagonal_qr,
                                                                  numerical::svd_method::jacobi};
}

TEST(symmetric_eigen_of_a_diagonal_matrix)
{
  const matrix<double> a {{5.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 8.0}};
  for (const numerical::eigen_method method : eigen_methods) {
    const symmetric_eigen e {a, method};
    EXPECT(approx_equal(e.values(), vector<double> {2.0, 5.0, 8.0})); // ascending
    EXPECT(approx_equal(reconstruct(e), a));
  }
}

TEST(symmetric_eigen_of_a_dense_matrix)
{
  const matrix<double> a {{2.0, 1.0}, {1.0, 2.0}};
  for (const numerical::eigen_method method : eigen_methods) {
    const symmetric_eigen e {a, method};
    EXPECT(approx_equal(e.values(), vector<double> {1.0, 3.0}));
    EXPECT(approx_equal(reconstruct(e), a));
  }
}

TEST(symmetric_eigen_handles_negative_rotations_and_zero_off_diagonals)
{
  // (0,1) gives a negative rotation angle; (0,2) and (1,2) are structurally zero, a column the tridiagonal
  // reduction leaves unreflected.
  const matrix<double> a {{5.0, 2.0, 0.0}, {2.0, 1.0, 0.0}, {0.0, 0.0, 3.0}};
  for (const numerical::eigen_method method : eigen_methods) {
    const symmetric_eigen e {a, method};
    EXPECT(approx_equal(reconstruct(e), a));
  }
}

// eigh reads the lower triangle, as numpy.linalg.eigh does by default: an upper triangle that does not
// mirror it changes nothing, under either method. Both are bounded: one Jacobi sweep does not diagonalise a
// dense 3 × 3, nor does one QL step its tridiagonal.
TEST(symmetric_eigen_reads_the_lower_triangle_and_bounds_its_iterations)
{
  const matrix<double> lower_only {{2.0, 99.0, -7.0}, {1.0, 3.0, 50.0}, {0.5, -1.0, 4.0}};
  const matrix<double> mirrored   {{2.0, 1.0, 0.5}, {1.0, 3.0, -1.0}, {0.5, -1.0, 4.0}};
  for (const numerical::eigen_method method : eigen_methods) {
    const symmetric_eigen e {lower_only, method};
    EXPECT(approx_equal(e.values(), symmetric_eigen {mirrored, method}.values()));
    EXPECT(approx_equal(reconstruct(e), mirrored));
  }
  std::vector<double> a {2.0, 1.0, 0.5, 1.0, 3.0, -1.0, 0.5, -1.0, 4.0};
  std::vector<double> v {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  EXPECT_THROWS(numerical::detail::cyclic_jacobi(a, v, 3, 1.0, 1), numerical::convergence_error);
  std::vector<double> d {2.0, 3.0, 4.0};
  std::vector<double> e {1.0, -1.0, 0.0};
  EXPECT_THROWS(numerical::detail::tridiagonal_ql(d, e, v, 3, 1), numerical::convergence_error);
}

// The two methods agree, and the default reaches rounding: a 40 × 40 Hilbert matrix (eigenvalues down to
// 1e-18 of the largest) with half its diagonal shifted keeps orthonormal vectors and residuals near machine
// precision.
TEST(symmetric_eigen_methods_agree_and_reach_rounding)
{
  const std::size_t   n {40};
  std::vector<double> entries(n * n);
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j < n; ++j) {
      entries[(i * n) + j] = 1.0 / static_cast<double>(1 + i + j); // Hilbert
    }
  }
  for (std::size_t i {0}; i < n / 2; ++i) {
    entries[(i * n) + i] += 2.0; // half the spectrum shifted away from the other half
  }
  const matrix<double>  a      {n, n, std::move(entries)};
  const symmetric_eigen ql     {a};
  const symmetric_eigen jacobi {a, numerical::eigen_method::jacobi};
  EXPECT(approx_equal(ql.values(), jacobi.values(), 1e-10));
  EXPECT(orthonormality_error(ql.vectors()) < 1e-13);
  EXPECT(approx_equal(reconstruct(ql), a, 1e-13));
}

// A complex Hermitian matrix: real eigenvalues, a unitary V with A = V Λ Vᴴ. [[2, 1−i], [1+i, 3]] has
// eigenvalues 1 and 4. The lower triangle is read and the diagonal's imaginary part ignored, as NumPy reads
// it; a real symmetric matrix gives symmetric_eigen's values.
TEST(hermitian_eigen_reconstructs_with_a_unitary_basis)
{
  using c = std::complex<double>;
  const numerical::hermitian_eigen two {matrix<c> {{c {2.0, 0.0}, c {1.0, -1.0}}, {c {1.0, 1.0}, c {3.0, 0.0}}}};
  EXPECT(approx_equal(two.values(), vector<double> {1.0, 4.0}, 1e-14));
  const std::size_t n                  {30};
  std::vector<c>    entries(n * n);
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j <= i; ++j) {
      const c z {std::sin(1.0 + static_cast<double>((i * n) + j)), i == j ? 0.0 : std::cos(static_cast<double>(i + (3 * j)))};
      entries[(i * n) + j] = z;
      entries[(j * n) + i] = std::conj(z);
    }
  }
  const matrix<c>                  a {n, n, entries};
  const numerical::hermitian_eigen e {a};
  const matrix<c>&                 v {e.vectors()};
  EXPECT(orthonormality_error(v) < 1e-13);
  double residual                    {0.0};
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j < n; ++j) {
      c sum {};
      for (std::size_t k {0}; k < n; ++k) {
        sum += v(i, k) * e.values()[k] * std::conj(v(j, k));
      }
      residual = std::max(residual, std::abs(sum - a(i, j)));
    }
  }
  EXPECT(residual < 1e-13);
  for (std::size_t k {1}; k < n; ++k) {
    EXPECT(e.values()[k - 1] <= e.values()[k]);
  }
  // The upper triangle and the diagonal's imaginary part are not read.
  const numerical::hermitian_eigen lower       {matrix<c> {{c {2.0, 7.0}, c {99.0, 99.0}}, {c {1.0, 1.0}, c {3.0, -5.0}}}};
  EXPECT(approx_equal(lower.values(), two.values(), 1e-14));
  // Of order 3, so a reflection updates the diagonal: an imaginary part left there would move the values.
  const matrix<c> clean {{c {2.0}, c {1.0, -1.0}, c {0.5, 2.0}}, {c {1.0, 1.0}, c {3.0}, c {-1.0, 0.5}},
    {c {0.5, -2.0}, c {-1.0, -0.5}, c {4.0}}};
  const matrix<c> noisy {{c {2.0, 3.0}, c {1.0, -1.0}, c {0.5, 2.0}}, {c {1.0, 1.0}, c {3.0, -4.0}, c {-1.0, 0.5}},
    {c {0.5, -2.0}, c {-1.0, -0.5}, c {4.0, 5.0}}};
  EXPECT(approx_equal(numerical::hermitian_eigen {noisy}.values(), numerical::hermitian_eigen {clean}.values(), 1e-13));
  const matrix<double>             real_matrix {{2.0, 1.0, 0.5}, {1.0, 3.0, -1.0}, {0.5, -1.0, 4.0}};
  const numerical::hermitian_eigen as_complex  {matrix<c> {{c {2.0}, c {1.0}, c {0.5}}, {c {1.0}, c {3.0}, c {-1.0}}, {c {0.5}, c {-1.0}, c {4.0}}}};
  EXPECT(approx_equal(as_complex.values(), symmetric_eigen {real_matrix}.values(), 1e-13));
  EXPECT_THROWS(numerical::hermitian_eigen {(matrix<c> {{c {1.0}, c {2.0}}})}, numerical::dimension_error);
  EXPECT((numerical::hermitian_eigen {matrix<c> {0, 0, {}}}.values().size() == 0));
  EXPECT((numerical::hermitian_eigen {matrix<c> {{c {0.0}, c {0.0}}, {c {0.0}, c {0.0}}}}.values()[1] == 0.0));
  EXPECT_THROWS(numerical::hermitian_eigen {(matrix<c> {{c {std::numeric_limits<double>::quiet_NaN()}}})}, numerical::non_finite_error);
}

TEST(symmetric_eigen_rejects_non_square)
{
  const matrix<double> non_square {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  EXPECT_THROWS(symmetric_eigen {non_square}, numerical::dimension_error);
}

TEST(svd_reconstructs_with_orthonormal_factors)
{
  const matrix<double> a        {{1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}}; // 3×2, full column rank
  const matrix<double> identity {{1.0, 0.0}, {0.0, 1.0}};
  for (const numerical::svd_method method : svd_methods) {
    const svd_decomposition s {a, method};
    EXPECT(approx_equal(s.vh(), transpose(s.v())));           // Vᴴ is Vᵀ for a real A
    EXPECT(approx_equal(reconstruct(s), a));                  // U Σ Vᵀ == A
    EXPECT(approx_equal(transpose(s.v()) * s.v(), identity)); // Vᵀ V == I
    EXPECT(approx_equal(transpose(s.u()) * s.u(), identity)); // Uᵀ U == I (full rank)
    EXPECT(s.singular_values()[0] >= s.singular_values()[1]); // descending
  }
}

TEST(svd_handles_a_rank_deficient_matrix)
{
  const matrix<double> a   {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}}; // column 2 = 2 × column 1, rank 1
  const matrix<double> one {{1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  for (const numerical::svd_method method : svd_methods) {
    const svd_decomposition s {a, method};
    EXPECT(approx_equal(reconstruct(s), a));         // reconstruction holds despite a zero σ
    EXPECT(std::abs(s.singular_values()[1]) < 1e-9); // the second singular value is ~0
    EXPECT(orthonormality_error(s.u()) < 1e-14);     // and U is still orthonormal
    // An exactly zero σ leaves no column of AV to normalise: U is completed to an orthonormal basis.
    const svd_decomposition z {one, method};
    EXPECT(approx_equal(z.singular_values(), vector<double> {1.0, 0.0, 0.0}));
    EXPECT(orthonormality_error(z.u()) < 1e-14);
    EXPECT(orthonormality_error(z.v()) < 1e-14);
  }
}

// Neither method squares the condition, as a decomposition of AᵀA did (it returned 9.7e-9 and 0 for the
// last two): a diagonal with σ from 1 down to 1e-12, its rows and columns permuted, comes back exactly.
TEST(svd_keeps_small_singular_values)
{
  const matrix<double>      a    {{0.0, 0.0, 1e-8, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1e-12}, {0.0, 1e-4, 0.0, 0.0}};
  const std::vector<double> want {1.0, 1e-4, 1e-8, 1e-12};
  for (const numerical::svd_method method : svd_methods) {
    const svd_decomposition s {a, method};
    for (std::size_t i {0}; i < want.size(); ++i) {
      EXPECT(std::abs((s.singular_values()[i] / want[i]) - 1.0) < 1e-14);
    }
    const svd_decomposition h {hilbert(12), method};
    EXPECT(orthonormality_error(h.u()) < 1e-13);
    EXPECT(orthonormality_error(h.v()) < 1e-13);
    EXPECT(approx_equal(reconstruct(h), hilbert(12), 1e-14));
  }
}

// A wide or complex matrix decomposes too: U is m × k and V n × k with k = min(m, n), and A = U Σ Vᴴ.
TEST(svd_of_a_wide_and_of_a_complex_matrix)
{
  const matrix<double> wide {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  using c = std::complex<double>;
  const matrix<c> z         {{c {1.0, 1.0}, c {2.0, 0.0}, c {0.0, 1.0}}, {c {0.0, -1.0}, c {1.0, 0.0}, c {3.0, 2.0}},
    {c {2.0, 0.0}, c {0.0, -2.0}, c {1.0, 1.0}}, {c {1.0, 0.0}, c {1.0, 0.0}, c {1.0, 0.0}}};
  for (const numerical::svd_method method : svd_methods) {
    const svd_decomposition w {wide, method};
    EXPECT(w.u().rows() == 2 && w.u().cols() == 2 && w.v().rows() == 3 && w.v().cols() == 2);
    EXPECT(approx_equal(reconstruct(w), wide));
    const svd_decomposition s {z, method};
    EXPECT(approx_equal(reconstruct(s), z, 1e-13));
    EXPECT(orthonormality_error(s.u()) < 1e-14);
    EXPECT(orthonormality_error(s.v()) < 1e-14);
  }
}

namespace {

  //! \brief Runs the bidiagonal QR on (d, e) of order n and checks B = U Σ Vᵀ, Σ descending and non-negative.
  bool bidiagonal_qr_reconstructs(std::vector<double> d,
                                  std::vector<double> e)
  {
    const std::size_t   n {d.size()};
    std::vector<double> b(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      b[(i * n) + i] = d[i];
      if (i + 1 < n) {
        b[(i * n) + i + 1] = e[i];
      }
    }
    std::vector<double> ut(n * n, 0.0);
    std::vector<double> vt(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      ut[(i * n) + i] = 1.0;
      vt[(i * n) + i] = 1.0;
    }
    numerical::detail::bidiagonal_qr(d, e, ut, vt, n, 30 * n);
    bool ok {true};
    for (std::size_t i {0}; i < n; ++i) {
      ok = ok && d[i] >= 0.0 && (i == 0 || d[i - 1] >= d[i]);
      for (std::size_t j {0}; j < n; ++j) {
        double sum {0.0};
        for (std::size_t k {0}; k < n; ++k) {
          sum += ut[(k * n) + i] * d[k] * vt[(k * n) + j];
        }
        ok = ok && std::abs(sum - b[(i * n) + j]) < 1e-14;
      }
    }
    return ok;
  }
} // namespace

// The shift is what makes the iteration converge in about two steps per singular value: twenty bidiagonals
// of order 20 take 840 steps in all. A shift half as large takes 999, none 1070; the bound of 900 keeps the
// shift honest, which the answers alone cannot (every shift converges to the same values).
TEST(bidiagonal_qr_converges_in_about_two_steps_per_value)
{
  const auto least_steps = [](const std::vector<double>& d0, const std::vector<double>& e0) {
                             const std::size_t n {d0.size()};
                             for (std::size_t bound {0};; ++bound) {
                               std::vector<double> d {d0};
                               std::vector<double> e {e0};
                               std::vector<double> ut(n * n, 0.0);
                               std::vector<double> vt(n * n, 0.0);
                               try {
                                 numerical::detail::bidiagonal_qr(d, e, ut, vt, n, bound);
                                 return bound;
                               }
                               catch (const numerical::convergence_error&) { // the bound was short: one more step
                               }
                             }
                           };
  std::size_t total {0};
  for (std::size_t t {0}; t < 20; ++t) {
    std::vector<double> d(20);
    std::vector<double> e(20, 0.0);
    for (std::size_t i {0}; i < 20; ++i) {
      d[i] = std::sin(1.0 + static_cast<double>(i) + (31.0 * static_cast<double>(t)));
      if (i + 1 < 20) {
        e[i] = std::cos(2.0 + (3.0 * static_cast<double>(i)) + (17.0 * static_cast<double>(t)));
      }
    }
    total += least_steps(d, e);
  }
  EXPECT(total <= 900U);
}

// The deflations of the bidiagonal iteration each have a bidiagonal that reaches them: a zero at the end of
// the block (chased up from two rows away), a zero inside it (the block splits), a negative entry (its sign
// moves into V), an order to restore, and the two ways the shift's `b` vanishes. A matrix with no column
// has no singular value, by either method.
TEST(bidiagonal_qr_deflates_and_bounds_its_steps)
{
  EXPECT(bidiagonal_qr_reconstructs({-3.0}, {0.0}));
  EXPECT(bidiagonal_qr_reconstructs({1.0, 2.0, 0.0}, {1.0, 1.0, 0.0}));
  EXPECT(bidiagonal_qr_reconstructs({1.0, 0.0, 2.0}, {1.0, 1.0, 0.0}));
  EXPECT(bidiagonal_qr_reconstructs({1.0, 3.0, -2.0, 4.0}, {0.0, 0.0, 0.0, 0.0}));
  {
    // A chain falling by 1e-8 per entry: no link is negligible next to its neighbours, yet at its tail the
    // shift's terms underflow to zero together.
    std::vector<double> chain_d(13);
    std::vector<double> chain_e(13, 0.0);
    for (std::size_t i {0}; i < 13; ++i) {
      chain_d[i] = std::pow(1e-8, static_cast<double>(2 * i));
      if (i + 1 < 13) {
        chain_e[i] = std::pow(1e-8, static_cast<double>((2 * i) + 1));
      }
    }
    EXPECT(bidiagonal_qr_reconstructs(chain_d, chain_e));
  }
  EXPECT(bidiagonal_qr_reconstructs({8.0, 3.0, 5.0}, {1.0, 4.0, 0.0}));          // b = 0 exactly, c is not
  std::vector<double> d {2.0, 3.0, 4.0};
  std::vector<double> e {1.0, -1.0, 0.0};
  std::vector<double> ut(9, 0.0);
  std::vector<double> vt(9, 0.0);
  EXPECT_THROWS(numerical::detail::bidiagonal_qr(d, e, ut, vt, 3, 0), numerical::convergence_error);
  for (const numerical::svd_method method : svd_methods) {
    const svd_decomposition none {matrix<double> {3, 0, {}}, method};
    EXPECT(none.singular_values().size() == 0 && none.u().rows() == 3 && none.u().cols() == 0);
  }
}

// Jacobi's sweeps are bounded: one sweep does not orthogonalise a dense 3 × 3, which is then an error.
TEST(one_sided_jacobi_reports_a_sweep_bound_reached)
{
  std::vector<double> w {4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0};
  std::vector<double> v {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  EXPECT_THROWS(numerical::detail::one_sided_jacobi(w, v, 3, 1), numerical::convergence_error);
}

// Jacobi converges quadratically when each rotation takes the smaller angle: this dense 8 × 8 orthogonalises
// in 9 sweeps under clang and GCC, at -O0 and -O3, and in 11 to 16 with the larger angle.
TEST(one_sided_jacobi_converges_in_few_sweeps)
{
  const std::size_t   n {8};
  std::vector<double> w(n * n);
  std::vector<double> v(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    v[(i * n) + i] = 1.0;
    for (std::size_t j {0}; j < n; ++j) {
      w[(i * n) + j] = std::sin(1.0 + (0.7 * static_cast<double>(i)) + (1.3 * static_cast<double>(j * j)));
    }
  }
  numerical::detail::one_sided_jacobi(w, v, n, 10);
}

// slogdet, matrix_rank and pinv as numpy.linalg gives them: a determinant whose product underflows keeps
// its logarithm, a rank-deficient matrix its rank, and the pseudo-inverse satisfies A A⁺ A = A.
TEST(log_determinant_rank_and_pseudo_inverse)
{
  const std::size_t   n {200};
  std::vector<double> scaled(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    scaled[(i * n) + i] = 1e-2;
  }
  const matrix<double> small {n, n, scaled};
  EXPECT(determinant(small) == 0.0); // 1e-400 underflows
  const auto [sign, logarithm] {lu_decomposition<double> {small}.log_determinant()};
  EXPECT(sign == 1.0 && std::abs(logarithm - (200.0 * std::log(1e-2))) < 1e-9);
  const auto [negative, log_two] {lu_decomposition<double> {matrix<double> {{0.0, 2.0}, {1.0, 0.0}}}.log_determinant()};
  EXPECT(negative == -1.0 && std::abs(log_two - std::log(2.0)) < 1e-15);
  const auto [zero, minus_infinity] {lu_decomposition<double> {matrix<double> {{1.0, 2.0}, {2.0, 4.0}}}.log_determinant()};
  EXPECT(zero == 0.0 && minus_infinity == -std::numeric_limits<double>::infinity());
  using c = std::complex<double>;
  const auto [phase, log_magnitude] {lu_decomposition<c> {matrix<c> {{c {0.0, 2.0}}}}.log_determinant()};
  EXPECT(std::abs(phase - c {0.0, 1.0}) < 1e-15 && std::abs(log_magnitude - std::log(2.0)) < 1e-15);

  const matrix<double> rank_one {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}};
  EXPECT_EQ(numerical::matrix_rank(rank_one), 1U);
  EXPECT_EQ(numerical::matrix_rank(hilbert(8)), 8U);
  EXPECT_EQ(numerical::matrix_rank(matrix<double> {{0.0, 0.0}, {0.0, 0.0}}), 0U);
  for (const matrix<double>& a : {rank_one, hilbert(6), matrix<double> {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}}) {
    const matrix<double> plus {numerical::pinv(a)};
    EXPECT(plus.rows() == a.cols() && plus.cols() == a.rows());
    EXPECT(approx_equal(a * plus * a, a, 1e-9));
    double largest {0.0}; // A⁺ of an ill-conditioned A is large: its rounding scales with it
    for (std::size_t i {0}; i < plus.rows(); ++i) {
      for (std::size_t j {0}; j < plus.cols(); ++j) {
        largest = std::max(largest, std::abs(plus(i, j)));
      }
    }
    EXPECT(approx_equal(plus * a * plus, plus, 1e-9 * largest));
  }
  const matrix<c> z {{c {1.0, 1.0}, c {0.0, 2.0}}, {c {2.0, 0.0}, c {1.0, -1.0}}};
  EXPECT(approx_equal(numerical::pinv(z), numerical::inverse(z)));
}

// lstsq is NumPy's: the least-squares solution of a full-rank tall system, accurate on a Vandermonde
// fit (modified Gram–Schmidt left a coefficient error of 0.6 at degree 11), and the minimum-norm one
// of a rank-deficient or wide system.
// A matrix right-hand side is one system per column, from one factorisation: each column of the answer is the
// vector solve of that column.
TEST(solve_and_lstsq_take_one_right_hand_side_per_column)
{
  const matrix<double> a   {{3.0, 1.0}, {1.0, 2.0}};
  const matrix<double> rhs {{9.0, 1.0}, {8.0, 0.0}};
  const matrix<double> x   {numerical::solve(a, rhs)};
  EXPECT(approx_equal(a * x, rhs));
  EXPECT(approx_equal(vector<double> {x(0, 1), x(1, 1)}, numerical::solve(a, vector<double> {1.0, 0.0})));
  EXPECT_THROWS(numerical::solve(a, matrix<double> {{1.0}, {2.0}, {3.0}}), numerical::dimension_error);
  const matrix<double> wide {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  const matrix<double> y    {numerical::lstsq(wide, rhs)};
  EXPECT(y.rows() == 3 && y.cols() == 2);
  EXPECT(approx_equal(wide * y, rhs));
  EXPECT(approx_equal(vector<double> {y(0, 0), y(1, 0), y(2, 0)}, numerical::lstsq(wide, vector<double> {9.0, 8.0})));
  EXPECT_THROWS(numerical::lstsq(wide, matrix<double> {{1.0}, {2.0}, {3.0}}), numerical::dimension_error);
}

TEST(lstsq_is_the_minimum_norm_least_squares_solution)
{
  const std::size_t   rows   {50};
  const std::size_t   degree {11};
  std::vector<double> entries(rows * (degree + 1));
  std::vector<double> coefficients(degree + 1);
  std::vector<double> b(rows, 0.0);
  for (std::size_t k {0}; k <= degree; ++k) {
    coefficients[k] = static_cast<double>(k % 3) - 1.0 + (0.1 * static_cast<double>(k));
  }
  for (std::size_t i {0}; i < rows; ++i) {
    const double x     {-1.0 + (2.0 * static_cast<double>(i) / static_cast<double>(rows - 1))};
    double       power {1.0};
    for (std::size_t k {0}; k <= degree; ++k) {
      entries[(i * (degree + 1)) + k] = power;
      b[i]                           += coefficients[k] * power;
      power                          *= x;
    }
  }
  const vector<double> fit {numerical::lstsq(matrix<double> {rows, degree + 1, entries}, vector<double> {b})};
  for (std::size_t k {0}; k <= degree; ++k) {
    EXPECT(std::abs(fit[k] - coefficients[k]) < 1e-9);
  }
  // Rank 1: every x with x₀ + 2 x₁ = 1 fits exactly; the smallest is (1, 2) / 5.
  EXPECT(approx_equal(numerical::lstsq(matrix<double> {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}}, vector<double> {1.0, 2.0, 3.0}),
                      vector<double> {0.2, 0.4}));
  // Wide: one equation x₀ + x₁ + x₂ = 3, smallest solution (1, 1, 1).
  EXPECT(approx_equal(numerical::lstsq(matrix<double> {{1.0, 1.0, 1.0}}, vector<double> {3.0}), vector<double> {1.0, 1.0, 1.0}));
  using c = std::complex<double>;
  const matrix<c> square {{c {1.0, 1.0}, c {2.0, 0.0}}, {c {0.0, 1.0}, c {1.0, 0.0}}};
  const vector<c> rhs    {c {1.0, 0.0}, c {0.0, 1.0}};
  EXPECT(approx_equal(numerical::lstsq(square, rhs), solve(square, rhs)));
  EXPECT_THROWS(numerical::lstsq(square, vector<c> {c {1.0, 0.0}}), numerical::dimension_error);
  EXPECT_THROWS(numerical::lstsq(matrix<double> {{1.0}}, vector<double> {1.0, 2.0}), numerical::dimension_error);
}

TEST(svd_of_a_tall_matrix)
{
  // Enough columns that the left singular vectors are computed in parallel;
  // reconstruction confirms the threaded result is correct.
  const std::size_t   rows {40};
  const std::size_t   cols {20};
  std::vector<double> data(rows * cols);
  for (std::size_t i {0}; i < data.size(); ++i) {
    data[i] = std::sin(0.1 * static_cast<double>(i)) + 0.5;
  }
  const matrix<double> a {rows, cols, data};
  for (const numerical::svd_method method : svd_methods) {
    const svd_decomposition s {a, method};
    EXPECT(approx_equal(reconstruct(s), a, 1e-7));
  }
}

TEST(eigen_qr_and_svd_do_not_depend_on_the_matrix_scale)
{
  // A sum of squares overflows for entries near 1e155 and underflows to 0 near 1e-155. Each
  // decomposition works at the scale of the matrix, so a scaled matrix gives the scaled answer:
  // eigenvalues and singular values times the scale, a QR whose R carries it, and no spurious
  // "linearly dependent" refusal.
  for (const double scale : {1e-170, 1.0, 1e170, 1e200}) {
    const matrix<double>  a {{2.0 * scale, 1.0 * scale}, {1.0 * scale, 2.0 * scale}};
    const symmetric_eigen e {a};
    EXPECT(std::abs((e.values()[0] / scale) - 1.0) < 1e-12);
    EXPECT(std::abs((e.values()[1] / scale) - 3.0) < 1e-12);
    const auto qr_of_a                          {numerical::qr(a)};
    EXPECT(std::abs((qr_of_a.r()(0, 0) / scale) - std::sqrt(5.0)) < 1e-12);
           const numerical::svd_decomposition d {matrix<double> {{3.0 * scale, 0.0}, {0.0, 4.0 * scale}, {0.0, 0.0}}};
           EXPECT(std::abs((d.singular_values()[0] / scale) - 4.0) < 1e-12);
    EXPECT(std::abs((d.singular_values()[1] / scale) - 3.0) < 1e-12);
  }
}

TEST(decompositions_of_a_zero_or_non_finite_matrix_keep_their_scale_at_one)
{
  // A zero matrix has no largest entry to divide by, and an infinite one no finite scale: both are
  // decomposed as given. A zero column is dependent on any other, whatever the column norms' scale.
  const matrix<double>  zero {{0.0, 0.0}, {0.0, 0.0}};
  const symmetric_eigen e    {zero};
  EXPECT(approx_equal(e.values(), vector<double> {0.0, 0.0}));
  const svd_decomposition d  {matrix<double> {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}};
  EXPECT(approx_equal(d.singular_values(), vector<double> {0.0, 0.0}));
  EXPECT_THROWS(qr(matrix<double> {{0.0, 1.0}, {0.0, 2.0}}).solve(vector<double> {1.0, 2.0}),
                numerical::singular_matrix_error);
}

// One non-finite policy for every decomposition, checked where it starts: an infinite or NaN entry is a
// non_finite_error. The iterative ones would otherwise stop at once on a NaN off the diagonal and return
// plausible values (eigh of [[1, nan], [nan, 2]] read as 1 and 2).
TEST(every_decomposition_refuses_a_non_finite_entry)
{
  const double inf {std::numeric_limits<double>::infinity()};
  const double nan {std::numeric_limits<double>::quiet_NaN()};
  for (const double bad : {inf, -inf, nan}) {
    const matrix<double> off_diagonal {{1.0, bad}, {bad, 2.0}};
    const matrix<double> on_diagonal  {{bad, 0.0}, {0.0, 1.0}};
    for (const matrix<double>& m : {off_diagonal, on_diagonal}) {
      // Each refusal names the decomposition asked for, not one it builds on (lstsq runs an SVD, the SVD a QR).
      const auto refuses {[](auto&& decompose, std::string_view name) {
                            try {
                              decompose();
                            }
                            catch (const numerical::non_finite_error& error) {
                              return std::string_view {error.what()}.starts_with(name);
                            }
                            return false;
                          }};
      EXPECT(refuses([&] { static_cast<void>(symmetric_eigen {m}); }, "symmetric eigendecomposition"));
      EXPECT(refuses([&] { static_cast<void>(svd_decomposition {m}); }, "SVD"));
      EXPECT(refuses([&] { static_cast<void>(qr(m)); }, "QR"));
      EXPECT(refuses([&] { static_cast<void>(lu_decomposition<double> {m}); }, "LU"));
      EXPECT(refuses([&] { static_cast<void>(cholesky_decomposition<double> {m}); }, "Cholesky"));
      EXPECT(refuses([&] { static_cast<void>(numerical::lstsq(m, vector<double> {1.0, 2.0})); }, "lstsq"));
    }
  }
  // A complex entry is finite only when both of its parts are.
  using complex = std::complex<double>;
  const complex zero {0.0, 0.0};
  const complex one  {1.0, 0.0};
  const complex two  {2.0, 0.0};
  for (const complex bad : {complex {inf, 0.0}, complex {0.0, nan}}) {
    const matrix<complex> m {{one, bad}, {bad, two}};
    EXPECT_THROWS(lu_decomposition<complex> {m}, numerical::non_finite_error);
    EXPECT_THROWS(qr(m), numerical::non_finite_error);
  }
  // A zero column is dependent, complex as real: it factorises, and its solve refuses.
  const matrix<complex> zero_column {{zero, one}, {zero, two}};
  EXPECT_THROWS(qr(zero_column).solve(numerical::vector<complex> {one, two}), numerical::singular_matrix_error);
  bool named                        {false};
  try {
    static_cast<void>(symmetric_eigen {matrix<double> {{1.0, nan}, {nan, 2.0}}});
  }
  catch (const numerical::non_finite_error& error) {
    named = std::string {error.what()}.find("infinite or NaN") != std::string::npos;
  }
  EXPECT(named);
}

namespace {

  // The unblocked algorithm the blocked one regroups: partial pivoting, whole-row swaps, each update
  // applied at its step. Returns the determinant and the solution of `A x = b`.
  template <typename T>
  std::pair<T, std::vector<T>> unblocked_lu_solve(std::size_t    n,
                                                  std::vector<T> a,
                                                  std::vector<T> b)
  {
    T sign {1.0};
    for (std::size_t k {0}; k < n; ++k) {
      std::size_t pivot {k};
      for (std::size_t r {k + 1}; r < n; ++r) {
        if (std::abs(a[(r * n) + k]) > std::abs(a[(pivot * n) + k])) {
          pivot = r;
        }
      }
      if (pivot != k) {
        for (std::size_t c {0}; c < n; ++c) {
          std::swap(a[(k * n) + c], a[(pivot * n) + c]);
        }
        std::swap(b[k], b[pivot]);
        sign = -sign;
      }
      for (std::size_t r {k + 1}; r < n; ++r) {
        const T factor {a[(r * n) + k] / a[(k * n) + k]};
        a[(r * n) + k] = factor;
        for (std::size_t c {k + 1}; c < n; ++c) {
          a[(r * n) + c] -= factor * a[(k * n) + c];
        }
      }
    }
    T determinant {sign};
    for (std::size_t i {0}; i < n; ++i) {
      determinant *= a[(i * n) + i];
    }
    std::vector<T> x(n);
    for (std::size_t i {0}; i < n; ++i) {
      T sum {b[i]};
      for (std::size_t j {0}; j < i; ++j) {
        sum -= a[(i * n) + j] * x[j];
      }
      x[i] = sum;
    }
    for (std::size_t i {n}; i-- > 0;) {
      T sum {x[i]};
      for (std::size_t j {i + 1}; j < n; ++j) {
        sum -= a[(i * n) + j] * x[j];
      }
      x[i] = sum / a[(i * n) + i];
    }
    return {determinant, x};
  }

  // Uniform in [-1, 1) from a fixed linear congruential sequence: well conditioned (a random matrix of
  // order n has a condition number of order n) and not diagonally dominant, so partial pivoting brings rows
  // from far below, across panels.
  inline double next_uniform(std::uint64_t& state)
  {
    state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
    return (static_cast<double>(state >> 11U) / 4503599627370496.0) - 1.0; // 2^52: [0, 2) shifted to [-1, 1)
  }

  template <typename T>
  std::vector<T> pivoting_entries(std::size_t n);

  template <>
  std::vector<double> pivoting_entries<double>(std::size_t n)
  {
    std::uint64_t       state {12345};
    std::vector<double> entries(n * n);
    for (double& x : entries) {
      x = next_uniform(state);
    }
    return entries;
  }

  template <>
  std::vector<std::complex<double>> pivoting_entries<std::complex<double>>(std::size_t n)
  {
    std::uint64_t                     state {54321};
    std::vector<std::complex<double>> entries(n * n);
    for (std::complex<double>& x : entries) {
      const double re {next_uniform(state)};
      x = {re, next_uniform(state)};
    }
    return entries;
  }

  // The blocked factorisation regroups the unblocked one's updates without reordering any: each element
  // receives its updates in ascending order of the pivot, and adding the product of a negated multiplier
  // is subtracting the product. So the two agree to the bit (the test target builds without multiply-add
  // fusion, which a compiler may apply to one loop and not the other).
  template <typename T>
  bool blocked_lu_equals_unblocked(std::size_t n)
  {
    const std::vector<T>   entries            {pivoting_entries<T>(n)};
    const std::vector<T>   rhs(n, T           {1.0});
    const auto             [determinant, solution] {unblocked_lu_solve(n, entries, rhs)};
    const numerical::lu_decomposition<T> lu   {matrix<T> {n, n, entries}};
    const vector<T>                      x    {lu.solve(vector<T> {rhs})};
    bool                                 same {lu.determinant() == determinant};
    for (std::size_t i {0}; i < n; ++i) {
      same = same && x[i] == solution[i];
    }
    return same;
  }
} // namespace

// Past one panel: three panels, the last partial. On the portable path: on a fused one the trailing products fuse
// their multiply-adds and the panels do not, which a later test bounds instead.
TEST(blocked_lu_equals_the_unblocked_algorithm_to_the_bit)
{
  const std::size_t n   {(2 * numerical::detail::lu_block) + 22};
  (void)numerical::limit_gemm_path(numerical::gemm_path::portable);
  EXPECT(blocked_lu_equals_unblocked<double>(n));
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
  EXPECT(blocked_lu_equals_unblocked<std::complex<double>>(numerical::detail::lu_block + 40));
}

// A zero column in a later panel makes the matrix singular there, and the panels after it still run.
TEST(blocked_lu_meets_a_zero_column_in_a_later_panel)
{
  const std::size_t   n {(2 * numerical::detail::lu_block) + 22};
  std::vector<double> entries(n * n, 0.25);
  for (std::size_t i {0}; i < n; ++i) {
    entries[(i * n) + i] = static_cast<double>(n);
  }
  const std::size_t zero {numerical::detail::lu_block + 36};
  for (std::size_t i {0}; i < n; ++i) {
    entries[(i * n) + zero] = 0.0;
  }
  const numerical::lu_decomposition<double> lu {matrix<double> {n, n, entries}};
  EXPECT(lu.determinant() == 0.0);
  const vector<double> b                       {std::vector<double>(n, 1.0)};
  EXPECT_THROWS((void)lu.solve(b), numerical::singular_matrix_error);
}

namespace {

  // The unblocked Cholesky the blocked one regroups, row by row:
  // L[i][j] = (A[i][j] − Σₖ L[i][k]·conj(L[j][k])) / L[j][j], k ascending.
  template <typename T>
  std::vector<T> unblocked_cholesky(std::size_t           n,
                                    const std::vector<T>& a)
  {
    std::vector<T> l(n * n, T {});
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j <= i; ++j) {
        T sum {a[(i * n) + j]};
        for (std::size_t k {0}; k < j; ++k) {
          sum -= l[(i * n) + k] * numerical::detail::conj_of(l[(j * n) + k]);
        }
        l[(i * n) + j] = i == j ? T {std::sqrt(numerical::detail::real_part(sum))} : sum / l[(j * n) + j];
      }
    }
    return l;
  }

  // M Mᴴ + n·I: Hermitian positive definite, with no structure the panels could lean on.
  template <typename T>
  std::vector<T> hermitian_positive_definite(std::size_t n)
  {
    const std::vector<T> m          {pivoting_entries<T>(n)};
    std::vector<T>       a(n * n, T {});
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        T sum {i == j ? T {static_cast<double>(n)} : T {}};
        for (std::size_t k {0}; k < n; ++k) {
          sum += m[(i * n) + k] * numerical::detail::conj_of(m[(j * n) + k]);
        }
        a[(i * n) + j] = sum;
      }
    }
    return a;
  }

  template <typename T>
  bool blocked_cholesky_equals_unblocked(std::size_t n)
  {
    const std::vector<T> a     {hermitian_positive_definite<T>(n)};
    const matrix<T>      lower {numerical::cholesky(matrix<T> {n, n, a}).lower()};
    return lower == matrix<T> {n, n, unblocked_cholesky(n, a)};
  }
} // namespace

// Past one panel: the diagonal blocks, the rows below them by leaves (or, under many rows, as one leaf), and
// the trailing products.
TEST(blocked_cholesky_equals_the_unblocked_algorithm_to_the_bit)
{
  (void)numerical::limit_gemm_path(numerical::gemm_path::portable);
  EXPECT(blocked_cholesky_equals_unblocked<double>((2 * numerical::detail::cholesky_block) + 22));
  EXPECT(blocked_cholesky_equals_unblocked<double>((9 * numerical::detail::cholesky_block) + 5));
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
  EXPECT(blocked_cholesky_equals_unblocked<std::complex<double>>(numerical::detail::cholesky_block + 40));
}

// A diagonal that turns non-positive in a later panel is refused there.
TEST(blocked_cholesky_refuses_a_later_panel_that_is_not_positive_definite)
{
  const std::size_t   n {(2 * numerical::detail::cholesky_block) + 22};
  std::vector<double> entries(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    entries[(i * n) + i] = 2.0;
  }
  entries[((numerical::detail::cholesky_block + 36) * n) + numerical::detail::cholesky_block + 36] = -1.0;
  EXPECT_THROWS((void)numerical::cholesky(matrix<double> {n, n, entries}), numerical::not_positive_definite_error);
}

namespace {

  // The blocked solve for many right-hand sides against the vector solve, column by column, past one block of
  // rows and one chunk of columns: the backward pass orders each row's terms differently, so they agree to
  // rounding, relative to the solution's size.
  template <typename T>
  bool matrix_solve_agrees_with_column_solves(std::size_t n,
                                              std::size_t k)
  {
    const matrix<T>                      a     {n, n, pivoting_entries<T>(n)};
    const std::vector<T>                 rhs   {pivoting_entries<T>(n)};
    const matrix<T>                      b     {n, k, std::vector<T>(rhs.begin(), rhs.begin() + static_cast<std::ptrdiff_t>(n * k))};
    const numerical::lu_decomposition<T> lu    {a};
    const matrix<T>                      x     {lu.solve(b)};
    bool                                 close {true};
    for (std::size_t c {0}; c < k; ++c) {
      std::vector<T> column(n);
      for (std::size_t i {0}; i < n; ++i) {
        column[i] = b(i, c);
      }
      const vector<T> expected {lu.solve(vector<T> {column})};
      double          scale    {0.0};
      for (std::size_t i {0}; i < n; ++i) {
        scale = std::max(scale, std::abs(expected[i]));
      }
      for (std::size_t i {0}; i < n; ++i) {
        close = close && std::abs(x(i, c) - expected[i]) <= 1e-12 * scale;
      }
    }
    return close;
  }
} // namespace

TEST(blocked_matrix_solve_agrees_with_column_solves)
{
  EXPECT(matrix_solve_agrees_with_column_solves<double>((2 * numerical::detail::lu_block) + 22, 70));
  EXPECT(matrix_solve_agrees_with_column_solves<std::complex<double>>(numerical::detail::lu_block + 40, 3));
  const numerical::lu_decomposition<double> singular    {matrix<double> {{1.0, 2.0}, {2.0, 4.0}}};
  const matrix<double>                      two_columns {{1.0, 0.0}, {0.0, 1.0}};
  EXPECT_THROWS((void)singular.solve(two_columns), numerical::singular_matrix_error);
  EXPECT_THROWS((void)singular.solve(matrix<double> {{1.0, 0.0}}), numerical::dimension_error);
}

TEST(blocked_inverse_times_the_matrix_is_the_identity)
{
  const std::size_t    n {(2 * numerical::detail::lu_block) + 22};
  const matrix<double> a {n, n, pivoting_entries<double>(n)};
  std::vector<double>  unit(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    unit[(i * n) + i] = 1.0;
  }
  EXPECT(approx_equal(a * inverse(a), matrix<double> {n, n, unit}, 1e-10));
}

namespace {

  // Past one panel of reflectors: Q orthonormal, Q R = A, R upper trapezoidal with a non-negative diagonal,
  // real to rounding; every entry of Q R within rounding of A's scale.
  template <typename T>
  bool blocked_qr_holds(std::size_t rows,
                        std::size_t cols,
                        std::size_t zero_column)
  {
    std::vector<T> entries {pivoting_entries<T>(std::max(rows, cols))};
    entries.resize(rows * cols);
    for (std::size_t i {0}; i < rows && zero_column < cols; ++i) {
      entries[(i * cols) + zero_column] = T {};
    }
    const matrix<T>                      a        {rows, cols, entries};
    const numerical::qr_decomposition<T> factored {a};
    const matrix<T>                      q        {factored.q()};
    const matrix<T>                      r        {factored.r()};
    bool                                 holds    {orthonormality_error(q) < 1e-13};
    const matrix<T>                      product  {q * r};
    for (std::size_t i {0}; i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        holds = holds && std::abs(product(i, j) - a(i, j)) < 1e-12;
      }
    }
    for (std::size_t i {0}; i < r.rows(); ++i) {
      // Real to rounding: the phase that makes it real is itself rounded.
      holds = holds && std::abs(std::imag(std::complex<double> {r(i, i)})) < 1e-13 && std::real(std::complex<double> {r(i, i)}) >= 0.0;
      for (std::size_t j {0}; j < i && j < r.cols(); ++j) {
        holds = holds && r(i, j) == T {};
      }
    }
    return holds;
  }
} // namespace

TEST(blocked_qr_across_panels_any_shape)
{
  const std::size_t past {(2 * numerical::detail::qr_block) + 13};
  EXPECT(blocked_qr_holds<double>(past + 40, past, past));                       // tall
  EXPECT(blocked_qr_holds<double>(past - 20, past + 30, past + 30));             // wide: k = rows
  EXPECT(blocked_qr_holds<double>(past, past, numerical::detail::qr_block + 5)); // a zero column, later panel
  EXPECT(blocked_qr_holds<std::complex<double>>(past + 7, past, past));
}

// On every path this processor runs (fused trailing products on the AVX and AMX ones) the blocked factorisations solve to
// within a small multiple of the rounding: the bound the bit-identity above gives on the vector path alone.
TEST(blocked_factorisations_solve_accurately_on_every_path)
{
  const std::size_t n {(2 * numerical::detail::lu_block) + 22};
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue;
    }
    std::vector<double> entries(n * n);
    for (std::size_t i {0}; i < n * n; ++i) {
      entries[i] = std::sin(static_cast<double>(i) * 0.37) + (i % (n + 1) == 0 ? static_cast<double>(n) : 0.0);
    }
    const matrix<double> a     {n, n, entries};
    const vector<double> b     {std::vector<double>(n, 1.0)};
    const vector<double> x     {numerical::solve(a, b)};
    const vector<double> r     {a * x};
    double               worst {0.0};
    for (std::size_t i {0}; i < n; ++i) {
      worst = std::max(worst, std::abs(r[i] - 1.0));
    }
    EXPECT(worst < 1e-12);
    const std::vector<double> spd {hermitian_positive_definite<double>(n)};
    const matrix<double>      lower {numerical::cholesky(matrix<double> {n, n, spd}).lower()};
    const matrix<double>      back {lower * numerical::transpose(lower)};
    double                    gap {0.0};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        const double want {spd[(i * n) + j]};
        gap = std::max(gap, std::abs(back(i, j) - want) / std::max(1.0, std::abs(want)));
      }
    }
    EXPECT(gap < 1e-12);
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

namespace {

  // Divide and conquer against implicit QL on one matrix: the same eigenvalues, orthonormal vectors, and the
  // matrix rebuilt, each to a few rounding errors of its order.
  bool divide_and_conquer_agrees(const matrix<double>& a)
  {
    const double          tolerance {1e-12};
    const symmetric_eigen fast      {a, numerical::eigen_method::divide_and_conquer};
    const symmetric_eigen ql        {a, numerical::eigen_method::tridiagonal_ql};
    return approx_equal(fast.values(), ql.values(), tolerance) && orthonormality_error(fast.vectors()) < tolerance
           && approx_equal(reconstruct(fast), a, tolerance);
  }
} // namespace

// Past the order QL handles alone, so the halves merge: a dense matrix; Wilkinson's W⁺₂₀₁, whose eigenvalues
// come in pairs closer than rounding (poles rotated together, then deflated); two blocks with no coupling
// (every weight negligible); and three values repeated many times under a rank-one coupling (equal poles).
TEST(divide_and_conquer_merges_and_deflates)
{
  const std::size_t   n {201};
  std::vector<double> dense(n * n);
  std::vector<double> wilkinson(n * n, 0.0);
  std::vector<double> blocks(n * n, 0.0);
  std::vector<double> repeated(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j < n; ++j) {
      dense[(i * n) + j] = std::sin(static_cast<double>((i + 1) * (j + 1))) + (i == j ? 3.0 : 0.0);
    }
    wilkinson[(i * n) + i] = std::abs(static_cast<double>(i) - 100.0);
    if (i + 1 < n) {
      wilkinson[(i * n) + i + 1]   = 1.0;
      wilkinson[((i + 1) * n) + i] = 1.0;
    }
    blocks[(i * n) + i]   = static_cast<double>(i % 7);
    repeated[(i * n) + i] = static_cast<double>(i % 3);
  }
  for (std::size_t i {0}; i < n; ++i) { // the blocks: [0, 100) and [100, 201), coupled within only
    for (std::size_t j {0}; j < n; ++j) {
      if (i != j && (i < 100) == (j < 100)) {
        blocks[(i * n) + j] = 1.0 / static_cast<double>(2 + i + j);
      }
    }
  }
  for (std::size_t i {0}; i < n; ++i) { // three values, each many times, plus a rank-one coupling of them all
    for (std::size_t j {0}; j < n; ++j) {
      repeated[(i * n) + j] += 1e-3;
    }
  }
  EXPECT(divide_and_conquer_agrees(matrix<double> {n, n, dense}));
  EXPECT(divide_and_conquer_agrees(matrix<double> {n, n, wilkinson}));
  EXPECT(divide_and_conquer_agrees(matrix<double> {n, n, blocks}));
  EXPECT(divide_and_conquer_agrees(matrix<double> {n, n, repeated}));
  using c = std::complex<double>;
  const std::size_t m {70};
  std::vector<c>    entries(m * m);
  for (std::size_t i {0}; i < m; ++i) {
    for (std::size_t j {0}; j <= i; ++j) {
      entries[(i * m) + j] = i == j ? c {static_cast<double>(i % 5), 0.0}
                                    : c {std::cos(static_cast<double>(i + (2 * j))), std::sin(static_cast<double>(i * j))};
      entries[(j * m) + i] = std::conj(entries[(i * m) + j]);
    }
  }
  const matrix<c>                  h       {m, m, entries};
  const numerical::hermitian_eigen e       {h};
  matrix<c>                        rebuilt {m, m, std::vector<c>(m * m)};
  double                           worst   {0.0};
  for (std::size_t i {0}; i < m; ++i) {
    for (std::size_t j {0}; j < m; ++j) {
      c sum {};
      for (std::size_t k {0}; k < m; ++k) {
        sum += e.vectors()(i, k) * e.values()[k] * std::conj(e.vectors()(j, k));
      }
      worst = std::max(worst, std::abs(sum - h(i, j)));
    }
  }
  EXPECT(worst < 1e-12);
}

// Roots of the secular equation against references computed to 80 digits: one 5e-30 past a pole of 8e-13
// (found as its distance from the pole, which the eigenvector needs, not as a sum that rounds it away), and one
// whose fit takes the quadratic's other form.
TEST(the_secular_equation_finds_its_roots_from_the_nearer_pole)
{
  const std::vector<double> d {-8.122808264515302e-13, 6.04530122336367e-12, 5.219248898251512e-06};
  const std::vector<double> z {4.2711462701151146e-10, 0.48318236791802455, 0.8755197309787659};
  std::vector<double>       delta(3);
  (void)numerical::detail::secular_root(d, z, 283.3625584139991, 0, numerical::detail::secular_iterations, delta.data());
  EXPECT(std::abs(delta[0] - -5.3584091596269844e-30) < 1e-14 * 5.4e-30);
  EXPECT(std::abs(delta[1] - 6.8575820498152e-12) < 1e-14 * 6.9e-12);
  const std::vector<double> d6 {-0.8949247533967253, -0.16141637105958684, -0.0008046271488207429,
                                -1.6560228279009447e-08, -6.869457203538978e-09, 0.9963928059764295};
  const std::vector<double> z6 {-0.005944540988224872, -0.589814085218663, -0.016380413260955785,
                                0.45405424328337657, 0.6675689734376589, -0.0014488724105117632};
  std::vector<double>       delta6(6);
  const double              root {numerical::detail::secular_root(d6, z6, 1.0448293818862207, 4, numerical::detail::secular_iterations,
                                                                  delta6.data())};
  EXPECT(std::abs(root - 0.9932959321048296) < 1e-14); // to a few rounding errors of the equation's terms
}

// The secular equation's iterations are bounded: one step does not reach a root.
TEST(the_secular_equation_bounds_its_iterations)
{
  const std::vector<double> d {0.0, 1.0, 2.0};
  const std::vector<double> z {0.6, 0.48, 0.64};
  std::vector<double>       delta(3);
  EXPECT_THROWS((void)numerical::detail::secular_root(d, z, 1.0, 0, 1, delta.data()), numerical::convergence_error);
}

namespace {

  // Divide and conquer against implicit-shift QR on one matrix: the same singular values, orthonormal vectors,
  // and the matrix rebuilt, each to a few rounding errors of its order.
  template <typename T>
  bool svd_divide_and_conquer_agrees(const matrix<T>& a)
  {
    const double                tolerance {1e-12};
    const svd_decomposition<T>  fast      {a, numerical::svd_method::divide_and_conquer};
    const svd_decomposition<T>  qr        {a, numerical::svd_method::bidiagonal_qr};
    return approx_equal(fast.singular_values(), qr.singular_values(), tolerance) && orthonormality_error(fast.u()) < tolerance
           && orthonormality_error(fast.v()) < tolerance && approx_equal(reconstruct(fast), a, tolerance);
  }
} // namespace

// Past the order QR handles alone, so the halves merge (and a half with an extra column is squared by
// rotations first): a dense matrix; one whose bidiagonal has zeros on its diagonal (values near the zero pole,
// rotated into it); repeated singular values under a small coupling (close values rotated together); a
// rank-deficient one (negligible weights); and a complex one.
TEST(svd_divide_and_conquer_merges_and_deflates)
{
  const std::size_t   n {150};
  std::vector<double> dense(n * n);
  std::vector<double> repeated(n * n, 0.0);
  std::vector<double> deficient(n * n);
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j < n; ++j) {
      dense[(i * n) + j]     = std::sin(static_cast<double>((i + 1) * (j + 2)));
      deficient[(i * n) + j] = (std::cos(static_cast<double>(i)) * std::sin(static_cast<double>(j)))
                               + (std::sin(static_cast<double>(2 * i)) * std::cos(static_cast<double>(3 * j)));
      repeated[(i * n) + j]  = (i == j ? static_cast<double>(1 + (i % 3)) : 0.0) + 1e-4;
    }
  }
  EXPECT(svd_divide_and_conquer_agrees(matrix<double> {n, n, dense}));
  EXPECT(svd_divide_and_conquer_agrees(matrix<double> {n, n, repeated}));
  EXPECT(svd_divide_and_conquer_agrees(matrix<double> {n, n, deficient}));
  std::vector<double> d(n);
  std::vector<double> e(n);
  for (std::size_t i {0}; i < n; ++i) {
    d[i] = i % 5 == 2 ? 0.0 : std::cos(static_cast<double>(i));
    e[i] = i + 1 < n ? 1.0 / static_cast<double>(i + 2) : 0.0;
  }
  std::vector<double> sigma;
  std::vector<double> u;
  std::vector<double> v;
  numerical::detail::bidiagonal_divide(d.data(), e.data(), n, 0, sigma, u, v);
  double worst {0.0}; // B v_j − σ_j u_j
  for (std::size_t j {0}; j < n; ++j) {
    for (std::size_t i {0}; i < n; ++i) {
      const double product {(d[i] * v[(i * n) + j]) + (i + 1 < n ? e[i] * v[((i + 1) * n) + j] : 0.0)};
      worst = std::max(worst, std::abs(product - (sigma[j] * u[(i * n) + j])));
    }
  }
  EXPECT(worst < 1e-13);
  EXPECT(std::ranges::is_sorted(sigma));
  // A row with neither entry at a split of a half with an extra column (no weight on either null vector), and a
  // leaf whose extra column and last diagonal entry are both zero (nothing to rotate).
  const std::size_t   r {70};
  std::vector<double> rd(r);
  std::vector<double> re(r);
  for (std::size_t i {0}; i < r; ++i) {
    rd[i] = i == 16 || i == 17 ? 0.0 : std::cos(static_cast<double>(i));
    re[i] = i == 16 || i == 17 || i + 1 == r ? 0.0 : 1.0 / static_cast<double>(i + 2);
  }
  numerical::detail::bidiagonal_divide(rd.data(), re.data(), r, 0, sigma, u, v);
  worst = 0.0;
  for (std::size_t j {0}; j < r; ++j) {
    for (std::size_t i {0}; i < r; ++i) {
      const double product {(rd[i] * v[(i * r) + j]) + (i + 1 < r ? re[i] * v[((i + 1) * r) + j] : 0.0)};
      worst = std::max(worst, std::abs(product - (sigma[j] * u[(i * r) + j])));
    }
  }
  EXPECT(worst < 1e-13);
  EXPECT(orthonormality_error(matrix<double> {r, r, u}) < 1e-13 && orthonormality_error(matrix<double> {r, r, v}) < 1e-13);
  using c = std::complex<double>;
  const std::size_t m {60};
  std::vector<c>    z(m * (m + 7));
  for (std::size_t i {0}; i < z.size(); ++i) {
    z[i] = c {std::sin(static_cast<double>(i)), std::cos(static_cast<double>(3 * i))};
  }
  EXPECT(svd_divide_and_conquer_agrees(matrix<c> {m, m + 7, z}));
  // Complex, already bidiagonal (no reflection needed on either side), and with no columns at all.
  const matrix<c>            bidiagonal {{c {1.0, 1.0}, c {0.0, 2.0}, c {}}, {c {}, c {3.0, 0.0}, c {1.0, -1.0}}, {c {}, c {}, c {0.0, -2.0}}};
  const svd_decomposition<c> two_sided  {bidiagonal};
  EXPECT(approx_equal(reconstruct(two_sided), bidiagonal, 1e-14));
  const svd_decomposition<c> none       {matrix<c> {3, 0, {}}};
  EXPECT(none.singular_values().size() == 0 && none.u().rows() == 3 && none.u().cols() == 0);
}
