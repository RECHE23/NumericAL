// LU factorisation and what it powers: reusing one factorisation for several
// right-hand sides, the determinant (incl. a singular matrix), the inverse
// (checked by A·A⁻¹ ≈ I), and the rejections (singular, non-square).
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
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

TEST(qr_rejects_wide_and_ill_sized_and_solves_only_full_rank)
{
  const matrix<double> wide      {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};                    // 2×3, rows < columns
  EXPECT_THROWS(qr(wide), numerical::dimension_error);
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

TEST(symmetric_eigen_of_a_diagonal_matrix)
{
  const matrix<double>  a {{5.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 8.0}};
  const symmetric_eigen e {a};
  EXPECT(approx_equal(e.values(), vector<double> {2.0, 5.0, 8.0})); // ascending
  EXPECT(approx_equal(reconstruct(e), a));
}

TEST(symmetric_eigen_of_a_dense_matrix)
{
  const matrix<double>  a {{2.0, 1.0}, {1.0, 2.0}};
  const symmetric_eigen e {a};
  EXPECT(approx_equal(e.values(), vector<double> {1.0, 3.0}));
  EXPECT(approx_equal(reconstruct(e), a));
}

TEST(symmetric_eigen_handles_negative_rotations_and_zero_off_diagonals)
{
  // (0,1) gives a negative rotation angle; (0,2) and (1,2) are structurally zero.
  const matrix<double>  a {{5.0, 2.0, 0.0}, {2.0, 1.0, 0.0}, {0.0, 0.0, 3.0}};
  const symmetric_eigen e {a};
  EXPECT(approx_equal(reconstruct(e), a));
}

// eigh reads the lower triangle, as numpy.linalg.eigh does by default: an upper triangle that does not
// mirror it changes nothing. Its sweeps are bounded: one sweep does not diagonalise a dense 3 × 3.
TEST(symmetric_eigen_reads_the_lower_triangle_and_bounds_its_sweeps)
{
  const matrix<double>  lower_only {{2.0, 99.0, -7.0}, {1.0, 3.0, 50.0}, {0.5, -1.0, 4.0}};
  const matrix<double>  mirrored   {{2.0, 1.0, 0.5}, {1.0, 3.0, -1.0}, {0.5, -1.0, 4.0}};
  const symmetric_eigen e          {lower_only};
  EXPECT(approx_equal(e.values(), symmetric_eigen {mirrored}.values()));
  EXPECT(approx_equal(reconstruct(e), mirrored));
  EXPECT_THROWS(symmetric_eigen(mirrored, 1), numerical::convergence_error);
}

TEST(symmetric_eigen_rejects_non_square)
{
  const matrix<double> non_square {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  EXPECT_THROWS(symmetric_eigen {non_square}, numerical::dimension_error);
}

TEST(svd_reconstructs_with_orthonormal_factors)
{
  const matrix<double>    a        {{1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}}; // 3×2, full column rank
  const svd_decomposition s        {a};
  EXPECT(approx_equal(s.vh(), transpose(s.v())));                        // Vᴴ is Vᵀ for a real A
  const matrix<double>    identity {{1.0, 0.0}, {0.0, 1.0}};
  EXPECT(approx_equal(reconstruct(s), a));                               // U Σ Vᵀ == A
  EXPECT(approx_equal(transpose(s.v()) * s.v(), identity));              // Vᵀ V == I
  EXPECT(approx_equal(transpose(s.u()) * s.u(), identity));              // Uᵀ U == I (full rank)
  EXPECT(s.singular_values()[0] >= s.singular_values()[1]);              // descending
}

TEST(svd_handles_a_rank_deficient_matrix)
{
  const matrix<double>    a {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}}; // column 2 = 2 × column 1, rank 1
  const svd_decomposition s {a};
  EXPECT(approx_equal(reconstruct(s), a));                        // reconstruction holds despite a zero σ
  EXPECT(std::abs(s.singular_values()[1]) < 1e-9);                // the second singular value is ~0
  EXPECT(orthonormality_error(s.u()) < 1e-14);                    // and U is still orthonormal
  // An exactly zero σ leaves no column of AV to normalise: U is completed to an orthonormal basis.
  const svd_decomposition z {matrix<double> {{1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
  EXPECT(approx_equal(z.singular_values(), vector<double> {1.0, 0.0, 0.0}));
  EXPECT(orthonormality_error(z.u()) < 1e-14);
  EXPECT(orthonormality_error(z.v()) < 1e-14);
}

// One-sided Jacobi on R gives each singular value to high relative accuracy, the small ones too: a
// diagonal with σ from 1 down to 1e-12, its rows and columns permuted, comes back exactly. A
// decomposition of AᵀA squared the condition and returned 9.7e-9 and 0 for the last two.
TEST(svd_keeps_small_singular_values)
{
  const matrix<double>      a    {{0.0, 0.0, 1e-8, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1e-12}, {0.0, 1e-4, 0.0, 0.0}};
  const svd_decomposition   s    {a};
  const std::vector<double> want {1.0, 1e-4, 1e-8, 1e-12};
  for (std::size_t i {0}; i < want.size(); ++i) {
    EXPECT(std::abs((s.singular_values()[i] / want[i]) - 1.0) < 1e-14);
  }
  const svd_decomposition h {hilbert(12)};
  EXPECT(orthonormality_error(h.u()) < 1e-13);
  EXPECT(orthonormality_error(h.v()) < 1e-13);
  EXPECT(approx_equal(reconstruct(h), hilbert(12), 1e-14));
}

// A wide or complex matrix decomposes too: U is m × k and V n × k with k = min(m, n), and A = U Σ Vᴴ.
TEST(svd_of_a_wide_and_of_a_complex_matrix)
{
  const matrix<double>    wide {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  const svd_decomposition w    {wide};
  EXPECT(w.u().rows() == 2 && w.u().cols() == 2 && w.v().rows() == 3 && w.v().cols() == 2);
  EXPECT(approx_equal(reconstruct(w), wide));
  using c = std::complex<double>;
  const matrix<c>         z {{c {1.0, 1.0}, c {2.0, 0.0}, c {0.0, 1.0}}, {c {0.0, -1.0}, c {1.0, 0.0}, c {3.0, 2.0}},
    {c {2.0, 0.0}, c {0.0, -2.0}, c {1.0, 1.0}}, {c {1.0, 0.0}, c {1.0, 0.0}, c {1.0, 0.0}}};
  const svd_decomposition s {z};
  EXPECT(approx_equal(reconstruct(s), z, 1e-13));
  EXPECT(orthonormality_error(s.u()) < 1e-14);
  EXPECT(orthonormality_error(s.v()) < 1e-14);
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

// lstsq is NumPy's: the least-squares solution of a full-rank tall system, accurate on a Vandermonde
// fit (modified Gram–Schmidt left a coefficient error of 0.6 at degree 11), and the minimum-norm one
// of a rank-deficient or wide system.
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
  const matrix<double>    a {rows, cols, data};
  const svd_decomposition s {a};
  EXPECT(approx_equal(reconstruct(s), a, 1e-7));
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
