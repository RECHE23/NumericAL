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
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
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

// The reduction's spelled-out complex corrections are subtract_products' to the bit on finite parts: the same
// statements, without the operator's rescue of a NaN product, which finite parts never take.
TEST(the_reduction_s_corrections_are_subtract_products)
{
  using c = std::complex<double>;
  std::uniform_real_distribution<double> part {-1.0, 1.0};
  for (const auto& [terms, count] : {std::array<std::size_t, 2> {1, 1}, {4, 7}, {6, 33}, {62, 40}}) {
    std::mt19937      draw {static_cast<std::mt19937::result_type>((terms * 131) + count)};
    const std::size_t ldx  {count + 3};
    std::vector<c>    f(terms);
    std::vector<c>    x(terms * ldx);
    std::vector<c>    y(count);
    for (c& value : f) {
      value = {part(draw), part(draw)};
    }
    for (c& value : x) {
      value = {part(draw), part(draw)};
    }
    for (c& value : y) {
      value = {part(draw), part(draw)};
    }
    std::vector<c> want {y};
    numerical::detail::subtract_products(want.data(), f.data(), x.data(), ldx, terms, count);
    numerical::detail::subtract_corrections(y.data(), f.data(), x.data(), ldx, terms, count);
    EXPECT(y == want);
    std::vector<double> ry(count, 0.5);
    std::vector<double> rwant {ry};
    std::vector<double> rf(terms, -0.25);
    std::vector<double> rx(terms * ldx, 0.75);
    numerical::detail::subtract_products(rwant.data(), rf.data(), rx.data(), ldx, terms, count);
    numerical::detail::subtract_corrections(ry.data(), rf.data(), rx.data(), ldx, terms, count);
    EXPECT(ry == rwant);
  }
}

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

// A singular matrix whose other pivots overflow together: 0, as its documentation says, where their product times the
// zero pivot was ∞·0, a NaN; a finite product kept as it was (its zero's sign the pivots').
TEST(a_singular_determinant_is_zero_where_the_other_pivots_overflow)
{
  const matrix<double> a   {{1e200, 0.0, 0.0}, {0.0, 1e200, 0.0}, {0.0, 0.0, 0.0}};
  const double         det {lu_decomposition<double> {a}.determinant()};
  EXPECT(det == 0.0 && !std::signbit(det));
  const matrix<double> b   {{1e200, 0.0, 0.0}, {0.0, -1e200, 0.0}, {0.0, 0.0, 0.0}};
  EXPECT(std::signbit(lu_decomposition<double> {b}.determinant()));
  using C = std::complex<double>;
  const matrix<C> c           {{C {1e200, 0.0}, C {}, C {}}, {C {}, C {0.0, 1e200}, C {}}, {C {}, C {}, C {}}};
  const C         complex_det {lu_decomposition<C> {c}.determinant()};
  EXPECT(complex_det.real() == 0.0 && complex_det.imag() == 0.0);
}

// A complex pivot under the smallest normal number: divided by, its reciprocal (2¹⁰³⁰, past the largest double) not
// taken, in the factorisation and in both solves; x = (1, 1) exactly.
TEST(a_subnormal_complex_pivot_divides_where_its_reciprocal_would_overflow)
{
  using C = std::complex<double>;
  const C                     p   {std::ldexp(1.0, -1030), 0.0};
  const matrix<C>             a   {{p, C {}}, {p, C {1.0, 0.0}}};
  const lu_decomposition<C>   lu  {a};
  const numerical::vector<C>  x   {lu.solve(numerical::vector<C> {std::vector<C> {p, p + C {1.0, 0.0}}})};
  const C                     one {1.0, 0.0};
  EXPECT(x[0] == one && x[1] == one);
  const matrix<C> many            {lu.solve(matrix<C> {{p, p}, {p + C {1.0, 0.0}, p + C {1.0, 0.0}}})};
  EXPECT(many(0, 0) == one && many(1, 1) == one);
  // A zero pivot: its column left, the determinant 0.
  const lu_decomposition<C> singular {matrix<C> {{one, one}, {one, one}}};
  EXPECT(singular.determinant() == C {});
}

TEST(a_zero_pivot_column_gives_no_terms_not_even_a_signed_zero)
{
  // Column 1 is zero from its diagonal down: its terms, ±0, would turn column 2's −0 into +0. The unblocked
  // factorisation leaves such a column out, so U's last pivot stays −0, and the determinant 1·0·(−0) with it.
  const matrix<double> a   {{1.0, 1.0, 0.0}, {1.0, 1.0, -0.0}, {1.0, 1.0, -0.0}};
  const double         det {lu_decomposition<double> {a}.determinant()};
  EXPECT(det == 0.0);
  EXPECT(std::signbit(det));
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

TEST(large_matrices_are_checked_for_non_finite_entries_on_the_threads)
{
  // Past two chunks of the shared check: a NaN in the triangle read is refused, one above it alone is not read.
  const std::size_t   n {300};
  std::vector<double> entries(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    entries[(i * n) + i] = 2.0;
  }
  std::vector<double> above           {entries};
  above[(5 * n) + 250] = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> below           {entries};
  below[(290 * n) + 3] = std::numeric_limits<double>::infinity();
  const matrix<double> nan_above      {n, n, above};
  const matrix<double> infinite_below {n, n, below};
  const matrix<double> finite         {n, n, entries};
  EXPECT(numerical::cholesky_decomposition<double> {nan_above}.lower()(n - 1, n - 1) == std::sqrt(2.0));
         EXPECT_THROWS(numerical::cholesky_decomposition<double> {infinite_below}, numerical::non_finite_error);
  EXPECT_THROWS(numerical::lu_decomposition<double> {nan_above}, numerical::non_finite_error);
  EXPECT(numerical::lu_decomposition<double> {finite}.determinant() > 0.0);
}

// A NaN or an infinity is found wherever it lies: in each of the lanes a step checks (eight doubles, sixteen floats),
// and in the values past the last whole step; the largest finite value and the smallest subnormal are finite.
template <typename R>
bool finiteness_check_sees_every_position()
{
  bool all {true};
  for (const std::size_t count : {std::size_t {7}, std::size_t {8}, std::size_t {16}, std::size_t {19}, std::size_t {37}}) {
    std::vector<R> values(count, R(1.5));
    values.front() = std::numeric_limits<R>::max();
    values.back()  = -std::numeric_limits<R>::denorm_min();
    all            = all && numerical::detail::all_finite(values.data(), count);
    for (std::size_t at {0}; at < count; ++at) {
      for (const R bad : {std::numeric_limits<R>::quiet_NaN(), -std::numeric_limits<R>::infinity()}) {
        std::vector<R> flawed {values};
        flawed[at] = bad;
        all        = all && !numerical::detail::all_finite(flawed.data(), count);
      }
    }
  }
  return all;
}

TEST(the_finiteness_check_sees_every_position)
{
  EXPECT(finiteness_check_sees_every_position<double>());
  EXPECT(finiteness_check_sees_every_position<float>());
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

namespace {
  //! \brief A positive-definite (Hermitian) matrix of order \p n of \p T, and \p cols right-hand sides, drawn.
  template <typename T>
  std::pair<matrix<T>, matrix<T>> definite_system(std::size_t n,
                                                  std::size_t cols,
                                                  std::size_t seed)
  {
    // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
    std::mt19937_64                  g {seed};
    std::normal_distribution<double> d;
    const auto                       draw = [&]() -> T {
                                              if constexpr (std::is_same_v<T, double>) {
                                                return d(g);
                                              }
                                              else {
                                                return T {d(g), d(g)};
                                              }
                                            };
    std::vector<T> g_entries(n * n);
    std::vector<T> b_entries(n * cols);
    for (T& v : g_entries) {
      v = draw();
    }
    for (T& v : b_entries) {
      v = draw();
    }
    std::vector<T> a(n * n);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        T sum {i == j ? T(static_cast<double>(n)) : T {}};
        for (std::size_t k {0}; k < n; ++k) {
          sum += g_entries[(i * n) + k] * numerical::detail::conj_of(g_entries[(j * n) + k]);
        }
        a[(i * n) + j] = sum;
      }
    }
    return {matrix<T> {n, n, a}, matrix<T> {n, cols, b_entries}};
  }

  //! \brief Whether each column of the Cholesky decomposition's solve of every column at once is its vector solve's,
  //!        to the bit.
  template <typename T>
  bool cholesky_solves_every_column_as_one(std::size_t n,
                                           std::size_t cols)
  {
    const auto [a, b] {definite_system<T>(n, cols, n + cols)};
    const cholesky_decomposition<T> chol {a};
    const matrix<T>                 x    {chol.solve(b)};
    bool                            same {x.rows() == n && x.cols() == cols};
    for (std::size_t c {0}; same && c < cols; ++c) {
      std::vector<T> column(n);
      for (std::size_t i {0}; i < n; ++i) {
        column[i] = b(i, c);
      }
      const vector<T> alone {chol.solve(vector<T> {column})};
      for (std::size_t i {0}; i < n; ++i) {
        same = same && x(i, c) == alone[i];
      }
    }
    return same;
  }
} // namespace

// The Cholesky and QR decompositions solve every column of B at once: Cholesky's each column its vector solve's bits
// (one column; nine; forty, the threads sharing them), real and complex; QR's to rounding; their errors as the vector's.
TEST(cholesky_and_qr_solve_every_right_hand_side_at_once)
{
  for (const std::size_t cols : {std::size_t {1}, std::size_t {9}, std::size_t {40}}) {
    EXPECT(cholesky_solves_every_column_as_one<double>(23, cols));
    EXPECT(cholesky_solves_every_column_as_one<std::complex<double>>(17, cols));
  }
  const auto [a, b] {definite_system<double>(12, 5, 3)};
  EXPECT_THROWS((void)cholesky(a).solve(matrix<double> {3, 2, std::vector<double>(6, 1.0)}), numerical::dimension_error);
  const matrix<double> tall {{1.0, 2.0}, {3.0, 4.0}, {5.0, 7.0}, {1.0, 0.0}};
  const matrix<double> rhs  {{1.0, 0.0, 2.0}, {0.0, 1.0, 1.0}, {2.0, 1.0, 0.0}, {1.0, 1.0, 1.0}};
  const auto           q    {qr(tall)};
  const matrix<double> x    {q.solve(rhs)};
  for (std::size_t c {0}; c < 3; ++c) {
    const vector<double> alone {q.solve(vector<double> {rhs(0, c), rhs(1, c), rhs(2, c), rhs(3, c)})};
    EXPECT(std::abs(x(0, c) - alone[0]) < 1e-12 && std::abs(x(1, c) - alone[1]) < 1e-12);
  }
  using C = std::complex<double>;
  const matrix<C> complex_tall {{C {1.0, 1.0}, C {0.0, 2.0}}, {C {2.0, 0.0}, C {1.0, -1.0}}, {C {0.0, 1.0}, C {3.0, 0.0}}};
  const matrix<C> complex_rhs  {{C {1.0, 0.0}, C {0.0, 1.0}}, {C {2.0, 1.0}, C {1.0, 0.0}}, {C {0.5, 0.0}, C {0.0, -1.0}}};
  const auto      complex_q    {qr(complex_tall)};
  const matrix<C> y            {complex_q.solve(complex_rhs)};
  for (std::size_t c {0}; c < 2; ++c) {
    const vector<C> alone {complex_q.solve(vector<C> {complex_rhs(0, c), complex_rhs(1, c), complex_rhs(2, c)})};
    EXPECT(std::abs(y(0, c) - alone[0]) < 1e-12 && std::abs(y(1, c) - alone[1]) < 1e-12);
  }
  EXPECT_THROWS((void)q.solve(matrix<double> {3, 1, std::vector<double>(3, 1.0)}), numerical::dimension_error);
  EXPECT_THROWS((void)qr(transpose(tall)).solve(matrix<double> {2, 1, std::vector<double>(2, 1.0)}), numerical::dimension_error);
  const matrix<double> deficient {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}};
  EXPECT_THROWS((void)qr(deficient).solve(matrix<double> {3, 1, std::vector<double>(3, 1.0)}), numerical::singular_matrix_error);
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

// A complex Cholesky factor divides by its diagonal, real and positive: each part over it, one rounding each, in the
// factor and in both triangular solves (a complex division rounded twice under libc++).
TEST(complex_cholesky_divides_each_part_by_the_real_diagonal)
{
  using c = std::complex<double>;
  for (int t {1}; t <= 64; ++t) {
    const double    d     {1.0 + (0.37 * t)};
    const c         z     {std::sin(static_cast<double>(t)), std::cos(3.0 * t)};
    const c         below {d * z};
    const matrix<c> a     {{c {d * d, 0.0}, std::conj(below)}, {below, c {std::norm(z) + 1.0, 0.0}}};
    const matrix<c> l     {cholesky(a).lower()};
    const double    pivot {l(0, 0).real()};
    EXPECT(l(1, 0) == (c {below.real() / pivot, below.imag() / pivot}));
    const matrix<c> one   {{c {d * d, 0.0}}};
    const c         x     {cholesky(one).solve(vector<c> {z})[0]};
    const c         y     {z.real() / pivot, z.imag() / pivot};
    EXPECT(x == (c {y.real() / pivot, y.imag() / pivot}));
  }
}

TEST(qr_factor_is_orthonormal_and_reconstructs_the_matrix)
{
  const matrix<double> a        {{1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}};     // 3×2, full column rank
  const auto           factored {qr(a)};
  const matrix<double> identity {{1.0, 0.0}, {0.0, 1.0}};
  EXPECT(approx_equal(factored.q() * factored.r(), a));                   // Q R == A
  EXPECT(approx_equal(transpose(factored.q()) * factored.q(), identity)); // Qᵀ Q == I
}

TEST(qr_and_svd_of_columns_of_the_identity_are_exact)
{
  // e₁, e₂, e₅ of order 5: each reflection maps a unit vector onto a unit vector, w = (1, 0, …, 1) with τ = 1 as
  // LAPACK's dlarfg forms it; a unit Householder vector there holds 1/√2 and its products round. SciPy's
  // subspace_angles takes an arcsin of the singular values: 1 − 2⁻⁵² there is an angle off by 2e-8.
  const matrix<double> a {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}};
  const auto           exact {[](const matrix<double>& m) {
                                return std::ranges::all_of(m, [](double x) { return x == 0.0 || x == 1.0 || x == -1.0; });
                              }};
  const auto factored               {qr(a)};
  EXPECT(exact(factored.q()) && exact(factored.r()));
  const svd_decomposition<double> s {a};
  EXPECT(s.singular_values()[0] == 1.0 && s.singular_values()[1] == 1.0 && s.singular_values()[2] == 1.0);
  EXPECT(exact(s.u()) && exact(s.vh()));
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

// qr_r is the decomposition's R to the bit, Q never formed: tall, wide, square past one block of reflectors, complex,
// a rank-deficient matrix; a non-finite entry refused as the decomposition refuses it.
TEST(qr_r_is_the_decompositions_r_to_the_bit)
{
  using c = std::complex<double>;
  const auto real_case {[](std::size_t rows, std::size_t cols) {
                          std::vector<double> e(rows * cols);
                          for (std::size_t i {0}; i < e.size(); ++i) {
                            e[i] = std::sin(static_cast<double>(i) * 0.37) + ((i % 7 == 0) ? 1.0 : 0.0);
                          }
                          const matrix<double> a {rows, cols, e};
                          return numerical::qr_r(a) == qr(a).r();
                        }};
  EXPECT(real_case(70, 40) && real_case(40, 70) && real_case(65, 65) && real_case(1, 1));
  std::vector<c> e(std::size_t {33} *29);
  for (std::size_t i {0}; i < e.size(); ++i) {
    e[i] = c {std::cos(static_cast<double>(i) * 0.11), std::sin(static_cast<double>(i) * 0.23)};
  }
  const matrix<c> complex_a      {33, 29, e};
  EXPECT(numerical::qr_r(complex_a) == qr(complex_a).r());
  const matrix<double> dependent {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}};
  EXPECT(numerical::qr_r(dependent) == qr(dependent).r());
  const matrix<double> bad       {{1.0, std::numeric_limits<double>::quiet_NaN()}, {0.0, 1.0}};
  EXPECT_THROWS(numerical::qr_r(bad), numerical::non_finite_error);
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

namespace {
  // Whether the column-major factorisation (the x86-64 units') and the row-major one (arm64's) of the same rows × cols
  // matrix leave the same R and reflectors in place, and the same blocks' T, to rounding: past one block of reflectors,
  // tall and wide.
  template <typename T>
  bool qr_layouts_agree(std::size_t rows,
                        std::size_t cols)
  {
    numerical::buffer<T> by_rows(rows * cols);
    numerical::buffer<T> by_columns(rows * cols);
    for (std::size_t i {0}; i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        const double x {std::sin(static_cast<double>((i * 5) + (j * 11) + 1))};
        by_rows[(i * cols) + j]    = numerical::detail::from_carried<T>(x) + (i == j ? T {2} : T {});
        by_columns[(j * rows) + i] = by_rows[(i * cols) + j];
      }
    }
    if constexpr (!std::is_floating_point_v<T>) { // an imaginary part, so that the adjoints are not transposes
      for (std::size_t e {0}; e < rows * cols; e += 3) {
        by_rows[e] += T {0.0, 0.5};
      }
      for (std::size_t i {0}; i < rows; ++i) {
        for (std::size_t j {0}; j < cols; ++j) {
          by_columns[(j * rows) + i] = by_rows[(i * cols) + j];
        }
      }
    }
    const auto row_blocks    {numerical::detail::reflect_in_rows(by_rows, rows, cols)};
    const auto column_blocks {numerical::detail::reflect_in_columns(by_columns, rows, cols)};
    bool       agree         {row_blocks.size() == column_blocks.size()};
    for (std::size_t i {0}; i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        agree = agree && std::abs(by_rows[(i * cols) + j] - by_columns[(j * rows) + i]) < 1e-12;
      }
    }
    for (std::size_t b {0}; agree && b < row_blocks.size(); ++b) {
      for (std::size_t e {0}; e < row_blocks[b].t.size(); ++e) {
        agree = agree && std::abs(row_blocks[b].t[e] - column_blocks[b].t[e]) < 1e-12;
      }
    }
    return agree;
  }
} // namespace

TEST(qr_factorises_alike_column_major_and_row_major)
{
  EXPECT(qr_layouts_agree<double>(70, 40));
  EXPECT(qr_layouts_agree<double>(40, 70));
  EXPECT(qr_layouts_agree<double>(33, 33));
  EXPECT((qr_layouts_agree<std::complex<double>>(70, 40)));
  EXPECT((qr_layouts_agree<std::complex<double>>(40, 70)));
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

// A diagonal matrix's eigenvalues and singular values exactly its entries: the matrix divided by a power of two, and
// the values multiplied back by it, exactly, where a division by the largest entry rounded 7/25 and 14/25 and the
// product back missed 7 and 14.
TEST(a_diagonal_matrix_decomposes_to_its_entries_exactly)
{
  const matrix<double> a {{7.0, 0.0, 0.0}, {0.0, 14.0, 0.0}, {0.0, 0.0, 25.0}};
  for (const numerical::eigen_method method : eigen_methods) {
    const symmetric_eigen e {a, method};
    EXPECT(e.values()[0] == 7.0 && e.values()[1] == 14.0 && e.values()[2] == 25.0);
  }
  const vector<double> sigma            {singular_values(a)};
  EXPECT(sigma[0] == 25.0 && sigma[1] == 14.0 && sigma[2] == 7.0);
  const vector<double> lambda           {symmetric_eigenvalues(a)};
  EXPECT(lambda[0] == 7.0 && lambda[1] == 14.0 && lambda[2] == 25.0);
  const svd_decomposition<double> s     {a};
  EXPECT(s.singular_values()[0] == 25.0 && s.singular_values()[1] == 14.0 && s.singular_values()[2] == 7.0);
  const matrix<std::complex<double>> h  {{{7.0, 0.0}, {}, {}}, {{}, {14.0, 0.0}, {}}, {{}, {}, {25.0, 0.0}}};
  const numerical::hermitian_eigen   he {h};
  EXPECT(he.values()[0] == 7.0 && he.values()[1] == 14.0 && he.values()[2] == 25.0);
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
  numerical::buffer<double> a {2.0, 1.0, 0.5, 1.0, 3.0, -1.0, 0.5, -1.0, 4.0};
  numerical::buffer<double> v {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  EXPECT_THROWS(numerical::detail::cyclic_jacobi(a, v, 3, 1.0, 1), numerical::convergence_error);
  numerical::buffer<double> d {2.0, 3.0, 4.0};
  numerical::buffer<double> e {1.0, -1.0, 0.0};
  EXPECT_THROWS(numerical::detail::tridiagonal_ql(d, e, v, 3, 1), numerical::convergence_error);
}

// The two methods agree, and the default reaches rounding: a 40 × 40 Hilbert matrix (eigenvalues down to
// 1e-18 of the largest) with half its diagonal shifted keeps orthonormal vectors and residuals near machine
// precision.
TEST(symmetric_eigen_methods_agree_and_reach_rounding)
{
  const std::size_t         n {40};
  numerical::buffer<double> entries(n * n, 0.0);
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

// Orders whose reductions share their steps with the threads (the coefficients among the products' tasks, the
// corrections and the next column by bands): the same bits on one thread as on all, an orthonormal basis, A back.
TEST(a_reduction_large_enough_to_share_its_steps)
{
  const std::size_t         n     {520};
  numerical::buffer<double> entries(n * n, 0.0);
  std::uint64_t             state {7};
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j <= i; ++j) {
      state                  = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
      entries[(i * n) + j]   = (static_cast<double>(state >> 11U) * 0x1.0p-52) - 0.5;
      entries[(j * n) + i]   = entries[(i * n) + j];
    }
  }
  const matrix<double>  a   {n, n, std::move(entries)};
  const symmetric_eigen all {a};
  numerical::set_max_threads(1);
  const symmetric_eigen one {a};
  numerical::set_max_threads(0);
  EXPECT(all.values() == one.values());
  EXPECT(all.vectors() == one.vectors());
  EXPECT(orthonormality_error(all.vectors()) < 1e-12);
  EXPECT(approx_equal(reconstruct(all), a, 1e-11));
  // And a bidiagonal reduction whose steps share theirs on x86-64 (from 256 rows): the same values on one thread.
  const std::size_t   m {300};
  std::vector<double> general(m * m);
  for (double& entry : general) {
    state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
    entry = (static_cast<double>(state >> 11U) * 0x1.0p-52) - 0.5;
  }
  const matrix<double> g      {m, m, general};
  const vector<double> values {numerical::singular_values(g)};
  numerical::set_max_threads(1);
  const vector<double> alone  {numerical::singular_values(g)};
  numerical::set_max_threads(0);
  EXPECT(values == alone);
  EXPECT(values.size() == m && values[0] >= values[m - 1] && values[m - 1] > 0.0);
}

// Implicit QL on request gives the divide and conquer's eigenvalues, past the order where the divide and conquer
// splits (QL alone below it); the Jacobi method has no complex form and is refused.
TEST(hermitian_eigen_by_implicit_ql_on_request)
{
  using c = std::complex<double>;
  const std::size_t n {70};
  std::vector<c>    entries(n * n);
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j <= i; ++j) {
      const c z {std::sin(2.0 + static_cast<double>((i * n) + j)), i == j ? 0.0 : std::cos(static_cast<double>((2 * i) + j))};
      entries[(i * n) + j] = z;
      entries[(j * n) + i] = std::conj(z);
    }
  }
  const matrix<c>                  a  {n, n, entries};
  const numerical::hermitian_eigen ql {a, numerical::eigen_method::tridiagonal_ql};
  const numerical::hermitian_eigen dc {a};
  EXPECT(orthonormality_error(ql.vectors()) < 1e-13);
  EXPECT(approx_equal(ql.values(), dc.values(), 1e-12));
  EXPECT_THROWS(numerical::hermitian_eigen(a, numerical::eigen_method::jacobi), std::invalid_argument);
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
  EXPECT_THROWS(numerical::hermitian_eigen {(matrix<c> {{c {1.0}, c {}}, {c {std::numeric_limits<double>::infinity()}, c {1.0}}})},
                numerical::non_finite_error); // a lower entry's real part
  EXPECT_THROWS(numerical::hermitian_eigen {(matrix<c> {{c {1.0}, c {}}, {c {0.0, std::numeric_limits<double>::quiet_NaN()}, c {1.0}}})},
                numerical::non_finite_error); // and its imaginary part
  // The diagonal's imaginary part is ignored, as NumPy's eigh ignores it: a huge or infinite one changes no value.
  for (const double odd : {1e300, std::numeric_limits<double>::infinity()}) {
    const matrix<c> real_diagonal {{c {2.0}, c {}}, {c {1.0, 1.0}, c {3.0}}};
    const matrix<c> odd_diagonal  {{c {2.0, odd}, c {}}, {c {1.0, 1.0}, c {3.0, -odd}}};
    EXPECT(approx_equal(numerical::hermitian_eigen {odd_diagonal}.values(), numerical::hermitian_eigen {real_diagonal}.values(), 1e-14));
  }
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

// A complex matrix past the bidiagonal reduction's crossover is reduced by panels, its bidiagonal turned real by unit
// phases on its rows and columns: A = U Σ Vᴴ with U and V unitary, the singular values those of the unblocked
// reduction's bidiagonal, and the same on one thread as on all (past 256 rows the steps share theirs on x86-64).
TEST(a_complex_matrix_past_the_crossover_is_reduced_by_panels)
{
  using c = std::complex<double>;
  std::uint64_t state {11};
  for (const std::size_t n : {numerical::detail::bidiagonal_crossover + 3, std::size_t {300}}) {
    std::vector<c> entries(n * n);
    for (c& entry : entries) {
      state                   = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
      const double re {(static_cast<double>(state >> 11U) * 0x1.0p-52) - 0.5};
      state                   = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
      entry                   = c {re, (static_cast<double>(state >> 11U) * 0x1.0p-52) - 0.5};
    }
    const matrix<c>         a {n, n, entries};
    const svd_decomposition s {a, numerical::svd_method::bidiagonal_qr};
    EXPECT(approx_equal(reconstruct(s), a, 1e-12));
    EXPECT(orthonormality_error(s.u()) < 1e-13);
    EXPECT(orthonormality_error(s.v()) < 1e-13);
    numerical::buffer<c>      b(entries.begin(), entries.end());
    numerical::buffer<c>      none;
    numerical::buffer<c>      none_too;
    numerical::buffer<double> d(n);
    numerical::buffer<double> e(n);
    numerical::detail::bidiagonalize_unblocked(b, n, d, e, none, none_too, false);
    numerical::detail::bidiagonal_qr(d, e, none, none_too, n, 30 * n);
    const vector<double> values {numerical::singular_values(a)};
    double               gap    {0.0};
    for (std::size_t k {0}; k < n; ++k) {
      gap = std::max(gap, std::abs(values[k] - d[k]));
    }
    EXPECT(gap < 1e-12 * d[0]);
    numerical::set_max_threads(1);
    const vector<double> alone {numerical::singular_values(a)};
    numerical::set_max_threads(0);
    EXPECT(values == alone);
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
  bool bidiagonal_qr_reconstructs(numerical::buffer<double> d,
                                  numerical::buffer<double> e)
  {
    const std::size_t   n {d.size()};
    std::vector<double> b(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      b[(i * n) + i] = d[i];
      if (i + 1 < n) {
        b[(i * n) + i + 1] = e[i];
      }
    }
    numerical::buffer<double> ut(n * n, 0.0);
    numerical::buffer<double> vt(n * n, 0.0);
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
                               numerical::buffer<double> d(d0.begin(), d0.end());
                               numerical::buffer<double> e(e0.begin(), e0.end());
                               numerical::buffer<double> ut(n * n, 0.0);
                               numerical::buffer<double> vt(n * n, 0.0);
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

// A bidiagonal graded up, each diagonal entry 2²⁶ times the one above it: the bulge chased from the top never brought
// its small entries down (a convergence_error from every method but Jacobi), from its larger end the iteration
// converges. Graded down, the same. U Σ Vᵀ = A with U and V orthonormal either way.
TEST(svd_converges_on_a_bidiagonal_graded_either_way)
{
  constexpr std::size_t n {17};
  for (const bool up : {true, false}) {
    std::vector<double> entries(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      const int place {static_cast<int>(up ? n - 1 - i : i)};
      entries[(i * n) + i] = std::ldexp(1.0, -26 * place);
      if (i + 1 < n) {
        entries[(i * n) + i + 1] = std::ldexp(1.0, (-26 * place) + (up ? 13 : -13));
      }
    }
    const matrix<double> a {n, n, entries};
    for (const numerical::svd_method method : svd_methods) {
      const svd_decomposition s {a, method};
      EXPECT(approx_equal(reconstruct(s), a, 1e-15));
      EXPECT(orthonormality_error(s.u()) < 1e-14);
      EXPECT(orthonormality_error(s.v()) < 1e-14);
      EXPECT(std::abs((s.singular_values()[n - 1] / 5.9091062713565978771e-126) - 1.0) < 1e-12);
    }
  }
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
    numerical::buffer<double> chain_d(13, 0.0);
    numerical::buffer<double> chain_e(13, 0.0);
    for (std::size_t i {0}; i < 13; ++i) {
      chain_d[i] = std::pow(1e-8, static_cast<double>(2 * i));
      if (i + 1 < 13) {
        chain_e[i] = std::pow(1e-8, static_cast<double>((2 * i) + 1));
      }
    }
    EXPECT(bidiagonal_qr_reconstructs(chain_d, chain_e));
  }
  EXPECT(bidiagonal_qr_reconstructs({8.0, 3.0, 5.0}, {1.0, 4.0, 0.0}));          // b = 0 exactly, c is not
  numerical::buffer<double> d {2.0, 3.0, 4.0};
  numerical::buffer<double> e {1.0, -1.0, 0.0};
  numerical::buffer<double> ut(9, 0.0);
  numerical::buffer<double> vt(9, 0.0);
  EXPECT_THROWS(numerical::detail::bidiagonal_qr(d, e, ut, vt, 3, 0), numerical::convergence_error);
  for (const numerical::svd_method method : svd_methods) {
    const svd_decomposition none {matrix<double> {3, 0, {}}, method};
    EXPECT(none.singular_values().size() == 0 && none.u().rows() == 3 && none.u().cols() == 0);
  }
}

// Jacobi's sweeps are bounded: one sweep does not orthogonalise a dense 3 × 3, which is then an error.
TEST(one_sided_jacobi_reports_a_sweep_bound_reached)
{
  numerical::buffer<double> w {4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0};
  numerical::buffer<double> v {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  EXPECT_THROWS(numerical::detail::one_sided_jacobi(w, v, 3, 1), numerical::convergence_error);
}

// Jacobi converges quadratically when each rotation takes the smaller angle: this dense 8 × 8 orthogonalises
// in 9 sweeps under clang and GCC, at -O0 and -O3, and in 11 to 16 with the larger angle.
TEST(one_sided_jacobi_converges_in_few_sweeps)
{
  const std::size_t         n               {8};
  numerical::buffer<double> w(n * n, double {});
  numerical::buffer<double> v(n * n, 0.0);
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

// A phase is of unit modulus at any magnitude a scalar holds, and 1 at zero: in float the factor that rescues a
// subnormal once overflowed to infinity, and zero times it was NaN.
TEST(the_phase_of_a_scalar_is_of_unit_modulus_in_every_precision)
{
  using numerical::detail::phase_of;
  using cf = std::complex<float>;
  EXPECT(phase_of(0.0F) == 1.0F && phase_of(cf {}) == cf {1.0F});
  EXPECT(phase_of(0.0) == 1.0 && phase_of(std::complex<double> {}) == std::complex<double> {1.0});
  const float tiny {std::numeric_limits<float>::denorm_min() * 7.0F};
  EXPECT(phase_of(-tiny) == -1.0F);
  EXPECT(std::abs(std::abs(phase_of(cf {tiny, tiny})) - 1.0F) < 4.0F * std::numeric_limits<float>::epsilon());
  EXPECT(std::abs(std::abs(phase_of(std::complex<double> {1e-320, -3e-320})) - 1.0) < 4.0 * std::numeric_limits<double>::epsilon());
}

// A cutoff is NumPy's by default (the largest singular value times ε times the larger dimension), relative to the
// largest as rcond and rtol, or absolute as tol; a singular value at or under it counts as zero in the rank, the
// pseudo-inverse and the least-squares solution, on the QR's road as on the decomposition's.
TEST(rank_pseudo_inverse_and_lstsq_take_numpys_cutoffs)
{
  using numerical::rank_cutoff;
  const matrix<double> d {diagonal_matrix(vector<double> {1.0, 1e-3, 1e-9, 0.0})};
  EXPECT_EQ(numerical::matrix_rank(d), 3U);
  EXPECT_EQ(numerical::matrix_rank(d, rank_cutoff::relative(1e-6)), 2U);
  EXPECT_EQ(numerical::matrix_rank(d, rank_cutoff::relative(1e-3)), 1U); // at the cutoff counts as zero
  EXPECT_EQ(numerical::matrix_rank(d, rank_cutoff::relative(0.0)), 3U);
  EXPECT_EQ(numerical::matrix_rank(d, rank_cutoff::absolute(1e-9)), 2U);
  EXPECT_EQ(numerical::matrix_rank(d, rank_cutoff::absolute(2.0)), 0U);
  EXPECT_THROWS(static_cast<void>(rank_cutoff::relative(-1e-3)), std::invalid_argument);
  EXPECT_THROWS(static_cast<void>(rank_cutoff::absolute(std::numeric_limits<double>::quiet_NaN())), std::invalid_argument);
  EXPECT(approx_equal(numerical::pinv(d, rank_cutoff::relative(1e-6)), diagonal_matrix(vector<double> {1.0, 1e3, 0.0, 0.0})));
  const vector<double> ones                           {1.0, 1.0, 1.0, 1.0};
  EXPECT(approx_equal(numerical::lstsq(d, ones, rank_cutoff::relative(1e-6)), vector<double> {1.0, 1e3, 0.0, 0.0}));
  const numerical::least_squares_solution<double> cut {
    numerical::least_squares(d, matrix<double> {4, 1, std::vector<double>(4, 1.0)}, rank_cutoff::relative(1e-6))};
  EXPECT_EQ(cut.rank, 2U);
  // Tall and full-rank: the QR's road, which must still drop what the cutoff drops, at any scale.
  const matrix<double> tall {{2.0, 0.0}, {0.0, 1e-3}, {0.0, 0.0}};
  EXPECT(approx_equal(numerical::lstsq(tall, vector<double> {2.0, 1.0, 5.0}), vector<double> {1.0, 1e3}));
  EXPECT(approx_equal(numerical::lstsq(tall, vector<double> {2.0, 1.0, 5.0}, rank_cutoff::absolute(1e-2)),
                      vector<double> {1.0, 0.0}));
  const matrix<double> huge {{1e300, 0.0}, {0.0, 1e297}, {0.0, 0.0}};
  EXPECT(approx_equal(numerical::lstsq(huge, vector<double> {1e300, 1e297, 0.0}, rank_cutoff::absolute(1e298)),
                      vector<double> {1.0, 0.0}));
  EXPECT(approx_equal(numerical::lstsq(huge, vector<double> {1e300, 1e297, 0.0}, rank_cutoff::absolute(1e296)),
                      vector<double> {1.0, 1.0}));
  // Tiny: unscaled, the cutoff would look negligible beside the scaled values, and the QR's road keep what it drops.
  const matrix<double> tiny {{1e-300, 0.0}, {0.0, 1e-303}, {0.0, 0.0}};
  EXPECT(approx_equal(numerical::lstsq(tiny, vector<double> {1e-300, 1e-303, 0.0}, rank_cutoff::absolute(1e-302)),
                      vector<double> {1.0, 0.0}));
  EXPECT(approx_equal(numerical::lstsq(tiny, vector<double> {1e-300, 1e-303, 0.0}, rank_cutoff::absolute(1e-304)),
                      vector<double> {1.0, 1.0}));
  const numerical::least_squares_solution<double> full {
    numerical::least_squares(tall, matrix<double> {{2.0}, {1.0}, {5.0}}, rank_cutoff::relative(1e-6))};
  EXPECT(full.rank == 2U && full.residuals.size() == 1U && std::abs(full.residuals[0] - 25.0) < 1e-9);
}

// hermitian, as NumPy's: the rank from the eigenvalues' magnitudes and the pseudo-inverse from the eigendecomposition,
// the lower triangle alone read (what lies above it, a NaN even, is not looked at); square or refused.
TEST(rank_and_pseudo_inverse_of_a_hermitian_matrix_read_its_lower_triangle)
{
  const double         nan {std::numeric_limits<double>::quiet_NaN()};
  const matrix<double> s   {{2.0, 1.0, 0.0}, {1.0, -3.0, 0.0}, {0.0, 0.0, 0.0}}; // indefinite, rank 2
  const matrix<double> low {{2.0, 99.0, nan}, {1.0, -3.0, 7.0}, {0.0, 0.0, 0.0}};
  EXPECT_EQ(numerical::matrix_rank(low, {}, true), 2U);
  EXPECT(approx_equal(numerical::pinv(low, {}, true), numerical::pinv(s), 1e-12));
  EXPECT_EQ(numerical::matrix_rank(s, numerical::rank_cutoff::absolute(2.5), true), 1U); // |−3.2| kept, 2.2 not
  using c = std::complex<double>;
  const matrix<c> z      {{c {2.0, 0.0}, c {1.0, 1.0}}, {c {1.0, -1.0}, c {3.0, 0.0}}};
  const matrix<c> z_low  {{c {2.0, 5.0}, c {nan, 0.0}}, {c {1.0, -1.0}, c {3.0, 0.0}}};  // the diagonal's real part
  EXPECT(approx_equal(numerical::pinv(z_low, {}, true), numerical::inverse(z), 1e-12));
  EXPECT_EQ(numerical::matrix_rank(z_low, {}, true), 2U);
  const matrix<double> wide {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  EXPECT_THROWS(static_cast<void>(numerical::pinv(wide, {}, true)), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::matrix_rank(wide, {}, true)), numerical::dimension_error);
  const matrix<double> nan_below {{1.0, 0.0}, {nan, 1.0}};
  EXPECT_THROWS(static_cast<void>(numerical::pinv(nan_below, {}, true)), numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::matrix_rank(nan_below, {}, true)), numerical::non_finite_error);
}

// lstsq is NumPy's: the least-squares solution of a full-rank tall system, accurate on a Vandermonde
// fit (modified Gram–Schmidt left a coefficient error of 0.6 at degree 11), and the minimum-norm one
// of a rank-deficient or wide system.
// A matrix right-hand side is one system per column, from one factorisation: each column of the answer is the
// vector solve of that column.
// A rank-deficient system's least squares, several right-hand sides at once (the singular value decomposition's
// coefficients and solution through the matrix product): pinv(a)·b, real and complex; a zero matrix's, zeros.
TEST(a_rank_deficient_lstsq_of_several_right_hand_sides_is_the_pseudo_inverse_times_them)
{
  const matrix<double> a {{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}; // rank 2: column 3 = 1 + 2
  const matrix<double> b {{1.0, 0.0, 2.0}, {0.0, 1.0, -1.0}, {3.0, 1.0, 0.0}, {1.0, 2.0, 1.0}};
  EXPECT(approx_equal(numerical::lstsq(a, b), numerical::pinv(a) * b, 1e-12));
  using C = std::complex<double>;
  const matrix<C> z         {{C {1.0, 1.0}, C {2.0, 2.0}}, {C {0.0, 1.0}, C {0.0, 2.0}}, {C {1.0, 0.0}, C {2.0, 0.0}}}; // rank 1
  const matrix<C> w         {{C {1.0, 0.0}, C {0.0, 1.0}}, {C {2.0, -1.0}, C {1.0, 1.0}}, {C {0.5, 0.0}, C {-1.0, 0.0}}};
  EXPECT(approx_equal(numerical::lstsq(z, w), numerical::pinv(z) * w, 1e-12));
  const matrix<double> none {numerical::lstsq(matrix<double> {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, b)};
  EXPECT(none.rows() == 2 && none.cols() == 3 && std::ranges::all_of(none, [](double v) { return v == 0.0; }));
}

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

namespace {
  // A deterministic matrix of the given shape, far from rank-deficient.
  template <typename T>
  matrix<T> spread(std::size_t rows,
                   std::size_t cols)
  {
    std::vector<T> entries(rows * cols);
    for (std::size_t i {0}; i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        const double angle {static_cast<double>((i * 7) + (j * 13) + ((i * j) % 11))};
        if constexpr (std::is_same_v<T, double>) {
          entries[(i * cols) + j] = std::sin(angle) + (i == j ? 3.0 : 0.0);
        }
        else {
          entries[(i * cols) + j] = T {std::sin(angle) + (i == j ? 3.0 : 0.0), std::cos(angle * 0.5)};
        }
      }
    }
    return matrix<T> {rows, cols, std::move(entries)};
  }

  // lstsq against the pseudo-inverse (from the SVD): the same solution, to rounding.
  template <typename T>
  bool lstsq_is_pinv_times(const matrix<T>& a)
  {
    std::vector<T> b(a.rows());
    for (std::size_t i {0}; i < a.rows(); ++i) {
      b[i] = T {std::cos(static_cast<double>(i))};
    }
    const vector<T> got  {numerical::lstsq(a, vector<T> {b})};
    const matrix<T> want {numerical::pinv(a) * matrix<T> {a.rows(), 1, b}};
    bool            same {got.size() == a.cols()};
    for (std::size_t j {0}; j < a.cols(); ++j) {
      same = same && std::abs(got[j] - want(j, 0)) < 1e-10;
    }
    return same;
  }
} // namespace

// Where a bound on the condition number shows the rank full with room to spare, lstsq answers by one QR (of A, or
// of Aᴴ for a wide A); elsewhere by the SVD: a zero matrix, a scale past a power of two's range, a zero on R's
// diagonal, a bound too near NumPy's rank tolerance.
TEST(lstsq_takes_a_qr_where_the_rank_is_surely_full)
{
  using c = std::complex<double>;
  EXPECT(lstsq_is_pinv_times(spread<double>(150, 100))); // R⁻¹ in two chunks of columns
  EXPECT(lstsq_is_pinv_times(spread<c>(40, 30)));
  EXPECT(lstsq_is_pinv_times(spread<c>(30, 40)));
  EXPECT(lstsq_is_pinv_times(spread<double>(20, 90)));
  EXPECT(approx_equal(numerical::lstsq(matrix<double> {3, 2, std::vector<double>(6, 0.0)}, vector<double> {1.0, 2.0, 3.0}),
                      vector<double> {0.0, 0.0}));
  EXPECT(approx_equal(numerical::lstsq(matrix<double> {{1e-310, 0.0}, {0.0, 2e-310}}, vector<double> {1e-310, 2e-310}),
                      vector<double> {1.0, 1.0}));
  EXPECT(approx_equal(numerical::lstsq(matrix<double> {{1e305, 0.0}, {0.0, 2e305}}, vector<double> {1e305, 2e305}),
                      vector<double> {1.0, 1.0}));
  EXPECT(approx_equal(numerical::lstsq(matrix<double> {{1.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}}, vector<double> {1.0, 1.0, 2.0}),
                      vector<double> {1.0, 0.0}));
  // κ about 4e14: full rank for NumPy, too near its tolerance for the bound; the residual stays small.
  const matrix<double> near {{1.0, 1.0}, {1.0, 1.0 + 1e-14}, {0.0, 0.0}};
  const vector<double> rhs  {2.0, 2.0 + 1e-14, 0.0};
  const vector<double> x    {numerical::lstsq(near, rhs)};
  EXPECT(std::abs((x[0] + x[1]) - 2.0) < 1e-6);
}

// An expiring matrix or decomposition hands its storage over: the same factors as by copy, the source left empty.
TEST(decompositions_take_and_give_their_storage_without_copies)
{
  const matrix<double>            spd   {{4.0, 2.0, 0.0}, {2.0, 5.0, 1.0}, {0.0, 1.0, 3.0}};
  matrix<double>                  spare {spd};
  const numerical::buffer<double> taken {std::move(spare).elements()};
  EXPECT(taken == std::vector<double>(spd.begin(), spd.end()));
  EXPECT(spare.rows() == 0U && spare.cols() == 0U); // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved): the state a move leaves is the point
  const numerical::tensor<double> moved {matrix<double> {spd}};
  EXPECT(moved == numerical::tensor<double> {spd});
  EXPECT(numerical::cholesky_decomposition<double> {matrix<double> {spd}}.lower() == numerical::cholesky_decomposition<double> {spd}.lower());
  EXPECT(numerical::cholesky(matrix<double> {spd}).lower() == numerical::cholesky(spd).lower());
  EXPECT(numerical::lu_decomposition<double> {matrix<double> {spd}}.inverse() == numerical::inverse(spd));
  EXPECT(numerical::determinant(matrix<double> {spd}) == numerical::determinant(spd));
  EXPECT(numerical::solve(matrix<double> {spd}, vector<double> {1.0, 2.0, 3.0}) == numerical::solve(spd, vector<double> {1.0, 2.0, 3.0}));
  numerical::qr_decomposition<double> qr     {matrix<double> {spd}};
  const matrix<double>                q_copy {qr.q()};
  const matrix<double>                r_copy {qr.r()};
  EXPECT(std::move(qr).q() == q_copy);
  EXPECT(std::move(qr).r() == r_copy); // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved): q() && leaves R
  numerical::symmetric_eigen eigen   {spd};
  const matrix<double>       vectors {eigen.vectors()};
  EXPECT(std::move(eigen).vectors() == vectors);
  using c = std::complex<double>;
  numerical::hermitian_eigen complex_eigen   {matrix<c> {{c {2.0, 0.0}, c {0.0, 1.0}}, {c {0.0, -1.0}, c {3.0, 0.0}}}};
  const matrix<c>            complex_vectors {complex_eigen.vectors()};
  EXPECT(std::move(complex_eigen).vectors() == complex_vectors);
  numerical::svd_decomposition<double> svd   {spd};
  const matrix<double>                 u     {svd.u()};
  EXPECT(std::move(svd).u() == u);
  EXPECT(svd.v().rows() == 3U); // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved): u() && leaves V
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

// The scale past one band of rows: the largest entries in the first rows alone, the rest of order one. The singular
// values are those of the matrix at unit scale, times the scale.
// Rows of scales far apart: the largest entries in the first 32 rows alone, the rest of order one. Divided by the
// largest, the small rows' squares underflow, and one-sided Jacobi judged such a pair by sums of squares of 0 (and past
// a ratio of 1e150 its tangent overflowed to no rotation): it did not converge. Its singular values are the others',
// the large ones at least, whatever the method.
TEST(svd_of_rows_of_scales_far_apart)
{
  const std::size_t   n     {40};
  std::vector<double> unit(n * n, 0.0);
  std::uint64_t       state {7};
  for (double& x : unit) {
    state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
    x     = (static_cast<double>(state >> 11U) * 0x1.0p-53) - 0.5;
  }
  for (const double scale : {1e100, 1e150, 1e200, 1e300}) {
    std::vector<double> scaled {unit};
    for (std::size_t e {0}; e < 32 * n; ++e) {
      scaled[e] *= scale;
    }
    std::vector<double> columns {unit}; // the first columns large instead
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n / 2; ++j) {
        columns[(i * n) + j] *= scale;
      }
    }
    const svd_decomposition by_columns     {matrix<double> {n, n, columns}, numerical::svd_method::jacobi};
    bool                    columns_finite {std::isfinite(by_columns.singular_values()[0])};
    for (std::size_t i {0}; i < n; ++i) {
      columns_finite = columns_finite && std::isfinite(by_columns.u()(i, 0)) && std::isfinite(by_columns.v()(i, 0));
    }
    EXPECT(columns_finite);
    const svd_decomposition jacobi {matrix<double> {n, n, scaled}, numerical::svd_method::jacobi};
    const svd_decomposition other  {matrix<double> {n, n, scaled}};
    bool                    same   {true};
    for (std::size_t i {0}; i < 32; ++i) {
      same = same && std::abs((jacobi.singular_values()[i] / other.singular_values()[i]) - 1.0) < 1e-12;
    }
    bool finite {true};
    for (std::size_t i {0}; i < n; ++i) {
      finite = finite && std::isfinite(jacobi.singular_values()[i]) && std::isfinite(jacobi.u()(i, 0)) && std::isfinite(jacobi.v()(i, 0));
    }
    EXPECT(same && finite);
  }
  // Orthonormal factors to rounding while the small rows stay above the subnormals (to a ratio of 1e250), each
  // row reconstructed to its own scale.
  for (const double scale : {1e100, 1e150, 1e200, 1e250}) {
    for (const bool by_rows : {true, false}) {
      std::vector<double> apart {unit};
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          apart[(i * n) + j] *= (by_rows ? i : j) < n / 2 ? scale : 1.0;
        }
      }
      const svd_decomposition jacobi {matrix<double> {n, n, apart}, numerical::svd_method::jacobi};
      double                  off    {0.0};
      for (std::size_t a {0}; a < n; ++a) {
        for (std::size_t b {0}; b < n; ++b) {
          double uu {a == b ? -1.0 : 0.0};
          double vv {a == b ? -1.0 : 0.0};
          for (std::size_t k {0}; k < n; ++k) {
            uu += jacobi.u()(k, a) * jacobi.u()(k, b);
            vv += jacobi.v()(k, a) * jacobi.v()(k, b);
          }
          off = std::max({off, std::abs(uu), std::abs(vv)});
        }
      }
      EXPECT(off < 1e-12);
    }
  }
  // A pair whose ratio ζ is near the largest double: its tangent underflowed to no rotation, and the sweeps never ended.
  std::vector<double> near  {unit};
  std::uint64_t       other {40};
  for (double& x : near) {
    other = (other * 6364136223846793005ULL) + 1442695040888963407ULL;
    x     = (static_cast<double>(other >> 11U) * 0x1.0p-53) - 0.5;
  }
  for (std::size_t e {0}; e < (n / 2) * n; ++e) {
    near[e] *= 1e300;
  }
  const svd_decomposition far {matrix<double> {n, n, near}, numerical::svd_method::jacobi};
  EXPECT(std::isfinite(far.singular_values()[0]) && far.singular_values()[0] > 1e299);
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
  // Another scalar, checked entry by entry: a long double LU refuses an infinity as a double one does.
  using wide = long double;
  EXPECT_THROWS((lu_decomposition<wide> {matrix<wide> {{1.0L, std::numeric_limits<wide>::infinity()}, {0.0L, 1.0L}}}),
                numerical::non_finite_error);
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
  // applied at its step; a complex pivot's reciprocal taken once and multiplied by, as LAPACK's zgetf2 (a real
  // pivot divides). Returns the determinant and the solution of `A x = b`.
  template <typename T>
  T over(const T& x,
         const T& pivot)
  {
    if constexpr (std::is_same_v<T, std::complex<double>>) {
      return x * (T {1.0} / pivot);
    }
    else {
      return x / pivot;
    }
  }

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
        const T factor {over(a[(r * n) + k], a[(k * n) + k])};
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
      x[i] = over(sum, a[(i * n) + i]);
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

// The LU's panel width by order: narrow panels up to lu_small_order, 64 columns above, and 128 from lu_wide_order,
// where the trailing updates' memory traffic sets the time.
TEST(lu_panel_width_follows_the_order)
{
  using numerical::detail::lu_block_for;
  EXPECT_EQ(lu_block_for(numerical::detail::lu_small_order), numerical::detail::lu_small_block);
  EXPECT_EQ(lu_block_for(numerical::detail::lu_small_order + 1), numerical::detail::lu_block);
  EXPECT_EQ(lu_block_for(numerical::detail::lu_wide_order - 1), numerical::detail::lu_block);
  EXPECT_EQ(lu_block_for(numerical::detail::lu_wide_order), numerical::detail::lu_wide_block);
}

// Past one panel: panels of `lu_block_for(n)` columns, the last partial. On the portable path: on a fused one the trailing products fuse
// their multiply-adds and the panels do not, which a later test bounds instead, and complex products sum the real
// and imaginary parts apart.
TEST(blocked_lu_equals_the_unblocked_algorithm_to_the_bit)
{
  const std::size_t n   {(2 * numerical::detail::lu_block) + 22};
  (void)numerical::limit_gemm_path(numerical::gemm_path::portable);
  EXPECT(blocked_lu_equals_unblocked<double>(n));
  EXPECT(blocked_lu_equals_unblocked<std::complex<double>>(numerical::detail::lu_block + 40));
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
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
        l[(i * n) + j] = i == j ? T {std::sqrt(numerical::detail::real_part(sum))} : numerical::detail::over_real_diagonal(sum, l[(j * n) + j]);
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

// Past numerical::detail::lu_lookahead_columns: the first steps update the trailing block as one product, the last
// ones factorise the next panel beside the update of the rest. The same factors on one thread as on all of them,
// and a solve to rounding.
TEST(lu_solves_an_expiring_right_hand_side_in_its_own_storage)
{
  // Rows permuted in place, cycle by cycle: the same bits as the copying solve, the factors moved out the same too.
  std::vector<double> entries(std::size_t {60} *60);
  for (std::size_t i {0}; i < entries.size(); ++i) {
    entries[i] = std::sin(static_cast<double>(i) * 0.37) + std::cos(static_cast<double>(i * i) * 0.011);
  }
  const matrix<double>                  a               {60, 60, entries};
  numerical::lu_decomposition<double>   lu              {a};
  std::vector<double>                   rhs(std::size_t {60} *7);
  for (std::size_t i {0}; i < rhs.size(); ++i) {
    rhs[i] = std::cos(static_cast<double>(i) * 0.5);
  }
  const matrix<double> b                             {60, 7, rhs};
  const matrix<double> copied                        {lu.solve(b)};
  const matrix<double> taken                         {lu.solve(matrix<double> {b})};
  EXPECT(std::equal(copied.begin(), copied.end(), taken.begin()));
  const matrix<double> kept                          {lu.factors()};
  const matrix<double> moved                         {std::move(lu).factors()};
  EXPECT(std::equal(kept.begin(), kept.end(), moved.begin()));
  const numerical::lu_decomposition<double> again    {a};
  EXPECT_THROWS((void)again.solve(matrix<double> {59, 1, std::vector<double>(59, 1.0)}), numerical::dimension_error);
  const numerical::lu_decomposition<double> singular {matrix<double> {{1.0, 2.0}, {2.0, 4.0}}};
  EXPECT_THROWS((void)singular.solve(matrix<double> {2, 1, std::vector<double>(2, 1.0)}), numerical::singular_matrix_error);
}

TEST(lu_looks_ahead_one_panel_with_the_same_factors)
{
  const std::size_t   n     {numerical::detail::lu_lookahead_columns + (2 * numerical::detail::lu_block) + 5};
  std::vector<double> entries(n * n);
  std::uint64_t       state {12345};
  for (std::size_t i {0}; i < n * n; ++i) { // pseudo-random in [-1, 1), a diagonal of 40 on top: well-conditioned
    state      = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
    entries[i] = (static_cast<double>(state >> 11U) * 0x1.0p-52) - 1.0 + (i % (n + 1) == 0 ? 40.0 : 0.0);
  }
  const matrix<double> a   {n, n, entries};
  const vector<double> b   {std::vector<double>(n, 1.0)};
  const vector<double> all {numerical::lu_decomposition<double> {a}.solve(b)};
  numerical::set_max_threads(1);
  const vector<double> one {numerical::lu_decomposition<double> {a}.solve(b)};
  numerical::set_max_threads(0);
  EXPECT(all == one);
  double residual {0.0};
  for (std::size_t i {0}; i < n; ++i) {
    double sum {0.0};
    for (std::size_t j {0}; j < n; ++j) {
      sum += a(i, j) * all[j];
    }
    residual = std::max(residual, std::abs(sum - 1.0));
  }
  EXPECT(residual < 1e-8);
}

// The divide and conquer of eigh and of the SVD solves blocks of its tree on separate threads, each on one thread
// below a depth and the joins above it a level at a time; on one thread it is the recursion's order. Every block
// takes the same inputs either way: the same values and vectors, to the bit. Past eight threads' worth of leaves, so
// a thread's block has halves of its own.
TEST(divide_and_conquer_gives_the_same_bits_on_one_thread_as_on_all)
{
  const std::size_t   n     {(16 * numerical::detail::divide_leaf) + 9};
  std::vector<double> entries(n * n);
  std::uint64_t       state {2024};
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j <= i; ++j) {
      entries[(i * n) + j] = entries[(j * n) + i] = next_uniform(state);
    }
  }
  const matrix<double> a                          {n, n, entries};
  const matrix<double> wide                       {n - 40, n, std::vector<double>(entries.begin(), entries.begin() + static_cast<std::ptrdiff_t>((n - 40) * n))};
  numerical::set_max_threads(1);
  const symmetric_eigen                 eigen_one {a};
  const svd_decomposition<double>       svd_one   {wide};
  numerical::set_max_threads(0);
  const symmetric_eigen                 eigen_all {a};
  const svd_decomposition<double>       svd_all   {wide};
  EXPECT(eigen_one.values() == eigen_all.values());
  EXPECT(eigen_one.vectors() == eigen_all.vectors());
  EXPECT(svd_one.singular_values() == svd_all.singular_values());
  EXPECT(svd_one.u() == svd_all.u());
  EXPECT(svd_one.v() == svd_all.v());
}

// A complex inverse splits its columns by the threads (an LU solve's slices): past the portable path every complex
// product, whatever its shape, runs on the one kernel, so each element is the same however narrow its slice, and the
// inverse is the same to the bit on one thread as on all. At order 131 a slice of 64 columns rounded its products in
// the plain loop where the whole width took the real kernel. A product with an empty side leaves C alone.
TEST(complex_inverse_gives_the_same_bits_on_one_thread_as_on_all)
{
  using complex = std::complex<double>;
  const std::size_t    n     {131};
  std::vector<complex> entries(n * n);
  std::uint64_t        state {77};
  for (complex& z : entries) {
    const double re {next_uniform(state)};
    z = complex {re, next_uniform(state)};
  }
  const matrix<complex> a   {n, n, entries};
  numerical::set_max_threads(1);
  const matrix<complex> one {inverse(a)};
  numerical::set_max_threads(0);
  const matrix<complex> all {inverse(a)};
  EXPECT(one == all);
  const auto leaves_c       {[]<typename Z>(Z /*type*/) {
                               const std::vector<Z> operand(4, Z {1, 1});
                               Z                    untouched {2, -3};
                               for (const auto& [rows, cols, depth] : {std::array<std::size_t, 3> {0, 1, 4}, {1, 0, 4}, {1, 1, 0}}) {
                                 numerical::detail::gemm<Z>(rows, cols, depth, operand.data(), 4, operand.data(), 1, &untouched, 1);
                               }
                               return untouched == Z {2, -3};
                             }};
  EXPECT(leaves_c(complex {}));
  EXPECT(leaves_c(std::complex<float> {}));
  // On the portable path a small complex product keeps the plain loop, shallow or not: whole complex products,
  // summed in order (small integers, so exactly).
  numerical::limit_gemm_path(numerical::gemm_path::portable);
  for (const std::size_t depth : {std::size_t {2}, numerical::detail::gemm_min_depth}) {
    std::vector<std::complex<float>> left(depth);
    std::vector<std::complex<float>> right(depth);
    std::complex<float>              want {};
    for (std::size_t p {0}; p < depth; ++p) {
      left[p]  = {static_cast<float>(p + 1), static_cast<float>(2 - static_cast<int>(p))};
      right[p] = {static_cast<float>(p % 3), -1.0F};
      want    += left[p] * right[p];
    }
    std::complex<float> sum {};
    numerical::detail::gemm<std::complex<float>>(1, 1, depth, left.data(), depth, right.data(), 1, &sum, 1);
    EXPECT(sum == want);
  }
  numerical::limit_gemm_path(numerical::gemm_path::amx);
}

// Each thread keeps the joins' buffers from one decomposition to the next, and lets a large one go after it: at order
// 700 the last join's buffers are larger than it keeps. The decompositions on either side of it are the same to the
// bit, whatever the buffers held before.
TEST(eigh_keeps_its_buffers_and_lets_the_large_ones_go)
{
  const auto symmetric {[](std::size_t n, std::uint64_t seed) {
                          std::vector<double> entries(n * n);
                          std::uint64_t       state {seed};
                          for (std::size_t i {0}; i < n; ++i) {
                            for (std::size_t j {0}; j <= i; ++j) {
                              entries[(i * n) + j] = entries[(j * n) + i] = next_uniform(state);
                            }
                          }
                          return matrix<double> {n, n, entries};
                        }};
  const matrix<double>  small  {symmetric(200, 7)};
  const symmetric_eigen before {small};
  const symmetric_eigen large  {symmetric(700, 11)};
  const symmetric_eigen after  {small};
  EXPECT(before.values() == after.values());
  EXPECT(before.vectors() == after.vectors());
  EXPECT(approx_equal(reconstruct(large), symmetric(700, 11), 1e-9));
}

// The SVD's joins keep their buffers too, and let a large one go: an svd of order 700 between two of the same smaller
// matrix, taller than wide so both bases are joined, leaves the second the first's to the bit.
TEST(svd_keeps_its_buffers_and_lets_the_large_ones_go)
{
  const auto general {[](std::size_t rows, std::size_t cols, std::uint64_t seed) {
                        std::vector<double> entries(rows * cols);
                        std::uint64_t       state {seed};
                        for (double& x : entries) {
                          x = next_uniform(state) - 0.5;
                        }
                        return matrix<double> {rows, cols, entries};
                      }};
  const matrix<double>                 small  {general(230, 200, 3)};
  const svd_decomposition<double>      before {small};
  const matrix<double>                 big    {general(700, 700, 5)};
  const svd_decomposition<double>      large  {big};
  const svd_decomposition<double>      after  {small};
  EXPECT(before.singular_values() == after.singular_values());
  EXPECT(before.u() == after.u());
  EXPECT(before.v() == after.v());
  std::vector<double> us(std::size_t {700} *700);
  for (std::size_t i {0}; i < 700; ++i) {
    for (std::size_t j {0}; j < 700; ++j) {
      us[(i * 700) + j] = large.u()(i, j) * large.singular_values()[j];
    }
  }
  EXPECT(approx_equal(matrix<double> {700, 700, us} *large.vh(), big, 1e-9));
}

// Complete (NumPy's mode="complete"): Q is m × m, its columns past min(m, n) an orthonormal basis of the complement,
// and R m × n with zero rows past min(m, n); the reduced factors are its first columns and rows, to the bit.
TEST(qr_complete_extends_the_reduced_factors)
{
  for (const auto& [rows, cols] : {std::pair<std::size_t, std::size_t> {7, 3}, {3, 7}, {5, 5}, {80, 41}}) {
    std::vector<double> entries(rows * cols);
    std::uint64_t       state {(rows * 100) + cols};
    for (double& x : entries) {
      x = next_uniform(state) - 0.5;
    }
    const matrix<double> a        {rows, cols, entries};
    const auto           reduced  {numerical::qr_decomposition<double> {a}};
    const auto           complete {numerical::qr_decomposition<double> {a, numerical::qr_mode::complete}};
    const std::size_t    k        {std::min(rows, cols)};
    EXPECT(complete.q().rows() == rows && complete.q().cols() == rows);
    EXPECT(complete.r().rows() == rows && complete.r().cols() == cols);
    EXPECT(orthonormality_error(complete.q()) < 1e-13);
    EXPECT(approx_equal(complete.q() * complete.r(), a, 1e-12));
    bool same {true};
    for (std::size_t i {0}; i < rows; ++i) {
      for (std::size_t j {0}; j < k; ++j) {
        same = same && complete.q()(i, j) == reduced.q()(i, j);
      }
      for (std::size_t j {0}; j < cols; ++j) {
        same = same && complete.r()(i, j) == (i < k ? reduced.r()(i, j) : 0.0);
      }
    }
    EXPECT(same);
  }
}

// Full (NumPy's full_matrices=True): U is m × m and V n × n, orthonormal, U Σ Vᴴ = A with Σ padded to m × n; the thin
// factors are their first columns, and the singular values the same, to the bit, by every method.
TEST(svd_full_extends_the_thin_factors)
{
  for (const auto method : {numerical::svd_method::divide_and_conquer, numerical::svd_method::bidiagonal_qr,
                            numerical::svd_method::jacobi}) {
    for (const auto& [rows, cols] : {std::pair<std::size_t, std::size_t> {9, 4}, {4, 9}, {6, 6}, {150, 70}}) {
      std::vector<double> entries(rows * cols);
      std::uint64_t       state {(rows * 1000) + cols};
      for (double& x : entries) {
        x = next_uniform(state) - 0.5;
      }
      const matrix<double>                       a    {rows, cols, entries};
      const numerical::svd_decomposition<double> thin {a, method};
      const numerical::svd_decomposition<double> full {a, method, true};
      const std::size_t                          k    {std::min(rows, cols)};
      EXPECT(full.u().rows() == rows && full.u().cols() == rows);
      EXPECT(full.v().rows() == cols && full.v().cols() == cols);
      EXPECT(orthonormality_error(full.u()) < 1e-12 && orthonormality_error(full.v()) < 1e-12);
      EXPECT(full.singular_values() == thin.singular_values());
      std::vector<double> padded(rows * cols, 0.0);
      for (std::size_t i {0}; i < k; ++i) {
        padded[(i * cols) + i] = full.singular_values()[i];
      }
      const matrix<double> sigma {rows, cols, padded};
      EXPECT(approx_equal(full.u() * sigma * full.vh(), a, 1e-11));
      bool same                  {true};
      for (std::size_t j {0}; j < k; ++j) {
        for (std::size_t i {0}; i < rows; ++i) {
          same = same && full.u()(i, j) == thin.u()(i, j);
        }
        for (std::size_t i {0}; i < cols; ++i) {
          same = same && full.v()(i, j) == thin.v()(i, j);
        }
      }
      EXPECT(same);
    }
  }
}

// Past one panel: the diagonal blocks, the rows below them by leaves (or, under many rows, as one leaf), and
// the trailing products.
TEST(blocked_cholesky_equals_the_unblocked_algorithm_to_the_bit)
{
  (void)numerical::limit_gemm_path(numerical::gemm_path::portable);
  EXPECT(blocked_cholesky_equals_unblocked<double>((2 * numerical::detail::cholesky_block) + 22));
  EXPECT(blocked_cholesky_equals_unblocked<double>((9 * numerical::detail::cholesky_block) + 5));
  EXPECT(blocked_cholesky_equals_unblocked<std::complex<double>>(numerical::detail::cholesky_block + 40));
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
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

// A few right-hand sides go row by row, each row's terms in the blocked passes' order: the solution of the first
// sixteen columns is the blocked solve's of seventeen to the bit, past one block of rows, on every path whose
// products fuse as the rows do (the portable path, fusing neither; AVX2 and AVX-512, both, where the rows go so at
// any order) and at orders whose blocked products all run the matrix kernel (192 is whole blocks, 200 a last
// block deep enough); off the x86-64 units the rows go so at any order too.
TEST(narrow_solve_equals_the_blocked_solve_to_the_bit)
{
  const numerical::lu_decomposition<double> small {matrix<double> {{2.0, 1.0}, {1.0, 3.0}}};
  const matrix<double>                      none  {small.solve(matrix<double> {2, 0, std::vector<double> {}})};
  EXPECT(none.rows() == 2U && none.cols() == 0U); // no right-hand side: nothing to solve, by either road
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::avx2, numerical::gemm_path::avx512}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue;
    }
    for (const std::size_t n : {std::size_t {192}, std::size_t {200}}) {
      const std::size_t                         wide {numerical::detail::narrow_solve + 1};
      const numerical::lu_decomposition<double> lu   {matrix<double> {n, n, pivoting_entries<double>(n)}};
      const std::vector<double>                 rhs  {pivoting_entries<double>(n)};
      const matrix<double>                      b    {n, wide, std::vector<double>(rhs.begin(), rhs.begin() + static_cast<std::ptrdiff_t>(n * wide))};
      std::vector<double>                       first(n * numerical::detail::narrow_solve);
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t c {0}; c < numerical::detail::narrow_solve; ++c) {
          first[(i * numerical::detail::narrow_solve) + c] = b(i, c);
        }
      }
      const matrix<double> blocked {lu.solve(b)};
      const matrix<double> narrow  {lu.solve(matrix<double> {n, numerical::detail::narrow_solve, first})};
      bool                 same    {true};
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t c {0}; c < numerical::detail::narrow_solve; ++c) {
          same = same && narrow(i, c) == blocked(i, c);
        }
      }
      EXPECT(same);
    }
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

// Many right-hand sides go to the threads by column slices, each through every block of rows: the same bits as
// one slice on one thread, whether the slices are capped by the threads (two) or by the columns (all threads).
TEST(lu_solves_column_slices_to_the_same_bits)
{
  const std::size_t    n {300};
  const matrix<double> a {n, n, pivoting_entries<double>(n)};
  std::vector<double>  rhs(n * 200);
  for (std::size_t i {0}; i < rhs.size(); ++i) {
    rhs[i] = (static_cast<double>((i * 7919U) % 1000U) / 500.0) - 1.0;
  }
  const matrix<double>                       b  {n, 200, rhs};
  const numerical::lu_decomposition<double>  lu {a};
  numerical::set_max_threads(1);
  const matrix<double> inverse_one              {lu.inverse()};
  const matrix<double> solve_one                {lu.solve(b)};
  for (const std::size_t threads : {std::size_t {2}, std::size_t {0}}) {
    numerical::set_max_threads(threads);
    EXPECT(lu.inverse() == inverse_one);
    EXPECT(lu.solve(b) == solve_one);
  }
  numerical::set_max_threads(0);
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
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
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
  const numerical::buffer<double> d               {-8.122808264515302e-13, 6.04530122336367e-12, 5.219248898251512e-06};
  const numerical::buffer<double> z               {4.2711462701151146e-10, 0.48318236791802455, 0.8755197309787659};
  numerical::buffer<double>       delta(3, double {});
  (void)numerical::detail::secular_root(d, z, 283.3625584139991, 0, numerical::detail::secular_iterations, delta.data());
  EXPECT(std::abs(delta[0] - -5.3584091596269844e-30) < 1e-14 * 5.4e-30);
  EXPECT(std::abs(delta[1] - 6.8575820498152e-12) < 1e-14 * 6.9e-12);
  const numerical::buffer<double> d6 {-0.8949247533967253, -0.16141637105958684, -0.0008046271488207429,
                                      -1.6560228279009447e-08, -6.869457203538978e-09, 0.9963928059764295};
  const numerical::buffer<double> z6 {-0.005944540988224872, -0.589814085218663, -0.016380413260955785,
                                      0.45405424328337657, 0.6675689734376589, -0.0014488724105117632};
  std::vector<double>       delta6(6);
  const double              root {numerical::detail::secular_root(d6, z6, 1.0448293818862207, 4, numerical::detail::secular_iterations,
                                                                  delta6.data())};
  EXPECT(std::abs(root - 0.9932959321048296) < 1e-14); // to a few rounding errors of the equation's terms
}

// The secular equations' iterations are bounded: one step does not reach a root.
TEST(the_secular_equation_bounds_its_iterations)
{
  const numerical::buffer<double> d               {0.0, 1.0, 2.0};
  const numerical::buffer<double> z               {0.6, 0.48, 0.64};
  numerical::buffer<double>       delta(3, double {});
  EXPECT_THROWS((void)numerical::detail::secular_root(d, z, 1.0, 0, 1, delta.data()), numerical::convergence_error);
  numerical::buffer<double> above(3, double       {}); // the singular values' equation likewise
  EXPECT_THROWS((void)numerical::detail::singular_secular_root(d, z, 0, 1, delta.data(), above.data()), numerical::convergence_error);
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
  numerical::buffer<double> d(n, 0.0);
  numerical::buffer<double> e(n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    d[i] = i % 5 == 2 ? 0.0 : std::cos(static_cast<double>(i));
    e[i] = i + 1 < n ? 1.0 / static_cast<double>(i + 2) : 0.0;
  }
  numerical::buffer<double> sigma;
  numerical::buffer<double> u;
  numerical::buffer<double> v;
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
  const std::size_t         r            {70};
  numerical::buffer<double> rd(r, double {});
  numerical::buffer<double> re(r, double {});
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
  // A zero matrix past the leaves: every arrow is zero (no weight anywhere, nothing to solve), its singular
  // values zeros and its vectors orthonormal.
  const svd_decomposition<double> zero {matrix<double> {70, 70, std::vector<double>(4900, 0.0)}};
  EXPECT(std::ranges::all_of(zero.singular_values(), [](double value) { return value == 0.0; }));
  EXPECT(orthonormality_error(zero.u()) < 1e-15 && orthonormality_error(zero.v()) < 1e-15);
  // Complex, already bidiagonal (no reflection needed on either side), and with no columns at all.
  const matrix<c>            bidiagonal {{c {1.0, 1.0}, c {0.0, 2.0}, c {}}, {c {}, c {3.0, 0.0}, c {1.0, -1.0}}, {c {}, c {}, c {0.0, -2.0}}};
  const svd_decomposition<c> two_sided  {bidiagonal};
  EXPECT(approx_equal(reconstruct(two_sided), bidiagonal, 1e-14));
  const svd_decomposition<c> none       {matrix<c> {3, 0, {}}};
  EXPECT(none.singular_values().size() == 0 && none.u().rows() == 3 && none.u().cols() == 0);
}

// Subnormal entries: a Householder vector or a phase computed among subnormal numbers is off unit length by the
// coarse grid's error, which no input scaling removes when one column alone is tiny. Every reflector and phase is
// scaled up first, exactly: the bases stay orthonormal.
TEST(subnormal_columns_keep_the_bases_orthonormal)
{
  using c = std::complex<double>;
  const std::size_t   n {40};
  std::vector<double> sym(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    sym[(i * n) + i] = 1.0 + (0.01 * static_cast<double>(i));
  }
  for (std::size_t i {1}; i < n; ++i) {
    sym[i * n] = sym[i] = 1e-320 * static_cast<double>((i % 7) + 1);
  }
  const symmetric_eigen tiny_column {matrix<double> {n, n, sym}};
  EXPECT(orthonormality_error(tiny_column.vectors()) < 1e-13);
  EXPECT(approx_equal(reconstruct(tiny_column), matrix<double> {n, n, sym}, 1e-13));
  std::vector<double> small(36);
  for (std::size_t i {0}; i < small.size(); ++i) {
    small[i] = 1e-318 * std::sin(static_cast<double>(i + 1));
  }
  const auto factored {numerical::qr(matrix<double> {6, 6, small})};
  EXPECT(orthonormality_error(factored.q()) < 1e-14);
  std::vector<double> bidiagonal(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    bidiagonal[(i * n) + i] = 1.0 + (0.01 * static_cast<double>(i));
    if (i > 0) {
      bidiagonal[i * n] = 1e-320 * static_cast<double>(i);
    }
  }
  const svd_decomposition<double> tiny_rows {matrix<double> {n, n, bidiagonal}};
  EXPECT(orthonormality_error(tiny_rows.u()) < 1e-13 && orthonormality_error(tiny_rows.v()) < 1e-13);
  std::vector<c> hermitian(n * n);
  for (std::size_t i {0}; i < n; ++i) {
    hermitian[(i * n) + i] = c {1.0 + (0.01 * static_cast<double>(i)), 0.0};
    if (i + 1 < n) {
      hermitian[((i + 1) * n) + i] = c {1e-321, 1e-321} *static_cast<double>((i % 3) + 1);
      hermitian[(i * n) + i + 1]   = std::conj(hermitian[((i + 1) * n) + i]);
    }
  }
  const numerical::hermitian_eigen tiny_phases {matrix<c> {n, n, hermitian}};
  EXPECT(orthonormality_error(tiny_phases.vectors()) < 1e-13);
  std::vector<c> upper(n * n);
  for (std::size_t i {0}; i < n; ++i) {
    upper[(i * n) + i] = c {1.0 + (0.01 * static_cast<double>(i)), 0.0};
    if (i + 1 < n) {
      upper[(i * n) + i + 1] = c {1e-321, 1e-321};
    }
  }
  const svd_decomposition<c> complex_tiny {matrix<c> {n, n, upper}};
  EXPECT(orthonormality_error(complex_tiny.v()) < 1e-13 && orthonormality_error(complex_tiny.u()) < 1e-13);
}

// eigh reads the lower triangle and the diagonal's real part, as numpy.linalg.eigh does: what lies elsewhere may
// be anything, a NaN included; a non-finite entry it reads is refused.
TEST(eigh_checks_only_what_it_reads)
{
  using c        = std::complex<double>;
  const double         nan       {std::numeric_limits<double>::quiet_NaN()};
  const matrix<double> upper_nan {{1.0, nan}, {2.0, 1.0}};
  EXPECT(approx_equal(symmetric_eigen {upper_nan}.values(), vector<double> {-1.0, 3.0}, 1e-14));
  EXPECT_THROWS(symmetric_eigen(matrix<double> {{1.0, 0.0}, {nan, 1.0}}), numerical::non_finite_error);
  EXPECT_THROWS(symmetric_eigen(matrix<double> {{1.0, 0.0}, {0.0, nan}}), numerical::non_finite_error);
  const matrix<c> imaginary_nan {{c {1.0, nan}, c {nan, nan}}, {c {2.0, 0.0}, c {1.0, 0.0}}};
  EXPECT(approx_equal(numerical::hermitian_eigen {imaginary_nan}.values(), vector<double> {-1.0, 3.0}, 1e-14));
  EXPECT_THROWS(numerical::hermitian_eigen(matrix<c> {{c {1.0, 0.0}, c {}}, {c {nan, 0.0}, c {1.0, 0.0}}}),
                numerical::non_finite_error);
}

// A norm summed plainly where its squares neither overflow nor underflow, and at scale where they would: the same
// norm, to rounding, from 1e-300 to 1e300, real and complex, and for a type summed at scale alone.
TEST(scaled_norm_is_exact_to_rounding_at_every_scale)
{
  for (const double size : {1e-300, 1e-160, 1.0, 1e160, 1e300}) {
    numerical::buffer<double>               real(37, double                  {});
    numerical::buffer<std::complex<double>> complex(37, std::complex<double> {});
    numerical::buffer<long double>          wide(37, 0.0L);
    double                                  want                             {0.0}; // Σ (3 + i mod 5)², at unit size
    for (std::size_t i {0}; i < real.size(); ++i) {
      const double unit {3.0 + static_cast<double>(i % 5)};
      real[i]     = unit * size;
      complex[i]  = std::complex<double> {unit * 0.6, unit * 0.8} *size;
      wide[i]     = static_cast<long double>(unit * size);
      want       += unit * unit;
    }
    want = std::sqrt(want) * size;
    EXPECT(std::abs(numerical::detail::scaled_norm(real) - want) <= 1e-14 * want);
    EXPECT(std::abs(numerical::detail::scaled_norm(complex) - want) <= 1e-14 * want);
    EXPECT(std::abs(numerical::detail::scaled_norm(wide) - want) <= 1e-14 * want);
  }
  EXPECT(numerical::detail::scaled_norm(numerical::buffer<double>(3, 0.0)) == 0.0);
}

// Divide and conquer called on a tridiagonal far from unit size: it works on the tridiagonal scaled to it, by an
// exact power of two, so its deflation tolerance stays relative.
TEST(divide_and_conquer_is_relative_to_the_tridiagonal_size)
{
  for (const double size : {1e-15, 1e15}) {
    const std::size_t         n {80};
    std::vector<double>       d(n);
    numerical::buffer<double> e(n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      d[i] = size * std::cos(static_cast<double>(i));
      e[i] = size * (i + 1 < n ? 1.0 / static_cast<double>(i + 2) : 0.0);
    }
    numerical::buffer<double> values(d.begin(), d.end());
    numerical::buffer<double> q;
    numerical::detail::divide_and_conquer(values.data(), e.data(), n, q);
    double worst {0.0}; // ‖T q − λ q‖ over ‖T‖
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t i {0}; i < n; ++i) {
        const double tq {(d[i] * q[(i * n) + j]) + (i > 0 ? e[i - 1] * q[((i - 1) * n) + j] : 0.0)
                         + (i + 1 < n ? e[i] * q[((i + 1) * n) + j] : 0.0)};
        worst = std::max(worst, std::abs(tq - (values[j] * q[(i * n) + j])) / size);
      }
    }
    EXPECT(worst < 1e-13);
  }
}

// The symmetric product of the tridiagonal reduction on every path this processor runs (its row kernel there, the
// portable one's weight off it): the same decomposition to rounding, A V = V Λ.
TEST(eigh_runs_every_kernel_path)
{
  const std::size_t   n {48};
  std::vector<double> entries(n * n);
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j <= i; ++j) {
      entries[(i * n) + j] = entries[(j * n) + i] = std::sin(static_cast<double>((i * 31) + (j * 17)));
    }
  }
  const matrix<double> a {n, n, entries};
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue;
    }
    const numerical::symmetric_eigen eigen {a};
    double                           worst {0.0};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        double av {0.0};
        for (std::size_t k {0}; k < n; ++k) {
          av += a(i, k) * eigen.vectors()(k, j);
        }
        worst = std::max(worst, std::abs(av - (eigen.vectors()(i, j) * eigen.values()[j])));
      }
    }
    EXPECT(worst < 1e-12 * static_cast<double>(n));
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

// A graded symmetric matrix, scaled to its largest entry yet spanning 1e-200 to 1: a QL shift's ratio reaches 5e199,
// whose square overflows, so a rotation's length is taken without squaring there. Both iterations find every
// eigenvalue, A V = V Λ to rounding of the largest.
TEST(eigh_of_a_graded_matrix)
{
  const matrix<double> two {{1e-200, 1e-200}, {1e-200, 1.0}};
  for (const numerical::eigen_method method : {numerical::eigen_method::tridiagonal_ql, numerical::eigen_method::divide_and_conquer}) {
    const numerical::symmetric_eigen eigen {two, method};
    EXPECT(std::abs(eigen.values()[0] - 1e-200) <= 1e-214 && std::abs(eigen.values()[1] - 1.0) <= 1e-15);
  }
  for (const std::size_t n : {std::size_t {5}, std::size_t {40}}) {
    std::vector<double> entries(n * n);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j <= i; ++j) {
        const double scale {std::pow(10.0, -200.0 * static_cast<double>(n - 1 - i) / static_cast<double>(n - 1))
                            * std::pow(10.0, -200.0 * static_cast<double>(n - 1 - j) / static_cast<double>(n - 1))};
        entries[(i * n) + j] = entries[(j * n) + i] = scale * (i == j ? 2.0 : std::sin(static_cast<double>((i * 7) + j)));
      }
    }
    const matrix<double> a {n, n, entries};
    for (const numerical::eigen_method method : {numerical::eigen_method::tridiagonal_ql, numerical::eigen_method::divide_and_conquer}) {
      const numerical::symmetric_eigen eigen {a, method};
      double                           worst {0.0};
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          double av {0.0};
          for (std::size_t k {0}; k < n; ++k) {
            av += a(i, k) * eigen.vectors()(k, j);
          }
          worst = std::max(worst, std::abs(av - (eigen.vectors()(i, j) * eigen.values()[j])));
        }
      }
      EXPECT(std::isfinite(worst) && worst < 1e-12 * static_cast<double>(n));
    }
  }
}

// The singular values without the vectors are the decomposition's, for every shape, a complex matrix, a zero one, and
// one far past the range where squares overflow; the rank follows from them.
TEST(symmetric_eigenvalues_alone_are_the_decompositions)
{
  // Fixed seed: the matrices are part of the test, so a failure names the same one every run.
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64                  generator {11};
  std::normal_distribution<double> normal;
  for (const std::size_t n : {std::size_t {1}, std::size_t {3}, std::size_t {40}, std::size_t {150}}) {
    std::vector<double>               real(n * n);
    std::vector<std::complex<double>> complex(n * n);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j <= i; ++j) {
        real[(i * n) + j]    = normal(generator);
        real[(j * n) + i]    = real[(i * n) + j];
        complex[(i * n) + j] = i == j ? std::complex<double> {normal(generator), 0.0} : std::complex<double> {normal(generator), normal(generator)};
        complex[(j * n) + i] = std::conj(complex[(i * n) + j]);
      }
    }
    const matrix<double>               a(n, n, real);
    const matrix<std::complex<double>> h(n, n, complex);
    const vector<double>               alone           {numerical::symmetric_eigenvalues(a)};
    const vector<double>               hermitian_alone {numerical::symmetric_eigenvalues(h)};
    const numerical::symmetric_eigen   full            {a};
    const numerical::hermitian_eigen   hermitian_full  {h};
    for (std::size_t i {0}; i < n; ++i) {
      EXPECT(std::abs(alone[i] - full.values()[i]) <= 1e-12 * static_cast<double>(n));
      EXPECT(std::abs(hermitian_alone[i] - hermitian_full.values()[i]) <= 1e-12 * static_cast<double>(n));
      EXPECT(i == 0 || alone[i - 1] <= alone[i]);
    }
  }
  EXPECT(numerical::symmetric_eigenvalues(matrix<double>(0, 0, std::vector<double> {})).size() == 0);
  const vector<double> scaled {numerical::symmetric_eigenvalues(matrix<double> {{2e200, 0.0}, {0.0, -3e200}})};
  EXPECT(std::abs(scaled[0] + 3e200) <= 1e-14 * 3e200 && std::abs(scaled[1] - 2e200) <= 1e-14 * 3e200);
  const vector<double> zero   {numerical::symmetric_eigenvalues(matrix<double> {{0.0, 0.0}, {0.0, 0.0}})};
  EXPECT(zero[0] == 0.0 && zero[1] == 0.0);
  EXPECT_THROWS(numerical::symmetric_eigenvalues(matrix<double> {{1.0, 2.0, 3.0}}), numerical::dimension_error);
  EXPECT_THROWS(numerical::symmetric_eigenvalues(matrix<double> {{1.0, 0.0}, {std::nan(""), 1.0}}), numerical::non_finite_error);
}

TEST(singular_values_alone_are_the_decompositions)
{
  // Fixed seed: the matrices are part of the test, so a failure names the same one every run.
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64                  generator {7};
  std::normal_distribution<double> normal;
  for (const auto& [rows, cols] : std::vector<std::pair<std::size_t, std::size_t>> {{1, 1}, {4, 4}, {9, 4}, {4, 9},
         {150, 150}, {200, 140}, {140, 200}}) {
    std::vector<double> entries(rows * cols);
    for (double& x : entries) {
      x = normal(generator);
    }
    const matrix<double>                       a(rows, cols, entries);
    const vector<double>                       alone         {numerical::singular_values(a)};
    const numerical::svd_decomposition<double> decomposition {a};
    const vector<double>&                      full          {decomposition.singular_values()};
    EXPECT(alone.size() == std::min(rows, cols));
    for (std::size_t i {0}; i < alone.size(); ++i) {
      EXPECT(std::abs(alone[i] - full[i]) <= 1e-13 * full[0]);
    }
  }
  const matrix<std::complex<double>>                       c                     {{{1.0, 2.0}, {0.0, -1.0}}, {{3.0, 0.5}, {2.0, 2.0}}, {{-1.0, 1.0}, {0.5, 0.0}}};
  const vector<double>                                     complex_alone         {numerical::singular_values(c)};
  const numerical::svd_decomposition<std::complex<double>> complex_decomposition {c};
  const vector<double>&                                    complex_full          {complex_decomposition.singular_values()};
  EXPECT(std::abs(complex_alone[0] - complex_full[0]) <= 1e-13 * complex_full[0]
         && std::abs(complex_alone[1] - complex_full[1]) <= 1e-13 * complex_full[0]);
  const vector<double> zero {numerical::singular_values(matrix<double> {{0.0, 0.0}, {0.0, 0.0}})};
  EXPECT(zero[0] == 0.0 && zero[1] == 0.0);
  // Already bidiagonal, a zero inside its diagonal: the iteration splits there, and with no basis records nothing.
  const matrix<double> split       {{1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}}; // σ = √2, √2, 0
  const vector<double> split_alone {numerical::singular_values(split)};
  EXPECT(std::abs(split_alone[0] - std::sqrt(2.0)) <= 1e-15 && std::abs(split_alone[1] - std::sqrt(2.0)) <= 1e-15
         && std::abs(split_alone[2]) <= 1e-15);
  EXPECT(split_alone == numerical::svd_decomposition<double> {split}.singular_values());
  const vector<double> huge {numerical::singular_values(matrix<double> {{3e200, 0.0}, {0.0, 4e200}})};
  EXPECT(std::abs(huge[0] - 4e200) <= 1e-14 * 4e200 && std::abs(huge[1] - 3e200) <= 1e-14 * 4e200);
  EXPECT_EQ(numerical::matrix_rank(matrix<double> {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}}), 1U);
  EXPECT_THROWS((void)numerical::singular_values(matrix<double> {{1.0, std::nan("")}}), numerical::non_finite_error);
}

// least_squares adds NumPy's residuals, rank and singular values to lstsq's solution: the residual sums of squares for a
// tall matrix of full column rank, one per right-hand side, and none for a square, wide or rank-deficient one.
TEST(least_squares_reports_what_numpy_lstsq_reports)
{
  const matrix<double>                            tall  {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
  const matrix<double>                            b     {{1.0, 2.0}, {1.0, 0.0}, {1.0, 1.0}};
  const numerical::least_squares_solution<double> found {numerical::least_squares(tall, b)};
  EXPECT(approx_equal(found.solution, numerical::lstsq(tall, b)));
  EXPECT_EQ(found.rank, 2U);
  EXPECT(found.residuals.size() == 2);
  for (std::size_t c {0}; c < 2; ++c) { // ‖b − A x‖² per column, from the solution
    double sum {0.0};
    for (std::size_t i {0}; i < 3; ++i) {
      const double r {b(i, c) - ((tall(i, 0) * found.solution(0, c)) + (tall(i, 1) * found.solution(1, c)))};
      sum += r * r;
    }
    EXPECT(std::abs(found.residuals[c] - sum) <= 1e-14);
  }
  EXPECT(approx_equal(found.singular_values, numerical::singular_values(tall)));
  // From the QR's R of A over a power of two, scaled back: a matrix far from 1 keeps its singular values.
  const matrix<double> large        {{3e10, 0.0}, {0.0, 4e10}, {0.0, 0.0}};
  const vector<double> large_values {numerical::least_squares(large, matrix<double> {{1.0}, {1.0}, {1.0}}).singular_values};
  EXPECT(large_values[0] == 4e10 && large_values[1] == 3e10);
  EXPECT(numerical::least_squares(matrix<double> {{2.0, 1.0}, {1.0, 3.0}}, matrix<double> {{1.0}, {2.0}}).residuals.size() == 0);
  EXPECT(numerical::least_squares(matrix<double> {{1.0, 2.0, 3.0}}, matrix<double> {{1.0}}).residuals.size() == 0);
  const numerical::least_squares_solution<double> deficient {
    numerical::least_squares(matrix<double> {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}}, matrix<double> {{1.0}, {0.0}, {1.0}})};
  EXPECT(deficient.rank == 1 && deficient.residuals.size() == 0);
  const matrix<std::complex<double>>                            c             {{{1.0, 1.0}}, {{0.0, 2.0}}, {{1.0, 0.0}}};
  const numerical::least_squares_solution<std::complex<double>> complex_found {
    numerical::least_squares(c, matrix<std::complex<double>> {{{1.0, 0.0}}, {{0.0, 0.0}}, {{0.0, 1.0}}})};
  EXPECT(complex_found.rank == 1 && complex_found.residuals.size() == 1 && complex_found.residuals[0] > 0.0);
  EXPECT_THROWS((void)numerical::least_squares(tall, matrix<double> {{1.0}}), numerical::dimension_error);
}

// A matrix without rows or columns decomposes into empty factors of NumPy's shapes, each method, thin or full: 0 × 0
// read the largest of no band of rows.
TEST(svd_of_empty_matrices_has_empty_factors)
{
  for (const auto method : {numerical::svd_method::divide_and_conquer, numerical::svd_method::bidiagonal_qr,
                            numerical::svd_method::jacobi}) {
    for (const auto& [rows, cols] : {std::pair<std::size_t, std::size_t> {0, 0}, {0, 3}, {3, 0}}) {
      for (const bool full : {false, true}) {
        const matrix<double>                       a   {rows, cols, std::vector<double> {}};
        const numerical::svd_decomposition<double> svd {a, method, full};
        const std::size_t                          k   {std::min(rows, cols)};
        EXPECT(svd.singular_values().size() == 0);
        EXPECT(svd.u().rows() == rows && svd.u().cols() == (full ? rows : k));
        EXPECT(svd.v().rows() == cols && svd.v().cols() == (full ? cols : k));
      }
    }
  }
}

// A decomposition of an expiring matrix, scaled in its own storage, gives the bits of the decomposition of a constant
// one: svd of tall, wide and square matrices by each method (complex too), symmetric_eigen by each method (its upper
// triangle holding values it must not read), eigen_decomposition with its vectors on both sides.
template <typename T>
numerical::matrix<T> drawn_matrix(std::size_t rows,
                                  std::size_t cols,
                                  unsigned    seed)
{
  std::mt19937_64     generator {seed};
  std::vector<double> parts(2 * rows * cols);
  for (double& part : parts) {
    part = std::ldexp(static_cast<double>(generator() >> 11U), -53) - 0.5;
  }
  std::vector<T> entries(rows * cols);
  for (std::size_t e {0}; e < entries.size(); ++e) {
    if constexpr (numerical::detail::split_complex<T>) {
      entries[e] = T {parts[2 * e], parts[(2 * e) + 1]};
    }
    else {
      entries[e] = parts[2 * e];
    }
  }
  return numerical::matrix<T> {rows, cols, entries};
}

template <typename T>
bool expiring_svd_has_the_bits_of_a_constant_one()
{
  bool all {true};
  for (const auto& [rows, cols] : {std::pair<std::size_t, std::size_t> {9, 5}, {5, 9}, {7, 7}}) {
    for (const numerical::svd_method method : {numerical::svd_method::divide_and_conquer, numerical::svd_method::bidiagonal_qr, numerical::svd_method::jacobi}) {
      for (const bool full : {false, true}) {
        const numerical::matrix<T>            a        {drawn_matrix<T>(rows, cols, 7U)};
        const numerical::svd_decomposition<T> constant {a, method, full};
        const numerical::svd_decomposition<T> expiring {drawn_matrix<T>(rows, cols, 7U), method, full};
        all = all && constant.singular_values() == expiring.singular_values() && constant.u() == expiring.u() && constant.v() == expiring.v();
      }
    }
  }
  return all;
}

TEST(a_decomposition_of_an_expiring_matrix_has_the_bits_of_a_constant_one)
{
  EXPECT(expiring_svd_has_the_bits_of_a_constant_one<double>());
  EXPECT(expiring_svd_has_the_bits_of_a_constant_one<std::complex<double>>());
  constexpr std::size_t           order             {11};
  std::vector<double>             lower(order * order);
  const numerical::matrix<double> drawn             {drawn_matrix<double>(order, order, 3U)};
  for (std::size_t i {0}; i < order; ++i) {
    for (std::size_t j {0}; j < order; ++j) {
      lower[(i * order) + j] = j <= i ? drawn(i, j) : 1e300; // above the diagonal: never read
    }
  }
  const numerical::matrix<double> symmetric {order, order, lower};
  for (const numerical::eigen_method method : {numerical::eigen_method::divide_and_conquer, numerical::eigen_method::tridiagonal_ql, numerical::eigen_method::jacobi}) {
    const numerical::symmetric_eigen constant {symmetric, method};
    const numerical::symmetric_eigen expiring {numerical::matrix<double> {symmetric}, method};
    EXPECT(constant.values() == expiring.values() && constant.vectors() == expiring.vectors());
  }
  const numerical::matrix<double>                   square                    {drawn_matrix<double>(13, 13, 5U)};
  const numerical::eigen_decomposition<double>      constant                  {square, true, true};
  const numerical::eigen_decomposition<double>      expiring                  {numerical::matrix<double> {square}, true, true};
  EXPECT(constant.values() == expiring.values() && constant.vectors() == expiring.vectors() && constant.left_vectors() == expiring.left_vectors());
  const numerical::matrix<std::complex<double>>              complex_square   {drawn_matrix<std::complex<double>>(6, 6, 9U)};
  const numerical::eigen_decomposition<std::complex<double>> complex_constant {complex_square};
  const numerical::eigen_decomposition<std::complex<double>> complex_expiring {numerical::matrix<std::complex<double>> {complex_square}};
  EXPECT(complex_constant.values() == complex_expiring.values() && complex_constant.vectors() == complex_expiring.vectors());
}

// Divide and conquer on Wilkinson's W61 (diagonal |−30|…|30|, ones beside it): its residual ‖T V − V Λ‖_F within
// 0.15 n ε ‖T‖, LAPACK's dstevd's level (0.098); leaves of 32 rows by QL left it at 0.22, those of 25 (dstedc's
// smlsiz) at 0.10.
TEST(divide_and_conquer_holds_lapacks_residual_on_wilkinsons_w61)
{
  constexpr std::size_t n {61};
  std::vector<double>   entries(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    entries[(i * n) + i] = std::abs(static_cast<double>(i) - 30.0);
    if (i + 1 < n) {
      entries[(i * n) + i + 1]   = 1.0;
      entries[((i + 1) * n) + i] = 1.0;
    }
  }
  const numerical::matrix<double>  t       {n, n, entries};
  const numerical::symmetric_eigen e       {t};
  long double                      squares {0.0L};
  for (std::size_t j {0}; j < n; ++j) {
    for (std::size_t i {0}; i < n; ++i) {
      // (d_i − λ) v_i first, its difference exact where v_i is large (λ near d_i), then the neighbours times 1.
      long double r {(static_cast<long double>(t(i, i)) - e.values()[j]) * e.vectors()(i, j)};
      if (i > 0) {
        r += e.vectors()(i - 1, j);
      }
      if (i + 1 < n) {
        r += e.vectors()(i + 1, j);
      }
      squares += r * r;
    }
  }
  const long double norm {32.0L}; // the largest row sum: 30 + 1 + 1
  EXPECT(std::sqrt(squares) / (norm * n * std::numeric_limits<double>::epsilon()) < 0.15L);
}
