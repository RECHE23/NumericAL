// expm against closed forms: a diagonal, a rotation, a nilpotent block, a scalar times the identity, an upper triangular
// matrix with equal and distinct diagonal entries, and a symmetric matrix of large norm through its eigenvalues; each
// degree of the approximant and the squarings, the backward error bound's extra squarings, the refusals. The Fréchet
// derivative of the exponential and its condition number, Sylvester's equation, the square root, and funm.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::matrix;
  using C = std::complex<double>;

  //! \brief The largest modulus of `x − y` over that of `y` (or 1).
  template <typename T>
  double relative_gap(const matrix<T>& x,
                      const matrix<T>& y)
  {
    double gap  {0.0};
    double size {0.0};
    for (std::size_t i {0}; i < y.rows(); ++i) {
      for (std::size_t j {0}; j < y.cols(); ++j) {
        gap  = std::max(gap, std::abs(x(i, j) - y(i, j)));
        size = std::max(size, std::abs(y(i, j)));
      }
    }
    return gap / std::max(size, 1.0);
  }
} // namespace

// The degrees by the norm of the powers: 1e-3 (3), 0.1 (5), 0.8 (7), 2 (9), 5 and 40 (13, squared); a diagonal matrix's
// exponential its entries'.
TEST(expm_of_a_diagonal_matrix_is_its_entries_exponentials)
{
  for (const double scale : {1e-3, 0.1, 0.8, 2.0, 5.0, 40.0}) {
    const std::vector<double> d {scale, -0.5 * scale, 0.25 * scale};
    std::vector<double>       a(9, 0.0);
    std::vector<double>       e(9, 0.0);
    for (std::size_t i {0}; i < 3; ++i) {
      a[(i * 3) + i] = d[i];
      e[(i * 3) + i] = std::exp(d[i]);
    }
    EXPECT(relative_gap(numerical::expm(matrix<double> {3, 3, a}), matrix<double> {3, 3, e}) < 1e-14);
  }
}

// A rotation's generator gives the rotation, a nilpotent block 1 + N, a complex scalar times the identity e^z I; the
// zero matrix the identity, an empty one an empty one.
TEST(expm_of_closed_forms)
{
  for (const double angle : {0.3, 2.0, 25.0}) {
    const matrix<double> generator {{0.0, -angle}, {angle, 0.0}};
    const matrix<double> rotation  {{std::cos(angle), -std::sin(angle)}, {std::sin(angle), std::cos(angle)}};
    EXPECT(relative_gap(numerical::expm(generator), rotation) < 1e-13);
  }
  const matrix<double> nilpotent {{0.0, 1.0}, {0.0, 0.0}};
  EXPECT((numerical::expm(nilpotent) == matrix<double> {{1.0, 1.0}, {0.0, 1.0}}));
  const C         z              {0.5, 3.0};
  const matrix<C> scalar         {{z, C {}}, {C {}, z}};
  EXPECT(relative_gap(numerical::expm(scalar), matrix<C> {{std::exp(z), C {}}, {C {}, std::exp(z)}}) < 1e-15);
  EXPECT((numerical::expm(matrix<double> {2, 2, std::vector<double>(4, 0.0)}) == matrix<double> {{1.0, 0.0}, {0.0, 1.0}}));
  EXPECT(numerical::expm(matrix<double> {0, 0, std::vector<double> {}}).rows() == 0);
}

// An upper triangular matrix: its diagonal and first superdiagonal written exactly at each squaring, e^a and
// b e^{(a + c)/2} sinch((a − c)/2), equal diagonal entries (the series of sinch) and distinct ones.
TEST(expm_of_an_upper_triangular_matrix)
{
  for (const auto& [a, c] : {std::pair {3.0, 3.0}, std::pair {12.0, -7.0}, std::pair {0.01, 0.01}}) {
    const double         b      {5.0};
    const matrix<double> t      {{a, b}, {0.0, c}};
    const double         corner {a == c ? b * std::exp(a) : b * (std::exp(a) - std::exp(c)) / (a - c)};
    const matrix<double> e      {{std::exp(a), corner}, {0.0, std::exp(c)}};
    EXPECT(relative_gap(numerical::expm(t), e) < 1e-14);
  }
  const matrix<C> complex_t {{C {1.0, 2.0}, C {3.0}}, {C {}, C {-2.0, 1.0}}};
  const C         a         {1.0, 2.0};
  const C         c         {-2.0, 1.0};
  const matrix<C> e         {{std::exp(a), C {3.0} *(std::exp(a) - std::exp(c)) / (a - c)}, {C {}, std::exp(c)}};
  EXPECT(relative_gap(numerical::expm(complex_t), e) < 1e-14);
}

// A symmetric matrix of norm some 30, through its eigenvalues: V e^Λ Vᵀ.
TEST(expm_of_a_symmetric_matrix_through_its_eigenvalues)
{
  const std::size_t   n {12};
  std::vector<double> entries(n * n);
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j <= i; ++j) {
      entries[(i * n) + j] = entries[(j * n) + i] = 4.0 * std::sin(1.0 + static_cast<double>((i * n) + j));
    }
  }
  const matrix<double>             a {n, n, entries};
  const numerical::symmetric_eigen e {a};
  std::vector<double>              expected(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t k {0}; k < n; ++k) {
        expected[(i * n) + j] += e.vectors()(i, k) * std::exp(e.values()[k]) * e.vectors()(j, k);
      }
    }
  }
  EXPECT(relative_gap(numerical::expm(a), matrix<double> {n, n, expected}) < 1e-13);
}

// A matrix whose powers are small but far from normal: c [[1, 1], [−1, −1]] squares to 0, so the norms of its powers
// admit the lowest degree, which the backward error bound (‖ |A|²ᵐ⁺¹ ‖) refuses, then the squarings it adds: e^A = I + A
// to rounding, where the lowest degree alone left 5e-10 of it at c = 10⁴. [[δ, b], [0, −δ]] likewise.
TEST(expm_adds_the_squarings_the_backward_error_bound_asks)
{
  const double         c         {1e4};
  const matrix<double> nilpotent {{c, c}, {-c, -c}};
  EXPECT(relative_gap(numerical::expm(nilpotent), matrix<double> {{1.0 + c, c}, {-c, 1.0 - c}}) < 1e-13);
  const double         delta     {1e-3};
  const double         b         {1e3};
  const matrix<double> a         {{delta, b}, {0.0, -delta}};
  const double         corner    {b * (std::exp(delta) - std::exp(-delta)) / (2.0 * delta)};
  const matrix<double> e         {{std::exp(delta), corner}, {0.0, std::exp(-delta)}};
  EXPECT(relative_gap(numerical::expm(a), e) < 1e-14);
  const matrix<double> full      {{delta, b, 0.0}, {0.0, -delta, 0.0}, {1e-3, 0.0, delta}}; // not triangular: squared whole
  const matrix<double> image     {numerical::expm(full)};
  EXPECT(std::abs(image(0, 1) - corner) < 1e-11 * corner);
}

// The degree-13 cases: η₃ ≤ θ₁₃ (3 I, no squaring whatever ‖A¹⁰‖), d₈ ≥ d₆ (6 I: every dₚ is 6) and d₈ < d₆ with η₃ past
// θ₁₃ (5 I plus a nilpotent 200, ‖Aᵖ‖₁ = 5ᵖ (1 + 40 p)), where A¹⁰ is formed for η₅.
TEST(expm_forms_the_tenth_power_only_where_it_decides)
{
  for (const double c : {3.0, 6.0}) {
    EXPECT(relative_gap(numerical::expm(matrix<double> {{c, 0.0}, {0.0, c}}), matrix<double> {{std::exp(c), 0.0}, {0.0, std::exp(c)}}) < 1e-14);
  }
  const double e5 {std::exp(5.0)};
  EXPECT(relative_gap(numerical::expm(matrix<double> {{5.0, 200.0}, {0.0, 5.0}}), matrix<double> {{e5, 200.0 * e5}, {0.0, e5}}) < 1e-14);
}

TEST(expm_refuses_what_it_cannot_take)
{
  EXPECT_THROWS(static_cast<void>(numerical::expm(matrix<double> {{1.0, 2.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm(matrix<double> {{std::numeric_limits<double>::quiet_NaN()}})), numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm(matrix<C> {{C {std::numeric_limits<double>::infinity()}}})), numerical::non_finite_error);
}

namespace {

  //! \brief `‖A X + X B − Q‖ / (‖A‖ ‖X‖ + ‖X‖ ‖B‖ + ‖Q‖)` (largest moduli) for \p x = solve_sylvester(a, b, q).
  template <typename T>
  double sylvester_residual(const matrix<T>& a,
                            const matrix<T>& b,
                            const matrix<T>& q,
                            const matrix<T>& x)
  {
    const matrix<T> r {(a * x) + (x * b)};
    const auto      largest = [](const matrix<T>& m) {
                                double value {0.0};
                                for (std::size_t i {0}; i < m.rows(); ++i) {
                                  for (std::size_t j {0}; j < m.cols(); ++j) {
                                    value = std::max(value, std::abs(m(i, j)));
                                  }
                                }
                                return value;
                              };
    return largest(r - q) / ((largest(a) * largest(x)) + (largest(x) * largest(b)) + largest(q));
  }

  //! \brief An `rows × cols` matrix of `sin` and `cos` of its entries' index, offset by \p seed.
  template <typename T>
  matrix<T> trigonometric(std::size_t rows,
                          std::size_t cols,
                          double      seed)
  {
    std::vector<T> entries(rows * cols);
    for (std::size_t e {0}; e < entries.size(); ++e) {
      const double angle {seed + (0.7 * static_cast<double>(e))};
      if constexpr (std::is_same_v<T, double>) {
        entries[e] = std::sin(angle);
      }
      else {
        entries[e] = T {std::sin(angle), std::cos(1.3 * angle)};
      }
    }
    return matrix<T> {rows, cols, entries};
  }
} // namespace

// A X + X B = Q, as scipy.linalg.solve_sylvester: real matrices (their Schur forms with 2 × 2 blocks, R's and S's) and
// complex ones, square and not, to rounding; a scalar equation; A and −B sharing an eigenvalue, a pivot perturbed as
// LAPACK's trsyl and the answer large but finite; empty operands; the refusals.
TEST(solve_sylvester_solves_the_equation)
{
  for (const auto& [m, n] : {std::pair<std::size_t, std::size_t> {1, 1}, {2, 3}, {7, 5}, {30, 40}}) {
    const matrix<double> a {trigonometric<double>(m, m, 1.0)};
    const matrix<double> b {trigonometric<double>(n, n, 2.0)};
    const matrix<double> q {trigonometric<double>(m, n, 3.0)};
    EXPECT(sylvester_residual(a, b, q, numerical::solve_sylvester(a, b, q)) < 1e-14);
    const matrix<C> ac     {trigonometric<C>(m, m, 1.0)};
    const matrix<C> bc     {trigonometric<C>(n, n, 2.0)};
    const matrix<C> qc     {trigonometric<C>(m, n, 3.0)};
    EXPECT(sylvester_residual(ac, bc, qc, numerical::solve_sylvester(ac, bc, qc)) < 1e-14);
  }
  EXPECT((numerical::solve_sylvester(matrix<double> {{2.0}}, matrix<double> {{3.0}}, matrix<double> {{10.0}}) == matrix<double> {{2.0}}));
  const matrix<double> shared {numerical::solve_sylvester(matrix<double> {{1.0}}, matrix<double> {{-1.0}}, matrix<double> {{1.0}})};
  EXPECT(std::isfinite(shared(0, 0)) && std::abs(shared(0, 0)) > 1e12);
  EXPECT(numerical::solve_sylvester(matrix<double> {0, 0, std::vector<double> {}}, matrix<double> {{1.0}}, matrix<double> {0, 1, std::vector<double> {}}).rows() == 0);
  EXPECT(numerical::solve_sylvester(matrix<double> {{1.0}}, matrix<double> {0, 0, std::vector<double> {}}, matrix<double> {1, 0, std::vector<double> {}}).cols() == 0);
  EXPECT_THROWS(static_cast<void>(numerical::solve_sylvester(matrix<double> {{1.0, 2.0}}, matrix<double> {{1.0}}, matrix<double> {{1.0}})),
                numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_sylvester(matrix<double> {{1.0}}, matrix<double> {{1.0}}, matrix<double> {{1.0, 2.0}})),
                numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_sylvester(matrix<double> {{1.0}}, matrix<double> {{1.0, 2.0}}, matrix<double> {{1.0}})),
                numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_sylvester(matrix<double> {{1.0}}, matrix<double> {{1.0}}, matrix<double> {{1.0}, {2.0}})),
                numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_sylvester(matrix<double> {{1.0}}, matrix<double> {{std::numeric_limits<double>::quiet_NaN()}},
                                                             matrix<double> {{1.0}})),
                numerical::non_finite_error);
}

// The principal square root, as scipy.linalg.sqrtm: X² = A for a symmetric positive-definite matrix, a general real
// one with complex pairs (a rotation's root the half rotation), a complex one; a real eigenvalue below 0 refused in
// real arithmetic and its root +i √|λ| in complex; no root (a zero in a Jordan block) infinite there, the zero matrix's
// zero, an upper triangular matrix its own Schur form, an empty one empty; the refusals.
TEST(sqrtm_squares_back)
{
  const auto squares = []<typename T>(const matrix<T>& a, const matrix<T>& x) { return relative_gap(x * x, a) < 1e-13; };
  for (const std::size_t n : {std::size_t {1}, std::size_t {5}, std::size_t {30}}) {
    const matrix<double> m {trigonometric<double>(n, n, 1.0)};
    std::vector<double>  eye(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      eye[(i * n) + i] = 1.0;
    }
    const matrix<double> identity {n, n, eye};
    const matrix<double> spd      {(m * numerical::transpose(m)) + identity};
    EXPECT(squares(spd, numerical::sqrtm(spd)));
    const matrix<double> shifted  {m + (3.0 * identity)}; // complex pairs, none below 0
    EXPECT(squares(shifted, numerical::sqrtm(shifted)));
    std::vector<C> shift(n * n);
    for (std::size_t i {0}; i < n; ++i) {
      shift[(i * n) + i] = C {3.0};
    }
    const matrix<C> complex {trigonometric<C>(n, n, 2.0) + matrix<C> {n, n, shift}}; // away from singular (its root ill-conditioned there)
    EXPECT(squares(complex, numerical::sqrtm(complex)));
  }
  const double         half     {std::sqrt(0.5)};
  const matrix<double> rotation {{0.0, -1.0}, {1.0, 0.0}};
  EXPECT(relative_gap(numerical::sqrtm(rotation), matrix<double> {{half, -half}, {half, half}}) < 1e-15);
  const matrix<double> negative {{-4.0, 1.0}, {0.0, -9.0}};
  EXPECT_THROWS(static_cast<void>(numerical::sqrtm(negative)), std::domain_error);
  EXPECT((numerical::complex_sqrtm(negative) == matrix<C> {{C {0.0, 2.0}, C {0.0, -0.2}}, {C {}, C {0.0, 3.0}}}));
  const matrix<double> pair_below {{-1.0, 2.0, 0.5}, {-3.0, -1.0, 1.0}, {0.0, 0.0, -2.0}}; // a pair and -2 (rsf2csf)
  const matrix<C>      root       {numerical::complex_sqrtm(pair_below)};
  std::vector<C>       entries(9);
  for (std::size_t i {0}; i < 3; ++i) {
    for (std::size_t j {0}; j < 3; ++j) {
      entries[(i * 3) + j] = C {pair_below(i, j)};
    }
  }
  EXPECT(relative_gap(root * root, matrix<C> {3, 3, entries}) < 1e-14);
  const matrix<double> jordan {numerical::sqrtm(matrix<double> {{0.0, 1.0}, {0.0, 0.0}})};
  EXPECT(jordan(0, 0) == 0.0 && std::isinf(jordan(0, 1)) && jordan(1, 0) == 0.0 && jordan(1, 1) == 0.0);
  EXPECT((numerical::sqrtm(matrix<double> {2, 2, std::vector<double>(4, 0.0)}) == matrix<double> {2, 2, std::vector<double>(4, 0.0)}));
  EXPECT((numerical::sqrtm(matrix<double> {{4.0, 0.0}, {0.0, 9.0}}) == matrix<double> {{2.0, 0.0}, {0.0, 3.0}}));
  EXPECT((numerical::sqrtm(matrix<C> {{C {-1.0}}}) == matrix<C> {{C {0.0, 1.0}}}));
  EXPECT(numerical::sqrtm(matrix<double> {0, 0, std::vector<double> {}}).rows() == 0);
  EXPECT(numerical::complex_sqrtm(matrix<double> {0, 0, std::vector<double> {}}).rows() == 0);
  EXPECT_THROWS(static_cast<void>(numerical::sqrtm(matrix<double> {{1.0, 2.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::sqrtm(matrix<C> {{C {1.0}, C {2.0}}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::complex_sqrtm(matrix<double> {{1.0, 2.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::sqrtm(matrix<double> {{std::numeric_limits<double>::infinity()}})), numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::sqrtm(matrix<C> {{C {std::numeric_limits<double>::quiet_NaN()}}})), numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::complex_sqrtm(matrix<double> {{std::numeric_limits<double>::quiet_NaN()}})), numerical::non_finite_error);
}

namespace {

  //! \brief Whether `expm_frechet(a, e)` holds for \p a and \p e (`n × n`): `expm(a)` beside it, a central difference
  //!        of expm within \p tolerance, and the block matrix's exponential the same.
  template <typename T>
  bool frechet_holds(const matrix<T>& a,
                     const matrix<T>& e,
                     double           tolerance)
  {
    const auto [x, l] {numerical::expm_frechet(a, e)};
    double     largest {1.0};
    for (const T& v : a) {
      largest = std::max(largest, std::abs(v));
    }
    const double    h          {1e-6 * largest};
    const matrix<T> difference {T {1.0 / (2.0 * h)} *(numerical::expm(a + (T {h} *e)) - numerical::expm(a - (T {h} *e)))};
    const auto [xb, lb] {numerical::expm_frechet(a, e, numerical::frechet_method::block_enlarge)};
    return relative_gap(x, numerical::expm(a)) < 1e-13 && relative_gap(l, difference) < tolerance && relative_gap(xb, x) < 1e-12
           && relative_gap(lb, l) < 1e-11;
  }
} // namespace

// The Fréchet derivative of the exponential, as scipy.linalg.expm_frechet: a central difference of expm, real and
// complex, at norms taking each degree (3, 5, 7, 9) and 13 without and with squarings; expm itself beside it; the
// block matrix's exponential the same; empty operands; the refusals.
TEST(expm_frechet_is_the_derivative)
{
  for (const double scale : {1e-3, 0.03, 0.15, 0.35, 1.0, 20.0}) {
    EXPECT(frechet_holds(scale * trigonometric<double>(4, 4, 1.0), trigonometric<double>(4, 4, 2.0), 1e-5));
    EXPECT(frechet_holds(C {scale} *trigonometric<C>(4, 4, 1.0), trigonometric<C>(4, 4, 2.0), 1e-5));
  }
  const matrix<C> empty_c {0, 0, std::vector<C> {}};
  EXPECT(numerical::expm_frechet(empty_c, empty_c).second.rows() == 0);
  EXPECT_THROWS(static_cast<void>(numerical::expm_frechet(matrix<C> {{C {1.0}, C {2.0}}}, matrix<C> {{C {1.0}, C {2.0}}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm_frechet(matrix<C> {{C {1.0}}}, matrix<C> {{C {1.0}, C {2.0}}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm_frechet(matrix<C> {{C {1.0}}}, matrix<C> {{C {1.0}}, {C {2.0}}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm_frechet(matrix<C> {{C {1.0}}}, matrix<C> {{C {std::numeric_limits<double>::quiet_NaN()}}})),
                numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm_frechet(matrix<C> {{C {std::numeric_limits<double>::infinity()}}}, matrix<C> {{C {1.0}}})),
                numerical::non_finite_error);
  const matrix<double> empty {0, 0, std::vector<double> {}};
  EXPECT(numerical::expm_frechet(empty, empty).second.rows() == 0);
  EXPECT_THROWS(static_cast<void>(numerical::expm_frechet(matrix<double> {{1.0, 2.0}}, matrix<double> {{1.0, 2.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm_frechet(matrix<double> {{1.0}}, matrix<double> {{1.0, 2.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm_frechet(matrix<double> {{1.0}}, matrix<double> {{1.0}, {2.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm_frechet(matrix<double> {{1.0}}, matrix<double> {{std::numeric_limits<double>::quiet_NaN()}})),
                numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm_frechet(matrix<double> {{std::numeric_limits<double>::infinity()}}, matrix<double> {{1.0}})),
                numerical::non_finite_error);
}

// The exponential's condition number, as scipy.linalg.expm_cond: a scalar's |a|; a diagonal matrix's from the divided
// differences of e^x (K diagonal, its entries (e^a − e^b) / (a − b) and e^a); complex; an empty matrix's 0; refusals.
TEST(expm_cond_of_closed_forms)
{
  EXPECT(std::abs(numerical::expm_cond(matrix<double> {{2.0}}) - 2.0) < 1e-14);
  EXPECT(std::abs(numerical::expm_cond(matrix<C> {{C {0.0, 3.0}}}) - 3.0) < 1e-14);
  const double gap {(std::exp(2.0) - std::numbers::e) / 1.0};
  const double k {std::max({std::numbers::e, std::exp(2.0), gap})};
  const double expected {k * std::sqrt(5.0) / std::sqrt(std::exp(2.0) + std::exp(4.0))};
  EXPECT(std::abs(numerical::expm_cond(matrix<double> {{1.0, 0.0}, {0.0, 2.0}}) - expected) < 1e-13 * expected);
  EXPECT(numerical::expm_cond(matrix<double> {0, 0, std::vector<double> {}}) == 0.0);
  EXPECT_THROWS(static_cast<void>(numerical::expm_cond(matrix<double> {{1.0, 2.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm_cond(matrix<double> {{std::numeric_limits<double>::quiet_NaN()}})), numerical::non_finite_error);
}

// A function of a matrix by its Schur form and Parlett's recurrence, as scipy.linalg.funm: x² is A² (SciPy's example),
// e^x is expm with a small error estimate, a complex matrix's; a repeated eigenvalue (a zero denominator, left
// undivided as SciPy's) estimated at 1; an infinite value at infinity; an empty matrix; the refusals, f's count of values
// among them.
TEST(funm_by_parlett)
{
  using values = std::vector<C>;
  const auto square = [](const values& x) {
                        values y(x.size());
                        std::ranges::transform(x, y.begin(), [](const C& z) { return z * z; });
                        return y;
                      };
  const auto exponential = [](const values& x) {
                             values y(x.size());
                             std::ranges::transform(x, y.begin(), [](const C& z) { return std::exp(z); });
                             return y;
                           };
  const matrix<double>         a       {{1.0, 3.0}, {1.0, 4.0}};
  const numerical::funm_result squared {numerical::funm(a, square)};
  EXPECT(relative_gap(squared.f, numerical::detail::as_complex(a * a)) < 1e-14 && squared.error < 1e-14);
  const matrix<double>         b       {trigonometric<double>(6, 6, 1.0) + matrix<double> {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 2.0, 0.0, 0.0, 0.0, 0.0},
                                          {0.0, 0.0, 3.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 4.0, 0.0, 0.0},
                                          {0.0, 0.0, 0.0, 0.0, 5.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 6.0}}};
  const numerical::funm_result exp_b {numerical::funm(b, exponential)};
  EXPECT(relative_gap(exp_b.f, numerical::detail::as_complex(numerical::expm(b))) < 1e3 * exp_b.error && exp_b.error < 1e-12);
  const matrix<C>              c {trigonometric<C>(5, 5, 2.0)};
  EXPECT(relative_gap(numerical::funm(c, exponential).f, numerical::expm(c)) < 1e-10);
  const numerical::funm_result jordan {numerical::funm(matrix<double> {{2.0, 1.0}, {0.0, 2.0}}, exponential)};
  EXPECT(jordan.error == 1.0);
  const auto                   reciprocal = [](const values& x) {
                                              values y(x.size());
                                              std::ranges::transform(x, y.begin(), [](const C& z) { return C {1.0} / z.real(); });
                                              return y;
                                            };
  EXPECT(std::isinf(numerical::funm(matrix<double> {{0.0, 1.0}, {0.0, 3.0}}, reciprocal).error));
  EXPECT(numerical::funm(matrix<double> {0, 0, std::vector<double> {}}, exponential).f.rows() == 0);
  EXPECT_THROWS(static_cast<void>(numerical::funm(matrix<double> {{1.0, 2.0}}, exponential)), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::funm(matrix<C> {{C {std::numeric_limits<double>::quiet_NaN()}}}, exponential)), numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::funm(matrix<double> {{1.0}}, [](const values&) { return values {}; })), numerical::dimension_error);
  const matrix<C> one {{C {1.0}}};
  EXPECT_THROWS(static_cast<void>(numerical::funm_from_schur(matrix<C> {{C {1.0}, C {2.0}}}, one, values {C {1.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::funm_from_schur(one, matrix<C> {{C {1.0}, C {2.0}}}, values {C {1.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::funm_from_schur(one, matrix<C> {{C {1.0}}, {C {2.0}}}, values {C {1.0}})), numerical::dimension_error);
}

namespace {

  //! \brief Whether `expm(logm(a))` is \p a within \p tolerance (relative to its largest entry).
  template <typename T>
  bool logm_inverts(const matrix<T>& a,
                    double           tolerance)
  {
    const numerical::logm_result r {numerical::logm(a)};
    return relative_gap(numerical::expm(r.log), numerical::detail::as_complex(a)) < tolerance && !r.exactly_singular && !r.nearly_singular;
  }
} // namespace

// The principal logarithm, as scipy.linalg.logm: expm undoes it for real and complex matrices of every scale (the
// square roots and the Padé degrees each takes), real ones with complex pairs and with negative eigenvalues (a complex
// logarithm), upper triangular ones (no Schur form), near the identity (degrees 1 and 2); closed forms: a Jordan block
// (equal eigenvalues), -I (iπ I), eigenvalues close together and across the negative real axis (the unwinding number);
// zero and tiny eigenvalues reported; the refusals.
TEST(logm_inverts_expm)
{
  const matrix<double> identity3 {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  for (const double scale : {1e-9, 1e-4, 1e-2, 0.1, 0.5, 2.0, 10.0, 1e3}) {
    for (const std::size_t n : {std::size_t {1}, std::size_t {3}, std::size_t {8}}) {
      matrix<double>      a {trigonometric<double>(n, n, 1.0)};
      std::vector<double> shifted(a.begin(), a.end());
      for (std::size_t i {0}; i < n; ++i) {
        shifted[(i * n) + i] += 2.0 + static_cast<double>(i);
      }
      const matrix<double> b {scale * matrix<double> {n, n, shifted}};
      EXPECT(logm_inverts(b, 1e-12));
    }
    EXPECT(logm_inverts(identity3 + (scale * trigonometric<double>(3, 3, 4.0)), 1e-10));
    EXPECT(logm_inverts(C {scale} *(trigonometric<C>(4, 4, 2.0) + C {0.0, 1.0} *numerical::detail::as_complex(trigonometric<double>(4, 4, 3.0))), 1e-11));
  }
  const matrix<double> rotation   {{0.0, -2.0}, {2.0, 0.0}}; // eigenvalues ±2i
  EXPECT(logm_inverts(rotation, 1e-14));
  const matrix<double> negative   {{-2.0, 1.0, 0.0}, {0.5, -3.0, 0.2}, {0.0, 0.1, 4.0}};
  EXPECT(logm_inverts(negative, 1e-13));
  const matrix<double> triangular {{2.0, 1.0, 3.0}, {0.0, 3.0, 1.0}, {0.0, 0.0, 0.5}};
  EXPECT(logm_inverts(triangular, 1e-14));
  EXPECT(logm_inverts(matrix<double> {{-2.0, 1.0}, {0.0, 3.0}}, 1e-14));
  const numerical::logm_result jordan {numerical::logm(matrix<double> {{2.0, 1.0}, {0.0, 2.0}})};
  EXPECT(relative_gap(jordan.log, matrix<C> {{C {std::log(2.0)}, C {0.5}}, {C {}, C {std::log(2.0)}}}) < 1e-15);
  const numerical::logm_result minus  {numerical::logm(matrix<double> {{-1.0, 0.0}, {0.0, -1.0}})};
  EXPECT(relative_gap(minus.log, matrix<C> {{C {0.0, std::numbers::pi}, C {}}, {C {}, C {0.0, std::numbers::pi}}}) < 1e-15);
  EXPECT(logm_inverts(matrix<double> {{1.0, 1.0}, {0.0, 1.0001}}, 1e-14));
  const C l1 {-1.0, 0.01};
  const C l2 {-1.0, -0.01};
  EXPECT(logm_inverts(matrix<C> {{l1, C {1.0}}, {C {}, l2}}, 1e-13));
  EXPECT(logm_inverts(matrix<C> {{C {-1.0, 0.5}, C {2.0}}, {C {}, C {-1.0, 0.5}}}, 1e-13));
  const numerical::logm_result zero {numerical::logm(matrix<double> {{0.0, 1.0}, {0.0, 1.0}})};
  EXPECT(zero.exactly_singular && !zero.nearly_singular);
  const numerical::logm_result tiny {numerical::logm(matrix<double> {{1e-25, 0.0}, {0.0, 1.0}})};
  EXPECT(tiny.nearly_singular && !tiny.exactly_singular);
  EXPECT(numerical::logm(matrix<double> {0, 0, std::vector<double> {}}).log.rows() == 0);
  EXPECT_THROWS(static_cast<void>(numerical::logm(matrix<double> {{1.0, 2.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::logm(matrix<C> {{C {std::numeric_limits<double>::quiet_NaN()}}})), numerical::non_finite_error);
}

// The moduli the norms of expm take: √(x² + y²) where the largest part is within [2⁻⁵⁰⁰, 2⁵⁰⁰], hypot outside it, where
// a square would overflow (3e300 + 4e300 i, 5e300) or vanish (3e-310 + 4e-310 i, 5e-310); a real entry's |x|.
TEST(expm_moduli_keep_to_the_range_of_their_squares)
{
  using numerical::buffer;
  EXPECT((numerical::detail::entry_moduli(buffer<C> {C {3.0, 4.0}, C {0.0, -1.0}}) == buffer<double> {5.0, 1.0}));
  EXPECT((numerical::detail::entry_moduli(buffer<C> {C {3e300, 4e300}, C {1.0}}) == buffer<double> {std::abs(C {3e300, 4e300}), 1.0}));
  EXPECT(numerical::detail::entry_moduli(buffer<C> {C {3e-310, 4e-310}})[0] == std::abs(C {3e-310, 4e-310}));
  EXPECT(numerical::detail::entry_moduli(buffer<C> {C {3e-310, 4e-310}})[0] > 0.0);
  EXPECT((numerical::detail::entry_moduli(buffer<double> {-2.0, 3.0}) == buffer<double> {2.0, 3.0}));
  EXPECT(numerical::detail::one_norm(buffer<C> {C {3.0, 4.0}, C {}, C {0.0, 1.0}, C {1.0}}, 2) == 6.0);
}

// A real form's 2 × 2 blocks scaled before their eigenvalue (b·c past √DBL_MAX overflowed, the rotation NaN): logm and
// the complex square root of matrices from 10⁻³⁰⁰ to 10³⁰⁰ with a pair of complex eigenvalues; log(s R) = ln s I + π/2 R
// for the rotation R by a quarter turn. A triangular form whose square roots are not finite refused, not iterated for
// ever.
TEST(logm_and_sqrtm_hold_at_extreme_scales)
{
  for (const double s : {1e-300, 1e-200, 1e150, 1e160, 1e200, 1e300}) {
    const matrix<double> rotation {{0.0, -s}, {s, 0.0}};
    const double         half_pi  {std::numbers::pi / 2.0};
    const matrix<C>      want     {{C {std::log(s)}, C {-half_pi}}, {C {half_pi}, C {std::log(s)}}};
    EXPECT(relative_gap(numerical::logm(rotation).log, want) < 1e-14);
    const matrix<double> pair     {{-s, 0.5 * s}, {-0.5 * s, -s}};
    const matrix<C>      root     {numerical::complex_sqrtm(pair)};
    EXPECT(relative_gap(root * root, numerical::detail::as_complex(pair)) < 1e-14);
  }
  const numerical::buffer<C> broken {C {4.0}, C {std::numeric_limits<double>::quiet_NaN()}, C {}, C {9.0}};
  EXPECT_THROWS(static_cast<void>(numerical::detail::inverse_squaring(broken, 2)), numerical::convergence_error);
}
