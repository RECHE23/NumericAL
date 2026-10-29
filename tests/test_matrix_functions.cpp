// expm against closed forms: a diagonal, a rotation, a nilpotent block, a scalar times the identity, an upper triangular
// matrix with equal and distinct diagonal entries, and a symmetric matrix of large norm through its eigenvalues; each
// degree of the approximant and the squarings, the backward error bound's extra squarings, the refusals.

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
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

TEST(expm_refuses_what_it_cannot_take)
{
  EXPECT_THROWS(static_cast<void>(numerical::expm(matrix<double> {{1.0, 2.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm(matrix<double> {{std::numeric_limits<double>::quiet_NaN()}})), numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::expm(matrix<C> {{C {std::numeric_limits<double>::infinity()}}})), numerical::non_finite_error);
}
