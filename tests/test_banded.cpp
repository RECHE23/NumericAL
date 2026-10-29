// solve_banded and solveh_banded against the dense solve of the same matrices: bands of every shape (none below or
// above the diagonal, one each, wider ones), real and complex, vectors and matrices of right-hand sides, pivoting and
// not; a singular band, a band not positive definite, the shapes refused.

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::matrix;
  using C = std::complex<double>;

  //! \brief A pseudo-random value in [-1, 1), the state advanced.
  double drawn(std::uint64_t& state)
  {
    state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
    return (static_cast<double>(state >> 11U) * 0x1.0p-52) - 1.0;
  }

  template <typename T>
  T drawn_scalar(std::uint64_t& state)
  {
    if constexpr (std::is_same_v<T, double>) {
      return drawn(state);
    }
    else {
      const double re {drawn(state)};
      return T {re, drawn(state)};
    }
  }

  //! \brief The largest modulus of `x − y` over that of `y`.
  template <typename T>
  double gap(const matrix<T>& x,
             const matrix<T>& y)
  {
    double difference {0.0};
    double size       {0.0};
    for (std::size_t i {0}; i < y.rows(); ++i) {
      for (std::size_t j {0}; j < y.cols(); ++j) {
        difference = std::max(difference, std::abs(x(i, j) - y(i, j)));
        size       = std::max(size, std::abs(y(i, j)));
      }
    }
    return difference / size;
  }

  //! \brief A band of \p lower and \p upper diagonals of order \p n solved by solve_banded and by the dense solve.
  template <typename T>
  bool band_agrees(std::size_t n,
                   std::size_t lower,
                   std::size_t upper,
                   std::size_t k,
                   bool        dominant)
  {
    std::uint64_t  state                           {(n * 131) + (lower * 17) + upper};
    std::vector<T> dense(n * n, T                  {});
    std::vector<T> band((lower + upper + 1) * n, T {std::numeric_limits<double>::max()}); // corners: not read
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {i > lower ? i - lower : 0}; j < std::min(n, i + upper + 1); ++j) {
        T value {drawn_scalar<T>(state)};
        if (dominant && i == j) {
          value += T {static_cast<double>(lower + upper + 2)};
        }
        dense[(i * n) + j]                   = value;
        band[((upper + i - j) * n) + j]      = value;
      }
    }
    std::vector<T> rhs(n * k);
    for (T& value : rhs) {
      value = drawn_scalar<T>(state);
    }
    const matrix<T> b {n, k, rhs};
    return gap(numerical::solve_banded(lower, upper, matrix<T> {lower + upper + 1, n, band}, b), numerical::solve(matrix<T> {n, n, dense}, b)) < 1e-12;
  }

  //! \brief A Hermitian positive-definite band of \p kd diagonals each side, by solveh_banded from its upper and its
  //!        lower form and by the dense solve.
  template <typename T>
  bool hermitian_band_agrees(std::size_t n,
                             std::size_t kd)
  {
    std::uint64_t  state                      {(n * 7) + kd};
    std::vector<T> dense(n * n, T             {});
    std::vector<T> upper_form((kd + 1) * n, T {1e300}); // corners: not read
    std::vector<T> lower_form((kd + 1) * n, T {1e300});
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {i > kd ? i - kd : 0}; j < i; ++j) {
        const T value {drawn_scalar<T>(state)};
        dense[(i * n) + j] = value;
        dense[(j * n) + i] = numerical::detail::conj_of(value);
      }
      dense[(i * n) + i] = T {static_cast<double>((4 * kd) + 2)};
    }
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t i {j > kd ? j - kd : 0}; i <= j; ++i) {
        upper_form[((kd + i - j) * n) + j] = dense[(i * n) + j];
      }
      for (std::size_t i {j}; i < std::min(n, j + kd + 1); ++i) {
        lower_form[((i - j) * n) + j] = dense[(i * n) + j];
      }
    }
    std::vector<T> rhs(n * 2);
    for (T& value : rhs) {
      value = drawn_scalar<T>(state);
    }
    const matrix<T> b {n, 2, rhs};
    const matrix<T> x {numerical::solve(matrix<T> {n, n, dense}, b)};
    return gap(numerical::solveh_banded(matrix<T> {kd + 1, n, upper_form}, b), x) < 1e-12
           && gap(numerical::solveh_banded(matrix<T> {kd + 1, n, lower_form}, b, true), x) < 1e-12;
  }
} // namespace

TEST(solve_banded_is_the_dense_solve)
{
  for (const auto& [n, lower, upper] : {std::array<std::size_t, 3> {1, 0, 0}, {5, 1, 2}, {30, 3, 1}, {40, 1, 1}, {12, 0, 2}, {12, 2, 0}, {60, 4, 5}}) {
    for (const bool dominant : {false, true}) { // dominant: no interchange; not: some
      EXPECT((band_agrees<double>(n, lower, upper, 1, dominant)));
      EXPECT((band_agrees<double>(n, lower, upper, 3, dominant)));
      EXPECT((band_agrees<C>(n, lower, upper, 2, dominant)));
    }
  }
  const matrix<double> example {{0, 0, -1, -1, -1}, {0, 2, 2, 2, 2}, {5, 4, 3, 2, 1}, {1, 1, 1, 1, 0}}; // SciPy's
  const matrix<double> x       {numerical::solve_banded(1, 2, example, matrix<double> {5, 1, std::vector<double> {0, 1, 2, 2, 3}})};
  EXPECT(std::abs(x(2, 0) + 4.0) < 1e-14);
  EXPECT_THROWS(static_cast<void>(numerical::solve_banded(1, 0, matrix<double> {{1.0, 0.0}, {1.0, 0.0}}, matrix<double> {2, 1, std::vector<double> {1.0, 1.0}})),
                numerical::singular_matrix_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_banded(1, 1, example, matrix<double> {5, 1, std::vector<double>(5, 1.0)})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_banded(1, 2, example, matrix<double> {4, 1, std::vector<double>(4, 1.0)})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_banded(1, 2, example, matrix<double> {5, 1, std::vector<double>(5, std::numeric_limits<double>::infinity())})),
                numerical::non_finite_error);
}

TEST(solveh_banded_is_the_dense_solve)
{
  for (const auto& [n, kd] : {std::pair<std::size_t, std::size_t> {1, 0}, {6, 2}, {30, 3}, {10, 1}, {5, 0}}) {
    EXPECT((hermitian_band_agrees<double>(n, kd)));
    EXPECT((hermitian_band_agrees<C>(n, kd)));
  }
  EXPECT_THROWS(static_cast<void>(numerical::solveh_banded(matrix<double> {{0.0, 1.0}, {1.0, -1.0}}, matrix<double> {2, 1, std::vector<double> {1.0, 1.0}})),
                numerical::not_positive_definite_error);
  EXPECT_THROWS(static_cast<void>(numerical::solveh_banded(matrix<double> {0, 3, std::vector<double> {}}, matrix<double> {3, 1, std::vector<double>(3, 1.0)})),
                numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solveh_banded(matrix<double> {{1.0, 1.0}}, matrix<double> {3, 1, std::vector<double>(3, 1.0)})), numerical::dimension_error);
}
