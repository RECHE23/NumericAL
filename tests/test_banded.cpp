// solve_banded and solveh_banded against the dense solve of the same matrices: bands of every shape (none below or
// above the diagonal, one each — LAPACK's gtsv and ptsv, as SciPy — wider ones), real and complex, vectors and matrices
// of right-hand sides, pivoting and not; a singular band, a band not positive definite, the shapes refused.

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
  for (const auto& [n, lower, upper] :
       {std::array<std::size_t, 3> {1, 0, 0}, {5, 1, 2}, {30, 3, 1}, {40, 1, 1}, {1, 1, 1}, {2, 1, 1}, {3, 1, 1}, {12, 0, 2}, {12, 2, 0}, {60, 4, 5}}) {
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
  // one sub- and one superdiagonal (gtsv): rows exchanged at the last step ([[1, 1], [2, 1]]) and before it; a zero
  // pivot with a zero below it, and one the elimination leaves last ([[1, 1], [1, 1]]); no row at all
  const matrix<double> two   {numerical::solve_banded(1, 1, matrix<double> {{0.0, 1.0}, {1.0, 1.0}, {2.0, 0.0}}, matrix<double> {2, 1, std::vector<double> {2.0, 3.0}})};
  EXPECT(std::abs(two(0, 0) - 1.0) < 1e-15 && std::abs(two(1, 0) - 1.0) < 1e-15);
  const matrix<double> three {numerical::solve_banded(1, 1, matrix<double> {{0.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 0.0}},
                                                      matrix<double> {3, 1, std::vector<double> {2.0, 4.0, 3.0}})};
  EXPECT(std::abs(three(0, 0) - 1.0) < 1e-15 && std::abs(three(1, 0) - 1.0) < 1e-15 && std::abs(three(2, 0) - 1.0) < 1e-15);
  EXPECT_THROWS(static_cast<void>(numerical::solve_banded(1, 1, matrix<double> {{0.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}}, matrix<double> {2, 1, std::vector<double> {1.0, 1.0}})),
                numerical::singular_matrix_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_banded(1, 1, matrix<double> {{0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}}, matrix<double> {2, 1, std::vector<double> {1.0, 1.0}})),
                numerical::singular_matrix_error);
  EXPECT(numerical::solve_banded(1, 1, matrix<double> {3, 0, std::vector<double> {}}, matrix<double> {0, 1, std::vector<double> {}}).rows() == 0);
  EXPECT_THROWS(static_cast<void>(numerical::solve_banded(1, 1, example, matrix<double> {5, 1, std::vector<double>(5, 1.0)})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_banded(1, 2, example, matrix<double> {4, 1, std::vector<double>(4, 1.0)})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_banded(1, 2, example, matrix<double> {5, 1, std::vector<double>(5, std::numeric_limits<double>::infinity())})),
                numerical::non_finite_error);
}

TEST(solveh_banded_is_the_dense_solve)
{
  for (const auto& [n, kd] : {std::pair<std::size_t, std::size_t> {1, 0}, {6, 2}, {30, 3}, {10, 1}, {1, 1}, {5, 0}}) {
    EXPECT((hermitian_band_agrees<double>(n, kd)));
    EXPECT((hermitian_band_agrees<C>(n, kd)));
  }
  EXPECT_THROWS(static_cast<void>(numerical::solveh_banded(matrix<double> {{0.0, 1.0}, {1.0, -1.0}}, matrix<double> {2, 1, std::vector<double> {1.0, 1.0}})),
                numerical::not_positive_definite_error);
  EXPECT(numerical::solveh_banded(matrix<double> {2, 0, std::vector<double> {}}, matrix<double> {0, 1, std::vector<double> {}}).rows() == 0);
  EXPECT_THROWS(static_cast<void>(numerical::solveh_banded(matrix<double> {0, 3, std::vector<double> {}}, matrix<double> {3, 1, std::vector<double>(3, 1.0)})),
                numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solveh_banded(matrix<double> {{1.0, 1.0}}, matrix<double> {3, 1, std::vector<double>(3, 1.0)})), numerical::dimension_error);
}

namespace {

  //! \brief A Toeplitz system of order \p n (first column and row drawn, the diagonal \p n past them) by
  //!        solve_toeplitz and by the dense solve.
  template <typename T>
  bool toeplitz_agrees(std::size_t n,
                       std::size_t k)
  {
    std::uint64_t  state {(n * 3) + k};
    std::vector<T> c(n);
    std::vector<T> r(n);
    for (std::size_t i {0}; i < n; ++i) {
      c[i] = drawn_scalar<T>(state);
      r[i] = drawn_scalar<T>(state);
    }
    c[0] += T {static_cast<double>(n)};
    r[0]  = T {1e300}; // not read
    std::vector<T> dense(n * n);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        dense[(i * n) + j] = i >= j ? c[i - j] : r[j - i];
      }
    }
    std::vector<T> rhs(n * k);
    for (T& value : rhs) {
      value = drawn_scalar<T>(state);
    }
    const matrix<T> b {n, k, rhs};
    return gap(numerical::solve_toeplitz(numerical::vector<T> {c}, numerical::vector<T> {r}, b), numerical::solve(matrix<T> {n, n, dense}, b)) < 1e-13;
  }

  //! \brief A circulant system of order \p n by solve_circulant and by the dense solve.
  template <typename T>
  bool circulant_agrees(std::size_t n)
  {
    std::uint64_t  state {(n * 5) + 1};
    std::vector<T> c(n);
    for (T& value : c) {
      value = drawn_scalar<T>(state);
    }
    c[0] += T {static_cast<double>(n)};
    std::vector<T> dense(n * n);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        dense[(i * n) + j] = c[(i + n - j) % n];
      }
    }
    std::vector<T> rhs(n * 2);
    for (T& value : rhs) {
      value = drawn_scalar<T>(state);
    }
    const matrix<T> b {n, 2, rhs};
    return gap(numerical::solve_circulant(numerical::vector<T> {c}, b), numerical::solve(matrix<T> {n, n, dense}, b)) < 1e-13;
  }
} // namespace

// Levinson's recursion against the dense solve, orders 1 to 40, real and complex, one and several right-hand sides;
// a singular leading minor refused (though T is not singular), at the first step and later; the refusals.
TEST(solve_toeplitz_is_the_dense_solve)
{
  for (const std::size_t n : {std::size_t {1}, std::size_t {2}, std::size_t {3}, std::size_t {7}, std::size_t {40}}) {
    EXPECT((toeplitz_agrees<double>(n, 1)));
    EXPECT((toeplitz_agrees<double>(n, 3)));
    EXPECT((toeplitz_agrees<C>(n, 2)));
  }
  const auto           vec = [](const std::vector<double>& values) { return numerical::vector<double> {values}; };
  const matrix<double> b  {2, 1, std::vector<double> {1.0, 2.0}};
  EXPECT_THROWS(static_cast<void>(numerical::solve_toeplitz(vec({0.0, 1.0}), vec({0.0, 1.0}), b)), numerical::singular_matrix_error);            // [[0, 1], [1, 0]]
  const matrix<double> b3 {3, 1, std::vector<double> {1.0, 2.0, 3.0}};
  EXPECT_THROWS(static_cast<void>(numerical::solve_toeplitz(vec({1.0, 1.0, 0.5}), vec({1.0, 1.0, 2.0}), b3)), numerical::singular_matrix_error); // [[1, 1], [1, 1]] leads
  EXPECT_THROWS(static_cast<void>(numerical::solve_toeplitz(vec({-2.0, 0.5, 1.0, -2.0}), vec({-2.0, 2.0, 0.0, 1.0}), matrix<double> {4, 1, std::vector<double>(4, 1.0)})),
                numerical::singular_matrix_error);                                                                                               // a backward denominator of zero
  EXPECT(numerical::solve_toeplitz(vec({}), vec({}), matrix<double> {0, 2, std::vector<double> {}}).cols() == 2);
  EXPECT_THROWS(static_cast<void>(numerical::solve_toeplitz(vec({1.0, 2.0}), vec({1.0}), b)), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_toeplitz(vec({1.0}), vec({1.0}), b)), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_toeplitz(vec({1.0, std::numeric_limits<double>::quiet_NaN()}), vec({1.0, 2.0}), b)), numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_toeplitz(vec({1.0, 2.0}), vec({1.0, std::numeric_limits<double>::infinity()}), b)), numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_toeplitz(vec({1.0, 2.0}), vec({1.0, 2.0}), matrix<double> {2, 1, std::vector<double> {1.0, std::numeric_limits<double>::quiet_NaN()}})),
                numerical::non_finite_error);
}

// The circulant system by the Fourier transform against the dense solve; a singular circulant refused, or its least
// squares (SciPy's example), a tolerance given; real inputs giving a real answer; the refusals.
TEST(solve_circulant_is_the_dense_solve)
{
  for (const std::size_t n : {std::size_t {1}, std::size_t {2}, std::size_t {7}, std::size_t {40}}) {
    EXPECT(circulant_agrees<double>(n));
    EXPECT(circulant_agrees<C>(n));
  }
  const numerical::vector<double> singular {std::vector<double> {1.0, 1.0, 0.0, 0.0}};
  const matrix<double>            b        {4, 1, std::vector<double> {1.0, 2.0, 3.0, 4.0}};
  EXPECT_THROWS(static_cast<void>(numerical::solve_circulant(singular, b)), numerical::singular_matrix_error);
  const matrix<double> least               {numerical::solve_circulant(singular, b, true)};
  EXPECT(gap(least, matrix<double> {4, 1, std::vector<double> {0.25, 1.25, 2.25, 1.25}}) < 1e-14);
  const matrix<double> loose               {numerical::solve_circulant(numerical::vector<double> {std::vector<double> {2.0, 2.0, 5.0}}, matrix<double> {3, 1, std::vector<double> {1.0, 2.0, 3.0}}, true, 10.0)};
  EXPECT(loose(0, 0) == 0.0 && loose(1, 0) == 0.0); // every eigenvalue under the tolerance: all left out
  EXPECT(numerical::solve_circulant(numerical::vector<double> {std::vector<double> {}}, matrix<double> {0, 1, std::vector<double> {}}).rows() == 0);
  EXPECT(numerical::solve_circulant(numerical::vector<double> {std::vector<double> {1.0}}, matrix<double> {1, 0, std::vector<double> {}}).cols() == 0);
  EXPECT_THROWS(static_cast<void>(numerical::solve_circulant(singular, matrix<double> {3, 1, std::vector<double>(3, 1.0)})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_circulant(numerical::vector<double> {std::vector<double> {1.0, std::numeric_limits<double>::infinity()}}, matrix<double> {2, 1, std::vector<double>(2, 1.0)})),
                numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_circulant(numerical::vector<double> {std::vector<double> {1.0, 2.0}}, matrix<double> {2, 1, std::vector<double> {1.0, std::numeric_limits<double>::quiet_NaN()}})),
                numerical::non_finite_error);
}

// The structured matrices as scipy.linalg forms them: a Toeplitz matrix of column and row (r[0] not read), square and
// not; a Hankel matrix (its last row from r's second entry); a circulant; a block diagonal matrix, an empty block
// among them and none at all.
TEST(structured_matrices_are_scipys)
{
  const numerical::vector<double> c {std::vector<double> {1.0, 2.0, 3.0}};
  const numerical::vector<double> r {std::vector<double> {9.0, 4.0, 5.0, 6.0}};
  EXPECT((numerical::toeplitz(c, r) == matrix<double> {{1.0, 4.0, 5.0, 6.0}, {2.0, 1.0, 4.0, 5.0}, {3.0, 2.0, 1.0, 4.0}}));
  EXPECT((numerical::toeplitz(c, c) == matrix<double> {{1.0, 2.0, 3.0}, {2.0, 1.0, 2.0}, {3.0, 2.0, 1.0}}));
  EXPECT((numerical::hankel(c, r) == matrix<double> {{1.0, 2.0, 3.0, 4.0}, {2.0, 3.0, 4.0, 5.0}, {3.0, 4.0, 5.0, 6.0}}));
  EXPECT((numerical::circulant(c) == matrix<double> {{1.0, 3.0, 2.0}, {2.0, 1.0, 3.0}, {3.0, 2.0, 1.0}}));
  const matrix<C> complex     {numerical::circulant(numerical::vector<C> {std::vector<C> {C {0.0, 1.0}, C {2.0}}})};
  EXPECT((complex == matrix<C> {{C {0.0, 1.0}, C {2.0}}, {C {2.0}, C {0.0, 1.0}}}));
  const matrix<double> blocks {numerical::block_diag(std::vector<matrix<double>> {matrix<double> {{1.0, 2.0}, {3.0, 4.0}}, matrix<double> {0, 1, std::vector<double> {}},
                                                                                  matrix<double> {{5.0}}})};
  EXPECT((blocks == matrix<double> {{1.0, 2.0, 0.0, 0.0}, {3.0, 4.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 5.0}}));
  EXPECT(numerical::block_diag(std::vector<matrix<double>> {}).rows() == 0);
}

namespace {

  //! \brief Whether pivoted QR of \p a holds: `A P = Q R` to rounding, `Q`'s columns orthonormal, `R` upper trapezoidal
  //!        with a real, non-negative diagonal that does not grow, `P` a permutation.
  template <typename T>
  bool pivoted_qr_holds(const matrix<T>&   a,
                        numerical::qr_mode mode)
  {
    const numerical::pivoted_qr_decomposition<T> f     {a, mode};
    const matrix<T>&                             q     {f.q()};
    const matrix<T>&                             r     {f.r()};
    const std::vector<std::size_t>&              p     {f.permutation()};
    const std::size_t                            m     {a.rows()};
    const std::size_t                            n     {a.cols()};
    std::vector<bool>                            seen(n, false);
    bool                                         holds {p.size() == n && q.rows() == m && r.cols() == n && q.cols() == r.rows()};
    for (const std::size_t origin : p) {
      holds                         = holds && origin < n && !seen[origin];
      seen[origin < n ? origin : 0] = true;
    }
    double size {1.0};
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        size = std::max(size, std::abs(a(i, j)));
      }
    }
    for (std::size_t i {0}; holds && i < m; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        T value {};
        for (std::size_t t {0}; t < r.rows(); ++t) {
          value += q(i, t) * r(t, j);
        }
        holds = holds && std::abs(value - a(i, p[j])) < 1e-13 * size * static_cast<double>(n + 1);
      }
    }
    for (std::size_t i {0}; holds && i < q.cols(); ++i) {
      for (std::size_t j {0}; j < q.cols(); ++j) {
        T dot {};
        for (std::size_t t {0}; t < m; ++t) {
          dot += numerical::detail::conj_of(q(t, i)) * q(t, j);
        }
        holds = holds && std::abs(dot - (i == j ? T {1.0} : T {})) < 1e-13 * static_cast<double>(m);
      }
    }
    for (std::size_t i {0}; holds && i < r.rows(); ++i) {
      for (std::size_t j {0}; j < std::min(i, n); ++j) {
        holds = holds && r(i, j) == T {};
      }
      if (i < std::min(m, n)) {
        holds = holds && std::imag(r(i, i)) == 0.0 && std::real(r(i, i)) >= 0.0;
        holds = holds && (i == 0 || std::abs(r(i, i)) <= std::abs(r(i - 1, i - 1)) * (1.0 + 1e-12));
      }
    }
    return holds;
  }
} // namespace

// QR with column pivoting, as scipy.linalg.qr(a, pivoting=True): square, tall and wide, real and complex, reduced and
// complete; a rank-deficient matrix's rank on R's diagonal (its norms downdated until the downdate has lost too much,
// then recomputed); a zero matrix; SciPy's permutation on a small example.
TEST(pivoted_qr_reveals_the_rank)
{
  using numerical::qr_mode;
  for (const auto& [m, n] : {std::pair<std::size_t, std::size_t> {1, 1}, {6, 6}, {9, 4}, {4, 9}, {40, 30}}) {
    std::uint64_t       state {(m * 13) + n};
    std::vector<double> real(m * n);
    std::vector<C>      complex(m * n);
    for (std::size_t e {0}; e < m * n; ++e) {
      real[e]    = drawn(state);
      complex[e] = drawn_scalar<C>(state);
    }
    for (const qr_mode mode : {qr_mode::reduced, qr_mode::complete}) {
      EXPECT(pivoted_qr_holds(matrix<double> {m, n, real}, mode));
      EXPECT(pivoted_qr_holds(matrix<C> {m, n, complex}, mode));
    }
  }
  std::vector<double> low(std::size_t   {20} *12); // rank 3: three columns' combinations
  std::uint64_t       state             {5};
  std::vector<double> basis(std::size_t {20} *3);
  for (double& value : basis) {
    value = drawn(state);
  }
  for (std::size_t i {0}; i < 20; ++i) {
    for (std::size_t j {0}; j < 12; ++j) {
      low[(i * 12) + j] = (basis[(i * 3) + (j % 3)] * static_cast<double>(j + 1)) + (j % 4 == 3 ? basis[(i * 3) + 1] : 0.0);
    }
  }
  const matrix<double>                              deficient {20, 12, low};
  const numerical::pivoted_qr_decomposition<double> f         {deficient};
  EXPECT(pivoted_qr_holds(deficient, qr_mode::reduced));
  EXPECT(f.r()(2, 2) > 1e-8 && f.r()(3, 3) < 1e-12 * f.r()(0, 0));
         EXPECT(pivoted_qr_holds(matrix<double> {3, 2, std::vector<double>(6, 0.0)}, qr_mode::complete));
  EXPECT(pivoted_qr_holds(matrix<C> {3, 2, std::vector<C>(6, C {})}, qr_mode::complete)); // a real diagonal, nothing below
  // past 128 columns, blocks of 32 as geqp3's laqps: a full matrix; one of rank 40 (its downdates lose everything after
  // the 40th step, the block ends, the norms recomputed); one whose last 220 columns are 0 (reflections the identity)
  for (const auto& [m, n] : {std::pair<std::size_t, std::size_t> {300, 200}, {170, 260}}) {
    std::uint64_t       draws {m + n};
    std::vector<double> real(m * n);
    std::vector<C>      complex(m * n);
    for (std::size_t e {0}; e < m * n; ++e) {
      real[e]    = drawn(draws);
      complex[e] = drawn_scalar<C>(draws);
    }
    for (const qr_mode mode : {qr_mode::reduced, qr_mode::complete}) {
      EXPECT(pivoted_qr_holds(matrix<double> {m, n, real}, mode));
      EXPECT(pivoted_qr_holds(matrix<C> {m, n, complex}, mode));
    }
  }
  std::vector<double> wide_basis(std::size_t {300} *40);
  for (double& value : wide_basis) {
    value = drawn(state);
  }
  std::vector<double> rank_forty(std::size_t {300} *260);
  std::vector<double> zero_tail(std::size_t  {300} *260, 0.0);
  for (std::size_t i {0}; i < 300; ++i) {
    for (std::size_t j {0}; j < 260; ++j) {
      rank_forty[(i * 260) + j] = wide_basis[(i * 40) + (j % 40)] * (1.0 + (0.01 * static_cast<double>(j)));
    }
    for (std::size_t j {0}; j < 40; ++j) {
      zero_tail[(i * 260) + j] = wide_basis[(i * 40) + j];
    }
  }
  const numerical::pivoted_qr_decomposition<double> forty {matrix<double> {300, 260, rank_forty}};
  EXPECT(pivoted_qr_holds(matrix<double> {300, 260, rank_forty}, qr_mode::reduced));
  EXPECT(forty.r()(39, 39) > 1e-8 && forty.r()(40, 40) < 1e-12 * forty.r()(0, 0));
         EXPECT(pivoted_qr_holds(matrix<double> {300, 260, zero_tail}, qr_mode::reduced));
  const numerical::pivoted_qr_decomposition<double> example {matrix<double> {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}, {1, 0, 1}}};
  EXPECT((example.permutation() == std::vector<std::size_t> {2, 0, 1}));
  EXPECT(std::abs(example.r()(0, 0) - 12.08304597) < 1e-8);
         EXPECT_THROWS((numerical::pivoted_qr_decomposition<double> {matrix<double> {{std::numeric_limits<double>::quiet_NaN()}}}), numerical::non_finite_error);
}
