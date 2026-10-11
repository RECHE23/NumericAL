// Every decomposition and every solver built on one, instantiated for each scalar the library is generic
// over (float, double, long double and their complex forms): each compiles, and each answers to the
// precision of its type. Norms, pivot magnitudes and singular values are carried in double whatever the
// scalar, so a long double result is checked to double's precision, not its own.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::matrix;
  using numerical::vector;

  template <typename T>
  struct real_of
  {
    using type = T;
  };

  template <typename U>
  struct real_of<std::complex<U>>
  {
    using type = U;
  };

  template <typename T>
  using real_t = real_of<T>::type;

  constexpr std::size_t side {12};

  //! \brief The bound on a residual: a small multiple of the matrix side times the coarser of T's and double's epsilon.
  template <typename T>
  double tolerance()
  {
    const double epsilon {std::max(static_cast<double>(std::numeric_limits<real_t<T>>::epsilon()),
                                   std::numeric_limits<double>::epsilon())};
    return 64.0 * static_cast<double>(side) * epsilon;
  }

  //! \brief A deterministic entry with a nonzero imaginary part when T is complex.
  template <typename T>
  T entry(std::size_t i,
          std::size_t j)
  {
    const auto re {static_cast<real_t<T>>(std::sin(static_cast<double>((7 * i) + (3 * j) + 1)))};
    if constexpr (std::is_same_v<T, real_t<T>>) {
      return re;
    }
    else {
      return T {re, static_cast<real_t<T>>(std::cos(static_cast<double>((5 * i) + (11 * j) + 2)))};
    }
  }

  template <typename T>
  T conjugate(const T& x)
  {
    if constexpr (std::is_same_v<T, real_t<T>>) {
      return x;
    }
    else {
      return std::conj(x);
    }
  }

  //! \brief A well-conditioned `rows × cols` matrix: deterministic entries plus a dominant diagonal.
  template <typename T>
  matrix<T> general(std::size_t rows,
                    std::size_t cols)
  {
    std::vector<T> data(rows * cols);
    for (std::size_t i {0}; i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        data[(i * cols) + j] = entry<T>(i, j) + (i == j ? T(real_t<T>(4 * side)) : T {});
      }
    }
    return matrix<T> {rows, cols, std::move(data)};
  }

  //! \brief A Hermitian positive-definite matrix: `G Gᴴ` plus its side on the diagonal.
  template <typename T>
  matrix<T> hermitian_positive()
  {
    const matrix<T> g {general<T>(side, side)};
    std::vector<T>  data(side * side);
    for (std::size_t i {0}; i < side; ++i) {
      for (std::size_t j {0}; j < side; ++j) {
        T sum {};
        for (std::size_t k {0}; k < side; ++k) {
          sum += g(i, k) * conjugate(g(j, k));
        }
        data[(i * side) + j] = sum + (i == j ? T(real_t<T>(side)) : T {});
      }
    }
    return matrix<T> {side, side, std::move(data)};
  }

  template <typename T>
  vector<T> right_hand_side(std::size_t n)
  {
    std::vector<T> data(n);
    for (std::size_t i {0}; i < n; ++i) {
      data[i] = entry<T>(i, n + i);
    }
    return vector<T> {std::move(data)};
  }

  //! \brief The largest entry magnitude of `a - b`, relative to the largest of `b`.
  template <typename T>
  double relative_gap(const matrix<T>& a,
                      const matrix<T>& b)
  {
    double gap   {0.0};
    double scale {0.0};
    for (std::size_t i {0}; i < b.rows(); ++i) {
      for (std::size_t j {0}; j < b.cols(); ++j) {
        gap   = std::max(gap, static_cast<double>(std::abs(a(i, j) - b(i, j))));
        scale = std::max(scale, static_cast<double>(std::abs(b(i, j))));
      }
    }
    return gap / scale;
  }

  template <typename T>
  double relative_gap(const vector<T>& a,
                      const vector<T>& b)
  {
    double gap   {0.0};
    double scale {0.0};
    for (std::size_t i {0}; i < b.size(); ++i) {
      gap   = std::max(gap, static_cast<double>(std::abs(a[i] - b[i])));
      scale = std::max(scale, static_cast<double>(std::abs(b[i])));
    }
    return gap / scale;
  }

  template <typename T>
  matrix<T> identity(std::size_t n)
  {
    std::vector<T> data(n * n, T {});
    for (std::size_t i {0}; i < n; ++i) {
      data[(i * n) + i] = T(real_t<T>(1));
    }
    return matrix<T> {n, n, std::move(data)};
  }

  template <typename T>
  bool lu_answers()
  {
    const matrix<T>                      a     {general<T>(side, side)};
    const vector<T>                      b     {right_hand_side<T>(side)};
    const numerical::lu_decomposition<T> lu    {a};
    const vector<T>                      x     {lu.solve(b)};
    const matrix<T>                      inv   {numerical::inverse(a)};
    const matrix<T>                      many  {numerical::solve(a, a)};
    const T                              det   {numerical::determinant(a)};
    return relative_gap(a * x, b) < tolerance<T>() && relative_gap(a * inv, identity<T>(side)) < tolerance<T>()
           && relative_gap(many, identity<T>(side)) < tolerance<T>() && std::abs(det) > real_t<T>(0)
           && relative_gap(numerical::solve(a, b), x) < tolerance<T>();
  }

  template <typename T>
  bool cholesky_answers()
  {
    const matrix<T> a {hermitian_positive<T>()};
    const vector<T> b {right_hand_side<T>(side)};
    const auto      c {numerical::cholesky(a)};
    const matrix<T> l {c.lower()};
    std::vector<T>  lh(side * side);
    for (std::size_t i {0}; i < side; ++i) {
      for (std::size_t j {0}; j < side; ++j) {
        lh[(i * side) + j] = conjugate(l(j, i));
      }
    }
    return relative_gap(l * matrix<T> {side, side, std::move(lh)}, a) < tolerance<T>()
           && relative_gap(a * c.solve(b), b) < tolerance<T>();
  }

  template <typename T>
  bool qr_answers()
  {
    const matrix<T> a      {general<T>(side + 3, side)};
    const auto      f      {numerical::qr(a)};
    const matrix<T> square {general<T>(side, side)};
    const vector<T> b      {right_hand_side<T>(side)};
    return relative_gap(f.q() * f.r(), a) < tolerance<T>()
           && relative_gap(square * numerical::qr(square).solve(b), b) < tolerance<T>();
  }

  template <typename T>
  bool svd_answers()
  {
    const matrix<T>                       a {general<T>(side + 2, side)};
    const numerical::svd_decomposition<T> d {a};
    const std::size_t                     k {d.singular_values().size()};
    std::vector<T>                        us(d.u().rows() * k);
    for (std::size_t i {0}; i < d.u().rows(); ++i) {
      for (std::size_t j {0}; j < k; ++j) {
        us[(i * k) + j] = d.u()(i, j) * T(static_cast<real_t<T>>(d.singular_values()[j]));
      }
    }
    const matrix<T> rebuilt {matrix<T> {d.u().rows(), k, std::move(us)} *d.vh()};
    const vector<T> b       {right_hand_side<T>(side + 2)};
    const vector<T> x       {numerical::lstsq(a, b)};
    const matrix<T> p       {numerical::pinv(a)};
    return relative_gap(rebuilt, a) < tolerance<T>() && numerical::matrix_rank(a) == side
           && relative_gap(p * b, x) < tolerance<T>() && relative_gap(p * a, identity<T>(side)) < tolerance<T>();
  }

  template <typename T>
  bool every_decomposition_answers()
  {
    return lu_answers<T>() && cholesky_answers<T>() && qr_answers<T>() && svd_answers<T>();
  }
} // namespace

TEST(every_decomposition_answers_for_float)
{
  EXPECT(every_decomposition_answers<float>());
}

TEST(every_decomposition_answers_for_double)
{
  EXPECT(every_decomposition_answers<double>());
}

TEST(every_decomposition_answers_for_long_double)
{
  EXPECT(every_decomposition_answers<long double>());
}

TEST(every_decomposition_answers_for_complex_float)
{
  EXPECT(every_decomposition_answers<std::complex<float>>());
}

TEST(every_decomposition_answers_for_complex_double)
{
  EXPECT(every_decomposition_answers<std::complex<double>>());
}

TEST(every_decomposition_answers_for_complex_long_double)
{
  EXPECT(every_decomposition_answers<std::complex<long double>>());
}
