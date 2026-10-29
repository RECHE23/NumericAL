/*!
 * \file matrix_functions.hpp
 * \brief Functions of a square matrix: the exponential, by scaling and squaring of a Padé approximant.
 */
#ifndef NUMERICAL_MATRIX_FUNCTIONS_HPP
#define NUMERICAL_MATRIX_FUNCTIONS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <utility>

#include "decomposition.hpp"

namespace numerical {

  namespace detail {

    //! \brief The 1-norm of the `n × n` row-major \p a (`n` from 1): its largest column sum of moduli.
    template <typename T>
    [[nodiscard]] double one_norm(const buffer<T>& a,
                                  std::size_t      n)
    {
      buffer<double> sums(n, 0.0);
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          sums[j] += std::abs(a[(i * n) + j]);
        }
      }
      return *std::max_element(sums.begin(), sums.end());
    }

    //! \brief `‖ |A|ᵖ ‖₁` of the `n × n` row-major \p a (`n` from 1) times \p scale, exactly: the row `1ᵀ |scale A|ᵖ` by \p p products of a
    //!        vector and `|A|`, a matrix of moduli's norm its largest entry.
    template <typename T>
    [[nodiscard]] double modulus_power_norm(const buffer<T>& a,
                                            std::size_t      n,
                                            std::size_t      p,
                                            double           scale)
    {
      buffer<double> row(n, 1.0);
      buffer<double> next(n, 0.0);
      for (std::size_t step {0}; step < p; ++step) {
        std::fill(next.begin(), next.end(), 0.0);
        for (std::size_t i {0}; i < n; ++i) {
          for (std::size_t j {0}; j < n; ++j) {
            next[j] += row[i] * std::abs(a[(i * n) + j]) * scale;
          }
        }
        std::swap(row, next);
      }
      return *std::max_element(row.begin(), row.end());
    }

    //! \brief The `n × n` product `A B` of row-major operands.
    template <typename T>
    [[nodiscard]] buffer<T> square_product(const buffer<T>& a,
                                           const buffer<T>& b,
                                           std::size_t      n)
    {
      buffer<T> c(n * n, T {});
      gemm(n, n, n, a.data(), n, b.data(), n, c.data(), n);
      return c;
    }

    //! \brief The coefficients `b₀ … b_m` of the `[m/m]` Padé approximant of the exponential, as Higham's (2005).
    [[nodiscard]] inline const double* pade_coefficients(std::size_t m)
    {
      static constexpr std::array<double, 4>  three    {120.0, 60.0, 12.0, 1.0};
      static constexpr std::array<double, 6>  five     {30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0};
      static constexpr std::array<double, 8>  seven    {17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0};
      static constexpr std::array<double, 10> nine     {17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
                                                        2162160.0,     110880.0,     3960.0,       90.0,        1.0};
      static constexpr std::array<double, 14> thirteen {64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 1187353796428800.0,
                                                        129060195264000.0,   10559470521600.0,    670442572800.0,     33522128640.0,
                                                        1323241920.0,        40840800.0,          960960.0,           16380.0,
                                                        182.0,               1.0};
      switch (m) {
        case 3: return three.data();
        case 5: return five.data();
        case 7: return seven.data();
        case 9: return nine.data();
        default: return thirteen.data();
      }
    }

    /*!
     * \brief Al-Mohy and Higham's `ℓ(A, m)` (2009): the squarings to add so that the `[m/m]` approximant's backward error
     *        bound, from the leading term of its series `|c₂ₘ₊₁| ‖ |A|²ᵐ⁺¹ ‖₁ / ‖A‖₁`, falls below the unit roundoff;
     *        \p a times \p scale (a power of two), its norms exact.
     */
    template <typename T>
    [[nodiscard]] std::size_t extra_squarings(const buffer<T>& a,
                                              std::size_t      n,
                                              std::size_t      m,
                                              double           scale,
                                              double           norm)
    {
      const auto   order       {static_cast<double>(m)}; // c = (m!)² / ((2m)! (2m + 1)!), by the logarithms of the factorials
      const double c           {std::exp((2.0 * std::lgamma(order + 1.0)) - std::lgamma((2.0 * order) + 1.0) - std::lgamma((2.0 * order) + 2.0))};
      const double scaled_norm {norm * scale};
      if (scaled_norm == 0.0) {
        return 0;
      }
      const double alpha {c * modulus_power_norm(a, n, (2 * m) + 1, scale) / scaled_norm};
      const double unit {std::ldexp(1.0, -53)};
      if (alpha <= unit) {
        return 0;
      }
      return static_cast<std::size_t>(std::ceil(std::log2(alpha / unit) / static_cast<double>(2 * m)));
    }

    /*!
     * \brief The `[m/m]` Padé approximant `(V − U)⁻¹ (V + U)` of the exponential of \p a, its even powers \p powers
     *        (`A²`, `A⁴`, `A⁶`, and `A⁸` for `m = 9`), as Higham's (2005): `U` the odd terms, `V` the even, `A¹³`'s
     *        terms by `A⁶` (`A⁶ (b₁₃ A⁶ + b₁₁ A⁴ + b₉ A²) + …`).
     */
    template <typename T>
    [[nodiscard]] buffer<T> pade_approximant(const buffer<T>&                       a,
                                             const std::array<const buffer<T>*, 4>& powers,
                                             std::size_t                            n,
                                             std::size_t                            m)
    {
      const double* const b {pade_coefficients(m)};
      buffer<T>           odd(n * n, T {});  // the sum U is A times
      buffer<T>           even(n * n, T {}); // V
      const auto          add = [&](buffer<T>& into, const buffer<T>& power, double coefficient) {
                                  for (std::size_t e {0}; e < n * n; ++e) {
                                    into[e] += T {coefficient} *power[e];
                                  }
                                };
      if (m == 13) {
        const buffer<T>& a2                 {*powers[0]};
        const buffer<T>& a4                 {*powers[1]};
        const buffer<T>& a6                 {*powers[2]};
        buffer<T>        high_odd(n * n, T  {});
        buffer<T>        high_even(n * n, T {});
        add(high_odd, a6, b[13]);
        add(high_odd, a4, b[11]);
        add(high_odd, a2, b[9]);
        add(high_even, a6, b[12]);
        add(high_even, a4, b[10]);
        add(high_even, a2, b[8]);
        odd  = square_product(a6, high_odd, n);
        even = square_product(a6, high_even, n);
        add(odd, a6, b[7]);
        add(odd, a4, b[5]);
        add(odd, a2, b[3]);
        add(even, a6, b[6]);
        add(even, a4, b[4]);
        add(even, a2, b[2]);
      }
      else {
        for (std::size_t k {2}; k < m; k += 2) { // b_{k+1} A^k into U's sum, b_k A^k into V
          add(odd, *powers[(k / 2) - 1], b[k + 1]);
          add(even, *powers[(k / 2) - 1], b[k]);
        }
      }
      for (std::size_t i {0}; i < n; ++i) {
        odd[(i * n) + i]  += T {b[1]};
        even[(i * n) + i] += T {b[0]};
      }
      const buffer<T> u {square_product(a, odd, n)};
      buffer<T>       numerator(n * n);
      buffer<T>       denominator(n * n);
      for (std::size_t e {0}; e < n * n; ++e) {
        numerator[e]   = even[e] + u[e];
        denominator[e] = even[e] - u[e];
      }
      const lu_decomposition<T> lu {matrix<T> {n, n, std::move(denominator)}};
      return lu.solve(matrix<T> {n, n, std::move(numerator)}).elements();
    }

    //! \brief `sinh(x) / x`, 1 at 0, by its series where `x` is small.
    template <typename T>
    [[nodiscard]] T sinch(const T& x)
    {
      if (std::abs(x) < 1e-4) {
        const T square {x * x};
        return T {1.0} + (square / T {6.0}) * (T {1.0} + (square / T {20.0}));
      }
      return std::sinh(x) / x;
    }

    /*!
     * \brief The diagonal and first superdiagonal of `exp(scale · T)` written exactly into \p x for the upper triangular
     *        \p t, as Al-Mohy and Higham's code fragment 2.1: `e^{λᵢ}`, and `t_{i,i+1} e^{(λᵢ + λᵢ₊₁)/2} sinch((λᵢ − λᵢ₊₁)/2)`.
     */
    template <typename T>
    void exact_triangular_bands(buffer<T>&       x,
                                const buffer<T>& t,
                                std::size_t      n,
                                double           scale)
    {
      for (std::size_t i {0}; i < n; ++i) {
        const T lambda {T {scale} *t[(i * n) + i]};
        x[(i * n) + i] = std::exp(lambda);
        if (i + 1 < n) {
          const T next     {T {scale} *t[((i + 1) * n) + i + 1]};
          const T half_sum {(lambda + next) / T {2.0}};
          const T half_gap {(lambda - next) / T {2.0}};
          x[(i * n) + i + 1] = T {scale} *t[(i * n) + i + 1] * std::exp(half_sum) * sinch(half_gap);
        }
      }
    }
  } // namespace detail

  /*!
   * \brief The exponential `e^A` of a square matrix, as `scipy.linalg.expm`: scaling and squaring of a Padé
   *        approximant, as Al-Mohy and Higham's algorithm (2009): the degree 3, 5, 7, 9 or 13 and the squarings `s`
   *        chosen from `‖A^{2k}‖₁^{1/2k}` (exact, `A²`, `A⁴` and `A⁶` formed, `A⁸` and `A¹⁰` where needed) and the
   *        approximant's backward error bound (`ℓ`), then `r(2⁻ˢ A)` squared `s` times; an upper triangular `A`'s
   *        diagonal and first superdiagonal written exactly at each squaring.
   * \tparam T `double` or `std::complex<double>` (the degrees' bounds are double precision's).
   * \throws dimension_error   If \p a is not square.
   * \throws non_finite_error  If an entry is infinite or NaN.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] matrix<T> expm(const matrix<T>& a)
  {
    const std::size_t n {a.rows()};
    if (a.cols() != n) {
      throw dimension_error("expm requires a square matrix");
    }
    detail::require_finite(a, "expm");
    if (n == 0) {
      return matrix<T> {0, 0, buffer<T> {}};
    }
    const buffer<T> a0(a.begin(), a.end());
    bool            triangular {true};
    for (std::size_t i {1}; i < n && triangular; ++i) {
      for (std::size_t j {0}; j < i; ++j) {
        triangular = triangular && a0[(i * n) + j] == T {};
      }
    }
    const double                    norm {detail::one_norm(a0, n)};
    buffer<T>                       a2 {detail::square_product(a0, a0, n)};
    buffer<T>                       a4 {detail::square_product(a2, a2, n)};
    buffer<T>                       a6 {detail::square_product(a4, a2, n)};
    const double                    d4 {std::pow(detail::one_norm(a4, n), 1.0 / 4.0)};
    const double                    d6 {std::pow(detail::one_norm(a6, n), 1.0 / 6.0)};
    const double                    eta1 {std::max(d4, d6)};
    constexpr std::array<double, 5> theta {1.495585217958292e-2, 2.539398330063230e-1, 9.504178996162932e-1, 2.097847961257068e0,
                                           5.371920351148152e0};
    const auto                      finish = [&](buffer<T> x) { return matrix<T> {n, n, std::move(x)}; };
    std::array<const buffer<T>*, 4> powers {&a2, &a4, &a6, nullptr};
    if (eta1 <= theta[0] && detail::extra_squarings(a0, n, 3, 1.0, norm) == 0) {
      return finish(detail::pade_approximant(a0, powers, n, 3));
    }
    if (eta1 <= theta[1] && detail::extra_squarings(a0, n, 5, 1.0, norm) == 0) {
      return finish(detail::pade_approximant(a0, powers, n, 5));
    }
    const buffer<T> a8 {detail::square_product(a4, a4, n)};
    powers[3] = &a8;
    const double d8    {std::pow(detail::one_norm(a8, n), 1.0 / 8.0)};
    const double eta3  {std::max(d6, d8)};
    if (eta3 <= theta[2] && detail::extra_squarings(a0, n, 7, 1.0, norm) == 0) {
      return finish(detail::pade_approximant(a0, powers, n, 7));
    }
    if (eta3 <= theta[3] && detail::extra_squarings(a0, n, 9, 1.0, norm) == 0) {
      return finish(detail::pade_approximant(a0, powers, n, 9));
    }
    const double d10  {std::pow(detail::one_norm(detail::square_product(a4, a6, n), n), 1.0 / 10.0)};
    const double eta5 {std::min(eta3, std::max(d8, d10))};
    std::size_t  s    {eta5 <= theta[4] ? 0 : static_cast<std::size_t>(std::ceil(std::log2(eta5 / theta[4])))};
    s += detail::extra_squarings(a0, n, 13, std::ldexp(1.0, -static_cast<int>(s)), norm);
    const int power   {static_cast<int>(s)};
    buffer<T> scaled  {a0};
    for (std::size_t e {0}; e < n * n; ++e) {
      scaled[e] *= T {std::ldexp(1.0, -power)};
      a2[e]     *= T {std::ldexp(1.0, -2 * power)};
      a4[e]     *= T {std::ldexp(1.0, -4 * power)};
      a6[e]     *= T {std::ldexp(1.0, -6 * power)};
    }
    buffer<T> x {detail::pade_approximant(scaled, powers, n, 13)};
    if (triangular) {
      detail::exact_triangular_bands(x, a0, n, std::ldexp(1.0, -power));
    }
    for (int j {power - 1}; j >= 0; --j) {
      x = detail::square_product(x, x, n);
      if (triangular) {
        detail::exact_triangular_bands(x, a0, n, std::ldexp(1.0, -j));
      }
    }
    return finish(std::move(x));
  }
} // namespace numerical

#endif // NUMERICAL_MATRIX_FUNCTIONS_HPP
