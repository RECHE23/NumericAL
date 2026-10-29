/*!
 * \file matrix_functions.hpp
 * \brief Functions of a square matrix and the matrix equations solved through Schur forms: the exponential, by
 *        scaling and squaring of a Padé approximant; the principal square root, by the Schur method; the Sylvester
 *        equation `A X + X B = Q`, as Bartels and Stewart.
 */
#ifndef NUMERICAL_MATRIX_FUNCTIONS_HPP
#define NUMERICAL_MATRIX_FUNCTIONS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "complex_math.hpp"
#include "eigen.hpp"

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

  namespace detail {

    //! \brief `Mᴴ`: the transpose, conjugated.
    template <typename T>
    [[nodiscard]] matrix<T> adjoint_of(const matrix<T>& m)
    {
      buffer<T> out(m.rows() * m.cols());
      for (std::size_t i {0}; i < m.rows(); ++i) {
        for (std::size_t j {0}; j < m.cols(); ++j) {
          out[(j * m.rows()) + i] = conj_of(m(i, j));
        }
      }
      return matrix<T> {m.cols(), m.rows(), std::move(out)};
    }

    //! \brief The order of the diagonal block of the upper quasi-triangular \p t (`n × n`, row-major) that starts at
    //!        \p k: 2 where the entry below its diagonal is nonzero (a real Schur form's pair).
    [[nodiscard]] inline std::size_t schur_block(const buffer<double>& t,
                                                 std::size_t           n,
                                                 std::size_t           k)
    {
      return k + 1 < n && t[((k + 1) * n) + k] != 0.0 ? 2 : 1;
    }

    //! \brief The same for a complex Schur form, triangular: 1.
    [[nodiscard]] inline std::size_t schur_block(const buffer<std::complex<double>>& /*t*/,
                                                 std::size_t /*n*/,
                                                 std::size_t /*k*/)
    {
      return 1;
    }

    /*!
     * \brief The \p rows × \p cols `y` of `R y + y S = c` (each at most 2, row-major in \p c, overwritten) for the
     *        diagonal blocks \p r (row stride \p ldr) and \p s (row stride \p lds): the Kronecker system of order
     *        `rows · cols` by Gaussian elimination with complete pivoting, a pivot under \p smin taken as \p smin, as
     *        LAPACK's `dlasy2` perturbs one (the eigenvalues of `R` and `−S` close).
     */
    template <typename T>
    void small_sylvester_block(const T*    r,
                               std::size_t ldr,
                               const T*    s,
                               std::size_t lds,
                               std::size_t rows,
                               std::size_t cols,
                               T*          c,
                               std::size_t ldc,
                               double      smin)
    {
      const std::size_t               size  {rows * cols};
      std::array<std::array<T, 4>, 4> k     {};
      std::array<T, 4>                rhs   {};
      std::array<std::size_t, 4>      order {0, 1, 2, 3}; // the unknown in each column after the column swaps
      for (std::size_t i {0}; i < rows; ++i) {
        for (std::size_t j {0}; j < cols; ++j) {
          const std::size_t e {(i * cols) + j};
          rhs[e] = c[(i * ldc) + j];
          for (std::size_t p {0}; p < rows; ++p) {
            k[e][(p * cols) + j] += r[(i * ldr) + p];
          }
          for (std::size_t q {0}; q < cols; ++q) {
            k[e][(i * cols) + q] += s[(q * lds) + j];
          }
        }
      }
      for (std::size_t step {0}; step < size; ++step) {
        std::size_t pivot_row {step};
        std::size_t pivot_col {step};
        double      largest   {-1.0};
        for (std::size_t i {step}; i < size; ++i) {
          for (std::size_t j {step}; j < size; ++j) {
            if (std::abs(k[i][j]) > largest) {
              largest   = std::abs(k[i][j]);
              pivot_row = i;
              pivot_col = j;
            }
          }
        }
        std::swap(k[step], k[pivot_row]);
        std::swap(rhs[step], rhs[pivot_row]);
        for (std::size_t i {0}; i < size; ++i) {
          std::swap(k[i][step], k[i][pivot_col]);
        }
        std::swap(order[step], order[pivot_col]);
        if (std::abs(k[step][step]) < smin) {
          k[step][step] = T {smin};
        }
        for (std::size_t i {step + 1}; i < size; ++i) {
          const T factor {k[i][step] / k[step][step]};
          for (std::size_t j {step}; j < size; ++j) {
            k[i][j] -= factor * k[step][j];
          }
          rhs[i] -= factor * rhs[step];
        }
      }
      std::array<T, 4> y {};
      for (std::size_t back {size}; back > 0; --back) {
        const std::size_t i     {back - 1};
        T                 value {rhs[i]};
        for (std::size_t j {i + 1}; j < size; ++j) {
          value -= k[i][j] * y[j];
        }
        y[i] = value / k[i][i];
      }
      for (std::size_t e {0}; e < size; ++e) {
        const std::size_t unknown {order[e]};
        c[((unknown / cols) * ldc) + (unknown % cols)] = y[e];
      }
    }

    /*!
     * \brief `Y` of `R Y + Y S = F` in place in \p f (`m × n`, row-major) for the upper quasi-triangular (real) or
     *        triangular (complex) \p r (`m × m`) and \p s (`n × n`), as LAPACK's `trsyl` (no transposes, `+`): by
     *        blocks of columns of `S` from the first, each first relieved of the columns solved before it through the
     *        matrix product, then by blocks of rows of `R` from the last, each a system of at most four unknowns
     *        (\ref small_sylvester_block), `R` and `−S` close in an eigenvalue perturbing its pivot as `trsyl` does.
     */
    template <typename T>
    void triangular_sylvester(const buffer<T>& r,
                              std::size_t      m,
                              const buffer<T>& s,
                              std::size_t      n,
                              buffer<T>&       f)
    {
      double largest {0.0};
      for (const T& entry : r) {
        largest = std::max(largest, std::abs(entry));
      }
      for (const T& entry : s) {
        largest = std::max(largest, std::abs(entry));
      }
      const double smin {std::max(std::numeric_limits<double>::epsilon() * largest, std::numeric_limits<double>::min())};
      buffer<T>    minus_s;
      buffer<T>    columns; // Y's columns of the block, each contiguous
      for (std::size_t l {0}; l < n;) {
        const std::size_t nl {schur_block(s, n, l)};
        if (l > 0) { // F(:, l…) −= Y(:, 0…l) S(0…l, l…)
          minus_s.resize(l * nl);
          for (std::size_t q {0}; q < l; ++q) {
            for (std::size_t j {0}; j < nl; ++j) {
              minus_s[(q * nl) + j] = -s[(q * n) + l + j];
            }
          }
          gemm(m, nl, l, f.data(), n, minus_s.data(), nl, f.data() + l, n);
        }
        columns.resize(nl * m);
        for (std::size_t end {m}; end > 0;) {
          const std::size_t kb {end >= 2 && schur_block(r, m, end - 2) == 2 ? std::size_t {2} : std::size_t {1}};
          const std::size_t k0 {end - kb};
          for (std::size_t i {k0}; i < end; ++i) { // F(k, l) −= R(k, end…m) Y(end…m, l)
            for (std::size_t j {0}; j < nl; ++j) {
              f[(i * n) + l + j] -= dot_product(r.data() + (i * m) + end, columns.data() + (j * m) + end, m - end);
            }
          }
          small_sylvester_block(r.data() + (k0 * m) + k0, m, s.data() + (l * n) + l, n, kb, nl, f.data() + (k0 * n) + l, n, smin);
          for (std::size_t i {k0}; i < end; ++i) {
            for (std::size_t j {0}; j < nl; ++j) {
              columns[(j * m) + i] = f[(i * n) + l + j];
            }
          }
          end = k0;
        }
        l += nl;
      }
    }
  } // namespace detail

  /*!
   * \brief The solution `X` (`m × n`) of the Sylvester equation `A X + X B = Q`, as `scipy.linalg.solve_sylvester`:
   *        by Bartels and Stewart's method, the Schur forms `A = U R Uᴴ` and `B = V S Vᴴ` (\ref schur_decomposition),
   *        `R Y + Y S = Uᴴ Q V` solved by blocks (LAPACK's `trsyl`), then `X = U Y Vᴴ`. Real throughout for real
   *        operands (`R` and `S` quasi-triangular), complex where any is. Where `A` and `−B` share an eigenvalue the
   *        equation has no unique solution: a pivot below the rounding of the forms is perturbed, as LAPACK's, and the
   *        answer grows accordingly.
   * \tparam T `double` or `std::complex<double>`.
   * \throws dimension_error   If \p a or \p b is not square or \p q is not `m × n`.
   * \throws non_finite_error  If an entry is infinite or NaN.
   * \throws convergence_error If a Schur form's iterations did not converge.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] matrix<T> solve_sylvester(const matrix<T>& a,
                                          const matrix<T>& b,
                                          const matrix<T>& q)
  {
    const std::size_t m {a.rows()};
    const std::size_t n {b.rows()};
    if (a.cols() != m || b.cols() != n || q.rows() != m || q.cols() != n) {
      throw dimension_error("solve_sylvester requires a square a (m × m), a square b (n × n) and q (m × n)");
    }
    detail::require_finite(a, "solve_sylvester");
    detail::require_finite(b, "solve_sylvester");
    detail::require_finite(q, "solve_sylvester");
    if (m == 0 || n == 0) {
      return matrix<T> {m, n, buffer<T>(m * n, T {})};
    }
    const schur_decomposition<T> left  {a};
    const schur_decomposition<T> right {b};
    const matrix<T>&             u     {left.z()};
    const matrix<T>&             v     {right.z()};
    matrix<T>                    f     {detail::adjoint_of(u) * q * v};
    buffer<T>                    y     {std::move(f).elements()};
    detail::triangular_sylvester(buffer<T>(left.t().begin(), left.t().end()), m, buffer<T>(right.t().begin(), right.t().end()), n, y);
    return u * matrix<T> {m, n, std::move(y)} *detail::adjoint_of(v);
  }

  namespace detail {

    //! \brief `‖A‖₁` bounds of the `[m/m]` Padé approximants of the Fréchet derivative of the exponential, for `m` from
    //!        1 to 20, as Al-Mohy and Higham's table 6.1 (2009) and SciPy's `ell_table_61`.
    [[nodiscard]] inline double frechet_bound(std::size_t m)
    {
      static constexpr std::array<double, 21> table {0.0,    2.11e-8, 3.56e-4, 1.08e-2, 6.49e-2, 2.00e-1, 4.37e-1, 7.83e-1, 1.23e0, 1.78e0, 2.42e0,
                                                     3.13e0, 3.90e0,  4.74e0,  5.63e0,  6.56e0,  7.52e0,  8.53e0,  9.56e0, 1.06e1, 1.17e1};
      return table[m];
    }

    //! \brief `into += coefficient · term` over the `n × n` buffers.
    template <typename T>
    void add_scaled(buffer<T>&       into,
                    const buffer<T>& term,
                    double           coefficient)
    {
      for (std::size_t e {0}; e < into.size(); ++e) {
        into[e] += T {coefficient} *term[e];
      }
    }

    //! \brief `P Q + Q P` of `n × n` buffers.
    template <typename T>
    [[nodiscard]] buffer<T> both_products(const buffer<T>& p,
                                          const buffer<T>& q,
                                          std::size_t      n)
    {
      buffer<T>       sum   {square_product(p, q, n)};
      const buffer<T> other {square_product(q, p, n)};
      for (std::size_t e {0}; e < sum.size(); ++e) {
        sum[e] += other[e];
      }
      return sum;
    }

    /*!
     * \brief `e^A` and its Fréchet derivative `L(A, E)` from the `[m/m]` Padé approximant, as Al-Mohy and Higham's
     *        algorithm 6.4 (2009) and SciPy's `expm_frechet_algo_64`: `U`, `V` and their derivatives `Lu`, `Lv` along
     *        \p e (`M₂ = A E + E A`, `M₄ = A² M₂ + M₂ A²`, `M₆ = A⁴ M₂ + M₄ A²`, `M₈ = A⁴ M₄ + M₄ A⁴`), each sum from
     *        its highest term, then `R = (V − U)⁻¹ (U + V)` and `L = (V − U)⁻¹ (Lu + Lv + (Lu − Lv) R)` by one LU
     *        factorisation; for `m = 13`, \p a and \p e already scaled, `A⁶`'s nesting as \ref pade_approximant's.
     */
    template <typename T>
    [[nodiscard]] std::pair<buffer<T>, buffer<T>> frechet_pade(const buffer<T>& a,
                                                               const buffer<T>& e,
                                                               std::size_t      n,
                                                               std::size_t      m)
    {
      const double* const b                 {pade_coefficients(m)};
      buffer<T>           identity(n * n, T {});
      for (std::size_t i {0}; i < n; ++i) {
        identity[(i * n) + i] = T {1.0};
      }
      const buffer<T> a2    {square_product(a, a, n)};
      const buffer<T> m2    {both_products(a, e, n)};
      const buffer<T> a4    {square_product(a2, a2, n)};
      const buffer<T> m4    {both_products(a2, m2, n)};
      const buffer<T> a6    {square_product(a2, a4, n)};
      buffer<T>       m6    {square_product(a4, m2, n)};
      add_scaled(m6, square_product(m4, a2, n), 1.0);
      buffer<T> w(n * n, T  {}); // U = A W
      buffer<T> v(n * n, T  {});
      buffer<T> lw(n * n, T {}); // Lu = A Lw + E W
      buffer<T> lv(n * n, T {});
      if (m == 13) {
        buffer<T> w1(n * n, T  {});
        buffer<T> z1(n * n, T  {});
        buffer<T> lw1(n * n, T {});
        buffer<T> lz1(n * n, T {});
        add_scaled(w1, a6, b[13]);
        add_scaled(w1, a4, b[11]);
        add_scaled(w1, a2, b[9]);
        add_scaled(z1, a6, b[12]);
        add_scaled(z1, a4, b[10]);
        add_scaled(z1, a2, b[8]);
        add_scaled(lw1, m6, b[13]);
        add_scaled(lw1, m4, b[11]);
        add_scaled(lw1, m2, b[9]);
        add_scaled(lz1, m6, b[12]);
        add_scaled(lz1, m4, b[10]);
        add_scaled(lz1, m2, b[8]);
        w  = square_product(a6, w1, n);
        v  = square_product(a6, z1, n);
        lw = square_product(a6, lw1, n);
        add_scaled(lw, square_product(m6, w1, n), 1.0);
        add_scaled(lv, square_product(a6, lz1, n), 1.0);
        add_scaled(lv, square_product(m6, z1, n), 1.0);
        const auto lower_terms = [&](const buffer<T>& power, const buffer<T>& derivative, std::size_t k) { // W₂, Z₂, Lw₂, Lz₂
                                   add_scaled(w, power, b[k + 1]);
                                   add_scaled(v, power, b[k]);
                                   add_scaled(lw, derivative, b[k + 1]);
                                   add_scaled(lv, derivative, b[k]);
                                 };
        lower_terms(a6, m6, 6);
        lower_terms(a4, m4, 4);
        lower_terms(a2, m2, 2);
      }
      else {
        const buffer<T>                       a8          {square_product(a4, a4, n)};
        const buffer<T>                       m8          {both_products(a4, m4, n)};
        const std::array<const buffer<T>*, 4> powers      {&a2, &a4, &a6, &a8};
        const std::array<const buffer<T>*, 4> derivatives {&m2, &m4, &m6, &m8};
        for (std::size_t k {m - 1}; k >= 2; k -= 2) { // b_{k+1} A^k into W, b_k A^k into V, from the highest
          add_scaled(w, *powers[(k / 2) - 1], b[k + 1]);
          add_scaled(v, *powers[(k / 2) - 1], b[k]);
          add_scaled(lw, *derivatives[(k / 2) - 1], b[k + 1]);
          add_scaled(lv, *derivatives[(k / 2) - 1], b[k]);
        }
      }
      add_scaled(w, identity, b[1]);
      add_scaled(v, identity, b[0]);
      const buffer<T> u  {square_product(a, w, n)};
      buffer<T>       lu {square_product(a, lw, n)};
      add_scaled(lu, square_product(e, w, n), 1.0);
      buffer<T> numerator(n * n);
      buffer<T> denominator(n * n);
      for (std::size_t k {0}; k < n * n; ++k) {
        numerator[k]   = u[k] + v[k];
        denominator[k] = v[k] - u[k];
      }
      const lu_decomposition<T> factor {matrix<T> {n, n, std::move(denominator)}};
      buffer<T>                 r      {factor.solve(matrix<T> {n, n, std::move(numerator)}).elements()};
      buffer<T>                 difference(n * n);
      for (std::size_t k {0}; k < n * n; ++k) {
        difference[k] = lu[k] - lv[k];
      }
      buffer<T> derivative_terms {square_product(difference, r, n)};
      for (std::size_t k {0}; k < n * n; ++k) {
        derivative_terms[k] = lu[k] + lv[k] + derivative_terms[k];
      }
      buffer<T> l {factor.solve(matrix<T> {n, n, std::move(derivative_terms)}).elements()};
      return {std::move(r), std::move(l)};
    }
  } // namespace detail

  //! \brief How `expm_frechet` computes: Al-Mohy and Higham's algorithm (SciPy's `"SPS"`), or the exponential of the
  //!        block matrix `[[A, E], [0, A]]` (`"blockEnlarge"`).
  enum class frechet_method
  {
    scaling_pade_squaring, //!< Al-Mohy and Higham's algorithm 6.4.
    block_enlarge          //!< `expm` of the `2n × 2n` block matrix.
  };

  /*!
   * \brief `e^A` and the Fréchet derivative `L(A, E)` of the exponential at \p a along \p e, as
   *        `scipy.linalg.expm_frechet`: by default Al-Mohy and Higham's algorithm 6.4 (2009) as SciPy's: the `[m/m]`
   *        Padé approximant (`m` 3, 5, 7 or 9) where `‖A‖₁` is within its bound, else `m = 13` on `2⁻ˢ A` and
   *        `2⁻ˢ E`, `s = ⌈log₂(‖A‖₁ / 4.74)⌉`, then `s` squarings `L ← R L + L R`, `R ← R²`; or the exponential of
   *        `[[A, E], [0, A]]`, whose blocks are `e^A` and `L(A, E)`.
   * \tparam T `double` or `std::complex<double>`.
   * \throws dimension_error  If \p a or \p e is not square or their orders differ.
   * \throws non_finite_error If an entry is infinite or NaN.
   * \throws singular_matrix_error If the approximant's denominator is singular.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] std::pair<matrix<T>, matrix<T>> expm_frechet(const matrix<T>& a,
                                                             const matrix<T>& e,
                                                             frechet_method   method = frechet_method::scaling_pade_squaring)
  {
    const std::size_t n {a.rows()};
    if (a.cols() != n || e.rows() != n || e.cols() != n) {
      throw dimension_error("expm_frechet requires square matrices a and e of one order");
    }
    detail::require_finite(a, "expm_frechet");
    detail::require_finite(e, "expm_frechet");
    if (n == 0) {
      return {matrix<T> {0, 0, buffer<T> {}}, matrix<T> {0, 0, buffer<T> {}}};
    }
    if (method == frechet_method::block_enlarge) {
      buffer<T> block(4 * n * n, T {});
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          block[(i * 2 * n) + j]           = a(i, j);
          block[(i * 2 * n) + n + j]       = e(i, j);
          block[((n + i) * 2 * n) + n + j] = a(i, j);
        }
      }
      const matrix<T> x {expm(matrix<T> {2 * n, 2 * n, std::move(block)})};
      buffer<T>       top(n * n);
      buffer<T>       right(n * n);
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          top[(i * n) + j]   = x(i, j);
          right[(i * n) + j] = x(i, n + j);
        }
      }
      return {matrix<T> {n, n, std::move(top)}, matrix<T> {n, n, std::move(right)}};
    }
    buffer<T>    a0(a.begin(), a.end());
    buffer<T>    e0(e.begin(), e.end());
    const double norm {detail::one_norm(a0, n)};
    for (const std::size_t m : {std::size_t {3}, std::size_t {5}, std::size_t {7}, std::size_t {9}}) {
      if (norm <= detail::frechet_bound(m)) {
        auto [r, l] {detail::frechet_pade(a0, e0, n, m)};
        return {matrix<T> {n, n, std::move(r)}, matrix<T> {n, n, std::move(l)}};
      }
    }
    const int s {std::max(0, static_cast<int>(std::ceil(std::log2(norm / detail::frechet_bound(13)))))};
    for (std::size_t k {0}; k < n * n; ++k) {
      a0[k] *= T {std::ldexp(1.0, -s)};
      e0[k] *= T {std::ldexp(1.0, -s)};
    }
    auto [r, l] {detail::frechet_pade(a0, e0, n, 13)};
    for (int k {0}; k < s; ++k) {
      l = detail::both_products(r, l, n);
      r = detail::square_product(r, r, n);
    }
    return {matrix<T> {n, n, std::move(r)}, matrix<T> {n, n, std::move(l)}};
  }

  /*!
   * \brief The relative condition number of the exponential at \p a in the Frobenius norm, as `scipy.linalg.expm_cond`:
   *        `‖K‖₂ ‖A‖_F / ‖e^A‖_F`, `K` the `n² × n²` Kronecker form of the Fréchet derivative (its column `i n + j`
   *        the column-major `vec` of `L(A, eᵢ eⱼᵀ)`), `‖K‖₂` its largest singular value.
   *        An empty matrix's is 0.
   * \throws dimension_error  If \p a is not square.
   * \throws non_finite_error If an entry is infinite or NaN.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] double expm_cond(const matrix<T>& a)
  {
    const std::size_t n {a.rows()};
    if (a.cols() != n) {
      throw dimension_error("expm_cond requires a square matrix");
    }
    detail::require_finite(a, "expm_cond");
    if (n == 0) { // no direction to perturb along
      return 0.0;
    }
    const std::size_t columns {n * n};
    buffer<T>         k(columns * columns);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        buffer<T> unit(columns, T {});
        unit[(i * n) + j] = T {1.0};
        const matrix<T>   l       {expm_frechet(a, matrix<T> {n, n, std::move(unit)}).second};
        for (std::size_t q {0}; q < n; ++q) { // vec, column-major: entry (p, q) at q n + p
          for (std::size_t p {0}; p < n; ++p) {
            k[(((q * n) + p) * columns) + (i * n) + j] = l(p, q);
          }
        }
      }
    }
    const auto frobenius = [](const matrix<T>& m) {
                             double sum {0.0};
                             for (const T& x : m) {
                               sum += std::norm(x);
                             }
                             return std::sqrt(sum);
                           };
    const double kronecker {singular_values(matrix<T> {columns, columns, std::move(k)})[0]};
    return kronecker * frobenius(a) / frobenius(expm(a));
  }

  namespace detail {

    /*!
     * \brief The off-diagonal blocks of the square root \p r (`n × n`, row-major) of the upper (quasi-)triangular \p t,
     *        its diagonal blocks already in \p r, column of blocks by column, each block row from the last: `R_ii X +
     *        X R_jj = T_ij − Σ R_ik R_kj` (Higham, 1987), between two scalars `x = c / (r_ii + r_jj)`, a zero over a
     *        zero taken as 0 and a nonzero over a zero as infinite (no root there, as SciPy answers), else the small
     *        Sylvester system of the blocks (\ref small_sylvester_block; principal roots of eigenvalues off the
     *        negative axis never sum to 0 there).
     */
    template <typename T>
    void square_root_columns(const buffer<T>&                t,
                             buffer<T>&                      r,
                             std::size_t                     n,
                             const std::vector<std::size_t>& starts)
    {
      const std::size_t blocks {starts.size()};
      for (std::size_t bj {0}; bj < blocks; ++bj) {
        const std::size_t j0 {starts[bj]};
        const std::size_t nj {(bj + 1 < blocks ? starts[bj + 1] : n) - j0};
        for (std::size_t bi {bj}; bi > 0; --bi) {
          const std::size_t i0 {starts[bi - 1]};
          const std::size_t ni {starts[bi] - i0};
          std::array<T, 4>  c  {};
          for (std::size_t i {0}; i < ni; ++i) {
            for (std::size_t j {0}; j < nj; ++j) {
              T value {t[((i0 + i) * n) + j0 + j]};
              for (std::size_t k {starts[bi]}; k < j0; ++k) {
                value -= r[((i0 + i) * n) + k] * r[(k * n) + j0 + j];
              }
              c[(i * nj) + j] = value;
            }
          }
          if (ni == 1 && nj == 1) {
            const T sum {r[(i0 * n) + i0] + r[(j0 * n) + j0]};
            r[(i0 * n) + j0] = sum == T {} && c[0] == T {} ? T {} : c[0] / sum;
            continue;
          }
          small_sylvester_block(r.data() + (i0 * n) + i0, n, r.data() + (j0 * n) + j0, n, ni, nj, c.data(), nj, 0.0);
          for (std::size_t i {0}; i < ni; ++i) {
            for (std::size_t j {0}; j < nj; ++j) {
              r[((i0 + i) * n) + j0 + j] = c[(i * nj) + j];
            }
          }
        }
      }
    }

    /*!
     * \brief The real principal square root of the real \p a from its real Schur form (Higham, 1987), or none where
     *        `a` has a real eigenvalue below 0 (its principal root complex): a 1 × 1 block's root `√t`, a 2 × 2 block
     *        `[θ b; c θ]` of eigenvalues `θ ± iμ` the root `α I + (T − θ I) / 2α` for `α + iβ = √(θ + iμ)`.
     */
    [[nodiscard]] inline std::optional<matrix<double>> real_square_root(const matrix<double>& form,
                                                                        const matrix<double>* z)
    {
      const std::size_t        n {form.rows()};
      const buffer<double>     t(form.begin(), form.end());
      buffer<double>           r(n * n, 0.0);
      std::vector<std::size_t> starts;
      for (std::size_t k {0}; k < n; k += schur_block(t, n, k)) {
        starts.push_back(k);
        if (schur_block(t, n, k) == 1) {
          if (t[(k * n) + k] < 0.0) {
            return std::nullopt;
          }
          r[(k * n) + k] = std::sqrt(t[(k * n) + k]);
          continue;
        }
        const double theta {t[(k * n) + k]};
        const double mu    {std::sqrt(std::abs(t[(k * n) + k + 1])) * std::sqrt(std::abs(t[((k + 1) * n) + k]))};
        const double alpha {complex_sqrt(std::complex<double> {theta, mu}).real()};
        r[(k * n) + k]             = alpha;
        r[((k + 1) * n) + k + 1]   = alpha;
        r[(k * n) + k + 1]         = t[(k * n) + k + 1] / (2.0 * alpha);
        r[((k + 1) * n) + k]       = t[((k + 1) * n) + k] / (2.0 * alpha);
      }
      square_root_columns(t, r, n, starts);
      const matrix<double>                           root {n, n, std::move(r)};
      return z == nullptr ? root : *z * root       * transpose(*z);
    }

    //! \brief The principal square root of a complex matrix from its Schur form: `√t_ii` on the diagonal, the rest by
    //!        \ref square_root_columns.
    [[nodiscard]] inline matrix<std::complex<double>> complex_square_root(const matrix<std::complex<double>>& form,
                                                                          const matrix<std::complex<double>>* z)
    {
      const std::size_t                  n                             {form.rows()};
      const buffer<std::complex<double>> t(form.begin(), form.end());
      buffer<std::complex<double>>       r(n * n, std::complex<double> {});
      std::vector<std::size_t>           starts(n);
      for (std::size_t k {0}; k < n; ++k) {
        starts[k]      = k;
        r[(k * n) + k] = complex_sqrt(t[(k * n) + k]); // csqrt's, exact on the axes as NumPy's and SciPy's
      }
      square_root_columns(t, r, n, starts);
      const matrix<std::complex<double>>             root {n, n, std::move(r)};
      return z == nullptr ? root : *z * root       * adjoint_of(*z);
    }

    /*!
     * \brief The complex Schur form of a real one, as SciPy's `rsf2csf`: each 2 × 2 block of \p t (a conjugate pair)
     *        triangularised by a unitary rotation of its two rows and columns, the columns of \p z turning with it;
     *        a real eigenvalue's diagonal entry kept exactly real. \return `(T, Z)` complex.
     */
    [[nodiscard]] inline std::pair<matrix<std::complex<double>>, matrix<std::complex<double>>> complex_schur_form(const matrix<double>& t,
                                                                                                                  const matrix<double>& z)
    {
      using C = std::complex<double>;
      const std::size_t n {t.rows()};
      buffer<C>         tc(n * n);
      buffer<C>         zc(n * n);
      for (std::size_t e {0}; e < n * n; ++e) {
        tc[e] = C {t.begin()[static_cast<std::ptrdiff_t>(e)]};
        zc[e] = C {z.begin()[static_cast<std::ptrdiff_t>(e)]};
      }
      for (std::size_t k {n}; k > 1; --k) { // the rows m = k − 1 from the last to the second
        const std::size_t m     {k - 1};
        const C           below {tc[(m * n) + m - 1]};
        if (below == C {}) {
          continue;
        }
        const C a         {tc[((m - 1) * n) + m - 1]};
        const C b         {tc[((m - 1) * n) + m]};
        const C d         {tc[(m * n) + m]};
        // the first eigenvalue of the block less its last diagonal entry, as rsf2csf's mu[0]
        const C      half {(a - d) / 2.0};
        const C      mu   {half + complex_sqrt((half * half) + (b * below))};
        const double r    {std::hypot(std::abs(mu), std::abs(below))};
        const C      c    {mu / r};
        const C      sn   {below / r};
        for (std::size_t j {m - 1}; j < n; ++j) { // G = [c̄ s; −s c] on rows m − 1 and m
          const C upper {tc[((m - 1) * n) + j]};
          const C lower {tc[(m * n) + j]};
          tc[((m - 1) * n) + j] = (std::conj(c) * upper) + (sn * lower);
          tc[(m * n) + j]       = (-sn * upper) + (c * lower);
        }
        for (std::size_t i {0}; i <= m; ++i) { // · Gᴴ on columns m − 1 and m
          const C left  {tc[(i * n) + m - 1]};
          const C right {tc[(i * n) + m]};
          tc[(i * n) + m - 1] = (c * left) + (std::conj(sn) * right);
          tc[(i * n) + m]     = (-sn * left) + (std::conj(c) * right);
        }
        for (std::size_t i {0}; i < n; ++i) {
          const C left  {zc[(i * n) + m - 1]};
          const C right {zc[(i * n) + m]};
          zc[(i * n) + m - 1] = (c * left) + (std::conj(sn) * right);
          zc[(i * n) + m]     = (-sn * left) + (std::conj(c) * right);
        }
        tc[(m * n) + m - 1] = C {};
      }
      return {matrix<C> {n, n, std::move(tc)}, matrix<C> {n, n, std::move(zc)}};
    }

    //! \brief Whether the square \p a is upper triangular: its Schur form as it stands, `Z = I` (as SciPy takes it, so
    //!        an infinite entry of the root reaches no other).
    template <typename T>
    [[nodiscard]] bool upper_triangular(const matrix<T>& a)
    {
      for (std::size_t i {1}; i < a.rows(); ++i) {
        for (std::size_t j {0}; j < i; ++j) {
          if (a(i, j) != T {}) {
            return false;
          }
        }
      }
      return true;
    }
  } // namespace detail

  /*!
   * \brief The principal square root `X` of a complex square matrix (`X² = A`, its eigenvalues of nonnegative real
   *        part), as `scipy.linalg.sqrtm`: by the Schur method, `A = Z T Zᴴ`, the triangular `R² = T` column by column
   *        (Björck and Hammarling), `X = Z R Zᴴ`. Where `A` has no square root (a zero eigenvalue in a Jordan block)
   *        the entries that would need one are infinite, as SciPy's.
   * \throws dimension_error   If \p a is not square.
   * \throws non_finite_error  If an entry is infinite or NaN.
   * \throws convergence_error If the Schur form's iterations did not converge.
   */
  [[nodiscard]] inline matrix<std::complex<double>> sqrtm(const matrix<std::complex<double>>& a)
  {
    if (a.rows() != a.cols()) {
      throw dimension_error("sqrtm requires a square matrix");
    }
    detail::require_finite(a, "sqrtm");
    if (detail::upper_triangular(a)) {
      return detail::complex_square_root(a, nullptr);
    }
    const schur_decomposition<std::complex<double>> s {a};
    return detail::complex_square_root(s.t(), &s.z());
  }

  /*!
   * \brief The complex principal square root of a real square matrix, as `scipy.linalg.sqrtm` answers one with a real
   *        eigenvalue below 0: from its real Schur form made complex (SciPy's `rsf2csf`), its real eigenvalues exactly
   *        real, so each root of one below 0 is `+i √|λ|`.
   * \throws dimension_error   If \p a is not square.
   * \throws non_finite_error  If an entry is infinite or NaN.
   * \throws convergence_error If the Schur form's iterations did not converge.
   */
  [[nodiscard]] inline matrix<std::complex<double>> complex_sqrtm(const matrix<double>& a)
  {
    if (a.rows() != a.cols()) {
      throw dimension_error("sqrtm requires a square matrix");
    }
    detail::require_finite(a, "sqrtm");
    if (detail::upper_triangular(a)) {
      const matrix<std::complex<double>> form {detail::complex_schur_form(a, a).first};
      return detail::complex_square_root(form, nullptr);
    }
    const schur_decomposition<double> s {a};
    const auto [t, z] {detail::complex_schur_form(s.t(), s.z())};
    return detail::complex_square_root(t, &z);
  }

  /*!
   * \brief The real principal square root of a real square matrix, as `scipy.linalg.sqrtm` answers one: in real
   *        arithmetic on its real Schur form (Higham, 1987), a complex pair's block by its own real root.
   * \throws dimension_error   If \p a is not square.
   * \throws non_finite_error  If an entry is infinite or NaN.
   * \throws std::domain_error If \p a has a real eigenvalue below 0: its principal root is complex
   *                           (\ref complex_sqrtm).
   * \throws convergence_error If the Schur form's iterations did not converge.
   */
  [[nodiscard]] inline matrix<double> sqrtm(const matrix<double>& a)
  {
    if (a.rows() != a.cols()) {
      throw dimension_error("sqrtm requires a square matrix");
    }
    detail::require_finite(a, "sqrtm");
    std::optional<matrix<double>> root;
    if (detail::upper_triangular(a)) {
      root = detail::real_square_root(a, nullptr);
    }
    else {
      const schur_decomposition<double> s {a};
      root = detail::real_square_root(s.t(), &s.z());
    }
    if (!root) {
      throw std::domain_error("sqrtm: a real eigenvalue below 0, whose principal square root is complex");
    }
    return std::move(*root);
  }

  /*!
   * \brief A function of a square matrix from its complex Schur form, as `scipy.linalg.funm` forms it: `f(A)` and the
   *        estimate of its relative error.
   */
  struct funm_result
  {
    matrix<std::complex<double>> f;           //!< `f(A)`, complex (SciPy's real part where `A` is real and it is negligible).
    double                       error {0.0}; //!< `‖err‖₁ / ‖A‖₁` estimated as SciPy's: `min(1, max(ε, ε ‖strict triu T‖₁ / δ))`.
  };

  namespace detail {

    //! \brief A real matrix's complex Schur form as `scipy.linalg.funm` takes it: the real form made complex (`rsf2csf`).
    [[nodiscard]] inline std::pair<matrix<std::complex<double>>, matrix<std::complex<double>>> function_schur_form(const matrix<double>& a)
    {
      const schur_decomposition<double> s {a};
      return complex_schur_form(s.t(), s.z());
    }

    //! \brief A complex matrix's Schur form.
    [[nodiscard]] inline std::pair<matrix<std::complex<double>>, matrix<std::complex<double>>> function_schur_form(const matrix<std::complex<double>>& a)
    {
      const schur_decomposition<std::complex<double>> s {a};
      return {s.t(), s.z()};
    }
  } // namespace detail

  /*!
   * \brief `f(A)` from the complex Schur form `A = Z T Zᴴ` and `f` at `T`'s diagonal (\p values), as `scipy.linalg.funm`
   *        completes it: Parlett's recurrence (Golub and Van Loan's algorithm 11.1.1) on `F`, a superdiagonal at a time,
   *        `F(i, j) = (T(i, j) (F(j, j) − F(i, i)) + Σ T(i, k) F(k, j) − Σ F(i, k) T(k, j)) / (T(j, j) − T(i, i))` (not
   *        divided where two diagonal entries are equal, as SciPy's, the error estimate then 1), then `Z F Zᴴ`; `δ` the
   *        smallest `|T(j, j) − T(i, i)|` met (or `|T(0, 0)|`, `ε` where 0).
   * \throws dimension_error If \p t and \p z are not square of one order or \p values does not hold `n` entries.
   */
  [[nodiscard]] inline funm_result funm_from_schur(const matrix<std::complex<double>>&      t,
                                                   const matrix<std::complex<double>>&      z,
                                                   const std::vector<std::complex<double>>& values)
  {
    using C = std::complex<double>;
    const std::size_t n {t.rows()};
    if (t.cols() != n || z.rows() != n || z.cols() != n || values.size() != n) {
      throw dimension_error("funm requires a square Schur form T, Z of its order and one value per eigenvalue");
    }
    const double tolerance {std::numeric_limits<double>::epsilon()};
    if (n == 0) {
      return {matrix<C> {0, 0, buffer<C> {}}, tolerance};
    }
    buffer<C> f(n * n, C {});
    for (std::size_t i {0}; i < n; ++i) {
      f[(i * n) + i] = values[i];
    }
    double smallest {std::abs(t(0, 0))};
    for (std::size_t p {1}; p < n; ++p) {
      for (std::size_t i {0}; i + p < n; ++i) {
        const std::size_t j     {i + p};
        C                 sum   {t(i, j) * (f[(j * n) + j] - f[(i * n) + i])};
        C                 left  {};
        C                 right {};
        for (std::size_t k {i + 1}; k < j; ++k) {
          left  += t(i, k) * f[(k * n) + j];
          right += f[(i * n) + k] * t(k, j);
        }
        sum += left - right;
        const C gap {t(j, j) - t(i, i)};
        if (gap != C {}) {
          sum /= gap;
        }
        f[(i * n) + j] = sum;
        smallest       = std::min(smallest, std::abs(gap));
      }
    }
    smallest = smallest == 0.0 ? tolerance : smallest;
    double upper {0.0}; // ‖strict upper triangle of T‖₁
    for (std::size_t j {1}; j < n; ++j) {
      double column {0.0};
      for (std::size_t i {0}; i < j; ++i) {
        column += std::abs(t(i, j));
      }
      upper = std::max(upper, column);
    }
    const matrix<C> result {z * matrix<C> {n, n, std::move(f)} *detail::adjoint_of(z)};
    double          error  {std::min(1.0, std::max(tolerance, (tolerance / smallest) * upper))};
    for (const C& x : result) { // x − x is 0 for a finite part and NaN for an infinite or NaN one, without overflow
      error = (x.real() - x.real()) + (x.imag() - x.imag()) == 0.0 ? error : std::numeric_limits<double>::infinity();
    }
    return {result, error};
  }

  /*!
   * \brief A function of a square matrix, as `scipy.linalg.funm`: the Schur form (a real matrix's real form made complex
   *        as `rsf2csf`), \p f at its diagonal, called once with the eigenvalues in the form's order and returning one
   *        value each, then \ref funm_from_schur. Accurate where the eigenvalues are well apart (the estimate says how).
   * \tparam T `double` or `std::complex<double>`.
   * \throws dimension_error   If \p a is not square, or \p f does not return one value per eigenvalue.
   * \throws non_finite_error  If an entry is infinite or NaN.
   * \throws convergence_error If the Schur form's iterations did not converge.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] funm_result funm(const matrix<T>&                                                                                  a,
                                 const std::function<std::vector<std::complex<double>>(const std::vector<std::complex<double>>&)>& f)
  {
    if (a.rows() != a.cols()) {
      throw dimension_error("funm requires a square matrix");
    }
    detail::require_finite(a, "funm");
    const auto [t, z] {detail::function_schur_form(a)};
    std::vector<std::complex<double>> diagonal(t.rows());
    for (std::size_t i {0}; i < t.rows(); ++i) {
      diagonal[i] = t(i, i);
    }
    return funm_from_schur(t, z, f(diagonal));
  }

  /*!
   * \brief The principal logarithm of a square matrix with what SciPy warns of: `log A` (complex), and whether `A`'s
   *        triangular form was exactly singular (its zero diagonal entries taken as `10⁻²⁰`, as SciPy's) or nearly
   *        (a diagonal entry below `10⁻²⁰` in modulus).
   */
  struct logm_result
  {
    matrix<std::complex<double>> log;                      //!< `log A`.
    bool                         exactly_singular {false}; //!< A zero diagonal entry of the triangular form was replaced.
    bool                         nearly_singular  {false}; //!< A diagonal entry of the triangular form is below `10⁻²⁰`.
  };

  namespace detail {

    //! \brief The bounds `θ_m` of the `[m/m]` Padé approximant of `log(I + X)` (Al-Mohy and Higham's table 2.1, 2012),
    //!        `m` from 1 to 7.
    [[nodiscard]] inline double log_pade_bound(std::size_t m)
    {
      static constexpr std::array<double, 8> theta {0.0, 1.59e-5, 2.31e-3, 1.94e-2, 6.21e-2, 1.28e-1, 2.06e-1, 2.88e-1};
      return theta[m];
    }

    //! \brief `ceil((Im z − π) / 2π)`, the unwinding number of `z` (SciPy's `_unwindk`).
    [[nodiscard]] inline double unwinding(const std::complex<double>& z)
    {
      return std::ceil((z.imag() - std::numbers::pi) / (2.0 * std::numbers::pi));
    }

    /*!
     * \brief `a^(1/2^k) − 1` without the cancellation of the subtraction, as SciPy's `_briggs_helper_function` (Al-Mohy
     *        and Higham's algorithm 2, 2012): `(a − 1) / Π (1 + a^(1/2^j))`, a first square root taken where `arg a ≥
     *        π/2`.
     */
    [[nodiscard]] inline std::complex<double> briggs(std::complex<double> a,
                                                     std::size_t          k)
    {
      using C = std::complex<double>;
      if (k == 0) {
        return a - C {1.0};
      }
      if (k == 1) {
        return complex_sqrt(a) - C {1.0};
      }
      std::size_t remaining {k};
      if (std::arg(a) >= std::numbers::pi / 2.0) {
        a = complex_sqrt(a);
        --remaining;
      }
      const C z0 {a - C {1.0}};
      a = complex_sqrt(a);
      C product  {C {1.0} + a};
      for (std::size_t j {1}; j < remaining; ++j) {
        a        = complex_sqrt(a);
        product *= C {1.0} + a;
      }
      return z0 / product;
    }

    //! \brief The superdiagonal entry of `T^p` for the 2 × 2 upper triangular `[[l1, t12], [0, l2]]`, as SciPy's
    //!         `_fractional_power_superdiag_entry` (Higham and Lin's equation 5.5, 2011).
    [[nodiscard]] inline std::complex<double> fractional_power_superdiagonal(const std::complex<double>& l1,
                                                                             const std::complex<double>& l2,
                                                                             const std::complex<double>& t12,
                                                                             double                      p)
    {
      using C = std::complex<double>;
      if (l1 == l2) {
        return t12 * p * complex_exp((p - 1.0) * complex_log(l1));
      }
      if (std::abs(l2 - l1) > std::abs(l1 + l2) / 2.0) {
        return t12 * (complex_exp(p * complex_log(l2)) - complex_exp(p * complex_log(l1))) / (l2 - l1);
      }
      const C      z {(l2 - l1) / (l2 + l1)};
      const C      log1 {complex_log(l1)};
      const C      log2 {complex_log(l2)};
      const C      scaled {t12 * complex_exp((p / 2.0) * (log2 + log1))};
      const double u {unwinding(log2 - log1)};
      const C      b {p * (complex_atanh(z) + (std::numbers::pi * C {0.0, 1.0} *u))};
      return scaled * (C {2.0} *complex_sinh(b) / (l2 - l1));
    }

    //! \brief The superdiagonal entry of `log T` for the 2 × 2 upper triangular `[[l1, t12], [0, l2]]`, as SciPy's
    //!         `_logm_superdiag_entry`.
    [[nodiscard]] inline std::complex<double> log_superdiagonal(const std::complex<double>& l1,
                                                                const std::complex<double>& l2,
                                                                const std::complex<double>& t12)
    {
      using C = std::complex<double>;
      if (l1 == l2) {
        return t12 / l1;
      }
      if (std::abs(l2 - l1) > std::abs(l1 + l2) / 2.0) {
        return t12 * (complex_log(l2) - complex_log(l1)) / (l2 - l1);
      }
      const C      z {(l2 - l1) / (l2 + l1)};
      const double u {unwinding(complex_log(l2) - complex_log(l1))};
      return t12 * C {2.0} *(complex_atanh(z) + (std::numbers::pi * C {0.0, 1.0} *u)) / (l2 - l1);
    }

    //! \brief `‖(T − I)ᵖ‖₁^(1/p)` of the upper triangular `n × n` \p t, exactly (SciPy estimates it, `onenormest`).
    [[nodiscard]] inline double shifted_power_norm(const buffer<std::complex<double>>& t,
                                                   std::size_t                         n,
                                                   std::size_t                         p)
    {
      buffer<std::complex<double>> shifted {t};
      for (std::size_t i {0}; i < n; ++i) {
        shifted[(i * n) + i] -= std::complex<double> {1.0};
      }
      buffer<std::complex<double>> power {shifted};
      for (std::size_t k {1}; k < p; ++k) {
        power = square_product(power, shifted, n);
      }
      return std::pow(one_norm(power, n), 1.0 / static_cast<double>(p));
    }

    //! \brief Whether `log` has its principal branch at every diagonal entry of \p t (none real and at most 0).
    [[nodiscard]] inline bool principal_branch(const buffer<std::complex<double>>& t,
                                               std::size_t                         n)
    {
      for (std::size_t i {0}; i < n; ++i) {
        const std::complex<double> lambda {t[(i * n) + i]};
        if (!(lambda.real() > 0.0 || lambda.imag() != 0.0)) {
          return false;
        }
      }
      return true;
    }

    /*!
     * \brief Al-Mohy and Higham's inverse scaling and squaring (2012, algorithm 4.1) on the upper triangular \p t0, as
     *        SciPy's `_inverse_squaring_helper`: `s` square roots of `T` until `T − I` is small enough for the `[m/m]`
     *        Padé approximant of `log(I + X)` (`m` from 1 to 7), the norms `‖(T − I)ᵖ‖₁^(1/p)` exact; `R = T − I` with
     *        its diagonal and superdiagonal recomputed without cancellation where the principal branch exists.
     * \return `R`, `s` and `m`.
     */
    [[nodiscard]] inline std::tuple<buffer<std::complex<double>>, std::size_t, std::size_t> inverse_squaring(const buffer<std::complex<double>>& t0,
                                                                                                             std::size_t                         n)
    {
      using C = std::complex<double>;
      std::size_t s0 {0};
      buffer<C>   diagonal(n);
      for (std::size_t i {0}; i < n; ++i) {
        diagonal[i] = t0[(i * n) + i];
      }
      const auto far = [&diagonal]() {
                         double largest {0.0};
                         for (const C& d : diagonal) {
                           largest = std::max(largest, std::abs(d - C {1.0}));
                         }
                         return largest > log_pade_bound(7);
                       };
      while (far()) {
        for (C& d : diagonal) {
          d = complex_sqrt(d);
        }
        ++s0;
      }
      const auto root = [n](const buffer<C>& t) {
                          return buffer<C> {complex_square_root(matrix<C> {n, n, buffer<C> {t}}, nullptr).elements()};
                        };
      buffer<C> t {t0};
      for (std::size_t i {0}; i < s0; ++i) {
        t = root(t);
      }
      std::size_t  s  {s0};
      double       d3 {shifted_power_norm(t, n, 3)};
      const double a2 {std::max(shifted_power_norm(t, n, 2), d3)};
      std::size_t  m  {a2 <= log_pade_bound(1) ? std::size_t {1} : (a2 <= log_pade_bound(2) ? std::size_t {2} : std::size_t {0})};
      while (m == 0) {
        if (s > s0) {
          d3 = shifted_power_norm(t, n, 3);
        }
        const double d4 {shifted_power_norm(t, n, 4)};
        const double a3 {std::max(d3, d4)};
        if (a3 <= log_pade_bound(7)) {
          std::size_t j1 {3};
          while (a3 > log_pade_bound(j1)) {
            ++j1;
          }
          if (j1 <= 6) {
            m = j1;
            break;
          }
          // a square root about halves a3, so one in (θ₆, θ₇] falls to θ₆ or below: the paper's cap of two such roots
          // (SciPy's k < 2) is never reached
          if (a3 / 2.0 <= log_pade_bound(5)) {
            t = root(t);
            ++s;
            continue;
          }
        }
        const double d5  {shifted_power_norm(t, n, 5)};
        const double eta {std::min(a3, std::max(d4, d5))};
        m = eta <= log_pade_bound(6) ? std::size_t {6} : (eta <= log_pade_bound(7) ? std::size_t {7} : std::size_t {0});
        if (m == 0) {
          t = root(t);
          ++s;
        }
      }
      for (std::size_t i {0}; i < n; ++i) {
        t[(i * n) + i] -= C {1.0};
      }
      if (principal_branch(t0, n)) {
        const double p {std::ldexp(1.0, -static_cast<int>(s))};
        for (std::size_t j {0}; j < n; ++j) {
          t[(j * n) + j] = briggs(t0[(j * n) + j], s);
          if (j + 1 < n) {
            t[(j * n) + j + 1] = fractional_power_superdiagonal(t0[(j * n) + j], t0[((j + 1) * n) + j + 1], t0[(j * n) + j + 1], p);
          }
        }
      }
      return {std::move(t), s, m};
    }

    /*!
     * \brief The logarithm of the upper triangular \p t0 (no zero on its diagonal), as SciPy's `_logm_triu`: `R`, `s`
     *        and `m` by \ref inverse_squaring, `U = 2ˢ Σ αₖ (I + βₖ R)⁻¹ R` (the `[m/m]` Padé approximant as the
     *        partial fractions of `m`-point Gauss–Legendre quadrature on [0, 1], SciPy's nodes and weights), then the
     *        diagonal `log λᵢ` and the superdiagonal recomputed exactly where the principal branch exists.
     */
    [[nodiscard]] inline buffer<std::complex<double>> triangular_log(const buffer<std::complex<double>>& t0,
                                                                     std::size_t                         n)
    {
      using C = std::complex<double>;
      static constexpr std::array<std::array<double, 7>, 7> nodes {{
        {0.5},
        {0.21132486540518708, 0.7886751345948129},
        {0.1127016653792583, 0.5, 0.8872983346207417},
        {0.06943184420297371, 0.33000947820757187, 0.6699905217924281, 0.9305681557970262},
        {0.04691007703066802, 0.23076534494715845, 0.5, 0.7692346550528415, 0.9530899229693319},
        {0.033765242898423975, 0.16939530676686776, 0.3806904069584015, 0.6193095930415985, 0.8306046932331322, 0.966234757101576},
        {0.0254460438286207, 0.12923440720030277, 0.2970774243113014, 0.5, 0.7029225756886985, 0.8707655927996972, 0.9745539561713793}}};
      static constexpr std::array<std::array<double, 7>, 7> weights {{
        {1.0},
        {0.5, 0.5},
        {0.2777777777777779, 0.44444444444444414, 0.2777777777777779},
        {0.17392742256872679, 0.3260725774312732, 0.3260725774312732, 0.17392742256872679},
        {0.11846344252809478, 0.2393143352496831, 0.2844444444444443, 0.2393143352496831, 0.11846344252809478},
        {0.08566224618958473, 0.1803807865240696, 0.2339569672863457, 0.2339569672863457, 0.1803807865240696, 0.08566224618958473},
        {0.06474248308443482, 0.13985269574463827, 0.19091502525255954, 0.20897959183673465, 0.19091502525255954, 0.13985269574463827,
         0.06474248308443482}}};
      auto [r, s, m] {inverse_squaring(t0, n)};
      buffer<C> u(n * n, C {});
      for (std::size_t q {0}; q < m; ++q) {
        const double alpha {weights[m - 1][q]};
        const double beta  {nodes[m - 1][q]};
        buffer<C>    shifted(n * n);
        buffer<C>    x(n * n);
        for (std::size_t e {0}; e < n * n; ++e) {
          shifted[e] = beta * r[e];
          x[e]       = alpha * r[e];
        }
        for (std::size_t i {0}; i < n; ++i) {
          shifted[(i * n) + i] += C {1.0};
        }
        solve_triangular_in_place(shifted.data(), n, false, triangular_op::none, false, x.data(), n);
        for (std::size_t e {0}; e < n * n; ++e) {
          u[e] += x[e];
        }
      }
      const double power {std::ldexp(1.0, static_cast<int>(s))};
      for (C& x : u) {
        x *= power;
      }
      if (principal_branch(t0, n)) {
        for (std::size_t i {0}; i < n; ++i) {
          u[(i * n) + i] = complex_log(t0[(i * n) + i]);
          if (i + 1 < n) {
            u[(i * n) + i + 1] = log_superdiagonal(t0[(i * n) + i], t0[((i + 1) * n) + i + 1], t0[(i * n) + i + 1]);
          }
        }
      }
      return u;
    }

    //! \brief The triangular form \p t made nonsingular as SciPy's `_logm_force_nonsingular_triangular_matrix`: a zero
    //!         diagonal entry replaced by `10⁻²⁰`. \return Whether one was (exactly singular), and whether one was
    //!         below `10⁻²⁰` otherwise (nearly singular).
    [[nodiscard]] inline std::pair<bool, bool> force_nonsingular(buffer<std::complex<double>>& t,
                                                                 std::size_t                   n)
    {
      constexpr double tiny  {1e-20};
      bool             exact {false};
      bool             near  {false};
      for (std::size_t i {0}; i < n; ++i) {
        const std::complex<double>& d {t[(i * n) + i]};
        exact = exact || std::abs(d) == 0.0;
        near  = near || std::abs(d) < tiny;
      }
      if (exact) {
        for (std::size_t i {0}; i < n; ++i) {
          t[(i * n) + i] = std::abs(t[(i * n) + i]) == 0.0 ? std::complex<double> {tiny} : t[(i * n) + i];
        }
      }
      return {exact, near && !exact};
    }
  } // namespace detail

  /*!
   * \brief The principal logarithm of a square matrix, as `scipy.linalg.logm`: an upper triangular `A` directly, else
   *        its Schur form `A = Z T Zᴴ` (a real matrix's real form made complex) and `Z log(T) Zᴴ`; `log T` by Al-Mohy and
   *        Higham's inverse scaling and squaring (2012) as SciPy's `_logm_triu`, the norms that choose the square roots
   *        and the degree exact where SciPy estimates them. A zero eigenvalue is taken as `10⁻²⁰` (no logarithm exists
   *        there), as SciPy's, and reported; complex, SciPy's real part where `A` is real and it is negligible.
   * \tparam T `double` or `std::complex<double>`.
   * \throws dimension_error   If \p a is not square.
   * \throws non_finite_error  If an entry is infinite or NaN.
   * \throws convergence_error If the Schur form's iterations did not converge.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] logm_result logm(const matrix<T>& a)
  {
    using C = std::complex<double>;
    const std::size_t n {a.rows()};
    if (a.cols() != n) {
      throw dimension_error("logm requires a square matrix");
    }
    detail::require_finite(a, "logm");
    if (n == 0) {
      return {matrix<C> {0, 0, buffer<C> {}}, false, false};
    }
    if (detail::upper_triangular(a)) {
      buffer<C> t(a.begin(), a.end());
      const auto [exact, near] {detail::force_nonsingular(t, n)};
      return {matrix<C> {n, n, detail::triangular_log(t, n)}, exact, near};
    }
    const auto [form, z] {detail::function_schur_form(a)};
    buffer<C> t(form.begin(), form.end());
    const auto [exact, near] {detail::force_nonsingular(t, n)};
    const matrix<C> u {n, n, detail::triangular_log(t, n)};
    return {z * u * detail::adjoint_of(z), exact, near};
  }
} // namespace numerical

#endif // NUMERICAL_MATRIX_FUNCTIONS_HPP
