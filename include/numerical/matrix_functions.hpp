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
#include <limits>
#include <optional>
#include <stdexcept>
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
} // namespace numerical

#endif // NUMERICAL_MATRIX_FUNCTIONS_HPP
