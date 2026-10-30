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
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "complex_math.hpp"
#include "eigen.hpp"

namespace numerical {

  namespace detail {

    //! \brief The moduli of a real matrix's entries.
    [[nodiscard]] inline buffer<double> entry_moduli(const buffer<double>& a)
    {
      buffer<double> moduli(a.size());
      std::transform(a.begin(), a.end(), moduli.begin(), [](double x) { return std::abs(x); });
      return moduli;
    }

    /*!
     * \brief The moduli of a complex matrix's entries: `√(x² + y²)`, which vectorises, where the largest part lies in
     *        [2⁻⁵⁰⁰, 2⁵⁰⁰] (no square overflows; one that underflows is of an entry below 2⁻⁵⁰⁰ times the largest's,
     *        nothing to a norm); `hypot` otherwise, as `std::abs` takes it.
     */
    [[nodiscard]] inline buffer<double> entry_moduli(const buffer<std::complex<double>>& a)
    {
      double largest {0.0};
      for (const std::complex<double>& z : a) {
        largest = std::max({largest, std::abs(z.real()), std::abs(z.imag())});
      }
      buffer<double> moduli(a.size());
      if (largest >= std::ldexp(1.0, -500) && largest <= std::ldexp(1.0, 500)) {
        std::transform(a.begin(), a.end(), moduli.begin(), [](const std::complex<double>& z) {
                         const double square {z.real() * z.real()};
                         return std::sqrt(square + (z.imag() * z.imag()));
                       });
      }
      else {
        std::transform(a.begin(), a.end(), moduli.begin(), [](const std::complex<double>& z) { return std::abs(z); });
      }
      return moduli;
    }

    //! \brief The 1-norm of the `n × n` row-major matrix whose entries' moduli are \p moduli (`n` from 1): its largest
    //!         column sum.
    [[nodiscard]] inline double column_sum_norm(const buffer<double>& moduli,
                                                std::size_t           n)
    {
      buffer<double> sums(n, 0.0);
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          sums[j] += moduli[(i * n) + j];
        }
      }
      return *std::max_element(sums.begin(), sums.end());
    }

    //! \brief The 1-norm of the `n × n` row-major \p a (`n` from 1): its largest column sum of moduli.
    template <typename T>
    [[nodiscard]] double one_norm(const buffer<T>& a,
                                  std::size_t      n)
    {
      return column_sum_norm(entry_moduli(a), n);
    }

    //! \brief `‖ |A|ᵖ ‖₁` of the `n × n` row-major \p moduli `|A|` (`n` from 1) times \p scale, exactly: the row
    //!        `1ᵀ |scale A|ᵖ` by \p p products of a vector and `|A|`, a matrix of moduli's norm its largest entry.
    [[nodiscard]] inline double modulus_power_norm(const buffer<double>& moduli,
                                                   std::size_t           n,
                                                   std::size_t           p,
                                                   double                scale)
    {
      buffer<double> row(n, 1.0);
      buffer<double> next(n, 0.0);
      for (std::size_t step {0}; step < p; ++step) {
        std::fill(next.begin(), next.end(), 0.0);
        for (std::size_t i {0}; i < n; ++i) {
          for (std::size_t j {0}; j < n; ++j) {
            next[j] += row[i] * moduli[(i * n) + j] * scale;
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
     *        `A` times \p scale (a power of two), its norms exact, from its \p moduli (`|A|`, formed once for every
     *        degree tried: a complex entry's modulus is a `hypot`).
     */
    [[nodiscard]] inline std::size_t extra_squarings(const buffer<double>& moduli,
                                                     std::size_t           n,
                                                     std::size_t           m,
                                                     double                scale,
                                                     double                norm)
    {
      const auto   order       {static_cast<double>(m)}; // c = (m!)² / ((2m)! (2m + 1)!), by the logarithms of the factorials
      const double c           {std::exp((2.0 * std::lgamma(order + 1.0)) - std::lgamma((2.0 * order) + 1.0) - std::lgamma((2.0 * order) + 2.0))};
      const double scaled_norm {norm * scale};
      if (scaled_norm == 0.0) {
        return 0;
      }
      const double alpha {c * modulus_power_norm(moduli, n, (2 * m) + 1, scale) / scaled_norm};
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

    //! \brief Whether \p a is Hermitian to the bit (symmetric where real), its diagonal real: a function of it is then
    //!        `V f(Λ) Vᴴ` from one eigendecomposition (Armadillo's `expmat_sym`, `logmat_sympd`, `sqrtmat_sympd`).
    template <typename T>
    [[nodiscard]] bool exactly_hermitian(const matrix<T>& a)
    {
      const std::size_t n {a.rows()};
      for (std::size_t i {0}; i < n; ++i) {
        if (a(i, i) != conj_of(a(i, i))) {
          return false;
        }
        for (std::size_t j {0}; j < i; ++j) {
          if (a(i, j) != conj_of(a(j, i))) {
            return false;
          }
        }
      }
      return true;
    }

    //! \brief The eigenvalues (ascending) and orthonormal eigenvectors of the real symmetric \p a.
    [[nodiscard]] inline std::pair<vector<double>, matrix<double>> hermitian_system(const matrix<double>& a)
    {
      symmetric_eigen e      {a};
      vector<double>  values {e.values()};
      return {std::move(values), std::move(e).vectors()};
    }

    //! \brief The eigenvalues (ascending) and orthonormal eigenvectors of the complex Hermitian \p a.
    [[nodiscard]] inline std::pair<vector<double>, matrix<std::complex<double>>> hermitian_system(const matrix<std::complex<double>>& a)
    {
      hermitian_eigen e      {a};
      vector<double>  values {e.values()};
      return {std::move(values), std::move(e).vectors()};
    }

    /*!
     * \brief `V diag(f) Vᴴ` in the arithmetic of \p R (`double` only for real \p v and \p f), the eigenvectors \p v
     *        orthonormal: one product. Made Hermitian to the bit where \p hermitian (every `f` real).
     */
    template <typename R, typename T>
    [[nodiscard]] matrix<R> spectral(const matrix<T>&      v,
                                     const std::vector<R>& f,
                                     bool                  hermitian)
    {
      const std::size_t n {v.rows()};
      buffer<R>         scaled(n * n);
      buffer<R>         adjoint(n * n);
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t k {0}; k < n; ++k) {
          scaled[(i * n) + k]  = R {v(i, k)} *f[k];
          adjoint[(k * n) + i] = conj_of(R {v(i, k)});
        }
      }
      matrix<R> x {matrix<R> {n, n, std::move(scaled)} *matrix<R> {n, n, std::move(adjoint)}};
      if (!hermitian) {
        return x;
      }
      buffer<R> out(n * n);
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t k {0}; k < n; ++k) {
          out[(i * n) + k] = (x(i, k) + conj_of(x(k, i))) / 2.0;
        }
      }
      return matrix<R> {n, n, std::move(out)};
    }
  } // namespace detail

  //! \brief The order from which \ref expm of a Hermitian matrix takes its eigendecomposition (below, the Padé path is
  //!        faster: an order-32 \ref symmetric_eigen costs more than the few products of a small Padé degree).
  inline constexpr std::size_t hermitian_expm_least {64};

  /*!
   * \brief The exponential `e^A` of a square matrix, as `scipy.linalg.expm`: scaling and squaring of a Padé
   *        approximant, as Al-Mohy and Higham's algorithm (2009): the degree 3, 5, 7, 9 or 13 and the squarings `s`
   *        chosen from `‖A^{2k}‖₁^{1/2k}` (exact, `A²`, `A⁴` and `A⁶` formed, `A⁸` and `A¹⁰` where needed) and the
   *        approximant's backward error bound (`ℓ`), then `r(2⁻ˢ A)` squared `s` times; an upper triangular `A`'s
   *        diagonal and first superdiagonal written exactly at each squaring. A Hermitian `A` (to the bit) of order
   *        \ref hermitian_expm_least or more: `V e^Λ Vᴴ` from its eigendecomposition.
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
    if (n >= hermitian_expm_least && detail::exactly_hermitian(a)) { // one eigendecomposition: below, the Padé path is faster
      const auto [values, vectors] {detail::hermitian_system(a)};
      std::vector<T> powers(n);
      for (std::size_t k {0}; k < n; ++k) {
        powers[k] = T {std::exp(values[k])};
      }
      return detail::spectral<T>(vectors, powers, true);
    }
    const buffer<T> a0(a.begin(), a.end());
    bool            triangular {true};
    for (std::size_t i {1}; i < n && triangular; ++i) {
      for (std::size_t j {0}; j < i; ++j) {
        triangular = triangular && a0[(i * n) + j] == T {};
      }
    }
    const buffer<double>            moduli {detail::entry_moduli(a0)};
    const double                    norm {detail::column_sum_norm(moduli, n)};
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
    if (eta1 <= theta[0] && detail::extra_squarings(moduli, n, 3, 1.0, norm) == 0) {
      return finish(detail::pade_approximant(a0, powers, n, 3));
    }
    if (eta1 <= theta[1] && detail::extra_squarings(moduli, n, 5, 1.0, norm) == 0) {
      return finish(detail::pade_approximant(a0, powers, n, 5));
    }
    const buffer<T> a8 {detail::square_product(a4, a4, n)};
    powers[3] = &a8;
    const double d8    {std::pow(detail::one_norm(a8, n), 1.0 / 8.0)};
    const double eta3  {std::max(d6, d8)};
    if (eta3 <= theta[2] && detail::extra_squarings(moduli, n, 7, 1.0, norm) == 0) {
      return finish(detail::pade_approximant(a0, powers, n, 7));
    }
    if (eta3 <= theta[3] && detail::extra_squarings(moduli, n, 9, 1.0, norm) == 0) {
      return finish(detail::pade_approximant(a0, powers, n, 9));
    }
    // η₅ = min(η₃, max(d₈, d₁₀)) is η₃ where η₃ ≤ θ₁₃ (s = 0 whatever d₁₀) or d₈ ≥ η₃: A¹⁰ formed only otherwise
    const auto   d10  = [&] { return std::pow(detail::one_norm(detail::square_product(a4, a6, n), n), 1.0 / 10.0); };
    const double eta5 {eta3 <= theta[4] || d8 >= eta3 ? eta3 : std::min(eta3, std::max(d8, d10()))};
    std::size_t  s    {eta5 <= theta[4] ? 0 : static_cast<std::size_t>(std::ceil(std::log2(eta5 / theta[4])))};
    s += detail::extra_squarings(moduli, n, 13, std::ldexp(1.0, -static_cast<int>(s)), norm);
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
      buffer<T>         columns(2 * n); // the block column's entries above its diagonal block, by column: contiguous
      for (std::size_t bj {0}; bj < blocks; ++bj) {
        const std::size_t j0 {starts[bj]};
        const std::size_t nj {(bj + 1 < blocks ? starts[bj + 1] : n) - j0};
        for (std::size_t bi {bj}; bi > 0; --bi) {
          const std::size_t i0 {starts[bi - 1]};
          const std::size_t ni {starts[bi] - i0};
          std::array<T, 4>  c  {};
          for (std::size_t i {0}; i < ni; ++i) {
            for (std::size_t j {0}; j < nj; ++j) {
              T              value  {t[((i0 + i) * n) + j0 + j]};
              const T* const row    {r.data() + ((i0 + i) * n)};
              const T* const column {columns.data() + (j * n)};
              for (std::size_t k {starts[bi]}; k < j0; ++k) {
                value -= row[k] * column[k];
              }
              c[(i * nj) + j] = value;
            }
          }
          if (ni == 1 && nj == 1) {
            const T sum {r[(i0 * n) + i0] + r[(j0 * n) + j0]};
            c[0] = sum == T {} && c[0] == T {} ? T {} : c[0] / sum;
          }
          else {
            small_sylvester_block(r.data() + (i0 * n) + i0, n, r.data() + (j0 * n) + j0, n, ni, nj, c.data(), nj, 0.0);
          }
          for (std::size_t i {0}; i < ni; ++i) {
            for (std::size_t j {0}; j < nj; ++j) {
              r[((i0 + i) * n) + j0 + j] = c[(i * nj) + j];
              columns[(j * n) + i0 + i]  = c[(i * nj) + j];
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

    //! \brief `√x` of a real \p x at least 0.
    [[nodiscard]] inline double scalar_sqrt(double x)
    {
      return std::sqrt(x);
    }

    //! \brief csqrt's `√z`, exact on the axes as NumPy's and SciPy's.
    [[nodiscard]] inline std::complex<double> scalar_sqrt(const std::complex<double>& z)
    {
      return complex_sqrt(z);
    }

    //! \brief The principal square root of the upper triangular `n × n` row-major \p t (no real eigenvalue below 0
    //!        where `T` is real), entry by entry: `√t_ii` on the diagonal, the rest by \ref square_root_columns.
    template <typename T>
    [[nodiscard]] buffer<T> pointwise_square_root(const buffer<T>& t,
                                                  std::size_t      n)
    {
      buffer<T>                r(n * n, T {});
      std::vector<std::size_t> starts(n);
      for (std::size_t k {0}; k < n; ++k) {
        starts[k]      = k;
        r[(k * n) + k] = scalar_sqrt(t[(k * n) + k]);
      }
      square_root_columns(t, r, n, starts);
      return r;
    }

    //! \brief The order of the diagonal blocks of \ref triangular_square_root (SciPy's `_sqrtm_triu`'s).
    inline constexpr std::size_t square_root_block {64};

    /*!
     * \brief The principal square root of the upper triangular `n × n` row-major \p t (no real eigenvalue below 0
     *        where `T` is real): \ref pointwise_square_root up to \ref square_root_block or where a diagonal entry is 0
     *        (a zero over a zero taken as 0 there); else by blocks (Deadman, Higham and Ralha, 2013, as SciPy's
     *        `_sqrtm_triu`): each diagonal block's root entry by entry, then each block column from the diagonal up,
     *        `R_ii X + X R_jj = T_ij − R_i,(i+1…j−1) R_(i+1…j−1),j`, the product by \ref gemm and the Sylvester equation by
     *        \ref triangular_sylvester (`R_ii + R_jj` has no eigenvalue at 0: principal roots of nonzero eigenvalues).
     */
    template <typename T>
    [[nodiscard]] buffer<T> triangular_square_root(const buffer<T>& t,
                                                   std::size_t      n)
    {
      bool singular {false};
      for (std::size_t k {0}; k < n; ++k) {
        singular = singular || t[(k * n) + k] == T {};
      }
      if (n <= square_root_block || singular) {
        return pointwise_square_root(t, n);
      }
      const std::size_t      blocks {(n + square_root_block - 1) / square_root_block};
      const auto             start  = [n, blocks](std::size_t b) { return b == blocks ? n : b * square_root_block; };
      std::vector<buffer<T>> roots(blocks); // R_bb, contiguous
      buffer<T>              r(n * n, T {});
      for (std::size_t b {0}; b < blocks; ++b) {
        const std::size_t b0 {start(b)};
        const std::size_t nb {start(b + 1) - b0};
        buffer<T>         block(nb * nb);
        for (std::size_t i {0}; i < nb; ++i) {
          std::copy_n(t.data() + ((b0 + i) * n) + b0, nb, block.data() + (i * nb));
        }
        roots[b] = pointwise_square_root(block, nb);
        for (std::size_t i {0}; i < nb; ++i) {
          std::copy_n(roots[b].data() + (i * nb), nb, r.data() + ((b0 + i) * n) + b0);
        }
      }
      buffer<T> f;
      buffer<T> product;
      for (std::size_t bj {1}; bj < blocks; ++bj) {
        const std::size_t j0 {start(bj)};
        const std::size_t nj {start(bj + 1) - j0};
        for (std::size_t bi {bj}; bi > 0; --bi) {
          const std::size_t i0 {start(bi - 1)};
          const std::size_t ni {start(bi) - i0};
          f.resize(ni * nj);
          product.resize(ni * nj);
          gemm(ni, nj, j0 - start(bi), r.data() + (i0 * n) + start(bi), n, r.data() + (start(bi) * n) + j0, n, product.data(), nj, c_start::zero);
          for (std::size_t i {0}; i < ni; ++i) {
            for (std::size_t j {0}; j < nj; ++j) {
              f[(i * nj) + j] = t[((i0 + i) * n) + j0 + j] - product[(i * nj) + j];
            }
          }
          triangular_sylvester(roots[bi - 1], ni, roots[bj], nj, f);
          for (std::size_t i {0}; i < ni; ++i) {
            std::copy_n(f.data() + (i * nj), nj, r.data() + ((i0 + i) * n) + j0);
          }
        }
      }
      return r;
    }

    //! \brief The principal square root of a complex matrix from its Schur form (\ref triangular_square_root).
    [[nodiscard]] inline matrix<std::complex<double>> complex_square_root(const matrix<std::complex<double>>& form,
                                                                          const matrix<std::complex<double>>* z)
    {
      const std::size_t                              n                             {form.rows()};
      const buffer<std::complex<double>>             t(form.begin(), form.end());
      const matrix<std::complex<double>>             root                          {n, n, triangular_square_root(t, n)};
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
        // the block scaled by a power of two near its largest entry: b·c can neither overflow (an entry past √DBL_MAX)
        // nor vanish, and the rotation, a ratio, is unchanged (a power of two rounds nothing)
        const double largest {std::max({std::abs(tc[((m - 1) * n) + m - 1]), std::abs(tc[((m - 1) * n) + m]), std::abs(tc[(m * n) + m]),
                                        std::abs(below)})};
        const double scale   {std::ldexp(1.0, -std::ilogb(largest))};
        const C      a       {tc[((m - 1) * n) + m - 1] * scale};
        const C      b       {tc[((m - 1) * n) + m] * scale};
        const C      d       {tc[(m * n) + m] * scale};
        const C      c_s     {below * scale};
        // the first eigenvalue of the block less its last diagonal entry, as rsf2csf's mu[0]
        const C      half    {(a - d) / 2.0};
        const C      mu      {half + complex_sqrt((half * half) + (b * c_s))};
        const double r       {std::hypot(std::abs(mu), std::abs(c_s))};
        const C      c       {mu / r};
        const C      sn      {c_s / r};
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

  namespace detail {

    //! \brief The principal square root of the Hermitian \p a, `V √Λ Vᴴ`: Hermitian where no eigenvalue is below 0.
    template <typename T>
    [[nodiscard]] matrix<std::complex<double>> hermitian_root(const matrix<T>& a)
    {
      const auto [values, vectors] {hermitian_system(a)};
      std::vector<std::complex<double>> roots(values.size());
      for (std::size_t k {0}; k < roots.size(); ++k) {
        roots[k] = complex_sqrt(std::complex<double> {values[k]});
      }
      return spectral<std::complex<double>>(vectors, roots, std::all_of(values.begin(), values.end(), [](double v) { return v >= 0.0; }));
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
    if (detail::exactly_hermitian(a)) {
      return detail::hermitian_root(a);
    }
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
    if (detail::exactly_hermitian(a)) {
      return detail::hermitian_root(a);
    }
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
    if (detail::exactly_hermitian(a)) {
      const auto [values, vectors] {detail::hermitian_system(a)};
      if (!std::all_of(values.begin(), values.end(), [](double v) { return v >= 0.0; })) {
        throw std::domain_error("sqrtm: a real eigenvalue below 0, whose principal square root is complex");
      }
      std::vector<double> roots(values.size());
      for (std::size_t k {0}; k < roots.size(); ++k) {
        roots[k] = std::sqrt(values[k]);
      }
      return detail::spectral<double>(vectors, roots, true);
    }
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
    buffer<C> tc(n * n); // T and F by columns: the recurrence's sums contiguous along both operands
    buffer<C> fc(n * n, C {});
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        tc[(j * n) + i] = t(i, j);
      }
      fc[(i * n) + i] = values[i];
    }
    double smallest {std::abs(t(0, 0))};
    for (std::size_t p {1}; p < n; ++p) {
      for (std::size_t i {0}; i + p < n; ++i) {
        const std::size_t j     {i + p};
        C                 sum   {t(i, j) * (f[(j * n) + j] - f[(i * n) + i])};
        C                 left  {};
        C                 right {};
        const C* const    t_row {&t(i, 0)};
        const C* const    f_row {f.data() + (i * n)};
        const C* const    t_col {tc.data() + (j * n)};
        const C* const    f_col {fc.data() + (j * n)};
        for (std::size_t k {i + 1}; k < j; ++k) {
          left  += t_row[k] * f_col[k];
          right += f_row[k] * t_col[k];
        }
        sum += left - right;
        const C gap {t(j, j) - t(i, i)};
        if (gap != C {}) {
          sum /= gap;
        }
        f[(i * n) + j]  = sum;
        fc[(j * n) + i] = sum;
        smallest        = std::min(smallest, std::abs(gap));
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

  namespace detail {

    //! \brief The clusters of \p values joined where two lie within \p delta (transitively), numbered by their first
    //!        member's place.
    [[nodiscard]] inline std::vector<std::size_t> eigenvalue_clusters(const std::vector<std::complex<double>>& values,
                                                                      double                                   delta)
    {
      const std::size_t        n {values.size()};
      std::vector<std::size_t> id(n);
      for (std::size_t i {0}; i < n; ++i) {
        id[i] = i;
      }
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {i + 1}; j < n; ++j) {
          if (std::abs(values[i] - values[j]) <= delta && id[j] != id[i]) { // j's cluster joins i's
            const std::size_t from {id[j]};
            const std::size_t to   {id[i]};
            for (std::size_t& k : id) {
              k = k == from ? to : k;
            }
          }
        }
      }
      std::vector<std::size_t> label(n, n);
      std::size_t              next {0};
      for (std::size_t i {0}; i < n; ++i) {
        if (label[id[i]] == n) {
          label[id[i]] = next++;
        }
        id[i] = label[id[i]];
      }
      return id;
    }

    //! \brief The clusters' bound: eigenvalues within `10⁻³` of the largest modulus (or of 1) gather.
    [[nodiscard]] inline double cluster_bound(const std::vector<std::complex<double>>& values)
    {
      double scale {1.0};
      for (const std::complex<double>& v : values) {
        scale = std::max(scale, std::abs(v));
      }
      return 1e-3 * scale;
    }

    //! \brief The diagonal of the square \p t.
    [[nodiscard]] inline std::vector<std::complex<double>> diagonal_of(const matrix<std::complex<double>>& t)
    {
      std::vector<std::complex<double>> values(t.rows());
      for (std::size_t i {0}; i < t.rows(); ++i) {
        values[i] = t(i, i);
      }
      return values;
    }

    //! \brief The circles of Cauchy's integral about the clusters of the reordered Schur form \p t (each a run of its
    //!        diagonal): the runs' starts (and `n`), and each run's centre and radius (0 for a single eigenvalue).
    struct cluster_circles
    {
      std::vector<std::size_t>          starts;  //!< Each run's first place, then `n`.
      std::vector<std::complex<double>> centres; //!< Each run's eigenvalues' mean.
      std::vector<double>               radii;   //!< Each run's circle's radius; 0 for a single eigenvalue.
    };

    //! \brief The trapezoidal rule's points on each circle of Cauchy's integral.
    inline constexpr std::size_t contour_points {64};

    /*!
     * \brief \ref cluster_circles of the Schur form \p t whose clusters are runs of its diagonal: a circle about each
     *        cluster of two or more, its radius the cluster's spread plus a margin as wide as the spread, the cluster
     *        bound, or the block's coupling (its largest entry above the diagonal: the resolvent's terms `‖N‖ᵏ / rᵏ⁺¹`
     *        then stay near `1/r` and cancel without loss), half the way to the nearest other eigenvalue at most.
     */
    [[nodiscard]] inline cluster_circles circles_of(const matrix<std::complex<double>>& t)
    {
      using C = std::complex<double>;
      const std::size_t              n       {t.rows()};
      const std::vector<C>           values  {diagonal_of(t)};
      const double                   delta   {cluster_bound(values)};
      const std::vector<std::size_t> cluster {eigenvalue_clusters(values, delta)};
      cluster_circles                circles;
      for (std::size_t i {0}; i < n; ++i) {
        if (i == 0 || cluster[i] != cluster[i - 1]) {
          circles.starts.push_back(i);
        }
      }
      const std::size_t blocks {circles.starts.size()};
      circles.starts.push_back(n);
      circles.centres.assign(blocks, C {});
      circles.radii.assign(blocks, 0.0);
      for (std::size_t b {0}; b < blocks; ++b) {
        const std::size_t first {circles.starts[b]};
        const std::size_t last  {circles.starts[b + 1]};
        if (last - first == 1) {
          continue;
        }
        C centre {};
        for (std::size_t i {first}; i < last; ++i) {
          centre += values[i];
        }
        centre /= static_cast<double>(last - first);
        double spread   {0.0};
        double outside  {std::numeric_limits<double>::infinity()};
        double coupling {0.0};
        for (std::size_t i {0}; i < n; ++i) {
          const bool   member   {i >= first && i < last};
          const double distance {std::abs(values[i] - centre)};
          spread  = member ? std::max(spread, distance) : spread;
          outside = member ? outside : std::min(outside, distance);
          for (std::size_t j {i + 1}; member && j < last; ++j) {
            coupling = std::max(coupling, std::abs(t(i, j)));
          }
        }
        circles.centres[b] = centre;
        circles.radii[b]   = spread + std::min(std::max({spread, delta, coupling}), (outside - spread) / 2.0);
      }
      return circles;
    }

    /*!
     * \brief Reorders the complex Schur form \p t, \p z so each cluster of eigenvalues (\ref eigenvalue_clusters, within
     *        \ref cluster_bound) is a run of the diagonal, each in turn moved to the first place past those placed.
     *        \return Whether any cluster holds two or more eigenvalues (none moved otherwise).
     */
    [[nodiscard]] inline bool gather_clusters(matrix<std::complex<double>>& t,
                                              matrix<std::complex<double>>& z)
    {
      using C = std::complex<double>;
      const std::size_t              n      {t.rows()};
      std::vector<C>                 values {diagonal_of(t)};
      const double                   delta  {cluster_bound(values)};
      const std::vector<std::size_t> first  {eigenvalue_clusters(values, delta)};
      if (*std::max_element(first.begin(), first.end()) + 1 == n) { // every cluster a single eigenvalue (n ≥ 1: an empty matrix is Hermitian)
        return false;
      }
      buffer<C>   form(t.begin(), t.end());
      buffer<C>   zt     {transpose(z).elements()};
      std::size_t placed {0};
      while (placed < n) {
        for (std::size_t i {0}; i < n; ++i) {
          values[i] = form[(i * n) + i];
        }
        const std::vector<std::size_t> cluster {eigenvalue_clusters(values, delta)};
        std::vector<bool>              chosen(n);
        for (std::size_t i {0}; i < n; ++i) {
          chosen[i] = i < placed || cluster[i] == cluster[placed];
        }
        placed = move_leading(form, zt, n, chosen);
      }
      t = matrix<C> {n, n, std::move(form)};
      z = transpose(matrix<C> {n, n, std::move(zt)});
      return true;
    }

    /*!
     * \brief `f(T)` of the Schur form \p t with clusters as runs of its diagonal, from \p images, `f` at the eigenvalues
     *        then at each cluster's circle (\ref circles_of), as Davies and Higham's Schur–Parlett algorithm (2003):
     *        each cluster's block by Cauchy's integral `(1/2πi) ∮ f(z) (zI − T)⁻¹ dz`, the trapezoidal rule at
     *        \ref contour_points points (Hale, Higham and Trefethen, 2008), its 32 even points' result and its diagonal
     *        against `f` at the eigenvalues the estimate of its error, where Davies and Higham's Taylor series wants
     *        `f`'s derivatives; the blocks above the diagonal by the block recurrence, a triangular Sylvester equation
     *        each. The estimate: SciPy's for Parlett's recurrence with the clusters' separation, at least the
     *        integrals'; ∞ where `f` or the result is not finite. \return `f(T)` and the estimate.
     */
    [[nodiscard]] inline std::pair<buffer<std::complex<double>>, double> clustered_function(const matrix<std::complex<double>>&      t,
                                                                                            const std::vector<std::complex<double>>& images)
    {
      using C = std::complex<double>;
      const std::size_t     n             {t.rows()};
      const cluster_circles circles       {circles_of(t)};
      const std::size_t     blocks        {circles.radii.size()};
      buffer<C>             fx(n * n, C   {});
      double                contour_error {0.0};
      std::size_t           next          {n}; // the next circle's first image
      for (std::size_t b {0}; b < blocks; ++b) {
        const std::size_t first {circles.starts[b]};
        const std::size_t p     {circles.starts[b + 1] - first};
        if (p == 1) {
          fx[(first * n) + first] = images[first];
          continue;
        }
        buffer<C> whole(p * p, C {});
        buffer<C> half(p * p, C  {});
        for (std::size_t k {0}; k < contour_points; ++k) {
          const C   point            {circles.centres[b] + std::polar(circles.radii[b], 2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(contour_points))};
          buffer<C> shifted(p * p, C {});
          buffer<C> inverse(p * p, C {});
          for (std::size_t i {0}; i < p; ++i) {
            for (std::size_t j {i}; j < p; ++j) {
              shifted[(i * p) + j] = (i == j ? point : C {}) - t(first + i, first + j);
            }
            inverse[(i * p) + i] = C {1.0};
          }
          solve_triangular_in_place(shifted.data(), p, false, triangular_op::none, false, inverse.data(), p);
          const C weight {images[next + k] * (point - circles.centres[b])};
          for (std::size_t e {0}; e < p * p; ++e) {
            whole[e] += weight * inverse[e];
            half[e]  += k % 2 == 0 ? weight * inverse[e] : C {};
          }
        }
        next += contour_points;
        double size {0.0};
        double gap  {0.0};
        for (std::size_t e {0}; e < p * p; ++e) {
          whole[e] /= static_cast<double>(contour_points);
          half[e]  /= static_cast<double>(contour_points) / 2.0;
          size      = std::max(size, std::abs(whole[e]));
          gap       = std::max(gap, std::abs(whole[e] - half[e]));
        }
        for (std::size_t i {0}; i < p; ++i) { // a singularity of f in the disk (no answer there) breaks the diagonal
          gap = std::max(gap, std::abs(whole[(i * p) + i] - images[first + i]));
        }
        contour_error = std::max(contour_error, size > 0.0 ? gap / size : gap);
        for (std::size_t i {0}; i < p; ++i) {
          std::copy_n(whole.data() + (i * p), p, fx.data() + ((first + i) * n) + first);
        }
      }
      double separation {std::numeric_limits<double>::infinity()};
      for (std::size_t d {1}; d < blocks; ++d) { // the block recurrence, a block superdiagonal at a time
        for (std::size_t bi {0}; bi + d < blocks; ++bi) {
          const std::size_t bj             {bi + d};
          const std::size_t i0             {circles.starts[bi]};
          const std::size_t pi             {circles.starts[bi + 1] - i0};
          const std::size_t j0             {circles.starts[bj]};
          const std::size_t pj             {circles.starts[bj + 1] - j0};
          buffer<C>         rhs(pi * pj, C {}); // F_ii T_ij − T_ij F_jj + Σ (F_ik T_kj − T_ik F_kj), F_ij still 0
          for (std::size_t i {0}; i < pi; ++i) {
            for (std::size_t j {0}; j < pj; ++j) {
              C sum {};
              for (std::size_t q {i0}; q < j0 + pj; ++q) {
                sum += (fx[((i0 + i) * n) + q] * t(q, j0 + j)) - (t(i0 + i, q) * fx[(q * n) + j0 + j]);
              }
              rhs[(i * pj) + j] = sum;
              separation        = std::min(separation, std::abs(t(i0 + i, i0 + i) - t(j0 + j, j0 + j)));
            }
          }
          buffer<C> left(pi * pi);
          buffer<C> right(pj * pj);
          for (std::size_t i {0}; i < pi; ++i) {
            for (std::size_t j {0}; j < pi; ++j) {
              left[(i * pi) + j] = t(i0 + i, i0 + j);
            }
          }
          for (std::size_t i {0}; i < pj; ++i) {
            for (std::size_t j {0}; j < pj; ++j) {
              right[(i * pj) + j] = -t(j0 + i, j0 + j);
            }
          }
          triangular_sylvester(left, pi, right, pj, rhs);
          for (std::size_t i {0}; i < pi; ++i) {
            std::copy_n(rhs.data() + (i * pj), pj, fx.data() + ((i0 + i) * n) + j0);
          }
        }
      }
      double upper {0.0}; // ‖strict upper triangle of T‖₁, as funm_from_schur's estimate
      for (std::size_t j {1}; j < n; ++j) {
        double column {0.0};
        for (std::size_t i {0}; i < j; ++i) {
          column += std::abs(t(i, j));
        }
        upper = std::max(upper, column);
      }
      const double tolerance {std::numeric_limits<double>::epsilon()};
      return {std::move(fx), std::min(1.0, std::max({tolerance, (tolerance / separation) * upper, contour_error}))};
    }
  } // namespace detail

  /*!
   * \brief Where \ref funm wants `f`, and the forms it completes: a Hermitian `A`'s eigendecomposition (\p t `Λ`, \p z
   *        `V`); else its complex Schur form (a real matrix's real form made complex, as `rsf2csf`), reordered so the
   *        eigenvalues within `10⁻³` of the largest modulus (or of 1) gather in runs. \p points: the eigenvalues in
   *        the form's order, then \ref detail::contour_points on the circle about each run of two or more.
   */
  struct funm_plan
  {
    matrix<std::complex<double>>      t;                 //!< `T` (`Λ`, for a Hermitian `A`).
    matrix<std::complex<double>>      z;                 //!< `Z` (`V`).
    std::vector<std::complex<double>> points;            //!< Where `f` is wanted.
    bool                              hermitian {false}; //!< Whether `A` was Hermitian, `T` then `Λ`.
  };

  /*!
   * \brief The \ref funm_plan of \p a: `f` is then wanted at its points, and \ref finish_funm completes `f(A)`.
   * \tparam T `double` or `std::complex<double>`.
   * \throws dimension_error   If \p a is not square.
   * \throws non_finite_error  If an entry is infinite or NaN.
   * \throws convergence_error If the Schur form's iterations did not converge.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] funm_plan plan_funm(const matrix<T>& a)
  {
    using C = std::complex<double>;
    if (a.rows() != a.cols()) {
      throw dimension_error("funm requires a square matrix");
    }
    detail::require_finite(a, "funm");
    const std::size_t n {a.rows()};
    funm_plan         plan;
    if (detail::exactly_hermitian(a)) {
      const auto [values, vectors] {detail::hermitian_system(a)};
      buffer<C> diagonal(n * n, C {});
      for (std::size_t i {0}; i < n; ++i) {
        diagonal[(i * n) + i] = C {values[i]};
      }
      plan.t         = matrix<C> {n, n, std::move(diagonal)};
      plan.z         = matrix<C> {n, n, buffer<C>(vectors.begin(), vectors.end())};
      plan.points    = detail::diagonal_of(plan.t);
      plan.hermitian = true;
      return plan;
    }
    std::tie(plan.t, plan.z) = detail::function_schur_form(a);
    plan.points              = detail::diagonal_of(plan.t);
    if (detail::gather_clusters(plan.t, plan.z)) {
      plan.points                                 = detail::diagonal_of(plan.t);
      const detail::cluster_circles circles {detail::circles_of(plan.t)};
      for (std::size_t b {0}; b < circles.radii.size(); ++b) {
        for (std::size_t k {0}; circles.radii[b] > 0.0 && k < detail::contour_points; ++k) {
          plan.points.push_back(circles.centres[b] + std::polar(circles.radii[b], 2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(detail::contour_points)));
        }
      }
    }
    return plan;
  }

  /*!
   * \brief `f(A)` from its \ref funm_plan and \p images, `f` at the plan's points: a Hermitian `A`'s `V f(Λ) Vᴴ`, its
   *        estimate `ε`; where eigenvalues gather, Davies and Higham's Schur–Parlett algorithm with each cluster by
   *        Cauchy's integral (\ref detail::clustered_function); else Parlett's recurrence, as SciPy's
   *        (\ref funm_from_schur). The estimate ∞ where `f` at an eigenvalue, or the result, is not finite.
   * \throws dimension_error If \p images does not hold one value per point.
   */
  [[nodiscard]] inline funm_result finish_funm(const funm_plan&                         plan,
                                               const std::vector<std::complex<double>>& images)
  {
    using C = std::complex<double>;
    const std::size_t n {plan.t.rows()};
    if (images.size() != plan.points.size()) {
      throw dimension_error("funm requires a square Schur form T, Z of its order and one value per eigenvalue");
    }
    const std::vector<C> at_eigenvalues(images.begin(), images.begin() + static_cast<std::ptrdiff_t>(n));
    if (!plan.hermitian && images.size() == n) {
      return funm_from_schur(plan.t, plan.z, at_eigenvalues);
    }
    matrix<C> x;
    double    error {std::numeric_limits<double>::epsilon()};
    if (plan.hermitian) {
      x = detail::spectral<C>(plan.z, at_eigenvalues, false);
    }
    else {
      auto [f_of_t, estimate] {detail::clustered_function(plan.t, images)};
      x     = plan.z * matrix<C> {n, n, std::move(f_of_t)} *detail::adjoint_of(plan.z);
      error = estimate;
    }
    for (const C& entry : x) {
      error = detail::is_finite_entry(entry) ? error : std::numeric_limits<double>::infinity();
    }
    for (const C& image : at_eigenvalues) { // f not finite on the spectrum: no f(A), whatever the circles gave
      error = detail::is_finite_entry(image) ? error : std::numeric_limits<double>::infinity();
    }
    return {std::move(x), error};
  }

  /*!
   * \brief A function of a square matrix, as `scipy.linalg.funm`: \ref plan_funm, \p f called once with the plan's
   *        points (the eigenvalues, then the circles' where eigenvalues gather) returning one value each, then
   *        \ref finish_funm. \p f must be analytic on each circle's disk. Accurate where the eigenvalues are well apart
   *        or gather tightly; between, Parlett's recurrence (the estimate says how).
   * \tparam T `double` or `std::complex<double>`.
   * \throws dimension_error   If \p a is not square, or \p f does not return one value per point.
   * \throws non_finite_error  If an entry is infinite or NaN.
   * \throws convergence_error If the Schur form's iterations did not converge.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] funm_result funm(const matrix<T>&                                                                                  a,
                                 const std::function<std::vector<std::complex<double>>(const std::vector<std::complex<double>>&)>& f)
  {
    const funm_plan plan {plan_funm(a)};
    return finish_funm(plan, f(plan.points));
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
    inline constexpr std::array<double, 8> log_pade_bounds {0.0, 1.59e-5, 2.31e-3, 1.94e-2, 6.21e-2, 1.28e-1, 2.06e-1, 2.88e-1};

    //! \brief The bounds `θ_m` of the `[m/m]` Padé approximant of `(I − X)ᵗ` (Higham and Lin's table 3.1, 2013, as
    //!        SciPy's `_remainder_matrix_power_triu`), `m` from 1 to 7.
    inline constexpr std::array<double, 8> power_pade_bounds {0.0, 1.51e-5, 2.24e-3, 1.88e-2, 6.04e-2, 1.24e-1, 2.00e-1, 2.79e-1};


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

    //! \brief \ref briggs of a real \p a above 0: `(a − 1) / Π (1 + a^(1/2^j))`, for `k = 1` too (where `√a − 1`
    //!        would cancel).
    [[nodiscard]] inline double briggs(double      a,
                                       std::size_t k)
    {
      if (k == 0) {
        return a - 1.0;
      }
      const double z0 {a - 1.0};
      a = std::sqrt(a);
      double product  {1.0 + a};
      for (std::size_t j {1}; j < k; ++j) {
        a        = std::sqrt(a);
        product *= 1.0 + a;
      }
      return z0 / product;
    }

    //! \brief \ref fractional_power_superdiagonal of real \p l1 and \p l2 above 0 (the unwinding number 0).
    [[nodiscard]] inline double fractional_power_superdiagonal(double l1,
                                                               double l2,
                                                               double t12,
                                                               double p)
    {
      if (l1 == l2) {
        return t12 * p * std::exp((p - 1.0) * std::log(l1));
      }
      if (std::abs(l2 - l1) > std::abs(l1 + l2) / 2.0) {
        return t12 * (std::exp(p * std::log(l2)) - std::exp(p * std::log(l1))) / (l2 - l1);
      }
      const double scaled {t12 * std::exp((p / 2.0) * (std::log(l2) + std::log(l1)))};
      return scaled * (2.0 * std::sinh(p * std::atanh((l2 - l1) / (l2 + l1))) / (l2 - l1));
    }

    //! \brief \ref log_superdiagonal of real \p l1 and \p l2 above 0 (the unwinding number 0).
    [[nodiscard]] inline double log_superdiagonal(double l1,
                                                  double l2,
                                                  double t12)
    {
      if (l1 == l2) {
        return t12 / l1;
      }
      if (std::abs(l2 - l1) > std::abs(l1 + l2) / 2.0) {
        return t12 * (std::log(l2) - std::log(l1)) / (l2 - l1);
      }
      return t12 * 2.0 * std::atanh((l2 - l1) / (l2 + l1)) / (l2 - l1);
    }

    //! \brief `log x` of a real \p x above 0.
    [[nodiscard]] inline double scalar_log(double x)
    {
      return std::log(x);
    }

    //! \brief \ref complex_log of \p z.
    [[nodiscard]] inline std::complex<double> scalar_log(const std::complex<double>& z)
    {
      return complex_log(z);
    }

    /*!
     * \brief `‖(T − I)ᵖ‖₁^(1/p)` of the upper triangular `n × n` \p t, `p` from 2, exactly (SciPy estimates them,
     *        `onenormest`): each power the one before times `T − I`, formed once whatever order the norms are asked in.
     */
    template <typename T>
    class shifted_power_norms
    {
    public:

      shifted_power_norms(const buffer<T>& t,
                          std::size_t      n) : shifted_ {t}, n_ {n}
      {
        for (std::size_t i {0}; i < n; ++i) {
          shifted_[(i * n) + i] -= T {1.0};
        }
        power_ = shifted_;
      }

      [[nodiscard]] double operator()(std::size_t p)
      {
        while (norms_.size() + 1 < p) {
          power_ = square_product(power_, shifted_, n_);
          norms_.push_back(one_norm(power_, n_));
        }
        return std::pow(norms_[p - 2], 1.0 / static_cast<double>(p));
      }

    private:

      buffer<T>           shifted_;
      buffer<T>           power_;
      std::size_t         n_;
      std::vector<double> norms_; //!< `‖(T − I)ᵖ‖₁` for `p` from 2.
    };

    //! \brief `log` has its principal branch at every diagonal entry of a real triangular form \ref logm keeps real:
    //!        none below 0, and its zeros taken as `10⁻²⁰` (\ref force_nonsingular).
    [[nodiscard]] inline bool principal_branch(const buffer<double>& /*t*/,
                                               std::size_t /*n*/)
    {
      return true;
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
     *        Padé approximant whose bounds are \p theta (`log(I + X)`'s, or `(I − X)ᵗ`'s; `m` from 1 to 7), the norms `‖(T − I)ᵖ‖₁^(1/p)` exact; `R = T − I` with
     *        its diagonal and superdiagonal recomputed without cancellation where the principal branch exists.
     * \return `R`, `s` and `m`.
     */
    template <typename T>
    [[nodiscard]] std::tuple<buffer<T>, std::size_t, std::size_t> inverse_squaring(const buffer<T>&             t0,
                                                                                   std::size_t                  n,
                                                                                   const std::array<double, 8>& theta = log_pade_bounds,
                                                                                   const char*                  name  = "logm")
    {
      const auto  bound = [&theta](std::size_t m) { return theta[m]; };
      std::size_t s0 {0};
      buffer<T>   diagonal(n);
      for (std::size_t i {0}; i < n; ++i) {
        diagonal[i] = t0[(i * n) + i];
      }
      const auto far = [&diagonal, &bound]() {
                         double largest {0.0};
                         for (const T& d : diagonal) {
                           largest = std::max(largest, std::abs(d - T {1.0}));
                         }
                         return largest > bound(7);
                       };
      while (far()) {
        for (T& d : diagonal) {
          d = scalar_sqrt(d);
        }
        ++s0;
      }
      buffer<T> t {t0};
      for (std::size_t i {0}; i < s0; ++i) {
        t = triangular_square_root(t, n);
      }
      std::size_t            s     {s0};
      shifted_power_norms<T> norms {t, n};
      const double           a2    {std::max(norms(2), norms(3))};
      std::size_t            m     {a2 <= bound(1) ? std::size_t {1} : (a2 <= bound(2) ? std::size_t {2} : std::size_t {0})};
      while (m == 0) {
        const double d3 {norms(3)};
        const double d4 {norms(4)};
        const double a3 {std::max(d3, d4)};
        if (!std::isfinite(a3)) { // NaN compares false with every bound: the roots would go on for ever
          throw convergence_error(std::string {name} + ": the square roots of the triangular form are not finite");
        }
        if (a3 <= bound(7)) {
          std::size_t j1 {3};
          while (a3 > bound(j1)) {
            ++j1;
          }
          if (j1 <= 6) {
            m = j1;
            break;
          }
          // a square root about halves a3, so one in (θ₆, θ₇] falls to θ₆ or below: the paper's cap of two such roots
          // (SciPy's k < 2) is never reached
          if (a3 / 2.0 <= bound(5)) {
            t     = triangular_square_root(t, n);
            norms = shifted_power_norms<T> {t, n};
            ++s;
            continue;
          }
        }
        const double d5  {norms(5)};
        const double eta {std::min(a3, std::max(d4, d5))};
        m = eta <= bound(6) ? std::size_t {6} : (eta <= bound(7) ? std::size_t {7} : std::size_t {0});
        if (m == 0) {
          t     = triangular_square_root(t, n);
          norms = shifted_power_norms<T> {t, n};
          ++s;
        }
      }
      for (std::size_t i {0}; i < n; ++i) {
        t[(i * n) + i] -= T {1.0};
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
    template <typename T>
    [[nodiscard]] buffer<T> triangular_log(const buffer<T>& t0,
                                           std::size_t      n)
    {
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
      buffer<T> u(n * n, T {});
      buffer<T> shifted(n * n); // each node's I + β R and α R, written whole
      buffer<T> term(n * n);
      for (std::size_t q {0}; q < m; ++q) {
        const double alpha {weights[m - 1][q]};
        const double beta  {nodes[m - 1][q]};
        for (std::size_t e {0}; e < n * n; ++e) {
          shifted[e] = beta * r[e];
          term[e]    = alpha * r[e];
        }
        for (std::size_t i {0}; i < n; ++i) {
          shifted[(i * n) + i] += T {1.0};
        }
        solve_triangular_in_place(shifted.data(), n, false, triangular_op::none, false, term.data(), n);
        for (std::size_t e {0}; e < n * n; ++e) {
          u[e] += term[e];
        }
      }
      const double power {std::ldexp(1.0, static_cast<int>(s))};
      for (T& x : u) {
        x *= power;
      }
      if (principal_branch(t0, n)) {
        for (std::size_t i {0}; i < n; ++i) {
          u[(i * n) + i] = scalar_log(t0[(i * n) + i]);
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
    template <typename T>
    [[nodiscard]] std::pair<bool, bool> force_nonsingular(buffer<T>&  t,
                                                          std::size_t n)
    {
      constexpr double tiny  {1e-20};
      bool             exact {false};
      bool             near  {false};
      for (std::size_t i {0}; i < n; ++i) {
        const T& d {t[(i * n) + i]};
        exact = exact || std::abs(d) == 0.0;
        near  = near || std::abs(d) < tiny;
      }
      if (exact) {
        for (std::size_t i {0}; i < n; ++i) {
          t[(i * n) + i] = std::abs(t[(i * n) + i]) == 0.0 ? T {tiny} : t[(i * n) + i];
        }
      }
      return {exact, near && !exact};
    }
  } // namespace detail

  namespace detail {

    //! \brief \p a itself: already complex.
    [[nodiscard]] inline const matrix<std::complex<double>>& as_complex(const matrix<std::complex<double>>& a)
    {
      return a;
    }

    //! \brief \p a's entries as complex numbers.
    [[nodiscard]] inline matrix<std::complex<double>> as_complex(const matrix<double>& a)
    {
      return matrix<std::complex<double>> {a.rows(), a.cols(), buffer<std::complex<double>>(a.begin(), a.end())};
    }

    //! \brief `z` to the real power \p p, principal: `exp(p log z)` (0 for `z = 0` and `p > 0`).
    [[nodiscard]] inline std::complex<double> principal_power(const std::complex<double>& z,
                                                              double                      p)
    {
      return complex_exp(p * complex_log(z));
    }

    //! \brief The coefficient `cᵢ` (`i ≥ 1`) of the continued fraction of `(1 − x)ᵗ`, as SciPy's
    //!        `_fractional_power_pade_constant` (Higham and Lin, 2011, equation 4.1).
    [[nodiscard]] inline double power_pade_constant(std::size_t i,
                                                    double      t)
    {
      if (i == 1) {
        return -t;
      }
      if (i % 2 == 0) {
        const double j {static_cast<double>(i) / 2.0}; // i even: exact
        return (t - j) / (2.0 * ((2.0 * j) - 1.0));
      }
      const double j {(static_cast<double>(i) - 1.0) / 2.0}; // i odd: exact
      return (-j - t) / (2.0 * ((2.0 * j) + 1.0));
    }

    /*!
     * \brief The `[m/m]` Padé approximant of `(I − X)ᵗ` for the upper triangular \p x, as SciPy's
     *        `_fractional_power_pade`: its continued fraction from the bottom, `Y = c₂ₘ X`, then `Y = (I + Y)⁻¹ cⱼ X`
     *        for `j = 2m − 1, …, 1`, and `I + Y`.
     */
    [[nodiscard]] inline buffer<std::complex<double>> power_pade(const buffer<std::complex<double>>& x,
                                                                 std::size_t                         n,
                                                                 double                              t,
                                                                 std::size_t                         m)
    {
      using C = std::complex<double>;
      buffer<C> y(n * n);
      for (std::size_t e {0}; e < n * n; ++e) {
        y[e] = x[e] * power_pade_constant(2 * m, t);
      }
      buffer<C> rhs(n * n); // each step's cⱼ X, written whole, then Y's
      for (std::size_t j {(2 * m) - 1}; j >= 1; --j) {
        for (std::size_t e {0}; e < n * n; ++e) {
          rhs[e] = x[e] * power_pade_constant(j, t);
        }
        for (std::size_t i {0}; i < n; ++i) {
          y[(i * n) + i] += C {1.0};
        }
        solve_triangular_in_place(y.data(), n, false, triangular_op::none, false, rhs.data(), n);
        std::swap(y, rhs);
      }
      for (std::size_t i {0}; i < n; ++i) {
        y[(i * n) + i] += C {1.0};
      }
      return y;
    }

    /*!
     * \brief `T^t` (`−1 < t < 1`) of the upper triangular \p t0 (no zero on its diagonal), as SciPy's
     *        `_remainder_matrix_power_triu` (Higham and Lin's Schur–Padé algorithm, 2013): a diagonal `T` entry by entry;
     *        else `R = T^(1/2ˢ) − I` and `m` by \ref inverse_squaring with the bounds of `(I − X)ᵗ`, the `[m/m]` Padé
     *        approximant at `−R`, then `s` squarings. Where the principal branch exists, the diagonal `λᵢ^(t/2ⁱ)` and the
     *        superdiagonal (\ref fractional_power_superdiagonal) are written exactly after each squaring, as Higham and
     *        Lin's algorithm 5.1 does (SciPy writes them before the first squaring only).
     */
    [[nodiscard]] inline buffer<std::complex<double>> triangular_power(const buffer<std::complex<double>>& t0,
                                                                       std::size_t                         n,
                                                                       double                              t)
    {
      using C = std::complex<double>;
      bool diagonal {true};
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {i + 1}; j < n; ++j) {
          diagonal = diagonal && t0[(i * n) + j] == C {};
        }
      }
      if (diagonal) {
        buffer<C> u(n * n, C {});
        for (std::size_t i {0}; i < n; ++i) {
          u[(i * n) + i] = principal_power(t0[(i * n) + i], t);
        }
        return u;
      }
      auto [r, s, m] {inverse_squaring(t0, n, power_pade_bounds, "fractional_matrix_power")};
      for (C& x : r) {
        x = -x;
      }
      buffer<C>  u     {power_pade(r, n, t, m)};
      const bool exact {principal_branch(t0, n)};
      for (std::size_t i {s + 1}; i-- > 0;) {
        if (i < s) {
          u = square_product(u, u, n);
        }
        if (exact) {
          const double p {std::ldexp(t, -static_cast<int>(i))};
          for (std::size_t j {0}; j < n; ++j) {
            u[(j * n) + j] = principal_power(t0[(j * n) + j], p);
            if (j + 1 < n) {
              u[(j * n) + j + 1] = fractional_power_superdiagonal(t0[(j * n) + j], t0[((j + 1) * n) + j + 1], t0[(j * n) + j + 1], p);
            }
          }
        }
      }
      return u;
    }

    //! \brief `A^t` (`−1 < t < 1`) by the Schur form, as SciPy's `_remainder_matrix_power`: an upper triangular `A`
    //!        directly, else `Z T^t Zᴴ`; none where the triangular form has a zero on its diagonal.
    template <typename T>
    [[nodiscard]] std::optional<matrix<std::complex<double>>> remainder_power(const matrix<T>& a,
                                                                              double           t)
    {
      using C = std::complex<double>;
      const std::size_t n          {a.rows()};
      const bool        triangular {upper_triangular(a)};
      matrix<C>         form;
      matrix<C>         z;
      if (triangular) {
        form = as_complex(a);
      }
      else {
        std::tie(form, z) = function_schur_form(a);
      }
      for (std::size_t i {0}; i < n; ++i) {
        if (form(i, i) == C {}) {
          return std::nullopt;
        }
      }
      matrix<C> u {n, n, triangular_power(buffer<C>(form.begin(), form.end()), n, t)};
      if (triangular) {
        return u;
      }
      return z * u * adjoint_of(z);
    }

    //! \brief \p a to the integer power \p e (a finite whole number) by repeated squaring, as
    //!        `numpy.linalg.matrix_power`: the identity for 0, the inverse's powers for `e < 0`.
    template <typename T>
    [[nodiscard]] matrix<T> integer_power(const matrix<T>& a,
                                          double           e)
    {
      const std::size_t        n      {a.rows()};
      matrix<T>                square {e < 0.0 ? inverse(a) : a};
      std::optional<matrix<T>> result;
      for (double k {std::abs(e)}; k > 0.0;) {
        if (std::fmod(k, 2.0) == 1.0) {
          result = result ? *result * square : square;
        }
        k = std::floor(k / 2.0);
        if (k > 0.0) {
          square = square * square;
        }
      }
      if (result) {
        return *result;
      }
      buffer<T> unit(n * n, T {});
      for (std::size_t i {0}; i < n; ++i) {
        unit[(i * n) + i] = T {1.0};
      }
      return matrix<T> {n, n, std::move(unit)};
    }
  } // namespace detail

  /*!
   * \brief `A^p` for a real \p p, as `scipy.linalg.fractional_matrix_power`: a whole \p p by repeated squaring;
   *        else `A^a A^b` with `a` the floor or the ceiling of \p p, chosen by `A`'s condition as SciPy chooses it, and
   *        `A^b` (`|b| < 1`) by Higham and Lin's Schur–Padé algorithm (2013) on the Schur form. Where the triangular form
   *        has a zero on its diagonal (`A` singular), a positive \p p falls back to \ref funm of `x^(p − ⌊p⌋)`, as
   *        SciPy's; a negative one has no answer and is refused (SciPy returns a matrix of NaN). Complex: a real
   *        matrix's power may be complex.
   * \tparam T `double` or `std::complex<double>`.
   * \throws dimension_error       If \p a is not square.
   * \throws non_finite_error      If an entry is infinite or NaN.
   * \throws std::invalid_argument If \p p is infinite or NaN.
   * \throws singular_matrix_error If \p p is negative and \p a singular.
   * \throws convergence_error     If a Schur form's iterations did not converge, or the square roots of its
   *                               triangular form are not finite.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] matrix<std::complex<double>> fractional_matrix_power(const matrix<T>& a,
                                                                     double           p)
  {
    using C = std::complex<double>;
    const std::size_t n {a.rows()};
    if (a.cols() != n) {
      throw dimension_error("fractional_matrix_power requires a square matrix");
    }
    detail::require_finite(a, "fractional_matrix_power");
    if (!std::isfinite(p)) {
      throw std::invalid_argument("fractional_matrix_power: the power must be finite");
    }
    if (n == 0) {
      return matrix<C> {0, 0, buffer<C> {}};
    }
    if (p == std::floor(p)) {
      return detail::as_complex(detail::integer_power(a, p));
    }
    const vector<double> sigma {singular_values(a)};
    if (sigma[n - 1] != 0.0) {
      const double k2 {sigma[0] / sigma[n - 1]}; // the condition chooses between ⌊p⌋ and ⌈p⌉, as SciPy's
      const double p1 {p - std::floor(p)};
      const double p2 {p - std::ceil(p)};
      const bool   below {p1 * std::pow(k2, 1.0 - p1) <= -p2 * k2};
      const double whole {below ? std::floor(p) : std::ceil(p)};
      if (const auto remainder {detail::remainder_power(a, below ? p1 : p2)}) {
        return detail::as_complex(detail::integer_power(a, whole)) * *remainder;
      }
    }
    if (p < 0.0) {
      throw singular_matrix_error("fractional_matrix_power: a negative power of a singular matrix");
    }
    const double fraction {p - std::floor(p)};
    const auto   power = [fraction](const std::vector<C>& values) {
                           std::vector<C> powers(values.size());
                           std::transform(values.begin(), values.end(), powers.begin(), [fraction](const C& z) { return detail::principal_power(z, fraction); });
                           return powers;
                         };
    return detail::as_complex(detail::integer_power(a, std::floor(p))) * funm(a, power).f;
  }

  namespace detail {

    //! \brief `log A` from the upper triangular `n × n` \p t of `A = Z T Zᴴ` (\p z, or none where `T` is `A`), its
    //!        zero diagonal entries taken as `10⁻²⁰` (\ref force_nonsingular).
    template <typename T>
    [[nodiscard]] logm_result schur_logm(buffer<T>        t,
                                         std::size_t      n,
                                         const matrix<T>* z)
    {
      const auto [exact, near] {force_nonsingular(t, n)};
      const matrix<T> u {n, n, triangular_log(t, n)};
      return {as_complex(z == nullptr ? u : *z * u * adjoint_of(*z)), exact, near};
    }

    //! \brief Whether no diagonal entry of \p t is below 0.
    [[nodiscard]] inline bool nonnegative_diagonal(const matrix<double>& t)
    {
      for (std::size_t i {0}; i < t.rows(); ++i) {
        if (t(i, i) < 0.0) {
          return false;
        }
      }
      return true;
    }

    //! \brief \ref logm of a complex matrix past its Hermitian path: an upper triangular one directly, else its Schur
    //!        form's.
    [[nodiscard]] inline logm_result general_logm(const matrix<std::complex<double>>& a)
    {
      using C = std::complex<double>;
      if (upper_triangular(a)) {
        return schur_logm<C>(buffer<C>(a.begin(), a.end()), a.rows(), nullptr);
      }
      const schur_decomposition<C> s {a};
      return schur_logm(buffer<C>(s.t().begin(), s.t().end()), a.rows(), &s.z());
    }

    //! \brief \ref logm of a real matrix past its Hermitian path: in real arithmetic where its real Schur form (or the
    //!        matrix, upper triangular) is triangular with no diagonal entry below 0, as SciPy's (`_logm`,
    //!        `_logm_triu`), else in the complex form (`rsf2csf`).
    [[nodiscard]] inline logm_result general_logm(const matrix<double>& a)
    {
      using C = std::complex<double>;
      if (upper_triangular(a)) {
        if (nonnegative_diagonal(a)) {
          return schur_logm<double>(buffer<double>(a.begin(), a.end()), a.rows(), nullptr);
        }
        return schur_logm<C>(buffer<C>(a.begin(), a.end()), a.rows(), nullptr);
      }
      const schur_decomposition<double> s {a};
      if (upper_triangular(s.t()) && nonnegative_diagonal(s.t())) {
        return schur_logm(buffer<double>(s.t().begin(), s.t().end()), a.rows(), &s.z());
      }
      const auto [form, z] {complex_schur_form(s.t(), s.z())};
      return schur_logm(buffer<C>(form.begin(), form.end()), a.rows(), &z);
    }
  } // namespace detail

  /*!
   * \brief The principal logarithm of a square matrix, as `scipy.linalg.logm`: an upper triangular `A` directly, else
   *        its Schur form `A = Z T Zᴴ` and `Z log(T) Zᴴ`: in real arithmetic where `A` is real and its real Schur form
   *        triangular with no eigenvalue below 0, else in the complex form (the real one made complex), as SciPy's.
   *        `log T` by Al-Mohy and Higham's inverse scaling and squaring (2012) as SciPy's `_logm_triu`, the norms that
   *        choose the square roots
   *        and the degree exact where SciPy estimates them. A zero eigenvalue is taken as `10⁻²⁰` (no logarithm exists
   *        there), as SciPy's, and reported; complex, SciPy's real part where `A` is real and it is negligible.
   * \tparam T `double` or `std::complex<double>`.
   * \throws dimension_error   If \p a is not square.
   * \throws non_finite_error  If an entry is infinite or NaN.
   * \throws convergence_error If the Schur form's iterations did not converge, or the square roots of its triangular form
   *                           are not finite.
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
    if (detail::exactly_hermitian(a)) { // V log(Λ) Vᴴ, unless an eigenvalue is below 10⁻²⁰ (SciPy's singular cases)
      const auto [values, vectors] {detail::hermitian_system(a)};
      std::vector<C> logs(n);
      bool           regular {true};
      for (std::size_t k {0}; k < n; ++k) {
        regular = regular && std::abs(values[k]) >= 1e-20;
        logs[k] = detail::complex_log(C {values[k]});
      }
      if (regular) {
        return {detail::spectral<C>(vectors, logs, values[0] > 0.0), false, false};
      }
    }
    return detail::general_logm(a);
  }
} // namespace numerical

#endif // NUMERICAL_MATRIX_FUNCTIONS_HPP
