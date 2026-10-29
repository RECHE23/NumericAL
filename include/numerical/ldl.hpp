/*!
 * \file ldl.hpp
 * \brief The symmetric indefinite factorisation `A = L D Lᵀ` (or `L D Lᴴ`) of a symmetric or Hermitian matrix with
 *        Bunch and Kaufman's diagonal pivoting, `D` block diagonal with blocks of order 1 and 2, as LAPACK's `sytf2` and
 *        `hetf2` and as `scipy.linalg.ldl` returns it.
 */
#ifndef NUMERICAL_LDL_HPP
#define NUMERICAL_LDL_HPP

#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "factorizations.hpp"

namespace numerical {

  namespace detail {

    //! \brief LAPACK's measure of a real entry, `|x|`.
    [[nodiscard]] inline double pivot_size(double x)
    {
      return std::abs(x);
    }

    //! \brief LAPACK's measure of a complex entry, `CABS1`: `|Re z| + |Im z|`.
    [[nodiscard]] inline double pivot_size(const std::complex<double>& z)
    {
      return std::abs(z.real()) + std::abs(z.imag());
    }

    //! \brief A real diagonal entry's measure, `|x|`, whatever \p hermitian.
    [[nodiscard]] inline double diagonal_size(double x,
                                              bool /*hermitian*/)
    {
      return std::abs(x);
    }

    //! \brief A complex diagonal entry's measure: `|Re z|` for a Hermitian matrix (`hetf2`), `CABS1` for a symmetric
    //!        one (`sytf2`).
    [[nodiscard]] inline double diagonal_size(const std::complex<double>& z,
                                              bool                        hermitian)
    {
      return hermitian ? std::abs(z.real()) : pivot_size(z);
    }

    //! \brief A real \p x as it is.
    [[nodiscard]] inline double conjugated_if(double x,
                                              bool /*hermitian*/)
    {
      return x;
    }

    //! \brief \p z's conjugate for a Hermitian matrix, \p z for a symmetric one.
    [[nodiscard]] inline std::complex<double> conjugated_if(const std::complex<double>& z,
                                                            bool                        hermitian)
    {
      return hermitian ? std::conj(z) : z;
    }

    //! \brief A real \p x as it is.
    [[nodiscard]] inline double diagonal_kept(double x,
                                              bool /*hermitian*/)
    {
      return x;
    }

    //! \brief A Hermitian matrix's diagonal entry \p z taken as real (`hetf2` ignores its imaginary part), a symmetric
    //!        one's as it is.
    [[nodiscard]] inline std::complex<double> diagonal_kept(const std::complex<double>& z,
                                                            bool                        hermitian)
    {
      return hermitian ? std::complex<double> {z.real()} : z;
    }

    /*!
     * \brief The trailing matrix's update by a pivot of order 1 at \p k, as `sytf2` (`syr`, then `L`'s column scaled):
     *        `A ← A − x xᵀ / d` on and below the diagonal, `x / d` into column \p k; the arithmetic of \p T. Row by row
     *        (`a` is row-major), each entry's one operation as `syr`'s column by column: the factors `−x_j / d` first, in
     *        \p w (`n` entries).
     */
    template <typename T>
    void symmetric_rank_one(buffer<T>&  a,
                            std::size_t n,
                            std::size_t k,
                            buffer<T>&  w)
    {
      const T r {T {1.0} / a[(k * n) + k]};
      for (std::size_t j {k + 1}; j < n; ++j) {
        w[j] = -r * a[(j * n) + k];
      }
      for (std::size_t i {k + 1}; i < n; ++i) {
        const T  x   {a[(i * n) + k]};
        T* const row {a.data() + (i * n)};
        for (std::size_t j {k + 1}; j <= i; ++j) {
          row[j] += x * w[j];
        }
        row[k] = x * r;
      }
    }

    //! \brief As \ref symmetric_rank_one for a Hermitian matrix, as `hetf2` (`her`): `d` real, `A ← A − x xᴴ / d`, the
    //!        diagonal kept real.
    inline void hermitian_rank_one(buffer<std::complex<double>>& a,
                                   std::size_t                   n,
                                   std::size_t                   k,
                                   buffer<std::complex<double>>& w)
    {
      using C = std::complex<double>;
      const double r {1.0 / a[(k * n) + k].real()};
      for (std::size_t j {k + 1}; j < n; ++j) {
        w[j] = -r * std::conj(a[(j * n) + k]);
      }
      for (std::size_t i {k + 1}; i < n; ++i) {
        const C  x   {a[(i * n) + k]};
        C* const row {a.data() + (i * n)};
        for (std::size_t j {k + 1}; j < i; ++j) {
          row[j] += x * w[j];
        }
        row[i] = C {row[i].real() + (x * w[i]).real()};
        row[k] = x * r;
      }
    }

    /*!
     * \brief The trailing matrix's update by a pivot of order 2 at rows \p k and \p k + 1, as `sytf2`: `D21` scaled out,
     *        `W = [x y] D⁻¹` from `D11 / D21`, `D22 / D21` and `t = 1 / (D11 D22 / D21² − 1)`, `A ← A − [x y] Wᵀ`, `W`
     *        into the two columns; the arithmetic of \p T. Row by row, `W` first, in \p w (`2n` entries), each row's
     *        `x` and `y` read before its two entries of `W` replace them.
     */
    template <typename T>
    void symmetric_rank_two(buffer<T>&  a,
                            std::size_t n,
                            std::size_t k,
                            buffer<T>&  w)
    {
      const T d21 {a[((k + 1) * n) + k]};
      const T d11 {a[((k + 1) * n) + k + 1] / d21};
      const T d22 {a[(k * n) + k] / d21};
      const T t   {T {1.0} / ((d11 * d22) - T {1.0})};
      for (std::size_t j {k + 2}; j < n; ++j) {
        w[2 * j]       = t * (((d11 * a[(j * n) + k]) - a[(j * n) + k + 1]) / d21);
        w[(2 * j) + 1] = t * (((d22 * a[(j * n) + k + 1]) - a[(j * n) + k]) / d21);
      }
      for (std::size_t i {k + 2}; i < n; ++i) {
        T* const row {a.data() + (i * n)};
        const T  x   {row[k]};
        const T  y   {row[k + 1]};
        for (std::size_t j {k + 2}; j <= i; ++j) {
          row[j] = row[j] - (x * w[2 * j]) - (y * w[(2 * j) + 1]);
        }
        row[k]     = w[2 * i];
        row[k + 1] = w[(2 * i) + 1];
      }
    }

    //! \brief As \ref symmetric_rank_two for a Hermitian matrix, as `hetf2`: `|D21|` scaled out in real arithmetic, the
    //!         conjugates where `Aᴴ` has them, the diagonal kept real.
    inline void hermitian_rank_two(buffer<std::complex<double>>& a,
                                   std::size_t                   n,
                                   std::size_t                   k,
                                   buffer<std::complex<double>>& w)
    {
      using C = std::complex<double>;
      const C      below {a[((k + 1) * n) + k]};
      const double d     {std::hypot(below.real(), below.imag())};
      const double d11   {a[((k + 1) * n) + k + 1].real() / d};
      const double d22   {a[(k * n) + k].real() / d};
      const double t     {1.0 / ((d11 * d22) - 1.0)};
      const C      d21   {below / d};
      for (std::size_t j {k + 2}; j < n; ++j) {
        w[2 * j]       = t * (((d11 * a[(j * n) + k]) - (d21 * a[(j * n) + k + 1])) / d);
        w[(2 * j) + 1] = t * (((d22 * a[(j * n) + k + 1]) - (std::conj(d21) * a[(j * n) + k])) / d);
      }
      for (std::size_t i {k + 2}; i < n; ++i) {
        C* const row {a.data() + (i * n)};
        const C  x   {row[k]};
        const C  y   {row[k + 1]};
        for (std::size_t j {k + 2}; j <= i; ++j) {
          row[j] = row[j] - (x * std::conj(w[2 * j])) - (y * std::conj(w[(2 * j) + 1]));
        }
        row[k]     = w[2 * i];
        row[k + 1] = w[(2 * i) + 1];
        row[i]     = C {row[i].real()};
      }
    }

    //! \brief A real matrix's update by a pivot of order 1 or 2 (\p step), `sytf2`'s whatever \p hermitian.
    inline void eliminate_pivot(buffer<double>& a,
                                std::size_t     n,
                                std::size_t     k,
                                std::size_t     step,
                                bool /*hermitian*/,
                                buffer<double>& w)
    {
      if (step == 1) {
        symmetric_rank_one(a, n, k, w);
      }
      else {
        symmetric_rank_two(a, n, k, w);
      }
    }

    //! \brief A complex matrix's update by a pivot of order 1 or 2 (\p step): `hetf2`'s if \p hermitian, else `sytf2`'s.
    inline void eliminate_pivot(buffer<std::complex<double>>& a,
                                std::size_t                   n,
                                std::size_t                   k,
                                std::size_t                   step,
                                bool                          hermitian,
                                buffer<std::complex<double>>& w)
    {
      if (step == 1) {
        if (hermitian) {
          hermitian_rank_one(a, n, k, w);
        }
        else {
          symmetric_rank_one(a, n, k, w);
        }
      }
      else if (hermitian) {
        hermitian_rank_two(a, n, k, w);
      }
      else {
        symmetric_rank_two(a, n, k, w);
      }
    }

    /*!
     * \brief Bunch and Kaufman's factorisation of the lower triangle of \p a (`n × n`, row-major; entries `(i, j)` with
     *        `i ≥ j` read and written), as LAPACK's `sytf2` and `hetf2` on the lower triangle: at each step a pivot of
     *        order 1 or 2 chosen with `α = (1 + √17) / 8`, the rows and columns exchanged, the trailing matrix updated
     *        by the pivot's rank-1 or rank-2 correction, `L`'s columns left below the pivot and `D` on and next to the
     *        diagonal. The largest entry of a column is its first (`idamax`) or, where \p last_on_ties, its last of
     *        equal size: the order of the upper triangle's search when \p a holds it reversed.
     * \return LAPACK's `ipiv` (from 1): `k` the row exchanged with row `k` for a block of order 1, `−k` on both rows of
     *         a block of order 2.
     */
    template <typename T>
    [[nodiscard]] std::vector<std::ptrdiff_t> bunch_kaufman_lower(buffer<T>&  a,
                                                                  std::size_t n,
                                                                  bool        hermitian,
                                                                  bool        last_on_ties)
    {
      const double                alpha {(1.0 + std::sqrt(17.0)) / 8.0};
      std::vector<std::ptrdiff_t> pivots(n);
      buffer<T>                   w(2 * n); // the rank updates' factors
      const auto                  at = [&a, n](std::size_t i, std::size_t j) -> T& { return a[(i * n) + j]; };
      for (std::size_t k {0}; k < n;) {
        std::size_t  step   {1};
        std::size_t  pivot  {k};
        const double absakk {diagonal_size(at(k, k), hermitian)};
        std::size_t  imax   {k};
        double       colmax {0.0};
        for (std::size_t i {k + 1}; i < n; ++i) {
          const double size {pivot_size(at(i, k))};
          if (size > colmax || (last_on_ties && size == colmax && i > k + 1)) {
            colmax = size;
            imax   = i;
          }
        }
        if (std::max(absakk, colmax) == 0.0 || std::isnan(absakk)) { // a zero (or NaN) column: no elimination
          at(k, k) = diagonal_kept(at(k, k), hermitian);
        }
        else {
          if (absakk < alpha * colmax) {
            double rowmax {0.0}; // the largest entry of row and column imax off the diagonal
            for (std::size_t j {k}; j < imax; ++j) {
              rowmax = std::max(rowmax, pivot_size(at(imax, j)));
            }
            for (std::size_t i {imax + 1}; i < n; ++i) {
              rowmax = std::max(rowmax, pivot_size(at(i, imax)));
            }
            if (absakk < alpha * colmax * (colmax / rowmax)) {
              pivot = imax;
              step  = diagonal_size(at(imax, imax), hermitian) >= alpha * rowmax ? 1 : 2;
            }
          }
          const std::size_t kk {k + step - 1};
          if (pivot != kk) { // rows and columns kk and pivot exchanged in the lower triangle
            for (std::size_t i {pivot + 1}; i < n; ++i) {
              std::swap(at(i, kk), at(i, pivot));
            }
            for (std::size_t j {kk + 1}; j < pivot; ++j) {
              const T t {conjugated_if(at(j, kk), hermitian)};
              at(j, kk)    = conjugated_if(at(pivot, j), hermitian);
              at(pivot, j) = t;
            }
            at(pivot, kk) = conjugated_if(at(pivot, kk), hermitian);
            const T t {diagonal_kept(at(kk, kk), hermitian)};
            at(kk, kk)       = diagonal_kept(at(pivot, pivot), hermitian);
            at(pivot, pivot) = t;
            if (step == 2) {
              at(k, k) = diagonal_kept(at(k, k), hermitian);
              std::swap(at(k + 1, k), at(pivot, k));
            }
          }
          else {
            at(k, k)   = diagonal_kept(at(k, k), hermitian);
            at(kk, kk) = diagonal_kept(at(kk, kk), hermitian);
          }
          eliminate_pivot(a, n, k, step, hermitian, w);
        }
        const auto origin {static_cast<std::ptrdiff_t>(pivot + 1)};
        pivots[k] = step == 1 ? origin : -origin;
        if (step == 2) {
          pivots[k + 1] = -origin;
        }
        k += step;
      }
      return pivots;
    }
  } // namespace detail

  /*!
   * \brief The symmetric indefinite factorisation of a symmetric or Hermitian matrix, as `scipy.linalg.ldl`:
   *        `A = L D Lᴴ` (Hermitian) or `A = L D Lᵀ` (symmetric), Bunch and Kaufman's diagonal pivoting (LAPACK's
   *        `sytf2`, `hetf2`) on the lower triangle, or on the upper one (`lower = false`, `L` then upper triangular); `D`
   *        block diagonal with blocks of order 1 and 2; `L` with a unit diagonal once its rows are permuted
   *        (\ref permutation), as SciPy returns them.
   * \tparam T `double` or `std::complex<double>`.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  class ldl_decomposition
  {
  public:

    /*!
     * \brief Factorises \p a, of which only the triangle \p lower names is read.
     * \param[in] a         The square matrix.
     * \param[in] lower     The lower triangle (and a lower `L`), or the upper.
     * \param[in] hermitian For a complex matrix, `A = Aᴴ` (the diagonal's imaginary parts ignored) or `A = Aᵀ`; a real
     *                      one's either.
     * \throws dimension_error  If \p a is not square.
     * \throws non_finite_error If an entry is infinite or NaN.
     */
    explicit ldl_decomposition(const matrix<T>& a,
                               bool             lower     = true,
                               bool             hermitian = true)
      : n_(a.rows())
    {
      if (a.cols() != n_) {
        throw dimension_error("ldl requires a square matrix");
      }
      detail::require_finite(a, "ldl");
      const std::size_t n {n_};
      buffer<T>         w(n * n);
      for (std::size_t i {0}; i < n; ++i) { // the upper triangle reversed, J A J, is a lower one searched backwards
        for (std::size_t j {0}; j < n; ++j) {
          w[(i * n) + j] = lower ? a(i, j) : a(n - 1 - i, n - 1 - j);
        }
      }
      std::vector<std::ptrdiff_t> found {detail::bunch_kaufman_lower(w, n, hermitian, !lower)};
      buffer<T>                   ldu(n * n);
      std::vector<std::ptrdiff_t> ipiv(n);
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          ldu[(i * n) + j] = lower ? w[(i * n) + j] : w[((n - 1 - i) * n) + (n - 1 - j)];
        }
        const std::ptrdiff_t p   {lower ? found[i] : found[n - 1 - i]};
        const std::ptrdiff_t row {lower ? std::abs(p) : static_cast<std::ptrdiff_t>(n) + 1 - std::abs(p)};
        ipiv[i] = p > 0 ? row : -row;
      }
      shape(ldu, ipiv, lower, hermitian);
    }

    //! \brief `L`, permuted (SciPy's `lu`): `L[permutation()]` is triangular with a unit diagonal.
    [[nodiscard]] const matrix<T>& factor() const noexcept
    {
      return l_;
    }

    //! \brief `D`, block diagonal (blocks of order 1 and 2).
    [[nodiscard]] const matrix<T>& d() const noexcept
    {
      return d_;
    }

    //! \brief The rows' order that makes \ref factor triangular (SciPy's `perm`).
    [[nodiscard]] const std::vector<std::size_t>& permutation() const noexcept
    {
      return permutation_;
    }

  private:

    /*!
     * \brief `L`, `D` and the permutation from LAPACK's packed factors \p ldu and \p ipiv, as SciPy's
     *        `_ldl_sanitize_ipiv`, `_ldl_get_d_and_l` and `_ldl_construct_tri_factor` form them.
     */
    void shape(const buffer<T>&                   ldu,
               const std::vector<std::ptrdiff_t>& ipiv,
               bool                               lower,
               bool                               hermitian)
    {
      const std::size_t        n {n_};
      std::vector<std::size_t> swaps(n);
      std::vector<int>         blocks(n, 0);
      std::iota(swaps.begin(), swaps.end(), std::size_t {0});
      // the swaps and the blocks' orders, read from the first row of each block (the last for the upper triangle)
      for (std::size_t visited {0}; visited < n;) {
        const std::size_t    index {lower ? visited : n - 1 - visited};
        const std::ptrdiff_t value {ipiv[index]};
        if (value > 0) {
          swaps[index]  = swaps[static_cast<std::size_t>(value - 1)];
          blocks[index] = 1;
          ++visited;
          continue;
        }
        // a block of order 2 never starts on the last row it is read from, so the partner row exists; at() lets a
        // compiler that inlines a matrix of order 1 see it as well
        const std::size_t partner {lower ? index + 1 : index - 1};
        swaps.at(partner)                  = swaps[static_cast<std::size_t>(-value - 1)];
        blocks[lower ? index : partner]    = 2;
        visited                           += 2;
      }
      buffer<T> d(n * n, T {});
      buffer<T> l(n * n, T {});
      for (std::size_t i {0}; i < n; ++i) {
        d[(i * n) + i] = ldu[(i * n) + i];
        l[(i * n) + i] = T {1.0};
        for (std::size_t j {0}; j < n; ++j) {
          if (lower ? j < i : j > i) {
            l[(i * n) + j] = ldu[(i * n) + j];
          }
        }
      }
      for (std::size_t start {0}; start < n;) { // a block of order 2: its off-diagonal entry into D, out of L
        const int order {blocks[start]};        // a block's order is marked on its first row, the second row's 0
        if (order == 2) {
          const std::size_t below {start + 1};
          const T           entry {lower ? ldu[(below * n) + start] : ldu[(start * n) + below]};
          d[(below * n) + start]                               = lower ? entry : detail::conjugated_if(entry, hermitian);
          d[(start * n) + below]                               = lower ? detail::conjugated_if(entry, hermitian) : entry;
          l[lower ? (below * n) + start : (start * n) + below] = T {};
        }
        start += static_cast<std::size_t>(order);
      }
      std::vector<std::size_t> order(n);
      std::iota(order.begin(), order.end(), std::size_t {0});
      for (std::size_t visited {0}; visited < n; ++visited) { // the swaps applied to L's rows, as SciPy reads them
        const std::size_t index  {lower ? n - 1 - visited : visited};
        const std::size_t target {swaps[index]};
        if (target == index) {
          continue;
        }
        std::size_t first {lower ? index : 0};
        std::size_t last  {lower ? n : index + 1};
        if (blocks[index] == (lower ? 0 : 2)) { // the second row of a block of order 2: both its columns
          first -= lower ? 1 : 0;
          last  += lower ? 0 : 1;
        }
        for (std::size_t j {first}; j < last; ++j) {
          std::swap(l[(target * n) + j], l[(index * n) + j]);
        }
        std::swap(order[target], order[index]);
      }
      permutation_.assign(n, 0);
      for (std::size_t i {0}; i < n; ++i) { // argsort of a permutation: its inverse
        permutation_[order[i]] = i;
      }
      l_ = matrix<T> {n, n, std::move(l)};
      d_ = matrix<T> {n, n, std::move(d)};
    }

    std::size_t              n_;           //!< The order.
    matrix<T>                l_;           //!< L, permuted.
    matrix<T>                d_;           //!< D.
    std::vector<std::size_t> permutation_; //!< The rows' order making L triangular.
  };
} // namespace numerical

#endif // NUMERICAL_LDL_HPP
