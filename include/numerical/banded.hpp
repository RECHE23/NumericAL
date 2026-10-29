/*!
 * \file banded.hpp
 * \brief Systems of banded matrices, in LAPACK's diagonal-ordered storage: the general band by its LU factorisation
 *        with partial pivoting (`gbsv`), the Hermitian positive-definite band by its Cholesky factor (`pbsv`).
 */
#ifndef NUMERICAL_BANDED_HPP
#define NUMERICAL_BANDED_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "factorizations.hpp"

namespace numerical {

  namespace detail {

    //! \brief A band's entries in LAPACK's column-major band storage: `AB(i, j)` (from 1) at `(i − 1) + (j − 1) ldab`.
    template <typename T>
    struct band_storage
    {
      buffer<T>   entries; //!< The columns, \ref ldab apart.
      std::size_t ldab;    //!< The rows of the storage.

      //! \brief `AB(i, j)`, from 1.
      [[nodiscard]] T& operator()(std::size_t i,
                                  std::size_t j)
      {
        return entries[(i - 1) + ((j - 1) * ldab)];
      }
    };

    /*!
     * \brief The LU factorisation with partial pivoting of the `n × n` band of \p kl subdiagonals and \p ku
     *        superdiagonals in \p ab (`2 kl + ku + 1` rows, the band from row `kl + 1`, the rows above for the fill
     *        the interchanges bring), in place, as LAPACK's `gbtf2`: each column's pivot the entry of largest modulus
     *        (the first of them) within the band below the diagonal, its row swapped to the reach of the rows it
     *        touches, the multipliers below and the rank-1 update right of it. \p pivots (from 1) the rows swapped.
     * \throws singular_matrix_error If a pivot is zero (`gbtf2`'s `info`; LAPACK goes on, SciPy then raises).
     */
    template <typename T>
    void band_lu(band_storage<T>&          ab,
                 std::size_t               n,
                 std::size_t               kl,
                 std::size_t               ku,
                 std::vector<std::size_t>& pivots)
    {
      const std::size_t kv {ku + kl};
      pivots.assign(n, 0);
      std::size_t ju       {1};
      for (std::size_t j {1}; j <= n; ++j) {
        if (j + kv <= n) {
          for (std::size_t i {1}; i <= kl; ++i) {
            ab(i, j + kv) = T {};
          }
        }
        const std::size_t km      {std::min(kl, n - j)};
        std::size_t       jp      {1};
        double            largest {std::abs(ab(kv + 1, j))};
        for (std::size_t r {2}; r <= km + 1; ++r) {
          if (std::abs(ab(kv + r, j)) > largest) {
            largest = std::abs(ab(kv + r, j));
            jp      = r;
          }
        }
        pivots[j - 1] = jp + j - 1;
        if (ab(kv + jp, j) == T {}) {
          throw singular_matrix_error("singular matrix");
        }
        ju = std::max(ju, std::min(j + ku + jp - 1, n));
        if (jp != 1) { // rows j and j + jp − 1 over columns j … ju, along the storage's rows
          for (std::size_t c {0}; c <= ju - j; ++c) {
            std::swap(ab(kv + jp - c, j + c), ab(kv + 1 - c, j + c));
          }
        }
        const T pivot {ab(kv + 1, j)};
        for (std::size_t r {kv + 2}; r <= kv + 1 + km; ++r) {
          ab(r, j) /= pivot;
        }
        for (std::size_t c {1}; c <= ju - j; ++c) { // the rows below take the multiples of row j, column by column
          const T factor {ab(kv + 1 - c, j + c)};
          for (std::size_t r {1}; r <= km; ++r) {
            ab(kv + 1 - c + r, j + c) -= ab(kv + 1 + r, j) * factor;
          }
        }
      }
    }

    /*!
     * \brief The \p k right-hand sides of \p b (`n × k`, row-major) solved in place against \ref band_lu's factors,
     *        as LAPACK's `gbtrs` (no transpose): the interchanges and `L` column by column, then `U`, a band of
     *        `kl + ku` superdiagonals, by back substitution.
     */
    template <typename T>
    void band_lu_solve(band_storage<T>&                ab,
                       std::size_t                     n,
                       std::size_t                     kl,
                       std::size_t                     ku,
                       const std::vector<std::size_t>& pivots,
                       buffer<T>&                      b,
                       std::size_t                     k)
    {
      const std::size_t kd {ku + kl + 1};
      for (std::size_t j {1}; j < n; ++j) {
        const std::size_t lm {std::min(kl, n - j)};
        const std::size_t l  {pivots[j - 1]};
        if (l != j) {
          std::swap_ranges(b.begin() + static_cast<std::ptrdiff_t>((l - 1) * k), b.begin() + static_cast<std::ptrdiff_t>(l * k),
                           b.begin() + static_cast<std::ptrdiff_t>((j - 1) * k));
        }
        for (std::size_t r {1}; r <= lm; ++r) {
          subtract_product(b.data() + ((j - 1 + r) * k), ab(kd + r, j), b.data() + ((j - 1) * k), k);
        }
      }
      for (std::size_t j {n}; j >= 1; --j) {
        const T diagonal {ab(kd, j)};
        for (std::size_t c {0}; c < k; ++c) {
          b[((j - 1) * k) + c] /= diagonal;
        }
        const std::size_t first {j > kl + ku ? j - kl - ku : 1};
        for (std::size_t i {first}; i < j; ++i) {
          subtract_product(b.data() + ((i - 1) * k), ab(kd + i - j, j), b.data() + ((j - 1) * k), k);
        }
      }
    }

    /*!
     * \brief The Cholesky factor `U` (`A = Uᴴ U`) of the Hermitian positive-definite band of \p kd superdiagonals in
     *        \p ab (`kd + 1` rows, `AB(kd + 1 + i − j, j) = A(i, j)`), in place, as LAPACK's `pbtf2` (upper): each
     *        diagonal entry's square root, the row right of it scaled, the trailing band less its outer product.
     * \throws not_positive_definite_error If a diagonal entry is not positive when reached.
     */
    template <typename T>
    void band_cholesky(band_storage<T>& ab,
                       std::size_t      n,
                       std::size_t      kd)
    {
      for (std::size_t j {1}; j <= n; ++j) {
        const double ajj {std::real(ab(kd + 1, j))};
        if (!(ajj > 0.0)) {
          throw not_positive_definite_error("solveh_banded: the matrix is not positive definite (leading minor " + std::to_string(j) + ")");
        }
        const double root    {std::sqrt(ajj)};
        ab(kd + 1, j) = T {root};
        const std::size_t kn {std::min(kd, n - j)};
        for (std::size_t c {1}; c <= kn; ++c) { // row j of U right of the diagonal: AB(kd + 1 − c, j + c)
          ab(kd + 1 - c, j + c) /= T {root};
        }
        for (std::size_t c {1}; c <= kn; ++c) { // A(j + r, j + c) −= conj(U(j, j + r)) U(j, j + c), r ≤ c, the upper band
          const T right {ab(kd + 1 - c, j + c)};
          for (std::size_t r {1}; r <= c; ++r) {
            ab(kd + 1 + r - c, j + c) -= conj_of(ab(kd + 1 - r, j + r)) * right;
          }
        }
      }
    }

    /*!
     * \brief The \p k right-hand sides of \p b (`n × k`, row-major) solved in place against \ref band_cholesky's `U`,
     *        as LAPACK's `pbtrs`: `Uᴴ y = b` forward, then `U x = y` back.
     */
    template <typename T>
    void band_cholesky_solve(band_storage<T>& ab,
                             std::size_t      n,
                             std::size_t      kd,
                             buffer<T>&       b,
                             std::size_t      k)
    {
      for (std::size_t j {1}; j <= n; ++j) { // Uᴴ: column j of Uᴴ is row j of U, conjugated
        const std::size_t first {j > kd ? j - kd : 1};
        for (std::size_t i {first}; i < j; ++i) {
          subtract_product(b.data() + ((j - 1) * k), conj_of(ab(kd + 1 + i - j, j)), b.data() + ((i - 1) * k), k);
        }
        const T diagonal {ab(kd + 1, j)};
        for (std::size_t c {0}; c < k; ++c) {
          b[((j - 1) * k) + c] /= diagonal;
        }
      }
      for (std::size_t j {n}; j >= 1; --j) {
        const T diagonal {ab(kd + 1, j)};
        for (std::size_t c {0}; c < k; ++c) {
          b[((j - 1) * k) + c] /= diagonal;
        }
        const std::size_t first {j > kd ? j - kd : 1};
        for (std::size_t i {first}; i < j; ++i) {
          subtract_product(b.data() + ((i - 1) * k), ab(kd + 1 + i - j, j), b.data() + ((j - 1) * k), k);
        }
      }
    }

    //! \brief Refuses a band that is not \p rows × `n` and right-hand sides not of `n` rows.
    template <typename T>
    void require_band(const matrix<T>&   ab,
                      std::size_t        rows,
                      const matrix<T>&   b,
                      const std::string& name)
    {
      if (ab.rows() != rows) {
        throw dimension_error(name + ": the band must have " + std::to_string(rows) + " rows");
      }
      if (b.rows() != ab.cols()) {
        throw dimension_error(name + ": the right-hand sides must have as many rows as the band has columns");
      }
      require_finite(ab, name.c_str());
      require_finite(b, name.c_str());
    }
  } // namespace detail

  /*!
   * \brief The solution `X` of `A X = B` for the `n × n` band `A` of \p lower subdiagonals and \p upper superdiagonals,
   *        given as `scipy.linalg.solve_banded` takes it: \p ab of `lower + upper + 1` rows and `n` columns,
   *        `ab[upper + i − j][j] = A(i, j)` (its corners, outside the matrix, not read). By the LU factorisation with
   *        partial pivoting in LAPACK's band storage (`gbsv`): `O(n · lower · (lower + upper))` operations.
   * \tparam T `double` or `std::complex<double>`.
   * \throws dimension_error       If \p ab has not `lower + upper + 1` rows or \p b not `n` rows.
   * \throws non_finite_error      If an entry is infinite or NaN.
   * \throws singular_matrix_error If a pivot is zero.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] matrix<T> solve_banded(std::size_t      lower,
                                       std::size_t      upper,
                                       const matrix<T>& ab,
                                       const matrix<T>& b)
  {
    detail::require_band(ab, lower + upper + 1, b, "solve_banded");
    const std::size_t        n    {ab.cols()};
    const std::size_t        ldab {(2 * lower) + upper + 1};
    detail::band_storage<T>  work {buffer<T>(ldab * n, T {}), ldab};
    for (std::size_t r {0}; r < lower + upper + 1; ++r) {
      for (std::size_t j {0}; j < n; ++j) {
        work(lower + r + 1, j + 1) = ab(r, j);
      }
    }
    std::vector<std::size_t> pivots;
    detail::band_lu(work, n, lower, upper, pivots);
    buffer<T> x(b.begin(), b.end());
    detail::band_lu_solve(work, n, lower, upper, pivots, x, b.cols());
    return matrix<T> {b.rows(), b.cols(), std::move(x)};
  }

  /*!
   * \brief The solution `X` of `A X = B` for the Hermitian positive-definite band `A`, given as
   *        `scipy.linalg.solveh_banded` takes it: \p ab of `kd + 1` rows and `n` columns, its upper form
   *        `ab[kd + i − j][j] = A(i, j)` for `i ≤ j` or, \p lower_form, its lower `ab[i − j][j] = A(i, j)` for `i ≥ j`
   *        (the diagonal's imaginary part, and the corners, not read). By the band's Cholesky factor (LAPACK's `pbsv`).
   * \tparam T `double` or `std::complex<double>`.
   * \throws dimension_error              If \p b has not `n` rows, or \p ab has no row.
   * \throws non_finite_error             If an entry is infinite or NaN.
   * \throws not_positive_definite_error  If `A` is not positive definite.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] matrix<T> solveh_banded(const matrix<T>& ab,
                                        const matrix<T>& b,
                                        bool             lower_form = false)
  {
    if (ab.rows() == 0) {
      throw dimension_error("solveh_banded: the band must have at least one row");
    }
    const std::size_t kd         {ab.rows() - 1};
    detail::require_band(ab, kd + 1, b, "solveh_banded");
    const std::size_t       n    {ab.cols()};
    detail::band_storage<T> work {buffer<T>((kd + 1) * n, T {}), kd + 1};
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t i {j > kd ? j - kd : 0}; i <= j; ++i) { // A(i, j), i ≤ j, from either form
        work(kd + 1 + i - j, j + 1) = lower_form ? detail::conj_of(ab(j - i, i)) : ab(kd + i - j, j);
      }
    }
    detail::band_cholesky(work, n, kd);
    buffer<T> x(b.begin(), b.end());
    detail::band_cholesky_solve(work, n, kd, x, b.cols());
    return matrix<T> {b.rows(), b.cols(), std::move(x)};
  }
} // namespace numerical

#endif // NUMERICAL_BANDED_HPP
