/*!
 * \file decomposition.hpp
 * \brief Matrix factorisations and the algorithms built on them.
 *
 * \ref numerical::lu_decomposition factorises a square matrix once (Gaussian
 * elimination with partial pivoting — an in-place `P A = L U`), then answers any
 * number of queries cheaply: \ref numerical::lu_decomposition::solve a system,
 * its \ref numerical::lu_decomposition::determinant, or its
 * \ref numerical::lu_decomposition::inverse. Free \ref numerical::solve,
 * \ref numerical::determinant and \ref numerical::inverse are one-shot conveniences
 * over it.
 *
 * The scalar `T` must be a field (support division) — `double`,
 * `std::complex<double>`, … — and the partial pivoting is numerically stable for
 * well-conditioned systems. A singular matrix has a zero determinant; asking it
 * to solve or invert throws \ref numerical::singular_matrix_error.
 *
 * Every factorisation here refuses an infinite or NaN entry where it starts, with a
 * \ref numerical::non_finite_error, since none of them means anything on one and the
 * iterative ones would stop at once and return plausible values.
 */
#ifndef NUMERICAL_DECOMPOSITION_HPP
#define NUMERICAL_DECOMPOSITION_HPP

#include <algorithm>
#include <cmath>
#include <complex> // so std::abs resolves for a complex scalar, regardless of include order
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "matrix.hpp"
#include "parallel.hpp"
#include "vector.hpp"

namespace numerical {

  namespace detail {

    //! \brief The columns of one panel of the blocked LU: the trailing update, a product of this depth, is
    //!        deep enough for the packed kernel while the unblocked panel stays a small part of the work.
    inline constexpr std::size_t lu_block {64};

    //! \brief The columns of one panel of the blocked Cholesky factorisation (as \ref lu_block).
    inline constexpr std::size_t cholesky_block {64};


    // conj_of / real_part are written as overloads (not `if constexpr`) so each
    // is a single, fully-covered expression: a complex scalar selects the
    // std::complex overload, any other scalar the primary template.

    //! \brief The conjugate of a real (or general) scalar — the value itself.
    template <typename T>
    [[nodiscard]] T conj_of(const T& x)
    {
      return x;
    }

    //! \brief The conjugate of a complex scalar.
    template <typename U>
    [[nodiscard]] std::complex<U> conj_of(const std::complex<U>& x)
    {
      return std::conj(x);
    }

    //! \brief The real part of a real (or general) scalar as a `double`.
    template <typename T>
    [[nodiscard]] double real_part(const T& x)
    {
      return static_cast<double>(x);
    }

    //! \brief The real part of a complex scalar.
    template <typename U>
    [[nodiscard]] double real_part(const std::complex<U>& x)
    {
      return x.real();
    }

    //! \brief The unit-modulus factor of \p x: `x / |x|` (its sign, for a real scalar), and 1 at 0.
    template <typename T>
    [[nodiscard]] T phase_of(const T& x)
    {
      const double magnitude {static_cast<double>(std::abs(x))};
      return magnitude == 0.0 ? T {1} : x / static_cast<T>(magnitude);
    }

    //! \brief The Euclidean norm of \p x, summed at the scale of its largest magnitude: a plain sum of
    //!        squares overflows near 1e155 and underflows to 0 near 1e-155.
    template <typename T>
    [[nodiscard]] double scaled_norm(const std::vector<T>& x)
    {
      double largest {0.0};
      for (const T& element : x) {
        largest = std::max(largest, static_cast<double>(std::abs(element)));
      }
      if (largest == 0.0) { // the entries are finite: require_finite runs where every caller starts
        return largest;
      }
      double sum {0.0};
      for (const T& element : x) {
        const double ratio {static_cast<double>(std::abs(element)) / largest};
        sum += ratio * ratio;
      }
      return largest * std::sqrt(sum);
    }
  } // namespace detail

  //! \brief Thrown when a matrix has no unique solution (it is singular).
  class singular_matrix_error : public std::runtime_error
  {
  public:

    //! \brief Builds the error. \param[in] message Human-readable cause.
    explicit singular_matrix_error(const std::string& message)
      : std::runtime_error(message)
    {}
  };

  //! \brief Thrown when a decomposition's input holds an infinite or NaN entry: no factorisation of it means
  //!        anything, and the iterative ones (eigh, SVD) would stop at once and return plausible values.
  class non_finite_error : public std::domain_error
  {
  public:

    //! \brief Builds the error. \param[in] message Human-readable cause.
    explicit non_finite_error(const std::string& message)
      : std::domain_error(message)
    {}
  };

  namespace detail {

    //! \brief Whether \p x is finite. \param[in] x The entry.
    template <typename T>
    [[nodiscard]] bool is_finite_entry(const T& x)
    {
      return std::isfinite(x);
    }

    //! \brief Whether \p x is finite: both of its parts. \param[in] x The entry.
    template <typename U>
    [[nodiscard]] bool is_finite_entry(const std::complex<U>& x)
    {
      return std::isfinite(x.real()) && std::isfinite(x.imag());
    }

    //! \brief The one non-finite policy, checked where every decomposition starts.
    //! \param[in] a    The input.
    //! \param[in] what The decomposition, for the message.
    //! \throws non_finite_error If an entry is infinite or NaN.
    template <typename T>
    void require_finite(const matrix<T>& a,
                        const char*      what)
    {
      for (std::size_t i {0}; i < a.rows(); ++i) {
        for (std::size_t j {0}; j < a.cols(); ++j) {
          if (!is_finite_entry(a(i, j))) {
            throw non_finite_error(std::string {what} + ": the matrix holds an infinite or NaN entry");
          }
        }
      }
    }
  } // namespace detail

  /*!
   * \brief An LU factorisation with partial pivoting of a square matrix.
   * \tparam T The scalar element type (a field: `double`, `std::complex<double>`, …).
   *
   * Factorises once on construction; \ref solve, \ref determinant and \ref inverse
   * then reuse the factors. Built for the common case of one matrix and several
   * right-hand sides.
   */
  template <typename T>
  class lu_decomposition
  {
  public:

    /*!
     * \brief Factorises \p a (in place, into combined L\\U with a row permutation).
     * \param[in] a The square matrix to factorise.
     * \throws dimension_error If \p a is not square.
     */
    explicit lu_decomposition(const matrix<T>& a)
      : n_(a.rows())
    {
      if (a.cols() != n_) {
        throw dimension_error("LU decomposition requires a square matrix");
      }
      detail::require_finite(a, "LU decomposition");
      lu_.resize(n_ * n_);
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j < n_; ++j) {
          lu_[(i * n_) + j] = a(i, j);
        }
      }
      pivots_.resize(n_);
      for (std::size_t i {0}; i < n_; ++i) {
        pivots_[i] = i;
      }
      factorize();
    }

    /*!
     * \brief Solves `A x = b` using the stored factors.
     * \param[in] b The right-hand side.
     * \return The solution `x`.
     * \throws dimension_error If `b`'s size differs from the matrix.
     * \throws singular_matrix_error If the matrix is singular.
     */
    [[nodiscard]] vector<T> solve(const vector<T>& b) const
    {
      if (b.size() != n_) {
        throw dimension_error("solve: right-hand side size differs from the matrix");
      }
      if (singular_) {
        throw singular_matrix_error("solve: matrix is singular");
      }
      std::vector<T> y(n_);
      for (std::size_t i {0}; i < n_; ++i) { // forward substitution (unit lower L)
        T sum {b[pivots_[i]]};
        for (std::size_t j {0}; j < i; ++j) {
          sum -= lu_[(i * n_) + j] * y[j];
        }
        y[i] = sum;
      }
      for (std::size_t i {n_}; i-- > 0;) { // back substitution (upper U)
        T sum {y[i]};
        for (std::size_t j {i + 1}; j < n_; ++j) {
          sum -= lu_[(i * n_) + j] * y[j];
        }
        y[i] = sum / lu_[(i * n_) + i];
      }
      return vector<T> {std::move(y)};
    }

    //! \brief The determinant (the pivot sign times the product of U's diagonal; 0 if singular).
    [[nodiscard]] T determinant() const
    {
      T value {static_cast<T>(sign_)};
      for (std::size_t i {0}; i < n_; ++i) {
        value *= lu_[(i * n_) + i];
      }
      return value;
    }

    /*!
     * \brief The determinant as its unit-modulus sign and the natural logarithm of its magnitude, as
     *        `numpy.linalg.slogdet` gives them: a sum of logarithms does not underflow or overflow where
     *        the product does (`det(1e-2·I)` of order 200 is 0.0; its logarithm is −921.0).
     * \return The sign (`+1` or `−1` for a real matrix, a unit complex number for a complex one) and the
     *         logarithm; a singular matrix gives a sign of 0 and a logarithm of −∞.
     */
    [[nodiscard]] std::pair<T, double> log_determinant() const
    {
      if (singular_) {
        return {T {0}, -std::numeric_limits<double>::infinity()};
      }
      T      sign      {static_cast<T>(sign_)};
      double logarithm {0.0};
      for (std::size_t i {0}; i < n_; ++i) {
        const T pivot {lu_[(i * n_) + i]};
        sign      *= detail::phase_of(pivot);
        logarithm += std::log(static_cast<double>(std::abs(pivot)));
      }
      return {sign, logarithm};
    }

    /*!
     * \brief The inverse, solved column by column against the identity.
     * \return The inverse matrix.
     * \throws singular_matrix_error If the matrix is singular.
     */
    [[nodiscard]] matrix<T> inverse() const
    {
      if (singular_) {
        throw singular_matrix_error("inverse: matrix is singular");
      }
      std::vector<T> rows(n_ * n_, T {}); // P·I: row i of the identity's row pivots_[i]
      for (std::size_t i {0}; i < n_; ++i) {
        rows[(i * n_) + pivots_[i]] = T {1};
      }
      solve_rows(rows, n_);
      return matrix<T> {n_, n_, std::move(rows)};
    }

    /*!
     * \brief Solves `A X = B` for every column of \p b at once.
     *
     * By blocks of rows, the products of a block's solution into the rows still to solve go through the
     * matrix product kernel. The forward substitution is the vector \ref solve's, term for term; the backward
     * one gives each row the farther blocks' terms first, so a column agrees with the vector solve to rounding.
     * \param[in] b The right-hand sides (`n × k`).
     * \return `X` (`n × k`).
     * \throws dimension_error If `b`'s row count differs from the matrix.
     * \throws singular_matrix_error If the matrix is singular.
     */
    [[nodiscard]] matrix<T> solve(const matrix<T>& b) const
    {
      if (b.rows() != n_) {
        throw dimension_error("solve: right-hand side row count differs from the matrix");
      }
      if (singular_) {
        throw singular_matrix_error("solve: matrix is singular");
      }
      const std::size_t cols {b.cols()};
      std::vector<T>    rows(n_ * cols);
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t c {0}; c < cols; ++c) {
          rows[(i * cols) + c] = b(pivots_[i], c);
        }
      }
      solve_rows(rows, cols);
      return matrix<T> {n_, cols, std::move(rows)};
    }

  private:

    //! \brief Performs the pivoted elimination, filling \ref lu_ and \ref pivots_.
    /*!
     * \brief `L U X = Y` in place, \p y being `n × cols`, row-major, already permuted. Within a block the
     *        rows are solved one after another, in parallel chunks of columns; across blocks, through the
     *        matrix product kernel.
     */
    void solve_rows(std::vector<T>& y,
                    std::size_t     cols) const
    {
      constexpr std::size_t width  {64};
      const std::size_t     chunks {(cols + width - 1) / width};
      for (std::size_t k0 {0}; k0 < n_; k0 += detail::lu_block) { // forward: unit-lower L
        const std::size_t end {std::min(k0 + detail::lu_block, n_)};
        detail::parallel_for(chunks, (end - k0) * (end - k0) * width / 2, [&](std::size_t chunk) {
                               const std::size_t first {chunk * width};
                               const std::size_t last {std::min(first + width, cols)};
                               for (std::size_t i {k0 + 1}; i < end; ++i) {
                                 for (std::size_t j {k0}; j < i; ++j) {
                                   const T factor {lu_[(i * n_) + j]};
                                   for (std::size_t c {first}; c < last; ++c) {
                                     y[(i * cols) + c] -= factor * y[(j * cols) + c];
                                   }
                                 }
                               }
                             });
        if (end < n_) {
          const std::vector<T> minus {negated_block(end, n_, k0, end)};
          detail::gemm(n_ - end, cols, end - k0, minus.data(), end - k0, y.data() + (k0 * cols), cols,
                       y.data() + (end * cols), cols);
        }
      }
      for (std::size_t k1 {n_}; k1 > 0;) { // backward: upper U, the blocks of the forward pass, last first
        const std::size_t k0 {(k1 - 1) / detail::lu_block * detail::lu_block};
        detail::parallel_for(chunks, (k1 - k0) * (k1 - k0) * width / 2, [&](std::size_t chunk) {
                               const std::size_t first {chunk * width};
                               const std::size_t last {std::min(first + width, cols)};
                               for (std::size_t i {k1}; i-- > k0;) {
                                 for (std::size_t j {i + 1}; j < k1; ++j) {
                                   const T factor {lu_[(i * n_) + j]};
                                   for (std::size_t c {first}; c < last; ++c) {
                                     y[(i * cols) + c] -= factor * y[(j * cols) + c];
                                   }
                                 }
                                 const T pivot {lu_[(i * n_) + i]};
                                 for (std::size_t c {first}; c < last; ++c) {
                                   y[(i * cols) + c] = y[(i * cols) + c] / pivot;
                                 }
                               }
                             });
        if (k0 > 0) {
          const std::vector<T> minus {negated_block(0, k0, k0, k1)};
          detail::gemm(k0, cols, k1 - k0, minus.data(), k1 - k0, y.data() + (k0 * cols), cols, y.data(), cols);
        }
        k1 = k0;
      }
    }

    //! \brief The rows `[r0, r1)` and columns `[c0, c1)` of the factor, negated, row-major and contiguous.
    [[nodiscard]] std::vector<T> negated_block(std::size_t r0,
                                               std::size_t r1,
                                               std::size_t c0,
                                               std::size_t c1) const
    {
      std::vector<T> out((r1 - r0) * (c1 - c0));
      for (std::size_t r {r0}; r < r1; ++r) {
        for (std::size_t c {c0}; c < c1; ++c) {
          out[((r - r0) * (c1 - c0)) + (c - c0)] = -lu_[(r * n_) + c];
        }
      }
      return out;
    }

    // Right-looking, by panels of detail::lu_block columns, as LAPACK's dgetrf: each panel is factorised
    // with partial pivoting (whole rows swapped), its rows of U to the right are solved against the panel's
    // unit-lower L, and the trailing block takes the product of the two through detail::gemm, where the
    // work of a large factorisation is. Each element still receives its updates in ascending order of the
    // pivot, so without multiply-add fusion the factors are the unblocked algorithm's to the bit; an order
    // up to one panel is that algorithm.
    void factorize()
    {
      for (std::size_t k0 {0}; k0 < n_; k0 += detail::lu_block) {
        const std::size_t end {std::min(k0 + detail::lu_block, n_)};
        factorize_panel(k0, end);
        if (end < n_) {
          solve_panel_rows(k0, end);
          update_trailing(k0, end);
        }
      }
    }

    /*!
     * \brief Unblocked LU with partial pivoting of the columns `[k0, end)`, every row below `k0`.
     *
     * The panel is factorised in a column-major copy, where a pivot search and every update run along a
     * column at unit stride (row-major, each reads one element per row: four times slower at order 2048);
     * a pivot's row swap reaches the columns outside the panel in place. The arithmetic is the in-place
     * algorithm's, step for step.
     */
    void factorize_panel(std::size_t k0,
                         std::size_t end)
    {
      const std::size_t rows  {n_ - k0};
      const std::size_t width {end - k0};
      std::vector<T>    panel(rows * width); // column j of the panel at panel[j * rows]
      for (std::size_t r {0}; r < rows; ++r) {
        for (std::size_t j {0}; j < width; ++j) {
          panel[(j * rows) + r] = lu_[((k0 + r) * n_) + k0 + j];
        }
      }
      for (std::size_t j {0}; j < width; ++j) {
        T*          column {panel.data() + (j * rows)};
        std::size_t pivot  {j};
        double      best   {std::abs(column[j])};
        for (std::size_t r {j + 1}; r < rows; ++r) {
          const double mag {std::abs(column[r])};
          if (mag > best) {
            best  = mag;
            pivot = r;
          }
        }
        if (best == 0.0) {
          singular_ = true;
          continue; // a zero pivot column: leave it, the determinant becomes 0
        }
        if (pivot != j) {
          for (std::size_t c {0}; c < width; ++c) {
            std::swap(panel[(c * rows) + j], panel[(c * rows) + pivot]);
          }
          swap_rows_outside(k0 + j, k0 + pivot, k0, end);
          std::swap(pivots_[k0 + j], pivots_[k0 + pivot]);
          sign_ = -sign_;
        }
        for (std::size_t r {j + 1}; r < rows; ++r) {
          column[r] = column[r] / column[j]; // the multiplier, in L's slot
        }
        for (std::size_t c {j + 1}; c < width; ++c) {
          T*      target {panel.data() + (c * rows)};
          const T u      {target[j]};
          for (std::size_t r {j + 1}; r < rows; ++r) {
            target[r] -= column[r] * u;
          }
        }
      }
      for (std::size_t r {0}; r < rows; ++r) {
        for (std::size_t j {0}; j < width; ++j) {
          lu_[((k0 + r) * n_) + k0 + j] = panel[(j * rows) + r];
        }
      }
    }

    //! \brief Swaps rows \p a and \p b of the factor outside the panel's columns `[k0, end)`.
    void swap_rows_outside(std::size_t a,
                           std::size_t b,
                           std::size_t k0,
                           std::size_t end)
    {
      for (std::size_t c {0}; c < k0; ++c) {
        std::swap(lu_[(a * n_) + c], lu_[(b * n_) + c]);
      }
      for (std::size_t c {end}; c < n_; ++c) {
        std::swap(lu_[(a * n_) + c], lu_[(b * n_) + c]);
      }
    }

    //! \brief `U₁₂ = L₁₁⁻¹ A₁₂`: the panel's rows right of it, by forward substitution with its unit-lower L.
    //!        Each block of columns is independent, so blocks run in parallel, each in the sequential order.
    void solve_panel_rows(std::size_t k0,
                          std::size_t end)
    {
      constexpr std::size_t width  {64};
      const std::size_t     blocks {((n_ - end) + width - 1) / width};
      const std::size_t     depth  {end - k0};
      detail::parallel_for(blocks, depth * depth * width / 2, [&](std::size_t block) {
                             const std::size_t first {end + (block * width)};
                             const std::size_t last {std::min(first + width, n_)};
                             for (std::size_t i {k0 + 1}; i < end; ++i) {
                               for (std::size_t p {k0}; p < i; ++p) {
                                 const T factor {lu_[(i * n_) + p]};
                                 for (std::size_t c {first}; c < last; ++c) {
                                   lu_[(i * n_) + c] -= factor * lu_[(p * n_) + c];
                                 }
                               }
                             }
                           });
    }

    //! \brief `A₂₂ −= L₂₁ U₁₂`, as `A₂₂ += (−L₂₁) U₁₂` through the matrix product kernel.
    void update_trailing(std::size_t k0,
                         std::size_t end)
    {
      const std::size_t rows  {n_ - end};
      const std::size_t depth {end - k0};
      std::vector<T>    minus_l(rows * depth);
      for (std::size_t r {0}; r < rows; ++r) {
        for (std::size_t p {0}; p < depth; ++p) {
          minus_l[(r * depth) + p] = -lu_[((end + r) * n_) + k0 + p];
        }
      }
      detail::gemm(rows, rows, depth, minus_l.data(), depth, lu_.data() + (k0 * n_) + end, n_,
                   lu_.data() + (end * n_) + end, n_);
    }

    std::size_t              n_;                //!< The matrix order.
    std::vector<T>           lu_;               //!< Combined L (below diag, unit) and U (on/above), row-major.
    std::vector<std::size_t> pivots_;           //!< Row permutation: row now at i came from pivots_[i].
    int                      sign_     {1};     //!< Permutation sign, for the determinant.
    bool                     singular_ {false}; //!< Set if a zero pivot was met.
  };

  /*!
   * \brief Solves the linear system `A x = b` for a square `A`.
   * \param[in] a The square coefficient matrix.
   * \param[in] b The right-hand side.
   * \return The solution `x`.
   * \throws dimension_error If `a` is not square or `b`'s size differs from it.
   * \throws singular_matrix_error If `a` is singular.
   */
  template <typename T>
  [[nodiscard]] vector<T> solve(const matrix<T>& a,
                                const vector<T>& b)
  {
    return lu_decomposition<T> {a}.solve(b);
  }

  /*!
   * \brief Solves `A X = B` for a square `A` and every column of `B` at once, as `numpy.linalg.solve` with a
   *        matrix right-hand side: one factorisation of `A`, then one substitution per column.
   * \param[in] a The square coefficient matrix (`n × n`).
   * \param[in] b The right-hand sides (`n × k`).
   * \return `X` (`n × k`).
   * \throws dimension_error If `a` is not square or `b`'s row count differs from it.
   * \throws singular_matrix_error If `a` is singular.
   */
  template <typename T>
  [[nodiscard]] matrix<T> solve(const matrix<T>& a,
                                const matrix<T>& b)
  {
    return lu_decomposition<T> {a}.solve(b);
  }

  //! \brief The determinant of a square matrix. \throws dimension_error If not square.
  template <typename T>
  [[nodiscard]] T determinant(const matrix<T>& a)
  {
    return lu_decomposition<T> {a}.determinant();
  }

  /*!
   * \brief The inverse of a square matrix.
   * \param[in] a The matrix.
   * \return Its inverse.
   * \throws dimension_error If `a` is not square.
   * \throws singular_matrix_error If `a` is singular.
   */
  template <typename T>
  [[nodiscard]] matrix<T> inverse(const matrix<T>& a)
  {
    return lu_decomposition<T> {a}.inverse();
  }

  //! \brief Thrown when a Cholesky factorisation finds the matrix not positive-definite.
  class not_positive_definite_error : public std::runtime_error
  {
  public:

    //! \brief Builds the error. \param[in] message Human-readable cause.
    explicit not_positive_definite_error(const std::string& message)
      : std::runtime_error(message)
    {}
  };

  /*!
   * \brief A Cholesky factorisation `A = L Lᴴ` of a Hermitian positive-definite matrix.
   * \tparam T The scalar element type (a field: `double`, `std::complex<double>`, …).
   *
   * `L` is lower-triangular with a real, positive diagonal. Only the lower triangle
   * of \p a is read, so \p a is assumed Hermitian (symmetric, for a real scalar);
   * the factorisation is about twice as cheap as an LU for this special case and
   * cannot be singular. A matrix that is not positive-definite throws
   * \ref not_positive_definite_error.
   */
  template <typename T>
  class cholesky_decomposition
  {
  public:

    /*!
     * \brief Factorises \p a. \param[in] a The Hermitian positive-definite matrix.
     * \throws dimension_error If \p a is not square.
     * \throws not_positive_definite_error If \p a is not positive-definite.
     */
    explicit cholesky_decomposition(const matrix<T>& a)
      : n_(a.rows())
    {
      if (a.cols() != n_) {
        throw dimension_error("Cholesky decomposition requires a square matrix");
      }
      detail::require_finite(a, "Cholesky decomposition");
      lower_.assign(n_ * n_, T {}); // lower-triangular; the upper triangle is zero once factorised
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j <= i; ++j) {
          lower_[(i * n_) + j] = a(i, j);
        }
      }
      factorize();
    }

    //! \brief The lower-triangular factor `L` (with `L Lᴴ == A`).
    [[nodiscard]] matrix<T> lower() const
    {
      return matrix<T> {n_, n_, lower_};
    }

    /*!
     * \brief Solves `A x = b` via `L y = b` then `Lᴴ x = y`.
     * \param[in] b The right-hand side.
     * \return The solution `x`.
     * \throws dimension_error If `b`'s size differs from the matrix.
     */
    [[nodiscard]] vector<T> solve(const vector<T>& b) const
    {
      if (b.size() != n_) {
        throw dimension_error("solve: right-hand side size differs from the matrix");
      }
      std::vector<T> y(n_);
      for (std::size_t i {0}; i < n_; ++i) { // forward: L y = b
        T sum {b[i]};
        for (std::size_t k {0}; k < i; ++k) {
          sum -= lower_[(i * n_) + k] * y[k];
        }
        y[i] = sum / lower_[(i * n_) + i];
      }
      std::vector<T> x(n_);
      for (std::size_t i {n_}; i-- > 0;) { // back: Lᴴ x = y, (Lᴴ)[i][k] = conj(L[k][i])
        T sum {y[i]};
        for (std::size_t k {i + 1}; k < n_; ++k) {
          sum -= detail::conj_of(lower_[(k * n_) + i]) * x[k];
        }
        x[i] = sum / detail::conj_of(lower_[(i * n_) + i]);
      }
      return vector<T> {std::move(x)};
    }

    //! \brief The determinant — the product of the squared diagonal of `L` (real, positive).
    [[nodiscard]] T determinant() const
    {
      T value {static_cast<T>(1)};
      for (std::size_t i {0}; i < n_; ++i) {
        value *= lower_[(i * n_) + i] * lower_[(i * n_) + i];
      }
      return value;
    }

  private:

    // By panels of detail::cholesky_block columns: the panel's diagonal block row by row, the rows below it
    // in parallel (each depends only on the diagonal block), then the lower part of the trailing block takes
    // A22 += (-L21) L21ᴴ from the matrix product kernel. Every element still receives its terms in ascending
    // order, so without multiply-add fusion the factor is the unblocked algorithm's to the bit.
    void factorize()
    {
      for (std::size_t k0 {0}; k0 < n_; k0 += detail::cholesky_block) {
        const std::size_t end {std::min(k0 + detail::cholesky_block, n_)};
        for (std::size_t i {k0}; i < end; ++i) {
          factor_row(i, k0, std::min(i + 1, end));
        }
        if (end < n_) {
          detail::parallel_for(n_ - end, (end - k0) * (end - k0) / 2,
                               [&](std::size_t r) { factor_row(end + r, k0, end); });
          update_trailing(k0, end);
        }
      }
      for (std::size_t i {0}; i < n_; ++i) {
        std::fill(lower_.begin() + static_cast<std::ptrdiff_t>((i * n_) + i + 1),
                  lower_.begin() + static_cast<std::ptrdiff_t>((i + 1) * n_), T {});
      }
    }

    /*!
     * \brief Row \p i of L in the columns `[k0, last)`, the earlier panels' terms already subtracted: the
     *        unblocked step, `L[i][j] = (A[i][j] − Σₖ L[i][k]·conj(L[j][k])) / L[j][j]` over `k ∈ [k0, j)`.
     * \throws not_positive_definite_error At a diagonal that is not positive.
     */
    void factor_row(std::size_t i,
                    std::size_t k0,
                    std::size_t last)
    {
      for (std::size_t j {k0}; j < last; ++j) {
        T sum {lower_[(i * n_) + j]};
        for (std::size_t k {k0}; k < j; ++k) {
          sum -= lower_[(i * n_) + k] * detail::conj_of(lower_[(j * n_) + k]);
        }
        if (i == j) {
          const double diagonal {detail::real_part(sum)};
          if (diagonal <= 0.0) {
            throw not_positive_definite_error("Cholesky: matrix is not positive-definite");
          }
          lower_[(i * n_) + i] = static_cast<T>(std::sqrt(diagonal));
        }
        else {
          lower_[(i * n_) + j] = sum / lower_[(j * n_) + j];
        }
      }
    }

    //! \brief `A22 += (−L21) L21ᴴ`, block column by block column, on and below the diagonal blocks.
    void update_trailing(std::size_t k0,
                         std::size_t end)
    {
      const std::size_t rows  {n_ - end};
      const std::size_t depth {end - k0};
      std::vector<T>    minus_l(rows * depth);
      std::vector<T>    l_adjoint(depth * rows);
      for (std::size_t r {0}; r < rows; ++r) {
        for (std::size_t p {0}; p < depth; ++p) {
          const T value {lower_[((end + r) * n_) + k0 + p]};
          minus_l[(r * depth) + p]  = -value;
          l_adjoint[(p * rows) + r] = detail::conj_of(value);
        }
      }
      for (std::size_t jb {0}; jb < rows; jb += detail::cholesky_block) {
        const std::size_t width {std::min(detail::cholesky_block, rows - jb)};
        detail::gemm(rows - jb, width, depth, minus_l.data() + (jb * depth), depth, l_adjoint.data() + jb, rows,
                     lower_.data() + ((end + jb) * n_) + end + jb, n_);
      }
    }

    std::size_t    n_;     //!< The matrix order.
    std::vector<T> lower_; //!< The lower-triangular factor L, row-major (upper triangle zero).
  };

  //! \brief Cholesky-factorises a Hermitian positive-definite matrix. \see cholesky_decomposition
  template <typename T>
  [[nodiscard]] cholesky_decomposition<T> cholesky(const matrix<T>& a)
  {
    return cholesky_decomposition<T> {a};
  }

  namespace detail {

    //! \brief A thin QR factorisation, row-major: `q` is `rows × cols` with orthonormal columns, `r` is
    //!        `cols × cols` upper triangular with a real, non-negative diagonal.
    template <typename T>
    struct thin_qr
    {
      std::vector<T> q; //!< The orthonormal factor.
      std::vector<T> r; //!< The triangular factor.
    };

    /*!
     * \brief Applies `I − 2 v vᴴ` from the left to the rows `first…first+|v|-1` of a row-major array, on its
     *        columns `from…cols-1`.
     *
     * Row by row: the projection `w = vᴴ A` accumulates one contiguous row at a time, then each row takes
     * `−2 vᵢ w`. A column-by-column walk strides by `cols` on every access and runs several times slower.
     * \param[in,out] a     The array, `cols` entries per row.
     * \param[in]     cols  Its row length.
     * \param[in]     v     The unit reflector.
     * \param[in]     first The first row it acts on.
     * \param[in]     from  The first column it acts on.
     */
    template <typename T>
    void reflect_rows(std::vector<T>&       a,
                      std::size_t           cols,
                      const std::vector<T>& v,
                      std::size_t           first,
                      std::size_t           from)
    {
      std::vector<T> w(cols - from, T {});
      for (std::size_t i {0}; i < v.size(); ++i) {
        const T  weight {conj_of(v[i])};
        const T* row    {a.data() + ((first + i) * cols) + from};
        for (std::size_t j {0}; j < w.size(); ++j) {
          w[j] += weight * row[j];
        }
      }
      for (std::size_t i {0}; i < v.size(); ++i) {
        const T scale {static_cast<T>(2) * v[i]};
        T*      row   {a.data() + ((first + i) * cols) + from};
        for (std::size_t j {0}; j < w.size(); ++j) {
          row[j] -= scale * w[j];
        }
      }
    }

    /*!
     * \brief Factorises the `rows × cols` row-major \p a by Householder reflections, `A = Q R` with
     *        `k = min(rows, cols)`: `Q` is `rows × k` with orthonormal columns and `R` `k × cols` upper
     *        trapezoidal. Each column below the diagonal is reflected onto it, so `Q` is orthonormal to
     *        rounding whatever the conditioning of `A`, and a column dependent on the earlier ones leaves a
     *        zero on `R`'s diagonal.
     * \param[in] a    The matrix, row-major.
     * \param[in] rows Its row count.
     * \param[in] cols Its column count.
     * \return `Q` and `R`.
     */
    template <typename T>
    [[nodiscard]] thin_qr<T> householder_qr(std::vector<T> a,
                                            std::size_t    rows,
                                            std::size_t    cols)
    {
      const std::size_t           depth {std::min(rows, cols)};
      std::vector<std::vector<T>> reflectors(depth); // v_k on rows k…m-1; empty where the column was zero
      for (std::size_t k {0}; k < depth; ++k) {
        std::vector<T> v(rows - k);
        for (std::size_t i {k}; i < rows; ++i) {
          v[i - k] = a[(i * cols) + k];
        }
        const double norm {scaled_norm(v)};
        if (norm == 0.0) {
          continue; // nothing to reflect: the column is already zero below and on the diagonal
        }
        v[0] += phase_of(v[0]) * static_cast<T>(norm); // x − α e₁ with α = −phase(x₀)‖x‖: no cancellation
        const auto length {static_cast<T>(scaled_norm(v))};
        for (T& element : v) {
          element /= length;
        }
        reflect_rows(a, cols, v, k, k); // A ← (I − 2 v vᴴ) A on rows k…m-1, columns k…n-1
        reflectors[k] = std::move(v);
      }
      thin_qr<T> out {.q = std::vector<T>(rows * depth, T {}), .r = std::vector<T>(depth * cols, T {})};
      for (std::size_t i {0}; i < depth; ++i) {
        out.q[(i * depth) + i] = T {1};
        for (std::size_t j {i}; j < cols; ++j) {
          out.r[(i * cols) + j] = a[(i * cols) + j];
        }
      }
      for (std::size_t k {depth}; k-- > 0;) { // Q = H₀ H₁ … applied to the first columns of I
        const std::vector<T>& v {reflectors[k]};
        if (v.empty()) {
          continue;
        }
        reflect_rows(out.q, depth, v, k, 0);
      }
      for (std::size_t k {0}; k < depth; ++k) { // R's diagonal real and non-negative: rotate row k, column k
        const T phase {phase_of(out.r[(k * cols) + k])};
        for (std::size_t j {k}; j < cols; ++j) {
          out.r[(k * cols) + j] *= conj_of(phase);
        }
        for (std::size_t i {0}; i < rows; ++i) {
          out.q[(i * depth) + k] *= phase;
        }
      }
      return out;
    }

    //! \brief The magnitude under which a diagonal entry of `R` or a singular value counts as zero, as
    //!        NumPy draws it: the largest times the machine epsilon times the larger dimension.
    [[nodiscard]] inline double rank_tolerance(double      largest,
                                               std::size_t rows,
                                               std::size_t cols)
    {
      return largest * std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(rows, cols));
    }
  } // namespace detail

  /*!
   * \brief A QR factorisation `A = Q R` of an `m × n` matrix, as NumPy's reduced `qr`.
   * \tparam T The scalar element type (a field: `double`, `std::complex<double>`, …).
   *
   * With `k = min(m, n)`, `Q` is `m × k` with orthonormal columns (`Qᴴ Q = I`) and `R` is `k × n` upper
   * trapezoidal (triangular when `m ≥ n`) with a real, non-negative diagonal. Computed by Householder
   * reflections, so `Q` stays orthonormal to rounding however ill-conditioned `A` is, and a rank-deficient
   * `A` factorises too (with zeros on `R`'s diagonal). \ref solve returns the least-squares solution of
   * `A x ≈ b` for a full-column-rank `A` (`m ≥ n`); for a rank-deficient or wide one, \ref numerical::lstsq
   * gives the minimum-norm solution.
   */
  template <typename T>
  class qr_decomposition
  {
  public:

    /*!
     * \brief Factorises \p a. \param[in] a The matrix, any shape.
     * \throws non_finite_error If an entry is infinite or NaN.
     */
    explicit qr_decomposition(const matrix<T>& a)
      : rows_(a.rows()), cols_(a.cols()), depth_(std::min(rows_, cols_))
    {
      detail::require_finite(a, "QR decomposition");
      std::vector<T> entries(rows_ * cols_);
      for (std::size_t i {0}; i < rows_; ++i) {
        for (std::size_t j {0}; j < cols_; ++j) {
          entries[(i * cols_) + j] = a(i, j);
        }
      }
      detail::thin_qr<T> factors {detail::householder_qr(std::move(entries), rows_, cols_)};
      q_ = std::move(factors.q);
      r_ = std::move(factors.r);
    }

    //! \brief The `m × k` factor `Q` (orthonormal columns), `k = min(m, n)`.
    [[nodiscard]] matrix<T> q() const
    {
      return matrix<T> {rows_, depth_, q_};
    }

    //! \brief The `k × n` upper-trapezoidal factor `R`, `k = min(m, n)`.
    [[nodiscard]] matrix<T> r() const
    {
      return matrix<T> {depth_, cols_, r_};
    }

    /*!
     * \brief The least-squares solution of `A x ≈ b` (`x = R⁻¹ Qᴴ b`) for a full-rank `A`.
     * \param[in] b The right-hand side (size `m`).
     * \return The solution `x` (size `n`).
     * \throws dimension_error If `b`'s size differs from the row count, or `A` is wide (`m < n`: its system is
     *         underdetermined, and \ref numerical::lstsq gives its minimum-norm solution).
     * \throws singular_matrix_error If `A` is rank-deficient: a diagonal entry of `R` is under
     *         \ref detail::rank_tolerance (\ref numerical::lstsq solves such a system).
     */
    [[nodiscard]] vector<T> solve(const vector<T>& b) const
    {
      if (b.size() != rows_) {
        throw dimension_error("solve: right-hand side size differs from the matrix");
      }
      if (rows_ < cols_) {
        throw dimension_error("QR solve: the system is underdetermined (lstsq gives its minimum-norm solution)");
      }
      double largest {0.0};
      for (std::size_t i {0}; i < cols_; ++i) {
        largest = std::max(largest, detail::real_part(r_[(i * cols_) + i]));
      }
      const double   tolerance {detail::rank_tolerance(largest, rows_, cols_)};
      std::vector<T> y(cols_); // y = Qᴴ b
      for (std::size_t i {0}; i < cols_; ++i) {
        if (detail::real_part(r_[(i * cols_) + i]) <= tolerance) {
          throw singular_matrix_error("QR solve: the matrix is rank-deficient (lstsq gives its minimum-norm solution)");
        }
        T sum {};
        for (std::size_t k {0}; k < rows_; ++k) {
          sum += detail::conj_of(q_[(k * depth_) + i]) * b[k];
        }
        y[i] = sum;
      }
      std::vector<T> x(cols_); // back substitution R x = y
      for (std::size_t i {cols_}; i-- > 0;) {
        T sum {y[i]};
        for (std::size_t j {i + 1}; j < cols_; ++j) {
          sum -= r_[(i * cols_) + j] * x[j];
        }
        x[i] = sum / r_[(i * cols_) + i];
      }
      return vector<T> {std::move(x)};
    }

  private:

    std::size_t    rows_;  //!< The number of rows (m).
    std::size_t    cols_;  //!< The number of columns (n).
    std::size_t    depth_; //!< min(m, n).
    std::vector<T> q_;     //!< The m×k factor Q, row-major.
    std::vector<T> r_;     //!< The k×n upper-trapezoidal factor R, row-major.
  };

  //! \brief QR-factorises an `m × n` matrix. \see qr_decomposition
  template <typename T>
  [[nodiscard]] qr_decomposition<T> qr(const matrix<T>& a)
  {
    return qr_decomposition<T> {a};
  }

  //! \brief Thrown when an iterative algorithm has not converged within its bound of sweeps.
  class convergence_error : public std::runtime_error
  {
  public:

    //! \brief Builds the error. \param[in] message Human-readable cause.
    explicit convergence_error(const std::string& message)
      : std::runtime_error(message)
    {}
  };

  namespace detail {

    //! \brief The sweeps a Jacobi method may take: it converges quadratically, in under 15 on every
    //!        matrix measured; the bound turns a non-convergence into an error instead of a hang.
    inline constexpr std::size_t jacobi_sweeps {60};

    //! \brief The implicit QL steps the tridiagonal eigensolver may take per eigenvalue, as LAPACK's
    //!        `dsteqr` bounds them (30 n in all): it converges cubically, in two or three per eigenvalue.
    inline constexpr std::size_t ql_steps_per_value {30};

    /*!
     * \brief A real subdiagonal entry of the tridiagonal: kept with its sign (the rotations take any sign).
     * \param[in] t The entry. \return \p t.
     */
    inline double real_subdiagonal(double t,
                                   double /*delta*/,
                                   double& /*next*/)
    {
      return t;
    }

    /*!
     * \brief A complex subdiagonal entry made real by the next row's unit phase: `next = delta · conj(phase(t))`.
     * \param[in]  t     The entry.
     * \param[in]  delta The phase of the row above.
     * \param[out] next  The phase of the row below.
     * \return `|t|`.
     */
    inline double real_subdiagonal(std::complex<double>  t,
                                   std::complex<double>  delta,
                                   std::complex<double>& next)
    {
      next = delta * std::conj(phase_of(t));
      return std::abs(t);
    }

    /*!
     * \brief Reduces the Hermitian (real: symmetric) `n × n` row-major \p a, both triangles filled, to a real
     *        symmetric tridiagonal by Householder reflections, `A = M T Mᴴ` with `M` unitary.
     *
     * Each reflection zeroes one column below the subdiagonal and updates the trailing block by the rank-2
     * `B − v wᴴ − w vᴴ`, row by row on contiguous memory. A complex subdiagonal entry is then made real by a
     * unit phase per row, as LAPACK's `zhetrd` leaves it for a real iteration; a real one keeps its sign.
     * \param[in,out] a The matrix; destroyed.
     * \param[in]     n Its order (at least 1).
     * \param[out]    d The diagonal of `T` (size \p n).
     * \param[out]    e The subdiagonal of `T` in `e[0…n-2]`, and `e[n-1] = 0` (size \p n).
     * \return `Mᵀ`, row-major: its rows are the columns of `M`.
     */
    template <typename T>
    std::vector<T> tridiagonalize(std::vector<T>&      a,
                                  std::size_t          n,
                                  std::vector<double>& d,
                                  std::vector<double>& e)
    {
      std::vector<std::vector<T>> reflectors(n);
      std::vector<T>              sub(n, T {}); // T's subdiagonal before the phases
      for (std::size_t k {0}; k + 1 < n; ++k) {
        const std::size_t m {n - k - 1};
        std::vector<T>    v(m); // column k below the diagonal: the conjugate of row k right of it
        for (std::size_t i {0}; i < m; ++i) {
          v[i] = conj_of(a[(k * n) + k + 1 + i]);
        }
        const double below {m > 1 ? scaled_norm(std::vector<T>(v.begin() + 1, v.end())) : 0.0};
        if (below == 0.0) {
          sub[k] = v[0]; // already tridiagonal in this column: nothing to reflect
          continue;
        }
        const T alpha {-phase_of(v[0]) * static_cast<T>(scaled_norm(v))}; // the sign that avoids cancellation
        sub[k]  = alpha;
        v[0]   -= alpha;
        const auto length {static_cast<T>(scaled_norm(v))};
        for (T& element : v) {
          element /= length;
        }
        T*             block  {a.data() + ((k + 1) * n) + k + 1}; // the trailing block, row stride n
        std::vector<T> w(m, T {});
        for (std::size_t i {0}; i < m; ++i) { // w = B v
          const T* row {block + (i * n)};
          T        sum {};
          for (std::size_t j {0}; j < m; ++j) {
            sum += row[j] * v[j];
          }
          w[i] = sum;
        }
        T c {};
        for (std::size_t i {0}; i < m; ++i) {
          c += conj_of(v[i]) * w[i];
        }
        for (std::size_t i {0}; i < m; ++i) { // w = 2 B v − 2 (vᴴ B v) v, so H B H = B − v wᴴ − w vᴴ
          w[i] = (static_cast<T>(2) * w[i]) - (static_cast<T>(2) * c * v[i]);
        }
        for (std::size_t i {0}; i < m; ++i) {
          T* row {block + (i * n)};
          for (std::size_t j {0}; j < m; ++j) {
            row[j] -= (v[i] * conj_of(w[j])) + (w[i] * conj_of(v[j]));
          }
        }
        reflectors[k] = std::move(v);
      }
      for (std::size_t i {0}; i < n; ++i) {
        d[i] = real_part(a[(i * n) + i]);
      }
      e[n - 1] = 0.0;
      std::vector<T> q(n * n, T {}); // Q = H₀ H₁ …, built from the last reflection back
      for (std::size_t i {0}; i < n; ++i) {
        q[(i * n) + i] = T {1};
      }
      for (std::size_t k {n}; k-- > 0;) {
        if (!reflectors[k].empty()) {
          reflect_rows(q, n, reflectors[k], k + 1, 0);
        }
      }
      // D T Dᴴ is real with D = diag(δ): δ₀ = 1, δ_{k+1} = δ_k · conj(phase(t_k)); then M = Q Dᴴ.
      std::vector<T> delta(n, T {1});
      for (std::size_t k {0}; k + 1 < n; ++k) {
        e[k] = real_subdiagonal(sub[k], delta[k], delta[k + 1]);
      }
      std::vector<T> mt(n * n);
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          mt[(j * n) + i] = q[(i * n) + j] * conj_of(delta[j]);
        }
      }
      return mt;
    }

    /*!
     * \brief The eigenvalues of the symmetric tridiagonal (\p d, \p e) by implicit QL with Wilkinson-type
     *        shifts (EISPACK's `tql2`), the rotations accumulated into the rows of \p zt.
     * \param[in,out] d         The diagonal; on return, the eigenvalues (unsorted).
     * \param[in,out] e         The subdiagonal in `e[0…n-2]`, `e[n-1] = 0`; destroyed.
     * \param[in,out] zt        Row-major `n × n` whose rows are rotated together (`Qᵀ` from
     *                          \ref tridiagonalize); on return, row `j` is the eigenvector of `d[j]`.
     * \param[in]     n         The order.
     * \param[in]     max_steps The bound of QL steps in all.
     * \throws convergence_error If a subdiagonal entry is not negligible after \p max_steps steps.
     */
    template <typename T>
    void tridiagonal_ql(std::vector<double>& d,
                        std::vector<double>& e,
                        std::vector<T>&      zt,
                        std::size_t          n,
                        std::size_t          max_steps)
    {
      const double epsilon {std::numeric_limits<double>::epsilon()};
      double       shift   {0.0};
      double       largest {0.0};
      std::size_t  steps   {0};
      for (std::size_t l {0}; l < n; ++l) {
        largest = std::max(largest, std::abs(d[l]) + std::abs(e[l]));
        std::size_t m {l};
        while (std::abs(e[m]) > epsilon * largest) { // e[n-1] = 0 stops it
          ++m;
        }
        while (m > l && std::abs(e[l]) > epsilon * largest) {
          if (steps == max_steps) {
            throw convergence_error("symmetric eigendecomposition: QL did not converge within its steps");
          }
          ++steps;
          const double g0   {d[l]};
          double       p    {(d[l + 1] - g0) / (2.0 * e[l])};
          double       r    {std::hypot(p, 1.0)};
          r                  = p < 0.0 ? -r : r;
          d[l]               = e[l] / (p + r);
          d[l + 1]           = e[l] * (p + r);
          const double dl1  {d[l + 1]};
          const double h0   {g0 - d[l]};
          for (std::size_t i {l + 2}; i < n; ++i) {
            d[i] -= h0;
          }
          shift             += h0;
          p                  = d[m];
          double       c    {1.0};
          double       c2   {1.0};
          double       c3   {1.0};
          const double el1  {e[l + 1]};
          double       s    {0.0};
          double       s2   {0.0};
          for (std::size_t i {m}; i-- > l;) {
            c3       = c2;
            c2       = c;
            s2       = s;
            const double g {c * e[i]};
            const double h {c * p};
            r        = std::hypot(p, e[i]);
            e[i + 1] = s * r;
            s        = e[i] / r;
            c        = p / r;
            p        = (c * d[i]) - (s * g);
            d[i + 1] = h + (s * ((c * g) + (s * d[i])));
            T* upper {zt.data() + (i * n)};
            T* lower {zt.data() + ((i + 1) * n)};
            for (std::size_t k {0}; k < n; ++k) { // two contiguous rows: columns i and i+1 of Z
              const T zk1 {lower[k]};
              lower[k] = (s * upper[k]) + (c * zk1);
              upper[k] = (c * upper[k]) - (s * zk1);
            }
          }
          p    = -s * s2 * c3 * el1 * e[l] / dl1;
          e[l] = s * p;
          d[l] = c * p;
        }
        d[l] += shift;
        e[l]  = 0.0;
      }
    }

    //! \brief The Frobenius norm of the strict off-diagonal of the `n × n` \p a.
    [[nodiscard]] inline double jacobi_off_diagonal(const std::vector<double>& a,
                                                    std::size_t                n)
    {
      double sum {0.0};
      for (std::size_t p {0}; p < n; ++p) {
        for (std::size_t q {p + 1}; q < n; ++q) {
          sum += a[(p * n) + q] * a[(p * n) + q];
        }
      }
      return std::sqrt(2.0 * sum);
    }

    //! \brief One Jacobi rotation of the `n × n` \p a zeroing `a[p][q]`, accumulated into \p v.
    inline void jacobi_rotate(std::vector<double>& a,
                              std::vector<double>& v,
                              std::size_t          n,
                              std::size_t          p,
                              std::size_t          q)
    {
      const double apq {a[(p * n) + q]};
      if (apq == 0.0) {
        return; // already zero — nothing to rotate
      }
      const double app   {a[(p * n) + p]};
      const double aqq   {a[(q * n) + q]};
      const double theta {(aqq - app) / (2.0 * apq)};
      const double t     {(theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt((theta * theta) + 1.0))};
      const double c     {1.0 / std::sqrt((t * t) + 1.0)};
      const double s     {t * c};
      a[(p * n) + p] = (c * c * app) - (2.0 * s * c * apq) + (s * s * aqq);
      a[(q * n) + q] = (s * s * app) + (2.0 * s * c * apq) + (c * c * aqq);
      a[(p * n) + q] = 0.0;
      a[(q * n) + p] = 0.0;
      for (std::size_t i {0}; i < n; ++i) {
        if (i != p && i != q) {
          const double aip {a[(i * n) + p]};
          const double aiq {a[(i * n) + q]};
          a[(i * n) + p] = (c * aip) - (s * aiq);
          a[(p * n) + i] = a[(i * n) + p];
          a[(i * n) + q] = (s * aip) + (c * aiq);
          a[(q * n) + i] = a[(i * n) + q];
        }
      }
      for (std::size_t i {0}; i < n; ++i) {
        const double vip {v[(i * n) + p]};
        const double viq {v[(i * n) + q]};
        v[(i * n) + p] = (c * vip) - (s * viq);
        v[(i * n) + q] = (s * vip) + (c * viq);
      }
    }

    /*!
     * \brief Cyclic Jacobi on the full symmetric \p a, the rotations accumulated into \p v.
     * \param[in,out] a          The matrix; on return, diagonal to rounding.
     * \param[in,out] v          The rotations (start from the identity).
     * \param[in]     n          The order.
     * \param[in]     frobenius  The Frobenius norm of \p a, the scale of the stopping threshold.
     * \param[in]     max_sweeps The bound of sweeps.
     * \throws convergence_error If the off-diagonal is not negligible after \p max_sweeps sweeps.
     */
    inline void cyclic_jacobi(std::vector<double>& a,
                              std::vector<double>& v,
                              std::size_t          n,
                              double               frobenius,
                              std::size_t          max_sweeps)
    {
      // The off-diagonal norm strictly decreases each rotation and converges to 0,
      // so this terminates; the threshold is relative to the matrix scale.
      const double threshold {1e-12 * frobenius};
      for (std::size_t sweeps {0}; jacobi_off_diagonal(a, n) > threshold; ++sweeps) {
        if (sweeps == max_sweeps) {
          throw convergence_error("symmetric eigendecomposition: Jacobi did not converge within its sweeps");
        }
        for (std::size_t p {0}; p < n; ++p) {
          for (std::size_t q {p + 1}; q < n; ++q) {
            jacobi_rotate(a, v, n, p, q);
          }
        }
      }
    }
  } // namespace detail

  //! \brief How \ref symmetric_eigen diagonalises.
  enum class eigen_method
  {
    tridiagonal_ql, //!< Householder tridiagonalisation then implicit QL, as LAPACK's `dsyev`: O(n³), the default.
    jacobi          //!< Cyclic Jacobi rotations: several times slower, kept as an independent second method.
  };

  /*!
   * \brief The eigenvalues and eigenvectors of a real symmetric matrix.
   *
   * By default the matrix is reduced to tridiagonal form by Householder reflections and the tridiagonal is
   * diagonalised by implicit QL (\ref eigen_method::tridiagonal_ql), the method of LAPACK's `dsyev`; cyclic
   * Jacobi rotations are the alternative (\ref eigen_method::jacobi). Either way the eigenvalues are real
   * and the eigenvectors orthonormal. The eigenvalues are returned in ascending order, with \ref vectors
   * `j`-th column the eigenvector of \ref values `[j]` (`A = V Λ Vᵀ`). Real symmetric only — the complex
   * Hermitian case is a deliberate future addition. Only the lower triangle is read, as `numpy.linalg.eigh`
   * reads it by default: the upper one is taken to mirror it.
   */
  class symmetric_eigen
  {
  public:

    /*!
     * \brief Diagonalises \p input.
     * \param[in] input  A real symmetric matrix (its lower triangle is read).
     * \param[in] method The algorithm.
     * \throws dimension_error   If \p input is not square.
     * \throws convergence_error If the method did not converge within its bound.
     */
    explicit symmetric_eigen(const matrix<double>& input,
                             eigen_method          method = eigen_method::tridiagonal_ql)
      : n_(input.rows())
    {
      if (input.cols() != n_) {
        throw dimension_error("symmetric eigendecomposition requires a square matrix");
      }
      detail::require_finite(input, "symmetric eigendecomposition");
      std::vector<double> a(n_ * n_);
      std::vector<double> v(n_ * n_, 0.0);
      // Work on the matrix divided by its largest entry: a sum of squares of entries near 1e170
      // overflows and of entries near 1e-170 underflows to 0, and either stops the sweeps before a
      // single rotation. The eigenvalues are scaled back; the eigenvectors do not depend on it.
      double scale {0.0};
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j <= i; ++j) {
          scale = std::max(scale, std::abs(input(i, j)));
        }
      }
      if (scale == 0.0) {
        scale = 1.0;
      }
      double frobenius_squared {0.0};
      for (std::size_t i {0}; i < n_; ++i) {
        v[(i * n_) + i] = 1.0;
        for (std::size_t j {0}; j < n_; ++j) {
          a[(i * n_) + j]    = input(std::max(i, j), std::min(i, j)) / scale; // the lower triangle, mirrored
          frobenius_squared += a[(i * n_) + j] * a[(i * n_) + j];
        }
      }
      std::vector<double> diagonal(n_);
      if (method == eigen_method::jacobi) {
        detail::cyclic_jacobi(a, v, n_, std::sqrt(frobenius_squared), detail::jacobi_sweeps);
        for (std::size_t i {0}; i < n_; ++i) {
          diagonal[i] = a[(i * n_) + i];
        }
      }
      else if (n_ > 0) {
        std::vector<double> off(n_);
        std::vector<double> zt {detail::tridiagonalize(a, n_, diagonal, off)};
        detail::tridiagonal_ql(diagonal, off, zt, n_, detail::ql_steps_per_value * n_);
        for (std::size_t i {0}; i < n_; ++i) { // the rows of zt are the eigenvectors: make them columns
          for (std::size_t j {0}; j < n_; ++j) {
            v[(j * n_) + i] = zt[(i * n_) + j];
          }
        }
      }
      for (double& value : diagonal) {
        value *= scale;
      }
      extract_sorted(diagonal, v);
    }

    //! \brief The eigenvalues, ascending.
    [[nodiscard]] const vector<double>& values() const noexcept
    {
      return values_;
    }

    //! \brief The eigenvectors as columns, ordered to match \ref values.
    [[nodiscard]] const matrix<double>& vectors() const noexcept
    {
      return vectors_;
    }

  private:

    //! \brief Sorts the eigenvalues \p diagonal ascending and reorders \p v's columns to match.
    void extract_sorted(const std::vector<double>& diagonal,
                        const std::vector<double>& v)
    {
      std::vector<std::size_t> order(n_);
      for (std::size_t i {0}; i < n_; ++i) {
        order[i] = i;
      }
      std::sort(order.begin(), order.end(),
                [&diagonal](std::size_t x, std::size_t y) {
                  return diagonal[x] < diagonal[y];
                });
      std::vector<double> values(n_);
      std::vector<double> vectors(n_ * n_);
      for (std::size_t col {0}; col < n_; ++col) {
        values[col] = diagonal[order[col]];
        for (std::size_t row {0}; row < n_; ++row) {
          vectors[(row * n_) + col] = v[(row * n_) + order[col]];
        }
      }
      values_  = vector<double> {std::move(values)};
      vectors_ = matrix<double> {n_, n_, std::move(vectors)};
    }

    std::size_t    n_;       //!< The matrix order.
    vector<double> values_;  //!< Eigenvalues, ascending.
    matrix<double> vectors_; //!< Eigenvectors as columns.
  };

  /*!
   * \brief The eigenvalues and eigenvectors of a complex Hermitian matrix.
   *
   * The matrix is reduced to a real symmetric tridiagonal by Householder reflections and unit phases (the
   * method of LAPACK's `zheev`), which the implicit QL of \ref symmetric_eigen diagonalises. The eigenvalues
   * are real and returned in ascending order, with \ref vectors `j`-th column the eigenvector of
   * \ref values `[j]` (`A = V Λ Vᴴ`, `V` unitary). Only the lower triangle is read and the imaginary part of
   * the diagonal is ignored, as `numpy.linalg.eigh` reads a matrix by default.
   */
  class hermitian_eigen
  {
  public:

    /*!
     * \brief Diagonalises \p input.
     * \param[in] input A complex Hermitian matrix (its lower triangle is read).
     * \throws dimension_error   If \p input is not square.
     * \throws non_finite_error  If an entry is infinite or NaN.
     * \throws convergence_error If QL did not converge within its bound (not observed).
     */
    explicit hermitian_eigen(const matrix<std::complex<double>>& input)
      : n_(input.rows())
    {
      if (input.cols() != n_) {
        throw dimension_error("Hermitian eigendecomposition requires a square matrix");
      }
      detail::require_finite(input, "Hermitian eigendecomposition");
      // Work on the matrix divided by its largest entry, as symmetric_eigen does; the vectors do not depend
      // on it.
      double scale {0.0};
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j <= i; ++j) {
          scale = std::max(scale, std::abs(input(i, j)));
        }
      }
      if (scale == 0.0) {
        scale = 1.0;
      }
      std::vector<std::complex<double>> a(n_ * n_);
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j < n_; ++j) {
          const std::complex<double> lower {i >= j ? input(i, j) : std::conj(input(j, i))};
          a[(i * n_) + j] = (i == j ? std::complex<double> {lower.real(), 0.0} : lower) / scale;
        }
      }
      std::vector<double>               diagonal(n_);
      std::vector<std::complex<double>> zt;
      if (n_ > 0) {
        std::vector<double> off(n_);
        zt = detail::tridiagonalize(a, n_, diagonal, off);
        detail::tridiagonal_ql(diagonal, off, zt, n_, detail::ql_steps_per_value * n_);
      }
      std::vector<std::size_t> order(n_);
      for (std::size_t i {0}; i < n_; ++i) {
        order[i] = i;
      }
      std::sort(order.begin(), order.end(), [&diagonal](std::size_t x, std::size_t y) { return diagonal[x] < diagonal[y]; });
      std::vector<double>               values(n_);
      std::vector<std::complex<double>> vectors(n_ * n_);
      for (std::size_t col {0}; col < n_; ++col) {
        values[col] = diagonal[order[col]] * scale;
        for (std::size_t row {0}; row < n_; ++row) {
          vectors[(row * n_) + col] = zt[(order[col] * n_) + row]; // row `order[col]` of zt is that vector
        }
      }
      values_  = vector<double> {std::move(values)};
      vectors_ = matrix<std::complex<double>> {n_, n_, std::move(vectors)};
    }

    //! \brief The eigenvalues, ascending.
    [[nodiscard]] const vector<double>& values() const noexcept
    {
      return values_;
    }

    //! \brief The eigenvectors as columns, ordered to match \ref values.
    [[nodiscard]] const matrix<std::complex<double>>& vectors() const noexcept
    {
      return vectors_;
    }

  private:

    std::size_t                  n_;       //!< The matrix order.
    vector<double>               values_;  //!< Eigenvalues, ascending.
    matrix<std::complex<double>> vectors_; //!< Eigenvectors as columns.
  };

  namespace detail {

    /*!
     * \brief One-sided (Hestenes) Jacobi on the columns of the `n × n` row-major \p w: plane rotations,
     *        accumulated into \p v, until every pair of columns is orthogonal to rounding. The column
     *        norms are then the singular values of the input, each to high relative accuracy.
     * \param[in,out] w          The matrix whose columns are orthogonalised.
     * \param[in,out] v          The rotations, accumulated (start from the identity).
     * \param[in]     n          The order.
     * \param[in]     max_sweeps The bound of sweeps.
     * \throws convergence_error If a sweep still rotated at the bound.
     */
    template <typename T>
    void one_sided_jacobi(std::vector<T>& w,
                          std::vector<T>& v,
                          std::size_t     n,
                          std::size_t     max_sweeps)
    {
      // The sweeps rotate columns: work on the transposes, where each column is a contiguous row. A walk
      // down the columns of the row-major arrays strides by n on every access and runs several times slower.
      const auto transpose_square {[n](std::vector<T>& m) {
                                     for (std::size_t i {0}; i < n; ++i) {
                                       for (std::size_t j {i + 1}; j < n; ++j) {
                                         std::swap(m[(i * n) + j], m[(j * n) + i]);
                                       }
                                     }
                                   }};
      transpose_square(w);
      transpose_square(v);
      const double epsilon {std::numeric_limits<double>::epsilon()};
      for (std::size_t sweep {1};; ++sweep) {
        bool rotated {false};
        for (std::size_t p {0}; p < n; ++p) {
          for (std::size_t q {p + 1}; q < n; ++q) {
            const T* wp    {w.data() + (p * n)};
            const T* wq    {w.data() + (q * n)};
            double   alpha {0.0};
            double   beta  {0.0};
            T        gamma {};
            for (std::size_t i {0}; i < n; ++i) {
              alpha += std::norm(wp[i]);
              beta  += std::norm(wq[i]);
              gamma += conj_of(wp[i]) * wq[i];
            }
            const double magnitude {static_cast<double>(std::abs(gamma))};
            if (magnitude <= epsilon * std::sqrt(alpha) * std::sqrt(beta)) {
              continue; // already orthogonal to rounding (or a zero column)
            }
            rotated = true;
            // Turn column q by the phase of γ so that the pair's inner product is real, then rotate.
            const T      turn {conj_of(gamma) / static_cast<T>(magnitude)};
            const double zeta {(beta - alpha) / (2.0 * magnitude)};
            const double t    {(zeta >= 0.0 ? 1.0 : -1.0) / (std::abs(zeta) + std::sqrt(1.0 + (zeta * zeta)))};
            const double c    {1.0 / std::sqrt(1.0 + (t * t))};
            const double s    {c * t};
            for (std::vector<T>* side : {&w, &v}) {
              T* mp {side->data() + (p * n)};
              T* mq {side->data() + (q * n)};
              for (std::size_t i {0}; i < n; ++i) {
                const T left  {mp[i]};
                const T right {turn * mq[i]};
                mp[i] = (static_cast<T>(c) * left) - (static_cast<T>(s) * right);
                mq[i] = (static_cast<T>(s) * left) + (static_cast<T>(c) * right);
              }
            }
          }
        }
        if (!rotated) {
          transpose_square(w);
          transpose_square(v);
          return;
        }
        if (sweep == max_sweeps) {
          throw convergence_error("SVD: one-sided Jacobi did not converge within its sweeps");
        }
      }
    }

    //! \brief The QR steps the bidiagonal singular value iteration may take per value: it converges in two
    //!        or three; the bound turns a non-convergence into an error instead of a hang.
    inline constexpr std::size_t svd_qr_steps_per_value {30};

    /*!
     * \brief Reduces the square `n × n` row-major \p b to real upper-bidiagonal form, `B = U_b D V_bᴴ` with
     *        `D` real, by Householder reflections alternately from the left and the right (Golub–Kahan).
     *
     * A complex entry left on the bidiagonal is turned real by a unit phase on its row or column, as LAPACK
     * does, so the iteration that follows needs only real rotations.
     * \param[in,out] b  The matrix; destroyed.
     * \param[in]     n  Its order.
     * \param[out]    d  The diagonal (size \p n).
     * \param[out]    e  The superdiagonal in `e[0…n-2]`, and `e[n-1] = 0` (size \p n).
     * \param[out]    ut `U_bᵀ`, row-major: its rows are the columns of `U_b`.
     * \param[out]    vt `V_bᵀ`, row-major: its rows are the columns of `V_b`.
     */
    template <typename T>
    void bidiagonalize(std::vector<T>&      b,
                       std::size_t          n,
                       std::vector<double>& d,
                       std::vector<double>& e,
                       std::vector<T>&      ut,
                       std::vector<T>&      vt)
    {
      std::vector<std::vector<T>> left(n);  // the left reflector of step k, empty where none was needed
      std::vector<std::vector<T>> right(n); // the right reflector acting on columns k+1…, likewise
      std::vector<T>              left_phase(n, T {1});
      std::vector<T>              right_phase(n, T {1});
      const auto                  reflector {[](std::vector<T>& v) { // unit v with (I − 2 v vᴴ) x ∥ e₁
                                               const double norm {scaled_norm(v)};
                                               v[0] += phase_of(v[0]) * static_cast<T>(norm);
                                               const auto length {static_cast<T>(scaled_norm(v))};
                                               for (T & element : v) {
                                                 element /= length;
                                               }
                                             }};
      for (std::size_t k {0}; k < n; ++k) {
        std::vector<T> x(n - k); // column k from the diagonal down
        for (std::size_t i {k}; i < n; ++i) {
          x[i - k] = b[(i * n) + k];
        }
        if (scaled_norm(std::vector<T>(x.begin() + 1, x.end())) > 0.0) {
          reflector(x);
          reflect_rows(b, n, x, k, k);
          left[k] = std::move(x);
        }
        left_phase[k] = conj_of(phase_of(b[(k * n) + k])); // row k times it puts |b_kk| on the diagonal
        for (std::size_t j {k}; j < n; ++j) {
          b[(k * n) + j] *= left_phase[k];
        }
        d[k] = std::abs(b[(k * n) + k]);
        if (k + 1 == n) {
          break;
        }
        std::vector<T> y(n - k - 1); // row k right of the diagonal, conjugated: B ← B H with H yᴴ ∥ e₁
        for (std::size_t j {k + 1}; j < n; ++j) {
          y[j - k - 1] = conj_of(b[(k * n) + j]);
        }
        if (y.size() > 1 && scaled_norm(std::vector<T>(y.begin() + 1, y.end())) > 0.0) {
          reflector(y);
          for (std::size_t i {k}; i < n; ++i) { // row ← row (I − 2 y yᴴ), rows above k are zero there
            T* row {b.data() + (i * n) + k + 1};
            T  dot {};
            for (std::size_t j {0}; j < y.size(); ++j) {
              dot += row[j] * y[j];
            }
            const T scale {static_cast<T>(2) * dot};
            for (std::size_t j {0}; j < y.size(); ++j) {
              row[j] -= scale * conj_of(y[j]);
            }
          }
          right[k] = std::move(y);
        }
        right_phase[k] = conj_of(phase_of(b[(k * n) + k + 1])); // column k+1 times it: |b_k,k+1| above
        for (std::size_t i {k}; i < n; ++i) {
          b[(i * n) + k + 1] *= right_phase[k];
        }
        e[k] = std::abs(b[(k * n) + k + 1]);
      }
      if (n > 0) {
        e[n - 1] = 0.0;
      }
      // U_b = L₀ᴴ L₁ᴴ … with L_k = diag(phase at k) H_k, and V_b = R₀ R₁ … with R_k = H'_k diag(phase at
      // k+1): each built on the identity from the last factor back, row by row.
      std::vector<T> u(n * n, T {});
      std::vector<T> v(n * n, T {});
      for (std::size_t i {0}; i < n; ++i) {
        u[(i * n) + i] = T {1};
        v[(i * n) + i] = T {1};
      }
      for (std::size_t k {n}; k-- > 0;) {
        for (std::size_t j {0}; j < n; ++j) {
          u[(k * n) + j] *= conj_of(left_phase[k]);
        }
        if (!left[k].empty()) {
          reflect_rows(u, n, left[k], k, 0);
        }
        if (k + 1 < n) {
          for (std::size_t j {0}; j < n; ++j) {
            v[((k + 1) * n) + j] *= right_phase[k];
          }
          if (!right[k].empty()) {
            reflect_rows(v, n, right[k], k + 1, 0);
          }
        }
      }
      ut.assign(n * n, T {});
      vt.assign(n * n, T {});
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          ut[(j * n) + i] = u[(i * n) + j];
          vt[(j * n) + i] = v[(i * n) + j];
        }
      }
    }

    /*!
     * \brief Rotates rows \p i and \p j of the row-major \p m (`n` per row) by the plane rotation
     *        `(c, s)`: `row_i ← c row_i + s row_j`, `row_j ← −s row_i + c row_j`.
     */
    template <typename T>
    void rotate_rows(std::vector<T>& m,
                     std::size_t     n,
                     std::size_t     i,
                     std::size_t     j,
                     double          c,
                     double          s)
    {
      T* first  {m.data() + (i * n)};
      T* second {m.data() + (j * n)};
      for (std::size_t k {0}; k < n; ++k) {
        const T a {first[k]};
        const T b {second[k]};
        first[k]  = (static_cast<T>(c) * a) + (static_cast<T>(s) * b);
        second[k] = (static_cast<T>(c) * b) - (static_cast<T>(s) * a);
      }
    }

    /*!
     * \brief The singular values of the real upper bidiagonal (\p d, \p e) by implicit-shift QR with
     *        deflation (LINPACK's `dsvdc`, the Golub–Kahan–Reinsch iteration), the rotations accumulated
     *        into the rows of \p ut and \p vt.
     * \param[in,out] d         The diagonal; on return, the singular values, descending and non-negative.
     * \param[in,out] e         The superdiagonal in `e[0…n-2]`, `e[n-1] = 0`; destroyed.
     * \param[in,out] ut        Row-major `n × n`; on return, row `j` is the left vector of `d[j]`.
     * \param[in,out] vt        Row-major `n × n`; on return, row `j` is the right vector of `d[j]`.
     * \param[in]     n         The order.
     * \param[in]     max_steps The bound of QR steps in all.
     * \throws convergence_error If the iteration has not converged after \p max_steps steps.
     */
    template <typename T>
    void bidiagonal_qr(std::vector<double>& d,
                       std::vector<double>& e,
                       std::vector<T>&      ut,
                       std::vector<T>&      vt,
                       std::size_t          n,
                       std::size_t          max_steps)
    {
      const double          epsilon {std::numeric_limits<double>::epsilon()};
      const double          tiny    {std::numeric_limits<double>::min() / epsilon};
      constexpr std::size_t none    {std::numeric_limits<std::size_t>::max()};
      std::size_t           p       {n};
      std::size_t           steps   {0};
      while (p > 0) {
        // The largest k < p−1 with a negligible e[k] (none: the loop ran out and wrapped).
        std::size_t k {p - 1};
        while (k-- > 0) {
          if (std::abs(e[k]) <= tiny + (epsilon * (std::abs(d[k]) + std::abs(d[k + 1])))) {
            e[k] = 0.0;
            break;
          }
        }
        const std::size_t top {k == none ? 0 : k + 1}; // the unreduced block is top…p−1
        if (top == p - 1) { // d[p−1] has converged: make it non-negative, sort it into place
          if (d[top] < 0.0) {
            d[top] = -d[top];
            for (std::size_t i {0}; i < n; ++i) {
              vt[(top * n) + i] = -vt[(top * n) + i];
            }
          }
          for (std::size_t j {top}; j + 1 < n && d[j] < d[j + 1]; ++j) {
            std::swap(d[j], d[j + 1]);
            std::swap_ranges(vt.begin() + static_cast<std::ptrdiff_t>(j * n),
                             vt.begin() + static_cast<std::ptrdiff_t>((j + 1) * n),
                             vt.begin() + static_cast<std::ptrdiff_t>((j + 1) * n));
            std::swap_ranges(ut.begin() + static_cast<std::ptrdiff_t>(j * n),
                             ut.begin() + static_cast<std::ptrdiff_t>((j + 1) * n),
                             ut.begin() + static_cast<std::ptrdiff_t>((j + 1) * n));
          }
          --p;
          continue;
        }
        // The last negligible d[zero] in the block (none: there is none).
        std::size_t zero {none};
        for (std::size_t ks {p}; ks-- > top;) {
          const double around {std::abs(e[ks]) + (ks != top ? std::abs(e[ks - 1]) : 0.0)};
          if (std::abs(d[ks]) <= tiny + (epsilon * around)) {
            d[ks] = 0.0;
            zero  = ks;
            break;
          }
        }
        if (zero == p - 1) { // chase e[p−2] up the block with rotations on the right
          double f {e[p - 2]};
          e[p - 2] = 0.0;
          for (std::size_t j {p - 1}; j-- > top;) {
            const double t  {std::hypot(d[j], f)};
            const double cs {d[j] / t};
            const double sn {f / t};
            d[j] = t;
            if (j != top) {
              f        = -sn * e[j - 1];
              e[j - 1] = cs * e[j - 1];
            }
            rotate_rows(vt, n, j, p - 1, cs, sn);
          }
          continue;
        }
        if (zero != none) { // split at d[zero] = 0: chase e[zero] along the row with rotations on the left
          double f {e[zero]};
          e[zero] = 0.0;
          for (std::size_t j {zero + 1}; j < p; ++j) {
            const double t  {std::hypot(d[j], f)};
            const double cs {d[j] / t};
            const double sn {f / t};
            d[j] = t;
            f    = -sn * e[j];
            e[j] = cs * e[j];
            rotate_rows(ut, n, j, zero, cs, sn);
          }
          continue;
        }
        if (steps == max_steps) {
          throw convergence_error("SVD: the bidiagonal QR iteration did not converge within its steps");
        }
        ++steps;
        // One implicit QR step on top…p−1, shifted by the eigenvalue of the trailing 2 × 2 of BᵀB nearer
        // its last diagonal entry.
        const double scale {std::max({std::abs(d[p - 1]), std::abs(d[p - 2]), std::abs(e[p - 2]), std::abs(d[top]),
                                      std::abs(e[top])})};
        const double sp    {d[p - 1] / scale};
        const double spm1  {d[p - 2] / scale};
        const double epm1  {e[p - 2] / scale};
        const double sk    {d[top] / scale};
        const double ek    {e[top] / scale};
        const double bb    {(((spm1 + sp) * (spm1 - sp)) + (epm1 * epm1)) / 2.0};
        const double cc    {(sp * epm1) * (sp * epm1)};
        double       shift {0.0};
        if (bb != 0.0 || cc != 0.0) {
          shift = std::sqrt((bb * bb) + cc);
          shift = cc / (bb < 0.0 ? bb - shift : bb + shift);
        }
        double f {((sk + sp) * (sk - sp)) + shift};
        double g {sk * ek};
        for (std::size_t j {top}; j + 1 < p; ++j) {
          double t  {std::hypot(f, g)};
          double cs {f / t};
          double sn {g / t};
          if (j != top) {
            e[j - 1] = t;
          }
          f        = (cs * d[j]) + (sn * e[j]);
          e[j]     = (cs * e[j]) - (sn * d[j]);
          g        = sn * d[j + 1];
          d[j + 1] = cs * d[j + 1];
          rotate_rows(vt, n, j, j + 1, cs, sn);
          t        = std::hypot(f, g);
          cs       = f / t;
          sn       = g / t;
          d[j]     = t;
          f        = (cs * e[j]) + (sn * d[j + 1]);
          d[j + 1] = (cs * d[j + 1]) - (sn * e[j]);
          g        = sn * e[j + 1];
          e[j + 1] = cs * e[j + 1];
          rotate_rows(ut, n, j, j + 1, cs, sn);
        }
        e[p - 2] = f;
      }
    }

    //! \brief The conjugate transpose of \p a.
    template <typename T>
    [[nodiscard]] matrix<T> adjoint(const matrix<T>& a)
    {
      std::vector<T> entries(a.rows() * a.cols());
      for (std::size_t i {0}; i < a.rows(); ++i) {
        for (std::size_t j {0}; j < a.cols(); ++j) {
          entries[(j * a.rows()) + i] = conj_of(a(i, j));
        }
      }
      return matrix<T> {a.cols(), a.rows(), std::move(entries)};
    }
  } // namespace detail

  //! \brief How \ref svd_decomposition computes the decomposition of `R`, after a Householder QR.
  enum class svd_method
  {
    bidiagonal_qr, //!< Golub–Kahan bidiagonalisation then implicit-shift QR, as LAPACK's `dgesvd`: the default.
    jacobi         //!< One-sided Jacobi: several times slower, each singular value to high relative accuracy.
  };

  /*!
   * \brief The singular value decomposition `A = U Σ Vᴴ` of an `m × n` matrix.
   * \tparam T The scalar element type (`double` or `std::complex<double>`).
   *
   * With `k = min(m, n)`: the `k` singular values, descending; `U` (`m × k`) and `V` (`n × k`) with
   * orthonormal columns, also where `A` is rank-deficient. Computed after a Householder QR of `A` (of `Aᴴ`
   * when `m < n`): by default `R` is reduced to bidiagonal form and the bidiagonal diagonalised by
   * implicit-shift QR (\ref svd_method::bidiagonal_qr), the method of LAPACK's `dgesvd`, each singular value
   * to an absolute accuracy of the largest times the machine epsilon. One-sided Jacobi on `R`
   * (\ref svd_method::jacobi) gives every singular value to high relative accuracy — the small ones of a
   * graded matrix too — at several times the cost. \ref vh is `Vᴴ`, the third factor NumPy's `svd` returns.
   */
  template <typename T>
  class svd_decomposition
  {
  public:

    /*!
     * \brief Decomposes \p a.
     * \param[in] a      The matrix.
     * \param[in] method The algorithm for `R`.
     * \throws non_finite_error If an entry is infinite or NaN.
     * \throws convergence_error If the iteration does not converge within its bound (not observed).
     */
    explicit svd_decomposition(const matrix<T>& a,
                               svd_method       method = svd_method::bidiagonal_qr)
    {
      detail::require_finite(a, "SVD");
      if (a.rows() >= a.cols()) {
        factor(a, method);
      }
      else { // A = (Aᴴ)ᴴ = (U' Σ V'ᴴ)ᴴ = V' Σ U'ᴴ
        factor(detail::adjoint(a), method);
        std::swap(u_, v_);
      }
    }

    //! \brief The singular values, descending.
    [[nodiscard]] const vector<double>& singular_values() const noexcept
    {
      return singular_values_;
    }

    //! \brief The `m × k` left singular vectors `U` (columns).
    [[nodiscard]] const matrix<T>& u() const noexcept
    {
      return u_;
    }

    //! \brief The `n × k` right singular vectors `V` (columns).
    [[nodiscard]] const matrix<T>& v() const noexcept
    {
      return v_;
    }

    //! \brief `Vᴴ` (`k × n`), so that `A = U Σ Vᴴ` reads `u() * diag(singular_values()) * vh()`.
    [[nodiscard]] matrix<T> vh() const
    {
      return detail::adjoint(v_);
    }

  private:

    //! \brief Decomposes an `m × n` \p a with `m ≥ n` into \ref u_, \ref singular_values_ and \ref v_.
    void factor(const matrix<T>& a,
                svd_method       method)
    {
      const std::size_t rows {a.rows()};
      const std::size_t cols {a.cols()};
      // Work on A divided by its largest magnitude: the sums of squares in the sweeps overflow near
      // 1e155 and underflow near 1e-155. The singular values are scaled back; the vectors do not change.
      double scale {0.0};
      for (std::size_t i {0}; i < rows; ++i) {
        for (std::size_t j {0}; j < cols; ++j) {
          scale = std::max(scale, static_cast<double>(std::abs(a(i, j))));
        }
      }
      if (scale == 0.0) {
        scale = 1.0;
      }
      std::vector<T> entries(rows * cols);
      for (std::size_t i {0}; i < rows; ++i) {
        for (std::size_t j {0}; j < cols; ++j) {
          entries[(i * cols) + j] = a(i, j) / static_cast<T>(scale);
        }
      }
      detail::thin_qr<T> factors {detail::householder_qr(std::move(entries), rows, cols)};
      if (method == svd_method::bidiagonal_qr) {
        std::vector<double> d(cols);
        std::vector<double> e(cols);
        std::vector<T>      ut;
        std::vector<T>      vt;
        detail::bidiagonalize(factors.r, cols, d, e, ut, vt);
        detail::bidiagonal_qr(d, e, ut, vt, cols, detail::svd_qr_steps_per_value * cols);
        std::vector<T> left(cols * cols);
        std::vector<T> right(cols * cols);
        for (std::size_t i {0}; i < cols; ++i) { // rows of ut and vt are the vectors: make them columns
          for (std::size_t j {0}; j < cols; ++j) {
            left[(i * cols) + j]  = ut[(j * cols) + i];
            right[(i * cols) + j] = vt[(j * cols) + i];
          }
        }
        for (double& value : d) {
          value *= scale;
        }
        singular_values_ = vector<double> {std::move(d)};
        u_               = matrix<T> {rows, cols, std::move(factors.q)} *matrix<T> {cols, cols, std::move(left)};
        v_               = matrix<T> {cols, cols, std::move(right)};
        return;
      }
      std::vector<T>     w                    {std::move(factors.r)}; // A = Q R, and R V = W with orthogonal columns
      std::vector<T>     right(cols * cols, T {});
      for (std::size_t i {0}; i < cols; ++i) {
        right[(i * cols) + i] = T {1};
      }
      detail::one_sided_jacobi(w, right, cols, detail::jacobi_sweeps);

      std::vector<double> norms(cols);
      for (std::size_t j {0}; j < cols; ++j) {
        std::vector<T> column(cols);
        for (std::size_t i {0}; i < cols; ++i) {
          column[i] = w[(i * cols) + j];
        }
        norms[j] = detail::scaled_norm(column);
      }
      std::vector<std::size_t> order(cols);
      for (std::size_t j {0}; j < cols; ++j) {
        order[j] = j;
      }
      std::stable_sort(order.begin(), order.end(), [&norms](std::size_t x, std::size_t y) { return norms[x] > norms[y]; });

      // The left vectors of R: each nonzero column of W over its norm; a zero one has none, and takes
      // the unit vector that leaves the largest residual against the columns already set, made
      // orthogonal to them.
      std::vector<T>      left(cols * cols, T {});
      std::vector<T>      sorted_right(cols * cols);
      std::vector<double> values(cols);
      for (std::size_t d {0}; d < cols; ++d) {
        const std::size_t j {order[d]};
        values[d] = norms[j] * scale;
        for (std::size_t i {0}; i < cols; ++i) {
          sorted_right[(i * cols) + d] = right[(i * cols) + j];
        }
        if (norms[j] > 0.0) {
          for (std::size_t i {0}; i < cols; ++i) {
            left[(i * cols) + d] = w[(i * cols) + j] / static_cast<T>(norms[j]);
          }
          continue;
        }
        std::vector<T> best;
        double         best_norm {-1.0};
        for (std::size_t k {0}; k < cols; ++k) {
          std::vector<T> candidate(cols, T {});
          candidate[k] = T {1};
          orthogonalise(candidate, left, cols, d);
          const double norm {detail::scaled_norm(candidate)};
          if (norm > best_norm) {
            best      = std::move(candidate);
            best_norm = norm;
          }
        }
        orthogonalise(best, left, cols, d); // a second pass restores orthogonality lost to rounding
        const double norm {detail::scaled_norm(best)};
        for (std::size_t i {0}; i < cols; ++i) {
          left[(i * cols) + d] = best[i] / static_cast<T>(norm);
        }
      }
      singular_values_ = vector<double> {std::move(values)};
      u_               = matrix<T> {rows, cols, std::move(factors.q)} *matrix<T> {cols, cols, std::move(left)};
      v_               = matrix<T> {cols, cols, std::move(sorted_right)};
    }

    //! \brief Removes from \p x its components along the first \p count columns of the `n × n` \p basis.
    static void orthogonalise(std::vector<T>&       x,
                              const std::vector<T>& basis,
                              std::size_t           n,
                              std::size_t           count)
    {
      for (std::size_t c {0}; c < count; ++c) {
        T projection {};
        for (std::size_t i {0}; i < n; ++i) {
          projection += detail::conj_of(basis[(i * n) + c]) * x[i];
        }
        for (std::size_t i {0}; i < n; ++i) {
          x[i] -= projection * basis[(i * n) + c];
        }
      }
    }

    vector<double> singular_values_; //!< Singular values, descending.
    matrix<T>      u_;               //!< m×k left singular vectors.
    matrix<T>      v_;               //!< n×k right singular vectors.
  };

  namespace detail {

    //! \brief How many of the singular values \p sigma of an `rows × cols` matrix exceed \ref rank_tolerance
    //!        of the largest: its numerical rank (the first that many, \p sigma being descending).
    [[nodiscard]] inline std::size_t numerical_rank(const vector<double>& sigma,
                                                    std::size_t           rows,
                                                    std::size_t           cols)
    {
      double largest {0.0};
      for (std::size_t d {0}; d < sigma.size(); ++d) {
        largest = std::max(largest, sigma[d]);
      }
      const double tolerance {rank_tolerance(largest, rows, cols)};
      std::size_t  rank      {0};
      for (std::size_t d {0}; d < sigma.size(); ++d) {
        rank += static_cast<std::size_t>(sigma[d] > tolerance);
      }
      return rank;
    }
  } // namespace detail

  /*!
   * \brief The minimum-norm least-squares solution of `A x ≈ b`, as NumPy's `lstsq` gives it.
   * \tparam T The scalar element type (`double` or `std::complex<double>`).
   *
   * From the SVD of `A`: `x = V Σ⁺ Uᴴ b`, where a singular value under \ref detail::rank_tolerance of the
   * largest counts as zero. Any shape, any rank: an overdetermined full-rank system gives its least-squares
   * solution, an underdetermined or rank-deficient one the solution of smallest norm.
   * \param[in] a The `m × n` matrix.
   * \param[in] b The right-hand side (size `m`).
   * \return `x` (size `n`).
   * \throws dimension_error If `b`'s size differs from the row count.
   * \throws non_finite_error If an entry of \p a is infinite or NaN.
   */
  template <typename T>
  [[nodiscard]] vector<T> lstsq(const matrix<T>& a,
                                const vector<T>& b)
  {
    const matrix<T> column {lstsq(a, matrix<T> {b.size(), 1, std::vector<T>(b.begin(), b.end())})};
    return vector<T> {std::vector<T>(column.begin(), column.end())};
  }

  /*!
   * \brief The minimum-norm least-squares solution of `A X ≈ B` for every column of `B` at once, as
   *        `numpy.linalg.lstsq` with a matrix right-hand side: one SVD of `A`, `X = V Σ⁺ Uᴴ B`.
   * \tparam T The scalar element type (`double` or `std::complex<double>`).
   * \param[in] a The `m × n` matrix.
   * \param[in] b The right-hand sides (`m × k`).
   * \return `X` (`n × k`).
   * \throws dimension_error If `b`'s row count differs from `a`'s.
   * \throws non_finite_error If an entry of \p a is infinite or NaN.
   */
  template <typename T>
  [[nodiscard]] matrix<T> lstsq(const matrix<T>& a,
                                const matrix<T>& b)
  {
    if (b.rows() != a.rows()) {
      throw dimension_error("lstsq: right-hand side size differs from the matrix");
    }
    detail::require_finite(a, "lstsq");
    const svd_decomposition<T> s                        {a};
    const vector<double>&      sigma                    {s.singular_values()};
    const std::size_t          rank                     {detail::numerical_rank(sigma, a.rows(), a.cols())};
    std::vector<T>             x(a.cols() * b.cols(), T {});
    for (std::size_t c {0}; c < b.cols(); ++c) {
      for (std::size_t d {0}; d < rank; ++d) {
        T coefficient {}; // (u_dᴴ b_c) / σ_d
        for (std::size_t i {0}; i < a.rows(); ++i) {
          coefficient += detail::conj_of(s.u()(i, d)) * b(i, c);
        }
        coefficient /= static_cast<T>(sigma[d]);
        for (std::size_t j {0}; j < a.cols(); ++j) {
          x[(j * b.cols()) + c] += coefficient * s.v()(j, d);
        }
      }
    }
    return matrix<T> {a.cols(), b.cols(), std::move(x)};
  }

  /*!
   * \brief The numerical rank of \p a, as `numpy.linalg.matrix_rank` counts it: its singular values above
   *        \ref detail::rank_tolerance of the largest.
   * \throws non_finite_error If an entry of \p a is infinite or NaN.
   */
  template <typename T>
  [[nodiscard]] std::size_t matrix_rank(const matrix<T>& a)
  {
    detail::require_finite(a, "matrix_rank");
    return detail::numerical_rank(svd_decomposition<T> {a}.singular_values(), a.rows(), a.cols());
  }

  /*!
   * \brief The Moore–Penrose pseudo-inverse of the `m × n` \p a (`n × m`), as `numpy.linalg.pinv` gives it:
   *        `V Σ⁺ Uᴴ`, a singular value under \ref detail::rank_tolerance of the largest counting as zero.
   * \throws non_finite_error If an entry of \p a is infinite or NaN.
   */
  template <typename T>
  [[nodiscard]] matrix<T> pinv(const matrix<T>& a)
  {
    detail::require_finite(a, "pinv");
    const svd_decomposition<T> s                          {a};
    const std::size_t          rank                       {detail::numerical_rank(s.singular_values(), a.rows(), a.cols())};
    std::vector<T>             out(a.cols() * a.rows(), T {});
    for (std::size_t d {0}; d < rank; ++d) {
      const auto inverse {static_cast<T>(1.0 / s.singular_values()[d])};
      for (std::size_t i {0}; i < a.cols(); ++i) {
        const T left {s.v()(i, d) * inverse};
        for (std::size_t j {0}; j < a.rows(); ++j) {
          out[(i * a.rows()) + j] += left * detail::conj_of(s.u()(j, d));
        }
      }
    }
    return matrix<T> {a.cols(), a.rows(), std::move(out)};
  }
} // namespace numerical

#endif // NUMERICAL_DECOMPOSITION_HPP
