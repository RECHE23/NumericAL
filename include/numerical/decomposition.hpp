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
#include <array>
#include <cmath>
#include <complex> // so std::abs resolves for a complex scalar, regardless of include order
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "matrix.hpp"
#include "simd.hpp"
#include "parallel.hpp"
#include "vector.hpp"

namespace numerical {

  namespace detail {

    //! \brief The columns of one panel of the blocked LU: the trailing update, a product of this depth, is
    //!        deep enough for the packed kernel while the unblocked panel stays a small part of the work.
    inline constexpr std::size_t lu_block {64};

    //! \brief The rows of one leaf of the triangular solves against many right-hand sides (\ref lu_block's), and
    //!        the columns of one leaf of the recursive LU panel.
    inline constexpr std::size_t solve_leaf {16};

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

    //! \brief The type `std::abs` gives for a `T`: the real type of a complex one, `T` itself otherwise.
    template <typename T>
    using magnitude_t = decltype(std::abs(std::declval<const T&>()));

    //! \brief \p x, a magnitude carried in `double`, as the scalar `T`: converted to `T`'s real type first, so a
    //!        `std::complex<float>` is built from a `float` (a `double` argument narrows inside its constructor).
    template <typename T>
    [[nodiscard]] T from_double(double x)
    {
      return static_cast<T>(static_cast<magnitude_t<T>>(x));
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
      return static_cast<double>(x.real());
    }

    //! \brief The unit-modulus factor of \p x: `x / |x|` (its sign, for a real scalar), and 1 at 0.
    template <typename T>
    [[nodiscard]] T phase_of(const T& x)
    {
      const double magnitude {static_cast<double>(std::abs(x))};
      return magnitude == 0.0 ? T {1} : x / detail::from_double<T>(magnitude);
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
      T value {detail::from_double<T>(sign_)};
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
     * \brief `L U X = Y` in place, \p y being `n × cols`, row-major, already permuted. The rows go by blocks of
     *        \ref detail::lu_block, the blocks by leaves of \ref detail::solve_leaf rows. Within a leaf the rows are
     *        solved one after another, in parallel chunks of columns; from a leaf to the rest of its block, and
     *        from a block to the rows beyond it, through the matrix product kernel.
     */
    void solve_rows(std::vector<T>& y,
                    std::size_t     cols) const
    {
      constexpr std::size_t width  {64};
      const std::size_t     chunks {(cols + width - 1) / width};
      for (std::size_t k0 {0}; k0 < n_; k0 += detail::lu_block) { // forward: unit-lower L
        const std::size_t end {std::min(k0 + detail::lu_block, n_)};
        for (std::size_t s0 {k0}; s0 < end; s0 += detail::solve_leaf) {
          const std::size_t s1 {std::min(s0 + detail::solve_leaf, end)};
          detail::parallel_for(chunks, (s1 - s0) * (s1 - s0) * width / 2, [&](std::size_t chunk) {
                                 const std::size_t first {chunk * width};
                                 const std::size_t last {std::min(first + width, cols)};
                                 for (std::size_t i {s0 + 1}; i < s1; ++i) {
                                   for (std::size_t j {s0}; j < i; ++j) {
                                     detail::subtract_product(y.data() + (i * cols) + first, lu_[(i * n_) + j], y.data() + (j * cols) + first, last - first);
                                   }
                                 }
                               });
          subtract_block(y, cols, s1, end, s0, s1);
        }
        subtract_block(y, cols, end, n_, k0, end);
      }
      for (std::size_t k1 {n_}; k1 > 0;) { // backward: upper U, the blocks and leaves of the forward pass, last first
        const std::size_t k0 {(k1 - 1) / detail::lu_block * detail::lu_block};
        for (std::size_t s1 {k1}; s1 > k0;) {
          const std::size_t s0 {k0 + ((s1 - 1 - k0) / detail::solve_leaf * detail::solve_leaf)};
          detail::parallel_for(chunks, (s1 - s0) * (s1 - s0) * width / 2, [&](std::size_t chunk) {
                                 const std::size_t first {chunk * width};
                                 const std::size_t last {std::min(first + width, cols)};
                                 for (std::size_t i {s1}; i-- > s0;) {
                                   for (std::size_t j {i + 1}; j < s1; ++j) {
                                     detail::subtract_product(y.data() + (i * cols) + first, lu_[(i * n_) + j], y.data() + (j * cols) + first, last - first);
                                   }
                                   const T pivot {lu_[(i * n_) + i]};
                                   for (std::size_t c {first}; c < last; ++c) {
                                     y[(i * cols) + c] = y[(i * cols) + c] / pivot;
                                   }
                                 }
                               });
          subtract_block(y, cols, k0, s0, s0, s1);
          s1 = s0;
        }
        subtract_block(y, cols, 0, k0, k0, k1);
        k1 = k0;
      }
    }

    //! \brief `Y[r0, r1) −= F[r0, r1) × [c0, c1) · Y[c0, c1)`, F the factor, through the matrix product kernel (nothing
    //!        when the rows are empty).
    void subtract_block(std::vector<T>& y,
                        std::size_t     cols,
                        std::size_t     r0,
                        std::size_t     r1,
                        std::size_t     c0,
                        std::size_t     c1) const
    {
      if (r0 < r1) {
        const std::vector<T> minus {negated_block(r0, r1, c0, c1)};
        detail::gemm(r1 - r0, cols, c1 - c0, minus.data(), c1 - c0, y.data() + (c0 * cols), cols, y.data() + (r0 * cols),
                     cols);
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
      factorize_columns(panel, rows, 0, width, k0, end);
      for (std::size_t r {0}; r < rows; ++r) {
        for (std::size_t j {0}; j < width; ++j) {
          lu_[((k0 + r) * n_) + k0 + j] = panel[(j * rows) + r];
        }
      }
    }

    /*!
     * \brief Factorises the columns `[first, last)` of the panel (column-major, \p rows rows), every row from
     *        `first` down, each pivot's row swap applied to the whole panel and outside it.
     *
     * Recursively, as LAPACK's dgetrf2: the left half is factorised, the top of the right half solved against its
     * unit-lower L, the bottom updated by a product through the matrix product kernel (in parallel, where the
     * work is), then the right half factorised; a leaf of \ref detail::solve_leaf columns, unblocked. Each element
     * still receives its updates in ascending order of the pivot.
     */
    void factorize_columns(std::vector<T>& panel,
                           std::size_t     rows,
                           std::size_t     first,
                           std::size_t     last,
                           std::size_t     k0,
                           std::size_t     end)
    {
      // A leaf, or a panel too short for its products to pay for packing (order 64 on arm64: 30 µs unblocked,
      // 37 recursive).
      if (last - first <= detail::solve_leaf || rows - first < 2 * detail::lu_block) {
        for (std::size_t j {first}; j < last; ++j) {
          T           *          column {panel.data() + (j * rows)};
          std::size_t            pivot  {j};
          detail::magnitude_t<T> best   {std::abs(column[j])};
          for (std::size_t r {j + 1}; r < rows; ++r) {
            const detail::magnitude_t<T> mag {std::abs(column[r])};
            if (mag > best) {
              best  = mag;
              pivot = r;
            }
          }
          if (best == detail::magnitude_t<T> {}) {
            singular_ = true;
            continue; // a zero pivot column: leave it, the determinant becomes 0
          }
          if (pivot != j) {
            for (std::size_t c {0}; c < end - k0; ++c) { // the whole panel row
              std::swap(panel[(c * rows) + j], panel[(c * rows) + pivot]);
            }
            swap_rows_outside(k0 + j, k0 + pivot, k0, end);
            std::swap(pivots_[k0 + j], pivots_[k0 + pivot]);
            sign_ = -sign_;
          }
          for (std::size_t r {j + 1}; r < rows; ++r) {
            column[r] = column[r] / column[j]; // the multiplier, in L's slot
          }
          for (std::size_t c {j + 1}; c < last; ++c) {
            T*      target {panel.data() + (c * rows)};
            detail::subtract_product(target + j + 1, column + j + 1, target[j], rows - (j + 1));
          }
        }
        return;
      }
      const std::size_t mid {first + ((last - first) / 2)};
      factorize_columns(panel, rows, first, mid, k0, end);
      for (std::size_t c {mid}; c < last; ++c) { // the top of the right half: rows [first, mid)
        T* target {panel.data() + (c * rows)};
        for (std::size_t j {first}; j < mid; ++j) {
          detail::subtract_product(target + j + 1, panel.data() + (j * rows) + j + 1, target[j], mid - (j + 1));
        }
      }
      // The bottom, rows [mid, rows): A₂₂ −= L₂₁ U₁₂, column-major, so as its transpose, row-major:
      // A₂₂ᵀ += (−U₁₂ᵀ) L₂₁ᵀ.
      const std::size_t depth {mid - first};
      std::vector<T>    minus_u((last - mid) * depth);
      for (std::size_t c {mid}; c < last; ++c) {
        for (std::size_t j {first}; j < mid; ++j) {
          minus_u[((c - mid) * depth) + (j - first)] = -panel[(c * rows) + j];
        }
      }
      detail::gemm(last - mid, rows - mid, depth, minus_u.data(), depth, panel.data() + (first * rows) + mid, rows,
                   panel.data() + (mid * rows) + mid, rows);
      factorize_columns(panel, rows, mid, last, k0, end);
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

    //! \brief `U₁₂ = L₁₁⁻¹ A₁₂`: the panel's rows right of it, by forward substitution with its unit-lower L, by
    //!        leaves of \ref detail::solve_leaf rows. Within a leaf, each block of columns is independent, so blocks
    //!        run in parallel, each in the sequential order; from a leaf to the panel's rows below it, through the
    //!        matrix product kernel. Each element still takes its terms in ascending order of the pivot.
    void solve_panel_rows(std::size_t k0,
                          std::size_t end)
    {
      constexpr std::size_t width  {64};
      const std::size_t     blocks {((n_ - end) + width - 1) / width};
      for (std::size_t s0 {k0}; s0 < end; s0 += detail::solve_leaf) {
        const std::size_t s1 {std::min(s0 + detail::solve_leaf, end)};
        detail::parallel_for(blocks, (s1 - s0) * (s1 - s0) * width / 2, [&](std::size_t block) {
                               const std::size_t first {end + (block * width)};
                               const std::size_t last {std::min(first + width, n_)};
                               for (std::size_t i {s0 + 1}; i < s1; ++i) {
                                 for (std::size_t p {s0}; p < i; ++p) {
                                   detail::subtract_product(lu_.data() + (i * n_) + first, lu_[(i * n_) + p], lu_.data() + (p * n_) + first, last - first);
                                 }
                               }
                             });
        if (s1 < end) {
          const std::vector<T> minus {negated_block(s1, end, s0, s1)};
          detail::gemm(end - s1, n_ - end, s1 - s0, minus.data(), s1 - s0, lu_.data() + (s0 * n_) + end, n_,
                       lu_.data() + (s1 * n_) + end, n_);
        }
      }
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
    // by leaves (each depends only on the diagonal block; factor_rows_below), then the lower part of the trailing block takes
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
          factor_rows_below(k0, end);
          update_trailing(k0, end);
        }
      }
      for (std::size_t i {0}; i < n_; ++i) {
        std::fill(lower_.begin() + static_cast<std::ptrdiff_t>((i * n_) + i + 1),
                  lower_.begin() + static_cast<std::ptrdiff_t>((i + 1) * n_), T {});
      }
    }

    /*!
     * \brief Row \p i of L in the columns `[k0, last)`, the terms of the columns before \p k0 already subtracted: the
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
          lower_[(i * n_) + i] = detail::from_double<T>(std::sqrt(diagonal));
        }
        else {
          lower_[(i * n_) + j] = sum / lower_[(j * n_) + j];
        }
      }
    }

    /*!
     * \brief The rows below the panel `[k0, end)`, by leaves of \ref detail::solve_leaf columns (fewer than eight
     *        blocks of rows) or of the whole panel: a leaf's
     *        terms from the panel's earlier columns, `L[i][j] −= Σₖ L[i][k]·conj(L[j][k])`, through the matrix
     *        product kernel, then the leaf row by row in parallel (\ref factor_row). Each element still takes its
     *        terms in ascending order.
     */
    void factor_rows_below(std::size_t k0,
                           std::size_t end)
    {
      const std::size_t rows {n_ - end};
      // Under many rows the leaves' products stream the panel again and again, which the row-by-row step does not
      // (order 1024 on arm64: 13.4 ms row by row, 15.1 by leaves); there, one leaf of the whole panel.
      const std::size_t leaf {rows < 8 * detail::cholesky_block ? detail::solve_leaf : end - k0};
      for (std::size_t s0 {k0}; s0 < end; s0 += leaf) {
        const std::size_t s1 {std::min(s0 + leaf, end)};
        if (s0 > k0) {
          const std::size_t depth {s0 - k0};
          std::vector<T>    minus_adjoint(depth * (s1 - s0)); // −conj(L[j][k]), k by row, j by column
          for (std::size_t k {0}; k < depth; ++k) {
            for (std::size_t j {s0}; j < s1; ++j) {
              minus_adjoint[(k * (s1 - s0)) + (j - s0)] = -detail::conj_of(lower_[(j * n_) + k0 + k]);
            }
          }
          detail::gemm(rows, s1 - s0, depth, lower_.data() + (end * n_) + k0, n_, minus_adjoint.data(), s1 - s0,
                       lower_.data() + (end * n_) + s0, n_);
        }
        detail::parallel_for(rows, (s1 - s0) * (s1 - s0) / 2, [&](std::size_t r) { factor_row(end + r, s0, s1); });
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
     * `−2 vᵢ w`. A column-by-column walk strides by `cols` on every access and runs several times slower. The
     * columns are independent, so chunks of them run in parallel, each element's arithmetic in the same order.
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
      constexpr std::size_t width {64};
      const std::size_t     span  {cols - from};
      parallel_for((span + width - 1) / width, 2 * v.size() * width, [&](std::size_t chunk) {
                     const std::size_t start {from + (chunk * width)};
                     const std::size_t count {std::min(width, cols - start)};
                     std::vector<T>    w(count, T {});
                     for (std::size_t i {0}; i < v.size(); ++i) {
                       const T  weight {conj_of(v[i])};
                       const T* row    {a.data() + ((first + i) * cols) + start};
                       add_product(w.data(), weight, row, count);
                     }
                     for (std::size_t i {0}; i < v.size(); ++i) {
                       const T scale {static_cast<T>(2) * v[i]};
                       T*      row   {a.data() + ((first + i) * cols) + start};
                       subtract_product(row, scale, w.data(), count);
                     }
                   });
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
    //! \brief The reflectors of one panel of the blocked QR, `I − V T Vᴴ` as LAPACK's `dlarft` forms it.
    template <typename T>
    struct reflector_block
    {
      std::size_t    first;  //!< The first row the block acts on.
      std::size_t    height; //!< The rows it acts on, `first…first+height-1`.
      std::size_t    width;  //!< Its reflector count.
      std::vector<T> v;      //!< `V`, `height × width` row-major, reflector j from row j (zeros above).
      std::vector<T> t;      //!< `T`, `width × width` upper triangular.
    };

    /*!
     * \brief Gathers the unit reflectors `reflectors[k0…k1-1]` (each `I − 2 v vᴴ`, an empty one the
     *        identity) into `I − V T Vᴴ = H_{k0} H_{k0+1} … H_{k1-1}`.
     */
    template <typename T>
    [[nodiscard]] reflector_block<T> gather_reflectors(const std::vector<std::vector<T>>& reflectors,
                                                       std::size_t                        k0,
                                                       std::size_t                        k1,
                                                       std::size_t                        rows)
    {
      reflector_block<T> block {.first = k0, .height = rows - k0, .width = k1 - k0, .v = {}, .t = {}};
      const std::size_t  b     {block.width};
      block.v.assign(block.height * b, T {});
      block.t.assign(b * b, T {});
      for (std::size_t j {0}; j < b; ++j) {
        const std::vector<T>& v {reflectors[k0 + j]};
        for (std::size_t r {0}; r < v.size(); ++r) {
          block.v[((j + r) * b) + j] = v[r];
        }
      }
      // Vᴴ V once, through the matrix product kernel: its column j above the diagonal is V(:, 0…j-1)ᴴ v_j.
      std::vector<T> v_adjoint(b * block.height);
      for (std::size_t r {0}; r < block.height; ++r) {
        for (std::size_t j {0}; j < b; ++j) {
          v_adjoint[(j * block.height) + r] = conj_of(block.v[(r * b) + j]);
        }
      }
      std::vector<T> gram(b * b, T {});
      gemm(b, b, block.height, v_adjoint.data(), block.height, block.v.data(), b, gram.data(), b);
      for (std::size_t j {0}; j < b; ++j) {
        if (reflectors[k0 + j].empty()) {
          continue;                           // the identity: a zero column of T
        }
        for (std::size_t i {0}; i < j; ++i) { // T(0…j-1, j) = −2 T(0…j-1, 0…j-1) z, z = V(:, 0…j-1)ᴴ v_j
          T sum {};
          for (std::size_t p {i}; p < j; ++p) {
            sum += block.t[(i * b) + p] * gram[(p * b) + j];
          }
          block.t[(i * b) + j] = static_cast<T>(-2) * sum;
        }
        block.t[(j * b) + j] = static_cast<T>(2);
      }
      return block;
    }

    /*!
     * \brief Applies `I − V T Vᴴ` (or its adjoint `I − V Tᴴ Vᴴ`) from the left to the block's rows of a
     *        row-major array, on its columns `from…to-1`: `W = Vᴴ A`, then `W ← T W` or `Tᴴ W`, then
     *        `A −= V W`, the two outer products through the matrix product kernel.
     */
    template <typename T>
    void apply_reflector_block(const reflector_block<T>& block,
                               bool                      adjoint,
                               std::vector<T>&           a,
                               std::size_t               cols,
                               std::size_t               from,
                               std::size_t               to)
    {
      const std::size_t b {block.width};
      const std::size_t h {block.height};
      const std::size_t w {to - from};
      std::vector<T>    v_adjoint(b * h);
      std::vector<T>    minus_v(h * b);
      for (std::size_t r {0}; r < h; ++r) {
        for (std::size_t j {0}; j < b; ++j) {
          v_adjoint[(j * h) + r]  = conj_of(block.v[(r * b) + j]);
          minus_v[(r * b) + j]    = -block.v[(r * b) + j];
        }
      }
      T*             top                 {a.data() + (block.first * cols) + from};
      std::vector<T> projection(b * w, T {});
      gemm(b, w, h, v_adjoint.data(), h, top, cols, projection.data(), w);
      std::vector<T> scaled(b * w, T     {});
      for (std::size_t i {0}; i < b; ++i) {
        for (std::size_t p {0}; p < b; ++p) {
          const T factor {adjoint ? conj_of(block.t[(p * b) + i]) : block.t[(i * b) + p]};
          if (factor == T {}) {
            continue; // T is triangular: half of these are zero
          }
          add_product(scaled.data() + (i * w), factor, projection.data() + (p * w), w);
        }
      }
      gemm(h, w, b, minus_v.data(), b, scaled.data(), w, top, cols);
    }

    /*!
     * \brief Applies `I − 2 v vᴴ` from the left to the rows `first…first+|v|-1`, on the columns `from…to-1`
     *        only: \ref reflect_rows confined to one panel.
     */
    template <typename T>
    void reflect_panel(std::vector<T>&       a,
                       std::size_t           cols,
                       const std::vector<T>& v,
                       std::size_t           first,
                       std::size_t           from,
                       std::size_t           to)
    {
      std::vector<T> w(to - from, T {});
      for (std::size_t i {0}; i < v.size(); ++i) {
        const T  weight {conj_of(v[i])};
        const T* row    {a.data() + ((first + i) * cols) + from};
        add_product(w.data(), weight, row, w.size());
      }
      for (std::size_t i {0}; i < v.size(); ++i) {
        const T scale {static_cast<T>(2) * v[i]};
        T*      row   {a.data() + ((first + i) * cols) + from};
        subtract_product(row, scale, w.data(), w.size());
      }
    }

    //! \brief The reflectors of one panel of the blocked QR.
    inline constexpr std::size_t qr_block {32};

    /*!
     * \brief Factorises the `rows × cols` row-major \p a by Householder reflections, `A = Q R` with
     *        `k = min(rows, cols)`: `Q` is `rows × k` with orthonormal columns and `R` `k × cols` upper
     *        trapezoidal. Each column below the diagonal is reflected onto it, so `Q` is orthonormal to
     *        rounding whatever the conditioning of `A`, and a column dependent on the earlier ones leaves a
     *        zero on `R`'s diagonal.
     *
     * Blocked as LAPACK's `dgeqrf`: the reflectors of a panel of \ref qr_block columns act on the panel
     * one by one, then on the columns right of it at once, as `I − V T Vᴴ` through the matrix product
     * kernel; `Q` is formed block by block the same way.
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
      const std::size_t               depth {std::min(rows, cols)};
      std::vector<std::vector<T>>     reflectors(depth); // v_k on rows k…m-1; empty where the column was zero
      std::vector<reflector_block<T>> blocks;
      for (std::size_t k0 {0}; k0 < depth; k0 += qr_block) {
        const std::size_t k1 {std::min(k0 + qr_block, depth)};
        for (std::size_t k {k0}; k < k1; ++k) {
          std::vector<T> v(rows - k);
          for (std::size_t i {k}; i < rows; ++i) {
            v[i - k] = a[(i * cols) + k];
          }
          const double norm {scaled_norm(v)};
          if (norm == 0.0) {
            continue; // nothing to reflect: the column is already zero below and on the diagonal
          }
          v[0] += phase_of(v[0]) * detail::from_double<T>(norm); // x − α e₁ with α = −phase(x₀)‖x‖: no cancellation
          const auto length {detail::from_double<T>(scaled_norm(v))};
          for (T& element : v) {
            element /= length;
          }
          reflect_panel(a, cols, v, k, k, k1); // the panel: A ← (I − 2 v vᴴ) A, columns k…k1-1
          reflectors[k] = std::move(v);
        }
        blocks.push_back(gather_reflectors(reflectors, k0, k1, rows));
        if (k1 < cols) { // right of the panel: H_{k1-1} … H_{k0} A = (I − V T Vᴴ)ᴴ A
          apply_reflector_block(blocks.back(), true, a, cols, k1, cols);
        }
      }
      thin_qr<T> out {.q = std::vector<T>(rows * depth, T {}), .r = std::vector<T>(depth * cols, T {})};
      for (std::size_t i {0}; i < depth; ++i) {
        out.q[(i * depth) + i] = T {1};
        for (std::size_t j {i}; j < cols; ++j) {
          out.r[(i * cols) + j] = a[(i * cols) + j];
        }
      }
      for (std::size_t b {blocks.size()}; b-- > 0;) { // Q = H₀ H₁ … applied to the first columns of I
        apply_reflector_block(blocks[b], false, out.q, depth, blocks[b].first, depth);
      }
      for (std::size_t k {0}; k < depth; ++k) {       // R's diagonal real and non-negative: rotate row k, column k
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

    //! \brief `y[i] += conj(x[i]) · f` for a real `T`, where the conjugate is the value: \ref add_product.
    template <typename T>
    requires std::is_floating_point_v<T>
    void add_conjugate_product(T*          y,
                               const T*    x,
                               const T&    f,
                               std::size_t count)
    {
      add_product(y, f, x, count);
    }

    //! \brief `y[i] += conj(x[i]) · f`.
    template <typename T>
    void add_conjugate_product(T*          y,
                               const T*    x,
                               const T&    f,
                               std::size_t count)
    {
      for (std::size_t i {0}; i < count; ++i) {
        y[i] += conj_of(x[i]) * f;
      }
    }

    //! \brief The reflections of one panel of the blocked tridiagonal reduction.
    inline constexpr std::size_t tridiagonal_block {32};

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
     * Each reflection zeroes one column below the subdiagonal; the trailing block takes the rank-2 updates
     * `B − v wᴴ − w vᴴ` of a panel of \ref tridiagonal_block reflections at once, through the matrix product
     * kernel, as LAPACK's `dsytrd`; `M` is then formed by blocks of reflections the same way, as `dorgtr`. A complex subdiagonal entry is then made real by a
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
      std::vector<std::vector<T>> reflectors(n); // reflectors[k + 1] zeroes column k, acting on rows k + 1…
      std::vector<T>              sub(n, T {});  // T's subdiagonal before the phases
      for (std::size_t k0 {0}; k0 + 1 < n; k0 += tridiagonal_block) {
        // A panel of reflections, as LAPACK's dlatrd: the matrix the panel's later steps see is
        // A − V Wᴴ − W Vᴴ, applied to the one column and the one product each step needs; the whole trailing
        // block takes it after the panel, through the matrix product kernel.
        const std::size_t k1                 {std::min(k0 + tridiagonal_block, n - 1)};
        const std::size_t width              {k1 - k0};
        std::vector<T>    big_v(n * width, T {}); // V: row i, column p is v of step k0 + p at row i (zero above)
        std::vector<T>    big_w(n * width, T {}); // W, likewise
        for (std::size_t k {k0}; k < k1; ++k) {
          const std::size_t j {k - k0};
          const std::size_t m {n - k - 1};
          std::vector<T>    column(m + 1); // column k from the diagonal down, the panel's earlier steps applied
          for (std::size_t i {k}; i < n; ++i) {
            T value {a[(i * n) + k]};
            for (std::size_t p {0}; p < j; ++p) {
              value -= (big_v[(i * width) + p] * conj_of(big_w[(k * width) + p]))
                       + (big_w[(i * width) + p] * conj_of(big_v[(k * width) + p]));
            }
            column[i - k] = value;
          }
          a[(k * n) + k] = column[0];
          std::vector<T> v(column.begin() + 1, column.end()); // column k below the diagonal
          const double   below {m > 1 ? scaled_norm(std::vector<T>(v.begin() + 1, v.end())) : 0.0};
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
          const T       *       block       {a.data() + ((k + 1) * n) + k + 1}; // the trailing block, row stride n
          std::vector<T>        w(m, T      {});
          constexpr std::size_t chunk_width {64};
          // w = A v on the trailing block, as Σₗ conj(row l) vₗ (A is Hermitian): contiguous updates rather than
          // a dependent sum per row, in parallel by chunks of w.
          parallel_for((m + chunk_width - 1) / chunk_width, m * chunk_width, [&](std::size_t chunk) {
                         const std::size_t first {chunk * chunk_width};
                         const std::size_t count {std::min(chunk_width, m - first)};
                         for (std::size_t l {0}; l < m; ++l) {
                           add_conjugate_product(w.data() + first, block + (l * n) + first, v[l], count);
                         }
                       });
          std::vector<T> by_w(j, T {}); // Wᴴ v and Vᴴ v over the earlier steps
          std::vector<T> by_v(j, T {});
          for (std::size_t l {0}; l < m; ++l) {
            for (std::size_t p {0}; p < j; ++p) {
              by_w[p] += conj_of(big_w[((k + 1 + l) * width) + p]) * v[l];
              by_v[p] += conj_of(big_v[((k + 1 + l) * width) + p]) * v[l];
            }
          }
          for (std::size_t i {0}; i < m; ++i) { // w = (A − V Wᴴ − W Vᴴ) v
            for (std::size_t p {0}; p < j; ++p) {
              w[i] -= (big_v[((k + 1 + i) * width) + p] * by_w[p]) + (big_w[((k + 1 + i) * width) + p] * by_v[p]);
            }
          }
          T c {};
          for (std::size_t i {0}; i < m; ++i) {
            c += conj_of(v[i]) * w[i];
          }
          for (std::size_t i {0}; i < m; ++i) { // w = 2 B v − 2 (vᴴ B v) v, so H B H = B − v wᴴ − w vᴴ
            w[i]                             = (static_cast<T>(2) * w[i]) - (static_cast<T>(2) * c * v[i]);
            big_v[((k + 1 + i) * width) + j] = v[i];
            big_w[((k + 1 + i) * width) + j] = w[i];
          }
          reflectors[k + 1] = std::move(v); // acting from row k + 1, as householder_qr's from row k
        }
        // The trailing block from row and column k1: A −= V Wᴴ + W Vᴴ.
        const std::size_t rest {n - k1};
        std::vector<T>    minus_v(rest * width);
        std::vector<T>    minus_w(rest * width);
        std::vector<T>    v_adjoint(width * rest);
        std::vector<T>    w_adjoint(width * rest);
        for (std::size_t i {0}; i < rest; ++i) {
          for (std::size_t p {0}; p < width; ++p) {
            minus_v[(i * width) + p]  = -big_v[((k1 + i) * width) + p];
            minus_w[(i * width) + p]  = -big_w[((k1 + i) * width) + p];
            v_adjoint[(p * rest) + i] = conj_of(big_v[((k1 + i) * width) + p]);
            w_adjoint[(p * rest) + i] = conj_of(big_w[((k1 + i) * width) + p]);
          }
        }
        T* trailing {a.data() + (k1 * n) + k1};
        gemm(rest, rest, width, minus_v.data(), width, w_adjoint.data(), rest, trailing, n);
        gemm(rest, rest, width, minus_w.data(), width, v_adjoint.data(), rest, trailing, n);
      }
      for (std::size_t i {0}; i < n; ++i) {
        d[i] = real_part(a[(i * n) + i]);
      }
      e[n - 1] = 0.0;
      std::vector<T> q(n * n, T {}); // Q = H₁ H₂ …, by blocks of reflectors from the last back, as dorgtr
      for (std::size_t i {0}; i < n; ++i) {
        q[(i * n) + i] = T {1};
      }
      for (std::size_t k1 {n}; k1 > 1;) {
        const std::size_t k0 {k1 - std::min(k1 - 1, qr_block)};
        apply_reflector_block(gather_reflectors(reflectors, k0, k1, n), false, q, n, k0, n);
        k1 = k0;
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

    //! \brief One operation of \ref tridiagonal_ql or \ref bidiagonal_qr on the rows of a vector matrix,
    //!        recorded to be applied later.
    struct row_operation
    {
      enum class kind : std::uint8_t
      {
        rotate, //!< `row_i ← c row_i + s row_j`, `row_j ← c row_j − s row_i`
        negate, //!< row `i` ← −row `i`
        swap,   //!< rows `i` and `j` exchanged
      };

      kind        what; //!< The operation.
      std::size_t i;    //!< The first row.
      std::size_t j;    //!< The second row (unused by `negate`).
      double      c;    //!< The cosine of a rotation.
      double      s;    //!< The sine of a rotation.
    };

    //! \brief A run of recorded rotations in which each shares a row with the next, as a sweep of the QL or QR
    //!        iteration gives them: rows `(i, i+1)` for `i` going down (`down`), or going up.
    struct rotation_run
    {
      std::size_t first; //!< Its first operation.
      std::size_t count; //!< Its operations (one for an operation outside any run).
      bool        down;  //!< Whether each next rotation's second row is this one's first.
    };

    //! \brief The operations split into runs of chained rotations, every other operation a run of one.
    [[nodiscard]] inline std::vector<rotation_run> rotation_runs(const std::vector<row_operation>& operations)
    {
      std::vector<rotation_run> runs;
      for (std::size_t t {0}; t < operations.size();) {
        const row_operation& op    {operations[t]};
        std::size_t          count {1};
        bool                 down  {false};
        if (op.what == row_operation::kind::rotate && t + 1 < operations.size()
            && operations[t + 1].what == row_operation::kind::rotate) {
          down = operations[t + 1].j == op.i;
          for (; t + count < operations.size(); ++count) {
            const row_operation& previous {operations[t + count - 1]};
            const row_operation& next     {operations[t + count]};
            const bool           chained  {down ? next.j == previous.i : next.i == previous.j};
            if (next.what != row_operation::kind::rotate || !chained) {
              break;
            }
          }
        }
        runs.push_back({.first = t, .count = count, .down = down});
        t                     += count;
      }
      return runs;
    }

    /*!
     * \brief Applies the recorded operations, in order, to the row-major \p m (`n` per row). Each column takes
     *        them independently of the others, so chunks of columns run in parallel, every element receiving
     *        the same operations in the same order as applying each in place would give it.
     *
     * A run of chained rotations goes as LAPACK's `dlasr`: the row each rotation hands the next stays in a
     * buffer the size of the chunk, so every row of the run is read and written once rather than twice per
     * rotation.
     */
    template <typename T>
    void apply_row_operations(std::vector<T>&                   m,
                              std::size_t                       n,
                              const std::vector<row_operation>& operations)
    {
      constexpr std::size_t           width {64};
      const std::vector<rotation_run> runs  {rotation_runs(operations)};
      parallel_for((n + width - 1) / width, operations.size() * width, [&](std::size_t chunk) {
                     const std::size_t     first {chunk * width};
                     const std::size_t     count {std::min(first + width, n) - first};
                     std::array<T, width>  carry {};
                     for (const rotation_run& run : runs) {
                       const row_operation& start {operations[run.first]};
                       if (run.count > 1) {
                         // Down: rotation u's first row is rotation u+1's second; up: its second is the next's first.
                         const T* source {m.data() + ((run.down ? start.j : start.i) * n) + first};
                         std::copy(source, source + count, carry.begin());
                         for (std::size_t u {run.first}; u < run.first + run.count; ++u) {
                           const row_operation& op {operations[u]};
                           const T              c  {detail::from_double<T>(op.c)};
                           const T              s  {detail::from_double<T>(op.s)};
                           if (run.down) { // the carried row is op.j: op.i is read, op.j written
                             const T* row   {m.data() + (op.i * n) + first};
                             T*       other {m.data() + (op.j * n) + first};
                             for (std::size_t k {0}; k < count; ++k) {
                               const T a {row[k]};
                               const T b {carry[k]};
                               other[k] = (c * b) - (s * a);
                               carry[k] = (c * a) + (s * b);
                             }
                           }
                           else { // the carried row is op.i: op.j is read, op.i written
                             T*       row   {m.data() + (op.i * n) + first};
                             const T* other {m.data() + (op.j * n) + first};
                             for (std::size_t k {0}; k < count; ++k) {
                               const T a {carry[k]};
                               const T b {other[k]};
                               row[k]   = (c * a) + (s * b);
                               carry[k] = (c * b) - (s * a);
                             }
                           }
                         }
                         const row_operation& end    {operations[run.first + run.count - 1]};
                         T*                   target {m.data() + ((run.down ? end.i : end.j) * n) + first};
                         std::copy(carry.begin(), carry.begin() + static_cast<std::ptrdiff_t>(count), target);
                         continue;
                       }
                       T* row   {m.data() + (start.i * n) + first};
                       T* other {m.data() + (start.j * n) + first};
                       // The kind is tested outside the column loop, which then vectorises.
                       if (start.what == row_operation::kind::rotate) {
                         const T c {detail::from_double<T>(start.c)};
                         const T s {detail::from_double<T>(start.s)};
                         for (std::size_t k {0}; k < count; ++k) {
                           const T a {row[k]};
                           const T b {other[k]};
                           row[k]   = (c * a) + (s * b);
                           other[k] = (c * b) - (s * a);
                         }
                       }
                       else if (start.what == row_operation::kind::negate) {
                         for (std::size_t k {0}; k < count; ++k) {
                           row[k] = -row[k];
                         }
                       }
                       else {
                         std::swap_ranges(row, row + count, other);
                       }
                     }
                   });
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
      // The iteration reads only d and e: its rotations of zt's rows are recorded and applied once at the
      // end, in parallel by columns.
      std::vector<row_operation> on_zt;
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
            // rows i and i+1 (columns i and i+1 of Z): upper ← c upper − s lower, lower ← c lower + s upper
            on_zt.push_back({row_operation::kind::rotate, i, i + 1, c, -s});
          }
          p    = -s * s2 * c3 * el1 * e[l] / dl1;
          e[l] = s * p;
          d[l] = c * p;
        }
        d[l] += shift;
        e[l]  = 0.0;
      }
      apply_row_operations(zt, n, on_zt);
    }

    //! \brief The order up to which \ref tridiagonal_divide_and_conquer diagonalises by implicit QL.
    inline constexpr std::size_t divide_leaf {32};

    //! \brief The iterations one root of the secular equation may take: it converges in a few; the bound turns
    //!        a non-convergence into an error instead of a hang.
    inline constexpr std::size_t secular_iterations {100};

    //! \brief A root of a secular equation: the pole it was sought from, and its offset from that pole.
    struct secular_solution
    {
      std::size_t origin; //!< The pole.
      double      offset; //!< The root minus the pole.
    };

    /*!
     * \brief Root \p i of `1/ρ + Σⱼ zⱼ²/(Pⱼ − x) = 0`, poles `P` strictly ascending, no `zⱼ` zero, `ρ > 0`, as
     *        LAPACK's `dlaed4` and `dlasd4` find it. The poles are given only as their offsets from one another,
     *        `offset(o, j) = Pⱼ − P_o`, which the caller computes accurately.
     *
     * The root lies in `(Pᵢ, Pᵢ₊₁)`, or for the last within `2ρ‖z‖²` past its pole. It is sought as an offset `τ`
     * from the nearer end `P_o`, so that every `Pⱼ − x = offset(o, j) − τ` is computed with a small relative error,
     * which keeps the vectors built from them orthogonal. Each step fits the equation with its two nearest poles
     * (Gragg's) and falls back to bisection when the fit leaves the bracket, which shrinks at every step.
     * \param[in]  z              The weights.
     * \param[in]  rho            `ρ`.
     * \param[in]  i              The root.
     * \param[in]  max_iterations The bound on the steps.
     * \param[out] delta          `Pⱼ − x` for every `j` (size `z.size()`).
     * \param[in]  offset         The pole offsets.
     * \return The pole and the root's offset from it.
     * \throws convergence_error If the root is not reached within \p max_iterations steps.
     */
    template <typename Offset>
    [[nodiscard]] secular_solution solve_secular(const std::vector<double>& z,
                                                 double                     rho,
                                                 std::size_t                i,
                                                 std::size_t                max_iterations,
                                                 double*                    delta,
                                                 const Offset&              offset)
    {
      const std::size_t k      {z.size()};
      const double      rhoinv {1.0 / rho};
      const bool        last   {i + 1 == k};
      double            weight {0.0};
      for (const double element : z) {
        weight += element * element;
      }
      std::size_t origin {i};
      double      lower  {0.0};
      double      upper  {2.0 * rho * weight}; // the last root is at most ρ‖z‖² past its pole: the equation is positive here
      if (!last) {
        const double gap {offset(i, i + 1)};
        double       f   {rhoinv};
        for (std::size_t j {0}; j < k; ++j) {
          f += z[j] * z[j] / (offset(i, j) - (gap / 2.0));
        }
        upper = gap / 2.0;
        if (f < 0.0) { // the root is nearer the pole past it
          origin = i + 1;
          lower  = -gap / 2.0;
          upper  = 0.0;
        }
      }
      double tau {(lower + upper) / 2.0};
      for (std::size_t iteration {0}; iteration < max_iterations; ++iteration) {
        double psi  {0.0}; // the terms of the poles up to i, and their derivative
        double dpsi {0.0};
        double phi  {0.0}; // those of the poles past i
        double dphi {0.0};
        for (std::size_t j {0}; j < k; ++j) {
          delta[j] = offset(origin, j) - tau;
          const double t {z[j] / delta[j]};
          (j <= i ? psi : phi)   += z[j] * t;
          (j <= i ? dpsi : dphi) += t * t;
        }
        const double w {rhoinv + psi + phi};
        if (std::abs(w) <= std::numeric_limits<double>::epsilon()
            * ((8.0 * (phi - psi + rhoinv)) + (3.0 * std::abs(tau) * (dpsi + dphi)))) {
          return {.origin = origin, .offset = tau};
        }
        (w < 0.0 ? lower : upper) = tau; // the equation increases with x
        // The fit c + s₁/(δᵢ − η) + s₂/(δᵢ₊₁ − η) matching the value and the slope of each side: C η² − A η + B = 0
        // (the last root has no pole to its right: its δᵢ₊₁ term is absent, A = δᵢ w − δᵢ² ψ' over C = w − δᵢ ψ').
        const double a1   {delta[i]};
        const double a2   {last ? 0.0 : delta[i + 1]};
        const double c    {w - (a1 * dpsi) - (a2 * dphi)};
        const double a    {last ? (a1 * c) + (a1 * a1 * dpsi) : ((a1 + a2) * w) - (a1 * a2 * (dpsi + dphi))};
        const double b    {last ? 0.0 : a1 * a2 * w};
        const double root {std::sqrt(std::abs((a * a) - (4.0 * b * c)))};
        // The smaller root of the quadratic, in the form that does not cancel.
        const double eta  {last ? a / c : (a <= 0.0 ? (a - root) / (2.0 * c) : (2.0 * b) / (a + root))};
        const double next {tau + eta};
        // A step out of the bracket (or not a number): bisect. The test above stops within a few units in the
        // last place of the root, before the bracket runs out of points.
        tau = next > lower && next < upper ? next : (lower + upper) / 2.0;
      }
      throw convergence_error("the secular equation did not converge");
    }

    /*!
     * \brief Root \p i of `1/ρ + Σⱼ zⱼ²/(dⱼ − λ) = 0` (`d` strictly ascending, `‖z‖ = 1`), the `i`-th eigenvalue
     *        of `diag(d) + ρ z zᵀ`, by \ref solve_secular.
     * \param[in]  d              The poles.
     * \param[in]  z              The weights.
     * \param[in]  rho            `ρ`.
     * \param[in]  i              The root.
     * \param[in]  max_iterations The bound on the steps.
     * \param[out] delta          `dⱼ − λ` for every `j`.
     * \return `λ`.
     */
    [[nodiscard]] inline double secular_root(const std::vector<double>& d,
                                             const std::vector<double>& z,
                                             double                     rho,
                                             std::size_t                i,
                                             std::size_t                max_iterations,
                                             double*                    delta)
    {
      const secular_solution root {solve_secular(z, rho, i, max_iterations, delta,
                                                 [&d](std::size_t o, std::size_t j) { return d[j] - d[o]; })};
      return d[root.origin] + root.offset;
    }

    /*!
     * \brief The eigenvalues of `diag(d) + ρ z zᵀ`, ascending, into \p d, and its eigenvectors times \p q (the
     *        row-major `n × n` whose columns span the halves) into \p q, as LAPACK's `dlaed1`.
     *
     * Deflation first (`dlaed2`): a negligible `zⱼ` leaves `dⱼ` and its column as they are, and two nearly equal
     * poles are rotated so that one of them does. The remaining `k` roots are found by \ref secular_root, in
     * parallel; `z` is then recomputed from them (Löwner's formula, Gu and Eisenstat), so that the eigenvectors
     * `(dⱼ − λᵢ)⁻¹ ẑⱼ` are orthogonal to working precision, and they reach `q` by one matrix product.
     */
    inline void merge_rank_one(std::vector<double>& d,
                               std::vector<double>  z,
                               double               rho,
                               std::vector<double>& q,
                               std::size_t          n)
    {
      for (double& element : z) { // ‖z‖ = √2: the rows of two orthogonal matrices
        element /= std::numbers::sqrt2;
      }
      rho *= 2.0;
      std::vector<std::size_t> order(n);
      for (std::size_t j {0}; j < n; ++j) {
        order[j] = j;
      }
      std::sort(order.begin(), order.end(), [&d](std::size_t x, std::size_t y) { return d[x] < d[y]; });
      double largest_d {0.0};
      double largest_z {0.0};
      for (std::size_t j {0}; j < n; ++j) {
        largest_d = std::max(largest_d, std::abs(d[j]));
        largest_z = std::max(largest_z, std::abs(z[j]));
      }
      const double             tolerance {8.0 * std::numeric_limits<double>::epsilon() * std::max(largest_d, largest_z)};
      std::vector<std::size_t> kept;        // the columns the secular equation updates, by ascending d
      std::vector<std::size_t> deflated;    // the columns that keep their eigenvalue
      std::size_t              pending {n}; // the last column kept so far, still open to a rotation with the next
      for (const std::size_t j : order) {
        if (rho * std::abs(z[j]) <= tolerance) {
          deflated.push_back(j);
          continue;
        }
        if (pending == n) {
          pending = j;
          continue;
        }
        const double length {std::hypot(z[j], z[pending])};
        const double c      {z[j] / length};
        const double s      {-z[pending] / length};
        if (std::abs((d[j] - d[pending]) * c * s) <= tolerance) {
          // Close poles: a rotation of the two columns zeroes the pending one's weight, which then deflates.
          z[j]       = length;
          z[pending] = 0.0;
          for (std::size_t r {0}; r < n; ++r) {
            const double x {q[(r * n) + pending]};
            const double y {q[(r * n) + j]};
            q[(r * n) + pending] = (c * x) + (s * y);
            q[(r * n) + j]       = (c * y) - (s * x);
          }
          const double rotated {(d[pending] * c * c) + (d[j] * s * s)};
          d[j]       = (d[pending] * s * s) + (d[j] * c * c);
          d[pending] = rotated;
          deflated.push_back(pending);
        }
        else {
          kept.push_back(pending);
        }
        pending = j;
      }
      if (pending != n) {
        kept.push_back(pending);
      }
      const std::size_t   k {kept.size()};
      std::vector<double> poles(k);
      std::vector<double> weights(k);
      double              norm {0.0};
      for (std::size_t i {0}; i < k; ++i) {
        poles[i]    = d[kept[i]];
        weights[i]  = z[kept[i]];
        norm       += weights[i] * weights[i];
      }
      norm = std::sqrt(norm);
      for (double& weight : weights) {
        weight /= norm;
      }
      const double        scaled_rho {rho * norm * norm};
      std::vector<double> roots(k);
      std::vector<double> gaps(k * k); // gaps[i * k + j] = poles[j] − roots[i]
      parallel_for(k, 8 * k, [&](std::size_t i) {
                     roots[i] = secular_root(poles, weights, scaled_rho, i, secular_iterations, gaps.data() + (i * k));
                   });
      std::vector<double> u(k * k); // column i: the eigenvector of roots[i] in the kept columns' basis
      for (std::size_t j {0}; j < k; ++j) {
        double product {gaps[(j * k) + j]}; // Löwner: ẑⱼ² = −Πᵢ (dⱼ − λᵢ) / Πᵢ≠ⱼ (dⱼ − dᵢ), each pole paired with its root
        for (std::size_t i {0}; i < k; ++i) {
          if (i != j) {
            product *= gaps[(i * k) + j] / (poles[j] - poles[i]);
          }
        }
        const double weight {std::copysign(std::sqrt(-product), weights[j])};
        for (std::size_t i {0}; i < k; ++i) {
          u[(j * k) + i] = weight / gaps[(i * k) + j];
        }
      }
      for (std::size_t i {0}; i < k; ++i) {
        double sum {0.0};
        for (std::size_t j {0}; j < k; ++j) {
          sum += u[(j * k) + i] * u[(j * k) + i];
        }
        const double length {std::sqrt(sum)};
        for (std::size_t j {0}; j < k; ++j) {
          u[(j * k) + i] /= length;
        }
      }
      std::vector<double> kept_columns(n * k);
      for (std::size_t r {0}; r < n; ++r) {
        for (std::size_t i {0}; i < k; ++i) {
          kept_columns[(r * k) + i] = q[(r * n) + kept[i]];
        }
      }
      std::vector<double> updated(n * k, 0.0);
      gemm(n, k, k, kept_columns.data(), k, u.data(), k, updated.data(), k);
      // Every eigenpair, then by ascending eigenvalue.
      std::vector<double> values(n);
      std::vector<double> vectors(n * n);
      for (std::size_t i {0}; i < k; ++i) {
        values[i] = roots[i];
        for (std::size_t r {0}; r < n; ++r) {
          vectors[(r * n) + i] = updated[(r * k) + i];
        }
      }
      for (std::size_t i {0}; i < deflated.size(); ++i) {
        values[k + i] = d[deflated[i]];
        for (std::size_t r {0}; r < n; ++r) {
          vectors[(r * n) + k + i] = q[(r * n) + deflated[i]];
        }
      }
      for (std::size_t j {0}; j < n; ++j) {
        order[j] = j;
      }
      std::sort(order.begin(), order.end(), [&values](std::size_t x, std::size_t y) { return values[x] < values[y]; });
      for (std::size_t j {0}; j < n; ++j) {
        d[j] = values[order[j]];
        for (std::size_t r {0}; r < n; ++r) {
          q[(r * n) + j] = vectors[(r * n) + order[j]];
        }
      }
    }

    /*!
     * \brief The eigenvalues, ascending, of the symmetric tridiagonal with diagonal `d[0…n-1]` and subdiagonal
     *        `e[0…n-2]` into \p d, and its eigenvectors as the columns of the row-major \p q (`n × n`), by
     *        Cuppen's divide and conquer: the tridiagonal is two halves plus a rank-one coupling, each half is
     *        diagonalised the same way (by implicit QL up to \ref divide_leaf), and \ref merge_rank_one joins
     *        them.
     */
    inline void divide_and_conquer(double*              d,
                                   const double*        e,
                                   std::size_t          n,
                                   std::vector<double>& q)
    {
      q.assign(n * n, 0.0);
      if (n <= divide_leaf) {
        std::vector<double> diagonal(d, d + n);
        std::vector<double> sub(n, 0.0);
        std::copy(e, e + (n - 1), sub.begin());
        std::vector<double> zt(n * n, 0.0);
        for (std::size_t i {0}; i < n; ++i) {
          zt[(i * n) + i] = 1.0;
        }
        tridiagonal_ql(diagonal, sub, zt, n, ql_steps_per_value * n);
        std::vector<std::size_t> order(n);
        for (std::size_t i {0}; i < n; ++i) {
          order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&diagonal](std::size_t x, std::size_t y) { return diagonal[x] < diagonal[y]; });
        for (std::size_t j {0}; j < n; ++j) {
          d[j] = diagonal[order[j]];
          for (std::size_t r {0}; r < n; ++r) {
            q[(r * n) + j] = zt[(order[j] * n) + r];
          }
        }
        return;
      }
      // T = diag(T₁ − ρ e_a e_aᵀ, T₂ − ρ e_b e_bᵀ) + ρ u uᵀ, u = e_a + sign(β) e_b, ρ = |β|, β the coupling.
      const std::size_t n1   {n / 2};
      const std::size_t n2   {n - n1};
      const double      beta {e[n1 - 1]};
      const double      rho  {std::abs(beta)};
      d[n1 - 1] -= rho;
      d[n1]     -= rho;
      std::vector<double> q1;
      std::vector<double> q2;
      divide_and_conquer(d, e, n1, q1);
      divide_and_conquer(d + n1, e + n1, n2, q2);
      std::vector<double> z(n);
      for (std::size_t i {0}; i < n1; ++i) {
        z[i] = q1[((n1 - 1) * n1) + i];
        for (std::size_t r {0}; r < n1; ++r) {
          q[(r * n) + i] = q1[(r * n1) + i];
        }
      }
      for (std::size_t i {0}; i < n2; ++i) {
        z[n1 + i] = std::copysign(1.0, beta) * q2[i];
        for (std::size_t r {0}; r < n2; ++r) {
          q[((n1 + r) * n) + n1 + i] = q2[(r * n2) + i];
        }
      }
      std::vector<double> values(d, d + n);
      merge_rank_one(values, std::move(z), rho, q, n);
      std::copy(values.begin(), values.end(), d);
    }

    //! \brief \p zt ← \p left · \p zt, `n × n`, row-major, for any other element type: the real factor converted.
    template <typename T>
    void multiply_real_left(const std::vector<double>& left,
                            std::vector<T>&            zt,
                            std::size_t                n)
    {
      std::vector<T> converted(n * n);
      for (std::size_t i {0}; i < n * n; ++i) {
        converted[i] = from_double<T>(left[i]);
      }
      std::vector<T> product(n * n, T {});
      gemm(n, n, n, converted.data(), n, zt.data(), n, product.data(), n);
      zt = std::move(product);
    }

    //! \brief \p zt ← \p left · \p zt, `n × n`, row-major, for a real left factor.
    inline void multiply_real_left(const std::vector<double>& left,
                                   std::vector<double>&       zt,
                                   std::size_t                n)
    {
      std::vector<double> product(n * n, 0.0);
      gemm(n, n, n, left.data(), n, zt.data(), n, product.data(), n);
      zt = std::move(product);
    }

    //! \brief \p zt ← \p left · \p zt for a complex \p zt: the real and imaginary parts by a real product each.
    inline void multiply_real_left(const std::vector<double>&         left,
                                   std::vector<std::complex<double>>& zt,
                                   std::size_t                        n)
    {
      std::vector<double> re(n * n);
      std::vector<double> im(n * n);
      for (std::size_t i {0}; i < n * n; ++i) {
        re[i] = zt[i].real();
        im[i] = zt[i].imag();
      }
      multiply_real_left(left, re, n);
      multiply_real_left(left, im, n);
      for (std::size_t i {0}; i < n * n; ++i) {
        zt[i] = {re[i], im[i]};
      }
    }

    /*!
     * \brief The eigenvalues of the symmetric tridiagonal (\p d, \p e) into \p d, the eigenvectors accumulated
     *        into the rows of \p zt as \ref tridiagonal_ql accumulates them, by divide and conquer (LAPACK's
     *        `dstedc`): most of the work in matrix products. Up to \ref divide_leaf, \ref tridiagonal_ql itself.
     * \throws convergence_error If an iteration did not converge within its bound.
     */
    template <typename T>
    void tridiagonal_divide_and_conquer(std::vector<double>& d,
                                        std::vector<double>& e,
                                        std::vector<T>&      zt,
                                        std::size_t          n)
    {
      if (n <= divide_leaf) {
        tridiagonal_ql(d, e, zt, n, ql_steps_per_value * n);
        return;
      }
      std::vector<double> q;
      divide_and_conquer(d.data(), e.data(), n, q);
      std::vector<double> qt(n * n); // row j: the eigenvector of d[j] in the tridiagonal's basis
      for (std::size_t r {0}; r < n; ++r) {
        for (std::size_t j {0}; j < n; ++j) {
          qt[(j * n) + r] = q[(r * n) + j];
        }
      }
      multiply_real_left(qt, zt, n);
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
    divide_and_conquer, //!< Householder tridiagonalisation then divide and conquer, as LAPACK's `dsyevd`: the default.
    tridiagonal_ql,     //!< Householder tridiagonalisation then implicit QL, as LAPACK's `dsyev`: the vectors by rotations.
    jacobi              //!< Cyclic Jacobi rotations: several times slower, kept as an independent second method.
  };

  /*!
   * \brief The eigenvalues and eigenvectors of a real symmetric matrix.
   *
   * By default the matrix is reduced to tridiagonal form by Householder reflections and the tridiagonal is
   * diagonalised by divide and conquer (\ref eigen_method::divide_and_conquer), the method of LAPACK's
   * `dsyevd`, most of its work in matrix products; implicit QL (\ref eigen_method::tridiagonal_ql, `dsyev`'s)
   * and cyclic Jacobi rotations (\ref eigen_method::jacobi) are the alternatives. Either way the eigenvalues are real
   * and the eigenvectors orthonormal. The eigenvalues are returned in ascending order, with \ref vectors
   * `j`-th column the eigenvector of \ref values `[j]` (`A = V Λ Vᵀ`). Real symmetric only (the complex
   * Hermitian case is \ref hermitian_eigen). Only the lower triangle is read, as `numpy.linalg.eigh`
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
                             eigen_method          method = eigen_method::divide_and_conquer)
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
        if (method == eigen_method::tridiagonal_ql) {
          detail::tridiagonal_ql(diagonal, off, zt, n_, detail::ql_steps_per_value * n_);
        }
        else {
          detail::tridiagonal_divide_and_conquer(diagonal, off, zt, n_);
        }
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
   * method of LAPACK's `zheevd`), which the divide and conquer of \ref symmetric_eigen diagonalises. The eigenvalues
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
        detail::tridiagonal_divide_and_conquer(diagonal, off, zt, n_);
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
              alpha += static_cast<double>(std::norm(wp[i]));
              beta  += static_cast<double>(std::norm(wq[i]));
              gamma += conj_of(wp[i]) * wq[i];
            }
            const double magnitude {static_cast<double>(std::abs(gamma))};
            if (magnitude <= epsilon * std::sqrt(alpha) * std::sqrt(beta)) {
              continue; // already orthogonal to rounding (or a zero column)
            }
            rotated = true;
            // Turn column q by the phase of γ so that the pair's inner product is real, then rotate.
            const T      turn {conj_of(gamma) / detail::from_double<T>(magnitude)};
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
                mp[i] = (detail::from_double<T>(c) * left) - (detail::from_double<T>(s) * right);
                mq[i] = (detail::from_double<T>(s) * left) + (detail::from_double<T>(c) * right);
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
     * \brief The bases of a bidiagonal reduction from its reflectors: `U_b = (H₀ H₁ …) diag(conj left phases)` and
     *        `V_b = (H'₀ H'₁ …) diag(right phases)`, each formed by blocks of reflectors through the matrix product
     *        kernel, as QR's `Q` is.
     * \param[in]  left        The left reflector of each step (acting from its row), empty where none.
     * \param[in]  right       The right reflector of each step (acting from the column past it), likewise.
     * \param[in]  left_phase  The unit phase each row took.
     * \param[in]  right_phase The unit phase each column past the diagonal took.
     * \param[in]  n           The order.
     * \param[out] ut          `U_bᵀ`, row-major: its rows are the columns of `U_b`.
     * \param[out] vt          `V_bᵀ`, likewise.
     */
    template <typename T>
    void form_bidiagonal_bases(const std::vector<std::vector<T>>& left,
                               std::vector<std::vector<T>>        right,
                               const std::vector<T>&              left_phase,
                               const std::vector<T>&              right_phase,
                               std::size_t                        n,
                               std::vector<T>&                    ut,
                               std::vector<T>&                    vt)
    {
      // U_b = L₀ᴴ L₁ᴴ … with L_kᴴ = H_k diag(conj phase at k), and V_b = R₀ R₁ … with R_k = H'_k diag(phase
      // at k+1). A phase on row k commutes with every later reflector, which acts on rows past it, so
      // U_b = (H₀ H₁ …) diag(conj phases) and V_b = (H'₀ H'₁ …) diag(phases): the reflectors are applied to the
      // diagonal by blocks, through the matrix product kernel, as QR's Q is formed.
      std::vector<T> u(n * n, T {});
      std::vector<T> v(n * n, T {});
      for (std::size_t k {0}; k < n; ++k) {
        u[(k * n) + k] = conj_of(left_phase[k]);
        v[(k * n) + k] = k == 0 ? T {1} : right_phase[k - 1];
      }
      std::vector<std::vector<T>> shifted(n); // right[k] acts from row k+1: as reflector k+1 of a QR
      for (std::size_t k {0}; k + 1 < n; ++k) {
        shifted[k + 1] = std::move(right[k]);
      }
      for (std::size_t k1 {n}; k1 > 0;) { // blocks of qr_block reflectors, the last first
        const std::size_t k0 {(k1 - 1) / qr_block * qr_block};
        apply_reflector_block(gather_reflectors(left, k0, k1, n), false, u, n, k0, n);
        apply_reflector_block(gather_reflectors(shifted, k0, k1, n), false, v, n, k0, n);
        k1 = k0;
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

    //! \brief The reflections from each side in one panel of the blocked bidiagonal reduction.
    inline constexpr std::size_t bidiagonal_block {32};

    /*!
     * \brief The unit Householder vector `v` with `(I − 2 v vᵀ) x = β e₁`, in place of \p x, for a real `x`.
     * \return `β = −sign(x₀) ‖x‖`.
     */
    template <typename T>
    [[nodiscard]] T householder_in_place(std::vector<T>& x)
    {
      const auto norm   {static_cast<T>(scaled_norm(x))};
      const T    beta   {-phase_of(x[0]) * norm};
      x[0] -= beta;
      const auto length {static_cast<T>(scaled_norm(x))};
      for (T& element : x) {
        element /= length;
      }
      return beta;
    }

    /*!
     * \brief Reduces the real square `n × n` row-major \p b to upper-bidiagonal form, `B = U_b D V_bᵀ`, by panels
     *        of \ref bidiagonal_block reflections from each side, as LAPACK's `dgebrd` and `dlabrd`: the matrix
     *        a panel's later steps see is `B − V Yᵀ − X Uᵀ` (`V`, `U` the reflectors, `Y`, `X` what they move),
     *        applied to the one column and the one row each step needs; the trailing block takes it after the
     *        panel, through the matrix product kernel. The bidiagonal keeps its signs, which the iterations that
     *        follow take.
     * \param[in,out] b  The matrix; destroyed.
     * \param[in]     n  Its order.
     * \param[out]    d  The diagonal (size \p n).
     * \param[out]    e  The superdiagonal in `e[0…n-2]`, and `e[n-1] = 0` (size \p n).
     * \param[out]    ut `U_bᵀ`, row-major: its rows are the columns of `U_b`.
     * \param[out]    vt `V_bᵀ`, row-major: its rows are the columns of `V_b`.
     */
    template <typename T>
    requires std::is_floating_point_v<T>
    void bidiagonalize(std::vector<T>&      b,
                       std::size_t          n,
                       std::vector<double>& d,
                       std::vector<double>& e,
                       std::vector<T>&      ut,
                       std::vector<T>&      vt)
    {
      std::vector<std::vector<T>> left(n);  // the left reflector of step k, on rows k…, empty where none was needed
      std::vector<std::vector<T>> right(n); // the right reflector of step k, on columns k+1…, likewise
      for (std::size_t k0 {0}; k0 < n; k0 += bidiagonal_block) {
        const std::size_t k1    {std::min(k0 + bidiagonal_block, n)};
        const std::size_t width {k1 - k0};
        // By reflector: row p of each holds step k0 + p's vector over the rows (v, x) or the columns (u, y), so every
        // term of the panel is a contiguous update.
        std::vector<T> by_row_v(width * n, T {}); // the left reflectors, at their rows
        std::vector<T> by_row_y(width * n, T {}); // what they move, 2 Bᵀ v, at the columns
        std::vector<T> by_row_u(width * n, T {}); // the right reflectors, at their columns
        std::vector<T> by_row_x(width * n, T {}); // what they move, 2 B u, at the rows
        const auto     row_of {[n](std::vector<T>& rows, std::size_t p, std::size_t from) { return rows.data() + (p * n) + from; }};
        for (std::size_t k {k0}; k < k1; ++k) {
          const std::size_t i {k - k0};
          std::vector<T>    x(n - k); // column k from the diagonal down, the panel's earlier steps applied
          for (std::size_t r {k}; r < n; ++r) {
            x[r - k] = b[(r * n) + k];
          }
          for (std::size_t p {0}; p < i; ++p) {
            subtract_product(x.data(), by_row_y[(p * n) + k], row_of(by_row_v, p, k), n - k);
            subtract_product(x.data(), by_row_u[(p * n) + k], row_of(by_row_x, p, k), n - k);
          }
          if (scaled_norm(std::vector<T>(x.begin() + 1, x.end())) > 0.0) {
            d[k] = static_cast<double>(householder_in_place(x));
            // y = 2 (B − V Yᵀ − X Uᵀ)ᵀ v on the columns past k: Bᵀ v by contiguous rows, then the panel's terms.
            std::vector<T> by_v(i, T {}); // Vᵀ v and Xᵀ v over the earlier steps
            std::vector<T> by_x(i, T {});
            for (std::size_t p {0}; p < i; ++p) {
              by_v[p] = dot_product(row_of(by_row_v, p, k), x.data(), n - k);
              by_x[p] = dot_product(row_of(by_row_x, p, k), x.data(), n - k);
            }
            const std::size_t     span        {n - k - 1};
            constexpr std::size_t chunk_width {64};
            parallel_for((span + chunk_width - 1) / chunk_width, (n - k) * chunk_width, [&](std::size_t chunk) {
                           const std::size_t first {k + 1 + (chunk * chunk_width)};
                           const std::size_t count {std::min(chunk_width, n - first)};
                           std::vector<T>    sum(count, T {});
                           for (std::size_t r {k}; r < n; ++r) {
                             add_product(sum.data(), x[r - k], b.data() + (r * n) + first, count);
                           }
                           for (std::size_t p {0}; p < i; ++p) {
                             subtract_product(sum.data(), by_v[p], row_of(by_row_y, p, first), count);
                             subtract_product(sum.data(), by_x[p], row_of(by_row_u, p, first), count);
                           }
                           for (std::size_t c {0}; c < count; ++c) {
                             by_row_y[(i * n) + first + c] = static_cast<T>(2) * sum[c];
                           }
                         });
            std::copy(x.begin(), x.end(), row_of(by_row_v, i, k));
            left[k] = std::move(x);
          }
          else {
            d[k] = static_cast<double>(x[0]);
          }
          if (k + 1 == n) {
            break;
          }
          std::vector<T> y(b.begin() + static_cast<std::ptrdiff_t>((k * n) + k + 1), // row k past the diagonal, the left
                           b.begin() + static_cast<std::ptrdiff_t>((k + 1) * n));    // reflection included
          for (std::size_t p {0}; p <= i; ++p) {
            subtract_product(y.data(), by_row_v[(p * n) + k], row_of(by_row_y, p, k + 1), n - k - 1);
          }
          for (std::size_t p {0}; p < i; ++p) {
            subtract_product(y.data(), by_row_x[(p * n) + k], row_of(by_row_u, p, k + 1), n - k - 1);
          }
          if (y.size() > 1 && scaled_norm(std::vector<T>(y.begin() + 1, y.end())) > 0.0) {
            e[k] = static_cast<double>(householder_in_place(y));
            // x = 2 (B − V Yᵀ − X Uᵀ) u on the rows past k: B u by rows, then the panel's terms.
            std::vector<T> by_y(i + 1, T {}); // Yᵀ u over this and the earlier steps, Uᵀ u over the earlier
            std::vector<T> by_u(i, T     {});
            for (std::size_t p {0}; p <= i; ++p) {
              by_y[p] = dot_product(row_of(by_row_y, p, k + 1), y.data(), y.size());
            }
            for (std::size_t p {0}; p < i; ++p) {
              by_u[p] = dot_product(row_of(by_row_u, p, k + 1), y.data(), y.size());
            }
            std::vector<T> moved(n - k - 1);
            parallel_for(n - k - 1, 2 * (n - k), [&](std::size_t row) {
                           moved[row] = dot_product(b.data() + ((k + 1 + row) * n) + k + 1, y.data(), y.size());
                         });
            for (std::size_t p {0}; p <= i; ++p) {
              subtract_product(moved.data(), by_y[p], row_of(by_row_v, p, k + 1), n - k - 1);
            }
            for (std::size_t p {0}; p < i; ++p) {
              subtract_product(moved.data(), by_u[p], row_of(by_row_x, p, k + 1), n - k - 1);
            }
            for (std::size_t r {0}; r < n - k - 1; ++r) {
              by_row_x[(i * n) + k + 1 + r] = static_cast<T>(2) * moved[r];
            }
            std::copy(y.begin(), y.end(), row_of(by_row_u, i, k + 1));
            right[k] = std::move(y);
          }
          else {
            e[k] = static_cast<double>(y[0]);
          }
        }
        if (k1 < n) { // the trailing block from row and column k1: B −= V Yᵀ + X Uᵀ
          const std::size_t rest {n - k1};
          std::vector<T>    minus_v(rest * width);
          std::vector<T>    minus_x(rest * width);
          for (std::size_t r {0}; r < rest; ++r) {
            for (std::size_t p {0}; p < width; ++p) {
              minus_v[(r * width) + p] = -by_row_v[(p * n) + k1 + r];
              minus_x[(r * width) + p] = -by_row_x[(p * n) + k1 + r];
            }
          }
          T* trailing {b.data() + (k1 * n) + k1};
          gemm(rest, rest, width, minus_v.data(), width, by_row_y.data() + k1, n, trailing, n);
          gemm(rest, rest, width, minus_x.data(), width, by_row_u.data() + k1, n, trailing, n);
        }
      }
      if (n > 0) {
        e[n - 1] = 0.0;
      }
      form_bidiagonal_bases(left, std::move(right), std::vector<T>(n, T {1}), std::vector<T>(n, T {1}), n, ut, vt);
    }

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
                                               v[0] += phase_of(v[0]) * detail::from_double<T>(norm);
                                               const auto length {detail::from_double<T>(scaled_norm(v))};
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
        d[k] = static_cast<double>(std::abs(b[(k * n) + k]));
        if (k + 1 == n) {
          break;
        }
        std::vector<T> y(n - k - 1); // row k right of the diagonal, conjugated: B ← B H with H yᴴ ∥ e₁
        for (std::size_t j {k + 1}; j < n; ++j) {
          y[j - k - 1] = conj_of(b[(k * n) + j]);
        }
        if (y.size() > 1 && scaled_norm(std::vector<T>(y.begin() + 1, y.end())) > 0.0) {
          reflector(y);
          parallel_for(n - k, 2 * y.size(), [&](std::size_t r) { // row ← row (I − 2 y yᴴ); rows above k are zero there
                         T*      row   {b.data() + ((k + r) * n) + k + 1};
                         const T scale {static_cast<T>(2) * dot_product(row, y.data(), y.size())};
                         for (std::size_t j {0}; j < y.size(); ++j) {
                           row[j] -= scale * conj_of(y[j]);
                         }
                       });
          right[k] = std::move(y);
        }
        right_phase[k] = conj_of(phase_of(b[(k * n) + k + 1])); // column k+1 times it: |b_k,k+1| above
        for (std::size_t i {k}; i < n; ++i) {
          b[(i * n) + k + 1] *= right_phase[k];
        }
        e[k] = static_cast<double>(std::abs(b[(k * n) + k + 1]));
      }
      if (n > 0) {
        e[n - 1] = 0.0;
      }
      form_bidiagonal_bases(left, std::move(right), left_phase, right_phase, n, ut, vt);
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
      // The iteration reads only d and e: what it does to the rows of vt and ut is recorded and applied once
      // at the end, in parallel by columns.
      std::vector<row_operation> on_vt;
      std::vector<row_operation> on_ut;
      const double               epsilon {std::numeric_limits<double>::epsilon()};
      const double               tiny    {std::numeric_limits<double>::min() / epsilon};
      constexpr std::size_t      none    {std::numeric_limits<std::size_t>::max()};
      std::size_t                p       {n};
      std::size_t                steps   {0};
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
            on_vt.push_back({row_operation::kind::negate, top, top, 0.0, 0.0});
          }
          for (std::size_t j {top}; j + 1 < n && d[j] < d[j + 1]; ++j) {
            std::swap(d[j], d[j + 1]);
            on_vt.push_back({row_operation::kind::swap, j, j + 1, 0.0, 0.0});
            on_ut.push_back({row_operation::kind::swap, j, j + 1, 0.0, 0.0});
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
            on_vt.push_back({row_operation::kind::rotate, j, p - 1, cs, sn});
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
            on_ut.push_back({row_operation::kind::rotate, j, zero, cs, sn});
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
          on_vt.push_back({row_operation::kind::rotate, j, j + 1, cs, sn});
          t        = std::hypot(f, g);
          cs       = f / t;
          sn       = g / t;
          d[j]     = t;
          f        = (cs * e[j]) + (sn * d[j + 1]);
          d[j + 1] = (cs * d[j + 1]) - (sn * e[j]);
          g        = sn * e[j + 1];
          e[j + 1] = cs * e[j + 1];
          on_ut.push_back({row_operation::kind::rotate, j, j + 1, cs, sn});
        }
        e[p - 2] = f;
      }
      apply_row_operations(vt, n, on_vt);
      apply_row_operations(ut, n, on_ut);
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

    /*!
     * \brief Singular value \p i of the arrow `M` whose first row is `z` and whose diagonal is `d` (`d₀ = 0`, `d`
     *        strictly ascending): root `σ` of `1 + Σⱼ zⱼ²/(dⱼ² − σ²) = 0`, as LAPACK's `dlasd4`.
     *
     * Solved by \ref solve_secular in `μ = σ² − d_o²`, the poles' offsets `(dⱼ − d_o)(dⱼ + d_o)` computed from their
     * two accurate factors, so that both `dⱼ − σ` and `dⱼ + σ` come out with small relative errors.
     * \param[in]  d              The diagonal (the poles).
     * \param[in]  z              The first row (the weights).
     * \param[in]  i              The singular value.
     * \param[in]  max_iterations The bound on the steps.
     * \param[out] below          `dⱼ − σ` for every `j`.
     * \param[out] above          `dⱼ + σ` for every `j`.
     * \return `σ`.
     */
    [[nodiscard]] inline double singular_secular_root(const std::vector<double>& d,
                                                      const std::vector<double>& z,
                                                      std::size_t                i,
                                                      std::size_t                max_iterations,
                                                      double*                    below,
                                                      double*                    above)
    {
      std::vector<double>    delta(d.size());
      const secular_solution root  {solve_secular(z, 1.0, i, max_iterations, delta.data(), [&d](std::size_t o, std::size_t j) {
                                                    return (d[j] - d[o]) * (d[j] + d[o]);
                                                  })};
      const double           pole  {d[root.origin]};
      const double           shift {root.offset / (pole + std::sqrt((pole * pole) + root.offset))}; // σ − d_o from σ² − d_o²
      for (std::size_t j {0}; j < d.size(); ++j) {
        below[j] = (d[j] - pole) - shift;
        above[j] = (d[j] + pole) + shift;
      }
      return pole + shift;
    }

    //! \brief Columns \p a and \p b of the row-major \p m (\p cols per row) rotated: `a ← c a + s b`, `b ← c b − s a`.
    inline void rotate_columns(std::vector<double>& m,
                               std::size_t          cols,
                               std::size_t          a,
                               std::size_t          b,
                               double               c,
                               double               s)
    {
      for (std::size_t r {0}; r < m.size() / cols; ++r) {
        const double x {m[(r * cols) + a]};
        const double y {m[(r * cols) + b]};
        m[(r * cols) + a] = (c * x) + (s * y);
        m[(r * cols) + b] = (c * y) - (s * x);
      }
    }

    /*!
     * \brief The singular value decomposition of the arrow `M` (first row `z`, diagonal `d`, `d₀ = 0`), carried into
     *        the bases \p u (`n × n`) and \p v (`m × m`, its first `n` columns the arrow's; a last, null column
     *        when `m = n + 1`), as LAPACK's `dlasd1`–`dlasd3`: on return \p d holds the singular values ascending
     *        and \p u, \p v the singular vectors, \p v's null column last.
     *
     * Deflation first (`dlasd2`): a negligible `zⱼ` leaves `dⱼ` and its vectors as they are, a `dⱼ` near zero is
     * rotated into the zero pole, and two nearly equal `dⱼ` are rotated together so that one of them deflates.
     * The remaining roots are found by \ref singular_secular_root, in parallel; `z` is recomputed from them
     * (Löwner's formula), so that the vectors are orthogonal to working precision, and they reach \p u and \p v by
     * a matrix product each.
     * \param[in,out] d          The diagonal; on return, the singular values.
     * \param[in]     z          The first row.
     * \param[in]     alpha,beta The entries the arrow was built from: they set the deflation tolerance.
     * \param[in,out] u          The left basis; on return, the left singular vectors.
     * \param[in,out] v          The right basis; on return, the right singular vectors.
     * \param[in]     n          The arrow's order.
     * \param[in]     m          The right basis's order (`n`, or `n + 1` with a null column).
     */
    inline void merge_arrow(std::vector<double>& d,
                            std::vector<double>  z,
                            double               alpha,
                            double               beta,
                            std::vector<double>& u,
                            std::vector<double>& v,
                            std::size_t          n,
                            std::size_t          m)
    {
      std::vector<std::size_t> order(n - 1);
      for (std::size_t j {1}; j < n; ++j) {
        order[j - 1] = j;
      }
      std::sort(order.begin(), order.end(), [&d](std::size_t x, std::size_t y) { return d[x] < d[y]; });
      double largest {std::max(std::abs(alpha), std::abs(beta))};
      for (const double value : d) {
        largest = std::max(largest, value);
      }
      const double tolerance {8.0 * std::numeric_limits<double>::epsilon() * largest};
      if (std::abs(z[0]) <= tolerance) {
        z[0] = tolerance; // the zero pole keeps a weight: the arrow moves by no more than the tolerance
      }
      std::vector<std::size_t> kept    {0}; // by ascending d, the zero pole first
      std::vector<std::size_t> deflated;    // the columns that keep their singular value
      std::size_t              pending {n}; // the last column kept so far, still open to a rotation with the next
      for (const std::size_t j : order) {
        if (std::abs(z[j]) <= tolerance) {
          deflated.push_back(j);
          continue;
        }
        if (d[j] <= tolerance) {
          // Nearly the zero pole: its weight joins the zero pole's, and dⱼ deflates.
          const double length {std::hypot(z[0], z[j])};
          rotate_columns(v, m, 0, j, z[0] / length, z[j] / length);
          z[0] = length;
          z[j] = 0.0;
          deflated.push_back(j);
          continue;
        }
        if (pending != n && d[j] - d[pending] <= tolerance) {
          // Close values: one rotation of both rows and columns zeroes the pending one's weight.
          const double length {std::hypot(z[j], z[pending])};
          const double c      {z[j] / length};
          const double s      {z[pending] / length};
          rotate_columns(v, m, j, pending, c, s);
          rotate_columns(u, n, j, pending, c, s);
          z[j]       = length;
          z[pending] = 0.0;
          deflated.push_back(pending);
        }
        else if (pending != n) {
          kept.push_back(pending);
        }
        pending = j;
      }
      if (pending != n) {
        kept.push_back(pending);
      }
      const std::size_t   k {kept.size()};
      std::vector<double> poles(k);
      std::vector<double> weights(k);
      for (std::size_t i {0}; i < k; ++i) {
        poles[i]   = d[kept[i]];
        weights[i] = z[kept[i]];
      }
      std::vector<double> roots(k);
      std::vector<double> below(k * k); // below[i * k + j] = poles[j] − roots[i]
      std::vector<double> above(k * k); // above[i * k + j] = poles[j] + roots[i]
      parallel_for(k, 8 * k, [&](std::size_t i) {
                     roots[i] = singular_secular_root(poles, weights, i, secular_iterations, below.data() + (i * k),
                                                      above.data() + (i * k));
                   });
      std::vector<double> left(k * k);  // column i: the left vector of roots[i], in the kept columns' basis
      std::vector<double> right(k * k); // and the right one
      for (std::size_t j {0}; j < k; ++j) {
        // Löwner: ẑⱼ² = Πᵢ (σᵢ² − dⱼ²) / Πᵢ≠ⱼ (dᵢ² − dⱼ²), each pole paired with its root.
        double product {-below[(j * k) + j] * above[(j * k) + j]};
        for (std::size_t i {0}; i < k; ++i) {
          if (i != j) {
            product *= (-below[(i * k) + j] * above[(i * k) + j]) / ((poles[i] - poles[j]) * (poles[i] + poles[j]));
          }
        }
        const double weight {std::copysign(std::sqrt(product), weights[j])};
        for (std::size_t i {0}; i < k; ++i) {
          const double ratio {weight / (below[(i * k) + j] * above[(i * k) + j])}; // ẑⱼ / (dⱼ² − σᵢ²)
          right[(j * k) + i] = ratio;
          left[(j * k) + i]  = j == 0 ? -1.0 : poles[j] * ratio;
        }
      }
      for (std::vector<double>* vectors : {&left, &right}) {
        for (std::size_t i {0}; i < k; ++i) {
          double sum {0.0};
          for (std::size_t j {0}; j < k; ++j) {
            sum += (*vectors)[(j * k) + i] * (*vectors)[(j * k) + i];
          }
          const double length {std::sqrt(sum)};
          for (std::size_t j {0}; j < k; ++j) {
            (*vectors)[(j * k) + i] /= length;
          }
        }
      }
      // The kept columns times the arrow's vectors; the deflated columns as they are; then by ascending value.
      const auto carried {[&](const std::vector<double>& basis, std::size_t rows, const std::vector<double>& small) {
                            std::vector<double> columns(rows * k);
                            for (std::size_t r {0}; r < rows; ++r) {
                              for (std::size_t i {0}; i < k; ++i) {
                                columns[(r * k) + i] = basis[(r * rows) + kept[i]];
                              }
                            }
                            std::vector<double> product(rows * k, 0.0);
                            gemm(rows, k, k, columns.data(), k, small.data(), k, product.data(), k);
                            return product;
                          }};
      const std::vector<double> new_u {carried(u, n, left)};
      const std::vector<double> new_v {carried(v, m, right)};
      std::vector<double>       values(n);
      std::vector<std::size_t>  from(n); // where each value's vectors are: a kept root (< k) or a deflated column
      for (std::size_t i {0}; i < k; ++i) {
        values[i] = roots[i];
        from[i]   = i;
      }
      for (std::size_t i {0}; i < deflated.size(); ++i) {
        values[k + i] = d[deflated[i]];
        from[k + i]   = k + i;
      }
      std::vector<std::size_t> rank(n);
      for (std::size_t j {0}; j < n; ++j) {
        rank[j] = j;
      }
      std::sort(rank.begin(), rank.end(), [&values](std::size_t x, std::size_t y) { return values[x] < values[y]; });
      std::vector<double> sorted_u(n * n);
      std::vector<double> sorted_v(v); // the null column, if any, stays last
      for (std::size_t c {0}; c < n; ++c) {
        const std::size_t source {from[rank[c]]};
        d[c] = values[rank[c]];
        for (std::size_t r {0}; r < n; ++r) {
          sorted_u[(r * n) + c] = source < k ? new_u[(r * k) + source] : u[(r * n) + deflated[source - k]];
        }
        for (std::size_t r {0}; r < m; ++r) {
          sorted_v[(r * m) + c] = source < k ? new_v[(r * k) + source] : v[(r * m) + deflated[source - k]];
        }
      }
      u = std::move(sorted_u);
      v = std::move(sorted_v);
    }

    //! \brief The order up to which \ref bidiagonal_divide decomposes by implicit-shift QR.
    inline constexpr std::size_t bidiagonal_leaf {32};

    /*!
     * \brief The singular value decomposition of the upper bidiagonal `n × (n + extra)` (`extra` 0 or 1) with
     *        diagonal `d[0…n-1]` and superdiagonal `e[0…n+extra-2]` (`e[n-1]` the last column's entry when `extra`
     *        is 1), by divide and conquer, as LAPACK's `dbdsdc`: \p sigma ascending, \p u (`n × n`) and \p v
     *        (`m × m`, `m = n + extra`, the null vector last) row-major, their columns the singular vectors.
     *
     * The rows split at a middle row `k`: above it an `k × (k + 1)` bidiagonal, below it an `(n − k − 1) × (…)` one,
     * each decomposed the same way (by implicit-shift QR up to \ref bidiagonal_leaf); row `k`'s two entries, in
     * their right bases, with the two null vectors rotated into one, make an arrow, which \ref merge_arrow
     * decomposes.
     */
    inline void bidiagonal_divide(const double*        d,
                                  const double*        e,
                                  std::size_t          n,
                                  std::size_t          extra,
                                  std::vector<double>& sigma,
                                  std::vector<double>& u,
                                  std::vector<double>& v)
    {
      // Work near unit size, scaled by a power of two (exactly): the secular equation works on squares, which
      // overflow past 1e154.
      double largest {0.0};
      for (std::size_t i {0}; i < n; ++i) {
        largest = std::max({largest, std::abs(d[i]), i + 1 < n + extra ? std::abs(e[i]) : 0.0});
      }
      int exponent {0};
      (void)std::frexp(largest, &exponent);
      if (exponent != 0) {
        std::vector<double> scaled_d(n);
        std::vector<double> scaled_e(n, 0.0);
        for (std::size_t i {0}; i < n; ++i) {
          scaled_d[i] = std::ldexp(d[i], -exponent);
          scaled_e[i] = i + 1 < n + extra ? std::ldexp(e[i], -exponent) : 0.0;
        }
        bidiagonal_divide(scaled_d.data(), scaled_e.data(), n, extra, sigma, u, v);
        for (double& value : sigma) {
          value = std::ldexp(value, exponent);
        }
        return;
      }
      const std::size_t m {n + extra};
      sigma.assign(n, 0.0);
      u.assign(n * n, 0.0);
      v.assign(m * m, 0.0);
      for (std::size_t i {0}; i < m; ++i) {
        v[(i * m) + i] = 1.0;
      }
      if (n <= bidiagonal_leaf) {
        std::vector<double> diagonal(d, d + n);
        std::vector<double> super(n, 0.0);
        std::copy_n(e, n - std::min<std::size_t>(n, 1), super.begin());
        // An extra column first: rotations of it against each column, from the last row up, chase its one entry
        // out (B G = [B' 0]), so the square B' remains.
        double fill {extra == 1 ? e[n - 1] : 0.0};
        for (std::size_t j {n}; extra == 1 && j-- > 0;) {
          const double length {std::hypot(diagonal[j], fill)};
          const double c      {length == 0.0 ? 1.0 : diagonal[j] / length};
          const double s      {length == 0.0 ? 0.0 : fill / length};
          diagonal[j] = length;
          rotate_columns(v, m, j, n, c, s);
          if (j > 0) {
            fill         = -s * super[j - 1];
            super[j - 1] = c * super[j - 1];
          }
        }
        std::vector<double> ut(n * n, 0.0);
        std::vector<double> vt(n * n, 0.0);
        for (std::size_t i {0}; i < n; ++i) {
          ut[(i * n) + i] = 1.0;
          vt[(i * n) + i] = 1.0;
        }
        bidiagonal_qr(diagonal, super, ut, vt, n, svd_qr_steps_per_value * n); // descending
        std::vector<double> rotated(v);
        for (std::size_t c {0}; c < n; ++c) {
          const std::size_t source {n - 1 - c};
          sigma[c] = diagonal[source];
          for (std::size_t r {0}; r < n; ++r) {
            u[(r * n) + c] = ut[(source * n) + r];
          }
          for (std::size_t r {0}; r < m; ++r) { // G times B''s right vector
            double sum {0.0};
            for (std::size_t t {0}; t < n; ++t) {
              sum += rotated[(r * m) + t] * vt[(source * n) + t];
            }
            v[(r * m) + c] = sum;
          }
        }
        return;
      }
      const std::size_t   k     {n / 2};
      const std::size_t   lower {n - k - 1};
      const double        alpha {d[k]};
      const double        beta  {e[k]};
      std::vector<double> sigma1;
      std::vector<double> u1;
      std::vector<double> v1;
      std::vector<double> sigma2;
      std::vector<double> u2;
      std::vector<double> v2;
      bidiagonal_divide(d, e, k, 1, sigma1, u1, v1);
      bidiagonal_divide(d + k + 1, e + k + 1, lower, extra, sigma2, u2, v2);
      const std::size_t   m1 {k + 1};
      const std::size_t   m2 {lower + extra};
      std::vector<double> arrow(n, 0.0);
      std::vector<double> z(n, 0.0);
      // The bases: row k is the arrow's first row; the arrow's columns are the null vector, then the halves'.
      u[k * n] = 1.0;
      for (std::size_t c {0}; c < k; ++c) {
        arrow[1 + c] = sigma1[c];
        z[1 + c]     = alpha * v1[(k * m1) + c];
        for (std::size_t r {0}; r < k; ++r) {
          u[(r * n) + 1 + c] = u1[(r * k) + c];
        }
      }
      for (std::size_t c {0}; c < lower; ++c) {
        arrow[k + 1 + c] = sigma2[c];
        z[k + 1 + c]     = beta * v2[c];
        for (std::size_t r {0}; r < lower; ++r) {
          u[((k + 1 + r) * n) + k + 1 + c] = u2[(r * lower) + c];
        }
      }
      std::fill(v.begin(), v.end(), 0.0);
      for (std::size_t r {0}; r < m1; ++r) {
        v[(r * m) + 0] = v1[(r * m1) + k];
        for (std::size_t c {0}; c < k; ++c) {
          v[(r * m) + 1 + c] = v1[(r * m1) + c];
        }
      }
      for (std::size_t r {0}; r < m2; ++r) {
        for (std::size_t c {0}; c < lower; ++c) {
          v[((m1 + r) * m) + k + 1 + c] = v2[(r * m2) + c];
        }
      }
      z[0] = alpha * v1[(k * m1) + k];
      if (extra == 1) { // the two null vectors: one joins the arrow, the other is the whole's
        for (std::size_t r {0}; r < m2; ++r) {
          v[((m1 + r) * m) + n] = v2[(r * m2) + lower];
        }
        const double other  {beta * v2[lower]};
        const double length {std::hypot(z[0], other)};
        const double c      {length == 0.0 ? 1.0 : z[0] / length};
        const double s      {length == 0.0 ? 0.0 : other / length};
        rotate_columns(v, m, 0, n, c, s);
        z[0] = length;
      }
      merge_arrow(arrow, std::move(z), alpha, beta, u, v, n, m);
      sigma = std::move(arrow);
    }

    /*!
     * \brief The singular values of the upper bidiagonal (\p d, \p e) into \p d, descending, the singular vectors
     *        accumulated into the rows of \p ut and \p vt as \ref bidiagonal_qr accumulates them, by divide and
     *        conquer (\ref bidiagonal_divide): most of the work in matrix products.
     * \throws convergence_error If an iteration did not converge within its bound.
     */
    template <typename T>
    void bidiagonal_divide_and_conquer(std::vector<double>& d,
                                       std::vector<double>& e,
                                       std::vector<T>&      ut,
                                       std::vector<T>&      vt,
                                       std::size_t          n)
    {
      if (n == 0) {
        return;
      }
      std::vector<double> sigma;
      std::vector<double> u;
      std::vector<double> v;
      bidiagonal_divide(d.data(), e.data(), n, 0, sigma, u, v);
      std::vector<double> left(n * n); // row c: the vectors of the c-th largest value
      std::vector<double> right(n * n);
      for (std::size_t c {0}; c < n; ++c) {
        d[c] = sigma[n - 1 - c];
        for (std::size_t r {0}; r < n; ++r) {
          left[(c * n) + r]  = u[(r * n) + (n - 1 - c)];
          right[(c * n) + r] = v[(r * n) + (n - 1 - c)];
        }
      }
      multiply_real_left(left, ut, n);
      multiply_real_left(right, vt, n);
    }
  } // namespace detail

  //! \brief How \ref svd_decomposition computes the decomposition of `R`, after a Householder QR.
  enum class svd_method
  {
    divide_and_conquer, //!< Golub–Kahan bidiagonalisation then divide and conquer, as LAPACK's `dgesdd`: the default.
    bidiagonal_qr,      //!< Golub–Kahan bidiagonalisation then implicit-shift QR, as LAPACK's `dgesvd`.
    jacobi              //!< One-sided Jacobi: several times slower, each singular value to high relative accuracy.
  };

  /*!
   * \brief The singular value decomposition `A = U Σ Vᴴ` of an `m × n` matrix.
   * \tparam T The scalar element type (`double` or `std::complex<double>`).
   *
   * With `k = min(m, n)`: the `k` singular values, descending; `U` (`m × k`) and `V` (`n × k`) with
   * orthonormal columns, also where `A` is rank-deficient. Computed after a Householder QR of `A` (of `Aᴴ`
   * when `m < n`): by default `R` is reduced to bidiagonal form and the bidiagonal decomposed by divide and
   * conquer (\ref svd_method::divide_and_conquer), the method of LAPACK's `dgesdd`, most of its work in matrix
   * products, or by implicit-shift QR (\ref svd_method::bidiagonal_qr, `dgesvd`'s); either way each singular
   * value to an absolute accuracy of the largest times the machine epsilon. One-sided Jacobi on `R`
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
                               svd_method       method = svd_method::divide_and_conquer)
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
          entries[(i * cols) + j] = a(i, j) / detail::from_double<T>(scale);
        }
      }
      detail::thin_qr<T> factors {detail::householder_qr(std::move(entries), rows, cols)};
      if (method != svd_method::jacobi) {
        std::vector<double> d(cols);
        std::vector<double> e(cols);
        std::vector<T>      ut;
        std::vector<T>      vt;
        detail::bidiagonalize(factors.r, cols, d, e, ut, vt);
        if (method == svd_method::bidiagonal_qr) {
          detail::bidiagonal_qr(d, e, ut, vt, cols, detail::svd_qr_steps_per_value * cols);
        }
        else {
          detail::bidiagonal_divide_and_conquer(d, e, ut, vt, cols);
        }
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
            left[(i * cols) + d] = w[(i * cols) + j] / detail::from_double<T>(norms[j]);
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
          left[(i * cols) + d] = best[i] / detail::from_double<T>(norm);
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
        coefficient /= detail::from_double<T>(sigma[d]);
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
      const auto inverse {detail::from_double<T>(1.0 / s.singular_values()[d])};
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
