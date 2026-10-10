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
     * \brief The inverse, solved column by column against the identity.
     * \return The inverse matrix.
     * \throws singular_matrix_error If the matrix is singular.
     */
    [[nodiscard]] matrix<T> inverse() const
    {
      if (singular_) {
        throw singular_matrix_error("inverse: matrix is singular");
      }
      std::vector<T> data(n_ * n_);
      // Each column of the inverse is an independent solve against a unit vector,
      // writing a disjoint column of `data` — so the columns run in parallel.
      detail::parallel_for(n_, n_ * n_, [this, &data](std::size_t col) { // a solve: n² multiply-adds
                             std::vector<T> unit(n_, T {});
                             unit[col] = T {1};
                             const vector<T> column {solve(vector<T> {std::move(unit)})};
                             for (std::size_t row {0}; row < n_; ++row) {
                               data[(row * n_) + col] = column[row];
                             }
                           });
      return matrix<T> {n_, n_, std::move(data)};
    }

  private:

    //! \brief Performs the pivoted elimination, filling \ref lu_ and \ref pivots_.
    void factorize()
    {
      for (std::size_t k {0}; k < n_; ++k) {
        std::size_t pivot {k};
        double      best  {std::abs(lu_[(k * n_) + k])};
        for (std::size_t r {k + 1}; r < n_; ++r) {
          const double mag {std::abs(lu_[(r * n_) + k])};
          if (mag > best) {
            best  = mag;
            pivot = r;
          }
        }
        if (best == 0.0) {
          singular_ = true;
          continue; // a zero pivot column: leave it, the determinant becomes 0
        }
        if (pivot != k) {
          for (std::size_t c {0}; c < n_; ++c) {
            std::swap(lu_[(k * n_) + c], lu_[(pivot * n_) + c]);
          }
          std::swap(pivots_[k], pivots_[pivot]);
          sign_ = -sign_;
        }
        for (std::size_t r {k + 1}; r < n_; ++r) {
          const T factor {lu_[(r * n_) + k] / lu_[(k * n_) + k]};
          lu_[(r * n_) + k] = factor; // store the multiplier in L's slot
          for (std::size_t c {k + 1}; c < n_; ++c) {
            lu_[(r * n_) + c] -= factor * lu_[(k * n_) + c];
          }
        }
      }
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
      lower_.assign(n_ * n_, T {}); // lower-triangular; the upper triangle stays zero
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j <= i; ++j) {
          T sum {a(i, j)};
          for (std::size_t k {0}; k < j; ++k) {
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

    //! \brief The unit-modulus factor of \p x: `x / |x|` (its sign, for a real scalar), and 1 at 0.
    template <typename T>
    [[nodiscard]] T phase_of(const T& x)
    {
      const double magnitude {static_cast<double>(std::abs(x))};
      return magnitude == 0.0 ? T {1} : x / static_cast<T>(magnitude);
    }

    //! \brief A thin QR factorisation, row-major: `q` is `rows × cols` with orthonormal columns, `r` is
    //!        `cols × cols` upper triangular with a real, non-negative diagonal.
    template <typename T>
    struct thin_qr
    {
      std::vector<T> q; //!< The orthonormal factor.
      std::vector<T> r; //!< The triangular factor.
    };

    /*!
     * \brief Factorises the `rows × cols` row-major \p a (`rows ≥ cols`) by Householder reflections: each
     *        column below the diagonal is reflected onto it, so `Q` is orthonormal to rounding whatever the
     *        conditioning of `A`, and a column dependent on the earlier ones leaves a zero on `R`'s diagonal.
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
      std::vector<std::vector<T>> reflectors(cols); // v_k on rows k…m-1; empty where the column was zero
      for (std::size_t k {0}; k < cols; ++k) {
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
        for (std::size_t j {k}; j < cols; ++j) { // A ← (I − 2 v vᴴ) A on rows k…m-1
          T projection {};
          for (std::size_t i {k}; i < rows; ++i) {
            projection += conj_of(v[i - k]) * a[(i * cols) + j];
          }
          for (std::size_t i {k}; i < rows; ++i) {
            a[(i * cols) + j] -= static_cast<T>(2) * v[i - k] * projection;
          }
        }
        reflectors[k] = std::move(v);
      }
      thin_qr<T> out {.q = std::vector<T>(rows * cols, T {}), .r = std::vector<T>(cols * cols, T {})};
      for (std::size_t i {0}; i < cols; ++i) {
        out.q[(i * cols) + i] = T {1};
        for (std::size_t j {i}; j < cols; ++j) {
          out.r[(i * cols) + j] = a[(i * cols) + j];
        }
      }
      for (std::size_t k {cols}; k-- > 0;) { // Q = H₀ H₁ … applied to the first columns of I
        const std::vector<T>& v {reflectors[k]};
        if (v.empty()) {
          continue;
        }
        for (std::size_t j {0}; j < cols; ++j) {
          T projection {};
          for (std::size_t i {k}; i < rows; ++i) {
            projection += conj_of(v[i - k]) * out.q[(i * cols) + j];
          }
          for (std::size_t i {k}; i < rows; ++i) {
            out.q[(i * cols) + j] -= static_cast<T>(2) * v[i - k] * projection;
          }
        }
      }
      for (std::size_t k {0}; k < cols; ++k) { // R's diagonal real and non-negative: rotate row k, column k
        const T phase {phase_of(out.r[(k * cols) + k])};
        for (std::size_t j {k}; j < cols; ++j) {
          out.r[(k * cols) + j] *= conj_of(phase);
        }
        for (std::size_t i {0}; i < rows; ++i) {
          out.q[(i * cols) + k] *= phase;
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
   * \brief A QR factorisation `A = Q R` of an `m × n` matrix with `m ≥ n`.
   * \tparam T The scalar element type (a field: `double`, `std::complex<double>`, …).
   *
   * `Q` is `m × n` with orthonormal columns (`Qᴴ Q = I`) and `R` is `n × n` upper triangular with a real,
   * non-negative diagonal. Computed by Householder reflections, so `Q` stays orthonormal to rounding
   * however ill-conditioned `A` is, and a rank-deficient `A` factorises too (with zeros on `R`'s
   * diagonal). \ref solve returns the least-squares solution of `A x ≈ b` for a full-rank `A`; for a
   * rank-deficient one, \ref numerical::lstsq gives the minimum-norm solution.
   */
  template <typename T>
  class qr_decomposition
  {
  public:

    /*!
     * \brief Factorises \p a. \param[in] a The matrix (rows ≥ columns).
     * \throws dimension_error If \p a has fewer rows than columns.
     */
    explicit qr_decomposition(const matrix<T>& a)
      : rows_(a.rows()), cols_(a.cols())
    {
      if (rows_ < cols_) {
        throw dimension_error("QR decomposition requires rows >= columns");
      }
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

    //! \brief The `m × n` factor `Q` (orthonormal columns).
    [[nodiscard]] matrix<T> q() const
    {
      return matrix<T> {rows_, cols_, q_};
    }

    //! \brief The `n × n` upper-triangular factor `R`.
    [[nodiscard]] matrix<T> r() const
    {
      return matrix<T> {cols_, cols_, r_};
    }

    /*!
     * \brief The least-squares solution of `A x ≈ b` (`x = R⁻¹ Qᴴ b`) for a full-rank `A`.
     * \param[in] b The right-hand side (size `m`).
     * \return The solution `x` (size `n`).
     * \throws dimension_error If `b`'s size differs from the row count.
     * \throws singular_matrix_error If `A` is rank-deficient: a diagonal entry of `R` is under
     *         \ref detail::rank_tolerance (\ref numerical::lstsq solves such a system).
     */
    [[nodiscard]] vector<T> solve(const vector<T>& b) const
    {
      if (b.size() != rows_) {
        throw dimension_error("solve: right-hand side size differs from the matrix");
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
          sum += detail::conj_of(q_[(k * cols_) + i]) * b[k];
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

    std::size_t    rows_; //!< The number of rows (m).
    std::size_t    cols_; //!< The number of columns (n).
    std::vector<T> q_;    //!< The m×n factor Q, row-major.
    std::vector<T> r_;    //!< The n×n upper-triangular factor R, row-major.
  };

  //! \brief QR-factorises an `m × n` matrix (`m ≥ n`). \see qr_decomposition
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
  } // namespace detail

  /*!
   * \brief The eigenvalues and eigenvectors of a real symmetric matrix.
   *
   * Computed by the cyclic Jacobi algorithm: orthogonal plane rotations drive the
   * off-diagonal entries to zero, so the eigenvalues are real and the
   * eigenvectors orthonormal. The eigenvalues are returned in ascending order,
   * with \ref vectors `j`-th column the eigenvector of \ref values `[j]`
   * (`A = V Λ Vᵀ`). Real symmetric only — the complex Hermitian case (complex
   * rotations) is a deliberate future addition. Only the lower triangle is read, as
   * `numpy.linalg.eigh` reads it by default: the upper one is taken to mirror it.
   */
  class symmetric_eigen
  {
  public:

    /*!
     * \brief Diagonalises \p input.
     * \param[in] input      A real symmetric matrix (its lower triangle is read).
     * \param[in] max_sweeps The bound of Jacobi sweeps.
     * \throws dimension_error   If \p input is not square.
     * \throws convergence_error If the off-diagonal is not negligible after \p max_sweeps sweeps.
     */
    explicit symmetric_eigen(const matrix<double>& input,
                             std::size_t           max_sweeps = detail::jacobi_sweeps)
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
      // The off-diagonal norm strictly decreases each rotation and converges to 0,
      // so this terminates; the threshold is relative to the matrix scale.
      const double threshold {1e-12 * std::sqrt(frobenius_squared)};
      for (std::size_t sweeps {0}; off_diagonal(a) > threshold; ++sweeps) {
        if (sweeps == max_sweeps) {
          throw convergence_error("symmetric eigendecomposition: Jacobi did not converge within its sweeps");
        }
        for (std::size_t p {0}; p < n_; ++p) {
          for (std::size_t q {p + 1}; q < n_; ++q) {
            sweep(a, v, p, q);
          }
        }
      }
      for (std::size_t i {0}; i < n_; ++i) {
        a[(i * n_) + i] *= scale;
      }
      extract_sorted(a, v);
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

    //! \brief The Frobenius norm of the strict off-diagonal of \p a.
    [[nodiscard]] double off_diagonal(const std::vector<double>& a) const
    {
      double sum {0.0};
      for (std::size_t p {0}; p < n_; ++p) {
        for (std::size_t q {p + 1}; q < n_; ++q) {
          sum += a[(p * n_) + q] * a[(p * n_) + q];
        }
      }
      return std::sqrt(2.0 * sum);
    }

    //! \brief One Jacobi rotation zeroing `a[p][q]`, accumulated into \p v.
    void sweep(std::vector<double>& a,
               std::vector<double>& v,
               std::size_t          p,
               std::size_t          q) const
    {
      const double apq {a[(p * n_) + q]};
      if (apq == 0.0) {
        return; // already zero — nothing to rotate
      }
      const double app {a[(p * n_) + p]};
      const double aqq {a[(q * n_) + q]};
      const double theta {(aqq - app) / (2.0 * apq)};
      const double t {(theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt((theta * theta) + 1.0))};
      const double c {1.0 / std::sqrt((t * t) + 1.0)};
      const double s {t * c};
      a[(p * n_) + p] = (c * c * app) - (2.0 * s * c * apq) + (s * s * aqq);
      a[(q * n_) + q] = (s * s * app) + (2.0 * s * c * apq) + (c * c * aqq);
      a[(p * n_) + q] = 0.0;
      a[(q * n_) + p] = 0.0;
      for (std::size_t i {0}; i < n_; ++i) {
        if (i != p && i != q) {
          const double aip {a[(i * n_) + p]};
          const double aiq {a[(i * n_) + q]};
          a[(i * n_) + p] = (c * aip) - (s * aiq);
          a[(p * n_) + i] = a[(i * n_) + p];
          a[(i * n_) + q] = (s * aip) + (c * aiq);
          a[(q * n_) + i] = a[(i * n_) + q];
        }
      }
      for (std::size_t i {0}; i < n_; ++i) {
        const double vip {v[(i * n_) + p]};
        const double viq {v[(i * n_) + q]};
        v[(i * n_) + p] = (c * vip) - (s * viq);
        v[(i * n_) + q] = (s * vip) + (c * viq);
      }
    }

    //! \brief Sorts eigenvalues ascending and reorders \p v's columns to match.
    void extract_sorted(const std::vector<double>& a,
                        const std::vector<double>& v)
    {
      std::vector<std::size_t> order(n_);
      for (std::size_t i {0}; i < n_; ++i) {
        order[i] = i;
      }
      std::sort(order.begin(), order.end(),
                [&a, this](std::size_t x, std::size_t y) {
                  return a[(x * n_) + x] < a[(y * n_) + y];
                });
      std::vector<double> values(n_);
      std::vector<double> vectors(n_ * n_);
      for (std::size_t col {0}; col < n_; ++col) {
        values[col] = a[(order[col] * n_) + order[col]];
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
      const double epsilon {std::numeric_limits<double>::epsilon()};
      for (std::size_t sweep {1};; ++sweep) {
        bool rotated {false};
        for (std::size_t p {0}; p < n; ++p) {
          for (std::size_t q {p + 1}; q < n; ++q) {
            double alpha {0.0};
            double beta  {0.0};
            T      gamma {};
            for (std::size_t i {0}; i < n; ++i) {
              alpha += std::norm(w[(i * n) + p]);
              beta  += std::norm(w[(i * n) + q]);
              gamma += conj_of(w[(i * n) + p]) * w[(i * n) + q];
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
              std::vector<T>& m {*side};
              for (std::size_t i {0}; i < n; ++i) {
                const T left  {m[(i * n) + p]};
                const T right {turn * m[(i * n) + q]};
                m[(i * n) + p] = (static_cast<T>(c) * left) - (static_cast<T>(s) * right);
                m[(i * n) + q] = (static_cast<T>(s) * left) + (static_cast<T>(c) * right);
              }
            }
          }
        }
        if (!rotated) {
          return;
        }
        if (sweep == max_sweeps) {
          throw convergence_error("SVD: one-sided Jacobi did not converge within its sweeps");
        }
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

  /*!
   * \brief The singular value decomposition `A = U Σ Vᴴ` of an `m × n` matrix.
   * \tparam T The scalar element type (`double` or `std::complex<double>`).
   *
   * With `k = min(m, n)`: the `k` singular values, descending; `U` (`m × k`) and `V` (`n × k`) with
   * orthonormal columns, also where `A` is rank-deficient (the columns for a zero singular value
   * complete the basis). Computed as NumPy's reference LAPACK routes recommend for accuracy: a
   * Householder QR of `A` (of `Aᴴ` when `m < n`), then one-sided Jacobi on `R`, which gives every
   * singular value to high relative accuracy — the small ones too, which a decomposition of `Aᴴ A`
   * loses. \ref vh is `Vᴴ`, the third factor NumPy's `svd` returns.
   */
  template <typename T>
  class svd_decomposition
  {
  public:

    /*!
     * \brief Decomposes \p a. \param[in] a The matrix.
     * \throws non_finite_error If an entry is infinite or NaN.
     * \throws convergence_error If the Jacobi sweeps do not converge (not observed).
     */
    explicit svd_decomposition(const matrix<T>& a)
    {
      detail::require_finite(a, "SVD");
      if (a.rows() >= a.cols()) {
        factor(a);
      }
      else { // A = (Aᴴ)ᴴ = (U' Σ V'ᴴ)ᴴ = V' Σ U'ᴴ
        factor(detail::adjoint(a));
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
    void factor(const matrix<T>& a)
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
      detail::thin_qr<T> factors              {detail::householder_qr(std::move(entries), rows, cols)};
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
    if (b.size() != a.rows()) {
      throw dimension_error("lstsq: right-hand side size differs from the matrix");
    }
    detail::require_finite(a, "lstsq");
    const svd_decomposition<T> s       {a};
    const vector<double>&      sigma   {s.singular_values()};
    const std::size_t          k       {sigma.size()};
    double                     largest {0.0};
    for (const double value : sigma) {
      largest = std::max(largest, value);
    }
    const double               tolerance     {detail::rank_tolerance(largest, a.rows(), a.cols())};
    std::vector<T>             x(a.cols(), T {});
    for (std::size_t d {0}; d < k && sigma[d] > tolerance; ++d) {
      T coefficient {}; // (u_dᴴ b) / σ_d
      for (std::size_t i {0}; i < a.rows(); ++i) {
        coefficient += detail::conj_of(s.u()(i, d)) * b[i];
      }
      coefficient /= static_cast<T>(sigma[d]);
      for (std::size_t j {0}; j < a.cols(); ++j) {
        x[j] += coefficient * s.v()(j, d);
      }
    }
    return vector<T> {std::move(x)};
  }
} // namespace numerical

#endif // NUMERICAL_DECOMPOSITION_HPP
