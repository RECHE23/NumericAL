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
#include <bit>
#include <cmath>
#include <complex> // so std::abs resolves for a complex scalar, regardless of include order
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
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

    //! \brief The order up to which the LU takes panels of \ref lu_small_block columns: there the panels, a column
    //!        at a time on one thread, are most of the time, and narrower ones halve their work.
    inline constexpr std::size_t lu_small_order {256};

    //! \brief The columns of one panel of the LU of an order up to \ref lu_small_order.
    inline constexpr std::size_t lu_small_block {32};

    //! \brief The panel width of the LU of order \p n: a power of two, so a row's panel is a mask away, not a division.
    [[nodiscard]] constexpr std::size_t lu_block_for(std::size_t n) noexcept
    {
      static_assert(std::has_single_bit(lu_small_block) && std::has_single_bit(lu_block), "a panel is a mask away");
      return n <= lu_small_order ? lu_small_block : lu_block;
    }

    //! \brief The trailing columns at most past which the LU factorises its next panel beside the update of the rest
    //!        (a look-ahead): a wider update runs well as one product, and cut in blocks beside the panel it ran
    //!        longer (order 2048 on six x86-64 cores, 9 %); a narrower one leaves threads idle while the panel runs.
    inline constexpr std::size_t lu_lookahead_columns {1024};

    //! \brief The multiply-adds of an LU's trailing update one of the tile's dependent ones stands for, as threads are
    //!        weighed (\ref gemm_kernel_speedup): the AMX tile's, on arm64; the AVX tiles' quarter of it on x86-64,
    //!        where the update of order 256 waited on one thread (six cores: 690 → 622 µs, 1024: 12.5 → 12.1 ms).
#ifdef NUMERICAL_X86_COMPILED
    inline constexpr std::size_t lu_update_speedup {gemm_kernel_speedup / 4};
#else
    inline constexpr std::size_t lu_update_speedup {gemm_kernel_speedup};
#endif

    //! \brief The most right-hand sides an LU solve takes row by row rather than by blocks through the product kernel.
    inline constexpr std::size_t narrow_solve {16};

    //! \brief The fewest columns an LU solve gives a thread of its own: a narrower slice starves the product kernel
    //!        (arm64, order 256 and 64 right-hand sides in eight slices: 112 → 194 µs).
    inline constexpr std::size_t slice_columns {64};

#ifdef NUMERICAL_X86_COMPILED
    //! \brief The order up to which an LU solve gives each thread its own columns on the x86-64 units: each thread
    //!        rereads the whole factor, which the memory stops paying for past it (six AVX2 cores, an inverse of order
    //!        768 6 % slower, 1024 17 %). On arm64 the slices win at every order measured (to 1024).
    inline constexpr std::size_t slice_solve_order {384};
#endif

    //! \brief The order up to which an LU solve goes row by row off the x86-64 units: past it the blocked products win
    //!        (arm64, four columns: order 128 49 → 37 µs, 192 93 → 86; order 256 145 → 160).
    inline constexpr std::size_t narrow_solve_order {192};

    //! \brief The columns of one leaf of the recursive LU panel.
    inline constexpr std::size_t solve_leaf {16};

#ifdef NUMERICAL_X86_COMPILED
    //! \brief The rows an LU panel's copy takes together, so that it writes each column of the panel eight `double`s,
    //!        a 64-byte line, at a time, where row by row each write fell on another line (six AVX2 cores, order 768:
    //!        5.08 → 4.74 ms; tiles of 16 or 32 rows were slower again, order 512 by 7 %).
    inline constexpr std::size_t panel_copy_rows {8};
#else
    //! \brief The rows an LU panel's copy takes together: one, row by row, on arm64, where tiles of rows were no
    //!        faster (eight cores, order 256: tiles of 8 rows 3 % slower, of 16 2 %).
    inline constexpr std::size_t panel_copy_rows {1};
#endif

    //! \brief The columns of one panel of the blocked Cholesky factorisation (as \ref lu_block).
    inline constexpr std::size_t cholesky_block {64};

    //! \brief The order below which the Cholesky factorisation is one panel: no trailing product, whose packing the
    //!        few panels of a small order do not repay (order 17 on one thread: 0.8 µs whole, 1.3 µs in panels of 16).
    //!        Panels of 16 win from order 40 on x86-64 and from 48 on arm64.
#ifdef NUMERICAL_X86_COMPILED
    inline constexpr std::size_t cholesky_one_panel {40};
#else
    inline constexpr std::size_t cholesky_one_panel {48};
#endif
    static_assert(cholesky_one_panel <= cholesky_block, "one panel of cholesky_block columns holds the whole matrix");

    //! \brief The panel width of the Cholesky factorisation of order \p n: narrower up to order 256, where the
    //!        panels, row by row on one thread, are most of the time; a power of two, which the product kernels'
    //!        tiles divide (widths of 24, 40 or 56 left the AMX's 32 × 16 tiles part empty: order 320, 20 % longer).
    [[nodiscard]] constexpr std::size_t cholesky_block_for(std::size_t n) noexcept
    {
      if (n < cholesky_one_panel) {
        return cholesky_block;
      }
      if (n <= 128) {
        return cholesky_block / 4;
      }
      return n <= 256 ? cholesky_block / 2 : cholesky_block;
    }

    //! \brief The rows of one chunk of a Cholesky panel's rows below it, solved together along each column.
    inline constexpr std::size_t cholesky_rows {64};

    //! \brief The fewest columns of the trailing block one Cholesky update's product takes: each product packs every
    //!        row below its columns again, a panel's width apart the packing outweighed the products on the AMX
    //!        (M1 Pro, order 128: 61 → 53 µs), and the triangle a wider block computes past the diagonal is wasted.
    inline constexpr std::size_t cholesky_update_columns {64};


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

    //! \brief The exact power of two that brings a magnitude below 2⁻⁵⁰⁰ up among normal numbers, as LAPACK's dlarfg
    //!        rescales; 1 for any larger magnitude.
    [[nodiscard]] inline double rescue_factor(double largest)
    {
      return largest < 0x1p-500 ? std::ldexp(1.0, -std::ilogb(std::max(largest, std::numeric_limits<double>::min()))) : 1.0;
    }

    /*!
     * \brief The unit-modulus factor of \p x: `x / |x|` (its sign, for a real scalar), and 1 at 0. A subnormal `x` is
     *        scaled up first, exactly: its modulus would round to the coarse subnormal grid, and `x / |x|` miss unit
     *        modulus by as much.
     */
    template <typename T>
    [[nodiscard]] T phase_of(const T& x)
    {
      const T      scaled    {x * detail::from_double<T>(rescue_factor(static_cast<double>(std::abs(x))))};
      const double magnitude {static_cast<double>(std::abs(scaled))};
      return magnitude == 0.0 ? T {1} : scaled / detail::from_double<T>(magnitude);
    }

    /*!
     * \brief Scales \p x by \ref rescue_factor of its largest magnitude, exactly, so that a Householder vector built
     *        from it is computed among normal numbers: a subnormal norm rounds to the coarse subnormal grid, and the
     *        reflector is then not orthogonal. \return The factor (1 unless the entries are all below 2⁻⁵⁰⁰).
     */
    template <typename T>
    double rescale_tiny(buffer<T>& x)
    {
      double largest {0.0};
      for (const T& element : x) {
        largest = std::max(largest, static_cast<double>(std::abs(element)));
      }
      const double factor {rescue_factor(largest)};
      for (T& element : x) {
        element *= detail::from_double<T>(factor);
      }
      return factor;
    }

    //! \brief The Euclidean norm of the \p count elements at \p x, summed at the scale of their largest magnitude:
    //!        a plain sum of squares overflows near 1e155 and underflows to 0 near 1e-155.
    template <typename T>
    [[nodiscard]] double norm_at_scale(const T*    x,
                                       std::size_t count)
    {
      double largest {0.0};
      for (std::size_t i {0}; i < count; ++i) {
        largest = std::max(largest, static_cast<double>(std::abs(x[i])));
      }
      if (largest == 0.0) { // the entries are finite: require_finite runs where every caller starts
        return largest;
      }
      double sum {0.0};
      for (std::size_t i {0}; i < count; ++i) {
        const double ratio {static_cast<double>(std::abs(x[i])) / largest};
        sum += ratio * ratio;
      }
      return largest * std::sqrt(sum);
    }

    //! \brief The smallest plain sum of squares \ref plain_or_at_scale keeps: a term that underflowed is below
    //!        2⁻¹⁰²², so a sum this large loses at most `count · 2⁻¹²²` of itself to them.
    inline constexpr double plain_squares_low {0x1p-900};

    //! \brief The largest plain sum of squares \ref plain_or_at_scale keeps: no term of it overflowed.
    inline constexpr double plain_squares_high {0x1p900};

    //! \brief `√plain`, the plain sum of the squares of the \p count elements at \p x, where it neither overflowed
    //!        nor underflowed (between \ref plain_squares_low and \ref plain_squares_high); else \ref norm_at_scale.
    template <typename T>
    [[nodiscard]] double plain_or_at_scale(double      plain,
                                           const T*    x,
                                           std::size_t count)
    {
      if (plain >= plain_squares_low && plain <= plain_squares_high) {
        return std::sqrt(plain);
      }
      return norm_at_scale(x, count);
    }

    //! \brief The Euclidean norm of the \p count elements at \p x, by \ref norm_at_scale for this type.
    template <typename T>
    [[nodiscard]] double scaled_norm(const T*    x,
                                     std::size_t count)
    {
      return norm_at_scale(x, count);
    }

    //! \brief The same for `double`: the plain sum of squares in one pass of \ref dot_product where it holds.
    [[nodiscard]] inline double scaled_norm(const double* x,
                                            std::size_t   count)
    {
      return plain_or_at_scale(dot_product(x, x, count), x, count);
    }

    //! \brief The same for `std::complex<double>`: \ref dot_product over the parts, which an array of them holds in
    //!        order ([complex.numbers]).
    [[nodiscard]] inline double scaled_norm(const std::complex<double>* x,
                                            std::size_t                 count)
    {
      const auto* const parts {reinterpret_cast<const double*>(x)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
      return plain_or_at_scale(dot_product(parts, parts, 2 * count), x, count);
    }

    //! \brief \ref scaled_norm of a whole vector.
    template <typename T>
    [[nodiscard]] double scaled_norm(const buffer<T>& x)
    {
      return scaled_norm(x.data(), x.size());
    }

    /*!
     * \brief `√(a² + b²)` for a plane rotation of the QR and QL iterations, as LAPACK's `dlartg` takes it: the plain
     *        sum of squares where the larger magnitude lies within `2^±500`, so its square neither overflows nor
     *        underflows and the smaller one's underflow is below its rounding; `std::hypot`, which scales every time,
     *        outside. A graded matrix reaches both ends though scaled to its largest entry: a QL shift's ratio
     *        `(d₁ − d₀) / 2e` came to 5e199 on `[[1e-200, 1e-200], [1e-200, 1]]`.
     */
    [[nodiscard]] inline double rotation_length(double a,
                                                double b) noexcept
    {
      constexpr double low    {0x1p-500};
      constexpr double high   {0x1p500};
      const double     larger {std::max(std::abs(a), std::abs(b))};
      if (larger > low && larger < high) {
        return std::sqrt((a * a) + (b * b));
      }
      return std::hypot(a, b);
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

    //! \brief Whether the \p count values at \p x are all finite: a NaN or an infinity has every exponent bit set. One
    //!        pass with no early exit, an integer OR the compilers vectorise (a check per element and an exit at the
    //!        first was a tenth of an LU of order 128).
    [[nodiscard]] inline bool all_finite(const double* x,
                                         std::size_t   count) noexcept
    {
      constexpr std::uint64_t exponent {0x7FF0000000000000ULL};
      std::uint64_t           flagged  {0};
      for (std::size_t i {0}; i < count; ++i) {
        std::uint64_t bits {};
        std::memcpy(&bits, x + i, sizeof(bits));
        flagged |= static_cast<std::uint64_t>((bits & exponent) == exponent);
      }
      return flagged == 0;
    }

    //! \brief \ref require_finite for `double`, its elements contiguous: \ref all_finite.
    inline void require_finite(const matrix<double>& a,
                               const char*           what)
    {
      if (!all_finite(std::to_address(a.begin()), a.rows() * a.cols())) {
        throw non_finite_error(std::string {what} + ": the matrix holds an infinite or NaN entry");
      }
    }

    //! \brief The same for `std::complex<double>`, whose array holds its parts in order ([complex.numbers]).
    inline void require_finite(const matrix<std::complex<double>>& a,
                               const char*                         what)
    {
      const auto* const parts {reinterpret_cast<const double*>(std::to_address(a.begin()))}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
      if (!all_finite(parts, 2 * a.rows() * a.cols())) {
        throw non_finite_error(std::string {what} + ": the matrix holds an infinite or NaN entry");
      }
    }

    //! \brief \ref require_finite over what a Hermitian eigensolver reads: the lower triangle, and the diagonal's real
    //!        part (as `numpy.linalg.eigh` reads them, whatever the rest holds).
    //! \param[in] a    The square input.
    //! \param[in] what The decomposition, for the message.
    //! \throws non_finite_error If an entry read is infinite or NaN.
    template <typename T>
    void require_finite_lower(const matrix<T>& a,
                              const char*      what)
    {
      for (std::size_t i {0}; i < a.rows(); ++i) {
        bool finite {std::isfinite(real_part(a(i, i)))};
        for (std::size_t j {0}; j < i; ++j) {
          finite = finite && is_finite_entry(a(i, j));
        }
        if (!finite) {
          throw non_finite_error(std::string {what} + ": the matrix holds an infinite or NaN entry");
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
      : lu_decomposition(matrix<T> {a})
    {}

    /*!
     * \brief Factorises \p a in its own storage, an expiring matrix's elements taken over: no copy of them.
     * \param[in] a The square matrix to factorise.
     * \throws dimension_error If \p a is not square.
     */
    explicit lu_decomposition(matrix<T>&& a)
      : n_(a.rows())
    {
      if (a.cols() != n_) {
        throw dimension_error("LU decomposition requires a square matrix");
      }
      detail::require_finite(a, "LU decomposition");
      lu_ = std::move(a).elements();
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
      buffer<T> y(n_, T {});
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
     * \brief The inverse, `U⁻¹ L⁻¹ P`, by blocks of rows through the matrix product kernel: `L⁻¹` from the identity,
     *        whose row `i` stays zero past column `i` in the forward pass, so that pass reads only the triangle it
     *        fills (a third of the products of a full one, as LAPACK's `dgetri` spares them); then `U⁻¹` on it, and
     *        `P` as a permutation of the columns. The terms left out are products with zeros.
     * \return The inverse matrix.
     * \throws singular_matrix_error If the matrix is singular.
     */
    [[nodiscard]] matrix<T> inverse() const
    {
      if (singular_) {
        throw singular_matrix_error("inverse: matrix is singular");
      }
      buffer<T> rows(n_ * n_, T {});
      for (std::size_t i {0}; i < n_; ++i) {
        rows[(i * n_) + i] = T {1};
      }
      solve_rows(rows, n_, true);
      buffer<T> inverted(n_ * n_, T {});
      detail::parallel_for(n_, n_, [&](std::size_t r) { // column pivots_[i] of A⁻¹ is column i of U⁻¹ L⁻¹
                             for (std::size_t i {0}; i < n_; ++i) {
                               inverted[(r * n_) + pivots_[i]] = rows[(r * n_) + i];
                             }
                           });
      return matrix<T> {n_, n_, std::move(inverted)};
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
      const std::size_t cols              {b.cols()};
      buffer<T>         rows(n_ * cols, T {});
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
     *        `block_` rows: within a block the rows are solved one after another, in parallel chunks of
     *        columns (each chunk's columns depend on no other's); from a block to the rows beyond it, through the
     *        matrix product kernel. When \p lower, row `i` of \p y is zero past column `i` (the identity's rows):
     *        the forward pass then works on those columns alone. Many columns go to the threads by slices
     *        (\ref column_slices), each through every block (\ref solve_columns); a column's terms come in the same
     *        order either way.
     */
    void solve_rows(buffer<T>&      y,
                    std::size_t     cols,
                    bool            lower = false) const
    {
      if constexpr (std::is_same_v<T, double>) { // there subtract_products holds its row in registers through the terms
        if (!lower && cols <= detail::narrow_solve && narrow_pays()) {
          solve_narrow(y, cols);
          return;
        }
      }
      const std::size_t slices {column_slices(cols)};
      if (slices > 1) { // independent right-hand sides: each thread its own columns, all the blocks, no join per block
        const std::size_t slice {(((cols + slices - 1) / slices) + 7) / 8 * 8};
        detail::parallel_for((cols + slice - 1) / slice, n_ * n_ * slice / detail::gemm_kernel_speedup, [&](std::size_t s) {
                               solve_columns(y, cols, s * slice, std::min(cols, (s + 1) * slice), lower);
                             });
        return;
      }
      solve_columns(y, cols, 0, cols, lower);
    }

    /*!
     * \brief The column slices \ref solve_rows splits \p cols right-hand sides into, one a thread, each of
     *        \ref detail::slice_columns or more (on the x86-64 units, to \c detail::slice_solve_order): the blocks of rows then join
     *        the threads once, where each block joined them for products too small to share (six AVX2 cores, an
     *        inverse of order 256: 1.87 → 1.60 ms; arm64, eight cores, 698 → 503 µs).
     */
    [[nodiscard]] std::size_t column_slices(std::size_t cols) const
    {
#ifdef NUMERICAL_X86_COMPILED
      if (n_ > detail::slice_solve_order) {
        return 1;
      }
#endif
      if (cols < 2 * detail::slice_columns) {
        return 1;
      }
      return std::min(max_threads(), cols / detail::slice_columns);
    }

    /*!
     * \brief \ref solve_rows on the columns `[c0, c1)` of \p y (`n × cols`, row-major): the blocked passes, their
     *        products through the matrix kernel, each element's terms in the same order whatever the slice.
     */
    void solve_columns(buffer<T>&      y,
                       std::size_t     cols,
                       std::size_t     c0,
                       std::size_t     c1,
                       bool            lower) const
    {
      constexpr std::size_t width  {64};
      const std::size_t     chunks {(c1 - c0 + width - 1) / width};
      for (std::size_t k0 {0}; k0 < n_; k0 += block_) {           // forward: unit-lower L
        const std::size_t end   {std::min(k0 + block_, n_)};
        const std::size_t reach {lower ? std::min(end, c1) : c1}; // the block's rows' columns that can be nonzero
        if (reach <= c0) {
          continue; // the slice's columns are zero in these rows and every row past them
        }
        detail::parallel_for((reach - c0 + width - 1) / width, (end - k0) * (end - k0) * width / 2, [&](std::size_t chunk) {
                               const std::size_t first {c0 + (chunk * width)};
                               const std::size_t last {std::min(first + width, reach)};
                               for (std::size_t i {k0 + 1}; i < end; ++i) {
                                 detail::subtract_products(y.data() + (i * cols) + first, lu_.data() + (i * n_) + k0, y.data() + (k0 * cols) + first, cols, i - k0, last - first);
                               }
                             });
        subtract_block(y, cols, end, n_, k0, end, c0, reach);
      }
      for (std::size_t k1 {n_}; k1 > 0;) { // backward: upper U, the blocks of the forward pass, last first
        const std::size_t k0 {panel_of(k1 - 1)};
        detail::parallel_for(chunks, (k1 - k0) * (k1 - k0) * width / 2, [&](std::size_t chunk) {
                               const std::size_t first {c0 + (chunk * width)};
                               const std::size_t last {std::min(first + width, c1)};
                               for (std::size_t i {k1}; i-- > k0;) {
                                 detail::subtract_products(y.data() + (i * cols) + first, lu_.data() + (i * n_) + i + 1, y.data() + ((i + 1) * cols) + first, cols, k1 - (i + 1), last - first);
                                 const T pivot {lu_[(i * n_) + i]};
                                 for (std::size_t c {first}; c < last; ++c) {
                                   y[(i * cols) + c] = y[(i * cols) + c] / pivot;
                                 }
                               }
                             });
        subtract_block(y, cols, 0, k0, k0, k1, c0, c1);
        k1 = k0;
      }
    }

    //! \brief Whether \ref solve_narrow beats the blocked passes here: on the x86-64 units at any order (the portable
    //!        path's products there fuse nothing, the rows would), elsewhere up to \ref detail::narrow_solve_order.
    [[nodiscard]] bool narrow_pays() const
    {
#ifdef NUMERICAL_X86_COMPILED
      return detail::gemm_path_current() != gemm_path::portable;
#else
      return n_ <= detail::narrow_solve_order;
#endif
    }

    /*!
     * \brief \ref solve_rows for a few right-hand sides: row by row, each one's terms in the order the blocked passes
     *        give them (forward, ascending; backward, the blocks from the last down to its own, each ascending),
     *        without a product that packs the factor's blocks for a few columns (order 256 and four columns on one
     *        x86-64 core: 112 → 72 µs, the factorisation 750). The solution is the blocked passes' to the bit wherever
     *        their products run the matrix kernel; one the plain loop takes (under \ref detail::gemm_packed_threshold
     *        multiply-adds, or shallower than \ref detail::gemm_min_depth, as a last block of a few rows) rounds each
     *        term apart where the x86-64 rows fuse it, so there the two differ in the last bits.
     */
    void solve_narrow(buffer<T>&      y,
                      std::size_t     cols) const
    {
      for (std::size_t i {1}; i < n_; ++i) {
        detail::subtract_products(y.data() + (i * cols), lu_.data() + (i * n_), y.data(), cols, i, cols);
      }
      const std::size_t last {panel_of(n_ - 1)};
      for (std::size_t i {n_}; i-- > 0;) {
        const std::size_t own {panel_of(i)};
        for (std::size_t b0 {last};; b0 -= block_) {
          const std::size_t from {std::max(b0, i + 1)};
          const std::size_t to   {std::min(b0 + block_, n_)};
          if (from < to) {
            detail::subtract_products(y.data() + (i * cols), lu_.data() + (i * n_) + from, y.data() + (from * cols), cols, to - from, cols);
          }
          if (b0 == own) {
            break;
          }
        }
        const T pivot {lu_[(i * n_) + i]};
        for (std::size_t c {0}; c < cols; ++c) {
          y[(i * cols) + c] = y[(i * cols) + c] / pivot;
        }
      }
    }

    //! \brief The first row of the panel that holds row \p i.
    [[nodiscard]] std::size_t panel_of(std::size_t i) const noexcept
    {
      return i & ~(block_ - 1);
    }

    //! \brief `Y[r0, r1) −= F[r0, r1) × [c0, c1) · Y[c0, c1)`, F the factor, through the matrix product kernel (nothing
    //!        when the rows are empty), on the columns `[first, last)` of `Y`, `first < last`.
    void subtract_block(buffer<T>&      y,
                        std::size_t     cols,
                        std::size_t     r0,
                        std::size_t     r1,
                        std::size_t     c0,
                        std::size_t     c1,
                        std::size_t     first,
                        std::size_t     last) const
    {
      if (r0 < r1) {
        const buffer<T> minus {negated_block(r0, r1, c0, c1)};
        detail::gemm(r1 - r0, last - first, c1 - c0, minus.data(), c1 - c0, y.data() + (c0 * cols) + first, cols,
                     y.data() + (r0 * cols) + first, cols);
      }
    }

    //! \brief The rows `[r0, r1)` and columns `[c0, c1)` of the factor, negated, row-major and contiguous.
    [[nodiscard]] buffer<T> negated_block(std::size_t r0,
                                          std::size_t r1,
                                          std::size_t c0,
                                          std::size_t c1) const
    {
      buffer<T> out((r1 - r0) * (c1 - c0), T {});
      for (std::size_t r {r0}; r < r1; ++r) {
        for (std::size_t c {c0}; c < c1; ++c) {
          out[((r - r0) * (c1 - c0)) + (c - c0)] = -lu_[(r * n_) + c];
        }
      }
      return out;
    }

    // Right-looking, by panels of block_ columns, as LAPACK's dgetrf: each panel is factorised
    // with partial pivoting (whole rows swapped), its rows of U to the right are solved against the panel's
    // unit-lower L, and the trailing block takes the product of the two through detail::gemm, where the
    // work of a large factorisation is. Each element still receives its updates in ascending order of the
    // pivot, so without multiply-add fusion the factors are the unblocked algorithm's to the bit; an order
    // up to one panel is that algorithm.
    //
    // With a look-ahead of one panel: the trailing update reaches the next panel's columns first, then the
    // next panel is factorised on the calling thread while the workers update the columns right of it (the
    // panel, a column at a time, would otherwise hold every other thread idle). Its row swaps reach those
    // columns once both are done: an update reads its rows of L from a copy and writes row by row, so a swap
    // after it is the swap before it, and the factors stay the ones above, to the bit.
    //
    // A pivot's swap reaches the columns left of its panel only at the end (\ref swap_left_columns): nothing reads
    // them before, and on the calling thread each swap held up the panel the other threads wait for.
    void factorize()
    {
      std::size_t end {std::min(block_, n_)};
      factorize_panel(0, end);
      for (std::size_t k0 {0}; end < n_;) {
        const std::size_t next       {std::min(end + block_, n_)};
        solve_panel_rows(k0, end);
        const buffer<T> minus_l      {negated_block(end, n_, k0, end)};
        update_columns(minus_l, k0, end, end, next);
        const std::size_t right      {n_ - next};
        if (right > detail::lu_lookahead_columns) { // one product of the whole trailing block, then the panel
          update_columns(minus_l, k0, end, next, n_);
          factorize_panel(end, next);
          k0  = end;
          end = next;
          continue;
        }
        // About two blocks of columns for each thread the work pays for: each block's product packs L₂₁ again, and a
        // product too small to share ran one block for each thread the machine has (order 256 on eight arm64
        // threads: 329 → 311 µs); two, not one, as on x86-64 a block of half the columns stays in its cache.
        const std::size_t work    {(n_ - end) * right * (end - k0) / detail::lu_update_speedup};
        const std::size_t helpers {std::clamp(work / detail::work_per_thread, std::size_t {1}, max_threads())};
        const std::size_t width   {std::max(block_, (right + (2 * helpers) - 1) / (2 * helpers))};
        const std::size_t blocks  {(right + width - 1) / width};
        swap_limit_ = next;
        detail::parallel_for_beside(blocks, (n_ - end) * width * (end - k0) / detail::lu_update_speedup,
                                    [&](std::size_t block) {
                                      const std::size_t first {next + (block * width)};
                                      update_columns(minus_l, k0, end, first, std::min(first + width, n_));
                                    },
                                    [&] { factorize_panel(end, next); });
        swap_limit_ = n_;
        for (const auto& [a, b] : deferred_swaps_) {
          for (std::size_t c {next}; c < n_; ++c) {
            std::swap(lu_[(a * n_) + c], lu_[(b * n_) + c]);
          }
        }
        deferred_swaps_.clear();
        k0  = end;
        end = next;
      }
      swap_left_columns();
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
      const std::size_t rows                  {n_ - k0};
      const std::size_t width                 {end - k0};
      buffer<T>         panel(rows * width, T {}); // column j of the panel at panel[j * rows]
      transpose_panel(panel, rows, k0, end, true);
      factorize_columns(panel, rows, 0, width, k0, end);
      transpose_panel(panel, rows, k0, end, false);
    }

    //! \brief Copies the columns `[k0, end)` of the factor, every row from `k0`, into \p panel (column-major, \p rows
    //!        rows) when \p in, back otherwise, by tiles of \ref detail::panel_copy_rows rows.
    void transpose_panel(buffer<T>&      panel,
                         std::size_t     rows,
                         std::size_t     k0,
                         std::size_t     end,
                         bool            in)
    {
      const std::size_t width {end - k0};
      T* const          base  {lu_.data() + (k0 * n_) + k0};
      for (std::size_t r0 {0}; r0 < rows; r0 += detail::panel_copy_rows) {
        const std::size_t r1 {std::min(r0 + detail::panel_copy_rows, rows)};
        if (in) {
          for (std::size_t j {0}; j < width; ++j) {
            for (std::size_t r {r0}; r < r1; ++r) {
              panel[(j * rows) + r] = base[(r * n_) + j];
            }
          }
        }
        else {
          for (std::size_t j {0}; j < width; ++j) {
            for (std::size_t r {r0}; r < r1; ++r) {
              base[(r * n_) + j] = panel[(j * rows) + r];
            }
          }
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
    void factorize_columns(buffer<T>&      panel,
                           std::size_t     rows,
                           std::size_t     first,
                           std::size_t     last,
                           std::size_t     k0,
                           std::size_t     end)
    {
      // A leaf, or a panel too short for its products to pay for packing (order 64 on arm64: 30 µs unblocked,
      // 37 recursive).
      if (last - first <= detail::solve_leaf || rows - first < 2 * block_) {
        for (std::size_t j {first}; j < last; ++j) {
          T           *          column {panel.data() + (j * rows)};
          std::size_t            pivot  {0}; // from row j
          detail::magnitude_t<T> best   {std::abs(column[j])};
          for (std::size_t r {detail::largest_magnitude_vectors(column + j, rows - j, best, pivot)}; r < rows - j; ++r) {
            const detail::magnitude_t<T> mag {std::abs(column[j + r])};
            if (mag > best) {
              best  = mag;
              pivot = r;
            }
          }
          pivot += j;
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
      const std::size_t depth                           {mid - first};
      buffer<T>         minus_u((last - mid) * depth, T {});
      for (std::size_t c {mid}; c < last; ++c) {
        for (std::size_t j {first}; j < mid; ++j) {
          minus_u[((c - mid) * depth) + (j - first)] = -panel[(c * rows) + j];
        }
      }
      detail::gemm(last - mid, rows - mid, depth, minus_u.data(), depth, panel.data() + (first * rows) + mid, rows,
                   panel.data() + (mid * rows) + mid, rows);
      factorize_columns(panel, rows, mid, last, k0, end);
    }

    //! \brief Swaps rows \p a and \p b of the factor outside the panel's columns `[k0, end)`: right of the panel up
    //!        to \ref swap_limit_, the rest left in \ref deferred_swaps_ (columns another thread is updating); left of
    //!        it, in \ref left_swaps_, for \ref swap_left_columns.
    void swap_rows_outside(std::size_t a,
                           std::size_t b,
                           std::size_t k0,
                           std::size_t end)
    {
      if (k0 > 0) {
        left_swaps_.emplace_back(a, b);
      }
      for (std::size_t c {end}; c < swap_limit_; ++c) {
        std::swap(lu_[(a * n_) + c], lu_[(b * n_) + c]);
      }
      if (swap_limit_ < n_) {
        deferred_swaps_.emplace_back(a, b);
      }
    }

    /*!
     * \brief Applies \ref left_swaps_ to the columns left of each swap's panel: a panel's columns take, in order, the
     *        swaps of the panels after it, which are the swaps from its last column on. The panels' columns are
     *        independent, so they go to the threads; a column sees its swaps in the order the pivots were chosen.
     */
    void swap_left_columns()
    {
      if (left_swaps_.empty()) {
        return;
      }
      const std::size_t panels {(n_ - 1) / block_}; // every panel but the last, which no later swap reaches
      detail::parallel_for(panels, left_swaps_.size() * block_, [&](std::size_t p) {
                             const std::size_t c0 {p * block_};
                             const std::size_t c1 {c0 + block_};
                             const auto first {std::lower_bound(left_swaps_.begin(), left_swaps_.end(), c1,
                                                                [](const auto& swap, std::size_t row) { return swap.first < row; })};
                             for (auto swap {first}; swap != left_swaps_.end(); ++swap) {
                               std::swap_ranges(lu_.data() + (swap->first * n_) + c0, lu_.data() + (swap->first * n_) + c1,
                                                lu_.data() + (swap->second * n_) + c0);
                             }
                           });
      left_swaps_.clear();
    }

    //! \brief `U₁₂ = L₁₁⁻¹ A₁₂`: the panel's rows right of it, by forward substitution with its unit-lower L. Each
    //!        block of columns is independent, so blocks run in parallel, each in the sequential order: each element
    //!        still takes its terms in ascending order of the pivot.
    void solve_panel_rows(std::size_t k0,
                          std::size_t end)
    {
      constexpr std::size_t width  {64};
      const std::size_t     blocks {((n_ - end) + width - 1) / width};
      detail::parallel_for(blocks, (end - k0) * (end - k0) * width / 2, [&](std::size_t block) {
                             const std::size_t first {end + (block * width)};
                             const std::size_t last {std::min(first + width, n_)};
                             for (std::size_t i {k0 + 1}; i < end; ++i) {
                               detail::subtract_products(lu_.data() + (i * n_) + first, lu_.data() + (i * n_) + k0, lu_.data() + (k0 * n_) + first, n_, i - k0, last - first);
                             }
                           });
    }

    //! \brief `A₂₂ −= L₂₁ U₁₂` on the trailing columns `[c0, c1)`, as `A₂₂ += (−L₂₁) U₁₂` through the matrix product
    //!        kernel, \p minus_l being `−L₂₁` (the rows below the panel `[k0, end)`).
    void update_columns(const buffer<T>&      minus_l,
                        std::size_t           k0,
                        std::size_t           end,
                        std::size_t           c0,
                        std::size_t           c1)
    {
      detail::gemm(n_ - end, c1 - c0, end - k0, minus_l.data(), end - k0, lu_.data() + (k0 * n_) + c0, n_,
                   lu_.data() + (end * n_) + c0, n_);
    }

    std::size_t                                      n_;                                //!< The matrix order.
    std::size_t                                      block_ {detail::lu_block_for(n_)}; //!< The panel width.
    buffer<T>                                        lu_;                               //!< Combined L (below diag, unit) and U (on/above), row-major.
    std::vector<std::size_t>                         pivots_;                           //!< Row permutation: row now at i came from pivots_[i].
    int                                              sign_       {1};                   //!< Permutation sign, for the determinant.
    bool                                             singular_   {false};               //!< Set if a zero pivot was met.
    std::size_t                                      swap_limit_ {n_};                  //!< The columns a panel's row swaps reach now.
    std::vector<std::pair<std::size_t, std::size_t>> deferred_swaps_;                   //!< Swaps left for the columns past it.
    std::vector<std::pair<std::size_t, std::size_t>> left_swaps_;                       //!< Swaps left for the columns before their panel.
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
  [[nodiscard]] vector<T> solve(matrix<T>        a,
                                const vector<T>& b)
  {
    return lu_decomposition<T> {std::move(a)}.solve(b);
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
  [[nodiscard]] matrix<T> solve(matrix<T>        a,
                                const matrix<T>& b)
  {
    return lu_decomposition<T> {std::move(a)}.solve(b);
  }

  //! \brief The determinant of a square matrix. \throws dimension_error If not square.
  template <typename T>
  [[nodiscard]] T determinant(matrix<T> a)
  {
    return lu_decomposition<T> {std::move(a)}.determinant();
  }

  /*!
   * \brief The inverse of a square matrix.
   * \param[in] a The matrix.
   * \return Its inverse.
   * \throws dimension_error If `a` is not square.
   * \throws singular_matrix_error If `a` is singular.
   */
  template <typename T>
  [[nodiscard]] matrix<T> inverse(matrix<T> a)
  {
    return lu_decomposition<T> {std::move(a)}.inverse();
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
      : cholesky_decomposition(matrix<T> {a})
    {}

    /*!
     * \brief Factorises \p a in its own storage, an expiring matrix's elements taken over: no copy of them.
     * \param[in] a The Hermitian positive-definite matrix.
     * \throws dimension_error If \p a is not square.
     * \throws not_positive_definite_error If \p a is not positive-definite.
     */
    explicit cholesky_decomposition(matrix<T>&& a)
      : n_(a.rows())
    {
      if (a.cols() != n_) {
        throw dimension_error("Cholesky decomposition requires a square matrix");
      }
      detail::require_finite(a, "Cholesky decomposition");
      lower_ = std::move(a).elements(); // the upper triangle is never read, and zeroed once factorised
      factorize();
    }

    //! \brief The lower-triangular factor `L` (with `L Lᴴ == A`).
    [[nodiscard]] matrix<T> lower() const&
    {
      return matrix<T> {n_, n_, lower_};
    }

    //! \brief The lower-triangular factor `L`, moved out of an expiring decomposition.
    [[nodiscard]] matrix<T> lower() &&
    {
      return matrix<T> {n_, n_, std::move(lower_)};
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
      buffer<T> y(n_, T {});
      for (std::size_t i {0}; i < n_; ++i) { // forward: L y = b
        T sum {b[i]};
        for (std::size_t k {0}; k < i; ++k) {
          sum -= lower_[(i * n_) + k] * y[k];
        }
        y[i] = sum / lower_[(i * n_) + i];
      }
      buffer<T> x(n_, T {});
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

    // By panels of block_ columns: the panel's diagonal block row by row, the rows below it
    // by leaves (each depends only on the diagonal block; factor_rows_below), then the lower part of the trailing block takes
    // A22 += (-L21) L21ᴴ from the matrix product kernel. Every element still receives its terms in ascending
    // order, so without multiply-add fusion the factor is the unblocked algorithm's to the bit.
    void factorize()
    {
      for (std::size_t k0 {0}; k0 < n_; k0 += block_) {
        const std::size_t end {std::min(k0 + block_, n_)};
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
     * \brief The rows below the panel `[k0, end)`, `L₂₁ = A₂₁ L₁₁⁻ᴴ`: chunks of \ref detail::cholesky_rows rows in
     *        parallel, each in a column-major copy where column `j` takes `L[i][j] −= L[i][k]·conj(L[j][k])` for
     *        `k` ascending, then the division by `L[j][j]`, for every row of the chunk at once along the column
     *        (\ref detail::subtract_product): each element the unblocked step's arithmetic, in its order; row by
     *        row, each sum waited on its last addition.
     */
    void factor_rows_below(std::size_t k0,
                           std::size_t end)
    {
      const std::size_t width  {end - k0};
      const std::size_t chunks {((n_ - end) + detail::cholesky_rows - 1) / detail::cholesky_rows};
      detail::parallel_for(chunks, detail::cholesky_rows * width * width / 2, [&](std::size_t chunk) {
                             const std::size_t first {end + (chunk * detail::cholesky_rows)};
                             const std::size_t height {std::min(detail::cholesky_rows, n_ - first)};
                             buffer<T>    panel(height * width, T {}); // column j at panel[j * height]
                             for (std::size_t r {0}; r < height; ++r) {
                               for (std::size_t j {0}; j < width; ++j) {
                                 panel[(j * height) + r] = lower_[((first + r) * n_) + k0 + j];
                               }
                             }
                             for (std::size_t j {0}; j < width; ++j) {
                               T* const column {panel.data() + (j * height)};
                               for (std::size_t k {0}; k < j; ++k) {
                                 detail::subtract_product(column, panel.data() + (k * height), detail::conj_of(lower_[((k0 + j) * n_) + k0 + k]), height);
                               }
                               const T pivot {lower_[((k0 + j) * n_) + k0 + j]};
                               for (std::size_t r {0}; r < height; ++r) {
                                 column[r] = column[r] / pivot;
                               }
                             }
                             for (std::size_t r {0}; r < height; ++r) {
                               for (std::size_t j {0}; j < width; ++j) {
                                 lower_[((first + r) * n_) + k0 + j] = panel[(j * height) + r];
                               }
                             }
                           });
    }

    //! \brief `A22 += (−L21) L21ᴴ`, by blocks of \ref detail::cholesky_update_columns columns, on and below the
    //!        diagonal blocks (the upper triangle, never read, takes what falls in a block past the diagonal).
    void update_trailing(std::size_t k0,
                         std::size_t end)
    {
      const std::size_t rows                      {n_ - end};
      const std::size_t depth                     {end - k0};
      buffer<T>         minus_l(rows * depth, T   {});
      buffer<T>         l_adjoint(depth * rows, T {});
      for (std::size_t r {0}; r < rows; ++r) {
        for (std::size_t p {0}; p < depth; ++p) {
          const T value {lower_[((end + r) * n_) + k0 + p]};
          minus_l[(r * depth) + p]  = -value;
          l_adjoint[(p * rows) + r] = detail::conj_of(value);
        }
      }
      const std::size_t step {std::max(block_, detail::cholesky_update_columns)};
      for (std::size_t jb {0}; jb < rows; jb += step) {
        const std::size_t width {std::min(step, rows - jb)};
        detail::gemm(rows - jb, width, depth, minus_l.data() + (jb * depth), depth, l_adjoint.data() + jb, rows,
                     lower_.data() + ((end + jb) * n_) + end + jb, n_);
      }
    }

    std::size_t    n_;                                      //!< The matrix order.
    std::size_t    block_ {detail::cholesky_block_for(n_)}; //!< The panel width.
    buffer<T>      lower_;                                  //!< The lower-triangular factor L, row-major (upper triangle zero).
  };

  //! \brief Cholesky-factorises a Hermitian positive-definite matrix. \see cholesky_decomposition
  template <typename T>
  [[nodiscard]] cholesky_decomposition<T> cholesky(matrix<T> a)
  {
    return cholesky_decomposition<T> {std::move(a)};
  }

  namespace detail {

    //! \brief A thin QR factorisation, row-major: `q` is `rows × cols` with orthonormal columns, `r` is
    //!        `cols × cols` upper triangular with a real, non-negative diagonal.
    template <typename T>
    struct thin_qr
    {
      buffer<T> q; //!< The orthonormal factor.
      buffer<T> r; //!< The triangular factor.
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
    void reflect_rows(buffer<T>&            a,
                      std::size_t           cols,
                      const buffer<T>&      v,
                      std::size_t           first,
                      std::size_t           from)
    {
      constexpr std::size_t width {64};
      const std::size_t     span  {cols - from};
      parallel_for((span + width - 1) / width, 2 * v.size() * width, [&](std::size_t chunk) {
                     const std::size_t start {from + (chunk * width)};
                     const std::size_t count {std::min(width, cols - start)};
                     buffer<T>    w(count, T {});
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

    //! \brief The reflectors of one panel of the blocked QR, `I − V T Vᴴ` as LAPACK's `dlarft` forms it.
    template <typename T>
    struct reflector_block
    {
      std::size_t    first;  //!< The first row the block acts on.
      std::size_t    height; //!< The rows it acts on, `first…first+height-1`.
      std::size_t    width;  //!< Its reflector count.
      buffer<T>      v;      //!< `V`, `height × width` row-major, reflector j from row j (zeros above).
      buffer<T>      t;      //!< `T`, `width × width` upper triangular.
    };

    /*!
     * \brief Gathers the unit reflectors `reflectors[k0…k1-1]` (each `I − 2 v vᴴ`, an empty one the
     *        identity) into `I − V T Vᴴ = H_{k0} H_{k0+1} … H_{k1-1}`.
     */
    template <typename T>
    [[nodiscard]] reflector_block<T> gather_reflectors(const std::vector<buffer<T>>&      reflectors,
                                                       std::size_t                        k0,
                                                       std::size_t                        k1,
                                                       std::size_t                        rows)
    {
      reflector_block<T> block {.first = k0, .height = rows - k0, .width = k1 - k0, .v = {}, .t = {}};
      const std::size_t  b     {block.width};
      block.v.assign(block.height * b, T {});
      block.t.assign(b * b, T {});
      for (std::size_t j {0}; j < b; ++j) {
        const buffer<T>& v {reflectors[k0 + j]};
        for (std::size_t r {0}; r < v.size(); ++r) {
          block.v[((j + r) * b) + j] = v[r];
        }
      }
      // Vᴴ V once, through the matrix product kernel: its column j above the diagonal is V(:, 0…j-1)ᴴ v_j.
      buffer<T> v_adjoint(b * block.height, T {});
      for (std::size_t r {0}; r < block.height; ++r) {
        for (std::size_t j {0}; j < b; ++j) {
          v_adjoint[(j * block.height) + r] = conj_of(block.v[(r * b) + j]);
        }
      }
      buffer<T> gram(b * b, T {});
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
                               buffer<T>&                a,
                               std::size_t               cols,
                               std::size_t               from,
                               std::size_t               to)
    {
      const std::size_t b                  {block.width};
      const std::size_t h                  {block.height};
      const std::size_t w                  {to - from};
      buffer<T>         v_adjoint(b * h, T {});
      buffer<T>         minus_v(h * b, T   {});
      for (std::size_t r {0}; r < h; ++r) {
        for (std::size_t j {0}; j < b; ++j) {
          v_adjoint[(j * h) + r]  = conj_of(block.v[(r * b) + j]);
          minus_v[(r * b) + j]    = -block.v[(r * b) + j];
        }
      }
      T*             top                 {a.data() + (block.first * cols) + from};
      buffer<T>      projection(b * w, T {});
      gemm(b, w, h, v_adjoint.data(), h, top, cols, projection.data(), w);
      buffer<T> scaled(b * w, T          {});
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
     * \brief \ref apply_reflector_block to `A` held transposed: \p at is `Aᵀ` (row-major, \p cols columns), and
     *        `A ← (I − V T Vᴴ) A` on `A`'s rows `first…` and columns `from…to-1` is `Aᵀ ← Aᵀ (I − conj(V) Tᵀ Vᵀ)`
     *        on `Aᵀ`'s columns `first…` and rows `from…to-1`. Its three products have `to − from` rows, which the
     *        matrix product kernel runs well and on every core; \ref apply_reflector_block's first has the
     *        block's few (a reflector block's 32 rows by 1000 deep: 54 GFLOP/s on twenty x86-64 threads).
     */
    template <typename T>
    void apply_transposed_reflector_block(const reflector_block<T>& block,
                                          buffer<T>&                at,
                                          std::size_t               cols,
                                          std::size_t               from,
                                          std::size_t               to)
    {
      const std::size_t b            {block.width};
      const std::size_t h            {block.height};
      const std::size_t r            {to - from};
      const auto        v_conjugate  {std::make_unique_for_overwrite<T[]>(h * b)}; // conj(V), h × b, written whole
      const auto        minus_v_t    {std::make_unique_for_overwrite<T[]>(b * h)}; // −Vᵀ, b × h
      const auto        t_transposed {std::make_unique_for_overwrite<T[]>(b * b)}; // Tᵀ
      for (std::size_t row {0}; row < h; ++row) {
        for (std::size_t j {0}; j < b; ++j) {
          v_conjugate[(row * b) + j] = conj_of(block.v[(row * b) + j]);
          minus_v_t[(j * h) + row]   = -block.v[(row * b) + j];
        }
      }
      for (std::size_t i {0}; i < b; ++i) {
        for (std::size_t p {0}; p < b; ++p) {
          t_transposed[(p * b) + i] = block.t[(i * b) + p];
        }
      }
      T*             left                {at.data() + (from * cols) + block.first};
      buffer<T>      projection(r * b, T {});
      gemm(r, b, h, left, cols, v_conjugate.get(), b, projection.data(), b);
      buffer<T> scaled(r * b, T          {});
      gemm(r, b, b, projection.data(), b, t_transposed.get(), b, scaled.data(), b);
      gemm(r, h, b, scaled.data(), b, minus_v_t.get(), h, left, cols);
    }

    //! \brief `Σ conj(x[i]) · y[i]` for a real `T`, where the conjugate is the value: \ref dot_product.
    template <typename T>
    requires std::is_floating_point_v<T>
    [[nodiscard]] T conjugate_dot(const T*    x,
                                  const T*    y,
                                  std::size_t count)
    {
      return dot_product(x, y, count);
    }

    //! \brief `Σ conj(x[i]) · y[i]`, in order.
    template <typename T>
    [[nodiscard]] T conjugate_dot(const T*    x,
                                  const T*    y,
                                  std::size_t count)
    {
      T total {};
      for (std::size_t i {0}; i < count; ++i) {
        total += conj_of(x[i]) * y[i];
      }
      return total;
    }

    /*!
     * \brief Applies `I − 2 v vᴴ` from the left to the columns `from…to-1` of a column-major panel of \p height rows,
     *        on its rows `first…first+|v|-1`: each column a dot product with `v`, then `v` scaled out of it, both
     *        along the column at unit stride.
     */
    template <typename T>
    void reflect_columns(buffer<T>&            panel,
                         std::size_t           height,
                         const buffer<T>&      v,
                         std::size_t           first,
                         std::size_t           from,
                         std::size_t           to)
    {
      for (std::size_t j {from}; j < to; ++j) {
        T* const column {panel.data() + (j * height) + first};
        const T  twice  {static_cast<T>(2) * conjugate_dot(v.data(), column, v.size())};
        subtract_product(column, twice, v.data(), v.size());
      }
    }

    //! \brief The reflectors of one panel of the blocked QR.
    inline constexpr std::size_t qr_block {32};

    /*!
     * \brief Householder reflections of the panel's columns `k0…k1-1` of the row-major \p a, rows from `k0`, one by
     *        one, into `reflectors[k0…k1-1]`: in a column-major copy, where a reflector meets each column at unit
     *        stride (row by row, each row's part of the panel was a call of a few elements). Recursive halves, the
     *        left's reflectors applied to the right at once, took longer at these widths (32 columns).
     */
    template <typename T>
    void factor_qr_panel(buffer<T>&                   a,
                         std::size_t                  rows,
                         std::size_t                  cols,
                         std::vector<buffer<T>>&      reflectors,
                         std::size_t                  k0,
                         std::size_t                  k1)
    {
      const std::size_t height                  {rows - k0};
      const std::size_t width                   {k1 - k0};
      buffer<T>         panel(height * width, T {});
      for (std::size_t i {0}; i < height; ++i) {
        for (std::size_t j {0}; j < width; ++j) {
          panel[(j * height) + i] = a[((k0 + i) * cols) + k0 + j];
        }
      }
      for (std::size_t k {k0}; k < k1; ++k) {
        const std::size_t c {k - k0};
        buffer<T>         v(panel.begin() + static_cast<std::ptrdiff_t>((c * height) + c),
                            panel.begin() + static_cast<std::ptrdiff_t>((c + 1) * height));
        (void)rescale_tiny(v); // the reflector is the same at any scale
        const double norm {scaled_norm(v)};
        if (norm == 0.0) {
          continue; // nothing to reflect: the column is already zero below and on the diagonal
        }
        v[0] += phase_of(v[0]) * detail::from_double<T>(norm); // x − α e₁ with α = −phase(x₀)‖x‖: no cancellation
        const auto length {detail::from_double<T>(scaled_norm(v))};
        for (T& element : v) {
          element /= length;
        }
        reflect_columns(panel, height, v, c, c, width); // the panel: A ← (I − 2 v vᴴ) A, columns k…k1-1
        reflectors[k] = std::move(v);
      }
      for (std::size_t i {0}; i < height; ++i) {
        for (std::size_t j {0}; j < width; ++j) {
          a[((k0 + i) * cols) + k0 + j] = panel[(j * height) + i];
        }
      }
    }

    //! \brief The reflections of a Householder QR, `Q` left as its blocks of reflectors.
    template <typename T>
    struct qr_reflections
    {
      buffer<T>                       r;      //!< `R`, `k × cols` upper trapezoidal, its diagonal as it comes.
      std::vector<reflector_block<T>> blocks; //!< `Q = H₀ H₁ …`, block by block of \ref qr_block reflectors.
    };

    /*!
     * \brief The reflections of \ref householder_qr, without forming `Q`: `A = Q R` with `Q` the product of the
     *        blocks' reflectors applied to the first `k = min(rows, cols)` columns of the identity, `R`'s diagonal
     *        not yet made real (a product with `Qᴴ`, or with `Q` of a block padded with zeros, applies the blocks
     *        instead).
     */
    template <typename T>
    [[nodiscard]] qr_reflections<T> householder_reflect(buffer<T>      a,
                                                        std::size_t    rows,
                                                        std::size_t    cols)
    {
      const std::size_t           depth {std::min(rows, cols)};
      std::vector<buffer<T>>      reflectors(depth); // v_k on rows k…m-1; empty where the column was zero
      qr_reflections<T>           out   {.r = buffer<T>(depth * cols, T {}), .blocks = {}};
      for (std::size_t k0 {0}; k0 < depth; k0 += qr_block) {
        const std::size_t k1 {std::min(k0 + qr_block, depth)};
        factor_qr_panel(a, rows, cols, reflectors, k0, k1);
        out.blocks.push_back(gather_reflectors(reflectors, k0, k1, rows));
        if (k1 < cols) { // right of the panel: H_{k1-1} … H_{k0} A = (I − V T Vᴴ)ᴴ A
          apply_reflector_block(out.blocks.back(), true, a, cols, k1, cols);
        }
      }
      for (std::size_t i {0}; i < depth; ++i) {
        for (std::size_t j {i}; j < cols; ++j) {
          out.r[(i * cols) + j] = a[(i * cols) + j];
        }
      }
      return out;
    }

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
    [[nodiscard]] thin_qr<T> householder_qr(buffer<T>      a,
                                            std::size_t    rows,
                                            std::size_t    cols)
    {
      const std::size_t depth   {std::min(rows, cols)};
      qr_reflections<T> factors {householder_reflect(std::move(a), rows, cols)};
      thin_qr<T>        out     {.q = buffer<T>(rows * depth, T {}), .r = std::move(factors.r)};
      for (std::size_t i {0}; i < depth; ++i) {
        out.q[(i * depth) + i] = T {1};
      }
      for (std::size_t b {factors.blocks.size()}; b-- > 0;) { // Q = H₀ H₁ … applied to the first columns of I
        apply_reflector_block(factors.blocks[b], false, out.q, depth, factors.blocks[b].first, depth);
      }
      for (std::size_t k {0}; k < depth; ++k) {               // R's diagonal real and non-negative: rotate row k, column k
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
      : qr_decomposition(matrix<T> {a})
    {}

    /*!
     * \brief Factorises \p a in its own storage, an expiring matrix's elements taken over: no copy of them.
     * \param[in] a The matrix, any shape.
     * \throws non_finite_error If an entry is infinite or NaN.
     */
    explicit qr_decomposition(matrix<T>&& a)
      : rows_(a.rows()), cols_(a.cols()), depth_(std::min(rows_, cols_))
    {
      detail::require_finite(a, "QR decomposition");
      detail::thin_qr<T> factors {detail::householder_qr(std::move(a).elements(), rows_, cols_)};
      q_ = std::move(factors.q);
      r_ = std::move(factors.r);
    }

    //! \brief The `m × k` factor `Q` (orthonormal columns), `k = min(m, n)`.
    [[nodiscard]] matrix<T> q() const&
    {
      return matrix<T> {rows_, depth_, q_};
    }

    //! \brief `Q`, moved out of an expiring decomposition (\ref r stays).
    [[nodiscard]] matrix<T> q() &&
    {
      return matrix<T> {rows_, depth_, std::move(q_)};
    }

    //! \brief The `k × n` upper-trapezoidal factor `R`, `k = min(m, n)`.
    [[nodiscard]] matrix<T> r() const&
    {
      return matrix<T> {depth_, cols_, r_};
    }

    //! \brief `R`, moved out of an expiring decomposition (\ref q stays).
    [[nodiscard]] matrix<T> r() &&
    {
      return matrix<T> {depth_, cols_, std::move(r_)};
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
      const double   tolerance  {detail::rank_tolerance(largest, rows_, cols_)};
      buffer<T>      y(cols_, T {}); // y = Qᴴ b
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
      buffer<T> x(cols_, T {}); // back substitution R x = y
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
    buffer<T>      q_;     //!< The m×k factor Q, row-major.
    buffer<T>      r_;     //!< The k×n upper-trapezoidal factor R, row-major.
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

    //! \brief A kernel of \ref symmetric_row's form: \ref symmetric_row itself, or an x86-64 one.
    using symmetric_row_kernel = double (*)(const double*, const double*, double, double*, std::size_t);

    //! \brief A kernel of \ref symmetric_rows' form: \ref symmetric_rows itself, or an x86-64 one.
    using symmetric_rows_kernel = std::array<double, symmetric_row_group> (*)(const double*, std::size_t, const double*,
                                                                              double*, std::size_t);

    /*!
     * \brief Row \p i's terms of a Hermitian product from the lower triangle: `out[j] += conj(row[j]) v[i]` for
     *        `j < i`, and the return `Σⱼ≤ᵢ row[j] v[j]`, for a complex `T`.
     */
    template <typename T>
    [[nodiscard]] T hermitian_row(symmetric_row_kernel /*kernel*/,
                                  const T*             row,
                                  const T*             v,
                                  T*                   out,
                                  std::size_t          i)
    {
      add_conjugate_product(out, row, v[i], i);
      return dot_product(row, v, i + 1);
    }

    /*!
     * \brief The same for a `std::complex<double>` row: in one pass on AVX-512 or AVX2 with FMA where the matrix
     *        product's kernel runs there (`avx512_hermitian_row`), else as the template.
     */
    [[nodiscard]] inline std::complex<double> hermitian_row(symmetric_row_kernel        kernel,
                                                            const std::complex<double>* row,
                                                            const std::complex<double>* v,
                                                            std::complex<double>*       out,
                                                            std::size_t                 i)
    {
#ifdef NUMERICAL_X86_COMPILED
      // A complex number's parts are an array of two (the standard's layout guarantee): addressed as doubles.
      const gemm_path path {gemm_path_current()};
      if (path == gemm_path::avx512 || path == gemm_path::avx2) {
        const auto* const          values {reinterpret_cast<const double*>(v)};   // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
        const auto* const          parts  {reinterpret_cast<const double*>(row)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
        auto* const                into   {reinterpret_cast<double*>(out)};       // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
        const std::complex<double> lower  {path == gemm_path::avx512
                                          ? avx512_hermitian_row(parts, values, v[i].real(), v[i].imag(), into, i)
                                          : avx2_hermitian_row(parts, values, v[i].real(), v[i].imag(), into, i)};
        return lower + (row[i] * v[i]);
      }
#endif
      return hermitian_row<std::complex<double>>(kernel, row, v, out, i);
    }

    //! \brief The same for a real row, in one pass through \p kernel.
    [[nodiscard]] inline double hermitian_row(symmetric_row_kernel kernel,
                                              const double*        row,
                                              const double*        v,
                                              double*              out,
                                              std::size_t          i)
    {
      return kernel(row, v, v[i], out, i) + (row[i] * v[i]);
    }

    /*!
     * \brief The weight of a symmetric row's multiply-adds in `hermitian_product`: on the x86-64 units, where its
     *        rows are paced by one core's loads, \ref vector_speedup, as more threads pay than their multiply-adds
     *        alone would pay for (an i9-9900K's reduction of order 512 took 17 % less, an i9-10900X's 7 %); 1
     *        elsewhere (arm64's, weighed so, took as long or longer).
     */
    template <typename T>
    [[nodiscard]] std::size_t symmetric_row_speedup()
    {
#ifdef NUMERICAL_X86_COMPILED
      if constexpr (std::is_same_v<T, double>) {
        if (gemm_path_current() != gemm_path::portable) {
          return vector_speedup;
        }
      }
#endif
      return 1;
    }

    /*!
     * \brief `w = B v` for the Hermitian `m × m` \p block (row stride \p n) from its lower triangle alone, as LAPACK's
     *        `dsymv`: each row's lower part is read once and serves twice, `w[i] += Σⱼ≤ᵢ B[i][j] v[j]` and
     *        `w[j] += conj(B[i][j]) v[i]` for `j < i`, so a reduction step reads half the block it would read whole.
     *        Rows go by tasks of fixed size, each into its own partial `w`, summed in a fixed order: the result does
     *        not depend on the number of threads. A real row is read once for both (\ref symmetric_row), on the
     *        widest x86-64 units the matrix product's kernel may use (\ref gemm_path_current).
     * \param[in]     block   The block's first element; its rows \p n apart.
     * \param[in]     n       The row stride.
     * \param[in]     m       The block's order.
     * \param[in]     v       The vector (\p m elements).
     * \param[out]    w       `B v` (\p m elements).
     * \param[in,out] scratch The partials, kept by the caller from one step of a reduction to the next: a fresh
     *        buffer of their size each step is mapped anew, page by page.
     */
    template <typename T>
    void hermitian_product(const T*        block,
                           std::size_t     n,
                           std::size_t     m,
                           const T*        v,
                           T*              w,
                           buffer<T>&      scratch)
    {
      constexpr std::size_t rows_per_task {32};
      const std::size_t     tasks         {(m + rows_per_task - 1) / rows_per_task};
      if (scratch.size() < tasks * m) {
        scratch.resize(tasks * m, T {});
      }
      symmetric_row_kernel kernel {[](const double* row, const double* x, double f, double* out, std::size_t count) {
                                     return symmetric_row(row, x, f, out, count);
                                   }};
      symmetric_rows_kernel group {[](const double* rows, std::size_t ld, const double* x, double* out, std::size_t i) {
                                     return symmetric_rows(rows, ld, x, out, i);
                                   }};
#ifdef NUMERICAL_X86_COMPILED
      if constexpr (std::is_same_v<T, double>) {
        const gemm_path path {gemm_path_current()};
        if (path == gemm_path::avx512) {
          kernel = &avx512_symmetric_row;
          group  = &avx512_symmetric_rows;
        }
        else if (path == gemm_path::avx2) {
          kernel = &avx2_symmetric_row;
          group  = &avx2_symmetric_rows;
        }
      }
#endif
      parallel_for(tasks, rows_per_task * m * symmetric_row_speedup<T>(), [&](std::size_t task) {
                     const std::size_t first {task * rows_per_task};
                     const std::size_t last {std::min(first + rows_per_task, m)};
                     T*                own {scratch.data() + (task * m)};
                     std::fill(own, own + last, T {}); // a task's rows reach w[0…last) alone
                     std::size_t i {first};
                     if constexpr (std::is_same_v<T, double>) { // rows by groups: v and own read once a group
                       for (; i + symmetric_row_group <= last; i += symmetric_row_group) {
                         const std::array<double, symmetric_row_group> sums {group(block + (i * n), n, v, own, i)};
                         for (std::size_t r {0}; r < symmetric_row_group; ++r) {
                           own[i + r] += sums[r];
                         }
                       }
                     }
                     for (; i < last; ++i) {
                       own[i] += hermitian_row(kernel, block + (i * n), v, own, i);
                     }
                   });
      std::fill(w, w + m, T {});
      for (std::size_t task {0}; task < tasks; ++task) { // the partials in order, each a contiguous update
        add_product(w, T {1}, scratch.data() + (task * m), std::min((task + 1) * rows_per_task, m));
      }
    }

    //! \brief The reflections of one panel of the blocked tridiagonal reduction.
    inline constexpr std::size_t tridiagonal_block {32};

    //! \brief The rows of each band of the tridiagonal reduction's trailing update: a band reaches its diagonal,
    //!        so a wider one computes more of the upper triangle the reduction never reads, a narrower one runs
    //!        the product kernel on smaller products.
    inline constexpr std::size_t tridiagonal_band {512};

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

    //! \brief The unitary `M = Q Dᴴ` of a reduction to real tridiagonal form, `A = M T Mᴴ`, kept as its factors.
    template <typename T>
    struct tridiagonal_reduction
    {
      std::vector<buffer<T>> reflectors; //!< `Q = H₁ H₂ …`: `reflectors[k + 1]` zeroes column k, from row k + 1.
      buffer<T>              delta;      //!< `D = diag(δ)`, the unit phases that make `T` real.
    };

    /*!
     * \brief Reduces the Hermitian (real: symmetric) `n × n` row-major \p a, of which the lower triangle is read, to
     *        a real symmetric tridiagonal by Householder reflections, `A = M T Mᴴ` with `M` unitary.
     *
     * Each reflection zeroes one column below the subdiagonal; the trailing block takes the rank-2 updates
     * `B − v wᴴ − w vᴴ` of a panel of \ref tridiagonal_block reflections at once, through the matrix product
     * kernel, as LAPACK's `dsytrd`. A complex subdiagonal entry is then made real by a unit phase per row, as
     * LAPACK's `zhetrd` leaves it for a real iteration; a real one keeps its sign.
     * \param[in,out] a The matrix; destroyed.
     * \param[in]     n Its order (at least 1).
     * \param[out]    d The diagonal of `T` (size \p n).
     * \param[out]    e The subdiagonal of `T` in `e[0…n-2]`, and `e[n-1] = 0` (size \p n).
     * \return `M`'s reflectors and phases: \ref tridiagonal_basis forms `M`, \ref to_matrix_basis applies it.
     */
    template <typename T>
    tridiagonal_reduction<T> tridiagonalize(buffer<T>&           a,
                                            std::size_t          n,
                                            buffer<double>&      d,
                                            buffer<double>&      e)
    {
      std::vector<buffer<T>> reflectors(n);                                                                  // reflectors[k + 1] zeroes column k, acting on rows k + 1…
      buffer<T>              symv_scratch;                                                                   // hermitian_product's partials, kept from step to step
      buffer<T>              sub(n, T      {});                                                              // T's subdiagonal before the phases
      // The trailing update's factors, for the first panel's size, the largest: one buffer each, mapped once.
      const auto left                      {std::make_unique_for_overwrite<T[]>(n * 2 * tridiagonal_block)}; // row i: −V's then −W's
      const auto right                     {std::make_unique_for_overwrite<T[]>(2 * tridiagonal_block * n)}; // Wᴴ's rows, then Vᴴ's
      for (std::size_t k0 {0}; k0 + 1 < n; k0 += tridiagonal_block) {
        // A panel of reflections, as LAPACK's dlatrd: the matrix the panel's later steps see is
        // A − V Wᴴ − W Vᴴ, applied to the one column and the one product each step needs; the whole trailing
        // block takes it after the panel, through the matrix product kernel.
        const std::size_t k1                 {std::min(k0 + tridiagonal_block, n - 1)};
        const std::size_t width              {k1 - k0};
        // By reflection: row p of each holds step k0 + p's vector over the rows (zero above), so every term of the
        // panel is a contiguous update.
        buffer<T> by_row_v(width * n, T {});
        buffer<T> by_row_w(width * n, T {});
        for (std::size_t k {k0}; k < k1; ++k) {
          const std::size_t j               {k - k0};
          const std::size_t m               {n - k - 1};
          buffer<T>         column(m + 1, T {}); // column k from the diagonal down, the panel's earlier steps applied
          for (std::size_t i {k}; i < n; ++i) {
            column[i - k] = a[(i * n) + k];
          }
          for (std::size_t p {0}; p < j; ++p) {
            subtract_product(column.data(), by_row_v.data() + (p * n) + k, conj_of(by_row_w[(p * n) + k]), m + 1);
            subtract_product(column.data(), by_row_w.data() + (p * n) + k, conj_of(by_row_v[(p * n) + k]), m + 1);
          }
          a[(k * n) + k] = column[0];
          buffer<T>      v(column.begin() + 1, column.end()); // column k below the diagonal
          const double   below {m > 1 ? scaled_norm(v.data() + 1, m - 1) : 0.0};
          if (below == 0.0) {
            sub[k] = v[0]; // already tridiagonal in this column: nothing to reflect
            continue;
          }
          const double up    {rescale_tiny(v)};                                  // the reflector is the same at any scale; α is scaled back
          const T      alpha {-phase_of(v[0]) * static_cast<T>(scaled_norm(v))}; // the sign that avoids cancellation
          sub[k]  = alpha / static_cast<T>(up);
          v[0]   -= alpha;
          const auto length {static_cast<T>(scaled_norm(v))};
          for (T& element : v) {
            element /= length;
          }
          const T*       block  {a.data() + ((k + 1) * n) + k + 1};         // the trailing block, row stride n
          buffer<T>      w(m, T {});
          hermitian_product(block, n, m, v.data(), w.data(), symv_scratch); // w = A v on the trailing block
          for (std::size_t p {0}; p < j; ++p) {                                           // w = (A − V Wᴴ − W Vᴴ) v
            const T by_w {conjugate_dot(by_row_w.data() + (p * n) + k + 1, v.data(), m)}; // (Wᴴ v)ₚ
            const T by_v {conjugate_dot(by_row_v.data() + (p * n) + k + 1, v.data(), m)}; // (Vᴴ v)ₚ
            subtract_product(w.data(), by_row_v.data() + (p * n) + k + 1, by_w, m);
            subtract_product(w.data(), by_row_w.data() + (p * n) + k + 1, by_v, m);
          }
          const T c {conjugate_dot(v.data(), w.data(), m)};
          for (std::size_t i {0}; i < m; ++i) { // w = 2 B v − 2 (vᴴ B v) v, so H B H = B − v wᴴ − w vᴴ
            w[i]                                = (static_cast<T>(2) * w[i]) - (static_cast<T>(2) * c * v[i]);
            by_row_v[(j * n) + k + 1 + i]       = v[i];
            by_row_w[(j * n) + k + 1 + i]       = w[i];
          }
          reflectors[k + 1] = std::move(v); // acting from row k + 1, as householder_qr's from row k
        }
        // The trailing block from row and column k1: A −= V Wᴴ + W Vᴴ, one product of depth 2 · width,
        // [−V −W] [Wᴴ; Vᴴ], on the lower triangle alone, as LAPACK's dsyr2k: the reduction reads nothing above the
        // diagonal. By bands of rows, each to the diagonal: about half the full block's operations.
        const std::size_t rest  {n - k1};
        const std::size_t depth {2 * width};
        for (std::size_t i {0}; i < rest; ++i) { // each row of left written whole, in order
          for (std::size_t p {0}; p < width; ++p) {
            left[(i * depth) + p]         = -by_row_v[(p * n) + k1 + i];
            left[(i * depth) + width + p] = -by_row_w[(p * n) + k1 + i];
          }
        }
        for (std::size_t p {0}; p < width; ++p) {
          for (std::size_t i {0}; i < rest; ++i) {
            right[(p * rest) + i]           = conj_of(by_row_w[(p * n) + k1 + i]);
            right[((width + p) * rest) + i] = conj_of(by_row_v[(p * n) + k1 + i]);
          }
        }
        T* trailing {a.data() + (k1 * n) + k1};
        for (std::size_t r0 {0}; r0 < rest; r0 += tridiagonal_band) {
          const std::size_t rows {std::min(tridiagonal_band, rest - r0)};
          gemm(rows, r0 + rows, depth, left.get() + (r0 * depth), depth, right.get(), rest, trailing + (r0 * n), n);
        }
      }
      for (std::size_t i {0}; i < n; ++i) {
        d[i] = real_part(a[(i * n) + i]);
      }
      e[n - 1] = 0.0;
      // D T Dᴴ is real with D = diag(δ): δ₀ = 1, δ_{k+1} = δ_k · conj(phase(t_k)); then M = Q Dᴴ.
      buffer<T> delta(n, T {1});
      for (std::size_t k {0}; k + 1 < n; ++k) {
        e[k] = real_subdiagonal(sub[k], delta[k], delta[k + 1]);
      }
      return {.reflectors = std::move(reflectors), .delta = std::move(delta)};
    }

    /*!
     * \brief `Mᵀ` of a reduction, row-major (its rows are the columns of `M`): `Qᵀ` by blocks of reflectors from the
     *        last back, as LAPACK's `dorgtr` forms `Q`, each block on the rows it reaches of the identity, then row j
     *        times `conj(δⱼ)`.
     */
    template <typename T>
    [[nodiscard]] buffer<T> tridiagonal_basis(const tridiagonal_reduction<T>& reduction,
                                              std::size_t                     n)
    {
      buffer<T> mt(n * n, T {});
      for (std::size_t i {0}; i < n; ++i) {
        mt[(i * n) + i] = T {1};
      }
      for (std::size_t k1 {n}; k1 > 1;) {
        const std::size_t k0 {k1 - std::min(k1 - 1, qr_block)};
        apply_transposed_reflector_block(gather_reflectors(reduction.reflectors, k0, k1, n), mt, n, k0, n);
        k1 = k0;
      }
      for (std::size_t j {0}; j < n; ++j) { // row j of Mᵀ = Dᴴ Qᵀ: row j of Qᵀ times conj(δⱼ)
        const T factor {conj_of(reduction.delta[j])};
        for (std::size_t i {0}; i < n; ++i) {
          mt[(j * n) + i] *= factor;
        }
      }
      return mt;
    }

    /*!
     * \brief The rows of \p at, vectors in the tridiagonal's basis, taken to the matrix's: `at ← at Mᵀ = at Dᴴ Qᵀ`,
     *        column j times `conj(δⱼ)`, then the blocks of reflectors from the last back on every row, as LAPACK's
     *        `dormtr`: two thirds of the operations of forming `M` (\ref tridiagonal_basis) and multiplying by it.
     */
    template <typename T>
    void to_matrix_basis(const tridiagonal_reduction<T>& reduction,
                         buffer<T>&                      at,
                         std::size_t                     n)
    {
      if constexpr (!std::is_floating_point_v<T>) { // a real reduction's phases are all 1
        parallel_for(n, n, [&](std::size_t i) {
                       for (std::size_t j {0}; j < n; ++j) {
                         at[(i * n) + j] *= conj_of(reduction.delta[j]);
                       }
                     });
      }
      for (std::size_t k1 {n}; k1 > 1;) {
        const std::size_t k0 {k1 - std::min(k1 - 1, qr_block)};
        apply_transposed_reflector_block(gather_reflectors(reduction.reflectors, k0, k1, n), at, n, 0, n);
        k1 = k0;
      }
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
    void apply_row_operations(buffer<T>&                        m,
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
    void tridiagonal_ql(buffer<double>&      d,
                        buffer<double>&      e,
                        buffer<T>&           zt,
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
      const bool                 record {!zt.empty()}; // the eigenvalues alone record nothing
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
          double       r    {rotation_length(p, 1.0)};
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
            r        = rotation_length(p, e[i]);
            e[i + 1] = s * r;
            s        = e[i] / r;
            c        = p / r;
            p        = (c * d[i]) - (s * g);
            d[i + 1] = h + (s * ((c * g) + (s * d[i])));
            // rows i and i+1 (columns i and i+1 of Z): upper ← c upper − s lower, lower ← c lower + s upper
            if (record) {
              on_zt.push_back({row_operation::kind::rotate, i, i + 1, c, -s});
            }
          }
          p    = -s * s2 * c3 * el1 * e[l] / dl1;
          e[l] = s * p;
          d[l] = c * p;
        }
        d[l] += shift;
        e[l]  = 0.0;
      }
      if (record) {
        apply_row_operations(zt, n, on_zt);
      }
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
     * \brief The smaller root `η` of `C η² − A η + B = 0`, in the form that does not cancel: a step of
     *        \ref solve_secular's fit (shared by its instantiations, whose data reach one form or the other).
     * \param[in] a `A`.
     * \param[in] b `B`.
     * \param[in] c `C`.
     * \return `η`.
     */
    [[nodiscard]] inline double quadratic_step(double a,
                                               double b,
                                               double c)
    {
      const double root {std::sqrt(std::abs((a * a) - (4.0 * b * c)))};
      return a <= 0.0 ? (a - root) / (2.0 * c) : (2.0 * b) / (a + root);
    }

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
    [[nodiscard]] secular_solution solve_secular(const buffer<double>&      z,
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
        // The two sides apart, each in its order: a term chose its sum by a branch (an SVD of order 128 on one arm64
        // core: 1.87 → 1.68 ms, eigh 1.02 → 0.82).
        for (std::size_t j {0}; j <= i; ++j) {
          delta[j] = offset(origin, j) - tau;
          const double t {z[j] / delta[j]};
          psi  += z[j] * t;
          dpsi += t * t;
        }
        for (std::size_t j {i + 1}; j < k; ++j) {
          delta[j] = offset(origin, j) - tau;
          const double t {z[j] / delta[j]};
          phi  += z[j] * t;
          dphi += t * t;
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
        const double next {tau + (last ? a / c : quadratic_step(a, b, c))};
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
    [[nodiscard]] inline double secular_root(const buffer<double>&      d,
                                             const buffer<double>&      z,
                                             double                     rho,
                                             std::size_t                i,
                                             std::size_t                max_iterations,
                                             double*                    delta)
    {
      const secular_solution root {solve_secular(z, rho, i, max_iterations, delta,
                                                 [&d](std::size_t o, std::size_t j) { return d[j] - d[o]; })};
      return d[root.origin] + root.offset;
    }

    //! \brief The poles a task of \ref merge_rank_one's Löwner products takes, and the side of its tiles of `u`.
    inline constexpr std::size_t lowner_block {32};

    //! \brief One buffer of `merge_workspace`: its storage and how many elements it holds.
    struct workspace_buffer
    {
      std::unique_ptr<double[]> data; //!< The elements.
      std::size_t               size; //!< Their number.
    };

    //! \brief The buffers of \ref merge_rank_one, kept from one join to the next: a fresh buffer per join is mapped
    //!        anew, page by page. Each is written before it is read (`updated` zeroed row by row by the gathers).
    struct merge_workspace
    {
      workspace_buffer gaps    {}; //!< `poles[j] − roots[i]`, row by root, then the eigenvectors in its place.
      workspace_buffer updated {}; //!< The kept columns of `q` times `u`.
      workspace_buffer columns {}; //!< One half's rows of the kept columns that reach it.
      workspace_buffer factor  {}; //!< The rows of `u` for those columns.
    };

    //! \brief The elements a task of \ref workspace_of touches first: 64 pages of 4 KiB, so a buffer of a few
    //!        megabytes spreads over every core.
    inline constexpr std::size_t first_touch_chunk {std::size_t {1} << 15};

    /*!
     * \brief \p buffer at least \p size elements long, its storage kept for the next join. A larger one is mapped
     *        fresh, and its pages are touched by every thread, chunk by chunk: on Linux a page costs about a
     *        microsecond the first time it is written, and the join that first wrote it waited on them one thread at a
     *        time (an `eigh` of order 1024 mapped some 30 MB of these buffers on one x86-64 core).
     */
    [[nodiscard]] inline double* workspace_of(workspace_buffer& buffer,
                                              std::size_t       size)
    {
      if (buffer.size < size) {
        buffer.data = std::make_unique_for_overwrite<double[]>(size);
        buffer.size = size;
        double* const elements {buffer.data.get()};
        parallel_for((size + first_touch_chunk - 1) / first_touch_chunk, first_touch_chunk, [&](std::size_t chunk) {
                       const std::size_t first {chunk * first_touch_chunk};
                       std::fill(elements + first, elements + std::min(first + first_touch_chunk, size), 0.0);
                     });
      }
      return buffer.data.get();
    }

    /*!
     * \brief The eigenvalues of `diag(d) + ρ z zᵀ`, ascending, into \p d, and its eigenvectors times \p q (the
     *        row-major `n × n` block of rows \p ldq apart whose columns span the halves) into \p q, as LAPACK's
     *        `dlaed1`.
     *
     * Deflation first (`dlaed2`): a negligible `zⱼ` leaves `dⱼ` and its column as they are, and two nearly equal
     * poles are rotated so that one of them does. The remaining `k` roots are found by \ref secular_root, in
     * parallel; `z` is then recomputed from them (Löwner's formula, Gu and Eisenstat), so that the eigenvectors
     * `(dⱼ − λᵢ)⁻¹ ẑⱼ` are orthogonal to working precision, and they reach `q` by a matrix product per half:
     * \p q is block diagonal, its first \p split columns zero below row \p split and the others above it, and
     * only a deflating rotation mixes a column of each (`dlaed3` multiplies the halves apart the same way).
     */
    inline void merge_rank_one(buffer<double>&      d,
                               buffer<double>       z,
                               double               rho,
                               double*              q,
                               std::size_t          ldq,
                               std::size_t          n,
                               std::size_t          split,
                               merge_workspace&     work)
    {
      // Which halves of the rows each column reaches: 1 the upper, 2 the lower.
      std::vector<unsigned> reach(n);
      for (std::size_t j {0}; j < n; ++j) {
        reach[j] = j < split ? 1U : 2U;
      }
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
            const double x {q[(r * ldq) + pending]};
            const double y {q[(r * ldq) + j]};
            q[(r * ldq) + pending] = (c * x) + (s * y);
            q[(r * ldq) + j]       = (c * y) - (s * x);
          }
          reach[j] = reach[pending] = reach[j] | reach[pending];
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
      const std::size_t   k                 {kept.size()};
      buffer<double>      poles(k, double   {});
      buffer<double>      weights(k, double {});
      double              norm              {0.0};
      for (std::size_t i {0}; i < k; ++i) {
        poles[i]    = d[kept[i]];
        weights[i]  = z[kept[i]];
        norm       += weights[i] * weights[i];
      }
      norm = std::sqrt(norm);
      for (double& weight : weights) {
        weight /= norm;
      }
      const double        scaled_rho      {rho * norm * norm};
      buffer<double>      roots(k, double {});
      // gaps[i * k + j] = poles[j] − roots[i], every element written by its root's solve.
      double* const gaps                  {workspace_of(work.gaps, k * k)};
      parallel_for(k, 8 * k, [&](std::size_t i) {
                     roots[i] = secular_root(poles, weights, scaled_rho, i, secular_iterations, gaps + (i * k));
                   });
      // Löwner: ẑⱼ² = −Πᵢ (dⱼ − λᵢ) / Πᵢ≠ⱼ (dⱼ − dᵢ), each pole paired with its root, by blocks of poles, each pole's
      // product in ascending order of the roots (a row of gaps at a time, along the block): the same operations in
      // the same order on any number of threads.
      buffer<double> signed_weights(k, double {});
      parallel_for((k + lowner_block - 1) / lowner_block, lowner_block * k, [&](std::size_t block) {
                     const std::size_t first {block * lowner_block};
                     const std::size_t last {std::min(first + lowner_block, k)};
                     std::array<double, lowner_block> product {};
                     for (std::size_t j {first}; j < last; ++j) {
                       product[j - first] = gaps[(j * k) + j];
                     }
                     for (std::size_t i {0}; i < k; ++i) {
                       for (std::size_t j {first}; j < last; ++j) {
                         if (i != j) {
                           product[j - first] *= gaps[(i * k) + j] / (poles[j] - poles[i]);
                         }
                       }
                     }
                     for (std::size_t j {first}; j < last; ++j) {
                       signed_weights[j] = std::copysign(std::sqrt(-product[j - first]), weights[j]);
                     }
                   });
      // u[j][i] = ẑⱼ / (dⱼ − λᵢ), column i roots[i]'s eigenvector in the kept basis, in gaps' place: gaps[i][j] and
      // gaps[j][i] are read, then each written as the other's quotient, tile by tile across the diagonal.
      double* const     u     {gaps};
      const std::size_t tiles {(k + lowner_block - 1) / lowner_block};
      parallel_for(tiles * tiles, lowner_block * lowner_block, [&](std::size_t tile) {
                     const std::size_t ti {tile / tiles};
                     const std::size_t tj {tile % tiles};
                     if (ti > tj) {
                       return; // the pair of tiles is (tj, ti)'s
                     }
                     for (std::size_t i {ti * lowner_block}; i < std::min((ti + 1) * lowner_block, k); ++i) {
                       for (std::size_t j {std::max(tj * lowner_block, ti == tj ? i : 0)}; j < std::min((tj + 1) * lowner_block, k); ++j) {
                         const double above {gaps[(i * k) + j]};
                         const double below {gaps[(j * k) + i]};
                         u[(j * k) + i] = signed_weights[j] / above;
                         u[(i * k) + j] = signed_weights[i] / below;
                       }
                     }
                   });
      buffer<double> length(k, 0.0);
      for (std::size_t j {0}; j < k; ++j) { // by rows of u, each a contiguous update of every column's sum
        for (std::size_t i {0}; i < k; ++i) {
          length[i] += u[(j * k) + i] * u[(j * k) + i];
        }
      }
      for (double& l : length) {
        l = std::sqrt(l);
      }
      parallel_for(k, k, [&](std::size_t j) {
                     for (std::size_t i {0}; i < k; ++i) {
                       u[(j * k) + i] /= length[i];
                     }
                   });
      // updated = q(:, kept) · u, by halves of the rows: a half's rows take the kept columns that reach it, and
      // u's rows for those, the others being zero there.
      double* const updated {workspace_of(work.updated, n * k)};
      for (const auto& [half, first, rows] : {std::tuple {1U, std::size_t {0}, split}, std::tuple {2U, split, n - split}}) {
        std::vector<std::size_t> reaching;
        for (std::size_t i {0}; i < k; ++i) {
          if ((reach[kept[i]] & half) != 0U) {
            reaching.push_back(i);
          }
        }
        const std::size_t width   {reaching.size()};
        double* const     columns {workspace_of(work.columns, rows * width)};
        double* const     factor  {workspace_of(work.factor, width * k)};
        parallel_for(rows, width + k, [&](std::size_t r) {
                       for (std::size_t c {0}; c < width; ++c) {
                         columns[(r * width) + c] = q[((first + r) * ldq) + kept[reaching[c]]];
                       }
                       std::fill_n(updated + ((first + r) * k), k, 0.0); // the product adds to it
                     });
        parallel_for(width, k, [&](std::size_t c) { std::copy_n(u + (reaching[c] * k), k, factor + (c * k)); });
        gemm(rows, k, width, columns, width, factor, k, updated + (first * k), k);
      }
      // Every eigenpair, then by ascending eigenvalue: column j of the result is column source[j] of the
      // updated ones (below k) or deflated column source[j] − k of q, written row by row.
      buffer<double> values(n, double {});
      for (std::size_t i {0}; i < k; ++i) {
        values[i] = roots[i];
      }
      for (std::size_t i {0}; i < deflated.size(); ++i) {
        values[k + i] = d[deflated[i]];
      }
      for (std::size_t j {0}; j < n; ++j) {
        order[j] = j;
      }
      std::sort(order.begin(), order.end(), [&values](std::size_t x, std::size_t y) { return values[x] < values[y]; });
      for (std::size_t j {0}; j < n; ++j) {
        d[j] = values[order[j]];
      }
      parallel_for(n, n, [&](std::size_t r) {
                     const buffer<double> before(q + (r * ldq), q + (r * ldq) + n);
                     for (std::size_t j {0}; j < n; ++j) {
                       const std::size_t source {order[j]};
                       q[(r * ldq) + j] = source < k ? updated[(r * k) + source] : before[deflated[source - k]];
                     }
                   });
    }

    /*!
     * \brief One block of \ref divide_and_conquer's tree: its rows `[0, n)` of the diagonal \p d its parent reads,
     *        the diagonal and subdiagonal it is solved on (its own copies scaled by a power of two, or \p d and the
     *        parent's), and its diagonal block of the eigenvector matrix, \p q, rows \p ldq apart.
     */
    struct divide_node
    {
      double*        d        {};  //!< The diagonal its parent reads, the eigenvalues once solved.
      double*        work_d   {};  //!< The diagonal it is solved on.
      const double*  work_e   {};  //!< The subdiagonal it is solved on.
      std::size_t    n        {};  //!< Its order.
      double*        q        {};  //!< Its diagonal block of the eigenvectors.
      std::size_t    ldq      {};  //!< The distance between the rows of \p q.
      int            exponent {0}; //!< The power of two it is scaled down by.
      buffer<double> scaled_d {};  //!< The scaled diagonal, when \p exponent is not 0.
      buffer<double> scaled_e {};  //!< The scaled subdiagonal, when \p exponent is not 0.
      std::size_t    n1       {0}; //!< The order of its first half; 0 for a leaf.
      double         beta     {};  //!< The coupling of its halves.
      std::size_t    depth    {};  //!< Its distance from the root.
      std::size_t    first    {};  //!< Its first half.
      std::size_t    second   {};  //!< Its second half.
    };

    /*!
     * \brief Appends the block (\p d, \p e, \p n) and, before it is solved, everything below it to \p nodes, parents
     *        before children: each scaled to work near unit size by a power of two (exactly), as the deflation
     *        tolerance is relative to the weights, which have unit norm, and so to the tridiagonal only at that size;
     *        each half has its own. A block past \ref divide_leaf is split into two halves plus a rank-one coupling,
     *        as LAPACK's `dlaed0`: `T = diag(T₁ − ρ e_a e_aᵀ, T₂ − ρ e_b e_bᵀ) + ρ u uᵀ`, `u = e_a + sign(β) e_b`,
     *        `ρ = |β|`, `β` the coupling.
     */
    inline void plan_divide(std::vector<divide_node>& nodes,
                            double*                   d,
                            const double*             e,
                            std::size_t               n,
                            double*                   q,
                            std::size_t               ldq,
                            std::size_t               depth)
    {
      double largest {0.0};
      for (std::size_t i {0}; i < n; ++i) {
        largest = std::max({largest, std::abs(d[i]), i + 1 < n ? std::abs(e[i]) : 0.0});
      }
      divide_node node {.d = d, .work_d = d, .work_e = e, .n = n, .q = q, .ldq = ldq, .depth = depth};
      (void)std::frexp(largest, &node.exponent);
      if (node.exponent != 0) {
        node.scaled_d.assign(n, double {});
        node.scaled_e.assign(n, 0.0);
        for (std::size_t i {0}; i < n; ++i) {
          node.scaled_d[i] = std::ldexp(d[i], -node.exponent);
          node.scaled_e[i] = i + 1 < n ? std::ldexp(e[i], -node.exponent) : 0.0;
        }
        node.work_d = node.scaled_d.data(); // the buffers' storage stays put as the node moves into the vector
        node.work_e = node.scaled_e.data();
      }
      if (n <= divide_leaf) {
        nodes.push_back(std::move(node));
        return;
      }
      node.n1                       = n / 2;
      node.beta                     = node.work_e[node.n1 - 1];
      node.work_d[node.n1 - 1]     -= std::abs(node.beta);
      node.work_d[node.n1]         -= std::abs(node.beta);
      double* const        work_d   {node.work_d};
      const double* const  work_e   {node.work_e};
      const std::size_t    n1       {node.n1};
      const std::size_t    index    {nodes.size()};
      nodes.push_back(std::move(node));
      nodes[index].first  = nodes.size();
      plan_divide(nodes, work_d, work_e, n1, q, ldq, depth + 1);
      nodes[index].second = nodes.size();
      plan_divide(nodes, work_d + n1, work_e + n1, n - n1, q + (n1 * ldq) + n1, ldq, depth + 1);
    }

    //! \brief Solves the leaf \p node by implicit QL: its eigenvalues ascending into its diagonal, their
    //!        eigenvectors as the columns of its block of `q`.
    inline void solve_divide_leaf(const divide_node& node)
    {
      const std::size_t n {node.n};
      buffer<double>    diagonal(node.work_d, node.work_d + n);
      buffer<double>    sub(n, 0.0);
      std::copy(node.work_e, node.work_e + (n - 1), sub.begin());
      buffer<double> zt(n * n, 0.0);
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
        node.work_d[j] = diagonal[order[j]];
        for (std::size_t r {0}; r < n; ++r) {
          node.q[(r * node.ldq) + j] = zt[(order[j] * n) + r];
        }
      }
    }

    //! \brief Joins the two solved halves of \p node by \ref merge_rank_one, in \p work.
    inline void join_divide_node(const divide_node& node,
                                 merge_workspace&   work)
    {
      const std::size_t n           {node.n};
      const std::size_t n1          {node.n1};
      const std::size_t n2          {n - n1};
      double* const     q           {node.q};
      buffer<double>    z(n, double {});
      for (std::size_t i {0}; i < n1; ++i) {
        z[i] = q[((n1 - 1) * node.ldq) + i];
      }
      for (std::size_t i {0}; i < n2; ++i) {
        z[n1 + i] = std::copysign(1.0, node.beta) * q[(n1 * node.ldq) + n1 + i];
      }
      buffer<double> values(node.work_d, node.work_d + n);
      merge_rank_one(values, std::move(z), std::abs(node.beta), q, node.ldq, n, n1, work);
      std::copy(values.begin(), values.end(), node.work_d);
    }

    //! \brief The eigenvalues of a solved \p node, scaled back, where its parent reads them.
    inline void unscale_divide_node(const divide_node& node)
    {
      if (node.exponent != 0) {
        for (std::size_t i {0}; i < node.n; ++i) {
          node.d[i] = std::ldexp(node.work_d[i], node.exponent);
        }
      }
    }

    //! \brief Solves the block \p index of \p nodes and everything below it, each block right after its halves.
    inline void solve_divide_subtree(std::vector<divide_node>& nodes,
                                     std::size_t               index,
                                     merge_workspace&          work)
    {
      const divide_node& node {nodes[index]};
      if (node.n1 == 0) {
        solve_divide_leaf(node);
      }
      else {
        solve_divide_subtree(nodes, node.first, work);
        solve_divide_subtree(nodes, node.second, work);
        join_divide_node(node, work);
      }
      unscale_divide_node(node);
    }

    /*!
     * \brief The eigenvalues, ascending, of the symmetric tridiagonal with diagonal `d[0…n-1]` and subdiagonal
     *        `e[0…n-2]` into \p d, and its eigenvectors as the columns of the row-major \p q (`n × n`), by
     *        Cuppen's divide and conquer: the tridiagonal is two halves plus a rank-one coupling, each half is
     *        diagonalised the same way (by implicit QL up to \ref divide_leaf), and \ref merge_rank_one joins
     *        them, each half into its own diagonal block of \p q, so the joins copy no half and leave the blocks off
     *        the diagonal as they found them until the join of their two halves fills them.
     *
     * The tree is laid out first (\ref plan_divide). The blocks at the depth that gives each thread one, and the
     * leaves above it, are independent of one another and go to the threads together, each solved on one thread
     * with its own buffers, a block right after its halves while they are still in its caches; the joins above them
     * then come a level at a time, those of one level together, a level of one join sharing it among the threads.
     * One after another each block had run alone (six AVX2 cores, an `eigh` of order 256: the leaves and the joins
     * below the last, 1.5 ms of the 4 ms the eigenvectors took, no faster than on one). Every block takes the same
     * inputs either way, so the result is the same to the bit; on one thread the order is the recursion's.
     */
    inline void divide_and_conquer(double*              d,
                                   const double*        e,
                                   std::size_t          n,
                                   buffer<double>&      q)
    {
      q.assign(n * n, 0.0);
      std::vector<divide_node> nodes;
      plan_divide(nodes, d, e, n, q.data(), n, 0);
      const auto               cut {static_cast<std::size_t>(std::bit_width(max_threads() - 1))};
      std::vector<std::size_t> tasks;
      for (std::size_t i {0}; i < nodes.size(); ++i) {
        if (nodes[i].depth == cut || (nodes[i].depth < cut && nodes[i].n1 == 0)) {
          tasks.push_back(i);
        }
      }
      std::vector<merge_workspace> own(tasks.size());
      const std::size_t            task_order {nodes[tasks[0]].n};
      parallel_for(tasks.size(), tasks.size() > 1 ? task_order * task_order * task_order : 0, [&](std::size_t t) {
                     solve_divide_subtree(nodes, tasks[t], own[t]);
                   });
      for (std::size_t level {cut}; level-- > 0;) {
        std::vector<std::size_t> joins;
        for (std::size_t i {0}; i < nodes.size(); ++i) {
          if (nodes[i].n1 != 0 && nodes[i].depth == level) {
            joins.push_back(i);
          }
        }
        if (joins.empty()) { // a tree shallower than the cut
          continue;
        }
        if (joins.size() == 1) {
          join_divide_node(nodes[joins[0]], own[0]);
          unscale_divide_node(nodes[joins[0]]);
          continue;
        }
        const std::size_t order {nodes[joins[0]].n};
        parallel_for(joins.size(), order * order * order / 4, [&](std::size_t j) {
                       join_divide_node(nodes[joins[j]], own[j]);
                       unscale_divide_node(nodes[joins[j]]);
                     });
      }
    }

    //! \brief \p zt ← \p left · \p zt, `n × n`, row-major, for any other element type: the real factor converted.
    template <typename T>
    void multiply_real_left(const buffer<double>&      left,
                            buffer<T>&                 zt,
                            std::size_t                n)
    {
      buffer<T> converted(n * n, T {});
      for (std::size_t i {0}; i < n * n; ++i) {
        converted[i] = from_double<T>(left[i]);
      }
      buffer<T> product(n * n, T {});
      gemm(n, n, n, converted.data(), n, zt.data(), n, product.data(), n);
      zt = std::move(product);
    }

    //! \brief \p zt ← \p left · \p zt, `n × n`, row-major, for a real left factor.
    inline void multiply_real_left(const buffer<double>&      left,
                                   buffer<double>&            zt,
                                   std::size_t                n)
    {
      buffer<double> product(n * n, 0.0);
      gemm(n, n, n, left.data(), n, zt.data(), n, product.data(), n);
      zt = std::move(product);
    }

    //! \brief \p zt ← \p left · \p zt for a complex \p zt: the real and imaginary parts by a real product each.
    inline void multiply_real_left(const buffer<double>&              left,
                                   buffer<std::complex<double>>&      zt,
                                   std::size_t                        n)
    {
      buffer<double> re(n * n, double {});
      buffer<double> im(n * n, double {});
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
     * \brief The eigenvalues of the tridiagonal (\p d, \p e) of \p reduction into \p d, and the matrix's
     *        eigenvectors as the rows of the result, by divide and conquer (LAPACK's `dstedc`, then `dormtr`):
     *        most of the work in matrix products. Up to \ref divide_leaf, \ref tridiagonal_ql on `Mᵀ` itself.
     * \throws convergence_error If an iteration did not converge within its bound.
     */
    template <typename T>
    [[nodiscard]] buffer<T> tridiagonal_divide_and_conquer(buffer<double>&                 d,
                                                           buffer<double>&                 e,
                                                           const tridiagonal_reduction<T>& reduction,
                                                           std::size_t                     n)
    {
      if (n <= divide_leaf) {
        buffer<T> zt {tridiagonal_basis(reduction, n)};
        tridiagonal_ql(d, e, zt, n, ql_steps_per_value * n);
        return zt;
      }
      buffer<double> q;
      divide_and_conquer(d.data(), e.data(), n, q);
      buffer<T> at(n * n, T {}); // row j: the eigenvector of d[j], in the tridiagonal's basis, then the matrix's
      parallel_for(n, n, [&](std::size_t j) {
                     for (std::size_t r {0}; r < n; ++r) {
                       at[(j * n) + r] = from_double<T>(q[(r * n) + j]);
                     }
                   });
      to_matrix_basis(reduction, at, n);
      return at;
    }

    //! \brief The Frobenius norm of the strict off-diagonal of the `n × n` \p a.
    [[nodiscard]] inline double jacobi_off_diagonal(const buffer<double>&      a,
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
    inline void jacobi_rotate(buffer<double>&      a,
                              buffer<double>&      v,
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
    inline void cyclic_jacobi(buffer<double>&      a,
                              buffer<double>&      v,
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

    //! \brief The indices of \p diagonal by ascending value.
    [[nodiscard]] inline std::vector<std::size_t> ascending_order(const buffer<double>& diagonal)
    {
      std::vector<std::size_t> order(diagonal.size());
      for (std::size_t i {0}; i < order.size(); ++i) {
        order[i] = i;
      }
      std::sort(order.begin(), order.end(), [&diagonal](std::size_t x, std::size_t y) { return diagonal[x] < diagonal[y]; });
      return order;
    }

    /*!
     * \brief The `n × n` row-major matrix whose column c is row `order[c]` of \p zt (the eigenvectors as rows):
     *        by tiles, the rows of each on its own core. Element by element, each write was a row apart, and
     *        the copy of order 1024 took a tenth of an `eigh` on six x86-64 cores.
     */
    template <typename T>
    [[nodiscard]] buffer<T> columns_by_value(const buffer<T>&                zt,
                                             const std::vector<std::size_t>& order,
                                             std::size_t                     n)
    {
      buffer<T> out(n * n, T {});
      parallel_for((n + transpose_tile - 1) / transpose_tile, transpose_tile * n, [&](std::size_t block) {
                     const std::size_t r0 {block * transpose_tile};
                     const std::size_t r1 {std::min(r0 + transpose_tile, n)};
                     for (std::size_t c0 {0}; c0 < n; c0 += transpose_tile) {
                       for (std::size_t c {c0}; c < std::min(c0 + transpose_tile, n); ++c) {
                         const T* const source {zt.data() + (order[c] * n)};
                         for (std::size_t r {r0}; r < r1; ++r) {
                           out[(r * n) + c] = source[r];
                         }
                       }
                     }
                   });
      return out;
    }
  } // namespace detail

  //! \brief How \ref symmetric_eigen diagonalises.
  enum class eigen_method
  {
    tridiagonal_ql,    //!< Householder tridiagonalisation then implicit QL, as LAPACK's `dsyev`: the vectors by rotations.
    jacobi,            //!< Cyclic Jacobi rotations: several times slower, kept as an independent second method.
    divide_and_conquer //!< Householder tridiagonalisation then divide and conquer, as LAPACK's `dsyevd`: the default.
  };

  /*!
   * \brief The eigenvalues and eigenvectors of a real symmetric matrix.
   *
   * By default the matrix is reduced to tridiagonal form by Householder reflections and the tridiagonal is
   * diagonalised by divide and conquer (\ref eigen_method::divide_and_conquer), the method of LAPACK's
   * `dsyevd`, most of its work in matrix products; implicit QL (\ref eigen_method::tridiagonal_ql, `dsyev`'s)
   * and cyclic Jacobi rotations (\ref eigen_method::jacobi) are the alternatives. Whichever, the eigenvalues are real
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
      detail::require_finite_lower(input, "symmetric eigendecomposition");
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
      buffer<double> a(n_ * n_, double   {});
      buffer<double> diagonal(n_, double {});
      if (method == eigen_method::jacobi) {
        buffer<double>      v(n_ * n_, 0.0);
        double              frobenius_squared {0.0};
        for (std::size_t i {0}; i < n_; ++i) {
          v[(i * n_) + i] = 1.0;
          for (std::size_t j {0}; j < n_; ++j) {
            a[(i * n_) + j]    = input(std::max(i, j), std::min(i, j)) / scale; // the lower triangle, mirrored
            frobenius_squared += a[(i * n_) + j] * a[(i * n_) + j];
          }
        }
        detail::cyclic_jacobi(a, v, n_, std::sqrt(frobenius_squared), detail::jacobi_sweeps);
        for (std::size_t i {0}; i < n_; ++i) {
          diagonal[i] = a[(i * n_) + i] * scale;
        }
        extract_sorted(diagonal, v);
        return;
      }
      for (std::size_t i {0}; i < n_; ++i) { // the lower triangle alone: the reduction reads nothing above it
        for (std::size_t j {0}; j <= i; ++j) {
          a[(i * n_) + j] = input(i, j) / scale;
        }
      }
      buffer<double> zt;
      if (n_ > 0) {
        buffer<double>                               off(n_, double {});
        const detail::tridiagonal_reduction<double>  reduction      {detail::tridiagonalize(a, n_, diagonal, off)};
        if (method == eigen_method::tridiagonal_ql) {
          zt = detail::tridiagonal_basis(reduction, n_);
          detail::tridiagonal_ql(diagonal, off, zt, n_, detail::ql_steps_per_value * n_);
        }
        else {
          zt = detail::tridiagonal_divide_and_conquer(diagonal, off, reduction, n_);
        }
      }
      const std::vector<std::size_t> order             {detail::ascending_order(diagonal)};
      buffer<double>                 values(n_, double {});
      for (std::size_t c {0}; c < n_; ++c) {
        values[c] = diagonal[order[c]] * scale;
      }
      values_  = vector<double> {std::move(values)};
      vectors_ = matrix<double> {n_, n_, detail::columns_by_value(zt, order, n_)};
    }

    //! \brief The eigenvalues, ascending.
    [[nodiscard]] const vector<double>& values() const noexcept
    {
      return values_;
    }

    //! \brief The eigenvectors as columns, ordered to match \ref values.
    [[nodiscard]] const matrix<double>& vectors() const& noexcept
    {
      return vectors_;
    }

    //! \brief The eigenvectors, moved out of an expiring decomposition.
    [[nodiscard]] matrix<double> vectors() &&
    {
      return std::move(vectors_);
    }

  private:

    //! \brief Sorts the eigenvalues \p diagonal ascending and reorders \p v's columns to match.
    void extract_sorted(const buffer<double>& diagonal,
                        const buffer<double>& v)
    {
      std::vector<std::size_t> order(n_);
      for (std::size_t i {0}; i < n_; ++i) {
        order[i] = i;
      }
      std::sort(order.begin(), order.end(),
                [&diagonal](std::size_t x, std::size_t y) {
                  return diagonal[x] < diagonal[y];
                });
      buffer<double> values(n_, double       {});
      buffer<double> vectors(n_ * n_, double {});
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
   * method of LAPACK's `zheevd`), which the divide and conquer of \ref symmetric_eigen diagonalises (implicit QL,
   * `zheev`'s, on request). The eigenvalues
   * are real and returned in ascending order, with \ref vectors `j`-th column the eigenvector of
   * \ref values `[j]` (`A = V Λ Vᴴ`, `V` unitary). Only the lower triangle is read and the imaginary part of
   * the diagonal is ignored, as `numpy.linalg.eigh` reads a matrix by default.
   */
  class hermitian_eigen
  {
  public:

    /*!
     * \brief Diagonalises \p input.
     * \param[in] input  A complex Hermitian matrix (its lower triangle is read).
     * \param[in] method The algorithm for the tridiagonal: \ref eigen_method::divide_and_conquer or
     *                   \ref eigen_method::tridiagonal_ql.
     * \throws std::invalid_argument If \p method is \ref eigen_method::jacobi, which has no complex form here.
     * \throws dimension_error   If \p input is not square.
     * \throws non_finite_error  If an entry is infinite or NaN.
     * \throws convergence_error If an iteration (implicit QL or the secular equation) did not converge within its
     *         bound (not observed).
     */
    explicit hermitian_eigen(const matrix<std::complex<double>>& input,
                             eigen_method                        method = eigen_method::divide_and_conquer)
      : n_(input.rows())
    {
      if (method == eigen_method::jacobi) {
        throw std::invalid_argument("Hermitian eigendecomposition: the Jacobi method is for real symmetric matrices");
      }
      if (input.cols() != n_) {
        throw dimension_error("Hermitian eigendecomposition requires a square matrix");
      }
      detail::require_finite_lower(input, "Hermitian eigendecomposition");
      // Work on the matrix divided by its largest entry, as symmetric_eigen does; the vectors do not depend
      // on it. The entries read: the lower triangle, and the diagonal's real part (its imaginary part is ignored,
      // so it scales nothing: an infinite one made every value NaN, a huge one wrong).
      double scale {0.0};
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j < i; ++j) {
          scale = std::max(scale, std::abs(input(i, j)));
        }
        scale = std::max(scale, std::abs(input(i, i).real()));
      }
      if (scale == 0.0) {
        scale = 1.0;
      }
      buffer<std::complex<double>> a(n_ * n_, std::complex<double> {});
      for (std::size_t i {0}; i < n_; ++i) { // the lower triangle alone, a real diagonal: the reduction reads no more
        for (std::size_t j {0}; j <= i; ++j) {
          a[(i * n_) + j] = (i == j ? std::complex<double> {input(i, j).real(), 0.0} : input(i, j)) / scale;
        }
      }
      buffer<double>               diagonal(n_, double {});
      buffer<std::complex<double>> zt;
      if (n_ > 0) {
        buffer<double>                                             off(n_, double {});
        const detail::tridiagonal_reduction<std::complex<double>>  reduction      {detail::tridiagonalize(a, n_, diagonal, off)};
        if (method == eigen_method::tridiagonal_ql) {
          zt = detail::tridiagonal_basis(reduction, n_);
          detail::tridiagonal_ql(diagonal, off, zt, n_, detail::ql_steps_per_value * n_);
        }
        else {
          zt = detail::tridiagonal_divide_and_conquer(diagonal, off, reduction, n_);
        }
      }
      const std::vector<std::size_t> order             {detail::ascending_order(diagonal)};
      buffer<double>                 values(n_, double {});
      for (std::size_t c {0}; c < n_; ++c) {
        values[c] = diagonal[order[c]] * scale;
      }
      values_  = vector<double> {std::move(values)};
      vectors_ = matrix<std::complex<double>> {n_, n_, detail::columns_by_value(zt, order, n_)};
    }

    //! \brief The eigenvalues, ascending.
    [[nodiscard]] const vector<double>& values() const noexcept
    {
      return values_;
    }

    //! \brief The eigenvectors as columns, ordered to match \ref values.
    [[nodiscard]] const matrix<std::complex<double>>& vectors() const& noexcept
    {
      return vectors_;
    }

    //! \brief The eigenvectors, moved out of an expiring decomposition.
    [[nodiscard]] matrix<std::complex<double>> vectors() &&
    {
      return std::move(vectors_);
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
    void one_sided_jacobi(buffer<T>&      w,
                          buffer<T>&      v,
                          std::size_t     n,
                          std::size_t     max_sweeps)
    {
      // The sweeps rotate columns: work on the transposes, where each column is a contiguous row. A walk
      // down the columns of the row-major arrays strides by n on every access and runs several times slower.
      const auto transpose_square {[n](buffer<T>& m) {
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
            for (buffer<T>* side : {&w, &v}) {
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

    //! \brief The order below which a real bidiagonal reduction goes a reflection at a time, as LAPACK's `dgebrd`
    //!        falls back to `dgebd2`: the panels' products are too small to pay for packing. Order 64: 33 % shorter
    //!        on six x86-64 cores, 7 % on arm64; order 160: 30 % longer on arm64, where x86-64 still gains 8 %.
    inline constexpr std::size_t bidiagonal_crossover {128};

    //! \brief The order below which the bases of a bidiagonal reduction are formed a reflector at a time: the blocks'
    //!        packing and triangular factor cost more than they save. Order 32: 11 % shorter on six x86-64 cores,
    //!        12 % on arm64; order 96: 7 % longer on both.
    inline constexpr std::size_t bidiagonal_basis_crossover {64};

    /*!
     * \brief The bases of a bidiagonal reduction from its reflectors: `U_b = (H₀ H₁ …) diag(conj left phases)` and
     *        `V_b = (H'₀ H'₁ …) diag(right phases)`, each formed by blocks of reflectors through the matrix product
     *        kernel, as QR's `Q` is, or a reflector at a time below \ref bidiagonal_basis_crossover.
     * \param[in]  left        The left reflector of each step (acting from its row), empty where none.
     * \param[in]  right       The right reflector of each step (acting from the column past it), likewise.
     * \param[in]  left_phase  The unit phase each row took.
     * \param[in]  right_phase The unit phase each column past the diagonal took.
     * \param[in]  n           The order.
     * \param[out] ut          `U_bᵀ`, row-major: its rows are the columns of `U_b`.
     * \param[out] vt          `V_bᵀ`, likewise.
     */
    template <typename T>
    void form_bidiagonal_bases(const std::vector<buffer<T>>&      left,
                               std::vector<buffer<T>>             right,
                               const buffer<T>&                   left_phase,
                               const buffer<T>&                   right_phase,
                               std::size_t                        n,
                               buffer<T>&                         ut,
                               buffer<T>&                         vt)
    {
      // U_b = L₀ᴴ L₁ᴴ … with L_kᴴ = H_k diag(conj phase at k), and V_b = R₀ R₁ … with R_k = H'_k diag(phase
      // at k+1). A phase on row k commutes with every later reflector, which acts on rows past it, so
      // U_b = (H₀ H₁ …) diag(conj phases) and V_b = (H'₀ H'₁ …) diag(phases): the reflectors are applied to the
      // diagonal by blocks, through the matrix product kernel, as QR's Q is formed.
      ut.assign(n * n, T {}); // Uᵀ and Vᵀ, formed transposed: diagonal to start, the reflectors applied to them
      vt.assign(n * n, T {});
      for (std::size_t k {0}; k < n; ++k) {
        ut[(k * n) + k] = conj_of(left_phase[k]);
        vt[(k * n) + k] = k == 0 ? T {1} : right_phase[k - 1];
      }
      std::vector<buffer<T>> shifted(n); // right[k] acts from row k+1: as reflector k+1 of a QR
      for (std::size_t k {0}; k + 1 < n; ++k) {
        shifted[k + 1] = std::move(right[k]);
      }
      if (n < bidiagonal_basis_crossover) { // a reflector at a time, the last first, as LAPACK's dorg2r
        for (std::size_t k {n}; k-- > 0;) {
          reflect_columns(ut, n, left[k], k, k, n);
          reflect_columns(vt, n, shifted[k], k, k, n);
        }
        return;
      }
      for (std::size_t k1 {n}; k1 > 0;) { // blocks of qr_block reflectors, the last first
        const std::size_t k0 {(k1 - 1) / qr_block * qr_block};
        apply_transposed_reflector_block(gather_reflectors(left, k0, k1, n), ut, n, k0, n);
        apply_transposed_reflector_block(gather_reflectors(shifted, k0, k1, n), vt, n, k0, n);
        k1 = k0;
      }
    }

    //! \brief The reflections from each side in one panel of the blocked bidiagonal reduction.
    inline constexpr std::size_t bidiagonal_block {32};

    //! \brief The weight of a multiply-add of the bidiagonal reduction's matrix-vector products, which read the trailing
    //!        block once for each column: on x86-64 a thread shares them from a quarter of the work (six cores,
    //!        singular values of order 256: 8.1 → 7.3 ms, 512: 35 → 30); on arm64 that slowed order 256 by 10 %.
#ifdef NUMERICAL_X86_COMPILED
    inline constexpr std::size_t bidiagonal_product_weight {4};
#else
    inline constexpr std::size_t bidiagonal_product_weight {1};
#endif

    template <typename T>
    void bidiagonalize_unblocked(buffer<T>&           b,
                                 std::size_t          n,
                                 buffer<double>&      d,
                                 buffer<double>&      e,
                                 buffer<T>&           ut,
                                 buffer<T>&           vt,
                                 bool                 form = true);

    /*!
     * \brief The unit Householder vector `v` with `(I − 2 v vᵀ) x = β e₁`, in place of \p x, for a real `x`.
     * \return `β = −sign(x₀) ‖x‖`.
     */
    template <typename T>
    [[nodiscard]] T householder_in_place(buffer<T>& x)
    {
      const double up     {rescale_tiny(x)}; // the reflector is the same at any scale; β is scaled back
      const auto   norm   {static_cast<T>(scaled_norm(x))};
      const T      beta   {-phase_of(x[0]) * norm};
      x[0] -= beta;
      const auto length   {static_cast<T>(scaled_norm(x))};
      for (T& element : x) {
        element /= length;
      }
      return beta / static_cast<T>(up);
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
     * \param[out]    ut `U_bᵀ`, row-major: its rows are the columns of `U_b` (left as it is without \p form).
     * \param[out]    vt `V_bᵀ`, row-major: its rows are the columns of `V_b` (likewise).
     * \param[in]     form Whether to form \p ut and \p vt; the bidiagonal alone gives the singular values.
     */
    template <typename T>
    requires std::is_floating_point_v<T>
    void bidiagonalize(buffer<T>&           b,
                       std::size_t          n,
                       buffer<double>&      d,
                       buffer<double>&      e,
                       buffer<T>&           ut,
                       buffer<T>&           vt,
                       bool                 form = true)
    {
      if (n < bidiagonal_crossover) {
        bidiagonalize_unblocked(b, n, d, e, ut, vt, form);
        return;
      }
      std::vector<buffer<T>>      left(n);  // the left reflector of step k, on rows k…, empty where none was needed
      std::vector<buffer<T>>      right(n); // the right reflector of step k, on columns k+1…, likewise
      const auto                  nonzero {[](const T& value) { return value != T {}; }};
      for (std::size_t k0 {0}; k0 < n; k0 += bidiagonal_block) {
        const std::size_t k1    {std::min(k0 + bidiagonal_block, n)};
        const std::size_t width {k1 - k0};
        // By reflector: row p of each holds step k0 + p's vector over the rows (v, x) or the columns (u, y), so every
        // term of the panel is a contiguous update.
        buffer<T>      by_row_v(width * n, T {}); // the left reflectors, at their rows
        buffer<T>      by_row_y(width * n, T {}); // what they move, 2 Bᵀ v, at the columns
        buffer<T>      by_row_u(width * n, T {}); // the right reflectors, at their columns
        buffer<T>      by_row_x(width * n, T {}); // what they move, 2 B u, at the rows
        const auto     row_of {[n](buffer<T>& rows, std::size_t p, std::size_t from) { return rows.data() + (p * n) + from; }};
        for (std::size_t k {k0}; k < k1; ++k) {
          const std::size_t i          {k - k0};
          buffer<T>         x(n - k, T {}); // column k from the diagonal down, the panel's earlier steps applied
          for (std::size_t r {k}; r < n; ++r) {
            x[r - k] = b[(r * n) + k];
          }
          for (std::size_t p {0}; p < i; ++p) {
            subtract_product(x.data(), by_row_y[(p * n) + k], row_of(by_row_v, p, k), n - k);
            subtract_product(x.data(), by_row_u[(p * n) + k], row_of(by_row_x, p, k), n - k);
          }
          if (std::any_of(x.begin() + 1, x.end(), nonzero)) { // something below the diagonal to reflect
            d[k] = static_cast<double>(householder_in_place(x));
            // y = 2 (B − V Yᵀ − X Uᵀ)ᵀ v on the columns past k: Bᵀ v by contiguous rows, then the panel's terms.
            buffer<T> by_v(i, T {});                          // Vᵀ v and Xᵀ v over the earlier steps
            buffer<T> by_x(i, T {});
            for (std::size_t p {0}; p < i; ++p) {
              by_v[p] = dot_product(row_of(by_row_v, p, k), x.data(), n - k);
              by_x[p] = dot_product(row_of(by_row_x, p, k), x.data(), n - k);
            }
            const std::size_t     span        {n - k - 1};
            constexpr std::size_t chunk_width {64};
            // Bᵀ v as the terms −(−v)·B, the chunk held through them (negation is exact, so each column takes the same
            // additions): row by row, each term stored the chunk back (singular values of order 512 on one x86-64
            // core: 46 → 42 ms).
            buffer<T> minus(x.size(), T {});
            std::transform(x.begin(), x.end(), minus.begin(), std::negate<T> {});
            parallel_for((span + chunk_width - 1) / chunk_width, bidiagonal_product_weight * (n - k) * chunk_width, [&](std::size_t chunk) {
                           const std::size_t first {k + 1 + (chunk * chunk_width)};
                           const std::size_t count {std::min(chunk_width, n - first)};
                           buffer<T>    sum(count, T {});
                           subtract_products(sum.data(), minus.data(), b.data() + (k * n) + first, n, n - k, count);
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
          buffer<T> y(b.begin() + static_cast<std::ptrdiff_t>((k * n) + k + 1), // row k past the diagonal, the left
                      b.begin() + static_cast<std::ptrdiff_t>((k + 1) * n));    // reflection included
          for (std::size_t p {0}; p <= i; ++p) {
            subtract_product(y.data(), by_row_v[(p * n) + k], row_of(by_row_y, p, k + 1), n - k - 1);
          }
          for (std::size_t p {0}; p < i; ++p) {
            subtract_product(y.data(), by_row_x[(p * n) + k], row_of(by_row_u, p, k + 1), n - k - 1);
          }
          if (std::any_of(y.begin() + 1, y.end(), nonzero)) { // something past the superdiagonal to reflect
            e[k] = static_cast<double>(householder_in_place(y));
            // x = 2 (B − V Yᵀ − X Uᵀ) u on the rows past k: B u by rows, then the panel's terms.
            buffer<T> by_y(i + 1, T {});                      // Yᵀ u over this and the earlier steps, Uᵀ u over the earlier
            buffer<T> by_u(i, T     {});
            for (std::size_t p {0}; p <= i; ++p) {
              by_y[p] = dot_product(row_of(by_row_y, p, k + 1), y.data(), y.size());
            }
            for (std::size_t p {0}; p < i; ++p) {
              by_u[p] = dot_product(row_of(by_row_u, p, k + 1), y.data(), y.size());
            }
            buffer<T> moved(n - k - 1, T {});
            parallel_for(n - k - 1, bidiagonal_product_weight * 2 * (n - k), [&](std::size_t row) {
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
          const std::size_t rest                    {n - k1};
          buffer<T>         minus_v(rest * width, T {});
          buffer<T>         minus_x(rest * width, T {});
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
      e[n - 1] = 0.0;
      if (form) {
        form_bidiagonal_bases(left, std::move(right), buffer<T>(n, T {1}), buffer<T>(n, T {1}), n, ut, vt);
      }
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
     * \param[out]    ut `U_bᵀ`, row-major: its rows are the columns of `U_b` (left as it is without \p form).
     * \param[out]    vt `V_bᵀ`, row-major: its rows are the columns of `V_b` (likewise).
     * \param[in]     form Whether to form \p ut and \p vt.
     */
    template <typename T>
    void bidiagonalize_unblocked(buffer<T>&           b,
                                 std::size_t          n,
                                 buffer<double>&      d,
                                 buffer<double>&      e,
                                 buffer<T>&           ut,
                                 buffer<T>&           vt,
                                 bool                 form)
    {
      std::vector<buffer<T>>      left(n);  // the left reflector of step k, empty where none was needed
      std::vector<buffer<T>>      right(n); // the right reflector acting on columns k+1…, likewise
      buffer<T>                   left_phase(n, T {1});
      buffer<T>                   right_phase(n, T {1});
      const auto                  reflector {[](buffer<T>& v) {       // unit v with (I − 2 v vᴴ) x ∥ e₁
                                               (void)rescale_tiny(v); // the same reflector at any scale
                                               const double norm {scaled_norm(v)};
                                               v[0] += phase_of(v[0]) * detail::from_double<T>(norm);
                                               const auto length {detail::from_double<T>(scaled_norm(v))};
                                               for (T & element : v) {
                                                 element /= length;
                                               }
                                             }};
      for (std::size_t k {0}; k < n; ++k) {
        buffer<T> x(n - k, T {}); // column k from the diagonal down
        for (std::size_t i {k}; i < n; ++i) {
          x[i - k] = b[(i * n) + k];
        }
        if (scaled_norm(x.data() + 1, x.size() - 1) > 0.0) {
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
        buffer<T> y(n - k - 1, T {}); // row k right of the diagonal, conjugated: B ← B H with H yᴴ ∥ e₁
        for (std::size_t j {k + 1}; j < n; ++j) {
          y[j - k - 1] = conj_of(b[(k * n) + j]);
        }
        if (y.size() > 1 && scaled_norm(y.data() + 1, y.size() - 1) > 0.0) {
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
      if (form) {
        form_bidiagonal_bases(left, std::move(right), left_phase, right_phase, n, ut, vt);
      }
    }

    //! \brief \ref bidiagonalize_unblocked, for a complex or any other `T` the panels do not take.
    template <typename T>
    requires(!std::is_floating_point_v<T>)
    void bidiagonalize(buffer<T>&           b,
                       std::size_t          n,
                       buffer<double>&      d,
                       buffer<double>&      e,
                       buffer<T>&           ut,
                       buffer<T>&           vt,
                       bool                 form = true)
    {
      bidiagonalize_unblocked(b, n, d, e, ut, vt, form);
    }

    /*!
     * \brief The singular values of the real upper bidiagonal (\p d, \p e) by implicit-shift QR with
     *        deflation (LINPACK's `dsvdc`, the Golub–Kahan–Reinsch iteration), the rotations accumulated
     *        into the rows of \p ut and \p vt.
     * \param[in,out] d         The diagonal; on return, the singular values, descending and non-negative.
     * \param[in,out] e         The superdiagonal in `e[0…n-2]`, `e[n-1] = 0`; destroyed.
     * \param[in,out] ut        Row-major `n × n`; on return, row `j` is the left vector of `d[j]`. Empty, with \p vt:
     *                          the singular values alone.
     * \param[in,out] vt        Row-major `n × n`; on return, row `j` is the right vector of `d[j]`.
     * \param[in]     n         The order.
     * \param[in]     max_steps The bound of QR steps in all.
     * \throws convergence_error If the iteration has not converged after \p max_steps steps.
     */
    template <typename T>
    void bidiagonal_qr(buffer<double>&      d,
                       buffer<double>&      e,
                       buffer<T>&           ut,
                       buffer<T>&           vt,
                       std::size_t          n,
                       std::size_t          max_steps)
    {
      // The iteration reads only d and e: what it does to the rows of vt and ut is recorded and applied once
      // at the end, in parallel by columns.
      std::vector<row_operation> on_vt;
      std::vector<row_operation> on_ut;
      const bool                 record  {!vt.empty()}; // the singular values alone record nothing
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
            if (record) {
              on_vt.push_back({row_operation::kind::negate, top, top, 0.0, 0.0});
            }
          }
          for (std::size_t j {top}; j + 1 < n && d[j] < d[j + 1]; ++j) {
            std::swap(d[j], d[j + 1]);
            if (record) {
              on_vt.push_back({row_operation::kind::swap, j, j + 1, 0.0, 0.0});
              on_ut.push_back({row_operation::kind::swap, j, j + 1, 0.0, 0.0});
            }
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
            const double t  {rotation_length(d[j], f)};
            const double cs {d[j] / t};
            const double sn {f / t};
            d[j] = t;
            if (j != top) {
              f        = -sn * e[j - 1];
              e[j - 1] = cs * e[j - 1];
            }
            if (record) {
              on_vt.push_back({row_operation::kind::rotate, j, p - 1, cs, sn});
            }
          }
          continue;
        }
        if (zero != none) { // split at d[zero] = 0: chase e[zero] along the row with rotations on the left
          double f {e[zero]};
          e[zero] = 0.0;
          for (std::size_t j {zero + 1}; j < p; ++j) {
            const double t  {rotation_length(d[j], f)};
            const double cs {d[j] / t};
            const double sn {f / t};
            d[j] = t;
            f    = -sn * e[j];
            e[j] = cs * e[j];
            if (record) {
              on_ut.push_back({row_operation::kind::rotate, j, zero, cs, sn});
            }
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
          double t  {rotation_length(f, g)};
          double cs {f / t};
          double sn {g / t};
          if (j != top) {
            e[j - 1] = t;
          }
          f        = (cs * d[j]) + (sn * e[j]);
          e[j]     = (cs * e[j]) - (sn * d[j]);
          g        = sn * d[j + 1];
          d[j + 1] = cs * d[j + 1];
          if (record) {
            on_vt.push_back({row_operation::kind::rotate, j, j + 1, cs, sn});
          }
          t        = rotation_length(f, g);
          cs       = f / t;
          sn       = g / t;
          d[j]     = t;
          f        = (cs * e[j]) + (sn * d[j + 1]);
          d[j + 1] = (cs * d[j + 1]) - (sn * e[j]);
          g        = sn * e[j + 1];
          e[j + 1] = cs * e[j + 1];
          if (record) {
            on_ut.push_back({row_operation::kind::rotate, j, j + 1, cs, sn});
          }
        }
        e[p - 2] = f;
      }
      if (!vt.empty()) {
        apply_row_operations(vt, n, on_vt);
        apply_row_operations(ut, n, on_ut);
      }
    }

    //! \brief The conjugate transpose of \p a.
    template <typename T>
    [[nodiscard]] matrix<T> adjoint(const matrix<T>& a)
    {
      buffer<T> entries(a.rows() * a.cols(), T {});
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
    [[nodiscard]] inline double singular_secular_root(const buffer<double>&      d,
                                                      const buffer<double>&      z,
                                                      std::size_t                i,
                                                      std::size_t                max_iterations,
                                                      double*                    below,
                                                      double*                    above)
    {
      buffer<double>         delta(d.size(), double {});
      const secular_solution root                   {solve_secular(z, 1.0, i, max_iterations, delta.data(), [&d](std::size_t o, std::size_t j) {
                                                                     return (d[j] - d[o]) * (d[j] + d[o]);
                                                                   })};
      const double           pole                   {d[root.origin]};
      const double           shift                  {root.offset / (pole + std::sqrt((pole * pole) + root.offset))}; // σ − d_o from σ² − d_o²
      for (std::size_t j {0}; j < d.size(); ++j) {
        below[j] = (d[j] - pole) - shift;
        above[j] = (d[j] + pole) + shift;
      }
      return pole + shift;
    }

    //! \brief Columns \p a and \p b of the row-major \p m (\p cols per row) rotated: `a ← c a + s b`, `b ← c b − s a`.
    inline void rotate_columns(buffer<double>&      m,
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
     * \param[in]     split      The row of \p u the arrow's first row is (its upper half above, its lower below):
     *                           the bases are block diagonal by it, and each half's rows take only the columns
     *                           that reach them (`dlasd3` multiplies the halves apart the same way).
     */
    inline void merge_arrow(buffer<double>&      d,
                            buffer<double>       z,
                            double               alpha,
                            double               beta,
                            buffer<double>&      u,
                            buffer<double>&      v,
                            std::size_t          n,
                            std::size_t          m,
                            std::size_t          split)
    {
      // The rows each column reaches: 1 the upper half's, 2 u's row `split` (the arrow's first), 4 the lower half's.
      // A rotation of two columns makes each reach the rows of both.
      std::vector<unsigned> reach_u(n, 4U);
      std::vector<unsigned> reach_v(m, 4U);
      reach_u[0] = 2U;
      reach_v[0] = m > n ? 5U : 1U; // the upper half's null vector, turned with the lower one's when there is one
      for (std::size_t j {1}; j <= split; ++j) {
        reach_u[j] = 1U;
        reach_v[j] = 1U;
      }
      const auto turned {[](std::vector<unsigned>& reach, std::size_t a, std::size_t b) {
                           reach[a] = reach[b] = reach[a] | reach[b];
                         }};
      std::vector<std::size_t> order(n - 1);
      for (std::size_t j {1}; j < n; ++j) {
        order[j - 1] = j;
      }
      std::sort(order.begin(), order.end(), [&d](std::size_t x, std::size_t y) { return d[x] < d[y]; });
      double largest {std::max(std::abs(alpha), std::abs(beta))};
      for (const double value : d) {
        largest = std::max(largest, value);
      }
      if (largest == 0.0) {
        return; // a zero arrow: its singular values are zeros, its vectors the bases as they are
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
          turned(reach_v, 0, j);
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
          turned(reach_v, j, pending);
          turned(reach_u, j, pending);
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
      const std::size_t   k                 {kept.size()};
      buffer<double>      poles(k, double   {});
      buffer<double>      weights(k, double {});
      for (std::size_t i {0}; i < k; ++i) {
        poles[i]   = d[kept[i]];
        weights[i] = z[kept[i]];
      }
      buffer<double> roots(k, double     {});
      buffer<double> below(k * k, double {}); // below[i * k + j] = poles[j] − roots[i]
      buffer<double> above(k * k, double {}); // above[i * k + j] = poles[j] + roots[i]
      parallel_for(k, 8 * k, [&](std::size_t i) {
                     roots[i] = singular_secular_root(poles, weights, i, secular_iterations, below.data() + (i * k),
                                                      above.data() + (i * k));
                   });
      buffer<double> left(k * k, double  {}); // column i: the left vector of roots[i], in the kept columns' basis
      buffer<double> right(k * k, double {}); // and the right one
      // Löwner: ẑⱼ² = Πᵢ (σᵢ² − dⱼ²) / Πᵢ≠ⱼ (dᵢ² − dⱼ²), each pole paired with its root, by blocks of poles on every
      // core, each pole's product in ascending order of the roots (a row of the gaps at a time, along the block).
      buffer<double> signed_weights(k, double {});
      parallel_for((k + lowner_block - 1) / lowner_block, lowner_block * k, [&](std::size_t block) {
                     const std::size_t                first {block * lowner_block};
                     const std::size_t                last {std::min(first + lowner_block, k)};
                     std::array<double, lowner_block> product {};
                     for (std::size_t j {first}; j < last; ++j) {
                       product[j - first] = -below[(j * k) + j] * above[(j * k) + j];
                     }
                     for (std::size_t i {0}; i < k; ++i) {
                       for (std::size_t j {first}; j < last; ++j) {
                         if (i != j) {
                           product[j - first] *= (-below[(i * k) + j] * above[(i * k) + j]) / ((poles[i] - poles[j]) * (poles[i] + poles[j]));
                         }
                       }
                     }
                     for (std::size_t j {first}; j < last; ++j) {
                       signed_weights[j] = std::copysign(std::sqrt(product[j - first]), weights[j]);
                     }
                   });
      // The vectors by rows of poles, each on its own core: right[j][i] = ẑⱼ / (dⱼ² − σᵢ²), read a column of the gaps.
      parallel_for(k, 4 * k, [&](std::size_t j) {
                     for (std::size_t i {0}; i < k; ++i) {
                       const double ratio {signed_weights[j] / (below[(i * k) + j] * above[(i * k) + j])}; // ẑⱼ / (dⱼ² − σᵢ²)
                       right[(j * k) + i] = ratio;
                       left[(j * k) + i]  = j == 0 ? -1.0 : poles[j] * ratio;
                     }
                   });
      for (buffer<double>* vectors : {&left, &right}) {
        buffer<double> length(k, 0.0);
        for (std::size_t j {0}; j < k; ++j) { // by rows, each a contiguous update of every column's sum
          for (std::size_t i {0}; i < k; ++i) {
            length[i] += (*vectors)[(j * k) + i] * (*vectors)[(j * k) + i];
          }
        }
        for (double& l : length) {
          l = std::sqrt(l);
        }
        parallel_for(k, k, [&](std::size_t j) {
                       for (std::size_t i {0}; i < k; ++i) {
                         (*vectors)[(j * k) + i] /= length[i];
                       }
                     });
      }
      // The kept columns times the arrow's vectors; the deflated columns as they are; then by ascending value.
      // By segments of rows, each taking the kept columns that reach it and their rows of the arrow's vectors: the
      // others are zero there, and an element's sum without them is the same to the bit.
      using segment = std::tuple<std::size_t, std::size_t, unsigned>; // first row, rows, the bit of reach
      const auto carried {[&](const buffer<double>& basis, std::size_t rows, const buffer<double>& small,
                              const std::vector<unsigned>& reach, const std::vector<segment>& segments) {
                            buffer<double> product(rows * k, double {});
                            for (const auto& [first, count, bit] : segments) {
                              std::vector<std::size_t> reaching;
                              for (std::size_t i {0}; i < k; ++i) {
                                if ((reach[kept[i]] & bit) != 0U) {
                                  reaching.push_back(i);
                                }
                              }
                              const std::size_t   width {reaching.size()};
                              buffer<double> columns(count * width, double {});
                              buffer<double> factor(width * k, double      {});
                              parallel_for(count, width + k, [&](std::size_t r) {
                                             for (std::size_t c {0}; c < width; ++c) {
                                               columns[(r * width) + c] = basis[((first + r) * rows) + kept[reaching[c]]];
                                             }
                                             std::fill_n(product.data() + ((first + r) * k), k, 0.0); // the product adds to it
                                           });
                              for (std::size_t c {0}; c < width; ++c) {
                                std::copy_n(small.data() + (reaching[c] * k), k, factor.data() + (c * k));
                              }
                              gemm(count, k, width, columns.data(), width, factor.data(), k, product.data() + (first * k), k);
                            }
                            return product;
                          }};
      const buffer<double>      new_u            {carried(u, n, left, reach_u, {{0, split, 1U}, {split, 1, 2U}, {split + 1, n - split - 1, 4U}})};
      const buffer<double>      new_v            {carried(v, m, right, reach_v, {{0, split + 1, 1U}, {split + 1, m - split - 1, 4U}})};
      buffer<double>            values(n, double {});
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
      std::vector<std::size_t> source(n); // by sorted column: a kept root (< k) or a deflated column (k + its place)
      for (std::size_t c {0}; c < n; ++c) {
        source[c] = from[rank[c]];
        d[c]      = values[rank[c]];
      }
      // Row by row on every core, each row written whole (by columns, each write was a row apart).
      buffer<double> sorted_u(n * n, double {});
      buffer<double> sorted_v(v); // the null column, if any, stays last
      parallel_for(m, n, [&](std::size_t r) { // m ≥ n: the right basis has the rows of both
                     for (std::size_t c {0}; c < n; ++c) {
                       if (r < n) {
                         sorted_u[(r * n) + c] = source[c] < k ? new_u[(r * k) + source[c]] : u[(r * n) + deflated[source[c] - k]];
                       }
                       sorted_v[(r * m) + c] = source[c] < k ? new_v[(r * k) + source[c]] : v[(r * m) + deflated[source[c] - k]];
                     }
                   });
      u = std::move(sorted_u);
      v = std::move(sorted_v);
    }

    //! \brief The order up to which \ref bidiagonal_divide decomposes by implicit-shift QR.
    inline constexpr std::size_t bidiagonal_leaf {32};

    /*!
     * \brief One block of \ref bidiagonal_divide's tree: the bidiagonal it decomposes (its own copies scaled by a
     *        power of two, or its parent's), the two halves it splits into, and its decomposition once solved.
     */
    struct bidiagonal_node
    {
      const double*  d        {};  //!< The diagonal it is solved on.
      const double*  e        {};  //!< The superdiagonal it is solved on.
      std::size_t    n        {};  //!< Its rows.
      std::size_t    extra    {};  //!< Its columns past its rows, 0 or 1.
      int            exponent {0}; //!< The power of two it is scaled down by.
      buffer<double> scaled_d {};  //!< The scaled diagonal, when \p exponent is not 0.
      buffer<double> scaled_e {};  //!< The scaled superdiagonal, when \p exponent is not 0.
      std::size_t    k        {0}; //!< The row it splits at; 0 for a leaf.
      std::size_t    first    {};  //!< The block above row \p k.
      std::size_t    second   {};  //!< The block below row \p k.
      std::size_t    depth    {};  //!< Its distance from the root.
      buffer<double> sigma    {};  //!< Its singular values, ascending.
      buffer<double> u        {};  //!< Its left singular vectors, as columns.
      buffer<double> v        {};  //!< Its right singular vectors, as columns.
    };

    /*!
     * \brief Appends the block (\p d, \p e, \p n, \p extra) and everything below it to \p nodes, parents before
     *        children, each scaled to work near unit size by a power of two (exactly): the secular equation works on
     *        squares, which overflow past 1e154.
     * \return Its index in \p nodes.
     */
    inline std::size_t plan_bidiagonal(std::vector<bidiagonal_node>& nodes,
                                       const double*                 d,
                                       const double*                 e,
                                       std::size_t                   n,
                                       std::size_t                   extra,
                                       std::size_t                   depth)
    {
      double largest {0.0};
      for (std::size_t i {0}; i < n; ++i) {
        largest = std::max({largest, std::abs(d[i]), i + 1 < n + extra ? std::abs(e[i]) : 0.0});
      }
      bidiagonal_node node {.d = d, .e = e, .n = n, .extra = extra, .depth = depth};
      (void)std::frexp(largest, &node.exponent);
      if (node.exponent != 0) {
        node.scaled_d.assign(n, double {});
        node.scaled_e.assign(n, 0.0);
        for (std::size_t i {0}; i < n; ++i) {
          node.scaled_d[i] = std::ldexp(d[i], -node.exponent);
          node.scaled_e[i] = i + 1 < n + extra ? std::ldexp(e[i], -node.exponent) : 0.0;
        }
        node.d = node.scaled_d.data(); // the buffers' storage stays put as the node moves into the vector
        node.e = node.scaled_e.data();
      }
      const std::size_t index {nodes.size()};
      if (n <= bidiagonal_leaf) {
        nodes.push_back(std::move(node));
        return index;
      }
      node.k = n / 2;
      const double* const work_d {node.d};
      const double* const work_e {node.e};
      const std::size_t   k      {node.k};
      nodes.push_back(std::move(node));
      const std::size_t first    {plan_bidiagonal(nodes, work_d, work_e, k, 1, depth + 1)};
      const std::size_t second   {plan_bidiagonal(nodes, work_d + k + 1, work_e + k + 1, n - k - 1, extra, depth + 1)};
      nodes[index].first  = first;
      nodes[index].second = second;
      return index;
    }

    //! \brief Its singular values and vectors into \p node's, from scratch: \p u zero, \p v the identity.
    inline void start_bidiagonal_node(bidiagonal_node& node)
    {
      const std::size_t m {node.n + node.extra};
      node.sigma.assign(node.n, 0.0);
      node.u.assign(node.n * node.n, 0.0);
      node.v.assign(m * m, 0.0);
      for (std::size_t i {0}; i < m; ++i) {
        node.v[(i * m) + i] = 1.0;
      }
    }

    //! \brief Decomposes the leaf \p node by implicit-shift QR.
    inline void solve_bidiagonal_leaf(bidiagonal_node& node)
    {
      start_bidiagonal_node(node);
      const double* const d     {node.d};
      const double* const e     {node.e};
      const std::size_t   n     {node.n};
      const std::size_t   extra {node.extra};
      const std::size_t   m     {n + extra};
      buffer<double>&     sigma {node.sigma};
      buffer<double>&     u     {node.u};
      buffer<double>&     v     {node.v};
      buffer<double>      diagonal(d, d + n);
      buffer<double>      super(n, 0.0);
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
      buffer<double> ut(n * n, 0.0);
      buffer<double> vt(n * n, 0.0);
      for (std::size_t i {0}; i < n; ++i) {
        ut[(i * n) + i] = 1.0;
        vt[(i * n) + i] = 1.0;
      }
      bidiagonal_qr(diagonal, super, ut, vt, n, svd_qr_steps_per_value * n);   // descending
      buffer<double> rotated(v);
      for (std::size_t c {0}; c < n; ++c) {
        const std::size_t source {n - 1 - c};
        sigma[c] = diagonal[source];
        for (std::size_t r {0}; r < n; ++r) {
          u[(r * n) + c] = ut[(source * n) + r];
        }
        for (std::size_t r {0}; r < m; ++r) {   // G times B''s right vector
          double sum {0.0};
          for (std::size_t t {0}; t < n; ++t) {
            sum += rotated[(r * m) + t] * vt[(source * n) + t];
          }
          v[(r * m) + c] = sum;
        }
      }
    }

    //! \brief Joins the two decomposed halves of \p node, \p one and \p two, by \ref merge_arrow, \p node started
    //!        (\ref start_bidiagonal_node).
    inline void join_bidiagonal_node(bidiagonal_node&       node,
                                     const bidiagonal_node& one,
                                     const bidiagonal_node& two)
    {
      const std::size_t     n      {node.n};
      const std::size_t     extra  {node.extra};
      const std::size_t     m      {n + extra};
      const std::size_t     k      {node.k};
      const std::size_t     lower  {n - k - 1};
      const double          alpha  {node.d[k]};
      const double          beta   {node.e[k]};
      const buffer<double>& sigma1 {one.sigma};
      const buffer<double>& u1     {one.u};
      const buffer<double>& v1     {one.v};
      const buffer<double>& sigma2 {two.sigma};
      const buffer<double>& u2     {two.u};
      const buffer<double>& v2     {two.v};
      buffer<double>&       u      {node.u};
      buffer<double>&       v      {node.v};
      const std::size_t     m1     {k + 1};
      const std::size_t     m2     {lower + extra};
      buffer<double>        arrow(n, 0.0);
      buffer<double>        z(n, 0.0);
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
      merge_arrow(arrow, std::move(z), alpha, beta, u, v, n, m, k);
      node.sigma = std::move(arrow);
    }

    //! \brief The singular values of a solved \p node scaled back.
    inline void unscale_bidiagonal_node(bidiagonal_node& node)
    {
      if (node.exponent != 0) {
        for (double& value : node.sigma) {
          value = std::ldexp(value, node.exponent);
        }
      }
    }

    //! \brief Joins the halves of \p nodes' block \p index, then lets them go.
    inline void join_bidiagonal_halves(std::vector<bidiagonal_node>& nodes,
                                       std::size_t                   index)
    {
      bidiagonal_node& node {nodes[index]};
      join_bidiagonal_node(node, nodes[node.first], nodes[node.second]);
      unscale_bidiagonal_node(node);
      nodes[node.first]  = bidiagonal_node {}; // read by no other join
      nodes[node.second] = bidiagonal_node {};
    }

    //! \brief Decomposes the block \p index of \p nodes and everything below it, each block right after its halves.
    inline void solve_bidiagonal_subtree(std::vector<bidiagonal_node>& nodes,
                                         std::size_t                   index)
    {
      if (nodes[index].k == 0) {
        solve_bidiagonal_leaf(nodes[index]);
        unscale_bidiagonal_node(nodes[index]);
        return;
      }
      start_bidiagonal_node(nodes[index]); // before its halves, as the recursion did: their buffers then come back to it
      solve_bidiagonal_subtree(nodes, nodes[index].first);
      solve_bidiagonal_subtree(nodes, nodes[index].second);
      join_bidiagonal_halves(nodes, index);
    }

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
     *
     * The tree is laid out first (\ref plan_bidiagonal). The blocks at the depth that gives each thread one, and the
     * leaves above it, are independent of one another and go to the threads together, each decomposed on one
     * thread, a block right after its halves while they are still in its caches; the joins above them then come a
     * level at a time, those of one level together, a level of one join sharing it among the threads. Every block
     * takes the same inputs either way, so the result is the same to the bit; on one thread the order is the
     * recursion's.
     */
    inline void bidiagonal_divide(const double*        d,
                                  const double*        e,
                                  std::size_t          n,
                                  std::size_t          extra,
                                  buffer<double>&      sigma,
                                  buffer<double>&      u,
                                  buffer<double>&      v)
    {
      std::vector<bidiagonal_node> nodes;
      (void)plan_bidiagonal(nodes, d, e, n, extra, 0);
      const auto               cut {static_cast<std::size_t>(std::bit_width(max_threads() - 1))};
      std::vector<std::size_t> tasks;
      for (std::size_t i {0}; i < nodes.size(); ++i) {
        if (nodes[i].depth == cut || (nodes[i].depth < cut && nodes[i].k == 0)) {
          tasks.push_back(i);
        }
      }
      const std::size_t task_order {nodes[tasks[0]].n};
      parallel_for(tasks.size(), tasks.size() > 1 ? task_order * task_order * task_order : 0, [&](std::size_t t) {
                     solve_bidiagonal_subtree(nodes, tasks[t]);
                   });
      for (std::size_t level {cut}; level-- > 0;) {
        std::vector<std::size_t> joins;
        for (std::size_t i {0}; i < nodes.size(); ++i) {
          if (nodes[i].k != 0 && nodes[i].depth == level) {
            joins.push_back(i);
          }
        }
        const std::size_t order {joins.empty() ? 0 : nodes[joins[0]].n};
        parallel_for(joins.size(), joins.size() > 1 ? order * order * order / 4 : 0, [&](std::size_t j) {
                       start_bidiagonal_node(nodes[joins[j]]);
                       join_bidiagonal_halves(nodes, joins[j]);
                     });
      }
      sigma = std::move(nodes[0].sigma);
      u     = std::move(nodes[0].u);
      v     = std::move(nodes[0].v);
    }

    /*!
     * \brief The singular values of the upper bidiagonal (\p d, \p e) into \p d, descending, the singular vectors
     *        accumulated into the rows of \p ut and \p vt as \ref bidiagonal_qr accumulates them, by divide and
     *        conquer (\ref bidiagonal_divide): most of the work in matrix products.
     * \throws convergence_error If an iteration did not converge within its bound.
     */
    template <typename T>
    void bidiagonal_divide_and_conquer(buffer<double>&      d,
                                       buffer<double>&      e,
                                       buffer<T>&           ut,
                                       buffer<T>&           vt,
                                       std::size_t          n)
    {
      if (n == 0) {
        return;
      }
      buffer<double> sigma;
      buffer<double> u;
      buffer<double> v;
      bidiagonal_divide(d.data(), e.data(), n, 0, sigma, u, v);
      buffer<double> left(n * n, double  {}); // row c: the vectors of the c-th largest value
      buffer<double> right(n * n, double {});
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
    bidiagonal_qr,     //!< Golub–Kahan bidiagonalisation then implicit-shift QR, as LAPACK's `dgesvd`.
    jacobi,            //!< One-sided Jacobi: several times slower, each singular value to high relative accuracy.
    divide_and_conquer //!< Golub–Kahan bidiagonalisation then divide and conquer, as LAPACK's `dgesdd`: the default.
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
    [[nodiscard]] const matrix<T>& u() const& noexcept
    {
      return u_;
    }

    //! \brief `U`, moved out of an expiring decomposition (\ref v and the singular values stay).
    [[nodiscard]] matrix<T> u() &&
    {
      return std::move(u_);
    }

    //! \brief The `n × k` right singular vectors `V` (columns).
    [[nodiscard]] const matrix<T>& v() const& noexcept
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
      buffer<T> entries(rows * cols, T {});
      for (std::size_t i {0}; i < rows; ++i) {
        for (std::size_t j {0}; j < cols; ++j) {
          entries[(i * cols) + j] = a(i, j) / detail::from_double<T>(scale);
        }
      }
      // A square matrix is reduced to bidiagonal form as it is (as LAPACK's dgesdd does short of a tall one): a QR
      // first, its Q formed and multiplied in at the end, took a third of the decomposition's time at order 1024.
      const bool         direct  {rows == cols && method != svd_method::jacobi};
      detail::thin_qr<T> factors {direct ? detail::thin_qr<T> {.q = {}, .r = std::move(entries)}
                                         : detail::householder_qr(std::move(entries), rows, cols)};
      if (method != svd_method::jacobi) {
        buffer<double> d(cols, double {});
        buffer<double> e(cols, double {});
        buffer<T>      ut;
        buffer<T>      vt;
        detail::bidiagonalize(factors.r, cols, d, e, ut, vt);
        if (method == svd_method::bidiagonal_qr) {
          detail::bidiagonal_qr(d, e, ut, vt, cols, detail::svd_qr_steps_per_value * cols);
        }
        else {
          detail::bidiagonal_divide_and_conquer(d, e, ut, vt, cols);
        }
        buffer<T> left(cols * cols, T  {});
        buffer<T> right(cols * cols, T {});
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
        u_               = direct ? matrix<T> {cols, cols, std::move(left)}
                                  : matrix<T> {rows, cols, std::move(factors.q)} *matrix<T> {cols, cols, std::move(left)};
        v_               = matrix<T> {cols, cols, std::move(right)};
        return;
      }
      buffer<T>     w                    {std::move(factors.r)}; // A = Q R, and R V = W with orthogonal columns
      buffer<T>     right(cols * cols, T {});
      for (std::size_t i {0}; i < cols; ++i) {
        right[(i * cols) + i] = T {1};
      }
      detail::one_sided_jacobi(w, right, cols, detail::jacobi_sweeps);

      buffer<double> norms(cols, double {});
      for (std::size_t j {0}; j < cols; ++j) {
        buffer<T> column(cols, T {});
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
      buffer<T>      left(cols * cols, T         {});
      buffer<T>      sorted_right(cols * cols, T {});
      buffer<double> values(cols, double         {});
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
        buffer<T>      best;
        double         best_norm {-1.0};
        for (std::size_t k {0}; k < cols; ++k) {
          buffer<T> candidate(cols, T {});
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
    static void orthogonalise(buffer<T>&            x,
                              const buffer<T>&      basis,
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

    //! \brief How far under the reciprocal of \ref rank_tolerance a bound on the condition number must stay for
    //!        \ref lstsq_by_qr to answer: nearer, the singular value decomposition decides the rank.
    inline constexpr double qr_rank_margin {100.0};

    //! \brief \p x times two to the \p exponent, exactly where the result is a normal number (no power of two is
    //!        formed, so none overflows).
    template <typename T>
    [[nodiscard]] T scaled_by_two(const T& x,
                                  int      exponent)
    {
      return std::scalbn(x, exponent);
    }

    //! \brief \ref scaled_by_two for a complex number, part by part.
    template <typename U>
    [[nodiscard]] std::complex<U> scaled_by_two(const std::complex<U>& x,
                                                int                    exponent)
    {
      return std::complex<U> {std::scalbn(x.real(), exponent), std::scalbn(x.imag(), exponent)};
    }

    /*!
     * \brief `R⁻¹`, upper triangular, of the `n × n` upper-triangular \p r (row-major): back substitution against
     *        the identity in parallel chunks of columns, each row's terms in registers (\ref subtract_products). A
     *        chunk's rows past its last column stay zero; a zero on the diagonal leaves infinities or NaN.
     */
    template <typename T>
    [[nodiscard]] buffer<T> upper_inverse(const buffer<T>&      r,
                                          std::size_t           n)
    {
      constexpr std::size_t width      {64};
      buffer<T>             x(n * n, T {});
      parallel_for((n + width - 1) / width, n * n * width / 2, [&](std::size_t chunk) {
                     const std::size_t first {chunk * width};
                     const std::size_t last {std::min(first + width, n)};
                     for (std::size_t i {last}; i-- > 0;) {
                       T* const row {x.data() + (i * n) + first};
                       if (i >= first) {
                         row[i - first] = T {1};
                       }
                       subtract_products(row, r.data() + (i * n) + i + 1, x.data() + ((i + 1) * n) + first, n, last - (i + 1), last - first);
                       for (std::size_t c {0}; c < last - first; ++c) {
                         row[c] = row[c] / r[(i * n) + i];
                       }
                     }
                   });
      return x;
    }

    /*!
     * \brief The least-squares solution of `A X ≈ B` by a QR factorisation, of `A`, or of `Aᴴ` for a wide `A` (the
     *        minimum-norm solution), where it is the one the singular value decomposition gives: with `A` divided by
     *        a power of two near its largest magnitude, `κ₂(R) ≤ ‖R‖_F ‖R⁻¹‖_F` must stay \ref qr_rank_margin times
     *        under the reciprocal of \ref rank_tolerance, so that every singular value is far above it and the rank
     *        full (a zero on `R`'s diagonal makes the bound infinite or NaN, which fails). Else nothing. A fraction
     *        of the decomposition's work, the rest in matrix products.
     * \param[in]  a The `m × n` matrix `A`.
     * \param[in]  b The right-hand sides, one per column.
     * \param[out] r Where given, the square `R` factorised, of `A` divided by `2^exponent`, and that exponent: its
     *              singular values are `A`'s over `2^exponent`.
     * \return The solution, or nothing where the decomposition must decide the rank.
     */
    template <typename T>
    [[nodiscard]] std::optional<matrix<T>> lstsq_by_qr(const matrix<T>&                       a,
                                                       const matrix<T>&                       b,
                                                       std::pair<buffer<T>, int>* const       r = nullptr)
    {
      const std::size_t m       {a.rows()};
      const std::size_t n       {a.cols()};
      const bool        wide    {m < n};
      const std::size_t rows    {wide ? n : m}; // of the matrix factorised
      const std::size_t cols    {wide ? m : n};
      double            largest {0.0};
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          largest = std::max(largest, static_cast<double>(std::abs(a(i, j))));
        }
      }
      if (largest == 0.0) {
        return std::nullopt;
      }
      const int      exponent               {std::ilogb(largest)};
      buffer<T>      entries(rows * cols, T {});
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          if (wide) {
            entries[(j * cols) + i] = scaled_by_two(conj_of(a(i, j)), -exponent);
          }
          else {
            entries[(i * cols) + j] = scaled_by_two(a(i, j), -exponent);
          }
        }
      }
      qr_reflections<T>       factors         {householder_reflect(std::move(entries), rows, cols)};
      const buffer<T>         inverse         {upper_inverse(factors.r, cols)};
      double                  r_squares       {0.0};
      double                  inverse_squares {0.0};
      for (std::size_t e {0}; e < cols * cols; ++e) {
        r_squares       += static_cast<double>(std::norm(factors.r[e]));
        inverse_squares += static_cast<double>(std::norm(inverse[e]));
      }
      const double condition {std::sqrt(r_squares) * std::sqrt(inverse_squares)};
      if (!(condition * rank_tolerance(1.0, m, n) * qr_rank_margin < 1.0)) {
        return std::nullopt;
      }
      const std::size_t k          {b.cols()};
      buffer<T>         x(n * k, T {});
      if (wide) { // x = Q [R⁻ᴴ B; 0]: the blocks applied last first
        const matrix<T>      inverse_adjoint {adjoint(matrix<T> {cols, cols, inverse})};
        const buffer<T>      right(b.begin(), b.end());
        gemm(cols, k, cols, &*inverse_adjoint.begin(), cols, right.data(), k, x.data(), k);
        for (std::size_t block {factors.blocks.size()}; block-- > 0;) {
          apply_reflector_block(factors.blocks[block], false, x, k, 0, k);
        }
      }
      else { // x = R⁻¹ times the first rows of Qᴴ B: the blocks' adjoints applied first first
        buffer<T> y(b.begin(), b.end());
        for (const reflector_block<T>& block : factors.blocks) {
          apply_reflector_block(block, true, y, k, 0, k);
        }
        gemm(n, k, cols, inverse.data(), cols, y.data(), k, x.data(), k);
      }
      for (T& value : x) {
        value = scaled_by_two(value, -exponent);
      }
      if (r != nullptr) {
        *r = {std::move(factors.r), exponent};
      }
      return matrix<T> {n, k, std::move(x)};
    }

    /*!
     * \brief The singular values of the square `n × n` row-major \p b, descending, without its vectors: its Golub–Kahan
     *        bidiagonal and the implicit-shift QR iteration on it, no basis formed nor rotated.
     * \throws convergence_error If the iteration does not converge within its bound (not observed).
     */
    template <typename T>
    [[nodiscard]] buffer<double> square_singular_values(buffer<T>      b,
                                                        std::size_t    n)
    {
      buffer<double> d(n, double {});
      buffer<double> e(n, double {});
      buffer<T>      none;
      buffer<T>      none_too;
      bidiagonalize(b, n, d, e, none, none_too, false);
      bidiagonal_qr(d, e, none, none_too, n, svd_qr_steps_per_value * n);
      return d;
    }

    /*!
     * \brief The minimum-norm least-squares solution of `A X ≈ B` from the singular value decomposition \p s of `A`
     *        (`m × n`), its first \p rank triplets: `X = V Σ⁺ Uᴴ B`.
     */
    template <typename T>
    [[nodiscard]] matrix<T> lstsq_from_svd(const svd_decomposition<T>& s,
                                           std::size_t                 rank,
                                           const matrix<T>&            b)
    {
      const vector<double>& sigma                {s.singular_values()};
      const std::size_t     rows                 {s.u().rows()};
      const std::size_t     cols                 {s.v().rows()};
      buffer<T>             x(cols * b.cols(), T {});
      for (std::size_t c {0}; c < b.cols(); ++c) {
        for (std::size_t d {0}; d < rank; ++d) {
          T coefficient {}; // (u_dᴴ b_c) / σ_d
          for (std::size_t i {0}; i < rows; ++i) {
            coefficient += conj_of(s.u()(i, d)) * b(i, c);
          }
          coefficient /= from_double<T>(sigma[d]);
          for (std::size_t j {0}; j < cols; ++j) {
            x[(j * b.cols()) + c] += coefficient * s.v()(j, d);
          }
        }
      }
      return matrix<T> {cols, b.cols(), std::move(x)};
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
    const matrix<T> column {lstsq(a, matrix<T> {b.size(), 1, buffer<T>(b.begin(), b.end())})};
    return vector<T> {buffer<T>(column.begin(), column.end())};
  }

  /*!
   * \brief The minimum-norm least-squares solution of `A X ≈ B` for every column of `B` at once, as
   *        `numpy.linalg.lstsq` with a matrix right-hand side: one SVD of `A`, `X = V Σ⁺ Uᴴ B`; or, where a bound
   *        on the condition number shows the rank full with room to spare, the same solution to rounding by one
   *        QR factorisation (\ref detail::lstsq_by_qr).
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
    if (std::optional<matrix<T>> fast {detail::lstsq_by_qr(a, b)}) {
      return *std::move(fast);
    }
    const svd_decomposition<T> s {a};
    return detail::lstsq_from_svd(s, detail::numerical_rank(s.singular_values(), a.rows(), a.cols()), b);
  }

  /*!
   * \brief The singular values of \p a, descending, without its singular vectors, as
   *        `numpy.linalg.svd(a, compute_uv=False)`: the Householder `R` of a tall `A` (of `Aᴴ` when wide), its
   *        Golub–Kahan bidiagonal and the implicit-shift QR iteration on it, no basis formed nor rotated. Each value
   *        to an absolute accuracy of the largest times the machine epsilon, as \ref svd_decomposition's.
   * \throws non_finite_error If an entry of \p a is infinite or NaN.
   * \throws convergence_error If the iteration does not converge within its bound (not observed).
   */
  template <typename T>
  [[nodiscard]] vector<double> singular_values(const matrix<T>& a)
  {
    detail::require_finite(a, "singular values");
    const bool        tall {a.rows() >= a.cols()};
    const std::size_t rows {tall ? a.rows() : a.cols()};
    const std::size_t cols {tall ? a.cols() : a.rows()};
    // Divided by its largest magnitude, as the decomposition is: the sums of squares overflow near 1e155.
    double scale           {0.0};
    for (std::size_t i {0}; i < a.rows(); ++i) {
      for (std::size_t j {0}; j < a.cols(); ++j) {
        scale = std::max(scale, static_cast<double>(std::abs(a(i, j))));
      }
    }
    scale = scale == 0.0 ? 1.0 : scale;
    buffer<T> entries(rows * cols, T {});
    for (std::size_t i {0}; i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        entries[(i * cols) + j] = (tall ? a(i, j) : detail::conj_of(a(j, i))) / detail::from_double<T>(scale);
      }
    }
    buffer<T>      r {rows == cols ? std::move(entries) : detail::householder_reflect(std::move(entries), rows, cols).r};
    buffer<double> d {detail::square_singular_values(std::move(r), cols)};
    for (double& value : d) {
      value *= scale;
    }
    return vector<double> {std::move(d)};
  }

  /*!
   * \brief The eigenvalues of the real symmetric or complex Hermitian \p a, ascending, without its eigenvectors, as
   *        `numpy.linalg.eigvalsh`: its Householder tridiagonal (no basis formed) and the implicit QL iteration on it,
   *        no rotation kept. Only the lower triangle is read, and of the diagonal its real part. Each value to an
   *        absolute accuracy of the largest magnitude times the machine epsilon, as \ref symmetric_eigen's.
   * \throws dimension_error   If \p a is not square.
   * \throws non_finite_error  If an entry read is infinite or NaN.
   * \throws convergence_error If the iteration does not converge within its bound (not observed).
   */
  template <typename T>
  [[nodiscard]] vector<double> symmetric_eigenvalues(const matrix<T>& a)
  {
    const std::size_t n {a.rows()};
    if (a.cols() != n) {
      throw dimension_error("symmetric eigenvalues require a square matrix");
    }
    detail::require_finite_lower(a, "symmetric eigenvalues");
    // Divided by its largest magnitude read, as the decompositions are: a sum of squares overflows near 1e155.
    double scale {0.0};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < i; ++j) {
        scale = std::max(scale, static_cast<double>(std::abs(a(i, j))));
      }
      scale = std::max(scale, std::abs(detail::real_part(a(i, i))));
    }
    scale = scale == 0.0 ? 1.0 : scale;
    buffer<T> entries(n * n, T {});
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < i; ++j) {
        entries[(i * n) + j] = a(i, j) / detail::from_double<T>(scale);
      }
      entries[(i * n) + i] = detail::from_double<T>(detail::real_part(a(i, i)) / scale);
    }
    buffer<double> d(n, double {});
    if (n > 0) {
      buffer<double> off(n, double {});
      buffer<double> none;
      static_cast<void>(detail::tridiagonalize(entries, n, d, off));
      detail::tridiagonal_ql(d, off, none, n, detail::ql_steps_per_value * n);
    }
    std::sort(d.begin(), d.end());
    for (double& value : d) {
      value *= scale;
    }
    return vector<double> {std::move(d)};
  }

  //! \brief A least-squares solution and what `numpy.linalg.lstsq` reports beside it (\ref least_squares).
  template <typename T>
  struct least_squares_solution
  {
    matrix<T>      solution;        //!< The minimum-norm `x` of `A x ≈ B` (`n × k`), \ref lstsq's.
    vector<double> residuals;       //!< Per column of `B`, `‖B − A x‖²`: for a tall `A` of full column rank, else empty.
    std::size_t    rank {0};        //!< The numerical rank of `A` (\ref matrix_rank).
    vector<double> singular_values; //!< The singular values of `A`, descending (\ref singular_values).
  };

  /*!
   * \brief \ref lstsq, with the residuals, the rank and the singular values `numpy.linalg.lstsq` returns beside the
   *        solution: the residual sums of squares are given only where the solution is unique and need not be exact,
   *        `A` tall (`m > n`) and of full column rank, as NumPy gives them.
   * \throws dimension_error If \p b's rows differ from \p a's.
   * \throws non_finite_error If an entry of \p a is infinite or NaN.
   */
  template <typename T>
  [[nodiscard]] least_squares_solution<T> least_squares(const matrix<T>& a,
                                                        const matrix<T>& b)
  {
    if (b.rows() != a.rows()) {
      throw dimension_error("lstsq: right-hand side size differs from the matrix");
    }
    detail::require_finite(a, "lstsq");
    // The singular values from what the solution took: the QR's R (A's over a power of two), else the decomposition.
    std::pair<buffer<T>, int>           r;
    std::optional<matrix<T>>            fast {detail::lstsq_by_qr(a, b, &r)};
    std::optional<svd_decomposition<T>> s;
    if (!fast) {
      s.emplace(a);
    }
    buffer<double> values {fast ? detail::square_singular_values(std::move(r.first), std::min(a.rows(), a.cols()))
                                     : buffer<double>(s->singular_values().begin(), s->singular_values().end())};
    if (fast) {
      for (double& value : values) {
        value = std::ldexp(value, r.second);
      }
    }
    const vector<double> sigma {std::move(values)};
    const std::size_t    rank  {detail::numerical_rank(sigma, a.rows(), a.cols())};
    matrix<T>            x     {fast ? *std::move(fast) : detail::lstsq_from_svd(*s, rank, b)};
    buffer<double>       residuals;
    if (rank == a.cols() && a.rows() > a.cols()) {
      const matrix<T> fitted {a * x};
      residuals.assign(b.cols(), 0.0);
      for (std::size_t i {0}; i < b.rows(); ++i) {
        for (std::size_t c {0}; c < b.cols(); ++c) {
          residuals[c] += std::norm(b(i, c) - fitted(i, c)); // |·|², real or complex
        }
      }
    }
    return least_squares_solution<T> {.solution        = std::move(x),
                                      .residuals       = vector<double> {std::move(residuals)},
                                      .rank            = rank,
                                      .singular_values = sigma};
  }

  /*!
   * \brief The numerical rank of \p a, as `numpy.linalg.matrix_rank` counts it: its singular values above
   *        \ref detail::rank_tolerance of the largest.
   * \throws non_finite_error If an entry of \p a is infinite or NaN.
   */
  template <typename T>
  [[nodiscard]] std::size_t matrix_rank(const matrix<T>& a)
  {
    return detail::numerical_rank(singular_values(a), a.rows(), a.cols());
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
    const svd_decomposition<T> s    {a};
    const std::size_t          rows {a.rows()};
    const std::size_t          cols {a.cols()};
    const std::size_t          rank {detail::numerical_rank(s.singular_values(), rows, cols)};
    // (V Σ⁺) · Uᴴ over the first `rank` singular triplets, through the matrix product kernel: summed term by term
    // it ran five times as long as NumPy's at order 512.
    buffer<T> scaled(cols * rank, T       {});
    buffer<T> left_adjoint(rank * rows, T {});
    for (std::size_t d {0}; d < rank; ++d) {
      const auto inverse {detail::from_double<T>(1.0 / s.singular_values()[d])};
      for (std::size_t i {0}; i < cols; ++i) {
        scaled[(i * rank) + d] = s.v()(i, d) * inverse;
      }
      for (std::size_t j {0}; j < rows; ++j) {
        left_adjoint[(d * rows) + j] = detail::conj_of(s.u()(j, d));
      }
    }
    buffer<T> out(cols * rows, T {});
    detail::gemm(cols, rows, rank, scaled.data(), rank, left_adjoint.data(), rows, out.data(), rows);
    return matrix<T> {cols, rows, std::move(out)};
  }
} // namespace numerical

#endif // NUMERICAL_DECOMPOSITION_HPP
