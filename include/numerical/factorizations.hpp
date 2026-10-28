/*!
 * \file factorizations.hpp
 * \brief The LU, Cholesky and QR factorisations and the symmetric and Hermitian eigen-decompositions.
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
#ifndef NUMERICAL_FACTORIZATIONS_HPP
#define NUMERICAL_FACTORIZATIONS_HPP

#include <algorithm>
#include <array>
#include <atomic>
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
#include <numeric>
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
    /*!
     * \brief The divisor \ref over_pivot takes for the pivot \p pivot. A complex pivot's reciprocal, once, as LAPACK's
     *        zgetf2 takes it (a complex division calls the C library's robust one, a quarter of a complex LU of order
     *        48 when it ran for every element); a real pivot itself, each element divided.
     */
    template <typename T>
    [[nodiscard]] T pivot_divisor(const T& pivot)
    {
      return pivot;
    }

    //! \brief A complex pivot as \ref over_pivot takes it: its reciprocal, or, where that would overflow (a subnormal
    //!        pivot), the pivot itself, divided by.
    template <typename R>
    struct complex_divisor
    {
      std::complex<R> reciprocal; //!< `1 / pivot`.
      std::complex<R> pivot;      //!< The pivot.
      bool            divides;    //!< Whether \ref over_pivot divides by the pivot: its parts under the smallest normal.
    };

    //! \brief \ref pivot_divisor of a complex pivot: its reciprocal, as zgetf2 takes one at least the smallest normal
    //!        number in modulus (`sfmin`), and one under it divided by, whose reciprocal would overflow. (A zero pivot
    //!        never comes here: the factorisation leaves its column, and the solves refuse a singular factor.)
    template <typename R>
    [[nodiscard]] complex_divisor<R> pivot_divisor(const std::complex<R>& pivot)
    {
      const R largest {std::max(std::abs(pivot.real()), std::abs(pivot.imag()))};
      return {std::complex<R> {1} / pivot, pivot, largest < std::numeric_limits<R>::min()};
    }

    //! \brief \p x over the pivot whose \ref pivot_divisor is \p divisor: the same quotient wherever the LU takes it.
    template <typename T>
    [[nodiscard]] T over_pivot(const T& x,
                               const T& divisor)
    {
      return x / divisor;
    }

    //! \brief \ref over_pivot of a complex \p x: times the pivot's reciprocal, the product in line; over a subnormal
    //!        pivot itself.
    template <typename R>
    [[nodiscard]] std::complex<R> over_pivot(const std::complex<R>&    x,
                                             const complex_divisor<R>& divisor)
    {
      return divisor.divides ? x / divisor.pivot : operator_times(x, divisor.reciprocal);
    }

    //! \brief \p x over a Cholesky factor's diagonal entry \p diagonal, real and positive whatever the scalar.
    template <typename T>
    [[nodiscard]] T over_real_diagonal(const T& x,
                                       const T& diagonal)
    {
      return x / diagonal;
    }

    /*!
     * \brief \ref over_real_diagonal of a complex \p x: each part over the diagonal's real part, one rounding each, where a
     *        complex division took the C library's scaled one (several roundings, and other bits under another
     *        library).
     */
    template <typename R>
    [[nodiscard]] std::complex<R> over_real_diagonal(const std::complex<R>& x,
                                                     const std::complex<R>& diagonal)
    {
      return std::complex<R> {x.real() / diagonal.real(), x.imag() / diagonal.real()};
    }

    //! \brief `sum −= a · b` in one statement, which the compiler may fuse (as it fuses the matrix solve's).
    template <typename T>
    void subtract_term(T&       sum,
                       const T& a,
                       const T& b)
    {
      sum -= a * b;
    }

    //! \brief \ref subtract_term of complex numbers: the product in line (\ref operator_times, the operator's rounding).
    template <typename R>
    void subtract_term(std::complex<R>&       sum,
                       const std::complex<R>& a,
                       const std::complex<R>& b)
    {
      sum -= operator_times(a, b);
    }

    //! \brief The columns of one panel of the blocked LU: the trailing update, a product of this depth, is
    //!        deep enough for the packed kernel while the unblocked panel stays a small part of the work.
    inline constexpr std::size_t lu_block {64};

#ifdef NUMERICAL_X86_COMPILED
    //! \brief The order up to which the LU takes panels of \ref lu_small_block columns: there the panels, a column
    //!        at a time on one thread, are most of the time, and narrower ones halve their work. On six AVX2 cores the
    //!        panel is the critical path further up (orders 300 and 384: 8 and 10 % faster than at 64 columns; 500, 5 %
    //!        slower).
    inline constexpr std::size_t lu_small_order {384};
#else
    //! \brief The order up to which the LU takes panels of \ref lu_small_block columns: there the panels, a column
    //!        at a time on one thread, are most of the time, and narrower ones halve their work (arm64, order 300 on
    //!        every core: 14 % slower at 32 columns than at 64).
    inline constexpr std::size_t lu_small_order {256};
#endif

    //! \brief The columns of one panel of the LU of an order up to \ref lu_small_order.
    inline constexpr std::size_t lu_small_block {32};

    //! \brief The order from which the LU takes panels of \ref lu_wide_block columns: each trailing update rereads the
    //!        whole trailing block, and from there the memory traffic of panels of 64, not their arithmetic, sets the
    //!        time; 128 halves it. Order 4096: ten AVX-512 cores 322 → 273 ms, six AVX2 cores 416 → 330, eight arm64
    //!        cores 285 → 252; order 3072 10 to 17 % faster; 2560 the same; 2048 10 % slower, the wider panel's own
    //!        work then the longer. The factors are the same to the bit at either width.
    inline constexpr std::size_t lu_wide_order {3072};

    //! \brief The columns of one panel of the LU from \ref lu_wide_order.
    inline constexpr std::size_t lu_wide_block {128};

    //! \brief The panel width of the LU of order \p n: a power of two, so a row's panel is a mask away, not a division.
    [[nodiscard]] constexpr std::size_t lu_block_for(std::size_t n) noexcept
    {
      static_assert(std::has_single_bit(lu_small_block) && std::has_single_bit(lu_block), "a panel is a mask away");
      static_assert(std::has_single_bit(lu_wide_block), "a panel is a mask away");
      if (n >= lu_wide_order) {
        return lu_wide_block;
      }
      return n <= lu_small_order ? lu_small_block : lu_block;
    }

    //! \brief The trailing columns at most past which the LU factorises its next panel beside the update of the rest
    //!        (a look-ahead): a wider update runs well as one product, and cut in blocks beside the panel it ran
    //!        longer (order 2048 on six x86-64 cores, 9 %); a narrower one leaves threads idle while the panel runs.
    inline constexpr std::size_t lu_lookahead_columns {1024};

    //! \brief The trailing columns at most past which each block of the update beside the panel solves its own rows of
    //!        U₁₂: a wider update keeps every thread busy past the panel, and its rows solved on all of them first ran
    //!        faster than on the workers alone (order 1000 on four AVX2 cores, 2 %).
    inline constexpr std::size_t lu_fused_columns {512};

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

#ifdef NUMERICAL_X86_COMPILED
    //! \brief The fewest columns an LU solve gives a thread of its own: on six AVX2 cores, 300 right-hand sides of
    //!        order 300 in six slices rather than four (of 64) took 0.54 ms instead of 0.62, an inverse 0.50 instead of
    //!        0.59; 32 did no better.
    inline constexpr std::size_t slice_columns {48};
#else
    //! \brief The fewest columns an LU solve gives a thread of its own: a narrower slice starves the product kernel
    //!        (arm64, order 256 and 64 right-hand sides in eight slices: 112 → 194 µs; 48 columns, order 300 and 300
    //!        right-hand sides: 8 % slower than 64).
    inline constexpr std::size_t slice_columns {64};
#endif

#ifdef NUMERICAL_X86_COMPILED
    //! \brief The order up to which an LU solve gives each thread its own columns on the x86-64 units: each thread
    //!        rereads the whole factor, which the memory stops paying for past it (six AVX2 cores, an inverse of order
    //!        768 6 % slower, 1024 17 %). On arm64 the slices win at every order measured (to 1024).
    inline constexpr std::size_t slice_solve_order {384};
#endif


    //! \brief The columns of one leaf of the recursive LU panel.
    inline constexpr std::size_t solve_leaf {16};

    //! \brief The rows an LU inverse writes its identity into, and permutes the columns of, on one thread at a time.
    inline constexpr std::size_t inverse_band {16};

    //! \brief The weight of an element a pass bound by the memory writes or moves, against a multiply-add, as the BLAS
    //!        routines' transposing copies (weighed as one, an LU inverse of order 300 permuted its columns on one thread).
    inline constexpr std::size_t memory_element_weight {4};

    //! \brief The rows of an SVD's input one thread scales and copies at a time.
    inline constexpr std::size_t svd_band {32};

    /*!
     * \brief The power of two a decomposition divides a matrix by whose largest magnitude is \p largest: its leading
     *        bit's, the largest entry then in [1, 2); 1 where \p largest is 0. Dividing and multiplying back by it are
     *        exact, where a division by the largest itself rounded every entry.
     */
    [[nodiscard]] inline double power_of_two_scale(double largest) noexcept
    {
      return largest == 0.0 ? 1.0 : std::ldexp(1.0, std::ilogb(largest));
    }

    //! \brief The rows of a Cholesky factor whose upper part one thread zeroes at a time.
    inline constexpr std::size_t zeroed_rows {32};

#ifdef NUMERICAL_X86_COMPILED
    //! \brief The rows an LU panel's copy takes together, so that it writes each column of the panel eight `double`s,
    //!        a 64-byte line, at a time, where row by row each write fell on another line (six AVX2 cores, order 768:
    //!        5.08 → 4.74 ms; tiles of 16 or 32 rows were slower again, order 512 by 7 %).
    inline constexpr std::size_t panel_copy_rows {8};

    //! \brief The fewest rows of a panel copied by tiles of \ref panel_copy_rows rows: a shorter panel stays in the L1
    //!        cache whichever way it is written, and the tiles' loops cost it (one x86-64 core, the LU of order 17
    //!        2.29 → 2.55 µs, of order 64 27.0 → 29.8 µs). Off the x86-64 units every panel goes row by row: tiles of
    //!        rows were no faster on arm64, and Apple clang compiled them worse.
    inline constexpr std::size_t panel_tile_rows {256};
#endif

    //! \brief The columns of one panel of the blocked Cholesky factorisation (as \ref lu_block).
    inline constexpr std::size_t cholesky_block {64};

    //! \brief The rows of a Cholesky diagonal block factorised together, one chain of sums each (one x86-64 AVX2 core,
    //!        order 300: the diagonal blocks 128 → 65 µs; eight rows did no better).
    inline constexpr std::size_t cholesky_diagonal_rows {4};

    //! \brief The narrowest Cholesky diagonal block whose rows go together: in a block of 8, the groups' loops cost
    //!        more than the chains save (order 8, one arm64 core: 0.21 → 0.23 µs).
    inline constexpr std::size_t cholesky_together_width {16};

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

    //! \brief The rows of one tile of a Cholesky update: the tiles of a panel's update, each one product on one thread,
    //!        share the threads in one parallel loop, where a parallel product for each block of columns started and
    //!        joined the threads some thirty times a panel.
    inline constexpr std::size_t cholesky_update_rows {256};


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

    //! \brief The magnitude partial pivoting compares: `|x|` of a real \p x.
    template <typename T>
    [[nodiscard]] magnitude_t<T> pivot_magnitude(const T& x)
    {
      return std::abs(x);
    }

    //! \brief The magnitude partial pivoting compares of a complex \p x: `|Re x| + |Im x|`, LAPACK's (`izamax`, `cabs1`),
    //!        so the same pivots as its `zgetrf`, with no square root.
    template <typename U>
    [[nodiscard]] U pivot_magnitude(const std::complex<U>& x)
    {
      return std::abs(x.real()) + std::abs(x.imag());
    }

    //! \brief The real type a `T`'s norms, reflector lengths and pivot magnitudes are carried in: `long double` for a
    //!        `long double` scalar, real or complex, whose precision a `double` would cut to its own; `double` otherwise.
    template <typename T>
    using carried_t = std::conditional_t<std::is_same_v<magnitude_t<T>, long double>, long double, double>;

    //! \brief \p x, a magnitude carried in \ref carried_t, as the scalar `T`: converted to `T`'s real type first, so a
    //!        `std::complex<float>` is built from a `float` (a `double` argument narrows inside its constructor).
    template <typename T>
    [[nodiscard]] T from_carried(carried_t<T> x)
    {
      return static_cast<T>(static_cast<magnitude_t<T>>(x));
    }

    //! \brief The real part of a real (or general) scalar, in \ref carried_t.
    template <typename T>
    [[nodiscard]] carried_t<T> real_part(const T& x)
    {
      return static_cast<carried_t<T>>(x);
    }

    //! \brief The real part of a complex scalar.
    template <typename U>
    [[nodiscard]] carried_t<U> real_part(const std::complex<U>& x)
    {
      return static_cast<carried_t<U>>(x.real());
    }

    //! \brief The real type whose range a `T`'s values take: `float` for a single-precision scalar, real or complex;
    //!        \ref carried_t otherwise.
    template <typename T>
    using value_range_t =
      std::conditional_t<std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>, float, carried_t<T>>;

    //! \brief The magnitude under which \ref rescue_factor scales a `T` up: about the square root of the smallest normal
    //!        number of its range, or below it (2⁻⁵⁰⁰ for a `long double`, whose range is the platform's).
    template <typename T>
    inline constexpr carried_t<T> rescue_below {std::is_same_v<value_range_t<T>, float> ? 0x1p-60 : 0x1p-500};

    //! \brief The exact power of two that brings a magnitude below about the square root of the smallest normal number
    //!        of \p T's range (2⁻⁵⁰⁰ in `double`, 2⁻⁶⁰ in `float`) up among normal numbers, as LAPACK's dlarfg
    //!        rescales; 1 for any larger magnitude. It is a `T` itself: a zero takes the factor of the smallest normal
    //!        number, which a `double`'s range holds and a `float`'s does not.
    template <typename T>
    [[nodiscard]] carried_t<T> rescue_factor(carried_t<T> largest)
    {
      using real = carried_t<T>;
      constexpr real least {static_cast<real>(std::numeric_limits<value_range_t<T>>::min())};
      return largest < rescue_below<T> ? std::ldexp(real {1}, -std::ilogb(std::max(largest, least))) : real {1};
    }

    /*!
     * \brief The unit-modulus factor of \p x: `x / |x|` (its sign, for a real scalar), and 1 at 0. A subnormal `x` is
     *        scaled up first, exactly: its modulus would round to the coarse subnormal grid, and `x / |x|` miss unit
     *        modulus by as much.
     */
    template <typename T>
    [[nodiscard]] T phase_of(const T& x)
    {
      const T            scaled    {x * detail::from_carried<T>(rescue_factor<T>(static_cast<carried_t<T>>(std::abs(x))))};
      const carried_t<T> magnitude {static_cast<carried_t<T>>(std::abs(scaled))};
      return magnitude == 0 ? T {1} : scaled / detail::from_carried<T>(magnitude);
    }

    /*!
     * \brief Scales \p x by \ref rescue_factor of its largest magnitude, exactly, so that a Householder vector built
     *        from it is computed among normal numbers: a subnormal norm rounds to the coarse subnormal grid, and the
     *        reflector is then not orthogonal. \return The factor (1 unless the entries are all below 2⁻⁵⁰⁰, 2⁻⁶⁰ in
     *        `float`).
     */
    template <typename T>
    carried_t<T> rescale_tiny(buffer<T>& x)
    {
      carried_t<T> largest {0};
      for (const T& element : x) {
        largest = std::max(largest, static_cast<carried_t<T>>(std::abs(element)));
      }
      const carried_t<T> factor {rescue_factor<T>(largest)};
      for (T& element : x) {
        element *= detail::from_carried<T>(factor);
      }
      return factor;
    }

    //! \brief The Euclidean norm of the \p count elements at \p x, summed at the scale of their largest magnitude:
    //!        a plain sum of squares overflows near 1e155 and underflows to 0 near 1e-155.
    template <typename T>
    [[nodiscard]] carried_t<T> norm_at_scale(const T*    x,
                                             std::size_t count)
    {
      using real = carried_t<T>;
      real largest {0};
      for (std::size_t i {0}; i < count; ++i) {
        largest = std::max(largest, static_cast<real>(std::abs(x[i])));
      }
      if (largest == 0) { // the entries are finite: require_finite runs where every caller starts
        return largest;
      }
      real sum {0};
      for (std::size_t i {0}; i < count; ++i) {
        const real ratio {static_cast<real>(std::abs(x[i])) / largest};
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
    [[nodiscard]] carried_t<T> scaled_norm(const T*    x,
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
    [[nodiscard]] carried_t<T> scaled_norm(const buffer<T>& x)
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

    /*!
     * \brief A copy of \p a whose pages the threads fault in (\ref parallel_copy), a decomposition's own storage: the
     *        matrix's copy faulted every one in on the calling thread (order 1000 on one arm64 core 0.64 ms, an eighth
     *        of the Cholesky factorisation; on eight cores 0.28 ms).
     */
    template <typename T>
    [[nodiscard]] matrix<T> parallel_copy_of(const matrix<T>& a)
    {
      buffer<T> elements(a.rows() * a.cols());
      parallel_copy(std::to_address(a.begin()), elements.size(), elements.data());
      return matrix<T> {a.rows(), a.cols(), std::move(elements)};
    }

    //! \brief The elements \ref all_finite_shared gives a thread at a time.
    inline constexpr std::size_t finite_chunk {std::size_t {1} << 15U};

    /*!
     * \brief \ref all_finite on the threads by chunks of \ref finite_chunk, past two of them: a pass bound by the memory,
     *        an element read weighed as half one copied (six x86-64 AVX2 cores, a Cholesky factorisation of order 500
     *        with its other passes on the threads too: 1.12 → 0.91 ms).
     */
    [[nodiscard]] inline bool all_finite_shared(const double* x,
                                                std::size_t   count)
    {
      if (count < 2 * finite_chunk) {
        return all_finite(x, count);
      }
      std::atomic<bool> finite {true};
      parallel_for((count + finite_chunk - 1) / finite_chunk, memory_element_weight * finite_chunk / 2, [x, count, &finite](std::size_t chunk) {
                     const std::size_t first {chunk * finite_chunk};
                     if (!all_finite(x + first, std::min(finite_chunk, count - first))) {
                       finite.store(false, std::memory_order_relaxed);
                     }
                   });
      return finite.load(std::memory_order_relaxed);
    }

    //! \brief \ref require_finite for `double`, its elements contiguous: \ref all_finite.
    inline void require_finite(const matrix<double>& a,
                               const char*           what)
    {
      if (!all_finite_shared(std::to_address(a.begin()), a.rows() * a.cols())) {
        throw non_finite_error(std::string {what} + ": the matrix holds an infinite or NaN entry");
      }
    }

    //! \brief The same for `std::complex<double>`, whose array holds its parts in order ([complex.numbers]).
    inline void require_finite(const matrix<std::complex<double>>& a,
                               const char*                         what)
    {
      const auto* const parts {reinterpret_cast<const double*>(std::to_address(a.begin()))}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
      if (!all_finite_shared(parts, 2 * a.rows() * a.cols())) {
        throw non_finite_error(std::string {what} + ": the matrix holds an infinite or NaN entry");
      }
    }

    //! \brief Whether the lower triangle of the square \p a, and its diagonal's real part, are finite: entry by entry,
    //!        kept out of line: the rare road of \ref require_finite_lower where the whole matrix is not finite.
    template <typename T>
    NUMERICAL_OUT_OF_LINE bool lower_is_finite(const matrix<T>& a)
    {
      for (std::size_t i {0}; i < a.rows(); ++i) {
        bool finite {std::isfinite(real_part(a(i, i)))};
        for (std::size_t j {0}; j < i; ++j) {
          finite = finite && is_finite_entry(a(i, j));
        }
        if (!finite) {
          return false;
        }
      }
      return true;
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
      if (!lower_is_finite(a)) {
        throw non_finite_error(std::string {what} + ": the matrix holds an infinite or NaN entry");
      }
    }

    //! \brief \ref require_finite_lower for `double`: \ref all_finite over the whole matrix, the one vectorised pass
    //!        \ref require_finite takes, and the triangle entry by entry only where it finds a NaN or an infinity. Out of
    //!        line, so that a decomposition's constructor stays small enough to be inlined: in it, the check kept GCC
    //!        from inlining the Cholesky's (order 8, 0.204 → 0.220 µs), as an entry-by-entry check (6 % at order 64 on
    //!        arm64) or a row-by-row one (21 % at order 8 under GCC) cost in time.
    NUMERICAL_OUT_OF_LINE inline void require_finite_lower(const matrix<double>& a,
                                                           const char*           what)
    {
      if (!all_finite_shared(std::to_address(a.begin()), a.rows() * a.cols()) && !lower_is_finite(a)) {
        throw non_finite_error(std::string {what} + ": the matrix holds an infinite or NaN entry");
      }
    }

    //! \brief The same for `std::complex<double>`, whose array holds its parts in order ([complex.numbers]).
    NUMERICAL_OUT_OF_LINE inline void require_finite_lower(const matrix<std::complex<double>>& a,
                                                           const char*                         what)
    {
      const auto* const parts {reinterpret_cast<const double*>(std::to_address(a.begin()))}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
      if (!all_finite_shared(parts, 2 * a.rows() * a.cols()) && !lower_is_finite(a)) {
        throw non_finite_error(std::string {what} + ": the matrix holds an infinite or NaN entry");
      }
    }

    /*!
     * \brief `column −= Σₜ x_t · conj(l[t])` over the \p terms columns \p columns (\p height elements each, one after
     *        another), term by term in ascending order: an elimination step of a Cholesky panel's rows below (an overload
     *        per scalar, as `conj_of`). Each term \ref subtract_product, the column read and written again for every one.
     */
    template <typename T>
    void subtract_panel_columns(T*          column,
                                const T*    columns,
                                std::size_t height,
                                const T*    l,
                                std::size_t terms)
    {
      for (std::size_t t {0}; t < terms; ++t) {
        subtract_product(column, columns + (t * height), conj_of(l[t]), height);
      }
    }

    //! \brief \ref subtract_panel_columns for `double`: \ref subtract_products, which holds the column in registers
    //!        through the terms, the same multiply-adds in the same order, so the same values (a complex one keeps the
    //!        term-by-term calls, whose `x · f` the other would round as `f · x`).
    inline void subtract_panel_columns(double*       column,
                                       const double* columns,
                                       std::size_t   height,
                                       const double* l,
                                       std::size_t   terms)
    {
      subtract_products(column, l, columns, height, terms, height);
    }
  } // namespace detail

  namespace detail {

    //! \brief Whether a solve of `T` goes by rows of real terms (\ref lu_decomposition's `solve_narrow`): `double` and
    //!        `float`.
    template <typename T>
    inline constexpr bool real_rows {std::is_same_v<T, double> || std::is_same_v<T, float>};
  } // namespace detail

  /*!
   * \brief An LU factorisation with partial pivoting of a square matrix: each column's pivot the first entry of the
   *        largest \ref detail::pivot_magnitude below the diagonal, `|Re| + |Im|` of a complex one, as LAPACK's.
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
      : lu_decomposition(detail::parallel_copy_of(a))
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
          detail::subtract_term(sum, lu_[(i * n_) + j], y[j]);
        }
        y[i] = sum;
      }
      for (std::size_t i {n_}; i-- > 0;) { // back substitution (upper U)
        T sum {y[i]};
        for (std::size_t j {i + 1}; j < n_; ++j) {
          detail::subtract_term(sum, lu_[(i * n_) + j], y[j]);
        }
        y[i] = detail::over_pivot(sum, detail::pivot_divisor(lu_[(i * n_) + i]));
      }
      return vector<T> {std::move(y)};
    }

    //! \brief The determinant (the pivot sign times the product of U's diagonal; 0 if singular).
    [[nodiscard]] T determinant() const
    {
      T value {detail::from_carried<T>(sign_)};
      for (std::size_t i {0}; i < n_; ++i) {
        value *= lu_[(i * n_) + i];
      }
      if (singular_ && !detail::is_finite_entry(value)) { // a zero pivot times others overflowing: 0·∞, not a NaN
        value = detail::from_carried<T>(sign_);
        for (std::size_t i {0}; i < n_; ++i) {
          const T pivot {lu_[(i * n_) + i]};
          value *= pivot == T {0} ? pivot : detail::phase_of(pivot); // the product's zero, its sign the pivots'
        }
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
      return inverse(buffer<T> {});
    }

    /*!
     * \brief \ref inverse in \p storage, its elements discarded: no fresh `n × n` mapped and faulted in where it holds
     *        that much already. \ref pivoted_inverse, then `P` applied in place, by bands of rows on the threads.
     * \param[in] storage The storage of the result, taken over.
     * \return The inverse matrix, in \p storage.
     * \throws singular_matrix_error If the matrix is singular.
     */
    [[nodiscard]] matrix<T> inverse(buffer<T> storage) const
    {
      buffer<T>                rows {pivoted_inverse(std::move(storage)).elements()};
      const std::size_t        n    {n_};
      T* const                 at   {rows.data()};
      const std::size_t* const to   {pivots_.data()};
      // Bounds captured by value: by reference, the stores might have changed them, and each would be read again.
      detail::parallel_for((n + detail::inverse_band - 1) / detail::inverse_band, detail::memory_element_weight * detail::inverse_band * n, [at, n, to](std::size_t band) {
                             buffer<T> held(n); // column to[i] of A⁻¹ is column i of U⁻¹ L⁻¹
                             for (std::size_t r {band * detail::inverse_band}; r < std::min(n, (band + 1) * detail::inverse_band); ++r) {
                               std::copy_n(at + (r * n), n, held.data());
                               for (std::size_t i {0}; i < n; ++i) {
                                 at[(r * n) + to[i]] = held[i];
                               }
                             }
                           });
      return matrix<T> {n, n, std::move(rows)};
    }

    /*!
     * \brief `(P A)⁻¹ = U⁻¹ L⁻¹`, the inverse of the rows in their pivoted order, in \p storage (its elements discarded):
     *        column `pivots()[i]` of `A⁻¹` is its column `i`, which \ref inverse moves there (a caller that copies the
     *        result anyway can move them as it copies). The identity written by bands of rows on the threads.
     * \param[in] storage The storage of the result, taken over.
     * \return `(P A)⁻¹`, in \p storage.
     * \throws singular_matrix_error If the matrix is singular.
     */
    [[nodiscard]] matrix<T> pivoted_inverse(buffer<T> storage) const
    {
      if (singular_) {
        throw singular_matrix_error("inverse: matrix is singular");
      }
      const std::size_t n    {n_};
      storage.resize(n * n);
      T* const          rows {storage.data()};
      detail::parallel_for((n + detail::inverse_band - 1) / detail::inverse_band, detail::memory_element_weight * detail::inverse_band * n, [rows, n](std::size_t band) {
                             for (std::size_t r {band * detail::inverse_band}; r < std::min(n, (band + 1) * detail::inverse_band); ++r) {
                               std::fill_n(rows + (r * n), n, T {});
                               rows[(r * n) + r] = T {1};
                             }
                           });
      solve_rows(storage, n, true);
      return matrix<T> {n, n, std::move(storage)};
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
      buffer<T>         rows(n_ * cols); // every element written below
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t c {0}; c < cols; ++c) {
          rows[(i * cols) + c] = b(pivots_[i], c);
        }
      }
      solve_rows(rows, cols);
      return matrix<T> {n_, cols, std::move(rows)};
    }

    /*!
     * \brief \ref solve on an expiring \p b, in its own storage: its rows permuted in place (cycle by cycle of the
     *        pivots, one row held aside), then solved there. No copy of `B`.
     * \param[in] b The right-hand sides (`n × k`), taken over.
     * \return `X` (`n × k`), in `b`'s storage.
     * \throws dimension_error If `b`'s row count differs from the matrix.
     * \throws singular_matrix_error If the matrix is singular.
     */
    [[nodiscard]] matrix<T> solve(matrix<T>&& b) const
    {
      if (b.rows() != n_) {
        throw dimension_error("solve: right-hand side row count differs from the matrix");
      }
      if (singular_) {
        throw singular_matrix_error("solve: matrix is singular");
      }
      const std::size_t cols {b.cols()};
      buffer<T>         rows {std::move(b).elements()};
      // Row i of P B is row pivots_[i] of B: each cycle i → pivots_[i] → … walked once, its first row held aside.
      std::vector<bool> placed(n_, false);
      buffer<T>         held(cols);
      for (std::size_t start {0}; start < n_; ++start) {
        if (placed[start] || pivots_[start] == start) {
          continue;
        }
        std::copy_n(rows.data() + (start * cols), cols, held.data());
        std::size_t i {start};
        for (std::size_t from {pivots_[i]}; from != start; i = from, from = pivots_[i]) {
          std::copy_n(rows.data() + (from * cols), cols, rows.data() + (i * cols));
          placed[i] = true;
        }
        std::copy_n(held.data(), cols, rows.data() + (i * cols));
        placed[i] = true;
      }
      solve_rows(rows, cols);
      return matrix<T> {n_, cols, std::move(rows)};
    }

    //! \brief `L` and `U` in one matrix, as LAPACK's `dgetrf` leaves them: `U` on and above the diagonal, `L` below it
    //!         (its unit diagonal not stored), of the rows in their pivoted order.
    [[nodiscard]] matrix<T> factors() const&
    {
      return matrix<T> {n_, n_, lu_};
    }

    //! \brief \ref factors, moved out of an expiring decomposition: no copy.
    [[nodiscard]] matrix<T> factors() &&
    {
      return matrix<T> {n_, n_, std::move(lu_)};
    }

    //! \brief The row permutation: row `i` of `P A = L U` is row `pivots()[i]` of `A`.
    [[nodiscard]] const std::vector<std::size_t>& pivots() const noexcept
    {
      return pivots_;
    }

    //! \brief Whether a pivot was zero: `U` is singular, and \ref solve and \ref inverse throw.
    [[nodiscard]] bool singular() const noexcept
    {
      return singular_;
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
      if constexpr (detail::real_rows<T>) {
        // The rows beat the blocked passes on the x86-64 units at any order (the portable path's products there fuse
        // nothing, the rows would), and at any order elsewhere, by groups (arm64, four columns: order 256 0.65 times the
        // blocked passes' time under Clang, 0.45 under GCC; order 1024 on every thread 0.91 and 0.62); in float, where
        // the blocked passes copy each block of the factor negated for a product of four columns, order 3000 on one
        // arm64 core 5.9 → 2.1 ms.
#ifdef NUMERICAL_X86_COMPILED
        const bool narrow {!lower && !y.empty() && cols <= detail::narrow_solve && detail::gemm_path_current() != gemm_path::portable};
#else
        const bool narrow {!lower && !y.empty() && cols <= detail::narrow_solve};
#endif
        if (narrow) {
          solve_narrow(y, cols);
          return;
        }
      }
      const std::size_t slices {column_slices(cols)};
      if (slices > 1) { // independent right-hand sides: each thread its own columns, all the blocks, no join per block
        const std::size_t slice {(((cols + slices - 1) / slices) + 7) / 8 * 8};
        const auto        start {[this, slice, slices, cols, lower](std::size_t s) {
                                   return lower ? lower_slice_start(s, slices, cols) : std::min(cols, s * slice);
                                 }};
        detail::parallel_for(lower ? slices : (cols + slice - 1) / slice, n_ * n_ * slice / detail::gemm_kernel_speedup, [&](std::size_t s) {
                               solve_columns(y, cols, start(s), start(s + 1), lower);
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
     * \brief The first column of slice \p s of \p slices (\p slices itself: \p cols) where the forward pass is lower:
     *        column `c` weighs `n² + (n − c)²`, the backward pass's triangle and the forward pass's below row `c`, and
     *        the slices weigh alike (by columns alike the first slice did twice the last's work), each from a multiple
     *        of 8 columns. A slice is never empty: it spans at least half of `cols / slices` columns, past 8.
     */
    [[nodiscard]] std::size_t lower_slice_start(std::size_t s,
                                                std::size_t slices,
                                                std::size_t cols) const
    {
      const auto   n {static_cast<double>(n_)};
      const auto   weight {[n](double c) { return (n * n * c) + (((n * n * n) - ((n - c) * (n - c) * (n - c))) / 3.0); }};
      const double target {weight(static_cast<double>(cols)) * static_cast<double>(s) / static_cast<double>(slices)};
      std::size_t  low    {0};
      std::size_t  high   {cols};
      while (low < high) { // the fewest columns that weigh the target
        const std::size_t middle {low + ((high - low) / 2)};
        if (weight(static_cast<double>(middle)) < target) {
          low = middle + 1;
        }
        else {
          high = middle;
        }
      }
      return std::min(cols, (low + 7) / 8 * 8);
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
                                 const auto divisor {detail::pivot_divisor(lu_[(i * n_) + i])};
                                 for (std::size_t c {first}; c < last; ++c) {
                                   y[(i * cols) + c] = detail::over_pivot(y[(i * cols) + c], divisor);
                                 }
                               }
                             });
        subtract_block(y, cols, 0, k0, k0, k1, c0, c1);
        k1 = k0;
      }
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
      // Forward, by groups of rows: their terms before the group are independent of one another, and go together
      // (\ref detail::subtract_rows), the group's own triangle after; each element's terms in ascending order still.
      constexpr std::size_t group {detail::product_rows};
      static_assert(detail::lu_small_block % group == 0 && detail::lu_block % group == 0, "a panel is a whole number of groups");
      static_assert(detail::lu_wide_block % group == 0, "a panel is a whole number of groups");
      std::size_t           i0    {0};
      for (; i0 + group <= n_; i0 += group) {
        if (i0 > 0) {
          detail::subtract_rows(y.data() + (i0 * cols), cols, lu_.data() + (i0 * n_), n_, y.data(), cols, i0, cols);
        }
        for (std::size_t i {i0 + 1}; i < i0 + group; ++i) {
          detail::subtract_products(y.data() + (i * cols), lu_.data() + (i * n_) + i0, y.data() + (i0 * cols), cols, i - i0, cols);
        }
      }
      for (std::size_t i {std::max<std::size_t>(i0, 1)}; i < n_; ++i) {
        detail::subtract_products(y.data() + (i * cols), lu_.data() + (i * n_), y.data(), cols, i, cols);
      }
      // Backward, each row's terms block by block from the last down to its own, ascending within each: a group's
      // terms in the blocks past its own are independent of one another and go together (\ref detail::subtract_rows),
      // in that same order; its own block's after, row by row (\p finish). The rows past the last whole group lie in
      // the last block, a panel being a whole number of groups: they have their own block's terms alone.
      const std::size_t last {panel_of(n_ - 1)};
      const auto        finish {[this, &y, cols](std::size_t i) {
                                  const std::size_t to {std::min(panel_of(i) + block_, n_)};
                                  if (i + 1 < to) {
                                    detail::subtract_products(y.data() + (i * cols), lu_.data() + (i * n_) + i + 1, y.data() + ((i + 1) * cols), cols, to - (i + 1), cols);
                                  }
                                  const auto divisor {detail::pivot_divisor(lu_[(i * n_) + i])};
                                  for (std::size_t c {0}; c < cols; ++c) {
                                    y[(i * cols) + c] = detail::over_pivot(y[(i * cols) + c], divisor);
                                  }
                                }};
      std::size_t i1 {n_};
      for (; i1 % group != 0; --i1) {
        finish(i1 - 1);
      }
      for (; i1 > 0; i1 -= group) {
        const std::size_t top {i1 - group};
        const std::size_t own {panel_of(top)};
        for (std::size_t b0 {last}; b0 > own; b0 -= block_) {
          detail::subtract_rows(y.data() + (top * cols), cols, lu_.data() + (top * n_) + b0, n_, y.data() + (b0 * cols), cols, std::min(block_, n_ - b0), cols);
        }
        for (std::size_t i {i1}; i-- > top;) {
          finish(i);
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

    //! \brief \ref negated_block into \p out, resized to it.
    void negated_block_into(buffer<T>&  out,
                            std::size_t r0,
                            std::size_t r1,
                            std::size_t c0,
                            std::size_t c1) const
    {
      out.resize((r1 - r0) * (c1 - c0));
      for (std::size_t r {r0}; r < r1; ++r) {
        for (std::size_t c {c0}; c < c1; ++c) {
          out[((r - r0) * (c1 - c0)) + (c - c0)] = -lu_[(r * n_) + c];
        }
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
      // Every panel in the one column-major storage, of the first (and tallest) one's size: a fresh one each panel was
      // mapped anew and faulted in page by page, on the critical path (order 1000 on six AVX2 cores: 24 µs of a panel's
      // 470, and its copy in 97 µs instead of about 40).
      buffer<T>         panel(n_ * std::min(block_, n_));
      const std::size_t end {std::min(block_, n_)};
      factorize_panel(panel, 0, end);
      if (end < n_) {
        factorize_trailing(panel, end);
      }
      swap_left_columns();
    }

    /*!
     * \brief The panels after the first (whose columns end at \p first_end), each with its trailing update. Out of
     *        line: in \ref factorize, the update's storage and calls cost an LU of one panel 5 % (orders 8 to 48 under
     *        clang on arm64), where none of them runs.
     */
    NUMERICAL_OUT_OF_LINE void factorize_trailing(buffer<T>&  panel,
                                                  std::size_t first_end)
    {
      // −L₂₁ and its packing for the product's tile, in storage kept from panel to panel: packed once, it serves every
      // block of the trailing update, where each block's product packed it again (6.9 % of the time on six AVX2 cores
      // at order 1000).
      buffer<T>   minus_l;
      buffer<T>   packing;
      std::size_t end {first_end};
      for (std::size_t k0 {0}; end < n_;) {
        const std::size_t next             {std::min(end + block_, n_)};
        negated_block_into(minus_l, end, n_, k0, end);
        const detail::packed_left<T> left  {detail::pack_left(n_ - end, block_, end - k0, minus_l.data(), end - k0, packing)};
        const std::size_t            right {n_ - next};
        if (right > detail::lu_lookahead_columns) { // one product of the whole trailing block, then the panel
          solve_panel_rows(k0, end);
          update_columns(left, k0, end, next);
          update_columns(left, k0, next, n_);
          factorize_panel(panel, end, next);
          k0  = end;
          end = next;
          continue;
        }
        // About two blocks of columns for each thread the work pays for: a product too small to share ran one block for
        // each thread the machine has (order 256 on eight arm64 threads: 329 → 311 µs); two, not one, as on x86-64 a
        // block of half the columns stays in its cache, and the balance beat one (six AVX2 cores, order 1000: 12 %).
        const std::size_t work    {(n_ - end) * right * (end - k0) / detail::lu_update_speedup};
        const std::size_t helpers {std::clamp(work / detail::work_per_thread, std::size_t {1}, max_threads())};
        const std::size_t width   {std::max(block_, (right + (2 * helpers) - 1) / (2 * helpers))};
        const std::size_t blocks  {(right + width - 1) / width};
        swap_limit_ = next;
        // The next panel's columns first, their update shared among the threads; then, where the blocks right of it
        // run beside the panel, each block solves its rows of U₁₂ and takes its update on one thread, one parallel loop
        // a panel where the rows of U₁₂ were solved in one of their own (order 531 on ten AVX-512 cores: 2.24 → 2.09 ms):
        // the columns of a block depend on no other's, and the rows they read from the panel's L are above every swap
        // the next panel makes. On one thread, all of U₁₂ first (in that order, orders 200 and 300 on one AVX-512 core:
        // 4 to 5 %).
        const std::size_t per_block {((n_ - end) * width * (end - k0) / detail::lu_update_speedup) + ((end - k0) * (end - k0) * width / 2)};
        const bool        fused     {right <= detail::lu_fused_columns && detail::parallel_threads(blocks, per_block) >= 2};
        if (fused) {
          solve_columns(k0, end, end, next);
        }
        else {
          solve_panel_rows(k0, end);
        }
        update_columns(left, k0, end, next);
        detail::parallel_for_beside(blocks, per_block,
                                    [&](std::size_t block) {
                                      const std::size_t first {next + (block * width)};
                                      const std::size_t last {std::min(first + width, n_)};
                                      if (fused) {
                                        solve_columns(k0, end, first, last);
                                      }
                                      update_columns(left, k0, first, last);
                                    },
                                    [&] { factorize_panel(panel, end, next); });
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
    }

    /*!
     * \brief Unblocked LU with partial pivoting of the columns `[k0, end)`, every row below `k0`.
     *
     * The panel is factorised in a column-major copy, where a pivot search and every update run along a
     * column at unit stride (row-major, each reads one element per row: four times slower at order 2048);
     * a pivot's row swap reaches the columns outside the panel in place. The arithmetic is the in-place
     * algorithm's, step for step. \p panel is the copy's storage, of `(n − k0) × (end − k0)` elements or more.
     */
    void factorize_panel(buffer<T>&      panel,
                         std::size_t     k0,
                         std::size_t     end)
    {
      const std::size_t rows                  {n_ - k0};
      const std::size_t width                 {end - k0};
      transpose_panel(panel, rows, k0, end, true); // column j of the panel at panel[j * rows], every element written
      factorize_columns(panel, rows, 0, width, k0, end);
      transpose_panel(panel, rows, k0, end, false);
    }

    //! \brief Copies the columns `[k0, end)` of the factor, every row from `k0`, into \p panel (column-major, \p rows
    //!        rows) when \p in, back otherwise: by tiles of rows on the x86-64 units from `panel_tile_rows` rows, row
    //!        by row otherwise, in the plainest loops, which Apple clang compiles best (tiles of one row there ran the
    //!        whole LU of order 17 1.1 → 1.7 µs, of order 512 1.8 → 2.1 ms, on one thread).
    void transpose_panel(buffer<T>&      panel,
                         std::size_t     rows,
                         std::size_t     k0,
                         std::size_t     end,
                         bool            in)
    {
      const std::size_t width {end - k0};
#ifdef NUMERICAL_X86_COMPILED
      if (rows >= detail::panel_tile_rows) {
        T* const base {lu_.data() + (k0 * n_) + k0};
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
        return;
      }
#endif
      for (std::size_t r {0}; r < rows; ++r) {
        for (std::size_t j {0}; j < width; ++j) {
          if (in) {
            panel[(j * rows) + r] = lu_[((k0 + r) * n_) + k0 + j];
          }
          else {
            lu_[((k0 + r) * n_) + k0 + j] = panel[(j * rows) + r];
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
          detail::magnitude_t<T> best   {detail::pivot_magnitude(column[j])};
          for (std::size_t r {detail::largest_magnitude_vectors(column + j, rows - j, best, pivot)}; r < rows - j; ++r) {
            const detail::magnitude_t<T> mag {detail::pivot_magnitude(column[j + r])};
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
          const auto divisor {detail::pivot_divisor(column[j])};
          for (std::size_t r {j + 1}; r < rows; ++r) {
            column[r] = detail::over_pivot(column[r], divisor); // the multiplier, in L's slot
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
      // The plain loop, which rounds as the tiles do (the same bits): a product this narrow packed a sliver of the
      // panel's whole height for each few columns (six x86-64 AVX2 cores, LU of order 1000: 6.86 → 6.36 ms).
      detail::gemm_rows(last - mid, rows - mid, depth, minus_u.data(), depth, panel.data() + (first * rows) + mid, rows,
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
                             solve_columns(k0, end, first, std::min(first + width, n_));
                           });
    }

    //! \brief The rows `[k0, end)` of U₁₂ on the columns `[first, last)`, against the panel's unit-lower L: each
    //!        column's elements those of the whole rows' solve, whatever the columns around it.
    void solve_columns(std::size_t k0,
                       std::size_t end,
                       std::size_t first,
                       std::size_t last)
    {
      for (std::size_t i {k0 + 1}; i < end; ++i) {
        detail::subtract_products(lu_.data() + (i * n_) + first, lu_.data() + (i * n_) + k0, lu_.data() + (k0 * n_) + first, n_, i - k0, last - first);
      }
    }

    //! \brief `A₂₂ −= L₂₁ U₁₂` on the trailing columns `[c0, c1)`, as `A₂₂ += (−L₂₁) U₁₂` through the matrix product
    //!        kernel, \p left being `−L₂₁` (the rows below the panel from \p k0) as \ref detail::pack_left packed it.
    void update_columns(const detail::packed_left<T>& left,
                        std::size_t                   k0,
                        std::size_t                   c0,
                        std::size_t                   c1)
    {
      const std::size_t end {k0 + left.k};
      detail::gemm_with(left, c1 - c0, lu_.data() + (k0 * n_) + c0, n_, lu_.data() + (end * n_) + c0, n_);
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

  namespace detail {

    /*!
     * \brief The finite \p m times `2⁻ᵉ` in single precision, `2ᵉ` the least power of two above its largest
     *        magnitude: every entry below 1 in magnitude, so none overflows `float`, and the scaling exact. \return
     *        The matrix and `e`.
     */
    [[nodiscard]] inline std::pair<matrix<float>, int> scaled_to_float(const matrix<double>& m)
    {
      const double*     x     {std::to_address(m.begin())};
      const std::size_t count {m.rows() * m.cols()};
      double            l0    {0.0}; // four chains: a maximum is exact in any order
      double            l1    {0.0};
      double            l2    {0.0};
      double            l3    {0.0};
      std::size_t       i     {0};
      for (; i + 4 <= count; i += 4) {
        l0 = std::max(l0, std::abs(x[i]));
        l1 = std::max(l1, std::abs(x[i + 1]));
        l2 = std::max(l2, std::abs(x[i + 2]));
        l3 = std::max(l3, std::abs(x[i + 3]));
      }
      for (; i < count; ++i) {
        l0 = std::max(l0, std::abs(x[i]));
      }
      int exponent {0};
      static_cast<void>(std::frexp(std::max(std::max(l0, l1), std::max(l2, l3)), &exponent));
      exponent = std::max(exponent, -1021); // 2⁻ᵉ finite: entries below 2⁻¹⁰²¹ vanish in single precision whatever the scale
      const double  scale {std::ldexp(1.0, -exponent)};
      buffer<float> out(count);
      std::transform(m.begin(), m.end(), out.begin(), [scale](double v) { return static_cast<float>(v * scale); });
      return {matrix<float> {m.rows(), m.cols(), std::move(out)}, exponent};
    }

    //! \brief The largest of `‖r‖∞ / ‖x‖∞` over the columns of the residual \p r and the solution \p x (LAPACK's
    //!        `dsgesv` compares each with `‖A‖∞ ε √n`); NaN, which no bound meets, where an entry of \p r is not finite.
    [[nodiscard]] inline double residual_ratio(const matrix<double>& x,
                                               const matrix<double>& r)
    {
      // An infinity or a NaN of the solution reaches the residual: no column of A is zero (A is not singular).
      if (!all_finite(std::to_address(r.begin()), r.rows() * r.cols())) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      double ratio {0.0};
      for (std::size_t j {0}; j < x.cols(); ++j) {
        double x_norm {0.0};
        double r_norm {0.0};
        for (std::size_t i {0}; i < x.rows(); ++i) {
          x_norm = std::max(x_norm, std::abs(x(i, j)));
          r_norm = std::max(r_norm, std::abs(r(i, j)));
        }
        ratio = std::max(ratio, r_norm / x_norm);
      }
      return ratio;
    }

    /*!
     * \brief `X` of `A X = B` (`A` square, `B` of its rows) as LAPACK's `dsgesv`: `A` factorised in single precision, its solution corrected in
     *        double precision by the residual `R = B − A X` (each correction a single-precision solve of `A D = R`)
     *        until every column meets `‖R‖∞ ≤ ‖X‖∞ ‖A‖∞ ε √n` (`ε` double's unit roundoff). Each operand goes to
     *        single precision scaled by a power of two (\ref scaled_to_float): none overflows, where `dsgesv` falls
     *        back. A correction must divide the largest `‖R‖∞ / ‖X‖∞` by 8 at least, which bounds their count (about
     *        a dozen from single precision to the bound) where `dsgesv` allows 30: an `A` too ill-conditioned for
     *        single precision converges slowly or not at all, and the double factorisation is then the faster.
     *        The single factorisation costs half the double one's; each correction a product and a solve of `n² k`.
     * \return `X`, or none where an entry of `A` or `B` is not finite, `A` is singular in single precision, or a
     *         correction falls short: the caller solves in double (and meets what the double solve meets).
     */
    [[nodiscard]] inline std::optional<matrix<double>> mixed_precision_solve(const matrix<double>& a,
                                                                             const matrix<double>& b)
    {
      const std::size_t n      {a.rows()};
      double            a_norm {0.0};
      for (auto row = a.begin(); row != a.end(); row += static_cast<std::ptrdiff_t>(n)) {
        std::array<double, 4> sums {}; // four chains: the bound needs no particular rounding of the sum
        for (std::size_t j {0}; j < n; ++j) {
          sums[j % 4] += std::abs(row[static_cast<std::ptrdiff_t>(j)]);
        }
        a_norm = std::max(a_norm, (sums[0] + sums[1]) + (sums[2] + sums[3]));
      }
      if (!all_finite(std::to_address(a.begin()), n * n) || !all_finite(std::to_address(b.begin()), b.rows() * b.cols())) {
        return std::nullopt;
      }
      const double bound                     {a_norm * (std::numeric_limits<double>::epsilon() / 2.0) * std::sqrt(static_cast<double>(n))};
      auto [single_a, a_exponent] {scaled_to_float(a)};
      const lu_decomposition<float> factored {std::move(single_a)};
      if (factored.singular()) {
        return std::nullopt;
      }
      const auto solved = [&factored, a_exponent](std::pair<matrix<float>, int> rhs) { // A⁻¹ times the unscaled rhs
                            const matrix<float> y    {factored.solve(std::move(rhs.first))};
                            const double        up   {std::ldexp(1.0, rhs.second)};
                            const double        down {std::ldexp(1.0, -a_exponent)};
                            buffer<double>      out(y.rows() * y.cols());
                            std::transform(y.begin(), y.end(), out.begin(), [up, down](float v) { return (static_cast<double>(v) * up) * down; });
                            return matrix<double> {y.rows(), y.cols(), std::move(out)};
                          };
      matrix<double> x        {solved(scaled_to_float(b))};
      matrix<double> r        {b - (a * x)};
      double         ratio    {residual_ratio(x, r)};
      double         previous {std::numeric_limits<double>::infinity()};
      while (!(ratio <= bound)) {
        if (!(ratio <= previous / 8.0)) { // the last correction fell short, or the residual is not finite
          return std::nullopt;
        }
        previous = ratio;
        x        = x + solved(scaled_to_float(r));
        r        = b - (a * x);
        ratio    = residual_ratio(x, r);
      }
      return x;
    }
  } // namespace detail

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

  //! \brief How `solve_triangular` reads its matrix: as it lies, transposed, or conjugate-transposed (SciPy's `trans`
  //!        0, 1 and 2).
  enum class triangular_op : unsigned char
  {
    none,      //!< `A X = B`.
    transpose, //!< `Aᵀ X = B`.
    adjoint    //!< `Aᴴ X = B`.
  };

  namespace detail {

    //! \brief The rows \ref triangular_halves solves one by one; past them, by halves of a multiple of them.
    inline constexpr std::size_t triangular_block {32};

    //! \brief The columns of the right-hand sides a thread takes in \ref triangular_halves' rows one by one.
    inline constexpr std::size_t triangular_slice {64};

    /*!
     * \brief `A X = B` in place in \p x (`n × k`, row-major), for the `n × n` row-major triangle of \p a, its \p lower
     *        one or its upper one (the other never read), its diagonal taken as ones where \p unit: by halves
     *        (\ref triangular_halves), each half's terms out of the other by one matrix product as deep as it, as a
     *        recursive `trsm`; one column by a dot product of each row with the unknowns solved.
     */
    /*!
     * \brief Rows \p first to \p last of the triangular solve of \ref triangular_rows (`M` lower where \p lower, `M` the
     *        `n × n` row-major \p a or, \p transposed, its transpose read in place), the terms of the rows outside them
     *        already taken: up to \ref triangular_block rows one by one, a row of terms at a time; past them by halves,
     *        the half solved first taken from the other by one matrix product as deep as the half, then the other half
     *        solved. \p x holds \p k columns, its rows \p ldx apart; \p minus a half's solution negated.
     */
    template <typename T>
    void triangular_halves(const T*    a,
                           std::size_t n,
                           bool        lower,
                           bool        transposed,
                           bool        unit,
                           T*          x,
                           std::size_t ldx,
                           std::size_t k,
                           std::size_t first,
                           std::size_t last,
                           T*          minus)
    {
      const std::size_t width {last - first};
      if (width <= triangular_block) {
        std::array<T, triangular_block> terms {}; // a column of a, gathered, where M is its transpose
        for (std::size_t step {0}; step < width; ++step) {
          const std::size_t i    {lower ? first + step : last - 1 - step};
          T* const          row  {x + (i * ldx)};
          const std::size_t from {lower ? first : i + 1};
          const std::size_t to   {lower ? i : last};
          const T*          f    {a + (i * n) + from};
          if (transposed) {
            for (std::size_t j {from}; j < to; ++j) {
              terms[j - from] = a[(j * n) + i];
            }
            f = terms.data();
          }
          subtract_products(row, f, x + (from * ldx), ldx, to - from, k); // the terms in order, in registers
          if (!unit) {
            const T diagonal {a[(i * n) + i]};
            for (std::size_t c {0}; c < k; ++c) {
              row[c] /= diagonal;
            }
          }
        }
        return;
      }
      const std::size_t middle       {first + (((width / 2) + triangular_block - 1) / triangular_block * triangular_block)};
      const std::size_t solved_first {lower ? first : middle}; // the half solved first
      const std::size_t solved_last  {lower ? middle : last};
      const std::size_t other_first  {lower ? middle : first};
      const std::size_t other_last   {lower ? last : middle};
      triangular_halves(a, n, lower, transposed, unit, x, ldx, k, solved_first, solved_last, minus);
      const std::size_t depth        {solved_last - solved_first};
      for (std::size_t r {0}; r < depth; ++r) {
        for (std::size_t c {0}; c < k; ++c) {
          minus[(r * k) + c] = -x[((solved_first + r) * ldx) + c];
        }
      }
      const std::size_t rows {other_last - other_first};
      if (transposed) { // M's block is the transpose of a's block at (solved, other)
        gemm_at(rows, k, depth, a + (solved_first * n) + other_first, n, minus, k, x + (other_first * ldx), ldx);
      }
      else {
        gemm(rows, k, depth, a + (other_first * n) + solved_first, n, minus, k, x + (other_first * ldx), ldx);
      }
      triangular_halves(a, n, lower, transposed, unit, x, ldx, k, other_first, other_last, minus);
    }

    /*!
     * \brief `M X = B` in place in \p x (`n × k`, row-major) for `M` the `n × n` row-major \p a or, \p transposed, its
     *        transpose, read in place; \p lower whether \p a (not `M`) is lower-triangular, the other triangle never
     *        read, its diagonal taken as ones where \p unit. By halves (\ref triangular_halves), as a recursive `trsm`,
     *        the columns by blocks of \ref triangular_slice on the threads; one column of `M = a` by a dot product of
     *        each row with the unknowns solved, of `M = aᵀ` by a sweep, each unknown solved taken from the others by
     *        its row of \p a.
     */
    template <typename T>
    void triangular_rows(const T*    a,
                         std::size_t n,
                         bool        lower,
                         bool        transposed,
                         bool        unit,
                         T*          x,
                         std::size_t k)
    {
      if (k == 1 && !transposed) { // each unknown its row's dot product with those solved, the row read where it runs
        for (std::size_t step {0}; step < n; ++step) {
          const std::size_t i     {lower ? step : n - 1 - step};
          const std::size_t from  {lower ? 0 : i + 1};
          const std::size_t count {lower ? i : n - i - 1};
          const T           value {x[i] - dot_product(a + (i * n) + from, x + from, count)};
          x[i] = unit ? value : value / a[(i * n) + i];
        }
        return;
      }
      if (k == 1) { // aᵀ: each unknown solved, then its terms out of the rest along its row of a
        for (std::size_t step {0}; step < n; ++step) {
          const std::size_t j {lower ? n - 1 - step : step};
          if (!unit) {
            x[j] /= a[(j * n) + j];
          }
          const std::size_t from {lower ? 0 : j + 1};
          subtract_product(x + from, a + (j * n) + from, x[j], lower ? j : n - j - 1);
        }
        return;
      }
      const bool m_lower {lower != transposed};
      if (k <= triangular_slice) {                               // one block of columns: the products take the threads
        buffer<T> minus((((n + 1) / 2) + triangular_block) * k); // a half's solution, negated
        triangular_halves(a, n, m_lower, transposed, unit, x, k, k, 0, n, minus.data());
        return;
      }
      // The columns apart, a block of them to a thread, each solved by halves: the same operations on each element
      // whatever the threads.
      parallel_for((k + triangular_slice - 1) / triangular_slice, n * n * triangular_slice / 2, [&](std::size_t block) {
                     const std::size_t start {block * triangular_slice};
                     const std::size_t count {std::min(triangular_slice, k - start)};
                     buffer<T>         minus((((n + 1) / 2) + triangular_block) * count); // a half's solution, negated
                     triangular_halves(a, n, m_lower, transposed, unit, x + start, k, count, 0, n, minus.data());
                   });
    }
  } // namespace detail

  namespace detail {

    //! \brief Whether the \p count elements at \p x are finite, one at a time: a type no vectorised pass takes.
    template <typename T>
    [[nodiscard]] bool elements_finite(const T*    x,
                                       std::size_t count)
    {
      return std::all_of(x, x + count, [](const T& value) { return is_finite_entry(value); });
    }

    //! \brief \ref elements_finite of `double`: \ref all_finite_shared, vectorised.
    [[nodiscard]] inline bool elements_finite(const double* x,
                                              std::size_t   count)
    {
      return all_finite_shared(x, count);
    }

    //! \brief The same for `std::complex<double>`, whose array holds its parts in order ([complex.numbers]).
    [[nodiscard]] inline bool elements_finite(const std::complex<double>* x,
                                              std::size_t                 count)
    {
      return all_finite_shared(reinterpret_cast<const double*>(x), 2 * count); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
    }

    //! \brief Whether the triangle of the row-major `n × n` \p a a solve reads (the diagonal with it) is finite, entry
    //!        by entry: the rare road of \ref require_finite_triangle, where the whole matrix is not.
    template <typename T>
    NUMERICAL_OUT_OF_LINE bool triangle_is_finite(const T*    a,
                                                  std::size_t n,
                                                  bool        lower)
    {
      for (std::size_t i {0}; i < n; ++i) {
        const T* const row {a + (i * n)};
        if (!(lower ? elements_finite(row, i + 1) : elements_finite(row + i, n - i))) {
          return false;
        }
      }
      return true;
    }

    /*!
     * \brief Refuses an infinite or NaN entry of the triangle of \p a a triangular solve reads, or of its right-hand
     *        side \p x (\p count elements): one vectorised pass over the whole matrix, then the triangle entry by entry
     *        only where that pass finds one (the other triangle may hold anything); the right-hand side in one pass.
     * \throws non_finite_error If an entry read is infinite or NaN.
     */
    template <typename T>
    void require_finite_triangle(const T*    a,
                                 std::size_t n,
                                 bool        lower,
                                 const T*    x,
                                 std::size_t count)
    {
      if (!elements_finite(a, n * n) && !triangle_is_finite(a, n, lower)) {
        throw non_finite_error("solve_triangular: the matrix holds an infinite or NaN entry");
      }
      if (!elements_finite(x, count)) {
        throw non_finite_error("solve_triangular: the right-hand side holds an infinite or NaN entry");
      }
    }

    /*!
     * \brief \ref solve_triangular_in_place of a triangle and a right-hand side known finite (\ref
     *        require_finite_triangle, or a check that covered them): a zero on the diagonal still refused.
     */
    template <typename T>
    void solve_finite_triangular_in_place(const T*      a,
                                          std::size_t   n,
                                          bool          lower,
                                          triangular_op op,
                                          bool          unit_diagonal,
                                          T*            x,
                                          std::size_t   k)
    {
      for (std::size_t i {0}; i < n && !unit_diagonal; ++i) {
        if (a[(i * n) + i] == T {}) {
          throw singular_matrix_error("singular matrix: resolution failed at diagonal " + std::to_string(i + 1));
        }
      }
      const bool adjoint {op == triangular_op::adjoint}; // a real one's conjugates are itself
      if (adjoint) {
        std::transform(x, x + (n * k), x, [](const T& value) { return conj_of(value); });
      }
      triangular_rows(a, n, lower, op != triangular_op::none, unit_diagonal, x, k);
      if (adjoint) {
        std::transform(x, x + (n * k), x, [](const T& value) { return conj_of(value); });
      }
    }

    /*!
     * \brief \ref numerical::solve_triangular on the `n × n` row-major \p a, in place in \p x (`n × k`, row-major):
     *        the triangle checked, then read where it lies, transposed in place; the adjoint by the transpose on the
     *        conjugates (`Aᴴ X = B` is `Aᵀ conj X = conj B`).
     */
    template <typename T>
    void solve_triangular_in_place(const T*      a,
                                   std::size_t   n,
                                   bool          lower,
                                   triangular_op op,
                                   bool          unit_diagonal,
                                   T*            x,
                                   std::size_t   k)
    {
      require_finite_triangle(a, n, lower, x, n * k);
      solve_finite_triangular_in_place(a, n, lower, op, unit_diagonal, x, k);
    }
  } // namespace detail

  /*!
   * \brief Solves `A X = B`, `Aᵀ X = B` or `Aᴴ X = B` (\p op) for a triangular `A`, as `scipy.linalg.solve_triangular`:
   *        its \p lower triangle or its upper one is read, the other never, and its diagonal as ones where
   *        \p unit_diagonal. By substitution, a block's rows by rows of terms and the rows past it through the matrix
   *        product kernel: `n² k` multiply-adds, where `solve` factorises `A` first.
   * \param[in] a             The square matrix (`n × n`).
   * \param[in] b             The right-hand sides (`n × k`).
   * \param[in] lower         Whether `A` is lower triangular, else upper.
   * \param[in] op            `A` as it lies, transposed or conjugate-transposed.
   * \param[in] unit_diagonal Whether `A`'s diagonal is taken as ones (not read).
   * \return `X` (`n × k`).
   * \throws dimension_error       If `a` is not square or `b`'s row count differs from it.
   * \throws non_finite_error      If an entry of `a`'s triangle or of `b` is infinite or NaN.
   * \throws singular_matrix_error If a diagonal entry read is zero (SciPy's message, its 1-based row).
   */
  template <typename T>
  [[nodiscard]] matrix<T> solve_triangular(const matrix<T>& a,
                                           matrix<T>        b,
                                           bool             lower,
                                           triangular_op    op            = triangular_op::none,
                                           bool             unit_diagonal = false)
  {
    const std::size_t n {a.rows()};
    if (a.cols() != n) {
      throw dimension_error("solve_triangular: expected a square matrix");
    }
    if (b.rows() != n) {
      throw dimension_error("solve_triangular: right-hand side row count differs from the matrix");
    }
    const std::size_t k {b.cols()};
    buffer<T>         x {std::move(b).elements()};
    detail::solve_triangular_in_place(std::to_address(a.begin()), n, lower, op, unit_diagonal, x.data(), k);
    return matrix<T> {n, k, std::move(x)};
  }

  //! \brief \ref solve_triangular of one right-hand side.
  template <typename T>
  [[nodiscard]] vector<T> solve_triangular(const matrix<T>& a,
                                           const vector<T>& b,
                                           bool             lower,
                                           triangular_op    op            = triangular_op::none,
                                           bool             unit_diagonal = false)
  {
    matrix<T> column {solve_triangular(a, matrix<T> {b.size(), 1, buffer<T>(b.begin(), b.end())}, lower, op, unit_diagonal)};
    return vector<T> {std::move(column).elements()};
  }

  //! \brief The pivoted LU factorisation of an `m × n` matrix (\ref lu_factors): `A = P L U`.
  template <typename T>
  struct rectangular_lu
  {
    std::vector<std::size_t> rows; //!< Row `i` of `P⁻¹ A = L U` is row `rows[i]` of `A`.
    matrix<T>                l;    //!< `m × k` (`k = min(m, n)`), lower trapezoidal with a unit diagonal.
    matrix<T>                u;    //!< `k × n`, upper trapezoidal.
  };

  namespace detail {

    //! \brief The rows of \p lu (`rows × cols` row-major, `L` below its diagonal, `U` on and above) as `L`, its first
    //!        \p k columns (unit diagonal), and `U`, its first \p k rows.
    template <typename T>
    [[nodiscard]] std::pair<matrix<T>, matrix<T>> split_lu(const T*    lu,
                                                           std::size_t rows,
                                                           std::size_t cols,
                                                           std::size_t k)
    {
      buffer<T> l(rows * k, T {});
      buffer<T> u(k * cols, T {});
      for (std::size_t i {0}; i < rows; ++i) {
        for (std::size_t j {0}; j < k; ++j) {
          l[(i * k) + j] = i == j ? T {1} : (j < i ? lu[(i * cols) + j] : T {});
        }
      }
      for (std::size_t i {0}; i < k; ++i) {
        for (std::size_t j {i}; j < cols; ++j) {
          u[(i * cols) + j] = lu[(i * cols) + j];
        }
      }
      return {matrix<T> {rows, k, std::move(l)}, matrix<T> {k, cols, std::move(u)}};
    }

    /*!
     * \brief Factorises the columns `[first, last)` of the `m × n` column-major \p panel (`m ≥ n`; column `j` at
     *        `panel[j m]`) by partial pivoting, in place, as LAPACK's `dgetrf2`: the left half factorised, the top of
     *        the right half solved against its unit-lower `L`, the bottom updated by one product, then the right half;
     *        a leaf of \ref solve_leaf columns, or a panel too short for its products to pay, unblocked. Each pivot's
     *        row swap reaches every column and \p rows (row `i` of the factors is row `rows[i]` of the input); a zero
     *        pivot column is left as it is.
     */
    template <typename T>
    void tall_lu_columns(T*                        panel,
                         std::size_t               m,
                         std::size_t               n,
                         std::size_t               first,
                         std::size_t               last,
                         std::vector<std::size_t>& rows)
    {
      if (last - first <= solve_leaf || m - first < 2 * lu_block_for(m)) {
        for (std::size_t j {first}; j < last; ++j) {
          T* const       column {panel + (j * m)};
          std::size_t    pivot  {0}; // from row j
          magnitude_t<T> best   {std::abs(column[j])};
          for (std::size_t r {largest_magnitude_vectors(column + j, m - j, best, pivot)}; r < m - j; ++r) {
            const magnitude_t<T> mag {std::abs(column[j + r])};
            if (mag > best) {
              best  = mag;
              pivot = r;
            }
          }
          pivot += j;
          if (best == magnitude_t<T> {}) {
            continue; // a zero column below the diagonal: nothing to eliminate
          }
          if (pivot != j) {
            for (std::size_t c {0}; c < n; ++c) {
              std::swap(panel[(c * m) + j], panel[(c * m) + pivot]);
            }
            std::swap(rows[j], rows[pivot]);
          }
          const auto divisor {pivot_divisor(column[j])};
          for (std::size_t r {j + 1}; r < m; ++r) {
            column[r] = over_pivot(column[r], divisor); // the multiplier, in L's slot
          }
          for (std::size_t c {j + 1}; c < last; ++c) {
            T* const target {panel + (c * m)};
            subtract_product(target + j + 1, column + j + 1, target[j], m - (j + 1));
          }
        }
        return;
      }
      const std::size_t mid {first + ((last - first) / 2)};
      tall_lu_columns(panel, m, n, first, mid, rows);
      for (std::size_t c {mid}; c < last; ++c) { // the top of the right half: rows [first, mid)
        T* const target {panel + (c * m)};
        for (std::size_t j {first}; j < mid; ++j) {
          subtract_product(target + j + 1, panel + (j * m) + j + 1, target[j], mid - (j + 1));
        }
      }
      // The bottom, rows [mid, m): A₂₂ −= L₂₁ U₁₂, column-major, so as its transpose, row-major: A₂₂ᵀ += (−U₁₂ᵀ) L₂₁ᵀ.
      const std::size_t depth                           {mid - first};
      buffer<T>         minus_u((last - mid) * depth, T {});
      for (std::size_t c {mid}; c < last; ++c) {
        for (std::size_t j {first}; j < mid; ++j) {
          minus_u[((c - mid) * depth) + (j - first)] = -panel[(c * m) + j];
        }
      }
      gemm_rows(last - mid, m - mid, depth, minus_u.data(), depth, panel + (first * m) + mid, m, panel + (mid * m) + mid, m);
      tall_lu_columns(panel, m, n, mid, last, rows);
    }

    //! \brief The ratio of rows to columns up to which \ref lu_factors factorises a tall matrix as the square of it
    //!        beside zero columns (`m³ / 3`, by the blocked LU), past which by \ref tall_lu_columns (`m n²` on narrow
    //!        products): the two even near twice as tall as wide (arm64, 1200 rows).
    inline constexpr std::size_t lu_tall_ratio {2};
  } // namespace detail

  /*!
   * \brief The pivoted LU factorisation `A = P L U` of an `m × n` matrix, as `scipy.linalg.lu`: `L` `m × k` lower
   *        trapezoidal with a unit diagonal, `U` `k × n` upper trapezoidal, `k = min(m, n)`, rows chosen by partial
   *        pivoting. A wide matrix's square part by \ref lu_decomposition and `U`'s other columns by `L⁻¹ P A` (a
   *        triangular solve); a tall one by \ref lu_decomposition of it beside zero columns (its first `n` columns factorised
   *        alike) up to \ref detail::lu_tall_ratio times as tall as wide, by `dgetrf2`'s recursion over its columns
   *        (\ref detail::tall_lu_columns) past. A zero pivot leaves its column of `L` zero, as LAPACK's.
   * \tparam T `double` or `std::complex<double>`.
   * \throws non_finite_error If an entry is infinite or NaN.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] rectangular_lu<T> lu_factors(const matrix<T>& a)
  {
    const std::size_t m {a.rows()};
    const std::size_t n {a.cols()};
    const std::size_t k {std::min(m, n)};
    detail::require_finite(a, "LU decomposition");
    std::vector<std::size_t> identity(m);
    std::iota(identity.begin(), identity.end(), std::size_t {0});
    if (k == 0) {
      return {std::move(identity), matrix<T> {m, k, buffer<T>(m * k)}, matrix<T> {k, n, buffer<T>(k * n)}};
    }
    if (m <= n) { // the square part's factorisation; U's other columns L⁻¹ (its rows in the pivoted order)
      buffer<T> square(m * m);
      for (std::size_t i {0}; i < m; ++i) {
        std::copy_n(&a(i, 0), m, square.data() + (i * m));
      }
      const lu_decomposition<T> f  {matrix<T> {m, m, std::move(square)}};
      const matrix<T>           lu {f.factors()};
      buffer<T>                 combined(m * n);
      for (std::size_t i {0}; i < m; ++i) {
        std::copy_n(&lu(i, 0), m, combined.data() + (i * n));
      }
      if (n > m) {
        buffer<T> rest(m * (n - m));
        for (std::size_t i {0}; i < m; ++i) {
          std::copy_n(&a(f.pivots()[i], m), n - m, rest.data() + (i * (n - m)));
        }
        detail::solve_triangular_in_place(std::to_address(lu.begin()), m, true, triangular_op::none, true, rest.data(), n - m); // L⁻¹
        for (std::size_t i {0}; i < m; ++i) {
          std::copy_n(rest.data() + (i * (n - m)), n - m, combined.data() + (i * n) + m);
        }
      }
      auto [l, u] {detail::split_lu(combined.data(), m, n, k)};
      return {f.pivots(), std::move(l), std::move(u)};
    }
    if (m <= detail::lu_tall_ratio * n) { // beside zero columns, square
      buffer<T> padded(m * m, T {});
      for (std::size_t i {0}; i < m; ++i) {
        std::copy_n(&a(i, 0), n, padded.data() + (i * m));
      }
      const lu_decomposition<T> f  {matrix<T> {m, m, std::move(padded)}};
      const matrix<T>           lu {f.factors()};
      auto [l, u] {detail::split_lu(std::to_address(lu.begin()), m, m, k)};
      buffer<T> top(k * n); // U's first n columns
      for (std::size_t i {0}; i < k; ++i) {
        std::copy_n(&u(i, 0), n, top.data() + (i * n));
      }
      return {f.pivots(), std::move(l), matrix<T> {k, n, std::move(top)}};
    }
    buffer<T> panel(m * n); // column-major: column j at panel[j m]
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        panel[(j * m) + i] = a(i, j);
      }
    }
    detail::tall_lu_columns(panel.data(), m, n, 0, n, identity);
    buffer<T> combined(m * n);
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        combined[(i * n) + j] = panel[(j * m) + i];
      }
    }
    auto [l, u] {detail::split_lu(combined.data(), m, n, k)};
    return {std::move(identity), std::move(l), std::move(u)};
  }

  //! \brief Thrown when a Cholesky factorisation finds the matrix not positive-definite.
  class not_positive_definite_error : public std::runtime_error
  {
  public:

    //! \brief Builds the error. \param[in] message Human-readable cause. \param[in] order As \ref order gives it.
    explicit not_positive_definite_error(const std::string& message,
                                         std::size_t        order = 0)
      : std::runtime_error(message), order_ {order}
    {}

    //! \brief The order of the first leading minor found not positive definite (LAPACK's `info`), 0 where unknown.
    [[nodiscard]] std::size_t order() const noexcept
    {
      return order_;
    }

  private:

    std::size_t order_; //!< The order of the failing leading minor, or 0.
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
     * \brief Factorises \p a. \param[in] a The Hermitian positive-definite matrix, its lower triangle alone read.
     * \throws dimension_error If \p a is not square.
     * \throws non_finite_error If an entry of its lower triangle is infinite or NaN.
     * \throws not_positive_definite_error If \p a is not positive-definite.
     */
    explicit cholesky_decomposition(const matrix<T>& a)
      : cholesky_decomposition(detail::parallel_copy_of(a))
    {}

    /*!
     * \brief Factorises \p a in its own storage, an expiring matrix's elements taken over: no copy of them.
     * \param[in] a The Hermitian positive-definite matrix, its lower triangle alone read.
     * \throws dimension_error If \p a is not square.
     * \throws non_finite_error If an entry of its lower triangle is infinite or NaN.
     * \throws not_positive_definite_error If \p a is not positive-definite.
     */
    explicit cholesky_decomposition(matrix<T>&& a)
      : n_(a.rows())
    {
      if (a.cols() != n_) {
        throw dimension_error("Cholesky decomposition requires a square matrix");
      }
      detail::require_finite_lower(a, "Cholesky decomposition");
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
        y[i] = detail::over_real_diagonal(sum, lower_[(i * n_) + i]);
      }
      buffer<T> x(n_, T {});
      for (std::size_t i {n_}; i-- > 0;) { // back: Lᴴ x = y, (Lᴴ)[i][k] = conj(L[k][i])
        T sum {y[i]};
        for (std::size_t k {i + 1}; k < n_; ++k) {
          sum -= detail::conj_of(lower_[(k * n_) + i]) * x[k];
        }
        x[i] = detail::over_real_diagonal(sum, lower_[(i * n_) + i]); // its own conjugate
      }
      return vector<T> {std::move(x)};
    }

    /*!
     * \brief Solves `A X = B` for every column of \p b at once: `L Y = B`, then `Lᴴ X = Y`, row by row, each row of the
     *        solution its column's terms in \ref solve's order (so each column \ref solve's bits), a row of terms
     *        at a time over the columns; the threads share the columns by slices of 8 and more.
     * \param[in] b The right-hand sides (`n × k`).
     * \return `X` (`n × k`).
     * \throws dimension_error If `b`'s row count differs from the matrix.
     */
    [[nodiscard]] matrix<T> solve(const matrix<T>& b) const
    {
      if (b.rows() != n_) {
        throw dimension_error("solve: right-hand side row count differs from the matrix");
      }
      const std::size_t cols   {b.cols()};
      buffer<T>         x(b.begin(), b.end());
      const std::size_t slices {std::max<std::size_t>(1, std::min(max_threads(), cols / 8))};
      const std::size_t slice  {(cols + slices - 1) / slices};
      detail::parallel_for(slices, n_ * n_ * slice, [&](std::size_t s) {
                             const std::size_t c0 {s * slice};
                             const std::size_t c1 {std::min(cols, c0 + slice)};
                             for (std::size_t i {0}; i < n_; ++i) { // forward: L Y = B
                               T* const row {x.data() + (i * cols)};
                               for (std::size_t k {0}; k < i; ++k) {
                                 const T        l     {lower_[(i * n_) + k]};
                                 const T* const above {x.data() + (k * cols)};
                                 for (std::size_t c {c0}; c < c1; ++c) {
                                   row[c] -= l * above[c];
                                 }
                               }
                               for (std::size_t c {c0}; c < c1; ++c) {
                                 row[c] = detail::over_real_diagonal(row[c], lower_[(i * n_) + i]);
                               }
                             }
                             for (std::size_t i {n_}; i-- > 0;) { // back: Lᴴ X = Y, (Lᴴ)[i][k] = conj(L[k][i])
                               T* const row {x.data() + (i * cols)};
                               for (std::size_t k {i + 1}; k < n_; ++k) {
                                 const T        l     {detail::conj_of(lower_[(k * n_) + i])};
                                 const T* const below {x.data() + (k * cols)};
                                 for (std::size_t c {c0}; c < c1; ++c) {
                                   row[c] -= l * below[c];
                                 }
                               }
                               for (std::size_t c {c0}; c < c1; ++c) {
                                 row[c] = detail::over_real_diagonal(row[c], lower_[(i * n_) + i]);
                               }
                             }
                           });
      return matrix<T> {n_, cols, std::move(x)};
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
      buffer<T> packing;        // L₂₁ packed for the trailing update's tiles, kept from panel to panel
      buffer<T> minus_adjoint;  // −L₂₁ᴴ, row-major, the same
      for (std::size_t k0 {0}; k0 < n_; k0 += block_) {
        const std::size_t end {std::min(k0 + block_, n_)};
        std::size_t       i   {k0};
        for (; end - k0 >= detail::cholesky_together_width && i + detail::cholesky_diagonal_rows <= end; i += detail::cholesky_diagonal_rows) {
          factor_rows_together(i, k0);
          for (std::size_t r {0}; r < detail::cholesky_diagonal_rows; ++r) { // their triangle, row after row
            factor_row(i + r, k0, i, i + r + 1);
          }
        }
        for (; i < end; ++i) {
          factor_row(i, k0, k0, i + 1);
        }
        if (end < n_) {
          minus_adjoint.resize((end - k0) * (n_ - end));
          factor_rows_below(k0, end, minus_adjoint);
          update_trailing(k0, end, minus_adjoint, packing);
        }
      }
      T* const          rows {lower_.data()};
      const std::size_t n    {n_};
      // The upper triangle zeroed by bands of rows on the threads: a pass bound by the memory, half a row an element.
      detail::parallel_for((n + detail::zeroed_rows - 1) / detail::zeroed_rows, detail::memory_element_weight * detail::zeroed_rows * n / 2, [rows, n](std::size_t band) {
                             for (std::size_t i {band * detail::zeroed_rows}; i < std::min(n, (band + 1) * detail::zeroed_rows); ++i) {
                               std::fill(rows + (i * n) + i + 1, rows + ((i + 1) * n), T {});
                             }
                           });
    }

    /*!
     * \brief The columns `[k0, i)` of the \ref detail::cholesky_diagonal_rows rows from \p i, together: each element
     *        depends on its own row's earlier columns and on the rows above \p i alone, so the rows' sums are chains of
     *        their own, where row by row each term waited on the last. Each element is \ref factor_row's, its terms in
     *        the same order. The rows' loops unrolled: rolled, GCC kept the sums in memory (order 33 on one arm64
     *        core: 8.3 µs, against 2.8 unrolled and 4.6 row by row).
     */
    void factor_rows_together(std::size_t i,
                              std::size_t k0)
    {
      std::array<T, detail::cholesky_diagonal_rows> sums {};
      for (std::size_t j {k0}; j < i; ++j) {
        const T* const above {lower_.data() + (j * n_)};
        NUMERICAL_UNROLL_BLOCK
        for (std::size_t r {0}; r < detail::cholesky_diagonal_rows; ++r) {
          sums[r] = lower_[((i + r) * n_) + j];
        }
        for (std::size_t k {k0}; k < j; ++k) {
          const T other {detail::conj_of(above[k])};
          NUMERICAL_UNROLL_BLOCK
          for (std::size_t r {0}; r < detail::cholesky_diagonal_rows; ++r) {
            sums[r] -= lower_[((i + r) * n_) + k] * other;
          }
        }
        NUMERICAL_UNROLL_BLOCK
        for (std::size_t r {0}; r < detail::cholesky_diagonal_rows; ++r) {
          lower_[((i + r) * n_) + j] = detail::over_real_diagonal(sums[r], above[j]);
        }
      }
    }

    /*!
     * \brief Row \p i of L in the columns `[first, last)`, the terms of the columns before \p k0 already subtracted:
     *        the unblocked step, `L[i][j] = (A[i][j] − Σₖ L[i][k]·conj(L[j][k])) / L[j][j]` over `k ∈ [k0, j)`.
     * \throws not_positive_definite_error At a diagonal that is not positive.
     */
    void factor_row(std::size_t i,
                    std::size_t k0,
                    std::size_t first,
                    std::size_t last)
    {
      for (std::size_t j {first}; j < last; ++j) {
        T sum {lower_[(i * n_) + j]};
        for (std::size_t k {k0}; k < j; ++k) {
          sum -= lower_[(i * n_) + k] * detail::conj_of(lower_[(j * n_) + k]);
        }
        if (i == j) {
          const detail::carried_t<T> diagonal {detail::real_part(sum)};
          if (diagonal <= 0) {
            throw not_positive_definite_error("Cholesky: matrix is not positive-definite", i + 1);
          }
          lower_[(i * n_) + i] = detail::from_carried<T>(std::sqrt(diagonal));
        }
        else {
          lower_[(i * n_) + j] = detail::over_real_diagonal(sum, lower_[(j * n_) + j]);
        }
      }
    }

    /*!
     * \brief The rows below the panel `[k0, end)`, `L₂₁ = A₂₁ L₁₁⁻ᴴ`: chunks of \ref detail::cholesky_rows rows in
     *        parallel, each in a column-major copy where column `j` takes `L[i][j] −= L[i][k]·conj(L[j][k])` for
     *        `k` ascending, then the division by `L[j][j]`, for every row of the chunk at once along the column: by
     *        sub-blocks of 16 columns, the terms of the sub-blocks before one through the matrix product's plain loop, its
     *        own column by column (\ref detail::subtract_product), each element's terms in ascending order, each fused
     *        as the column step fuses it, so the unblocked step's arithmetic, in its order (one core of six AVX2, order
     *        1000: 4.95 → 3.95 ms; all column by column, the solve took 37 % of the factorisation for a tenth of its
     *        work); row by row, each sum waited on its last addition. Each chunk also writes its columns of
     *        \p minus_adjoint (`−L₂₁ᴴ`, row-major), the trailing update's right operand, which a copy on one thread after
     *        it took 15 to 20 % of the factorisation on six x86-64 cores (order 1024 to 2048).
     */
    void factor_rows_below(std::size_t k0,
                           std::size_t end,
                           buffer<T>&  minus_adjoint)
    {
      const std::size_t width  {end - k0};
      const std::size_t chunks {((n_ - end) + detail::cholesky_rows - 1) / detail::cholesky_rows};
      detail::parallel_for(chunks, detail::cholesky_rows * width * width / 2, [&](std::size_t chunk) {
                             const std::size_t first {end + (chunk * detail::cholesky_rows)};
                             const std::size_t height {std::min(detail::cholesky_rows, n_ - first)};
                             buffer<T>    panel(height * width); // column j at panel[j * height], every element written
                             for (std::size_t r {0}; r < height; ++r) {
                               for (std::size_t j {0}; j < width; ++j) {
                                 panel[(j * height) + r] = lower_[((first + r) * n_) + k0 + j];
                               }
                             }
                             constexpr std::size_t sub {16};
                             buffer<T>             minus_conj;
                             for (std::size_t s0 {0}; s0 < width; s0 += sub) {
                               const std::size_t s1 {std::min(s0 + sub, width)};
                               if (s0 > 0) { // the earlier sub-blocks' terms, through the product kernel: Pᵀ[s0, s1) += (−conj L₁₁[s0, s1) × [0, s0)) Pᵀ[0, s0)
                                 minus_conj.resize((s1 - s0) * s0);
                                 for (std::size_t j {s0}; j < s1; ++j) {
                                   for (std::size_t k {0}; k < s0; ++k) {
                                     minus_conj[((j - s0) * s0) + k] = -detail::conj_of(lower_[((k0 + j) * n_) + k0 + k]);
                                   }
                                 }
                                 // The plain loop, which rounds as the tiles do (the same bits), where the packed
                                 // product packed slivers too small to pay for it (six AVX2 cores' one, order 1000:
                                 // this pass 2.82 → 2.23 ms).
                                 detail::gemm_rows(s1 - s0, height, s0, minus_conj.data(), s0, panel.data(), height, panel.data() + (s0 * height), height);
                               }
                               for (std::size_t j {s0}; j < s1; ++j) {
                                 T* const column {panel.data() + (j * height)};
                                 detail::subtract_panel_columns(column, panel.data() + (s0 * height), height, lower_.data() + ((k0 + j) * n_) + k0 + s0, j - s0);
                                 const T pivot {lower_[((k0 + j) * n_) + k0 + j]};
                                 for (std::size_t r {0}; r < height; ++r) {
                                   column[r] = detail::over_real_diagonal(column[r], pivot);
                                 }
                               }
                             }
                             const std::size_t below {n_ - end};
#ifdef NUMERICAL_X86_COMPILED
                             // On the x86-64 units the factor's rows, then −L₂₁ᴴ's columns, each written along: written
                             // together, one of them went across (one AVX2 core, order 1000: this pass 2.23 → 1.97 ms).
                             // Together on arm64, where apart was 3 % slower under GCC 14 (orders 48 to 128).
                             for (std::size_t r {0}; r < height; ++r) {
                               for (std::size_t j {0}; j < width; ++j) {
                                 lower_[((first + r) * n_) + k0 + j] = panel[(j * height) + r];
                               }
                             }
                             for (std::size_t j {0}; j < width; ++j) {
                               const T* const from {panel.data() + (j * height)};
                               T* const       to {minus_adjoint.data() + (j * below) + (first - end)};
                               for (std::size_t r {0}; r < height; ++r) {
                                 to[r] = -detail::conj_of(from[r]);
                               }
                             }
#else
                             for (std::size_t r {0}; r < height; ++r) {
                               for (std::size_t j {0}; j < width; ++j) {
                                 const T value {panel[(j * height) + r]};
                                 lower_[((first + r) * n_) + k0 + j]            = value;
                                 minus_adjoint[(j * below) + (first - end + r)] = -detail::conj_of(value);
                               }
                             }
#endif
                           });
    }

    /*!
     * \brief `A22 += L21 (−L21ᴴ)`, by tiles of \ref detail::cholesky_update_columns columns and
     *        \ref detail::cholesky_update_rows rows on and below the diagonal blocks (the upper triangle, never read,
     *        takes what falls in a block past the diagonal), in parallel, each element's terms in the order of one
     *        product, `L₂₁` read in the factor and \p minus_adjoint as \ref factor_rows_below wrote it: the sign on the
     *        right, `(−a) b` being `a (−b)` exactly, so no negated copy of `L₂₁` (one AVX2 core, order 100: 48 → 45 µs,
     *        300 −4 %). `L₂₁` is packed once, in
     *        \p packing, for the tile the products run on (\ref detail::pack_left), and every tile reads its rows there
     *        (a tile packed its own, again for each block of columns: 8 % of the factorisation on one AVX2 core at order
     *        1000): so a tile's rows start at a multiple of the tile's slivers, a few rows above the diagonal block.
     */
    void update_trailing(std::size_t      k0,
                         std::size_t      end,
                         const buffer<T>& minus_adjoint,
                         buffer<T>&       packing)
    {
      const std::size_t                                rows    {n_ - end};
      const std::size_t                                depth   {end - k0};
#ifdef NUMERICAL_X86_COMPILED
      // On the x86-64 units a real product the kernel packs runs on the triangle's tiles alone (detail::gemm_triangle,
      // each element the tiles' sum below), where the blocks below computed the part of each diagonal block past the
      // diagonal: six AVX2 cores, order 500 787 → 721 µs; a float one on the float tiles, where the blocks' narrow
      // kernel held it near the double's time. On arm64 the blocks stay, faster there (all cores, order 500:
      // 0.85 ms, by the triangle 1.03: the AMX takes −L₂₁ packed once for every block). A smaller or complex product by
      // the blocks.
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
        if (depth >= detail::gemm_min_depth && rows * rows >= ((detail::gemm_packed_threshold + depth) - 1) / depth) {
          detail::gemm_triangle<detail::a_rows, detail::b_rows>(rows, depth, lower_.data() + (end * n_) + k0, n_, minus_adjoint.data(), rows,
                                                                lower_.data() + (end * n_) + end, n_, true);
          return;
        }
      }
#endif
      const std::size_t                                step    {std::max(block_, detail::cholesky_update_columns)};
      // Packed once where a block of rows serves several tiles: past two tiles' rows (one AVX2 core, the shared packing
      // cost order 300 5 % and 500 3 %, where a block of columns has one or two tiles), else each tile packs its own.
      const bool                                       shared  {rows > 2 * detail::cholesky_update_rows};
      const T* const                                   l21     {lower_.data() + (end * n_) + k0};
      const detail::packed_left<T>                     left    {shared ? detail::pack_left(rows, step, depth, l21, n_, packing)
                                                                       : detail::read_in_place(rows, depth, l21, n_)};
      const std::size_t                                granule {std::max<std::size_t>(1, left.granule)}; // a row at least
      const std::size_t                                height  {std::max(granule, detail::cholesky_update_rows / granule * granule)};
      std::vector<std::pair<std::size_t, std::size_t>> tiles;                                            // each one's first column and first row, below the panel
      for (std::size_t jb {0}; jb < rows; jb += step) {
        for (std::size_t r0 {jb / granule * granule}; r0 < rows; r0 += height) {
          tiles.emplace_back(jb, r0);
        }
      }
      detail::parallel_for(tiles.size(), height * step * depth / detail::gemm_kernel_speedup, [&](std::size_t t) {
                             const auto [jb, r0] {tiles[t]};
                             detail::gemm_with(detail::left_rows(left, r0, std::min(height, rows - r0)), std::min(step, rows - jb),
                                               minus_adjoint.data() + jb, rows, lower_.data() + ((end + r0) * n_) + end + jb, n_);
                           });
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
     * \brief Applies `I − s v vᴴ` from the left (\p s 2 for a unit `v`) to the rows `first…first+|v|-1` of a
     *        row-major array, on its columns `from…cols-1`.
     *
     * Row by row: the projection `w = vᴴ A` accumulates one contiguous row at a time, then each row takes
     * `−s vᵢ w`. A column-by-column walk strides by `cols` on every access and runs several times slower. The
     * columns are independent, so chunks of them run in parallel, each element's arithmetic in the same order.
     * \param[in,out] a     The array, `cols` entries per row.
     * \param[in]     cols  Its row length.
     * \param[in]     v     The reflector (unit where \p s is 2).
     * \param[in]     first The first row it acts on.
     * \param[in]     from  The first column it acts on.
     * \param[in]     s     The reflector's factor.
     */
    template <typename T>
    void reflect_rows(buffer<T>&            a,
                      std::size_t           cols,
                      const buffer<T>&      v,
                      std::size_t           first,
                      std::size_t           from,
                      const T&              s = T {2})
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
                       const T scale {s * v[i]};
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
     * \brief Gathers the reflectors `reflectors[k0…k1-1]` (each `I − τ_k v vᴴ`: `τ_k` from \p taus, else 2 for a unit
     *        `v`; an empty one the identity) into `I − V T Vᴴ = H_{k0} H_{k0+1} … H_{k1-1}`, `T` as LAPACK's `dlarft`
     *        forms it.
     */
    template <typename T>
    [[nodiscard]] reflector_block<T> gather_reflectors(const std::vector<buffer<T>>&      reflectors,
                                                       std::size_t                        k0,
                                                       std::size_t                        k1,
                                                       std::size_t                        rows,
                                                       const std::vector<T>*              taus = nullptr)
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
        const T tau {taus == nullptr ? static_cast<T>(2) : (*taus)[k0 + j]};
        for (std::size_t i {0}; i < j; ++i) { // T(0…j-1, j) = −τ_j T(0…j-1, 0…j-1) z, z = V(:, 0…j-1)ᴴ v_j
          T sum {};
          for (std::size_t p {i}; p < j; ++p) {
            sum += block.t[(i * b) + p] * gram[(p * b) + j];
          }
          block.t[(i * b) + j] = -tau * sum;
        }
        block.t[(j * b) + j] = tau;
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
     *        on `Aᵀ`'s columns `first…` and rows `from…to-1` (with the \p adjoint `(I − V Tᴴ Vᴴ) A`, `conj(T)` for
     *        `Tᵀ`). Its three products have `to − from` rows, which the
     *        matrix product kernel runs well and on every core; \ref apply_reflector_block's first has the
     *        block's few (a reflector block's 32 rows by 1000 deep: 54 GFLOP/s on twenty x86-64 threads).
     */
    template <typename T>
    void apply_transposed_reflector_block(const reflector_block<T>& block,
                                          buffer<T>&                at,
                                          std::size_t               cols,
                                          std::size_t               from,
                                          std::size_t               to,
                                          bool                      adjoint = false)
    {
      const std::size_t b            {block.width};
      const std::size_t h            {block.height};
      const std::size_t r            {to - from};
      const auto        v_conjugate  {std::make_unique_for_overwrite<T[]>(h * b)}; // conj(V), h × b, written whole
      const auto        minus_v_t    {std::make_unique_for_overwrite<T[]>(b * h)}; // −Vᵀ, b × h
      const auto        t_transposed {std::make_unique_for_overwrite<T[]>(b * b)}; // Tᵀ, or conj(T)
      for (std::size_t row {0}; row < h; ++row) {
        for (std::size_t j {0}; j < b; ++j) {
          v_conjugate[(row * b) + j] = conj_of(block.v[(row * b) + j]);
          minus_v_t[(j * h) + row]   = -block.v[(row * b) + j];
        }
      }
      for (std::size_t i {0}; i < b; ++i) {
        for (std::size_t p {0}; p < b; ++p) {
          t_transposed[(p * b) + i] = adjoint ? conj_of(block.t[(p * b) + i]) : block.t[(i * b) + p];
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

    //! \brief `Σ conj(x[i]) · y[i]` of complex numbers (\ref conjugate_dot_product).
    template <typename T>
    [[nodiscard]] T conjugate_dot(const T*    x,
                                  const T*    y,
                                  std::size_t count)
    {
      return conjugate_dot_product(x, y, count);
    }

    /*!
     * \brief The reflector LAPACK's `dlarfg` (`zlarfg`) forms of \p w, in place: `H = I − τ w wᴴ` with `w₀ = 1` and
     *        `Hᴴ x = β e₁`, `β` real, of the sign opposite to `Re x₀`. Exact inputs stay exact where LAPACK's do (`e₂`
     *        reflected onto `e₁`: `w = (1, 1)`, `τ = 1`, where a unit `v` holds `1/√2`, rounded). For an `x` with
     *        something below `x₀`, or `x₀` complex (the identity otherwise, which the caller keeps).
     * \return `β` (as `T`, its imaginary part 0) and `τ`.
     */
    template <typename T>
    [[nodiscard]] std::pair<T, T> larfg_in_place(buffer<T>& w)
    {
      const carried_t<T> factor {rescale_tiny(w)}; // w, τ and the signs the same at any scale; β scaled back
      const T            alpha  {w[0]};
      const carried_t<T> beta   {-std::copysign(scaled_norm(w), detail::real_part(alpha))};
      const T            tau    {(detail::from_carried<T>(beta) - alpha) / detail::from_carried<T>(beta)};
      const T            scale  {T {1} / (alpha - detail::from_carried<T>(beta))}; // |α − β| ≥ ‖x‖: no cancellation
      w[0] = T {1};
      for (std::size_t r {1}; r < w.size(); ++r) {
        w[r] *= scale;
      }
      return {detail::from_carried<T>(beta / factor), tau};
    }

    /*!
     * \brief Applies `I − s v vᴴ` from the left (\p s 2 for a unit `v`) to the columns `from…to-1` of the
     *        column-major \p panel (\p height rows per column), on its rows `first…first+|v|-1`: each column a dot
     *        product with `v`, then `v` scaled out of it, both along the column at unit stride.
     */
    template <typename T>
    void reflect_columns(T*                    panel,
                         std::size_t           height,
                         const buffer<T>&      v,
                         std::size_t           first,
                         std::size_t           from,
                         std::size_t           to,
                         const T&              s = T {2})
    {
      for (std::size_t j {from}; j < to; ++j) {
        T* const column {panel + (j * height) + first};
        const T  scaled {s * conjugate_dot(v.data(), column, v.size())};
        subtract_product(column, scaled, v.data(), v.size());
      }
    }

    //! \brief The reflectors of one panel of the blocked QR.
    inline constexpr std::size_t qr_block {32};

    /*!
     * \brief Householder reflections of the columns `k0…k1-1` of a column-major matrix, rows from `k0`, one by one, in
     *        place, into `reflectors[k0…k1-1]` and \p taus: \p panel holds column `k0`, \p ld elements per column from
     *        row \p top (0 in the whole matrix, `k0` in a copy of the panel). A reflector meets each column at unit
     *        stride. Recursive halves, the left's reflectors applied to the right at once, took longer at these widths
     *        (32 columns).
     *
     * Each as LAPACK's `dlarfg` (`zlarfg`) forms it: `H = I − τ w wᴴ`, `w₀ = 1`, with `Hᴴ x = β e₁` for a real `β` of
     * the sign opposite to `Re x₀`; the identity (`τ = 0`, an empty reflector) where nothing lies below the diagonal
     * and `x₀` is real. Exact inputs stay exact where LAPACK's do (`e₃` reflected onto `e₁`: `w = (1, 0, 1)`, `τ = 1`;
     * a unit `v` there holds `1/√2`, rounded), and the factors are LAPACK's to rounding. `β` goes on the diagonal and
     * `w₁…` below it, where `dgeqrf` leaves them.
     */
    template <typename T>
    void factor_qr_panel(T*                           panel,
                         std::size_t                  ld,
                         std::size_t                  top,
                         std::vector<buffer<T>>&      reflectors,
                         std::vector<T>&              taus,
                         std::size_t                  k0,
                         std::size_t                  k1)
    {
      for (std::size_t k {k0}; k < k1; ++k) {
        const std::size_t  c      {k - k0};
        T* const           x      {panel + (c * ld) + (k - top)};
        buffer<T>          w(x, x + (ld + top - k));
        const T            alpha  {w[0]};
        if (scaled_norm(w.data() + 1, w.size() - 1) == 0 && alpha == detail::from_carried<T>(detail::real_part(alpha))) {
          continue; // the identity: nothing below the diagonal, and the diagonal real
        }
        const auto [beta, tau] {larfg_in_place(w)};
        reflect_columns(panel, ld, w, k - top, c + 1, k1 - k0, conj_of(tau)); // Hᴴ on the panel's columns past k
        x[0] = beta;
        std::copy(w.begin() + 1, w.end(), x + 1);
        reflectors[k] = std::move(w);
        taus[k]       = tau;
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
     * \brief The fewest threads for which \ref reflect_in_columns looks ahead, a panel factorised while the threads
     *        update the columns past it: on two, one thread would update the rest alone (ten AVX-512 cores, order
     *        1024: 29.4 ms as it is, 36.3 looking ahead), from three a panel no longer waits for the update (four
     *        threads 22.1 → 16.5 ms, eight 19.8 → 13.9).
     */
    inline constexpr std::size_t qr_look_ahead_threads {3};

    //! \brief The fewest columns a task of \ref reflect_in_columns' look-ahead updates: each task repacks the block, which
    //!        narrow slices pay for too often (fixed slices of 64 columns on one thread, order 1024: 41 → 61 ms; with a
    //!        slice per thread, a least of 32 or of 64 alike).
    inline constexpr std::size_t qr_update_slice {32};

    /*!
     * \brief The blocks of reflectors of a Householder QR of the column-major \p columns (\p rows per column, \p cols
     *        columns), factorised in place as LAPACK's `dgeqrf` leaves it: `R` on and above the diagonal, each
     *        reflector's `w₁…` below it. The columns right of a panel take its block as `(I − V T Vᴴ)ᴴ`, held transposed
     *        (\ref apply_transposed_reflector_block): the products then have as many rows as there are columns to
     *        update, where the row-major form's first product has the block's 32 (one x86-64 core: 39 ms against 46 for
     *        the updates of order 1000). From \ref qr_look_ahead_threads threads, the next panel is updated and
     *        factorised by one task while the others update the columns past it, a slice each. Where \p reflectors is false
     *        (`R` alone asked), a block no update reads is not gathered.
     */
    template <typename T>
    [[nodiscard]] std::vector<reflector_block<T>> reflect_in_columns(buffer<T>&  columns,
                                                                     std::size_t rows,
                                                                     std::size_t cols,
                                                                     bool        reflectors = true)
    {
      const std::size_t               depth         {std::min(rows, cols)};
      std::vector<buffer<T>>          reflections(depth); // w_k on rows k…m-1; empty for the identity
      std::vector<T>                  taus(depth, T {});
      std::vector<reflector_block<T>> blocks;
      if (depth == 0) {
        return blocks;
      }
      factor_qr_panel(columns.data(), rows, 0, reflections, taus, 0, std::min(qr_block, depth));
      if (reflectors || std::min(qr_block, depth) < cols) {          // a block no update reads, where R alone is asked: not gathered
        blocks.push_back(gather_reflectors(reflections, 0, std::min(qr_block, depth), rows, &taus));
      }
      for (std::size_t k1 {std::min(qr_block, depth)}; k1 < cols;) { // the panels left of k1 factorised, the last not applied
        if (k1 >= depth) { // a wide matrix: its last block on the columns past every panel
          apply_transposed_reflector_block(blocks.back(), columns, rows, k1, cols, true);
          break;
        }
        const std::size_t n1      {std::min(k1 + qr_block, depth)}; // the next panel: columns k1…n1-1
        const std::size_t threads {max_threads()};
        if (threads < qr_look_ahead_threads) { // right of the panel: H_{k1-1} … H_{k0} A = (I − V T Vᴴ)ᴴ A, then the next
          apply_transposed_reflector_block(blocks.back(), columns, rows, k1, cols, true);
          factor_qr_panel(columns.data() + (k1 * rows), rows, 0, reflections, taus, k1, n1);
          if (reflectors || n1 < cols) {
            blocks.push_back(gather_reflectors(reflections, k1, n1, rows, &taus));
          }
          k1 = n1;
          continue;
        }
        // Look-ahead: one task updates the next panel's columns and factorises them while the others update the rest,
        // a slice of columns each (the same products on fewer columns: the same bits).
        const std::size_t         rest   {cols - n1};
        const std::size_t         slice  {std::max(qr_update_slice, (rest + threads - 2) / (threads - 1))};
        const std::size_t         slices {(rest + slice - 1) / slice};
        const reflector_block<T>& block  {blocks.back()};
        reflector_block<T>        next   {};
        parallel_for(1 + slices, rows * slice * qr_block / 4, [&](std::size_t task) {
                       if (task == 0) {
                         apply_transposed_reflector_block(block, columns, rows, k1, n1, true);
                         factor_qr_panel(columns.data() + (k1 * rows), rows, 0, reflections, taus, k1, n1);
                         if (reflectors || n1 < cols) {
                           next = gather_reflectors(reflections, k1, n1, rows, &taus);
                         }
                         return;
                       }
                       const std::size_t from {n1 + ((task - 1) * slice)};
                       apply_transposed_reflector_block(block, columns, rows, from, std::min(from + slice, cols), true);
                     });
        if (reflectors || n1 < cols) {
          blocks.push_back(std::move(next));
        }
        k1 = n1;
      }
      return blocks;
    }

    //! \brief The side of the tiles in which the QR transposes (\ref transpose_into): a tile's rows and columns both
    //!        stay in the cache, where a whole column read across rows misses on every element.
    inline constexpr std::size_t qr_transpose_tile {32};

    //! \brief The row-major `rows × cols` \p from into \p to as its transpose, `cols × rows` row-major, by tiles,
    //!        bands of them on the threads (a pass bound by the memory, its bounds captured by value).
    template <typename T>
    void transpose_into(const T*    from,
                        std::size_t rows,
                        std::size_t cols,
                        T*          to)
    {
      parallel_for((rows + qr_transpose_tile - 1) / qr_transpose_tile, memory_element_weight * qr_transpose_tile * cols, [from, rows, cols, to](std::size_t band) {
                     const std::size_t i0 {band * qr_transpose_tile};
                     for (std::size_t j0 {0}; j0 < cols; j0 += qr_transpose_tile) {
                       for (std::size_t j {j0}; j < std::min(j0 + qr_transpose_tile, cols); ++j) {
                         for (std::size_t i {i0}; i < std::min(i0 + qr_transpose_tile, rows); ++i) {
                           to[(j * rows) + i] = from[(i * cols) + j];
                         }
                       }
                     }
                   });
    }

    /*!
     * \brief \ref factor_qr_panel on the columns `k0…k1-1` of the row-major \p a, rows from `k0`, in a column-major copy
     *        of the panel, copied back. A function of its own: written into \ref reflect_in_rows' loop, the same
     *        instructions took 5 % more cycles at order 128 (clang, arm64).
     */
    template <typename T>
    void factor_row_major_panel(buffer<T>&              a,
                                std::size_t             rows,
                                std::size_t             cols,
                                std::vector<buffer<T>>& reflectors,
                                std::vector<T>&         taus,
                                std::size_t             k0,
                                std::size_t             k1)
    {
      const std::size_t height                  {rows - k0};
      const std::size_t width                   {k1 - k0};
      buffer<T>         panel(height * width, T {});
      for (std::size_t i {0}; i < height; ++i) {
        for (std::size_t j {0}; j < width; ++j) {
          panel[(j * height) + i] = a[((k0 + i) * cols) + k0 + j];
        }
      }
      factor_qr_panel(panel.data(), height, k0, reflectors, taus, k0, k1);
      for (std::size_t i {0}; i < height; ++i) {
        for (std::size_t j {0}; j < width; ++j) {
          a[((k0 + i) * cols) + k0 + j] = panel[(j * height) + i];
        }
      }
    }

    /*!
     * \brief The blocks of reflectors of a Householder QR of the row-major \p a (\p rows × \p cols), factorised in place
     *        as \ref reflect_in_columns leaves the column-major one (`R` on and above the diagonal, each reflector's `w₁…`
     *        below it): each panel in a column-major copy, the columns right of it updated by \ref apply_reflector_block,
     *        whose projection reads `A` as the product's right operand. The AMX tile reads that in place, and NEON's packs
     *        it faster than the column-major form's left one (arm64, order 512: 6.3 ms against 7.2 under clang, 16.4
     *        against 17.9 under GCC). Where \p r is given, it is `R` (`min(rows, cols) × cols` row-major, zeros below
     *        the diagonal).
     */
    template <typename T>
    [[nodiscard]] std::vector<reflector_block<T>> reflect_in_rows(buffer<T>&  a,
                                                                  std::size_t rows,
                                                                  std::size_t cols,
                                                                  buffer<T>*  r          = nullptr,
                                                                  bool        reflectors = true)
    {
      const std::size_t      depth         {std::min(rows, cols)};
      std::vector<buffer<T>> reflections(depth); // w_k on rows k…m-1; empty for the identity
      std::vector<T>         taus(depth, T {});
      if (r != nullptr) {
        *r = buffer<T>(depth * cols, T {}); // constructed, not assigned: the generic fill of assign took 4 % of qr at order 128
      }
      std::vector<reflector_block<T>> blocks;
      if (depth > 0) {
        factor_row_major_panel(a, rows, cols, reflections, taus, 0, std::min(qr_block, depth));
        if (reflectors || std::min(qr_block, depth) < cols) { // as reflect_in_columns
          blocks.push_back(gather_reflectors(reflections, 0, std::min(qr_block, depth), rows, &taus));
        }
      }
      for (std::size_t k1 {std::min(qr_block, depth)}; depth > 0 && k1 < cols;) { // as reflect_in_columns (none if empty)
        if (k1 >= depth) {
          apply_reflector_block(blocks.back(), true, a, cols, k1, cols);
          break;
        }
        const std::size_t n1      {std::min(k1 + qr_block, depth)};
        const std::size_t threads {max_threads()};
        if (threads < qr_look_ahead_threads) {
          apply_reflector_block(blocks.back(), true, a, cols, k1, cols);
          factor_row_major_panel(a, rows, cols, reflections, taus, k1, n1);
          if (reflectors || n1 < cols) {
            blocks.push_back(gather_reflectors(reflections, k1, n1, rows, &taus));
          }
          k1 = n1;
          continue;
        }
        const std::size_t         rest   {cols - n1};
        const std::size_t         slice  {std::max(qr_update_slice, (rest + threads - 2) / (threads - 1))};
        const std::size_t         slices {(rest + slice - 1) / slice};
        const reflector_block<T>& block  {blocks.back()};
        reflector_block<T>        next   {};
        parallel_for(1 + slices, rows * slice * qr_block / 4, [&](std::size_t task) {
                       if (task == 0) {
                         apply_reflector_block(block, true, a, cols, k1, n1);
                         factor_row_major_panel(a, rows, cols, reflections, taus, k1, n1);
                         if (reflectors || n1 < cols) {
                           next = gather_reflectors(reflections, k1, n1, rows, &taus);
                         }
                         return;
                       }
                       const std::size_t from {n1 + ((task - 1) * slice)};
                       apply_reflector_block(block, true, a, cols, from, std::min(from + slice, cols));
                     });
        if (reflectors || n1 < cols) {
          blocks.push_back(std::move(next));
        }
        k1 = n1;
      }
      for (std::size_t i {0}; r != nullptr && i < depth; ++i) {
        for (std::size_t j {i}; j < cols; ++j) {
          (*r)[(i * cols) + j] = a[(i * cols) + j];
        }
      }
      return blocks;
    }

    /*!
     * \brief The reflections of \ref householder_qr, without forming `Q`: `A = Q R` with `Q` the product of the
     *        blocks' reflectors applied to the first `k = min(rows, cols)` columns of the identity, `R`'s diagonal
     *        not yet made real (a product with `Qᴴ`, or with `Q` of a block padded with zeros, applies the blocks
     *        instead). \p a is row-major. On the x86-64 units it is transposed in by tiles and factorised column-major
     *        (\ref reflect_in_columns, whose updates' products run faster there: one AVX2 core, order 1000, 39 ms against
     *        46), elsewhere in place (\ref reflect_in_rows). Where \p reflectors is false, `R` alone is asked: a block no
     *        update reads is not gathered, and the blocks are not all there.
     */
    template <typename T>
    [[nodiscard]] qr_reflections<T> householder_reflect(buffer<T>   a,
                                                        std::size_t rows,
                                                        std::size_t cols,
                                                        bool        reflectors = true)
    {
#ifdef NUMERICAL_X86_COMPILED
      const std::size_t depth {std::min(rows, cols)};
      buffer<T>         columns(rows * cols);
      transpose_into(a.data(), rows, cols, columns.data());
      qr_reflections<T> out {.r = buffer<T>(depth * cols), .blocks = reflect_in_columns(columns, rows, cols, reflectors)};
      for (std::size_t j0 {0}; j0 < cols; j0 += qr_transpose_tile) { // R, row-major: written whole, zeros below
        for (std::size_t i0 {0}; i0 < depth; i0 += qr_transpose_tile) {
          for (std::size_t i {i0}; i < std::min(i0 + qr_transpose_tile, depth); ++i) {
            for (std::size_t j {j0}; j < std::min(j0 + qr_transpose_tile, cols); ++j) {
              out.r[(i * cols) + j] = i <= j ? columns[(j * rows) + i] : T {};
            }
          }
        }
      }
#else
      qr_reflections<T> out;
      out.blocks = reflect_in_rows(a, rows, cols, &out.r, reflectors);
#endif
      return out;
    }

    /*!
     * \brief The columns `[first, first + count)` of `Q = H₀ H₁ …`, the \p blocks applied (the last first) to those of
     *        the identity, into \p out: `rows × count` row-major, or where \p transposed `Qᵀ` row-major (`Q`
     *        column-major). Formed as \ref householder_reflect factorises, so that a `Q` is the same to the bit in either
     *        layout: as the rows of `Qᵀ` on the x86-64 units (\ref apply_transposed_reflector_block), row-major elsewhere
     *        (\ref apply_reflector_block), then transposed where the other layout is asked. The columns of the identity
     *        left of a block's rows are left as they are.
     */
    template <typename T>
    void form_q_into(buffer<T>&                             out,
                     const std::vector<reflector_block<T>>& blocks,
                     std::size_t                            rows,
                     std::size_t                            first,
                     std::size_t                            count,
                     bool                                   transposed)
    {
#ifdef NUMERICAL_X86_COMPILED
      buffer<T>  formed;
      buffer<T>& qt {transposed ? out : formed};
      qt = buffer<T>(count * rows, T {});
      for (std::size_t c {0}; c < count; ++c) {
        qt[(c * rows) + first + c] = T {1};
      }
      for (std::size_t b {blocks.size()}; b-- > 0;) {
        apply_transposed_reflector_block(blocks[b], qt, rows, std::max(first, blocks[b].first) - first, count);
      }
      if (!transposed) {
        out.resize(rows * count);
        transpose_into(qt.data(), count, rows, out.data());
      }
#else
      buffer<T>  formed;
      buffer<T>& q {transposed ? formed : out};
      q = buffer<T>(rows * count, T {});
      for (std::size_t c {0}; c < count; ++c) {
        q[((first + c) * count) + c] = T {1};
      }
      for (std::size_t b {blocks.size()}; b-- > 0;) {
        apply_reflector_block(blocks[b], false, q, count, std::max(first, blocks[b].first) - first, count);
      }
      if (transposed) {
        out.resize(count * rows);
        transpose_into(q.data(), rows, count, out.data());
      }
#endif
    }

    /*!
     * \brief `P ← Q P` for the `rows × count` row-major \p p, `Q = H₀ H₁ …` the product of the \p blocks' reflectors, the
     *        last applied first: a basis taken through `Q` without forming `Q`, as LAPACK's `dormqr` applies it where
     *        `dorgqr` forms it. As \ref form_q_into applies the blocks: on the x86-64 units to the rows of `Pᵀ`
     *        (\ref apply_transposed_reflector_block), transposed in and out, elsewhere in place (\ref apply_reflector_block).
     */
    template <typename T>
    void apply_q(buffer<T>&                             p,
                 const std::vector<reflector_block<T>>& blocks,
                 [[maybe_unused]] std::size_t           rows,
                 std::size_t                            count)
    {
#ifdef NUMERICAL_X86_COMPILED
      buffer<T> pt(count * rows);
      transpose_into(p.data(), rows, count, pt.data());
      for (std::size_t b {blocks.size()}; b-- > 0;) {
        apply_transposed_reflector_block(blocks[b], pt, rows, 0, count);
      }
      transpose_into(pt.data(), count, rows, p.data());
#else
      for (std::size_t b {blocks.size()}; b-- > 0;) {
        apply_reflector_block(blocks[b], false, p, count, 0, count);
      }
#endif
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
     * \param[in] a        The matrix, row-major.
     * \param[in] rows     Its row count.
     * \param[in] cols     Its column count.
     * \param[in] complete Whether `Q` is the whole `rows × rows` product of the reflections, its columns past `k` an
     *                     orthonormal basis of the complement of `A`'s column space (`numpy.linalg.qr(mode="complete")`).
     * \return `Q` and `R`.
     */
    template <typename T>
    [[nodiscard]] thin_qr<T> householder_qr(buffer<T>      a,
                                            std::size_t    rows,
                                            std::size_t    cols,
                                            bool           complete = false)
    {
      const std::size_t depth   {std::min(rows, cols)};
      const std::size_t width   {complete ? rows : depth};
      qr_reflections<T> factors {householder_reflect(std::move(a), rows, cols)};
      // Q = H₀ H₁ … applied to the columns [first, first + count) of I, each column's own product: the first `depth`
      // formed alone whether Q is complete or not, so that the complete Q's are the reduced one's to the bit (formed
      // together, the products' tiles differ, and a fused multiply-add may round a column otherwise).
      const auto formed {[&](std::size_t first, std::size_t count) {
                           buffer<T> columns;
                           form_q_into(columns, factors.blocks, rows, first, count, false);
                           return columns;
                         }};
      thin_qr<T> out {.q = formed(0, depth), .r = std::move(factors.r)};
      if (width > depth) {
        const buffer<T> rest                  {formed(depth, width - depth)};
        buffer<T>       whole(rows * width, T {});
        for (std::size_t i {0}; i < rows; ++i) {
          std::copy_n(out.q.data() + (i * depth), depth, whole.data() + (i * width));
          std::copy_n(rest.data() + (i * (width - depth)), width - depth, whole.data() + (i * width) + depth);
        }
        out.q = std::move(whole);
      }
      for (std::size_t k {0}; k < depth; ++k) {               // R's diagonal real and non-negative: rotate row k, column k
        const T phase {phase_of(out.r[(k * cols) + k])};
        for (std::size_t j {k}; j < cols; ++j) {
          out.r[(k * cols) + j] *= conj_of(phase);
        }
        for (std::size_t i {0}; i < rows; ++i) {
          out.q[(i * width) + k] *= phase;
        }
      }
      return out;
    }

    /*!
     * \brief \ref householder_qr's `R` alone, `k × cols`, `k = min(rows, cols)`: the same reflections and the same
     *        rotation of each row by its diagonal's phase, so the same bits, `Q` never formed (`numpy.linalg.qr`'s
     *        `mode="r"`; forming `Q` is about as much work again).
     * \param[in] a    The matrix, row-major.
     * \param[in] rows Its row count.
     * \param[in] cols Its column count.
     * \return `R`, row-major, its diagonal real and non-negative.
     */
    template <typename T>
    [[nodiscard]] buffer<T> householder_r(buffer<T>   a,
                                          std::size_t rows,
                                          std::size_t cols)
    {
      const std::size_t depth {std::min(rows, cols)};
      buffer<T>         r     {householder_reflect(std::move(a), rows, cols, false).r};
      for (std::size_t k {0}; k < depth; ++k) { // as householder_qr rotates R's row k
        const T phase {phase_of(r[(k * cols) + k])};
        for (std::size_t j {k}; j < cols; ++j) {
          r[(k * cols) + j] *= conj_of(phase);
        }
      }
      return r;
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
   * \brief The magnitude at or under which a singular value counts as zero, in a rank (\ref matrix_rank), a
   *        pseudo-inverse (\ref pinv) or a least-squares solution (\ref lstsq): by default NumPy's, the largest
   *        singular value times the machine epsilon times the larger dimension; \ref relative to the largest, as
   *        NumPy's `rcond` and `rtol`; or \ref absolute, as `matrix_rank`'s `tol`.
   */
  class rank_cutoff
  {
  public:

    //! \brief NumPy's default: the largest singular value times the machine epsilon times the larger dimension.
    constexpr rank_cutoff() noexcept = default;

    /*!
     * \brief The largest singular value times \p ratio.
     * \throws std::invalid_argument If \p ratio is negative or NaN.
     */
    [[nodiscard]] static rank_cutoff relative(double ratio)
    {
      return rank_cutoff {checked(ratio), false};
    }

    /*!
     * \brief \p magnitude itself, whatever the largest singular value.
     * \throws std::invalid_argument If \p magnitude is negative or NaN.
     */
    [[nodiscard]] static rank_cutoff absolute(double magnitude)
    {
      return rank_cutoff {checked(magnitude), true};
    }

    //! \brief The cutoff for a `rows × cols` matrix whose largest singular value is \p largest.
    [[nodiscard]] double magnitude(double      largest,
                                   std::size_t rows,
                                   std::size_t cols) const noexcept
    {
      if (absolute_) {
        return value_;
      }
      return value_ < 0.0 ? detail::rank_tolerance(largest, rows, cols) : largest * value_;
    }

    //! \brief The same cutoff for the matrix divided by `2^exponent`.
    [[nodiscard]] rank_cutoff scaled(int exponent) const noexcept
    {
      return absolute_ ? rank_cutoff {std::ldexp(value_, -exponent), true} : *this;
    }

  private:

    constexpr rank_cutoff(double value,
                          bool   absolute) noexcept
      : value_(value), absolute_(absolute)
    {}

    [[nodiscard]] static double checked(double value)
    {
      if (!(value >= 0.0)) {
        throw std::invalid_argument("rank cutoff: a tolerance is non-negative");
      }
      return value;
    }

    double value_    {-1.0};  //!< The ratio or the magnitude; negative for NumPy's default.
    bool   absolute_ {false}; //!< Whether \ref value_ is a magnitude.
  };

  //! \brief The factors a \ref numerical::qr_decomposition keeps, as `numpy.linalg.qr`'s modes name them.
  enum class qr_mode
  {
    reduced, //!< `Q` is `m × k`, `R` `k × n`, `k = min(m, n)`: NumPy's default.
    complete //!< `Q` is `m × m`, its columns past `k` a basis of the complement of `A`'s column space; `R` is `m × n`.
  };

  /*!
   * \brief A QR factorisation `A = Q R` of an `m × n` matrix, as NumPy's `qr`, reduced by default.
   * \tparam T The scalar element type (a field: `double`, `std::complex<double>`, …).
   *
   * With `k = min(m, n)`, `Q` is `m × k` with orthonormal columns (`Qᴴ Q = I`) and `R` is `k × n` upper
   * trapezoidal (triangular when `m ≥ n`) with a real, non-negative diagonal; complete (\ref qr_mode::complete),
   * `Q` is `m × m` and `R` `m × n`, its rows past `k` zero. Computed by Householder
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
     * \param[in] mode \ref qr_mode::reduced (`Q` `m × k`) or \ref qr_mode::complete (`Q` `m × m`).
     * \throws non_finite_error If an entry is infinite or NaN.
     */
    explicit qr_decomposition(const matrix<T>& a,
                              qr_mode          mode = qr_mode::reduced)
      : qr_decomposition(detail::parallel_copy_of(a),
                         mode)
    {}

    /*!
     * \brief Factorises \p a in its own storage, an expiring matrix's elements taken over: no copy of them.
     * \param[in] a The matrix, any shape.
     * \param[in] mode \ref qr_mode::reduced (`Q` `m × k`) or \ref qr_mode::complete (`Q` `m × m`).
     * \throws non_finite_error If an entry is infinite or NaN.
     */
    explicit qr_decomposition(matrix<T>&& a,
                              qr_mode     mode = qr_mode::reduced)
      : rows_(a.rows()), cols_(a.cols()), depth_(std::min(rows_, cols_)),
        width_(mode == qr_mode::complete ? rows_ : depth_)
    {
      detail::require_finite(a, "QR decomposition");
      detail::thin_qr<T> factors {detail::householder_qr(std::move(a).elements(), rows_, cols_, mode == qr_mode::complete)};
      q_ = std::move(factors.q);
      r_ = std::move(factors.r);
      r_.resize(width_ * cols_, T {}); // complete: R's rows past k are zero
    }

    //! \brief The factor `Q` (orthonormal columns): `m × k`, `k = min(m, n)`, or `m × m` complete.
    [[nodiscard]] matrix<T> q() const&
    {
      return matrix<T> {rows_, width_, q_};
    }

    //! \brief `Q`, moved out of an expiring decomposition (\ref r stays).
    [[nodiscard]] matrix<T> q() &&
    {
      return matrix<T> {rows_, width_, std::move(q_)};
    }

    //! \brief The upper-trapezoidal factor `R`: `k × n`, `k = min(m, n)`, or `m × n` complete.
    [[nodiscard]] matrix<T> r() const&
    {
      return matrix<T> {width_, cols_, r_};
    }

    //! \brief `R`, moved out of an expiring decomposition (\ref q stays).
    [[nodiscard]] matrix<T> r() &&
    {
      return matrix<T> {width_, cols_, std::move(r_)};
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
        largest = std::max(largest, static_cast<double>(detail::real_part(r_[(i * cols_) + i])));
      }
      const double   tolerance  {detail::rank_tolerance(largest, rows_, cols_)};
      buffer<T>      y(cols_, T {}); // y = Qᴴ b
      for (std::size_t i {0}; i < cols_; ++i) {
        if (detail::real_part(r_[(i * cols_) + i]) <= tolerance) {
          throw singular_matrix_error("QR solve: the matrix is rank-deficient (lstsq gives its minimum-norm solution)");
        }
        T sum {};
        for (std::size_t k {0}; k < rows_; ++k) {
          sum += detail::conj_of(q_[(k * width_) + i]) * b[k];
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

    /*!
     * \brief The least-squares solutions of `A X ≈ B` (`X = R⁻¹ Qᴴ B`) for a full-rank `A`, every column of \p b at
     *        once: `Qᴴ B` through the matrix product kernel, then the back substitution by rows of `R` over all the
     *        columns (each column \ref solve's to rounding: the product sums its terms in another order).
     * \param[in] b The right-hand sides (`m × k`).
     * \return `X` (`n × k`).
     * \throws dimension_error If `b`'s row count differs from the matrix's, or `A` is wide (as \ref solve).
     * \throws singular_matrix_error If `A` is rank-deficient (as \ref solve).
     */
    [[nodiscard]] matrix<T> solve(const matrix<T>& b) const
    {
      if (b.rows() != rows_) {
        throw dimension_error("solve: right-hand side row count differs from the matrix");
      }
      if (rows_ < cols_) {
        throw dimension_error("QR solve: the system is underdetermined (lstsq gives its minimum-norm solution)");
      }
      double largest {0.0};
      for (std::size_t i {0}; i < cols_; ++i) {
        largest = std::max(largest, static_cast<double>(detail::real_part(r_[(i * cols_) + i])));
      }
      const double tolerance {detail::rank_tolerance(largest, rows_, cols_)};
      for (std::size_t i {0}; i < cols_; ++i) {
        if (detail::real_part(r_[(i * cols_) + i]) <= tolerance) {
          throw singular_matrix_error("QR solve: the matrix is rank-deficient (lstsq gives its minimum-norm solution)");
        }
      }
      const std::size_t sides {b.cols()};
      buffer<T>         q_adjoint(cols_ * rows_); // the first n columns of Q, conjugated, as rows
      for (std::size_t k {0}; k < rows_; ++k) {
        for (std::size_t i {0}; i < cols_; ++i) {
          q_adjoint[(i * rows_) + k] = detail::conj_of(q_[(k * width_) + i]);
        }
      }
      buffer<T> x(cols_ * sides); // Y = Qᴴ B, then X in its place
      gemm(cols_, sides, rows_, q_adjoint.data(), rows_, std::to_address(b.begin()), sides, x.data(), sides, detail::c_start::zero);
      for (std::size_t i {cols_}; i-- > 0;) { // back substitution R X = Y, a row of R's terms at a time
        T* const row {x.data() + (i * sides)};
        for (std::size_t j {i + 1}; j < cols_; ++j) {
          const T        r     {r_[(i * cols_) + j]};
          const T* const below {x.data() + (j * sides)};
          for (std::size_t c {0}; c < sides; ++c) {
            row[c] -= r * below[c];
          }
        }
        for (std::size_t c {0}; c < sides; ++c) {
          row[c] /= r_[(i * cols_) + i];
        }
      }
      return matrix<T> {cols_, sides, std::move(x)};
    }

  private:

    std::size_t    rows_;  //!< The number of rows (m).
    std::size_t    cols_;  //!< The number of columns (n).
    std::size_t    depth_; //!< min(m, n).
    std::size_t    width_; //!< Q's columns and R's rows: k, or m complete.
    buffer<T>      q_;     //!< The factor Q, row-major, m × width_.
    buffer<T>      r_;     //!< The upper-trapezoidal factor R, row-major, width_ × n.
  };

  //! \brief QR-factorises an `m × n` matrix. \see qr_decomposition
  template <typename T>
  [[nodiscard]] qr_decomposition<T> qr(const matrix<T>& a)
  {
    return qr_decomposition<T> {a};
  }

  namespace detail {

    //! \brief `Σ x[i] y[i]`, vectorised (\ref dot_product): `xᴴ y` for real numbers.
    [[nodiscard]] inline double adjoint_dot(const double* x,
                                            const double* y,
                                            std::size_t   count)
    {
      return dot_product(x, y, count);
    }

    //! \brief `Σ conj(x[i]) y[i]`, vectorised (\ref conjugate_dot_product): `xᴴ y`.
    [[nodiscard]] inline std::complex<double> adjoint_dot(const std::complex<double>* x,
                                                          const std::complex<double>* y,
                                                          std::size_t                 count)
    {
      return conjugate_dot_product(x, y, count);
    }

    //! \brief The 2-norm of the \p count entries at \p x, scaled by the largest so no square overflows or underflows.
    template <typename T>
    [[nodiscard]] double contiguous_norm(const T*    x,
                                         std::size_t count)
    {
      double largest {0.0};
      for (std::size_t i {0}; i < count; ++i) {
        largest = std::max(largest, std::abs(x[i]));
      }
      if (largest == 0.0) {
        return 0.0;
      }
      double sum {0.0};
      for (std::size_t i {0}; i < count; ++i) {
        const double scaled {std::abs(x[i]) / largest};
        sum += scaled * scaled;
      }
      return largest * std::sqrt(sum);
    }
  } // namespace detail

  /*!
   * \brief A QR factorisation with column pivoting `A P = Q R` of an `m × n` matrix, as `scipy.linalg.qr(a,
   *        pivoting=True)` (LAPACK's `geqp3`): at each step the column of largest remaining norm moved first, so `R`'s
   *        diagonal does not grow in modulus down the diagonal and reveals `A`'s numerical rank; the norms downdated
   *        as LAPACK's `laqp2` downdates them, recomputed where the downdate has lost too much (`√ε`). Householder
   *        reflectors as `larfg` forms them; `R`'s diagonal then made real and non-negative, as \ref qr_decomposition's
   *        (`Q`'s columns taking the phases), where LAPACK's may be negative.
   */
  template <typename T>
  class pivoted_qr_decomposition
  {
  public:

    /*!
     * \brief Factorises \p a: `Q` `m × k` and `R` `k × n` (`k = min(m, n)`), or `Q` `m × m` and `R` `m × n`
     *        (\p mode complete).
     * \throws non_finite_error If an entry is infinite or NaN.
     */
    explicit pivoted_qr_decomposition(const matrix<T>& a,
                                      qr_mode          mode = qr_mode::reduced)
    {
      const std::size_t m {a.rows()};
      const std::size_t n {a.cols()};
      const std::size_t k {std::min(m, n)};
      detail::require_finite(a, "pivoted QR");
      buffer<T> w(m * n); // by columns, each contiguous: the steps' sums, updates and exchanges go down them
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          w[(j * m) + i] = a(i, j);
        }
      }
      buffer<double> current(n);
      buffer<double> reference(n);
      permutation_.resize(n);
      for (std::size_t j {0}; j < n; ++j) {
        permutation_[j] = j;
        current[j]      = detail::contiguous_norm(w.data() + (j * m), m);
        reference[j]    = current[j];
      }
      const double           tolerance {std::sqrt(std::numeric_limits<double>::epsilon())};
      std::vector<buffer<T>> reflectors(k);
      std::vector<T>         taus(k);
      const std::size_t      top   {k > pivoted_qr_crossover ? k - pivoted_qr_crossover : 0};
      std::size_t            start {0};
      while (start < top) {
        start += factor_block(w, m, n, start, std::min(pivoted_qr_block, top - start), current, reference, reflectors, taus);
      }
      for (std::size_t step {start}; step < k; ++step) {
        const std::size_t pivot {static_cast<std::size_t>(std::max_element(current.begin() + static_cast<std::ptrdiff_t>(step), current.end()) - current.begin())};
        T* const          head  {w.data() + (step * m)};
        if (pivot != step) {
          std::swap_ranges(head, head + m, w.data() + (pivot * m));
          std::swap(permutation_[step], permutation_[pivot]);
          std::swap(current[step], current[pivot]);
          std::swap(reference[step], reference[pivot]);
        }
        buffer<T> v(head + step, head + m);
        // The identity where nothing lies below the diagonal and the diagonal is real (a zero column among them), as
        // the QR's panel takes it: larfg would divide by a zero β.
        const T    alpha    {v[0]};
        const bool identity {detail::scaled_norm(v.data() + 1, v.size() - 1) == 0 && alpha == detail::from_carried<T>(detail::real_part(alpha))};
        T          tau      {};
        if (identity) {
          std::fill(v.begin(), v.end(), T {});
          v[0] = T {1};
        }
        else {
          const auto [beta, factor] {detail::larfg_in_place(v)};
          tau        = factor;
          head[step] = beta;
          std::fill(head + step + 1, head + m, T {});
        }
        for (std::size_t j {step + 1}; j < n; ++j) { // Hᴴ = I − τ̄ v vᴴ on the columns right of the pivot
          T* const column {w.data() + (j * m)};
          const T  dot    {detail::adjoint_dot(v.data(), column + step, m - step)};
          detail::subtract_product(column + step, detail::conj_of(tau) * dot, v.data(), m - step);
          if (current[j] != 0.0) { // LAPACK's downdate of the norm by the entry the reflection took into row step
            const double ratio {std::abs(column[step]) / current[j]};
            const double left  {std::max(0.0, (1.0 + ratio) * (1.0 - ratio))};
            const double drift {left * (current[j] / reference[j]) * (current[j] / reference[j])};
            if (drift <= tolerance) {
              current[j]   = detail::contiguous_norm(column + step + 1, m - step - 1);
              reference[j] = current[j];
            }
            else {
              current[j] *= std::sqrt(left);
            }
          }
        }
        reflectors[step] = std::move(v);
        taus[step]       = tau;
      }
      const std::size_t rows          {mode == qr_mode::complete ? m : k};
      buffer<T>         r(rows * n, T {});
      for (std::size_t i {0}; i < rows; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          r[(i * n) + j] = w[(j * m) + i];
        }
      }
      std::vector<detail::reflector_block<T>> blocks; // Q = H₀ H₁ …, formed by blocks as the QR's
      for (std::size_t k0 {0}; k0 < k; k0 += pivoted_qr_block) {
        blocks.push_back(detail::gather_reflectors(reflectors, k0, std::min(k0 + pivoted_qr_block, k), m, &taus));
      }
      buffer<T> q;
      detail::form_q_into(q, blocks, m, 0, rows, false);
      for (std::size_t i {0}; i < k; ++i) { // R's diagonal real and non-negative, Q's column i taking the phase
        const double modulus {std::abs(r[(i * n) + i])};
        if (modulus == 0.0) {
          continue;
        }
        const T phase {r[(i * n) + i] / T {modulus}};
        for (std::size_t j {i}; j < n; ++j) {
          r[(i * n) + j] *= detail::conj_of(phase);
        }
        r[(i * n) + i] = T {modulus};
        for (std::size_t row {0}; row < m; ++row) {
          q[(row * rows) + i] *= phase;
        }
      }
      q_ = matrix<T> {m, rows, std::move(q)};
      r_ = matrix<T> {rows, n, std::move(r)};
    }

    //! \brief `Q`, orthonormal columns: `m × k`, or `m × m` complete.
    [[nodiscard]] const matrix<T>& q() const noexcept
    {
      return q_;
    }

    //! \brief `R`, upper trapezoidal, its diagonal real, non-negative and not increasing: `k × n`, or `m × n`.
    [[nodiscard]] const matrix<T>& r() const noexcept
    {
      return r_;
    }

    //! \brief The column permutation: column `j` of `A P` is column `permutation()[j]` of `A`, as SciPy's `P`.
    [[nodiscard]] const std::vector<std::size_t>& permutation() const noexcept
    {
      return permutation_;
    }

  private:

    static constexpr std::size_t pivoted_qr_block     {32};  //!< `geqp3`'s block of columns (`NB`).
    static constexpr std::size_t pivoted_qr_crossover {128}; //!< The columns `geqp3` leaves to the unblocked steps (`NX`).
    static constexpr std::size_t pivoted_qr_band      {16};  //!< The columns one thread takes of a block's updates.

    /*!
     * \brief Up to \p width steps from column \p start of the column-major `m × n` \p w, as LAPACK's `laqps`: the
     *        reflections kept aside in `F` (`A ← A − V Fᴴ`, `F(j, ·) = τ A(:, j)ᴴ v` less the earlier reflectors'
     *        share), each step's pivot column and pivot row brought up to date alone, the rest of the matrix by one
     *        product at the end; a column whose norm downdate lost too much (`√ε`) ends the block, its norm recomputed
     *        after the product. \return The steps taken.
     */
    std::size_t factor_block(buffer<T>&              w,
                             std::size_t             m,
                             std::size_t             n,
                             std::size_t             start,
                             std::size_t             width,
                             buffer<double>&         current,
                             buffer<double>&         reference,
                             std::vector<buffer<T>>& reflectors,
                             std::vector<T>&         taus)
    {
      const double             tolerance                    {std::sqrt(std::numeric_limits<double>::epsilon())};
      const std::size_t        cols                         {n - start};
      buffer<T>                f(cols * pivoted_qr_block, T {}); // F by rows: F(j, t) at f[j·block + t], j from start
      buffer<T>                coefficients(pivoted_qr_block);
      std::vector<std::size_t> deferred;
      std::size_t              done {0};
      const auto               row_of_f = [&f](std::size_t j) { return f.data() + (j * pivoted_qr_block); };
      while (done < width && deferred.empty()) {
        const std::size_t kk    {done};
        const std::size_t step  {start + kk};
        const std::size_t pivot {static_cast<std::size_t>(std::max_element(current.begin() + static_cast<std::ptrdiff_t>(step), current.end()) - current.begin())};
        T* const          head  {w.data() + (step * m)};
        if (pivot != step) {
          std::swap_ranges(head, head + m, w.data() + (pivot * m));
          std::swap_ranges(row_of_f(kk), row_of_f(kk) + kk, row_of_f(pivot - start));
          std::swap(permutation_[step], permutation_[pivot]);
          std::swap(current[step], current[pivot]);
          std::swap(reference[step], reference[pivot]);
        }
        if (kk > 0) { // the column's share of the earlier reflections: A(step:, step) −= V(step:, ·) F(kk, ·)ᴴ
          for (std::size_t t {0}; t < kk; ++t) {
            coefficients[t] = detail::conj_of(row_of_f(kk)[t]);
          }
          detail::subtract_products(head + step, coefficients.data(), w.data() + (start * m) + step, m, kk, m - step);
        }
        buffer<T>  v(head + step, head + m);
        const T    alpha    {v[0]};
        const bool identity {detail::scaled_norm(v.data() + 1, v.size() - 1) == 0 && alpha == detail::from_carried<T>(detail::real_part(alpha))};
        T          tau      {};
        T          diagonal {alpha};
        if (identity) {
          std::fill(v.begin(), v.end(), T {});
          v[0] = T {1};
        }
        else {
          const auto [beta, factor] {detail::larfg_in_place(v)};
          tau      = factor;
          diagonal = beta;
        }
        std::copy(v.begin(), v.end(), head + step); // v in place (its 1 on the diagonal) while the block lasts
        const auto conjugate_dot = [&v, m, step](const T* column) { // A(step:, j)ᴴ v, vectorised
                                     return detail::adjoint_dot(column + step, v.data(), m - step);
                                   };
        // F(j, kk) = τ A(step:, j)ᴴ v, each column its own dot product: by bands of columns on the threads
        const std::size_t later {cols - (kk + 1)};
        detail::parallel_for((later + pivoted_qr_band - 1) / pivoted_qr_band, (m - step) * pivoted_qr_band, [&](std::size_t band) {
                               const std::size_t first {kk + 1 + (band * pivoted_qr_band)};
                               const std::size_t last {std::min(cols, first + pivoted_qr_band)};
                               for (std::size_t j {first}; j < last; ++j) {
                                 row_of_f(j)[kk] = tau * conjugate_dot(w.data() + ((start + j) * m));
                               }
                             });
        for (std::size_t j {0}; j <= kk; ++j) {
          row_of_f(j)[kk] = T {};
        }
        if (kk > 0) { // less the earlier reflectors' share: F(·, kk) += F(·, 0:kk) (−τ V(step:, 0:kk)ᴴ v)
          for (std::size_t t {0}; t < kk; ++t) {
            coefficients[t] = -tau * conjugate_dot(w.data() + ((start + t) * m));
          }
          for (std::size_t j {0}; j < cols; ++j) {
            T sum {};
            for (std::size_t t {0}; t < kk; ++t) {
              sum += row_of_f(j)[t] * coefficients[t];
            }
            row_of_f(j)[kk] += sum;
          }
        }
        for (std::size_t j {kk + 1}; j < cols; ++j) { // the pivot row: A(step, j) −= A(step, start:step+1) F(j, 0:kk+1)ᴴ
          T sum {};
          for (std::size_t t {0}; t <= kk; ++t) {
            sum += w[((start + t) * m) + step] * detail::conj_of(row_of_f(j)[t]);
          }
          w[((start + j) * m) + step] -= sum;
        }
        for (std::size_t j {step + 1}; j < n; ++j) { // the norms downdated as laqp2's; one that lost too much waits
          if (current[j] != 0.0) {                     // for the product (the blocks end 128 columns before the last)
            const double ratio {std::abs(w[(j * m) + step]) / current[j]};
            const double left  {std::max(0.0, (1.0 + ratio) * (1.0 - ratio))};
            const double drift {left * (current[j] / reference[j]) * (current[j] / reference[j])};
            if (drift <= tolerance) {
              deferred.push_back(j);
            }
            else {
              current[j] *= std::sqrt(left);
            }
          }
        }
        head[step]       = diagonal;
        reflectors[step] = std::move(v);
        taus[step]       = tau;
        ++done;
      }
      const std::size_t below {start + done}; // the rows below the block's, the columns right of it: A −= V Fᴴ
      const std::size_t right {cols - done};  // each column on its own: by bands of columns on the threads
      detail::parallel_for((right + pivoted_qr_band - 1) / pivoted_qr_band, (m - below) * done * pivoted_qr_band, [&](std::size_t band) {
                             std::array<T, pivoted_qr_block> terms {};
                             const std::size_t                first {done + (band * pivoted_qr_band)};
                             const std::size_t                last {std::min(cols, first + pivoted_qr_band)};
                             for (std::size_t j {first}; j < last; ++j) {
                               for (std::size_t t {0}; t < done; ++t) {
                                 terms[t] = detail::conj_of(row_of_f(j)[t]);
                               }
                               detail::subtract_products(w.data() + ((start + j) * m) + below, terms.data(), w.data() + (start * m) + below, m, done, m - below);
                             }
                           });
      for (std::size_t t {0}; t < done; ++t) { // R below the diagonal: the reflectors are kept apart
        T* const column {w.data() + ((start + t) * m)};
        std::fill(column + start + t + 1, column + m, T {});
      }
      for (const std::size_t j : deferred) {
        current[j]   = detail::contiguous_norm(w.data() + (j * m) + below, m - below);
        reference[j] = current[j];
      }
      return done;
    }

    matrix<T>                q_;           //!< Q.
    matrix<T>                r_;           //!< R.
    std::vector<std::size_t> permutation_; //!< P, by its columns' origins.
  };

  /*!
   * \brief The factor `R` of the QR decomposition of an `m × n` matrix, `k × n`, `k = min(m, n)`: \ref qr_decomposition's
   *        `R` to the bit, `Q` never formed.
   * \param[in] a The matrix, any shape, its elements taken over.
   * \return `R`, upper trapezoidal, its diagonal real and non-negative.
   * \throws non_finite_error If an entry is infinite or NaN.
   */
  template <typename T>
  [[nodiscard]] matrix<T> qr_r(matrix<T> a)
  {
    detail::require_finite(a, "QR decomposition");
    const std::size_t rows {a.rows()};
    const std::size_t cols {a.cols()};
    return matrix<T> {std::min(rows, cols), cols, detail::householder_r(std::move(a).elements(), rows, cols)};
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

    /*!
     * \brief `y[i] += conj(x[i]) · f`, each product's statements spelled out as the complex operator's (four part
     *        products rounded apart, `ac − bd` and `ad + bc`, then their sums with `y`): the same bits for finite parts
     *        whose products stay finite, where the operator's rescue of a product whose parts are both NaN never acts,
     *        and a loop the compiler vectorises (the reduction's rows, scaled by their largest magnitude, are such).
     */
    template <typename T>
    void add_conjugate_product(T*          y,
                               const T*    x,
                               const T&    f,
                               std::size_t count)
    {
      using R = typename T::value_type;
      const R c {f.real()};
      const R d {f.imag()};
      for (std::size_t i {0}; i < count; ++i) {
        const R a  {x[i].real()};
        const R b  {-x[i].imag()};
        const R ac {a * c};
        const R bd {b * d};
        const R ad {a * d};
        const R bc {b * c};
        const R re {ac - bd};
        const R im {ad + bc};
        y[i] = T {y[i].real() + re, y[i].imag() + im};
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

#ifdef NUMERICAL_NEON_COMPILED
    /*!
     * \brief \ref hermitian_row of a `std::complex<double>` row in one pass on the NEON units, as `avx2_hermitian_row`
     *        on AVX2: a number a register, `out[j] += conj(row[j]) v[i]` by two fused multiply-adds, and
     *        `Σⱼ<count row[j] v[j]` in two pairs of accumulators (each part product's terms apart, summed in a fixed
     *        order), the row read once for both; the return leaves out the diagonal's term.
     */
    inline std::complex<double> neon_hermitian_row(const double* row,
                                                   const double* v,
                                                   double        vr,
                                                   double        vi,
                                                   double*       out,
                                                   std::size_t   count) noexcept
    {
      const float64x2_t real      {vr, -vr};
      const float64x2_t imag      {vi, vi};
      float64x2_t       same[2]   {vdupq_n_f64(0.0), vdupq_n_f64(0.0)};
      float64x2_t       across[2] {vdupq_n_f64(0.0), vdupq_n_f64(0.0)};
      std::size_t       j         {0};
      for (; j + 2 <= count; j += 2) {
        for (std::size_t l {0}; l < 2; ++l) {
          const float64x2_t r {vld1q_f64(row + (2 * (j + l)))};
          const float64x2_t x {vld1q_f64(v + (2 * (j + l)))};
          const float64x2_t s {vextq_f64(r, r, 1)};
          float64x2_t       o {vld1q_f64(out + (2 * (j + l)))};
          o         = vfmaq_f64(o, real, r);
          o         = vfmaq_f64(o, imag, s);
          same[l]   = vfmaq_f64(same[l], r, x);
          across[l] = vfmaq_f64(across[l], s, x);
          vst1q_f64(out + (2 * (j + l)), o);
        }
      }
      const float64x2_t sum_same   {vaddq_f64(same[0], same[1])};
      const float64x2_t sum_across {vaddq_f64(across[0], across[1])};
      double            re         {vgetq_lane_f64(sum_same, 0) - vgetq_lane_f64(sum_same, 1)};
      double            im         {vgetq_lane_f64(sum_across, 0) + vgetq_lane_f64(sum_across, 1)};
      for (; j < count; ++j) {
        const double rr {row[2 * j]};
        const double ri {row[(2 * j) + 1]};
        out[2 * j]       = std::fma(vi, ri, std::fma(vr, rr, out[2 * j]));
        out[(2 * j) + 1] = std::fma(vi, rr, std::fma(-vr, ri, out[(2 * j) + 1]));
        re               = std::fma(-ri, v[(2 * j) + 1], std::fma(rr, v[2 * j], re));
        im               = std::fma(rr, v[(2 * j) + 1], std::fma(ri, v[2 * j], im));
      }
      return std::complex<double> {re, im};
    }

#endif

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
#ifdef NUMERICAL_NEON_COMPILED
      if (const gemm_path neon_path {gemm_path_current()}; neon_path == gemm_path::neon || neon_path == gemm_path::amx) {
        const auto* const          values {reinterpret_cast<const double*>(v)};   // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
        const auto* const          parts  {reinterpret_cast<const double*>(row)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
        auto* const                into   {reinterpret_cast<double*>(out)};       // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
        const std::complex<double> lower  {neon_hermitian_row(parts, values, v[i].real(), v[i].imag(), into, i)};
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
      // The tasks of the longest rows first: the threads take the indices in order, and the last rows' task, of some
      // 32 m terms where the first's has 528, left one thread working alone at the end.
      parallel_for(tasks, rows_per_task * m * symmetric_row_speedup<T>(), [&](std::size_t index) {
                     const std::size_t task {tasks - 1 - index};
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

    //! \brief The rows from which \ref corrected_hermitian_product shares its coefficients and bands with the threads:
    //!        on six x86-64 AVX2 cores from 256 (below, eigh of order 128 took 13 % longer), on an M1 Pro from 512
    //!        (from 256, eigh of order 512 on eight cores took 3 to 9 % longer).
#ifdef NUMERICAL_X86_COMPILED
    inline constexpr std::size_t corrected_product_least {256};
#else
    inline constexpr std::size_t corrected_product_least {512};
#endif

    //! \brief \ref subtract_products of a real `T`, where the reduction's corrections are those of its type.
    template <typename T>
    requires std::is_floating_point_v<T>
    void subtract_corrections(T*          y,
                              const T*    f,
                              const T*    x,
                              std::size_t ldx,
                              std::size_t terms,
                              std::size_t count)
    {
      subtract_products(y, f, x, ldx, terms, count);
    }

    /*!
     * \brief \ref subtract_products of complex numbers, each term `y[i] −= f · x[i]` spelled out as the complex
     *        operator's statements (four part products rounded apart, `ac − bd` and `ad + bc`, then their differences
     *        with `y`): the same bits for finite parts whose products stay finite, where the operator's rescue of a
     *        product whose parts are both NaN never acts, and loops the compiler vectorises (the reduction's rows,
     *        scaled by their largest magnitude, are such). Complex `double` on the x86-64 units keeps its fused kernel.
     */
    template <split_complex T>
    void subtract_corrections(T*          y,
                              const T*    f,
                              const T*    x,
                              std::size_t ldx,
                              std::size_t terms,
                              std::size_t count)
    {
#ifdef NUMERICAL_X86_COMPILED
      if constexpr (std::is_same_v<T, std::complex<double>>) { // its own fused kernel on the x86-64 units, as fast
        if (const gemm_path path {gemm_path_current()}; path == gemm_path::avx2 || path == gemm_path::avx512) {
          subtract_products(y, f, x, ldx, terms, count);
          return;
        }
      }
#endif
      using R = typename T::value_type;
      for (std::size_t t {0}; t < terms; ++t) {
        const R        c   {f[t].real()};
        const R        d   {f[t].imag()};
        const T* const row {x + (t * ldx)};
        for (std::size_t i {0}; i < count; ++i) {
          const R a  {row[i].real()};
          const R b  {row[i].imag()};
          const R ac {c * a};
          const R bd {d * b};
          const R ad {c * b};
          const R bc {d * a};
          const R re {ac - bd};
          const R im {ad + bc};
          y[i] = T {y[i].real() - re, y[i].imag() - im};
        }
      }
    }

    /*!
     * \brief \ref hermitian_product corrected by a panel's earlier steps: `w = B v − Σₜ fₜ cₜ`, the coefficients
     *        `f[2p] = (c[2p+1] · v)` and `f[2p+1] = (c[2p] · v)` (conjugated) computed beside the product, among its
     *        tasks, and the partials summed and the corrections taken by bands of rows: each element the operations,
     *        in the order, \ref hermitian_product, \ref conjugate_dot and \ref subtract_products give it apart.
     * \param[in]     block        The block's first element; its rows \p n apart.
     * \param[in]     n            The row stride.
     * \param[in]     m            The block's order.
     * \param[in]     v            The vector (\p m elements).
     * \param[out]    w            `B v` corrected (\p m elements).
     * \param[in,out] scratch      \ref hermitian_product's partials, kept by the caller from one step to the next.
     * \param[in]     corrections  The rows `c₀ … c_{terms−1}` (\p ldc apart), over the block's rows.
     * \param[in]     ldc          The distance between those rows.
     * \param[in]     terms        Their count, even (two a step: its `v` and its `w`).
     * \param[out]    factors      The coefficients (\p terms).
     * \param[in,out] next         The panel's next column over the block's rows, which takes the terms of these
     *                              corrections with \p next_factors in the same pass; none past a panel's last step.
     * \param[in]     next_factors Its coefficients of the rows (\p terms).
     */
    template <typename T>
    void corrected_hermitian_product(const T*        block,
                                     std::size_t     n,
                                     std::size_t     m,
                                     const T*        v,
                                     T*              w,
                                     buffer<T>&      scratch,
                                     const T*        corrections,
                                     std::size_t     ldc,
                                     std::size_t     terms,
                                     T*              factors,
                                     T*              next         = nullptr,
                                     const T*        next_factors = nullptr)
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
      // Below corrected_product_least rows the coefficients and the bands run on the caller: woken threads cost more
      // than they take (an eigh of order 128, six x86-64 cores, +13 %).
      const bool        share {m >= corrected_product_least};
      const std::size_t pairs {terms / 2};
      const auto        coefficients = [&](std::size_t p) {
                                         factors[2 * p]       = conjugate_dot(corrections + (((2 * p) + 1) * ldc), v, m);
                                         factors[(2 * p) + 1] = conjugate_dot(corrections + ((2 * p) * ldc), v, m);
                                       };
      parallel_for(tasks + (share ? pairs : 0), rows_per_task * m * symmetric_row_speedup<T>(), [&](std::size_t index) {
                     if (index >= tasks) { // a pair of coefficients: an earlier step's w and v against v
                       coefficients(index - tasks);
                       return;
                     }
                     const std::size_t task {tasks - 1 - index}; // the longest rows first, as hermitian_product
                     const std::size_t first {task * rows_per_task};
                     const std::size_t last {std::min(first + rows_per_task, m)};
                     T*                own {scratch.data() + (task * m)};
                     std::fill(own, own + last, T {});
                     std::size_t i {first};
                     if constexpr (std::is_same_v<T, double>) {
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
      for (std::size_t p {0}; !share && p < pairs; ++p) {
        coefficients(p);
      }
      constexpr std::size_t band {256};
      const auto            bands = [&](std::size_t b) {
                                      const std::size_t first {b * band};
                                      const std::size_t count {std::min(band, m - first)};
                                      std::fill(w + first, w + first + count, T {});
                                      for (std::size_t task {0}; task < tasks; ++task) { // the partials in order, as hermitian_product
                                        const std::size_t reach {std::min((task + 1) * rows_per_task, m)};
                                        if (reach > first) {
                                          add_product(w + first, T {1}, scratch.data() + (task * m) + first, std::min(count, reach - first));
                                        }
                                      }
                                      subtract_corrections(w + first, factors, corrections + first, ldc, terms, count);
                                      if (next != nullptr) { // the next step's column: these earlier steps' terms (next implies an earlier step)
                                        subtract_corrections(next + first, next_factors, corrections + first, ldc, terms, count);
                                      }
                                    };
      if (share) {
        parallel_for((m + band - 1) / band, band * (tasks + terms) * 8, bands);
        return;
      }
      for (std::size_t b {0}; b < (m + band - 1) / band; ++b) {
        bands(b);
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
      // By reflection: row 2p holds step k0 + p's vector v over the rows (zero above), row 2p + 1 its w, so that a
      // step's terms of the panel's earlier ones are one update through the rows (subtract_products), in their order.
      buffer<T> by_row(2 * tridiagonal_block * n);
      buffer<T> columns(tridiagonal_block * n);      // the panel's columns of A, gathered row by row
      buffer<T> factors(2 * tridiagonal_block);      // a step's coefficients of those rows
      buffer<T> next_factors(2 * tridiagonal_block); // the next column's, applied ahead
      bool      column_ahead {false};                // whether the step's column took its earlier terms ahead
      for (std::size_t k0 {0}; k0 + 1 < n; k0 += tridiagonal_block) {
        // A panel of reflections, as LAPACK's dlatrd: the matrix the panel's later steps see is
        // A − V Wᴴ − W Vᴴ, applied to the one column and the one product each step needs; the whole trailing
        // block takes it after the panel, through the matrix product kernel.
        const std::size_t k1                 {std::min(k0 + tridiagonal_block, n - 1)};
        const std::size_t width              {k1 - k0};
        std::fill_n(by_row.data(), 2 * width * n, T {});
        for (std::size_t i {k0}; i < n; ++i) { // column k from row k down, each row's part read once
          for (std::size_t k {k0}; k < std::min(k1, i + 1); ++k) {
            columns[((k - k0) * n) + i] = a[(i * n) + k];
          }
        }
        for (std::size_t k {k0}; k < k1; ++k) {
          const std::size_t j               {k - k0};
          const std::size_t m               {n - k - 1};
          T* const          column          {columns.data() + (j * n) + k}; // column k from the diagonal down, the panel's earlier steps applied
          // The steps before the last were applied during it (column_ahead); the last's two terms, or all, here.
          const std::size_t from            {column_ahead ? j - 1 : 0};
          for (std::size_t p {from}; p < j; ++p) {
            factors[2 * (p - from)]       = conj_of(by_row[(((2 * p) + 1) * n) + k]); // v's row by w's element
            factors[(2 * (p - from)) + 1] = conj_of(by_row[((2 * p) * n) + k]);
          }
          subtract_corrections(column, factors.data(), by_row.data() + ((2 * from) * n) + k, n, 2 * (j - from), m + 1);
          column_ahead   = false;
          a[(k * n) + k] = column[0];
          buffer<T>          v(column + 1, column + m + 1); // column k below the diagonal
          const carried_t<T> below {m > 1 ? scaled_norm(v.data() + 1, m - 1) : carried_t<T> {0}};
          if (below == 0.0) {
            sub[k] = v[0]; // already tridiagonal in this column: nothing to reflect
            continue;
          }
          const carried_t<T> up    {rescale_tiny(v)};                                  // the reflector is the same at any scale; α is scaled back
          const T            alpha {-phase_of(v[0]) * static_cast<T>(scaled_norm(v))}; // the sign that avoids cancellation
          sub[k]  = alpha / up;
          v[0]   -= alpha;
          const carried_t<T> length {scaled_norm(v)};                                  // real: each part divided, where a complex division scaled and divided both
          for (T& element : v) {
            element /= length;
          }
          const T*       block  {a.data() + ((k + 1) * n) + k + 1};         // the trailing block, row stride n
          buffer<T>      w(m, T {});
          // w = (A − V Wᴴ − W Vᴴ) v on the trailing block, (Wᴴ v)ₚ and (Vᴴ v)ₚ beside A v; the next column of the
          // panel takes these earlier steps' terms in the same pass (its rows are w's).
          T* const next {k + 1 < k1 && j > 0 ? columns.data() + ((j + 1) * n) + k + 1 : nullptr};
          for (std::size_t p {0}; next != nullptr && p < j; ++p) {
            next_factors[2 * p]       = conj_of(by_row[(((2 * p) + 1) * n) + k + 1]);
            next_factors[(2 * p) + 1] = conj_of(by_row[((2 * p) * n) + k + 1]);
          }
          corrected_hermitian_product(block, n, m, v.data(), w.data(), symv_scratch, by_row.data() + k + 1, n, 2 * j, factors.data(),
                                      next, next_factors.data());
          column_ahead = next != nullptr;
          const T c {conjugate_dot(v.data(), w.data(), m)};
          for (std::size_t i {0}; i < m; ++i) { // w = 2 B v − 2 (vᴴ B v) v, so H B H = B − v wᴴ − w vᴴ
            w[i]                                    = (static_cast<T>(2) * w[i]) - (static_cast<T>(2) * c * v[i]);
            by_row[((2 * j) * n) + k + 1 + i]       = v[i];
            by_row[(((2 * j) + 1) * n) + k + 1 + i] = w[i];
          }
          reflectors[k + 1] = std::move(v); // acting from row k + 1, as householder_qr's from row k
        }
        // The trailing block from row and column k1: A −= V Wᴴ + W Vᴴ, one product of depth 2 · width,
        // [−V −W] [Wᴴ; Vᴴ], on the lower triangle alone, as LAPACK's dsyr2k: the reduction reads nothing above the
        // diagonal. A real product the kernel packs runs on the triangle's tiles alone (\ref gemm_triangle, each
        // element the bands' sum); a smaller or complex one by bands of rows, each to the diagonal, past which a band
        // computes what is never read.
        const std::size_t rest  {n - k1};
        const std::size_t depth {2 * width};
        for (std::size_t i {0}; i < rest; ++i) { // each row of left written whole, in order
          for (std::size_t p {0}; p < width; ++p) {
            left[(i * depth) + p]         = -by_row[((2 * p) * n) + k1 + i];
            left[(i * depth) + width + p] = -by_row[(((2 * p) + 1) * n) + k1 + i];
          }
        }
        for (std::size_t p {0}; p < width; ++p) {
          for (std::size_t i {0}; i < rest; ++i) {
            right[(p * rest) + i]           = conj_of(by_row[(((2 * p) + 1) * n) + k1 + i]);
            right[((width + p) * rest) + i] = conj_of(by_row[((2 * p) * n) + k1 + i]);
          }
        }
        T* trailing {a.data() + (k1 * n) + k1};
        if constexpr (std::is_same_v<T, double>) {
          if (depth >= gemm_min_depth && rest * rest >= ((gemm_packed_threshold + depth) - 1) / depth) {
            gemm_triangle<a_rows, b_rows>(rest, depth, left.get(), depth, right.get(), rest, trailing, n, true);
            continue;
          }
        }
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
                           const T              c  {detail::from_carried<T>(op.c)};
                           const T              s  {detail::from_carried<T>(op.s)};
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
                         const T c {detail::from_carried<T>(start.c)};
                         const T s {detail::from_carried<T>(start.s)};
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
    inline constexpr std::size_t divide_leaf {25};

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

    //! \brief The most elements a buffer of a thread's \ref merge_workspace keeps from one decomposition to the next:
    //!        2 MiB, every buffer an `eigh` of order 512 needs, so a thread keeps at most 8 MiB.
    inline constexpr std::size_t kept_workspace {std::size_t {1} << 18};

    /*!
     * \brief The calling thread's \ref merge_workspace, kept from one decomposition to the next: mapping its pages anew
     *        took 8–11 % of the eigenvectors' time (six AVX2 cores, an `eigh` of order 256 or 512). A join uses it
     *        whole on one thread, and nests no other join; \ref release_large_workspace lets a large buffer go after.
     */
    [[nodiscard]] inline merge_workspace& thread_workspace()
    {
      thread_local merge_workspace work;
      return work;
    }

    //! \brief Releases the buffers of \p work longer than \ref kept_workspace.
    inline void release_large_workspace(merge_workspace& work)
    {
      for (workspace_buffer* buffer : {&work.gaps, &work.updated, &work.columns, &work.factor}) {
        if (buffer->size > kept_workspace) {
          buffer->data.reset();
          buffer->size = 0;
        }
      }
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
    NUMERICAL_OUT_OF_LINE inline void plan_divide(std::vector<divide_node>& nodes,
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
    NUMERICAL_OUT_OF_LINE inline void solve_divide_leaf(const divide_node& node)
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
    NUMERICAL_OUT_OF_LINE inline void join_divide_node(const divide_node& node,
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
    NUMERICAL_OUT_OF_LINE inline void unscale_divide_node(const divide_node& node)
    {
      if (node.exponent != 0) {
        for (std::size_t i {0}; i < node.n; ++i) {
          node.d[i] = std::ldexp(node.work_d[i], node.exponent);
        }
      }
    }

    //! \brief Solves the block \p index of \p nodes and everything below it, each block right after its halves.
    NUMERICAL_OUT_OF_LINE inline void solve_divide_subtree(std::vector<divide_node>& nodes,
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
      const std::size_t task_order {nodes[tasks[0]].n};
      parallel_for(tasks.size(), tasks.size() > 1 ? task_order * task_order * task_order : 0, [&](std::size_t t) {
                     solve_divide_subtree(nodes, tasks[t], thread_workspace());
                     release_large_workspace(thread_workspace());
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
          join_divide_node(nodes[joins[0]], thread_workspace());
          release_large_workspace(thread_workspace());
          unscale_divide_node(nodes[joins[0]]);
          continue;
        }
        const std::size_t order {nodes[joins[0]].n};
        parallel_for(joins.size(), order * order * order / 4, [&](std::size_t j) {
                       join_divide_node(nodes[joins[j]], thread_workspace());
                       release_large_workspace(thread_workspace());
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
        converted[i] = from_carried<T>(left[i]);
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
                       at[(j * n) + r] = from_carried<T>(q[(r * n) + j]);
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

  namespace detail {

    //! \brief The columns \ref reduce_pencil reduces a block at a time, as LAPACK's `sygst` by its block size.
    inline constexpr std::size_t pencil_block {64};


    /*!
     * \brief The lower triangle of the `n × n` \p c (row stride \p ldc) taken to `L⁻¹ C L⁻ᴴ` for the lower-triangular
     *        \p l (row stride \p ldl), one column after another, as LAPACK's `sygs2` (type 1): each diagonal element
     *        divided by `L`'s squared, the column below it scaled, corrected by half of it twice around the rank-2 update
     *        of the rest, then solved against the rest of `L`. \p c's other triangle is neither read nor written.
     */
    template <typename T>
    void reduce_pencil_columns(T*          c,
                               std::size_t ldc,
                               const T*    l,
                               std::size_t ldl,
                               std::size_t n)
    {
      buffer<T> x(n); // the column below the diagonal, conjugated, then solved
      buffer<T> y(n); // L's, conjugated
      for (std::size_t k {0}; k < n; ++k) {
        const auto bkk      {real_part(l[(k * ldl) + k])};
        const auto akk      {real_part(c[(k * ldc) + k]) / (bkk * bkk)};
        c[(k * ldc) + k] = T {akk};
        const auto inverse  {static_cast<carried_t<T>>(1) / bkk};
        const T    half_akk {static_cast<carried_t<T>>(-0.5) * akk};
        for (std::size_t i {k + 1}; i < n; ++i) {
          c[(i * ldc) + k] = (c[(i * ldc) + k] * T {inverse}) + (half_akk * l[(i * ldl) + k]);
          x[i]             = conj_of(c[(i * ldc) + k]);
          y[i]             = conj_of(l[(i * ldl) + k]);
        }
        for (std::size_t i {k + 1}; i < n; ++i) { // C₂₂ −= x yᴴ + y xᴴ, x the column, y L's, row by row to the diagonal
          T* const row {c + (i * ldc)};
          subtract_product(row + k + 1, c[(i * ldc) + k], y.data() + k + 1, i - k);
          subtract_product(row + k + 1, l[(i * ldl) + k], x.data() + k + 1, i - k);
        }
        for (std::size_t i {k + 1}; i < n; ++i) { // L₂₂ z = x, by forward substitution, the column gathered
          x[i]             = c[(i * ldc) + k] + (half_akk * l[(i * ldl) + k]);
          x[i]             = (x[i] - dot_product(l + (i * ldl) + k + 1, x.data() + k + 1, i - k - 1)) / l[(i * ldl) + i];
          c[(i * ldc) + k] = x[i];
        }
      }
    }

    /*!
     * \brief `C += A·B` on the lower triangle of the `m × m` \p c (row stride \p ldc), \p a `m × depth` and \p b
     *        `depth × m` row-major, \p depth two blocks of \ref pencil_block (past \ref gemm_min_depth), where the
     *        matrix product kernel packs a product of that size: on the triangle's tiles alone (\ref gemm_triangle).
     *        \return whether it ran; a scalar other than `double` never does here.
     */
    template <typename T>
    [[nodiscard]] bool lower_triangle_product(std::size_t /*m*/,
                                              std::size_t /*depth*/,
                                              const T* /*a*/,
                                              const T* /*b*/,
                                              T* /*c*/,
                                              std::size_t /*ldc*/)
    {
      return false;
    }

    //! \brief \ref lower_triangle_product for `double`.
    [[nodiscard]] inline bool lower_triangle_product(std::size_t   m,
                                                     std::size_t   depth,
                                                     const double* a,
                                                     const double* b,
                                                     double*       c,
                                                     std::size_t   ldc)
    {
      static_assert(2 * pencil_block >= gemm_min_depth);
      if (m * m < ((gemm_packed_threshold + depth) - 1) / depth) {
        return false;
      }
      gemm_triangle<a_rows, b_rows>(m, depth, a, depth, b, m, c, ldc, true);
      return true;
    }

    /*!
     * \brief The lower triangle of the Hermitian `n × n` \p c (row-major) taken to `L⁻¹ C L⁻ᴴ` for the lower-triangular
     *        \p l (`n × n`, row-major), as LAPACK's `sygst` (type 1, lower) by blocks of \ref pencil_block columns: a
     *        block's diagonal by \ref reduce_pencil_columns; the rows below it solved against its `L₁₁ᴴ` from the right,
     *        corrected by half of `L₂₁ C₁₁` twice around the trailing rank-2k update `C₂₂ −= C₂₁ L₂₁ᴴ + L₂₁ C₂₁ᴴ` (its lower
     *        triangle); once every block is through, each one's rows solved against its `L₂₂`, the blocks on the
     *        threads together. Some `n³` operations where two whole triangular solves took `2 n³`.
     *        \p c's other triangle is neither read nor written.
     */
    template <typename T>
    void reduce_pencil(T*          c,
                       const T*    l,
                       std::size_t n)
    {
      buffer<T> half(pencil_block * pencil_block);
      buffer<T> diagonal(pencil_block * pencil_block);
      buffer<T> turned;
      buffer<T> left;
      buffer<T> right;
      for (std::size_t k0 {0};; k0 += pencil_block) { // to the last block, whose rows below are none
        const std::size_t kb {std::min(pencil_block, n - k0)};
        const std::size_t k1 {k0 + kb};
        const std::size_t m  {n - k1};
        reduce_pencil_columns(c + (k0 * n) + k0, n, l + (k0 * n) + k0, n, kb);
        if (m == 0) {
          break;
        }
        T* const       c21 {c + (k1 * n) + k0};
        const T* const l21 {l + (k1 * n) + k0};
        // C₂₁ L₁₁⁻ᴴ: the rows of C₂₁ conjugated as columns, L₁₁ X = C₂₁ᴴ, turned back.
        for (std::size_t i {0}; i < kb; ++i) {
          for (std::size_t j {0}; j <= i; ++j) {
            diagonal[(i * kb) + j] = l[((k0 + i) * n) + k0 + j];
          }
          std::fill(diagonal.begin() + static_cast<std::ptrdiff_t>((i * kb) + i + 1), diagonal.begin() + static_cast<std::ptrdiff_t>((i + 1) * kb), T {});
        }
        turned.resize(kb * m);
        for (std::size_t r {0}; r < m; ++r) {
          for (std::size_t q {0}; q < kb; ++q) {
            turned[(q * m) + r] = conj_of(c21[(r * n) + q]);
          }
        }
        triangular_rows(diagonal.data(), kb, true, false, false, turned.data(), m);
        for (std::size_t r {0}; r < m; ++r) {
          for (std::size_t q {0}; q < kb; ++q) {
            c21[(r * n) + q] = conj_of(turned[(q * m) + r]);
          }
        }
        // −½ C₁₁, whole from its lower triangle.
        for (std::size_t i {0}; i < kb; ++i) {
          for (std::size_t j {0}; j < i; ++j) {
            const T value {static_cast<carried_t<T>>(-0.5) * c[((k0 + i) * n) + k0 + j]};
            half[(i * kb) + j] = value;
            half[(j * kb) + i] = conj_of(value);
          }
          half[(i * kb) + i] = T {static_cast<carried_t<T>>(-0.5) * real_part(c[((k0 + i) * n) + k0 + i])};
        }
        gemm(m, kb, kb, l21, n, half.data(), kb, c21, n);
        // C₂₂ −= C₂₁ L₂₁ᴴ + L₂₁ C₂₁ᴴ on the lower triangle: [−C₂₁ −L₂₁] [L₂₁ᴴ; C₂₁ᴴ].
        const std::size_t depth {2 * kb};
        left.resize(m * depth);
        right.resize(depth * m);
        for (std::size_t r {0}; r < m; ++r) {
          for (std::size_t q {0}; q < kb; ++q) {
            left[(r * depth) + q]      = -c21[(r * n) + q];
            left[(r * depth) + kb + q] = -l21[(r * n) + q];
            right[(q * m) + r]         = conj_of(l21[(r * n) + q]);
            right[((kb + q) * m) + r]  = conj_of(c21[(r * n) + q]);
          }
        }
        T* const   c22  {c + (k1 * n) + k1};
        const bool done {lower_triangle_product(m, depth, left.data(), right.data(), c22, n)};
        for (std::size_t r0 {0}; !done && r0 < m; r0 += tridiagonal_band) {
          const std::size_t rows {std::min(tridiagonal_band, m - r0)};
          gemm(rows, r0 + rows, depth, left.data() + (r0 * depth), depth, right.data(), m, c22 + (r0 * n), n);
        }
        gemm(m, kb, kb, l21, n, half.data(), kb, c21, n);
      }
      // Each block's L₂₂⁻¹ C₂₁, L₂₂ read where it lies: a block's rows below it are its own once the rank-2k update
      // has read them, so the blocks are solved together, the largest first, each on one thread.
      const std::size_t below {(std::max(n, std::size_t {1}) - 1) / pencil_block}; // the blocks with rows below them
      parallel_for(below, n * n * pencil_block / 2, [&](std::size_t index) {
                     const std::size_t k0 {index * pencil_block};
                     const std::size_t k1 {k0 + pencil_block};
                     const std::size_t m  {n - k1};
                     buffer<T>         minus((((m + 1) / 2) + triangular_block) * pencil_block);
                     triangular_halves(l + (k1 * n) + k1, n, true, false, false, c + (k1 * n) + k0, n, pencil_block, 0, m, minus.data());
                   });
    }

    /*!
     * \brief The order from which a pencil's standard form is reduced by blocks (\ref reduce_pencil): half the
     *        operations of two whole triangular solves, but a block at a time, where the two solves share their
     *        columns among the threads whole. Below it the solves ran faster: ten AVX-512 cores up to order 1100 or
     *        so, six AVX2 cores up to 500, the M1 Pro (its products on AMX) up to some 2000 real and 2500 complex. From
     *        it: order 3072, six AVX2 cores 683 → 335 ms (complex 1749 → 884), ten AVX-512 cores 423 → 224 ms (complex
     *        900 → 528), M1 Pro 204 → 161 ms (complex 724 → 684). One order whatever the threads: the bits of `C` do
     *        not depend on their number.
     */
    template <typename T>
    inline constexpr std::size_t pencil_reduction_least {is_complex_number<T> ? 3072 : 2048};

    /*!
     * \brief `C = L⁻¹ A L⁻ᴴ` whole, \p y holding `A`'s lower triangle (`n × n`, row-major, the diagonal real) and
     *        \p lower `L`: by blocks (\ref reduce_pencil, `C`'s lower triangle, then its upper from it) where
     *        \p blocks, else by two whole triangular solves, `L⁻¹ A` and `L⁻¹ (L⁻¹ A)ᴴ`.
     */
    template <typename T>
    [[nodiscard]] buffer<T> pencil_standard_form(buffer<T>   y,
                                                 const T*    lower,
                                                 std::size_t n,
                                                 bool        blocks)
    {
      if (blocks) {
        reduce_pencil(y.data(), lower, n);
        for (std::size_t i {0}; i < n; ++i) {
          for (std::size_t j {0}; j < i; ++j) {
            y[(j * n) + i] = conj_of(y[(i * n) + j]);
          }
        }
        return y;
      }
      for (std::size_t i {0}; i < n; ++i) { // A whole
        for (std::size_t j {0}; j < i; ++j) {
          y[(j * n) + i] = conj_of(y[(i * n) + j]);
        }
      }
      solve_triangular_in_place(lower, n, true, triangular_op::none, false, y.data(), n); // L⁻¹ A
      buffer<T> c(n * n, T {});
      for (std::size_t i {0}; i < n; ++i) { // (L⁻¹ A)ᴴ = A L⁻ᴴ
        for (std::size_t j {0}; j < n; ++j) {
          c[(i * n) + j] = conj_of(y[(j * n) + i]);
        }
      }
      solve_triangular_in_place(lower, n, true, triangular_op::none, false, c.data(), n); // L⁻¹ A L⁻ᴴ
      return c;
    }

    /*!
     * \brief The pencil `A − λ B` of a Hermitian `A` and a Hermitian positive-definite `B` (each read in its lower
     *        triangle, of the diagonal its real part) in standard form: `B = L Lᴴ` and `C = L⁻¹ A L⁻ᴴ`, Hermitian, as
     *        LAPACK's `sygst` (type 1) forms it from \ref pencil_reduction_least, by two triangular solves below
     *        (\ref pencil_standard_form). `C`'s eigenpairs `(λ, y)` are the pencil's
     *        `(λ, L⁻ᴴ y)`, those vectors `B`-orthonormal (`Xᴴ B X = I`), as `scipy.linalg.eigh(a, b)`'s.
     */
    template <typename T>
    struct definite_pencil
    {
      /*!
       * \throws dimension_error              If \p a or \p b is not square, or their orders differ.
       * \throws non_finite_error             If an entry read is infinite or NaN.
       * \throws not_positive_definite_error  If \p b is not positive definite.
       */
      definite_pencil(const matrix<T>& a,
                      const matrix<T>& b)
        : factor(checked_factor(a, b)),
          reduced(standard_form(a, factor))
      {}

      //! \brief The pencil's eigenvectors from \p v, the columns of `C`'s: `L⁻ᴴ V`.
      [[nodiscard]] matrix<T> vectors_of(matrix<T>&& v) const
      {
        const std::size_t n {factor.rows()};
        buffer<T>         x {std::move(v).elements()};
        solve_triangular_in_place(std::to_address(factor.begin()), n, true, triangular_op::adjoint, false, x.data(), n);
        return matrix<T> {n, n, std::move(x)};
      }

      matrix<T> factor;  //!< `L`, lower-triangular, `B = L Lᴴ`.
      matrix<T> reduced; //!< `C = L⁻¹ A L⁻ᴴ`, whole.

    private:

      [[nodiscard]] static matrix<T> checked_factor(const matrix<T>& a,
                                                    const matrix<T>& b)
      {
        if (a.rows() != a.cols() || b.rows() != b.cols() || b.rows() != a.rows()) {
          throw dimension_error("a definite pencil requires two square matrices of one order");
        }
        require_finite_lower(a, "a definite pencil");
        return cholesky_decomposition<T> {b}.lower();
      }

      [[nodiscard]] static matrix<T> standard_form(const matrix<T>& a,
                                                   const matrix<T>& l)
      {
        const std::size_t n          {a.rows()};
        buffer<T>         y(n * n, T {});
        for (std::size_t i {0}; i < n; ++i) { // A's lower triangle, a real diagonal
          for (std::size_t j {0}; j < i; ++j) {
            y[(i * n) + j] = a(i, j);
          }
          y[(i * n) + i] = T {real_part(a(i, i))};
        }
        return matrix<T> {n, n, pencil_standard_form(std::move(y), std::to_address(l.begin()), n, n >= pencil_reduction_least<T>)};
      }
    };
  } // namespace detail

  //! \brief How \ref numerical::symmetric_eigen diagonalises.
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
      : symmetric_eigen(matrix<double> {input},
                        method)
    {}

    /*!
     * \brief Diagonalises \p input in its own storage, an expiring matrix's elements taken over: its scaled copy is
     *        made in place, one matrix less at the peak.
     * \param[in] input  A real symmetric matrix (its lower triangle is read).
     * \param[in] method The algorithm.
     * \throws dimension_error   If \p input is not square.
     * \throws convergence_error If the method did not converge within its bound.
     */
    explicit symmetric_eigen(matrix<double>&& input,
                             eigen_method     method = eigen_method::divide_and_conquer)
      : n_(input.rows())
    {
      if (input.cols() != n_) {
        throw dimension_error("symmetric eigendecomposition requires a square matrix");
      }
      detail::require_finite_lower(input, "symmetric eigendecomposition");
      // Work on the matrix divided by the power of two of its largest entry: a sum of squares of entries near 1e170
      // overflows and of entries near 1e-170 underflows to 0, and either stops the sweeps before a
      // single rotation. The eigenvalues are scaled back; the eigenvectors do not depend on it.
      double scale {0.0};
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j <= i; ++j) {
          scale = std::max(scale, std::abs(input(i, j)));
        }
      }
      scale = detail::power_of_two_scale(scale);
      buffer<double> a                   {std::move(input).elements()};
      buffer<double> diagonal(n_, double {});
      for (std::size_t i {0}; i < n_; ++i) { // the lower triangle divided; above it, whatever the input held
        for (std::size_t j {0}; j <= i; ++j) {
          a[(i * n_) + j] /= scale;
        }
      }
      if (method == eigen_method::jacobi) {
        buffer<double>      v(n_ * n_, 0.0);
        double              frobenius_squared {0.0};
        for (std::size_t i {0}; i < n_; ++i) {
          v[(i * n_) + i] = 1.0;
          for (std::size_t j {0}; j < n_; ++j) {
            if (j > i) {
              a[(i * n_) + j] = a[(j * n_) + i]; // the lower triangle, mirrored
            }
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
        std::fill(a.begin() + static_cast<std::ptrdiff_t>((i * n_) + i + 1), a.begin() + static_cast<std::ptrdiff_t>((i + 1) * n_), 0.0);
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

    /*!
     * \brief Diagonalises the pencil `A − λ B`, `A x = λ B x`, as `scipy.linalg.eigh(a, b)`: `B = L Lᴴ` and the
     *        symmetric `L⁻¹ A L⁻ᴴ` diagonalised, its eigenvectors `y` the pencil's `L⁻ᴴ y`, `B`-orthonormal
     *        (`Xᵀ B X = I`). Each matrix's lower triangle alone is read.
     * \param[in] a      A real symmetric matrix.
     * \param[in] b      A real symmetric positive-definite matrix of its order.
     * \param[in] method The algorithm for the standard form.
     * \throws dimension_error              If \p a or \p b is not square, or their orders differ.
     * \throws non_finite_error             If an entry read is infinite or NaN.
     * \throws not_positive_definite_error  If \p b is not positive definite.
     * \throws convergence_error            If the method did not converge within its bound.
     */
    symmetric_eigen(const matrix<double>& a,
                    const matrix<double>& b,
                    eigen_method          method = eigen_method::divide_and_conquer)
      : symmetric_eigen(detail::definite_pencil<double> {a, b},
                        method)
    {}

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

    //! \brief The pencil's eigenpairs: its standard form's, the vectors carried back.
    symmetric_eigen(detail::definite_pencil<double> pencil,
                    eigen_method                    method)
      : symmetric_eigen(std::move(pencil.reduced),
                        method)
    {
      vectors_ = pencil.vectors_of(std::move(vectors_));
    }

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
      // Work on the matrix divided by the power of two of its largest entry, as symmetric_eigen does; the vectors do not depend
      // on it. The entries read: the lower triangle, and the diagonal's real part (its imaginary part is ignored,
      // so it scales nothing: an infinite one made every value NaN, a huge one wrong).
      double scale {0.0};
      for (std::size_t i {0}; i < n_; ++i) {
        for (std::size_t j {0}; j < i; ++j) {
          scale = std::max(scale, std::abs(input(i, j)));
        }
        scale = std::max(scale, std::abs(input(i, i).real()));
      }
      scale = detail::power_of_two_scale(scale);
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

    /*!
     * \brief Diagonalises the pencil `A − λ B`, `A x = λ B x`, as `scipy.linalg.eigh(a, b)`: `B = L Lᴴ` and the
     *        Hermitian `L⁻¹ A L⁻ᴴ` diagonalised, its eigenvectors `y` the pencil's `L⁻ᴴ y`, `B`-orthonormal
     *        (`Xᴴ B X = I`). Each matrix's lower triangle alone is read, of the diagonal its real part.
     * \param[in] a      A complex Hermitian matrix.
     * \param[in] b      A complex Hermitian positive-definite matrix of its order.
     * \param[in] method The algorithm for the tridiagonal, as for one matrix.
     * \throws std::invalid_argument        If \p method is \ref eigen_method::jacobi.
     * \throws dimension_error              If \p a or \p b is not square, or their orders differ.
     * \throws non_finite_error             If an entry read is infinite or NaN.
     * \throws not_positive_definite_error  If \p b is not positive definite.
     * \throws convergence_error            If an iteration did not converge within its bound (not observed).
     */
    hermitian_eigen(const matrix<std::complex<double>>& a,
                    const matrix<std::complex<double>>& b,
                    eigen_method                        method = eigen_method::divide_and_conquer)
      : hermitian_eigen(detail::definite_pencil<std::complex<double>> {a, b},
                        method)
    {}

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

    //! \brief The pencil's eigenpairs: its standard form's, the vectors carried back.
    hermitian_eigen(const detail::definite_pencil<std::complex<double>>& pencil,
                    eigen_method                                         method)
      : hermitian_eigen(pencil.reduced,
                        method)
    {
      vectors_ = pencil.vectors_of(std::move(vectors_));
    }

    std::size_t                  n_;       //!< The matrix order.
    vector<double>               values_;  //!< Eigenvalues, ascending.
    matrix<std::complex<double>> vectors_; //!< Eigenvectors as columns.
  };
} // namespace numerical

#endif // NUMERICAL_FACTORIZATIONS_HPP
