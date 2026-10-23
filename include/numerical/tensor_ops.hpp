/*!
 * \file tensor_ops.hpp
 * \brief Element-wise arithmetic with NumPy's broadcasting, reductions, reshaping and creation of tensors.
 *
 * Two shapes broadcast as NumPy's do: aligned on their last axes, each pair of extents equal or one of them
 * 1 (a missing leading axis counts as 1), the result taking the larger. A sum adds by pairwise halving, as
 * NumPy's reference algorithm does (NumPy's vector builds group the terms their own way, so the last bits
 * can differ by platform): its rounding error grows with the logarithm of the count instead of the count. A maximum or
 * a minimum is defined for an ordered scalar only, and an empty reduction without an identity throws.
 */
#ifndef NUMERICAL_TENSOR_OPS_HPP
#define NUMERICAL_TENSOR_OPS_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "complex_math.hpp"
#include "simd.hpp"
#include "tensor.hpp"

namespace numerical {

  namespace detail {

    //! \brief A shape as NumPy prints it, `(2, 3)`, for a message.
    [[nodiscard]] inline std::string shape_text(const std::vector<std::size_t>& shape)
    {
      std::string out {"("};
      for (std::size_t i {0}; i < shape.size(); ++i) {
        out += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
      }
      return out + (shape.size() == 1 ? ",)" : ")");
    }

    //! \brief The row-major strides of \p shape, with a stride of 0 on each axis of extent 1 (it broadcasts),
    //!        aligned on the last axis of a result of rank \p rank (missing leading axes stride 0 too).
    [[nodiscard]] inline std::vector<std::size_t> broadcast_strides(const std::vector<std::size_t>& shape,
                                                                    std::size_t                     rank)
    {
      std::vector<std::size_t> strides(rank, 0);
      std::size_t              stride {1};
      for (std::size_t i {shape.size()}; i-- > 0;) {
        strides[rank - shape.size() + i] = shape[i] == 1 ? 0 : stride;
        stride                          *= shape[i];
      }
      return strides;
    }

    //! \brief The extents and strides that walk a tensor of \p shape with axis \p axis taken out: `outer`
    //!        positions before it, `inner` after it, the axis itself `extent` long at stride `inner`.
    struct axis_walk
    {
      std::size_t outer;  //!< The product of the extents before the axis.
      std::size_t extent; //!< The axis' extent.
      std::size_t inner;  //!< The product of the extents after the axis.
    };

    //! \brief \ref axis_walk of \p shape along \p axis; `outer` and `inner` 0 where the result is empty.
    //!        \throws dimension_error If \p axis is not an axis, or the result's elements overflow.
    [[nodiscard]] inline axis_walk walk_along(const std::vector<std::size_t>& shape,
                                              std::size_t                     axis)
    {
      if (axis >= shape.size()) {
        throw dimension_error("axis " + std::to_string(axis) + " is out of bounds for a tensor of rank "
                              + std::to_string(shape.size()));
      }
      const auto before {shape.begin() + static_cast<std::ptrdiff_t>(axis)};
      axis_walk  walk   {.outer = 1, .extent = shape[axis], .inner = 1};
      // The result's elements, the axis taken out: none (and no line walked) where a kept extent is 0, whatever
      // the others; else their product, checked, which the outer and inner parts then divide.
      if (std::find(shape.begin(), before, std::size_t {0}) != before || std::find(before + 1, shape.end(), std::size_t {0}) != shape.end()) {
        walk.outer = 0;
        walk.inner = 0;
        return walk;
      }
      walk.outer = detail::element_count(shape.begin(), before);
      walk.inner = detail::element_count(before + 1, shape.end());
      static_cast<void>(detail::checked_mul(walk.outer, walk.inner));
      return walk;
    }

    //! \brief `x · y + z`: one fused multiply-add where \p fused, else the product rounded first (statements apart, so
    //!        no compiler contracts them).
    template <std::floating_point R>
    [[nodiscard]] R numpy_muladd(R    x,
                                 R    y,
                                 R    z,
                                 bool fused)
    {
      if (fused) {
        return std::fma(x, y, z);
      }
      const R product {x * y};
      return product + z;
    }

    /*!
     * \brief `a · b` of complex numbers as NumPy's array loop computes it: `ai·bi` and `ai·br` rounded, then
     *        `ar·br − ai·bi` and `ar·bi + ai·br`, each by a fused multiply-add where \p fused (NumPy's vectors on arm64
     *        and on x86-64 with FMA), else each product rounded first. No recovery of an infinity from a NaN, as
     *        `std::complex`'s (C Annex G) makes: `1j · (inf + inf·j)` is NaN, `(1e308 + 1e308j) · (1e308 − 1e308j)`
     *        `inf + inf·j` when fused.
     */
    template <split_complex C>
    [[nodiscard]] C numpy_multiply(const C& a,
                                   const C& b,
                                   bool     fused)
    {
      using R = typename C::value_type;
      const R ii {a.imag() * b.imag()};
      const R ir {a.imag() * b.real()};
      return C {numpy_muladd(a.real(), b.real(), -ii, fused), numpy_muladd(a.real(), b.imag(), ir, fused)};
    }

    //! \brief \p x divided by +0, as IEEE divides it, spelled out: a NaN for a zero or a NaN, else an infinity of
    //!        \p x's sign (a division by a zero the compiler sees is MSVC's C4723 under /WX).
    template <std::floating_point R>
    [[nodiscard]] R divided_by_zero(R x) noexcept
    {
      if (x == R {0} || std::isnan(x)) {
        return std::numeric_limits<R>::quiet_NaN();
      }
      return std::copysign(std::numeric_limits<R>::infinity(), x);
    }

    /*!
     * \brief `a / b` of complex numbers as NumPy's array loop computes it: Smith's division by the larger part of
     *        \p b (an exact zero divided part by part, to an infinity or a NaN), each `x + y·z` of it one fused
     *        multiply-add where \p fused (NumPy's build contracts them on arm64, not on x86-64).
     */
    template <split_complex C>
    [[nodiscard]] C numpy_divide(const C& a,
                                 const C& b,
                                 bool     fused)
    {
      using R = typename C::value_type;
      const R real_part      {std::abs(b.real())};
      const R imaginary_part {std::abs(b.imag())};
      if (real_part >= imaginary_part) {
        if (real_part == R {0}) { // the imaginary part's magnitude no larger: a zero too
          return C {divided_by_zero(a.real()), divided_by_zero(a.imag())};
        }
        const R ratio {b.imag() / b.real()};
        const R scale {R {1} / numpy_muladd(b.imag(), ratio, b.real(), fused)};
        return C {numpy_muladd(a.imag(), ratio, a.real(), fused) * scale, numpy_muladd(-a.real(), ratio, a.imag(), fused) * scale};
      }
      const R ratio {b.real() / b.imag()};
      const R scale {R {1} / numpy_muladd(b.real(), ratio, b.imag(), fused)};
      return C {numpy_muladd(a.real(), ratio, a.imag(), fused) * scale, numpy_muladd(a.imag(), ratio, -a.real(), fused) * scale};
    }

    /*!
     * \brief NumPy's complex reciprocal `1 / x` (its `reciprocal`, the loop `x ** -1` takes): the ratio of the smaller
     *        part to the larger, `d` the larger plus the smaller times it (one fused multiply-add where \p fused), then
     *        `1 / d` and `−ratio / d` — not \ref numpy_divide's quotient of 1.
     */
    template <split_complex C>
    [[nodiscard]] C numpy_reciprocal(const C& x,
                                     bool     fused)
    {
      using R = typename C::value_type;
      if (std::abs(x.imag()) <= std::abs(x.real())) {
        const R ratio {x.imag() / x.real()};
        const R d     {numpy_muladd(x.imag(), ratio, x.real(), fused)};
        return C {R {1} / d, -ratio / d};
      }
      const R ratio {x.real() / x.imag()};
      const R d     {numpy_muladd(x.real(), ratio, x.imag(), fused)};
      return C {ratio / d, R {-1} / d};
    }

    /*!
     * \brief NumPy's complex power `a ** b` (npymath's `npy_cpow`): 1 for a zero exponent, 0 or NaN for a zero base;
     *        an integer exponent below 100 in magnitude by products (\ref numpy_multiply, from 1 past the cubes) and,
     *        when negative, the quotient of 1 by them (\ref numpy_divide), each fused where \p fused; any other by
     *        `std::pow`, where NumPy calls the C library's `cpow`.
     */
    template <split_complex C>
    [[nodiscard]] C numpy_power(const C& a,
                                const C& b,
                                bool     fused)
    {
      using R = typename C::value_type;
      if (b.real() == R {0} && b.imag() == R {0}) {
        return C {R {1}, R {0}};
      }
      if (a.real() == R {0} && a.imag() == R {0}) {
        if (b.real() > R {0}) {
          return C {R {0}, R {0}};
        }
        return C {std::numeric_limits<R>::quiet_NaN(), std::numeric_limits<R>::quiet_NaN()};
      }
      if (b.imag() != R {0} || !(b.real() > R {-100} && b.real() < R {100}) || std::trunc(b.real()) != b.real()) {
        return std::pow(a, b);
      }
      const auto n {static_cast<int>(b.real())};
      if (n == 1) {
        return a;
      }
      if (n == 2) {
        return numpy_multiply(a, a, fused);
      }
      if (n == 3) {
        return numpy_multiply(a, numpy_multiply(a, a, fused), fused);
      }
      const auto magnitude {static_cast<unsigned>(n < 0 ? -n : n)};
      C          product   {R {1}, R {0}};
      C          power     {a};
      for (unsigned bit {1U};; bit <<= 1U) { // the binary powers of a, those of the exponent's bits multiplied in
        if ((magnitude & bit) != 0U) {
          product = numpy_multiply(product, power, fused);
        }
        if (magnitude < (bit << 1U)) {
          break;
        }
        power = numpy_multiply(power, power, fused);
      }
      if (n < 0) {
        return numpy_divide(C {R {1}, R {0}}, product, fused);
      }
      return product;
    }

    //! \brief The square root of a scalar, as `std::sqrt` gives it.
    template <typename T>
    [[nodiscard]] T root(const T& x)
    {
      return static_cast<T>(std::sqrt(x));
    }

    //! \brief The square root of a complex number of `float` or `double`: \ref complex_sqrt.
    template <split_complex T>
    [[nodiscard]] T root(const T& x)
    {
      return complex_sqrt(x);
    }

    //! \brief `exp` of a scalar, as `std::exp` gives it.
    template <typename T>
    [[nodiscard]] T elementary_exp(const T& x)
    {
      return static_cast<T>(std::exp(x));
    }

    //! \brief `exp` of a complex number of `float` or `double`: \ref complex_exp.
    template <split_complex T>
    [[nodiscard]] T elementary_exp(const T& x)
    {
      return complex_exp(x);
    }

    //! \brief `log` of a scalar, as `std::log` gives it.
    template <typename T>
    [[nodiscard]] T elementary_log(const T& x)
    {
      return static_cast<T>(std::log(x));
    }

    //! \brief `log` of a complex number of `float` or `double`: \ref complex_log.
    template <split_complex T>
    [[nodiscard]] T elementary_log(const T& x)
    {
      return complex_log(x);
    }

    //! \brief `sin` of a scalar, as `std::sin` gives it.
    template <typename T>
    [[nodiscard]] T elementary_sin(const T& x)
    {
      return static_cast<T>(std::sin(x));
    }

    //! \brief `sin` of a complex number of `float` or `double`: \ref complex_sin.
    template <split_complex T>
    [[nodiscard]] T elementary_sin(const T& x)
    {
      return complex_sin(x);
    }

    //! \brief `cos` of a scalar, as `std::cos` gives it.
    template <typename T>
    [[nodiscard]] T elementary_cos(const T& x)
    {
      return static_cast<T>(std::cos(x));
    }

    //! \brief `cos` of a complex number of `float` or `double`: \ref complex_cos.
    template <split_complex T>
    [[nodiscard]] T elementary_cos(const T& x)
    {
      return complex_cos(x);
    }

    //! \brief `tan` of a scalar, as `std::tan` gives it.
    template <typename T>
    [[nodiscard]] T elementary_tan(const T& x)
    {
      return static_cast<T>(std::tan(x));
    }

    //! \brief `tan` of a complex number of `float` or `double`: \ref complex_tan.
    template <split_complex T>
    [[nodiscard]] T elementary_tan(const T& x)
    {
      return complex_tan(x);
    }

    //! \brief `tanh` of a scalar, as `std::tanh` gives it.
    template <typename T>
    [[nodiscard]] T elementary_tanh(const T& x)
    {
      return static_cast<T>(std::tanh(x));
    }

    //! \brief `tanh` of a complex number of `float` or `double`: \ref complex_tanh.
    template <split_complex T>
    [[nodiscard]] T elementary_tanh(const T& x)
    {
      return complex_tanh(x);
    }

    /*!
     * \brief NumPy's complex magnitude (its `absolute`'s vector loop): the parts' magnitudes, an infinite one making
     *        both infinite and then a NaN both NaN, then the larger times `sqrt(ratio² + 1)` for the ratio of the
     *        smaller to it (0 where the larger is 0 or the smaller infinite), `ratio² + 1` one fused multiply-add
     *        where \p fused.
     */
    template <split_complex C>
    [[nodiscard]] typename C::value_type numpy_absolute(const C& x,
                                                        bool     fused)
    {
      using R = typename C::value_type;
      constexpr R inf         {std::numeric_limits<R>::infinity()};
      constexpr R nan         {std::numeric_limits<R>::quiet_NaN()};
      R           re          {std::abs(x.real())};
      R           im          {std::abs(x.imag())};
      const bool  re_infinite {re == inf};
      const bool  im_infinite {im == inf};
      if (re_infinite) {
        im = inf;
      }
      if (im_infinite) {
        re = inf;
      }
      const bool re_nan {std::isnan(re)};
      const bool im_nan {std::isnan(im)};
      if (re_nan) {
        im = nan;
      }
      if (im_nan) {
        re = nan;
      }
      const R larger  {std::max(re, im)};
      const R smaller {std::min(im, re)};
      R       ratio   {0};
      if (larger != R {0} && smaller != inf) {
        ratio = smaller / larger;
      }
      return std::sqrt(numpy_muladd(ratio, ratio, R {1}, fused)) * larger;
    }

    //! \brief Whether NumPy's complex product fuses its multiply-adds on this processor: on arm64, and on x86-64
    //!        where its AVX2 loop runs (FMA with it); elsewhere its loop rounds each product.
    [[nodiscard]] inline bool numpy_fuses_products() noexcept
    {
#if defined(__aarch64__) || defined(_M_ARM64)
      return true;
#elif defined(NUMERICAL_X86_COMPILED)
      return cpu_features().avx2_fma;
#else
      return false;
#endif
    }

    //! \brief Whether NumPy's complex product fuses its multiply-adds in its scalar loop (where \ref
    //!        numpy_loop_is_scalar holds), as its PyPI wheels are built: that loop sits beside the vector one, and
    //!        clang (macOS) and MSVC (Windows) contract it where the vector loop has FMA; GCC (Linux, ISO C) does not.
    [[nodiscard]] inline bool numpy_fuses_scalar_products() noexcept
    {
#if defined(__APPLE__) || defined(_WIN32)
      return numpy_fuses_products();
#else
      return false;
#endif
    }

    //! \brief Whether NumPy's complex quotient fuses its multiply-adds, as its PyPI wheels are built: on macOS arm64,
    //!        whose clang contracts them (FMA is arm64's baseline); not on Linux, whose GCC does not, nor on x86-64,
    //!        whose baseline has no FMA.
    [[nodiscard]] inline bool numpy_fuses_quotients() noexcept
    {
#if defined(__APPLE__) && defined(__aarch64__)
      return true;
#else
      return false;
#endif
    }

    /*!
     * \brief Whether NumPy multiplies \p a by \p b in its scalar loop rather than its vector one: when the result is
     *        one element and an operand of rank above 0 is broadcast to a higher rank, NumPy's iterator hands its loop
     *        an output step of 0, which the vector loop refuses.
     */
    template <typename T>
    [[nodiscard]] bool numpy_loop_is_scalar(const tensor<T>& a,
                                            const tensor<T>& b) noexcept
    {
      const auto        ones {[](const tensor<T>& t) { return std::ranges::all_of(t.shape(), [](std::size_t e) { return e == 1; }); }};
      const std::size_t rank {std::max(a.rank(), b.rank())};
      const auto        broadcast {[rank](const tensor<T>& t) { return t.rank() != 0 && t.rank() != rank; }};
      if (!ones(a) || !ones(b)) {
        return false;
      }
      return broadcast(a) || broadcast(b);
    }

    /*!
     * \brief A mean: \p value divided by the count \p n, in the wider of `T` and `double` and back to `T`, the count
     *        exact (to 2^53), as NumPy divides by its integer count; a `float`'s count rounded to a `float` past 2^24
     *        (2^24 + 1 ones averaged to 1, NumPy's 0.99999994).
     */
    template <std::floating_point T>
    [[nodiscard]] T divided_by_count(T           value,
                                     std::size_t n)
    {
      using wide = std::common_type_t<T, double>;
      return static_cast<T>(static_cast<wide>(value) / static_cast<wide>(n));
    }

    /*!
     * \brief \ref divided_by_count of a complex number of `float` or `double`, as NumPy's mean divides it: by the
     *        complex count `n + 0j`, in complex `double`, by its division (\ref numpy_divide), so a NaN part makes
     *        both NaN and each part is multiplied by the reciprocal of the count.
     */
    template <split_complex T>
    [[nodiscard]] T divided_by_count(T           value,
                                     std::size_t n)
    {
      using R = typename T::value_type;
      const std::complex<double> wide {static_cast<double>(value.real()), static_cast<double>(value.imag())};
      // Smith's division by n + 0j: a ratio of 0 and a scale of 1 / n, each product with that 0 exact, fused or not.
      const double scale              {1.0 / static_cast<double>(n)};
      const double zero               {0.0};
      return T {static_cast<R>((wide.real() + (wide.imag() * zero)) * scale), static_cast<R>((wide.imag() - (wide.real() * zero)) * scale)};
    }

    //! \brief \ref divided_by_count of any other scalar: divided by the count as a `T`.
    //!        \throws dimension_error If the count is 0: an integer has no NaN for the mean of nothing.
    template <typename T>
    [[nodiscard]] T divided_by_count(T           value,
                                     std::size_t n)
    {
      if (n == 0) {
        throw dimension_error("a mean over no element has no value in an integer type");
      }
      return value / static_cast<T>(n);
    }

    /*!
     * \brief The extreme \p value of \p count elements \p stride apart from \p first, its zero's sign settled as
     *        IEEE 754-2019's maximum and minimum settle it, and NumPy's on arm64: no zero is larger than the other, so
     *        the first met stayed; `max` gives +0 where one is among them, `min` −0, whatever their order. A NaN, or
     *        any other value, as it is.
     */
    template <std::floating_point T,
              typename Order>
    [[nodiscard]] T settled_zero(T            value,
                                 const T*     first,
                                 std::size_t  count,
                                 std::size_t  stride,
                                 const Order& larger)
    {
      if (value != T {0}) {
        return value;
      }
      const bool maximum {larger(T {1}, T {0})};
      for (std::size_t i {0}; i < count; ++i) {
        const T x {first[i * stride]};
        if (x == T {0} && std::signbit(x) != maximum) {
          return x;
        }
      }
      return value;
    }

    //! \brief \ref settled_zero of a type with one zero: \p value.
    template <typename T,
              typename Order>
    [[nodiscard]] T settled_zero(T value,
                                 const T* /*first*/,
                                 std::size_t /*count*/,
                                 std::size_t /*stride*/,
                                 const Order& /*larger*/)
    {
      return value;
    }

    //! \brief The shape with axis \p axis removed.
    [[nodiscard]] inline std::vector<std::size_t> without_axis(std::vector<std::size_t> shape,
                                                               std::size_t              axis)
    {
      shape.erase(shape.begin() + static_cast<std::ptrdiff_t>(axis));
      return shape;
    }

    /*!
     * \brief Reduces each line of \p t along \p axis by \p reduce, which takes the line's first element,
     *        its length and its stride.
     */
    template <typename T,
              typename Reduce>
    [[nodiscard]] tensor<T> reduce_along(const tensor<T>& t,
                                         std::size_t      axis,
                                         const Reduce&    reduce)
    {
      const axis_walk walk                           {walk_along(t.shape(), axis)};
      buffer<T>       out(walk.outer * walk.inner);
      for (std::size_t o {0}; o < walk.outer; ++o) {
        for (std::size_t i {0}; i < walk.inner; ++i) {
          out[(o * walk.inner) + i] = reduce(t.data().data() + (o * walk.extent * walk.inner) + i, walk.extent, walk.inner);
        }
      }
      return tensor<T> {without_axis(t.shape(), axis), std::move(out)};
    }

    //! \brief `x · x`, by the type's own product (a complex one's C Annex G, as the library's `*`).
    template <typename T>
    [[nodiscard]] T square(const T& x)
    {
      return static_cast<T>(x * x);
    }

    //! \brief A NaN? Never for a scalar without one.
    template <typename T>
    [[nodiscard]] bool is_nan(const T& /*x*/)
    {
      return false;
    }

    //! \brief A NaN?
    [[nodiscard]] inline bool is_nan(double x)
    {
      return std::isnan(x);
    }

#if defined(__GNUC__)
    //! \brief The lanes where \p x is greater than \p best (all ones there, zeros elsewhere).
    template <typename V,
              typename T>
    [[nodiscard]] auto better(const V& x,
                              const V& best,
                              const std::greater<T>& /*larger*/)
    {
      return x > best;
    }

    //! \brief The lanes where \p x is less than \p best.
    template <typename V,
              typename T>
    [[nodiscard]] auto better(const V& x,
                              const V& best,
                              const std::less<T>& /*smaller*/)
    {
      return x < best;
    }

    //! \brief A lane differs from another only when unordered: \p a against itself flags the NaN lanes.
    template <typename V>
    [[nodiscard]] auto unordered(const V& a,
                                 const V& b)
    {
      return a != b;
    }

#endif

    /*!
     * \brief The largest (\p larger is `std::greater`) or smallest of \p count contiguous elements, or a NaN if
     *        one is among them, as NumPy's: the comparisons run without a branch over native vectors (or eight
     *        scalar lanes), each lane its own running best, and the NaN is tracked apart.
     */
    template <typename T,
              typename Order>
    [[nodiscard]] T contiguous_extreme(const T*     first,
                                       std::size_t  count,
                                       const Order& larger)
    {
      T              result   {first[0]};
      bool           nan      {false};
      std::size_t    done     {0};
#if defined(__GNUC__)
      using lane = typename native_vector<T>::type;
      constexpr std::size_t width {native_vector_bytes / sizeof(T)};
      constexpr std::size_t lanes {4}; // independent vectors, so no comparison waits on the one before it
      if (count >= width * lanes) {
        using mask = decltype(lane {} > lane {}); // all ones in a lane where a comparison holds
        std::array<lane, lanes> best {};
        std::array<mask, lanes> nans {};
        for (std::size_t l {0}; l < lanes; ++l) {
          std::memcpy(&best[l], first + (l * width), sizeof(lane));
        }
        done = count - (count % (width * lanes));
        for (std::size_t i {0}; i < done; i += width * lanes) {
#  pragma GCC unroll 4 // the lanes in registers: GCC keeps a rolled array of vectors in memory
          for (std::size_t l {0}; l < lanes; ++l) {
            lane x;
            std::memcpy(&x, first + i + (l * width), sizeof(lane));
            // Blended through the comparison's all-ones / all-zeros mask: both compilers keep it in vector
            // registers (GCC's `?:` on vectors does not).
            const mask take {better(x, best[l], larger)};
            best[l]  = std::bit_cast<lane>((std::bit_cast<mask>(x) & take) | (std::bit_cast<mask>(best[l]) & ~take));
            nans[l] |= unordered(x, x);
          }
        }
        for (std::size_t l {0}; l < lanes; ++l) {
          for (std::size_t k {0}; k < width; ++k) {
            nan    = nan || nans[l][k] != 0;
            result = larger(best[l][k], result) ? best[l][k] : result;
          }
        }
      }
#endif
      std::array<T, 8>        best {};
      std::array<unsigned, 8> nans {};
      best.fill(result);
      for (std::size_t i {done}; i < count; ++i) {
        const T x {first[i]};
        nans[i % 8] |= static_cast<unsigned>(std::isnan(x));
        best[i % 8]  = larger(x, best[i % 8]) ? x : best[i % 8];
      }
      for (std::size_t l {0}; l < 8; ++l) {
        nan    = nan || nans[l] != 0U;
        result = larger(best[l], result) ? best[l] : result;
      }
      return nan ? std::numeric_limits<T>::quiet_NaN() : result;
    }

    //! \brief The largest (\p larger is `std::greater`) or smallest element of a line.
    template <typename T,
              typename Order>
    [[nodiscard]] T extreme(const T*     first,
                            std::size_t  count,
                            std::size_t  stride,
                            const Order& larger)
    {
      if (count == 0) {
        throw dimension_error("a zero-size reduction to a maximum or a minimum has no identity");
      }
      if constexpr (std::is_floating_point_v<T>) {
        if (stride == 1) {
          // A contiguous line: the comparison without a branch (so it vectorises) and the NaN tracked apart,
          // a NaN anywhere is the answer, as NumPy's.
          return settled_zero(contiguous_extreme(first, count, larger), first, count, 1, larger);
        }
      }
      T best {first[0]};
      for (std::size_t i {1}; i < count; ++i) {
        // A NaN wins, as in NumPy: every comparison with it is false, so a NaN candidate is taken apart, and
        // once taken nothing compares larger than it.
        const T& candidate {first[i * stride]};
        if (is_nan(candidate) || larger(candidate, best)) {
          best = candidate;
        }
      }
      return settled_zero(best, first, count, stride, larger);
    }

    /*!
     * \brief The largest (\p larger is `std::greater`) or smallest elements along \p axis. Along an inner axis
     *        each line is contiguous (\ref extreme); along an outer one the slabs are compared element by element,
     *        so the loop runs over contiguous memory rather than down a column.
     */
    template <typename T,
              typename Order>
    [[nodiscard]] tensor<T> extremes_along(const tensor<T>& t,
                                           std::size_t      axis,
                                           const Order&     larger)
    {
      const axis_walk walk {walk_along(t.shape(), axis)};
      // An empty axis has no maximum even when the result is empty too (NumPy refuses both).
      if (walk.extent == 0) {
        throw dimension_error("a zero-size reduction to a maximum or a minimum has no identity");
      }
      if (!std::is_floating_point_v<T> || walk.inner == 1) {
        return reduce_along(t, axis, [&larger](const T* first, std::size_t count, std::size_t stride) {
                              return extreme(first, count, stride, larger);
                            });
      }
      buffer<T>             out(walk.outer * walk.inner);
      std::vector<unsigned> nan(walk.inner);
      for (std::size_t o {0}; o < walk.outer; ++o) {
        const T* const slab {t.data().data() + (o * walk.extent * walk.inner)};
        T* const       best {out.data() + (o * walk.inner)};
        std::copy(slab, slab + walk.inner, best);
        std::fill(nan.begin(), nan.end(), 0U);
        for (std::size_t e {0}; e < walk.extent; ++e) {
          const T* const row {slab + (e * walk.inner)};
          for (std::size_t i {0}; i < walk.inner; ++i) {
            const T x {row[i]};
            nan[i]  |= static_cast<unsigned>(std::isnan(x));
            best[i]  = larger(x, best[i]) ? x : best[i];
          }
        }
        for (std::size_t i {0}; i < walk.inner; ++i) {
          best[i] = settled_zero(best[i], slab + i, walk.extent, walk.inner, larger);
          if (nan[i] != 0U) {
            best[i] = std::numeric_limits<T>::quiet_NaN();
          }
        }
      }
      return tensor<T> {without_axis(t.shape(), axis), std::move(out)};
    }
  } // namespace detail

  /*!
   * \brief The shape two shapes broadcast to, as NumPy's `broadcast_shapes`.
   * \throws dimension_error If an axis has two extents, neither of them 1.
   */
  [[nodiscard]] inline std::vector<std::size_t> broadcast_shape(const std::vector<std::size_t>& a,
                                                                const std::vector<std::size_t>& b)
  {
    const std::size_t        rank {std::max(a.size(), b.size())};
    std::vector<std::size_t> out(rank);
    for (std::size_t i {0}; i < rank; ++i) {
      const std::size_t x {i < rank - a.size() ? 1 : a[i - (rank - a.size())]};
      const std::size_t y {i < rank - b.size() ? 1 : b[i - (rank - b.size())]};
      if (x != y && x != 1 && y != 1) {
        throw dimension_error("operands could not be broadcast together with shapes " + detail::shape_text(a) + " "
                              + detail::shape_text(b));
      }
      out[i] = x == 1 ? y : x;
    }
    return out;
  }

  namespace detail {
    /*!
     * \brief `f(x)` for every element of \p t, by chunks of \ref elementwise_chunk among the threads (as
     *        \ref broadcast_apply's same-shape pairs): for a stateless \p f, the library's element-wise functions.
     */
    template <typename T,
              typename F>
    [[nodiscard]] auto parallel_map(const tensor<T>& t,
                                    const F&         f) -> tensor<std::decay_t<decltype(f(std::declval<const T&>()))>>
    {
      using R = std::decay_t<decltype(f(std::declval<const T&>()))>;
      buffer<R>         out(t.size());
      const std::size_t chunks {(t.size() + elementwise_chunk - 1) / elementwise_chunk};
      parallel_for(chunks, elementwise_chunk, [&](std::size_t c) {
                     const std::size_t first {c * elementwise_chunk};
                     const std::size_t last {std::min(first + elementwise_chunk, t.size())};
                     std::transform(t.data().begin() + static_cast<std::ptrdiff_t>(first), t.data().begin() + static_cast<std::ptrdiff_t>(last),
                                    out.begin() + static_cast<std::ptrdiff_t>(first), f);
                   });
      return tensor<R> {t.shape(), std::move(out)};
    }
  } // namespace detail

  /*!
   * \brief `op(a, b)` element by element, the operands broadcast against each other.
   * \throws dimension_error If the shapes do not broadcast.
   */
  template <typename T,
            typename Op>
  [[nodiscard]] tensor<T> broadcast_apply(const tensor<T>& a,
                                          const tensor<T>& b,
                                          const Op&        op)
  {
    if (a.shape() == b.shape()) {
      buffer<T>         out(a.size());
      const std::size_t chunks {(a.size() + detail::elementwise_chunk - 1) / detail::elementwise_chunk};
      detail::parallel_for(chunks, detail::elementwise_chunk, [&](std::size_t c) {
                             const std::size_t first {c * detail::elementwise_chunk};
                             const std::size_t last {std::min(first + detail::elementwise_chunk, a.size())};
                             std::transform(a.data().begin() + static_cast<std::ptrdiff_t>(first), a.data().begin() + static_cast<std::ptrdiff_t>(last),
                                            b.data().begin() + static_cast<std::ptrdiff_t>(first), out.begin() + static_cast<std::ptrdiff_t>(first), op);
                           });
      return tensor<T> {a.shape(), std::move(out)};
    }
    std::vector<std::size_t>       shape {broadcast_shape(a.shape(), b.shape())};
    const std::size_t              rank  {shape.size()};
    const std::vector<std::size_t> sa    {detail::broadcast_strides(a.shape(), rank)};
    const std::vector<std::size_t> sb    {detail::broadcast_strides(b.shape(), rank)};
    const std::size_t              count {detail::element_count(shape.begin(), shape.end())};
    buffer<T>                      out(count); // every element written below
    if (count == 0) {
      return tensor<T> {std::move(shape), std::move(out)};
    }
    // The innermost axis runs as one loop, each operand stepping 1 or 0 (a broadcast one) along it, so a
    // scalar or a row against a matrix is a plain vector loop; the odometer turns only between rows. The rows go
    // by tasks, each task's odometer set from its first row.
    const std::size_t inner          {shape[rank - 1]};
    const std::size_t step_a         {sa[rank - 1]};
    const std::size_t step_b         {sb[rank - 1]};
    const std::size_t rows           {count / inner};
    const std::size_t rows_per_task  {std::max<std::size_t>(1, detail::elementwise_chunk / inner)};
    detail::parallel_for((rows + rows_per_task - 1) / rows_per_task, rows_per_task * inner, [&](std::size_t task) {
                           const std::size_t        first {task * rows_per_task};
                           const std::size_t        last {std::min(first + rows_per_task, rows)};
                           std::vector<std::size_t> index(rank, 0);
                           std::size_t              ia {0};
                           std::size_t              ib {0};
                           for (std::size_t axis {rank - 1}, rest {first}; axis-- > 0;) { // the first row's position
                             index[axis]  = rest % shape[axis];
                             rest        /= shape[axis];
                             ia          += index[axis] * sa[axis];
                             ib          += index[axis] * sb[axis];
                           }
                           for (std::size_t row {first}; row < last; ++row) {
                             const T* const pa {a.data().data() + ia};
                             const T* const pb {b.data().data() + ib};
                             T* const       po {out.data() + (row * inner)};
                             if (step_a == 1 && step_b == 1) {
                               for (std::size_t i {0}; i < inner; ++i) {
                                 po[i] = op(pa[i], pb[i]);
                               }
                             }
                             else if (step_a == 1) {
                               const T y {pb[0]};
                               for (std::size_t i {0}; i < inner; ++i) {
                                 po[i] = op(pa[i], y);
                               }
                             }
                             else if (step_b == 1) {
                               const T x {pa[0]};
                               for (std::size_t i {0}; i < inner; ++i) {
                                 po[i] = op(x, pb[i]);
                               }
                             }
                             else {
                               std::fill(po, po + inner, op(pa[0], pb[0]));
                             }
                             for (std::size_t axis {rank - 1}; axis-- > 0;) { // odometer over the outer axes: carry on wrap
                               ++index[axis];
                               ia += sa[axis];
                               ib += sb[axis];
                               if (index[axis] < shape[axis]) {
                                 break;
                               }
                               ia          -= sa[axis] * shape[axis];
                               ib          -= sb[axis] * shape[axis];
                               index[axis]  = 0;
                             }
                           }
                         });
    return tensor<T> {std::move(shape), std::move(out)};
  }

  //! \brief Which of NumPy's complex loops fuse their multiply-adds: its product's vector loop, its product's scalar
  //!        loop (\ref detail::numpy_loop_is_scalar) and its quotient's. They follow how its wheels were built, not
  //!        only the processor, so the binding measures them on the NumPy it runs beside.
  struct numpy_loops {
    bool products;        //!< The product's vector loop.
    bool scalar_products; //!< The product's scalar loop.
    bool quotients;       //!< The quotient (Smith's algorithm, a scalar loop).
  };

  //! \brief NumPy's loops as its current PyPI wheels build them on this processor (\ref detail::numpy_fuses_products,
  //!        \ref detail::numpy_fuses_scalar_products, \ref detail::numpy_fuses_quotients).
  [[nodiscard]] inline numpy_loops numpy_wheel_loops() noexcept
  {
    return numpy_loops {.products        = detail::numpy_fuses_products(),
                        .scalar_products = detail::numpy_fuses_scalar_products(),
                        .quotients       = detail::numpy_fuses_quotients()};
  }

  /*!
   * \brief `a · b` of complex tensors, broadcast, each product as NumPy's array loop computes it
   *        (\ref detail::numpy_multiply, fused as \p loops says): the binding's `*`, where `std::complex`'s operator
   *        (this library's `*`) recovers infinities NumPy's makes NaN.
   */
  template <detail::split_complex C>
  [[nodiscard]] tensor<C> numpy_multiply(const tensor<C>&   a,
                                         const tensor<C>&   b,
                                         const numpy_loops& loops = numpy_wheel_loops())
  {
    const bool fused {detail::numpy_loop_is_scalar(a, b) ? loops.scalar_products : loops.products};
    return broadcast_apply(a, b, [fused](const C& x, const C& y) { return detail::numpy_multiply(x, y, fused); });
  }

  //! \brief `a / b` of complex tensors, broadcast, each quotient as NumPy's array loop computes it
  //!        (\ref detail::numpy_divide, fused as \p loops says): the binding's `/`.
  template <detail::split_complex C>
  [[nodiscard]] tensor<C> numpy_divide(const tensor<C>&   a,
                                       const tensor<C>&   b,
                                       const numpy_loops& loops = numpy_wheel_loops())
  {
    const bool fused {loops.quotients};
    return broadcast_apply(a, b, [fused](const C& x, const C& y) { return detail::numpy_divide(x, y, fused); });
  }

  //! \brief `|x|` of every element of a complex tensor as NumPy's `absolute` computes it
  //!        (\ref detail::numpy_absolute), fused as \p loops says of its vector loops.
  template <detail::split_complex C>
  [[nodiscard]] tensor<typename C::value_type> numpy_absolute(const tensor<C>&   a,
                                                              const numpy_loops& loops = numpy_wheel_loops())
  {
    const bool fused {loops.products};
    return detail::parallel_map(a, [fused](const C& x) { return detail::numpy_absolute(x, fused); });
  }

  //! \brief `x ** 2` of a complex tensor as NumPy's `square` computes it (the loop of an array to the Python integer
  //!        2): its product's vector loop, fused as \p loops says of it.
  template <detail::split_complex C>
  [[nodiscard]] tensor<C> numpy_square(const tensor<C>&   a,
                                       const numpy_loops& loops = numpy_wheel_loops())
  {
    const bool fused {loops.products};
    return detail::parallel_map(a, [fused](const C& x) { return detail::numpy_multiply(x, x, fused); });
  }

  //! \brief `x ** -1` of a complex tensor as NumPy's `reciprocal` computes it (the loop of an array to the Python
  //!        integer -1, \ref detail::numpy_reciprocal), fused as \p loops says of its scalar loops.
  template <detail::split_complex C>
  [[nodiscard]] tensor<C> numpy_reciprocal(const tensor<C>&   a,
                                           const numpy_loops& loops = numpy_wheel_loops())
  {
    const bool fused {loops.quotients};
    return detail::parallel_map(a, [fused](const C& x) { return detail::numpy_reciprocal(x, fused); });
  }

  //! \brief `a ** b` of complex tensors, broadcast, as NumPy's `power` computes each (\ref detail::numpy_power), fused
  //!        as \p loops says of its scalar loops.
  template <detail::split_complex C>
  [[nodiscard]] tensor<C> numpy_power(const tensor<C>&   a,
                                      const tensor<C>&   b,
                                      const numpy_loops& loops = numpy_wheel_loops())
  {
    const bool fused {loops.quotients};
    return broadcast_apply(a, b, [fused](const C& x, const C& y) { return detail::numpy_power(x, y, fused); });
  }

  //! \brief `exp` of every element: `std::exp` of a real one, \ref detail::complex_exp of a complex one.
  template <typename T>
  [[nodiscard]] tensor<T> exp(const tensor<T>& a)
  {
    return detail::parallel_map(a, [](const T& x) { return detail::elementary_exp(x); });
  }

  //! \brief `log` of every element: `std::log` of a real one, \ref detail::complex_log of a complex one.
  template <typename T>
  [[nodiscard]] tensor<T> log(const tensor<T>& a)
  {
    return detail::parallel_map(a, [](const T& x) { return detail::elementary_log(x); });
  }

  //! \brief `sin` of every element: `std::sin` of a real one, \ref detail::complex_sin of a complex one.
  template <typename T>
  [[nodiscard]] tensor<T> sin(const tensor<T>& a)
  {
    return detail::parallel_map(a, [](const T& x) { return detail::elementary_sin(x); });
  }

  //! \brief `cos` of every element: `std::cos` of a real one, \ref detail::complex_cos of a complex one.
  template <typename T>
  [[nodiscard]] tensor<T> cos(const tensor<T>& a)
  {
    return detail::parallel_map(a, [](const T& x) { return detail::elementary_cos(x); });
  }

  //! \brief `tan` of every element: `std::tan` of a real one, \ref detail::complex_tan of a complex one.
  template <typename T>
  [[nodiscard]] tensor<T> tan(const tensor<T>& a)
  {
    return detail::parallel_map(a, [](const T& x) { return detail::elementary_tan(x); });
  }

  //! \brief `tanh` of every element: `std::tanh` of a real one, \ref detail::complex_tanh of a complex one.
  template <typename T>
  [[nodiscard]] tensor<T> tanh(const tensor<T>& a)
  {
    return detail::parallel_map(a, [](const T& x) { return detail::elementary_tanh(x); });
  }

  //! \brief The square root of every element: `std::sqrt` of a real one, \ref detail::complex_sqrt of a complex one.
  template <typename T>
  [[nodiscard]] tensor<T> sqrt(const tensor<T>& a)
  {
    return detail::parallel_map(a, [](const T& x) { return detail::root(x); });
  }

  //! \brief Element-wise sum, broadcast. \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> operator+(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, std::plus<T> {});
  }

  //! \brief Element-wise difference, broadcast. \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> operator-(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, std::minus<T> {});
  }

  //! \brief Element-wise product, broadcast. \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> operator*(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, std::multiplies<T> {});
  }

  //! \brief Element-wise quotient, broadcast. \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> operator/(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, std::divides<T> {});
  }

  //! \brief Element-wise negation.
  template <typename T>
  [[nodiscard]] tensor<T> operator-(const tensor<T>& a)
  {
    return detail::parallel_map(a, std::negate<T> {});
  }

  //! \brief `f(x)` for every element, in a tensor of the same shape and of `f`'s result type.
  template <typename T,
            typename F>
  [[nodiscard]] auto map(const tensor<T>& t,
                         const F&         f) -> tensor<std::decay_t<decltype(f(std::declval<const T&>()))>>
  {
    using R = std::decay_t<decltype(f(std::declval<const T&>()))>;
    buffer<R> out(t.size());
    std::transform(t.data().begin(), t.data().end(), out.begin(), f);
    return tensor<R> {t.shape(), std::move(out)};
  }

  /*!
   * \brief `a` to the power `b` element by element, broadcast, as `std::pow`; a single exponent of 2, 0.5 or
   *        -1 as the library's own `x * x`, square root (\ref detail::complex_sqrt for a complex one) and `1 / x`
   *        (exact where `pow` rounds: a complex square has no stray real part). A complex one keeps C's Annex G, as
   *        the operators do: NumPy's loops are \ref numpy_square, \ref numpy_reciprocal and \ref numpy_power.
   */
  template <typename T>
  [[nodiscard]] tensor<T> pow(const tensor<T>& a,
                              const tensor<T>& b)
  {
    if constexpr (!std::is_integral_v<T>) {
      if (b.size() == 1 && b.shape().size() <= a.shape().size()) {
        const T y {b.data()[0]};
        if (y == T(2)) {
          return detail::parallel_map(a, [](const T& x) { return detail::square(x); });
        }
        if (y == T(0.5)) {
          return detail::parallel_map(a, [](const T& x) { return detail::root(x); });
        }
        if (y == T(-1)) {
          return detail::parallel_map(a, [](const T& x) { return static_cast<T>(T(1) / x); });
        }
      }
    }
    return broadcast_apply(a, b, [](const T& x, const T& y) { return static_cast<T>(std::pow(x, y)); });
  }

  //! \brief The sum of every element, by pairwise halving (0 for an empty tensor).
  template <typename T>
  [[nodiscard]] T sum(const tensor<T>& t)
  {
    return detail::pairwise_sum(t.data().data(), t.size(), 1);
  }

  //! \brief The sums along \p axis, by pairwise halving. \throws dimension_error If \p axis is not an axis.
  template <typename T>
  [[nodiscard]] tensor<T> sum(const tensor<T>& t,
                              std::size_t      axis)
  {
    const detail::axis_walk walk {detail::walk_along(t.shape(), axis)};
    buffer<T>               out(walk.outer * walk.inner);
    detail::pairwise_sums(t.data().data(), walk.outer, walk.extent, walk.inner, out.data());
    return tensor<T> {detail::without_axis(t.shape(), axis), std::move(out)};
  }

  /*!
   * \brief The mean of every element. An empty tensor's mean is NaN, as NumPy's (which also warns).
   */
  template <typename T>
  [[nodiscard]] T mean(const tensor<T>& t)
  {
    return detail::divided_by_count(sum(t), t.size());
  }

  //! \brief The means along \p axis. \throws dimension_error If \p axis is not an axis.
  template <typename T>
  [[nodiscard]] tensor<T> mean(const tensor<T>& t,
                               std::size_t      axis)
  {
    const detail::axis_walk walk  {detail::walk_along(t.shape(), axis)};
    buffer<T>               out(walk.outer * walk.inner);
    detail::pairwise_sums(t.data().data(), walk.outer, walk.extent, walk.inner, out.data());
    for (T& x : out) {
      x = detail::divided_by_count(x, walk.extent);
    }
    return tensor<T> {detail::without_axis(t.shape(), axis), std::move(out)};
  }

  //! \brief The largest element (a NaN, if there is one). \throws dimension_error If \p t is empty.
  template <std::totally_ordered T>
  [[nodiscard]] T max(const tensor<T>& t)
  {
    return detail::extreme(t.data().data(), t.size(), 1, std::greater<T> {});
  }

  //! \brief The smallest element (a NaN, if there is one). \throws dimension_error If \p t is empty.
  template <std::totally_ordered T>
  [[nodiscard]] T min(const tensor<T>& t)
  {
    return detail::extreme(t.data().data(), t.size(), 1, std::less<T> {});
  }

  //! \brief The largest elements along \p axis. \throws dimension_error If \p axis is not an axis or is empty.
  template <std::totally_ordered T>
  [[nodiscard]] tensor<T> max(const tensor<T>& t,
                              std::size_t      axis)
  {
    return detail::extremes_along(t, axis, std::greater<T> {});
  }

  //! \brief The smallest elements along \p axis. \throws dimension_error If \p axis is not an axis or is empty.
  template <std::totally_ordered T>
  [[nodiscard]] tensor<T> min(const tensor<T>& t,
                              std::size_t      axis)
  {
    return detail::extremes_along(t, axis, std::less<T> {});
  }

  /*!
   * \brief The same elements in the same order under another shape.
   * \throws dimension_error If the shapes hold different counts of elements.
   */
  template <typename T>
  [[nodiscard]] tensor<T> reshape(const tensor<T>&         t,
                                  std::vector<std::size_t> shape)
  {
    std::size_t count {1};
    for (const std::size_t extent : shape) {
      count = detail::checked_mul(count, extent);
    }
    if (count != t.size()) {
      throw dimension_error("cannot reshape a tensor of size " + std::to_string(t.size()) + " into shape "
                            + detail::shape_text(shape));
    }
    return tensor<T> {std::move(shape), t.data()};
  }

  //! \brief A tensor of \p shape with every element \p value.
  template <typename T>
  [[nodiscard]] tensor<T> full(std::vector<std::size_t> shape,
                               const T&                 value)
  {
    std::size_t count {1};
    for (const std::size_t extent : shape) {
      count = detail::checked_mul(count, extent);
    }
    return tensor<T> {std::move(shape), buffer<T>(count, value)};
  }

  //! \brief A tensor of \p shape full of zeros.
  template <typename T>
  [[nodiscard]] tensor<T> zeros(std::vector<std::size_t> shape)
  {
    return full<T>(std::move(shape), T {});
  }

  //! \brief A tensor of \p shape full of ones.
  template <typename T>
  [[nodiscard]] tensor<T> ones(std::vector<std::size_t> shape)
  {
    return full<T>(std::move(shape), static_cast<T>(1));
  }

  //! \brief The `n × n` identity.
  template <typename T>
  [[nodiscard]] tensor<T> eye(std::size_t n)
  {
    buffer<T> out(detail::checked_mul(n, n), T {});
    for (std::size_t i {0}; i < n; ++i) {
      out[(i * n) + i] = static_cast<T>(1);
    }
    return tensor<T> {{n, n}, std::move(out)};
  }

  /*!
   * \brief `start, start + step, …` below \p stop (above it for a negative step), as NumPy's `arange`: its
   *        length is `ceil((stop − start) / step)` and element `i` is `start + i·delta` with
   *        `delta = (start + step) − start`, the step as the first two elements hold it,
   *        rounded once (a fused multiply-add, as NumPy's arm64 builds compute it; one without fusion rounds
   *        the product first and can differ in the last bits).
   * \throws dimension_error If \p step is zero or not finite.
   */
  [[nodiscard]] inline tensor<double> arange(double start,
                                             double stop,
                                             double step)
  {
    if (step == 0.0 || !std::isfinite(step) || !std::isfinite(start) || !std::isfinite(stop)) {
      throw dimension_error("arange needs finite bounds and a finite, non-zero step");
    }
    const double        length            {std::ceil((stop - start) / step)};
    if (!(length < 0x1p64)) { // past std::size_t, or an infinite span: no conversion holds it
      throw dimension_error("arange: the length does not fit in std::size_t");
    }
    const std::size_t   count             {length > 0.0 ? static_cast<std::size_t>(length) : 0};
    buffer<double>      out(count);
    const double        delta             {(start + step) - start};
    for (std::size_t i {0}; i < count; ++i) {
      out[i] = std::fma(static_cast<double>(i), delta, start);
    }
    return tensor<double> {{count}, std::move(out)};
  }

  /*!
   * \brief \p count values evenly spaced from \p start to \p stop, both included, as NumPy's `linspace`:
   *        element `i` is `start + i·step` with `step = (stop − start) / (count − 1)`, and the last is
   *        \p stop exactly.
   */
  [[nodiscard]] inline tensor<double> linspace(double      start,
                                               double      stop,
                                               std::size_t count)
  {
    buffer<double>      out(count);
    const double        step              {count > 1 ? (stop - start) / static_cast<double>(count - 1) : 0.0};
    for (std::size_t i {0}; i < count; ++i) {
      // The product rounded before the sum, as NumPy's linspace computes it in two operations. A compiler
      // may fuse `i * step` into the addition after it (GCC does, across statements, wherever there is a
      // fused multiply-add); a product computed by fma, `i * step - 0`, is exactly the rounded one, signed
      // zero included, and no multiplication is left to fuse.
      const double offset {std::fma(static_cast<double>(i), step, -0.0)};
      out[i] = start + offset;
    }
    if (count > 1) {
      out[count - 1] = stop;
    }
    return tensor<double> {{count}, std::move(out)};
  }
} // namespace numerical

#endif // NUMERICAL_TENSOR_OPS_HPP
