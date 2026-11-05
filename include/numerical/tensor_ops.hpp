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
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "complex_math.hpp"
#include "simd.hpp"
#include "tensor.hpp"

namespace numerical {

  namespace detail {



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

    //! \brief `a · b` of complex numbers as NumPy's reduction loop computes it where MSVC contracts it (its Windows
    //!        wheels from 2.5): `ai·bi` and `ar·bi` rounded, then `ar·br − ai·bi` and `ai·br + ar·bi` each one fused
    //!        multiply-add (the imaginary part's other product fused than in \ref numpy_multiply).
    template <split_complex C>
    [[nodiscard]] C numpy_multiply_late(const C& a,
                                        const C& b)
    {
      using R = typename C::value_type;
      const R ii {a.imag() * b.imag()};
      const R ri {a.real() * b.imag()};
      return C {std::fma(a.real(), b.real(), -ii), std::fma(a.imag(), b.real(), ri)};
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

    //! \brief Whether NumPy's complex product fuses its multiply-adds in its reduction loop (the product's loop whose
    //!        output is its own first operand: `numpy.prod` along an axis nothing follows, `numpy.cumprod` past two
    //!        elements), as its PyPI wheels are built: a scalar loop, contracted as \ref numpy_fuses_scalar_products
    //!        says (GCC's on Linux rounds each product, though its vector loop fuses).
    [[nodiscard]] inline bool numpy_fuses_reductions() noexcept
    {
      return numpy_fuses_scalar_products();
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
      const axis_walk walk {walk_along(t.shape(), axis)};
      buffer<T>       out(walk.outer * walk.inner);
      parallel_for(walk.outer * walk.inner, walk.extent, [&](std::size_t at) { // each line its own, on whichever thread
                     const std::size_t o {at / walk.inner};
                     const std::size_t i {at % walk.inner};
                     out[at] = reduce(t.data().data() + (o * walk.extent * walk.inner) + i, walk.extent, walk.inner);
                   });
      return tensor<T> {without_axis(t.shape(), axis), std::move(out)};
    }

    //! \brief The columns of a band that one task of \ref fold_along and \ref scan_along walks row by row.
    inline constexpr std::size_t band_columns {256};

    /*!
     * \brief Folds each line of \p t along \p axis in order, `step(running, x)` from \p init: a tensor of the result
     *        type `R`, the axis taken out. Where the lines are columns (something follows the axis), each task takes a
     *        band of up to \ref band_columns of them and walks it row by row, contiguously; each line's steps are the
     *        same, in the same order, either way.
     */
    template <typename R,
              typename T,
              typename Step>
    [[nodiscard]] tensor<R> fold_along(const tensor<T>& t,
                                       std::size_t      axis,
                                       const R&         init,
                                       const Step&      step)
    {
      const axis_walk walk {walk_along(t.shape(), axis)};
      buffer<R>       out(walk.outer * walk.inner);
      const T* const  in   {t.data().data()};
      if (walk.inner == 1) {
        parallel_for(walk.outer, walk.extent, [&](std::size_t o) {
                       R running {init};
                       for (std::size_t k {0}; k < walk.extent; ++k) {
                         running = step(running, in[(o * walk.extent) + k]);
                       }
                       out[o] = running;
                     });
        return tensor<R> {without_axis(t.shape(), axis), std::move(out)};
      }
      const std::size_t bands {(walk.inner + band_columns - 1) / band_columns};
      parallel_for(walk.outer * bands, walk.extent * band_columns, [&](std::size_t at) {
                     const std::size_t o     {at / bands};
                     const std::size_t first {(at % bands) * band_columns};
                     const std::size_t width {std::min(band_columns, walk.inner - first)};
                     const std::size_t row   {(o * walk.inner) + first}; // indexed: a buffer of bools has no data()
                     for (std::size_t w {0}; w < width; ++w) {
                       out[row + w] = init;
                     }
                     for (std::size_t k {0}; k < walk.extent; ++k) {
                       const T* const x {in + (((o * walk.extent) + k) * walk.inner) + first};
                       for (std::size_t w {0}; w < width; ++w) {
                         out[row + w] = step(out[row + w], x[w]);
                       }
                     }
                   });
      return tensor<R> {without_axis(t.shape(), axis), std::move(out)};
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

#ifdef NUMERICAL_NEON_COMPILED
    //! \brief arm64's lane-wise `FMAX`: a NaN where one is met, +0 over −0.
    [[nodiscard]] inline float64x2_t lane_extreme(float64x2_t x,
                                                  float64x2_t y,
                                                  const std::greater<double>& /*larger*/) noexcept
    {
      return vmaxq_f64(x, y);
    }

    //! \brief arm64's lane-wise `FMIN`: a NaN where one is met, −0 under +0.
    [[nodiscard]] inline float64x2_t lane_extreme(float64x2_t x,
                                                  float64x2_t y,
                                                  const std::less<double>& /*smaller*/) noexcept
    {
      return vminq_f64(x, y);
    }

    //! \brief \ref lane_extreme for `float`'s lanes.
    [[nodiscard]] inline float32x4_t lane_extreme(float32x4_t x,
                                                  float32x4_t y,
                                                  const std::greater<float>& /*larger*/) noexcept
    {
      return vmaxq_f32(x, y);
    }

    //! \brief \ref lane_extreme for `float`'s lanes.
    [[nodiscard]] inline float32x4_t lane_extreme(float32x4_t x,
                                                  float32x4_t y,
                                                  const std::less<float>& /*smaller*/) noexcept
    {
      return vminq_f32(x, y);
    }

    //! \brief The extreme across a vector's lanes (`FMAXV`, a NaN where one is).
    [[nodiscard]] inline double across_lanes(float64x2_t x,
                                             const std::greater<double>& /*larger*/) noexcept
    {
      return vmaxvq_f64(x);
    }

    //! \brief The extreme across a vector's lanes (`FMINV`).
    [[nodiscard]] inline double across_lanes(float64x2_t x,
                                             const std::less<double>& /*smaller*/) noexcept
    {
      return vminvq_f64(x);
    }

    //! \brief \ref across_lanes for `float`'s lanes.
    [[nodiscard]] inline float across_lanes(float32x4_t x,
                                            const std::greater<float>& /*larger*/) noexcept
    {
      return vmaxvq_f32(x);
    }

    //! \brief \ref across_lanes for `float`'s lanes.
    [[nodiscard]] inline float across_lanes(float32x4_t x,
                                            const std::less<float>& /*smaller*/) noexcept
    {
      return vminvq_f32(x);
    }

    /*!
     * \brief The largest (\p larger is `std::greater`) or smallest of the whole blocks of four registers of \p count
     *        elements from \p first, into \p result and \p nan (set: whether a NaN was met), by arm64's `FMAX` and
     *        `FMIN`: they take a NaN where one is met and order the zeros as IEEE 754-2019's maximum and minimum, so
     *        the NaN needs no comparison of its own (a blend of compared lanes took twice NumPy's time).
     *        \return The elements done.
     */
    template <typename T,
              typename Order>
    [[nodiscard]] std::size_t neon_extreme(const T*     first,
                                           std::size_t  count,
                                           T&           result,
                                           bool&        nan,
                                           const Order& larger) noexcept
    {
      constexpr std::size_t width {16 / sizeof(T)}; // a register's lanes
      constexpr std::size_t block {4 * width};
      const std::size_t     done  {count - (count % block)};
      if (done == 0) {
        return 0;
      }
      auto b0 {neon_load(first)};
      auto b1 {neon_load(first + width)};
      auto b2 {neon_load(first + (2 * width))};
      auto b3 {neon_load(first + (3 * width))};
      for (std::size_t i {block}; i < done; i += block) {
        b0 = lane_extreme(b0, neon_load(first + i), larger);
        b1 = lane_extreme(b1, neon_load(first + i + width), larger);
        b2 = lane_extreme(b2, neon_load(first + i + (2 * width)), larger);
        b3 = lane_extreme(b3, neon_load(first + i + (3 * width)), larger);
      }
      const T value {across_lanes(lane_extreme(lane_extreme(b0, b1, larger), lane_extreme(b2, b3, larger), larger), larger)};
      nan    = std::isnan(value);
      result = larger(value, result) ? value : result;
      return done;
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
#if defined(NUMERICAL_NEON_COMPILED)
      done = neon_extreme(first, count, result, nan, larger);
#else
#  ifdef NUMERICAL_X86_COMPILED
      bool           wide     {false}; // the leading elements done by AVX2's wider vectors
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
        // The processor's AVX2, where the baseline build's vectors hold half as many: the same answer, whatever the
        // width (each lane its own best, a NaN flagged apart, a zero's sign settled after).
        if (const gemm_path path {gemm_path_current()}; path == gemm_path::avx2 || path == gemm_path::avx512) {
          done = avx2_extreme<std::is_same_v<Order, std::greater<T>>>(first, count, result, nan);
          wide = true;
        }
      }
#  endif
#  if defined(__GNUC__)
      using lane = typename native_vector<T>::type;
      constexpr std::size_t width {native_vector_bytes / sizeof(T)};
      constexpr std::size_t lanes {4}; // independent vectors, so no comparison waits on the one before it
#    ifdef NUMERICAL_X86_COMPILED
      if (!wide && count >= width * lanes) {
#    else
      if (count >= width * lanes) {
#    endif
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
#  endif
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

    /*!
     * \brief \ref contiguous_extreme of \p count elements on the threads: runs of \ref pairwise_leaf each its own,
     *        then their extremes'. Exact whatever the runs (a maximum rounds nothing), so the same answer on any
     *        number of threads. Four runs or fewer: on the caller, as \ref parallel_pairwise_sum.
     */
    template <typename T,
              typename Order>
    [[nodiscard]] T parallel_extreme(const T*     first,
                                     std::size_t  count,
                                     const Order& larger)
    {
      if (count <= 4 * pairwise_leaf) {
        return contiguous_extreme(first, count, larger);
      }
      const std::size_t runs {(count + pairwise_leaf - 1) / pairwise_leaf};
      buffer<T>         each(runs);
      parallel_for(runs, pairwise_leaf, [&](std::size_t r) {
                     const std::size_t start {r * pairwise_leaf};
                     each[r] = contiguous_extreme(first + start, std::min(pairwise_leaf, count - start), larger);
                   });
      return contiguous_extreme(each.data(), runs, larger);
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
          return settled_zero(parallel_extreme(first, count, larger), first, count, 1, larger);
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

    //! \brief Whether \ref extremes_along takes the reduction's path: an exact type, or an axis of single elements. A
    //!        function, where the test written in the condition joined a constant to the axis's, which MSVC 19.44
    //!        (Visual Studio 2022's) rejects (C4127, an error under /WX).
    template <typename T>
    [[nodiscard]] constexpr bool by_reduction(std::size_t inner) noexcept
    {
      return !std::is_floating_point_v<T> || inner == 1;
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
      if (by_reduction<T>(walk.inner)) {
        return reduce_along(t, axis, [&larger](const T* first, std::size_t count, std::size_t stride) {
                              return extreme(first, count, stride, larger);
                            });
      }
      // Each slab's runs of extreme_columns columns shared among the threads: each element compared alike by whichever.
      // Each slab a task, compared row by row along its whole width (a strip of its columns at a time read each row in
      // pieces, 2.6 times as long on one arm64 thread): each element compared alike by whichever thread.
      buffer<T>         out(walk.outer * walk.inner);
      const std::size_t inner  {walk.inner};
      const std::size_t extent {walk.extent};
      const T* const    data   {t.data().data()};
      T* const          into   {out.data()};
      parallel_for(walk.outer, extent * inner, [ =, &larger](std::size_t o) {
                     const T* const        slab {data + (o * extent * inner)};
                     T* const              best {into + (o * inner)};
                     std::vector<unsigned> nan(inner, 0U);
                     std::copy(slab, slab + inner, best);
                     for (std::size_t e {0}; e < extent; ++e) {
                       const T* const row {slab + (e * inner)};
                       for (std::size_t i {0}; i < inner; ++i) {
                         const T x {row[i]};
                         nan[i]  |= static_cast<unsigned>(std::isnan(x));
                         best[i]  = larger(x, best[i]) ? x : best[i];
                       }
                     }
                     for (std::size_t i {0}; i < inner; ++i) {
                       best[i] = settled_zero(best[i], slab + i, extent, inner, larger);
                       if (nan[i] != 0U) {
                         best[i] = std::numeric_limits<T>::quiet_NaN();
                       }
                     }
                   });
      return tensor<T> {without_axis(t.shape(), axis), std::move(out)};
    }
  } // namespace detail


  namespace detail {
    /*!
     * \brief `f(x)` for every one of the row-major elements at \p data, of shape \p shape (another array's storage, read
     *        where it lies).
     */
    template <typename T,
              typename F>
    [[nodiscard]] auto parallel_map(const T*                        data,
                                    const std::vector<std::size_t>& shape,
                                    const F&                        f) -> tensor<std::decay_t<decltype(f(std::declval<const T&>()))>>
    {
      using R = std::decay_t<decltype(f(std::declval<const T&>()))>;
      const std::size_t count  {element_count(shape.begin(), shape.end())};
      buffer<R>         out(count);
      const std::size_t chunks {(count + elementwise_chunk - 1) / elementwise_chunk};
      parallel_for(chunks, elementwise_chunk, [&](std::size_t c) {
                     const std::size_t first {c * elementwise_chunk};
                     const std::size_t last {std::min(first + elementwise_chunk, count)};
                     std::transform(data + first, data + last, out.begin() + static_cast<std::ptrdiff_t>(first), f);
                   });
      return tensor<R> {shape, std::move(out)};
    }

    /*!
     * \brief `f(x)` for every element of \p t, by chunks of \ref elementwise_chunk among the threads (as
     *        \ref broadcast_apply's same-shape pairs): for a stateless \p f, the library's element-wise functions.
     */
    template <typename T,
              typename F>
    [[nodiscard]] auto parallel_map(const tensor<T>& t,
                                    const F&         f) -> tensor<std::decay_t<decltype(f(std::declval<const T&>()))>>
    {
      return parallel_map(t.data().data(), t.shape(), f);
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
    return detail::broadcast_into(a.data().data(), a.shape(), b.data().data(), b.shape(), op);
  }

  namespace detail {
    // NumPy's loops over tensors, for the binding beside a NumPy: no part of the library's own algebra, which keeps
    // C's Annex G (the operators, pow).

    //! \brief Which of NumPy's complex loops fuse their multiply-adds: its product's vector loop, its product's scalar
    //!        loop (\ref detail::numpy_loop_is_scalar), its quotient's and its product's reduction loop. They follow how
    //!        its wheels were built, not only the processor, so the binding measures them on the NumPy it runs beside.
    struct numpy_loops {
      bool products;        //!< The product's vector loop.
      bool scalar_products; //!< The product's scalar loop.
      bool quotients;       //!< The quotient (Smith's algorithm, a scalar loop).
      bool reductions;      //!< The product's reduction loop (\ref detail::numpy_fuses_reductions).
      bool late_reductions; //!< Where it fuses, as \ref detail::numpy_multiply_late does (MSVC's contraction).
    };

    //! \brief `a · b` as NumPy's reduction loop computes it, fused as \p loops says.
    template <split_complex C>
    [[nodiscard]] C numpy_reduction_multiply(const C&           a,
                                             const C&           b,
                                             const numpy_loops& loops)
    {
      return loops.reductions && loops.late_reductions ? numpy_multiply_late(a, b)
                                                       : numpy_multiply(a, b, loops.reductions);
    }

    //! \brief NumPy's loops as its current PyPI wheels build them on this processor (\ref detail::numpy_fuses_products,
    //!        \ref detail::numpy_fuses_scalar_products, \ref detail::numpy_fuses_quotients).
    [[nodiscard]] inline numpy_loops numpy_wheel_loops() noexcept
    {
      return numpy_loops {.products        = detail::numpy_fuses_products(),
                          .scalar_products = detail::numpy_fuses_scalar_products(),
                          .quotients       = detail::numpy_fuses_quotients(),
                          .reductions      = detail::numpy_fuses_reductions(),
                          .late_reductions = false};
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

    /*!
     * \brief The products of a complex \p t along \p axis, each line in order from 1 (NumPy's identity, which turns
     *        an infinite element alone into a NaN part), each product as NumPy's loops compute it (\ref detail::numpy_multiply): its reduction loop's where
     *        nothing follows the axis, its vector loop's (between whole slices) where something does; fused as \p loops
     *        says. The binding's `prod`, where `std::complex`'s operator (\ref prod) recovers infinities NumPy's makes
     *        NaN. \throws dimension_error If \p axis is not an axis.
     */
    template <detail::split_complex C>
    [[nodiscard]] tensor<C> numpy_prod(const tensor<C>&   t,
                                       std::size_t        axis,
                                       const numpy_loops& loops = numpy_wheel_loops())
    {
      const bool reduction {walk_along(t.shape(), axis).inner == 1};
      return fold_along(t, axis, C {1}, [reduction, &loops](const C& product, const C& x) {
                          return reduction ? numpy_reduction_multiply(product, x, loops) : numpy_multiply(product, x, loops.products);
                        });
    }

    //! \brief The product of every element of a complex \p t, in order from 1 (NumPy's identity), as NumPy's
    //!        reduction loop computes it (\ref detail::numpy_multiply, fused as \p loops says).
    template <detail::split_complex C>
    [[nodiscard]] C numpy_prod(const tensor<C>&   t,
                               const numpy_loops& loops = numpy_wheel_loops())
    {
      C product {1};
      for (const C& x : t.data()) {
        product = numpy_reduction_multiply(product, x, loops);
      }
      return product;
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
  } // namespace detail

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

  namespace detail {
    //! \brief The element operations of \p T the tensor operators apply: the standard ones.
    template <typename T>
    struct arithmetic {
      using plus       = std::plus<T>;       //!< `a + b`.
      using minus      = std::minus<T>;      //!< `a - b`.
      using multiplies = std::multiplies<T>; //!< `a · b`.
      using negate     = std::negate<T>;     //!< `-a`.
    };

    //! \brief A signed integer's operations through its unsigned type, so that an overflow wraps modulo 2ⁿ as
    //!        NumPy's integers do (two's complement) where the signed operation is undefined.
    template <std::signed_integral T>
    struct arithmetic<T> {
      using unsigned_type = std::make_unsigned_t<decltype(T {} + 0U)>; //!< Wide enough that no promotion is signed.

      //! \brief `a + b`, wrapped.
      struct plus {
        [[nodiscard]] T operator()(T a,
                                   T b) const noexcept
        {
          return static_cast<T>(static_cast<unsigned_type>(a) + static_cast<unsigned_type>(b));
        }
      };

      //! \brief `a - b`, wrapped.
      struct minus {
        [[nodiscard]] T operator()(T a,
                                   T b) const noexcept
        {
          return static_cast<T>(static_cast<unsigned_type>(a) - static_cast<unsigned_type>(b));
        }
      };

      //! \brief `a · b`, wrapped.
      struct multiplies {
        [[nodiscard]] T operator()(T a,
                                   T b) const noexcept
        {
          return static_cast<T>(static_cast<unsigned_type>(a) * static_cast<unsigned_type>(b));
        }
      };

      //! \brief `-a`, wrapped (the most negative value its own negation).
      struct negate {
        [[nodiscard]] T operator()(T a) const noexcept
        {
          return static_cast<T>(unsigned_type {0} - static_cast<unsigned_type>(a));
        }
      };
    };
  } // namespace detail

  //! \brief Element-wise sum, broadcast (a signed integer's wrapped, \ref detail::arithmetic). \throws dimension_error
  //!        If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> operator+(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, typename detail::arithmetic<T>::plus {});
  }

  namespace detail {

    //! \brief The larger of \p a and \p b as IEEE 754-2019's maximum orders them: a NaN wins (\p a's where both are),
    //!        and +0 is over −0 whatever their order (`a + b` of two zeros: −0 only where both are).
    template <std::floating_point T>
    [[nodiscard]] T ieee_maximum(T a,
                                 T b) noexcept
    {
      if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) ? a : b;
      }
      if (a == b) {
        return a == T {0} ? a + b : a;
      }
      return a > b ? a : b;
    }

    //! \brief The smaller of \p a and \p b as IEEE 754-2019's minimum orders them: a NaN wins (\p a's where both
    //!        are), and −0 is under +0 whatever their order (`−(−a − b)` of two zeros: +0 only where both are).
    template <std::floating_point T>
    [[nodiscard]] T ieee_minimum(T a,
                                 T b) noexcept
    {
      if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) ? a : b;
      }
      if (a == b) {
        return a == T {0} ? -(-a - b) : a;
      }
      return a < b ? a : b;
    }
  } // namespace detail

  //! \brief The larger element of \p a and \p b, broadcast, as `numpy.maximum` (\ref detail::ieee_maximum; a NaN
  //!        wins, +0 over −0). \throws dimension_error If the shapes do not broadcast.
  template <std::floating_point T>
  [[nodiscard]] tensor<T> maximum(const tensor<T>& a,
                                  const tensor<T>& b)
  {
    if (b.shape().empty()) { // a number (clip's bound): a map over a, where broadcasting walks both
      const T bound {b.data()[0]};
      return detail::parallel_map(a.data().data(), a.shape(), [bound](T x) { return detail::ieee_maximum(x, bound); });
    }
    return broadcast_apply(a, b, [](T x, T y) { return detail::ieee_maximum(x, y); });
  }

  //! \brief The smaller element of \p a and \p b, broadcast, as `numpy.minimum` (\ref detail::ieee_minimum; a NaN
  //!        wins, −0 under +0). \throws dimension_error If the shapes do not broadcast.
  template <std::floating_point T>
  [[nodiscard]] tensor<T> minimum(const tensor<T>& a,
                                  const tensor<T>& b)
  {
    if (b.shape().empty()) { // a number (clip's bound): a map over a, where broadcasting walks both
      const T bound {b.data()[0]};
      return detail::parallel_map(a.data().data(), a.shape(), [bound](T x) { return detail::ieee_minimum(x, bound); });
    }
    return broadcast_apply(a, b, [](T x, T y) { return detail::ieee_minimum(x, y); });
  }

  namespace detail {
    /*!
     * \brief A real of type \p R rounded to \p decimals places, half to even, as `numpy.round`: `nearbyint(x · 10ᵈ) / 10ᵈ`
     *        (`nearbyint(x / 10⁻ᵈ) · 10⁻ᵈ` for a negative `d`), computed in \p C.
     *
     * An `x` of magnitude 2ᵖ · 10⁻ᵈ or more (p the digits of \p R's significand) has an ulp coarser than the places:
     * its decimal rounding lies within half a place of it, nearer than any other value, so it is kept. Its product by
     * 10ᵈ, which NumPy's computes, can round off its last bit or overflow (`round(1e307, 2)`); below, the product is
     * under 2ᵖ. A finite `x` rounds to ±0 where 10⁻ᵈ overflows \p C.
     */
    template <std::floating_point R,
              std::floating_point C>
    struct decimal_rounding {
      int decimals;   //!< The places; fewer than none round to tens, hundreds and so on.
      C   scale;      //!< 10 to the magnitude of \ref decimals, in \p C.
      C   exact_from; //!< The magnitude from which a value is its own rounding (\ref decimals above 0).

      explicit decimal_rounding(int places) noexcept
        : decimals {places}
          , scale {std::pow(C {10}, static_cast<C>(places < 0 ? -places : places))}
          , exact_from {std::ldexp(C {1}, std::numeric_limits<R>::digits) / scale}
      {}

      [[nodiscard]] R operator()(R x) const noexcept
      {
        const C wide {static_cast<C>(x)};
        if (decimals > 0) {
          return std::fabs(wide) < exact_from ? static_cast<R>(std::nearbyint(wide * scale) / scale) : x;
        }
        if (std::isinf(scale)) {
          return std::isfinite(x) ? std::copysign(R {0}, x) : x;
        }
        return static_cast<R>(std::nearbyint(wide / scale) * scale);
      }
    };

    //! \brief `f(rounding)` with the \ref decimal_rounding of \p R to \p decimals places: in \p R, as NumPy's, but in
    //!        double for a float whose 10ᵈ overflows it (a float under 2²⁴ · 10⁻ᵈ still has places to round there).
    template <std::floating_point R,
              typename F>
    [[nodiscard]] auto with_decimal_rounding(int      decimals,
                                             const F& f)
    {
      if constexpr (std::is_same_v<R, float>) {
        if (decimals > std::numeric_limits<float>::max_exponent10) {
          return f(decimal_rounding<float, double> {decimals});
        }
      }
      return f(decimal_rounding<R, R> {decimals});
    }
  } // namespace detail

  //! \brief Every element rounded to \p decimals places (tens, hundreds for a negative one), half to even, as
  //!        `numpy.round`; a value already exact to those places is kept (\ref detail::decimal_rounding).
  template <std::floating_point T>
  [[nodiscard]] tensor<T> round(const tensor<T>& a,
                                int              decimals = 0)
  {
    return detail::with_decimal_rounding<T>(decimals, [&a](const auto& rounding) { return detail::parallel_map(a, rounding); });
  }

  //! \brief Every element's real and imaginary parts rounded to \p decimals places apart, as `numpy.round` of a
  //!        complex array.
  template <detail::split_complex C>
  [[nodiscard]] tensor<C> round(const tensor<C>& a,
                                int              decimals = 0)
  {
    return detail::with_decimal_rounding<typename C::value_type>(decimals, [&a](const auto& rounding) {
                                                                   return detail::parallel_map(a, [&rounding](const C& z) { return C {rounding(z.real()), rounding(z.imag())}; });
                                                                 });
  }

  //! \brief The larger element of integer tensors \p a and \p b, broadcast, as `numpy.maximum`. \throws dimension_error
  //!        If the shapes do not broadcast.
  template <std::integral T>
  [[nodiscard]] tensor<T> maximum(const tensor<T>& a,
                                  const tensor<T>& b)
  {
    return broadcast_apply(a, b, [](T x, T y) { return std::max(x, y); });
  }

  //! \brief The smaller element of integer tensors \p a and \p b, broadcast, as `numpy.minimum`. \throws dimension_error
  //!        If the shapes do not broadcast.
  template <std::integral T>
  [[nodiscard]] tensor<T> minimum(const tensor<T>& a,
                                  const tensor<T>& b)
  {
    return broadcast_apply(a, b, [](T x, T y) { return std::min(x, y); });
  }

  //! \brief Element-wise difference, broadcast. \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> operator-(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, typename detail::arithmetic<T>::minus {});
  }

  //! \brief Element-wise product, broadcast. \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> operator*(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, typename detail::arithmetic<T>::multiplies {});
  }

  //! \brief Element-wise quotient, broadcast. \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> operator/(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, std::divides<T> {});
  }

  namespace detail {
    /*!
     * \brief `a // b` and `a % b` of reals as NumPy's `divmod` computes them (Python's: the quotient floored, the
     *        remainder of \p b's sign): `fmod` first, the quotient `(a − mod) / b` adjusted by the remainder's sign and
     *        snapped to the nearest integer, a zero of the right sign. \p b zero: `a / b` and `fmod(a, b)`.
     */
    template <std::floating_point T>
    [[nodiscard]] std::pair<T, T> floor_divmod(T a,
                                               T b) noexcept
    {
      T mod {std::fmod(a, b)};
      if (b == T {0}) {
        return {a / b, mod};
      }
      T quotient {(a - mod) / b};
      if (mod != T {0}) {
        if ((b < T {0}) != (mod < T {0})) {
          mod      += b;
          quotient -= T {1};
        }
      }
      else {
        mod = std::copysign(T {0}, b);
      }
      if (quotient == T {0}) {
        return {std::copysign(T {0}, a / b), mod};
      }
      T floored {std::floor(quotient)};
      if (quotient - floored > T {0.5}) {
        floored += T {1};
      }
      return {floored, mod};
    }

    /*!
     * \brief `a // b` and `a % b` of signed integers as NumPy's: the quotient floored, the remainder of \p b's sign; by
     *        zero both 0 (NumPy's answer, beside its warning), and the most negative value by −1 its own negation (0
     *        remainder), wrapped where C++'s would be undefined.
     */
    template <std::signed_integral T>
    [[nodiscard]] std::pair<T, T> floor_divmod(T a,
                                               T b) noexcept
    {
      if (b == T {0}) {
        return {T {0}, T {0}};
      }
      if (b == T {-1}) {
        return {typename arithmetic<T>::negate {}(a), T {0}};
      }
      T quotient {static_cast<T>(a / b)};
      T mod      {static_cast<T>(a % b)};
      if (mod != T {0} && ((mod < T {0}) != (b < T {0}))) {
        quotient = static_cast<T>(quotient - T {1});
        mod      = static_cast<T>(mod + b);
      }
      return {quotient, mod};
    }
  } // namespace detail

  //! \brief `a // b` element by element, broadcast, as `numpy.floor_divide` (\ref detail::floor_divmod).
  //!        \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> floor_divide(const tensor<T>& a,
                                       const tensor<T>& b)
  {
    return broadcast_apply(a, b, [](T x, T y) { return detail::floor_divmod(x, y).first; });
  }

  //! \brief `a % b` element by element, broadcast, as `numpy.remainder` (Python's: of \p b's sign; a real by zero
  //!        `fmod`'s NaN, \ref detail::floor_divmod). \throws dimension_error If the shapes do not broadcast.
  template <typename T>
  [[nodiscard]] tensor<T> remainder(const tensor<T>& a,
                                    const tensor<T>& b)
  {
    return broadcast_apply(a, b, [](T x, T y) { return detail::floor_divmod(x, y).second; });
  }

  //! \brief Element-wise negation.
  template <typename T>
  [[nodiscard]] tensor<T> operator-(const tensor<T>& a)
  {
    return detail::parallel_map(a, typename detail::arithmetic<T>::negate {});
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
   *        the operators do: NumPy's loops are \ref detail::numpy_square, \ref detail::numpy_reciprocal and \ref detail::numpy_power.
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

  namespace detail {
    //! \brief \p base to the power \p exponent (not negative) by repeated squaring, each product wrapped modulo 2ⁿ as
    //!        NumPy's integer power (\ref arithmetic).
    template <std::signed_integral T>
    [[nodiscard]] T integer_power(T base,
                                  T exponent) noexcept
    {
      const typename arithmetic<T>::multiplies times  {};
      T                                        result {1};
      while (exponent != 0) {
        if ((exponent & 1) != 0) {
          result = times(result, base);
        }
        base       = times(base, base);
        exponent >>= 1;
      }
      return result;
    }
  } // namespace detail

  /*!
   * \brief `a` to the power `b` of signed integers element by element, broadcast, as NumPy's: exactly, by repeated
   *        squaring, wrapped modulo 2ⁿ. \throws std::domain_error If an exponent is negative (NumPy's refusal: no
   *        integer is the answer). \throws dimension_error If the shapes do not broadcast.
   */
  template <std::signed_integral T>
  [[nodiscard]] tensor<T> pow(const tensor<T>& a,
                              const tensor<T>& b)
  {
    if (std::ranges::any_of(b.data(), [](T exponent) { return exponent < 0; })) {
      throw std::domain_error("Integers to negative integer powers are not allowed.");
    }
    return broadcast_apply(a, b, [](T x, T y) { return detail::integer_power(x, y); });
  }

  //! \brief The sum of every element, by pairwise halving (0 for an empty tensor), on the threads where they are many
  //!        (\ref detail::parallel_pairwise_sum): the same bits.
  template <typename T>
  [[nodiscard]] T sum(const tensor<T>& t)
  {
    return detail::parallel_pairwise_sum(t.data().data(), t.size());
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

  //! \brief The sum of every element of a signed integer tensor, wrapped modulo 2ⁿ as NumPy's (in any order the same
  //!        bits: the unsigned sum is associative), by chunks on the threads where they are many.
  template <std::signed_integral T>
  [[nodiscard]] T sum(const tensor<T>& t)
  {
    using unsigned_type = typename detail::arithmetic<T>::unsigned_type;
    const T* const             data   {t.data().data()};
    const std::size_t          count  {t.size()};
    const std::size_t          chunks {(count + detail::elementwise_chunk - 1) / detail::elementwise_chunk};
    std::vector<unsigned_type> partial(chunks, 0);
    detail::parallel_for(chunks, detail::elementwise_chunk, [&](std::size_t c) {
                           const std::size_t last {std::min((c + 1) * detail::elementwise_chunk, count)};
                           unsigned_type     total {0};
                           for (std::size_t i {c * detail::elementwise_chunk}; i < last; ++i) {
                             total += static_cast<unsigned_type>(data[i]);
                           }
                           partial[c] = total;
                         });
    unsigned_type total {0};
    for (const unsigned_type part : partial) {
      total += part;
    }
    return static_cast<T>(total);
  }

  //! \brief The sums along \p axis of a signed integer tensor, wrapped modulo 2ⁿ as NumPy's. \throws dimension_error If
  //!        \p axis is not an axis.
  template <std::signed_integral T>
  [[nodiscard]] tensor<T> sum(const tensor<T>& t,
                              std::size_t      axis)
  {
    return detail::reduce_along(t, axis, [](const T* first, std::size_t count, std::size_t stride) {
                                  const typename detail::arithmetic<T>::plus plus {};
                                  T                                          total {0};
                                  for (std::size_t i {0}; i < count; ++i) {
                                    total = plus(total, first[i * stride]);
                                  }
                                  return total;
                                });
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

  /*!
   * \brief The product of every element, in order (`numpy.prod`'s, whose multiplications are sequential: only its sums
   *        pair). An empty tensor's is 1.
   */
  template <typename T>
  [[nodiscard]] T prod(const tensor<T>& t)
  {
    const typename detail::arithmetic<T>::multiplies times   {};
    T                                                product {1};
    for (const T& x : t.data()) {
      product = times(product, x);
    }
    return product;
  }

  //! \brief The products along \p axis, each line in order. \throws dimension_error If \p axis is not an axis.
  template <typename T>
  [[nodiscard]] tensor<T> prod(const tensor<T>& t,
                               std::size_t      axis)
  {
    return detail::fold_along(t, axis, T {1}, typename detail::arithmetic<T>::multiplies {});
  }

  namespace detail {

    //! \brief Each line of \p t along \p axis replaced by its running \p step (`x₀`, `step(x₀, x₁)`, …), in order.
    template <typename T,
              typename Step>
    [[nodiscard]] tensor<T> scan_along(const tensor<T>& t,
                                       std::size_t      axis,
                                       const Step&      step)
    {
      const axis_walk walk {walk_along(t.shape(), axis)};
      buffer<T>       out(t.size());
      const T* const  in   {t.data().data()};
      if (walk.inner == 1) {
        parallel_for(walk.outer, walk.extent, [&](std::size_t o) { // each line its own, contiguous
                       const std::size_t first {o * walk.extent};
                       for (std::size_t k {0}; k < walk.extent; ++k) {
                         out[first + k] = k == 0 ? in[first] : step(out[first + k - 1], in[first + k]);
                       }
                     });
        return tensor<T> {t.shape(), std::move(out)};
      }
      // the lines are columns: a band of them per task, row after row (each line's steps in the same order)
      const std::size_t bands {(walk.inner + band_columns - 1) / band_columns};
      parallel_for(walk.outer * bands, walk.extent * band_columns, [&](std::size_t at) {
                     const std::size_t o     {at / bands};
                     const std::size_t first {(at % bands) * band_columns};
                     const std::size_t width {std::min(band_columns, walk.inner - first)};
                     const std::size_t base  {(o * walk.extent * walk.inner) + first};
                     for (std::size_t k {0}; k < walk.extent; ++k) {
                       const std::size_t row {base + (k * walk.inner)};
                       for (std::size_t w {0}; w < width; ++w) {
                         out[row + w] = k == 0 ? in[row + w] : step(out[row - walk.inner + w], in[row + w]);
                       }
                     }
                   });
      return tensor<T> {t.shape(), std::move(out)};
    }

    /*!
     * \brief The running products of a complex \p t along \p axis, in order, each as NumPy's loops compute it
     *        (\ref detail::numpy_multiply): its vector loop's for a line of two (one product), its reduction loop's
     *        past it; fused as \p loops says. The binding's `cumprod`. \throws dimension_error If \p axis is not an
     *        axis.
     */
    template <detail::split_complex C>
    [[nodiscard]] tensor<C> numpy_cumprod(const tensor<C>&   t,
                                          std::size_t        axis,
                                          const numpy_loops& loops = numpy_wheel_loops())
    {
      const bool vector {walk_along(t.shape(), axis).extent == 2};
      return scan_along(t, axis, [vector, &loops](const C& running, const C& x) {
                          return vector ? numpy_multiply(running, x, loops.products) : numpy_reduction_multiply(running, x, loops);
                        });
    }

    //! \brief Whether \p a sorts before \p b as NumPy sorts reals: ascending, NaNs last.
    template <std::floating_point T>
    [[nodiscard]] bool sorts_before(T a,
                                    T b) noexcept
    {
      return std::isnan(b) ? !std::isnan(a) : a < b;
    }

    //! \brief Whether \p a sorts before \p b as integers sort: ascending.
    template <std::integral T>
    [[nodiscard]] bool sorts_before(T a,
                                    T b) noexcept
    {
      return a < b;
    }

    //! \brief Whether \p a sorts before \p b as NumPy sorts complex numbers: by the real parts, then the imaginary
    //!        parts, each with NaNs last.
    template <split_complex C>
    [[nodiscard]] bool sorts_before(const C& a,
                                    const C& b) noexcept
    {
      const bool before {sorts_before(a.real(), b.real())};
      return before || (!sorts_before(b.real(), a.real()) && sorts_before(a.imag(), b.imag()));
    }
  } // namespace detail

  //! \brief The running sums along \p axis, in order, as `numpy.cumsum`. \throws dimension_error If \p axis is not an
  //!        axis.
  template <typename T>
  [[nodiscard]] tensor<T> cumsum(const tensor<T>& t,
                                 std::size_t      axis)
  {
    return detail::scan_along(t, axis, typename detail::arithmetic<T>::plus {});
  }

  //! \brief The running products along \p axis, in order, as `numpy.cumprod`. \throws dimension_error If \p axis is
  //!        not an axis.
  template <typename T>
  [[nodiscard]] tensor<T> cumprod(const tensor<T>& t,
                                  std::size_t      axis)
  {
    return detail::scan_along(t, axis, typename detail::arithmetic<T>::multiplies {});
  }

  namespace detail {
    //! \brief Runs this long or longer are sorted by their bits (\ref radix_sort); shorter ones by comparisons.
    inline constexpr std::size_t radix_sort_least {256};

    //! \brief The unsigned integer of \p T's width, whose order \ref order_key maps \p T's onto.
    template <typename T>
    using order_key_t = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;

    //! \brief The highest bit of \p U, a real's or an integer's sign.
    template <typename U>
    inline constexpr U sign_bit {U {1} << ((sizeof(U) * 8) - 1)};

    //! \brief \p x's bits as an unsigned integer ordered as \p x is by `<`: the sign bit set for a positive real,
    //!        every bit flipped for a negative one. A NaN and the zeros' signs are the caller's: −0 orders below +0.
    template <std::floating_point T>
    [[nodiscard]] order_key_t<T> order_key(T x) noexcept
    {
      using U = order_key_t<T>;
      const auto bits {std::bit_cast<U>(x)};
      return (bits & sign_bit<U>) != 0 ? static_cast<U>(~bits) : static_cast<U>(bits | sign_bit<U>);
    }

    //! \brief An integer's bits ordered as it is: its sign bit flipped.
    template <std::integral T>
    [[nodiscard]] order_key_t<T> order_key(T x) noexcept
    {
      using U = order_key_t<T>;
      return static_cast<U>(std::bit_cast<U>(x) ^ sign_bit<U>);
    }

    //! \brief The real whose \ref order_key is \p key.
    template <std::floating_point T>
    [[nodiscard]] T from_order_key(order_key_t<T> key) noexcept
    {
      using U = order_key_t<T>;
      return std::bit_cast<T>((key & sign_bit<U>) != 0 ? static_cast<U>(key & ~sign_bit<U>) : static_cast<U>(~key));
    }

    //! \brief The integer whose \ref order_key is \p key.
    template <std::integral T>
    [[nodiscard]] T from_order_key(order_key_t<T> key) noexcept
    {
      return std::bit_cast<T>(static_cast<order_key_t<T>>(key ^ sign_bit<order_key_t<T>>));
    }

    /*!
     * \brief Sorts `[first, last)` by \ref order_key, a byte at a time from the least significant (a stable counting
     *        sort each, so equal keys keep their order; a byte that every key shares skipped).
     */
    template <typename T>
    void radix_sort(T* first,
                    T* last)
    {
      using U = order_key_t<T>;
      const auto count {static_cast<std::size_t>(last - first)};
      buffer<U>  keys(count);
      buffer<U>  spare(count);
      for (std::size_t i {0}; i < count; ++i) {
        keys[i] = order_key(first[i]);
      }
      U* from {keys.data()};
      U* to   {spare.data()};
      for (std::size_t shift {0}; shift < sizeof(U) * 8; shift += 8) {
        std::array<std::size_t, 257> places {};
        for (std::size_t i {0}; i < count; ++i) {
          ++places[((from[i] >> shift) & 0xffU) + 1];
        }
        if (std::ranges::find(places, count) != places.end()) { // one byte value alone: in order already
          continue;
        }
        std::partial_sum(places.begin(), places.end(), places.begin());
        for (std::size_t i {0}; i < count; ++i) {
          to[places[(from[i] >> shift) & 0xffU]++] = from[i];
        }
        std::swap(from, to);
      }
      for (std::size_t i {0}; i < count; ++i) {
        first[i] = from_order_key<T>(from[i]);
      }
    }

    /*!
     * \brief Sorts `[first, last)` stably as \ref sorts_before orders reals: the NaNs moved behind the rest in their
     *        order first, the rest by `<`, which tests for no NaN. A long run by \ref radix_sort, its zeros then
     *        rewritten with their signs in their order (the bits order −0 below +0, where `<` holds them equal); a
     *        short one by `std::stable_sort` with a lambda of its own, not the default `std::less`, which libc++ 22
     *        sorts by the bits from 1024 doubles on, −0 first.
     */
    template <std::floating_point T>
    void sort_run(T* first,
                  T* last)
    {
      T* const numbers_end {std::stable_partition(first, last, [](T x) { return !std::isnan(x); })};
      if (static_cast<std::size_t>(numbers_end - first) < radix_sort_least) {
        std::stable_sort(first, numbers_end, [](T a, T b) { return a < b; });
        return;
      }
      std::vector<bool> negative_zeros; // each zero's sign, in their order
      for (const T* x {first}; x != numbers_end; ++x) {
        if (*x == T {0}) {
          negative_zeros.push_back(std::signbit(*x));
        }
      }
      radix_sort(first, numbers_end);
      T* zero {std::lower_bound(first, numbers_end, T {0}, [](T a, T b) { return a < b; })};
      for (const bool negative : negative_zeros) {
        *zero++ = negative ? -T {0} : T {0};
      }
    }

    //! \brief The types whose indices `order_run` sorts by the elements' bits.
    template <typename T>
    concept radix_orderable = std::same_as<T, double> || std::same_as<T, float> || std::same_as<T, std::int64_t>;

    //! \brief A real's \ref order_key for an argsort: both zeros the same key, as `<` holds them equal (their indices
    //!        then order them).
    template <std::floating_point T>
    [[nodiscard]] order_key_t<T> argsort_key(T x) noexcept
    {
      return order_key(x == T {0} ? T {0} : x);
    }

    //! \brief An integer's \ref order_key for an argsort.
    [[nodiscard]] inline order_key_t<std::int64_t> argsort_key(std::int64_t x) noexcept
    {
      return order_key(x);
    }

    //! \brief Where a real's NaNs begin among the indices `[first, last)`, moved behind the rest in their order.
    template <std::floating_point T>
    [[nodiscard]] std::int64_t* numbers_end(std::int64_t* first,
                                            std::int64_t* last,
                                            const T*      values,
                                            std::size_t   stride)
    {
      return std::stable_partition(first, last, [&](std::int64_t k) { return !std::isnan(values[static_cast<std::size_t>(k) * stride]); });
    }

    //! \brief An integer has no NaN: every index is a number's.
    [[nodiscard]] inline std::int64_t* numbers_end(std::int64_t* /*first*/,
                                                   std::int64_t*       last,
                                                   const std::int64_t* /*values*/,
                                                   std::size_t /*stride*/)
    {
      return last;
    }

    /*!
     * \brief Sorts the indices `[first, last)` stably by their elements (`values[k · stride]`), as \ref sorts_before
     *        orders them: comparisons for a run shorter than \ref radix_sort_least, else the NaNs set apart in their
     *        order and the rest by \ref argsort_key, a byte at a time (a stable counting sort of the keys and their
     *        indices together; a byte that every key shares skipped): equal elements by index either way.
     */
    template <radix_orderable T,
              typename Less>
    void order_run(std::int64_t*  first,
                   std::int64_t*  last,
                   const T*       values,
                   std::size_t    stride,
                   const Less&    less)
    {
      if (static_cast<std::size_t>(last - first) < radix_sort_least) {
        std::stable_sort(first, last, less);
        return;
      }
      using U = order_key_t<T>;
      std::int64_t* const  end   {numbers_end(first, last, values, stride)};
      const auto           count {static_cast<std::size_t>(end - first)};
      buffer<U>            keys(count);
      buffer<U>            key_spare(count);
      buffer<std::int64_t> index_spare(count);
      for (std::size_t i {0}; i < count; ++i) {
        keys[i] = argsort_key(values[static_cast<std::size_t>(first[i]) * stride]);
      }
      U*            key_from   {keys.data()};
      U*            key_to     {key_spare.data()};
      std::int64_t* index_from {first};
      std::int64_t* index_to   {index_spare.data()};
      for (std::size_t shift {0}; shift < sizeof(U) * 8; shift += 8) {
        std::array<std::size_t, 257> places {};
        for (std::size_t i {0}; i < count; ++i) {
          ++places[((key_from[i] >> shift) & 0xffU) + 1];
        }
        if (std::ranges::find(places, count) != places.end()) {
          continue;
        }
        std::partial_sum(places.begin(), places.end(), places.begin());
        for (std::size_t i {0}; i < count; ++i) {
          const std::size_t at {places[(key_from[i] >> shift) & 0xffU]++};
          key_to[at]   = key_from[i];
          index_to[at] = index_from[i];
        }
        std::swap(key_from, key_to);
        std::swap(index_from, index_to);
      }
      if (index_from != first) {
        std::copy_n(index_from, count, first);
      }
    }

    //! \brief Sorts the indices `[first, last)` stably by their elements, by comparisons (any other type).
    template <typename T,
              typename Less>
    void order_run(std::int64_t* first,
                   std::int64_t* last,
                   const T* /*values*/,
                   std::size_t /*stride*/,
                   const Less&   less)
    {
      std::stable_sort(first, last, less);
    }

    //! \brief Sorts `[first, last)` stably as integers sort.
    template <std::integral T>
    void sort_run(T* first,
                  T* last)
    {
      std::stable_sort(first, last);
    }

    //! \brief Sorts `[first, last)` as int64 sort: a long run by \ref radix_sort (equal integers are the same bits, so
    //!        any order of them is the stable one).
    inline void sort_run(std::int64_t* first,
                         std::int64_t* last)
    {
      if (static_cast<std::size_t>(last - first) < radix_sort_least) {
        std::stable_sort(first, last);
        return;
      }
      radix_sort(first, last);
    }

    //! \brief Sorts `[first, last)` stably as \ref sorts_before orders complex numbers.
    template <split_complex C>
    void sort_run(C* first,
                  C* last)
    {
      std::stable_sort(first, last, [](const C& a, const C& b) { return sorts_before(a, b); });
    }

    /*!
     * \brief Sorts the \p count elements at \p data stably in \p pieces, a power of four, among the threads: each piece
     *        sorted apart by \p sort_piece, then the pieces merged in pairs by \p less (the order \p sort_piece sorts
     *        by), round after round, between \p data and a scratch buffer
     *        (`std::merge` takes the first piece's element of two equal ones, so the merge keeps their order). The
     *        rounds are even in number, so the last writes \p data.
     */
    template <typename T,
              typename SortRun,
              typename Less>
    void parallel_sort_run(T*             data,
                           std::size_t    count,
                           std::size_t    pieces,
                           const SortRun& sort_piece,
                           const Less&    less)
    {
      std::vector<std::size_t> bounds(pieces + 1);
      for (std::size_t k {0}; k <= pieces; ++k) {
        bounds[k] = (count / pieces * k) + (std::min(k, count % pieces));
      }
      parallel_for(pieces, count / pieces * 16, [&](std::size_t k) { sort_piece(data + bounds[k], data + bounds[k + 1]); });
      buffer<T> scratch(count);
      T*        from {data};
      T*        to   {scratch.data()};
      for (std::size_t width {1}; width < pieces; width *= 2) {
        const std::size_t pairs {(pieces + (2 * width) - 1) / (2 * width)};
        parallel_for(pairs, count / pairs, [&](std::size_t pair) {
                       const std::size_t first  {bounds[pair * 2 * width]};
                       const std::size_t middle {bounds[std::min((pair * 2 * width) + width, pieces)]};
                       const std::size_t last   {bounds[std::min((pair + 1) * 2 * width, pieces)]};
                       std::merge(from + first, from + middle, from + middle, from + last, to + first, less);
                     });
        std::swap(from, to);
      }
    }
  } // namespace detail

  /*!
   * \brief The elements sorted along \p axis, as `numpy.sort`: ascending, NaNs last, equal elements in their order
   *        (NumPy's `kind="stable"`; its default sort may order equal ones otherwise, which only a sign of zero shows).
   *
   * Lines enough for the threads are sorted one to a task; fewer, each is sorted in turn among the threads
   * (\ref detail::parallel_sort_run). A line along the last axis is sorted where it lies in the result.
   * \throws dimension_error If \p axis is not an axis.
   */
  template <typename T>
  [[nodiscard]] tensor<T> sort(const tensor<T>& t,
                               std::size_t      axis)
  {
    const detail::axis_walk walk {detail::walk_along(t.shape(), axis)};
    const std::size_t       lines {walk.outer * walk.inner};
    buffer<T>               out(t.size());
    const T* const          in   {t.data().data()};
    const auto              gather {[&](std::size_t line, T* into) {
                                      const std::size_t first {((line / walk.inner) * walk.extent * walk.inner) + (line % walk.inner)};
                                      for (std::size_t k {0}; k < walk.extent; ++k) {
                                        into[k] = in[first + (k * walk.inner)];
                                      }
                                    }};
    const auto              scatter {[&](std::size_t line, const T* from) {
                                       const std::size_t first {((line / walk.inner) * walk.extent * walk.inner) + (line % walk.inner)};
                                       for (std::size_t k {0}; k < walk.extent; ++k) {
                                         out[first + (k * walk.inner)] = from[k];
                                       }
                                     }};
    const std::size_t       threads {detail::parallel_threads(walk.extent, 16)};
    if (threads >= 2 && detail::parallel_threads(lines, walk.extent * 16) * 2 <= threads) {
      std::size_t pieces {4};
      while (pieces < threads) {
        pieces *= 4;
      }
      buffer<T> line_buffer(walk.inner == 1 ? 0 : walk.extent);
      for (std::size_t line {0}; line < lines; ++line) {
        T* const run {walk.inner == 1 ? out.data() + (line * walk.extent) : line_buffer.data()};
        gather(line, run);
        detail::parallel_sort_run(run, walk.extent, pieces, [](T* first, T* last) { detail::sort_run(first, last); },
                                  [](const T& a, const T& b) { return detail::sorts_before(a, b); });
        if (walk.inner != 1) {
          scatter(line, run);
        }
      }
      return tensor<T> {t.shape(), std::move(out)};
    }
    detail::parallel_for(lines, walk.extent * 16, [&](std::size_t line) {
                           if (walk.inner == 1) {
                             T* const run {out.data() + (line * walk.extent)};
                             gather(line, run);
                             detail::sort_run(run, run + walk.extent);
                             return;
                           }
                           buffer<T> run(walk.extent);
                           gather(line, run.data());
                           detail::sort_run(run.data(), run.data() + walk.extent);
                           scatter(line, run.data());
                         });
    return tensor<T> {t.shape(), std::move(out)};
  }

  /*!
   * \brief The indices that sort the elements along \p axis, as `numpy.argsort` (its `kind="stable"`): of each line,
   *        in the order \ref sort puts its elements, equal ones by index. A long line is sorted among the threads
   *        (\ref detail::parallel_sort_run). \throws dimension_error If \p axis is not an axis.
   */
  template <typename T>
  [[nodiscard]] tensor<std::int64_t> argsort(const tensor<T>& t,
                                             std::size_t      axis)
  {
    const detail::axis_walk walk {detail::walk_along(t.shape(), axis)};
    const std::size_t       lines {walk.outer * walk.inner};
    buffer<std::int64_t>    out(t.size());
    const T* const          in {t.data().data()};
    const std::size_t       threads {detail::parallel_threads(walk.extent, 16)};
    const bool              shared {threads >= 2 && detail::parallel_threads(lines, walk.extent * 16) * 2 <= threads};
    const auto              order_line {[&](std::size_t line, std::size_t pieces) {
                                          const std::size_t first {((line / walk.inner) * walk.extent * walk.inner) + (line % walk.inner)};
                                          const auto        value {[&](std::int64_t k) -> const T& {
                                                                     return in[first + (static_cast<std::size_t>(k) * walk.inner)];
                                                                   }};
                                          const auto        less {[&](std::int64_t a, std::int64_t b) { return detail::sorts_before(value(a), value(b)); }};
                                          buffer<std::int64_t> order(walk.extent);
                                          std::iota(order.begin(), order.end(), std::int64_t {0});
                                          const T* const line_values {in + first};
                                          const auto     sort_piece {[&](std::int64_t* from, std::int64_t* to) {
                                                                       detail::order_run(from, to, line_values, walk.inner, less);
                                                                     }};
                                          if (pieces > 1) {
                                            detail::parallel_sort_run(order.data(), walk.extent, pieces, sort_piece, less);
                                          }
                                          else {
                                            sort_piece(order.data(), order.data() + walk.extent);
                                          }
                                          for (std::size_t k {0}; k < walk.extent; ++k) {
                                            out[first + (k * walk.inner)] = order[k];
                                          }
                                        }};
    if (shared) {
      std::size_t pieces {4};
      while (pieces < threads) {
        pieces *= 4;
      }
      for (std::size_t line {0}; line < lines; ++line) {
        order_line(line, pieces);
      }
    }
    else {
      detail::parallel_for(lines, walk.extent * 16, [&](std::size_t line) { order_line(line, 1); });
    }
    return tensor<std::int64_t> {t.shape(), std::move(out)};
  }

  namespace detail {
    //! \brief A real's NaN, which wins `numpy.argmax` and `numpy.argmin`.
    template <std::floating_point T>
    [[nodiscard]] bool wins_any_order(T x) noexcept
    {
      return std::isnan(x);
    }

    //! \brief An integer's: none.
    template <std::integral T>
    [[nodiscard]] bool wins_any_order(T /*x*/) noexcept
    {
      return false;
    }

    //! \brief The index of the first largest (\p larger `std::greater`) or smallest element of a line of \p count at
    //!        stride \p stride, as NumPy's: the first NaN where there is one.
    template <typename T,
              typename Order>
    [[nodiscard]] std::int64_t extreme_index(const T*     first,
                                             std::size_t  count,
                                             std::size_t  stride,
                                             const Order& larger)
    {
      std::size_t best {0};
      for (std::size_t i {0}; i < count; ++i) {
        const T& x {first[i * stride]};
        if (wins_any_order(x)) {
          return static_cast<std::int64_t>(i);
        }
        if (larger(x, first[best * stride])) {
          best = i;
        }
      }
      return static_cast<std::int64_t>(best);
    }

    //! \brief \ref argmax or \ref argmin along \p axis by \p larger.
    template <typename T,
              typename Order>
    [[nodiscard]] tensor<std::int64_t> extreme_indices(const tensor<T>& t,
                                                       std::size_t      axis,
                                                       const Order&     larger)
    {
      const axis_walk walk {walk_along(t.shape(), axis)};
      if (walk.extent == 0) {
        throw dimension_error("attempt to get argmax or argmin of an empty sequence");
      }
      buffer<std::int64_t> out(walk.outer * walk.inner);
      const T* const       in {t.data().data()};
      if (walk.inner == 1) {
        parallel_for(walk.outer, walk.extent, [&](std::size_t line) {
                       out[line] = extreme_index(in + (line * walk.extent), walk.extent, 1, larger);
                     });
        return tensor<std::int64_t> {without_axis(t.shape(), axis), std::move(out)};
      }
      // the lines are columns: a band of them per task, row after row, each column's first winner kept as its own
      // line's walk keeps it (a NaN's at once, else the first of equals)
      const std::size_t bands {(walk.inner + band_columns - 1) / band_columns};
      parallel_for(walk.outer * bands, walk.extent * band_columns, [&](std::size_t at) {
                     const std::size_t                     o     {at / bands};
                     const std::size_t                     first {(at % bands) * band_columns};
                     const std::size_t                     width {std::min(band_columns, walk.inner - first)};
                     const T* const                        base  {in + (o * walk.extent * walk.inner) + first};
                     std::array<std::size_t, band_columns> best {};
                     std::array<bool, band_columns>        won {};
                     for (std::size_t w {0}; w < width; ++w) {
                       won[w] = wins_any_order(base[w]); // the first element itself a NaN
                     }
                     for (std::size_t k {1}; k < walk.extent; ++k) {
                       const T* const row {base + (k * walk.inner)};
                       for (std::size_t w {0}; w < width; ++w) {
                         if (won[w]) {
                           continue;
                         }
                         if (wins_any_order(row[w])) {
                           best[w] = k;
                           won[w]  = true;
                         }
                         else if (larger(row[w], base[(best[w] * walk.inner) + w])) {
                           best[w] = k;
                         }
                       }
                     }
                     for (std::size_t w {0}; w < width; ++w) {
                       out[(o * walk.inner) + first + w] = static_cast<std::int64_t>(best[w]);
                     }
                   });
      return tensor<std::int64_t> {without_axis(t.shape(), axis), std::move(out)};
    }

    //! \brief \ref extreme_index of the \p count contiguous elements at \p first, by chunks on the threads: each chunk's
    //!        own, then the first of the chunks' (a NaN's chunk wins at once, the earliest of equal ones otherwise).
    template <typename T,
              typename Order>
    [[nodiscard]] std::int64_t parallel_extreme_index(const T*     first,
                                                      std::size_t  count,
                                                      const Order& larger)
    {
      const std::size_t         chunks {(count + elementwise_chunk - 1) / elementwise_chunk};
      std::vector<std::int64_t> best(chunks);
      parallel_for(chunks, elementwise_chunk, [&](std::size_t c) {
                     const std::size_t start {c * elementwise_chunk};
                     best[c] = static_cast<std::int64_t>(start) + extreme_index(first + start, std::min(elementwise_chunk, count - start), 1, larger);
                   });
      std::int64_t winner {best[0]};
      for (const std::int64_t candidate : best) {
        const T& x {first[candidate]};
        if (wins_any_order(x)) {
          return candidate; // the earliest chunk's NaN: chunks are in order
        }
        if (larger(x, first[winner])) {
          winner = candidate;
        }
      }
      return winner;
    }
  } // namespace detail

  //! \brief The flat index of the first largest element, as `numpy.argmax` (the first NaN where there is one).
  //!        \throws dimension_error If \p t is empty.
  template <typename T>
  [[nodiscard]] std::int64_t argmax(const tensor<T>& t)
  {
    if (t.size() == 0) {
      throw dimension_error("attempt to get argmax of an empty sequence");
    }
    return detail::parallel_extreme_index(t.data().data(), t.size(), std::greater<T> {});
  }

  //! \brief The flat index of the first smallest element, as `numpy.argmin` (the first NaN where there is one).
  //!        \throws dimension_error If \p t is empty.
  template <typename T>
  [[nodiscard]] std::int64_t argmin(const tensor<T>& t)
  {
    if (t.size() == 0) {
      throw dimension_error("attempt to get argmin of an empty sequence");
    }
    return detail::parallel_extreme_index(t.data().data(), t.size(), std::less<T> {});
  }

  //! \brief The index of the first largest element of every line along \p axis, as `numpy.argmax`. \throws
  //!        dimension_error If \p axis is not an axis, or is empty.
  template <typename T>
  [[nodiscard]] tensor<std::int64_t> argmax(const tensor<T>& t,
                                            std::size_t      axis)
  {
    return detail::extreme_indices(t, axis, std::greater<T> {});
  }

  //! \brief The index of the first smallest element of every line along \p axis, as `numpy.argmin`. \throws
  //!        dimension_error If \p axis is not an axis, or is empty.
  template <typename T>
  [[nodiscard]] tensor<std::int64_t> argmin(const tensor<T>& t,
                                            std::size_t      axis)
  {
    return detail::extreme_indices(t, axis, std::less<T> {});
  }

  /*!
   * \brief The indices of the nonzero elements (`x != T {}`: a NaN is one), one tensor per axis, in C order, as
   *        `numpy.nonzero`: by chunks on the threads, each chunk's counted, then its indices written at its offset.
   *        A chunk walks its elements' indices as an odometer (one division per axis at its start, none after) and
   *        writes every element's, the place advancing by one where it is nonzero, without a branch; into a buffer of
   *        its own, whose last write may run one past its part.
   */
  template <typename T>
  [[nodiscard]] std::vector<tensor<std::int64_t>> nonzero(const tensor<T>& t)
  {
    const std::vector<std::size_t>& shape  {t.shape()};
    const T* const                  data   {t.data().data()};
    const std::size_t               count  {t.size()};
    const std::size_t               rank   {shape.size()};
    const std::size_t               chunk  {detail::elementwise_chunk};
    const std::size_t               chunks {(count + chunk - 1) / chunk};
    std::vector<std::size_t>        offsets(chunks + 1, 0);
    detail::parallel_for(chunks, chunk, [&](std::size_t c) {
                           const std::size_t last {std::min((c + 1) * chunk, count)};
                           offsets[c + 1] = static_cast<std::size_t>(std::count_if(data + (c * chunk), data + last, [](const T& x) { return x != T {}; }));
                         });
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    const std::size_t                 found {offsets.back()};
    std::vector<buffer<std::int64_t>> out(rank);
    for (buffer<std::int64_t>& axis_out : out) {
      axis_out = buffer<std::int64_t>(found);
    }
    detail::parallel_for(chunks, chunk * rank, [&](std::size_t c) {
                           const std::size_t        first {c * chunk};
                           const std::size_t        last  {std::min(first + chunk, count)};
                           const std::size_t        span  {last - first + 1}; // one past the part: the last write's
                           buffer<std::int64_t>     scratch(span * rank);
                           std::vector<std::size_t> coordinate(rank);
                           std::size_t              rest  {first};
                           for (std::size_t a {rank}; a-- > 0;) {
                             coordinate[a] = rest % shape[a];
                             rest         /= shape[a];
                           }
                           std::size_t at {0};
                           for (std::size_t k {first}; k < last; ++k) {
                             for (std::size_t a {0}; a < rank; ++a) {
                               scratch[(a * span) + at] = static_cast<std::int64_t>(coordinate[a]);
                             }
                             at += static_cast<std::size_t>(data[k] != T {});
                             for (std::size_t a {rank}; a-- > 0;) { // the next element's indices
                               if (++coordinate[a] < shape[a]) {
                                 break;
                               }
                               coordinate[a] = 0;
                             }
                           }
                           for (std::size_t a {0}; a < rank; ++a) {
                             std::copy_n(scratch.data() + (a * span), at, out[a].data() + offsets[c]);
                           }
                         });
    std::vector<tensor<std::int64_t>> indices;
    indices.reserve(rank);
    for (buffer<std::int64_t>& axis_out : out) {
      indices.emplace_back(std::vector<std::size_t> {found}, std::move(axis_out));
    }
    return indices;
  }

  //! \brief The number of nonzero elements (`x != T {}`) along \p axis, as `numpy.count_nonzero`. \throws
  //!        dimension_error If \p axis is not an axis.
  template <typename T>
  [[nodiscard]] tensor<std::int64_t> count_nonzero(const tensor<T>& t,
                                                   std::size_t      axis)
  {
    return detail::fold_along(t, axis, std::int64_t {0},
                              [](std::int64_t total, const T& x) { return total + static_cast<std::int64_t>(x != T {}); });
  }

  /*!
   * \brief The elements of \p t at \p indices along \p axis, the other axes matched, as `numpy.take_along_axis`:
   *        \p indices of \p t's rank and shape but along the axis (any extent there), each in `[0, extent)`; read
   *        directly, on the threads.
   * \throws dimension_error If the shapes do not match so, or an index is out of its range.
   */
  template <typename T>
  [[nodiscard]] tensor<T> take_along_axis(const tensor<T>&            t,
                                          const tensor<std::int64_t>& indices,
                                          std::size_t                 axis)
  {
    const detail::axis_walk walk    {detail::walk_along(t.shape(), axis)};
    bool                    matched {indices.rank() == t.rank()};
    for (std::size_t k {0}; matched && k < t.rank(); ++k) {
      matched = k == axis || indices.shape()[k] == t.shape()[k];
    }
    if (!matched) {
      throw dimension_error("take_along_axis: the indices' shape is the tensor's but along the axis");
    }
    const auto extent {static_cast<std::int64_t>(t.shape()[axis])};
    if (std::ranges::any_of(indices.data(), [extent](std::int64_t i) { return i < 0 || i >= extent; })) {
      throw dimension_error("take_along_axis: an index out of the axis's range");
    }
    const std::size_t   count {indices.shape()[axis]};
    const std::size_t   inner {walk.inner};
    buffer<T>           out(indices.size());
    const T* const      in    {t.data().data()};
    const std::int64_t* at    {indices.data().data()};
    detail::parallel_for(indices.size() / std::max<std::size_t>(inner, 1), inner, [&](std::size_t row) {
                           const std::size_t o {row / std::max<std::size_t>(count, 1)};
                           for (std::size_t i {0}; i < inner; ++i) {
                             const auto index {static_cast<std::size_t>(at[(row * inner) + i])};
                             out[(row * inner) + i] = in[(((o * walk.extent) + index) * inner) + i];
                           }
                         });
    return tensor<T> {indices.shape(), std::move(out)};
  }

  namespace detail {
    //! \brief A line with more positions than these is sorted whole rather than selected position by position.
    inline constexpr std::size_t select_positions_sorted {8};

    /*!
     * \brief Of the \p count elements at \p row, the one a stable sort (\ref sort_run) puts at \p k, from the one
     *        \p chosen that a selection put there, equal to it under \ref sorts_before. A zero (either sign) or a
     *        NaN (any payload) is not alone in its class, and the class's member of the same rank in \p row's
     *        order is the stable sort's.
     */
    template <std::floating_point T>
    [[nodiscard]] T stable_choice(const T*    row,
                                  std::size_t count,
                                  std::size_t k,
                                  T           chosen)
    {
      const bool nan {std::isnan(chosen)};
      if (!nan && chosen != T {0}) {
        return chosen;
      }
      std::size_t first {0}; // the class's first rank: the elements below a zero, or every one that is no NaN
      for (std::size_t i {0}; i < count; ++i) {
        first += static_cast<std::size_t>(nan ? !std::isnan(row[i]) : row[i] < T {0});
      }
      std::size_t rank {k - first};
      std::size_t i    {0};
      for (;; ++i) { // the class holds rank k: its member of that rank is met before the row ends
        if (nan ? std::isnan(row[i]) : row[i] == T {0}) {
          if (rank == 0) {
            break;
          }
          --rank;
        }
      }
      return row[i];
    }

    //! \brief An integer's class under `<` is itself alone.
    template <std::integral T>
    [[nodiscard]] T stable_choice(const T* /*row*/,
                                  std::size_t /*count*/,
                                  std::size_t /*k*/,
                                  T chosen)
    {
      return chosen;
    }

    /*!
     * \brief Into \p out, the elements of the \p count at \p row that a stable sort puts at the \p m positions
     *        \p at, \p work holding \p count elements of scratch: `std::nth_element` at each distinct position,
     *        smallest first, each over the part above the one before (the least or the largest of that part by a
     *        pass alone), then \ref stable_choice; more positions than \ref select_positions_sorted, the row sorted
     *        whole.
     */
    template <typename T>
    requires std::floating_point<T> || std::integral<T>
    void select_row(const T*            row,
                    std::size_t         count,
                    const std::int64_t* at,
                    std::size_t         m,
                    T*                  work,
                    T*                  out)
    {
      std::copy(row, row + count, work);
      if (m > select_positions_sorted) {
        sort_run(work, work + count);
        for (std::size_t j {0}; j < m; ++j) {
          out[j] = work[static_cast<std::size_t>(at[j])];
        }
        return;
      }
      std::array<std::size_t, select_positions_sorted> order {};
      for (std::size_t j {0}; j < m; ++j) { // sorted by insertion: eight at most
        const auto  k {static_cast<std::size_t>(at[j])};
        std::size_t i {j};
        for (; i > 0 && order[i - 1] > k; --i) {
          order[i] = order[i - 1];
        }
        order[i] = k;
      }
      const auto  before {[](T a, T b) { return sorts_before(a, b); }};
      std::size_t from   {0};
      for (std::size_t j {0}; j < m; ++j) {
        const std::size_t k {order[j]};
        if (k < from) {                    // a position named twice: in place already
          continue;
        }
        if (k == from || k + 1 == count) { // the least or the largest of the part left: one pass finds it
          std::iter_swap(work + k, k == from ? std::min_element(work + from, work + count, before)
                                             : std::max_element(work + from, work + count, before));
        }
        else {
          std::nth_element(work + from, work + k, work + count, before);
        }
        from = k + 1;
      }
      for (std::size_t j {0}; j < m; ++j) {
        const auto k {static_cast<std::size_t>(at[j])};
        out[j] = stable_choice(row, count, k, work[k]);
      }
    }

    //! \brief A complex row is sorted whole: its classes under \ref sorts_before are zeros and NaNs in either part.
    template <split_complex C>
    void select_row(const C*            row,
                    std::size_t         count,
                    const std::int64_t* at,
                    std::size_t         m,
                    C*                  work,
                    C*                  out)
    {
      std::copy(row, row + count, work);
      sort_run(work, work + count);
      for (std::size_t j {0}; j < m; ++j) {
        out[j] = work[static_cast<std::size_t>(at[j])];
      }
    }
  } // namespace detail

  /*!
   * \brief Of each line along the last axis of \p t, the elements its stable sort (\ref sort) puts at the
   *        positions \p positions names, as `sort(t)[..., positions]` reads them, without the sort: each line's
   *        positions found by selection (\ref detail::select_row), the lines on the threads. \p positions has shape
   *        `(m,)`, the same for every line, or the lines' shape and `m`; the result, the lines' shape and `m`.
   * \throws dimension_error If \p t has no axis, \p positions has neither shape, or a position is not in a line.
   */
  template <typename T>
  [[nodiscard]] tensor<T> select_along_last(const tensor<T>&            t,
                                            const tensor<std::int64_t>& positions)
  {
    if (t.rank() == 0 || positions.rank() == 0) {
      throw dimension_error("select_along_last: a tensor and positions of rank 1 or more");
    }
    const std::vector<std::size_t> lead(t.shape().begin(), t.shape().end() - 1);
    const std::size_t              count {t.shape().back()};
    const std::size_t              m     {positions.shape().back()};
    const bool                     every {positions.rank() == 1};
    if (!every && !std::equal(lead.begin(), lead.end(), positions.shape().begin(), positions.shape().end() - 1)) {
      throw dimension_error("select_along_last: positions of shape (m,) or the lines' shape and m");
    }
    const auto extent {static_cast<std::int64_t>(count)};
    if (std::ranges::any_of(positions.data(), [extent](std::int64_t k) { return k < 0 || k >= extent; })) {
      throw dimension_error("select_along_last: a position out of the lines");
    }
    std::vector<std::size_t> shape {lead};
    shape.push_back(m);
    const std::size_t   lines      {std::accumulate(lead.begin(), lead.end(), std::size_t {1}, std::multiplies {})};
    buffer<T>           out(lines * m);
    const T* const      in         {t.data().data()};
    const std::int64_t* at         {positions.data().data()};
    const std::size_t   block      {std::max<std::size_t>(1, 4096 / std::max<std::size_t>(count, 1))}; // rows a task takes
    const std::size_t   blocks     {(lines + block - 1) / block};
    detail::parallel_for(blocks, block * count * 16, [&](std::size_t b) {
                           buffer<T> work(count);
                           for (std::size_t line {b * block}; line < std::min(lines, (b + 1) * block); ++line) {
                             detail::select_row(in + (line * count), count, at + (every ? 0 : line * m), m, work.data(),
                                                out.data() + (line * m));
                           }
                         });
    return tensor<T> {std::move(shape), std::move(out)};
  }

  namespace detail {
    //! \brief `sum + x · y` of a real or an integer (wrapped) correlation's terms.
    template <typename T>
    [[nodiscard]] T correlation_step(const T& sum,
                                     const T& x,
                                     const T& y)
    {
      return typename arithmetic<T>::plus {}(sum, typename arithmetic<T>::multiplies {}(x, y));
    }

    //! \brief `sum + x · conj(y)` of a complex correlation's terms, NumPy's product rounded.
    template <split_complex C>
    [[nodiscard]] C correlation_step(const C& sum,
                                     const C& x,
                                     const C& y)
    {
      return sum + numpy_multiply(x, std::conj(y), false);
    }
  } // namespace detail

  /*!
   * \brief The cross-correlation `c[k] = Σₙ a[n + k] · conj(v[n])` of two vectors over the lags \p first …
   *        `first + count − 1` (from `−(len v − 1)`, the full range's first, to `len a − 1`), as `numpy.correlate`'s:
   *        each lag's sum in the order of `v`'s terms, a complex product NumPy's rounded one (no infinity recovered),
   *        an integer's wrapped; the lags on the threads.
   */
  template <typename T>
  [[nodiscard]] tensor<T> correlate(const tensor<T>& a,
                                    const tensor<T>& v,
                                    std::ptrdiff_t   first,
                                    std::size_t      count)
  {
    const auto     n {static_cast<std::ptrdiff_t>(a.size())};
    const auto     m {static_cast<std::ptrdiff_t>(v.size())};
    const T* const x {a.data().data()};
    const T* const y {v.data().data()};
    buffer<T>      out(count);
    detail::parallel_for(count, static_cast<std::size_t>(m), [&](std::size_t lag) {
                           const std::ptrdiff_t k {first + static_cast<std::ptrdiff_t>(lag)};
                           T                    sum {};
                           for (std::ptrdiff_t j {std::max<std::ptrdiff_t>(0, -k)}; j < std::min(m, n - k); ++j) {
                             sum = detail::correlation_step(sum, x[j + k], y[j]);
                           }
                           out[lag] = sum;
                         });
    return tensor<T> {{count}, std::move(out)};
  }

  /*!
   * \brief Where each element of \p v would go in the sorted line \p a to keep it sorted, as `numpy.searchsorted`: the
   *        first such place (\p right false) or the last, in \ref detail::sorts_before's order (NaNs last).
   *        \throws dimension_error If \p a is not a vector.
   */
  template <typename T>
  [[nodiscard]] tensor<std::int64_t> searchsorted(const tensor<T>& a,
                                                  const tensor<T>& v,
                                                  bool             right = false)
  {
    if (a.rank() != 1) {
      throw dimension_error("searchsorted: a sorted vector");
    }
    const T* const first {a.data().data()};
    const T* const last {first + a.size()};
    const auto     less {[](const T& x, const T& y) { return detail::sorts_before(x, y); }};
    return detail::parallel_map(v, [&](const T& x) {
                                  const T* const at {right ? std::upper_bound(first, last, x, less) : std::lower_bound(first, last, x, less)};
                                  return static_cast<std::int64_t>(at - first);
                                });
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
   * \brief \p t repeated along its axes of extent 1 and new leading ones to \p shape, as `numpy.broadcast_to` (a copy:
   *        a tensor's elements are contiguous): each row of the result copied from its row of \p t, or filled with
   *        its one element where the last axis broadcasts, the rows on the threads.
   * \throws dimension_error If \p t's shape does not broadcast to \p shape.
   */
  template <typename T>
  [[nodiscard]] tensor<T> broadcast_to(const tensor<T>&         t,
                                       std::vector<std::size_t> shape)
  {
    const std::vector<std::size_t>& from {t.shape()};
    bool                            fits {from.size() <= shape.size()};
    for (std::size_t i {0}; fits && i < from.size(); ++i) {
      const std::size_t have {from[from.size() - 1 - i]};
      fits = have == 1 || have == shape[shape.size() - 1 - i];
    }
    if (!fits) {
      throw dimension_error("broadcast_to: the shape " + detail::shape_text(from) + " does not broadcast to "
                            + detail::shape_text(shape));
    }
    const std::size_t total {detail::element_count(shape.begin(), shape.end())};
    buffer<T>         out(total);
    if (total == 0) {
      return tensor<T> {std::move(shape), std::move(out)};
    }
    const std::vector<std::size_t> strides {detail::broadcast_strides(from, shape.size())};
    const std::size_t              inner   {shape.empty() ? 1 : shape.back()};
    const bool                     spread  {!shape.empty() && strides.back() == 0}; // the last axis repeats one element
    const T* const                 in      {t.data().data()};
    detail::parallel_for(total / inner, inner, [&](std::size_t row) {
                           std::size_t rest {row};
                           std::size_t at {0};
                           for (std::size_t axis {shape.size() - (shape.empty() ? 0 : 1)}; axis-- > 0;) {
                             at   += (rest % shape[axis]) * strides[axis];
                             rest /= shape[axis];
                           }
                           T* const into {out.data() + (row * inner)};
                           if (spread) {
                             std::fill_n(into, inner, in[at]);
                           }
                           else {
                             std::copy_n(in + at, inner, into);
                           }
                         });
    return tensor<T> {std::move(shape), std::move(out)};
  }

  /*!
   * \brief The same elements in the same order under another shape, shared with \p t (no copy).
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
    return t.reshaped(std::move(shape)); // the same elements, shared: no copy
  }

  /*!
   * \brief The \p offset diagonal of each matrix of \p t (its last two axes), as `numpy.linalg.diagonal`: above the
   *        main diagonal for a positive \p offset, below for a negative one, read where it lies (no copy of the
   *        matrices); the batch's shape, then the diagonal's length (0 where \p offset passes the matrix).
   * \throws dimension_error If \p t has fewer than two axes.
   */
  template <typename T>
  [[nodiscard]] tensor<T> diagonal(const tensor<T>& t,
                                   std::ptrdiff_t   offset)
  {
    const std::vector<std::size_t>& shape {t.shape()};
    if (shape.size() < 2) {
      throw dimension_error("diagonal: a matrix or a stack of matrices (rank >= 2)");
    }
    const std::size_t rows      {shape[shape.size() - 2]};
    const std::size_t cols      {shape[shape.size() - 1]};
    const std::size_t magnitude {offset < 0 ? std::size_t {0} - static_cast<std::size_t>(offset) : static_cast<std::size_t>(offset)};
    const std::size_t r0        {offset < 0 ? magnitude : 0};
    const std::size_t c0        {offset < 0 ? 0 : magnitude};
    const std::size_t length    {r0 < rows && c0 < cols ? std::min(rows - r0, cols - c0) : 0};
    std::size_t       batch     {1};
    for (std::size_t axis {0}; axis + 2 < shape.size(); ++axis) {
      batch = detail::checked_mul(batch, shape[axis]);
    }
    std::vector<std::size_t> out(shape.begin(), shape.end() - 2);
    out.push_back(length);
    buffer<T>      values(batch * length);
    const T* const data {t.data().data()};
    for (std::size_t b {0}; b < batch; ++b) {
      const T* const first {data + (b * rows * cols) + (r0 * cols) + c0};
      for (std::size_t i {0}; i < length; ++i) {
        values[(b * length) + i] = first[i * (cols + 1)];
      }
    }
    return tensor<T> {std::move(out), std::move(values)};
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
    buffer<T> values(count); // its pages first touched by the threads that fill them
    detail::parallel_fill(values.data(), count, value);
    return tensor<T> {std::move(shape), std::move(values)};
  }

  //! \brief A tensor of \p shape full of zeros (each thread's share zeroed as the C library zeroes).
  template <typename T>
  [[nodiscard]] tensor<T> zeros(std::vector<std::size_t> shape)
  {
    std::size_t count {1};
    for (const std::size_t extent : shape) {
      count = detail::checked_mul(count, extent);
    }
    buffer<T> values(count);
    detail::parallel_zero(values.data(), count);
    return tensor<T> {std::move(shape), std::move(values)};
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
   *        length is `ceil((stop − start) / step)`, its first two elements \p start itself (a zero's sign
   *        kept) and `start + step`, and element `i` after them `start + i·delta` with
   *        `delta = (start + step) − start`, the step as the first two elements hold it: rounded once where
   *        \p fused (a fused multiply-add, as NumPy's macOS arm64 wheels compute it), else the product rounded
   *        first (NumPy's other wheels), which can differ in the last bits.
   * \throws dimension_error If \p step is zero or not finite.
   */
  [[nodiscard]] inline tensor<double> arange(double start,
                                             double stop,
                                             double step,
                                             bool   fused = true)
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
    const double        second            {start + step};
    const double        delta             {second - start};
    if (count > 0) {
      out[0] = start;
    }
    if (count > 1) {
      out[1] = second;
    }
    if (fused) {
      for (std::size_t i {2}; i < count; ++i) {
        out[i] = std::fma(static_cast<double>(i), delta, start);
      }
    }
    else {
      for (std::size_t i {2}; i < count; ++i) {
        // the product rounded by itself, `i · delta − 0`, so that no compiler fuses it into the sum
        out[i] = start + std::fma(static_cast<double>(i), delta, -0.0);
      }
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
