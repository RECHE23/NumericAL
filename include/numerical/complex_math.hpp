/*!
 * \file complex_math.hpp
 * \brief The elementary functions of a complex number of `float` or `double` as the GNU C library computes them
 *        (NumPy's on Linux, to the bit): C's Annex G for the infinities and NaNs, scaled against overflow, the
 *        logarithm's real part near the unit circle from an almost exact `x² + y² − 1`.
 *
 * `std::exp`, `std::log`, `std::sin`, `std::cos`, `std::tan` and `std::tanh` of a complex number are libc++'s on macOS
 * and MSVC's on Windows, which lose a part where glibc's do not: `tan(1e308 + 0j)` was NaN for −0.509, `sin(1e308 j)`
 * had a NaN real part for 0, `log(1 + 1e-20 j)` a real part of 0 for 5e-41. Each sum below is a statement of its own,
 * so that no build contracts it into a fused multiply-add glibc's does not compute.
 */
#ifndef NUMERICAL_COMPLEX_MATH_HPP
#define NUMERICAL_COMPLEX_MATH_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numbers>

namespace numerical::detail {

  //! \brief The real part past which `exp` of it overflows, rounded down: 709 for `double`, 88 for `float`.
  template <std::floating_point R>
  inline constexpr R exp_bound {static_cast<R>(static_cast<int>(static_cast<R>(std::numeric_limits<R>::max_exponent - 1) * std::numbers::ln2_v<R>))};

  //! \brief Half of it, past which `tanh`'s parts are ±1 and a quotient by `exp(2x)`: 354 for `double`, 44 for `float`.
  template <std::floating_point R>
  inline constexpr R half_exp_bound {static_cast<R>(static_cast<int>(static_cast<R>(std::numeric_limits<R>::max_exponent - 1) * std::numbers::ln2_v<R> / R {2}))};

  //! \brief `a·b + c·d` as the GNU C library's build computes it: `fma(a, b, c·d)` on arm64, where GCC contracts it
  //!        (FMA is in the baseline); each product rounded on x86-64, whose baseline has no FMA.
  template <std::floating_point R>
  [[nodiscard]] R libm_sum_of_products(R a,
                                       R b,
                                       R c,
                                       R d)
  {
    const R second {c * d};
#if defined(__aarch64__) || defined(_M_ARM64)
    return std::fma(a, b, second);
#else
    const R first {a * b};
    return first + second;
#endif
  }

  //! \brief `s + a·b` as glibc's build computes `s += a·b`: fused on arm64, as \ref libm_sum_of_products.
  template <std::floating_point R>
  [[nodiscard]] R libm_add_product(R s,
                                   R a,
                                   R b)
  {
#if defined(__aarch64__) || defined(_M_ARM64)
    return std::fma(a, b, s);
#else
    const R product {a * b};
    return s + product;
#endif
  }

  //! \brief `sin` and `cos` of \p x together, as glibc's complex functions take them: its `sincos` under glibc (whose
  //!        single-precision one strayed from `sinf` and `cosf` by a unit in the last place before 2.29), the system's
  //!        `__sincos` on macOS (a compiler may fuse a `sin` and a `cos` into it where it inlines them, and it strays
  //!        from them by a unit: the same call in every context), `std::sin` and `std::cos` elsewhere.
  inline void sine_and_cosine(double  x,
                              double& sine,
                              double& cosine)
  {
#if defined(__GLIBC__)
    ::sincos(x, &sine, &cosine);
#elif defined(__APPLE__)
    ::__sincos(x, &sine, &cosine);
#else
    sine   = std::sin(x);
    cosine = std::cos(x);
#endif
  }

  //! \brief \ref sine_and_cosine of a `float`.
  inline void sine_and_cosine(float  x,
                              float& sine,
                              float& cosine)
  {
#if defined(__GLIBC__)
    ::sincosf(x, &sine, &cosine);
#elif defined(__APPLE__)
    ::__sincosf(x, &sine, &cosine);
#else
    sine   = std::sin(x);
    cosine = std::cos(x);
#endif
  }

  //! \brief `sin` and `cos` of \p x (\ref sine_and_cosine), or \p x and 1 for one under the smallest normal value (an
  //!        exact sine there).
  template <std::floating_point R>
  void small_sincos(R  x,
                    R& sine,
                    R& cosine)
  {
    if (std::abs(x) > std::numeric_limits<R>::min()) {
      sine_and_cosine(x, sine, cosine);
      return;
    }
    sine   = x;
    cosine = R {1};
  }

  /*!
   * \brief `exp(z)`: `exp(re)` times `cos(im) + i sin(im)`, a real part past \ref exp_bound taken off in steps of it
   *        (each a factor `exp(bound)` on the sine and the cosine), so that a finite product does not overflow early.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_exp(const std::complex<R>& z)
  {
    constexpr R inf {std::numeric_limits<R>::infinity()};
    constexpr R nan {std::numeric_limits<R>::quiet_NaN()};
    R           x   {z.real()};
    const R     y   {z.imag()};
    if (std::isfinite(x)) {
      if (!std::isfinite(y)) {
        return {nan, nan};
      }
      R sine            {};
      R cosine          {};
      small_sincos(y, sine, cosine);
      constexpr R bound {exp_bound<R>};
      if (x > bound) {
        const R step {std::exp(bound)};
        x      -= bound;
        sine   *= step;
        cosine *= step;
        if (x > bound) {
          x      -= bound;
          sine   *= step;
          cosine *= step;
        }
      }
      if (x > bound) { // past three bounds: it overflows
        return {std::numeric_limits<R>::max() * cosine, std::numeric_limits<R>::max() * sine};
      }
      const R scale {std::exp(x)};
      return {scale * cosine, scale * sine};
    }
    if (std::isinf(x)) {
      if (std::isfinite(y)) {
        const R magnitude {std::signbit(x) ? R {0} : inf};
        if (y == R {0}) {
          return {magnitude, y};
        }
        return {std::copysign(magnitude, std::cos(y)), std::copysign(magnitude, std::sin(y))};
      }
      if (!std::signbit(x)) {
        return {inf, y - y};
      }
      return {R {0}, std::copysign(R {0}, y)};
    }
    if (y == R {0}) {
      return {nan, y};
    }
    return {nan, nan};
  }

  /*!
   * \brief `cosh(z)`: `cosh(re) cos(im) + i sinh(re) sin(im)`; past \ref exp_bound, `exp(|re|) / 2` times the cosine and
   *        the sine, taken off in steps as \ref complex_exp does.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_cosh(const std::complex<R>& z)
  {
    constexpr R inf {std::numeric_limits<R>::infinity()};
    constexpr R nan {std::numeric_limits<R>::quiet_NaN()};
    const R     x   {z.real()};
    const R     y   {z.imag()};
    if (std::isfinite(x)) {
      if (!std::isfinite(y)) {
        return {y - y, x == R {0} ? R {0} : nan};
      }
      R sine            {};
      R cosine          {};
      small_sincos(y, sine, cosine);
      constexpr R bound {exp_bound<R>};
      if (std::abs(x) > bound) {
        const R step {std::exp(bound)};
        R       rx   {std::abs(x)};
        if (std::signbit(x)) {
          sine = -sine;
        }
        rx     -= bound;
        sine   *= step / R {2};
        cosine *= step / R {2};
        if (rx > bound) {
          rx     -= bound;
          sine   *= step;
          cosine *= step;
        }
        if (rx > bound) {
          return {std::numeric_limits<R>::max() * cosine, std::numeric_limits<R>::max() * sine};
        }
        const R scale {std::exp(rx)};
        return {scale * cosine, scale * sine};
      }
      return {std::cosh(x) * cosine, std::sinh(x) * sine};
    }
    if (std::isinf(x)) {
      if (std::isfinite(y) && y != R {0}) {
        return {std::copysign(inf, std::cos(y)), std::copysign(inf, std::sin(y)) * std::copysign(R {1}, x)};
      }
      if (y == R {0}) {
        return {inf, y * std::copysign(R {1}, x)};
      }
      return {inf, y - y};
    }
    return {nan, y == R {0} ? y : nan};
  }

  /*!
   * \brief `sinh(z)`: `sinh(re) cos(im) + i cosh(re) sin(im)` of `|re|`, the sign of `re` carried by the cosine; past
   *        \ref exp_bound as \ref complex_cosh.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_sinh(const std::complex<R>& z)
  {
    constexpr R inf      {std::numeric_limits<R>::infinity()};
    constexpr R nan      {std::numeric_limits<R>::quiet_NaN()};
    const bool  negative {std::signbit(z.real())};
    const R     x        {std::abs(z.real())};
    const R     y        {z.imag()};
    if (std::isfinite(x)) {
      if (!std::isfinite(y)) {
        if (x == R {0}) {
          return {negative ? R {-0.0} : R {0}, y - y};
        }
        return {nan, nan};
      }
      R sine   {};
      R cosine {};
      small_sincos(y, sine, cosine);
      if (negative) {
        cosine = -cosine;
      }
      constexpr R bound {exp_bound<R>};
      if (x > bound) {
        const R step {std::exp(bound)};
        R       rx   {x - bound};
        sine   *= step / R {2};
        cosine *= step / R {2};
        if (rx > bound) {
          rx     -= bound;
          sine   *= step;
          cosine *= step;
        }
        if (rx > bound) {
          return {std::numeric_limits<R>::max() * cosine, std::numeric_limits<R>::max() * sine};
        }
        const R scale {std::exp(rx)};
        return {scale * cosine, scale * sine};
      }
      return {std::sinh(x) * cosine, std::cosh(x) * sine};
    }
    if (std::isinf(x)) {
      if (std::isfinite(y) && y != R {0}) {
        R sine     {};
        R cosine   {};
        small_sincos(y, sine, cosine);
        const R re {std::copysign(inf, cosine)};
        return {negative ? -re : re, std::copysign(inf, sine)};
      }
      if (y == R {0}) {
        return {negative ? -inf : inf, y};
      }
      return {inf, y - y};
    }
    return {nan, y == R {0} ? y : nan};
  }

  //! \brief `cos(z)`, as `cosh(i z)`: \ref complex_cosh of `−im + i re`.
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_cos(const std::complex<R>& z)
  {
    return complex_cosh(std::complex<R> {-z.imag(), z.real()});
  }

  /*!
   * \brief `sin(z)`: `sin(re) cosh(im) + i cos(re) sinh(im)` of `|re|`, the sign of `re` carried by the sine; an
   *        imaginary part past \ref exp_bound as \ref complex_sinh's real part.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_sin(const std::complex<R>& z)
  {
    constexpr R inf      {std::numeric_limits<R>::infinity()};
    constexpr R nan      {std::numeric_limits<R>::quiet_NaN()};
    const bool  negative {std::signbit(z.real())};
    const R     x        {std::abs(z.real())};
    const R     y        {z.imag()};
    if (std::isfinite(y)) {
      if (!std::isfinite(x)) {
        if (y == R {0}) {
          return {x - x, y};
        }
        return {nan, nan};
      }
      R sine   {};
      R cosine {};
      small_sincos(x, sine, cosine);
      if (negative) {
        sine = -sine;
      }
      constexpr R bound {exp_bound<R>};
      if (std::abs(y) > bound) {
        const R step {std::exp(bound)};
        R       iy   {std::abs(y) - bound};
        if (std::signbit(y)) {
          cosine = -cosine;
        }
        sine   *= step / R {2};
        cosine *= step / R {2};
        if (iy > bound) {
          iy     -= bound;
          sine   *= step;
          cosine *= step;
        }
        if (iy > bound) {
          return {std::numeric_limits<R>::max() * sine, std::numeric_limits<R>::max() * cosine};
        }
        const R scale {std::exp(iy)};
        return {scale * sine, scale * cosine};
      }
      return {std::cosh(y) * sine, std::sinh(y) * cosine};
    }
    if (std::isinf(y)) {
      if (x == R {0}) {
        return {negative ? R {-0.0} : R {0}, y};
      }
      if (std::isfinite(x)) {
        R sine     {};
        R cosine   {};
        small_sincos(x, sine, cosine);
        const R re {std::copysign(inf, sine)};
        const R im {std::copysign(inf, cosine)};
        return {negative ? -re : re, std::signbit(y) ? -im : im};
      }
      return {x - x, inf};
    }
    return {x == R {0} ? (negative ? R {-0.0} : R {0}) : nan, nan};
  }

  /*!
   * \brief `tanh(z)`: `(sinh(re) cosh(re) + i sin(im) cos(im)) / (sinh(re)² + cos(im)²)`; past \ref half_exp_bound the
   *        real part is ±1 and the imaginary one `4 sin(im) cos(im) / exp(2|re|)`, divided in steps.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_tanh(const std::complex<R>& z)
  {
    constexpr R nan {std::numeric_limits<R>::quiet_NaN()};
    const R     x   {z.real()};
    const R     y   {z.imag()};
    if (!std::isfinite(x) || !std::isfinite(y)) {
      if (std::isinf(x)) {
        if (std::isfinite(y) && std::abs(y) > R {1}) {
          const R product {std::sin(y) * std::cos(y)};
          return {std::copysign(R {1}, x), std::copysign(R {0}, product)};
        }
        return {std::copysign(R {1}, x), std::copysign(R {0}, y)};
      }
      if (y == R {0}) {
        return z;
      }
      return {x == R {0} ? x : nan, nan};
    }
    R sine            {};
    R cosine          {};
    small_sincos(y, sine, cosine);
    constexpr R bound {half_exp_bound<R>};
    if (std::abs(x) > bound) {
      const R step {std::exp(R {2} *bound)};
      R       im   {R {4} *sine};
      im       *= cosine;
      R rx         {std::abs(x) - bound};
      im /= step;
      if (rx > bound) { // the imaginary part underflows
        im /= step;
      }
      else {
        im /= std::exp(R {2} *rx);
      }
      return {std::copysign(R {1}, x), im};
    }
    R sinh_x {x};
    R cosh_x {1};
    if (std::abs(x) > std::numeric_limits<R>::min()) {
      sinh_x = std::sinh(x);
      cosh_x = std::cosh(x);
    }
    R den {cosine * cosine};
    if (std::abs(sinh_x) > std::abs(cosine) * std::numeric_limits<R>::epsilon()) {
      den = libm_sum_of_products(sinh_x, sinh_x, cosine, cosine);
    }
    const R re {sinh_x * cosh_x};
    const R im {sine * cosine};
    return {re / den, im / den};
  }

  /*!
   * \brief `tan(z)`: `(sin(re) cos(re) + i sinh(im) cosh(im)) / (cos(re)² + sinh(im)²)`; past \ref half_exp_bound the
   *        imaginary part is ±1 and the real one `4 sin(re) cos(re) / exp(2|im|)`, divided in steps.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_tan(const std::complex<R>& z)
  {
    constexpr R nan {std::numeric_limits<R>::quiet_NaN()};
    const R     x   {z.real()};
    const R     y   {z.imag()};
    if (!std::isfinite(x) || !std::isfinite(y)) {
      if (std::isinf(y)) {
        if (std::isfinite(x) && std::abs(x) > R {1}) {
          const R product {std::sin(x) * std::cos(x)};
          return {std::copysign(R {0}, product), std::copysign(R {1}, y)};
        }
        return {std::copysign(R {0}, x), std::copysign(R {1}, y)};
      }
      if (x == R {0}) {
        return z;
      }
      return {nan, y == R {0} ? y : nan};
    }
    R sine            {};
    R cosine          {};
    small_sincos(x, sine, cosine);
    constexpr R bound {half_exp_bound<R>};
    if (std::abs(y) > bound) {
      const R step {std::exp(R {2} *bound)};
      R       re   {R {4} *sine};
      re       *= cosine;
      R iy         {std::abs(y) - bound};
      re /= step;
      if (iy > bound) { // the real part underflows
        re /= step;
      }
      else {
        re /= std::exp(R {2} *iy);
      }
      return {re, std::copysign(R {1}, y)};
    }
    R sinh_y {y};
    R cosh_y {1};
    if (std::abs(y) > std::numeric_limits<R>::min()) {
      sinh_y = std::sinh(y);
      cosh_y = std::cosh(y);
    }
    R den {cosine * cosine};
    if (std::abs(sinh_y) > std::abs(cosine) * std::numeric_limits<R>::epsilon()) {
      den = libm_sum_of_products(sinh_y, sinh_y, cosine, cosine);
    }
    const R re {sine * cosine};
    const R im {sinh_y * cosh_y};
    return {re / den, im / den};
  }

  /*!
   * \brief `x² + y² − 1` for `0.5 ≤ x < 1`, `y ≤ x` and `x² + y² ≥ 0.5`, almost exactly: each square split into its
   *        rounded value and its error (a fused multiply-add), the five terms summed by magnitude, each pair's error
   *        carried to the next.
   */
  [[nodiscard]] inline double unit_circle_offset(double x,
                                                 double y)
  {
    std::array<double, 5> terms {};
    terms[1] = x * x;
    terms[0] = std::fma(x, x, -terms[1]);
    terms[3] = y * y;
    terms[2] = std::fma(y, y, -terms[3]);
    terms[4] = -1.0;
    const auto by_magnitude_from {[&terms](std::size_t first) { // insertion, five terms at most
                                    for (std::size_t i {first + 1}; i < terms.size(); ++i) {
                                      const double term {terms[i]};
                                      std::size_t  j {i};
                                      while (j > first && std::abs(terms[j - 1]) > std::abs(term)) {
                                        terms[j] = terms[j - 1];
                                        --j;
                                      }
                                      terms[j] = term;
                                    }
                                  }};
    by_magnitude_from(0);
    for (std::size_t i {0}; i <= 3; ++i) {
      const double sum   {terms[i + 1] + terms[i]};
      const double error {(terms[i + 1] - sum) + terms[i]};
      terms[i + 1] = sum;
      terms[i]     = error;
      by_magnitude_from(i + 1);
    }
    double total {terms[4] + terms[3]};
    total += terms[2];
    total += terms[1];
    total += terms[0];
    return total;
  }

  //! \brief \ref unit_circle_offset of `float`s, in `double`: `(x − 1)(x + 1) + y²`.
  [[nodiscard]] inline float unit_circle_offset(float x,
                                                float y)
  {
    const double wide_x  {x};
    const double wide_y  {y};
    return static_cast<float>(libm_sum_of_products(wide_x - 1.0, wide_x + 1.0, wide_y, wide_y));
  }

  //! \brief Whether `x² + y² ≥ 0.5`, as glibc's build computes the sum (\ref libm_sum_of_products).
  template <std::floating_point R>
  [[nodiscard]] bool unit_circle_squares_reach_half(R x,
                                                    R y)
  {
    return libm_sum_of_products(x, x, y, y) >= R {0.5};
  }

  /*!
   * \brief `log(z)`: `log|z| + i arg(z)`, `log|z|` as `log1p(|z|² − 1) / 2` near the unit circle (`|z|² − 1` exact or
   *        almost, \ref unit_circle_offset) and as `log(hypot)` elsewhere, the larger part scaled by 2 near overflow
   *        and both by 2^digits under the smallest normal value.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_log(const std::complex<R>& z)
  {
    constexpr R inf {std::numeric_limits<R>::infinity()};
    constexpr R nan {std::numeric_limits<R>::quiet_NaN()};
    const R     x   {z.real()};
    const R     y   {z.imag()};
    if (x == R {0} && y == R {0}) {
      const R angle {std::signbit(x) ? std::numbers::pi_v<R> : R {0}};
      return {-R {1} / std::abs(x), std::copysign(angle, y)};
    }
    if (std::isnan(x) || std::isnan(y)) {
      return {std::isinf(x) || std::isinf(y) ? inf : nan, nan};
    }
    R   larger  {std::abs(x)};
    R   smaller {std::abs(y)};
    int scale   {0};
    if (larger < smaller) {
      std::swap(larger, smaller);
    }
    constexpr R least {std::numeric_limits<R>::min()};
    if (larger > std::numeric_limits<R>::max() / R {2}) {
      scale   = -1;
      larger  = std::ldexp(larger, scale);
      smaller = smaller >= least * R {2} ? std::ldexp(smaller, scale) : R {0};
    }
    else if (larger < least) { // the smaller part too
      scale   = std::numeric_limits<R>::digits;
      larger  = std::ldexp(larger, scale);
      smaller = std::ldexp(smaller, scale);
    }
    R re {};
    if (larger == R {1}) { // scaled, the larger part is far from 1: these four branches take scale 0
      const R square {smaller * smaller};
      re = std::log1p(square) / R {2};
    }
    else if (larger > R {1} && larger < R {2} && smaller < R {1}) {
      R offset {(larger - R {1}) * (larger + R {1})};
      if (smaller >= std::numeric_limits<R>::epsilon()) {
        offset = libm_add_product(offset, smaller, smaller);
      }
      re = std::log1p(offset) / R {2};
    }
    else if (larger < R {1} && larger >= R {0.5} && smaller < std::numeric_limits<R>::epsilon() / R {2}) {
      const R offset {(larger - R {1}) * (larger + R {1})};
      re = std::log1p(offset) / R {2};
    }
    else if (larger < R {1} && larger >= R {0.5} && unit_circle_squares_reach_half(larger, smaller)) {
      re = std::log1p(unit_circle_offset(larger, smaller)) / R {2};
    }
    else {
      const R correction {static_cast<R>(scale) * std::numbers::ln2_v<R>};
      re = std::log(std::hypot(larger, smaller)) - correction;
    }
    return {re, std::atan2(y, x)};
  }

  /*!
   * \brief The square root of a complex number, as the GNU C library's `csqrt` computes it (NumPy's on Linux, to the
   *        bit): the infinities and NaNs of C's Annex G; a real or an imaginary argument apart (the root of `-1` is
   *        `i` exactly, that of `y i` has equal parts); otherwise `sqrt((|z| ± re) / 2)` for the larger part and the
   *        imaginary part over twice it for the other, scaled by 4 near overflow and by a power of 2 among subnormal
   *        parts. `std::sqrt` of libc++ goes through the polar form: `sqrt(-1e20 + 1j)` in single precision had a
   *        real part of 755 for 5e-11.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_sqrt(const std::complex<R>& z)
  {
    constexpr R inf   {std::numeric_limits<R>::infinity()};
    constexpr R nan   {std::numeric_limits<R>::quiet_NaN()};
    constexpr R least {std::numeric_limits<R>::min()};
    constexpr R half  {0.5};
    R           x     {z.real()};
    R           y     {z.imag()};
    if (std::isinf(y)) {
      return {inf, y};
    }
    if (std::isinf(x)) {
      if (x < R {0}) {
        return {std::isnan(y) ? nan : R {0}, std::copysign(inf, y)};
      }
      return {x, std::isnan(y) ? nan : std::copysign(R {0}, y)};
    }
    if (std::isnan(x) || std::isnan(y)) {
      return {nan, nan};
    }
    if (y == R {0}) {
      if (x < R {0}) {
        return {R {0}, std::copysign(std::sqrt(-x), y)};
      }
      return {std::abs(std::sqrt(x)), y};
    }
    if (x == R {0}) {
      const R part {std::abs(y) >= R {2} *least ? std::sqrt(half * std::abs(y)) : half * std::sqrt(R {2} *std::abs(y))};
      return {part, std::copysign(part, y)};
    }
    int scale {0};
    if (std::abs(x) > std::numeric_limits<R>::max() / R {4}) {
      scale = 1;
      x     = std::ldexp(x, -2);
      y     = std::ldexp(y, -2);
    }
    else if (std::abs(y) > std::numeric_limits<R>::max() / R {4}) {
      scale = 1;
      x     = std::abs(x) >= R {4} *least ? std::ldexp(x, -2) : R {0};
      y     = std::ldexp(y, -2);
    }
    else if (std::abs(x) < R {2} *least && std::abs(y) < R {2} *least) {
      scale = -((std::numeric_limits<R>::digits + 1) / 2);
      x     = std::ldexp(x, -2 * scale);
      y     = std::ldexp(y, -2 * scale);
    }
    const R modulus {std::hypot(x, y)};
    R       re      {};
    R       im      {};
    if (x > R {0}) { // re from |z| + x, im = y / (2 re): no cancellation
      re = std::sqrt(half * (modulus + x));
      if (scale == 1 && std::abs(y) < R {1}) {
        im    = y / re;
        re    = std::ldexp(re, scale);
        scale = 0;
      }
      else {
        im = half * (y / re);
      }
    }
    else { // im from |z| − x, re = |y| / (2 im)
      im = std::sqrt(half * (modulus - x));
      if (scale == 1 && std::abs(y) < R {1}) {
        re    = std::abs(y / im);
        im    = std::ldexp(im, scale);
        scale = 0;
      }
      else {
        re = std::abs(half * (y / im));
      }
    }
    if (scale != 0) {
      re = std::ldexp(re, scale);
      im = std::ldexp(im, scale);
    }
    return {re, std::copysign(im, y)};
  }

  /*!
   * \brief `atanh(z)`, as glibc's `catanh`: C's Annex G for the infinities and NaNs; `π/2` and the reciprocal past
   *        `16/ε`; else the real part `log(|1 + z|² / |1 − z|²) / 4` (by `log1p(4x / |1 − z|²)` where the quotient is
   *        at least ½), the imaginary part `atan2(2y, 1 − |z|²) / 2`, `1 − |z|²` from \ref unit_circle_offset where it
   *        cancels.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_atanh(const std::complex<R>& z)
  {
    const R      x        {z.real()};
    const R      y        {z.imag()};
    const R      half_pi  {std::numbers::pi_v<R> / R {2}};
    const R      eps      {std::numeric_limits<R>::epsilon()};
    const bool   x_finite {std::isfinite(x)};
    const bool   y_finite {std::isfinite(y)};
    if (!x_finite || !y_finite) {
      if (std::isinf(y)) {
        return {std::copysign(R {0}, x), std::copysign(half_pi, y)};
      }
      if (std::isinf(x) || x == R {0}) { // y NaN where x is 0
        return {std::copysign(R {0}, x), y_finite ? std::copysign(half_pi, y) : std::numeric_limits<R>::quiet_NaN()};
      }
      return {std::numeric_limits<R>::quiet_NaN(), std::numeric_limits<R>::quiet_NaN()};
    }
    if (x == R {0} && y == R {0}) {
      return z;
    }
    if (std::abs(x) >= R {16} / eps || std::abs(y) >= R {16} / eps) {
      R re {};
      if (std::abs(y) <= R {1}) {
        re = R {1} / x;
      }
      else if (std::abs(x) <= R {1}) {
        re = x / y / y;
      }
      else {
        const R h {std::hypot(x / R {2}, y / R {2})};
        re = x / h / h / R {4};
      }
      return {re, std::copysign(half_pi, y)};
    }
    R re {};
    if (std::abs(x) == R {1} && std::abs(y) < eps * eps) {
      re = std::copysign(R {0.5}, x) * (std::numbers::ln2_v<R> - std::log(std::abs(y)));
    }
    else {
      const R i2    {std::abs(y) >= eps * eps ? y * y : R {0}};
      const R plus  {R {1} + x};
      const R minus {R {1} - x};
      const R num   {libm_add_product(i2, plus, plus)};
      const R den   {libm_add_product(i2, minus, minus)};
      const R f     {num / den};
      if (f < R {0.5}) {
        re = R {0.25} *std::log(f);
      }
      else {
        const R four_x {R {4} *x};
        re = R {0.25} *std::log1p(four_x / den);
      }
    }
    const R absx {std::max(std::abs(x), std::abs(y))};
    const R absy {std::min(std::abs(x), std::abs(y))};
    R       den  {};
    if (absy < eps / R {2}) {
      const R product {(R {1} - absx) * (R {1} + absx)};
      den = product == R {0} ? R {0} : product; // +0 for −0, as glibc's
    }
    else if (absx >= R {1} || (absx < R {0.75} && absy < R {0.5})) {
      den = libm_sum_of_products(R {1} - absx, R {1} + absx, -absy, absy);
    }
    else {
      den = -unit_circle_offset(absx, absy);
    }
    const R twice_y {R {2} *y};
    return {re, R {0.5} *std::atan2(twice_y, den)};
  }
} // namespace numerical::detail

#endif // NUMERICAL_COMPLEX_MATH_HPP
