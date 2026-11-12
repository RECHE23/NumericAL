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

  /*!
   * \brief Whether the C library's complex functions fuse their sums of products, `a·b + c·d` as `fma(a, b, c·d)`,
   *        off arm64 (where FMA is in every build's baseline, and they do): under glibc as its build does, read once from
   *        its `ctan` of a number whose `cos² + sinh²` the two round apart. glibc 2.43 fuses there on an AMD EPYC 7763
   *        (GitHub's Ubuntu 26.04 runners) and not on an Intel Core i5-8500, the same library; NumPy's complex
   *        functions are glibc's. Elsewhere no.
   */
#if !defined(__aarch64__) && !defined(_M_ARM64)
  [[nodiscard]] inline bool libm_fuses() noexcept
  {
#  if defined(__GLIBC__) && defined(_GLIBCXX_USE_C99_COMPLEX) && _GLIBCXX_USE_C99_COMPLEX
    static const bool fused {[] {
                               const volatile double re {0.00028503236970950325}; // read at run time: no folding of the call
                               const volatile double im {-0.7408643619401744};
                               const std::complex<double> z      {re, im};
                               const std::complex<double> answer {std::tan(z)};   // libstdc++'s: glibc's ctan
                               double                     sine {};
                               double                     cosine {};
                               sine_and_cosine(z.real(), sine, cosine);
                               const double sinh_y {std::sinh(z.imag())};
                               const double den    {std::fma(sinh_y, sinh_y, cosine * cosine)};
                               return answer == std::complex<double> {(sine * cosine) / den, (sinh_y * std::cosh(z.imag())) / den};
                             }()};
    return fused;
#  else
    return false;
#  endif
  }

#endif

  //! \brief `a·b + c·d` as the C library's build computes it: `fma(a, b, c·d)` where it fuses (\ref libm_fuses), each
  //!        product rounded elsewhere.
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
    if (libm_fuses()) {
      return std::fma(a, b, second);
    }
    const R first {a * b};
    return first + second;
#endif
  }

  //! \brief `s + a·b` as the C library's build computes `s += a·b`: fused where it fuses, as \ref libm_sum_of_products.
  template <std::floating_point R>
  [[nodiscard]] R libm_add_product(R s,
                                   R a,
                                   R b)
  {
#if defined(__aarch64__) || defined(_M_ARM64)
    return std::fma(a, b, s);
#else
    if (libm_fuses()) {
      return std::fma(a, b, s);
    }
    const R product {a * b};
    return s + product;
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
    const double wide_x  {static_cast<double>(x)};
    const double wide_y  {static_cast<double>(y)};
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

  /*!
   * \brief `asinh(z)` of a finite nonzero \p z, as glibc's `__kernel_casinh`, its imaginary part taken from π/2 where
   *        \p adjust (`acos` and `acosh` through it): the first quadrant's `|re|` and `|im|`, by the region they lie
   *        in, `log(2z)` past 1/ε, `log(z + √(1 + z²))` by `log1p` and `atan2` of parts that do not cancel nearer 1,
   *        else that sum by \ref complex_sqrt and \ref complex_log; the signs of \p z's parts given back last.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> kernel_asinh(const std::complex<R>& z,
                                             bool                   adjust)
  {
    const R eps  {std::numeric_limits<R>::epsilon()};
    const R rx   {std::abs(z.real())};
    const R ix   {std::abs(z.imag())};
    const R sign {z.imag()};
    R       re   {};
    R       im   {};
    // the imaginary part from (a, b) as atan2(b, a), or atan2(a, ±b) where adjusted
    const auto angle {[&](R a, R b) { return adjust ? std::atan2(a, std::copysign(b, sign)) : std::atan2(b, a); }};
    const auto through_log {[&](std::complex<R> y) {
                              if (adjust) {
                                y = {std::copysign(y.imag(), sign), y.real()};
                              }
                              return complex_log(y);
                            }};
    if (rx >= R {1} / eps || ix >= R {1} / eps) { // z + √(1 + z²) as 2z, without the square's overflow
      const std::complex<R> res {through_log({rx, ix})};
      re = res.real() + std::numbers::ln2_v<R>;
      im = res.imag();
    }
    else if (rx >= R {0.5} && ix < eps / R {8}) {
      const R s {std::hypot(R {1}, rx)};
      re = std::log(rx + s);
      im = adjust ? std::atan2(s, z.imag()) : std::atan2(ix, s);
    }
    else if (rx < eps / R {8} && ix >= R {1.5}) {
      const R s {std::sqrt((ix + R {1}) * (ix - R {1}))};
      re = std::log(ix + s);
      im = angle(rx, s);
    }
    else if (ix > R {1} && ix < R {1.5} && rx < R {0.5}) {
      const R ix2m1 {(ix + R {1}) * (ix - R {1})};
      if (rx < eps * eps) {
        const R s {std::sqrt(ix2m1)};
        re = std::log1p(R {2} *(ix2m1 + (ix * s))) / R {2};
        im = angle(rx, s);
      }
      else {
        const R rx2 {rx * rx};
        const R f   {rx2 * (R {2} + rx2 + (R {2} *ix * ix))};
        const R d   {std::sqrt((ix2m1 * ix2m1) + f)};
        const R dp  {d + ix2m1};
        const R dm  {f / dp};
        const R r1  {std::sqrt((dm + rx2) / R {2})};
        const R r2  {rx * ix / r1};
        re = std::log1p(rx2 + dp + (R {2} *((rx * r1) + (ix * r2)))) / R {2};
        im = angle(rx + r1, ix + r2);
      }
    }
    else if (ix == R {1} && rx < R {0.5}) {
      if (rx < eps / R {8}) {
        const R root {std::sqrt(rx)};
        re = std::log1p(R {2} *(rx + root)) / R {2};
        im = angle(root, R {1});
      }
      else {
        const R d  {rx * std::sqrt(R {4} + (rx * rx))};
        const R s1 {std::sqrt((d + (rx * rx)) / R {2})};
        const R s2 {std::sqrt((d - (rx * rx)) / R {2})};
        re = std::log1p((rx * rx) + d + (R {2} *((rx * s1) + s2))) / R {2};
        im = angle(rx + s1, R {1} + s2);
      }
    }
    else if (ix < R {1} && rx < R {0.5}) {
      if (ix < eps) {
        const R s {std::hypot(R {1}, rx)};
        re = std::log1p(R {2} *rx * (rx + s)) / R {2};
        im = adjust ? std::atan2(s, z.imag()) : std::atan2(ix, s);
      }
      else if (rx < eps * eps) {
        const R onemix2 {(R {1} + ix) * (R {1} - ix)};
        const R s       {std::sqrt(onemix2)};
        re = std::log1p(R {2} *rx / s) / R {2};
        im = adjust ? std::atan2(s, z.imag()) : std::atan2(ix, s);
      }
      else {
        const R onemix2 {(R {1} + ix) * (R {1} - ix)};
        const R rx2     {rx * rx};
        const R f       {rx2 * (R {2} + rx2 + (R {2} *ix * ix))};
        const R d       {std::sqrt((onemix2 * onemix2) + f)};
        const R dp      {d + onemix2};
        const R dm      {f / dp};
        const R r1      {std::sqrt((dp + rx2) / R {2})};
        const R r2      {rx * ix / r1};
        re = std::log1p(rx2 + dm + (R {2} *((rx * r1) + (ix * r2)))) / R {2};
        im = angle(rx + r1, ix + r2);
      }
    }
    else { // z + √(1 + z²), of 1 + z² = (rx − ix)(rx + ix) + 1 + 2 rx ix i
      const std::complex<R> root {complex_sqrt(std::complex<R> {((rx - ix) * (rx + ix)) + R {1}, R {2} *rx * ix})};
      const std::complex<R> res  {through_log({root.real() + rx, root.imag() + ix})};
      re = res.real();
      im = res.imag();
    }
    return {std::copysign(re, z.real()), std::copysign(im, adjust ? R {1} : z.imag())};
  }

  /*!
   * \brief `asinh(z)`, as glibc's `casinh`: C's Annex G for the infinities and NaNs, z itself at ±0 ± 0i, else
   *        \ref kernel_asinh.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_asinh(const std::complex<R>& z)
  {
    const R x   {z.real()};
    const R y   {z.imag()};
    const R nan {std::numeric_limits<R>::quiet_NaN()};
    if (!std::isfinite(x) || !std::isfinite(y)) {
      if (std::isinf(y)) {
        const R quarter {x == R {0} || std::isfinite(x) ? std::numbers::pi_v<R> / R {2} : std::numbers::pi_v<R> / R {4}};
        return {std::copysign(std::numeric_limits<R>::infinity(), x), std::isnan(x) ? nan : std::copysign(quarter, y)};
      }
      if (!std::isfinite(x)) { // y finite or NaN
        return {x, ((std::isinf(x) && std::isfinite(y)) || y == R {0}) ? std::copysign(R {0}, y) : nan};
      }
      return {nan, nan}; // x finite, y NaN
    }
    if (x == R {0} && y == R {0}) {
      return z;
    }
    return kernel_asinh(z, false);
  }

  //! \brief `asin(z)`, as glibc's `casin`: `−i asinh(iz)`; a NaN part as Annex G has it.
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_asin(const std::complex<R>& z)
  {
    const R nan {std::numeric_limits<R>::quiet_NaN()};
    if (std::isnan(z.real()) || std::isnan(z.imag())) {
      if (z.real() == R {0}) {
        return z;
      }
      if (std::isinf(z.real()) || std::isinf(z.imag())) {
        return {nan, std::copysign(std::numeric_limits<R>::infinity(), z.imag())};
      }
      return {nan, nan};
    }
    const std::complex<R> y {complex_asinh(std::complex<R> {-z.imag(), z.real()})};
    return {y.imag(), -y.real()};
  }

  //! \brief `acos(z)`, as glibc's `cacos`: `π/2 − asin(z)` for the infinities, NaNs and zeros, else \ref kernel_asinh
  //!        of `iz` adjusted.
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_acos(const std::complex<R>& z)
  {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()) || (z.real() == R {0} && z.imag() == R {0})) {
      const std::complex<R> y  {complex_asin(z)};
      const R               re {(std::numbers::pi_v<R> / R {2}) - y.real()};
      return {re == R {0} ? R {0} : re, -y.imag()}; // +0 for −0, as glibc's
    }
    const std::complex<R> y {kernel_asinh(std::complex<R> {-z.imag(), z.real()}, true)};
    return {y.imag(), y.real()};
  }

  //! \brief `acosh(z)`, as glibc's `cacosh`: Annex G for the infinities, NaNs and zeros, else \ref kernel_asinh of `iz`
  //!        adjusted, its sign by `z`'s imaginary part's.
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_acosh(const std::complex<R>& z)
  {
    const R x        {z.real()};
    const R y        {z.imag()};
    const R nan      {std::numeric_limits<R>::quiet_NaN()};
    const R inf      {std::numeric_limits<R>::infinity()};
    const R half_pi  {std::numbers::pi_v<R> / R {2}};
    const R quarter  {std::numbers::pi_v<R> / R {4}};
    if (!std::isfinite(x) || !std::isfinite(y)) {
      if (std::isinf(y)) {
        const R angle {std::isinf(x) ? (x < R {0} ? std::numbers::pi_v<R> - quarter : quarter) : half_pi};
        return {inf, std::isnan(x) ? nan : std::copysign(angle, y)};
      }
      if (std::isinf(x)) {
        return {inf, std::isnan(y) ? nan : std::copysign(std::signbit(x) ? std::numbers::pi_v<R> : R {0}, y)};
      }
      return {nan, x == R {0} ? half_pi : nan}; // a NaN part, the other finite
    }
    if (x == R {0} && y == R {0}) {
      return {R {0}, std::copysign(half_pi, y)};
    }
    const std::complex<R> w {kernel_asinh(std::complex<R> {-y, x}, true)};
    return std::signbit(y) ? std::complex<R> {w.real(), -w.imag()} : std::complex<R> {-w.real(), w.imag()};
  }

  /*!
   * \brief `atan(z)`, as glibc's `catan`: C's Annex G for the infinities and NaNs; `π/2` and the reciprocal past
   *        `16/ε`; else the real part `atan2(2x, 1 − |z|²) / 2`, `1 − |z|²` from \ref unit_circle_offset where it
   *        cancels, the imaginary part `log(|z + i|² / |z − i|²) / 4` (by `log1p(4y / |z − i|²)` where the quotient is
   *        at least ½): \ref complex_atanh's turned a quarter, each operation in glibc's order.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_atan(const std::complex<R>& z)
  {
    const R x       {z.real()};
    const R y       {z.imag()};
    const R half_pi {std::numbers::pi_v<R> / R {2}};
    const R eps     {std::numeric_limits<R>::epsilon()};
    const R nan     {std::numeric_limits<R>::quiet_NaN()};
    if (!std::isfinite(x) || !std::isfinite(y)) {
      if (std::isinf(x)) {
        return {std::copysign(half_pi, x), std::copysign(R {0}, y)};
      }
      if (std::isinf(y)) {
        return {std::isnan(x) ? nan : std::copysign(half_pi, x), std::copysign(R {0}, y)};
      }
      return {nan, y == R {0} ? std::copysign(R {0}, y) : nan}; // a NaN part, the other finite
    }
    if (x == R {0} && y == R {0}) {
      return z;
    }
    if (std::abs(x) >= R {16} / eps || std::abs(y) >= R {16} / eps) {
      R im {};
      if (std::abs(x) <= R {1}) {
        im = R {1} / y;
      }
      else if (std::abs(y) <= R {1}) {
        im = y / x / x;
      }
      else {
        const R h {std::hypot(x / R {2}, y / R {2})};
        im = y / h / h / R {4};
      }
      return {std::copysign(half_pi, x), im};
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
    const R twice_x {R {2} *x};
    const R re      {R {0.5} *std::atan2(twice_x, den)};
    R       im      {};
    if (std::abs(y) == R {1} && std::abs(x) < eps * eps) {
      im = std::copysign(R {0.5}, y) * (std::numbers::ln2_v<R> - std::log(std::abs(x)));
    }
    else {
      const R r2    {std::abs(x) >= eps * eps ? x * x : R {0}};
      const R plus  {y + R {1}};
      const R minus {y - R {1}};
      const R num   {libm_add_product(r2, plus, plus)};
      const R below {libm_add_product(r2, minus, minus)};
      const R f     {num / below};
      if (f < R {0.5}) {
        im = R {0.25} *std::log(f);
      }
      else {
        const R four_y {R {4} *y};
        im = R {0.25} *std::log1p(four_y / below);
      }
    }
    return {re, im};
  }

  //! \brief `log2(z)`, as NumPy's: \ref complex_log times log₂ e, each part.
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_log2(const std::complex<R>& z)
  {
    const std::complex<R> w {complex_log(z)};
    return {w.real() * std::numbers::log2e_v<R>, w.imag() * std::numbers::log2e_v<R>};
  }

  //! \brief `log10(z)`, as NumPy's: \ref complex_log times log₁₀ e, each part.
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_log10(const std::complex<R>& z)
  {
    const std::complex<R> w {complex_log(z)};
    return {w.real() * std::numbers::log10e_v<R>, w.imag() * std::numbers::log10e_v<R>};
  }

  //! \brief `2^z`, as NumPy's: \ref complex_exp of `z` times ln 2, each part.
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_exp2(const std::complex<R>& z)
  {
    return complex_exp(std::complex<R> {z.real() * std::numbers::ln2_v<R>, z.imag() * std::numbers::ln2_v<R>});
  }

  //! \brief `exp(z) − 1`, as NumPy's: `expm1(re) cos(im) − 2 sin²(im/2) + i exp(re) sin(im)`.
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_expm1(const std::complex<R>& z)
  {
    const R a    {std::sin(z.imag() / R {2})};
    const R real {std::expm1(z.real()) * std::cos(z.imag())};
    const R half {R {2} *a * a};
    return {real - half, std::exp(z.real()) * std::sin(z.imag())};
  }

  /*!
   * \brief `log(1 + z)`: the imaginary part `atan2(im, 1 + re)`, as NumPy's; the real part `log|1 + z|`, by
   *        `log1p(d) / 2` of `d = |1 + z|² − 1 = 2re + re² + im²` where `|re|` and `|im|` are at most 3 and `d` at least
   *        −½ (the squares exact by fused multiply-adds, the sum compensated: `d` keeps its digits where it cancels,
   *        about the circle `|1 + z| = 1` and about 0, where NumPy's `log(hypot(1 + re, im))` loses them all); else that,
   *        `|1 + z|` there far from 1 or `1 + re` exact.
   */
  template <std::floating_point R>
  [[nodiscard]] std::complex<R> complex_log1p(const std::complex<R>& z)
  {
    const R x  {z.real()};
    const R y  {z.imag()};
    R       re {};
    R       d  {-R {1}};
    if (std::abs(x) <= R {3} && std::abs(y) <= R {3}) {
      const R xx      {x * x};
      const R xx_err  {std::fma(x, x, -xx)};
      const R yy      {y * y};
      const R yy_err  {std::fma(y, y, -yy)};
      const R twice   {R {2} *x};
      const R first   {twice + yy}; // two-sums: each rounding error kept
      const R bv      {first - twice};
      const R first_e {(twice - (first - bv)) + (yy - bv)};
      const R sum     {first + xx};
      const R cv      {sum - first};
      const R sum_e   {(first - (sum - cv)) + (xx - cv)};
      d = sum + (((first_e + sum_e) + xx_err) + yy_err);
    }
    if (d >= R {-0.5}) {
      re = std::log1p(d) / R {2};
    }
    else {
      re = std::log(std::hypot(x + R {1}, y));
    }
    return {re, std::atan2(y, x + R {1})};
  }
} // namespace numerical::detail

#endif // NUMERICAL_COMPLEX_MATH_HPP
