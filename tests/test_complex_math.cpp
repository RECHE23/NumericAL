// The elementary functions of a complex number as the GNU C library computes them: Annex G's infinities and NaNs
// exactly, each finite branch (scaled near overflow, among subnormal parts, near the unit circle) against a reference
// that does not overflow. NumPy's to the bit on Linux is the Python suite's.
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <complex>
#include <limits>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  // Within `ulps` units in the last place of `want` (both NaN, or equal infinities, count).
  template <typename R>
  bool near(R   got,
            R   want,
            int ulps = 8)
  {
    if (std::isnan(want) || std::isnan(got)) {
      return std::isnan(want) && std::isnan(got);
    }
    if (std::isinf(want) || std::isinf(got)) {
      return got == want;
    }
    const R scale {std::max(std::abs(want), std::numeric_limits<R>::min())};
    return std::abs(got - want) <= static_cast<R>(ulps) * std::numeric_limits<R>::epsilon() * scale;
  }

  template <typename R>
  bool near(const std::complex<R>& got,
            const std::complex<R>& want,
            int                    ulps = 8)
  {
    return near(got.real(), want.real(), ulps) && near(got.imag(), want.imag(), ulps);
  }

  template <typename R>
  bool both_nan(const std::complex<R>& z)
  {
    return std::isnan(z.real()) && std::isnan(z.imag());
  }

  // The same value, the sign of a zero and the NaNs taken into account.
  template <typename R>
  bool same(const std::complex<R>& got,
            const std::complex<R>& want)
  {
    const auto part {[](R a, R b) { return (std::isnan(a) && std::isnan(b)) || (a == b && std::signbit(a) == std::signbit(b)); }};
    return part(got.real(), want.real()) && part(got.imag(), want.imag());
  }

  template <typename R>
  bool exponentials_hold()
  {
    using C = std::complex<R>;
    using numerical::detail::complex_exp;
    const R inf   {std::numeric_limits<R>::infinity()};
    const R nan   {std::numeric_limits<R>::quiet_NaN()};
    const R bound {numerical::detail::exp_bound<R>};
    const R tiny  {std::numeric_limits<R>::denorm_min()};
    bool    ok    {true};
    // finite: the plain product, a subnormal imaginary part taken as its own sine
    ok = ok && near(complex_exp(C {R {1}, R {2}}), C {std::exp(R {1}) * std::cos(R {2}), std::exp(R {1}) * std::sin(R {2})});
    ok = ok && same(complex_exp(C {R {0}, tiny}), C {R {1}, tiny});
    // past the bound once, twice and three times: exp(x / 2)² times the cosine and the sine, or overflow
    for (const R x : {bound + R {1}, (R {2} *bound) + R {1}}) {
      const R half   {std::exp(x / R {2})};
      const R y      {(std::numbers::pi_v<R> / R {2}) - static_cast<R>(1e-30)};
      const C got    {complex_exp(C {x - (R {2} *bound), y})};
      ok = ok && near(got, C {std::exp(x - (R {2} *bound)) * std::cos(y), std::exp(x - (R {2} *bound)) * std::sin(y)});
      const C scaled {complex_exp(C {x, R {-3}})};
      ok = ok && std::isfinite(scaled.real()) == std::isfinite(half * half * std::cos(R {-3}));
    }
    const C small_cosine {complex_exp(C {bound + R {1}, std::numbers::pi_v<R> / R {2}})};
    ok = ok && std::isfinite(small_cosine.real()) && small_cosine.imag() == inf;
    const C twice        {complex_exp(C {(R {2} *bound) + R {1}, std::numbers::pi_v<R> / R {2}})};
    ok = ok && std::isfinite(twice.real()) == (std::abs(std::cos(std::numbers::pi_v<R> / R {2})) < static_cast<R>(1e-30F));
    ok = ok && complex_exp(C {(R {3} *bound) + R {1}, R {1}}) == C {inf, inf};
    // Annex G
    ok = ok && both_nan(complex_exp(C {R {1}, inf})) && both_nan(complex_exp(C {R {1}, nan}));
    ok = ok && same(complex_exp(C {inf, R {0}}), C {inf, R {0}}) && same(complex_exp(C {-inf, static_cast<R>(-0.0)}), C {R {0}, static_cast<R>(-0.0)});
    ok = ok && same(complex_exp(C {inf, R {2}}), C {-inf, inf}) && same(complex_exp(C {-inf, R {2}}), C {static_cast<R>(-0.0), R {0}});
    ok = ok && complex_exp(C {inf, inf}).real() == inf && std::isnan(complex_exp(C {inf, nan}).imag());
    ok = ok && same(complex_exp(C {-inf, -inf}), C {R {0}, static_cast<R>(-0.0)});
    ok = ok && same(complex_exp(C {nan, static_cast<R>(-0.0)}), C {nan, static_cast<R>(-0.0)}) && both_nan(complex_exp(C {nan, R {1}}));
    return ok;
  }

  template <typename R>
  bool hyperbolics_hold()
  {
    using C = std::complex<R>;
    using numerical::detail::complex_cos;
    using numerical::detail::complex_cosh;
    using numerical::detail::complex_sin;
    using numerical::detail::complex_sinh;
    const R inf   {std::numeric_limits<R>::infinity()};
    const R nan   {std::numeric_limits<R>::quiet_NaN()};
    const R bound {numerical::detail::exp_bound<R>};
    const R tiny  {std::numeric_limits<R>::denorm_min()};
    bool    ok    {true};
    // finite, small: the textbook products
    for (const C& z : {C {static_cast<R>(0.5), static_cast<R>(-1.25)}, C {R {-2}, R {3}}, C {R {1}, tiny}}) {
      const R x {z.real()};
      const R y {z.imag()};
      const R s {std::abs(y) > std::numeric_limits<R>::min() ? std::sin(y) : y};
      const R c {std::abs(y) > std::numeric_limits<R>::min() ? std::cos(y) : R {1}};
      ok = ok && near(complex_cosh(z), C {std::cosh(x) * c, std::sinh(x) * s});
      ok = ok && near(complex_sinh(z), C {std::sinh(x) * c, std::cosh(x) * s});
      // sin(z) = −i sinh(i z) and cos(z) = cosh(i z)
      ok = ok && near(complex_sin(C {y, x}), C {std::cosh(x) * s, std::sinh(x) * c});
      ok = ok && near(complex_cos(C {-y, x}), C {std::cosh(x) * c, std::sinh(x) * s});
    }
    // past the bound: exp(|x|) / 2, in one, two or three steps, either sign
    for (const R x : {bound + R {1}, -(bound + R {1}), (R {2} *bound) + R {1}, -((R {3} *bound) + R {1})}) {
      const C ch {complex_cosh(C {x, static_cast<R>(1.5)})};
      const C sh {complex_sinh(C {x, static_cast<R>(1.5)})};
      const C sn {complex_sin(C {static_cast<R>(1.5), x})};
      ok = ok && std::signbit(ch.imag()) == std::signbit(x) && std::signbit(sh.real()) == std::signbit(x);
      ok = ok && std::signbit(sn.imag()) == std::signbit(x) && !std::signbit(sn.real());
      ok = ok && std::abs(ch.real()) >= std::abs(sh.imag()) * std::cos(static_cast<R>(1.5)) / R {2};
    }
    const C once {complex_cosh(C {bound + R {1}, static_cast<R>(1.5)})};
    ok = ok && near(once, C {std::exp((bound + R {1}) / R {2}) * (std::exp((bound + R {1}) / R {2}) / R {2}) * std::cos(static_cast<R>(1.5)), once.imag()}, 16);
    // Annex G
    ok = ok && same(complex_cosh(C {R {0}, inf}), C {nan, R {0}}) && both_nan(complex_cosh(C {R {1}, nan}));
    ok = ok && same(complex_cosh(C {inf, R {2}}), C {-inf, inf}) && same(complex_cosh(C {-inf, R {2}}), C {-inf, -inf});
    ok = ok && same(complex_cosh(C {-inf, R {0}}), C {inf, static_cast<R>(-0.0)}) && std::isnan(complex_cosh(C {inf, nan}).imag());
    ok = ok && same(complex_cosh(C {nan, R {0}}), C {nan, R {0}}) && both_nan(complex_cosh(C {nan, R {1}}));
    ok = ok && same(complex_sinh(C {static_cast<R>(-0.0), inf}), C {static_cast<R>(-0.0), nan}) && both_nan(complex_sinh(C {R {1}, inf}));
    ok = ok && same(complex_sinh(C {R {0}, -inf}), C {R {0}, nan});
    ok = ok && same(complex_sinh(C {inf, R {2}}), C {-inf, inf}) && same(complex_sinh(C {-inf, R {2}}), C {inf, inf});
    ok = ok && same(complex_sinh(C {-inf, R {0}}), C {-inf, R {0}}) && same(complex_sinh(C {inf, static_cast<R>(-0.0)}), C {inf, static_cast<R>(-0.0)});
    ok = ok && complex_sinh(C {inf, nan}).real() == inf && same(complex_sinh(C {nan, static_cast<R>(-0.0)}), C {nan, static_cast<R>(-0.0)});
    ok = ok && both_nan(complex_sinh(C {nan, R {1}}));
    ok = ok && same(complex_sin(C {inf, R {0}}), C {nan, R {0}}) && both_nan(complex_sin(C {inf, R {1}}));
    ok = ok && same(complex_sin(C {R {0}, -inf}), C {R {0}, -inf});
    ok = ok && same(complex_sin(C {static_cast<R>(-0.0), inf}), C {static_cast<R>(-0.0), inf}) && same(complex_sin(C {R {2}, -inf}), C {inf, inf});
    ok = ok && same(complex_sin(C {R {-2}, inf}), C {-inf, -inf}) && std::isnan(complex_sin(C {nan, inf}).real());
    ok = ok && same(complex_sin(C {static_cast<R>(-0.0), nan}), C {static_cast<R>(-0.0), nan}) && both_nan(complex_sin(C {R {1}, nan}));
    ok = ok && same(complex_sin(C {R {0}, nan}), C {R {0}, nan});
    return ok;
  }

  template <typename R>
  bool tangents_hold()
  {
    using C = std::complex<R>;
    using numerical::detail::complex_tan;
    using numerical::detail::complex_tanh;
    const R inf   {std::numeric_limits<R>::infinity()};
    const R nan   {std::numeric_limits<R>::quiet_NaN()};
    const R bound {numerical::detail::half_exp_bound<R>};
    bool    ok    {true};
    // finite: (sinh·cosh + i sin·cos) / (sinh² + cos²), a real part too small to count beside the cosine apart
    for (const C& z : {C {static_cast<R>(0.5), static_cast<R>(-1.25)}, C {R {-2}, R {3}}, C {std::numeric_limits<R>::denorm_min(), static_cast<R>(0.5)}, C {static_cast<R>(1e-30F), R {0}}}) {
      const R x   {z.real()};
      const R y   {z.imag()};
      const R den {(std::sinh(x) * std::sinh(x)) + (std::cos(y) * std::cos(y))};
      ok = ok && near(complex_tanh(z), C {std::sinh(x) * std::cosh(x) / den, std::sin(y) * std::cos(y) / den});
      ok = ok && near(complex_tan(C {y, x}), C {std::sin(y) * std::cos(y) / den, std::sinh(x) * std::cosh(x) / den});
    }
    // past half the bound: ±1, and 4 sin cos / exp(2|x|) in one or two quotients
    for (const R x : {bound + R {1}, -(bound + R {5}), (R {2} *bound) + R {1}}) {
      const C th {complex_tanh(C {x, static_cast<R>(0.5)})};
      const C tn {complex_tan(C {static_cast<R>(0.5), x})};
      ok = ok && th.real() == std::copysign(R {1}, x) && tn.imag() == std::copysign(R {1}, x);
      ok = ok && th.imag() >= R {0} && th.imag() == tn.real();
    }
    // Annex G
    ok = ok && same(complex_tanh(C {inf, R {2}}), C {R {1}, static_cast<R>(-0.0)}) && same(complex_tanh(C {-inf, static_cast<R>(0.5)}), C {R {-1}, R {0}});
    ok = ok && same(complex_tanh(C {inf, nan}), C {R {1}, std::copysign(R {0}, nan)}) && same(complex_tanh(C {nan, static_cast<R>(-0.0)}), C {nan, static_cast<R>(-0.0)});
    ok = ok && same(complex_tanh(C {static_cast<R>(-0.0), inf}), C {static_cast<R>(-0.0), nan}) && both_nan(complex_tanh(C {R {1}, inf}));
    ok = ok && same(complex_tan(C {R {2}, inf}), C {static_cast<R>(-0.0), R {1}}) && same(complex_tan(C {static_cast<R>(0.5), -inf}), C {R {0}, R {-1}});
    ok = ok && same(complex_tan(C {nan, inf}), C {std::copysign(R {0}, nan), R {1}}) && same(complex_tan(C {static_cast<R>(-0.0), nan}), C {static_cast<R>(-0.0), nan});
    ok = ok && same(complex_tan(C {inf, static_cast<R>(-0.0)}), C {nan, static_cast<R>(-0.0)}) && both_nan(complex_tan(C {inf, R {1}}));
    return ok;
  }

  template <typename R>
  bool logarithms_hold()
  {
    using C = std::complex<R>;
    using numerical::detail::complex_log;
    const R    inf {std::numeric_limits<R>::infinity()};
    const R    nan {std::numeric_limits<R>::quiet_NaN()};
    const R    big {std::numeric_limits<R>::max()};
    const R    least {std::numeric_limits<R>::min()};
    const R    eps {std::numeric_limits<R>::epsilon()};
    const R    pi {std::numbers::pi_v<R>};
    bool       ok {true};
    const auto reference {[](R x, R y) { return C {static_cast<R>(std::log(std::hypot(static_cast<double>(x), static_cast<double>(y)))), std::atan2(y, x)}; }};
    // zeros, NaNs and infinities
    ok = ok && same(complex_log(C {static_cast<R>(-0.0), static_cast<R>(-0.0)}), C {-inf, -pi}) && same(complex_log(C {R {0}, R {0}}), C {-inf, R {0}});
    ok = ok && same(complex_log(C {nan, inf}), C {inf, nan}) && same(complex_log(C {-inf, nan}), C {inf, nan});
    ok = ok && both_nan(complex_log(C {nan, R {1}}));
    ok = ok && near(complex_log(C {R {0}, R {2}}), C {std::log(R {2}), std::atan2(R {2}, R {0})}, 2);
    ok = ok && near(complex_log(C {static_cast<R>(1.5), static_cast<R>(-1.25)}), reference(static_cast<R>(1.5), static_cast<R>(-1.25)));
    ok = ok && complex_log(C {-inf, R {1}}).real() == inf && near(complex_log(C {-inf, R {1}}).imag(), pi, 2); // the C library's π
    // each branch: on the circle, just outside (an imaginary part that counts or not), just inside (negligible or
    // near the circle: x² + y² − 1 almost exactly), far from it, scaled near overflow and among subnormal parts
    const R thousandth {static_cast<R>(1e-3F)};
    ok = ok && near(complex_log(C {R {1}, thousandth}), C {std::log1p(thousandth * thousandth) / R {2}, std::atan2(thousandth, R {1})});
    ok = ok && near(complex_log(C {static_cast<R>(1.5), static_cast<R>(0.5)}), reference(static_cast<R>(1.5), static_cast<R>(0.5)));
    ok = ok && near(complex_log(C {static_cast<R>(-1.25), eps / R {4}}), reference(static_cast<R>(-1.25), eps / R {4}));
    ok = ok && near(complex_log(C {static_cast<R>(0.75), eps / R {4}}), reference(static_cast<R>(0.75), eps / R {4}));
    ok = ok && near(complex_log(C {static_cast<R>(0.6), static_cast<R>(-0.8)}), C {complex_log(C {static_cast<R>(0.6), static_cast<R>(-0.8)}).real(), std::atan2(static_cast<R>(-0.8), static_cast<R>(0.6))});
    ok = ok && std::abs(complex_log(C {static_cast<R>(0.6), static_cast<R>(-0.8)}).real()) < static_cast<R>(1e-6F);
    ok = ok && near(complex_log(C {static_cast<R>(-0.7), static_cast<R>(0.7)}), reference(static_cast<R>(-0.7), static_cast<R>(0.7)));
    ok = ok && near(complex_log(C {static_cast<R>(0.5), static_cast<R>(0.25)}), reference(static_cast<R>(0.5), static_cast<R>(0.25)));
    ok = ok && near(complex_log(C {R {3}, R {-4}}), reference(R {3}, R {-4}));
    ok = ok && near(complex_log(C {big, big}).real(), std::log(big) + (std::numbers::ln2_v<R> / R {2}));
    ok = ok && near(complex_log(C {big, least}).real(), std::log(big)) && near(complex_log(C {-big, least * R {4}}).real(), std::log(big));
    ok = ok && near(complex_log(C {least / R {4}, least / R {4}}).real(), std::log(least / R {4}) + (std::numbers::ln2_v<R> / R {2}), 16);
    // the exact x² + y² − 1 near the unit circle: (1 − 2^-20)² + (2^-10)² − 1 = 2^-40 − 2^-20 exactly
    const R x {R {1} - std::ldexp(R {1}, -20)};
    const R y {std::ldexp(R {1}, -10)};
    ok = ok && numerical::detail::unit_circle_offset(x, y) == std::ldexp(R {1}, -40) - std::ldexp(R {1}, -20);
    return ok;
  }

  template <typename R>
  bool tensors_take_them()
  {
    using C = std::complex<R>;
    const numerical::tensor<C> t  {{2}, numerical::buffer<C> {C {static_cast<R>(0.5), R {-1}}, C {R {2}, static_cast<R>(0.25)}}};
    const numerical::tensor<R> r  {{2}, numerical::buffer<R> {static_cast<R>(0.5), R {2}}};
    bool                       ok {true};
    ok = ok && numerical::exp(t).data()[1] == numerical::detail::complex_exp(t.data()[1]) && numerical::exp(r).data()[1] == std::exp(R {2});
    ok = ok && numerical::log(t).data()[1] == numerical::detail::complex_log(t.data()[1]) && numerical::log(r).data()[1] == std::log(R {2});
    ok = ok && numerical::sin(t).data()[1] == numerical::detail::complex_sin(t.data()[1]) && numerical::sin(r).data()[1] == std::sin(R {2});
    ok = ok && numerical::cos(t).data()[1] == numerical::detail::complex_cos(t.data()[1]) && numerical::cos(r).data()[1] == std::cos(R {2});
    ok = ok && numerical::tan(t).data()[1] == numerical::detail::complex_tan(t.data()[1]) && numerical::tan(r).data()[1] == std::tan(R {2});
    ok = ok && numerical::tanh(t).data()[1] == numerical::detail::complex_tanh(t.data()[1]) && numerical::tanh(r).data()[1] == std::tanh(R {2});
    // pow of a complex tensor: the operators' square and reciprocal, the root, std::pow past them, and an exponent
    // tensor of more elements or of a higher rank taken element by element
    const auto exponent {[](C y) { return numerical::tensor<C> {std::vector<std::size_t> {}, numerical::buffer<C> {y}}; }};
    ok = ok && numerical::pow(t, exponent(C {R {2}, R {0}})).data()[1] == t.data()[1] * t.data()[1];
    ok = ok && numerical::pow(t, exponent(C {R {-1}, R {0}})).data()[1] == C {R {1}, R {0}} / t.data()[1];
    ok = ok && numerical::pow(t, exponent(C {static_cast<R>(0.5), R {0}})).data()[1] == numerical::detail::complex_sqrt(t.data()[1]);
    ok = ok && numerical::pow(t, exponent(C {R {3}, R {0}})).data()[1] == std::pow(t.data()[1], C {R {3}, R {0}});
    const numerical::tensor<C> pair   {{2}, numerical::buffer<C> {C {R {2}, R {0}}, C {R {3}, R {0}}}};
    ok = ok && numerical::pow(t, pair).data()[0] == std::pow(t.data()[0], C {R {2}, R {0}});
    const numerical::tensor<C> deeper {{1, 1}, numerical::buffer<C> {C {R {2}, R {0}}}};
    ok = ok && numerical::pow(t, deeper).shape() == std::vector<std::size_t> {1, 2};
    return ok;
  }

  // atanh through each of glibc's branches against the value to 200 bits: |1 + z|² / |1 − z|² at least ½ (log1p) and
  // below it (log); 1 − |z|² by the plain products, inside |z| ≥ 1 and near the unit circle (x² + y² − 1); y below ε/2;
  // x = ±1 with y below ε²; past 16/ε; and Annex G's zeros, infinities and NaNs.
  template <typename R>
  bool inverse_tangents_hold()
  {
    using C = std::complex<R>;
    const R                                    inf {std::numeric_limits<R>::infinity()};
    const R                                    nan {std::numeric_limits<R>::quiet_NaN()};
    const R                                    half_pi {std::numbers::pi_v<R> / R {2}};
    const auto                                 v = [](double x) { return static_cast<R>(x); }; // the references rounded to R
    const std::vector<std::pair<C, C>>         finite {
      {C {v(0.3), v(0.2)}, C {v(0.2957499202364143), v(0.21547449370018826)}},
      {C {v(-0.5), v(0.1)}, C {v(-0.5406096153127009), v(0.1319818618128523)}},
      {C {v(0.8), v(0.3)}, C {v(0.8107982831297268), v(0.573971200330978)}},
      {C {v(-0.6), v(-0.6)}, C {v(-0.4313775209217136), v(-0.6707821967589506)}},
      {C {v(2.0), v(0.1)}, C {v(0.5470961851917696), v(1.5376224984884392)}},
      {C {v(0.5), v(1e-17)}, C {v(0.5493061443340549), v(1.3333333333333335e-17)}},
      {C {v(1.0), v(1e-40)}, C {v(46.39827545016089), v(0.7853981633974483)}},
      {C {v(-1.0), v(-1e-40)}, C {v(-46.39827545016089), v(-0.7853981633974483)}},
      {C {v(1e+20), v(0.5)}, C {v(1e-20), v(1.5707963267948966)}},
      {C {v(0.5), v(1e+20)}, C {v(5e-41), v(1.5707963267948966)}},
      {C {v(1e+20), v(1e+20)}, C {v(5e-21), v(1.5707963267948966)}},
      {C {v(0.25), v(-3.0)}, C {v(0.024865198382659503), v(-1.2509114353815838)}},
      {C {v(1.0), v(0.5)}, C {v(0.708303336014054), v(0.9078874949608804)}},
      {C {v(0.0), v(2.0)}, C {v(0.0), v(1.1071487177940904)}}};
    bool ok {true};
    for (const auto& [z, want] : finite) {
      ok = ok && near(numerical::detail::complex_atanh(z), want);
    }
    const C zero {numerical::detail::complex_atanh(C {R {-0.0}, R {0.0}})};
    ok = ok && zero.real() == R {0} && std::signbit(zero.real()) && zero.imag() == R {0};
    ok = ok && numerical::detail::complex_atanh(C {inf, R {1}}) == C {R {0}, half_pi};
    ok = ok && numerical::detail::complex_atanh(C {R {1}, -inf}) == C {R {0}, -half_pi};
    const C infinite_nan {numerical::detail::complex_atanh(C {-inf, nan})};
    ok = ok && infinite_nan.real() == R {0} && std::signbit(infinite_nan.real()) && std::isnan(infinite_nan.imag());
    const C zero_nan     {numerical::detail::complex_atanh(C {R {0}, nan})};
    ok = ok && zero_nan.real() == R {0} && std::isnan(zero_nan.imag());
    ok = ok && both_nan(numerical::detail::complex_atanh(C {nan, R {1}}));
    ok = ok && both_nan(numerical::detail::complex_atanh(C {R {1}, nan}));
    return ok;
  }

  // The inverse functions through each of glibc's branches of __kernel_casinh (past 1/ε; |re| from ½ beside a tiny
  // |im|; |im| from 1.5 beside a tiny |re|; |im| in (1, 1.5), exactly 1, below 1, each beside a tiny |re| and not;
  // the square root's) as asin, acos (adjusted), asinh, acosh (adjusted), and atan's; NumPy's formulas of log2,
  // log10, exp2, expm1 and log1p's: within 8 units in the last place of the value to 300 bits. Annex G's zeros,
  // infinities and NaNs as glibc 2.41 gives them, the signs of the zeros too (the narrower type's within an ulp of
  // them, rounded).
  template <typename R>
  bool inverse_functions_hold()
  {
    using C                = std::complex<R>;
    using F                = C (*)(const C&);
    const R    inf         {std::numeric_limits<R>::infinity()};
    const R    nan         {std::numeric_limits<R>::quiet_NaN()};
    const R    zero        {R {0}};
    const R    minus_zero  {-R {0}};
    // a double's bits, rounded to R: the references as IEEE patterns, which no named constant stands for
    const auto b           = [](std::uint64_t bits) { return static_cast<R>(std::bit_cast<double>(bits)); };
    namespace d            = numerical::detail;
    const std::array<F, 10> finite_functions {&d::complex_asin<R>,  &d::complex_acos<R>, &d::complex_atan<R>, &d::complex_asinh<R>,
                                              &d::complex_acosh<R>, &d::complex_log1p<R>, &d::complex_log2<R>, &d::complex_log10<R>,
                                              &d::complex_exp2<R>,  &d::complex_expm1<R>};
    const std::array<F, 6>  special_functions {&d::complex_asin<R>,  &d::complex_acos<R>,  &d::complex_atan<R>,
                                               &d::complex_asinh<R>, &d::complex_acosh<R>, &d::complex_atanh<R>};
    struct value {
      std::size_t function;
      C           z;
      C           want;
    };
    const std::vector<value> finite {
      // asin
      {0, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x4047386e22edf4a6U)}},
      {0, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x3bc0000000000000U), b(0x4047386e22edf4a6U)}},
      {0, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0xbff921fb54442d18U), b(0xc047386e22edf4a6U)}},
      {0, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0xbbc0000000000000U), b(0xc047386e22edf4a6U)}},
      {0, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x3bc0000000000000U), b(0x4047386e22edf4a6U)}},
      {0, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x4047386e22edf4a6U)}},
      {0, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0xbbc0000000000000U), b(0xc047386e22edf4a6U)}},
      {0, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0xbff921fb54442d18U), b(0xc047386e22edf4a6U)}},
      {0, C {b(0x3fe8000000000000U), b(0x3c30000000000000U)}, C {b(0x3feb235315c680dcU), b(0x3c383091e6a7f7e7U)}},
      {0, C {b(0x3c30000000000000U), b(0x3fe8000000000000U)}, C {b(0x3c2999999999999aU), b(0x3fe62e42fefa39efU)}},
      {0, C {b(0xbfe8000000000000U), b(0xbc30000000000000U)}, C {b(0xbfeb235315c680dcU), b(0xbc383091e6a7f7e7U)}},
      {0, C {b(0xbc30000000000000U), b(0xbfe8000000000000U)}, C {b(0xbc2999999999999aU), b(0xbfe62e42fefa39efU)}},
      {0, C {b(0x3c30000000000000U), b(0x4000000000000000U)}, C {b(0x3c1c9f25c5bfedd9U), b(0x3ff719218313d087U)}},
      {0, C {b(0x4000000000000000U), b(0x3c30000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3ff5124271980435U)}},
      {0, C {b(0xbc30000000000000U), b(0xc000000000000000U)}, C {b(0xbc1c9f25c5bfedd9U), b(0xbff719218313d087U)}},
      {0, C {b(0xc000000000000000U), b(0xbc30000000000000U)}, C {b(0xbff921fb54442d18U), b(0xbff5124271980435U)}},
      {0, C {b(0x37a0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3793fd8077e70577U), b(0x3ff0c2f0e40d8c01U)}},
      {0, C {b(0x3ff4000000000000U), b(0x37a0000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3fe62e42fefa39efU)}},
      {0, C {b(0xb7a0000000000000U), b(0xbff4000000000000U)}, C {b(0xb793fd8077e70577U), b(0xbff0c2f0e40d8c01U)}},
      {0, C {b(0xbff4000000000000U), b(0xb7a0000000000000U)}, C {b(0xbff921fb54442d18U), b(0xbfe62e42fefa39efU)}},
      {0, C {b(0x3fc0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3fb3f92da25f5c40U), b(0x3ff0ccb107f734e5U)}},
      {0, C {b(0x3ff4000000000000U), b(0x3fc0000000000000U)}, C {b(0x3ff68c4ecfd6412fU), b(0x3fe6df2b17dbd9b2U)}},
      {0, C {b(0xbfc0000000000000U), b(0xbff4000000000000U)}, C {b(0xbfb3f92da25f5c40U), b(0xbff0ccb107f734e5U)}},
      {0, C {b(0xbff4000000000000U), b(0xbfc0000000000000U)}, C {b(0xbff68c4ecfd6412fU), b(0xbfe6df2b17dbd9b2U)}},
      {0, C {b(0x3bd0000000000000U), b(0x3ff0000000000000U)}, C {b(0x3bc6a09e667f3bcdU), b(0x3fec34366179d427U)}},
      {0, C {b(0x3ff0000000000000U), b(0x3bd0000000000000U)}, C {b(0x3ff921fb543c2d18U), b(0x3de0000000000000U)}},
      {0, C {b(0xbbd0000000000000U), b(0xbff0000000000000U)}, C {b(0xbbc6a09e667f3bcdU), b(0xbfec34366179d427U)}},
      {0, C {b(0xbff0000000000000U), b(0xbbd0000000000000U)}, C {b(0xbff921fb543c2d18U), b(0xbde0000000000000U)}},
      {0, C {b(0x3fd4000000000000U), b(0x3ff0000000000000U)}, C {b(0x3fcc29ed5eae8250U), b(0x3fecc26fc2d68070U)}},
      {0, C {b(0x3ff0000000000000U), b(0x3fd4000000000000U)}, C {b(0x3ff06fa4a34900b3U), b(0x3fe251a675b7c290U)}},
      {0, C {b(0xbfd4000000000000U), b(0xbff0000000000000U)}, C {b(0xbfcc29ed5eae8250U), b(0xbfecc26fc2d68070U)}},
      {0, C {b(0xbff0000000000000U), b(0xbfd4000000000000U)}, C {b(0xbff06fa4a34900b3U), b(0xbfe251a675b7c290U)}},
      {0, C {b(0x3fc8000000000000U), b(0x3c70000000000000U)}, C {b(0x3fc82494ed0e78fcU), b(0x3c7049f4b0cadb3bU)}},
      {0, C {b(0x3c70000000000000U), b(0x3fc8000000000000U)}, C {b(0x3c6f73b05f60fd39U), b(0x3fc7dc8ed1f20329U)}},
      {0, C {b(0xbfc8000000000000U), b(0xbc70000000000000U)}, C {b(0xbfc82494ed0e78fcU), b(0xbc7049f4b0cadb3bU)}},
      {0, C {b(0xbc70000000000000U), b(0xbfc8000000000000U)}, C {b(0xbc6f73b05f60fd39U), b(0xbfc7dc8ed1f20329U)}},
      {0, C {b(0x37a0000000000000U), b(0x3fe4000000000000U)}, C {b(0x379b22cd56d87d7eU), b(0x3fe2e27502cbfb33U)}},
      {0, C {b(0x3fe4000000000000U), b(0x37a0000000000000U)}, C {b(0x3fe59aad71ced00fU), b(0x37a47f144fe17f9fU)}},
      {0, C {b(0xb7a0000000000000U), b(0xbfe4000000000000U)}, C {b(0xb79b22cd56d87d7eU), b(0xbfe2e27502cbfb33U)}},
      {0, C {b(0xbfe4000000000000U), b(0xb7a0000000000000U)}, C {b(0xbfe59aad71ced00fU), b(0xb7a47f144fe17f9fU)}},
      {0, C {b(0x3fc8000000000000U), b(0x3fe4000000000000U)}, C {b(0x3fc45d5450a1ff76U), b(0x3fe319e47c2763fbU)}},
      {0, C {b(0x3fe4000000000000U), b(0x3fc8000000000000U)}, C {b(0x3fe4ec78067e9f5dU), b(0x3fcdf658b9cd23bfU)}},
      {0, C {b(0xbfc8000000000000U), b(0xbfe4000000000000U)}, C {b(0xbfc45d5450a1ff76U), b(0xbfe319e47c2763fbU)}},
      {0, C {b(0xbfe4000000000000U), b(0xbfc8000000000000U)}, C {b(0xbfe4ec78067e9f5dU), b(0xbfcdf658b9cd23bfU)}},
      {0, C {b(0x3fea000000000000U), b(0x3fec000000000000U)}, C {b(0x3fe2fa4b662f529eU), b(0x3fed6f835850b8aaU)}},
      {0, C {b(0x3fec000000000000U), b(0x3fea000000000000U)}, C {b(0x3fe505c22427ff72U), b(0x3fecca3e1c011349U)}},
      {0, C {b(0xbfea000000000000U), b(0xbfec000000000000U)}, C {b(0xbfe2fa4b662f529eU), b(0xbfed6f835850b8aaU)}},
      {0, C {b(0xbfec000000000000U), b(0xbfea000000000000U)}, C {b(0xbfe505c22427ff72U), b(0xbfecca3e1c011349U)}},
      {0, C {b(0x4000000000000000U), b(0x4008000000000000U)}, C {b(0x3fe242c9a0c0f98fU), b(0x3fffbbf409ccd560U)}},
      {0, C {b(0x4008000000000000U), b(0x4000000000000000U)}, C {b(0x3feede7b8307a548U), b(0x3fff7f8a7b4f255fU)}},
      {0, C {b(0xc000000000000000U), b(0xc008000000000000U)}, C {b(0xbfe242c9a0c0f98fU), b(0xbfffbbf409ccd560U)}},
      {0, C {b(0xc008000000000000U), b(0xc000000000000000U)}, C {b(0xbfeede7b8307a548U), b(0xbfff7f8a7b4f255fU)}},
      {0, C {b(0x3fe6000000000000U), b(0x0000000000000000U)}, C {b(0x3fe841deb5114bb4U), b(0x0000000000000000U)}},
      {0, C {b(0xbfe6000000000000U), b(0x0000000000000000U)}, C {b(0xbfe841deb5114bb4U), b(0x0000000000000000U)}},
      {0, C {b(0x0000000000000000U), b(0x3fe6000000000000U)}, C {b(0x0000000000000000U), b(0x3fe48e82454a7953U)}},
      {0, C {b(0x0000000000000000U), b(0x3ff8000000000000U)}, C {b(0x0000000000000000U), b(0x3ff31dc0090b63d8U)}},
      // acos
      {1, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x3bc0000000000000U), b(0xc047386e22edf4a6U)}},
      {1, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x3ff921fb54442d18U), b(0xc047386e22edf4a6U)}},
      {1, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0x400921fb54442d18U), b(0x4047386e22edf4a6U)}},
      {1, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x4047386e22edf4a6U)}},
      {1, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x3ff921fb54442d18U), b(0xc047386e22edf4a6U)}},
      {1, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x3bc0000000000000U), b(0xc047386e22edf4a6U)}},
      {1, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x4047386e22edf4a6U)}},
      {1, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0x400921fb54442d18U), b(0x4047386e22edf4a6U)}},
      {1, C {b(0x3fe8000000000000U), b(0x3c30000000000000U)}, C {b(0x3fe720a392c1d955U), b(0xbc383091e6a7f7e7U)}},
      {1, C {b(0x3c30000000000000U), b(0x3fe8000000000000U)}, C {b(0x3ff921fb54442d18U), b(0xbfe62e42fefa39efU)}},
      {1, C {b(0xbfe8000000000000U), b(0xbc30000000000000U)}, C {b(0x400359d26f93b6c3U), b(0x3c383091e6a7f7e7U)}},
      {1, C {b(0xbc30000000000000U), b(0xbfe8000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3fe62e42fefa39efU)}},
      {1, C {b(0x3c30000000000000U), b(0x4000000000000000U)}, C {b(0x3ff921fb54442d18U), b(0xbff719218313d087U)}},
      {1, C {b(0x4000000000000000U), b(0x3c30000000000000U)}, C {b(0x3c2279a74590331cU), b(0xbff5124271980435U)}},
      {1, C {b(0xbc30000000000000U), b(0xc000000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3ff719218313d087U)}},
      {1, C {b(0xc000000000000000U), b(0xbc30000000000000U)}, C {b(0x400921fb54442d18U), b(0x3ff5124271980435U)}},
      {1, C {b(0x37a0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3ff921fb54442d18U), b(0xbff0c2f0e40d8c01U)}},
      {1, C {b(0x3ff4000000000000U), b(0x37a0000000000000U)}, C {b(0x37a5555555555555U), b(0xbfe62e42fefa39efU)}},
      {1, C {b(0xb7a0000000000000U), b(0xbff4000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3ff0c2f0e40d8c01U)}},
      {1, C {b(0xbff4000000000000U), b(0xb7a0000000000000U)}, C {b(0x400921fb54442d18U), b(0x3fe62e42fefa39efU)}},
      {1, C {b(0x3fc0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3ff7e2687a1e3754U), b(0xbff0ccb107f734e5U)}},
      {1, C {b(0x3ff4000000000000U), b(0x3fc0000000000000U)}, C {b(0x3fc4ad64236f5f49U), b(0xbfe6df2b17dbd9b2U)}},
      {1, C {b(0xbfc0000000000000U), b(0xbff4000000000000U)}, C {b(0x3ffa618e2e6a22dcU), b(0x3ff0ccb107f734e5U)}},
      {1, C {b(0xbff4000000000000U), b(0xbfc0000000000000U)}, C {b(0x4007d725120d3724U), b(0x3fe6df2b17dbd9b2U)}},
      {1, C {b(0x3bd0000000000000U), b(0x3ff0000000000000U)}, C {b(0x3ff921fb54442d18U), b(0xbfec34366179d427U)}},
      {1, C {b(0x3ff0000000000000U), b(0x3bd0000000000000U)}, C {b(0x3de0000000000000U), b(0xbde0000000000000U)}},
      {1, C {b(0xbbd0000000000000U), b(0xbff0000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3fec34366179d427U)}},
      {1, C {b(0xbff0000000000000U), b(0xbbd0000000000000U)}, C {b(0x400921fb54402d18U), b(0x3de0000000000000U)}},
      {1, C {b(0x3fd4000000000000U), b(0x3ff0000000000000U)}, C {b(0x3ff59cbda86e5cceU), b(0xbfecc26fc2d68070U)}},
      {1, C {b(0x3ff0000000000000U), b(0x3fd4000000000000U)}, C {b(0x3fe164ad61f658cbU), b(0xbfe251a675b7c290U)}},
      {1, C {b(0xbfd4000000000000U), b(0xbff0000000000000U)}, C {b(0x3ffca7390019fd62U), b(0x3fecc26fc2d68070U)}},
      {1, C {b(0xbff0000000000000U), b(0xbfd4000000000000U)}, C {b(0x4004c8cffbc696e6U), b(0x3fe251a675b7c290U)}},
      {1, C {b(0x3fc8000000000000U), b(0x3c70000000000000U)}, C {b(0x3ff61d68b6a25df9U), b(0xbc7049f4b0cadb3bU)}},
      {1, C {b(0x3c70000000000000U), b(0x3fc8000000000000U)}, C {b(0x3ff921fb54442d18U), b(0xbfc7dc8ed1f20329U)}},
      {1, C {b(0xbfc8000000000000U), b(0xbc70000000000000U)}, C {b(0x3ffc268df1e5fc38U), b(0x3c7049f4b0cadb3bU)}},
      {1, C {b(0xbc70000000000000U), b(0xbfc8000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3fc7dc8ed1f20329U)}},
      {1, C {b(0x37a0000000000000U), b(0x3fe4000000000000U)}, C {b(0x3ff921fb54442d18U), b(0xbfe2e27502cbfb33U)}},
      {1, C {b(0x3fe4000000000000U), b(0x37a0000000000000U)}, C {b(0x3feca94936b98a22U), b(0xb7a47f144fe17f9fU)}},
      {1, C {b(0xb7a0000000000000U), b(0xbfe4000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3fe2e27502cbfb33U)}},
      {1, C {b(0xbfe4000000000000U), b(0xb7a0000000000000U)}, C {b(0x4001f7a90695ca90U), b(0x37a47f144fe17f9fU)}},
      {1, C {b(0x3fc8000000000000U), b(0x3fe4000000000000U)}, C {b(0x3ff69650ca2fed2aU), b(0xbfe319e47c2763fbU)}},
      {1, C {b(0x3fe4000000000000U), b(0x3fc8000000000000U)}, C {b(0x3fed577ea209bad4U), b(0xbfcdf658b9cd23bfU)}},
      {1, C {b(0xbfc8000000000000U), b(0xbfe4000000000000U)}, C {b(0x3ffbada5de586d07U), b(0x3fe319e47c2763fbU)}},
      {1, C {b(0xbfe4000000000000U), b(0xbfc8000000000000U)}, C {b(0x4001cc1babc1be63U), b(0x3fcdf658b9cd23bfU)}},
      {1, C {b(0x3fea000000000000U), b(0x3fec000000000000U)}, C {b(0x3fef49ab42590792U), b(0xbfed6f835850b8aaU)}},
      {1, C {b(0x3fec000000000000U), b(0x3fea000000000000U)}, C {b(0x3fed3e3484605abeU), b(0xbfecca3e1c011349U)}},
      {1, C {b(0xbfea000000000000U), b(0xbfec000000000000U)}, C {b(0x40014f9083adeb34U), b(0x3fed6f835850b8aaU)}},
      {1, C {b(0xbfec000000000000U), b(0xbfea000000000000U)}, C {b(0x4001d26e332c1669U), b(0x3fecca3e1c011349U)}},
      {1, C {b(0x4000000000000000U), b(0x4008000000000000U)}, C {b(0x3ff0009683e3b051U), b(0xbfffbbf409ccd560U)}},
      {1, C {b(0x4008000000000000U), b(0x4000000000000000U)}, C {b(0x3fe3657b2580b4e9U), b(0xbfff7f8a7b4f255fU)}},
      {1, C {b(0xc000000000000000U), b(0xc008000000000000U)}, C {b(0x400121b0125254f0U), b(0x3fffbbf409ccd560U)}},
      {1, C {b(0xc008000000000000U), b(0xc000000000000000U)}, C {b(0x4004489c8ae3ffdeU), b(0x3fff7f8a7b4f255fU)}},
      {1, C {b(0x3fe6000000000000U), b(0x0000000000000000U)}, C {b(0x3fea0217f3770e7dU), b(0x0000000000000000U)}},
      {1, C {b(0xbfe6000000000000U), b(0x0000000000000000U)}, C {b(0x4002a17557666979U), b(0x0000000000000000U)}},
      {1, C {b(0x0000000000000000U), b(0x3fe6000000000000U)}, C {b(0x3ff921fb54442d18U), b(0xbfe48e82454a7953U)}},
      {1, C {b(0x0000000000000000U), b(0x3ff8000000000000U)}, C {b(0x3ff921fb54442d18U), b(0xbff31dc0090b63d8U)}},
      // atan
      {2, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x37a0000000000000U)}},
      {2, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3bd0000000000000U)}},
      {2, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0xbff921fb54442d18U), b(0xb7a0000000000000U)}},
      {2, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0xbff921fb54442d18U), b(0xbbd0000000000000U)}},
      {2, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3bd0000000000000U)}},
      {2, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x37a0000000000000U)}},
      {2, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0xbff921fb54442d18U), b(0xbbd0000000000000U)}},
      {2, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0xbff921fb54442d18U), b(0xb7a0000000000000U)}},
      {2, C {b(0x3fe8000000000000U), b(0x3c30000000000000U)}, C {b(0x3fe4978fa3269ee1U), b(0x3c247ae147ae147bU)}},
      {2, C {b(0x3c30000000000000U), b(0x3fe8000000000000U)}, C {b(0x3c42492492492492U), b(0x3fef2272ae325a57U)}},
      {2, C {b(0xbfe8000000000000U), b(0xbc30000000000000U)}, C {b(0xbfe4978fa3269ee1U), b(0xbc247ae147ae147bU)}},
      {2, C {b(0xbc30000000000000U), b(0xbfe8000000000000U)}, C {b(0xbc42492492492492U), b(0xbfef2272ae325a57U)}},
      {2, C {b(0x3c30000000000000U), b(0x4000000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3fe193ea7aad030bU)}},
      {2, C {b(0x4000000000000000U), b(0x3c30000000000000U)}, C {b(0x3ff1b6e192ebbe44U), b(0x3c0999999999999aU)}},
      {2, C {b(0xbc30000000000000U), b(0xc000000000000000U)}, C {b(0xbff921fb54442d18U), b(0xbfe193ea7aad030bU)}},
      {2, C {b(0xc000000000000000U), b(0xbc30000000000000U)}, C {b(0xbff1b6e192ebbe44U), b(0xbc0999999999999aU)}},
      {2, C {b(0x37a0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3ff193ea7aad030bU)}},
      {2, C {b(0x3ff4000000000000U), b(0x37a0000000000000U)}, C {b(0x3fecac7c57846f9eU), b(0x3788f9c18f9c18faU)}},
      {2, C {b(0xb7a0000000000000U), b(0xbff4000000000000U)}, C {b(0xbff921fb54442d18U), b(0xbff193ea7aad030bU)}},
      {2, C {b(0xbff4000000000000U), b(0xb7a0000000000000U)}, C {b(0xbfecac7c57846f9eU), b(0xb788f9c18f9c18faU)}},
      {2, C {b(0x3fc0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3ff5de17a6e00481U), b(0x3ff0b29293942975U)}},
      {2, C {b(0x3ff4000000000000U), b(0x3fc0000000000000U)}, C {b(0x3fecc4d1bdc76dfbU), b(0x3fa8e709c554ffb6U)}},
      {2, C {b(0xbfc0000000000000U), b(0xbff4000000000000U)}, C {b(0xbff5de17a6e00481U), b(0xbff0b29293942975U)}},
      {2, C {b(0xbff4000000000000U), b(0xbfc0000000000000U)}, C {b(0xbfecc4d1bdc76dfbU), b(0xbfa8e709c554ffb6U)}},
      {2, C {b(0x3bd0000000000000U), b(0x3ff0000000000000U)}, C {b(0x3fe921fb54442d18U), b(0x4037386e22edf4a6U)}},
      {2, C {b(0x3ff0000000000000U), b(0x3bd0000000000000U)}, C {b(0x3fe921fb54442d18U), b(0x3bc0000000000000U)}},
      {2, C {b(0xbbd0000000000000U), b(0xbff0000000000000U)}, C {b(0xbfe921fb54442d18U), b(0xc037386e22edf4a6U)}},
      {2, C {b(0xbff0000000000000U), b(0xbbd0000000000000U)}, C {b(0xbfe921fb54442d18U), b(0xbbc0000000000000U)}},
      {2, C {b(0x3fd4000000000000U), b(0x3ff0000000000000U)}, C {b(0x3feb9cd9315cf966U), b(0x3fede4cbc4dd5703U)}},
      {2, C {b(0x3ff0000000000000U), b(0x3fd4000000000000U)}, C {b(0x3fe9e9d2b279b8eeU), b(0x3fc3aa504226855fU)}},
      {2, C {b(0xbfd4000000000000U), b(0xbff0000000000000U)}, C {b(0xbfeb9cd9315cf966U), b(0xbfede4cbc4dd5703U)}},
      {2, C {b(0xbff0000000000000U), b(0xbfd4000000000000U)}, C {b(0xbfe9e9d2b279b8eeU), b(0xbfc3aa504226855fU)}},
      {2, C {b(0x3fc8000000000000U), b(0x3c70000000000000U)}, C {b(0x3fc7b97b4bce5b02U), b(0x3c6ee9c7f8458e02U)}},
      {2, C {b(0x3c70000000000000U), b(0x3fc8000000000000U)}, C {b(0x3c70953f39010954U), b(0x3fc8498ed69936dcU)}},
      {2, C {b(0xbfc8000000000000U), b(0xbc70000000000000U)}, C {b(0xbfc7b97b4bce5b02U), b(0xbc6ee9c7f8458e02U)}},
      {2, C {b(0xbc70000000000000U), b(0xbfc8000000000000U)}, C {b(0xbc70953f39010954U), b(0xbfc8498ed69936dcU)}},
      {2, C {b(0x37a0000000000000U), b(0x3fe4000000000000U)}, C {b(0x37aa41a41a41a41aU), b(0x3fe7761ddbb61597U)}},
      {2, C {b(0x3fe4000000000000U), b(0x37a0000000000000U)}, C {b(0x3fe1e00babdefeb4U), b(0x379702e05c0b8170U)}},
      {2, C {b(0xb7a0000000000000U), b(0xbfe4000000000000U)}, C {b(0xb7aa41a41a41a41aU), b(0xbfe7761ddbb61597U)}},
      {2, C {b(0xbfe4000000000000U), b(0xb7a0000000000000U)}, C {b(0xbfe1e00babdefeb4U), b(0xb79702e05c0b8170U)}},
      {2, C {b(0x3fc8000000000000U), b(0x3fe4000000000000U)}, C {b(0x3fd28345369bc7c5U), b(0x3fe5c8347afdfe2fU)}},
      {2, C {b(0x3fe4000000000000U), b(0x3fc8000000000000U)}, C {b(0x3fe23e26a03af7a6U), b(0x3fc13cebd0424e5fU)}},
      {2, C {b(0xbfc8000000000000U), b(0xbfe4000000000000U)}, C {b(0xbfd28345369bc7c5U), b(0xbfe5c8347afdfe2fU)}},
      {2, C {b(0xbfe4000000000000U), b(0xbfc8000000000000U)}, C {b(0xbfe23e26a03af7a6U), b(0xbfc13cebd0424e5fU)}},
      {2, C {b(0x3fea000000000000U), b(0x3fec000000000000U)}, C {b(0x3fed3b9dd7b7835aU), b(0x3fdd23954fc5fe66U)}},
      {2, C {b(0x3fec000000000000U), b(0x3fea000000000000U)}, C {b(0x3fecf38f0fedcbcdU), b(0x3fd9efeead33c958U)}},
      {2, C {b(0xbfea000000000000U), b(0xbfec000000000000U)}, C {b(0xbfed3b9dd7b7835aU), b(0xbfdd23954fc5fe66U)}},
      {2, C {b(0xbfec000000000000U), b(0xbfea000000000000U)}, C {b(0xbfecf38f0fedcbcdU), b(0xbfd9efeead33c958U)}},
      {2, C {b(0x4000000000000000U), b(0x4008000000000000U)}, C {b(0x3ff68f095fdf593cU), b(0x3fcd5240f0e0e078U)}},
      {2, C {b(0x4008000000000000U), b(0x4000000000000000U)}, C {b(0x3ff56c6e7397f5aeU), b(0x3fc2cf25fad8f1c4U)}},
      {2, C {b(0xc000000000000000U), b(0xc008000000000000U)}, C {b(0xbff68f095fdf593cU), b(0xbfcd5240f0e0e078U)}},
      {2, C {b(0xc008000000000000U), b(0xc000000000000000U)}, C {b(0xbff56c6e7397f5aeU), b(0xbfc2cf25fad8f1c4U)}},
      {2, C {b(0x3fe6000000000000U), b(0x0000000000000000U)}, C {b(0x3fe345f01cce37bbU), b(0x0000000000000000U)}},
      {2, C {b(0xbfe6000000000000U), b(0x0000000000000000U)}, C {b(0xbfe345f01cce37bbU), b(0x0000000000000000U)}},
      {2, C {b(0x0000000000000000U), b(0x3fe6000000000000U)}, C {b(0x0000000000000000U), b(0x3feafb7d78197becU)}},
      {2, C {b(0x3ff8000000000000U), b(0x0000000000000000U)}, C {b(0x3fef730bd281f69bU), b(0x0000000000000000U)}},
      {2, C {b(0xbff8000000000000U), b(0x0000000000000000U)}, C {b(0xbfef730bd281f69bU), b(0x0000000000000000U)}},
      {2, C {b(0x0000000000000000U), b(0x3ff8000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3fe9c041f7ed8d33U)}},
      // asinh
      {3, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x4047386e22edf4a6U), b(0x3bc0000000000000U)}},
      {3, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x4047386e22edf4a6U), b(0x3ff921fb54442d18U)}},
      {3, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0xc047386e22edf4a6U), b(0xbbc0000000000000U)}},
      {3, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0xc047386e22edf4a6U), b(0xbff921fb54442d18U)}},
      {3, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x4047386e22edf4a6U), b(0x3ff921fb54442d18U)}},
      {3, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x4047386e22edf4a6U), b(0x3bc0000000000000U)}},
      {3, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0xc047386e22edf4a6U), b(0xbff921fb54442d18U)}},
      {3, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0xc047386e22edf4a6U), b(0xbbc0000000000000U)}},
      {3, C {b(0x3fe8000000000000U), b(0x3c30000000000000U)}, C {b(0x3fe62e42fefa39efU), b(0x3c2999999999999aU)}},
      {3, C {b(0x3c30000000000000U), b(0x3fe8000000000000U)}, C {b(0x3c383091e6a7f7e7U), b(0x3feb235315c680dcU)}},
      {3, C {b(0xbfe8000000000000U), b(0xbc30000000000000U)}, C {b(0xbfe62e42fefa39efU), b(0xbc2999999999999aU)}},
      {3, C {b(0xbc30000000000000U), b(0xbfe8000000000000U)}, C {b(0xbc383091e6a7f7e7U), b(0xbfeb235315c680dcU)}},
      {3, C {b(0x3c30000000000000U), b(0x4000000000000000U)}, C {b(0x3ff5124271980435U), b(0x3ff921fb54442d18U)}},
      {3, C {b(0x4000000000000000U), b(0x3c30000000000000U)}, C {b(0x3ff719218313d087U), b(0x3c1c9f25c5bfedd9U)}},
      {3, C {b(0xbc30000000000000U), b(0xc000000000000000U)}, C {b(0xbff5124271980435U), b(0xbff921fb54442d18U)}},
      {3, C {b(0xc000000000000000U), b(0xbc30000000000000U)}, C {b(0xbff719218313d087U), b(0xbc1c9f25c5bfedd9U)}},
      {3, C {b(0x37a0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3fe62e42fefa39efU), b(0x3ff921fb54442d18U)}},
      {3, C {b(0x3ff4000000000000U), b(0x37a0000000000000U)}, C {b(0x3ff0c2f0e40d8c01U), b(0x3793fd8077e70577U)}},
      {3, C {b(0xb7a0000000000000U), b(0xbff4000000000000U)}, C {b(0xbfe62e42fefa39efU), b(0xbff921fb54442d18U)}},
      {3, C {b(0xbff4000000000000U), b(0xb7a0000000000000U)}, C {b(0xbff0c2f0e40d8c01U), b(0xb793fd8077e70577U)}},
      {3, C {b(0x3fc0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3fe6df2b17dbd9b2U), b(0x3ff68c4ecfd6412fU)}},
      {3, C {b(0x3ff4000000000000U), b(0x3fc0000000000000U)}, C {b(0x3ff0ccb107f734e5U), b(0x3fb3f92da25f5c40U)}},
      {3, C {b(0xbfc0000000000000U), b(0xbff4000000000000U)}, C {b(0xbfe6df2b17dbd9b2U), b(0xbff68c4ecfd6412fU)}},
      {3, C {b(0xbff4000000000000U), b(0xbfc0000000000000U)}, C {b(0xbff0ccb107f734e5U), b(0xbfb3f92da25f5c40U)}},
      {3, C {b(0x3bd0000000000000U), b(0x3ff0000000000000U)}, C {b(0x3de0000000000000U), b(0x3ff921fb543c2d18U)}},
      {3, C {b(0x3ff0000000000000U), b(0x3bd0000000000000U)}, C {b(0x3fec34366179d427U), b(0x3bc6a09e667f3bcdU)}},
      {3, C {b(0xbbd0000000000000U), b(0xbff0000000000000U)}, C {b(0xbde0000000000000U), b(0xbff921fb543c2d18U)}},
      {3, C {b(0xbff0000000000000U), b(0xbbd0000000000000U)}, C {b(0xbfec34366179d427U), b(0xbbc6a09e667f3bcdU)}},
      {3, C {b(0x3fd4000000000000U), b(0x3ff0000000000000U)}, C {b(0x3fe251a675b7c290U), b(0x3ff06fa4a34900b3U)}},
      {3, C {b(0x3ff0000000000000U), b(0x3fd4000000000000U)}, C {b(0x3fecc26fc2d68070U), b(0x3fcc29ed5eae8250U)}},
      {3, C {b(0xbfd4000000000000U), b(0xbff0000000000000U)}, C {b(0xbfe251a675b7c290U), b(0xbff06fa4a34900b3U)}},
      {3, C {b(0xbff0000000000000U), b(0xbfd4000000000000U)}, C {b(0xbfecc26fc2d68070U), b(0xbfcc29ed5eae8250U)}},
      {3, C {b(0x3fc8000000000000U), b(0x3c70000000000000U)}, C {b(0x3fc7dc8ed1f20329U), b(0x3c6f73b05f60fd39U)}},
      {3, C {b(0x3c70000000000000U), b(0x3fc8000000000000U)}, C {b(0x3c7049f4b0cadb3bU), b(0x3fc82494ed0e78fcU)}},
      {3, C {b(0xbfc8000000000000U), b(0xbc70000000000000U)}, C {b(0xbfc7dc8ed1f20329U), b(0xbc6f73b05f60fd39U)}},
      {3, C {b(0xbc70000000000000U), b(0xbfc8000000000000U)}, C {b(0xbc7049f4b0cadb3bU), b(0xbfc82494ed0e78fcU)}},
      {3, C {b(0x37a0000000000000U), b(0x3fe4000000000000U)}, C {b(0x37a47f144fe17f9fU), b(0x3fe59aad71ced00fU)}},
      {3, C {b(0x3fe4000000000000U), b(0x37a0000000000000U)}, C {b(0x3fe2e27502cbfb33U), b(0x379b22cd56d87d7eU)}},
      {3, C {b(0xb7a0000000000000U), b(0xbfe4000000000000U)}, C {b(0xb7a47f144fe17f9fU), b(0xbfe59aad71ced00fU)}},
      {3, C {b(0xbfe4000000000000U), b(0xb7a0000000000000U)}, C {b(0xbfe2e27502cbfb33U), b(0xb79b22cd56d87d7eU)}},
      {3, C {b(0x3fc8000000000000U), b(0x3fe4000000000000U)}, C {b(0x3fcdf658b9cd23bfU), b(0x3fe4ec78067e9f5dU)}},
      {3, C {b(0x3fe4000000000000U), b(0x3fc8000000000000U)}, C {b(0x3fe319e47c2763fbU), b(0x3fc45d5450a1ff76U)}},
      {3, C {b(0xbfc8000000000000U), b(0xbfe4000000000000U)}, C {b(0xbfcdf658b9cd23bfU), b(0xbfe4ec78067e9f5dU)}},
      {3, C {b(0xbfe4000000000000U), b(0xbfc8000000000000U)}, C {b(0xbfe319e47c2763fbU), b(0xbfc45d5450a1ff76U)}},
      {3, C {b(0x3fea000000000000U), b(0x3fec000000000000U)}, C {b(0x3fecca3e1c011349U), b(0x3fe505c22427ff72U)}},
      {3, C {b(0x3fec000000000000U), b(0x3fea000000000000U)}, C {b(0x3fed6f835850b8aaU), b(0x3fe2fa4b662f529eU)}},
      {3, C {b(0xbfea000000000000U), b(0xbfec000000000000U)}, C {b(0xbfecca3e1c011349U), b(0xbfe505c22427ff72U)}},
      {3, C {b(0xbfec000000000000U), b(0xbfea000000000000U)}, C {b(0xbfed6f835850b8aaU), b(0xbfe2fa4b662f529eU)}},
      {3, C {b(0x4000000000000000U), b(0x4008000000000000U)}, C {b(0x3fff7f8a7b4f255fU), b(0x3feede7b8307a548U)}},
      {3, C {b(0x4008000000000000U), b(0x4000000000000000U)}, C {b(0x3fffbbf409ccd560U), b(0x3fe242c9a0c0f98fU)}},
      {3, C {b(0xc000000000000000U), b(0xc008000000000000U)}, C {b(0xbfff7f8a7b4f255fU), b(0xbfeede7b8307a548U)}},
      {3, C {b(0xc008000000000000U), b(0xc000000000000000U)}, C {b(0xbfffbbf409ccd560U), b(0xbfe242c9a0c0f98fU)}},
      {3, C {b(0x3fe6000000000000U), b(0x0000000000000000U)}, C {b(0x3fe48e82454a7953U), b(0x0000000000000000U)}},
      {3, C {b(0xbfe6000000000000U), b(0x0000000000000000U)}, C {b(0xbfe48e82454a7953U), b(0x0000000000000000U)}},
      {3, C {b(0x0000000000000000U), b(0x3fe6000000000000U)}, C {b(0x0000000000000000U), b(0x3fe841deb5114bb4U)}},
      {3, C {b(0x3ff8000000000000U), b(0x0000000000000000U)}, C {b(0x3ff31dc0090b63d8U), b(0x0000000000000000U)}},
      {3, C {b(0xbff8000000000000U), b(0x0000000000000000U)}, C {b(0xbff31dc0090b63d8U), b(0x0000000000000000U)}},
      {3, C {b(0x0000000000000000U), b(0x3ff8000000000000U)}, C {b(0x3feecc2caec5160aU), b(0x3ff921fb54442d18U)}},
      // acosh
      {4, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x4047386e22edf4a6U), b(0x3bc0000000000000U)}},
      {4, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x4047386e22edf4a6U), b(0x3ff921fb54442d18U)}},
      {4, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0x4047386e22edf4a6U), b(0xc00921fb54442d18U)}},
      {4, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0x4047386e22edf4a6U), b(0xbff921fb54442d18U)}},
      {4, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x4047386e22edf4a6U), b(0x3ff921fb54442d18U)}},
      {4, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x4047386e22edf4a6U), b(0x3bc0000000000000U)}},
      {4, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0x4047386e22edf4a6U), b(0xbff921fb54442d18U)}},
      {4, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0x4047386e22edf4a6U), b(0xc00921fb54442d18U)}},
      {4, C {b(0x3fe8000000000000U), b(0x3c30000000000000U)}, C {b(0x3c383091e6a7f7e7U), b(0x3fe720a392c1d955U)}},
      {4, C {b(0x3c30000000000000U), b(0x3fe8000000000000U)}, C {b(0x3fe62e42fefa39efU), b(0x3ff921fb54442d18U)}},
      {4, C {b(0xbfe8000000000000U), b(0xbc30000000000000U)}, C {b(0x3c383091e6a7f7e7U), b(0xc00359d26f93b6c3U)}},
      {4, C {b(0xbc30000000000000U), b(0xbfe8000000000000U)}, C {b(0x3fe62e42fefa39efU), b(0xbff921fb54442d18U)}},
      {4, C {b(0x3c30000000000000U), b(0x4000000000000000U)}, C {b(0x3ff719218313d087U), b(0x3ff921fb54442d18U)}},
      {4, C {b(0x4000000000000000U), b(0x3c30000000000000U)}, C {b(0x3ff5124271980435U), b(0x3c2279a74590331cU)}},
      {4, C {b(0xbc30000000000000U), b(0xc000000000000000U)}, C {b(0x3ff719218313d087U), b(0xbff921fb54442d18U)}},
      {4, C {b(0xc000000000000000U), b(0xbc30000000000000U)}, C {b(0x3ff5124271980435U), b(0xc00921fb54442d18U)}},
      {4, C {b(0x37a0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3ff0c2f0e40d8c01U), b(0x3ff921fb54442d18U)}},
      {4, C {b(0x3ff4000000000000U), b(0x37a0000000000000U)}, C {b(0x3fe62e42fefa39efU), b(0x37a5555555555555U)}},
      {4, C {b(0xb7a0000000000000U), b(0xbff4000000000000U)}, C {b(0x3ff0c2f0e40d8c01U), b(0xbff921fb54442d18U)}},
      {4, C {b(0xbff4000000000000U), b(0xb7a0000000000000U)}, C {b(0x3fe62e42fefa39efU), b(0xc00921fb54442d18U)}},
      {4, C {b(0x3fc0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3ff0ccb107f734e5U), b(0x3ff7e2687a1e3754U)}},
      {4, C {b(0x3ff4000000000000U), b(0x3fc0000000000000U)}, C {b(0x3fe6df2b17dbd9b2U), b(0x3fc4ad64236f5f49U)}},
      {4, C {b(0xbfc0000000000000U), b(0xbff4000000000000U)}, C {b(0x3ff0ccb107f734e5U), b(0xbffa618e2e6a22dcU)}},
      {4, C {b(0xbff4000000000000U), b(0xbfc0000000000000U)}, C {b(0x3fe6df2b17dbd9b2U), b(0xc007d725120d3724U)}},
      {4, C {b(0x3bd0000000000000U), b(0x3ff0000000000000U)}, C {b(0x3fec34366179d427U), b(0x3ff921fb54442d18U)}},
      {4, C {b(0x3ff0000000000000U), b(0x3bd0000000000000U)}, C {b(0x3de0000000000000U), b(0x3de0000000000000U)}},
      {4, C {b(0xbbd0000000000000U), b(0xbff0000000000000U)}, C {b(0x3fec34366179d427U), b(0xbff921fb54442d18U)}},
      {4, C {b(0xbff0000000000000U), b(0xbbd0000000000000U)}, C {b(0x3de0000000000000U), b(0xc00921fb54402d18U)}},
      {4, C {b(0x3fd4000000000000U), b(0x3ff0000000000000U)}, C {b(0x3fecc26fc2d68070U), b(0x3ff59cbda86e5cceU)}},
      {4, C {b(0x3ff0000000000000U), b(0x3fd4000000000000U)}, C {b(0x3fe251a675b7c290U), b(0x3fe164ad61f658cbU)}},
      {4, C {b(0xbfd4000000000000U), b(0xbff0000000000000U)}, C {b(0x3fecc26fc2d68070U), b(0xbffca7390019fd62U)}},
      {4, C {b(0xbff0000000000000U), b(0xbfd4000000000000U)}, C {b(0x3fe251a675b7c290U), b(0xc004c8cffbc696e6U)}},
      {4, C {b(0x3fc8000000000000U), b(0x3c70000000000000U)}, C {b(0x3c7049f4b0cadb3bU), b(0x3ff61d68b6a25df9U)}},
      {4, C {b(0x3c70000000000000U), b(0x3fc8000000000000U)}, C {b(0x3fc7dc8ed1f20329U), b(0x3ff921fb54442d18U)}},
      {4, C {b(0xbfc8000000000000U), b(0xbc70000000000000U)}, C {b(0x3c7049f4b0cadb3bU), b(0xbffc268df1e5fc38U)}},
      {4, C {b(0xbc70000000000000U), b(0xbfc8000000000000U)}, C {b(0x3fc7dc8ed1f20329U), b(0xbff921fb54442d18U)}},
      {4, C {b(0x37a0000000000000U), b(0x3fe4000000000000U)}, C {b(0x3fe2e27502cbfb33U), b(0x3ff921fb54442d18U)}},
      {4, C {b(0x3fe4000000000000U), b(0x37a0000000000000U)}, C {b(0x37a47f144fe17f9fU), b(0x3feca94936b98a22U)}},
      {4, C {b(0xb7a0000000000000U), b(0xbfe4000000000000U)}, C {b(0x3fe2e27502cbfb33U), b(0xbff921fb54442d18U)}},
      {4, C {b(0xbfe4000000000000U), b(0xb7a0000000000000U)}, C {b(0x37a47f144fe17f9fU), b(0xc001f7a90695ca90U)}},
      {4, C {b(0x3fc8000000000000U), b(0x3fe4000000000000U)}, C {b(0x3fe319e47c2763fbU), b(0x3ff69650ca2fed2aU)}},
      {4, C {b(0x3fe4000000000000U), b(0x3fc8000000000000U)}, C {b(0x3fcdf658b9cd23bfU), b(0x3fed577ea209bad4U)}},
      {4, C {b(0xbfc8000000000000U), b(0xbfe4000000000000U)}, C {b(0x3fe319e47c2763fbU), b(0xbffbada5de586d07U)}},
      {4, C {b(0xbfe4000000000000U), b(0xbfc8000000000000U)}, C {b(0x3fcdf658b9cd23bfU), b(0xc001cc1babc1be63U)}},
      {4, C {b(0x3fea000000000000U), b(0x3fec000000000000U)}, C {b(0x3fed6f835850b8aaU), b(0x3fef49ab42590792U)}},
      {4, C {b(0x3fec000000000000U), b(0x3fea000000000000U)}, C {b(0x3fecca3e1c011349U), b(0x3fed3e3484605abeU)}},
      {4, C {b(0xbfea000000000000U), b(0xbfec000000000000U)}, C {b(0x3fed6f835850b8aaU), b(0xc0014f9083adeb34U)}},
      {4, C {b(0xbfec000000000000U), b(0xbfea000000000000U)}, C {b(0x3fecca3e1c011349U), b(0xc001d26e332c1669U)}},
      {4, C {b(0x4000000000000000U), b(0x4008000000000000U)}, C {b(0x3fffbbf409ccd560U), b(0x3ff0009683e3b051U)}},
      {4, C {b(0x4008000000000000U), b(0x4000000000000000U)}, C {b(0x3fff7f8a7b4f255fU), b(0x3fe3657b2580b4e9U)}},
      {4, C {b(0xc000000000000000U), b(0xc008000000000000U)}, C {b(0x3fffbbf409ccd560U), b(0xc00121b0125254f0U)}},
      {4, C {b(0xc008000000000000U), b(0xc000000000000000U)}, C {b(0x3fff7f8a7b4f255fU), b(0xc004489c8ae3ffdeU)}},
      {4, C {b(0x3fe6000000000000U), b(0x0000000000000000U)}, C {b(0x0000000000000000U), b(0x3fea0217f3770e7dU)}},
      {4, C {b(0xbfe6000000000000U), b(0x0000000000000000U)}, C {b(0x0000000000000000U), b(0x4002a17557666979U)}},
      {4, C {b(0x0000000000000000U), b(0x3fe6000000000000U)}, C {b(0x3fe48e82454a7953U), b(0x3ff921fb54442d18U)}},
      {4, C {b(0x0000000000000000U), b(0x3ff8000000000000U)}, C {b(0x3ff31dc0090b63d8U), b(0x3ff921fb54442d18U)}},
      // log1p
      {5, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x4046dfb516f20bbfU), b(0x3bc0000000000000U)}},
      {5, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x4046dfb516f20bbfU), b(0x3ff921fb54442d18U)}},
      {5, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0x4046dfb516f20bbfU), b(0xc00921fb54442d18U)}},
      {5, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0x4046dfb516f20bbfU), b(0xbff921fb54442d18U)}},
      {5, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x4046dfb516f20bbfU), b(0x3ff921fb54442d18U)}},
      {5, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x4046dfb516f20bbfU), b(0x3bc0000000000000U)}},
      {5, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0x4046dfb516f20bbfU), b(0xbff921fb54442d18U)}},
      {5, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0x4046dfb516f20bbfU), b(0xc00921fb54442d18U)}},
      {5, C {b(0x3fe8000000000000U), b(0x3c30000000000000U)}, C {b(0x3fe1e85f5e7040d0U), b(0x3c22492492492492U)}},
      {5, C {b(0x3c30000000000000U), b(0x3fe8000000000000U)}, C {b(0x3fcc8ff7c79a9a22U), b(0x3fe4978fa3269ee1U)}},
      {5, C {b(0xbfe8000000000000U), b(0xbc30000000000000U)}, C {b(0xbff62e42fefa39efU), b(0xbc50000000000000U)}},
      {5, C {b(0xbc30000000000000U), b(0xbfe8000000000000U)}, C {b(0x3fcc8ff7c79a9a22U), b(0xbfe4978fa3269ee1U)}},
      {5, C {b(0x3c30000000000000U), b(0x4000000000000000U)}, C {b(0x3fe9c041f7ed8d33U), b(0x3ff1b6e192ebbe44U)}},
      {5, C {b(0x4000000000000000U), b(0x3c30000000000000U)}, C {b(0x3ff193ea7aad030bU), b(0x3c15555555555555U)}},
      {5, C {b(0xbc30000000000000U), b(0xc000000000000000U)}, C {b(0x3fe9c041f7ed8d33U), b(0xbff1b6e192ebbe44U)}},
      {5, C {b(0xc000000000000000U), b(0xbc30000000000000U)}, C {b(0x3860000000000000U), b(0xc00921fb54442d18U)}},
      {5, C {b(0x37a0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3fde1c891a51c97aU), b(0x3fecac7c57846f9eU)}},
      {5, C {b(0x3ff4000000000000U), b(0x37a0000000000000U)}, C {b(0x3fe9f323ecbf984cU), b(0x378c71c71c71c71cU)}},
      {5, C {b(0xb7a0000000000000U), b(0xbff4000000000000U)}, C {b(0x3fde1c891a51c97aU), b(0xbfecac7c57846f9eU)}},
      {5, C {b(0xbff4000000000000U), b(0xb7a0000000000000U)}, C {b(0xbff62e42fefa39efU), b(0xc00921fb54442d18U)}},
      {5, C {b(0x3fc0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3fe0a2423c2b8eddU), b(0x3fead0be00825672U)}},
      {5, C {b(0x3ff4000000000000U), b(0x3fc0000000000000U)}, C {b(0x3fe9ffc3494f853bU), b(0x3fac6a4cd203b4bbU)}},
      {5, C {b(0xbfc0000000000000U), b(0xbff4000000000000U)}, C {b(0x3fdb0ac20701ddccU), b(0xbfeeb8e57b0c8583U)}},
      {5, C {b(0xbff4000000000000U), b(0xbfc0000000000000U)}, C {b(0xbff465438280904dU), b(0xc0056c6e7397f5aeU)}},
      {5, C {b(0x3bd0000000000000U), b(0x3ff0000000000000U)}, C {b(0x3fd62e42fefa39efU), b(0x3fe921fb54442d18U)}},
      {5, C {b(0x3ff0000000000000U), b(0x3bd0000000000000U)}, C {b(0x3fe62e42fefa39efU), b(0x3bc0000000000000U)}},
      {5, C {b(0xbbd0000000000000U), b(0xbff0000000000000U)}, C {b(0x3fd62e42fefa39efU), b(0xbfe921fb54442d18U)}},
      {5, C {b(0xbff0000000000000U), b(0xbbd0000000000000U)}, C {b(0xc046dfb516f20bbfU), b(0xbff921fb54442d18U)}},
      {5, C {b(0x3fd4000000000000U), b(0x3ff0000000000000U)}, C {b(0x3fe0069613296fd2U), b(0x3fe4d59ed9394f67U)}},
      {5, C {b(0x3ff0000000000000U), b(0x3fd4000000000000U)}, C {b(0x3fe6910f7dace0afU), b(0x3fc3d6eee8c6626cU)}},
      {5, C {b(0xbfd4000000000000U), b(0xbff0000000000000U)}, C {b(0x3fc8c5b7c858b48bU), b(0xbfeefe068bba2275U)}},
      {5, C {b(0xbff0000000000000U), b(0xbfd4000000000000U)}, C {b(0xbff29c440606e6abU), b(0xbff921fb54442d18U)}},
      {5, C {b(0x3fc8000000000000U), b(0x3c70000000000000U)}, C {b(0x3fc5ff3070a793d4U), b(0x3c6af286bca1af28U)}},
      {5, C {b(0x3c70000000000000U), b(0x3fc8000000000000U)}, C {b(0x3f91b0d98923d984U), b(0x3fc7b97b4bce5b02U)}},
      {5, C {b(0xbfc8000000000000U), b(0xbc70000000000000U)}, C {b(0xbfca93ed3c8ad9e3U), b(0xbc73b13b13b13b14U)}},
      {5, C {b(0xbc70000000000000U), b(0xbfc8000000000000U)}, C {b(0x3f91b0d98923d97cU), b(0xbfc7b97b4bce5b02U)}},
      {5, C {b(0x37a0000000000000U), b(0x3fe4000000000000U)}, C {b(0x3fc51aad872df82dU), b(0x3fe1e00babdefeb4U)}},
      {5, C {b(0x3fe4000000000000U), b(0x37a0000000000000U)}, C {b(0x3fdf128f5faf06edU), b(0x3793b13b13b13b14U)}},
      {5, C {b(0xb7a0000000000000U), b(0xbfe4000000000000U)}, C {b(0x3fc51aad872df82dU), b(0xbfe1e00babdefeb4U)}},
      {5, C {b(0xbfe4000000000000U), b(0xb7a0000000000000U)}, C {b(0xbfef62f40794a7b8U), b(0xb7b5555555555555U)}},
      {5, C {b(0x3fc8000000000000U), b(0x3fe4000000000000U)}, C {b(0x3fd2d2b4012edc9eU), b(0x3fdf01afb73642bdU)}},
      {5, C {b(0x3fe4000000000000U), b(0x3fc8000000000000U)}, C {b(0x3fdf7ee7ceb54fd4U), b(0x3fbd688d9f5750e7U)}},
      {5, C {b(0xbfc8000000000000U), b(0xbfe4000000000000U)}, C {b(0x3f995c830ec8e3ebU), b(0xbfe4fb7564dacdedU)}},
      {5, C {b(0xbfe4000000000000U), b(0xbfc8000000000000U)}, C {b(0xbfebd0f50ea15474U), b(0xbfddac670561bb4fU)}},
      {5, C {b(0x3fea000000000000U), b(0x3fec000000000000U)}, C {b(0x3fe661ef3455c76aU), b(0x3fdcc8dc8b4c94a6U)}},
      {5, C {b(0x3fec000000000000U), b(0x3fea000000000000U)}, C {b(0x3fe6de6b36ef00c4U), b(0x3fda2b8bbe1b0ba9U)}},
      {5, C {b(0xbfea000000000000U), b(0xbfec000000000000U)}, C {b(0xbfbc6ffbc6f00f71U), b(0xbff5c157ed1aa6a3U)}},
      {5, C {b(0xbfec000000000000U), b(0xbfea000000000000U)}, C {b(0xbfc914a8635bf68aU), b(0xbff6b0bae830c070U)}},
      {5, C {b(0x4000000000000000U), b(0x4008000000000000U)}, C {b(0x3ff71f7b3a6b9186U), b(0x3fe921fb54442d18U)}},
      {5, C {b(0x4008000000000000U), b(0x4000000000000000U)}, C {b(0x3ff7f7427b73e391U), b(0x3fddac670561bb4fU)}},
      {5, C {b(0xc000000000000000U), b(0xc008000000000000U)}, C {b(0x3ff26bb1bbb55516U), b(0xbffe47df3d0dd4d1U)}},
      {5, C {b(0xc008000000000000U), b(0xc000000000000000U)}, C {b(0x3ff0a2b23f3bab73U), b(0xc002d97c7f3321d2U)}},
      {5, C {b(0x3fe6000000000000U), b(0x0000000000000000U)}, C {b(0x3fe0be72e4252a83U), b(0x0000000000000000U)}},
      {5, C {b(0xbfe6000000000000U), b(0x0000000000000000U)}, C {b(0xbff29c440606e6abU), b(0x0000000000000000U)}},
      {5, C {b(0x0000000000000000U), b(0x3fe6000000000000U)}, C {b(0x3fc8c5b7c858b48bU), b(0x3fe345f01cce37bbU)}},
      {5, C {b(0x3ff8000000000000U), b(0x0000000000000000U)}, C {b(0x3fed5240f0e0e078U), b(0x0000000000000000U)}},
      {5, C {b(0xbff8000000000000U), b(0x0000000000000000U)}, C {b(0xbfe62e42fefa39efU), b(0x400921fb54442d18U)}},
      {5, C {b(0x0000000000000000U), b(0x3ff8000000000000U)}, C {b(0x3fe2dbc55768deb3U), b(0x3fef730bd281f69bU)}},
      // log2
      {6, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x4050800000000000U), b(0x3bc71547652b82feU)}},
      {6, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x4050800000000000U), b(0x40022123045b5decU)}},
      {6, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0x4050800000000000U), b(0xc0122123045b5decU)}},
      {6, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0x4050800000000000U), b(0xc0022123045b5decU)}},
      {6, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x4050800000000000U), b(0x40022123045b5decU)}},
      {6, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x4050800000000000U), b(0x3bc71547652b82feU)}},
      {6, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0x4050800000000000U), b(0xc0022123045b5decU)}},
      {6, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0x4050800000000000U), b(0xc0122123045b5decU)}},
      {6, C {b(0x3fe8000000000000U), b(0x3c30000000000000U)}, C {b(0xbfda8ff971810a5eU), b(0x3c3ec709dc3a03fdU)}},
      {6, C {b(0x3c30000000000000U), b(0x3fe8000000000000U)}, C {b(0xbfda8ff971810a5eU), b(0x40022123045b5decU)}},
      {6, C {b(0xbfe8000000000000U), b(0xbc30000000000000U)}, C {b(0xbfda8ff971810a5eU), b(0xc0122123045b5decU)}},
      {6, C {b(0xbc30000000000000U), b(0xbfe8000000000000U)}, C {b(0xbfda8ff971810a5eU), b(0xc0022123045b5decU)}},
      {6, C {b(0x3c30000000000000U), b(0x4000000000000000U)}, C {b(0x3ff0000000000000U), b(0x40022123045b5decU)}},
      {6, C {b(0x4000000000000000U), b(0x3c30000000000000U)}, C {b(0x3ff0000000000000U), b(0x3c271547652b82feU)}},
      {6, C {b(0xbc30000000000000U), b(0xc000000000000000U)}, C {b(0x3ff0000000000000U), b(0xc0022123045b5decU)}},
      {6, C {b(0xc000000000000000U), b(0xbc30000000000000U)}, C {b(0x3ff0000000000000U), b(0xc0122123045b5decU)}},
      {6, C {b(0x37a0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3fd49a784bcd1b8bU), b(0x40022123045b5decU)}},
      {6, C {b(0x3ff4000000000000U), b(0x37a0000000000000U)}, C {b(0x3fd49a784bcd1b8bU), b(0x37a2776c50ef9bfeU)}},
      {6, C {b(0xb7a0000000000000U), b(0xbff4000000000000U)}, C {b(0x3fd49a784bcd1b8bU), b(0xc0022123045b5decU)}},
      {6, C {b(0xbff4000000000000U), b(0xb7a0000000000000U)}, C {b(0x3fd49a784bcd1b8bU), b(0xc0122123045b5decU)}},
      {6, C {b(0x3fc0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3fd510118708a8f9U), b(0x4000faa6dfdb3701U)}},
      {6, C {b(0x3ff4000000000000U), b(0x3fc0000000000000U)}, C {b(0x3fd510118708a8f9U), b(0x3fc267c248026eacU)}},
      {6, C {b(0xbfc0000000000000U), b(0xbff4000000000000U)}, C {b(0x3fd510118708a8f9U), b(0xc003479f28db84d6U)}},
      {6, C {b(0xbff4000000000000U), b(0xbfc0000000000000U)}, C {b(0x3fd510118708a8f9U), b(0xc0118de4f21b4a76U)}},
      {6, C {b(0x3bd0000000000000U), b(0x3ff0000000000000U)}, C {b(0x37a71547652b82feU), b(0x40022123045b5decU)}},
      {6, C {b(0x3ff0000000000000U), b(0x3bd0000000000000U)}, C {b(0x37a71547652b82feU), b(0x3bd71547652b82feU)}},
      {6, C {b(0xbbd0000000000000U), b(0xbff0000000000000U)}, C {b(0x37a71547652b82feU), b(0xc0022123045b5decU)}},
      {6, C {b(0xbff0000000000000U), b(0xbbd0000000000000U)}, C {b(0x37a71547652b82feU), b(0xc0122123045b5decU)}},
      {6, C {b(0x3fd4000000000000U), b(0x3ff0000000000000U)}, C {b(0x3fb134e1b489062eU), b(0x3ffd447141f1017cU)}},
      {6, C {b(0x3ff0000000000000U), b(0x3fd4000000000000U)}, C {b(0x3fb134e1b489062eU), b(0x3fdbf7531b16e96dU)}},
      {6, C {b(0xbfd4000000000000U), b(0xbff0000000000000U)}, C {b(0x3fb134e1b489062eU), b(0xc005a00d67be3b19U)}},
      {6, C {b(0xbff0000000000000U), b(0xbfd4000000000000U)}, C {b(0x3fb134e1b489062eU), b(0xc01061add2a9ef55U)}},
      {6, C {b(0x3fc8000000000000U), b(0x3c70000000000000U)}, C {b(0xc00351ff2e30214cU), b(0x3c9ec709dc3a03fdU)}},
      {6, C {b(0x3c70000000000000U), b(0x3fc8000000000000U)}, C {b(0xc00351ff2e30214cU), b(0x40022123045b5debU)}},
      {6, C {b(0xbfc8000000000000U), b(0xbc70000000000000U)}, C {b(0xc00351ff2e30214cU), b(0xc0122123045b5debU)}},
      {6, C {b(0xbc70000000000000U), b(0xbfc8000000000000U)}, C {b(0xc00351ff2e30214cU), b(0xc0022123045b5decU)}},
      {6, C {b(0x37a0000000000000U), b(0x3fe4000000000000U)}, C {b(0xbfe5b2c3da19723bU), b(0x40022123045b5decU)}},
      {6, C {b(0x3fe4000000000000U), b(0x37a0000000000000U)}, C {b(0xbfe5b2c3da19723bU), b(0x37b2776c50ef9bfeU)}},
      {6, C {b(0xb7a0000000000000U), b(0xbfe4000000000000U)}, C {b(0xbfe5b2c3da19723bU), b(0xc0022123045b5decU)}},
      {6, C {b(0xbfe4000000000000U), b(0xb7a0000000000000U)}, C {b(0xbfe5b2c3da19723bU), b(0xc0122123045b5decU)}},
      {6, C {b(0x3fc8000000000000U), b(0x3fe4000000000000U)}, C {b(0xbfe3b5845a7c883aU), b(0x3ffd87f95dd8d2e4U)}},
      {6, C {b(0x3fe4000000000000U), b(0x3fc8000000000000U)}, C {b(0xbfe3b5845a7c883aU), b(0x3fdae932ab77a3cfU)}},
      {6, C {b(0xbfc8000000000000U), b(0xbfe4000000000000U)}, C {b(0xbfe3b5845a7c883aU), b(0xc0057e4959ca5265U)}},
      {6, C {b(0xbfe4000000000000U), b(0xbfc8000000000000U)}, C {b(0xbfe3b5845a7c883aU), b(0xc010728fd9a3e3afU)}},
      {6, C {b(0x3fea000000000000U), b(0x3fec000000000000U)}, C {b(0x3fd0604719f24eb2U), b(0x3ff2fbe6232f4d06U)}},
      {6, C {b(0x3fec000000000000U), b(0x3fea000000000000U)}, C {b(0x3fd0604719f24eb2U), b(0x3ff1465fe5876ed2U)}},
      {6, C {b(0xbfea000000000000U), b(0xbfec000000000000U)}, C {b(0x3fd0604719f24eb2U), b(0xc00ac452f71f1554U)}},
      {6, C {b(0xbfec000000000000U), b(0xbfea000000000000U)}, C {b(0x3fd0604719f24eb2U), b(0xc00b9f1615f3046eU)}},
      {6, C {b(0x4000000000000000U), b(0x4008000000000000U)}, C {b(0x3ffd9a802391e233U), b(0x3ff6af9a29bfaa3fU)}},
      {6, C {b(0x4008000000000000U), b(0x4000000000000000U)}, C {b(0x3ffd9a802391e233U), b(0x3feb2557bdee2330U)}},
      {6, C {b(0xc000000000000000U), b(0xc008000000000000U)}, C {b(0x3ffd9a802391e233U), b(0xc008ea78f3d6e6b8U)}},
      {6, C {b(0xc008000000000000U), b(0xc000000000000000U)}, C {b(0x3ffd9a802391e233U), b(0xc00d78f0193b330bU)}},
      {6, C {b(0x3fe6000000000000U), b(0x0000000000000000U)}, C {b(0xbfe14c560fe68af9U), b(0x0000000000000000U)}},
      {6, C {b(0xbfe6000000000000U), b(0x0000000000000000U)}, C {b(0xbfe14c560fe68af9U), b(0x40122123045b5decU)}},
      {6, C {b(0x0000000000000000U), b(0x3fe6000000000000U)}, C {b(0xbfe14c560fe68af9U), b(0x40022123045b5decU)}},
      {6, C {b(0x3ff8000000000000U), b(0x0000000000000000U)}, C {b(0x3fe2b803473f7ad1U), b(0x0000000000000000U)}},
      {6, C {b(0xbff8000000000000U), b(0x0000000000000000U)}, C {b(0x3fe2b803473f7ad1U), b(0x40122123045b5decU)}},
      {6, C {b(0x0000000000000000U), b(0x3ff8000000000000U)}, C {b(0x3fe2b803473f7ad1U), b(0x40022123045b5decU)}},
      // log10
      {7, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x4033de33eb2475cfU), b(0x3babcb7b1526e50eU)}},
      {7, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x4033de33eb2475cfU), b(0x3fe5d47c4cb2fba1U)}},
      {7, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0x4033de33eb2475cfU), b(0xbff5d47c4cb2fba1U)}},
      {7, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0x4033de33eb2475cfU), b(0xbfe5d47c4cb2fba1U)}},
      {7, C {b(0x3fe0000000000000U), b(0x4410000000000000U)}, C {b(0x4033de33eb2475cfU), b(0x3fe5d47c4cb2fba1U)}},
      {7, C {b(0x4410000000000000U), b(0x3fe0000000000000U)}, C {b(0x4033de33eb2475cfU), b(0x3babcb7b1526e50eU)}},
      {7, C {b(0xbfe0000000000000U), b(0xc410000000000000U)}, C {b(0x4033de33eb2475cfU), b(0xbfe5d47c4cb2fba1U)}},
      {7, C {b(0xc410000000000000U), b(0xbfe0000000000000U)}, C {b(0x4033de33eb2475cfU), b(0xbff5d47c4cb2fba1U)}},
      {7, C {b(0x3fe8000000000000U), b(0x3c30000000000000U)}, C {b(0xbfbffbfc2bbc7803U), b(0x3c2287a7636f435fU)}},
      {7, C {b(0x3c30000000000000U), b(0x3fe8000000000000U)}, C {b(0xbfbffbfc2bbc7803U), b(0x3fe5d47c4cb2fba1U)}},
      {7, C {b(0xbfe8000000000000U), b(0xbc30000000000000U)}, C {b(0xbfbffbfc2bbc7803U), b(0xbff5d47c4cb2fba1U)}},
      {7, C {b(0xbc30000000000000U), b(0xbfe8000000000000U)}, C {b(0xbfbffbfc2bbc7803U), b(0xbfe5d47c4cb2fba1U)}},
      {7, C {b(0x3c30000000000000U), b(0x4000000000000000U)}, C {b(0x3fd34413509f79ffU), b(0x3fe5d47c4cb2fba1U)}},
      {7, C {b(0x4000000000000000U), b(0x3c30000000000000U)}, C {b(0x3fd34413509f79ffU), b(0x3c0bcb7b1526e50eU)}},
      {7, C {b(0xbc30000000000000U), b(0xc000000000000000U)}, C {b(0x3fd34413509f79ffU), b(0xbfe5d47c4cb2fba1U)}},
      {7, C {b(0xc000000000000000U), b(0xbc30000000000000U)}, C {b(0x3fd34413509f79ffU), b(0xbff5d47c4cb2fba1U)}},
      {7, C {b(0x37a0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3fb8cf183886480dU), b(0x3fe5d47c4cb2fba1U)}},
      {7, C {b(0x3ff4000000000000U), b(0x37a0000000000000U)}, C {b(0x3fb8cf183886480dU), b(0x37863c62775250d8U)}},
      {7, C {b(0xb7a0000000000000U), b(0xbff4000000000000U)}, C {b(0x3fb8cf183886480dU), b(0xbfe5d47c4cb2fba1U)}},
      {7, C {b(0xbff4000000000000U), b(0xb7a0000000000000U)}, C {b(0x3fb8cf183886480dU), b(0xbff5d47c4cb2fba1U)}},
      {7, C {b(0x3fc0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3fb95cb2880f45baU), b(0x3fe471e3ee405d7fU)}},
      {7, C {b(0x3ff4000000000000U), b(0x3fc0000000000000U)}, C {b(0x3fb95cb2880f45baU), b(0x3fa62985e729e21eU)}},
      {7, C {b(0xbfc0000000000000U), b(0xbff4000000000000U)}, C {b(0x3fb95cb2880f45baU), b(0xbfe73714ab2599c3U)}},
      {7, C {b(0xbff4000000000000U), b(0xbfc0000000000000U)}, C {b(0x3fb95cb2880f45baU), b(0xbff523301d79ac90U)}},
      {7, C {b(0x3bd0000000000000U), b(0x3ff0000000000000U)}, C {b(0x378bcb7b1526e50eU), b(0x3fe5d47c4cb2fba1U)}},
      {7, C {b(0x3ff0000000000000U), b(0x3bd0000000000000U)}, C {b(0x378bcb7b1526e50eU), b(0x3bbbcb7b1526e50eU)}},
      {7, C {b(0xbbd0000000000000U), b(0xbff0000000000000U)}, C {b(0x378bcb7b1526e50eU), b(0xbfe5d47c4cb2fba1U)}},
      {7, C {b(0xbff0000000000000U), b(0xbbd0000000000000U)}, C {b(0x378bcb7b1526e50eU), b(0xbff5d47c4cb2fba1U)}},
      {7, C {b(0x3fd4000000000000U), b(0x3ff0000000000000U)}, C {b(0x3f94b8018b21ed4fU), b(0x3fe19ee65a336ef7U)}},
      {7, C {b(0x3ff0000000000000U), b(0x3fd4000000000000U)}, C {b(0x3f94b8018b21ed4fU), b(0x3fc0d657c9fe32a5U)}},
      {7, C {b(0xbfd4000000000000U), b(0xbff0000000000000U)}, C {b(0x3f94b8018b21ed4fU), b(0xbfea0a123f32884aU)}},
      {7, C {b(0xbff0000000000000U), b(0xbfd4000000000000U)}, C {b(0x3f94b8018b21ed4fU), b(0xbff3b9b15373354cU)}},
      {7, C {b(0x3fc8000000000000U), b(0x3c70000000000000U)}, C {b(0xbfe74392d61708ffU), b(0x3c8287a7636f435fU)}},
      {7, C {b(0x3c70000000000000U), b(0x3fc8000000000000U)}, C {b(0xbfe74392d61708ffU), b(0x3fe5d47c4cb2fba0U)}},
      {7, C {b(0xbfc8000000000000U), b(0xbc70000000000000U)}, C {b(0xbfe74392d61708ffU), b(0xbff5d47c4cb2fba1U)}},
      {7, C {b(0xbc70000000000000U), b(0xbfc8000000000000U)}, C {b(0xbfe74392d61708ffU), b(0xbfe5d47c4cb2fba1U)}},
      {7, C {b(0x37a0000000000000U), b(0x3fe4000000000000U)}, C {b(0xbfca209a84fbcff8U), b(0x3fe5d47c4cb2fba1U)}},
      {7, C {b(0x3fe4000000000000U), b(0x37a0000000000000U)}, C {b(0xbfca209a84fbcff8U), b(0x37963c62775250d8U)}},
      {7, C {b(0xb7a0000000000000U), b(0xbfe4000000000000U)}, C {b(0xbfca209a84fbcff8U), b(0xbfe5d47c4cb2fba1U)}},
      {7, C {b(0xbfe4000000000000U), b(0xb7a0000000000000U)}, C {b(0xbfca209a84fbcff8U), b(0xbff5d47c4cb2fba1U)}},
      {7, C {b(0x3fc8000000000000U), b(0x3fe4000000000000U)}, C {b(0xbfc7bb6868d1a65bU), b(0x3fe1c78ed4c3554fU)}},
      {7, C {b(0x3fe4000000000000U), b(0x3fc8000000000000U)}, C {b(0xbfc7bb6868d1a65bU), b(0x3fc033b5dfbe9947U)}},
      {7, C {b(0xbfc8000000000000U), b(0xbfe4000000000000U)}, C {b(0xbfc7bb6868d1a65bU), b(0xbfe9e169c4a2a1f2U)}},
      {7, C {b(0xbfe4000000000000U), b(0xbfc8000000000000U)}, C {b(0xbfc7bb6868d1a65bU), b(0xbff3ce0590bb2878U)}},
      {7, C {b(0x3fea000000000000U), b(0x3fec000000000000U)}, C {b(0x3fb3b80161d70f33U), b(0x3fd6dbe6c6a7b4f5U)}},
      {7, C {b(0x3fec000000000000U), b(0x3fea000000000000U)}, C {b(0x3fb3b80161d70f33U), b(0x3fd4cd11d2be424dU)}},
      {7, C {b(0xbfea000000000000U), b(0xbfec000000000000U)}, C {b(0x3fb3b80161d70f33U), b(0xbff01d829b090e64U)}},
      {7, C {b(0xbfec000000000000U), b(0xbfea000000000000U)}, C {b(0x3fb3b80161d70f33U), b(0xbff0a137d8036b0dU)}},
      {7, C {b(0x4000000000000000U), b(0x4008000000000000U)}, C {b(0x3fe1d2b643bc124fU), b(0x3fdb510cc39c3b28U)}},
      {7, C {b(0x4008000000000000U), b(0x4000000000000000U)}, C {b(0x3fe1d2b643bc124fU), b(0x3fd057ebd5c9bc19U)}},
      {7, C {b(0xc000000000000000U), b(0xc008000000000000U)}, C {b(0x3fe1d2b643bc124fU), b(0xbfee00723797d9adU)}},
      {7, C {b(0xc008000000000000U), b(0xc000000000000000U)}, C {b(0x3fe1d2b643bc124fU), b(0xbff1be8157408c9aU)}},
      {7, C {b(0x3fe6000000000000U), b(0x0000000000000000U)}, C {b(0xbfc4d43f8275a483U), b(0x0000000000000000U)}},
      {7, C {b(0xbfe6000000000000U), b(0x0000000000000000U)}, C {b(0xbfc4d43f8275a483U), b(0x3ff5d47c4cb2fba1U)}},
      {7, C {b(0x0000000000000000U), b(0x3fe6000000000000U)}, C {b(0xbfc4d43f8275a483U), b(0x3fe5d47c4cb2fba1U)}},
      {7, C {b(0x3ff8000000000000U), b(0x0000000000000000U)}, C {b(0x3fc68a288b60b7fcU), b(0x0000000000000000U)}},
      {7, C {b(0xbff8000000000000U), b(0x0000000000000000U)}, C {b(0x3fc68a288b60b7fcU), b(0x3ff5d47c4cb2fba1U)}},
      {7, C {b(0x0000000000000000U), b(0x3ff8000000000000U)}, C {b(0x3fc68a288b60b7fcU), b(0x3fe5d47c4cb2fba1U)}},
      // exp2
      {8, C {b(0x3fe8000000000000U), b(0x3c30000000000000U)}, C {b(0x3ffae89f995ad3adU), b(0x3c32a6d4750ced4aU)}},
      {8, C {b(0x3c30000000000000U), b(0x3fe8000000000000U)}, C {b(0x3febc5be318ed250U), b(0x3fdfcae625887c12U)}},
      {8, C {b(0xbfe8000000000000U), b(0xbc30000000000000U)}, C {b(0x3fe306fe0a31b715U), b(0xbc1a60a1147227c1U)}},
      {8, C {b(0xbc30000000000000U), b(0xbfe8000000000000U)}, C {b(0x3febc5be318ed250U), b(0xbfdfcae625887c12U)}},
      {8, C {b(0x3c30000000000000U), b(0x4000000000000000U)}, C {b(0x3fc77b84a55f6f1fU), b(0x3fef74f6978410f4U)}},
      {8, C {b(0x4000000000000000U), b(0x3c30000000000000U)}, C {b(0x4010000000000000U), b(0x3c462e42fefa39efU)}},
      {8, C {b(0xbc30000000000000U), b(0xc000000000000000U)}, C {b(0x3fc77b84a55f6f1fU), b(0xbfef74f6978410f4U)}},
      {8, C {b(0xc000000000000000U), b(0xbc30000000000000U)}, C {b(0x3fd0000000000000U), b(0xbc062e42fefa39efU)}},
      {8, C {b(0x37a0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3fe4b8b6b3f41dd5U), b(0x3fe8628172003ed1U)}},
      {8, C {b(0x3ff4000000000000U), b(0x37a0000000000000U)}, C {b(0x400306fe0a31b715U), b(0x37aa60a1147227c1U)}},
      {8, C {b(0xb7a0000000000000U), b(0xbff4000000000000U)}, C {b(0x3fe4b8b6b3f41dd5U), b(0xbfe8628172003ed1U)}},
      {8, C {b(0xbff4000000000000U), b(0xb7a0000000000000U)}, C {b(0x3fdae89f995ad3adU), b(0xb782a6d4750ced4aU)}},
      {8, C {b(0x3fc0000000000000U), b(0x3ff4000000000000U)}, C {b(0x3fe698d4cf819a3aU), b(0x3fea97802c6d86adU)}},
      {8, C {b(0x3ff4000000000000U), b(0x3fc0000000000000U)}, C {b(0x4002f4b866bb891fU), b(0x3fca582f02c7c97fU)}},
      {8, C {b(0xbfc0000000000000U), b(0xbff4000000000000U)}, C {b(0x3fe30071a325fccaU), b(0xbfe65c67258fcb11U)}},
      {8, C {b(0xbff4000000000000U), b(0xbfc0000000000000U)}, C {b(0x3fdacec86b50fb40U), b(0xbfa2a0dba4c1343fU)}},
      {8, C {b(0x3bd0000000000000U), b(0x3ff0000000000000U)}, C {b(0x3fe89d9ae6856a55U), b(0x3fe4725eeb25adecU)}},
      {8, C {b(0x3ff0000000000000U), b(0x3bd0000000000000U)}, C {b(0x4000000000000000U), b(0x3bd62e42fefa39efU)}},
      {8, C {b(0xbbd0000000000000U), b(0xbff0000000000000U)}, C {b(0x3fe89d9ae6856a55U), b(0xbfe4725eeb25adecU)}},
      {8, C {b(0xbff0000000000000U), b(0xbbd0000000000000U)}, C {b(0x3fe0000000000000U), b(0xbbb62e42fefa39efU)}},
      {8, C {b(0x3fd4000000000000U), b(0x3ff0000000000000U)}, C {b(0x3fee91b28f2ca25bU), b(0x3fe96458200ab724U)}},
      {8, C {b(0x3ff0000000000000U), b(0x3fd4000000000000U)}, C {b(0x3fff4091ac613baeU), b(0x3fdb8273fd682508U)}},
      {8, C {b(0xbfd4000000000000U), b(0xbff0000000000000U)}, C {b(0x3fe3d256478071edU), b(0xbfe076f3a6cd3211U)}},
      {8, C {b(0xbff0000000000000U), b(0xbfd4000000000000U)}, C {b(0x3fdf4091ac613baeU), b(0xbfbb8273fd682508U)}},
      {8, C {b(0x3fc8000000000000U), b(0x3c70000000000000U)}, C {b(0x3ff2387a6e756238U), b(0x3c69425703358c38U)}},
      {8, C {b(0x3c70000000000000U), b(0x3fc8000000000000U)}, C {b(0x3fefbae97f0a7559U), b(0x3fc096b7aeccd5a4U)}},
      {8, C {b(0xbfc8000000000000U), b(0xbc70000000000000U)}, C {b(0x3fec199bdd85529cU), b(0xbc637a3acd129175U)}},
      {8, C {b(0xbc70000000000000U), b(0xbfc8000000000000U)}, C {b(0x3fefbae97f0a7559U), b(0xbfc096b7aeccd5a4U)}},
      {8, C {b(0x37a0000000000000U), b(0x3fe4000000000000U)}, C {b(0x3fed0b390e8a1cc4U), b(0x3fdadde280c60e99U)}},
      {8, C {b(0x3fe4000000000000U), b(0x37a0000000000000U)}, C {b(0x3ff8ace5422aa0dbU), b(0x37a11a89d042937fU)}},
      {8, C {b(0xb7a0000000000000U), b(0xbfe4000000000000U)}, C {b(0x3fed0b390e8a1cc4U), b(0xbfdadde280c60e99U)}},
      {8, C {b(0xbfe4000000000000U), b(0xb7a0000000000000U)}, C {b(0x3fe4bfdad5362a27U), b(0xb78cc3cb5c605804U)}},
      {8, C {b(0x3fc8000000000000U), b(0x3fe4000000000000U)}, C {b(0x3ff08992db11c79fU), b(0x3fde9874fefb2c31U)}},
      {8, C {b(0x3fe4000000000000U), b(0x3fc8000000000000U)}, C {b(0x3ff8779f19856bc5U), b(0x3fc995556e89385dU)}},
      {8, C {b(0xbfc8000000000000U), b(0xbfe4000000000000U)}, C {b(0x3fe9811028b7172bU), b(0xbfd797a6665f5b1bU)}},
      {8, C {b(0xbfe4000000000000U), b(0xbfc8000000000000U)}, C {b(0x3fe4930e8dd558a7U), b(0xbfb5834fadcbfb92U)}},
      {8, C {b(0x3fea000000000000U), b(0x3fec000000000000U)}, C {b(0x3ff7169959a89197U), b(0x3ff0045844898989U)}},
      {8, C {b(0x3fec000000000000U), b(0x3fea000000000000U)}, C {b(0x3ff8cfeed5006949U), b(0x3fef551c2455bf52U)}},
      {8, C {b(0xbfea000000000000U), b(0xbfec000000000000U)}, C {b(0x3fddf1193e456a73U), b(0xbfd4c57d4400cb83U)}},
      {8, C {b(0xbfec000000000000U), b(0xbfea000000000000U)}, C {b(0x3fdd81c38b3ee090U), b(0xbfd2a16168646077U)}},
      {8, C {b(0x4000000000000000U), b(0x4008000000000000U)}, C {b(0xbfff2aeaa29fd873U), b(0x400bf2ef36cc43d1U)}},
      {8, C {b(0x4008000000000000U), b(0x4000000000000000U)}, C {b(0x3ff77b84a55f6f1fU), b(0x401f74f6978410f4U)}},
      {8, C {b(0xc000000000000000U), b(0xc008000000000000U)}, C {b(0xbfbf2aeaa29fd873U), b(0xbfcbf2ef36cc43d1U)}},
      {8, C {b(0xc008000000000000U), b(0xc000000000000000U)}, C {b(0x3f977b84a55f6f1fU), b(0xbfbf74f6978410f4U)}},
      {8, C {b(0x3fe6000000000000U), b(0x0000000000000000U)}, C {b(0x3ff9c49182a3f090U), b(0x0000000000000000U)}},
      {8, C {b(0xbfe6000000000000U), b(0x0000000000000000U)}, C {b(0x3fe3dea64c123422U), b(0x0000000000000000U)}},
      {8, C {b(0x0000000000000000U), b(0x3fe6000000000000U)}, C {b(0x3fec6f50091ad94eU), b(0x3fdd5b71737d93f6U)}},
      {8, C {b(0x3ff8000000000000U), b(0x0000000000000000U)}, C {b(0x4006a09e667f3bcdU), b(0x0000000000000000U)}},
      {8, C {b(0xbff8000000000000U), b(0x0000000000000000U)}, C {b(0x3fd6a09e667f3bcdU), b(0x0000000000000000U)}},
      {8, C {b(0x0000000000000000U), b(0x3ff8000000000000U)}, C {b(0x3fe034edcb9b3c64U), b(0x3feb97a83e754d80U)}},
      // expm1
      {9, C {b(0x3fe8000000000000U), b(0x3c30000000000000U)}, C {b(0x3ff1df3b68cfb9efU), b(0x3c40ef9db467dcf8U)}},
      {9, C {b(0x3c30000000000000U), b(0x3fe8000000000000U)}, C {b(0xbfd12c027355bdc2U), b(0x3fe5cffc16bf8f0dU)}},
      {9, C {b(0xbfe8000000000000U), b(0xbc30000000000000U)}, C {b(0xbfe0e25f8a081941U), b(0xbc1e3b40ebefcd7eU)}},
      {9, C {b(0xbc30000000000000U), b(0xbfe8000000000000U)}, C {b(0xbfd12c027355bdc2U), b(0xbfe5cffc16bf8f0dU)}},
      {9, C {b(0x3c30000000000000U), b(0x4000000000000000U)}, C {b(0xbff6a88995d4dc81U), b(0x3fed18f6ead1b446U)}},
      {9, C {b(0x4000000000000000U), b(0x3c30000000000000U)}, C {b(0x40198e64b8d4ddaeU), b(0x3c5d8e64b8d4ddaeU)}},
      {9, C {b(0xbc30000000000000U), b(0xc000000000000000U)}, C {b(0xbff6a88995d4dc81U), b(0xbfed18f6ead1b446U)}},
      {9, C {b(0xc000000000000000U), b(0xbc30000000000000U)}, C {b(0xbfebab5557101f8dU), b(0xbc0152aaa3bf81ccU)}},
      {9, C {b(0x37a0000000000000U), b(0x3ff4000000000000U)}, C {b(0xbfe5e8e113ba1357U), b(0x3fee5e14fe11418cU)}},
      {9, C {b(0x3ff4000000000000U), b(0x37a0000000000000U)}, C {b(0x4003ec38edb0faf0U), b(0x37bbec38edb0faf0U)}},
      {9, C {b(0xb7a0000000000000U), b(0xbff4000000000000U)}, C {b(0xbfe5e8e113ba1357U), b(0xbfee5e14fe11418cU)}},
      {9, C {b(0xbff4000000000000U), b(0xb7a0000000000000U)}, C {b(0xbfe6d4f3e46ad3d8U), b(0xb7825618372a584fU)}},
      {9, C {b(0x3fc0000000000000U), b(0x3ff4000000000000U)}, C {b(0xbfe490f0cfb96488U), b(0x3ff1349830f72e7cU)}},
      {9, C {b(0x3ff4000000000000U), b(0x3fc0000000000000U)}, C {b(0x4003b47316d5a9b6U), b(0x3fdbd99f2b25c228U)}},
      {9, C {b(0xbfc0000000000000U), b(0xbff4000000000000U)}, C {b(0xbfe71867661515eaU), b(0xbfeacc9a71042fc0U)}},
      {9, C {b(0xbff4000000000000U), b(0xbfc0000000000000U)}, C {b(0xbfe6e743e0bfd29aU), b(0xbfa249e14355bc48U)}},
      {9, C {b(0x3bd0000000000000U), b(0x3ff0000000000000U)}, C {b(0xbfdd6bafe095f2e9U), b(0x3feaed548f090ceeU)}},
      {9, C {b(0x3ff0000000000000U), b(0x3bd0000000000000U)}, C {b(0x3ffb7e151628aed3U), b(0x3be5bf0a8b145769U)}},
      {9, C {b(0xbbd0000000000000U), b(0xbff0000000000000U)}, C {b(0xbfdd6bafe095f2e9U), b(0xbfeaed548f090ceeU)}},
      {9, C {b(0xbff0000000000000U), b(0xbbd0000000000000U)}, C {b(0xbfe43a54e4e98864U), b(0xbbb78b56362cef38U)}},
      {9, C {b(0x3fd4000000000000U), b(0x3ff0000000000000U)}, C {b(0xbfd0bc529feb8f4dU), b(0x3ff267085f343da0U)}},
      {9, C {b(0x3ff0000000000000U), b(0x3fd4000000000000U)}, C {b(0x3ff962d5fe6f5837U), b(0x3feabe176bde9958U)}},
      {9, C {b(0xbfd4000000000000U), b(0xbff0000000000000U)}, C {b(0xbfe359c13660c4edU), b(0xbfe3b344a8004421U)}},
      {9, C {b(0xbff0000000000000U), b(0xbfd4000000000000U)}, C {b(0xbfe4cc4a2dd07227U), b(0xbfbcf424500ed6a0U)}},
      {9, C {b(0x3fc8000000000000U), b(0x3c70000000000000U)}, C {b(0x3fca65c0b85ac1a9U), b(0x3c734cb8170b5835U)}},
      {9, C {b(0x3c70000000000000U), b(0x3fc8000000000000U)}, C {b(0xbf91f2840c263c87U), b(0x3fc7dc102fbaf2b5U)}},
      {9, C {b(0xbfc8000000000000U), b(0xbc70000000000000U)}, C {b(0xbfc5e25fb4fde211U), b(0xbc6a876812c0877cU)}},
      {9, C {b(0xbc70000000000000U), b(0xbfc8000000000000U)}, C {b(0xbf91f2840c263c8fU), b(0xbfc7dc102fbaf2b5U)}},
      {9, C {b(0x37a0000000000000U), b(0x3fe4000000000000U)}, C {b(0xbfc8325c49bb41edU), b(0x3fe2b91dea88421eU)}},
      {9, C {b(0x3fe4000000000000U), b(0x37a0000000000000U)}, C {b(0x3febc8abbf01c781U), b(0x37ade455df80e3c0U)}},
      {9, C {b(0xb7a0000000000000U), b(0xbfe4000000000000U)}, C {b(0xbfc8325c49bb41edU), b(0xbfe2b91dea88421eU)}},
      {9, C {b(0xbfe4000000000000U), b(0xb7a0000000000000U)}, C {b(0xbfddbe46d96cd831U), b(0xb79120dc934993e8U)}},
      {9, C {b(0x3fc8000000000000U), b(0x3fe4000000000000U)}, C {b(0xbf965093dc85958dU), b(0x3fe6959a406b405aU)}},
      {9, C {b(0x3fe4000000000000U), b(0x3fc8000000000000U)}, C {b(0x3feabc6e42eba2e7U), b(0x3fd649aea5c6299cU)}},
      {9, C {b(0xbfc8000000000000U), b(0xbfe4000000000000U)}, C {b(0xbfd4f8d6ec0b76e3U), b(0xbfdf0b4452bc4b2dU)}},
      {9, C {b(0xbfe4000000000000U), b(0xbfc8000000000000U)}, C {b(0xbfde57fb1f7351fcU), b(0xbfb98ad240983560U)}},
      {9, C {b(0x3fea000000000000U), b(0x3fec000000000000U)}, C {b(0x3fdc72d4b20aa5d3U), b(0x3ffbaccb35948d8dU)}},
      {9, C {b(0x3fec000000000000U), b(0x3fea000000000000U)}, C {b(0x3fe4ca1cbc42c8f6U), b(0x3ffbdd9c64a94fceU)}},
      {9, C {b(0xbfea000000000000U), b(0xbfec000000000000U)}, C {b(0xbfe6e5dcc2b6acddU), b(0xbfd5cc507fe9237aU)}},
      {9, C {b(0xbfec000000000000U), b(0xbfea000000000000U)}, C {b(0xbfe6d39973c0b473U), b(0xbfd35e8af4b0741dU)}},
      {9, C {b(0x4000000000000000U), b(0x4008000000000000U)}, C {b(0xc020a1561c407ab9U), b(0x3ff0af13f8da5098U)}},
      {9, C {b(0x4008000000000000U), b(0x4000000000000000U)}, C {b(0xc022b7919774a197U), b(0x403243839d86bc10U)}},
      {9, C {b(0xc000000000000000U), b(0xc008000000000000U)}, C {b(0xbff224c92bfe9196U), b(0xbf938e8fbaf98c75U)}},
      {9, C {b(0xc008000000000000U), b(0xc000000000000000U)}, C {b(0xbff054dd2a011babU), b(0xbfa72dcb2f823af7U)}},
      {9, C {b(0x3fe6000000000000U), b(0x0000000000000000U)}, C {b(0x3fefa3bcc305f191U), b(0x0000000000000000U)}},
      {9, C {b(0xbfe6000000000000U), b(0x0000000000000000U)}, C {b(0xbfdfd19b804dffbfU), b(0x0000000000000000U)}},
      {9, C {b(0x0000000000000000U), b(0x3fe6000000000000U)}, C {b(0xbfcd13be9688dbfeU), b(0x3fe44eb381cf386bU)}},
      {9, C {b(0x3ff8000000000000U), b(0x0000000000000000U)}, C {b(0x400bda7fcc9f8a83U), b(0x0000000000000000U)}},
      {9, C {b(0xbff8000000000000U), b(0x0000000000000000U)}, C {b(0xbfe8dc1e236d28f9U), b(0x0000000000000000U)}},
      {9, C {b(0x0000000000000000U), b(0x3ff8000000000000U)}, C {b(0xbfedbc85560740cbU), b(0x3fefeb7a9b2c6d8bU)}},
      {3, C {b(0x3fe8000000000000U), b(0x3ff4000000000000U)}, C {b(0x3ff081eb4b421591U), b(0x3fed2cf5c7c70f0cU)}}, // a branch of its own
      {3, C {b(0x3fe8000000000000U), b(0x3ff0000000000000U)}, C {b(0x3fed1b64054a4e02U), b(0x3fe880250fe32e4dU)}}, // a branch of its own
      {2, C {b(0x4410000000000000U), b(0x4410000000000000U)}, C {b(0x3ff921fb54442d18U), b(0x3bc0000000000000U)}}, // a branch of its own
      {2, C {b(0x3fe4000000000000U), b(0x3fe2000000000000U)}, C {b(0x3fe5730098602231U), b(0x3fd950cb870779f1U)}}, // a branch of its own
      {2, C {b(0x37a0000000000000U), b(0x3ff0000000000000U)}, C {b(0x3fe921fb54442d18U), b(0x4047386e22edf4a6U)}}, // a branch of its own
    };
    const std::vector<value> special {
      // asin
      {0, C {zero, zero}, C {zero, zero}},
      {0, C {zero, minus_zero}, C {zero, minus_zero}},
      {0, C {zero, inf}, C {zero, inf}},
      {0, C {zero, -inf}, C {zero, -inf}},
      {0, C {zero, nan}, C {zero, nan}},
      {0, C {minus_zero, zero}, C {minus_zero, zero}},
      {0, C {minus_zero, minus_zero}, C {minus_zero, minus_zero}},
      {0, C {minus_zero, inf}, C {minus_zero, inf}},
      {0, C {minus_zero, -inf}, C {minus_zero, -inf}},
      {0, C {minus_zero, nan}, C {minus_zero, nan}},
      {0, C {inf, zero}, C {b(0x3ff921fb54442d18U), inf}},
      {0, C {inf, minus_zero}, C {b(0x3ff921fb54442d18U), -inf}},
      {0, C {inf, inf}, C {b(0x3fe921fb54442d18U), inf}},
      {0, C {inf, -inf}, C {b(0x3fe921fb54442d18U), -inf}},
      {0, C {inf, nan}, C {nan, inf}},
      {0, C {inf, b(0x3ff0000000000000U)}, C {b(0x3ff921fb54442d18U), inf}},
      {0, C {inf, b(0xbff0000000000000U)}, C {b(0x3ff921fb54442d18U), -inf}},
      {0, C {inf, b(0x3fe0000000000000U)}, C {b(0x3ff921fb54442d18U), inf}},
      {0, C {-inf, zero}, C {b(0xbff921fb54442d18U), inf}},
      {0, C {-inf, minus_zero}, C {b(0xbff921fb54442d18U), -inf}},
      {0, C {-inf, inf}, C {b(0xbfe921fb54442d18U), inf}},
      {0, C {-inf, -inf}, C {b(0xbfe921fb54442d18U), -inf}},
      {0, C {-inf, nan}, C {nan, inf}},
      {0, C {-inf, b(0x3ff0000000000000U)}, C {b(0xbff921fb54442d18U), inf}},
      {0, C {-inf, b(0xbff0000000000000U)}, C {b(0xbff921fb54442d18U), -inf}},
      {0, C {-inf, b(0x3fe0000000000000U)}, C {b(0xbff921fb54442d18U), inf}},
      {0, C {nan, zero}, C {nan, nan}},
      {0, C {nan, minus_zero}, C {nan, nan}},
      {0, C {nan, inf}, C {nan, inf}},
      {0, C {nan, -inf}, C {nan, -inf}},
      {0, C {nan, nan}, C {nan, nan}},
      {0, C {nan, b(0x3ff0000000000000U)}, C {nan, nan}},
      {0, C {nan, b(0xbff0000000000000U)}, C {nan, nan}},
      {0, C {nan, b(0x3fe0000000000000U)}, C {nan, nan}},
      {0, C {b(0x3ff0000000000000U), inf}, C {zero, inf}},
      {0, C {b(0x3ff0000000000000U), -inf}, C {zero, -inf}},
      {0, C {b(0x3ff0000000000000U), nan}, C {nan, nan}},
      {0, C {b(0xbff0000000000000U), inf}, C {minus_zero, inf}},
      {0, C {b(0xbff0000000000000U), -inf}, C {minus_zero, -inf}},
      {0, C {b(0xbff0000000000000U), nan}, C {nan, nan}},
      {0, C {b(0x3fe0000000000000U), inf}, C {zero, inf}},
      {0, C {b(0x3fe0000000000000U), -inf}, C {zero, -inf}},
      {0, C {b(0x3fe0000000000000U), nan}, C {nan, nan}},
      // acos
      {1, C {zero, zero}, C {b(0x3ff921fb54442d18U), minus_zero}},
      {1, C {zero, minus_zero}, C {b(0x3ff921fb54442d18U), zero}},
      {1, C {zero, inf}, C {b(0x3ff921fb54442d18U), -inf}},
      {1, C {zero, -inf}, C {b(0x3ff921fb54442d18U), inf}},
      {1, C {zero, nan}, C {b(0x3ff921fb54442d18U), nan}},
      {1, C {minus_zero, zero}, C {b(0x3ff921fb54442d18U), minus_zero}},
      {1, C {minus_zero, minus_zero}, C {b(0x3ff921fb54442d18U), zero}},
      {1, C {minus_zero, inf}, C {b(0x3ff921fb54442d18U), -inf}},
      {1, C {minus_zero, -inf}, C {b(0x3ff921fb54442d18U), inf}},
      {1, C {minus_zero, nan}, C {b(0x3ff921fb54442d18U), nan}},
      {1, C {inf, zero}, C {zero, -inf}},
      {1, C {inf, minus_zero}, C {zero, inf}},
      {1, C {inf, inf}, C {b(0x3fe921fb54442d18U), -inf}},
      {1, C {inf, -inf}, C {b(0x3fe921fb54442d18U), inf}},
      {1, C {inf, nan}, C {nan, -inf}},
      {1, C {inf, b(0x3ff0000000000000U)}, C {zero, -inf}},
      {1, C {inf, b(0xbff0000000000000U)}, C {zero, inf}},
      {1, C {inf, b(0x3fe0000000000000U)}, C {zero, -inf}},
      {1, C {-inf, zero}, C {b(0x400921fb54442d18U), -inf}},
      {1, C {-inf, minus_zero}, C {b(0x400921fb54442d18U), inf}},
      {1, C {-inf, inf}, C {b(0x4002d97c7f3321d2U), -inf}},
      {1, C {-inf, -inf}, C {b(0x4002d97c7f3321d2U), inf}},
      {1, C {-inf, nan}, C {nan, -inf}},
      {1, C {-inf, b(0x3ff0000000000000U)}, C {b(0x400921fb54442d18U), -inf}},
      {1, C {-inf, b(0xbff0000000000000U)}, C {b(0x400921fb54442d18U), inf}},
      {1, C {-inf, b(0x3fe0000000000000U)}, C {b(0x400921fb54442d18U), -inf}},
      {1, C {nan, zero}, C {nan, nan}},
      {1, C {nan, minus_zero}, C {nan, nan}},
      {1, C {nan, inf}, C {nan, -inf}},
      {1, C {nan, -inf}, C {nan, inf}},
      {1, C {nan, nan}, C {nan, nan}},
      {1, C {nan, b(0x3ff0000000000000U)}, C {nan, nan}},
      {1, C {nan, b(0xbff0000000000000U)}, C {nan, nan}},
      {1, C {nan, b(0x3fe0000000000000U)}, C {nan, nan}},
      {1, C {b(0x3ff0000000000000U), inf}, C {b(0x3ff921fb54442d18U), -inf}},
      {1, C {b(0x3ff0000000000000U), -inf}, C {b(0x3ff921fb54442d18U), inf}},
      {1, C {b(0x3ff0000000000000U), nan}, C {nan, nan}},
      {1, C {b(0xbff0000000000000U), inf}, C {b(0x3ff921fb54442d18U), -inf}},
      {1, C {b(0xbff0000000000000U), -inf}, C {b(0x3ff921fb54442d18U), inf}},
      {1, C {b(0xbff0000000000000U), nan}, C {nan, nan}},
      {1, C {b(0x3fe0000000000000U), inf}, C {b(0x3ff921fb54442d18U), -inf}},
      {1, C {b(0x3fe0000000000000U), -inf}, C {b(0x3ff921fb54442d18U), inf}},
      {1, C {b(0x3fe0000000000000U), nan}, C {nan, nan}},
      // atan
      {2, C {zero, zero}, C {zero, zero}},
      {2, C {zero, minus_zero}, C {zero, minus_zero}},
      {2, C {zero, inf}, C {b(0x3ff921fb54442d18U), zero}},
      {2, C {zero, -inf}, C {b(0x3ff921fb54442d18U), minus_zero}},
      {2, C {zero, nan}, C {nan, nan}},
      {2, C {minus_zero, zero}, C {minus_zero, zero}},
      {2, C {minus_zero, minus_zero}, C {minus_zero, minus_zero}},
      {2, C {minus_zero, inf}, C {b(0xbff921fb54442d18U), zero}},
      {2, C {minus_zero, -inf}, C {b(0xbff921fb54442d18U), minus_zero}},
      {2, C {minus_zero, nan}, C {nan, nan}},
      {2, C {inf, zero}, C {b(0x3ff921fb54442d18U), zero}},
      {2, C {inf, minus_zero}, C {b(0x3ff921fb54442d18U), minus_zero}},
      {2, C {inf, inf}, C {b(0x3ff921fb54442d18U), zero}},
      {2, C {inf, -inf}, C {b(0x3ff921fb54442d18U), minus_zero}},
      {2, C {inf, nan}, C {b(0x3ff921fb54442d18U), zero}},
      {2, C {inf, b(0x3ff0000000000000U)}, C {b(0x3ff921fb54442d18U), zero}},
      {2, C {inf, b(0xbff0000000000000U)}, C {b(0x3ff921fb54442d18U), minus_zero}},
      {2, C {inf, b(0x3fe0000000000000U)}, C {b(0x3ff921fb54442d18U), zero}},
      {2, C {-inf, zero}, C {b(0xbff921fb54442d18U), zero}},
      {2, C {-inf, minus_zero}, C {b(0xbff921fb54442d18U), minus_zero}},
      {2, C {-inf, inf}, C {b(0xbff921fb54442d18U), zero}},
      {2, C {-inf, -inf}, C {b(0xbff921fb54442d18U), minus_zero}},
      {2, C {-inf, nan}, C {b(0xbff921fb54442d18U), zero}},
      {2, C {-inf, b(0x3ff0000000000000U)}, C {b(0xbff921fb54442d18U), zero}},
      {2, C {-inf, b(0xbff0000000000000U)}, C {b(0xbff921fb54442d18U), minus_zero}},
      {2, C {-inf, b(0x3fe0000000000000U)}, C {b(0xbff921fb54442d18U), zero}},
      {2, C {nan, zero}, C {nan, zero}},
      {2, C {nan, minus_zero}, C {nan, minus_zero}},
      {2, C {nan, inf}, C {nan, zero}},
      {2, C {nan, -inf}, C {nan, minus_zero}},
      {2, C {nan, nan}, C {nan, nan}},
      {2, C {nan, b(0x3ff0000000000000U)}, C {nan, nan}},
      {2, C {nan, b(0xbff0000000000000U)}, C {nan, nan}},
      {2, C {nan, b(0x3fe0000000000000U)}, C {nan, nan}},
      {2, C {b(0x3ff0000000000000U), inf}, C {b(0x3ff921fb54442d18U), zero}},
      {2, C {b(0x3ff0000000000000U), -inf}, C {b(0x3ff921fb54442d18U), minus_zero}},
      {2, C {b(0x3ff0000000000000U), nan}, C {nan, nan}},
      {2, C {b(0xbff0000000000000U), inf}, C {b(0xbff921fb54442d18U), zero}},
      {2, C {b(0xbff0000000000000U), -inf}, C {b(0xbff921fb54442d18U), minus_zero}},
      {2, C {b(0xbff0000000000000U), nan}, C {nan, nan}},
      {2, C {b(0x3fe0000000000000U), inf}, C {b(0x3ff921fb54442d18U), zero}},
      {2, C {b(0x3fe0000000000000U), -inf}, C {b(0x3ff921fb54442d18U), minus_zero}},
      {2, C {b(0x3fe0000000000000U), nan}, C {nan, nan}},
      // asinh
      {3, C {zero, zero}, C {zero, zero}},
      {3, C {zero, minus_zero}, C {zero, minus_zero}},
      {3, C {zero, inf}, C {inf, b(0x3ff921fb54442d18U)}},
      {3, C {zero, -inf}, C {inf, b(0xbff921fb54442d18U)}},
      {3, C {zero, nan}, C {nan, nan}},
      {3, C {minus_zero, zero}, C {minus_zero, zero}},
      {3, C {minus_zero, minus_zero}, C {minus_zero, minus_zero}},
      {3, C {minus_zero, inf}, C {-inf, b(0x3ff921fb54442d18U)}},
      {3, C {minus_zero, -inf}, C {-inf, b(0xbff921fb54442d18U)}},
      {3, C {minus_zero, nan}, C {nan, nan}},
      {3, C {inf, zero}, C {inf, zero}},
      {3, C {inf, minus_zero}, C {inf, minus_zero}},
      {3, C {inf, inf}, C {inf, b(0x3fe921fb54442d18U)}},
      {3, C {inf, -inf}, C {inf, b(0xbfe921fb54442d18U)}},
      {3, C {inf, nan}, C {inf, nan}},
      {3, C {inf, b(0x3ff0000000000000U)}, C {inf, zero}},
      {3, C {inf, b(0xbff0000000000000U)}, C {inf, minus_zero}},
      {3, C {inf, b(0x3fe0000000000000U)}, C {inf, zero}},
      {3, C {-inf, zero}, C {-inf, zero}},
      {3, C {-inf, minus_zero}, C {-inf, minus_zero}},
      {3, C {-inf, inf}, C {-inf, b(0x3fe921fb54442d18U)}},
      {3, C {-inf, -inf}, C {-inf, b(0xbfe921fb54442d18U)}},
      {3, C {-inf, nan}, C {-inf, nan}},
      {3, C {-inf, b(0x3ff0000000000000U)}, C {-inf, zero}},
      {3, C {-inf, b(0xbff0000000000000U)}, C {-inf, minus_zero}},
      {3, C {-inf, b(0x3fe0000000000000U)}, C {-inf, zero}},
      {3, C {nan, zero}, C {nan, zero}},
      {3, C {nan, minus_zero}, C {nan, minus_zero}},
      {3, C {nan, inf}, C {inf, nan}},
      {3, C {nan, -inf}, C {inf, nan}},
      {3, C {nan, nan}, C {nan, nan}},
      {3, C {nan, b(0x3ff0000000000000U)}, C {nan, nan}},
      {3, C {nan, b(0xbff0000000000000U)}, C {nan, nan}},
      {3, C {nan, b(0x3fe0000000000000U)}, C {nan, nan}},
      {3, C {b(0x3ff0000000000000U), inf}, C {inf, b(0x3ff921fb54442d18U)}},
      {3, C {b(0x3ff0000000000000U), -inf}, C {inf, b(0xbff921fb54442d18U)}},
      {3, C {b(0x3ff0000000000000U), nan}, C {nan, nan}},
      {3, C {b(0xbff0000000000000U), inf}, C {-inf, b(0x3ff921fb54442d18U)}},
      {3, C {b(0xbff0000000000000U), -inf}, C {-inf, b(0xbff921fb54442d18U)}},
      {3, C {b(0xbff0000000000000U), nan}, C {nan, nan}},
      {3, C {b(0x3fe0000000000000U), inf}, C {inf, b(0x3ff921fb54442d18U)}},
      {3, C {b(0x3fe0000000000000U), -inf}, C {inf, b(0xbff921fb54442d18U)}},
      {3, C {b(0x3fe0000000000000U), nan}, C {nan, nan}},
      // acosh
      {4, C {zero, zero}, C {zero, b(0x3ff921fb54442d18U)}},
      {4, C {zero, minus_zero}, C {zero, b(0xbff921fb54442d18U)}},
      {4, C {zero, inf}, C {inf, b(0x3ff921fb54442d18U)}},
      {4, C {zero, -inf}, C {inf, b(0xbff921fb54442d18U)}},
      {4, C {zero, nan}, C {nan, b(0x3ff921fb54442d18U)}},
      {4, C {minus_zero, zero}, C {zero, b(0x3ff921fb54442d18U)}},
      {4, C {minus_zero, minus_zero}, C {zero, b(0xbff921fb54442d18U)}},
      {4, C {minus_zero, inf}, C {inf, b(0x3ff921fb54442d18U)}},
      {4, C {minus_zero, -inf}, C {inf, b(0xbff921fb54442d18U)}},
      {4, C {minus_zero, nan}, C {nan, b(0x3ff921fb54442d18U)}},
      {4, C {inf, zero}, C {inf, zero}},
      {4, C {inf, minus_zero}, C {inf, minus_zero}},
      {4, C {inf, inf}, C {inf, b(0x3fe921fb54442d18U)}},
      {4, C {inf, -inf}, C {inf, b(0xbfe921fb54442d18U)}},
      {4, C {inf, nan}, C {inf, nan}},
      {4, C {inf, b(0x3ff0000000000000U)}, C {inf, zero}},
      {4, C {inf, b(0xbff0000000000000U)}, C {inf, minus_zero}},
      {4, C {inf, b(0x3fe0000000000000U)}, C {inf, zero}},
      {4, C {-inf, zero}, C {inf, b(0x400921fb54442d18U)}},
      {4, C {-inf, minus_zero}, C {inf, b(0xc00921fb54442d18U)}},
      {4, C {-inf, inf}, C {inf, b(0x4002d97c7f3321d2U)}},
      {4, C {-inf, -inf}, C {inf, b(0xc002d97c7f3321d2U)}},
      {4, C {-inf, nan}, C {inf, nan}},
      {4, C {-inf, b(0x3ff0000000000000U)}, C {inf, b(0x400921fb54442d18U)}},
      {4, C {-inf, b(0xbff0000000000000U)}, C {inf, b(0xc00921fb54442d18U)}},
      {4, C {-inf, b(0x3fe0000000000000U)}, C {inf, b(0x400921fb54442d18U)}},
      {4, C {nan, zero}, C {nan, nan}},
      {4, C {nan, minus_zero}, C {nan, nan}},
      {4, C {nan, inf}, C {inf, nan}},
      {4, C {nan, -inf}, C {inf, nan}},
      {4, C {nan, nan}, C {nan, nan}},
      {4, C {nan, b(0x3ff0000000000000U)}, C {nan, nan}},
      {4, C {nan, b(0xbff0000000000000U)}, C {nan, nan}},
      {4, C {nan, b(0x3fe0000000000000U)}, C {nan, nan}},
      {4, C {b(0x3ff0000000000000U), inf}, C {inf, b(0x3ff921fb54442d18U)}},
      {4, C {b(0x3ff0000000000000U), -inf}, C {inf, b(0xbff921fb54442d18U)}},
      {4, C {b(0x3ff0000000000000U), nan}, C {nan, nan}},
      {4, C {b(0xbff0000000000000U), inf}, C {inf, b(0x3ff921fb54442d18U)}},
      {4, C {b(0xbff0000000000000U), -inf}, C {inf, b(0xbff921fb54442d18U)}},
      {4, C {b(0xbff0000000000000U), nan}, C {nan, nan}},
      {4, C {b(0x3fe0000000000000U), inf}, C {inf, b(0x3ff921fb54442d18U)}},
      {4, C {b(0x3fe0000000000000U), -inf}, C {inf, b(0xbff921fb54442d18U)}},
      {4, C {b(0x3fe0000000000000U), nan}, C {nan, nan}},
      // atanh
      {5, C {zero, zero}, C {zero, zero}},
      {5, C {zero, minus_zero}, C {zero, minus_zero}},
      {5, C {zero, inf}, C {zero, b(0x3ff921fb54442d18U)}},
      {5, C {zero, -inf}, C {zero, b(0xbff921fb54442d18U)}},
      {5, C {zero, nan}, C {zero, nan}},
      {5, C {minus_zero, zero}, C {minus_zero, zero}},
      {5, C {minus_zero, minus_zero}, C {minus_zero, minus_zero}},
      {5, C {minus_zero, inf}, C {minus_zero, b(0x3ff921fb54442d18U)}},
      {5, C {minus_zero, -inf}, C {minus_zero, b(0xbff921fb54442d18U)}},
      {5, C {minus_zero, nan}, C {minus_zero, nan}},
      {5, C {inf, zero}, C {zero, b(0x3ff921fb54442d18U)}},
      {5, C {inf, minus_zero}, C {zero, b(0xbff921fb54442d18U)}},
      {5, C {inf, inf}, C {zero, b(0x3ff921fb54442d18U)}},
      {5, C {inf, -inf}, C {zero, b(0xbff921fb54442d18U)}},
      {5, C {inf, nan}, C {zero, nan}},
      {5, C {inf, b(0x3ff0000000000000U)}, C {zero, b(0x3ff921fb54442d18U)}},
      {5, C {inf, b(0xbff0000000000000U)}, C {zero, b(0xbff921fb54442d18U)}},
      {5, C {inf, b(0x3fe0000000000000U)}, C {zero, b(0x3ff921fb54442d18U)}},
      {5, C {-inf, zero}, C {minus_zero, b(0x3ff921fb54442d18U)}},
      {5, C {-inf, minus_zero}, C {minus_zero, b(0xbff921fb54442d18U)}},
      {5, C {-inf, inf}, C {minus_zero, b(0x3ff921fb54442d18U)}},
      {5, C {-inf, -inf}, C {minus_zero, b(0xbff921fb54442d18U)}},
      {5, C {-inf, nan}, C {minus_zero, nan}},
      {5, C {-inf, b(0x3ff0000000000000U)}, C {minus_zero, b(0x3ff921fb54442d18U)}},
      {5, C {-inf, b(0xbff0000000000000U)}, C {minus_zero, b(0xbff921fb54442d18U)}},
      {5, C {-inf, b(0x3fe0000000000000U)}, C {minus_zero, b(0x3ff921fb54442d18U)}},
      {5, C {nan, zero}, C {nan, nan}},
      {5, C {nan, minus_zero}, C {nan, nan}},
      {5, C {nan, inf}, C {zero, b(0x3ff921fb54442d18U)}},
      {5, C {nan, -inf}, C {zero, b(0xbff921fb54442d18U)}},
      {5, C {nan, nan}, C {nan, nan}},
      {5, C {nan, b(0x3ff0000000000000U)}, C {nan, nan}},
      {5, C {nan, b(0xbff0000000000000U)}, C {nan, nan}},
      {5, C {nan, b(0x3fe0000000000000U)}, C {nan, nan}},
      {5, C {b(0x3ff0000000000000U), inf}, C {zero, b(0x3ff921fb54442d18U)}},
      {5, C {b(0x3ff0000000000000U), -inf}, C {zero, b(0xbff921fb54442d18U)}},
      {5, C {b(0x3ff0000000000000U), nan}, C {nan, nan}},
      {5, C {b(0xbff0000000000000U), inf}, C {minus_zero, b(0x3ff921fb54442d18U)}},
      {5, C {b(0xbff0000000000000U), -inf}, C {minus_zero, b(0xbff921fb54442d18U)}},
      {5, C {b(0xbff0000000000000U), nan}, C {nan, nan}},
      {5, C {b(0x3fe0000000000000U), inf}, C {zero, b(0x3ff921fb54442d18U)}},
      {5, C {b(0x3fe0000000000000U), -inf}, C {zero, b(0xbff921fb54442d18U)}},
      {5, C {b(0x3fe0000000000000U), nan}, C {nan, nan}},
    };
    bool ok {true};
    for (const value& each : finite) {
      ok = ok && near(finite_functions.at(each.function)(each.z), each.want);
    }
    for (const value& each : special) {
      const C got {special_functions.at(each.function)(each.z)};
      ok = ok && (std::is_same_v<R, double> ? same(got, each.want) : near(got, each.want, 1));
    }
    // log1p of a zero is that zero, its signs kept; of m (1 + i) past hypot's overflow, ln|1 + z| finite (to 300 bits)
    for (const C& z : {C {minus_zero, zero}, C {minus_zero, minus_zero}, C {zero, minus_zero}, C {zero, zero}}) {
      ok = ok && same(d::complex_log1p(z), z);
    }
    if constexpr (std::is_same_v<R, double>) {
      const R m {b(0x7feab36d48e1acf0U)}; // 1.5e308
      ok = ok && near(d::complex_log1p(C {m, m}), C {b(0x40862f9602b3a198U), std::numbers::pi_v<R> / R {4}});
    }
    else {
      const R m {std::bit_cast<R>(0x7f61b1e6U)}; // 3e38
      ok = ok && near(d::complex_log1p(C {m, m}), C {std::bit_cast<R>(0x42b1e308U), std::numbers::pi_v<R> / R {4}});
    }
    return ok;
  }

  // sign as NumPy 2's: z / |z|, 0 for 0, NaN for a NaN modulus, ±1 along the infinite part (NaN where both are);
  // expm1 past exp's overflow
  template <typename R>
  bool signs_hold()
  {
    using C = std::complex<R>;
    using numerical::detail::complex_sign;
    const R inf {std::numeric_limits<R>::infinity()};
    const R nan {std::numeric_limits<R>::quiet_NaN()};
    bool    ok  {near(complex_sign(C {R {3}, R {-4}}), C {R {3} / R {5}, R {-4} / R {5}})};
    ok = ok && same(complex_sign(C {R {-0.0}, R {0}}), C {R {0}, R {0}});
    ok = ok && same(complex_sign(C {-inf, R {2}}), C {R {-1}, R {0}});
    ok = ok && same(complex_sign(C {inf, nan}), C {R {1}, R {0}});
    ok = ok && same(complex_sign(C {R {2}, -inf}), C {R {0}, R {-1}});
    ok = ok && same(complex_sign(C {nan, inf}), C {R {0}, R {1}});
    ok = ok && both_nan(complex_sign(C {inf, -inf}));
    ok = ok && both_nan(complex_sign(C {nan, R {1}}));
    // expm1 past exp's overflow: exp(z) − 1, scaled, where exp(re) times the sine overflowed
    using numerical::detail::complex_expm1;
    ok = ok && same(complex_expm1(C {inf, R {0}}), C {inf, R {0}});
    const R past {numerical::detail::exp_bound<R> + R {2}};
    ok = ok && same(complex_expm1(C {past, R {0}}), C {inf, R {0}});
    ok = ok && both_nan(complex_expm1(C {inf, nan})); // NumPy's formula, an imaginary part not finite
    return ok;
  }
} // namespace

TEST(complex_exponentials_as_glibc_computes_them)
{
  EXPECT(exponentials_hold<double>());
  EXPECT(exponentials_hold<float>());
}

TEST(complex_hyperbolic_and_circular_functions_as_glibc_computes_them)
{
  EXPECT(hyperbolics_hold<double>());
  EXPECT(hyperbolics_hold<float>());
}

TEST(complex_tangents_as_glibc_computes_them)
{
  EXPECT(tangents_hold<double>());
  EXPECT(tangents_hold<float>());
}

TEST(complex_logarithms_as_glibc_computes_them)
{
  EXPECT(logarithms_hold<double>());
  EXPECT(logarithms_hold<float>());
}

TEST(complex_inverse_hyperbolic_tangents_as_glibc_computes_them)
{
  EXPECT(inverse_tangents_hold<double>());
  EXPECT(inverse_tangents_hold<float>());
}

TEST(complex_inverse_functions_as_glibc_computes_them)
{
  EXPECT(inverse_functions_hold<double>());
  EXPECT(inverse_functions_hold<float>());
}

TEST(complex_signs_as_numpy_2_gives_them)
{
  EXPECT(signs_hold<double>());
  EXPECT(signs_hold<float>());
}

TEST(tensors_take_each_element_s_elementary_function)
{
  EXPECT(tensors_take_them<double>());
  EXPECT(tensors_take_them<float>());
}
