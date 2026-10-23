// The elementary functions of a complex number as the GNU C library computes them: Annex G's infinities and NaNs
// exactly, each finite branch (scaled near overflow, among subnormal parts, near the unit circle) against a reference
// that does not overflow. NumPy's to the bit on Linux is the Python suite's.
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>

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

TEST(tensors_take_each_element_s_elementary_function)
{
  EXPECT(tensors_take_them<double>());
  EXPECT(tensors_take_them<float>());
}
