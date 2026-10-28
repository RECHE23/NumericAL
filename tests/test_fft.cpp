// The discrete Fourier transform (fft.hpp): accuracy against the transform's sum in double-double arithmetic, the
// twiddle factors' exact values and symmetries, NumPy's points, axes and scalings, and a line's bits the same alone,
// among others and on any number of threads.

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {
  using numerical::fft_norm;
  using numerical::tensor;
  using numerical::detail::double_double;

  // The transform of x, its sum taken in double-double arithmetic with double-double roots: exact to some 10⁻³⁰.
  // The n roots computed once, each term then a product and sums: n² evaluations of the roots' series took a third
  // of the suite's time.
  std::vector<std::complex<double>> double_double_dft(const std::vector<std::complex<double>>& x)
  {
    using numerical::detail::dd_add;
    using numerical::detail::dd_mul;
    using numerical::detail::dd_negate;
    const std::size_t                                    n {x.size()};
    std::vector<std::complex<double>>                    y(n);
    std::vector<std::pair<double_double, double_double>> roots(n);
    for (std::size_t m {0}; m < n; ++m) {
      roots[m] = numerical::detail::unit_root_parts(m, n);
    }
    for (std::size_t k {0}; k < n; ++k) {
      double_double re {.hi = 0.0, .lo = 0.0};
      double_double im {.hi = 0.0, .lo = 0.0};
      for (std::size_t j {0}; j < n; ++j) {
        const auto&         parts {roots[(j * k) % n]};
        const double_double xr    {.hi = x[j].real(), .lo = 0.0};
        const double_double xi    {.hi = x[j].imag(), .lo = 0.0};
        re = dd_add(re, dd_add(dd_mul(xr, parts.first), dd_negate(dd_mul(xi, parts.second))));
        im = dd_add(im, dd_add(dd_mul(xr, parts.second), dd_mul(xi, parts.first)));
      }
      y[k] = {re.hi, im.hi};
    }
    return y;
  }

  // n drawn complex numbers of parts R.
  template <typename R>
  std::vector<std::complex<R>> drawn(std::size_t      n,
                                     std::mt19937_64& draws)
  {
    std::normal_distribution<double> normal;
    std::vector<std::complex<R>>     x(n);
    for (auto& value : x) {
      value = {static_cast<R>(normal(draws)), static_cast<R>(normal(draws))};
    }
    return x;
  }

  // The relative RMS error of the transform of n drawn numbers against double_double_dft, in units of R's epsilon
  // times log₂ n (1 for n of 1 or 2).
  template <typename R>
  double relative_error(std::size_t      n,
                        std::mt19937_64& draws)
  {
    const std::vector<std::complex<R>> x {drawn<R>(n, draws)};
    std::vector<std::complex<double>>  wide(n);
    std::transform(x.begin(), x.end(), wide.begin(), [](const std::complex<R>& v) { return std::complex<double>(v); });
    const std::vector<std::complex<double>> want  {double_double_dft(wide)};
    const tensor<std::complex<R>>           got   {numerical::fft(tensor<std::complex<R>> {{n}, x})};
    double                                  error {0.0};
    double                                  size  {0.0};
    for (std::size_t k {0}; k < n; ++k) {
      error += std::norm(std::complex<double>(got.data()[k]) - want[k]);
      size  += std::norm(want[k]);
    }
    return std::sqrt(error / size) / static_cast<double>(std::numeric_limits<R>::epsilon()) / std::max(1.0, std::log2(static_cast<double>(n)));
  }

  // The same bits, part by part: complex or real values, x's and y's.
  template <typename X,
            typename Y>
  bool same_bits(const X& x,
                 const Y& y)
  {
    using V = std::ranges::range_value_t<X>;
    using R = decltype(std::real(V {}));
    using U = std::conditional_t<sizeof(R) == 8, std::uint64_t, std::uint32_t>;
    return std::ranges::equal(x, y, [](const V& p, const V& q) {
                                return std::bit_cast<U>(std::real(p)) == std::bit_cast<U>(std::real(q)) && std::bit_cast<U>(std::imag(p)) == std::bit_cast<U>(std::imag(q));
                              });
  }
} // namespace

// Within half of epsilon · log₂ n in RMS of the exact transform (a single addition at n = 2), in both precisions: every power of two to 4096 (both
// parities of the exponent: radix 4 alone, and its last pass of radix 2), lengths of mixed radices (2, 3, 5, 7, 31, a
// pass of radix 2 before others) and lengths with a prime factor past the largest radix (Bluestein's convolution).
TEST(fft_is_within_a_fraction_of_epsilon_log_n_of_the_exact_transform)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
  std::mt19937_64 draws {20261025};
  for (std::size_t n {1}; n <= 4096; n *= 2) {
    EXPECT(relative_error<double>(n, draws) < 0.5);
    EXPECT(relative_error<float>(n, draws) < 0.5);
  }
  for (const std::size_t n : {3UL, 5UL, 6UL, 7UL, 12UL, 31UL, 100UL, 360UL, 1000UL, 2187UL, 67UL, 127UL, 1009UL, 2018UL}) {
    EXPECT(relative_error<double>(n, draws) < 0.5);
    EXPECT(relative_error<float>(n, draws) < 0.5);
  }
}

// The roots of unity are exact where their value is (±1, ±i, √½ on the diagonals, with +0 for a zero part), the
// conjugates of their mirror images to the bit, and within one unit of the library's sine and cosine elsewhere.
TEST(the_twiddle_factors_are_exact_where_their_value_is_and_symmetric)
{
  for (const std::size_t n : {1UL, 2UL, 3UL, 8UL, 12UL, 1000UL, 1024UL}) {
    for (std::size_t k {0}; k < n; ++k) {
      const std::complex<double> w      {numerical::detail::unit_root(k, n)};
      const std::complex<double> mirror {numerical::detail::unit_root(n - k, n)};
      EXPECT(mirror.real() == w.real() && mirror.imag() == -w.imag());
      const double angle                {-2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(n)};
      EXPECT(std::abs(w.real() - std::cos(angle)) < 1e-15 && std::abs(w.imag() - std::sin(angle)) < 1e-15);
    }
  }
  const std::complex<double> quarter {numerical::detail::unit_root(2, 8)};
  EXPECT(quarter.real() == 0.0 && !std::signbit(quarter.real()) && quarter.imag() == -1.0);
  const std::complex<double> half    {numerical::detail::unit_root(4, 8)};
  EXPECT(half.real() == -1.0 && half.imag() == 0.0 && !std::signbit(half.imag()));
  const std::complex<double> eighth  {numerical::detail::unit_root(1, 8)};
  EXPECT(eighth.real() == std::sqrt(0.5) && eighth.imag() == -std::sqrt(0.5));
  EXPECT(numerical::detail::unit_root(0, 5) == std::complex<double>(1.0, 0.0) && !std::signbit(numerical::detail::unit_root(0, 5).imag()));
}

// ifft undoes fft, and the scalings are NumPy's: the forward 1/n, both 1/√n (ortho), or the inverse 1/n (backward); a
// power of two's 1/n exact, so the forward-scaled transform is the backward one's bits times 1/n.
TEST(ifft_undoes_fft_and_the_scalings_are_numpys)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
  std::mt19937_64 draws {7};
  for (const std::size_t n : {1UL, 8UL, 32UL, 12UL}) {
    const tensor<std::complex<double>> x {{n}, drawn<double>(n, draws)};
    for (const fft_norm norm : {fft_norm::backward, fft_norm::ortho, fft_norm::forward}) {
      const tensor<std::complex<double>> back {numerical::ifft(numerical::fft(x, std::nullopt, -1, norm), std::nullopt, -1, norm)};
      for (std::size_t k {0}; k < n; ++k) {
        EXPECT(std::abs(back.data()[k] - x.data()[k]) < 1e-14);
      }
    }
    const tensor<std::complex<double>> plain  {numerical::fft(x)};
    const tensor<std::complex<double>> scaled {numerical::fft(x, std::nullopt, 0, fft_norm::forward)};
    if (std::has_single_bit(n)) {
      numerical::buffer<std::complex<double>> divided(plain.data());
      for (auto& value : divided) {
        value = {value.real() / static_cast<double>(n), value.imag() / static_cast<double>(n)};
      }
      EXPECT(same_bits(scaled.data(), divided));
    }
    const tensor<std::complex<double>> unitary    {numerical::fft(x, std::nullopt, -1, fft_norm::ortho)};
    double                             energy_in  {0.0};
    double                             energy_out {0.0};
    for (std::size_t k {0}; k < n; ++k) {
      energy_in  += std::norm(x.data()[k]);
      energy_out += std::norm(unitary.data()[k]);
    }
    EXPECT(std::abs(energy_in - energy_out) < 1e-13 * energy_in); // Parseval
  }
}

// n points: the axis cut to them or padded with zeros, as NumPy's; a real tensor transformed as its complex copy,
// to the bit; float in single precision.
namespace {
  // The transform of a real tensor against that of its complex copy, within a few units of R's epsilon times the input's
  // norm, its frequencies past n/2 the conjugates of those before to the bit; its inverse the conjugate of its transform
  // over n, to the bit (n a power of two, so 1/n exact).
  template <std::floating_point R>
  bool real_input_agrees(const std::vector<R>& values,
                         std::size_t           points)
  {
    using C = std::complex<R>;
    std::vector<C> copy(values.size());
    double         norm {0.0};
    for (std::size_t j {0}; j < values.size(); ++j) {
      copy[j]  = C {values[j], R(0)};
      norm    += static_cast<double>(values[j]) * static_cast<double>(values[j]);
    }
    const tensor<R> real     {{values.size()}, values};
    const auto      spectrum {numerical::fft(real, points)};
    const auto      complex  {numerical::fft(tensor<C> {{values.size()}, copy}, points)};
    const auto      inverse  {numerical::ifft(real, points)};
    const double    bound    {8.0 * static_cast<double>(std::numeric_limits<R>::epsilon()) * std::sqrt(norm) * std::max(1.0, std::log2(static_cast<double>(points)))};
    bool            good     {spectrum.shape() == complex.shape() && spectrum.data()[0].imag() == R(0)};
    for (std::size_t k {0}; k < points; ++k) {
      good = good && std::abs(std::complex<double>(spectrum.data()[k] - complex.data()[k])) <= bound;
      if (2 * k == points) {
        good = good && spectrum.data()[k].imag() == R(0); // the last frequency of an even n: real
      }
      else if (k != 0) {
        good = good && same_bits(std::vector {spectrum.data()[points - k]}, std::vector {std::conj(spectrum.data()[k])});
      }
      const C over_n {std::conj(spectrum.data()[k]) / static_cast<R>(points)};
      good = good && (std::popcount(points) != 1 || same_bits(std::vector {inverse.data()[k]}, std::vector {over_n}));
    }
    return good;
  }
} // namespace

// A real input's transform, as NumPy takes it: its real transform and the Hermitian symmetry, within rounding of the
// complex copy's; cut, padded, of either parity, both precisions; exact where the sums are.
TEST(fft_takes_numpys_points_and_real_input)
{
  const std::vector<double> values {1.0, -2.0, 0.5, 4.0, 3.0};
  for (const std::size_t points : {5UL, 8UL, 2UL, 1UL, 7UL, 16UL}) {
    EXPECT(real_input_agrees(values, points));
  }
  EXPECT(real_input_agrees(std::vector<float> {1.0F, -2.0F, 0.5F, 4.0F, 3.0F, 9.0F}, 6));
  const tensor<double> real                  {{5}, values};
  EXPECT(numerical::fft(real, 2).data()[1] == std::complex<double>(3.0, 0.0));
  const tensor<float>               single   {{4}, std::vector<float> {1.0F, 2.0F, 3.0F, 4.0F}};
  const tensor<std::complex<float>> spectrum {numerical::fft(single)};
  EXPECT(spectrum.data()[0] == std::complex<float>(10.0F, 0.0F) && spectrum.data()[2] == std::complex<float>(-2.0F, 0.0F));
  EXPECT(numerical::ifft(single).data()[0] == std::complex<float>(2.5F, 0.0F));
  EXPECT(numerical::fft(real, std::nullopt, 0, numerical::fft_norm::ortho).shape() == (std::vector<std::size_t> {5}));
  EXPECT_THROWS((void)numerical::fft(real, 0), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::ifft(real, 0), numerical::dimension_error);
}

// Along any axis of a stack, each line's transform is that line's alone to the bit, on one thread or all of them; an
// empty stack transforms to an empty one of the new shape.
TEST(a_line_has_the_same_bits_alone_among_others_and_on_any_threads)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
  std::mt19937_64                          draws  {11};
  const std::size_t                        rows   {6};
  const std::size_t                        cols   {64};
  const std::vector<std::complex<double>>  values {drawn<double>(rows * cols, draws)};
  const tensor<std::complex<double>>       stack  {{rows, cols}, values};
  numerical::set_max_threads(1);
  const tensor<std::complex<double>> one_thread   {numerical::fft(stack)};
  numerical::set_max_threads(0);
  const tensor<std::complex<double>> all_threads  {numerical::fft(stack)};
  EXPECT(same_bits(one_thread.data(), all_threads.data()));
  for (std::size_t r {0}; r < rows; ++r) {
    const tensor<std::complex<double>> line  {{cols}, std::vector<std::complex<double>>(values.begin() + static_cast<std::ptrdiff_t>(r * cols),
                                                                                        values.begin() + static_cast<std::ptrdiff_t>((r + 1) * cols))};
    const tensor<std::complex<double>> alone {numerical::fft(line)};
    EXPECT(std::equal(alone.data().begin(), alone.data().end(), all_threads.data().begin() + static_cast<std::ptrdiff_t>(r * cols)));
  }
  const tensor<std::complex<double>> columns {numerical::fft(stack, std::nullopt, 0)};
  for (std::size_t c {0}; c < cols; c += 13) {
    std::vector<std::complex<double>> column(rows);
    for (std::size_t r {0}; r < rows; ++r) {
      column[r] = values[(r * cols) + c];
    }
    const tensor<std::complex<double>> alone {numerical::fft(tensor<std::complex<double>> {{rows}, column})};
    for (std::size_t r {0}; r < rows; ++r) {
      EXPECT(columns.data()[(r * cols) + c] == alone.data()[r]);
    }
  }
  const tensor<double> empty                   {{0, 4}, std::vector<double> {}};
  EXPECT((numerical::fft(empty, 8).shape() == std::vector<std::size_t> {0, 8}));
  const tensor<double>               no_points {{3, 0}, std::vector<double> {}};
  const tensor<std::complex<double>> padded    {numerical::fft(no_points, 2)}; // an empty axis padded: zeros
  EXPECT(padded.shape() == (std::vector<std::size_t> {3, 2}) && padded.data()[5] == std::complex<double>(0.0, 0.0));
}

// Many lengths in turn: the least recently used plan dropped past the store's size, the one in use kept; a plan held
// outlives its eviction.
namespace {
  // A value of T drawn from the normal distribution: complex parts drawn in turn.
  template <typename T>
  T drawn_value(std::mt19937_64& draws)
  {
    std::normal_distribution<double> normal;
    if constexpr (std::is_floating_point_v<T>) {
      return static_cast<T>(normal(draws));
    }
    else {
      const auto re {static_cast<T::value_type>(normal(draws))};
      return T {re, static_cast<T::value_type>(normal(draws))};
    }
  }

  // fft, or ifft where `inverse`, of x to `points` along `axis`.
  template <typename T>
  auto transformed(const tensor<T>& x,
                   bool             inverse,
                   std::size_t      points,
                   std::ptrdiff_t   axis,
                   fft_norm         norm)
  {
    return inverse ? numerical::ifft(x, points, axis, norm) : numerical::fft(x, points, axis, norm);
  }

  // Each line of a batch of `lines` lines of n points of T, along the last axis (`along_last`) or the middle one of a
  // (2, n, lines) tensor, transformed to `points` (fft, or ifft where `inverse`) has the bits of its transform alone.
  template <typename T>
  bool batch_lines_are_alone(std::size_t n,
                             std::size_t lines,
                             std::size_t points,
                             bool        inverse,
                             bool        along_last,
                             fft_norm    norm)
  {
    // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
    std::mt19937_64   draws {(n * 131) + lines};
    const std::size_t outer {along_last ? std::size_t {1} : std::size_t {2}};
    std::vector<T>    values(outer * n * lines);
    for (T& v : values) {
      v = drawn_value<T>(draws);
    }
    std::vector<std::size_t> shape {lines, n};
    if (!along_last) {
      shape = {2, n, lines};
    }
    const tensor<T> batch      {shape, values};
    const auto      all        {transformed(batch, inverse, points, along_last ? -1 : 1, norm)};
    bool            good       {true};
    for (std::size_t o {0}; o < outer; ++o) {
      for (std::size_t l {0}; l < lines; ++l) {
        std::vector<T> line(n);
        for (std::size_t k {0}; k < n; ++k) {
          line[k] = along_last ? values[(l * n) + k] : values[(o * n * lines) + (k * lines) + l];
        }
        const auto alone {transformed(tensor<T> {{n}, line}, inverse, points, -1, norm)};
        for (std::size_t k {0}; k < points; ++k) {
          const std::size_t at {along_last ? (l * points) + k : (o * points * lines) + (k * lines) + l};
          good = good && same_bits(std::vector {alone.data()[k]}, std::vector {all.data()[at]});
        }
      }
    }
    return good;
  }

  // batch_lines_are_alone over lengths of every kind, both axes, cut and padded, each scaling.
  template <typename T>
  bool every_batch_line_is_alone()
  {
    bool good {true};
    for (const std::size_t n : {1UL, 2UL, 12UL, 16UL, 32UL, 60UL, 64UL, 67UL, 96UL, 100UL}) {
      for (const std::size_t lines : {1UL, 9UL, 17UL}) {
        good = good && batch_lines_are_alone<T>(n, lines, n, false, true, fft_norm::backward);
        good = good && batch_lines_are_alone<T>(n, lines, n + 3, true, true, fft_norm::ortho);
        good = good && batch_lines_are_alone<T>(n, lines, (n / 2) + 1, false, false, fft_norm::forward);
        good = good && batch_lines_are_alone<T>(n, lines, n, true, false, fft_norm::backward);
        good = good && batch_lines_are_alone<T>(n, lines, n + 5, false, false, fft_norm::ortho); // columns padded
      }
    }
    good = good && batch_lines_are_alone<T>(1024, 9, 1024, false, true, fft_norm::backward);  // passes of 16 past their first butterfly
    good = good && batch_lines_are_alone<T>(1024, 9, 1024, true, false, fft_norm::ortho);
    good = good && batch_lines_are_alone<T>(8192, 8, 8192, false, true, fft_norm::backward);  // past the lanes' bytes along the last axis
    return good && batch_lines_are_alone<T>(4100, 8, 4100, false, false, fft_norm::backward); // past their length across it
  }
} // namespace

// On every path this processor runs: each line of a batch, along either kind of axis, has the bits of its transform
// alone, whether a register of lines or a line alone transformed it (the lanes of the vector paths: every input type,
// cut, padded and scaled, lines past a whole register of them, lengths past the lanes' bounds and Bluestein's).
TEST(every_line_of_a_batch_has_its_bits_alone_on_every_path)
{
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue; // a path this processor does not run
    }
    EXPECT(every_batch_line_is_alone<std::complex<double>>());
    EXPECT(every_batch_line_is_alone<std::complex<float>>());
    EXPECT(every_batch_line_is_alone<double>());
    EXPECT(every_batch_line_is_alone<float>());
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

namespace {
  // One pass of radix r (3 or 5) over `current` points at stride `stride`, by the generic pass of an odd radix and by
  // the radix's own butterflies, as Fused: the same bits.
  template <bool Fused,
            std::floating_point R>
  bool generic_pass_is_the_butterflys(std::size_t radix,
                                      std::size_t current,
                                      std::size_t stride)
  {
    using C = std::complex<R>;
    // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
    std::mt19937_64                  draws {(radix * 7) + current};
    std::normal_distribution<double> normal;
    std::vector<C>                   in(current * stride);
    std::vector<C>                   generic(in.size());
    std::vector<C>                   own(in.size());
    std::vector<C>                   twiddles((radix - 1) * (current / radix));
    std::vector<C>                   roots(radix);
    for (C& v : in) {
      v = drawn_value<C>(draws);
    }
    for (C& v : twiddles) {
      v = drawn_value<C>(draws);
    }
    for (std::size_t m {0}; m < radix; ++m) {
      const std::complex<double> root {numerical::detail::unit_root(m, radix)};
      roots[m] = C {static_cast<R>(root.real()), static_cast<R>(root.imag())};
    }
    const auto constants {numerical::detail::make_small_radix_constants<R>()};
    numerical::detail::any_radix_pass_body<Fused>(radix, current, stride, in.data(), generic.data(), twiddles.data(), roots.data());
    if (radix == 3) {
      numerical::detail::radix_three_pass_body<Fused>(current, stride, in.data(), own.data(), twiddles.data(), constants);
    }
    else {
      numerical::detail::radix_five_pass_body<Fused>(current, stride, in.data(), own.data(), twiddles.data(), constants);
    }
    return same_bits(generic, own);
  }
} // namespace

// The pass of any odd radix computes radices 3 and 5 as their own butterflies do, to the bit, fused or not, past a
// pass' first butterfly and along a stride: the symmetry of the roots its sums take is theirs.
TEST(the_generic_pass_gives_radices_three_and_five_their_butterflies_bits)
{
  for (const std::size_t radix : {3UL, 5UL}) {
    for (const std::size_t current : {radix, 3 * radix, 4 * radix * radix}) {
      for (const std::size_t stride : {1UL, 4UL}) {
        EXPECT((generic_pass_is_the_butterflys<false, double>(radix, current, stride)));
        EXPECT((generic_pass_is_the_butterflys<true, double>(radix, current, stride)));
        EXPECT((generic_pass_is_the_butterflys<false, float>(radix, current, stride)));
        EXPECT((generic_pass_is_the_butterflys<true, float>(radix, current, stride)));
      }
    }
  }
}

TEST(fft_plans_are_kept_and_renewed)
{
  const auto kept {numerical::detail::shared_fft_plan<double>(7)};
  const auto held {numerical::detail::shared_fft_plan<double>(9)};
  for (std::size_t n {100}; n < 100 + numerical::detail::fft_plans_kept; ++n) {
    EXPECT(numerical::detail::shared_fft_plan<double>(7) == kept); // used at every step: never the least recent
    (void)numerical::detail::shared_fft_plan<double>(n);
  }
  EXPECT(numerical::detail::shared_fft_plan<double>(9) != held && held->length == 9); // dropped, rebuilt; ours lives on
  for (std::size_t n {1}; n <= numerical::detail::fft_plans_kept + 8; ++n) {
    const tensor<double>               ones     {{n}, std::vector<double>(n, 1.0)};
    const tensor<std::complex<double>> spectrum {numerical::fft(ones)};
    EXPECT(std::abs(spectrum.data()[0] - std::complex<double>(static_cast<double>(n), 0.0)) < 1e-12);
  }
}

// A real transform of an even length whose line holds an infinity takes the complex transform of its points, as
// NumPy's rfft carries it: [∞, 0, 0, 0] to ∞ at every frequency, where the recombination of its pairs gave ∞ − ∞, NaN;
// its inverse from a spectrum holding one as well; both precisions.
TEST(real_transforms_carry_an_infinity_as_the_complex_one)
{
  const double                       inf      {std::numeric_limits<double>::infinity()};
  const tensor<std::complex<double>> spectrum {numerical::rfft(tensor<double> {{4}, std::vector<double> {inf, 0.0, 0.0, 0.0}})};
  for (const std::complex<double>& value : spectrum.data()) {
    EXPECT(std::isinf(value.real()) && value.imag() == 0.0);
  }
  const tensor<std::complex<float>> single {numerical::rfft(tensor<float> {{2}, std::vector<float> {std::numeric_limits<float>::infinity(), 1.0F}})};
  EXPECT(std::isinf(single.data()[0].real()) && std::isinf(single.data()[1].real()));
  const tensor<double> back                {numerical::irfft(tensor<std::complex<double>> {{3}, std::vector<std::complex<double>> {{inf, 0.0}, {0.0, 0.0}, {0.0, 0.0}}})};
  for (const double value : back.data()) {
    EXPECT(std::isinf(value));
  }
  const tensor<float> narrow {numerical::irfft(tensor<std::complex<float>> {{2}, std::vector<std::complex<float>> {{1.0F, 0.0F}, {std::numeric_limits<float>::infinity(), 0.0F}}})};
  EXPECT(std::isinf(narrow.data()[0]) && std::isinf(narrow.data()[1]));
  // An infinity at an odd point, and in either part of a frequency between the first and the last: the complex
  // transform's answer, value for value (an infinity, a NaN, or the same number).
  const auto                         alike = [](double a, double b) { return std::isnan(a) ? std::isnan(b) : a == b; };
  const std::vector<double>          odd     {0.0, inf, 0.0, 0.0};
  const tensor<std::complex<double>> half    {numerical::rfft(tensor<double> {{4}, odd})};
  const tensor<std::complex<double>> whole   {numerical::fft(tensor<std::complex<double>> {{4}, std::vector<std::complex<double>>(odd.begin(), odd.end())})};
  for (std::size_t k {0}; k < 3; ++k) {
    EXPECT(alike(half.data()[k].real(), whole.data()[k].real()) && alike(half.data()[k].imag(), whole.data()[k].imag()));
  }
  for (const std::complex<double> middle : {std::complex<double> {inf, 0.0}, std::complex<double> {0.0, inf}}) {
    const tensor<double>               real    {numerical::irfft(tensor<std::complex<double>> {{3}, std::vector<std::complex<double>> {{1.0, 0.0}, middle, {0.0, 0.0}}})};
    const tensor<std::complex<double>> complex {numerical::ifft(tensor<std::complex<double>> {{4}, std::vector<std::complex<double>> {{1.0, 0.0}, middle, {0.0, 0.0}, std::conj(middle)}})};
    for (std::size_t j {0}; j < 4; ++j) {
      EXPECT(alike(real.data()[j], complex.data()[j].real()));
    }
  }
  const tensor<double>               last         {numerical::irfft(tensor<std::complex<double>> {{3}, std::vector<std::complex<double>> {{1.0, 0.0}, {0.0, 0.0}, {inf, 0.0}}})};
  const tensor<std::complex<double>> last_complex {numerical::ifft(tensor<std::complex<double>> {{4}, std::vector<std::complex<double>> {{1.0, 0.0}, {0.0, 0.0}, {inf, 0.0}, {0.0, 0.0}}})};
  for (std::size_t j {0}; j < 4; ++j) {
    EXPECT(alike(last.data()[j], last_complex.data()[j].real())); // the last frequency's infinity
  }
}

// An infinite input is carried, not turned into NaN by a factor of 1: [∞, 0, 0, 0] transforms to ∞ everywhere.
TEST(fft_carries_an_infinity_through_factors_of_one)
{
  const tensor<double>               impulse  {{4}, std::vector<double> {std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0}};
  const tensor<std::complex<double>> spectrum {numerical::fft(impulse)};
  for (const std::complex<double>& value : spectrum.data()) {
    EXPECT(std::isinf(value.real()) && value.imag() == 0.0);
  }
}

// NumPy's errors: an axis out of range and no points.
TEST(fft_refuses_a_missing_axis_and_no_points)
{
  const tensor<double> x {{2, 3}, std::vector<double>(6, 1.0)};
  EXPECT_THROWS((void)numerical::fft(x, std::nullopt, 2), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::fft(x, std::nullopt, -3), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::ifft(x, 0), numerical::dimension_error);
  EXPECT(numerical::fft(x, std::nullopt, -2).shape() == (std::vector<std::size_t> {2, 3}));
  const tensor<std::complex<double>> z {{2, 3}, std::vector<std::complex<double>>(6, {1.0, -1.0})};
  EXPECT_THROWS((void)numerical::fft(z, std::nullopt, 2), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::fft(z, std::nullopt, -3), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::ifft(z, 0), numerical::dimension_error);
  const tensor<std::complex<double>> none {{0, 3}, std::vector<std::complex<double>> {}};
  EXPECT(numerical::fft(none, 5).shape() == (std::vector<std::size_t> {0, 5}));
}

namespace {
  // A 3 × 4 tensor of T transformed along a positive axis padded to 8 points and inverted along a negative one cut to
  // 2, ortho: each matches the same lines transformed one by one as complex numbers, within a few units.
  template <typename T>
  bool every_axis_point_and_scaling_runs(numerical::fft_norm norm)
  {
    std::vector<T> values(12);
    for (std::size_t i {0}; i < values.size(); ++i) {
      values[i] = T(static_cast<float>(i) - 5.5F);
    }
    const tensor<T> x {{3, 4}, values};
    const auto      padded {numerical::fft(x, 8, 0, norm)};
    const auto      cut    {numerical::ifft(x, 2, -1, norm)};
    using C = numerical::detail::fft_complex<T>::type;
    const auto ok {[](const C& p, const C& q) { return std::abs(p - q) < 1e-4F; }};
    bool       good {padded.shape() == (std::vector<std::size_t> {8, 4}) && cut.shape() == (std::vector<std::size_t> {3, 2})};
    for (std::size_t c {0}; c < 4; ++c) {
      std::vector<C> column(8);
      for (std::size_t r {0}; r < 3; ++r) {
        column[r] = C(values[(r * 4) + c]);
      }
      const tensor<C> alone {numerical::fft(tensor<C> {{8}, column}, std::nullopt, -1, norm)};
      for (std::size_t r {0}; r < 8; ++r) {
        good = good && ok(padded.data()[(r * 4) + c], alone.data()[r]);
      }
    }
    // Round trips along the lines apart (axis 0) and along the lines together, padded: the input back, zeros past it.
    const tensor<C> back_columns {numerical::ifft(numerical::fft(x, 8, 0, norm), std::nullopt, 0, norm)};
    const tensor<C> back_rows    {numerical::ifft(numerical::fft(x, 8, -1, norm), std::nullopt, -1, norm)};
    for (std::size_t r {0}; r < 8; ++r) {
      for (std::size_t c {0}; c < 4; ++c) {
        good = good && ok(back_columns.data()[(r * 4) + c], r < 3 ? C(values[(r * 4) + c]) : C {});
      }
    }
    for (std::size_t r {0}; r < 3; ++r) {
      for (std::size_t c {0}; c < 8; ++c) {
        good = good && ok(back_rows.data()[(r * 8) + c], c < 4 ? C(values[(r * 4) + c]) : C {});
      }
    }
    for (std::size_t r {0}; r < 3; ++r) {
      const tensor<C> alone {numerical::ifft(tensor<C> {{2}, std::vector<C> {C(values[r * 4]), C(values[(r * 4) + 1])}}, std::nullopt, 0, norm)};
      good = good && ok(cut.data()[r * 2], alone.data()[0]) && ok(cut.data()[(r * 2) + 1], alone.data()[1]);
    }
    return good;
  }
} // namespace

// Each type through a positive and a negative axis, padding and cutting, forward and inverse, every scaling.
TEST(fft_runs_every_axis_point_and_scaling_in_each_type)
{
  for (const fft_norm norm : {fft_norm::backward, fft_norm::ortho, fft_norm::forward}) {
    EXPECT(every_axis_point_and_scaling_runs<float>(norm));
    EXPECT(every_axis_point_and_scaling_runs<double>(norm));
    EXPECT(every_axis_point_and_scaling_runs<std::complex<float>>(norm));
    EXPECT(every_axis_point_and_scaling_runs<std::complex<double>>(norm));
  }
}

// On every path this processor runs, the transform is as accurate; the fused paths (NEON, AMX, AVX2, AVX-512: each
// factor's product fused alike) give the same bits as each other, whatever their vectors' width (float: passes of 3 and
// 5 at strides a register of AVX-512 fills, 4800 = 4·16·3·5·5 and 10000 = 16·5⁴; odd radices past 5 at odd strides,
// 1001 = 7·11·13, and at even ones, 1984 = 64·31 and 15616 = 256·61).
TEST(every_path_is_accurate_and_the_fused_ones_agree_to_the_bit)
{
  std::vector<numerical::buffer<std::complex<double>>> fused;
  std::vector<numerical::buffer<std::complex<float>>>  fused_float;
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue; // a path this processor does not run
    }
    // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
    std::mt19937_64 draws {3};
    for (std::size_t n {1}; n <= 2048; n *= 2) {
      EXPECT(relative_error<double>(n, draws) < 0.5);
      EXPECT(relative_error<float>(n, draws) < 0.5);
    }
    for (const std::size_t n : {360UL, 1009UL}) { // mixed radices; Bluestein's
      EXPECT(relative_error<double>(n, draws) < 0.5);
      EXPECT(relative_error<float>(n, draws) < 0.5);
    }
    if (path != numerical::gemm_path::portable) {
      numerical::buffer<std::complex<double>> spectra;
      for (const std::size_t n : {1024UL, 1000UL, 1009UL, 1001UL, 15616UL}) { // radices 4 and 16; mixed; Bluestein's; 7·11·13; 61·256
        // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
        std::mt19937_64                    same {5};
        const tensor<std::complex<double>> x    {{n}, drawn<double>(n, same)};
        const tensor<std::complex<double>> y    {numerical::fft(x)};
        spectra.insert(spectra.end(), y.data().begin(), y.data().end());
      }
      fused.push_back(spectra);
      numerical::buffer<std::complex<float>> spectra_float;
      for (const std::size_t n : {4800UL, 10000UL, 1001UL, 1984UL}) {
        // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
        std::mt19937_64                   same {7};
        const tensor<std::complex<float>> x    {{n}, drawn<float>(n, same)};
        const tensor<std::complex<float>> y    {numerical::fft(x)};
        spectra_float.insert(spectra_float.end(), y.data().begin(), y.data().end());
      }
      fused_float.push_back(spectra_float);
    }
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
  for (const auto& bits : fused) {
    EXPECT(same_bits(bits, fused.front()));
  }
  for (const auto& bits : fused_float) {
    EXPECT(same_bits(bits, fused_float.front()));
  }
}

// A transform past the working space a thread keeps (fft_scratch_kept): its own allocation, the same transform.
TEST(a_long_transform_takes_its_own_working_space)
{
  const std::size_t                  n        {2 * numerical::detail::fft_scratch_kept};
  const tensor<double>               ones     {{n}, std::vector<double>(n, 1.0)};
  const tensor<std::complex<double>> spectrum {numerical::fft(ones)};
  EXPECT(spectrum.data()[0] == std::complex<double>(static_cast<double>(n), 0.0));
  EXPECT(std::all_of(spectrum.data().begin() + 1, spectrum.data().end(), [](const std::complex<double>& v) { return std::abs(v) < 1e-9; }));
}

// Every length from 1 to 300 in both precisions, on every path this processor runs: ifft undoes fft. The registers'
// loops end at whole registers, whatever the radices and strides (a length past them once wrote out of bounds).
TEST(every_length_round_trips_on_every_path)
{
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue; // a path this processor does not run
    }
    // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
    std::mt19937_64 draws {17};
    for (std::size_t n {1}; n <= 300; ++n) {
      const tensor<std::complex<double>> x       {{n}, drawn<double>(n, draws)};
      const tensor<std::complex<float>>  y       {{n}, drawn<float>(n, draws)};
      const tensor<std::complex<double>> x_back  {numerical::ifft(numerical::fft(x))};
      const tensor<std::complex<float>>  y_back  {numerical::ifft(numerical::fft(y))};
      double                             worst_x {0.0};
      double                             worst_y {0.0};
      for (std::size_t j {0}; j < n; ++j) {
        worst_x = std::max(worst_x, std::abs(x_back.data()[j] - x.data()[j]));
        worst_y = std::max(worst_y, static_cast<double>(std::abs(y_back.data()[j] - y.data()[j])));
      }
      EXPECT(worst_x < 1e-13);
      EXPECT(worst_y < 1e-5);
    }
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

namespace {
  // Output k of the transform of x, its sum in double-double arithmetic with double-double roots (one row of
  // double_double_dft): exact to some 10⁻³⁰ of the sum of the terms' magnitudes.
  std::complex<double> double_double_bin(const std::vector<std::complex<double>>&  x,
                                         const numerical::detail::unit_roots&      roots,
                                         std::size_t                               k)
  {
    using numerical::detail::dd_add;
    using numerical::detail::dd_mul;
    using numerical::detail::dd_negate;
    const std::size_t n     {x.size()};
    double_double     re    {.hi = 0.0, .lo = 0.0};
    double_double     im    {.hi = 0.0, .lo = 0.0};
    std::size_t       power {0}; // j·k mod n
    for (std::size_t j {0}; j < n; ++j) {
      const auto          parts {roots.parts(power)};
      const double_double xr    {.hi = x[j].real(), .lo = 0.0};
      const double_double xi    {.hi = x[j].imag(), .lo = 0.0};
      re    = dd_add(re, dd_add(dd_mul(xr, parts.first), dd_negate(dd_mul(xi, parts.second))));
      im    = dd_add(im, dd_add(dd_mul(xr, parts.second), dd_mul(xi, parts.first)));
      power = (power + k) % n;
    }
    return {re.hi, im.hi};
  }
} // namespace

// Bluestein's convolution runs at the power of two at least 2n − 1, or at a length of factors 2, 3 and 5 at most 3/5 of
// it from 1024; a length with a prime factor p past 61, at least 16 times p and 8192 points, in two steps, Bluestein's
// along the longer side (the rows where p is), a shorter one over its whole length.
TEST(bluestein_lengths_and_splits_follow_the_lengths_alone)
{
  EXPECT(numerical::detail::bluestein_length(67) == 256);    // 135 under 1024
  EXPECT(numerical::detail::bluestein_length(257) == 540);   // 2²·3³·5 against 1024
  EXPECT(numerical::detail::bluestein_length(1009) == 2048); // 2025 past 3/5 of 2048
  EXPECT(numerical::detail::bluestein_length(4099) == 8640);
  EXPECT(numerical::detail::bluestein_length(10007) == 32768);
  EXPECT(numerical::detail::largest_prime_factor(2) == 2 && numerical::detail::largest_prime_factor(67UL * 67UL) == 67 &&
         numerical::detail::largest_prime_factor(1024) == 2 && numerical::detail::largest_prime_factor(3UL * 1009UL) == 1009);
  const auto shape = [](std::size_t n) {
                       const auto plan {numerical::detail::shared_fft_plan<double>(n)};
                       return plan->four_step ? std::pair {plan->four_step->columns, plan->four_step->rows} : std::pair {std::size_t {0}, std::size_t {0}};
                     };
  EXPECT(shape(1009UL * 16UL) == (std::pair {std::size_t {1009}, std::size_t {16}}));                              // rows of 1009 points
  EXPECT(shape(67UL * 128UL) == (std::pair {std::size_t {128}, std::size_t {67}}));                                // columns of 67 points
  EXPECT(shape(67UL * 64UL).first == 0 && numerical::detail::shared_fft_plan<double>(67UL * 64UL)->bluestein);     // under 8192 points
  EXPECT(shape(10007UL * 2UL).first == 0 && numerical::detail::shared_fft_plan<double>(10007UL * 2UL)->bluestein); // twice p
}

// Long transforms, one of each kind of length (radices 4 and 16; 3; 3, 5 and 7; Bluestein's over a power of two, over
// a length of factors 2, 3 and 5, along the rows of two steps and down their columns): 16 drawn outputs against
// their sums in double-double arithmetic, each within 2·ε·log₂ n of the input's norm, as a transform's error grows.
TEST(long_transforms_are_within_epsilon_log_n_of_their_exact_outputs)
{
  for (const std::size_t n : {std::size_t {1} << 20, std::size_t {3} << 18, std::size_t {44100}, std::size_t {20014}, std::size_t {4099},
                              std::size_t {1009UL * 16UL}, std::size_t {67UL * 128UL}}) {
    // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
    std::mt19937_64                         draws {n};
    const std::vector<std::complex<double>> x     {drawn<double>(n, draws)};
    const tensor<std::complex<double>>      got   {numerical::fft(tensor<std::complex<double>> {{n}, x})};
    const numerical::detail::unit_roots     roots {n};
    double                                  norm  {0.0};
    for (const std::complex<double>& v : x) {
      norm += std::norm(v);
    }
    const double                               bound {2.0 * std::numeric_limits<double>::epsilon() * std::log2(static_cast<double>(n)) * std::sqrt(norm)};
    std::uniform_int_distribution<std::size_t> pick  {0, n - 1};
    for (int sample {0}; sample < 16; ++sample) {
      const std::size_t k {pick(draws)};
      EXPECT(std::abs(got.data()[k] - double_double_bin(x, roots, k)) < bound);
    }
  }
}

namespace {
  // A transform in four steps (fft_four_step_least bytes and more) of n points of R: 6 outputs within 2·ε·log₂ n·‖x‖ of
  // the exact ones, and every one of the radix-2 split of two shorter transforms; the same bits on one thread as on all,
  // and alone as among two lines; with inverse_too, its inverse and a line padded with zeros to it as well.
  template <std::floating_point R>
  bool four_step_transform_holds(std::size_t n,
                                 bool        inverse_too = false)
  {
    using C = std::complex<R>;
    // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
    std::mt19937_64                   draws {n};
    std::vector<C>                    x     {drawn<R>(n, draws)};
    std::vector<std::complex<double>> wide  (n);
    std::vector<C>                    twice (2 * n);
    double                            norm  {0.0};
    for (std::size_t j {0}; j < n; ++j) {
      wide[j]         = std::complex<double>(x[j]);
      twice[j]        = x[n - 1 - j];
      twice[n + j]    = x[j];
      norm           += std::norm(wide[j]);
    }
    numerical::set_max_threads(1);
    const tensor<C> alone_one                        {numerical::fft(tensor<C> {{n}, x})};
    numerical::set_max_threads(0);
    const tensor<C> alone                            {numerical::fft(tensor<C> {{n}, x})};
    const tensor<C> pair                             {numerical::fft(tensor<C> {{2, n}, twice})};
    bool            good                             {same_bits(alone.data(), alone_one.data()) &&
                                                      same_bits(std::vector<C>(pair.data().begin() + static_cast<std::ptrdiff_t>(n), pair.data().end()), alone.data())};
    const numerical::detail::unit_roots        roots {n};
    const double                               bound {2.0 * static_cast<double>(std::numeric_limits<R>::epsilon()) * std::log2(static_cast<double>(n)) * std::sqrt(norm)};
    std::uniform_int_distribution<std::size_t> pick  {0, n - 1};
    for (int sample {0}; sample < 6; ++sample) {
      const std::size_t k {pick(draws)};
      good = good && std::abs(std::complex<double>(alone.data()[k]) - double_double_bin(wide, roots, k)) < bound;
    }
    // Every output against the radix-2 split of two transforms of n/2 points (in passes, shorter than four steps take):
    // X_k = E_k + w^k·O_k, X_{k+n/2} = E_k − w^k·O_k, in double with the exact roots.
    std::vector<C> even(n / 2);
    std::vector<C> odd(n / 2);
    for (std::size_t j {0}; j < n / 2; ++j) {
      even[j] = x[2 * j];
      odd[j]  = x[(2 * j) + 1];
    }
    const tensor<C> evens {numerical::fft(tensor<C> {{n / 2}, even})};
    const tensor<C> odds  {numerical::fft(tensor<C> {{n / 2}, odd})};
    for (std::size_t k {0}; k < n / 2; ++k) {
      const std::complex<double> product {roots.root(k) * std::complex<double>(odds.data()[k])};
      good = good && std::abs(std::complex<double>(alone.data()[k]) - (std::complex<double>(evens.data()[k]) + product)) < bound &&
             std::abs(std::complex<double>(alone.data()[k + (n / 2)]) - (std::complex<double>(evens.data()[k]) - product)) < bound;
    }
    if (!inverse_too) {
      return good;
    }
    // The inverse: conj(fft(conj x)) times 1/n, element for element; a line padded with zeros: its padded copy's.
    std::vector<C> conjugated(n);
    std::vector<C> padded(x);
    std::transform(x.begin(), x.end(), conjugated.begin(), [](const C& v) { return std::conj(v); });
    std::fill(padded.end() - 5, padded.end(), C {});
    const tensor<C> back    {numerical::ifft(tensor<C> {{n}, x})};
    const tensor<C> forward {numerical::fft(tensor<C> {{n}, conjugated})};
    const R         scale   {R(1) / static_cast<R>(n)};
    std::vector<C>  wanted  (n);
    std::transform(forward.data().begin(), forward.data().end(), wanted.begin(), [scale](const C& v) { return C {v.real() * scale, -v.imag() * scale}; });
    good = good && same_bits(back.data(), wanted);
    const tensor<C> cut     {{n - 5}, std::vector<C>(x.begin(), x.end() - 5)};
    good = good && same_bits(numerical::fft(cut, n).data(), numerical::fft(tensor<C> {{n}, padded}).data());
    return good;
  }
} // namespace

// On every path this processor runs: transforms in four steps (2²⁰ and 3·2¹⁹ points of double, 2²¹ of float: a square
// and an oblong matrix of points; the shortest, 2¹⁸ of double and 2¹⁹ of float, and 2⁴·3·5³·7² of double; 67·2¹⁴, a
// prime past the passes', whose Bluestein's convolution takes them) within ε·log₂ n of the exact transform, their bits
// a line's own on any threads.
TEST(four_step_transforms_are_exact_to_epsilon_log_n_and_their_own_on_any_threads)
{
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue; // a path this processor does not run
    }
    EXPECT(four_step_transform_holds<double>(std::size_t {1} << 20));
    EXPECT(four_step_transform_holds<double>(std::size_t {3} << 19));
    EXPECT(four_step_transform_holds<float>(std::size_t {1} << 21));
    EXPECT(four_step_transform_holds<double>(std::size_t {67} << 14));      // Bluestein's, its transforms in four steps
    EXPECT(four_step_transform_holds<double>(std::size_t {1} << 18, true)); // its inverse and a padded line too
    EXPECT(four_step_transform_holds<float>(std::size_t {1} << 19));
    EXPECT(four_step_transform_holds<double>(std::size_t {294000}));        // 2⁴·3·5³·7²
    const std::size_t   n {std::size_t {1} << 20};
    std::vector<double> real(n);
    for (std::size_t j {0}; j < n; ++j) {
      real[j] = static_cast<double>((j * 37) % 101) - 50.0;
    }
    std::vector<std::complex<double>> copy(real.begin(), real.end());
    const auto                        spectrum {numerical::fft(tensor<double> {{n}, real})};
    const auto                        complex  {numerical::fft(tensor<std::complex<double>> {{n}, copy})};
    EXPECT(same_bits(spectrum.data(), complex.data())); // a real input of four steps: the complex transform's own bits
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

// fftshift moves element 0 to the middle along an axis, as numpy.fft.fftshift (j to (j + n/2) mod n), ifftshift moves
// it back (by −n/2: the two differ on an odd extent); the other axes untouched; an empty axis and a missing one.
TEST(fftshift_rolls_an_axis_by_half_its_extent)
{
  const tensor<double> odd {{5}, std::vector<double> {0, 1, 2, 3, 4}};
  EXPECT(numerical::fftshift(odd).data() == (numerical::buffer<double>({3, 4, 0, 1, 2})));
  EXPECT(numerical::fftshift(odd, 0, true).data() == (numerical::buffer<double>({2, 3, 4, 0, 1})));
  EXPECT(numerical::fftshift(numerical::fftshift(odd), -1, true) == odd);
  const tensor<double> grid {{2, 4}, std::vector<double> {0, 1, 2, 3, 4, 5, 6, 7}};
  EXPECT(numerical::fftshift(grid, 1).data() == (numerical::buffer<double>({2, 3, 0, 1, 6, 7, 4, 5})));
  EXPECT(numerical::fftshift(grid, -2).data() == (numerical::buffer<double>({4, 5, 6, 7, 0, 1, 2, 3})));
  EXPECT(numerical::fftshift(tensor<double> {{3, 0}, std::vector<double> {}}, 1).shape() == (std::vector<std::size_t> {3, 0}));
  EXPECT(numerical::fftshift(tensor<double> {{0, 3}, std::vector<double> {}}, 1).shape() == (std::vector<std::size_t> {0, 3}));
  EXPECT_THROWS((void)numerical::fftshift(grid, 2), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::fftshift(grid, -3), numerical::dimension_error);
  // Several axes in one copy: the rolls of each in turn; an axis given twice rolled twice; none, a copy; a number.
  std::vector<double> values(std::size_t {60}); // 3 × 4 × 5
  for (std::size_t j {0}; j < values.size(); ++j) {
    values[j] = static_cast<double>(j);
  }
  const tensor<double> cube {{3, 4, 5}, values};
  EXPECT(numerical::fftshift(cube, std::vector<std::ptrdiff_t> {0, 2}) == numerical::fftshift(numerical::fftshift(cube, 0), 2));
  EXPECT(numerical::fftshift(cube, std::vector<std::ptrdiff_t> {-1, 1, 0}, true) ==
         numerical::fftshift(numerical::fftshift(numerical::fftshift(cube, 2, true), 1, true), 0, true));
  EXPECT(numerical::fftshift(cube, std::vector<std::ptrdiff_t> {2, 2}) == numerical::fftshift(numerical::fftshift(cube, 2), 2));
  EXPECT(numerical::fftshift(cube, std::vector<std::ptrdiff_t> {}) == cube);
  const tensor<double> number {{}, std::vector<double> {7.0}};
  EXPECT(numerical::fftshift(number, std::vector<std::ptrdiff_t> {}) == number);
  EXPECT_THROWS((void)numerical::fftshift(cube, std::vector<std::ptrdiff_t> {0, 3}), numerical::dimension_error);
}

namespace {
  // rfft of n real points against the first n/2 + 1 outputs of fft, and irfft back, within a few units of R's epsilon
  // times the input's norm, for norm; odd and even n, along the last axis and along the first.
  template <std::floating_point R>
  bool real_transforms_agree(std::size_t n,
                             fft_norm    norm)
  {
    // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
    std::mt19937_64                  draws {n};
    std::normal_distribution<double> normal;
    std::vector<R>                   values(2 * n);
    for (R& v : values) {
      v = static_cast<R>(normal(draws));
    }
    const tensor<R>               x       {{2, n}, values};
    const tensor<std::complex<R>> half    {numerical::rfft(x, std::nullopt, -1, norm)};
    const tensor<std::complex<R>> full    {numerical::fft(x, std::nullopt, -1, norm)};
    const tensor<R>               back    {numerical::irfft(half, n, -1, norm)};
    const tensor<R>               columns {numerical::irfft(numerical::rfft(numerical::fftshift(x, 0), std::nullopt, 0, norm), 2, 0, norm)};
    const double                  bound   {64.0 * static_cast<double>(std::numeric_limits<R>::epsilon()) * std::sqrt(static_cast<double>(n))};
    bool                          good    {half.shape() == (std::vector<std::size_t> {2, (n / 2) + 1})};
    for (std::size_t r {0}; r < 2; ++r) {
      for (std::size_t k {0}; k <= n / 2; ++k) {
        good = good && std::abs(half.data()[(r * ((n / 2) + 1)) + k] - full.data()[(r * n) + k]) < bound * std::sqrt(static_cast<double>(n));
      }
      for (std::size_t j {0}; j < n; ++j) {
        good = good && std::abs(back.data()[(r * n) + j] - values[(r * n) + j]) < bound;
        good = good && std::abs(columns.data()[(r * n) + j] - values[((1 - r) * n) + j]) < bound;
      }
    }
    return good;
  }
} // namespace

// rfft is fft's first n/2 + 1 outputs and irfft undoes it, for every length to 64, both precisions, every scaling,
// along either kind of axis; irfft reads the imaginary parts of the zero and the last frequency as 0, as NumPy's.
TEST(rfft_is_the_first_half_of_fft_and_irfft_undoes_it)
{
  for (std::size_t n {1}; n <= 64; ++n) {
    for (const fft_norm norm : {fft_norm::backward, fft_norm::ortho, fft_norm::forward}) {
      EXPECT(real_transforms_agree<double>(n, norm));
      EXPECT(real_transforms_agree<float>(n, norm));
    }
  }
  const tensor<std::complex<double>> spectrum  {{3}, std::vector<std::complex<double>> {{1.0, 5.0}, {2.0, -1.0}, {3.0, 7.0}}};
  const tensor<std::complex<double>> real_ends {{3}, std::vector<std::complex<double>> {{1.0, 0.0}, {2.0, -1.0}, {3.0, 0.0}}};
  EXPECT(numerical::irfft(spectrum).data() == numerical::irfft(real_ends).data());
  const tensor<std::complex<double>> real_zero {{3}, std::vector<std::complex<double>> {{1.0, 0.0}, {2.0, -1.0}, {3.0, 7.0}}};
  EXPECT(numerical::irfft(spectrum, 5).data() == numerical::irfft(real_zero, 5).data()); // an odd n: the zero frequency's alone
}

// hfft is irfft of the conjugates, ihfft the conjugates of rfft, each with its scaling's direction exchanged.
TEST(hfft_and_ihfft_are_numpys_definitions)
{
  const tensor<std::complex<double>> spectrum   {{4}, std::vector<std::complex<double>> {{1.0, 0.0}, {2.0, -1.0}, {0.5, 3.0}, {3.0, 0.0}}};
  const tensor<std::complex<double>> conjugates {{4}, std::vector<std::complex<double>> {{1.0, 0.0}, {2.0, 1.0}, {0.5, -3.0}, {3.0, 0.0}}};
  EXPECT(numerical::hfft(spectrum).data() == numerical::irfft(conjugates, std::nullopt, -1, fft_norm::forward).data());
  EXPECT(numerical::hfft(spectrum, 7, -1, fft_norm::ortho).data() == numerical::irfft(conjugates, 7, -1, fft_norm::ortho).data());
  const tensor<double>               x       {{5}, std::vector<double> {1.0, -2.0, 0.5, 4.0, 3.0}};
  const tensor<std::complex<double>> forward {numerical::rfft(x, std::nullopt, -1, fft_norm::backward)};
  const tensor<std::complex<double>> inverse {numerical::ihfft(x, std::nullopt, -1, fft_norm::forward)};
  for (std::size_t k {0}; k < 3; ++k) {
    EXPECT(inverse.data()[k] == std::conj(forward.data()[k]));
  }
  EXPECT(numerical::ihfft(x).data()[0] == std::complex<double>(6.5 / 5.0, -0.0));
}

// The real transforms' errors, empty tensors, and their plans renewed as the complex ones are.
TEST(real_transforms_refuse_no_points_and_keep_their_plans)
{
  const tensor<double> x {{2, 3}, std::vector<double>(6, 1.0)};
  EXPECT_THROWS((void)numerical::rfft(x, 0), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::rfft(x, std::nullopt, 2), numerical::dimension_error);
  const tensor<std::complex<double>> one {{1}, std::vector<std::complex<double>> {1.0}};
  EXPECT_THROWS((void)numerical::irfft(one), numerical::dimension_error);                                                     // 2·(1 − 1) points
  EXPECT_THROWS((void)numerical::irfft(tensor<std::complex<double>> {{0}, std::vector<std::complex<double>> {}}), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::hfft(one, std::nullopt, -2), numerical::dimension_error);
  EXPECT(numerical::rfft(tensor<double> {{0, 4}, std::vector<double> {}}).shape() == (std::vector<std::size_t> {0, 3}));
  for (std::size_t n {1}; n <= numerical::detail::fft_plans_kept + 8; ++n) {
    EXPECT(numerical::rfft(tensor<double> {{n}, std::vector<double>(n, 1.0)}).data()[0] == std::complex<double>(static_cast<double>(n), 0.0));
  }
}

namespace {
  // The real transforms of R on the path in use: rfft padded (n past the axis) is fft's first half, irfft padded (more
  // frequencies asked than given) undoes it, odd and even n; their errors.
  template <std::floating_point R>
  bool real_transforms_pad_and_refuse()
  {
    const tensor<R> x    {{5}, std::vector<R> {R(1), R(-2), R(0.5), R(4), R(3)}};
    bool            good {true};
    for (const std::size_t n : {7UL, 8UL}) {
      const tensor<std::complex<R>> half {numerical::rfft(x, n)};
      const tensor<std::complex<R>> full {numerical::fft(x, n)};
      for (std::size_t k {0}; k <= n / 2; ++k) {
        good = good && std::abs(half.data()[k] - full.data()[k]) < R(1e-4);
      }
      const tensor<R> back {numerical::irfft(half, n + 4)}; // its frequencies padded with zeros
      good = good && back.shape() == (std::vector<std::size_t> {n + 4});
    }
    bool refused {false};
    try {
      (void)numerical::rfft(x, 0);
    }
    catch (const numerical::dimension_error&) {
      refused = true;
    }
    return good && refused;
  }
} // namespace

namespace {
  // rfft and irfft of each even n to 2·limit against the scalar recombination's loop (real_forward_body and
  // real_inverse_body with no_registers), to the bit: the registers' recombination, where a path has one, rounds as
  // the loop, about every count of complex numbers left past the registers.
  template <std::floating_point R>
  bool recombinations_match_the_loop(std::size_t limit)
  {
    using C = std::complex<R>;
    bool good {true};
    for (std::size_t n {2}; n <= 2 * limit; n += 2) {
      // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
      std::mt19937_64                  draws {n};
      std::normal_distribution<double> normal;
      std::vector<R>                   values(n);
      for (R& v : values) {
        v = static_cast<R>(normal(draws));
      }
      const auto                     plan     {numerical::detail::shared_real_fft_plan<R>(n)};
      const tensor<C>                spectrum {numerical::rfft(tensor<R> {{n}, values})};
      const tensor<R>                back     {numerical::irfft(spectrum, n)};
      std::vector<C>                 work((n / 2) + 1 + plan->complex->work);
      std::vector<C>                 loop((n / 2) + 1);
      std::vector<R>                 inverse(n);
      numerical::detail::real_forward_body<true>(*plan, values.data(), loop.data(), work.data(), numerical::detail::no_registers<R>);
      numerical::detail::real_inverse_body<true>(*plan, spectrum.data().data(), inverse.data(), work.data(), R(2) / static_cast<R>(n),
                                                 numerical::detail::no_registers<R>);
      good = good && same_bits(spectrum.data(), loop) && same_bits(back.data(), inverse);
    }
    return good;
  }
} // namespace

// On every fused path this processor runs: the recombination of an even n's real transforms is the loop's, to the bit.
TEST(the_real_recombination_has_the_loops_bits_on_every_fused_path)
{
  for (const numerical::gemm_path path : {numerical::gemm_path::neon, numerical::gemm_path::avx2, numerical::gemm_path::avx512,
                                          numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue; // a path this processor does not run
    }
    EXPECT(recombinations_match_the_loop<double>(40));
    EXPECT(recombinations_match_the_loop<float>(40));
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

// On every path this processor runs (the portable one's products apart): padded real transforms, both precisions.
TEST(real_transforms_pad_on_every_path)
{
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue; // a path this processor does not run
    }
    EXPECT(real_transforms_pad_and_refuse<double>());
    EXPECT(real_transforms_pad_and_refuse<float>());
    EXPECT(real_transforms_agree<double>(12, fft_norm::backward)); // even n
    EXPECT(real_transforms_agree<float>(9, fft_norm::ortho));      // odd n
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

namespace {

  //! \brief The unscaled transform of \p x (sines where \p sine) of \p type by its sum, in long double.
  std::vector<double> trig_sum(bool                       sine,
                               int                        type,
                               const std::vector<double>& x)
  {
    const std::size_t   n  {x.size()};
    const long double   pi {std::numbers::pi_v<long double>};
    std::vector<double> y(n);
    for (std::size_t k {0}; k < n; ++k) {
      long double total {0.0L};
      for (std::size_t j {0}; j < n; ++j) {
        const long double value {x[j]};
        const auto        jj    {static_cast<long double>(j)};
        const auto        kk    {static_cast<long double>(k)};
        const auto        nn    {static_cast<long double>(n)};
        long double       term  {};
        if (!sine) {
          if (type == 1) {
            term = value * std::cos(pi * jj * kk / (nn - 1)) * (j == 0 || j == n - 1 ? 1.0L : 2.0L);
          }
          else if (type == 2) {
            term = 2 * value * std::cos(pi * kk * ((2 * jj) + 1) / (2 * nn));
          }
          else if (type == 3) {
            term = j == 0 ? value : 2 * value * std::cos(pi * jj * ((2 * kk) + 1) / (2 * nn));
          }
          else {
            term = 2 * value * std::cos(pi * ((2 * jj) + 1) * ((2 * kk) + 1) / (4 * nn));
          }
        }
        else if (type == 1) {
          term = 2 * value * std::sin(pi * (jj + 1) * (kk + 1) / (nn + 1));
        }
        else if (type == 2) {
          term = 2 * value * std::sin(pi * (kk + 1) * ((2 * jj) + 1) / (2 * nn));
        }
        else if (type == 3) {
          const long double sign {k % 2 == 0 ? 1.0L : -1.0L};
          term = j == n - 1 ? sign * value : 2 * value * std::sin(pi * (jj + 1) * ((2 * kk) + 1) / (2 * nn));
        }
        else {
          term = 2 * value * std::sin(pi * ((2 * jj) + 1) * ((2 * kk) + 1) / (4 * nn));
        }
        total += term;
      }
      y[k] = static_cast<double>(total);
    }
    return y;
  }
} // namespace

// The cosine and sine transforms of scipy.fft, every type: each its sum to 32 n ε on lengths odd and even, a power of
// two and not (DCT-IV's even and odd forms), and the inverse undoing it; float; along a first axis (lines gathered) and
// padded; each norm, orthogonalize on and off; the errors.
TEST(cosine_and_sine_transforms_are_their_sums_and_invert)
{
  using numerical::dct;
  using numerical::dst;
  using numerical::idct;
  using numerical::idst;
  double worst {0.0};
  double back  {0.0};
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::amx}) { // each path's products, a line shared
    (void)numerical::limit_gemm_path(path);
    for (const std::size_t n : {std::size_t {1}, std::size_t {2}, std::size_t {3}, std::size_t {8}, std::size_t {9}, std::size_t {30},
                                std::size_t {97}, std::size_t {128}}) {
      std::vector<double> x(n);
      for (std::size_t j {0}; j < n; ++j) {
        x[j] = std::sin(1.0 + (0.7 * static_cast<double>(j)));
      }
      const tensor<double> line {{n}, numerical::buffer<double>(x.begin(), x.end())};
      for (const bool sine : {false, true}) {
        for (int type {1}; type <= 4; ++type) {
          if (!sine && type == 1 && n < 2) {
            continue;
          }
          const tensor<double>      y        {sine ? dst(line, type) : dct(line, type)};
          const tensor<double>      x_again  {sine ? idst(y, type) : idct(y, type)};
          const std::vector<double> expected {trig_sum(sine, type, x)};
          double                    size     {1.0};
          for (const double value : expected) {
            size = std::max(size, std::abs(value));
          }
          for (std::size_t k {0}; k < n; ++k) {
            worst = std::max(worst, std::abs(y.data()[k] - expected[k]) / (size * static_cast<double>(n)));
            back  = std::max(back, std::abs(x_again.data()[k] - x[k]) / static_cast<double>(n));
          }
        }
      }
    }
  }
  EXPECT(worst < 32 * std::numeric_limits<double>::epsilon());
  EXPECT(back < 32 * std::numeric_limits<double>::epsilon());
  // Along the first axis of a 9 × 2 tensor, padded to 12: each column its own line's transform, cut and padded.
  const std::vector<double> column {1.0, -2.0, 0.5, 3.0, 0.25, -1.0, 2.0, 0.0, 1.5};
  numerical::buffer<double> pairs(18);
  for (std::size_t j {0}; j < 9; ++j) {
    pairs[2 * j]       = column[j];
    pairs[(2 * j) + 1] = -column[j];
  }
  const tensor<double> two_columns {{9, 2}, std::move(pairs)};
  std::vector<double>  padded(column);
  padded.resize(12, 0.0);
  for (int type {1}; type <= 4; ++type) {
    const tensor<double>      along    {dst(two_columns, type, 12, 0)};
    const std::vector<double> expected {trig_sum(true, type, padded)};
    for (std::size_t k {0}; k < 12; ++k) {
      EXPECT(std::abs(along.data()[2 * k] - expected[k]) < 1e-13 && std::abs(along.data()[(2 * k) + 1] + expected[k]) < 1e-13);
    }
  }
  // The scalings: ortho by 1/√(2n) with the first value weighted (DCT-II), forward by 1/2n, and orthogonalize alone.
  const tensor<double>      four       {{4}, numerical::buffer<double> {1.0, 2.0, 3.0, 4.0}};
  const std::vector<double> plain      {trig_sum(false, 2, {1.0, 2.0, 3.0, 4.0})};
  const tensor<double>      ortho      {dct(four, 2, std::nullopt, -1, fft_norm::ortho)};
  const tensor<double>      forward    {dct(four, 2, std::nullopt, -1, fft_norm::forward)};
  const tensor<double>      weighted   {dct(four, 2, std::nullopt, -1, fft_norm::backward, true)};
  const tensor<double>      unweighted {dct(four, 2, std::nullopt, -1, fft_norm::ortho, false)};
  EXPECT(std::abs(ortho.data()[0] - (plain[0] / std::sqrt(2.0) / std::sqrt(8.0))) < 1e-14);
  EXPECT(std::abs(ortho.data()[1] - (plain[1] / std::sqrt(8.0))) < 1e-14);
  EXPECT(std::abs(forward.data()[1] - (plain[1] / 8.0)) < 1e-14);
  EXPECT(std::abs(weighted.data()[0] - (plain[0] / std::sqrt(2.0))) < 1e-14 && std::abs(weighted.data()[1] - plain[1]) < 1e-14);
  EXPECT(std::abs(unweighted.data()[0] - (plain[0] / std::sqrt(8.0))) < 1e-14);
  // An orthonormal pair: ortho's inverse undoes it, for every family and type, the weighted ends (DCT-I's both, DCT-III's
  // first, DST-II's and DST-III's last) included.
  for (const bool sine : {false, true}) {
    for (int type {1}; type <= 4; ++type) {
      const tensor<double> y          {sine ? dst(four, type, std::nullopt, -1, fft_norm::ortho) : dct(four, type, std::nullopt, -1, fft_norm::ortho)};
      const tensor<double> again      {sine ? idst(y, type, std::nullopt, -1, fft_norm::ortho) : idct(y, type, std::nullopt, -1, fft_norm::ortho)};
      double               energy_in  {0.0};
      double               energy_out {0.0};
      for (std::size_t k {0}; k < 4; ++k) {
        EXPECT(std::abs(again.data()[k] - four.data()[k]) < 1e-14);
        energy_in  += four.data()[k] * four.data()[k];
        energy_out += y.data()[k] * y.data()[k];
      }
      EXPECT(std::abs(energy_in - energy_out) < 1e-12); // orthogonal: the norm kept
    }
  }
  const tensor<float> single      {{5}, numerical::buffer<float> {1.0F, -1.0F, 2.0F, 0.5F, 3.0F}};
  const tensor<float> single_back {idct(dct(single, 4), 4)};
  for (std::size_t k {0}; k < 5; ++k) {
    EXPECT(std::abs(single_back.data()[k] - single.data()[k]) < 1e-5F);
  }
  EXPECT_THROWS(dct(four, 0), std::invalid_argument);
  EXPECT_THROWS(dst(four, 5), std::invalid_argument);
  EXPECT_THROWS(dct(tensor<double> {{1}, numerical::buffer<double> {1.0}}, 1), numerical::dimension_error);
  EXPECT_THROWS(dst(four, 2, 0), numerical::dimension_error);
  EXPECT_THROWS(idct(four, 2, std::nullopt, 1), numerical::dimension_error);
}
