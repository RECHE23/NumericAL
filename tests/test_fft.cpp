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
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {
  using numerical::fft_norm;
  using numerical::tensor;
  using numerical::detail::double_double;

  // The transform of x, its sum taken in double-double arithmetic with double-double roots: exact to some 10⁻³⁰.
  std::vector<std::complex<double>> double_double_dft(const std::vector<std::complex<double>>& x)
  {
    using numerical::detail::dd_add;
    using numerical::detail::dd_mul;
    using numerical::detail::dd_negate;
    const std::size_t                 n {x.size()};
    std::vector<std::complex<double>> y(n);
    for (std::size_t k {0}; k < n; ++k) {
      double_double re {.hi = 0.0, .lo = 0.0};
      double_double im {.hi = 0.0, .lo = 0.0};
      for (std::size_t j {0}; j < n; ++j) {
        const auto          parts {numerical::detail::unit_root_parts((j * k) % n, n)};
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
TEST(fft_takes_numpys_points_and_real_input)
{
  const tensor<double>               real   {{5}, std::vector<double> {1.0, -2.0, 0.5, 4.0, 3.0}};
  const tensor<std::complex<double>> copy   {{5}, std::vector<std::complex<double>> {1.0, -2.0, 0.5, 4.0, 3.0}};
  EXPECT(same_bits(numerical::fft(real).data(), numerical::fft(copy).data()));
  const tensor<std::complex<double>> padded {{8}, std::vector<std::complex<double>> {1.0, -2.0, 0.5, 4.0, 3.0, 0.0, 0.0, 0.0}};
  EXPECT(same_bits(numerical::fft(real, 8).data(), numerical::fft(padded).data()));
  const tensor<std::complex<double>> cut    {{2}, std::vector<std::complex<double>> {1.0, -2.0}};
  EXPECT(same_bits(numerical::fft(real, 2).data(), numerical::fft(cut).data()));
  EXPECT(numerical::fft(real, 2).data()[1] == std::complex<double>(3.0, 0.0));
  const tensor<float>               single   {{4}, std::vector<float> {1.0F, 2.0F, 3.0F, 4.0F}};
  const tensor<std::complex<float>> spectrum {numerical::fft(single)};
  EXPECT(spectrum.data()[0] == std::complex<float>(10.0F, 0.0F) && spectrum.data()[2] == std::complex<float>(-2.0F, 0.0F));
  EXPECT(numerical::ifft(single).data()[0] == std::complex<float>(2.5F, 0.0F));
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
// factor's product fused alike) give the same bits as each other, whatever their vectors' width.
TEST(every_path_is_accurate_and_the_fused_ones_agree_to_the_bit)
{
  std::vector<numerical::buffer<std::complex<double>>> fused;
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
      for (const std::size_t n : {1024UL, 1000UL, 1009UL}) { // radices 4 and 16; mixed; Bluestein's
        // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed): a fixed draw, the same on every run
        std::mt19937_64                    same {5};
        const tensor<std::complex<double>> x    {{n}, drawn<double>(n, same)};
        const tensor<std::complex<double>> y    {numerical::fft(x)};
        spectra.insert(spectra.end(), y.data().begin(), y.data().end());
      }
      fused.push_back(spectra);
    }
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
  for (const auto& bits : fused) {
    EXPECT(same_bits(bits, fused.front()));
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

// Long transforms, one of each kind of length (radices 4 and 16; 3; 3, 5 and 7; Bluestein's): 16 drawn outputs against
// their sums in double-double arithmetic, each within 2·ε·log₂ n of the input's norm, as a transform's error grows.
TEST(long_transforms_are_within_epsilon_log_n_of_their_exact_outputs)
{
  for (const std::size_t n : {std::size_t {1} << 20, std::size_t {3} << 18, std::size_t {44100}, std::size_t {20014}}) {
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
