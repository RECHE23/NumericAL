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
      double_double re {0.0, 0.0};
      double_double im {0.0, 0.0};
      for (std::size_t j {0}; j < n; ++j) {
        const auto          parts {numerical::detail::unit_root_parts((j * k) % n, n)};
        const double_double xr    {x[j].real(), 0.0};
        const double_double xi    {x[j].imag(), 0.0};
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

  // The same bits, part by part.
  template <typename C>
  bool same_bits(const numerical::buffer<C>& x,
                 const numerical::buffer<C>& y)
  {
    using R = typename C::value_type;
    using U = std::conditional_t<sizeof(R) == 8, std::uint64_t, std::uint32_t>;
    return x.size() == y.size() && std::equal(x.begin(), x.end(), y.begin(), [](const C& p, const C& q) {
                                                return std::bit_cast<U>(p.real()) == std::bit_cast<U>(q.real()) && std::bit_cast<U>(p.imag()) == std::bit_cast<U>(q.imag());
                                              });
  }
} // namespace

// Within half of epsilon · log₂ n in RMS of the exact transform (a single addition at n = 2), in both precisions: every power of two to 4096 (both
// parities of the exponent: radix 4 alone, and its last pass of radix 2), lengths of mixed radices (2, 3, 5, 7, 31, a
// pass of radix 2 before others) and lengths with a prime factor past the largest radix (Bluestein's convolution).
TEST(fft_is_within_a_fraction_of_epsilon_log_n_of_the_exact_transform)
{
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
    using C = typename numerical::detail::fft_complex<T>::type;
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
