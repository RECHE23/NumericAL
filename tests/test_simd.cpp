// The scaled-row updates (simd.hpp): over native vectors or not, every element equals, to the bit, the scalar
// statement's result, for a length with both a whole number of vectors and a remainder.
#include <complex>
#include <cstddef>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  // Products that are not exactly representable, of both signs: a reordered or fused operation shows in the
  // last bits.
  template <typename T>
  std::vector<T> values(std::size_t count,
                        double      offset)
  {
    std::vector<T> out(count);
    for (std::size_t i {0}; i < count; ++i) {
      out[i] = T((1.0 / (static_cast<double>(i) + offset)) - 0.21);
    }
    return out;
  }

  template <typename T>
  bool updates_match_the_scalar_statements(std::size_t count)
  {
    const std::vector<T> x            {values<T>(count, 3.0)};
    const T              f            {T(1.0 / 7.0)};
    std::vector<T>       down         {values<T>(count, 5.0)};
    std::vector<T>       up           {down};
    std::vector<T>       down_wanted  {down};
    std::vector<T>       up_wanted    {down};
    std::vector<T>       right        {down};
    std::vector<T>       right_wanted {down};
    for (std::size_t i {0}; i < count; ++i) {
      down_wanted[i]  -= f * x[i];
      up_wanted[i]    += f * x[i];
      right_wanted[i] -= x[i] * f;
    }
    numerical::detail::subtract_product(down.data(), f, x.data(), count);
    numerical::detail::add_product(up.data(), f, x.data(), count);
    numerical::detail::subtract_product(right.data(), x.data(), f, count);
    return down == down_wanted && up == up_wanted && right == right_wanted;
  }
} // namespace

TEST(scaled_row_updates_equal_the_scalar_statements_to_the_bit)
{
  for (const std::size_t count : {std::size_t {0}, std::size_t {1}, std::size_t {37}}) {
    EXPECT(updates_match_the_scalar_statements<double>(count));
    EXPECT(updates_match_the_scalar_statements<float>(count));
    EXPECT(updates_match_the_scalar_statements<std::complex<double>>(count));
  }
}

// Whole numbers small enough that every partial sum is exact: any order of summation gives the exact dot
// product, so the vector form (four accumulators, then the lanes, then the remainder) must equal it.
TEST(dot_product_sums_every_term)
{
  for (const std::size_t count : {std::size_t {0}, std::size_t {3}, std::size_t {64}, std::size_t {77}}) {
    std::vector<double>               x(count);
    std::vector<double>               y(count);
    std::vector<std::complex<double>> zx(count);
    std::vector<std::complex<double>> zy(count);
    double                            want {0.0};
    for (std::size_t i {0}; i < count; ++i) {
      x[i]   = static_cast<double>(i % 7) - 3.0;
      y[i]   = static_cast<double>(i % 5) + 1.0;
      zx[i]  = {x[i], 1.0};
      zy[i]  = {y[i], 0.0};
      want  += x[i] * y[i];
    }
    EXPECT(numerical::detail::dot_product(x.data(), y.data(), count) == want);
    EXPECT(numerical::detail::dot_product(zx.data(), zy.data(), count).real() == want);
  }
}
