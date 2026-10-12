// The row kernels the decompositions share: the scaled-row updates (simd.hpp) equal, to the bit, the scalar
// statements, over native vectors or not; the dot product sums every term; recorded row operations applied by
// runs equal them applied one by one.
#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
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

// A symmetric product's row: the sum of every term and the scaled row added to out, in one pass, on every
// kernel this processor runs. Small integers: every order sums them exactly.
TEST(symmetric_row_sums_every_term_and_adds_the_scaled_row)
{
  using kernel = double (*)(const double*, const double*, double, double*, std::size_t);
  std::vector<kernel> kernels {[](const double* row, const double* v, double f, double* out, std::size_t count) {
                                 return numerical::detail::symmetric_row(row, v, f, out, count);
                               }};
#ifdef NUMERICAL_X86_COMPILED
  if (numerical::detail::cpu_features().avx2_fma) {
    kernels.push_back(&numerical::detail::avx2_symmetric_row);
  }
  if (numerical::detail::cpu_features().avx512f) {
    kernels.push_back(&numerical::detail::avx512_symmetric_row);
  }
#endif
  for (const kernel run : kernels) {
    for (const std::size_t count : {std::size_t {0}, std::size_t {3}, std::size_t {64}, std::size_t {77}, std::size_t {130}}) {
      std::vector<double> row(count);
      std::vector<double> v(count);
      std::vector<double> out(count);
      double              want {0.0};
      for (std::size_t i {0}; i < count; ++i) {
        row[i]  = static_cast<double>(i % 7) - 3.0;
        v[i]    = static_cast<double>(i % 5) + 1.0;
        out[i]  = static_cast<double>(i % 3);
        want   += row[i] * v[i];
      }
      EXPECT(run(row.data(), v.data(), 2.0, out.data(), count) == want);
      bool added {true};
      for (std::size_t i {0}; i < count; ++i) {
        added = added && out[i] == static_cast<double>(i % 3) + (2.0 * row[i]);
      }
      EXPECT(added);
    }
  }
  std::vector<std::complex<double>> z {{1.0, 2.0}, {3.0, -1.0}};
  std::vector<std::complex<double>> zout(2);
  EXPECT(numerical::detail::symmetric_row(z.data(), z.data(), std::complex<double> {2.0, 0.0}, zout.data(), 2)
         == (z[0] * z[0]) + (z[1] * z[1]));
  EXPECT(zout[1] == z[1] * 2.0);
}

namespace {

  using numerical::detail::row_operation;

  row_operation rotation(std::size_t i,
                         std::size_t j,
                         double      angle)
  {
    return {.what = row_operation::kind::rotate, .i = i, .j = j, .c = std::cos(angle), .s = std::sin(angle)};
  }

  // The operations one at a time, each on whole rows: the definition the runs must reproduce to the bit.
  std::vector<double> applied_one_by_one(std::vector<double>               m,
                                         std::size_t                       n,
                                         const std::vector<row_operation>& operations)
  {
    for (const row_operation& op : operations) {
      for (std::size_t k {0}; k < n; ++k) {
        double& a {m[(op.i * n) + k]};
        double& b {m[(op.j * n) + k]};
        if (op.what == row_operation::kind::rotate) {
          const double x {a};
          const double y {b};
          a = (op.c * x) + (op.s * y);
          b = (op.c * y) - (op.s * x);
        }
        else if (op.what == row_operation::kind::negate) {
          a = -a;
        }
        else {
          std::swap(a, b);
        }
      }
    }
    return m;
  }
} // namespace

// Chained rotations going down (as a QL sweep records them) or up (as a QR sweep), a run ending the list, and
// operations that chain to nothing: split into runs, and applied by runs exactly as one by one.
TEST(rotation_runs_apply_as_the_operations_one_by_one)
{
  const std::vector<row_operation> down {rotation(3, 4, 0.3), rotation(2, 3, -1.1), rotation(1, 2, 0.7)};
  const std::vector<row_operation> up   {rotation(0, 1, 0.2), rotation(1, 2, 2.1), rotation(2, 3, -0.4),
                                         {.what = row_operation::kind::negate, .i = 2, .j = 2, .c = 0.0, .s = 0.0},
                                         {.what = row_operation::kind::swap, .i = 0, .j = 4, .c = 0.0, .s = 0.0},
                                         rotation(0, 3, 0.9), rotation(1, 2, 0.5)};
  const auto down_runs {numerical::detail::rotation_runs(down)};
  const auto up_runs   {numerical::detail::rotation_runs(up)};
  EXPECT(down_runs.size() == 1U && down_runs[0].count == 3 && down_runs[0].down);
  EXPECT(up_runs.size() == 5U && up_runs[0].count == 3 && !up_runs[0].down);
  for (const std::size_t n : {std::size_t {5}, std::size_t {70}}) { // one chunk of columns, then two
    std::vector<double> m(n * n);
    for (std::size_t i {0}; i < m.size(); ++i) {
      m[i] = (1.0 / static_cast<double>(i + 3)) - 0.17;
    }
    for (const std::vector<row_operation>* operations : {&down, &up}) {
      std::vector<double> runs {m};
      numerical::detail::apply_row_operations(runs, n, *operations);
      EXPECT(runs == applied_one_by_one(m, n, *operations));
    }
  }
}
