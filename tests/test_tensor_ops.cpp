// Element-wise arithmetic with broadcasting, reductions, reshaping and creation of tensors, each against
// NumPy's definition of it.
#include <algorithm>
#include <functional>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::tensor;
  using shape = std::vector<std::size_t>;

  tensor<double> counting(shape s)
  {
    std::size_t count {1};
    for (const std::size_t extent : s) {
      count *= extent;
    }
    numerical::buffer<double> values(count, 0.0);
    for (std::size_t i {0}; i < count; ++i) {
      values[i] = static_cast<double>(i);
    }
    return tensor<double> {std::move(s), std::move(values)};
  }
} // namespace

// A mean divides by its count in the wider of its type and double, the count exact: a float sum of 2^24 + 1 ones is
// 2^24, and divided by 2^24 + 1 it is NumPy's 0.99999994, where the count rounded to a float gave 1. Each part of a
// complex number so; a long double stays one; an integer divides by its count.
TEST(means_divide_by_an_exact_count)
{
  const std::size_t n        {(std::size_t {1} << 24) + 1};
  const float       expected {static_cast<float>(16777216.0 / 16777217.0)};
  EXPECT(numerical::detail::divided_by_count(16777216.0F, n) == expected);
  EXPECT(expected != 1.0F);
  EXPECT((numerical::detail::divided_by_count(std::complex<float> {16777216.0F, 33554432.0F}, n)
          == std::complex<float> {expected, static_cast<float>(33554432.0 / 16777217.0)}));
  EXPECT(numerical::detail::divided_by_count(3.0, 4) == 0.75);
  EXPECT((numerical::detail::divided_by_count(std::complex<double> {3.0, -1.0}, 4) == std::complex<double> {0.75, -0.25}));
  EXPECT(numerical::detail::divided_by_count(1.0L, 3) == 1.0L / 3.0L);
  EXPECT(numerical::detail::divided_by_count(7, 2) == 3);
  const numerical::tensor<int> no_ints {{2, 0, 3}, numerical::buffer<int> {}};          // no NaN to give: refused
  EXPECT_THROWS((void)numerical::mean(no_ints, 1), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::mean(no_ints), numerical::dimension_error);
  const numerical::tensor<double> no_doubles {{2, 0, 3}, numerical::buffer<double> {}}; // NaNs, as NumPy's
  EXPECT(std::isnan(numerical::mean(no_doubles, 1).data()[0]) && std::isnan(numerical::mean(no_doubles)));
  const numerical::tensor<float> twos        {{3, 2}, numerical::buffer<float> {2.0F, 4.0F, 6.0F, 8.0F, 10.0F, 12.0F}};
  EXPECT(numerical::mean(twos) == 7.0F);
  EXPECT((numerical::mean(twos, 0).data() == numerical::buffer<float> {6.0F, 8.0F}));
}

// max gives +0 and min −0 wherever both zeros are among the elements, whatever their order, as NumPy's: no zero is
// larger than the other, so the first met stayed. On a short line, a long one (the vectors' lanes) and along an
// outer axis (slabs compared element by element); a NaN still wins.
namespace {
  template <typename T>
  bool zeros_settle()
  {
    const T zero     {0};
    const T negative {-zero};
    bool    ok       {true};
    for (const std::size_t count : {std::size_t {2}, std::size_t {64}}) {
      for (const bool positive_first : {true, false}) {
        numerical::buffer<T> values(count);
        for (std::size_t i {0}; i < count; ++i) {
          values[i] = (i < count / 2) == positive_first ? zero : negative;
        }
        const numerical::tensor<T> line {{count}, values};
        ok = ok && !std::signbit(numerical::max(line)) && std::signbit(numerical::min(line));
        const numerical::tensor<T> rows {{count, 2}, [&] {
            numerical::buffer<T> pairs(2 * count);
            for (std::size_t i {0}; i < count; ++i) {
              pairs[2 * i]       = values[i];
              pairs[(2 * i) + 1] = T {1};
            }
            return pairs;
          }()};
        ok = ok && !std::signbit(numerical::max(rows, 0).data()[0]) && std::signbit(numerical::min(rows, 0).data()[0]);
        ok = ok && !std::signbit(numerical::max(line, 0).data()[0]);
      }
    }
    const numerical::tensor<T> only_negative {{3}, numerical::buffer<T> {negative, negative, T {-1}}};
    ok = ok && std::signbit(numerical::max(only_negative)) && numerical::max(only_negative) == zero;
    const numerical::tensor<T> with_nan      {{3}, numerical::buffer<T> {zero, std::numeric_limits<T>::quiet_NaN(), negative}};
    return ok && std::isnan(numerical::max(with_nan)) && std::isnan(numerical::min(with_nan));
  }
} // namespace

TEST(extremes_settle_a_zero_s_sign_as_numpy)
{
  EXPECT(zeros_settle<double>());
  EXPECT(zeros_settle<float>());
}

// Element-wise operations past a task's elements run by tasks: a same-shape pair by chunks, a broadcast one by rows,
// each task's odometer set from its first row (a row starting partway through the outer axes), the last task partial.
// Each element op(a, b) of its operands, as the plain loop gives it, on one thread as on all.
TEST(element_wise_operations_by_tasks_are_each_element_s_op)
{
  using numerical::tensor;
  const auto filled = [](std::vector<std::size_t> shape, double seed) {
                        std::size_t count {1};
                        for (const std::size_t extent : shape) {
                          count *= extent;
                        }
                        numerical::buffer<double> values(count);
                        for (std::size_t i {0}; i < count; ++i) {
                          values[i] = std::sin(seed + static_cast<double>(i));
                        }
                        return tensor<double> {std::move(shape), std::move(values)};
                      };
  const tensor<double> a      {filled({7, 301, 157}, 0.5)};
  const tensor<double> same   {filled({7, 301, 157}, 1.5)};
  const tensor<double> column {filled({7, 301, 1}, 2.5)};
  const tensor<double> row    {filled({1, 157}, 3.5)};
  const tensor<double> plane  {filled({301, 157}, 4.5)};
  for (const std::size_t threads : {std::size_t {1}, std::size_t {0}}) {
    numerical::set_max_threads(threads);
    const tensor<double> sum       {a + same};
    const tensor<double> by_column {a * column};
    const tensor<double> by_row    {a - row};
    const tensor<double> by_plane  {plane / a};
    bool                 each      {true};
    for (std::size_t p {0}; p < 7; ++p) {
      for (std::size_t r {0}; r < 301; ++r) {
        for (std::size_t c {0}; c < 157; ++c) {
          const std::size_t k {(((p * 301) + r) * 157) + c};
          each = each && sum.data()[k] == a.data()[k] + same.data()[k];
          each = each && by_column.data()[k] == a.data()[k] * column.data()[(p * 301) + r];
          each = each && by_row.data()[k] == a.data()[k] - row.data()[c];
          each = each && by_plane.data()[k] == plane.data()[(r * 157) + c] / a.data()[k];
        }
      }
    }
    EXPECT(each);
  }
  numerical::set_max_threads(0);
}

// A tensor holding 0 elements along one axis may hold any number along the others: a reduction keeping axes whose
// extents multiply past std::size_t refuses, where a wrapped size gave a buffer too small, written past (its end
// at address 0); one keeping an empty axis is empty, whatever the others. In each type, int among them.
namespace {
  template <typename T>
  bool huge_reductions_refuse_or_are_empty()
  {
    using numerical::tensor;
    using extents = std::vector<std::size_t>;
    const std::size_t big    {std::size_t {1} << 32};
    const tensor<T>   middle {{big, 0, big}, numerical::buffer<T> {}};
    bool              ok     {true};
    for (const auto& refused : {std::function<void()> {[&] { (void)numerical::sum(middle, 1); }},
                                std::function<void()> {[&] { (void)numerical::mean(middle, 1); }},
                                std::function<void()> {[&] { (void)numerical::einsum("ijk->ik", middle); }}}) {
      try {
        refused();
        ok = false;
      }
      catch (const numerical::dimension_error&) {}
    }
    ok = ok && numerical::sum(middle, 0).shape() == extents {0, big} && numerical::sum(middle, 0).data().empty();
    ok = ok && numerical::sum(middle, 2).shape() == extents {big, 0};
    const std::size_t huge {std::size_t {1} << 40};
    const tensor<T>   last {{huge, huge, 3, 0}, numerical::buffer<T> {}};
    ok = ok && numerical::sum(last, 2).shape() == extents {huge, huge, 0} && numerical::mean(last, 2).shape() == extents {huge, huge, 0};
    ok = ok && numerical::einsum("ijkl->ijl", last).shape() == extents {huge, huge, 0} && numerical::einsum("ijkl->ijl", last).data().empty();
    const tensor<T> diagonal {{0, 0, huge}, numerical::buffer<T> {}}; // einsum's general loop
    ok = ok && numerical::einsum("iij->ij", diagonal).shape() == extents {0, huge};
    // A contraction routed to the matrix product: empty where the result is, refused where it is past std::size_t.
    const tensor<T> left {{huge, huge, 0}, numerical::buffer<T> {}};
    const tensor<T> none {{0, 0}, numerical::buffer<T> {}};
    const tensor<T> five {{0, 5}, numerical::buffer<T> {}};
    ok = ok && numerical::einsum("ijk,kl->ijl", left, none).shape() == extents {huge, huge, 0};
    try {
      (void)numerical::einsum("ijk,kl->ijl", left, five);
      ok = false;
    }
    catch (const numerical::dimension_error&) {}
    return ok;
  }
} // namespace

TEST(reductions_keeping_huge_extents_refuse_or_are_empty)
{
  EXPECT(huge_reductions_refuse_or_are_empty<double>());
  EXPECT(huge_reductions_refuse_or_are_empty<float>());
  EXPECT(huge_reductions_refuse_or_are_empty<std::complex<double>>());
  EXPECT(huge_reductions_refuse_or_are_empty<std::complex<float>>());
  EXPECT(huge_reductions_refuse_or_are_empty<int>());
}

// Products and broadcasts whose result's extents multiply past std::size_t refuse before writing; an empty one is
// empty. arange refuses a length past std::size_t (or an infinite span), where the conversion was undefined.
TEST(products_broadcasts_and_ranges_past_size_t_refuse)
{
  const std::size_t               big  {std::size_t {1} << 33};
  const numerical::tensor<double> tall {{big, 0}, numerical::buffer<double> {}};
  const numerical::tensor<double> wide {{0, big}, numerical::buffer<double> {}};
  EXPECT_THROWS((void)numerical::matmul(tall, wide), numerical::dimension_error);
  EXPECT((numerical::matmul(wide, tall).shape() == std::vector<std::size_t> {0, 0}));
  const numerical::tensor<double> left  {{big, 1, 0}, numerical::buffer<double> {}};
  const numerical::tensor<double> right {{1, big, 0}, numerical::buffer<double> {}};
  const numerical::tensor<double> both  {left + right};
  EXPECT((both.shape() == std::vector<std::size_t> {big, big, 0}));
  EXPECT(both.data().empty());
  EXPECT_THROWS((void)numerical::arange(0.0, 1e20, 1.0), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::arange(-1e308, 1e308, 1.0), numerical::dimension_error);
  EXPECT(numerical::arange(0.0, 3.0, 1.0).size() == 3);
}

TEST(broadcast_shape_aligns_trailing_axes)
{
  EXPECT(numerical::broadcast_shape({3, 1}, {1, 4}) == (shape {3, 4}));
  EXPECT(numerical::broadcast_shape({2, 3}, {3}) == (shape {2, 3}));
  EXPECT(numerical::broadcast_shape({}, {2, 2}) == (shape {2, 2}));
  EXPECT(numerical::broadcast_shape({5, 1, 2}, {4, 1}) == (shape {5, 4, 2}));
  EXPECT_THROWS((void)numerical::broadcast_shape({2, 3}, {2}), numerical::dimension_error);
  try {
    (void)numerical::broadcast_shape({2, 3}, {4});
  }
  catch (const numerical::dimension_error& e) {
    EXPECT(std::string {e.what()} == "operands could not be broadcast together with shapes (2, 3) (4,)");
  }
}

TEST(arithmetic_broadcasts_as_numpy)
{
  const tensor<double> column {{3, 1}, {1.0, 2.0, 3.0}};
  const tensor<double> row    {{1, 4}, {10.0, 20.0, 30.0, 40.0}};
  const tensor<double> sum    {column + row};
  EXPECT(sum.shape() == (shape {3, 4}));
  EXPECT(sum(std::vector<std::size_t> {2, 3}) == 43.0);
  const tensor<double> product {column * row};
  EXPECT(product(std::vector<std::size_t> {1, 2}) == 60.0);
  EXPECT((row - column)(std::vector<std::size_t> {0, 0}) == 9.0);
  EXPECT((row / column)(std::vector<std::size_t> {1, 3}) == 20.0);
  const tensor<double> scalar {{}, {2.0}};
  EXPECT((counting({2, 3}) * scalar) == (tensor<double> {{2, 3}, {0.0, 2.0, 4.0, 6.0, 8.0, 10.0}}));
  EXPECT((counting({2, 3}) + counting({2, 3})) == (counting({2, 3}) * scalar)); // the same-shape path
  EXPECT(-counting({2}) == (tensor<double> {{2}, {-0.0, -1.0}}));
  EXPECT_THROWS((void)(counting({2, 3}) + counting({2})), numerical::dimension_error);
  const tensor<std::complex<double>> z {{2}, {{1.0, 1.0}, {0.0, 2.0}}};
  EXPECT((z * z) == (tensor<std::complex<double>> {{2}, {{0.0, 2.0}, {-4.0, 0.0}}}));
}

TEST(sum_is_pairwise_and_reduces_along_an_axis)
{
  // 0.1 a million times: summed in order the error grows with the count; pairwise it stays near rounding.
  const tensor<double> tenths {numerical::full<double>({1000000}, 0.1)};
  double               naive  {0.0};
  for (std::size_t i {0}; i < tenths.size(); ++i) {
    naive += 0.1;
  }
  const double pairwise {numerical::sum(tenths)};
  EXPECT(std::abs(pairwise - 100000.0) < 1e-6);
  EXPECT(std::abs(naive - 100000.0) > 1e-6);

  EXPECT(numerical::sum(counting({13})) == 78.0);           // eight partial sums, then a remainder of five in order
  const tensor<double> cube      {counting({2, 3, 4})};
  EXPECT(numerical::sum(cube) == 276.0);
  const tensor<double> over_rows {numerical::sum(cube, 1)}; // (2, 4)
  EXPECT(over_rows.shape() == (shape {2, 4}));
  EXPECT(over_rows(std::vector<std::size_t> {1, 2}) == 14.0 + 18.0 + 22.0);
  EXPECT(numerical::sum(cube, 0)(std::vector<std::size_t> {2, 3}) == 11.0 + 23.0);
  EXPECT(numerical::sum(cube, 2)(std::vector<std::size_t> {0, 1}) == 4.0 + 5.0 + 6.0 + 7.0);
  EXPECT(numerical::mean(cube) == 11.5);
  EXPECT(numerical::mean(cube, 2)(std::vector<std::size_t> {1, 0}) == 13.5);
  EXPECT(std::isnan(numerical::mean(numerical::zeros<double>({0}))));
  EXPECT(numerical::sum(numerical::zeros<double>({0})) == 0.0);
  EXPECT_THROWS((void)numerical::sum(cube, 3), numerical::dimension_error);
}

TEST(max_and_min_propagate_nan_and_refuse_an_empty_reduction)
{
  const tensor<double> cube {counting({2, 3, 4})};
  EXPECT(numerical::max(cube) == 23.0);
  EXPECT(numerical::min(cube) == 0.0);
  EXPECT(numerical::max(cube, 1)(std::vector<std::size_t> {0, 3}) == 11.0);
  EXPECT(numerical::min(cube, 0)(std::vector<std::size_t> {2, 1}) == 9.0);
  const tensor<double> unsorted {{3}, {3.0, 1.0, 2.0}};
  EXPECT(numerical::max(unsorted) == 3.0);
  EXPECT(numerical::min(unsorted) == 1.0);
  const double         nan      {std::numeric_limits<double>::quiet_NaN()};
  const tensor<double> with_nan {{4}, {1.0, nan, 3.0, 2.0}};
  EXPECT(std::isnan(numerical::max(with_nan)));
  EXPECT(std::isnan(numerical::min(with_nan)));
  const tensor<double> nan_first {{3}, {nan, 5.0, 1.0}};
  EXPECT(std::isnan(numerical::max(nan_first)));
  const tensor<int> integers     {{3}, {4, -2, 9}};
  EXPECT(numerical::max(integers) == 9);
  EXPECT(numerical::min(integers) == -2);
  EXPECT_THROWS((void)numerical::max(numerical::zeros<double>({0})), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::min(numerical::zeros<double>({2, 0}), 1), numerical::dimension_error);
}

TEST(reshape_keeps_the_order_and_checks_the_count)
{
  const tensor<double> line {counting({6})};
  const tensor<double> grid {numerical::reshape(line, {2, 3})};
  EXPECT(grid.shape() == (shape {2, 3}));
  EXPECT(grid(std::vector<std::size_t> {1, 0}) == 3.0);
  EXPECT_THROWS((void)numerical::reshape(line, {4}), numerical::dimension_error);
}

TEST(creation_routines)
{
  EXPECT(numerical::zeros<double>({2, 2}) == (tensor<double> {{2, 2}, {0.0, 0.0, 0.0, 0.0}}));
  EXPECT(numerical::ones<double>({3}) == (tensor<double> {{3}, {1.0, 1.0, 1.0}}));
  EXPECT(numerical::full<double>({}, 7.0) == (tensor<double> {{}, {7.0}}));
  EXPECT(numerical::eye<double>(2) == (tensor<double> {{2, 2}, {1.0, 0.0, 0.0, 1.0}}));
  EXPECT(numerical::arange(0.0, 1.0, 0.25) == (tensor<double> {{4}, {0.0, 0.25, 0.5, 0.75}}));
  EXPECT(numerical::arange(3.0, 0.0, -1.0) == (tensor<double> {{3}, {3.0, 2.0, 1.0}}));
  // The step is the one the first two elements hold, (start + step) − start, and each element is rounded
  // once: 0.3 from −3.5 differs from i·0.3 + (−3.5) rounded twice at several elements.
  const tensor<double> stepped {numerical::arange(-3.5, 7.2, 0.3)};
  const double         delta   {(-3.5 + 0.3) - -3.5};
  bool                 fused   {stepped.size() == 36U && stepped.data()[1] == -3.5 + 0.3};
  bool                 twice   {true};
  for (std::size_t i {2}; i < stepped.size(); ++i) {
    fused = fused && stepped.data()[i] == std::fma(static_cast<double>(i), delta, -3.5);
    const double product {static_cast<double>(i) * 0.3};
    twice = twice && stepped.data()[i] == -3.5 + product;
  }
  EXPECT(fused);
  EXPECT(!twice); // the case tells the two roundings apart
  EXPECT(numerical::arange(1.0, 1.0, 1.0).size() == 0U);
  EXPECT(numerical::arange(2.0, 1.0, 1.0).size() == 0U);
  EXPECT_THROWS((void)numerical::arange(0.0, 1.0, 0.0), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::arange(0.0, std::numeric_limits<double>::infinity(), 1.0), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::arange(0.0, 1.0, std::numeric_limits<double>::infinity()), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::arange(std::numeric_limits<double>::quiet_NaN(), 1.0, 1.0), numerical::dimension_error);
  const tensor<double> five {numerical::linspace(0.0, 1.0, 5)};
  EXPECT(five == (tensor<double> {{5}, {0.0, 0.25, 0.5, 0.75, 1.0}}));
  EXPECT(numerical::linspace(0.0, 0.3, 7).data().back() == 0.3);
  EXPECT(numerical::linspace(2.0, 3.0, 1) == (tensor<double> {{1}, {2.0}}));
  EXPECT(numerical::linspace(2.0, 3.0, 0).size() == 0U);
}

TEST(map_and_pow)
{
  const tensor<double> t               {{3}, {1.0, 4.0, 9.0}};
  EXPECT(numerical::map(t, [](double x) { return std::sqrt(x); }) == (tensor<double> {{3}, {1.0, 2.0, 3.0}}));
  const tensor<std::complex<double>> z {{2}, {{3.0, 4.0}, {0.0, -2.0}}};
  EXPECT(numerical::map(z, [](const std::complex<double>& x) { return std::abs(x); }) == (tensor<double> {{2}, {5.0, 2.0}}));
  EXPECT(numerical::pow(t, tensor<double> {{}, {2.0}}) == (tensor<double> {{3}, {1.0, 16.0, 81.0}}));
}

namespace {
  // A thousand values (vector lanes then a scalar tail on every build), the best and a NaN placed in turn at
  // the start, inside the vector region and in the tail, against a plain scan.
  template <typename T>
  bool contiguous_extremes_hold()
  {
    const T nan {std::numeric_limits<T>::quiet_NaN()};
    const T far {std::numeric_limits<T>::max() / T(2)};
    bool    ok  {true};
    for (const std::size_t count : {1U, 7U, 31U, 33U, 1000U, 1003U}) {
      std::vector<T> values(count);
      for (std::size_t i {0}; i < count; ++i) {
        values[i] = static_cast<T>(std::sin(static_cast<double>(i) * 1.7) * 10.0);
      }
      const tensor<T> plain {{count}, values};
      ok = ok && numerical::max(plain) == *std::ranges::max_element(values);
      ok = ok && numerical::min(plain) == *std::ranges::min_element(values);
      for (const std::size_t at : {std::size_t {0}, count / 2, count - 1}) {
        std::vector<T> peak {values};
        peak[at] = far;
        ok       = ok && numerical::max(tensor<T> {{count}, peak}) == far;
        peak[at] = -far;
        ok       = ok && numerical::min(tensor<T> {{count}, peak}) == -far;
        peak[at] = nan;
        ok       = ok && std::isnan(numerical::max(tensor<T> {{count}, peak}));
        ok       = ok && std::isnan(numerical::min(tensor<T> {{count}, peak}));
      }
    }
    return ok;
  }
} // namespace

TEST(a_contiguous_extreme_finds_nan_and_the_best_in_every_lane_and_the_tail)
{
  EXPECT(contiguous_extremes_hold<double>());
  EXPECT(contiguous_extremes_hold<float>());
}

TEST(extremes_along_an_outer_axis_compare_slab_by_slab)
{
  // (3, 4, 5) along each axis against the line-by-line reduction, a NaN in one line only.
  std::vector<double> values(60);
  for (std::size_t i {0}; i < values.size(); ++i) {
    values[i] = std::cos(static_cast<double>(i) * 0.9);
  }
  values[23] = std::numeric_limits<double>::quiet_NaN(); // index (1, 0, 3)
  const tensor<double> cube {{3, 4, 5}, values};
  for (std::size_t axis {0}; axis < 3; ++axis) {
    const tensor<double> got  {numerical::max(cube, axis)};
    const tensor<double> want {numerical::detail::reduce_along(cube, axis, [](const double* first, std::size_t count, std::size_t stride) {
                                                                 return numerical::detail::extreme(first, count, stride, std::greater<double> {});
                                                               })};
    bool same                 {got.shape() == want.shape()};
    for (std::size_t i {0}; same && i < got.size(); ++i) {
      same = (got.data()[i] == want.data()[i]) || (std::isnan(got.data()[i]) && std::isnan(want.data()[i]));
    }
    EXPECT(same);
  }
  EXPECT(std::isnan(numerical::min(cube, 0)(std::vector<std::size_t> {0, 3})));
  EXPECT(!std::isnan(numerical::min(cube, 0)(std::vector<std::size_t> {0, 2})));
  EXPECT_THROWS((void)numerical::max(numerical::zeros<double>({0, 3}), 0), numerical::dimension_error);
  // An empty axis is refused even when the result would be empty too, as NumPy refuses it.
  EXPECT_THROWS((void)numerical::min(numerical::zeros<double>({3, 0, 0}), 1), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::max(numerical::zeros<double>({0, 0}), 0), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::max(numerical::zeros<int>({2, 0}), 1), numerical::dimension_error);
}

TEST(a_scalar_exponent_of_two_a_half_or_minus_one_is_computed_as_numpy_does)
{
  const tensor<double> x   {{4}, {0.3, 2.0, 7.25, 1e-3}};
  const tensor<double> two {{}, {2.0}};
  bool                 ok  {true};
  for (std::size_t i {0}; i < x.size(); ++i) {
    const double v {x.data()[i]};
    ok = ok && numerical::pow(x, two).data()[i] == v * v;
    ok = ok && numerical::pow(x, tensor<double> {{}, {0.5}}).data()[i] == std::sqrt(v);
    ok = ok && numerical::pow(x, tensor<double> {{}, {-1.0}}).data()[i] == 1.0 / v;
    ok = ok && numerical::pow(x, tensor<double> {{}, {3.0}}).data()[i] == std::pow(v, 3.0);
  }
  EXPECT(ok);
  using cplx = std::complex<double>;
  const tensor<cplx> z {{2}, {cplx {1.0, 1.0}, cplx {3.0, -2.0}}};
  EXPECT(numerical::pow(z, tensor<cplx> {{}, {cplx {2.0, 0.0}}}) == (tensor<cplx> {{2}, {cplx {0.0, 2.0}, cplx {5.0, -12.0}}}));
  // An exponent per element is a power per element.
  EXPECT(numerical::pow(x, tensor<double> {{4}, {2.0, 2.0, 0.5, -1.0}})
         == (tensor<double> {{4}, {std::pow(0.3, 2.0), std::pow(2.0, 2.0), std::pow(7.25, 0.5), std::pow(1e-3, -1.0)}}));
  // An exponent of higher rank broadcasts the result to it: not the element-wise shortcut.
  EXPECT(numerical::pow(x, tensor<double> {{1, 1}, {2.0}}).shape() == (shape {1, 4}));
  const tensor<int> n {{2}, {3, 4}};
  EXPECT(numerical::pow(n, tensor<int> {{}, {0}}) == (tensor<int> {{2}, {1, 1}})); // an integer power keeps pow
}

TEST(broadcasting_steps_each_operand_or_holds_it_along_the_inner_axis)
{
  // Every way through the inner loop with one operator (coverage counts each operator's instantiation apart):
  // both stepping, the right held, the left held, both held, an empty result.
  const tensor<double> row4 {counting({4})};
  EXPECT((counting({2, 4}) + row4) == (tensor<double> {{2, 4}, {0.0, 2.0, 4.0, 6.0, 4.0, 6.0, 8.0, 10.0}}));
  EXPECT((counting({2, 1}) + row4) == (tensor<double> {{2, 4}, {0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0}}));
  EXPECT((row4 + counting({2, 1})) == (tensor<double> {{2, 4}, {0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0}}));
  EXPECT((counting({2, 1}) + counting({1, 1})) == (tensor<double> {{2, 1}, {0.0, 1.0}}));
  EXPECT((numerical::zeros<double>({0, 4}) + row4).size() == 0U);
  const tensor<double> column {{3, 1}, {1.0, 2.0, 3.0}};
  const tensor<double> one    {{1, 1}, {10.0}};
  EXPECT((column + one) == (tensor<double> {{3, 1}, {11.0, 12.0, 13.0}}));           // both held: a filled run
  EXPECT((one - counting({4})) == (tensor<double> {{1, 4}, {10.0, 9.0, 8.0, 7.0}})); // the left held
  EXPECT((counting({2, 3}) * tensor<double> {{3}, {1.0, 0.0, 2.0}})
         == (tensor<double> {{2, 3}, {0.0, 0.0, 4.0, 3.0, 0.0, 10.0}}));             // both stepping
  const tensor<double> cube {numerical::zeros<double>({2, 1, 1}) + counting({1, 3, 1})};
  EXPECT(cube.shape() == (shape {2, 3, 1}));
  EXPECT(cube(std::vector<std::size_t> {1, 2, 0}) == 2.0);
  EXPECT((numerical::zeros<double>({0, 3}) + counting({3})).size() == 0U);         // an empty result
}
