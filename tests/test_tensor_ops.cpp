// Element-wise arithmetic with broadcasting, reductions, reshaping and creation of tensors, each against
// NumPy's definition of it.
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
    std::vector<double> values(count);
    for (std::size_t i {0}; i < count; ++i) {
      values[i] = static_cast<double>(i);
    }
    return tensor<double> {std::move(s), std::move(values)};
  }
} // namespace

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
