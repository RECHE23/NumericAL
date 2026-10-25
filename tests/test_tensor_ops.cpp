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

// einsum's general loop takes each term's first factor itself, then multiplies the others in: a complex 1 times an
// element was no identity (0 · inf a NaN, −0 − (−0) a +0). A complex trace keeps a finite imaginary part beside an
// infinite real one; a product of two operands through the same loop (a label repeated) is each term's product.
namespace {
  template <typename T>
  bool general_einsum_keeps_its_factors()
  {
    const auto value {[](double re, [[maybe_unused]] double im) {
                        if constexpr (numerical::detail::split_complex<T>) {
                          using R = T::value_type;
                          return T {static_cast<R>(re), static_cast<R>(im)};
                        }
                        else {
                          return static_cast<T>(re);
                        }
                      }};
    const numerical::tensor<T> square {{2, 2}, numerical::buffer<T> {value(4, 1), value(0, 0), value(0, 0), value(-1, -1)}};
    const numerical::tensor<T> twice  {{2}, numerical::buffer<T> {value(2, 0), value(3, 0)}};
    const T                    trace  {numerical::einsum("ii->", square).data()[0]};
    const numerical::tensor<T> rows   {numerical::einsum("ii,i->i", square, twice)};
    return trace == value(3, 0) && rows.data()[0] == value(8, 2) && rows.data()[1] == value(-3, -3);
  }
} // namespace

TEST(general_einsum_takes_each_first_factor_itself)
{
  using c = std::complex<double>;
  EXPECT(general_einsum_keeps_its_factors<double>());
  EXPECT(general_einsum_keeps_its_factors<float>());
  EXPECT(general_einsum_keeps_its_factors<c>());
  EXPECT(general_einsum_keeps_its_factors<std::complex<float>>());
  EXPECT(general_einsum_keeps_its_factors<int>());
  const double               inf      {std::numeric_limits<double>::infinity()};
  const numerical::tensor<c> infinite {{2, 2}, numerical::buffer<c> {c {inf, 1.0}, c {}, c {}, c {-0.0, -1.0}}};
  const c                    trace    {numerical::einsum("ii->", infinite).data()[0]};
  EXPECT(trace.real() == inf && trace.imag() == 0.0);
}

// NumPy's complex product and quotient, fused or not: no recovery of an infinity (1j · (inf + inf·j) is NaN either
// way, where std::complex's gives one), the fused product's overflow an infinity in both parts where the rounded one
// leaves a NaN; Smith's quotient exact on exact parts, a zero divisor divided part by part. The tensors' broadcast
// operations take this processor's choice.
namespace {
  template <typename C>
  bool numpy_complex_arithmetic_holds()
  {
    using R = C::value_type;
    const R    inf {std::numeric_limits<R>::infinity()};
    const R    big {std::numeric_limits<R>::max() / R {2}};
    const auto nan_both {[](const C& z) { return std::isnan(z.real()) && std::isnan(z.imag()); }};
    bool       ok {true};
    for (const bool fused : {false, true}) {
      ok = ok && nan_both(numerical::detail::numpy_multiply(C {R {0}, R {1}}, C {inf, inf}, fused));
      ok = ok && numerical::detail::numpy_multiply(C {R {3}, R {-2}}, C {R {0.5}, R {4}}, fused) == C {R {9.5}, R {11}};
      ok = ok && numerical::detail::numpy_divide(C {R {2}, R {4}}, C {R {1}, R {1}}, fused) == C {R {3}, R {1}};
      ok = ok && numerical::detail::numpy_divide(C {R {6}, R {3}}, C {R {0}, R {3}}, fused) == C {R {1}, R {-2}};
      ok = ok && numerical::detail::numpy_divide(C {R {1}, R {-1}}, C {R {0}, R {0}}, fused) == C {inf, -inf};
      ok = ok && nan_both(numerical::detail::numpy_divide(C {R {0}, R {0}}, C {R {0}, R {0}}, fused));
      const C nan_over_zero {numerical::detail::numpy_divide(C {std::numeric_limits<R>::quiet_NaN(), R {-2}}, C {R {0}, R {0}}, fused)};
      ok = ok && std::isnan(nan_over_zero.real()) && nan_over_zero.imag() == -inf;
    }
    const C wide                        {numerical::detail::numpy_multiply(C {big, big}, C {big, -big}, true)};
    const C rounded                     {numerical::detail::numpy_multiply(C {big, big}, C {big, -big}, false)};
    ok = ok && wide.real() == inf && wide.imag() == inf && rounded.real() == inf && std::isnan(rounded.imag());
    const numerical::tensor<C> a        {{2, 1}, numerical::buffer<C> {C {R {1}, R {2}}, C {R {-3}, R {0.5}}}};
    const numerical::tensor<C> b        {{3}, numerical::buffer<C> {C {R {2}, R {0}}, C {R {0}, R {-1}}, C {R {0.25}, R {4}}}};
    const numerical::tensor<C> product  {numerical::detail::numpy_multiply(a, b)};
    const numerical::tensor<C> quotient {numerical::detail::numpy_divide(a, b)};
    for (std::size_t i {0}; i < 2; ++i) {
      for (std::size_t j {0}; j < 3; ++j) {
        ok = ok && product.data()[(i * 3) + j]
             == numerical::detail::numpy_multiply(a.data()[i], b.data()[j], numerical::detail::numpy_fuses_products());
        ok = ok && quotient.data()[(i * 3) + j]
             == numerical::detail::numpy_divide(a.data()[i], b.data()[j], numerical::detail::numpy_wheel_loops().quotients);
      }
    }
    // One element from a broadcast operand of rank above 0 is NumPy's scalar loop's; from equal ranks or a rank-0
    // operand, its vector loop's. (1 + e)(1 - e) - 1 is -e² fused and 0 rounded.
    const R                    e           {std::numeric_limits<R>::epsilon() * R {4}};
    const C                    x           {R {1} + e, R {1}};
    const C                    y           {R {1} - e, R {1}};
    const C                    scalar_loop {numerical::detail::numpy_multiply(x, y, numerical::detail::numpy_fuses_scalar_products())};
    const C                    vector_loop {numerical::detail::numpy_multiply(x, y, numerical::detail::numpy_fuses_products())};
    const numerical::tensor<C> one_one     {{1, 1}, numerical::buffer<C> {x}};
    const numerical::tensor<C> other_one   {{1, 1}, numerical::buffer<C> {y}};
    const numerical::tensor<C> one         {{1}, numerical::buffer<C> {y}};
    const numerical::tensor<C> zero_rank   {std::vector<std::size_t> {}, numerical::buffer<C> {y}};
    const numerical::tensor<C> three       {{3}, numerical::buffer<C> {y, y, y}};
    ok = ok && numerical::detail::numpy_multiply(x, y, true) != numerical::detail::numpy_multiply(x, y, false);
    ok = ok && numerical::detail::numpy_multiply(one_one, one).data()[0] == scalar_loop;
    ok = ok && numerical::detail::numpy_multiply(one, one_one).data()[0] == scalar_loop;
    ok = ok && numerical::detail::numpy_multiply(one_one, zero_rank).data()[0] == vector_loop;
    ok = ok && numerical::detail::numpy_multiply(one_one, other_one).data()[0] == vector_loop;
    ok = ok && numerical::detail::numpy_multiply(one_one, three).data()[2] == vector_loop;
    // The loops as measured beside a NumPy, whichever way each fuses.
    const C fused_product   {numerical::detail::numpy_multiply(x, y, true)};
    const C rounded_product {numerical::detail::numpy_multiply(x, y, false)};
    for (const bool scalar : {false, true}) {
      const numerical::detail::numpy_loops loops {.products = !scalar, .scalar_products = scalar, .quotients = scalar};
      ok = ok && numerical::detail::numpy_multiply(one_one, one, loops).data()[0] == (scalar ? fused_product : rounded_product);
      ok = ok && numerical::detail::numpy_multiply(one_one, other_one, loops).data()[0] == (scalar ? rounded_product : fused_product);
      ok = ok && numerical::detail::numpy_divide(one_one, other_one, loops).data()[0] == numerical::detail::numpy_divide(x, y, scalar);
    }
    return ok;
  }
} // namespace

TEST(complex_arithmetic_as_numpy_computes_it)
{
  EXPECT(numpy_complex_arithmetic_holds<std::complex<double>>());
  EXPECT(numpy_complex_arithmetic_holds<std::complex<float>>());
}

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
// The element-wise functions share a large tensor among the threads by chunks, a last one partial: each element is the
// sequential map's, on one thread and on all of them.
TEST(element_wise_functions_by_chunks_are_the_sequential_map_s)
{
  using cplx = std::complex<double>;
  const std::size_t          count {(3 * numerical::detail::elementwise_chunk) + 7};
  numerical::buffer<double>  reals(count);
  numerical::buffer<cplx>    complexes(count);
  for (std::size_t i {0}; i < count; ++i) {
    reals[i]     = 1.5 + std::sin(static_cast<double>(i)); // positive: each logarithm finite
    complexes[i] = cplx {reals[i], std::cos(static_cast<double>(i))};
  }
  const numerical::tensor<double> r       {{count}, std::move(reals)};
  const numerical::tensor<cplx>   z       {{count}, std::move(complexes)};
  const std::size_t               threads {numerical::max_threads()};
  for (const std::size_t n : {std::size_t {1}, threads}) {
    numerical::set_max_threads(n);
    EXPECT(numerical::exp(z) == numerical::map(z, [](const cplx& x) { return numerical::detail::complex_exp(x); }));
    EXPECT(numerical::log(r) == numerical::map(r, [](double x) { return std::log(x); }));
    EXPECT(numerical::tanh(z) == numerical::map(z, [](const cplx& x) { return numerical::detail::complex_tanh(x); }));
    EXPECT(-z == numerical::map(z, [](const cplx& x) { return -x; }));
    EXPECT(numerical::detail::numpy_absolute(z) == numerical::map(z, [](const cplx& x) { return numerical::detail::numpy_absolute(x, numerical::detail::numpy_wheel_loops().products); }));
  }
  numerical::set_max_threads(threads);
}

// NumPy's complex magnitude: an infinite part makes it infinite whatever the other, NaN, and a NaN part then NaN;
// a zero is 0; otherwise the larger part times sqrt(ratio² + 1).
namespace {
  template <typename R>
  bool numpy_magnitudes_hold()
  {
    using C = std::complex<R>;
    const R inf {std::numeric_limits<R>::infinity()};
    const R nan {std::numeric_limits<R>::quiet_NaN()};
    bool    ok  {true};
    for (const bool fused : {false, true}) {
      const auto magnitude {[fused](C z) { return numerical::detail::numpy_absolute(z, fused); }};
      ok = ok && magnitude(C {inf, nan}) == inf && magnitude(C {nan, -inf}) == inf && magnitude(C {R {1}, inf}) == inf;
      ok = ok && std::isnan(magnitude(C {nan, R {1}})) && std::isnan(magnitude(C {R {2}, nan}));
      ok = ok && magnitude(C {R {0}, R {-0.0}}) == R {0} && magnitude(C {R {-3}, R {4}}) == R {5};
    }
    return ok;
  }
} // namespace

TEST(complex_magnitudes_as_numpy_computes_them)
{
  EXPECT(numpy_magnitudes_hold<double>());
  EXPECT(numpy_magnitudes_hold<float>());
}

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
    for (const std::size_t count : {1U, 7U, 16U, 17U, 31U, 32U, 33U, 64U, 65U, 1000U, 1003U}) {
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
        std::vector<T> zeros(count, T(-0.0)); // −0 everywhere but one +0: max's +0, min's −0, wherever it lies
        zeros[at] = T(0.0);
        ok        = ok && !std::signbit(numerical::max(tensor<T> {{count}, zeros}));
        ok        = ok && (count == 1 || std::signbit(numerical::min(tensor<T> {{count}, zeros})));
      }
    }
    return ok;
  }
} // namespace

// The extremes of contiguous values on every path this processor runs (the x86-64 ones in AVX2's wider vectors):
// the same answers, the best and a NaN anywhere, a zero's sign settled as NumPy settles it.
TEST(a_contiguous_extreme_finds_nan_and_the_best_in_every_lane_and_the_tail)
{
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue; // a path this processor does not run
    }
    EXPECT(contiguous_extremes_hold<double>());
    EXPECT(contiguous_extremes_hold<float>());
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
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
  // a complex one keeps the operators' Annex G, the square and the reciprocal alike (NumPy's are the numpy_ forms)
  const tensor<cplx> extreme {{2}, {cplx {1e308, 1e308}, cplx {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()}}};
  const tensor<cplx> squared {numerical::pow(extreme, tensor<cplx> {{}, {cplx {2.0, 0.0}}})};
  const tensor<cplx> inverted {numerical::pow(extreme, tensor<cplx> {{}, {cplx {-1.0, 0.0}}})};
  const auto         same_parts {[](cplx a, cplx b) {
                                   const auto part {[](double p, double q) { return (std::isnan(p) && std::isnan(q)) || p == q; }};
                                   return part(a.real(), b.real()) && part(a.imag(), b.imag());
                                 }};
  for (std::size_t i {0}; i < 2; ++i) {
    EXPECT(same_parts(squared.data()[i], extreme.data()[i] * extreme.data()[i]));
    EXPECT(same_parts(inverted.data()[i], cplx {1.0, 0.0} / extreme.data()[i]));
  }
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

namespace {
  // a + b and a − b broadcast against the sum and difference element by element (each output index's operands found by
  // its own coordinates).
  bool sum_is_each_elements(const shape& sa,
                            const shape& sb)
  {
    const auto values {[](const shape& s, double phase) {
                         std::size_t count {1};
                         for (const std::size_t e : s) {
                           count *= e;
                         }
                         std::vector<double> v(count);
                         for (std::size_t i {0}; i < count; ++i) {
                           v[i] = std::sin((static_cast<double>(i) * 0.3) + phase);
                         }
                         return tensor<double> {s, v};
                       }};
    const tensor<double> a          {values(sa, 0.1)};
    const tensor<double> b          {values(sb, 0.9)};
    const tensor<double> got        {a + b};
    const tensor<double> difference {a - b};
    const shape          out        {numerical::broadcast_shape(sa, sb)};
    const std::size_t    rank       {out.size()};
    bool                 same       {got.shape() == out};
    for (std::size_t flat {0}; same && flat < got.size(); ++flat) {
      std::size_t ia     {0};
      std::size_t ib     {0};
      std::size_t step_a {1};
      std::size_t step_b {1};
      std::size_t rest   {flat};
      for (std::size_t axis {rank}; axis-- > 0;) {
        const std::size_t index  {rest % out[axis]};
        rest /= out[axis];
        const std::size_t lead_a {rank - sa.size()};
        const std::size_t lead_b {rank - sb.size()};
        if (axis >= lead_a) {
          const std::size_t extent {sa[axis - lead_a]};
          ia     += (extent == 1 ? 0 : index) * step_a;
          step_a *= extent;
        }
        if (axis >= lead_b) {
          const std::size_t extent {sb[axis - lead_b]};
          ib     += (extent == 1 ? 0 : index) * step_b;
          step_b *= extent;
        }
      }
      same = got.data()[flat] == a.data()[ia] + b.data()[ib] && difference.data()[flat] == a.data()[ia] - b.data()[ib];
    }
    return same;
  }
} // namespace

// Broadcasting merges the axes both operands step through as one, and runs short rows of one operand repeated over the
// other's, or a column's elements spread over the other's rows, as one loop: every element the sum of its own operands,
// whichever operand repeats or spreads, rows of 1 to 32 elements, merged axes and not, more than one task.
TEST(broadcasting_gives_each_elements_sum)
{
  for (const auto& [sa, sb] : std::vector<std::pair<shape, shape>> {{{50000, 3}, {3}}, {{3}, {50000, 3}}, {{20000, 1}, {1}}, {{20000, 31}, {31}},
         {{20000, 32}, {32}}, {{6, 31}, {1, 31}}, {{1000, 4, 3}, {1, 4, 3}}, {{1, 4, 3}, {1000, 4, 3}},
         {{1000, 3}, {1000, 1}}, {{4, 100, 100}, {100, 100}}, {{2, 3, 4, 5}, {3, 1, 5}},
         {{3, 1, 7}, {1, 4, 1}}, {{5, 6}, {5, 1}}, {{1}, {9, 9}}, {{1, 3}, {3}}, {{50000, 3}, {50000, 1}},
         {{50000, 1}, {50000, 2}}, {{20000, 31}, {20000, 1}}, {{20000, 32}, {20000, 1}}, {{7, 5}, {7, 1}}, {{5000, 1}, {1, 6}}}) {
    EXPECT(sum_is_each_elements(sa, sb));
  }
}

namespace {
  // |root² − z| within a few rounding errors of |z|, the real part not negative, the imaginary part of z's sign.
  template <typename R>
  bool is_the_root(const std::complex<R>& root,
                   const std::complex<R>& z)
  {
    using wide = long double;
    const std::complex<wide> r      {root.real(), root.imag()};
    const std::complex<wide> square {(r.real() * r.real()) - (r.imag() * r.imag()), 2 * r.real() * r.imag()};
    const wide               error  {std::abs(square - std::complex<wide> {z.real(), z.imag()})};
    return error <= 8 * static_cast<wide>(std::numeric_limits<R>::epsilon()) * std::abs(std::complex<wide> {z.real(), z.imag()})
           && !std::signbit(root.real()) && std::signbit(root.imag()) == std::signbit(z.imag());
  }

  template <typename R>
  bool complex_roots_hold()
  {
    using C = std::complex<R>;
    const R    inf {std::numeric_limits<R>::infinity()};
    const R    nan {std::numeric_limits<R>::quiet_NaN()};
    const R    big {std::numeric_limits<R>::max()};
    const R    least {std::numeric_limits<R>::min()};
    const R    tiny {std::numeric_limits<R>::denorm_min()};
    bool       ok {true};
    const auto root {[](const C& z) { return numerical::detail::complex_sqrt(z); }};
    // exact ones: the real and the imaginary axes apart
    ok = ok && root(C {R {-1}, R {0}}) == C {R {0}, R {1}} && root(C {R {-4}, R {-0.0}}) == C {R {0}, R {-2}};
    ok = ok && root(C {R {4}, R {0}}) == C {R {2}, R {0}} && std::signbit(root(C {R {4}, R {-0.0}}).imag());
    ok = ok && root(C {R {0}, R {2}}) == C {R {1}, R {1}} && root(C {R {-0.0}, R {-8}}) == C {R {2}, R {-2}};
    ok = ok && root(C {R {0}, R {0}}) == C {R {0}, R {0}};
    // Annex G's infinities and NaNs
    ok = ok && root(C {nan, inf}) == C {inf, inf} && root(C {R {1}, -inf}) == C {inf, -inf};
    ok = ok && root(C {-inf, R {2}}) == C {R {0}, inf} && root(C {-inf, R {-2}}) == C {R {0}, -inf};
    ok = ok && std::isnan(root(C {-inf, nan}).real()) && std::isinf(root(C {-inf, nan}).imag());
    ok = ok && root(C {inf, R {-2}}) == C {inf, R {-0.0}} && std::signbit(root(C {inf, R {-2}}).imag());
    ok = ok && root(C {inf, nan}).real() == inf && std::isnan(root(C {inf, nan}).imag());
    ok = ok && std::isnan(root(C {nan, R {1}}).real()) && std::isnan(root(C {R {1}, nan}).imag());
    // each branch: either sign of the real part, scaled near overflow (by either part, the imaginary one small or
    // not, the real one negligible or not) and among subnormal parts, and an imaginary axis of subnormal values
    for (const C& z : {C {R {3}, R {4}}, C {R {-3}, R {4}}, C {static_cast<R>(-1e3), static_cast<R>(1e-3)}, C {big, R {0.5}}, C {-big, R {0.5}},
                       C {big, big}, C {-big, big / R {2}}, C {R {1}, big}, C {least, -big}, C {tiny, tiny},
                       C {-tiny * R {3}, tiny}, C {R {0}, tiny}, C {R {-0.0}, least}, C {tiny, R {1}}, C {-tiny, R {-2}}}) {
      ok = ok && is_the_root(root(z), z);
    }
    // the polar form's loss near the negative real axis: the real part is |y| / (2 √|x|)
    const C near {root(C {static_cast<R>(-1e20), R {1}})};
    ok = ok && std::abs(near.real() - static_cast<R>(5e-11)) < static_cast<R>(1e-15) && near.imag() == std::sqrt(static_cast<R>(1e20));
    // the tensor's square root and the exponent 0.5 take it
    const numerical::tensor<C> t    {{2}, numerical::buffer<C> {C {R {-1}, R {0}}, C {static_cast<R>(-1e20), R {1}}}};
    ok = ok && numerical::sqrt(t).data()[0] == C {R {0}, R {1}} && numerical::sqrt(t).data()[1] == near;
    const numerical::tensor<C> half {std::vector<std::size_t> {}, numerical::buffer<C> {C {R {0.5}, R {0}}}};
    ok = ok && numerical::pow(t, half).data()[1] == near;
    const numerical::tensor<R> real {{1}, numerical::buffer<R> {R {9}}};
    ok = ok && numerical::sqrt(real).data()[0] == R {3};
    return ok;
  }

  template <typename R>
  bool numpy_powers_hold()
  {
    using C = std::complex<R>;
    using numerical::detail::numpy_divide;
    using numerical::detail::numpy_multiply;
    using numerical::detail::numpy_power;
    using numerical::detail::numpy_reciprocal;
    const R nan  {std::numeric_limits<R>::quiet_NaN()};
    const C a    {R {1.25}, R {-0.75}};
    const C wide {R {3}, R {0.5}};
    bool    ok   {true};
    for (const bool fused : {false, true}) {
      const C square {numpy_multiply(a, a, fused)};
      ok = ok && numpy_power(a, C {R {0}, R {0}}, fused) == C {R {1}, R {0}};
      ok = ok && numpy_power(a, C {R {0}, R {1}}, fused).imag() != R {0}; // an exponent of i, by std::pow
      ok = ok && numpy_power(C {R {0}, R {2}}, C {R {2}, R {0}}, fused) == numpy_multiply(C {R {0}, R {2}}, C {R {0}, R {2}}, fused);
      ok = ok && numpy_power(C {R {0}, R {0}}, C {R {2}, R {1}}, fused) == C {R {0}, R {0}};
      const C nan_power {numpy_power(C {R {0}, R {-0.0}}, C {R {-1}, R {0}}, fused)};
      ok = ok && std::isnan(nan_power.real()) && std::isnan(nan_power.imag());
      ok = ok && numpy_power(a, C {R {1}, R {0}}, fused) == a && numpy_power(a, C {R {2}, R {0}}, fused) == square;
      ok = ok && numpy_power(a, C {R {3}, R {0}}, fused) == numpy_multiply(a, square, fused);
      // past the cubes, by the binary powers from 1; a negative exponent's quotient of 1 by them
      const C fourth {numpy_multiply(square, square, fused)};
      const C fifth  {numpy_multiply(numpy_multiply(C {R {1}, R {0}}, a, fused), fourth, fused)};
      ok = ok && numpy_power(a, C {R {5}, R {0}}, fused) == fifth;
      ok = ok && numpy_power(a, C {R {-1}, R {0}}, fused) == numpy_divide(C {R {1}, R {0}}, numpy_multiply(C {R {1}, R {0}}, a, fused), fused);
      ok = ok && numpy_power(a, C {R {-2}, R {0}}, fused) == numpy_divide(C {R {1}, R {0}}, numpy_multiply(C {R {1}, R {0}}, square, fused), fused);
      // any other exponent by std::pow: a fraction, one of 100 or more, an imaginary part
      // (read back through volatile parts: GCC folds std::pow of constants exactly, where a call rounds as the C
      // library's cpow does)
      for (const C& exponent : {C {R {0.5}, R {0}}, C {R {100}, R {0}}, C {R {-100}, R {0}}, C {R {2}, R {1}}, C {nan, R {0}}}) {
        volatile R real_part      {exponent.real()};
        volatile R imaginary_part {exponent.imag()};
        volatile R base_real      {a.real()};
        volatile R base_imaginary {a.imag()};
        const C    base           {static_cast<R>(base_real), static_cast<R>(base_imaginary)};
        const C    b              {static_cast<R>(real_part), static_cast<R>(imaginary_part)};
        const C    want           {std::pow(base, b)};
        const C    got            {numpy_power(base, b, fused)};
        ok = ok && ((got == want) || (std::isnan(got.real()) && std::isnan(want.real())));
      }
      // the reciprocal: the smaller part over the larger, either way round
      for (const C& x : {a, wide, C {R {0.5}, R {3}}}) {
        const C    r           {numpy_reciprocal(x, fused)};
        const bool real_larger {std::abs(x.imag()) <= std::abs(x.real())};
        const R    ratio       {real_larger ? x.imag() / x.real() : x.real() / x.imag()};
        const R    d           {real_larger ? numerical::detail::numpy_muladd(x.imag(), ratio, x.real(), fused)
                                  : numerical::detail::numpy_muladd(x.real(), ratio, x.imag(), fused)};
        ok = ok && r == (real_larger ? C {R {1} / d, -ratio / d} : C {ratio / d, R {-1} / d});
      }
    }
    // the tensors' forms, fused as the loops say
    const numerical::tensor<C>   t     {{2}, numerical::buffer<C> {a, wide}};
    const numerical::tensor<C>   three {std::vector<std::size_t> {}, numerical::buffer<C> {C {R {3}, R {0}}}};
    for (const bool fused : {false, true}) {
      const numerical::detail::numpy_loops loops {.products = fused, .scalar_products = !fused, .quotients = !fused};
      ok = ok && numerical::detail::numpy_square(t, loops).data()[1] == numpy_multiply(wide, wide, fused);
      ok = ok && numerical::detail::numpy_reciprocal(t, loops).data()[1] == numpy_reciprocal(wide, !fused);
      ok = ok && numerical::detail::numpy_power(t, three, loops).data()[0] == numpy_power(a, C {R {3}, R {0}}, !fused);
    }
    return ok;
  }
} // namespace

// The complex square root as glibc's csqrt (NumPy's on Linux): exact on the axes, Annex G's specials, each scaling.
TEST(complex_square_roots_are_glibc_s)
{
  EXPECT(complex_roots_hold<double>());
  EXPECT(complex_roots_hold<float>());
}

// x ** 2, x ** -1 and x ** n as NumPy's square, reciprocal and npy_cpow compute them.
TEST(complex_powers_as_numpy_computes_them)
{
  EXPECT(numpy_powers_hold<double>());
  EXPECT(numpy_powers_hold<float>());
}

// A permutation of three axes, each element where its labels say: the last two exchanged (by tiles, each slice of the
// first), and two that move the first (row by row of the last label).
TEST(einsum_permutes_three_axes)
{
  const std::size_t   a {3};
  const std::size_t   b {4};
  const std::size_t   c {5};
  std::vector<double> values(a * b * c);
  for (std::size_t n {0}; n < values.size(); ++n) {
    values[n] = static_cast<double>(n);
  }
  const numerical::tensor<double> t   {{a, b, c}, values};
  const auto                      ikj {numerical::einsum("ijk->ikj", t)};
  const auto                      kji {numerical::einsum("ijk->kji", t)};
  const auto                      jik {numerical::einsum("ijk->jik", t)};
  bool                            all {ikj.shape() == (std::vector<std::size_t> {a, c, b}) && kji.shape() == (std::vector<std::size_t> {c, b, a}) &&
                                       jik.shape() == (std::vector<std::size_t> {b, a, c})};
  for (std::size_t i {0}; i < a; ++i) {
    for (std::size_t j {0}; j < b; ++j) {
      for (std::size_t k {0}; k < c; ++k) {
        const double x {values[(((i * b) + j) * c) + k]};
        all = all && ikj.data()[(((i * c) + k) * b) + j] == x && kji.data()[(((k * b) + j) * a) + i] == x &&
              jik.data()[(((j * a) + i) * c) + k] == x;
      }
    }
  }
  EXPECT(all);
}
