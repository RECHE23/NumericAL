// Element-wise arithmetic with broadcasting, reductions, reshaping and creation of tensors, each against
// NumPy's definition of it.
#include <algorithm>
#include <array>
#include <bit>
#include <functional>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
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
      const numerical::detail::numpy_loops loops {.products = !scalar, .scalar_products = scalar, .quotients = scalar,
                                                  .reductions = scalar, .late_reductions = false};
      ok                                                      = ok && numerical::detail::numpy_multiply(one_one, one, loops).data()[0] == (scalar ? fused_product : rounded_product);
      ok                                                      = ok && numerical::detail::numpy_multiply(one_one, other_one, loops).data()[0] == (scalar ? rounded_product : fused_product);
      ok                                                      = ok && numerical::detail::numpy_divide(one_one, other_one, loops).data()[0] == numerical::detail::numpy_divide(x, y, scalar);
    }
    return ok;
  }
} // namespace

TEST(complex_arithmetic_as_numpy_computes_it)
{
  EXPECT(numpy_complex_arithmetic_holds<std::complex<double>>());
  EXPECT(numpy_complex_arithmetic_holds<std::complex<float>>());
}

// NumPy's prod and cumprod of complex numbers: a product from 1, a running product from its first element, each
// product NumPy's, with no infinity recovered ((inf − inf·j) · 3 is NaN, where C Annex G's keeps the infinities),
// fused as the loop that runs it: the reduction loop along an axis nothing follows and for a running product past two
// elements, the vector loop between slices and for a running product of two.
namespace {
  template <typename C>
  bool numpy_products_hold()
  {
    using R = C::value_type;
    const R    inf      {std::numeric_limits<R>::infinity()};
    const R    e        {std::numeric_limits<R>::epsilon() * R {4}};
    const C    x        {R {1} + e, R {1}};
    const C    y        {R {1} - e, R {1}};
    const C    one      {R {1}, R {0}};
    const C    fused    {numerical::detail::numpy_multiply(x, y, true)};
    const C    rounded  {numerical::detail::numpy_multiply(x, y, false)};
    const auto nan_both {[](const C& z) { return std::isnan(z.real()) && std::isnan(z.imag()); }};
    bool       ok       {fused != rounded};
    for (const bool reduction : {false, true}) {
      const numerical::detail::numpy_loops loops {.products = !reduction, .scalar_products = false, .quotients = false,
                                                  .reductions = reduction, .late_reductions = false};
      const C                    by_reduction    {reduction ? fused : rounded};
      const C                    by_vector       {reduction ? rounded : fused};
      const numerical::tensor<C> line            {{2}, numerical::buffer<C> {x, y}};
      const numerical::tensor<C> column          {{2, 1}, numerical::buffer<C> {x, y}};
      const numerical::tensor<C> rows            {{2, 2}, numerical::buffer<C> {x, x, y, y}};
      const numerical::tensor<C> three           {{3}, numerical::buffer<C> {one, x, y}};
      ok = ok && numerical::detail::numpy_prod(line, loops) == by_reduction;
      ok = ok && numerical::detail::numpy_prod(line, 0, loops).data()[0] == by_reduction;
      ok = ok && numerical::detail::numpy_prod(column, 0, loops).data()[0] == by_reduction;
      ok = ok && numerical::detail::numpy_prod(rows, 0, loops).data()[1] == by_vector;
      ok = ok && numerical::detail::numpy_cumprod(line, 0, loops).data()[1] == by_vector;
      ok = ok && numerical::detail::numpy_cumprod(three, 0, loops).data()[2] == by_reduction;
      ok = ok && numerical::detail::numpy_cumprod(rows, 0, loops).data()[3] == by_vector;
    }
    // A transpose's lines where its elements lie: each loop the one its own shape asks for, as its copy's.
    const numerical::detail::numpy_loops split {.products = true, .scalar_products = false, .quotients = false,
                                                .reductions = false, .late_reductions = false};
    const numerical::tensor<C> pairs           {numerical::tensor<C> {{2, 2}, numerical::buffer<C> {x, y, x, y}}.transposed()};
    const numerical::tensor<C> runs            {numerical::tensor<C> {{2, 3}, numerical::buffer<C> {one, x, y, one, x, y}}.transposed()};
    ok = ok && numerical::detail::numpy_prod(pairs, 0, split).data()[1] == fused;     // a vector loop between rows
    ok = ok && numerical::detail::numpy_cumprod(runs, 0, split).data()[5] == rounded; // the reduction loop past two
    for (const numerical::tensor<C>& view : {pairs, runs}) {
      const numerical::tensor<C> copied {view.shape(), view.data()};
      for (const std::size_t axis : {std::size_t {0}, std::size_t {1}}) {
        ok = ok && numerical::detail::numpy_prod(view, axis, split) == numerical::detail::numpy_prod(copied, axis, split);
        ok = ok && numerical::detail::numpy_cumprod(view, axis, split) == numerical::detail::numpy_cumprod(copied, axis, split);
      }
    }
    const numerical::tensor<C> infinite {{2}, numerical::buffer<C> {C {inf, -inf}, C {R {3}, R {0}}}};
    ok = ok && nan_both(numerical::detail::numpy_prod(infinite));
    ok = ok && nan_both(numerical::detail::numpy_cumprod(infinite, 0).data()[1]);
    const numerical::tensor<C> alone         {{1}, numerical::buffer<C> {C {inf, R {1}}}};
    const C                    alone_product {numerical::detail::numpy_prod(alone)};
    ok = ok && alone_product.real() == inf && std::isnan(alone_product.imag());  // 1 · (inf + j), as NumPy's from 1
    ok = ok && numerical::detail::numpy_prod(numerical::zeros<C>({0})) == one;
    const numerical::tensor<C> empty_lines {numerical::detail::numpy_prod(numerical::zeros<C>({2, 0}), 1)};
    ok = ok && empty_lines.shape() == std::vector<std::size_t> {2} && empty_lines.data()[0] == one;
    ok = ok && numerical::detail::numpy_prod(numerical::zeros<C>({0, 2}), 1).size() == 0;
    // MSVC's contraction of the reduction loop: the imaginary part's second product fused, ai·br + ar·bi (e² here
    // where the other order and the rounded one give 0); a running product of two keeps the vector loop.
    const C                              a     {R {1}, R {1} + e};
    const C                              b     {-(R {1} - e), R {1}};
    const numerical::detail::numpy_loops late  {.products = true, .scalar_products = true, .quotients = true,
                                                .reductions = true, .late_reductions = true};
    const numerical::tensor<C>           ab    {{2}, numerical::buffer<C> {a, b}};
    ok = ok && numerical::detail::numpy_multiply_late(a, b).imag() != R {0};
    ok = ok && numerical::detail::numpy_multiply(a, b, true).imag() == R {0};
    ok = ok && numerical::detail::numpy_prod(ab, late) == numerical::detail::numpy_multiply_late(a, b);
    ok = ok && numerical::detail::numpy_prod(ab, 0, late).data()[0] == numerical::detail::numpy_multiply_late(a, b);
    ok = ok && numerical::detail::numpy_cumprod(ab, 0, late).data()[1] == numerical::detail::numpy_multiply(a, b, true);
    const numerical::tensor<C> abb  {{3}, numerical::buffer<C> {a, b, one}};
    ok = ok && numerical::detail::numpy_cumprod(abb, 0, late).data()[1] == numerical::detail::numpy_multiply_late(a, b);
    const numerical::tensor<C> pair {{2}, numerical::buffer<C> {x, y}};
    ok = ok && numerical::detail::numpy_prod(pair) == numerical::detail::numpy_multiply(x, y, numerical::detail::numpy_fuses_reductions());
    return ok;
  }
} // namespace

TEST(complex_prod_and_cumprod_as_numpy_computes_them)
{
  EXPECT(numpy_products_hold<std::complex<double>>());
  EXPECT(numpy_products_hold<std::complex<float>>());
  EXPECT_THROWS((void)numerical::detail::numpy_prod(numerical::zeros<std::complex<double>>({2}), 1), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::detail::numpy_cumprod(numerical::zeros<std::complex<double>>({2}), 1), numerical::dimension_error);
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
  const numerical::tensor<double> r            {{count}, std::move(reals)};
  const numerical::tensor<cplx>   z            {{count}, std::move(complexes)};
  const tensor<double>            zero_largest {{3}, {-1.0, -0.0, -2.0}}; // a zero the extreme among nonzero elements
  const tensor<double>            zero_least   {{3}, {1.0, 0.0, 2.0}};
  EXPECT(numerical::max(zero_largest) == 0.0 && std::signbit(numerical::max(zero_largest)));
  EXPECT(numerical::min(zero_least) == 0.0 && !std::signbit(numerical::min(zero_least)));
  const std::size_t               threads {numerical::max_threads()};
  for (const std::size_t n : {std::size_t {1}, threads}) {
    numerical::set_max_threads(n);
    EXPECT(numerical::exp(z) == numerical::map(z, [](const cplx& x) { return numerical::detail::complex_exp(x); }));
    EXPECT(numerical::log(r) == numerical::map(r, [](double x) { return std::log(x); }));
    EXPECT(numerical::exp(r) == numerical::map(r, [](double x) { return numerical::detail::elementary_exp(x); }));
    EXPECT(numerical::tanh(z) == numerical::map(z, [](const cplx& x) { return numerical::detail::complex_tanh(x); }));
    EXPECT(-z == numerical::map(z, [](const cplx& x) { return -x; }));
    EXPECT(numerical::detail::numpy_absolute(z) == numerical::map(z, [](const cplx& x) { return numerical::detail::numpy_absolute(x, numerical::detail::numpy_wheel_loops().products); }));
  }
  numerical::set_max_threads(0); // the default again
}

// exp of a double by the table: within an ulp of the C library's (each within half an ulp or so of the exact value),
// its own value at the ends (overflow past ln of the largest double, the subnormal and zero results the library's,
// NaN itself), and the same bits from a block, in vectors, as from one call.
TEST(exp_of_a_double_by_the_table)
{
  std::vector<double> x;
  for (int i {0}; i < 20000; ++i) {
    const double t {static_cast<double>(i) / 20000.0};
    x.push_back(-760.0 + (1480.0 * t)); // across the whole range, past both ends
    x.push_back((2.0 * t) - 1.0);
    x.push_back(-709.0 + t);            // about the bound below which a result is subnormal
  }
  const double inf {std::numeric_limits<double>::infinity()};
  for (const double v : {0.0, -0.0, inf, -inf, std::numeric_limits<double>::quiet_NaN(), 709.782712893384, 709.79, 800.0,
                         -745.1332191019411, -745.14, -800.0, 1e-300, -1e-300, numerical::detail::exp_normal_bound,
                         std::nextafter(numerical::detail::exp_normal_bound, -inf)}) {
    x.push_back(v);
  }
  std::vector<double> block(x.size());
  numerical::detail::exp_into(x.data(), block.data(), x.size());
  bool same  {true};
  bool close {true};
  for (std::size_t i {0}; i < x.size(); ++i) {
    const double one     {numerical::detail::elementary_exp(x[i])};
    same = same && std::bit_cast<std::uint64_t>(one) == std::bit_cast<std::uint64_t>(block[i]);
    const double library {std::exp(x[i])};
    if (std::isnan(x[i]) || library == 0.0 || std::isinf(library) || library < std::numeric_limits<double>::min()) {
      close = close && std::bit_cast<std::uint64_t>(one) == std::bit_cast<std::uint64_t>(library);
    }
    else {
      close = close && std::fabs(one - library) <= std::nextafter(library, inf) - library;
    }
  }
  EXPECT(same);
  EXPECT(close);
  EXPECT(numerical::detail::elementary_exp(0.0) == 1.0 && numerical::detail::elementary_exp(-0.0) == 1.0);
  EXPECT(numerical::detail::elementary_exp(709.79) == inf && numerical::detail::elementary_exp(-inf) == 0.0);
  EXPECT(numerical::detail::elementary_exp(1.0) == 0x1.5bf0a8b145769p+1); // e, rounded to nearest
}

// expm1 of a double by the table: within an ulp of the C library's (each within half an ulp or so of the exact value),
// the library's own bits at the ends (±0 and the smallest values themselves, −1 far below, overflow past ln of the
// largest double, NaN itself), and the same bits from a block, in vectors, as from one call.
TEST(expm1_of_a_double_by_the_table)
{
  std::vector<double> x;
  for (int i {0}; i < 20000; ++i) {
    const double t {static_cast<double>(i) / 20000.0};
    x.push_back(-760.0 + (1480.0 * t)); // across the whole range, past both ends
    x.push_back((2.0 * t) - 1.0);
    x.push_back((0.02 * t) - 0.01);     // about 0, where 2^(j/128) − 1 cancels the most
    x.push_back(-41.0 + (5.0 * t));     // about −38, past which the result is −1
  }
  const double inf {std::numeric_limits<double>::infinity()};
  for (const double v : {0.0, -0.0, inf, -inf, std::numeric_limits<double>::quiet_NaN(), 709.782712893384, 709.79, 800.0,
                         -40.0, -800.0, 1e-300, -1e-300, 0x1p-54, -0x1p-54, 0x1p-53, std::numbers::ln2,
                         -std::numbers::ln2, 5e-324, 2.5}) { // an odd count: the last one past the pairs
    x.push_back(v);
  }
  std::vector<double> block(x.size());
  numerical::detail::expm1_into(x.data(), block.data(), x.size());
  bool same  {true};
  bool close {true};
  for (std::size_t i {0}; i < x.size(); ++i) {
    const double one     {numerical::detail::elementary_expm1(x[i])};
    same = same && std::bit_cast<std::uint64_t>(one) == std::bit_cast<std::uint64_t>(block[i]);
    const double library {std::expm1(x[i])};
    if (std::isnan(x[i]) || std::isinf(library) || library == -1.0 || std::fabs(x[i]) < 0x1p-54) {
      close = close && std::bit_cast<std::uint64_t>(one) == std::bit_cast<std::uint64_t>(library);
    }
    else {
      close = close && std::fabs(one - library) <= std::fabs(std::nextafter(library, inf) - library);
    }
  }
  EXPECT(same);
  EXPECT(close);
  EXPECT(std::signbit(numerical::detail::elementary_expm1(-0.0)) && numerical::detail::elementary_expm1(-inf) == -1.0);
  EXPECT(numerical::detail::elementary_expm1(709.79) == inf);
  // the table's own values, whichever function elementary_expm1 takes (MSVC's std::expm1 strays an ulp at 1 and −0.5)
  EXPECT(numerical::detail::table_expm1(1.0) == 0x1.b7e151628aed3p+0);    // e − 1, rounded to nearest
  EXPECT(numerical::detail::table_expm1(1e-10) == 0x1.b7cdfd9dda4e3p-34); // rounded to nearest, as the next
  EXPECT(numerical::detail::table_expm1(-0.5) == -0x1.92e9a0720d3ecp-2);
  const numerical::tensor<double> t {{2, 2}, {0.5, -0.0, 1e-20, -3.0}};
  EXPECT(numerical::expm1(t) == numerical::map(t, [](double v) { return numerical::detail::elementary_expm1(v); }));
}

// tanh of a double by exp's table: within an ulp of the C library's tanh in long double where that is wider (glibc's of
// a double strays 2 ulps from the exact value, macOS's half an ulp or so, the reference there), the reference's own
// bits at the ends (±0 and the smallest values themselves, ±1 far out, NaN itself), and the same bits from a block, in
// vectors, as from one call.
TEST(tanh_of_a_double_by_the_table)
{
  std::vector<double> x;
  for (int i {0}; i < 20000; ++i) {
    const double t {static_cast<double>(i) / 20000.0};
    x.push_back(-30.0 + (60.0 * t)); // across the whole range, past both ends
    x.push_back((2.0 * t) - 1.0);
    x.push_back((0.02 * t) - 0.01);  // about 0, where expm1(2x) cancels the most
    x.push_back(17.0 + (6.0 * t));   // about 19, past which the result is 1
  }
  const double inf {std::numeric_limits<double>::infinity()};
  for (const double v : {0.0, -0.0, inf, -inf, std::numeric_limits<double>::quiet_NaN(), 22.0, -22.0, 800.0, -800.0,
                         1e-300, -1e-300, 0x1p-27, -0x1p-27, 0x1.fffffffffffffp-28, 5e-324, 2.5,
                         -std::numbers::ln2}) { // an odd count: the last one past the pairs
    x.push_back(v);
  }
  std::vector<double> block(x.size());
  numerical::detail::tanh_into(x.data(), block.data(), x.size());
  bool same  {true};
  bool close {true};
  for (std::size_t i {0}; i < x.size(); ++i) {
    const double one     {numerical::detail::elementary_tanh(x[i])};
    same = same && std::bit_cast<std::uint64_t>(one) == std::bit_cast<std::uint64_t>(block[i]);
    const double library {std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits
                            ? static_cast<double>(std::tanh(static_cast<long double>(x[i])))
                            : std::tanh(x[i])};
    if (std::isnan(x[i]) || std::fabs(library) == 1.0 || std::fabs(x[i]) < 0x1p-27) {
      close = close && std::bit_cast<std::uint64_t>(one) == std::bit_cast<std::uint64_t>(library);
    }
    else {
      close = close && std::fabs(one - library) <= std::fabs(std::nextafter(library, inf) - library);
    }
  }
  EXPECT(same);
  EXPECT(close);
  EXPECT(std::signbit(numerical::detail::elementary_tanh(-0.0)) && numerical::detail::elementary_tanh(-inf) == -1.0);
  EXPECT(numerical::detail::table_tanh(1.0) == 0x1.85efab514f394p-1); // the table's, rounded to nearest, as the next
  EXPECT(numerical::detail::table_tanh(-0.001) == -0x1.0624d77516ce2p-10);
  const numerical::tensor<double> t {{2, 2}, {0.5, -0.0, 1e-20, -3.0}};
  EXPECT(numerical::tanh(t) == numerical::map(t, [](double v) { return numerical::detail::elementary_tanh(v); }));
}

//! \brief Whether \p into of \p x (blocks, in vectors) gives \p one's bits for each value, \p one within an ulp of
//!        \p reference (the C library's in long double where that is wider) and its own bits where \p exact says so.
template <typename Into,
          typename One,
          typename Reference,
          typename Exact>
bool table_function_holds(const std::vector<double>& x,
                          const Into&                into,
                          const One&                 one,
                          const Reference&           reference,
                          const Exact&               exact)
{
  const double        inf {std::numeric_limits<double>::infinity()};
  std::vector<double> block(x.size());
  into(x.data(), block.data(), x.size());
  bool holds {true};
  for (std::size_t i {0}; i < x.size(); ++i) {
    const double value   {one(x[i])};
    holds = holds && std::bit_cast<std::uint64_t>(value) == std::bit_cast<std::uint64_t>(block[i]);
    const double library {reference(x[i])};
    if (std::isnan(x[i]) || std::isinf(library) || exact(x[i], library)) {
      holds = holds && (std::bit_cast<std::uint64_t>(value) == std::bit_cast<std::uint64_t>(library)
                        || (std::isnan(value) && std::isnan(library)));
    }
    else {
      holds = holds && std::fabs(value - library) <= std::fabs(std::nextafter(library, inf) - library);
    }
  }
  return holds;
}

// sinh, cosh and exp2 of a double by exp's table: within an ulp of the C library's in long double where that is wider,
// its own bits at the ends (±0 and the smallest values of sinh themselves, cosh 1 there, the overflow past 710.47 where
// e^x alone overflows from 709.78, 2^x at every integer, NaN), and the same bits from a block, in vectors, as from one
// call; values rounded to nearest, whichever function the elementary ones take.
TEST(sinh_cosh_and_exp2_of_a_double_by_the_table)
{
  constexpr bool      wider {std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits};
  const double        inf   {std::numeric_limits<double>::infinity()};
  std::vector<double> x;
  for (int i {0}; i < 20000; ++i) {
    const double t {static_cast<double>(i) / 20000.0};
    x.push_back(-720.0 + (1440.0 * t)); // across the whole range, past both ends
    x.push_back((2.0 * t) - 1.0);
    x.push_back((0.02 * t) - 0.01);     // about 0, where d − u cancels the most
    x.push_back(35.0 + (20.0 * t));     // about 44, where t passes 2^64
    x.push_back(709.0 + (2.0 * t));     // about the overflow
  }
  for (const double v : {0.0, -0.0, inf, -inf, std::numeric_limits<double>::quiet_NaN(), 710.4758600739439, 710.476, 1e-300,
                         -1e-300, 0x1p-27, -0x1p-27, 0x1.fffffffffffffp-28, 5e-324, 2.5, -1022.0, -1022.5, 1023.0, 1024.0,
                         -1100.0}) { // an odd count: the last one past the pairs
    x.push_back(v);
  }
  EXPECT(table_function_holds(
           x, [](const double* in, double* out, std::size_t count) { numerical::detail::sinh_into(in, out, count); },
           [](double v) { return numerical::detail::elementary_sinh(v); },
           [](double v) { return wider ? static_cast<double>(std::sinh(static_cast<long double>(v))) : std::sinh(v); },
           [](double v, double /*library*/) { return std::fabs(v) < 0x1p-27; }));
  EXPECT(table_function_holds(
           x, [](const double* in, double* out, std::size_t count) { numerical::detail::cosh_into(in, out, count); },
           [](double v) { return numerical::detail::elementary_cosh(v); },
           [](double v) { return wider ? static_cast<double>(std::cosh(static_cast<long double>(v))) : std::cosh(v); },
           [](double v, double /*library*/) { return std::fabs(v) < 0x1p-27; }));
  EXPECT(table_function_holds(
           x, [](const double* in, double* out, std::size_t count) { numerical::detail::exp2_into(in, out, count); },
           [](double v) { return numerical::detail::elementary_exp2(v); },
           [](double v) { return std::exp2(v); }, // exact where the result is subnormal (std::exp2's own) and at integers
           [](double v, double /*library*/) { return v < numerical::detail::exp2_normal_bound || v == std::trunc(v); }));
  EXPECT(std::signbit(numerical::detail::elementary_sinh(-0.0)) && numerical::detail::elementary_sinh(-inf) == -inf);
  EXPECT(numerical::detail::elementary_cosh(-inf) == inf && numerical::detail::elementary_exp2(-inf) == 0.0);
  EXPECT(numerical::detail::table_sinh(1.0) == 0x1.2cd9fc44eb982p+0); // rounded to nearest, as the next
  EXPECT(numerical::detail::table_sinh(-0.001) == -0x1.0624e00c1c9e3p-10);
  EXPECT(numerical::detail::table_sinh(710.0) == 0x1.3e21a464507f9p+1023);
  EXPECT(numerical::detail::table_cosh(1.0) == 0x1.8b07551d9f55p+0);
  EXPECT(numerical::detail::table_cosh(-0.001) == 0x1.000008637bdc1p+0);
  EXPECT(numerical::detail::table_exp2(0.5) == 0x1.6a09e667f3bcdp+0);
  EXPECT(numerical::detail::table_exp2(-3.3) == 0x1.9fdf8bcce533ep-4);
  EXPECT(numerical::detail::table_exp2(1023.5) == 0x1.6a09e667f3bcdp+1023);
  const numerical::tensor<double> t {{2, 2}, {0.5, -0.0, 1e-20, -3.0}};
  EXPECT(numerical::sinh(t) == numerical::map(t, [](double v) { return numerical::detail::elementary_sinh(v); }));
  EXPECT(numerical::cosh(t) == numerical::map(t, [](double v) { return numerical::detail::elementary_cosh(v); }));
  EXPECT(numerical::exp2(t) == numerical::map(t, [](double v) { return numerical::detail::elementary_exp2(v); }));
  const numerical::tensor<float> f {{2}, {0.5F, -3.0F}};
  EXPECT(numerical::sinh(f).data()[1] == std::sinh(-3.0F) && numerical::cosh(f).data()[0] == std::cosh(0.5F)
         && numerical::exp2(f).data()[1] == std::exp2(-3.0F));
}

// atan of a double by its table: within an ulp of the C library's atan in long double where that is wider (the
// library's of a double strays further, macOS's 0.86 ulp), the reference's own bits at the ends (±0 and the smallest
// values themselves, ±π/2 far out, NaN itself), and the same bits from a block, in vectors, as from one call.
TEST(atan_of_a_double_by_the_table)
{
  std::vector<double> x;
  for (int i {0}; i < 20000; ++i) {
    const double t {static_cast<double>(i) / 20000.0};
    x.push_back(-20.0 + (40.0 * t));                  // across the inverses' range and the table's
    x.push_back((2.0 * t) - 1.0);
    x.push_back(0.98 + (0.04 * t));                   // about 1, where the inverse takes over
    x.push_back(std::ldexp(1.0 + t, (i % 120) - 60)); // from 2^−60 to 2^60
  }
  const double inf {std::numeric_limits<double>::infinity()};
  for (const double v : {0.0, -0.0, inf, -inf, std::numeric_limits<double>::quiet_NaN(), 1.0, -1.0, 0x1p54, 0x1p55,
                         1e300, 1e-300, -1e-300, 0x1p-27, 0x1.fffffffffffffp-28, 5e-324, 2.5,
                         std::nextafter(1.0, 2.0)}) { // an odd count: the last one past the pairs
    x.push_back(v);
  }
  std::vector<double> block(x.size());
  numerical::detail::atan_into(x.data(), block.data(), x.size());
  bool same  {true};
  bool close {true};
  for (std::size_t i {0}; i < x.size(); ++i) {
    const double one     {numerical::detail::elementary_atan(x[i])};
    same = same && std::bit_cast<std::uint64_t>(one) == std::bit_cast<std::uint64_t>(block[i]);
    const double library {std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits
                            ? static_cast<double>(std::atan(static_cast<long double>(x[i])))
                            : std::atan(x[i])};
    if (std::isnan(x[i]) || std::fabs(x[i]) < 0x1p-27 || std::fabs(x[i]) > 0x1p54) {
      close = close && std::bit_cast<std::uint64_t>(one) == std::bit_cast<std::uint64_t>(library);
    }
    else {
      close = close && std::fabs(one - library) <= std::fabs(std::nextafter(library, inf) - library);
    }
  }
  EXPECT(same);
  EXPECT(close);
  EXPECT(std::signbit(numerical::detail::elementary_atan(-0.0)));
  // the table's own values, whichever function elementary_atan takes
  EXPECT(numerical::detail::table_atan(0.5) == 0x1.dac670561bb4fp-2); // rounded to nearest, as the next
  EXPECT(numerical::detail::table_atan(3.0) == 0x1.3fc176b7a8560p+0);
  EXPECT(numerical::detail::table_atan(-0.00001) == -0x1.4f8b588e06854p-17);
  EXPECT(numerical::detail::table_atan(1e10) == 0x1.921fb543d4de0p+0);
  EXPECT(numerical::detail::table_atan(-inf) == -numerical::detail::atan_half_pi_high);
  const numerical::tensor<double> t {{2, 2}, {0.5, -0.0, 1e-20, -3.0}};
  EXPECT(numerical::atan(t) == numerical::map(t, [](double v) { return numerical::detail::elementary_atan(v); }));
}

// sin and cos of a double by a table of sin(jπ/32) and cos(jπ/32): within an ulp of the C library's in long double
// where that is wider, two where it is not (macOS's own strays 0.86 ulp), the library's bits past 2^20 and for an
// infinity or a NaN, x itself (and 1) under 2^−27 with ±0 kept, and the same bits from a block as from one call;
// values rounded to nearest, whichever function elementary_sin takes.
TEST(sin_and_cos_of_a_double_by_the_table)
{
  std::vector<double> x;
  for (int i {0}; i < 20000; ++i) {
    const double t {static_cast<double>(i) / 20000.0};
    x.push_back(-8.0 + (16.0 * t));                                         // the table's every entry, both signs
    x.push_back(static_cast<double>(i) * 0x1.921fb54442d18p-4);             // on multiples of π/32
    x.push_back(std::ldexp(1.0 + t, (i % 80) - 30));                        // from 2^−30 to 2^49, past the bound
    x.push_back(((static_cast<double>(i % 7) - 3.0) * 1e5) + (1000.0 * t)); // far in, below the bound
  }
  const double inf {std::numeric_limits<double>::infinity()};
  for (const double v : {0.0, -0.0, inf, -inf, std::numeric_limits<double>::quiet_NaN(), 0x1p-27, 0x1.fffffffffffffp-28,
                         -0x1p-30, 5e-324, 0x1p20, std::nextafter(0x1p20, inf), -0x1p20, 1e300}) { // an odd count
    x.push_back(v);
  }
  const bool wider {std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits};
  // the C library's sin or cos of v in double, and rounded from long double where that is wider
  const auto in_double {[](bool cosine, double v) { return cosine ? std::cos(v) : std::sin(v); }};
  const auto reference {[&](bool cosine, double v) {
                          if (!wider) {
                            return in_double(cosine, v);
                          }
                          const long double at {static_cast<long double>(v)};
                          return static_cast<double>(cosine ? std::cos(at) : std::sin(at));
                        }};
  for (const bool cosine : {false, true}) {
    std::vector<double> block(x.size());
    if (cosine) {
      numerical::detail::double_sines_into<true>(x.data(), block.data(), x.size());
    }
    else {
      numerical::detail::double_sines_into<false>(x.data(), block.data(), x.size());
    }
    bool same  {true};
    bool close {true};
    for (std::size_t i {0}; i < x.size(); ++i) {
      const double one          {cosine ? numerical::detail::elementary_cos(x[i]) : numerical::detail::elementary_sin(x[i])};
      same = same && std::bit_cast<std::uint64_t>(one) == std::bit_cast<std::uint64_t>(block[i]);
      const double library      {reference(cosine, x[i])};
      if (!(std::fabs(x[i]) <= numerical::detail::sine_reduction_bound)) { // the library's own, NaN for NaN and ±∞
        close = close && (std::isnan(one) ? std::isnan(library) : one == in_double(cosine, x[i]));
      }
      else {
        const double ulp {std::fabs(std::nextafter(library, inf) - library)};
        close = close && std::fabs(one - library) <= (wider ? 1.0 : 2.0) * ulp;
      }
    }
    EXPECT(same);
    EXPECT(close);
  }
  EXPECT(std::signbit(numerical::detail::elementary_sin(-0.0)) && numerical::detail::elementary_cos(-0.0) == 1.0);
  EXPECT(numerical::detail::elementary_sin(0x1.fffffffffffffp-28) == 0x1.fffffffffffffp-28);
  // the table's own values, rounded to nearest, whichever function elementary_sin takes
  EXPECT(numerical::detail::table_sine<false>(0.5) == 0x1.eaee8744b05f0p-2);
  EXPECT(numerical::detail::table_sine<true>(0.5) == 0x1.c1528065b7d50p-1);
  EXPECT(numerical::detail::table_sine<false>(3.0) == 0x1.210386db6d55bp-3);
  EXPECT(numerical::detail::table_sine<true>(3.0) == -0x1.fae04be85e5d2p-1);
  EXPECT(numerical::detail::table_sine<false>(-1e5) == -0x1.24daa9c527e96p-5);
  EXPECT(numerical::detail::table_sine<true>(1e6) == 0x1.df9df9906d32cp-1);
  const numerical::tensor<double> t {{2, 2}, {0.5, -0.0, 1e-20, 3e7}};
  EXPECT(numerical::sin(t) == numerical::map(t, [](double v) { return numerical::detail::elementary_sin(v); }));
  EXPECT(numerical::cos(t) == numerical::map(t, [](double v) { return numerical::detail::elementary_cos(v); }));
  const numerical::tensor<double> wide   {numerical::arange(0.0, 2048.0, 1.0).reshaped({32, 64})};
  const numerical::tensor<double> wide_t {wide.transposed({1, 0})};
  EXPECT(numerical::sin(wide_t) == numerical::sin(wide_t.materialized()) && numerical::cos(wide_t) == numerical::cos(wide_t.materialized()));
}

// log1p of a double by a table of ln(k/128): within an ulp of the C library's in long double where that is wider, its
// own bits at the ends (x itself under 2^−54, ±0 kept, −∞ at −1, NaN below), and the same bits from a block, in vectors,
// as from one call; values rounded to nearest, whichever function elementary_log1p takes.
TEST(log1p_of_a_double_by_the_table)
{
  std::vector<double> x;
  for (int i {0}; i < 20000; ++i) {
    const double t {static_cast<double>(i) / 20000.0};
    x.push_back(-0.999 + (6.0 * t));
    x.push_back(-0.01 + (0.02 * t));                   // about 0, where 1 + x rounds
    x.push_back(std::ldexp(1.0 + t, (i % 120) - 60));  // from 2^−60 to 2^60
    x.push_back(-std::ldexp(1.0 + t, -(i % 60) - 2));  // toward −1 and toward 0 from below
  }
  const double inf {std::numeric_limits<double>::infinity()};
  for (const double v : {0.0, -0.0, inf, -inf, std::numeric_limits<double>::quiet_NaN(), -1.0, -2.0, 1.0, 0x1p-54,
                         0x1.fffffffffffffp-55, -0x1p-54, 5e-324, std::numeric_limits<double>::max(),
                         0x1.9e59973ebc448p-53, std::nextafter(-1.0, 0.0)}) { // an odd count: the last past the pairs
    x.push_back(v);
  }
  constexpr bool wider {std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits};
  EXPECT(table_function_holds(
           x, [](const double* in, double* out, std::size_t count) { numerical::detail::log1p_into(in, out, count); },
           [](double v) { return numerical::detail::elementary_log1p(v); },
           [](double v) { return wider ? static_cast<double>(std::log1p(static_cast<long double>(v))) : std::log1p(v); },
           [](double v, double library) { return std::fabs(v) < 0x1p-54 || std::isnan(library); })); // NaN below −1
  EXPECT(std::signbit(numerical::detail::elementary_log1p(-0.0)));
  // the table's own values, whichever function elementary_log1p takes: each rounded to nearest
  EXPECT(numerical::detail::table_log1p(-0.5) == -0x1.62e42fefa39efp-1);
  EXPECT(numerical::detail::table_log1p(1e-10) == 0x1.b7cdfd9d1d693p-34);
  EXPECT(numerical::detail::table_log1p(0x1.9e59973ebc448p-53) == 0x1.9e59973ebc447p-53); // 1 + x's error as large as r
  EXPECT(numerical::detail::table_log1p(1e300) == 0x1.5963447f87fb5p+9);
  EXPECT(numerical::detail::table_log1p(-0x1.fdc45e9aa3829p-1) == -0x1.5bd7317316e85p+2);
  EXPECT(numerical::detail::table_log1p(0x1.68aa500d14646p-8) == 0x1.67ad2df83eb5p-8);
  EXPECT(numerical::detail::table_log1p(-1.0) == -inf);
  EXPECT(std::isnan(numerical::detail::table_log1p(-1.5)));
  const numerical::tensor<double> t {{4}, {0.5, -0.0, 1e-20, -0.75}}; // an even count: no lane alone
  EXPECT(numerical::log1p(t) == numerical::map(t, [](double v) { return numerical::detail::elementary_log1p(v); }));
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
  // One element of more axes than a transpose beside it: the result takes its axes, the transpose read as it lies.
  const tensor<double> turned {counting({2, 3}).transposed()};
  const tensor<double> ten    {{1, 1, 1}, {10.0}};
  const tensor<double> lifted {{1, 3, 2}, {10.0, 13.0, 11.0, 14.0, 12.0, 15.0}};
  EXPECT((turned + ten) == lifted);
  EXPECT((ten + turned) == lifted);
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

// A reshape of a view reads it where it lies wherever NumPy's is a view: axes of extent 1 put in or taken out, an axis
// split, axes joined whose elements lie one stride from the next (from a slice's offset, along a reversed axis); any
// other is the view's row-major copy, shared. The elements, in row-major order, the view's either way.
TEST(a_reshape_of_a_view_reads_it_where_it_lies)
{
  const tensor<double> cube     {counting({8, 16, 10})};      // its slices past 512 elements: views
  const tensor<double> permuted {cube.transposed({2, 0, 1})}; // (10, 8, 16), strides (1, 160, 10)
  const tensor<double> rows     {cube.sliced({1, 0, 0}, {2, 1, 1}, {4, 16, 10}, std::vector<bool>(3, true))};
  const tensor<double> reversed {cube.sliced({7, 0, 0}, {-1, 1, 1}, {8, 16, 10}, std::vector<bool>(3, true))};
  struct reshaping {
    const tensor<double>* view;
    shape                 into;
    bool                  in_place;
  };
  const std::vector<reshaping> reshapings {
    {.view = &permuted, .into = {1, 10, 8, 16, 1}, .in_place = true},
    {.view = &permuted, .into = {10, 128}, .in_place = true},
    {.view = &permuted, .into = {10, 2, 4, 16}, .in_place = true},
    {.view = &permuted, .into = {1, 10, 1, 128}, .in_place = true},
    {.view = &permuted, .into = {80, 16}, .in_place = false},
    {.view = &permuted, .into = {1280}, .in_place = false},
    {.view = &rows, .into = {4, 160}, .in_place = true},
    {.view = &rows, .into = {1, 4, 4, 4, 10}, .in_place = true},
    {.view = &rows, .into = {640}, .in_place = false},
    {.view = &reversed, .into = {8, 160}, .in_place = true},
    {.view = &reversed, .into = {1280}, .in_place = false},
    {.view = &reversed, .into = {2, 4, 16, 10}, .in_place = true}};
  for (const reshaping& each : reshapings) {
    const tensor<double> reshaped {each.view->reshaped(each.into)};
    EXPECT(reshaped.shape() == each.into && reshaped.data() == each.view->data());
    EXPECT(reshaped.contiguous() != each.in_place);
    EXPECT(each.in_place ? reshaped.origin() == each.view->origin() : reshaped.data().data() == each.view->data().data());
  }
  const tensor<double> squeezed {permuted.reshaped({1, 10, 1, 8, 16}).reshaped({10, 8, 16})};
  EXPECT(!squeezed.contiguous() && squeezed.strides() == permuted.strides() && squeezed.data() == permuted.data());
  EXPECT(permuted.reshaped({10, 128}).strides() == (std::vector<std::ptrdiff_t> {1, 10}));
  EXPECT(numerical::sum(permuted.reshaped({1, 10, 128}), 2) == numerical::sum(tensor<double> {{1, 10, 128}, permuted.data()}, 2));
  EXPECT_THROWS((void)permuted.reshaped({1281}), numerical::dimension_error);
  EXPECT_THROWS((void)rows.reshaped({639}), numerical::dimension_error);
}

TEST(reshape_keeps_the_order_and_checks_the_count)
{
  const tensor<double> line {counting({6})};
  const tensor<double> grid {numerical::reshape(line, {2, 3})};
  EXPECT(grid.shape() == (shape {2, 3}));
  EXPECT(grid(std::vector<std::size_t> {1, 0}) == 3.0);
  EXPECT_THROWS((void)numerical::reshape(line, {4}), numerical::dimension_error);
}

// Tensors share their immutable elements: a copy and a reshape hold the one buffer (no element copied), which lives
// as long as one of them does; a reshape to another count of elements is refused.
TEST(tensors_share_their_elements)
{
  const tensor<double> line {{6}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  const tensor<double> grid {numerical::reshape(line, {2, 3})};
  const tensor<double> copy {line}; // NOLINT(performance-unnecessary-copy-initialization): the copy is the case tested
  EXPECT(grid.data().data() == line.data().data() && copy.data().data() == line.data().data());
  EXPECT(grid(std::vector<std::size_t> {1, 0}) == 4.0 && grid.shape() == (std::vector<std::size_t> {2, 3}));
  const tensor<double> kept {[] {
                               const tensor<double> transient {{4}, {7.0, 8.0, 9.0, 10.0}};
                               return transient.reshaped({2, 2});
                             }()};
  EXPECT(kept(std::vector<std::size_t> {1, 1}) == 10.0); // the buffer outlived the tensor it was built in
  EXPECT_THROWS((void)line.reshaped({4}), numerical::dimension_error);
  // From an expiring matrix: its extents read before its elements move out, whatever the compiler's order.
  const tensor<double> taken {numerical::matrix<double> {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  EXPECT(taken.shape() == (std::vector<std::size_t> {2, 3}) && taken(std::vector<std::size_t> {1, 2}) == 6.0);
}

// A permutation of the axes is a view: read where the elements lie, copied out in row-major order by the first data()
// call alone and shared with the view's copies and reshapes; one that leaves them row-major (moving axes of extent 1,
// undoing another, a tensor without elements) shares them as they lie; anything but a permutation is refused.
TEST(a_permutation_of_the_axes_is_a_view)
{
  const tensor<double> cube {counting({2, 3, 4})};
  const tensor<double> view {cube.transposed({2, 0, 1})};
  EXPECT(!view.contiguous() && view.shape() == (shape {4, 2, 3}) && view.size() == 24);
  using steps = std::vector<std::ptrdiff_t>;
  EXPECT(view.strides() == (steps {1, 12, 4}) && view.origin() == cube.data().data() && cube.strides() == (steps {12, 4, 1}));
  EXPECT(view(std::vector<std::size_t> {3, 1, 2}) == cube(std::vector<std::size_t> {1, 2, 3}));
  numerical::buffer<double> expected;
  for (std::size_t k {0}; k < 4; ++k) {
    for (std::size_t i {0}; i < 2; ++i) {
      for (std::size_t j {0}; j < 3; ++j) {
        expected.push_back(cube(std::vector<std::size_t> {i, j, k}));
      }
    }
  }
  const double* const copied {view.data().data()};
  EXPECT(view.data() == expected && copied != cube.data().data() && view.data().data() == copied);
  const tensor<double> again {view};                                                          // NOLINT(performance-unnecessary-copy-initialization): the copy is the case tested
  EXPECT(again.data().data() == copied && view.reshaped({24}).data().data() == copied);
  EXPECT(view == (tensor<double> {{4, 2, 3}, expected}) && !(view == cube.reshaped({4, 2, 3})));
  const tensor<double> flat    {[&] { return cube.transposed({2, 0, 1}).reshaped({24}); }()}; // the copy outlives the view
  EXPECT(flat.data() == expected);
  const tensor<double> back    {view.transposed({1, 2, 0})};                                  // undone: the elements as they lie
  EXPECT(back.contiguous() && back.data().data() == cube.data().data());
  const tensor<double> matrix  {counting({2, 3})};
  EXPECT(matrix.transposed().data() == (numerical::buffer<double> {0.0, 3.0, 1.0, 4.0, 2.0, 5.0}));
  const tensor<double> ones    {counting({1, 3, 1, 2})};
  EXPECT(ones.transposed({2, 1, 0, 3}).contiguous() && ones.transposed({2, 1, 0, 3}).data().data() == ones.data().data());
  const tensor<double> between {counting({2, 1, 3}).transposed({2, 1, 0})}; // an axis of extent 1 among them
  EXPECT(!between.contiguous() && between.data() == (numerical::buffer<double> {0.0, 3.0, 1.0, 4.0, 2.0, 5.0}));
  const tensor<double> empty   {counting({2, 0, 3})};
  EXPECT(empty.transposed().contiguous() && empty.transposed().shape() == (shape {3, 0, 2}) && empty.transposed().size() == 0);
  EXPECT((tensor<double> {{}, {7.0}}).transposed().data() == (numerical::buffer<double> {7.0}));
  EXPECT_THROWS((void)matrix.transposed({0, 0}), numerical::dimension_error);
  EXPECT_THROWS((void)matrix.transposed({0, 2}), numerical::dimension_error);
  EXPECT_THROWS((void)matrix.transposed({0}), numerical::dimension_error);
}

// A transpose is laid out as its elements lie: its axes from the widest stride, the tensor over the same elements
// row-major in that order, none copied; an axis of extent 1 among its ties anywhere. A tensor row-major already, a slice
// skipping elements and one starting past the first are not.
TEST(a_transpose_is_laid_out_as_its_elements_lie)
{
  using order = std::vector<std::size_t>;
  using all   = std::vector<bool>;
  const tensor<double> cube {counting({8, 9, 10})};
  const auto           laid {cube.transposed({2, 0, 1}).laid_out()};
  EXPECT(laid && laid->second == (order {1, 2, 0}) && laid->first.shape() == (shape {8, 9, 10}) && laid->first.contiguous()
         && laid->first.origin() == cube.origin());
  const tensor<double> flat {counting({1, 30, 40})};
  const auto           tied {flat.transposed({2, 1, 0}).laid_out()};
  EXPECT(tied && tied->second == (order {2, 1, 0}) && tied->first.shape() == (shape {1, 30, 40}) && tied->first.origin() == flat.origin());
  EXPECT(!cube.laid_out());
  const tensor<double> grid {counting({40, 40})};
  EXPECT(!grid.sliced({0, 0}, {1, 2}, {40, 20}, all(2, true)).laid_out());
  EXPECT(!grid.sliced({0, 39}, {1, -1}, {40, 40}, all(2, true)).laid_out());
}

// Along an axis of a transpose, its elements as they lie along the matching axis, none copied: the scans, the sorts,
// the partitions, the reductions, the extremes' indices and the counts of every axis of transposes of order 2 and 3 the
// bits of their contiguous copies'; of every element, the extremes and an integer sum (in any order the same); an axis
// past the rank refused alike.
TEST(a_transpose_along_an_axis_reads_its_elements_as_they_lie)
{
  numerical::buffer<double>       values(std::size_t {30} *40 * 7);
  numerical::buffer<std::int64_t> integers(values.size());
  for (std::size_t i {0}; i < values.size(); ++i) {
    const auto drawn {static_cast<std::int64_t>((i * 7919U) % 101U) - 50};
    values[i]   = static_cast<double>(drawn) * 0.37; // ties, zeros, negatives, sums rounded
    integers[i] = drawn * 3037000493LL;              // sums wrapped
  }
  const tensor<double>           cube {{30, 40, 7}, values};
  const auto                     same = [](const auto& x, const auto& y) { return x.shape() == y.shape() && x.data() == y.data(); };
  const std::vector<std::size_t> kths {3, 5};
  bool                           all  {true};
  for (const tensor<double>& view : {cube.transposed({2, 0, 1}), cube.transposed({1, 2, 0}), cube.transposed({1, 0, 2}),
                                     cube.reshaped({1200, 7}).transposed()}) {
    const tensor<double> copy {view.shape(), view.data()};
    EXPECT(view.laid_out().has_value());
    for (std::size_t axis {0}; axis < view.rank(); ++axis) {
      all = all && same(numerical::cumsum(view, axis), numerical::cumsum(copy, axis)) && same(numerical::cumprod(view, axis), numerical::cumprod(copy, axis))
            && same(numerical::sort(view, axis), numerical::sort(copy, axis)) && same(numerical::argsort(view, axis), numerical::argsort(copy, axis))
            && same(numerical::partition(view, kths, axis), numerical::partition(copy, kths, axis))
            && same(numerical::argpartition(view, kths, axis), numerical::argpartition(copy, kths, axis))
            && same(numerical::argmax(view, axis), numerical::argmax(copy, axis)) && same(numerical::argmin(view, axis), numerical::argmin(copy, axis))
            && same(numerical::count_nonzero(view, axis), numerical::count_nonzero(copy, axis))
            && same(numerical::sum(view, axis), numerical::sum(copy, axis)) && same(numerical::mean(view, axis), numerical::mean(copy, axis))
            && same(numerical::max(view, axis), numerical::max(copy, axis)) && same(numerical::min(view, axis), numerical::min(copy, axis))
            && same(numerical::prod(view, axis), numerical::prod(copy, axis));
    }
    EXPECT_THROWS((void)numerical::argmax(view, view.rank()), numerical::dimension_error);
    EXPECT_THROWS((void)numerical::cumsum(view, view.rank()), numerical::dimension_error);
    EXPECT_THROWS((void)numerical::sum(view, view.rank()), numerical::dimension_error);
    EXPECT(numerical::count_nonzero(view) == numerical::count_nonzero(copy));
    EXPECT(numerical::max(view) == numerical::max(copy) && numerical::min(view) == numerical::min(copy));
  }
  const tensor<std::int64_t> whole {{30, 40, 7}, integers};
  for (const tensor<std::int64_t>& view : {whole.transposed({2, 0, 1}), whole.reshaped({1200, 7}).transposed()}) {
    const tensor<std::int64_t> copy {view.shape(), view.data()};
    EXPECT(numerical::sum(view) == numerical::sum(copy) && numerical::max(view) == numerical::max(copy));
    for (std::size_t axis {0}; axis < view.rank(); ++axis) {
      all = all && same(numerical::sum(view, axis), numerical::sum(copy, axis)) && same(numerical::prod(view, axis), numerical::prod(copy, axis));
    }
  }
  EXPECT(all);
}

// An operation of two tensors reads a transpose where its elements lie: against one element (a number) as the
// element-wise functions do, against a transpose alike pair by pair; the bits, and a comparison's bools, of the
// contiguous copies'. One element of more axes broadcasts them; a transpose against another permutation's or a
// slice's is copied.
TEST(an_operation_of_two_tensors_reads_transposes_where_they_lie)
{
  using all = std::vector<bool>;
  numerical::buffer<double> values(std::size_t {40} *30 * 7);
  for (std::size_t i {0}; i < values.size(); ++i) {
    values[i] = (static_cast<double>((i * 7919U) % 2001U) * 0.005) - 5.0;
  }
  const tensor<double> cube   {{40, 30, 7}, values};
  const tensor<double> other  {cube * tensor<double> {{}, {1.5}}};
  const tensor<double> number {{}, {0.75}};
  const tensor<double> one    {{1, 1, 1}, {-2.25}};
  const auto           same   = [](const auto& x, const auto& y) {
                                  return x.shape() == y.shape() && std::memcmp(x.data().data(), y.data().data(), x.size() * sizeof(x.data()[0])) == 0;
                                };
  const auto           copied = [](const tensor<double>& t) { return tensor<double> {t.shape(), t.data()}; };
  const auto           less   = [](double x, double y) { return static_cast<std::uint8_t>(x < y); }; // a comparison's
  bool                 holds  {true};
  for (const tensor<double>& view : {cube.transposed({2, 0, 1}), cube.reshaped({1200, 7}).transposed(),
                                     cube.sliced({0, 0, 0}, {1, 2, 1}, {40, 15, 7}, all(3, true))}) {
    const tensor<double> copy {copied(view)};
    holds = holds && same(view + number, copy + number) && same(number - view, number - copy) && same(view * one, copy * one)
            && same(one / view, one / copy) && same(numerical::broadcast_apply(view, number, less), numerical::broadcast_apply(copy, number, less));
  }
  const tensor<double> a {cube.transposed({2, 0, 1})};
  const tensor<double> b {other.transposed({2, 0, 1})};
  const tensor<double> c {tensor<double> {{7, 30, 40}, values}.transposed({0, 2, 1})};
  EXPECT(a.shape() == c.shape() && a.laid_out() && c.laid_out() && a.laid_out()->second != c.laid_out()->second);
  holds = holds && same(a + b, copied(a) + copied(b)) && same(a + c, copied(a) + copied(c)) && same(a - c, copied(a) - copied(c)) && same(b * a, copied(b) * copied(a))
          && same(numerical::broadcast_apply(a, b, less), numerical::broadcast_apply(copied(a), copied(b), less));
  EXPECT(holds);
  const tensor<double> row {{3}, {1.0, 2.0, 3.0}};
  EXPECT((row + one).shape() == (shape {1, 1, 3}) && (one * row).data() == (numerical::buffer<double> {-2.25, -4.5, -6.75}));
  EXPECT((tensor<double> {{0, 3}, {}} + number).shape() == (shape {0, 3}));
  const tensor<double> column {{2, 1}, {10.0, 20.0}}; // a column against short rows, on either side
  const tensor<double> rows   {{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  EXPECT((column + rows).data() == (numerical::buffer<double> {11.0, 12.0, 13.0, 24.0, 25.0, 26.0}) && (rows + column).data() == (column + rows).data());
  // a column read where it lies, every other element of a longer one, against rows together; a contiguous column
  // against rows apart (the first 3 of 40): the copies' bits
  using all_axes = std::vector<bool>;
  const tensor<double> tall        {cube.reshaped({8400, 1})};
  const tensor<double> wide        {cube.reshaped({1050, 8})};
  const tensor<double> every_other {tall.sliced({0, 0}, {2, 1}, {1200, 1}, all_axes(2, true))};
  const tensor<double> first_three {wide.sliced({0, 0}, {1, 1}, {1050, 3}, all_axes(2, true))};
  const tensor<double> short_rows  {copied(tall.sliced({0, 0}, {1, 1}, {1200, 1}, all_axes(2, true))) * tensor<double> {{1, 3}, {1.0, 2.0, 3.0}}};
  EXPECT(!every_other.contiguous() && !first_three.contiguous() && short_rows.shape() == (shape {1200, 3}));
  EXPECT(same(every_other + short_rows, copied(every_other) + short_rows) && same(short_rows + every_other, short_rows + copied(every_other)));
  const tensor<double> part     {copied(tall.sliced({0, 0}, {1, 1}, {1050, 1}, all_axes(2, true)))};
  EXPECT(same(part + first_three, part + copied(first_three)) && same(first_three + part, copied(first_three) + part));
  const tensor<double> stepping {wide.sliced({0, 0}, {1, 2}, {1050, 3}, all_axes(2, true))}; // short rows stepping 2
  EXPECT(same(part + stepping, part + copied(stepping)) && same(first_three + copied(first_three), copied(first_three) + copied(first_three)));
  EXPECT(same(one + row, (tensor<double> {{1, 1, 3}, {-1.25, -0.25, 0.75}})));
}

// The triangle of a transpose is the other triangle of its elements as they lie, the diagonal's side reversed; of a
// stack whose matrices' axes lie last, theirs; any other permutation's copied. The bits of the contiguous copy's, every
// diagonal from past one corner to past the other.
TEST(the_triangle_of_a_transpose_is_its_elements_other_triangle)
{
  numerical::buffer<double> values(std::size_t {3} *4 * 9 * 11);
  for (std::size_t i {0}; i < values.size(); ++i) {
    values[i] = i % 5 == 0 ? -0.0 : static_cast<double>(i) - 200.0;
  }
  const tensor<double> stack {{3, 4, 9, 11}, values};
  const auto           same = [](const tensor<double>& x, const tensor<double>& y) {
                                return x.shape() == y.shape() && std::memcmp(x.data().data(), y.data().data(), x.size() * sizeof(double)) == 0;
                              };
  bool                 holds {true};
  for (const tensor<double>& view : {stack.reshaped({108, 11}).transposed(), stack.transposed({1, 0, 3, 2}), stack.transposed({1, 0, 2, 3}),
                                     stack.transposed({0, 2, 1, 3}), stack.reshaped({12, 9, 11}).transposed({1, 2, 0}),
                                     stack.reshaped({12, 9, 11}).transposed({2, 1, 0}), stack.reshaped({12, 9, 11}).transposed({2, 0, 1})}) {
    const tensor<double> copy {view.shape(), view.data()};
    for (const std::ptrdiff_t k : {std::numeric_limits<std::ptrdiff_t>::min(), std::ptrdiff_t {-120}, std::ptrdiff_t {-4}, std::ptrdiff_t {0},
                                   std::ptrdiff_t {3}, std::ptrdiff_t {120}, std::numeric_limits<std::ptrdiff_t>::max()}) {
      for (const bool lower : {true, false}) {
        holds = holds && same(numerical::triangle(view, k, lower), numerical::triangle(copy, k, lower));
      }
    }
  }
  EXPECT(holds);
}

// The functions by a table read a view where its elements lie, row by row (gathered by pieces where its last axis
// steps over elements, copied first where it lies far apart): the same bits, double and float, as of its contiguous
// copy, over rows that a block of the threads begins midway, rows wider than a gathered piece, backwards and transposed;
// and the bits of the kernels over a pointer to the elements (a borrowed NumPy array's).
TEST(the_table_functions_read_a_view_where_it_lies)
{
  using all   = std::vector<bool>;
  const auto drawn = [](shape s) {
                       numerical::buffer<double> values(counting(s).size(), 0.0);
                       for (std::size_t i {0}; i < values.size(); ++i) {
                         values[i] = (static_cast<double>((i * 7919U) % 2001U) * 0.005) - 5.0;
                       }
                       return tensor<double> {std::move(s), std::move(values)};
                     };
  const tensor<double> cube {drawn({60, 50, 43})};
  const tensor<double> wide {drawn({30, 1601})};
  const auto           views = [&](const auto& c, const auto& w) {
                                 using V = std::remove_cvref_t<decltype(c)>;
                                 return std::vector<V> {
                                   c.sliced({0, 0, 0}, {1, 1, 2}, {60, 50, 22}, all(3, true)),                       // a step of 2 along each row
                                   c.sliced({0, 0, 1}, {1, 1, 1}, {60, 50, 42}, all(3, true)),                       // rows of 42 of 43
                                   c.sliced({0, 49, 42}, {1, -3, -1}, {60, 17, 43}, all(3, true)),                   // backwards
                                   c.transposed({2, 0, 1}),                                                          // far apart, laid out
                                   c.transposed({2, 0, 1}).sliced({1, 0, 0}, {2, 1, 1}, {21, 60, 50}, all(3, true)), // far apart, not
                                   w.sliced({0, 0}, {1, 2}, {30, 801}, all(2, true)),                                // rows past a piece
                                 };
                               };
  const auto same_bits = [](const auto& x, const auto& y) {
                           return x.shape() == y.shape() && x.size() == y.size() &&
                                  std::memcmp(x.data().data(), y.data().data(), x.size() * sizeof(x.data()[0])) == 0;
                         };
  // the kernels over elements the caller holds (a NumPy array's, borrowed): the bits of the view's
  const auto of_pointer = [&](const auto& view, const auto& copy) {
                            namespace d = numerical::detail;
                            const auto* const p {copy.data().data()};
                            const auto&       s {copy.shape()};
                            return same_bits(d::exp_of(p, s), numerical::exp(view)) && same_bits(d::expm1_of(p, s), numerical::expm1(view))
                                   && same_bits(d::atan_of(p, s), numerical::atan(view)) && same_bits(d::sinh_of(p, s), numerical::sinh(view))
                                   && same_bits(d::cosh_of(p, s), numerical::cosh(view)) && same_bits(d::exp2_of(p, s), numerical::exp2(view))
                                   && same_bits(d::log1p_of(p, s), numerical::log1p(view)) && same_bits(d::sin_of(p, s), numerical::sin(view))
                                   && same_bits(d::cos_of(p, s), numerical::cos(view)) && same_bits(d::tanh_of(p, s), numerical::tanh(view));
                          };
  const auto holds = [&](const auto& view) {
                       using V = std::remove_cvref_t<decltype(view)>;
                       const V copy {view.shape(), view.data()};
                       using E = std::remove_cvref_t<decltype(view.data()[0])>;
                       return same_bits(numerical::exp(view), numerical::exp(copy)) &&
                              same_bits(numerical::expm1(view), numerical::expm1(copy)) &&
                              same_bits(numerical::atan(view), numerical::atan(copy)) &&
                              same_bits(numerical::sinh(view), numerical::sinh(copy)) &&
                              same_bits(numerical::cosh(view), numerical::cosh(copy)) &&
                              same_bits(numerical::exp2(view), numerical::exp2(copy)) &&
                              same_bits(numerical::log1p(view), numerical::log1p(copy)) &&
                              same_bits(numerical::sin(view), numerical::sin(copy)) &&
                              same_bits(numerical::cos(view), numerical::cos(copy)) &&
                              same_bits(numerical::tanh(view), numerical::tanh(copy)) && of_pointer(view, copy) &&
                              same_bits(numerical::clip(view, E {-0.5}, E {0.5}), numerical::clip(copy, E {-0.5}, E {0.5}));
                     };
  for (const tensor<double>& view : views(cube, wide)) {
    EXPECT(!view.contiguous() && holds(view));
  }
  const tensor<float> cube32 {numerical::detail::converted<float>(cube)};
  const tensor<float> wide32 {numerical::detail::converted<float>(wide)};
  for (const tensor<float>& view : views(cube32, wide32)) {
    EXPECT(!view.contiguous() && holds(view));
  }
  const tensor<double> whole {cube.sliced({0, 0, 0}, {1, 1, 1}, {60, 50, 43}, all(3, true))}; // contiguous as it lies
  EXPECT(same_bits(numerical::exp(whole), numerical::exp(tensor<double> {whole.shape(), whole.data()})));
}

// A slice is a view past sliced_copy_most elements: each axis's range read where the elements lie, from its start by
// its step (backwards for a negative one), an integer index dropping its axis; copied out in row-major order on first
// need, by rows on the threads past the direct reads. A slice of a transpose and a transpose of a slice compose. A few
// elements, an empty range and a rank-0 result are copied at once; a range out of its axis or a dropped axis of more
// than one index is refused.
TEST(a_slice_is_a_view)
{
  using steps = std::vector<std::ptrdiff_t>;
  const tensor<double> cube {counting({20, 30, 40})};
  // the elements of `slice` at every index against cube's at the index `at` maps it to
  const auto reads = [&](const tensor<double>& slice, const auto& at) {
                       bool                     same {slice.size() == slice.data().size()};
                       std::vector<std::size_t> index(slice.rank(), 0);
                       for (std::size_t flat {0}; same && flat < slice.size(); ++flat) {
                         std::size_t rest {flat};
                         for (std::size_t axis {slice.rank()}; axis-- > 0;) {
                           index[axis] = rest % slice.shape()[axis];
                           rest       /= slice.shape()[axis];
                         }
                         same = slice.data()[flat] == cube(at(index)) && slice(index) == cube(at(index));
                       }
                       return same;
                     };
  // cube[1:20:2, ::-2, 3]: rows 1 to 19, columns 29 down to 1, the depth 3 dropped
  const tensor<double> part {cube.sliced({1, 29, 3}, {2, -2, 1}, {10, 15, 1}, std::vector<bool> {true, true, false})};
  EXPECT(part.shape() == (shape {10, 15}) && part.contiguous()); // 150 elements: copied at once
  EXPECT(reads(part, [](const std::vector<std::size_t>& i) { return std::vector<std::size_t> {1 + (2 * i[0]), 29 - (2 * i[1]), 3}; }));
  // cube[1:20:2, ::-1, 3:40:3]: 3900 elements, a view
  const tensor<double> view {cube.sliced({1, 29, 3}, {2, -1, 3}, {10, 30, 13}, std::vector<bool> {true, true, true})};
  EXPECT(!view.contiguous() && view.shape() == (shape {10, 30, 13}) && view.size() == 3900);
  EXPECT(view.strides() == (steps {2400, -40, 3}) && view.origin() == cube.data().data() + 1200 + 1160 + 3);
  EXPECT(reads(view, [](const std::vector<std::size_t>& i) { return std::vector<std::size_t> {1 + (2 * i[0]), 29 - i[1], 3 + (3 * i[2])}; }));
  const tensor<double> turned              {view.transposed({2, 0, 1})};                                                                           // a transpose of a slice
  EXPECT(turned(std::vector<std::size_t> {12, 9, 0}) == cube(std::vector<std::size_t> {19, 29, 39}));
  const tensor<double> back                {cube.transposed().sliced({39, 0, 0}, {-1, 1, 1}, {40, 30, 20}, std::vector<bool> {true, true, true})}; // a slice of a transpose
  EXPECT(reads(back, [](const std::vector<std::size_t>& i) { return std::vector<std::size_t> {i[2], i[1], 39 - i[0]}; }));
  const tensor<double> rows                {cube.sliced({1, 0, 0}, {1, 1, 1}, {2, 30, 40}, std::vector<bool> {true, true, true})};                 // contiguous, from an offset
  EXPECT(!rows.contiguous() && rows.data() == (numerical::buffer<double>(cube.data().begin() + 1200, cube.data().begin() + 3600)));
  const tensor<double> first               {cube.sliced({0, 0, 0}, {1, 1, 1}, {2, 30, 40}, std::vector<bool> {true, true, true})};                 // from the first, not all
  EXPECT(!first.contiguous() && first.data() == (numerical::buffer<double>(cube.data().begin(), cube.data().begin() + 2400)));
  const tensor<double> rows_from_the_first {counting({400, 300}).sliced({0, 0}, {1, 1}, {399, 300}, std::vector<bool> {true, true})};
  EXPECT(rows_from_the_first.data().size() == 119700 && rows_from_the_first.data()[119699] == 119699.0);                                           // past the direct reads
  const tensor<double> whole               {cube.sliced({0, 0, 0}, {1, 1, 1}, {20, 30, 40}, std::vector<bool> {true, true, true})};                // the whole: as it lies
  EXPECT(whole.contiguous() && whole.data().data() == cube.data().data());
  const tensor<double> one                 {cube.sliced({2, 3, 4}, {1, 1, 1}, {1, 1, 1}, std::vector<bool> {false, false, false})};                // every axis dropped
  EXPECT(one.rank() == 0 && one.data() == (numerical::buffer<double> {cube(std::vector<std::size_t> {2, 3, 4})}));
  const tensor<double> none                {cube.sliced({0, 2, 0}, {1, 1, 1}, {20, 0, 40}, std::vector<bool> {true, true, true})};
  EXPECT(none.contiguous() && none.shape() == (shape {20, 0, 40}) && none.size() == 0 && none.data().empty());
  const tensor<double> line                {counting({1000})};     // a rank-1 view, read backwards by threes
  EXPECT(line.sliced({999}, {-3}, {334}, std::vector<bool> {true}).data()[333] == 0.0);
  const tensor<double> wide                {counting({300, 200})}; // past the direct reads: by rows on the threads
  const tensor<double> every               {wide.sliced({299, 1}, {-2, 3}, {150, 66}, std::vector<bool> {true, true})};
  bool                 same                {every.data().size() == std::size_t {150} *66};
  for (std::size_t i {0}; i < 150; ++i) {
    for (std::size_t j {0}; j < 66; ++j) {
      same = same && every.data()[(i * 66) + j] == wide(std::vector<std::size_t> {299 - (2 * i), 1 + (3 * j)});
    }
  }
  EXPECT(same);
  const tensor<double> wider  {counting({400, 300})}; // a reversal of the whole: no permutation, read by rows
  EXPECT(wider.sliced({399, 0}, {-1, 1}, {400, 300}, std::vector<bool> {true, true}).data()[0] == wider(std::vector<std::size_t> {399, 0}));
  const std::vector<bool> all {true, true, true};
  EXPECT_THROWS((void)cube.sliced({0, 0}, {1, 1}, {4, 5}, std::vector<bool> {true, true}), numerical::dimension_error);
  EXPECT_THROWS((void)cube.sliced({0, 0, 0}, {1, 1}, {4, 5, 6}, all), numerical::dimension_error);
  EXPECT_THROWS((void)cube.sliced({0, 0, 0}, {1, 1, 1}, {4, 5}, all), numerical::dimension_error);
  EXPECT_THROWS((void)cube.sliced({0, 0, 0}, {1, 1, 1}, {4, 5, 6}, std::vector<bool> {true, true}), numerical::dimension_error);
  EXPECT_THROWS((void)cube.sliced({0, 0, 0}, {1, 1, 1}, {2, 5, 6}, std::vector<bool> {false, true, true}), numerical::dimension_error);
  EXPECT_THROWS((void)cube.sliced({20, 0, 0}, {1, 1, 1}, {1, 5, 6}, all), std::out_of_range);
  EXPECT_THROWS((void)cube.sliced({0, 0, 0}, {1, 1, 2}, {4, 5, 21}, all), std::out_of_range);
  EXPECT_THROWS((void)cube.sliced({0, 0, 1}, {1, 1, -2}, {4, 5, 2}, all), std::out_of_range);
}

TEST(an_element_wise_operation_reads_a_view_where_it_lies)
{
  using keep = std::vector<bool>;
  // each operation on a view against the same on its row-major copy: every element alone, the same bits
  const auto copy_of = [](const tensor<double>& t) { return tensor<double> {t.shape(), numerical::buffer<double>(t.data().begin(), t.data().end())}; };
  const auto seventh = [](double x) { return (x / 7.0) + 0.25; };
  // x op y against the same on the copies, for each of + − × ÷: each operation one instance that meets every pair
  const auto same_four = [&](const tensor<double>& x, const tensor<double>& y) {
                           const tensor<double> cx {copy_of(x)};
                           const tensor<double> cy {copy_of(y)};
                           const auto           same {[](const tensor<double>& p, const tensor<double>& q) {
                                                        return p.shape() == q.shape() && p.data() == q.data();
                                                      }};
                           return same(x + y, cx + cy) && same(x - y, cx - cy) && same(x * y, cx * cy) && same(x / y, cx / cy);
                         };
  const tensor<double>              wide  {numerical::detail::parallel_map(counting({300, 200}), seventh)};
  const tensor<double>              line  {numerical::detail::parallel_map(counting({40000}), seventh)};
  const tensor<double>              cube  {numerical::detail::parallel_map(counting({20, 30, 40}), seventh)};
  const std::vector<tensor<double>> views {
    wide.sliced({0, 0}, {2, 1}, {150, 200}, keep {true, true}),        // rows apart
    wide.sliced({299, 199}, {-1, -1}, {300, 200}, keep {true, true}),  // both axes reversed
    wide.sliced({0, 10}, {1, 1}, {300, 100}, keep {true, true}),       // a band of columns
    wide.sliced({0, 0}, {1, 3}, {300, 67}, keep {true, true}),         // every third column
    wide.sliced({1, 0}, {2, 1}, {149, 8}, keep {true, true}),          // short rows apart
    wide.transposed(),                                                 // copied out first
    line.sliced({39999}, {-1}, {40000}, keep {true}),                  // a long row reversed: by pieces
    line.sliced({1}, {2}, {19999}, keep {true}),                       // every other element
    cube.sliced({1, 29, 0}, {2, -1, 1}, {9, 30, 40}, keep {true, true, true}),
  };
  const tensor<double> scalar {shape {}, {2.5}};
  for (const tensor<double>& view : views) {
    EXPECT(!view.contiguous());
    const tensor<double> copy {copy_of(view)};
    const tensor<double> row  {numerical::detail::parallel_map(counting({view.shape().back()}), seventh)};
    EXPECT(same_four(view, view) && same_four(view, copy) && same_four(copy, view));
    EXPECT(same_four(view, scalar) && same_four(scalar, view) && same_four(view, row) && same_four(row, view));
    const tensor<double> mapped {numerical::detail::parallel_map(view, seventh)};
    EXPECT(mapped.shape() == view.shape() && mapped.data() == numerical::detail::parallel_map(copy, seventh).data());
    if (view.rank() == 2) { // against a column: one element a row
      const std::size_t    rows   {view.shape()[0]};
      const tensor<double> column {numerical::detail::parallel_map(counting({rows, 1}), seventh)};
      EXPECT(same_four(view, column) && same_four(column, view));
      EXPECT(same_four(view, view.sliced({0, 0}, {1, 1}, {rows, 1}, keep {true, true}))); // its own first column
    }
  }
  // a column of more than 512 elements, a view whose last axis holds one: copied out, an axis of one element
  const tensor<double> tall         {numerical::detail::parallel_map(counting({2000, 3}), seventh)};
  const tensor<double> middle       {tall.sliced({0, 1}, {1, 1}, {2000, 1}, keep {true, true})};
  EXPECT(!middle.contiguous() && same_four(middle, middle) && same_four(middle, tall) && same_four(tall, middle));
  const tensor<double> rows_of_cube {cube.sliced({0, 0, 0}, {1, 1, 1}, {20, 30, 1}, keep {true, true, true})};
  EXPECT(same_four(cube, rows_of_cube) && same_four(rows_of_cube, cube)); // broadcast along the last axis
}

// A view's first data() calls at once, from threads of their own and from a parallel_for's bodies (a run nested in the
// copy's own parallel_for takes its chunks itself): one copy, the one every call returns.
TEST(a_view_is_copied_out_once_whatever_the_threads)
{
  const tensor<double>       big  {counting({300, 200})};
  const tensor<double>       view {big.transposed()};
  std::vector<const double*> seen(8, nullptr);
  {
    std::vector<std::thread> threads;
    for (std::size_t k {0}; k < seen.size(); ++k) {
      threads.emplace_back([&, k] { seen[k] = view.data().data(); });
    }
    for (std::thread& thread : threads) {
      thread.join();
    }
  }
  EXPECT(std::ranges::all_of(seen, [&](const double* p) { return p == seen[0]; }));
  EXPECT(view(std::vector<std::size_t> {199, 299}) == view.data()[(199 * 300) + 299]);
  const tensor<double> slab   {counting({150, 1, 120})}; // past the direct reads, an axis of extent 1 among the others
  const tensor<double> turned {slab.transposed({2, 1, 0})};
  bool                 same   {true};
  for (std::size_t k {0}; k < 120; ++k) {
    for (std::size_t i {0}; i < 150; ++i) {
      same = same && turned.data()[(k * 150) + i] == slab(std::vector<std::size_t> {i, 0, k});
    }
  }
  EXPECT(same && !turned.contiguous());
  const tensor<double>       other {big.transposed()};
  std::vector<const double*> nested(64, nullptr);
  numerical::detail::parallel_for(nested.size(), std::size_t {1} << 20, [&](std::size_t k) { nested[k] = other.data().data(); });
  EXPECT(std::ranges::all_of(nested, [&](const double* p) { return p == nested[0]; }) && other.data() == view.data());
}

// broadcast_to repeats a tensor along its axes of extent 1 and new leading ones: rows copied, or filled with their one
// element where the last axis broadcasts; an empty result, a scalar, and a shape that does not fit.
TEST(broadcast_to_as_numpy)
{
  const tensor<double> row    {{3}, {1.0, 2.0, 3.0}};
  EXPECT(numerical::broadcast_to(row, {2, 3}).data() == (numerical::buffer<double> {1.0, 2.0, 3.0, 1.0, 2.0, 3.0}));
  const tensor<double> column {{2, 1}, {4.0, 5.0}};
  EXPECT(numerical::broadcast_to(column, {2, 3}).data() == (numerical::buffer<double> {4.0, 4.0, 4.0, 5.0, 5.0, 5.0}));
  EXPECT(numerical::broadcast_to(column, {3, 2, 2}).shape() == (std::vector<std::size_t> {3, 2, 2}));
  EXPECT(numerical::broadcast_to(tensor<double> {{}, {7.0}}, {}).data() == (numerical::buffer<double> {7.0}));
  EXPECT(numerical::broadcast_to(tensor<double> {{1}, {7.0}}, {0}).size() == 0);
  EXPECT_THROWS((void)numerical::broadcast_to(row, {2, 4}), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::broadcast_to(column, {2}), numerical::dimension_error);
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
  // Unfused, as NumPy's wheels other than macOS arm64's: each product rounded before its sum.
  const tensor<double> apart   {numerical::arange(-3.5, 7.2, 0.3, false)};
  bool                 rounded {apart.size() == 36U && apart.data()[0] == -3.5 && apart.data()[1] == -3.5 + 0.3};
  for (std::size_t i {2}; i < apart.size(); ++i) {
    const double product {std::fma(static_cast<double>(i), delta, -0.0)};
    rounded = rounded && apart.data()[i] == -3.5 + product;
  }
  EXPECT(rounded);
  EXPECT(!(apart == stepped));
  EXPECT(numerical::arange(1.0, 1.0, 1.0).size() == 0U);
  EXPECT(numerical::arange(2.0, 1.0, 1.0).size() == 0U);
  // The first element is the start itself, a negative zero's sign kept, as NumPy's: −0 + 0·delta would be +0.
  const tensor<double> from_zero {numerical::arange(-0.0, 0.5, 1.0)};
  EXPECT(from_zero.size() == 1U && std::signbit(from_zero.data()[0]));
  EXPECT(std::signbit(numerical::arange(-0.0, 3.0, 1.5).data()[0]));
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
      ok = ok && numpy_power(a, C {R {0}, R {1}}, fused).imag() != R {0}; // an exponent of i, by the C library's cpow
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
      // any other exponent by the C library's cpow (library_power: std::pow but under Apple, whose libc++ computes
      // its own): a fraction, one of 100 or more, an imaginary part (read back through volatile parts: GCC folds
      // std::pow of constants exactly, where a call rounds as the C library's cpow does)
      for (const C& exponent : {C {R {0.5}, R {0}}, C {R {100}, R {0}}, C {R {-100}, R {0}}, C {R {2}, R {1}}, C {nan, R {0}}}) {
        volatile R real_part      {exponent.real()};
        volatile R imaginary_part {exponent.imag()};
        volatile R base_real      {a.real()};
        volatile R base_imaginary {a.imag()};
        const C    base           {static_cast<R>(base_real), static_cast<R>(base_imaginary)};
        const C    b              {static_cast<R>(real_part), static_cast<R>(imaginary_part)};
        const C    want           {numerical::detail::library_power(base, b)};
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
      const numerical::detail::numpy_loops loops {.products = fused, .scalar_products = !fused, .quotients = !fused,
                                                  .reductions = !fused, .late_reductions = false};
      ok                                                      = ok && numerical::detail::numpy_square(t, loops).data()[1] == numpy_multiply(wide, wide, fused);
      ok                                                      = ok && numerical::detail::numpy_reciprocal(t, loops).data()[1] == numpy_reciprocal(wide, !fused);
      ok                                                      = ok && numerical::detail::numpy_power(t, three, loops).data()[0] == numpy_power(a, C {R {3}, R {0}}, !fused);
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

// The offset diagonal of each matrix, a view of it past a few elements: the main one, above and below, a rectangle's,
// a stack's, a transpose's and a slice's, an offset past the matrix or no matrix (empty), and a vector refused.
TEST(diagonal_reads_each_matrix_s_offset_diagonal)
{
  const tensor<double> grid {counting({3, 4})};
  EXPECT(numerical::diagonal(grid, 0).data() == (numerical::buffer<double> {0.0, 5.0, 10.0}));
  EXPECT(numerical::diagonal(grid, 1).data() == (numerical::buffer<double> {1.0, 6.0, 11.0}));
  EXPECT(numerical::diagonal(grid, -1).data() == (numerical::buffer<double> {4.0, 9.0}));
  EXPECT(numerical::diagonal(grid, 3).data() == (numerical::buffer<double> {3.0}));
  const tensor<double> stack {counting({2, 2, 3})};
  const tensor<double> both  {numerical::diagonal(stack, 0)};
  EXPECT(both.shape() == (shape {2, 2}) && both.data() == (numerical::buffer<double> {0.0, 4.0, 6.0, 10.0}));
  EXPECT(numerical::diagonal(grid, 4).shape() == (shape {0}));
  EXPECT(numerical::diagonal(grid, -3).shape() == (shape {0}));
  EXPECT_THROWS((void)numerical::diagonal(counting({3}), 0), numerical::dimension_error);
  // past sliced_copy_most elements a view, of a view too: a transposed stack's, from a slice's start
  const tensor<double> cube    {counting({3, 700, 700})};
  const tensor<double> flipped {numerical::diagonal(cube.transposed({0, 2, 1}), 2)};
  bool                 along   {flipped.shape() == (shape {3, 698}) && !flipped.contiguous()};
  for (std::size_t b {0}; b < 3; ++b) {
    for (std::size_t i {0}; i < 698; ++i) {
      along = along && flipped({b, i}) == cube({b, i + 2, i}) && flipped.data()[(b * 698) + i] == cube({b, i + 2, i});
    }
  }
  EXPECT(along);
  const tensor<double> part {cube.sliced({1, 5, 0}, {1, 1, 2}, {2, 600, 350}, std::vector<bool> {true, true, true})};
  const tensor<double> down {numerical::diagonal(part, -3)};
  bool                 kept {down.shape() == (shape {2, 350}) && !down.contiguous()};
  for (std::size_t b {0}; b < 2; ++b) {
    for (std::size_t i {0}; i < 350; ++i) {
      kept = kept && down({b, i}) == cube({1 + b, 5 + 3 + i, 2 * i});
    }
  }
  EXPECT(kept);
  const tensor<double> none {numerical::diagonal(counting({0, 5, 5}), 1)}; // no matrix: empty, its shape kept
  EXPECT(none.shape() == (shape {0, 4}) && none.size() == 0 && none.contiguous());
}

// A contiguous extreme past four runs of 2¹⁵ elements, each run on a thread: the largest in the last run, the
// smallest in the first, a NaN in a middle run, and zeros of both signs alone; the same answers on one thread.
TEST(a_long_extreme_is_the_same_on_the_threads)
{
  const std::size_t         count {(5 * (std::size_t {1} << 15U)) + 7};
  numerical::buffer<double> values(count);
  for (std::size_t i {0}; i < count; ++i) {
    values[i] = std::sin(static_cast<double>(i));
  }
  values[count - 3] = 2.0;
  values[11]        = -2.0;
  const tensor<double>      line     {{count}, values};
  numerical::buffer<double> with_nan {values};
  with_nan[(3 * (std::size_t {1} << 15U)) + 5] = std::numeric_limits<double>::quiet_NaN();
  const tensor<double>      nan_line {{count}, with_nan};
  numerical::buffer<double> zeros(count, -0.0);
  zeros[count / 2] = 0.0;
  const tensor<double> zero_line {{count}, zeros};
  const std::size_t    threads   {numerical::max_threads()};
  for (const std::size_t limit : {threads, std::size_t {1}}) {
    numerical::set_max_threads(limit);
    EXPECT(numerical::max(line) == 2.0 && numerical::min(line) == -2.0);
    EXPECT(std::isnan(numerical::max(nan_line)) && std::isnan(numerical::min(nan_line)));
    EXPECT(!std::signbit(numerical::max(zero_line)) && std::signbit(numerical::min(zero_line)));
  }
  numerical::set_max_threads(0); // the default again
}

// numpy.maximum and numpy.minimum, broadcast: a NaN wins (the first's where both are), the zeros ordered as IEEE
// 754-2019 (+0 over −0 for maximum, −0 under +0 for minimum, whatever their order); float32 alike.
TEST(maximum_and_minimum_order_nans_and_zeros_as_ieee)
{
  const double         nan     {std::numeric_limits<double>::quiet_NaN()};
  const tensor<double> a       {{2, 4}, {1.0, nan, 0.0, -0.0, 5.0, nan, 3.0, -0.0}};
  const tensor<double> b       {{2, 4}, {2.0, 1.0, -0.0, 0.0, 4.0, -nan, 3.0, -0.0}};
  const tensor<double> larger  {numerical::maximum(a, b)};
  const tensor<double> smaller {numerical::minimum(a, b)};
  EXPECT(larger.data()[0] == 2.0 && std::isnan(larger.data()[1]) && larger.data()[4] == 5.0 && larger.data()[6] == 3.0);
  EXPECT(smaller.data()[0] == 1.0 && std::isnan(smaller.data()[1]) && smaller.data()[4] == 4.0 && smaller.data()[6] == 3.0);
  EXPECT(!std::signbit(larger.data()[2]) && !std::signbit(larger.data()[3]) && std::signbit(larger.data()[7]));
  EXPECT(std::signbit(smaller.data()[2]) && std::signbit(smaller.data()[3]) && std::signbit(smaller.data()[7]));
  EXPECT(!std::signbit(larger.data()[5]) && !std::signbit(smaller.data()[5])); // the first NaN, +NaN, not b's −NaN
  const tensor<double> right {{3}, {nan, 2.0, 0.0}};                           // b's NaN where a is a number
  const tensor<double> row   {{1, 3}, {1.0, 1.0, 1.0}};
  EXPECT(std::isnan(numerical::maximum(row, right).data()[0]) && std::isnan(numerical::minimum(row, right).data()[0]));
  EXPECT(numerical::maximum(row, right).shape() == (shape {1, 3}) && numerical::minimum(row, right).data()[2] == 0.0);
  const tensor<float> f      {{2}, {1.0F, -2.0F}};
  const tensor<float> g      {{2}, {0.5F, -1.0F}};
  EXPECT(numerical::maximum(f, g).data()[1] == -1.0F && numerical::minimum(f, g).data()[0] == 0.5F);
  const tensor<double> bound {std::vector<std::size_t> {}, {0.5}};                // a number: mapped over a
  EXPECT(numerical::maximum(a, bound).data()[0] == 1.0 && numerical::maximum(a, bound).data()[2] == 0.5);
  EXPECT(numerical::minimum(a, bound).data()[0] == 0.5 && std::isnan(numerical::minimum(a, bound).data()[1]));
  EXPECT_THROWS((void)numerical::maximum(counting({2}), counting({3})), numerical::dimension_error);
}

// clip between two numbers in one pass: the bits of minimum(maximum(a, low), high), a NaN kept, the zeros ordered as
// those two order them, a NaN bound winning; float32 and int64 alike.
// A line's extreme: reals contiguous by vectors, strided walked (a NaN wins either way), integers walked; an empty
// line refused for both.
TEST(a_line_extreme_strided_or_not_and_of_integers)
{
  namespace d = numerical::detail;
  const double              nan   {std::numeric_limits<double>::quiet_NaN()};
  const std::vector<double> reals {3.0, nan, -1.0, 7.0, 2.0};
  EXPECT(d::extreme(reals.data(), 3, 2, std::less<double> {}) == -1.0 && d::extreme(reals.data(), 3, 2, std::greater<double> {}) == 3.0);
  EXPECT(std::isnan(d::extreme(reals.data(), 2, 1, std::less<double> {})) && std::isnan(d::extreme(reals.data() + 1, 2, 3, std::greater<double> {})));
  const std::vector<std::int64_t> integers {4, -2, 9};
  EXPECT(d::extreme(integers.data(), 3, 1, std::less<std::int64_t> {}) == -2 && d::extreme(integers.data(), 2, 2, std::greater<std::int64_t> {}) == 9);
  EXPECT_THROWS((void)d::extreme(reals.data(), 0, 1, std::less<double> {}), numerical::dimension_error);
  EXPECT_THROWS((void)d::extreme(integers.data(), 0, 1, std::less<std::int64_t> {}), numerical::dimension_error);
}

TEST(clip_is_minimum_of_maximum_in_one_pass)
{
  const double         nan {std::numeric_limits<double>::quiet_NaN()};
  const tensor<double> a   {{8}, {-3.0, -0.0, 0.0, 0.25, nan, 7.0, -std::numeric_limits<double>::infinity(), 1e-310}};
  const auto           same_bits = [](const tensor<double>& x, const tensor<double>& y) {
                                     return std::memcmp(x.data().data(), y.data().data(), x.size() * sizeof(double)) == 0;
                                   };
  for (const auto& [low, high] : std::vector<std::pair<double, double>> {{-1.0, 1.0}, {0.0, 0.0}, {-0.0, -0.0}, {-0.0, 0.0}, {0.0, -0.0}, {2.0, -2.0}, {nan, 1.0}, {-1.0, nan}}) {
    const tensor<double> lower {std::vector<std::size_t> {}, {low}};
    const tensor<double> upper {std::vector<std::size_t> {}, {high}};
    EXPECT(same_bits(numerical::clip(a, low, high), numerical::minimum(numerical::maximum(a, lower), upper)));
  }
  const tensor<float> f {{4}, {-2.0F, -0.0F, 0.5F, std::numeric_limits<float>::quiet_NaN()}};
  const tensor<float> c {numerical::clip(f, 0.0F, 1.0F)};
  EXPECT(c.data()[0] == 0.0F && !std::signbit(c.data()[1]) && c.data()[2] == 0.5F && std::isnan(c.data()[3]));
  EXPECT(std::isnan(numerical::clip(f, std::numeric_limits<float>::quiet_NaN(), 1.0F).data()[0]));
  const tensor<std::int64_t> n {{4}, {-9, 0, 4, 9}};
  EXPECT(numerical::clip(n, std::int64_t {-2}, std::int64_t {5}).data() == (numerical::buffer<std::int64_t> {-2, 0, 4, 5}));
  EXPECT(numerical::clip(n, std::int64_t {3}, std::int64_t {1}).data() == (numerical::buffer<std::int64_t> {1, 1, 1, 1}));
}

// einsum of one operand's diagonal: …ii->…i the diagonals, …ii->… their sums by pairwise halving, the same bits as
// sum() of the diagonal (trace's); an empty matrix's trace 0; specs it does not take go the general way.
TEST(einsum_takes_a_diagonal_and_its_pairwise_sum)
{
  const tensor<double> m     {counting({300, 300})};
  const tensor<double> along {numerical::einsum("ii->i", m)};
  EXPECT(along.shape() == (shape {300}) && along.data()[7] == 7.0 * 301.0);
  EXPECT(numerical::einsum("ii->", m).data()[0] == numerical::sum(along));
  const tensor<double> stack {counting({3, 4, 4})};
  const tensor<double> sums  {numerical::einsum("aii->a", stack)};
  EXPECT(sums.shape() == (shape {3}) && sums.data()[1] == (16.0 + 21.0 + 26.0 + 31.0));
  EXPECT(numerical::einsum("aii->ai", stack).shape() == (shape {3, 4}));
  EXPECT(numerical::einsum("ii->", numerical::zeros<double>({0, 0})).data()[0] == 0.0);
  EXPECT(numerical::einsum("aii->ia", stack).shape() == (shape {4, 3}));                                 // not in order: the general loop
  EXPECT(numerical::einsum("aai->i", counting({2, 2, 3})).data()[0] == 0.0 + 9.0);
  EXPECT(numerical::einsum("iii->", counting({2, 2, 2})).data()[0] == 0.0 + 7.0);                        // a label thrice: the general loop
  EXPECT(numerical::einsum("->", tensor<double> {std::vector<std::size_t> {}, {2.5}}).data()[0] == 2.5); // rank 0
}

// prod in order (an empty one 1), along an axis; cumsum and cumprod in order along an axis; sort along an axis,
// ascending with NaNs last and equal elements in their order, complex numbers by real then imaginary part.
// NumPy's pairwise sum of complex numbers counts their parts: four partial sums and blocks of 64 where a real's are
// eight and 128, its halves a multiple of four long. Integers sum exactly in any order: the tree's every branch runs,
// whole sums and columns alike (the bits against NumPy's are the Python suite's).
TEST(complex_sums_take_numpys_tree)
{
  using cplx = std::complex<double>;
  for (const std::size_t count : {3UL, 7UL, 64UL, 65UL, 200UL, 70000UL, 300000UL}) {
    numerical::buffer<cplx> values(count);
    for (std::size_t k {0}; k < count; ++k) {
      values[k] = cplx {static_cast<double>(k % 5), -static_cast<double>(k % 3)};
    }
    double real      {0.0};
    double imaginary {0.0};
    for (const cplx& z : values) {
      real      += z.real();
      imaginary += z.imag();
    }
    const tensor<cplx> line   {{count}, numerical::buffer<cplx>(values)};
    EXPECT(numerical::sum(line) == cplx(real, imaginary));
    const tensor<cplx> column {{count, 2}, numerical::buffer<cplx>(2 * count, cplx {1.0, -1.0})};
    EXPECT(numerical::sum(column, 0).data()[1] == cplx(static_cast<double>(count), -static_cast<double>(count)));
  }
  EXPECT(numerical::detail::pairwise_tree<cplx>::half(100) == 48 && numerical::detail::pairwise_tree<double>::half(100) == 48);
  EXPECT(numerical::detail::pairwise_tree<cplx>::half(200) == 100 && numerical::detail::pairwise_tree<double>::half(200) == 96);
}

// argmax and argmin along an axis something follows walk bands of columns row by row: each column's first winner
// (its first NaN at once, else the first of equals), across a band's edge (300 columns: 256 and 44), as a line's walk.
TEST(arg_extremes_of_columns_by_bands)
{
  const double              nan   {std::numeric_limits<double>::quiet_NaN()};
  const std::size_t         rows  {4};
  const std::size_t         width {300};
  numerical::buffer<double> values(rows * width);
  for (std::size_t k {0}; k < rows; ++k) {
    for (std::size_t w {0}; w < width; ++w) {
      values[(k * width) + w] = static_cast<double>((w + (k * 7)) % 5);
    }
  }
  values[(0 * width) + 3]  = nan; // a NaN first: it wins at once
  values[(2 * width) + 10] = nan; // a NaN later: it wins over the larger ones before it
  values[(3 * width) + 10] = nan;
  const tensor<double>       t        {{rows, width}, numerical::buffer<double>(values)};
  const tensor<std::int64_t> largest  {numerical::argmax(t, 0)};
  const tensor<std::int64_t> smallest {numerical::argmin(t, 0)};
  bool                       ok       {true};
  for (std::size_t w {0}; w < width; ++w) {
    std::size_t most  {0};
    std::size_t least {0};
    bool        found {false};
    for (std::size_t k {0}; k < rows && !found; ++k) {
      const double x {values[(k * width) + w]};
      if (std::isnan(x)) {
        most         = least = k;
        found        = true;
      }
      else {
        most  = x > values[(most * width) + w] ? k : most;
        least = x < values[(least * width) + w] ? k : least;
      }
    }
    ok = ok && std::cmp_equal(largest.data()[w], most) && std::cmp_equal(smallest.data()[w], least);
  }
  EXPECT(ok);
  const tensor<std::int64_t> integers {{2, 3}, {3, 1, 2, 3, 4, 2}};
  EXPECT(numerical::argmax(integers, 0).data() == (numerical::buffer<std::int64_t> {0, 1, 0}));
  const tensor<double> lines          {{2, 3}, {1.0, 3.0, 3.0, nan, 0.0, 5.0}}; // along the last axis: each line its own walk
  EXPECT(numerical::argmax(lines, 1).data() == (numerical::buffer<std::int64_t> {1, 0}));
  EXPECT(numerical::argmin(lines, 1).data() == (numerical::buffer<std::int64_t> {0, 0}));
}

// take_along_axis reads each index's element along the axis, the other axes matched; correlate sums each lag's terms
// in v's order (a complex product NumPy's, conjugating v; an integer's wrapped), over the lags asked alone.
TEST(take_along_axis_and_correlate)
{
  const tensor<double>       t       {{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  const tensor<std::int64_t> rows    {{2, 2}, {2, 0, 1, 1}};
  EXPECT(numerical::take_along_axis(t, rows, 1).data() == (numerical::buffer<double> {3.0, 1.0, 5.0, 5.0}));
  const tensor<std::int64_t> columns {{1, 3}, {1, 0, 1}};
  EXPECT(numerical::take_along_axis(t, columns, 0).data() == (numerical::buffer<double> {4.0, 2.0, 6.0}));
  EXPECT(numerical::take_along_axis(numerical::zeros<double>({0, 3}), tensor<std::int64_t> {{0, 2}, numerical::buffer<std::int64_t> {}}, 1).size() == 0);
  EXPECT_THROWS((void)numerical::take_along_axis(t, tensor<std::int64_t> {{2, 1}, {3, 0}}, 1), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::take_along_axis(t, tensor<std::int64_t> {{2, 1}, {-1, 0}}, 1), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::take_along_axis(t, tensor<std::int64_t> {{3, 1}, {0, 0, 0}}, 1), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::take_along_axis(t, tensor<std::int64_t> {{2}, {0, 0}}, 1), numerical::dimension_error);
  const tensor<double> a {{3}, {1.0, 2.0, 3.0}};
  const tensor<double> v {{2}, {0.5, 1.0}};
  EXPECT(numerical::correlate(a, v, -1, 4).data() == (numerical::buffer<double> {1.0, 2.5, 4.0, 1.5}));
  EXPECT(numerical::correlate(a, v, 0, 2).data() == (numerical::buffer<double> {2.5, 4.0}));
  using cplx = std::complex<double>;
  const tensor<cplx> z           {{1}, {cplx {0.0, 1.0}}};
  EXPECT(numerical::correlate(z, z, 0, 1).data()[0] == cplx(1.0, 0.0)); // i · conj(i)
  const tensor<std::int64_t> big {{2}, {std::numeric_limits<std::int64_t>::max(), 1}};
  const tensor<std::int64_t> one {{1}, {2}};
  EXPECT(numerical::correlate(big, one, 0, 2).data()[0] == -2);         // wrapped, as NumPy's
  // the lags by blocks where v lies within a, one by one at the ends: each sum the bits of its own in v's order
  std::vector<double> long_a(45);
  std::vector<double> long_v(6);
  for (std::size_t i {0}; i < long_a.size(); ++i) {
    long_a[i] = std::sin(static_cast<double>(i) * 1.7) * 1e3;
  }
  for (std::size_t i {0}; i < long_v.size(); ++i) {
    long_v[i] = std::cos(static_cast<double>(i) * 0.9) / 3.0;
  }
  const tensor<double> along {{long_a.size()}, numerical::buffer<double>(long_a.begin(), long_a.end())};
  const tensor<double> vlong {{long_v.size()}, numerical::buffer<double>(long_v.begin(), long_v.end())};
  const auto           full  {numerical::correlate(along, vlong, -5, 50)};
  bool                 same  {true};
  for (std::ptrdiff_t k {-5}; k < 45; ++k) {
    double sum {};
    for (std::ptrdiff_t j {std::max<std::ptrdiff_t>(0, -k)}; j < std::min<std::ptrdiff_t>(6, 45 - k); ++j) {
      const double term {long_a[static_cast<std::size_t>(j + k)] * long_v[static_cast<std::size_t>(j)]};
      sum = sum + term;
    }
    same = same && std::bit_cast<std::uint64_t>(sum) == std::bit_cast<std::uint64_t>(full.data()[static_cast<std::size_t>(k + 5)]);
  }
  EXPECT(same);
}

// take reads each index once (negative ones from the end; modulo the extent or clipped as asked) and copies the
// elements along the axis, the result shaped by the indices there; NumPy's errors, NumPy's exemptions.
TEST(take_by_mode)
{
  using numerical::take_mode;
  const tensor<double>       t       {{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  const tensor<std::int64_t> some    {{2, 2}, {2, -3, -1, 0}};
  const tensor<double>       picked  {numerical::take(t, some, 1, take_mode::raise)};
  EXPECT(picked.shape() == (std::vector<std::size_t> {2, 2, 2}));
  EXPECT(picked.data() == (numerical::buffer<double> {3.0, 1.0, 3.0, 1.0, 6.0, 4.0, 6.0, 4.0}));
  const tensor<std::int64_t> far     {{4}, {7, -4, 1, -1}};
  EXPECT(numerical::take(t, far, 1, take_mode::wrap).data() == (numerical::buffer<double> {2.0, 3.0, 2.0, 3.0, 5.0, 6.0, 5.0, 6.0}));
  EXPECT(numerical::take(t, far, 1, take_mode::clip).data() == (numerical::buffer<double> {3.0, 1.0, 2.0, 1.0, 6.0, 4.0, 5.0, 4.0}));
  EXPECT(numerical::take(t, tensor<std::int64_t> {{1}, {-2}}, 0, take_mode::raise).data() == (numerical::buffer<double> {1.0, 2.0, 3.0}));
  const tensor<std::int64_t> scalar  {std::vector<std::size_t> {}, numerical::buffer<std::int64_t> {1}};
  EXPECT(numerical::take(t, scalar, 0, take_mode::raise).shape() == (std::vector<std::size_t> {3}));
  try {
    (void)numerical::take(t, far, 1, take_mode::raise);
    EXPECT(false);
  }
  catch (const std::out_of_range& outside) {
    EXPECT(std::string(outside.what()) == "index 7 is out of bounds for axis 1 with size 3");
  }
  EXPECT_THROWS((void)numerical::take(t, tensor<std::int64_t> {{1}, {-4}}, 1, take_mode::raise), std::out_of_range);
  EXPECT_THROWS((void)numerical::take(t, some, 2, take_mode::raise), numerical::dimension_error);
  // no element taken: an index outside still refused where a slice of the axes before stands, not where none does
  const tensor<double>       flat_rows {numerical::zeros<double>({2, 3, 0})};
  EXPECT_THROWS((void)numerical::take(flat_rows, far, 1, take_mode::raise), std::out_of_range);
  EXPECT_THROWS((void)numerical::take(flat_rows, tensor<std::int64_t> {{1}, {-4}}, 1, take_mode::raise), std::out_of_range);
  EXPECT(numerical::take(flat_rows, some, 1, take_mode::raise).shape() == (std::vector<std::size_t> {2, 2, 2, 0}));
  EXPECT(numerical::take(numerical::zeros<double>({0, 3}), far, 1, take_mode::raise).size() == 0);
  EXPECT(numerical::take(flat_rows, far, 1, take_mode::wrap).shape() == (std::vector<std::size_t> {2, 4, 0}));
  EXPECT(numerical::take(t, tensor<std::int64_t> {{0}, numerical::buffer<std::int64_t> {}}, 1, take_mode::raise).shape()
         == (std::vector<std::size_t> {2, 0}));
  // an empty axis: nothing to take from where the result has elements, nor under wrap with any index
  const tensor<double>       hollow  {numerical::zeros<double>({2, 0})};
  const tensor<std::int64_t> none    {{0}, numerical::buffer<std::int64_t> {}};
  EXPECT_THROWS((void)numerical::take(hollow, far, 1, take_mode::clip), std::out_of_range);
  EXPECT_THROWS((void)numerical::take(numerical::zeros<double>({0, 0}), far, 1, take_mode::wrap), std::out_of_range);
  EXPECT(numerical::take(numerical::zeros<double>({0, 0}), far, 1, take_mode::clip).shape() == (std::vector<std::size_t> {0, 4}));
  EXPECT(numerical::take(hollow, none, 1, take_mode::wrap).shape() == (std::vector<std::size_t> {2, 0}));
}

// A long run is sorted by its bits: the same bits as the stable sort by `<`, NaNs behind in their order, the zeros'
// signs in theirs, for float64, float32 and int64 (its extremes too), in one part or several among the threads; a
// short one by comparisons.
template <typename T>
bool radix_sorted_as_stable(std::vector<T> values,
                            std::size_t    parts = 1)
{
  std::vector<T> want    {values};
  T* const       numbers {std::stable_partition(want.data(), want.data() + want.size(), [](T x) { return !std::isnan(static_cast<double>(x)); })};
  std::stable_sort(want.data(), numbers, [](T a, T b) { return a < b; });
  numerical::detail::sort_run(values.data(), values.data() + values.size(), parts);
  return std::memcmp(values.data(), want.data(), values.size() * sizeof(T)) == 0;
}

// The indices of a line sorted in parts (order_run): those of the stable sort, whatever the parts.
template <typename T>
bool ordered_as_stable(const std::vector<T>& values,
                       std::size_t           parts)
{
  const auto                less {[&](std::int64_t a, std::int64_t b) {
                                    return numerical::detail::sorts_before(values[static_cast<std::size_t>(a)], values[static_cast<std::size_t>(b)]);
                                  }};
  std::vector<std::int64_t> want(values.size());
  std::iota(want.begin(), want.end(), std::int64_t {0});
  std::vector<std::int64_t> order {want};
  std::stable_sort(want.begin(), want.end(), less);
  numerical::detail::order_run(order.data(), order.data() + order.size(), values.data(), 1, less, parts);
  return order == want;
}

template <typename T>
bool argsort_as_stable(const std::vector<T>& values)
{
  std::vector<std::int64_t> want(values.size());
  std::iota(want.begin(), want.end(), std::int64_t {0});
  std::stable_sort(want.begin(), want.end(), [&](std::int64_t a, std::int64_t b) {
                     return numerical::detail::sorts_before(values[static_cast<std::size_t>(a)], values[static_cast<std::size_t>(b)]);
                   });
  const tensor<T> line {{values.size()}, numerical::buffer<T>(values.begin(), values.end())};
  return std::ranges::equal(numerical::argsort(line, 0).data(), want);
}

TEST(sort_runs_by_bits_as_the_stable_sort)
{
  const double                nan_a {std::bit_cast<double>(std::uint64_t {0x7ff8000000000001})};
  const double                nan_b {std::bit_cast<double>(std::uint64_t {0xfff8000000000002})}; // a negative NaN: last all the same
  std::vector<double>         wide;
  std::vector<float>          single;
  std::vector<std::int64_t>   integers;
  const std::array<double, 7> specials {0.0, -0.0, nan_a, nan_b, std::numeric_limits<double>::infinity(), -1e300, 5e-324};
  for (std::size_t k {0}; k < 3 * numerical::detail::radix_sort_least; ++k) {
    const double value {k % 5 == 0 ? specials[k % 7] : (static_cast<double>((k * 7919) % 1001) / 7.0) - 70.0};
    wide.push_back(value);
    single.push_back(static_cast<float>(value));
    integers.push_back(k % 11 == 0 ? std::numeric_limits<std::int64_t>::min() + static_cast<std::int64_t>(k % 3)
                                   : static_cast<std::int64_t>((k * 7919) % 1001) - 500);
  }
  EXPECT(radix_sorted_as_stable(wide));
  EXPECT(radix_sorted_as_stable(single));
  EXPECT(radix_sorted_as_stable(integers));
  EXPECT(radix_sorted_as_stable(std::vector<double>(numerical::detail::radix_sort_least, 2.0))); // every byte shared
  EXPECT(radix_sorted_as_stable(std::vector<std::int64_t> {3, -1, 2}));                          // short: compared
  std::vector<std::int64_t> low_byte(numerical::detail::radix_sort_least);
  for (std::size_t k {0}; k < low_byte.size(); ++k) {
    low_byte[k] = static_cast<std::int64_t>((k * 37) % 256);
  }
  EXPECT(radix_sorted_as_stable(low_byte)); // one byte that differs: one pass, the keys sorted in the spare buffer
  // argsort of a long line: its indices by the bits, equal elements (both zeros among them) by index, NaNs last.
  EXPECT(argsort_as_stable(wide) && argsort_as_stable(single) && argsort_as_stable(integers));
  EXPECT(argsort_as_stable(std::vector<double>(numerical::detail::radix_sort_least, -0.0)));
  // In five parts of unequal lengths: each pass's places part after part, the parts' bytes counted again once moved.
  EXPECT(radix_sorted_as_stable(wide, 5) && radix_sorted_as_stable(single, 5) && radix_sorted_as_stable(integers, 5));
  EXPECT(ordered_as_stable(wide, 5) && ordered_as_stable(single, 5) && ordered_as_stable(integers, 5));
  EXPECT(radix_sorted_as_stable(std::vector<double>(numerical::detail::radix_sort_least, 2.0), 3)); // no pass moves
  std::vector<double> mostly_nan(numerical::detail::radix_sort_least, nan_a);
  EXPECT(ordered_as_stable(mostly_nan, 4));                                                         // no number: one part
  mostly_nan[7]  = 2.0;
  mostly_nan[90] = -1.0;
  EXPECT(ordered_as_stable(mostly_nan, 4));                                                         // two numbers: two parts
}

// nonzero over many chunks of a rank-3 tensor: each chunk starts its indices mid-row (its odometer's carries across
// the chunks' edges), and the indices are those of a plain walk, in order.
TEST(nonzero_across_chunks_as_a_plain_walk)
{
  const std::size_t                      depth {3};
  const std::size_t                      rows  {7};
  const std::size_t                      width {((3 * numerical::detail::elementwise_chunk) / (depth * rows)) + 13};
  numerical::buffer<double>              values(depth * rows * width);
  std::vector<std::vector<std::int64_t>> want(3);
  for (std::size_t k {0}; k < values.size(); ++k) {
    values[k] = (k % 3 == 0 || k % 7 == 1) ? 1.0 : 0.0;
    if (values[k] != 0.0) {
      want[0].push_back(static_cast<std::int64_t>(k / (rows * width)));
      want[1].push_back(static_cast<std::int64_t>((k / width) % rows));
      want[2].push_back(static_cast<std::int64_t>(k % width));
    }
  }
  const std::vector<tensor<std::int64_t>> found {numerical::nonzero(tensor<double> {{depth, rows, width}, std::move(values)})};
  bool                                    same  {found.size() == 3};
  for (std::size_t a {0}; same && a < 3; ++a) {
    same = std::equal(found[a].data().begin(), found[a].data().end(), want[a].begin(), want[a].end());
  }
  EXPECT(same);
  EXPECT(numerical::nonzero(tensor<double> {{}, {2.0}}).empty()); // a scalar has no axis to index
}

// count_nonzero without an axis over many chunks (a NaN and a negative zero among them: one counts, one does not), and
// of an empty tensor and a scalar.
TEST(count_nonzero_of_every_element_across_chunks)
{
  const std::size_t         size {(3 * numerical::detail::elementwise_chunk) + 17};
  numerical::buffer<double> values(size);
  std::size_t               want {0};
  for (std::size_t k {0}; k < size; ++k) {
    values[k] = k % 11 == 3 ? std::numeric_limits<double>::quiet_NaN() : static_cast<double>(k % 4);
    if (k % 5 == 0) {
      values[k] = -0.0;
    }
    want += static_cast<std::size_t>(values[k] != 0.0);
  }
  EXPECT(numerical::count_nonzero(tensor<double> {{size}, std::move(values)}) == want);
  EXPECT(numerical::count_nonzero(tensor<double> {{0, 3}, {}}) == 0);
  EXPECT(numerical::count_nonzero(tensor<std::int64_t> {{}, {7}}) == 1);
}

// triangle keeps each matrix's part to and below (above) its k diagonal and zeros the rest, as NumPy's tril and triu:
// every k from past the lower-left corner to past the upper-right one (the extremes clamped, so i + k cannot overflow),
// a stack of matrices, a wide and a tall one, a negative zero kept where it lies; an empty matrix, and a vector refused.
TEST(triangle_keeps_one_side_of_a_diagonal)
{
  for (const auto& [rows, cols] : {std::pair<std::size_t, std::size_t> {3, 5}, {5, 3}, {4, 4}}) {
    const std::size_t         batch {2};
    numerical::buffer<double> values(batch * rows * cols);
    for (std::size_t k {0}; k < values.size(); ++k) {
      values[k] = k % 6 == 0 ? -0.0 : static_cast<double>(k + 1);
    }
    const tensor<double> t {{batch, rows, cols}, values};
    for (const std::ptrdiff_t k : {std::numeric_limits<std::ptrdiff_t>::min(), std::ptrdiff_t {-6}, std::ptrdiff_t {-1},
                                   std::ptrdiff_t {0}, std::ptrdiff_t {2}, std::ptrdiff_t {6},
                                   std::numeric_limits<std::ptrdiff_t>::max()}) {
      for (const bool lower : {true, false}) {
        const tensor<double> got  {numerical::triangle(t, k, lower)};
        bool                 same {got.shape() == t.shape()};
        for (std::size_t at {0}; same && at < values.size(); ++at) {
          const auto   i    {static_cast<double>((at / cols) % rows)};
          const auto   j    {static_cast<double>(at % cols)};
          const auto   band {static_cast<double>(k)};
          const bool   kept {lower ? j <= i + band : j >= i + band};
          const double want {kept ? values[at] : 0.0};
          same = std::bit_cast<std::uint64_t>(got.data()[at]) == std::bit_cast<std::uint64_t>(want);
        }
        EXPECT(same);
      }
    }
  }
  EXPECT(numerical::triangle(tensor<double> {{0, 4}, {}}, 0, true).shape() == (std::vector<std::size_t> {0, 4}));
  EXPECT_THROWS(static_cast<void>(numerical::triangle(tensor<double> {{3}, {1.0, 2.0, 3.0}}, 0, true)),
                numerical::dimension_error);
}

// triangle_indices lists the part to and below (above) the k diagonal in row-major order, as NumPy's tril_indices and
// triu_indices: square, wide, tall and empty matrices, k past either corner.
TEST(triangle_indices_in_row_major_order)
{
  for (const auto& [rows, cols] : {std::pair<std::size_t, std::size_t> {4, 4}, {3, 6}, {6, 2}, {0, 3}, {3, 0}}) {
    for (const std::ptrdiff_t k : {std::numeric_limits<std::ptrdiff_t>::min(), std::ptrdiff_t {-2}, std::ptrdiff_t {0},
                                   std::ptrdiff_t {1}, std::numeric_limits<std::ptrdiff_t>::max()}) {
      for (const bool lower : {true, false}) {
        std::vector<std::int64_t> want_rows;
        std::vector<std::int64_t> want_cols;
        for (std::size_t i {0}; i < rows; ++i) {
          for (std::size_t j {0}; j < cols; ++j) {
            const auto band {static_cast<double>(k)};
            if (lower ? static_cast<double>(j) <= static_cast<double>(i) + band : static_cast<double>(j) >= static_cast<double>(i) + band) {
              want_rows.push_back(static_cast<std::int64_t>(i));
              want_cols.push_back(static_cast<std::int64_t>(j));
            }
          }
        }
        const auto found {numerical::triangle_indices(rows, cols, k, lower)};
        EXPECT(std::ranges::equal(found[0].data(), want_rows) && std::ranges::equal(found[1].data(), want_cols));
      }
    }
  }
}

// diag puts a vector on the k diagonal of a square of order len + |k|, zeros elsewhere, as NumPy's diag of a vector:
// on, above and below the main diagonal, an empty vector's (a zero square of order |k|); a matrix and a square too
// large refused.
TEST(diag_puts_a_vector_on_a_diagonal)
{
  const tensor<std::int64_t> v {{3}, {4, 5, 6}};
  for (const std::ptrdiff_t k : {std::ptrdiff_t {-2}, std::ptrdiff_t {0}, std::ptrdiff_t {1}}) {
    const tensor<std::int64_t> got  {numerical::diag(v, k)};
    const std::size_t          n    {3 + static_cast<std::size_t>(k < 0 ? -k : k)};
    bool                       same {got.shape() == std::vector<std::size_t> {n, n}};
    for (std::size_t i {0}; same && i < n; ++i) {
      for (std::size_t j {0}; same && j < n; ++j) {
        const std::ptrdiff_t at {static_cast<std::ptrdiff_t>(k < 0 ? j : i)};
        const bool           on {static_cast<std::ptrdiff_t>(j) - static_cast<std::ptrdiff_t>(i) == k};
        same = got.data()[(i * n) + j] == (on ? v.data()[static_cast<std::size_t>(at)] : 0);
      }
    }
    EXPECT(same);
  }
  EXPECT(numerical::diag(tensor<std::int64_t> {{0}, {}}, -2).shape() == (std::vector<std::size_t> {2, 2}));
  EXPECT_THROWS(static_cast<void>(numerical::diag(tensor<std::int64_t> {{1, 1}, {1}}, 0)), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::diag(tensor<std::int64_t> {{1}, {1}}, std::numeric_limits<std::ptrdiff_t>::min())),
                numerical::dimension_error); // a square of order 2⁶³ + 1
}

//! \brief Whether argsort_by of \p order by \p key along \p axis is each line of \p order sorted stably by the elements
//!        of \p key it indexes.
template <typename T>
bool sorted_by_as_stable(const tensor<std::int64_t>& order,
                         const tensor<T>&            key,
                         std::size_t                 axis)
{
  const tensor<std::int64_t>         got  {numerical::argsort_by(order, key, axis)};
  const numerical::detail::axis_walk walk {numerical::detail::walk_along(key.shape(), axis)};
  bool                               same {got.shape() == key.shape()};
  for (std::size_t line {0}; same && line < walk.outer * walk.inner; ++line) {
    const std::size_t         first {((line / walk.inner) * walk.extent * walk.inner) + (line % walk.inner)};
    std::vector<std::int64_t> want(walk.extent);
    for (std::size_t k {0}; k < walk.extent; ++k) {
      want[k] = order.data()[first + (k * walk.inner)];
    }
    std::stable_sort(want.begin(), want.end(), [&](std::int64_t a, std::int64_t b) {
                       return numerical::detail::sorts_before(key.data()[first + (static_cast<std::size_t>(a) * walk.inner)],
                                                              key.data()[first + (static_cast<std::size_t>(b) * walk.inner)]);
                     });
    for (std::size_t k {0}; same && k < walk.extent; ++k) {
      same = got.data()[first + (k * walk.inner)] == want[k];
    }
  }
  return same;
}

// argsort_by sorts an order stably by a key read where it lies, as each key of numpy.lexsort: along either axis of a
// matrix whose order is a reversal (equal keys keep the reversed places), the lines one to a task, short or long; one
// line long enough for its radix sort among the threads (both zeros and NaNs among its keys), of reals, floats and integers; a complex
// key by comparisons. Shapes that differ and an index past the axis refused.
TEST(argsort_by_sorts_an_order_stably_by_a_key)
{
  const std::size_t               rows {37};
  const std::size_t               cols {41};
  numerical::buffer<double>       values(rows * cols);
  numerical::buffer<std::int64_t> reversed_rows(rows * cols);
  numerical::buffer<std::int64_t> reversed_cols(rows * cols);
  for (std::size_t k {0}; k < values.size(); ++k) {
    values[k]        = k % 7 == 0 ? -0.0 : static_cast<double>(k % 5);
    if (k % 13 == 0) {
      values[k] = std::numeric_limits<double>::quiet_NaN();
    }
    reversed_rows[k] = static_cast<std::int64_t>(rows - 1 - (k / cols));
    reversed_cols[k] = static_cast<std::int64_t>(cols - 1 - (k % cols));
  }
  const tensor<double> key {{rows, cols}, values};
  EXPECT(sorted_by_as_stable(tensor<std::int64_t> {{rows, cols}, reversed_rows}, key, 0));
  EXPECT(sorted_by_as_stable(tensor<std::int64_t> {{rows, cols}, reversed_cols}, key, 1));
  const std::size_t               length {std::max<std::size_t>(3 * numerical::detail::radix_sort_least, 1U << 15)};
  numerical::buffer<double>       wide(length);
  numerical::buffer<float>        single(length);
  numerical::buffer<std::int64_t> integers(length);
  numerical::buffer<std::int64_t> reversed(length);
  for (std::size_t k {0}; k < length; ++k) {
    wide[k]     = k % 3 == 0 ? -0.0 : static_cast<double>((k * 7919) % 101);
    if (k % 17 == 0) {
      wide[k] = std::numeric_limits<double>::quiet_NaN();
    }
    single[k]   = static_cast<float>(wide[k]);
    integers[k] = static_cast<std::int64_t>((k * 7919) % 101) - 50;
    reversed[k] = static_cast<std::int64_t>(length - 1 - k);
  }
  const tensor<std::int64_t> order {{length}, reversed};
  EXPECT(sorted_by_as_stable(order, tensor<double> {{length}, wide}, 0));
  numerical::buffer<double>       lines(8 * length);
  numerical::buffer<std::int64_t> reversed_lines(8 * length);
  for (std::size_t k {0}; k < lines.size(); ++k) {
    lines[k]          = wide[(k * 31) % length];
    reversed_lines[k] = reversed[k % length];
  }
  EXPECT(sorted_by_as_stable(tensor<std::int64_t> {{8, length}, reversed_lines}, tensor<double> {{8, length}, lines}, 1)); // long lines, one to a task
  EXPECT(sorted_by_as_stable(order, tensor<float> {{length}, single}, 0));
  EXPECT(sorted_by_as_stable(order, tensor<std::int64_t> {{length}, integers}, 0));
  EXPECT(sorted_by_as_stable(tensor<std::int64_t> {{3}, {2, 0, 1}}, tensor<std::complex<double>> {{3}, {std::complex<double> {1.0, 2.0}, std::complex<double> {1.0, 1.0}, std::complex<double> {0.0, 5.0}}}, 0));
  EXPECT_THROWS(static_cast<void>(numerical::argsort_by(tensor<std::int64_t> {{2}, {0, 1}}, tensor<double> {{3}, {1.0, 2.0, 3.0}}, 0)),
                numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::argsort_by(tensor<std::int64_t> {{2}, {0, 2}}, tensor<double> {{2}, {1.0, 2.0}}, 0)),
                numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::argsort_by(tensor<std::int64_t> {{2}, {0, -1}}, tensor<double> {{2}, {1.0, 2.0}}, 0)),
                numerical::dimension_error);
}

// A selection answers what a stable sort puts at each position, to the bit: among zeros of either sign and NaNs of
// different payloads, the one of the same rank in the line's order. Positions alike for every line or one row per
// line, repeated ones, more than a selection takes (the line sorted), integers and complex numbers.
TEST(select_along_last_as_the_stable_sort)
{
  const double         nan_a  {std::bit_cast<double>(std::uint64_t {0x7ff8000000000001})};
  const double         nan_b  {std::bit_cast<double>(std::uint64_t {0x7ff8000000000002})};
  const tensor<double> t      {{2, 7}, {0.0, -1.0, nan_a, -0.0, 2.0, nan_b, 0.0, -0.0, 3.0, 0.0, nan_b, -2.0, nan_a, -0.0}};
  const tensor<double> sorted {numerical::sort(t, 1)};
  bool                 every  {true};
  for (std::int64_t k {0}; k < 7; ++k) {
    const tensor<double> picked {numerical::select_along_last(t, tensor<std::int64_t> {{1}, {k}})};
    for (std::size_t line {0}; line < 2; ++line) {
      every = every && std::bit_cast<std::uint64_t>(picked.data()[line]) ==
              std::bit_cast<std::uint64_t>(sorted.data()[(line * 7) + static_cast<std::size_t>(k)]);
    }
  }
  EXPECT(every);
  const tensor<std::int64_t> rows   {{2, 3}, {4, 1, 1, 6, 0, 5}}; // one row per line, a position twice
  const tensor<double>       picked {numerical::select_along_last(t, rows)};
  bool                       each   {picked.shape() == (std::vector<std::size_t> {2, 3})};
  for (std::size_t j {0}; j < 6; ++j) {
    const std::size_t line {j / 3};
    each = each && std::bit_cast<std::uint64_t>(picked.data()[j]) ==
           std::bit_cast<std::uint64_t>(sorted.data()[(line * 7) + static_cast<std::size_t>(rows.data()[j])]);
  }
  EXPECT(each);
  const tensor<std::int64_t> many {{9}, {6, 5, 4, 3, 2, 1, 0, 0, 6}}; // past the positions a selection takes: sorted
  const tensor<double>       all  {numerical::select_along_last(t, many)};
  bool                       same {true};
  for (std::size_t j {0}; j < 18; ++j) {
    same = same && std::bit_cast<std::uint64_t>(all.data()[j]) ==
           std::bit_cast<std::uint64_t>(sorted.data()[((j / 9) * 7) + static_cast<std::size_t>(many.data()[j % 9])]);
  }
  EXPECT(same);
  const tensor<std::int64_t> ints {{5}, {3, -1, 3, 0, 2}};
  EXPECT(numerical::select_along_last(ints, tensor<std::int64_t> {{2}, {0, 4}}).data() == (numerical::buffer<std::int64_t> {-1, 3}));
  using cplx = std::complex<double>;
  const tensor<cplx> z {{3}, {cplx {1.0, 0.0}, cplx {0.0, 2.0}, cplx {0.0, 1.0}}};
  EXPECT(numerical::select_along_last(z, tensor<std::int64_t> {{1}, {1}}).data()[0] == cplx(0.0, 2.0));
  EXPECT(numerical::select_along_last(numerical::zeros<double>({0, 3}), tensor<std::int64_t> {{1}, {2}}).shape() == (std::vector<std::size_t> {0, 1}));
  EXPECT_THROWS((void)numerical::select_along_last(tensor<double> {{}, {1.0}}, tensor<std::int64_t> {{1}, {0}}), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::select_along_last(t, tensor<std::int64_t> {{}, {0}}), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::select_along_last(t, tensor<std::int64_t> {{3, 1}, {0, 0, 0}}), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::select_along_last(t, tensor<std::int64_t> {{1}, {7}}), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::select_along_last(t, tensor<std::int64_t> {{1}, {-1}}), numerical::dimension_error);
}

namespace {
  // Whether select_along_last of drawn lines of T, at drawn positions, gives the stable sort's bits: lines of 1 to 300
  // elements of few values (zeros of either sign, NaNs of two payloads and signs, repeats), with no NaN, some and only
  // NaNs, at 1 to 8 positions drawn among the edges (the first, the last that is no NaN, the first NaN, the last) and
  // the rest, repeats included.
  template <typename T>
  bool selections_drawn_match_the_sort()
  {
    using bits = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
    const T                nan_a {std::bit_cast<T>(static_cast<bits>(sizeof(T) == 8 ? 0x7ff8000000000001ULL : 0x7fc00001U))};
    const T                nan_b {std::bit_cast<T>(static_cast<bits>(sizeof(T) == 8 ? 0xfff8000000000002ULL : 0xffc00002U))};
    const std::array<T, 8> values {T {-2}, T {-1}, static_cast<T>(-0.0), T {0}, T {1}, static_cast<T>(3.5), static_cast<T>(1e30),
                                   static_cast<T>(-1e-30)};
    std::uint64_t          state {12345};
    const auto             draw {[&state](std::size_t bound) {
                                   state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
                                   return static_cast<std::size_t>((state >> 33U) % bound);
                                 }};
    bool every {true};
    for (const std::size_t count : {std::size_t {1}, std::size_t {2}, std::size_t {61}, std::size_t {300}, std::size_t {2000}}) {
      for (const std::size_t nans_in : {std::size_t {0}, std::size_t {5}, std::size_t {100}}) { // percent
        for (int round {0}; round < 12; ++round) {
          numerical::buffer<T> line(2 * count);
          for (T& v : line) {
            const T nan {draw(2) == 0 ? nan_a : nan_b};
            v = draw(100) < nans_in ? nan : values[draw(values.size())];
          }
          const tensor<T>                  t      {{2, count}, std::move(line)};
          const tensor<T>                  sorted {numerical::sort(t, 1)};
          const auto                       first  {sorted.data().begin()};
          const auto                       valid  {static_cast<std::size_t>(std::count_if(first, first + static_cast<std::ptrdiff_t>(count), [](T v) { return !std::isnan(v); }))};
          const std::array<std::size_t, 4> edges  {0, valid == 0 ? 0 : valid - 1, std::min(valid, count - 1), count - 1};
          std::vector<std::int64_t>        at(1 + draw(8));
          for (std::int64_t& k : at) {
            k = static_cast<std::int64_t>(draw(2) == 0 ? edges[draw(edges.size())] : draw(count));
          }
          const std::size_t m      {at.size()};
          const tensor<T>   picked {numerical::select_along_last(t, tensor<std::int64_t> {{m}, numerical::buffer<std::int64_t>(at.begin(), at.end())})};
          every = every && picked.shape() == (std::vector<std::size_t> {2, m});
          for (std::size_t j {0}; every && j < 2 * m; ++j) {
            const std::size_t k {static_cast<std::size_t>(at[j % m])};
            every = std::bit_cast<bits>(picked.data()[j]) == std::bit_cast<bits>(sorted.data()[((j / m) * count) + k]);
          }
        }
      }
    }
    return every;
  }
} // namespace

// Drawn lines and positions, float and double: the stable sort's bits, NaNs and zeros among them.
TEST(select_along_last_of_drawn_lines_as_the_stable_sort)
{
  EXPECT(selections_drawn_match_the_sort<double>());
  EXPECT(selections_drawn_match_the_sort<float>());
}

// select_keys at every depth, its partitions' bound reached at once (std::nth_element key by key) or after one: the
// sort's elements at the keys, for distinct values, repeats and one value alone, doubles and integers.
TEST(select_keys_at_every_depth_as_the_sort)
{
  const auto selected_as_sorted {[]<typename T>(std::vector<T> line, std::size_t depth) {
                                   std::vector<T> sorted                 {line};
                                   std::sort(sorted.begin(), sorted.end());
                                   const std::array<std::size_t, 5> keys {0, 3, 40, 41, line.size() - 1};
                                   numerical::detail::select_keys(line.data(), 0, line.size(), keys.data(), keys.size(), depth);
                                   return std::ranges::all_of(keys, [&](std::size_t k) { return line[k] == sorted[k]; });
                                 }};
  std::vector<double>       drawn(100);
  std::vector<std::int64_t> repeats(100);
  for (std::size_t i {0}; i < 100; ++i) {
    drawn[i]   = static_cast<double>((i * 37) % 101) - 50.0;
    repeats[i] = static_cast<std::int64_t>((i * 7) % 3);
  }
  bool every {true};
  for (const std::size_t depth : {std::size_t {0}, std::size_t {1}, std::size_t {14}}) {
    every = every && selected_as_sorted(drawn, depth) && selected_as_sorted(repeats, depth)
            && selected_as_sorted(std::vector<double>(100, 2.5), depth);
  }
  EXPECT(every);
}

// Lines that are columns are walked in bands of 256, row after row: each line's sums, products and counts are those
// of its elements in order, the bands' edges anywhere (a band of 256 and one of 44 here, under 2 outer slices).
TEST(columns_fold_and_scan_in_bands_as_line_by_line)
{
  const std::size_t         outer {2};
  const std::size_t         rows  {5};
  const std::size_t         width {300};
  numerical::buffer<double> values(outer * rows * width);
  for (std::size_t k {0}; k < values.size(); ++k) {
    values[k] = (k % 7 == 0) ? 0.0 : 1.0 + (static_cast<double>(k % 13) / 3.0);
  }
  const tensor<double>       t        {{outer, rows, width}, numerical::buffer<double>(values)};
  const tensor<double>       sums     {numerical::cumsum(t, 1)};
  const tensor<double>       products {numerical::prod(t, 1)};
  const tensor<std::int64_t> counts   {numerical::count_nonzero(t, 1)};
  bool                       ok       {true};
  for (std::size_t o {0}; o < outer; ++o) {
    for (std::size_t w {0}; w < width; ++w) {
      double       running {0.0};
      double       product {1.0};
      std::int64_t nonzero {0};
      for (std::size_t k {0}; k < rows; ++k) {
        const double x {values[(((o * rows) + k) * width) + w]};
        running  = k == 0 ? x : running + x;
        product *= x;
        nonzero += x != 0.0 ? 1 : 0;
        ok       = ok && sums.data()[(((o * rows) + k) * width) + w] == running;
      }
      ok = ok && products.data()[(o * width) + w] == product && counts.data()[(o * width) + w] == nonzero;
    }
  }
  EXPECT(ok);
  EXPECT(numerical::count_nonzero(numerical::zeros<double>({3, 0}), 0).size() == 0);
  EXPECT(numerical::count_nonzero(numerical::zeros<double>({0, 3}), 0).data() == (numerical::buffer<std::int64_t> {0, 0, 0}));
}

TEST(prod_cumulative_sums_and_sort_along_an_axis)
{
  const tensor<double> grid {{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  EXPECT(numerical::prod(grid) == 720.0 && numerical::prod(numerical::zeros<double>({0})) == 1.0);
  EXPECT(numerical::prod(grid, 0).data() == (numerical::buffer<double> {4.0, 10.0, 18.0}));
  EXPECT(numerical::prod(grid, 1).data() == (numerical::buffer<double> {6.0, 120.0}));
  EXPECT(numerical::cumsum(grid, 1).data() == (numerical::buffer<double> {1.0, 3.0, 6.0, 4.0, 9.0, 15.0}));
  EXPECT(numerical::cumsum(grid, 0).data() == (numerical::buffer<double> {1.0, 2.0, 3.0, 5.0, 7.0, 9.0}));
  EXPECT(numerical::cumprod(grid, 1).data() == (numerical::buffer<double> {1.0, 2.0, 6.0, 4.0, 20.0, 120.0}));
  const double         nan   {std::numeric_limits<double>::quiet_NaN()};
  const tensor<double> mixed {{2, 4}, {3.0, nan, -1.0, 2.0, 0.0, -0.0, 5.0, -2.0}};
  const tensor<double> rows  {numerical::sort(mixed, 1)};
  EXPECT(rows.data()[0] == -1.0 && rows.data()[2] == 3.0 && std::isnan(rows.data()[3]));
  EXPECT(rows.data()[4] == -2.0 && !std::signbit(rows.data()[5]) && std::signbit(rows.data()[6]) && rows.data()[7] == 5.0); // 0, −0 kept
  EXPECT(numerical::sort(mixed, 0).data()[1] == -0.0 && std::isnan(numerical::sort(mixed, 0).data()[5]));
  using cplx = std::complex<double>;
  const tensor<cplx> complex_line {{4}, {cplx {1.0, 2.0}, cplx {1.0, -1.0}, cplx {nan, 0.0}, cplx {0.0, 5.0}}};
  const tensor<cplx> sorted       {numerical::sort(complex_line, 0)};
  EXPECT(sorted.data()[0] == cplx(0.0, 5.0) && sorted.data()[1] == cplx(1.0, -1.0) && sorted.data()[2] == cplx(1.0, 2.0));
  EXPECT(std::isnan(sorted.data()[3].real()));
  const tensor<cplx> imaginary {{3}, {cplx {1.0, nan}, cplx {1.0, 1.0}, cplx {1.0, 1.0}}}; // a NaN imaginary part last
  EXPECT(std::isnan(numerical::sort(imaginary, 0).data()[2].imag()));
  // NumPy's order (npysort's CLT): no NaN, then a NaN imaginary part by the real parts, a NaN real part by the
  // imaginary parts, both NaN; every pair against its places, and the line sorted from its reverse.
  const double            inf     {std::numeric_limits<double>::infinity()};
  const std::vector<cplx> order   {cplx {0.0, 0.0}, cplx {0.0, 1.0}, cplx {1.0, -1.0}, cplx {1.0, 0.0}, cplx {-inf, nan},
                                   cplx {0.0, nan}, cplx {1.0, nan}, cplx {nan, 0.0}, cplx {nan, 1.0}, cplx {nan, nan}};
  bool                    ordered {true};
  for (std::size_t i {0}; i < order.size(); ++i) {
    for (std::size_t j {0}; j < order.size(); ++j) {
      ordered = ordered && numerical::detail::sorts_before(order[i], order[j]) == (i < j);
    }
  }
  EXPECT(ordered);
  const tensor<cplx> reversed {{order.size()}, numerical::buffer<cplx>(order.rbegin(), order.rend())};
  const tensor<cplx> resorted {numerical::sort(reversed, 0)};
  const auto         same_place {[](const cplx& a, const cplx& b) {
                                   const auto part {[](double x, double y) { return x == y || (std::isnan(x) && std::isnan(y)); }};
                                   return part(a.real(), b.real()) && part(a.imag(), b.imag());
                                 }};
  EXPECT(std::ranges::equal(resorted.data(), order, same_place));
  const tensor<float> single   {{3}, {2.0F, -1.0F, 0.5F}};
  EXPECT(numerical::sort(single, 0).data()[0] == -1.0F && numerical::cumsum(single, 0).data()[2] == 1.5F);
  EXPECT_THROWS((void)numerical::cumsum(grid, 2), numerical::dimension_error);
  numerical::buffer<double> zeros(4096); // +0, −0, +0, …: past the size where libc++ 22 sorts doubles by their bits
  for (std::size_t k {0}; k < zeros.size(); ++k) {
    zeros[k] = k % 2 == 0 ? 0.0 : -0.0;
  }
  const tensor<double> signed_zeros {numerical::sort(tensor<double> {{zeros.size()}, numerical::buffer<double>(zeros)}, 0)};
  bool                 in_order     {true};
  for (std::size_t k {0}; k < zeros.size(); ++k) {
    in_order = in_order && std::signbit(signed_zeros.data()[k]) == (k % 2 == 1);
  }
  EXPECT(in_order);
}

// A line long enough for the threads sorts among them to the bits of one stable sort: equal values (zeros of either
// sign among them) in their order, NaNs last in theirs. Alone, or two lines along the last axis or along the first,
// with the threads' default and with two: a real's radix passes shared out, a complex line in pieces merged in pairs
// (four pieces with two threads, one merge round fewer), its argsort too.
TEST(sort_of_long_lines_among_the_threads)
{
  const std::size_t   n     {std::size_t {1} << 17};
  std::vector<double> values(2 * n);
  std::uint64_t       state {12345};
  for (std::size_t k {0}; k < values.size(); ++k) {
    state     = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
    values[k] = static_cast<double>(static_cast<std::int64_t>(state >> 54) - 512); // 1024 values, each many times
    if (k % 97 == 0) {
      values[k] = std::numeric_limits<double>::quiet_NaN();
    }
    else if (values[k] == 0.0 && k % 2 == 0) {
      values[k] = -0.0;
    }
  }
  const auto stably_sorted {[](std::vector<double> line) {
                              std::ranges::stable_sort(line, [](double a, double b) { return numerical::detail::sorts_before(a, b); });
                              return line;
                            }};
  const auto same_bits {[](const double* got, const std::vector<double>& want, std::size_t stride) {
                          for (std::size_t k {0}; k < want.size(); ++k) {
                            if (std::bit_cast<std::uint64_t>(got[k * stride]) != std::bit_cast<std::uint64_t>(want[k])) {
                              return false;
                            }
                          }
                          return true;
                        }};
  const std::vector<double> first(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(n));
  const std::vector<double> second(values.begin() + static_cast<std::ptrdiff_t>(n), values.end());
  const std::vector<double> first_sorted  {stably_sorted(first)};
  const std::vector<double> second_sorted {stably_sorted(second)};
  for (const std::size_t threads : {std::size_t {0}, std::size_t {2}}) {
    numerical::set_max_threads(threads);
    const tensor<double> line {numerical::sort(tensor<double> {{n}, numerical::buffer<double>(first.begin(), first.end())}, 0)};
    EXPECT(same_bits(line.data().data(), first_sorted, 1));
    const tensor<double> rows {numerical::sort(tensor<double> {{2, n}, numerical::buffer<double>(values.begin(), values.end())}, 1)};
    EXPECT(same_bits(rows.data().data(), first_sorted, 1) && same_bits(rows.data().data() + n, second_sorted, 1));
    numerical::buffer<double> interleaved(2 * n);
    for (std::size_t k {0}; k < n; ++k) {
      interleaved[2 * k]       = first[k];
      interleaved[(2 * k) + 1] = second[k];
    }
    const tensor<double> columns {numerical::sort(tensor<double> {{n, 2}, std::move(interleaved)}, 0)};
    EXPECT(same_bits(columns.data().data(), first_sorted, 2) && same_bits(columns.data().data() + 1, second_sorted, 2));
    using cplx = std::complex<double>;
    numerical::buffer<cplx> mixed(n);
    for (std::size_t k {0}; k < n; ++k) {
      mixed[k] = cplx {first[k], second[k]};
    }
    const auto                cplx_less {[](const cplx& a, const cplx& b) { return numerical::detail::sorts_before(a, b); }};
    std::vector<cplx>         want(mixed.begin(), mixed.end());
    std::vector<std::int64_t> order(n);
    std::iota(order.begin(), order.end(), std::int64_t {0});
    std::ranges::stable_sort(want, cplx_less);
    std::ranges::stable_sort(order, [&](std::int64_t a, std::int64_t b) {
                               return cplx_less(mixed[static_cast<std::size_t>(a)], mixed[static_cast<std::size_t>(b)]);
                             });
    const tensor<cplx> complex_line {{n}, numerical::buffer<cplx>(mixed)};
    const tensor<cplx> sorted       {numerical::sort(complex_line, 0)};
    EXPECT(std::ranges::equal(sorted.data(), want, [](const cplx& a, const cplx& b) {
                                return std::bit_cast<std::uint64_t>(a.real()) == std::bit_cast<std::uint64_t>(b.real())
                                       && std::bit_cast<std::uint64_t>(a.imag()) == std::bit_cast<std::uint64_t>(b.imag());
                              }));
    EXPECT(std::ranges::equal(numerical::argsort(complex_line, 0).data(), order));
  }
  numerical::set_max_threads(0);
}

// A signed integer's +, −, ·, negation, sums, products and running ones wrap modulo 2⁶⁴ as NumPy's integers (where
// C++'s signed overflow is undefined): the same bits whatever the order or the threads. Integers sort ascending.
TEST(signed_integers_wrap_as_numpy)
{
  using big = std::int64_t;
  const big            top    {std::numeric_limits<big>::max()};
  const big            bottom {std::numeric_limits<big>::min()};
  const tensor<big>    edges  {{3}, {top, bottom, 3}};
  const tensor<big>    ones   {{3}, {1, 1, 2}};
  EXPECT((edges + ones).data() == (numerical::buffer<big> {bottom, bottom + 1, 5}));
  EXPECT((edges - ones).data() == (numerical::buffer<big> {top - 1, top, 1}));
  EXPECT((edges * tensor<big> {{3}, {2, 2, -4}}).data() == (numerical::buffer<big> {-2, 0, -12}));
  EXPECT((-edges).data() == (numerical::buffer<big> {-top, bottom, -3}));
  EXPECT(numerical::sum(edges) == 2 && numerical::prod(edges) == bottom); // top + bottom + 3; top · bottom · 3 wraps
  EXPECT(numerical::sum(tensor<big> {{2, 2}, {top, 1, 1, 2}}, 0).data() == (numerical::buffer<big> {bottom, 3}));
  EXPECT(numerical::prod(tensor<big> {{2, 2}, {top, 3, 2, 5}}, 0).data() == (numerical::buffer<big> {-2, 15}));
  EXPECT(numerical::cumsum(edges, 0).data() == (numerical::buffer<big> {top, -1, 2}));
  EXPECT(numerical::cumprod(edges, 0).data() == (numerical::buffer<big> {top, bottom, bottom}));
  // Many elements, summed in chunks on the threads: the wrapped total in any order.
  const std::size_t      n        {std::size_t {1} << 18};
  numerical::buffer<big> many(n);
  std::uint64_t          expected {0};
  for (std::size_t k {0}; k < n; ++k) {
    many[k]   = static_cast<big>((k * 0x9E3779B97F4A7C15ULL) ^ (k << 7));
    expected += static_cast<std::uint64_t>(many[k]);
  }
  const tensor<big> line {{n}, std::move(many)};
  EXPECT(numerical::sum(line) == static_cast<big>(expected));
  EXPECT(numerical::sum(tensor<big> {{0}, numerical::buffer<big> {}}) == 0);
  const tensor<short> narrow {{2}, {30000, 30000}}; // promoted through unsigned int, no signed product
  EXPECT((narrow * narrow).data()[0] == static_cast<short>(900000000U & 0xFFFFU) && numerical::sum(narrow) == static_cast<short>(60000));
  // Sorted ascending, alone or among the threads (the merge's integer order).
  numerical::buffer<big> shuffled(n);
  for (std::size_t k {0}; k < n; ++k) {
    shuffled[k] = static_cast<big>((k * 2654435761U) % 1000) - 500;
  }
  std::vector<big>  want(shuffled.begin(), shuffled.end());
  std::ranges::sort(want);
  const tensor<big> sorted {numerical::sort(tensor<big> {{n}, std::move(shuffled)}, 0)};
  EXPECT(std::ranges::equal(sorted.data(), want));
  EXPECT(numerical::sort(tensor<big> {{2, 2}, {4, -1, 0, 2}}, 1).data() == (numerical::buffer<big> {-1, 4, 0, 2}));
  // Powers exact by squaring, wrapped; a negative exponent refused (no integer is the answer).
  const tensor<big> bases     {{5}, {3, -2, 7, 0, 2}};
  const tensor<big> exponents {{5}, {4, 3, 0, 0, 64}};
  EXPECT(numerical::pow(bases, exponents).data() == (numerical::buffer<big> {81, -8, 1, 1, 0}));                    // 2⁶⁴ wraps to 0
  EXPECT(numerical::pow(tensor<big> {{1}, {3}}, tensor<big> {{1}, {39}}).data()[0] == big {4052555153018976267LL}); // 3³⁹, just within int64
  EXPECT_THROWS((void)numerical::pow(bases, tensor<big> {{1}, {-1}}), std::domain_error);
  // The larger and smaller of two, broadcast.
  EXPECT(numerical::maximum(bases, tensor<big> {{1}, {1}}).data() == (numerical::buffer<big> {3, 1, 7, 1, 2}));
  EXPECT(numerical::minimum(bases, exponents).data() == (numerical::buffer<big> {3, -2, 0, 0, 2}));
}

// floor_divide and remainder as NumPy's: the quotient floored and the remainder of the divisor's sign. A real by zero
// a / b and fmod's NaN; a zero quotient or remainder of its sign; the quotient snapped to the nearest integer. A signed
// integer by zero 0 and 0, the most negative by −1 itself (wrapped), 0 remainder.
TEST(floor_divide_and_remainder_as_numpy)
{
  const double         inf {std::numeric_limits<double>::infinity()};
  const tensor<double> a   {{11}, {9.142325629204539, 7.0, -7.0, 7.0, 6.0, -6.0, 0.5, -0.5, 1.0, -0.0, 6.0}};
  const tensor<double> b   {{11}, {0.1, 2.0, 2.0, -2.0, 3.0, 3.0, 2.0, 2.0, 0.0, 2.0, -3.0}};
  const tensor<double> q   {numerical::floor_divide(a, b)};
  const tensor<double> r   {numerical::remainder(a, b)};
  EXPECT(q.data()[0] == 91.0 && r.data()[0] == 0.04232562920453822); // (a − mod) / b just under 91: snapped up
  EXPECT(q.data()[1] == 3.0 && r.data()[1] == 1.0 && q.data()[2] == -4.0 && r.data()[2] == 1.0);
  EXPECT(q.data()[3] == -4.0 && r.data()[3] == -1.0 && q.data()[4] == 2.0 && r.data()[4] == 0.0 && !std::signbit(r.data()[4]));
  EXPECT(q.data()[5] == -2.0 && r.data()[5] == 0.0 && q.data()[6] == 0.0 && !std::signbit(q.data()[6]) && r.data()[6] == 0.5);
  EXPECT(q.data()[7] == -1.0 && r.data()[7] == 1.5 && q.data()[8] == inf && std::isnan(r.data()[8]));
  EXPECT(q.data()[9] == 0.0 && std::signbit(q.data()[9]) && r.data()[9] == 0.0 && !std::signbit(r.data()[9]));
  EXPECT(q.data()[10] == -2.0 && r.data()[10] == 0.0 && std::signbit(r.data()[10])); // a zero remainder of b's sign
  using big = std::int64_t;
  const big         bottom {std::numeric_limits<big>::min()};
  const tensor<big> x      {{8}, {7, -7, 7, -7, 6, 5, bottom, -9}};
  const tensor<big> y      {{8}, {2, 2, -2, -2, 3, 0, -1, -1}};
  EXPECT(numerical::floor_divide(x, y).data() == (numerical::buffer<big> {3, -4, -4, 3, 2, 0, bottom, 9}));
  EXPECT(numerical::remainder(x, y).data() == (numerical::buffer<big> {1, 1, -1, -1, 0, 0, 0, 0}));
  EXPECT(numerical::remainder(tensor<float> {{1}, {-7.0F}}, tensor<float> {{1}, {2.0F}}).data()[0] == 1.0F);
}

// The index functions as NumPy's, in int64: argmax and argmin the first extreme (the first NaN where there is one),
// flat or along an axis, on the threads by chunks; argsort stable, a long line among the threads; nonzero by axis,
// count_nonzero along an axis, searchsorted's first or last place (NaNs last).
TEST(index_functions_as_numpy)
{
  using big = std::int64_t;
  const double         nan  {std::numeric_limits<double>::quiet_NaN()};
  const tensor<double> grid {{2, 3}, {3.0, 1.0, 3.0, 2.0, nan, 5.0}};
  EXPECT(numerical::argmax(grid) == 4 && numerical::argmin(grid) == 4); // the NaN
  EXPECT(numerical::argmax(grid, 0).data() == (numerical::buffer<big> {0, 1, 1}));
  EXPECT(numerical::argmin(grid, 1).data() == (numerical::buffer<big> {1, 1}));
  const tensor<big> ints {{5}, {2, 7, -1, 7, -1}};
  EXPECT(numerical::argmax(ints) == 1 && numerical::argmin(ints) == 2); // the first of equal ones
  // zeros of both signs are equal: the first of either, whichever sign the extreme took
  EXPECT(numerical::argmin(tensor<double> {{3}, {1.0, 0.0, -0.0}}) == 1 && numerical::argmax(tensor<double> {{3}, {-1.0, -0.0, 0.0}}) == 1);
  EXPECT(numerical::argmin(tensor<float> {{3}, {1.0F, -0.0F, 0.0F}}) == 1 && numerical::argmax(tensor<float> {{2}, {0.0F, -0.0F}}) == 0);
  EXPECT_THROWS((void)numerical::argmax(tensor<double> {{0}, numerical::buffer<double> {}}), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::argmin(tensor<double> {{0}, numerical::buffer<double> {}}), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::argmax(tensor<double> {{2, 0}, numerical::buffer<double> {}}, 1), numerical::dimension_error);
  // Many elements, by chunks: a later chunk's larger one, an earlier chunk's equal one, an early NaN.
  const std::size_t      n {std::size_t {1} << 17};
  numerical::buffer<big> many(n, 0);
  many[5]     = 9;
  many[n - 3] = 9;
  many[n - 2] = 10;
  many[70000] = -4;
  const tensor<big> line {{n}, std::move(many)};
  EXPECT(numerical::argmax(line) == static_cast<big>(n - 2) && numerical::argmin(line) == 70000);
  numerical::buffer<double> reals(n, 1.0);
  reals[100]   = 2.0;
  reals[50000] = 2.0;
  EXPECT(numerical::argmax(tensor<double> {{n}, numerical::buffer<double>(reals)}) == 100);
  reals[20000] = 0.5; // a later chunk's smaller one: found past the first chunk; then a NaN's chunk, which wins
  EXPECT(numerical::argmin(tensor<double> {{n}, numerical::buffer<double>(reals)}) == 20000);
  reals[40000] = nan;
  reals[90000] = nan;
  EXPECT(numerical::argmin(tensor<double> {{n}, std::move(reals)}) == 40000);
  // argsort: equal elements by index, NaNs last; along either axis; a long line among the threads.
  EXPECT(numerical::argsort(grid, 1).data() == (numerical::buffer<big> {1, 0, 2, 0, 2, 1}));
  EXPECT(numerical::argsort(grid, 0).data() == (numerical::buffer<big> {1, 0, 0, 0, 1, 1}));
  numerical::buffer<double> shuffled(n);
  for (std::size_t k {0}; k < n; ++k) {
    shuffled[k] = static_cast<double>((k * 2654435761U) % 1000);
  }
  shuffled[3] = nan;
  std::vector<big> want(n);
  std::iota(want.begin(), want.end(), big {0});
  std::ranges::stable_sort(want, [&](big a, big b) {
                             return numerical::detail::sorts_before(shuffled[static_cast<std::size_t>(a)], shuffled[static_cast<std::size_t>(b)]);
                           });
  const tensor<big> order {numerical::argsort(tensor<double> {{n}, std::move(shuffled)}, 0)};
  EXPECT(std::ranges::equal(order.data(), want) && order.data()[n - 1] == 3);
  // Lines enough for the threads, each long: one to a task.
  const std::size_t         rows  {16};
  const std::size_t         width {std::size_t {1} << 14};
  numerical::buffer<double> wide(rows * width);
  for (std::size_t k {0}; k < wide.size(); ++k) {
    wide[k] = static_cast<double>((k * 40503U) % 977);
  }
  const tensor<big> each    {numerical::argsort(tensor<double> {{rows, width}, numerical::buffer<double>(wide)}, 1)};
  bool              ordered {true};
  for (std::size_t r {0}; r < rows; ++r) {
    for (std::size_t k {1}; k < width; ++k) {
      const double before {wide[(r * width) + static_cast<std::size_t>(each.data()[(r * width) + k - 1])]};
      const double after  {wide[(r * width) + static_cast<std::size_t>(each.data()[(r * width) + k])]};
      ordered = ordered && (before < after || (before == after && each.data()[(r * width) + k - 1] < each.data()[(r * width) + k]));
    }
  }
  EXPECT(ordered);
  using cplx = std::complex<double>;
  EXPECT(numerical::argsort(tensor<cplx> {{3}, {cplx {1.0, 2.0}, cplx {1.0, -1.0}, cplx {0.0, 9.0}}}, 0).data() == (numerical::buffer<big> {2, 1, 0}));
  // nonzero by axis; none at all (an empty extent too); count_nonzero along an axis.
  const std::vector<tensor<big>> places {numerical::nonzero(tensor<double> {{2, 3}, {0.0, 1.0, 0.0, nan, 0.0, -2.0}})};
  EXPECT(places.size() == 2 && places[0].data() == (numerical::buffer<big> {0, 1, 1}) && places[1].data() == (numerical::buffer<big> {1, 0, 2}));
  const std::vector<tensor<big>> none   {numerical::nonzero(tensor<double> {{2, 0}, numerical::buffer<double> {}})};
  EXPECT(none.size() == 2 && none[0].size() == 0 && none[1].size() == 0);
  EXPECT(numerical::nonzero(tensor<big> {{3}, {0, 0, 0}})[0].size() == 0);
  EXPECT(numerical::count_nonzero(grid, 1).data() == (numerical::buffer<big> {3, 3}));
  EXPECT(numerical::count_nonzero(tensor<big> {{2, 2}, {0, 1, 0, 0}}, 0).data() == (numerical::buffer<big> {0, 1}));
  // searchsorted: the first or the last place, NaNs last; a vector only.
  const tensor<double> sorted {{4}, {1.0, 2.0, 2.0, nan}};
  const tensor<double> values {{3}, {2.0, nan, 0.5}};
  EXPECT(numerical::searchsorted(sorted, values).data() == (numerical::buffer<big> {1, 3, 0}));
  EXPECT(numerical::searchsorted(sorted, values, true).data() == (numerical::buffer<big> {3, 4, 0}));
  EXPECT_THROWS((void)numerical::searchsorted(grid, values), numerical::dimension_error);
}

// round to decimal places as numpy.round: half to even through the scaled value; a value already exact to the places
// kept (no product to overflow); a finite value 0 where 10⁻ᵈ overflows, a NaN or an infinity as it is; a float past
// 10³⁸ places through double; a complex number's parts apart.
TEST(round_to_decimal_places)
{
  const double         inf      {std::numeric_limits<double>::infinity()};
  const double         nan      {std::numeric_limits<double>::quiet_NaN()};
  const tensor<double> halves   {{4}, {1.25, 1.35, 2.5, -0.5}};
  EXPECT(numerical::round(halves, 1).data() == (numerical::buffer<double> {12.0 / 10.0, 14.0 / 10.0, 2.5, -0.5}));
  const tensor<double> whole    {numerical::round(halves)};
  EXPECT(whole.data()[0] == 1.0 && whole.data()[1] == 1.0 && whole.data()[2] == 2.0 && std::signbit(whole.data()[3]));
  const tensor<double> tens     {{3}, {1250.0, 1350.0, -49.0}};
  const tensor<double> hundreds {numerical::round(tens, -2)};
  EXPECT(hundreds.data()[0] == 1200.0 && hundreds.data()[1] == 1400.0 && hundreds.data()[2] == 0.0 && std::signbit(hundreds.data()[2]));
  const tensor<double> edges    {{4}, {1e307, inf, nan, 0.125}};
  const tensor<double> places   {numerical::round(edges, 2)};
  EXPECT(places.data()[0] == 1e307 && places.data()[1] == inf && std::isnan(places.data()[2]) && places.data()[3] == 0.12);
  const tensor<double> far      {{4}, {1.5, -2.25, inf, 0.0}};
  EXPECT(numerical::round(far, 400).data() == far.data()); // 10⁴⁰⁰ overflows: every value its own rounding
  const tensor<double> none     {numerical::round(edges, -400)};
  EXPECT(none.data()[0] == 0.0 && none.data()[1] == inf && std::isnan(none.data()[2]) && none.data()[3] == 0.0);
  EXPECT(std::signbit(numerical::round(tensor<double> {{1}, {-5.0}}, -400).data()[0]));
  const tensor<float> single {{3}, {1.25F, 1.234e-39F, 3e-39F}};
  EXPECT(numerical::round(single, 1).data()[0] == 12.0F / 10.0F);
  const tensor<float> tiny   {numerical::round(single, 40)};
  EXPECT(tiny.data()[0] == 1.25F && tiny.data()[1] == static_cast<float>(12.0 / 1e40) && tiny.data()[2] == 3e-39F);
  EXPECT(numerical::round(single, 38).data()[2] == 0.0F);
  using cplx = std::complex<double>;
  const tensor<cplx> parts                 {numerical::round(tensor<cplx> {{2}, {cplx {1.26, -2.34}, cplx {nan, 0.5}}}, 1)};
  EXPECT(parts.data()[0] == cplx(13.0 / 10.0, -23.0 / 10.0) && std::isnan(parts.data()[1].real()) && parts.data()[1].imag() == 0.5);
  const tensor<std::complex<float>> narrow {numerical::round(tensor<std::complex<float>> {{1}, {std::complex<float> {2.5F, 1e-39F}}}, 40)};
  EXPECT(narrow.data()[0] == std::complex<float>(2.5F, static_cast<float>(1e-39)));
}

namespace {
  //! \brief Whether \p whole (a float function of tensors) gives, for each of \p x, \p one's bits (the scalar float
  //!        function) and the double \p wide rounded once, within half a float's ulp of \p reference (the C library's
  //!        double, within two of its own ulps) and 2⁻²⁸ more.
  template <typename Whole,
            typename One,
            typename Wide,
            typename Reference>
  bool through_doubles_holds(const std::vector<float>& x,
                             Whole                     whole,
                             One                       one,
                             Wide                      wide,
                             Reference                 reference)
  {
    const numerical::tensor<float> values {{x.size()}, numerical::buffer<float>(x.begin(), x.end())};
    const numerical::tensor<float> image  {whole(values)};
    bool                           holds  {true};
    for (std::size_t i {0}; i < x.size(); ++i) {
      const float  got     {image.data()[i]};
      const float  rounded {static_cast<float>(wide(static_cast<double>(x[i])))};
      const double exact   {reference(static_cast<double>(x[i]))};
      holds = holds && std::bit_cast<std::uint32_t>(got) == std::bit_cast<std::uint32_t>(one(x[i]))
              && std::bit_cast<std::uint32_t>(got) == std::bit_cast<std::uint32_t>(rounded);
      if (std::isfinite(exact) && std::fabs(exact) <= std::numeric_limits<float>::max()) {
        const auto   magnitude {static_cast<float>(std::fabs(exact))};
        const double ulp       {static_cast<double>(std::nextafter(magnitude, std::numeric_limits<float>::infinity()) - magnitude)};
        holds = holds && std::fabs(static_cast<double>(got) - exact) <= ulp * (0.5 + 0x1p-28);
      }
      else {
        holds = holds && (std::isnan(exact) ? std::isnan(got) : got == static_cast<float>(exact));
      }
    }
    return holds;
  }
} // namespace

// A float's atan, sinh, cosh, expm1, log1p, tanh, sin and cos are the double's by the tables, rounded once: within half a float's ulp
// (and 2⁻²⁸), where glibc's float functions are within one or two, and each tensor's element the scalar's bits.
// 3011 values: pieces of 512 and a last one shorter
TEST(float_functions_are_the_doubles_rounded_once)
{
  namespace d = numerical::detail;
  std::vector<float> x;
  for (int i {0}; i < 3000; ++i) {
    const float t {static_cast<float>(i) / 3000.0F};
    x.push_back((t < 0.5F ? -1.0F : 1.0F) * std::ldexp(1.0F + t, (i % 300) - 150)); // magnitudes 2^-150 to 2^150
  }
  for (const float v : {0.0F, -0.0F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
                        -1.0F, 89.5F, -89.5F, 1e-45F, 252.898208F, 0x1p20F, -0x1.000002p20F}) {
    x.push_back(v);
  }
  x.resize(3011);
  EXPECT(through_doubles_holds(
           x, [](const auto& t) { return numerical::atan(t); }, [](float v) { return d::elementary_atan(v); },
           [](double v) { return d::elementary_atan(v); }, [](double v) { return std::atan(v); }));
  EXPECT(through_doubles_holds(
           x, [](const auto& t) { return numerical::sinh(t); }, [](float v) { return d::elementary_sinh(v); },
           [](double v) { return d::elementary_sinh(v); }, [](double v) { return std::sinh(v); }));
  EXPECT(through_doubles_holds(
           x, [](const auto& t) { return numerical::cosh(t); }, [](float v) { return d::elementary_cosh(v); },
           [](double v) { return d::elementary_cosh(v); }, [](double v) { return std::cosh(v); }));
  EXPECT(through_doubles_holds(
           x, [](const auto& t) { return numerical::tanh(t); }, [](float v) { return d::elementary_tanh(v); },
           [](double v) { return d::elementary_tanh(v); }, [](double v) { return std::tanh(v); }));
  EXPECT(through_doubles_holds(
           x, [](const auto& t) { return numerical::expm1(t); }, [](float v) { return d::elementary_expm1(v); },
           [](double v) { return d::elementary_expm1(v); }, [](double v) { return std::expm1(v); }));
  // sin and cos: reduced by π/2 to 2²⁰ (252.898… the float nearest a multiple, 161·π/2), the C library's past it
  EXPECT(through_doubles_holds(
           x, [](const auto& t) { return numerical::sin(t); }, [](float v) { return d::elementary_sin(v); },
           [](double v) { return d::float_sine<false>(v); }, [](double v) { return std::sin(v); }));
  EXPECT(through_doubles_holds(
           x, [](const auto& t) { return numerical::cos(t); }, [](float v) { return d::elementary_cos(v); },
           [](double v) { return d::float_sine<true>(v); }, [](double v) { return std::cos(v); }));
  EXPECT(std::signbit(numerical::sin(tensor<float> {{1}, {-0.0F}}).data()[0]) && numerical::cos(tensor<float> {{1}, {-0.0F}}).data()[0] == 1.0F);
  std::vector<float> above(x.size());
  std::ranges::transform(x, above.begin(), [](float v) {
                           return std::isnan(v) || v > -1.0F ? v : -1.0F - v; // log1p's domain, -1 included
                         });
  EXPECT(through_doubles_holds(
           above, [](const auto& t) { return numerical::log1p(t); }, [](float v) { return d::elementary_log1p(v); },
           [](double v) { return d::elementary_log1p(v); }, [](double v) { return std::log1p(v); }));
}

namespace {
  //! \brief Whether numerical::partition and numerical::argpartition of \p t along \p axis at \p kths put at each
  //!        position the element numerical::sort puts there, none before it that sorts after it, none after it that
  //!        sorts before, every line a permutation of its own.
  template <typename T>
  bool partition_holds(const numerical::tensor<T>&     t,
                       const std::vector<std::size_t>& kths,
                       std::size_t                     axis)
  {
    namespace d = numerical::detail;
    const numerical::tensor<T>            sorted {numerical::sort(t, axis)};
    const numerical::tensor<T>            parted {numerical::partition(t, kths, axis)};
    const numerical::tensor<std::int64_t> order {numerical::argpartition(t, kths, axis)};
    const d::axis_walk                    walk {d::walk_along(t.shape(), axis)};
    const auto                            same {[](const T& a, const T& b) { return !d::sorts_before(a, b) && !d::sorts_before(b, a); }};
    const numerical::tensor<T>            resorted {numerical::sort(parted, axis)};
    bool                                  holds    {std::equal(resorted.data().begin(), resorted.data().end(), sorted.data().begin(), same)};
    for (std::size_t line {0}; line < walk.outer * walk.inner; ++line) {
      const std::size_t first {((line / walk.inner) * walk.extent * walk.inner) + (line % walk.inner)};
      const auto        at {[&](const auto& source, std::size_t k) { return source.data()[first + (k * walk.inner)]; }};
      const auto        indexed {[&](std::size_t k) {
                                   return t.data()[first + (static_cast<std::size_t>(at(order, k)) * walk.inner)];
                                 }};
      std::vector<std::int64_t> taken;
      for (std::size_t k {0}; k < walk.extent; ++k) {
        taken.push_back(at(order, k));
      }
      std::ranges::sort(taken);
      for (std::size_t k {0}; k < walk.extent; ++k) {
        holds = holds && std::cmp_equal(taken[k], k);
      }
      for (const std::size_t kth : kths) {
        const T pivot {at(sorted, kth)};
        holds = holds && same(at(parted, kth), pivot) && same(indexed(kth), pivot);
        for (std::size_t k {0}; k < walk.extent; ++k) {
          const bool before {k < kth};
          holds = holds && (k == kth || (before ? !d::sorts_before(pivot, at(parted, k)) : !d::sorts_before(at(parted, k), pivot)));
          holds = holds && (k == kth || (before ? !d::sorts_before(pivot, indexed(k)) : !d::sorts_before(indexed(k), pivot)));
        }
      }
    }
    return holds;
  }
} // namespace

// partition and argpartition select (std::nth_element between neighbouring positions) where sort sorted the whole
// line: the element at each position the sort's, NaNs last, a complex number by its parts in turn; positions given
// in any order and repeated, along either axis, and refused past the extent
TEST(partition_selects_as_a_sort_would_place)
{
  const double        nan {std::numeric_limits<double>::quiet_NaN()};
  std::vector<double> values;
  for (std::size_t i {0}; i < std::size_t {280}; ++i) { // 7 lines of 40
    values.push_back(i % 11 == 3 ? nan : static_cast<double>((i * 37) % 23) - 11.0);
  }
  const numerical::tensor<double> wide {{7, 40}, numerical::buffer<double>(values.begin(), values.end())};
  EXPECT(partition_holds(wide, {20}, 1));
  EXPECT(partition_holds(wide, {39, 0, 13, 13, 5}, 1));
  EXPECT(partition_holds(wide, {3, 6}, 0));
  EXPECT(partition_holds(wide, {}, 1));
  std::vector<std::int64_t> integers(values.size());
  std::ranges::transform(values, integers.begin(),
                         [](double v) { return std::isnan(v) ? std::int64_t {99} : static_cast<std::int64_t>(v); });
  EXPECT(partition_holds(numerical::tensor<std::int64_t> {{40, 7}, numerical::buffer<std::int64_t>(integers.begin(), integers.end())},
                         {0, 6, 2}, 1));
  std::vector<std::complex<double>> complexes;
  for (std::size_t i {0}; i < 60; ++i) {
    complexes.emplace_back(static_cast<double>(i % 3), i % 17 == 4 ? nan : static_cast<double>((i * 7) % 5));
  }
  EXPECT(partition_holds(numerical::tensor<std::complex<double>> {{60}, numerical::buffer<std::complex<double>>(complexes.begin(), complexes.end())},
                         {30, 59}, 0));
  EXPECT(partition_holds(numerical::tensor<float> {{5}, numerical::buffer<float> {2.0F, -1.0F, 2.0F, 0.0F, -0.0F}}, {2}, 0));
  EXPECT(partition_holds(numerical::tensor<double> {{0, 3}}, {}, 0));
  bool refused {false};
  try {
    static_cast<void>(numerical::partition(wide, {40}, 1));
  }
  catch (const numerical::index_error&) {
    refused = true;
  }
  EXPECT(refused);
}

// vdot sums conj(a) b by the chunks of a dot product, the conjugate taken in each chunk's sums (no conjugated copy):
// the chunks' own conjugated dot products added in order, one chunk and three; reals by the plain dot product
TEST(vdot_conjugates_in_the_sum)
{
  namespace d = numerical::detail;
  using cplx  = std::complex<double>;
  for (const std::size_t count : {std::size_t {5}, (2 * d::dot_chunk) + 7}) {
    std::vector<cplx>   a;
    std::vector<cplx>   b;
    std::vector<double> r;
    for (std::size_t i {0}; i < count; ++i) {
      a.emplace_back(std::sin(static_cast<double>(i)), std::cos(static_cast<double>(i) * 0.7));
      b.emplace_back(std::cos(static_cast<double>(i) * 1.3), -std::sin(static_cast<double>(i) * 0.4));
      r.push_back(std::sin(static_cast<double>(i) * 0.9));
    }
    cplx expected {};
    for (std::size_t first {0}; first < count; first += d::dot_chunk) {
      expected += d::conjugate_dot_product(a.data() + first, b.data() + first, std::min(d::dot_chunk, count - first));
    }
    const numerical::tensor<cplx>   ta {{count}, numerical::buffer<cplx>(a.begin(), a.end())};
    const numerical::tensor<cplx>   tb {{count}, numerical::buffer<cplx>(b.begin(), b.end())};
    const numerical::tensor<double> tr {{count}, numerical::buffer<double>(r.begin(), r.end())};
    EXPECT(numerical::vdot(ta, tb) == expected);
    EXPECT(numerical::vdot(tr, tr) == d::chunked_dot(r.data(), r.data(), count));
    cplx plain {};
    for (std::size_t i {0}; i < count; ++i) {
      plain += std::conj(a[i]) * b[i];
    }
    EXPECT(std::abs(numerical::vdot(ta, tb) - plain) <= 1e-12 * static_cast<double>(count));
  }
  bool refused {false};
  try {
    static_cast<void>(numerical::vdot(numerical::tensor<double> {{2}, {1.0, 2.0}}, numerical::tensor<double> {{3}, {1.0, 2.0, 3.0}}));
  }
  catch (const numerical::dimension_error&) {
    refused = true;
  }
  EXPECT(refused);
}

// A range of sliced is checked without overflow: a step of PTRDIFF_MIN or PTRDIFF_MAX past the first index refused,
// as a count no axis holds; a range of one index takes any step, its stride the axis's.
TEST(sliced_checks_its_ranges_without_overflow)
{
  using numerical::detail::range_within;
  constexpr std::ptrdiff_t least {std::numeric_limits<std::ptrdiff_t>::min()};
  constexpr std::ptrdiff_t most  {std::numeric_limits<std::ptrdiff_t>::max()};
  EXPECT(range_within(9, 0, least, 4)); // no index
  EXPECT(!range_within(4, 1, 1, 4));    // first outside
  EXPECT(range_within(1, 3, 1, 4));     // 1, 2, 3
  EXPECT(!range_within(1, 4, 1, 4));    // 1 to 4
  EXPECT(range_within(3, 2, -3, 4));    // 3, 0
  EXPECT(!range_within(3, 2, -4, 4));   // 3, -1
  EXPECT(range_within(2, 5, 0, 4));     // a step of 0 stays
  EXPECT(!range_within(0, 3, least, 4));
  EXPECT(!range_within(3, 3, most, 4));
  EXPECT(!range_within(0, std::numeric_limits<std::size_t>::max(), 1, 4));
  const tensor<double> line {numerical::arange(0.0, 10.0, 1.0)};
  EXPECT_THROWS((void)line.sliced({0}, {least}, {3}, std::vector<bool> {true}), numerical::index_error);
  EXPECT_THROWS((void)line.sliced({9}, {most}, {2}, std::vector<bool> {true}), numerical::index_error);
  constexpr std::size_t huge    {std::numeric_limits<std::size_t>::max() / 2};
  const tensor<double>  grid    {numerical::arange(0.0, 4.0, 1.0).reshaped({2, 2})};
  EXPECT_THROWS((void)grid.sliced({0, 0}, {0, 0}, {huge, 4}, std::vector<bool> {true, true}), numerical::dimension_error); // a step of 0 repeating past std::size_t
  const tensor<double> repeated {grid.sliced({1, 0}, {0, 1}, {1000, 2}, std::vector<bool> {true, true})};                  // as many as wanted within it
  EXPECT(repeated.size() == 2000 && repeated.data()[1999] == 3.0);
  const tensor<double> one      {line.sliced({7}, {least}, {1}, std::vector<bool> {true})};
  EXPECT(one.shape() == std::vector<std::size_t> {1} && one.data()[0] == 7.0);
}

// materialized: a tensor holding its elements row-major is itself (the same buffer); a view's is its row-major copy,
// contiguous, which keeps the view's base no longer (its only owner once the view and the base are gone).
TEST(a_view_materialized_holds_its_own_elements)
{
  const tensor<double> base {numerical::arange(0.0, 2400.0, 1.0).reshaped({40, 60})}; // past a copied slice's 512
  const tensor<double> same {base.materialized()};
  EXPECT(same.contiguous() && same.data().data() == base.data().data());
  const tensor<double> rows {base.sliced({0, 0}, {2, 1}, {20, 60}, std::vector<bool> {true, true})};
  const tensor<double> copy {rows.materialized()};
  EXPECT(!rows.contiguous() && copy.contiguous() && copy.shape() == rows.shape() && copy.data() == rows.data());
  EXPECT(copy.data().data() != base.data().data());
}

namespace {
  template <typename T>
  bool same_bits_as_tensor(const tensor<T>& x,
                           const tensor<T>& y)
  {
    return x.shape() == y.shape() && std::memcmp(x.data().data(), y.data().data(), x.size() * sizeof(T)) == 0;
  }

  // einsum of transposed views against the same of their copies, to the bit: a matrix's sums over one axis, its
  // diagonal and its permutation read on the matrix under it; a sum of every element, a rank-3 view and the pairs
  // keep the copy.
  template <typename T>
  bool einsum_of_views_gives_their_copies_bits()
  {
    const auto made = [](std::vector<std::size_t> extents, int seed) {
                        const std::size_t    count {numerical::detail::element_count(extents.begin(), extents.end())};
                        numerical::buffer<T> values(count);
                        for (std::size_t i {0}; i < count; ++i) {
                          values[i] = T(std::sin(static_cast<double>((i * 7) + static_cast<std::size_t>(seed))) / 3.0);
                        }
                        return tensor<T> {std::move(extents), std::move(values)};
                      };
    const tensor<T> a        {made({37, 53}, 1)};
    const tensor<T> b        {made({53, 41}, 2)};
    const tensor<T> cube     {made({11, 13, 17}, 3)};
    const tensor<T> square   {made({29, 29}, 4)};
    const tensor<T> a_t      {a.transposed({1, 0})};
    const tensor<T> square_t {square.transposed({1, 0})};
    const tensor<T> cube_t   {cube.transposed({2, 0, 1})};
    bool            same     {true};
    for (const char* spec : {"ij->j", "ij->i", "ij->", "ij->ji", "ij->ij"}) {
      same = same && same_bits_as_tensor(numerical::einsum(spec, a_t), numerical::einsum(spec, a_t.materialized()));
    }
    for (const char* spec : {"ii->i", "ii->"}) {
      same = same && same_bits_as_tensor(numerical::einsum(spec, square_t), numerical::einsum(spec, square_t.materialized()));
    }
    for (const char* spec : {"ijk->k", "ijk->ij", "ijk->"}) {
      same = same && same_bits_as_tensor(numerical::einsum(spec, cube_t), numerical::einsum(spec, cube_t.materialized()));
    }
    same = same && same_bits_as_tensor(numerical::einsum("ji,jk->ik", a_t, b), numerical::einsum("ji,jk->ik", a_t.materialized(), b));
    return same;
  }
} // namespace

TEST(einsum_reads_a_transposed_matrix_where_it_lies_with_its_copys_bits)
{
  const tensor<int>      grid            {{3, 4}, numerical::buffer<int> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  const tensor<int>      grid_t          {grid.transposed({1, 0})};
  const tensor<unsigned> unsigned_grid   {{3, 4}, numerical::buffer<unsigned> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  const tensor<unsigned> unsigned_grid_t {unsigned_grid.transposed({1, 0})};
  for (const char* spec : {"ij->j", "ij->", "ij->ji"}) { // integers: the same sums and the same order
    EXPECT(numerical::einsum(spec, grid_t) == numerical::einsum(spec, grid_t.materialized()));
    EXPECT(numerical::einsum(spec, unsigned_grid_t) == numerical::einsum(spec, unsigned_grid_t.materialized()));
  }
  EXPECT(einsum_of_views_gives_their_copies_bits<double>());
  EXPECT(einsum_of_views_gives_their_copies_bits<float>());
  EXPECT(einsum_of_views_gives_their_copies_bits<std::complex<double>>());
}

// Views that start at the first element and count as many as their buffer without reaching each once: a step of 0
// repeating a row or a column, beside a reversed axis and a transpose of each, each element read through the view's
// strides (t(i, j)), not through a copy; and below and past each size that changes the path (a row of 64 bytes, 512
// elements, 2^14).
TEST(views_that_repeat_elements_are_not_taken_for_a_transpose)
{
  struct slice
  {
    std::array<std::size_t, 2>    starts;
    std::array<std::ptrdiff_t, 2> steps;
  };
  bool                                 right   {true};
  constexpr std::array<std::size_t, 3> lengths {4, 300, 9000};
  for (const std::size_t n : lengths) {
    std::vector<double> values(2 * n);
    std::iota(values.begin(), values.end(), 0.0);
    const numerical::tensor<double> t      {{2, n}, values};
    const std::array<slice, 4>      slices {slice {.starts = {0, 0}, .steps = {0, 1}}, slice {.starts = {0, 0}, .steps = {1, 0}},
                                            slice {.starts = {1, 0}, .steps = {-1, 1}}, slice {.starts = {0, n - 1}, .steps = {1, -1}}};
    for (const slice& s : slices) {
      const auto at {[&values, &s, n](std::size_t i, std::size_t j) {
                       const auto row {static_cast<std::ptrdiff_t>(s.starts[0]) + (s.steps[0] * static_cast<std::ptrdiff_t>(i))};
                       const auto col {static_cast<std::ptrdiff_t>(s.starts[1]) + (s.steps[1] * static_cast<std::ptrdiff_t>(j))};
                       return values.at((static_cast<std::size_t>(row) * n) + static_cast<std::size_t>(col));
                     }};
      const numerical::tensor<double> v           {t.sliced({s.starts[0], s.starts[1]}, {s.steps[0], s.steps[1]}, {2, n}, std::vector<bool> {true, true})};
      const numerical::tensor<double> vt          {v.transposed()};
      const numerical::tensor<double> copied      {v.materialized()};
      const numerical::tensor<double> doubled     {v + v};
      const numerical::tensor<double> doubled_t   {vt + vt};
      const numerical::tensor<double> flipped     {einsum("ij->ji", v)};
      const numerical::tensor<double> columns     {einsum("ij->j", v)};
      const numerical::tensor<double> rows        {einsum("ij->i", v)};
      const numerical::tensor<double> column_sums {numerical::sum(v, 0)};
      const numerical::tensor<double> row_sums    {numerical::sum(vt, 0)};
      std::array<double, 2>           row_total   {0.0, 0.0};
      for (std::size_t j {0}; j < n; ++j) {
        right         = right && v.data()[n + j] == at(1, j) && copied.data()[n + j] == at(1, j) && vt.data()[(2 * j) + 1] == at(1, j);
        right         = right && doubled({1, j}) == 2.0 * at(1, j) && doubled_t({j, 1}) == 2.0 * at(1, j);
        right         = right && flipped({j, 1}) == at(1, j) && columns({j}) == at(0, j) + at(1, j) && column_sums({j}) == columns({j});
        row_total[0] += at(0, j);
        row_total[1] += at(1, j);
      }
      right = right && rows({0}) == row_total[0] && rows({1}) == row_total[1] && row_sums({1}) == row_total[1];
    }
  }
  EXPECT(right);
}
