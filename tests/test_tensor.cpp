// The N-dimensional tensor and Einstein summation: construction (incl. from the
// vector/matrix special cases), access, equality, tolerance comparison, and
// einsum across its many forms (matrix product, dot, transpose, trace, diagonal,
// reductions, outer product, multi-axis contraction, three operands, mixed-case
// labels) together with every malformed-spec error path.
//
// Tensor brace literals are kept in named locals; the one literal that appears
// inside a macro is wrapped in an extra pair of parentheses. einsum(...) calls
// inside EXPECT macros are safe as-is (their commas sit inside their own parens
// or a string literal).
#include <cmath>
#include <complex>
#include <cstddef>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <tuple>
#include <string>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::approx_equal;
  using numerical::einsum;
  using numerical::matrix;
  using numerical::tensor;
  using numerical::transpose;
  using numerical::vector;
} // namespace

TEST(construction_shape_and_access)
{
  const tensor<int> t {{2, 3}, {1, 2, 3, 4, 5, 6}};
  EXPECT_EQ(t.rank(), 2U);
  EXPECT_EQ(t.size(), 6U);
  EXPECT((t.shape() == std::vector<std::size_t> {2, 3}));
  EXPECT_EQ(t({0, 0}), 1);
  EXPECT_EQ(t({1, 2}), 6);

  const tensor<int> cube {{2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7}};
  EXPECT_EQ(cube({1, 0, 1}), 5);

  const tensor<int> scalar {std::vector<std::size_t> {}, std::vector<int> {42}};
  EXPECT_EQ(scalar.rank(), 0U);
  EXPECT_EQ(scalar({}), 42);
}

TEST(buffer_size_mismatch_throws)
{
  EXPECT_THROWS((tensor<int> {std::vector<std::size_t> {2, 2}, std::vector<int> {1, 2, 3}}),
                numerical::dimension_error);
}

TEST(zero_dimension_tensor_is_valid_and_empty)
{
  // A 0 in the shape gives 0 elements — and exercises checked_mul's a == 0 short-circuit
  // (which also keeps its max / a division well-defined).
  const tensor<int> empty {std::vector<std::size_t> {2, 0}, std::vector<int> {}};
  EXPECT_EQ(empty.rank(), 2U);
  EXPECT_EQ(empty.size(), 0U);
}

TEST(shape_product_overflow_throws)
{
  // An absurd shape whose element-count product overflows size_t is rejected, not
  // silently wrapped to a small (mismatching) total.
  const std::size_t huge {std::size_t {1} << 40};
  EXPECT_THROWS((tensor<int> {std::vector<std::size_t> {huge, huge}, std::vector<int> {1}}),
                numerical::dimension_error);
}

TEST(built_from_the_vector_and_matrix_special_cases)
{
  const vector<int> v  {1, 2, 3};
  const tensor<int> tv {v};
  EXPECT_EQ(tv.rank(), 1U);
  EXPECT((tv.shape() == std::vector<std::size_t> {3}));
  EXPECT((tv.data() == std::vector<int> {1, 2, 3}));

  const matrix<int> m  {{1, 2, 3}, {4, 5, 6}};
  const tensor<int> tm {m};
  EXPECT_EQ(tm.rank(), 2U);
  EXPECT((tm.shape() == std::vector<std::size_t> {2, 3}));
  EXPECT_EQ(tm({1, 2}), 6);
}

TEST(equality_is_shape_and_element_wise)
{
  const tensor<int> a           {{2, 2}, {1, 2, 3, 4}};
  const tensor<int> same        {{2, 2}, {1, 2, 3, 4}};
  const tensor<int> other_shape {{4}, {1, 2, 3, 4}};
  const tensor<int> other_data  {{2, 2}, {1, 2, 3, 9}};
  EXPECT(a == same);
  EXPECT(!(a == other_shape)); // shape differs
  EXPECT(!(a == other_data));  // same shape, an element differs
}

TEST(approx_equal_compares_within_a_tolerance)
{
  const tensor<double> a     {{2}, {1.0, 2.0}};
  const tensor<double> close {{2}, {1.0 + 1e-12, 2.0 - 1e-12}};
  const tensor<double> off   {{2}, {1.0, 2.5}};
  const tensor<double> wrong {{3}, {1.0, 2.0, 3.0}};
  EXPECT(approx_equal(a, close));                                                 // within default tolerance
  EXPECT(!approx_equal(a, off));                                                  // an element exceeds it
  EXPECT(!approx_equal(a, wrong));                                                // shapes differ
  const double nan {std::numeric_limits<double>::quiet_NaN()};
  EXPECT(!approx_equal(a, tensor<double> {{2}, {nan, 2.0}}));                     // a NaN is close to nothing
  const double inf {std::numeric_limits<double>::infinity()};
  EXPECT(approx_equal(tensor<double> {{1}, {inf}}, tensor<double> {{1}, {inf}})); // equal infinities
  EXPECT(!approx_equal(tensor<double> {{1}, {inf}}, tensor<double> {{1}, {-inf}}));
}

TEST(approx_equal_compares_complex_tensors)
{
  using c = std::complex<double>;
  const tensor<c> a {{2}, {c {1.0, 1.0}, c {2.0, 0.0}}};
  EXPECT(approx_equal(a, a));                                                     // equal
  EXPECT(approx_equal(a, tensor<c> {{2}, {c {1.0, 1.0 + 1e-12}, c {2.0, 0.0}}})); // close
  EXPECT(!approx_equal(a, tensor<c> {{2}, {c {1.0, 1.5}, c {2.0, 0.0}}}));        // far
  EXPECT(!approx_equal(a, tensor<c> {{1}, {c {1.0, 1.0}}}));                      // shape mismatch
}

// Without '->' the output is every label that appears once, in ASCII order, as NumPy reads the spec.
TEST(einsum_implicit_mode_as_numpy)
{
  const matrix<int> a  {{1, 2, 3}, {4, 5, 6}};
  const matrix<int> b  {{1, 0}, {0, 1}, {2, 2}};
  const matrix<int> sq {{1, 2}, {3, 4}};
  const tensor<int> ta {a};
  const tensor<int> tb {b};
  EXPECT(einsum("ij,jk", ta, tb) == einsum("ij,jk->ik", ta, tb));        // a product
  EXPECT(einsum("ba", ta) == einsum("ba->ab", ta));                      // a transpose
  EXPECT(einsum("ij", ta) == ta);
  EXPECT_EQ(einsum("ii", tensor<int> {sq}).data()[0], 5);                // a trace: i twice, no output
  EXPECT(einsum("Ba", ta).shape() == (std::vector<std::size_t> {2, 3})); // B before a in ASCII
  EXPECT(einsum("bA", ta).shape() == (std::vector<std::size_t> {3, 2}));
  const tensor<int> tv {vector<int> {1, 2, 3}};
  EXPECT_EQ(einsum("i,i", tv, tv).data()[0], 14);                        // a dot product
}

TEST(einsum_matrix_product)
{
  const matrix<int> a        {{1, 2}, {3, 4}};
  const matrix<int> b        {{5, 6}, {7, 8}};
  const tensor<int> ta       {a};
  const tensor<int> tb       {b};
  const tensor<int> expected {a * b};
  EXPECT(einsum("ij,jk->ik", ta, tb) == expected);
}

// A product whose result is empty, rows or columns none, is an empty tensor of its shape: with no element it is no
// dot product (one read past an empty operand).
TEST(einsum_with_an_empty_result)
{
  const tensor<double> rows {{0, 3}, {}};
  const tensor<double> wide {{3, 5}, std::vector<double>(15, 1.0)};
  const tensor<double> tall {{3, 3}, std::vector<double>(9, 1.0)};
  const tensor<double> none {{3, 0}, {}};
  EXPECT(einsum("ij,jk->ik", rows, wide).shape() == (std::vector<std::size_t> {0, 5}));
  EXPECT(einsum("ij,jk->ik", tall, none).shape() == (std::vector<std::size_t> {3, 0}));
  EXPECT(einsum("a,ab->b", tensor<double> {{1}, {2.0}}, tensor<double> {{1, 0}, {}}).shape() == (std::vector<std::size_t> {0}));
}

TEST(einsum_tolerates_whitespace_in_the_spec)
{
  // NumPy accepts spaces anywhere in the spec; einsum strips them before parsing,
  // so "ij, jk -> ik" is equivalent to "ij,jk->ik".
  const matrix<int> a  {{1, 2}, {3, 4}};
  const matrix<int> b  {{5, 6}, {7, 8}};
  const tensor<int> ta {a};
  const tensor<int> tb {b};
  EXPECT(einsum("ij, jk -> ik", ta, tb) == einsum("ij,jk->ik", ta, tb));
}

TEST(einsum_dot_product_yields_a_scalar)
{
  const vector<int> v  {1, 2, 3};
  const tensor<int> tv {v};
  const tensor<int> r  {einsum("i,i->", tv, tv)};
  EXPECT_EQ(r.rank(), 0U);
  EXPECT_EQ(r.data()[0], 14); // 1 + 4 + 9
}

// A dot product of several chunks of numerical::detail::dot_chunk elements: whole numbers keep every partial sum
// exact, so any order gives the ascending sum; values that round give the same bits on one thread as on all of them.
TEST(einsum_dot_product_sums_by_fixed_chunks)
{
  const std::size_t   n {(3 * numerical::detail::dot_chunk) + 77};
  std::vector<double> whole(n);
  std::vector<double> other(n);
  std::vector<double> rough(n);
  double              exact {0.0};
  for (std::size_t i {0}; i < n; ++i) {
    whole[i]  = static_cast<double>(i % 7) - 3.0;
    other[i]  = static_cast<double>((i % 5) + 1);
    rough[i]  = 1.0 / static_cast<double>(i + 3);
    exact    += whole[i] * other[i];
  }
  const tensor<double> tw {{n}, whole};
  const tensor<double> to {{n}, other};
  const tensor<double> tr {{n}, rough};
  EXPECT(einsum("i,i->", tw, to).data()[0] == exact);
  const double all        {einsum("i,i->", tr, to).data()[0]};
  numerical::set_max_threads(1);
  const double one        {einsum("i,i->", tr, to).data()[0]};
  numerical::set_max_threads(0);
  EXPECT(all == one);
  const tensor<double> empty {{std::size_t {0}}, std::vector<double> {}};
  EXPECT(einsum("i,i->", empty, empty).data()[0] == 0.0);
}

TEST(einsum_transpose_and_mixed_case_labels)
{
  const matrix<int> m        {{1, 2, 3}, {4, 5, 6}};
  const tensor<int> tm       {m};
  const tensor<int> expected {transpose(m)};
  EXPECT(einsum("ij->ji", tm) == expected);
  EXPECT(einsum("aB->Ba", tm) == expected); // uppercase labels are valid too
}

TEST(einsum_permutes_by_tiles_and_copies_an_identity)
{
  // Larger than a tile each way and ragged at both edges; a permutation of three axes and the identity too.
  const std::size_t   rows {(numerical::detail::transpose_tile * 2) + 5};
  const std::size_t   cols {numerical::detail::transpose_tile + 3};
  std::vector<double> values(rows * cols);
  for (std::size_t i {0}; i < values.size(); ++i) {
    values[i] = static_cast<double>(i) * 0.5;
  }
  const tensor<double> t       {{rows, cols}, values};
  const tensor<double> flipped {einsum("ij->ji", t)};
  EXPECT(flipped.shape() == (std::vector<std::size_t> {cols, rows}));
  bool same                    {true};
  for (std::size_t r {0}; r < rows; ++r) {
    for (std::size_t c {0}; c < cols; ++c) {
      same = same && flipped.data()[(c * rows) + r] == values[(r * cols) + c];
    }
  }
  EXPECT(same);
  EXPECT(einsum("ij->ij", t) == t);
  const tensor<double> cube {{2, 3, 4}, std::vector<double>(24, 1.0)};
  EXPECT(einsum("abc->cab", cube).shape() == (std::vector<std::size_t> {4, 2, 3}));
}

// A transpose whose rows lie 8 KiB apart, walked by pieces of 8 on every processor (transpose_piece), ragged at both
// edges, through einsum and the matrix's transpose, either way round.
TEST(a_transpose_of_aliased_rows_moves_every_element)
{
  for (const auto& [rows, cols] : {std::pair<std::size_t, std::size_t> {37, 1024}, {1024, 37}}) {
    std::vector<double> values(rows * cols);
    for (std::size_t i {0}; i < values.size(); ++i) {
      values[i] = static_cast<double>(i) * 0.25;
    }
    const tensor<double>              t       {{rows, cols}, values};
    const tensor<double>              flipped {einsum("ij->ji", t)};
    const numerical::matrix<double>   m       {rows, cols, values};
    const numerical::matrix<double>   mt      {numerical::transpose(m)};
    bool                              same    {mt.rows() == cols && mt.cols() == rows};
    for (std::size_t r {0}; r < rows; ++r) {
      for (std::size_t c {0}; c < cols; ++c) {
        same = same && flipped.data()[(c * rows) + r] == values[(r * cols) + c] && mt(c, r) == values[(r * cols) + c];
      }
    }
    EXPECT(same);
  }
}

TEST(einsum_sums_one_axis_under_several_free_ones)
{
  // The general loop's odometer over two outer free labels (a, b), the last free one (c) swept inside.
  std::vector<double> values(std::size_t {120}); // 2 × 3 × 4 × 5
  for (std::size_t i {0}; i < values.size(); ++i) {
    values[i] = static_cast<double>(i);
  }
  const tensor<double> t    {{2, 3, 4, 5}, values};
  const tensor<double> sums {einsum("abcd->abc", t)};
  EXPECT(sums.shape() == (std::vector<std::size_t> {2, 3, 4}));
  bool same                 {true};
  for (std::size_t abc {0}; abc < 24; ++abc) {
    const double first {static_cast<double>(abc * 5)};
    same = same && sums.data()[abc] == (5.0 * first) + 10.0; // first + (first+1) + … + (first+4)
  }
  EXPECT(same);
  // The same for integers, whose einsum the coverage counts apart.
  const tensor<int> ints {{2, 3, 4, 5}, std::vector<int>(120, 1)};
  EXPECT(einsum("abcd->abc", ints) == (tensor<int> {{2, 3, 4}, std::vector<int>(24, 5)}));
  // Summed labels apart (a and c, around b): the general loop, an odometer over the outer free label, the last swept.
  const tensor<double> apart      {einsum("abcd->bd", t)};
  bool                 apart_same {apart.shape() == (std::vector<std::size_t> {3, 5})};
  for (std::size_t b {0}; b < 3; ++b) {
    for (std::size_t d {0}; d < 5; ++d) {
      double sum {0.0};
      for (std::size_t a {0}; a < 2; ++a) {
        for (std::size_t c {0}; c < 4; ++c) {
          sum += static_cast<double>((a * 60) + (b * 20) + (c * 5) + d);
        }
      }
      apart_same = apart_same && apart.data()[(b * 5) + d] == sum;
    }
  }
  EXPECT(apart_same);
  EXPECT(einsum("abcd->bd", ints) == (tensor<int> {{3, 5}, std::vector<int>(15, 8)}));
}

// sum_over_run, the reduction a NumPy array is read in place for: a run of axes summed into the others in order, and
// nothing for a spec that sums nothing, sums labels apart or reorders the output.
TEST(sum_over_run_takes_a_run_of_axes_and_declines_the_rest)
{
  const std::vector<double>      values {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2 × 3
  const std::vector<std::size_t> shape  {2, 3};
  const auto                     rows   {numerical::detail::sum_over_run<double>("ij", "i", shape, values.data())};
  EXPECT(rows.has_value() && rows->shape() == (std::vector<std::size_t> {2}) && rows->data()[0] == 6.0 && rows->data()[1] == 15.0);
  const auto columns                    {numerical::detail::sum_over_run<double>("ij", "j", shape, values.data())};
  EXPECT(columns.has_value() && columns->data()[2] == 9.0);
  EXPECT(!numerical::detail::sum_over_run<double>("ij", "ij", shape, values.data()).has_value()); // nothing summed
  EXPECT(!numerical::detail::sum_over_run<double>("ij", "ji", shape, values.data()).has_value()); // nothing summed either
  const std::vector<std::size_t> cube {1, 2, 3};
  EXPECT(!numerical::detail::sum_over_run<double>("ijk", "j", cube, values.data()).has_value());  // i and k apart
}

namespace {
  // `a` (inner) times the row-major B (inner × cols) as add_product sums it, B's rows scaled and added in order in
  // chunks of vector_chunk columns: the arithmetic the column kernels must give to the bit.
  template <typename T>
  std::vector<T> rows_added_in_order(const std::vector<T>& a,
                                     const std::vector<T>& b,
                                     std::size_t           cols)
  {
    std::vector<T> y(cols, T {});
    for (std::size_t first {0}; first < cols; first += numerical::detail::vector_chunk) {
      const std::size_t count {std::min(numerical::detail::vector_chunk, cols - first)};
      for (std::size_t k {0}; k < a.size(); ++k) {
        numerical::detail::add_product(y.data() + first, a[k], b.data() + (k * cols) + first, count);
      }
    }
    return y;
  }

  // canonical_product of `a` (inner) and the row-major B, and vector_times_transposed of `a` and B given as its
  // transpose (cols × inner, rows padded): B's rows added in order, to the bit.
  template <typename T>
  bool vector_products_add_the_rows_in_order(std::size_t inner,
                                             std::size_t cols)
  {
    using R = decltype(std::abs(T {})); // the parts' type
    const auto        part {[](double value) { return static_cast<R>(value); }};
    const std::size_t ldbt {inner + 3};
    std::vector<T>    a(inner);
    std::vector<T>    b(inner * cols);
    std::vector<T>    bt(cols * ldbt, T(-9)); // the padding is never read
    for (std::size_t k {0}; k < inner; ++k) {
      a[k] = T(part(std::sin(static_cast<double>(k) + 0.5)));
      if constexpr (!std::is_floating_point_v<T>) {
        a[k] += T(R {0}, part(std::cos(static_cast<double>(k) * 1.3)));
      }
      for (std::size_t j {0}; j < cols; ++j) {
        T value {T(part(std::cos((static_cast<double>(k) * 0.7) + (static_cast<double>(j) * 1.9))))};
        if constexpr (!std::is_floating_point_v<T>) {
          value += T(R {0}, part(std::sin(static_cast<double>(j + k))));
        }
        b[(k * cols) + j]   = value;
        bt[(j * ldbt) + k]  = value;
      }
    }
    const std::vector<T>       want       {rows_added_in_order(a, b, cols)};
    const numerical::tensor<T> canonical  {numerical::detail::canonical_product<T>(a.data(), inner, b.data(), cols, 1, inner, cols, {cols})};
    const numerical::buffer<T> transposed {numerical::detail::vector_times_transposed(a.data(), bt.data(), ldbt, inner, cols)};
    return std::equal(canonical.data().begin(), canonical.data().end(), want.begin(), want.end()) &&
           std::equal(transposed.begin(), transposed.end(), want.begin(), want.end());
  }
} // namespace

namespace
{
  // canonical_product of A (rows × inner) and B (inner × cols) laid with their rows apart (NaN between, never read)
  // against the same operands together: the same bits on every route (a matrix times a vector, a vector times a narrow
  // or a wide matrix, the matrix product, a dot product).
  template <typename T>
  bool rows_apart_give_the_contiguous_bits(std::size_t rows,
                                           std::size_t inner,
                                           std::size_t cols)
  {
    using R = decltype(std::abs(T {})); // the parts' type
    const std::size_t lda {inner + 3};
    const std::size_t ldb {cols + 5};
    std::vector<T>    a(rows * inner);
    std::vector<T>    b(inner * cols);
    std::vector<T>    a_apart(rows * lda, T(std::numeric_limits<R>::quiet_NaN()));
    std::vector<T>    b_apart(inner * ldb, T(std::numeric_limits<R>::quiet_NaN()));
    for (std::size_t i {0}; i < rows; ++i) {
      for (std::size_t k {0}; k < inner; ++k) {
        a[(i * inner) + k] = a_apart[(i * lda) + k] = T(static_cast<R>(std::sin(static_cast<double>((i * 7) + k) + 0.5)));
      }
    }
    for (std::size_t k {0}; k < inner; ++k) {
      for (std::size_t j {0}; j < cols; ++j) {
        b[(k * cols) + j] = b_apart[(k * ldb) + j] = T(static_cast<R>(std::cos((static_cast<double>(k) * 0.7) + (static_cast<double>(j) * 1.9))));
      }
    }
    // A vector (one row of A, one column of B) lies together: its ld unused, given as its length.
    const std::size_t          b_ld     {cols == 1 ? 1 : ldb};
    const T* const             b_at     {cols == 1 ? b.data() : b_apart.data()};
    const numerical::tensor<T> together {numerical::detail::canonical_product<T>(a.data(), inner, b.data(), cols, rows, inner, cols, {rows * cols})};
    const numerical::tensor<T> apart    {numerical::detail::canonical_product<T>(a_apart.data(), lda, b_at, b_ld, rows, inner, cols, {rows * cols})};
    return together.data().size() == apart.data().size() && std::equal(together.data().begin(), together.data().end(), apart.data().begin());
  }
} // namespace

// Operands whose rows lie apart (every other row of a larger NumPy array, read where it lies) give the bits of the
// same operands together, on every route of canonical_product, in each type, on every path this processor runs.
TEST(a_product_of_rows_apart_gives_the_contiguous_bits)
{
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue; // a path this processor does not run
    }
    // the rows of the second longer than a chunk of the dot product (chunked_dot's partial sums)
    for (const auto& [rows, inner, cols] : {std::array<std::size_t, 3> {40, 50, 1}, {300, 700, 1}, {3, numerical::detail::dot_chunk + 9, 1},
                                            {1, 37, 5}, {1, 300, 200}, {169, 579, 2}, {64, 64, 64}, {130, 101, 600}, {1, 9, 1}}) {
      EXPECT(rows_apart_give_the_contiguous_bits<double>(rows, inner, cols));
      EXPECT(rows_apart_give_the_contiguous_bits<float>(rows, inner, cols));
      EXPECT(rows_apart_give_the_contiguous_bits<std::complex<double>>(rows, inner, cols));
    }
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

// A vector times a matrix, row-major (narrow: its columns' sums held) and given as its transpose (a Fortran-ordered
// NumPy matrix): B's rows added in order, the same bits, in each type, on every path this processor runs; every width
// of a partial group of columns, the narrow kernel's last width and the first past it, several chunks, a depth of one
// and an empty one (zeros).
TEST(a_vector_times_a_matrix_adds_its_rows_in_order)
{
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue; // a path this processor does not run
    }
    std::vector<std::array<std::size_t, 2>> shapes {{406, 583}, {200, 130}, {764, 2}};
    for (const std::size_t inner : {std::size_t {0}, std::size_t {1}, std::size_t {5}, std::size_t {37}}) {
      for (std::size_t cols {2}; cols <= numerical::detail::column_sums_below; ++cols) {
        shapes.push_back({inner, cols});
      }
    }
    for (const auto& [inner, cols] : shapes) {
      EXPECT(vector_products_add_the_rows_in_order<double>(inner, cols));
      EXPECT(vector_products_add_the_rows_in_order<float>(inner, cols));
      EXPECT(vector_products_add_the_rows_in_order<std::complex<double>>(inner, cols));
      EXPECT(vector_products_add_the_rows_in_order<std::complex<float>>(inner, cols));
    }
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

// One operand summed over a run of its axes (ij->, ij->i, ij->j, ijk->ik): each output element the pairwise sum of its
// run, the bits of sum() and of sum(t, axis), every column summed down a block by rows as alone.
TEST(einsum_reductions_sum_pairwise_as_sum_does)
{
  std::vector<double> values(std::size_t {300} *700);
  std::uint64_t       state              {7};
  for (double& x : values) {
    state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
    x     = (static_cast<double>(state >> 11U) * 0x1.0p-53) - 0.5;
  }
  const tensor<double> m {{300, 700}, values};
  EXPECT(einsum("ij->", m).data()[0] == numerical::sum(m));
  EXPECT(einsum("ij->i", m) == numerical::sum(m, 1));
  EXPECT(einsum("ij->j", m) == numerical::sum(m, 0));
  const tensor<double> cube {{6, 50, 700}, values};
  EXPECT(einsum("ijk->ik", cube) == numerical::sum(cube, 1));
  EXPECT(einsum("ijk->i", cube).shape() == (std::vector<std::size_t> {6}));
  // Every column of sum(t, 0) as pairwise_sum walks it down alone, across the shapes the halving takes (fewer than 8
  // rows, up to 128, more) and widths either side of the 512 columns summed at once.
  bool same {true};
  for (const std::size_t rows : {std::size_t {3}, std::size_t {100}, std::size_t {129}, std::size_t {300}}) {
    for (const std::size_t cols : {std::size_t {1}, std::size_t {5}, std::size_t {512}, std::size_t {700}}) {
      const tensor<double> block {{rows, cols}, std::vector<double>(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(rows * cols))};
      const tensor<double> sums  {numerical::sum(block, 0)};
      for (std::size_t c {0}; c < cols; ++c) {
        same = same && sums.data()[c] == numerical::detail::pairwise_sum(block.data().data() + c, rows, cols);
      }
    }
  }
  EXPECT(same);
  const tensor<double> means {numerical::mean(m, 0)};
  EXPECT(means.data()[3] == numerical::sum(m, 0).data()[3] / 300.0);
}

TEST(einsum_matrix_vector_rows_are_their_dot_products)
{
  // Each row of A·v is that row's dot product with v, four rows at a time and the rest alone; v·A sums its rows in
  // ascending order. A row longer than a chunk takes the chunked sum, as "i,i->" does.
  for (const auto& [rows, n] : std::vector<std::pair<std::size_t, std::size_t>> {{7, 9}, {70, 33}, {3, 130}}) {
    std::vector<double> a(rows * n);
    std::vector<double> v(n);
    std::vector<double> w(rows);
    for (std::size_t i {0}; i < a.size(); ++i) {
      a[i] = 1.0 / static_cast<double>(i + 3);
    }
    for (std::size_t i {0}; i < n; ++i) {
      v[i] = static_cast<double>(i % 5) - 1.5;
    }
    for (std::size_t i {0}; i < rows; ++i) {
      w[i] = static_cast<double>(i % 3) + 0.25;
    }
    const tensor<double> ta   {{rows, n}, a};
    const tensor<double> av   {einsum("ij,j->i", ta, tensor<double> {{n}, v})};
    const tensor<double> wa   {einsum("i,ij->j", tensor<double> {{rows}, w}, ta)};
    bool                 same {true};
    for (std::size_t r {0}; r < rows; ++r) {
      same = same && av.data()[r] == numerical::detail::dot_product(a.data() + (r * n), v.data(), n);
    }
    for (std::size_t c {0}; c < n; ++c) {
      double sum {0.0};
      for (std::size_t r {0}; r < rows; ++r) {
        sum += w[r] * a[(r * n) + c];
      }
      same = same && std::abs(wa.data()[c] - sum) <= 1e-14 * static_cast<double>(rows);
    }
    EXPECT(same);
  }
  const std::size_t    long_row {numerical::detail::dot_chunk + 5};
  std::vector<double>  two(2 * long_row, 1.0);
  const tensor<double> long_av  {einsum("ij,j->i", tensor<double> {{2, long_row}, two}, tensor<double> {{long_row}, std::vector<double>(long_row, 2.0)})};
  EXPECT(long_av.data()[0] == 2.0 * static_cast<double>(long_row) && long_av.data()[1] == long_av.data()[0]);
  const tensor<double> none     {einsum("ij,j->i", tensor<double> {{3, 0}, std::vector<double> {}}, tensor<double> {{0}, std::vector<double> {}})};
  EXPECT(none.data() == (std::vector<double> {0.0, 0.0, 0.0}));
  // Integers, whose einsum the coverage counts apart: rows of A·v, one longer than a chunk, and v·A.
  const tensor<int> im {{3, 4}, std::vector<int> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  EXPECT(einsum("ij,j->i", im, tensor<int> {{4}, std::vector<int> {1, 0, 2, 1}}) == (tensor<int> {{3}, std::vector<int> {11, 27, 43}}));
  EXPECT(einsum("i,ij->j", tensor<int> {{3}, std::vector<int> {1, 1, 2}}, im) == (tensor<int> {{4}, std::vector<int> {24, 28, 32, 36}}));
  EXPECT(einsum("ij,j->i", tensor<int> {{3, 0}, std::vector<int> {}}, tensor<int> {{0}, std::vector<int> {}}) == (tensor<int> {{3}, std::vector<int>(3, 0)}));
  EXPECT(einsum("i,ij->j", tensor<int> {{0}, std::vector<int> {}}, tensor<int> {{0, 4}, std::vector<int> {}}) == (tensor<int> {{4}, std::vector<int>(4, 0)}));
  const std::size_t long_ints {numerical::detail::dot_chunk + 3};
  EXPECT(einsum("ij,j->i", tensor<int> {{2, long_ints}, std::vector<int>(2 * long_ints, 1)}, tensor<int> {{long_ints}, std::vector<int>(long_ints, 3)})
         == (tensor<int> {{2}, std::vector<int>(2, static_cast<int>(3 * long_ints))}));
}

TEST(einsum_trace_and_diagonal)
{
  const matrix<int> m     {{1, 2}, {3, 4}};
  const tensor<int> tm    {m};
  const tensor<int> trace {einsum("ii->", tm)};
  EXPECT_EQ(trace.rank(), 0U);
  EXPECT_EQ(trace.data()[0], 5); // 1 + 4

  const tensor<int> diagonal {einsum("ii->i", tm)};
  const tensor<int> expected {{2}, {1, 4}};
  EXPECT(diagonal == expected);
}

TEST(einsum_reductions)
{
  const matrix<int> m     {{1, 2, 3}, {4, 5, 6}};
  const tensor<int> tm    {m};

  const tensor<int> total {einsum("ij->", tm)};
  EXPECT_EQ(total.data()[0], 21);

  const tensor<int> row_sums      {einsum("ij->i", tm)};
  const tensor<int> rows_expected {{2}, {6, 15}};
  EXPECT(row_sums == rows_expected);

  const tensor<int> col_sums      {einsum("ij->j", tm)};
  const tensor<int> cols_expected {{3}, {5, 7, 9}};
  EXPECT(col_sums == cols_expected);
}

TEST(einsum_outer_product)
{
  const vector<int> v        {1, 2};
  const vector<int> w        {3, 4, 5};
  const tensor<int> tv       {v};
  const tensor<int> tw       {w};
  const matrix<int> outer    {{3, 4, 5}, {6, 8, 10}};
  const tensor<int> expected {outer};
  EXPECT(einsum("i,j->ij", tv, tw) == expected);
}

TEST(einsum_full_contraction_over_two_axes)
{
  // Frobenius inner product: sum over both i and j (a multi-axis contraction).
  const matrix<int> m  {{1, 2, 3}, {4, 5, 6}};
  const tensor<int> tm {m};
  const tensor<int> r  {einsum("ij,ij->", tm, tm)};
  EXPECT_EQ(r.rank(), 0U);
  EXPECT_EQ(r.data()[0], 91); // 1+4+9+16+25+36
}

TEST(einsum_chains_three_operands)
{
  const matrix<int> a        {{1, 2}, {3, 4}};
  const matrix<int> b        {{5, 6}, {7, 8}};
  const matrix<int> c        {{1, 0}, {1, 1}};
  const tensor<int> ta       {a};
  const tensor<int> tb       {b};
  const tensor<int> tc       {c};
  const tensor<int> expected {(a * b) * c};
  EXPECT(einsum("ij,jk,kl->il", ta, tb, tc) == expected);
}

TEST(einsum_over_rank_three_tensors)
{
  const tensor<int> t {{2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7}};
  // Identity: all labels free (outer-free spans two axes — exercises the carry).
  EXPECT(einsum("ijk->ijk", t) == t);
  // Full contraction over three axes: the sum of every element's square.
  const tensor<int> sq {einsum("ijk,ijk->", t, t)};
  EXPECT_EQ(sq.rank(), 0U);
  EXPECT_EQ(sq.data()[0], 140); // 0+1+4+9+16+25+36+49
}

TEST(einsum_matmul_uses_the_tiled_gemm_across_tiles)
{
  // Dimensions spanning several GEMM tiles exercise the blocked kernel and its
  // parallel path (≥ 3 row-blocks); checked against a plain reference matmul.
  const std::size_t rows  {96};
  const std::size_t inner {96};
  const std::size_t cols  {96};
  std::vector<int>  av(rows * inner);
  std::vector<int>  bv(inner * cols);
  for (std::size_t i {0}; i < av.size(); ++i) {
    av[i] = static_cast<int>(i % 7) - 3;
  }
  for (std::size_t i {0}; i < bv.size(); ++i) {
    bv[i] = static_cast<int>(i % 5) - 2;
  }
  std::vector<int> ref(rows * cols, 0);
  for (std::size_t m {0}; m < rows; ++m) {
    for (std::size_t k {0}; k < inner; ++k) {
      for (std::size_t n {0}; n < cols; ++n) {
        ref[(m * cols) + n] += av[(m * inner) + k] * bv[(k * cols) + n];
      }
    }
  }
  const tensor<int> ta       {{rows, inner}, av};
  const tensor<int> tb       {{inner, cols}, bv};
  const tensor<int> expected {{rows, cols}, ref};
  EXPECT(einsum("ij,jk->ik", ta, tb) == expected);
}

TEST(einsum_regroups_non_canonical_contractions)
{
  // None of these match the canonical GEMM layout: their operands and result are regrouped around a product.
  const tensor<int> a             {{2, 2}, {1, 2, 3, 4}};
  const tensor<int> b             {{2, 2}, {5, 6, 7, 8}};
  const tensor<int> a_transpose_b {{2, 2}, {26, 30, 38, 44}}; // "ji,jk->ik": op0 not ending in the contracted label
  const tensor<int> a_b_transpose {{2, 2}, {17, 23, 39, 53}}; // "ij,kj->ik": op1 not starting with it
  const tensor<int> swapped_out   {{2, 2}, {19, 43, 22, 50}}; // "ij,jk->ki": output axes swapped
  EXPECT(einsum("ji,jk->ik", a, b) == a_transpose_b);
  EXPECT(einsum("ij,kj->ik", a, b) == a_b_transpose);
  EXPECT(einsum("ij,jk->ki", a, b) == swapped_out);

  // A label summed by one operand alone leaves the pair to the general engine.
  const tensor<int> v      {{3}, {1, 2, 3}};
  const tensor<int> scalar {std::vector<std::size_t> {}, std::vector<int> {10}};
  const tensor<int> left   {einsum("i,->", v, scalar)}; // op1 is rank-0
  const tensor<int> right  {einsum(",i->", scalar, v)}; // op0 is rank-0
  EXPECT_EQ(left.rank(), 0U);
  EXPECT_EQ(left.data()[0], 60);                        // (1 + 2 + 3) * 10
  EXPECT_EQ(right.data()[0], 60);
}

namespace {
  // A matrix of doubles that round, its element (i, j) from i and j alone.
  numerical::tensor<double> rounding(std::size_t rows,
                                     std::size_t cols,
                                     double      seed)
  {
    numerical::buffer<double> values(rows * cols, 0.0);
    for (std::size_t i {0}; i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        values[(i * cols) + j] = 1.0 / (seed + static_cast<double>((i * 7) + (j * 3)));
      }
    }
    return numerical::tensor<double> {{rows, cols}, std::move(values)};
  }

  // The transpose, by an einsum the general engine takes.
  numerical::tensor<double> transposed(const numerical::tensor<double>& t)
  {
    return einsum("ij->ji", t);
  }
} // namespace

// A pair regrouped around a product takes each element's terms in the order the canonical layout does: the same bits
// as the canonical product of the operands transposed by hand, batch by batch, with the output laid out last.
TEST(einsum_regrouped_pairs_keep_the_canonical_sums)
{
  const tensor<double> a {rounding(37, 29, 3.0)};
  const tensor<double> b {rounding(29, 41, 5.0)};
  EXPECT(einsum("ij,kj->ik", a, transposed(b)) == einsum("ij,jk->ik", a, b));
  EXPECT(einsum("ji,jk->ik", transposed(a), b) == einsum("ij,jk->ik", a, b));
  EXPECT(einsum("ij,jk->ki", a, b) == transposed(einsum("ij,jk->ik", a, b)));
  EXPECT(!(einsum("ij,jk->ki", a, b) == einsum("ij,jk->ik", a, b)));                     // another shape
  EXPECT(!(einsum("ij,jk->ik", a, b) == einsum("ij,jk->ik", a, rounding(29, 41, 6.0)))); // other values
  EXPECT_THROWS(tensor<double>({2}, {1.0, 2.0, 3.0}), numerical::dimension_error);
  // Three batches, stacked: each slice the canonical product of its slices.
  std::vector<double> left;
  std::vector<double> right;
  std::vector<double> wanted;
  for (std::size_t p {0}; p < 3; ++p) {
    const tensor<double> x {rounding(5, 6, 2.0 + static_cast<double>(p))};
    const tensor<double> y {rounding(6, 4, 7.0 + static_cast<double>(p))};
    const tensor<double> z {einsum("ij,jk->ik", x, y)};
    left.insert(left.end(), x.data().begin(), x.data().end());
    right.insert(right.end(), y.data().begin(), y.data().end());
    wanted.insert(wanted.end(), z.data().begin(), z.data().end());
  }
  const tensor<double> stacked {einsum("bij,bjk->bik", tensor<double> {{3, 5, 6}, left}, tensor<double> {{3, 6, 4}, right})};
  EXPECT(stacked == tensor<double> ({3, 5, 4}, wanted));
  // The batch label inside the output: the stacked result laid out anew.
  EXPECT(einsum("bij,bjk->ibk", tensor<double> {{3, 5, 6}, left}, tensor<double> {{3, 6, 4}, right}) ==
         einsum("bik->ibk", stacked));
  // Two summed labels, in ascending order; an outer product; a dot product per batch; an empty extent.
  const tensor<double> cube {{2, 3, 4}, std::vector<double>(24, 0.5)};
  EXPECT(einsum("ijk,jkl->il", cube, tensor<double> {{3, 4, 2}, std::vector<double>(24, 2.0)}) ==
         tensor<double>({2, 2}, std::vector<double>(4, 12.0)));
  EXPECT(einsum("i,j->ij", tensor<double> {{2}, {2.0, 3.0}}, tensor<double> {{3}, {1.0, 4.0, 5.0}}) ==
         tensor<double>({2, 3}, {2.0, 8.0, 10.0, 3.0, 12.0, 15.0}));
  EXPECT(einsum("bi,bi->b", tensor<double> {{2, 3}, {1, 2, 3, 4, 5, 6}}, tensor<double> {{2, 3}, {1, 1, 1, 2, 2, 2}}) ==
         tensor<double>({2}, {6.0, 30.0}));
  EXPECT(einsum("ij,kj->ik", tensor<double> {{0, 3}, std::vector<double> {}}, tensor<double> {{2, 3}, std::vector<double>(6, 1.0)}).data().empty());
  EXPECT(einsum("ij,kj->ik", tensor<double> {{2, 3}, std::vector<double>(6, 1.0)}, tensor<double> {{0, 3}, std::vector<double> {}}).data().empty());
  // A label summed by one operand alone, or twice in one operand (a diagonal), leaves the pair to the general engine.
  EXPECT(einsum("ij,j->", tensor<double> {{2, 2}, {1, 2, 3, 4}}, tensor<double> {{2}, {10, 100}}) ==
         tensor<double>(std::vector<std::size_t> {}, {640.0}));
  EXPECT(einsum("j,ij->", tensor<double> {{2}, {10, 100}}, tensor<double> {{2, 2}, {1, 2, 3, 4}}) ==
         tensor<double>(std::vector<std::size_t> {}, {640.0}));
  EXPECT(einsum("i,ii->i", tensor<double> {{2}, {10, 100}}, tensor<double> {{2, 2}, {1, 2, 3, 4}}) ==
         tensor<double>({2}, {10.0, 400.0}));
  EXPECT(einsum("j,kj->k", tensor<double> {{2}, {1, 2}}, tensor<double> {{3, 2}, {1, 2, 3, 4, 5, 6}}) ==
         tensor<double>({3}, {5.0, 11.0, 17.0}));
  EXPECT(einsum("ii,i->i", tensor<double> {{2, 2}, {1, 2, 3, 4}}, tensor<double> {{2}, {10, 100}}) ==
         tensor<double>({2}, {10.0, 400.0}));
}

TEST(einsum_is_generic_over_a_complex_scalar)
{
  using c = std::complex<double>;
  const matrix<c> a        {{c {1.0, 1.0}, c {0.0, 0.0}}, {c {0.0, 0.0}, c {1.0, -1.0}}};
  const matrix<c> b        {{c {2.0, 0.0}, c {0.0, 0.0}}, {c {0.0, 0.0}, c {2.0, 0.0}}};
  const tensor<c> ta       {a};
  const tensor<c> tb       {b};
  const tensor<c> expected {a * b};
  EXPECT(approx_equal(einsum("ij,jk->ik", ta, tb), expected));
}

TEST(einsum_rejects_malformed_specs)
{
  const matrix<int> sq {{1, 2}, {3, 4}};
  const tensor<int> ta {sq};
  const vector<int> v  {1, 2};
  const tensor<int> tv {v};

  EXPECT_THROWS(einsum("i1", ta), numerical::einsum_error);            // bad input char, implicit output
  EXPECT_THROWS(einsum("i1->i", ta), numerical::einsum_error);         // bad input char
  EXPECT_THROWS(einsum("ij,jk->ik", ta), numerical::einsum_error);     // operand count
  EXPECT_THROWS(einsum("ij->ij", tv), numerical::einsum_error);        // operand rank
  EXPECT_THROWS(einsum("ij->i1", ta), numerical::einsum_error);        // bad output char
  EXPECT_THROWS(einsum("ij->k", ta), numerical::einsum_error);         // output label not in inputs
  EXPECT_THROWS(einsum("ij->ii", ta), numerical::einsum_error);        // repeated output label
}

TEST(einsum_rejects_inconsistent_extents)
{
  const matrix<int> p  {{1, 2, 3}, {4, 5, 6}}; // 2×3 ⇒ i=2, j=3
  const matrix<int> q  {{1, 2}, {3, 4}};       // 2×2 ⇒ j=2 (conflicts)
  const tensor<int> tp {p};
  const tensor<int> tq {q};
  EXPECT_THROWS(einsum("ij,jk->ik", tp, tq), numerical::dimension_error);
}

// Five summed labels of 65 536 make 2⁸⁰ terms in one loop (whose count wrapped to 0 in a size_t, and the sum
// came back 0.0). Each operand is summed alone first, so the product is exact: 65 536⁵ = 2⁸⁰.
TEST(einsum_sums_each_operand_over_its_own_labels_first)
{
  const tensor<double> ones {std::vector<std::size_t> {std::size_t {1} << 16},
                             std::vector<double>(std::size_t {1} << 16, 1.0)};
  EXPECT(einsum("i,j,k,l,m->", ones, ones, ones, ones, ones).data().front() == std::ldexp(1.0, 80));
}

// Three operands or more are contracted two at a time: a chain of products, a shared label kept for a later
// operand, a diagonal and an output in another order give what the products written out give.
TEST(einsum_contracts_many_operands_pairwise)
{
  const tensor<int> a {matrix<int> {{1, 2}, {3, 4}}};
  const tensor<int> b {matrix<int> {{0, 1}, {1, 0}}};
  const tensor<int> c {matrix<int> {{2, 0}, {1, 3}}};
  EXPECT(einsum("ij,jk,kl->il", a, b, c).data() == einsum("ik,kl->il", einsum("ij,jk->ik", a, b), c).data());
  EXPECT(einsum("ij,jk,kl->li", a, b, c).data() == einsum("il->li", einsum("ij,jk,kl->il", a, b, c)).data());
  // i is in the first and the third operand: the intermediate keeps it.
  EXPECT(einsum("ij,jk,ik->i", a, b, c).data() == (std::vector<int> {4, 13})); // Σⱼₖ aᵢⱼ bⱼₖ cᵢₖ
  // A diagonal whose label the others carry too: aᵢᵢ Σⱼ aᵢⱼ vⱼ.
  EXPECT(einsum("ii,ij,j->i", a, a, tensor<int> {vector<int> {1, 2}}).data() == (std::vector<int> {5, 44}));
  EXPECT(einsum("ii,j,j->", a, tensor<int> {vector<int> {1, 2}}, tensor<int> {vector<int> {3, 4}}).data()
         == std::vector<int> {(1 + 4) * ((1 * 3) + (2 * 4))});
}

// The contraction order is the cheapest: in `ij,jk,kl` with a narrow j, the last two first (0.8 million multiply-adds
// against 8.4); past four operands, the cheapest pair at each step. The result is the same in any order here (whole
// numbers), as many operands as there are.
TEST(einsum_contracts_in_the_cheapest_order)
{
  using numerical::detail::contraction_path;
  const std::map<char, std::size_t> narrow {{'i', 200}, {'j', 10}, {'k', 200}, {'l', 200}};
  const auto                        path   {contraction_path({"ij", "jk", "kl"}, "il", narrow)};
  EXPECT(path.size() == 2U && path[0].first == 1U && path[0].second == 2U && path[1].first == 0U && path[1].second == 1U);
  // Five operands, a wide middle: the narrow ends first, each step the cheapest pair.
  const std::map<char, std::size_t> wide   {{'a', 2}, {'b', 3}, {'c', 50}, {'d', 50}, {'e', 3}, {'f', 2}};
  const auto                        steps {[](const std::vector<numerical::detail::contraction_step>& order) {
                                             std::vector<std::pair<std::size_t, std::size_t>> pairs;
                                             pairs.reserve(order.size());
                                             for (const numerical::detail::contraction_step step : order) {
                                               pairs.emplace_back(step.first, step.second);
                                             }
                                             return pairs;
                                           }};
  using pairs = std::vector<std::pair<std::size_t, std::size_t>>;
  EXPECT(steps(contraction_path({"ab", "bc", "cd", "de", "ef"}, "af", wide)) == (pairs {{0, 1}, {1, 2}, {0, 1}, {0, 1}}));
  // Equal costs: the smaller intermediate decides, over whole paths and pair by pair.
  const std::map<char, std::size_t> even  {{'a', 1}, {'b', 2}, {'c', 3}, {'d', 1}, {'e', 1}};
  EXPECT(steps(contraction_path({"ab", "bc", "cd", "de"}, "ae", even)) == (pairs {{1, 2}, {0, 2}, {0, 1}}));
  const std::map<char, std::size_t> level {{'a', 1}, {'b', 1}, {'c', 1}, {'d', 3}, {'e', 2}, {'f', 1}};
  EXPECT(steps(contraction_path({"ab", "bc", "cd", "de", "ef"}, "af", level)) == (pairs {{0, 1}, {0, 3}, {0, 2}, {0, 1}}));
  // No pair sharing a label: outer products, cheapest first.
  const std::map<char, std::size_t> apart {{'a', 4}, {'b', 1}, {'c', 2}, {'d', 3}, {'e', 5}};
  EXPECT(steps(contraction_path({"a", "b", "c", "d", "e"}, "abcde", apart)) == (pairs {{1, 2}, {1, 3}, {0, 1}, {0, 1}}));
  std::vector<tensor<int>> chain;
  chain.reserve(5);
  for (int p {0}; p < 5; ++p) {
    chain.emplace_back(matrix<int> {{1 + p, 2}, {0, 1 - p}});
  }
  const tensor<int> pairwise {einsum("ab,bc->ac", einsum("ab,bc->ac", einsum("ab,bc->ac", einsum("ab,bc->ac", chain[0], chain[1]), chain[2]), chain[3]), chain[4])};
  EXPECT(einsum("ab,bc,cd,de,ef->af", chain[0], chain[1], chain[2], chain[3], chain[4]) == pairwise);
}

// The operands may be counted at run time, as an interpreter holds them: the same contraction, and a
// count that disagrees with the spec, none included, is an einsum_error.
TEST(einsum_takes_operands_counted_at_run_time)
{
  const tensor<double>                     a    {matrix<double> {{1.0, 2.0}, {3.0, 4.0}}};
  const tensor<double>                     b    {matrix<double> {{0.0, 1.0}, {1.0, 0.0}}};
  const std::vector<const tensor<double>*> both {&a, &b};
  EXPECT(numerical::einsum("ij,jk->ik", both).data() == einsum("ij,jk->ik", a, b).data());
  EXPECT_THROWS(numerical::einsum("ij,jk->ik", std::vector<const tensor<double>*> {&a}), numerical::einsum_error);
  EXPECT_THROWS(numerical::einsum("->", std::vector<const tensor<double>*> {}), numerical::einsum_error);
}

namespace {

  //! \brief A tensor of \p shape holding 0, 1, 2, … times \p scale plus \p shift.
  tensor<double> counting(const std::vector<std::size_t>& shape,
                          double                          scale = 1.0,
                          double                          shift = 0.0)
  {
    std::size_t size {1};
    for (const std::size_t extent : shape) {
      size *= extent;
    }
    std::vector<double> values(size);
    for (std::size_t i {0}; i < size; ++i) {
      values[i] = (static_cast<double>(i) * scale) + shift;
    }
    return tensor<double> {shape, values};
  }

  //! \brief `a @ b` by loops, for two operands of rank 3 whose leading axes are equal or 1.
  std::vector<double> stacked_product(const tensor<double>& a,
                                      const tensor<double>& b,
                                      std::size_t           count)
  {
    const std::size_t   m {a.shape()[1]};
    const std::size_t   k {a.shape()[2]};
    const std::size_t   n {b.shape()[2]};
    std::vector<double> out(count * m * n, 0.0);
    for (std::size_t p {0}; p < count; ++p) {
      const std::size_t pa {a.shape()[0] == 1 ? 0 : p};
      const std::size_t pb {b.shape()[0] == 1 ? 0 : p};
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < n; ++j) {
          for (std::size_t q {0}; q < k; ++q) {
            out[(((p * m) + i) * n) + j] += a.data()[(((pa * m) + i) * k) + q] * b.data()[(((pb * k) + q) * n) + j];
          }
        }
      }
    }
    return out;
  }
} // namespace

TEST(matmul_of_stacks_broadcasts_their_leading_axes)
{
  const tensor<double> a           {counting({3, 2, 4}, 0.5, -1.0)};
  const tensor<double> b           {counting({1, 4, 5}, 0.25, 0.5)};
  const tensor<double> product     {numerical::matmul(a, b)};
  EXPECT(product.shape() == (std::vector<std::size_t> {3, 2, 5}));
  const std::vector<double> wanted {stacked_product(a, b, 3)};
  EXPECT(product.data() == numerical::buffer<double>(wanted.begin(), wanted.end()));
  const tensor<double> c           {numerical::matmul(counting({4, 1, 3, 2}), counting({5, 2, 6}))}; // leading axes 4 × 1 and 5
  EXPECT(c.shape() == (std::vector<std::size_t> {4, 5, 3, 6}));
  // a vector first is a row, a vector second a column, two vectors their dot product
  EXPECT(numerical::matmul(counting({4}), counting({2, 4, 3})).shape() == (std::vector<std::size_t> {2, 3}));
  EXPECT(numerical::matmul(counting({2, 3, 4}), counting({4})).shape() == (std::vector<std::size_t> {2, 3}));
  const tensor<double> dot {numerical::matmul(counting({3}), counting({3}))};
  EXPECT(dot.shape().empty() && dot.data()[0] == 5.0);
  EXPECT(numerical::matmul(counting({0, 2, 3}), counting({3, 4})).shape() == (std::vector<std::size_t> {0, 2, 4}));
  EXPECT(numerical::matmul(counting({2, 3, 0}), counting({0, 4})).data() == numerical::buffer<double>(24, 0.0));
  EXPECT(numerical::matmul(counting({2, 3, 4}), counting({2, 4, 5})).shape() == (std::vector<std::size_t> {2, 3, 5})); // equal axes
  EXPECT_THROWS((void)numerical::matmul(tensor<double> {{}, std::vector<double> {1.0}}, counting({2, 2})), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::matmul(counting({2, 2}), tensor<double> {{}, std::vector<double> {1.0}}), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::matmul(counting({2, 3}), counting({2, 3})), numerical::dimension_error);
  EXPECT_THROWS((void)numerical::matmul(counting({2, 2, 3}), counting({3, 3, 4})), numerical::dimension_error);
}

namespace
{
  // A stack's product against each of its matrices' products alone (one matmul of matrices), to the bit: (stack, m, k)
  // by (stack or 1, k, n), A of `a_stack` matrices (1: repeated), B of `b_stack` (1: shared).
  bool stack_gives_each_products_bits(std::size_t a_stack,
                                      std::size_t b_stack,
                                      std::size_t m,
                                      std::size_t k,
                                      std::size_t n)
  {
    const auto values {[](std::size_t count, double phase) {
                         std::vector<double> v(count);
                         for (std::size_t i {0}; i < count; ++i) {
                           v[i] = std::sin((static_cast<double>(i) * 0.37) + phase);
                         }
                         return v;
                       }};
    const std::size_t    stack     {std::max(a_stack, b_stack)};
    const tensor<double> a         {{a_stack, m, k}, values(a_stack * m * k, 0.1)};
    const tensor<double> b         {{b_stack, k, n}, values(b_stack * k * n, 0.7)};
    const tensor<double> by_matmul {numerical::matmul(a, b)};
    const tensor<double> by_einsum {einsum("...ij,...jk->...ik", a, b)};
    bool                 same      {by_matmul.data().size() == stack * m * n && by_einsum.data() == by_matmul.data()};
    for (std::size_t p {0}; p < stack; ++p) {
      const std::size_t    pa    {a_stack == 1 ? 0 : p};
      const std::size_t    pb    {b_stack == 1 ? 0 : p};
      const tensor<double> one_a {{m, k}, std::vector<double>(a.data().begin() + static_cast<std::ptrdiff_t>(pa * m * k),
                                                              a.data().begin() + static_cast<std::ptrdiff_t>((pa + 1) * m * k))};
      const tensor<double> one_b {{k, n}, std::vector<double>(b.data().begin() + static_cast<std::ptrdiff_t>(pb * k * n),
                                                              b.data().begin() + static_cast<std::ptrdiff_t>((pb + 1) * k * n))};
      const tensor<double> alone {numerical::matmul(one_a, one_b)};
      same = same && std::equal(alone.data().begin(), alone.data().end(), by_matmul.data().begin() + static_cast<std::ptrdiff_t>(p * m * n));
    }
    return same;
  }
} // namespace

// A stack of products gives each matrix's product alone: B shared by the whole stack (one product of all A's rows),
// A repeated, both stacked; fewer products than threads (one after another, each on all of them) and more.
TEST(a_stack_of_products_gives_each_products_bits)
{
  numerical::set_max_threads(4);
  for (const auto& [a_stack, b_stack, m, k, n] : {std::array<std::size_t, 5> {9, 1, 3, 64, 64}, {2, 1, 40, 300, 50},
                                                  {1, 9, 7, 30, 20}, {2, 2, 130, 101, 120}, {9, 9, 5, 40, 33}, {3, 1, 1, 20, 30}}) {
    EXPECT(stack_gives_each_products_bits(a_stack, b_stack, m, k, n));
  }
  numerical::set_max_threads(0); // every thread again
}

// A sum of many elements on the threads gives the bits of pairwise_sum on one: the whole tensor's and along an axis whose
// rows are single elements, past the four subtrees summed on the caller, an odd count among them; float and complex too.
TEST(a_parallel_sum_gives_the_pairwise_sums_bits)
{
  const auto check {[](auto zero, std::size_t count) {
                      using T = decltype(zero);
                      std::vector<T> values(count);
                      for (std::size_t i {0}; i < count; ++i) {
                        const double x {std::sin(static_cast<double>(i) * 0.61) * 1e3};
                        if constexpr (std::is_arithmetic_v<T>) {
                          values[i] = static_cast<T>(x);
                        }
                        else {
                          values[i] = T(static_cast<T::value_type>(x), static_cast<T::value_type>(std::cos(x)));
                        }
                      }
                      const T              alone {numerical::detail::pairwise_sum(values.data(), count, 1)};
                      const tensor<T>      whole   {{count}, values};
                      const tensor<T>      column  {{count, 1}, values};
                      const tensor<T>      by_axis {numerical::sum(column, 0)};
                      return numerical::sum(whole) == alone && by_axis.data()[0] == alone;
                    }};
  for (const std::size_t count : {(4 * numerical::detail::pairwise_leaf) + 1, std::size_t {3000017}, std::size_t {1} << 20U}) {
    EXPECT(check(0.0, count));
    EXPECT(check(0.0F, count));
    EXPECT(check(std::complex<double> {}, count));
  }
}

namespace
{
  // einsum of a product that sums nothing against each element's own product of depth one through gemm (0 + x·y as
  // gemm rounds a first term): signed zeros, products that underflow to a zero of either sign, ordinary values.
  template <typename T>
  bool each_product_is_gemms(const std::string&       spec,
                             std::vector<std::size_t> sa,
                             std::vector<std::size_t> sb)
  {
    const auto values {[](const std::vector<std::size_t>& s, std::size_t phase) {
                         std::size_t count {1};
                         for (const std::size_t e : s) {
                           count *= e;
                         }
                         std::vector<T> v(count);
                         for (std::size_t i {0}; i < count; ++i) {
                           const std::size_t k {(i * 7) + phase};
                           if constexpr (std::is_floating_point_v<T>) {
                             const T tiny {std::numeric_limits<T>::min() * T(0.75)};
                             const T table[] {T(-0.0), T(0.0), tiny, -tiny, T(1.5), T(-2.25), T(1) / T(3), std::numeric_limits<T>::min()};
                             v[i] = table[k % 8];
                           }
                           else {
                             v[i] = static_cast<T>(static_cast<int>(k % 11) - 5);
                           }
                         }
                         return tensor<T> {s, v};
                       }};
    const tensor<T> a                  {values(sa, 1)};
    const tensor<T> b                  {values(sb, 4)};
    const tensor<T> got                {numerical::einsum(spec, a, b)};
    // the reference: the output's labels give each element's operands
    const std::string           inputs {spec.substr(0, spec.find("->"))};
    const std::string           la     {inputs.substr(0, inputs.find(','))};
    const std::string           lb     {inputs.substr(inputs.find(',') + 1)};
    const std::string           out    {spec.substr(spec.find("->") + 2)};
    std::map<char, std::size_t> size;
    for (std::size_t i {0}; i < la.size(); ++i) {
      size[la[i]] = sa[i];
    }
    for (std::size_t i {0}; i < lb.size(); ++i) {
      size[lb[i]] = sb[i];
    }
    const auto offset {[&](const std::string& labels, const std::map<char, std::size_t>& at) {
                         std::size_t o {0};
                         for (const char l : labels) {
                           o = (o * size.at(l)) + at.at(l);
                         }
                         return o;
                       }};
    bool same {true};
    for (std::size_t flat {0}; flat < got.size(); ++flat) {
      std::map<char, std::size_t> at;
      std::size_t                 rest {flat};
      for (std::size_t i {out.size()}; i-- > 0;) {
        at[out[i]]  = rest % size.at(out[i]);
        rest       /= size.at(out[i]);
      }
      const T x    {a.data()[offset(la, at)]};
      const T y    {b.data()[offset(lb, at)]};
      T       want {};
      numerical::detail::gemm(1, 1, 1, &x, 1, &y, 1, &want, 1, numerical::detail::c_start::zero);
      const T have {got.data()[flat]};
      if constexpr (std::is_floating_point_v<T>) { // the same value and the same sign, a zero's included (no NaN here)
        same = same && have == want && std::signbit(have) == std::signbit(want);
      }
      else {
        same = same && have == want;
      }
    }
    return same;
  }
} // namespace

namespace {
  // fused_from_zero against std::fma(p, q, +0), value and sign, for every pair among zeros of either sign, products
  // that underflow to zero of either sign, subnormal, normal, huge and infinite values and NaN.
  template <typename T>
  bool fused_from_zero_is_fma()
  {
    const T                 inf    {std::numeric_limits<T>::infinity()};
    const T                 tiny   {std::numeric_limits<T>::denorm_min()};
    const T                 least  {std::numeric_limits<T>::min()};
    const std::array<T, 14> values {T {0}, -T {0}, tiny, -tiny, least, -least, T {0.75}, -T {1.5}, std::sqrt(least), -std::sqrt(least),
                                    std::numeric_limits<T>::max(), -inf, inf, std::numeric_limits<T>::quiet_NaN()};
    bool ok                        {true};
    for (const T p : values) {
      for (const T q : values) {
        const T got  {numerical::detail::fused_from_zero(p, q)};
        const T want {std::fma(p, q, T {})};
        ok = ok && ((std::isnan(got) && std::isnan(want)) || (got == want && std::signbit(got) == std::signbit(want)));
      }
    }
    return ok;
  }
} // namespace

TEST(a_product_from_zero_is_the_fused_one)
{
  EXPECT(fused_from_zero_is_fma<double>());
  EXPECT(fused_from_zero_is_fma<float>());
}

// An einsum product that sums nothing (each element one product) gives, on every path, each element's product of
// depth one through gemm to the bit, its zero's sign included: as matrix products of depth one gave it before.
TEST(an_einsum_that_sums_nothing_gives_gemms_first_terms)
{
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    if (numerical::limit_gemm_path(path) != path) {
      continue; // a path this processor does not run
    }
    for (const auto& [spec, sa, sb] : std::vector<std::tuple<std::string, std::vector<std::size_t>, std::vector<std::size_t>>> {
      {"ij,j->ij", {40, 30}, {30}}, {"i,j->ij", {17}, {23}}, {"ij,i->ij", {40, 30}, {40}}, {"ji,j->ij", {30, 40}, {30}},
      {"ijk,ik->ijk", {6, 5, 7}, {6, 7}}, {"j,ij->ij", {30}, {40, 30}}, {"ij,jk->ijk", {4, 5}, {5, 6}}, {"ij,ij->ij", {9, 7}, {9, 7}}}) {
      EXPECT(each_product_is_gemms<double>(spec, sa, sb));
      EXPECT(each_product_is_gemms<float>(spec, sa, sb));
      EXPECT(each_product_is_gemms<int>(spec, sa, sb));
    }
  }
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
}

TEST(einsum_reads_an_ellipsis_as_numpy)
{
  const tensor<double> a           {counting({2, 3, 4})};
  const tensor<double> b           {counting({1, 4, 5}, 0.5)};
  const tensor<double> batched     {einsum("...ij,...jk->...ik", a, b)}; // b's leading 1 repeated
  EXPECT(batched.shape() == (std::vector<std::size_t> {2, 3, 5}));
  const std::vector<double> wanted {stacked_product(a, b, 2)};
  EXPECT(batched.data() == numerical::buffer<double>(wanted.begin(), wanted.end()));
  EXPECT(einsum("...ij,jk", a, counting({4, 5})).shape() == (std::vector<std::size_t> {2, 3, 5}));                   // implicit
  EXPECT(einsum("i...->...", counting({4, 2, 3})).shape() == (std::vector<std::size_t> {2, 3}));
  EXPECT(einsum("...", counting({1, 1})).shape() == (std::vector<std::size_t> {1, 1}));                              // every operand's 1 kept
  EXPECT(einsum("...ii->...i", counting({2, 3, 3})).data() == numerical::buffer<double>({0, 4, 8, 9, 13, 17}));
  EXPECT(einsum("i,...->...i", counting({3}), counting({2})).shape() == (std::vector<std::size_t> {2, 3}));
  EXPECT(einsum("...i,...i->...", counting({2, 3}), counting({2, 3})).data() == numerical::buffer<double>({5, 50})); // equal axes
  EXPECT(einsum("...ij->ij", counting({2, 3})).shape() == (std::vector<std::size_t> {2, 3}));                        // an ellipsis of no axis
  const tensor<int> whole {{2, 2}, std::vector<int> {1, 2, 3, 4}};                                                   // the same for every scalar type
  EXPECT(einsum("...i->...", whole).data() == numerical::buffer<int>({3, 7}));
  EXPECT_THROWS((void)einsum("..i", counting({2})), numerical::einsum_error);
  EXPECT_THROWS((void)einsum("...i...", counting({2, 3})), numerical::einsum_error);
  EXPECT_THROWS((void)einsum("...ij", counting({2})), numerical::einsum_error);                    // fewer axes than letters
  EXPECT_THROWS((void)einsum("...i->i", counting({2, 3})), numerical::einsum_error);               // NumPy keeps them
  EXPECT_THROWS((void)einsum("...,...", counting({2, 3})), numerical::einsum_error);               // one operand for two
  EXPECT_THROWS((void)einsum("...i,...i->...", counting({2, 3}), counting({4, 3})), numerical::dimension_error);
  EXPECT_THROWS((void)einsum("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz...->...",
                             counting(std::vector<std::size_t>(53, 1))), numerical::einsum_error); // no letter left
}
