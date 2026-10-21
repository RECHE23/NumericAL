// The dense matrix: construction/access, equality, element-wise arithmetic,
// scalar / matrix–vector / matrix–matrix products, transpose, tolerance
// comparison, shape checks, and the LU-based linear solve (with pivoting,
// singular detection, and genericity over a complex scalar).
//
// Brace `{a, b}` literals are kept in named locals so their commas are not seen
// as macro arguments; where a literal must appear inside a macro it is wrapped
// in an extra pair of parentheses.
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::approx_equal;
  using numerical::matrix;
  using numerical::solve;
  using numerical::transpose;
  using numerical::vector;
} // namespace

TEST(construction_and_access)
{
  const matrix<int> m {{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ(m.rows(), 2U);
  EXPECT_EQ(m.cols(), 3U);
  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(1, 2), 6);

  const matrix<int> empty;
  EXPECT_EQ(empty.rows(), 0U);
  EXPECT_EQ(empty.cols(), 0U);

  const matrix<int> from_buffer {2, 2, std::vector<int> {1, 2, 3, 4}};
  EXPECT_EQ(from_buffer(1, 1), 4);
}

TEST(ill_formed_construction_throws)
{
  EXPECT_THROWS((matrix<int> {{1, 2}, {3}}), numerical::dimension_error);                      // ragged rows
  EXPECT_THROWS((matrix<int> {2, 2, std::vector<int> {1, 2, 3}}), numerical::dimension_error); // bad buffer size
}

TEST(dimensions_whose_product_overflows_are_rejected)
{
  // rows*cols overflowing size_t is rejected at construction, not silently wrapped to a
  // small (mismatching) total that would later index out of bounds.
  const std::size_t huge {std::size_t {1} << 40};
  EXPECT_THROWS((matrix<int> {huge, huge, std::vector<int> {1}}), numerical::dimension_error);
}

TEST(equality_is_shape_and_element_wise)
{
  const matrix<int> a          {{1, 2, 3}, {4, 5, 6}};
  const matrix<int> same       {{1, 2, 3}, {4, 5, 6}};
  const matrix<int> fewer_rows {{1, 2, 3}};
  const matrix<int> fewer_cols {{1, 2}, {4, 5}};
  const matrix<int> other_data {{1, 2, 3}, {4, 5, 9}};
  EXPECT(a == same);
  EXPECT(!(a == fewer_rows));  // rows differ
  EXPECT(!(a == fewer_cols));  // same rows, cols differ
  EXPECT(!(a == other_data));  // same shape, an element differs
}

TEST(elementwise_add_and_subtract)
{
  const matrix<int> a    {{1, 2}, {3, 4}};
  const matrix<int> b    {{10, 20}, {30, 40}};
  const matrix<int> sum  {{11, 22}, {33, 44}};
  const matrix<int> diff {{9, 18}, {27, 36}};
  EXPECT(a + b == sum);
  EXPECT(b - a == diff);
}

TEST(arithmetic_shape_mismatch_throws)
{
  const matrix<int> a          {{1, 2}, {3, 4}};
  const matrix<int> wrong_rows {{1, 2}};               // rows differ
  const matrix<int> wrong_cols {{1, 2, 3}, {4, 5, 6}}; // same rows, cols differ
  EXPECT_THROWS(a + wrong_rows, numerical::dimension_error);
  EXPECT_THROWS(a + wrong_cols, numerical::dimension_error);
  EXPECT_THROWS(a - wrong_rows, numerical::dimension_error);
  EXPECT_THROWS(a - wrong_cols, numerical::dimension_error);
}

TEST(scalar_multiplication_on_either_side)
{
  const matrix<int> m      {{1, 2}, {3, 4}};
  const matrix<int> scaled {{3, 6}, {9, 12}};
  EXPECT(3 * m == scaled);
  EXPECT(m * 3 == scaled);
}

TEST(matrix_vector_product)
{
  const matrix<int> a        {{1, 2, 3}, {4, 5, 6}};
  const vector<int> v        {1, 0, 1};
  const vector<int> expected {4, 10}; // [1+3, 4+6]
  EXPECT(a * v == expected);

  const vector<int> wrong {1, 2};
  EXPECT_THROWS(a * wrong, numerical::dimension_error);
  // No columns: each element an empty sum.
  EXPECT(matrix<int>(2, 0, std::vector<int> {}) * vector<int>(std::vector<int> {}) == (vector<int> {0, 0}));
  // Each element is its row's dot product, to the bit, over several groups of rows.
  const std::size_t   rows {131};
  const std::size_t   cols {37};
  std::vector<double> entries(rows * cols);
  for (std::size_t i {0}; i < entries.size(); ++i) {
    entries[i] = 1.0 / static_cast<double>(i + 3);
  }
  std::vector<double> x(cols);
  for (std::size_t j {0}; j < cols; ++j) {
    x[j] = static_cast<double>(j % 5) - 1.5;
  }
  const vector<double> y    {matrix<double>(rows, cols, entries) * vector<double>(x)};
  bool                 same {true};
  for (std::size_t r {0}; r < rows; ++r) {
    same = same && y[r] == numerical::detail::dot_product(entries.data() + (r * cols), x.data(), cols);
  }
  EXPECT(same);
}

TEST(matrix_matrix_product)
{
  const matrix<int> a        {{1, 2}, {3, 4}};
  const matrix<int> b        {{5, 6}, {7, 8}};
  const matrix<int> expected {{19, 22}, {43, 50}};
  EXPECT(a * b == expected);

  const matrix<int> wrong {{1, 2, 3}};
  EXPECT_THROWS(a * wrong, numerical::dimension_error);
}

// By blocks, partial ones at either edge, and by the threads' shares: each element where it belongs.
TEST(transpose_by_blocks_puts_each_element_where_it_belongs)
{
  for (const auto& [rows, cols] : {std::array<std::size_t, 2> {1, 1}, {33, 65}, {700, 1003}, {1, 513}}) {
    numerical::buffer<double> values(rows * cols);
    for (std::size_t k {0}; k < values.size(); ++k) {
      values[k] = static_cast<double>(k);
    }
    const numerical::matrix<double> m {rows, cols, std::move(values)};
    for (const std::size_t threads : {std::size_t {1}, std::size_t {0}}) {
      numerical::set_max_threads(threads);
      const numerical::matrix<double> t    {numerical::transpose(m)};
      bool                            each {t.rows() == cols && t.cols() == rows};
      for (std::size_t i {0}; each && i < rows; ++i) {
        for (std::size_t j {0}; j < cols; ++j) {
          each = each && t(j, i) == m(i, j);
        }
      }
      EXPECT(each);
    }
    numerical::set_max_threads(0);
  }
}

TEST(transpose_swaps_rows_and_columns)
{
  const matrix<int> m {{1, 2, 3}, {4, 5, 6}};
  const matrix<int> t {{1, 4}, {2, 5}, {3, 6}};
  EXPECT(transpose(m) == t);
}

TEST(approx_equal_compares_within_a_tolerance)
{
  const matrix<double> a          {{1.0, 2.0}, {3.0, 4.0}};
  const matrix<double> close      {{1.0 + 1e-12, 2.0}, {3.0, 4.0 - 1e-12}};
  const matrix<double> off        {{1.0, 2.0}, {3.0, 4.5}};
  const matrix<double> wrong_rows {{1.0, 2.0}};
  const matrix<double> wrong_cols {{1.0, 2.0, 3.0}, {3.0, 4.0, 5.0}};
  EXPECT(approx_equal(a, close));                                       // within default tolerance
  EXPECT(!approx_equal(a, off));                                        // an element exceeds it
  EXPECT(!approx_equal(a, wrong_rows));                                 // rows differ
  EXPECT(!approx_equal(a, wrong_cols));                                 // same rows, cols differ
  const double nan {std::numeric_limits<double>::quiet_NaN()};
  EXPECT(!approx_equal(a, matrix<double> {{1.0, 2.0}, {nan, 4.0}}));    // a NaN is close to nothing
  const double inf {std::numeric_limits<double>::infinity()};
  EXPECT(approx_equal(matrix<double> {{inf}}, matrix<double> {{inf}})); // equal infinities, a NaN difference
  EXPECT(!approx_equal(matrix<double> {{inf}}, matrix<double> {{-inf}}));
}

TEST(approx_equal_compares_complex_matrices)
{
  using c = std::complex<double>;
  const matrix<c> a {{c {1.0, 1.0}, c {2.0, 0.0}}};
  EXPECT(approx_equal(a, a));                                                       // equal
  EXPECT(approx_equal(a, matrix<c> {{c {1.0, 1.0 + 1e-12}, c {2.0, 0.0}}}));        // close
  EXPECT(!approx_equal(a, matrix<c> {{c {1.0, 1.5}, c {2.0, 0.0}}}));               // far
  EXPECT(!approx_equal(a, matrix<c> {{c {1.0, 1.0}}}));                             // columns differ
  EXPECT(!approx_equal(a, matrix<c> {{c {1.0, 1.0}, c {2.0, 0.0}}, {c {}, c {}}})); // rows differ
}

TEST(solve_a_well_conditioned_system)
{
  // Diagonally dominant ⇒ no row pivoting needed; x = [1, 2, 3].
  const matrix<double> a {{2.0, 1.0, 1.0}, {1.0, 3.0, 1.0}, {1.0, 1.0, 4.0}};
  const vector<double> b {7.0, 10.0, 15.0};
  const vector<double> x {1.0, 2.0, 3.0};
  EXPECT(approx_equal(solve(a, b), x));
}

TEST(solve_requires_a_row_swap)
{
  // A zero pivot in column 0 forces partial pivoting to swap rows; x = [5, 7].
  const matrix<double> a {{0.0, 1.0}, {1.0, 0.0}};
  const vector<double> b {7.0, 5.0};
  const vector<double> x {5.0, 7.0};
  EXPECT(approx_equal(solve(a, b), x));
}

TEST(solve_is_generic_over_a_complex_scalar)
{
  using c = std::complex<double>;
  const matrix<c> a {{c {1.0, 1.0}, c {0.0, 0.0}}, {c {0.0, 0.0}, c {2.0, 0.0}}};
  const vector<c> b {c {1.0, 1.0}, c {4.0, 0.0}};
  const vector<c> x {c {1.0, 0.0}, c {2.0, 0.0}};
  EXPECT(approx_equal(solve(a, b), x));
}

TEST(solve_rejects_singular_and_ill_shaped_systems)
{
  const matrix<double> singular {{1.0, 2.0}, {2.0, 4.0}}; // rank-deficient
  const vector<double> b2       {1.0, 2.0};
  EXPECT_THROWS(solve(singular, b2), numerical::singular_matrix_error);

  const matrix<double> non_square {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  EXPECT_THROWS(solve(non_square, b2), numerical::dimension_error);

  const matrix<double> square {{1.0, 0.0}, {0.0, 1.0}};
  const vector<double> b3     {1.0, 2.0, 3.0};
  EXPECT_THROWS(solve(square, b3), numerical::dimension_error);
}

TEST(matrix_matrix_product_large)
{
  // Large enough that the product runs its rows in parallel; checked against a
  // plain reference, confirming the threaded result is correct (deterministic).
  const std::size_t n {64};
  std::vector<int>  av(n * n);
  std::vector<int>  bv(n * n);
  for (std::size_t i {0}; i < n * n; ++i) {
    av[i] = static_cast<int>(i % 5) - 2;
    bv[i] = static_cast<int>(i % 3) - 1;
  }
  std::vector<int> ref(n * n, 0);
  for (std::size_t m {0}; m < n; ++m) {
    for (std::size_t k {0}; k < n; ++k) {
      for (std::size_t j {0}; j < n; ++j) {
        ref[(m * n) + j] += av[(m * n) + k] * bv[(k * n) + j];
      }
    }
  }
  const matrix<int> a        {n, n, av};
  const matrix<int> b        {n, n, bv};
  const matrix<int> expected {n, n, ref};
  EXPECT(a * b == expected);
}

TEST(matrix_matrix_product_double_matches_independent_reference)
{
  // operator* runs its rows in parallel and accumulates each in i-k-j order. This
  // checks that the threaded, reordered accumulation still equals an independent
  // definition-order (i-j-k) reference, within tolerance, on varied non-square
  // operands. The fill is a deterministic sin/cos of the index — reproducible, and
  // the same values feed both sides, so the test is platform-independent.
  const std::size_t   n {100}; // rows are past the parallel threshold
  const std::size_t   p {33};
  const std::size_t   m {40};  // (n×p)·(p×m)
  std::vector<double> av(n * p);
  std::vector<double> bv(p * m);
  for (std::size_t i {0}; i < av.size(); ++i) { av[i] = std::sin((static_cast<double>(i) * 1.3) + 0.4); }
  for (std::size_t i {0}; i < bv.size(); ++i) { bv[i] = std::cos((static_cast<double>(i) * 0.9) + 0.2); }

  std::vector<double> ref(n * m, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j < m; ++j) {
      double sum {0.0};
      for (std::size_t k {0}; k < p; ++k) { sum += av[(i * p) + k] * bv[(k * m) + j]; }
      ref[(i * m) + j] = sum;
    }
  }

  const matrix<double> a        {n, p, av};
  const matrix<double> b        {p, m, bv};
  const matrix<double> expected {n, m, ref};
  EXPECT(approx_equal(a * b, expected));
}
