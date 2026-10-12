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
#include <limits>
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

TEST(einsum_matrix_product)
{
  const matrix<int> a        {{1, 2}, {3, 4}};
  const matrix<int> b        {{5, 6}, {7, 8}};
  const tensor<int> ta       {a};
  const tensor<int> tb       {b};
  const tensor<int> expected {a * b};
  EXPECT(einsum("ij,jk->ik", ta, tb) == expected);
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

TEST(einsum_falls_back_for_non_canonical_contractions)
{
  // None of these match the canonical GEMM layout, so they take the general
  // engine; the results confirm that fall-back path stays correct.
  const tensor<int> a             {{2, 2}, {1, 2, 3, 4}};
  const tensor<int> b             {{2, 2}, {5, 6, 7, 8}};
  const tensor<int> a_transpose_b {{2, 2}, {26, 30, 38, 44}}; // "ji,jk->ik": op0 not ending in the contracted label
  const tensor<int> a_b_transpose {{2, 2}, {17, 23, 39, 53}}; // "ij,kj->ik": op1 not starting with it
  const tensor<int> swapped_out   {{2, 2}, {19, 43, 22, 50}}; // "ij,jk->ki": output axes swapped
  EXPECT(einsum("ji,jk->ik", a, b) == a_transpose_b);
  EXPECT(einsum("ij,kj->ik", a, b) == a_b_transpose);
  EXPECT(einsum("ij,jk->ki", a, b) == swapped_out);

  // A rank-0 operand cannot carry the contracted label, so neither is a GEMM.
  const tensor<int> v      {{3}, {1, 2, 3}};
  const tensor<int> scalar {std::vector<std::size_t> {}, std::vector<int> {10}};
  const tensor<int> left   {einsum("i,->", v, scalar)}; // op1 is rank-0
  const tensor<int> right  {einsum(",i->", scalar, v)}; // op0 is rank-0
  EXPECT_EQ(left.rank(), 0U);
  EXPECT_EQ(left.data()[0], 60);                        // (1 + 2 + 3) * 10
  EXPECT_EQ(right.data()[0], 60);
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

  EXPECT_THROWS(einsum("ij", ta), numerical::einsum_error);            // no '->'
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
