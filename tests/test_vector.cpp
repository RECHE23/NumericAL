// The dense vector: construction/access, equality, element-wise arithmetic,
// scalar multiplication, dot product, tolerance comparison, and shape checks.
// Exercised over int (exact), double (tolerance), and complex (genericity).
//
// EXPECT(...) conditions containing braced `{a, b}` literals are wrapped in an
// extra pair of parentheses, so their commas are not seen as macro arguments.
#include <complex>
#include <limits>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::approx_equal;
  using numerical::dot;
  using numerical::vector;
} // namespace

TEST(construction_and_access)
{
  const vector<int> v {10, 20, 30};
  EXPECT_EQ(v.size(), 3U);
  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[2], 30);
  const vector<int> empty;
  EXPECT_EQ(empty.size(), 0U);
  const vector<int> from_buffer {std::vector<int> {1, 2}};
  EXPECT_EQ(from_buffer.size(), 2U);
  EXPECT_EQ(from_buffer[1], 2);
}

TEST(equality_is_size_and_element_wise)
{
  EXPECT((vector<int> {1, 2, 3} == vector<int> {1, 2, 3}));
  EXPECT((!(vector<int> {1, 2, 3} == vector<int> {1, 2, 4}))); // an element differs
  EXPECT((!(vector<int> {1, 2} == vector<int> {1, 2, 3})));    // a size differs
}

TEST(elementwise_add_and_subtract)
{
  EXPECT((vector<int> {1, 2, 3} + vector<int> {10, 20, 30} == vector<int> {11, 22, 33}));
  EXPECT((vector<int> {10, 20, 30} - vector<int> {1, 2, 3} == vector<int> {9, 18, 27}));
}

TEST(scalar_multiplication_on_either_side)
{
  EXPECT((3 * vector<int> {1, 2, 3} == vector<int> {3, 6, 9}));
  EXPECT((vector<int> {1, 2, 3} *3 == vector<int> {3, 6, 9}));
}

TEST(dot_product)
{
  EXPECT_EQ(dot(vector<int> {1, 2, 3}, vector<int> {4, 5, 6}), 32); // 4 + 10 + 18
  EXPECT_EQ(dot(vector<int> {}, vector<int> {}), 0);                // empty ⇒ zero
}

TEST(generic_over_the_scalar_complex)
{
  using c = std::complex<double>;
  const vector<c> v {c {0.0, 1.0}, c {1.0, 0.0}}; // [i, 1]
  EXPECT((dot(v, v) == c {0.0, 0.0}));            // i² + 1 = -1 + 1 = 0
  EXPECT((approx_equal(v + v, vector<c> {c {0.0, 2.0}, c {2.0, 0.0}})));
}

TEST(approx_equal_compares_within_a_tolerance)
{
  const vector<double> a {1.0, 2.0};
  EXPECT((approx_equal(a, vector<double> {1.0 + 1e-12, 2.0 - 1e-12})));         // within default tol
  EXPECT((!approx_equal(a, vector<double> {1.0, 2.5})));                        // outside tol
  EXPECT((!approx_equal(a, vector<double> {1.0})));                             // size mismatch
  const double nan {std::numeric_limits<double>::quiet_NaN()};
  const double inf {std::numeric_limits<double>::infinity()};
  EXPECT((!approx_equal(a, vector<double> {nan, 2.0})));                        // a NaN is close to nothing
  EXPECT((!approx_equal(vector<double> {nan}, vector<double> {nan})));          // not even a NaN
  EXPECT((approx_equal(vector<double> {inf, 1.0}, vector<double> {inf, 1.0}))); // equal infinities
  EXPECT((!approx_equal(vector<double> {inf}, vector<double> {-inf})));
}

TEST(approx_equal_compares_complex_vectors)
{
  using c = std::complex<double>;
  const vector<c> a {c {1.0, 1.0}, c {2.0, 0.0}};
  EXPECT((approx_equal(a, a)));                                                 // equal
  EXPECT((approx_equal(a, vector<c> {c {1.0, 1.0 + 1e-12}, c {2.0, 0.0}})));    // close
  EXPECT((!approx_equal(a, vector<c> {c {1.0, 1.5}, c {2.0, 0.0}})));           // far
  EXPECT((!approx_equal(a, vector<c> {c {1.0, 1.0}})));                         // size mismatch
}

TEST(shape_mismatch_throws)
{
  EXPECT_THROWS((vector<int> {1, 2} + vector<int> {1, 2, 3}), numerical::dimension_error);
  EXPECT_THROWS((vector<int> {1, 2} - vector<int> {1, 2, 3}), numerical::dimension_error);
  EXPECT_THROWS((dot(vector<int> {1, 2}, vector<int> {1, 2, 3})), numerical::dimension_error);
}
