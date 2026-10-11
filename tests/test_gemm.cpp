// The matrix product kernel (gemm.hpp): every element of C is summed in ascending order of the inner
// index, so the packed kernel equals the plain i-k-j loop to the bit, across several depth panels,
// several column blocks, partial register tiles, every scalar it is instantiated for, and any number of
// threads. Exact equality is the property; a tolerance would hide a reordered sum.
#include <complex>
#include <cstddef>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::matrix;
  using numerical::tensor;

  // Deterministic, sign-varying, not exactly representable sums: a reordering shows in the last bits.
  template <typename T>
  T fill(std::size_t i)
  {
    return T((1.0 / static_cast<double>((i % 13) + 1)) - 0.27);
  }

  template <>
  int fill<int>(std::size_t i)
  {
    return static_cast<int>(i % 11) - 5;
  }

  // The definition the kernel must reproduce: C starts at zero and adds a(i,p)·b(p,j) for p ascending.
  template <typename T>
  std::vector<T> reference(std::size_t           m,
                           std::size_t           n,
                           std::size_t           k,
                           const std::vector<T>& a,
                           const std::vector<T>& b)
  {
    std::vector<T> c(m * n, T {});
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t p {0}; p < k; ++p) {
        const T aip {a[(i * k) + p]};
        for (std::size_t j {0}; j < n; ++j) {
          c[(i * n) + j] += aip * b[(p * n) + j];
        }
      }
    }
    return c;
  }

  template <typename T>
  bool product_is_exact(std::size_t m,
                        std::size_t n,
                        std::size_t k)
  {
    std::vector<T> a(m * k);
    std::vector<T> b(k * n);
    for (std::size_t i {0}; i < a.size(); ++i) {
      a[i] = fill<T>(i);
    }
    for (std::size_t i {0}; i < b.size(); ++i) {
      b[i] = fill<T>((i * 7) + 3);
    }
    const matrix<T> expected  {m, n, reference(m, n, k, a, b)};
    const bool      by_matrix {matrix<T> {m, k, a} *matrix<T> {k, n, b} == expected};
    const bool      by_einsum {numerical::einsum("ij,jk->ik", tensor<T> {{m, k}, a}, tensor<T> {{k, n}, b}) ==
                               tensor<T> {{m, n}, std::vector<T>(expected.begin(), expected.end())}};
    return by_matrix && by_einsum;
  }
} // namespace

// Past the packing threshold: three depth panels (the last partial), a last partial register tile in
// both directions, and enough tasks to run in parallel.
TEST(packed_product_equals_the_ascending_sum_to_the_bit)
{
  const std::size_t k {(numerical::detail::gemm_kc * 2) + 17};
  EXPECT(product_is_exact<double>(129, 101, k));
  EXPECT(product_is_exact<float>(67, 75, k));
  EXPECT(product_is_exact<std::complex<double>>(37, 29, k));
  EXPECT(product_is_exact<int>(70, 50, k));
}

// More columns than one packed block of B holds: the column blocks are summed independently.
TEST(packed_product_spans_several_column_blocks)
{
  const std::size_t n {numerical::detail::gemm_nc<double> + 13};
  EXPECT(product_is_exact<double>(9, n, 40));
}

// A small or a shallow product takes the plain loop, in the same order.
TEST(small_or_shallow_product_takes_the_plain_loop_and_the_same_order)
{
  EXPECT(std::size_t {3} *5 * 9 < numerical::detail::gemm_packed_threshold);
  EXPECT(product_is_exact<double>(3, 5, 9));
  EXPECT(std::size_t {5} < numerical::detail::gemm_min_depth);
  EXPECT(product_is_exact<double>(70, 60, 5));
}

// An empty inner dimension leaves C at zero; an empty outer one leaves it empty.
TEST(product_over_an_empty_dimension)
{
  const matrix<double> wide {2, 0, std::vector<double> {}};
  const matrix<double> tall {0, 3, std::vector<double> {}};
  EXPECT(wide * tall == (matrix<double> {2, 3, std::vector<double>(6, 0.0)}));
  const matrix<double> none {0, 4, std::vector<double> {}};
  const matrix<double> four {4, 2, std::vector<double>(8, 1.0)};
  EXPECT((none * four).rows() == 0U);
  EXPECT_THROWS(wide * four, numerical::dimension_error); // the shape check precedes the empty cases
  EXPECT(numerical::einsum("ij,jk->ik", tensor<double> {{2, 0}, {}}, tensor<double> {{0, 3}, {}}) ==
         (tensor<double> {{2, 3}, std::vector<double>(6, 0.0)}));
}
