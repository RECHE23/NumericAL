// The matrix product kernel (gemm.hpp): every element of C is summed in ascending order of the inner
// index, so the packed kernel equals the plain i-k-j loop to the bit, across several depth panels,
// several column blocks, partial register tiles, every scalar it is instantiated for, and any number of
// threads. Exact equality is the property; a tolerance would hide a reordered sum.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>
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

  // The definition the kernel must reproduce: C starts at zero and adds a(i,p)·b(p,j) for p ascending, each
  // product rounded before its sum, or fused with it where the product runs on a fused path (AVX2, AVX-512 or the
  // AMX: a double product large enough to be packed; a small one runs the plain loop on any chip).
  template <typename T>
  std::vector<T> reference(std::size_t           m,
                           std::size_t           n,
                           std::size_t           k,
                           const std::vector<T>& a,
                           const std::vector<T>& b)
  {
    const bool packed         {k >= numerical::detail::gemm_min_depth
                               && m * n >= ((numerical::detail::gemm_packed_threshold + k) - 1) / k};
    const bool     fused      {std::is_same_v<T, double> && packed && numerical::gemm_path_in_use() != numerical::gemm_path::portable};
    std::vector<T> c(m * n, T {});
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t p {0}; p < k; ++p) {
        const T aip {a[(i * k) + p]};
        for (std::size_t j {0}; j < n; ++j) {
          if constexpr (std::is_floating_point_v<T>) {
            c[(i * n) + j] = fused ? std::fma(aip, b[(p * n) + j], c[(i * n) + j]) : c[(i * n) + j] + (aip * b[(p * n) + j]);
          }
          else {
            c[(i * n) + j] += aip * b[(p * n) + j];
          }
        }
      }
    }
    return c;
  }

  // Each check once on every path this processor can run (limit_gemm_path answers which one it took), then the
  // best path again.
  template <typename Check>
  bool on_every_path(const Check& check)
  {
    bool ok {true};
    for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::avx2,
                                            numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
      if (numerical::limit_gemm_path(path) == path) {
        ok = ok && check();
      }
    }
    (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
    return ok;
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
  EXPECT(on_every_path([&] { return product_is_exact<double>(129, 101, k); }));
  EXPECT(on_every_path([&] { return product_is_exact<double>(31, 15, k); })); // one partial AMX tile
  EXPECT(product_is_exact<float>(67, 75, k));
  (void)numerical::limit_gemm_path(numerical::gemm_path::portable);           // complex: past it, through the real kernel
  EXPECT(product_is_exact<std::complex<double>>(37, 29, k));
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
  EXPECT(product_is_exact<int>(70, 50, k));
}

namespace {
  // A complex product past the portable path, which sums the real and imaginary parts of A apart, on small
  // multiples of an eighth: every product and partial sum is exact, so it must equal the ascending sum, on one
  // thread as on all of them.
  template <typename Z>
  bool split_product_is_exact(std::size_t m,
                              std::size_t n,
                              std::size_t k)
  {
    using R = Z::value_type;
    std::vector<Z> a(m * k);
    std::vector<Z> b(k * n);
    for (std::size_t i {0}; i < a.size(); ++i) {
      a[i] = Z {static_cast<R>(i % 7) - R {3}, static_cast<R>(i % 5) / R {4}};
    }
    for (std::size_t i {0}; i < b.size(); ++i) {
      b[i] = Z {static_cast<R>(i % 3) / R {2}, R {1} - static_cast<R>(i % 11)};
    }
    std::vector<Z> wanted(m * n, Z {R {1}, R {-2}});
    std::vector<Z> all             {wanted};
    std::vector<Z> alone           {wanted};
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t p {0}; p < k; ++p) {
        for (std::size_t j {0}; j < n; ++j) {
          wanted[(i * n) + j] += a[(i * k) + p] * b[(p * n) + j];
        }
      }
    }
    numerical::detail::gemm(m, n, k, a.data(), k, b.data(), n, all.data(), n);
    numerical::set_max_threads(1);
    numerical::detail::gemm(m, n, k, a.data(), k, b.data(), n, alone.data(), n);
    numerical::set_max_threads(0);
    return all == wanted && alone == wanted;
  }
} // namespace

// Complex products past the portable path: several column blocks and depth blocks of the real products, each
// partial.
TEST(complex_products_run_through_the_real_kernel)
{
  const std::size_t n {numerical::detail::split_columns + 37};
  const std::size_t k {numerical::detail::split_depth + 19};
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<double>>(45, n, k); }));
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<float>>(21, 70, k); }));
}

// The complex product's real operands stay with the calling thread for the next product, unless one outgrew the
// limit: every product exact whatever the operands held before, a large one in between.
TEST(complex_products_keep_their_real_operands_within_the_limit)
{
  using numerical::detail::split_storage;
  const std::size_t k {numerical::detail::split_depth};
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<double>>(40, 70, 50); }));
  EXPECT(!split_storage<double>().c.empty());
  const std::size_t rows {(numerical::detail::split_kept / (2 * k)) + 8}; // [Re A | −Im A] past the limit
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<double>>(rows, 30, k); }));
  EXPECT(split_storage<double>().a.empty());
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<double>>(40, 70, 50); }));
  EXPECT(!split_storage<double>().c.empty());
}

// More columns than one packed block of B holds: the column blocks are summed independently.
TEST(packed_product_spans_several_column_blocks)
{
  const std::size_t n {std::max(numerical::detail::gemm_nc<double>, std::size_t {16} *128) + 13};
  EXPECT(on_every_path([&] { return product_is_exact<double>(9, n, 40); }));
}

// A small or a shallow product takes the plain loop, in the same order.
TEST(small_or_shallow_product_takes_the_plain_loop_and_the_same_order)
{
  EXPECT(std::size_t {3} *5 * 9 < numerical::detail::gemm_packed_threshold);
  EXPECT(on_every_path([] { return product_is_exact<double>(3, 5, 9); }));
  EXPECT(std::size_t {5} < numerical::detail::gemm_min_depth);
  EXPECT(on_every_path([] { return product_is_exact<double>(70, 60, 5); }));
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

// The packing storage stays with the calling thread for the next product, unless it outgrew the limit.
TEST(the_calling_thread_keeps_its_packing_storage_within_the_limit)
{
  using storage = numerical::detail::packing_storage<double>;
  EXPECT(product_is_exact<double>(64, 64, 64));
  EXPECT(!storage::kept().empty());
  // A tall product packs more of A than the limit: correct, then nothing kept.
  const std::size_t rows {(storage::kept_limit / numerical::detail::gemm_kc) + 64};
  EXPECT(product_is_exact<double>(rows, 3, numerical::detail::gemm_kc));
  EXPECT(storage::kept().empty());
  EXPECT(product_is_exact<double>(64, 64, 64));
  EXPECT(!storage::kept().empty());
}

// A zero dimension leaves C as it is, whichever it is (the decompositions call it so when nothing is left).
TEST(an_empty_dimension_leaves_c_unchanged)
{
  const std::vector<double> a(64, 1.0);
  const std::vector<double> b(64, 1.0);
  std::vector<double>       c(64, 7.0);
  numerical::detail::gemm(0, 8, 8, a.data(), 8, b.data(), 8, c.data(), 8);
  numerical::detail::gemm(8, 0, 8, a.data(), 8, b.data(), 8, c.data(), 8);
  numerical::detail::gemm(8, 8, 0, a.data(), 8, b.data(), 8, c.data(), 8);
  numerical::detail::gemm(64, 64, 0, a.data(), 1, b.data(), 1, c.data(), 1);
  EXPECT(std::ranges::all_of(c, [](double value) { return value == 7.0; }));
}
