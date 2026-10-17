// The matrix product kernel (gemm.hpp): every element of C is summed in ascending order of the inner
// index, so the packed kernel equals the plain i-k-j loop to the bit, across several depth panels,
// several column blocks, partial register tiles, every scalar it is instantiated for, and any number of
// threads. Exact equality is the property; a tolerance would hide a reordered sum.
#include <algorithm>
#include <array>
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
  // product rounded before its sum, or fused with it where the product runs on a fused path (NEON, AVX2, AVX-512 or
  // the AMX: a double product large enough to be packed, or a float one on arm64's; a small one runs the plain loop
  // on any chip).
  template <typename T>
  std::vector<T> reference(std::size_t           m,
                           std::size_t           n,
                           std::size_t           k,
                           const std::vector<T>& a,
                           const std::vector<T>& b)
  {
    const bool packed         {k >= numerical::detail::gemm_min_depth
                               && m * n >= ((numerical::detail::gemm_packed_threshold + k) - 1) / k};
    const bool     fused      {(std::is_same_v<T, double> || (std::is_same_v<T, float> && numerical::detail::neon_compiled)) && packed
                               && numerical::gemm_path_in_use() != numerical::gemm_path::portable};
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
    for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
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
  EXPECT(on_every_path([&] { return product_is_exact<float>(67, 75, k); }));
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

// Complex products past the portable path, through the real kernel or the x86-64 complex tiles: several column
// blocks and depth blocks, each partial, and rows ending in a partial tile of either.
namespace {
  // Whether gemm_with on A packed once (pack_left, for products of about n_hint columns) gives gemm's bits for a
  // product of n columns, A and B with padded row distances, C added to.
  template <typename T>
  bool shared_left_equals_gemm(std::size_t m,
                               std::size_t n_hint,
                               std::size_t n,
                               std::size_t k)
  {
    const std::size_t lda {k + 3};
    const std::size_t ldb {n + 2};
    const std::size_t ldc {n + 5};
    std::vector<T>    a(m * lda);
    std::vector<T>    b(k * ldb);
    std::vector<T>    c(m * ldc);
    for (std::size_t i {0}; i < a.size(); ++i) {
      a[i] = fill<T>(i);
    }
    for (std::size_t i {0}; i < b.size(); ++i) {
      b[i] = fill<T>(i + 7);
    }
    for (std::size_t i {0}; i < c.size(); ++i) {
      c[i] = fill<T>(i + 3);
    }
    std::vector<T> want {c};
    numerical::detail::gemm(m, n, k, a.data(), lda, b.data(), ldb, want.data(), ldc);
    numerical::buffer<T>                    storage;
    const numerical::detail::packed_left<T> left {numerical::detail::pack_left(m, n_hint, k, a.data(), lda, storage)};
    numerical::detail::gemm_with(left, n, b.data(), ldb, c.data(), ldc);
    numerical::detail::gemm_with(left, n, b.data(), ldb, c.data(), ldc); // A's packing read twice, unchanged
    numerical::detail::gemm(m, n, k, a.data(), lda, b.data(), ldb, want.data(), ldc);
    return c == want;
  }
} // namespace

TEST(a_left_operand_packed_once_gives_gemm_s_bits_on_every_path)
{
  // Packed for the tile in use (the depth of an LU panel, 32 or 64, and a deep one); a narrow product whose
  // tile differs from the hint's (on the AMX path, too few columns to suit it) or that the plain loop takes (7 × 1
  // and 20 × 2 by 64: a packing read there would round otherwise); a
  // depth past one panel, and other scalars, read A in place.
  for (const auto& [m, hint, n, k] : {std::array<std::size_t, 4> {936, 64, 64, 64}, {936, 64, 200, 64}, {300, 32, 97, 32},
                                      {130, 64, 3, 64}, {7, 64, 1, 64}, {20, 64, 2, 64}, {7, 64, 5, 4}, {260, 64, 64, 300}, {64, 64, 64, 256}}) {
    EXPECT(shared_left_equals_gemm<double>(m, hint, n, k));
  }
  EXPECT(shared_left_equals_gemm<float>(200, 64, 64, 64));
  EXPECT(shared_left_equals_gemm<std::complex<double>>(100, 32, 40, 32));
  numerical::buffer<double> storage;
  const std::vector<double> a(64, 1.0);
  EXPECT(numerical::detail::pack_left(8, 8, 8, a.data(), 8, storage).kernel == numerical::detail::left_kernel::plain);
  for (const numerical::gemm_path path : {numerical::gemm_path::portable, numerical::gemm_path::neon, numerical::gemm_path::avx2,
                                          numerical::gemm_path::avx512, numerical::gemm_path::amx}) {
    (void)numerical::limit_gemm_path(path); // every path this machine runs, the portable vector tile among them
    EXPECT(shared_left_equals_gemm<double>(300, 64, 128, 64));
  }
}

#ifdef NUMERICAL_X86_COMPILED
// The AVX2 tile's task rows follow the panel's depth and it prefetches the next row's C: the hooks are found (a
// signature they no longer match would fall back in silence), and other tiles keep their block and prefetch nothing.
static_assert(numerical::detail::task_rows_cap<numerical::detail::avx2_tile_kernel>(64) == 36);
static_assert(numerical::detail::task_rows_cap<numerical::detail::avx2_tile_kernel>(128) == 36);
static_assert(numerical::detail::task_rows_cap<numerical::detail::avx2_tile_kernel>(256) == numerical::detail::avx2_tile_kernel::block_rows);
static_assert(numerical::detail::task_rows_cap<numerical::detail::avx512_tile_kernel>(64) == numerical::detail::avx512_tile_kernel::block_rows);
static_assert(numerical::detail::prefetches_next_row_of_c<numerical::detail::avx2_tile_kernel>);
static_assert(!numerical::detail::prefetches_next_row_of_c<numerical::detail::avx512_tile_kernel>);
#endif

TEST(complex_products_past_the_portable_path)
{
  const std::size_t n {numerical::detail::split_columns + 37};
  const std::size_t k {numerical::detail::split_depth + 19};
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<double>>(45, n, k); }));
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<double>>(46, 70, 135); }));
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<float>>(21, 70, k); }));
}

// The complex product's real operands stay with the calling thread for the next product, unless one outgrew the
// limit: every product exact whatever the operands held before, a large one in between.
TEST(complex_products_keep_their_real_operands_within_the_limit)
{
  using numerical::detail::split_storage;
  // Only past the portable path does a complex product go through the real kernel (an arm64 Linux processor has none);
  // complex float on every such path (complex double has a tile of its own on the x86-64 units).
  const bool        split {numerical::limit_gemm_path(numerical::gemm_path::amx) != numerical::gemm_path::portable};
  const std::size_t k     {numerical::detail::split_depth};
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<float>>(40, 70, 50); }));
  EXPECT(!split || !split_storage<float>().c.empty());
  const std::size_t rows {(numerical::detail::split_kept / (2 * k)) + 8}; // [Re A | −Im A] past the limit
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<float>>(rows, 30, k); }));
  EXPECT(split_storage<float>().a.empty());
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<float>>(40, 70, 50); }));
  EXPECT(!split || !split_storage<float>().c.empty());
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
