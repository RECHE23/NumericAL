// The matrix product kernel (gemm.hpp): every element of C is summed in ascending order of the inner
// index, so the packed kernel equals the plain i-k-j loop to the bit, across several depth panels,
// several column blocks, partial register tiles, every scalar it is instantiated for, and any number of
// threads. Exact equality is the property; a tolerance would hide a reordered sum.
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
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

  // fill's value, its imaginary part filled too for a complex T (a zero one would hide how its products round).
  template <typename T>
  T entry(std::size_t i)
  {
    if constexpr (std::is_floating_point_v<T>) {
      return fill<T>(i);
    }
    else {
      using R = T::value_type;
      return T {fill<R>(i), fill<R>((i * 3) + 1)};
    }
  }

  // The definition the kernel must reproduce: C starts at zero and adds a(i,p)·b(p,j) for p ascending, each
  // product rounded before its sum, or fused with it where the product runs on a fused path (NEON, AVX2, AVX-512 or
  // the AMX, double and float alike), small or packed alike: the plain loop rounds each term as the tile of the path
  // does, so an element is the same whatever the shape of the product.
  template <typename T>
  std::vector<T> reference(std::size_t           m,
                           std::size_t           n,
                           std::size_t           k,
                           const std::vector<T>& a,
                           const std::vector<T>& b)
  {
    const bool     fused      {(std::is_same_v<T, double> || std::is_same_v<T, float>)
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

// A product of up to gemm_rows_columns columns runs by rows (gemm_rows), where the packed kernel's tile left most of
// its width idle: the packed product's bits, on every path, past one depth panel, any number of rows and threads, every
// width of a last block of columns after whole blocks of rows and after a last row alone.
TEST(narrow_products_by_rows_give_the_packed_bits)
{
  EXPECT(on_every_path([] {
                         bool same {true};
                         for (const std::size_t threads : {std::size_t {1}, std::size_t {4}}) {
                           numerical::set_max_threads(threads);
                           for (const auto& [m, n, k] : {std::array<std::size_t, 3> {169, 2, 579}, {37, 7, 37}, {300, 8, 300}, {130, 13, 600},
                                                         {41, 3, 90}, {23, 14, 70}, {9, 11, 50}, {42, 6, 33},
                                                         {64, numerical::detail::gemm_rows_columns, (numerical::detail::gemm_kc * 2) + 5},
                                                         {500, numerical::detail::gemm_rows_columns, 40}}) {
                             std::vector<double> a(m * k);
                             std::vector<double> b(k * n);
                             for (std::size_t i {0}; i < a.size(); ++i) {
                               a[i] = fill<double>(i);
                             }
                             for (std::size_t i {0}; i < b.size(); ++i) {
                               b[i] = fill<double>((i * 7) + 3);
                             }
                             std::vector<double> by_rows(m * n, 0.0);
                             std::vector<double> packed(m * n, 0.0);
                             numerical::detail::gemm(m, n, k, a.data(), k, b.data(), n, by_rows.data(), n);
                             numerical::detail::gemm_large(m, n, k, a.data(), k, b.data(), n, packed.data(), n);
                             same = same && by_rows == packed;
                           }
                         }
                         numerical::set_max_threads(0); // every thread again
                         return same;
                       }));
}

// Past the packing threshold: three depth panels (the last partial), a last partial register tile in
// both directions, and enough tasks to run in parallel.
TEST(packed_product_equals_the_ascending_sum_to_the_bit)
{
  const std::size_t k {(numerical::detail::gemm_kc * 2) + 17};
  EXPECT(on_every_path([&] { return product_is_exact<double>(129, 101, k); }));
  EXPECT(on_every_path([&] { return product_is_exact<double>(33, 15, k); }));                                        // a whole AMX tile's rows and a partial one
  EXPECT(on_every_path([&] { return product_is_exact<double>(31, 15, k); }));                                        // within one tile: the vector tile's
  EXPECT(on_every_path([] { return product_is_exact<double>(50, numerical::detail::gemm_rows_columns + 16, 2); }));  // wide, shallow
  EXPECT(on_every_path([] { return product_is_exact<double>(3, numerical::detail::gemm_rows_columns + 40, 300); })); // few rows: B in place
  EXPECT(on_every_path([&] { return product_is_exact<float>(67, 75, k); }));
  (void)numerical::limit_gemm_path(numerical::gemm_path::portable);                                                  // complex: past it, through the real kernel
  EXPECT(product_is_exact<std::complex<double>>(37, 29, k));
  EXPECT(product_is_exact<std::complex<double>>(50, numerical::detail::gemm_rows_columns + 16, 2));                  // wide, shallow
  EXPECT(product_is_exact<std::complex<double>>(3, numerical::detail::gemm_rows_columns + 16, 30));                  // wide, below packing
  (void)numerical::limit_gemm_path(numerical::gemm_path::amx);
  EXPECT(product_is_exact<int>(70, 50, k));
  EXPECT(product_is_exact<int>(50, numerical::detail::gemm_rows_columns + 16, 2));                                   // wider than by rows, shallow
  EXPECT(product_is_exact<int>(3, numerical::detail::gemm_rows_columns + 16, 30));                                   // wider, below the packing threshold
}

// float products on every path, the AMX's among them: whole and partial tiles in place, a last column cutting a
// tile's first half row or its second, A's slivers by streaming stores to a depth past a multiple of four.
TEST(float_products_equal_the_ascending_sum_to_the_bit)
{
  for (const auto& [rows, cols, depth] : {std::array<std::size_t, 3> {40, 45, 300}, std::array<std::size_t, 3> {70, 60, 130},
                                          std::array<std::size_t, 3> {33, 64, 7}, std::array<std::size_t, 3> {64, 100, 130},
                                          std::array<std::size_t, 3> {128, 24, 64}, std::array<std::size_t, 3> {96, 96, 96}}) {
    EXPECT(on_every_path([&] { return product_is_exact<float>(rows, cols, depth); }));
  }
}

namespace {
  // A complex product past the portable path, which sums the real and imaginary parts of A apart, on small
  // multiples of an eighth: every product and partial sum is exact, so it must equal the ascending sum, on one
  // thread as on all of them; through gemm, or through gemm_split itself where \p Split.
  template <typename Z,
            bool Split = false>
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
    const auto product {[&](std::vector<Z>& c) {
                          if constexpr (Split) {
                            numerical::detail::gemm_split(m, n, k, a.data(), k, b.data(), n, c.data(), n);
                          }
                          else {
                            numerical::detail::gemm(m, n, k, a.data(), k, b.data(), n, c.data(), n);
                          }
                        }};
    product(all);
    numerical::set_max_threads(1);
    product(alone);
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

namespace {
  // Whether the product with A, B or both given as their transposes (row-major, their rows padded) adds to C what
  // gemm of A and B does, to the bit: gemm_bt, gemm_at, gemm_atbt as `which` is 0, 1, 2.
  template <typename T>
  bool transposed_operands_equal_gemm(std::size_t m,
                                      std::size_t n,
                                      std::size_t k,
                                      int         which)
  {
    const std::size_t ldbt {k + 3};
    const std::size_t ldat {m + 2};
    std::vector<T>    a(m * k);
    std::vector<T>    at(k * ldat, T(-5)); // the padding past each row of a transpose is never read
    std::vector<T>    b(k * n);
    std::vector<T>    bt(n * ldbt, T(-7));
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t p {0}; p < k; ++p) {
        a[(i * k) + p]     = entry<T>((i * k) + p);
        at[(p * ldat) + i] = a[(i * k) + p];
      }
    }
    for (std::size_t p {0}; p < k; ++p) {
      for (std::size_t j {0}; j < n; ++j) {
        b[(p * n) + j]     = entry<T>((p * 7) + (j * 11) + 3);
        bt[(j * ldbt) + p] = b[(p * n) + j];
      }
    }
    std::vector<T> want(m * n);
    for (std::size_t e {0}; e < want.size(); ++e) {
      want[e] = entry<T>(e + 5);
    }
    std::vector<T> got {want};
    numerical::detail::gemm(m, n, k, a.data(), k, b.data(), n, want.data(), n);
    if (which == 0) {
      numerical::detail::gemm_bt(m, n, k, a.data(), k, bt.data(), ldbt, got.data(), n);
    }
    else if (which == 1) {
      numerical::detail::gemm_at(m, n, k, at.data(), ldat, b.data(), n, got.data(), n);
    }
    else {
      numerical::detail::gemm_atbt(m, n, k, at.data(), ldat, bt.data(), ldbt, got.data(), n);
    }
    return got == want;
  }
} // namespace

// Either operand given transposed: the plain loop's small products, the tiles' (partial slivers of A and B, several
// panels of depth and blocks of columns), and for complex numbers gemm_split's copies of the parts on either side
// of its choice, on every path this machine runs, in each of the four types; an empty product leaves C as it is.
TEST(transposed_operands_give_gemm_s_bits_on_every_path)
{
  for (const int which : {0, 1, 2}) {
    for (const auto& [m, n, k] : {std::array<std::size_t, 3> {3, 4, 5}, {7, 1, 64}, {64, 64, 64}, {129, 101, 300}, {31, 15, 40},
                                  {20, 1030, 260}, {200, 9, 70}, {2, 600, 300}, {0, 4, 5}, {3, 0, 5}, {3, 4, 0}}) {
      EXPECT(on_every_path([&] { return transposed_operands_equal_gemm<double>(m, n, k, which); }));
      EXPECT(on_every_path([&] { return transposed_operands_equal_gemm<float>(m, n, k, which); }));
      EXPECT(on_every_path([&] { return transposed_operands_equal_gemm<std::complex<double>>(m, n, k, which); }));
      EXPECT(on_every_path([&] { return transposed_operands_equal_gemm<std::complex<float>>(m, n, k, which); }));
    }
  }
}

namespace {
  // Whether gemm_triangle of A (n × k) and B (k × n), the layouts `which` picks (0: both row-major, 1: B given
  // transposed, 2: A given transposed), adds to its triangle of C what gemm adds there, to the bit, and leaves the
  // other triangle as it was.
  bool triangle_equals_gemm(std::size_t n,
                            std::size_t k,
                            int         which,
                            bool        lower)
  {
    std::vector<double> a(n * k);
    std::vector<double> at(k * n);
    std::vector<double> b(k * n);
    std::vector<double> bt(n * k);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t p {0}; p < k; ++p) {
        a[(i * k) + p]  = fill<double>((i * k) + p);
        at[(p * n) + i] = a[(i * k) + p];
        b[(p * n) + i]  = fill<double>((p * 7) + (i * 11) + 3);
        bt[(i * k) + p] = b[(p * n) + i];
      }
    }
    const std::size_t   ldc {n + 2};
    std::vector<double> want(n * ldc);
    for (std::size_t e {0}; e < want.size(); ++e) {
      want[e] = fill<double>(e + 5);
    }
    const std::vector<double> before {want};
    std::vector<double>       got    {want};
    numerical::detail::gemm(n, n, k, a.data(), k, b.data(), n, want.data(), ldc);
    if (which == 0) {
      numerical::detail::gemm_triangle<numerical::detail::a_rows, numerical::detail::b_rows>(n, k, a.data(), k, b.data(), n, got.data(), ldc, lower);
    }
    else if (which == 1) {
      numerical::detail::gemm_triangle<numerical::detail::a_rows, numerical::detail::b_transposed>(n, k, a.data(), k, bt.data(), k, got.data(), ldc, lower);
    }
    else {
      numerical::detail::gemm_triangle<numerical::detail::a_transposed, numerical::detail::b_rows>(n, k, at.data(), n, b.data(), n, got.data(), ldc, lower);
    }
    bool same {true};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < ldc; ++j) {
        const bool   in    {j < n && (lower ? j <= i : j >= i)};
        const double right {in ? want[(i * ldc) + j] : before[(i * ldc) + j]};
        same = same && got[(i * ldc) + j] == right;
      }
    }
    return same;
  }
} // namespace

// The triangle of a product: tiles across the diagonal, partial tiles (a last row of tiles one row deep, whose
// one element in the triangle is its diagonal's: orders 65 and 129), one packed but too small for the AMX (on its path, the NEON
// tile's) and one past every path's block of columns (2100: row blocks with no tile in the triangle left unpacked), several panels of depth and blocks of columns,
// each triangle and layout, on every path this machine runs.
TEST(the_triangle_of_a_product_gives_gemm_s_bits_and_leaves_the_other_alone)
{
  for (const bool lower : {true, false}) {
    for (const int which : {0, 1, 2}) {
      for (const auto& [n, k] : {std::array<std::size_t, 2> {67, 40}, {65, 30}, {129, 40}, {130, 300}, {1100, 70}, {15, 40}, {2100, 9}}) {
        EXPECT(on_every_path([&] { return triangle_equals_gemm(n, k, which, lower); }));
      }
    }
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
  // gemm_split called itself, on every path: arm64 takes it past the portable path, the x86-64 units have complex
  // tiles of their own.
  const std::size_t k {numerical::detail::split_depth};
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<float>, true>(40, 70, 50); }));
  EXPECT(!split_storage<float>().c.empty());
  const std::size_t rows {(numerical::detail::split_kept / (2 * k)) + 8}; // [Re A | −Im A] past the limit
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<float>, true>(rows, 30, k); }));
  EXPECT(split_storage<float>().a.empty());
  EXPECT(on_every_path([&] { return split_product_is_exact<std::complex<float>, true>(40, 70, 50); }));
  EXPECT(!split_storage<float>().c.empty());
}

// gemm_split gives each element of C the chains of one real product of each block's expanded operands,
// `[Re C | Im C] += [Re A | −Im A] · [[Re B, Im B], [Im B, −Re B]]`, by four real products of its parts too (as
// split_by_parts chooses): the same bits, on drawn parts that round, on every path; a row of A or a few, either side of
// each path's choice, and several blocks of columns and of depth.
namespace {
  template <typename Z>
  bool split_product_is_the_expanded_one(std::size_t m,
                                         std::size_t n,
                                         std::size_t k)
  {
    using R = Z::value_type;
    std::mt19937                      draw {static_cast<std::mt19937::result_type>((m * 7919) + (n * 31) + k)};
    std::uniform_real_distribution<R> part {R {-1}, R {1}};
    const auto                        drawn {[&](std::size_t count) {
                                               std::vector<Z> values(count);
                                               for (Z & value : values) {
                                                 value = Z {part(draw), part(draw)};
                                               }
                                               return values;
                                             }};
    const std::vector<Z> a {drawn(m * k)};
    const std::vector<Z> b {drawn(k * n)};
    std::vector<Z>       c {drawn(m * n)};
    std::vector<R>       expanded_c(m * 2 * n);
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        expanded_c[(i * 2 * n) + j]     = c[(i * n) + j].real();
        expanded_c[(i * 2 * n) + n + j] = c[(i * n) + j].imag();
      }
    }
    for (std::size_t jc {0}; jc < n; jc += numerical::detail::split_columns) {
      const std::size_t nb {std::min(numerical::detail::split_columns, n - jc)};
      for (std::size_t pc {0}; pc < k; pc += numerical::detail::split_depth) {
        const std::size_t kb {std::min(numerical::detail::split_depth, k - pc)};
        std::vector<R>    ar(m * 2 * kb);
        std::vector<R>    br(2 * kb * 2 * nb);
        for (std::size_t i {0}; i < m; ++i) {
          for (std::size_t p {0}; p < kb; ++p) {
            ar[(i * 2 * kb) + p]      = a[(i * k) + pc + p].real();
            ar[(i * 2 * kb) + kb + p] = -a[(i * k) + pc + p].imag();
          }
        }
        for (std::size_t p {0}; p < kb; ++p) {
          for (std::size_t j {0}; j < nb; ++j) {
            const Z value {b[((pc + p) * n) + jc + j]};
            br[(p * 2 * nb) + j]             = value.real();
            br[(p * 2 * nb) + nb + j]        = value.imag();
            br[((kb + p) * 2 * nb) + j]      = value.imag();
            br[((kb + p) * 2 * nb) + nb + j] = -value.real();
          }
        }
        std::vector<R> cr(m * 2 * nb);
        for (std::size_t i {0}; i < m; ++i) {
          for (std::size_t j {0}; j < nb; ++j) {
            cr[(i * 2 * nb) + j]      = expanded_c[(i * 2 * n) + jc + j];
            cr[(i * 2 * nb) + nb + j] = expanded_c[(i * 2 * n) + n + jc + j];
          }
        }
        numerical::detail::gemm(m, 2 * nb, 2 * kb, ar.data(), 2 * kb, br.data(), 2 * nb, cr.data(), 2 * nb);
        for (std::size_t i {0}; i < m; ++i) {
          for (std::size_t j {0}; j < nb; ++j) {
            expanded_c[(i * 2 * n) + jc + j]     = cr[(i * 2 * nb) + j];
            expanded_c[(i * 2 * n) + n + jc + j] = cr[(i * 2 * nb) + nb + j];
          }
        }
      }
    }
    numerical::detail::gemm_split(m, n, k, a.data(), k, b.data(), n, c.data(), n);
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        if (c[(i * n) + j] != Z {expanded_c[(i * 2 * n) + j], expanded_c[(i * 2 * n) + n + j]}) {
          return false;
        }
      }
    }
    return true;
  }
} // namespace

TEST(split_products_give_the_expanded_product_s_bits)
{
  const std::size_t n {numerical::detail::split_columns + 37};
  const std::size_t k {numerical::detail::split_depth + 19};
  for (const std::size_t m : {std::size_t {1}, std::size_t {3}, std::size_t {5}, std::size_t {45}, numerical::detail::split_parts_rows,
                              numerical::detail::split_parts_rows + 1}) {
    EXPECT(on_every_path([&] { return split_product_is_the_expanded_one<std::complex<float>>(m, n, k); }));
    EXPECT(on_every_path([&] { return split_product_is_the_expanded_one<std::complex<double>>(m, n, k); }));
    EXPECT(on_every_path([&] { return split_product_is_the_expanded_one<std::complex<float>>(m, 70, 135); }));
    EXPECT(on_every_path([&] { return split_product_is_the_expanded_one<std::complex<double>>(m, 5, 9); }));
  }
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

namespace {

  // C += A·B whole, and again as two products of its columns cut at each multiple of 8: the same bits, so that a
  // caller splitting its columns by the threads (an LU inverse's slices) has the same answer on any number of them.
  // The pieces run the plain loop and the packed tiles in turn (16 rows, depth 16: 131 columns pack, 8 do not).
  template <typename T>
  bool columns_cut_anywhere_give_the_same_bits()
  {
    const std::size_t m {16};
    const std::size_t k {16};
    const std::size_t n {131};
    std::vector<T>    a(m * k);
    std::vector<T>    b(k * n);
    std::vector<T>    start(m * n);
    for (std::size_t i {0}; i < a.size(); ++i) {
      a[i] = entry<T>((i * 13) + 1) / T {7};
    }
    for (std::size_t i {0}; i < b.size(); ++i) {
      b[i] = entry<T>((i * 5) + 2) / T {3};
    }
    for (std::size_t i {0}; i < start.size(); ++i) {
      start[i] = entry<T>(i) / T {11};
    }
    std::vector<T> whole {start};
    numerical::detail::gemm(m, n, k, a.data(), k, b.data(), n, whole.data(), n);
    bool same            {true};
    for (std::size_t cut {8}; cut < n; cut += 8) {
      std::vector<T> pieces {start};
      numerical::detail::gemm(m, cut, k, a.data(), k, b.data(), n, pieces.data(), n);
      numerical::detail::gemm(m, n - cut, k, a.data(), k, b.data() + cut, n, pieces.data() + cut, n);
      same = same && pieces == whole;
    }
    return same;
  }
} // namespace

TEST(a_product_cut_by_its_columns_gives_the_same_bits)
{
  EXPECT(on_every_path([] { return columns_cut_anywhere_give_the_same_bits<double>(); }));
  EXPECT(on_every_path([] { return columns_cut_anywhere_give_the_same_bits<float>(); }));
  EXPECT(on_every_path([] { return columns_cut_anywhere_give_the_same_bits<std::complex<double>>(); }));
  EXPECT(on_every_path([] { return columns_cut_anywhere_give_the_same_bits<std::complex<float>>(); }));
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

// The packing storage stays with the calling thread for the next product, unless it outgrew the limit. The kept
// products are 128 deep with 128 rows: every tile packs them (a shallower or narrower one may run in the plain loop).
TEST(the_calling_thread_keeps_its_packing_storage_within_the_limit)
{
  using storage = numerical::detail::packing_storage<double>;
  EXPECT(on_every_path([] {
                         bool ok {product_is_exact<double>(128, 128, 128) && !storage::kept().empty()};
                         // A tall product packs more of A than the limit: correct, then nothing kept.
                         const std::size_t rows {(storage::kept_limit / numerical::detail::gemm_kc) + 64};
                         ok = ok && product_is_exact<double>(rows, numerical::detail::gemm_rows_columns + 1, numerical::detail::gemm_kc) && storage::kept().empty();
                         return ok && product_is_exact<double>(128, 128, 128) && !storage::kept().empty();
                       }));
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

namespace {
  // Whether the complex dot products of type C are close to a long double sum at every length about the vectors' widths
  // (the native vectors' parts, AVX2's two or four complex numbers, four accumulators of each), each row of row_dots
  // and of matrix_times_vector its dot_product to the bit.
  template <typename C>
  bool complex_dots_hold()
  {
    using R = C::value_type;
    const auto value = [](std::size_t i, std::size_t salt) {
                         const auto phase {static_cast<double>(((i * 2654435761U) + salt) % 1000U) / 1000.0};
                         return C {static_cast<R>(phase - 0.5), static_cast<R>(0.25 - (phase * phase))};
                       };
    bool ok {true};
    for (const std::size_t count : {std::size_t {0}, std::size_t {1}, std::size_t {3}, std::size_t {15}, std::size_t {16},
                                    std::size_t {17}, std::size_t {33}, std::size_t {1000}}) {
      std::vector<C>            x(count);
      std::vector<C>            y(count);
      std::complex<long double> want {};
      for (std::size_t i {0}; i < count; ++i) {
        x[i]  = value(i, 1);
        y[i]  = value(i, 7);
        want += std::complex<long double> {x[i].real(), x[i].imag()} *std::complex<long double> {y[i].real(), y[i].imag()};
      }
      const C           got   {numerical::detail::dot_product(x.data(), y.data(), count)};
      const long double bound {(std::is_same_v<R, float> ? 1e-4L : 1e-12L) * static_cast<long double>(count + 1)};
      ok = ok && std::abs(std::complex<long double> {got.real(), got.imag()} - want) <= bound;
    }
    constexpr std::size_t rows {189};
    constexpr std::size_t cols {206};
    std::vector<C>        m(rows * cols);
    std::vector<C>        v(cols);
    for (std::size_t i {0}; i < m.size(); ++i) {
      m[i] = value(i, 3);
    }
    for (std::size_t i {0}; i < cols; ++i) {
      v[i] = value(i, 5);
    }
    std::vector<C> by_rows(rows);
    std::vector<C> product(rows);
    numerical::detail::row_dots(m.data(), cols, rows, v.data(), cols, by_rows.data());
    numerical::detail::matrix_times_vector(m.data(), rows, cols, v.data(), product.data());
    for (std::size_t r {0}; r < rows; ++r) {
      const C alone {numerical::detail::dot_product(m.data() + (r * cols), v.data(), cols)};
      ok = ok && by_rows[r] == alone && product[r] == alone;
    }
    return ok;
  }
} // namespace

// The complex dot products, on every path (AVX2's fused chains on the x86-64 units, the native vectors' elsewhere).
TEST(complex_dot_products_on_every_path)
{
  EXPECT(on_every_path([] { return complex_dots_hold<std::complex<double>>(); }));
  EXPECT(on_every_path([] { return complex_dots_hold<std::complex<float>>(); }));
}
