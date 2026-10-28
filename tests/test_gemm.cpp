// The matrix product kernel (gemm.hpp): every element of C is summed in ascending order of the inner
// index, so the packed kernel equals the plain i-k-j loop to the bit, across several depth panels,
// several column blocks, partial register tiles, every scalar it is instantiated for, and any number of
// threads. Exact equality is the property; a tolerance would hide a reordered sum.
#include <algorithm>
#include <array>
#include <cmath>
#include <bit>
#include <complex>
#include <cstdint>
#include <limits>
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

  // Whether a and b are the same bits: a real number's, or each part of a complex one's.
  template <typename T>
  bool same_bits(T a,
                 T b)
  {
    if constexpr (std::is_floating_point_v<T>) {
      using bits = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
      return std::bit_cast<bits>(a) == std::bit_cast<bits>(b);
    }
    else {
      return same_bits(a.real(), b.real()) && same_bits(a.imag(), b.imag());
    }
  }

  // fill's value, its imaginary part filled too for a complex T (a zero one would hide how its products round).
  template <typename T>
  T entry(std::size_t i)
  {
    if constexpr (std::is_arithmetic_v<T>) {
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

// A product of up to gemm_rows_columns columns, or of up to gemm_rows_work multiply-adds from narrow_group rows, runs
// by rows (gemm_rows): the packed product's bits, on every path, past one depth panel, any number of rows and threads,
// every width of a last block of columns after whole blocks of rows and after a last row alone.
TEST(narrow_products_by_rows_give_the_packed_bits)
{
  EXPECT(on_every_path([] {
                         bool same {true};
                         for (const std::size_t threads : {std::size_t {1}, std::size_t {4}}) {
                           numerical::set_max_threads(threads);
                           for (const auto& [m, n, k] : {std::array<std::size_t, 3> {169, 2, 579}, {37, 7, 37}, {300, 8, 300}, {130, 13, 600},
                                                         {41, 3, 90}, {23, 14, 70}, {9, 11, 50}, {42, 6, 33},
                                                         {64, 64, 64}, {80, 80, 80}, {40, 100, 60}, {4, 40, 75}, {3, 138, 137},
                                                         {128, 128, 128}, {150, 100, 120}, {12, 300, 300},                      // by rows on four threads
                                                         {64, numerical::detail::gemm_rows_columns, (numerical::detail::gemm_kc * 2) + 5},
                                                         {500, numerical::detail::gemm_rows_columns, 40},
                                                         {1, 517, 300}, {2, 84, 987}, {3, 200, 50}, {4, 301, 300}, {1, 33, 40}, // a few rows by blocks of columns
                                                         {3, numerical::detail::gemm_rows_columns + 1, 70}}) {
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

// Complex products by rows (gemm_rows, which the LU's and the Cholesky factorisation's panels call): each element its
// start (C's own, or zero) plus a(i,p)·b(p,j) for p ascending, the library's product rounded before its sum, on every
// path and width (the x86-64 vectors' two columns apiece and a last one alone, past four columns in two parts), and
// with infinities whose product has two NaN parts, which the library rescues where it follows C's Annex G: the block
// then left to the scalar loop.
TEST(complex_products_by_rows_give_the_ascending_sum)
{
  using C = std::complex<double>;
  const auto same_part = [](double x, double y) { return (std::isnan(x) && std::isnan(y)) || (x == y && std::signbit(x) == std::signbit(y)); };
  EXPECT(on_every_path([&] {
                         bool ok {true};
                         for (const std::size_t n : {1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U, 13U, 24U}) {
                           for (const bool infinite : {false, true}) {
                             const std::size_t m {11};
                             const std::size_t k {37};
                             std::vector<C>    a(m * k);
                             std::vector<C>    b(k * n);
                             std::vector<C>    c(m * n);
                             for (std::size_t i {0}; i < a.size(); ++i) {
                               a[i] = entry<C>(i);
                             }
                             for (std::size_t i {0}; i < b.size(); ++i) {
                               b[i] = entry<C>((i * 7) + 3);
                             }
                             for (std::size_t i {0}; i < c.size(); ++i) {
                               c[i] = entry<C>((i * 5) + 2);
                             }
                             if (infinite) { // (inf, inf)·(0, 1): both parts NaN, the library's rescue an infinity
                               const double inf {std::numeric_limits<double>::infinity()};
                               a[(3 * k) + 4]       = C {inf, inf};
                               b[(4 * n) + (n - 1)] = C {0.0, 1.0};
                             }
                             std::vector<C> want {c};
                             for (std::size_t i {0}; i < m; ++i) {
                               for (std::size_t p {0}; p < k; ++p) {
                                 for (std::size_t j {0}; j < n; ++j) {
                                   want[(i * n) + j] = want[(i * n) + j] + (a[(i * k) + p] * b[(p * n) + j]);
                                 }
                               }
                             }
                             std::vector<C> kept {c};
                             numerical::detail::gemm_rows(m, n, k, a.data(), k, b.data(), n, kept.data(), n);
                             std::vector<C> zero(m * n, C {std::numeric_limits<double>::quiet_NaN(), 0.0}); // not read
                             numerical::detail::gemm_rows(m, n, k, a.data(), k, b.data(), n, zero.data(), n, numerical::detail::c_start::zero);
                             std::vector<C>  from_zero(m * n);
                             for (std::size_t i {0}; i < m; ++i) {
                               for (std::size_t p {0}; p < k; ++p) {
                                 for (std::size_t j {0}; j < n; ++j) {
                                   from_zero[(i * n) + j] = (p == 0 ? C {} : from_zero[(i * n) + j]) + (a[(i * k) + p] * b[(p * n) + j]);
                                 }
                               }
                             }
                             for (std::size_t e {0}; e < m * n; ++e) {
                               ok = ok && same_part(kept[e].real(), want[e].real()) && same_part(kept[e].imag(), want[e].imag());
                               ok = ok && same_part(zero[e].real(), from_zero[e].real()) && same_part(zero[e].imag(), from_zero[e].imag());
                             }
                             if (infinite) { // where the library rescues the product (C's Annex G; MSVC's does not), the sum is infinite
                               const double inf {std::numeric_limits<double>::infinity()};
                               ok = ok && std::isinf(want[(3 * n) + (n - 1)].real()) == std::isinf((C {inf, inf} *C {0.0, 1.0}).real());
                             }
                           }
                         }
                         return ok;
                       }));
}

// A product of one to four rows of float, by blocks of columns on the threads where the x86-64 units run it: the
// packed product's bits, a last block of columns narrower, one thread and four.
TEST(float_products_of_a_few_rows_give_the_packed_bits)
{
  EXPECT(on_every_path([] {
                         bool same {true};
                         for (const std::size_t threads : {std::size_t {1}, std::size_t {4}}) {
                           numerical::set_max_threads(threads);
                           for (const auto& [m, n, k] : {std::array<std::size_t, 3> {1, 517, 300}, {2, 84, 987}, {3, 200, 50}, {4, 301, 300}}) {
                             std::vector<float> a(m * k);
                             std::vector<float> b(k * n);
                             for (std::size_t i {0}; i < a.size(); ++i) {
                               a[i] = fill<float>(i);
                             }
                             for (std::size_t i {0}; i < b.size(); ++i) {
                               b[i] = fill<float>((i * 7) + 3);
                             }
                             std::vector<float> by_blocks(m * n, 0.0F);
                             std::vector<float> packed(m * n, 0.0F);
                             numerical::detail::gemm(m, n, k, a.data(), k, b.data(), n, by_blocks.data(), n);
                             numerical::detail::gemm_large(m, n, k, a.data(), k, b.data(), n, packed.data(), n);
                             same = same && by_blocks == packed;
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

namespace
{
  // The pieces run a few rows, a tile short of rows (each count up to the widest tile's twelve) and whole tiles in
  // turn: every element of C the same as in the whole product.
  template <typename T>
  bool rows_cut_anywhere_give_the_same_bits()
  {
    const std::size_t m {30};
    const std::size_t k {64};
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
    for (std::size_t cut {1}; cut < m; ++cut) {
      std::vector<T> pieces {start};
      numerical::detail::gemm(cut, n, k, a.data(), k, b.data(), n, pieces.data(), n);
      numerical::detail::gemm(m - cut, n, k, a.data() + (cut * k), k, b.data(), n, pieces.data() + (cut * n), n);
      same = same && pieces == whole;
    }
    return same;
  }
} // namespace

namespace
{
  // A product of a few columns of B, A row-major or given as its transpose (a sliver of columns: A read where it lies
  // on the x86-64 complex tiles), gives the columns of the wider product, whose A is packed: rows short of a tile,
  // several panels of depth.
  template <typename T>
  bool a_few_columns_give_the_wide_products_bits()
  {
    const std::size_t m    {29};
    const std::size_t k    {300};
    const std::size_t wide {40};
    const std::size_t ldat {m + 3};
    std::vector<T>    a(m * k);
    std::vector<T>    at(k * ldat, T(-5));
    std::vector<T>    b(k * wide);
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t p {0}; p < k; ++p) {
        a[(i * k) + p]     = entry<T>((i * 13) + (p * 3) + 1) / T {7};
        at[(p * ldat) + i] = a[(i * k) + p];
      }
    }
    for (std::size_t e {0}; e < b.size(); ++e) {
      b[e] = entry<T>((e * 5) + 2) / T {3};
    }
    std::vector<T> whole(m * wide);
    for (std::size_t e {0}; e < whole.size(); ++e) {
      whole[e] = entry<T>(e) / T {11};
    }
    const std::vector<T> start {whole};
    numerical::detail::gemm(m, wide, k, a.data(), k, b.data(), wide, whole.data(), wide);
    bool same                  {true};
    for (std::size_t n {1}; n <= 17; ++n) {
      std::vector<T> rows       {start};
      std::vector<T> transposed {start};
      numerical::detail::gemm(m, n, k, a.data(), k, b.data(), wide, rows.data(), wide);
      numerical::detail::gemm_at(m, n, k, at.data(), ldat, b.data(), wide, transposed.data(), wide);
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < wide; ++j) {
          const T expected {j < n ? whole[(i * wide) + j] : start[(i * wide) + j]};
          same = same && rows[(i * wide) + j] == expected && transposed[(i * wide) + j] == expected;
        }
      }
    }
    return same;
  }
} // namespace

TEST(a_product_of_a_few_columns_gives_the_wide_products_bits)
{
  EXPECT(on_every_path([] { return a_few_columns_give_the_wide_products_bits<double>(); }));
  EXPECT(on_every_path([] { return a_few_columns_give_the_wide_products_bits<float>(); }));
  EXPECT(on_every_path([] { return a_few_columns_give_the_wide_products_bits<std::complex<double>>(); }));
  EXPECT(on_every_path([] { return a_few_columns_give_the_wide_products_bits<std::complex<float>>(); }));
}

namespace
{
  // Products of one to four rows (on x86-64, complex ones stream B's rows, three and four rows past few_rows_complex_least
  // elements of B) give the rows of a product of eleven (the tiles), A, B and C each a block of a wider matrix, C kept
  // or zeroed first; B of 300 × 260, several panels of depth.
  template <typename T>
  bool a_few_rows_give_a_tall_products_bits()
  {
    const std::size_t m   {11};
    const std::size_t k   {300};
    const std::size_t n   {260};
    const std::size_t lda {k + 5};
    const std::size_t ldb {n + 3};
    const std::size_t ldc {n + 7};
    std::vector<T>    a(m * lda);
    std::vector<T>    b(k * ldb);
    std::vector<T>    start(m * ldc);
    for (std::size_t e {0}; e < a.size(); ++e) {
      a[e] = entry<T>((e * 13) + 1) / T {7};
    }
    for (std::size_t e {0}; e < b.size(); ++e) {
      b[e] = entry<T>((e * 5) + 2) / T {3};
    }
    for (std::size_t e {0}; e < start.size(); ++e) {
      start[e] = entry<T>(e) / T {11};
    }
    bool same {true};
    for (const auto begin : {numerical::detail::c_start::kept, numerical::detail::c_start::zero}) {
      std::vector<T> tall {start};
      numerical::detail::gemm(m, n, k, a.data(), lda, b.data(), ldb, tall.data(), ldc, begin);
      for (std::size_t rows {1}; rows <= 4; ++rows) {
        for (const std::size_t first : {std::size_t {0}, m - rows}) {
          std::vector<T> few {start};
          numerical::detail::gemm(rows, n, k, a.data() + (first * lda), lda, b.data(), ldb, few.data() + (first * ldc), ldc, begin);
          for (std::size_t i {0}; i < m; ++i) {
            for (std::size_t j {0}; j < ldc; ++j) {
              const bool computed {i >= first && i < first + rows && j < n};
              same = same && few[(i * ldc) + j] == (computed ? tall[(i * ldc) + j] : start[(i * ldc) + j]);
            }
          }
        }
      }
    }
    return same;
  }
} // namespace

TEST(a_product_of_a_few_rows_gives_a_tall_products_bits)
{
  EXPECT(on_every_path([] { return a_few_rows_give_a_tall_products_bits<double>(); }));
  EXPECT(on_every_path([] { return a_few_rows_give_a_tall_products_bits<float>(); }));
  EXPECT(on_every_path([] { return a_few_rows_give_a_tall_products_bits<std::complex<double>>(); }));
  EXPECT(on_every_path([] { return a_few_rows_give_a_tall_products_bits<std::complex<float>>(); }));
}

TEST(a_product_cut_by_its_rows_gives_the_same_bits)
{
  EXPECT(on_every_path([] { return rows_cut_anywhere_give_the_same_bits<double>(); }));
  EXPECT(on_every_path([] { return rows_cut_anywhere_give_the_same_bits<float>(); }));
  EXPECT(on_every_path([] { return rows_cut_anywhere_give_the_same_bits<std::complex<double>>(); }));
  EXPECT(on_every_path([] { return rows_cut_anywhere_give_the_same_bits<std::complex<float>>(); }));
}

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
// products are 128 deep with 128 rows, on one thread: every tile packs them (a shallower or narrower one may run in
// the plain loop, and on threads a product of order 128 by rows).
TEST(the_calling_thread_keeps_its_packing_storage_within_the_limit)
{
  using storage = numerical::detail::packing_storage<double>;
  numerical::set_max_threads(1);
  EXPECT(on_every_path([] {
                         bool ok {product_is_exact<double>(128, 128, 128) && !storage::kept().empty()};
                         // A tall product packs more of A than the limit: correct, then nothing kept.
                         const std::size_t rows {(storage::kept_limit / numerical::detail::gemm_kc) + 64};
                         ok = ok && product_is_exact<double>(rows, numerical::detail::gemm_rows_columns + 1, numerical::detail::gemm_kc) && storage::kept().empty();
                         return ok && product_is_exact<double>(128, 128, 128) && !storage::kept().empty();
                       }));
  numerical::set_max_threads(0); // every thread again
}

namespace
{
  // A value C holds before a product that does not read it: NaN, or for int a value no product below reaches.
  template <typename T>
  T stale()
  {
    if constexpr (std::is_same_v<T, int>) {
      return 1 << 30;
    }
    else if constexpr (std::is_floating_point_v<T>) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    else {
      return T {stale<typename T::value_type>(), stale<typename T::value_type>()};
    }
  }

  // Whether \p value is still stale: NaN, or for int the value stale gave.
  template <typename T>
  bool is_stale(const T& value)
  {
    if constexpr (std::is_same_v<T, int>) {
      return value == stale<int>();
    }
    else if constexpr (std::is_floating_point_v<T>) {
      return std::isnan(value);
    }
    else {
      return is_stale(value.real()) && is_stale(value.imag());
    }
  }

  // gemm with c_start::zero into a C of stale values, its rows \p ldc apart: the bits of C zeroed then the product
  // added, and the columns past it untouched.
  template <typename T>
  bool product_into_stale_c_is_zeroed_then_added_apart(std::size_t m,
                                                       std::size_t n,
                                                       std::size_t k,
                                                       std::size_t ldc)
  {
    std::vector<T>    a(m * k);
    std::vector<T>    b(k * n);
    for (std::size_t i {0}; i < a.size(); ++i) {
      a[i] = entry<T>(i);
    }
    for (std::size_t i {0}; i < b.size(); ++i) {
      b[i] = entry<T>((i * 7) + 3);
    }
    std::vector<T> added(m * ldc, T {});
    std::vector<T> into(m * ldc, stale<T>());
    numerical::detail::gemm(m, n, k, a.data(), k, b.data(), n, added.data(), ldc);
    numerical::detail::gemm(m, n, k, a.data(), k, b.data(), n, into.data(), ldc, numerical::detail::c_start::zero);
    bool same {true};
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t j {0}; j < ldc; ++j) {
        const T& got {into[(i * ldc) + j]};
        same = same && (j < n ? got == added[(i * ldc) + j] : is_stale(got));
      }
    }
    return same;
  }

  // The same, C a whole matrix and a block of a wider one.
  template <typename T>
  bool product_into_stale_c_is_zeroed_then_added(std::size_t m,
                                                 std::size_t n,
                                                 std::size_t k)
  {
    return product_into_stale_c_is_zeroed_then_added_apart<T>(m, n, k, n) && product_into_stale_c_is_zeroed_then_added_apart<T>(m, n, k, n + 3);
  }
} // namespace

// gemm into a C it does not read (c_start::zero), each part of C zeroed by the thread that computes it: the bits of C
// zeroed then the product added, on every path, one and four threads, by rows (narrow, small, of depth one), packed
// (one depth panel and three, several blocks of columns), and of depth zero; C a whole matrix and a block of a wider one.
TEST(a_product_into_c_gives_the_bits_of_c_zeroed_then_added)
{
  const std::size_t wide {std::max(numerical::detail::gemm_nc<double>, std::size_t {16} *128) + 13};
  const std::size_t deep {(numerical::detail::gemm_kc * 2) + 17};
  EXPECT(on_every_path([&] {
                         bool same {true};
                         for (const std::size_t threads : {std::size_t {1}, std::size_t {4}}) {
                           numerical::set_max_threads(threads);
                           for (const auto& [m, n, k] : {std::array<std::size_t, 3> {3, 5, 7}, {169, 2, 579}, {64, 64, 64}, {40, 40, 1},
                                                         {130, 101, 600}, {129, 67, deep}, {9, wide, 40}, {5, 6, 0}}) {
                             same = same && product_into_stale_c_is_zeroed_then_added<double>(m, n, k) &&
                                    product_into_stale_c_is_zeroed_then_added<float>(m, n, k) &&
                                    product_into_stale_c_is_zeroed_then_added<std::complex<double>>(m, n, k) &&
                                    product_into_stale_c_is_zeroed_then_added<std::complex<float>>(m, n, k) &&
                                    product_into_stale_c_is_zeroed_then_added<int>(m, n, k);
                           }
                         }
                         numerical::set_max_threads(0); // every thread again
                         return same;
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
    numerical::detail::matrix_times_vector(m.data(), cols, rows, cols, v.data(), product.data());
    for (std::size_t r {0}; r < rows; ++r) {
      const C alone {numerical::detail::dot_product(m.data() + (r * cols), v.data(), cols)};
      ok = ok && by_rows[r] == alone && product[r] == alone;
    }
    return ok;
  }
} // namespace

namespace
{
  // row_dots of rows ld apart: each row the bits of its own dot_product (complex_dot_product for a complex T).
  template <typename T>
  bool each_row_is_its_dot_product(std::size_t rows,
                                   std::size_t count)
  {
    const std::size_t ld {count + 3};
    std::vector<T>    m(rows * ld);
    std::vector<T>    x(count);
    for (std::size_t i {0}; i < m.size(); ++i) {
      m[i] = entry<T>(i);
    }
    for (std::size_t i {0}; i < count; ++i) {
      x[i] = entry<T>((i * 5) + 2);
    }
    std::vector<T> out(rows);
    numerical::detail::row_dots(m.data(), ld, rows, x.data(), count, out.data());
    bool same {true};
    for (std::size_t r {0}; r < rows; ++r) {
      if constexpr (std::is_floating_point_v<T>) {
        same = same && out[r] == numerical::detail::dot_product(m.data() + (r * ld), x.data(), count);
      }
      else {
        same = same && out[r] == numerical::detail::complex_dot_product(m.data() + (r * ld), x.data(), count);
      }
    }
    return same;
  }
} // namespace

namespace {

  // columns_times_vector of T against matrix_times_vector of the row-major copy, bit for bit where it runs, nothing
  // written where it declines.
  template <typename T>
  bool column_major_gives_copys_bits()
  {
    bool same {true};
    for (const std::size_t cols : {std::size_t {1}, std::size_t {15}, std::size_t {16}, std::size_t {17}, std::size_t {31},
                                   std::size_t {32}, std::size_t {33}, std::size_t {63}, std::size_t {64}, std::size_t {65},
                                   std::size_t {129}, std::size_t {275}}) {
      for (const std::size_t rows : {std::size_t {1}, std::size_t {3}, std::size_t {4}, std::size_t {9}, std::size_t {37},
                                     numerical::detail::column_rows + 3}) {
        std::vector<T> columns(rows * cols);
        std::vector<T> row_major(rows * cols);
        std::vector<T> x(cols);
        for (std::size_t i {0}; i < rows; ++i) {
          for (std::size_t j {0}; j < cols; ++j) {
            columns[i + (j * rows)]   = entry<T>((i * cols) + j);
            row_major[(i * cols) + j] = columns[i + (j * rows)];
          }
        }
        for (std::size_t j {0}; j < cols; ++j) {
          x[j] = entry<T>((j * 5) + 2);
        }
        std::vector<T> want(rows);
        std::vector<T> got(rows, T {-1});
        numerical::detail::matrix_times_vector(row_major.data(), cols, rows, cols, x.data(), want.data());
        if (numerical::detail::columns_times_vector(columns.data(), rows, rows, cols, x.data(), got.data())) {
          for (std::size_t i {0}; i < rows; ++i) {
            same = same && same_bits(got[i], want[i]);
          }
        }
        else {
          same = same && got == std::vector<T>(rows, T {-1}); // declined: nothing written
        }
      }
    }
    { // past the AMX's range of sizes (2^18 elements): the portable vectors' partials again on arm64
      const std::size_t rows {640};
      const std::size_t cols {450};
      std::vector<T>    columns(rows * cols);
      std::vector<T>    row_major(rows * cols);
      std::vector<T>    x(cols);
      for (std::size_t i {0}; i < rows; ++i) {
        for (std::size_t j {0}; j < cols; ++j) {
          columns[i + (j * rows)]   = entry<T>((i * cols) + j);
          row_major[(i * cols) + j] = columns[i + (j * rows)];
        }
      }
      for (std::size_t j {0}; j < cols; ++j) {
        x[j] = entry<T>(j + 3);
      }
      std::vector<T> want(rows);
      std::vector<T> got(rows, T {-1});
      numerical::detail::matrix_times_vector(row_major.data(), cols, rows, cols, x.data(), want.data());
      if (numerical::detail::columns_times_vector(columns.data(), rows, rows, cols, x.data(), got.data())) {
        for (std::size_t i {0}; i < rows; ++i) {
          same = same && same_bits(got[i], want[i]);
        }
      }
    }
    std::vector<T> long_row(numerical::detail::dot_chunk + 1, T {1});
    T              out                                          {};
    return same && !numerical::detail::columns_times_vector(long_row.data(), 1, 1, long_row.size(), long_row.data(), &out);
  }
} // namespace

// A column-major matrix times a vector (columns_times_vector), in the four types: each element the bits of its row-major
// copy's product (matrix_times_vector), for every length about the kernels' partial sums and a complex row's slots, rows
// past whole blocks and vectors, and on every path; rows longer than a chunk, and the paths it does not run, decline.
TEST(a_column_major_matrix_times_a_vector_gives_its_copys_bits)
{
  EXPECT(on_every_path([] { return column_major_gives_copys_bits<double>() && column_major_gives_copys_bits<float>(); }));
  EXPECT(on_every_path([] {
                         return column_major_gives_copys_bits<std::complex<double>>() && column_major_gives_copys_bits<std::complex<float>>();
                       }));
}

// The rows of a matrix times a vector (row_dots), four at a time on the x86-64 units with their remainders together:
// each row the bits of its dot product alone, on every path, for every length about the 16-, 32- and 64-element
// blocks and a number of rows past whole groups of four.
TEST(row_dots_give_each_rows_dot_product)
{
  EXPECT(on_every_path([] {
                         bool same {true};
                         for (const std::size_t count : {std::size_t {0}, std::size_t {1}, std::size_t {3}, std::size_t {15},
                                                         std::size_t {16}, std::size_t {17}, std::size_t {31}, std::size_t {32},
                                                         std::size_t {33}, std::size_t {59}, std::size_t {63}, std::size_t {64},
                                                         std::size_t {65}, std::size_t {275}}) {
                           for (const std::size_t rows : {std::size_t {1}, std::size_t {4}, std::size_t {37}}) {
                             same = same && each_row_is_its_dot_product<double>(rows, count) && each_row_is_its_dot_product<float>(rows, count) &&
                                    each_row_is_its_dot_product<std::complex<double>>(rows, count) &&
                                    each_row_is_its_dot_product<std::complex<float>>(rows, count);
                           }
                         }
                         return same;
                       }));
}

// Whole numbers whose every partial sum is exact in float: any order of summation gives the exact dot product, so
// float's kernels (four accumulators of 8 or 16 on the x86-64 units, the portable vectors elsewhere), their lanes and
// their remainder must each equal it, on every path, about the 32- and 64-element blocks.
TEST(float_dot_products_sum_every_term_on_every_path)
{
  EXPECT(on_every_path([] {
                         bool same {true};
                         for (const std::size_t count : {std::size_t {0}, std::size_t {5}, std::size_t {31}, std::size_t {32},
                                                         std::size_t {33}, std::size_t {64}, std::size_t {77}, std::size_t {200}}) {
                           std::vector<float> x(count);
                           std::vector<float> y(count);
                           float              want {0.0F};
                           for (std::size_t i {0}; i < count; ++i) {
                             x[i]  = static_cast<float>(i % 7) - 3.0F;
                             y[i]  = static_cast<float>(i % 5) + 1.0F;
                             want += x[i] * y[i];
                           }
                           same = same && numerical::detail::dot_product(x.data(), y.data(), count) == want;
                         }
                         return same;
                       }));
}

// The complex dot products, on every path (AVX2's fused chains on the x86-64 units, the native vectors' elsewhere).
TEST(complex_dot_products_on_every_path)
{
  EXPECT(on_every_path([] { return complex_dots_hold<std::complex<double>>(); }));
  EXPECT(on_every_path([] { return complex_dots_hold<std::complex<float>>(); }));
}
