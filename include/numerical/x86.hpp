/*!
 * \file x86.hpp
 * \brief The x86-64 tiles of the `double` matrix product: AVX-512 and AVX2 with FMA, chosen at run time.
 *
 * A wheel is built for the baseline x86-64 (SSE2), so the wider units are reached function by function: each
 * tile is compiled for its instruction set (GCC's and Clang's `target` attribute; MSVC emits the intrinsics as
 * written) and used only once `cpuid` reports the instructions and `xgetbv` reports the system saving their
 * registers. Each element of C is then a chain of fused multiply-adds in ascending order of the depth, the same on
 * any number of threads.
 */
#ifndef NUMERICAL_X86_HPP
#define NUMERICAL_X86_HPP

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>

#include "simd.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#  define NUMERICAL_X86_COMPILED // the x86-64 tiles are compiled in (tested with #ifdef only)
#  include <immintrin.h>
#  if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#    define NUMERICAL_TARGET(isa)
#    define NUMERICAL_UNROLL_TILE
#    define NUMERICAL_UNROLL_DEPTH
#    define NUMERICAL_UNROLL_DEPTH_AVX2
#  else
#    include <cpuid.h>
#    define NUMERICAL_TARGET(isa) __attribute__((target(isa))) // NOLINT(cppcoreguidelines-macro-usage): an attribute, which no function can stand for
// A tile loop unrolled whole, so each accumulator stays in its register (rolled, GCC keeps them in memory).
#    define NUMERICAL_UNROLL_TILE _Pragma("GCC unroll 24")
// The depth loop unrolled by two: half the loop's own instructions, which compete with the multiply-adds for
// issue slots (an AVX-512 tile in the L1 cache: 103 GFLOP/s rolled, 115 unrolled, on one x86-64 core).
#    define NUMERICAL_UNROLL_DEPTH _Pragma("GCC unroll 2")
// The AVX2 tile's depth loop unrolled by four, as OpenBLAS's Haswell kernel unrolls it (by eight): a product of order
// 1000 on one AVX2 core, 50.5 → 52.9 GFLOP/s, by two; by eight no better.
#    define NUMERICAL_UNROLL_DEPTH_AVX2 _Pragma("GCC unroll 4")
#  endif
#endif

namespace numerical::detail {

#ifdef NUMERICAL_X86_COMPILED
  //! \brief What `cpuid` and `xgetbv` report of the wide vector units.
  struct x86_features
  {
    bool avx2_fma {false}; //!< AVX2 and FMA, their 256-bit registers saved by the system.
    bool avx512f  {false}; //!< AVX-512 Foundation, its 512-bit and mask registers saved by the system.
  };

  //! \brief `cpuid` leaf \p leaf, sub-leaf \p sub: EAX, EBX, ECX, EDX.
  [[nodiscard]] inline std::array<std::uint32_t, 4> cpuid(std::uint32_t leaf,
                                                          std::uint32_t sub) noexcept
  {
    std::array<std::uint32_t, 4> r {};
#  if defined(_MSC_VER) && !defined(__clang__)
    std::array<int, 4> regs        {};
    __cpuidex(regs.data(), static_cast<int>(leaf), static_cast<int>(sub));
    for (std::size_t i {0}; i < 4; ++i) {
      r[i] = static_cast<std::uint32_t>(regs[i]);
    }
#  else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#  endif
    return r;
  }

  //! \brief The register state the system saves (`xgetbv(0)`); 0 when it does not enable `xgetbv`.
  [[nodiscard]] NUMERICAL_TARGET("xsave") inline std::uint64_t saved_state(std::uint32_t leaf1_ecx) noexcept
  {
    if ((leaf1_ecx & (1U << 27U)) == 0U) { // OSXSAVE
      return 0;
    }
    return _xgetbv(0);
  }

  //! \brief The wide units this processor has and this system saves, from the raw `cpuid` words.
  [[nodiscard]] constexpr x86_features decode_features(std::uint32_t leaf1_ecx,
                                                       std::uint32_t leaf7_ebx,
                                                       std::uint64_t state) noexcept
  {
    const bool avx     {(leaf1_ecx & (1U << 28U)) != 0U};
    const bool ymm     {avx && (state & 0x6U) == 0x6U};   // the AVX instructions, and SSE and AVX state
    const bool zmm     {avx && (state & 0xe6U) == 0xe6U}; // and the opmask and both halves of the 512-bit registers
    const bool fma     {(leaf1_ecx & (1U << 12U)) != 0U};
    const bool avx2    {(leaf7_ebx & (1U << 5U)) != 0U};
    const bool avx512f {(leaf7_ebx & (1U << 16U)) != 0U};
    return {.avx2_fma = ymm && avx2 && fma, .avx512f = zmm && avx512f};
  }

  //! \brief This processor's wide units, read once.
  [[nodiscard]] inline x86_features cpu_features() noexcept
  {
    static const x86_features features {[] {
                                          const std::array<std::uint32_t, 4> leaf1 {cpuid(1, 0)};
                                          // Leaf 7 only where the processor has it: past its last leaf, cpuid repeats
                                          // another leaf's words.
                                          const std::array<std::uint32_t, 4> leaf7 {cpuid(0, 0)[0] >= 7U ? cpuid(7, 0)
                                                                                    : std::array<std::uint32_t, 4> {}};
                                          return decode_features(leaf1[2], leaf7[1], saved_state(leaf1[2]));
                                        }()};
    return features;
  }

  //! \brief A 512-bit register as an element of `std::array` (a vector type as a template argument loses its
  //!        alignment attribute).
  struct zmm_register
  {
    __m512d v; //!< The register.
  };

  //! \brief A 256-bit register as an element of `std::array`.
  struct ymm_register
  {
    __m256d v; //!< The register.
  };

  //! \brief The rows of C one AVX-512 tile covers.
  inline constexpr std::size_t avx512_mr {12};

  //! \brief The columns of C one AVX-512 tile covers: two registers of eight `double`.
  inline constexpr std::size_t avx512_nr {16};

  //! \brief The rows of C one AVX2 tile covers.
  inline constexpr std::size_t avx2_mr {6};

  //! \brief The columns of C one AVX2 tile covers: two registers of four `double`.
  inline constexpr std::size_t avx2_nr {8};

  /*!
   * \brief `C += Aₚ·Bₚ` for one whole tile of C (\ref avx512_mr × \ref avx512_nr) over a
   *        panel of depth \p kc: 24 accumulators of eight `double`, each step one fused multiply-add per
   *        accumulator.
   */
  NUMERICAL_TARGET("avx512f") inline void avx512_full_tile_f64(std::size_t   kc,
                                                               const double* ap,
                                                               const double* bp,
                                                               std::size_t   ldb,
                                                               double*       tile,
                                                               std::size_t   ld) noexcept
  {
    std::array<zmm_register, avx512_mr * 2> acc {};
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < avx512_mr; ++i) {
      acc[2 * i].v       = _mm512_loadu_pd(tile + (i * ld));
      acc[(2 * i) + 1].v = _mm512_loadu_pd(tile + (i * ld) + 8);
    }
    NUMERICAL_UNROLL_DEPTH
    for (std::size_t p {0}; p < kc; ++p) {
      const __m512d b0 {_mm512_loadu_pd(bp + (p * ldb))};
      const __m512d b1 {_mm512_loadu_pd(bp + (p * ldb) + 8)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < avx512_mr; ++i) {
        const __m512d a {_mm512_set1_pd(ap[(p * avx512_mr) + i])};
        acc[2 * i].v       = _mm512_fmadd_pd(a, b0, acc[2 * i].v);
        acc[(2 * i) + 1].v = _mm512_fmadd_pd(a, b1, acc[(2 * i) + 1].v);
      }
    }
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < avx512_mr; ++i) {
      _mm512_storeu_pd(tile + (i * ld), acc[2 * i].v);
      _mm512_storeu_pd(tile + (i * ld) + 8, acc[(2 * i) + 1].v);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

  //! \brief \ref avx512_full_tile_f64 for a tile of \p rows × \p cols (at most \ref avx512_mr × \ref avx512_nr), a partial
  //!        one through a whole one whose outside is zero (\ref through_edge).
  inline void avx512_tile_f64(std::size_t   kc,
                              const double* ap,
                              const double* bp,
                              std::size_t   ldb,
                              double*       c,
                              std::size_t   ldc,
                              std::size_t   rows,
                              std::size_t   cols) noexcept
  {
    through_edge<avx512_mr, avx512_nr>(c, ldc, rows, cols, [&](double* tile, std::size_t ld) noexcept { avx512_full_tile_f64(kc, ap, bp, ldb, tile, ld); });
  }

  //! \brief How far ahead, in `double`s (512 bytes), the AVX2 tile fetches its slivers of A and B.
  inline constexpr std::size_t avx2_prefetch_ahead {64};

  /*!
   * \brief `C += Aₚ·Bₚ` for one whole tile of C (\ref avx2_mr × \ref avx2_nr) over a panel of
   *        depth \p kc: 12 accumulators of four `double`, each step one fused multiply-add per accumulator.
   */
  NUMERICAL_TARGET("avx2,fma") inline void avx2_full_tile_f64(std::size_t   kc,
                                                              const double* ap,
                                                              const double* bp,
                                                              std::size_t   ldb,
                                                              double*       tile,
                                                              std::size_t   ld) noexcept
  {
    std::array<ymm_register, avx2_mr * 2> acc {};
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < avx2_mr; ++i) {
      acc[2 * i].v       = _mm256_loadu_pd(tile + (i * ld));
      acc[(2 * i) + 1].v = _mm256_loadu_pd(tile + (i * ld) + 4);
    }
    NUMERICAL_UNROLL_DEPTH_AVX2
    for (std::size_t p {0}; p < kc; ++p) {
      // A and B a few steps ahead, into the L1 cache: the hardware's streams start a step late at each sliver (as
      // OpenBLAS's Haswell kernel fetches both 512 bytes ahead; with the unrolling, order 1000 52.9 → 53.9 GFLOP/s).
      _mm_prefetch(reinterpret_cast<const char*>(bp + (p * ldb) + avx2_prefetch_ahead), _MM_HINT_T0);     // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the intrinsic takes a char pointer
      _mm_prefetch(reinterpret_cast<const char*>(ap + (p * avx2_mr) + avx2_prefetch_ahead), _MM_HINT_T0); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
      const __m256d b0 {_mm256_loadu_pd(bp + (p * ldb))};
      const __m256d b1 {_mm256_loadu_pd(bp + (p * ldb) + 4)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < avx2_mr; ++i) {
        const __m256d a {_mm256_broadcast_sd(ap + (p * avx2_mr) + i)};
        acc[2 * i].v       = _mm256_fmadd_pd(a, b0, acc[2 * i].v);
        acc[(2 * i) + 1].v = _mm256_fmadd_pd(a, b1, acc[(2 * i) + 1].v);
      }
    }
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < avx2_mr; ++i) {
      _mm256_storeu_pd(tile + (i * ld), acc[2 * i].v);
      _mm256_storeu_pd(tile + (i * ld) + 4, acc[(2 * i) + 1].v);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

  //! \brief \ref avx2_full_tile_f64 for a tile of \p rows × \p cols (at most \ref avx2_mr × \ref avx2_nr), a partial
  //!        one through a whole one whose outside is zero (\ref through_edge).
  inline void avx2_tile_f64(std::size_t   kc,
                            const double* ap,
                            const double* bp,
                            std::size_t   ldb,
                            double*       c,
                            std::size_t   ldc,
                            std::size_t   rows,
                            std::size_t   cols) noexcept
  {
    through_edge<avx2_mr, avx2_nr>(c, ldc, rows, cols, [&](double* tile, std::size_t ld) noexcept { avx2_full_tile_f64(kc, ap, bp, ldb, tile, ld); });
  }

  //! \brief A 512-bit register of `float` as an element of `std::array`.
  struct zmm_register_f32
  {
    __m512 v; //!< The register.
  };

  //! \brief A 256-bit register of `float` as an element of `std::array`.
  struct ymm_register_f32
  {
    __m256 v; //!< The register.
  };

  //! \brief The columns of C one AVX-512 `float` tile covers: two registers of sixteen `float`.
  inline constexpr std::size_t avx512_nr_f32 {32};

  //! \brief The columns of C one AVX2 `float` tile covers: two registers of eight `float`.
  inline constexpr std::size_t avx2_nr_f32 {16};

  //! \brief \ref avx512_full_tile_f64 for `float`: \ref avx512_mr × \ref avx512_nr_f32, 24 accumulators of sixteen.
  NUMERICAL_TARGET("avx512f") inline void avx512_full_tile_f32(std::size_t  kc,
                                                               const float* ap,
                                                               const float* bp,
                                                               std::size_t  ldb,
                                                               float*       tile,
                                                               std::size_t  ld) noexcept
  {
    std::array<zmm_register_f32, avx512_mr * 2> acc {};
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < avx512_mr; ++i) {
      acc[2 * i].v       = _mm512_loadu_ps(tile + (i * ld));
      acc[(2 * i) + 1].v = _mm512_loadu_ps(tile + (i * ld) + 16);
    }
    NUMERICAL_UNROLL_DEPTH
    for (std::size_t p {0}; p < kc; ++p) {
      const __m512 b0 {_mm512_loadu_ps(bp + (p * ldb))};
      const __m512 b1 {_mm512_loadu_ps(bp + (p * ldb) + 16)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < avx512_mr; ++i) {
        const __m512 a {_mm512_set1_ps(ap[(p * avx512_mr) + i])};
        acc[2 * i].v       = _mm512_fmadd_ps(a, b0, acc[2 * i].v);
        acc[(2 * i) + 1].v = _mm512_fmadd_ps(a, b1, acc[(2 * i) + 1].v);
      }
    }
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < avx512_mr; ++i) {
      _mm512_storeu_ps(tile + (i * ld), acc[2 * i].v);
      _mm512_storeu_ps(tile + (i * ld) + 16, acc[(2 * i) + 1].v);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

  //! \brief \ref avx512_full_tile_f32 for a tile of \p rows × \p cols, a partial one through a whole one (\ref through_edge).
  inline void avx512_tile_f32(std::size_t  kc,
                              const float* ap,
                              const float* bp,
                              std::size_t  ldb,
                              float*       c,
                              std::size_t  ldc,
                              std::size_t  rows,
                              std::size_t  cols) noexcept
  {
    through_edge<avx512_mr, avx512_nr_f32>(c, ldc, rows, cols, [&](float* tile, std::size_t ld) noexcept { avx512_full_tile_f32(kc, ap, bp, ldb, tile, ld); });
  }

  //! \brief \ref avx2_full_tile_f64 for `float`: \ref avx2_mr × \ref avx2_nr_f32, 12 accumulators of eight.
  NUMERICAL_TARGET("avx2,fma") inline void avx2_full_tile_f32(std::size_t  kc,
                                                              const float* ap,
                                                              const float* bp,
                                                              std::size_t  ldb,
                                                              float*       tile,
                                                              std::size_t  ld) noexcept
  {
    std::array<ymm_register_f32, avx2_mr * 2> acc {};
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < avx2_mr; ++i) {
      acc[2 * i].v       = _mm256_loadu_ps(tile + (i * ld));
      acc[(2 * i) + 1].v = _mm256_loadu_ps(tile + (i * ld) + 8);
    }
    NUMERICAL_UNROLL_DEPTH_AVX2
    for (std::size_t p {0}; p < kc; ++p) {
      _mm_prefetch(reinterpret_cast<const char*>(bp + (p * ldb) + (2 * avx2_prefetch_ahead)), _MM_HINT_T0);     // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the intrinsic takes a char pointer
      _mm_prefetch(reinterpret_cast<const char*>(ap + (p * avx2_mr) + (2 * avx2_prefetch_ahead)), _MM_HINT_T0); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
      const __m256 b0 {_mm256_loadu_ps(bp + (p * ldb))};
      const __m256 b1 {_mm256_loadu_ps(bp + (p * ldb) + 8)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < avx2_mr; ++i) {
        const __m256 a {_mm256_broadcast_ss(ap + (p * avx2_mr) + i)};
        acc[2 * i].v       = _mm256_fmadd_ps(a, b0, acc[2 * i].v);
        acc[(2 * i) + 1].v = _mm256_fmadd_ps(a, b1, acc[(2 * i) + 1].v);
      }
    }
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < avx2_mr; ++i) {
      _mm256_storeu_ps(tile + (i * ld), acc[2 * i].v);
      _mm256_storeu_ps(tile + (i * ld) + 8, acc[(2 * i) + 1].v);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

  //! \brief \ref avx2_full_tile_f32 for a tile of \p rows × \p cols, a partial one through a whole one (\ref through_edge).
  inline void avx2_tile_f32(std::size_t  kc,
                            const float* ap,
                            const float* bp,
                            std::size_t  ldb,
                            float*       c,
                            std::size_t  ldc,
                            std::size_t  rows,
                            std::size_t  cols) noexcept
  {
    through_edge<avx2_mr, avx2_nr_f32>(c, ldc, rows, cols, [&](float* tile, std::size_t ld) noexcept { avx2_full_tile_f32(kc, ap, bp, ldb, tile, ld); });
  }

  //! \brief The rows of C one AVX2 complex tile covers.
  inline constexpr std::size_t avx2_complex_mr {3};

  //! \brief The columns of C one AVX2 complex tile covers: two registers of two `std::complex<double>`.
  inline constexpr std::size_t avx2_complex_nr {4};

  /*!
   * \brief `C += Aₚ·Bₚ` for one whole tile of complex C (\ref avx2_complex_mr × \ref avx2_complex_nr) over a panel of
   *        depth \p kc, the operands read as their parts in order ([complex.numbers]): each row accumulates `Re a · b`
   *        and `Im a · b` apart, 12 accumulators of four `double`, each step one fused multiply-add per accumulator;
   *        their sum `Re a · b + i Im a · b` is formed once, by exchanging the second's parts and subtracting or
   *        adding them (`addsub`).
   */
  NUMERICAL_TARGET("avx2,fma") inline void avx2_full_tile_c64(std::size_t                 kc,
                                                              const std::complex<double>* ap,
                                                              const std::complex<double>* bp,
                                                              std::size_t                 ldb,
                                                              std::complex<double>*       tile,
                                                              std::size_t                 ld) noexcept
  {
    const auto* const                             a   {reinterpret_cast<const double*>(ap)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
    const auto* const                             b   {reinterpret_cast<const double*>(bp)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    auto* const                                   c   {reinterpret_cast<double*>(tile)};     // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    std::array<ymm_register, avx2_complex_mr * 4> acc {};
    NUMERICAL_UNROLL_DEPTH
    for (std::size_t p {0}; p < kc; ++p) {
      const __m256d b0 {_mm256_loadu_pd(b + (2 * p * ldb))};
      const __m256d b1 {_mm256_loadu_pd(b + (2 * p * ldb) + 4)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < avx2_complex_mr; ++i) {
        const __m256d re {_mm256_broadcast_sd(a + (2 * ((p * avx2_complex_mr) + i)))};
        const __m256d im {_mm256_broadcast_sd(a + (2 * ((p * avx2_complex_mr) + i)) + 1)};
        acc[4 * i].v       = _mm256_fmadd_pd(re, b0, acc[4 * i].v);
        acc[(4 * i) + 1].v = _mm256_fmadd_pd(re, b1, acc[(4 * i) + 1].v);
        acc[(4 * i) + 2].v = _mm256_fmadd_pd(im, b0, acc[(4 * i) + 2].v);
        acc[(4 * i) + 3].v = _mm256_fmadd_pd(im, b1, acc[(4 * i) + 3].v);
      }
    }
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < avx2_complex_mr; ++i) {
      double* const row {c + (2 * i * ld)};
      const __m256d lo  {_mm256_addsub_pd(acc[4 * i].v, _mm256_permute_pd(acc[(4 * i) + 2].v, 0b0101))};
      const __m256d hi  {_mm256_addsub_pd(acc[(4 * i) + 1].v, _mm256_permute_pd(acc[(4 * i) + 3].v, 0b0101))};
      _mm256_storeu_pd(row, _mm256_add_pd(_mm256_loadu_pd(row), lo));
      _mm256_storeu_pd(row + 4, _mm256_add_pd(_mm256_loadu_pd(row + 4), hi));
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

  //! \brief \ref avx2_full_tile_c64 for a tile of \p rows × \p cols (at most \ref avx2_complex_mr ×
  //!        \ref avx2_complex_nr), a partial one through a whole one whose outside is zero (\ref through_edge).
  inline void avx2_tile_c64(std::size_t                 kc,
                            const std::complex<double>* ap,
                            const std::complex<double>* bp,
                            std::size_t                 ldb,
                            std::complex<double>*       c,
                            std::size_t                 ldc,
                            std::size_t                 rows,
                            std::size_t                 cols) noexcept
  {
    through_edge<avx2_complex_mr, avx2_complex_nr>(c, ldc, rows, cols, [&](std::complex<double>* tile, std::size_t ld) noexcept {
                                                     avx2_full_tile_c64(kc, ap, bp, ldb, tile, ld);
                                                   });
  }

  //! \brief The rows of C one AVX-512 complex tile covers.
  inline constexpr std::size_t avx512_complex_mr {6};

  //! \brief The columns of C one AVX-512 complex tile covers: two registers of four `std::complex<double>`.
  inline constexpr std::size_t avx512_complex_nr {8};

  /*!
   * \brief \ref avx2_full_tile_c64 on the AVX-512 units: a tile of \ref avx512_complex_mr × \ref avx512_complex_nr,
   *        24 accumulators of eight `double`; with no `addsub` there, the second's exchanged parts are added times
   *        `(−1, 1, …)` by one fused multiply-add, as exact.
   */
  NUMERICAL_TARGET("avx512f") inline void avx512_full_tile_c64(std::size_t                 kc,
                                                               const std::complex<double>* ap,
                                                               const std::complex<double>* bp,
                                                               std::size_t                 ldb,
                                                               std::complex<double>*       tile,
                                                               std::size_t                 ld) noexcept
  {
    const auto* const                               a   {reinterpret_cast<const double*>(ap)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
    const auto* const                               b   {reinterpret_cast<const double*>(bp)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    auto* const                                     c   {reinterpret_cast<double*>(tile)};     // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    std::array<zmm_register, avx512_complex_mr * 4> acc {};
    NUMERICAL_UNROLL_DEPTH
    for (std::size_t p {0}; p < kc; ++p) {
      const __m512d b0 {_mm512_loadu_pd(b + (2 * p * ldb))};
      const __m512d b1 {_mm512_loadu_pd(b + (2 * p * ldb) + 8)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < avx512_complex_mr; ++i) {
        const __m512d re {_mm512_set1_pd(a[2 * ((p * avx512_complex_mr) + i)])};
        const __m512d im {_mm512_set1_pd(a[(2 * ((p * avx512_complex_mr) + i)) + 1])};
        acc[4 * i].v       = _mm512_fmadd_pd(re, b0, acc[4 * i].v);
        acc[(4 * i) + 1].v = _mm512_fmadd_pd(re, b1, acc[(4 * i) + 1].v);
        acc[(4 * i) + 2].v = _mm512_fmadd_pd(im, b0, acc[(4 * i) + 2].v);
        acc[(4 * i) + 3].v = _mm512_fmadd_pd(im, b1, acc[(4 * i) + 3].v);
      }
    }
    const __m512d signs {_mm512_set_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)}; // −1 on the real parts
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < avx512_complex_mr; ++i) {
      double* const row {c + (2 * i * ld)};
      const __m512d lo  {_mm512_fmadd_pd(_mm512_permute_pd(acc[(4 * i) + 2].v, 0x55), signs, acc[4 * i].v)};
      const __m512d hi  {_mm512_fmadd_pd(_mm512_permute_pd(acc[(4 * i) + 3].v, 0x55), signs, acc[(4 * i) + 1].v)};
      _mm512_storeu_pd(row, _mm512_add_pd(_mm512_loadu_pd(row), lo));
      _mm512_storeu_pd(row + 8, _mm512_add_pd(_mm512_loadu_pd(row + 8), hi));
    }
  }

  //! \brief \ref avx512_full_tile_c64 for a tile of \p rows × \p cols, a partial one through a whole one (\ref through_edge).
  inline void avx512_tile_c64(std::size_t                 kc,
                              const std::complex<double>* ap,
                              const std::complex<double>* bp,
                              std::size_t                 ldb,
                              std::complex<double>*       c,
                              std::size_t                 ldc,
                              std::size_t                 rows,
                              std::size_t                 cols) noexcept
  {
    through_edge<avx512_complex_mr, avx512_complex_nr>(c, ldc, rows, cols, [&](std::complex<double>* tile, std::size_t ld) noexcept {
                                                         avx512_full_tile_c64(kc, ap, bp, ldb, tile, ld);
                                                       });
  }

  /*!
   * \brief `Σ row[i] · v[i]`, and `out[i] += row[i] · f`, for \p count elements in one pass over \p row (see
   *        \ref symmetric_row): four accumulators of eight `double`, each a chain of fused multiply-adds, summed
   *        in a fixed order, as is the remainder.
   */
  NUMERICAL_TARGET("avx512f") inline double avx512_symmetric_row(const double* row,
                                                                 const double* v,
                                                                 double        f,
                                                                 double*       out,
                                                                 std::size_t   count) noexcept
  {
    constexpr std::size_t           width {8};
    constexpr std::size_t           lanes {4};
    const __m512d                   splat {_mm512_set1_pd(f)};
    std::array<zmm_register, lanes> sum   {};
    const std::size_t               done  {count - (count % (width * lanes))};
    for (std::size_t i {0}; i < done; i += width * lanes) {
      NUMERICAL_UNROLL_TILE
      for (std::size_t l {0}; l < lanes; ++l) {
        const __m512d r {_mm512_loadu_pd(row + i + (l * width))};
        sum[l].v = _mm512_fmadd_pd(r, _mm512_loadu_pd(v + i + (l * width)), sum[l].v);
        _mm512_storeu_pd(out + i + (l * width), _mm512_fmadd_pd(r, splat, _mm512_loadu_pd(out + i + (l * width))));
      }
    }
    double total {0.0};
    for (std::size_t l {0}; l < lanes; ++l) {
      total += _mm512_reduce_add_pd(sum[l].v);
    }
    for (std::size_t i {done}; i < count; ++i) {
      total  += row[i] * v[i];
      out[i] += row[i] * f;
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return total;
  }

  //! \brief \ref avx512_symmetric_row with four accumulators of four `double` (AVX2 with FMA).
  NUMERICAL_TARGET("avx2,fma") inline double avx2_symmetric_row(const double* row,
                                                                const double* v,
                                                                double        f,
                                                                double*       out,
                                                                std::size_t   count) noexcept
  {
    constexpr std::size_t           width {4};
    constexpr std::size_t           lanes {4};
    const __m256d                   splat {_mm256_set1_pd(f)};
    std::array<ymm_register, lanes> sum   {};
    const std::size_t               done  {count - (count % (width * lanes))};
    for (std::size_t i {0}; i < done; i += width * lanes) {
      NUMERICAL_UNROLL_TILE
      for (std::size_t l {0}; l < lanes; ++l) {
        const __m256d r {_mm256_loadu_pd(row + i + (l * width))};
        sum[l].v = _mm256_fmadd_pd(r, _mm256_loadu_pd(v + i + (l * width)), sum[l].v);
        _mm256_storeu_pd(out + i + (l * width), _mm256_fmadd_pd(r, splat, _mm256_loadu_pd(out + i + (l * width))));
      }
    }
    double total {0.0};
    for (std::size_t l {0}; l < lanes; ++l) {
      alignas(32) std::array<double, width> part {}; // a register stored whole: some compilers store it aligned
      _mm256_storeu_pd(part.data(), sum[l].v);
      total += (part[0] + part[1]) + (part[2] + part[3]);
    }
    for (std::size_t i {done}; i < count; ++i) {
      total  += row[i] * v[i];
      out[i] += row[i] * f;
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return total;
  }

  /*!
   * \brief \ref symmetric_rows on AVX-512: the shared columns by eight, a fused multiply-add for each row's sum and
   *        each row's term of `out`, the rows in order.
   */
  NUMERICAL_TARGET("avx512f") inline std::array<double, symmetric_row_group> avx512_symmetric_rows(const double* rows,
                                                                                                   std::size_t   n,
                                                                                                   const double* v,
                                                                                                   double*       out,
                                                                                                   std::size_t   i) noexcept
  {
    constexpr std::size_t                         width {8};
    std::array<zmm_register, symmetric_row_group> sum   {};
    std::array<zmm_register, symmetric_row_group> splat {};
    for (std::size_t r {0}; r < symmetric_row_group; ++r) {
      splat[r].v = _mm512_set1_pd(v[i + r]);
    }
    const std::size_t done {i - (i % width)};
    for (std::size_t j {0}; j < done; j += width) {
      const __m512d vv {_mm512_loadu_pd(v + j)};
      __m512d       ov {_mm512_loadu_pd(out + j)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t r {0}; r < symmetric_row_group; ++r) {
        const __m512d rv {_mm512_loadu_pd(rows + (r * n) + j)};
        sum[r].v = _mm512_fmadd_pd(rv, vv, sum[r].v);
        ov       = _mm512_fmadd_pd(rv, splat[r].v, ov);
      }
      _mm512_storeu_pd(out + j, ov);
    }
    std::array<double, symmetric_row_group> sums {};
    for (std::size_t r {0}; r < symmetric_row_group; ++r) {
      sums[r] = _mm512_reduce_add_pd(sum[r].v);
    }
    _mm256_zeroupper(); // the upper halves cleared: the tail may be SSE alone
    return symmetric_rows_tail(rows, n, v, out, i, done, sums);
  }

  //! \brief \ref avx512_symmetric_rows by four columns (AVX2 with FMA).
  NUMERICAL_TARGET("avx2,fma") inline std::array<double, symmetric_row_group> avx2_symmetric_rows(const double* rows,
                                                                                                  std::size_t   n,
                                                                                                  const double* v,
                                                                                                  double*       out,
                                                                                                  std::size_t   i) noexcept
  {
    constexpr std::size_t                         width {4};
    std::array<ymm_register, symmetric_row_group> sum   {};
    std::array<ymm_register, symmetric_row_group> splat {};
    for (std::size_t r {0}; r < symmetric_row_group; ++r) {
      splat[r].v = _mm256_set1_pd(v[i + r]);
    }
    const std::size_t done {i - (i % width)};
    for (std::size_t j {0}; j < done; j += width) {
      const __m256d vv {_mm256_loadu_pd(v + j)};
      __m256d       ov {_mm256_loadu_pd(out + j)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t r {0}; r < symmetric_row_group; ++r) {
        const __m256d rv {_mm256_loadu_pd(rows + (r * n) + j)};
        sum[r].v = _mm256_fmadd_pd(rv, vv, sum[r].v);
        ov       = _mm256_fmadd_pd(rv, splat[r].v, ov);
      }
      _mm256_storeu_pd(out + j, ov);
    }
    std::array<double, symmetric_row_group> sums {};
    for (std::size_t r {0}; r < symmetric_row_group; ++r) {
      alignas(32) std::array<double, width> part {}; // a register stored whole: some compilers store it aligned
      _mm256_storeu_pd(part.data(), sum[r].v);
      sums[r] = (part[0] + part[1]) + (part[2] + part[3]);
    }
    _mm256_zeroupper(); // the upper halves cleared: the tail may be SSE alone
    return symmetric_rows_tail(rows, n, v, out, i, done, sums);
  }

  //! \brief `y[i] += f · x[i]` for \p count elements, each one fused multiply-add (a subtraction passes `−f`,
  //!        whose negation is exact).
  NUMERICAL_TARGET("avx512f") inline void avx512_scaled_add(double*       y,
                                                            const double* x,
                                                            double        f,
                                                            std::size_t   count) noexcept
  {
    const __m512d     splat {_mm512_set1_pd(f)};
    const std::size_t done  {count - (count % 8)};
    for (std::size_t i {0}; i < done; i += 8) {
      _mm512_storeu_pd(y + i, _mm512_fmadd_pd(splat, _mm512_loadu_pd(x + i), _mm512_loadu_pd(y + i)));
    }
    for (std::size_t i {done}; i < count; ++i) {
      y[i] = std::fma(f, x[i], y[i]);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  /*!
   * \brief `y[i] -= Σₜ f[t] · x[(t · ldx) + i]` for \p count elements, the terms in ascending order, each one fused
   *        multiply-add (as \ref avx512_scaled_add with `−f[t]`, term after term): four registers of `y` held through
   *        the terms, then one, then the remainder.
   */
  NUMERICAL_TARGET("avx512f") inline void avx512_subtract_products(double*       y,
                                                                   const double* f,
                                                                   const double* x,
                                                                   std::size_t   ldx,
                                                                   std::size_t   terms,
                                                                   std::size_t   count) noexcept
  {
    std::size_t i {0};
    for (; i + 32 <= count; i += 32) {
      __m512d y0 {_mm512_loadu_pd(y + i)};
      __m512d y1 {_mm512_loadu_pd(y + i + 8)};
      __m512d y2 {_mm512_loadu_pd(y + i + 16)};
      __m512d y3 {_mm512_loadu_pd(y + i + 24)};
      for (std::size_t t {0}; t < terms; ++t) {
        const __m512d       minus {_mm512_set1_pd(-f[t])};
        const double* const row   {x + (t * ldx) + i};
        y0 = _mm512_fmadd_pd(minus, _mm512_loadu_pd(row), y0);
        y1 = _mm512_fmadd_pd(minus, _mm512_loadu_pd(row + 8), y1);
        y2 = _mm512_fmadd_pd(minus, _mm512_loadu_pd(row + 16), y2);
        y3 = _mm512_fmadd_pd(minus, _mm512_loadu_pd(row + 24), y3);
      }
      _mm512_storeu_pd(y + i, y0);
      _mm512_storeu_pd(y + i + 8, y1);
      _mm512_storeu_pd(y + i + 16, y2);
      _mm512_storeu_pd(y + i + 24, y3);
    }
    for (; i + 8 <= count; i += 8) {
      __m512d sum {_mm512_loadu_pd(y + i)};
      for (std::size_t t {0}; t < terms; ++t) {
        sum = _mm512_fmadd_pd(_mm512_set1_pd(-f[t]), _mm512_loadu_pd(x + (t * ldx) + i), sum);
      }
      _mm512_storeu_pd(y + i, sum);
    }
    for (; i < count; ++i) {
      for (std::size_t t {0}; t < terms; ++t) {
        y[i] = std::fma(-f[t], x[(t * ldx) + i], y[i]);
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  /*!
   * \brief \ref avx512_subtract_products for eight rows of `y` at once, \p ldy apart, each with its own factors (\p ldf
   *        apart): a register of eight columns per row through the terms, the last columns masked. Each element is
   *        the same chain of fused multiply-adds, so the same value.
   */
  NUMERICAL_TARGET("avx512f") inline void avx512_subtract_rows(double*       y,
                                                               std::size_t   ldy,
                                                               const double* f,
                                                               std::size_t   ldf,
                                                               const double* x,
                                                               std::size_t   ldx,
                                                               std::size_t   terms,
                                                               std::size_t   count) noexcept
  {
    struct lane // a register in a struct: an array of the register type itself loses its alignment attribute
    {
      __m512d value;
    };
    for (std::size_t i {0}; i < count; i += 8) {
      const auto               mask {static_cast<__mmask8>(count - i >= 8 ? 0xFFU : (1U << (count - i)) - 1U)};
      std::array<lane, 8>      held {};
      for (std::size_t r {0}; r < 8; ++r) {
        held[r].value = _mm512_maskz_loadu_pd(mask, y + (r * ldy) + i);
      }
      for (std::size_t t {0}; t < terms; ++t) {
        const __m512d row {_mm512_maskz_loadu_pd(mask, x + (t * ldx) + i)};
        for (std::size_t r {0}; r < 8; ++r) {
          held[r].value = _mm512_fmadd_pd(_mm512_set1_pd(-f[(r * ldf) + t]), row, held[r].value);
        }
      }
      for (std::size_t r {0}; r < 8; ++r) {
        _mm512_mask_storeu_pd(y + (r * ldy) + i, mask, held[r].value);
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief \ref avx512_subtract_rows on AVX2 with FMA: a register of four columns per row, then a column at a time.
  NUMERICAL_TARGET("avx2,fma") inline void avx2_subtract_rows(double*       y,
                                                              std::size_t   ldy,
                                                              const double* f,
                                                              std::size_t   ldf,
                                                              const double* x,
                                                              std::size_t   ldx,
                                                              std::size_t   terms,
                                                              std::size_t   count) noexcept
  {
    struct lane // as avx512_subtract_rows's
    {
      __m256d value;
    };
    std::size_t i {0};
    for (; i + 4 <= count; i += 4) {
      std::array<lane, 8> held {};
      for (std::size_t r {0}; r < 8; ++r) {
        held[r].value = _mm256_loadu_pd(y + (r * ldy) + i);
      }
      for (std::size_t t {0}; t < terms; ++t) {
        const __m256d row {_mm256_loadu_pd(x + (t * ldx) + i)};
        for (std::size_t r {0}; r < 8; ++r) {
          held[r].value = _mm256_fmadd_pd(_mm256_set1_pd(-f[(r * ldf) + t]), row, held[r].value);
        }
      }
      for (std::size_t r {0}; r < 8; ++r) {
        _mm256_storeu_pd(y + (r * ldy) + i, held[r].value);
      }
    }
    for (; i < count; ++i) {
      std::array<double, 8> held {};
      for (std::size_t r {0}; r < 8; ++r) {
        held[r] = y[(r * ldy) + i];
      }
      for (std::size_t t {0}; t < terms; ++t) {
        const double value {x[(t * ldx) + i]};
        for (std::size_t r {0}; r < 8; ++r) {
          held[r] = std::fma(-f[(r * ldf) + t], value, held[r]);
        }
      }
      for (std::size_t r {0}; r < 8; ++r) {
        y[(r * ldy) + i] = held[r];
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief \ref avx512_subtract_products on AVX2 with FMA: four registers of four `double`, then one.
  NUMERICAL_TARGET("avx2,fma") inline void avx2_subtract_products(double*       y,
                                                                  const double* f,
                                                                  const double* x,
                                                                  std::size_t   ldx,
                                                                  std::size_t   terms,
                                                                  std::size_t   count) noexcept
  {
    std::size_t i {0};
    for (; i + 16 <= count; i += 16) {
      __m256d y0 {_mm256_loadu_pd(y + i)};
      __m256d y1 {_mm256_loadu_pd(y + i + 4)};
      __m256d y2 {_mm256_loadu_pd(y + i + 8)};
      __m256d y3 {_mm256_loadu_pd(y + i + 12)};
      for (std::size_t t {0}; t < terms; ++t) {
        const __m256d       minus {_mm256_set1_pd(-f[t])};
        const double* const row   {x + (t * ldx) + i};
        y0 = _mm256_fmadd_pd(minus, _mm256_loadu_pd(row), y0);
        y1 = _mm256_fmadd_pd(minus, _mm256_loadu_pd(row + 4), y1);
        y2 = _mm256_fmadd_pd(minus, _mm256_loadu_pd(row + 8), y2);
        y3 = _mm256_fmadd_pd(minus, _mm256_loadu_pd(row + 12), y3);
      }
      _mm256_storeu_pd(y + i, y0);
      _mm256_storeu_pd(y + i + 4, y1);
      _mm256_storeu_pd(y + i + 8, y2);
      _mm256_storeu_pd(y + i + 12, y3);
    }
    for (; i + 4 <= count; i += 4) {
      __m256d sum {_mm256_loadu_pd(y + i)};
      for (std::size_t t {0}; t < terms; ++t) {
        sum = _mm256_fmadd_pd(_mm256_set1_pd(-f[t]), _mm256_loadu_pd(x + (t * ldx) + i), sum);
      }
      _mm256_storeu_pd(y + i, sum);
    }
    for (; i < count; ++i) {
      for (std::size_t t {0}; t < terms; ++t) {
        y[i] = std::fma(-f[t], x[(t * ldx) + i], y[i]);
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  /*!
   * \brief `y[i] -= Σₜ f[t] · x[(t · ldx) + i]` for \p count complex numbers (each its real then imaginary part), the
   *        terms in ascending order: per term, `y −= Re f · x` then `y −= (−Im f, Im f) · x'`, `x'` each number's
   *        parts swapped, each one fused multiply-add per part. Four registers of `y` held through the terms, then
   *        one, then the remainder, the same operations one number at a time.
   */
  NUMERICAL_TARGET("avx512f") inline void avx512_subtract_complex_products(double*       y,
                                                                           const double* f,
                                                                           const double* x,
                                                                           std::size_t   ldx,
                                                                           std::size_t   terms,
                                                                           std::size_t   count) noexcept
  {
    const __m512d sign {_mm512_set_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)}; // (−, +) for each number
    std::size_t   i    {0};
    for (; i + 16 <= count; i += 16) {
      std::array<zmm_register, 4> acc {};
      NUMERICAL_UNROLL_TILE
      for (std::size_t r {0}; r < 4; ++r) {
        acc[r].v = _mm512_loadu_pd(y + (2 * i) + (8 * r));
      }
      for (std::size_t t {0}; t < terms; ++t) {
        const __m512d       real   {_mm512_set1_pd(f[2 * t])};
        const __m512d       imag   {_mm512_mul_pd(_mm512_set1_pd(f[(2 * t) + 1]), sign)};
        const double* const row    {x + (2 * ((t * ldx) + i))};
        NUMERICAL_UNROLL_TILE
        for (std::size_t r {0}; r < 4; ++r) {
          const __m512d value {_mm512_loadu_pd(row + (8 * r))};
          acc[r].v = _mm512_fnmadd_pd(real, value, acc[r].v);
          acc[r].v = _mm512_fnmadd_pd(imag, _mm512_permute_pd(value, 0x55), acc[r].v);
        }
      }
      NUMERICAL_UNROLL_TILE
      for (std::size_t r {0}; r < 4; ++r) {
        _mm512_storeu_pd(y + (2 * i) + (8 * r), acc[r].v);
      }
    }
    for (; i + 4 <= count; i += 4) {
      __m512d sum {_mm512_loadu_pd(y + (2 * i))};
      for (std::size_t t {0}; t < terms; ++t) {
        const __m512d value {_mm512_loadu_pd(x + (2 * ((t * ldx) + i)))};
        sum = _mm512_fnmadd_pd(_mm512_set1_pd(f[2 * t]), value, sum);
        sum = _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_set1_pd(f[(2 * t) + 1]), sign), _mm512_permute_pd(value, 0x55), sum);
      }
      _mm512_storeu_pd(y + (2 * i), sum);
    }
    for (; i < count; ++i) {
      double real {y[2 * i]};
      double imag {y[(2 * i) + 1]};
      for (std::size_t t {0}; t < terms; ++t) {
        const double xr {x[2 * ((t * ldx) + i)]};
        const double xi {x[(2 * ((t * ldx) + i)) + 1]};
        real = std::fma(-f[2 * t], xr, real);
        imag = std::fma(-f[2 * t], xi, imag);
        real = std::fma(f[(2 * t) + 1], xi, real);
        imag = std::fma(-f[(2 * t) + 1], xr, imag);
      }
      y[2 * i]       = real;
      y[(2 * i) + 1] = imag;
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief \ref avx512_subtract_complex_products on AVX2 with FMA: four registers of two complex numbers, then one.
  NUMERICAL_TARGET("avx2,fma") inline void avx2_subtract_complex_products(double*       y,
                                                                          const double* f,
                                                                          const double* x,
                                                                          std::size_t   ldx,
                                                                          std::size_t   terms,
                                                                          std::size_t   count) noexcept
  {
    const __m256d sign {_mm256_set_pd(1.0, -1.0, 1.0, -1.0)};
    std::size_t   i    {0};
    for (; i + 8 <= count; i += 8) {
      std::array<ymm_register, 4> acc {};
      NUMERICAL_UNROLL_TILE
      for (std::size_t r {0}; r < 4; ++r) {
        acc[r].v = _mm256_loadu_pd(y + (2 * i) + (4 * r));
      }
      for (std::size_t t {0}; t < terms; ++t) {
        const __m256d       real {_mm256_set1_pd(f[2 * t])};
        const __m256d       imag {_mm256_mul_pd(_mm256_set1_pd(f[(2 * t) + 1]), sign)};
        const double* const row  {x + (2 * ((t * ldx) + i))};
        NUMERICAL_UNROLL_TILE
        for (std::size_t r {0}; r < 4; ++r) {
          const __m256d value {_mm256_loadu_pd(row + (4 * r))};
          acc[r].v = _mm256_fnmadd_pd(real, value, acc[r].v);
          acc[r].v = _mm256_fnmadd_pd(imag, _mm256_permute_pd(value, 0x5), acc[r].v);
        }
      }
      NUMERICAL_UNROLL_TILE
      for (std::size_t r {0}; r < 4; ++r) {
        _mm256_storeu_pd(y + (2 * i) + (4 * r), acc[r].v);
      }
    }
    for (; i + 2 <= count; i += 2) {
      __m256d sum {_mm256_loadu_pd(y + (2 * i))};
      for (std::size_t t {0}; t < terms; ++t) {
        const __m256d value {_mm256_loadu_pd(x + (2 * ((t * ldx) + i)))};
        sum = _mm256_fnmadd_pd(_mm256_set1_pd(f[2 * t]), value, sum);
        sum = _mm256_fnmadd_pd(_mm256_mul_pd(_mm256_set1_pd(f[(2 * t) + 1]), sign), _mm256_permute_pd(value, 0x5), sum);
      }
      _mm256_storeu_pd(y + (2 * i), sum);
    }
    for (; i < count; ++i) {
      double real {y[2 * i]};
      double imag {y[(2 * i) + 1]};
      for (std::size_t t {0}; t < terms; ++t) {
        const double xr {x[2 * ((t * ldx) + i)]};
        const double xi {x[(2 * ((t * ldx) + i)) + 1]};
        real = std::fma(-f[2 * t], xr, real);
        imag = std::fma(-f[2 * t], xi, imag);
        real = std::fma(f[(2 * t) + 1], xi, real);
        imag = std::fma(-f[(2 * t) + 1], xr, imag);
      }
      y[2 * i]       = real;
      y[(2 * i) + 1] = imag;
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  /*!
   * \brief A complex row of a Hermitian product in one pass (as \ref avx512_symmetric_row for a real one):
   *        `out[j] += conj(row[j]) · (vr + i vi)` for \p count numbers, each part two fused multiply-adds, and the
   *        return `Σ row[j] · v[j]`, its products' parts summed in two registers (`rr·vr, ri·vi` and `rr·vi, ri·vr`)
   *        then their lanes, and the remainder, in a fixed order. Numbers are their real then imaginary parts.
   */
  NUMERICAL_TARGET("avx512f") inline std::complex<double> avx512_hermitian_row(const double* row,
                                                                               const double* v,
                                                                               double        vr,
                                                                               double        vi,
                                                                               double*       out,
                                                                               std::size_t   count) noexcept
  {
    const __m512d real   {_mm512_set_pd(-vr, vr, -vr, vr, -vr, vr, -vr, vr)}; // (vr, −vr) for each number
    const __m512d imag   {_mm512_set1_pd(vi)};
    __m512d       same   {_mm512_setzero_pd()};                               // rr·vr, ri·vi
    __m512d       across {_mm512_setzero_pd()};                               // rr·vi, ri·vr
    std::size_t   j      {0};
    for (; j + 4 <= count; j += 4) {
      const __m512d r {_mm512_loadu_pd(row + (2 * j))};
      const __m512d x {_mm512_loadu_pd(v + (2 * j))};
      const __m512d s {_mm512_permute_pd(r, 0x55)};
      __m512d       o {_mm512_loadu_pd(out + (2 * j))};
      o      = _mm512_fmadd_pd(real, r, o);
      o      = _mm512_fmadd_pd(imag, s, o);
      same   = _mm512_fmadd_pd(r, x, same);
      across = _mm512_fmadd_pd(s, x, across);
      _mm512_storeu_pd(out + (2 * j), o);
    }
    // Registers stored whole: GCC 14 stores the second aligned (in C++20, under the address sanitizer), so both are.
    alignas(64) std::array<double, 8> lanes_same   {};
    alignas(64) std::array<double, 8> lanes_across {};
    _mm512_storeu_pd(lanes_same.data(), same);
    _mm512_storeu_pd(lanes_across.data(), across);
    double re {0.0};
    double im {0.0};
    for (std::size_t l {0}; l < 8; l += 2) {
      re += lanes_same[l] - lanes_same[l + 1];
      im += lanes_across[l] + lanes_across[l + 1];
    }
    for (; j < count; ++j) {
      const double rr {row[2 * j]};
      const double ri {row[(2 * j) + 1]};
      out[2 * j]       = std::fma(vi, ri, std::fma(vr, rr, out[2 * j]));
      out[(2 * j) + 1] = std::fma(vi, rr, std::fma(-vr, ri, out[(2 * j) + 1]));
      re              += (rr * v[2 * j]) - (ri * v[(2 * j) + 1]);
      im              += (ri * v[2 * j]) + (rr * v[(2 * j) + 1]);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return std::complex<double> {re, im};
  }

  //! \brief \ref avx512_hermitian_row on AVX2 with FMA: two numbers a register.
  NUMERICAL_TARGET("avx2,fma") inline std::complex<double> avx2_hermitian_row(const double* row,
                                                                              const double* v,
                                                                              double        vr,
                                                                              double        vi,
                                                                              double*       out,
                                                                              std::size_t   count) noexcept
  {
    const __m256d real   {_mm256_set_pd(-vr, vr, -vr, vr)};
    const __m256d imag   {_mm256_set1_pd(vi)};
    __m256d       same   {_mm256_setzero_pd()};
    __m256d       across {_mm256_setzero_pd()};
    std::size_t   j      {0};
    for (; j + 2 <= count; j += 2) {
      const __m256d r {_mm256_loadu_pd(row + (2 * j))};
      const __m256d x {_mm256_loadu_pd(v + (2 * j))};
      const __m256d s {_mm256_permute_pd(r, 0x5)};
      __m256d       o {_mm256_loadu_pd(out + (2 * j))};
      o      = _mm256_fmadd_pd(real, r, o);
      o      = _mm256_fmadd_pd(imag, s, o);
      same   = _mm256_fmadd_pd(r, x, same);
      across = _mm256_fmadd_pd(s, x, across);
      _mm256_storeu_pd(out + (2 * j), o);
    }
    alignas(32) std::array<double, 4> lanes_same   {}; // registers stored whole, aligned (as above)
    alignas(32) std::array<double, 4> lanes_across {};
    _mm256_storeu_pd(lanes_same.data(), same);
    _mm256_storeu_pd(lanes_across.data(), across);
    double re {(lanes_same[0] - lanes_same[1]) + (lanes_same[2] - lanes_same[3])};
    double im {(lanes_across[0] + lanes_across[1]) + (lanes_across[2] + lanes_across[3])};
    for (; j < count; ++j) {
      const double rr {row[2 * j]};
      const double ri {row[(2 * j) + 1]};
      out[2 * j]       = std::fma(vi, ri, std::fma(vr, rr, out[2 * j]));
      out[(2 * j) + 1] = std::fma(vi, rr, std::fma(-vr, ri, out[(2 * j) + 1]));
      re              += (rr * v[2 * j]) - (ri * v[(2 * j) + 1]);
      im              += (ri * v[2 * j]) + (rr * v[(2 * j) + 1]);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return std::complex<double> {re, im};
  }

  //! \brief \ref avx512_scaled_add on AVX2 with FMA.
  NUMERICAL_TARGET("avx2,fma") inline void avx2_scaled_add(double*       y,
                                                           const double* x,
                                                           double        f,
                                                           std::size_t   count) noexcept
  {
    const __m256d     splat {_mm256_set1_pd(f)};
    const std::size_t done  {count - (count % 4)};
    for (std::size_t i {0}; i < done; i += 4) {
      _mm256_storeu_pd(y + i, _mm256_fmadd_pd(splat, _mm256_loadu_pd(x + i), _mm256_loadu_pd(y + i)));
    }
    for (std::size_t i {done}; i < count; ++i) {
      y[i] = std::fma(f, x[i], y[i]);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief `Σ x[i] · y[i]`: four accumulators of eight `double`, chains of fused multiply-adds summed in a fixed
  //!        order, then the remainder.
  NUMERICAL_TARGET("avx512f") inline double avx512_dot(const double* x,
                                                       const double* y,
                                                       std::size_t   count) noexcept
  {
    constexpr std::size_t           width {8};
    constexpr std::size_t           lanes {4};
    std::array<zmm_register, lanes> sum   {};
    const std::size_t               done  {count - (count % (width * lanes))};
    for (std::size_t i {0}; i < done; i += width * lanes) {
      NUMERICAL_UNROLL_TILE
      for (std::size_t l {0}; l < lanes; ++l) {
        sum[l].v = _mm512_fmadd_pd(_mm512_loadu_pd(x + i + (l * width)), _mm512_loadu_pd(y + i + (l * width)), sum[l].v);
      }
    }
    double total {0.0};
    for (std::size_t l {0}; l < lanes; ++l) {
      total += _mm512_reduce_add_pd(sum[l].v);
    }
    for (std::size_t i {done}; i < count; ++i) {
      total += x[i] * y[i];
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return total;
  }

  //! \brief \ref avx512_dot with four accumulators of four `double` (AVX2 with FMA).
  NUMERICAL_TARGET("avx2,fma") inline double avx2_dot(const double* x,
                                                      const double* y,
                                                      std::size_t   count) noexcept
  {
    constexpr std::size_t           width {4};
    constexpr std::size_t           lanes {4};
    std::array<ymm_register, lanes> sum   {};
    const std::size_t               done  {count - (count % (width * lanes))};
    for (std::size_t i {0}; i < done; i += width * lanes) {
      NUMERICAL_UNROLL_TILE
      for (std::size_t l {0}; l < lanes; ++l) {
        sum[l].v = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + (l * width)), _mm256_loadu_pd(y + i + (l * width)), sum[l].v);
      }
    }
    double total {0.0};
    for (std::size_t l {0}; l < lanes; ++l) {
      alignas(32) std::array<double, width> part {}; // a register stored whole: some compilers store it aligned
      _mm256_storeu_pd(part.data(), sum[l].v);
      total += (part[0] + part[1]) + (part[2] + part[3]);
    }
    for (std::size_t i {done}; i < count; ++i) {
      total += x[i] * y[i];
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return total;
  }

  /*!
   * \brief `Σ x[i] · y[i]` of complex numbers (AVX2 with FMA), their parts interleaved as the standard lays them out:
   *        four accumulators take `x · y` part by part (`re·re` in the even lanes, `im·im` in the odd), four `x` times
   *        `y` with its parts swapped (`re·im`, `im·re`), as chains of fused multiply-adds; their lanes, then the
   *        remainder, summed in a fixed order, the real part the first two sums' difference, the imaginary part the
   *        other two's sum (\ref dot_product's of complex numbers, fused).
   */
  NUMERICAL_TARGET("avx2,fma") inline std::complex<double> avx2_complex_dot(const std::complex<double>* x,
                                                                            const std::complex<double>* y,
                                                                            std::size_t                 count) noexcept
  {
    constexpr std::size_t           width {4};                                  // parts in a register: two complex numbers
    constexpr std::size_t           lanes {4};
    const auto* const               a     {reinterpret_cast<const double*>(x)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is an array of its two parts
    const auto* const               b     {reinterpret_cast<const double*>(y)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): as above
    const std::size_t               parts {2 * count};
    std::array<ymm_register, lanes> same  {};
    std::array<ymm_register, lanes> cross {};
    const std::size_t               done  {parts - (parts % (width * lanes))};
    for (std::size_t i {0}; i < done; i += width * lanes) {
      NUMERICAL_UNROLL_TILE
      for (std::size_t l {0}; l < lanes; ++l) {
        const __m256d av {_mm256_loadu_pd(a + i + (l * width))};
        const __m256d bv {_mm256_loadu_pd(b + i + (l * width))};
        same[l].v  = _mm256_fmadd_pd(av, bv, same[l].v);
        cross[l].v = _mm256_fmadd_pd(av, _mm256_permute_pd(bv, 0x5), cross[l].v); // each pair's parts swapped
      }
    }
    double rr {0.0};
    double ii {0.0};
    double ri {0.0};
    double ir {0.0};
    for (std::size_t l {0}; l < lanes; ++l) {
      alignas(32) std::array<double, width> s {}; // a register stored whole: some compilers store it aligned
      alignas(32) std::array<double, width> c {};
      _mm256_storeu_pd(s.data(), same[l].v);
      _mm256_storeu_pd(c.data(), cross[l].v);
      rr += s[0] + s[2];
      ii += s[1] + s[3];
      ri += c[0] + c[2];
      ir += c[1] + c[3];
    }
    for (std::size_t i {done}; i < parts; i += 2) {
      rr += a[i] * b[i];
      ii += a[i + 1] * b[i + 1];
      ri += a[i] * b[i + 1];
      ir += a[i + 1] * b[i];
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return {rr - ii, ri + ir};
  }

  //! \brief \ref avx2_complex_dot of `std::complex<float>`: four complex numbers a register.
  NUMERICAL_TARGET("avx2,fma") inline std::complex<float> avx2_complex_dot(const std::complex<float>* x,
                                                                           const std::complex<float>* y,
                                                                           std::size_t                count) noexcept
  {
    constexpr std::size_t               width {8};
    constexpr std::size_t               lanes {4};
    const auto* const                   a     {reinterpret_cast<const float*>(x)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is an array of its two parts
    const auto* const                   b     {reinterpret_cast<const float*>(y)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): as above
    const std::size_t                   parts {2 * count};
    std::array<ymm_register_f32, lanes> same  {};
    std::array<ymm_register_f32, lanes> cross {};
    const std::size_t                   done  {parts - (parts % (width * lanes))};
    for (std::size_t i {0}; i < done; i += width * lanes) {
      NUMERICAL_UNROLL_TILE
      for (std::size_t l {0}; l < lanes; ++l) {
        const __m256 av {_mm256_loadu_ps(a + i + (l * width))};
        const __m256 bv {_mm256_loadu_ps(b + i + (l * width))};
        same[l].v  = _mm256_fmadd_ps(av, bv, same[l].v);
        cross[l].v = _mm256_fmadd_ps(av, _mm256_permute_ps(bv, 0xB1), cross[l].v); // each pair's parts swapped
      }
    }
    float rr {0.0F};
    float ii {0.0F};
    float ri {0.0F};
    float ir {0.0F};
    for (std::size_t l {0}; l < lanes; ++l) {
      alignas(32) std::array<float, width> s {};
      alignas(32) std::array<float, width> c {};
      _mm256_storeu_ps(s.data(), same[l].v);
      _mm256_storeu_ps(c.data(), cross[l].v);
      rr += (s[0] + s[2]) + (s[4] + s[6]);
      ii += (s[1] + s[3]) + (s[5] + s[7]);
      ri += (c[0] + c[2]) + (c[4] + c[6]);
      ir += (c[1] + c[3]) + (c[5] + c[7]);
    }
    for (std::size_t i {done}; i < parts; i += 2) {
      rr += a[i] * b[i];
      ii += a[i + 1] * b[i + 1];
      ri += a[i] * b[i + 1];
      ir += a[i + 1] * b[i];
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return {rr - ii, ri + ir};
  }

#endif
} // namespace numerical::detail

#endif // NUMERICAL_X86_HPP
