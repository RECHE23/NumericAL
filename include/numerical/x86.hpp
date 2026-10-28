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

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

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

  /*!
   * \brief `body(index)` in a function built for AVX2 and FMA, where `std::fma` is the instruction: code outside such
   *        functions targets the base x86-64, on which `std::fma` calls the C library's, a call for each multiply-add,
   *        which also keeps the loop around it from being vectorised. \p body is inlined into it
   *        (`NUMERICAL_INLINE_LAMBDA`, and so must be what it calls). Only on a processor with both (\ref
   *        cpu_features), and for a body whose multiply-adds are all `std::fma`'s: the compiler may contract an
   *        `x·y + z` there into one. The same values: a fused multiply-add is rounded once, called or not.
   */
  template <typename Body>
  NUMERICAL_TARGET("avx2,fma") void with_fma_instruction(const Body& body,
                                                         std::size_t index)
  {
    body(index);
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

  //! \brief A 128-bit register as an element of `std::array`.
  struct xmm_register
  {
    __m128d v; //!< The register.
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
   *        accumulator. Of a tile short of rows, its first \p Rows alone (\ref through_rows), A's slivers still
   *        \ref avx512_mr wide: so each tile below.
   */
  template <std::size_t Rows = avx512_mr>
  NUMERICAL_TARGET("avx512f") inline void avx512_full_tile_f64(std::size_t   kc,
                                                               const double* ap,
                                                               const double* bp,
                                                               std::size_t   ldb,
                                                               double*       tile,
                                                               std::size_t   ld,
                                                               bool          fresh = false) noexcept
  {
    std::array<zmm_register, Rows * 2> acc {};
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < Rows; ++i) {
      acc[2 * i].v       = fresh ? _mm512_setzero_pd() : _mm512_loadu_pd(tile + (i * ld));
      acc[(2 * i) + 1].v = fresh ? _mm512_setzero_pd() : _mm512_loadu_pd(tile + (i * ld) + 8);
    }
    NUMERICAL_UNROLL_DEPTH
    for (std::size_t p {0}; p < kc; ++p) {
      const __m512d b0 {_mm512_loadu_pd(bp + (p * ldb))};
      const __m512d b1 {_mm512_loadu_pd(bp + (p * ldb) + 8)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < Rows; ++i) {
        const __m512d a {_mm512_set1_pd(ap[(p * avx512_mr) + i])};
        acc[2 * i].v       = _mm512_fmadd_pd(a, b0, acc[2 * i].v);
        acc[(2 * i) + 1].v = _mm512_fmadd_pd(a, b1, acc[(2 * i) + 1].v);
      }
    }
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < Rows; ++i) {
      _mm512_storeu_pd(tile + (i * ld), acc[2 * i].v);
      _mm512_storeu_pd(tile + (i * ld) + 8, acc[(2 * i) + 1].v);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

  //! \brief \ref avx512_full_tile_f64 for a tile of \p rows × \p cols (at most \ref avx512_mr × \ref avx512_nr), a partial
  //!        one by its rows alone or through a whole one whose outside is zero (\ref through_rows).
  inline void avx512_tile_f64(std::size_t   kc,
                              const double* ap,
                              const double* bp,
                              std::size_t   ldb,
                              double*       c,
                              std::size_t   ldc,
                              std::size_t   rows,
                              std::size_t   cols,
                              bool          fresh = false) noexcept
  {
    through_rows<avx512_mr, avx512_nr>(c, ldc, rows, cols, [&]<std::size_t Rows>(double* tile, std::size_t ld) noexcept { avx512_full_tile_f64<Rows>(kc, ap, bp, ldb, tile, ld, fresh); });
  }

  //! \brief How far ahead, in `double`s (512 bytes), the AVX2 tile fetches its slivers of A and B.
  inline constexpr std::size_t avx2_prefetch_ahead {64};

  /*!
   * \brief `C += Aₚ·Bₚ` for one whole tile of C (\ref avx2_mr × \ref avx2_nr) over a panel of
   *        depth \p kc: 12 accumulators of four `double`, each step one fused multiply-add per accumulator.
   */
  template <std::size_t Rows = avx2_mr>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_full_tile_f64(std::size_t   kc,
                                                              const double* ap,
                                                              const double* bp,
                                                              std::size_t   ldb,
                                                              double*       tile,
                                                              std::size_t   ld,
                                                              bool          fresh = false) noexcept
  {
    std::array<ymm_register, Rows * 2> acc {};
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < Rows; ++i) {
      acc[2 * i].v       = fresh ? _mm256_setzero_pd() : _mm256_loadu_pd(tile + (i * ld));
      acc[(2 * i) + 1].v = fresh ? _mm256_setzero_pd() : _mm256_loadu_pd(tile + (i * ld) + 4);
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
      for (std::size_t i {0}; i < Rows; ++i) {
        const __m256d a {_mm256_broadcast_sd(ap + (p * avx2_mr) + i)};
        acc[2 * i].v       = _mm256_fmadd_pd(a, b0, acc[2 * i].v);
        acc[(2 * i) + 1].v = _mm256_fmadd_pd(a, b1, acc[(2 * i) + 1].v);
      }
    }
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < Rows; ++i) {
      _mm256_storeu_pd(tile + (i * ld), acc[2 * i].v);
      _mm256_storeu_pd(tile + (i * ld) + 4, acc[(2 * i) + 1].v);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

  //! \brief \ref avx2_full_tile_f64 for a tile of \p rows × \p cols (at most \ref avx2_mr × \ref avx2_nr), a partial
  //!        one by its rows alone or through a whole one whose outside is zero (\ref through_rows).
  inline void avx2_tile_f64(std::size_t   kc,
                            const double* ap,
                            const double* bp,
                            std::size_t   ldb,
                            double*       c,
                            std::size_t   ldc,
                            std::size_t   rows,
                            std::size_t   cols,
                            bool          fresh = false) noexcept
  {
    through_rows<avx2_mr, avx2_nr>(c, ldc, rows, cols, [&]<std::size_t Rows>(double* tile, std::size_t ld) noexcept { avx2_full_tile_f64<Rows>(kc, ap, bp, ldb, tile, ld, fresh); });
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
  template <std::size_t Rows = avx512_mr>
  NUMERICAL_TARGET("avx512f") inline void avx512_full_tile_f32(std::size_t  kc,
                                                               const float* ap,
                                                               const float* bp,
                                                               std::size_t  ldb,
                                                               float*       tile,
                                                               std::size_t  ld,
                                                               bool         fresh = false) noexcept
  {
    std::array<zmm_register_f32, Rows * 2> acc {};
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < Rows; ++i) {
      acc[2 * i].v       = fresh ? _mm512_setzero_ps() : _mm512_loadu_ps(tile + (i * ld));
      acc[(2 * i) + 1].v = fresh ? _mm512_setzero_ps() : _mm512_loadu_ps(tile + (i * ld) + 16);
    }
    NUMERICAL_UNROLL_DEPTH
    for (std::size_t p {0}; p < kc; ++p) {
      const __m512 b0 {_mm512_loadu_ps(bp + (p * ldb))};
      const __m512 b1 {_mm512_loadu_ps(bp + (p * ldb) + 16)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < Rows; ++i) {
        const __m512 a {_mm512_set1_ps(ap[(p * avx512_mr) + i])};
        acc[2 * i].v       = _mm512_fmadd_ps(a, b0, acc[2 * i].v);
        acc[(2 * i) + 1].v = _mm512_fmadd_ps(a, b1, acc[(2 * i) + 1].v);
      }
    }
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < Rows; ++i) {
      _mm512_storeu_ps(tile + (i * ld), acc[2 * i].v);
      _mm512_storeu_ps(tile + (i * ld) + 16, acc[(2 * i) + 1].v);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

  //! \brief \ref avx512_full_tile_f32 for a tile of \p rows × \p cols, a partial one by its rows alone or through a whole one (\ref through_rows).
  inline void avx512_tile_f32(std::size_t  kc,
                              const float* ap,
                              const float* bp,
                              std::size_t  ldb,
                              float*       c,
                              std::size_t  ldc,
                              std::size_t  rows,
                              std::size_t  cols,
                              bool         fresh = false) noexcept
  {
    through_rows<avx512_mr, avx512_nr_f32>(c, ldc, rows, cols, [&]<std::size_t Rows>(float* tile, std::size_t ld) noexcept { avx512_full_tile_f32<Rows>(kc, ap, bp, ldb, tile, ld, fresh); });
  }

  //! \brief \ref avx2_full_tile_f64 for `float`: \ref avx2_mr × \ref avx2_nr_f32, 12 accumulators of eight.
  template <std::size_t Rows = avx2_mr>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_full_tile_f32(std::size_t  kc,
                                                              const float* ap,
                                                              const float* bp,
                                                              std::size_t  ldb,
                                                              float*       tile,
                                                              std::size_t  ld,
                                                              bool         fresh = false) noexcept
  {
    std::array<ymm_register_f32, Rows * 2> acc {};
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < Rows; ++i) {
      acc[2 * i].v       = fresh ? _mm256_setzero_ps() : _mm256_loadu_ps(tile + (i * ld));
      acc[(2 * i) + 1].v = fresh ? _mm256_setzero_ps() : _mm256_loadu_ps(tile + (i * ld) + 8);
    }
    NUMERICAL_UNROLL_DEPTH_AVX2
    for (std::size_t p {0}; p < kc; ++p) {
      _mm_prefetch(reinterpret_cast<const char*>(bp + (p * ldb) + (2 * avx2_prefetch_ahead)), _MM_HINT_T0);     // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the intrinsic takes a char pointer
      _mm_prefetch(reinterpret_cast<const char*>(ap + (p * avx2_mr) + (2 * avx2_prefetch_ahead)), _MM_HINT_T0); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
      const __m256 b0 {_mm256_loadu_ps(bp + (p * ldb))};
      const __m256 b1 {_mm256_loadu_ps(bp + (p * ldb) + 8)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < Rows; ++i) {
        const __m256 a {_mm256_broadcast_ss(ap + (p * avx2_mr) + i)};
        acc[2 * i].v       = _mm256_fmadd_ps(a, b0, acc[2 * i].v);
        acc[(2 * i) + 1].v = _mm256_fmadd_ps(a, b1, acc[(2 * i) + 1].v);
      }
    }
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < Rows; ++i) {
      _mm256_storeu_ps(tile + (i * ld), acc[2 * i].v);
      _mm256_storeu_ps(tile + (i * ld) + 8, acc[(2 * i) + 1].v);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

  //! \brief \ref avx2_full_tile_f32 for a tile of \p rows × \p cols, a partial one by its rows alone or through a whole one (\ref through_rows).
  inline void avx2_tile_f32(std::size_t  kc,
                            const float* ap,
                            const float* bp,
                            std::size_t  ldb,
                            float*       c,
                            std::size_t  ldc,
                            std::size_t  rows,
                            std::size_t  cols,
                            bool         fresh = false) noexcept
  {
    through_rows<avx2_mr, avx2_nr_f32>(c, ldc, rows, cols, [&]<std::size_t Rows>(float* tile, std::size_t ld) noexcept { avx2_full_tile_f32<Rows>(kc, ap, bp, ldb, tile, ld, fresh); });
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
  template <std::size_t Rows = avx2_complex_mr,
            bool        AInPlace = false>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_full_tile_c64(std::size_t                 kc,
                                                              const std::complex<double>* ap,
                                                              const std::complex<double>* bp,
                                                              std::size_t                 ldb,
                                                              std::complex<double>*       tile,
                                                              std::size_t                 ld,
                                                              std::size_t                 a_step                        = avx2_complex_mr,
                                                              std::size_t                 a_row                         = 1,
                                                              bool                        fresh                         = false) noexcept
  {
    const std::size_t                             step   {AInPlace ? a_step : avx2_complex_mr}; // A's elements apart from one step to the next
    const std::size_t                             across {AInPlace ? a_row : 1};                // and from one row to the next
    const auto* const                             a      {reinterpret_cast<const double*>(ap)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
    const auto* const                             b      {reinterpret_cast<const double*>(bp)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    auto* const                                   c      {reinterpret_cast<double*>(tile)};     // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    std::array<ymm_register, Rows * 4>            acc    {};
    NUMERICAL_UNROLL_DEPTH
    for (std::size_t p {0}; p < kc; ++p) {
      const __m256d b0 {_mm256_loadu_pd(b + (2 * p * ldb))};
      const __m256d b1 {_mm256_loadu_pd(b + (2 * p * ldb) + 4)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < Rows; ++i) {
        const __m256d re {_mm256_broadcast_sd(a + (2 * ((p * step) + (i * across))))};
        const __m256d im {_mm256_broadcast_sd(a + (2 * ((p * step) + (i * across))) + 1)};
        acc[4 * i].v       = _mm256_fmadd_pd(re, b0, acc[4 * i].v);
        acc[(4 * i) + 1].v = _mm256_fmadd_pd(re, b1, acc[(4 * i) + 1].v);
        acc[(4 * i) + 2].v = _mm256_fmadd_pd(im, b0, acc[(4 * i) + 2].v);
        acc[(4 * i) + 3].v = _mm256_fmadd_pd(im, b1, acc[(4 * i) + 3].v);
      }
    }
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < Rows; ++i) {
      double* const row {c + (2 * i * ld)};
      const __m256d lo  {_mm256_addsub_pd(acc[4 * i].v, _mm256_permute_pd(acc[(4 * i) + 2].v, 0b0101))};
      const __m256d hi  {_mm256_addsub_pd(acc[(4 * i) + 1].v, _mm256_permute_pd(acc[(4 * i) + 3].v, 0b0101))};
      _mm256_storeu_pd(row, _mm256_add_pd(fresh ? _mm256_setzero_pd() : _mm256_loadu_pd(row), lo));
      _mm256_storeu_pd(row + 4, _mm256_add_pd(fresh ? _mm256_setzero_pd() : _mm256_loadu_pd(row + 4), hi));
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

  //! \brief \ref avx2_full_tile_c64 for a tile of \p rows × \p cols (at most \ref avx2_complex_mr ×
  //!        \ref avx2_complex_nr), a partial one by its rows alone or through a whole one whose outside is zero (\ref through_rows).
  inline void avx2_tile_c64(std::size_t                 kc,
                            const std::complex<double>* ap,
                            const std::complex<double>* bp,
                            std::size_t                 ldb,
                            std::complex<double>*       c,
                            std::size_t                 ldc,
                            std::size_t                 rows,
                            std::size_t                 cols,
                            bool                        fresh = false) noexcept
  {
    through_rows<avx2_complex_mr, avx2_complex_nr>(c, ldc, rows, cols, [&]<std::size_t Rows>(std::complex<double>* tile, std::size_t ld) noexcept {
                                                     avx2_full_tile_c64<Rows>(kc, ap, bp, ldb, tile, ld, avx2_complex_mr, 1, fresh);
                                                   });
  }

  //! \brief \ref avx2_tile_c64 with A read where it lies: element `(i, p)` of the tile's rows at `a[(p · a_step) + (i · a_row)]`,
  //!        its rows alone computed (no row past A's last is read).
  inline void avx2_tile_c64_a_in_place(std::size_t                 kc,
                                       const std::complex<double>* a,
                                       std::size_t                 a_step,
                                       std::size_t                 a_row,
                                       const std::complex<double>* bp,
                                       std::size_t                 ldb,
                                       std::complex<double>      * c,
                                       std::size_t                 ldc,
                                       std::size_t                 rows,
                                       std::size_t                 cols,
                                       bool                        fresh = false) noexcept
  {
    through_rows<avx2_complex_mr, avx2_complex_nr>(c, ldc, rows, cols, [&]<std::size_t Rows>(std::complex<double>* tile, std::size_t ld) noexcept {
                                                     avx2_full_tile_c64<Rows, true>(kc, a, bp, ldb, tile, ld, a_step, a_row, fresh);
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
  template <std::size_t Rows = avx512_complex_mr,
            bool        AInPlace = false>
  NUMERICAL_TARGET("avx512f") inline void avx512_full_tile_c64(std::size_t                 kc,
                                                               const std::complex<double>* ap,
                                                               const std::complex<double>* bp,
                                                               std::size_t                 ldb,
                                                               std::complex<double>*       tile,
                                                               std::size_t                 ld,
                                                               std::size_t                 a_step                        = avx512_complex_mr,
                                                               std::size_t                 a_row                         = 1,
                                                               bool                        fresh                         = false) noexcept
  {
    const std::size_t                               step   {AInPlace ? a_step : avx512_complex_mr}; // A's elements apart from one step to the next
    const std::size_t                               across {AInPlace ? a_row : 1};                  // and from one row to the next
    const auto* const                               a      {reinterpret_cast<const double*>(ap)};   // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
    const auto* const                               b      {reinterpret_cast<const double*>(bp)};   // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    auto* const                                     c      {reinterpret_cast<double*>(tile)};       // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    std::array<zmm_register, Rows * 4>              acc    {};
    NUMERICAL_UNROLL_DEPTH
    for (std::size_t p {0}; p < kc; ++p) {
      const __m512d b0 {_mm512_loadu_pd(b + (2 * p * ldb))};
      const __m512d b1 {_mm512_loadu_pd(b + (2 * p * ldb) + 8)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < Rows; ++i) {
        const __m512d re {_mm512_set1_pd(a[2 * ((p * step) + (i * across))])};
        const __m512d im {_mm512_set1_pd(a[(2 * ((p * step) + (i * across))) + 1])};
        acc[4 * i].v       = _mm512_fmadd_pd(re, b0, acc[4 * i].v);
        acc[(4 * i) + 1].v = _mm512_fmadd_pd(re, b1, acc[(4 * i) + 1].v);
        acc[(4 * i) + 2].v = _mm512_fmadd_pd(im, b0, acc[(4 * i) + 2].v);
        acc[(4 * i) + 3].v = _mm512_fmadd_pd(im, b1, acc[(4 * i) + 3].v);
      }
    }
    const __m512d signs {_mm512_set_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)}; // −1 on the real parts
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < Rows; ++i) {
      double* const row {c + (2 * i * ld)};
      const __m512d lo  {_mm512_fmadd_pd(_mm512_permute_pd(acc[(4 * i) + 2].v, 0x55), signs, acc[4 * i].v)};
      const __m512d hi  {_mm512_fmadd_pd(_mm512_permute_pd(acc[(4 * i) + 3].v, 0x55), signs, acc[(4 * i) + 1].v)};
      _mm512_storeu_pd(row, _mm512_add_pd(fresh ? _mm512_setzero_pd() : _mm512_loadu_pd(row), lo));
      _mm512_storeu_pd(row + 8, _mm512_add_pd(fresh ? _mm512_setzero_pd() : _mm512_loadu_pd(row + 8), hi));
    }
  }

  //! \brief \ref avx512_full_tile_c64 for a tile of \p rows × \p cols, a partial one by its rows alone or through a whole one (\ref through_rows).
  inline void avx512_tile_c64(std::size_t                 kc,
                              const std::complex<double>* ap,
                              const std::complex<double>* bp,
                              std::size_t                 ldb,
                              std::complex<double>*       c,
                              std::size_t                 ldc,
                              std::size_t                 rows,
                              std::size_t                 cols,
                              bool                        fresh = false) noexcept
  {
    through_rows<avx512_complex_mr, avx512_complex_nr>(c, ldc, rows, cols, [&]<std::size_t Rows>(std::complex<double>* tile, std::size_t ld) noexcept {
                                                         avx512_full_tile_c64<Rows>(kc, ap, bp, ldb, tile, ld, avx512_complex_mr, 1, fresh);
                                                       });
  }

  //! \brief \ref avx512_tile_c64 with A read where it lies: element `(i, p)` of the tile's rows at `a[(p · a_step) + (i · a_row)]`,
  //!        its rows alone computed (no row past A's last is read).
  inline void avx512_tile_c64_a_in_place(std::size_t                 kc,
                                         const std::complex<double>* a,
                                         std::size_t                 a_step,
                                         std::size_t                 a_row,
                                         const std::complex<double>* bp,
                                         std::size_t                 ldb,
                                         std::complex<double>      * c,
                                         std::size_t                 ldc,
                                         std::size_t                 rows,
                                         std::size_t                 cols,
                                         bool                        fresh = false) noexcept
  {
    through_rows<avx512_complex_mr, avx512_complex_nr>(c, ldc, rows, cols, [&]<std::size_t Rows>(std::complex<double>* tile, std::size_t ld) noexcept {
                                                         avx512_full_tile_c64<Rows, true>(kc, a, bp, ldb, tile, ld, a_step, a_row, fresh);
                                                       });
  }

  //! \brief The columns of C one AVX2 complex `float` tile covers: two registers of four `std::complex<float>`.
  inline constexpr std::size_t avx2_complex_nr_f32 {8};

  /*!
   * \brief \ref avx2_full_tile_c64 for complex `float`: \ref avx2_complex_mr × \ref avx2_complex_nr_f32, 12
   *        accumulators of eight `float`, the same arithmetic.
   */
  template <std::size_t Rows = avx2_complex_mr,
            bool        AInPlace = false>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_full_tile_c32(std::size_t                kc,
                                                              const std::complex<float>* ap,
                                                              const std::complex<float>* bp,
                                                              std::size_t                ldb,
                                                              std::complex<float>*       tile,
                                                              std::size_t                ld,
                                                              std::size_t                a_step                       = avx2_complex_mr,
                                                              std::size_t                a_row                        = 1,
                                                              bool                       fresh                        = false) noexcept
  {
    const std::size_t                                 step   {AInPlace ? a_step : avx2_complex_mr}; // A's elements apart from one step to the next
    const std::size_t                                 across {AInPlace ? a_row : 1};                // and from one row to the next
    const auto* const                                 a      {reinterpret_cast<const float*>(ap)};  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee of [complex.numbers]
    const auto* const                                 b      {reinterpret_cast<const float*>(bp)};  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    auto* const                                       c      {reinterpret_cast<float*>(tile)};      // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    std::array<ymm_register_f32, Rows * 4>            acc    {};
    NUMERICAL_UNROLL_DEPTH
    for (std::size_t p {0}; p < kc; ++p) {
      const __m256 b0 {_mm256_loadu_ps(b + (2 * p * ldb))};
      const __m256 b1 {_mm256_loadu_ps(b + (2 * p * ldb) + 8)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < Rows; ++i) {
        const __m256 re {_mm256_broadcast_ss(a + (2 * ((p * step) + (i * across))))};
        const __m256 im {_mm256_broadcast_ss(a + (2 * ((p * step) + (i * across))) + 1)};
        acc[4 * i].v       = _mm256_fmadd_ps(re, b0, acc[4 * i].v);
        acc[(4 * i) + 1].v = _mm256_fmadd_ps(re, b1, acc[(4 * i) + 1].v);
        acc[(4 * i) + 2].v = _mm256_fmadd_ps(im, b0, acc[(4 * i) + 2].v);
        acc[(4 * i) + 3].v = _mm256_fmadd_ps(im, b1, acc[(4 * i) + 3].v);
      }
    }
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < Rows; ++i) {
      float* const row {c + (2 * i * ld)};
      const __m256 lo  {_mm256_addsub_ps(acc[4 * i].v, _mm256_permute_ps(acc[(4 * i) + 2].v, 0xB1))};
      const __m256 hi  {_mm256_addsub_ps(acc[(4 * i) + 1].v, _mm256_permute_ps(acc[(4 * i) + 3].v, 0xB1))};
      _mm256_storeu_ps(row, _mm256_add_ps(fresh ? _mm256_setzero_ps() : _mm256_loadu_ps(row), lo));
      _mm256_storeu_ps(row + 8, _mm256_add_ps(fresh ? _mm256_setzero_ps() : _mm256_loadu_ps(row + 8), hi));
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

  //! \brief \ref avx2_full_tile_c32 for a tile of \p rows × \p cols, a partial one by its rows alone or through a whole one (\ref through_rows).
  inline void avx2_tile_c32(std::size_t                kc,
                            const std::complex<float>* ap,
                            const std::complex<float>* bp,
                            std::size_t                ldb,
                            std::complex<float>*       c,
                            std::size_t                ldc,
                            std::size_t                rows,
                            std::size_t                cols,
                            bool                       fresh = false) noexcept
  {
    through_rows<avx2_complex_mr, avx2_complex_nr_f32>(c, ldc, rows, cols, [&]<std::size_t Rows>(std::complex<float>* tile, std::size_t ld) noexcept {
                                                         avx2_full_tile_c32<Rows>(kc, ap, bp, ldb, tile, ld, avx2_complex_mr, 1, fresh);
                                                       });
  }

  //! \brief \ref avx2_tile_c32 with A read where it lies: element `(i, p)` of the tile's rows at `a[(p · a_step) + (i · a_row)]`,
  //!        its rows alone computed (no row past A's last is read).
  inline void avx2_tile_c32_a_in_place(std::size_t                kc,
                                       const std::complex<float>* a,
                                       std::size_t                a_step,
                                       std::size_t                a_row,
                                       const std::complex<float>* bp,
                                       std::size_t                ldb,
                                       std::complex<float>      * c,
                                       std::size_t                ldc,
                                       std::size_t                rows,
                                       std::size_t                cols,
                                       bool                       fresh = false) noexcept
  {
    through_rows<avx2_complex_mr, avx2_complex_nr_f32>(c, ldc, rows, cols, [&]<std::size_t Rows>(std::complex<float>* tile, std::size_t ld) noexcept {
                                                         avx2_full_tile_c32<Rows, true>(kc, a, bp, ldb, tile, ld, a_step, a_row, fresh);
                                                       });
  }

  //! \brief The columns of C one AVX-512 complex `float` tile covers: two registers of eight `std::complex<float>`.
  inline constexpr std::size_t avx512_complex_nr_f32 {16};

  //! \brief \ref avx512_full_tile_c64 for complex `float`: \ref avx512_complex_mr × \ref avx512_complex_nr_f32, 24
  //!        accumulators of sixteen `float`, the same arithmetic.
  template <std::size_t Rows = avx512_complex_mr,
            bool        AInPlace = false>
  NUMERICAL_TARGET("avx512f") inline void avx512_full_tile_c32(std::size_t                kc,
                                                               const std::complex<float>* ap,
                                                               const std::complex<float>* bp,
                                                               std::size_t                ldb,
                                                               std::complex<float>*       tile,
                                                               std::size_t                ld,
                                                               std::size_t                a_step                       = avx512_complex_mr,
                                                               std::size_t                a_row                        = 1,
                                                               bool                       fresh                        = false) noexcept
  {
    const std::size_t                                   step   {AInPlace ? a_step : avx512_complex_mr}; // A's elements apart from one step to the next
    const std::size_t                                   across {AInPlace ? a_row : 1};                  // and from one row to the next
    const auto* const                                   a      {reinterpret_cast<const float*>(ap)};    // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee of [complex.numbers]
    const auto* const                                   b      {reinterpret_cast<const float*>(bp)};    // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    auto* const                                         c      {reinterpret_cast<float*>(tile)};        // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    std::array<zmm_register_f32, Rows * 4>              acc    {};
    NUMERICAL_UNROLL_DEPTH
    for (std::size_t p {0}; p < kc; ++p) {
      const __m512 b0 {_mm512_loadu_ps(b + (2 * p * ldb))};
      const __m512 b1 {_mm512_loadu_ps(b + (2 * p * ldb) + 16)};
      NUMERICAL_UNROLL_TILE
      for (std::size_t i {0}; i < Rows; ++i) {
        const __m512 re {_mm512_set1_ps(a[2 * ((p * step) + (i * across))])};
        const __m512 im {_mm512_set1_ps(a[(2 * ((p * step) + (i * across))) + 1])};
        acc[4 * i].v       = _mm512_fmadd_ps(re, b0, acc[4 * i].v);
        acc[(4 * i) + 1].v = _mm512_fmadd_ps(re, b1, acc[(4 * i) + 1].v);
        acc[(4 * i) + 2].v = _mm512_fmadd_ps(im, b0, acc[(4 * i) + 2].v);
        acc[(4 * i) + 3].v = _mm512_fmadd_ps(im, b1, acc[(4 * i) + 3].v);
      }
    }
    const __m512 signs {_mm512_set_ps(1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F)}; // −1 on the real parts
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < Rows; ++i) {
      float* const row {c + (2 * i * ld)};
      const __m512 lo  {_mm512_fmadd_ps(_mm512_permute_ps(acc[(4 * i) + 2].v, 0xB1), signs, acc[4 * i].v)};
      const __m512 hi  {_mm512_fmadd_ps(_mm512_permute_ps(acc[(4 * i) + 3].v, 0xB1), signs, acc[(4 * i) + 1].v)};
      _mm512_storeu_ps(row, _mm512_add_ps(fresh ? _mm512_setzero_ps() : _mm512_loadu_ps(row), lo));
      _mm512_storeu_ps(row + 16, _mm512_add_ps(fresh ? _mm512_setzero_ps() : _mm512_loadu_ps(row + 16), hi));
    }
  }

  //! \brief \ref avx512_full_tile_c32 for a tile of \p rows × \p cols, a partial one by its rows alone or through a whole one (\ref through_rows).
  inline void avx512_tile_c32(std::size_t                kc,
                              const std::complex<float>* ap,
                              const std::complex<float>* bp,
                              std::size_t                ldb,
                              std::complex<float>*       c,
                              std::size_t                ldc,
                              std::size_t                rows,
                              std::size_t                cols,
                              bool                       fresh = false) noexcept
  {
    through_rows<avx512_complex_mr, avx512_complex_nr_f32>(c, ldc, rows, cols, [&]<std::size_t Rows>(std::complex<float>* tile, std::size_t ld) noexcept {
                                                             avx512_full_tile_c32<Rows>(kc, ap, bp, ldb, tile, ld, avx512_complex_mr, 1, fresh);
                                                           });
  }

  //! \brief \ref avx512_tile_c32 with A read where it lies: element `(i, p)` of the tile's rows at `a[(p · a_step) + (i · a_row)]`,
  //!        its rows alone computed (no row past A's last is read).
  inline void avx512_tile_c32_a_in_place(std::size_t                kc,
                                         const std::complex<float>* a,
                                         std::size_t                a_step,
                                         std::size_t                a_row,
                                         const std::complex<float>* bp,
                                         std::size_t                ldb,
                                         std::complex<float>      * c,
                                         std::size_t                ldc,
                                         std::size_t                rows,
                                         std::size_t                cols,
                                         bool                       fresh = false) noexcept
  {
    through_rows<avx512_complex_mr, avx512_complex_nr_f32>(c, ldc, rows, cols, [&]<std::size_t Rows>(std::complex<float>* tile, std::size_t ld) noexcept {
                                                             avx512_full_tile_c32<Rows, true>(kc, a, bp, ldb, tile, ld, a_step, a_row, fresh);
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

  /*!
   * \brief The leading whole blocks of 16 of \p count contiguous `double`s for \ref contiguous_extreme: their largest
   *        (\p Maximum) or smallest into \p result and whether a NaN is among them into \p nan, four AVX2 vectors each
   *        its own running best, a NaN never kept (`max` returns its second operand, the best, when the new one is a
   *        NaN) but flagged apart. A tie keeps the best met, as the portable lanes do; a zero's sign is settled after.
   * \return The elements done.
   */
  template <bool Maximum>
  NUMERICAL_TARGET("avx2") inline std::size_t avx2_extreme(const double* first,
                                                           std::size_t   count,
                                                           double&       result,
                                                           bool&         nan) noexcept
  {
    constexpr std::size_t block {16};
    const std::size_t     done  {count - (count % block)};
    if (done == 0) {
      return 0;
    }
    std::array<ymm_register, 4> best {};
    std::array<ymm_register, 4> nans {};
    for (std::size_t l {0}; l < 4; ++l) {
      best[l].v = _mm256_loadu_pd(first + (l * 4));
      nans[l].v = _mm256_setzero_pd();
    }
    for (std::size_t i {0}; i < done; i += block) {
      for (std::size_t l {0}; l < 4; ++l) {
        const __m256d x {_mm256_loadu_pd(first + i + (l * 4))};
        best[l].v = Maximum ? _mm256_max_pd(x, best[l].v) : _mm256_min_pd(x, best[l].v);
        nans[l].v = _mm256_or_pd(nans[l].v, _mm256_cmp_pd(x, x, _CMP_UNORD_Q));
      }
    }
    std::array<double, block> values {};
    std::array<double, block> flags  {};
    for (std::size_t l {0}; l < 4; ++l) {
      _mm256_storeu_pd(values.data() + (l * 4), best[l].v);
      _mm256_storeu_pd(flags.data() + (l * 4), nans[l].v);
    }
    _mm256_zeroupper(); // the upper halves cleared: the tail may be SSE alone
    for (std::size_t k {0}; k < block; ++k) {
      nan    = nan || std::bit_cast<std::uint64_t>(flags[k]) != 0U;
      result = (Maximum ? values[k] > result : values[k] < result) ? values[k] : result;
    }
    return done;
  }

  //! \brief \ref avx2_extreme for `float`, whole blocks of 32.
  template <bool Maximum>
  NUMERICAL_TARGET("avx2") inline std::size_t avx2_extreme(const float* first,
                                                           std::size_t  count,
                                                           float&       result,
                                                           bool&        nan) noexcept
  {
    constexpr std::size_t block {32};
    const std::size_t     done  {count - (count % block)};
    if (done == 0) {
      return 0;
    }
    std::array<ymm_register_f32, 4> best {};
    std::array<ymm_register_f32, 4> nans {};
    for (std::size_t l {0}; l < 4; ++l) {
      best[l].v = _mm256_loadu_ps(first + (l * 8));
      nans[l].v = _mm256_setzero_ps();
    }
    for (std::size_t i {0}; i < done; i += block) {
      for (std::size_t l {0}; l < 4; ++l) {
        const __m256 x {_mm256_loadu_ps(first + i + (l * 8))};
        best[l].v = Maximum ? _mm256_max_ps(x, best[l].v) : _mm256_min_ps(x, best[l].v);
        nans[l].v = _mm256_or_ps(nans[l].v, _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
      }
    }
    std::array<float, block> values {};
    std::array<float, block> flags  {};
    for (std::size_t l {0}; l < 4; ++l) {
      _mm256_storeu_ps(values.data() + (l * 8), best[l].v);
      _mm256_storeu_ps(flags.data() + (l * 8), nans[l].v);
    }
    _mm256_zeroupper(); // the upper halves cleared: the tail may be SSE alone
    for (std::size_t k {0}; k < block; ++k) {
      nan    = nan || std::bit_cast<std::uint32_t>(flags[k]) != 0U;
      result = (Maximum ? values[k] > result : values[k] < result) ? values[k] : result;
    }
    return done;
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

  //! \brief \ref avx2_scaled_add for `float`, as the x86-64 `float` tiles round their terms.
  NUMERICAL_TARGET("avx2,fma") inline void avx2_scaled_add(float*       y,
                                                           const float* x,
                                                           float        f,
                                                           std::size_t  count) noexcept
  {
    const __m256      splat {_mm256_set1_ps(f)};
    const std::size_t done  {count - (count % 8)};
    for (std::size_t i {0}; i < done; i += 8) {
      _mm256_storeu_ps(y + i, _mm256_fmadd_ps(splat, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
    for (std::size_t i {done}; i < count; ++i) {
      y[i] = std::fma(f, x[i], y[i]);
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  /*!
   * \brief The leading elements of `y[i] += f · x[i]` for complex `float`, whole vectors of four: how many it did. Each
   *        product's part products rounded apart (`f.re · x` and `f.im ·` x's exchanged parts), then `re·re − im·im` and
   *        `re·im + im·re` by one `addsub`, then added to `y`: \ref complex_scaled_add's arithmetic, so the same bits as
   *        the scalar loop. AVX2 alone, no fused multiply-add: none may be contracted.
   */
  NUMERICAL_TARGET("avx2") inline std::size_t avx2_scaled_add_c32(std::complex<float>*       y,
                                                                  const std::complex<float>* x,
                                                                  std::complex<float>        f,
                                                                  std::size_t                count) noexcept
  {
    auto* const       out  {reinterpret_cast<float*>(y)};       // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee of [complex.numbers]
    const auto* const in   {reinterpret_cast<const float*>(x)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    const __m256      re   {_mm256_set1_ps(f.real())};
    const __m256      im   {_mm256_set1_ps(f.imag())};
    const std::size_t done {count - (count % 4)};
    for (std::size_t i {0}; i < 2 * done; i += 8) {
      const __m256 xv      {_mm256_loadu_ps(in + i)};
      const __m256 product {_mm256_addsub_ps(_mm256_mul_ps(re, xv), _mm256_mul_ps(im, _mm256_permute_ps(xv, 0xB1)))};
      _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(out + i), product));
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return done;
  }

  //! \brief \ref avx2_scaled_add_c32 for complex `double`: whole vectors of two.
  NUMERICAL_TARGET("avx2") inline std::size_t avx2_scaled_add_c64(std::complex<double>*       y,
                                                                  const std::complex<double>* x,
                                                                  std::complex<double>        f,
                                                                  std::size_t                 count) noexcept
  {
    auto* const       out  {reinterpret_cast<double*>(y)};       // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee of [complex.numbers]
    const auto* const in   {reinterpret_cast<const double*>(x)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    const __m256d     re   {_mm256_set1_pd(f.real())};
    const __m256d     im   {_mm256_set1_pd(f.imag())};
    const std::size_t done {count - (count % 2)};
    for (std::size_t i {0}; i < 2 * done; i += 4) {
      const __m256d xv      {_mm256_loadu_pd(in + i)};
      const __m256d product {_mm256_addsub_pd(_mm256_mul_pd(re, xv), _mm256_mul_pd(im, _mm256_permute_pd(xv, 0b0101)))};
      _mm256_storeu_pd(out + i, _mm256_add_pd(_mm256_loadu_pd(out + i), product));
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return done;
  }

  //! \brief Columns \p j to `j + Width` of \ref avx2_strided_columns into \p y, their sums held through the rows.
  template <std::size_t Width,
            typename T>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_strided_column_group(const T*    a,
                                                                     const T*    b,
                                                                     std::size_t row_step,
                                                                     std::size_t column_step,
                                                                     std::size_t inner,
                                                                     std::size_t j,
                                                                     T*          y) noexcept
  {
    std::array<T, Width> sums {};
    for (std::size_t k {0}; k < inner; ++k) {
      const T        f   {a[k]};
      const T* const row {b + (k * row_step) + (j * column_step)};
      for (std::size_t c {0}; c < Width; ++c) {
        sums[c] = std::fma(f, row[c * column_step], sums[c]);
      }
    }
    std::copy(sums.begin(), sums.end(), y + j);
  }

  /*!
   * \brief Columns \p first to \p last of a vector \p a (\p inner) times a matrix \p b whose element `B[k][j]` is
   *        `b[(k · row_step) + (j · column_step)]`, into \p y: each element `a[k] · B[k][j]` added in ascending order
   *        of `k` by one fused multiply-add, as \ref avx2_scaled_add adds each row (and so \ref avx512_scaled_add,
   *        whose processors have these units too); eight columns' chains at once, the rest as one group of its width.
   *        `double` or `float`.
   */
  template <typename T>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_strided_columns(const T*    a,
                                                                const T*    b,
                                                                std::size_t row_step,
                                                                std::size_t column_step,
                                                                std::size_t inner,
                                                                std::size_t first,
                                                                std::size_t last,
                                                                T*          y) noexcept
  {
    constexpr std::size_t group {8}; // the switch below names each narrower last group
    std::size_t           j     {first};
    for (; j + group <= last; j += group) {
      avx2_strided_column_group<group>(a, b, row_step, column_step, inner, j, y);
    }
    switch (last - j) {
      case 1: avx2_strided_column_group<1>(a, b, row_step, column_step, inner, j, y); break;
      case 2: avx2_strided_column_group<2>(a, b, row_step, column_step, inner, j, y); break;
      case 3: avx2_strided_column_group<3>(a, b, row_step, column_step, inner, j, y); break;
      case 4: avx2_strided_column_group<4>(a, b, row_step, column_step, inner, j, y); break;
      case 5: avx2_strided_column_group<5>(a, b, row_step, column_step, inner, j, y); break;
      case 6: avx2_strided_column_group<6>(a, b, row_step, column_step, inner, j, y); break;
      case 7: avx2_strided_column_group<7>(a, b, row_step, column_step, inner, j, y); break;
      default: break; // a whole number of groups
    }
  }

  //! \brief The blocks of \ref avx512_dot: `x · y` over the first \p done elements (a multiple of 32), four
  //!        accumulators of eight `double`, chains of fused multiply-adds, summed in a fixed order from +0 (none: +0).
  NUMERICAL_TARGET("avx512f") inline double avx512_dot_blocks(const double* x,
                                                              const double* y,
                                                              std::size_t   done) noexcept
  {
    constexpr std::size_t           width {8};
    constexpr std::size_t           lanes {4};
    std::array<zmm_register, lanes> sum   {};
    for (std::size_t i {0}; i < done; i += width * lanes) {
      NUMERICAL_UNROLL_TILE
      for (std::size_t l {0}; l < lanes; ++l) {
        sum[l].v = _mm512_fmadd_pd(_mm512_loadu_pd(x + i + (l * width)), _mm512_loadu_pd(y + i + (l * width)), sum[l].v);
      }
    }
    double total {0.0};
    for (std::size_t l {0}; done != 0 && l < lanes; ++l) { // none summed: +0 left as it is
      total += _mm512_reduce_add_pd(sum[l].v);
    }
    return total;
  }

  //! \brief `Σ x[i] · y[i]`: its blocks of 32 (\ref avx512_dot_blocks), then the remainder added in order.
  NUMERICAL_TARGET("avx512f") inline double avx512_dot(const double* x,
                                                       const double* y,
                                                       std::size_t   count) noexcept
  {
    const std::size_t done  {count - (count % 32)};
    double            total {avx512_dot_blocks(x, y, done)};
    for (std::size_t i {done}; i < count; ++i) {
      total += x[i] * y[i];
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return total;
  }

  //! \brief \ref avx512_dot_blocks with four accumulators of four `double` (AVX2 with FMA), \p done a multiple of 16.
  NUMERICAL_TARGET("avx2,fma") inline double avx2_dot_blocks(const double* x,
                                                             const double* y,
                                                             std::size_t   done) noexcept
  {
    constexpr std::size_t           width {4};
    constexpr std::size_t           lanes {4};
    std::array<ymm_register, lanes> sum   {};
    for (std::size_t i {0}; i < done; i += width * lanes) {
      NUMERICAL_UNROLL_TILE
      for (std::size_t l {0}; l < lanes; ++l) {
        sum[l].v = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + (l * width)), _mm256_loadu_pd(y + i + (l * width)), sum[l].v);
      }
    }
    double total {0.0};
    for (std::size_t l {0}; done != 0 && l < lanes; ++l) { // none summed: +0 left as it is
      alignas(32) std::array<double, width> part {};       // a register stored whole: some compilers store it aligned
      _mm256_storeu_pd(part.data(), sum[l].v);
      total += (part[0] + part[1]) + (part[2] + part[3]);
    }
    return total;
  }

  //! \brief \ref avx512_dot on AVX2 with FMA: blocks of 16 (\ref avx2_dot_blocks), then the remainder in order.
  NUMERICAL_TARGET("avx2,fma") inline double avx2_dot(const double* x,
                                                      const double* y,
                                                      std::size_t   count) noexcept
  {
    const std::size_t done  {count - (count % 16)};
    double            total {avx2_dot_blocks(x, y, done)};
    for (std::size_t i {done}; i < count; ++i) {
      total += x[i] * y[i];
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return total;
  }

  //! \brief \ref avx512_dot_blocks of `float`: four accumulators of sixteen, \p done a multiple of 64, chains of fused
  //!        multiply-adds summed in a fixed order from +0 (none: +0).
  NUMERICAL_TARGET("avx512f") inline float avx512_dot_blocks(const float* x,
                                                             const float* y,
                                                             std::size_t  done) noexcept
  {
    constexpr std::size_t               width {16};
    constexpr std::size_t               lanes {4};
    std::array<zmm_register_f32, lanes> sum   {};
    for (std::size_t i {0}; i < done; i += width * lanes) {
      NUMERICAL_UNROLL_TILE
      for (std::size_t l {0}; l < lanes; ++l) {
        sum[l].v = _mm512_fmadd_ps(_mm512_loadu_ps(x + i + (l * width)), _mm512_loadu_ps(y + i + (l * width)), sum[l].v);
      }
    }
    float total {0.0F};
    for (std::size_t l {0}; done != 0 && l < lanes; ++l) { // none summed: +0 left as it is
      total += _mm512_reduce_add_ps(sum[l].v);
    }
    return total;
  }

  //! \brief \ref avx512_dot of `float`: its blocks of 64, then the remainder added in order.
  NUMERICAL_TARGET("avx512f") inline float avx512_dot(const float* x,
                                                      const float* y,
                                                      std::size_t  count) noexcept
  {
    const std::size_t done  {count - (count % 64)};
    float             total {avx512_dot_blocks(x, y, done)};
    for (std::size_t i {done}; i < count; ++i) {
      total += x[i] * y[i];
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return total;
  }

  //! \brief \ref avx2_dot_blocks of `float`: four accumulators of eight, \p done a multiple of 32, each register's eight
  //!        summed as a tree of pairs.
  NUMERICAL_TARGET("avx2,fma") inline float avx2_dot_blocks(const float* x,
                                                            const float* y,
                                                            std::size_t  done) noexcept
  {
    constexpr std::size_t               width {8};
    constexpr std::size_t               lanes {4};
    std::array<ymm_register_f32, lanes> sum   {};
    for (std::size_t i {0}; i < done; i += width * lanes) {
      NUMERICAL_UNROLL_TILE
      for (std::size_t l {0}; l < lanes; ++l) {
        sum[l].v = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + (l * width)), _mm256_loadu_ps(y + i + (l * width)), sum[l].v);
      }
    }
    float total {0.0F};
    for (std::size_t l {0}; done != 0 && l < lanes; ++l) { // none summed: +0 left as it is
      alignas(32) std::array<float, width> part {};        // a register stored whole: some compilers store it aligned
      _mm256_storeu_ps(part.data(), sum[l].v);
      total += ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
    }
    return total;
  }

  //! \brief \ref avx2_dot of `float`: its blocks of 32, then the remainder added in order.
  NUMERICAL_TARGET("avx2,fma") inline float avx2_dot(const float* x,
                                                     const float* y,
                                                     std::size_t  count) noexcept
  {
    const std::size_t done  {count - (count % 32)};
    float             total {avx2_dot_blocks(x, y, done)};
    for (std::size_t i {done}; i < count; ++i) {
      total += x[i] * y[i];
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return total;
  }

  /*!
   * \brief The partial sums \ref avx2_dot_blocks (\p Partials 16) or \ref avx512_dot_blocks (32) keep for each of
   *        \p rows rows of the column-major \p a (column `j` at `a + j · lda`) times \p x, over its first \p done
   *        columns (a multiple of \p Partials): element `j` of a row into its partial `j mod Partials` by one fused
   *        multiply-add, in ascending order, as the kernels' lanes take it; \p sums (`Partials × rows`, partial `p` of row
   *        `r` at `p · rows + r`) zero before. A column's rows lie together, so four rows a vector.
   */
  template <std::size_t Partials>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_column_partials(const double* a,
                                                                std::size_t   lda,
                                                                std::size_t   rows,
                                                                std::size_t   done,
                                                                const double* x,
                                                                double*       sums) noexcept
  {
    for (std::size_t j {0}; j < done; ++j) {
      double* const       s   {sums + ((j % Partials) * rows)};
      const double* const col {a + (j * lda)};
      const __m256d       xv  {_mm256_set1_pd(x[j])};
      std::size_t         r   {0};
      for (; r + 4 <= rows; r += 4) {
        _mm256_storeu_pd(s + r, _mm256_fmadd_pd(_mm256_loadu_pd(col + r), xv, _mm256_loadu_pd(s + r)));
      }
      for (; r < rows; ++r) {
        s[r] = std::fma(col[r], x[j], s[r]);
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  /*!
   * \brief \ref avx2_dot of each row of the column-major \p a with \p x, into \p y, from \ref avx2_column_partials'
   *        sixteen partial sums: each vector's four summed as \ref avx2_dot_blocks sums them, the vectors in order from +0
   *        (none: +0), then the columns past \p done added in order by the same statement as the kernel's, so its
   *        bits.
   */
  NUMERICAL_TARGET("avx2,fma") inline void avx2_columns_finish(const double* a,
                                                               std::size_t   lda,
                                                               std::size_t   rows,
                                                               std::size_t   done,
                                                               std::size_t   cols,
                                                               const double* x,
                                                               const double* sums,
                                                               double*       y) noexcept
  {
    constexpr std::size_t width {4};
    constexpr std::size_t lanes {4};
    for (std::size_t r {0}; r < rows; ++r) {
      double total {0.0};
      for (std::size_t l {0}; done != 0 && l < lanes; ++l) {
        const double* const part {sums + (l * width * rows) + r}; // partial l·width + k of row r at part[k · rows]
        total += (part[0] + part[rows]) + (part[2 * rows] + part[3 * rows]);
      }
      y[r] = total;
    }
    for (std::size_t j {done}; j < cols; ++j) {
      const double* const col {a + (j * lda)};
      for (std::size_t r {0}; r < rows; ++r) {
        y[r] += col[r] * x[j];
      }
    }
  }

  /*!
   * \brief \ref avx2_columns_finish for \ref avx512_dot: thirty-two partial sums, each register's eight by the kernel's
   *        own reduction (`_mm512_reduce_add_pd`), the registers in order from +0, then the remainder in order.
   */
  NUMERICAL_TARGET("avx512f") inline void avx512_columns_finish(const double* a,
                                                                std::size_t   lda,
                                                                std::size_t   rows,
                                                                std::size_t   done,
                                                                std::size_t   cols,
                                                                const double* x,
                                                                const double* sums,
                                                                double*       y) noexcept
  {
    constexpr std::size_t width {8};
    constexpr std::size_t lanes {4};
    for (std::size_t r {0}; r < rows; ++r) {
      double total {0.0};
      for (std::size_t l {0}; done != 0 && l < lanes; ++l) {
        alignas(64) std::array<double, width> part {};
        for (std::size_t k {0}; k < width; ++k) {
          part[k] = sums[(((l * width) + k) * rows) + r];
        }
        total += _mm512_reduce_add_pd(_mm512_load_pd(part.data()));
      }
      y[r] = total;
    }
    for (std::size_t j {done}; j < cols; ++j) {
      const double* const col {a + (j * lda)};
      for (std::size_t r {0}; r < rows; ++r) {
        y[r] += col[r] * x[j];
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  /*!
   * \brief The partial sums `avx2_dot_blocks` (\p Partials 32) or `avx512_dot_blocks` (64) of `float` keep for each of
   *        \p rows rows of the column-major \p a times \p x, over its first \p done columns: as the `double` form, eight
   *        rows a vector.
   */
  template <std::size_t Partials>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_column_partials(const float* a,
                                                                std::size_t  lda,
                                                                std::size_t  rows,
                                                                std::size_t  done,
                                                                const float* x,
                                                                float*       sums) noexcept
  {
    for (std::size_t j {0}; j < done; ++j) {
      float* const       s   {sums + ((j % Partials) * rows)};
      const float* const col {a + (j * lda)};
      const __m256       xv  {_mm256_set1_ps(x[j])};
      std::size_t        r   {0};
      for (; r + 8 <= rows; r += 8) {
        _mm256_storeu_ps(s + r, _mm256_fmadd_ps(_mm256_loadu_ps(col + r), xv, _mm256_loadu_ps(s + r)));
      }
      for (; r < rows; ++r) {
        s[r] = std::fma(col[r], x[j], s[r]);
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief `avx2_dot` of `float` of each row of the column-major \p a with \p x, into \p y, from
  //!        \ref avx2_column_partials' thirty-two partial sums: each vector's eight summed as `avx2_dot_blocks` sums
  //!        them, the vectors in order from +0 (none: +0), then the columns past \p done in order, so its bits.
  NUMERICAL_TARGET("avx2,fma") inline void avx2_columns_finish(const float* a,
                                                               std::size_t  lda,
                                                               std::size_t  rows,
                                                               std::size_t  done,
                                                               std::size_t  cols,
                                                               const float* x,
                                                               const float* sums,
                                                               float*       y) noexcept
  {
    constexpr std::size_t width {8};
    constexpr std::size_t lanes {4};
    for (std::size_t r {0}; r < rows; ++r) {
      float total {0.0F};
      for (std::size_t l {0}; done != 0 && l < lanes; ++l) {
        const float* const part {sums + (l * width * rows) + r}; // partial l·width + k of row r at part[k · rows]
        total += ((part[0] + part[rows]) + (part[2 * rows] + part[3 * rows]))
                 + ((part[4 * rows] + part[5 * rows]) + (part[6 * rows] + part[7 * rows]));
      }
      y[r] = total;
    }
    for (std::size_t j {done}; j < cols; ++j) {
      const float* const col {a + (j * lda)};
      for (std::size_t r {0}; r < rows; ++r) {
        y[r] += col[r] * x[j];
      }
    }
  }

  //! \brief \ref avx2_columns_finish for `avx512_dot` of `float`: sixty-four partial sums, each register's sixteen by
  //!        the kernel's own `_mm512_reduce_add_ps`.
  NUMERICAL_TARGET("avx512f") inline void avx512_columns_finish(const float* a,
                                                                std::size_t  lda,
                                                                std::size_t  rows,
                                                                std::size_t  done,
                                                                std::size_t  cols,
                                                                const float* x,
                                                                const float* sums,
                                                                float*       y) noexcept
  {
    constexpr std::size_t width {16};
    constexpr std::size_t lanes {4};
    for (std::size_t r {0}; r < rows; ++r) {
      float total {0.0F};
      for (std::size_t l {0}; done != 0 && l < lanes; ++l) {
        alignas(64) std::array<float, width> part {};
        for (std::size_t k {0}; k < width; ++k) {
          part[k] = sums[(((l * width) + k) * rows) + r];
        }
        total += _mm512_reduce_add_ps(_mm512_load_ps(part.data()));
      }
      y[r] = total;
    }
    for (std::size_t j {done}; j < cols; ++j) {
      const float* const col {a + (j * lda)};
      for (std::size_t r {0}; r < rows; ++r) {
        y[r] += col[r] * x[j];
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  /*!
   * \brief \ref avx2_dot of each of \p rows rows of \p m (\p ld apart) with \p x, into \p out, four rows at a time: each
   *        row's blocks, then the four rows' remainders added in order together, their chains interleaved where one
   *        row's waited on its last addition. Each row's sum the kernel's. \p Width the blocks' elements, \p Blocks
   *        their kernel.
   */
  template <typename T, std::size_t Width, T (*Blocks)(const T*, const T*, std::size_t) noexcept>
  NUMERICAL_TARGET("avx2,fma") void interleaved_row_dots(const T*    m,
                                                         std::size_t ld,
                                                         std::size_t rows,
                                                         const T*    x,
                                                         std::size_t count,
                                                         T*          out) noexcept
  {
    const std::size_t done {count - (count % Width)};
    std::size_t       r    {0};
    for (; r + 4 <= rows; r += 4) {
      const T* const m0 {m + (r * ld)};
      T              t0 {done == 0 ? T {} : Blocks(m0, x, done)};
      T              t1 {done == 0 ? T {} : Blocks(m0 + ld, x, done)};
      T              t2 {done == 0 ? T {} : Blocks(m0 + (2 * ld), x, done)};
      T              t3 {done == 0 ? T {} : Blocks(m0 + (3 * ld), x, done)};
      for (std::size_t i {done}; i < count; ++i) {
        t0 += m0[i] * x[i];
        t1 += m0[ld + i] * x[i];
        t2 += m0[(2 * ld) + i] * x[i];
        t3 += m0[(3 * ld) + i] * x[i];
      }
      out[r]     = t0;
      out[r + 1] = t1;
      out[r + 2] = t2;
      out[r + 3] = t3;
    }
    for (; r < rows; ++r) {
      T total {Blocks(m + (r * ld), x, done)};
      for (std::size_t i {done}; i < count; ++i) {
        total += m[(r * ld) + i] * x[i];
      }
      out[r] = total;
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
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

  //! \brief \ref avx2_complex_dot of each of \p rows rows of \p m (\p ld apart) with \p x, into \p out: the path read
  //!        once and the kernel inlined in the loop, each row's sums its own (a row of a few elements paid a call and the
  //!        path's load: 652 × 3 complex `double` times a vector, six AVX2 cores, 7.4 → 3.3 µs).
  template <typename C>
  NUMERICAL_TARGET("avx2,fma") void avx2_complex_row_dots(const C*    m,
                                                          std::size_t ld,
                                                          std::size_t rows,
                                                          const C*    x,
                                                          std::size_t count,
                                                          C*          out) noexcept
  {
    for (std::size_t r {0}; r < rows; ++r) {
      out[r] = avx2_complex_dot(m + (r * ld), x, count);
    }
  }

  //! \brief A register of `double` or `float` from \p p (`_mm256_loadu_pd`, `_mm256_loadu_ps`).
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_load(const double* p) noexcept
  {
    return _mm256_loadu_pd(p);
  }

  //! \brief \ref avx2_load of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_load(const float* p) noexcept
  {
    return _mm256_loadu_ps(p);
  }

  //! \brief \p v stored at \p p.
  NUMERICAL_TARGET("avx2,fma") inline void avx2_store(double* p,
                                                      __m256d v) noexcept
  {
    _mm256_storeu_pd(p, v);
  }

  //! \brief \ref avx2_store of `float`.
  NUMERICAL_TARGET("avx2,fma") inline void avx2_store(float* p,
                                                      __m256 v) noexcept
  {
    _mm256_storeu_ps(p, v);
  }

  //! \brief `m · v + s`, one fused multiply-add a lane.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_fused(__m256d m,
                                                         __m256d v,
                                                         __m256d s) noexcept
  {
    return _mm256_fmadd_pd(m, v, s);
  }

  //! \brief \ref avx2_fused of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_fused(__m256 m,
                                                        __m256 v,
                                                        __m256 s) noexcept
  {
    return _mm256_fmadd_ps(m, v, s);
  }

  //! \brief \p first, \p second, \p first, … across a register: a complex number's parts, or them swapped.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_pairs(double first,
                                                         double second) noexcept
  {
    return _mm256_setr_pd(first, second, first, second);
  }

  //! \brief \ref avx2_pairs of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_pairs(float first,
                                                        float second) noexcept
  {
    return _mm256_setr_ps(first, second, first, second, first, second, first, second);
  }

  /*!
   * \brief The slots \ref avx2_complex_dot keeps for each of \p rows complex rows of the column-major \p a (column `j`
   *        at `a + j · lda`) times \p x, over its first \p done columns (a multiple of \p Slots, two complex numbers a
   *        register's lane for `double`, four for `float`): element `j` of a row into slot `j mod Slots`, its parts by one
   *        fused multiply-add each with `x[j]`'s, then with them swapped, as the kernel's lanes take it; \p sums
   *        (`2 · Slots` vectors of `2 · rows` parts, slot `q`'s `x · y` sums of row `r` at `2 · (q · 2 · rows + r)`, its
   *        crossed ones `2 · rows` further) zero before. A column's rows lie together, their parts interleaved as a
   *        row's are.
   */
  template <std::size_t Slots,
            typename R>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_complex_column_slots(const R*    a,
                                                                     std::size_t lda,
                                                                     std::size_t rows,
                                                                     std::size_t done,
                                                                     const R*    x,
                                                                     R*          sums) noexcept
  {
    // Each slot its own sum over its columns in ascending order: a slot at a time over four registers of rows, its sums
    // in registers, each column read once and each slot written once.
    using vec = decltype(avx2_load(a));           // __m256d or __m256
    constexpr std::size_t width {32 / sizeof(R)}; // parts in a register
    constexpr std::size_t tile  {4 * width};      // parts of the rows a slot takes at once
    const std::size_t     parts {2 * rows};
    for (std::size_t q {0}; q < Slots; ++q) {
      R* const    same  {sums + (2 * q * parts)};
      R* const    cross {same + parts};
      std::size_t p     {0};
      for (; p + tile <= parts; p += tile) {
        using held = std::conditional_t<std::is_same_v<R, double>, ymm_register, ymm_register_f32>; // a register in an array
        std::array<held, 4> s {};
        std::array<held, 4> c {};
        for (std::size_t j {q}; j < done; j += Slots) {
          const R* const col {a + (2 * j * lda) + p};
          const vec      xs  {avx2_pairs(x[2 * j], x[(2 * j) + 1])};
          const vec      xc  {avx2_pairs(x[(2 * j) + 1], x[2 * j])};
          NUMERICAL_UNROLL_TILE
          for (std::size_t t {0}; t < 4; ++t) {
            const vec mv {avx2_load(col + (t * width))};
            s[t].v = avx2_fused(mv, xs, s[t].v);
            c[t].v = avx2_fused(mv, xc, c[t].v);
          }
        }
        for (std::size_t t {0}; t < 4; ++t) {
          avx2_store(same + p + (t * width), s[t].v);
          avx2_store(cross + p + (t * width), c[t].v);
        }
      }
      for (; p < parts; p += 2) { // the rows past whole tiles: the same fused multiply-adds, one at a time
        for (std::size_t j {q}; j < done; j += Slots) {
          const R* const col {a + (2 * j * lda) + p};
          const R        xr  {x[2 * j]};
          const R        xi  {x[(2 * j) + 1]};
          same[p]      = std::fma(col[0], xr, same[p]);
          same[p + 1]  = std::fma(col[1], xi, same[p + 1]);
          cross[p]     = std::fma(col[0], xi, cross[p]);
          cross[p + 1] = std::fma(col[1], xr, cross[p + 1]);
        }
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  /*!
   * \brief \ref avx2_complex_dot of each complex row of the column-major \p a with \p x, into \p y, from
   *        \ref avx2_complex_column_slots' sums: each lane's slots summed as the kernel sums a register's (in pairs,
   *        `float`'s pairs then summed), the lanes in order from +0, then the columns past \p done added by the kernel's
   *        statements, and the parts' difference and sum; so its bits.
   */
  template <std::size_t Slots,
            typename R>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_complex_columns_finish(const R*                 a,
                                                                       std::size_t              lda,
                                                                       std::size_t              rows,
                                                                       std::size_t              done,
                                                                       std::size_t              cols,
                                                                       const R*                 x,
                                                                       const R*                 sums,
                                                                       std::complex<R>*         y) noexcept
  {
    constexpr std::size_t lanes {4};
    constexpr std::size_t per   {Slots / lanes}; // the slots of one register: two complex numbers, or four
    const auto            slot  {[&](std::size_t q, std::size_t kind, std::size_t r, std::size_t part) {
                                   return sums[(2 * q * 2 * rows) + (kind * 2 * rows) + (2 * r) + part];
                                 }};
    std::vector<R>        acc (4 * rows);
    for (std::size_t r {0}; r < rows; ++r) {
      R rr {0};
      R ii {0};
      R ri {0};
      R ir {0};
      for (std::size_t l {0}; done != 0 && l < lanes; ++l) {
        const std::size_t q {l * per};
        if constexpr (per == 2) {
          rr += slot(q, 0, r, 0) + slot(q + 1, 0, r, 0);
          ii += slot(q, 0, r, 1) + slot(q + 1, 0, r, 1);
          ri += slot(q, 1, r, 0) + slot(q + 1, 1, r, 0);
          ir += slot(q, 1, r, 1) + slot(q + 1, 1, r, 1);
        }
        else {
          rr += (slot(q, 0, r, 0) + slot(q + 1, 0, r, 0)) + (slot(q + 2, 0, r, 0) + slot(q + 3, 0, r, 0));
          ii += (slot(q, 0, r, 1) + slot(q + 1, 0, r, 1)) + (slot(q + 2, 0, r, 1) + slot(q + 3, 0, r, 1));
          ri += (slot(q, 1, r, 0) + slot(q + 1, 1, r, 0)) + (slot(q + 2, 1, r, 0) + slot(q + 3, 1, r, 0));
          ir += (slot(q, 1, r, 1) + slot(q + 1, 1, r, 1)) + (slot(q + 2, 1, r, 1) + slot(q + 3, 1, r, 1));
        }
      }
      acc[4 * r]       = rr;
      acc[(4 * r) + 1] = ii;
      acc[(4 * r) + 2] = ri;
      acc[(4 * r) + 3] = ir;
    }
    for (std::size_t j {done}; j < cols; ++j) {
      const R* const col {a + (2 * j * lda)};
      const R        xr  {x[2 * j]};
      const R        xi  {x[(2 * j) + 1]};
      for (std::size_t r {0}; r < rows; ++r) {
        acc[4 * r]       += col[2 * r] * xr;
        acc[(4 * r) + 1] += col[(2 * r) + 1] * xi;
        acc[(4 * r) + 2] += col[2 * r] * xi;
        acc[(4 * r) + 3] += col[(2 * r) + 1] * xr;
      }
    }
    for (std::size_t r {0}; r < rows; ++r) {
      y[r] = {acc[4 * r] - acc[(4 * r) + 1], acc[(4 * r) + 2] + acc[(4 * r) + 3]};
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief \p sum plus `re − im` in its real parts and `re + im` in its imaginary ones, \p im's parts exchanged first:
  //!        the complex tiles' `Re a · b + i Im a · b` added to C (\ref avx2_full_tile_c64).
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_crossed_sum(__m256d sum,
                                                               __m256d re,
                                                               __m256d im) noexcept
  {
    return _mm256_add_pd(sum, _mm256_addsub_pd(re, _mm256_permute_pd(im, 0b0101)));
  }

  //! \brief \ref avx2_crossed_sum of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_crossed_sum(__m256 sum,
                                                              __m256 re,
                                                              __m256 im) noexcept
  {
    return _mm256_add_ps(sum, _mm256_addsub_ps(re, _mm256_permute_ps(im, 0xB1)));
  }

  //! \brief The complex numbers at \p at and \p apart parts on (two of them), their parts interleaved: a register of
  //!        complex numbers from as many columns whose elements lie together.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_complex_columns(const double* at,
                                                                   std::size_t   apart) noexcept
  {
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(at)), _mm_loadu_pd(at + apart), 1);
  }

  //! \brief \ref avx2_complex_columns of `float`: four complex numbers, \p apart parts from one to the next.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_complex_columns(const float* at,
                                                                  std::size_t  apart) noexcept
  {
    const auto pair {[](const float* lo, const float* hi) {
                       return _mm_castpd_ps(_mm_loadh_pd(_mm_load_sd(reinterpret_cast<const double*>(lo)), reinterpret_cast<const double*>(hi))); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex float's two parts loaded as 64 bits, unaligned loads
                     }};
    return _mm256_insertf128_ps(_mm256_castps128_ps256(pair(at, at + apart)), pair(at + (2 * apart), at + (3 * apart)), 1);
  }

  /*!
   * \brief The complex vector \p x (\p inner elements) times columns `0` to \p count of B, into \p y, each element
   *        summed as the complex tiles sum it (\ref avx2_full_tile_c64, \ref avx512_full_tile_c64 and their `_c32`):
   *        by panels of \p depth, `Re x · B` and `Im x · B` from zero, one fused multiply-add a part and a term, then
   *        `Re x·Re B − Im x·Im B` and `Re x·Im B + Im x·Re B` added to the element, from +0. Every operand's parts
   *        interleaved; the parts of B's element `[k][j]` at `b + 2 · ((k · row_step) + (j · column_step))`: a row's
   *        columns by registers, read where they lie with \p column_step 1, else each register's complex numbers from
   *        as many columns (\ref avx2_complex_columns).
   */
  template <typename R>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_complex_row_panels(const R*    x,
                                                                   const R*    b,
                                                                   std::size_t row_step,
                                                                   std::size_t column_step,
                                                                   std::size_t inner,
                                                                   std::size_t depth,
                                                                   std::size_t count,
                                                                   R*          y) noexcept
  {
    using vec  = decltype(avx2_load(x));                                                        // __m256d or __m256
    using held = std::conditional_t<std::is_same_v<R, double>, ymm_register, ymm_register_f32>; // a register in an array
    constexpr std::size_t width {32 / sizeof(R)};                                               // parts in a register
    constexpr std::size_t tile  {4 * width};                                                    // parts held through a panel
    const std::size_t     parts {2 * count};
    const held            zero  {};
    for (std::size_t pc {0}; pc < inner; pc += depth) { // a panel of B's rows at a time, read once
      const std::size_t end   {std::min(pc + depth, inner)};
      const bool        first {pc == 0};                // y holds nothing before the first panel
      std::size_t       p     {0};
      for (; p + tile <= parts; p += tile) {
        std::array<held, 4> re {};
        std::array<held, 4> im {};
        for (std::size_t k {pc}; k < end; ++k) {
          const R* const row {b + (2 * k * row_step) + (p * column_step)};
          const vec      xr  {avx2_pairs(x[2 * k], x[2 * k])};
          const vec      xi  {avx2_pairs(x[(2 * k) + 1], x[(2 * k) + 1])};
          NUMERICAL_UNROLL_TILE
          for (std::size_t t {0}; t < 4; ++t) {
            const vec bv {column_step == 1 ? avx2_load(row + (t * width)) : avx2_complex_columns(row + (t * width * column_step), 2 * column_step)};
            re[t].v = avx2_fused(xr, bv, re[t].v);
            im[t].v = avx2_fused(xi, bv, im[t].v);
          }
        }
        for (std::size_t t {0}; t < 4; ++t) {
          R* const at {y + p + (t * width)};
          avx2_store(at, avx2_crossed_sum(first ? zero.v : avx2_load(at), re[t].v, im[t].v));
        }
      }
      for (; p < parts; p += 2) { // the columns past whole registers: the same terms, one at a time
        const R* const col {b + (p * column_step)};
        R              rr  {0};
        R              ri  {0};
        R              ir  {0};
        R              ii  {0};
        for (std::size_t k {pc}; k < end; ++k) {
          const R br {col[2 * k * row_step]};
          const R bi {col[(2 * k * row_step) + 1]};
          rr = std::fma(x[2 * k], br, rr);
          ri = std::fma(x[2 * k], bi, ri);
          ir = std::fma(x[(2 * k) + 1], br, ir);
          ii = std::fma(x[(2 * k) + 1], bi, ii);
        }
        y[p]     = (first ? R {0} : y[p]) + (rr - ii);
        y[p + 1] = (first ? R {0} : y[p + 1]) + (ri + ir);
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief `a + b`, lane by lane.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_add(__m256d a,
                                                       __m256d b) noexcept
  {
    return _mm256_add_pd(a, b);
  }

  //! \brief \ref avx2_add of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_add(__m256 a,
                                                      __m256 b) noexcept
  {
    return _mm256_add_ps(a, b);
  }

  //! \brief `a − b`, lane by lane.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_sub(__m256d a,
                                                       __m256d b) noexcept
  {
    return _mm256_sub_pd(a, b);
  }

  //! \brief \ref avx2_sub of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_sub(__m256 a,
                                                      __m256 b) noexcept
  {
    return _mm256_sub_ps(a, b);
  }

  //! \brief The complex numbers of \p v times i: each `(−im, re)`, exact.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_times_i(__m256d v) noexcept
  {
    return _mm256_xor_pd(_mm256_permute_pd(v, 0b0101), _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0));
  }

  //! \brief \ref avx2_times_i of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_times_i(__m256 v) noexcept
  {
    return _mm256_xor_ps(_mm256_permute_ps(v, 0xB1), _mm256_setr_ps(-0.0F, 0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F, 0.0F));
  }

  //! \brief `w · v` for each complex number of \p v, \p re and \p im `w`'s parts in every lane: `w.im · v`'s parts
  //!        exchanged and rounded, then `w.re · v` fused with them (subtracted from the real part, added to the
  //!        imaginary), so `fused_product`'s arithmetic.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_twiddled(__m256d re,
                                                            __m256d im,
                                                            __m256d v) noexcept
  {
    return _mm256_fmaddsub_pd(re, v, _mm256_mul_pd(im, _mm256_permute_pd(v, 0b0101)));
  }

  //! \brief \ref avx2_twiddled of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_twiddled(__m256 re,
                                                           __m256 im,
                                                           __m256 v) noexcept
  {
    return _mm256_fmaddsub_ps(re, v, _mm256_mul_ps(im, _mm256_permute_ps(v, 0xB1)));
  }

  //! \brief The real parts of the complex numbers in \p v, each in both of its lanes.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_real_parts(__m256d v) noexcept
  {
    return _mm256_movedup_pd(v);
  }

  //! \brief \ref avx2_real_parts of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_real_parts(__m256 v) noexcept
  {
    return _mm256_moveldup_ps(v);
  }

  //! \brief The imaginary parts of the complex numbers in \p v, each in both of its lanes.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_imaginary_parts(__m256d v) noexcept
  {
    return _mm256_permute_pd(v, 0b1111);
  }

  //! \brief \ref avx2_imaginary_parts of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_imaginary_parts(__m256 v) noexcept
  {
    return _mm256_movehdup_ps(v);
  }

  //! \brief The outputs of two radix-4 butterflies, \p y0 to \p y3 their `j`-th outputs (a complex `double` of each per
  //!        register), stored butterfly by butterfly at \p to: its four outputs together, then the next's.
  NUMERICAL_TARGET("avx2,fma") inline void avx2_store_butterflies(double* to,
                                                                  __m256d y0,
                                                                  __m256d y1,
                                                                  __m256d y2,
                                                                  __m256d y3) noexcept
  {
    _mm256_storeu_pd(to, _mm256_permute2f128_pd(y0, y1, 0x20));
    _mm256_storeu_pd(to + 4, _mm256_permute2f128_pd(y2, y3, 0x20));
    _mm256_storeu_pd(to + 8, _mm256_permute2f128_pd(y0, y1, 0x31));
    _mm256_storeu_pd(to + 12, _mm256_permute2f128_pd(y2, y3, 0x31));
  }

  //! \brief \ref avx2_store_butterflies of four butterflies of complex `float`: a transpose of their 64-bit complex numbers.
  NUMERICAL_TARGET("avx2,fma") inline void avx2_store_butterflies(float* to,
                                                                  __m256 y0,
                                                                  __m256 y1,
                                                                  __m256 y2,
                                                                  __m256 y3) noexcept
  {
    const __m256d t0 {_mm256_unpacklo_pd(_mm256_castps_pd(y0), _mm256_castps_pd(y1))};
    const __m256d t1 {_mm256_unpackhi_pd(_mm256_castps_pd(y0), _mm256_castps_pd(y1))};
    const __m256d t2 {_mm256_unpacklo_pd(_mm256_castps_pd(y2), _mm256_castps_pd(y3))};
    const __m256d t3 {_mm256_unpackhi_pd(_mm256_castps_pd(y2), _mm256_castps_pd(y3))};
    _mm256_storeu_ps(to, _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t2, 0x20)));
    _mm256_storeu_ps(to + 8, _mm256_castpd_ps(_mm256_permute2f128_pd(t1, t3, 0x20)));
    _mm256_storeu_ps(to + 16, _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t2, 0x31)));
    _mm256_storeu_ps(to + 24, _mm256_castpd_ps(_mm256_permute2f128_pd(t1, t3, 0x31)));
  }

  //! \brief `w · v` of complex numbers as \ref avx2_twiddled computes each lane, one at a time.
  template <typename R>
  NUMERICAL_TARGET("avx2,fma") inline std::complex<R> avx2_fused_product(const std::complex<R>& w,
                                                                         const std::complex<R>& v) noexcept
  {
    const R ii {w.imag() * v.imag()};
    const R ir {w.imag() * v.real()};
    return {std::fma(w.real(), v.real(), -ii), std::fma(w.real(), v.imag(), ir)};
  }

  //! \brief The complex numbers of \p v times −i: each `(im, −re)`, exact.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_times_minus_i(__m256d v) noexcept
  {
    return _mm256_xor_pd(_mm256_permute_pd(v, 0b0101), _mm256_setr_pd(0.0, -0.0, 0.0, -0.0));
  }

  //! \brief \ref avx2_times_minus_i of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_times_minus_i(__m256 v) noexcept
  {
    return _mm256_xor_ps(_mm256_permute_ps(v, 0xB1), _mm256_setr_ps(0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F));
  }

  //! \brief `fft.hpp`'s `four_point` over registers of complex numbers: \p v0 to \p v3 into \p y0 to \p y3.
  template <typename V>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_four_point(V  v0,
                                                           V  v1,
                                                           V  v2,
                                                           V  v3,
                                                           V& y0,
                                                           V& y1,
                                                           V& y2,
                                                           V& y3) noexcept
  {
    const V a_plus_c  {avx2_add(v0, v2)};
    const V a_minus_c {avx2_sub(v0, v2)};
    const V b_plus_d  {avx2_add(v1, v3)};
    const V i_b_d     {avx2_times_i(avx2_sub(v1, v3))};
    y0 = avx2_add(a_plus_c, b_plus_d);
    y1 = avx2_sub(a_minus_c, i_b_d);
    y2 = avx2_sub(a_plus_c, b_plus_d);
    y3 = avx2_add(a_minus_c, i_b_d);
  }

  //! \brief A register of \p c in every lane.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_splat(double c) noexcept
  {
    return _mm256_set1_pd(c);
  }

  //! \brief \ref avx2_splat of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_splat(float c) noexcept
  {
    return _mm256_set1_ps(c);
  }

  //! \brief `a · b`, lane by lane.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_mul(__m256d a,
                                                       __m256d b) noexcept
  {
    return _mm256_mul_pd(a, b);
  }

  //! \brief \ref avx2_mul of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_mul(__m256 a,
                                                      __m256 b) noexcept
  {
    return _mm256_mul_ps(a, b);
  }

  //! \brief `c + a · b`, lane by lane, one fused multiply-add.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_fma(__m256d a,
                                                       __m256d b,
                                                       __m256d c) noexcept
  {
    return _mm256_fmadd_pd(a, b, c);
  }

  //! \brief \ref avx2_fma of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_fma(__m256 a,
                                                      __m256 b,
                                                      __m256 c) noexcept
  {
    return _mm256_fmadd_ps(a, b, c);
  }

  //! \brief `s + c · v` of complex numbers and a real \p c, each part one fused multiply-add (`fft.hpp`'s fused
  //!        `scaled_add`).
  template <typename R>
  NUMERICAL_TARGET("avx2,fma") inline std::complex<R> avx2_scaled_add(const std::complex<R>& s,
                                                                      R                      c,
                                                                      const std::complex<R>& v) noexcept
  {
    return {std::fma(c, v.real(), s.real()), std::fma(c, v.imag(), s.imag())};
  }

  //! \brief Output \p k of butterfly \p p of a pass, \p value, stored at \p place: times `w^{kp}` (its factors \p twiddles,
  //!        a run of \p run each power, as \ref avx2_twiddled) unless \p p is 0.
  template <typename R>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_store_twiddled(R*                                                place,
                                                               decltype(avx2_load(static_cast<const R*>(nullptr))) value,
                                                               const std::complex<R>                           * twiddles,
                                                               std::size_t                                       run,
                                                               std::size_t                                       p,
                                                               std::size_t                                       k) noexcept
  {
    if (p != 0) {
      const std::complex<R> w {twiddles[((k - 1) * run) + p - 1]};
      value = avx2_twiddled(avx2_pairs(w.real(), w.real()), avx2_pairs(w.imag(), w.imag()), value);
    }
    avx2_store(place, value);
  }

  /*!
   * \brief One Stockham pass of radix 3 of `fft.hpp`'s (`radix_three_pass_body`), fused, a register of complex numbers
   *        along the stride at a time where it holds one, else one at a time: the same operations in the same order.
   *        \p half_root_three is √3/2.
   */
  template <typename R>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_radix_three_pass(std::size_t            current,
                                                                 std::size_t            stride,
                                                                 const std::complex<R>* in,
                                                                 std::complex<R>*       out,
                                                                 const std::complex<R>* twiddles,
                                                                 R                      half_root_three) noexcept
  {
    using C   = std::complex<R>;
    using vec = decltype(avx2_load(static_cast<const R*>(nullptr))); // __m256d or __m256
    constexpr std::size_t width {32 / sizeof(C)};
    const std::size_t     part  {current / 3};
    const std::size_t     run   {part - 1};
    const std::size_t     apart {stride * part};
    const vec             half  {avx2_splat(R(0.5))};
    const vec             root  {avx2_splat(half_root_three)};
    for (std::size_t p {0}; p < part; ++p) {
      const C* const in_p {in + (stride * p)};
      C* const       to   {out + (stride * 3 * p)};
      std::size_t    q    {0};
      if (stride >= width) {
        const C   w1    {p == 0 ? C {1} : twiddles[p - 1]};
        const C   w2    {p == 0 ? C {1} : twiddles[run + p - 1]};
        const vec w1_re {avx2_pairs(w1.real(), w1.real())};
        const vec w1_im {avx2_pairs(w1.imag(), w1.imag())};
        const vec w2_re {avx2_pairs(w2.real(), w2.real())};
        const vec w2_im {avx2_pairs(w2.imag(), w2.imag())};
        for (; q + width <= stride; q += width) {
          const R* const at {reinterpret_cast<const R*>(in_p + q)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
          const vec      x0 {avx2_load(at)};
          const vec      x1 {avx2_load(at + (2 * apart))};
          const vec      x2 {avx2_load(at + (4 * apart))};
          const vec      t  {avx2_add(x1, x2)};
          const vec      v  {avx2_sub(x0, avx2_mul(half, t))};
          const vec      iu {avx2_times_i(avx2_mul(root, avx2_sub(x1, x2)))};
          vec            y1 {avx2_sub(v, iu)};
          vec            y2 {avx2_add(v, iu)};
          if (p != 0) {
            y1 = avx2_twiddled(w1_re, w1_im, y1);
            y2 = avx2_twiddled(w2_re, w2_im, y2);
          }
          R* const place {reinterpret_cast<R*>(to + q)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
          avx2_store(place, avx2_add(x0, t));
          avx2_store(place + (2 * stride), y1);
          avx2_store(place + (4 * stride), y2);
        }
      }
      for (; q < stride; ++q) { // a stride narrower than a register
        const C x0 {in_p[q]};
        const C x1 {in_p[q + apart]};
        const C x2 {in_p[q + (2 * apart)]};
        const C t  {x1 + x2};
        const C v  {x0 - C {R(0.5) * t.real(), R(0.5) * t.imag()}};
        const C d  {x1 - x2};
        const C u  {half_root_three * d.real(), half_root_three * d.imag()};
        const C iu {-u.imag(), u.real()};
        to[q]              = x0 + t;
        to[q + stride]     = p == 0 ? v - iu : avx2_fused_product(twiddles[p - 1], v - iu);
        to[q + 2 * stride] = p == 0 ? v + iu : avx2_fused_product(twiddles[run + p - 1], v + iu);
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  /*!
   * \brief One Stockham pass of radix 5 of `fft.hpp`'s (`radix_five_pass_body`), fused, as \ref avx2_radix_three_pass:
   *        \p c1, \p c2 the cosines and \p s1, \p s2 the sines of 2π/5 and 4π/5.
   */
  template <typename R>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_radix_five_pass(std::size_t            current,
                                                                std::size_t            stride,
                                                                const std::complex<R>* in,
                                                                std::complex<R>*       out,
                                                                const std::complex<R>* twiddles,
                                                                R                      c1,
                                                                R                      c2,
                                                                R                      s1,
                                                                R                      s2) noexcept
  {
    using C   = std::complex<R>;
    using vec = decltype(avx2_load(static_cast<const R*>(nullptr))); // __m256d or __m256
    constexpr std::size_t width {32 / sizeof(C)};
    const std::size_t     part  {current / 5};
    const std::size_t     run   {part - 1};
    const std::size_t     apart {stride * part};
    const vec             vc1   {avx2_splat(c1)};
    const vec             vc2   {avx2_splat(c2)};
    const vec             vs1   {avx2_splat(s1)};
    const vec             vs2   {avx2_splat(s2)};
    const vec             vms1  {avx2_splat(-s1)};
    for (std::size_t p {0}; p < part; ++p) {
      const C* const in_p {in + (stride * p)};
      C* const       to   {out + (stride * 5 * p)};
      std::size_t    q    {0};
      if (stride >= width) {
        for (; q + width <= stride; q += width) {
          const R* const at    {reinterpret_cast<const R*>(in_p + q)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
          const vec      x0    {avx2_load(at)};
          const vec      x1    {avx2_load(at + (2 * apart))};
          const vec      x2    {avx2_load(at + (4 * apart))};
          const vec      x3    {avx2_load(at + (6 * apart))};
          const vec      x4    {avx2_load(at + (8 * apart))};
          const vec      t1    {avx2_add(x1, x4)};
          const vec      t2    {avx2_add(x2, x3)};
          const vec      t3    {avx2_sub(x1, x4)};
          const vec      t4    {avx2_sub(x2, x3)};
          const vec      a1    {avx2_fma(vc2, t2, avx2_fma(vc1, t1, x0))};
          const vec      a2    {avx2_fma(vc1, t2, avx2_fma(vc2, t1, x0))};
          const vec      ib1   {avx2_times_i(avx2_fma(vs2, t4, avx2_mul(vs1, t3)))};
          const vec      ib2   {avx2_times_i(avx2_fma(vms1, t4, avx2_mul(vs2, t3)))};
          R* const       place {reinterpret_cast<R*>(to + q)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
          avx2_store(place, avx2_add(avx2_add(x0, t1), t2));
          avx2_store_twiddled(place + (2 * stride), avx2_sub(a1, ib1), twiddles, run, p, 1);
          avx2_store_twiddled(place + (4 * stride), avx2_sub(a2, ib2), twiddles, run, p, 2);
          avx2_store_twiddled(place + (6 * stride), avx2_add(a2, ib2), twiddles, run, p, 3);
          avx2_store_twiddled(place + (8 * stride), avx2_add(a1, ib1), twiddles, run, p, 4);
        }
      }
      for (; q < stride; ++q) { // a stride narrower than a register
        const C                x0  {in_p[q]};
        const C                x1  {in_p[q + apart]};
        const C                x2  {in_p[q + (2 * apart)]};
        const C                x3  {in_p[q + (3 * apart)]};
        const C                x4  {in_p[q + (4 * apart)]};
        const C                t1  {x1 + x4};
        const C                t2  {x2 + x3};
        const C                t3  {x1 - x4};
        const C                t4  {x2 - x3};
        const C                a1  {avx2_scaled_add(avx2_scaled_add(x0, c1, t1), c2, t2)};
        const C                a2  {avx2_scaled_add(avx2_scaled_add(x0, c2, t1), c1, t2)};
        const C                b1  {avx2_scaled_add(C {s1 * t3.real(), s1 * t3.imag()}, s2, t4)};
        const C                b2  {avx2_scaled_add(C {s2 * t3.real(), s2 * t3.imag()}, -s1, t4)};
        const C                ib1 {-b1.imag(), b1.real()};
        const C                ib2 {-b2.imag(), b2.real()};
        const std::array<C, 4> y   {a1 - ib1, a2 - ib2, a2 + ib2, a1 + ib1};
        to[q] = (x0 + t1) + t2;
        for (std::size_t j {1}; j < 5; ++j) {
          to[q + (j * stride)] = p == 0 ? y[j - 1] : avx2_fused_product(twiddles[((j - 1) * run) + p - 1], y[j - 1]);
        }
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief The even lanes of \p even, the odd ones of \p odd: each complex number's real part from one, imaginary from
  //!        the other.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_parts_of(__m256d even,
                                                            __m256d odd) noexcept
  {
    return _mm256_blend_pd(even, odd, 0b1010);
  }

  //! \brief \ref avx2_parts_of of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_parts_of(__m256 even,
                                                           __m256 odd) noexcept
  {
    return _mm256_blend_ps(even, odd, 0b10101010);
  }

  //! \brief \p v with each complex number's imaginary part negated, exact.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_conjugated(__m256d v) noexcept
  {
    return _mm256_xor_pd(v, _mm256_setr_pd(0.0, -0.0, 0.0, -0.0));
  }

  //! \brief \ref avx2_conjugated of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_conjugated(__m256 v) noexcept
  {
    return _mm256_xor_ps(v, _mm256_setr_ps(0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F, 0.0F, -0.0F));
  }

  //! \brief The complex numbers of \p v, their parts exchanged.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_exchanged(__m256d v) noexcept
  {
    return _mm256_permute_pd(v, 0b0101);
  }

  //! \brief \ref avx2_exchanged of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_exchanged(__m256 v) noexcept
  {
    return _mm256_permute_ps(v, 0xB1);
  }

  /*!
   * \brief One Stockham pass of radix 8 of `fft.hpp`'s (`radix_eight_pass_body`), fused, a register of complex numbers
   *        along the stride at a time (the stride a power of two of at least 4: powers of two run first), the same
   *        operations in the same order. \p half_root_two is √½.
   */
  template <typename R>
  NUMERICAL_TARGET("avx2,fma") inline void avx2_radix_eight_pass(std::size_t            current,
                                                                 std::size_t            stride,
                                                                 const std::complex<R>* in,
                                                                 std::complex<R>*       out,
                                                                 const std::complex<R>* twiddles,
                                                                 R                      half_root_two) noexcept
  {
    using vec = decltype(avx2_load(static_cast<const R*>(nullptr))); // __m256d or __m256
    constexpr std::size_t width {32 / sizeof(std::complex<R>)};
    const std::size_t     part  {current / 8};
    const std::size_t     run   {part - 1};
    const std::size_t     apart {2 * stride * part};              // parts between a butterfly's inputs
    const vec             root  {avx2_splat(half_root_two)};
    const R* const        from  {reinterpret_cast<const R*>(in)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
    R* const              into  {reinterpret_cast<R*>(out)};      // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    for (std::size_t p {0}; p < part; ++p) {
      const R* const in_p {from + (2 * stride * p)};
      R* const       to   {into + (2 * stride * 8 * p)};
      for (std::size_t q {0}; q + width <= stride; q += width) {
        const R* const at {in_p + (2 * q)};
        const vec      x0 {avx2_load(at)};
        const vec      x1 {avx2_load(at + apart)};
        const vec      x2 {avx2_load(at + (2 * apart))};
        const vec      x3 {avx2_load(at + (3 * apart))};
        const vec      x4 {avx2_load(at + (4 * apart))};
        const vec      x5 {avx2_load(at + (5 * apart))};
        const vec      x6 {avx2_load(at + (6 * apart))};
        const vec      x7 {avx2_load(at + (7 * apart))};
        const vec      b1 {avx2_sub(x1, x5)};
        const vec      b3 {avx2_sub(x3, x7)};
        const vec      s1 {avx2_add(b1, avx2_exchanged(b1))}; // re + im in both lanes
        const vec      d1 {avx2_sub(b1, avx2_exchanged(b1))}; // re − im, im − re
        const vec      s3 {avx2_add(b3, avx2_exchanged(b3))};
        const vec      d3 {avx2_sub(avx2_exchanged(b3), b3)}; // im − re, re − im
        const vec      w1 {avx2_mul(root, avx2_parts_of(s1, d1))};
        const vec      w3 {avx2_conjugated(avx2_mul(root, avx2_parts_of(d3, s3)))};
        vec            e0 {};
        vec            e1 {};
        vec            e2 {};
        vec            e3 {};
        vec            o0 {};
        vec            o1 {};
        vec            o2 {};
        vec            o3 {};
        avx2_four_point(avx2_add(x0, x4), avx2_add(x1, x5), avx2_add(x2, x6), avx2_add(x3, x7), e0, e1, e2, e3);
        avx2_four_point(avx2_sub(x0, x4), w1, avx2_times_minus_i(avx2_sub(x2, x6)), w3, o0, o1, o2, o3);
        R* const place {to + (2 * q)};
        avx2_store(place, e0);
        avx2_store_twiddled(place + (2 * stride), o0, twiddles, run, p, 1);
        avx2_store_twiddled(place + (4 * stride), e1, twiddles, run, p, 2);
        avx2_store_twiddled(place + (6 * stride), o1, twiddles, run, p, 3);
        avx2_store_twiddled(place + (8 * stride), e2, twiddles, run, p, 4);
        avx2_store_twiddled(place + (10 * stride), o2, twiddles, run, p, 5);
        avx2_store_twiddled(place + (12 * stride), e3, twiddles, run, p, 6);
        avx2_store_twiddled(place + (14 * stride), o3, twiddles, run, p, 7);
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
  }

  //! \brief The complex numbers of \p v in the reverse order.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_reversed(__m256d v) noexcept
  {
    return _mm256_permute2f128_pd(v, v, 0x01);
  }

  //! \brief \ref avx2_reversed of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_reversed(__m256 v) noexcept
  {
    return _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(v), 0x1B));
  }

  //! \brief \p v negated, exact.
  NUMERICAL_TARGET("avx2,fma") inline __m256d avx2_negated(__m256d v) noexcept
  {
    return _mm256_xor_pd(v, _mm256_set1_pd(-0.0));
  }

  //! \brief \ref avx2_negated of `float`.
  NUMERICAL_TARGET("avx2,fma") inline __m256 avx2_negated(__m256 v) noexcept
  {
    return _mm256_xor_ps(v, _mm256_set1_ps(-0.0F));
  }

  /*!
   * \brief `fft.hpp`'s recombination of a real transform of `2h` points (`real_forward_body`) a register at a time:
   *        `X_k = E_k + w^k·O_k` from \p z (`Z`, the transform of the `h` pairs) and \p roots (`w^k`) into \p X, for
   *        `k` from 1 while a register fits below \p h, the same operations in the same order (`Z_{h−k}` read as a
   *        register reversed). Returns the first `k` left.
   */
  template <typename R>
  NUMERICAL_TARGET("avx2,fma") inline std::size_t avx2_real_forward_registers(std::size_t            h,
                                                                              const std::complex<R>* z,
                                                                              const std::complex<R>* roots,
                                                                              std::complex<R>*       X) noexcept
  {
    using vec = decltype(avx2_load(static_cast<const R*>(nullptr))); // __m256d or __m256
    constexpr std::size_t width {32 / sizeof(std::complex<R>)};
    const vec             half  {avx2_splat(R(0.5))};
    const R* const        from  {reinterpret_cast<const R*>(z)};     // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
    const R* const        root  {reinterpret_cast<const R*>(roots)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    R* const              into  {reinterpret_cast<R*>(X)};           // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    std::size_t           k     {1};
    for (; k + width <= h; k += width) {
      const vec a    {avx2_load(from + (2 * k))};
      const vec b    {avx2_conjugated(avx2_reversed(avx2_load(from + (2 * (h - k - (width - 1))))))};
      const vec even {avx2_mul(avx2_add(a, b), half)};
      const vec odd  {avx2_mul(avx2_times_minus_i(avx2_sub(a, b)), half)};
      const vec w    {avx2_load(root + (2 * k))};
      avx2_store(into + (2 * k), avx2_add(even, avx2_twiddled(avx2_real_parts(w), avx2_imaginary_parts(w), odd)));
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return k;
  }

  /*!
   * \brief `fft.hpp`'s recombination before an inverse real transform of `2h` points (`real_inverse_body`) a register
   *        at a time: `conj Z_k = conj(E_k + i·O_k)`, `O_k = conj(w^k)·(X_k − conj X_{h−k})/2`, from \p X and \p roots
   *        (`w^k`) into \p z, for `k` from 1 while a register fits below \p h, the same operations in the same order.
   *        Returns the first `k` left.
   */
  template <typename R>
  NUMERICAL_TARGET("avx2,fma") inline std::size_t avx2_real_inverse_registers(std::size_t            h,
                                                                              const std::complex<R>* X,
                                                                              const std::complex<R>* roots,
                                                                              std::complex<R>*       z) noexcept
  {
    using vec = decltype(avx2_load(static_cast<const R*>(nullptr))); // __m256d or __m256
    constexpr std::size_t width {32 / sizeof(std::complex<R>)};
    const vec             half  {avx2_splat(R(0.5))};
    const R* const        from  {reinterpret_cast<const R*>(X)};     // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex number is its two parts ([complex.numbers])
    const R* const        root  {reinterpret_cast<const R*>(roots)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    R* const              into  {reinterpret_cast<R*>(z)};           // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same
    std::size_t           k     {1};
    for (; k + width <= h; k += width) {
      const vec a    {avx2_load(from + (2 * k))};
      const vec b    {avx2_conjugated(avx2_reversed(avx2_load(from + (2 * (h - k - (width - 1))))))};
      const vec even {avx2_mul(avx2_add(a, b), half)};
      const vec w    {avx2_load(root + (2 * k))};
      const vec odd  {avx2_twiddled(avx2_real_parts(w), avx2_negated(avx2_imaginary_parts(w)), avx2_mul(avx2_sub(a, b), half))};
      avx2_store(into + (2 * k), avx2_conjugated(avx2_add(even, avx2_times_i(odd))));
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the kernel may be SSE alone
    return k;
  }

  // The transform's kernels, written once for AVX2 (__m256d, __m256) and AVX-512 (__m512d, __m512): each operation below
  // is overloaded on the register and built for its unit; the kernels' bodies, built for none, are inlined into the
  // functions built for each (fft_avx2_pass, fft_avx512_pass), where the operations then inline. The same operations
  // lane by lane, whatever the width: the same bits.

  //! \brief AVX2's registers of the transform's kernels.
  struct fft_avx2
  {
    static constexpr bool whole_strides {false}; //!< An odd stride leaves a register half filled.
  };

  //! \brief AVX-512's registers of the transform's kernels.
  struct fft_avx512
  {
    //! \brief Its passes of 3 and 5 take strides its registers fill (any_radix_pass_avx512; AVX2's kernels the others).
    static constexpr bool whole_strides {true};
  };

  /*!
   * \brief A register of complex numbers of \p R that \p Isa holds, as the kernels pass it: in a structure, so that no
   *        function built for no unit (the kernels' bodies) passes a register of a unit by value, and no register's type
   *        is a template's argument (GCC drops its alignment there).
   */
  template <typename Isa,
            typename R>
  struct fft_reg;

  //! \brief Two complex `double` (AVX2).
  template <>
  struct fft_reg<fft_avx2, double>
  {
    __m256d v; //!< The register.
  };

  //! \brief Four complex `float` (AVX2).
  template <>
  struct fft_reg<fft_avx2, float>
  {
    __m256 v; //!< The register.
  };

  //! \brief Four complex `double` (AVX-512).
  template <>
  struct fft_reg<fft_avx512, double>
  {
    __m512d v; //!< The register.
  };

  //! \brief Eight complex `float` (AVX-512).
  template <>
  struct fft_reg<fft_avx512, float>
  {
    __m512 v; //!< The register.
  };

  //! \brief The register at \p p.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, double> fft_load(fft_avx2 /*isa*/,
                                                                         const double* p) noexcept
  {
    return {_mm256_loadu_pd(p)};
  }

  //! \brief \p c in every lane.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, double> fft_splat(fft_avx2 /*isa*/,
                                                                          double c) noexcept
  {
    return {_mm256_set1_pd(c)};
  }

  //! \brief \p v stored at \p p.
  NUMERICAL_TARGET("avx2,fma") inline void fft_store(double                  * p,
                                                     fft_reg<fft_avx2, double> v) noexcept
  {
    _mm256_storeu_pd(p, v.v);
  }

  //! \brief `a + b`, lane by lane.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, double> fft_add(fft_reg<fft_avx2, double> a,
                                                                        fft_reg<fft_avx2, double> b) noexcept
  {
    return {_mm256_add_pd(a.v, b.v)};
  }

  //! \brief `a − b`, lane by lane.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, double> fft_sub(fft_reg<fft_avx2, double> a,
                                                                        fft_reg<fft_avx2, double> b) noexcept
  {
    return {_mm256_sub_pd(a.v, b.v)};
  }

  //! \brief The complex numbers of \p v times i: each `(−im, re)`, exact.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, double> fft_times_i(fft_reg<fft_avx2, double> v) noexcept
  {
    return {avx2_times_i(v.v)};
  }

  //! \brief The complex numbers of \p v times −i: each `(im, −re)`, exact.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, double> fft_times_minus_i(fft_reg<fft_avx2, double> v) noexcept
  {
    return {avx2_times_minus_i(v.v)};
  }

  //! \brief `w · v` of each complex number of \p v, \p re and \p im `w`'s parts in every lane, as `fused_product`.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, double> fft_twiddled(fft_reg<fft_avx2, double> re,
                                                                             fft_reg<fft_avx2, double> im,
                                                                             fft_reg<fft_avx2, double> v) noexcept
  {
    return {avx2_twiddled(re.v, im.v, v.v)};
  }

  //! \brief The real parts of the complex numbers of \p v, each in both of its lanes.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, double> fft_real_parts(fft_reg<fft_avx2, double> v) noexcept
  {
    return {_mm256_movedup_pd(v.v)};
  }

  //! \brief The imaginary parts of the complex numbers of \p v, each in both of its lanes.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, double> fft_imaginary_parts(fft_reg<fft_avx2, double> v) noexcept
  {
    return {_mm256_permute_pd(v.v, 0b1111)};
  }

  //! \brief The outputs of a register of radix-4 butterflies, \p y0 to \p y3 their `j`-th outputs, stored butterfly by butterfly at \p to: a transpose.
  NUMERICAL_TARGET("avx2,fma") inline void fft_store_butterflies(double                  * to,
                                                                 fft_reg<fft_avx2, double> y0,
                                                                 fft_reg<fft_avx2, double> y1,
                                                                 fft_reg<fft_avx2, double> y2,
                                                                 fft_reg<fft_avx2, double> y3) noexcept
  {
    avx2_store_butterflies(to, y0.v, y1.v, y2.v, y3.v);
  }

  //! \brief The register at \p p.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, float> fft_load(fft_avx2 /*isa*/,
                                                                        const float* p) noexcept
  {
    return {_mm256_loadu_ps(p)};
  }

  //! \brief \p c in every lane.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, float> fft_splat(fft_avx2 /*isa*/,
                                                                         float c) noexcept
  {
    return {_mm256_set1_ps(c)};
  }

  //! \brief \p v stored at \p p.
  NUMERICAL_TARGET("avx2,fma") inline void fft_store(float                  * p,
                                                     fft_reg<fft_avx2, float> v) noexcept
  {
    _mm256_storeu_ps(p, v.v);
  }

  //! \brief `a + b`, lane by lane.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, float> fft_add(fft_reg<fft_avx2, float> a,
                                                                       fft_reg<fft_avx2, float> b) noexcept
  {
    return {_mm256_add_ps(a.v, b.v)};
  }

  //! \brief `a − b`, lane by lane.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, float> fft_sub(fft_reg<fft_avx2, float> a,
                                                                       fft_reg<fft_avx2, float> b) noexcept
  {
    return {_mm256_sub_ps(a.v, b.v)};
  }

  //! \brief The complex numbers of \p v times i: each `(−im, re)`, exact.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, float> fft_times_i(fft_reg<fft_avx2, float> v) noexcept
  {
    return {avx2_times_i(v.v)};
  }

  //! \brief The complex numbers of \p v times −i: each `(im, −re)`, exact.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, float> fft_times_minus_i(fft_reg<fft_avx2, float> v) noexcept
  {
    return {avx2_times_minus_i(v.v)};
  }

  //! \brief `w · v` of each complex number of \p v, \p re and \p im `w`'s parts in every lane, as `fused_product`.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, float> fft_twiddled(fft_reg<fft_avx2, float> re,
                                                                            fft_reg<fft_avx2, float> im,
                                                                            fft_reg<fft_avx2, float> v) noexcept
  {
    return {avx2_twiddled(re.v, im.v, v.v)};
  }

  //! \brief The real parts of the complex numbers of \p v, each in both of its lanes.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, float> fft_real_parts(fft_reg<fft_avx2, float> v) noexcept
  {
    return {_mm256_moveldup_ps(v.v)};
  }

  //! \brief The imaginary parts of the complex numbers of \p v, each in both of its lanes.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, float> fft_imaginary_parts(fft_reg<fft_avx2, float> v) noexcept
  {
    return {_mm256_movehdup_ps(v.v)};
  }

  //! \brief The outputs of a register of radix-4 butterflies, \p y0 to \p y3 their `j`-th outputs, stored butterfly by butterfly at \p to: a transpose.
  NUMERICAL_TARGET("avx2,fma") inline void fft_store_butterflies(float                  * to,
                                                                 fft_reg<fft_avx2, float> y0,
                                                                 fft_reg<fft_avx2, float> y1,
                                                                 fft_reg<fft_avx2, float> y2,
                                                                 fft_reg<fft_avx2, float> y3) noexcept
  {
    avx2_store_butterflies(to, y0.v, y1.v, y2.v, y3.v);
  }

  //! \brief The register at \p p.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_load(fft_avx512 /*isa*/,
                                                                          const double* p) noexcept
  {
    return {_mm512_loadu_pd(p)};
  }

  //! \brief \p c in every lane.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_splat(fft_avx512 /*isa*/,
                                                                           double c) noexcept
  {
    return {_mm512_set1_pd(c)};
  }

  //! \brief \p v stored at \p p.
  NUMERICAL_TARGET("avx512f") inline void fft_store(double                    * p,
                                                    fft_reg<fft_avx512, double> v) noexcept
  {
    _mm512_storeu_pd(p, v.v);
  }

  //! \brief `a + b`, lane by lane.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_add(fft_reg<fft_avx512, double> a,
                                                                         fft_reg<fft_avx512, double> b) noexcept
  {
    return {_mm512_add_pd(a.v, b.v)};
  }

  //! \brief `a − b`, lane by lane.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_sub(fft_reg<fft_avx512, double> a,
                                                                         fft_reg<fft_avx512, double> b) noexcept
  {
    return {_mm512_sub_pd(a.v, b.v)};
  }

  //! \brief The complex numbers of \p v times i: each `(−im, re)`, exact.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_times_i(fft_reg<fft_avx512, double> v) noexcept
  {
    return {_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(_mm512_permute_pd(v.v, 0x55)), _mm512_set_epi64(0, INT64_MIN, 0, INT64_MIN, 0, INT64_MIN, 0, INT64_MIN)))};
  }

  //! \brief The complex numbers of \p v times −i: each `(im, −re)`, exact.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_times_minus_i(fft_reg<fft_avx512, double> v) noexcept
  {
    return {_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(_mm512_permute_pd(v.v, 0x55)), _mm512_set_epi64(INT64_MIN, 0, INT64_MIN, 0, INT64_MIN, 0, INT64_MIN, 0)))};
  }

  //! \brief `w · v` of each complex number of \p v, \p re and \p im `w`'s parts in every lane, as `fused_product`.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_twiddled(fft_reg<fft_avx512, double> re,
                                                                              fft_reg<fft_avx512, double> im,
                                                                              fft_reg<fft_avx512, double> v) noexcept
  {
    return {_mm512_fmaddsub_pd(re.v, v.v, _mm512_mul_pd(im.v, _mm512_permute_pd(v.v, 0x55)))};
  }

  //! \brief The real parts of the complex numbers of \p v, each in both of its lanes.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_real_parts(fft_reg<fft_avx512, double> v) noexcept
  {
    return {_mm512_movedup_pd(v.v)};
  }

  //! \brief The imaginary parts of the complex numbers of \p v, each in both of its lanes.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_imaginary_parts(fft_reg<fft_avx512, double> v) noexcept
  {
    return {_mm512_permute_pd(v.v, 0xFF)};
  }

  //! \brief The outputs of a register of radix-4 butterflies, \p y0 to \p y3 their `j`-th outputs, stored butterfly by butterfly at \p to: a transpose.
  NUMERICAL_TARGET("avx512f") inline void fft_store_butterflies(double                    * to,
                                                                fft_reg<fft_avx512, double> y0,
                                                                fft_reg<fft_avx512, double> y1,
                                                                fft_reg<fft_avx512, double> y2,
                                                                fft_reg<fft_avx512, double> y3) noexcept
  {
    const __m512d low01  {_mm512_shuffle_f64x2(y0.v, y1.v, 0x44)};         // y0's butterflies 0 1, y1's 0 1
    const __m512d high01 {_mm512_shuffle_f64x2(y0.v, y1.v, 0xEE)};         // y0's 2 3, y1's 2 3
    const __m512d low23  {_mm512_shuffle_f64x2(y2.v, y3.v, 0x44)};
    const __m512d high23 {_mm512_shuffle_f64x2(y2.v, y3.v, 0xEE)};
    _mm512_storeu_pd(to, _mm512_shuffle_f64x2(low01, low23, 0x88));        // butterfly 0: y0 y1 y2 y3
    _mm512_storeu_pd(to + 8, _mm512_shuffle_f64x2(low01, low23, 0xDD));    // butterfly 1
    _mm512_storeu_pd(to + 16, _mm512_shuffle_f64x2(high01, high23, 0x88)); // butterfly 2
    _mm512_storeu_pd(to + 24, _mm512_shuffle_f64x2(high01, high23, 0xDD)); // butterfly 3
  }

  //! \brief The register at \p p.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_load(fft_avx512 /*isa*/,
                                                                         const float* p) noexcept
  {
    return {_mm512_loadu_ps(p)};
  }

  //! \brief \p c in every lane.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_splat(fft_avx512 /*isa*/,
                                                                          float c) noexcept
  {
    return {_mm512_set1_ps(c)};
  }

  //! \brief \p v stored at \p p.
  NUMERICAL_TARGET("avx512f") inline void fft_store(float                    * p,
                                                    fft_reg<fft_avx512, float> v) noexcept
  {
    _mm512_storeu_ps(p, v.v);
  }

  //! \brief `a + b`, lane by lane.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_add(fft_reg<fft_avx512, float> a,
                                                                        fft_reg<fft_avx512, float> b) noexcept
  {
    return {_mm512_add_ps(a.v, b.v)};
  }

  //! \brief `a − b`, lane by lane.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_sub(fft_reg<fft_avx512, float> a,
                                                                        fft_reg<fft_avx512, float> b) noexcept
  {
    return {_mm512_sub_ps(a.v, b.v)};
  }

  //! \brief The complex numbers of \p v times i: each `(−im, re)`, exact.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_times_i(fft_reg<fft_avx512, float> v) noexcept
  {
    return {_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(_mm512_permute_ps(v.v, 0xB1)), _mm512_set1_epi64(static_cast<std::int64_t>(0x0000000080000000))))};
  }

  //! \brief The complex numbers of \p v times −i: each `(im, −re)`, exact.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_times_minus_i(fft_reg<fft_avx512, float> v) noexcept
  {
    return {_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(_mm512_permute_ps(v.v, 0xB1)), _mm512_set1_epi64(INT64_MIN)))};
  }

  //! \brief `w · v` of each complex number of \p v, \p re and \p im `w`'s parts in every lane, as `fused_product`.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_twiddled(fft_reg<fft_avx512, float> re,
                                                                             fft_reg<fft_avx512, float> im,
                                                                             fft_reg<fft_avx512, float> v) noexcept
  {
    return {_mm512_fmaddsub_ps(re.v, v.v, _mm512_mul_ps(im.v, _mm512_permute_ps(v.v, 0xB1)))};
  }

  //! \brief The real parts of the complex numbers of \p v, each in both of its lanes.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_real_parts(fft_reg<fft_avx512, float> v) noexcept
  {
    return {_mm512_moveldup_ps(v.v)};
  }

  //! \brief The imaginary parts of the complex numbers of \p v, each in both of its lanes.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_imaginary_parts(fft_reg<fft_avx512, float> v) noexcept
  {
    return {_mm512_movehdup_ps(v.v)};
  }

  //! \brief The outputs of a register of radix-4 butterflies, \p y0 to \p y3 their `j`-th outputs, stored butterfly by butterfly at \p to: a transpose.
  NUMERICAL_TARGET("avx512f") inline void fft_store_butterflies(float                    * to,
                                                                fft_reg<fft_avx512, float> y0,
                                                                fft_reg<fft_avx512, float> y1,
                                                                fft_reg<fft_avx512, float> y2,
                                                                fft_reg<fft_avx512, float> y3) noexcept
  {
    const __m512d even01    {_mm512_unpacklo_pd(_mm512_castps_pd(y0.v), _mm512_castps_pd(y1.v))}; // butterflies 0 2 4 6: y0 y1
    const __m512d odd01     {_mm512_unpackhi_pd(_mm512_castps_pd(y0.v), _mm512_castps_pd(y1.v))}; // 1 3 5 7
    const __m512d even23    {_mm512_unpacklo_pd(_mm512_castps_pd(y2.v), _mm512_castps_pd(y3.v))};
    const __m512d odd23     {_mm512_unpackhi_pd(_mm512_castps_pd(y2.v), _mm512_castps_pd(y3.v))};
    const __m512d even_low  {_mm512_shuffle_f64x2(even01, even23, 0x44)};                         // 0 2, each y0 y1 then y2 y3
    const __m512d odd_low   {_mm512_shuffle_f64x2(odd01, odd23, 0x44)};                           // 1 3
    const __m512d even_high {_mm512_shuffle_f64x2(even01, even23, 0xEE)};                         // 4 6
    const __m512d odd_high  {_mm512_shuffle_f64x2(odd01, odd23, 0xEE)};                           // 5 7
    double* const into      {reinterpret_cast<double*>(to)};                                      // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): a complex float is eight bytes, stored whole
    _mm512_storeu_pd(into, _mm512_shuffle_f64x2(even_low, odd_low, 0x88));                        // butterflies 0 1
    _mm512_storeu_pd(into + 8, _mm512_shuffle_f64x2(even_low, odd_low, 0xDD));                    // 2 3
    _mm512_storeu_pd(into + 16, _mm512_shuffle_f64x2(even_high, odd_high, 0x88));                 // 4 5
    _mm512_storeu_pd(into + 24, _mm512_shuffle_f64x2(even_high, odd_high, 0xDD));                 // 6 7
  }

  //! \brief `a · b`, lane by lane, rounded.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, double> fft_mul(fft_reg<fft_avx2, double> a,
                                                                        fft_reg<fft_avx2, double> b) noexcept
  {
    return {_mm256_mul_pd(a.v, b.v)};
  }

  //! \brief `a · b + c`, lane by lane, one rounding.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, double> fft_fused(fft_reg<fft_avx2, double> a,
                                                                          fft_reg<fft_avx2, double> b,
                                                                          fft_reg<fft_avx2, double> c) noexcept
  {
    return {_mm256_fmadd_pd(a.v, b.v, c.v)};
  }

  //! \brief `a · b`, lane by lane, rounded.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, float> fft_mul(fft_reg<fft_avx2, float> a,
                                                                       fft_reg<fft_avx2, float> b) noexcept
  {
    return {_mm256_mul_ps(a.v, b.v)};
  }

  //! \brief `a · b + c`, lane by lane, one rounding.
  NUMERICAL_TARGET("avx2,fma") inline fft_reg<fft_avx2, float> fft_fused(fft_reg<fft_avx2, float> a,
                                                                         fft_reg<fft_avx2, float> b,
                                                                         fft_reg<fft_avx2, float> c) noexcept
  {
    return {_mm256_fmadd_ps(a.v, b.v, c.v)};
  }

  //! \brief `a · b`, lane by lane, rounded.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_mul(fft_reg<fft_avx512, double> a,
                                                                         fft_reg<fft_avx512, double> b) noexcept
  {
    return {_mm512_mul_pd(a.v, b.v)};
  }

  //! \brief `a · b + c`, lane by lane, one rounding.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_fused(fft_reg<fft_avx512, double> a,
                                                                           fft_reg<fft_avx512, double> b,
                                                                           fft_reg<fft_avx512, double> c) noexcept
  {
    return {_mm512_fmadd_pd(a.v, b.v, c.v)};
  }

  //! \brief Each complex number's real part from \p even, its imaginary part from \p odd.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_parts_of(fft_reg<fft_avx512, double> even,
                                                                              fft_reg<fft_avx512, double> odd) noexcept
  {
    return {_mm512_mask_blend_pd(0xAA, even.v, odd.v)};
  }

  //! \brief The complex numbers of \p v conjugated, exact.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_conjugated(fft_reg<fft_avx512, double> v) noexcept
  {
    return {_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(v.v), _mm512_set_epi64(INT64_MIN, 0, INT64_MIN, 0, INT64_MIN, 0, INT64_MIN, 0)))};
  }

  //! \brief Each complex number of \p v, its parts exchanged.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, double> fft_exchanged(fft_reg<fft_avx512, double> v) noexcept
  {
    return {_mm512_permute_pd(v.v, 0x55)};
  }

  //! \brief `a · b`, lane by lane, rounded.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_mul(fft_reg<fft_avx512, float> a,
                                                                        fft_reg<fft_avx512, float> b) noexcept
  {
    return {_mm512_mul_ps(a.v, b.v)};
  }

  //! \brief `a · b + c`, lane by lane, one rounding.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_fused(fft_reg<fft_avx512, float> a,
                                                                          fft_reg<fft_avx512, float> b,
                                                                          fft_reg<fft_avx512, float> c) noexcept
  {
    return {_mm512_fmadd_ps(a.v, b.v, c.v)};
  }

  //! \brief Each complex number's real part from \p even, its imaginary part from \p odd.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_parts_of(fft_reg<fft_avx512, float> even,
                                                                             fft_reg<fft_avx512, float> odd) noexcept
  {
    return {_mm512_mask_blend_ps(0xAAAA, even.v, odd.v)};
  }

  //! \brief The complex numbers of \p v conjugated, exact.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_conjugated(fft_reg<fft_avx512, float> v) noexcept
  {
    return {_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(v.v), _mm512_set1_epi64(INT64_MIN)))};
  }

  //! \brief Each complex number of \p v, its parts exchanged.
  NUMERICAL_TARGET("avx512f") inline fft_reg<fft_avx512, float> fft_exchanged(fft_reg<fft_avx512, float> v) noexcept
  {
    return {_mm512_permute_ps(v.v, 0xB1)};
  }

#endif
} // namespace numerical::detail

#endif // NUMERICAL_X86_HPP
