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
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#  define NUMERICAL_X86_COMPILED // the x86-64 tiles are compiled in (tested with #ifdef only)
#  include <immintrin.h>
#  if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#    define NUMERICAL_TARGET(isa)
#    define NUMERICAL_UNROLL_TILE
#    define NUMERICAL_UNROLL_DEPTH
#  else
#    include <cpuid.h>
#    define NUMERICAL_TARGET(isa) __attribute__((target(isa))) // NOLINT(cppcoreguidelines-macro-usage): an attribute, which no function can stand for
// A tile loop unrolled whole, so each accumulator stays in its register (rolled, GCC keeps them in memory).
#    define NUMERICAL_UNROLL_TILE _Pragma("GCC unroll 24")
// The depth loop unrolled by two: half the loop's own instructions, which compete with the multiply-adds for
// issue slots (an AVX-512 tile in the L1 cache: 103 GFLOP/s rolled, 115 unrolled, on one x86-64 core).
#    define NUMERICAL_UNROLL_DEPTH _Pragma("GCC unroll 2")
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
   * \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols, at most \ref avx512_mr × \ref avx512_nr) over a
   *        panel of depth \p kc: 24 accumulators of eight `double`, each step one fused multiply-add per
   *        accumulator. A partial tile goes through a full one, its outside left at zero.
   */
  NUMERICAL_TARGET("avx512f") inline void avx512_tile_f64(std::size_t   kc,
                                                          const double* ap,
                                                          const double* bp,
                                                          double*       c,
                                                          std::size_t   ldc,
                                                          std::size_t   rows,
                                                          std::size_t   cols) noexcept
  {
    std::array<double, avx512_mr * avx512_nr> edge {};
    const bool                                full {rows == avx512_mr && cols == avx512_nr};
    double* const                             tile {full ? c : edge.data()};
    const std::size_t                         ld   {full ? ldc : avx512_nr};
    for (std::size_t i {0}; !full && i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        edge[(i * avx512_nr) + j] = c[(i * ldc) + j];
      }
    }
    std::array<zmm_register, avx512_mr * 2> acc {};
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < avx512_mr; ++i) {
      acc[2 * i].v       = _mm512_loadu_pd(tile + (i * ld));
      acc[(2 * i) + 1].v = _mm512_loadu_pd(tile + (i * ld) + 8);
    }
    NUMERICAL_UNROLL_DEPTH
    for (std::size_t p {0}; p < kc; ++p) {
      const __m512d b0 {_mm512_loadu_pd(bp + (p * avx512_nr))};
      const __m512d b1 {_mm512_loadu_pd(bp + (p * avx512_nr) + 8)};
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
    for (std::size_t i {0}; !full && i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        c[(i * ldc) + j] = edge[(i * avx512_nr) + j];
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

  /*!
   * \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols, at most \ref avx2_mr × \ref avx2_nr) over a panel of
   *        depth \p kc: 12 accumulators of four `double`, each step one fused multiply-add per accumulator.
   */
  NUMERICAL_TARGET("avx2,fma") inline void avx2_tile_f64(std::size_t   kc,
                                                         const double* ap,
                                                         const double* bp,
                                                         double*       c,
                                                         std::size_t   ldc,
                                                         std::size_t   rows,
                                                         std::size_t   cols) noexcept
  {
    std::array<double, avx2_mr * avx2_nr> edge {};
    const bool                            full {rows == avx2_mr && cols == avx2_nr};
    double* const                         tile {full ? c : edge.data()};
    const std::size_t                     ld   {full ? ldc : avx2_nr};
    for (std::size_t i {0}; !full && i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        edge[(i * avx2_nr) + j] = c[(i * ldc) + j];
      }
    }
    std::array<ymm_register, avx2_mr * 2> acc {};
    NUMERICAL_UNROLL_TILE
    for (std::size_t i {0}; i < avx2_mr; ++i) {
      acc[2 * i].v       = _mm256_loadu_pd(tile + (i * ld));
      acc[(2 * i) + 1].v = _mm256_loadu_pd(tile + (i * ld) + 4);
    }
    NUMERICAL_UNROLL_DEPTH
    for (std::size_t p {0}; p < kc; ++p) {
      const __m256d b0 {_mm256_loadu_pd(bp + (p * avx2_nr))};
      const __m256d b1 {_mm256_loadu_pd(bp + (p * avx2_nr) + 4)};
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
    for (std::size_t i {0}; !full && i < rows; ++i) {
      for (std::size_t j {0}; j < cols; ++j) {
        c[(i * ldc) + j] = edge[(i * avx2_nr) + j];
      }
    }
    _mm256_zeroupper(); // the upper halves cleared: code past the tile may be SSE alone
  }

#endif
} // namespace numerical::detail

#endif // NUMERICAL_X86_HPP
