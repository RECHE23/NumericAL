/*!
 * \file amx.hpp
 * \brief The matrix coprocessor of Apple's M-series chips (AMX), for the `double` matrix product.
 *
 * Apple documents no instruction for it; the encodings used here are the ones the AMX has kept from the M1
 * to the M4 for loads, stores and the 64-bit fused multiply-add. Each core reaches it through instructions
 * the assembler emits as raw words, so no compiler support is needed and the library stays header-only.
 *
 * Because the instructions are not documented, the path is used only where it is known to work:
 * - it is compiled only for arm64 macOS, and not at all under `NUMERICAL_NO_AMX`;
 * - it is used only on a chip whose CPU family is one of the M1 to M4 families (which the A14 and A15 share), so
 *   no other chip can meet an instruction it lacks, and not in a virtual machine;
 * - on first use a product computed through it must equal, to the bit, the same product computed by
 *   `std::fma` in ascending order of the depth; else the path stays off for the process;
 * - \ref numerical::limit_gemm_path keeps products off it at run time, and the Python package reads
 *   `NUMERICAL_AMX=0`: under such a cap, not even the self-test runs.
 *
 * Every element of C is a chain of fused multiply-adds in ascending order of the depth, so a product through
 * the AMX is the same on any number of threads; it differs from the vector path's separately rounded products
 * in the last bits, as any fused product does.
 */
#ifndef NUMERICAL_AMX_HPP
#define NUMERICAL_AMX_HPP

#include <algorithm>
#include <array>
#include <cstring>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "simd.hpp"

#if defined(__APPLE__)
#  include <TargetConditionals.h>
#endif
#if defined(__APPLE__) && defined(__aarch64__) && TARGET_OS_OSX && !defined(NUMERICAL_NO_AMX)
#  define NUMERICAL_AMX_COMPILED // the AMX path is compiled in (tested with #ifdef only)
#  include <sys/sysctl.h>
#endif

namespace numerical::detail {

#ifdef NUMERICAL_AMX_COMPILED
  //! \brief The AMX instruction numbers this file emits.
  enum class amx_op : unsigned
  {
    ldx   = 0,  //!< Load 64 bytes into an X register.
    ldy   = 1,  //!< Load 64 bytes into a Y register.
    ldz   = 4,  //!< Load 64 bytes into a Z row.
    stz   = 5,  //!< Store a Z row's 64 bytes.
    fma64 = 10, //!< `double` fused multiply-add, as an outer product in matrix mode, or lane by lane in vector mode.
  };

  /*!
   * \brief Emits AMX instruction \p Op with \p operand in a general register. The instruction word carries the
   *        register's number: `0%1` reads the assembler's `xN` as the literal `0xN`, which the arithmetic turns
   *        back into N (`0x1N` is 16 + N, less 6).
   */
  template <amx_op Op>
  inline void amx(std::uint64_t operand) noexcept
  {
    __asm__ volatile (".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" // NOLINT(hicpp-no-assembler): no compiler emits the AMX
                      :
                      : "i" (static_cast<unsigned>(Op)), "r" (operand)
                      : "memory");
  }

  /*!
   * \brief Turns the AMX on for this thread (\p Off false) or off (true): its registers exist only between the
   *        two, and the system saves them with the thread's state.
   */
  template <bool Off>
  inline void amx_switch() noexcept
  {
    __asm__ volatile ("nop\nnop\nnop\n.word (0x201000 + (17 << 5) + %0)" // NOLINT(hicpp-no-assembler): no compiler emits the AMX
                      :
                      : "i" (static_cast<unsigned>(Off))
                      : "memory");
  }

  //! \brief The AMX on for a scope of one thread.
  class amx_session
  {
  public:

    amx_session() noexcept
    {
      amx_switch<false>();
    }

    amx_session(const amx_session&)            = delete;
    amx_session& operator=(const amx_session&) = delete;
    amx_session(amx_session&&)                 = delete;
    amx_session& operator=(amx_session&&)      = delete;

    ~amx_session()
    {
      amx_switch<true>();
    }
  };

  //! \brief The operand of a load or store: the address in the low 56 bits, the register or row above.
  [[nodiscard]] inline std::uint64_t amx_address(const void  * at,
                                                 std::uint64_t index) noexcept
  {
    return (index << 56U) | reinterpret_cast<std::uintptr_t>(at); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the instruction takes the address as an integer
  }

  //! \brief The rows of one group of \ref amx_dot_rows_f64: one Z row each.
  inline constexpr std::size_t amx_dot_group {8};

  /*!
   * \brief `out[i] = Σⱼ m[i·ld + j] · x[j]` for \p rows rows of \p count elements, on the AMX of the calling thread
   *        (inside an \ref amx_session), eight rows at a time: each row's Z row takes, lane by lane, one fused
   *        multiply-add per eight elements (`fma64` in vector mode), x loaded once for the eight rows. Lane `l` of a
   *        row sums its elements `j ≡ l (mod 8)` in ascending order of `j`, as a chain of fused multiply-adds (the
   *        last ones, past a multiple of eight, by `std::fma`), and the row's sum is its lanes added from 0 to 7.
   *        From the L2 cache, a row's 64 bytes a load: one M1 core, order 256, 5.7 µs, eight rows of NEON 8.1.
   */
  inline void amx_dot_rows_f64(const double* m,
                               std::size_t   ld,
                               std::size_t   rows,
                               const double* x,
                               std::size_t   count,
                               double*       out) noexcept
  {
    alignas(64) static constexpr std::array<double, 8> zero            {};
    alignas(64) std::array<std::array<double, 8>, amx_dot_group> lanes {};
    const std::size_t whole                                            {count - (count % 8)};
    for (std::size_t r0 {0}; r0 < rows; r0 += amx_dot_group) {
      const std::size_t group {std::min(amx_dot_group, rows - r0)};
      for (std::size_t r {0}; r < group; ++r) {
        amx<amx_op::ldz>(amx_address(zero.data(), r));
      }
      for (std::size_t j {0}; j < whole; j += 8) {
        amx<amx_op::ldx>(amx_address(x + j, 0));
        for (std::size_t r {0}; r < group; ++r) {
          amx<amx_op::ldy>(amx_address(m + ((r0 + r) * ld) + j, r));
          amx<amx_op::fma64>((std::uint64_t {1} << 63U) | (std::uint64_t {r} << 20U) | (std::uint64_t {r} *64U)); // vector mode: Z row r += X0 ∘ Y r
        }
      }
      for (std::size_t r {0}; r < group; ++r) { // every row stored before any is read: a read right behind its store waited on it
        amx<amx_op::stz>(amx_address(lanes[r].data(), r));
      }
      for (std::size_t r {0}; r < group; ++r) {
        for (std::size_t j {whole}; j < count; ++j) {
          lanes[r][j % 8] = std::fma(m[((r0 + r) * ld) + j], x[j], lanes[r][j % 8]);
        }
        double sum {lanes[r][0]};
        for (std::size_t l {1}; l < 8; ++l) {
          sum += lanes[r][l];
        }
        out[r0 + r] = sum;
      }
    }
  }

  //! \brief The rows of C one AMX tile covers: four Y registers of eight `double`.
  inline constexpr std::size_t amx_mr {32};

  //! \brief The columns of C one AMX tile covers: two X registers of eight `double`.
  inline constexpr std::size_t amx_nr {16};

  //! \brief The fewest rows and columns of C, together, for which the AMX runs a product of fewer than
  //!        \ref amx_least_work multiply-adds: half its tile's rows and all its columns. A shorter or narrower product
  //!        computes whole tiles into a scratch block, and with the AMX's start-up (about 0.35 µs a product) it lost
  //!        to the vector tile up to five times over (M1 Pro, 8 × 166 by 166 × 8: 7.0 µs against 1.3).
  inline constexpr std::size_t amx_least_extent {16};

  //! \brief The multiply-adds from which the AMX runs a product of any shape. With \ref amx_least_extent, against
  //!        the faster of the AMX and the vector tile on 302 random shapes of 2^13 to 2^22 multiply-adds (M1 Pro), the
  //!        products took 1.05 times as long in all, the slowest 1.8 times; on the AMX always, 1.09 and 5.5.
  inline constexpr std::size_t amx_least_work {std::size_t {1} << 16U};

  //! \brief Whether the AMX runs a product of \p m × \p k by \p k × \p n (\ref amx_least_extent, \ref amx_least_work).
  [[nodiscard]] constexpr bool amx_suits(std::size_t m,
                                         std::size_t n,
                                         std::size_t k) noexcept
  {
    return (m >= amx_least_extent && n >= amx_least_extent) || m * n * k >= amx_least_work;
  }

  //! \brief The longest rows of B the AMX reads in place, unpacked: X loads a row's 16 elements as they lie, and
  //!        packing them took a fifth of a small product (M1 Pro, order 64: 5.5 → 4.3 µs, 256: 130 → 104 µs); rows
  //!        further apart cost more in translations than packing (order 1024: +8 %, 2048: +35 %).
  inline constexpr std::size_t amx_in_place_stride {512};

  /*!
   * \brief The depth loop of a tile: for each step, A's 32 elements into Y and B's 16 into X, and one fused outer
   *        product per accumulator of Z (see \ref amx_full_tile_f64), the one loop of the whole and the partial
   *        tiles. Out of line and aligned, so the loop, a few instructions past the entry, spans two fetch blocks of
   *        64 bytes wherever the linker puts it: inline, it fell where its caller fell, three blocks in one binary.
   * \param[in] kc  The panel depth.
   * \param[in] ap  The A sliver: \p kc columns of \ref amx_mr elements.
   * \param[in] bp  The B sliver: \p kc rows of \ref amx_nr elements.
   * \param[in] ldb The distance between the sliver's rows.
   */
  [[gnu::noinline, gnu::aligned(64)]] inline void amx_depth_f64(std::size_t   kc,
                                                                const double* ap,
                                                                const double* bp,
                                                                std::size_t   ldb) noexcept
  {
    for (std::size_t p {0}; p < kc; ++p) {
      const double* const a {ap + (p * amx_mr)};
      const double* const b {bp + (p * ldb)};
      for (std::uint64_t ya {0}; ya < 4; ++ya) {
        amx<amx_op::ldy>(amx_address(a + (ya * 8), ya));
      }
      amx<amx_op::ldx>(amx_address(b, 0));
      amx<amx_op::ldx>(amx_address(b + 8, 1));
      for (std::uint64_t ya {0}; ya < 4; ++ya) {
        for (std::uint64_t xb {0}; xb < 2; ++xb) {
          // Matrix mode: Z row (2·ya + xb), X register xb (64 bytes each), Y register ya.
          amx<amx_op::fma64>((((ya * 2) + xb) << 20U) | ((xb * 64) << 10U) | (ya * 64));
        }
      }
    }
  }

  /*!
   * \brief `C += Aₚ·Bₚ` for one whole tile of C (\ref amx_mr × \ref amx_nr) over a panel of depth \p kc,
   *        on the AMX of the calling thread (inside an \ref amx_session).
   *
   * The eight 8×8 accumulators of the Z registers hold the tile: accumulator `2·ya + xb` holds rows
   * `8·ya …` and columns `8·xb …`, its row `j` in Z row `8·j + 2·ya + xb`. Each step of the depth loads A's
   * 32 elements into Y and B's 16 into X, and one fused outer product per accumulator adds them in.
   * \param[in]     kc   The panel depth.
   * \param[in]     ap   The A sliver: \p kc columns of \ref amx_mr elements.
   * \param[in]     bp   The B sliver: \p kc rows of \ref amx_nr elements.
   * \param[in]     ldb  The distance between the sliver's rows: \ref amx_nr packed, B's own read in place.
   * \param[in,out] tile The tile's first element, in a row-major C.
   * \param[in]     ld   The distance between C's rows.
   */
  inline void amx_full_tile_f64(std::size_t   kc,
                                const double* ap,
                                const double* bp,
                                std::size_t   ldb,
                                double*       tile,
                                std::size_t   ld) noexcept
  {
    for (std::uint64_t ya {0}; ya < 4; ++ya) {
      for (std::uint64_t xb {0}; xb < 2; ++xb) {
        for (std::uint64_t j {0}; j < 8; ++j) {
          amx<amx_op::ldz>(amx_address(tile + (((ya * 8) + j) * ld) + (xb * 8), (j * 8) + (ya * 2) + xb));
        }
      }
    }
    amx_depth_f64(kc, ap, bp, ldb);
    for (std::uint64_t ya {0}; ya < 4; ++ya) {
      for (std::uint64_t xb {0}; xb < 2; ++xb) {
        for (std::uint64_t j {0}; j < 8; ++j) {
          amx<amx_op::stz>(amx_address(tile + (((ya * 8) + j) * ld) + (xb * 8), (j * 8) + (ya * 2) + xb));
        }
      }
    }
  }

  /*!
   * \brief \ref amx_full_tile_f64 for a partial tile of \p rows × \p cols, in C itself: each half row of eight
   *        elements inside C is loaded and stored in place, one the last column cuts through a buffer, and those
   *        outside are neither (their Z rows sum the zeros beyond A's and B's slivers, and are never read).
   * \param[in]     kc   The panel depth.
   * \param[in]     ap   The A sliver: \p kc columns of \ref amx_mr elements.
   * \param[in]     bp   The B sliver: \p kc rows of \ref amx_nr elements.
   * \param[in]     ldb  The distance between the sliver's rows.
   * \param[in,out] c    The tile's first element, in a row-major C.
   * \param[in]     ldc  The distance between C's rows.
   * \param[in]     rows The tile's rows inside C.
   * \param[in]     cols The tile's columns inside C.
   */
  inline void amx_partial_tile_f64(std::size_t   kc,
                                   const double* ap,
                                   const double* bp,
                                   std::size_t   ldb,
                                   double*       c,
                                   std::size_t   ldc,
                                   std::size_t   rows,
                                   std::size_t   cols) noexcept
  {
    alignas(64) std::array<double, amx_mr * 8> cut {}; // the half row the last column cuts, per row
    const std::uint64_t whole_halves {cols / 8};       // halves wholly inside C
    const std::size_t   cut_width {cols % 8};          // the cut half's elements inside C
    const auto          each_half = [&](auto&& at) {
                                      for (std::uint64_t ya {0}; ya < 4; ++ya) {
                                        for (std::uint64_t j {0}; j < 8; ++j) {
                                          const std::size_t row {(ya * 8) + j};
                                          if (row >= rows) {
                                            continue;
                                          }
                                          for (std::uint64_t xb {0}; xb < 2; ++xb) {
                                            if (xb < whole_halves) {
                                              at(c + (row * ldc) + (xb * 8), (j * 8) + (ya * 2) + xb);
                                            }
                                            else if (xb == whole_halves && cut_width != 0) {
                                              at(cut.data() + (row * 8), (j * 8) + (ya * 2) + xb);
                                            }
                                          }
                                        }
                                      }
                                    };
    if (cut_width != 0) {
      for (std::size_t row {0}; row < rows; ++row) {
        for (std::size_t e {0}; e < cut_width; ++e) {
          cut[(row * 8) + e] = c[(row * ldc) + (whole_halves * 8) + e];
        }
      }
    }
    each_half([](double* half, std::uint64_t z) { amx<amx_op::ldz>(amx_address(half, z)); });
    amx_depth_f64(kc, ap, bp, ldb);
    each_half([](double* half, std::uint64_t z) { amx<amx_op::stz>(amx_address(half, z)); });
    if (cut_width != 0) {
      for (std::size_t row {0}; row < rows; ++row) {
        for (std::size_t e {0}; e < cut_width; ++e) {
          c[(row * ldc) + (whole_halves * 8) + e] = cut[(row * 8) + e];
        }
      }
    }
  }

  /*!
   * \brief \ref amx_full_tile_f64 for a tile of \p rows × \p cols (at most \ref amx_mr × \ref amx_nr); the slivers
   *        beyond it are zero. A whole tile is computed in place; a partial one in place too (\ref amx_partial_tile_f64)
   *        unless its last column cuts its second half row, then through a whole one whose outside is zero
   *        (\ref through_edge). A copy of the tile, written by the core and read by the AMX, then the other way, held
   *        each partial tile behind its lines' write-back to the L2 the AMX reads (M1 Pro, a product of order 48: 4.2
   *        → 2.8 µs, 65: 10.3 → 7.8); a second half cut in place, its first half stored by the AMX and its second
   *        by the core, was slower still than the copy (order 31: 2.7 → 3.2 µs). A tile of eight rows or fewer, the
   *        last of a product whose rows are not a multiple of \ref amx_mr, runs in place too: through the copy, each
   *        such tile cost a microsecond (one M1 Pro core, products of order 100 19.2 → 15.9 µs, 129 32.9 → 28.8, 33
   *        3.7 → 2.5; an SVD or a QR of order 33 −16 %, of order 8 level).
   */
  inline void amx_tile_f64(std::size_t   kc,
                           const double* ap,
                           const double* bp,
                           std::size_t   ldb,
                           double*       c,
                           std::size_t   ldc,
                           std::size_t   rows,
                           std::size_t   cols) noexcept
  {
    if ((rows < amx_mr || cols < amx_nr) && (cols < 8 || cols % 8 == 0)) {
      amx_partial_tile_f64(kc, ap, bp, ldb, c, ldc, rows, cols);
      return;
    }
    through_edge<amx_mr, amx_nr>(c, ldc, rows, cols, [&](double* tile, std::size_t ld) noexcept { amx_full_tile_f64(kc, ap, bp, ldb, tile, ld); });
  }

  //! \brief The fewest elements of A one packing writes past the L1 (\ref amx_pack_a): below, a block of a few
  //!        kilobytes, the round trip to beyond the L2 costs more than the dirty lines it spares (M1 Pro, a product of
  //!        64 × 16 by 16 × 64: +17 % streamed; 64 × 32 by 32 × 64: −21 %).
  inline constexpr std::size_t amx_stream_least {2048};

  /*!
   * \brief A's slivers for \ref amx_full_tile_f64 (\ref amx_mr rows each, stored column by column). From
   *        \ref amx_stream_least elements, the whole slivers are written by non-temporal pair stores: the AMX reads
   *        its operands from the L2 cache, and A's lines left dirty in the core's L1 by ordinary stores held each
   *        product's first tiles back while they were written back (M1 Pro, a product of order 64: 3.78 → 2.88 µs).
   *        A smaller block's slivers, and a last partial one, are written by ordinary stores, the partial one padded
   *        with zeros.
   */
  inline void amx_pack_a(const double* a,
                         std::size_t   lda,
                         std::size_t   rows,
                         std::size_t   kc,
                         double*       out) noexcept
  {
    using pair = double __attribute__((vector_size(16)));
    std::size_t ir {0};
    for (; rows * kc >= amx_stream_least && ir + amx_mr <= rows; ir += amx_mr) {
      double* const sliver {out + (ir * kc)};
      std::size_t   p      {0};
      for (; p + 2 <= kc; p += 2) { // two columns of four rows a step: the rows' pairs read whole, their lanes swapped
        for (std::size_t i {0}; i < amx_mr; i += 4) {
          std::array<pair, 4> row {};
          for (std::size_t r {0}; r < 4; ++r) {
            std::memcpy(&row.at(r), a + ((ir + i + r) * lda) + p, sizeof(pair));
          }
          const pair here_low  {__builtin_shufflevector(row[0], row[1], 0, 2)};
          const pair here_high {__builtin_shufflevector(row[2], row[3], 0, 2)};
          const pair next_low  {__builtin_shufflevector(row[0], row[1], 1, 3)};
          const pair next_high {__builtin_shufflevector(row[2], row[3], 1, 3)};
          __asm__ volatile ("stnp %q0, %q1, [%2]" // NOLINT(hicpp-no-assembler): as below, four elements at once
                            :
                            : "w" (here_low), "w" (here_high), "r" (sliver + (p * amx_mr) + i)
                            : "memory");
          __asm__ volatile ("stnp %q0, %q1, [%2]" // NOLINT(hicpp-no-assembler): as below, four elements at once
                            :
                            : "w" (next_low), "w" (next_high), "r" (sliver + ((p + 1) * amx_mr) + i)
                            : "memory");
        }
      }
      for (; p < kc; ++p) {
        for (std::size_t i {0}; i < amx_mr; i += 2) {
          const double first  {a[((ir + i) * lda) + p]};
          const double second {a[((ir + i + 1) * lda) + p]};
          __asm__ volatile ("stnp %d0, %d1, [%2]" // NOLINT(hicpp-no-assembler): no builtin emits a pair past the L1 under both compilers
                            :
                            : "w" (first), "w" (second), "r" (sliver + (p * amx_mr) + i)
                            : "memory");
        }
      }
    }
    for (; ir + amx_mr <= rows; ir += amx_mr) { // whole slivers below the threshold, the loop free of the padding's test
      double* const sliver {out + (ir * kc)};
      std::size_t   p      {0};
      for (; p + 2 <= kc; p += 2) { // as above, by ordinary stores
        for (std::size_t i {0}; i < amx_mr; i += 2) {
          pair first;
          pair second;
          std::memcpy(&first, a + ((ir + i) * lda) + p, sizeof(pair));
          std::memcpy(&second, a + ((ir + i + 1) * lda) + p, sizeof(pair));
          const pair here {__builtin_shufflevector(first, second, 0, 2)};
          const pair next {__builtin_shufflevector(first, second, 1, 3)};
          std::memcpy(sliver + (p * amx_mr) + i, &here, sizeof(pair));
          std::memcpy(sliver + ((p + 1) * amx_mr) + i, &next, sizeof(pair));
        }
      }
      for (; p < kc; ++p) {
        for (std::size_t i {0}; i < amx_mr; ++i) {
          sliver[(p * amx_mr) + i] = a[((ir + i) * lda) + p];
        }
      }
    }
    if (ir < rows) {
      double* const sliver {out + (ir * kc)};
      for (std::size_t p {0}; p < kc; ++p) {
        for (std::size_t i {0}; i < amx_mr; ++i) {
          sliver[(p * amx_mr) + i] = ir + i < rows ? a[((ir + i) * lda) + p] : 0.0;
        }
      }
    }
  }

  //! \brief Whether \p family (Apple's CPUFAMILY_ARM_* values) is one of the M1 to M4 families.
  [[nodiscard]] constexpr bool amx_family_known(std::uint32_t family) noexcept
  {
    switch (family) {
      case 0x1b588bb3U: // FIRESTORM_ICESTORM: M1, M1 Pro, Max, Ultra
      case 0xda33d83dU: // BLIZZARD_AVALANCHE: M2 family
      case 0xfa33415eU: // IBIZA: M3
      case 0x5f4dea93U: // LOBOS: M3 Pro
      case 0x72015832U: // PALMA: M3 Max
      case 0x6f5129acU: // DONAN: M4
      case 0x17d5b93aU: // BRAVA: M4 Pro, Max
        return true;
      default:
        return false;
    }
  }

  //! \brief This chip's CPU family, or 0 (no known family) when the system does not say.
  [[nodiscard]] inline std::uint32_t cpu_family() noexcept
  {
    std::uint32_t family {0};
    std::size_t   size   {sizeof family};
    (void)sysctlbyname("hw.cpufamily", &family, &size, nullptr, 0); // on failure family stays 0
    return family;
  }

  /*!
   * \brief Rows' dot products through the AMX (\ref amx_dot_rows_f64) equal, to the bit, the same lanes summed by
   *        `std::fma`: a whole group of rows and a partial one, a tail past a multiple of eight.
   */
  [[nodiscard]] inline bool amx_dot_self_test() noexcept
  {
    constexpr std::size_t                    rows  {11};
    constexpr std::size_t                    count {21};
    std::array<double, rows * count>         m     {};
    std::array<double, count>                x     {};
    std::array<double, rows>                 got   {};
    std::array<double, rows>                 want  {};
    for (std::size_t i {0}; i < m.size(); ++i) {
      m[i] = std::ldexp(static_cast<double>((i * 37) % 101) - 50.0, -static_cast<int>(i % 9));
    }
    for (std::size_t j {0}; j < count; ++j) {
      x[j] = 1.0 / static_cast<double>((j % 13) + 3);
    }
    for (std::size_t r {0}; r < rows; ++r) {
      std::array<double, 8> lane {};
      for (std::size_t j {0}; j < count; ++j) {
        lane[j % 8] = std::fma(m[(r * count) + j], x[j], lane[j % 8]);
      }
      double sum {lane[0]};
      for (std::size_t l {1}; l < 8; ++l) {
        sum += lane[l];
      }
      want[r] = sum;
    }
    {
      const amx_session on {};
      amx_dot_rows_f64(m.data(), count, rows, x.data(), count, got.data());
    }
    return got == want;
  }

  /*!
   * \brief A product through the AMX equals, to the bit, the same product by `std::fma` in ascending order of
   *        the depth: a full tile and a partial one, every accumulator, signs and magnitudes mixed; and the
   *        rows' dot products their lanes by `std::fma` (\ref amx_dot_self_test).
   */
  [[nodiscard]] inline bool amx_self_test() noexcept
  {
    constexpr std::size_t               depth                 {7};
    std::array<double, depth * amx_mr>  a                     {};
    std::array<double, depth * amx_nr>  b                     {};
    std::array<double, amx_mr * amx_nr> c                     {};
    std::array<double, amx_mr * amx_nr> want                  {};
    for (std::size_t i {0}; i < depth * amx_mr; ++i) {
      a[i] = std::ldexp(static_cast<double>((i * 37) % 101) - 50.0, -static_cast<int>(i % 9));
    }
    for (std::size_t i {0}; i < depth * amx_nr; ++i) {
      b[i] = 1.0 / static_cast<double>((i % 13) + 3);
    }
    for (std::size_t i {0}; i < amx_mr * amx_nr; ++i) {
      c[i]    = static_cast<double>(i % 17) * 0.1;
      want[i] = c[i];
    }
    for (const std::size_t rows : {amx_mr, std::size_t {19}}) {
      const std::size_t cols {rows == amx_mr ? amx_nr : std::size_t {11}};
      for (std::size_t i {0}; i < rows; ++i) {
        for (std::size_t j {0}; j < cols; ++j) {
          for (std::size_t p {0}; p < depth; ++p) {
            want[(i * amx_nr) + j] = std::fma(a[(p * amx_mr) + i], b[(p * amx_nr) + j], want[(i * amx_nr) + j]);
          }
        }
      }
      {
        const amx_session on {};
        amx_tile_f64(depth, a.data(), b.data(), amx_nr, c.data(), amx_nr, rows, cols);
      }
    }
    const std::array<bool, 2> passed {c == want, amx_dot_self_test()};
    return std::count(passed.begin(), passed.end(), true) == 2; // no branch on a check that only a fault fails
  }

  //! \brief Whether this system runs in a virtual machine, whose host decides what the guest's processor offers.
  [[nodiscard]] inline bool running_virtualised() noexcept
  {
    int         present {0};
    std::size_t size    {sizeof present};
    (void)sysctlbyname("kern.hv_vmm_present", &present, &size, nullptr, 0); // on failure present stays 0
    return present != 0;
  }

  //! \brief Whether the AMX's self-test has run in this process: a cap below the AMX keeps it from running.
  [[nodiscard]] inline std::atomic<bool>& amx_probed() noexcept
  {
    static std::atomic<bool> probed {false};
    return probed;
  }

  //! \brief Whether the AMX of a chip of \p family can be used: a known family, outside a virtual machine, then a
  //!        passing self-test (never run elsewhere: another chip, or a host, may not provide the instructions).
  [[nodiscard]] inline bool amx_usable_on(std::uint32_t family,
                                          bool          virtualised) noexcept
  {
    if (!amx_family_known(family) || virtualised) {
      return false;
    }
    amx_probed().store(true);
    return amx_self_test();
  }

#endif

  //! \brief Whether this chip's AMX can be used: known family and a passing self-test, decided once.
  [[nodiscard]] inline bool amx_usable() noexcept
  {
#ifdef NUMERICAL_AMX_COMPILED
    static const bool usable {amx_usable_on(cpu_family(), running_virtualised())};
    return usable;
#else
    return false;
#endif
  }
} // namespace numerical::detail

#endif // NUMERICAL_AMX_HPP
