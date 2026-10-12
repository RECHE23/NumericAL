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
 * - it is used only on a chip whose CPU family is one of the M1 to M4 families, so no other chip can meet an
 *   instruction it lacks;
 * - on first use a product computed through it must equal, to the bit, the same product computed by
 *   `std::fma` in ascending order of the depth; else the path stays off for the process;
 * - \ref numerical::limit_gemm_path keeps products off it at run time, and the Python package reads
 *   `NUMERICAL_AMX=0`.
 *
 * Every element of C is a chain of fused multiply-adds in ascending order of the depth, so a product through
 * the AMX is the same on any number of threads; it differs from the vector path's separately rounded products
 * in the last bits, as any fused product does.
 */
#ifndef NUMERICAL_AMX_HPP
#define NUMERICAL_AMX_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#if defined(__APPLE__) && defined(__aarch64__) && !defined(NUMERICAL_NO_AMX)
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
    fma64 = 10, //!< `double` fused multiply-add, as an outer product in matrix mode.
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

  //! \brief The rows of C one AMX tile covers: four Y registers of eight `double`.
  inline constexpr std::size_t amx_mr {32};

  //! \brief The columns of C one AMX tile covers: two X registers of eight `double`.
  inline constexpr std::size_t amx_nr {16};

  /*!
   * \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols, at most \ref amx_mr × \ref amx_nr) over a panel of
   *        depth \p kc, on the AMX of the calling thread (inside an \ref amx_session).
   *
   * The eight 8×8 accumulators of the Z registers hold the tile: accumulator `2·ya + xb` holds rows
   * `8·ya …` and columns `8·xb …`, its row `j` in Z row `8·j + 2·ya + xb`. Each step of the depth loads A's
   * 32 elements into Y and B's 16 into X, and one fused outer product per accumulator adds them in.
   * \param[in]     kc   The panel depth.
   * \param[in]     ap   The A sliver: \p kc columns of \ref amx_mr elements.
   * \param[in]     bp   The B sliver: \p kc rows of \ref amx_nr elements.
   * \param[in,out] c    The tile's first element, in a row-major C.
   * \param[in]     ldc  The distance between C's rows.
   * \param[in]     rows The tile's rows (the sliver beyond them is zero).
   * \param[in]     cols The tile's columns (likewise).
   */
  inline void amx_tile_f64(std::size_t   kc,
                           const double* ap,
                           const double* bp,
                           double*       c,
                           std::size_t   ldc,
                           std::size_t   rows,
                           std::size_t   cols) noexcept
  {
    // A partial tile goes through a full one, its outside left at zero; a full one is read and written in place.
    std::array<double, amx_mr * amx_nr> edge                  {};
    const bool                          full                  {rows == amx_mr && cols == amx_nr};
    double* const                       tile                  {full ? c : edge.data()};
    const std::size_t                   ld                    {full ? ldc : amx_nr};
    if (!full) {
      for (std::size_t i {0}; i < rows; ++i) {
        for (std::size_t j {0}; j < cols; ++j) {
          edge[(i * amx_nr) + j] = c[(i * ldc) + j];
        }
      }
    }
    for (std::uint64_t ya {0}; ya < 4; ++ya) {
      for (std::uint64_t xb {0}; xb < 2; ++xb) {
        for (std::uint64_t j {0}; j < 8; ++j) {
          amx<amx_op::ldz>(amx_address(tile + (((ya * 8) + j) * ld) + (xb * 8), (j * 8) + (ya * 2) + xb));
        }
      }
    }
    for (std::size_t p {0}; p < kc; ++p) {
      const double* const a {ap + (p * amx_mr)};
      const double* const b {bp + (p * amx_nr)};
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
    for (std::uint64_t ya {0}; ya < 4; ++ya) {
      for (std::uint64_t xb {0}; xb < 2; ++xb) {
        for (std::uint64_t j {0}; j < 8; ++j) {
          amx<amx_op::stz>(amx_address(tile + (((ya * 8) + j) * ld) + (xb * 8), (j * 8) + (ya * 2) + xb));
        }
      }
    }
    if (!full) {
      for (std::size_t i {0}; i < rows; ++i) {
        for (std::size_t j {0}; j < cols; ++j) {
          c[(i * ldc) + j] = edge[(i * amx_nr) + j];
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
   * \brief A product through the AMX equals, to the bit, the same product by `std::fma` in ascending order of
   *        the depth: a full tile and a partial one, every accumulator, signs and magnitudes mixed.
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
        amx_tile_f64(depth, a.data(), b.data(), c.data(), amx_nr, rows, cols);
      }
    }
    return c == want;
  }

  //! \brief Whether the AMX of a chip of \p family can be used: a known family, then a passing self-test (never
  //!        run on another family, which may lack the instructions).
  [[nodiscard]] inline bool amx_usable_on(std::uint32_t family) noexcept
  {
    return amx_family_known(family) ? amx_self_test() : false;
  }

#endif

  //! \brief Whether this chip's AMX can be used: known family and a passing self-test, decided once.
  [[nodiscard]] inline bool amx_usable() noexcept
  {
#ifdef NUMERICAL_AMX_COMPILED
    static const bool usable {amx_usable_on(cpu_family())};
    return usable;
#else
    return false;
#endif
  }
} // namespace numerical::detail

#endif // NUMERICAL_AMX_HPP
