/*!
 * \file gemm.hpp
 * \brief The matrix product kernel behind `matrix * matrix` and the einsum matmul path.
 *
 * `C += A·B` over row-major buffers, in the loop structure of BLIS (Van Zee and van de Geijn, 2015):
 * the depth is cut into panels of \ref numerical::detail::gemm_kc, each operand's panel is copied once
 * into a contiguous layout the inner loop reads at unit stride (A in slivers of \ref
 * numerical::detail::gemm_mr rows, B in slivers of \ref numerical::detail::gemm_nr columns), and a
 * register tile of `gemm_mr × gemm_nr` elements of C accumulates a whole panel's depth before it is
 * written back. Each element of C is read into the tile at the start of a panel and written back at its
 * end, so every element is summed in ascending order of the depth, exactly as the plain i-k-j loop
 * does, on any number of threads: with the compiler's multiply-add contraction off, the two agree to
 * the bit.
 *
 * For `float` and `double` under GCC and Clang the tile is held in vectors of the target's native width
 * (16 bytes, 32 with AVX), which both compilers keep in registers; elsewhere it is a plain array.
 */
#ifndef NUMERICAL_GEMM_HPP
#define NUMERICAL_GEMM_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__GNUC__) && defined(__aarch64__)
#  include <arm_neon.h>
#  define NUMERICAL_NEON_COMPILED // the arm64 fused tile is compiled in (tested with #ifdef only)
#endif

#include "parallel.hpp"
#include "amx.hpp"
#include "buffer.hpp"
#include "simd.hpp"
#include "x86.hpp"

// A function kept out of its callers: inlined, it spent GCC's growth budget for the whole translation unit, and the
// products, the Cholesky factorisation and the LU solve lost their own inlining (one arm64 thread, GCC 14: a product of
// order 8 0.29 → 0.33 µs, a Cholesky factorisation of order 17 0.77 → 0.87 µs, a solve of order 512 0.47 → 0.54 ms).
#if defined(_MSC_VER) && !defined(__clang__)
#  define NUMERICAL_OUT_OF_LINE __declspec(noinline)
#else
#  define NUMERICAL_OUT_OF_LINE [[gnu::noinline]] // NOLINT(cppcoreguidelines-macro-usage): an attribute, which no function can stand for
#endif

#if defined(__clang__)
// A block's depth loop kept scalar: vectorised, Clang sums its terms in order but splits each multiply-add (which
// -ffp-contract=on lets it fuse, not makes it) into a product and a sum, the last terms fused: one element rounded
// two ways, where the packed kernels fuse every term.
#  define NUMERICAL_SCALAR_DEPTH _Pragma("clang loop vectorize(disable) interleave(disable)")
#else
#  define NUMERICAL_SCALAR_DEPTH
#endif

#if defined(__GNUC__)
// A block's loops unrolled whole, so that each of its sums stays in a register: GCC at -O2 does not unroll them,
// and kept the sums in memory (a product of order 17 on one arm64 core: 2.0 µs, against 0.31 under Clang).
#  define NUMERICAL_UNROLL_BLOCK _Pragma("GCC unroll 32")
#else
#  define NUMERICAL_UNROLL_BLOCK
#endif

namespace numerical {

  /*!
   * \brief The kernels a `double` matrix product can run on (a `float` one too on arm64), from the most portable. Each is used only where the
   *        processor has it: NEON on every arm64 chip, the AVX ones where `cpuid` reports them, the Apple AMX on an
   *        M1 to M4 chip whose self-test passes (see x86.hpp and amx.hpp).
   */
  enum class gemm_path : std::uint8_t
  {
    portable = 0, //!< The compiler's vectors (16 bytes on x86-64 wheels), as the compiler rounds `c + a·b` (see gemm).
    neon     = 1, //!< arm64's vectors with fused multiply-adds, for `double` and `float`.
    avx2     = 2, //!< AVX2 with fused multiply-adds.
    avx512   = 3, //!< AVX-512 with fused multiply-adds.
    amx      = 4, //!< The Apple AMX coprocessor, fused multiply-adds; the products it does not suit on the NEON path's tile.
  };
} // namespace numerical

namespace numerical::detail {

  //! \brief Whether the NEON path's tile is compiled in: on every arm64 target GCC or Clang builds.
#ifdef NUMERICAL_NEON_COMPILED
  inline constexpr bool neon_compiled {true};
#else
  inline constexpr bool neon_compiled {false};
#endif

  //! \brief Whether \p path can run here.
  [[nodiscard]] inline bool gemm_path_usable(gemm_path path) noexcept
  {
    switch (path) {
#ifdef NUMERICAL_X86_COMPILED
      case gemm_path::avx2:   return cpu_features().avx2_fma;
      case gemm_path::avx512: return cpu_features().avx512f;
#endif
      case gemm_path::neon:   return neon_compiled;
      case gemm_path::amx:    return amx_usable();
      default:                return path == gemm_path::portable;
    }
  }

  //! \brief The most capable path that can run here, \p most at the highest.
  [[nodiscard]] inline gemm_path best_gemm_path(gemm_path most) noexcept
  {
    for (auto p {static_cast<int>(most)}; p > 0; --p) {
      if (gemm_path_usable(static_cast<gemm_path>(p))) {
        return static_cast<gemm_path>(p);
      }
    }
    return gemm_path::portable;
  }

  //! \brief The most capable path products may take, as \ref numerical::limit_gemm_path set it: none is probed until a
  //!        product (or a question) reaches it, so a cap below the AMX keeps its instructions from ever running.
  [[nodiscard]] inline std::atomic<gemm_path>& gemm_path_cap() noexcept
  {
    static std::atomic<gemm_path> cap {gemm_path::amx};
    return cap;
  }

  //! \brief The path the next `double` product takes: the most capable one under the cap that runs here (each
  //!        path's probe is made once).
  [[nodiscard]] inline gemm_path gemm_path_current() noexcept
  {
    return best_gemm_path(gemm_path_cap().load(std::memory_order_relaxed));
  }

  /*!
   * \brief `y[i] −= f · x[i]` for `double` on the widest x86-64 units the matrix product's kernel may use
   *        (\ref gemm_path_current), each element one fused multiply-add there; elsewhere \ref subtract_product's
   *        template. The decompositions' row updates take it: a wheel built for the baseline x86-64 holds two
   *        `double` per vector and fuses none.
   */
  inline void subtract_product(double*       y,
                               const double& f,
                               const double* x,
                               std::size_t   count)
  {
    const double factor  {f}; // read once: it may be an element of y's array
#ifdef NUMERICAL_X86_COMPILED
    const gemm_path path {gemm_path_current()};
    if (path == gemm_path::avx512) {
      avx512_scaled_add(y, x, -factor, count);
      return;
    }
    if (path == gemm_path::avx2) {
      avx2_scaled_add(y, x, -factor, count);
      return;
    }
#endif
    subtract_product<double>(y, factor, x, count);
  }

  //! \brief `y[i] −= x[i] · f` for `double`, as the other (a real product is the same either way round).
  inline void subtract_product(double*       y,
                               const double* x,
                               const double& f,
                               std::size_t   count)
  {
    subtract_product(y, f, x, count);
  }

  //! \brief `y[i] -= Σₜ f[t] · x[(t · ldx) + i]` for `double`, the units chosen as \ref subtract_product chooses them,
  //!        the elements held in registers through the terms there.
  inline void subtract_products(double*       y,
                                const double* f,
                                const double* x,
                                std::size_t   ldx,
                                std::size_t   terms,
                                std::size_t   count)
  {
#ifdef NUMERICAL_X86_COMPILED
    const gemm_path path {gemm_path_current()};
    if (path == gemm_path::avx512) {
      avx512_subtract_products(y, f, x, ldx, terms, count);
      return;
    }
    if (path == gemm_path::avx2) {
      avx2_subtract_products(y, f, x, ldx, terms, count);
      return;
    }
#endif
    subtract_products<double>(y, f, x, ldx, terms, count);
  }

  //! \brief \ref subtract_rows for `double`, the units chosen as \ref subtract_products chooses them: each element the
  //!        value \ref subtract_products gives it.
  inline void subtract_rows(double*       y,
                            std::size_t   ldy,
                            const double* f,
                            std::size_t   ldf,
                            const double* x,
                            std::size_t   ldx,
                            std::size_t   terms,
                            std::size_t   count)
  {
#ifdef NUMERICAL_X86_COMPILED
    const gemm_path path {gemm_path_current()};
    if (path == gemm_path::avx512) {
      avx512_subtract_rows(y, ldy, f, ldf, x, ldx, terms, count);
      return;
    }
    if (path == gemm_path::avx2) {
      avx2_subtract_rows(y, ldy, f, ldf, x, ldx, terms, count);
      return;
    }
#endif
    subtract_rows<double>(y, ldy, f, ldf, x, ldx, terms, count);
  }

  /*!
   * \brief \ref subtract_products for `std::complex<double>`: on AVX-512 or AVX2 with FMA where the matrix product's
   *        kernel runs there, each part of each term one fused multiply-add (`Re f · x`, then `Im f` times the
   *        swapped parts), the elements in registers through the terms; elsewhere the template, whole complex
   *        products as the compiler rounds them.
   */
  inline void subtract_products(std::complex<double>*       y,
                                const std::complex<double>* f,
                                const std::complex<double>* x,
                                std::size_t                 ldx,
                                std::size_t                 terms,
                                std::size_t                 count)
  {
#ifdef NUMERICAL_X86_COMPILED
    // A complex number's parts are an array of two (the standard's layout guarantee): addressed as doubles.
    const gemm_path path {gemm_path_current()};
    if (path == gemm_path::avx512 || path == gemm_path::avx2) {
      auto* const       into   {reinterpret_cast<double*>(y)};       // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
      const auto* const scales {reinterpret_cast<const double*>(f)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
      const auto* const rows   {reinterpret_cast<const double*>(x)}; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the layout guarantee above
      if (path == gemm_path::avx512) {
        avx512_subtract_complex_products(into, scales, rows, ldx, terms, count);
      }
      else {
        avx2_subtract_complex_products(into, scales, rows, ldx, terms, count);
      }
      return;
    }
#endif
    subtract_products<std::complex<double>>(y, f, x, ldx, terms, count);
  }

  //! \brief `y[i] −= f · x[i]` for `std::complex<double>`: one term of \ref subtract_products on AVX-512 or AVX2
  //!        where the matrix product's kernel runs there, else the template.
  inline void subtract_product(std::complex<double>*       y,
                               const std::complex<double>& f,
                               const std::complex<double>* x,
                               std::size_t                 count)
  {
    const std::complex<double> factor {f}; // read once: it may be an element of y's array
#ifdef NUMERICAL_X86_COMPILED
    const gemm_path path              {gemm_path_current()};
    if (path == gemm_path::avx512 || path == gemm_path::avx2) {
      subtract_products(y, &factor, x, 0, 1, count);
      return;
    }
#endif
    subtract_product<std::complex<double>>(y, factor, x, count);
  }

  //! \brief `y[i] −= x[i] · f` for `std::complex<double>`, as the other (the template keeps the expression's order).
  inline void subtract_product(std::complex<double>*       y,
                               const std::complex<double>* x,
                               const std::complex<double>& f,
                               std::size_t                 count)
  {
    const std::complex<double> factor {f}; // read once: it may be an element of y's array
#ifdef NUMERICAL_X86_COMPILED
    const gemm_path path              {gemm_path_current()};
    if (path == gemm_path::avx512 || path == gemm_path::avx2) {
      subtract_products(y, &factor, x, 0, 1, count);
      return;
    }
#endif
    subtract_product<std::complex<double>>(y, x, factor, count);
  }

  //! \brief `y[i] += f · x[i]` for `double`, as \ref subtract_product.
  inline void add_product(double*       y,
                          const double& f,
                          const double* x,
                          std::size_t   count)
  {
    const double factor  {f}; // read once: it may be an element of y's array
#ifdef NUMERICAL_X86_COMPILED
    const gemm_path path {gemm_path_current()};
    if (path == gemm_path::avx512) {
      avx512_scaled_add(y, x, factor, count);
      return;
    }
    if (path == gemm_path::avx2) {
      avx2_scaled_add(y, x, factor, count);
      return;
    }
#endif
    add_product<double>(y, factor, x, count);
  }

  //! \brief `Σ x[i] · y[i]` for `double`, as \ref subtract_product chooses its units (in another order of
  //!        summation there, as fixed).
  [[nodiscard]] inline double dot_product(const double* x,
                                          const double* y,
                                          std::size_t   count)
  {
#ifdef NUMERICAL_X86_COMPILED
    const gemm_path path {gemm_path_current()};
    if (path == gemm_path::avx512) {
      return avx512_dot(x, y, count);
    }
    if (path == gemm_path::avx2) {
      return avx2_dot(x, y, count);
    }
#endif
    return dot_product<double>(x, y, count);
  }

  //! \brief \ref row_dots for `double`: on the x86-64 units each row by their \ref dot_product (wider than four rows
  //!        of the portable vectors), elsewhere four rows at a time.
  inline void row_dots(const double* m,
                       std::size_t   ld,
                       std::size_t   rows,
                       const double* x,
                       std::size_t   count,
                       double*       out)
  {
#ifdef NUMERICAL_X86_COMPILED
    if (gemm_path_current() != gemm_path::portable) {
      for (std::size_t r {0}; r < rows; ++r) {
        out[r] = dot_product(m + (r * ld), x, count);
      }
      return;
    }
#endif
    row_dots<double>(m, ld, rows, x, count, out);
  }

  //! \brief The rows of C one register tile covers.
  inline constexpr std::size_t gemm_mr {4};

  //! \brief The columns of C one register tile covers: 96 bytes of elements (12 `double`, 24 `float`),
  //!        which with \ref gemm_mr rows fills the vector registers of both arm64 and x86-64 with AVX.
  template <typename T>
  inline constexpr std::size_t gemm_nr {sizeof(T) >= 16 ? std::size_t {6} : std::size_t {96} / sizeof(T)};

  //! \brief The depth of one panel: a B sliver (`gemm_kc × gemm_nr`) stays in the L1 cache.
  inline constexpr std::size_t gemm_kc {256};

  //! \brief The rows of one task: an A block (`gemm_mc × gemm_kc`) stays in the L2 cache.
  inline constexpr std::size_t gemm_mc {64};

  //! \brief The B slivers of one task.
  inline constexpr std::size_t gemm_slivers_per_task {4};

  //! \brief The tasks a product's compute pass hands each thread at least, while its rows allow: a slower core
  //!        then holds back no one.
  inline constexpr std::size_t tasks_per_thread {4};

  //! \brief The multiply-adds the register tile does in the time of one in \ref work_per_thread's dependent
  //!        chain (about 20 G/s on one arm64 or x86-64 AVX2 core): a task weighs its work by it, so a product
  //!        of 0.1 ms does not start eight threads.
  inline constexpr std::size_t gemm_kernel_speedup {16};

  //! \brief The columns of B packed at once, a whole number of slivers.
  template <typename T>
  inline constexpr std::size_t gemm_nc {gemm_nr<T> * 128};

  //! \brief Below this many multiply-adds the plain loop (\ref gemm_rows, by blocks of rows held in registers) runs
  //!        instead of the packed kernel, both summing in the same order. Against the packed kernel on 400 random
  //!        shapes from 512 multiply-adds, the plain loop took 0.25 to 0.41 times as long below 2^13 on arm64 (M1 Pro,
  //!        geometric means by octave, the slowest shape 0.91); above, it lost up to 2.5 times on shapes the AMX
  //!        suits. On x86-64 the plain loop has the baseline's vectors and the packed one the AVX2 or AVX-512 units:
  //!        there it took 0.43 to 0.55 times as long below 2^11, 0.86 above with shapes up to 2 times slower (six
  //!        AVX2 cores, GCC 13).
#ifdef NUMERICAL_X86_COMPILED
  inline constexpr std::size_t gemm_packed_threshold {std::size_t {1} << 11U};
#else
  inline constexpr std::size_t gemm_packed_threshold {std::size_t {1} << 13U};
#endif

  //! \brief Below this depth the plain loop runs too: each panel reads and writes its tiles of C once,
  //!        which a shallow product does not amortise (1000×1000×1 ran six times slower packed, ×4 still
  //!        lost 10 %).
  inline constexpr std::size_t gemm_min_depth {8};

  template <typename T>
  using gemm_tile = std::array<T, gemm_mr * gemm_nr<T>>;

  /*!
   * \brief `tile += Aₚ·Bₚ` over one panel, for any scalar.
   * \param[in]     kc   The panel depth.
   * \param[in]     ap   The A sliver, `kc` columns of \ref gemm_mr elements.
   * \param[in]     bp   The B sliver, `kc` rows of \ref gemm_nr elements.
   * \param[in]     ldb  The distance between its rows.
   * \param[in,out] tile The accumulators, row-major.
   */
  template <typename T>
  void gemm_kernel(std::size_t   kc,
                   const T*      ap,
                   const T*      bp,
                   std::size_t   ldb,
                   gemm_tile<T>& tile)
  {
    constexpr std::size_t nr {gemm_nr<T>};
    for (std::size_t p {0}; p < kc; ++p) {
      for (std::size_t i {0}; i < gemm_mr; ++i) {
        const T ai {ap[(p * gemm_mr) + i]};
        for (std::size_t j {0}; j < nr; ++j) {
          tile[(i * nr) + j] += ai * bp[(p * ldb) + j];
        }
      }
    }
  }

#if defined(__GNUC__)
  /*!
   * \brief \ref gemm_kernel for `float` and `double`, the tile held in native vectors. Each lane sums
   *        its element in the same order as the scalar kernel.
   */
  template <typename T>
  requires std::is_same_v<T, float> || std::is_same_v<T, double>
  void gemm_kernel(std::size_t   kc,
                   const T*      ap,
                   const T*      bp,
                   std::size_t   ldb,
                   gemm_tile<T>& tile)
  {
    using lane                 = typename native_vector<T>::type;
    constexpr std::size_t             nr    {gemm_nr<T>};
    constexpr std::size_t             lanes {nr * sizeof(T) / native_vector_bytes};
    // The tile moves in and out one vector at a time: a single copy of the whole array would leave it in
    // memory under GCC, which then keeps no accumulator in a register (7.4 GFLOP/s instead of 11.8 on one
    // x86-64 core).
    std::array<lane, gemm_mr * lanes> acc {};
#  pragma GCC unroll 32
    for (std::size_t q {0}; q < acc.size(); ++q) {
      lane v;
      std::memcpy(&v, tile.data() + (q * (native_vector_bytes / sizeof(T))), sizeof(v));
      acc[q] = v;
    }
    // Unrolled, so that every accumulator has a fixed index and stays in a register: GCC at -O2 does not
    // unroll on its own, and then reads and writes the tile through memory.
    for (std::size_t p {0}; p < kc; ++p) {
      std::array<lane, lanes> b {};
      std::memcpy(b.data(), bp + (p * ldb), sizeof(b));
#  pragma GCC unroll 8
      for (std::size_t i {0}; i < gemm_mr; ++i) {
        const T ai {ap[(p * gemm_mr) + i]};
#  pragma GCC unroll 8
        for (std::size_t j {0}; j < lanes; ++j) {
          acc[(i * lanes) + j] += ai * b[j];
        }
      }
    }
#  pragma GCC unroll 32
    for (std::size_t q {0}; q < acc.size(); ++q) {
      const lane v {acc[q]};
      std::memcpy(tile.data() + (q * (native_vector_bytes / sizeof(T))), &v, sizeof(v));
    }
  }

#endif

#ifdef NUMERICAL_NEON_COMPILED
  /*!
   * \brief \ref gemm_kernel for `double` on the NEON path: the tile in 24 vectors of two, each multiply-add fused, by
   *        A's element in its lane (A's column in two vectors). GCC fuses no product of vector extensions into its
   *        sum, even under -ffp-contract=on: the portable tile ran at half speed there, its accumulators spilled
   *        (one M1 core, a product of order 512: 23 GFLOP/s; fused, 46). Each element is the AMX's chain of
   *        multiply-adds, in the same order as the scalar kernel.
   */
  inline void neon_kernel(std::size_t        kc,
                          const double*      ap,
                          const double*      bp,
                          std::size_t        ldb,
                          gemm_tile<double>& tile) noexcept
  {
    static_assert(gemm_mr == 4 && gemm_nr<double> == 12, "two rows of A a vector, twelve columns six vectors");
    constexpr std::size_t              lanes {6};
    std::array<float64x2_t, 4 * lanes> acc   {};
    NUMERICAL_UNROLL_BLOCK
    for (std::size_t q {0}; q < acc.size(); ++q) {
      acc[q] = vld1q_f64(tile.data() + (2 * q));
    }
    for (std::size_t p {0}; p < kc; ++p) {
      const float64x2_t top    {vld1q_f64(ap + (p * gemm_mr))};
      const float64x2_t bottom {vld1q_f64(ap + (p * gemm_mr) + 2)};
      NUMERICAL_UNROLL_BLOCK
      for (std::size_t j {0}; j < lanes; ++j) {
        const float64x2_t b {vld1q_f64(bp + (p * ldb) + (2 * j))};
        acc[j]               = vfmaq_laneq_f64(acc[j], b, top, 0);
        acc[lanes + j]       = vfmaq_laneq_f64(acc[lanes + j], b, top, 1);
        acc[(2 * lanes) + j] = vfmaq_laneq_f64(acc[(2 * lanes) + j], b, bottom, 0);
        acc[(3 * lanes) + j] = vfmaq_laneq_f64(acc[(3 * lanes) + j], b, bottom, 1);
      }
    }
    NUMERICAL_UNROLL_BLOCK
    for (std::size_t q {0}; q < acc.size(); ++q) {
      vst1q_f64(tile.data() + (2 * q), acc[q]);
    }
  }

  //! \brief \ref neon_kernel for `float`: the tile in 24 vectors of four, A's column one vector (GCC's portable
  //!        tile, unfused: 33 GFLOP/s on one M1 core at order 512, Clang's fused 65).
  inline void neon_kernel(std::size_t       kc,
                          const float*      ap,
                          const float*      bp,
                          std::size_t       ldb,
                          gemm_tile<float>& tile) noexcept
  {
    static_assert(gemm_mr == 4 && gemm_nr<float> == 24, "four rows of A a vector, 24 columns six vectors");
    constexpr std::size_t              lanes {6};
    std::array<float32x4_t, 4 * lanes> acc   {};
    NUMERICAL_UNROLL_BLOCK
    for (std::size_t q {0}; q < acc.size(); ++q) {
      acc[q] = vld1q_f32(tile.data() + (4 * q));
    }
    for (std::size_t p {0}; p < kc; ++p) {
      const float32x4_t column {vld1q_f32(ap + (p * gemm_mr))};
      NUMERICAL_UNROLL_BLOCK
      for (std::size_t j {0}; j < lanes; ++j) {
        const float32x4_t b {vld1q_f32(bp + (p * ldb) + (4 * j))};
        acc[j]               = vfmaq_laneq_f32(acc[j], b, column, 0);
        acc[lanes + j]       = vfmaq_laneq_f32(acc[lanes + j], b, column, 1);
        acc[(2 * lanes) + j] = vfmaq_laneq_f32(acc[(2 * lanes) + j], b, column, 2);
        acc[(3 * lanes) + j] = vfmaq_laneq_f32(acc[(3 * lanes) + j], b, column, 3);
      }
    }
    NUMERICAL_UNROLL_BLOCK
    for (std::size_t q {0}; q < acc.size(); ++q) {
      vst1q_f32(tile.data() + (4 * q), acc[q]);
    }
  }

#endif

  /*!
   * \brief Copies `rows × kc` of A (from row-major, leading dimension \p lda) into slivers of \p Mr rows, each
   *        stored column by column; a last partial sliver is padded with zeros.
   */
  template <std::size_t Mr,
            typename T>
  void gemm_pack_a(const T*    a,
                   std::size_t lda,
                   std::size_t rows,
                   std::size_t kc,
                   T*          out)
  {
    for (std::size_t ir {0}; ir < rows; ir += Mr) {
      const std::size_t mr     {std::min(Mr, rows - ir)};
      T*                sliver {out + (ir * kc)};
      if (mr == Mr) { // a whole sliver, its loop free of the padding's test (M1 Pro, order 64 for the AMX: 1.21 → 0.96 µs)
        for (std::size_t p {0}; p < kc; ++p) {
          for (std::size_t i {0}; i < Mr; ++i) {
            sliver[(p * Mr) + i] = a[((ir + i) * lda) + p];
          }
        }
        continue;
      }
      for (std::size_t p {0}; p < kc; ++p) {
        for (std::size_t i {0}; i < Mr; ++i) {
          sliver[(p * Mr) + i] = i < mr ? a[((ir + i) * lda) + p] : T {};
        }
      }
    }
  }

  /*!
   * \brief Copies `kc × cols` of B (from row-major, leading dimension \p ldb) into slivers of \p Nr columns,
   *        each stored row by row; a last partial sliver is padded with zeros.
   */
  template <std::size_t Nr,
            typename T>
  void gemm_pack_b(const T*    b,
                   std::size_t ldb,
                   std::size_t kc,
                   std::size_t cols,
                   T*          out)
  {
    for (std::size_t jr {0}; jr < cols; jr += Nr) {
      const std::size_t width  {std::min(Nr, cols - jr)};
      T*                sliver {out + (jr * kc)};
      for (std::size_t p {0}; p < kc; ++p) {
        for (std::size_t j {0}; j < Nr; ++j) {
          sliver[(p * Nr) + j] = j < width ? b[(p * ldb) + jr + j] : T {};
        }
      }
    }
  }

  /*!
   * \brief The vector path's tile: \ref gemm_mr × \ref gemm_nr elements of C in registers (\ref gemm_kernel),
   *        each product rounded before its sum unless the compiler contracts `c + a·b` into a fused multiply-add.
   */
  template <typename T>
  struct vector_tile_kernel
  {
    static constexpr std::size_t mr           {gemm_mr};             //!< The tile's rows.
    static constexpr std::size_t nr           {gemm_nr<T>};          //!< The tile's columns.
    static constexpr std::size_t depth        {gemm_kc};             //!< The panel depth.
    static constexpr std::size_t block_rows   {gemm_mc};             //!< The rows of one task, a whole number of tiles.
    static constexpr std::size_t speedup      {gemm_kernel_speedup}; //!< The weight of its work (\ref gemm_kernel_speedup).
    static constexpr std::size_t serial_below {1};                   //!< Every product with work shares it (\ref gemm_shares).

    //! \brief Nothing to turn on around a task.
    struct session
    {};

    //! \brief Whether B's rows \p ldb apart are read in place: never, its slivers are packed.
    static constexpr bool reads_b_in_place(std::size_t /*ldb*/) noexcept
    {
      return false;
    }

    //! \brief A's slivers for this tile (\ref gemm_pack_a).
    static void pack_a(const T*    a,
                       std::size_t lda,
                       std::size_t rows,
                       std::size_t kc,
                       T*          out)
    {
      gemm_pack_a<mr>(a, lda, rows, kc, out);
    }

    //! \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols) over a panel of depth \p kc, B's rows \p ldb apart.
    static void tile(std::size_t kc,
                     const T*    ap,
                     const T*    bp,
                     std::size_t ldb,
                     T*          c,
                     std::size_t ldc,
                     std::size_t rows,
                     std::size_t cols)
    {
      gemm_tile<T> accumulators;
      accumulators.fill(T {});
      for (std::size_t i {0}; i < rows; ++i) {
        std::copy_n(c + (i * ldc), cols, accumulators.data() + (i * nr));
      }
      gemm_kernel(kc, ap, bp, ldb, accumulators);
      for (std::size_t i {0}; i < rows; ++i) {
        std::copy_n(accumulators.data() + (i * nr), cols, c + (i * ldc));
      }
    }
  };

#ifdef NUMERICAL_NEON_COMPILED
  //! \brief The NEON path's tile (\ref neon_kernel): the vector tile's shape, packing and panels, each multiply-add fused.
  template <typename T>
  struct neon_tile_kernel : vector_tile_kernel<T>
  {
    //! \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols) over a panel of depth \p kc, B's rows \p ldb apart.
    static void tile(std::size_t kc,
                     const T*    ap,
                     const T*    bp,
                     std::size_t ldb,
                     T*          c,
                     std::size_t ldc,
                     std::size_t rows,
                     std::size_t cols) noexcept
    {
      constexpr std::size_t nr           {vector_tile_kernel<T>::nr};
      gemm_tile<T>          accumulators {};
      for (std::size_t i {0}; i < rows; ++i) {
        std::copy_n(c + (i * ldc), cols, accumulators.data() + (i * nr));
      }
      neon_kernel(kc, ap, bp, ldb, accumulators);
      for (std::size_t i {0}; i < rows; ++i) {
        std::copy_n(accumulators.data() + (i * nr), cols, c + (i * ldc));
      }
    }
  };
#endif

#ifdef NUMERICAL_AMX_COMPILED
  //! \brief The AMX path's tile (\ref amx_tile_f64): \ref amx_mr × \ref amx_nr elements of C in the Z registers,
  //!        each element a chain of fused multiply-adds.
  struct amx_tile_kernel
  {
    static constexpr std::size_t mr {amx_mr}; //!< The tile's rows.
    static constexpr std::size_t nr {amx_nr}; //!< The tile's columns.
    //! \brief The panel depth: the AMX reads its operands from the L2 cache, where a deeper panel still fits, and
    //!        each panel loads and stores its tiles of C once.
    static constexpr std::size_t depth      {512};
    static constexpr std::size_t block_rows {gemm_mc};             //!< The rows of one task, a whole number of tiles.
    static constexpr std::size_t speedup    {gemm_kernel_speedup}; //!< The weight of its work (\ref gemm_kernel_speedup):
                                                                   //!< fewer threads paid AMX more (QR of order 256: 2.34 ms, 2.62 at a quarter).
    //! \brief The multiply-adds below which the product runs on the caller alone: a cluster of cores shares one
    //!        AMX, so threads beside the caller pay only when the product spans clusters (M1 Pro, order 192:
    //!        70 µs on one thread, 84 on its threads; order 248: 151 against 174; order 256: 149 against 129).
    static constexpr std::size_t serial_below {std::size_t {1} << 24};
    using session = amx_session; //!< The AMX on for a task.

    //! \brief Whether B's rows \p ldb apart are read in place: up to \ref amx_in_place_stride.
    static constexpr bool reads_b_in_place(std::size_t ldb) noexcept
    {
      return ldb <= amx_in_place_stride;
    }

    //! \brief A's slivers for this tile, written past the L1 (\ref amx_pack_a).
    static void pack_a(const double* a,
                       std::size_t   lda,
                       std::size_t   rows,
                       std::size_t   kc,
                       double*       out) noexcept
    {
      amx_pack_a(a, lda, rows, kc, out);
    }

    //! \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols) over a panel of depth \p kc, B's rows \p ldb apart.
    static void tile(std::size_t   kc,
                     const double* ap,
                     const double* bp,
                     std::size_t   ldb,
                     double*       c,
                     std::size_t   ldc,
                     std::size_t   rows,
                     std::size_t   cols) noexcept
    {
      amx_tile_f64(kc, ap, bp, ldb, c, ldc, rows, cols);
    }
  };
#endif

  //! \brief The least depth at which rows of C wider than one block (\ref narrow_columns) run by blocks in
  //!        \ref gemm_rows; one block or narrower takes any depth past 1. Against the plain loop of unknown width, on
  //!        500 random shapes below the packed kernel, the blocks took 0.11 to 0.63 times as long from a depth of 2
  //!        on arm64 (M1 Pro, geometric means by width); on x86-64, where the loop has the baseline's vectors, 0.35 to
  //!        0.90 from a depth of 8 but 1.08 to 1.31 below it past one block (six AVX2 cores, GCC 13).
#ifdef NUMERICAL_X86_COMPILED
  inline constexpr std::size_t narrow_least_depth {8};
#else
  inline constexpr std::size_t narrow_least_depth {2};
#endif

  //! \brief The columns of C one block of \ref gemm_rows keeps in registers over the whole depth.
  inline constexpr std::size_t narrow_columns {8};

  //! \brief The rows of C one block of \ref gemm_rows takes together: each element a chain of multiply-adds that
  //!         waits on its last, four rows give the units four times the chains (M1 Pro, order 8 to 24: 1.6 to 1.9
  //!         times as fast as one row; two rows on six x86-64 AVX2 cores, the fastest there, were within 7 %).
  inline constexpr std::size_t narrow_group {4};

  //! \brief `C += A·B` for \p R rows and \p N columns of C, each sum in ascending order of the depth, as the plain
  //!        loop's: the block held in registers, of a size the compiler knows. Out of line: its unrolled sizes,
  //!        inlined, spent GCC's growth budget, and the LU factorisation, which runs none of them, lost 3 to 8 % at
  //!        every order (one arm64 thread, GCC 14).
  template <std::size_t R, std::size_t N, typename T>
  NUMERICAL_OUT_OF_LINE void narrow_block(std::size_t k,
                                          const T*    a,
                                          std::size_t lda,
                                          const T*    b,
                                          std::size_t ldb,
                                          T*          c,
                                          std::size_t ldc) noexcept
  {
    std::array<T, R * N> sums {};
    NUMERICAL_UNROLL_BLOCK
    for (std::size_t e {0}; e < R * N; ++e) {
      sums[e] = c[((e / N) * ldc) + (e % N)];
    }
    NUMERICAL_SCALAR_DEPTH
    for (std::size_t p {0}; p < k; ++p) {
      NUMERICAL_UNROLL_BLOCK
      for (std::size_t r {0}; r < R; ++r) {
        const T ap {a[(r * lda) + p]};
        NUMERICAL_UNROLL_BLOCK
        for (std::size_t j {0}; j < N; ++j) {
          sums[(r * N) + j] += ap * b[(p * ldb) + j];
        }
      }
    }
    NUMERICAL_UNROLL_BLOCK
    for (std::size_t e {0}; e < R * N; ++e) {
      c[((e / N) * ldc) + (e % N)] = sums[e];
    }
  }

  //! \brief `C += A·B` for \p R whole rows of C of \p n columns: blocks of \ref narrow_columns, the last narrower.
  template <std::size_t R, typename T>
  void narrow_rows(std::size_t n,
                   std::size_t k,
                   const T*    a,
                   std::size_t lda,
                   const T*    b,
                   std::size_t ldb,
                   T*          c,
                   std::size_t ldc) noexcept
  {
    static_assert(narrow_columns == 8, "the switch below names each narrower last block");
    std::size_t j {0};
    for (; j + narrow_columns <= n; j += narrow_columns) {
      narrow_block<R, narrow_columns>(k, a, lda, b + j, ldb, c + j, ldc);
    }
    switch (n - j) {
      case 1: narrow_block<R, 1>(k, a, lda, b + j, ldb, c + j, ldc); break;
      case 2: narrow_block<R, 2>(k, a, lda, b + j, ldb, c + j, ldc); break;
      case 3: narrow_block<R, 3>(k, a, lda, b + j, ldb, c + j, ldc); break;
      case 4: narrow_block<R, 4>(k, a, lda, b + j, ldb, c + j, ldc); break;
      case 5: narrow_block<R, 5>(k, a, lda, b + j, ldb, c + j, ldc); break;
      case 6: narrow_block<R, 6>(k, a, lda, b + j, ldb, c + j, ldc); break;
      case 7: narrow_block<R, 7>(k, a, lda, b + j, ldb, c + j, ldc); break;
      default: break; // a whole number of blocks
    }
  }

  /*!
   * \brief `C += A·B` by the plain i-k-j loop, parallel over the rows of C. Past a depth of 1 (\ref
   *        narrow_least_depth), the rows run \ref narrow_group at a time by blocks (\ref narrow_rows), the last rows
   *        one by one: the same sums, six to ten times as fast from order 4 to 16 (M1 Pro) as the loop of unknown
   *        width, whose sums the compiler cannot keep in registers. A depth of 1 streams as fast through that loop.
   * \param[in]     m,n,k       The shapes: A is `m×k`, B `k×n`, C `m×n`, all row-major.
   * \param[in]     a,lda,b,ldb The operands and the distance between their rows.
   * \param[in,out] c,ldc       The result and the distance between its rows.
   */
  template <typename T>
  void gemm_rows(std::size_t m,
                 std::size_t n,
                 std::size_t k,
                 const T*    a,
                 std::size_t lda,
                 const T*    b,
                 std::size_t ldb,
                 T*          c,
                 std::size_t ldc)
  {
    if (k >= (n <= narrow_columns ? std::size_t {2} : narrow_least_depth)) {
      parallel_for((m + narrow_group - 1) / narrow_group, narrow_group * k * n / vector_speedup, [&](std::size_t g) {
                     const std::size_t first {g * narrow_group};
                     if (first + narrow_group <= m) {
                       narrow_rows<narrow_group>(n, k, a + (first * lda), lda, b, ldb, c + (first * ldc), ldc);
                       return;
                     }
                     for (std::size_t i {first}; i < m; ++i) {
                       narrow_rows<1>(n, k, a + (i * lda), lda, b, ldb, c + (i * ldc), ldc);
                     }
                   });
      return;
    }
    parallel_for(m, k * n / vector_speedup, [&](std::size_t i) {
                   for (std::size_t p {0}; p < k; ++p) {
                     const T aip {a[(i * lda) + p]};
                     for (std::size_t j {0}; j < n; ++j) {
                       c[(i * ldc) + j] += aip * b[(p * ldb) + j];
                     }
                   }
                 });
  }

  /*!
   * \brief The storage of a product's packed operands, kept by the calling thread from one product to the next.
   *
   * A fresh allocation of megabytes is mapped anew and faulted in page by page on every product (at order 512 on
   * twenty x86-64 threads, a product took 3.6 times as long). Each operand starts a cache line, so a vector load
   * of a packed row never straddles two (one in two did, which cost an AVX-512 product a fifth of its speed).
   * The storage leaves the thread's keeping while a product uses it: a product nested in another on the same
   * thread (a scalar whose arithmetic multiplies matrices) finds none kept and takes its own. Storage past
   * \ref kept_limit is released, so one huge product holds no memory afterwards.
   */
  template <typename T>
  class packing_storage
  {
  public:

    //! \brief The most elements a thread keeps between products (32 MiB).
    static constexpr std::size_t kept_limit {(std::size_t {32} << 20U) / sizeof(T)};

    //! \brief Room for \p a_count packed elements of A and \p b_count of B.
    packing_storage(std::size_t a_count,
                    std::size_t b_count)
      : storage_ {grown(std::exchange(kept(), buffer<T> {}), a_count + padding + b_count + padding)},
        a_ {aligned(storage_.data())},
        b_ {aligned(storage_.data() + a_count + padding)}
    {}

    packing_storage(const packing_storage&)            = delete;
    packing_storage& operator=(const packing_storage&) = delete;
    packing_storage(packing_storage&&)                 = delete;
    packing_storage& operator=(packing_storage&&)      = delete;

    ~packing_storage()
    {
      kept() = storage_.size() <= kept_limit ? std::move(storage_) : buffer<T> {};
    }

    //! \brief A's packed elements.
    [[nodiscard]] T* a() const noexcept
    {
      return a_;
    }

    //! \brief B's packed elements.
    [[nodiscard]] T* b() const noexcept
    {
      return b_;
    }

    //! \brief The storage this thread keeps between products.
    [[nodiscard]] static buffer<T>& kept() noexcept
    {
      thread_local buffer<T> storage;
      return storage;
    }

  private:

    static constexpr std::size_t cache_line {64};
    static constexpr std::size_t padding    {(cache_line / sizeof(T)) + 1}; //!< Elements enough to reach a line.

    //! \brief \p storage with at least \p count elements (it only grows, so a reuse writes nothing).
    [[nodiscard]] static buffer<T> grown(buffer<T>      storage,
                                         std::size_t    count)
    {
      storage.resize(std::max(storage.size(), count), T {});
      return storage;
    }

    //! \brief The first element of \p at's run that starts a cache line (or \p at's nearest element towards one,
    //!        for an element whose size does not divide the allocator's alignment).
    [[nodiscard]] static T* aligned(T* at) noexcept
    {
      const std::size_t miss {static_cast<std::size_t>(std::bit_cast<std::uintptr_t>(at) % cache_line)};
      return at + (((cache_line - miss) % cache_line) / sizeof(T));
    }

    buffer<T>      storage_;
    T*             a_;
    T*             b_;
  };

  //! \brief Whether the product's threads pay: below \p Kernel's `serial_below` multiply-adds, the product runs on the
  //!        caller alone.
  template <typename Kernel>
  [[nodiscard]] constexpr bool gemm_shares(std::size_t m,
                                           std::size_t n,
                                           std::size_t k) noexcept
  {
    return m * n * k >= Kernel::serial_below;
  }

  /*!
   * \brief The packed product with the tile \p Kernel computes: for each block of B's columns and each panel of
   *        the depth, both operands are packed (in parallel, by the blocks the product takes), then tasks of
   *        \ref gemm_mc rows by \ref gemm_slivers_per_task slivers of columns run in parallel, each writing a
   *        disjoint block of C, one tile at a time.
   */
  template <typename Kernel,
            typename T>
  void gemm_packed(std::size_t m,
                   std::size_t n,
                   std::size_t k,
                   const T*    a,
                   std::size_t lda,
                   const T*    b,
                   std::size_t ldb,
                   T*          c,
                   std::size_t ldc)
  {
    constexpr std::size_t    mr     {Kernel::mr};
    constexpr std::size_t    nr     {Kernel::nr};
    constexpr std::size_t    nc_max {nr * 128};
    const std::size_t        a_rows {((m + mr) - 1) / mr * mr};
    const std::size_t        b_cols {((std::min(nc_max, n) + nr) - 1) / nr * nr};
    const packing_storage<T> packed {a_rows * std::min(Kernel::depth, k), b_cols * std::min(Kernel::depth, k)};
    const std::size_t        shared   {gemm_shares<Kernel>(m, n, k) ? 1U : 0U}; // a weight of zero starts no thread
    const bool               in_place {Kernel::reads_b_in_place(ldb)};          // B's whole slivers unpacked
    for (std::size_t jc {0}; jc < n; jc += nc_max) {
      const std::size_t nc      {std::min(nc_max, n - jc)};
      const std::size_t slivers {((nc + nr) - 1) / nr};
      for (std::size_t pc {0}; pc < k; pc += Kernel::depth) {
        const std::size_t kc         {std::min(Kernel::depth, k - pc)};
        const std::size_t row_blocks {((m + Kernel::block_rows) - 1) / Kernel::block_rows};
        const std::size_t groups     {((slivers + gemm_slivers_per_task) - 1) / gemm_slivers_per_task};
        // One pass packs both: B's groups of slivers first, then A's row blocks.
        // The packing weighed as the tile's work is (\p Kernel's speedup): a small product then packs in parallel too
        // (eight AVX2 cores: order 128 40 → 30 µs, 192 92 → 62).
        parallel_for(groups + row_blocks, shared * Kernel::block_rows * kc * gemm_kernel_speedup / Kernel::speedup, [&](std::size_t piece) {
                       if (piece < groups) {
                         const std::size_t jr    {piece * gemm_slivers_per_task * nr};
                         const std::size_t width {std::min(gemm_slivers_per_task * nr, nc - jr)};
                         if (!in_place || width % nr != 0) { // read in place, a partial sliver is still packed: its outside zero
                           gemm_pack_b<nr>(b + (pc * ldb) + jc + jr, ldb, kc, width, packed.b() + (jr * kc));
                         }
                       }
                       else {
                         const std::size_t ic {(piece - groups) * Kernel::block_rows};
                         Kernel::pack_a(a + (ic * lda) + pc, lda, std::min(Kernel::block_rows, m - ic), kc, packed.a() + (ic * kc));
                       }
                     });
        // A task covers task_rows rows: a whole block, or fewer whole tiles when blocks would leave a thread fewer
        // than tasks_per_thread tasks (order 256 on twenty x86-64 threads: twelve tasks took 116 µs, 44 took 104).
        const std::size_t tiles      {((m + mr) - 1) / mr};
        const std::size_t row_tasks  {(((tasks_per_thread * max_threads()) + groups) - 1) / groups};
        const std::size_t task_rows  {std::clamp(((tiles + row_tasks) - 1) / row_tasks, std::size_t {1}, Kernel::block_rows / mr) * mr};
        const std::size_t task_count {((m + task_rows) - 1) / task_rows};
        parallel_for(task_count * groups, shared * task_rows * gemm_slivers_per_task * nr * kc / Kernel::speedup, [&](std::size_t task) {
                       const std::size_t                   ic     {(task / groups) * task_rows};
                       const std::size_t                   first  {(task % groups) * gemm_slivers_per_task};
                       const std::size_t                   last   {std::min(first + gemm_slivers_per_task, slivers)};
                       const std::size_t                   ic_end {std::min(ic + task_rows, m)};
                       [[maybe_unused]] const typename Kernel::session on {};
                       // A sliver (the operand each step broadcasts) stays in the L1 cache across the task's B slivers,
                       // which stream from the L2, as BLIS orders its loops: B sliver outer, its 32 KiB filled the L1
                       // (a product of order 2048 on one AVX-512 core: 72 GFLOP/s, 77 in this order; AVX2 and AMX gain too).
                       for (std::size_t ir {ic}; ir < ic_end; ir += mr) {
                         for (std::size_t s {first}; s < last; ++s) {
                           const std::size_t jr {s * nr};
                           const bool direct {in_place && jr + nr <= nc};
                           Kernel::tile(kc, packed.a() + (ir * kc), direct ? b + (pc * ldb) + jc + jr : packed.b() + (jr * kc),
                                        direct ? ldb : nr, c + (ir * ldc) + jc + jr, ldc, std::min(mr, m - ir), std::min(nr, nc - jr));
                         }
                       }
                     });
      }
    }
  }

#ifdef NUMERICAL_X86_COMPILED
  //! \brief The AVX-512 tile (\ref avx512_tile_f64): \ref avx512_mr × \ref avx512_nr elements of C in 24 registers.
  struct avx512_tile_kernel
  {
    static constexpr std::size_t mr         {avx512_mr};           //!< The tile's rows.
    static constexpr std::size_t nr         {avx512_nr};           //!< The tile's columns.
    static constexpr std::size_t depth      {gemm_kc};             //!< The panel depth.
    static constexpr std::size_t block_rows {96};                  //!< The rows of one task, a whole number of tiles.
    //! \brief The weight of its work, a quarter of \ref gemm_kernel_speedup, as AVX2's (ten AVX-512 cores: order 128
    //!        90 → 51 µs, 192 155 → 94, 384 580 → 450, 512 940 → 860; an eighth slowed order 64 15 → 24 µs).
    static constexpr std::size_t speedup      {gemm_kernel_speedup / 4};
    static constexpr std::size_t serial_below {1}; //!< Every product with work shares it (\ref gemm_shares).

    //! \brief Nothing to turn on around a task.
    struct session
    {};

    //! \brief Whether B's rows \p ldb apart are read in place: never, its slivers are packed.
    static constexpr bool reads_b_in_place(std::size_t /*ldb*/) noexcept
    {
      return false;
    }

    //! \brief A's slivers for this tile (\ref gemm_pack_a).
    static void pack_a(const double* a,
                       std::size_t   lda,
                       std::size_t   rows,
                       std::size_t   kc,
                       double*       out)
    {
      gemm_pack_a<mr>(a, lda, rows, kc, out);
    }

    //! \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols) over a panel of depth \p kc, B's rows \p ldb apart.
    static void tile(std::size_t   kc,
                     const double* ap,
                     const double* bp,
                     std::size_t   ldb,
                     double*       c,
                     std::size_t   ldc,
                     std::size_t   rows,
                     std::size_t   cols) noexcept
    {
      avx512_tile_f64(kc, ap, bp, ldb, c, ldc, rows, cols);
    }
  };

  //! \brief The AVX2 tile (\ref avx2_tile_f64): \ref avx2_mr × \ref avx2_nr elements of C in 12 registers.
  struct avx2_tile_kernel
  {
    static constexpr std::size_t mr         {avx2_mr}; //!< The tile's rows.
    static constexpr std::size_t nr         {avx2_nr}; //!< The tile's columns.
    static constexpr std::size_t depth      {gemm_kc}; //!< The panel depth.
    static constexpr std::size_t block_rows {72};      //!< The rows of one task, a whole number of tiles.
    //! \brief The weight of its work, a quarter of \ref gemm_kernel_speedup: more threads paid on physical cores (eight
    //!        AVX2 cores: order 128 73 → 39 µs, 192 107 → 90; Cholesky of order 1024 20 → 16 ms).
    static constexpr std::size_t speedup      {gemm_kernel_speedup / 4};
    static constexpr std::size_t serial_below {1}; //!< Every product with work shares it (\ref gemm_shares).

    //! \brief Nothing to turn on around a task.
    struct session
    {};

    //! \brief Whether B's rows \p ldb apart are read in place: never, its slivers are packed.
    static constexpr bool reads_b_in_place(std::size_t /*ldb*/) noexcept
    {
      return false;
    }

    //! \brief A's slivers for this tile (\ref gemm_pack_a).
    static void pack_a(const double* a,
                       std::size_t   lda,
                       std::size_t   rows,
                       std::size_t   kc,
                       double*       out)
    {
      gemm_pack_a<mr>(a, lda, rows, kc, out);
    }

    //! \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols) over a panel of depth \p kc, B's rows \p ldb apart.
    static void tile(std::size_t   kc,
                     const double* ap,
                     const double* bp,
                     std::size_t   ldb,
                     double*       c,
                     std::size_t   ldc,
                     std::size_t   rows,
                     std::size_t   cols) noexcept
    {
      avx2_tile_f64(kc, ap, bp, ldb, c, ldc, rows, cols);
    }
  };
  //! \brief The AVX-512 complex tile (\ref avx512_full_tile_c64).
  struct avx512_complex_tile_kernel
  {
    static constexpr std::size_t mr           {avx512_complex_mr};        //!< The tile's rows.
    static constexpr std::size_t nr           {avx512_complex_nr};        //!< The tile's columns.
    static constexpr std::size_t depth        {gemm_kc / 2};              //!< The panel depth: an element is two `double`.
    static constexpr std::size_t block_rows   {96};                       //!< The rows of one task, a whole number of tiles.
    static constexpr std::size_t speedup      {gemm_kernel_speedup / 16}; //!< A complex multiply-add is four real ones.
    static constexpr std::size_t serial_below {1};                        //!< Every product with work shares it.

    //! \brief Nothing to turn on around a task.
    struct session
    {};

    //! \brief Whether B's rows \p ldb apart are read in place: never, its slivers are packed.
    static constexpr bool reads_b_in_place(std::size_t /*ldb*/) noexcept
    {
      return false;
    }

    //! \brief A's slivers for this tile (\ref gemm_pack_a).
    static void pack_a(const std::complex<double>* a,
                       std::size_t                 lda,
                       std::size_t                 rows,
                       std::size_t                 kc,
                       std::complex<double>*       out)
    {
      gemm_pack_a<mr>(a, lda, rows, kc, out);
    }

    //! \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols) over a panel of depth \p kc, B's rows \p ldb apart.
    static void tile(std::size_t                 kc,
                     const std::complex<double>* ap,
                     const std::complex<double>* bp,
                     std::size_t                 ldb,
                     std::complex<double>*       c,
                     std::size_t                 ldc,
                     std::size_t                 rows,
                     std::size_t                 cols) noexcept
    {
      avx512_tile_c64(kc, ap, bp, ldb, c, ldc, rows, cols);
    }
  };

  //! \brief The AVX2 complex tile (\ref avx2_full_tile_c64).
  struct avx2_complex_tile_kernel
  {
    static constexpr std::size_t mr           {avx2_complex_mr};          //!< The tile's rows.
    static constexpr std::size_t nr           {avx2_complex_nr};          //!< The tile's columns.
    static constexpr std::size_t depth        {gemm_kc / 2};              //!< The panel depth: an element is two `double`.
    static constexpr std::size_t block_rows   {72};                       //!< The rows of one task, a whole number of tiles.
    static constexpr std::size_t speedup      {gemm_kernel_speedup / 16}; //!< A complex multiply-add is four real ones.
    static constexpr std::size_t serial_below {1};                        //!< Every product with work shares it.

    //! \brief Nothing to turn on around a task.
    struct session
    {};

    //! \brief Whether B's rows \p ldb apart are read in place: never, its slivers are packed.
    static constexpr bool reads_b_in_place(std::size_t /*ldb*/) noexcept
    {
      return false;
    }

    //! \brief A's slivers for this tile (\ref gemm_pack_a).
    static void pack_a(const std::complex<double>* a,
                       std::size_t                 lda,
                       std::size_t                 rows,
                       std::size_t                 kc,
                       std::complex<double>*       out)
    {
      gemm_pack_a<mr>(a, lda, rows, kc, out);
    }

    //! \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols) over a panel of depth \p kc, B's rows \p ldb apart.
    static void tile(std::size_t                 kc,
                     const std::complex<double>* ap,
                     const std::complex<double>* bp,
                     std::size_t                 ldb,
                     std::complex<double>*       c,
                     std::size_t                 ldc,
                     std::size_t                 rows,
                     std::size_t                 cols) noexcept
    {
      avx2_tile_c64(kc, ap, bp, ldb, c, ldc, rows, cols);
    }
  };
#endif

  //! \brief A product a faster path than the vector one takes, if any: none for this scalar.
  template <typename T>
  [[nodiscard]] bool gemm_elsewhere(std::size_t m,
                                    std::size_t n,
                                    std::size_t k,
                                    const T*    a,
                                    std::size_t lda,
                                    const T*    b,
                                    std::size_t ldb,
                                    T*          c,
                                    std::size_t ldc)
  {
    (void)m, (void)n, (void)k, (void)a, (void)lda, (void)b, (void)ldb, (void)c, (void)ldc;
    return false;
  }

  //! \brief A `double` product on the path in use (\ref numerical::gemm_path_in_use), unless it is the portable one.
  //!        \return Whether it ran there.
  [[nodiscard]] inline bool gemm_elsewhere(std::size_t   m,
                                           std::size_t   n,
                                           std::size_t   k,
                                           const double* a,
                                           std::size_t   lda,
                                           const double* b,
                                           std::size_t   ldb,
                                           double*       c,
                                           std::size_t   ldc)
  {
    const gemm_path path {gemm_path_current()};
#ifdef NUMERICAL_AMX_COMPILED
    if (path == gemm_path::amx && amx_suits(m, n, k)) { // else the NEON tile
      gemm_packed<amx_tile_kernel>(m, n, k, a, lda, b, ldb, c, ldc);
      return true;
    }
#endif
#ifdef NUMERICAL_NEON_COMPILED
    if (path == gemm_path::neon || path == gemm_path::amx) {
      gemm_packed<neon_tile_kernel<double>>(m, n, k, a, lda, b, ldb, c, ldc);
      return true;
    }
#endif
#ifdef NUMERICAL_X86_COMPILED
    if (path == gemm_path::avx512) {
      gemm_packed<avx512_tile_kernel>(m, n, k, a, lda, b, ldb, c, ldc);
      return true;
    }
    if (path == gemm_path::avx2) {
      gemm_packed<avx2_tile_kernel>(m, n, k, a, lda, b, ldb, c, ldc);
      return true;
    }
#endif
    (void)path;
    (void)m, (void)n, (void)k, (void)a, (void)lda, (void)b, (void)ldb, (void)c, (void)ldc;
    return false;
  }

  //! \brief A `float` product on the NEON tile on arm64's fused paths (the NEON path, and the AMX path, whose AMX takes
  //!        `double` alone); elsewhere the vector tile. \return Whether it ran on the NEON tile.
  [[nodiscard]] inline bool gemm_elsewhere(std::size_t  m,
                                           std::size_t  n,
                                           std::size_t  k,
                                           const float* a,
                                           std::size_t  lda,
                                           const float* b,
                                           std::size_t  ldb,
                                           float*       c,
                                           std::size_t  ldc)
  {
#ifdef NUMERICAL_NEON_COMPILED
    const gemm_path path {gemm_path_current()};
    if (path == gemm_path::neon || path == gemm_path::amx) {
      gemm_packed<neon_tile_kernel<float>>(m, n, k, a, lda, b, ldb, c, ldc);
      return true;
    }
#endif
    (void)m, (void)n, (void)k, (void)a, (void)lda, (void)b, (void)ldb, (void)c, (void)ldc;
    return false;
  }

  template <typename T>
  void gemm(std::size_t m,
            std::size_t n,
            std::size_t k,
            const T*    a,
            std::size_t lda,
            const T*    b,
            std::size_t ldb,
            T*          c,
            std::size_t ldc);

  //! \brief The complex numbers `gemm_split` multiplies through the real kernel.
  template <typename T>
  concept split_complex = std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

  //! \brief The columns of B and C, and the depth, of one real product of `gemm_split`: its real operands
  //!         stay a few megabytes whatever the shapes.
  inline constexpr std::size_t split_columns {512};

  //! \brief The depth of one real product of `gemm_split` (\ref split_columns).
  inline constexpr std::size_t split_depth {256};

  //! \brief The most elements an operand of `gemm_split` keeps from one product to the next on a thread: 2 MiB of
  //!        `double`, every operand of a product of order 512; a larger one is let go after its product.
  inline constexpr std::size_t split_kept {std::size_t {1} << 18};

  //! \brief The real operands of `gemm_split`, kept by each thread from one product to the next (\ref split_storage):
  //!        mapping their pages anew took a third to a half of a complex product of order 64 to 256 on one x86-64 core
  //!        (six AVX2 cores, all threads: order 128 0.52 → 0.17 ms), 15–20 % of one of order 256 on arm64.
  template <typename R>
  struct split_operands
  {
    buffer<R> a {}; //!< `[Re A | −Im A]`, a block of depth.
    buffer<R> b {}; //!< `[[Re B, Im B], [Im B, −Re B]]`, a block of columns and of depth.
    buffer<R> c {}; //!< `[Re C | Im C]`, a block of columns.
  };

  /*!
   * \brief The calling thread's \ref split_operands. A product uses them whole on one thread, and the real product it
   *        runs on them nests no complex one.
   */
  template <typename R>
  [[nodiscard]] split_operands<R>& split_storage()
  {
    thread_local split_operands<R> kept;
    return kept;
  }

  /*!
   * \brief `C += A·B` for complex numbers through the real kernel, by blocks of \ref split_columns columns and
   *        \ref split_depth of depth: each block is one real product, `[Re C | Im C] += [Re A | −Im A] ·
   *        [[Re B, Im B], [Im B, −Re B]]`, of the multiply-adds the complex product has, on the real path's
   *        units and threads (a scalar complex tile has none). Within a block, each element of C takes the
   *        products of the real parts of A in ascending order of the depth, then those of the imaginary parts.
   */
  template <split_complex Z>
  void gemm_split(std::size_t m,
                  std::size_t n,
                  std::size_t k,
                  const Z*    a,
                  std::size_t lda,
                  const Z*    b,
                  std::size_t ldb,
                  Z*          c,
                  std::size_t ldc)
  {
    using R = typename Z::value_type;
    const std::size_t           widest  {std::min(n, split_columns)};
    const std::size_t           deepest {std::min(k, split_depth)};
    split_operands<R>&          kept    {split_storage<R>()};
    kept.a.resize(std::max(kept.a.size(), m * 2 * deepest)); // grown only, its elements written before they are read
    kept.b.resize(std::max(kept.b.size(), 4 * deepest * widest));
    kept.c.resize(std::max(kept.c.size(), m * 2 * widest));
    R* const                    ar      {kept.a.data()};
    R* const                    br      {kept.b.data()};
    R* const                    cr      {kept.c.data()};
    for (std::size_t jc {0}; jc < n; jc += split_columns) {
      const std::size_t nb {std::min(split_columns, n - jc)};
      parallel_for(m, nb, [&](std::size_t i) {
                     for (std::size_t j {0}; j < nb; ++j) {
                       cr[(i * 2 * nb) + j]      = c[(i * ldc) + jc + j].real();
                       cr[(i * 2 * nb) + nb + j] = c[(i * ldc) + jc + j].imag();
                     }
                   });
      for (std::size_t pc {0}; pc < k; pc += split_depth) {
        const std::size_t kb {std::min(split_depth, k - pc)};
        parallel_for(m, kb, [&](std::size_t i) {
                       for (std::size_t p {0}; p < kb; ++p) {
                         ar[(i * 2 * kb) + p]      = a[(i * lda) + pc + p].real();
                         ar[(i * 2 * kb) + kb + p] = -a[(i * lda) + pc + p].imag();
                       }
                     });
        parallel_for(kb, 2 * nb, [&](std::size_t p) {
                       R* const top {br + (p * 2 * nb)};
                       R* const bottom {br + ((kb + p) * 2 * nb)};
                       for (std::size_t j {0}; j < nb; ++j) {
                         const Z value {b[((pc + p) * ldb) + jc + j]};
                         top[j]         = value.real();
                         top[nb + j]    = value.imag();
                         bottom[j]      = value.imag();
                         bottom[nb + j] = -value.real();
                       }
                     });
        gemm(m, 2 * nb, 2 * kb, ar, 2 * kb, br, 2 * nb, cr, 2 * nb);
      }
      parallel_for(m, nb, [&](std::size_t i) {
                     for (std::size_t j {0}; j < nb; ++j) {
                       c[(i * ldc) + jc + j] = Z {cr[(i * 2 * nb) + j], cr[(i * 2 * nb) + nb + j]};
                     }
                   });
    }
    for (buffer<R>* operand : {&kept.a, &kept.b, &kept.c}) {
      if (operand->size() > split_kept) {
        *operand = buffer<R> {};
      }
    }
  }

  //! \brief A product past the plain loop: on the AMX or x86-64 tiles where \ref gemm_elsewhere takes it, else on
  //!        the vector tile.
  template <typename T>
  void gemm_large(std::size_t m,
                  std::size_t n,
                  std::size_t k,
                  const T*    a,
                  std::size_t lda,
                  const T*    b,
                  std::size_t ldb,
                  T*          c,
                  std::size_t ldc)
  {
    if (!gemm_elsewhere(m, n, k, a, lda, b, ldb, c, ldc)) {
      gemm_packed<vector_tile_kernel<T>>(m, n, k, a, lda, b, ldb, c, ldc);
    }
  }

  //! \brief A complex product past the plain loop: on the vector tile on the portable path, whole complex products
  //!        summed in order; complex `double` on the x86-64 complex tiles; else through the real kernel (`gemm_split`).
  template <split_complex T>
  void gemm_large(std::size_t m,
                  std::size_t n,
                  std::size_t k,
                  const T*    a,
                  std::size_t lda,
                  const T*    b,
                  std::size_t ldb,
                  T*          c,
                  std::size_t ldc)
  {
    const gemm_path path {gemm_path_current()};
    if (path == gemm_path::portable) {
      gemm_packed<vector_tile_kernel<T>>(m, n, k, a, lda, b, ldb, c, ldc);
      return;
    }
#ifdef NUMERICAL_X86_COMPILED
    if constexpr (std::is_same_v<T, std::complex<double>>) { // a complex tile of its own on the x86-64 units
      if (path == gemm_path::avx512) {
        gemm_packed<avx512_complex_tile_kernel>(m, n, k, a, lda, b, ldb, c, ldc);
        return;
      }
      if (path == gemm_path::avx2) {
        gemm_packed<avx2_complex_tile_kernel>(m, n, k, a, lda, b, ldb, c, ldc);
        return;
      }
    }
#endif
    gemm_split(m, n, k, a, lda, b, ldb, c, ldc);
  }

  /*!
   * \brief `C += A·B`, every element of C summed in ascending order of the depth.
   *
   * Products of fewer than \ref gemm_packed_threshold multiply-adds, or shallower than \ref gemm_min_depth, run
   * \ref gemm_rows. The rest run packed (\ref gemm_packed). A `double` product runs on the AMX where it is in use
   * and suits the product (`amx_suits`), else on the NEON or AVX tiles: each element a chain of fused
   * multiply-adds; a `float` one on the NEON tile on arm64's NEON and AMX paths. On the portable path either runs on the vector tile, which rounds `c + a·b` as the compiler does:
   * each product before its sum with contraction off, as the plain loop then does; fused under Clang's
   * -ffp-contract=on, as the plain loop then is; under GCC's, apart, where the plain loop's scalar statement is
   * fused. Every element is summed in ascending order of the depth on any number of threads. Complex `double` on the x86-64
   * paths runs on their complex tiles (`avx2_full_tile_c64`, `avx512_full_tile_c64`); complex `float` and
   * `double` past the portable path otherwise run through the real kernel (`gemm_split`). Either is the same on
   * any number of threads too, but no longer the plain loop's to the bit.
   *
   * \param[in]     m,n,k       The shapes: A is `m×k`, B `k×n`, C `m×n`, all row-major.
   * \param[in]     a,lda,b,ldb The operands and the distance between their rows.
   * \param[in,out] c,ldc       The result and the distance between its rows (C may be a block of a
   *                             larger matrix that A and B do not overlap).
   * A zero `m`, `n` or `k` leaves C as it is.
   */
  template <typename T>
  void gemm(std::size_t m,
            std::size_t n,
            std::size_t k,
            const T*    a,
            std::size_t lda,
            const T*    b,
            std::size_t ldb,
            T*          c,
            std::size_t ldc)
  {
    if (k < gemm_min_depth || m * n < ((gemm_packed_threshold + k) - 1) / k) {
      gemm_rows(m, n, k, a, lda, b, ldb, c, ldc);
      return;
    }
    gemm_large(m, n, k, a, lda, b, ldb, c, ldc);
  }

  //! \brief The elements of one partial sum of `chunked_dot`: fixed, so the sum does not depend on the threads.
  inline constexpr std::size_t dot_chunk {std::size_t {1} << 14U};

  /*!
   * \brief `Σ a[i] · b[i]` for a plain dot product: chunks of \ref dot_chunk elements in parallel, each by
   *        \ref dot_product (several vector accumulators, on the widest x86-64 units for `double`), their sums
   *        added in the chunks' order. The same on any number of threads; in the last bits, not the ascending
   *        sum, which waits on each addition before the next (ten times slower at 100 000 elements).
   */
  template <typename T>
  [[nodiscard]] T chunked_dot(const T*    a,
                              const T*    b,
                              std::size_t count)
  {
    const std::size_t chunks {(count + dot_chunk - 1) / dot_chunk};
    if (chunks <= 1) {
      return dot_product(a, b, count);
    }
    buffer<T> partial(chunks);
    parallel_for(chunks, dot_chunk, [&](std::size_t c) {
                   const std::size_t first {c * dot_chunk};
                   partial[c] = dot_product(a + first, b + first, std::min(dot_chunk, count - first));
                 });
    T total {};
    for (const T& sum : partial) {
      total += sum;
    }
    return total;
  }

  //! \brief The rows a thread takes of a matrix times a vector (\ref matrix_times_vector).
  inline constexpr std::size_t row_group {64};

  /*!
   * \brief `y = A x` for the row-major `rows × cols` \p a: each element the dot product of its row with \p x, as
   *        \ref chunked_dot sums it (a row of a chunk or less is one \ref dot_product), the rows in parallel by
   *        groups (\ref row_dots, which share each load of \p x). The same on any number of threads.
   */
  template <typename T>
  void matrix_times_vector(const T*    a,
                           std::size_t rows,
                           std::size_t cols,
                           const T*    x,
                           T*          y)
  {
    if (cols > dot_chunk) {   // rows longer than a chunk: chunked_dot's partial sums
      parallel_for(rows, cols, [&](std::size_t r) { y[r] = chunked_dot(a + (r * cols), x, cols); });
      return;
    }
    parallel_for((rows + row_group - 1) / row_group, row_group * cols, [&](std::size_t group) {
                   const std::size_t first {group * row_group};
                   row_dots(a + (first * cols), cols, std::min(row_group, rows - first), x, cols, y + first);
                 });
  }
} // namespace numerical::detail

namespace numerical {

  /*!
   * \brief Keeps `double` matrix products on \p most or a more portable path: the most capable path at or below
   *        it that can run here. `gemm_path::amx` restores the best one; `gemm_path::portable` keeps them on the
   *        compiler's own vectors, with no instruction chosen at run time. Paths below the cap are not probed.
   * \return The path now in use.
   */
  inline gemm_path limit_gemm_path(gemm_path most) noexcept
  {
    detail::gemm_path_cap().store(most, std::memory_order_relaxed);
    return detail::best_gemm_path(most); // probes only the paths at or below the cap
  }

  //! \brief The path `double` matrix products take.
  [[nodiscard]] inline gemm_path gemm_path_in_use() noexcept
  {
    return detail::gemm_path_current();
  }
} // namespace numerical

#endif // NUMERICAL_GEMM_HPP
