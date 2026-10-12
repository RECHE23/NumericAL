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
#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "parallel.hpp"
#include "amx.hpp"
#include "simd.hpp"

namespace numerical::detail {

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

  //! \brief The multiply-adds the register tile does in the time of one in \ref work_per_thread's dependent
  //!        chain (about 20 G/s on one arm64 or x86-64 AVX2 core): a task weighs its work by it, so a product
  //!        of 0.1 ms does not start eight threads.
  inline constexpr std::size_t gemm_kernel_speedup {16};

  //! \brief The columns of B packed at once, a whole number of slivers.
  template <typename T>
  inline constexpr std::size_t gemm_nc {gemm_nr<T> * 128};

  //! \brief Below this many multiply-adds the plain i-k-j loop runs instead of the packed kernel (both
  //!        sum in the same order): the kernel already wins at order 12 (3× on arm64), and ties at 8.
  inline constexpr std::size_t gemm_packed_threshold {std::size_t {8} *8 * 8};

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
   * \param[in,out] tile The accumulators, row-major.
   */
  template <typename T>
  void gemm_kernel(std::size_t   kc,
                   const T*      ap,
                   const T*      bp,
                   gemm_tile<T>& tile)
  {
    constexpr std::size_t nr {gemm_nr<T>};
    for (std::size_t p {0}; p < kc; ++p) {
      for (std::size_t i {0}; i < gemm_mr; ++i) {
        const T ai {ap[(p * gemm_mr) + i]};
        for (std::size_t j {0}; j < nr; ++j) {
          tile[(i * nr) + j] += ai * bp[(p * nr) + j];
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
                   gemm_tile<T>& tile)
  {
    using lane                 = typename native_vector<T>::type;
    constexpr std::size_t             nr    {gemm_nr<T>};
    constexpr std::size_t             lanes {nr * sizeof(T) / native_vector_bytes};
    // The tile moves in and out one vector at a time: a single copy of the whole array would leave it in
    // memory under GCC, which then keeps no accumulator in a register (7.4 GFLOP/s instead of 11.8 on one
    // x86-64 core).
    std::array<lane, gemm_mr * lanes> acc;
#  pragma GCC unroll 32
    for (std::size_t q {0}; q < acc.size(); ++q) {
      lane v;
      std::memcpy(&v, tile.data() + (q * (native_vector_bytes / sizeof(T))), sizeof(v));
      acc[q] = v;
    }
    // Unrolled, so that every accumulator has a fixed index and stays in a register: GCC at -O2 does not
    // unroll on its own, and then reads and writes the tile through memory.
    for (std::size_t p {0}; p < kc; ++p) {
      std::array<lane, lanes> b;
      std::memcpy(b.data(), bp + (p * nr), sizeof(b));
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
   *        each product rounded before its sum.
   */
  template <typename T>
  struct vector_tile_kernel
  {
    static constexpr std::size_t mr    {gemm_mr};    //!< The tile's rows.
    static constexpr std::size_t nr    {gemm_nr<T>}; //!< The tile's columns.
    static constexpr std::size_t depth {gemm_kc};    //!< The panel depth.

    //! \brief Nothing to turn on around a task.
    struct session
    {};

    //! \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols) over a panel of depth \p kc.
    static void tile(std::size_t kc,
                     const T*    ap,
                     const T*    bp,
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
      gemm_kernel(kc, ap, bp, accumulators);
      for (std::size_t i {0}; i < rows; ++i) {
        std::copy_n(accumulators.data() + (i * nr), cols, c + (i * ldc));
      }
    }
  };

#ifdef NUMERICAL_AMX_COMPILED
  //! \brief The AMX path's tile (\ref amx_tile_f64): \ref amx_mr × \ref amx_nr elements of C in the Z registers,
  //!        each element a chain of fused multiply-adds.
  struct amx_tile_kernel
  {
    static constexpr std::size_t mr {amx_mr}; //!< The tile's rows.
    static constexpr std::size_t nr {amx_nr}; //!< The tile's columns.
    //! \brief The panel depth: the AMX reads its operands from the L2 cache, where a deeper panel still fits, and
    //!        each panel loads and stores its tiles of C once.
    static constexpr std::size_t depth {512};
    using session = amx_session;              //!< The AMX on for a task.

    //! \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols) over a panel of depth \p kc.
    static void tile(std::size_t   kc,
                     const double* ap,
                     const double* bp,
                     double*       c,
                     std::size_t   ldc,
                     std::size_t   rows,
                     std::size_t   cols) noexcept
    {
      amx_tile_f64(kc, ap, bp, c, ldc, rows, cols);
    }
  };
#endif

  /*!
   * \brief `C += A·B` by the plain i-k-j loop, parallel over the rows of C.
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
    constexpr std::size_t mr     {Kernel::mr};
    constexpr std::size_t nr     {Kernel::nr};
    constexpr std::size_t nc_max {nr * 128};
    const std::size_t     a_rows {((m + mr) - 1) / mr * mr};
    const std::size_t     b_cols {((std::min(nc_max, n) + nr) - 1) / nr * nr};
    // Left unset: packing writes every element before a tile reads it (zeroing megabytes per call cost a
    // tenth of a product of order 512).
    const std::unique_ptr<T[]> a_packed {std::make_unique_for_overwrite<T[]>(a_rows * std::min(Kernel::depth, k))};
    const std::unique_ptr<T[]> b_packed {std::make_unique_for_overwrite<T[]>(b_cols * std::min(Kernel::depth, k))};
    for (std::size_t jc {0}; jc < n; jc += nc_max) {
      const std::size_t nc      {std::min(nc_max, n - jc)};
      const std::size_t slivers {((nc + nr) - 1) / nr};
      for (std::size_t pc {0}; pc < k; pc += Kernel::depth) {
        const std::size_t kc         {std::min(Kernel::depth, k - pc)};
        const std::size_t row_blocks {((m + gemm_mc) - 1) / gemm_mc};
        const std::size_t groups     {((slivers + gemm_slivers_per_task) - 1) / gemm_slivers_per_task};
        // One pass packs both: B's groups of slivers first, then A's row blocks.
        parallel_for(groups + row_blocks, gemm_mc * kc, [&](std::size_t piece) {
                       if (piece < groups) {
                         const std::size_t jr    {piece * gemm_slivers_per_task * nr};
                         const std::size_t width {std::min(gemm_slivers_per_task * nr, nc - jr)};
                         gemm_pack_b<nr>(b + (pc * ldb) + jc + jr, ldb, kc, width, b_packed.get() + (jr * kc));
                       }
                       else {
                         const std::size_t ic {(piece - groups) * gemm_mc};
                         gemm_pack_a<mr>(a + (ic * lda) + pc, lda, std::min(gemm_mc, m - ic), kc, a_packed.get() + (ic * kc));
                       }
                     });
        parallel_for(row_blocks * groups, gemm_mc * gemm_slivers_per_task * nr * kc / gemm_kernel_speedup, [&](std::size_t task) {
                       const std::size_t                   ic     {(task / groups) * gemm_mc};
                       const std::size_t                   first  {(task % groups) * gemm_slivers_per_task};
                       const std::size_t                   last   {std::min(first + gemm_slivers_per_task, slivers)};
                       const std::size_t                   ic_end {std::min(ic + gemm_mc, m)};
                       [[maybe_unused]] const typename Kernel::session on {};
                       for (std::size_t s {first}; s < last; ++s) {
                         const std::size_t jr {s * nr};
                         for (std::size_t ir {ic}; ir < ic_end; ir += mr) {
                           Kernel::tile(kc, a_packed.get() + (ir * kc), b_packed.get() + (jr * kc), c + (ir * ldc) + jc + jr, ldc,
                                        std::min(mr, m - ir), std::min(nr, nc - jr));
                         }
                       }
                     });
      }
    }
  }

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

  //! \brief A `double` product on the AMX, where it is in use (\ref amx_active). \return Whether it ran there.
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
#ifdef NUMERICAL_AMX_COMPILED
    if (amx_active()) {
      gemm_packed<amx_tile_kernel>(m, n, k, a, lda, b, ldb, c, ldc);
      return true;
    }
#endif
    (void)m, (void)n, (void)k, (void)a, (void)lda, (void)b, (void)ldb, (void)c, (void)ldc;
    return false;
  }

  /*!
   * \brief `C += A·B`, every element of C summed in ascending order of the depth.
   *
   * Products of fewer than \ref gemm_packed_threshold multiply-adds, or shallower than \ref gemm_min_depth, run
   * \ref gemm_rows. The rest run packed (\ref gemm_packed): on the AMX for `double` where it is in use (each
   * element a chain of fused multiply-adds), else on the vector tile (each product rounded before its sum, so
   * the plain loop and the packed one agree to the bit with the compiler's contraction off).
   *
   * \param[in]     m,n,k       The shapes: A is `m×k`, B `k×n`, C `m×n`, all row-major.
   * \param[in]     a,lda,b,ldb The operands and the distance between their rows.
   * \param[in,out] c,ldc       The result and the distance between its rows (C may be a block of a
   *                             larger matrix that A and B do not overlap).
   * \pre `m`, `n` and `k` are not zero.
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
    if (!gemm_elsewhere(m, n, k, a, lda, b, ldb, c, ldc)) {
      gemm_packed<vector_tile_kernel<T>>(m, n, k, a, lda, b, ldb, c, ldc);
    }
  }
} // namespace numerical::detail

#endif // NUMERICAL_GEMM_HPP
