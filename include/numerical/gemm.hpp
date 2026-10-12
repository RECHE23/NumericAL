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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "parallel.hpp"
#include "amx.hpp"
#include "simd.hpp"
#include "x86.hpp"

namespace numerical {

  /*!
   * \brief The kernels a `double` matrix product can run on, from the most portable. Each is used only where the
   *        processor has it: the AVX ones where `cpuid` reports them, the Apple AMX on an M1 to M4 chip whose
   *        self-test passes (see x86.hpp and amx.hpp).
   */
  enum class gemm_path : std::uint8_t
  {
    portable = 0, //!< The compiler's vectors (16 bytes on x86-64 wheels), as the compiler rounds `c + a·b` (see gemm).
    avx2     = 1, //!< AVX2 with fused multiply-adds.
    avx512   = 2, //!< AVX-512 with fused multiply-adds.
    amx      = 3, //!< The Apple AMX coprocessor, fused multiply-adds.
  };
} // namespace numerical

namespace numerical::detail {

  //! \brief Whether \p path can run here.
  [[nodiscard]] inline bool gemm_path_usable(gemm_path path) noexcept
  {
    switch (path) {
#ifdef NUMERICAL_X86_COMPILED
      case gemm_path::avx2:   return cpu_features().avx2_fma;
      case gemm_path::avx512: return cpu_features().avx512f;
#endif
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
   *        each product rounded before its sum unless the compiler contracts `c + a·b` into a fused multiply-add.
   */
  template <typename T>
  struct vector_tile_kernel
  {
    static constexpr std::size_t mr         {gemm_mr};    //!< The tile's rows.
    static constexpr std::size_t nr         {gemm_nr<T>}; //!< The tile's columns.
    static constexpr std::size_t depth      {gemm_kc};    //!< The panel depth.
    static constexpr std::size_t block_rows {gemm_mc};    //!< The rows of one task, a whole number of tiles.

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
    static constexpr std::size_t depth      {512};
    static constexpr std::size_t block_rows {gemm_mc}; //!< The rows of one task, a whole number of tiles.
    using session = amx_session;                       //!< The AMX on for a task.

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
      : storage_ {grown(std::exchange(kept(), std::vector<T> {}), a_count + padding + b_count + padding)},
        a_ {aligned(storage_.data())},
        b_ {aligned(storage_.data() + a_count + padding)}
    {}

    packing_storage(const packing_storage&)            = delete;
    packing_storage& operator=(const packing_storage&) = delete;
    packing_storage(packing_storage&&)                 = delete;
    packing_storage& operator=(packing_storage&&)      = delete;

    ~packing_storage()
    {
      kept() = storage_.size() <= kept_limit ? std::move(storage_) : std::vector<T> {};
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
    [[nodiscard]] static std::vector<T>& kept() noexcept
    {
      thread_local std::vector<T> storage;
      return storage;
    }

  private:

    static constexpr std::size_t cache_line {64};
    static constexpr std::size_t padding    {(cache_line / sizeof(T)) + 1}; //!< Elements enough to reach a line.

    //! \brief \p storage with at least \p count elements (it only grows, so a reuse writes nothing).
    [[nodiscard]] static std::vector<T> grown(std::vector<T> storage,
                                              std::size_t    count)
    {
      storage.resize(std::max(storage.size(), count));
      return storage;
    }

    //! \brief The first element of \p at's run that starts a cache line (or \p at's nearest element towards one,
    //!        for an element whose size does not divide the allocator's alignment).
    [[nodiscard]] static T* aligned(T* at) noexcept
    {
      const std::size_t miss {static_cast<std::size_t>(std::bit_cast<std::uintptr_t>(at) % cache_line)};
      return at + (((cache_line - miss) % cache_line) / sizeof(T));
    }

    std::vector<T> storage_;
    T*             a_;
    T*             b_;
  };

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
    for (std::size_t jc {0}; jc < n; jc += nc_max) {
      const std::size_t nc      {std::min(nc_max, n - jc)};
      const std::size_t slivers {((nc + nr) - 1) / nr};
      for (std::size_t pc {0}; pc < k; pc += Kernel::depth) {
        const std::size_t kc         {std::min(Kernel::depth, k - pc)};
        const std::size_t row_blocks {((m + Kernel::block_rows) - 1) / Kernel::block_rows};
        const std::size_t groups     {((slivers + gemm_slivers_per_task) - 1) / gemm_slivers_per_task};
        // One pass packs both: B's groups of slivers first, then A's row blocks.
        parallel_for(groups + row_blocks, Kernel::block_rows * kc, [&](std::size_t piece) {
                       if (piece < groups) {
                         const std::size_t jr    {piece * gemm_slivers_per_task * nr};
                         const std::size_t width {std::min(gemm_slivers_per_task * nr, nc - jr)};
                         gemm_pack_b<nr>(b + (pc * ldb) + jc + jr, ldb, kc, width, packed.b() + (jr * kc));
                       }
                       else {
                         const std::size_t ic {(piece - groups) * Kernel::block_rows};
                         gemm_pack_a<mr>(a + (ic * lda) + pc, lda, std::min(Kernel::block_rows, m - ic), kc, packed.a() + (ic * kc));
                       }
                     });
        // A task covers task_rows rows: a whole block, or fewer whole tiles when blocks would leave a thread fewer
        // than tasks_per_thread tasks (order 256 on twenty x86-64 threads: twelve tasks took 116 µs, 44 took 104).
        const std::size_t tiles      {((m + mr) - 1) / mr};
        const std::size_t row_tasks  {(((tasks_per_thread * max_threads()) + groups) - 1) / groups};
        const std::size_t task_rows  {std::clamp(((tiles + row_tasks) - 1) / row_tasks, std::size_t {1}, Kernel::block_rows / mr) * mr};
        const std::size_t task_count {((m + task_rows) - 1) / task_rows};
        parallel_for(task_count * groups, task_rows * gemm_slivers_per_task * nr * kc / gemm_kernel_speedup, [&](std::size_t task) {
                       const std::size_t                   ic     {(task / groups) * task_rows};
                       const std::size_t                   first  {(task % groups) * gemm_slivers_per_task};
                       const std::size_t                   last   {std::min(first + gemm_slivers_per_task, slivers)};
                       const std::size_t                   ic_end {std::min(ic + task_rows, m)};
                       [[maybe_unused]] const typename Kernel::session on {};
                       for (std::size_t s {first}; s < last; ++s) {
                         const std::size_t jr {s * nr};
                         for (std::size_t ir {ic}; ir < ic_end; ir += mr) {
                           Kernel::tile(kc, packed.a() + (ir * kc), packed.b() + (jr * kc), c + (ir * ldc) + jc + jr, ldc,
                                        std::min(mr, m - ir), std::min(nr, nc - jr));
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
    static constexpr std::size_t mr         {avx512_mr}; //!< The tile's rows.
    static constexpr std::size_t nr         {avx512_nr}; //!< The tile's columns.
    static constexpr std::size_t depth      {gemm_kc};   //!< The panel depth.
    static constexpr std::size_t block_rows {96};        //!< The rows of one task, a whole number of tiles.

    //! \brief Nothing to turn on around a task.
    struct session
    {};

    //! \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols) over a panel of depth \p kc.
    static void tile(std::size_t   kc,
                     const double* ap,
                     const double* bp,
                     double*       c,
                     std::size_t   ldc,
                     std::size_t   rows,
                     std::size_t   cols) noexcept
    {
      avx512_tile_f64(kc, ap, bp, c, ldc, rows, cols);
    }
  };

  //! \brief The AVX2 tile (\ref avx2_tile_f64): \ref avx2_mr × \ref avx2_nr elements of C in 12 registers.
  struct avx2_tile_kernel
  {
    static constexpr std::size_t mr         {avx2_mr}; //!< The tile's rows.
    static constexpr std::size_t nr         {avx2_nr}; //!< The tile's columns.
    static constexpr std::size_t depth      {gemm_kc}; //!< The panel depth.
    static constexpr std::size_t block_rows {72};      //!< The rows of one task, a whole number of tiles.

    //! \brief Nothing to turn on around a task.
    struct session
    {};

    //! \brief `C += Aₚ·Bₚ` for one tile of C (\p rows × \p cols) over a panel of depth \p kc.
    static void tile(std::size_t   kc,
                     const double* ap,
                     const double* bp,
                     double*       c,
                     std::size_t   ldc,
                     std::size_t   rows,
                     std::size_t   cols) noexcept
    {
      avx2_tile_f64(kc, ap, bp, c, ldc, rows, cols);
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
    if (path == gemm_path::amx) {
      gemm_packed<amx_tile_kernel>(m, n, k, a, lda, b, ldb, c, ldc);
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

  /*!
   * \brief `C += A·B`, every element of C summed in ascending order of the depth.
   *
   * Products of fewer than \ref gemm_packed_threshold multiply-adds, or shallower than \ref gemm_min_depth, run
   * \ref gemm_rows. The rest run packed (\ref gemm_packed): on the AMX for `double` where it is in use (each
   * element a chain of fused multiply-adds), else on the vector tile, which rounds `c + a·b` as the compiler
   * does: each product before its sum with contraction off (GCC's default in ISO C++ mode), fused where it
   * contracts (Clang's default). Either way the plain loop and the packed one agree to the bit, and every
   * element is summed in ascending order of the depth on any number of threads.
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
    if (!gemm_elsewhere(m, n, k, a, lda, b, ldb, c, ldc)) {
      gemm_packed<vector_tile_kernel<T>>(m, n, k, a, lda, b, ldb, c, ldc);
    }
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
