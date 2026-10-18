/*!
 * \file blas.hpp
 * \brief The column-major semantics of the BLAS and LAPACK routines NumericAL stands in for (`ddot`, `dgemm`, `dsyrk`, `dgemv`, `dtrsm`, `dgesv`,
 *        `dgetrf`, `dgetrs`, `dpotrf`, `dpotrs`, `dsyevd`, `dgesdd`, `dgeqrf`, `dorgqr`, `dgelsd`, `dgeev`), as their reference
 *        implementations define them, over NumericAL's kernels: what the optional `numerical_blas` library exports
 *        with the Fortran calling convention (blas/numerical_blas.cpp), so that a program linked against BLAS and
 *        LAPACK, or a switch such as FlexiBLAS, can run on NumericAL.
 *
 * Each function checks its arguments as the reference does and returns the position of the first illegal one (the
 * number the reference passes to `XERBLA`), negated for the LAPACK routines as their `INFO`; it computes nothing then.
 * A matrix holding a NaN or an infinity, which NumericAL's decompositions refuse, is an illegal value of that
 * argument.
 */
#ifndef NUMERICAL_BLAS_HPP
#define NUMERICAL_BLAS_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "decomposition.hpp"
#include "eigen.hpp"
#include "gemm.hpp"
#include "matrix.hpp"
#include "parallel.hpp"

namespace numerical::blas {

  //! \brief LAPACK's relative machine precision, `dlamch('E')`: the unit roundoff `2⁻⁵³`, half of
  //!        `std::numeric_limits<double>::epsilon()`. The default rank cutoff of `dgelsd` is relative to it.
  inline constexpr double lapack_precision {0x1p-53};

  //! \brief The columns of \ref syrk's smallest diagonal blocks, formed whole: their wasted half, at most
  //!        `64 n k` multiply-adds, stays small beside the `n² k / 2` of the triangle.
  inline constexpr std::size_t syrk_block {128};

  //! \brief The most elements of \ref syrk's operand copy a thread keeps between calls (32 MiB).
  inline constexpr std::size_t syrk_kept_limit {(std::size_t {32} << 20U) / sizeof(double)};

  namespace detail {

    //! \brief The storage of \ref syrk's operand copy this thread keeps between calls.
    [[nodiscard]] inline buffer<double>& syrk_operand() noexcept
    {
      thread_local buffer<double> storage;
      return storage;
    }

    //! \brief Whether \p op is one of BLAS's `N`, `T` or `C` (either case): no transpose, transpose, conjugate one.
    [[nodiscard]] constexpr bool is_operation(char op) noexcept
    {
      return op == 'N' || op == 'n' || op == 'T' || op == 't' || op == 'C' || op == 'c';
    }

    //! \brief Whether \p op transposes (`T` or `C`: a real matrix's conjugate transpose is its transpose).
    [[nodiscard]] constexpr bool transposes(char op) noexcept
    {
      return op != 'N' && op != 'n';
    }

    //! \brief Whether \p op is BLAS's `C`: the conjugate transpose (a real matrix's is its transpose).
    [[nodiscard]] constexpr bool conjugates(char op) noexcept
    {
      return op == 'C' || op == 'c';
    }

    //! \brief Whether the real \p x is finite (an overload per kind of scalar, as `conj_of`: each one expression).
    template <typename T>
    [[nodiscard]] bool finite(const T& x)
    {
      return std::isfinite(x);
    }

    //! \brief Whether both parts of the complex \p x are finite.
    template <typename U>
    [[nodiscard]] bool finite(const std::complex<U>& x)
    {
      return std::isfinite(x.real()) && std::isfinite(x.imag());
    }

    /*!
     * \brief Whether the \p count reals from \p x are all finite: an exponent of all ones is infinite or NaN. Each chunk
     *        is tested without a branch, the flags or-ed as integers, so that the loop runs on vectors (an exit at every
     *        element kept it scalar: 0.9 ms for a million).
     */
    [[nodiscard]] inline bool all_finite(const double* x,
                                         std::size_t   count)
    {
      constexpr std::uint64_t exponent {0x7FF0'0000'0000'0000ULL};
      constexpr std::size_t   chunk    {512};
      for (std::size_t i0 {0}; i0 < count; i0 += chunk) {
        std::uint64_t infinite {0};
        for (std::size_t i {i0}; i < std::min(i0 + chunk, count); ++i) {
          infinite |= static_cast<std::uint64_t>((std::bit_cast<std::uint64_t>(x[i]) & exponent) == exponent);
        }
        if (infinite != 0) {
          return false;
        }
      }
      return true;
    }

    //! \brief The tile side of the transposing copies between column-major arguments and row-major storage: a tile's
    //!        rows and columns both stay in the cache, where a whole column read across rows `lda` apart misses on
    //!        every element (a 1000 × 1000 copy took several milliseconds).
    inline constexpr std::size_t transpose_tile {32};

    //! \brief The weight of an element of the transposing copies against a multiply-add: bound by the memory, a copy
    //!        spreads over threads from a few hundred thousand elements (an inverse of order 1000 on six AVX2 cores
    //!        spent a fifth of its time in its four copies on one thread). The bands capture their bounds by value: by
    //!        reference, the stores might have changed them, and each was read again (order 100: 5.9 → 10 µs).
    inline constexpr std::size_t transpose_weight {4};

    //! \brief The column-major `rows × cols` \p a (\p lda apart) into the row-major \p out (resized to it), by tiles,
    //!        each element times \p scale (conjugated first where \p conjugate).
    template <typename T>
    void row_major_into(buffer<T>&     out,
                        const T*       a,
                        std::ptrdiff_t lda,
                        std::ptrdiff_t rows,
                        std::ptrdiff_t cols,
                        T              scale,
                        bool           conjugate = false)
    {
      const auto r  {static_cast<std::size_t>(rows)};
      const auto c  {static_cast<std::size_t>(cols)};
      const auto ld {static_cast<std::size_t>(lda)};
      out.resize(r * c);
      T* const to   {out.data()};
      numerical::detail::parallel_for((r + transpose_tile - 1) / transpose_tile, transpose_weight * transpose_tile * c, [a, ld, r, c, to, scale, conjugate](std::size_t band) {
                                        const std::size_t i0 {band * transpose_tile};
                                        for (std::size_t j0 {0}; j0 < c; j0 += transpose_tile) {
                                          for (std::size_t j {j0}; j < std::min(j0 + transpose_tile, c); ++j) {
                                            for (std::size_t i {i0}; i < std::min(i0 + transpose_tile, r); ++i) {
                                              const T x {a[i + (j * ld)]};
                                              to[(i * c) + j] = scale * (conjugate ? numerical::detail::conj_of(x) : x);
                                            }
                                          }
                                        }
                                      });
    }

    //! \brief `rows × cols` of the column-major \p a (\p lda apart), as a row-major buffer, each element times
    //!        \p scale (conjugated first where \p conjugate).
    template <typename T>
    [[nodiscard]] buffer<T> row_major(const T*       a,
                                      std::ptrdiff_t lda,
                                      std::ptrdiff_t rows,
                                      std::ptrdiff_t cols,
                                      T              scale,
                                      bool           conjugate = false)
    {
      buffer<T> out;
      row_major_into(out, a, lda, rows, cols, scale, conjugate);
      return out;
    }

    //! \brief The most bytes a thread keeps in each of \ref kept_storage's slots between calls (32 MiB).
    inline constexpr std::size_t kept_bytes {std::size_t {32} << 20U};

    /*!
     * \brief Row-major storage of the scalar \p T this thread keeps between calls, in \p slot (0 for a call's `A`, 1
     *        for its `B`): a fresh one of megabytes is mapped anew and faulted in page by page on every call (a solve
     *        of order 1000 spent 2.5 ms of its 30 on each).
     */
    template <typename T>
    [[nodiscard]] buffer<T>& kept_storage(std::size_t slot) noexcept
    {
      thread_local std::array<buffer<T>, 2> kept;
      return kept[slot];
    }

    //! \brief \ref kept_storage's \p slot, taken over by the caller (left empty there meanwhile).
    template <typename T>
    [[nodiscard]] buffer<T> take_kept(std::size_t slot)
    {
      return std::exchange(kept_storage<T>(slot), buffer<T> {});
    }

    //! \brief \p storage given back to \ref kept_storage's \p slot, unless past \ref kept_bytes.
    template <typename T>
    void give_back(std::size_t slot,
                   buffer<T>&& storage)
    {
      if (storage.capacity() <= kept_bytes / sizeof(T)) {
        kept_storage<T>(slot) = std::move(storage);
      }
    }

    //! \brief The row-major `rows × cols` \p from into the column-major \p a (\p lda apart), by tiles.
    template <typename T>
    void column_major_into(const T*    from,
                           std::size_t rows,
                           std::size_t cols,
                           T*          a,
                           std::size_t lda)
    {
      numerical::detail::parallel_for((cols + transpose_tile - 1) / transpose_tile, transpose_weight * transpose_tile * rows, [from, rows, cols, a, lda](std::size_t band) {
                                        const std::size_t j0 {band * transpose_tile};
                                        for (std::size_t i0 {0}; i0 < rows; i0 += transpose_tile) {
                                          for (std::size_t i {i0}; i < std::min(i0 + transpose_tile, rows); ++i) {
                                            for (std::size_t j {j0}; j < std::min(j0 + transpose_tile, cols); ++j) {
                                              a[i + (j * lda)] = from[(i * cols) + j];
                                            }
                                          }
                                        }
                                      });
    }

    /*!
     * \brief `C += L op(A)ᵀ` (`C` row-major `m × n`, `L` `m × k`), \p a the column-major `op(A)`'s storage (\p lda apart):
     *        `op(A)ᵀ` is A read in place where \p transa does not transpose, else a copy of its transpose (an overload
     *        per scalar, as `conj_of`).
     */
    template <typename T>
    void product_by_op_a(char        transa,
                         std::size_t m,
                         std::size_t n,
                         std::size_t k,
                         const T*    l,
                         std::size_t ldl,
                         const T*    a,
                         std::size_t lda,
                         T*          c,
                         std::size_t ldc)
    {
      const bool      copied {transposes(transa)};
      const buffer<T> right  {copied ? row_major(a, static_cast<std::ptrdiff_t>(lda), static_cast<std::ptrdiff_t>(k), static_cast<std::ptrdiff_t>(n), T {1}, conjugates(transa))
                                     : buffer<T> {}};
      numerical::detail::gemm(m, n, k, l, ldl, copied ? right.data() : a, copied ? n : lda, c, ldc);
    }

    //! \brief \ref product_by_op_a for `double`: a transposed A read in place too, as `op(A)ᵀ`'s transpose (\ref
    //!        numerical::detail::gemm_bt, the bits of the product by the copy; NumPy's `a @ b.T`).
    inline void product_by_op_a(char          transa,
                                std::size_t   m,
                                std::size_t   n,
                                std::size_t   k,
                                const double* l,
                                std::size_t   ldl,
                                const double* a,
                                std::size_t   lda,
                                double*       c,
                                std::size_t   ldc)
    {
      if (transposes(transa)) {
        numerical::detail::gemm_bt(m, n, k, l, ldl, a, lda, c, ldc);
        return;
      }
      numerical::detail::gemm(m, n, k, l, ldl, a, lda, c, ldc);
    }

    //! \brief The lower triangle of the column-major `n × n` \p a (\p lda apart) into that of the row-major \p rows
    //!        (`n` apart) where \p in, out of it back into \p a otherwise, by tiles; the other triangles untouched.
    template <typename T>
    void lower_triangle_rows(T*          a,
                             std::size_t lda,
                             std::size_t n,
                             T*          rows,
                             bool        in)
    {
      for (std::size_t j0 {0}; j0 < n; j0 += transpose_tile) {
        for (std::size_t i0 {j0}; i0 < n; i0 += transpose_tile) {
          for (std::size_t j {j0}; j < std::min(j0 + transpose_tile, n); ++j) {
            for (std::size_t i {std::max(i0, j)}; i < std::min(i0 + transpose_tile, n); ++i) {
              if (in) {
                rows[(i * n) + j] = a[i + (j * lda)];
              }
              else {
                a[i + (j * lda)] = rows[(i * n) + j];
              }
            }
          }
        }
      }
    }

    //! \brief The \p cols columns of \p rows elements of the column-major \p a (\p lda apart) as the rows of a
    //!        row-major buffer, each element times \p scale: `aᵀ`, row-major.
    template <typename T>
    [[nodiscard]] buffer<T> columns_as_rows(const T*       a,
                                            std::ptrdiff_t lda,
                                            std::ptrdiff_t rows,
                                            std::ptrdiff_t cols,
                                            T              scale)
    {
      const auto r {static_cast<std::size_t>(rows)};
      const auto c {static_cast<std::size_t>(cols)};
      buffer<T>  out(r * c);
      for (std::size_t j {0}; j < c; ++j) {
        for (std::size_t i {0}; i < r; ++i) {
          out[(j * r) + i] = scale * a[i + (j * static_cast<std::size_t>(lda))];
        }
      }
      return out;
    }

    /*!
     * \brief `C += α op(B)ᵀ op(A)ᵀ`, row-major `C` (`m × n`, \p ldc apart), the column-major `C` of BLAS's `gemm` as
     *        its transpose. A column-major matrix is row-major as its transpose, so `op(B)ᵀ` is B read in place when B
     *        is not transposed (unless α scales a copy), and `op(A)ᵀ` is A read in place when A is not; a transposed one
     *        is copied (an overload per scalar, as `conj_of`).
     */
    template <typename T>
    void product_of_ops(char        transa,
                        char        transb,
                        T           alpha,
                        std::size_t m,
                        std::size_t n,
                        std::size_t k,
                        const T*    a,
                        std::size_t lda,
                        const T*    b,
                        std::size_t ldb,
                        T*          c,
                        std::size_t ldc)
    {
      const bool copy_b {transposes(transb) || alpha != T {1}};
      buffer<T>  left   {};
      if (transposes(transb)) {
        left = row_major(b, static_cast<std::ptrdiff_t>(ldb), static_cast<std::ptrdiff_t>(m), static_cast<std::ptrdiff_t>(k), alpha,
                         conjugates(transb));
      }
      else if (copy_b) {
        left = columns_as_rows(b, static_cast<std::ptrdiff_t>(ldb), static_cast<std::ptrdiff_t>(k), static_cast<std::ptrdiff_t>(m), alpha);
      }
      product_by_op_a(transa, m, n, k, copy_b ? left.data() : b, copy_b ? k : ldb, a, lda, c, ldc);
    }

    //! \brief \ref product_of_ops for `double`: with α = 1 a transposed B is read in place too, as `op(B)ᵀ`'s
    //!        transpose (\ref numerical::detail::gemm_at, or \ref numerical::detail::gemm_atbt with a transposed A:
    //!        the bits of the products by the copies; NumPy's `a.T @ b`).
    inline void product_of_ops(char          transa,
                               char          transb,
                               double        alpha,
                               std::size_t   m,
                               std::size_t   n,
                               std::size_t   k,
                               const double* a,
                               std::size_t   lda,
                               const double* b,
                               std::size_t   ldb,
                               double*       c,
                               std::size_t   ldc)
    {
      if (!transposes(transb) || alpha != 1.0) {
        product_of_ops<double>(transa, transb, alpha, m, n, k, a, lda, b, ldb, c, ldc);
        return;
      }
      if (transposes(transa)) {
        numerical::detail::gemm_atbt(m, n, k, b, ldb, a, lda, c, ldc);
        return;
      }
      numerical::detail::gemm_at(m, n, k, b, ldb, a, lda, c, ldc);
    }
  } // namespace detail

  namespace detail {

    //! \brief The permutation \p pivots (row `i` of `P A` is row `pivots[i]` of `A`) as LAPACK's successive row
    //!        interchanges, 1-based, in \p ipiv: at step `i`, row `i` was interchanged with row `ipiv[i]`.
    template <typename Index>
    void interchanges(const std::vector<std::size_t>& pivots,
                      Index*                          ipiv)
    {
      const std::size_t        size {pivots.size()};
      std::vector<std::size_t> at(size);    // the original row at each position
      std::vector<std::size_t> where(size); // the position of each original row
      for (std::size_t i {0}; i < size; ++i) {
        at[i]    = i;
        where[i] = i;
      }
      for (std::size_t i {0}; i < size; ++i) {
        const std::size_t from {where[pivots[i]]};
        ipiv[i]          = static_cast<Index>(from + 1);
        where[at[i]]     = from;
        where[pivots[i]] = i;
        std::swap(at[i], at[from]);
      }
    }

    /*!
     * \brief The LU factorisation with partial pivoting of a rectangular column-major `A` in place, as LAPACK's
     *        `dgetf2`, column by column: NumericAL's blocked LU takes square matrices. \return `INFO` as `dgetrf`'s.
     */
    template <typename T, typename Index>
    int rectangular_lu(std::size_t m,
                       std::size_t n,
                       T*          a,
                       std::size_t lda,
                       Index*      ipiv)
    {
      int info {0};
      for (std::size_t j {0}; j < std::min(m, n); ++j) {
        T* const    column {a + (j * lda)};
        std::size_t pivot  {j};
        for (std::size_t i {j + 1}; i < m; ++i) {
          if (std::abs(column[i]) > std::abs(column[pivot])) {
            pivot = i;
          }
        }
        ipiv[j] = static_cast<Index>(pivot + 1);
        if (column[pivot] == T {}) {
          info = info == 0 ? static_cast<int>(j + 1) : info;
          continue;
        }
        for (std::size_t c {0}; c < n; ++c) { // the whole rows interchanged, as dgetf2's dswap
          std::swap(a[j + (c * lda)], a[pivot + (c * lda)]);
        }
        for (std::size_t i {j + 1}; i < m; ++i) {
          column[i] = column[i] / column[j];
        }
        for (std::size_t c {j + 1}; c < n; ++c) {
          T* const target {a + (c * lda)};
          const T  factor {target[j]};
          for (std::size_t i {j + 1}; i < m; ++i) {
            target[i] -= column[i] * factor;
          }
        }
      }
      return info;
    }
  } // namespace detail

  /*!
   * \brief `C ← α op(A) op(B) + β C`, column-major, as BLAS's `dgemm`: `op(A)` is `m × k`, `op(B)` `k × n`, `C`
   *        `m × n`. With β zero, `C` is not read (a NaN there does not survive), as the reference's. Each term is
   *        `(α b) a`, α applied to B's element first, as the reference applies it; the sums run through NumericAL's
   *        product kernel, each element's terms in ascending order of the depth.
   * \return 0, or the position of the first illegal argument as `XERBLA` numbers it (then nothing is computed).
   */
  template <typename T>
  int gemm(char           transa,
           char           transb,
           std::ptrdiff_t m,
           std::ptrdiff_t n,
           std::ptrdiff_t k,
           T              alpha,
           const T*       a,
           std::ptrdiff_t lda,
           const T*       b,
           std::ptrdiff_t ldb,
           T              beta,
           T*             c,
           std::ptrdiff_t ldc)
  {
    const std::ptrdiff_t a_rows  {detail::transposes(transa) ? k : m};
    const std::ptrdiff_t b_rows  {detail::transposes(transb) ? n : k};
    int                  illegal {0};
    if (!detail::is_operation(transa)) {
      illegal = 1;
    }
    else if (!detail::is_operation(transb)) {
      illegal = 2;
    }
    else if (m < 0) {
      illegal = 3;
    }
    else if (n < 0) {
      illegal = 4;
    }
    else if (k < 0) {
      illegal = 5;
    }
    else if (lda < std::max<std::ptrdiff_t>(1, a_rows)) {
      illegal = 8;
    }
    else if (ldb < std::max<std::ptrdiff_t>(1, b_rows)) {
      illegal = 10;
    }
    else if (ldc < std::max<std::ptrdiff_t>(1, m)) {
      illegal = 13;
    }
    if (illegal != 0 || m == 0 || n == 0) {
      return illegal;
    }
    const auto rows   {static_cast<std::size_t>(m)};
    const auto cols   {static_cast<std::size_t>(n)};
    const auto stride {static_cast<std::size_t>(ldc)};
    for (std::size_t j {0}; j < cols; ++j) { // C ← β C, column by column
      T* const column {c + (j * stride)};
      if (beta == T {}) {
        std::fill_n(column, rows, T {});
      }
      else if (beta != T {1}) {
        std::transform(column, column + rows, column, [beta](T x) { return beta * x; });
      }
    }
    if (alpha == T {} || k == 0) {
      return 0;
    }
    detail::product_of_ops(transa, transb, alpha, cols, rows, static_cast<std::size_t>(k), a, static_cast<std::size_t>(lda), b,
                           static_cast<std::size_t>(ldb), c, stride);
    return 0;
  }

  namespace detail {

    //! \brief Index of element \p i of a BLAS vector of \p length elements \p inc apart: from the last one where
    //!        \p inc is negative, as the reference walks it.
    [[nodiscard]] constexpr std::size_t strided(std::size_t    i,
                                                std::size_t    length,
                                                std::ptrdiff_t inc) noexcept
    {
      return inc > 0 ? i * static_cast<std::size_t>(inc) : (length - 1 - i) * static_cast<std::size_t>(-inc);
    }
  } // namespace detail

  /*!
   * \brief `Σ x_i y_i` over \p n elements \p incx and \p incy apart (from the last where negative), as BLAS's `ddot`;
   *        0 where `n ≤ 0`. Contiguous vectors by \ref numerical::detail::chunked_dot (the vector units, the chunks in
   *        parallel, the same sum on any number of threads); others gathered first.
   */
  [[nodiscard]] inline double dot(std::ptrdiff_t n,
                                  const double*  x,
                                  std::ptrdiff_t incx,
                                  const double*  y,
                                  std::ptrdiff_t incy)
  {
    if (n <= 0) {
      return 0.0;
    }
    const auto count {static_cast<std::size_t>(n)};
    if (incx == 1 && incy == 1) {
      return numerical::detail::chunked_dot(x, y, count);
    }
    buffer<double> xs(count, 0.0);
    buffer<double> ys(count, 0.0);
    for (std::size_t i {0}; i < count; ++i) {
      xs[i] = x[detail::strided(i, count, incx)];
      ys[i] = y[detail::strided(i, count, incy)];
    }
    return numerical::detail::chunked_dot(xs.data(), ys.data(), count);
  }

  //! \brief The rows (or, for `Aᵀ x`, the columns) each task of \ref gemv takes, 64 so that an order of a few
  //!        hundred shares out (256 left order 300 two tasks).
  inline constexpr std::size_t gemv_rows {64};

  //! \brief The weight of \ref gemv's multiply-adds against a compute-bound product's: each reads an element of `A`
  //!        once, so the product is bound by the memory, and spreads over more threads' caches for its work (six
  //!        x86-64 AVX2 cores, `Aᵀ x`: order 300 17.7 → 7.5 µs, order 1000 74 → 60 µs).
  inline constexpr std::size_t gemv_weight {4};

  /*!
   * \brief `y ← α op(A) x + β y`, column-major `A` `m × n`, as BLAS's `dgemv`: the vectors \p incx and \p incy
   *        apart (from the last element where negative), `y` not read where `β = 0`. `Aᵀ x` is each column's dot
   *        product with `x` (\ref numerical::detail::row_dots, `A`'s columns the rows of `Aᵀ`); `A x` adds `α x_j`
   *        times each column, by groups of \ref gemv_rows rows in parallel.
   * \return 0, or the position of the first illegal argument, as the reference passes it to `XERBLA`.
   */
  inline int gemv(char           trans,
                  std::ptrdiff_t m,
                  std::ptrdiff_t n,
                  double         alpha,
                  const double*  a,
                  std::ptrdiff_t lda,
                  const double*  x,
                  std::ptrdiff_t incx,
                  double         beta,
                  double*        y,
                  std::ptrdiff_t incy)
  {
    if (!detail::is_operation(trans)) {
      return 1;
    }
    if (m < 0) {
      return 2;
    }
    if (n < 0) {
      return 3;
    }
    if (lda < std::max<std::ptrdiff_t>(1, m)) {
      return 6;
    }
    if (incx == 0) {
      return 8;
    }
    if (incy == 0) {
      return 11;
    }
    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0)) {
      return 0;
    }
    const bool        transposed {detail::transposes(trans)};
    const auto        rows       {static_cast<std::size_t>(m)};
    const auto        cols       {static_cast<std::size_t>(n)};
    const auto        ld         {static_cast<std::size_t>(lda)};
    const std::size_t x_length   {transposed ? rows : cols};
    const std::size_t y_length   {transposed ? cols : rows};
    buffer<double>    out(y_length, 0.0);
    for (std::size_t i {0}; i < y_length && beta != 0.0; ++i) {
      out[i] = beta * y[detail::strided(i, y_length, incy)];
    }
    if (alpha != 0.0) {
      buffer<double> scaled(x_length, 0.0); // α x, gathered
      for (std::size_t i {0}; i < x_length; ++i) {
        scaled[i] = alpha * x[detail::strided(i, x_length, incx)];
      }
      if (transposed) {
        buffer<double> dots(cols, 0.0);
        numerical::detail::parallel_for((cols + gemv_rows - 1) / gemv_rows, gemv_weight * gemv_rows * rows, [&](std::size_t group) {
                                          const std::size_t first {group * gemv_rows};
                                          numerical::detail::row_dots(a + (first * ld), ld, std::min(gemv_rows, cols - first),
                                                                      scaled.data(), rows, dots.data() + first);
                                        });
        for (std::size_t j {0}; j < cols; ++j) {
          out[j] += dots[j];
        }
      }
      else { // A x: each task's rows of y held through every column (numerical::detail::subtract_products, the
             // factors −α x_j), each element fma(α x_j, a_ij, y_i) for j ascending, as column by column
        for (double& factor : scaled) {
          factor = -factor;
        }
        numerical::detail::parallel_for((rows + gemv_rows - 1) / gemv_rows, gemv_weight * gemv_rows * cols, [&](std::size_t group) {
                                          const std::size_t first {group * gemv_rows};
                                          numerical::detail::subtract_products(out.data() + first, scaled.data(), a + first, ld, cols,
                                                                               std::min(gemv_rows, rows - first));
                                        });
      }
    }
    for (std::size_t i {0}; i < y_length; ++i) {
      y[detail::strided(i, y_length, incy)] = out[i];
    }
    return 0;
  }

  namespace detail {

    //! \brief The operands of \ref syrk_triangle, which are row-major `P = op(A)` and `Q = op(A)ᵀ` (one of them
    //!        scaled by α), the column-major `C` and which triangle of it is written.
    struct syrk_operands
    {
      const double* p;       //!< `P`, `n × depth`.
      std::size_t   ldp;     //!< `P`'s row distance.
      const double* q;       //!< `Q`, `depth × n`.
      std::size_t   ldq;     //!< `Q`'s row distance.
      std::size_t   depth;   //!< `k`.
      double*       c;       //!< `C`, column-major.
      std::size_t   ldc;     //!< `C`'s column distance.
      bool          lower;   //!< The lower triangle (else the upper).
      bool          q_is_pt; //!< `Q` is `Pᵀ` read from `P`'s rows (\ref numerical::detail::gemm_bt), \p q unused.
      bool          p_is_qt; //!< `P` is `Qᵀ` read from `Q`'s columns (\ref numerical::detail::gemm_at), \p p unused.
    };

    //! \brief `P(c0…) Q(…, r0…)` added to the `cols × rows` row-major block at \p out (\p ldo apart): \ref
    //!        numerical::detail::gemm, or, where `Q` is `Pᵀ`, \ref numerical::detail::gemm_bt from `P`'s rows (the same
    //!        bits, no copy of `Q`).
    inline void syrk_product(const syrk_operands& operands,
                             std::size_t          c0,
                             std::size_t          cols,
                             std::size_t          r0,
                             std::size_t          rows,
                             double*              out,
                             std::size_t          ldo)
    {
      if (operands.q_is_pt) {
        numerical::detail::gemm_bt(cols, rows, operands.depth, operands.p + (c0 * operands.ldp), operands.ldp, operands.p + (r0 * operands.ldp),
                                   operands.ldp, out, ldo);
        return;
      }
      if (operands.p_is_qt) {
        numerical::detail::gemm_at(cols, rows, operands.depth, operands.q + c0, operands.ldq, operands.q + r0, operands.ldq, out, ldo);
        return;
      }
      numerical::detail::gemm(cols, rows, operands.depth, operands.p + (c0 * operands.ldp), operands.ldp, operands.q + r0, operands.ldq, out, ldo);
    }

    /*!
     * \brief Adds `P Q` to the triangle of the diagonal block `[j0, j0 + w)` of `C`: halved until it is a block of
     *        \ref syrk_block columns or fewer, formed whole aside; above that, the off-diagonal half is one product of
     *        `(w / 2)²` entries, which the matrix product kernel runs on every core. A block of `C` (rows `r0…`,
     *        columns `c0…`), column-major, is row-major `Cᵀ`: it adds `P(c0…) Q(…, r0…)`.
     */
    inline void syrk_triangle(syrk_operands operands,
                              std::size_t   j0,
                              std::size_t   w)
    {
      if (w <= syrk_block) {
        buffer<double> diagonal(w * w, 0.0); // row-major: (i, j) is C(j0 + j, j0 + i)
        syrk_product(operands, j0, w, j0, w, diagonal.data(), w);
        for (std::size_t j {0}; j < w; ++j) {
          for (std::size_t i {operands.lower ? j : 0}; i < (operands.lower ? w : j + 1); ++i) {
            operands.c[(j0 + i) + ((j0 + j) * operands.ldc)] += diagonal[(j * w) + i];
          }
        }
        return;
      }
      const std::size_t h    {w / 2};
      syrk_triangle(operands, j0, h);
      const std::size_t r0   {operands.lower ? j0 + h : j0}; // below the first half (L), or right of it (U)
      const std::size_t c0   {operands.lower ? j0 : j0 + h};
      const std::size_t rows {operands.lower ? w - h : h};
      const std::size_t cols {operands.lower ? h : w - h};
      syrk_product(operands, c0, cols, r0, rows, operands.c + r0 + (c0 * operands.ldc), operands.ldc);
      syrk_triangle(operands, j0 + h, w - h);
    }

    /*!
     * \brief The whole triangle of `C` (order \p n) from \p operands: a product the matrix product kernel packs runs in
     *        one pass over the triangle's tiles (\ref numerical::detail::gemm_triangle; each operand read where it lies,
     *        as \ref syrk_product reads it), a smaller one by halves (\ref syrk_triangle). Column-major `C` is row-major
     *        `Cᵀ = P Q`, so its upper triangle is the product's lower one.
     */
    inline void syrk_all(const syrk_operands& operands,
                         std::size_t          n)
    {
      const std::size_t k {operands.depth};
      if (k < numerical::detail::gemm_min_depth || n * n < ((numerical::detail::gemm_packed_threshold + k) - 1) / k) {
        syrk_triangle(operands, 0, n);
        return;
      }
      const bool m_lower {!operands.lower};
      if (operands.q_is_pt) {
        numerical::detail::gemm_triangle<numerical::detail::a_rows, numerical::detail::b_transposed>(
          n, k, operands.p, operands.ldp, operands.p, operands.ldp, operands.c, operands.ldc, m_lower);
        return;
      }
      if (operands.p_is_qt) {
        numerical::detail::gemm_triangle<numerical::detail::a_transposed, numerical::detail::b_rows>(
          n, k, operands.q, operands.ldq, operands.q, operands.ldq, operands.c, operands.ldc, m_lower);
        return;
      }
      numerical::detail::gemm_triangle<numerical::detail::a_rows, numerical::detail::b_rows>(
        n, k, operands.p, operands.ldp, operands.q, operands.ldq, operands.c, operands.ldc, m_lower);
    }
  } // namespace detail

  /*!
   * \brief `C ← α op(A) op(A)ᵀ + β C` on the triangle \p uplo (`U` or `L`) of the column-major `n × n` `C`, `op(A)`
   *        `n × k`, as BLAS's `dsyrk` (NumPy's `a @ a.T`): the other triangle neither read nor written, `C` not read
   *        where `β = 0`. By blocks of \ref syrk_block columns: a diagonal block formed whole aside and its triangle
   *        kept, the rows past it (`L`) or before it (`U`) by \ref gemm.
   * \return 0, or the position of the first illegal argument, as the reference passes it to `XERBLA`.
   */
  inline int syrk(char           uplo,
                  char           trans,
                  std::ptrdiff_t n,
                  std::ptrdiff_t k,
                  double         alpha,
                  const double*  a,
                  std::ptrdiff_t lda,
                  double         beta,
                  double*        c,
                  std::ptrdiff_t ldc)
  {
    const bool lower      {uplo == 'L' || uplo == 'l'};
    const bool transposed {detail::transposes(trans)};
    if (!lower && uplo != 'U' && uplo != 'u') {
      return 1;
    }
    if (!detail::is_operation(trans)) {
      return 2;
    }
    if (n < 0) {
      return 3;
    }
    if (k < 0) {
      return 4;
    }
    if (lda < std::max<std::ptrdiff_t>(1, transposed ? k : n)) {
      return 7;
    }
    if (ldc < std::max<std::ptrdiff_t>(1, n)) {
      return 10;
    }
    if (n == 0) {
      return 0;
    }
    const auto order {static_cast<std::size_t>(n)};
    const auto ld    {static_cast<std::size_t>(ldc)};
    if (alpha == 0.0 || k == 0) { // C ← β C on the triangle
      for (std::size_t j {0}; j < order && beta != 1.0; ++j) {
        for (std::size_t i {lower ? j : 0}; i < (lower ? order : j + 1); ++i) {
          c[i + (j * ld)] = beta == 0.0 ? 0.0 : beta * c[i + (j * ld)];
        }
      }
      return 0;
    }
    // C(r, c) = α Σ_l op(A)(r, l) op(A)(c, l). The kernel takes row-major operands: P = op(A) (n × k) and
    // Q = op(A)ᵀ (k × n). A column-major A is one of them read in place (Q for N, P for T). With α = 1 the other is
    // read from it as its transpose (detail::syrk_product; NumPy's a @ a.T is T): no copy, the same bits. Otherwise
    // the other is copied once, α folded into it.
    const auto depth {static_cast<std::size_t>(k)};
    const auto lda_s {static_cast<std::size_t>(lda)};
    for (std::size_t j {0}; j < order; ++j) { // C ← β C on the triangle; the products then add into it
      for (std::size_t i {lower ? j : 0}; i < (lower ? order : j + 1); ++i) {
        double& entry {c[i + (j * ld)]};
        entry = beta == 0.0 ? 0.0 : beta * entry;
      }
    }
    if (alpha == 1.0) {
      detail::syrk_all({transposed ? a : nullptr, lda_s, transposed ? nullptr : a, lda_s, depth, c, ld, lower, transposed, !transposed}, order);
      return 0;
    }
    // The copy in storage the thread keeps between calls (as the kernel's packing_storage): a fresh one of a megabyte
    // is mapped anew and faulted in page by page on every call.
    buffer<double>    copy      {std::exchange(detail::syrk_operand(), buffer<double> {})};
    const std::size_t copy_rows {transposed ? depth : order}; // row-major: α A as it lies, k × n (T) or n × k (N)
    const std::size_t copy_cols {transposed ? order : depth};
    if (copy.size() < copy_rows * copy_cols) {
      copy = buffer<double>(copy_rows * copy_cols);
    }
    for (std::size_t i {0}; i < copy_rows; ++i) {
      for (std::size_t j {0}; j < copy_cols; ++j) {
        copy[(i * copy_cols) + j] = alpha * a[i + (j * lda_s)];
      }
    }
    const double* const p_rows {transposed ? a : copy.data()};
    const std::size_t   ldp    {transposed ? lda_s : depth};
    const double* const q_cols {transposed ? copy.data() : a};
    const std::size_t   ldq    {transposed ? order : lda_s};
    detail::syrk_all({p_rows, ldp, q_cols, ldq, depth, c, ld, lower, false, false}, order);
    if (copy.size() <= syrk_kept_limit) {
      detail::syrk_operand() = std::move(copy);
    }
    return 0;
  }

  //! \brief The rows of \ref trsm's blocks: within one, each column solved alone; past it, the matrix product kernel.
  inline constexpr std::size_t trsm_block {64};

  namespace detail {

    /*!
     * \brief `A X = B` in place for the column-major triangular `k × k` \p a (\p lower or upper, a \p unit diagonal
     *        not read) and `k × n` \p b: by blocks of \ref trsm_block rows, from the top (lower) or the bottom
     *        (upper); a block's diagonal triangle column by column of `B`, the columns in parallel, then its terms out
     *        of the rows still to solve through \ref gemm.
     */
    inline void triangular_solve(bool          lower,
                                 bool          unit,
                                 std::size_t   k,
                                 std::size_t   n,
                                 const double* a,
                                 std::size_t   lda,
                                 double*       b,
                                 std::size_t   ldb)
    {
      for (std::size_t done {0}; done < k; done += trsm_block) {
        const std::size_t w     {std::min(trsm_block, k - done)};
        const std::size_t k0    {lower ? done : k - done - w};
        const std::size_t k1    {k0 + w};
        // The block's triangle, each column of B alone, the columns in parallel.
        numerical::detail::parallel_for(n, w * w, [a, b, lda, ldb, k0, k1, lower, unit](std::size_t j) {
                                          double* x {b + (j * ldb)};
                                          for (std::size_t s {0}; s < k1 - k0; ++s) {
                                            const std::size_t p {lower ? k0 + s : k1 - 1 - s};
                                            if (!unit) {
                                              x[p] /= a[p + (p * lda)];
                                            }
                                            const std::size_t from {lower ? p + 1 : k0};
                                            const std::size_t to {lower ? k1 : p};
                                            numerical::detail::add_product(x + from, -x[p], a + from + (p * lda), to - from);
                                          }
                                        });
        const std::size_t first {lower ? k1 : 0};
        const std::size_t count {lower ? k - k1 : k0};
        if (count > 0) { // B(rows still to solve) −= A(those rows, block) X(block)
          (void)gemm('N', 'N', static_cast<std::ptrdiff_t>(count), static_cast<std::ptrdiff_t>(n), static_cast<std::ptrdiff_t>(w), -1.0,
                     a + first + (k0 * lda), static_cast<std::ptrdiff_t>(lda), b + k0, static_cast<std::ptrdiff_t>(ldb), 1.0, b + first,
                     static_cast<std::ptrdiff_t>(ldb));
        }
      }
    }
  } // namespace detail

  /*!
   * \brief `B ← α op(A)⁻¹ B` (\p side `L`) or `α B op(A)⁻¹` (`R`), column-major, `B` `m × n` and the triangular `A`
   *        (\p uplo `U` or `L`, its other triangle not read; a \p diag `U`nit diagonal not read either) `m × m` or
   *        `n × n`, as BLAS's `dtrsm`; `A` not read where `α = 0`. Every case is `A' X = B'` with `A'` triangular:
   *        `A` or a transposed copy of it, `B` or a transposed copy of it (`X op(A) = B` is `op(A)ᵀ Xᵀ = Bᵀ`), solved
   *        by \ref detail::triangular_solve. A zero on the diagonal gives infinities, as the reference's division does.
   * \return 0, or the position of the first illegal argument, as the reference passes it to `XERBLA`.
   */
  inline int trsm(char           side,
                  char           uplo,
                  char           transa,
                  char           diag,
                  std::ptrdiff_t m,
                  std::ptrdiff_t n,
                  double         alpha,
                  const double*  a,
                  std::ptrdiff_t lda,
                  double*        b,
                  std::ptrdiff_t ldb)
  {
    const bool left  {side == 'L' || side == 'l'};
    const bool lower {uplo == 'L' || uplo == 'l'};
    const bool unit  {diag == 'U' || diag == 'u'};
    if (!left && side != 'R' && side != 'r') {
      return 1;
    }
    if (!lower && uplo != 'U' && uplo != 'u') {
      return 2;
    }
    if (!detail::is_operation(transa)) {
      return 3;
    }
    if (!unit && diag != 'N' && diag != 'n') {
      return 4;
    }
    if (m < 0) {
      return 5;
    }
    if (n < 0) {
      return 6;
    }
    if (lda < std::max<std::ptrdiff_t>(1, left ? m : n)) {
      return 9;
    }
    if (ldb < std::max<std::ptrdiff_t>(1, m)) {
      return 11;
    }
    if (m == 0 || n == 0) {
      return 0;
    }
    const auto rows {static_cast<std::size_t>(m)};
    const auto cols {static_cast<std::size_t>(n)};
    const auto ld   {static_cast<std::size_t>(ldb)};
    for (std::size_t j {0}; j < cols; ++j) {
      for (std::size_t i {0}; i < rows; ++i) {
        double& entry {b[i + (j * ld)]};
        entry = alpha == 0.0 ? 0.0 : alpha * entry;
      }
    }
    if (alpha == 0.0) {
      return 0;
    }
    const std::size_t k            {left ? rows : cols};
    const bool        flip_a       {left == detail::transposes(transa)}; // A' = Aᵀ: op(A) = Aᵀ on the left, A on the right
    buffer<double>    transposed_a;
    const double*     a_used       {a};
    std::size_t       lda_used     {static_cast<std::size_t>(lda)};
    if (flip_a) {
      transposed_a = buffer<double>(k * k, 0.0);
      for (std::size_t j {0}; j < k; ++j) {
        for (std::size_t i {lower ? j : 0}; i < (lower ? k : j + 1); ++i) { // A's triangle, transposed
          transposed_a[j + (i * k)] = a[i + (j * static_cast<std::size_t>(lda))];
        }
      }
      a_used   = transposed_a.data();
      lda_used = k;
    }
    if (left) {
      detail::triangular_solve(lower != flip_a, unit, k, cols, a_used, lda_used, b, ld);
      return 0;
    }
    buffer<double> bt(cols * rows, 0.0); // Bᵀ, column-major n × m
    for (std::size_t j {0}; j < cols; ++j) {
      for (std::size_t i {0}; i < rows; ++i) {
        bt[j + (i * cols)] = b[i + (j * ld)];
      }
    }
    detail::triangular_solve(lower != flip_a, unit, k, rows, a_used, lda_used, bt.data(), cols);
    for (std::size_t j {0}; j < cols; ++j) {
      for (std::size_t i {0}; i < rows; ++i) {
        b[i + (j * ld)] = bt[j + (i * cols)];
      }
    }
    return 0;
  }

  /*!
   * \brief `A X = B`, column-major, as LAPACK's `dgesv`: `A` (`n × n`, \p lda apart) is factorised as `P A = L U` by
   *        NumericAL's blocked LU with partial pivoting, the factors left in \p a as `dgetrf` leaves them (`U` on and
   *        above the diagonal, `L` below, its unit diagonal not stored) and the row interchanges in \p ipiv (1-based:
   *        row `i` was interchanged with row `ipiv[i]`, in order); then `B` (`n × nrhs`, \p ldb apart) is overwritten
   *        by `X`. A zero pivot leaves `B` as it was.
   * \return `INFO`: 0; `i > 0` where `U(i, i)` is exactly zero; `-i` where argument `i` is illegal (`-3`: `A` holds a
   *         NaN or an infinity, which NumericAL's LU refuses), nothing then computed.
   */
  template <typename T, typename Index>
  int gesv(std::ptrdiff_t n,
           std::ptrdiff_t nrhs,
           T*             a,
           std::ptrdiff_t lda,
           Index*         ipiv,
           T*             b,
           std::ptrdiff_t ldb)
  {
    if (n < 0) {
      return -1;
    }
    if (nrhs < 0) {
      return -2;
    }
    if (lda < std::max<std::ptrdiff_t>(1, n)) {
      return -4;
    }
    if (ldb < std::max<std::ptrdiff_t>(1, n)) {
      return -7;
    }
    if (n == 0) {
      return 0;
    }
    const auto               size {static_cast<std::size_t>(n)};
    const auto               ld   {static_cast<std::size_t>(lda)};
    std::vector<std::size_t> pivots;
    matrix<T>                factors;
    bool                     singular {false};
    try { // B and A each transposed in and out once, in storage the thread keeps: the LU solves in B's copy and gives
          // its factors up
      buffer<T> a_rows       {detail::take_kept<T>(0)};
      detail::row_major_into(a_rows, a, lda, n, n, T {1});
      lu_decomposition<T> lu {matrix<T> {size, size, std::move(a_rows)}};
      pivots   = lu.pivots();
      singular = lu.singular();
      if (!singular && nrhs > 0) {
        buffer<T> b_rows {detail::take_kept<T>(1)};
        detail::row_major_into(b_rows, b, ldb, n, nrhs, T {1});
        matrix<T> x      {lu.solve(matrix<T> {size, static_cast<std::size_t>(nrhs), std::move(b_rows)})};
        detail::column_major_into(&*x.begin(), size, static_cast<std::size_t>(nrhs), b, static_cast<std::size_t>(ldb));
        detail::give_back(1, std::move(x).elements());
      }
      factors = std::move(lu).factors();
    }
    catch (const non_finite_error&) {
      return -3;
    }
    detail::column_major_into(&*factors.begin(), size, size, a, ld);
    detail::give_back(0, std::move(factors).elements());
    detail::interchanges(pivots, ipiv);
    for (std::size_t i {0}; i < size; ++i) {
      if (a[i + (i * ld)] == T {}) { // U(i, i), as just written
        return static_cast<int>(i + 1);
      }
    }
    return 0;
  }

  /*!
   * \brief The Cholesky factor of a symmetric positive-definite `A`, column-major, as LAPACK's `dpotrf`: with \p uplo
   *        `L`, `A = L Lᵀ` read from and written to `A`'s lower triangle; with `U`, `A = Uᵀ U` and the upper one. The
   *        other triangle is not referenced. Where `A` is not positive definite, `A` is left as it was (LAPACK leaves
   *        the factorisation it reached).
   * \return `INFO`: 0; `i > 0` where the leading minor of order `i` is not positive definite; `-i` where argument `i`
   *         is illegal (`-3`: the triangle read holds a NaN or an infinity).
   */
  template <typename T>
  int potrf(char           uplo,
            std::ptrdiff_t n,
            T*             a,
            std::ptrdiff_t lda)
  {
    const bool lower {uplo == 'L' || uplo == 'l'};
    if (!lower && uplo != 'U' && uplo != 'u') {
      return -1;
    }
    if (n < 0) {
      return -2;
    }
    if (lda < std::max<std::ptrdiff_t>(1, n)) {
      return -4;
    }
    if (n == 0) {
      return 0;
    }
    const auto size {static_cast<std::size_t>(n)};
    const auto ld   {static_cast<std::size_t>(lda)};
    // A's lower triangle, row-major (the upper one's conjugate, Hermitian), in storage the thread keeps: the decomposition
    // reads no more of it and zeroes the rest. L's triangle by tiles (a row of it read across columns lda apart missed on
    // every element), U's columns as they lie; the factor written back the same ways.
    buffer<T> lower_rows {detail::take_kept<T>(0)};
    lower_rows.resize(size * size);
    if (lower) {
      detail::lower_triangle_rows(a, ld, size, lower_rows.data(), true);
    }
    else {
      for (std::size_t i {0}; i < size; ++i) {
        for (std::size_t j {0}; j <= i; ++j) {
          lower_rows[(i * size) + j] = numerical::detail::conj_of(a[j + (i * ld)]);
        }
      }
    }
    try {
      buffer<T> l {cholesky_decomposition<T> {matrix<T> {size, size, std::move(lower_rows)}}.lower().elements()};
      if (lower) {
        detail::lower_triangle_rows(a, ld, size, l.data(), false);
      }
      else { // U = Lᴴ
        for (std::size_t i {0}; i < size; ++i) {
          for (std::size_t j {0}; j <= i; ++j) {
            a[j + (i * ld)] = numerical::detail::conj_of(l[(i * size) + j]);
          }
        }
      }
      detail::give_back(0, std::move(l));
    }
    catch (const not_positive_definite_error& failed) {
      return static_cast<int>(failed.order());
    }
    catch (const non_finite_error&) {
      return -3;
    }
    return 0;
  }

  /*!
   * \brief `P A = L U` of the column-major `m × n` `A` in place, as LAPACK's `dgetrf`: the factors as `dgetrf` leaves
   *        them and the row interchanges in \p ipiv (1-based, `min(m, n)` of them). A square `A` by NumericAL's
   *        blocked LU, a rectangular one column by column.
   * \return `INFO`: 0; `i > 0` where `U(i, i)` is exactly zero; `-i` where argument `i` is illegal (`-3`: a NaN or an
   *         infinity in `A`).
   */
  template <typename T, typename Index>
  int getrf(std::ptrdiff_t m,
            std::ptrdiff_t n,
            T*             a,
            std::ptrdiff_t lda,
            Index*         ipiv)
  {
    if (m < 0) {
      return -1;
    }
    if (n < 0) {
      return -2;
    }
    if (lda < std::max<std::ptrdiff_t>(1, m)) {
      return -4;
    }
    if (m == 0 || n == 0) {
      return 0;
    }
    const auto rows {static_cast<std::size_t>(m)};
    const auto cols {static_cast<std::size_t>(n)};
    const auto ld   {static_cast<std::size_t>(lda)};
    for (std::size_t j {0}; j < cols; ++j) {
      for (std::size_t i {0}; i < rows; ++i) {
        if (!detail::finite(a[i + (j * ld)])) {
          return -3;
        }
      }
    }
    if (rows != cols) {
      return detail::rectangular_lu(rows, cols, a, ld, ipiv);
    }
    buffer<T> a_rows       {detail::take_kept<T>(0)}; // transposed in and out once, in storage the thread keeps
    detail::row_major_into(a_rows, a, lda, m, n, T {1});
    lu_decomposition<T> lu {matrix<T> {rows, cols, std::move(a_rows)}};
    detail::interchanges(lu.pivots(), ipiv);
    matrix<T> factors      {std::move(lu).factors()};
    detail::column_major_into(&*factors.begin(), rows, cols, a, ld);
    detail::give_back(0, std::move(factors).elements());
    for (std::size_t i {0}; i < rows; ++i) {
      if (a[i + (i * ld)] == T {}) { // U(i, i), as just written
        return static_cast<int>(i + 1);
      }
    }
    return 0;
  }

  /*!
   * \brief `A X = B`, `Aᵀ X = B` or `Aᴴ X = B` (\p trans `N`, `T` or `C`), column-major, from the factors and interchanges of a
   *        `dgetrf` (by any implementation), as LAPACK's `dgetrs`: `B` (`n × nrhs`) overwritten by `X`, each column by
   *        forward then backward substitution.
   * \return `INFO`: 0, or `-i` where argument `i` is illegal.
   */
  template <typename T, typename Index>
  int getrs(char           trans,
            std::ptrdiff_t n,
            std::ptrdiff_t nrhs,
            const T*       a,
            std::ptrdiff_t lda,
            const Index*   ipiv,
            T*             b,
            std::ptrdiff_t ldb)
  {
    if (!detail::is_operation(trans)) {
      return -1;
    }
    if (n < 0) {
      return -2;
    }
    if (nrhs < 0) {
      return -3;
    }
    if (lda < std::max<std::ptrdiff_t>(1, n)) {
      return -5;
    }
    if (ldb < std::max<std::ptrdiff_t>(1, n)) {
      return -8;
    }
    const auto size {static_cast<std::size_t>(n)};
    const auto ld   {static_cast<std::size_t>(lda)};
    const bool conjugate {detail::conjugates(trans)};
    const auto at        = [&](std::size_t i, std::size_t j) { return a[i + (j * ld)]; };
    // An element of op(A)'s factor read transposed: conjugated for Aᴴ.
    const auto across    = [&](std::size_t i, std::size_t j) { return conjugate ? numerical::detail::conj_of(at(i, j)) : at(i, j); };
    for (std::size_t c {0}; c < static_cast<std::size_t>(nrhs); ++c) {
      T* const x {b + (c * static_cast<std::size_t>(ldb))};
      if (!detail::transposes(trans)) { // L U x = P b
        for (std::size_t i {0}; i < size; ++i) {
          std::swap(x[i], x[static_cast<std::size_t>(ipiv[i]) - 1]);
        }
        for (std::size_t j {0}; j < size; ++j) { // L, unit lower, by columns
          for (std::size_t i {j + 1}; i < size; ++i) {
            x[i] -= at(i, j) * x[j];
          }
        }
        for (std::size_t j {size}; j-- > 0;) { // U, by columns
          x[j] /= at(j, j);
          for (std::size_t i {0}; i < j; ++i) {
            x[i] -= at(i, j) * x[j];
          }
        }
        continue;
      }
      for (std::size_t i {0}; i < size; ++i) { // Uᵀ (Uᴴ), by rows of U's columns
        T sum {x[i]};
        for (std::size_t k {0}; k < i; ++k) {
          sum -= across(k, i) * x[k];
        }
        x[i] = sum / across(i, i);
      }
      for (std::size_t i {size}; i-- > 0;) { // Lᵀ (Lᴴ)
        T sum {x[i]};
        for (std::size_t k {i + 1}; k < size; ++k) {
          sum -= across(k, i) * x[k];
        }
        x[i] = sum;
      }
      for (std::size_t i {size}; i-- > 0;) { // Pᵀ: the interchanges undone in reverse
        std::swap(x[i], x[static_cast<std::size_t>(ipiv[i]) - 1]);
      }
    }
    return 0;
  }

  /*!
   * \brief `A X = B` from the Cholesky factor of a `dpotrf` (by any implementation), column-major, as LAPACK's
   *        `dpotrs`: \p uplo `L` for `A = L Lᵀ` in the lower triangle, `U` for `A = Uᵀ U` in the upper one; `B`
   *        (`n × nrhs`) overwritten by `X`.
   * \return `INFO`: 0, or `-i` where argument `i` is illegal.
   */
  template <typename T>
  int potrs(char           uplo,
            std::ptrdiff_t n,
            std::ptrdiff_t nrhs,
            const T*       a,
            std::ptrdiff_t lda,
            T*             b,
            std::ptrdiff_t ldb)
  {
    const bool lower {uplo == 'L' || uplo == 'l'};
    if (!lower && uplo != 'U' && uplo != 'u') {
      return -1;
    }
    if (n < 0) {
      return -2;
    }
    if (nrhs < 0) {
      return -3;
    }
    if (lda < std::max<std::ptrdiff_t>(1, n)) {
      return -5;
    }
    if (ldb < std::max<std::ptrdiff_t>(1, n)) {
      return -7;
    }
    const auto size {static_cast<std::size_t>(n)};
    const auto ld   {static_cast<std::size_t>(lda)};
    // L(i, j), i ≥ j: U(j, i) conjugated where U = Lᴴ is stored.
    const auto l = [&](std::size_t i, std::size_t j) { return lower ? a[i + (j * ld)] : numerical::detail::conj_of(a[j + (i * ld)]); };
    for (std::size_t c {0}; c < static_cast<std::size_t>(nrhs); ++c) {
      T* const x {b + (c * static_cast<std::size_t>(ldb))};
      for (std::size_t i {0}; i < size; ++i) { // L y = b
        T sum {x[i]};
        for (std::size_t k {0}; k < i; ++k) {
          sum -= l(i, k) * x[k];
        }
        x[i] = sum / l(i, i);
      }
      for (std::size_t i {size}; i-- > 0;) { // Lᴴ x = y
        T sum {x[i]};
        for (std::size_t k {i + 1}; k < size; ++k) {
          sum -= numerical::detail::conj_of(l(k, i)) * x[k];
        }
        x[i] = sum / numerical::detail::conj_of(l(i, i));
      }
    }
    return 0;
  }

  /*!
   * \brief The eigenvalues, and with \p jobz `V` the eigenvectors, of a symmetric column-major `A`, as LAPACK's
   *        `dsyevd`: the triangle \p uplo (`L` or `U`) read; the eigenvalues ascending in \p w; the orthonormal
   *        eigenvectors overwriting `A`, column `j` that of `w[j]`; with `N`, `A` left as it was (LAPACK leaves it
   *        overwritten). NumericAL's divide and conquer computes them; the workspaces are not used, but their sizes
   *        are `dsyevd`'s: an `lwork` or `liwork` of -1 asks them (written to `work[0]` and `iwork[0]`), and a
   *        smaller one is illegal.
   * \return `INFO`: 0, or `-i` where argument `i` is illegal (`-4`: a NaN or an infinity in the triangle read).
   */
  template <typename Index>
  int syevd(char           jobz,
            char           uplo,
            std::ptrdiff_t n,
            double*        a,
            std::ptrdiff_t lda,
            double*        w,
            double*        work,
            std::ptrdiff_t lwork,
            Index*         iwork,
            std::ptrdiff_t liwork)
  {
    const bool vectors {jobz == 'V' || jobz == 'v'};
    const bool lower   {uplo == 'L' || uplo == 'l'};
    if (!vectors && jobz != 'N' && jobz != 'n') {
      return -1;
    }
    if (!lower && uplo != 'U' && uplo != 'u') {
      return -2;
    }
    if (n < 0) {
      return -3;
    }
    if (lda < std::max<std::ptrdiff_t>(1, n)) {
      return -5;
    }
    std::ptrdiff_t lwork_least  {1};
    std::ptrdiff_t liwork_least {1};
    if (n > 1) {
      lwork_least  = vectors ? 1 + (6 * n) + (2 * n * n) : (2 * n) + 1;
      liwork_least = vectors ? 3 + (5 * n) : 1;
    }
    if (lwork == -1 || liwork == -1) {
      work[0]  = static_cast<double>(lwork_least);
      iwork[0] = static_cast<Index>(liwork_least);
      return 0;
    }
    if (lwork < lwork_least) {
      return -8;
    }
    if (liwork < liwork_least) {
      return -10;
    }
    if (n == 0) {
      return 0;
    }
    const auto     size {static_cast<std::size_t>(n)};
    const auto     ld   {static_cast<std::size_t>(lda)};
    buffer<double> lower_rows(size * size, 0.0); // the triangle read, as the lower one, row-major
    for (std::size_t i {0}; i < size; ++i) {
      for (std::size_t j {0}; j <= i; ++j) {
        lower_rows[(i * size) + j] = lower ? a[i + (j * ld)] : a[j + (i * ld)];
      }
    }
    try {
      const matrix<double> symmetric {size, size, std::move(lower_rows)};
      if (!vectors) {
        const vector<double> values {symmetric_eigenvalues(symmetric)};
        std::copy_n(&values[0], size, w);
        return 0;
      }
      const symmetric_eigen eigen {symmetric};
      std::copy_n(&eigen.values()[0], size, w);
      for (std::size_t j {0}; j < size; ++j) {
        for (std::size_t i {0}; i < size; ++i) {
          a[i + (j * ld)] = eigen.vectors()(i, j);
        }
      }
    }
    catch (const non_finite_error&) {
      return -4;
    }
    return 0;
  }

  /*!
   * \brief The singular value decomposition `A = U Σ Vᵀ` of a column-major `m × n` `A`, as LAPACK's `dgesdd`: the
   *        singular values descending in \p s; with \p jobz `A`, all of `U` (`m × m`) and `Vᵀ` (`n × n`); with `S`,
   *        their first `min(m, n)` columns and rows; with `O`, those overwriting `A` (`U`'s where `m ≥ n`, else
   *        `Vᵀ`'s) and the other in \p u or \p vt; with `N`, the values alone (`A` then left as it was). NumericAL's
   *        divide and conquer computes them; the workspaces are not used, but their sizes are `dgesdd`'s: an `lwork`
   *        of -1 asks it (written to `work[0]`), and a smaller one is illegal.
   * \return `INFO`: 0, or `-i` where argument `i` is illegal (`-4`: a NaN or an infinity in `A`).
   */
  template <typename Index>
  int gesdd(char           jobz,
            std::ptrdiff_t m,
            std::ptrdiff_t n,
            double*        a,
            std::ptrdiff_t lda,
            double*        s,
            double*        u,
            std::ptrdiff_t ldu,
            double*        vt,
            std::ptrdiff_t ldvt,
            double*        work,
            std::ptrdiff_t lwork,
            Index*         iwork)
  {
    (void)iwork; // dgesdd's integer workspace, of 8 min(m, n): not used
    const bool all       {jobz == 'A' || jobz == 'a'};
    const bool some      {jobz == 'S' || jobz == 's'};
    const bool overwrite {jobz == 'O' || jobz == 'o'};
    const bool none      {jobz == 'N' || jobz == 'n'};
    if (!all && !some && !overwrite && !none) {
      return -1;
    }
    if (m < 0) {
      return -2;
    }
    if (n < 0) {
      return -3;
    }
    if (lda < std::max<std::ptrdiff_t>(1, m)) {
      return -5;
    }
    const std::ptrdiff_t least  {std::min(m, n)};
    const std::ptrdiff_t most   {std::max(m, n)};
    const bool           u_out  {all || some || (overwrite && m < n)};
    const bool           vt_out {all || some || (overwrite && m >= n)};
    if (ldu < std::max<std::ptrdiff_t>(1, u_out ? m : 1)) {
      return -8;
    }
    std::ptrdiff_t vt_rows {1};
    if (some) {
      vt_rows = least;
    }
    else if (vt_out) { // A, or O with m ≥ n: Vᵀ whole
      vt_rows = n;
    }
    if (ldvt < std::max<std::ptrdiff_t>(1, vt_rows)) {
      return -10;
    }
    std::ptrdiff_t lwork_least {(3 * least) + std::max(most, 7 * least)};
    if (overwrite) {
      lwork_least = (3 * least) + std::max(most, (5 * least * least) + (4 * least));
    }
    else if (all || some) {
      lwork_least = (3 * least) + std::max(most, (4 * least * least) + (4 * least));
    }
    if (lwork == -1) {
      work[0] = static_cast<double>(std::max<std::ptrdiff_t>(1, lwork_least));
      return 0;
    }
    if (lwork < std::max<std::ptrdiff_t>(1, lwork_least)) {
      return -12;
    }
    if (least == 0) {
      return 0;
    }
    const auto rows {static_cast<std::size_t>(m)};
    const auto cols {static_cast<std::size_t>(n)};
    const auto k    {static_cast<std::size_t>(least)};
    try {
      const matrix<double> source {rows, cols, detail::row_major(a, lda, m, n, 1.0)};
      if (none) {
        const vector<double> values {singular_values(source)};
        std::copy_n(&values[0], k, s);
        return 0;
      }
      const svd_decomposition<double> svd {source, svd_method::divide_and_conquer, all};
      std::copy_n(&svd.singular_values()[0], k, s);
      const matrix<double> left           {svd.u()};
      const matrix<double> right          {svd.vh()};
      // With O, the factor that fits A's storage goes there, the other to its own argument.
      double* const        u_into         {overwrite && m >= n ? a : u};
      const std::size_t    ld_u           {overwrite && m >= n ? static_cast<std::size_t>(lda) : static_cast<std::size_t>(ldu)};
      double* const        vt_into        {overwrite && m < n ? a : vt};
      const std::size_t    ld_vt          {overwrite && m < n ? static_cast<std::size_t>(lda) : static_cast<std::size_t>(ldvt)};
      for (std::size_t j {0}; j < left.cols(); ++j) {
        for (std::size_t i {0}; i < left.rows(); ++i) {
          u_into[i + (j * ld_u)] = left(i, j);
        }
      }
      for (std::size_t j {0}; j < right.cols(); ++j) {
        for (std::size_t i {0}; i < right.rows(); ++i) {
          vt_into[i + (j * ld_vt)] = right(i, j);
        }
      }
    }
    catch (const non_finite_error&) {
      return -4;
    }
    return 0;
  }

  /*!
   * \brief The QR factorisation `A = Q R` of a column-major `m × n` `A`, as LAPACK's `dgeqrf`: `R` overwrites the upper
   *        trapezoid of `A`; `Q = H₀ H₁ … H_{k-1}` (`k = min(m, n)`) is left as its reflectors `H_i = I − τ_i w wᵀ`,
   *        `w` 1 at row `i` and below it held in `A`'s column `i`, `τ_i` in \p tau (0 for the identity, which a column with
   *        nothing below the diagonal takes, as in `dlarfg`) — the form
   *        `dorgqr` and `dormqr` read. NumericAL's blocked Householder reflections compute them; the workspace is not
   *        used, but its size is `dgeqrf`'s least, `n`: an `lwork` of -1 asks it (written to `work[0]`), and a smaller
   *        one is illegal.
   * \return `INFO`: 0, or `-i` where argument `i` is illegal (`-3`: a NaN or an infinity in `A`).
   */
  inline int geqrf(std::ptrdiff_t m,
                   std::ptrdiff_t n,
                   double*        a,
                   std::ptrdiff_t lda,
                   double*        tau,
                   double*        work,
                   std::ptrdiff_t lwork)
  {
    if (m < 0) {
      return -1;
    }
    if (n < 0) {
      return -2;
    }
    if (lda < std::max<std::ptrdiff_t>(1, m)) {
      return -4;
    }
    const std::ptrdiff_t lwork_least {std::min(m, n) == 0 ? 1 : n};
    if (lwork == -1) {
      work[0] = static_cast<double>(lwork_least);
      return 0;
    }
    if (lwork < lwork_least) {
      return -7;
    }
    if (std::min(m, n) == 0) {
      return 0;
    }
    const auto     rows    {static_cast<std::size_t>(m)};
    const auto     cols    {static_cast<std::size_t>(n)};
    const auto     ld      {static_cast<std::size_t>(lda)};
    // In the layout the factorisation runs in (numerical::detail::householder_reflect), in storage the thread keeps: R and
    // the reflectors' w₁… where dgeqrf leaves them (numerical::detail::factor_qr_panel, as dlarfg); τ on each block's T
    // diagonal, 0 for the identity.
    buffer<double> entries {detail::take_kept<double>(0)};
#ifdef NUMERICAL_X86_COMPILED
    entries.resize(rows * cols); // column-major: copied in and out by columns
    for (std::size_t j {0}; j < cols; ++j) {
      std::copy_n(a + (j * ld), rows, entries.data() + (j * rows));
    }
#else
    detail::row_major_into(entries, a, lda, m, n, 1.0);
#endif
    if (!detail::all_finite(entries.data(), entries.size())) {
      detail::give_back(0, std::move(entries));
      return -3;
    }
#ifdef NUMERICAL_X86_COMPILED
    const std::vector<numerical::detail::reflector_block<double>> blocks {numerical::detail::reflect_in_columns(entries, rows, cols)};
    for (std::size_t j {0}; j < cols; ++j) {
      std::copy_n(entries.data() + (j * rows), rows, a + (j * ld));
    }
#else
    const std::vector<numerical::detail::reflector_block<double>> blocks {numerical::detail::reflect_in_rows(entries, rows, cols)};
    detail::column_major_into(entries.data(), rows, cols, a, ld);
#endif
    detail::give_back(0, std::move(entries));
    for (const numerical::detail::reflector_block<double>& block : blocks) {
      for (std::size_t j {0}; j < block.width; ++j) {
        tau[block.first + j] = block.t[(j * block.width) + j];
      }
    }
    return 0;
  }

  /*!
   * \brief The first \p n columns of `Q = H₀ H₁ … H_{k-1}`, from the reflectors of a `dgeqrf` (\ref geqrf's or any
   *        implementation's: column `i` of `A` below row `i`, `τ_i` in \p tau), overwriting the column-major `m × n`
   *        `A`, as LAPACK's `dorgqr`: the reflectors gathered by blocks into `I − V T Vᵀ` (`T` as `dlarft` forms it)
   *        and applied through the matrix product kernel. The workspace is not used, but its size is `dorgqr`'s
   *        least, `n`: an `lwork` of -1 asks it (written to `work[0]`), and a smaller one is illegal.
   * \return `INFO`: 0, or `-i` where argument `i` is illegal (`-4`: a NaN or an infinity in a reflector, `-6` in
   *         \p tau).
   */
  inline int orgqr(std::ptrdiff_t m,
                   std::ptrdiff_t n,
                   std::ptrdiff_t k,
                   double*        a,
                   std::ptrdiff_t lda,
                   const double*  tau,
                   double*        work,
                   std::ptrdiff_t lwork)
  {
    if (m < 0) {
      return -1;
    }
    if (n < 0 || n > m) {
      return -2;
    }
    if (k < 0 || k > n) {
      return -3;
    }
    if (lda < std::max<std::ptrdiff_t>(1, m)) {
      return -5;
    }
    const std::ptrdiff_t lwork_least {std::max<std::ptrdiff_t>(1, n)};
    if (lwork == -1) {
      work[0] = static_cast<double>(lwork_least);
      return 0;
    }
    if (lwork < lwork_least) {
      return -8;
    }
    if (n == 0) {
      return 0;
    }
    const auto rows  {static_cast<std::size_t>(m)};
    const auto cols  {static_cast<std::size_t>(n)};
    const auto depth {static_cast<std::size_t>(k)};
    const auto ld    {static_cast<std::size_t>(lda)};
    for (std::size_t j {0}; j < depth; ++j) {
      if (!std::isfinite(tau[j])) {
        return -6;
      }
      for (std::size_t i {j + 1}; i < rows; ++i) {
        if (!std::isfinite(a[i + (j * ld)])) {
          return -4;
        }
      }
    }
    // Each reflector w_j = (1, A(j+1…m-1, j)) and τ_j, gathered by blocks into I − V T Vᵀ (T as dlarft forms it).
    std::vector<buffer<double>> reflectors(depth);
    const std::vector<double>   taus(tau, tau + depth);
    for (std::size_t j {0}; j < depth; ++j) {
      reflectors[j].resize(rows - j);
      reflectors[j][0] = 1.0;
      std::copy_n(a + (j + 1) + (j * ld), rows - j - 1, reflectors[j].data() + 1);
    }
    std::vector<numerical::detail::reflector_block<double>> blocks;
    for (std::size_t k0 {0}; k0 < depth; k0 += numerical::detail::qr_block) {
      blocks.push_back(numerical::detail::gather_reflectors(reflectors, k0, std::min(k0 + numerical::detail::qr_block, depth), rows, &taus));
    }
    buffer<double> columns {detail::take_kept<double>(0)}; // Q column-major, as numerical::qr() forms it
    numerical::detail::form_q_into(columns, blocks, rows, 0, cols, true);
    for (std::size_t j {0}; j < cols; ++j) {
      std::copy_n(columns.data() + (j * rows), rows, a + (j * ld));
    }
    detail::give_back(0, std::move(columns));
    return 0;
  }

  /*!
   * \brief The minimum-norm least-squares solutions of `A X ≈ B`, column-major `A` `m × n` and `B` `max(m, n) × nrhs`,
   *        as LAPACK's `dgelsd`: a singular value at or under \p rcond times the largest counts as zero (`rcond ≤ 0` or
   *        `≥ 1`: \ref lapack_precision); `X` overwrites `B`'s first `n` rows, the singular values (descending) go to
   *        \p s and the effective rank to \p rank. Where `m > n`, `B`'s rows `n…m-1` of each column hold its residual
   *        as LAPACK leaves it, a vector whose norm is `‖b − A x‖` (here that norm, then zeros), from which NumPy's
   *        `lstsq` takes its residuals. `A` is left as it was. NumericAL's \ref least_squares computes them; the
   *        workspaces are not used, but their sizes are `dgelsd`'s: an `lwork` of -1 asks them (written to `work[0]`
   *        and `iwork[0]`), and a smaller `lwork` is illegal.
   * \return `INFO`: 0, or `-i` where argument `i` is illegal (`-4`, `-6`: a NaN or an infinity in `A`, in `B`).
   */
  template <typename Index>
  int gelsd(std::ptrdiff_t m,
            std::ptrdiff_t n,
            std::ptrdiff_t nrhs,
            const double*  a,
            std::ptrdiff_t lda,
            double*        b,
            std::ptrdiff_t ldb,
            double*        s,
            double         rcond,
            Index*         rank,
            double*        work,
            std::ptrdiff_t lwork,
            Index*         iwork)
  {
    if (m < 0) {
      return -1;
    }
    if (n < 0) {
      return -2;
    }
    if (nrhs < 0) {
      return -3;
    }
    if (lda < std::max<std::ptrdiff_t>(1, m)) {
      return -5;
    }
    if (ldb < std::max<std::ptrdiff_t>({1, m, n})) {
      return -7;
    }
    // dgelsd's sizes: its divide and conquer's leaves of 25 values, NLVL levels above them.
    constexpr std::ptrdiff_t leaf    {25};
    const std::ptrdiff_t     least   {std::max<std::ptrdiff_t>(1, std::min(m, n))};
    const auto               levels  {std::max<std::ptrdiff_t>(
                                        static_cast<std::ptrdiff_t>(std::log2(static_cast<double>(least) / static_cast<double>(leaf + 1))) + 1, 0)};
    const std::ptrdiff_t lwork_least {(3 * least) + (9 * least) + (2 * least * leaf) + (8 * least * levels) + (least * nrhs)
                                      + ((leaf + 1) * (leaf + 1))};
    if (lwork == -1) {
      work[0]  = static_cast<double>(lwork_least);
      iwork[0] = static_cast<Index>((3 * least * levels) + (11 * least));
      return 0;
    }
    if (lwork < lwork_least) {
      return -12;
    }
    *rank = 0;
    if (m == 0 || n == 0) {
      return 0;
    }
    const auto     rows {static_cast<std::size_t>(m)};
    const auto     cols {static_cast<std::size_t>(n)};
    const auto     rhs  {static_cast<std::size_t>(nrhs)};
    const auto     ld   {static_cast<std::size_t>(ldb)};
    buffer<double> right(rows * rhs, 0.0);
    for (std::size_t i {0}; i < rows; ++i) {
      for (std::size_t c {0}; c < rhs; ++c) {
        right[(i * rhs) + c] = b[i + (c * ld)];
      }
    }
    if (!std::ranges::all_of(right, [](double x) { return std::isfinite(x); })) {
      return -6;
    }
    const rank_cutoff cutoff {rank_cutoff::relative(rcond > 0.0 && rcond < 1.0 ? rcond : lapack_precision)};
    try {
      const matrix<double>                    left {rows, cols, detail::row_major(a, lda, m, n, 1.0)};
      const least_squares_solution<double>    fit  {least_squares(left, matrix<double> {rows, rhs, std::move(right)}, cutoff)};
      std::copy_n(&fit.singular_values[0], fit.singular_values.size(), s);
      *rank = static_cast<Index>(fit.rank);
      for (std::size_t c {0}; c < rhs; ++c) {
        for (std::size_t i {0}; i < cols; ++i) {
          b[i + (c * ld)] = fit.solution(i, c);
        }
        for (std::size_t i {cols}; i < rows; ++i) {
          b[i + (c * ld)] = i == cols && fit.residuals.size() > 0 ? std::sqrt(fit.residuals[c]) : 0.0;
        }
      }
    }
    catch (const non_finite_error&) {
      return -4;
    }
    return 0;
  }

  namespace detail {

    //! \brief The eigenvectors \p v (columns, complex) in `dgeev`'s real form, into the column-major \p out: column `j`
    //!        of a real eigenvalue its vector; of a pair `λ_j`, `λ_{j+1} = conj(λ_j)` (positive imaginary part first)
    //!        the real and imaginary parts of `λ_j`'s vector; conjugated where \p conjugate (the left vectors).
    inline void real_form(const matrix<std::complex<double>>&       v,
                          const vector<std::complex<double>>&       values,
                          const std::vector<std::size_t>&           column_of,
                          bool                                      conjugate,
                          double*                                   out,
                          std::size_t                               ld)
    {
      const std::size_t n    {values.size()};
      const double      sign {conjugate ? -1.0 : 1.0};
      for (std::size_t j {0}; j < n; ++j) {
        const std::size_t c {column_of[j]};
        if (values[j].imag() == 0.0) {
          for (std::size_t i {0}; i < n; ++i) {
            out[i + (j * ld)] = v(i, c).real();
          }
        }
        else if (values[j].imag() > 0.0) {
          for (std::size_t i {0}; i < n; ++i) {
            out[i + (j * ld)]       = v(i, c).real();
            out[i + ((j + 1) * ld)] = sign * v(i, c).imag();
          }
        }
      }
    }
  } // namespace detail

  /*!
   * \brief The eigenvalues, and with \p jobvl or \p jobvr `V` the left or right eigenvectors, of a column-major
   *        `n × n` `A`, as LAPACK's `dgeev`: the eigenvalues as \p wr + i \p wi, a conjugate pair adjacent with its
   *        positive imaginary part first; a right eigenvector `v` (`A v = λ v`) and a left one `u` (`uᴴ A = λ uᴴ`) of
   *        unit norm with its largest component real, in `VR` and `VL` as `dgeev` packs them (a real eigenvalue's
   *        column its vector, a pair's two columns the real and imaginary parts of the first's). NumericAL's
   *        \ref eigen_decomposition computes them; the left vectors are `Aᵀ`'s right ones, conjugated, each matched to
   *        the nearest unmatched eigenvalue. `A` is left as it was. The workspace is not used, but its size is
   *        `dgeev`'s least, `3n` (`4n` with vectors): an `lwork` of -1 asks it (written to `work[0]`), and a smaller
   *        one is illegal.
   * \return `INFO`: 0, or `-i` where argument `i` is illegal (`-4`: a NaN or an infinity in `A`).
   */
  inline int geev(char           jobvl,
                  char           jobvr,
                  std::ptrdiff_t n,
                  const double*  a,
                  std::ptrdiff_t lda,
                  double*        wr,
                  double*        wi,
                  double*        vl,
                  std::ptrdiff_t ldvl,
                  double*        vr,
                  std::ptrdiff_t ldvr,
                  double*        work,
                  std::ptrdiff_t lwork)
  {
    const bool left  {jobvl == 'V' || jobvl == 'v'};
    const bool right {jobvr == 'V' || jobvr == 'v'};
    if (!left && jobvl != 'N' && jobvl != 'n') {
      return -1;
    }
    if (!right && jobvr != 'N' && jobvr != 'n') {
      return -2;
    }
    if (n < 0) {
      return -3;
    }
    if (lda < std::max<std::ptrdiff_t>(1, n)) {
      return -5;
    }
    if (ldvl < 1 || (left && ldvl < n)) {
      return -9;
    }
    if (ldvr < 1 || (right && ldvr < n)) {
      return -11;
    }
    const std::ptrdiff_t lwork_least {std::max<std::ptrdiff_t>(1, (left || right ? 4 : 3) * n)};
    if (lwork == -1) {
      work[0] = static_cast<double>(lwork_least);
      return 0;
    }
    if (lwork < lwork_least) {
      return -13;
    }
    if (n == 0) {
      return 0;
    }
    const auto size {static_cast<std::size_t>(n)};
    try {
      const matrix<double>                      source {size, size, detail::row_major(a, lda, n, n, 1.0)};
      const eigen_decomposition<double>         eigen  {source, right};
      const vector<std::complex<double>>&       values {eigen.values()};
      for (std::size_t j {0}; j < size; ++j) {
        wr[j] = values[j].real();
        wi[j] = values[j].imag();
      }
      std::vector<std::size_t> same(size);
      for (std::size_t j {0}; j < size; ++j) {
        same[j] = j;
      }
      if (right) {
        detail::real_form(eigen.vectors(), values, same, false, vr, static_cast<std::size_t>(ldvr));
      }
      if (left) {
        const eigen_decomposition<double> transposed {matrix<double> {size, size, detail::columns_as_rows(a, lda, n, n, 1.0)}, true};
        // Aᵀ w = λ w is (conj w)ᴴ A = λ (conj w)ᴴ: each eigenvalue's left vector is the conjugate of Aᵀ's right one
        // for the same eigenvalue, the nearest one not yet taken (a multiple eigenvalue takes its vectors in turn).
        std::vector<std::size_t> column_of(size);
        std::vector<bool>        taken(size, false);
        for (std::size_t j {0}; j < size; ++j) {
          std::size_t best {size};
          for (std::size_t p {0}; p < size; ++p) {
            if (!taken[p] && (best == size || std::abs(transposed.values()[p] - values[j]) < std::abs(transposed.values()[best] - values[j]))) {
              best = p;
            }
          }
          taken[best]  = true;
          column_of[j] = best;
        }
        detail::real_form(transposed.vectors(), values, column_of, true, vl, static_cast<std::size_t>(ldvl));
      }
    }
    catch (const non_finite_error&) {
      return -4;
    }
    return 0;
  }
} // namespace numerical::blas

#endif // NUMERICAL_BLAS_HPP
