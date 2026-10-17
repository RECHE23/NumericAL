/*!
 * \file blas.hpp
 * \brief The column-major semantics of the BLAS and LAPACK routines NumericAL stands in for (`dgemm`, `dgesv`,
 *        `dgetrf`, `dgetrs`, `dpotrf`, `dpotrs`, `dsyevd`, `dgesdd`, `dgeqrf`, `dorgqr`), as their reference
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
#include <cmath>
#include <cstddef>
#include <vector>

#include "decomposition.hpp"
#include "gemm.hpp"
#include "matrix.hpp"

namespace numerical::blas {

  namespace detail {

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

    //! \brief `rows × cols` of the column-major \p a (\p lda apart), as a row-major buffer, each element times
    //!        \p scale.
    [[nodiscard]] inline buffer<double> row_major(const double * a,
                                                  std::ptrdiff_t lda,
                                                  std::ptrdiff_t rows,
                                                  std::ptrdiff_t cols,
                                                  double         scale)
    {
      const auto     r {static_cast<std::size_t>(rows)};
      const auto     c {static_cast<std::size_t>(cols)};
      buffer<double> out(r * c);
      for (std::size_t i {0}; i < r; ++i) {
        for (std::size_t j {0}; j < c; ++j) {
          out[(i * c) + j] = scale * a[i + (j * static_cast<std::size_t>(lda))];
        }
      }
      return out;
    }

    //! \brief The \p cols columns of \p rows elements of the column-major \p a (\p lda apart) as the rows of a
    //!        row-major buffer, each element times \p scale: `aᵀ`, row-major.
    [[nodiscard]] inline buffer<double> columns_as_rows(const double*  a,
                                                        std::ptrdiff_t lda,
                                                        std::ptrdiff_t rows,
                                                        std::ptrdiff_t cols,
                                                        double         scale)
    {
      const auto     r {static_cast<std::size_t>(rows)};
      const auto     c {static_cast<std::size_t>(cols)};
      buffer<double> out(r * c);
      for (std::size_t j {0}; j < c; ++j) {
        for (std::size_t i {0}; i < r; ++i) {
          out[(j * r) + i] = scale * a[i + (j * static_cast<std::size_t>(lda))];
        }
      }
      return out;
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
    template <typename Index>
    int rectangular_lu(std::size_t m,
                       std::size_t n,
                       double*     a,
                       std::size_t lda,
                       Index*      ipiv)
    {
      int info {0};
      for (std::size_t j {0}; j < std::min(m, n); ++j) {
        double* const column {a + (j * lda)};
        std::size_t   pivot  {j};
        for (std::size_t i {j + 1}; i < m; ++i) {
          if (std::abs(column[i]) > std::abs(column[pivot])) {
            pivot = i;
          }
        }
        ipiv[j] = static_cast<Index>(pivot + 1);
        if (column[pivot] == 0.0) {
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
          double* const target {a + (c * lda)};
          const double  factor {target[j]};
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
  inline int gemm(char           transa,
                  char           transb,
                  std::ptrdiff_t m,
                  std::ptrdiff_t n,
                  std::ptrdiff_t k,
                  double         alpha,
                  const double*  a,
                  std::ptrdiff_t lda,
                  const double*  b,
                  std::ptrdiff_t ldb,
                  double         beta,
                  double*        c,
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
      double* const column {c + (j * stride)};
      if (beta == 0.0) {
        std::fill_n(column, rows, 0.0);
      }
      else if (beta != 1.0) {
        std::transform(column, column + rows, column, [beta](double x) { return beta * x; });
      }
    }
    if (alpha == 0.0 || k == 0) {
      return 0;
    }
    // Column-major C is row-major Cᵀ (n × m): Cᵀ += (α op(B))ᵀ op(A)ᵀ. A column-major matrix is row-major as its
    // transpose, so op(B)ᵀ is B read in place when B is not transposed (unless α scales a copy), and op(A)ᵀ is A
    // read in place when A is not; a transposed one is copied.
    const bool     copy_b {detail::transposes(transb) || alpha != 1.0};
    buffer<double> left   {};
    if (detail::transposes(transb)) {
      left = detail::row_major(b, ldb, n, k, alpha);
    }
    else if (copy_b) {
      left = detail::columns_as_rows(b, ldb, k, n, alpha);
    }
    const double* const  l      {copy_b ? left.data() : b};
    const std::size_t    ldl    {copy_b ? static_cast<std::size_t>(k) : static_cast<std::size_t>(ldb)};
    const buffer<double> right  {detail::transposes(transa) ? detail::row_major(a, lda, k, m, 1.0) : buffer<double> {}};
    const double* const  r      {detail::transposes(transa) ? right.data() : a};
    const std::size_t    ldr    {detail::transposes(transa) ? static_cast<std::size_t>(m) : static_cast<std::size_t>(lda)};
    numerical::detail::gemm(cols, rows, static_cast<std::size_t>(k), l, ldl, r, ldr, c, stride);
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
  template <typename Index>
  int gesv(std::ptrdiff_t n,
           std::ptrdiff_t nrhs,
           double*        a,
           std::ptrdiff_t lda,
           Index*         ipiv,
           double*        b,
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
    matrix<double>           factors;
    bool                     singular {false};
    try {
      const lu_decomposition<double> lu {matrix<double> {size, size, detail::row_major(a, lda, n, n, 1.0)}};
      factors  = lu.factors();
      pivots   = lu.pivots();
      singular = lu.singular();
      if (!singular && nrhs > 0) {
        const matrix<double> x {lu.solve(matrix<double> {size, static_cast<std::size_t>(nrhs), detail::row_major(b, ldb, n, nrhs, 1.0)})};
        for (std::size_t j {0}; j < static_cast<std::size_t>(nrhs); ++j) {
          for (std::size_t i {0}; i < size; ++i) {
            b[i + (j * static_cast<std::size_t>(ldb))] = x(i, j);
          }
        }
      }
    }
    catch (const non_finite_error&) {
      return -3;
    }
    for (std::size_t j {0}; j < size; ++j) {
      for (std::size_t i {0}; i < size; ++i) {
        a[i + (j * ld)] = factors(i, j);
      }
    }
    detail::interchanges(pivots, ipiv);
    for (std::size_t i {0}; i < size; ++i) {
      if (factors(i, i) == 0.0) {
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
  inline int potrf(char           uplo,
                   std::ptrdiff_t n,
                   double*        a,
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
    const auto     size {static_cast<std::size_t>(n)};
    const auto     ld   {static_cast<std::size_t>(lda)};
    buffer<double> lower_rows(size * size, 0.0); // A's lower triangle, row-major
    for (std::size_t i {0}; i < size; ++i) {
      for (std::size_t j {0}; j <= i; ++j) {
        lower_rows[(i * size) + j] = lower ? a[i + (j * ld)] : a[j + (i * ld)];
      }
    }
    try {
      const matrix<double> l {cholesky_decomposition<double> {matrix<double> {size, size, std::move(lower_rows)}}.lower()};
      for (std::size_t i {0}; i < size; ++i) {
        for (std::size_t j {0}; j <= i; ++j) {
          (lower ? a[i + (j * ld)] : a[j + (i * ld)]) = l(i, j);
        }
      }
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
  template <typename Index>
  int getrf(std::ptrdiff_t m,
            std::ptrdiff_t n,
            double*        a,
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
        if (!std::isfinite(a[i + (j * ld)])) {
          return -3;
        }
      }
    }
    if (rows != cols) {
      return detail::rectangular_lu(rows, cols, a, ld, ipiv);
    }
    const lu_decomposition<double> lu      {matrix<double> {rows, cols, detail::row_major(a, lda, m, n, 1.0)}};
    const matrix<double>           factors {lu.factors()};
    for (std::size_t j {0}; j < cols; ++j) {
      for (std::size_t i {0}; i < rows; ++i) {
        a[i + (j * ld)] = factors(i, j);
      }
    }
    detail::interchanges(lu.pivots(), ipiv);
    for (std::size_t i {0}; i < rows; ++i) {
      if (factors(i, i) == 0.0) {
        return static_cast<int>(i + 1);
      }
    }
    return 0;
  }

  /*!
   * \brief `A X = B` or `Aᵀ X = B` (\p trans `N`, or `T`/`C`), column-major, from the factors and interchanges of a
   *        `dgetrf` (by any implementation), as LAPACK's `dgetrs`: `B` (`n × nrhs`) overwritten by `X`, each column by
   *        forward then backward substitution.
   * \return `INFO`: 0, or `-i` where argument `i` is illegal.
   */
  template <typename Index>
  int getrs(char           trans,
            std::ptrdiff_t n,
            std::ptrdiff_t nrhs,
            const double*  a,
            std::ptrdiff_t lda,
            const Index*   ipiv,
            double*        b,
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
    const auto at   = [&](std::size_t i, std::size_t j) { return a[i + (j * ld)]; };
    for (std::size_t c {0}; c < static_cast<std::size_t>(nrhs); ++c) {
      double* const x {b + (c * static_cast<std::size_t>(ldb))};
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
      for (std::size_t i {0}; i < size; ++i) { // Uᵀ, by rows of U's columns
        double sum {x[i]};
        for (std::size_t k {0}; k < i; ++k) {
          sum -= at(k, i) * x[k];
        }
        x[i] = sum / at(i, i);
      }
      for (std::size_t i {size}; i-- > 0;) { // Lᵀ
        double sum {x[i]};
        for (std::size_t k {i + 1}; k < size; ++k) {
          sum -= at(k, i) * x[k];
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
  inline int potrs(char           uplo,
                   std::ptrdiff_t n,
                   std::ptrdiff_t nrhs,
                   const double*  a,
                   std::ptrdiff_t lda,
                   double*        b,
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
    const auto l    = [&](std::size_t i, std::size_t j) { return lower ? a[i + (j * ld)] : a[j + (i * ld)]; }; // L(i, j), i ≥ j
    for (std::size_t c {0}; c < static_cast<std::size_t>(nrhs); ++c) {
      double* const x {b + (c * static_cast<std::size_t>(ldb))};
      for (std::size_t i {0}; i < size; ++i) { // L y = b
        double sum {x[i]};
        for (std::size_t k {0}; k < i; ++k) {
          sum -= l(i, k) * x[k];
        }
        x[i] = sum / l(i, i);
      }
      for (std::size_t i {size}; i-- > 0;) { // Lᵀ x = y
        double sum {x[i]};
        for (std::size_t k {i + 1}; k < size; ++k) {
          sum -= l(k, i) * x[k];
        }
        x[i] = sum / l(i, i);
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
   *        `w` 1 at row `i` and below it held in `A`'s column `i`, `τ_i` in \p tau (0 for the identity) — the form
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
    const auto           rows    {static_cast<std::size_t>(m)};
    const auto           cols    {static_cast<std::size_t>(n)};
    const auto           ld      {static_cast<std::size_t>(lda)};
    buffer<double>       entries {detail::row_major(a, lda, m, n, 1.0)};
    if (!std::ranges::all_of(entries, [](double x) { return std::isfinite(x); })) {
      return -3;
    }
    const numerical::detail::qr_reflections<double> factors {numerical::detail::householder_reflect(std::move(entries), rows, cols)};
    const std::size_t                               depth   {std::min(rows, cols)};
    for (std::size_t i {0}; i < depth; ++i) {
      for (std::size_t j {i}; j < cols; ++j) {
        a[i + (j * ld)] = factors.r[(i * cols) + j];
      }
    }
    // A unit reflector I − 2 v vᵀ is I − τ w wᵀ with w = v / v₀ and τ = 2 v₀²; v₀ is 0 only for an empty one (the
    // identity).
    for (const numerical::detail::reflector_block<double>& block : factors.blocks) {
      for (std::size_t j {0}; j < block.width; ++j) {
        const std::size_t k   {block.first + j};
        const double      top {block.v[(j * block.width) + j]};
        tau[k] = 2.0 * top * top; // 0 for an empty reflector (the identity), its v zero
        for (std::size_t r {1}; j + r < block.height; ++r) {
          a[(k + r) + (k * ld)] = tau[k] == 0.0 ? 0.0 : block.v[((j + r) * block.width) + j] / top;
        }
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
    std::vector<numerical::detail::reflector_block<double>> blocks;
    for (std::size_t k0 {0}; k0 < depth; k0 += numerical::detail::qr_block) {
      const std::size_t                          width {std::min(numerical::detail::qr_block, depth - k0)};
      numerical::detail::reflector_block<double> block {.first = k0, .height = rows - k0, .width = width, .v = {}, .t = {}};
      block.v.assign(block.height * width, 0.0);
      block.t.assign(width * width, 0.0);
      for (std::size_t j {0}; j < width; ++j) {
        block.v[(j * width) + j] = 1.0;
        for (std::size_t r {j + 1}; r < block.height; ++r) {
          block.v[(r * width) + j] = a[(k0 + r) + ((k0 + j) * ld)];
        }
      }
      // T(j, j) = τ_j; T(0…j-1, j) = −τ_j T(0…j-1, 0…j-1) V(:, 0…j-1)ᵀ v_j.
      for (std::size_t j {0}; j < width; ++j) {
        const double t_j {tau[k0 + j]};
        block.t[(j * width) + j] = t_j;
        std::vector<double> z(j, 0.0);
        for (std::size_t p {0}; p < j; ++p) {
          for (std::size_t r {j}; r < block.height; ++r) {
            z[p] += block.v[(r * width) + p] * block.v[(r * width) + j];
          }
        }
        for (std::size_t i {0}; i < j; ++i) {
          double sum {0.0};
          for (std::size_t p {i}; p < j; ++p) {
            sum += block.t[(i * width) + p] * z[p];
          }
          block.t[(i * width) + j] = -t_j * sum;
        }
      }
      blocks.push_back(std::move(block));
    }
    buffer<double> q(rows * cols, 0.0); // the first n columns of I, row-major, then Q's
    for (std::size_t c {0}; c < cols; ++c) {
      q[(c * cols) + c] = 1.0;
    }
    // Q = B₀ B₁ … applied right to left; the columns of I left of a block's first row are left as they are.
    for (std::size_t b {blocks.size()}; b-- > 0;) {
      numerical::detail::apply_reflector_block(blocks[b], false, q, cols, blocks[b].first, cols);
    }
    for (std::size_t j {0}; j < cols; ++j) {
      for (std::size_t i {0}; i < rows; ++i) {
        a[i + (j * ld)] = q[(i * cols) + j];
      }
    }
    return 0;
  }
} // namespace numerical::blas

#endif // NUMERICAL_BLAS_HPP
