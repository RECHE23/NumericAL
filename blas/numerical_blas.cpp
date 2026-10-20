// The Fortran calling convention of the BLAS and LAPACK routines NumericAL stands in for (ddot, dgeev, dgelsd, dgemm and
// sgemm, cgemm, zgemm, dgemv, sgesv, cgesv, zgesv, sgetrf…, sgetrs…, spotrf…, spotrs…, dgeqrf, dgesdd, dgesv, dgetrf, dgetrs, dorgqr, dpotrf, dpotrs, dsyevd, dsyrk, dtrsm): every argument by address, 32-bit integers (LP64), and a hidden
// length after the arguments for each character one, as gfortran passes it; under the same names ending in `64_`
// (dgemm_64_), 64-bit integers (ILP64). The semantics are numerical/blas.hpp's. An
// illegal argument is reported as the reference XERBLA reports it, on the standard error, and the call returns (as
// OpenBLAS's does) rather than stopping the program; no C++ exception leaves these functions.
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>

#include "numerical/blas.hpp"
#include "numerical_blas.hpp"

// Built with hidden visibility (the wheels' library), only the Fortran routines are visible (numerical_blas.hpp
// declares them so): NumericAL's own functions stay inside, where no other library in the process can bind to them
// (numerical_blas.map hides the standard library's instantiations, which keep their default visibility).

namespace {
  //! \brief The reference XERBLA's message for the illegal argument \p position of the routine \p name.
  void report_illegal(const char*    name,
                      std::ptrdiff_t position) noexcept
  {
    if (position > 0) {
      (void)std::fprintf(stderr, " ** On entry to %s parameter number %2d had an illegal value\n", name, static_cast<int>(position)); // NOLINT(cppcoreguidelines-pro-type-vararg): the reference's message
    }
  }

  //! \brief A failure no argument explains (memory exhausted): reported, the call then returns.
  void report_failure(const char* name) noexcept
  {
    (void)std::fprintf(stderr, " ** %s failed: %s\n", name, "out of memory or an internal error"); // NOLINT(cppcoreguidelines-pro-type-vararg): a message on the standard error
  }
} // namespace

extern "C" {
void dgemm_(const char*   transa,
            const char*   transb,
            const int*    m,
            const int*    n,
            const int*    k,
            const double* alpha,
            const double* a,
            const int*    lda,
            const double* b,
            const int*    ldb,
            const double* beta,
            double*       c,
            const int*    ldc,
            std::size_t /*transa_length*/,
            std::size_t /*transb_length*/) noexcept
{
  try {
    report_illegal("DGEMM ", numerical::blas::gemm(*transa, *transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc));
  }
  catch (...) {
    report_failure("DGEMM");
  }
}
}

namespace {
  //! \brief The `?gemm` of the scalar \p T and the integer \p Index (`int`, or 64 bits under the ILP64 names),
  //!        reported under \p name: the Fortran routines differ in nothing else.
  template <typename T, typename Index>
  void any_gemm(const char  *    name,
                const char  *    transa,
                const char  *    transb,
                const Index*     m,
                const Index*     n,
                const Index*     k,
                const T  *       alpha,
                const T  *       a,
                const Index*     lda,
                const T  *       b,
                const Index*     ldb,
                const T  *       beta,
                T  *             c,
                const Index*     ldc) noexcept
  {
    try {
      report_illegal(name, numerical::blas::gemm(*transa, *transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc));
    }
    catch (...) {
      report_failure(name);
    }
  }

  //! \brief The `?gesv` of the scalar \p T, reported under \p name.
  template <typename T, typename Index>
  void any_gesv(const char  * name,
                const Index*  n,
                const Index*  nrhs,
                T  *          a,
                const Index*  lda,
                Index*        ipiv,
                T  *          b,
                const Index*  ldb,
                Index*        info) noexcept
  {
    try {
      *info = numerical::blas::gesv(*n, *nrhs, a, *lda, ipiv, b, *ldb);
      report_illegal(name, -*info);
    }
    catch (...) {
      report_failure(name);
      *info = -1;
    }
  }

  //! \brief The `?getrf` of the scalar \p T, reported under \p name.
  template <typename T, typename Index>
  void any_getrf(const char  * name,
                 const Index*  m,
                 const Index*  n,
                 T  *          a,
                 const Index*  lda,
                 Index*        ipiv,
                 Index*        info) noexcept
  {
    try {
      *info = numerical::blas::getrf(*m, *n, a, *lda, ipiv);
      report_illegal(name, -*info);
    }
    catch (...) {
      report_failure(name);
      *info = -1;
    }
  }

  //! \brief The `?getrs` of the scalar \p T, reported under \p name.
  template <typename T, typename Index>
  void any_getrs(const char  * name,
                 const char  * trans,
                 const Index*  n,
                 const Index*  nrhs,
                 const T  *    a,
                 const Index*  lda,
                 const Index*  ipiv,
                 T  *          b,
                 const Index*  ldb,
                 Index*        info) noexcept
  {
    try {
      *info = numerical::blas::getrs(*trans, *n, *nrhs, a, *lda, ipiv, b, *ldb);
      report_illegal(name, -*info);
    }
    catch (...) {
      report_failure(name);
      *info = -1;
    }
  }

  //! \brief The `?potrf` of the scalar \p T, reported under \p name.
  template <typename T, typename Index>
  void any_potrf(const char  * name,
                 const char  * uplo,
                 const Index*  n,
                 T  *          a,
                 const Index*  lda,
                 Index*        info) noexcept
  {
    try {
      *info = numerical::blas::potrf(*uplo, *n, a, *lda);
      report_illegal(name, -*info);
    }
    catch (...) {
      report_failure(name);
      *info = -1;
    }
  }

  //! \brief The `?potrs` of the scalar \p T, reported under \p name.
  template <typename T, typename Index>
  void any_potrs(const char  * name,
                 const char  * uplo,
                 const Index*  n,
                 const Index*  nrhs,
                 const T  *    a,
                 const Index*  lda,
                 T  *          b,
                 const Index*  ldb,
                 Index*        info) noexcept
  {
    try {
      *info = numerical::blas::potrs(*uplo, *n, *nrhs, a, *lda, b, *ldb);
      report_illegal(name, -*info);
    }
    catch (...) {
      report_failure(name);
      *info = -1;
    }
  }

  //! \brief The complex dot product of \p n elements, conjugating \p x where \p conjugate (`?dotc`), as gfortran
  //!        returns it: \p Result holds its two parts.
  template <typename Result, typename T, typename Index>
  Result any_dot(bool                   conjugate,
                 const Index*           n,
                 const std::complex<T>* x,
                 const Index*           incx,
                 const std::complex<T>* y,
                 const Index*           incy) noexcept
  {
    const std::complex<T> sum {numerical::blas::complex_dot(conjugate, *n, x, *incx, y, *incy)};
    return Result {sum.real(), sum.imag()};
  }
} // namespace

extern "C" {
void sgemm_(const char*  transa,
            const char*  transb,
            const int*   m,
            const int*   n,
            const int*   k,
            const float* alpha,
            const float* a,
            const int*   lda,
            const float* b,
            const int*   ldb,
            const float* beta,
            float*       c,
            const int*   ldc,
            std::size_t /*transa_length*/,
            std::size_t /*transb_length*/) noexcept
{
  any_gemm("SGEMM ", transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemm_(const char*                transa,
            const char*                transb,
            const int*                 m,
            const int*                 n,
            const int*                 k,
            const std::complex<float>* alpha,
            const std::complex<float>* a,
            const int*                 lda,
            const std::complex<float>* b,
            const int*                 ldb,
            const std::complex<float>* beta,
            std::complex<float>*       c,
            const int*                 ldc,
            std::size_t /*transa_length*/,
            std::size_t /*transb_length*/) noexcept
{
  any_gemm("CGEMM ", transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zgemm_(const char*                 transa,
            const char*                 transb,
            const int*                  m,
            const int*                  n,
            const int*                  k,
            const std::complex<double>* alpha,
            const std::complex<double>* a,
            const int*                  lda,
            const std::complex<double>* b,
            const int*                  ldb,
            const std::complex<double>* beta,
            std::complex<double>*       c,
            const int*                  ldc,
            std::size_t /*transa_length*/,
            std::size_t /*transb_length*/) noexcept
{
  any_gemm("ZGEMM ", transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void sgesv_(const int* n,
            const int* nrhs,
            float    * a,
            const int* lda,
            int      * ipiv,
            float    * b,
            const int* ldb,
            int      * info) noexcept
{
  any_gesv("SGESV ", n, nrhs, a, lda, ipiv, b, ldb, info);
}

void sgetrf_(const int* m,
             const int* n,
             float    * a,
             const int* lda,
             int      * ipiv,
             int      * info) noexcept
{
  any_getrf("SGETRF", m, n, a, lda, ipiv, info);
}

void sgetrs_(const char * trans,
             const int  * n,
             const int  * nrhs,
             const float* a,
             const int  * lda,
             const int  * ipiv,
             float      * b,
             const int  * ldb,
             int        * info,
             std::size_t /*trans_length*/) noexcept
{
  any_getrs("SGETRS", trans, n, nrhs, a, lda, ipiv, b, ldb, info);
}

void cgesv_(const int          * n,
            const int          * nrhs,
            std::complex<float>* a,
            const int          * lda,
            int                * ipiv,
            std::complex<float>* b,
            const int          * ldb,
            int                * info) noexcept
{
  any_gesv("CGESV ", n, nrhs, a, lda, ipiv, b, ldb, info);
}

void cgetrf_(const int          * m,
             const int          * n,
             std::complex<float>* a,
             const int          * lda,
             int                * ipiv,
             int                * info) noexcept
{
  any_getrf("CGETRF", m, n, a, lda, ipiv, info);
}

void cgetrs_(const char               * trans,
             const int                * n,
             const int                * nrhs,
             const std::complex<float>* a,
             const int                * lda,
             const int                * ipiv,
             std::complex<float>      * b,
             const int                * ldb,
             int                      * info,
             std::size_t /*trans_length*/) noexcept
{
  any_getrs("CGETRS", trans, n, nrhs, a, lda, ipiv, b, ldb, info);
}

void zgesv_(const int           * n,
            const int           * nrhs,
            std::complex<double>* a,
            const int           * lda,
            int                 * ipiv,
            std::complex<double>* b,
            const int           * ldb,
            int                 * info) noexcept
{
  any_gesv("ZGESV ", n, nrhs, a, lda, ipiv, b, ldb, info);
}

void zgetrf_(const int           * m,
             const int           * n,
             std::complex<double>* a,
             const int           * lda,
             int                 * ipiv,
             int                 * info) noexcept
{
  any_getrf("ZGETRF", m, n, a, lda, ipiv, info);
}

void zgetrs_(const char                * trans,
             const int                 * n,
             const int                 * nrhs,
             const std::complex<double>* a,
             const int                 * lda,
             const int                 * ipiv,
             std::complex<double>      * b,
             const int                 * ldb,
             int                       * info,
             std::size_t /*trans_length*/) noexcept
{
  any_getrs("ZGETRS", trans, n, nrhs, a, lda, ipiv, b, ldb, info);
}

void spotrf_(const char* uplo,
             const int * n,
             float     * a,
             const int * lda,
             int       * info,
             std::size_t /*uplo_length*/) noexcept
{
  any_potrf("SPOTRF", uplo, n, a, lda, info);
}

void spotrs_(const char * uplo,
             const int  * n,
             const int  * nrhs,
             const float* a,
             const int  * lda,
             float      * b,
             const int  * ldb,
             int        * info,
             std::size_t /*uplo_length*/) noexcept
{
  any_potrs("SPOTRS", uplo, n, nrhs, a, lda, b, ldb, info);
}

void cpotrf_(const char         * uplo,
             const int          * n,
             std::complex<float>* a,
             const int          * lda,
             int                * info,
             std::size_t /*uplo_length*/) noexcept
{
  any_potrf("CPOTRF", uplo, n, a, lda, info);
}

void cpotrs_(const char               * uplo,
             const int                * n,
             const int                * nrhs,
             const std::complex<float>* a,
             const int                * lda,
             std::complex<float>      * b,
             const int                * ldb,
             int                      * info,
             std::size_t /*uplo_length*/) noexcept
{
  any_potrs("CPOTRS", uplo, n, nrhs, a, lda, b, ldb, info);
}

void zpotrf_(const char          * uplo,
             const int           * n,
             std::complex<double>* a,
             const int           * lda,
             int                 * info,
             std::size_t /*uplo_length*/) noexcept
{
  any_potrf("ZPOTRF", uplo, n, a, lda, info);
}

void zpotrs_(const char                * uplo,
             const int                 * n,
             const int                 * nrhs,
             const std::complex<double>* a,
             const int                 * lda,
             std::complex<double>      * b,
             const int                 * ldb,
             int                       * info,
             std::size_t /*uplo_length*/) noexcept
{
  any_potrs("ZPOTRS", uplo, n, nrhs, a, lda, b, ldb, info);
}

numerical_blas_complex_float cdotu_(const int                * n,
                                    const std::complex<float>* x,
                                    const int                * incx,
                                    const std::complex<float>* y,
                                    const int                * incy) noexcept
{
  return any_dot<numerical_blas_complex_float>(false, n, x, incx, y, incy);
}

numerical_blas_complex_float cdotc_(const int                * n,
                                    const std::complex<float>* x,
                                    const int                * incx,
                                    const std::complex<float>* y,
                                    const int                * incy) noexcept
{
  return any_dot<numerical_blas_complex_float>(true, n, x, incx, y, incy);
}

numerical_blas_complex_double zdotu_(const int                 * n,
                                     const std::complex<double>* x,
                                     const int                 * incx,
                                     const std::complex<double>* y,
                                     const int                 * incy) noexcept
{
  return any_dot<numerical_blas_complex_double>(false, n, x, incx, y, incy);
}

numerical_blas_complex_double zdotc_(const int                 * n,
                                     const std::complex<double>* x,
                                     const int                 * incx,
                                     const std::complex<double>* y,
                                     const int                 * incy) noexcept
{
  return any_dot<numerical_blas_complex_double>(true, n, x, incx, y, incy);
}

float sdot_(const int  * n,
            const float* x,
            const int  * incx,
            const float* y,
            const int  * incy) noexcept
{
  try {
    return numerical::blas::dot(*n, x, *incx, y, *incy);
  }
  catch (...) {
    report_failure("SDOT");
    return 0.0F;
  }
}

double ddot_(const int*    n,
             const double* x,
             const int*    incx,
             const double* y,
             const int*    incy) noexcept
{
  try {
    return numerical::blas::dot(*n, x, *incx, y, *incy);
  }
  catch (...) {
    report_failure("DDOT");
    return 0.0;
  }
}

void dgemv_(const char*   trans,
            const int*    m,
            const int*    n,
            const double* alpha,
            const double* a,
            const int*    lda,
            const double* x,
            const int*    incx,
            const double* beta,
            double*       y,
            const int*    incy,
            std::size_t /*trans_length*/) noexcept
{
  try {
    report_illegal("DGEMV ", numerical::blas::gemv(*trans, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy));
  }
  catch (...) {
    report_failure("DGEMV");
  }
}

void dsyrk_(const char*   uplo,
            const char*   trans,
            const int*    n,
            const int*    k,
            const double* alpha,
            const double* a,
            const int*    lda,
            const double* beta,
            double*       c,
            const int*    ldc,
            std::size_t /*uplo_length*/,
            std::size_t /*trans_length*/) noexcept
{
  try {
    report_illegal("DSYRK ", numerical::blas::syrk(*uplo, *trans, *n, *k, *alpha, a, *lda, *beta, c, *ldc));
  }
  catch (...) {
    report_failure("DSYRK");
  }
}

void dtrsm_(const char*   side,
            const char*   uplo,
            const char*   transa,
            const char*   diag,
            const int*    m,
            const int*    n,
            const double* alpha,
            const double* a,
            const int*    lda,
            double*       b,
            const int*    ldb,
            std::size_t /*side_length*/,
            std::size_t /*uplo_length*/,
            std::size_t /*transa_length*/,
            std::size_t /*diag_length*/) noexcept
{
  try {
    report_illegal("DTRSM ", numerical::blas::trsm(*side, *uplo, *transa, *diag, *m, *n, *alpha, a, *lda, b, *ldb));
  }
  catch (...) {
    report_failure("DTRSM");
  }
}

void dgesv_(const int* n,
            const int* nrhs,
            double*    a,
            const int* lda,
            int*       ipiv,
            double*    b,
            const int* ldb,
            int*       info) noexcept
{
  try {
    *info = numerical::blas::gesv(*n, *nrhs, a, *lda, ipiv, b, *ldb);
    report_illegal("DGESV ", -*info);
  }
  catch (...) {
    report_failure("DGESV");
    *info = -1;
  }
}

void dpotrf_(const char* uplo,
             const int*  n,
             double*     a,
             const int*  lda,
             int*        info,
             std::size_t /*uplo_length*/) noexcept
{
  try {
    *info = numerical::blas::potrf(*uplo, *n, a, *lda);
    report_illegal("DPOTRF", -*info);
  }
  catch (...) {
    report_failure("DPOTRF");
    *info = -1;
  }
}

void dgetrf_(const int* m,
             const int* n,
             double*    a,
             const int* lda,
             int*       ipiv,
             int*       info) noexcept
{
  try {
    *info = numerical::blas::getrf(*m, *n, a, *lda, ipiv);
    report_illegal("DGETRF", -*info);
  }
  catch (...) {
    report_failure("DGETRF");
    *info = -1;
  }
}

void dgetrs_(const char*   trans,
             const int*    n,
             const int*    nrhs,
             const double* a,
             const int*    lda,
             const int*    ipiv,
             double*       b,
             const int*    ldb,
             int*          info,
             std::size_t /*trans_length*/) noexcept
{
  try {
    *info = numerical::blas::getrs(*trans, *n, *nrhs, a, *lda, ipiv, b, *ldb);
    report_illegal("DGETRS", -*info);
  }
  catch (...) {
    report_failure("DGETRS");
    *info = -1;
  }
}

void dpotrs_(const char*   uplo,
             const int*    n,
             const int*    nrhs,
             const double* a,
             const int*    lda,
             double*       b,
             const int*    ldb,
             int*          info,
             std::size_t /*uplo_length*/) noexcept
{
  try {
    *info = numerical::blas::potrs(*uplo, *n, *nrhs, a, *lda, b, *ldb);
    report_illegal("DPOTRS", -*info);
  }
  catch (...) {
    report_failure("DPOTRS");
    *info = -1;
  }
}

void dsyevd_(const char* jobz,
             const char* uplo,
             const int*  n,
             double*     a,
             const int*  lda,
             double*     w,
             double*     work,
             const int*  lwork,
             int*        iwork,
             const int*  liwork,
             int*        info,
             std::size_t /*jobz_length*/,
             std::size_t /*uplo_length*/) noexcept
{
  try {
    *info = numerical::blas::syevd(*jobz, *uplo, *n, a, *lda, w, work, *lwork, iwork, *liwork);
    report_illegal("DSYEVD", -*info);
  }
  catch (...) {
    report_failure("DSYEVD");
    *info = -1;
  }
}

void dgesdd_(const char* jobz,
             const int*  m,
             const int*  n,
             double*     a,
             const int*  lda,
             double*     s,
             double*     u,
             const int*  ldu,
             double*     vt,
             const int*  ldvt,
             double*     work,
             const int*  lwork,
             int*        iwork,
             int*        info,
             std::size_t /*jobz_length*/) noexcept
{
  try {
    *info = numerical::blas::gesdd(*jobz, *m, *n, a, *lda, s, u, *ldu, vt, *ldvt, work, *lwork, iwork);
    report_illegal("DGESDD", -*info);
  }
  catch (...) {
    report_failure("DGESDD");
    *info = -1;
  }
}

void dgeqrf_(const int* m,
             const int* n,
             double*    a,
             const int* lda,
             double*    tau,
             double*    work,
             const int* lwork,
             int*       info) noexcept
{
  try {
    *info = numerical::blas::geqrf(*m, *n, a, *lda, tau, work, *lwork);
    report_illegal("DGEQRF", -*info);
  }
  catch (...) {
    report_failure("DGEQRF");
    *info = -1;
  }
}

void dorgqr_(const int*    m,
             const int*    n,
             const int*    k,
             double*       a,
             const int*    lda,
             const double* tau,
             double*       work,
             const int*    lwork,
             int*          info) noexcept
{
  try {
    *info = numerical::blas::orgqr(*m, *n, *k, a, *lda, tau, work, *lwork);
    report_illegal("DORGQR", -*info);
  }
  catch (...) {
    report_failure("DORGQR");
    *info = -1;
  }
}

void dgelsd_(const int*    m,
             const int*    n,
             const int*    nrhs,
             const double* a,
             const int*    lda,
             double*       b,
             const int*    ldb,
             double*       s,
             const double* rcond,
             int*          rank,
             double*       work,
             const int*    lwork,
             int*          iwork,
             int*          info) noexcept
{
  try {
    *info = numerical::blas::gelsd(*m, *n, *nrhs, a, *lda, b, *ldb, s, *rcond, rank, work, *lwork, iwork);
    report_illegal("DGELSD", -*info);
  }
  catch (...) {
    report_failure("DGELSD");
    *info = -1;
  }
}

void dgeev_(const char*   jobvl,
            const char*   jobvr,
            const int*    n,
            const double* a,
            const int*    lda,
            double*       wr,
            double*       wi,
            double*       vl,
            const int*    ldvl,
            double*       vr,
            const int*    ldvr,
            double*       work,
            const int*    lwork,
            int*          info,
            std::size_t /*jobvl_length*/,
            std::size_t /*jobvr_length*/) noexcept
{
  try {
    *info = numerical::blas::geev(*jobvl, *jobvr, *n, a, *lda, wr, wi, vl, *ldvl, vr, *ldvr, work, *lwork);
    report_illegal("DGEEV ", -*info);
  }
  catch (...) {
    report_failure("DGEEV");
    *info = -1;
  }
}
}

// The ILP64 interface (OpenBLAS's `64_` suffix, Julia's libblastrampoline): the same routines, every integer 64 bits.
extern "C" {
void dgemm_64_(const char         *   transa,
               const char         *   transb,
               const std::int64_t*    m,
               const std::int64_t*    n,
               const std::int64_t*    k,
               const double         * alpha,
               const double         * a,
               const std::int64_t*    lda,
               const double         * b,
               const std::int64_t*    ldb,
               const double         * beta,
               double         *       c,
               const std::int64_t*    ldc,
               std::size_t /*transa_length*/,
               std::size_t /*transb_length*/) noexcept
{
  try {
    report_illegal("DGEMM ", numerical::blas::gemm(*transa, *transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc));
  }
  catch (...) {
    report_failure("DGEMM");
  }
}
}

extern "C" {
void sgemm_64_(const char         *  transa,
               const char         *  transb,
               const std::int64_t*   m,
               const std::int64_t*   n,
               const std::int64_t*   k,
               const float         * alpha,
               const float         * a,
               const std::int64_t*   lda,
               const float         * b,
               const std::int64_t*   ldb,
               const float         * beta,
               float         *       c,
               const std::int64_t*   ldc,
               std::size_t /*transa_length*/,
               std::size_t /*transb_length*/) noexcept
{
  any_gemm("SGEMM ", transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemm_64_(const char         *                transa,
               const char         *                transb,
               const std::int64_t*                 m,
               const std::int64_t*                 n,
               const std::int64_t*                 k,
               const std::complex<float>         * alpha,
               const std::complex<float>         * a,
               const std::int64_t*                 lda,
               const std::complex<float>         * b,
               const std::int64_t*                 ldb,
               const std::complex<float>         * beta,
               std::complex<float>         *       c,
               const std::int64_t*                 ldc,
               std::size_t /*transa_length*/,
               std::size_t /*transb_length*/) noexcept
{
  any_gemm("CGEMM ", transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zgemm_64_(const char         *                 transa,
               const char         *                 transb,
               const std::int64_t*                  m,
               const std::int64_t*                  n,
               const std::int64_t*                  k,
               const std::complex<double>         * alpha,
               const std::complex<double>         * a,
               const std::int64_t*                  lda,
               const std::complex<double>         * b,
               const std::int64_t*                  ldb,
               const std::complex<double>         * beta,
               std::complex<double>         *       c,
               const std::int64_t*                  ldc,
               std::size_t /*transa_length*/,
               std::size_t /*transb_length*/) noexcept
{
  any_gemm("ZGEMM ", transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void sgesv_64_(const std::int64_t         * n,
               const std::int64_t         * nrhs,
               float                      * a,
               const std::int64_t         * lda,
               std::int64_t               * ipiv,
               float                      * b,
               const std::int64_t         * ldb,
               std::int64_t               * info) noexcept
{
  any_gesv("SGESV ", n, nrhs, a, lda, ipiv, b, ldb, info);
}

void sgetrf_64_(const std::int64_t         * m,
                const std::int64_t         * n,
                float                      * a,
                const std::int64_t         * lda,
                std::int64_t               * ipiv,
                std::int64_t               * info) noexcept
{
  any_getrf("SGETRF", m, n, a, lda, ipiv, info);
}

void sgetrs_64_(const char          * trans,
                const std::int64_t  * n,
                const std::int64_t  * nrhs,
                const float         * a,
                const std::int64_t  * lda,
                const std::int64_t  * ipiv,
                float               * b,
                const std::int64_t  * ldb,
                std::int64_t        * info,
                std::size_t /*trans_length*/) noexcept
{
  any_getrs("SGETRS", trans, n, nrhs, a, lda, ipiv, b, ldb, info);
}

void cgesv_64_(const std::int64_t          * n,
               const std::int64_t          * nrhs,
               std::complex<float>         * a,
               const std::int64_t          * lda,
               std::int64_t                * ipiv,
               std::complex<float>         * b,
               const std::int64_t          * ldb,
               std::int64_t                * info) noexcept
{
  any_gesv("CGESV ", n, nrhs, a, lda, ipiv, b, ldb, info);
}

void cgetrf_64_(const std::int64_t          * m,
                const std::int64_t          * n,
                std::complex<float>         * a,
                const std::int64_t          * lda,
                std::int64_t                * ipiv,
                std::int64_t                * info) noexcept
{
  any_getrf("CGETRF", m, n, a, lda, ipiv, info);
}

void cgetrs_64_(const char                        * trans,
                const std::int64_t                * n,
                const std::int64_t                * nrhs,
                const std::complex<float>         * a,
                const std::int64_t                * lda,
                const std::int64_t                * ipiv,
                std::complex<float>               * b,
                const std::int64_t                * ldb,
                std::int64_t                      * info,
                std::size_t /*trans_length*/) noexcept
{
  any_getrs("CGETRS", trans, n, nrhs, a, lda, ipiv, b, ldb, info);
}

void zgesv_64_(const std::int64_t           * n,
               const std::int64_t           * nrhs,
               std::complex<double>         * a,
               const std::int64_t           * lda,
               std::int64_t                 * ipiv,
               std::complex<double>         * b,
               const std::int64_t           * ldb,
               std::int64_t                 * info) noexcept
{
  any_gesv("ZGESV ", n, nrhs, a, lda, ipiv, b, ldb, info);
}

void zgetrf_64_(const std::int64_t           * m,
                const std::int64_t           * n,
                std::complex<double>         * a,
                const std::int64_t           * lda,
                std::int64_t                 * ipiv,
                std::int64_t                 * info) noexcept
{
  any_getrf("ZGETRF", m, n, a, lda, ipiv, info);
}

void zgetrs_64_(const char                         * trans,
                const std::int64_t                 * n,
                const std::int64_t                 * nrhs,
                const std::complex<double>         * a,
                const std::int64_t                 * lda,
                const std::int64_t                 * ipiv,
                std::complex<double>               * b,
                const std::int64_t                 * ldb,
                std::int64_t                       * info,
                std::size_t /*trans_length*/) noexcept
{
  any_getrs("ZGETRS", trans, n, nrhs, a, lda, ipiv, b, ldb, info);
}

void spotrf_64_(const char                  * uplo,
                const std::int64_t          * n,
                float                       * a,
                const std::int64_t          * lda,
                std::int64_t                * info,
                std::size_t /*uplo_length*/) noexcept
{
  any_potrf("SPOTRF", uplo, n, a, lda, info);
}

void spotrs_64_(const char          * uplo,
                const std::int64_t  * n,
                const std::int64_t  * nrhs,
                const float         * a,
                const std::int64_t  * lda,
                float               * b,
                const std::int64_t  * ldb,
                std::int64_t        * info,
                std::size_t /*uplo_length*/) noexcept
{
  any_potrs("SPOTRS", uplo, n, nrhs, a, lda, b, ldb, info);
}

void cpotrf_64_(const char                  * uplo,
                const std::int64_t          * n,
                std::complex<float>         * a,
                const std::int64_t          * lda,
                std::int64_t                * info,
                std::size_t /*uplo_length*/) noexcept
{
  any_potrf("CPOTRF", uplo, n, a, lda, info);
}

void cpotrs_64_(const char                        * uplo,
                const std::int64_t                * n,
                const std::int64_t                * nrhs,
                const std::complex<float>         * a,
                const std::int64_t                * lda,
                std::complex<float>               * b,
                const std::int64_t                * ldb,
                std::int64_t                      * info,
                std::size_t /*uplo_length*/) noexcept
{
  any_potrs("CPOTRS", uplo, n, nrhs, a, lda, b, ldb, info);
}

void zpotrf_64_(const char                   * uplo,
                const std::int64_t           * n,
                std::complex<double>         * a,
                const std::int64_t           * lda,
                std::int64_t                 * info,
                std::size_t /*uplo_length*/) noexcept
{
  any_potrf("ZPOTRF", uplo, n, a, lda, info);
}

void zpotrs_64_(const char                         * uplo,
                const std::int64_t                 * n,
                const std::int64_t                 * nrhs,
                const std::complex<double>         * a,
                const std::int64_t                 * lda,
                std::complex<double>               * b,
                const std::int64_t                 * ldb,
                std::int64_t                       * info,
                std::size_t /*uplo_length*/) noexcept
{
  any_potrs("ZPOTRS", uplo, n, nrhs, a, lda, b, ldb, info);
}

numerical_blas_complex_float cdotu_64_(const std::int64_t       * n,
                                       const std::complex<float>* x,
                                       const std::int64_t       * incx,
                                       const std::complex<float>* y,
                                       const std::int64_t       * incy) noexcept
{
  return any_dot<numerical_blas_complex_float>(false, n, x, incx, y, incy);
}

numerical_blas_complex_float cdotc_64_(const std::int64_t       * n,
                                       const std::complex<float>* x,
                                       const std::int64_t       * incx,
                                       const std::complex<float>* y,
                                       const std::int64_t       * incy) noexcept
{
  return any_dot<numerical_blas_complex_float>(true, n, x, incx, y, incy);
}

numerical_blas_complex_double zdotu_64_(const std::int64_t        * n,
                                        const std::complex<double>* x,
                                        const std::int64_t        * incx,
                                        const std::complex<double>* y,
                                        const std::int64_t        * incy) noexcept
{
  return any_dot<numerical_blas_complex_double>(false, n, x, incx, y, incy);
}

numerical_blas_complex_double zdotc_64_(const std::int64_t        * n,
                                        const std::complex<double>* x,
                                        const std::int64_t        * incx,
                                        const std::complex<double>* y,
                                        const std::int64_t        * incy) noexcept
{
  return any_dot<numerical_blas_complex_double>(true, n, x, incx, y, incy);
}

float sdot_64_(const std::int64_t* n,
               const float       * x,
               const std::int64_t* incx,
               const float       * y,
               const std::int64_t* incy) noexcept
{
  try {
    return numerical::blas::dot(*n, x, *incx, y, *incy);
  }
  catch (...) {
    report_failure("SDOT");
    return 0.0F;
  }
}

double ddot_64_(const std::int64_t*    n,
                const double         * x,
                const std::int64_t*    incx,
                const double         * y,
                const std::int64_t*    incy) noexcept
{
  try {
    return numerical::blas::dot(*n, x, *incx, y, *incy);
  }
  catch (...) {
    report_failure("DDOT");
    return 0.0;
  }
}

void dgemv_64_(const char         *   trans,
               const std::int64_t*    m,
               const std::int64_t*    n,
               const double         * alpha,
               const double         * a,
               const std::int64_t*    lda,
               const double         * x,
               const std::int64_t*    incx,
               const double         * beta,
               double         *       y,
               const std::int64_t*    incy,
               std::size_t /*trans_length*/) noexcept
{
  try {
    report_illegal("DGEMV ", numerical::blas::gemv(*trans, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy));
  }
  catch (...) {
    report_failure("DGEMV");
  }
}

void dsyrk_64_(const char         *   uplo,
               const char         *   trans,
               const std::int64_t*    n,
               const std::int64_t*    k,
               const double         * alpha,
               const double         * a,
               const std::int64_t*    lda,
               const double         * beta,
               double         *       c,
               const std::int64_t*    ldc,
               std::size_t /*uplo_length*/,
               std::size_t /*trans_length*/) noexcept
{
  try {
    report_illegal("DSYRK ", numerical::blas::syrk(*uplo, *trans, *n, *k, *alpha, a, *lda, *beta, c, *ldc));
  }
  catch (...) {
    report_failure("DSYRK");
  }
}

void dtrsm_64_(const char         *   side,
               const char         *   uplo,
               const char         *   transa,
               const char         *   diag,
               const std::int64_t*    m,
               const std::int64_t*    n,
               const double         * alpha,
               const double         * a,
               const std::int64_t*    lda,
               double         *       b,
               const std::int64_t*    ldb,
               std::size_t /*side_length*/,
               std::size_t /*uplo_length*/,
               std::size_t /*transa_length*/,
               std::size_t /*diag_length*/) noexcept
{
  try {
    report_illegal("DTRSM ", numerical::blas::trsm(*side, *uplo, *transa, *diag, *m, *n, *alpha, a, *lda, b, *ldb));
  }
  catch (...) {
    report_failure("DTRSM");
  }
}

void dgesv_64_(const std::int64_t* n,
               const std::int64_t* nrhs,
               double         *    a,
               const std::int64_t* lda,
               std::int64_t*       ipiv,
               double         *    b,
               const std::int64_t* ldb,
               std::int64_t*       info) noexcept
{
  try {
    *info = numerical::blas::gesv(*n, *nrhs, a, *lda, ipiv, b, *ldb);
    report_illegal("DGESV ", -*info);
  }
  catch (...) {
    report_failure("DGESV");
    *info = -1;
  }
}

void dpotrf_64_(const char         * uplo,
                const std::int64_t*  n,
                double         *     a,
                const std::int64_t*  lda,
                std::int64_t*        info,
                std::size_t /*uplo_length*/) noexcept
{
  try {
    *info = numerical::blas::potrf(*uplo, *n, a, *lda);
    report_illegal("DPOTRF", -*info);
  }
  catch (...) {
    report_failure("DPOTRF");
    *info = -1;
  }
}

void dgetrf_64_(const std::int64_t* m,
                const std::int64_t* n,
                double         *    a,
                const std::int64_t* lda,
                std::int64_t*       ipiv,
                std::int64_t*       info) noexcept
{
  try {
    *info = numerical::blas::getrf(*m, *n, a, *lda, ipiv);
    report_illegal("DGETRF", -*info);
  }
  catch (...) {
    report_failure("DGETRF");
    *info = -1;
  }
}

void dgetrs_64_(const char         *   trans,
                const std::int64_t*    n,
                const std::int64_t*    nrhs,
                const double         * a,
                const std::int64_t*    lda,
                const std::int64_t*    ipiv,
                double         *       b,
                const std::int64_t*    ldb,
                std::int64_t*          info,
                std::size_t /*trans_length*/) noexcept
{
  try {
    *info = numerical::blas::getrs(*trans, *n, *nrhs, a, *lda, ipiv, b, *ldb);
    report_illegal("DGETRS", -*info);
  }
  catch (...) {
    report_failure("DGETRS");
    *info = -1;
  }
}

void dpotrs_64_(const char         *   uplo,
                const std::int64_t*    n,
                const std::int64_t*    nrhs,
                const double         * a,
                const std::int64_t*    lda,
                double         *       b,
                const std::int64_t*    ldb,
                std::int64_t*          info,
                std::size_t /*uplo_length*/) noexcept
{
  try {
    *info = numerical::blas::potrs(*uplo, *n, *nrhs, a, *lda, b, *ldb);
    report_illegal("DPOTRS", -*info);
  }
  catch (...) {
    report_failure("DPOTRS");
    *info = -1;
  }
}

void dsyevd_64_(const char         * jobz,
                const char         * uplo,
                const std::int64_t*  n,
                double         *     a,
                const std::int64_t*  lda,
                double         *     w,
                double         *     work,
                const std::int64_t*  lwork,
                std::int64_t*        iwork,
                const std::int64_t*  liwork,
                std::int64_t*        info,
                std::size_t /*jobz_length*/,
                std::size_t /*uplo_length*/) noexcept
{
  try {
    *info = numerical::blas::syevd(*jobz, *uplo, *n, a, *lda, w, work, *lwork, iwork, *liwork);
    report_illegal("DSYEVD", -*info);
  }
  catch (...) {
    report_failure("DSYEVD");
    *info = -1;
  }
}

void dgesdd_64_(const char         * jobz,
                const std::int64_t*  m,
                const std::int64_t*  n,
                double         *     a,
                const std::int64_t*  lda,
                double         *     s,
                double         *     u,
                const std::int64_t*  ldu,
                double         *     vt,
                const std::int64_t*  ldvt,
                double         *     work,
                const std::int64_t*  lwork,
                std::int64_t*        iwork,
                std::int64_t*        info,
                std::size_t /*jobz_length*/) noexcept
{
  try {
    *info = numerical::blas::gesdd(*jobz, *m, *n, a, *lda, s, u, *ldu, vt, *ldvt, work, *lwork, iwork);
    report_illegal("DGESDD", -*info);
  }
  catch (...) {
    report_failure("DGESDD");
    *info = -1;
  }
}

void dgeqrf_64_(const std::int64_t* m,
                const std::int64_t* n,
                double         *    a,
                const std::int64_t* lda,
                double         *    tau,
                double         *    work,
                const std::int64_t* lwork,
                std::int64_t*       info) noexcept
{
  try {
    *info = numerical::blas::geqrf(*m, *n, a, *lda, tau, work, *lwork);
    report_illegal("DGEQRF", -*info);
  }
  catch (...) {
    report_failure("DGEQRF");
    *info = -1;
  }
}

void dorgqr_64_(const std::int64_t*    m,
                const std::int64_t*    n,
                const std::int64_t*    k,
                double         *       a,
                const std::int64_t*    lda,
                const double         * tau,
                double         *       work,
                const std::int64_t*    lwork,
                std::int64_t*          info) noexcept
{
  try {
    *info = numerical::blas::orgqr(*m, *n, *k, a, *lda, tau, work, *lwork);
    report_illegal("DORGQR", -*info);
  }
  catch (...) {
    report_failure("DORGQR");
    *info = -1;
  }
}

void dgelsd_64_(const std::int64_t*    m,
                const std::int64_t*    n,
                const std::int64_t*    nrhs,
                const double         * a,
                const std::int64_t*    lda,
                double         *       b,
                const std::int64_t*    ldb,
                double         *       s,
                const double         * rcond,
                std::int64_t*          rank,
                double         *       work,
                const std::int64_t*    lwork,
                std::int64_t*          iwork,
                std::int64_t*          info) noexcept
{
  try {
    *info = numerical::blas::gelsd(*m, *n, *nrhs, a, *lda, b, *ldb, s, *rcond, rank, work, *lwork, iwork);
    report_illegal("DGELSD", -*info);
  }
  catch (...) {
    report_failure("DGELSD");
    *info = -1;
  }
}

void dgeev_64_(const char         *   jobvl,
               const char         *   jobvr,
               const std::int64_t*    n,
               const double         * a,
               const std::int64_t*    lda,
               double         *       wr,
               double         *       wi,
               double         *       vl,
               const std::int64_t*    ldvl,
               double         *       vr,
               const std::int64_t*    ldvr,
               double         *       work,
               const std::int64_t*    lwork,
               std::int64_t*          info,
               std::size_t /*jobvl_length*/,
               std::size_t /*jobvr_length*/) noexcept
{
  try {
    *info = numerical::blas::geev(*jobvl, *jobvr, *n, a, *lda, wr, wi, vl, *ldvl, vr, *ldvr, work, *lwork);
    report_illegal("DGEEV ", -*info);
  }
  catch (...) {
    report_failure("DGEEV");
    *info = -1;
  }
}
}
