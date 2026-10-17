// The Fortran calling convention of the BLAS and LAPACK routines NumericAL stands in for (ddot, dgeev, dgelsd, dgemm,
// dgemv, dgeqrf, dgesdd, dgesv, dgetrf, dgetrs, dorgqr, dpotrf, dpotrs, dsyevd, dsyrk): every argument by address, 32-bit integers (LP64), and a hidden
// length after the arguments for each character one, as gfortran passes it. The semantics are numerical/blas.hpp's. An
// illegal argument is reported as the reference XERBLA reports it, on the standard error, and the call returns (as
// OpenBLAS's does) rather than stopping the program; no C++ exception leaves these functions.
#include <cstddef>
#include <cstdio>
#include <exception>

#include "numerical/blas.hpp"

namespace {
  //! \brief The reference XERBLA's message for the illegal argument \p position of the routine \p name.
  void report_illegal(const char* name,
                      int         position) noexcept
  {
    if (position > 0) {
      (void)std::fprintf(stderr, " ** On entry to %s parameter number %2d had an illegal value\n", name, position); // NOLINT(cppcoreguidelines-pro-type-vararg): the reference's message
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
