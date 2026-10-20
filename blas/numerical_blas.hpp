// The BLAS and LAPACK routines numerical_blas defines (numerical_blas.cpp), as a Fortran caller resolves them: every
// argument by address, a hidden length after the arguments for each character one, 32-bit integers under the
// reference names (LP64) and 64-bit ones under the names ending in `64_` (ILP64). Declared once, so each definition is
// checked against its declaration, and with default visibility, which the definitions take from it.
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

extern "C" {
//! \brief A complex result as gfortran returns it, by value: two floating-point numbers, as `float _Complex`.
struct numerical_blas_complex_float
{
  float re; //!< The real part.
  float im; //!< The imaginary part.
};

//! \brief A complex result as gfortran returns it, by value: two floating-point numbers, as `double _Complex`.
struct numerical_blas_complex_double
{
  double re; //!< The real part.
  double im; //!< The imaginary part.
};
void dgemm_(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*, std::size_t, std::size_t) noexcept;
void sgemm_(const char*, const char*, const int*, const int*, const int*, const float*, const float*, const int*, const float*, const int*, const float*, float*, const int*, std::size_t, std::size_t) noexcept;
void cgemm_(const char*, const char*, const int*, const int*, const int*, const std::complex<float>*, const std::complex<float>*, const int*, const std::complex<float>*, const int*, const std::complex<float>*, std::complex<float>*, const int*, std::size_t, std::size_t) noexcept;
void zgemm_(const char*, const char*, const int*, const int*, const int*, const std::complex<double>*, const std::complex<double>*, const int*, const std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*, std::size_t, std::size_t) noexcept;
void sgesv_(const int*,
            const int*,
            float*,
            const int*,
            int*,
            float*,
            const int*,
            int*) noexcept;
void sgetrf_(const int*,
             const int*,
             float*,
             const int*,
             int*,
             int*) noexcept;
void sgetrs_(const char*, const int*, const int*, const float*, const int*, const int*, float*, const int*, int*, std::size_t) noexcept;
void cgesv_(const int*,
            const int*,
            std::complex<float>*,
            const int*,
            int*,
            std::complex<float>*,
            const int*,
            int*) noexcept;
void cgetrf_(const int*,
             const int*,
             std::complex<float>*,
             const int*,
             int*,
             int*) noexcept;
void cgetrs_(const char*, const int*, const int*, const std::complex<float>*, const int*, const int*, std::complex<float>*, const int*, int*, std::size_t) noexcept;
void zgesv_(const int*,
            const int*,
            std::complex<double>*,
            const int*,
            int*,
            std::complex<double>*,
            const int*,
            int*) noexcept;
void zgetrf_(const int*,
             const int*,
             std::complex<double>*,
             const int*,
             int*,
             int*) noexcept;
void zgetrs_(const char*, const int*, const int*, const std::complex<double>*, const int*, const int*, std::complex<double>*, const int*, int*, std::size_t) noexcept;
void spotrf_(const char*, const int*, float*, const int*, int*, std::size_t) noexcept;
void spotrs_(const char*, const int*, const int*, const float*, const int*, float*, const int*, int*, std::size_t) noexcept;
void cpotrf_(const char*, const int*, std::complex<float>*, const int*, int*, std::size_t) noexcept;
void cpotrs_(const char*, const int*, const int*, const std::complex<float>*, const int*, std::complex<float>*, const int*, int*, std::size_t) noexcept;
void zpotrf_(const char*, const int*, std::complex<double>*, const int*, int*, std::size_t) noexcept;
void zpotrs_(const char*, const int*, const int*, const std::complex<double>*, const int*, std::complex<double>*, const int*, int*, std::size_t) noexcept;
double ddot_(const int*,
             const double*,
             const int*,
             const double*,
             const int*) noexcept;
void dgemv_(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*, std::size_t) noexcept;
void dsyrk_(const char*, const char*, const int*, const int*, const double*, const double*, const int*, const double*, double*, const int*, std::size_t, std::size_t) noexcept;
void dtrsm_(const char*, const char*, const char*, const char*, const int*, const int*, const double*, const double*, const int*, double*, const int*, std::size_t, std::size_t, std::size_t, std::size_t) noexcept;
void dgesv_(const int*,
            const int*,
            double*,
            const int*,
            int*,
            double*,
            const int*,
            int*) noexcept;
void dpotrf_(const char*, const int*, double*, const int*, int*, std::size_t) noexcept;
void dgetrf_(const int*,
             const int*,
             double*,
             const int*,
             int*,
             int*) noexcept;
void dgetrs_(const char*, const int*, const int*, const double*, const int*, const int*, double*, const int*, int*, std::size_t) noexcept;
void dpotrs_(const char*, const int*, const int*, const double*, const int*, double*, const int*, int*, std::size_t) noexcept;
void dsyevd_(const char*, const char*, const int*, double*, const int*, double*, double*, const int*, int*, const int*, int*, std::size_t, std::size_t) noexcept;
void dgesdd_(const char*, const int*, const int*, double*, const int*, double*, double*, const int*, double*, const int*, double*, const int*, int*, int*, std::size_t) noexcept;
void dgeqrf_(const int*,
             const int*,
             double*,
             const int*,
             double*,
             double*,
             const int*,
             int*) noexcept;
void dorgqr_(const int*,
             const int*,
             const int*,
             double*,
             const int*,
             const double*,
             double*,
             const int*,
             int*) noexcept;
void dgelsd_(const int*,
             const int*,
             const int*,
             const double*,
             const int*,
             double*,
             const int*,
             double*,
             const double*,
             int*,
             double*,
             const int*,
             int*,
             int*) noexcept;
void dgeev_(const char*, const char*, const int*, const double*, const int*, double*, double*, double*, const int*, double*, const int*, double*, const int*, int*, std::size_t, std::size_t) noexcept;
float sdot_(const int*,
            const float*,
            const int*,
            const float*,
            const int*) noexcept;
numerical_blas_complex_float cdotu_(const int*,
                                    const std::complex<float>*,
                                    const int*,
                                    const std::complex<float>*,
                                    const int*) noexcept;
numerical_blas_complex_float cdotc_(const int*,
                                    const std::complex<float>*,
                                    const int*,
                                    const std::complex<float>*,
                                    const int*) noexcept;
numerical_blas_complex_double zdotu_(const int*,
                                     const std::complex<double>*,
                                     const int*,
                                     const std::complex<double>*,
                                     const int*) noexcept;
numerical_blas_complex_double zdotc_(const int*,
                                     const std::complex<double>*,
                                     const int*,
                                     const std::complex<double>*,
                                     const int*) noexcept;

void dgemm_64_(const char*, const char*, const std::int64_t*, const std::int64_t*, const std::int64_t*, const double*, const double*, const std::int64_t*, const double*, const std::int64_t*, const double*, double*, const std::int64_t*, std::size_t, std::size_t) noexcept;
void sgemm_64_(const char*, const char*, const std::int64_t*, const std::int64_t*, const std::int64_t*, const float*, const float*, const std::int64_t*, const float*, const std::int64_t*, const float*, float*, const std::int64_t*, std::size_t, std::size_t) noexcept;
void cgemm_64_(const char*, const char*, const std::int64_t*, const std::int64_t*, const std::int64_t*, const std::complex<float>*, const std::complex<float>*, const std::int64_t*, const std::complex<float>*, const std::int64_t*, const std::complex<float>*, std::complex<float>*, const std::int64_t*, std::size_t, std::size_t) noexcept;
void zgemm_64_(const char*, const char*, const std::int64_t*, const std::int64_t*, const std::int64_t*, const std::complex<double>*, const std::complex<double>*, const std::int64_t*, const std::complex<double>*, const std::int64_t*, const std::complex<double>*, std::complex<double>*, const std::int64_t*, std::size_t, std::size_t) noexcept;
void sgesv_64_(const std::int64_t*,
               const std::int64_t*,
               float*,
               const std::int64_t*,
               std::int64_t*,
               float*,
               const std::int64_t*,
               std::int64_t*) noexcept;
void sgetrf_64_(const std::int64_t*,
                const std::int64_t*,
                float*,
                const std::int64_t*,
                std::int64_t*,
                std::int64_t*) noexcept;
void sgetrs_64_(const char*, const std::int64_t*, const std::int64_t*, const float*, const std::int64_t*, const std::int64_t*, float*, const std::int64_t*, std::int64_t*, std::size_t) noexcept;
void cgesv_64_(const std::int64_t*,
               const std::int64_t*,
               std::complex<float>*,
               const std::int64_t*,
               std::int64_t*,
               std::complex<float>*,
               const std::int64_t*,
               std::int64_t*) noexcept;
void cgetrf_64_(const std::int64_t*,
                const std::int64_t*,
                std::complex<float>*,
                const std::int64_t*,
                std::int64_t*,
                std::int64_t*) noexcept;
void cgetrs_64_(const char*, const std::int64_t*, const std::int64_t*, const std::complex<float>*, const std::int64_t*, const std::int64_t*, std::complex<float>*, const std::int64_t*, std::int64_t*, std::size_t) noexcept;
void zgesv_64_(const std::int64_t*,
               const std::int64_t*,
               std::complex<double>*,
               const std::int64_t*,
               std::int64_t*,
               std::complex<double>*,
               const std::int64_t*,
               std::int64_t*) noexcept;
void zgetrf_64_(const std::int64_t*,
                const std::int64_t*,
                std::complex<double>*,
                const std::int64_t*,
                std::int64_t*,
                std::int64_t*) noexcept;
void zgetrs_64_(const char*, const std::int64_t*, const std::int64_t*, const std::complex<double>*, const std::int64_t*, const std::int64_t*, std::complex<double>*, const std::int64_t*, std::int64_t*, std::size_t) noexcept;
void spotrf_64_(const char*, const std::int64_t*, float*, const std::int64_t*, std::int64_t*, std::size_t) noexcept;
void spotrs_64_(const char*, const std::int64_t*, const std::int64_t*, const float*, const std::int64_t*, float*, const std::int64_t*, std::int64_t*, std::size_t) noexcept;
void cpotrf_64_(const char*, const std::int64_t*, std::complex<float>*, const std::int64_t*, std::int64_t*, std::size_t) noexcept;
void cpotrs_64_(const char*, const std::int64_t*, const std::int64_t*, const std::complex<float>*, const std::int64_t*, std::complex<float>*, const std::int64_t*, std::int64_t*, std::size_t) noexcept;
void zpotrf_64_(const char*, const std::int64_t*, std::complex<double>*, const std::int64_t*, std::int64_t*, std::size_t) noexcept;
void zpotrs_64_(const char*, const std::int64_t*, const std::int64_t*, const std::complex<double>*, const std::int64_t*, std::complex<double>*, const std::int64_t*, std::int64_t*, std::size_t) noexcept;
double ddot_64_(const std::int64_t*,
                const double*,
                const std::int64_t*,
                const double*,
                const std::int64_t*) noexcept;
void dgemv_64_(const char*, const std::int64_t*, const std::int64_t*, const double*, const double*, const std::int64_t*, const double*, const std::int64_t*, const double*, double*, const std::int64_t*, std::size_t) noexcept;
void dsyrk_64_(const char*, const char*, const std::int64_t*, const std::int64_t*, const double*, const double*, const std::int64_t*, const double*, double*, const std::int64_t*, std::size_t, std::size_t) noexcept;
void dtrsm_64_(const char*, const char*, const char*, const char*, const std::int64_t*, const std::int64_t*, const double*, const double*, const std::int64_t*, double*, const std::int64_t*, std::size_t, std::size_t, std::size_t, std::size_t) noexcept;
void dgesv_64_(const std::int64_t*,
               const std::int64_t*,
               double*,
               const std::int64_t*,
               std::int64_t*,
               double*,
               const std::int64_t*,
               std::int64_t*) noexcept;
void dpotrf_64_(const char*, const std::int64_t*, double*, const std::int64_t*, std::int64_t*, std::size_t) noexcept;
void dgetrf_64_(const std::int64_t*,
                const std::int64_t*,
                double*,
                const std::int64_t*,
                std::int64_t*,
                std::int64_t*) noexcept;
void dgetrs_64_(const char*, const std::int64_t*, const std::int64_t*, const double*, const std::int64_t*, const std::int64_t*, double*, const std::int64_t*, std::int64_t*, std::size_t) noexcept;
void dpotrs_64_(const char*, const std::int64_t*, const std::int64_t*, const double*, const std::int64_t*, double*, const std::int64_t*, std::int64_t*, std::size_t) noexcept;
void dsyevd_64_(const char*, const char*, const std::int64_t*, double*, const std::int64_t*, double*, double*, const std::int64_t*, std::int64_t*, const std::int64_t*, std::int64_t*, std::size_t, std::size_t) noexcept;
void dgesdd_64_(const char*, const std::int64_t*, const std::int64_t*, double*, const std::int64_t*, double*, double*, const std::int64_t*, double*, const std::int64_t*, double*, const std::int64_t*, std::int64_t*, std::int64_t*, std::size_t) noexcept;
void dgeqrf_64_(const std::int64_t*,
                const std::int64_t*,
                double*,
                const std::int64_t*,
                double*,
                double*,
                const std::int64_t*,
                std::int64_t*) noexcept;
void dorgqr_64_(const std::int64_t*,
                const std::int64_t*,
                const std::int64_t*,
                double*,
                const std::int64_t*,
                const double*,
                double*,
                const std::int64_t*,
                std::int64_t*) noexcept;
void dgelsd_64_(const std::int64_t*,
                const std::int64_t*,
                const std::int64_t*,
                const double*,
                const std::int64_t*,
                double*,
                const std::int64_t*,
                double*,
                const double*,
                std::int64_t*,
                double*,
                const std::int64_t*,
                std::int64_t*,
                std::int64_t*) noexcept;
void dgeev_64_(const char*, const char*, const std::int64_t*, const double*, const std::int64_t*, double*, double*, double*, const std::int64_t*, double*, const std::int64_t*, double*, const std::int64_t*, std::int64_t*, std::size_t, std::size_t) noexcept;
float sdot_64_(const std::int64_t*,
               const float*,
               const std::int64_t*,
               const float*,
               const std::int64_t*) noexcept;
numerical_blas_complex_float cdotu_64_(const std::int64_t*,
                                       const std::complex<float>*,
                                       const std::int64_t*,
                                       const std::complex<float>*,
                                       const std::int64_t*) noexcept;
numerical_blas_complex_float cdotc_64_(const std::int64_t*,
                                       const std::complex<float>*,
                                       const std::int64_t*,
                                       const std::complex<float>*,
                                       const std::int64_t*) noexcept;
numerical_blas_complex_double zdotu_64_(const std::int64_t*,
                                        const std::complex<double>*,
                                        const std::int64_t*,
                                        const std::complex<double>*,
                                        const std::int64_t*) noexcept;
numerical_blas_complex_double zdotc_64_(const std::int64_t*,
                                        const std::complex<double>*,
                                        const std::int64_t*,
                                        const std::complex<double>*,
                                        const std::int64_t*) noexcept;
}

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
