// The BLAS/LAPACK stand-ins (numerical/blas.hpp and the Fortran shim of blas/numerical_blas.cpp): column-major
// semantics as the reference routines define them, checked against plain loops and against the factorisations'
// defining properties; every illegal argument reported by its position.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

extern "C" {
void dgemm_(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*, const double*,
            const int*, const double*, double*, const int*, std::size_t, std::size_t) noexcept;
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
void dsyevd_(const char*, const char*, const int*, double*, const int*, double*, double*, const int*, int*, const int*, int*, std::size_t,
             std::size_t) noexcept;
void dgesdd_(const char*, const int*, const int*, double*, const int*, double*, double*, const int*, double*, const int*, double*, const int*,
             int*, int*, std::size_t) noexcept;
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
void dgeev_(const char*, const char*, const int*, const double*, const int*, double*, double*, double*, const int*, double*, const int*,
            double*, const int*, int*, std::size_t, std::size_t) noexcept;
void dsyrk_(const char*, const char*, const int*, const int*, const double*, const double*, const int*, const double*, double*, const int*,
            std::size_t, std::size_t) noexcept;
void dgemv_(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*,
            const int*, std::size_t) noexcept;
double ddot_(const int*,
             const double*,
             const int*,
             const double*,
             const int*) noexcept;
void dtrsm_(const char*, const char*, const char*, const char*, const int*, const int*, const double*, const double*, const int*, double*,
            const int*, std::size_t, std::size_t, std::size_t, std::size_t) noexcept;
void sgemm_(const char*, const char*, const int*, const int*, const int*, const float*, const float*, const int*, const float*, const int*,
            const float*, float*, const int*, std::size_t, std::size_t) noexcept;
void zgemm_(const char*, const char*, const int*, const int*, const int*, const std::complex<double>*, const std::complex<double>*,
            const int*, const std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*,
            std::size_t, std::size_t) noexcept;
void cgemm_(const char*, const char*, const int*, const int*, const int*, const std::complex<float>*, const std::complex<float>*,
            const int*, const std::complex<float>*, const int*, const std::complex<float>*, std::complex<float>*, const int*,
            std::size_t, std::size_t) noexcept;
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
}

namespace {
  // Deterministic entries in [-1, 1), of full rank (a sine of a linear index would give a matrix of rank two).
  std::vector<double> entries(std::size_t count,
                              std::size_t seed)
  {
    std::vector<double> out(count);
    std::uint64_t       state {(seed * 0x9E3779B97F4A7C15ULL) + 1};
    for (double& x : out) {
      state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
      x     = (static_cast<double>(state >> 11U) * 0x1.0p-52) - 1.0;
    }
    return out;
  }

  // Element (i, j) of op(X), X column-major with leading dimension ld.
  double element(const std::vector<double>& x,
                 std::size_t                ld,
                 char                       op,
                 std::size_t                i,
                 std::size_t                j)
  {
    return (op == 'N' || op == 'n') ? x[i + (j * ld)] : x[j + (i * ld)];
  }

  // Whether gemm(transa, transb) of m × k by k × n, with leading dimensions padded past the rows, matches
  // α op(A) op(B) + β C by a plain loop.
  bool gemm_matches(char        transa,
                    char        transb,
                    std::size_t m,
                    std::size_t n,
                    std::size_t k,
                    double      alpha,
                    double      beta)
  {
    const bool                ta  {transa != 'N' && transa != 'n'};
    const bool                tb  {transb != 'N' && transb != 'n'};
    const std::size_t         lda {(ta ? k : m) + 2};
    const std::size_t         ldb {(tb ? n : k) + 1};
    const std::size_t         ldc {m + 3};
    const std::vector<double> a   {entries(lda * (ta ? m : k), 1)};
    const std::vector<double> b   {entries(ldb * (tb ? k : n), 2)};
    std::vector<double>       c   {entries(ldc * n, 3)};
    if (beta == 0.0) {
      std::ranges::fill(c, std::numeric_limits<double>::quiet_NaN()); // never read
    }
    std::vector<double> want {c};
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t i {0}; i < m; ++i) {
        double sum {0.0};
        for (std::size_t p {0}; p < k; ++p) {
          sum += element(a, lda, transa, i, p) * element(b, ldb, transb, p, j);
        }
        want[i + (j * ldc)] = (alpha * sum) + (beta == 0.0 ? 0.0 : beta * c[i + (j * ldc)]);
      }
    }
    const int code {numerical::blas::gemm(transa, transb, static_cast<std::ptrdiff_t>(m), static_cast<std::ptrdiff_t>(n), static_cast<std::ptrdiff_t>(k),
                                          alpha, a.data(), static_cast<std::ptrdiff_t>(lda), b.data(), static_cast<std::ptrdiff_t>(ldb), beta, c.data(),
                                          static_cast<std::ptrdiff_t>(ldc))};
    bool same {code == 0};
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t i {0}; i < ldc; ++i) {
        const double got {c[i + (j * ldc)]};
        if (i < m) {
          same = same && std::abs(got - want[i + (j * ldc)]) <= 1e-12 * (1.0 + static_cast<double>(k));
        }
        else { // past the rows, untouched
          same = same && (std::isnan(got) ? std::isnan(want[i + (j * ldc)]) : got == want[i + (j * ldc)]);
        }
      }
    }
    return same;
  }
} // namespace

// Every operation pair, α and β of 0, 1 and other values, shapes past the kernel's tiles and below its thresholds.
TEST(gemm_computes_alpha_op_a_op_b_plus_beta_c)
{
  for (const char ta : {'N', 'T', 'c'}) {
    for (const char tb : {'n', 't', 'C'}) {
      EXPECT(gemm_matches(ta, tb, 7, 5, 3, 1.0, 0.0));
      EXPECT(gemm_matches(ta, tb, 37, 29, 41, -0.5, 2.0));
      EXPECT(gemm_matches(ta, tb, 70, 66, 90, 1.0, 1.0));
    }
  }
  EXPECT(gemm_matches('N', 'N', 9, 8, 7, 0.0, 0.5)); // α zero: β C alone
  EXPECT(gemm_matches('N', 'N', 9, 8, 0, 2.0, 3.0)); // no depth: β C alone
}

TEST(gemm_reports_its_first_illegal_argument)
{
  std::vector<double> x(64, 1.0);
  const auto          call = [&](char ta, char tb, std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k, std::ptrdiff_t lda, std::ptrdiff_t ldb,
                                 std::ptrdiff_t ldc) { return numerical::blas::gemm(ta, tb, m, n, k, 1.0, x.data(), lda, x.data(), ldb, 0.0, x.data(), ldc); };
  EXPECT_EQ(call('X', 'N', 2, 2, 2, 2, 2, 2), 1);
  EXPECT_EQ(call('N', '?', 2, 2, 2, 2, 2, 2), 2);
  EXPECT_EQ(call('N', 'N', -1, 2, 2, 2, 2, 2), 3);
  EXPECT_EQ(call('N', 'N', 2, -1, 2, 2, 2, 2), 4);
  EXPECT_EQ(call('N', 'N', 2, 2, -1, 2, 2, 2), 5);
  EXPECT_EQ(call('N', 'N', 3, 2, 2, 2, 2, 3), 8);  // lda under m
  EXPECT_EQ(call('T', 'N', 3, 2, 4, 3, 4, 3), 8);  // lda under k, A transposed
  EXPECT_EQ(call('N', 'N', 2, 2, 3, 2, 2, 2), 10); // ldb under k
  EXPECT_EQ(call('N', 'T', 2, 3, 2, 2, 2, 2), 10); // ldb under n, B transposed
  EXPECT_EQ(call('N', 'N', 3, 2, 2, 3, 2, 2), 13);
  EXPECT_EQ(call('N', 'N', 0, 2, 2, 1, 2, 1), 0);  // an empty C: nothing to do
  EXPECT_EQ(call('N', 'N', 2, 0, 2, 2, 2, 2), 0);
}

namespace {
  // Whether gesv of order n, nrhs right-hand sides, padded leading dimensions, leaves P A = L U (the interchanges of
  // ipiv applied in order) and A X = B, each within the rounding of the order.
  bool gesv_solves(std::size_t n,
                   std::size_t nrhs)
  {
    const std::size_t         lda {n + 2};
    const std::size_t         ldb {n + 1};
    const std::vector<double> a   {entries(lda * n, 5)};
    const std::vector<double> b   {entries(ldb * nrhs, 6)};
    std::vector<double>       lu  {a};
    std::vector<double>       x   {b};
    std::vector<int>          ipiv(n);
    if (numerical::blas::gesv(static_cast<std::ptrdiff_t>(n), static_cast<std::ptrdiff_t>(nrhs), lu.data(), static_cast<std::ptrdiff_t>(lda), ipiv.data(),
                              x.data(), static_cast<std::ptrdiff_t>(ldb)) != 0) {
      return false;
    }
    std::vector<double> pa {a}; // P A: the interchanges applied to A's rows, in order
    for (std::size_t i {0}; i < n; ++i) {
      const auto other {static_cast<std::size_t>(ipiv[i] - 1)};
      for (std::size_t j {0}; j < n; ++j) {
        std::swap(pa[i + (j * lda)], pa[other + (j * lda)]);
      }
    }
    bool same {true};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        double sum {0.0};
        for (std::size_t p {0}; p <= std::min(i, j); ++p) {
          sum += (p == i ? 1.0 : lu[i + (p * lda)]) * lu[p + (j * lda)];
        }
        same = same && std::abs(sum - pa[i + (j * lda)]) <= 1e-12 * static_cast<double>(n);
      }
      for (std::size_t c {0}; c < nrhs; ++c) {
        double sum {0.0};
        for (std::size_t j {0}; j < n; ++j) {
          sum += a[i + (j * lda)] * x[j + (c * ldb)];
        }
        same = same && std::abs(sum - b[i + (c * ldb)]) <= 1e-9 * static_cast<double>(n);
      }
    }
    return same;
  }
} // namespace

TEST(gesv_factorises_as_dgetrf_and_solves)
{
  EXPECT(gesv_solves(1, 1));
  EXPECT(gesv_solves(7, 3));
  EXPECT(gesv_solves(150, 4)); // blocked, with look-ahead
  EXPECT(gesv_solves(9, 0));   // factors only
}

TEST(gesv_reports_a_zero_pivot_an_illegal_argument_and_a_nan)
{
  std::vector<double> a {1.0, 2.0, 0.0, 0.0}; // second column zero: U(2, 2) = 0
  std::vector<double> b {5.0, 6.0};
  std::vector<int>    ipiv(2);
  EXPECT_EQ(numerical::blas::gesv(2, 1, a.data(), 2, ipiv.data(), b.data(), 2), 2);
  EXPECT_EQ(b[0], 5.0); // not solved
  std::vector<double> nan {1.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0};
  EXPECT_EQ(numerical::blas::gesv(2, 1, nan.data(), 2, ipiv.data(), b.data(), 2), -3);
  EXPECT_EQ(numerical::blas::gesv(-1, 1, a.data(), 2, ipiv.data(), b.data(), 2), -1);
  EXPECT_EQ(numerical::blas::gesv(2, -1, a.data(), 2, ipiv.data(), b.data(), 2), -2);
  EXPECT_EQ(numerical::blas::gesv(2, 1, a.data(), 1, ipiv.data(), b.data(), 2), -4);
  EXPECT_EQ(numerical::blas::gesv(2, 1, a.data(), 2, ipiv.data(), b.data(), 1), -7);
  EXPECT_EQ(numerical::blas::gesv(0, 1, a.data(), 1, ipiv.data(), b.data(), 1), 0);
}

namespace {
  // Whether potrf(uplo) of order n gives the factor of a symmetric positive-definite A in its triangle, the other
  // triangle untouched.
  bool potrf_factorises(char        uplo,
                        std::size_t n)
  {
    const std::size_t   lda {n + 2};
    std::vector<double> g   {entries(n * n, 7)};
    std::vector<double> a(lda * n, 9.0); // 9 marks what must stay
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        double sum {i == j ? static_cast<double>(n) : 0.0};
        for (std::size_t p {0}; p < n; ++p) {
          sum += g[i + (p * n)] * g[j + (p * n)];
        }
        const bool kept {(uplo == 'L') ? i >= j : i <= j};
        if (kept) {
          a[i + (j * lda)] = sum;
        }
      }
    }
    const std::vector<double> original {a};
    if (numerical::blas::potrf(uplo, static_cast<std::ptrdiff_t>(n), a.data(), static_cast<std::ptrdiff_t>(lda)) != 0) {
      return false;
    }
    bool same {true};
    for (std::size_t i {0}; i < lda; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        const bool kept {i < n && ((uplo == 'L') ? i >= j : i <= j)};
        if (!kept) {
          same = same && a[i + (j * lda)] == original[i + (j * lda)];
          continue;
        }
        const auto factor = [&](std::size_t r, std::size_t q) { // L(r, q), stored as L or as U = Lᵀ
                              if (q > r) {
                                return 0.0;
                              }
                              return (uplo == 'L') ? a[r + (q * lda)] : a[q + (r * lda)];
                            };
        double sum {0.0}; // (L Lᵀ)(i, j) from the factor written
        for (std::size_t q {0}; q < n; ++q) {
          sum += factor(i, q) * factor(j, q);
        }
        same = same && std::abs(sum - original[i + (j * lda)]) <= 1e-10 * static_cast<double>(n * n);
      }
    }
    return same;
  }
} // namespace

TEST(potrf_factorises_either_triangle)
{
  EXPECT(potrf_factorises('L', 1));
  EXPECT(potrf_factorises('L', 9));
  EXPECT(potrf_factorises('U', 9));
  EXPECT(potrf_factorises('L', 140)); // in panels
  EXPECT(potrf_factorises('U', 140));
}

TEST(potrf_reports_the_failing_minor_an_illegal_argument_and_a_nan)
{
  std::vector<double> a   {4.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0}; // the minor of order 2 is singular
  EXPECT_EQ(numerical::blas::potrf('L', 3, a.data(), 3), 2);
  std::vector<double> nan {std::numeric_limits<double>::infinity()};
  EXPECT_EQ(numerical::blas::potrf('U', 1, nan.data(), 1), -3);
  EXPECT_EQ(numerical::blas::potrf('X', 1, a.data(), 1), -1);
  EXPECT_EQ(numerical::blas::potrf('L', -1, a.data(), 1), -2);
  EXPECT_EQ(numerical::blas::potrf('l', 2, a.data(), 1), -4);
  EXPECT_EQ(numerical::blas::potrf('u', 0, a.data(), 1), 0);
}

// The Fortran shim: the same answers through every argument passed by address, its illegal arguments reported on the
// standard error and in INFO.
TEST(the_fortran_routines_answer_as_the_cpp_ones)
{
  const int           two  {2};
  const double        one  {1.0};
  const double        zero {0.0};
  std::vector<double> a    {1.0, 3.0, 2.0, 4.0};                                                         // [[1, 2], [3, 4]], column-major
  std::vector<double> c(4, 0.0);
  dgemm_("N", "N", &two, &two, &two, &one, a.data(), &two, a.data(), &two, &zero, c.data(), &two, 1, 1);
  EXPECT_EQ(c[0], 7.0);                                                                                  // [[7, 10], [15, 22]]
  EXPECT_EQ(c[2], 10.0);
  const int bad            {-1};
  dgemm_("N", "N", &bad, &two, &two, &one, a.data(), &two, a.data(), &two, &zero, c.data(), &two, 1, 1); // reported
  std::vector<double> b    {5.0, 6.0};
  std::vector<int>    ipiv(2);
  const int           rhs  {1};
  int                 info {7};
  dgesv_(&two, &rhs, a.data(), &two, ipiv.data(), b.data(), &two, &info);
  EXPECT_EQ(info, 0);
  EXPECT(std::abs(b[0] + 4.0) < 1e-12 && std::abs(b[1] - 4.5) < 1e-12); // x = (-4, 4.5)
  dgesv_(&bad, &rhs, a.data(), &two, ipiv.data(), b.data(), &two, &info);
  EXPECT_EQ(info, -1);
  std::vector<double> s {4.0, 2.0, 2.0, 3.0};
  dpotrf_("L", &two, s.data(), &two, &info, 1);
  EXPECT_EQ(info, 0);
  EXPECT_EQ(s[0], 2.0);
  dpotrf_("Q", &two, s.data(), &two, &info, 1);
  EXPECT_EQ(info, -1);
}

namespace {
  // Whether getrf of the m × n A leaves P A = L U (L m × min(m, n) unit lower, U min(m, n) × n upper), the
  // interchanges of ipiv applied to A's rows in order.
  bool getrf_factorises(std::size_t m,
                        std::size_t n)
  {
    const std::size_t         lda   {m + 2};
    const std::size_t         steps {std::min(m, n)};
    const std::vector<double> a     {entries(lda * n, 11)};
    std::vector<double>       lu    {a};
    std::vector<int>          ipiv(steps);
    if (numerical::blas::getrf(static_cast<std::ptrdiff_t>(m), static_cast<std::ptrdiff_t>(n), lu.data(), static_cast<std::ptrdiff_t>(lda),
                               ipiv.data()) != 0) {
      return false;
    }
    std::vector<double> pa {a};
    for (std::size_t i {0}; i < steps; ++i) {
      const auto other {static_cast<std::size_t>(ipiv[i] - 1)};
      for (std::size_t j {0}; j < n; ++j) {
        std::swap(pa[i + (j * lda)], pa[other + (j * lda)]);
      }
    }
    bool same {true};
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        double sum {0.0};
        for (std::size_t p {0}; p <= std::min({i, j, steps - 1}); ++p) {
          sum += (p == i ? 1.0 : lu[i + (p * lda)]) * lu[p + (j * lda)];
        }
        same = same && std::abs(sum - pa[i + (j * lda)]) <= 1e-12 * static_cast<double>(m + n);
      }
    }
    return same;
  }

  // Whether getrs(trans) with getrf's factors of an order-n A solves A X = B (Aᵀ X = B), nrhs columns.
  bool getrs_solves(char        trans,
                    std::size_t n,
                    std::size_t nrhs)
  {
    const std::size_t         lda  {n + 1};
    const std::size_t         ldb  {n + 3};
    const std::vector<double> a    {entries(lda * n, 12)};
    const std::vector<double> b    {entries(ldb * nrhs, 13)};
    std::vector<double>       lu   {a};
    std::vector<double>       x    {b};
    std::vector<int>          ipiv(n);
    const auto                size {static_cast<std::ptrdiff_t>(n)};
    if (numerical::blas::getrf(size, size, lu.data(), static_cast<std::ptrdiff_t>(lda), ipiv.data()) != 0
        || numerical::blas::getrs(trans, size, static_cast<std::ptrdiff_t>(nrhs), lu.data(), static_cast<std::ptrdiff_t>(lda), ipiv.data(), x.data(),
                                  static_cast<std::ptrdiff_t>(ldb)) != 0) {
      return false;
    }
    const bool transposed {trans != 'N'};
    bool       same       {true};
    for (std::size_t c {0}; c < nrhs; ++c) {
      for (std::size_t i {0}; i < n; ++i) {
        double sum {0.0};
        for (std::size_t j {0}; j < n; ++j) {
          sum += (transposed ? a[j + (i * lda)] : a[i + (j * lda)]) * x[j + (c * ldb)];
        }
        same = same && std::abs(sum - b[i + (c * ldb)]) <= 1e-9 * static_cast<double>(n);
      }
    }
    return same;
  }

  // Whether potrs(uplo) with potrf's factor of an order-n symmetric positive-definite A solves A X = B.
  bool potrs_solves(char        uplo,
                    std::size_t n)
  {
    const std::size_t         lda {n};
    const std::vector<double> g   {entries(n * n, 14)};
    std::vector<double>       a(n * n);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        double sum {i == j ? static_cast<double>(n) : 0.0};
        for (std::size_t p {0}; p < n; ++p) {
          sum += g[i + (p * n)] * g[j + (p * n)];
        }
        a[i + (j * lda)] = sum;
      }
    }
    const std::vector<double> b      {entries(n * 2, 15)};
    std::vector<double>       factor {a};
    std::vector<double>       x      {b};
    const auto                size   {static_cast<std::ptrdiff_t>(n)};
    if (numerical::blas::potrf(uplo, size, factor.data(), size) != 0 || numerical::blas::potrs(uplo, size, 2, factor.data(), size, x.data(), size) != 0) {
      return false;
    }
    bool same {true};
    for (std::size_t c {0}; c < 2; ++c) {
      for (std::size_t i {0}; i < n; ++i) {
        double sum {0.0};
        for (std::size_t j {0}; j < n; ++j) {
          sum += a[i + (j * lda)] * x[j + (c * n)];
        }
        same = same && std::abs(sum - b[i + (c * n)]) <= 1e-9 * static_cast<double>(n);
      }
    }
    return same;
  }
} // namespace

TEST(getrf_factorises_square_tall_and_wide_matrices)
{
  EXPECT(getrf_factorises(9, 9));
  EXPECT(getrf_factorises(130, 130)); // blocked
  EXPECT(getrf_factorises(12, 7));    // tall
  EXPECT(getrf_factorises(7, 12));    // wide
}

TEST(getrf_reports_zero_pivots_illegal_arguments_and_a_nan)
{
  std::vector<int>    ipiv(3);
  std::vector<double> square {1.0, 2.0, 0.0, 0.0};           // second column zero
  EXPECT_EQ(numerical::blas::getrf(2, 2, square.data(), 2, ipiv.data()), 2);
  std::vector<double> tall   {0.0, 0.0, 0.0, 1.0, 2.0, 3.0}; // 3 × 2, first column zero
  EXPECT_EQ(numerical::blas::getrf(3, 2, tall.data(), 3, ipiv.data()), 1);
  std::vector<double> nan    {std::numeric_limits<double>::quiet_NaN(), 1.0};
  EXPECT_EQ(numerical::blas::getrf(2, 1, nan.data(), 2, ipiv.data()), -3);
  EXPECT_EQ(numerical::blas::getrf(-1, 2, square.data(), 2, ipiv.data()), -1);
  EXPECT_EQ(numerical::blas::getrf(2, -1, square.data(), 2, ipiv.data()), -2);
  EXPECT_EQ(numerical::blas::getrf(2, 2, square.data(), 1, ipiv.data()), -4);
  EXPECT_EQ(numerical::blas::getrf(0, 2, square.data(), 1, ipiv.data()), 0);
  EXPECT_EQ(numerical::blas::getrf(2, 0, square.data(), 2, ipiv.data()), 0);
  std::vector<double> two_zero {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // 3 × 2, both columns zero: INFO the first
  EXPECT_EQ(numerical::blas::getrf(3, 2, two_zero.data(), 3, ipiv.data()), 1);
}

TEST(getrs_and_potrs_solve_from_the_factors)
{
  EXPECT(getrs_solves('N', 8, 3));
  EXPECT(getrs_solves('T', 8, 3));
  EXPECT(getrs_solves('c', 70, 2));
  EXPECT(potrs_solves('L', 9));
  EXPECT(potrs_solves('U', 9));
  EXPECT(potrs_solves('l', 4));
  std::vector<double> a(4, 1.0);
  std::vector<int>    ipiv {1, 2};
  EXPECT_EQ(numerical::blas::getrs('X', 2, 1, a.data(), 2, ipiv.data(), a.data(), 2), -1);
  EXPECT_EQ(numerical::blas::getrs('N', -1, 1, a.data(), 2, ipiv.data(), a.data(), 2), -2);
  EXPECT_EQ(numerical::blas::getrs('N', 2, -1, a.data(), 2, ipiv.data(), a.data(), 2), -3);
  EXPECT_EQ(numerical::blas::getrs('N', 2, 1, a.data(), 1, ipiv.data(), a.data(), 2), -5);
  EXPECT_EQ(numerical::blas::getrs('N', 2, 1, a.data(), 2, ipiv.data(), a.data(), 1), -8);
  EXPECT_EQ(numerical::blas::potrs('X', 2, 1, a.data(), 2, a.data(), 2), -1);
  EXPECT_EQ(numerical::blas::potrs('L', -1, 1, a.data(), 2, a.data(), 2), -2);
  EXPECT_EQ(numerical::blas::potrs('L', 2, -1, a.data(), 2, a.data(), 2), -3);
  EXPECT_EQ(numerical::blas::potrs('U', 2, 1, a.data(), 1, a.data(), 2), -5);
  EXPECT_EQ(numerical::blas::potrs('u', 2, 1, a.data(), 2, a.data(), 1), -7);
}

TEST(the_fortran_solvers_answer_as_the_cpp_ones)
{
  const int           two  {2};
  const int           one  {1};
  std::vector<double> a    {4.0, 2.0, 2.0, 3.0};
  std::vector<int>    ipiv(2);
  int                 info {7};
  dgetrf_(&two, &two, a.data(), &two, ipiv.data(), &info);
  EXPECT_EQ(info, 0);
  std::vector<double> b {2.0, 1.0}; // [[4, 2], [2, 3]] x = (2, 1): x = (0.5, 0)
  dgetrs_("N", &two, &one, a.data(), &two, ipiv.data(), b.data(), &two, &info, 1);
  EXPECT(info == 0 && std::abs(b[0] - 0.5) < 1e-15 && std::abs(b[1]) < 1e-15);
  std::vector<double> s {4.0, 2.0, 2.0, 3.0};
  dpotrf_("L", &two, s.data(), &two, &info, 1);
  std::vector<double> c {2.0, 1.0};
  dpotrs_("L", &two, &one, s.data(), &two, c.data(), &two, &info, 1);
  EXPECT(info == 0 && std::abs(c[0] - 0.5) < 1e-15 && std::abs(c[1]) < 1e-15);
  const int bad {-1};
  dgetrf_(&bad, &two, a.data(), &two, ipiv.data(), &info);
  EXPECT_EQ(info, -1);
  dgetrs_("N", &bad, &one, a.data(), &two, ipiv.data(), b.data(), &two, &info, 1);
  EXPECT_EQ(info, -2);
  dpotrs_("L", &bad, &one, s.data(), &two, c.data(), &two, &info, 1);
  EXPECT_EQ(info, -2);
}

namespace {
  // Whether syevd(jobz, uplo) of a symmetric matrix of order n, the other triangle NaN (never read) and the leading
  // dimension padded, gives ascending eigenvalues and, with V, orthonormal eigenvectors with A v = λ v; with N, the
  // same values and A left as it was.
  bool syevd_decomposes(char        jobz,
                        char        uplo,
                        std::size_t n)
  {
    const bool          lower {uplo == 'L' || uplo == 'l'};
    const std::size_t   lda   {n + 2};
    std::vector<double> sym(n * n);
    const auto          draw  {entries(n * n, 13)};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j <= i; ++j) {
        sym[i + (j * n)] = draw[i + (j * n)];
        sym[j + (i * n)] = draw[i + (j * n)];
      }
    }
    std::vector<double> a(lda * n, 0.0);
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t i {0}; i < n; ++i) {
        const bool read {lower ? i >= j : i <= j};
        a[i + (j * lda)] = read ? sym[i + (j * n)] : std::numeric_limits<double>::quiet_NaN();
      }
    }
    const std::vector<double> original {a};
    std::vector<double>       w(n);
    std::vector<double>       work(1);
    std::vector<int>          iwork(1);
    const auto                ln {static_cast<std::ptrdiff_t>(n)};
    const auto                ld {static_cast<std::ptrdiff_t>(lda)};
    if (numerical::blas::syevd(jobz, uplo, ln, a.data(), ld, w.data(), work.data(), -1, iwork.data(), -1) != 0) {
      return false;
    }
    const auto lwork  {static_cast<std::ptrdiff_t>(work[0])};
    const auto liwork {static_cast<std::ptrdiff_t>(iwork[0])};
    work.resize(static_cast<std::size_t>(lwork));
    iwork.resize(static_cast<std::size_t>(liwork));
    if (numerical::blas::syevd(jobz, uplo, ln, a.data(), ld, w.data(), work.data(), lwork, iwork.data(), liwork) != 0) {
      return false;
    }
    bool good {std::ranges::is_sorted(w)};
    if (jobz == 'N' || jobz == 'n') {
      std::vector<double> again {original};
      std::vector<double> values(n);
      std::vector<double> big_work(1 + (6 * n) + (2 * n * n));
      std::vector<int>    big_iwork(3 + (5 * n));
      good = good && numerical::blas::syevd('V', uplo, ln, again.data(), ld, values.data(), big_work.data(),
                                            static_cast<std::ptrdiff_t>(big_work.size()), big_iwork.data(),
                                            static_cast<std::ptrdiff_t>(big_iwork.size())) == 0;
      for (std::size_t i {0}; i < n; ++i) {
        good = good && std::abs(values[i] - w[i]) <= 1e-12 * static_cast<double>(n);
      }
      for (std::size_t i {0}; i < a.size(); ++i) {
        good = good && (std::isnan(original[i]) ? std::isnan(a[i]) : a[i] == original[i]);
      }
      return good;
    }
    for (std::size_t p {0}; p < n; ++p) {
      for (std::size_t q {0}; q < n; ++q) {
        double gram     {0.0};
        double residual {0.0};
        for (std::size_t i {0}; i < n; ++i) {
          gram     += a[i + (p * lda)] * a[i + (q * lda)];
          residual += sym[p + (i * n)] * a[i + (q * lda)];
        }
        residual -= w[q] * a[p + (q * lda)];
        good      = good && std::abs(gram - (p == q ? 1.0 : 0.0)) <= 1e-12 * static_cast<double>(n);
        good      = good && std::abs(residual) <= 1e-12 * static_cast<double>(n);
      }
    }
    return good;
  }

  // Whether gesdd(jobz) of an m × n matrix, leading dimensions padded, gives descending singular values and factors
  // with orthonormal columns (U) and rows (Vᵀ) whose product with Σ is A; with N, the values of S and A left as it was.
  bool gesdd_decomposes(char        jobz,
                        std::size_t m,
                        std::size_t n)
  {
    const std::size_t         k        {std::min(m, n)};
    const bool                all      {jobz == 'A' || jobz == 'a'};
    const bool                over     {jobz == 'O' || jobz == 'o'};
    const std::size_t         lda      {m + 1};
    const std::size_t         ucol     {all ? m : k};
    const std::size_t         vrow     {all ? n : k};
    const std::size_t         ldu      {m + 2};
    const std::size_t         ldvt     {n + 3};
    std::vector<double>       a        {entries(lda * n, 17)};
    const std::vector<double> original {a};
    std::vector<double>       s(k);
    std::vector<double>       u(ldu * m, 0.0);
    std::vector<double>       vt(ldvt * n, 0.0);
    std::vector<double>       work(1);
    std::vector<int>          iwork(8 * k);
    const auto                lm {static_cast<std::ptrdiff_t>(m)};
    const auto                ln {static_cast<std::ptrdiff_t>(n)};
    if (numerical::blas::gesdd(jobz, lm, ln, a.data(), static_cast<std::ptrdiff_t>(lda), s.data(), u.data(),
                               static_cast<std::ptrdiff_t>(ldu), vt.data(), static_cast<std::ptrdiff_t>(ldvt), work.data(),
                               -1, iwork.data()) != 0) {
      return false;
    }
    const auto lwork {static_cast<std::ptrdiff_t>(work[0])};
    work.resize(static_cast<std::size_t>(lwork));
    if (numerical::blas::gesdd(jobz, lm, ln, a.data(), static_cast<std::ptrdiff_t>(lda), s.data(), u.data(),
                               static_cast<std::ptrdiff_t>(ldu), vt.data(), static_cast<std::ptrdiff_t>(ldvt), work.data(),
                               lwork, iwork.data()) != 0) {
      return false;
    }
    bool good {std::ranges::is_sorted(s, std::ranges::greater {})};
    if (jobz == 'N' || jobz == 'n') {
      std::vector<double> again                       {original};
      std::vector<double> values(k);
      std::vector<double> big_work((3 * k) + std::max({m, n, (4 * k * k) + (4 * k)}));
      good = good && numerical::blas::gesdd('S', lm, ln, again.data(), static_cast<std::ptrdiff_t>(lda), values.data(), u.data(),
                                            static_cast<std::ptrdiff_t>(ldu), vt.data(), static_cast<std::ptrdiff_t>(ldvt),
                                            big_work.data(), static_cast<std::ptrdiff_t>(big_work.size()), iwork.data()) == 0;
      for (std::size_t i {0}; i < k; ++i) {
        good = good && std::abs(values[i] - s[i]) <= 1e-12 * static_cast<double>(m + n);
      }
      return good && a == original;
    }
    // Where O put a factor into A, read it from there.
    const double*     uf  {over && m >= n ? a.data() : u.data()};
    const std::size_t lu  {over && m >= n ? lda : ldu};
    const double*     vf  {over && m < n ? a.data() : vt.data()};
    const std::size_t lv  {over && m < n ? lda : ldvt};
    const double      tol {1e-12 * static_cast<double>(m + n)};
    for (std::size_t p {0}; p < ucol; ++p) {
      for (std::size_t q {0}; q < ucol; ++q) {
        double gram {0.0};
        for (std::size_t i {0}; i < m; ++i) {
          gram += uf[i + (p * lu)] * uf[i + (q * lu)];
        }
        good = good && std::abs(gram - (p == q ? 1.0 : 0.0)) <= tol;
      }
    }
    for (std::size_t p {0}; p < vrow; ++p) {
      for (std::size_t q {0}; q < vrow; ++q) {
        double gram {0.0};
        for (std::size_t j {0}; j < n; ++j) {
          gram += vf[p + (j * lv)] * vf[q + (j * lv)];
        }
        good = good && std::abs(gram - (p == q ? 1.0 : 0.0)) <= tol;
      }
    }
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        double sum {0.0};
        for (std::size_t l {0}; l < k; ++l) {
          sum += uf[i + (l * lu)] * s[l] * vf[l + (j * lv)];
        }
        good = good && std::abs(sum - original[i + (j * lda)]) <= tol;
      }
    }
    return good;
  }
} // namespace

TEST(syevd_diagonalises_either_triangle)
{
  EXPECT(syevd_decomposes('V', 'L', 9));
  EXPECT(syevd_decomposes('v', 'U', 40));
  EXPECT(syevd_decomposes('N', 'L', 9));
  EXPECT(syevd_decomposes('n', 'u', 40));
  EXPECT(syevd_decomposes('V', 'l', 1));
  EXPECT(syevd_decomposes('N', 'U', 0));
}

TEST(syevd_sizes_its_workspaces_and_reports_an_illegal_argument_and_a_nan)
{
  std::vector<double> a(9, 1.0);
  std::vector<double> w(3);
  std::vector<double> work(100);
  std::vector<int>    iwork(100);
  EXPECT_EQ(numerical::blas::syevd('V', 'L', 3, a.data(), 3, w.data(), work.data(), -1, iwork.data(), 100), 0);
  EXPECT(work[0] == 1.0 + 18.0 + 18.0 && iwork[0] == 18);
  EXPECT_EQ(numerical::blas::syevd('N', 'L', 3, a.data(), 3, w.data(), work.data(), 100, iwork.data(), -1), 0);
  EXPECT(work[0] == 7.0 && iwork[0] == 1);
  EXPECT_EQ(numerical::blas::syevd('V', 'U', 1, a.data(), 1, w.data(), work.data(), -1, iwork.data(), -1), 0);
  EXPECT(work[0] == 1.0 && iwork[0] == 1);
  EXPECT_EQ(numerical::blas::syevd('X', 'L', 3, a.data(), 3, w.data(), work.data(), 100, iwork.data(), 100), -1);
  EXPECT_EQ(numerical::blas::syevd('V', 'X', 3, a.data(), 3, w.data(), work.data(), 100, iwork.data(), 100), -2);
  EXPECT_EQ(numerical::blas::syevd('V', 'L', -1, a.data(), 3, w.data(), work.data(), 100, iwork.data(), 100), -3);
  EXPECT_EQ(numerical::blas::syevd('V', 'L', 3, a.data(), 2, w.data(), work.data(), 100, iwork.data(), 100), -5);
  EXPECT_EQ(numerical::blas::syevd('V', 'L', 3, a.data(), 3, w.data(), work.data(), 36, iwork.data(), 100), -8);
  EXPECT_EQ(numerical::blas::syevd('V', 'L', 3, a.data(), 3, w.data(), work.data(), 37, iwork.data(), 17), -10);
  a[1] = std::numeric_limits<double>::quiet_NaN(); // (2, 1): in the lower triangle, not the upper
  EXPECT_EQ(numerical::blas::syevd('V', 'L', 3, a.data(), 3, w.data(), work.data(), 37, iwork.data(), 18), -4);
  EXPECT_EQ(numerical::blas::syevd('N', 'L', 3, a.data(), 3, w.data(), work.data(), 7, iwork.data(), 1), -4);
  EXPECT_EQ(numerical::blas::syevd('N', 'U', 3, a.data(), 3, w.data(), work.data(), 7, iwork.data(), 1), 0);
}

TEST(gesdd_decomposes_tall_wide_and_square_matrices_by_every_job)
{
  for (const char job : {'A', 'S', 'O', 'N', 'a', 's', 'o', 'n'}) {
    EXPECT(gesdd_decomposes(job, 9, 5));
    EXPECT(gesdd_decomposes(job, 5, 9));
    EXPECT(gesdd_decomposes(job, 6, 6));
  }
  EXPECT(gesdd_decomposes('S', 0, 4));
}

namespace {
  // The workspace gesdd(job) asks for an m × n matrix.
  double gesdd_query(char           job,
                     std::ptrdiff_t m,
                     std::ptrdiff_t n)
  {
    double work    {0.0};
    double unused  {0.0};
    int    iunused {0};
    (void)numerical::blas::gesdd(job, m, n, &unused, 4, &unused, &unused, 4, &unused, 4, &work, -1, &iunused);
    return work;
  }
} // namespace

TEST(gesdd_sizes_its_workspace_and_reports_an_illegal_argument_and_a_nan)
{
  std::vector<double> a(12, 1.0); // 4 × 3
  std::vector<double> s(3);
  std::vector<double> u(16);
  std::vector<double> vt(9);
  std::vector<double> work(200);
  std::vector<int>    iwork(24);
  EXPECT(gesdd_query('N', 4, 3) == 9.0 + 21.0);
  EXPECT(gesdd_query('O', 4, 3) == 9.0 + 45.0 + 12.0);
  EXPECT(gesdd_query('S', 4, 3) == 9.0 + 36.0 + 12.0);
  EXPECT(gesdd_query('A', 3, 4) == 9.0 + 36.0 + 12.0);
  EXPECT(gesdd_query('N', 0, 3) == 3.0);
  EXPECT_EQ(numerical::blas::gesdd('X', 4, 3, a.data(), 4, s.data(), u.data(), 4, vt.data(), 3, work.data(), 200, iwork.data()), -1);
  EXPECT_EQ(numerical::blas::gesdd('S', -1, 3, a.data(), 4, s.data(), u.data(), 4, vt.data(), 3, work.data(), 200, iwork.data()), -2);
  EXPECT_EQ(numerical::blas::gesdd('S', 4, -1, a.data(), 4, s.data(), u.data(), 4, vt.data(), 3, work.data(), 200, iwork.data()), -3);
  EXPECT_EQ(numerical::blas::gesdd('S', 4, 3, a.data(), 3, s.data(), u.data(), 4, vt.data(), 3, work.data(), 200, iwork.data()), -5);
  EXPECT_EQ(numerical::blas::gesdd('S', 4, 3, a.data(), 4, s.data(), u.data(), 3, vt.data(), 3, work.data(), 200, iwork.data()), -8);
  EXPECT_EQ(numerical::blas::gesdd('O', 3, 4, a.data(), 3, s.data(), u.data(), 2, vt.data(), 1, work.data(), 200, iwork.data()), -8);
  EXPECT_EQ(numerical::blas::gesdd('O', 4, 3, a.data(), 4, s.data(), u.data(), 1, vt.data(), 2, work.data(), 200, iwork.data()), -10);
  EXPECT_EQ(numerical::blas::gesdd('A', 3, 4, a.data(), 3, s.data(), u.data(), 3, vt.data(), 3, work.data(), 200, iwork.data()), -10);
  EXPECT_EQ(numerical::blas::gesdd('S', 3, 4, a.data(), 3, s.data(), u.data(), 3, vt.data(), 2, work.data(), 200, iwork.data()), -10);
  EXPECT_EQ(numerical::blas::gesdd('N', 4, 3, a.data(), 4, s.data(), u.data(), 1, vt.data(), 1, work.data(), 29, iwork.data()), -12);
  EXPECT_EQ(numerical::blas::gesdd('N', 4, 3, a.data(), 4, s.data(), u.data(), 1, vt.data(), 1, work.data(), 30, iwork.data()), 0);
  a[5] = std::numeric_limits<double>::infinity();
  EXPECT_EQ(numerical::blas::gesdd('S', 4, 3, a.data(), 4, s.data(), u.data(), 4, vt.data(), 3, work.data(), 200, iwork.data()), -4);
  EXPECT_EQ(numerical::blas::gesdd('N', 4, 3, a.data(), 4, s.data(), u.data(), 1, vt.data(), 1, work.data(), 30, iwork.data()), -4);
}

TEST(the_fortran_eigensolvers_answer_as_the_cpp_ones)
{
  const int           two    {2};
  const int           lwork  {21};                 // 1 + 6 n + 2 n²
  const int           liwork {13};
  std::vector<double> a      {2.0, 1.0, 1.0, 2.0}; // eigenvalues 1 and 3
  std::vector<double> w(2);
  std::vector<double> work(21);
  std::vector<int>    iwork(13);
  int                 info   {7};
  dsyevd_("V", "L", &two, a.data(), &two, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info, 1, 1);
  EXPECT(info == 0 && std::abs(w[0] - 1.0) < 1e-15 && std::abs(w[1] - 3.0) < 1e-15);
  std::vector<double> b {3.0, 0.0, 0.0, -4.0}; // singular values 4 and 3
  std::vector<double> s(2);
  std::vector<double> u(4);
  std::vector<double> vt(4);
  std::vector<double> big(64);
  const int           lbig {64};
  dgesdd_("A", &two, &two, b.data(), &two, s.data(), u.data(), &two, vt.data(), &two, big.data(), &lbig, iwork.data(), &info, 1);
  EXPECT(info == 0 && std::abs(s[0] - 4.0) < 1e-15 && std::abs(s[1] - 3.0) < 1e-15);
  const int bad {-1};
  dsyevd_("V", "L", &bad, a.data(), &two, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info, 1, 1);
  EXPECT_EQ(info, -3);
  dgesdd_("A", &bad, &two, b.data(), &two, s.data(), u.data(), &two, vt.data(), &two, big.data(), &lbig, iwork.data(), &info, 1);
  EXPECT_EQ(info, -2);
}

namespace {
  // The first n columns of H₀ H₁ … H_{k-1}, H_i = I − τ_i w wᵀ with w 1 at row i and A's column i below it,
  // column-major m × n, by plain loops: each reflector applied in turn, the last first.
  std::vector<double> reflected_identity(const std::vector<double>& a,
                                         std::size_t                lda,
                                         const std::vector<double>& tau,
                                         std::size_t                m,
                                         std::size_t                n,
                                         std::size_t                k)
  {
    std::vector<double> q(m * n, 0.0);
    for (std::size_t c {0}; c < n; ++c) {
      q[c + (c * m)] = 1.0;
    }
    for (std::size_t i {k}; i-- > 0;) {
      for (std::size_t c {0}; c < n; ++c) {
        double dot {q[i + (c * m)]};
        for (std::size_t r {i + 1}; r < m; ++r) {
          dot += a[r + (i * lda)] * q[r + (c * m)];
        }
        q[i + (c * m)] -= tau[i] * dot;
        for (std::size_t r {i + 1}; r < m; ++r) {
          q[r + (c * m)] -= tau[i] * a[r + (i * lda)] * dot;
        }
      }
    }
    return q;
  }

  // Whether geqrf of an m × n matrix (the leading dimension padded, \p zero_column zeroed if below n) leaves R and
  // reflectors whose product is A, Q orthonormal, and orgqr forming that Q.
  bool geqrf_factorises(std::size_t m,
                        std::size_t n,
                        std::size_t zero_column)
  {
    const std::size_t   lda {m + 3};
    const std::size_t   k   {std::min(m, n)};
    std::vector<double> a   {entries(lda * n, 23)};
    for (std::size_t i {0}; i < m && zero_column < n; ++i) {
      a[i + (zero_column * lda)] = 0.0;
    }
    const std::vector<double> original {a};
    std::vector<double>       tau(k);
    double                    query    {0.0};
    const auto                lm       {static_cast<std::ptrdiff_t>(m)};
    const auto                ln       {static_cast<std::ptrdiff_t>(n)};
    const auto                ld       {static_cast<std::ptrdiff_t>(lda)};
    if (numerical::blas::geqrf(lm, ln, a.data(), ld, tau.data(), &query, -1) != 0) {
      return false;
    }
    std::vector<double> work(static_cast<std::size_t>(query));
    if (numerical::blas::geqrf(lm, ln, a.data(), ld, tau.data(), work.data(), static_cast<std::ptrdiff_t>(work.size())) != 0) {
      return false;
    }
    const std::vector<double> q    {reflected_identity(a, lda, tau, m, k, k)};
    const double              tol  {1e-13 * static_cast<double>(m + n)};
    bool                      good {zero_column >= k || tau[zero_column] == 0.0};
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        double sum {0.0};
        for (std::size_t l {0}; l <= std::min(j, k - 1); ++l) {
          sum += q[i + (l * m)] * a[l + (j * lda)];
        }
        good = good && std::abs(sum - original[i + (j * lda)]) <= tol;
      }
    }
    for (std::size_t p {0}; p < k; ++p) {
      for (std::size_t c {0}; c < k; ++c) {
        double gram {0.0};
        for (std::size_t i {0}; i < m; ++i) {
          gram += q[i + (p * m)] * q[i + (c * m)];
        }
        good = good && std::abs(gram - (p == c ? 1.0 : 0.0)) <= tol;
      }
    }
    if (numerical::blas::orgqr(lm, static_cast<std::ptrdiff_t>(k), static_cast<std::ptrdiff_t>(k), a.data(), ld, tau.data(),
                               work.data(), static_cast<std::ptrdiff_t>(std::max<std::size_t>(1, k))) != 0) {
      return false;
    }
    for (std::size_t j {0}; j < k; ++j) {
      for (std::size_t i {0}; i < m; ++i) {
        good = good && std::abs(a[i + (j * lda)] - q[i + (j * m)]) <= tol;
      }
    }
    return good;
  }

  // Whether orgqr forms the product of k arbitrary reflectors (not only a QR's: τ_i not 2 / ‖w‖², each I − τ w wᵀ a
  // contraction, so that the product stays of order 1) as plain loops do.
  bool orgqr_forms(std::size_t m,
                   std::size_t n,
                   std::size_t k)
  {
    const std::size_t   lda {m + 1};
    std::vector<double> a   {entries(lda * n, 29)};
    std::vector<double> tau {entries(k, 31)};
    for (std::size_t i {0}; i < k; ++i) {
      double norm_squared {1.0};
      for (std::size_t r {i + 1}; r < m; ++r) {
        norm_squared += a[r + (i * lda)] * a[r + (i * lda)];
      }
      tau[i] = (tau[i] + 1.0) / norm_squared; // in [0, 2 / ‖w‖²)
    }
    const std::vector<double> want {reflected_identity(a, lda, tau, m, n, k)};
    std::vector<double>       work(n + 1);
    if (numerical::blas::orgqr(static_cast<std::ptrdiff_t>(m), static_cast<std::ptrdiff_t>(n), static_cast<std::ptrdiff_t>(k), a.data(),
                               static_cast<std::ptrdiff_t>(lda), tau.data(), work.data(), static_cast<std::ptrdiff_t>(work.size())) != 0) {
      return false;
    }
    bool good {true};
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t i {0}; i < m; ++i) {
        good = good && std::abs(a[i + (j * lda)] - want[i + (j * m)]) <= 1e-12 * static_cast<double>(m);
      }
    }
    return good;
  }
} // namespace

TEST(geqrf_factorises_and_orgqr_forms_its_q)
{
  EXPECT(geqrf_factorises(9, 5, 99));
  EXPECT(geqrf_factorises(5, 9, 99));
  EXPECT(geqrf_factorises(100, 70, 40)); // three blocks of reflectors, one column zero: its τ 0
  EXPECT(geqrf_factorises(70, 100, 3));
  EXPECT(geqrf_factorises(1, 1, 99));
}

TEST(geqrf_takes_the_identity_where_nothing_lies_below_the_diagonal)
{
  // An upper triangular A is its own R, every τ 0, as LAPACK's dlarfg leaves it (a reflection would flip R's signs:
  // SciPy compares the Q of its pivoted QR, the reference dgeqp3's, with dgeqrf's of the same columns).
  std::vector<double>       a {3.0, 0.0, 0.0, 1.0, -2.0, 0.0, 5.0, 4.0, -1.0}; // [[3, 1, 5], [0, −2, 4], [0, 0, −1]]
  const std::vector<double> r {a};
  std::vector<double>       tau(3, 7.0);
  std::vector<double>       work(3);
  EXPECT_EQ(numerical::blas::geqrf(3, 3, a.data(), 3, tau.data(), work.data(), 3), 0);
  EXPECT(a == r);
  EXPECT(tau[0] == 0.0 && tau[1] == 0.0 && tau[2] == 0.0);
  std::vector<double>       wide   {2.0, 0.0, 1.0, -3.0}; // [[2, 1], [0, −3]] and a third row of a taller one below
  const std::vector<double> wide_r {wide};
  EXPECT_EQ(numerical::blas::geqrf(2, 2, wide.data(), 2, tau.data(), work.data(), 3), 0);
  EXPECT(wide == wide_r && tau[0] == 0.0 && tau[1] == 0.0);
  // [[1, 7], [2, 0], [0, 0]]: the first reflection mixes rows 0 and 1, and leaves row 2 of column 1 zero, so the
  // second is the identity; Q R is still A.
  const std::vector<double> tall    {1.0, 2.0, 0.0, 7.0, 0.0, 0.0};
  std::vector<double>       factors {tall};
  EXPECT_EQ(numerical::blas::geqrf(3, 2, factors.data(), 3, tau.data(), work.data(), 3), 0);
  EXPECT(tau[0] != 0.0 && tau[1] == 0.0);
  std::vector<double> q {factors};
  EXPECT_EQ(numerical::blas::orgqr(3, 2, 2, q.data(), 3, tau.data(), work.data(), 3), 0);
  for (std::size_t i {0}; i < 3; ++i) {
    for (std::size_t j {0}; j < 2; ++j) {
      double sum {0.0};
      for (std::size_t l {0}; l <= j; ++l) {
        sum += q[i + (l * 3)] * factors[l + (j * 3)];
      }
      EXPECT(std::abs(sum - tall[i + (j * 3)]) < 1e-14);
    }
  }
}

TEST(orgqr_forms_any_product_of_reflectors)
{
  EXPECT(orgqr_forms(80, 70, 65));
  EXPECT(orgqr_forms(80, 70, 0));
  EXPECT(orgqr_forms(10, 10, 10));
  EXPECT(orgqr_forms(5, 0, 0));
}

TEST(geqrf_and_orgqr_size_their_workspaces_and_report_an_illegal_argument_and_a_nan)
{
  std::vector<double> a(12, 1.0); // 4 × 3
  std::vector<double> tau(3, 1.0);
  std::vector<double> work(8);
  EXPECT_EQ(numerical::blas::geqrf(4, 3, a.data(), 4, tau.data(), work.data(), -1), 0);
  EXPECT(work[0] == 3.0);
  EXPECT_EQ(numerical::blas::geqrf(0, 3, a.data(), 1, tau.data(), work.data(), -1), 0);
  EXPECT(work[0] == 1.0);
  EXPECT_EQ(numerical::blas::geqrf(0, 3, a.data(), 1, tau.data(), work.data(), 1), 0);
  EXPECT_EQ(numerical::blas::geqrf(-1, 3, a.data(), 4, tau.data(), work.data(), 8), -1);
  EXPECT_EQ(numerical::blas::geqrf(4, -1, a.data(), 4, tau.data(), work.data(), 8), -2);
  EXPECT_EQ(numerical::blas::geqrf(4, 3, a.data(), 3, tau.data(), work.data(), 8), -4);
  EXPECT_EQ(numerical::blas::geqrf(4, 3, a.data(), 4, tau.data(), work.data(), 2), -7);
  EXPECT_EQ(numerical::blas::orgqr(4, 3, 3, a.data(), 4, tau.data(), work.data(), -1), 0);
  EXPECT(work[0] == 3.0);
  EXPECT_EQ(numerical::blas::orgqr(-1, 0, 0, a.data(), 4, tau.data(), work.data(), 8), -1);
  EXPECT_EQ(numerical::blas::orgqr(4, -1, 0, a.data(), 4, tau.data(), work.data(), 8), -2);
  EXPECT_EQ(numerical::blas::orgqr(2, 3, 0, a.data(), 4, tau.data(), work.data(), 8), -2);
  EXPECT_EQ(numerical::blas::orgqr(4, 3, -1, a.data(), 4, tau.data(), work.data(), 8), -3);
  EXPECT_EQ(numerical::blas::orgqr(4, 2, 3, a.data(), 4, tau.data(), work.data(), 8), -3);
  EXPECT_EQ(numerical::blas::orgqr(4, 3, 3, a.data(), 3, tau.data(), work.data(), 8), -5);
  EXPECT_EQ(numerical::blas::orgqr(4, 3, 3, a.data(), 4, tau.data(), work.data(), 2), -8);
  a[0] = std::numeric_limits<double>::quiet_NaN(); // on the diagonal: R's, not a reflector's, for orgqr
  EXPECT_EQ(numerical::blas::geqrf(4, 3, a.data(), 4, tau.data(), work.data(), 8), -3);
  a[0] = 1.0;
  a[3] = std::numeric_limits<double>::infinity();  // below the diagonal of column 0: reflector 0's
  EXPECT_EQ(numerical::blas::orgqr(4, 3, 1, a.data(), 4, tau.data(), work.data(), 8), -4);
  a[3]   = 1.0;
  tau[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(numerical::blas::orgqr(4, 3, 2, a.data(), 4, tau.data(), work.data(), 8), -6);
  a[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(numerical::blas::orgqr(4, 3, 1, a.data(), 4, tau.data(), work.data(), 8), 0);
}

TEST(the_fortran_qr_routines_answer_as_the_cpp_ones)
{
  const int           two  {2};
  const int           lw   {2};
  std::vector<double> a    {3.0, 4.0, 1.0, 2.0}; // column 0 = (3, 4): R(0, 0) = −5
  std::vector<double> tau(2);
  std::vector<double> work(2);
  int                 info {7};
  dgeqrf_(&two, &two, a.data(), &two, tau.data(), work.data(), &lw, &info);
  EXPECT(info == 0 && std::abs(a[0] + 5.0) < 1e-15);
  dorgqr_(&two, &two, &two, a.data(), &two, tau.data(), work.data(), &lw, &info);
  EXPECT(info == 0 && std::abs(a[0] + 0.6) < 1e-15 && std::abs(a[1] + 0.8) < 1e-15);
  const int bad {-1};
  dgeqrf_(&bad, &two, a.data(), &two, tau.data(), work.data(), &lw, &info);
  EXPECT_EQ(info, -1);
  dorgqr_(&two, &two, &bad, a.data(), &two, tau.data(), work.data(), &lw, &info);
  EXPECT_EQ(info, -3);
}

namespace {
  // Whether gelsd of an m × n A of rank r (a product of m × r and r × n factors) and nrhs right-hand sides, the
  // leading dimensions padded, gives the rank, the singular values and V Σ⁺ Uᵀ B as gesdd's factors give them by plain
  // loops, A left as it was and, where m > n and the rank full, the residual's norm in row n.
  bool gelsd_solves(std::size_t m,
                    std::size_t n,
                    std::size_t r,
                    std::size_t nrhs,
                    double      rcond)
  {
    const std::size_t   lda {m + 1};
    const std::size_t   ldb {std::max(m, n) + 2};
    const std::size_t   k   {std::min(m, n)};
    const auto          f   {entries(m * r, 37)};
    const auto          g   {entries(r * n, 41)};
    std::vector<double> a(lda * n, 0.0);
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        for (std::size_t l {0}; l < r; ++l) {
          a[i + (j * lda)] += f[i + (l * m)] * g[l + (j * r)];
        }
      }
    }
    std::vector<double>       b      {entries(ldb * nrhs, 43)};
    const std::vector<double> a0     {a};
    const std::vector<double> b0     {b};
    std::vector<double>       s(k);
    int                       rank   {-1};
    double                    query  {0.0};
    int                       iquery {0};
    const auto                lm     {static_cast<std::ptrdiff_t>(m)};
    const auto                ln     {static_cast<std::ptrdiff_t>(n)};
    const auto                lr     {static_cast<std::ptrdiff_t>(nrhs)};
    if (numerical::blas::gelsd(lm, ln, lr, a.data(), static_cast<std::ptrdiff_t>(lda), b.data(), static_cast<std::ptrdiff_t>(ldb),
                               s.data(), rcond, &rank, &query, -1, &iquery) != 0) {
      return false;
    }
    std::vector<double> work(static_cast<std::size_t>(query));
    std::vector<int>    iwork(static_cast<std::size_t>(iquery));
    if (numerical::blas::gelsd(lm, ln, lr, a.data(), static_cast<std::ptrdiff_t>(lda), b.data(), static_cast<std::ptrdiff_t>(ldb),
                               s.data(), rcond, &rank, work.data(), static_cast<std::ptrdiff_t>(work.size()), iwork.data()) != 0) {
      return false;
    }
    // The reference: gesdd's U, Σ, Vᵀ, the values at or under the cutoff dropped.
    std::vector<double> copy {a0};
    std::vector<double> sigma(k);
    std::vector<double> u(m * k);
    std::vector<double> vt(k * n);
    std::vector<double> svd_work((3 * k) + std::max({m, n, (4 * k * k) + (4 * k)}));
    std::vector<int>    svd_iwork(8 * k);
    if (numerical::blas::gesdd('S', lm, ln, copy.data(), static_cast<std::ptrdiff_t>(lda), sigma.data(), u.data(), lm, vt.data(),
                               static_cast<std::ptrdiff_t>(k), svd_work.data(), static_cast<std::ptrdiff_t>(svd_work.size()),
                               svd_iwork.data()) != 0) {
      return false;
    }
    const double cutoff    {(rcond > 0.0 && rcond < 1.0 ? rcond : 0x1p-53) * sigma[0]};
    std::size_t  want_rank {0};
    bool         good      {a == a0};
    for (std::size_t d {0}; d < k; ++d) {
      want_rank += static_cast<std::size_t>(sigma[d] > cutoff);
      good       = good && std::abs(s[d] - sigma[d]) <= 1e-12 * static_cast<double>(m + n);
    }
    good = good && std::cmp_equal(rank, want_rank) && want_rank == std::min(r, k);
    for (std::size_t c {0}; c < nrhs; ++c) {
      for (std::size_t j {0}; j < n; ++j) {
        double x {0.0};
        for (std::size_t d {0}; d < want_rank; ++d) {
          double projection {0.0};
          for (std::size_t i {0}; i < m; ++i) {
            projection += u[i + (d * m)] * b0[i + (c * ldb)];
          }
          x += vt[d + (j * k)] * projection / sigma[d];
        }
        good = good && std::abs(b[j + (c * ldb)] - x) <= 1e-9;
      }
      if (m > n) {
        double residual {0.0};
        for (std::size_t i {0}; i < m; ++i) {
          double fitted {0.0};
          for (std::size_t j {0}; j < n; ++j) {
            fitted += a0[i + (j * lda)] * b[j + (c * ldb)];
          }
          residual += (b0[i + (c * ldb)] - fitted) * (b0[i + (c * ldb)] - fitted);
        }
        const double row_n {want_rank == n ? std::sqrt(residual) : 0.0};
        good = good && std::abs(b[n + (c * ldb)] - row_n) <= 1e-12 * static_cast<double>(m);
        for (std::size_t i {n + 1}; i < m; ++i) {
          good = good && b[i + (c * ldb)] == 0.0;
        }
      }
    }
    return good;
  }
} // namespace

TEST(gelsd_gives_the_minimum_norm_least_squares_solution_its_rank_and_residual)
{
  EXPECT(gelsd_solves(12, 5, 5, 3, -1.0));   // tall, full rank: residuals in row n
  EXPECT(gelsd_solves(5, 12, 5, 2, 0.0));    // wide, full rank
  EXPECT(gelsd_solves(30, 20, 7, 2, 1e-10)); // rank-deficient: the minimum-norm solution
  EXPECT(gelsd_solves(20, 30, 7, 1, 1e-10));
  EXPECT(gelsd_solves(9, 9, 9, 4, 2.0));     // rcond ≥ 1: LAPACK's precision, 2⁻⁵³
  EXPECT(gelsd_solves(80, 60, 60, 1, -1.0)); // past one leaf of dgelsd's divide and conquer
  EXPECT(gelsd_solves(6, 4, 4, 0, -1.0));    // no right-hand side: the values and the rank alone
}

TEST(gelsd_cuts_the_rank_at_lapack_s_precision_not_the_machine_epsilon)
{
  // Rank 3 (column 3 is 2 column 0 − column 1), its fourth singular value rounding noise near 5e-16: above 2⁻⁵³ times
  // the largest (4.9e-16), under the machine epsilon times it (9.9e-16). NumPy's test of rcond = -1 asks rank 4, as
  // LAPACK's dgelsd gives it.
  std::vector<double> a {0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0,
                         1.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0};
  std::vector<double> b {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> s(4);
  std::vector<double> work(1000);
  std::vector<int>    iwork(100);
  int                 rank {0};
  EXPECT_EQ(numerical::blas::gelsd(6, 4, 1, a.data(), 6, b.data(), 6, s.data(), -1.0, &rank, work.data(), 1000, iwork.data()), 0);
  EXPECT(s[3] > 0x1p-53 * s[0] && s[3] < 0x1p-52 * s[0]);
  EXPECT_EQ(rank, 4);
}

TEST(gelsd_sizes_its_workspaces_and_reports_an_illegal_argument_and_a_nan)
{
  std::vector<double> a(12, 1.0); // 4 × 3
  std::vector<double> b(8, 1.0);
  std::vector<double> s(3);
  std::vector<double> work(2000);
  std::vector<int>    iwork(100);
  int                 rank {-1};
  EXPECT_EQ(numerical::blas::gelsd(4, 3, 2, a.data(), 4, b.data(), 4, s.data(), -1.0, &rank, work.data(), -1, iwork.data()), 0);
  EXPECT(work[0] == 9.0 + 27.0 + 150.0 + 6.0 + 676.0 && iwork[0] == 33);           // 3 < 26: no level, 12·3 + 50·3 + 3·2 + 676
  EXPECT_EQ(numerical::blas::gelsd(200, 100, 1, a.data(), 200, b.data(), 200, s.data(), -1.0, &rank, work.data(), -1, iwork.data()), 0);
  EXPECT(work[0] == 1200.0 + 5000.0 + 1600.0 + 100.0 + 676.0 && iwork[0] == 1700); // log₂(100 / 26) → 2 levels
  EXPECT_EQ(numerical::blas::gelsd(-1, 3, 2, a.data(), 4, b.data(), 4, s.data(), -1.0, &rank, work.data(), 2000, iwork.data()), -1);
  EXPECT_EQ(numerical::blas::gelsd(4, -1, 2, a.data(), 4, b.data(), 4, s.data(), -1.0, &rank, work.data(), 2000, iwork.data()), -2);
  EXPECT_EQ(numerical::blas::gelsd(4, 3, -1, a.data(), 4, b.data(), 4, s.data(), -1.0, &rank, work.data(), 2000, iwork.data()), -3);
  EXPECT_EQ(numerical::blas::gelsd(4, 3, 2, a.data(), 3, b.data(), 4, s.data(), -1.0, &rank, work.data(), 2000, iwork.data()), -5);
  EXPECT_EQ(numerical::blas::gelsd(3, 4, 2, a.data(), 3, b.data(), 3, s.data(), -1.0, &rank, work.data(), 2000, iwork.data()), -7);
  EXPECT_EQ(numerical::blas::gelsd(4, 3, 2, a.data(), 4, b.data(), 4, s.data(), -1.0, &rank, work.data(), 867, iwork.data()), -12);
  EXPECT_EQ(numerical::blas::gelsd(0, 3, 2, a.data(), 1, b.data(), 3, s.data(), -1.0, &rank, work.data(), 2000, iwork.data()), 0);
  EXPECT_EQ(rank, 0);
  rank = -1;
  EXPECT_EQ(numerical::blas::gelsd(4, 0, 2, a.data(), 4, b.data(), 4, s.data(), -1.0, &rank, work.data(), 2000, iwork.data()), 0);
  EXPECT_EQ(rank, 0);
  b[5] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(numerical::blas::gelsd(4, 3, 2, a.data(), 4, b.data(), 4, s.data(), -1.0, &rank, work.data(), 868, iwork.data()), -6);
  b[5] = 1.0;
  a[7] = std::numeric_limits<double>::infinity();
  EXPECT_EQ(numerical::blas::gelsd(4, 3, 2, a.data(), 4, b.data(), 4, s.data(), -1.0, &rank, work.data(), 868, iwork.data()), -4);
}

TEST(the_fortran_least_squares_answers_as_the_cpp_one)
{
  const int           three {3};
  const int           one   {1};
  const int           lwork {1000};
  const double        rcond {-1.0};
  std::vector<double> a     {1.0, 1.0, 1.0, 0.0, 1.0, 2.0}; // fit c₀ + c₁ t at t = 0, 1, 2
  std::vector<double> b     {1.0, 2.0, 4.0};                // c = (5/6, 3/2), residual 1/6 in squares
  std::vector<double> s(2);
  std::vector<double> work(1000);
  std::vector<int>    iwork(100);
  int                 rank  {0};
  int                 info  {7};
  const int           two   {2};
  dgelsd_(&three, &two, &one, a.data(), &three, b.data(), &three, s.data(), &rcond, &rank, work.data(), &lwork, iwork.data(), &info);
  EXPECT(info == 0 && rank == 2 && std::abs(b[0] - (5.0 / 6.0)) < 1e-14 && std::abs(b[1] - 1.5) < 1e-14);
  EXPECT(std::abs((b[2] * b[2]) - (1.0 / 6.0)) < 1e-14);
  const int bad {-1};
  dgelsd_(&three, &bad, &one, a.data(), &three, b.data(), &three, s.data(), &rcond, &rank, work.data(), &lwork, iwork.data(), &info);
  EXPECT_EQ(info, -2);
}

namespace {
  // Eigenvector j of dgeev's packed real form (column-major, ld apart), as a complex vector.
  std::vector<std::complex<double>> packed_vector(const std::vector<double>& packed,
                                                  std::size_t                ld,
                                                  const std::vector<double>& wi,
                                                  std::size_t                j,
                                                  std::size_t                n)
  {
    std::vector<std::complex<double>> v(n);
    for (std::size_t i {0}; i < n; ++i) {
      if (wi[j] == 0.0) {
        v[i] = packed[i + (j * ld)];
      }
      else if (wi[j] > 0.0) {
        v[i] = {packed[i + (j * ld)], packed[i + ((j + 1) * ld)]};
      }
      else {
        v[i] = {packed[i + ((j - 1) * ld)], -packed[i + (j * ld)]};
      }
    }
    return v;
  }

  // Whether v has unit norm with its component of largest modulus real.
  bool unit_and_real_at_largest(const std::vector<std::complex<double>>& v)
  {
    double      norm_squared {0.0};
    std::size_t largest      {0};
    for (std::size_t i {0}; i < v.size(); ++i) {
      norm_squared += std::norm(v[i]);
      largest       = std::abs(v[i]) > std::abs(v[largest]) ? i : largest;
    }
    return std::abs(norm_squared - 1.0) <= 1e-13 && std::abs(v[largest].imag()) <= 1e-13;
  }

  // Whether geev(jobvl, jobvr) of a: its eigenvalues with conjugate pairs adjacent (positive imaginary part first);
  // A v = λ v and uᴴ A = λ uᴴ for the vectors asked, of unit norm and largest component real; the eigenvalues alone
  // the same; A left as it was.
  bool geev_decomposes(char                       jobvl,
                       char                       jobvr,
                       const std::vector<double>& a,
                       std::size_t                n)
  {
    const std::size_t   ld {n + 1};
    std::vector<double> padded(ld * n, 0.0);
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t i {0}; i < n; ++i) {
        padded[i + (j * ld)] = a[i + (j * n)];
      }
    }
    const std::vector<double> before {padded};
    std::vector<double>       wr(n);
    std::vector<double>       wi(n);
    std::vector<double>       vl(ld * n, 0.0);
    std::vector<double>       vr(ld * n, 0.0);
    double                    query {0.0};
    const auto                ln    {static_cast<std::ptrdiff_t>(n)};
    const auto                lld   {static_cast<std::ptrdiff_t>(ld)};
    if (numerical::blas::geev(jobvl, jobvr, ln, padded.data(), lld, wr.data(), wi.data(), vl.data(), lld, vr.data(), lld, &query, -1)
        != 0) {
      return false;
    }
    std::vector<double> work(static_cast<std::size_t>(query));
    if (numerical::blas::geev(jobvl, jobvr, ln, padded.data(), lld, wr.data(), wi.data(), vl.data(), lld, vr.data(), lld, work.data(),
                              static_cast<std::ptrdiff_t>(work.size())) != 0) {
      return false;
    }
    std::vector<double> wr0(n);
    std::vector<double> wi0(n);
    std::vector<double> none(1);
    std::vector<double> small_work(std::max<std::size_t>(1, 3 * n));
    bool                good {padded == before};
    good = good && numerical::blas::geev('N', 'N', ln, padded.data(), lld, wr0.data(), wi0.data(), none.data(), 1, none.data(), 1,
                                         small_work.data(), static_cast<std::ptrdiff_t>(small_work.size())) == 0;
    const double scale {1e-12 * static_cast<double>(n)};
    for (std::size_t j {0}; j < n; ++j) {
      good = good && std::abs(wr[j] - wr0[j]) <= scale && std::abs(wi[j] - wi0[j]) <= scale;
      if (wi[j] > 0.0) {
        good = good && j + 1 < n && wr[j + 1] == wr[j] && wi[j + 1] == -wi[j];
      }
      const std::complex<double> lambda {wr[j], wi[j]};
      if (jobvr == 'V' || jobvr == 'v') {
        const auto v {packed_vector(vr, ld, wi, j, n)};
        good = good && unit_and_real_at_largest(v);
        for (std::size_t i {0}; i < n; ++i) {
          std::complex<double> av {};
          for (std::size_t l {0}; l < n; ++l) {
            av += a[i + (l * n)] * v[l];
          }
          good = good && std::abs(av - (lambda * v[i])) <= scale;
        }
      }
      if (jobvl == 'V' || jobvl == 'v') {
        const auto u {packed_vector(vl, ld, wi, j, n)};
        good = good && unit_and_real_at_largest(u);
        for (std::size_t c {0}; c < n; ++c) { // (uᴴ A)_c = Σ conj(u_l) A(l, c)
          std::complex<double> ua {};
          for (std::size_t l {0}; l < n; ++l) {
            ua += std::conj(u[l]) * a[l + (c * n)];
          }
          good = good && std::abs(ua - (lambda * std::conj(u[c]))) <= scale;
        }
      }
    }
    return good;
  }

  // X diag(2, 2, 3, −1, 5) X⁻¹ with X lower unit triangular: a double eigenvalue, not symmetric.
  std::vector<double> with_a_double_eigenvalue()
  {
    const std::size_t   n   {5};
    const double        d[] {2.0, 2.0, 3.0, -1.0, 5.0};
    const auto          l   {entries(n * n, 47)};
    std::vector<double> x(n * n, 0.0);
    std::vector<double> inverse(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      x[i + (i * n)] = 1.0;
      for (std::size_t j {0}; j < i; ++j) {
        x[i + (j * n)] = l[i + (j * n)];
      }
    }
    for (std::size_t c {0}; c < n; ++c) { // X⁻¹ column by column, forward substitution
      for (std::size_t i {0}; i < n; ++i) {
        double sum {i == c ? 1.0 : 0.0};
        for (std::size_t j {0}; j < i; ++j) {
          sum -= x[i + (j * n)] * inverse[j + (c * n)];
        }
        inverse[i + (c * n)] = sum;
      }
    }
    std::vector<double> a(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        for (std::size_t p {0}; p < n; ++p) {
          a[i + (j * n)] += x[i + (p * n)] * d[p] * inverse[p + (j * n)];
        }
      }
    }
    return a;
  }
} // namespace

TEST(geev_gives_the_eigenvalues_and_either_side_s_vectors)
{
  const auto general {entries(std::size_t {40} *40, 53)}; // conjugate pairs and real eigenvalues
  for (const char left : {'N', 'V', 'n', 'v'}) {
    for (const char right : {'N', 'V', 'n', 'v'}) {
      EXPECT(geev_decomposes(left, right, general, 40));
    }
  }
  EXPECT(geev_decomposes('V', 'V', with_a_double_eigenvalue(), 5));
  EXPECT(geev_decomposes('V', 'V', {0.0, 1.0, -1.0, 0.0}, 2)); // ±i alone
  EXPECT(geev_decomposes('V', 'V', {4.0}, 1));
  EXPECT(geev_decomposes('V', 'V', {}, 0));
}

TEST(geev_sizes_its_workspace_and_reports_an_illegal_argument_and_a_nan)
{
  std::vector<double> a(9, 1.0);
  std::vector<double> w(6);
  std::vector<double> v(9);
  std::vector<double> work(12);
  EXPECT_EQ(numerical::blas::geev('N', 'N', 3, a.data(), 3, w.data(), w.data() + 3, v.data(), 1, v.data(), 1, work.data(), -1), 0);
  EXPECT(work[0] == 9.0);
  EXPECT_EQ(numerical::blas::geev('N', 'V', 3, a.data(), 3, w.data(), w.data() + 3, v.data(), 1, v.data(), 3, work.data(), -1), 0);
  EXPECT(work[0] == 12.0);
  EXPECT_EQ(numerical::blas::geev('N', 'N', 0, a.data(), 1, w.data(), w.data() + 3, v.data(), 1, v.data(), 1, work.data(), -1), 0);
  EXPECT(work[0] == 1.0);
  EXPECT_EQ(numerical::blas::geev('X', 'N', 3, a.data(), 3, w.data(), w.data() + 3, v.data(), 1, v.data(), 1, work.data(), 12), -1);
  EXPECT_EQ(numerical::blas::geev('N', 'X', 3, a.data(), 3, w.data(), w.data() + 3, v.data(), 1, v.data(), 1, work.data(), 12), -2);
  EXPECT_EQ(numerical::blas::geev('N', 'N', -1, a.data(), 3, w.data(), w.data() + 3, v.data(), 1, v.data(), 1, work.data(), 12), -3);
  EXPECT_EQ(numerical::blas::geev('N', 'N', 3, a.data(), 2, w.data(), w.data() + 3, v.data(), 1, v.data(), 1, work.data(), 12), -5);
  EXPECT_EQ(numerical::blas::geev('N', 'N', 3, a.data(), 3, w.data(), w.data() + 3, v.data(), 0, v.data(), 1, work.data(), 12), -9);
  EXPECT_EQ(numerical::blas::geev('V', 'N', 3, a.data(), 3, w.data(), w.data() + 3, v.data(), 2, v.data(), 1, work.data(), 12), -9);
  EXPECT_EQ(numerical::blas::geev('N', 'N', 3, a.data(), 3, w.data(), w.data() + 3, v.data(), 1, v.data(), 0, work.data(), 12), -11);
  EXPECT_EQ(numerical::blas::geev('N', 'V', 3, a.data(), 3, w.data(), w.data() + 3, v.data(), 1, v.data(), 2, work.data(), 12), -11);
  EXPECT_EQ(numerical::blas::geev('N', 'N', 3, a.data(), 3, w.data(), w.data() + 3, v.data(), 1, v.data(), 1, work.data(), 8), -13);
  EXPECT_EQ(numerical::blas::geev('V', 'N', 3, a.data(), 3, w.data(), w.data() + 3, v.data(), 3, v.data(), 1, work.data(), 11), -13);
  a[4] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(numerical::blas::geev('N', 'N', 3, a.data(), 3, w.data(), w.data() + 3, v.data(), 1, v.data(), 1, work.data(), 12), -4);
}

TEST(the_fortran_eigenvalue_routine_answers_as_the_cpp_one)
{
  const int           two   {2};
  const int           one   {1};
  const int           lwork {8};
  std::vector<double> a     {0.0, -2.0, 2.0, 0.0}; // [[0, 2], [−2, 0]]: ±2i
  std::vector<double> wr(2);
  std::vector<double> wi(2);
  std::vector<double> vr(4);
  std::vector<double> none(1);
  std::vector<double> work(8);
  int                 info  {7};
  dgeev_("N", "V", &two, a.data(), &two, wr.data(), wi.data(), none.data(), &one, vr.data(), &two, work.data(), &lwork, &info, 1, 1);
  EXPECT(info == 0 && wr[0] == 0.0 && std::abs(wi[0] - 2.0) < 1e-15 && wi[1] == -wi[0]);
  const int bad {-1};
  dgeev_("N", "V", &bad, a.data(), &two, wr.data(), wi.data(), none.data(), &one, vr.data(), &two, work.data(), &lwork, &info, 1, 1);
  EXPECT_EQ(info, -3);
}

namespace {
  // Whether syrk(uplo, trans) of order n and depth k, the leading dimensions padded, gives α op(A) op(A)ᵀ + β C on the
  // triangle by a plain loop and leaves the other triangle as it was (NaN where β = 0: never read on the triangle).
  bool syrk_matches(char        uplo,
                    char        trans,
                    std::size_t n,
                    std::size_t k,
                    double      alpha,
                    double      beta)
  {
    const bool                lower {uplo == 'L' || uplo == 'l'};
    const bool                t     {trans != 'N' && trans != 'n'};
    const std::size_t         lda   {(t ? k : n) + 2};
    const std::size_t         ldc   {n + 1};
    const std::vector<double> a     {entries(lda * (t ? n : k), 59)};
    std::vector<double>       c     {entries(ldc * n, 61)};
    for (std::size_t j {0}; j < n && beta == 0.0; ++j) {
      for (std::size_t i {lower ? j : 0}; i < (lower ? n : j + 1); ++i) {
        c[i + (j * ldc)] = std::numeric_limits<double>::quiet_NaN();
      }
    }
    const std::vector<double> before {c};
    if (numerical::blas::syrk(uplo, trans, static_cast<std::ptrdiff_t>(n), static_cast<std::ptrdiff_t>(k), alpha, a.data(),
                              static_cast<std::ptrdiff_t>(lda), beta, c.data(), static_cast<std::ptrdiff_t>(ldc)) != 0) {
      return false;
    }
    bool good {true};
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t i {0}; i < n; ++i) {
        const std::size_t at {i + (j * ldc)};
        if (lower ? i < j : i > j) {
          good = good && c[at] == before[at];
          continue;
        }
        double sum {0.0};
        for (std::size_t l {0}; l < k; ++l) {
          sum += element(a, lda, trans, i, l) * element(a, lda, trans, j, l);
        }
        const double want {(alpha * sum) + (beta == 0.0 ? 0.0 : beta * before[at])};
        good = good && std::abs(c[at] - want) <= 1e-12 * static_cast<double>(k + 1);
      }
    }
    return good && std::ranges::equal(c.begin() + static_cast<std::ptrdiff_t>(n * ldc), c.end(),
                                      before.begin() + static_cast<std::ptrdiff_t>(n * ldc), before.end());
  }
} // namespace

TEST(syrk_updates_one_triangle_by_alpha_op_a_op_a_transposed)
{
  for (const char uplo : {'L', 'U', 'l', 'u'}) {
    for (const char trans : {'N', 'T', 'c'}) {
      EXPECT(syrk_matches(uplo, trans, 7, 5, 1.5, 0.5));
      EXPECT(syrk_matches(uplo, trans, 300, 40, -1.0, 0.0)); // three blocks, the last partial; C not read
    }
  }
  EXPECT(syrk_matches('L', 'N', 9, 0, 1.0, 2.0));  // no depth: C ← β C
  EXPECT(syrk_matches('U', 'T', 9, 4, 0.0, 0.0));  // α = 0, β = 0: zeros
  EXPECT(syrk_matches('U', 'N', 9, 4, 0.0, 1.0));  // nothing to do
  EXPECT(syrk_matches('L', 'N', 0, 4, 1.0, 1.0));
}

TEST(syrk_reports_its_first_illegal_argument)
{
  std::vector<double> x(16, 1.0);
  EXPECT_EQ(numerical::blas::syrk('X', 'N', 2, 2, 1.0, x.data(), 2, 0.0, x.data(), 2), 1);
  EXPECT_EQ(numerical::blas::syrk('L', 'X', 2, 2, 1.0, x.data(), 2, 0.0, x.data(), 2), 2);
  EXPECT_EQ(numerical::blas::syrk('L', 'N', -1, 2, 1.0, x.data(), 2, 0.0, x.data(), 2), 3);
  EXPECT_EQ(numerical::blas::syrk('L', 'N', 2, -1, 1.0, x.data(), 2, 0.0, x.data(), 2), 4);
  EXPECT_EQ(numerical::blas::syrk('L', 'N', 3, 2, 1.0, x.data(), 2, 0.0, x.data(), 3), 7);
  EXPECT_EQ(numerical::blas::syrk('L', 'T', 2, 3, 1.0, x.data(), 2, 0.0, x.data(), 2), 7);
  EXPECT_EQ(numerical::blas::syrk('L', 'N', 3, 2, 1.0, x.data(), 3, 0.0, x.data(), 2), 10);
  const int           two   {2};
  const int           bad   {-1};
  const double        one   {1.0};
  const double        zero  {0.0};
  std::vector<double> a     {1.0, 2.0, 3.0, 4.0}; // [[1, 3], [2, 4]]: A Aᵀ = [[10, 14], [14, 20]]
  std::vector<double> c(4, -7.0);
  dsyrk_("L", "N", &two, &two, &one, a.data(), &two, &zero, c.data(), &two, 1, 1);
  EXPECT(c[0] == 10.0 && c[1] == 14.0 && c[2] == -7.0 && c[3] == 20.0);
  dsyrk_("L", "N", &bad, &two, &one, a.data(), &two, &zero, c.data(), &two, 1, 1); // reported, C untouched
  EXPECT(c[0] == 10.0);
}

namespace {
  // Whether gemv(trans) of an m × n A (its leading dimension padded) with vectors incx and incy apart (from the last
  // element where negative) gives α op(A) x + β y by a plain loop, the elements between the vector's left as they were
  // (and y not read where β = 0).
  bool gemv_matches(char           trans,
                    std::size_t    m,
                    std::size_t    n,
                    double         alpha,
                    double         beta,
                    std::ptrdiff_t incx,
                    std::ptrdiff_t incy)
  {
    const bool                t        {trans != 'N' && trans != 'n'};
    const std::size_t         lda      {m + 3};
    const std::size_t         x_length {t ? m : n};
    const std::size_t         y_length {t ? n : m};
    const auto                step_x   {static_cast<std::size_t>(std::abs(incx))};
    const auto                step_y   {static_cast<std::size_t>(std::abs(incy))};
    const std::vector<double> a        {entries(lda * n, 67)};
    const std::vector<double> x        {entries((x_length * step_x) + 1, 71)};
    std::vector<double>       y        {entries((y_length * step_y) + 1, 73)};
    const auto                at       {[](std::size_t i, std::size_t length, std::ptrdiff_t inc) {
                                          return inc > 0 ? i * static_cast<std::size_t>(inc) : (length - 1 - i) * static_cast<std::size_t>(-inc);
                                        }};
    for (std::size_t i {0}; i < y_length && beta == 0.0; ++i) {
      y[at(i, y_length, incy)] = std::numeric_limits<double>::quiet_NaN();
    }
    const std::vector<double> before {y};
    if (numerical::blas::gemv(trans, static_cast<std::ptrdiff_t>(m), static_cast<std::ptrdiff_t>(n), alpha, a.data(),
                              static_cast<std::ptrdiff_t>(lda), x.data(), incx, beta, y.data(), incy) != 0) {
      return false;
    }
    std::vector<bool> touched(y.size(), false);
    bool              good {true};
    for (std::size_t i {0}; i < y_length && m > 0 && n > 0; ++i) {
      double sum {0.0};
      for (std::size_t l {0}; l < x_length; ++l) {
        sum += (t ? a[l + (i * lda)] : a[i + (l * lda)]) * x[at(l, x_length, incx)];
      }
      const std::size_t k    {at(i, y_length, incy)};
      const double      want {(alpha * sum) + (beta == 0.0 ? 0.0 : beta * before[k])};
      touched[k] = true;
      good       = good && std::abs(y[k] - want) <= 1e-12 * static_cast<double>(x_length + 1);
    }
    for (std::size_t k {0}; k < y.size(); ++k) {
      good = good && (touched[k] || y[k] == before[k] || (std::isnan(y[k]) && std::isnan(before[k])));
    }
    return good;
  }
} // namespace

TEST(gemv_computes_alpha_op_a_x_plus_beta_y_with_any_increments)
{
  for (const char trans : {'N', 'T', 'n', 'c'}) {
    EXPECT(gemv_matches(trans, 7, 5, 1.5, 0.5, 1, 1));
    EXPECT(gemv_matches(trans, 7, 5, -1.0, 0.0, 2, -3));   // y not read
    EXPECT(gemv_matches(trans, 7, 5, 2.0, 1.0, -1, 2));
    EXPECT(gemv_matches(trans, 600, 530, 1.0, 1.0, 1, 1)); // several row groups, the last partial
    EXPECT(gemv_matches(trans, 7, 5, 0.0, 2.0, 1, 1));     // α = 0: y ← β y
    EXPECT(gemv_matches(trans, 7, 5, 0.0, 1.0, 1, 1));     // nothing to do
    EXPECT(gemv_matches(trans, 0, 5, 1.0, 0.0, 1, 1));     // empty: y untouched, as the reference leaves it
    EXPECT(gemv_matches(trans, 5, 0, 1.0, 0.0, 1, 1));
  }
}

TEST(gemv_reports_its_first_illegal_argument)
{
  std::vector<double> v(16, 1.0);
  EXPECT_EQ(numerical::blas::gemv('X', 2, 2, 1.0, v.data(), 2, v.data(), 1, 0.0, v.data(), 1), 1);
  EXPECT_EQ(numerical::blas::gemv('N', -1, 2, 1.0, v.data(), 2, v.data(), 1, 0.0, v.data(), 1), 2);
  EXPECT_EQ(numerical::blas::gemv('N', 2, -1, 1.0, v.data(), 2, v.data(), 1, 0.0, v.data(), 1), 3);
  EXPECT_EQ(numerical::blas::gemv('N', 3, 2, 1.0, v.data(), 2, v.data(), 1, 0.0, v.data(), 1), 6);
  EXPECT_EQ(numerical::blas::gemv('N', 2, 2, 1.0, v.data(), 2, v.data(), 0, 0.0, v.data(), 1), 8);
  EXPECT_EQ(numerical::blas::gemv('N', 2, 2, 1.0, v.data(), 2, v.data(), 1, 0.0, v.data(), 0), 11);
  const int           two  {2};
  const int           one  {1};
  const int           bad  {0};
  const double        unit {1.0};
  const double        zero {0.0};
  std::vector<double> a    {1.0, 2.0, 3.0, 4.0}; // [[1, 3], [2, 4]]
  std::vector<double> x    {1.0, 1.0};
  std::vector<double> y(2, -7.0);
  dgemv_("N", &two, &two, &unit, a.data(), &two, x.data(), &one, &zero, y.data(), &one, 1);
  EXPECT(y[0] == 4.0 && y[1] == 6.0);
  dgemv_("N", &two, &two, &unit, a.data(), &two, x.data(), &bad, &zero, y.data(), &one, 1); // reported, y untouched
  EXPECT(y[0] == 4.0);
}

TEST(dot_sums_the_products_with_any_increments)
{
  const auto x        {entries(std::size_t {3} *100000, 79)}; // past one chunk of the parallel sum
  const auto y        {entries(std::size_t {2} *100000, 83)};
  double     plain    {0.0};
  double     strided  {0.0};
  double     reversed {0.0};
  double     mixed    {0.0};
  for (std::size_t i {0}; i < 100000; ++i) {
    plain    += x[i] * y[i];
    strided  += x[3 * i] * y[2 * i];
    reversed += x[3 * i] * y[2 * (99999 - i)];
    mixed    += x[i] * y[2 * i];
  }
  EXPECT(std::abs(numerical::blas::dot(100000, x.data(), 1, y.data(), 1) - plain) < 1e-9);
  EXPECT(std::abs(numerical::blas::dot(100000, x.data(), 3, y.data(), 2) - strided) < 1e-9);
  EXPECT(std::abs(numerical::blas::dot(100000, x.data(), 3, y.data(), -2) - reversed) < 1e-9);
  EXPECT(std::abs(numerical::blas::dot(100000, x.data(), 1, y.data(), 2) - mixed) < 1e-9);
  EXPECT_EQ(numerical::blas::dot(0, x.data(), 1, y.data(), 1), 0.0);
  EXPECT_EQ(numerical::blas::dot(-3, x.data(), 1, y.data(), 1), 0.0);
  const int                 three {3};
  const int                 one   {1};
  const std::vector<double> u     {1.0, 2.0, 3.0};
  const std::vector<double> v     {4.0, 5.0, 6.0};
  EXPECT_EQ(ddot_(&three, u.data(), &one, v.data(), &one), 32.0);
}

namespace {
  // Whether trsm(side, uplo, transa, diag) of an m × n B, the leading dimensions padded, leaves X with op(A) X = α B
  // (left) or X op(A) = α B (right), checked by multiplying back with a plain loop; A's other triangle (and a unit
  // diagonal) NaN, never read.
  bool trsm_solves(char        side,
                   char        uplo,
                   char        transa,
                   char        diag,
                   std::size_t m,
                   std::size_t n,
                   double      alpha)
  {
    const bool          left  {side == 'L' || side == 'l'};
    const bool          lower {uplo == 'L' || uplo == 'l'};
    const bool          unit  {diag == 'U' || diag == 'u'};
    const bool          t     {transa != 'N' && transa != 'n'};
    const std::size_t   k     {left ? m : n};
    const std::size_t   lda   {k + 2};
    const std::size_t   ldb   {m + 1};
    std::vector<double> a     {entries(lda * k, 89)};
    for (std::size_t j {0}; j < k; ++j) {
      for (std::size_t i {0}; i < k; ++i) {
        const bool inside {lower ? i > j : i < j};
        if (i == j) {
          a[i + (j * lda)] = unit ? std::numeric_limits<double>::quiet_NaN() : 2.0 + a[i + (j * lda)]; // well away from 0
        }
        else if (!inside) {
          a[i + (j * lda)] = std::numeric_limits<double>::quiet_NaN();
        }
        else {
          a[i + (j * lda)] /= static_cast<double>(k); // a well-conditioned triangle
        }
      }
    }
    // op(A)(i, j), the triangle's entries alone (a unit diagonal as 1).
    const auto op_a {[&](std::size_t i, std::size_t j) {
                       const std::size_t r {t ? j : i};
                       const std::size_t c {t ? i : j};
                       if (r == c) {
                         return unit ? 1.0 : a[r + (c * lda)];
                       }
                       const bool inside {lower ? r > c : r < c};
                       return inside ? a[r + (c * lda)] : 0.0;
                     }};
    const std::vector<double> b0 {entries(ldb * n, 97)};
    std::vector<double>       x  {b0};
    if (numerical::blas::trsm(side, uplo, transa, diag, static_cast<std::ptrdiff_t>(m), static_cast<std::ptrdiff_t>(n), alpha, a.data(),
                              static_cast<std::ptrdiff_t>(lda), x.data(), static_cast<std::ptrdiff_t>(ldb)) != 0) {
      return false;
    }
    bool good {true};
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t i {0}; i < m; ++i) {
        double product {0.0};
        for (std::size_t l {0}; l < k; ++l) {
          product += left ? op_a(i, l) * x[l + (j * ldb)] : x[i + (l * ldb)] * op_a(l, j);
        }
        good = good && std::abs(product - (alpha * b0[i + (j * ldb)])) <= 1e-12 * static_cast<double>(k);
      }
      good = good && x[m + (j * ldb)] == b0[m + (j * ldb)]; // the padding untouched
    }
    return good;
  }
} // namespace

TEST(trsm_solves_every_side_triangle_and_transposition)
{
  for (const char side : {'L', 'R', 'l', 'r'}) {
    for (const char uplo : {'L', 'U', 'l', 'u'}) {
      for (const char transa : {'N', 'T', 'c'}) {
        for (const char diag : {'N', 'U', 'n', 'u'}) {
          EXPECT(trsm_solves(side, uplo, transa, diag, 9, 6, 1.5));
        }
      }
      EXPECT(trsm_solves(side, uplo, 'N', 'N', 150, 140, -1.0)); // three blocks, the last partial
      EXPECT(trsm_solves(side, uplo, 'T', 'N', 150, 140, 2.0));
    }
  }
}

TEST(trsm_with_alpha_zero_clears_b_and_reports_its_first_illegal_argument)
{
  std::vector<double> nan(4, std::numeric_limits<double>::quiet_NaN()); // A, never read where α = 0
  std::vector<double> b {1.0, 2.0, 3.0, 4.0};
  EXPECT_EQ(numerical::blas::trsm('L', 'L', 'N', 'N', 2, 2, 0.0, nan.data(), 2, b.data(), 2), 0);
  EXPECT(std::ranges::all_of(b, [](double x) { return x == 0.0; }));
  EXPECT_EQ(numerical::blas::trsm('L', 'L', 'N', 'N', 0, 2, 1.0, nan.data(), 1, b.data(), 1), 0);
  EXPECT_EQ(numerical::blas::trsm('R', 'L', 'N', 'N', 2, 0, 1.0, nan.data(), 1, b.data(), 2), 0);
  EXPECT_EQ(numerical::blas::trsm('X', 'L', 'N', 'N', 2, 2, 1.0, b.data(), 2, b.data(), 2), 1);
  EXPECT_EQ(numerical::blas::trsm('L', 'X', 'N', 'N', 2, 2, 1.0, b.data(), 2, b.data(), 2), 2);
  EXPECT_EQ(numerical::blas::trsm('L', 'L', 'X', 'N', 2, 2, 1.0, b.data(), 2, b.data(), 2), 3);
  EXPECT_EQ(numerical::blas::trsm('L', 'L', 'N', 'X', 2, 2, 1.0, b.data(), 2, b.data(), 2), 4);
  EXPECT_EQ(numerical::blas::trsm('L', 'L', 'N', 'N', -1, 2, 1.0, b.data(), 2, b.data(), 2), 5);
  EXPECT_EQ(numerical::blas::trsm('L', 'L', 'N', 'N', 2, -1, 1.0, b.data(), 2, b.data(), 2), 6);
  EXPECT_EQ(numerical::blas::trsm('L', 'L', 'N', 'N', 3, 1, 1.0, b.data(), 2, b.data(), 3), 9);
  EXPECT_EQ(numerical::blas::trsm('R', 'L', 'N', 'N', 1, 3, 1.0, b.data(), 2, b.data(), 1), 9);
  EXPECT_EQ(numerical::blas::trsm('L', 'L', 'N', 'N', 2, 1, 1.0, b.data(), 2, b.data(), 1), 11);
  const int           two  {2};
  const int           bad  {-1};
  const double        unit {1.0};
  std::vector<double> l    {2.0, 1.0, 0.0, 4.0}; // [[2, 0], [1, 4]]
  std::vector<double> y    {2.0, 9.0};           // L x = (2, 9): x = (1, 2)
  const int           one  {1};
  dtrsm_("L", "L", "N", "N", &two, &one, &unit, l.data(), &two, y.data(), &two, 1, 1, 1, 1);
  EXPECT(y[0] == 1.0 && y[1] == 2.0);
  dtrsm_("L", "L", "N", "N", &bad, &one, &unit, l.data(), &two, y.data(), &two, 1, 1, 1, 1); // reported, y untouched
  EXPECT(y[0] == 1.0);
}

namespace {
  // Deterministic entries of the scalar T: entries()'s, two of them a complex number.
  template <typename T>
  std::vector<T> scalars(std::size_t count,
                         std::size_t seed)
  {
    const auto     parts {entries(2 * count, seed)};
    std::vector<T> out(count);
    for (std::size_t i {0}; i < count; ++i) {
      if constexpr (std::is_floating_point_v<T>) {
        out[i] = static_cast<T>(parts[i]);
      }
      else {
        using R = T::value_type;
        out[i]  = T {static_cast<R>(parts[2 * i]), static_cast<R>(parts[(2 * i) + 1])};
      }
    }
    return out;
  }

  // Whether gemm<T>(transa, transb) of m × k by k × n, the leading dimensions padded, matches α op(A) op(B) + β C by
  // a plain loop, op C conjugating, to the type's precision.
  template <typename T>
  bool typed_gemm_matches(char        transa,
                          char        transb,
                          std::size_t m,
                          std::size_t n,
                          std::size_t k,
                          T           alpha,
                          T           beta)
  {
    const bool        ta  {transa != 'N' && transa != 'n'};
    const bool        tb  {transb != 'N' && transb != 'n'};
    const std::size_t lda {(ta ? k : m) + 2};
    const std::size_t ldb {(tb ? n : k) + 1};
    const std::size_t ldc {m + 3};
    const auto        a   {scalars<T>(lda * (ta ? m : k), 101)};
    const auto        b   {scalars<T>(ldb * (tb ? k : n), 103)};
    auto              c   {scalars<T>(ldc * n, 107)};
    const auto        c0  {c};
    const auto        op  {[](const std::vector<T>& x, std::size_t ld, char o, std::size_t i, std::size_t j) {
                             if (o == 'N' || o == 'n') {
                               return x[i + (j * ld)];
                             }
                             const T element {x[j + (i * ld)]};
                             return (o == 'C' || o == 'c') ? numerical::detail::conj_of(element) : element;
                           }};
    if (numerical::blas::gemm(transa, transb, static_cast<std::ptrdiff_t>(m), static_cast<std::ptrdiff_t>(n), static_cast<std::ptrdiff_t>(k),
                              alpha, a.data(), static_cast<std::ptrdiff_t>(lda), b.data(), static_cast<std::ptrdiff_t>(ldb), beta, c.data(),
                              static_cast<std::ptrdiff_t>(ldc))
        != 0) {
      return false;
    }
    const double tol  {(std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>> ? 1e-5 : 1e-13) * static_cast<double>(k + 1) * 4};
    bool         good {true};
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t i {0}; i < m; ++i) {
        std::complex<double> sum {};
        for (std::size_t l {0}; l < k; ++l) {
          sum += std::complex<double> {op(a, lda, transa, i, l)} *std::complex<double> {op(b, ldb, transb, l, j)};
        }
        const std::complex<double> want {(std::complex<double> {alpha} *sum) + (std::complex<double> {beta} *std::complex<double> {c0[i + (j * ldc)]})};
        good = good && std::abs(std::complex<double> {c[i + (j * ldc)]} - want) <= tol;
      }
      good = good && c[m + (j * ldc)] == c0[m + (j * ldc)];
    }
    return good;
  }

  // Whether each argument check of gemm<T> answers, in the reference's order.
  template <typename T>
  bool gemm_checks_in_order(T alpha,
                            T beta)
  {
    std::vector<T> x(64, T {1});
    T* const       p       {x.data()};
    return numerical::blas::gemm('X', 'N', 2, 2, 2, alpha, p, 2, p, 2, beta, p, 2) == 1
           && numerical::blas::gemm('N', 'X', 2, 2, 2, alpha, p, 2, p, 2, beta, p, 2) == 2
           && numerical::blas::gemm('N', 'N', -1, 2, 2, alpha, p, 2, p, 2, beta, p, 2) == 3
           && numerical::blas::gemm('N', 'N', 2, -1, 2, alpha, p, 2, p, 2, beta, p, 2) == 4
           && numerical::blas::gemm('N', 'N', 2, 2, -1, alpha, p, 2, p, 2, beta, p, 2) == 5
           && numerical::blas::gemm('N', 'N', 2, 2, 2, alpha, p, 1, p, 2, beta, p, 2) == 8
           && numerical::blas::gemm('N', 'N', 2, 2, 2, alpha, p, 2, p, 1, beta, p, 2) == 10
           && numerical::blas::gemm('N', 'N', 2, 2, 2, alpha, p, 2, p, 2, beta, p, 1) == 13;
  }

  template <typename T>
  bool every_operation_matches(T alpha,
                               T beta)
  {
    bool good {true};
    for (const char ta : {'N', 'T', 'C', 'c'}) {
      for (const char tb : {'N', 't', 'C'}) {
        good = good && typed_gemm_matches<T>(ta, tb, 7, 5, 6, alpha, beta) && typed_gemm_matches<T>(ta, tb, 70, 90, 80, alpha, beta);
      }
    }
    good = good && typed_gemm_matches<T>('N', 'N', 4, 3, 2, T {}, beta) && typed_gemm_matches<T>('T', 'N', 4, 3, 2, alpha, T {1})
           && typed_gemm_matches<T>('N', 'N', 4, 3, 0, alpha, beta) && typed_gemm_matches<T>('N', 'N', 0, 3, 2, alpha, beta)
           && typed_gemm_matches<T>('N', 'N', 4, 0, 2, alpha, beta);
    return good && gemm_checks_in_order<T>(alpha, beta);
  }
} // namespace

TEST(gemm_of_every_scalar_conjugates_where_asked)
{
  EXPECT(every_operation_matches<float>(1.5F, 0.5F));
  EXPECT(every_operation_matches<double>(1.0, 0.0));
  EXPECT(every_operation_matches<std::complex<float>>({0.5F, -1.0F}, {1.0F, 0.0F}));
  EXPECT(every_operation_matches<std::complex<double>>({0.5, -1.0}, {0.25, 2.0}));
  EXPECT(every_operation_matches<std::complex<double>>({1.0, 0.0}, {0.0, 0.0}));
}

TEST(the_fortran_sgemm_cgemm_zgemm_answer_as_the_cpp_one)
{
  const int                          two  {2};
  const int                          bad  {-1};
  const float                        one  {1.0F};
  const float                        zero {0.0F};
  const std::vector<float>           a    {1.0F, 3.0F, 2.0F, 4.0F}; // [[1, 2], [3, 4]]
  std::vector<float>                 c(4, 9.0F);
  sgemm_("N", "N", &two, &two, &two, &one, a.data(), &two, a.data(), &two, &zero, c.data(), &two, 1, 1);
  EXPECT(c[0] == 7.0F && c[1] == 15.0F && c[2] == 10.0F && c[3] == 22.0F);
  sgemm_("N", "N", &bad, &two, &two, &one, a.data(), &two, a.data(), &two, &zero, c.data(), &two, 1, 1);
  EXPECT(c[0] == 7.0F);
  const std::complex<double>              unit   {1.0, 0.0};
  const std::complex<double>              none   {};
  const std::vector<std::complex<double>> z      {{0.0, 1.0}}; // i: i·conj(i) = 1, i·i = −1
  std::vector<std::complex<double>>       w(1);
  const int                               single {1};
  zgemm_("N", "C", &single, &single, &single, &unit, z.data(), &single, z.data(), &single, &none, w.data(), &single, 1, 1);
  EXPECT(w[0] == unit);
  zgemm_("N", "T", &single, &single, &single, &unit, z.data(), &single, z.data(), &single, &none, w.data(), &single, 1, 1);
  EXPECT(w[0] == -unit);
  const std::complex<float>              unit_f {1.0F, 0.0F};
  const std::complex<float>              none_f {};
  const std::vector<std::complex<float>> zf     {{0.0F, 1.0F}};
  std::vector<std::complex<float>>       wf(1);
  cgemm_("C", "N", &single, &single, &single, &unit_f, zf.data(), &single, zf.data(), &single, &none_f, wf.data(), &single, 1, 1);
  EXPECT(wf[0] == unit_f);
}

namespace {
  // The tolerance of a residual of order n in the scalar T: its precision times the order, with room.
  template <typename T>
  double precision_of(std::size_t n)
  {
    const bool single {std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>};
    return (single ? 1e-4 : 1e-11) * static_cast<double>(n);
  }

  // Whether getrf<T> of an m × n A gives P A = L U as LAPACK lays the factors and interchanges out, and, square,
  // gesv<T> solves A X = B and getrs<T> each op(A) X = B from those factors.
  template <typename T>
  bool typed_lu_holds(std::size_t m,
                      std::size_t n)
  {
    const std::size_t lda {m + 1};
    const auto        a0  {scalars<T>(lda * n, 109)};
    auto              f   {a0};
    std::vector<int>  ipiv(std::min(m, n));
    if (numerical::blas::getrf(static_cast<std::ptrdiff_t>(m), static_cast<std::ptrdiff_t>(n), f.data(), static_cast<std::ptrdiff_t>(lda),
                               ipiv.data())
        != 0) {
      return false;
    }
    auto pa {a0}; // P A: the interchanges applied in order
    for (std::size_t i {0}; i < ipiv.size(); ++i) {
      for (std::size_t c {0}; c < n; ++c) {
        std::swap(pa[i + (c * lda)], pa[static_cast<std::size_t>(ipiv[i] - 1) + (c * lda)]);
      }
    }
    bool good {true};
    for (std::size_t i {0}; i < m; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        std::complex<double> sum {};
        for (std::size_t l {0}; l <= std::min({i, j, std::min(m, n) - 1}); ++l) {
          const std::complex<double> lower {l == i ? std::complex<double> {1.0} : std::complex<double> {f[i + (l * lda)]}};
          sum += lower * std::complex<double> {f[l + (j * lda)]};
        }
        good = good && std::abs(sum - std::complex<double> {pa[i + (j * lda)]}) <= precision_of<T>(n);
      }
    }
    if (m != n) {
      return good;
    }
    const std::size_t ldb {n + 2};
    const auto        b0  {scalars<T>(ldb * 3, 113)};
    for (const char trans : {'N', 'T', 'C'}) {
      auto x {b0};
      good = good && numerical::blas::getrs(trans, static_cast<std::ptrdiff_t>(n), 3, f.data(), static_cast<std::ptrdiff_t>(lda), ipiv.data(),
                                            x.data(), static_cast<std::ptrdiff_t>(ldb)) == 0;
      for (std::size_t c {0}; c < 3; ++c) {
        for (std::size_t i {0}; i < n; ++i) {
          std::complex<double> sum {};
          for (std::size_t l {0}; l < n; ++l) {
            const T element {trans == 'N' ? a0[i + (l * lda)] : a0[l + (i * lda)]};
            sum += std::complex<double> {trans == 'C' ? numerical::detail::conj_of(element) : element} *std::complex<double> {x[l + (c * ldb)]};
          }
          good = good && std::abs(sum - std::complex<double> {b0[i + (c * ldb)]}) <= precision_of<T>(n) * 10;
        }
      }
    }
    auto             g {a0};
    auto             x {b0};
    std::vector<int> pivots(n);
    good = good && numerical::blas::gesv(static_cast<std::ptrdiff_t>(n), 3, g.data(), static_cast<std::ptrdiff_t>(lda), pivots.data(), x.data(),
                                         static_cast<std::ptrdiff_t>(ldb)) == 0
           && pivots == ipiv;
    for (std::size_t c {0}; c < 3; ++c) {
      for (std::size_t i {0}; i < n; ++i) {
        std::complex<double> sum {};
        for (std::size_t l {0}; l < n; ++l) {
          sum += std::complex<double> {a0[i + (l * lda)]} *std::complex<double> {x[l + (c * ldb)]};
        }
        good = good && std::abs(sum - std::complex<double> {b0[i + (c * ldb)]}) <= precision_of<T>(n) * 10;
      }
    }
    return good;
  }

  // Whether gesv<T> and getrf<T> report a zero pivot, an illegal argument and a NaN, as for double.
  template <typename T>
  bool typed_lu_reports()
  {
    std::vector<T>   a    {T {1}, T {2}, T {}, T {}}; // second column zero: U(2, 2) = 0
    std::vector<T>   b    {T {5}, T {6}};
    std::vector<int> ipiv(2);
    bool             good {numerical::blas::gesv(2, 1, a.data(), 2, ipiv.data(), b.data(), 2) == 2 && b[0] == T {5}};
    good = good && numerical::blas::gesv(-1, 1, a.data(), 2, ipiv.data(), b.data(), 2) == -1;
    std::vector<T> nan    {T {1}, T {std::numeric_limits<float>::quiet_NaN()}, T {}, T {1}};
    good = good && numerical::blas::gesv(2, 1, nan.data(), 2, ipiv.data(), b.data(), 2) == -3;
    good = good && numerical::blas::getrf(2, 2, nan.data(), 2, ipiv.data()) == -3;
    std::vector<T> tall   {T {}, T {}, T {}, T {1}, T {2}, T {3}}; // 3 × 2, first column zero
    good = good && numerical::blas::getrf(3, 2, tall.data(), 3, ipiv.data()) == 1;
    std::vector<T> square {T {1}, T {2}, T {}, T {}};
    good = good && numerical::blas::getrf(2, 2, square.data(), 2, ipiv.data()) == 2;
    // Every argument check and the empty sizes, for this scalar too.
    std::vector<T> x(16, T {1});
    T* const       p       {x.data()};
    int* const     q       {ipiv.data()};
    good = good && numerical::blas::gesv(2, -1, p, 2, q, p, 2) == -2 && numerical::blas::gesv(2, 1, p, 1, q, p, 2) == -4
           && numerical::blas::gesv(2, 1, p, 2, q, p, 1) == -7 && numerical::blas::gesv(0, 1, p, 1, q, p, 1) == 0;
    std::vector<T> identity {T {1}, T {}, T {}, T {1}};
    good = good && numerical::blas::gesv(2, 0, identity.data(), 2, q, p, 2) == 0;
    good = good && numerical::blas::getrf(-1, 2, p, 2, q) == -1 && numerical::blas::getrf(2, -1, p, 2, q) == -2
           && numerical::blas::getrf(2, 2, p, 1, q) == -4 && numerical::blas::getrf(0, 2, p, 1, q) == 0;
    const std::vector<int> unit {1, 2};
    good = good && numerical::blas::getrs('X', 2, 1, p, 2, unit.data(), p, 2) == -1 && numerical::blas::getrs('N', -1, 1, p, 2, unit.data(), p, 2) == -2
           && numerical::blas::getrs('N', 2, -1, p, 2, unit.data(), p, 2) == -3 && numerical::blas::getrs('N', 2, 1, p, 1, unit.data(), p, 2) == -5
           && numerical::blas::getrs('N', 2, 1, p, 2, unit.data(), p, 1) == -8;
    return good;
  }
} // namespace

TEST(lu_routines_of_every_scalar_factorise_and_solve)
{
  EXPECT(typed_lu_holds<float>(40, 40) && typed_lu_holds<float>(9, 5) && typed_lu_holds<float>(5, 9));
  EXPECT(typed_lu_holds<std::complex<float>>(40, 40) && typed_lu_holds<std::complex<float>>(9, 5));
  EXPECT(typed_lu_holds<std::complex<double>>(130, 130) && typed_lu_holds<std::complex<double>>(5, 9));
  EXPECT(typed_lu_holds<double>(40, 40));
  EXPECT(typed_lu_reports<float>() && typed_lu_reports<std::complex<float>>() && typed_lu_reports<std::complex<double>>());
  std::vector<std::complex<double>> imaginary_infinity {{1.0, std::numeric_limits<double>::infinity()}}; // the real part finite
  std::vector<int>                  ipiv(1);
  EXPECT_EQ(numerical::blas::getrf(1, 1, imaginary_infinity.data(), 1, ipiv.data()), -3);
}

TEST(the_fortran_lu_routines_of_every_scalar_answer_as_the_cpp_ones)
{
  const int                         two  {2};
  const int                         one  {1};
  const int                         bad  {-1};
  int                               info {7};
  std::vector<int>                  ipiv(2);
  std::vector<std::complex<double>> a    {{4.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}};
  std::vector<std::complex<double>> b    {{2.0, 2.0}, {1.0, 1.0}}; // (1 + i) (2, 1): x = (0.5 + 0.5i, 0)
  zgesv_(&two, &one, a.data(), &two, ipiv.data(), b.data(), &two, &info);
  EXPECT(info == 0 && std::abs(b[0] - std::complex<double> {0.5, 0.5}) < 1e-15 && std::abs(b[1]) < 1e-15);
  std::vector<std::complex<double>> c {{2.0, 2.0}, {1.0, 1.0}};
  zgetrs_("C", &two, &one, a.data(), &two, ipiv.data(), c.data(), &two, &info, 1); // A is real symmetric: Aᴴ = A
  EXPECT(info == 0 && std::abs(c[0] - std::complex<double> {0.5, 0.5}) < 1e-15);
  zgetrf_(&bad, &two, a.data(), &two, ipiv.data(), &info);
  EXPECT_EQ(info, -1);
  std::vector<float> f {4.0F, 2.0F, 2.0F, 3.0F};
  std::vector<float> y {2.0F, 1.0F};
  sgesv_(&two, &one, f.data(), &two, ipiv.data(), y.data(), &two, &info);
  EXPECT(info == 0 && std::abs(y[0] - 0.5F) < 1e-6F);
  sgetrs_("N", &two, &one, f.data(), &two, ipiv.data(), y.data(), &two, &info, 1);
  sgetrf_(&two, &two, f.data(), &two, ipiv.data(), &info);
  EXPECT_EQ(info, 0);
  std::vector<std::complex<float>> g {{4.0F, 0.0F}, {2.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}};
  std::vector<std::complex<float>> h {{2.0F, 0.0F}, {1.0F, 0.0F}};
  cgesv_(&two, &one, g.data(), &two, ipiv.data(), h.data(), &two, &info);
  EXPECT(info == 0 && std::abs(h[0] - std::complex<float> {0.5F, 0.0F}) < 1e-6F);
  cgetrs_("T", &two, &one, g.data(), &two, ipiv.data(), h.data(), &two, &info, 1);
  cgetrf_(&bad, &two, g.data(), &two, ipiv.data(), &info);
  EXPECT_EQ(info, -1);
  sgesv_(&bad, &one, f.data(), &two, ipiv.data(), y.data(), &two, &info);
  EXPECT_EQ(info, -1);
}

namespace {
  // Whether potrf<T>(uplo) of the Hermitian positive definite Bᴴ B + n I (its other triangle NaN, never read) gives
  // L Lᴴ (or Uᴴ U) = A, and potrs<T> from it solves A X = B.
  template <typename T>
  bool typed_cholesky_holds(char        uplo,
                            std::size_t n)
  {
    const bool        lower {uplo == 'L' || uplo == 'l'};
    const std::size_t lda   {n + 1};
    const auto        g     {scalars<T>(n * n, 127)};
    std::vector<T>    a(lda * n);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        T sum {i == j ? T {static_cast<float>(n)} : T {}};
        for (std::size_t l {0}; l < n; ++l) {
          sum += numerical::detail::conj_of(g[l + (i * n)]) * g[l + (j * n)];
        }
        a[i + (j * lda)] = sum;
      }
    }
    const auto a0 {a};
    for (std::size_t j {0}; j < n; ++j) {
      for (std::size_t i {0}; i < n; ++i) {
        if (lower ? i < j : i > j) {
          a[i + (j * lda)] = T {std::numeric_limits<float>::quiet_NaN()};
        }
      }
    }
    if (numerical::blas::potrf(uplo, static_cast<std::ptrdiff_t>(n), a.data(), static_cast<std::ptrdiff_t>(lda)) != 0) {
      return false;
    }
    // L(i, j) from the triangle written.
    const auto l {[&](std::size_t i, std::size_t j) {
                    if (j > i) {
                      return std::complex<double> {};
                    }
                    return std::complex<double> {lower ? a[i + (j * lda)] : numerical::detail::conj_of(a[j + (i * lda)])};
                  }};
    bool good {true};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        std::complex<double> sum {};
        for (std::size_t k {0}; k < n; ++k) {
          sum += l(i, k) * std::conj(l(j, k));
        }
        good = good && std::abs(sum - std::complex<double> {a0[i + (j * lda)]}) <= precision_of<T>(n) * static_cast<double>(n);
      }
    }
    const std::size_t ldb {n + 2};
    const auto        b0  {scalars<T>(ldb * 2, 131)};
    auto              x   {b0};
    good = good && numerical::blas::potrs(uplo, static_cast<std::ptrdiff_t>(n), 2, a.data(), static_cast<std::ptrdiff_t>(lda), x.data(),
                                          static_cast<std::ptrdiff_t>(ldb)) == 0;
    for (std::size_t c {0}; c < 2; ++c) {
      for (std::size_t i {0}; i < n; ++i) {
        std::complex<double> sum {};
        for (std::size_t k {0}; k < n; ++k) {
          sum += std::complex<double> {a0[i + (k * lda)]} *std::complex<double> {x[k + (c * ldb)]};
        }
        good = good && std::abs(sum - std::complex<double> {b0[i + (c * ldb)]}) <= precision_of<T>(n) * static_cast<double>(n);
      }
    }
    return good;
  }

  // Whether potrf<T> and potrs<T> report a failing minor, a NaN and every illegal argument, as for double.
  template <typename T>
  bool typed_cholesky_reports()
  {
    std::vector<T> indefinite {T {1}, T {2}, T {2}, T {1}};
    std::vector<T> nan        {T {std::numeric_limits<float>::quiet_NaN()}};
    std::vector<T> x(16, T    {1});
    T* const       p          {x.data()};
    return numerical::blas::potrf('L', 2, indefinite.data(), 2) == 2 && numerical::blas::potrf('U', 1, nan.data(), 1) == -3
           && numerical::blas::potrf('X', 2, p, 2) == -1 && numerical::blas::potrf('L', -1, p, 2) == -2 && numerical::blas::potrf('L', 2, p, 1) == -4
           && numerical::blas::potrf('L', 0, p, 1) == 0 && numerical::blas::potrs('X', 2, 1, p, 2, p, 2) == -1
           && numerical::blas::potrs('L', -1, 1, p, 2, p, 2) == -2 && numerical::blas::potrs('L', 2, -1, p, 2, p, 2) == -3
           && numerical::blas::potrs('U', 2, 1, p, 1, p, 2) == -5 && numerical::blas::potrs('u', 2, 1, p, 2, p, 1) == -7;
  }
} // namespace

TEST(cholesky_routines_of_every_scalar_factorise_either_triangle_and_solve)
{
  for (const char uplo : {'L', 'U', 'l', 'u'}) {
    EXPECT(typed_cholesky_holds<float>(uplo, 30) && typed_cholesky_holds<std::complex<float>>(uplo, 30));
    EXPECT(typed_cholesky_holds<std::complex<double>>(uplo, 70) && typed_cholesky_holds<double>(uplo, 20));
  }
  EXPECT(typed_cholesky_reports<float>() && typed_cholesky_reports<std::complex<float>>() && typed_cholesky_reports<std::complex<double>>());
}

TEST(the_fortran_cholesky_routines_of_every_scalar_answer_as_the_cpp_ones)
{
  const int                         two  {2};
  const int                         one  {1};
  const int                         bad  {-1};
  int                               info {7};
  std::vector<std::complex<double>> z    {{4.0, 0.0}, {0.0, 2.0}, {0.0, -2.0}, {3.0, 0.0}}; // [[4, −2i], [2i, 3]]
  zpotrf_("L", &two, z.data(), &two, &info, 1);
  EXPECT(info == 0 && z[0] == std::complex<double>(2.0, 0.0) && std::abs(z[1] - std::complex<double>(0.0, 1.0)) < 1e-15);
  std::vector<std::complex<double>> y {{6.0, 0.0}, {0.0, 5.0}};                             // A (1, i) = (4 − 2i·i, 2i + 3i)
  zpotrs_("L", &two, &one, z.data(), &two, y.data(), &two, &info, 1);
  EXPECT(info == 0 && std::abs(y[0] - std::complex<double> {1.0, 0.0}) < 1e-15 && std::abs(y[1] - std::complex<double> {0.0, 1.0}) < 1e-15);
  std::vector<float> f {4.0F, 2.0F, 2.0F, 3.0F};
  spotrf_("U", &two, f.data(), &two, &info, 1);
  std::vector<float> w {2.0F, 1.0F};
  spotrs_("U", &two, &one, f.data(), &two, w.data(), &two, &info, 1);
  EXPECT(info == 0 && std::abs(w[0] - 0.5F) < 1e-6F && std::abs(w[1]) < 1e-6F);
  std::vector<std::complex<float>> c {{4.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F}, {9.0F, 0.0F}};
  cpotrf_("L", &two, c.data(), &two, &info, 1);
  std::vector<std::complex<float>> v {{4.0F, 0.0F}, {9.0F, 0.0F}};
  cpotrs_("L", &two, &one, c.data(), &two, v.data(), &two, &info, 1);
  EXPECT(info == 0 && v[0] == std::complex<float>(1.0F, 0.0F) && v[1] == std::complex<float>(1.0F, 0.0F));
  cpotrf_("L", &bad, c.data(), &two, &info, 1);
  EXPECT_EQ(info, -2);
}
