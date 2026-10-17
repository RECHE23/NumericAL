// The BLAS/LAPACK stand-ins (numerical/blas.hpp and the Fortran shim of blas/numerical_blas.cpp): column-major
// semantics as the reference routines define them, checked against plain loops and against the factorisations'
// defining properties; every illegal argument reported by its position.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
