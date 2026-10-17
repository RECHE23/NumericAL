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
