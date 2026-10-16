// The general eigendecomposition: residuals at rounding on random real and complex matrices, unit eigenvectors with
// their largest component real, a real matrix's conjugate pairs; the matrices that strain it (triangular, defective,
// cyclic, graded, a complex pair nearly real); and its pieces alone: the balancing, the 2 × 2 standard form, the small
// reflections, the bounded iterations.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::eigen_decomposition;
  using numerical::matrix;
  using C = std::complex<double>;

  //! \brief `|(A V)(r, c) − λ_c V(r, c)|`.
  template <typename T>
  double entry_residual(const matrix<T>&              a,
                        const eigen_decomposition<T>& e,
                        std::size_t                   r,
                        std::size_t                   c)
  {
    C sum {};
    for (std::size_t k {0}; k < a.rows(); ++k) {
      sum += e.vectors()(k, c) * a(r, k);
    }
    return std::abs(sum - (e.values()[c] * e.vectors()(r, c)));
  }

  //! \brief `max |A v − λ v|` over the pairs, relative to `n ‖A‖_max ε`; 1 where a vector is not of unit norm with its
  //!        largest component real.
  template <typename T>
  double residual(const matrix<T>&              a,
                  const eigen_decomposition<T>& e)
  {
    const std::size_t n    {a.rows()};
    double            size {0.0};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        size = std::max(size, static_cast<double>(std::abs(a(i, j))));
      }
    }
    double worst {0.0};
    for (std::size_t c {0}; c < n; ++c) {
      double      norm    {0.0};
      std::size_t largest {0};
      for (std::size_t r {0}; r < n; ++r) {
        worst  = std::max(worst, entry_residual(a, e, r, c));
        norm  += std::norm(e.vectors()(r, c));
                           if (std::norm(e.vectors()(r, c)) > std::norm(e.vectors()(largest, c))) {
          largest = r;
        }
      }
      if (std::abs(std::sqrt(norm) - 1.0) > 1e-14 || e.vectors()(largest, c).imag() != 0.0) {
        return 1.0; // not normalised as LAPACK's
      }
    }
    return worst / (std::max(size, 1e-300) * static_cast<double>(n) * 2.220446049250313e-16);
  }

  matrix<double> random_real(std::size_t      n,
                             std::mt19937_64& g)
  {
    std::normal_distribution<double> d;
    std::vector<double>              x(n * n);
    for (double& v : x) {
      v = d(g);
    }
    return matrix<double> {n, n, x};
  }

  matrix<C> random_complex(std::size_t      n,
                           std::mt19937_64& g)
  {
    std::normal_distribution<double> d;
    std::vector<C>                   x(n * n);
    for (C& v : x) {
      v = C {d(g), d(g)};
    }
    return matrix<C> {n, n, x};
  }

  //! \brief Whether the eigenvalues of \p e are \p want, in any order, within \p tolerance.
  template <typename T>
  bool same_values(const eigen_decomposition<T>& e,
                   std::vector<C>                want,
                   double                        tolerance)
  {
    std::vector<C> got(e.values().begin(), e.values().end());
    const auto     order {[](const C& x, const C& y) { return x.real() != y.real() ? x.real() < y.real() : x.imag() < y.imag(); }};
    std::sort(got.begin(), got.end(), order);
    std::sort(want.begin(), want.end(), order);
    bool same {got.size() == want.size()};
    for (std::size_t i {0}; same && i < got.size(); ++i) {
      same = std::abs(got[i] - want[i]) <= tolerance;
    }
    return same;
  }
} // namespace

TEST(eig_reaches_rounding_on_random_matrices)
{
  // Fixed seed: the matrices are part of the test, so a failure names the same one every run.
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64 g {7};
  for (const std::size_t n : {1U, 2U, 3U, 4U, 7U, 16U, 33U, 70U}) {
    const matrix<double>              a {random_real(n, g)};
    const eigen_decomposition<double> e {a};
    EXPECT(residual(a, e) < 10.0);
    const matrix<C>              b      {random_complex(n, g)};
    const eigen_decomposition<C> f      {b};
    EXPECT(residual(b, f) < 10.0);
    EXPECT(!f.real());
    const numerical::vector<C> alone {numerical::eigenvalues(b)};
    for (std::size_t i {0}; i < n; ++i) {
      EXPECT_EQ(alone[i], f.values()[i]); // the same Schur form, with or without the vectors
    }
  }
}

TEST(eig_of_a_real_matrix_gives_conjugate_pairs)
{
  const matrix<double>              rotation {{0.0, -1.0}, {1.0, 0.0}};
  const eigen_decomposition<double> e        {rotation};
  EXPECT(!e.real());
  EXPECT_EQ(e.values()[0], C(0.0, 1.0)); // the positive imaginary part first
  EXPECT_EQ(e.values()[1], C(0.0, -1.0));
  for (std::size_t r {0}; r < 2; ++r) {
    EXPECT_EQ(e.vectors()(r, 1), std::conj(e.vectors()(r, 0)));
  }
  EXPECT(residual(rotation, e) < 10.0);
  const matrix<double>              real_values {{2.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 4.0}};
  const eigen_decomposition<double> s           {real_values};
  EXPECT(s.real());
  for (std::size_t r {0}; r < 3; ++r) {
    EXPECT_EQ(s.vectors()(r, 0).imag(), 0.0);
  }
  // a pair whose block has its larger entry below the diagonal, then above: either start of the pair's vector
  const matrix<double> below {{1.0, -0.5, 3.0}, {8.0, 1.0, -2.0}, {0.0, 0.0, 5.0}};
  const matrix<double> above {{1.0, -8.0, 3.0}, {0.5, 1.0, -2.0}, {0.0, 0.0, 5.0}};
  EXPECT(residual(below, eigen_decomposition<double> {below}) < 10.0);
  EXPECT(residual(above, eigen_decomposition<double> {above}) < 10.0);
}

TEST(eig_isolates_triangular_matrices_exactly)
{
  const matrix<double> upper {{1.0, 2.0, 3.0}, {0.0, 4.0, 5.0}, {0.0, 0.0, 6.0}};
  const matrix<double> lower {{1.0, 0.0, 0.0}, {2.0, 4.0, 0.0}, {3.0, 5.0, 6.0}};
  for (const matrix<double>* a : {&upper, &lower}) {
    const eigen_decomposition<double> e {*a};
    EXPECT(same_values(e, {1.0, 4.0, 6.0}, 0.0));
    EXPECT(residual(*a, e) < 10.0);
  }
  const matrix<C>              cupper {{C(1.0, 1.0), C(2.0, 0.0)}, {C(0.0, 0.0), C(3.0, -1.0)}};
  const eigen_decomposition<C> f      {cupper};
  EXPECT(same_values(f, {C(1.0, 1.0), C(3.0, -1.0)}, 0.0));
  EXPECT(residual(cupper, f) < 10.0);
}

TEST(eig_of_degenerate_matrices)
{
  const eigen_decomposition<double> empty {matrix<double> {0, 0, std::vector<double> {}}};
  EXPECT_EQ(empty.values().size(), 0U);
  const matrix<double>              zero  {3, 3, std::vector<double>(9, 0.0)};
  const eigen_decomposition<double> z     {zero};
  EXPECT(same_values(z, {0.0, 0.0, 0.0}, 0.0));
  EXPECT(residual(zero, z) < 10.0);
  // a Jordan block: one eigenvector, as often as its order; the back-substitution's growth, 1/ε a step, scaled down
  const std::size_t   n {40};
  std::vector<double> jordan(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    jordan[(i * n) + i] = 2.0;
    if (i + 1 < n) {
      jordan[(i * n) + i + 1] = 1.0;
    }
  }
  const eigen_decomposition<double> j {matrix<double> {n, n, jordan}};
  for (std::size_t c {0}; c < n; ++c) {
    EXPECT(std::isfinite(j.vectors()(0, c).real()));
           EXPECT_EQ(j.values()[c], C(2.0));
  }
  std::vector<C>               cjordan(jordan.begin(), jordan.end());
  const eigen_decomposition<C> cj {matrix<C> {n, n, cjordan}};
  EXPECT(std::isfinite(cj.vectors()(0, n - 1).real()));
}

TEST(eig_converges_where_the_standard_shifts_stall)
{
  // The cyclic permutation: every standard shift is zero and gains nothing; the exceptional shifts move it.
  for (const std::size_t n : {3U, 4U, 6U, 9U, 19U}) { // 19: twenty iterations, the second exceptional shift
    std::vector<double> cyclic(n * n, 0.0);
    for (std::size_t i {0}; i + 1 < n; ++i) {
      cyclic[((i + 1) * n) + i] = 1.0;
    }
    cyclic[n - 1] = 1.0;
    const matrix<double>              a {n, n, cyclic};
    const eigen_decomposition<double> e {a};
    EXPECT(residual(a, e) < 50.0);
    const matrix<C>              b      {n, n, std::vector<C>(cyclic.begin(), cyclic.end())};
    const eigen_decomposition<C> f      {b};
    EXPECT(residual(b, f) < 50.0);
  }
}

TEST(eig_balances_graded_matrices)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64                  g {3};
  std::normal_distribution<double> d;
  const std::size_t                n {12};
  std::vector<double>              graded(n * n);
  for (std::size_t i {0}; i < n; ++i) {
    for (std::size_t j {0}; j < n; ++j) {
      graded[(i * n) + j] = d(g) * std::pow(10.0, 1.5 * (static_cast<double>(i) - static_cast<double>(j)));
    }
  }
  const eigen_decomposition<double> e {matrix<double> {n, n, graded}};
  EXPECT(std::isfinite(e.values()[0].real()));
  // entries 1e-300 apart: the scaling stops at its limit
  const matrix<double>              far  {{0.0, 1.0}, {1e-300, 0.0}};
  const eigen_decomposition<double> f    {far};
  EXPECT(same_values(f, {-1e-150, 1e-150}, 1e-164));
  const matrix<double> far_below         {{0.0, 1e-300}, {1.0, 0.0}};
  EXPECT(same_values(eigen_decomposition<double> {far_below}, {-1e-150, 1e-150}, 1e-164));
  const matrix<double>              huge {{1e300, 2e300}, {-3e300, 1e300}};
  EXPECT(residual(huge, eigen_decomposition<double> {huge}) < 10.0);
}

TEST(eig_of_complex_pairs_nearly_real)
{
  // 1 ± 1e-20 i, each block a near-multiple of the identity, above a real eigenvalue 1 and below one another: the
  // back-substitution's pivots fall to their floor and its components are scaled down.
  const std::size_t   blocks {12};
  const std::size_t   n      {(2 * blocks) + 1};
  std::vector<double> t(n * n, 0.0);
  for (std::size_t b {0}; b < blocks; ++b) {
    const std::size_t k {2 * b};
    t[(k * n) + k]             = 1.0;
    t[((k + 1) * n) + k + 1]   = 1.0;
    t[(k * n) + k + 1]         = 1e-20;
    t[((k + 1) * n) + k]       = -1e-20;
    for (std::size_t j {k + 2}; j < n; ++j) {
      t[(k * n) + j]       = 1.0;
      t[((k + 1) * n) + j] = 1.0;
    }
  }
  t[(n * n) - 1] = 1.0;
  const eigen_decomposition<double> e {matrix<double> {n, n, t}};
  for (std::size_t c {0}; c < n; ++c) {
    EXPECT(std::isfinite(e.vectors()(0, c).real()) && std::isfinite(e.vectors()(0, c).imag()));
  }
}

TEST(eig_rejects_what_it_cannot_decompose)
{
  EXPECT_THROWS((eigen_decomposition<double> {matrix<double> {2, 3, std::vector<double>(6, 1.0)}}), numerical::dimension_error);
  EXPECT_THROWS((eigen_decomposition<C> {matrix<C> {2, 3, std::vector<C>(6, C(1.0))}}), numerical::dimension_error);
  EXPECT_THROWS((eigen_decomposition<double> {matrix<double> {{1.0, std::nan("")}, {0.0, 1.0}}}), numerical::non_finite_error);
  numerical::buffer<double>                     h {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 7.0, 8.0};
  numerical::buffer<double>                     re(3, 0.0);
  numerical::buffer<double>                     im(3, 0.0);
  const numerical::detail::schur_target<double> t {.h = &h, .n = 3, .low = 0, .high = 3, .full = false, .zt = nullptr};
  EXPECT_THROWS(numerical::detail::real_schur(t, re, im, 0), numerical::convergence_error);
  numerical::buffer<C>                     hc     {C(1.0), C(2.0), C(3.0), C(4.0), C(5.0), C(6.0), C(0.0), C(7.0), C(8.0)};
  numerical::buffer<C>                     w(3);
  const numerical::detail::schur_target<C> tc     {.h = &hc, .n = 3, .low = 0, .high = 3, .full = false, .zt = nullptr};
  EXPECT_THROWS(numerical::detail::complex_schur(tc, w, 0), numerical::convergence_error);
  // A NaN never deflates: through both exceptional shifts, to the bound
  numerical::buffer<double>                     never {1.0, 2.0, 3.0, std::nan(""), 5.0, 6.0, 0.0, 7.0, 8.0};
  const numerical::detail::schur_target<double> tn    {.h = &never, .n = 3, .low = 0, .high = 3, .full = true, .zt = nullptr};
  EXPECT_THROWS(numerical::detail::real_schur(tn, re, im, 25), numerical::convergence_error);
  numerical::buffer<C>                     cnever     {C(1.0), C(2.0), C(3.0), C(std::nan("")), C(5.0), C(6.0), C(0.0), C(7.0), C(8.0)};
  const numerical::detail::schur_target<C> tcn        {.h = &cnever, .n = 3, .low = 0, .high = 3, .full = true, .zt = nullptr};
  EXPECT_THROWS(numerical::detail::complex_schur(tcn, w, 25), numerical::convergence_error);
}

TEST(eigenvalues_alone_are_the_decompositions)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64 g {21};
  for (const std::size_t n : {0U, 2U, 5U, 12U}) {
    const matrix<double>              a     {random_real(n, g)};
    const numerical::vector<C>        alone {numerical::eigenvalues(a)};
    const eigen_decomposition<double> e     {a};
    for (std::size_t i {0}; i < n; ++i) {
      EXPECT_EQ(alone[i], e.values()[i]);
    }
  }
  const matrix<double> rotation {{0.0, -1.0}, {1.0, 0.0}};
  EXPECT_EQ(numerical::eigenvalues(rotation)[0], C(0.0, 1.0));
}

TEST(eig_of_complex_matrices_that_strain_it)
{
  const eigen_decomposition<C> empty    {matrix<C> {0, 0, std::vector<C> {}}};
  EXPECT_EQ(empty.values().size(), 0U);
  const matrix<C>              zero     {2, 2, std::vector<C>(4, C {})};
  EXPECT(same_values(eigen_decomposition<C> {zero}, {C {}, C {}}, 0.0));
  const matrix<C> tiny_subdiagonal      {{C(1.0), C(2.0)}, {C(1e-300), C(3.0)}};
  EXPECT(residual(tiny_subdiagonal, eigen_decomposition<C> {tiny_subdiagonal}) < 10.0);
  const matrix<C> equal_diagonal        {{C(1.0), C(2.0)}, {C(3.0), C(1.0)}}; // the shift's half-difference is zero
  EXPECT(residual(equal_diagonal, eigen_decomposition<C> {equal_diagonal}) < 10.0);
  const matrix<C> far                   {{C(0.0), C(1.0)}, {C(1e-300), C(0.0)}};
  EXPECT(same_values(eigen_decomposition<C> {far}, {C(-1e-150), C(1e-150)}, 1e-164));
  const matrix<C> far_below             {{C(0.0), C(1e-300)}, {C(1.0), C(0.0)}};
  EXPECT(same_values(eigen_decomposition<C> {far_below}, {C(-1e-150), C(1e-150)}, 1e-164));
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64                     g {5};
  std::uniform_int_distribution<int>  u(-2, 2);
  for (int trial {0}; trial < 200; ++trial) { // small integers: exact cancellations, repeated eigenvalues
    const std::size_t n {2 + static_cast<std::size_t>(trial % 6)};
    std::vector<C>    y(n * n);
    for (C& v : y) {
      v = C(static_cast<double>(u(g)), static_cast<double>(u(g)));
    }
    const matrix<C>              b {n, n, y};
    const eigen_decomposition<C> f {b};
    EXPECT(std::isfinite(f.values()[0].real()));
  }
}

TEST(eig_isolates_a_column_as_well_as_a_row)
{
  // no row lacks an off-diagonal entry, but the first column does: its eigenvalue goes to the top
  const matrix<double>              a {{1.0, 1.0, 1.0}, {0.0, 2.0, 1.0}, {0.0, 1.0, 2.0}};
  const eigen_decomposition<double> e {a};
  EXPECT(same_values(e, {1.0, 1.0, 3.0}, 1e-15));
  EXPECT(residual(a, e) < 10.0);
  const matrix<C> b {{C(1.0), C(1.0), C(1.0)}, {C(0.0), C(2.0), C(1.0)}, {C(0.0), C(1.0), C(2.0)}};
  EXPECT(residual(b, eigen_decomposition<C> {b}) < 10.0);
  // graded complex rows and columns, both ways
  std::vector<C> graded(16);
  for (std::size_t i {0}; i < 4; ++i) {
    for (std::size_t j {0}; j < 4; ++j) {
      graded[(i * 4) + j] = C(1.0 + static_cast<double>(i), static_cast<double>(j)) * std::pow(1e4, static_cast<double>(i) - static_cast<double>(j));
    }
  }
  const matrix<C> g {4, 4, graded};
  EXPECT(residual(g, eigen_decomposition<C> {g}) < 1e6);
}

TEST(small_reflections_leave_a_zero_tail_alone)
{
  double v[3] {2.0, 0.0, 0.0};
  EXPECT_EQ(numerical::detail::small_reflector(v, 3), 0.0);
  EXPECT_EQ(v[0], 2.0);
  C w[2] {C(2.0), C(0.0)};
  EXPECT_EQ(numerical::detail::complex_reflector(w), C(0.0));
  C x[2] {C(0.0, 2.0), C(0.0)}; // α not real: a reflection still, to a real β
  EXPECT(numerical::detail::complex_reflector(x) != C(0.0));
  EXPECT_EQ(x[0].imag(), 0.0);
}

TEST(the_standard_form_of_every_kind_of_block)
{
  const auto check {[](double a, double b, double c, double d) {
                      const double                               a0 {a};
                      const double                               b0 {b};
                      const double                               c0 {c};
                      const double                               d0 {d};
                      const numerical::detail::standard_block s {numerical::detail::standardize_block(a, b, c, d)};
                      // [a b; c d] = Rᵀ [a0 b0; c0 d0] R, R = [cs −sn; sn cs]
                      const double ra {(s.cs * ((a0 * s.cs) + (b0 * s.sn))) + (s.sn * ((c0 * s.cs) + (d0 * s.sn)))};
                      const double rc {(-s.sn * ((a0 * s.cs) + (b0 * s.sn))) + (s.cs * ((c0 * s.cs) + (d0 * s.sn)))};
                      const double rd {(-s.sn * ((-a0 * s.sn) + (b0 * s.cs))) + (s.cs * ((-c0 * s.sn) + (d0 * s.cs)))};
                      const double size {std::abs(a0) + std::abs(b0) + std::abs(c0) + std::abs(d0)};
                      const bool   standard {c == 0.0 || (a == d && std::signbit(b) != std::signbit(c))};
                      return standard && std::abs(ra - a) <= 1e-15 * size && std::abs(rc - c) <= 1e-15 * size && std::abs(rd - d) <= 1e-15 * size
                             && std::abs((a0 + d0) - (s.re1 + s.re2)) <= 1e-15 * size;
                    }};
  EXPECT(check(1.0, 2.0, 0.0, 3.0));     // triangular already
  EXPECT(check(1.0, 0.0, 0.0, 3.0));     // diagonal
  EXPECT(check(2.0, 1.0, 1.0, 2.0));     // equal diagonal entries, real eigenvalues
  EXPECT(check(1.0, 0.0, 2.0, 3.0));     // lower triangular: exchanged
  EXPECT(check(1.0, 2.0, -3.0, 1.0));    // standard already
  EXPECT(check(4.0, 1.0, 2.0, 1.0));     // real eigenvalues
  EXPECT(check(1.0, 2.0, -3.0, 1.5));    // complex
  EXPECT(check(1.0, 1e-17, 1e-17, 1.0)); // nearly equal real eigenvalues, triangular after the rotation
  EXPECT(check(1.0, -1e-17, 1e-17, 1.0 + 1e-16));
  EXPECT(check(1.0, 1.0, -1.0, 3.0));    // a double eigenvalue
  EXPECT(check(2.0, -0.5, 2.0, 0.0));    // a double eigenvalue whose rotation zeroes the upper entry: exchanged
  EXPECT(check(-6.0, -18.0, 0.5, 0.0));  // one whose rotation zeroes the lower entry: triangular as it is
}
