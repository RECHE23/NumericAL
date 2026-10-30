// The general eigendecomposition: residuals at rounding on random real and complex matrices, unit eigenvectors with
// their largest component real, a real matrix's conjugate pairs; the matrices that strain it (triangular, defective,
// cyclic, graded, a complex pair nearly real); and its pieces alone: the balancing, the 2 × 2 standard form, the small
// reflections, the bounded iterations.
#include <algorithm>
#include <array>
#include <type_traits>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::eigen_decomposition;
  using numerical::matrix;
  using C = std::complex<double>;
  using numerical::detail::lapack_index; // std::ptrdiff_t: long long where long is 32 bits

  //! \brief Whether column \p c of \p v has a real component of the largest modulus, \p most its square: the one
  //!        LAPACK's normalisation made real, ties among moduli equal to rounding (a cyclic matrix's vectors, every
  //!        component of modulus 1/√n) taken as such, the rotation having moved their last bits.
  inline bool real_among_largest(const matrix<C>& v,
                                 std::size_t      c,
                                 double           most)
  {
    for (std::size_t r {0}; r < v.rows(); ++r) {
      if (v(r, c).imag() == 0.0 && std::norm(v(r, c)) >= most * (1.0 - 1e-12)) {
        return true;
      }
    }
    return false;
  }

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

  //! \brief `max |A v − λ v|` over the pairs, relative to `n ‖A‖_max ε`; infinite where a vector is not of unit norm with its
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
      if (std::abs(std::sqrt(norm) - 1.0) > 1e-14 || !real_among_largest(e.vectors(), c, std::norm(e.vectors()(largest, c)))) {
        return std::numeric_limits<double>::infinity(); // not normalised as LAPACK's: no bound holds it
      }
    }
    return worst / (std::max(size, 1e-300) * static_cast<double>(n) * 2.220446049250313e-16);
  }

  //! \brief `max |uᴴ A − λ uᴴ|` over the left pairs (\ref eigen_decomposition::left_vectors), relative to
  //!        `n ‖A‖_max ε`; infinite where a vector is not of unit norm with its largest component real.
  template <typename T>
  double left_residual(const matrix<T>&              a,
                       const eigen_decomposition<T>& e)
  {
    const std::size_t n    {a.rows()};
    double            size {0.0};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        size = std::max(size, static_cast<double>(std::abs(a(i, j))));
      }
    }
    const matrix<C>& u     {e.left_vectors()};
    double           worst {0.0};
    for (std::size_t c {0}; c < n; ++c) {
      double      norm    {0.0};
      std::size_t largest {0};
      for (std::size_t k {0}; k < n; ++k) {
        C sum {};
        for (std::size_t r {0}; r < n; ++r) {
          sum += std::conj(u(r, c)) * a(r, k);
        }
        worst  = std::max(worst, std::abs(sum - (e.values()[c] * std::conj(u(k, c)))));
        norm  += std::norm(u(k, c));
        if (std::norm(u(k, c)) > std::norm(u(largest, c))) {
          largest = k;
        }
      }
      if (std::abs(std::sqrt(norm) - 1.0) > 1e-14 || !real_among_largest(u, c, std::norm(u(largest, c)))) {
        return std::numeric_limits<double>::infinity(); // not normalised as LAPACK's: no bound holds it
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

// The left eigenvectors, from the same Schur form as the right ones (S = J Tᴴ J): uᴴ A = λ uᴴ to rounding for each
// eigenvalue in its place, of unit norm with the largest component real; alone or with the right ones (the same
// right vectors either way); through a balancing that scales (a graded matrix); a double eigenvalue of two vectors,
// whose left ones a second decomposition of Aᵀ matched to the eigenvalues by distance alone.
TEST(eig_left_vectors_match_their_eigenvalues)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64 g {17};
  for (const std::size_t n : {1U, 2U, 3U, 7U, 16U, 33U}) {
    const matrix<double>              a    {random_real(n, g)};
    const eigen_decomposition<double> both {a, true, true};
    EXPECT(left_residual(a, both) < 10.0 && residual(a, both) < 10.0);
    const eigen_decomposition<double> left {a, false, true};
    EXPECT(left_residual(a, left) < 10.0 && left.vectors().rows() == 0);
    EXPECT(eigen_decomposition<double> {a}.vectors() == both.vectors());
    const matrix<C>              b       {random_complex(n, g)};
    const eigen_decomposition<C> complex {b, true, true};
    EXPECT(left_residual(b, complex) < 10.0 && residual(b, complex) < 10.0);
  }
  const matrix<double> base {random_real(6, g)};
  std::vector<double>  entries(36);
  for (std::size_t i {0}; i < 6; ++i) {
    for (std::size_t j {0}; j < 6; ++j) {
      entries[(i * 6) + j] = base(i, j) * std::ldexp(1.0, 8 * (static_cast<int>(j) - static_cast<int>(i)));
    }
  }
  const matrix<double> graded  {6, 6, entries};
  // Through a balancing that scales, its residual too small to see beside entries of 2⁴⁰: each left vector against
  // Aᵀ's right one for the same (simple) eigenvalue, conjugated, equal up to a unit factor.
  const eigen_decomposition<double> graded_left {graded, false, true};
  std::vector<double>               transposed(36);
  for (std::size_t i {0}; i < 6; ++i) {
    for (std::size_t j {0}; j < 6; ++j) {
      transposed[(j * 6) + i] = graded(i, j);
    }
  }
  const eigen_decomposition<double> right_of_transpose {matrix<double> {6, 6, transposed}};
  for (std::size_t c {0}; c < 6; ++c) {
    std::size_t nearest {0};
    for (std::size_t p {1}; p < 6; ++p) {
      if (std::abs(right_of_transpose.values()[p] - graded_left.values()[c]) < std::abs(right_of_transpose.values()[nearest] - graded_left.values()[c])) {
        nearest = p;
      }
    }
    C overlap {};
    for (std::size_t r {0}; r < 6; ++r) {
      overlap += std::conj(graded_left.left_vectors()(r, c)) * std::conj(right_of_transpose.vectors()(r, nearest));
    }
    EXPECT(std::abs(std::abs(overlap) - 1.0) < 1e-8);
  }
  const matrix<double> s       {{1.0, 2.0, 0.5}, {0.0, 1.0, 3.0}, {2.0, 1.0, 1.0}};
  const matrix<double> double_ {s * matrix<double> {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 2.0}} *numerical::inverse(s)};
  EXPECT(left_residual(double_, eigen_decomposition<double> {double_, true, true}) < 100.0);
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

namespace {

  //! \brief A random upper Hessenberg matrix of order \p n, row-major, with zeros on the subdiagonal where \p split.
  template <typename T>
  numerical::buffer<T> random_hessenberg(std::size_t      n,
                                         std::mt19937_64& g,
                                         std::size_t      split = 0)
  {
    std::normal_distribution<double> d;
    numerical::buffer<T>             h(n * n, T {});
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {i == 0 ? 0 : i - 1}; j < n; ++j) {
        if constexpr (std::is_same_v<T, double>) {
          h[(i * n) + j] = d(g);
        }
        else {
          h[(i * n) + j] = T(d(g), d(g));
        }
      }
      if (split != 0 && i > 0 && i % split == 0) {
        h[(i * n) + i - 1] = T {};
      }
    }
    return h;
  }

  //! \brief Whether \p t is the Schur form of \p h by \p zt (`Zᵀ`): `Zᴴ H Z = T` and `Zᴴ Z = I` to rounding, `T` upper
  //!        triangular (quasi-triangular with standard 2 × 2 blocks for a real `T`).
  template <typename T>
  bool is_schur_form(const numerical::buffer<T>& h,
                     const numerical::buffer<T>& t,
                     const numerical::buffer<T>& zt,
                     std::size_t                 n)
  {
    double size {1e-300};
    for (const T& x : h) {
      size = std::max(size, static_cast<double>(std::abs(x)));
    }
    const double tolerance {1e-13 * static_cast<double>(n) * size};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        C similar {};
        C gram    {};
        for (std::size_t k {0}; k < n; ++k) {
          C row {};
          for (std::size_t l {0}; l < n; ++l) {
            row += std::conj(C {zt[(i * n) + k]}) * C {h[(k * n) + l]} *C {zt[(j * n) + l]};
          }
          similar += row;
          gram    += std::conj(C {zt[(i * n) + k]}) * C {zt[(j * n) + k]};
        }
        if (std::abs(similar - C {t[(i * n) + j]}) > tolerance || std::abs(gram - (i == j ? 1.0 : 0.0)) > 1e-13 * static_cast<double>(n)) {
          return false;
        }
        const bool below {i > j + 1 || (i == j + 1 && (std::is_same_v<T, C> || (j > 0 && t[(j * n) + j - 1] != T {})))};
        if (below && t[(i * n) + j] != T {}) {
          return false;
        }
        if constexpr (std::is_same_v<T, double>) {
          if (i == j + 1 && t[(i * n) + j] != 0.0 && (t[(j * n) + j] != t[(i * n) + i] || std::signbit(t[(j * n) + i]) == std::signbit(t[(i * n) + j]))) {
            return false; // a 2 × 2 block not standard
          }
        }
      }
    }
    return true;
  }

  //! \brief The multishift Schur form of \p h0 (order \p n), checked; \p full and \p recursive as given.
  template <typename T>
  bool multishift_checks(const numerical::buffer<T>& h0,
                         std::size_t                 n,
                         bool                        recursive = true)
  {
    numerical::buffer<T> h           {h0};
    numerical::buffer<T> zt(n * n, T {});
    for (std::size_t i {0}; i < n; ++i) {
      zt[(i * n) + i] = T {1};
    }
    numerical::buffer<C>                     w(n);
    const numerical::detail::schur_target<T> target {.h = &h, .n = n, .low = 0, .high = n, .full = true, .zt = &zt};
    numerical::detail::multishift_schur(target, w.data(), recursive);
    return is_schur_form(h0, h, zt, n);
  }
} // namespace

TEST(the_multishift_schur_form_of_random_hessenberg_matrices)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64 g {31};
  for (const std::size_t n : {16U, 30U, 61U, 80U}) {
    EXPECT(multishift_checks(random_hessenberg<double>(n, g), n));
    EXPECT(multishift_checks(random_hessenberg<C>(n, g), n));
    EXPECT(multishift_checks(random_hessenberg<double>(n, g, 7), n, false)); // zeros on the subdiagonal: blocks apart
    EXPECT(multishift_checks(random_hessenberg<C>(n, g, 7), n, false));
  }
  EXPECT(multishift_checks(numerical::buffer<double>(400, 0.0), 20)); // everything deflates at once
}

TEST(eig_of_matrices_past_every_crossover)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64                   g {37};
  const matrix<double>              a {random_real(205, g)}; // multishift, blocked reduction
  const eigen_decomposition<double> e {a};
  EXPECT(residual(a, e) < 10.0);
  const numerical::vector<C> alone    {numerical::eigenvalues(a)};
  for (std::size_t i {0}; i < 205; ++i) {
    EXPECT_EQ(alone[i], e.values()[i]); // the whole Schur form for the eigenvalues alone too
  }
  const matrix<C>              b {random_complex(130, g)};
  const eigen_decomposition<C> f {b};
  EXPECT(residual(b, f) < 10.0);
}

TEST(plane_rotations_take_a_pair_to_its_length)
{
  using numerical::detail::givens;
  const auto real_zero_g    {givens(3.0, 0.0)};
  EXPECT(real_zero_g.c == 1.0 && real_zero_g.s == 0.0 && real_zero_g.r == 3.0);
  const auto real_zero_f    {givens(0.0, -2.0)};
  EXPECT(real_zero_f.c == 0.0 && real_zero_f.s == -1.0 && real_zero_f.r == 2.0);
  const auto real_both      {givens(-3.0, 4.0)};
  EXPECT(std::abs(real_both.r + 5.0) < 1e-15 && std::abs((real_both.c * -3.0) + (real_both.s * 4.0) - real_both.r) < 1e-15);
  const auto complex_zero_g {givens(C(1.0, 2.0), C {})};
  EXPECT(complex_zero_g.c == 1.0 && complex_zero_g.r == C(1.0, 2.0));
  const auto complex_zero_f {givens(C {}, C(0.0, 2.0))};
  EXPECT(complex_zero_f.c == 0.0 && complex_zero_f.r == C(2.0) && std::abs((complex_zero_f.s * C(0.0, 2.0)) - complex_zero_f.r) < 1e-15);
  const auto complex_both   {givens(C(1.0, 1.0), C(2.0, -1.0))};
  EXPECT(std::abs((complex_both.c * C(1.0, 1.0)) + (complex_both.s * C(2.0, -1.0)) - complex_both.r) < 1e-15);
  EXPECT(std::abs((-std::conj(complex_both.s) * C(1.0, 1.0)) + (complex_both.c * C(2.0, -1.0))) < 1e-15);
}

namespace {

  //! \brief The largest residual of `TL X − X TR = B` for the \p n1 × \p n2 solution of \ref small_sylvester.
  double sylvester_residual(std::array<double, 4>         tl,
                            std::array<double, 4>         tr,
                            std::array<double, 4>         b,
                            lapack_index                  n1,
                            lapack_index                  n2)
  {
    using numerical::detail::one_based;
    double x[4]  {};
    numerical::detail::small_sylvester(one_based<double> {tl.data(), 2}, one_based<double> {tr.data(), 2}, one_based<double> {b.data(), 2}, n1, n2, x);
    double worst {0.0};
    for (lapack_index i {0}; i < n1; ++i) {
      for (lapack_index j {0}; j < n2; ++j) {
        double lhs {0.0};
        for (lapack_index k {0}; k < n1; ++k) {
          lhs += tl[static_cast<std::size_t>((i * 2) + k)] * x[(k * n2) + j];
        }
        for (lapack_index k {0}; k < n2; ++k) {
          lhs -= x[(i * n2) + k] * tr[static_cast<std::size_t>((k * 2) + j)];
        }
        worst = std::max(worst, std::abs(lhs - b[static_cast<std::size_t>((i * 2) + j)]));
      }
    }
    return worst;
  }
} // namespace

TEST(the_small_sylvester_equations_of_a_block_exchange)
{
  const std::array<double, 4> tl {1.0, 2.0, -3.0, 1.0};
  const std::array<double, 4> tr {4.0, 1.0, -2.0, 4.0};
  const std::array<double, 4> b  {1.0, 2.0, 3.0, 4.0};
  EXPECT(sylvester_residual(tl, tr, b, 1, 2) < 1e-14);
  EXPECT(sylvester_residual(tl, tr, b, 2, 1) < 1e-14);
  EXPECT(sylvester_residual(tl, tr, b, 2, 2) < 1e-14);
  EXPECT(std::isfinite(sylvester_residual(tl, tl, b, 2, 2))); // TL = TR: singular, a pivot floored, the answer finite
}

TEST(a_block_exchange_that_would_not_hold_is_refused)
{
  using numerical::detail::one_based;
  // a 2 × 2 block of eigenvalues 1 ± 6e-6 i above one of 1.0000043 ± 1.4e-5 i: the swap would leave them coupled
  const std::vector<double> before {1, 2.9258991580319096, -0.99582361335424341, 0.6660776232485035, -3.4177527863695918e-11, 1, -0.63530008058277909,
                                    0.22425311739723819, 0, 0, 1.0000043005630168, 0.68529066378232517, 0, 0, -2.0478021681892508e-10, 1.0000043005630168};
  std::vector<double>       t      {before};
  std::vector<double>       q(16, 0.0);
  for (std::size_t i {0}; i < 4; ++i) {
    q[(i * 4) + i] = 1.0;
  }
  EXPECT(!numerical::detail::exchange_blocks(one_based<double> {t.data(), 4}, one_based<double> {q.data(), 4}, 4, 1, 2, 2));
  EXPECT(t == before);
  lapack_index last {1};
  EXPECT(!numerical::detail::reorder(one_based<double> {t.data(), 4}, one_based<double> {q.data(), 4}, 4, 3, last));
  EXPECT_EQ(last, lapack_index {3}); // where the block stayed
  last = 4;
  EXPECT(!numerical::detail::reorder(one_based<double> {t.data(), 4}, one_based<double> {q.data(), 4}, 4, 1, last));
  EXPECT_EQ(last, lapack_index {1});
}

TEST(reordering_a_schur_form_keeps_it_one)
{
  using numerical::detail::one_based;
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64                                g {17};
  std::uniform_real_distribution<double>         u(-1.0, 1.0);
  const lapack_index                             n {7};
  for (int trial {0}; trial < 400; ++trial) { // blocks nearly real among them split as they move
    std::vector<double>     t(49, 0.0);
    std::vector<double>     q(49, 0.0);
    const one_based<double> tt {t.data(), n};
    for (lapack_index i {1}; i <= n; ++i) {
      q[static_cast<std::size_t>((i - 1) * 8)] = 1.0;
      for (lapack_index j {i}; j <= n; ++j) {
        tt(i, j) = u(g);
      }
    }
    for (lapack_index i {1}; i < n;) {
      if (g() % 2 == 1) {
        const double w {std::pow(10.0, -static_cast<double>(g() % 17))};
        const double b {1.0 + u(g)};
        tt(i, i + 1)     = b;
        tt(i + 1, i)     = -w * w / b;
        tt(i + 1, i + 1) = tt(i, i);
        i               += 2;
      }
      else {
        ++i;
      }
    }
    const std::vector<double>         before {t};
    lapack_index                      last {1 + static_cast<lapack_index>(g() % n)};
    lapack_index                      first {1 + static_cast<lapack_index>(g() % n)};
    first = first > 1 && tt(first, first - 1) != 0.0 ? first - 1 : first;
    const lapack_index size {first < n && tt(first + 1, first) != 0.0 ? 2 : 1};
    const auto         trace_and_det {[&tt, size](lapack_index at) { // of the size × size diagonal block at `at`
                                        return size == 1 ? std::pair {tt(at, at), tt(at, at)}
                       : std::pair {tt(at, at) + tt(at + 1, at + 1), (tt(at, at) * tt(at + 1, at + 1)) - (tt(at, at + 1) * tt(at + 1, at))};
                                      }};
    const auto moved   {trace_and_det(first)};
    EXPECT(numerical::detail::reorder(tt, one_based<double> {q.data(), n}, n, first, last));
    const auto arrived {trace_and_det(last)}; // the block, or what it split into, where reorder says it ended
    EXPECT(std::abs(arrived.first - moved.first) < 1e-12 && std::abs(arrived.second - moved.second) < 1e-12);
    // rows of Qᵀ: T = Qᵀ T₀ Q, the form quasi-triangular
    EXPECT(is_schur_form(numerical::buffer<double>(before.begin(), before.end()), numerical::buffer<double>(t.begin(), t.end()),
                         numerical::buffer<double>(q.begin(), q.end()), 7));
  }
  std::vector<C> c(25, C  {});
  std::vector<C> cq(25, C {});
  for (std::size_t i {0}; i < 5; ++i) {
    cq[(i * 5) + i] = C(1.0);
    for (std::size_t j {i}; j < 5; ++j) {
      c[(i * 5) + j] = C(u(g), u(g));
    }
  }
  const std::vector<C>         before {c};
  lapack_index                 last   {5};
  EXPECT(numerical::detail::reorder(one_based<C> {c.data(), 5}, one_based<C> {cq.data(), 5}, 5, 1, last)); // down
  EXPECT(c[24] == before[0]);                                                                              // to the end
  last = 1;
  const C fourth {c[18]};
  EXPECT(numerical::detail::reorder(one_based<C> {c.data(), 5}, one_based<C> {cq.data(), 5}, 5, 4, last)); // and up
  EXPECT(c[0] == fourth);
  EXPECT(is_schur_form(numerical::buffer<C>(before.begin(), before.end()), numerical::buffer<C>(c.begin(), c.end()),
                       numerical::buffer<C>(cq.begin(), cq.end()), 5));
}

TEST(the_bulge_columns_of_zero_shifts_on_zero_entries)
{
  using numerical::detail::one_based;
  double zeros[9] {};
  double v[3]     {1.0, 1.0, 1.0};
  numerical::detail::bulge_column(one_based<double> {zeros, 3}, 3, C {}, C {}, v);
  EXPECT(v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0);
  v[0] = 1.0;
  numerical::detail::bulge_column(one_based<double> {zeros, 3}, 2, C {}, C {}, v);
  EXPECT(v[0] == 0.0 && v[1] == 0.0);
  C czeros[9] {};
  C cv[3]     {C(1.0), C(1.0), C(1.0)};
  numerical::detail::bulge_column(one_based<C> {czeros, 3}, 3, C {}, C {}, cv);
  EXPECT(cv[0] == C {} && cv[2] == C {});
  cv[0] = C(1.0);
  numerical::detail::bulge_column(one_based<C> {czeros, 3}, 2, C {}, C {}, cv);
  EXPECT(cv[0] == C {} && cv[1] == C {});
}

TEST(sweeps_of_many_shifts_keep_a_schur_similarity)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64 g {41};
  for (const bool accumulate : {false, true}) {
    for (const lapack_index top : {lapack_index {1}, lapack_index {9}}) { // the whole matrix, and an active block below a zero subdiagonal
      for (const std::size_t n : {24U, 7U}) {
        numerical::buffer<double> h0 {random_hessenberg<double>(n, g)};
        if (top > 1 && n > 9) {
          h0[(8 * n) + 7] = 0.0;
        }
        if (n < 9) {
          h0[(5 * n) + 4] = 0.0; // the trailing 2 × 2 apart
        }
        numerical::buffer<double> h {h0};
        numerical::buffer<double> zt(n * n, 0.0);
        for (std::size_t i {0}; i < n; ++i) {
          zt[(i * n) + i] = 1.0;
        }
        // real shifts out of pairs, then a conjugate pair: the sweep pairs them
        std::vector<C>                                        shifts {C(0.5), C(0.3, 0.2), C(0.3, -0.2), C(-0.1), C(2.0), C(-1.5)};
        const numerical::detail::schur_target<double>         target {.h = &h, .n = n, .low = 0, .high = n, .full = true, .zt = &zt};
        const lapack_index                                    ktop   {n > 9 ? top : static_cast<lapack_index>(n) - 1}; // the order-7 block: its last 2 × 2
        numerical::detail::chase_bulges(target, ktop, static_cast<lapack_index>(n), shifts.data(), n > 9 ? 6 : 2, accumulate);
        // H stays similar to H₀ through Z, Hessenberg below its first subdiagonal
        bool similar {true};
        for (std::size_t i {0}; i < n && similar; ++i) {
          for (std::size_t j {0}; j < n && similar; ++j) {
            double s {0.0};
            for (std::size_t k {0}; k < n; ++k) {
              for (std::size_t l {0}; l < n; ++l) {
                s += zt[(i * n) + k] * h0[(k * n) + l] * zt[(j * n) + l];
              }
            }
            similar = std::abs(s - h[(i * n) + j]) < 1e-12 && (i <= j + 1 || h[(i * n) + j] == 0.0);
          }
        }
        EXPECT(similar);
      }
    }
  }
  numerical::buffer<C> ch0       {random_hessenberg<C>(9, g)};
  numerical::buffer<C> ch        {ch0};
  numerical::buffer<C> czt(81, C {});
  for (std::size_t i {0}; i < 9; ++i) {
    czt[(i * 9) + i] = C(1.0);
  }
  std::vector<C>                           cshifts {C(0.5, 1.0), C(-0.5, 0.2), C(1.0), C(0.0, -1.0)};
  const numerical::detail::schur_target<C> ctarget {.h = &ch, .n = 9, .low = 0, .high = 9, .full = false, .zt = &czt};
  numerical::detail::chase_bulges(ctarget, 1, 9, cshifts.data(), 4, true);
  EXPECT(std::isfinite(ch[0].real()));
}

TEST(the_multishift_iterations_where_the_shifts_stall)
{
  for (const std::size_t n : {40U, 64U}) { // the cyclic permutation: exceptional shifts, windows grown and shrunk
    numerical::buffer<double> cyclic(n * n, 0.0);
    for (std::size_t i {0}; i + 1 < n; ++i) {
      cyclic[((i + 1) * n) + i] = 1.0;
    }
    cyclic[n - 1] = 1.0;
    // as a Hessenberg matrix: rotate the corner entry in by a similarity is not needed, the corner is in the first row
    EXPECT(multishift_checks(cyclic, n));
    EXPECT(multishift_checks(numerical::buffer<C>(cyclic.begin(), cyclic.end()), n));
  }
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64           g          {43};
  numerical::buffer<double> zero_last  {random_hessenberg<double>(30, g)}; // an exact zero eigenvalue at the bottom
  numerical::buffer<C>      czero_last {random_hessenberg<C>(30, g)};
  for (std::size_t i {0}; i < 30; ++i) {
    zero_last[(i * 30) + 29]  = 0.0;
    czero_last[(i * 30) + 29] = C {};
  }
  EXPECT(multishift_checks(zero_last, 30));
  EXPECT(multishift_checks(czero_last, 30));
  numerical::buffer<double>                     h       {random_hessenberg<double>(30, g)};
  numerical::buffer<C>                          w(30);
  const numerical::detail::schur_target<double> bounded {.h = &h, .n = 30, .low = 0, .high = 30, .full = false, .zt = nullptr};
  EXPECT_THROWS(numerical::detail::multishift_schur(bounded, w.data(), true, 1), numerical::convergence_error);
}

TEST(a_wide_deflation_window_is_itself_taken_by_the_multishift_iterations)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64                           g      {47};
  const std::size_t                         n      {110};
  numerical::buffer<C>                      h      {random_hessenberg<C>(n, g)};
  numerical::buffer<C>                      w(n);
  const numerical::detail::schur_target<C>  target {.h = &h, .n = n, .low = 0, .high = n, .full = false, .zt = nullptr};
  const numerical::detail::deflation_count  found  {numerical::detail::early_deflation(target, 1, static_cast<lapack_index>(n), 105, w.data(), true)};
  EXPECT_EQ(found.shifts + found.deflated, lapack_index {105});
  EXPECT_EQ(numerical::detail::recommended_shifts(20), lapack_index {2});
  EXPECT_EQ(numerical::detail::recommended_shifts(40), lapack_index {4});
  EXPECT_EQ(numerical::detail::recommended_shifts(100), lapack_index {10});
  EXPECT_EQ(numerical::detail::recommended_shifts(1024), lapack_index {64});
  EXPECT_EQ(numerical::detail::recommended_shifts(4000), lapack_index {128});
  EXPECT_EQ(numerical::detail::recommended_shifts(7000), lapack_index {256});
}

namespace {

  //! \brief The cyclic permutation of order \p n plus \p eps times an exact pattern on its Hessenberg part: shifts that
  //!        stall for dozens of iterations, the same on every platform.
  numerical::buffer<double> perturbed_cyclic(std::size_t         n,
                                             double              eps,
                                             lapack_index        seed)
  {
    numerical::buffer<double> h(n * n, 0.0);
    for (std::size_t i {0}; i + 1 < n; ++i) {
      h[((i + 1) * n) + i] = 1.0;
    }
    h[n - 1] = 1.0;
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {i == 0 ? 0 : i - 1}; j < n; ++j) {
        const lapack_index pattern {((static_cast<lapack_index>(i) * 131) + (static_cast<lapack_index>(j) * 71) + (seed * 29)) % 97};
        h[(i * n) + j] += eps * ((static_cast<double>(pattern) / 48.0) - 1.0);
      }
    }
    return h;
  }
} // namespace

TEST(long_stalls_take_their_shifts_elsewhere)
{
  // after many iterations without deflation the window has shrunk below half the shifts: the shifts are the trailing
  // submatrix's eigenvalues; two real shifts, the one nearer the corner twice
  const numerical::buffer<double> real_trailing    {perturbed_cyclic(30, 1e-3, 0)};
  EXPECT(multishift_checks(real_trailing, 30));
  const numerical::buffer<double> complex_trailing {perturbed_cyclic(40, 1e-12, 0)};
  EXPECT(multishift_checks(numerical::buffer<C>(complex_trailing.begin(), complex_trailing.end()), 40));
  EXPECT(multishift_checks(perturbed_cyclic(25, 1e-15, 3), 25));
  const numerical::buffer<double> complex_nearer {perturbed_cyclic(16, 1e-6, 2)};
  EXPECT(multishift_checks(numerical::buffer<C>(complex_nearer.begin(), complex_nearer.end()), 16));
}

TEST(a_complex_reflection_makes_its_first_entry_real)
{
  C          alpha   {0.0, 2.0};
  C          tail[1] {C {}};
  const C    tau     {numerical::detail::reflector_tau(alpha, tail, 1)};
  EXPECT(tau != C {});
  EXPECT(alpha.imag() == 0.0 && std::abs(std::abs(alpha) - 2.0) < 1e-15);
  C          real_alpha {3.0};
  EXPECT(numerical::detail::reflector_tau(real_alpha, tail, 1) == C {});
}

TEST(vigilant_deflation_zeroes_a_negligible_subdiagonal_entry)
{
  using numerical::detail::one_based;
  const auto deflated {[](auto entries) { // entries: a 4 × 4 row-major, k = 2: whether h(3, 2) was set to zero
                         using T = std::decay_t<decltype(entries[0])>;
                         std::vector<T> h(entries.begin(), entries.end());
                         numerical::detail::vigilant_deflation(one_based<T> {h.data(), 4}, 2, 1, 4, 1e-290);
                         return h[(2 * 4) + 1] == T {};
                       }};
  EXPECT(deflated(std::array<double, 16> {1, 1, 1, 1, 1, 1, 1, 1, 0, 1e-20, 2, 1, 0, 0, 1, 1}));          // negligible: zeroed
  EXPECT(!deflated(std::array<double, 16> {1, 1, 1, 1, 1, 1, 1, 1, 0, 1e-3, 2, 1, 0, 0, 1, 1}));          // not negligible
  EXPECT(deflated(std::array<double, 16> {1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1}));              // already zero
  EXPECT(deflated(std::array<double, 16> {1, 1, 1, 1, 1e-30, 0, 5, 1, 0, 1e-300, 0, 1, 0, 0, 1e-30, 1})); // zero diagonal: the neighbours judge
  EXPECT(deflated(std::array<double, 16> {1, 1, 1, 1, 1, 2, 1, 1, 0, 1e-20, 2, 1, 0, 0, 1, 1}));          // equal diagonal entries: test2 = 0
  EXPECT(!deflated(std::array<double, 16> {1, 1, 1, 1, 1, 1, 1e18, 1, 0, 1e-17, 1.0001, 1, 0, 0, 1, 1})); // small, but its product with the
                                                                                                          // entry above is not
  EXPECT(deflated(std::array<C, 16> {C(1), C(1), C(1), C(1), C(1), C(1), C(1), C(1), C {}, C(1e-20), C(2), C(1), C {}, C {}, C(1), C(1)}));
  EXPECT(!deflated(std::array<C, 16> {C(1), C(1), C(1), C(1), C(1), C(1), C(1e18), C(1), C {}, C(1e-17), C(1.0001), C(1), C {}, C {}, C(1), C(1)}));
  EXPECT(deflated(std::array<C, 16> {C(1), C(1), C(1), C(1), C(1), C(2), C(1), C(1), C {}, C(1e-20), C(2), C(1), C {}, C {}, C(1), C(1)}));
}

TEST(sweeps_of_a_complex_matrix_with_and_without_z)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64 g {53};
  for (const bool accumulate : {false, true}) {
    for (const bool full : {false, true}) {
      for (const bool with_z : {false, true}) {
        for (const std::size_t n : {7U, 24U}) {
          numerical::buffer<C> h {random_hessenberg<C>(n, g)};
          if (n < 9) {
            h[(5 * n) + 4] = C {}; // a 2 × 2 active block at the bottom
          }
          numerical::buffer<C> zt(n * n, C {});
          for (std::size_t i {0}; i < n; ++i) {
            zt[(i * n) + i] = C(1.0);
          }
          std::vector<C>                           shifts {C(0.5, 1.0), C(-0.5, 0.2), C(1.0), C(0.0, -1.0)};
          const numerical::detail::schur_target<C> target {.h = &h, .n = n, .low = 0, .high = n, .full = full, .zt = with_z ? &zt : nullptr};
          numerical::detail::chase_bulges(target, n < 9 ? 6 : 1, static_cast<lapack_index>(n), shifts.data(), n < 9 ? 2 : 4, accumulate);
          EXPECT(std::isfinite(h[(n * n) - 1].real()));
        }
      }
    }
  }
  for (const bool full : {false, true}) { // real accumulated sweeps, without Z
    numerical::buffer<double>                     h      {random_hessenberg<double>(24, g)};
    std::vector<C>                                shifts {C(0.5), C(-0.5), C(0.3, 0.4), C(0.3, -0.4)};
    const numerical::detail::schur_target<double> target {.h = &h, .n = 24, .low = 0, .high = 24, .full = full, .zt = nullptr};
    numerical::detail::chase_bulges(target, 1, 24, shifts.data(), 4, true);
    EXPECT(std::isfinite(h[575]));
  }
}

TEST(the_multishift_eigenvalues_alone)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64 g {59};
  for (const std::size_t n : {30U, 160U}) { // 160: accumulated sweeps
    for (int kind {0}; kind < 2; ++kind) {
      numerical::buffer<double>                     h           {random_hessenberg<double>(n, g)};
      numerical::buffer<C>                          hc          {random_hessenberg<C>(n, g)};
      numerical::buffer<C>                          w(n);
      const numerical::detail::schur_target<double> real_target {.h = &h, .n = n, .low = 0, .high = n, .full = kind == 1, .zt = nullptr};
      numerical::detail::multishift_schur(real_target, w.data(), true);
      EXPECT(std::isfinite(w[0].real()));
      const numerical::detail::schur_target<C> complex_target {.h = &hc, .n = n, .low = 0, .high = n, .full = kind == 1, .zt = nullptr};
      numerical::detail::multishift_schur(complex_target, w.data(), true);
      EXPECT(std::isfinite(w[0].real()));
    }
  }
  numerical::buffer<C>                     hc      {random_hessenberg<C>(30, g)};
  numerical::buffer<C>                     w(30);
  const numerical::detail::schur_target<C> bounded {.h = &hc, .n = 30, .low = 0, .high = 30, .full = false, .zt = nullptr};
  EXPECT_THROWS(numerical::detail::multishift_schur(bounded, w.data(), true, 1), numerical::convergence_error);
  const matrix<C>              b                   {random_complex(130, g)};
  const numerical::vector<C>   alone               {numerical::eigenvalues(b)};
  const eigen_decomposition<C> f                   {b};
  for (std::size_t i {0}; i < 130; ++i) {
    EXPECT_EQ(alone[i], f.values()[i]);
  }
  EXPECT_EQ(numerical::detail::recommended_window(400, 40), lapack_index {40});
  EXPECT_EQ(numerical::detail::recommended_window(600, 64), lapack_index {96});
}

TEST(a_refused_exchange_while_sorting_the_window)
{
  using numerical::detail::one_based;
  // two 2 × 2 blocks that will not exchange, the larger below, over a 1 × 1 that deflates
  std::vector<double> t(25, 0.0);
  const double        blocks[16] {1, 2.9258991580319096, -0.99582361335424341, 0.6660776232485035, -3.4177527863695918e-11, 1, -0.63530008058277909,
                                  0.22425311739723819, 0, 0, 1.0000043005630168, 0.68529066378232517, 0, 0, -2.0478021681892508e-10, 1.0000043005630168};
  for (std::size_t i {0}; i < 4; ++i) {
    for (std::size_t j {0}; j < 4; ++j) {
      t[(i * 5) + j] = blocks[(i * 4) + j];
    }
    t[(i * 5) + 4] = 0.5;
  }
  t[24] = 3.0;
  std::vector<double> vt(25, 0.0);
  for (std::size_t i {0}; i < 4; ++i) {
    vt[i * 5] = 1.0; // the first column of V: the spike, undeflatable but for the last row
  }
  std::vector<C> w(5);
  EXPECT_EQ(numerical::detail::undeflated(one_based<double> {t.data(), 5}, one_based<double> {vt.data(), 5}, 5, 1.0, 1e-290, w.data()), lapack_index {4});
}

TEST(a_real_window_past_the_crossover)
{
  // NOLINTNEXTLINE(cert-msc51-cpp,cert-msc32-c,bugprone-random-generator-seed)
  std::mt19937_64                               g      {61};
  const std::size_t                             n      {205};
  numerical::buffer<double>                     h      {random_hessenberg<double>(n, g)};
  numerical::buffer<C>                          w(n);
  const numerical::detail::schur_target<double> target {.h = &h, .n = n, .low = 0, .high = n, .full = false, .zt = nullptr};
  const numerical::detail::deflation_count      found  {numerical::detail::early_deflation(target, 1, static_cast<lapack_index>(n), 202, w.data(), true)};
  EXPECT_EQ(found.shifts + found.deflated, lapack_index {202});
}

TEST(a_nearly_real_pair_below_a_chain_of_its_real_part)
{
  // 1 ± 1e-20 i at the bottom of a Jordan chain of eigenvalue 1: the pair's vector meets pivots of 1e-20 row after
  // row, floored, its components scaled down as they grow
  const std::size_t   n {30};
  std::vector<double> t(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    t[(i * n) + i] = 1.0;
    if (i + 1 < n) {
      t[(i * n) + i + 1] = 1.0;
    }
  }
  t[((n - 2) * n) + n - 1] = 1e-20;
  t[((n - 1) * n) + n - 2] = -1e-20;
  const eigen_decomposition<double> e {matrix<double> {n, n, t}};
  for (std::size_t c {0}; c < n; ++c) {
    EXPECT(std::isfinite(e.vectors()(0, c).real()) && std::isfinite(e.vectors()(0, c).imag()));
  }
}

namespace {

  //! \brief Whether the Schur decomposition of \p a holds: `A = Z T Zᴴ` and `Zᴴ Z = I` to `64 n ε`, `T` zero below its
  //!        subdiagonal, a real `T`'s 2 × 2 blocks in standard form and a complex `T` triangular, and its eigenvalues
  //!        those of \ref eigen_decomposition (as sets).
  template <typename T>
  bool schur_holds(const matrix<T>& a)
  {
    const std::size_t                         n             {a.rows()};
    const numerical::schur_decomposition<T>   s             {a};
    const matrix<T>&                          t             {s.t()};
    const matrix<T>&                          z             {s.z()};
    double                                    residual      {0.0};
    double                                    orthogonality {0.0};
    double                                    size          {0.0};
    std::vector<T>                            zt(n * n); // Z T
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t k {0}; k < n; ++k) {
        for (std::size_t l {0}; l < n; ++l) {
          zt[(i * n) + l] += z(i, k) * t(k, l);
        }
      }
    }
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        T sum  {};
        T gram {};
        for (std::size_t k {0}; k < n; ++k) {
          sum  += zt[(i * n) + k] * numerical::detail::conj_of(z(j, k));
          gram += numerical::detail::conj_of(z(k, i)) * z(k, j);
        }
        residual      = std::max(residual, std::abs(sum - a(i, j)));
        orthogonality = std::max(orthogonality, std::abs(gram - (i == j ? T {1.0} : T {})));
        size          = std::max(size, std::abs(a(i, j)));
      }
    }
    bool                  shaped {true};
    std::vector<C>        values;
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j + 1 < i; ++j) {
        shaped = shaped && t(i, j) == T {};
      }
    }
    for (std::size_t i {0}; i < n; ++i) {
      if constexpr (std::is_same_v<T, double>) {
        if (i + 1 < n && t(i + 1, i) != 0.0) { // a pair: equal diagonal, off-diagonal of opposite signs
          shaped = shaped && t(i, i) == t(i + 1, i + 1) && t(i, i + 1) * t(i + 1, i) < 0.0 && (i + 2 >= n || t(i + 2, i + 1) == 0.0);
          const double imaginary {std::sqrt(-t(i, i + 1) * t(i + 1, i))};
          values.emplace_back(t(i, i), imaginary);
          values.emplace_back(t(i, i), -imaginary);
          ++i;
          continue;
        }
        values.emplace_back(t(i, i));
      }
      else {
        shaped = shaped && (i + 1 >= n || t(i + 1, i) == T {});
        values.push_back(t(i, i));
      }
    }
    const auto                 by_parts = [](const C& x, const C& y) { return x.real() < y.real() || (x.real() == y.real() && x.imag() < y.imag()); };
    const numerical::vector<C> computed {numerical::eigenvalues(a)};
    std::vector<C>             expected(computed.begin(), computed.end());
    std::sort(values.begin(), values.end(), by_parts);
    std::sort(expected.begin(), expected.end(), by_parts);
    double apart {0.0};
    for (std::size_t k {0}; k < n; ++k) {
      apart = std::max(apart, std::abs(values[k] - expected[k]));
    }
    const double bound {64.0 * static_cast<double>(std::max<std::size_t>(n, 1)) * std::numeric_limits<double>::epsilon()};
    return shaped && residual <= bound * size && orthogonality <= bound && apart <= 1e-9 * std::max(size, 1.0);
  }
} // namespace

// The Schur decomposition, as scipy.linalg.schur: random real and complex matrices of orders past the multishift
// crossovers (100 complex, 200 real), a rotation's pair, a triangular matrix left as it is, a scaled one, the empty and the errors.
TEST(schur_decomposition_reconstructs_with_an_orthonormal_basis)
{
  for (const std::size_t n : {std::size_t {1}, std::size_t {2}, std::size_t {5}, std::size_t {40}, std::size_t {130}, std::size_t {210}}) {
    std::vector<double> real_entries(n * n);
    std::vector<C>      complex_entries(n * n);
    for (std::size_t e {0}; e < n * n; ++e) {
      real_entries[e]    = std::sin(1.0 + (0.7 * static_cast<double>(e)));
      complex_entries[e] = C {std::sin(1.0 + (0.7 * static_cast<double>(e))), std::cos(0.3 * static_cast<double>(e))};
    }
    EXPECT(schur_holds(matrix<double> {n, n, real_entries}));
    EXPECT(schur_holds(matrix<C> {n, n, complex_entries}));
  }
  const matrix<double> rotation {{0.0, -2.0}, {2.0, 0.0}}; // ±2i: one 2 × 2 block
  EXPECT(schur_holds(rotation));
  EXPECT((numerical::schur_decomposition<double> {rotation}.t()(1, 0) != 0.0));
         const matrix<double> triangular {{1.0, 2.0, 3.0}, {0.0, 4.0, 5.0}, {0.0, 0.0, 6.0}};
         EXPECT(schur_holds(triangular));
  EXPECT(schur_holds(matrix<double> {{1e200, 3e199}, {-2e199, 5e199}}));
  EXPECT(schur_holds(matrix<double> {{0.0, 0.0}, {0.0, 0.0}}));
  EXPECT((numerical::schur_decomposition<double> {matrix<double> {0, 0, {}}}.t().rows() == 0));
  const matrix<C> pair {{C {1.0, 1.0}, C {2.0}}, {C {0.0, -1.0}, C {3.0}}};
  const matrix<C> z    {numerical::schur_decomposition<C> {pair}.z()}; // each moved out of an expiring decomposition
  const matrix<C> t    {numerical::schur_decomposition<C> {pair}.t()};
  EXPECT(z.rows() == 2 && t.rows() == 2 && t(1, 0) == C {});
  EXPECT_THROWS((numerical::schur_decomposition<double> {matrix<double> {{1.0, 2.0}}}), numerical::dimension_error);
  EXPECT_THROWS((numerical::schur_decomposition<double> {matrix<double> {{std::numeric_limits<double>::quiet_NaN()}}}),
                numerical::non_finite_error);
}

namespace {

  //! \brief Whether \p s still decomposes \p a (`A = Z T Zᴴ`, `Z` unitary, to rounding), its first \p sdim eigenvalues
  //!        the ones \p chosen holds and the rest not, a real `T` quasi-triangular, a complex one triangular.
  template <typename T, typename Chosen>
  bool sorted_schur_holds(const matrix<T>&                         a,
                          const numerical::schur_decomposition<T>& s,
                          std::size_t                              sdim,
                          Chosen                                   chosen)
  {
    const std::size_t n             {a.rows()};
    const matrix<T>&  t             {s.t()};
    const matrix<T>&  z             {s.z()};
    double            residual      {0.0};
    double            orthogonality {0.0};
    double            size          {0.0};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        T sum  {};
        T gram {};
        for (std::size_t k {0}; k < n; ++k) {
          for (std::size_t l {0}; l < n; ++l) {
            sum += z(i, k) * t(k, l) * numerical::detail::conj_of(z(j, l));
          }
          gram += numerical::detail::conj_of(z(k, i)) * z(k, j);
        }
        residual      = std::max(residual, std::abs(sum - a(i, j)));
        orthogonality = std::max(orthogonality, std::abs(gram - (i == j ? T {1.0} : T {})));
        size          = std::max(size, std::abs(a(i, j)));
      }
    }
    bool                 ordered {true};
    const std::vector<C> values  {s.values()};
    for (std::size_t k {0}; k < n; ++k) {
      ordered = ordered && chosen(values[k]) == (k < sdim);
      for (std::size_t j {0}; j + 1 < k; ++j) {
        ordered = ordered && t(k, j) == T {};
      }
    }
    const double                          bound {64.0 * static_cast<double>(n) * std::numeric_limits<double>::epsilon()};
    return ordered && residual <= bound * std::max(size, 1.0) && orthogonality <= bound;
  }
} // namespace

namespace {

  /*!
   * \brief A matrix of order \p n whose eigenvalues lie clear of the selections' boundaries (real parts from 0.2 and
   *        moduli from 0.2 off 0 and 1, as rounding moves none across): `H T₀ H` for the Householder reflection `H` of
   *        a fixed vector and `T₀` upper (quasi-)triangular, its diagonal cycling through −2.5, −0.5 ± 0.5i, 0.4,
   *        0.2 ± 2i, −0.3 (a real `T` the pairs as 2 × 2 blocks `[a ω; −ω a]`, 3 where a block would not fit), each
   *        cycle's moved off the last's away from the boundaries: distinct, so well conditioned.
   */
  template <typename T>
  matrix<T> clear_of_boundaries(std::size_t n)
  {
    std::vector<T> t0(n * n, T {});
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {i + 1}; j < n; ++j) {
        t0[(i * n) + j] = T {0.1 * std::sin(static_cast<double>(i + (2 * j)))};
      }
    }
    // per slot: the real part and its step a cycle, the imaginary part (a pair where nonzero) and its step
    constexpr std::array<std::array<double, 4>, 6> slots {{{-2.5, -0.05, 0.0, 0.0},
      {-0.5, -0.01, 0.5, -0.01},
      {3.0, 0.05, 0.0, 0.0},
      {0.4, 0.02, 0.0, 0.0},
      {0.2, 0.05, 2.0, 0.05},
      {-0.3, -0.02, 0.0, 0.0}}};
    for (std::size_t i {0}; i < n; ++i) {
      const std::size_t              cycle {i / 6}; // each cycle's values off the last's, away from the boundaries
      const double                   c     {static_cast<double>(cycle)};
      const std::array<double, 4>&   slot  {slots[i % 6]};
      const bool                     pair  {slot[2] != 0.0};
      const double                   re    {slot[0] + (slot[1] * c)};
      const double                   im    {slot[2] + (slot[3] * c)};
      if constexpr (std::is_same_v<T, double>) {
        if (pair && i + 1 < n) {
          t0[(i * n) + i]           = re;
          t0[(i * n) + i + 1]       = im;
          t0[((i + 1) * n) + i]     = -im;
          t0[((i + 1) * n) + i + 1] = re;
          ++i;
          continue;
        }
        t0[(i * n) + i] = pair ? slots[2][0] + (slots[2][1] * c) : re;
      }
      else {
        t0[(i * n) + i] = pair ? T {re, im} : T {re};
      }
    }
    std::vector<T> v(n);
    double         square {0.0};
    for (std::size_t i {0}; i < n; ++i) {
      v[i] = T {0.5 + std::sin(1.0 + static_cast<double>(i))};
      if constexpr (!std::is_same_v<T, double>) {
        v[i] += T {0.0, std::cos(static_cast<double>(i))};
      }
      square += std::norm(v[i]);
    }
    std::vector<T> h(n * n); // I − 2 v vᴴ / vᴴ v
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        h[(i * n) + j] = (i == j ? T {1.0} : T {}) - (T {2.0 / square} *v[i] * numerical::detail::conj_of(v[j]));
      }
    }
    std::vector<T> a(n * n, T {});
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        for (std::size_t k {0}; k < n; ++k) {
          for (std::size_t l {0}; l < n; ++l) {
            a[(i * n) + j] += h[(i * n) + k] * t0[(k * n) + l] * h[(l * n) + j];
          }
        }
      }
    }
    return matrix<T> {n, n, a};
  }
} // namespace

// The eigenvalues a selection holds moved to the leading block of T, as scipy.linalg.schur(a, sort=...) (LAPACK's
// gees): its four selections by half plane and unit circle, real and complex, on matrices whose eigenvalues lie clear of
// the boundaries; a real pair moved, and counted twice, where either of its two is chosen; values() in T's order.
TEST(a_sorted_schur_form_leads_with_the_selected_eigenvalues)
{
  for (const std::size_t n : {std::size_t {1}, std::size_t {6}, std::size_t {20}}) {
    const matrix<double> real {clear_of_boundaries<double>(n)};
    const matrix<C>      complex {clear_of_boundaries<C>(n)};
    const auto           check = [&]<typename T>(const matrix<T>& a, auto chosen) {
                                   const numerical::schur_decomposition<T> s {a, chosen};
                                   return sorted_schur_holds(a, s, s.leading(), chosen);
                                 };
    for (const auto& chosen : std::vector<bool (*)(const C&)> {[](const C& z) { return z.real() < 0.0; }, [](const C& z) { return z.real() >= 0.0; },
                                                               [](const C& z) { return std::abs(z) <= 1.0; }, [](const C& z) { return std::abs(z) > 1.0; }}) {
      EXPECT(check(real, chosen));
      EXPECT(check(complex, chosen));
    }
  }
  // ±2i over 3: the pair chosen by its second eigenvalue alone still moves whole, and counts twice.
  const matrix<double>                   rotation {{3.0, 1.0, 0.5}, {0.0, 0.0, -2.0}, {0.0, 2.0, 0.0}};
  const auto                             second   {[](const C& z) { return z.imag() < 0.0; }};
  numerical::schur_decomposition<double> s        {rotation};
  EXPECT_EQ(s.sort(second), std::size_t {2});
  EXPECT(sorted_schur_holds(rotation, s, 2, [](const C& z) { return z.imag() != 0.0; }));
  EXPECT(s.values()[0].imag() > 0.0 && s.values()[1] == std::conj(s.values()[0]));
  EXPECT_EQ(s.reorder(std::vector<bool> {false, false, true}), std::size_t {1}); // 3 back to the front
  EXPECT(s.values()[0] == C {3.0});
  EXPECT_THROWS(s.reorder(std::vector<bool> {true}), numerical::dimension_error);
  EXPECT_EQ((numerical::schur_decomposition<double> {matrix<double> {0, 0, {}}, second}.leading()), std::size_t {0});
}

// Two blocks too close to exchange (the pair of 1 ± 6e-6 i above that of 1.0000043 ± 1.4e-5 i) refuse the move, as
// trsen's info: the decomposition stays one. A selection the exchanges' rounding no longer meets past an eigenvalue not
// selected (here a selection that answers otherwise the second time) is refused as dgees's info n + 2.
TEST(a_sorted_schur_form_refuses_what_it_cannot_order)
{
  const std::vector<double> close {1, 2.9258991580319096, -0.99582361335424341, 0.6660776232485035, -3.4177527863695918e-11, 1, -0.63530008058277909,
                                   0.22425311739723819, 0, 0, 1.0000043005630168, 0.68529066378232517, 0, 0, -2.0478021681892508e-10, 1.0000043005630168};
  numerical::buffer<double> t     {close.begin(), close.end()};
  numerical::buffer<double> qt(16, 0.0);
  for (std::size_t i {0}; i < 4; ++i) {
    qt[(i * 4) + i] = 1.0;
  }
  EXPECT_THROWS(numerical::detail::move_leading(t, qt, 4, std::vector<bool> {false, false, true, true}), numerical::convergence_error);
  const matrix<double>                   a {4, 4, close};
  numerical::schur_decomposition<double> s {a};
  const std::vector<C>                   values {s.values()};
  const C                                last {values[3]};
  const auto                             lower {[&](const C& z) { return z.imag() != 0.0 && std::abs(z.real() - last.real()) < 1e-12; }};
  EXPECT_THROWS(static_cast<void>(s.sort(lower)), numerical::convergence_error);
  EXPECT(sorted_schur_holds(a, s, 0, [](const C& /*z*/) { return false; }));
  std::size_t                            calls  {0};
  const auto                             fickle {[&calls](const C& /*z*/) { return calls++ != 2; }}; // the second pass: false, then true
  numerical::schur_decomposition<double> d      {matrix<double> {{1.0, 0.0}, {0.0, 2.0}}};
  EXPECT_THROWS(static_cast<void>(d.sort(fickle)), numerical::convergence_error);
}

// matrix_balance: B = T⁻¹ A T exactly, T = P D, B[i][j] = A[p(i)][p(j)] d_j / d_i, for a matrix whose rows and columns
// isolate eigenvalues at both ends and whose middle block is graded over 2⁶⁰, real and complex, with and without each of
// the permutation and the scaling; the scaling brings the block's row and column norms within a factor of 4.
TEST(matrix_balance_is_an_exact_similarity)
{
  // The middle rows 1 … 3 graded; row 4 and column 0 have no off-diagonal entry in their block (an isolated
  // eigenvalue each, moved to the ends), row 2 none at all after them.
  const matrix<double> a {{1.0, 0x1p30, 3.0, 0x1p-30, 5.0},
    {0.0, 2.0, 0x1p40, 1.0, 7.0},
    {0.0, 0x1p-40, 3.0, 2.0, 1.0},
    {0.0, 1.0, 0x1p-20, 4.0, 2.0},
    {0.0, 0.0, 0.0, 0.0, 5.0}};
  const matrix<C>      h {{C {1.0, 1.0}, C {0x1p30, 0.0}, 3.0, 1.0, 5.0},
    {0.0, 2.0, C {0.0, 0x1p40}, 1.0, 7.0},
    {0.0, 0x1p-40, 3.0, 2.0, 1.0},
    {0.0, 1.0, C {0x1p-20, 1.0}, 4.0, 2.0},
    {0.0, 0.0, 0.0, 0.0, 5.0}};
  const auto check {[](const auto& x, bool permute, bool scale) {
                      const std::size_t n {x.rows()};
                      auto              balanced {numerical::matrix_balance(x, permute, scale)};
                      std::vector<bool> seen(n, false);
                      for (std::size_t c {0}; c < n; ++c) {
                        seen[balanced.permutation[c]] = true;
                        EXPECT(permute || balanced.permutation[c] == c);
                        EXPECT(scale || balanced.scale[c] == 1.0);
                        EXPECT(std::exp2(std::ilogb(balanced.scale[c])) == balanced.scale[c]); // a power of two
                      }
                      EXPECT(std::ranges::find(seen, false) == seen.end());
                      for (std::size_t i {0}; i < n; ++i) {
                        for (std::size_t j {0}; j < n; ++j) {
                          const auto expected {x(balanced.permutation[i], balanced.permutation[j]) * balanced.scale[j] / balanced.scale[i]};
                          EXPECT(balanced.b(i, j) == expected);
                        }
                      }
                      return balanced;
                    }};
  for (const bool permute : {true, false}) {
    for (const bool scale : {true, false}) {
      const auto real {check(a, permute, scale)};
      (void)check(h, permute, scale);
      if (permute) { // column 0 isolates an eigenvalue at the top, row 4 at the bottom
        EXPECT(real.permutation[0] == 0 && real.permutation[4] == 4);
      }
    }
  }
  const auto graded {numerical::matrix_balance(a, false, true)};
  EXPECT(graded.scale[1] != 1.0 || graded.scale[2] != 1.0 || graded.scale[3] != 1.0);
  // Row 1's only nonzero entries off its block's diagonal: a permutation that moves it, its swap's other index too.
  const matrix<double> moved   {{2.0, 1.0, 0.0}, {0.0, 3.0, 0.0}, {4.0, 5.0, 6.0}};
  const auto           swapped {numerical::matrix_balance(moved)};
  EXPECT(swapped.permutation != (std::vector<std::size_t> {0, 1, 2}));
  EXPECT_THROWS(numerical::matrix_balance(matrix<double> {{1.0, 2.0}}), numerical::dimension_error);
  EXPECT_THROWS(numerical::matrix_balance(matrix<double> {{1.0, std::nan("")}, {0.0, 1.0}}), numerical::non_finite_error);
  EXPECT(numerical::matrix_balance(matrix<double>(0, 0, std::vector<double> {})).permutation.empty());
}
