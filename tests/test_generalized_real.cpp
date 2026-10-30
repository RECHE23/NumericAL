// The generalized real Schur decomposition (real QZ, LAPACK's dgges) against its definition: A = Q S Zᵀ, B = Q T Zᵀ,
// Q and Z orthogonal, T upper triangular, S quasi-triangular with a 2 × 2 block for each conjugate pair, T diagonal on
// it; T's diagonal non-negative before a reordering (dhgeqz), a 1 × 1 block's after one (dtgsen); the eigenvalues each
// block's. Singular B (infinite eigenvalues), pencils that need exceptional shifts, the deflations of dhgeqz, the 2 × 2
// helpers (dlasv2, dlag2, dlagv2) at their branches, reordering (dtgexc, dtgex2) and its refusals.

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::matrix;
  using C = std::complex<double>;

  //! \brief The largest magnitude of \p m's entries, infinite if one is not finite.
  double largest_entry(const matrix<double>& m)
  {
    double value {0.0};
    for (const double x : m) {
      value = std::isfinite(x) ? std::max(value, std::abs(x)) : std::numeric_limits<double>::infinity();
    }
    return value;
  }

  //! \brief The `n × n` identity.
  matrix<double> unit(std::size_t n)
  {
    std::vector<double> entries(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      entries[(i * n) + i] = 1.0;
    }
    return matrix<double> {n, n, entries};
  }

  //! \brief An `n × n` matrix of `sin` of a quadratic in its entries' index, offset by \p seed: of full rank.
  matrix<double> waves(std::size_t n,
                       double      seed)
  {
    std::vector<double> entries(n * n);
    for (std::size_t e {0}; e < entries.size(); ++e) {
      const double k {static_cast<double>(e)};
      entries[e] = std::sin(seed + (0.7 * k) + (0.013 * k * k));
    }
    return matrix<double> {n, n, entries};
  }

  //! \brief `|det(β S − α T)|` of the 2 × 2 block at \p k of \p s and \p t (`T` diagonal there), each of `S` and `T`
  //!        divided by its largest entry and `(α, β)` scaled to match and to the larger of the two, so that no product
  //!        overflows, over the sum of the terms' magnitudes: a backward error, whatever the eigenvalue's size or
  //!        conditioning.
  double block_residual(const matrix<double>& s,
                        const matrix<double>& t,
                        std::size_t           k,
                        C                     alpha,
                        double                beta)
  {
    const double ns    {std::max({std::abs(s(k, k)), std::abs(s(k, k + 1)), std::abs(s(k + 1, k)), std::abs(s(k + 1, k + 1))})};
    const double nt    {std::max(std::abs(t(k, k)), std::abs(t(k + 1, k + 1)))};
    const double first {std::max(std::abs(alpha), beta)}; // β S − α T = (β ns) S / ns − (α nt) T / nt, kept in range
    alpha = (alpha / first) * nt;
    beta  = (beta / first) * ns;
    const double larger {std::max(std::abs(alpha), beta)};
    alpha /= larger;
    beta  /= larger;
    const C      d11  {(beta * s(k, k) / ns) - (alpha * t(k, k) / nt)};
    const C      d22  {(beta * s(k + 1, k + 1) / ns) - (alpha * t(k + 1, k + 1) / nt)};
    const double off  {beta * beta * (s(k, k + 1) / ns) * (s(k + 1, k) / ns)};
    const double size {((std::abs(beta * s(k, k) / ns) + std::abs(alpha * t(k, k) / nt)) * (std::abs(beta * s(k + 1, k + 1) / ns) + std::abs(alpha * t(k + 1, k + 1) / nt)))
                       + std::abs(off)};
    return std::abs((d11 * d22) - off) / size;
  }

  //! \brief Whether \p d holds a generalized real Schur decomposition of \p a and \p b to \p tolerance (relative to
  //!        their norms): the products, the orthogonal factors, the structure, `T`'s diagonal non-negative (on every
  //!        block unless \p reordered, then on the 1 × 1 ones), the values each block's.
  bool decomposes(const numerical::real_generalized_schur& d,
                  const matrix<double>&                    a,
                  const matrix<double>&                    b,
                  double                                   tolerance,
                  bool                                     reordered = false)
  {
    const std::size_t                                                                  n {a.rows()};
    const matrix<double>                                                               q {d.q()};
    const matrix<double>                                                               z {d.z()};
    const matrix<double>                                                               s {d.s()};
    const matrix<double>                                                               t {d.t()};
    bool                                                                               fine {largest_entry((q * s * numerical::transpose(z)) - a) <= tolerance * std::max(largest_entry(a), 1.0)};
    fine = fine && largest_entry((q * t * numerical::transpose(z)) - b) <= tolerance * std::max(largest_entry(b), 1.0);
    fine = fine && largest_entry((numerical::transpose(q) * q) - unit(n)) <= tolerance;
    fine = fine && largest_entry((numerical::transpose(z) * z) - unit(n)) <= tolerance;
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < i; ++j) {
        fine = fine && t(i, j) == 0.0 && (j + 1 == i || s(i, j) == 0.0);
      }
    }
    for (std::size_t k {0}; k < n; ++k) {
      const auto& value {d.values()[k]};
      if (k + 1 < n && s(k + 1, k) != 0.0) { // a 2 × 2 block: a conjugate pair, T diagonal there
        fine = fine && (k == 0 || s(k, k - 1) == 0.0) && (k + 2 >= n || s(k + 2, k + 1) == 0.0) && t(k, k + 1) == 0.0;
        fine = fine && (reordered || (t(k, k) >= 0.0 && t(k + 1, k + 1) >= 0.0));
        const auto& next {d.values()[k + 1]};
        // each with its own β (dhgeqz's), both eigenvalues of the block
        fine = fine && value.alphai > 0.0 && next.alphai < 0.0 && value.beta > 0.0 && next.beta > 0.0;
        fine = fine && block_residual(s, t, k, C {value.alphar, value.alphai}, value.beta) <= 100.0 * tolerance;
        fine = fine && block_residual(s, t, k, C {next.alphar, next.alphai}, next.beta) <= 100.0 * tolerance;
        ++k;
        continue;
      }
      fine = fine && value.alphai == 0.0 && value.alphar == s(k, k) && value.beta == t(k, k) && t(k, k) >= 0.0;
    }
    return fine;
  }

  //! \brief The count of infinite eigenvalues (`β` within \p tolerance of 0, relative to \p b) of \p d.
  std::size_t infinite_count(const numerical::real_generalized_schur& d,
                             const matrix<double>&                    b,
                             double                                   tolerance)
  {
    std::size_t count {0};
    for (const auto& value : d.values()) {
      count += value.beta <= (tolerance * largest_entry(b)) + 1e-300 ? 1 : 0;
    }
    return count;
  }
} // namespace

// Pencils of orders 0 to 40, B invertible: the decomposition to rounding, conjugate pairs and real eigenvalues both.
TEST(real_generalized_schur_decomposes_a_regular_pencil)
{
  for (const std::size_t n : {std::size_t {0}, std::size_t {1}, std::size_t {2}, std::size_t {3}, std::size_t {5}, std::size_t {12}, std::size_t {40}}) {
    const matrix<double>                    a {waves(n, 1.0)};
    const matrix<double>                    b {waves(n, 2.0) + (3.0 * unit(n))};
    const numerical::real_generalized_schur d {a, b};
    EXPECT(decomposes(d, a, b, 1e-13));
    EXPECT(infinite_count(d, b, 1e-13) == 0);
  }
  // A = B = I: every eigenvalue 1, no iteration needed past the first deflations
  const numerical::real_generalized_schur same {unit(4), unit(4)};
  EXPECT(decomposes(same, unit(4), unit(4), 1e-15));
}

// B singular: infinite eigenvalues (β = 0), its rank from 0 to n − 1; B zero altogether; B upper triangular with zeros
// on its diagonal at the top, the middle and the bottom; A and B sharing a null vector (a singular pencil).
TEST(real_generalized_schur_takes_a_singular_b)
{
  const std::size_t n {6};
  for (const std::size_t rank : {std::size_t {0}, std::size_t {1}, std::size_t {3}, std::size_t {5}}) {
    std::vector<double>  entries(n * n, 0.0);
    const matrix<double> full {waves(n, 4.0)};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        for (std::size_t k {0}; k < rank; ++k) {
          entries[(i * n) + j] += full(i, k) * full(j, k);
        }
      }
    }
    const matrix<double>                    b {n, n, entries};
    const matrix<double>                    a {waves(n, 1.0)};
    const numerical::real_generalized_schur d {a, b};
    EXPECT(decomposes(d, a, b, 1e-12));
    EXPECT(infinite_count(d, b, 1e-12) == n - rank);
  }
  for (const std::vector<std::size_t>& zeros : {std::vector<std::size_t> {0}, std::vector<std::size_t> {2}, std::vector<std::size_t> {4},
                                                std::vector<std::size_t> {1, 2}, std::vector<std::size_t> {0, 4}, std::vector<std::size_t> {0, 1, 2}}) {
    std::vector<double> bt(25, 0.0);
    std::vector<double> ah(25, 0.0);
    for (std::size_t i {0}; i < 5; ++i) {
      for (std::size_t j {0}; j < 5; ++j) {
        bt[(i * 5) + j] = j >= i ? 1.0 + static_cast<double>(i + (2 * j)) : 0.0;
        ah[(i * 5) + j] = j + 1 >= i ? std::sin(1.0 + static_cast<double>((5 * i) + j)) : 0.0;
      }
    }
    for (const std::size_t z : zeros) {
      bt[(z * 5) + z] = 0.0;
    }
    const matrix<double> a {5, 5, ah};
    const matrix<double> b {5, 5, bt};
    EXPECT(decomposes(numerical::real_generalized_schur {a, b}, a, b, 1e-12));
  }
  std::vector<double> as(16);
  std::vector<double> bs(16);
  for (std::size_t i {0}; i < 4; ++i) {
    for (std::size_t j {0}; j < 4; ++j) {
      as[(i * 4) + j] = j == 3 ? 0.0 : std::cos(static_cast<double>(i + (3 * j)));
      if (j != 3) {
        bs[(i * 4) + j] = i == j ? 2.0 : 0.1;
      }
    }
  }
  EXPECT(decomposes(numerical::real_generalized_schur {matrix<double> {4, 4, as}, matrix<double> {4, 4, bs}}, matrix<double> {4, 4, as},
                    matrix<double> {4, 4, bs}, 1e-13));
}

// The cyclic shift against the identity: eigenvalues the n-th roots of unity, all of one modulus, on which the standard
// shifts stall: the exceptional shifts break the symmetry. A bound of one iteration refuses.
TEST(real_generalized_schur_needs_exceptional_shifts)
{
  for (const std::size_t n : {std::size_t {2}, std::size_t {3}, std::size_t {4}, std::size_t {7}}) {
    std::vector<double> shift(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      shift[(((i + 1) % n) * n) + i] = 1.0;
    }
    const matrix<double>                    a {n, n, shift};
    const numerical::real_generalized_schur d {a, unit(n)};
    EXPECT(decomposes(d, a, unit(n), 1e-13));
    for (const auto& value : d.values()) {
      EXPECT(std::abs(std::abs(C {value.alphar, value.alphai} / value.beta) - 1.0) < 1e-13);
    }
    EXPECT_THROWS(static_cast<void>(numerical::real_generalized_schur(a, unit(n), 1)), numerical::convergence_error);
  }
}

// Reordering as ordqz: the chosen eigenvalues first, a conjugate pair moved whole, the decomposition kept; by a
// predicate (the left half-plane, inside the unit circle), the last block alone, nothing, everything; one choice per
// eigenvalue required.
TEST(real_generalized_schur_reorders)
{
  for (const std::size_t n : {std::size_t {4}, std::size_t {9}, std::size_t {30}}) {
    const matrix<double>              a        {waves(n, 1.0)};
    const matrix<double>              b        {waves(n, 2.0) + (3.0 * unit(n))};
    numerical::real_generalized_schur d        {a, b};
    std::size_t                       expected {0};
    for (const auto& value : d.values()) {
      expected += value.alphar / value.beta < 0.0 ? 1 : 0;
    }
    const std::size_t chosen {d.sort([](const C& alpha, const C& beta) { return (alpha / beta).real() < 0.0; })};
    EXPECT(decomposes(d, a, b, 1e-12, true));
    EXPECT(chosen == expected);
    for (std::size_t k {0}; k < n; ++k) {
      EXPECT((d.values()[k].alphar / d.values()[k].beta < 0.0) == (k < chosen));
    }
    EXPECT(d.sort([](const C& alpha, const C& beta) { return std::abs(alpha) < std::abs(beta); }) <= n);
    EXPECT(decomposes(d, a, b, 1e-12, true));
    for (std::size_t k {0}; k + 1 < n; ++k) {
      const bool inside      {std::abs(C {d.values()[k].alphar, d.values()[k].alphai}) < d.values()[k].beta};
      const bool next_inside {std::abs(C {d.values()[k + 1].alphar, d.values()[k + 1].alphai}) < d.values()[k + 1].beta};
      EXPECT(inside || !next_inside); // the chosen ones first
    }
    std::vector<bool> last(n, false);
    last[n - 1] = true;
    EXPECT(d.reorder(last) >= 1);
    EXPECT(decomposes(d, a, b, 1e-12, true));
    EXPECT(d.sort([](const C&, const C&) { return false; }) == 0);
    EXPECT(d.sort([](const C&, const C&) { return true; }) == n);
    EXPECT(decomposes(d, a, b, 1e-12, true));
    EXPECT_THROWS(static_cast<void>(d.reorder(std::vector<bool>(n - 1, true))), numerical::dimension_error);
  }
}

// The deflations of dhgeqz: a zero of T at the top chased down to the bottom, T's corner cancelled on the way; two zeros
// of T in a row; a zero of T under two small subdiagonal entries of A (their product below the tolerance) chased down;
// a pencil whose last eigenvalue takes twenty iterations (the second exceptional shift).
TEST(real_generalized_schur_deflates_as_dhgeqz)
{
  const matrix<double> cancelled_a {{1.0, 2.0}, {1.0, 3.0}};
  const matrix<double> cancelled_b {{0.0, 1.0}, {0.0, 1.0}};
  EXPECT(decomposes(numerical::real_generalized_schur {cancelled_a, cancelled_b}, cancelled_a, cancelled_b, 1e-15));
  const matrix<double> twice_a     {{1.0, 2.0, 3.0}, {1.0, 3.0, 1.0}, {0.0, 2.0, 5.0}};
  const matrix<double> twice_b     {{0.0, 0.0, 1.0}, {0.0, 0.0, 2.0}, {0.0, 0.0, 1.0}};
  EXPECT(decomposes(numerical::real_generalized_schur {twice_a, twice_b}, twice_a, twice_b, 1e-15));
  const matrix<double> small_a     {{1.0, 0.5, 0.3}, {1e-9, 2.0, 0.7}, {0.0, 1e-9, 3.0}};
  const matrix<double> small_b     {{1.0, 0.2, 0.1}, {0.0, 0.0, 0.4}, {0.0, 0.0, 1.0}};
  EXPECT(decomposes(numerical::real_generalized_schur {small_a, small_b}, small_a, small_b, 1e-15));
  const std::vector<double> slow_a {
    -1.7873903515026938e-09, 3.789057791421365e-09, -6.242438565071149e-09, 3.535112342135423e-13,
    2.511712430191712e-15, -0.07190763098391408, -1.4836301505366966e-06, -2.5893836356842995e-13,
    -4.2160010685157253e-10, -4.387850467891631e-12, 1.0117570852776248e-07, 2.1913183348759464e-08,
    3.272764127646231e-05, -0.04265277432373959, 0.0, 4.219549457216899e-05, -8.932931652572282e-15,
    1.509911510636513e-07, -1.1360161720140982e-12, 8.015826986780645e-15, -4.184709887013448e-11, 0.0, 0.0,
    -3.009673705286672e-06, 1.7690447911648228e-14, -0.004721943299768884, 5.487830273111145e-09,
    9.885945196058933e-15, 0.0, 0.0, 0.0, 0.08893501896090081, 0.0002714628054453393, 0.016656805777091204,
    3.5688585548585558e-12, 0.0, 0.0, 0.0, 0.0, 5.2810465592346945e-11, -0.05372919185949465,
    -0.00010443032543891785, 0.0, 0.0, 0.0, 0.0, 0.0, 0.024883069044305383, 1.5236675706288e-07};
  const std::vector<double> slow_b {
    0.1186392189348664, 0.8211869122513609, -0.4872971934542164, -0.9834216171060512, -0.27806726090319567,
    -0.8903616594588939, -0.6576534882926882, 0.0, 0.3377260776447546, 0.7473442865969249, 0.023279934673831493,
    -0.6399498978915432, 0.5940374902954368, 0.3496605114735054, 0.0, 0.0, -9.328183133003254e-21,
    -0.17019478464835502, -0.7062756173536571, -0.12781538352641852, -0.9880157359407798, 0.0, 0.0, 0.0, 0.0,
    -0.03957727347596862, 0.8675182225386471, -0.03876352617756873, 0.0, 0.0, 0.0, 0.0, 0.3068465063957049,
    0.15057825956006643, -0.15788517285106007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12988830060955214, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.01333800386302686};
  const matrix<double> a {7, 7, slow_a};
  const matrix<double> b {7, 7, slow_b};
  EXPECT(decomposes(numerical::real_generalized_schur {a, b}, a, b, 1e-13));
  // a near-double eigenvalue: complex by dlag2 before the 2 × 2 block's standardization, real after it, so that the
  // next iteration splits the block (found by a search)
  const matrix<double>                    near_a {{-0.014254250588293634, -0.71349484027968002}, {4.7237779598755056e-17, -0.014477072681084148}};
  const matrix<double>                    near_b {{1.1150342198652257, 0.99351089534026427}, {0.0, 1.1324644072233323}};
  const numerical::real_generalized_schur near   {near_a, near_b};
  EXPECT(decomposes(near, near_a, near_b, 1e-15));
  // A some 10³¹⁰ times B, its eigenvalues of one modulus: the exceptional shift past maxit · safmin times the
  // subdiagonal grows by 1 / (safmin maxit), scaled down to the range
  for (const std::size_t n : {std::size_t {3}, std::size_t {5}}) {
    std::vector<double> shift(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      shift[(((i + 1) % n) * n) + i] = 1e300;
    }
    const matrix<double>                    big   {n, n, shift};
    const matrix<double>                    small {1e-10 * unit(n)};
    const numerical::real_generalized_schur d     {big, small};
    EXPECT(decomposes(d, big, small, 1e-13));
    // B of norm past 15 n with a small diagonal entry: the exceptional shift then past half the range over ‖B‖
    std::vector<double> graded(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      graded[(i * n) + i] = i + 2 == n ? 1e-7 : 100.0;
    }
    const matrix<double>                    lopsided {n, n, graded};
    const numerical::real_generalized_schur e        {big, lopsided};
    EXPECT(decomposes(e, big, lopsided, 1e-13));
  }
}

// Pencils past the square root of the range either way, brought into it as dgges does and taken back: the products
// within rounding of A and B themselves; each block's eigenvalues its own. Pairs whose α or β would leave the range
// when taken back, rescaled whole first (dgges's): a tiny real part, a tiny imaginary part, a tiny β, α past 1 / safmin;
// and a pair of αr = 0, which takes the imaginary part's test.
TEST(real_generalized_schur_scales_as_dgges)
{
  const auto check = [](const matrix<double>& a, const matrix<double>& b) {
                       const numerical::real_generalized_schur d    {a, b};
                       const matrix<double>                    s    {d.s()};
                       const matrix<double>                    t    {d.t()};
                       bool                                    fine {largest_entry((d.q() * s * numerical::transpose(d.z())) - a) <= 1e-13 * largest_entry(a)};
                       fine = fine && largest_entry((d.q() * t * numerical::transpose(d.z())) - b) <= 1e-13 * largest_entry(b);
                       for (std::size_t k {0}; k < a.rows(); ++k) {
                         const auto& v {d.values()[k]};
                         if (v.alphai == 0.0) {
                           fine = fine && v.alphar == s(k, k) && v.beta == t(k, k);
                           continue;
                         }
                         fine = fine && std::isfinite(v.alphar) && std::isfinite(v.alphai) && v.beta > 0.0 && std::isfinite(v.beta);
                         fine = fine && block_residual(s, t, v.alphai > 0.0 ? k : k - 1, C {v.alphar, v.alphai}, v.beta) <= 1e-12;
                       }
                       return fine;
                     };
  for (const double scale : {1e-300, 3e-308, 1e300, 1e307}) {
    for (const bool on_b : {false, true}) {
      EXPECT(check((on_b ? 1.0 : scale) * waves(9, 1.0), (on_b ? scale : 1.0) * (waves(9, 2.0) + (3.0 * unit(9)))));
    }
  }
  EXPECT(check(1e-300 * matrix<double> {{1e-9, 1.0}, {-1.0, 1e-9}}, unit(2)));                                 // αr tiny
  EXPECT(check(1e-300 * matrix<double> {{1.0, 1e-9}, {-1e-9, 1.0}}, unit(2)));                                 // αi tiny
  EXPECT(check(matrix<double> {{0.0, 1.0}, {-1.0, 0.0}}, 1e-300 * matrix<double> {{1.0, 0.0}, {0.0, 3e-16}})); // β tiny
  EXPECT(check(1e-300 * matrix<double> {{0.0, 1.0}, {-1.0, 0.0}}, unit(2)));                                   // αr = 0
  EXPECT(check(1.5e308 * matrix<double> {{0.6, 0.8}, {-0.8, 0.6}}, unit(2)));                                  // α past 1 / safmin
  EXPECT(check(1.5e308 * matrix<double> {{1e-3, 1.0}, {-1.0, 1e-3}}, unit(2)));                                // αi alone past it
  EXPECT(check(matrix<double> {{0.0, 1.0}, {-1.0, 0.0}}, 1e308 * unit(2)));                                    // β past it
  // ‖A‖ 10¹³⁸ with an active block of 10⁻¹⁷⁵: the single shift's scale past half the range over ‖A‖, scaled down
  EXPECT(check(matrix<double> {{1e138, 0.0, 0.0}, {0.0, 2e-175, 1e-175}, {0.0, 1e-175, 3e-175}}, unit(3)));
}

// An exchange the strong test refuses at any common scale (entries 10²⁰⁰ apart, found by a search): reorder throws,
// the decomposition as it was.
TEST(real_generalized_schur_refuses_an_unstable_exchange)
{
  const matrix<double>              a {{4.0433738246568382e-290, -4.1728728744498313e-195}, {0.0, -7.7732712242752404e-158}};
  const matrix<double>              b {{-1.3394732758970938e-222, -2.3968178730587086e-295}, {0.0, -7.9217174196872459e-184}};
  numerical::real_generalized_schur d {a, b};
  const matrix<double>              s {d.s()};
  const matrix<double>              z {d.z()};
  EXPECT_THROWS(static_cast<void>(d.reorder(std::vector<bool> {false, true})), numerical::convergence_error);
  EXPECT((d.s() == s) && (d.z() == z));
}

// The refusals of the decomposition.
TEST(real_generalized_schur_refuses_what_it_cannot_take)
{
  EXPECT_THROWS(static_cast<void>(numerical::real_generalized_schur(matrix<double> {{1.0, 2.0}}, unit(1))), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::real_generalized_schur(unit(2), unit(1))), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::real_generalized_schur(unit(1), matrix<double> {{1.0}, {2.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::real_generalized_schur(unit(1), matrix<double> {{1.0, 2.0}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::real_generalized_schur(matrix<double> {{std::numeric_limits<double>::quiet_NaN()}}, unit(1))),
                numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::real_generalized_schur(unit(1), matrix<double> {{std::numeric_limits<double>::infinity()}})),
                numerical::non_finite_error);
}

// dlasv2 at each of its branches: a diagonal triangle (g = 0); g dominant past ε, h above and below 1; g negligible
// beside equal and unequal diagonal entries (its square underflowing); g the largest without dominating, after the
// swap of a larger h. The rotations take the triangle to diag(ssmax, ssmin), |ssmax| ≥ |ssmin|.
TEST(real_qz_svd_of_a_triangle)
{
  const std::vector<std::array<double, 3>> triangles {{3.0, 0.0, -2.0},   {2.0, 1e20, 1.5},     {2.0, -1e20, 0.5},  {1.0, 1e-170, 1.0}, {-1.0, 1e-170, 0.5},
    {1.0, 3.0, 2.0},    {-0.5, 2.0, 4.0},     {1.0, 1.0, 1.0},    {5.0, -1.0, 0.0},   {0.0, 1.0, 2.0}};
  for (const auto& [f, g, h] : triangles) {
    const numerical::detail::triangle_svd svd {numerical::detail::svd_of_triangle(f, g, h)};
    // [csl snl; −snl csl] [f g; 0 h] [csr −snr; snr csr]
    const double a00                          {(svd.csl * f)};
    const double a01                          {(svd.csl * g) + (svd.snl * h)};
    const double a10                          {-svd.snl * f};
    const double a11                          {(-svd.snl * g) + (svd.csl * h)};
    const double d00                          {(a00 * svd.csr) + (a01 * svd.snr)};
    const double d01                          {(-a00 * svd.snr) + (a01 * svd.csr)};
    const double d10                          {(a10 * svd.csr) + (a11 * svd.snr)};
    const double d11                          {(-a10 * svd.snr) + (a11 * svd.csr)};
    const double size                         {std::max({std::abs(f), std::abs(g), std::abs(h)})};
    EXPECT(std::abs(d00 - svd.ssmax) <= 1e-14 * size && std::abs(d11 - svd.ssmin) <= 1e-14 * size);
    EXPECT(std::abs(d01) <= 1e-14 * size && std::abs(d10) <= 1e-14 * size);
    EXPECT(std::abs(svd.ssmax) >= std::abs(svd.ssmin));
  }
}

// dlag2 against the roots of det(A − w B) at its scalings: a tiny diagonal entry of B lifted to its threshold (first and
// second), the discriminant past the square root of the range, eigenvalues large enough that no scaling is needed, a
// conjugate pair.
TEST(real_qz_pencil_eigenvalues)
{
  const double safmin {std::numeric_limits<double>::min()};
  struct pencil
  {
    std::array<double, 4> a;
    std::array<double, 3> b;
  };
  const std::vector<pencil> pencils {{.a = {1.0, 0.5, 0.25, 2.0}, .b = {1.0, 0.3, 1.5}},   {.a = {1.0, 2.0, 3.0, 4.0}, .b = {1e-200, 1.0, 1.0}}, {.a = {1.0, 2.0, 3.0, 4.0}, .b = {1.0, 1.0, 1e-200}},
    {.a = {1.0, 0.0, 0.0, 10.0}, .b = {1.0, 0.0, 1e-200}}, {.a = {1.0, 0.0, 0.0, 1.0}, .b = {1.0, 0.0, 0.25}},  {.a = {0.0, 1.0, -4.0, 0.0}, .b = {1.0, 0.0, 0.1}},
    {.a = {1.0, -2.0, 3.0, 1.0}, .b = {2.0, 0.5, 1.0}},
    // B₂₂ lifted to √safmin = 2⁻⁵¹¹ exactly: pp √safmin = 1, the discriminant taken scaled
    {.a = {0.0, 0.0, -1.0, 1.0}, .b = {1.0, 1.0, 1e-200}},
    // A below safmin, B past 1: each eigenvalue's scale past 1 (wsize > 1)
    {.a = {1e-310, 0.0, 0.0, 2e-310}, .b = {4.0, 0.0, 2.0}}};
  for (const pencil& p : pencils) {
    const auto [a11, a12, a21, a22] {p.a};
    const auto [b11, b12, b22] {p.b};
    const numerical::detail::pencil_pair pair {numerical::detail::pencil_eigenvalues(a11, a12, a21, a22, b11, b12, b22, safmin)};
    EXPECT(std::isfinite(pair.wr1) && std::isfinite(pair.wr2) && std::isfinite(pair.wi) && pair.scale1 > 0.0 && pair.scale2 > 0.0);
    // det(A − w B) at w: (a11 − w b11)(a22 − w b22) − (a12 − w b12) a21, over its terms' sizes
    const auto residual = [&](const C& w) {
                            const C      det  {((a11 - (w * b11)) * (a22 - (w * b22))) - ((a12 - (w * b12)) * a21)};
                            const double size {((std::abs(a11) + (std::abs(w) * std::abs(b11))) * (std::abs(a22) + (std::abs(w) * std::abs(b22))))
                                               + ((std::abs(a12) + (std::abs(w) * std::abs(b12))) * std::abs(a21))};
                            return std::abs(det) / size;
                          };
    // a root of the lifted pencil, not of this one; A subnormal, its roots to its few digits
    const bool lifted {std::abs(b11) < 1e-150 || std::abs(b22) < 1e-150 || std::abs(a11) < 1e-300};
    if (!lifted) {
      EXPECT(residual(C {pair.wr1, pair.wi} / pair.scale1) <= 1e-14 && residual(C {pair.wr2, -pair.wi} / pair.scale2) <= 1e-14);
    }
  }
}

// dlagv2 at its branches: A's subdiagonal negligible; B's first or second diagonal entry negligible; two real
// eigenvalues, the right rotation from either row of A − w B and the left from B or from A; a conjugate pair. The
// rotations take the pencil to triangles (B diagonal for a pair), the pencil rebuilt from them.
TEST(real_qz_standardize_pencil)
{
  using block = std::array<std::array<double, 2>, 2>;
  const std::vector<std::pair<block, block>> pencils {
    {{{{1.0, 2.0}, {1e-20, 3.0}}}, {{{1.0, 1.0}, {0.0, 2.0}}}},  {{{{1.0, 2.0}, {3.0, 4.0}}}, {{{1e-20, 1.0}, {0.0, 2.0}}}},
    {{{{1.0, 2.0}, {3.0, 4.0}}}, {{{2.0, 1.0}, {0.0, 1e-20}}}},  {{{{2.0, 1.0}, {1.0, 2.0}}}, {{{1.0, 0.0}, {0.0, 1.0}}}},
    {{{{2.0, 1.0}, {4.0, 2.0}}}, {{{1.0, 3.0}, {0.0, 1.0}}}},    {{{{1.0, 5.0}, {1.0, -3.0}}}, {{{4.0, 0.5}, {0.0, 0.25}}}},
    {{{{0.0, 1.0}, {-1.0, 0.0}}}, {{{1.0, 0.0}, {0.0, 1.0}}}},   {{{{1.0, -3.0}, {2.0, 1.0}}}, {{{2.0, 0.7}, {0.0, 0.5}}}}};
  for (const auto& [a0, b0] : pencils) {
    block                                     a {a0};
    block                                     b {b0};
    const numerical::detail::pencil_rotations g {numerical::detail::standardize_pencil(a, b)};
    EXPECT(a[1][0] == 0.0 || (b[1][0] == 0.0 && b[0][1] == 0.0)); // triangles, or B diagonal beside a pair
    EXPECT(b[1][0] == 0.0);
    // A = Lᵀ A' Rᵀ, L = [csl snl; −snl csl], R = [csr −snr; snr csr]
    for (const auto& [before, after] : {std::pair {a0, a}, std::pair {b0, b}}) {
      for (std::size_t i {0}; i < 2; ++i) {
        for (std::size_t j {0}; j < 2; ++j) {
          double rebuilt {0.0};
          for (std::size_t k {0}; k < 2; ++k) {
            for (std::size_t l {0}; l < 2; ++l) {
              const double off_left  {k == 0 ? g.snl : -g.snl};  // Lᵀ(i, k) = L(k, i) off the diagonal
              const double off_right {l == 0 ? g.snr : -g.snr};  // Rᵀ(l, j) off it
              const double left      {k == i ? g.csl : off_left};
              const double right     {l == j ? g.csr : off_right};
              rebuilt += left * after[k][l] * right;
            }
          }
          EXPECT(std::abs(rebuilt - before[i][j]) <= 1e-14 * 8.0);
        }
      }
    }
  }
}

namespace {

  //! \brief A real QZ form of \p s and \p t (`n × n` row-major), `Q = Z = I`.
  numerical::detail::real_qz_form form_of(std::size_t                n,
                                          const std::vector<double>& s,
                                          const std::vector<double>& t)
  {
    numerical::detail::real_qz_form f;
    f.n  = n;
    f.s  = numerical::buffer<double>(s.begin(), s.end());
    f.t  = numerical::buffer<double>(t.begin(), t.end());
    f.qt = numerical::buffer<double>(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      f.qt[(i * n) + i] = 1.0;
    }
    f.zt = f.qt;
    return f;
  }

  //! \brief Whether \p f still factors the pencil (\p s, \p t) it was formed from (`Q = Z = I` then) to \p tolerance
  //!        of their largest entries, `Q` and `Z` orthogonal.
  bool still_factors(const numerical::detail::real_qz_form& f,
                     const std::vector<double>&             s,
                     const std::vector<double>&             t,
                     double                                 tolerance)
  {
    const std::size_t                                                                                          n {f.n};
    const matrix<double>                                                                                       q {numerical::transpose(matrix<double> {n, n, f.qt})};
    const matrix<double>                                                                                       z {numerical::transpose(matrix<double> {n, n, f.zt})};
    const matrix<double>                                                                                       a {n, n, s};
    const matrix<double>                                                                                       b {n, n, t};
    bool                                                                                                       fine {largest_entry((q * matrix<double> {n, n, f.s} *numerical::transpose(z)) - a) <= tolerance * largest_entry(a)};
    fine = fine && largest_entry((q * matrix<double> {n, n, f.t} *numerical::transpose(z)) - b) <= tolerance * largest_entry(b);
    fine = fine && largest_entry((numerical::transpose(q) * q) - unit(n)) <= tolerance;
    return fine && largest_entry((numerical::transpose(z) * z) - unit(n)) <= tolerance;
  }
} // namespace

// dhgeqz on a pencil left unscaled (as real_generalized_schur scales it, these stay out of its reach): A some 10³¹⁰
// times B, eigenvalues of one modulus, so that the exceptional shift grows by 1 / (safmin maxit); B also of norm past
// 15 n with a small diagonal entry, so that the shift is scaled to the range over ‖B‖.
TEST(real_qz_iteration_takes_extreme_ratios)
{
  for (const std::size_t n : {std::size_t {3}, std::size_t {5}}) {
    std::vector<double> shift(n * n, 0.0);
    std::vector<double> small(n * n, 0.0);
    std::vector<double> graded(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      shift[(((i + 1) % n) * n) + i] = 1e300;
      small[(i * n) + i]             = 1e-10;
      graded[(i * n) + i]            = i + 2 == n ? 1e-7 : 100.0;
    }
    for (const std::vector<double>* b : {&small, &graded}) {
      numerical::detail::real_qz_form f      {numerical::detail::real_hessenberg_triangular(matrix<double> {n, n, shift}, matrix<double> {n, n, *b})};
      const auto                      values {numerical::detail::real_qz_iteration(f, 30 * n)};
      EXPECT(values.size() == n);
      // the form against the pencil: Q S Zᵀ = A, Q T Zᵀ = B
      const matrix<double> q {numerical::transpose(matrix<double> {n, n, f.qt})};
      const matrix<double> z {numerical::transpose(matrix<double> {n, n, f.zt})};
      EXPECT(largest_entry((q * matrix<double> {n, n, f.s} *numerical::transpose(z)) - matrix<double> {n, n, shift}) <= 1e-13 * 1e300);
      EXPECT(largest_entry((q * matrix<double> {n, n, f.t} *numerical::transpose(z)) - matrix<double> {n, n, *b}) <= 1e-13 * largest_entry(matrix<double> {n, n, *b}));
    }
  }
}

// The double-shift sweep where the 2 × 2 of T below the bulge has no null vector to take: every entry below safmin
// (the reflection then e₁ ↔ e₂), and of rank one exactly (w₂₂ eliminated to 0). A(2, 1) = 0 makes the first
// reflection the identity, so that block is T's own; A's second column 0 keeps the shifts finite beside it, and the
// order 5 keeps the trailing 2 × 2 the shifts read apart from it. The sweep keeps A = Q S Zᵀ and B = Q T Zᵀ.
TEST(real_qz_sweep_takes_a_singular_block)
{
  const std::vector<double> h {4.0, 0.0, 2.0, 3.0, 1.0,  0.0, 0.0, 1.0, 2.0, 0.5,  0.0, 0.0, 5.0, 1.0, 0.3,
                               0.0, 0.0, 2.0, 6.0, 1.0,  0.0, 0.0, 0.0, 1.5, 3.0};
  for (const std::array<double, 3>& w : {std::array<double, 3> {1e-310, 2e-310, 3e-310}, std::array<double, 3> {2.0, 2.0, 0.0}}) {
    // T upper triangular: T(1, 1) = 1, its row past it 0, the 2 × 2 at (2, 2) the block, then 1 and 2
    const std::vector<double> t       {1.0, 0.0,  0.0,  0.5, 0.1,  0.0, w[0], w[1], 0.3, 0.2,  0.0, 0.0, w[2], 0.2, 0.4,
                                       0.0, 0.0,  0.0,  1.0, 0.6,  0.0, 0.0,  0.0,  0.0, 2.0};
    numerical::detail::real_qz_form f {form_of(5, h, t)};
    numerical::detail::double_shift_sweep(f, 1, 5, 1.0 / 8.0, 1.0 / 3.0, std::numeric_limits<double>::min());
    EXPECT(still_factors(f, h, t, 1e-14));
  }
}

// dhgeqz's α and β of a degenerate pair: the rotations' guards where c₂₂ and the first column of the rotated block both
// vanish (an imaginary part of 10⁻³¹⁰); the values finite, the β positive.
TEST(real_qz_pair_values_take_a_degenerate_block)
{
  numerical::detail::real_qz_form      f    {form_of(2, {1.0, 0.0, 0.0, 1.0}, {1.0, 0.0, 0.0, 1.0})};
  const numerical::detail::pencil_pair pair {.scale1 = 1.0, .scale2 = 1.0, .wr1 = 1.0, .wr2 = 1.0, .wi = 1e-310};
  const auto [first, second] {numerical::detail::pair_values(f, 2, pair, 1.0, 1.0)};
  for (const auto& v : {first, second}) {
    EXPECT(std::isfinite(v.alphar) && std::isfinite(v.alphai) && v.beta > 0.0 && std::isfinite(v.beta));
  }
}

// The generalized Sylvester equations of dtgsy2: solved for blocks apart, refused for two equal blocks (a pivot below
// ε times the largest) and for a solution past the range; an exchange of two equal 2 × 2 blocks refused, the form kept.
TEST(real_qz_generalized_sylvester)
{
  numerical::detail::small_square s;
  numerical::detail::small_square t;
  const std::array<double, 16>    blocks {1.0, 2.0, 0.5, 0.25, -3.0, 1.0, 0.75, -0.5, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, -2.0, 4.0};
  for (std::size_t e {0}; e < 16; ++e) {
    s.e[e] = blocks[e];
    const bool coupled {e == 2 || e == 7};
    t.e[e] = coupled ? 0.3 : 0.0;
    if (e % 5 == 0) {
      t.e[e] = 1.0;
    }
  }
  const auto solved {numerical::detail::generalized_sylvester(s, t, 2, 2)};
  EXPECT(solved.has_value());
  if (solved) { // S₁₁ R − L S₂₂ = S₁₂, T₁₁ R − L T₂₂ = T₁₂
    const auto& [r, l] {*solved};
    for (std::size_t i {0}; i < 2; ++i) {
      for (std::size_t j {0}; j < 2; ++j) {
        double first  {-s(i, 2 + j)};
        double second {-t(i, 2 + j)};
        for (std::size_t k {0}; k < 2; ++k) {
          first  += (s(i, k) * r(k, j)) - (l(i, k) * s(2 + k, 2 + j));
          second += (t(i, k) * r(k, j)) - (l(i, k) * t(2 + k, 2 + j));
        }
        EXPECT(std::abs(first) <= 1e-14 * 8.0 && std::abs(second) <= 1e-14 * 8.0);
      }
    }
  }
  numerical::detail::small_square same {s};
  for (std::size_t i {0}; i < 2; ++i) {
    for (std::size_t j {0}; j < 2; ++j) {
      same(2 + i, 2 + j) = s(i, j);
    }
  }
  numerical::detail::small_square same_t {numerical::detail::small_square::identity(4)};
  EXPECT(!numerical::detail::generalized_sylvester(same, same_t, 2, 2).has_value());
  numerical::detail::small_square far    {s}; // the blocks 10⁻³ of theirs, S₁₂ 10³⁰⁸: the solution past the range
  numerical::detail::small_square far_t  {t};
  for (std::size_t i {0}; i < 4; ++i) {
    for (std::size_t j {0}; j < 4; ++j) {
      const bool   block {(i < 2) == (j < 2)};
      const double off   {i < 2 ? 1e308 : 0.0};
      far(i, j)   = block ? 1e-3 * s(i, j) : off;
      far_t(i, j) = block ? 1e-3 * t(i, j) : t(i, j);
    }
  }
  EXPECT(!numerical::detail::generalized_sylvester(far, far_t, 2, 2).has_value());
  // two equal pairs: the exchange refused, the form as it was
  const std::vector<double>       pair_s {1.0, 2.0, 0.5, 0.25, -3.0, 1.0, 0.75, -0.5, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, -3.0, 1.0};
  const std::vector<double>       pair_t {1.0, 0.0, 0.3, 0.0, 0.0, 1.0, 0.0, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  numerical::detail::real_qz_form f      {form_of(4, pair_s, pair_t)};
  EXPECT(!numerical::detail::exchange_real_blocks(f, 1, 2, 2));
  EXPECT(f.s == numerical::buffer<double>(pair_s.begin(), pair_s.end()) && f.t == numerical::buffer<double>(pair_t.begin(), pair_t.end()));
}

// dtgex2's refusals at each test (found by a search over blocks of scales 10±³⁰⁰): two 1 × 1 blocks whose new
// subdiagonal entries pass the weak test, S's and T's alone; a 2 × 2 block past a 1 × 1 one whose RQ and QR both leave
// too much below S's diagonal, the QR's the smaller. The form left as it was.
TEST(real_qz_exchange_refusals)
{
  struct pencil
  {
    std::size_t         n;
    std::size_t         n1;
    std::vector<double> s;
    std::vector<double> t;
  };
  const std::vector<pencil> refused {
    {.n = 2, .n1 = 1, .s = {-3.0057259403010486, -1.8999426808341242e+291, 0.0, -5.9155888389379269e+255}, .t = {5.7429595585228079e+19, 1.112278457023524e-22, 0.0, 1.9094161655057504e+127}},
    {.n = 2, .n1 = 1, .s = {-6.8511926541261825e-223, 4.0411364574960022e-298, 0.0, 8.4742347027664837e-274}, .t = {5.4784635402511913e-88, 1.1742168413169882e-91, 0.0, 1.3076170457115753e-142}},
    {.n = 3, .n1 = 2, .s =
     {-4.6565778441985185e-09, -6304939.5291651562, -7.9434156338329713e+24, 3.0711053328576424e+21, -2.6573477638074699e+20, 1.6985825987068891e-26, 0.0, 0.0,
      3.9040422157117758e-21},
     .t = {8.6268949490307261e-34, 0.0, -0.11597725162406085, 0.0, 4.2605037562985242e-19, 6.8626317255168812e-10, 0.0, 0.0, 83087676672206640.0}},
    {.n = 3, .n1 = 2, .s =
     {-1.5231999285009249e-09, 0.16556998522173935, 0.00025655080115448084, -6.4166032052487772e-06, -1.2566692469877067e-29, 4.9514418809284183e-05, 0.0, 0.0,
      3.7628437696331179e-11},
     .t = {6.4112259458608051e-09, 0.0, 2.8724281039258287e+47, 0.0, 0.00069850816609372756, 2.9881277124864555e-51, 0.0, 0.0, 6.9467197679413046e-63}}};
  for (const pencil& p : refused) {
    numerical::detail::real_qz_form f {form_of(p.n, p.s, p.t)};
    EXPECT(!numerical::detail::exchange_real_blocks(f, 1, p.n1, p.n - p.n1));
    EXPECT(f.s == numerical::buffer<double>(p.s.begin(), p.s.end()) && f.t == numerical::buffer<double>(p.t.begin(), p.t.end()));
  }
}

namespace {

  //! \brief The eigenvalue `S(k, k) / T(k, k)` of a 1 × 1 block of \p f (from 0).
  double single_value(const numerical::detail::real_qz_form& f,
                      std::size_t                            k)
  {
    return f.s[(k * f.n) + k] / f.t[(k * f.n) + k];
  }
} // namespace

// dtgexc's moves up of a block: a place on a 2 × 2 block's second row taken to its first, for the block moved and for
// its destination; a "2 × 2" block of real eigenvalues (as a form's block of a nearly double pair can be) that splits as
// it is standardized, then moved as its two 1 × 1 ones past a 1 × 1 block, past a conjugate pair kept whole, and past
// another such block that splits in passing. T = I, so each block's eigenvalues are S's. The form still factors the
// pencil; the eigenvalues moved lead.
TEST(real_qz_moves_blocks_up)
{
  const auto check = [](std::size_t n, const std::vector<double>& s, std::size_t from, std::size_t to, const std::vector<double>& leading) {
                       std::vector<double> t(n * n, 0.0);
                       for (std::size_t i {0}; i < n; ++i) {
                         t[(i * n) + i] = 1.0;
                       }
                       numerical::detail::real_qz_form f {form_of(n, s, t)};
                       numerical::detail::move_real_block_up(f, from, to);
                       bool fine                         {still_factors(f, s, t, 1e-13)};
                       for (std::size_t k {0}; k < leading.size(); ++k) { // as a set: the split block's two in either order
                         bool found {false};
                         for (const double value : leading) {
                           found = found || std::abs(single_value(f, k) - value) <= 1e-12 * std::max(1.0, std::abs(value));
                         }
                         fine = fine && found && (k + 1 >= n || f.s[((k + 1) * n) + k] == 0.0);
                       }
                       return fine;
                     };
  // two conjugate pairs, the second moved up from its second row to the first's: 4 ± i√2 then leads
  const std::vector<double> pairs {1.0, 2.0, 0.5, 0.3, -3.0, 1.0, 0.2, 0.1, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, -2.0, 4.0};
  std::vector<double>       t4(16, 0.0);
  for (std::size_t i {0}; i < 4; ++i) {
    t4[(i * 4) + i] = 1.0;
  }
  numerical::detail::real_qz_form f {form_of(4, pairs, t4)};
  numerical::detail::move_real_block_up(f, 4, 2);
  EXPECT(still_factors(f, pairs, t4, 1e-13) && f.s[4] != 0.0 && std::abs((((f.s[0] / f.t[0]) + (f.s[5] / f.t[5])) / 2.0) - 4.0) <= 1e-12);
  // a block of eigenvalues 6 and 8 that splits, past 1 × 1 blocks of 7 and 5
  EXPECT(check(4, {7.0, 0.4, 0.2, 0.1, 0.0, 5.0, 0.3, 0.2, 0.0, 0.0, 7.0, 1.0, 0.0, 0.0, 1.0, 7.0}, 3, 1, {6.0, 8.0}));
  // the same past a 1 × 1 block of 5 and a conjugate pair ±i√2 above it
  EXPECT(check(5, {0.0, 1.0, 0.3, 0.2, 0.1, -2.0, 0.0, 0.4, 0.1, 0.2, 0.0, 0.0, 5.0, 0.3, 0.2, 0.0, 0.0, 0.0, 7.0, 1.0, 0.0, 0.0, 0.0, 1.0, 7.0}, 4, 1,
               {6.0, 8.0}));
  // past a 1 × 1 block of 5 and a block of eigenvalues 1 and 3 that splits in passing
  EXPECT(check(5, {2.0, 1.0, 0.3, 0.2, 0.1, 1.0, 2.0, 0.4, 0.1, 0.2, 0.0, 0.0, 5.0, 0.3, 0.2, 0.0, 0.0, 0.0, 7.0, 1.0, 0.0, 0.0, 0.0, 1.0, 7.0}, 4, 1,
               {6.0, 8.0}));
}
