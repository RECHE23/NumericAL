// The symmetric indefinite factorisation against its definition and SciPy's: A = L D Lᴴ (Hermitian) or L D Lᵀ
// (symmetric) for real, complex Hermitian and complex symmetric matrices, from either triangle; L triangular with a unit
// diagonal once its rows are permuted, D block diagonal with blocks of order 1 and 2; SciPy's answers where ties decide
// the pivots; zero columns; the diagonal's imaginary parts ignored for a Hermitian matrix; the refusals.

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::matrix;
  using C = std::complex<double>;

  //! \brief A value in [−1, 1) from \p state (a linear congruential sequence, the same on every platform).
  double drawn(std::uint64_t& state)
  {
    state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
    return (static_cast<double>(state >> 11) / static_cast<double>(1ULL << 53) * 2.0) - 1.0;
  }

  //! \brief \p z conjugated where \p hermitian (a real number as it is).
  C transposed_entry(const C& z,
                     bool     hermitian)
  {
    return hermitian ? std::conj(z) : z;
  }

  //! \brief An `n × n` matrix with `A(j, i) = A(i, j)` (conjugated where \p hermitian), complex entries where \p complex.
  matrix<C> symmetric(std::size_t    n,
                      bool           complex,
                      bool           hermitian,
                      std::uint64_t& state)
  {
    std::vector<C> a(n * n);
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j <= i; ++j) {
        C z {drawn(state), complex ? drawn(state) : 0.0};
        if (i == j && hermitian) {
          z = C {z.real()};
        }
        a[(i * n) + j] = z;
        a[(j * n) + i] = transposed_entry(z, hermitian);
      }
    }
    return matrix<C> {n, n, a};
  }

  //! \brief \p m's real parts.
  matrix<double> real_of(const matrix<C>& m)
  {
    std::vector<double> entries;
    for (const C& z : m) {
      entries.push_back(z.real());
    }
    return matrix<double> {m.rows(), m.cols(), entries};
  }

  //! \brief \p m's entries as complex numbers.
  matrix<C> as_complex(const matrix<double>& m)
  {
    return matrix<C> {m.rows(), m.cols(), std::vector<C>(m.begin(), m.end())};
  }

  //! \brief A copy of \p m.
  matrix<C> as_complex(const matrix<C>& m)
  {
    return m;
  }

  /*!
   * \brief Whether \p f factorises \p a (`n × n`): `L[perm]` triangular (lower or upper as \p lower) with a unit
   *        diagonal, `D` block diagonal with blocks of order 1 and 2 (symmetric, or Hermitian where \p hermitian), and
   *        `L D Lᴴ` (`L D Lᵀ`) within \p tolerance of `A` relative to its largest entry.
   */
  template <typename T>
  bool factorises(const numerical::ldl_decomposition<T>& f,
                  const matrix<T>&                       a,
                  bool                                   lower,
                  bool                                   hermitian,
                  double                                 tolerance)
  {
    const std::size_t               n    {a.rows()};
    const matrix<C>                 l    {as_complex(f.factor())};
    const matrix<C>                 d    {as_complex(f.d())};
    const std::vector<std::size_t>& p    {f.permutation()};
    bool                            fine {p.size() == n && l.rows() == n && d.rows() == n};
    for (std::size_t i {0}; i < n && fine; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        const C entry   {l(p[i], j)};
        fine = fine && (i == j ? entry == C {1.0} : ((lower ? j < i : j > i) || entry == C {}));
        const bool near {(i > j ? i - j : j - i) <= 1};
        fine = fine && (near || d(i, j) == C {});
        fine = fine && d(j, i) == transposed_entry(d(i, j), hermitian);
      }
    }
    for (std::size_t i {0}; i + 1 < n && fine; ++i) { // a block of order 2 is followed by one of order 1 or another
      fine = fine && (d(i + 1, i) == C {} || i + 2 >= n || d(i + 2, i + 1) == C {});
    }
    std::vector<C> conjugate(n * n); // Lᴴ (Lᵀ)
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        conjugate[(j * n) + i] = transposed_entry(l(i, j), hermitian);
      }
    }
    const matrix<C> product {(l * d) * matrix<C> {n, n, conjugate}};
    double          gap     {0.0};
    double          size    {1.0};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        gap  = std::max(gap, std::abs(product(i, j) - C {as_complex(a)(i, j)}));
        size = std::max(size, std::abs(C {as_complex(a)(i, j)}));
      }
    }
    return fine && gap <= tolerance * size;
  }

  // Orders 0 to 30, real (symmetric), complex Hermitian and complex symmetric, each triangle: the definition to rounding.
  TEST(ldl_factorises)
  {
    std::uint64_t state {3};
    for (const std::size_t n : {std::size_t {0}, std::size_t {1}, std::size_t {2}, std::size_t {5}, std::size_t {12}, std::size_t {30}}) {
      for (const bool lower : {true, false}) {
        const matrix<C> real {symmetric(n, false, true, state)};
        for (const bool hermitian : {true, false}) { // a real matrix's factorisation either way
          EXPECT(factorises(numerical::ldl_decomposition<double> {real_of(real), lower, hermitian}, real_of(real), lower, hermitian, 1e-13));
        }
        for (const bool hermitian : {true, false}) {
          const matrix<C> complex {symmetric(n, true, hermitian, state)};
          EXPECT(factorises(numerical::ldl_decomposition<C> {complex, lower, hermitian}, complex, lower, hermitian, 1e-13));
        }
      }
    }
  }

  /*!
   * \brief Whether the panels (\ref numerical::detail::bunch_kaufman_blocked_lower) and the unblocked steps
   *        (\ref numerical::detail::bunch_kaufman_lower), each checked against the definition elsewhere, factorise the
   *        lower triangle of \p a alike: the same pivots, and each entry of the triangle within 1e-12 of the largest.
   */
  template <typename T>
  bool panels_agree(const matrix<T>& a,
                    bool             hermitian,
                    bool             last_on_ties)
  {
    const std::size_t        n {a.rows()};
    numerical::buffer<T>     blocked(a.begin(), a.end());
    numerical::buffer<T>     stepped(a.begin(), a.end());
    const auto               by_panels {numerical::detail::bunch_kaufman_blocked_lower(blocked, n, hermitian, last_on_ties)};
    const auto               by_steps  {numerical::detail::bunch_kaufman_lower(stepped, n, hermitian, last_on_ties)};
    double                   largest   {1.0};
    double                   gap       {0.0};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j <= i; ++j) {
        const C x {blocked[(i * n) + j]};
        const C y {stepped[(i * n) + j]};
        largest = std::max(largest, std::abs(y));
        gap     = std::max(gap, std::abs(x - y));
      }
    }
    return by_panels == by_steps && gap <= 1e-12 * largest;
  }
} // namespace

// By panels: real, complex Hermitian and complex symmetric, the first (or last, an upper triangle's order) of equal
// entries, at orders whose last panel leaves 6 or 7 columns to the trailing product (by bands of rows), and 13 or more
// (a real one on the triangle's tiles): the unblocked steps' factorisation. A matrix of equal entries off a zero
// diagonal (every pivot of order 2, every search a tie), with zero columns inside the first panel and at its edge; an
// overflow that leaves a NaN on the diagonal (taken as a zero column) the same pivots. From order 384 ldl_decomposition takes
// the panels: A v = L D Lᵀ v for a vector v.
TEST(ldl_factorises_by_panels)
{
  std::uint64_t state {41};
  for (const std::size_t n : {std::size_t {70}, std::size_t {100}, std::size_t {141}}) {
    for (const bool last_on_ties : {false, true}) {
      const matrix<C> real {symmetric(n, false, true, state)};
      for (const bool hermitian : {true, false}) {
        EXPECT(panels_agree(real_of(real), hermitian, last_on_ties));
        EXPECT(panels_agree(symmetric(n, true, hermitian, state), hermitian, last_on_ties));
      }
    }
  }
  const std::size_t   n {100};
  std::vector<double> ties(n * n, 1.0);
  for (std::size_t i {0}; i < n; ++i) {
    ties[(i * n) + i] = 0.0;
  }
  std::vector<double> gaps {ties};
  for (const std::size_t zero : {std::size_t {5}, std::size_t {62}, std::size_t {63}}) {
    for (std::size_t i {0}; i < n; ++i) {
      gaps[(i * n) + zero] = 0.0;
      gaps[(zero * n) + i] = 0.0;
    }
  }
  std::vector<double> big(n * n, 0.0); // ldl_takes_an_overflow_as_sytf2_does's matrix, then the identity
  const double        b         {1.5e308};
  const double        corner[9] {b, b, -b, b, -b, b, -b, b, b};
  for (std::size_t i {0}; i < n; ++i) {
    big[(i * n) + i] = 1.0;
  }
  for (std::size_t i {0}; i < 3; ++i) {
    for (std::size_t j {0}; j < 3; ++j) {
      big[(i * n) + j] = corner[(i * 3) + j];
    }
  }
  for (const bool last_on_ties : {false, true}) {
    for (const std::vector<double>* entries : {&ties, &gaps}) {
      EXPECT(panels_agree(matrix<double> {n, n, *entries}, true, last_on_ties));
    }
    // the panels keep the NaN column in W, as sytrf does, and its later updates carry the NaN on: the same pivots
    numerical::buffer<double> blocked(big.begin(), big.end());
    numerical::buffer<double> stepped(big.begin(), big.end());
    EXPECT(numerical::detail::bunch_kaufman_blocked_lower(blocked, n, true, last_on_ties)
           == numerical::detail::bunch_kaufman_lower(stepped, n, true, last_on_ties));
    EXPECT(std::isnan(blocked[(2 * n) + 2]) && std::isnan(stepped[(2 * n) + 2]));
  }
  const std::size_t                          order {384};
  const matrix<double>                       a     {real_of(symmetric(order, false, true, state))};
  const numerical::ldl_decomposition<double> f     {a};
  std::vector<double>                        v(order);
  for (double& x : v) {
    x = drawn(state);
  }
  const matrix<double> column {order, 1, v};
  const matrix<double> lt     {numerical::transpose(f.factor())};
  const matrix<double> gap    {(a * column) - (f.factor() * (f.d() * (lt * column)))};
  double               worst  {0.0};
  for (const double x : gap) {
    worst = std::max(worst, std::abs(x));
  }
  EXPECT(worst <= 1e-12 * static_cast<double>(order));
}

// SciPy's answers: its documentation's example from the upper triangle, and a matrix of equal entries off a zero
// diagonal (every pivot search a tie: the first of equal entries below the diagonal, the last above).
TEST(ldl_matches_scipy)
{
  const numerical::ldl_decomposition<double> example {matrix<double> {{2.0, -1.0, 3.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 1.0}}, false};
  EXPECT((example.factor() == matrix<double> {{0.0, 0.0, 1.0}, {0.0, 1.0, -0.5}, {1.0, 1.0, 1.5}}));
  EXPECT((example.d() == matrix<double> {{-5.0, 0.0, 0.0}, {0.0, 1.5, 0.0}, {0.0, 0.0, 2.0}}));
  EXPECT((example.permutation() == std::vector<std::size_t> {2, 1, 0}));
  const matrix<double>                       ties  {{0.0, 1.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}};
  const numerical::ldl_decomposition<double> below {ties};
  EXPECT((below.factor() == matrix<double> {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 1.0}}));
  EXPECT((below.d() == matrix<double> {{0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, -2.0}}));
  EXPECT((below.permutation() == std::vector<std::size_t> {0, 1, 2}));
  const numerical::ldl_decomposition<double> above {ties, false};
  EXPECT((above.factor() == matrix<double> {{0.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, {0.0, 0.0, 1.0}}));
  EXPECT((above.d() == matrix<double> {{-2.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}}));
  EXPECT((above.permutation() == std::vector<std::size_t> {1, 0, 2}));
}

// A zero column needs no elimination (D's entry 0, L's column the identity's), first, in the middle and last; the zero
// matrix's D zero; a Hermitian matrix's diagonal imaginary parts ignored (its factorisation the real diagonal's).
TEST(ldl_takes_zero_columns_and_a_hermitian_diagonal)
{
  for (const std::size_t zero : {std::size_t {0}, std::size_t {2}, std::size_t {4}}) {
    std::uint64_t       state {zero + 11};
    const matrix<C>     full  {symmetric(5, false, true, state)};
    std::vector<double> entries(25);
    for (std::size_t i {0}; i < 5; ++i) {
      for (std::size_t j {0}; j < 5; ++j) {
        entries[(i * 5) + j] = (i == zero || j == zero) ? 0.0 : full(i, j).real();
      }
    }
    const matrix<double> a {5, 5, entries};
    for (const bool lower : {true, false}) {
      const numerical::ldl_decomposition<double> f {a, lower};
      EXPECT(factorises(f, a, lower, true, 1e-14));
    }
  }
  const matrix<double> zeros {3, 3, std::vector<double>(9, 0.0)};
  EXPECT((numerical::ldl_decomposition<double> {zeros}.d() == zeros));
  std::uint64_t   state      {21};
  const matrix<C> hermitian  {symmetric(6, true, true, state)};
  std::vector<C>  tilted(hermitian.begin(), hermitian.end());
  for (std::size_t i {0}; i < 6; ++i) {
    tilted[(i * 6) + i] += C {0.0, 1.0 + static_cast<double>(i)};
  }
  for (const bool lower : {true, false}) {
    const numerical::ldl_decomposition<C> f {matrix<C> {6, 6, tilted}, lower};
    const numerical::ldl_decomposition<C> g {hermitian, lower};
    EXPECT((f.factor() == g.factor()) && (f.d() == g.d()) && (f.permutation() == g.permutation()));
  }
}

// Finite entries whose elimination overflows: the last diagonal entry becomes NaN (∞ − ∞), and sytf2 takes it as it
// takes a zero column, with no pivot search.
TEST(ldl_takes_an_overflow_as_sytf2_does)
{
  const double                               big {1.5e308};
  const numerical::ldl_decomposition<double> f   {matrix<double> {{big, big, -big}, {big, -big, big}, {-big, big, big}}};
  EXPECT(std::isnan(f.d()(2, 2)));
         EXPECT((f.permutation() == std::vector<std::size_t> {0, 1, 2}));
}

// The refusals.
TEST(ldl_refuses_what_it_cannot_take)
{
  EXPECT_THROWS((numerical::ldl_decomposition<double> {matrix<double> {{1.0, 2.0}}}), numerical::dimension_error);
  EXPECT_THROWS((numerical::ldl_decomposition<C> {matrix<C> {{C {std::numeric_limits<double>::quiet_NaN()}}}}), numerical::non_finite_error);
}
