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
    double gap  {0.0};
    double size {1.0};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        C sum {};
        for (std::size_t k {0}; k < n; ++k) {
          for (std::size_t m {0}; m < n; ++m) {
            sum += l(i, k) * d(k, m) * transposed_entry(l(j, m), hermitian);
          }
        }
        gap  = std::max(gap, std::abs(sum - C {as_complex(a)(i, j)}));
        size = std::max(size, std::abs(C {as_complex(a)(i, j)}));
      }
    }
    return fine && gap <= tolerance * size;
  }
} // namespace

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
