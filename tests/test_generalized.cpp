// The generalized Schur decomposition (complex QZ) against its definition: A = Q S Zᴴ, B = Q T Zᴴ, Q and Z unitary, S
// and T triangular, T's diagonal real and non-negative; singular B (infinite eigenvalues, zeros of T chased down and
// up), a pencil that needs exceptional shifts, reordering and its refusals. The algebraic Riccati equations against
// their residuals, with and without E and S, real and complex, balanced or not; the case SciPy's balancing gets wrong
// with E; the refusals.

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::matrix;
  using C = std::complex<double>;

  //! \brief The largest modulus of \p m's entries, infinite if one is not finite.
  template <typename T>
  double largest(const matrix<T>& m)
  {
    double value {0.0};
    for (std::size_t i {0}; i < m.rows(); ++i) {
      for (std::size_t j {0}; j < m.cols(); ++j) {
        const double size {std::abs(m(i, j))};
        value = std::isfinite(size) ? std::max(value, size) : std::numeric_limits<double>::infinity();
      }
    }
    return value;
  }

  //! \brief The `n × n` identity.
  template <typename T>
  matrix<T> identity(std::size_t n)
  {
    std::vector<T> entries(n * n, T {});
    for (std::size_t i {0}; i < n; ++i) {
      entries[(i * n) + i] = T {1.0};
    }
    return matrix<T> {n, n, entries};
  }

  //! \brief An `rows × cols` matrix of `sin` (real part) and `cos` (imaginary part, if \p complex) of a quadratic in its
  //!        entries' index, offset by \p seed.
  matrix<C> trigonometric(std::size_t rows,
                          std::size_t cols,
                          double      seed,
                          bool        complex)
  {
    std::vector<C> entries(rows * cols);
    for (std::size_t e {0}; e < entries.size(); ++e) {
      const double k     {static_cast<double>(e)};
      const double angle {seed + (0.7 * k) + (0.013 * k * k)}; // a quadratic phase: of full rank, as a linear one is not
      entries[e] = C {std::sin(angle), complex ? std::cos(1.3 * angle) : 0.0};
    }
    return matrix<C> {rows, cols, entries};
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

  //! \brief Whether \p d holds a generalized Schur decomposition of \p a and \p b to \p tolerance (relative to their
  //!        norms): the products, the unitary factors, the triangles, `T`'s diagonal, the values the diagonals.
  bool decomposes(const numerical::generalized_schur& d,
                  const matrix<C>&                    a,
                  const matrix<C>&                    b,
                  double                              tolerance)
  {
    const std::size_t                                               n {a.rows()};
    const matrix<C>                                                 q {d.q()};
    const matrix<C>                                                 z {d.z()};
    const matrix<C>                                                 s {d.s()};
    const matrix<C>                                                 t {d.t()};
    const auto                                                      adjoint = [](const matrix<C>& m) { return numerical::detail::adjoint_of(m); };
    bool                                                            fine {largest((q * s * adjoint(z)) - a) <= tolerance * std::max(largest(a), 1.0)};
    fine = fine && largest((q * t * adjoint(z)) - b) <= tolerance * std::max(largest(b), 1.0);
    fine = fine && largest((adjoint(q) * q) - identity<C>(n)) <= tolerance;
    fine = fine && largest((adjoint(z) * z) - identity<C>(n)) <= tolerance;
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < i; ++j) {
        fine = fine && s(i, j) == C {} && t(i, j) == C {};
      }
      fine = fine && t(i, i).imag() == 0.0 && t(i, i).real() >= 0.0;
      fine = fine && d.values()[i].first == s(i, i) && d.values()[i].second == t(i, i);
    }
    return fine;
  }
} // namespace

// Pencils of orders 0 to 30, real and complex entries, B invertible: the decomposition to rounding.
TEST(generalized_schur_decomposes_a_regular_pencil)
{
  for (const std::size_t n : {std::size_t {0}, std::size_t {1}, std::size_t {2}, std::size_t {3}, std::size_t {8}, std::size_t {30}}) {
    for (const bool complex : {false, true}) {
      const matrix<C>                    a {trigonometric(n, n, 1.0, complex)};
      const matrix<C>                    b {trigonometric(n, n, 2.0, complex) + (C {3.0} *identity<C>(n))};
      const numerical::generalized_schur d {a, b};
      EXPECT(decomposes(d, a, b, 1e-13));
    }
  }
}

// B singular: infinite eigenvalues (β = 0), its zero diagonal entries met at the bottom of the active block, inside
// it, and in pairs; B zero altogether; A and B sharing a null vector (a singular pencil, every eigenvalue 0/0 there).
TEST(generalized_schur_takes_a_singular_b)
{
  const std::size_t n {6};
  for (const std::size_t rank : {std::size_t {0}, std::size_t {1}, std::size_t {3}, std::size_t {5}}) {
    std::vector<C>  entries(n * n, C {});
    const matrix<C> full             {trigonometric(n, n, 4.0, true)};
    for (std::size_t i {0}; i < n; ++i) {
      for (std::size_t j {0}; j < n; ++j) {
        for (std::size_t k {0}; k < rank; ++k) {
          entries[(i * n) + j] += full(i, k) * full(j, k);
        }
      }
    }
    const matrix<C>                    b {n, n, entries};
    const matrix<C>                    a {trigonometric(n, n, 1.0, true)};
    const numerical::generalized_schur d {a, b};
    EXPECT(decomposes(d, a, b, 1e-12));
    std::size_t infinite                 {0};
    for (const auto& [alpha, beta] : d.values()) {
      infinite += std::abs(beta) <= (1e-12 * largest(b)) + 1e-300 ? 1 : 0;
    }
    EXPECT(infinite == n - rank);
  }
  // B upper triangular with zeros on its diagonal at the top, the middle and the bottom, A Hessenberg
  for (const std::vector<std::size_t>& zeros : {std::vector<std::size_t> {0}, std::vector<std::size_t> {2}, std::vector<std::size_t> {4},
                                                std::vector<std::size_t> {1, 2}, std::vector<std::size_t> {0, 4}}) {
    std::vector<C> bt(25, C {});
    std::vector<C> ah(25, C {});
    for (std::size_t i {0}; i < 5; ++i) {
      for (std::size_t j {0}; j < 5; ++j) {
        bt[(i * 5) + j] = j >= i ? C {1.0 + static_cast<double>(i + (2 * j)), 0.5} : C {};
        ah[(i * 5) + j] = j + 1 >= i ? C {std::sin(1.0 + static_cast<double>((5 * i) + j)), 0.25} : C {};
      }
    }
    for (const std::size_t z : zeros) {
      bt[(z * 5) + z] = C {};
    }
    const matrix<C> a {5, 5, ah};
    const matrix<C> b {5, 5, bt};
    EXPECT(decomposes(numerical::generalized_schur {a, b}, a, b, 1e-12));
  }
  // a singular pencil: A and B both zero in their last column
  std::vector<C> as(16);
  std::vector<C> bs(16);
  for (std::size_t i {0}; i < 4; ++i) {
    for (std::size_t j {0}; j < 4; ++j) {
      as[(i * 4) + j] = j == 3 ? C {} : C {std::cos(static_cast<double>(i + (3 * j))), 0.0};
      bs[(i * 4) + j] = j == 3 ? C {} : C {i == j ? 2.0 : 0.1, 0.0};
    }
  }
  EXPECT(decomposes(numerical::generalized_schur {matrix<C> {4, 4, as}, matrix<C> {4, 4, bs}}, matrix<C> {4, 4, as}, matrix<C> {4, 4, bs}, 1e-13));
}

// The cyclic shift against the identity: eigenvalues the n-th roots of unity, all of one modulus, on which the standard
// shifts stall: the exceptional shifts (every tenth iteration) break the symmetry. A bound of one iteration refuses.
TEST(generalized_schur_needs_exceptional_shifts)
{
  for (const std::size_t n : {std::size_t {2}, std::size_t {4}, std::size_t {7}}) {
    std::vector<C> shift(n * n, C {});
    for (std::size_t i {0}; i < n; ++i) {
      shift[(((i + 1) % n) * n) + i] = C {1.0};
    }
    const matrix<C>                    a {n, n, shift};
    const matrix<C>                    b {identity<C>(n)};
    const numerical::generalized_schur d {a, b};
    EXPECT(decomposes(d, a, b, 1e-13));
    for (const auto& [alpha, beta] : d.values()) {
      EXPECT(std::abs(std::abs(alpha / beta) - 1.0) < 1e-13);
    }
    EXPECT_THROWS(static_cast<void>(numerical::generalized_schur(a, b, 1)), numerical::convergence_error);
  }
}

// Reordering as ordqz: the chosen eigenvalues first, in their order, the decomposition kept; sort by a predicate (the
// left half-plane, inside the unit circle); nothing or everything chosen; one choice per eigenvalue required; two equal
// eigenvalues exchanged.
TEST(generalized_schur_reorders)
{
  const std::size_t            n {8};
  const matrix<C>              a {trigonometric(n, n, 1.0, true)};
  const matrix<C>              b {trigonometric(n, n, 2.0, true) + (C {3.0} *identity<C>(n))};
  numerical::generalized_schur d {a, b};
  std::vector<C>               before;
  for (const auto& [alpha, beta] : d.values()) {
    before.push_back(alpha / beta);
  }
  const std::size_t chosen {d.sort([](const C& alpha, const C& beta) { return (alpha * std::conj(beta)).real() < 0.0; })};
  EXPECT(decomposes(d, a, b, 1e-12));
  std::size_t expected     {0};
  for (const C& lambda : before) {
    expected += lambda.real() < 0.0 ? 1 : 0;
  }
  EXPECT(chosen == expected && chosen > 0 && chosen < n);
  for (std::size_t k {0}; k < n; ++k) {
    const C lambda {d.values()[k].first / d.values()[k].second};
    EXPECT((lambda.real() < 0.0) == (k < chosen));
  }
  // the last eigenvalue alone moved to the front: the others keep their order
  std::vector<C> was;
  for (const auto& [alpha, beta] : d.values()) {
    was.push_back(alpha / beta);
  }
  std::vector<bool> last(n, false);
  last[n - 1] = true;
  EXPECT(d.reorder(last) == 1);
  EXPECT(decomposes(d, a, b, 1e-12));
  EXPECT(std::abs((d.values()[0].first / d.values()[0].second) - was[n - 1]) < 1e-10);
  for (std::size_t k {1}; k < n; ++k) {
    EXPECT(std::abs((d.values()[k].first / d.values()[k].second) - was[k - 1]) < 1e-10);
  }
  EXPECT(d.sort([](const C&, const C&) { return false; }) == 0);
  EXPECT(d.sort([](const C&, const C&) { return true; }) == n);
  EXPECT(decomposes(d, a, b, 1e-12));
  EXPECT_THROWS(static_cast<void>(d.reorder(std::vector<bool>(n - 1, true))), numerical::dimension_error);
  // a double eigenvalue: its two copies exchanged
  const matrix<C>              twice {{C {2.0}, C {1.0}}, {C {}, C {2.0}}};
  numerical::generalized_schur e     {twice, identity<C>(2)};
  EXPECT(e.reorder(std::vector<bool> {false, true}) == 1);
  EXPECT(decomposes(e, twice, identity<C>(2), 1e-14));
}

// An exchange whose rotations do not keep the blocks apart is refused (LAPACK's ztgex2 tests, NaN failing them), the
// decomposition left as it was: entries of scales 10²⁰⁰ apart (found by a search), refused at any common scale.
TEST(generalized_schur_refuses_an_unstable_exchange)
{
  const matrix<C>              a {{C {4.0433738246568382e-290}, C {-4.1728728744498313e-195}}, {C {}, C {-7.7732712242752404e-158}}};
  const matrix<C>              b {{C {-1.3394732758970938e-222}, C {-2.3968178730587086e-295}}, {C {}, C {-7.9217174196872459e-184}}};
  numerical::generalized_schur d {a, b};
  const matrix<C>              s {d.s()};
  const matrix<C>              z {d.z()};
  EXPECT_THROWS(static_cast<void>(d.reorder(std::vector<bool> {false, true})), numerical::convergence_error);
  EXPECT((d.s() == s) && (d.z() == z));
  EXPECT(decomposes(d, a, b, 1e-15));
}

// A pencil whose entries lie past the square root of the range, either way: brought into it as zgges does, S, T and
// the eigenvalues taken back; the products within rounding of A and B themselves (unscaled, an overflow of the norms'
// squares had left B's 45 % off).
TEST(generalized_schur_scales_as_zgges)
{
  for (const double scale : {1e-300, 3e-308, 1e300, 1e307}) {
    for (const bool on_b : {false, true}) {
      const matrix<C>                    a {C {on_b ? 1.0 : scale} *trigonometric(9, 9, 1.0, true)};
      const matrix<C>                    b {C {on_b ? scale : 1.0} *(trigonometric(9, 9, 2.0, true) + (C {3.0} *identity<C>(9)))};
      const numerical::generalized_schur d {a, b};
      const auto                         adjoint = [](const matrix<C>& m) { return numerical::detail::adjoint_of(m); };
      EXPECT(largest((d.q() * d.s() * adjoint(d.z())) - a) <= 1e-13 * largest(a));
      EXPECT(largest((d.q() * d.t() * adjoint(d.z())) - b) <= 1e-13 * largest(b));
      for (std::size_t k {0}; k < 9; ++k) {
        EXPECT(d.values()[k].first == d.s()(k, k) && d.values()[k].second == d.t()(k, k));
      }
    }
  }
}

// The deflations of zhgeqz: a zero of T at the top chased down to the bottom, T's corner cancelled on the way (A's
// first column at 45° against T's equal last column); two zeros of T in a row; a zero of T under two small
// subdiagonal entries of A (their product below the tolerance) chased down; a pencil whose last eigenvalue takes
// twenty iterations (the second exceptional shift).
TEST(generalized_schur_deflates_as_zhgeqz)
{
  const matrix<C> cancelled_a      {{C {1.0}, C {2.0}}, {C {1.0}, C {3.0}}};
  const matrix<C> cancelled_b      {{C {}, C {1.0}}, {C {}, C {1.0}}};
  EXPECT(decomposes(numerical::generalized_schur {cancelled_a, cancelled_b}, cancelled_a, cancelled_b, 1e-15));
  const matrix<C> twice_a          {{C {1.0}, C {2.0}, C {3.0}}, {C {1.0}, C {3.0}, C {1.0}}, {C {}, C {2.0}, C {5.0}}};
  const matrix<C> twice_b          {{C {}, C {}, C {1.0}}, {C {}, C {}, C {2.0}}, {C {}, C {}, C {1.0}}};
  EXPECT(decomposes(numerical::generalized_schur {twice_a, twice_b}, twice_a, twice_b, 1e-15));
  const matrix<C> small_a          {{C {1.0}, C {0.5}, C {0.3}}, {C {1e-9}, C {2.0}, C {0.7}}, {C {}, C {1e-9}, C {3.0}}};
  const matrix<C> small_b          {{C {1.0}, C {0.2}, C {0.1}}, {C {}, C {}, C {0.4}}, {C {}, C {}, C {1.0}}};
  EXPECT(decomposes(numerical::generalized_schur {small_a, small_b}, small_a, small_b, 1e-15));
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
  const matrix<C> a {7, 7, std::vector<C>(slow_a.begin(), slow_a.end())};
  const matrix<C> b {7, 7, std::vector<C>(slow_b.begin(), slow_b.end())};
  EXPECT(decomposes(numerical::generalized_schur {a, b}, a, b, 1e-13));
}

// ztgex2's tests on blocks of widely different scales (found by searches): S's new subdiagonal small but not the
// rotations' rebuilding of it (the strong test), the same of T, S's new subdiagonal not small, and T's not where S's is
// (the weak test); the pencil left as it was.
TEST(generalized_schur_exchange_refused_by_the_strong_test)
{
  const std::vector<std::vector<C>> blocks {
    {C {-1.5184070081802398e-156}, C {6.3620712533652165e-60}, C {5.0957381973536265e-215}, C {-3.7285531778325993e+87},
     C {4.491083877498144e-232, 2.0144018663795549e-232}, C {1.3722051491558582e-252, -5.8124886344425557e-253}},
    {C {1.4168451659752015e+250}, C {-5.313705309083561e-236, -2.5441570759711414e-235}, C {4.0512040151661126e-161},
     C {7.5064066864147823e+281}, C {-3.4550193118442166e-183}, C {6.4046429987321238e-89}},
    {C {7.7839978399627143e-294, 9.6449676117005718e-294}, C {3.990189972275741e-300}, C {-3.1390256356239303e-287},
     C {2.7553991424053377e-280, 3.0573459930565973e-280}, C {5.2767074943069092e-59, -1.8613206037500929e-58},
     C {-5.6155289535791438e-18, -2.1634029033105651e-17}},
    {C {2.4031823726337465e-135}, C {4.9206202281730338e-98, -2.9828187753186243e-98}, C {7.7043445286263324e-64},
     C {7.292612240243964e-259}, C {1.1830952176537991e-251}, C {6.3618424291582671e-269}},
    // S's new subdiagonal not small (the weak test on S)
    {C {-2.1070735516868428e-73}, C {-6.2852733730141101e+150}, C {9.4030832948394708e+207}, C {1.6521185955162518e+196},
     C {-1.7831359330900348e+218}, C {2.150972062082325e+156}},
    // S's new subdiagonal small, T's not (the weak test on T alone)
    {C {7.6312369938252286e-134}, C {-4.6471336851529855e-128}, C {-3.0993675071219111e-103}, C {3.0857879948018701e-221},
     C {-6.9181806678442579e-209}, C {6.8993726162757166e-274}}};
  for (const std::vector<C>& c : blocks) {
    numerical::detail::qz_form f;
    f.n  = 2;
    f.s  = numerical::buffer<C> {c[0], c[1], C {}, c[2]};
    f.t  = numerical::buffer<C> {c[3], c[4], C {}, c[5]};
    f.qt = numerical::buffer<C> {C {1.0}, C {}, C {}, C {1.0}};
    f.zt = f.qt;
    const numerical::buffer<C> s {f.s};
    const numerical::buffer<C> t {f.t};
    EXPECT(!numerical::detail::exchange_pencil_diagonal(f, 1));
    EXPECT(f.s == s && f.t == t);
  }
}

// The refusals of the decomposition.
TEST(generalized_schur_refuses_what_it_cannot_take)
{
  EXPECT_THROWS(static_cast<void>(numerical::generalized_schur(matrix<C> {{C {1.0}, C {2.0}}}, identity<C>(1))), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::generalized_schur(identity<C>(2), identity<C>(1))), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::generalized_schur(identity<C>(1), matrix<C> {{C {1.0}}, {C {2.0}}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::generalized_schur(identity<C>(1), matrix<C> {{C {1.0}, C {2.0}}})), numerical::dimension_error);
  EXPECT_THROWS(static_cast<void>(numerical::generalized_schur(matrix<C> {{C {std::numeric_limits<double>::quiet_NaN()}}}, identity<C>(1))),
                numerical::non_finite_error);
  EXPECT_THROWS(static_cast<void>(numerical::generalized_schur(identity<C>(1), matrix<C> {{C {std::numeric_limits<double>::infinity()}}})),
                numerical::non_finite_error);
}

namespace {

  //! \brief The operands of one Riccati equation.
  template <typename T>
  struct riccati_case
  {
    matrix<T> a;
    matrix<T> b;
    matrix<T> q;
    matrix<T> r;
    matrix<T> e;
    matrix<T> s;
  };

  //! \brief An equation of order \p m with \p n inputs, `Q` and `R` positive definite, `E` well away from singular.
  riccati_case<C> riccati_operands(std::size_t m,
                                   std::size_t n,
                                   bool        complex)
  {
    const auto      adjoint = [](const matrix<C>& x) { return numerical::detail::adjoint_of(x); };
    const matrix<C> q0 {trigonometric(m, m, 3.0, complex)};
    const matrix<C> r0 {trigonometric(n, n, 4.0, complex)};
    return {.a = trigonometric(m, m, 1.0, complex),
            .b = trigonometric(m, n, 2.0, complex),
            .q = (adjoint(q0) * q0) + identity<C>(m),
            .r = (adjoint(r0) * r0) + identity<C>(n),
            .e = trigonometric(m, m, 5.0, complex) + (C {static_cast<double>(m) + 1.0} *identity<C>(m)),
            .s = C {0.1} *trigonometric(m, n, 6.0, complex)};
  }

  //! \brief The continuous equation's residual `Aᴴ X E + Eᴴ X A − (Eᴴ X B + S) R⁻¹ (Bᴴ X E + Sᴴ) + Q` over its terms'
  //!        sizes, its stability: the closed loop `A − B K` (`K = R⁻¹ (Bᴴ X E + Sᴴ)`) of `E⁻¹` with its eigenvalues in
  //!        the left half-plane.
  double continuous_residual(const riccati_case<C>& c,
                             const matrix<C>&       x,
                             bool                   with_e,
                             bool                   with_s)
  {
    const auto      adjoint = [](const matrix<C>& m) { return numerical::detail::adjoint_of(m); };
    const matrix<C> e        {with_e ? c.e : identity<C>(c.a.rows())};
    const matrix<C> s        {with_s ? c.s : C {0.0} *c.s};
    const matrix<C> g        {(adjoint(e) * x * c.b) + s};
    const matrix<C> gain     {numerical::solve(c.r, adjoint(g))};
    const matrix<C> residual {(adjoint(c.a) * x * e) + (adjoint(e) * x * c.a) - (g * gain) + c.q};
    const matrix<C> loop     {numerical::solve(e, c.a - (c.b * gain))};
    for (const C& lambda : numerical::eigenvalues(loop)) {
      if (!(lambda.real() < 0.0)) {
        return std::numeric_limits<double>::infinity();
      }
    }
    return largest(residual) / ((2.0 * largest(c.a) * largest(x) * largest(e)) + largest(g * gain) + largest(c.q));
  }

  //! \brief The discrete equation's residual `Aᴴ X A − Eᴴ X E − (Aᴴ X B + S) (R + Bᴴ X B)⁻¹ (Bᴴ X A + Sᴴ) + Q` over its
  //!        terms' sizes, its stability: the closed loop of `E⁻¹` with its eigenvalues inside the unit circle.
  double discrete_residual(const riccati_case<C>& c,
                           const matrix<C>&       x,
                           bool                   with_e,
                           bool                   with_s)
  {
    const auto      adjoint = [](const matrix<C>& m) { return numerical::detail::adjoint_of(m); };
    const matrix<C> e        {with_e ? c.e : identity<C>(c.a.rows())};
    const matrix<C> s        {with_s ? c.s : C {0.0} *c.s};
    const matrix<C> g        {(adjoint(c.a) * x * c.b) + s};
    const matrix<C> gain     {numerical::solve(c.r + (adjoint(c.b) * x * c.b), adjoint(g))};
    const matrix<C> residual {(adjoint(c.a) * x * c.a) - (adjoint(e) * x * e) - (g * gain) + c.q};
    const matrix<C> loop     {numerical::solve(e, c.a - (c.b * gain))};
    for (const C& lambda : numerical::eigenvalues(loop)) {
      if (!(std::abs(lambda) < 1.0)) {
        return std::numeric_limits<double>::infinity();
      }
    }
    return largest(residual) / ((largest(c.a) * largest(c.a) * largest(x)) + (largest(e) * largest(e) * largest(x)) + largest(g * gain) + largest(c.q));
  }
} // namespace

// The stabilising solutions, real and complex, orders 1 to 20, E and S each present or not, balanced and not: a
// Hermitian X with the residual within 1e-11 of its terms (SciPy 1.18 reaches 3e-12 at order 20) and the closed loop
// stable; a real equation's solution real.
TEST(riccati_equations_are_solved)
{
  for (const auto& [m, n] : {std::pair<std::size_t, std::size_t> {1, 1}, {2, 1}, {5, 2}, {20, 4}}) {
    for (const bool complex : {false, true}) {
      riccati_case<C> c {riccati_operands(m, n, complex)};
      c.a = C {0.3} *c.a; // A inside the unit circle as E's scale has it, and the continuous equation's too
      for (int flags {0}; flags < 8; ++flags) {
        const bool       with_e   {(flags & 1) != 0};
        const bool       with_s   {(flags & 2) != 0};
        const bool       balanced {(flags & 4) == 0};
        const matrix<C>* e        {with_e ? &c.e : nullptr};
        const matrix<C>* s        {with_s ? &c.s : nullptr};
        if (complex) {
          const matrix<C> x {numerical::solve_continuous_are(c.a, c.b, c.q, c.r, e, s, balanced)};
          EXPECT(continuous_residual(c, x, with_e, with_s) < 1e-11);
          EXPECT(largest(x - numerical::detail::adjoint_of(x)) == 0.0);
          const matrix<C> y {numerical::solve_discrete_are(c.a, c.b, c.q, c.r, e, s, balanced)};
          EXPECT(discrete_residual(c, y, with_e, with_s) < 1e-11);
          EXPECT(largest(y - numerical::detail::adjoint_of(y)) == 0.0);
          continue;
        }
        const matrix<double> ar {real_of(c.a)};
        const matrix<double> br {real_of(c.b)};
        const matrix<double> qr {real_of(c.q)};
        const matrix<double> rr {real_of(c.r)};
        const matrix<double> er {real_of(c.e)};
        const matrix<double> sr {real_of(c.s)};
        const matrix<double> x  {numerical::solve_continuous_are(ar, br, qr, rr, with_e ? &er : nullptr, with_s ? &sr : nullptr, balanced)};
        EXPECT(continuous_residual(c, numerical::detail::as_complex(x), with_e, with_s) < 1e-11);
        EXPECT((x == numerical::transpose(x)));
        const matrix<double> y {numerical::solve_discrete_are(ar, br, qr, rr, with_e ? &er : nullptr, with_s ? &sr : nullptr, balanced)};
        EXPECT(discrete_residual(c, numerical::detail::as_complex(y), with_e, with_s) < 1e-11);
        EXPECT((y == numerical::transpose(y)));
      }
    }
  }
}

// E scaled by the balancing: the basis of the balanced pencil goes with D₁ E D₁⁻¹, not E (SciPy 1.18 answers this
// equation with residual 3 continuous and 2 discrete, scipy/scipy#26555).
TEST(riccati_balancing_scales_e)
{
  riccati_case<C> c {.a = matrix<C> {{C {3.0}, C {-3.0}}, {C {-1.0}, C {-1.0}}},
                     .b = matrix<C> {{C {3.0}}, {C {-2.0}}},
                     .q = identity<C>(2),
                     .r = identity<C>(1),
                     .e = matrix<C> {{C {1.0}, C {-2.0}}, {C {}, C {100.0}}},
                     .s = matrix<C> {{C {}}, {C {}}}};
  const matrix<double> a {real_of(c.a)};
  const matrix<double> b {real_of(c.b)};
  const matrix<double> e {real_of(c.e)};
  const matrix<double> x {numerical::solve_continuous_are(a, b, identity<double>(2), identity<double>(1), &e)};
  EXPECT(continuous_residual(c, numerical::detail::as_complex(x), true, false) < 1e-14);
  const matrix<double> y {numerical::solve_discrete_are(a, b, identity<double>(2), identity<double>(1), &e)};
  EXPECT(discrete_residual(c, numerical::detail::as_complex(y), true, false) < 1e-14);
}

// A graded A (eigenvalues near −2.6·10⁸ and −6.2·10⁸, X near 10⁻⁹): the generalized Schur form's answer has a
// residual of 1.3·10⁻⁷ of its terms (SciPy 1.18's 9.8·10⁻⁹, its real QZ's), one step of Newton's method brings it to
// rounding. An exact answer (Q = 0, X = 0) is kept: the step does not lower a zero residual. An X near 2·10¹⁵⁴ (A = 10¹⁵⁴
// unstable) is kept as the Schur method gives it: its residual's terms overflow, and no step is taken.
TEST(riccati_continuous_takes_a_step_of_newton)
{
  riccati_case<C> c {.a = matrix<C> {{C {-2.5723633516085407e+08}, C {4.4375165314791058e-04}},
                       {C {1.2072959802483012e+03}, C {-6.2102344766426814e+08}}},
                     .b = matrix<C> {{C {0.44663486401116476}}, {C {0.285525462894814}}},
                     .q = matrix<C> {{C {2.998057577496829}, C {0.5524690732831364}}, {C {0.5524690732831364}, C {3.127499944446211}}},
                     .r = matrix<C> {{C {1.0528016657039019}}},
                     .e = identity<C>(2),
                     .s = matrix<C> {{C {}}, {C {}}}};
  const matrix<double> x {numerical::solve_continuous_are(real_of(c.a), real_of(c.b), real_of(c.q), real_of(c.r))};
  EXPECT(continuous_residual(c, numerical::detail::as_complex(x), false, false) < 4.0 * std::numeric_limits<double>::epsilon());
  using M = matrix<double>;
  const M zero {numerical::solve_continuous_are(M {{-1.0, 0.5}, {0.0, -2.0}}, M {{1.0}, {1.0}}, M {{0.0, 0.0}, {0.0, 0.0}}, M {{1.0}})};
  EXPECT((zero == M {{0.0, 0.0}, {0.0, 0.0}}));
  const M huge {numerical::solve_continuous_are(M {{1e154}}, M {{1.0}}, M {{1.0}}, M {{1.0}})};
  EXPECT(std::abs((huge(0, 0) / 2e154) - 1.0) < 1e-15);
  // an R of condition 10¹⁰, past 1/√ε: the extended pencil, not the Hamiltonian (whose G = B R⁻¹ Bᴴ would lose it)
  riccati_case<C> stiff {.a = matrix<C> {{C {-1.0}, C {0.5}}, {C {0.0}, C {-2.0}}},
                         .b = matrix<C> {{C {1.0}, C {0.0}}, {C {0.0}, C {1.0}}},
                         .q = matrix<C> {{C {1.0}, C {0.0}}, {C {0.0}, C {1.0}}},
                         .r = matrix<C> {{C {1.0}, C {0.0}}, {C {0.0}, C {1e-10}}},
                         .e = identity<C>(2),
                         .s = matrix<C> {{C {}, C {}}, {C {}, C {}}}};
  const matrix<double> y {numerical::solve_continuous_are(real_of(stiff.a), real_of(stiff.b), real_of(stiff.q), real_of(stiff.r))};
  EXPECT(continuous_residual(stiff, numerical::detail::as_complex(y), false, false) < 1e-14);
}

// No stabilising solution: an unstable mode B cannot reach (U₀₀ singular, exactly where the mode is decoupled), and a negative Q putting the pencil's two
// eigenvalues on the imaginary axis (±i√2) or the unit circle (a conjugate pair): a real pencil's pair moves whole, so
// that the sort takes 0 or 2 eigenvalues, not 1; a complex one's takes 1, whose eigenvector is complex and U₀₀ᴴ U₁₀ not
// Hermitian (SciPy 1.18's discrete solver, in real arithmetic, answers 1 + √2 there for every Q < 0: a 1 × 1 real
// product is always symmetric).
TEST(riccati_equations_without_a_stabilising_solution_are_refused)
{
  using M = matrix<double>;
  // an unstable mode decoupled from the rest (B and Q zero there): the pencil diagonal there, U₀₀ exactly singular
  EXPECT_THROWS(static_cast<void>(numerical::solve_continuous_are(M {{1.0}}, M {{0.0}}, M {{0.0}}, M {{1.0}})), numerical::singular_matrix_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_discrete_are(M {{2.0}}, M {{0.0}}, M {{0.0}}, M {{1.0}})), numerical::singular_matrix_error);
  const matrix<C> unit_c {{C {1.0}}};
  EXPECT_THROWS(static_cast<void>(numerical::solve_discrete_are(matrix<C> {{C {2.0}}}, matrix<C> {{C {}}}, matrix<C> {{C {}}}, unit_c)),
                numerical::singular_matrix_error); // complex: U₀₀ singular there too
  const M other {{0.0, 0.0}, {0.0, 1.0}};
  EXPECT_THROWS(static_cast<void>(numerical::solve_continuous_are(M {{1.0, 0.0}, {0.0, -1.0}}, M {{0.0}, {1.0}}, other, M {{1.0}})),
                numerical::singular_matrix_error);
  EXPECT_THROWS(static_cast<void>(numerical::solve_discrete_are(M {{2.0, 0.0}, {0.0, 0.5}}, M {{0.0}, {1.0}}, other, M {{1.0}})),
                numerical::singular_matrix_error);
  for (const double q : {-0.5, -2.0}) {
    const M unit {{1.0}};
    EXPECT_THROWS(static_cast<void>(numerical::solve_continuous_are(M {{0.0}}, M {{1.0}}, M {{q}}, M {{1.0}}, &unit)), numerical::convergence_error); // the pencil
    EXPECT_THROWS(static_cast<void>(numerical::solve_continuous_are(M {{0.0}}, M {{1.0}}, M {{q}}, M {{1.0}})), numerical::convergence_error);
    EXPECT_THROWS(static_cast<void>(numerical::solve_discrete_are(M {{1.0}}, M {{1.0}}, M {{q}}, M {{1.0}})), numerical::convergence_error);
    // complex: one eigenvalue of the pair taken, its eigenvector's halves failing the symmetry test
    const matrix<C> one {{C {1.0}}};
    EXPECT_THROWS(static_cast<void>(numerical::solve_continuous_are(matrix<C> {{C {}}}, one, matrix<C> {{C {q}}}, one, &one)), numerical::convergence_error);
    EXPECT_THROWS(static_cast<void>(numerical::solve_discrete_are(one, one, matrix<C> {{C {q}}}, one)), numerical::convergence_error);
  }
}

// The refusals of the operands: shapes (each operand's, and m or n zero), entries not finite, Q or R not Hermitian, E
// singular.
TEST(riccati_equations_refuse_what_they_cannot_take)
{
  using M = matrix<double>;
  const M one    {{1.0}};
  const M two    {identity<double>(2)};
  const M column {{1.0}, {1.0}};
  const M empty  {0, 0, std::vector<double> {}};
  for (const bool continuous : {true, false}) {
    const auto solve = [continuous](const M& a, const M& b, const M& q, const M& r, const M* e, const M* s) {
                         return continuous ? numerical::solve_continuous_are(a, b, q, r, e, s) : numerical::solve_discrete_are(a, b, q, r, e, s);
                       };
    EXPECT_THROWS(static_cast<void>(solve(empty, M {0, 1, std::vector<double> {}}, empty, one, nullptr, nullptr)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(one, M {1, 0, std::vector<double> {}}, one, empty, nullptr, nullptr)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(column, one, one, one, nullptr, nullptr)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(two, one, two, one, nullptr, nullptr)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(two, column, one, one, nullptr, nullptr)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(two, column, M {{1.0, 0.0}}, one, nullptr, nullptr)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(two, column, column, one, nullptr, nullptr)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(two, column, two, M {{1.0, 0.0}}, nullptr, nullptr)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(two, column, two, two, nullptr, nullptr)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(two, column, two, column, nullptr, nullptr)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(two, column, two, one, &one, nullptr)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(two, column, two, one, &column, nullptr)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(two, column, two, one, nullptr, &two)), numerical::dimension_error);
    EXPECT_THROWS(static_cast<void>(solve(two, column, two, one, nullptr, &one)), numerical::dimension_error);
    const M nan {{std::numeric_limits<double>::quiet_NaN()}};
    EXPECT_THROWS(static_cast<void>(solve(nan, one, one, one, nullptr, nullptr)), numerical::non_finite_error);
    EXPECT_THROWS(static_cast<void>(solve(one, one, one, one, &nan, nullptr)), numerical::non_finite_error);
    EXPECT_THROWS(static_cast<void>(solve(one, one, one, one, nullptr, &nan)), numerical::non_finite_error);
    EXPECT_THROWS(static_cast<void>(solve(two, column, M {{1.0, 1.0}, {0.0, 1.0}}, one, nullptr, nullptr)), std::invalid_argument);
    EXPECT_THROWS(static_cast<void>(solve(one, M {{1.0, 1.0}}, one, M {{1.0, 2.0}, {0.0, 1.0}}, nullptr, nullptr)),
                  std::invalid_argument);
    const M singular {{1.0, 2.0}, {2.0, 4.0}};
    EXPECT_THROWS(static_cast<void>(solve(two, column, two, one, &singular, nullptr)), numerical::singular_matrix_error);
  }
  const matrix<C> hermitian  {{C {1.0}, C {0.0, 1.0}}, {C {0.0, -1.0}, C {2.0}}};
  const matrix<C> transposed {{C {1.0}, C {0.0, 1.0}}, {C {0.0, 1.0}, C {2.0}}}; // symmetric, not Hermitian
  const matrix<C> inputs     {{C {1.0}}, {C {0.5, 0.5}}};
  EXPECT(numerical::solve_continuous_are(C {-1.0} *identity<C>(2), inputs, hermitian, identity<C>(1)).rows() == 2);
  EXPECT_THROWS(static_cast<void>(numerical::solve_continuous_are(C {-1.0} *identity<C>(2), inputs, transposed, identity<C>(1))), std::invalid_argument);
  // the continuous equation's R⁻¹: a numerically singular r refused as SciPy refuses it, a zero one too
  EXPECT_THROWS(static_cast<void>(numerical::solve_continuous_are(-1.0 * two, two, two, M {{1.0, 2.0}, {2.0, 4.0}})), std::invalid_argument);
  EXPECT_THROWS(static_cast<void>(numerical::solve_continuous_are(-1.0 * two, two, two, M {{0.0, 0.0}, {0.0, 0.0}})), std::invalid_argument);
}

// The scaling of dgebal('S') at the edges of the range: a zero row or column left alone; scales stopped where the
// row's or the column's norm would leave the safe range, from either side; accumulated scales refused past it (a
// matrix needing more than one sweep). Each scale a power of two, the scaled matrix D⁻¹ M D finite.
TEST(scaling_balance_keeps_to_the_safe_range)
{
  const std::vector<std::vector<double>> cases {
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 1e300, 1e-300, 0.0},
    {0.0, 1e-290, 1e-300, 0.0},
    {0.0, 1e290, 1e300, 0.0},
    {0.0, 1e-300, 1e-290, 0.0},
    {0.0, 2.6084675210142215e-59, 1.3033222867197861e-130, 0.0, 1.8716888205735587e-36, 0.0, 0.0, 6.1575455797815257e+26, 0.0,
     6.5783819696480919e-117, 0.0, 0.0, 6.0639328053507447e-68, 2.3372294121524485e-288, 2.6928191476905506e+103, 0.0},
    {0.0, 6.5356851434793315e+76, 9.601987026629549e-171, 0.0, 1.3333566355247427e+42, 0.0, 0.0, 0.0, 4.1104916880713947e+273,
     1.0876119885754736e-63, 0.0, 0.0, 4.5118383718757068e+31, 1.1482732101160237e+109, 3.6058230797178956e+262, 0.0},
    {0.0, 0.0, 3.5463108278188681e-230, 6.080431252282603e-204, 0.0, 0.0, 5.2611905161421423e+111, 2.67928047914497e+290, 0.0}};
  for (const std::vector<double>& entries : cases) {
    const std::size_t               n     {static_cast<std::size_t>(std::lround(std::sqrt(static_cast<double>(entries.size()))))};
    numerical::buffer<double>       m(entries.begin(), entries.end());
    const numerical::buffer<double> scale {numerical::detail::scaling_balance(m, n)};
    for (const double d : scale) {
      int exponent {0};
      EXPECT(std::isfinite(d) && d > 0.0 && std::frexp(d, &exponent) == 0.5);
    }
    for (const double v : m) {
      EXPECT(std::isfinite(v));
    }
  }
}
