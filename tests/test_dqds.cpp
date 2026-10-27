// The singular values of a bidiagonal by dqds: each to high relative accuracy (graded bidiagonals against values to
// 100 digits), the determinant and the Frobenius norm they keep, QR's values to its absolute accuracy, on bidiagonals
// drawn to reach each of the iteration's cases (its shifts, deflations, splits and flips, its transforms past
// overflow), and QR in its place past the bound of steps.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <sciforge/test/framework.hpp>
#include "numerical/numerical.hpp"

namespace {

  using numerical::buffer;
  using numerical::matrix;

  //! \brief A bidiagonal drawn from \p seed, the same on every platform: `mt19937_64`'s outputs and exact operations.
  struct drawn_bidiagonal
  {
    std::vector<double> d; //!< The diagonal.
    std::vector<double> e; //!< The superdiagonal, its last entry zero.
  };

  drawn_bidiagonal draw_bidiagonal(std::uint64_t seed)
  {
    std::mt19937_64  g {seed};
    const auto       unit = [&] { return static_cast<double>(g() >> 11U) * 0x1p-53; }; // [0, 1)
    const auto       pick = [&](std::uint64_t k) { return static_cast<int>(g() % k); };
    const int        kind {pick(17)};
    const int        n    {3 + pick(seed % 4 != 3 ? 5 : 30)};
    const int        lo   {500 + pick(575)};
    drawn_bidiagonal b {.d = std::vector<double>(static_cast<std::size_t>(n)), .e = std::vector<double>(static_cast<std::size_t>(n), 0.0)};
    const auto       entry = [&](int i, bool diagonal) {
                               double v {(2.0 * unit()) - 1.0};
                               switch (kind) {
                                 case 1: v  = std::ldexp(v, pick(600) - 300); break;                                                        // wildly graded
                                 case 2: v  = pick(3) == 0 ? 0.0 : v; break;                                                                // zeros
                                 case 3: v  = pick(2) != 0 ? 1.0 : v; break;                                                                // repeated
                                 case 4: v  = std::ldexp(1.0, -13 * (diagonal ? 2 * i : (2 * i) + 1)); break;                               // graded down
                                 case 5: v  = std::ldexp(1.0, -13 * (diagonal ? 2 * (n - 1 - i) : (2 * (n - 1 - i)) - 1)); break;           // up
                                 case 6: v  = 1.0 + std::ldexp(v, -50); break;                                                              // clustered
                                 case 7: v  = static_cast<double>(pick(3)); break;                                                          // small integers
                                 case 8: v  = std::ldexp(v, -pick(1070)); break;                                                            // down to the subnormals
                                 case 9: v  = diagonal ? 1.0 : std::ldexp(v, -27); break;                                                   // nearly diagonal
                                 case 10: v = pick(2) != 0 ? 0.5 + (0.5 * std::abs(v)) : std::ldexp(0.5 + (0.5 * std::abs(v)), -lo); break; // two scales far apart
                                 case 11: v = std::ldexp(pick(2) != 0 ? 1.0 : -1.0, pick(2098) - 1074); break;                              // any power of two
                                 case 12: v = std::ldexp(1.0, -i * pick(60)); break;
                                 case 13: v = pick(4) == 0 ? std::ldexp(1.0, -1000) : v; break;
                                 case 14: v = (pick(3) == 0 ? std::ldexp(1.0, -pick(1074)) : 1.0) * (pick(2) != 0 ? 1.0 : 1.0 + std::ldexp(v, -10)); break;
                                 case 15: v = std::ldexp(diagonal ? 1.0 : v, -pick(8) * 64); break;
                                 case 16: v = static_cast<double>(1 + pick(2)) * std::ldexp(1.0, -pick(3) * pick(400)); break;
                                 default: break;
                               }
                               return v;
                             };
    for (int i {0}; i < n; ++i) {
      b.d[static_cast<std::size_t>(i)] = entry(i, true);
    }
    for (int i {0}; i + 1 < n; ++i) {
      b.e[static_cast<std::size_t>(i)] = entry(i, false);
    }
    return b;
  }

  //! \brief Seeds whose bidiagonals reach the iteration's rarer cases: each estimate of the shift that gives up on
  //!        a ratio past one, a two-value deflation by either test, a split, the transforms whose numbers are not
  //!        finite (retried unshifted, then guarded), a pair whose half-sum underflows, a superdiagonal whose squares
  //!        all underflow.
  constexpr std::array<std::uint64_t, 25> rare_seeds {951277027U, 400000235U, 400010847U, 6127U, 1208655U, 60000173U, 120000081U, 120000009U, 60000091U,
                                                      7275U, 120000019U, 79463U, 120013568U, 60000579U, 120000875U,
                                                      8011U, 60000403U, 1655U, 60062823U, 10U, 120000006U,
                                                      120000047U, 120000012U, 121109667U, 118325055U};

  //! \brief The values of the bidiagonal (\p d, \p e), its largest entry \p largest, by the implicit-shift QR iteration
  //!        on it over that entry's power of two, as the decomposition scales a matrix (QR's tests of negligible
  //!        entries are absolute below `DBL_MIN / ε`).
  buffer<double> by_qr(const std::vector<double>& d,
                       const std::vector<double>& e,
                       double                     largest)
  {
    const std::size_t n     {d.size()};
    const int         power {std::ilogb(largest)};
    buffer<double>    qd(n);
    buffer<double>    qe(n);
    for (std::size_t i {0}; i < n; ++i) {
      qd[i] = std::ldexp(d[i], -power);
      qe[i] = std::ldexp(e[i], -power);
    }
    buffer<double> none;
    buffer<double> none_too;
    numerical::detail::bidiagonal_qr(qd, qe, none, none_too, n, 30 * n);
    for (double& value : qd) {
      value = std::ldexp(value, power);
    }
    return qd;
  }

  //! \brief The product of \p values' magnitudes as a fraction in [½, 1) and a power of two, \p power: exact but for
  //!        the fraction's roundings, however far the values range.
  double product_of_magnitudes(const auto& values,
                               int&        power)
  {
    double fraction {1.0};
    power = 0;
    for (const double value : values) {
      int scale {0};
      fraction *= std::frexp(std::abs(value), &scale);
      power    += scale;
      fraction  = std::frexp(fraction, &scale);
      power    += scale;
    }
    return fraction;
  }

  //! \brief Whether \p values are those of the bidiagonal (\p d, \p e): finite, non-negative and descending; their
  //!        squares summing to its Frobenius norm's; within `8nε` of the largest of QR's; their product |det| to
  //!        `4n²ε` where no diagonal entry is zero and no value below `2⁻⁴⁰⁰` of the largest entry (the squares of
  //!        the smallest stay normal at the scale dqds works at).
  bool are_its_values(const buffer<double>&      values,
                      const std::vector<double>& d,
                      const std::vector<double>& e)
  {
    const std::size_t n       {d.size()};
    const double      epsilon {std::numeric_limits<double>::epsilon()};
    double            largest {0.0};
    for (std::size_t i {0}; i < n; ++i) {
      largest = std::max({largest, std::abs(d[i]), std::abs(e[i])});
    }
    bool fine {true};
    for (std::size_t i {0}; i < n; ++i) {
      fine = fine && std::isfinite(values[i]) && values[i] >= 0.0 && (i == 0 || values[i] <= values[i - 1]);
    }
    if (!fine || largest == 0.0) {
      return fine;
    }
    double squares {0.0};
    double norm    {0.0};
    for (std::size_t i {0}; i < n; ++i) {
      squares += (values[i] / largest) * (values[i] / largest);
      norm    += ((d[i] / largest) * (d[i] / largest)) + ((e[i] / largest) * (e[i] / largest));
    }
    fine = std::abs(squares - norm) <= 4.0 * static_cast<double>(n) * epsilon * norm;
    const buffer<double> qr {by_qr(d, e, largest)};
    for (std::size_t i {0}; i < n; ++i) {
      fine = fine && std::abs(values[i] - qr[i]) <= (8.0 * static_cast<double>(n) * epsilon * qr[0]) + std::numeric_limits<double>::denorm_min();
    }
    bool nonsingular {true};
    for (std::size_t i {0}; i < n; ++i) {
      nonsingular = nonsingular && d[i] != 0.0 && values[i] >= std::ldexp(largest, -400);
    }
    if (!nonsingular) {
      return fine;
    }
    int          of_values   {0};
    int          of_diagonal {0};
    const double ratio       {product_of_magnitudes(std::vector<double>(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(n)), of_values)
                              / product_of_magnitudes(d, of_diagonal)};
    return fine && std::abs(std::ldexp(ratio, of_values - of_diagonal) - 1.0) <= 4.0 * static_cast<double>(n * n) * epsilon;
  }
} // namespace

// Graded bidiagonals, each value to 100 digits: every one, the smallest 10⁻²⁸ or 5·10⁻³³ of the largest, to a few
// units of its own last place, as LAPACK's dqds gives them (QR's iteration gave the smallest of the third to its
// absolute accuracy: none of its digits).
TEST(dqds_finds_each_value_of_a_graded_bidiagonal_to_relative_accuracy)
{
  constexpr std::size_t                      n    {8};
  const std::array<std::array<double, n>, 3> want {{
    {1.00003051711293262873, 1.22070312727359805066e-4, 1.49011611938493502125e-8, 1.81898940354585648845e-12,
     2.22044604925031308085e-16, 2.71050543121376108502e-20, 3.30872245021211069949e-24, 4.03884457333245464603e-28},
    {1.00000047683705162176, 1.220703125000555111e-4, 1.49011611938476562565e-8, 1.81898940354585647583e-12,
     2.22044604925031308085e-16, 2.71050543121376108502e-20, 3.30872245021211069949e-24, 4.03896590880114818019e-28},
    {2.94881639476674556563, 2.79793265193181340406, 2.55553002906644771184, 2.23606797749978969641, 1.86259664730172885413,
     1.47362575820790059316, 1.14213916400535577444, 5.33733434557999735037e-33},
  }};
  for (std::size_t c {0}; c < want.size(); ++c) {
    buffer<double> d(n, 0.0);
    buffer<double> e(n, 0.0);
    buffer<double> b(n * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      const int down {static_cast<int>(c == 0 ? i : n - 1 - i)};
      if (c == 2) { // diagonal 1 but the last, superdiagonal 2
        d[i] = i + 1 == n ? 0x1p-100 : 1.0;
        e[i] = i + 1 == n ? 0.0 : 2.0;
      }
      else {
        d[i] = std::ldexp(1.0, -13 * down);
        e[i] = i + 1 == n ? 0.0 : std::ldexp(1.0, (-13 * down) + (c == 0 ? -7 : 3));
      }
      b[(i * n) + i] = d[i];
      if (i + 1 < n) {
        b[(i * n) + i + 1] = e[i];
      }
    }
    const numerical::vector<double> through_a {numerical::singular_values(matrix<double> {n, n, std::move(b)})};
    EXPECT(numerical::detail::dqds_singular_values(d, e, n));
    for (std::size_t i {0}; i < n; ++i) {
      EXPECT(std::abs(d[i] - want[c][i]) <= 4.0 * std::numeric_limits<double>::epsilon() * want[c][i]);
      EXPECT(std::abs(through_a[i] - want[c][i]) <= 4.0 * std::numeric_limits<double>::epsilon() * want[c][i]);
    }
  }
}

// Bidiagonals drawn to reach each of the iteration's cases, then a sweep: the values are the bidiagonal's.
TEST(dqds_keeps_the_determinant_and_the_norm_and_agrees_with_qr)
{
  std::size_t wrong {0};
  const auto  check = [&](std::uint64_t seed) {
                        const drawn_bidiagonal b {draw_bidiagonal(seed)};
                        buffer<double>         d(b.d.begin(), b.d.end());
                        const buffer<double>   e(b.e.begin(), b.e.end());
                        EXPECT(numerical::detail::dqds_singular_values(d, e, b.d.size()));
                        wrong += are_its_values(d, b.d, b.e) ? 0U : 1U;
                      };
  for (const std::uint64_t seed : rare_seeds) {
    check(seed);
  }
  for (std::uint64_t seed {0}; seed < 4000; ++seed) {
    check(seed);
  }
  EXPECT(wrong == 0U);
}

// The 2 × 2 deflation leaves a block whose half-sum `((a − c) + b) / 2` underflows to zero as it is (its
// off-diagonal smallest of all, its two diagonal entries equal): the quotient by that half-sum would be infinite.
TEST(dqds_leaves_a_pair_whose_half_sum_underflows)
{
  double a {0x1p-1000};
  double c {0x1p-1000};
  numerical::detail::qd_pair(a, std::numeric_limits<double>::denorm_min(), c);
  EXPECT(a == 0x1p-1000 && c == 0x1p-1000);
  double larger  {1.0};
  double smaller {4.0};
  numerical::detail::qd_pair(larger, 1.0, smaller); // q₁ = 1, e₁ = 1, q₂ = 4: trace 6, determinant 4, ordered
  EXPECT(larger > smaller);
  EXPECT(std::abs((larger + smaller) - 6.0) < 1e-15 && std::abs((larger * smaller) - 4.0) < 1e-14);
}

// The guarded transform past a zero pivot (a block whose first q and e are zero, its first value zero): the pivot
// left at zero, the rest transformed on from the next q, the trace of the array kept, as the transform keeps it.
TEST(dqd_transform_passes_a_zero_pivot)
{
  buffer<double>                      z(16, 0.0);
  const numerical::detail::qd_array   at {z.data()};
  const std::array<double, 4>         q  {0.0, 1.0, 2.0, 3.0};
  const std::array<double, 3>         e  {0.0, 0.5, 0.25};
  for (std::size_t k {0}; k < 4; ++k) {
    at(static_cast<numerical::detail::qd_index>((4 * k) + 1)) = q[k];
    if (k < 3) {
      at(static_cast<numerical::detail::qd_index>((4 * k) + 3)) = e[k];
    }
  }
  numerical::detail::dqds_state state;
  double                        dmin {0.0};
  numerical::detail::dqd_transform(1, 4, at, 0, dmin, state);
  double trace                       {0.0};
  for (numerical::detail::qd_index k {1}; k <= 4; ++k) {
    trace += at((4 * k) - 2) + (k < 4 ? at(4 * k) : 0.0);
  }
  EXPECT(at(2) == 0.0 && at(4) == 0.0);
  EXPECT(std::abs(trace - 6.75) <= 8.0 * std::numeric_limits<double>::epsilon());
  EXPECT(dmin >= 0.0 && at(16) == 0.0);
}

// Orders 0 to 2, a diagonal, and a superdiagonal whose squares underflow at the scale dqds works at: the values all
// the same, each magnitude of the diagonal sorted where the superdiagonal is zero.
TEST(dqds_takes_small_orders_and_diagonals)
{
  buffer<double> none;
  EXPECT(numerical::detail::dqds_singular_values(none, none, 0));
  buffer<double> one {-3.0};
  EXPECT(numerical::detail::dqds_singular_values(one, buffer<double> {0.0}, 1) && one[0] == 3.0);
  buffer<double> two {3.0, 4.0};
  EXPECT(numerical::detail::dqds_singular_values(two, buffer<double> {2.0, 0.0}, 2));
  const double root  {std::sqrt(((25.0 + 4.0) * (25.0 + 4.0)) - (4.0 * 144.0))}; // σ² = (29 ± root) / 2
  EXPECT(std::abs(two[0] - std::sqrt((29.0 + root) / 2.0)) < 4e-16 * two[0]);
  EXPECT(std::abs(two[1] - std::sqrt((29.0 - root) / 2.0)) < 4e-16 * two[1]);
  buffer<double> diagonal {1.0, -5.0, 0.0, 2.0};
  EXPECT(numerical::detail::dqds_singular_values(diagonal, buffer<double>(4, 0.0), 4));
  EXPECT((diagonal == buffer<double> {5.0, 2.0, 1.0, 0.0}));
}

// Past the bound of steps dqds gives up, the diagonal untouched, and the singular values come from QR instead.
TEST(dqds_hands_over_to_qr_past_its_bound)
{
  const drawn_bidiagonal b {draw_bidiagonal(7)};
  const std::size_t      n {b.d.size()};
  buffer<double>         d(b.d.begin(), b.d.end());
  EXPECT(!numerical::detail::dqds_singular_values(d, buffer<double>(b.e.begin(), b.e.end()), n, 0));
  EXPECT(std::equal(d.begin(), d.end(), b.d.begin()));
  buffer<double> square(n * n, 0.0);
  for (std::size_t i {0}; i < n; ++i) {
    square[(i * n) + i] = b.d[i];
    if (i + 1 < n) {
      square[(i * n) + i + 1] = b.e[i];
    }
  }
  const buffer<double> handed {numerical::detail::square_singular_values(square, n, 0)};
  buffer<double>       qd(n, 0.0);
  buffer<double>       qe(n, 0.0);
  buffer<double>       none;
  buffer<double>       none_too;
  numerical::detail::bidiagonalize(square, n, qd, qe, none, none_too, false);
  numerical::detail::bidiagonal_qr(qd, qe, none, none_too, n, numerical::detail::svd_qr_steps_per_value * n);
  EXPECT(handed == qd);
}
