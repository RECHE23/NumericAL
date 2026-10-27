/*!
 * \file dqds.hpp
 * \brief The singular values of a real bidiagonal by the dqds algorithm (Fernando and Parlett's differential
 *        quotient-difference with shifts, as LAPACK's `dlasq1`–`dlasq6`): the eigenvalues of `BᵀB` from the qd array of
 *        the squares of `B`'s entries, every transform's numbers non-negative, so that each value is found to high
 *        relative accuracy. The routines keep LAPACK's one-based indices into the array `Z` of `4n` entries, so that
 *        each line reads against its original.
 */
#ifndef NUMERICAL_DQDS_HPP
#define NUMERICAL_DQDS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>

#include "buffer.hpp"

namespace numerical::detail {

  //! \brief An index into the qd array, one-based and signed as LAPACK's.
  using qd_index = std::ptrdiff_t;

  //! \brief The qd array `Z` of `4n` entries, read one-based: `Z(i)` is `z[i − 1]`.
  class qd_array
  {
  public:

    //! \brief The array over \p z.
    explicit qd_array(double* z) noexcept
      : z_ {z}
    {}

    //! \brief `Z(i)`.
    [[nodiscard]] double& operator()(qd_index i) const noexcept
    {
      return z_[i - 1];
    }

  private:

    double* z_; //!< The first entry.
  };

  //! \brief What one dqds step hands the next: the least `d`s of the last transform and the shift's memory.
  struct dqds_state
  {
    double   dmin1 {0.0}; //!< The least `d` but the last.
    double   dmin2 {0.0}; //!< The least `d` but the last two.
    double   dn    {0.0}; //!< The last `d`.
    double   dn1   {0.0}; //!< The `d` before it.
    double   dn2   {0.0}; //!< The `d` before that.
    double   g     {0.0}; //!< The fraction of the least `d` that a shift of type −6 takes.
    double   tau   {0.0}; //!< The shift.
    qd_index ttype {0};   //!< Which estimate gave the shift.
  };

  //! \brief LAPACK's bound of the off-diagonal against the diagonal under which a value deflates: `(100 ε)²`.
  inline constexpr double dqds_tolerance_squared {(std::numeric_limits<double>::epsilon() * 100.0) *
                                                  (std::numeric_limits<double>::epsilon() * 100.0)};

  //! \brief The steps of the inner iteration per value of its block before dqds gives up (LAPACK's 100).
  inline constexpr qd_index dqds_sweeps {100};

  /*!
   * \brief The eigenvalues of the 2 × 2 `[[a, √b], [√b, c]]`-like block of a qd array, `a = q₁`, `b = e₁`, `c = q₂`:
   *        the larger into \p a, the smaller into \p c, each to high relative accuracy. No change where \p b is
   *        negligible against \p c, nor where half the sum `(a − c) + b` underflows to zero.
   */
  inline void qd_pair(double& a,
                      double  b,
                      double& c) noexcept
  {
    if (c > a) {
      std::swap(a, c);
    }
    const double t {0.5 * ((a - c) + b)};
    if (b > c * dqds_tolerance_squared && t != 0.0) {
      double s {c * (b / t)};
      if (s <= t) {
        s = c * (b / (t * (1.0 + std::sqrt(1.0 + (s / t)))));
      }
      else {
        s = c * (b / (t + (std::sqrt(t) * std::sqrt(t + s))));
      }
      const double larger {a + (s + b)};
      c = c * (a / larger);
      a = larger;
    }
  }

  /*!
   * \brief The shift of the next transform of the block `i0…n0` (LAPACK's `dlasq4`): an estimate of the least
   *        eigenvalue from the last transform's least `d`s, by the case \p state's `ttype` records. Some cases return
   *        with the shift as it was, where a ratio of the array exceeds one.
   * \param[in]     i0    The block's first index.
   * \param[in]     n0    Its last.
   * \param[in]     z     The qd array.
   * \param[in]     pp    Which of the ping-pong halves holds the array (0 or 1).
   * \param[in]     n0in  The last index before this step's deflations.
   * \param[in]     dmin  The last transform's least `d`.
   * \param[in,out] state The other least `d`s; on return, the shift, its type and `g`.
   */
  inline void dqds_shift(qd_index        i0,
                         qd_index        n0,
                         const qd_array& z,
                         qd_index        pp,
                         qd_index        n0in,
                         double          dmin,
                         dqds_state&     state)
  {
    constexpr double cnst1  {0.5630};
    constexpr double cnst2  {1.010};
    constexpr double cnst3  {1.050};
    constexpr double qurtr  {0.25};
    constexpr double third  {0.3330};
    constexpr double half   {0.5};
    constexpr double hundrd {100.0};
    if (dmin <= 0.0) {
      state.tau   = -dmin;
      state.ttype = -1;
      return;
    }
    const double   dmin1 {state.dmin1};
    const double   dmin2 {state.dmin2};
    const double   dn    {state.dn};
    const double   dn1   {state.dn1};
    const double   dn2   {state.dn2};
    const qd_index nn    {(4 * n0) + pp};
    double         s     {0.0};
    double         a2    {0.0};
    double         b1    {0.0};
    double         b2    {0.0};
    double         gam   {0.0};
    double         gap1  {0.0};
    double         gap2  {0.0};
    qd_index       np    {0};
    if (n0in == n0) { // no value deflated
      if (dmin == dn || dmin == dn1) {
        b1 = std::sqrt(z(nn - 3)) * std::sqrt(z(nn - 5));
        b2 = std::sqrt(z(nn - 7)) * std::sqrt(z(nn - 9));
        a2 = z(nn - 7) + z(nn - 5);
        if (dmin == dn && dmin1 == dn1) { // the two least at the end: Gershgorin's discs of the trailing 2 × 2
          gap2 = dmin2 - a2 - (dmin2 * qurtr);
          if (gap2 > 0.0 && gap2 > b2) {
            gap1 = a2 - dn - ((b2 / gap2) * b2);
          }
          else {
            gap1 = a2 - dn - (b1 + b2);
          }
          if (gap1 > 0.0 && gap1 > b1) {
            s           = std::max(dn - ((b1 / gap1) * b1), half * dmin);
            state.ttype = -2;
          }
          else {
            s = 0.0;
            if (dn > b1) {
              s = dn - b1;
            }
            if (a2 > (b1 + b2)) {
              s = std::min(s, a2 - (b1 + b2));
            }
            s           = std::max(s, third * dmin);
            state.ttype = -3;
          }
        }
        else { // the least at one of the two ends: a Rayleigh-quotient-like estimate along the array
          state.ttype = -4;
          s           = qurtr * dmin;
          if (dmin == dn) {
            gam = dn;
            a2  = 0.0;
            if (z(nn - 5) > z(nn - 7)) {
              return;
            }
            b2 = z(nn - 5) / z(nn - 7);
            np = nn - 9;
          }
          else {
            np  = nn - (2 * pp);
            gam = dn1;
            if (z(np - 4) > z(np - 2)) {
              return;
            }
            a2 = z(np - 4) / z(np - 2);
            if (z(nn - 9) > z(nn - 11)) {
              return;
            }
            b2 = z(nn - 9) / z(nn - 11);
            np = nn - 13;
          }
          a2 = a2 + b2;
          for (qd_index i4 {np}; i4 >= (4 * i0) - 1 + pp; i4 -= 4) {
            if (b2 == 0.0) {
              break;
            }
            b1 = b2;
            if (z(i4) > z(i4 - 2)) {
              return;
            }
            b2 = b2 * (z(i4) / z(i4 - 2));
            a2 = a2 + b2;
            if (hundrd * std::max(b2, b1) < a2 || cnst1 < a2) {
              break;
            }
          }
          a2 = cnst3 * a2;
          if (a2 < cnst1) {
            s = gam * (1.0 - std::sqrt(a2)) / (1.0 + a2);
          }
        }
      }
      else if (dmin == dn2) {
        state.ttype = -5;
        s           = qurtr * dmin;
        np          = nn - (2 * pp);
        b1          = z(np - 2);
        b2          = z(np - 6);
        gam         = dn2;
        if (z(np - 8) > b2 || z(np - 4) > b1) {
          return;
        }
        a2 = (z(np - 8) / b2) * (1.0 + (z(np - 4) / b1));
        if (n0 - i0 > 2) {
          b2 = z(nn - 13) / z(nn - 15);
          a2 = a2 + b2;
          for (qd_index i4 {nn - 17}; i4 >= (4 * i0) - 1 + pp; i4 -= 4) {
            if (b2 == 0.0) {
              break;
            }
            b1 = b2;
            if (z(i4) > z(i4 - 2)) {
              return;
            }
            b2 = b2 * (z(i4) / z(i4 - 2));
            a2 = a2 + b2;
            if (hundrd * std::max(b2, b1) < a2 || cnst1 < a2) {
              break;
            }
          }
          a2 = cnst3 * a2;
        }
        if (a2 < cnst1) {
          s = gam * (1.0 - std::sqrt(a2)) / (1.0 + a2);
        }
      }
      else { // the least inside the array: a fraction of it, raised after each shift of this type
        if (state.ttype == -6) {
          state.g = state.g + (third * (1.0 - state.g));
        }
        else if (state.ttype == -18) {
          state.g = qurtr * third;
        }
        else {
          state.g = qurtr;
        }
        s           = state.g * dmin;
        state.ttype = -6;
      }
    }
    else if (n0in == n0 + 1) { // one value deflated
      if (dmin1 == dn1 && dmin2 == dn2) {
        state.ttype = -7;
        s           = third * dmin1;
        if (z(nn - 5) > z(nn - 7)) {
          return;
        }
        b1 = z(nn - 5) / z(nn - 7);
        b2 = b1;
        if (b2 != 0.0) {
          for (qd_index i4 {(4 * n0) - 9 + pp}; i4 >= (4 * i0) - 1 + pp; i4 -= 4) {
            a2 = b1;
            if (z(i4) > z(i4 - 2)) {
              return;
            }
            b1 = b1 * (z(i4) / z(i4 - 2));
            b2 = b2 + b1;
            if (hundrd * std::max(b1, a2) < b2) {
              break;
            }
          }
        }
        b2   = std::sqrt(cnst3 * b2);
        a2   = dmin1 / (1.0 + (b2 * b2));
        gap2 = (half * dmin2) - a2;
        if (gap2 > 0.0 && gap2 > b2 * a2) {
          s = std::max(s, a2 * (1.0 - (cnst2 * a2 * (b2 / gap2) * b2)));
        }
        else {
          s           = std::max(s, a2 * (1.0 - (cnst2 * b2)));
          state.ttype = -8;
        }
      }
      else {
        s = qurtr * dmin1;
        if (dmin1 == dn1) {
          s = half * dmin1;
        }
        state.ttype = -9;
      }
    }
    else if (n0in == n0 + 2) { // two values deflated
      if (dmin2 == dn2 && 2.0 * z(nn - 5) < z(nn - 7)) {
        state.ttype = -10;
        s           = third * dmin2;
        b1          = z(nn - 5) / z(nn - 7); // under one half: the condition above
        b2          = b1;
        if (b2 != 0.0) {
          for (qd_index i4 {(4 * n0) - 9 + pp}; i4 >= (4 * i0) - 1 + pp; i4 -= 4) {
            if (z(i4) > z(i4 - 2)) {
              return;
            }
            b1 = b1 * (z(i4) / z(i4 - 2));
            b2 = b2 + b1;
            if (hundrd * b1 < b2) {
              break;
            }
          }
        }
        b2   = std::sqrt(cnst3 * b2);
        a2   = dmin2 / (1.0 + (b2 * b2));
        gap2 = z(nn - 7) + z(nn - 9) - (std::sqrt(z(nn - 11)) * std::sqrt(z(nn - 9))) - a2;
        if (gap2 > 0.0 && gap2 > b2 * a2) {
          s = std::max(s, a2 * (1.0 - (cnst2 * a2 * (b2 / gap2) * b2)));
        }
        else {
          s = std::max(s, a2 * (1.0 - (cnst2 * b2)));
        }
      }
      else {
        s           = qurtr * dmin2;
        state.ttype = -11;
      }
    }
    else { // more than two deflated: no estimate
      s           = 0.0;
      state.ttype = -12;
    }
    state.tau = s;
  }

  /*!
   * \brief One dqds transform of the block `i0…n0` (at least three values) with the shift \p tau, from the half \p pp
   *        of the array into the other (LAPACK's `dlasq5`, its IEEE form): a zero shift where it is under half of
   *        `ε (σ + τ)`, the `d`s it takes below that bound set to zero. \p dmin is NaN where the last `d` is not
   *        finite: a `d + e` of zero, or past overflow, makes every `d` after it NaN, which a least of them alone
   *        does not show.
   */
  inline void dqds_transform(qd_index        i0,
                             qd_index        n0,
                             const qd_array& z,
                             qd_index        pp,
                             double&         tau,
                             double          sigma,
                             double&         dmin,
                             dqds_state&     state)
  {
    const double dthresh {std::numeric_limits<double>::epsilon() * (sigma + tau)};
    if (tau < dthresh * 0.5) {
      tau = 0.0;
    }
    qd_index   j4         {(4 * i0) + pp - 3};
    double     emin       {z(j4 + 4)};
    double     d          {z(j4) - tau};
    const bool zero_shift {tau == 0.0};
    dmin = d;
    if (pp == 0) {
      for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
        z(j4 - 2) = d + z(j4 - 1);
        const double temp {z(j4 + 1) / z(j4 - 2)};
        d = (d * temp) - tau;
        if (zero_shift && d < dthresh) {
          d = 0.0;
        }
        dmin  = std::min(dmin, d);
        z(j4) = z(j4 - 1) * temp;
        emin  = std::min(z(j4), emin);
      }
    }
    else {
      for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
        z(j4 - 3) = d + z(j4);
        const double temp {z(j4 + 2) / z(j4 - 3)};
        d = (d * temp) - tau;
        if (zero_shift && d < dthresh) {
          d = 0.0;
        }
        dmin      = std::min(dmin, d);
        z(j4 - 1) = z(j4) * temp;
        emin      = std::min(z(j4 - 1), emin);
      }
    }
    // The last two steps apart, their d's kept for the shift.
    state.dn2   = d;
    state.dmin2 = dmin;
    j4          = (4 * (n0 - 2)) - pp;
    qd_index j4p2 {j4 + (2 * pp) - 1};
    z(j4 - 2)          = state.dn2 + z(j4p2);
    z(j4)              = z(j4p2 + 2) * (z(j4p2) / z(j4 - 2));
    state.dn1          = (z(j4p2 + 2) * (state.dn2 / z(j4 - 2))) - tau;
    dmin               = std::min(dmin, state.dn1);
    state.dmin1        = dmin;
    j4                 = j4 + 4;
    j4p2               = j4 + (2 * pp) - 1;
    z(j4 - 2)          = state.dn1 + z(j4p2);
    z(j4)              = z(j4p2 + 2) * (z(j4p2) / z(j4 - 2));
    state.dn           = (z(j4p2 + 2) * (state.dn1 / z(j4 - 2))) - tau;
    dmin               = std::min(dmin, state.dn);
    z(j4 + 2)          = state.dn;
    z((4 * n0) - pp)   = emin;
    if (!std::isfinite(state.dn)) {
      dmin = std::numeric_limits<double>::quiet_NaN();
    }
  }

  /*!
   * \brief One dqd transform (no shift) of the block `i0…n0` (at least three values), from the half \p pp of the
   *        array into the other, each quotient guarded against underflow and overflow (LAPACK's `dlasq6`): the
   *        transform a shifted one falls back to when its numbers are not finite.
   */
  inline void dqd_transform(qd_index        i0,
                            qd_index        n0,
                            const qd_array& z,
                            qd_index        pp,
                            double&         dmin,
                            dqds_state&     state)
  {
    const double safmin {std::numeric_limits<double>::min()};
    qd_index     j4     {(4 * i0) + pp - 3};
    double       emin   {z(j4 + 4)};
    double       d      {z(j4)};
    dmin = d;
    if (pp == 0) {
      for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
        z(j4 - 2) = d + z(j4 - 1);
        if (z(j4 - 2) == 0.0) {
          z(j4) = 0.0;
          d     = z(j4 + 1);
          dmin  = d;
          emin  = 0.0;
        }
        else if (safmin * z(j4 + 1) < z(j4 - 2) && safmin * z(j4 - 2) < z(j4 + 1)) {
          const double temp {z(j4 + 1) / z(j4 - 2)};
          z(j4) = z(j4 - 1) * temp;
          d     = d * temp;
        }
        else {
          z(j4) = z(j4 + 1) * (z(j4 - 1) / z(j4 - 2));
          d     = z(j4 + 1) * (d / z(j4 - 2));
        }
        dmin = std::min(dmin, d);
        emin = std::min(emin, z(j4));
      }
    }
    else {
      for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
        z(j4 - 3) = d + z(j4);
        if (z(j4 - 3) == 0.0) {
          z(j4 - 1) = 0.0;
          d         = z(j4 + 2);
          dmin      = d;
          emin      = 0.0;
        }
        else if (safmin * z(j4 + 2) < z(j4 - 3) && safmin * z(j4 - 3) < z(j4 + 2)) {
          const double temp {z(j4 + 2) / z(j4 - 3)};
          z(j4 - 1) = z(j4) * temp;
          d         = d * temp;
        }
        else {
          z(j4 - 1) = z(j4 + 2) * (z(j4) / z(j4 - 3));
          d         = z(j4 + 2) * (d / z(j4 - 3));
        }
        dmin = std::min(dmin, d);
        emin = std::min(emin, z(j4 - 1));
      }
    }
    // The last two steps apart, their d's kept for the shift.
    state.dn2   = d;
    state.dmin2 = dmin;
    qd_index j4p2 {0};
    for (int last {0}; last < 2; ++last) {
      j4   = last == 0 ? (4 * (n0 - 2)) - pp : j4 + 4;
      j4p2 = j4 + (2 * pp) - 1;
      const double before {last == 0 ? state.dn2 : state.dn1};
      double       after  {0.0};
      z(j4 - 2) = before + z(j4p2);
      if (z(j4 - 2) == 0.0) {
        z(j4) = 0.0;
        after = z(j4p2 + 2);
        dmin  = after;
        emin  = 0.0;
      }
      else if (safmin * z(j4p2 + 2) < z(j4 - 2) && safmin * z(j4 - 2) < z(j4p2 + 2)) {
        const double temp {z(j4p2 + 2) / z(j4 - 2)};
        z(j4) = z(j4p2) * temp;
        after = before * temp;
      }
      else {
        z(j4) = z(j4p2 + 2) * (z(j4p2) / z(j4 - 2));
        after = z(j4p2 + 2) * (before / z(j4 - 2));
      }
      dmin = std::min(dmin, after);
      if (last == 0) {
        state.dn1   = after;
        state.dmin1 = dmin;
      }
      else {
        state.dn = after;
      }
    }
    z(j4 + 2)        = state.dn;
    z((4 * n0) - pp) = emin;
  }

  //! \brief Turns the block `i0…n0` of the array end for end, both halves (LAPACK's flip in `dlasq2` and `dlasq3`).
  inline void qd_flip(qd_index        i0,
                      qd_index        n0,
                      const qd_array& z) noexcept
  {
    const qd_index ipn4 {4 * (i0 + n0)};
    for (qd_index i4 {4 * i0}; i4 <= 2 * (i0 + n0 - 1); i4 += 4) {
      std::swap(z(i4 - 3), z(ipn4 - i4 - 3));
      std::swap(z(i4 - 2), z(ipn4 - i4 - 2));
      std::swap(z(i4 - 1), z(ipn4 - i4 - 5));
      std::swap(z(i4), z(ipn4 - i4 - 4));
    }
  }

  /*!
   * \brief One step on the block `i0…n0` (LAPACK's `dlasq3`): deflates the values its end has converged to (into
   *        `Z(4k − 3)`, \p sigma added back), flips the block where its far end is the smaller, then one transform with
   *        the shift of \ref dqds_shift. A shift that makes a `d` negative is lowered and the transform retried; one
   *        whose numbers are not finite is dropped, the transform then unshifted, guarded as \ref dqd_transform.
   * \param[in]     i0    The block's first index.
   * \param[in,out] n0    Its last; on return, less the values deflated.
   * \param[in]     z     The qd array.
   * \param[in,out] pp    The half that holds the array (2 on the first step after a flip in \ref dqds_squares).
   * \param[in,out] dmin  The last transform's least `d`.
   * \param[in,out] sigma The shifts so far, summed.
   * \param[in,out] desig The rounding error of that sum.
   * \param[in,out] qmax  The largest `q` of the block.
   * \param[in,out] state The rest of the last transform's results and the shift's memory.
   */
  inline void dqds_step(qd_index        i0,
                        qd_index&       n0,
                        const qd_array& z,
                        qd_index&       pp,
                        double&         dmin,
                        double&         sigma,
                        double&         desig,
                        double&         qmax,
                        dqds_state&     state)
  {
    constexpr double cbias {1.50};
    constexpr double qurtr {0.25};
    const double     eps   {std::numeric_limits<double>::epsilon()};
    const double     tol   {eps * 100.0};
    const double     tol2  {dqds_tolerance_squared};
    const qd_index   n0in  {n0};
    while (n0 >= i0) {
      if (n0 == i0) { // one value left
        z((4 * n0) - 3) = z((4 * n0) + pp - 3) + sigma;
        --n0;
        continue;
      }
      const qd_index nn {(4 * n0) + pp};
      if (n0 != i0 + 1) {
        if (z(nn - 5) <= tol2 * (sigma + z(nn - 3)) || z(nn - (2 * pp) - 4) <= tol2 * z(nn - 7)) { // the last
          z((4 * n0) - 3) = z((4 * n0) + pp - 3) + sigma;
          --n0;
          continue;
        }
        if (z(nn - 9) > tol2 * sigma && z(nn - (2 * pp) - 8) > tol2 * z(nn - 11)) {
          break; // nothing converged: on to the transform
        }
      }
      // The last two.
      qd_pair(z(nn - 7), z(nn - 5), z(nn - 3));
      z((4 * n0) - 7) = z(nn - 7) + sigma;
      z((4 * n0) - 3) = z(nn - 3) + sigma;
      n0             -= 2;
    }
    if (n0 < i0) {
      return;
    }
    if (pp == 2) {
      pp = 0;
    }
    if (dmin <= 0.0 || n0 < n0in) { // a new start: flip if the far end is much the smaller
      if (cbias * z((4 * i0) + pp - 3) < z((4 * n0) + pp - 3)) {
        qd_flip(i0, n0, z);
        if (n0 - i0 <= 4) {
          z((4 * n0) + pp - 1) = z((4 * i0) + pp - 1);
          z((4 * n0) - pp)     = z((4 * i0) - pp);
        }
        state.dmin2          = std::min(state.dmin2, z((4 * n0) + pp - 1));
        z((4 * n0) + pp - 1) = std::min({z((4 * n0) + pp - 1), z((4 * i0) + pp - 1), z((4 * i0) + pp + 3)});
        z((4 * n0) - pp)     = std::min({z((4 * n0) - pp), z((4 * i0) - pp), z((4 * i0) - pp + 4)});
        qmax                 = std::max({qmax, z((4 * i0) + pp - 3), z((4 * i0) + pp + 1)});
        dmin                 = -0.0;
      }
    }
    dqds_shift(i0, n0, z, pp, n0in, dmin, state);
    for (;;) {
      dqds_transform(i0, n0, z, pp, state.tau, sigma, dmin, state);
      if (std::isnan(dmin)) { // not finite: again unshifted, then guarded
        if (state.tau == 0.0) {
          dqd_transform(i0, n0, z, pp, dmin, state);
          break;
        }
        state.tau = 0.0;
        continue;
      }
      if (dmin >= 0.0) { // the least d of a finite transform bounds the others: all non-negative
        break;
      }
      if (state.dmin1 > 0.0 && z((4 * (n0 - 1)) - pp) < tol * (sigma + state.dn1) && std::abs(state.dn) < tol * sigma) {
        // Only the last d is negative, and negligible: the last value has converged to the shift.
        z((4 * (n0 - 1)) - pp + 2) = 0.0;
        dmin                       = 0.0;
        break;
      }
      // The shift was too large: lower it.
      if (state.ttype < -22) {
        state.tau = 0.0;
      }
      else if (state.dmin1 > 0.0) {
        state.tau    = (state.tau + dmin) * (1.0 - (2.0 * eps));
        state.ttype -= 11;
      }
      else {
        state.tau    = qurtr * state.tau;
        state.ttype -= 12;
      }
    }
    // sigma += tau, its rounding error carried in desig.
    double t {0.0};
    if (state.tau < sigma) {
      desig = desig + state.tau;
      t     = sigma + desig;
      desig = desig - (t - sigma);
    }
    else {
      t     = sigma + state.tau;
      desig = sigma - (t - state.tau) + desig;
    }
    sigma = t;
  }

  /*!
   * \brief The eigenvalues of the positive semi-definite tridiagonal whose qd array is in \p z (LAPACK's `dlasq2`):
   *        `q₁ e₁ q₂ e₂ … qₙ`, non-negative, in its first `2n − 1` entries, `n` at least two.
   * \param[in,out] z      The array, of at least `4n` entries; on success, the eigenvalues in its first `n`,
   *                       descending.
   * \param[in]     n      The order.
   * \param[in]     sweeps The steps of the inner iteration per value of its block before giving up.
   * \return Whether the iteration converged within that bound.
   */
  [[nodiscard]] inline bool dqds_squares(buffer<double>& z,
                                         qd_index        n,
                                         qd_index        sweeps)
  {
    const qd_array   zz         {z.data()};
    constexpr double cbias      {1.50};
    constexpr double half       {0.5};
    constexpr double four       {4.0};
    const double     safmin     {std::numeric_limits<double>::min()};
    const double     tol2       {dqds_tolerance_squared};
    const auto       descending {[&] {
                                   std::sort(z.begin(), z.begin() + n, std::greater<> {});
                                   return true;
                                 }};
    if (n == 2) {
      qd_pair(zz(1), zz(2), zz(3));
      zz(2) = zz(3);
      return true;
    }
    zz(2 * n) = 0.0;
    double e  {0.0};
    for (qd_index k {2}; k <= 2 * (n - 1); k += 2) {
      e = e + zz(k);
    }
    if (e == 0.0) { // diagonal: the q's themselves
      for (qd_index k {2}; k <= n; ++k) {
        zz(k) = zz((2 * k) - 1);
      }
      return descending();
    }
    // Spread q₁ e₁ … qₙ to every fourth entry, both halves.
    for (qd_index k {2 * n}; k >= 2; k -= 2) {
      zz(2 * k)       = 0.0;
      zz((2 * k) - 1) = zz(k);
      zz((2 * k) - 2) = 0.0;
      zz((2 * k) - 3) = zz(k - 1);
    }
    qd_index i0 {1};
    qd_index n0 {n};
    if (cbias * zz((4 * i0) - 3) < zz((4 * n0) - 3)) { // the far end the smaller: flip
      const qd_index ipn4 {4 * (i0 + n0)};
      for (qd_index i4 {4 * i0}; i4 <= 2 * (i0 + n0 - 1); i4 += 4) {
        std::swap(zz(i4 - 3), zz(ipn4 - i4 - 3));
        std::swap(zz(i4 - 1), zz(ipn4 - i4 - 5));
      }
    }
    // Two dqd transforms, unshifted, each splitting at the negligible e's.
    qd_index pp   {0};
    double   d    {0.0};
    double   qmax {0.0};
    for (int k {1}; k <= 2; ++k) {
      d = zz((4 * n0) + pp - 3);
      for (qd_index i4 {(4 * (n0 - 1)) + pp}; i4 >= (4 * i0) + pp; i4 -= 4) {
        if (zz(i4 - 1) <= tol2 * d) {
          zz(i4 - 1) = -0.0;
          d          = zz(i4 - 3);
        }
        else {
          d = zz(i4 - 3) * (d / (d + zz(i4 - 1)));
        }
      }
      d = zz((4 * i0) + pp - 3);
      for (qd_index i4 {(4 * i0) + pp}; i4 <= (4 * (n0 - 1)) + pp; i4 += 4) {
        zz(i4 - (2 * pp) - 2) = d + zz(i4 - 1);
        if (zz(i4 - 1) <= tol2 * d) {
          zz(i4 - 1)            = -0.0;
          zz(i4 - (2 * pp) - 2) = d;
          zz(i4 - (2 * pp))     = 0.0;
          d                     = zz(i4 + 1);
        }
        else if (safmin * zz(i4 + 1) < zz(i4 - (2 * pp) - 2) && safmin * zz(i4 - (2 * pp) - 2) < zz(i4 + 1)) {
          const double temp {zz(i4 + 1) / zz(i4 - (2 * pp) - 2)};
          zz(i4 - (2 * pp)) = zz(i4 - 1) * temp;
          d                 = d * temp;
        }
        else {
          zz(i4 - (2 * pp)) = zz(i4 + 1) * (zz(i4 - 1) / zz(i4 - (2 * pp) - 2));
          d                 = zz(i4 + 1) * (d / zz(i4 - (2 * pp) - 2));
        }
      }
      zz((4 * n0) - pp - 2) = d;
      pp                    = 1 - pp;
    }
    dqds_state state;
    // Each pass takes the last unreduced block, n0 down to its split; n0 falls with each.
    while (n0 >= 1) {
      double desig {0.0};
      double sigma {n0 == n ? 0.0 : -zz((4 * n0) - 1)}; // a split stored −σ at its end
      double emax  {0.0};
      double qmin  {zz((4 * n0) - 3)};
      qmax = qmin;
      qd_index i4  {4 * n0};
      for (; i4 >= 8; i4 -= 4) {
        if (zz(i4 - 5) <= 0.0) {
          break;
        }
        if (qmin >= four * emax) {
          qmin = std::min(qmin, zz(i4 - 3));
          emax = std::max(emax, zz(i4 - 5));
        }
        qmax = std::max(qmax, zz(i4 - 7) + zz(i4 - 5));
      }
      i0 = i4 / 4;
      pp = 0;
      if (n0 - i0 > 1) { // flip if the least d of a dqd sits in the block's first third
        double   dee    {zz((4 * i0) - 3)};
        double   deemin {dee};
        qd_index kmin   {i0};
        for (qd_index j {(4 * i0) + 1}; j <= (4 * n0) - 3; j += 4) {
          dee = zz(j) * (dee / (dee + zz(j - 2)));
          if (dee <= deemin) {
            deemin = dee;
            kmin   = (j + 3) / 4;
          }
        }
        if ((kmin - i0) * 2 < n0 - kmin && deemin <= half * zz((4 * n0) - 3)) {
          pp = 2;
          qd_flip(i0, n0, zz);
        }
      }
      double         dmin  {-std::max(0.0, qmin - (2.0 * std::sqrt(qmin) * std::sqrt(emax)))};
      const qd_index nbig  {sweeps * (n0 - i0 + 1)};
      qd_index       steps {0};
      while (i0 <= n0) {
        if (steps++ == nbig) {
          return false;
        }
        dqds_step(i0, n0, zz, pp, dmin, sigma, desig, qmax, state);
        pp = 1 - pp;
        if (pp == 0 && n0 - i0 >= 3 && (zz(4 * n0) <= tol2 * qmax || zz((4 * n0) - 1) <= tol2 * sigma)) {
          // A negligible e inside the block: split there, the part above left for a later pass.
          qd_index splt   {i0 - 1};
          qmax            = zz((4 * i0) - 3);
          double   emin   {zz((4 * i0) - 1)};
          double   oldemn {zz(4 * i0)};
          for (qd_index j {4 * i0}; j <= 4 * (n0 - 3); j += 4) {
            if (zz(j) <= tol2 * zz(j - 3) || zz(j - 1) <= tol2 * sigma) {
              zz(j - 1) = -sigma;
              splt      = j / 4;
              qmax      = 0.0;
              emin      = zz(j + 3);
              oldemn    = zz(j + 4);
            }
            else {
              qmax   = std::max(qmax, zz(j + 1));
              emin   = std::min(emin, zz(j - 1));
              oldemn = std::min(oldemn, zz(j));
            }
          }
          zz((4 * n0) - 1) = emin;
          zz(4 * n0)       = oldemn;
          i0               = splt + 1;
        }
      }
    }
    for (qd_index k {2}; k <= n; ++k) {
      zz(k) = zz((4 * k) - 3);
    }
    return descending();
  }

  /*!
   * \brief The singular values of the real upper bidiagonal (\p d, \p e) of order \p n by dqds (LAPACK's `dlasq1`),
   *        descending into \p d: its entries' magnitudes scaled so that the largest is `√(ε / DBL_MIN)`, their
   *        squares' eigenvalues found by \ref dqds_squares, the square roots scaled back.
   * \param[in,out] d      The diagonal; on success, the singular values, descending. Untouched on failure.
   * \param[in]     e      The superdiagonal in `e[0…n-2]`.
   * \param[in]     n      The order, at least one.
   * \param[in]     sweeps As \ref dqds_squares takes it.
   * \return Whether dqds converged within that bound.
   */
  [[nodiscard]] inline bool dqds_singular_values(buffer<double>&       d,
                                                 const buffer<double>& e,
                                                 std::size_t           n,
                                                 qd_index              sweeps = dqds_sweeps)
  {
    double sigmx {0.0};
    for (std::size_t i {0}; i + 1 < n; ++i) {
      sigmx = std::max(sigmx, std::abs(e[i]));
    }
    if (sigmx == 0.0) { // diagonal
      for (std::size_t i {0}; i < n; ++i) {
        d[i] = std::abs(d[i]);
      }
      std::sort(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(n), std::greater<> {});
      return true;
    }
    for (std::size_t i {0}; i < n; ++i) {
      sigmx = std::max(sigmx, std::abs(d[i]));
    }
    // Each entry over the largest first: the scale's quotient by a subnormal largest would overflow.
    const double   scale {std::sqrt(std::numeric_limits<double>::epsilon() / std::numeric_limits<double>::min())};
    buffer<double> z(4 * n, 0.0);
    for (std::size_t i {0}; i < n; ++i) {
      const double v {(std::abs(d[i]) / sigmx) * scale};
      z[2 * i] = v * v;
    }
    for (std::size_t i {0}; i + 1 < n; ++i) {
      const double v {(std::abs(e[i]) / sigmx) * scale};
      z[(2 * i) + 1] = v * v;
    }
    if (!dqds_squares(z, static_cast<qd_index>(n), sweeps)) {
      return false;
    }
    for (std::size_t i {0}; i < n; ++i) {
      d[i] = (std::sqrt(z[i]) / scale) * sigmx;
    }
    return true;
  }
} // namespace numerical::detail

#endif // NUMERICAL_DQDS_HPP
