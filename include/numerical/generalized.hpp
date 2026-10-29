/*!
 * \file generalized.hpp
 * \brief The generalized Schur decomposition (QZ) of a complex pencil `A − λ B`, as LAPACK's `zgges`: the reduction to
 *        Hessenberg-triangular form (`zgghrd`), the single-shift QZ iteration (`zhgeqz`), and the reordering of its
 *        eigenvalues by exchanges of adjacent ones (`ztgexc`, `ztgsen`).
 */
#ifndef NUMERICAL_GENERALIZED_HPP
#define NUMERICAL_GENERALIZED_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "matrix_functions.hpp"
#include "svd.hpp"

namespace numerical {

  namespace detail {

    //! \brief `|Re z| + |Im z|`, LAPACK's `ABS1`.
    [[nodiscard]] inline double modulus_one(const std::complex<double>& z)
    {
      return std::abs(z.real()) + std::abs(z.imag());
    }

    /*!
     * \brief A complex pencil in generalized Schur form being formed: `S` and `T` (`n × n`, row-major) with
     *        `A = Q S Zᴴ` and `B = Q T Zᴴ`, \ref q and \ref z unitary.
     */
    struct qz_form
    {
      std::size_t                  n {0}; //!< The order.
      buffer<std::complex<double>> s;     //!< `S`, upper Hessenberg, then triangular.
      buffer<std::complex<double>> t;     //!< `T`, upper triangular.
      buffer<std::complex<double>> q;     //!< `Q`, by rows.
      buffer<std::complex<double>> z;     //!< `Z`, by rows.

      //! \brief `S(i, j)`, from 1.
      [[nodiscard]] std::complex<double>& h(std::size_t i,
                                            std::size_t j)
      {
        return s[((i - 1) * n) + (j - 1)];
      }

      //! \brief `T(i, j)`, from 1.
      [[nodiscard]] std::complex<double>& b(std::size_t i,
                                            std::size_t j)
      {
        return t[((i - 1) * n) + (j - 1)];
      }

      //! \brief Rows \p i and \p i + 1 (from 1) of \p m over the columns from \p first (from 1) to \p last.
      static void rotate_rows(buffer<std::complex<double>>& m,
                              std::size_t                   n,
                              std::size_t                   i,
                              std::size_t                   first,
                              std::size_t                   last,
                              double                        c,
                              const std::complex<double>&   sine)
      {
        if (first <= last) {
          rotate_pair(m.data() + ((i - 1) * n) + (first - 1), m.data() + (i * n) + (first - 1), last - first + 1, 1, c, sine);
        }
      }

      //! \brief Columns \p j and \p k (from 1) of \p m over the rows from \p first to \p last (from 1), `first ≤ last`:
      //!        every caller's span holds a row (its `last` a column index of at least 1).
      static void rotate_columns(buffer<std::complex<double>>& m,
                                 std::size_t                   n,
                                 std::size_t                   j,
                                 std::size_t                   k,
                                 std::size_t                   first,
                                 std::size_t                   last,
                                 double                        c,
                                 const std::complex<double>&   sine)
      {
        rotate_pair(m.data() + ((first - 1) * n) + (j - 1), m.data() + ((first - 1) * n) + (k - 1), last - first + 1, n, c, sine);
      }
    };

    /*!
     * \brief \p a and \p b (`n × n`, row-major) to Hessenberg-triangular form, as LAPACK's `zgges` begins: `B = Q R` by
     *        the QR factorisation, `A ← Qᴴ A`, then `zgghrd`: each column of `A` from the first cleared below its
     *        subdiagonal from the bottom by rotations of rows, the fill each leaves below `B`'s diagonal taken out by a
     *        rotation of columns; `Q` and `Z` accumulating them.
     */
    [[nodiscard]] inline qz_form hessenberg_triangular(const matrix<std::complex<double>>& a,
                                                       const matrix<std::complex<double>>& b)
    {
      using C = std::complex<double>;
      const std::size_t                      n      {a.rows()};
      const qr_decomposition<C>              factor {b, qr_mode::complete};
      const matrix<C>                        q      {factor.q()};
      const matrix<C>                        r      {factor.r()};
      qz_form                                form;
      form.n = n;
      form.t = buffer<C>(r.begin(), r.end());
      const matrix<C> s {adjoint_of(q) * a};
      form.s = buffer<C>(s.begin(), s.end());
      form.q = buffer<C>(q.begin(), q.end());
      form.z = buffer<C>(n * n, C {});
      for (std::size_t i {0}; i < n; ++i) {
        form.z[(i * n) + i] = C {1.0};
      }
      for (std::size_t column {1}; column + 2 <= n; ++column) {
        for (std::size_t row {n}; row >= column + 2; --row) {
          // rows row − 1 and row take out A(row, column)
          const plane_rotation<C> g {givens(form.h(row - 1, column), form.h(row, column))};
          form.h(row - 1, column) = g.r;
          form.h(row, column)     = C {};
          qz_form::rotate_rows(form.s, n, row - 1, column + 1, n, g.c, g.s);
          qz_form::rotate_rows(form.t, n, row - 1, row - 1, n, g.c, g.s);
          qz_form::rotate_columns(form.q, n, row - 1, row, 1, n, g.c, std::conj(g.s));
          // columns row and row − 1 take out the fill B(row, row − 1)
          const plane_rotation<C> h {givens(form.b(row, row), form.b(row, row - 1))};
          form.b(row, row)     = h.r;
          form.b(row, row - 1) = C {};
          qz_form::rotate_columns(form.s, n, row, row - 1, 1, n, h.c, h.s);
          qz_form::rotate_columns(form.t, n, row, row - 1, 1, row - 1, h.c, h.s);
          qz_form::rotate_columns(form.z, n, row, row - 1, 1, n, h.c, h.s);
        }
      }
      return form;
    }

    //! \brief `T(j, j)` made real and non-negative, as `zhgeqz` standardises it: row `j` of `T` and `S` above the
    //!        diagonal and column `j` of `S`, `T` and `Z` taking the phase. \return `(α, β)`, the eigenvalue `α/β`.
    inline std::pair<std::complex<double>, std::complex<double>> standardize(qz_form&    f,
                                                                             std::size_t j)
    {
      using C = std::complex<double>;
      const std::size_t n    {f.n};
      const double      size {std::abs(f.b(j, j))};
      if (size > std::numeric_limits<double>::min()) {
        const C sign {std::conj(f.b(j, j) / size)};
        f.b(j, j) = C {size};
        for (std::size_t i {1}; i < j; ++i) {
          f.b(i, j) *= sign;
        }
        for (std::size_t i {1}; i <= j; ++i) {
          f.h(i, j) *= sign;
        }
        for (std::size_t i {1}; i <= n; ++i) {
          f.z[((i - 1) * n) + (j - 1)] *= sign;
        }
      }
      else {
        f.b(j, j) = C {};
      }
      return {f.h(j, j), f.b(j, j)};
    }

    /*!
     * \brief The Hessenberg-triangular \p f to generalized Schur form, as LAPACK's `zhgeqz` (Schur form, `Q` and `Z`
     *        updated): single-shift QZ sweeps on the active block, the shift the eigenvalue of the trailing 2 × 2 of
     *        `S T⁻¹` nearer its corner (an exceptional one every tenth iteration), a negligible subdiagonal entry of `S`
     *        splitting the block, a negligible diagonal entry of `T` chased down (or up, after two small subdiagonal
     *        entries) to split off an infinite eigenvalue. \return the eigenvalues as `(α, β)` pairs, in order.
     *        \p steps bounds the iterations (`zhgeqz`'s `30 n`).
     * \throws convergence_error If the iterations did not converge within \p steps (`zhgeqz`'s `info`).
     */
    [[nodiscard]] inline std::vector<std::pair<std::complex<double>, std::complex<double>>> qz_iteration(qz_form&    f,
                                                                                                         std::size_t steps)
    {
      using C = std::complex<double>;
      const std::size_t            n {f.n};
      std::vector<std::pair<C, C>> values(n);
      if (n == 0) {
        return values;
      }
      const double safmin {std::numeric_limits<double>::min()};
      const double ulp    {std::numeric_limits<double>::epsilon()};
      double       anorm  {0.0};
      double       bnorm  {0.0};
      for (std::size_t i {1}; i <= n; ++i) {
        for (std::size_t j {i > 1 ? i - 1 : 1}; j <= n; ++j) {
          anorm += std::norm(f.h(i, j));
        }
        for (std::size_t j {i}; j <= n; ++j) {
          bnorm += std::norm(f.b(i, j));
        }
      }
      anorm = std::sqrt(anorm);
      bnorm = std::sqrt(bnorm);
      const double atol {std::max(safmin, ulp * anorm)};
      const double btol {std::max(safmin, ulp * bnorm)};
      const double ascale {1.0 / std::max(safmin, anorm)};
      const double bscale {1.0 / std::max(safmin, bnorm)};
      std::size_t  ilast {n};
      std::size_t  ifirst {1};
      std::size_t  iiter {0};
      C            eshift {};
      const auto   rotation_left = [&](std::size_t row, std::size_t first, double c, const C& sine) { // rows row, row + 1 of S and T
                                     qz_form::rotate_rows(f.s, n, row, first, n, c, sine);
                                     qz_form::rotate_rows(f.t, n, row, first, n, c, sine);
                                     qz_form::rotate_columns(f.q, n, row, row + 1, 1, n, c, std::conj(sine));
                                   };
      for (std::size_t jiter {0}; jiter < steps; ++jiter) {
        // the case to take: 0 a QZ step on ifirst…ilast, 1 T(ilast, ilast) zero, 2 a 1 × 1 block split at ilast
        int action {0};
        if (ilast == 1) {
          action = 2;
        }
        else if (modulus_one(f.h(ilast, ilast - 1)) <= std::max(safmin, ulp * (modulus_one(f.h(ilast, ilast)) + modulus_one(f.h(ilast - 1, ilast - 1))))) {
          f.h(ilast, ilast - 1) = C {};
          action                = 2;
        }
        else if (std::abs(f.b(ilast, ilast)) <= btol) {
          f.b(ilast, ilast) = C {};
          action            = 1;
        }
        else {
          for (std::size_t j {ilast - 1};; --j) { // every case ends the search at j = 1 at the latest
            bool ilazro {false};
            if (j == 1) {
              ilazro = true;
            }
            else if (modulus_one(f.h(j, j - 1)) <= std::max(safmin, ulp * (modulus_one(f.h(j, j)) + modulus_one(f.h(j - 1, j - 1))))) {
              f.h(j, j - 1) = C {};
              ilazro        = true;
            }
            if (std::abs(f.b(j, j)) < btol) {
              f.b(j, j) = C {};
              bool ilazr2 {false};
              if (!ilazro && modulus_one(f.h(j, j - 1)) * (ascale * modulus_one(f.h(j + 1, j))) <= modulus_one(f.h(j, j)) * (ascale * atol)) {
                ilazr2 = true;
              }
              if (ilazro || ilazr2) { // split a 1 × 1 block off at the top, the zero of T chased down
                bool settled {false};
                for (std::size_t jch {j}; jch < ilast; ++jch) {
                  const plane_rotation<C> g {givens(f.h(jch, jch), f.h(jch + 1, jch))};
                  f.h(jch, jch)     = g.r;
                  f.h(jch + 1, jch) = C {};
                  qz_form::rotate_rows(f.s, n, jch, jch + 1, n, g.c, g.s);
                  qz_form::rotate_rows(f.t, n, jch, jch + 1, n, g.c, g.s);
                  qz_form::rotate_columns(f.q, n, jch, jch + 1, 1, n, g.c, std::conj(g.s));
                  if (ilazr2) {
                    f.h(jch, jch - 1) *= g.c;
                  }
                  ilazr2 = false;
                  if (modulus_one(f.b(jch + 1, jch + 1)) >= btol) {
                    if (jch + 1 >= ilast) {
                      action = 2;
                    }
                    else {
                      ifirst = jch + 1;
                      action = 0;
                    }
                    settled = true;
                    break;
                  }
                  f.b(jch + 1, jch + 1) = C {};
                }
                if (!settled) {
                  action = 1;
                }
                break;
              }
              // only T(j, j) zero: chase it to T(ilast, ilast)
              for (std::size_t jch {j}; jch < ilast; ++jch) {
                const plane_rotation<C> g {givens(f.b(jch, jch + 1), f.b(jch + 1, jch + 1))};
                f.b(jch, jch + 1)     = g.r;
                f.b(jch + 1, jch + 1) = C {};
                qz_form::rotate_rows(f.t, n, jch, jch + 2, n, g.c, g.s);
                qz_form::rotate_rows(f.s, n, jch, jch - 1, n, g.c, g.s);
                qz_form::rotate_columns(f.q, n, jch, jch + 1, 1, n, g.c, std::conj(g.s));
                const plane_rotation<C> h {givens(f.h(jch + 1, jch), f.h(jch + 1, jch - 1))};
                f.h(jch + 1, jch)     = h.r;
                f.h(jch + 1, jch - 1) = C {};
                qz_form::rotate_columns(f.s, n, jch, jch - 1, 1, jch, h.c, h.s);
                qz_form::rotate_columns(f.t, n, jch, jch - 1, 1, jch - 1, h.c, h.s);
                qz_form::rotate_columns(f.z, n, jch, jch - 1, 1, n, h.c, h.s);
              }
              action = 1;
              break;
            }
            if (ilazro) { // only S's subdivision: a QZ step on j…ilast
              ifirst = j;
              action = 0;
              break;
            }
          }
        }
        if (action == 1) { // T(ilast, ilast) zero: clear S(ilast, ilast − 1) to split a 1 × 1 block
          const plane_rotation<C> g {givens(f.h(ilast, ilast), f.h(ilast, ilast - 1))};
          f.h(ilast, ilast)     = g.r;
          f.h(ilast, ilast - 1) = C {};
          qz_form::rotate_columns(f.s, n, ilast, ilast - 1, 1, ilast - 1, g.c, g.s);
          qz_form::rotate_columns(f.t, n, ilast, ilast - 1, 1, ilast - 1, g.c, g.s);
          qz_form::rotate_columns(f.z, n, ilast, ilast - 1, 1, n, g.c, g.s);
          action = 2;
        }
        if (action == 2) { // a 1 × 1 block: its eigenvalue, then the next block
          values[ilast - 1] = standardize(f, ilast);
          if (ilast == 1) {
            return values;
          }
          --ilast;
          iiter  = 0;
          eshift = C {};
          continue;
        }
        // a QZ step on ifirst…ilast
        ++iiter;
        C shift {};
        if (iiter % 10 != 0) { // the eigenvalue of the trailing 2 × 2 of S T⁻¹ nearer its corner
          const C u12   {(bscale * f.b(ilast - 1, ilast)) / (bscale * f.b(ilast, ilast))};
          const C ad11  {(ascale * f.h(ilast - 1, ilast - 1)) / (bscale * f.b(ilast - 1, ilast - 1))};
          const C ad21  {(ascale * f.h(ilast, ilast - 1)) / (bscale * f.b(ilast - 1, ilast - 1))};
          const C ad12  {(ascale * f.h(ilast - 1, ilast)) / (bscale * f.b(ilast, ilast))};
          const C ad22  {(ascale * f.h(ilast, ilast)) / (bscale * f.b(ilast, ilast))};
          const C abi22 {ad22 - (u12 * ad21)};
          const C abi12 {ad12 - (u12 * ad11)};
          shift = abi22;
          const C ctemp {std::sqrt(abi12) * std::sqrt(ad21)};
          double  temp  {modulus_one(ctemp)};
          if (ctemp != C {}) {
            const C      x {0.5 * (ad11 - shift)};
            const double temp2 {modulus_one(x)};
            temp = std::max(temp, temp2);
            C y {temp * std::sqrt(((x / temp) * (x / temp)) + ((ctemp / temp) * (ctemp / temp)))};
            if (temp2 > 0.0 && ((x / temp2).real() * y.real()) + ((x / temp2).imag() * y.imag()) < 0.0) {
              y = -y;
            }
            shift -= ctemp * (ctemp / (x + y));
          }
        }
        else { // exceptional
          // zhgeqz also asks bscale |T(ilast, ilast)| > safmin here: |T(ilast, ilast)| > btol was found this iteration,
          // so it is above ulp
          if (iiter % 20 == 0) {
            eshift += (ascale * f.h(ilast, ilast)) / (bscale * f.b(ilast, ilast));
          }
          else {
            eshift += (ascale * f.h(ilast, ilast - 1)) / (bscale * f.b(ilast - 1, ilast - 1));
          }
          shift = eshift;
        }
        // two consecutive small subdiagonal entries: start the sweep below them
        std::size_t istart {ifirst};
        C           ctemp  {(ascale * f.h(ifirst, ifirst)) - (shift * (bscale * f.b(ifirst, ifirst)))};
        for (std::size_t j {ilast - 1}; j > ifirst; --j) {
          const C      trial {(ascale * f.h(j, j)) - (shift * (bscale * f.b(j, j)))};
          double       temp {modulus_one(trial)};
          double       temp2 {ascale * modulus_one(f.h(j + 1, j))};
          const double tempr {std::max(temp, temp2)};
          if (tempr < 1.0) { // tempr > 0: the search just found S(j + 1, j) not negligible (zhgeqz also asks it)
            temp  /= tempr;
            temp2 /= tempr;
          }
          if (modulus_one(f.h(j, j - 1)) * temp2 <= temp * atol) {
            istart = j;
            ctemp  = trial;
            break;
          }
        }
        // the implicit-shift sweep
        const plane_rotation<C> first {givens(ctemp, ascale * f.h(istart + 1, istart))};
        double                  c     {first.c};
        C                       sine  {first.s};
        for (std::size_t j {istart}; j < ilast; ++j) {
          if (j > istart) {
            const plane_rotation<C> g {givens(f.h(j, j - 1), f.h(j + 1, j - 1))};
            f.h(j, j - 1)     = g.r;
            f.h(j + 1, j - 1) = C {};
            c                 = g.c;
            sine              = g.s;
          }
          rotation_left(j, j, c, sine);
          const plane_rotation<C> g {givens(f.b(j + 1, j + 1), f.b(j + 1, j))};
          f.b(j + 1, j + 1) = g.r;
          f.b(j + 1, j)     = C {};
          qz_form::rotate_columns(f.s, n, j + 1, j, 1, std::min(j + 2, ilast), g.c, g.s);
          qz_form::rotate_columns(f.t, n, j + 1, j, 1, j, g.c, g.s);
          qz_form::rotate_columns(f.z, n, j + 1, j, 1, n, g.c, g.s);
        }
      }
      throw convergence_error("QZ: the iterations did not converge (zhgeqz's info)");
    }

    //! \brief The Frobenius norm of the 2 × 2 block \p m.
    [[nodiscard]] inline double frobenius_two(const std::array<std::complex<double>, 4>& m)
    {
      return std::hypot(std::hypot(std::abs(m[0]), std::abs(m[1])), std::hypot(std::abs(m[2]), std::abs(m[3])));
    }

    /*!
     * \brief Exchanges the diagonal entries \p j and \p j + 1 (from 1) of the triangular pencil \p f, as LAPACK's
     *        `ztgex2`: columns `j`, `j + 1` rotated onto the right eigenvector of the second eigenvalue, then rows `j`,
     *        `j + 1` back to triangular, the rotation from `S`'s new first column where `|S₂₂ T₁₁| ≥ |S₁₁ T₂₂|`, else `T`'s; `Z`
     *        and `Q` turning with them. The diagonal of `T` is then complex: \ref standardize takes it back. Tried on
     *        the 2 × 2 blocks first and refused, \p f unchanged, unless it passes `ztgex2`'s weak test (the new
     *        subdiagonal entries within `20 ε` of the blocks' norms) and strong test (the blocks rebuilt from the
     *        rotations within the same bounds). \return Whether the exchange was made.
     */
    [[nodiscard]] inline bool exchange_pencil_diagonal(qz_form&    f,
                                                       std::size_t j)
    {
      using C     = std::complex<double>;
      using block = std::array<C, 4>; // (1,1) (1,2) (2,1) (2,2)
      const std::size_t       n {f.n};
      const block             s0 {f.h(j, j), f.h(j, j + 1), f.h(j + 1, j), f.h(j + 1, j + 1)};
      const block             t0 {f.b(j, j), f.b(j, j + 1), f.b(j + 1, j), f.b(j + 1, j + 1)};
      const double            smlnum {std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon()};
      const double            thresha {std::max(20.0 * std::numeric_limits<double>::epsilon() * frobenius_two(s0), smlnum)};
      const double            threshb {std::max(20.0 * std::numeric_limits<double>::epsilon() * frobenius_two(t0), smlnum)};
      const C                 ff {(s0[3] * t0[0]) - (t0[3] * s0[0])};
      const C                 gg {(s0[3] * t0[1]) - (t0[3] * s0[1])};
      const plane_rotation<C> z {givens(gg, ff)};
      const C                 zs {std::conj(-z.s)};
      // a block's columns turned by (c, s): column 1 ← c·column 1 + s·column 2, column 2 ← c·column 2 − s̄·column 1
      const auto columns = [](const block& m, double c, const C& sine) {
                             return block {(c * m[0]) + (sine * m[1]), (c * m[1]) - (std::conj(sine) * m[0]),
                                           (c * m[2]) + (sine * m[3]), (c * m[3]) - (std::conj(sine) * m[2])};
                           };
      // its rows turned by (c, s): row 1 ← c·row 1 + s·row 2, row 2 ← c·row 2 − s̄·row 1
      const auto rows = [](const block& m, double c, const C& sine) {
                          return block {(c * m[0]) + (sine * m[2]), (c * m[1]) + (sine * m[3]),
                                        (c * m[2]) - (std::conj(sine) * m[0]), (c * m[3]) - (std::conj(sine) * m[1])};
                        };
      const block             s1     {columns(s0, z.c, zs)};
      const block             t1     {columns(t0, z.c, zs)};
      const bool              from_s {std::abs(s0[3]) * std::abs(t0[0]) >= std::abs(s0[0]) * std::abs(t0[3])};
      const plane_rotation<C> q      {from_s ? givens(s1[0], s1[2]) : givens(t1[0], t1[2])};
      const block             s2     {rows(s1, q.c, q.s)};
      const block             t2     {rows(t1, q.c, q.s)};
      if (!(std::abs(s2[2]) <= thresha && std::abs(t2[2]) <= threshb)) { // NaN fails, as LAPACK
        return false;
      }
      const auto residue = [&](const block& now, const block& was) {
                             const block back {columns(rows(now, q.c, -q.s), z.c, -zs)};
                             return frobenius_two(block {back[0] - was[0], back[1] - was[1], back[2] - was[2], back[3] - was[3]});
                           };
      if (!(residue(s2, s0) <= thresha && residue(t2, t0) <= threshb)) {
        return false;
      }
      qz_form::rotate_columns(f.s, n, j, j + 1, 1, j + 1, z.c, zs);
      qz_form::rotate_columns(f.t, n, j, j + 1, 1, j + 1, z.c, zs);
      qz_form::rotate_columns(f.z, n, j, j + 1, 1, n, z.c, zs);
      qz_form::rotate_rows(f.s, n, j, j, n, q.c, q.s);
      qz_form::rotate_rows(f.t, n, j, j, n, q.c, q.s);
      qz_form::rotate_columns(f.q, n, j, j + 1, 1, n, q.c, std::conj(q.s));
      f.h(j + 1, j) = C {};
      f.b(j + 1, j) = C {};
      return true;
    }
  } // namespace detail

  /*!
   * \brief The generalized Schur decomposition of a complex pencil `A − λ B`, `A = Q S Zᴴ` and `B = Q T Zᴴ`, as
   *        `scipy.linalg.qz(a, b, output="complex")` (LAPACK's `zgges`): `Q` and `Z` unitary, `S` and `T` upper
   *        triangular, `T`'s diagonal real and non-negative; the generalized eigenvalues `α_k / β_k` on the diagonals
   *        (`β_k = 0` an infinite one). Reordered as `scipy.linalg.ordqz` by \ref sort or \ref reorder.
   */
  class generalized_schur
  {
  public:

    //! \brief The selection `scipy.linalg.ordqz` takes: true for `(α, β)` to lead.
    using selection = std::function<bool (const std::complex<double>&, const std::complex<double>&)>;

    /*!
     * \brief Decomposes the pencil of \p a and \p b.
     * \throws dimension_error   If \p a or \p b is not square or their orders differ.
     * \throws non_finite_error  If an entry is infinite or NaN.
     * \throws convergence_error If the QZ iterations did not converge.
     */
    generalized_schur(const matrix<std::complex<double>>& a,
                      const matrix<std::complex<double>>& b)
      : generalized_schur(a,
                          b,
                          30 * std::max<std::size_t>(a.rows(), 1))
    {}

    //! \brief Decomposes the pencil within \p steps QZ iterations (`zhgeqz`'s bound `30 n` by default).
    generalized_schur(const matrix<std::complex<double>>& a,
                      const matrix<std::complex<double>>& b,
                      std::size_t                         steps)
    {
      const std::size_t n {a.rows()};
      if (a.cols() != n || b.rows() != n || b.cols() != n) {
        throw dimension_error("the generalized Schur decomposition requires two square matrices of one order");
      }
      detail::require_finite(a, "generalized Schur decomposition");
      detail::require_finite(b, "generalized Schur decomposition");
      form_   = detail::hessenberg_triangular(a, b);
      values_ = detail::qz_iteration(form_, steps);
    }

    //! \brief `S`, upper triangular.
    [[nodiscard]] matrix<std::complex<double>> s() const
    {
      return matrix<std::complex<double>> {form_.n, form_.n, form_.s};
    }

    //! \brief `T`, upper triangular, its diagonal real and non-negative.
    [[nodiscard]] matrix<std::complex<double>> t() const
    {
      return matrix<std::complex<double>> {form_.n, form_.n, form_.t};
    }

    //! \brief `Q`, unitary: `A = Q S Zᴴ`.
    [[nodiscard]] matrix<std::complex<double>> q() const
    {
      return matrix<std::complex<double>> {form_.n, form_.n, form_.q};
    }

    //! \brief `Z`, unitary.
    [[nodiscard]] matrix<std::complex<double>> z() const
    {
      return matrix<std::complex<double>> {form_.n, form_.n, form_.z};
    }

    //! \brief The eigenvalues as `(α, β)`, in the diagonals' order: `α = S(k, k)`, `β = T(k, k)`.
    [[nodiscard]] const std::vector<std::pair<std::complex<double>, std::complex<double>>>& values() const noexcept
    {
      return values_;
    }

    /*!
     * \brief Moves the eigenvalues whose entries of \p chosen are set to the leading diagonal places, in their order, as
     *        LAPACK's `ztgsen` (by `ztgexc`'s exchanges), `Q` and `Z` turning with them, `T`'s diagonal made real and
     *        non-negative again. \return The eigenvalues moved.
     * \throws dimension_error   If \p chosen does not hold one entry per eigenvalue.
     * \throws convergence_error If an exchange is refused (`ztgex2`'s tests: two eigenvalues too close to separate);
     *                           the decomposition is then unchanged.
     */
    std::size_t reorder(const std::vector<bool>& chosen)
    {
      const std::size_t n {form_.n};
      if (chosen.size() != n) {
        throw dimension_error("generalized Schur reordering requires one choice per eigenvalue");
      }
      detail::qz_form work  {form_};
      std::size_t     moved {0};
      for (std::size_t k {0}; k < n; ++k) {
        if (!chosen[k]) {
          continue;
        }
        for (std::size_t j {k}; j > moved; --j) { // from place k up to place moved, a swap at a time
          if (!detail::exchange_pencil_diagonal(work, j)) {
            throw convergence_error("generalized Schur form: eigenvalues could not be separated for reordering");
          }
        }
        ++moved;
      }
      for (std::size_t k {1}; k <= n; ++k) {
        values_[k - 1] = detail::standardize(work, k);
      }
      form_ = std::move(work);
      return moved;
    }

    //! \brief The eigenvalues \p select holds moved first (\ref reorder), as `scipy.linalg.ordqz(…, sort=…)` sorts
    //!        them. \return Their count.
    std::size_t sort(const selection& select)
    {
      std::vector<bool> chosen(values_.size());
      for (std::size_t k {0}; k < values_.size(); ++k) {
        chosen[k] = select(values_[k].first, values_[k].second);
      }
      return reorder(chosen);
    }

  private:

    detail::qz_form                                                    form_;   //!< S, T, Q and Z.
    std::vector<std::pair<std::complex<double>, std::complex<double>>> values_; //!< The eigenvalues as (α, β).
  };

  namespace detail {

    /*!
     * \brief The scaling of LAPACK's `dgebal('S')` for the non-negative `n × n` \p m (row-major, scaled in place): rows
     *        and columns by powers of two until their 2-norms are within a factor of 2, a row or column of zeros left
     *        alone, the factors kept within the safe range.
     * \return The scales `D` of `D⁻¹ M D`.
     */
    [[nodiscard]] inline buffer<double> scaling_balance(buffer<double>& m,
                                                        std::size_t     n)
    {
      constexpr double radix  {2.0};
      const double     sfmin1 {std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon()};
      const double     sfmax1 {1.0 / sfmin1};
      const double     sfmin2 {sfmin1 * radix};
      const double     sfmax2 {1.0 / sfmin2};
      buffer<double>   scale(n, 1.0);
      for (bool unbalanced {true}; unbalanced;) {
        unbalanced = false;
        for (std::size_t i {0}; i < n; ++i) {
          double c  {0.0};
          double r  {0.0};
          double ca {0.0};
          double ra {0.0};
          for (std::size_t k {0}; k < n; ++k) {
            c  = std::hypot(c, m[(k * n) + i]);
            r  = std::hypot(r, m[(i * n) + k]);
            ca = std::max(ca, m[(k * n) + i]);
            ra = std::max(ra, m[(i * n) + k]);
          }
          if (c == 0.0 || r == 0.0) {
            continue;
          }
          const double before {c + r};
          double       f      {1.0};
          double       g      {r / radix};
          while (c < g && std::max({f, c, ca}) < sfmax2 && std::min({r, g, ra}) > sfmin2) {
            f  *= radix;
            c  *= radix;
            ca *= radix;
            r  /= radix;
            g  /= radix;
            ra /= radix;
          }
          g = c / radix;
          while (g >= r && std::max(r, ra) < sfmax2 && std::min({f, c, g, ca}) > sfmin2) {
            f  /= radix;
            c  /= radix;
            g  /= radix;
            ca /= radix;
            r  *= radix;
            ra *= radix;
          }
          const bool too_small {f < 1.0 && scale[i] < 1.0 && f * scale[i] <= sfmin1};
          const bool too_large {f > 1.0 && scale[i] > 1.0 && scale[i] >= sfmax1 / f};
          if (c + r >= 0.95 * before || too_small || too_large) {
            continue;
          }
          scale[i]  *= f;
          unbalanced = true;
          for (std::size_t k {0}; k < n; ++k) {
            m[(i * n) + k] /= f;
            m[(k * n) + i] *= f;
          }
        }
      }
      return scale;
    }

    //! \brief The 1-norm of \p a: its largest column sum of moduli.
    [[nodiscard]] inline double norm_one(const matrix<std::complex<double>>& a)
    {
      double largest {0.0};
      for (std::size_t j {0}; j < a.cols(); ++j) {
        double sum {0.0};
        for (std::size_t i {0}; i < a.rows(); ++i) {
          sum += std::abs(a(i, j));
        }
        largest = std::max(largest, sum);
      }
      return largest;
    }

    //! \brief The operands of an algebraic Riccati equation, `e` and `s` absent when null.
    struct riccati_terms
    {
      const matrix<std::complex<double>>& a;
      const matrix<std::complex<double>>& b;
      const matrix<std::complex<double>>& q;
      const matrix<std::complex<double>>& r;
      const matrix<std::complex<double>>* e;
      const matrix<std::complex<double>>* s;
    };

    /*!
     * \brief The extended pencil `(H, J)` of order `2m + n` of SciPy's `solve_continuous_are` (\p continuous) or
     *        `solve_discrete_are`, row-major.
     */
    [[nodiscard]] inline std::pair<buffer<std::complex<double>>, buffer<std::complex<double>>>
    riccati_pencil(const riccati_terms& terms,
                   bool                 continuous)
    {
      using C = std::complex<double>;
      const std::size_t m {terms.a.rows()};
      const std::size_t n {terms.b.cols()};
      const std::size_t order {(2 * m) + n};
      buffer<C>         h(order * order, C {});
      buffer<C>         j(order * order, C {});
      const auto        e_at = [&terms](std::size_t row, std::size_t col) {
                                 return terms.e != nullptr ? (*terms.e)(row, col) : C {row == col ? 1.0 : 0.0};
                               };
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t k {0}; k < m; ++k) {
          h[(i * order) + k]           = terms.a(i, k);
          h[((m + i) * order) + k]     = -terms.q(i, k);
          j[(i * order) + k]           = e_at(i, k);
          h[((m + i) * order) + m + k] = continuous ? -std::conj(terms.a(k, i)) : std::conj(e_at(k, i));
          j[((m + i) * order) + m + k] = continuous ? std::conj(e_at(k, i)) : std::conj(terms.a(k, i));
        }
        for (std::size_t k {0}; k < n; ++k) {
          h[(i * order) + (2 * m) + k] = terms.b(i, k);
          if (continuous) {
            h[(((2 * m) + k) * order) + m + i] = std::conj(terms.b(i, k));
          }
          else {
            j[(((2 * m) + k) * order) + m + i] = -std::conj(terms.b(i, k));
          }
          if (terms.s != nullptr) {
            h[((m + i) * order) + (2 * m) + k] = -(*terms.s)(i, k);
            h[(((2 * m) + k) * order) + i]     = std::conj((*terms.s)(i, k));
          }
        }
      }
      for (std::size_t i {0}; i < n; ++i) {
        for (std::size_t k {0}; k < n; ++k) {
          h[(((2 * m) + i) * order) + (2 * m) + k] = terms.r(i, k);
        }
      }
      return {std::move(h), std::move(j)};
    }

    /*!
     * \brief Balances the pencil `(H, J)` of order `2m + n` as SciPy does: `dgebal('S')` scaling of `|H| + |J|` off the
     *        diagonal, the first two blocks of `m` then given `D` and `D⁻¹` from half the difference of their exponents
     *        (Benner's symplectic constraint), the pencil scaled to `D⁻¹ (H, J) D`.
     * \return The scales `D`, all 1 when the scaling found none.
     */
    [[nodiscard]] inline buffer<double> balance_pencil(buffer<std::complex<double>>& h,
                                                       buffer<std::complex<double>>& j,
                                                       std::size_t                   m,
                                                       std::size_t                   order)
    {
      buffer<double> magnitudes(order * order, 0.0);
      for (std::size_t i {0}; i < order; ++i) {
        for (std::size_t k {0}; k < order; ++k) {
          magnitudes[(i * order) + k] = i == k ? 0.0 : std::abs(h[(i * order) + k]) + std::abs(j[(i * order) + k]);
        }
      }
      const buffer<double> found {scaling_balance(magnitudes, order)};
      buffer<double>       scale(order, 1.0);
      if (std::all_of(found.begin(), found.end(), [](double v) { return std::abs(v - 1.0) <= 1e-8 + 1e-5; })) {
        return scale;
      }
      for (std::size_t i {0}; i < m; ++i) {
        const double half {std::nearbyint((std::log2(found[m + i]) - std::log2(found[i])) / 2.0)};
        scale[i]     = std::exp2(half);
        scale[m + i] = std::exp2(-half);
      }
      std::copy(found.begin() + static_cast<std::ptrdiff_t>(2 * m), found.end(),
                scale.begin() + static_cast<std::ptrdiff_t>(2 * m));
      for (std::size_t i {0}; i < order; ++i) {
        for (std::size_t k {0}; k < order; ++k) {
          h[(i * order) + k] *= scale[i] / scale[k];
          j[(i * order) + k] *= scale[i] / scale[k];
        }
      }
      return scale;
    }

    /*!
     * \brief The stabilising solution of the continuous (\p continuous) or discrete algebraic Riccati equation by
     *        SciPy's algorithm: the extended pencil, balanced unless \p balanced is false, deflated by the QR
     *        factorisation of its last `n` columns, its generalized Schur form sorted with the eigenvalues in the left
     *        half-plane (inside the unit circle) first, and `X = U₁₀ U₀₀⁻¹` from the leading `m` columns of `Z` (with
     *        `E`, of the QR factorisation of `[Ẽ U₀₀; U₁₀]`), returned as `(X + Xᴴ)/2` in the original scaling. `Ẽ` is
     *        `E` as the balanced pencil holds it, `D₁ E D₁⁻¹`: the basis `U` is of the balanced pencil, so `X̃ Ẽ U₀₀ =
     *        U₁₀` for `X̃ = D₁⁻¹ X D₁⁻¹`, and the unscaled `E` there (as SciPy has it) answers a wrong `X` or none.
     * \throws singular_matrix_error If `U₀₀` is numerically singular (the condition of its LU factor `U` past `1/ε`, as
     *                               SciPy tests it): no finite solution. Where `(A, B)` cannot be stabilised, the
     *                               computed `U₀₀` is singular only to rounding, and its condition may stay below
     *                               `1/ε`: the answer is then a huge `X`, as SciPy's.
     * \throws convergence_error     If `U₀₀ᴴ U₁₀` is not Hermitian: the pencil's eigenvalues too close to the imaginary
     *                               axis (the unit circle).
     */
    [[nodiscard]] inline matrix<std::complex<double>> riccati(const riccati_terms& terms,
                                                              bool                 continuous,
                                                              bool                 balanced)
    {
      using C = std::complex<double>;
      const std::size_t m        {terms.a.rows()};
      const std::size_t n        {terms.b.cols()};
      const std::size_t order    {(2 * m) + n};
      auto [h, j] = riccati_pencil(terms, continuous);
      const buffer<double> scale {balanced ? balance_pencil(h, j, m, order) : buffer<double>(order, 1.0)};
      buffer<C>            last(order * n);
      for (std::size_t i {0}; i < order; ++i) {
        std::copy_n(h.data() + (i * order) + (2 * m), n, last.data() + (i * n));
      }
      const matrix<C> deflation       {qr_decomposition<C> {matrix<C> {order, n, std::move(last)}, qr_mode::complete}.q()};
      buffer<C>       kept(order * 2 * m); // the columns past the first n: Q₂
      buffer<C>       h_left(order * 2 * m);
      buffer<C>       j_left(order * 2 * m);
      for (std::size_t t {0}; t < order; ++t) {
        std::copy_n(&deflation(t, n), 2 * m, kept.data() + (t * 2 * m));
        std::copy_n(h.data() + (t * order), 2 * m, h_left.data() + (t * 2 * m));
        std::copy_n(j.data() + (t * order), 2 * m, j_left.data() + (t * 2 * m));
      }
      const matrix<C>   q2     {adjoint_of(matrix<C> {order, 2 * m, std::move(kept)})};
      generalized_schur pencil {q2 * matrix<C> {order, 2 * m, std::move(h_left)}, q2 * matrix<C> {order, 2 * m, std::move(j_left)}};
      if (continuous) {
        pencil.sort([](const C& alpha, const C& beta) { return (alpha * std::conj(beta)).real() < 0.0; });
      }
      else {
        pencil.sort([](const C& alpha, const C& beta) { return std::abs(alpha) < std::abs(beta); });
      }
      matrix<C> u {pencil.z()};
      if (terms.e != nullptr) { // [D₁ E D₁⁻¹ U₀₀; U₁₀], E scaled as the balancing scaled the pencil
        buffer<C> scaled_e(m * m);
        buffer<C> upper(m * m);
        for (std::size_t i {0}; i < m; ++i) {
          for (std::size_t t {0}; t < m; ++t) {
            scaled_e[(i * m) + t] = (*terms.e)(i, t) * (scale[i] / scale[t]);
            upper[(i * m) + t]    = u(i, t);
          }
        }
        const matrix<C> top {matrix<C> {m, m, std::move(scaled_e)} *matrix<C> {m, m, std::move(upper)}};
        buffer<C>       stacked(2 * m * m);
        std::copy(top.begin(), top.end(), stacked.begin());
        for (std::size_t i {0}; i < m; ++i) {
          for (std::size_t k {0}; k < m; ++k) {
            stacked[((m + i) * m) + k] = u(m + i, k);
          }
        }
        u = qr_decomposition<C> {matrix<C> {2 * m, m, std::move(stacked)}, qr_mode::complete}.q();
      }
      buffer<C> u00(m * m);
      buffer<C> u10(m * m);
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t k {0}; k < m; ++k) {
          u00[(i * m) + k] = u(i, k);
          u10[(i * m) + k] = u(m + i, k);
        }
      }
      const matrix<C> top            {m, m, std::move(u00)};
      const matrix<C> bottom         {m, m, std::move(u10)};
      const matrix<C> factors        {lu_decomposition<C> {top}.factors()};
      buffer<C>       upper(m * m, C {});
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t k {i}; k < m; ++k) {
          upper[(i * m) + k] = factors(i, k);
        }
      }
      const vector<double> sigma {singular_values(matrix<C> {m, m, std::move(upper)})};
      if (!(sigma[m - 1] > std::numeric_limits<double>::epsilon() * sigma[0])) { // 1/cond ≤ ε, as SciPy (a zero U too)
        throw singular_matrix_error("Failed to find a finite solution.");
      }
      const matrix<C> x         {adjoint_of(solve(adjoint_of(top), adjoint_of(bottom)))};
      const matrix<C> product   {adjoint_of(top) * bottom};
      const double    tolerance {std::max(std::ldexp(1.0, -43), 0.1 * norm_one(product))};
      if (!(norm_one(product - adjoint_of(product)) <= tolerance)) {
        throw convergence_error(continuous
                                  ? "The associated Hamiltonian pencil has eigenvalues too close to the imaginary axis"
                                  : "The associated symplectic pencil has eigenvalues too close to the unit circle");
      }
      buffer<C> out(m * m);
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t k {0}; k < m; ++k) {
          out[(i * m) + k] = (x(i, k) + std::conj(x(k, i))) * (scale[i] * scale[k] / 2.0);
        }
      }
      return matrix<C> {m, m, std::move(out)};
    }

    /*!
     * \brief The residual `Aᴴ X E + Eᴴ X A − (Eᴴ X B + S) R⁻¹ (Bᴴ X E + Sᴴ) + Q` of the continuous equation at the
     *        Hermitian \p x, and the closed loop `A − B K`, `K = R⁻¹ (Bᴴ X E + Sᴴ)`, into \p closed.
     */
    [[nodiscard]] inline matrix<std::complex<double>> continuous_residual(const riccati_terms&                terms,
                                                                          const matrix<std::complex<double>>& x,
                                                                          matrix<std::complex<double>>&       closed)
    {
      using C = std::complex<double>;
      const matrix<C> xe       {terms.e != nullptr ? x * *terms.e : x};
      matrix<C>       coupling {adjoint_of(terms.b) * xe};
      if (terms.s != nullptr) {
        coupling = coupling + adjoint_of(*terms.s);
      }
      const matrix<C> gain    {solve(terms.r, coupling)}; // r nonsingular: checked as SciPy checks it
      const matrix<C> product {adjoint_of(terms.a) * xe};
      closed = terms.a - (terms.b * gain);
      return ((product + adjoint_of(product)) - (adjoint_of(coupling) * gain)) + terms.q;
    }

    //! \brief Whether every entry of the matrices \p xs is finite.
    [[nodiscard]] inline bool entries_finite(std::initializer_list<const matrix<std::complex<double>>*> xs)
    {
      return std::all_of(xs.begin(), xs.end(), [](const matrix<std::complex<double>>* x) {
                           return std::all_of(x->begin(), x->end(), [](const std::complex<double>& z) { return is_finite_entry(z); });
                         });
    }

    /*!
     * \brief One step of Newton's method on the continuous equation from the Schur method's \p x (Kleinman, 1968), in
     *        its correction form: `A_Kᴴ Δ E + Eᴴ Δ A_K = −Res(X)` for the closed loop `A_K` (\ref continuous_residual),
     *        a Lyapunov equation solved for `Y = Eᴴ Δ E` as `Fᴴ Y + Y F = −Res(X)`, `F = E⁻¹ A_K`; `X + Δ` kept only
     *        where its residual's 1-norm is smaller. The generalized Schur form of a badly scaled equation (a graded
     *        `A`) loses digits that a step from a solution so near the stabilising one gains back.
     */
    [[nodiscard]] inline matrix<std::complex<double>> newton_refined(const riccati_terms&           terms,
                                                                     matrix<std::complex<double>>&& x)
    {
      using C = std::complex<double>;
      matrix<C>       closed;
      const matrix<C> residual {continuous_residual(terms, x, closed)};
      const matrix<C> f        {terms.e != nullptr ? solve(*terms.e, closed) : closed};
      if (!entries_finite({&residual, &f})) { // a term past the doubles (an X near 1e154): no step, the solver refuses it
        return std::move(x);
      }
      const matrix<C> y     {solve_sylvester(adjoint_of(f), f, C {-1.0} *residual)};
      const matrix<C> delta {terms.e != nullptr ? adjoint_of(solve(adjoint_of(*terms.e), adjoint_of(solve(adjoint_of(*terms.e), y))))
                                                : y};
      const std::size_t m   {x.rows()};
      buffer<C>         stepped(m * m);
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t k {0}; k < m; ++k) {
          stepped[(i * m) + k] = x(i, k) + ((delta(i, k) + std::conj(delta(k, i))) / 2.0);
        }
      }
      matrix<C> refined {m, m, std::move(stepped)};
      matrix<C> unused;
      if (norm_one(continuous_residual(terms, refined, unused)) < norm_one(residual)) {
        return refined;
      }
      return std::move(x);
    }
  } // namespace detail

  namespace detail {

    //! \brief \p x itself, for complex operands \p like.
    [[nodiscard]] inline matrix<std::complex<double>> answer_like(matrix<std::complex<double>>&& x,
                                                                  const matrix<std::complex<double>>& /*like*/)
    {
      return std::move(x);
    }

    //! \brief \p x's real part, for real operands \p like: the solution of a real equation is real.
    [[nodiscard]] inline matrix<double> answer_like(const matrix<std::complex<double>>& x,
                                                    const matrix<double>& /*like*/)
    {
      buffer<double> real(x.rows() * x.cols());
      std::transform(x.begin(), x.end(), real.begin(), [](const std::complex<double>& z) { return z.real(); });
      return matrix<double> {x.rows(), x.cols(), std::move(real)};
    }

    /*!
     * \brief Refuses \p x (named \p name in the message) unless Hermitian, as SciPy's `_are_validate_args`:
     *        `‖X − Xᴴ‖₁` within 100 spacings of `‖X‖₁`.
     * \throws std::invalid_argument If it is not.
     */
    inline void require_hermitian(const matrix<std::complex<double>>& x,
                                  const char*                         name)
    {
      const double size {norm_one(x)};
      if (norm_one(x - adjoint_of(x)) > 100.0 * (std::nextafter(size, std::numeric_limits<double>::infinity()) - size)) {
        throw std::invalid_argument(std::string {"Matrix "} + name + " should be symmetric/hermitian.");
      }
    }

    /*!
     * \brief The checks of SciPy's `_are_validate_args`, then \ref riccati on complex operands, the answer real for
     *        real ones.
     */
    template <typename T>
    [[nodiscard]] matrix<T> solve_are(const matrix<T>& a,
                                      const matrix<T>& b,
                                      const matrix<T>& q,
                                      const matrix<T>& r,
                                      const matrix<T>* e,
                                      const matrix<T>* s,
                                      bool             balanced,
                                      bool             continuous)
    {
      const char*       name   {continuous ? "solve_continuous_are" : "solve_discrete_are"};
      const std::size_t m      {a.rows()};
      const std::size_t n      {b.cols()};
      const bool        e_fits {e == nullptr || (e->rows() == m && e->cols() == m)};
      const bool        s_fits {s == nullptr || (s->rows() == m && s->cols() == n)};
      if (m == 0 || n == 0 || a.cols() != m || b.rows() != m || q.rows() != m || q.cols() != m || r.rows() != n
          || r.cols() != n || !e_fits || !s_fits) {
        throw dimension_error(std::string {name}
                              + " requires a square a (m × m, m ≥ 1), b (m × n, n ≥ 1), q (m × m), r (n × n), "
                              "e (m × m) and s (m × n)");
      }
      for (const matrix<T>* operand : {&a, &b, &q, &r, e, s}) {
        if (operand != nullptr) {
          require_finite(*operand, name);
        }
      }
      const auto& ca {as_complex(a)};
      const auto& cb {as_complex(b)};
      const auto& cq {as_complex(q)};
      const auto& cr {as_complex(r)};
      require_hermitian(cq, "q");
      require_hermitian(cr, "r");
      if (continuous) { // R⁻¹ is the equation's: SciPy refuses a numerically singular r
        const vector<double> sigma {singular_values(cr)};
        if (!(sigma[n - 1] >= std::numeric_limits<double>::epsilon() * norm_one(cr)) || sigma[n - 1] == 0.0) {
          throw std::invalid_argument("Matrix r is numerically singular.");
        }
      }
      std::optional<matrix<std::complex<double>>> ce;
      std::optional<matrix<std::complex<double>>> cs;
      if (e != nullptr) {
        ce.emplace(as_complex(*e));
        const vector<double> sigma {singular_values(*ce)};
        if (!(sigma[0] * std::numeric_limits<double>::epsilon() < sigma[m - 1])) {
          throw singular_matrix_error("Matrix e is numerically singular.");
        }
      }
      if (s != nullptr) {
        cs.emplace(as_complex(*s));
      }
      const riccati_terms          terms {ca, cb, cq, cr, ce ? &*ce : nullptr, cs ? &*cs : nullptr};
      matrix<std::complex<double>> x     {riccati(terms, continuous, balanced)};
      if (continuous) {
        x = newton_refined(terms, std::move(x));
      }
      return answer_like(std::move(x), a);
    }
  } // namespace detail

  /*!
   * \brief The stabilising solution `X` of the continuous algebraic Riccati equation `Aᴴ X E + Eᴴ X A − (Eᴴ X B + S)
   *        R⁻¹ (Bᴴ X E + Sᴴ) + Q = 0` (`E = I` and `S = 0` when \p e and \p s are null), as
   *        `scipy.linalg.solve_continuous_are`: the eigenvalues of the extended Hamiltonian pencil in the left
   *        half-plane moved first in its generalized Schur form (\ref generalized_schur), `R` never inverted.
   * \param[in] a        `m × m`.
   * \param[in] b        `m × n`.
   * \param[in] q        `m × m`, Hermitian.
   * \param[in] r        `n × n`, Hermitian.
   * \param[in] e        `m × m`, nonsingular, or null.
   * \param[in] s        `m × n`, or null.
   * \param[in] balanced Whether the pencil is balanced first (SciPy's default).
   * \tparam T `double` or `std::complex<double>`.
   * \throws dimension_error       If an operand's shape differs from those above, or `m` or `n` is 0.
   * \throws non_finite_error      If an entry is infinite or NaN.
   * \throws std::invalid_argument If \p q or \p r is not Hermitian.
   * \throws singular_matrix_error If \p e is numerically singular, or the equation has no finite stabilising
   *                               solution.
   * \throws convergence_error     If the QZ iterations did not converge, or the pencil has eigenvalues too close to
   *                               the imaginary axis.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] matrix<T> solve_continuous_are(const matrix<T>&                       a,
                                               const matrix<T>&                       b,
                                               const matrix<T>&                       q,
                                               const matrix<T>&                       r,
                                               const std::type_identity_t<matrix<T>>* e        = nullptr,
                                               const std::type_identity_t<matrix<T>>* s        = nullptr,
                                               bool                                   balanced = true)
  {
    return detail::solve_are(a, b, q, r, e, s, balanced, true);
  }

  /*!
   * \brief The stabilising solution `X` of the discrete algebraic Riccati equation `Aᴴ X A − Eᴴ X E − (Aᴴ X B + S)
   *        (R + Bᴴ X B)⁻¹ (Bᴴ X A + Sᴴ) + Q = 0` (`E = I` and `S = 0` when \p e and \p s are null), as
   *        `scipy.linalg.solve_discrete_are`: the eigenvalues of the extended symplectic pencil inside the unit circle
   *        moved first in its generalized Schur form (\ref generalized_schur), nothing inverted.
   * \param[in] a        `m × m`.
   * \param[in] b        `m × n`.
   * \param[in] q        `m × m`, Hermitian.
   * \param[in] r        `n × n`, Hermitian.
   * \param[in] e        `m × m`, nonsingular, or null.
   * \param[in] s        `m × n`, or null.
   * \param[in] balanced Whether the pencil is balanced first (SciPy's default).
   * \tparam T `double` or `std::complex<double>`.
   * \throws dimension_error       If an operand's shape differs from those above, or `m` or `n` is 0.
   * \throws non_finite_error      If an entry is infinite or NaN.
   * \throws std::invalid_argument If \p q or \p r is not Hermitian.
   * \throws singular_matrix_error If \p e is numerically singular, or the equation has no finite stabilising
   *                               solution.
   * \throws convergence_error     If the QZ iterations did not converge, or the pencil has eigenvalues too close to
   *                               the unit circle.
   */
  template <typename T>
  requires std::same_as<T, double> || std::same_as<T, std::complex<double>>
  [[nodiscard]] matrix<T> solve_discrete_are(const matrix<T>&                       a,
                                             const matrix<T>&                       b,
                                             const matrix<T>&                       q,
                                             const matrix<T>&                       r,
                                             const std::type_identity_t<matrix<T>>* e        = nullptr,
                                             const std::type_identity_t<matrix<T>>* s        = nullptr,
                                             bool                                   balanced = true)
  {
    return detail::solve_are(a, b, q, r, e, s, balanced, false);
  }
} // namespace numerical

#endif // NUMERICAL_GENERALIZED_HPP
