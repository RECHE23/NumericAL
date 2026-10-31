/*!
 * \file generalized.hpp
 * \brief The generalized Schur decomposition (QZ) of a pencil `A − λ B`, as LAPACK's `zgges` and `dgges`: the operands
 *        scaled into range, the reduction to Hessenberg-triangular form (`zgghrd`, `dgghrd`), the QZ iteration (`zhgeqz`
 *        single-shift, `dhgeqz` double-shift with 2 × 2 blocks for conjugate pairs), and the reordering of the
 *        eigenvalues by exchanges of adjacent blocks (`ztgexc`/`ztgsen`, `dtgexc`/`dtgex2`/`dtgsen`).
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
     *        `A = Q S Zᴴ` and `B = Q T Zᴴ`, `Q` and `Z` unitary (\ref qt, \ref zt).
     */
    struct qz_form
    {
      std::size_t                  n {0}; //!< The order.
      buffer<std::complex<double>> s;     //!< `S`, upper Hessenberg, then triangular.
      buffer<std::complex<double>> t;     //!< `T`, upper triangular.
      buffer<std::complex<double>> qt;    //!< `Qᵀ` by rows: `Q`'s columns, each contiguous for the rotations.
      buffer<std::complex<double>> zt;    //!< `Zᵀ` by rows.

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

      //! \brief Columns \p j and \p k (from 1) of `Q` or `Z` whole, rows \p j and \p k of \p mt, its transpose:
      //!        \ref rotate_columns's arithmetic, each column contiguous.
      static void rotate_vectors(buffer<std::complex<double>>& mt,
                                 std::size_t                   n,
                                 std::size_t                   j,
                                 std::size_t                   k,
                                 double                        c,
                                 const std::complex<double>&   sine)
      {
        rotate_pair(mt.data() + ((j - 1) * n), mt.data() + ((k - 1) * n), n, 1, c, sine);
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
      const matrix<C> s  {adjoint_of(q) * a};
      form.s = buffer<C>(s.begin(), s.end());
      const matrix<C> qt {transpose(q)};
      form.qt = buffer<C>(qt.begin(), qt.end());
      form.zt = buffer<C>(n * n, C {});
      for (std::size_t i {0}; i < n; ++i) {
        form.zt[(i * n) + i] = C {1.0};
      }
      for (std::size_t column {1}; column + 2 <= n; ++column) {
        for (std::size_t row {n}; row >= column + 2; --row) {
          // rows row − 1 and row take out A(row, column)
          const plane_rotation<C> g {givens(form.h(row - 1, column), form.h(row, column))};
          form.h(row - 1, column) = g.r;
          form.h(row, column)     = C {};
          qz_form::rotate_rows(form.s, n, row - 1, column + 1, n, g.c, g.s);
          qz_form::rotate_rows(form.t, n, row - 1, row - 1, n, g.c, g.s);
          qz_form::rotate_vectors(form.qt, n, row - 1, row, g.c, std::conj(g.s));
          // columns row and row − 1 take out the fill B(row, row − 1)
          const plane_rotation<C> h {givens(form.b(row, row), form.b(row, row - 1))};
          form.b(row, row)     = h.r;
          form.b(row, row - 1) = C {};
          qz_form::rotate_columns(form.s, n, row, row - 1, 1, n, h.c, h.s);
          qz_form::rotate_columns(form.t, n, row, row - 1, 1, row - 1, h.c, h.s);
          qz_form::rotate_vectors(form.zt, n, row, row - 1, h.c, h.s);
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
          f.zt[((j - 1) * n) + (i - 1)] *= sign;
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
                                     qz_form::rotate_vectors(f.qt, n, row, row + 1, c, std::conj(sine));
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
                  qz_form::rotate_vectors(f.qt, n, jch, jch + 1, g.c, std::conj(g.s));
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
                qz_form::rotate_vectors(f.qt, n, jch, jch + 1, g.c, std::conj(g.s));
                const plane_rotation<C> h {givens(f.h(jch + 1, jch), f.h(jch + 1, jch - 1))};
                f.h(jch + 1, jch)     = h.r;
                f.h(jch + 1, jch - 1) = C {};
                qz_form::rotate_columns(f.s, n, jch, jch - 1, 1, jch, h.c, h.s);
                qz_form::rotate_columns(f.t, n, jch, jch - 1, 1, jch - 1, h.c, h.s);
                qz_form::rotate_vectors(f.zt, n, jch, jch - 1, h.c, h.s);
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
          qz_form::rotate_vectors(f.zt, n, ilast, ilast - 1, g.c, g.s);
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
          qz_form::rotate_vectors(f.zt, n, j + 1, j, g.c, g.s);
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
      qz_form::rotate_vectors(f.zt, n, j, j + 1, z.c, zs);
      qz_form::rotate_rows(f.s, n, j, j, n, q.c, q.s);
      qz_form::rotate_rows(f.t, n, j, j, n, q.c, q.s);
      qz_form::rotate_vectors(f.qt, n, j, j + 1, q.c, std::conj(q.s));
      f.h(j + 1, j) = C {};
      f.b(j + 1, j) = C {};
      return true;
    }

    /*!
     * \brief A real pencil in generalized real Schur form being formed: `S` and `T` (`n × n`, row-major) with
     *        `A = Q S Zᵀ` and `B = Q T Zᵀ`, `Q` and `Z` orthogonal (\ref qt, \ref zt).
     */
    struct real_qz_form
    {
      std::size_t    n {0}; //!< The order.
      buffer<double> s;     //!< `S`, upper Hessenberg, then quasi-triangular.
      buffer<double> t;     //!< `T`, upper triangular.
      buffer<double> qt;    //!< `Qᵀ` by rows: `Q`'s columns, each contiguous for the rotations and reflections.
      buffer<double> zt;    //!< `Zᵀ` by rows.

      //! \brief `S(i, j)`, from 1.
      [[nodiscard]] double& h(std::size_t i,
                              std::size_t j)
      {
        return s[((i - 1) * n) + (j - 1)];
      }

      //! \brief `T(i, j)`, from 1.
      [[nodiscard]] double& b(std::size_t i,
                              std::size_t j)
      {
        return t[((i - 1) * n) + (j - 1)];
      }

      //! \brief LAPACK's `drot` on rows \p i and \p k (from 1) of \p m over the columns \p first … \p last:
      //!        `x ← c x + s y`, `y ← c y − s x`, `x` row \p i.
      static void rotate_rows(buffer<double>& m,
                              std::size_t     n,
                              std::size_t     i,
                              std::size_t     k,
                              std::size_t     first,
                              std::size_t     last,
                              double          c,
                              double          sine)
      {
        for (std::size_t j {first}; j <= last; ++j) {
          double&      x {m[((i - 1) * n) + (j - 1)]};
          double&      y {m[((k - 1) * n) + (j - 1)]};
          const double t {(c * x) + (sine * y)};
          y = (c * y) - (sine * x);
          x = t;
        }
      }

      //! \brief LAPACK's `drot` on columns \p j and \p k (from 1) of \p m over the rows \p first … \p last:
      //!        `x ← c x + s y`, `y ← c y − s x`, `x` column \p j.
      static void rotate_columns(buffer<double>& m,
                                 std::size_t     n,
                                 std::size_t     j,
                                 std::size_t     k,
                                 std::size_t     first,
                                 std::size_t     last,
                                 double          c,
                                 double          sine)
      {
        for (std::size_t i {first}; i <= last; ++i) {
          double&      x {m[((i - 1) * n) + (j - 1)]};
          double&      y {m[((i - 1) * n) + (k - 1)]};
          const double t {(c * x) + (sine * y)};
          y = (c * y) - (sine * x);
          x = t;
        }
      }
    };

    //! \brief The singular value decomposition of the upper triangular `[f g; 0 h]` (\ref svd_of_triangle).
    struct triangle_svd
    {
      double ssmin {0.0}; //!< The smaller singular value, signed.
      double ssmax {0.0}; //!< The larger, signed.
      double snr   {0.0}; //!< The right rotation's sine.
      double csr   {1.0}; //!< Its cosine.
      double snl   {0.0}; //!< The left rotation's sine.
      double csl   {1.0}; //!< Its cosine.
    };

    /*!
     * \brief The SVD of the upper triangular `[f g; 0 h]`, as LAPACK's `dlasv2`: `[csl snl; −snl csl] [f g; 0 h]
     *        [csr −snr; snr csr] = diag(ssmax, ssmin)`, `|ssmax| ≥ |ssmin|`, each accurate to a few ulps.
     */
    [[nodiscard]] inline triangle_svd svd_of_triangle(double f,
                                                      double g,
                                                      double h)
    {
      double     ft   {f};
      double     fa   {std::abs(ft)};
      double     ht   {h};
      double     ha   {std::abs(h)};
      int        pmax {1};
      const bool swap {ha > fa};
      if (swap) {
        pmax = 3;
        std::swap(ft, ht);
        std::swap(fa, ha);
      }
      const double gt    {g};
      const double ga    {std::abs(gt)};
      double       ssmin {0.0};
      double       ssmax {0.0};
      double       clt   {1.0};
      double       crt   {1.0};
      double       slt   {0.0};
      double       srt   {0.0};
      if (ga == 0.0) {
        ssmin = ha;
        ssmax = fa;
      }
      else {
        bool gasmal {true};
        if (ga > fa) {
          pmax = 2;
          if (fa / ga < std::numeric_limits<double>::epsilon()) {
            gasmal = false;
            ssmax  = ga;
            ssmin  = ha > 1.0 ? fa / (ga / ha) : (fa / ga) * ha;
            clt    = 1.0;
            slt    = ht / gt;
            srt    = 1.0;
            crt    = ft / gt;
          }
        }
        if (gasmal) {
          const double d  {fa - ha};
          double       l  {d == fa ? 1.0 : d / fa};
          const double m  {gt / ft};
          double       t  {2.0 - l};
          const double mm {m * m};
          const double tt {t * t};
          const double s  {std::sqrt(tt + mm)};
          const double r  {l == 0.0 ? std::abs(m) : std::sqrt((l * l) + mm)};
          const double a  {0.5 * (s + r)};
          ssmin = ha / a;
          ssmax = fa * a;
          if (mm == 0.0) {
            t = l == 0.0 ? std::copysign(2.0, ft) * std::copysign(1.0, gt) : (gt / std::copysign(d, ft)) + (m / t);
          }
          else {
            t = ((m / (s + t)) + (m / (r + l))) * (1.0 + a);
          }
          l   = std::sqrt((t * t) + 4.0);
          crt = 2.0 / l;
          srt = t / l;
          clt = (crt + (srt * m)) / a;
          slt = (ht / ft) * srt / a;
        }
      }
      triangle_svd out;
      if (swap) {
        out.csl = srt;
        out.snl = crt;
        out.csr = slt;
        out.snr = clt;
      }
      else {
        out.csl = clt;
        out.snl = slt;
        out.csr = crt;
        out.snr = srt;
      }
      double tsign {1.0};
      if (pmax == 1) {
        tsign = std::copysign(1.0, out.csr) * std::copysign(1.0, out.csl) * std::copysign(1.0, f);
      }
      else if (pmax == 2) {
        tsign = std::copysign(1.0, out.snr) * std::copysign(1.0, out.csl) * std::copysign(1.0, g);
      }
      else {
        tsign = std::copysign(1.0, out.snr) * std::copysign(1.0, out.snl) * std::copysign(1.0, h);
      }
      out.ssmax = std::copysign(ssmax, tsign);
      out.ssmin = std::copysign(ssmin, tsign * std::copysign(1.0, f) * std::copysign(1.0, h));
      return out;
    }

    //! \brief The eigenvalues of a 2 × 2 pencil, scaled (\ref pencil_eigenvalues): `wr1 / scale1` and `wr2 / scale2`, or
    //!        the pair `(wr1 ± i wi) / scale1`.
    struct pencil_pair
    {
      double scale1 {1.0}; //!< The first's scale.
      double scale2 {1.0}; //!< The second's.
      double wr1    {0.0}; //!< The first's scaled real part.
      double wr2    {0.0}; //!< The second's.
      double wi     {0.0}; //!< The pair's scaled imaginary part, non-negative; 0 for real ones.
    };

    //! \brief The bounds of `dlag2`'s `WSIZE` (\ref pencil_wsize), fixed by the pencil's scales.
    struct pencil_bounds
    {
      double floor {0.0}; //!< The divisor's least value.
      double c2    {0.0}; //!< `safmin · max(1, bnorm)`.
      double c3    {0.0}; //!< `bsize · safmin`.
      double c4    {0.0}; //!< The divisor's cap before the eigenvalue's own size.
      double c5    {0.0}; //!< The eigenvalue's least size counted.
    };

    //! \brief `dlag2`'s `WSIZE`: the divisor that keeps an eigenvalue of magnitude \p w in range.
    [[nodiscard]] inline double pencil_wsize(const pencil_bounds& b,
                                             double               w) noexcept
    {
      constexpr double fuzzy1 {1.0 + 1.0e-5};
      return std::max(b.floor, std::max(fuzzy1 * ((w * b.c2) + b.c3), std::min(b.c4, 0.5 * std::max(w, b.c5))));
    }

    /*!
     * \brief The eigenvalues of the pencil `A − w B`, `A` (\p a11 \p a12 / \p a21 \p a22) and the upper triangular
     *        `B` (\p b11 \p b12 / 0 \p b22), as LAPACK's `dlag2`, with scalings that keep `scale · w` from overflowing
     *        or underflowing; a diagonal entry of `B` below \p safmin's square root times its largest perturbed up to it.
     */
    [[nodiscard]] inline pencil_pair pencil_eigenvalues(double a11,
                                                        double a12,
                                                        double a21,
                                                        double a22,
                                                        double b11,
                                                        double b12,
                                                        double b22,
                                                        double safmin)
    {
      const double     rtmin  {std::sqrt(safmin)};
      const double     rtmax  {1.0 / rtmin};
      const double     safmax {1.0 / safmin};
      const double     anorm  {std::max({std::abs(a11) + std::abs(a21), std::abs(a12) + std::abs(a22), safmin})};
      const double     ascale {1.0 / anorm};
      a11 *= ascale;
      a21 *= ascale;
      a12 *= ascale;
      a22 *= ascale;
      const double bmin {rtmin * std::max({std::abs(b11), std::abs(b12), std::abs(b22), rtmin})};
      if (std::abs(b11) < bmin) {
        b11 = std::copysign(bmin, b11);
      }
      if (std::abs(b22) < bmin) {
        b22 = std::copysign(bmin, b22);
      }
      const double bnorm  {std::max({std::abs(b11), std::abs(b12) + std::abs(b22), safmin})};
      const double bsize  {std::max(std::abs(b11), std::abs(b22))};
      const double bscale {1.0 / bsize};
      b11 *= bscale;
      b12 *= bscale;
      b22 *= bscale;
      const double binv11 {1.0 / b11};
      const double binv22 {1.0 / b22};
      const double s1     {a11 * binv11};
      const double s2     {a22 * binv22};
      double       as12   {0.0};
      double       ss     {0.0};
      double       abi22  {0.0};
      double       pp     {0.0};
      double       shift  {0.0};
      if (std::abs(s1) <= std::abs(s2)) {
        as12  = a12 - (s1 * b12);
        const double as22 {a22 - (s1 * b22)};
        ss    = a21 * (binv11 * binv22);
        abi22 = (as22 * binv22) - (ss * b12);
        pp    = 0.5 * abi22;
        shift = s1;
      }
      else {
        as12  = a12 - (s2 * b12);
        const double as11 {a11 - (s2 * b11)};
        ss    = a21 * (binv11 * binv22);
        abi22 = -ss * b12;
        pp    = 0.5 * ((as11 * binv11) + abi22);
        shift = s2;
      }
      const double qq    {ss * as12};
      double       discr {0.0};
      double       r     {0.0};
      if (std::abs(pp * rtmin) >= 1.0) {
        discr = ((rtmin * pp) * (rtmin * pp)) + (qq * safmin);
        r     = std::sqrt(std::abs(discr)) * rtmax;
      }
      else if ((pp * pp) + std::abs(qq) <= safmin) {
        discr = ((rtmax * pp) * (rtmax * pp)) + (qq * safmax);
        r     = std::sqrt(std::abs(discr)) * rtmin;
      }
      else {
        discr = (pp * pp) + qq;
        r     = std::sqrt(std::abs(discr));
      }
      pencil_pair out;
      if (discr >= 0.0) { // dlag2 also takes r = 0, which is discr = 0: no square root above underflows to 0
        const double sum    {pp + std::copysign(r, pp)};
        const double diff   {pp - std::copysign(r, pp)};
        const double wbig   {shift + sum};
        double       wsmall {shift + diff};
        if (0.5 * std::abs(wbig) > std::max(std::abs(wsmall), safmin)) {
          const double wdet {((a11 * a22) - (a12 * a21)) * (binv11 * binv22)};
          wsmall = wdet / wbig;
        }
        if (pp > abi22) {
          out.wr1 = std::min(wbig, wsmall);
          out.wr2 = std::max(wbig, wsmall);
        }
        else {
          out.wr1 = std::max(wbig, wsmall);
          out.wr2 = std::min(wbig, wsmall);
        }
        out.wi = 0.0;
      }
      else {
        out.wr1 = shift + pp;
        out.wr2 = out.wr1;
        out.wi  = r;
      }
      const double        c1 {bsize * (safmin * std::max(1.0, ascale))};
      const double        c2 {safmin * std::max(1.0, bnorm)};
      const double        c3     {bsize * safmin};
      const double        c4     {ascale <= 1.0 && bsize <= 1.0 ? std::min(1.0, (ascale / safmin) * bsize) : 1.0};
      const double        c5     {ascale <= 1.0 || bsize <= 1.0 ? std::min(1.0, ascale * bsize) : 1.0};
      const pencil_bounds bounds {.floor = std::max(safmin, c1), .c2 = c2, .c3 = c3, .c4 = c4, .c5 = c5};
      const double        wabs   {std::abs(out.wr1) + std::abs(out.wi)};
      double              wsize  {pencil_wsize(bounds, wabs)};
      if (wsize != 1.0) {
        const double wscale {1.0 / wsize};
        out.scale1 = wsize > 1.0 ? (std::max(ascale, bsize) * wscale) * std::min(ascale, bsize) : (std::min(ascale, bsize) * wscale) * std::max(ascale, bsize);
        out.wr1   *= wscale;
        if (out.wi != 0.0) {
          out.wi     *= wscale;
          out.wr2     = out.wr1;
          out.scale2  = out.scale1;
        }
      }
      else {
        out.scale1 = ascale * bsize;
        out.scale2 = out.scale1;
      }
      if (out.wi == 0.0) {
        wsize = pencil_wsize(bounds, std::abs(out.wr2));
        if (wsize != 1.0) {
          const double wscale {1.0 / wsize};
          out.scale2 = wsize > 1.0 ? (std::max(ascale, bsize) * wscale) * std::min(ascale, bsize) : (std::min(ascale, bsize) * wscale) * std::max(ascale, bsize);
          out.wr2   *= wscale;
        }
        else {
          out.scale2 = ascale * bsize;
        }
      }
      return out;
    }

    //! \brief A generalized eigenvalue `(αr + i αi) / β` of a real pencil, as LAPACK's `dgges` gives it.
    struct real_pencil_value
    {
      double alphar {0.0}; //!< The real part of `α`.
      double alphai {0.0}; //!< Its imaginary part: of opposite signs within a conjugate pair, the first positive.
      double beta   {0.0}; //!< `β ≥ 0`, 0 for an infinite eigenvalue.
    };

    //! \brief `√(x² + y² + z²)` without overflow, LAPACK's `dlapy3`.
    [[nodiscard]] inline double hypot3(double x,
                                       double y,
                                       double z)
    {
      return std::hypot(std::hypot(x, y), z);
    }

    /*!
     * \brief The conjugate pair of the standardized 2 × 2 block ending at \p ilast of \p f (`T` diagonal, \p b11 and
     *        \p b22), as `dhgeqz` computes `α` and `β` from \p pair (`dlag2`'s): `β` the norms of the diagonal of `T`
     *        after the complex rotations that would triangularise the block, `α = (wr ± i wi) β / scale`.
     */
    [[nodiscard]] inline std::pair<real_pencil_value, real_pencil_value> pair_values(real_qz_form&      f,
                                                                                     std::size_t        ilast,
                                                                                     const pencil_pair& pair,
                                                                                     double             b11,
                                                                                     double             b22)
    {
      const double safmin {std::numeric_limits<double>::min()};
      const double s1     {pair.scale1};
      const double wr     {pair.wr1};
      const double wi     {pair.wi};
      const double s1inv  {1.0 / s1};
      const double a11    {f.h(ilast - 1, ilast - 1)};
      const double a21    {f.h(ilast, ilast - 1)};
      const double a12    {f.h(ilast - 1, ilast)};
      const double a22    {f.h(ilast, ilast)};
      const double c11r   {(s1 * a11) - (wr * b11)};
      const double c11i   {-wi * b11};
      const double c12    {s1 * a12};
      const double c21    {s1 * a21};
      const double c22r   {(s1 * a22) - (wr * b22)};
      const double c22i   {-wi * b22};
      double       cz     {0.0};
      double       szr    {0.0};
      double       szi    {0.0};
      if (std::abs(c11r) + std::abs(c11i) + std::abs(c12) > std::abs(c21) + std::abs(c22r) + std::abs(c22i)) {
        const double t1 {hypot3(c12, c11r, c11i)};
        cz  = c12 / t1;
        szr = -c11r / t1;
        szi = -c11i / t1;
      }
      else {
        cz = std::hypot(c22r, c22i);
        if (cz <= safmin) {
          cz  = 0.0;
          szr = 1.0;
          szi = 0.0;
        }
        else {
          const double tempr {c22r / cz};
          const double tempi {c22i / cz};
          const double t1    {std::hypot(cz, c21)};
          cz /= t1;
          szr = -c21 * tempr / t1;
          szi = c21 * tempi / t1;
        }
      }
      const double an   {std::abs(a11) + std::abs(a12) + std::abs(a21) + std::abs(a22)};
      const double bn   {std::abs(b11) + std::abs(b22)};
      const double wabs {std::abs(wr) + std::abs(wi)};
      double       cq   {0.0};
      double       sqr  {0.0};
      double       sqi  {0.0};
      if (s1 * an > wabs * bn) {
        cq  = cz * b11;
        sqr = szr * b22;
        sqi = -szi * b22;
      }
      else {
        const double a1r {(cz * a11) + (szr * a12)};
        const double a1i {szi * a12};
        const double a2r {(cz * a21) + (szr * a22)};
        const double a2i {szi * a22};
        cq = std::hypot(a1r, a1i);
        if (cq <= safmin) {
          cq  = 0.0;
          sqr = 1.0;
          sqi = 0.0;
        }
        else {
          const double tempr {a1r / cq};
          const double tempi {a1i / cq};
          sqr = (tempr * a2r) + (tempi * a2i);
          sqi = (tempi * a2r) - (tempr * a2i);
        }
      }
      const double t1 {hypot3(cq, sqr, sqi)};
      cq  /= t1;
      sqr /= t1;
      sqi /= t1;
      const double tempr {(sqr * szr) - (sqi * szi)};
      const double tempi {(sqr * szi) + (sqi * szr)};
      const double b1a   {std::hypot((cq * cz * b11) + (tempr * b22), tempi * b22)};
      const double b2a   {std::hypot((cq * cz * b22) + (tempr * b11), -tempi * b11)};
      return {real_pencil_value {(wr * b1a) * s1inv, (wi * b1a) * s1inv, b1a}, real_pencil_value {(wr * b2a) * s1inv, -(wi * b2a) * s1inv, b2a}};
    }

    /*!
     * \brief One double-shift QZ sweep (Moler and Stewart) on rows and columns \p ifirst … \p ilast of \p f, as
     *        `dhgeqz`'s: the shifts the eigenvalues of the trailing 2 × 2 of `S T⁻¹` and its leading one's first column,
     *        the bulge chased by reflections of three rows (`dlarfg`) and of three columns (from the null vector of
     *        `T`'s 2 × 2 below the bulge), a rotation pair for the last step.
     */
    inline void double_shift_sweep(real_qz_form& f,
                                   std::size_t   ifirst,
                                   std::size_t   ilast,
                                   double        ascale,
                                   double        bscale,
                                   double        safmin)
    {
      using form = real_qz_form;
      const std::size_t n     {f.n};
      const double      ad11  {(ascale * f.h(ilast - 1, ilast - 1)) / (bscale * f.b(ilast - 1, ilast - 1))};
      const double      ad21  {(ascale * f.h(ilast, ilast - 1)) / (bscale * f.b(ilast - 1, ilast - 1))};
      const double      ad12  {(ascale * f.h(ilast - 1, ilast)) / (bscale * f.b(ilast, ilast))};
      const double      ad22  {(ascale * f.h(ilast, ilast)) / (bscale * f.b(ilast, ilast))};
      const double      u12   {f.b(ilast - 1, ilast) / f.b(ilast, ilast)};
      const double      ad11l {(ascale * f.h(ifirst, ifirst)) / (bscale * f.b(ifirst, ifirst))};
      const double      ad21l {(ascale * f.h(ifirst + 1, ifirst)) / (bscale * f.b(ifirst, ifirst))};
      const double      ad12l {(ascale * f.h(ifirst, ifirst + 1)) / (bscale * f.b(ifirst + 1, ifirst + 1))};
      const double      ad22l {(ascale * f.h(ifirst + 1, ifirst + 1)) / (bscale * f.b(ifirst + 1, ifirst + 1))};
      const double      ad32l {(ascale * f.h(ifirst + 2, ifirst + 1)) / (bscale * f.b(ifirst + 1, ifirst + 1))};
      const double      u12l  {f.b(ifirst, ifirst + 1) / f.b(ifirst + 1, ifirst + 1)};
      double            v[3]  {((ad11 - ad11l) * (ad22 - ad11l)) - (ad12 * ad21) + (ad21 * u12 * ad11l) + ((ad12l - (ad11l * u12l)) * ad21l),
                               ((ad22l - ad11l) - (ad21l * u12l) - (ad11 - ad11l) - (ad22 - ad11l) + (ad21 * u12)) * ad21l, ad32l * ad21l};
      double            tau   {small_reflector(v, 3)};
      v[0] = 1.0;
      for (std::size_t j {ifirst}; j + 2 <= ilast; ++j) {
        if (j > ifirst) {
          v[0]              = f.h(j, j - 1);
          v[1]              = f.h(j + 1, j - 1);
          v[2]              = f.h(j + 2, j - 1);
          tau               = small_reflector(v, 3);
          f.h(j, j - 1)     = v[0];
          v[0]              = 1.0;
          f.h(j + 1, j - 1) = 0.0;
          f.h(j + 2, j - 1) = 0.0;
        }
        const double t2 {tau * v[1]};
        const double t3 {tau * v[2]};
        for (std::size_t jc {j}; jc <= n; ++jc) { // the reflection of rows j … j + 2 of S and T
          const double temp {f.h(j, jc) + (v[1] * f.h(j + 1, jc)) + (v[2] * f.h(j + 2, jc))};
          f.h(j, jc)     -= temp * tau;
          f.h(j + 1, jc) -= temp * t2;
          f.h(j + 2, jc) -= temp * t3;
          const double temp2 {f.b(j, jc) + (v[1] * f.b(j + 1, jc)) + (v[2] * f.b(j + 2, jc))};
          f.b(j, jc)     -= temp2 * tau;
          f.b(j + 1, jc) -= temp2 * t2;
          f.b(j + 2, jc) -= temp2 * t3;
        }
        double* const q0 {f.qt.data() + ((j - 1) * n)}; // Q's columns j … j + 2, rows of Qᵀ
        double* const q1 {q0 + n};
        double* const q2 {q1 + n};
        for (std::size_t r {0}; r < n; ++r) {
          const double temp {q0[r] + (v[1] * q1[r]) + (v[2] * q2[r])};
          q0[r] -= temp * tau;
          q1[r] -= temp * t2;
          q2[r] -= temp * t3;
        }
        // the reflection of columns j … j + 2 that takes T(j + 1 … j + 2, j) out: from the null vector of T's 2 × 2 below
        bool         ilpivt {false};
        double       scale  {1.0};
        double       u1     {1.0};
        double       u2     {0.0};
        const double top    {std::max(std::abs(f.b(j + 1, j + 1)), std::abs(f.b(j + 1, j + 2)))};
        const double bottom {std::max(std::abs(f.b(j + 2, j + 1)), std::abs(f.b(j + 2, j + 2)))};
        if (std::max(top, bottom) < safmin) {
          scale = 0.0;
          u1    = 1.0;
          u2    = 0.0;
        }
        else {
          double w11 {0.0};
          double w21 {0.0};
          double w12 {0.0};
          double w22 {0.0};
          if (top >= bottom) {
            w11 = f.b(j + 1, j + 1);
            w21 = f.b(j + 2, j + 1);
            w12 = f.b(j + 1, j + 2);
            w22 = f.b(j + 2, j + 2);
            u1  = f.b(j + 1, j);
            u2  = f.b(j + 2, j);
          }
          else {
            w21 = f.b(j + 1, j + 1);
            w11 = f.b(j + 2, j + 1);
            w22 = f.b(j + 1, j + 2);
            w12 = f.b(j + 2, j + 2);
            u2  = f.b(j + 1, j);
            u1  = f.b(j + 2, j);
          }
          if (std::abs(w12) > std::abs(w11)) {
            ilpivt = true;
            std::swap(w11, w12);
            std::swap(w21, w22);
          }
          const double ratio {w21 / w11};
          u2  -= ratio * u1;
          w22 -= ratio * w12;
          if (std::abs(w22) < safmin) {
            scale = 0.0;
            u2    = 1.0;
            u1    = -w12 / w11;
          }
          else {
            if (std::abs(w22) < std::abs(u2)) {
              scale = std::abs(w22 / u2);
            }
            if (std::abs(w11) < std::abs(u1)) {
              scale = std::min(scale, std::abs(w11 / u1));
            }
            u2 = (scale * u2) / w22;
            u1 = ((scale * u1) - (w12 * u2)) / w11;
          }
        }
        if (ilpivt) {
          std::swap(u1, u2);
        }
        const double t1  {std::sqrt((scale * scale) + (u1 * u1) + (u2 * u2))};
        const double tau2 {1.0 + (scale / t1)};
        const double vs  {-1.0 / (scale + t1)};
        const double w2  {vs * u1};
        const double w3  {vs * u2};
        const double c2  {tau2 * w2};
        const double c3  {tau2 * w3};
        const auto   reflect_columns = [&](buffer<double>& m, std::size_t first, std::size_t last) {
                                         for (std::size_t r {first}; r <= last; ++r) {
                                           double* const row  {m.data() + ((r - 1) * n) + (j - 1)};
                                           const double  temp {row[0] + (w2 * row[1]) + (w3 * row[2])};
                                           row[0] -= temp * tau2;
                                           row[1] -= temp * c2;
                                           row[2] -= temp * c3;
                                         }
                                       };
        reflect_columns(f.s, 1, std::min(j + 3, ilast));
        reflect_columns(f.t, 1, j + 2);
        double* const z0 {f.zt.data() + ((j - 1) * n)}; // Z's columns j … j + 2, rows of Zᵀ
        double* const z1 {z0 + n};
        double* const z2 {z1 + n};
        for (std::size_t r {0}; r < n; ++r) {
          const double temp {z0[r] + (w2 * z1[r]) + (w3 * z2[r])};
          z0[r] -= temp * tau2;
          z1[r] -= temp * c2;
          z2[r] -= temp * c3;
        }
        f.b(j + 1, j) = 0.0;
        f.b(j + 2, j) = 0.0;
      }
      // the last step: rotations of rows ilast − 1, ilast and of their columns
      const std::size_t            j {ilast - 1};
      const plane_rotation<double> g {givens(f.h(j, j - 1), f.h(j + 1, j - 1))};
      f.h(j, j - 1)     = g.r;
      f.h(j + 1, j - 1) = 0.0;
      form::rotate_rows(f.s, n, j, j + 1, j, n, g.c, g.s);
      form::rotate_rows(f.t, n, j, j + 1, j, n, g.c, g.s);
      form::rotate_rows(f.qt, n, j, j + 1, 1, n, g.c, g.s);
      const plane_rotation<double> h {givens(f.b(j + 1, j + 1), f.b(j + 1, j))};
      f.b(j + 1, j + 1) = h.r;
      f.b(j + 1, j)     = 0.0;
      form::rotate_columns(f.s, n, j + 1, j, 1, ilast, h.c, h.s);
      form::rotate_columns(f.t, n, j + 1, j, 1, ilast - 1, h.c, h.s);
      form::rotate_rows(f.zt, n, j + 1, j, 1, n, h.c, h.s);
    }

    /*!
     * \brief The real \p a and \p b to Hessenberg-triangular form, as LAPACK's `dgges` begins: `B = Q R` by the QR
     *        factorisation, `A ← Qᵀ A`, then `dgghrd`: each column of `A` from the first cleared below its subdiagonal
     *        from the bottom by rotations of rows, the fill each leaves below `B`'s diagonal taken out by a rotation of
     *        columns; `Q` and `Z` accumulating them.
     */
    [[nodiscard]] inline real_qz_form real_hessenberg_triangular(const matrix<double>& a,
                                                                 const matrix<double>& b)
    {
      const std::size_t              n      {a.rows()};
      const qr_decomposition<double> factor {b, qr_mode::complete};
      const matrix<double>           q      {factor.q()};
      const matrix<double>           r      {factor.r()};
      real_qz_form                   form;
      form.n = n;
      form.t = buffer<double>(r.begin(), r.end());
      const matrix<double> s  {transpose(q) * a};
      form.s = buffer<double>(s.begin(), s.end());
      const matrix<double> qt {transpose(q)};
      form.qt = buffer<double>(qt.begin(), qt.end());
      form.zt = buffer<double>(n * n, 0.0);
      for (std::size_t i {0}; i < n; ++i) {
        form.zt[(i * n) + i] = 1.0;
        for (std::size_t j {0}; j < i; ++j) { // R's zeros below its diagonal, as the factorisation leaves them
          form.t[(i * n) + j] = 0.0;
        }
      }
      for (std::size_t column {1}; column + 2 <= n; ++column) {
        for (std::size_t row {n}; row >= column + 2; --row) {
          // rows row − 1 and row take out A(row, column)
          const plane_rotation<double> g {givens(form.h(row - 1, column), form.h(row, column))};
          form.h(row - 1, column) = g.r;
          form.h(row, column)     = 0.0;
          real_qz_form::rotate_rows(form.s, n, row - 1, row, column + 1, n, g.c, g.s);
          real_qz_form::rotate_rows(form.t, n, row - 1, row, row - 1, n, g.c, g.s);
          real_qz_form::rotate_rows(form.qt, n, row - 1, row, 1, n, g.c, g.s);
          // columns row and row − 1 take out the fill B(row, row − 1)
          const plane_rotation<double> h {givens(form.b(row, row), form.b(row, row - 1))};
          form.b(row, row)     = h.r;
          form.b(row, row - 1) = 0.0;
          real_qz_form::rotate_columns(form.s, n, row, row - 1, 1, n, h.c, h.s);
          real_qz_form::rotate_columns(form.t, n, row, row - 1, 1, row - 1, h.c, h.s);
          real_qz_form::rotate_rows(form.zt, n, row, row - 1, 1, n, h.c, h.s);
        }
      }
      return form;
    }

    /*!
     * \brief The Hessenberg-triangular \p f to generalized real Schur form, as LAPACK's `dhgeqz` (Schur form, `Q` and
     *        `Z` updated): double-shift QZ sweeps (Moler and Stewart) on the active block, the shifts the eigenvalues of
     *        its trailing 2 × 2 (`dlag2`; a single shift where they are real, an exceptional one every tenth
     *        iteration), a negligible subdiagonal entry of `S` splitting it, a negligible diagonal entry of `T` chased
     *        to split off an infinite eigenvalue; a 2 × 2 block of a conjugate pair standardized with `T` diagonal and
     *        positive (`dlasv2`), every `T(j, j)` non-negative. \p steps bounds the iterations (`dhgeqz`'s `30 n`).
     * \return The eigenvalues, in the diagonal's order.
     * \throws convergence_error If the iterations did not converge within \p steps (`dhgeqz`'s `info`).
     */
    [[nodiscard]] inline std::vector<real_pencil_value> real_qz_iteration(real_qz_form& f,
                                                                          std::size_t   steps)
    {
      using form = real_qz_form;
      const std::size_t              n {f.n};
      std::vector<real_pencil_value> values(n);
      if (n == 0) {
        return values;
      }
      constexpr double safety {1.0e2};
      const double     safmin {std::numeric_limits<double>::min()};
      const double     safmax {1.0 / safmin};
      const double     ulp    {std::numeric_limits<double>::epsilon()};
      double           anorm  {0.0};
      double           bnorm  {0.0};
      for (std::size_t i {1}; i <= n; ++i) { // dlanhs('F'): the Hessenberg part of each
        for (std::size_t j {i > 1 ? i - 1 : 1}; j <= n; ++j) {
          anorm = std::hypot(anorm, f.h(i, j));
          bnorm = std::hypot(bnorm, f.b(i, j));
        }
      }
      const double atol   {std::max(safmin, ulp * anorm)};
      const double btol   {std::max(safmin, ulp * bnorm)};
      const double ascale {1.0 / std::max(safmin, anorm)};
      const double bscale {1.0 / std::max(safmin, bnorm)};
      const double maxit  {static_cast<double>(steps)};
      std::size_t  ilast  {n};
      std::size_t  ifirst {1};
      std::size_t  iiter  {0};
      double       eshift {0.0};
      // a 1 × 1 block at ilast: T(ilast, ilast) made non-negative, the eigenvalue recorded
      const auto   one_by_one = [&](std::size_t j) {
                                  if (f.b(j, j) < 0.0) {
                                    for (std::size_t i {1}; i <= j; ++i) {
                                      f.h(i, j) = -f.h(i, j);
                                      f.b(i, j) = -f.b(i, j);
                                    }
                                    for (std::size_t i {0}; i < n; ++i) {
                                      f.zt[((j - 1) * n) + i] = -f.zt[((j - 1) * n) + i];
                                    }
                                  }
                                  values[j - 1] = {f.h(j, j), 0.0, f.b(j, j)};
                                };
      const auto   negligible_below = [&](std::size_t j) { // S(j, j − 1) against its diagonal neighbours
                                        return std::abs(f.h(j, j - 1)) <= std::max(safmin, ulp * (std::abs(f.h(j, j)) + std::abs(f.h(j - 1, j - 1))));
                                      };
      for (std::size_t jiter {0}; jiter < steps; ++jiter) {
        // 0 a QZ step on ifirst…ilast; 1 T(ilast, ilast) zero (label 70); 2 a 1 × 1 block at ilast (label 80)
        int action {0};
        if (ilast == 1) {
          action = 2;
        }
        else if (negligible_below(ilast)) {
          f.h(ilast, ilast - 1) = 0.0;
          action                = 2;
        }
        else if (std::abs(f.b(ilast, ilast)) <= btol) {
          f.b(ilast, ilast) = 0.0;
          action            = 1;
        }
        else {
          for (std::size_t j {ilast - 1};; --j) { // every case ends the search at j = 1 at the latest
            bool ilazro {false};
            if (j == 1) {
              ilazro = true;
            }
            else if (negligible_below(j)) {
              f.h(j, j - 1) = 0.0;
              ilazro        = true;
            }
            if (std::abs(f.b(j, j)) < btol) {
              f.b(j, j) = 0.0;
              bool ilazr2 {false};
              if (!ilazro) {
                double       temp  {std::abs(f.h(j, j - 1))};
                double       temp2 {std::abs(f.h(j, j))};
                const double tempr {std::max(temp, temp2)};
                if (tempr < 1.0) { // tempr > 0: S(j, j − 1) was found not negligible (dhgeqz also asks it)
                  temp  /= tempr;
                  temp2 /= tempr;
                }
                ilazr2 = temp * (ascale * std::abs(f.h(j + 1, j))) <= temp2 * (ascale * atol);
              }
              if (ilazro || ilazr2) { // split a 1 × 1 block off at the top, the zero of T chased down
                bool settled {false};
                for (std::size_t jch {j}; jch < ilast; ++jch) {
                  const plane_rotation<double> g {givens(f.h(jch, jch), f.h(jch + 1, jch))};
                  f.h(jch, jch)     = g.r;
                  f.h(jch + 1, jch) = 0.0;
                  form::rotate_rows(f.s, n, jch, jch + 1, jch + 1, n, g.c, g.s);
                  form::rotate_rows(f.t, n, jch, jch + 1, jch + 1, n, g.c, g.s);
                  form::rotate_rows(f.qt, n, jch, jch + 1, 1, n, g.c, g.s);
                  if (ilazr2) {
                    f.h(jch, jch - 1) *= g.c;
                  }
                  ilazr2 = false;
                  if (std::abs(f.b(jch + 1, jch + 1)) >= btol) {
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
                  f.b(jch + 1, jch + 1) = 0.0;
                }
                if (!settled) {
                  action = 1;
                }
                break;
              }
              // only T(j, j) zero: chase it to T(ilast, ilast)
              for (std::size_t jch {j}; jch < ilast; ++jch) {
                const plane_rotation<double> g {givens(f.b(jch, jch + 1), f.b(jch + 1, jch + 1))};
                f.b(jch, jch + 1)     = g.r;
                f.b(jch + 1, jch + 1) = 0.0;
                if (jch + 1 < n) {
                  form::rotate_rows(f.t, n, jch, jch + 1, jch + 2, n, g.c, g.s);
                }
                form::rotate_rows(f.s, n, jch, jch + 1, jch - 1, n, g.c, g.s);
                form::rotate_rows(f.qt, n, jch, jch + 1, 1, n, g.c, g.s);
                const plane_rotation<double> h {givens(f.h(jch + 1, jch), f.h(jch + 1, jch - 1))};
                f.h(jch + 1, jch)     = h.r;
                f.h(jch + 1, jch - 1) = 0.0;
                form::rotate_columns(f.s, n, jch, jch - 1, 1, jch, h.c, h.s);
                form::rotate_columns(f.t, n, jch, jch - 1, 1, jch - 1, h.c, h.s); // jch ≥ 2: j = 1 sets ilazro
                form::rotate_rows(f.zt, n, jch, jch - 1, 1, n, h.c, h.s);
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
          const plane_rotation<double> g {givens(f.h(ilast, ilast), f.h(ilast, ilast - 1))};
          f.h(ilast, ilast)     = g.r;
          f.h(ilast, ilast - 1) = 0.0;
          form::rotate_columns(f.s, n, ilast, ilast - 1, 1, ilast - 1, g.c, g.s);
          form::rotate_columns(f.t, n, ilast, ilast - 1, 1, ilast - 1, g.c, g.s);
          form::rotate_rows(f.zt, n, ilast, ilast - 1, 1, n, g.c, g.s);
          action = 2;
        }
        if (action == 2) {
          one_by_one(ilast);
          if (ilast == 1) {
            return values;
          }
          --ilast;
          iiter  = 0;
          eshift = 0.0;
          continue;
        }
        // a QZ step on ifirst…ilast (label 110)
        ++iiter;
        double s1             {1.0};
        double wr             {0.0};
        bool   complex_shifts {false};
        if (iiter % 10 == 0) { // exceptional
          if ((maxit * safmin) * std::abs(f.h(ilast, ilast - 1)) < std::abs(f.b(ilast - 1, ilast - 1))) {
            eshift = f.h(ilast, ilast - 1) / f.b(ilast - 1, ilast - 1);
          }
          else {
            eshift += 1.0 / (safmin * maxit);
          }
          s1 = 1.0;
          wr = eshift;
        }
        else {
          pencil_pair shifts {pencil_eigenvalues(f.h(ilast - 1, ilast - 1), f.h(ilast - 1, ilast), f.h(ilast, ilast - 1), f.h(ilast, ilast),
                                                 f.b(ilast - 1, ilast - 1), f.b(ilast - 1, ilast), f.b(ilast, ilast), safmin * safety)};
          if (std::abs(((shifts.wr1 / shifts.scale1) * f.b(ilast, ilast)) - f.h(ilast, ilast))
              > std::abs(((shifts.wr2 / shifts.scale2) * f.b(ilast, ilast)) - f.h(ilast, ilast))) {
            std::swap(shifts.wr1, shifts.wr2);
            std::swap(shifts.scale1, shifts.scale2);
          }
          s1             = shifts.scale1;
          wr             = shifts.wr1;
          complex_shifts = shifts.wi != 0.0;
        }
        if (!complex_shifts) { // a single real shift: the implicit sweep of rotations
          double       temp  {std::min(ascale, 1.0) * (0.5 * safmax)};
          double       scale {s1 > temp ? temp / s1 : 1.0};
          temp = std::min(bscale, 1.0) * (0.5 * safmax);
          if (std::abs(wr) > temp) {
            scale = std::min(scale, temp / std::abs(wr));
          }
          s1 *= scale;
          wr *= scale;
          std::size_t istart {ifirst};
          for (std::size_t j {ilast - 1}; j > ifirst; --j) { // two consecutive small subdiagonal entries
            double       small  {std::abs(s1 * f.h(j, j - 1))};
            double       diag   {std::abs((s1 * f.h(j, j)) - (wr * f.b(j, j)))};
            const double tempr  {std::max(small, diag)};
            if (tempr < 1.0) { // dhgeqz also asks tempr ≠ 0, where both stay 0 and the test holds: as here past safmin
              small /= std::max(tempr, std::numeric_limits<double>::min());
              diag  /= std::max(tempr, std::numeric_limits<double>::min());
            }
            if (std::abs((ascale * f.h(j + 1, j)) * small) <= (ascale * atol) * diag) {
              istart = j;
              break;
            }
          }
          const plane_rotation<double> first {givens((s1 * f.h(istart, istart)) - (wr * f.b(istart, istart)), s1 * f.h(istart + 1, istart))};
          double                       c     {first.c};
          double                       sine  {first.s};
          for (std::size_t j {istart}; j < ilast; ++j) {
            if (j > istart) {
              const plane_rotation<double> g {givens(f.h(j, j - 1), f.h(j + 1, j - 1))};
              f.h(j, j - 1)     = g.r;
              f.h(j + 1, j - 1) = 0.0;
              c                 = g.c;
              sine              = g.s;
            }
            form::rotate_rows(f.s, n, j, j + 1, j, n, c, sine);
            form::rotate_rows(f.t, n, j, j + 1, j, n, c, sine);
            form::rotate_rows(f.qt, n, j, j + 1, 1, n, c, sine);
            const plane_rotation<double> g {givens(f.b(j + 1, j + 1), f.b(j + 1, j))};
            f.b(j + 1, j + 1) = g.r;
            f.b(j + 1, j)     = 0.0;
            form::rotate_columns(f.s, n, j + 1, j, 1, std::min(j + 2, ilast), g.c, g.s);
            form::rotate_columns(f.t, n, j + 1, j, 1, j, g.c, g.s);
            form::rotate_rows(f.zt, n, j + 1, j, 1, n, g.c, g.s);
          }
          continue;
        }
        if (ifirst + 1 == ilast) { // label 200: a 2 × 2 block of a conjugate pair, standardized and split off
          triangle_svd svd {svd_of_triangle(f.b(ilast - 1, ilast - 1), f.b(ilast - 1, ilast), f.b(ilast, ilast))};
          double       b22 {svd.ssmin};
          double       b11 {svd.ssmax};
          if (b11 < 0.0) {
            svd.csr = -svd.csr;
            svd.snr = -svd.snr;
            b11     = -b11;
            b22     = -b22;
          }
          form::rotate_rows(f.s, n, ilast - 1, ilast, ilast - 1, n, svd.csl, svd.snl);
          form::rotate_columns(f.s, n, ilast - 1, ilast, 1, ilast, svd.csr, svd.snr);
          if (ilast < n) {
            form::rotate_rows(f.t, n, ilast - 1, ilast, ilast + 1, n, svd.csl, svd.snl);
          }
          if (ilast > 2) {
            form::rotate_columns(f.t, n, ilast - 1, ilast, 1, ilast - 2, svd.csr, svd.snr);
          }
          form::rotate_rows(f.qt, n, ilast - 1, ilast, 1, n, svd.csl, svd.snl);
          form::rotate_rows(f.zt, n, ilast - 1, ilast, 1, n, svd.csr, svd.snr);
          f.b(ilast - 1, ilast - 1) = b11;
          f.b(ilast - 1, ilast)     = 0.0;
          f.b(ilast, ilast - 1)     = 0.0;
          f.b(ilast, ilast)         = b22;
          if (b22 < 0.0) {
            for (std::size_t i {1}; i <= ilast; ++i) {
              f.h(i, ilast) = -f.h(i, ilast);
              f.b(i, ilast) = -f.b(i, ilast);
            }
            for (std::size_t i {0}; i < n; ++i) {
              f.zt[((ilast - 1) * n) + i] = -f.zt[((ilast - 1) * n) + i];
            }
            b22 = -b22;
          }
          const pencil_pair pair {pencil_eigenvalues(f.h(ilast - 1, ilast - 1), f.h(ilast - 1, ilast), f.h(ilast, ilast - 1), f.h(ilast, ilast),
                                                     f.b(ilast - 1, ilast - 1), f.b(ilast - 1, ilast), f.b(ilast, ilast), safmin * safety)};
          if (pair.wi == 0.0) { // real after all: the next iteration splits the block
            continue;
          }
          const auto [first_value, second_value] {pair_values(f, ilast, pair, b11, b22)};
          values[ilast - 2] = first_value;
          values[ilast - 1] = second_value;
          ilast             = ifirst - 1;
          if (ilast == 0) {
            return values;
          }
          iiter  = 0;
          eshift = 0.0;
          continue;
        }
        double_shift_sweep(f, ifirst, ilast, ascale, bscale, safmin);
      }
      throw convergence_error("QZ: the iterations did not converge (dhgeqz's info)");
    }

    //! \brief A square of order 4 at most, row-major in a fixed stride: the blocks `dtgex2` exchanges and its transforms.
    struct small_square
    {
      std::array<double, 16> e {}; //!< The entries, `(i, j)` at `4 i + j`.

      //! \brief The entry `(i, j)`, from 0.
      [[nodiscard]] double& operator()(std::size_t i,
                                       std::size_t j)
      {
        return e[(i * 4) + j];
      }

      //! \brief The entry `(i, j)`, from 0.
      [[nodiscard]] double operator()(std::size_t i,
                                      std::size_t j) const
      {
        return e[(i * 4) + j];
      }

      //! \brief The identity of order \p m.
      [[nodiscard]] static small_square identity(std::size_t m)
      {
        small_square out;
        for (std::size_t i {0}; i < m; ++i) {
          out(i, i) = 1.0;
        }
        return out;
      }
    };

    //! \brief `op(x) op(y)` of order \p m, `op` the transpose where \p tx (\p ty).
    [[nodiscard]] inline small_square small_product(const small_square& x,
                                                    bool                tx,
                                                    const small_square& y,
                                                    bool                ty,
                                                    std::size_t         m)
    {
      small_square out;
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < m; ++j) {
          double sum {0.0};
          for (std::size_t k {0}; k < m; ++k) {
            sum += (tx ? x(k, i) : x(i, k)) * (ty ? y(j, k) : y(k, j));
          }
          out(i, j) = sum;
        }
      }
      return out;
    }

    //! \brief The Frobenius norm of rows \p r0 … \p r1 − 1 and columns \p c0 … \p c1 − 1 of \p x.
    [[nodiscard]] inline double small_norm(const small_square& x,
                                           std::size_t         r0,
                                           std::size_t         r1,
                                           std::size_t         c0,
                                           std::size_t         c1)
    {
      double norm {0.0};
      for (std::size_t i {r0}; i < r1; ++i) {
        for (std::size_t j {c0}; j < c1; ++j) {
          norm = std::hypot(norm, x(i, j));
        }
      }
      return norm;
    }

    /*!
     * \brief The orthogonal `Q` (order \p m) of the Householder QR factorisation of the first \p k columns of \p x
     *        (`m × k`, `k ≤ m`), as LAPACK's `dgeqr2` then `dorg2r`: its first \p k columns span theirs, `Qᵀ x` upper
     *        trapezoidal.
     */
    [[nodiscard]] inline small_square small_qr(small_square x,
                                               std::size_t  m,
                                               std::size_t  k)
    {
      small_square q {small_square::identity(m)};
      for (std::size_t j {0}; j < k; ++j) {
        double rest {0.0};
        for (std::size_t i {j + 1}; i < m; ++i) {
          rest = std::hypot(rest, x(i, j));
        }
        if (rest == 0.0) {
          continue;
        }
        const double          alpha {x(j, j)};
        const double          beta  {-std::copysign(std::hypot(alpha, rest), alpha)};
        const double          tau   {(beta - alpha) / beta};
        std::array<double, 4> v     {};
        v[j] = 1.0;
        for (std::size_t i {j + 1}; i < m; ++i) {
          v[i] = x(i, j) / (alpha - beta);
        }
        for (std::size_t c {j}; c < m; ++c) { // x ← (I − τ v vᵀ) x
          double dot {0.0};
          for (std::size_t i {j}; i < m; ++i) {
            dot += v[i] * x(i, c);
          }
          for (std::size_t i {j}; i < m; ++i) {
            x(i, c) -= tau * v[i] * dot;
          }
        }
        for (std::size_t r {0}; r < m; ++r) { // Q ← Q (I − τ v vᵀ)
          double dot {0.0};
          for (std::size_t i {j}; i < m; ++i) {
            dot += q(r, i) * v[i];
          }
          for (std::size_t i {j}; i < m; ++i) {
            q(r, i) -= tau * dot * v[i];
          }
        }
      }
      return q;
    }

    //! \brief The rotations `dlagv2` applies to a 2 × 2 pencil: `[csl snl; −snl csl] (A, B) [csr −snr; snr csr]`.
    struct pencil_rotations
    {
      double csl {1.0}; //!< The left cosine.
      double snl {0.0}; //!< The left sine.
      double csr {1.0}; //!< The right cosine.
      double snr {0.0}; //!< The right sine.
    };

    /*!
     * \brief The 2 × 2 pencil (\p a, \p b), `b` upper triangular, standardized, as LAPACK's `dlagv2`: two real eigenvalues
     *        made triangular, a complex pair left with `b` diagonal and positive; both changed in place.
     */
    [[nodiscard]] inline pencil_rotations standardize_pencil(std::array<std::array<double, 2>, 2>& a,
                                                             std::array<std::array<double, 2>, 2>& b)
    {
      const double safmin {std::numeric_limits<double>::min()};
      const double ulp    {std::numeric_limits<double>::epsilon()};
      const double anorm  {std::max({std::abs(a[0][0]) + std::abs(a[1][0]), std::abs(a[0][1]) + std::abs(a[1][1]), safmin})};
      const double bnorm  {std::max({std::abs(b[0][0]), std::abs(b[0][1]) + std::abs(b[1][1]), safmin})};
      for (auto& row : a) {
        for (double& x : row) {
          x *= 1.0 / anorm;
        }
      }
      b[0][0] *= 1.0 / bnorm;
      b[0][1] *= 1.0 / bnorm;
      b[1][1] *= 1.0 / bnorm;
      // rows by (c, s): row 0 ← c row 0 + s row 1, row 1 ← c row 1 − s row 0; columns likewise
      const auto rows    = [](std::array<std::array<double, 2>, 2>& m, double c, double s) {
                             for (std::size_t j {0}; j < 2; ++j) {
                               const double t {(c * m[0][j]) + (s * m[1][j])};
                               m[1][j] = (c * m[1][j]) - (s * m[0][j]);
                               m[0][j] = t;
                             }
                           };
      const auto columns = [](std::array<std::array<double, 2>, 2>& m, double c, double s) {
                             for (std::size_t i {0}; i < 2; ++i) {
                               const double t {(c * m[i][0]) + (s * m[i][1])};
                               m[i][1] = (c * m[i][1]) - (s * m[i][0]);
                               m[i][0] = t;
                             }
                           };
      pencil_rotations out;
      if (std::abs(a[1][0]) <= ulp) {
        a[1][0] = 0.0;
        b[1][0] = 0.0;
      }
      else if (std::abs(b[0][0]) <= ulp) {
        const plane_rotation<double> g {givens(a[0][0], a[1][0])};
        out.csl = g.c;
        out.snl = g.s;
        rows(a, out.csl, out.snl);
        rows(b, out.csl, out.snl);
        a[1][0] = 0.0;
        b[0][0] = 0.0;
        b[1][0] = 0.0;
      }
      else if (std::abs(b[1][1]) <= ulp) {
        const plane_rotation<double> g {givens(a[1][1], a[1][0])};
        out.csr = g.c;
        out.snr = -g.s;
        columns(a, out.csr, out.snr);
        columns(b, out.csr, out.snr);
        a[1][0] = 0.0;
        b[1][0] = 0.0;
        b[1][1] = 0.0;
      }
      else {
        const pencil_pair pair {pencil_eigenvalues(a[0][0], a[0][1], a[1][0], a[1][1], b[0][0], b[0][1], b[1][1], safmin)};
        if (pair.wi == 0.0) {
          const double                 h1 {(pair.scale1 * a[0][0]) - (pair.wr1 * b[0][0])};
          const double                 h2 {(pair.scale1 * a[0][1]) - (pair.wr1 * b[0][1])};
          const double                 h3 {(pair.scale1 * a[1][1]) - (pair.wr1 * b[1][1])};
          const double                 rr {std::hypot(h1, h2)};
          const double                 qq {std::hypot(pair.scale1 * a[1][0], h3)};
          const plane_rotation<double> g  {rr > qq ? givens(h2, h1) : givens(h3, pair.scale1 * a[1][0])};
          out.csr = g.c;
          out.snr = -g.s;
          columns(a, out.csr, out.snr);
          columns(b, out.csr, out.snr);
          const double                 ha {std::max(std::abs(a[0][0]) + std::abs(a[0][1]), std::abs(a[1][0]) + std::abs(a[1][1]))};
          const double                 hb {std::max(std::abs(b[0][0]) + std::abs(b[0][1]), std::abs(b[1][0]) + std::abs(b[1][1]))};
          const plane_rotation<double> h  {pair.scale1 * ha >= std::abs(pair.wr1) * hb ? givens(b[0][0], b[1][0]) : givens(a[0][0], a[1][0])};
          out.csl = h.c;
          out.snl = h.s;
          rows(a, out.csl, out.snl);
          rows(b, out.csl, out.snl);
          a[1][0] = 0.0;
          b[1][0] = 0.0;
        }
        else {
          const triangle_svd svd {svd_of_triangle(b[0][0], b[0][1], b[1][1])};
          out.csl = svd.csl;
          out.snl = svd.snl;
          out.csr = svd.csr;
          out.snr = svd.snr;
          rows(a, out.csl, out.snl);
          rows(b, out.csl, out.snl);
          columns(a, out.csr, out.snr);
          columns(b, out.csr, out.snr);
          b[1][0] = 0.0;
          b[0][1] = 0.0;
        }
      }
      for (auto& row : a) {
        for (double& x : row) {
          x *= anorm;
        }
      }
      for (auto& row : b) {
        for (double& x : row) {
          x *= bnorm;
        }
      }
      return out;
    }

    /*!
     * \brief The solution `(R, L)` (`n1 × n2` each) of the generalized Sylvester equations `S₁₁ R − L S₂₂ = S₁₂`,
     *        `T₁₁ R − L T₂₂ = T₁₂` of the `m × m` blocks of \p s and \p t (`m = n1 + n2`), as LAPACK's `dtgsy2` solves
     *        them: by elimination with complete pivoting of their Kronecker form; nothing where a pivot falls below
     *        `ε` times the largest entry (`dgetc2` would perturb it, and `dtgex2` then refuses the exchange).
     */
    [[nodiscard]] inline std::optional<std::pair<small_square, small_square>> generalized_sylvester(const small_square& s,
                                                                                                    const small_square& t,
                                                                                                    std::size_t         n1,
                                                                                                    std::size_t         n2)
    {
      const std::size_t                    k    {n1 * n2};
      const std::size_t                    size {2 * k};
      std::array<std::array<double, 9>, 8> z    {}; // the system and its right-hand side, column size
      for (std::size_t i {0}; i < n1; ++i) {
        for (std::size_t j {0}; j < n2; ++j) {
          const std::size_t row {(i * n2) + j};
          for (std::size_t p {0}; p < n1; ++p) { // S₁₁ R and T₁₁ R
            z[row][(p * n2) + j]     += s(i, p);
            z[row + k][(p * n2) + j] += t(i, p);
          }
          for (std::size_t q {0}; q < n2; ++q) { // − L S₂₂ and − L T₂₂
            z[row][k + (i * n2) + q]     -= s(n1 + q, n1 + j);
            z[row + k][k + (i * n2) + q] -= t(n1 + q, n1 + j);
          }
          z[row][size]     = s(i, n1 + j);
          z[row + k][size] = t(i, n1 + j);
        }
      }
      double largest {0.0};
      for (std::size_t r {0}; r < size; ++r) {
        for (std::size_t c {0}; c < size; ++c) {
          largest = std::max(largest, std::abs(z[r][c]));
        }
      }
      const double               smin  {std::max(std::numeric_limits<double>::epsilon() * largest, std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon())};
      std::array<std::size_t, 8> order {};
      for (std::size_t c {0}; c < size; ++c) {
        order[c] = c;
      }
      for (std::size_t p {0}; p < size; ++p) { // complete pivoting, as dgetc2
        std::size_t pr {p};
        std::size_t pc {p};
        for (std::size_t r {p}; r < size; ++r) {
          for (std::size_t c {p}; c < size; ++c) {
            if (std::abs(z[r][c]) > std::abs(z[pr][pc])) {
              pr = r;
              pc = c;
            }
          }
        }
        if (!(std::abs(z[pr][pc]) >= smin)) {
          return std::nullopt;
        }
        std::swap(z[p], z[pr]);
        for (std::size_t r {0}; r < size; ++r) {
          std::swap(z[r][p], z[r][pc]);
        }
        std::swap(order[p], order[pc]);
        for (std::size_t r {p + 1}; r < size; ++r) {
          const double factor {z[r][p] / z[p][p]};
          for (std::size_t c {p}; c <= size; ++c) {
            z[r][c] -= factor * z[p][c];
          }
        }
      }
      std::array<double, 8> x {};
      for (std::size_t p {size}; p-- > 0;) {
        double sum {z[p][size]};
        for (std::size_t c {p + 1}; c < size; ++c) {
          sum -= z[p][c] * x[c];
        }
        x[p] = sum / z[p][p];
      }
      small_square r;
      small_square l;
      for (std::size_t c {0}; c < size; ++c) {
        const std::size_t unknown {order[c]};
        if (!std::isfinite(x[c])) {
          return std::nullopt;
        }
        if (unknown < k) {
          r(unknown / n2, unknown % n2) = x[c];
        }
        else {
          l((unknown - k) / n2, (unknown - k) % n2) = x[c];
        }
      }
      return std::pair {r, l};
    }

    //! \brief An exchange of two diagonal blocks as \ref exchange_real_blocks plans it: the blocks after it, and the
    //!        transforms of rows and columns, `S' = LIᵀ S IR` (`IR` the columns' transform).
    struct block_exchange
    {
      small_square s;  //!< The `m × m` block of `S` after the exchange.
      small_square t;  //!< Of `T`.
      small_square li; //!< The rows' transform: `S' = LIᵀ S IR`, `Q ← Q LI`.
      small_square ir; //!< The columns': `Z ← Z IR`.
    };

    /*!
     * \brief Two 1 × 1 blocks' exchange, as LAPACK's `dtgex2` makes it: columns turned onto the second eigenvalue's right
     *        eigenvector, then rows back to triangular, the rotation from `S`'s new first column where `|S₂₂ T₁₁| ≥ |S₁₁
     *        T₂₂|`, else `T`'s. Nothing where the new subdiagonal entries pass \p thresha and \p threshb (the weak test).
     */
    [[nodiscard]] inline std::optional<block_exchange> plan_single_exchange(small_square s,
                                                                            small_square t,
                                                                            double       thresha,
                                                                            double       threshb)
    {
      const double                 ff {(s(1, 1) * t(0, 0)) - (t(1, 1) * s(0, 0))};
      const double                 gg {(s(1, 1) * t(0, 1)) - (t(1, 1) * s(0, 1))};
      const double                 sa {std::abs(s(1, 1)) * std::abs(t(0, 0))};
      const double                 sb {std::abs(s(0, 0)) * std::abs(t(1, 1))};
      const plane_rotation<double> z  {givens(ff, gg)}; // dtgex2's IR(1, 2) = c, IR(1, 1) = s: columns by (s, −c)
      small_square                 ir;
      ir(0, 0) = z.s;
      ir(1, 0) = -z.c;
      ir(0, 1) = z.c;
      ir(1, 1) = z.s;
      s        = small_product(s, false, ir, false, 2);
      t        = small_product(t, false, ir, false, 2);
      const plane_rotation<double> q {sa >= sb ? givens(s(0, 0), s(1, 0)) : givens(t(0, 0), t(1, 0))};
      small_square                 li; // rows by (c, s): S' = LIᵀ S
      li(0, 0) = q.c;
      li(1, 0) = q.s;
      li(0, 1) = -q.s;
      li(1, 1) = q.c;
      s        = small_product(li, true, s, false, 2);
      t        = small_product(li, true, t, false, 2);
      if (!(std::abs(s(1, 0)) <= thresha && std::abs(t(1, 0)) <= threshb)) { // NaN fails it
        return std::nullopt;
      }
      return block_exchange {s, t, li, ir}; // the entries below the diagonal, within the weak test's bounds, kept for the strong one
    }

    /*!
     * \brief The exchange of blocks of orders \p n1 and \p n2 (one of them 2), as LAPACK's `dtgex2` makes it: orthogonal
     *        bases of the deflating subspaces the generalized Sylvester equations give (\ref generalized_sylvester), `T`
     *        made triangular again by an RQ or a QR factorisation (the one that leaves the smaller block below `S`'s
     *        diagonal), each 2 × 2 block standardized (\ref standardize_pencil). Nothing where the equations have no
     *        solution or that block passes \p thresha (the weak test).
     */
    [[nodiscard]] inline std::optional<block_exchange> plan_block_exchange(small_square s,
                                                                           small_square t,
                                                                           std::size_t  n1,
                                                                           std::size_t  n2,
                                                                           double       thresha)
    {
      const std::size_t                                          m      {n1 + n2};
      const std::optional<std::pair<small_square, small_square>> solved {generalized_sylvester(s, t, n1, n2)};
      if (!solved) {
        return std::nullopt;
      }
      const auto& [r, l] {*solved};
      small_square lbasis; // [−L; I] (m × n2)
      for (std::size_t j {0}; j < n2; ++j) {
        for (std::size_t i {0}; i < n1; ++i) {
          lbasis(i, j) = -l(i, j);
        }
        lbasis(n1 + j, j) = 1.0;
      }
      small_square li {small_qr(lbasis, m, n2)}; // its first n2 columns span [−L; I]
      small_square rbasis;                       // [I R]ᵀ (m × n1)
      for (std::size_t i {0}; i < n1; ++i) {
        rbasis(i, i) = 1.0;
        for (std::size_t j {0}; j < n2; ++j) {
          rbasis(n1 + j, i) = r(i, j);
        }
      }
      const small_square qn {small_qr(rbasis, m, n1)};
      small_square       ir; // by rows: the last n1 span those of [I R], the first n2 their complement
      for (std::size_t c {0}; c < m; ++c) {
        for (std::size_t i {0}; i < n1; ++i) {
          ir(n2 + i, c) = qn(c, i);
        }
        for (std::size_t i {0}; i < n2; ++i) {
          ir(i, c) = qn(c, n1 + i);
        }
      }
      s = small_product(small_product(li, true, s, false, m), false, ir, true, m);
      t = small_product(small_product(li, true, t, false, m), false, ir, true, m);
      // T triangular again: by RQ (T = R Q: S ← S Qᵀ, IR ← Q IR) or by QR (T = Q R: S ← Qᵀ S, LI ← LI Q)
      small_square flipped; // J Tᵀ J, whose QR gives T's RQ
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < m; ++j) {
          flipped(i, j) = t(m - 1 - j, m - 1 - i);
        }
      }
      const small_square qf {small_qr(flipped, m, m)};
      small_square       qrq; // Q of T = R Q: J Q_fᵀ J
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < m; ++j) {
          qrq(i, j) = qf(m - 1 - j, m - 1 - i);
        }
      }
      const small_square qt     {small_qr(t, m, m)};
      const small_square s_rq   {small_product(s, false, qrq, true, m)};
      const small_square s_qr   {small_product(qt, true, s, false, m)};
      const double       brqa21 {small_norm(s_rq, n2, m, 0, n2)};
      const double       bqra21 {small_norm(s_qr, n2, m, 0, n2)};
      if (bqra21 <= brqa21 && bqra21 <= thresha) {
        s  = s_qr;
        t  = small_product(qt, true, t, false, m);
        li = small_product(li, false, qt, false, m);
      }
      else if (!(brqa21 < thresha)) { // NaN fails it
        return std::nullopt;
      }
      else {
        s  = s_rq;
        t  = small_product(t, false, qrq, true, m);
        ir = small_product(qrq, false, ir, false, m);
      }
      // T's below its diagonal (S's below the blocks kept for the strong test), over the whole 4 × 4, whose rows and
      // columns past m are zero: bounded by m, GCC 13 cannot tell the rows stay within it (-Warray-bounds)
      for (std::size_t i {1}; i < 4; ++i) {
        for (std::size_t j {0}; j < i; ++j) {
          t(i, j) = 0.0;
        }
      }
      // each 2 × 2 block standardized: the left rotations gathered in `left` (its transpose applied), the right in `right`
      small_square left {small_square::identity(m)};
      small_square right {small_square::identity(m)};
      const auto   standardize_at = [&](std::size_t at) {
                                      std::array<std::array<double, 2>, 2> a {{{s(at, at), s(at, at + 1)}, {s(at + 1, at), s(at + 1, at + 1)}}};
                                      std::array<std::array<double, 2>, 2> b {{{t(at, at), t(at, at + 1)}, {t(at + 1, at), t(at + 1, at + 1)}}};
                                      const pencil_rotations               g {standardize_pencil(a, b)};
                                      for (std::size_t i {0}; i < 2; ++i) {
                                        for (std::size_t j {0}; j < 2; ++j) {
                                          s(at + i, at + j) = a[i][j];
                                          t(at + i, at + j) = b[i][j];
                                        }
                                      }
                                      left(at, at)          = g.csl;
                                      left(at + 1, at)      = g.snl;
                                      left(at, at + 1)      = -g.snl;
                                      left(at + 1, at + 1)  = g.csl;
                                      right(at, at)         = g.csr;
                                      right(at + 1, at)     = g.snr;
                                      right(at, at + 1)     = -g.snr;
                                      right(at + 1, at + 1) = g.csr;
                                    };
      if (n2 > 1) {
        standardize_at(0);
      }
      if (n1 > 1) {
        standardize_at(n2);
      }
      // the block between them: leftᵀ on its rows, right on its columns
      const small_square s_rest {small_product(small_product(left, true, s, false, m), false, right, false, m)};
      const small_square t_rest {small_product(small_product(left, true, t, false, m), false, right, false, m)};
      for (std::size_t i {0}; i < n2; ++i) {
        for (std::size_t j {n2}; j < m; ++j) {
          s(i, j) = s_rest(i, j);
          t(i, j) = t_rest(i, j);
        }
      }
      return block_exchange {s, t, small_product(li, false, left, false, m), small_product(ir, true, right, false, m)};
    }

    /*!
     * \brief Exchanges the adjacent diagonal blocks of orders \p n1 and \p n2 (1 or 2) at \p j1 (from 1) of the real
     *        generalized Schur form \p f, as LAPACK's `dtgex2`: planned by \ref plan_single_exchange or \ref
     *        plan_block_exchange, then refused unless the blocks rebuilt from the transforms, `LI S' IRᵀ`, are within
     *        `20 ε` of the originals' norms (the strong test, after the 2 × 2 blocks' standardization where `dtgex2`
     *        makes it before: their rotations' rounding more); else the entries below the new blocks set to 0 and the
     *        rows and columns of `S`, `T`, `Q` and `Z` turned.
     * \return Whether the exchange was made, \p f unchanged where not.
     */
    [[nodiscard]] inline bool exchange_real_blocks(real_qz_form& f,
                                                   std::size_t   j1,
                                                   std::size_t   n1,
                                                   std::size_t   n2)
    {
      const std::size_t n {f.n};
      const std::size_t m {n1 + n2};
      small_square      s;
      small_square      t;
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < m; ++j) {
          s(i, j) = f.h(j1 + i, j1 + j);
          t(i, j) = f.b(j1 + i, j1 + j);
        }
      }
      const double                        eps     {std::numeric_limits<double>::epsilon()};
      const double                        smlnum  {std::numeric_limits<double>::min() / eps};
      const double                        thresha {std::max(20.0 * eps * small_norm(s, 0, m, 0, m), smlnum)};
      const double                        threshb {std::max(20.0 * eps * small_norm(t, 0, m, 0, m), smlnum)};
      const std::optional<block_exchange> plan    {m == 2 ? plan_single_exchange(s, t, thresha, threshb) : plan_block_exchange(s, t, n1, n2, thresha)};
      if (!plan) {
        return false;
      }
      for (const bool of_s : {true, false}) { // the strong test
        const small_square rebuilt {small_product(small_product(plan->li, false, of_s ? plan->s : plan->t, false, m), false, plan->ir, true, m)};
        small_square       gap;
        for (std::size_t i {0}; i < m; ++i) {
          for (std::size_t j {0}; j < m; ++j) {
            gap(i, j) = rebuilt(i, j) - (of_s ? s(i, j) : t(i, j));
          }
        }
        if (!(small_norm(gap, 0, m, 0, m) <= (of_s ? thresha : threshb))) {
          return false;
        }
      }
      const small_square& li   {plan->li};
      const small_square& ir   {plan->ir};
      const std::size_t   lead {m == 2 ? 1 : n2}; // the new leading block's order
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < m; ++j) {
          const bool below {i >= lead && j < lead};
          f.h(j1 + i, j1 + j) = below ? 0.0 : plan->s(i, j);
          f.b(j1 + i, j1 + j) = below ? 0.0 : plan->t(i, j);
        }
      }
      const auto right_times = [&](buffer<double>& x, std::size_t rows_last) { // columns j1 … of rows 1 … rows_last
                                 for (std::size_t row {1}; row <= rows_last; ++row) {
                                   std::array<double, 4> v {};
                                   for (std::size_t c {0}; c < m; ++c) {
                                     for (std::size_t k {0}; k < m; ++k) {
                                       v[c] += x[((row - 1) * n) + (j1 - 1 + k)] * ir(k, c);
                                     }
                                   }
                                   for (std::size_t c {0}; c < m; ++c) {
                                     x[((row - 1) * n) + (j1 - 1 + c)] = v[c];
                                   }
                                 }
                               };
      const auto left_times = [&](buffer<double>& x) { // LIᵀ on rows j1 … of the columns past the blocks
                                for (std::size_t col {j1 + m}; col <= n; ++col) {
                                  std::array<double, 4> v {};
                                  for (std::size_t r2 {0}; r2 < m; ++r2) {
                                    for (std::size_t k {0}; k < m; ++k) {
                                      v[r2] += li(k, r2) * x[((j1 - 1 + k) * n) + (col - 1)];
                                    }
                                  }
                                  for (std::size_t r2 {0}; r2 < m; ++r2) {
                                    x[((j1 - 1 + r2) * n) + (col - 1)] = v[r2];
                                  }
                                }
                              };
      right_times(f.s, j1 - 1);
      right_times(f.t, j1 - 1);
      left_times(f.s);
      left_times(f.t);
      const auto times_transposed = [&](buffer<double>& xt, const small_square& factor) { // X ← X F on Xᵀ's rows j1 …
                                      for (std::size_t row {1}; row <= n; ++row) {
                                        std::array<double, 4> v {};
                                        for (std::size_t c {0}; c < m; ++c) {
                                          for (std::size_t k {0}; k < m; ++k) {
                                            v[c] += xt[((j1 - 1 + k) * n) + (row - 1)] * factor(k, c);
                                          }
                                        }
                                        for (std::size_t c {0}; c < m; ++c) {
                                          xt[((j1 - 1 + c) * n) + (row - 1)] = v[c];
                                        }
                                      }
                                    };
      times_transposed(f.zt, ir);
      times_transposed(f.qt, li);
      return true;
    }

    //! \brief \ref exchange_real_blocks, a refusal thrown as \ref real_generalized_schur::reorder reports it.
    inline void exchange_or_refuse(real_qz_form& f,
                                   std::size_t   j1,
                                   std::size_t   n1,
                                   std::size_t   n2)
    {
      if (!exchange_real_blocks(f, j1, n1, n2)) {
        throw convergence_error("generalized Schur form: eigenvalues could not be separated for reordering");
      }
    }

    /*!
     * \brief Moves the diagonal block at \p ifst (from 1) of the real generalized Schur form \p f up to \p ilst, as LAPACK's
     *        `dtgexc` does it upwards: exchanged with each block above in turn (\ref exchange_real_blocks), a 2 × 2 block
     *        that splits on the way moved as its two 1 × 1 ones; \p ifst and \p ilst taken to the first row of their
     *        blocks.
     * \throws convergence_error If an exchange is refused (\ref exchange_or_refuse); \p f then partly moved.
     */
    inline void move_real_block_up(real_qz_form& f,
                                   std::size_t   ifst,
                                   std::size_t   ilst)
    {
      const std::size_t n {f.n};
      if (ifst > 1 && f.h(ifst, ifst - 1) != 0.0) {
        --ifst;
      }
      std::size_t nbf {ifst < n && f.h(ifst + 1, ifst) != 0.0 ? std::size_t {2} : std::size_t {1}};
      if (ilst > 1 && f.h(ilst, ilst - 1) != 0.0) {
        --ilst;
      }
      std::size_t here {ifst};
      while (here > ilst) {
        const std::size_t nbnext {here >= 3 && f.h(here - 1, here - 2) != 0.0 ? std::size_t {2} : std::size_t {1}};
        if (nbf != 3) {
          exchange_or_refuse(f, here - nbnext, nbnext, nbf);
          here -= nbnext;
          if (nbf == 2 && f.h(here + 1, here) == 0.0) { // the block split in two real eigenvalues
            nbf = 3;
          }
          continue;
        }
        // two 1 × 1 blocks, here and here + 1: the upper one past the block above, then the lower one
        exchange_or_refuse(f, here - nbnext, nbnext, 1);
        if (nbnext == 1) {
          exchange_or_refuse(f, here, 1, 1);
          --here;
        }
        else if (f.h(here, here - 1) != 0.0) {
          exchange_or_refuse(f, here - 1, 2, 1);
          here -= 2;
        }
        else { // the block above split in passing: its two halves past the lower one in turn
          exchange_or_refuse(f, here, 1, 1);
          --here;
          exchange_or_refuse(f, here, 1, 1);
          --here;
        }
      }
    }

    /*!
     * \brief The eigenvalues of the real generalized Schur form \p f after a reordering, as LAPACK's `dtgsen` gives them:
     *        a 2 × 2 block's pair by `dlag2` (`β` its scale), a 1 × 1 block's from its diagonal, `T(k, k)` made
     *        non-negative (row `k` of `S` and `T` and column `k` of `Q` negated).
     */
    [[nodiscard]] inline std::vector<real_pencil_value> real_values(real_qz_form& f)
    {
      const std::size_t              n {f.n};
      std::vector<real_pencil_value> values(n);
      for (std::size_t k {1}; k <= n; ++k) {
        if (k < n && f.h(k + 1, k) != 0.0) {
          const pencil_pair pair {pencil_eigenvalues(f.h(k, k), f.h(k, k + 1), f.h(k + 1, k), f.h(k + 1, k + 1), f.b(k, k), f.b(k, k + 1), f.b(k + 1, k + 1),
                                                     std::numeric_limits<double>::min())};
          values[k - 1] = {pair.wr1, pair.wi, pair.scale1};
          values[k]     = {pair.wr2, -pair.wi, pair.scale2};
          ++k;
          continue;
        }
        if (std::signbit(f.b(k, k))) {
          for (std::size_t i {1}; i <= n; ++i) {
            f.h(k, i)                     = -f.h(k, i);
            f.b(k, i)                     = -f.b(k, i);
            f.qt[((k - 1) * n) + (i - 1)] = -f.qt[((k - 1) * n) + (i - 1)];
          }
        }
        values[k - 1] = {f.h(k, k), 0.0, f.b(k, k)};
      }
      return values;
    }
  } // namespace detail

  namespace detail {
    /*!
     * \brief \p m brought into the range `[√safmin / ε, ε / √safmin]` of its largest magnitude where it lies outside,
     *        as LAPACK's `xgges` scales `A` and `B` before the QZ iterations (their norms' squares, the shifts and the
     *        rotations stay in range): the matrix, and the factor that takes the scaled one back (1 where none).
     */
    template <typename T>
    [[nodiscard]] std::pair<matrix<T>, double> pencil_operand_in_range(const matrix<T>& m)
    {
      const double smlnum {std::sqrt(std::numeric_limits<double>::min()) / std::numeric_limits<double>::epsilon()};
      const double bignum {1.0 / smlnum};
      double       norm   {0.0};
      for (const T& x : m) {
        norm = std::max(norm, static_cast<double>(std::abs(x)));
      }
      const double to {norm > 0.0 && norm < smlnum ? smlnum : norm > bignum ? bignum : norm};
      if (to == norm) {
        return {m, 1.0};
      }
      const double factor {to / norm};
      buffer<T>    entries(m.begin(), m.end());
      for (T& x : entries) {
        x *= factor;
      }
      return {matrix<T> {m.rows(), m.cols(), std::move(entries)}, norm / to};
    }

    //! \brief The entries of \p m multiplied by \p factor, `n × n` row-major (the scaled form taken back).
    template <typename T>
    [[nodiscard]] matrix<T> scaled_back(const buffer<T>& m,
                                        std::size_t      n,
                                        double           factor)
    {
      buffer<T> entries {m};
      for (T& x : entries) {
        x *= factor;
      }
      return matrix<T> {n, n, std::move(entries)};
    }
  } // namespace detail

  /*!
   * \brief The generalized Schur decomposition of a complex pencil `A − λ B`, `A = Q S Zᴴ` and `B = Q T Zᴴ`, as
   *        `scipy.linalg.qz(a, b, output="complex")` (LAPACK's `zgges`): `Q` and `Z` unitary, `S` and `T` upper
   *        triangular, `T`'s diagonal real and non-negative; the generalized eigenvalues `α_k / β_k` on the diagonals
   *        (`β_k = 0` an infinite one). Reordered as `scipy.linalg.ordqz` by \ref sort or \ref reorder. Each operand is
   *        first brought into the range `[√safmin / ε, ε / √safmin]` of its largest magnitude where it lies outside, as
   *        `zgges` scales it, and the form and the eigenvalues taken back.
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
      // as zgges: each operand within the range of its largest magnitude, S and T and the eigenvalues taken back
      const auto [scaled_a, a_back] {detail::pencil_operand_in_range(a)};
      const auto [scaled_b, b_back] {detail::pencil_operand_in_range(b)};
      a_back_ = a_back;
      b_back_ = b_back;
      form_   = detail::hessenberg_triangular(scaled_a, scaled_b);
      values_ = unscaled(detail::qz_iteration(form_, steps));
    }

    //! \brief `S`, upper triangular.
    [[nodiscard]] matrix<std::complex<double>> s() const
    {
      return detail::scaled_back(form_.s, form_.n, a_back_);
    }

    //! \brief `T`, upper triangular, its diagonal real and non-negative.
    [[nodiscard]] matrix<std::complex<double>> t() const
    {
      return detail::scaled_back(form_.t, form_.n, b_back_);
    }

    //! \brief `Q`, unitary: `A = Q S Zᴴ`.
    [[nodiscard]] matrix<std::complex<double>> q() const
    {
      return transpose(matrix<std::complex<double>> {form_.n, form_.n, form_.qt});
    }

    //! \brief `Z`, unitary.
    [[nodiscard]] matrix<std::complex<double>> z() const
    {
      return transpose(matrix<std::complex<double>> {form_.n, form_.n, form_.zt});
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
      std::vector<std::pair<std::complex<double>, std::complex<double>>> standardized(n);
      for (std::size_t k {1}; k <= n; ++k) {
        standardized[k - 1] = detail::standardize(work, k);
      }
      values_ = unscaled(std::move(standardized));
      form_   = std::move(work);
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

    //! \brief The eigenvalues of the scaled form \p raw taken back: `α` by `A`'s factor, `β` by `B`'s.
    [[nodiscard]] std::vector<std::pair<std::complex<double>, std::complex<double>>> unscaled(
      std::vector<std::pair<std::complex<double>, std::complex<double>>> raw) const
    {
      for (auto& [alpha, beta] : raw) {
        alpha *= a_back_;
        beta  *= b_back_;
      }
      return raw;
    }

    detail::qz_form                                                    form_;         //!< S, T, Q and Z, of the scaled pencil.
    std::vector<std::pair<std::complex<double>, std::complex<double>>> values_;       //!< The eigenvalues as (α, β).
    double                                                             a_back_ {1.0}; //!< The factor taking S back to A's scale.
    double                                                             b_back_ {1.0}; //!< T's.
  };

  namespace detail {

    /*!
     * \brief The right (or, where \p left, the left) eigenvectors of the triangular pencil `(s, t)` whose diagonal is
     *        \p values, as LAPACK's `ztgevc`: for each `(α, β)`, `x` with `(β S − α T) x = 0` and `x_k = 1` by back
     *        substitution (`yᴴ (β S − α T) = 0` forward for a left one), then `basis x` made of unit 2-norm; a diagonal
     *        entry of `β S − α T` below `ε (|β| ‖S‖ + |α| ‖T‖)` taken as that bound (`SMIN`), and `α` and `β` both zero
     *        (a singular pencil) given the unit vector of its place. The substitutions on the threads, each on its own;
     *        the basis applied to their nonzero parts by one product.
     * \return The vectors as columns.
     */
    [[nodiscard]] inline matrix<std::complex<double>> triangular_pencil_vectors(const matrix<std::complex<double>>&                                       s,
                                                                                const matrix<std::complex<double>>&                                       t,
                                                                                const std::vector<std::pair<std::complex<double>, std::complex<double>>>& values,
                                                                                const matrix<std::complex<double>>&                                       basis,
                                                                                bool                                                                      left)
    {
      using C = std::complex<double>;
      const std::size_t n      {s.rows()};
      double            s_norm {0.0};
      double            t_norm {0.0};
      for (const C& x : s) {
        s_norm = std::max(s_norm, std::abs(x));
      }
      for (const C& x : t) {
        t_norm = std::max(t_norm, std::abs(x));
      }
      buffer<C> solved(n * n, C {}); // column k the substitution's x: rows 0 … k for a right vector, k … n − 1 a left
      parallel_for(n, n * n / 2, [&](std::size_t k) {
                     const C alpha {values[k].first};
                     const C beta  {values[k].second};
                     C*      x     {solved.data() + k}; // x[i] at x + i n
                     x[k * n] = C {1.0};
                     if (std::abs(alpha) <= std::numeric_limits<double>::min() && std::abs(beta) <= std::numeric_limits<double>::min()) {
                       return;
                     }
                     const double smallest {std::numeric_limits<double>::epsilon() * ((std::abs(beta) * s_norm) + (std::abs(alpha) * t_norm))};
                     const auto   entry = [&](std::size_t i, std::size_t j) { return (beta * s(i, j)) - (alpha * t(i, j)); }; // β S − α T
                     const auto   pivot = [&](std::size_t i) {
                                            const C m {entry(i, i)};
                                            return std::abs(m) < smallest ? C {smallest} : m;
                                          };
                     if (!left) { // rows k − 1 … 0 of the upper triangular system
                       for (std::size_t i {k}; i-- > 0;) {
                         C sum {};
                         for (std::size_t j {i + 1}; j <= k; ++j) {
                           sum += entry(i, j) * x[j * n];
                         }
                         x[i * n] = -sum / pivot(i);
                       }
                     }
                     else { // yᴴ M = 0: (Mᴴ y)_i = 0 for i > k, Mᴴ lower triangular
                       for (std::size_t i {k + 1}; i < n; ++i) {
                         C sum {};
                         for (std::size_t j {k}; j < i; ++j) {
                           sum += std::conj(entry(j, i)) * x[j * n];
                         }
                         x[i * n] = -sum / std::conj(pivot(i));
                       }
                     }
                   });
      const matrix<C> product {basis * matrix<C> {n, n, std::move(solved)}}; // the zeros of each x add nothing
      buffer<C>       out(product.begin(), product.end());
      for (std::size_t k {0}; k < n; ++k) {
        double norm {0.0};
        for (std::size_t r {0}; r < n; ++r) {
          norm = std::hypot(norm, std::abs(out[(r * n) + k]));
        }
        for (std::size_t r {0}; r < n; ++r) {
          out[(r * n) + k] /= norm;
        }
      }
      return matrix<C> {n, n, std::move(out)};
    }
  } // namespace detail

  /*!
   * \brief The right (or, where \p left, the left) eigenvectors of the pencil \p d decomposes, as LAPACK's `ztgevc` and
   *        `scipy.linalg.eig(a, b)`: for each eigenvalue `(α, β)` in the diagonals' order, `x` with `(β S − α T) x = 0`
   *        and `x_k = 1` by back substitution (`yᴴ (β S − α T) = 0` forward for a left one), then `Z x` (`Q y`) made of
   *        unit 2-norm, as SciPy normalises `ggev`'s; a diagonal entry of `β S − α T` below `ε (|β| ‖S‖ + |α| ‖T‖)` taken
   *        as that bound (`ztgevc`'s `SMIN`), and an eigenvalue with `α` and `β` both zero (a singular pencil) given the
   *        unit vector of its place, as `ztgevc` gives it.
   * \return The vectors as columns.
   */
  [[nodiscard]] inline matrix<std::complex<double>> generalized_eigenvectors(const generalized_schur& d,
                                                                             bool                     left)
  {
    return detail::triangular_pencil_vectors(d.s(), d.t(), d.values(), left ? d.q() : d.z(), left);
  }

  /*!
   * \brief The generalized real Schur decomposition of a real pencil `A − λ B`, `A = Q S Zᵀ` and `B = Q T Zᵀ`, as
   *        `scipy.linalg.qz(a, b)` (LAPACK's `dgges`): `Q` and `Z` orthogonal, `T` upper triangular with a non-negative
   *        diagonal, `S` quasi-triangular, a 2 × 2 diagonal block for each conjugate pair of eigenvalues, `T` diagonal
   *        and positive there; the eigenvalues `(αr + i αi) / β` (`β = 0` an infinite one).
   */
  class real_generalized_schur
  {
  public:

    /*!
     * \brief Decomposes the pencil of \p a and \p b.
     * \throws dimension_error   If \p a or \p b is not square or their orders differ.
     * \throws non_finite_error  If an entry is infinite or NaN.
     * \throws convergence_error If the QZ iterations did not converge.
     */
    real_generalized_schur(const matrix<double>& a,
                           const matrix<double>& b)
      : real_generalized_schur(a,
                               b,
                               30 * std::max<std::size_t>(a.rows(), 1))
    {}

    //! \brief Decomposes the pencil within \p steps QZ iterations (`dhgeqz`'s bound `30 n` by default).
    real_generalized_schur(const matrix<double>& a,
                           const matrix<double>& b,
                           std::size_t           steps)
    {
      const std::size_t n {a.rows()};
      if (a.cols() != n || b.rows() != n || b.cols() != n) {
        throw dimension_error("the generalized Schur decomposition requires two square matrices of one order");
      }
      detail::require_finite(a, "generalized Schur decomposition");
      detail::require_finite(b, "generalized Schur decomposition");
      // as dgges: each operand within the range of its largest magnitude, S and T and the eigenvalues taken back
      const auto [scaled_a, a_back] {detail::pencil_operand_in_range(a)};
      const auto [scaled_b, b_back] {detail::pencil_operand_in_range(b)};
      a_back_ = a_back;
      b_back_ = b_back;
      form_   = detail::real_hessenberg_triangular(scaled_a, scaled_b);
      values_ = unscaled(detail::real_qz_iteration(form_, steps));
    }

    //! \brief `S`, quasi-triangular.
    [[nodiscard]] matrix<double> s() const
    {
      return detail::scaled_back(form_.s, form_.n, a_back_);
    }

    //! \brief `T`, upper triangular, its diagonal non-negative.
    [[nodiscard]] matrix<double> t() const
    {
      return detail::scaled_back(form_.t, form_.n, b_back_);
    }

    //! \brief `Q`, orthogonal: `A = Q S Zᵀ`.
    [[nodiscard]] matrix<double> q() const
    {
      return transpose(matrix<double> {form_.n, form_.n, form_.qt});
    }

    //! \brief `Z`, orthogonal.
    [[nodiscard]] matrix<double> z() const
    {
      return transpose(matrix<double> {form_.n, form_.n, form_.zt});
    }

    //! \brief The eigenvalues as `(αr, αi, β)`, in the diagonal's order.
    [[nodiscard]] const std::vector<detail::real_pencil_value>& values() const noexcept
    {
      return values_;
    }

    //! \brief The selection `scipy.linalg.ordqz` takes: true for `(α, β)` to lead, `α = αr + i αi`.
    using selection = std::function<bool (const std::complex<double>&, const std::complex<double>&)>;

    /*!
     * \brief Moves the eigenvalues whose entries of \p chosen are set to the leading diagonal places, in their order, as
     *        LAPACK's `dtgsen` (by `dtgexc`'s exchanges of blocks), a conjugate pair moved together where either of its
     *        two is chosen; `Q` and `Z` turning with them, the eigenvalues recomputed as `dtgsen` gives them.
     *        \return The eigenvalues moved.
     * \throws dimension_error   If \p chosen does not hold one entry per eigenvalue.
     * \throws convergence_error If an exchange is refused (`dtgex2`'s tests: two eigenvalues too close to separate);
     *                           the decomposition is then unchanged.
     */
    std::size_t reorder(const std::vector<bool>& chosen)
    {
      const std::size_t n {form_.n};
      if (chosen.size() != n) {
        throw dimension_error("generalized Schur reordering requires one choice per eigenvalue");
      }
      detail::real_qz_form work  {form_};
      std::size_t          moved {0};
      for (std::size_t k {1}; k <= n; ++k) {
        const bool pair {k < n && work.h(k + 1, k) != 0.0};
        if (chosen[k - 1] || (pair && chosen[k])) {
          detail::move_real_block_up(work, k, moved + 1);
          moved += pair ? 2 : 1;
        }
        k += pair ? 1 : 0;
      }
      values_ = unscaled(detail::real_values(work));
      form_   = std::move(work);
      return moved;
    }

    //! \brief The eigenvalues \p select holds moved first (\ref reorder), as `scipy.linalg.ordqz(…, sort=…)` sorts
    //!        them. \return Their count.
    std::size_t sort(const selection& select)
    {
      std::vector<bool> chosen(values_.size());
      for (std::size_t k {0}; k < values_.size(); ++k) {
        chosen[k] = select(std::complex<double> {values_[k].alphar, values_[k].alphai}, std::complex<double> {values_[k].beta});
      }
      return reorder(chosen);
    }

  private:

    /*!
     * \brief The eigenvalues of the scaled form \p raw taken back, as `dgges` takes them: `α` by `A`'s factor, `β` by
     *        `B`'s; a conjugate pair whose `α` or `β` would leave the range first rescaled as a whole (its ratio kept) to
     *        the size of its block's entries. A pair of `αr = 0` takes the second test (`dgges` divides by it).
     */
    [[nodiscard]] std::vector<detail::real_pencil_value> unscaled(std::vector<detail::real_pencil_value> raw) const
    {
      const double safmin {std::numeric_limits<double>::min()};
      const double safmax {1.0 / safmin};
      const auto   rescale = [](detail::real_pencil_value& v, double w) {
                               v.beta   *= w;
                               v.alphar *= w;
                               v.alphai *= w;
                             };
      const std::size_t n {form_.n};
      for (std::size_t i {0}; i < raw.size(); ++i) {
        detail::real_pencil_value& v {raw[i]};
        if (v.alphai == 0.0) {
          continue;
        }
        const double s_ii {form_.s[(i * n) + i]};
        if (a_back_ != 1.0) {
          if (v.alphar != 0.0 && ((v.alphar / safmax) > (1.0 / a_back_) || (safmin / v.alphar) > a_back_)) {
            rescale(v, std::abs(s_ii / v.alphar));
          }
          else if ((v.alphai / safmax) > (1.0 / a_back_) || (safmin / v.alphai) > a_back_) { // αi > 0: the pair's first
            rescale(v, std::abs(form_.s[(i * n) + i + 1] / v.alphai));
          }
        }
        if (b_back_ != 1.0 && ((v.beta / safmax) > (1.0 / b_back_) || (safmin / v.beta) > b_back_)) {
          rescale(v, std::abs(form_.t[(i * n) + i] / v.beta));
        }
      }
      for (detail::real_pencil_value& v : raw) {
        v.alphar *= a_back_;
        v.alphai *= a_back_;
        v.beta   *= b_back_;
      }
      return raw;
    }

    detail::real_qz_form                   form_;         //!< S, T, Q and Z, of the scaled pencil.
    std::vector<detail::real_pencil_value> values_;       //!< The eigenvalues.
    double                                 a_back_ {1.0}; //!< The factor taking S back to A's scale.
    double                                 b_back_ {1.0}; //!< T's.
  };

  /*!
   * \brief The right (or, where \p left, the left) eigenvectors of the real pencil \p d decomposes, complex, as
   *        `scipy.linalg.eig(a, b)` gives them for a real pencil (each of unit 2-norm): each 2 × 2 block of `S` and `T`
   *        made triangular first by unitary rotations from both sides (the left one along `T v`, the right one along
   *        the block's eigenvector `v`), carried through `S`, `T`, `Q` and `Z`, then the complex pencil's vectors
   *        (\ref generalized_eigenvectors). The columns in the diagonals' order, as \p d's values.
   */
  [[nodiscard]] inline matrix<std::complex<double>> generalized_eigenvectors(const real_generalized_schur& d,
                                                                             bool                          left)
  {
    using C = std::complex<double>;
    const matrix<double> real_s     {d.s()};
    const matrix<double> real_t     {d.t()};
    const matrix<double> real_basis {left ? d.q() : d.z()};
    const std::size_t    n          {real_s.rows()};
    buffer<C>            s(real_s.begin(), real_s.end()); // row-major, S(i, j) at i n + j
    buffer<C>            t(real_t.begin(), real_t.end());
    buffer<C>            basis(real_basis.begin(), real_basis.end());
    // Rows k and k + 1 of m, from column k, by Uᴴ for U = [u, (−ū1, ū0)].
    const auto rows_by = [n](buffer<C>& m, std::size_t k, const C& u0, const C& u1) {
                           for (std::size_t j {k}; j < n; ++j) {
                             const C a {m[(k * n) + j]};
                             const C b {m[((k + 1) * n) + j]};
                             m[(k * n) + j]       = (std::conj(u0) * a) + (std::conj(u1) * b);
                             m[((k + 1) * n) + j] = (-u1 * a) + (u0 * b);
                           }
                         };
    // Columns k and k + 1 of m's first `rows` rows by V = [v, (−v̄1, v̄0)].
    const auto columns_by = [n](buffer<C>& m, std::size_t k, std::size_t rows, const C& v0, const C& v1) {
                              for (std::size_t i {0}; i < rows; ++i) {
                                const C a {m[(i * n) + k]};
                                const C b {m[(i * n) + k + 1]};
                                m[(i * n) + k]     = (a * v0) + (b * v1);
                                m[(i * n) + k + 1] = (-a * std::conj(v1)) + (b * std::conj(v0));
                              }
                            };
    std::size_t k {0};
    while (k + 1 < n) {
      if (s[((k + 1) * n) + k] == C {}) { // a real eigenvalue's 1 × 1 block
        ++k;
        continue;
      }
      // (S_b − λ T_b) v = 0 by the larger row of the 2 × 2 block; u along T_b v, which S_b v is λ times
      const detail::real_pencil_value& value {d.values()[k]};
      const C                          lambda {C {value.alphar, value.alphai} / value.beta};
      const auto                       m = [&](std::size_t i, std::size_t j) { return s[(i * n) + j] - (lambda * t[(i * n) + j]); };
      const bool                       upper  {std::hypot(std::abs(m(k, k)), std::abs(m(k, k + 1))) >= std::hypot(std::abs(m(k + 1, k)), std::abs(m(k + 1, k + 1)))};
      C                                v0     {upper ? m(k, k + 1) : m(k + 1, k + 1)};
      C                                v1     {upper ? -m(k, k) : -m(k + 1, k)};
      const double                     v_norm {std::hypot(std::abs(v0), std::abs(v1))};
      v0 /= v_norm;
      v1 /= v_norm;
      C            u0     {(t[(k * n) + k] * v0) + (t[(k * n) + k + 1] * v1)};
      C            u1     {t[((k + 1) * n) + k + 1] * v1};
      const double u_norm {std::hypot(std::abs(u0), std::abs(u1))};
      u0 /= u_norm;
      u1 /= u_norm;
      rows_by(s, k, u0, u1);
      rows_by(t, k, u0, u1);
      columns_by(s, k, k + 2, v0, v1);
      columns_by(t, k, k + 2, v0, v1);
      s[((k + 1) * n) + k] = C {};
      t[((k + 1) * n) + k] = C {};
      if (left) {
        columns_by(basis, k, n, u0, u1);
      }
      else {
        columns_by(basis, k, n, v0, v1);
      }
      k += 2;
    }
    std::vector<std::pair<C, C>> diagonal(n);
    for (std::size_t i {0}; i < n; ++i) {
      diagonal[i] = {s[(i * n) + i], t[(i * n) + i]};
    }
    return detail::triangular_pencil_vectors(matrix<C> {n, n, std::move(s)}, matrix<C> {n, n, std::move(t)}, diagonal,
                                             matrix<C> {n, n, std::move(basis)}, left);
  }

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
    template <typename T>
    [[nodiscard]] double norm_one(const matrix<T>& a)
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

    //! \brief The operands of an algebraic Riccati equation in the arithmetic of \p T, `e` and `s` absent when null.
    template <typename T>
    struct riccati_operands
    {
      const matrix<T>& a;
      const matrix<T>& b;
      const matrix<T>& q;
      const matrix<T>& r;
      const matrix<T>* e;
      const matrix<T>* s;
    };

    //! \brief The operands of an algebraic Riccati equation as complex matrices (the residuals and Newton's step).
    using riccati_terms = riccati_operands<std::complex<double>>;

    //! \brief Refuses a Riccati equation whose pencil has eigenvalues on (or too close to) the boundary.
    //! \throws convergence_error Always.
    [[noreturn]] inline void refuse_boundary(bool continuous)
    {
      throw convergence_error(continuous ? "The associated Hamiltonian pencil has eigenvalues too close to the imaginary axis"
                                         : "The associated symplectic pencil has eigenvalues too close to the unit circle");
    }

    /*!
     * \brief `X = U₂₁ U₁₁⁻¹` from the `m` leading Schur vectors' halves \p top (`U₁₁`) and \p bottom (`U₂₁`) of a Riccati
     *        equation's Hamiltonian (\p continuous) or symplectic matrix or pencil, as SciPy's checks have it.
     * \throws singular_matrix_error If `U₁₁` is numerically singular (the condition of its LU factor `U` past `1/ε`, as
     *                               SciPy's): no finite stabilising solution.
     * \throws convergence_error     If `U₁₁ᴴ U₂₁` is not Hermitian: the eigenvalues too close to the imaginary axis (the
     *                               unit circle).
     */
    template <typename T>
    [[nodiscard]] matrix<T> stabilising_solution(const matrix<T>& top,
                                                 const matrix<T>& bottom,
                                                 bool             continuous)
    {
      const std::size_t m              {top.rows()};
      const matrix<T>   factors        {lu_decomposition<T> {top}.factors()};
      buffer<T>         upper(m * m, T {});
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t k {i}; k < m; ++k) {
          upper[(i * m) + k] = factors(i, k);
        }
      }
      const vector<double> sigma {singular_values(matrix<T> {m, m, std::move(upper)})};
      if (!(sigma[m - 1] > std::numeric_limits<double>::epsilon() * sigma[0])) { // 1/cond ≤ ε, as SciPy (a zero U too)
        throw singular_matrix_error("Failed to find a finite solution.");
      }
      const matrix<T> product   {adjoint_of(top) * bottom};
      const double    tolerance {std::max(std::ldexp(1.0, -43), 0.1 * norm_one(product))};
      if (!(norm_one(product - adjoint_of(product)) <= tolerance)) {
        refuse_boundary(continuous);
      }
      return adjoint_of(solve(adjoint_of(top), adjoint_of(bottom)));
    }

    /*!
     * \brief The extended pencil `(H, J)` of order `2m + n` of SciPy's `solve_continuous_are` (\p continuous) or
     *        `solve_discrete_are`, row-major.
     */
    template <typename T>
    [[nodiscard]] std::pair<buffer<T>, buffer<T>> riccati_pencil(const riccati_operands<T>& terms,
                                                                 bool                       continuous)
    {
      const std::size_t m {terms.a.rows()};
      const std::size_t n {terms.b.cols()};
      const std::size_t order {(2 * m) + n};
      buffer<T>         h(order * order, T {});
      buffer<T>         j(order * order, T {});
      const auto        e_at = [&terms](std::size_t row, std::size_t col) {
                                 return terms.e != nullptr ? (*terms.e)(row, col) : T {row == col ? 1.0 : 0.0};
                               };
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t k {0}; k < m; ++k) {
          h[(i * order) + k]           = terms.a(i, k);
          h[((m + i) * order) + k]     = -terms.q(i, k);
          j[(i * order) + k]           = e_at(i, k);
          h[((m + i) * order) + m + k] = continuous ? -conj_of(terms.a(k, i)) : conj_of(e_at(k, i));
          j[((m + i) * order) + m + k] = continuous ? conj_of(e_at(k, i)) : conj_of(terms.a(k, i));
        }
        for (std::size_t k {0}; k < n; ++k) {
          h[(i * order) + (2 * m) + k] = terms.b(i, k);
          if (continuous) {
            h[(((2 * m) + k) * order) + m + i] = conj_of(terms.b(i, k));
          }
          else {
            j[(((2 * m) + k) * order) + m + i] = -conj_of(terms.b(i, k));
          }
          if (terms.s != nullptr) {
            h[((m + i) * order) + (2 * m) + k] = -(*terms.s)(i, k);
            h[(((2 * m) + k) * order) + i]     = conj_of((*terms.s)(i, k));
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
    template <typename T>
    [[nodiscard]] buffer<double> balance_pencil(buffer<T>&  h,
                                                buffer<T>&  j,
                                                std::size_t m,
                                                std::size_t order)
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
    template <typename T>
    [[nodiscard]] matrix<T> riccati(const riccati_operands<T>& terms,
                                    bool                       continuous,
                                    bool                       balanced)
    {
      using form = std::conditional_t<std::is_same_v<T, double>, real_generalized_schur, generalized_schur>;
      const std::size_t m        {terms.a.rows()};
      const std::size_t n        {terms.b.cols()};
      const std::size_t order    {(2 * m) + n};
      auto [h, j] = riccati_pencil(terms, continuous);
      const buffer<double> scale {balanced ? balance_pencil(h, j, m, order) : buffer<double>(order, 1.0)};
      buffer<T>            last(order * n);
      for (std::size_t i {0}; i < order; ++i) {
        std::copy_n(h.data() + (i * order) + (2 * m), n, last.data() + (i * n));
      }
      const matrix<T> deflation       {qr_decomposition<T> {matrix<T> {order, n, std::move(last)}, qr_mode::complete}.q()};
      buffer<T>       kept(order * 2 * m); // the columns past the first n: Q₂
      buffer<T>       h_left(order * 2 * m);
      buffer<T>       j_left(order * 2 * m);
      for (std::size_t t {0}; t < order; ++t) {
        std::copy_n(&deflation(t, n), 2 * m, kept.data() + (t * 2 * m));
        std::copy_n(h.data() + (t * order), 2 * m, h_left.data() + (t * 2 * m));
        std::copy_n(j.data() + (t * order), 2 * m, j_left.data() + (t * 2 * m));
      }
      const matrix<T>   q2     {adjoint_of(matrix<T> {order, 2 * m, std::move(kept)})};
      form              pencil {q2 * matrix<T> {order, 2 * m, std::move(h_left)}, q2 * matrix<T> {order, 2 * m, std::move(j_left)}};
      const std::size_t chosen {continuous ? pencil.sort([](const std::complex<double>& alpha, const std::complex<double>& beta) {
                                                           return (alpha * std::conj(beta)).real() < 0.0;
                                                         })
                                           : pencil.sort([](const std::complex<double>& alpha, const std::complex<double>& beta) {
                                                           return std::abs(alpha) < std::abs(beta);
                                                         })};
      if (chosen != m) {             // a stable half of other than m eigenvalues: some on the boundary, where a real pencil's
        refuse_boundary(continuous); // conjugate pair moves whole and its basis passes the symmetry test
      }
      matrix<T> u {pencil.z()};
      if (terms.e != nullptr) {      // [D₁ E D₁⁻¹ U₀₀; U₁₀], E scaled as the balancing scaled the pencil
        buffer<T> scaled_e(m * m);
        buffer<T> upper(m * m);
        for (std::size_t i {0}; i < m; ++i) {
          for (std::size_t t {0}; t < m; ++t) {
            scaled_e[(i * m) + t] = (*terms.e)(i, t) * (scale[i] / scale[t]);
            upper[(i * m) + t]    = u(i, t);
          }
        }
        const matrix<T> top {matrix<T> {m, m, std::move(scaled_e)} *matrix<T> {m, m, std::move(upper)}};
        buffer<T>       stacked(2 * m * m);
        std::copy(top.begin(), top.end(), stacked.begin());
        for (std::size_t i {0}; i < m; ++i) {
          for (std::size_t k {0}; k < m; ++k) {
            stacked[((m + i) * m) + k] = u(m + i, k);
          }
        }
        u = qr_decomposition<T> {matrix<T> {2 * m, m, std::move(stacked)}, qr_mode::complete}.q();
      }
      buffer<T> u00(m * m);
      buffer<T> u10(m * m);
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t k {0}; k < m; ++k) {
          u00[(i * m) + k] = u(i, k);
          u10[(i * m) + k] = u(m + i, k);
        }
      }
      const matrix<T> x {stabilising_solution(matrix<T> {m, m, std::move(u00)}, matrix<T> {m, m, std::move(u10)}, continuous)};
      buffer<T>       out(m * m);
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t k {0}; k < m; ++k) {
          out[(i * m) + k] = (x(i, k) + conj_of(x(k, i))) * (scale[i] * scale[k] / 2.0);
        }
      }
      return matrix<T> {m, m, std::move(out)};
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
     * \brief The stabilising solution of the continuous equation without `E` from the Schur form of the Hamiltonian
     *        `H = [[Ã, −G], [−Q̃, −Ãᴴ]]` (Laub, 1979), `G = B R⁻¹ Bᴴ`, `Ã = A − B R⁻¹ Sᴴ` and `Q̃ = Q − S R⁻¹ Sᴴ` in the
     *        arithmetic of \p T (a real `H`'s real Schur form): its eigenvalues in the left half-plane moved first,
     *        `X = U₂₁ U₁₁⁻¹` from their `m` Schur vectors, made Hermitian. A Schur form of order `2m` by multishift QR
     *        where the pencil's QZ takes order `2m + n`.
     * \throws singular_matrix_error If `U₁₁` is numerically singular (\ref stabilising_solution).
     * \throws convergence_error     If the left half-plane holds other than `m` eigenvalues, or as
     *                               \ref stabilising_solution says (eigenvalues on or too close to the imaginary axis).
     */
    template <typename T>
    [[nodiscard]] matrix<T> hamiltonian_riccati(const matrix<T>& a,
                                                const matrix<T>& b,
                                                const matrix<T>& q,
                                                const matrix<T>& r,
                                                const matrix<T>* s)
    {
      const std::size_t m      {a.rows()};
      const matrix<T>   g      {b * solve(r, adjoint_of(b))};
      matrix<T>         shift  {a};
      matrix<T>         weight {q};
      if (s != nullptr) {
        const matrix<T> cross {solve(r, adjoint_of(*s))}; // R⁻¹ Sᴴ
        shift  = a - (b * cross);
        weight = q - (*s * cross);
      }
      const std::size_t order {2 * m};
      buffer<T>         h(order * order);
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t j {0}; j < m; ++j) {
          h[(i * order) + j]           = shift(i, j);
          h[(i * order) + m + j]       = -(g(i, j) + conj_of(g(j, i))) / 2.0;
          h[((m + i) * order) + j]     = -(weight(i, j) + conj_of(weight(j, i))) / 2.0;
          h[((m + i) * order) + m + j] = -conj_of(shift(j, i));
        }
      }
      schur_decomposition<T>                  form   {matrix<T> {order, order, std::move(h)}};
      const std::vector<std::complex<double>> values {form.values()};
      std::vector<bool>                       chosen(order);
      for (std::size_t k {0}; k < order; ++k) {
        chosen[k] = values[k].real() < 0.0;
      }
      if (form.reorder(chosen) != m) {
        throw convergence_error("The associated Hamiltonian matrix has eigenvalues too close to the imaginary axis");
      }
      buffer<T> u11(m * m);
      buffer<T> u21(m * m);
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t k {0}; k < m; ++k) {
          u11[(i * m) + k] = form.z()(i, k);
          u21[(i * m) + k] = form.z()(m + i, k);
        }
      }
      const matrix<T> x {stabilising_solution(matrix<T> {m, m, std::move(u11)}, matrix<T> {m, m, std::move(u21)}, true)};
      buffer<T>       out(m * m);
      for (std::size_t i {0}; i < m; ++i) {
        for (std::size_t k {0}; k < m; ++k) {
          out[(i * m) + k] = (x(i, k) + conj_of(x(k, i))) / 2.0;
        }
      }
      return matrix<T> {m, m, std::move(out)};
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
      bool hamiltonian {false};
      if (continuous) { // R⁻¹ is the equation's: SciPy refuses a numerically singular r
        const vector<double> sigma {singular_values(cr)};
        if (!(sigma[n - 1] >= std::numeric_limits<double>::epsilon() * norm_one(cr)) || sigma[n - 1] == 0.0) {
          throw std::invalid_argument("Matrix r is numerically singular.");
        }
        // a well-conditioned R (within √ε of singular) and no E: the Hamiltonian's Schur form, else the pencil's
        hamiltonian = e == nullptr && sigma[n - 1] >= std::sqrt(std::numeric_limits<double>::epsilon()) * sigma[0];
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
      // a real equation's pencil in real arithmetic, real_generalized_schur's double shifts, as SciPy's dgges
      matrix<std::complex<double>> x     {hamiltonian ? as_complex(hamiltonian_riccati(a, b, q, r, s))
                                                      : as_complex(riccati(riccati_operands<T> {a, b, q, r, e, s}, continuous, balanced))};
      // the discrete equation takes no step of Newton's (Hewer's): its residual, rounded at ε ‖A‖² ‖X‖ where the
      // continuous one is at ε ‖A‖ ‖X‖, steers the step away from X wherever A is large or graded
      if (continuous) {
        x = newton_refined(terms, std::move(x));
      }
      return answer_like(std::move(x), a);
    }
  } // namespace detail

  /*!
   * \brief The stabilising solution `X` of the continuous algebraic Riccati equation `Aᴴ X E + Eᴴ X A − (Eᴴ X B + S)
   *        R⁻¹ (Bᴴ X E + Sᴴ) + Q = 0` (`E = I` and `S = 0` when \p e and \p s are null), as
   *        `scipy.linalg.solve_continuous_are`: without \p e and with `R` well-conditioned (its condition below
   *        `1/√ε`), the Schur form of the Hamiltonian `[[Ã, −G], [−Q̃, −Ãᴴ]]` (Laub, 1979, in real arithmetic for real
   *        operands); else, as SciPy, the eigenvalues of the extended Hamiltonian pencil in the left half-plane moved
   *        first in its generalized Schur form (\ref generalized_schur), `R` never inverted. Then one step of
   *        Newton's method, kept where it lowers the residual.
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
